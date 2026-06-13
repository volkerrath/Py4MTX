#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mt_get_averages.py
=========
Load impedance invariants and off-diagonal components for all sites in a
collection NPZ, compute apparent resistivity and phase per site, form
frequency-wise log averages across sites, compute a global scalar log-average
rho_a, produce one rho/phase plot per active quantity, and save the averaged
curves to NPZ.

Quantities
----------
Zssq  — sum-of-squares invariant, (n_freq,) complex, mV/km/nT
        defined as  Zssq = sqrt( (Zxx² + Zxy² + Zyx² + Zyy²) / 2 )
Zdet  — determinant invariant,    (n_freq,) complex, mV/km/nT
        defined as  Zdet = sqrt( Zxx·Zyy − Zxy·Zyx )
Zavg  — off-diagonal geometric mean, (n_freq,) complex, mV/km/nT
        defined as  Zavg = sqrt( Zxy · Zyx )
Zxy   — off-diagonal element Z[:,0,1], (n_freq,) complex, mV/km/nT
Zyx   — off-diagonal element Z[:,1,0], (n_freq,) complex, mV/km/nT

For all quantities, apparent resistivity and phase are:
        rho_a(f) = mu0 * |Z|² * 1e6 / (2π f)   [Ω·m]
        phi(f)   = arg(Z)                        [°]

Averaging
---------
Per-frequency log average : geometric mean of rho_a over all sites that
                             have data at that bin (nearest-bin matching);
                             arithmetic mean of phase.
Global log average         : geometric mean of all per-site, all-frequency
                             rho_a values (scalar).

NPZ output
----------
For each active quantity a compressed NPZ is written alongside the plots,
containing: freq_avg [Hz], period_avg [s], rho_avg [Ω·m], phi_avg [°],
rho_global (scalar), n_sites (int).

@author:    vrath (VR)
@project:   py4mt — Python for Magnetotellurics
@created:   2026-06-12
@modified:  2026-06-12 — initial version; Claude Sonnet 4.6 (Anthropic)
@modified:  2026-06-12 — annotation to axes-fraction lower-left; LEGEND_SITES switch; Claude Sonnet 4.6 (Anthropic)
@modified:  2026-06-12 — legend fontsize/ncol scaled by LEGEND_SITES (fs=10/ncol=1 vs fs=5/ncol=4); Claude Sonnet 4.6 (Anthropic)
@modified:  2026-06-12 — PLOT_SSQ / PLOT_DET switches; unified _rho_phi_from_zinvar and _plot_invar; Claude Sonnet 4.6 (Anthropic)
@modified:  2026-06-13 — PLOT_AVG: Zavg = sqrt(Zxy*Zyx) off-diagonal geometric mean; _plot_invar returns averages; NPZ save per quantity; Claude Sonnet 4.6 (Anthropic)
@modified:  2026-06-13 — removed PLOT_XY/YX; added FREQ_RANGE frequency mask; Claude Sonnet 4.6 (Anthropic)
@modified:  2026-06-13 — PLOT_AVG switched to direct "Zavg" key lookup (analog to Zdet/Zssq); Claude Sonnet 4.6 (Anthropic)
@modified:  2026-06-13 — FIG_SIZE_CM parameter; PLOT_ERRORS/ERR_ALPHA: per-bin std shading on rho and phase for all three averages; rho_std/phi_std added to NPZ output; Claude Sonnet 4.6 (Anthropic)
@modified:  2026-06-13 — Zavg phase folded to [0°,90°] via fold_phase flag in _INVAR_SPEC; Claude Sonnet 4.6 (Anthropic)
@modified:  2026-06-13 — ERRORS switch ("std"/"bootstrap"/None); bootstrap std-of-the-mean per bin and globally; global interval in print/annotation; rho_global_std in NPZ; Claude Sonnet 4.6 (Anthropic)
@modified:  2026-06-13 — fix global error: resample per-site log-means (not pooled values), stay in log space, report multiplicative factor ×/÷ σ_g; Claude Sonnet 4.6 (Anthropic)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WORK_DIR   = "/home/vrath/Py4MTX/py4mt/data/rto/ubinas/edi/proc/"
COLL_FILE  = WORK_DIR + "Ubinas_collection.npz"
PLOT_DIR   = WORK_DIR + "../plots/"
DATA_DIR   = WORK_DIR                  # NPZ averages written here
NAME_STR   = "Ubinas_noHF_std"
PLOT_FORMAT = [".pdf", ".png"]

# Which quantities to process and plot
PLOT_SSQ = True    # Zssq — sum-of-squares invariant
PLOT_DET = True    # Zdet — determinant invariant
PLOT_AVG = True    # Zavg — off-diagonal geometric mean sqrt(Zxy*Zyx)

# Frequency range mask applied before averaging and plotting.
# Tuple (f_min, f_max) in Hz; set either bound to None for no limit.
FREQ_RANGE = (None, 100.)   # e.g. (1.e-3, 1.e+3) to restrict to 1 mHz–1 kHz

# Frequency-bin matching tolerance (fraction of a decade).
BIN_TOL_DEC = 0.1   # ±0.1 decade

# Plot appearance
XLIM       = (1.e-4, 1.e+4)   # period axis limits [s]  (None = auto)
YLIM_RHO   = (1.e+0, 1.e+4)   # rho axis limits [Ω·m]   (None = auto)
YLIM_PHASE = (0., 90.)         # phase axis limits [°]    (None = auto)

LEGEND_SITES = True    # True: all station names in legend; False: averages only

FIG_SIZE_CM  = (22., 18.)   # figure size in cm (width, height)

PLOT_ERRORS = True     # shade error band around the freq-wise averages
ERRORS      = "std" #"bootstrap"  # "std"       — sample std across sites (spatial spread)
                           # "bootstrap" — bootstrap std-of-the-mean (N_SIM resamples)
                           # None / False — no error band
ERR_NSIM    = 500      # number of bootstrap resamples (only used when ERRORS="bootstrap")
ERR_ALPHA   = 0.20     # opacity of the error shading

SITE_ALPHA  = 0.35
SITE_LW     = 1.8
AVG_LW      = 3.0
AVG_COLOR   = "red"
SITE_CMAP   = "tab20"

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
_MU0: float = 4.0 * np.pi * 1.0e-7
_Z_MT_TO_SI_SQ: float = _MU0 ** 2 * 1.0e6   # (mu0*1e3)² / mu0  → see data_proc

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rho_phi_from_zinvar(
    freq: np.ndarray,
    Zinvar: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute rho_a [Ω·m] and phase [°] from a scalar complex quantity."""
    omega = 2.0 * np.pi * freq
    rho = _Z_MT_TO_SI_SQ * np.abs(Zinvar) ** 2 / (_MU0 * omega)
    phi = np.degrees(np.angle(Zinvar))
    return rho, phi


def _period(freq: np.ndarray) -> np.ndarray:
    return 1.0 / freq


def _log_avg(values: list[np.ndarray]) -> float:
    """Global scalar geometric mean of all finite positive values."""
    all_vals = np.concatenate([v[np.isfinite(v) & (v > 0)] for v in values])
    if all_vals.size == 0:
        return np.nan
    return float(np.exp(np.mean(np.log(all_vals))))


def _build_freq_grid(all_freqs: list[np.ndarray]) -> np.ndarray:
    """Union of all site frequencies, sorted ascending, deduplicated (1 % tol)."""
    merged = np.sort(np.concatenate(all_freqs))
    keep = np.ones(len(merged), dtype=bool)
    for i in range(1, len(merged)):
        if merged[i] / merged[i - 1] < 1.01:
            keep[i] = False
    return merged[keep]


def _freqwise_log_avg(
    grid: np.ndarray,
    all_freqs: list[np.ndarray],
    all_rho:   list[np.ndarray],
    all_phi:   list[np.ndarray],
    tol_dec:   float = BIN_TOL_DEC,
    errors:    str | None = "bootstrap",
    n_sim:     int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-frequency geometric mean of rho_a and arithmetic mean of phase,
    with optional error estimation.

    Parameters
    ----------
    errors : {"std", "bootstrap", None}
        "std"       — sample std across sites (spatial spread / heterogeneity).
        "bootstrap" — bootstrap std-of-the-mean: n_sim resamples with
                      replacement from the contributing sites at each bin;
                      std of the bootstrap distribution of the mean.
        None        — no error computed; rho_std and phi_std are all NaN.
    n_sim : int
        Number of bootstrap resamples (used only when errors="bootstrap").

    Returns
    -------
    grid, rho_avg, phi_avg, rho_std, phi_std

    rho_std is a multiplicative geometric factor: band = rho_avg/rho_std … rho_avg*rho_std.
    phi_std is in degrees.  Both are NaN where count < 2.
    """
    n        = len(grid)
    log_grid = np.log10(grid)

    # --- first pass: collect per-bin site values ---
    # Store lists so bootstrap can resample; also accumulate sums for the mean.
    bin_log_rho: list[list[float]] = [[] for _ in range(n)]
    bin_phi:     list[list[float]] = [[] for _ in range(n)]

    for freq_s, rho_s, phi_s in zip(all_freqs, all_rho, all_phi):
        log_fs = np.log10(freq_s)
        for k, lf in enumerate(log_grid):
            dist = np.abs(log_fs - lf)
            idx  = np.argmin(dist)
            if dist[idx] > tol_dec:
                continue
            if not np.isfinite(rho_s[idx]) or rho_s[idx] <= 0:
                continue
            bin_log_rho[k].append(np.log(rho_s[idx]))
            bin_phi[k].append(phi_s[idx])

    # --- compute means ---
    counts  = np.array([len(b) for b in bin_log_rho], dtype=int)
    valid   = counts > 0
    rho_avg = np.full(n, np.nan)
    phi_avg = np.full(n, np.nan)
    for k in range(n):
        if counts[k] > 0:
            rho_avg[k] = np.exp(np.mean(bin_log_rho[k]))
            phi_avg[k] = np.mean(bin_phi[k])

    # --- compute error estimates ---
    rho_std = np.full(n, np.nan)
    phi_std = np.full(n, np.nan)
    enough  = counts >= 2

    _errors = str(errors).lower() if errors else None

    if _errors == "std":
        for k in range(n):
            if not enough[k]:
                continue
            lr = np.array(bin_log_rho[k])
            pp = np.array(bin_phi[k])
            rho_std[k] = np.exp(np.std(lr, ddof=1))
            phi_std[k] = np.std(pp, ddof=1)

    elif _errors == "bootstrap":
        rng_bs = np.random.default_rng()
        for k in range(n):
            if not enough[k]:
                continue
            lr = np.array(bin_log_rho[k])
            pp = np.array(bin_phi[k])
            boot_rho_means = np.empty(n_sim)
            boot_phi_means = np.empty(n_sim)
            for b in range(n_sim):
                idx_b = rng_bs.integers(0, len(lr), size=len(lr))
                boot_rho_means[b] = np.mean(lr[idx_b])
                boot_phi_means[b] = np.mean(pp[idx_b])
            rho_std[k] = np.exp(np.std(boot_rho_means, ddof=1))
            phi_std[k] = np.std(boot_phi_means, ddof=1)

    return grid, rho_avg, phi_avg, rho_std, phi_std


def _plot_invar(
    tag: str,
    label: str,
    stations: list[str],
    all_freqs: list[np.ndarray],
    all_rho:   list[np.ndarray],
    all_phi:   list[np.ndarray],
) -> dict:
    """Produce and save the rho/phase figure for one quantity.

    Returns a dict with the averaged curves and global scalar for NPZ export:
        freq_avg, period_avg, rho_avg, phi_avg  — 1-D arrays on the merged grid
        rho_std                                  — geometric std factor (multiplicative)
        phi_std                                  — std of phase [°]
        rho_global                               — scalar float
        n_sites                                  — int
    """
    freq_grid = _build_freq_grid(all_freqs)
    grid, rho_favg, phi_favg, rho_fstd, phi_fstd = _freqwise_log_avg(
        freq_grid, all_freqs, all_rho, all_phi,
        errors=ERRORS if PLOT_ERRORS else None,
        n_sim=ERR_NSIM,
    )
    rho_global = _log_avg(all_rho)
    n_sites    = len(stations)

    # Global error on log(rho_global): operate on per-site log-means (one value
    # per site), so frequency dependence does not inflate the spread estimate.
    _err_label = str(ERRORS).lower() if PLOT_ERRORS and ERRORS else None
    _site_lr   = np.array([np.mean(np.log(v[np.isfinite(v) & (v > 0)]))
                            for v in all_rho
                            if np.any(np.isfinite(v) & (v > 0))])

    if len(_site_lr) >= 2 and _err_label == "bootstrap":
        _rng_g = np.random.default_rng()
        _boot_means = np.array([
            np.mean(_rng_g.choice(_site_lr, size=len(_site_lr), replace=True))
            for _ in range(ERR_NSIM)
        ])
        # std of bootstrap means in log space → geometric multiplicative factor
        _rho_gstd_log = np.std(_boot_means, ddof=1)
        _rho_gstd     = np.exp(_rho_gstd_log)          # multiplicative factor
        _err_str = f" ×/÷ {_rho_gstd:.2f} (bootstrap)"
    elif len(_site_lr) >= 2 and _err_label == "std":
        # sample std of per-site log-means → geometric std factor
        _rho_gstd_log = np.std(_site_lr, ddof=1)
        _rho_gstd     = np.exp(_rho_gstd_log)
        _err_str = f" ×/÷ {_rho_gstd:.2f} (std)"
    else:
        _rho_gstd = np.nan
        _err_str  = ""

    print(f"  {label}: global log-average rho_a = {rho_global:.1f}{_err_str} Ω·m  "
          f"({n_sites} sites)")

    cmap   = plt.get_cmap(SITE_CMAP)
    colors = [cmap(i % cmap.N) for i in range(n_sites)]

    _fig_inch = (FIG_SIZE_CM[0] / 2.54, FIG_SIZE_CM[1] / 2.54)
    fig, (ax_rho, ax_phi) = plt.subplots(2, 1, figsize=_fig_inch, sharex=True)
    fig.suptitle(f"{label} — apparent resistivity & phase", fontsize=11)

    # --- rho panel ---
    for i, (freq_s, rho_s, sta) in enumerate(zip(all_freqs, all_rho, stations)):
        ax_rho.loglog(_period(freq_s), rho_s,
                      color=colors[i], alpha=SITE_ALPHA, lw=SITE_LW, label=sta)

    per_grid = _period(grid)
    valid = np.isfinite(rho_favg)
    ax_rho.loglog(per_grid[valid], rho_favg[valid],
                  color=AVG_COLOR, lw=AVG_LW, ls="--",
                  label="freq-wise avg", zorder=5)

    if PLOT_ERRORS:
        std_ok = valid & np.isfinite(rho_fstd)
        if std_ok.any():
            ax_rho.fill_between(
                per_grid[std_ok],
                rho_favg[std_ok] / rho_fstd[std_ok],
                rho_favg[std_ok] * rho_fstd[std_ok],
                color=AVG_COLOR, alpha=ERR_ALPHA, lw=0, zorder=4,
                label="±1 std",
            )

    ax_rho.axhline(rho_global, color="black", lw=3., ls=":", zorder=4, alpha=0.8)
    _annot = f"$\\bar{{\\rho}}_{{\\!a}}$ = {rho_global:.0f} Ω·m{_err_str}"
    ax_rho.annotate(
        _annot,
        xy=(0.02, 0.05), xycoords="axes fraction",
        fontsize=11, color="black", va="bottom", ha="left",
    )
    ax_rho.set_ylabel("$\\rho_a$ (Ω·m)")
    if YLIM_RHO:
        ax_rho.set_ylim(YLIM_RHO)
    if XLIM:
        ax_rho.set_xlim(XLIM)
    ax_rho.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())
    ax_rho.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)

    # --- phase panel ---
    for i, (freq_s, phi_s) in enumerate(zip(all_freqs, all_phi)):
        ax_phi.semilogx(_period(freq_s), phi_s,
                        color=colors[i], alpha=SITE_ALPHA, lw=SITE_LW)

    valid_p = np.isfinite(phi_favg)
    ax_phi.semilogx(per_grid[valid_p], phi_favg[valid_p],
                    color=AVG_COLOR, lw=AVG_LW, ls="--", zorder=5)

    if PLOT_ERRORS:
        std_ok_p = valid_p & np.isfinite(phi_fstd)
        if std_ok_p.any():
            ax_phi.fill_between(
                per_grid[std_ok_p],
                phi_favg[std_ok_p] - phi_fstd[std_ok_p],
                phi_favg[std_ok_p] + phi_fstd[std_ok_p],
                color=AVG_COLOR, alpha=ERR_ALPHA, lw=0, zorder=4,
            )

    ax_phi.axhline(0,  color="grey", lw=0.6, ls="-")
    ax_phi.axhline(45, color="grey", lw=0.6, ls=":")
    ax_phi.set_ylabel("Phase (°)")
    ax_phi.set_xlabel("Period (s)")
    if YLIM_PHASE:
        ax_phi.set_ylim(YLIM_PHASE)
    if XLIM:
        ax_phi.set_xlim(XLIM)
    ax_phi.yaxis.set_major_locator(mticker.MultipleLocator(45))
    ax_phi.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)

    # --- legend ---
    handles, labels_leg = ax_rho.get_legend_handles_labels()
    if not LEGEND_SITES:
        handles    = [h for h, l in zip(handles, labels_leg)
                      if l in ("freq-wise avg", "±1 std")]
        labels_leg = [l for l in labels_leg
                      if l in ("freq-wise avg", "±1 std")]
        _leg_fs, _leg_ncol = 10, 1
    else:
        _leg_fs, _leg_ncol = 5, 4
    ax_rho.legend(handles, labels_leg,
                  fontsize=_leg_fs, ncol=_leg_ncol,
                  loc="best", framealpha=0.7,
                  handlelength=1.2, columnspacing=0.6, labelspacing=0.2)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    for fmt in PLOT_FORMAT:
        out = os.path.join(PLOT_DIR, f"{NAME_STR}_{tag}{fmt}")
        plt.savefig(out, dpi=300)
        print(f"  Wrote {out}")

    plt.show()

    return {
        "freq_avg":      grid,
        "period_avg":    per_grid,
        "rho_avg":       rho_favg,
        "phi_avg":       phi_favg,
        "rho_std":       rho_fstd,
        "phi_std":       phi_fstd,
        "rho_global":    np.float64(rho_global),
        "rho_global_std": np.float64(_rho_gstd),
        "n_sites":       np.int32(n_sites),
    }


# ---------------------------------------------------------------------------
# Load collection
# ---------------------------------------------------------------------------
if not os.path.isfile(COLL_FILE):
    sys.exit(f" Collection file not found: {COLL_FILE}")

data    = np.load(COLL_FILE, allow_pickle=True)
records = data["records"]
print(f"Loaded {len(records)} stations from {COLL_FILE}")

# _INVAR_SPEC maps tag → (src_key, extractor, fold_phase)
#   src_key    : key to look up in the record dict
#   extractor  : callable(array) → (n_freq,) complex, or None for direct use
#   fold_phase : if True, fold arg(Z) into [0°, 90°] after computing phi
_INVAR_SPEC: dict[str, tuple[str, object, bool]] = {}
if PLOT_SSQ:
    _INVAR_SPEC["ssq"] = ("Zssq", None, False)
if PLOT_DET:
    _INVAR_SPEC["det"] = ("Zdet", None, False)
if PLOT_AVG:
    _INVAR_SPEC["avg"] = ("Zavg", None, True)   # sqrt branch cut → fold to [0°,90°]

if not _INVAR_SPEC:
    sys.exit(" All PLOT_* flags are False — nothing to do.")

_INVAR_LABELS = {
    "ssq": "Zssq", "det": "Zdet", "avg": "Zavg",
}

invar_data: dict[str, dict] = {
    tag: {"stations": [], "freqs": [], "rho": [], "phi": []}
    for tag in _INVAR_SPEC
}

for rec in records:
    freq = rec.get("freq")
    if freq is None:
        continue
    freq = np.asarray(freq, dtype=float)
    if freq.size == 0:
        continue

    # Apply frequency range mask
    mask = np.ones(freq.size, dtype=bool)
    if FREQ_RANGE[0] is not None:
        mask &= freq >= FREQ_RANGE[0]
    if FREQ_RANGE[1] is not None:
        mask &= freq <= FREQ_RANGE[1]
    if not mask.any():
        continue
    freq = freq[mask]

    sta = str(rec.get("station", "?"))

    for tag, (src_key, extractor, fold_phase) in _INVAR_SPEC.items():
        raw = rec.get(src_key)
        if raw is None:
            print(f"  Skipping {sta} for {tag} — key '{src_key}' absent")
            continue
        raw = np.asarray(raw, dtype=complex)
        if raw.size == 0:
            continue
        # slice to masked frequencies (raw is (n_freq,) or (n_freq,2,2))
        raw = raw[mask] if raw.ndim == 1 else raw[mask, ...]
        with np.errstate(invalid="ignore"):
            Z_invar = extractor(raw) if extractor is not None else raw
        rho, phi = _rho_phi_from_zinvar(freq, Z_invar)
        if fold_phase:
            # sqrt branch cut places arg(Zavg) anywhere in (-180°,180°];
            # fold into [0°, 90°]: map to [0°,180°] then mirror about 90°.
            phi = 90.0 - np.abs((phi % 180.0) - 90.0)
        invar_data[tag]["stations"].append(sta)
        invar_data[tag]["freqs"].append(freq)
        invar_data[tag]["rho"].append(rho)
        invar_data[tag]["phi"].append(phi)

# ---------------------------------------------------------------------------
# Compute averages, plot, and save NPZ
# ---------------------------------------------------------------------------
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

for tag in _INVAR_SPEC:
    d = invar_data[tag]
    ns_used = len(d["stations"])
    label   = _INVAR_LABELS[tag]
    print(f"\n{label}: {ns_used} stations")
    if ns_used == 0:
        print("  No data — skipping.")
        continue

    avg = _plot_invar(
        tag=tag,
        label=label,
        stations=d["stations"],
        all_freqs=d["freqs"],
        all_rho=d["rho"],
        all_phi=d["phi"],
    )

    npz_out = os.path.join(DATA_DIR, f"{NAME_STR}_{tag}_avg.npz")
    np.savez_compressed(npz_out, **avg)
    print(f"  Wrote {npz_out}")
