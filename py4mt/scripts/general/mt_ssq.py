#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mt_ssq.py
=========
Load the sum-of-squares impedance invariant (Zssq) for all sites in a
collection NPZ, compute apparent resistivity and phase per site, form
frequency-wise log averages across sites, compute a global scalar log-average
rho_a, and produce a single rho/phase plot with all sites overlaid and the
global average annotated.

Quantities
----------
Zssq  — complex scalar invariant, (n_freq,), mV/km/nT
        defined as  Zssq = sqrt( (Zxx² + Zxy² + Zyx² + Zyy²) / 2 )
rho_a — apparent resistivity from Zssq [Ω·m]:
        rho_a(f) = mu0 * |Zssq|² * 1e6 / (2π f)
phi   — phase of Zssq [°]: arctan(Im/Re)

Averaging
---------
Per-frequency log average : geometric mean over all sites that have data
                             at that frequency bin (nearest-bin matching).
Global log average         : geometric mean of all per-site, all-frequency
                             rho_a values (scalar).

@author:    vrath (VR)
@project:   py4mt — Python for Magnetotellurics
@created:   2026-06-12
@modified:  2026-06-12 — initial version; Claude Sonnet 4.6 (Anthropic)
@modified:  2026-06-12 — annotation to axes-fraction lower-left; LEGEND_SITES switch; Claude Sonnet 4.6 (Anthropic)
@modified:  2026-06-12 — legend fontsize/ncol scaled by LEGEND_SITES (fs=10/ncol=1 vs fs=5/ncol=4); Claude Sonnet 4.6 (Anthropic)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WORK_DIR   = "/home/vrath/Py4MTX/py4mt/data/rto/ubinas/edi/proc/"
COLL_FILE  = WORK_DIR + "TEST_test_proc_collection.npz"
PLOT_DIR   = WORK_DIR + "../plots/"
NAME_STR   = "Ubinas_ssq2"
PLOT_FORMAT = [".pdf", ".png"]

# Frequency-bin matching tolerance (fraction of a decade).
# Sites whose nearest grid-bin distance exceeds this are excluded from that bin.
BIN_TOL_DEC = 0.1   # ±0.1 decade

# Plot appearance
XLIM = (1.e-4, 1.e+4)   # period axis limits [s]  (None = auto)
YLIM_RHO   = (1.e+0, 1.e+4)
YLIM_PHASE = (0., 90.)

LEGEND_SITES = True   # if False, legend shows only the averages line

SITE_ALPHA  = 0.35
SITE_LW     = 1.8
AVG_LW      = 3.0
AVG_COLOR   = "red"
SITE_CMAP   = "tab20"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_MU0: float = 4.0 * np.pi * 1.0e-7
_Z_MT_TO_SI_SQ: float = _MU0 ** 2 * 1.0e6   # (mu0*1e3)^2 / mu0  → see data_proc


def _rho_phi_from_zssq(
    freq: np.ndarray,
    Zssq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute rho_a [Ω·m] and phase [°] from the scalar Zssq invariant."""
    omega = 2.0 * np.pi * freq
    rho = _Z_MT_TO_SI_SQ * np.abs(Zssq) ** 2 / (_MU0 * omega)
    phi = np.degrees(np.angle(Zssq))
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frequency log-average rho_a and arithmetic-average phase."""
    n = len(grid)
    rho_avg = np.full(n, np.nan)
    phi_avg = np.full(n, np.nan)
    counts  = np.zeros(n, dtype=int)

    log_grid = np.log10(grid)

    for freq_s, rho_s, phi_s in zip(all_freqs, all_rho, all_phi):
        log_fs = np.log10(freq_s)
        for k, lf in enumerate(log_grid):
            dist = np.abs(log_fs - lf)
            idx  = np.argmin(dist)
            if dist[idx] > tol_dec:
                continue
            if not np.isfinite(rho_s[idx]) or rho_s[idx] <= 0:
                continue
            # accumulate log-rho and phase for averaging
            if np.isnan(rho_avg[k]):
                rho_avg[k] = np.log(rho_s[idx])
                phi_avg[k] = phi_s[idx]
            else:
                rho_avg[k] += np.log(rho_s[idx])
                phi_avg[k] += phi_s[idx]
            counts[k] += 1

    valid = counts > 0
    rho_avg[valid] /= counts[valid]
    rho_avg[valid]  = np.exp(rho_avg[valid])
    phi_avg[valid] /= counts[valid]
    rho_avg[~valid] = np.nan
    phi_avg[~valid] = np.nan

    return grid, rho_avg, phi_avg


# ---------------------------------------------------------------------------
# Load collection
# ---------------------------------------------------------------------------
if not os.path.isfile(COLL_FILE):
    sys.exit(f" Collection file not found: {COLL_FILE}")

data = np.load(COLL_FILE, allow_pickle=True)
records = data["records"]
ns = len(records)
print(f"Loaded {ns} stations from {COLL_FILE}")

stations:  list[str]        = []
all_freqs: list[np.ndarray] = []
all_rho:   list[np.ndarray] = []
all_phi:   list[np.ndarray] = []

for rec in records:
    Zssq = rec.get("Zssq")
    freq = rec.get("freq")
    if Zssq is None or freq is None:
        print(f"  Skipping {rec.get('station','?')} — no Zssq")
        continue
    freq  = np.asarray(freq,  dtype=float)
    Zssq  = np.asarray(Zssq,  dtype=complex)
    if freq.size == 0 or Zssq.size == 0:
        continue

    rho, phi = _rho_phi_from_zssq(freq, Zssq)

    stations.append(str(rec.get("station", "?")))
    all_freqs.append(freq)
    all_rho.append(rho)
    all_phi.append(phi)

ns_used = len(stations)
print(f"Using {ns_used} stations with valid Zssq")

# ---------------------------------------------------------------------------
# (a) Per-site rho/phase  →  stored in all_rho / all_phi (done above)
# (b) Frequency-wise log average
# (c) Global log average of rho_a
# ---------------------------------------------------------------------------
freq_grid = _build_freq_grid(all_freqs)
grid, rho_favg, phi_favg = _freqwise_log_avg(
    freq_grid, all_freqs, all_rho, all_phi
)

rho_global = _log_avg(all_rho)
print(f"Global log-average rho_a = {rho_global:.1f} Ω·m")

# ---------------------------------------------------------------------------
# (d) Plot
# ---------------------------------------------------------------------------
os.makedirs(PLOT_DIR, exist_ok=True)

cmap   = plt.get_cmap(SITE_CMAP)
colors = [cmap(i % cmap.N) for i in range(ns_used)]

fig, (ax_rho, ax_phi) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
fig.suptitle("Zssq invariant — apparent resistivity & phase", fontsize=11)

# --- rho panel ---
for i, (freq_s, rho_s, sta) in enumerate(zip(all_freqs, all_rho, stations)):
    per_s = _period(freq_s)
    ax_rho.loglog(per_s, rho_s,
                  color=colors[i], alpha=SITE_ALPHA, lw=SITE_LW,
                  label=sta)

per_grid = _period(grid)
valid = np.isfinite(rho_favg)
ax_rho.loglog(per_grid[valid], rho_favg[valid],
              color=AVG_COLOR, lw=AVG_LW, ls="--",
              label="freq-wise avg", zorder=5)

ax_rho.axhline(rho_global, color="black", lw=3., ls=":",
               zorder=4, alpha=0.8)
ax_rho.annotate(
    f"$\\bar{{\\rho}}_{{\\!a}}$ = {rho_global:.0f} Ω·m",
    xy=(0.02, 0.05), xycoords="axes fraction",
    fontsize=16, color="black",
    va="bottom", ha="left",
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
    per_s = _period(freq_s)
    ax_phi.semilogx(per_s, phi_s,
                    color=colors[i], alpha=SITE_ALPHA, lw=SITE_LW)

valid_p = np.isfinite(phi_favg)
ax_phi.semilogx(per_grid[valid_p], phi_favg[valid_p],
                color=AVG_COLOR, lw=AVG_LW, ls="--", zorder=5)

ax_phi.axhline(0, color="grey", lw=0.6, ls="-")
ax_phi.axhline(45, color="grey", lw=0.6, ls=":")
ax_phi.set_ylabel("Phase (°)")
ax_phi.set_xlabel("Period (s)")
if YLIM_PHASE:
    ax_phi.set_ylim(YLIM_PHASE)
if XLIM:
    ax_phi.set_xlim(XLIM)
ax_phi.yaxis.set_major_locator(mticker.MultipleLocator(45))
ax_phi.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)

# Legend: always show averages; sites optional
handles, labels = ax_rho.get_legend_handles_labels()
if not LEGEND_SITES:
    handles = [h for h, l in zip(handles, labels) if l == "freq-wise avg"]
    labels  = ["freq-wise avg"]
    _leg_fs, _leg_ncol = 10, 1
else:
    _leg_fs, _leg_ncol = 5, 4
ax_rho.legend(handles, labels,
              fontsize=_leg_fs, ncol=_leg_ncol,
              loc="best", framealpha=0.7,
              handlelength=1.2, columnspacing=0.6, labelspacing=0.2)

fig.tight_layout(rect=[0, 0, 1, 0.96])

for fmt in PLOT_FORMAT:
    out = PLOT_DIR + NAME_STR + fmt
    plt.savefig(out, dpi=300)
    print(f"Wrote {out}")

plt.show()
