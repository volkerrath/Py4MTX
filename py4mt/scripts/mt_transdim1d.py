#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script-style driver for transdimensional (rjMCMC) 1-D MT inversion.

This is intentionally NOT a CLI.  Edit the USER CONFIG and run:

    python mt_transdim1d.py

Preserved conventions: PY4MTX environment variables, explicit sys.path
setup, startup title print, example in-file model (MODEL0 / MODEL0_ANISO).
Helpers are imported from transdim.py; plotting from transdim_viz.py.
Data I/O uses data_proc.py (load_edi, load_npz, compute_rhophas, compute_pt).

The sampler uses reversible-jump MCMC (Green 1995) so that the number of
layers *k* is itself a free parameter.  Multiple independent chains are
run in parallel via joblib.  Optionally uses the anisotropic 1-D MT
forward model from aniso.py (set USE_ANISO = True).

@author:    Volker Rath (DIAS)
@project:   py4mt — Python for Magnetotellurics
@created:   2026-03-07 — Claude (Opus 4.6, Anthropic)
@modified:  2026-03-07 — anisotropic example block, viz split
@modified:  2026-03-08 — data I/O via data_proc (load_edi, load_npz, compute_rhophas, compute_pt)
"""

from __future__ import annotations

import os
import sys
import inspect
import warnings
from pathlib import Path
import time

import numpy as np

# Optional: suppress noisy FutureWarnings
SUPPRESS_FUTUREWARNINGS = True
if SUPPRESS_FUTUREWARNINGS:
    warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
#  Environment variables / path settings
# =============================================================================
PY4MTX_DATA = os.environ.get("PY4MTX_DATA", "")
PY4MTX_ROOT = os.environ.get("PY4MTX_ROOT", "")

if not PY4MTX_ROOT:
    sys.exit("PY4MTX_ROOT not set! Exit.")
if not PY4MTX_DATA:
    sys.exit("PY4MTX_DATA not set! Exit.")

mypath = [
    str(Path(PY4MTX_ROOT) / "py4mt" / "modules"),
    str(Path(PY4MTX_ROOT) / "py4mt" / "scripts"),
]
for pth in mypath:
    if pth and pth not in sys.path and Path(pth).exists():
        sys.path.insert(0, pth)

import util
from version import versionstrg

import data_proc
from data_proc import (
    get_edi_list, load_edi, load_npz,
    compute_rhophas, compute_pt,
)

import transdim
import transdim_viz

version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())

titstrng = util.print_title(version=version, fname=fname, out=False)
print(titstrng + "\n\n")


# =============================================================================
#  Example starting model — ISOTROPIC
# =============================================================================
#
#  For the transdimensional sampler the starting model only defines the
#  *initial* number of layers and their properties; k is free to change.
#  The MODEL0 dict uses the same schema as mt_aniso1d_sampler.py so that
#  existing model NPZ files can be re-used.
#
N_LAYER = 4
H_M = np.r_[100.0, 400.0, 1000.0, 0.0]     # last entry = half-space (ignored)

RHO_BG = 100.0         # Ohm·m

MODEL0 = {
    "prior_name":  "transdim_start_iso",
    "h_m":         H_M,
    "sigma_min":   (1.0 / RHO_BG) * np.ones(N_LAYER, dtype=float),
    "sigma_max":   (1.0 / RHO_BG) * np.ones(N_LAYER, dtype=float),
    "strike_deg":  np.zeros(N_LAYER, dtype=float),
    "is_iso":      np.ones(N_LAYER, dtype=bool),
    "is_fix":      np.zeros(N_LAYER, dtype=bool),
}


# =============================================================================
#  Example starting model — ANISOTROPIC
# =============================================================================
#
#  Use this with USE_ANISO = True.
#
#  The model has 6 layers with moderate anisotropy (ratio ~ 3–5) in the
#  middle layers.  sigma_min corresponds to the *maximum* conductivity
#  direction (= minimum resistivity), sigma_max to the minimum
#  conductivity direction (= maximum resistivity).
#
#  To activate: set  MODEL_DIRECT = MODEL0_ANISO  in the USER CONFIG.
#
N_LAYER_ANISO = 6
H_M_ANISO = np.r_[
    50.0,       # shallow overburden
    150.0,      # transition
    500.0,      # anisotropic zone
    800.0,      # deep anisotropic zone
    1500.0,     # lower crust
    0.0,        # half-space (ignored)
]

#  rho_max (maximum horizontal resistivity, Ohm·m)
RHO_MAX_ANISO = np.array([200.0, 300.0, 500.0, 800.0, 300.0, 100.0])
#  rho_min (minimum horizontal resistivity, Ohm·m)
#  Ratio ≈ 1 for isotropic layers; > 1 for anisotropic layers.
RHO_MIN_ANISO = np.array([200.0, 100.0, 100.0, 200.0, 300.0, 100.0])
#  strike angles (degrees)
STRIKE_ANISO = np.array([0.0, 30.0, 45.0, 45.0, 0.0, 0.0])

MODEL0_ANISO = {
    "prior_name":  "transdim_start_aniso",
    "h_m":         H_M_ANISO,
    "sigma_min":   1.0 / RHO_MAX_ANISO,    # note: sigma_min ↔ rho_max
    "sigma_max":   1.0 / RHO_MIN_ANISO,    # note: sigma_max ↔ rho_min
    "strike_deg":  STRIKE_ANISO,
    "is_iso":      (RHO_MAX_ANISO == RHO_MIN_ANISO),
    "is_fix":      np.zeros(N_LAYER_ANISO, dtype=bool),
}


# =============================================================================
#  USER CONFIG — forward model
# =============================================================================

USE_ANISO = False                # True → anisotropic forward model (aniso.py)

# =============================================================================
#  USER CONFIG — data
# =============================================================================

MCMC_DATA = PY4MTX_ROOT + "/py4mt/data/edi/mcmc/"

# Input format: "edi" or "npz".
#   "edi" — read .edi files via data_proc.load_edi(); rho/phase computed
#           automatically from Z via data_proc.compute_rhophas().
#   "npz" — read .npz files via data_proc.load_npz(); expects arrays
#           written by data_proc.save_npz() (freq, Z, Z_err, rho, …).
INPUT_FORMAT = "npz"

# For INPUT_FORMAT = "edi":  directory scanned by data_proc.get_edi_list()
EDI_DIR = MCMC_DATA

# For INPUT_FORMAT = "npz":  glob pattern
INPUT_GLOB = MCMC_DATA + "*proc.npz"

MODEL_NPZ = MCMC_DATA + "model0.npz"

# Set MODEL_DIRECT to one of:
#   MODEL0         — isotropic starting model (default)
#   MODEL0_ANISO   — anisotropic starting model
#   None           — load from MODEL_NPZ file
MODEL_DIRECT = MODEL0
MODEL_DIRECT_SAVE_PATH = MODEL_NPZ
MODEL_DIRECT_OVERWRITE = True

# Data uncertainties (used as fallback when Z_err is absent or as floor)
NOISE_LEVEL = 0.02               # relative noise in log10(rho_a) space
SIGMA_FLOOR = 0.0

# Error estimation for rho/phase (passed to data_proc.compute_rhophas)
ERR_METHOD = "analytic"          # "none", "analytic", "bootstrap", "both"
ERR_NSIM = 200                   # Monte-Carlo samples (bootstrap only)

# Compute phase tensor from Z (for QC plots)
COMPUTE_PT = True

# =============================================================================
#  USER CONFIG — prior bounds
# =============================================================================

K_MIN = 1                        # minimum number of internal interfaces
K_MAX = 20                       # maximum number of internal interfaces
DEPTH_MIN = 5.0                  # shallowest allowed interface [m]
DEPTH_MAX = 5000.0               # deepest allowed interface [m]
LOG10_RHO_BOUNDS = (-1.0, 4.0)   # log10(Ohm·m) bounds

# Anisotropy prior bounds (only used when USE_ANISO = True)
LOG10_ANISO_BOUNDS = (0.0, 1.5)  # log10(rho_max/rho_min)
STRIKE_BOUNDS_DEG = (-90.0, 90.0)

# =============================================================================
#  USER CONFIG — sampler
# =============================================================================

N_ITERATIONS = 200_000
BURN_IN = 50_000
THIN = 10

# Proposal weights: (birth, death, move, change)
PROPOSAL_WEIGHTS = (0.20, 0.20, 0.25, 0.35)

# Proposal standard deviations — isotropic parameters
SIGMA_BIRTH_RHO = 0.10           # log10(rho) perturbation on birth
SIGMA_MOVE_Z = 100.0             # interface depth perturbation [m]
SIGMA_CHANGE_RHO = 0.15          # log10(rho) perturbation on change

# Proposal standard deviations — anisotropy parameters
# (only used when USE_ANISO = True)
SIGMA_BIRTH_ANISO = 0.10         # log10(ratio) perturbation on birth
SIGMA_BIRTH_STRIKE = 10.0        # strike perturbation on birth [deg]
SIGMA_CHANGE_ANISO = 0.05        # log10(ratio) perturbation on change
SIGMA_CHANGE_STRIKE = 5.0        # strike perturbation on change [deg]

# =============================================================================
#  USER CONFIG — parallel chains
# =============================================================================

N_CHAINS = 4
N_JOBS = -1                       # -1 = all CPUs; 1 = sequential (debug)
BASE_SEED = 42                    # chain i gets seed = BASE_SEED + i

# =============================================================================
#  USER CONFIG — output
# =============================================================================

OUTDIR = MCMC_DATA + "rjmcmc_" + ("aniso" if USE_ANISO else "iso")

DEPTH_GRID_MAX = 3000.0           # plotting depth limit [m]

# Quantile pairs for summary (mirroring mt_aniso1d_sampler.py convention)
QPAIRS = ((5, 95), (10, 90), (25, 75))

PROGRESSBAR = True                # verbose chain output


# =============================================================================
#  Build objects from USER CONFIG
# =============================================================================

prior = transdim.Prior(
    k_min=K_MIN,
    k_max=K_MAX,
    depth_min=DEPTH_MIN,
    depth_max=DEPTH_MAX,
    log_rho_min=LOG10_RHO_BOUNDS[0],
    log_rho_max=LOG10_RHO_BOUNDS[1],
    log_aniso_min=LOG10_ANISO_BOUNDS[0],
    log_aniso_max=LOG10_ANISO_BOUNDS[1],
    strike_min=STRIKE_BOUNDS_DEG[0],
    strike_max=STRIKE_BOUNDS_DEG[1],
)

config = transdim.RjMCMCConfig(
    n_iterations=N_ITERATIONS,
    burn_in=BURN_IN,
    thin=THIN,
    proposal_weights=PROPOSAL_WEIGHTS,
    sigma_birth_rho=SIGMA_BIRTH_RHO,
    sigma_move_z=SIGMA_MOVE_Z,
    sigma_change_rho=SIGMA_CHANGE_RHO,
    sigma_birth_aniso=SIGMA_BIRTH_ANISO,
    sigma_birth_strike=SIGMA_BIRTH_STRIKE,
    sigma_change_aniso=SIGMA_CHANGE_ANISO,
    sigma_change_strike=SIGMA_CHANGE_STRIKE,
    verbose=PROGRESSBAR,
)


# =============================================================================
#  Helper: convert MODEL0-style dict → transdim.LayeredModel
# =============================================================================

def _model0_to_layered(m0: dict, use_aniso: bool) -> transdim.LayeredModel:
    """Convert a py4mt model dict to a transdim LayeredModel.

    The dict is expected to have ``h_m``, ``sigma_min``, ``sigma_max``,
    ``strike_deg`` (same schema as mt_aniso1d_sampler.py MODEL0).
    """
    h_m = np.asarray(m0["h_m"], dtype=float)
    sigma_max = np.asarray(m0["sigma_max"], dtype=float)     # S/m
    sigma_min = np.asarray(m0["sigma_min"], dtype=float)     # S/m

    # Resistivities: rho_max = 1/sigma_min (the "larger" resistivity)
    tiny = np.finfo(float).tiny
    rho_max = 1.0 / np.maximum(sigma_min, tiny)

    # Interface depths from cumulative thicknesses (drop basement h=0)
    mask = h_m > 0
    thicknesses = h_m[mask]
    interfaces = np.cumsum(thicknesses)

    # log10(rho) per layer — one value per layer including basement
    log_rho = np.log10(rho_max)

    if use_aniso:
        rho_min = 1.0 / np.maximum(sigma_max, tiny)
        aniso_ratios = np.maximum(rho_max / rho_min, 1.0)
        strikes = np.asarray(m0["strike_deg"], dtype=float)
        return transdim.LayeredModel(interfaces, log_rho, aniso_ratios, strikes)

    return transdim.LayeredModel(interfaces, log_rho)


# =============================================================================
#  Helper: load site data via data_proc
# =============================================================================

def _load_site(path: str | Path) -> dict:
    """Load a site from an EDI or NPZ file using ``data_proc`` routines.

    Returns a dict with the keys expected by the transdim sampler and
    QC plots:

        frequencies, rho_a, sigma, station,
        rho_a_yx, sigma_yx   (if anisotropic data present)
        Z, Z_err             (if impedance tensor available)
        PT, PT_err           (if phase tensor available)
    """
    path = Path(path)
    ext = path.suffix.lower()

    # ---- Load raw data via data_proc --------------------------------------
    if ext == ".edi":
        dd = load_edi(str(path))
    elif ext == ".npz":
        dd = load_npz(str(path))
    else:
        raise ValueError(f"Unsupported input format: {ext!r} ({path})")

    # ---- Frequencies (data_proc uses "freq") ------------------------------
    if "freq" in dd:
        frequencies = np.asarray(dd["freq"], dtype=float).ravel()
    elif "frequencies" in dd:
        frequencies = np.asarray(dd["frequencies"], dtype=float).ravel()
    elif "period" in dd:
        frequencies = 1.0 / np.asarray(dd["period"], dtype=float).ravel()
    else:
        raise KeyError(f"No 'freq', 'frequencies', or 'period' in {path}")

    # ---- Station name -----------------------------------------------------
    station = dd.get("station", path.stem)
    if isinstance(station, np.ndarray):
        station = str(station.item()) if station.ndim == 0 else str(station)

    # ---- Impedance tensor Z -----------------------------------------------
    Z = dd.get("Z")
    Z_err = dd.get("Z_err")
    err_kind = str(dd.get("err_kind", "var")).strip().lower()
    if err_kind.startswith("std"):
        err_kind = "std"
    else:
        err_kind = "var"

    # ---- Compute apparent resistivity / phase from Z if not present -------
    rho = dd.get("rho")
    rho_err = dd.get("rho_err")

    if rho is None and Z is not None:
        Z = np.asarray(Z, dtype=complex)
        _Ze = np.asarray(Z_err) if Z_err is not None else None
        rho, phi, rho_err, phi_err = compute_rhophas(
            freq=frequencies, Z=Z, Z_err=_Ze,
            err_kind=err_kind,
            err_method=ERR_METHOD,
            nsim=ERR_NSIM,
        )

    # ---- Extract xy (and optionally yx) apparent resistivity ---------------
    if rho is not None:
        rho = np.asarray(rho)
        if rho.ndim == 3 and rho.shape[1:] == (2, 2):
            # rho has shape (n, 2, 2) — extract components
            rho_a_xy = rho[:, 0, 1]
            rho_a_yx = rho[:, 1, 0]
        elif rho.ndim == 1:
            rho_a_xy = rho
            rho_a_yx = None
        else:
            rho_a_xy = rho.ravel()
            rho_a_yx = None
    elif "rho_a" in dd:
        rho_a_xy = np.asarray(dd["rho_a"], dtype=float).ravel()
        rho_a_yx = dd.get("rho_a_yx")
        if rho_a_yx is not None:
            rho_a_yx = np.asarray(rho_a_yx, dtype=float).ravel()
    elif "rho_a_xy" in dd:
        rho_a_xy = np.asarray(dd["rho_a_xy"], dtype=float).ravel()
        rho_a_yx = dd.get("rho_a_yx")
        if rho_a_yx is not None:
            rho_a_yx = np.asarray(rho_a_yx, dtype=float).ravel()
    else:
        raise KeyError(
            f"Cannot determine apparent resistivity from {path}.  "
            "Need 'Z', 'rho', 'rho_a', or 'rho_a_xy'."
        )

    # ---- Uncertainties in log10(rho_a) space ------------------------------
    if rho_err is not None:
        rho_err = np.asarray(rho_err)
        if rho_err.ndim == 3 and rho_err.shape[1:] == (2, 2):
            rho_err_xy = rho_err[:, 0, 1]
            rho_err_yx = rho_err[:, 1, 0]
        elif rho_err.ndim == 1:
            rho_err_xy = rho_err
            rho_err_yx = None
        else:
            rho_err_xy = rho_err.ravel()
            rho_err_yx = None

        # Convert from linear rho_err to log10(rho) uncertainty:
        #   σ_log10 ≈ rho_err / (rho * ln(10))
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma_xy = np.where(
                rho_a_xy > 0,
                rho_err_xy / (rho_a_xy * np.log(10)),
                NOISE_LEVEL,
            )
        sigma_xy = np.maximum(sigma_xy, SIGMA_FLOOR)
        sigma_xy = np.where(np.isfinite(sigma_xy), sigma_xy, NOISE_LEVEL)

        sigma_yx = None
        if rho_err_yx is not None and rho_a_yx is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                sigma_yx = np.where(
                    rho_a_yx > 0,
                    rho_err_yx / (rho_a_yx * np.log(10)),
                    NOISE_LEVEL,
                )
            sigma_yx = np.maximum(sigma_yx, SIGMA_FLOOR)
            sigma_yx = np.where(np.isfinite(sigma_yx), sigma_yx, NOISE_LEVEL)
    elif "sigma" in dd:
        sigma_xy = np.maximum(
            np.asarray(dd["sigma"], dtype=float).ravel(), SIGMA_FLOOR)
        sigma_yx = dd.get("sigma_yx")
        if sigma_yx is not None:
            sigma_yx = np.maximum(
                np.asarray(sigma_yx, dtype=float).ravel(), SIGMA_FLOOR)
    else:
        sigma_xy = np.full(len(frequencies), NOISE_LEVEL)
        sigma_yx = None

    # ---- Phase tensor (for QC plots) --------------------------------------
    PT = dd.get("P", dd.get("PT"))
    PT_err = dd.get("P_err", dd.get("PT_err"))
    if PT is None and COMPUTE_PT and Z is not None:
        Z_arr = np.asarray(Z, dtype=complex)
        _Ze = np.asarray(Z_err) if Z_err is not None else None
        PT, PT_err = compute_pt(Z_arr, _Ze, err_kind=err_kind)

    # ---- Build output dict ------------------------------------------------
    result = {
        "station": station,
        "frequencies": frequencies,
        "rho_a": rho_a_xy,
        "sigma": sigma_xy,
    }

    if rho_a_yx is not None:
        result["rho_a_yx"] = rho_a_yx
    if sigma_yx is not None:
        result["sigma_yx"] = sigma_yx
    if Z is not None:
        result["Z"] = np.asarray(Z, dtype=complex)
    if Z_err is not None:
        result["Z_err"] = np.asarray(Z_err, dtype=float)
    if PT is not None:
        result["PT"] = np.asarray(PT, dtype=float)
    if PT_err is not None:
        result["PT_err"] = np.asarray(PT_err, dtype=float)

    return result


# =============================================================================
#  Ensure output directory exists
# =============================================================================

def _ensure_dir(d: str | Path) -> Path:
    p = Path(d)
    p.mkdir(parents=True, exist_ok=True)
    return p


# =============================================================================
#  Build quantile summary (mirrors mcmc.build_summary_npz convention)
# =============================================================================

def _build_summary(
    station: str,
    results: dict,
    depth_max: float,
    qpairs: tuple,
    use_aniso: bool,
) -> dict:
    """Build a summary dict with posterior statistics and quantiles."""
    depth_grid = np.linspace(1, depth_max, 500)
    prof = transdim.compute_posterior_profile(results["models"], depth_grid)

    summary = {
        "station": station,
        "depth_grid": depth_grid,
        "rho_mean": prof["mean"],
        "rho_median": prof["median"],
        "rho_p05": prof["p05"],
        "rho_p95": prof["p95"],
        "n_layers_trace": results["n_layers"],
        "n_layers_median": float(np.median(results["n_layers"])),
        "n_layers_mode": int(np.bincount(results["n_layers"]).argmax()),
        "gelman_rubin": results.get("gelman_rubin", np.nan),
        "elapsed_s": results.get("elapsed_s", 0.0),
        "acceptance": results["acceptance"],
    }

    # Quantile pairs
    log_ens = np.log10(prof["ensemble"])
    for qlo, qhi in qpairs:
        summary[f"rho_p{qlo:02d}"] = 10 ** np.percentile(log_ens, qlo, axis=0)
        summary[f"rho_p{qhi:02d}"] = 10 ** np.percentile(log_ens, qhi, axis=0)

    if use_aniso:
        aprof = transdim.compute_posterior_aniso_profile(
            results["models"], depth_grid)
        summary.update(aprof)

    return summary


# =============================================================================
#  Run sampler
# =============================================================================

outdir = _ensure_dir(OUTDIR)

# ---- Sanity checks ---------------------------------------------------------
if USE_ANISO and not transdim.has_aniso():
    sys.exit(
        "USE_ANISO=True but aniso.py not found on PYTHONPATH.  "
        "Place aniso.py in the working directory or set PYTHONPATH.  Exit."
    )

# ---- Starting model --------------------------------------------------------
if MODEL_DIRECT is not None:
    initial_model = _model0_to_layered(MODEL_DIRECT, USE_ANISO)
    if MODEL_DIRECT_OVERWRITE or not Path(MODEL_DIRECT_SAVE_PATH).exists():
        np.savez_compressed(
            MODEL_DIRECT_SAVE_PATH,
            **{k: np.asarray(v) for k, v in MODEL_DIRECT.items()},
        )
else:
    # Load from NPZ (same schema)
    with np.load(MODEL_NPZ, allow_pickle=True) as npz:
        m0 = {k: npz[k] for k in npz.files}
    initial_model = _model0_to_layered(m0, USE_ANISO)

print(f"Starting model: k={initial_model.k} interfaces, "
      f"{initial_model.n_layers} layers, "
      f"anisotropic={initial_model.is_anisotropic}")
if initial_model.is_anisotropic:
    print(f"  rho_max       = {initial_model.get_resistivities()}")
    print(f"  aniso_ratio   = {initial_model.aniso_ratios}")
    print(f"  strike [deg]  = {initial_model.strikes}")
print()

# ---- Discover input data files ---------------------------------------------
if INPUT_FORMAT.lower() == "edi":
    in_files = get_edi_list(EDI_DIR, fullpath=True, sort=True)
else:
    import glob
    in_files = sorted(glob.glob(INPUT_GLOB))

if not in_files:
    pat = EDI_DIR if INPUT_FORMAT.lower() == "edi" else INPUT_GLOB
    raise FileNotFoundError(f"No inputs matched: {pat}")

print(f"Found {len(in_files)} input file(s) ({INPUT_FORMAT}).\n")

# ---- Loop over sites -------------------------------------------------------
for f in in_files:
    site = _load_site(f)
    station = str(site.get("station", Path(f).stem))
    print(f"{'='*70}")
    print(f"  Station: {station}")
    print(f"{'='*70}")
    print(f"  Frequencies: {len(site['frequencies'])}, "
          f"range {site['frequencies'].min():.4f}–{site['frequencies'].max():.1f} Hz")
    print()

    # ---- Run parallel rjMCMC -----------------------------------------------
    kw_aniso = {}
    if USE_ANISO and "rho_a_yx" in site:
        kw_aniso["observed_yx"] = site["rho_a_yx"]
        kw_aniso["sigma_yx"] = site.get("sigma_yx", site["sigma"])

    results = transdim.run_parallel_rjmcmc(
        frequencies=site["frequencies"],
        observed=site["rho_a"],
        sigma=site["sigma"],
        prior=prior,
        config=config,
        n_chains=N_CHAINS,
        n_jobs=N_JOBS,
        base_seed=BASE_SEED,
        use_aniso=USE_ANISO,
        **kw_aniso,
    )

    # ---- Summary -----------------------------------------------------------
    print(f"\nPosterior summary for {station}:")
    print(f"  Median number of layers: {np.median(results['n_layers']):.0f}")
    print(f"  Mode number of layers:   "
          f"{np.bincount(results['n_layers']).argmax()}")
    print(f"  R-hat: {results.get('gelman_rubin', np.nan):.4f}")
    print()

    # ---- Save results ------------------------------------------------------
    npz_path = outdir / f"{station}_rjmcmc.npz"
    transdim.save_results_npz(results, npz_path)

    sum_dict = _build_summary(station, results, DEPTH_GRID_MAX, QPAIRS, USE_ANISO)
    sum_path = outdir / f"{station}_rjmcmc_summary.npz"
    np.savez_compressed(str(sum_path), **{
        k: np.asarray(v) if not isinstance(v, (str, dict)) else str(v)
        for k, v in sum_dict.items()
    })
    print(f"Wrote: {sum_path}")

    # ---- Plot (via transdim_viz) -------------------------------------------
    fig_path = outdir / f"{station}_rjmcmc.png"
    transdim_viz.plot_results(
        results,
        true_model=None,            # set to a LayeredModel for synthetic tests
        frequencies=site["frequencies"],
        observed=site["rho_a"],
        depth_max=DEPTH_GRID_MAX,
        use_aniso=USE_ANISO,
        save_path=str(fig_path),
    )

    # ---- QC plot (apparent rho, phase, misfit trace, k histogram) ----------
    qc_path = outdir / f"{station}_rjmcmc_qc.png"
    transdim_viz.plot_qc(
        results,
        frequencies=site["frequencies"],
        observed=site["rho_a"],
        sigma=site["sigma"],
        station=station,
        use_aniso=USE_ANISO,
        observed_Z=site.get("Z", None),
        observed_Z_err=site.get("Z_err", None),
        z_comps=("xy", "yx") if not USE_ANISO else ("xx", "xy", "yx", "yy"),
        show_pt=site.get("PT", None) is not None or (
            USE_ANISO and site.get("Z", None) is not None),
        observed_PT=site.get("PT", None),
        observed_PT_err=site.get("PT_err", None),
        save_path=str(qc_path),
    )

    # ---- Posterior model plot (2-D histogram + change-point frequency) ------
    model_path = outdir / f"{station}_rjmcmc_model.png"
    transdim_viz.plot_posterior_model(
        results,
        depth_max=DEPTH_GRID_MAX,
        true_model=None,            # set to a LayeredModel for synthetic tests
        station=station,
        use_aniso=USE_ANISO,
        save_path=str(model_path),
    )

    print()

print("\nDone.\n")
