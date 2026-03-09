#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script-style driver for transdimensional (rjMCMC) 1-D MT inversion.

This is intentionally NOT a CLI.  Edit the USER CONFIG and run:

    python mt_transdim1d.py

Preserved conventions: PY4MTX environment variables, explicit sys.path
setup, startup title print, example in-file model (MODEL0 / MODEL0_ANISO).
Helpers are imported from transdim.py; plotting from transdim_viz.py.
Data I/O uses data_proc.py (load_edi, load_npz, compute_rhophas, compute_pt)
via transdim.load_site().

The sampler uses reversible-jump MCMC (Green 1995) so that the number of
layers *k* is itself a free parameter.  Multiple independent chains are
run in parallel via joblib.

Likelihood modes
----------------
- **Isotropic** (``USE_ANISO = False``):
  The observed data are the determinant impedance Z_det = sqrt(det(Z)),
  or equivalently the apparent resistivity / phase derived from Z_det.
  The likelihood operates on Re/Im of Z_det (``likelihood_mode="Zdet"``)
  or on log10(ρ_a) from Z_det (``likelihood_mode="rhoa"``).

- **Anisotropic** (``USE_ANISO = True``):
  The observed data are a user-selected subset of Z components
  (default: all four), optionally supplemented by the phase tensor.
  This mirrors the mcmc.py / mt_aniso1d_sampler.py approach.
  The likelihood operates on Re/Im of the selected Z components and
  (optionally) phase tensor entries (``likelihood_mode="Z_comps"``).

@author:    Volker Rath (DIAS)
@project:   py4mt — Python for Magnetotellurics
@created:   2026-03-07 — Claude (Opus 4.6, Anthropic)
@modified:  2026-03-07 — anisotropic example block, viz split
@modified:  2026-03-08 — data I/O via data_proc
@modified:  2026-03-09 — Z_det likelihood for isotropic; Z-component + PT
                          likelihood for anisotropic; helpers moved to transdim.py
"""

from __future__ import annotations

import os
import sys
import inspect
import warnings
from pathlib import Path
import glob

import numpy as np

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
]
for pth in mypath:
    if pth and pth not in sys.path and Path(pth).exists():
        sys.path.append(pth)

import util
from version import versionstrg

import transdim
import transdim_viz

version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())

titstrng = util.print_title(version=version, fname=fname, out=False)
print(titstrng + "\n\n")


# =============================================================================
#  Example starting model — ISOTROPIC
# =============================================================================
N_LAYER = 4
H_M = np.r_[100.0, 400.0, 1000.0, 0.0]

RHO_BG = 100.0

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
N_LAYER_ANISO = 6
H_M_ANISO = np.r_[50.0, 150.0, 500.0, 800.0, 1500.0, 0.0]

RHO_MAX_ANISO = np.array([200.0, 300.0, 500.0, 800.0, 300.0, 100.0])
RHO_MIN_ANISO = np.array([200.0, 100.0, 100.0, 200.0, 300.0, 100.0])
STRIKE_ANISO = np.array([0.0, 30.0, 45.0, 45.0, 0.0, 0.0])

MODEL0_ANISO = {
    "prior_name":  "transdim_start_aniso",
    "h_m":         H_M_ANISO,
    "sigma_min":   1.0 / RHO_MAX_ANISO,
    "sigma_max":   1.0 / RHO_MIN_ANISO,
    "strike_deg":  STRIKE_ANISO,
    "is_iso":      (RHO_MAX_ANISO == RHO_MIN_ANISO),
    "is_fix":      np.zeros(N_LAYER_ANISO, dtype=bool),
}


# =============================================================================
#  USER CONFIG — forward model
# =============================================================================

USE_ANISO = False

# =============================================================================
#  USER CONFIG — likelihood
# =============================================================================
#
#  LIKELIHOOD_MODE controls which data and misfit function are used:
#
#    "Zdet"     — Re/Im of determinant impedance Z_det (isotropic default)
#    "rhoa"     — log10(ρ_a) from Z_det (isotropic alternative)
#    "Z_comps"  — Re/Im of selected Z components + optional PT (anisotropic)
#
#  None → auto-select: "Zdet" for isotropic, "Z_comps" for anisotropic.
#
LIKELIHOOD_MODE = None

# Z components for the "Z_comps" likelihood (anisotropic case)
Z_COMPS = ("xx", "xy", "yx", "yy")

# Include phase tensor in the "Z_comps" likelihood?
USE_PT = True

# Phase tensor components
PT_COMPS = ("xx", "xy", "yx", "yy")


# =============================================================================
#  USER CONFIG — data
# =============================================================================

MCMC_DATA = PY4MTX_ROOT + "/py4mt/data/edi/mcmc/"

INPUT_FORMAT = "npz"
INPUT_GLOB = MCMC_DATA + "*proc.npz"

MODEL_NPZ = MCMC_DATA + "model0.npz"

MODEL_DIRECT = MODEL0
MODEL_DIRECT_SAVE_PATH = MODEL_NPZ
MODEL_DIRECT_OVERWRITE = True

NOISE_LEVEL = 0.02
SIGMA_FLOOR = 0.0

ERR_METHOD = "analytic"
ERR_NSIM = 200

COMPUTE_PT = True

# =============================================================================
#  USER CONFIG — prior bounds
# =============================================================================

K_MIN = 1
K_MAX = 20
DEPTH_MIN = 5.0
DEPTH_MAX = 5000.0
LOG10_RHO_BOUNDS = (-1.0, 4.0)

LOG10_ANISO_BOUNDS = (0.0, 1.5)
STRIKE_BOUNDS_DEG = (-90.0, 90.0)

# =============================================================================
#  USER CONFIG — sampler
# =============================================================================

N_ITERATIONS = 200_000
BURN_IN = 50_000
THIN = 10

PROPOSAL_WEIGHTS = (0.20, 0.20, 0.25, 0.35)

SIGMA_BIRTH_RHO = 0.10
SIGMA_MOVE_Z = 100.0
SIGMA_CHANGE_RHO = 0.15

SIGMA_BIRTH_ANISO = 0.10
SIGMA_BIRTH_STRIKE = 10.0
SIGMA_CHANGE_ANISO = 0.05
SIGMA_CHANGE_STRIKE = 5.0

# =============================================================================
#  USER CONFIG — parallel chains
# =============================================================================

N_CHAINS = 4
N_JOBS = -1
BASE_SEED = 42

# =============================================================================
#  USER CONFIG — output
# =============================================================================

OUTDIR = MCMC_DATA + "rjmcmc_" + ("aniso" if USE_ANISO else "iso")

DEPTH_GRID_MAX = 3000.0

QPAIRS = ((5, 95), (10, 90), (25, 75))

PROGRESSBAR = True


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

# Resolve likelihood mode
if LIKELIHOOD_MODE is not None:
    likelihood_mode = LIKELIHOOD_MODE
elif USE_ANISO:
    likelihood_mode = "Z_comps"
else:
    likelihood_mode = "Zdet"

print(f"Likelihood mode: {likelihood_mode}")
if likelihood_mode.lower() == "z_comps":
    print(f"  Z components:  {Z_COMPS}")
    print(f"  Use PT:        {USE_PT}")
    if USE_PT:
        print(f"  PT components: {PT_COMPS}")
print()


# =============================================================================
#  Run
# =============================================================================

outdir = transdim.ensure_dir(OUTDIR)

# ---- Sanity checks ---------------------------------------------------------
if USE_ANISO and not transdim.has_aniso():
    sys.exit(
        "USE_ANISO=True but aniso.py not found on PYTHONPATH.  "
        "Place aniso.py in the working directory or set PYTHONPATH.  Exit."
    )

if likelihood_mode.lower() == "z_comps" and not USE_ANISO:
    print("WARNING: likelihood_mode='Z_comps' is designed for anisotropic "
          "inversion but USE_ANISO=False.  The isotropic forward model "
          "will be used (Zxx=Zyy=0, Zxy=Z, Zyx=-Z).\n")

# ---- Starting model --------------------------------------------------------
if MODEL_DIRECT is not None:
    initial_model = transdim.model0_to_layered(MODEL_DIRECT, USE_ANISO)
    if MODEL_DIRECT_OVERWRITE or not Path(MODEL_DIRECT_SAVE_PATH).exists():
        np.savez_compressed(
            MODEL_DIRECT_SAVE_PATH,
            **{k: np.asarray(v) for k, v in MODEL_DIRECT.items()},
        )
else:
    with np.load(MODEL_NPZ, allow_pickle=True) as npz:
        m0 = {k: npz[k] for k in npz.files}
    initial_model = transdim.model0_to_layered(m0, USE_ANISO)

print(f"Starting model: k={initial_model.k} interfaces, "
      f"{initial_model.n_layers} layers, "
      f"anisotropic={initial_model.is_anisotropic}")
if initial_model.is_anisotropic:
    print(f"  rho_max       = {initial_model.get_resistivities()}")
    print(f"  aniso_ratio   = {initial_model.aniso_ratios}")
    print(f"  strike [deg]  = {initial_model.strikes}")
print()

# ---- Discover input data files ---------------------------------------------
in_files = sorted(glob.glob(INPUT_GLOB))

if not in_files:
    pat = INPUT_GLOB
    raise FileNotFoundError(f"No inputs matched: {pat}")

print(f"Found {len(in_files)} input file(s) ({INPUT_FORMAT}).\n")

# ---- Loop over sites -------------------------------------------------------
for f in in_files:
    site = transdim.load_site(
        f,
        noise_level=NOISE_LEVEL,
        sigma_floor=SIGMA_FLOOR,
        err_method=ERR_METHOD,
        err_nsim=ERR_NSIM,
        do_compute_pt=COMPUTE_PT,
    )
    station = str(site.get("station", Path(f).stem))
    print(f"{'='*70}")
    print(f"  Station: {station}")
    print(f"{'='*70}")
    print(f"  Frequencies: {len(site['frequencies'])}, "
          f"range {site['frequencies'].min():.4f}–{site['frequencies'].max():.1f} Hz")

    has_Z = "Z" in site
    has_Zdet = "Zdet" in site
    has_PT = "PT" in site
    print(f"  Z tensor:     {'yes' if has_Z else 'no'}")
    print(f"  Z_det:        {'yes' if has_Zdet else 'no'}")
    print(f"  Phase tensor: {'yes' if has_PT else 'no'}")
    print()

    # ---- Prepare likelihood-specific arguments -----------------------------
    lmode = likelihood_mode.lower().strip()

    kw_sampler = dict(
        frequencies=site["frequencies"],
        observed=site["rho_a"],
        sigma=site["sigma"],
        prior=prior,
        config=config,
        n_chains=N_CHAINS,
        n_jobs=N_JOBS,
        base_seed=BASE_SEED,
        use_aniso=USE_ANISO,
        likelihood_mode=lmode,
    )

    if lmode == "zdet":
        # ---- Isotropic: Z_det likelihood -----------------------------------
        if not has_Zdet:
            print("  WARNING: Z_det not available; falling back to rhoa likelihood.")
            kw_sampler["likelihood_mode"] = "rhoa"
        else:
            kw_sampler["observed_Zdet"] = site["Zdet"]
            if "Zdet_err" in site:
                kw_sampler["Zdet_sigma"] = site["Zdet_err"]
            else:
                kw_sampler["Zdet_sigma"] = 0.05 * np.abs(site["Zdet"])
                print("  NOTE: No Zdet_err; using 5% of |Zdet| as uncertainty.")
            # Use Z_det-derived rho_a for the QC/rhoa fallback fields
            if "rho_a_det" in site:
                kw_sampler["observed"] = site["rho_a_det"]
                Zabs = np.abs(site["Zdet"])
                with np.errstate(divide="ignore", invalid="ignore"):
                    sig_log = np.where(
                        Zabs > 0,
                        2.0 * np.asarray(kw_sampler["Zdet_sigma"]) / (Zabs * np.log(10)),
                        NOISE_LEVEL,
                    )
                kw_sampler["sigma"] = np.maximum(sig_log, SIGMA_FLOOR)

    elif lmode == "z_comps":
        # ---- Anisotropic: Z-component + optional PT likelihood -------------
        if not has_Z:
            print("  WARNING: Z tensor not available; falling back to rhoa likelihood.")
            kw_sampler["likelihood_mode"] = "rhoa"
            if USE_ANISO and "rho_a_yx" in site:
                kw_sampler["observed_yx"] = site["rho_a_yx"]
                kw_sampler["sigma_yx"] = site.get("sigma_yx", site["sigma"])
        else:
            kw_sampler["observed_Z"] = site["Z"]
            if "Z_err" in site:
                kw_sampler["observed_Z_err"] = np.asarray(site["Z_err"], dtype=float)
            else:
                kw_sampler["observed_Z_err"] = 0.05 * np.abs(site["Z"]).astype(float)
                print("  NOTE: No Z_err; using 5% of |Z| as uncertainty.")
            kw_sampler["z_comps"] = Z_COMPS
            kw_sampler["use_pt"] = USE_PT and has_PT
            if kw_sampler["use_pt"]:
                kw_sampler["observed_PT"] = site["PT"]
                if "PT_err" in site:
                    kw_sampler["observed_PT_err"] = np.asarray(site["PT_err"], dtype=float)
                else:
                    kw_sampler["observed_PT_err"] = 0.05 * np.maximum(
                        np.abs(site["PT"]), 1e-6).astype(float)
                    print("  NOTE: No PT_err; using 5% of |PT| as uncertainty.")
                kw_sampler["pt_comps"] = PT_COMPS

    else:
        # ---- rhoa likelihood (legacy / fallback) ---------------------------
        if USE_ANISO and "rho_a_yx" in site:
            kw_sampler["observed_yx"] = site["rho_a_yx"]
            kw_sampler["sigma_yx"] = site.get("sigma_yx", site["sigma"])

    # ---- Run parallel rjMCMC -----------------------------------------------
    results = transdim.run_parallel_rjmcmc(**kw_sampler)

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

    sum_dict = transdim.build_rjmcmc_summary(
        station, results, DEPTH_GRID_MAX, QPAIRS, USE_ANISO)
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
        true_model=None,
        frequencies=site["frequencies"],
        observed=kw_sampler["observed"],
        depth_max=DEPTH_GRID_MAX,
        use_aniso=USE_ANISO,
        save_path=str(fig_path),
    )

    # ---- QC plot -----------------------------------------------------------
    qc_path = outdir / f"{station}_rjmcmc_qc.png"
    qc_z_comps = Z_COMPS if USE_ANISO else ("xy", "yx")

    transdim_viz.plot_qc(
        results,
        frequencies=site["frequencies"],
        observed=kw_sampler["observed"],
        sigma=kw_sampler["sigma"],
        station=station,
        use_aniso=USE_ANISO,
        observed_Z=site.get("Z", None),
        observed_Z_err=site.get("Z_err", None),
        z_comps=qc_z_comps,
        show_pt=has_PT or (USE_ANISO and has_Z),
        observed_PT=site.get("PT", None),
        observed_PT_err=site.get("PT_err", None),
        save_path=str(qc_path),
    )

    # ---- Posterior model plot ----------------------------------------------
    model_path = outdir / f"{station}_rjmcmc_model.png"
    transdim_viz.plot_posterior_model(
        results,
        depth_max=DEPTH_GRID_MAX,
        true_model=None,
        station=station,
        use_aniso=USE_ANISO,
        save_path=str(model_path),
    )

    print()

print("\nDone.\n")
