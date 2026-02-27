#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mt_aniso1d_sampler_strict.py

Script-style PyMC driver for anisotropic 1-D MT inversion.

This is intentionally NOT a CLI. Edit the USER CONFIG and run:

    python mt_aniso1d_sampler_strict.py

Preserved conventions
---------------------
- Environment variables:
    - PY4MTX_ROOT
    - PY4MTX_DATA
- Explicit sys.path setup
- Startup title print
- Example in-file model (Model0)

Implementation note
-------------------
Helpers are imported from `mcmc.py`

Author: Volker Rath (DIAS)
Created with the help of ChatGPT (GPT-5 Thinking) on 2026-02-12 (UTC)
"""

from __future__ import annotations

import os
import sys
import inspect
import warnings
from pathlib import Path
import time

import numpy as np

# Optional: suppress noisy FutureWarnings (e.g., from PyMC/PyTensor)
SUPPRESS_FUTUREWARNINGS = True
if SUPPRESS_FUTUREWARNINGS:
    warnings.filterwarnings("ignore", category=FutureWarning)


# -----------------------------------------------------------------------------
# Environment variables / path settings (keep)
# -----------------------------------------------------------------------------
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

# local modules
import data_proc  # noqa: F401
import mcmc
import util
from version import versionstrg

version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())

titstrng = util.print_title(version=version, fname=fname, out=False)
print(titstrng+'\n\n')


# -----------------------------------------------------------------------------
# Example Model1 (keep/edit)
# -----------------------------------------------------------------------------
nlayer = 8
# Convention: last entry is the basement "thickness" (ignored by the recursion).
# Keep it at 0.0 so that cumulative-depth plots remain well-defined.
h_m = np.r_[np.logspace(np.log10(500.0), np.log10(3000.0), nlayer - 1), 0.0]

# --- Simplified model template (preferred) -----------------------------------
#
# You can specify the starting model either in resistivity (rho) or
# conductivity (sigma) form:
#   - rho:   rho_min, rho_max, strike_deg
#   - sigma: sigma_min, sigma_max, strike_deg
#
# The sampler normalizes to a canonical representation containing both rho and sigma fields.

# Example: layered halfspace with constant isotropic background
rho_bg = 300.0  # Ohm·m
sigma_bg = 1.0 / rho_bg  # S/m

Model0 = {
    "prior_name": "model1",
    "h_m": h_m,
    # Conductivity parameterization (sigma_min <= sigma_max)
    "sigma_min": sigma_bg * np.ones_like(h_m, dtype=float),
    "sigma_max": sigma_bg * np.ones_like(h_m, dtype=float),
    "strike_deg": np.zeros_like(h_m, dtype=float),
    "is_iso": np.zeros_like(h_m, dtype=bool),
    "is_fix": np.zeros_like(h_m, dtype=bool),
}


# =============================================================================
# USER CONFIG
# =============================================================================

# MCMC_Data = "/home/vrath/Py4MTX/py4mt/data/edi/"
MCMC_Data =  "/home/vrath/Py4MTX/py4mt/data/edi/ann/mcmc/"

INPUT_GLOB = MCMC_Data + "*.npz"  # or *.npz

MODEL_NPZ = MCMC_Data+"model0.npz"

# Set MODEL_DIRECT = Model0 to use the in-file model template
MODEL_DIRECT = Model0
MODEL_DIRECT_SAVE_PATH = MODEL_NPZ
MODEL_DIRECT_OVERWRITE = True

USE_PT = True
PT_ERR_NSIM = 200
Z_COMPS = ("xx", "xy", "yx", "yy")
# Z_COMPS = ("xy", "yx")
PT_COMPS = ("xx", "xy", "yx", "yy")
# Phase tensor is always recomputed from Z as P = inv(Re(Z)) @ Im(Z)

PT_REG = 1e-8  # small diagonal regularization used inside PT gradients

FIX_H = False
SAMPLE_LAST_THICKNESS = False

# If True, sample a single global thickness scale H_m (total thickness except basement),
# while keeping the relative thickness profile fixed to the starting model.
# Constraint: requires FIX_H=True.
SAMPLE_H_M = False

LOG10_H_BOUNDS = (0.0, 3.0)        # per-layer thickness bounds (used only if FIX_H=False)
LOG10_H_TOTAL_BOUNDS = (0.0, 5.0)  # total thickness H_m bounds (used only if SAMPLE_H_M=True)
LOG10_RHO_BOUNDS = (-0., 5.0)
STRIKE_BOUNDS_DEG = (-180.0, 180.0)

SIGMA_FLOOR_Z = 0.0
SIGMA_FLOOR_P = 0.0

# STEP_METHOD = "nuts"  # or "nuts" when ENABLE_GRAD=True
# ENABLE_GRAD = True  # set True for NUTS/HMC (Z and optional PT)
STEP_METHOD = "demetropolis"  # or "nuts" when ENABLE_GRAD=True
ENABLE_GRAD = False  # set True for NUTS/HMC (Z and optional PT)
PRIOR_KIND = "default"  # NUTS-friendly soft priors
PARAM_DOMAIN = "rho"  # "rho" (default) or "sigma"

OUTDIR = MCMC_Data + "pmc_"+STEP_METHOD+"_h_zp"

DRAWS = 100000
TUNE = 10000
CHAINS = 10
CORES = CHAINS
TARGET_ACCEPT = 0.85
RANDOM_SEED = mcmc.generate_mcmc_seed() #int(time.time_ns()) #123
PROGRESSBAR = True

# -----------------------------------------------------------------------------
# Example pmc_dict presets (copy/paste into `pmc_dict` as needed)
# -----------------------------------------------------------------------------
# These presets are consistent with mcmc.sample_pymc() in this patched zip.
# They use only keys that are either consumed by the wrapper
#   - step_method, target_accept
# or forwarded to pm.sample(**kwargs)
#   - draws, tune, chains, cores, init, random_seed, progressbar, ...
#
# (a) 10-layer case, hfixed (FIX_H=True)
pmc_dict_nuts_10layer_hfixed = {
    "step_method": "nuts",
    "draws": 4000,
    "tune": 4000,
    "chains": 4,
    "cores": 4,
    "target_accept": 0.90,
    "init": "auto",  # (PyMC) initialization + mass-matrix adaptation strategy
    "random_seed": mcmc.generate_mcmc_seed(), #int(time.time_ns()) #123None,
    "progressbar": True,
    "discard_tuned_samples": True,
    "compute_convergence_checks": True,
}

pmc_dict_metropolis_10layer_hfixed = {
    "step_method": "metropolis",
    "draws": 120_000,
    "tune": 30_000,
    "chains": 4,
    "cores": 4,
    "random_seed": mcmc.generate_mcmc_seed(), #int(time.time_ns()) #123None,
    "progressbar": True,
    "discard_tuned_samples": True,
    "compute_convergence_checks": True,
}

pmc_dict_demetropolisz_10layer_hfixed = {
    "step_method": "demetropolisz",
    "draws": 60_000,
    "tune": 20_000,
    "chains": 8,   # DE-style methods typically benefit from more chains
    "cores": 8,
    "random_seed": mcmc.generate_mcmc_seed(), #int(time.time_ns()) #123None,
    "progressbar": True,
    "discard_tuned_samples": True,
    "compute_convergence_checks": True,
}

# (b) 6-layer case, sampling log(h_m) (FIX_H=False)
pmc_dict_nuts_6layer_loghm = {
    "step_method": "nuts",
    "draws": 6000,
    "tune": 6000,
    "chains": 4,
    "cores": 4,
    "target_accept": 0.93,
    "init": "auto",
    "random_seed": mcmc.generate_mcmc_seed(), #int(time.time_ns()) #123None,
    "progressbar": True,
    "discard_tuned_samples": True,
    "compute_convergence_checks": True,
}

pmc_dict_metropolis_6layer_loghm = {
    "step_method": "metropolis",
    "draws": 200_000,
    "tune": 80_000,
    "chains": 4,
    "cores": 4,
    "random_seed": mcmc.generate_mcmc_seed(), #int(time.time_ns()) #123None,
    "progressbar": True,
    "discard_tuned_samples": True,
    "compute_convergence_checks": True,
}

pmc_dict_demetropolisz_6layer_loghm = {
    "step_method": "demetropolisz",
    "draws": 100_000,
    "tune": 30_000,
    "chains": 10,
    "cores": 10,
    "random_seed": mcmc.generate_mcmc_seed(), #int(time.time_ns()) #123None,
    "progressbar": True,
    "discard_tuned_samples": True,
    "compute_convergence_checks": True,
}

# Collect all PyMC sampling controls in one dict (forwarded via **pmc_dict)
pmc_dict = {
    "draws": int(DRAWS),
    "tune": int(TUNE),
    "chains": int(CHAINS),
    "cores": int(CORES),
    "step_method": str(STEP_METHOD),
    "target_accept": float(TARGET_ACCEPT),
    "random_seed": int(RANDOM_SEED) if RANDOM_SEED is not None else None,
    "progressbar": bool(PROGRESSBAR),
}

# One shared quantile setting (requested)
QPAIRS = ((10, 90), (25, 75))
# QPAIRS accepts either quantiles in [0,1] or percentiles in [0,100].
outdir = mcmc.ensure_dir(OUTDIR)
in_files = mcmc.glob_inputs(INPUT_GLOB)
if not in_files:
    raise FileNotFoundError(f"No inputs matched: {INPUT_GLOB}")

# starting model
if MODEL_DIRECT is not None:
    model0 = mcmc.model_from_direct(MODEL_DIRECT)
    if MODEL_DIRECT_OVERWRITE or (not Path(MODEL_DIRECT_SAVE_PATH).exists()):
        mcmc.save_model_npz(model0, MODEL_DIRECT_SAVE_PATH)
else:
    model0 = mcmc.load_model_npz(MODEL_NPZ)

# Ensure all arrays are consistent: rop orientation and flag lengths.
model0 = mcmc.normalize_model(model0)

nl = int(np.asarray(model0["h_m"]).size)
# Sanity: global H_m sampling requires fixed relative thickness profile
if SAMPLE_H_M and (not FIX_H):
    raise ValueError("SAMPLE_H_M=True requires FIX_H=True (keep relative layer thicknesses fixed).")

spec = mcmc.ParamSpec(
    nl=nl,
    fix_h=bool(FIX_H),
    sample_H_m=bool(SAMPLE_H_M),
    sample_last_thickness=bool(SAMPLE_LAST_THICKNESS),
    log10_h_bounds=LOG10_H_BOUNDS,
    log10_H_bounds=LOG10_H_TOTAL_BOUNDS,
    log10_rho_bounds=LOG10_RHO_BOUNDS,
    strike_bounds_deg=STRIKE_BOUNDS_DEG,
)

for f in in_files:
    site = mcmc.load_site(f)
    station = str(site.get("station", Path(f).stem))
    print(f"--- {station} ---")

    if USE_PT:
        site = mcmc.ensure_phase_tensor(site, nsim=int(PT_ERR_NSIM))

    pm_model, info = mcmc.build_pymc_model(
        site,
        spec=spec,
        model0=model0,
        use_pt=bool(USE_PT),
        z_comps=Z_COMPS,
        pt_comps=PT_COMPS,
        pt_reg=float(PT_REG),
        sigma_floor_Z=float(SIGMA_FLOOR_Z),
        sigma_floor_P=float(SIGMA_FLOOR_P),
        enable_grad=bool(ENABLE_GRAD),
        prior_kind=str(PRIOR_KIND),
        param_domain=str(PARAM_DOMAIN),
    )

    idata = mcmc.sample_pymc(pm_model, pmc_dict=pmc_dict)
    nc_path = Path(outdir) / f"{station}_pmc_{STEP_METHOD}.nc"
    sum_path = Path(outdir) / f"{station}_pmc_{STEP_METHOD}_summary.npz"
    mcmc.save_idata(idata, nc_path)

    summary = mcmc.build_summary_npz(
        station=station,
        site=site,
        idata=idata,
        spec=spec,
        model0=model0,
        info=info,
        qpairs=QPAIRS,
    )
    mcmc.save_summary_npz(summary, sum_path)

    print(f"Wrote: {nc_path}")
    print(f"Wrote: {sum_path}")

print("\nDone.\n")
