#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script-style PyMC driver for anisotropic 1-D MT inversion.

This is intentionally NOT a CLI. Edit the USER CONFIG and run:

    python mt_aniso1d_sampler.py

Preserved conventions: PY4MTX environment variables, explicit sys.path
setup, startup title print, example in-file model (MODEL0).
Helpers are imported from mcmc.py.

@author:    Volker Rath (DIAS)
@project:   py4mt — Python for Magnetotellurics
@created:   2026-02-12 with the help of ChatGPT (GPT-5 Thinking)
@modified:  2026-03-01 — Gaussian prior option, Claude (Opus 4.6, Anthropic)
@modified:  2026-03-01 — Matérn covariance kernels, Claude (Opus 4.6, Anthropic)
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

import data_proc  # noqa: F401
import mcmc
import util
from version import versionstrg

version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())

titstrng = util.print_title(version=version, fname=fname, out=False)
print(titstrng + "\n\n")

# =============================================================================
#  Example starting model (edit as needed)
# =============================================================================
N_LAYER = 8
H_M = np.r_[np.logspace(np.log10(500.0), np.log10(3000.0), N_LAYER - 1), 0.0]

RHO_BG = 300.0       # Ohm·m
SIGMA_BG = 1.0 / RHO_BG  # S/m

MODEL0 = {
    "prior_name": "model1",
    "h_m": H_M,
    "sigma_min": SIGMA_BG * np.ones_like(H_M, dtype=float),
    "sigma_max": SIGMA_BG * np.ones_like(H_M, dtype=float),
    "strike_deg": np.zeros_like(H_M, dtype=float),
    "is_iso": np.zeros_like(H_M, dtype=bool),
    "is_fix": np.zeros_like(H_M, dtype=bool),
}

# =============================================================================
#  USER CONFIG
# =============================================================================

MCMC_DATA = "/home/vrath/Py4MTX/py4mt/data/edi/ann/mcmc/"

INPUT_GLOB = MCMC_DATA + "*.npz"

MODEL_NPZ = MCMC_DATA + "model0.npz"

# Set MODEL_DIRECT = MODEL0 to use the in-file model template
MODEL_DIRECT = MODEL0
MODEL_DIRECT_SAVE_PATH = MODEL_NPZ
MODEL_DIRECT_OVERWRITE = True

USE_PT = True
PT_ERR_NSIM = 200
Z_COMPS = ("xx", "xy", "yx", "yy")
# Z_COMPS = ("xy", "yx")
PT_COMPS = ("xx", "xy", "yx", "yy")

PT_REG = 1e-8  # small diagonal regularisation used inside PT gradients

FIX_H = False
SAMPLE_LAST_THICKNESS = False

# If True, sample a single global thickness scale H_m (total thickness except
# basement), while keeping the relative thickness profile fixed to the starting
# model.  Constraint: requires FIX_H=True.
SAMPLE_H_M = False

LOG10_H_BOUNDS = (0.0, 3.0)          # per-layer thickness bounds (FIX_H=False)
LOG10_H_TOTAL_BOUNDS = (0.0, 5.0)    # total thickness H_m bounds (SAMPLE_H_M=True)
LOG10_RHO_BOUNDS = (-0., 5.0)
STRIKE_BOUNDS_DEG = (-180.0, 180.0)

SIGMA_FLOOR_Z = 0.0
SIGMA_FLOOR_P = 0.0

STEP_METHOD = "demetropolis"  # or "nuts" when ENABLE_GRAD=True
ENABLE_GRAD = False           # set True for NUTS/HMC (Z and optional PT)
PRIOR_KIND = "default"        # "default", "uniform", or "gaussian"
PARAM_DOMAIN = "rho"          # "rho" or "sigma"

# ---- Gaussian prior (used only when PRIOR_KIND = "gaussian") ----------------
PRIOR_STD = {
    "log10_rho": 1.0,
    "strike_deg": 45.0,
    "log10_h": 0.5,
    "log10_H": 0.5,
}

# Full covariance matrix.  Set to a (3*N_LAYER, 3*N_LAYER) array to enable
# correlated Gaussian priors.  When set, overrides rho/strike entries in
# PRIOR_STD.  Build with mcmc.build_gaussian_cov().
#
# Correlation models: "exponential" (Matérn ν=½), "matern32" (ν=3/2),
# "matern52" (ν=5/2), "matern" (general, ν via nu kwarg), "gaussian" (ν→∞),
# None/"identity" (independent layers).
#
# See README for examples.
PRIOR_COV = None

# One shared quantile setting
QPAIRS = ((10, 90), (25, 75))

OUTDIR = MCMC_DATA + "pmc_" + STEP_METHOD + "_h_zp"

PROGRESSBAR = False

# =============================================================================
#  pmc_dict presets (copy desired preset into pmc_dict below)
# =============================================================================

# (a) 10-layer case, hfixed (FIX_H=True)
PMC_DICT_NUTS_10LAYER_HFIXED = {
    "step_method": "nuts",
    "draws": 4000,
    "tune": 4000,
    "chains": 4,
    "cores": 4,
    "target_accept": 0.90,
    "init": "auto",
    "random_seed": mcmc.generate_mcmc_seed(),
    "progressbar": True,
    "discard_tuned_samples": True,
    "compute_convergence_checks": True,
}

PMC_DICT_METROPOLIS_10LAYER_HFIXED = {
    "step_method": "metropolis",
    "draws": 120_000,
    "tune": 30_000,
    "chains": 4,
    "cores": 4,
    "random_seed": mcmc.generate_mcmc_seed(),
    "progressbar": True,
    "discard_tuned_samples": True,
    "compute_convergence_checks": True,
}

PMC_DICT_DEMETROPOLISZ_10LAYER_HFIXED = {
    "step_method": "demetropolisz",
    "draws": 60_000,
    "tune": 20_000,
    "chains": 8,
    "cores": 8,
    "random_seed": mcmc.generate_mcmc_seed(),
    "progressbar": True,
    "discard_tuned_samples": True,
    "compute_convergence_checks": True,
}

# (b) 6-layer case, sampling log(h_m) (FIX_H=False)
PMC_DICT_NUTS_6LAYER_LOGHM = {
    "step_method": "nuts",
    "draws": 6000,
    "tune": 6000,
    "chains": 4,
    "cores": 4,
    "target_accept": 0.93,
    "init": "auto",
    "random_seed": mcmc.generate_mcmc_seed(),
    "progressbar": True,
    "discard_tuned_samples": True,
    "compute_convergence_checks": True,
}

PMC_DICT_METROPOLIS_6LAYER_LOGHM = {
    "step_method": "metropolis",
    "draws": 200_000,
    "tune": 80_000,
    "chains": 4,
    "cores": 4,
    "random_seed": mcmc.generate_mcmc_seed(),
    "progressbar": True,
    "discard_tuned_samples": True,
    "compute_convergence_checks": True,
}

PMC_DICT_DEMETROPOLISZ_6LAYER_LOGHM = {
    "step_method": "demetropolisz",
    "draws": 100_000,
    "tune": 30_000,
    "chains": 10,
    "cores": 10,
    "random_seed": mcmc.generate_mcmc_seed(),
    "progressbar": True,
    "discard_tuned_samples": True,
    "compute_convergence_checks": True,
}

# Active preset
pmc_dict = PMC_DICT_DEMETROPOLISZ_10LAYER_HFIXED

# =============================================================================
#  Run sampler
# =============================================================================
outdir = mcmc.ensure_dir(OUTDIR)
in_files = mcmc.glob_inputs(INPUT_GLOB)
if not in_files:
    raise FileNotFoundError(f"No inputs matched: {INPUT_GLOB}")

# Starting model
if MODEL_DIRECT is not None:
    model0 = mcmc.model_from_direct(MODEL_DIRECT)
    if MODEL_DIRECT_OVERWRITE or (not Path(MODEL_DIRECT_SAVE_PATH).exists()):
        mcmc.save_model_npz(model0, MODEL_DIRECT_SAVE_PATH)
else:
    model0 = mcmc.load_model_npz(MODEL_NPZ)

model0 = mcmc.normalize_model(model0)

nl = int(np.asarray(model0["h_m"]).size)

# Sanity: global H_m sampling requires fixed relative thickness profile
if SAMPLE_H_M and (not FIX_H):
    raise ValueError(
        "SAMPLE_H_M=True requires FIX_H=True "
        "(keep relative layer thicknesses fixed)."
    )

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
        prior_std=PRIOR_STD if PRIOR_KIND.lower() == "gaussian" else None,
        prior_cov=PRIOR_COV if PRIOR_KIND.lower() == "gaussian" else None,
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
