#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mt_aniso1d_sampler.py

Script-style PyMC driver for anisotropic 1-D MT inversion.

This is intentionally NOT a CLI. Edit the USER CONFIG and run:

    python mt_aniso1d_sampler.py

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
Created with the help of ChatGPT (GPT-5 Thinking) on 2026-02-10 (UTC)
"""

from __future__ import annotations

import os
import sys
import inspect
import warnings
from pathlib import Path

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
# Example Model0 (keep/edit)
# -----------------------------------------------------------------------------
# Model0 = dict(
#     h_m=np.array([200.0, 400.0, 800.0, 0.0], dtype=float),
#     rop=np.array(
#         [
#             [100.0, 100.0, 100.0],
#             [100.0, 100.0, 100.0],
#             [100.0, 100.0, 100.0],
#             [100.0, 100.0, 100.0],
#         ],
#         dtype=float,
#     ),
#     ustr_deg=np.array([0.0, 0.0, 45.0, 0.0], dtype=float),
#     udip_deg=np.array([0.0, 0.0, 0.0, 0.0], dtype=float),
#     usla_deg=np.array([0.0, 0.0, 0.0, 0.0], dtype=float),
#     is_iso=np.array([True, True, False, True], dtype=bool),
#     is_fix=np.array([False, False, False, False], dtype=bool),
# )

# -----------------------------------------------------------------------------
# Example Model1 (keep/edit)
# -----------------------------------------------------------------------------
nlayer = 17
# Convention: last entry is the basement "thickness" (ignored by the recursion).
# Keep it at 0.0 so that cumulative-depth plots remain well-defined.
h_m = np.r_[np.logspace(np.log10(50.0), np.log10(500.0), nlayer - 1), 0.0]

# --- Simplified model template (preferred) -----------------------------------
#
# You can specify the starting model either in resistivity (rho) or
# conductivity (sigma) form:
#   - rho:   rho_min_ohmm, rho_max_ohmm, strike_deg
#   - sigma: sigma_min_Spm, sigma_max_Spm, strike_deg
#
# The sampler will normalize to the internal canonical form (rho_min/rho_max).

# Example: layered halfspace with constant isotropic background
rho_bg = 100.0  # Ohm·m
sigma_bg = 1.0 / rho_bg  # S/m

Model0 = {
    "prior_name": "model1_hfix",
    "h_m": h_m,
    # Conductivity parameterization (sigma_min <= sigma_max)
    "sigma_min_Spm": sigma_bg * np.ones_like(h_m, dtype=float),
    "sigma_max_Spm": sigma_bg * np.ones_like(h_m, dtype=float),
    "strike_deg": np.zeros_like(h_m, dtype=float),
    "is_iso": np.ones_like(h_m, dtype=bool),
    "is_fix": np.zeros_like(h_m, dtype=bool),
}


# =============================================================================
# USER CONFIG
# =============================================================================

MCMC_Data = "/home/vrath/Py4MTX/py4mt/data/edi/"

INPUT_GLOB = MCMC_Data + "Ann*.edi"  # or *.npz
OUTDIR = MCMC_Data + "pmc_met_hfix"
MODEL_NPZ = MCMC_Data+"model0.npz"

# Set MODEL_DIRECT = Model0 to use the in-file model template
MODEL_DIRECT = Model0
MODEL_DIRECT_SAVE_PATH = MODEL_NPZ
MODEL_DIRECT_OVERWRITE = True

USE_PT = True
PT_ERR_NSIM = 200
Z_COMPS = ("xx", "xy", "yx", "yy")
PT_COMPS = ("xx", "xy", "yx", "yy")
# Phase tensor is always recomputed from Z as P = inv(Re(Z)) @ Im(Z)

PT_REG = 1e-12  # small diagonal regularization used inside PT gradients

FIX_H = True
SAMPLE_LAST_THICKNESS = False

# If True, sample a single global thickness scale H_m (total thickness except basement),
# while keeping the relative thickness profile fixed to the starting model.
# Constraint: requires FIX_H=True.
SAMPLE_H_M = False

LOG10_H_BOUNDS = (0.0, 5.0)        # per-layer thickness bounds (used only if FIX_H=False)
LOG10_H_TOTAL_BOUNDS = (0.0, 5.0)  # total thickness H_m bounds (used only if SAMPLE_H_M=True)
LOG10_RHO_BOUNDS = (-0., 5.0)
STRIKE_BOUNDS_DEG = (-180.0, 180.0)

SIGMA_FLOOR_Z = 0.0
SIGMA_FLOOR_P = 0.0

STEP_METHOD = "demetropolis"  # or "nuts" when ENABLE_GRAD=True
DRAWS = 10000
TUNE = 1000
CHAINS = 8
CORES = CHAINS
TARGET_ACCEPT = 0.85
RANDOM_SEED = 123
PROGRESSBAR = True
ENABLE_GRAD = False  # set True for NUTS/HMC (Z and optional PT)
PRIOR_KIND = "default"  # NUTS-friendly soft priors
PARAM_DOMAIN = "rho"  # "rho" (default) or "sigma"

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

    idata = mcmc.sample_pymc(
        pm_model,
        draws=int(DRAWS),
        tune=int(TUNE),
        chains=int(CHAINS),
        cores=int(CORES),
        step_method=str(STEP_METHOD),
        target_accept=float(TARGET_ACCEPT),
        random_seed=int(RANDOM_SEED) if RANDOM_SEED is not None else None,
        progressbar=bool(PROGRESSBAR),
    )

    nc_path = Path(outdir) / f"{station}_pmc.nc"
    sum_path = Path(outdir) / f"{station}_pmc_summary.npz"
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
