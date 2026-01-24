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
Created with the help of ChatGPT (GPT-5 Thinking) on 2026-01-23
"""

from __future__ import annotations

import os
import sys
import inspect
from pathlib import Path

import numpy as np

# -----------------------------------------------------------------------------
# Environment variables / path settings (keep)
# -----------------------------------------------------------------------------
PY4MTX_DATA = os.environ.get("PY4MTX_DATA", "")
PY4MTX_ROOT = os.environ.get("PY4MTX_ROOT", "")

if not PY4MTX_ROOT:
    PY4MTX_ROOT = str(Path(__file__).resolve().parent.parent)
if not PY4MTX_DATA:
    PY4MTX_DATA = str(Path(PY4MTX_ROOT) / "data")

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
Model0 = dict(
    h_m=np.array([200.0, 400.0, 800.0, 0.0], dtype=float),
    rop=np.array(
        [
            [100.0, 100.0, 100.0],
            [300.0, 300.0, 300.0],
            [30.0, 300.0, 3000.0],
            [1000.0, 1000.0, 1000.0],
        ],
        dtype=float,
    ),
    ustr_deg=np.array([0.0, 0.0, 45.0, 0.0], dtype=float),
    udip_deg=np.array([0.0, 0.0, 0.0, 0.0], dtype=float),
    usla_deg=np.array([0.0, 0.0, 0.0, 0.0], dtype=float),
    is_iso=np.array([True, True, False, True], dtype=bool),
    is_fix=np.array([False, False, False, False], dtype=bool),
)

# =============================================================================
# USER CONFIG
# =============================================================================

INPUT_GLOB = str(Path(PY4MTX_DATA) / "*.edi")   # or *.npz
OUTDIR = str(Path(PY4MTX_DATA) / "pmc_out")



MODEL_NPZ = str(Path(PY4MTX_DATA) / "model0.npz")

# Set MODEL_DIRECT = Model0 to use the in-file model template
MODEL_DIRECT = None
MODEL_DIRECT_SAVE_PATH = MODEL_NPZ
MODEL_DIRECT_OVERWRITE = True

USE_PT = True
PT_ERR_NSIM = 200
Z_COMPS = ("xy", "yx")
PT_COMPS = ("xx", "xy", "yx", "yy")
COMPUTE_PT_IF_MISSING = True

FIX_H = True
SAMPLE_LAST_THICKNESS = False

LOG10_H_BOUNDS = (0.0, 5.0)
LOG10_RHO_BOUNDS = (-1.0, 6.0)
USTR_BOUNDS_DEG = (-180.0, 180.0)
UDIP_BOUNDS_DEG = (0.0, 90.0)
USLA_BOUNDS_DEG = (-180.0, 180.0)

SIGMA_FLOOR_Z = 0.0
SIGMA_FLOOR_P = 0.0

STEP_METHOD = "demetropolis"
DRAWS = 2000
TUNE = 1000
CHAINS = 10
CORES = CHAINS
TARGET_ACCEPT = 0.85
RANDOM_SEED = 123
PROGRESSBAR = True
ENABLE_GRAD = False
PRIOR_KIND = "uniform"

# One shared quantile setting (requested)
QPAIRS = ((0.1, 0.9), (0.25, 0.75))


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

nl = int(np.asarray(model0["rop"]).shape[0])
if "is_fix" not in model0:
    model0["is_fix"] = np.zeros(nl, dtype=bool)
spec = mcmc.ParamSpec(
    nl=nl,
    fix_h=bool(FIX_H),
    sample_last_thickness=bool(SAMPLE_LAST_THICKNESS),
    log10_h_bounds=LOG10_H_BOUNDS,
    log10_rho_bounds=LOG10_RHO_BOUNDS,
    ustr_bounds_deg=USTR_BOUNDS_DEG,
    udip_bounds_deg=UDIP_BOUNDS_DEG,
    usla_bounds_deg=USLA_BOUNDS_DEG,
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
        h_m0=np.asarray(model0.get("h_m", None)) if "h_m" in model0 else None,
        rop0=np.asarray(model0.get("rop", None)),
        ustr_deg0=np.asarray(model0.get("ustr_deg", None)) if "ustr_deg" in model0 else None,
        udip_deg0=np.asarray(model0.get("udip_deg", None)) if "udip_deg" in model0 else None,
        usla_deg0=np.asarray(model0.get("usla_deg", None)) if "usla_deg" in model0 else None,
        is_iso=np.asarray(model0.get("is_iso", None)) if "is_iso" in model0 else None,
        is_fix=np.asarray(model0.get("is_fix", None)) if "is_fix" in model0 else None,
        use_pt=bool(USE_PT),
        z_comps=Z_COMPS,
        pt_comps=PT_COMPS,
        compute_pt_if_missing=bool(COMPUTE_PT_IF_MISSING),
        sigma_floor_Z=float(SIGMA_FLOOR_Z),
        sigma_floor_P=float(SIGMA_FLOOR_P),
        enable_grad=bool(ENABLE_GRAD),
        prior_kind=str(PRIOR_KIND),
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
