#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deterministic driver for anisotropic 1-D MT inversion.

Script-style workflow: edit the USER CONFIG section and run:

    python mt_aniso1d_inversion.py

Supports Tikhonov and TSVD regularisation with automatic parameter
selection (GCV, L-curve, ABIC). Helpers are imported from inverse.py.

@author:    Volker Rath (DIAS)
@project:   py4mt — Python for Magnetotellurics
@created:   2026-02-13 with the help of ChatGPT (GPT-5 Thinking)
"""

from __future__ import annotations

import os
import sys
import inspect
from pathlib import Path

import numpy as np

# =============================================================================
#  Environment / path setup
# =============================================================================
PY4MTX_DATA = os.environ.get("PY4MTX_DATA", "")
PY4MTX_ROOT = os.environ.get("PY4MTX_ROOT", "")

if not PY4MTX_ROOT:
    sys.exit("PY4MTX_ROOT not set! Exit.")
if not PY4MTX_DATA:
    sys.exit("PY4MTX_DATA not set! Exit.")

for _base in [PY4MTX_ROOT + "/py4mt/modules/"]:
    for _p in [Path(_base), *Path(_base).rglob("*")]:
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

import inverse
import util
from version import versionstrg

version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())

titstrng = util.print_title(version=version, fname=fname, out=False)
print(titstrng + "\n\n")

# =============================================================================
#  Starting model
# =============================================================================
N_LAYER = 17
H_M = np.r_[np.logspace(np.log10(50.0), np.log10(500.0), N_LAYER - 1), 0.0]

# Conductivity domain (S/m). 100 Ohm·m isotropic => sigma = 0.01 S/m.
MODEL0 = {
    "prior_name": "model1_hfix",
    "h_m": H_M,
    "sigma_min": 0.01 * np.ones(N_LAYER, dtype=float),
    "sigma_max": 0.01 * np.ones(N_LAYER, dtype=float),
    "strike_deg": np.zeros(N_LAYER, dtype=float),
    "is_iso": np.zeros(N_LAYER, dtype=bool),
    "is_fix": np.zeros(N_LAYER, dtype=bool),
}

PLOT_RESULTS = True

# =============================================================================
#  USER CONFIG
# =============================================================================

DATA_DIR = "/home/vrath/Py4MTX/py4mt/data/edi/"

INPUT_GLOB = DATA_DIR + "Ann18*.npz"
OUTDIR = DATA_DIR + "detinv_hfix"
MODEL_NPZ = DATA_DIR + "model0.npz"

MODEL_DIRECT = MODEL0
MODEL_DIRECT_SAVE_PATH = MODEL_NPZ
MODEL_DIRECT_OVERWRITE = True

# Data options
USE_PT = True
PT_ERR_NSIM = 200
Z_COMPS = ("xx", "xy", "yx", "yy")
PT_COMPS = ("xx", "xy", "yx", "yy")

# Parameterisation options (match sampler)
PARAM_DOMAIN = "rho"            # "rho" or "sigma"
PARAM_SET = "minmax"            # "minmax" or "max_anifac"

FIX_H = True
SAMPLE_LAST_THICKNESS = False

# Bounds: log10(rho) if PARAM_DOMAIN="rho", log10(sigma) if "sigma"
LOG10_PARAM_BOUNDS = (0.0, 5.0)
LOG10_ANIFAC_BOUNDS = (0.0, 2.0)
LOG10_H_BOUNDS = (0.0, 5.0)
STRIKE_BOUNDS_DEG = (-180.0, 180.0)

SIGMA_FLOOR_Z = 0.0
SIGMA_FLOOR_P = 0.0

# Inversion method
INV_METHOD = "tikhonov"         # "tikhonov" or "tsvd"

# Tikhonov settings
LAMBDA = 1.0
ALPHA_SELECT = "gcv"            # "fixed", "lcurve", "gcv", "abic"
ALPHA_GRID = None
ALPHA_NGRID = 40
ALPHA_MIN = None
ALPHA_MAX = None
L_ORDER = 1
L_INCLUDE_THICKNESS = False

# TSVD settings
TSVD_K = None
TSVD_SELECT = "fixed"           # "fixed", "gcv", "lcurve"
TSVD_K_MIN = 1
TSVD_K_MAX = None
TSVD_RCOND = 1e-3

# Gauss-Newton settings
MAX_ITER = 15
TOL = 1e-3
STEP_SCALE = 1.0

# =============================================================================
#  Run inversion
# =============================================================================
outdir = inverse.ensure_dir(OUTDIR)
in_files = inverse.glob_inputs(INPUT_GLOB)
if not in_files:
    raise FileNotFoundError(f"No inputs matched: {INPUT_GLOB}")

# Starting model
if MODEL_DIRECT is not None:
    model0 = inverse.model_from_direct(MODEL_DIRECT)

    if MODEL_DIRECT_SAVE_PATH:
        p = Path(MODEL_DIRECT_SAVE_PATH).expanduser()
        if p.exists() and (not MODEL_DIRECT_OVERWRITE):
            raise FileExistsError(f"Refusing to overwrite existing model file: {p}")
        inverse.save_model_npz(model0, p.as_posix())
else:
    model0 = inverse.load_model_npz(MODEL_NPZ)

nl = int(np.asarray(model0["h_m"]).size)
spec = inverse.ParamSpec(
    nl=nl,
    fix_h=bool(FIX_H),
    sample_last_thickness=bool(SAMPLE_LAST_THICKNESS),
    log10_h_bounds=LOG10_H_BOUNDS,
    log10_param_bounds=LOG10_PARAM_BOUNDS,
    log10_anifac_bounds=LOG10_ANIFAC_BOUNDS,
    strike_bounds_deg=STRIKE_BOUNDS_DEG,
    param_domain=str(PARAM_DOMAIN),
    param_set=str(PARAM_SET),
)

for f in in_files:
    site = inverse.load_site(f)
    station = str(site.get("station", Path(f).stem))
    print(f"--- {station} ---")

    if USE_PT:
        site = inverse.ensure_phase_tensor(site, nsim=int(PT_ERR_NSIM))

    res = inverse.invert_site(
        site,
        spec=spec,
        model0=model0,
        method=str(INV_METHOD),
        lam=float(LAMBDA),
        lam_select=str(ALPHA_SELECT),
        lam_grid=None if ALPHA_GRID is None else np.asarray(ALPHA_GRID, dtype=float),
        lam_ngrid=int(ALPHA_NGRID),
        lam_min=ALPHA_MIN,
        lam_max=ALPHA_MAX,
        reg_order=int(L_ORDER),
        include_thickness_in_L=bool(L_INCLUDE_THICKNESS),
        tsvd_k=TSVD_K,
        tsvd_select=str(TSVD_SELECT),
        tsvd_k_min=int(TSVD_K_MIN),
        tsvd_k_max=TSVD_K_MAX,
        tsvd_rcond=float(TSVD_RCOND) if TSVD_RCOND is not None else None,
        max_iter=int(MAX_ITER),
        tol=float(TOL),
        step_scale=float(STEP_SCALE),
        use_pt=bool(USE_PT),
        z_comps=Z_COMPS,
        pt_comps=PT_COMPS,
        sigma_floor_Z=float(SIGMA_FLOOR_Z),
        sigma_floor_P=float(SIGMA_FLOOR_P),
    )

    out_path = Path(outdir) / f"{station}_inverse_{INV_METHOD}_{ALPHA_SELECT}.npz"
    inverse.save_inversion_npz(res, out_path.as_posix())
    print(f"Wrote: {out_path}")
    if PLOT_RESULTS:
        print("  (result plotting not yet implemented)")

print("\nDone.\n")
