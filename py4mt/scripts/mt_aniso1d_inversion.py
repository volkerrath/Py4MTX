#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mt_aniso1d_inversion.py

Script-style deterministic driver for anisotropic 1-D MT inversion.

This is intentionally NOT a CLI. Edit the USER CONFIG and run:

    python mt_aniso1d_inversion.py

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
Helpers are imported from `inv1d.py` (TSVD/Tikhonov deterministic inversion).

Author: Volker Rath (DIAS)
Created with the help of ChatGPT (GPT-5 Thinking) on 2026-02-08 (UTC)
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
import inv1d
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
# This driver uses the *simplified* parameterization consistent with the sampler:
# either (rho_min_ohmm, rho_max_ohmm, strike_deg) or (sigma_min_Spm, sigma_max_Spm, strike_deg).
#
# Convention: last entry is the basement "thickness" (ignored by the recursion).
# Keep it at 0.0 so that cumulative-depth plots remain well-defined.
nlayer = 17
h_m = np.r_[np.logspace(np.log10(50.0), np.log10(500.0), nlayer - 1), 0.0]

# Start model in conductivity domain (S/m). For 100 Ohm·m isotropic layers:
# sigma = 1/rho = 0.01 S/m.
Model0 = {
    "prior_name": "model1_hfix",
    "h_m": h_m,
    "sigma_min_Spm": 0.01 * np.ones(nlayer, dtype=float),
    "sigma_max_Spm": 0.01 * np.ones(nlayer, dtype=float),
    "strike_deg": np.zeros(nlayer, dtype=float),
    "is_iso": np.zeros(nlayer, dtype=bool),
    "is_fix": np.zeros(nlayer, dtype=bool),
}


# =============================================================================

# USER CONFIG
# =============================================================================

DATA_DIR = "/home/vrath/Py4MTX/py4mt/data/edi/"

INPUT_GLOB = DATA_DIR + "Ann*.edi"   # or "*.npz"
OUTDIR = DATA_DIR + "detinv_hfix"
MODEL_NPZ = DATA_DIR + "model0.npz"

# Set MODEL_DIRECT = Model0 to use the in-file model template
MODEL_DIRECT = Model0
MODEL_DIRECT_SAVE_PATH = MODEL_NPZ
MODEL_DIRECT_OVERWRITE = True

# Data options
USE_PT = True
PT_ERR_NSIM = 200
Z_COMPS = ("xx", "xy", "yx", "yy")
PT_COMPS = ("xx", "xy", "yx", "yy")

# Parameterization options (match sampler)
PARAM_DOMAIN = "rho"           # "rho" or "sigma"
PARAM_SET = "minmax"           # "minmax" or "max_anifac"

FIX_H = True
SAMPLE_LAST_THICKNESS = False

# Bounds apply to log10(rho) if PARAM_DOMAIN="rho", and to log10(sigma) if "sigma".
LOG10_PARAM_BOUNDS = (0.0, 5.0)
LOG10_ANIFAC_BOUNDS = (0.0, 2.0)   # only used for PARAM_SET="max_anifac"

# Backward-compatible name (kept)
LOG10_RHO_BOUNDS = LOG10_PARAM_BOUNDS

LOG10_H_BOUNDS = (0.0, 5.0)
STRIKE_BOUNDS_DEG = (-180.0, 180.0)

SIGMA_FLOOR_Z = 0.0
SIGMA_FLOOR_P = 0.0

# Deterministic inversion options
INV_METHOD = "tikhonov"   # "tikhonov" or "tsvd"

# Tikhonov settings
LAMBDA = 1.0
ALPHA_SELECT = "fixed"    # "fixed", "lcurve", "gcv", "abic"
ALPHA_GRID = None         # optional 1D array/list of lambda values
ALPHA_NGRID = 40          # used if ALPHA_GRID is None
ALPHA_MIN = None          # optional lower bound for auto grid
ALPHA_MAX = None          # optional upper bound for auto grid
L_ORDER = 1               # 1 (first-order) or 2 (second-order)
L_INCLUDE_THICKNESS = False

# TSVD settings
TSVD_K = None             # fixed truncation rank (int) or None
TSVD_SELECT = "fixed"     # "fixed", "gcv", "lcurve" (used if TSVD_K is None or TSVD_SELECT != "fixed")
TSVD_K_MIN = 1
TSVD_K_MAX = None
TSVD_RCOND = 1e-3         # used only for the "fixed" TSVD path when TSVD_K is provided

# Gauss-Newton settings
MAX_ITER = 15
TOL = 1e-3
STEP_SCALE = 1.0


outdir = inv1d.ensure_dir(OUTDIR)
in_files = inv1d.glob_inputs(INPUT_GLOB)
if not in_files:
    raise FileNotFoundError(f"No inputs matched: {INPUT_GLOB}")

# starting model
if MODEL_DIRECT is not None:
    model0 = inv1d.model_from_direct(MODEL_DIRECT)

    if MODEL_DIRECT_SAVE_PATH:
        p = Path(MODEL_DIRECT_SAVE_PATH).expanduser()
        if p.exists() and (not MODEL_DIRECT_OVERWRITE):
            raise FileExistsError(f"Refusing to overwrite existing model file: {p}")
        inv1d.save_model_npz(model0, p.as_posix())
else:
    model0 = inv1d.load_model_npz(MODEL_NPZ)

nl = int(np.asarray(model0["h_m"]).size)
spec = inv1d.ParamSpec(
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
    site = inv1d.load_site(f)
    station = str(site.get("station", Path(f).stem))
    print(f"--- {station} ---")

    if USE_PT:
        # Add P and (bootstrap) P_err if missing
        site = inv1d.ensure_phase_tensor(site, nsim=int(PT_ERR_NSIM))

    res = inv1d.invert_site(
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

    out_path = Path(outdir) / f"{station}_inv1d_{INV_METHOD}.npz"
    inv1d.save_inversion_npz(res, out_path.as_posix())
    print(f"Wrote: {out_path}")

print("\nDone.\n")
