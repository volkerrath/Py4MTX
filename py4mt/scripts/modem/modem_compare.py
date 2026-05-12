#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare two ModEM resistivity models.

Reads two ModEM .rho files, computes the log10-resistivity difference
and cross-gradient, and writes the results to new .rho files.

Author: vrath
Created: Fri Sep 11 15:41:25 2020
Provenance: cleaned/debugged with Claude (Anthropic), Mar 2026
"""

import os
import sys
from pathlib import Path
import inspect
import warnings
from datetime import datetime

import numpy as np

# ---------------------------------------------------------------------------
# Path configuration – adjust to your environment
# ---------------------------------------------------------------------------
PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

for _base in [PY4MTX_ROOT + "/py4mt/modules/"]:
    for _p in [Path(_base), *Path(_base).rglob("*")]:
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

import modem as mod
import util as utl
from version import versionstrg

version, _ = versionstrg()
title_str = utl.print_title(
    version=version,
    fname=inspect.getfile(inspect.currentframe()),
    out=False,
)
print(title_str + "\n\n")

rng = np.random.default_rng()
nan = np.nan

warnings.simplefilter(action="ignore", category=FutureWarning)

# ===========================================================================
# User parameters
# ===========================================================================
MODEL_IN1 = r"/home/vrath/work/MaurienneJCN/MaurPrior"
MODEL_IN2 = r"/home/vrath/work/MaurienneJCN/MaurPrior"

MODEL_OUT_DIFF = "CompareMod_Diff"
MODEL_OUT_CRG  = "CompareMod_CrossGrad"

# ---------------------------------------------------------------------------
# Read models
# ---------------------------------------------------------------------------
dx, dy, dz, rho1, reference, _ = mod.read_mod(MODEL_IN1, out=True)
_,  _,  _,  rho2, _,         _ = mod.read_mod(MODEL_IN2, out=True)

# ---------------------------------------------------------------------------
# Compute log10-resistivity difference
# ---------------------------------------------------------------------------
log_rho1 = np.log10(rho1)
log_rho2 = np.log10(rho2)
rho_diff = log_rho2 - log_rho1

# ---------------------------------------------------------------------------
# Write difference model
# ---------------------------------------------------------------------------
mod.write_mod(
    MODEL_OUT_DIFF,
    dx=dx, dy=dy, dz=dz,
    mval=rho_diff,
    reference=reference,
    trans="LINEAR",
    out=True,
)

# ---------------------------------------------------------------------------
# Gradients and cross-gradient
# ---------------------------------------------------------------------------
g1x, g1y, g1z = np.gradient(log_rho1)
ng1 = np.sqrt(g1x**2 + g1y**2 + g1z**2)

g2x, g2y, g2z = np.gradient(log_rho2)
ng2 = np.sqrt(g2x**2 + g2y**2 + g2z**2)

rho_crg, rho_crg_norm = mod.crossgrad(
    m1=rho1, m2=rho2, mesh=[dx, dy, dz],
)

mod.write_mod(
    MODEL_OUT_CRG,
    dx=dx, dy=dy, dz=dz,
    mval=rho_crg,
    reference=reference,
    trans="LINEAR",
    out=True,
)
