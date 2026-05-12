#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate depth-dependent smoothing alpha parameters for ModEM inversion.

Reads a ModEM model to obtain the mesh, then computes alpha_x and alpha_y
smoothing parameters that vary linearly with depth between user-specified
bounds.

@author: vrath (Nov 2024)
Cleanup: 4 Mar 2026 by Claude (Anthropic)
"""

import os
import sys
from pathlib import Path
import inspect

import numpy as np

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
titstrng = utl.print_title(
    version=version, fname=inspect.getfile(inspect.currentframe()), out=False
)
print(titstrng + "\n\n")

# =============================================================================
#  Configuration
# =============================================================================
MOD_DIR_IN = PY4MTX_DATA + "/Peru/Misti/"
MOD_DIR_OUT = MOD_DIR_IN

MOD_FILE_IN = MOD_DIR_IN + "Mis_50"
MOD_FILE_OUT = MOD_FILE_IN

if not os.path.isdir(MOD_DIR_OUT):
    print("Directory: %s does not exist, but will be created" % MOD_DIR_OUT)
    os.mkdir(MOD_DIR_OUT)

BEG_LIN = np.array([100.0, 0.1, 0.1])
END_LIN = np.array([25000.0, 0.6, 0.6])

# =============================================================================
#  Read model and generate alphas
# =============================================================================
dx, dy, dz, base_model, refmod, _ = mod.read_mod(MOD_FILE_IN, ".rho", trans="log10")

_, depth = mod.set_mesh(d=dz)

ax, ay = mod.generate_alphas(dz, beg_lin=BEG_LIN, end_lin=END_LIN)

print(ax)
