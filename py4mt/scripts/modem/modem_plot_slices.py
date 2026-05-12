#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot slices through a 3D ModEM resistivity model.

Reads a (possibly anisotropic) ModEM model and prepares horizontal/vertical
slices for visualisation.

Status: work-in-progress stub — model loading works; slice plotting is
not yet implemented.

@author: vrath
Cleanup: 4 Mar 2026 by Claude (Anthropic)
"""

import os
import sys
from pathlib import Path
import inspect

import numpy as np
from scipy.interpolate import RegularGridInterpolator

PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

for _base in [PY4MTX_ROOT + "/py4mt/modules/"]:
    for _p in [Path(_base), *Path(_base).rglob("*")]:
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

from version import versionstrg
import util as utl
import modem as mod

version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())
titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng + "\n\n")

# =============================================================================
#  Configuration
# =============================================================================
RHOAIR = 1.0e17
COMPONENTS = 3
USE_ANISO = True

WORK_DIR = "/home/vrath/FEMTIC_work/test/"
MOD_FILE = [WORK_DIR + "/Peru/1_feb_ell/TAC_100"]
PLOT_FILE = WORK_DIR + "XXX"

# =============================================================================
#  Read model
# =============================================================================
dx, dy, dz, rho, refmod, _ = mod.read_mod_aniso(
    MOD_FILE, components=COMPONENTS, trans="log10",
)
print(" Read", str(COMPONENTS), " model components from %s " % (MOD_FILE))
print(np.shape(rho))

aircells = np.where(rho > RHOAIR / 10)
rho_ref = np.mean(rho, axis=0)
print(np.shape(rho_ref))

# TODO: Get cell centres, create meshgrid, and implement slice plotting
for ii in np.arange(np.shape(rho)[0]):
    pass  # Slice plotting to be implemented
