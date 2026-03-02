#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot slices through a 3D ModEM resistivity model.

Reads a (possibly anisotropic) ModEM model and prepares horizontal/vertical
slices for visualisation.

@author: vrath
"""

import os
import sys
import inspect

import numpy as np
from scipy.interpolate import RegularGridInterpolator

mypath = ["/home/vrath/Py4MT/py4mt/modules/", "/home/vrath/Py4MT/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)

from version import versionstrg
import util as utl
import modem as mod

rhoair = 1.0e17
rng = np.random.default_rng()
nan = np.nan
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())
titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng + "\n\n")

# =============================================================================
#  Configuration
# =============================================================================
Components = 3
UseAniso = True

WorkDir = "/home/vrath/FEMTIC_work/test/"
ModFile = [WorkDir + "/Peru/1_feb_ell/TAC_100"]
PlotFile = WorkDir + "XXX"

# =============================================================================
#  Read model
# =============================================================================
dx, dy, dz, rho, refmod, _ = mod.read_mod_aniso(
    ModFile, components=Components, trans="log10",
)
print(" Read", str(Components), " model components from %s " % (ModFile))
print(np.shape(rho))

aircells = np.where(rho > rhoair / 10)
rho_ref = np.mean(rho, axis=0)
print(np.shape(rho_ref))

# TODO: Get cell centres, create meshgrid, and implement slice plotting
for ii in np.arange(np.shape(rho)[0]):
    pass  # Slice plotting to be implemented
