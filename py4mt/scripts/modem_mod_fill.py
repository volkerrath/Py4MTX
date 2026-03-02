#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fill a ModEM model with a uniform resistivity value, preserving air and sea cells.

Useful for creating prior/starting models from an existing mesh.

@author: vrath
"""

import os
import sys
import inspect
import time

import numpy as np

PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]
PY4MTX_DATA = os.environ["PY4MTX_DATA"]

mypath = [PY4MTX_ROOT + "/py4mt/modules/", PY4MTX_ROOT + "/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)

import modem as mod
from version import versionstrg
import util as utl

version, _ = versionstrg()
titstrng = utl.print_title(
    version=version, fname=inspect.getfile(inspect.currentframe()), out=False
)
print(titstrng + "\n\n")

# =============================================================================
#  Configuration
# =============================================================================
rhosea = 0.3
rhoair = 1.0e17
FillVal = 100.0

InMod = "/home/vrath/MT_Data/Fogo/FOG_best"
OutMod = "/home/vrath/MT_Data/Fogo/FOG_prior" + str(int(FillVal))

# =============================================================================
#  Read and fill model
# =============================================================================
print("\n\nTransforming ModEM model file:")
print(InMod)

start = time.perf_counter()
dx, dy, dz, rho, refmod, _ = mod.read_mod(InMod, ".rho", trans="linear")
dims = np.shape(rho)
elapsed = time.perf_counter() - start
print(" Used %7.4f s for reading MOD model from %s " % (elapsed, InMod))

allcells = np.where(rho)
aircells = np.where(np.isclose(rho, np.ones_like(rho) * rhoair, rtol=1e-04))
seacells = np.where(np.isclose(rho, np.ones_like(rho) * rhosea, rtol=1e-04))

rho_new = rho.copy()
rho_new[allcells] = FillVal
rho_new[seacells] = rhosea
rho_new[aircells] = rhoair

print("\n\nTransformed ModEM model file:")
print(OutMod)

mod.write_mod(
    OutMod, modext=".rho", trans="LOGE",
    dx=dx, dy=dy, dz=dz, mval=rho_new,
    reference=refmod, mvalair=rhoair, aircells=aircells, header="",
)
