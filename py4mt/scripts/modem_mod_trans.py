#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a ModEM model file to other formats (UBC, RLM/CGG).

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
rhoair = 1.0e17
blank = rhoair

InFmt = "mod"
OutFmt = ["rlm", "ubc"]

InMod = "/home/vrath/JacoPyAn/work/UBC_format_example/UBI8_Z_Alpha02_NLCG_014"
lat, lon = -16.346, -70.908

# =============================================================================
#  Read and convert
# =============================================================================
if "mod" in InFmt.lower():
    print("\n\nTransforming ModEM model file:")
    print(InMod)
    OutMod = InMod

    start = time.perf_counter()
    dx, dy, dz, rho, refmod, _ = mod.read_mod(
        InMod, ".rho", trans="linear", volumes=True
    )
    dims = np.shape(rho)
    elapsed = time.perf_counter() - start
    print(" Used %7.4f s for reading MOD model from %s " % (elapsed, InMod))

    aircells = np.where(rho > rhoair / 10)

    for fmt in OutFmt:
        if "ubc" in fmt.lower():
            print("\n\nTransforming to UBC model & mesh format")
            start = time.perf_counter()
            elev = -refmod[2]
            refcenter = [lat, lon, elev]
            mod.write_model_ubc(
                OutMod, ".mesh", ".mod",
                dx, dy, dz, rho, refcenter,
                mvalair=blank, aircells=aircells,
            )
            elapsed = time.perf_counter() - start
            print(" Used %7.4f s for Writing UBC model to %s " % (elapsed, OutMod))

        if "rlm" in fmt.lower() or "cgg" in fmt.lower():
            print("\n\nTransforming to RLM/CGG format")
            start = time.perf_counter()
            mod.write_rlm(
                OutMod, modext=".rlm",
                dx=dx, dy=dy, dz=dz, mval=rho,
                reference=refmod, mvalair=blank, aircells=aircells,
                comment="RLM format model",
            )
            elapsed = time.perf_counter() - start
            print(" Used %7.4f s for Writing RLM/CGG model to %s " % (elapsed, OutMod))
else:
    sys.exit(InFmt + " input format not yet implemented! Exit.")
