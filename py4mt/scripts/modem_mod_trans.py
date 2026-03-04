#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a ModEM model file to other formats (UBC, RLM/CGG).

@author: vrath
Cleanup: 4 Mar 2026 by Claude (Anthropic)
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
RHOAIR = 1.0e17
BLANK = RHOAIR

IN_FMT = "mod"
OUT_FMT = ["rlm", "ubc"]

IN_MOD = "/home/vrath/JacoPyAn/work/UBC_format_example/UBI8_Z_Alpha02_NLCG_014"
LAT, LON = -16.346, -70.908

# =============================================================================
#  Read and convert
# =============================================================================
if "mod" in IN_FMT.lower():
    print("\n\nTransforming ModEM model file:")
    print(IN_MOD)
    OUT_MOD = IN_MOD

    start = time.perf_counter()
    dx, dy, dz, rho, refmod, _ = mod.read_mod(
        IN_MOD, ".rho", trans="linear", volumes=True
    )
    dims = np.shape(rho)
    elapsed = time.perf_counter() - start
    print(" Used %7.4f s for reading MOD model from %s " % (elapsed, IN_MOD))

    aircells = np.where(rho > RHOAIR / 10)

    for fmt in OUT_FMT:
        if "ubc" in fmt.lower():
            print("\n\nTransforming to UBC model & mesh format")
            start = time.perf_counter()
            elev = -refmod[2]
            refcenter = [LAT, LON, elev]
            mod.write_model_ubc(
                OUT_MOD, ".mesh", ".mod",
                dx, dy, dz, rho, refcenter,
                mvalair=BLANK, aircells=aircells,
            )
            elapsed = time.perf_counter() - start
            print(" Used %7.4f s for Writing UBC model to %s " % (elapsed, OUT_MOD))

        if "rlm" in fmt.lower() or "cgg" in fmt.lower():
            print("\n\nTransforming to RLM/CGG format")
            start = time.perf_counter()
            mod.write_rlm(
                OUT_MOD, modext=".rlm",
                dx=dx, dy=dy, dz=dz, mval=rho,
                reference=refmod, mvalair=BLANK, aircells=aircells,
                comment="RLM format model",
            )
            elapsed = time.perf_counter() - start
            print(" Used %7.4f s for Writing RLM/CGG model to %s " % (elapsed, OUT_MOD))
else:
    sys.exit(IN_FMT + " input format not yet implemented! Exit.")
