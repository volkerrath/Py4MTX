#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Insert geometric bodies (ellipsoids or boxes) into a ModEM resistivity model.

Reads a ModEM model file, adds perturbation bodies with optional
smoothing, and writes the modified model.

@author: vrath
Cleanup: 4 Mar 2026 by Claude (Anthropic)
"""

import os
import sys
import inspect
import time

import numpy as np

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [PY4MTX_ROOT + "/py4mt/modules/", PY4MTX_ROOT + "/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)

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
RHOAIR = 1.0e17

MOD_FILE_IN = PY4MTX_DATA + "/Peru/1_feb_ell/TAC_100"
MOD_FILE_OUT = MOD_FILE_IN

# Body definitions:
#   ['type', action, condition, cx, cy, cz, ax, ay, az, ang1, ang2, ang3]
ACTION = ["rep", 1.0]
CONDITION = "val <= np.log(1.)"
ELL = [
    "ell", ACTION, CONDITION,
    0.0, 0.0, 10000.0,          # center (x, y, z)
    30000.0, 30000.0, 5000.0,   # semi-axes
    0.0, 0.0, 0.0,              # rotation angles
]

BODIES = [ELL]
ADDITIVE = False
SMOOTHER = ["uniform", 1]

# =============================================================================
#  Read model
# =============================================================================
total = 0.0
start = time.perf_counter()

dx, dy, dz, rho, refmod, _ = mod.read_mod(MOD_FILE_IN, ".rho", trans="linear")
aircells = np.where(rho > RHOAIR / 10)

elapsed = time.perf_counter() - start
total += elapsed
print(" Used %7.4f s for reading model from %s " % (elapsed, MOD_FILE_IN + ".rho"))

# =============================================================================
#  Insert bodies
# =============================================================================
nb = len(BODIES)
rho_in = mod.prepare_model(rho, rhoair=RHOAIR)

for ibody in range(nb):
    body = BODIES[ibody]

    if "add" not in ACTION:
        rho_out = mod.insert_body_condition(
            dx, dy, dz, rho_in, body, smooth=SMOOTHER, reference=refmod
        )
        Modout = MOD_FILE_OUT + "_" + body[0] + str(ibody) + "_" + SMOOTHER[0]
        mod.write_mod(
            Modout, modext="_new.rho", trans="LOGE",
            dx=dx, dy=dy, dz=dz, mval=rho_out,
            reference=refmod, mvalair=1e17, aircells=aircells, header="",
        )
    elif ibody > 0:
        rho_in = rho_out.copy()
        rho_out = mod.insert_body(
            dx, dy, dz, rho_in, body, smooth=SMOOTHER, reference=refmod
        )

    elapsed = time.perf_counter() - start
    print(" Used %7.4f s for processing/writing model to %s" % (elapsed, Modout))

if "add" in ACTION:
    Modout = MOD_FILE_OUT + "_final"
    mod.write_mod(
        Modout, modext="_new.rho", trans="LOGE",
        dx=dx, dy=dy, dz=dz, mval=rho_out,
        reference=refmod, mvalair=1e17, aircells=aircells, header="",
    )

total += elapsed
print(" Total time used:  %f s " % total)
