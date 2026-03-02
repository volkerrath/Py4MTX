#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Insert geometric bodies (ellipsoids or boxes) into a ModEM resistivity model.

Reads a ModEM model file, adds perturbation bodies with optional
smoothing, and writes the modified model.

@author: vrath
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
import jacproc as jac
from version import versionstrg

rng = np.random.default_rng()
nan = np.nan
version, _ = versionstrg()
titstrng = utl.print_title(
    version=version, fname=inspect.getfile(inspect.currentframe()), out=False
)
print(titstrng + "\n\n")

# =============================================================================
#  Configuration
# =============================================================================
rhoair = 1.0e17

ModFile_in = PY4MTX_DATA + "/Peru/1_feb_ell/TAC_100"
ModFile_out = ModFile_in

# Body definitions:
#   ['type', action, condition, cx, cy, cz, ax, ay, az, ang1, ang2, ang3]
action = ["rep", 1.0]
condition = "val <= np.log(1.)"
ell = [
    "ell", action, condition,
    0.0, 0.0, 10000.0,          # center (x, y, z)
    30000.0, 30000.0, 5000.0,   # semi-axes
    0.0, 0.0, 0.0,              # rotation angles
]

bodies = [ell]
additive = False
nb = len(bodies)

smoother = ["uniform", 1]

# =============================================================================
#  Read model
# =============================================================================
total = 0.0
start = time.perf_counter()

dx, dy, dz, rho, refmod, _ = mod.read_mod(ModFile_in, ".rho", trans="linear")
aircells = np.where(rho > rhoair / 10)

elapsed = time.perf_counter() - start
total += elapsed
print(" Used %7.4f s for reading model from %s " % (elapsed, ModFile_in + ".rho"))

# =============================================================================
#  Insert bodies
# =============================================================================
rho_in = mod.prepare_model(rho, rhoair=rhoair)

for ibody in range(nb):
    body = bodies[ibody]

    if "add" not in action:
        rho_out = mod.insert_body_condition(
            dx, dy, dz, rho_in, body, smooth=smoother, reference=refmod
        )
        Modout = ModFile_out + "_" + body[0] + str(ibody) + "_" + smoother[0]
        mod.write_mod(
            Modout, modext="_new.rho", trans="LOGE",
            dx=dx, dy=dy, dz=dz, mval=rho_out,
            reference=refmod, mvalair=1e17, aircells=aircells, header="",
        )
    elif ibody > 0:
        rho_in = rho_out.copy()
        rho_out = mod.insert_body(
            dx, dy, dz, rho_in, body, smooth=smoother, reference=refmod
        )

    elapsed = time.perf_counter() - start
    print(" Used %7.4f s for processing/writing model to %s" % (elapsed, Modout))

if "add" in action:
    Modout = ModFile_out + "_final"
    mod.write_mod(
        Modout, modext="_new.rho", trans="LOGE",
        dx=dx, dy=dy, dz=dz, mval=rho_out,
        reference=refmod, mvalair=1e17, aircells=aircells, header="",
    )

total += elapsed
print(" Total time used:  %f s " % total)
