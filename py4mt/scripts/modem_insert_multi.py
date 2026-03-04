#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate multiple perturbed ModEM models for resolution testing.

Reads a base model and Jacobian SVD, inserts distributed perturbation
bodies (regular or random), and projects them through the Jacobian
null space.

@author: vrath
Cleanup: 4 Mar 2026 by Claude (Anthropic)
"""

import os
import sys
import inspect
import time

import numpy as np
import scipy.sparse as scs

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [PY4MTX_ROOT + "/py4mt/modules/", PY4MTX_ROOT + "/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)

import jac_proc as jac
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
BLANK = 1.0e-30
RHOAIR = 1.0e17

MOD_DIR_IN = PY4MTX_DATA + "/Peru/Misti/"
MOD_DIR_OUT = MOD_DIR_IN + "/results_shuttle/"

MOD_FILE_IN = MOD_DIR_IN + "Misti10_best"
MOD_FILE_OUT = "Misti10_best"
MOD_ORIG = [-16.277300, -71.444397]  # Misti

SVD_FILE = MOD_DIR_IN + "Misti_best_Z5_nerr_sp-8"

MOD_OUT_SINGLE = True

if not os.path.isdir(MOD_DIR_OUT):
    print("Directory: %s does not exist, but will be created" % MOD_DIR_OUT)
    os.mkdir(MOD_DIR_OUT)

PADDING = [10, 10, 10, 10, 0, 20]
BODY_MASK = [3, 3, 5]
BODY_VAL = 0.2
FLIP = "alt"

# Random perturbed grid
MODEL_SET = 10
METHOD = [
    ["random", 25, [1, 1, 1, 1, 1, 1], "uniform", [3, 3, 5], 6],
]

# =============================================================================
#  Read model and Jacobian SVD
# =============================================================================
total = 0.0
start = time.perf_counter()
dx, dy, dz, base_model, refmod, _ = mod.read_mod(
    MOD_FILE_IN, ".rho", trans="log10"
)
mdims = np.shape(base_model)
aircells = np.where(base_model > np.log10(RHOAIR / 10.0))
jacmask = jac.set_airmask(
    rho=base_model, aircells=aircells, blank=np.log10(BLANK), flat=False, out=True
)
jacflat = jacmask.flatten(order="F")
elapsed = time.perf_counter() - start
total += elapsed
print(" Used %7.4f s for reading model from %s " % (elapsed, MOD_FILE_IN + ".rho"))

start = time.perf_counter()
print("Reading Jacobian SVD from " + SVD_FILE)
SVD = np.load(SVD_FILE)
U = SVD["U"]
S = SVD["S"]
print(np.shape(U), np.shape(S))
elapsed = time.perf_counter() - start
print(" Used %7.4f s for reading Jacobian/data from %s" % (elapsed, SVD_FILE))
total += elapsed

# =============================================================================
#  Generate perturbed models
# =============================================================================
for ibody in range(MODEL_SET):
    model = base_model.copy()
    templ = mod.distribute_bodies_ijk(model=model, method=METHOD)
    new_model = mod.insert_body_ijk(
        rho_in=model, template=templ, perturb=BODY_VAL, bodymask=BODY_MASK
    )
    new_model[aircells] = RHOAIR

    ModFile = MOD_DIR_OUT + MOD_FILE_OUT + "_" + str(ibody) + "+perturbed.rho"
    Header = "# " + ModFile

    rho_proj = jac.project_model(
        m=model, U=U, tst_sample=new_model, nsamp=1
    )

    # TODO: Write output model files here
    # mod.write_mod_npz(...)

elapsed = time.perf_counter() - start
total += elapsed
print(" Total time used:  %f s " % total)
