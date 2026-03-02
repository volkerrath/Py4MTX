#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate multiple perturbed ModEM models for resolution testing.

Reads a base model and Jacobian SVD, inserts distributed perturbation
bodies (regular or random), and projects them through the Jacobian
null space.

@author: vrath
"""

import os
import sys
import inspect
import time

import numpy as np
import numpy.linalg as npl
import scipy.sparse as scs

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [PY4MTX_ROOT + "/py4mt/modules/", PY4MTX_ROOT + "/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)

import jacproc as jac
import modem as mod
import util as utl
from version import versionstrg

rng = np.random.default_rng()
nan = np.nan
blank = 1.0e-30
rhoair = 1.0e17

version, _ = versionstrg()
titstrng = utl.print_title(
    version=version, fname=inspect.getfile(inspect.currentframe()), out=False
)
print(titstrng + "\n\n")

# =============================================================================
#  Configuration
# =============================================================================
ModDir_in = PY4MTX_DATA + "/Peru/Misti/"
ModDir_out = ModDir_in + "/results_shuttle/"

ModFile_in = ModDir_in + "Misti10_best"
ModFile_out = "Misti10_best"
ModOrig = [-16.277300, -71.444397]  # Misti

SVDFile = ModDir_in + "Misti_best_Z5_nerr_sp-8"

ModOutSingle = True

if not os.path.isdir(ModDir_out):
    print("File: %s does not exist, but will be created" % ModDir_out)
    os.mkdir(ModDir_out)

padding = [10, 10, 10, 10, 0, 20]
bodymask = [3, 3, 5]
bodyval = 0.2
flip = "alt"

# Random perturbed grid
model_set = 10
method = [
    ["random", 25, [1, 1, 1, 1, 1, 1], "uniform", [3, 3, 5], 6],
]

# =============================================================================
#  Read model and Jacobian SVD
# =============================================================================
total = 0.0
start = time.perf_counter()
dx, dy, dz, base_model, refmod, _ = mod.read_mod(
    ModFile_in, ".rho", trans="log10"
)
mdims = np.shape(base_model)
aircells = np.where(base_model > np.log10(rhoair / 10.0))
jacmask = jac.set_airmask(
    rho=base_model, aircells=aircells, blank=np.log10(blank), flat=False, out=True
)
jacflat = jacmask.flatten(order="F")
elapsed = time.perf_counter() - start
total += elapsed
print(" Used %7.4f s for reading model from %s " % (elapsed, ModFile_in + ".rho"))

start = time.perf_counter()
print("Reading Jacobian SVD from " + SVDFile)
SVD = np.load(SVDFile)
U = SVD["U"]
S = SVD["S"]
print(np.shape(U), np.shape(S))
elapsed = time.perf_counter() - start
print(" Used %7.4f s for reading Jacobian/data from %s" % (elapsed, SVDFile))
total += elapsed

# =============================================================================
#  Generate perturbed models
# =============================================================================
for ibody in range(model_set):
    model = base_model.copy()
    templ = mod.distribute_bodies_ijk(model=model, method=method)
    new_model = mod.insert_body_ijk(
        rho_in=model, template=templ, perturb=bodyval, bodymask=bodymask
    )
    new_model[aircells] = rhoair

    ModFile = ModDir_out + ModFile_out + "_" + str(ibody) + "+perturbed.rho"
    Header = "# " + ModFile

    rho_proj = jac.project_model(
        m=model, U=U, tst_sample=new_model, nsamp=1
    )

    # TODO: Write output model files here
    # mod.write_mod_npz(...)

elapsed = time.perf_counter() - start
total += elapsed
print(" Total time used:  %f s " % total)
