#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute randomised SVD of a ModEM Jacobian matrix.

Reads a processed (sparsified) Jacobian and computes truncated SVDs at
various ranks, oversampling factors, and subspace iterations. Reports
the operator-norm accuracy as a percentage of the full Jacobian.

@author: vrath
"""

import os

nthreads = 8
os.environ["OMP_NUM_THREADS"] = str(nthreads)
os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
os.environ["MKL_NUM_THREADS"] = str(nthreads)

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

import jac_proc as jac
import modem as mod
import util as utl
from version import versionstrg

version, _ = versionstrg()
titstrng = utl.print_title(
    version=version, fname=inspect.getfile(inspect.currentframe()), out=False
)
print(titstrng + "\n\n")

rng = np.random.default_rng()

# =============================================================================
#  Configuration
# =============================================================================
WorkDir = "/home/vrath/ModEM_work/Ub25_ZT_600_PT_jac/"
JName = "Ub25_ZPT_nerr_sp-6"
JFile = WorkDir + JName
MFile = WorkDir + "Ub_600ZT4_PT_NLCG_009"

OutName = "_SVD"
NumSingular = [100, 200, 300, 400, 500, 1000]
OverSample = [2]
SubspaceIt = [0]

# =============================================================================
#  Read Jacobian
# =============================================================================
total = 0.0
start = time.perf_counter()
print("\nReading Data from " + JFile)

Jac = scs.load_npz(JFile + "_jac.npz")
Dat = np.load(JFile + "_info.npz", allow_pickle=True)
Freq = Dat["Freq"]
Comp = Dat["Comp"]
Site = Dat["Site"]
DTyp = Dat["DTyp"]
Data = Dat["Data"]
Scale = Dat["Scale"]
Info = Dat["Info"]

elapsed = time.perf_counter() - start
total += elapsed
print(" Used %7.4f s for reading Jacobian from %s " % (elapsed, JFile))

# =============================================================================
#  Compute randomised SVDs
# =============================================================================
info = []

for noversmp in OverSample:
    for nsubspit in SubspaceIt:
        for nsingval in NumSingular:
            start = time.perf_counter()
            U, S, Vt = jac.rsvd(
                Jac.T, rank=nsingval,
                n_oversamples=noversmp * nsingval,
                n_subspace_iters=nsubspit,
            )
            elapsed = time.perf_counter() - start
            print("Used %7.4f s for calculating k = %i SVD " % (elapsed, nsingval))
            print("Oversampling factor = ", str(noversmp))
            print("Subspace iterations = ", str(nsubspit))

            D = U @ scs.diags(S[:]) @ Vt - Jac.T
            x_op = rng.normal(size=np.shape(D)[1])
            n_op = npl.norm(D @ x_op) / npl.norm(x_op)
            j_op = npl.norm(Jac.T @ x_op) / npl.norm(x_op)
            perc = 100.0 - n_op * 100.0 / j_op
            info.append([nsingval, noversmp, nsubspit, perc, elapsed])

            File = (
                JFile + "_SVD_k" + str(nsingval) + "_o" + str(noversmp)
                + "_s" + str(nsubspit) + "_" + str(np.around(perc, 1))
                + "percent.npz"
            )
            np.savez_compressed(File, U=U, S=S, V=Vt, Nop=perc)

np.savetxt(JFile + OutName + ".dat", np.vstack(info),
           fmt="%6i, %6i, %6i, %4.6g, %4.6g")
