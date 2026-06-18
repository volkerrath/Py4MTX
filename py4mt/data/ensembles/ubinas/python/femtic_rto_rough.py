#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Get prior model covariance for  randomize-then-optimize (RTO) algorithm:

See:

Bardsley, J. M.; Solonen, A.; Haario, H. & Laine, M.
    Randomize-Then-Optimize: a Method for Sampling from Posterior
    Distributions in Nonlinear Inverse Problems
    SIAM J. Sci. Comp., 2014, 36, A1895-A1910

Blatter, D.; Morzfeld, M.; Key, K. & Constable, S.
    Uncertainty quantification for regularized inversion of electromagnetic
    geophysical data. Part I: Motivation and Theory
    Geophysical Journal International, doi:10.1093/gji/ggac241, 2022

Blatter, D.; Morzfeld, M.; Key, K. & Constable, S.
    Uncertainty quantification for regularized inversion of electromagnetic
    geophysical data – Part II: application in 1-D and 2-D problems
    Geophysical Journal International, , doi:10.1093/gji/ggac242, 2022

vr July 2025

Created on Thu Jul 24 10:25:11 2025

@author: vrath

Provenance:
    2025-07-24  vrath   Created.
    2026-03-03  Claude  Renamed user-set parameters to UPPERCASE.
"""
import os
import sys
from pathlib import Path
import numpy as np
import inspect

import scipy.sparse as scs


PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

for _base in [PY4MTX_ROOT + "/py4mt/modules/"]:
    for _p in [Path(_base), *Path(_base).rglob("*")]:
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

#import modules
import femtic as fem
import util as utl
from version import versionstrg


N_THREADS = "32"
os.environ["OMP_NUM_THREADS"] = N_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = N_THREADS
os.environ["MKL_NUM_THREADS"] = N_THREADS

rng = np.random.default_rng()
nan = np.nan  # float("NaN")
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())

titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng+"\n\n")

WORK_DIR = "/home/vrath/Ensembles/misti_rto/work/" #PY4MTX_DATA+"Misti/MISTI_test/"
WORK_DIR = "/home/vrath/Py4MTX/py4mt/data/rto/ubinas/work/"
ROUGH_FILE = WORK_DIR + "roughening_matrix.out"



OUT_ROUGH = "R"
SPARSE_FORMAT = "coo"
ROUGH_NEW = WORK_DIR+OUT_ROUGH+"_"+SPARSE_FORMAT+".npz"


R   = fem.get_roughness(filerough=ROUGH_FILE,
                   spformat = SPARSE_FORMAT,
                   out=True)


if "q" in OUT_ROUGH.lower():
    Q = R.T@R
    fem.check_sparse_matrix(Q)
    ROUGH_NEW = WORK_DIR+"Q_"+SPARSE_FORMAT+".npz"
    print("saved to", ROUGH_NEW)
    print("Sparse format is", Q.format)
    scs.save_npz(ROUGH_NEW, Q)
else:
    fem.check_sparse_matrix(R)
    ROUGH_NEW = WORK_DIR+"R_"+SPARSE_FORMAT+".npz"
    print("saved to", ROUGH_NEW)
    print("Sparse format is", R.format)
    scs.save_npz(ROUGH_NEW, R)
