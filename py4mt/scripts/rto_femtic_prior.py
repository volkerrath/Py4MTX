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

"""
import os
import sys
import shutil
import numpy as np
import functools
import inspect


PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [PY4MTX_ROOT+"/py4mt/modules/", PY4MTX_ROOT+"/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)

#import modules
import femtic as fem
import util as utl
from version import versionstrg

import sklearn as skl
from sklearn.covariance import empirical_covariance


rng = np.random.default_rng()
nan = np.nan  # float("NaN")
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())

titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng+"\n\n")

ModelDir = PY4MTX_DATA+'Misti/test/'
RoughFile = ModelDir + "roughening_matrix.out"

ii, jj, val = fem.make_prior_cov(filerough=RoughFile , out=True)