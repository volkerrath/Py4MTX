#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''

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

'''
import os
import sys
import shutil
import numpy as np
import functools
import inspect
import time 

import sklearn as skl
import scipy.sparse as scs


PY4MTX_DATA = os.os.environ['PY4MTX_DATA']
PY4MTX_ROOT = os.os.environ['PY4MTX_ROOT']

mypath = [PY4MTX_ROOT+'/py4mt/modules/', PY4MTX_ROOT+'/py4mt/scripts/']
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)

#import modules
import femtic as fem
import util as utl
from version import versionstrg


N_THREADS = '32'
os.environ['OMP_NUM_THREADS'] = N_THREADS
os.environ['OPENBLAS_NUM_THREADS'] = N_THREADS
os.environ['MKL_NUM_THREADS'] = N_THREADS
#os.environ['VECLIB_MAXIMUM_THREADS'] = N_THREADS
#os.environ['NUMEXPR_NUM_THREADS'] = N_THREADS

rng = np.random.default_rng()
nan = np.nan  # float('NaN')
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())

titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng+'\n\n')

ModelDir = '/home/vrath/FEMTIC_work/test/' #PY4MTX_DATA+'Misti/MISTI_test/'

RtR = False


if RtR:
    RoughFile = ModelDir +'RTR.npz'
else:
    RoughFile = ModelDir +'R.npz'

RoughNew =  ModelDir +'Ri.npz'

R = scs.load_npz(RoughNew, matrix=r_out)
print('R sparse format is', R.getformat())

R_inv = fem.make_prior_cov(rough=R,
                   small = 1.e-5,
                   rtr = RtR,
                   out=True)


scs.save_npz(RoughNew, matrix=R_inv)
