#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
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
from scipy.sparse.linalg import norm

PY4MTX_DATA = os.environ['PY4MTX_DATA']
PY4MTX_ROOT = os.environ['PY4MTX_ROOT']

mypath = [PY4MTX_ROOT+'/py4mt/modules/', PY4MTX_ROOT+'/py4mt/scripts/']
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)

#import modules
import femtic as fem
import util as utl
from version import versionstrg


N_THREADS = '16'
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

WorkDir = '/home/vrath/FEMTIC_work/test/' #PY4MTX_DATA+'Misti/MISTI_test/'

RoughFile0 = WorkDir +'R_csc.npz'
RoughFile1 = WorkDir +'RT_csc.npz'
Roughfile2 = WorkDir +'RTR_csc.npz'

InvFile0 = WorkDir +'InvR.npz'
InvFile1 = WorkDir +'InvRT.npz'
Invfile2 = WorkDir +'InvRTR.npz'


R0 = scs.load_npz(RoughFile0)
R1 = scs.load_npz(RoughFile1)
R2 = scs.load_npz(RoughFile2)

R3 = R1@R0

Test = np.norm(R3-R2)
print('invR type is', type(invR))

