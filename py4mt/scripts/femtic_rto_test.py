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
from scipy.sparse import csr_array, csc_array, coo_array, eye_array, issparse
from scipy.sparse.linalg import inv, spsolve, factorized, splu, spilu
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
from scipy.sparse.linalg import norm

import matplotlib.pyplot as plt
from matspy import spy_to_mpl, spy



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

SparseFormat = 'coo'

RoughFile = WorkDir+'R_'+SparseFormat+'.npz'

#CovFile0 = WorkDir +'COV_'+SparseFormat+'.npz'



R = scs.load_npz(RoughFile)
fem.check_sparse_matrix(R)

R = coo_matrix(R)

T = R.T - R
print(' R-R^T max/min:', T.max(), T.min())
if T.max()+T.min()==0.:
    print('Matrix is symmetric!')


# Plotting
options = {'title': '$\mathbf{R}$, Sparsity Pattern',
           'figsize': 8.,      #  inches
           'dpi': 600,
           'shading': 'binary', # 'absolute' 'relative' 'binary'
           'spy_aa_tweaks_enabled': True,
           'color_full': 'black'} 

fig, ax = spy_to_mpl(R, **options)
fig.show()
fig.savefig(WorkDir+'R_spy.png', bbox_inches='tight')
fig.savefig(WorkDir+'R_spy.pdf', bbox_inches='tight')
plt.close()
