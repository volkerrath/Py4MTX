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

Provenance:
    2025-07-24  vrath   Created.
    2026-03-03  Claude  Renamed user-set parameters to UPPERCASE.
'''
import os
import sys
from pathlib import Path
import numpy as np
import inspect

import scipy.sparse as scs


PY4MTX_DATA = os.environ['PY4MTX_DATA']
PY4MTX_ROOT = os.environ['PY4MTX_ROOT']

for _base in [PY4MTX_ROOT + "/py4mt/modules/"]:
    for _p in [Path(_base), *Path(_base).rglob("*")]:
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

#import modules
import femtic as fem
import util as utl
from version import versionstrg


N_THREADS = '64'
os.environ['OMP_NUM_THREADS'] = N_THREADS
os.environ['OPENBLAS_NUM_THREADS'] = N_THREADS
os.environ['MKL_NUM_THREADS'] = N_THREADS
os.environ['MKL_PARDISO_OOC_MAX_CORE_SIZE'] = '10000'

n_threads = int(N_THREADS)

rng = np.random.default_rng()
nan = np.nan  # float('NaN')
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())

titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng+'\n\n')

WORK_DIR = '/home/vrath/FEMTIC_work/test/' #PY4MTX_DATA+'Misti/MISTI_test/'

MATRIX_IN = 'R'
FORMAT_IN =  'coo'
ROUGH_FILE = WORK_DIR +MATRIX_IN+'_'+FORMAT_IN+'.npz'

ALPHA = 1.
FACTOR = 1./ALPHA**2
REG_EPS = 1.e-4
SPARSIFY = [1.e-4, 10]
MATRIX_OUT = 'invRTR_'+str(int(np.abs(np.log10(SPARSIFY[0]))))+'-'+str(SPARSIFY[1])
FORMAT_OUT = 'csr'


ROUGH_NEW = ROUGH_FILE.replace(MATRIX_IN, MATRIX_OUT)
ROUGH_NEW = ROUGH_NEW.replace(FORMAT_IN, FORMAT_OUT)
print('Output M will be written to ', ROUGH_NEW)

R = scs.load_npz(ROUGH_FILE)
print(type(R))
print('Sparse format is', R.format)


M = fem.make_prior_cov(rough=R,
                          outmatrix = MATRIX_OUT,
                          regeps = REG_EPS,
                          spformat = FORMAT_OUT,
                          spthresh = SPARSIFY[0],
                          spfill = SPARSIFY[1],
                          spsolver = 'ilu',
                          spmeth = 'basic,area',
                          nthreads = n_threads,
                          out=True)

# fem.check_sparse_matrix(M)
print('matrix done')

M = FACTOR*M


if scs.issparse(M):
    print('M is sparse.')
    scs.save_npz(ROUGH_NEW, matrix=M)
else:
    print('M is dense.')
    np.savez_compressed(ROUGH_NEW, matrix=M)

print('all done')
