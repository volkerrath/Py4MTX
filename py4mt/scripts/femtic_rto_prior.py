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

# import sklearn as skl
import scipy.sparse as scs


PY4MTX_DATA = os.environ['PY4MTX_DATA']
PY4MTX_ROOT = os.environ['PY4MTX_ROOT']

mypath = [PY4MTX_ROOT+'/py4mt/modules/', PY4MTX_ROOT+'/py4mt/scripts/']
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)

#import modules
import femtic as fem
import ensembles as ens
import util as utl
#import inverse as inv
from version import versionstrg


N_THREADS = '64'
os.environ['OMP_NUM_THREADS'] = N_THREADS
os.environ['OPENBLAS_NUM_THREADS'] = N_THREADS
os.environ['MKL_NUM_THREADS'] = N_THREADS
os.environ['MKL_PARDISO_OOC_MAX_CORE_SIZE'] = '10000'
#os.environ['VECLIB_MAXIMUM_THREADS'] = N_THREADS
#os.environ['NUMEXPR_NUM_THREADS'] = N_THREADS

n_threads = int(N_THREADS)

rng = np.random.default_rng()
nan = np.nan  # float('NaN')
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())

titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng+'\n\n')

WorkDir = '/home/vrath/FEMTIC_work/test/' #PY4MTX_DATA+'Misti/MISTI_test/'

MatrixIn = 'R'
FormatIn =  'coo'
RoughFile = WorkDir +MatrixIn+'_'+FormatIn+'.npz'

Alpha = 1.
Factor = 1./Alpha**2
RegEps = 1.e-4
Sparsify = [1.e-4, 10]
MatrixOut = 'invRTR_'+str(int(np.abs(np.log10(Sparsify[0]))))+'-'+str(Sparsify[1])
FormatOut = 'csr'


RoughNew = RoughFile.replace(MatrixIn, MatrixOut)
RoughNew = RoughNew.replace(FormatIn, FormatOut)
print('Output M will be written to ', RoughNew)

R = scs.load_npz(RoughFile)
print(type(R))
print('Sparse format is', R.format)


M = fem.make_prior_cov(rough=R,
                          outmatrix = MatrixOut,
                          regeps = RegEps,
                          spformat = FormatOut,
                          spthresh = Sparsify[0],
                          spfill = Sparsify[1],
                          spsolver = 'ilu',
                          spmeth = 'basic,area',
                          nthreads = n_threads,
                          out=True)

# fem.check_sparse_matrix(M)
print('matrix done')

M = Factor*M


if scs.issparse(M):
    print('M is sparse.')
    scs.save_npz(RoughNew, matrix=M)
else:
    print('M is dense.')
    np.savez_compressed(RoughNew, matrix=M)

print('all done')
