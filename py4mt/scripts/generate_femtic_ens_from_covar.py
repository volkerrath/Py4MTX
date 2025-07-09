#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Wed Apr 30 16:33:13 2025
Generate equivalent ensemble based on covariance

References:
    
K Osypov, Y Yang, A Fournier, N Ivanova, R Bachrach, C E Yarman, Y You, 
    D Nichols & M Woodward (2013):  Model-uncertainty quantification in 
    seismic tomography: method and applications
    Geophysical Prospecting, 61, 1114--1134, doi: 10.1111/1365-2478.12058


@author: vrath
'''
import os
import sys
import shutil
import numpy as np


import sklearn as skl
import sklearn.covariance 
import sklearn.decomposition 


from sksparse.cholmod import cholesky

import scipy as sc
import scipy.linalg as scl
import scipy.ndimage as sci
import scipy.sparse as scs

PY4MTX_DATA = os.environ['PY4MTX_DATA']
PY4MTX_ROOT = os.environ['PY4MTX_ROOT']

mypath = [PY4MTX_ROOT+'/py4mt/modules/', PY4MTX_ROOT+'/py4mt/scripts/']
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)

import femtic as fem
import util as utl
from version import versionstrg



rng = np.random.default_rng()
nan = np.nan  # float('NaN')
version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng+'\n\n')

CovarDir = r'/home/vrath/work/Ensembles/RTO/'
CovarResults = CovarDir+'RTO_results.npz'

NewEnsembleResults = CovarDir+'RTO_results.npz'
NewEnsembleSize = 100
NewEnsembleDir = r'/home/vrath/work/Ensembles/OSY/'
NewEnsembleFile =  NewEnsembleDir+'OSY_ensemble.npz'


tmp = np.load(CovarResults)
cov = tmp['rto_cov']
ref = tmp['rto_avg']
var = tmp['rto_var']

# from sksparse.cholmod import cholesky

sqrtcov = cholesky(cov)


# now generte new ensemble after Osypov(2013).

model_size = np.shape(ref)[0]
for imod in np.arange(NewEnsembleSize):
    sample = ref + sqrtcov * np.random.normal(loc=0., scale=1., size=model_size)
    if imod==0:
        new_ens = sample
    else:
        new_ens = np.vstack((new_ens, sample))

ensemble_dict ={'new_ens' : new_ens,
                'sqrtcov' : sqrtcov,
                'ref' : ref}
np.savez_compressed(NewEnsembleFile, **ensemble_dict)


