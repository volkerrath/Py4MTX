#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Thu May 29 10:09:45 2025

@author: vrath
'''

# Import python modules
# edit according to your needs
import os
import sys

import time
from datetime import datetime
import warnings
import csv
import inspect
import argparse

# Import numerical or other specialised modules
import numpy as np
import scipy as sc
import numpy.linalg as nla

PY4MTX_DATA = os.environ['PY4MTX_DATA']
PY4MTX_ROOT = os.environ['PY4MTX_ROOT']

# add py4mt modules to pythonpath
mypath = [PY4MTX_ROOT + '/py4mt/modules/',
          PY4MTX_ROOT + '/py4mt/scripts/']
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)

# Import required py4mt modules for your script
import util as utl
import viz
import inverse as inv
import femtic as fem
import ensembles as ens
import femtic_viz as femviz

from version import versionstrg


rng = np.random.default_rng()
nan = np.nan  # float('NaN')
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())
titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng + '\n\n')


EnsembleDir = r'/home/vrath/FEMTIC_work/ens_annecy/'
EnsembleName = 'ann_'
EnsembleFile = 'AnnecyENS'
EnsembleEOF = 'AnnecyEOF'

OutStrng = '_trunc'
GetComponents = False
if GetComponents:
    OutStrng = '_comp'


SearchStrng = EnsembleName + '*'
dir_list = utl.get_filelist(searchstr=[SearchStrng], searchpath=EnsembleDir,
                            sortedlist=True, fullpath=True)

nens = len(dir_list)


ens_num = -1
for directory in dir_list:
    ens_num = ens_num + 1
    num_best, nrm_best = fem.get_nrms(directory)

    print('\n', directory)
    print('min(nrmse) = ', nrm_best, 'at iteration', num_best)
    mesh_file = directory + '/mesh.dat'
    modl_file = directory + '/resistivity_block_iter' + str(num_best) + '.dat'
    print(modl_file)

    modl = fem.read_model(
        model_file=modl_file,
        model_trans="log10")[2:,]

    modl = modl[:, None]
    if ens_num == 0:
        ensemble = modl
    else:
        ensemble = np.concatenate((ensemble, modl), axis=1)
    print(np.shape(ensemble))


storedict = {'ensemble': ensemble}
np.savez_compressed(EnsembleDir + EnsembleFile +'.npz', **storedict)

"""
Compute EOFs (Empirical Orthogonal Functions) from an ensemble matrix.

Assumes E has shape (ncells, nsamples), i.e. each column is one sample.
Returns spatial EOFs (patterns) and PCs (coefficients per sample).

Parameters
----------
E : ndarray, shape (ncells, nsamples)
    Ensemble matrix. Columns are samples.
k : int or None, optional
    Number of leading EOFs to return. If None, return all (<= nsamples).
method : {"svd", "sample_space"}, optional
    - "svd": thin SVD on anomaly matrix (recommended).
    - "sample_space": eigen-decomposition of nsamples×nsamples Gram matrix.
demean : bool, optional
    If True, remove per-cell mean across samples.
ddof : int, optional
    Degrees of freedom for covariance scaling. Use 1 for sample covariance
    (divide by nsamples-1), or 0 for population (divide by nsamples).
eps : float, optional
    Small cutoff to avoid division by ~0 eigenvalues.

Returns
-------
eofs : ndarray, shape (ncells, k)
    Spatial EOF patterns (columns). Orthonormal in Euclidean inner product.
pcs : ndarray, shape (k, nsamples)
    Principal components (coefficients per sample).
evals : ndarray, shape (k,)
    Eigenvalues of the (scaled) covariance matrix (variance explained).
frac : ndarray, shape (k,)
    Fraction of total variance explained.
mean : ndarray, shape (ncells,)
    Mean removed from each cell (zeros if demean=False).

Notes
-----
- EOFs are defined up to sign; both EOF and corresponding PC may flip sign.
- With nsamples << ncells, both methods are efficient. "svd" is simplest.
"""
eofs, pcs, w_k, frac, mean = ens.compute_eofs(
    E=ensemble, method="svd", demean=True,)
print('eofs:', np.shape(eofs))

storedict = {'eofs': eofs,
             'pcs': pcs,
             'frac': frac,
             'w_k': w_k,
             'mean': mean
             }
np.savez_compressed(EnsembleDir + EnsembleEOF + '.npz', **storedict)


"""
Draw one new physical-space sample from a truncated EOF model.

Parameters
----------
eofs : (ncells, r)
    EOF spatial modes (columns).
pcs : (r, nsamples)
    Training PCs. Used to estimate per-mode variance unless whitened=True.
mean : (ncells,), optional
    Mean field to add back.
k0, k1 : int
    Truncation window [k0, k1) in Python slicing convention.
groups : list[(k0,k1)] or None
    If provided, sample each window; return sum (default) or components.
rng : np.random.Generator, optional
    Random generator. If None, uses default_rng().
method : {"empirical_diag", "empirical_full"}
    - "empirical_diag": assume PCs independent; use per-mode variance.
    - "empirical_full": estimate full covariance across selected modes.
whitened : bool
    If True, assume PCs already have unit variance; sample N(0, I).
ddof : int
    ddof for variance/covariance estimation.
return_components : bool
    If True, return (ngroups, ncells) components; else sum to (ncells,).

Returns
-------
x : ndarray
    Sample in physical space: (ncells,) or (ngroups, ncells).
"""
eof_results = []
eof_norms = []


for ie in np.arange(nens):
    if GetComponents:
        k0, k1 = ie, ie + 1
    else:
        k0, k1 = 0, ie + 1

    tmp = ens.eof_sample(
        eofs, pcs, mean,
        k0=k0,
        k1=k1,
        method="empirical_diag",
        return_components=False,
    )
    eof_results.append(np.power(10., tmp))

    frac = ens.eof_captured_fraction_from_pcs(
        pcs,
        k0=k0,
        k1=k1,
    )



    eof_norms.append(frac)

    print(np.shape(eof_results))

enorms = np.array(eof_norms)
nnorms = enorms / enorms[-1]
# cnorm = np.array(cnorm)
print(enorms)
print(nnorms)


tens = len(eof_results)
for ie in np.arange(tens):
    file = EnsembleDir + EnsembleEOF + OutStrng + str(ie) + '.npz'
    fem.insert_model(
        template=EnsembleDir + 'resistivity_block_iter.dat',
        model=eof_results[ie],
        model_file=file,
    )
