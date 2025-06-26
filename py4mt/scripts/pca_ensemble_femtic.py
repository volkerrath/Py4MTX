#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Calculate PCA for model ensemble

Sat Jun  7 04:53:42 PM CEST 2025

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
from sklearn.decomposition import PCA, IncrementalPCA, SparsePCA, TruncatedSVD

rng = np.random.default_rng()
nan = np.nan  # float("NaN")
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())

titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng+"\n\n")


EnsembleDir = r'/home/vrath/work/Ensembles/RTO/'
EnsembleName = 'rto_*'
NRMSmax = 1.4
# Percentiles = numpy.array([10., 20., 30., 40., 50., 60., 70., 80., 90.]) # linear
Percentiles = [2.3, 15.9, 50., 84.1,97.7]                   # 95/68
EnsembleResults = EnsembleDir+'RTO_pca.npz'

dir_list = utl.get_filelist(
    searchstr=[EnsembleName],
    searchpath=EnsembleDir,
    fullpath=True)


model_list = []
model_count = -1
for dir in dir_list:
    print('\nInversion run',dir)
    cnv_file = dir+'/femtic.cnv'
    if not os.path.isfile(cnv_file):
        print(cnv_file, 'not found, run skipped.')
        continue
    
    with open(cnv_file) as file:
        cnv = file.readlines()
    info = cnv[-1].split()
    numit = int(info[0])
    nrms = float(info[8])


    if nrms > NRMSmax:
        print(dir,'nRMS =',nrms)
        print(dir,'not converged, run skipped.')
        continue
    model_count = model_count+1
    mod_file = dir+'/resistivity_block_iter'+str(numit)+'.dat'
    print( mod_file, ':')
    print(numit, nrms)
    model_list.append([mod_file,numit, nrms])
        
    model = fem.read_model(model_file=mod_file, model_trans='log10', out=True)
    
    if model_count==0:
        ensemble = model
    else:
        ensemble = np.vstack((ensemble, model))
    
for ipca in np.arange(1, model_count):
    pca = PCA(n_components=ipca)
    pca.fit(ensemble)
    print('\n')
    print(ipca, 'explained variance:')
    print(pca.explained_variance_ratio_)
    print(ipca, 'cummulative eplained variance:')
    print(np.cumsum(pca.explained_variance_ratio_))
    print (ipca, 'singular_values:')
    print(pca.singular_values_)

results_dict ={'model_list' : model_list,
    'ensemble' : ensemble,
}

np.savez_compressed(EnsembleResults, **results_dict)
