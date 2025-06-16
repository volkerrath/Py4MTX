#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Wed Apr 30 16:33:13 2025

@author: vrath
'''
import os
import sys
import shutil
import numpy as np


import sklearn as skl
from sklearn.covariance import empirical_covariance
from sklearn.decomposition import PCA, IncrementalPCA, SparsePCA, TruncatedSVD

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

EnsembleDir = r'/home/vrath/Ensembles/RTO/'
EnsembleName = 'rto_*'
NRMSmax = 1.4
# Percentiles = numpy.array([10., 20., 30., 40., 50., 60., 70., 80., 90.]) # linear
Percentiles = [2.3, 15.9, 50., 84.1,97.7]                   # 95/68
EnsembleResults = EnsembleDir+'RTO_results.npz'

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
        rto_ens = model
    else:
        rto_ens = np.vstack((rto_ens, model))

rto_cov = empirical_covariance(rto_ens)

# pca = PCA(n_components=6)
# pca.fit(X)

# print('\n')
# print('explained variance:')
# print(pca.explained_variance_ratio_)
# print('cummulative eplained variance:')
# print(np.cumsum(pca.explained_variance_ratio_))
# print ('singular_values:')
# print(pca.singular_values_)
    

ne = np.shape(rto_ens)
rto_avg = np.mean(rto_ens, axis=1)
# rto_std = np.std(rto_ens, axis=1)
rto_var = np.var(rto_ens, axis=1)
rto_med = np.median(rto_ens, axis=1)
# print(np.shape(rto_ens), np.shape(rto_med))
# print(ne)
rto_mad = np.median(
    np.abs(rto_ens.T - np.tile(rto_med, (ne[1], 1))))
rto_prc = np.percentile(rto_ens, Percentiles)

results_dict ={'model_list' : model_list,
    'rto_ens' : rto_ens,
    'rto_cov' : rto_cov,
    'rto_avg' : rto_avg,
    'rto_var' : rto_var,
    'rto_med' : rto_med,
    'rto_mad' : rto_mad,
    'rto_prc' : rto_prc}

np.savez_compressed(EnsembleResults, **results_dict)

