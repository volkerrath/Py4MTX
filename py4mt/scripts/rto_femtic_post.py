#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 16:33:13 2025

@author: vrath
"""
import os
import sys
import shutil
import numpy as np


import sklearn as skl
from sklearn.covariance import empirical_covariance
from sklearn.decomposition import PCA, IncrementalPCA, SparsePCA, TruncatedSVD

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [PY4MTX_ROOT+"/py4mt/modules/", PY4MTX_ROOT+"/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)

import femtic as fem
import util as utl
from version import versionstrg



rng = np.random.default_rng()
nan = np.nan  # float("NaN")
version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")

EnsembleDir = r'/home/vrath/work/Ensemble/Test/'
NRMSmax = 1.2


dir_list = utl.get_filelist(searchstr=["*ens*"], searchpath=EnsembleDir,fullpath=True)

model_list = []
model_count = -1
for dir in dir_list:
    print('\nInversion run',dir)
    cnv_file = dir+"/femtic.cnv"
    if not os.path.isfile(cnv_file):
        print(cnv_file, 'not found, run skipped.')
        continue
    
    with open(cnv_file) as file:
        cnv = file.readlines()
    info = cnv[-1].split()
    numit = int(info[0])
    nrms = float(info[8])

    if nrms > NRMSmax:
        print(dir,'not converged, run skipped.')
        continue
    model_count = model_count+1
    mod_file = dir+'/resistivity_block_iter'+str(numit)+'.dat'
    print( mod_file, ':')
    print(numit, nrms)
      
        
    model = fem.read_model(model_file=mod_file, model_trans='log10', out=True)
    
    if model_count==0:
        X = model
    else:
        X = np.vstack((X, model))
    
# print(np.shape(X))

# pca = PCA(n_components=6)
# pca.fit(X)

# print('\n')
# print('explained variance:')
# print(pca.explained_variance_ratio_)
# print('cummulative eplained variance:')
# print(np.cumsum(pca.explained_variance_ratio_))
# print ('singular_values:')
# print(pca.singular_values_)
    
C = empirical_covariance(X) 


# for isample in np.arange(N_samples):
#     if isample == 0:
#         rto_ens = m
#     else:
#         rto_ens = numpy.vstack((rto_ens, m))

# ne = numpy.shape(rto_ens)
# rto_avg = numpy.mean(rto_ens, axis=1)
# # rto_std = numpy.std(rto_ens, axis=1)
# rto_var = numpy.var(rto_ens, axis=1)
# rto_med = numpy.median(rto_ens, axis=1)
# # print(numpy.shape(rto_ens), numpy.shape(rto_med))
# # print(ne)
# mm = numpy.tile(rto_med, (ne[1], 1))
# # print(numpy.shape(mm))

# rto_mad = numpy.median(
#     numpy.abs(rto_ens.T - numpy.tile(rto_med, (ne[1], 1))))

# rto_prc = numpy.percentile(rto_ens, Percentiles)

# rtofields = (
#     ("rto_avg", rto_avg),
#     ("rto_var", rto_var),
#     ("rto_med", rto_med),
#     ("rto_mad", rto_mad),
#     ("rto_prc", rto_prc)
# )
# rto_results.update(rtofields)

# if "ens" in rto_out:
#     rto_results["jcn_ens"] = rto_ens

# return rto_results
