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


EnsembleDir = r'/home/vrath/work/Ensemble/Test/'
NRMSmax = 1.2


dir_list = utl.get_filelist(searchstr=["*ens*"], searchpath=EnsembleDir,fullpath=True)

model_list = []
model_count = -1
for dir in dir_list:
    print('\nInversion run',dir)
    cnv_file = dir+"/femtic.cnv"
    if not os.path.isfile(cnv_file.isfile()):
        continue
    
    with open(cnv_file) as file:
        cnv = file.readlines()
    info = cnv[-1].split()
    numit = int(info[0])
    nrms = float(info[8])

    if nrms > NRMSmax:
        print(dir,'not converged, file skipped.')
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
    
print(np.shape(X))

pca = PCA(n_components=6)
pca.fit(X)


print(pca.explained_variance_ratio_)
print(pca.singular_values_)
    
C = empirical_covariance(X) 


# model_ensemble = fem.generate_model_ensemble(dir_base=EnsembleDir+'ens_',
#                                           N_samples=N_samples,
#                                           file_in='resistivity_block_iter0.dat',
#                                           draw_from=Mod_pdf,
#                                           out=True)
