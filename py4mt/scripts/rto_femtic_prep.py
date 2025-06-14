#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Run the randomize-then-optimize (RTO) algorithm:

    for i = 1 : nsamples do
        Draw perturbed data set: d_pert∼ N (d, Cd)
        Draw prior model: m̃ ∼ N (0, 1/mu (LT L)^−1 )
        Solve determistic problem  to get the model m_i
    end

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

vr July 2022


Created on Wed Apr 30 16:33:13 2025

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


rng = np.random.default_rng()
nan = np.nan  # float("NaN")
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())

titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng+"\n\n")


N_samples = 32
EnsembleDir = r'/home/vrath/work/Ensemble/RTO/'
Templates = EnsembleDir+'templates/'
Files = ['control.dat',
         'observe.dat',
         'mesh.dat',
         'resistivity_block_iter0.dat',
         'distortion_iter0.dat',
         'run_femtic_dub.sh',
         'run_femtic_oar.sh']

EnsembleName = 'rto_'
PerturbMod = True
if PerturbMod:
    Mod_method = 'add'
    Mod_pdf = ['normal', 0., 0.3]
    
PerturbDat = True 
if PerturbDat: 
    Dat_method ='add',
    Dat_pdf = ['normal', 0., 1.0]

ResetErrors=True
if ResetErrors:
    Errors =[
        [15., 4.,  5., 15.],        # Impeedance in percent
        [0.03, 0.03],               # VTF
        [.5, .2,  .2, .5],          # PT
        ]
else:
    Errors =[]

# os.chdir(EnsembleDir)

dir_list = fem.generate_directories(
    dir_base=EnsembleDir+EnsembleName,
    templates=Templates,
    file_list=Files,
    N_samples=N_samples,
    out=True)




"""
Draw perturbed model sets: d  ̃ ∼ N (m, Cm)
"""


model_ensemble = fem.generate_model_ensemble(dir_base=EnsembleDir+EnsembleName,
                                          N_samples=N_samples,
                                          file_in='resistivity_block_iter0.dat',
                                          draw_from=Mod_pdf,
                                          method=Mod_method,
                                          out=True)
print('\n')

"""
Draw perturbed data sets: d  ̃ ∼ N (d, Cd)
"""

data_ensemble = fem.generate_data_ensemble(dir_base=EnsembleDir+EnsembleName,
                                          N_samples=N_samples,
                                          file_in='observe.dat',
                                          draw_from=Dat_pdf,
                                          method=Dat_method,
                                          errors = Errors,
                                          out=True)
