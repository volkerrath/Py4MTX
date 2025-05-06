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


N_samples = 3
EnsembleDir = r'./'
Templates = EnsembleDir+'templates/'
Files = ['control.dat', 
         'observe.dat', 
         'mesh.dat',     
         'resistivity_block_iter0.dat',
         'distortion_iter0.dat',
         'run_femtic_dub.sh',
         'run_femtic_oar.sh']

dir_list = fem.generate_directories(
        dir_base=EnsembleDir+'ens_',
        templates=Templates,
        file_list=Files, 
        N_samples=10, 
        out = True)

# """
# Draw perturbed data set: d  ̃ ∼ N (d, Cd)
# """
# d_ens = generate_data_ensemble(dref=d_obs, dact=d_act,
#                                        nens=nsamples,
#                                        perturb=["gauss", d_err],
#                                        out=OutInfo)
# """
# Draw prior model: m̃ ∼ N (0, 1 (LT L)−1 )
# """
# m_ens = generate_param_ensemble(mref=m_ref, mact=m_act,
#                                         nens=nsamples,
#                                         perturb=["gauss", c_ref,
#                                                  numpy.array([])],
#                                                  out=OutInfo)
