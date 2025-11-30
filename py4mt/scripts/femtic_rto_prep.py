#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''

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


Created on Wed Apr 30 16:33:13 2025

@author: vrath
'''

import os
import sys
import shutil
import numpy as np
import functools
import inspect

'''
specialized toolboxes settings and imports.
'''
# import sklearn as skl
# from sklearn.covariance import empirical_covariance


'''
Py4MTX-specific settings and imports.
'''
PY4MTX_DATA = os.environ['PY4MTX_DATA']
PY4MTX_ROOT = os.environ['PY4MTX_ROOT']

mypath = [PY4MTX_ROOT + '/py4mt/modules/', PY4MTX_ROOT + '/py4mt/scripts/']
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)

# import modules
from version import versionstrg
import util as utl
import femtic as fem

rng = np.random.default_rng()
nan = np.nan  # float('NaN')
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())

titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng + '\n\n')

'''
Base setup.
'''
N_samples = 32
EnsembleDir = r'/home/vrath/work/Ensemble/RTO/'
Templates = EnsembleDir + 'templates/'
Files = ['control.dat',
         'observe.dat',
         'mesh.dat',
         'resistivity_block_iter0.dat',
         'distortion_iter0.dat',
         'run_femtic_dub.sh',
         'run_femtic_oar.sh']

EnsembleName = 'rto_'

'''
Control number of ensemble members for increase of smple number or restart
of badly converged samples (see femtic_rto_post.py)
'''
FromTo = np.arange(0, N_samples)


'''
Set up mode of model perturbations.
'''
PerturbMod = True
if PerturbMod:
    Mod_method = 'add'
    # if ModCov is not None, this needs to be normal
    Mod_pdf = ['normal', 0., 0.3]
    # ['exp', L], ['gauss', L], ['matern], L, MatPars], ['femtic'], None
    Mod_R = 'femtic R'
    R_file = 'R_coo'
else:
    Mod_R = None

'''
Set up mode of data perturbations.
'''

PerturbDat = True
if PerturbDat:
    Dat_method = 'add',
    Dat_pdf = ['normal', 0., 1.0]

ResetErrors = True
if ResetErrors:
    Errors = [
        [15., 4., 5., 15.],        # Impedance in percent
        [0.03, 0.03],               # VTF
        [.5, .2, .2, .5],          # PT
    ]
else:
    Errors = []


'''
Generate ensemble directories and copy template files.
'''

dir_list = fem.generate_directories(
    dir_base=EnsembleDir + EnsembleName,
    templates=Templates,
    file_list=Files,
    n_samples=N_samples,
    fromto=FromTo,
    out=True)


'''
Draw perturbed model sets: d  ̃ ∼ N (m, Cm)

Read prior parameter precision Q = R^T@R for perturbations
if needed. If the femtic mode is chosen, the martix needs to be
read from external file.
'''
if 'Q' in Mod_R:
    Q = np.load(EnsembleDir + EnsembleName + R_file + '.npz')['R']
else:
    R = np.load(EnsembleDir + EnsembleName + R_file + '.npz')['R']
    Q = R.T@R


model_ensemble = fem.generate_model_ensemble(dir_base=EnsembleDir + EnsembleName,
                                             n_samples=N_samples,
                                             fromto=FromTo,
                                             file_in='resistivity_block_iter0.dat',
                                             draw_from=Mod_pdf,
                                             method=Mod_method,
                                             q=Q,
                                             out=True)
print('\n')

'''
Draw perturbed data sets: d  ̃ ∼ N (d, Cd)
'''

data_ensemble = fem.generate_data_ensemble(dir_base=EnsembleDir + EnsembleName,
                                           n_samples=N_samples,
                                           fromto=FromTo,
                                           file_in='observe.dat',
                                           draw_from=Dat_pdf,
                                           method=Dat_method,
                                           errors=Errors,
                                           out=True)
