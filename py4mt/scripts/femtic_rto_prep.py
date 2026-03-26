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

Provenance:
    2025-04-30  vrath   Created.
    2026-03-03  Claude  Renamed user-set parameters to UPPERCASE.
    2026-03-24  Claude  Added visualization config blocks (data + model
                        ensemble plots); helper functions live in
                        femtic_viz.plot_data_ensemble and
                        femtic_viz.plot_model_ensemble.
'''

import os
import sys
import numpy as np
import inspect

'''
Specialized toolboxes settings and imports.
'''
import scipy.sparse as scs

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
import ensembles as ens
import femtic_viz as fviz

from util import stop

N_THREADS = '32'
os.environ['OMP_NUM_THREADS'] = N_THREADS
os.environ['OPENBLAS_NUM_THREADS'] = N_THREADS
os.environ['MKL_NUM_THREADS'] = N_THREADS

rng = np.random.default_rng()
nan = np.nan  # float('NaN')
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())

titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng + '\n\n')

'''
Base setup.
'''
N_SAMPLES = 32
ENSEMBLE_DIR = r'/home/vrath/FEMTIC_work/ens_test/'
TEMPLATES = ENSEMBLE_DIR + 'templates/'
FILES = ['control.dat',
         'observe.dat',
         'mesh.dat',
         'referencemodel.dat',
         'resistivity_block_iter0.dat',
         'distortion_iter0.dat',
         'run_femtic_dias.sh',]
# 'run_femtic_kraken.sh']

ENSEMBLE_NAME = 'misti_rto_'

'''
Control number of ensemble members for increase of sample number or restart
of badly converged samples (see femtic_rto_post.py)
'''
FROM_TO = None


'''
Set up mode of model perturbations.
'''
PERTURB_MOD = True
if PERTURB_MOD:
    MOD_REF = 'resistivity_block_iter30.dat'
    MOD_METHOD = 'add'
    # if ModCov is not None, this needs to be normal
    MOD_PDF = ['normal', 0., 0.3]
    # ['exp', L], ['gauss', L], ['matern], L, MatPars], ['femtic'], None
    MOD_R = 'femtic R'
    R_FILE = 'R_coo'
else:
    MOD_R = None

'''
Set up mode of data perturbations.
'''
PERTURB_DAT = True
if PERTURB_DAT:
    DAT_METHOD = 'add',
    DAT_PDF = ['normal', 0., 1.0]

RESET_ERRORS = True
if RESET_ERRORS:
    ERRORS = [
        [0.15, .05, .05, 0.15]*2,        # Impedance
        [0.03, 0.03]*2,                   # VTF
        [.5, .2, .2, .5],                 # PT
    ]
else:
    ERRORS = []


'''
Generate ensemble directories and copy template files.
'''
dir_list = ens.generate_directories(alg='rto',
                                    dir_base=ENSEMBLE_DIR + ENSEMBLE_NAME,
                                    templates=TEMPLATES,
                                    file_list=FILES,
                                    n_samples=N_SAMPLES,
                                    fromto=FROM_TO,
                                    out=True)
print('\n')

'''
Draw perturbed data sets: d̃ ∼ N(d, Cd)
'''
data_ensemble = ens.generate_data_ensemble(alg='rto',
                                           dir_base=ENSEMBLE_DIR + ENSEMBLE_NAME,
                                           n_samples=N_SAMPLES,
                                           fromto=FROM_TO,
                                           file_in='observe.dat',
                                           draw_from=DAT_PDF,
                                           method=DAT_METHOD,
                                           errors=ERRORS,
                                           out=True)
print('data ensemble ready!')
print('\n')

'''
Draw perturbed model sets: m̃ ∼ N(m, Cm)

Read prior parameter precision Q = R^T@R for perturbations
if needed. If the femtic mode is chosen, the matrix needs to be
read from external file.
'''
if 'Q' in MOD_R:
    Q = scs.load_npz(ENSEMBLE_DIR + ENSEMBLE_NAME + R_FILE + '.npz')
else:
    R = scs.load_npz(ENSEMBLE_DIR + ENSEMBLE_NAME + R_FILE + '.npz')
    Q = R.T @ R

print('roughness loaded with shape:', np.shape(Q))

model_ensemble = ens.generate_model_ensemble(alg='rto',
                                             dir_base=ENSEMBLE_DIR + ENSEMBLE_NAME,
                                             n_samples=N_SAMPLES,
                                             fromto=FROM_TO,
                                             refmod=MOD_REF,
                                             method=MOD_METHOD,
                                             q=Q,
                                             out=True)
print('\n')
print('model ensemble ready!')


# =============================================================================
# Visualization configuration
# =============================================================================

'''
Select which ensemble members to visualise (fixed list of sample indices).
Applies to both the data plot and the model plot.
'''
VIZ_SAMPLES = [0, 1, 2, 3]   # adjust as needed; must be < N_SAMPLES

'''
Data visualization
------------------
Joint plot of original vs. perturbed observe.dat for the selected samples.
One subplot row per sample; original (solid) and perturbed (dashed) curves
are overlaid on the same axes.

Helper: femtic_viz.plot_data_ensemble
'''
PLOT_DATA = True
if PLOT_DATA:
    import matplotlib.pyplot as plt

    DAT_WHAT = 'rho'       # 'rho' | 'phase' | 'tipper' | 'pt'
    DAT_COMPS = 'xy,yx'
    DAT_SHOW_ERRORS = True

    dat_orig_file = TEMPLATES + 'observe.dat'
    dat_ens_files = [
        ENSEMBLE_DIR + ENSEMBLE_NAME + f'{i}/observe.dat'
        for i in range(N_SAMPLES)
    ]

    fig_dat, axs_dat = fviz.plot_data_ensemble(
        orig_file=dat_orig_file,
        ens_files=dat_ens_files,
        sample_indices=VIZ_SAMPLES,
        comps=DAT_COMPS,
        what=DAT_WHAT,
        show_errors=DAT_SHOW_ERRORS,
        out=True,
    )
    fig_dat.savefig(ENSEMBLE_DIR + 'rto_data_ensemble.pdf', bbox_inches='tight')
    plt.close(fig_dat)
    print('data ensemble plot saved.')

'''
Model visualization
-------------------
Joint plot of original vs. perturbed resistivity models for the selected
samples, shown across 1-5 user-defined slices (map or curtain).
Original and perturbed sit in adjacent rows; slices in columns.

Helper: femtic_viz.plot_model_ensemble
'''
PLOT_MODEL = True
if PLOT_MODEL:
    import matplotlib.pyplot as plt

    MOD_MESH = TEMPLATES + 'mesh.dat'
    MOD_ORIG = TEMPLATES + MOD_REF

    mod_ens_files = [
        ENSEMBLE_DIR + ENSEMBLE_NAME + f'{i}/{MOD_REF}'
        for i in range(N_SAMPLES)
    ]

    # Define 1-5 slices.  Each dict must have 'type': 'map' or 'curtain'
    # plus the keyword arguments for the corresponding femtic_viz function.
    # Adjust z0 / dz / polyline / width to match your model geometry.
    MOD_SLICES = [
        {'type': 'map',     'z0': -500,  'dz': 50},
        {'type': 'map',     'z0': -2000, 'dz': 50},
        {'type': 'curtain',
         'polyline': np.array([[0., 0.], [10000., 0.]]),
         'width': 500},
    ]

    fig_mod, axs_mod = fviz.plot_model_ensemble(
        orig_mod_file=MOD_ORIG,
        ens_mod_files=mod_ens_files,
        mesh_file=MOD_MESH,
        sample_indices=VIZ_SAMPLES,
        slices=MOD_SLICES,
        mode='tri',
        log10=True,
        cmap='jet_r',
        clim=None,      # auto-derive from original model
        out=True,
    )
    fig_mod.savefig(ENSEMBLE_DIR + 'rto_model_ensemble.pdf', bbox_inches='tight')
    plt.close(fig_mod)
    print('model ensemble plot saved.')
