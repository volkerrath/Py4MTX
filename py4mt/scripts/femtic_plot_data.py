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

import functools
import inspect
import time
from datetime import datetime
import warnings
import csv
'''
specialized toolboxes settings and imports.
'''
# import sklearn as skl
# from sklearn.covariance import empirical_covariance
import scipy.sparse as scs
from scipy.interpolate import make_smoothing_spline

# Import numerical or other specialised modules
import numpy as np
import scipy as sci
import pandas as pd
import matplotlib.pyplot as plt

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

from dataviz import add_phase, add_rho, add_tipper, add_pt
from dataproc import load_edi, save_edi, save_ncd, save_hdf
from dataproc import compute_pt, dataframe_from_arrays, interpolate_data
from dataproc import set_errors, estimate_errors, rotate_data


from util import stop

N_THREADS = '10'
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
N_samples = 1
DatDir = r'/home/vrath/FEMTIC_work/ens_misti/misti_rto_01/'
DatList= ['misti_rto_01/observation.dat']
NPZList =  ['misti_rto_01/observation.npz']


'''
plot data sets
'''
for site in NPZList:
    data = np.load(site)

    df = pd.DataFrame(data)


    fig, axs = plt.subplots(3, 2, figsize=(8, 14), sharex=True)

    add_rho(df, comps="xy,yx", ax=axs[0, 0])
    add_phase(df, comps="xy,yx", ax=axs[0, 1])
    add_rho(df, comps="xx,yy", ax=axs[1, 0])
    add_phase(df, comps="xx,yy", ax=axs[1, 1])
    add_tipper(df, ax=axs[2, 0])
    add_pt(df, ax=axs[2, 1])
    fig.suptitle(station)

# Remove empty axes
for ax in axs:
    if not ax.lines and not ax.images and not ax.collections:
        fig.delaxes(ax)

for f in PlotFormat:
    plt.savefig(WorkDir + station + String_out + f, dpi=600)

plt.show()


print('data plots ready!')
