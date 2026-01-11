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

# Import numerical or other specialised modules
import numpy as np
import scipy as sci
import pandas as pd
import matplotlib.pyplot as plt

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
import modem as mod
# import jacproc as jac
import mtproc as mtp
# import plotrjmcmc as plmc
# import viz
# import inverse as inv
import femtic as fem
from scipy.interpolate import make_smoothing_spline
from data_viz import add_phase, add_rho, add_tipper, add_pt
from data_proc import load_edi, save_edi, save_ncd, save_hdf, save_npz
from data_proc import compute_pt, compute_zdet, compute_zssq
from data_proc import dataframe_from_edi, interpolate_data
from data_proc import set_errors, estimate_errors, rotate_data

from version import versionstrg


rng = np.random.default_rng()
nan = np.nan  # float('NaN')
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())
titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng + '\n\n')

# WorkDir = '/home/vrath/ChatGPT_tests/'
WorkDir = '/home/vrath/Py4MTX/work/EPS_2025_Annecy/edi/Sites_Name/'

if not os.path.isdir(WorkDir):
    print(' File: %s does not exist, but will be created' % WorkDir)
    os.mkdir(WorkDir)


DataDir = WorkDir  # +'/edi/'
edi_files = mtp.get_edi_list(DataDir, fullpath=True)
ns = np.size(edi_files)

OutFiles = 'edi, ncd, hdf, npz'

Plot = True
if Plot:
    PlotFormat = ['.png', '.pdf']
# %%
NameStr = '_processed'


SetErrors = False
Errors = {'Zerr': [0.1, 0.1, 0.1, 0.1],
          'Terr': [0.03, 0.03, 0.03, 0.03],
          'PTerr': [0.1, 0.1, 0.1, 0.1]
          }

PhasTens = True
Invars = True

Interpolate = False
if Interpolate:
    FreqPerDec = 6
    IntMethod = [None, FreqPerDec]

EstimateErrors = False
if EstimateErrors:
    sys.exit('Work in progress! Exit')
    spread = 2.  # *std-dev
    ErrMethod = ['gcvspline', spread]


Rotate = False
if Rotate:
    Angle = 0.  # 2.68   #E
    DecDeg = True


for edi in edi_files:

    edi_dict = load_edi(edi, drop_invalid_periods=True)
    # def load_edi(
    #     path: str | Path,
    #     *,
    #     prefer_spectra: bool = True,
    #     ref: str = "RH",
    #     err_kind: str = "var",
    #     drop_invalid_periods: bool = True,
    #     invalid_sentinel: float = 1.0e30,
    # ) -> Dict[str, Any]:

    station = edi_dict['station']
    Z = edi_dict['Z']
    Zerr = edi_dict['Z_err']
    T = edi_dict['T']
    Terr = edi_dict['T_err']

    '''
    Task block
    '''

    if PhasTens:
        P, Perr = compute_pt(Z, Zerr)
        edi_dict['P'] = P

        edi_dict['P_err'] = Perr

    if Invars:

        Zdet, Zdeterr = compute_zdet(Z)
        edi_dict['Zdet'] = Zdet
        edi_dict['Zdet_err'] = Zdeterr
        Zssq, Zssqerr = compute_zssq(Z)
        edi_dict['Zssq'] = Zssq
        edi_dict['Zssq_err'] = Zssqerr

    if EstimateErrors:
        edi_dict = estimate_errors(edi_dict=edi_dict, method=ErrMethod)

    if SetErrors:
        edi_dict = set_errors(edi_dict=edi_dict, errors=Errors)

    if Interpolate:
        edi_dict = interpolate_data(edi_dict=edi_dict, method=IntMethod)

    if Rotate:
        edi_dict = rotate_data(edi_dict=edi_dict, angle=Angle)


    # print(np.shape(Z), np.shape(Zerr),
    #       np.shape(T), np.shape(Terr),
    #       np.shape(P), np.shape(Perr))
    # print(list(edi_dict.keys()))

    pltargs = {'show_errors': True}

    if 'edi' in OutFiles.lower():
        _ = save_edi(
            path=DataDir + station + NameStr+'.edi',
            edi=edi_dict
        )

    if 'ncd' in OutFiles.lower():
        _ = save_ncd(
            path = DataDir + station + NameStr+'.ncd',
            data_dict=edi_dict)

    if 'hdf' in OutFiles.lower():
        _ = save_hdf(
            path = DataDir + station + NameStr+'.hdf',
            data_dict=edi_dict)


    if 'npz' in OutFiles.lower():
            _ = save_npz(
            path = DataDir + station + NameStr+'.npz',
            data_dict = edi_dict)

    if Plot:
        fig, axs = plt.subplots(3, 2, figsize=(8, 14), sharex=True)
        add_rho(edi_dict, comps="xy,yx", ax=axs[0, 0], **pltargs)
        add_phase(edi_dict, comps="xy,yx", ax=axs[0, 1], **pltargs)
        add_rho(edi_dict, comps="xx,yy", ax=axs[1, 0], **pltargs)
        add_phase(edi_dict, comps="xx,yy", ax=axs[1, 1], **pltargs)
        add_tipper(edi_dict, ax=axs[2, 0], **pltargs)
        add_pt(edi_dict, ax=axs[2, 1], **pltargs)
        fig.suptitle(station + NameStr.replace('_', ' | '))

        # for ax in np.atleast_1d(axs).ravel():
        # Remove empty axes
        for ax in axs.flat:
            if not ax.lines and not ax.images and not ax.collections:
                fig.delaxes(ax)

        fig.tight_layout(rect=[0, 0, 1, 0.97])

        for f in PlotFormat:
            plt.savefig(WorkDir + station + NameStr + f, dpi=600)

        plt.show()
