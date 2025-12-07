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
# import femtic as fem
from scipy.interpolate import make_smoothing_spline
from ediviz import add_phase, add_rho, add_tipper, add_pt
from ediproc import load_edi, save_edi, save_ncd, save_hdf
from ediproc import compute_pt, dataframe_from_arrays, interpolate_data
from ediproc import set_errors, estimate_errors, rotate_data

from version import versionstrg


rng = np.random.default_rng()
nan = np.nan  # float('NaN')
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())
titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng + '\n\n')

WorkDir = '/home/vrath/ChatGPT_tests/'
if not os.path.isdir(WorkDir):
    print(' File: %s does not exist, but will be created' % WorkDir)
    os.mkdir(WorkDir)


EdiDir = WorkDir  # +'/edi/'
edi_files = mtp.get_edi_list(EdiDir, fullpath=True)
ns = np.size(edi_files)

OutFiles = 'edi, ncd, hdf'

Plot = True
if Plot:
    PlotFormat = ['.png', '.pdf']
String_out = '_processed'

SetErrors = False
Errors = {'Zerr': [0.1, 0.1, 0.1, 0.1],
          'Terr': [0.03, 0.03, 0.03, 0.03],
          'PTerr': [0.1, 0.1, 0.1, 0.1]
          }

PhasTens = True

Interpolate = True
if Interpolate:
    FreqPerDec = 6
    IntMethod = [None, FreqPerDec]

EstimateErrors = False
if EstimateErrors:
    sys.exit('Work in progress! Exit')
    spread = 2.  # *std-dev
    ErrMethod = ['gcvspline', spread]


Rotate = True
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
    Zerr = edi_dict['Zerr']
    T = edi_dict['T']
    Terr = edi_dict['Terr']

    '''
    Task block
    '''

    if PhasTens:
        # generate phase tensors+errors, if not already in edi_dict
        # def compute_pt(
        #     Z: np.ndarray,
        #     Z_err: Optional[np.ndarray] = None,
        #     *,
        #     err_kind: str = "var",
        #     nsim: int = 200,
        #     random_state: Optional[np.random.Generator] = None,
        # ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        PT, PTerr = compute_pt(Z)
        edi_dict['PT'] = PT
        edi_dict['PTerr'] = PTerr

    if EstimateErrors:
        edi_dict = estimate_errors(edi_dict=edi_dict, method=ErrMethod)

    if SetErrors:
        edi_dict = set_errors(edi_dict=edi_dict, errors=Errors)

    if Interpolate:
        edi_dict = interpolate_data(edi_dict=edi_dict, method=IntMethod)

    if Rotate:
        edi_dict = rotate_data(edi_dict=edi_dict, angle=Angle)

    '''
    store final dict into pandas data frame for export to hdf/netcdf
    and plotting
    '''

    df = pd.DataFrame(edi_dict)

    if 'edi' in OutFiles.lower():
        _ = save_edi(edi_dict,
                     station + '_processed.edi',
                     drop_invalid_periods=True,
                     rotate_deg=0.0,
                     )

    if 'ncd' in OutFiles.lower():
        _ = save_edi(edi_dict,
                     station + '_processed.ncd',
                     drop_invalid_periods=True,
                     rotate_deg=0.0,
                     )
        # df: pd.DataFrame,
        # path: str | Path,
        # *,
        # engine: Optional[str] = None,
        # dim: str = "period",
        # dataset_name: str = "mt",
    if 'hdf' in OutFiles.lower():
        _ = save_edi(edi_dict,
                     station + '_processed.hdf',
                     drop_invalid_periods=True,
                     rotate_deg=0.0,
                     )
        # df: pd.DataFrame,
        # path: str | Path,
        # *,
        # key: str = "mt",
        # mode: str = "w",
        # complevel: int = 4,
        # complib: str = "zlib",
        # **kwargs: Any,

    if Plot:
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
