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
import viz
import inverse as inv
import femtic as fem

from ediviz import add_phase, add_rho, add_tipper, add_pt
from ediproc import load_edi, dataframe_from_arrays, save_edi, compute_phase_tensor
# import cluster as fcm
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

Out = True
OutFile = '.edi'

Plot = True
if Plot:
    PlotFormat = ['.png', '.pdf']
String_out = '_processed'

PhasTens = True
Rotate = True
if Rotate:
    Declination = 0.  # 2.68   #E
    DecDeg = True

for edi in edi_files:

    freq, Z, T, station, _ = load_edi(edi, drop_invalid_periods=True)

    if PhasTens:
        PT = compute_phase_tensor(Z)
        df = dataframe_from_arrays(
            freq,
            Z,
            T=T,
            PT=PT,          # <- now PT columns appear again
            station=station,
        )
    else:
        df = dataframe_from_arrays(
            freq,
            Z,
            T=T,
            station=station,
        )



    if Out:
        if 'edi' in OutFile.lower():
            _ = save_edi(edi,
                         station + '_processed',
                         drop_invalid_periods=True,
                         rotate_deg=0.0,
                         )

    if Plot:
        fig, axs = plt.subplots(3, 2, figsize=(8, 14), sharex=True)
        add_rho(df, comps="xy,yx", ax=axs[0, 0])
        add_phase(df, comps="xy,yx", ax=axs[0, 1])
        add_rho(df, comps="xx,yy", ax=axs[1, 0])
        add_phase(df, comps="xx,yy", ax=axs[1, 1])
        add_tipper(df, ax=axs[2, 0])
        add_pt(df, ax=axs[2, 1])
        fig.suptitle(station)

        for f in PlotFormat:
            plt.savefig(station + String_out + f, dpi=600)

        plt.show()
