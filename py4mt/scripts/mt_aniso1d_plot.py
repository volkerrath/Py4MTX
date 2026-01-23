#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Thu Oct 23 15:19:01 2025

@author: vrath
'''



import os
import sys
import time
from datetime import datetime
import warnings
import csv
import inspect

# Import numerical or other specialised modules
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.cm as cm
import mpl_toolkits.axes_grid1
import matplotlib.ticker


PY4MTX_DATA = os.environ['PY4MTX_DATA']
PY4MTX_ROOT = os.environ['PY4MTX_ROOT']

mypath = [PY4MTX_ROOT+'/py4mt/modules/', PY4MTX_ROOT+'/py4mt/scripts/']
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)

from aniso import aniso1d_impedance_sens
from data_viz import add_phase, add_rho, add_tipper, add_pt
from data_proc import load_edi, save_edi, save_ncd, save_hdf, save_npz
from data_proc import compute_pt, interpolate_data
from data_proc import set_errors, estimate_errors, rotate_data
from data_proc import calc_rhoa_phas

# from mtproc import calc_rhoa_phas
#from mcmc_funcs import pack_model, unpack_model
import viz
import util as utl
from version import versionstrg

pi = np.pi
mu0 = 4e-7 * pi
rng = np.random.default_rng()
nan = np.nan  # float('NaN')
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())

titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng+'\n\n')
