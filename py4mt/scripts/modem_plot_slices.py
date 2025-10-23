#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Plot slices through 3D model (modem)

Created on Mon Oct 20 10:09:45 2025

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
# from mtpy.core.mt import MT
from scipy.interpolate import RegularGridInterpolator

# add py4mt modules to pythonpath
mypath = ['/home/vrath/Py4MT/py4mt/modules/',
          '/home/vrath/Py4MT/py4mt/scripts/']
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)
        
from version import versionstrg
import inverse as inv
import util as utl
import viz
import modem as mod

# Import required py4mt modules for your script

# import jacproc as jac
# import mtproc as proc
# import plotrjmcmc as plmc
# import femtic as fem
# import cluster as fcm

rhoair = 1.e17
rng = np.random.default_rng()
nan = np.nan  # float('NaN')
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())

titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng+'\n\n')

Components = 3
UseAniso = True
    

WorkDir = '/home/vrath/FEMTIC_work/test/'  # PY4MTX_DATA+'Misti/MISTI_test/'
ModFile = [WorkDir+'/Peru/1_feb_ell/TAC_100']

PlotFile = WorkDir+'XXX'


rho = []

dx, dy, dz, rho, refmod, _ = mod.read_mod_aniso(
    ModFile,
    components=Components,
    trans='log10')
print(' read',str(Components),' model components from %s ' % (ModFile + '.rho'))
# rhotmp = mod.prepare_model(rhotmp, rhoair=rhoair)

print(np.shape(rho))
aircells = np.where(rho > rhoair/10)

rho_ref = np.mean(rho, axis=0)
print(np.shape(rho_ref))


# get cell centers

# to meshgrid

for ii in np.arange(np.shape(rho)[0]):
    # do something







