#!/usr/bin/env python3

'''
Reads ModEM's Jacobian, does fancy things.

@author: vrath   Feb 2021

'''

# Import required modules

import os
import sys

import inspect

# import struct
import time
from datetime import datetime
import warnings
import gc

import numpy as np
import numpy.linalg as npl
import scipy.linalg as scl
import scipy.sparse as scs



PY4MTX_DATA = os.environ['PY4MTX_DATA']
PY4MTX_ROOT = os.environ['PY4MTX_ROOT']

mypath = [PY4MTX_ROOT+'/py4mt/modules/', PY4MTX_ROOT+'/py4mt/scripts/']
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)


import jacproc as jac
import modem as mod
from version import versionstrg
import util as utl

version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=inspect.getfile(inspect.currentframe()), out=False)
print(titstrng+'\n\n')


gc.enable()

rng = np.random.default_rng()
blank = np.nan
rhoair = 17
hpads = 8
vpads = 30


# WorkDir = PY4MTX_DATA+'/Annecy/Jacobians/'
# WorkDir = PY4MTX_DATA+'/Peru/Sabancaya//SABA8_Jac/'
WorkDir = '/home/vrath/work/MT_Data/Sabancaya/Saba_best/SABA13a/'

if not WorkDir.endswith('/'):
    WorkDir = WorkDir+'/'
    
# MFile = WorkDir + 'SABA8_best.rho'
MFile = WorkDir + 'SABA13a'



total = 0.0

start = time.perf_counter()
dx, dy, dz, rho, refmod, _ = mod.read_mod(MFile, trans='log10')
elapsed = time.perf_counter() - start
total = total + elapsed
print(' Used %7.4f s for reading model from %s ' % (elapsed, MFile))

dims = np.shape(rho)
sdims = np.size(rho)

aircells = np.where(rho>rhoair/10)
rho[aircells] = blank


rho[:hpads,:,:]  = blank
rho[dims[0]-hpads:dims[0],:,:]  = blank

rho[:,:hpads,:] = blank
rho[:,dims[1]-hpads:dims[1],:,]  = blank

rho[:,:,dims[2]-vpads:dims[2]]

print('\n\n')
rhoavg = np.nanmean(rho)
print('Mean resistivity is', np.power(10., rhoavg))
print('Mean log resistivity is', rhoavg)
rhostd = np.sqrt(np.nanvar(rho))
print('Std log resistivity is', rhostd)

print('\n\n')
rhomed = np.nanmedian(rho)
print('Median resistivity is', np.power(10., rhomed))
print('Median log resistivity is', rhomed)
rhoquant = np.nanquantile(rho,[0.16, 0.84] )
print('1-sigma quantiles:', rhoquant)
rhoquant = np.nanquantile(rho,[0.023, 0.977] )
print('2-sigma quantiles:', rhoquant)
print('\n\n')
print()
