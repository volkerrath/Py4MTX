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

PY4MTX_DATA = os.environ['PY4MTX_DATA']
PY4MTX_ROOT = os.environ['PY4MTX_ROOT']

mypath = [PY4MTX_ROOT+'/py4mt/modules/', PY4MTX_ROOT+'/py4mt/scripts/']
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)

from version import versionstrg

import util as utl
import viz
from aniso import cpanis, z1anis, dphase

pi = np.pi
mu0 = 4e-7 * pi
rng = np.random.default_rng()
nan = np.nan  # float('NaN')
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())

titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng+'\n\n')

WorkDir = PY4MTX_ROOT+'/aniso/'
ModFile = WorkDir+'ana.dat'
ResFile = ModFile.replace('.dat', '_summary.dat')
ImpFile = ModFile.replace('.dat', '_impedance.dat')
RhoFile = ModFile.replace('.dat', '_rhophas.dat')

Periods = np.logspace(-2., 2., 41)

'''
Model A from Pek, J. and Santos, F. A. M., 2002.
'''
NLayer = 4
Model = [
    [10.,  10000.,  10000.,  10000.,    0.,  0.,  0.,],
    [18.,    200.,  20000.,    200.,   15.,  0.,  0.,],
    [100.,  1000.,   2000.,   1000.,  -75.,  0.,  0.,],
    [0.,     100.,    100.,    100.,    0.,  0.,  0.,]
        ]


# '''
# Model B from Pek, J. and Santos, F. A. M., 2002.
# '''
# NLayer = 2
# Model = [
#     [10.,   1000.,   1000.,   1000.,    0.,  0.,  0.,],
#     [0.,     10.,    300.,    100.,   15., 60., 30.,]
# ]

model = np.array(Model) 

# Allocate arrays

h   = model[:,0]
rop = model[:,1:4]
ustr = model[:,4]
udip =  model[:,5]
usla =  model[:,6]



# === Compute conductivity tensors and effective parameters ===
sg, al, at, blt = cpanis(rop[:NLayer], ustr[:NLayer], udip[:NLayer], usla[:NLayer])

# === Write model summary ===
with open(ResFile, 'w') as f:
    line = '# Model parameters'
    f.write(line + "\n")    
    line = '# layer, thick (km)  res_max, res_min, strike, dip, slant' 
    f.write(line + "\n")
    for layer in range(NLayer):      
        pars =  f'{layer:5d} {h[layer]:12.4f}'
        rops = f'   {rop[layer,0]:12.4f} {rop[layer,1]:12.4f} {rop[layer,2]:12.4f} ' 
        angs = f'   {ustr[layer]:12.4f} {udip[layer]:12.4f} {usla[layer]:12.0f} ' 
        # for i in range(3):
        #     for j in range(3):
        #         pars = pars +f'  {sg[layer, i, j]:14.5f}'
        line = pars+rops+angs
        f.write(line + "\n")
# === Loop over periods ===
with open(ImpFile, 'w') as f:
    line = '#   PERIOD,  Re Zxx,  Im Zxx,  Re Zxy,  Im Zxy ,  Re Zyx,  Im Zyx,  Re Zyy,  Im Zyy'
    f.write(line + "\n")
    for per in Periods:
        z = z1anis(NLayer, h[:NLayer], al[:NLayer], at[:NLayer],
                   blt[:NLayer], np.array([per]))[:, :, 0]
        z_flat = z.flatten()
        interlaced = [f"{tmp.real:.5e} {tmp.imag:.5e}" for tmp in z_flat]
        line = f"{per:.5e} " + "  ".join(interlaced)
        f.write(line + "\n")

with open(RhoFile, 'w') as f:
    line = '#   PERIOD,  Rhoa xx,  Phs xx,  Rhoa xy,  Phs xy,  Rhoa yx,  Phs yx,  Rhoa yy,  Phs yy'
    f.write(line + "\n")
    for per in Periods:
        z = z1anis(NLayer, h[:NLayer], al[:NLayer], at[:NLayer],
                   blt[:NLayer], np.array([per]))[:, :, 0]
        omega = 2.0 * pi / per
        prev = 1.0 / (omega * mu0)
        rapp = np.abs(z)**2 * prev
        papp = np.array(
            [[180.0 * dphase(z[ii, jj]) / pi for jj in range(2)] for ii in range(2)])
        line = [f"{per:.5e}"]
        for i in range(2):
            for j in range(2):
                line.append(f"{rapp[i, j]:.5e}")
                line.append(f"{papp[i, j]:.1f}")
        f.write("  ".join(line) + "\n")

