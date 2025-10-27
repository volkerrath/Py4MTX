#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Thu Oct 23 15:19:01 2025

@author: vrath
'''

from aniso import *
import util as utl
from version import versionstrg
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
import modem 
from aniso import *

rhoair = 1.e17
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

Periods = np.logspace(-4., 4., 41)

pi = np.pi
mu0 = 4e-7 * pi
nlmax = 1001


SearchStrng = '*.edi'
dir_list = utl.get_filelist(searchstr=[SearchStrng], searchpath=WorkDir, 
                            sortedlist =True, fullpath=True)



# hstart = np.log10(100./1000.) 
# hfinal = np.log10(3000./1000.)
# h    = np.logspace(hstart, hfinal, NLayer)
# maxdepth = np.sum(h)
# print(maxdepth)

# # Allocate arrays
# h = np.zeros(nlmax)
# rop = np.zeros((nlmax, 3))
# ustr = np.zeros(nlmax)
# udip = np.zeros(nlmax)
# usla = np.zeros(nlmax)

# # === Read model parameters ===
# with open(ModFile, 'r') as f:
#     nl = int(f.readline())
#     for layer in range(nl):
#         line = f.readline().split()
#         h[layer] = float(line[0])
#         rop[layer, :] = list(map(float, line[1:4]))
#         ustr[layer] = float(line[4])
#         udip[layer] = float(line[5])
#         usla[layer] = float(line[6])

# # === Compute conductivity tensors and effective parameters ===
# sg, al, at, blt = cpanis(rop[:nl], ustr[:nl], udip[:nl], usla[:nl])

# # === Write model summary ===
# with open(ResFile, 'w') as f:
#     line = '# Model parameters'
#     f.write(line + "\n")    
#     line = '# layer, thick (km)  res_max, res_min, strike, dip, slant' 
#     f.write(line + "\n")
#     for layer in range(nl):      
#         pars =  f'{layer:5d} {h[layer]:12.4f}'
#         rops = f'   {rop[layer,0]:12.4f} {rop[layer,1]:12.4f} {rop[layer,2]:12.4f} ' 
#         angs = f'   {ustr[layer]:12.4f} {udip[layer]:12.4f} {usla[layer]:12.0f} ' 
#         # for i in range(3):
#         #     for j in range(3):
#         #         pars = pars +f'  {sg[layer, i, j]:14.5f}'
#         line = pars+rops+angs
#         f.write(line + "\n")
# # === Loop over periods ===
# with open(ImpFile, 'w') as f:
#     line = '#   PERIOD,  Re Zxx,  Im Zxx,  Re Zxy,  Im Zxy ,  Re Zyx,  Im Zyx,  Re Zyy,  Im Zyy'
#     f.write(line + "\n")
#     for per in Periods:
#         z = z1anis(nl, h[:nl], al[:nl], at[:nl],
#                    blt[:nl], np.array([per]))[:, :, 0]
#         z_flat = z.flatten()
#         interlaced = [f"{tmp.real:.5e} {tmp.imag:.5e}" for tmp in z_flat]
#         line = f"{per:.5e} " + "  ".join(interlaced)
#         f.write(line + "\n")

# with open(RhoFile, 'w') as f:
#     line = '#   PERIOD,  Rhoa xx,  Phs xx,  Rhoa xy,  Phs xy,  Rhoa yx,  Phs yx,  Rhoa yy,  Phs yy'
#     f.write(line + "\n")
#     for per in Periods:
#         z = z1anis(nl, h[:nl], al[:nl], at[:nl],
#                    blt[:nl], np.array([per]))[:, :, 0]
#         omega = 2.0 * pi / per
#         prev = 1.0 / (omega * mu0)
#         rapp = np.abs(z)**2 * prev
#         papp = np.array(
#             [[180.0 * dphase(z[ii, jj]) / pi for jj in range(2)] for ii in range(2)])
#         line = [f"{per:.5e}"]
#         for i in range(2):
#             for j in range(2):
#                 line.append(f"{rapp[i, j]:.5e}")
#                 line.append(f"{papp[i, j]:.1f}")
#         f.write("  ".join(line) + "\n")

