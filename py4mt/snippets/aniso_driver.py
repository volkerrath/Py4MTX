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

# add py4mt modules to pythonpath
mypath = ['/home/vrath/Py4MT/py4mt/modules/',
          '/home/vrath/Py4MT/py4mt/scripts/']
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)
        
# from version import versionstrg
# import inverse as inv
import util as utl
from aniso import *

rhoair = 1.e17
rng = np.random.default_rng()
nan = np.nan  # float('NaN')
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())

titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng+'\n\n')


#def z1a_driver(model_file='an.dat', res_file='an.res', tab_file='anr.dat'):
model_file='an.dat' 
res_file='an_python.res' 
tab_file='anr_python.dat'
pi = np.pi
mu0 = 4e-7 * pi
nlmax = 1001

# Allocate arrays
h = np.zeros(nlmax)
rop = np.zeros((nlmax, 3))
ustr = np.zeros(nlmax)
udip = np.zeros(nlmax)
usla = np.zeros(nlmax)

# === Read model parameters ===
with open(model_file, 'r') as f:
    nl = int(f.readline())
    for layer in range(nl):
        line = f.readline().split()
        h[layer] = float(line[0])
        rop[layer, :] = list(map(float, line[1:4]))
        ustr[layer] = float(line[4])
        udip[layer] = float(line[5])
        usla[layer] = float(line[6])

# === Compute conductivity tensors and effective parameters ===
sg, al, at, blt = cpanis(rop[:nl], ustr[:nl], udip[:nl], usla[:nl])

# === Write model summary ===
with open(res_file, 'w') as f2:
    f2.write('=== Model parameters ====================\n\n')
    for layer in range(nl):
        f2.write(f'+++> Layer = {layer+1:5d}, thickness in km = {h[layer]:12.4f}\n')
        f2.write('--- Conductivity tensor, in S/m ---------\n')
        for i in range(3):
            f2.write('  ' + '  '.join(f'{sg[layer, i, j]:14.5f}' for j in range(3)) + '\n')
        f2.write('--- CONDMAX,CONDMIN,ASTRIKE for eq.mod. --\n')
        f2.write(f'     {al[layer]:14.5f}  {at[layer]:14.5f}  {180.0 * blt[layer] / pi:14.5f}\n\n')
    f2.write('=========================================\n')

# === Loop over periods ===
with open(tab_file, 'w') as f3:
    header= '#   PERIOD     Re Zxx     Im Zxx     Re Zxy     Im Zxy    Re Zyx     Im Zyx     Re Zyy     Im Zyy'
    Z = []
    periods = np.logspace(-4., 4., 41)
    for per in periods:
        z = z1anis(nl, h[:nl], al[:nl], at[:nl], blt[:nl], np.array([per]))[:, :, 0]
        tmp = np.array([per, z.flat])  #np.insert(z.flat, 0, per, axis=0)
        Z.append(tmp)
        
        
    np.savetxt(tab_file, Z, 
               fmt = '%1.5e', 
               delimiter='   ', 
               header=header)

    R = []
    for iper in range(-30, 51, 2):
        z = z1anis(nl, h[:nl], al[:nl], at[:nl], blt[:nl], np.array([per]))[:, :, 0]
        # Apparent resistivity and phase
        omega = 2.0 * pi / per
        prev = 1.0 / (omega * mu0)
        rapp = np.abs(z)**2 * prev
        papp = np.array([[180.0 * dphase(z[ii, jj]) / pi for jj in range(2)] for ii in range(2)])
        
        
        # f3.write('\n=== Period and impedance tensor =========\n')
        # f3.write('        PERIOD     Re Zxx     Im Zxx     Re Zxy     Im Zxy\n')
        # f3.write('                  Re Zyx     Im Zyx     Re Zyy     Im Zyy\n')
        # f3.write(f'{per:14.5f}  {z[0,0].real:12.4e}  {z[0,0].imag:12.4e}  {z[0,1].real:12.4e}  {z[0,1].imag:12.4e}\n')
        # f3.write(f'              {z[1,0].real:12.4e}  {z[1,0].imag:12.4e}  {z[1,1].real:12.4e}  {z[1,1].imag:12.4e}\n')



        # f3.write('--- Period, resistivities and phases ----\n')
        # f3.write('        PERIOD     RHOAxx     RHOAxy     RHOAyx     RHOAyy\n')
        # f3.write('                  PHIAxx     PHIAxy     PHIAyx     PHIAyy\n')
        # f3.write(f'{per:14.5f}  {rapp[0,0]:12.4e}  {rapp[0,1]:12.4e}  {rapp[1,0]:12.4e}  {rapp[1,1]:12.4e}\n')
        # f3.write(f'              {papp[0,0]:12.2f}  {papp[0,1]:12.2f}  {papp[1,0]:12.2f}  {papp[1,1]:12.2f}\n')

        # f3.write(f'{per:14.5f}  ')
