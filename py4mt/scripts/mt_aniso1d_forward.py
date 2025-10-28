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

from aniso import cpanis, z1anis
from mtproc import calc_rhoa_phas
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

WorkDir = '/home/vrath/Py4MTX/aniso/'
if not os.path.isdir(WorkDir):
    print(' File: %s does not exist, but will be created' % WorkDir)
    os.mkdir(WorkDir)

SumOut = True
ResFile = WorkDir + 'summary.dat'

ImpOut = True
ImpFile = WorkDir + 'impedance.dat'
ImpPlt = True

RhoOut = True
RhoFile = WorkDir + 'rhophas.dat'  
RhoPlt = True

PlotFormat = ['.png']
PlotFile = WorkDir+'AnisoTest'

if RhoPlt or ImpPlt:
    pltargs = {
        'pltsize': [16., 16.],
        'fontsizes': [18, 20, 24, 18],  # axis, label,title
        'm_size': 10,
        'c_obs': ['b', 'r'],
        'm_obs': ['s', 'o'],
        'l_obs': ['-', 2],
        'c_cal': ['b', 'r'],
        'm_cal': ['.', '.'],
        'l_cal': ['-', 2],
        'nrms': [],
        'xlimits': [],  # [1e-3, 1e3],
        'ylimits': [],
        'suptitle': 'Anisotropic model Test',
    }

Periods = np.logspace(-3, 5., 41)
print('periods:', Periods)

'''
Testmodel
'''
NLayer = 4
# Model = [
#     [2.,  100.,  100.,  100.,    0.,  0.,  0., 1.],
#     [6.,  200.,  20.,    200.,   15.,  0.,  0., 1],
#     [6.,  1000.,   2000.,   10.,  -75.,  0.,  0., 1],
#     [0.,     100.,    100.,    100.,    0.,  0.,  0., 1]
# ]
NLayer = 4
Model = [
    [2.,  100.,  100.,  100.,    0.,  0.,  0., 1.],
    [6.,  100.,  10.,    10.,   15.,  0.,  0., 1],
    [6.,  100.,   1000., 10.,  -75.,  0.,  0., 1],
    [0.,     100.,    100.,    100.,    0.,  0.,  0., 1]
]
# '''
# Model A from Pek, J. and Santos, F. A. M., 2002.
# '''
# NLayer = 4
# Model = [
#     [10.,  10000.,  10000.,  10000.,    0.,  0.,  0., 1.],
#     [18.,    200.,  20000.,    200.,   15.,  0.,  0., 1],
#     [100.,  1000.,   2000.,   1000.,  -75.,  0.,  0., 1],
#     [0.,     100.,    100.,    100.,    0.,  0.,  0., 1]
#         ]

# '''
# Model B from Pek, J. and Santos, F. A. M., 2002.
# '''
# NLayer = 2
# Model = [
#     [10.,   1000.,   1000.,   1000.,    0.,  0.,  0., 1],
#     [0.,     10.,    300.,    100.,   15., 60., 30., 1]
# ]

model = np.array(Model)

# Allocate arrays

h = model[:, 0]
rop = model[:, 1:4]
ustr = model[:, 4]
udip = model[:, 5]
usla = model[:, 6]


# === Compute conductivity tensors and effective parameters ===
sg, al, at, blt = cpanis(rop[:NLayer], ustr[:NLayer],
                         udip[:NLayer], usla[:NLayer])

# === Write model summary ===
with open(ResFile, 'w') as f:
    line = '# Model parameters'
    f.write(line + '\n')
    line = '# layer, thick (km)  res_x, res_y, res_z, strike, dip, slant'
    f.write(line + '\n')
    for layer in range(NLayer):
        pars = f'{layer:5d} {h[layer]:12.4f}'
        rops = f'   {rop[layer,0]:12.4f} {rop[layer,1]:12.4f} {rop[layer,2]:12.4f} '
        angs = f'   {ustr[layer]:12.4f} {udip[layer]:12.4f} {usla[layer]:12.0f} '
        # for i in range(3):
        #     for j in range(3):
        #         pars = pars +f'  {sg[layer, i, j]:14.5f}'
        line = pars+rops+angs
        f.write(line + '\n')

# === Loop over periods for impedances ===
Z = []
for per in Periods:
    z = z1anis(NLayer, h[:NLayer], al[:NLayer], at[:NLayer],
               blt[:NLayer], np.array([per]))[:, :, 0]
    z_flat = z.flatten()
    Z.append(z_flat)
Z = np.array(Z)

# interlace Z as Re Im
shp = np.shape(Z)
interlaced = np.empty((shp[0], 2*shp[1]))
interlaced[:, 0::2] = Z.real
interlaced[:, 1::2] = Z.imag


if ImpOut:
    # interlace Z as Re Im
    Imp = interlaced

    with open(ImpFile, 'w') as f:
        line = '#   PERIOD,  Re Zxx,  Im Zxx,  Re Zxy,  Im Zxy ,  Re Zyx,  Im Zyx,  Re Zyy,  Im Zyy'
        f.write(line + '\n')
        for iper in np.arange(len(Periods)):
            real_imag = '  '.join(
                [f'{tmp.real:.5e} {tmp.imag:.5e}' for tmp in Imp[iper, :]])
            line = f'{per:14.5f} ' + real_imag
            f.write(line + '\n')

if ImpPlt:

    Imp = interlaced

    fig, ax = plt.subplots(2, 2, figsize=pltargs['pltsize'])
    fig.suptitle(pltargs['suptitle'], fontsize=pltargs['fontsizes'][2])

    pltargs['yscale'] = 'linear'
    pltargs['ylabel'] = r'impedance [$\Omega$]'
    pltargs['legend'] = ['real', 'imag']

    data = np.zeros((len(Periods), 3))
    data[:, 0] = Periods[:]

    data[:, 1] = Imp[:, 0]
    data[:, 2] = Imp[:, 1]
    pltargs['title'] = 'Zxx'
    viz.plot_impedance(thisaxis=ax[0, 0], data=data, **pltargs)

    data[:, 1] = Imp[:, 2]
    data[:, 2] = Imp[:, 3]
    pltargs['title'] = 'Zxy'

    viz.plot_impedance(thisaxis=ax[0, 1], data=data, **pltargs)

    data[:, 1] = Imp[:, 4]
    data[:, 2] = Imp[:, 5]
    pltargs['title'] = 'Zyx'
    viz.plot_impedance(thisaxis=ax[1, 0], data=data, **pltargs)

    data[:, 1] = Imp[:, 6]
    data[:, 2] = Imp[:, 7]
    pltargs['title'] = 'Zyy'
    viz.plot_impedance(thisaxis=ax[1, 1], data=data, **pltargs)
    
    for f in PlotFormat:
            plt.savefig(PlotFile+'_imped'+f)

if RhoOut:
    freqs = (1./Periods).reshape(-1, 1)
    rhoa, phas = calc_rhoa_phas(freq=freqs, Z=Z)
    with open(RhoFile, 'w') as f:
        line = '#   PERIOD,  Rhoa xx,  Phs xx,  Rhoa xy,  Phs xy,  Rhoa yx,  Phs yx,  Rhoa yy,  Phs yy'
        f.write(line + '\n')
        for iper in np.arange(len(Periods)):
            per = Periods[iper]

            # rhoa = np.ravel([[fac*np.abs(tmp)**2 for tmp in Z[iper,0::2]]])
            # phas = np.ravel([[deg*dphase(tmp) for tmp in Z[iper,1::2]]])
            print('rhoa:', rhoa[iper, :])
            print('phas:', phas[iper, :])
            rhoa_phas = ''.join(
                [f'{float(rhoa[iper,ii]):14.5e} {float(phas[iper,ii]):12.2f}' for ii in range(4)])
            line = f'{per:14.5f} ' + rhoa_phas
            f.write(''.join(line) + '\n')

if RhoPlt:

    freqs = (1./Periods).reshape(-1, 1)
    rhoa, phas = calc_rhoa_phas(freq=freqs, Z=Z)

    fig, ax = plt.subplots(2, 2, figsize=pltargs['pltsize'])
    fig.suptitle(pltargs['suptitle'], fontsize=pltargs['fontsizes'][2])

    data = np.zeros((len(Periods), 3))
    data[:, 0] = Periods[:]

    data[:, 1] = rhoa[:, 1]
    data[:, 2] = rhoa[:, 2]
    pltargs['title'] = 'Rho xy/yx'
    pltargs['legend'] = [r'$\rho_{a, xy}$', r'$\rho_{a, xy}$']
    pltargs['yscale'] = 'log'
    pltargs['ylimits'] = [1.e1, 1.e3]
    pltargs['ylabel'] = r'$\rho_a$  [$\Omega$ m]'
    viz.plot_rhophas(thisaxis=ax[0, 0], data=data, **pltargs)

    data[:, 1] = phas[:, 1]+90.
    data[:, 2] = phas[:, 2]-90.
    pltargs['title'] = 'Phas xy/yx'
    pltargs['legend'] = [r'$\phi_{xy}$', r'$\phi_{yx}$']
    pltargs['yscale'] = 'linear'
    pltargs['ylimits'] = []
    pltargs['ylabel'] = r'$\phi$ [$^\circ$]'
    viz.plot_rhophas(thisaxis=ax[1, 0], data=data, **pltargs)

    data[:, 1] = rhoa[:, 0]
    data[:, 2] = rhoa[:, 3]
    pltargs['title'] = 'Rho xx/yy'
    pltargs['legend'] = [r'$\rho_{a, xx}$', r'$\rho_{a, yy}$']
    pltargs['yscale'] = 'log'
    pltargs['ylimits'] = [1.e-4, 1.e1]
    pltargs['ylabel'] = r'$\rho_a$  [$\Omega$ m]'
    viz.plot_rhophas(thisaxis=ax[0, 1], data=data, **pltargs)

    data[:, 1] = phas[:, 0]-90.
    data[:, 2] = phas[:, 3]+90.
    pltargs['title'] = 'Phas xx/yy'
    pltargs['legend'] = [r'$\phi_{xx}$', r'$\phi_{yy}$']
    pltargs['yscale'] = 'linear'
    pltargs['ylimits'] = []
    pltargs['ylabel'] = r'$\phi$ [$^\circ$]'
    viz.plot_rhophas(thisaxis=ax[1, 1], data=data, **pltargs)

    for f in PlotFormat:
            plt.savefig(PlotFile+'_rhophas'+f)
