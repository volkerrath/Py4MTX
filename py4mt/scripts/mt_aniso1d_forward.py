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

WorkDir = '/home/vrath/Py4MTX/work/aniso/'
# WorkDir = '/home/vrath/Py4MTX/work/EPS_2025_Annecy/edi/Sites_Name/edi/'
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

PhTOut = True
PhTFile = WorkDir + 'rhophas.dat'
PhTPlt = True


PlotFormat = ['.png']
PlotFile = WorkDir+'AnisoTest'

if RhoPlt or ImpPlt or PhTPlt:
    pltargs = {
        'pltsize': [16., 16.],
        'fontsizes': [18, 20, 24, 18],  # axis, label, title
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
        'suptitle': 'Anisotropic model Rong et al. (2022)',
    }

'''
Testmodel
'''
# Periods = np.logspace(-3, 5., 41)
# print('periods:', Periods)
# Freqs = 1./Periods
# NLayer = 4
# Model = [
#     [2.,  100.,  100.,  100.,    0.,  0.,  0., 1.],
#     [6.,  200.,  20.,    200.,   15.,  0.,  0., 1],
#     [6.,  1000.,   2000.,   10.,  -75.,  0.,  0., 1],
#     [0.,     100.,    100.,    100.,    0.,  0.,  0., 1]
# ]


# NLayer = 4
# Model = {'h': np.array([2000,6000.,6000.,0]),
#          'rop': np.array([
#                         [ 100.,  100.,   100.],
#                         [ 100.,   10.,   100.],
#                         [ 100., 1000.,   100.],
#                         [ 100.,  100.,   100.]
#                         ]),
#          'ustr': np.array([0., 15., -75., 0.]),
#          'udip': np.array([0., 0., 0., 0.,]),
#          'usla' : np.array([0., 0., 0., 0.,]),
#          'is_iso': np.array([1, 1, 1, 0,])
#          }

# NLayer = 4
# Model = {'h': np.array([2000,6000.,6000.,0]),
#          'rop': np.array([
#                         [ 100.,  100.,   100.],
#                         [ 100.,   10.,   100.],
#                         [ 100., 1000.,   100.],
#                         [ 100.,  100.,   100.]
#                         ]),
#          'ustr': np.array([0., 15., -75., 0.]),
#          'udip': np.array([0., 0., 0., 0.,]),
#          'usla' : np.array([0., 0., 0., 0.,]),
#          'is_iso': np.array([1, 1, 1, 0,])
#          }

'''
Model from Rong et al, (2022), Fig 1.
'''
Periods = np.logspace(-2, 4., 41)
# print('periods:', Periods)
Freqs = 1./Periods
NLayer = 4
Model = [
    [10000.,   1000.,   1000.,   1000.,    0.,  0.,  0., 0],
    [18000.,    200.,   2000.,    200.,   15.,  0.,  0., 0],
    [100000.,  1000.,  10000.,   1000.,  -75.,  0.,  0., 0],
    [0.,     100.,    100.,    100.,    0.,  0.,  0., 0]
        ]


'''
Model A from Pek, J. and Santos, F. A. M., 2002.
'''
# Periods = np.logspace(-3, 5., 41)
# print('periods:', Periods)
# Freqs = 1./Periods
# NLayer = 4
# Model = [
#     [10000.,   10000.,   10000.,   10000.,    0.,  0.,  0., 0],
#     [18000.,    200.,   20000.,    200.,   15.,  0.,  0., 0],
#     [100000.,  1000.,  2000.,   1000.,  -75.,  0.,  0., 0],
#     [0.,     100.,    100.,    100.,    0.,  0.,  0., 0]
#         ]


'''
Model B from Pek, J. and Santos, F. A. M., 2002.
'''
# NLayer = 2
# Model = [
#     [10.,   1000.,   1000.,   1000.,    0.,  0.,  0., 0],
#     [0.,     10.,    300.,    100.,   15., 60., 30., 0]
# ]

model = np.array(Model)
periods = Periods.copy()

# to arrays

h = model[:, 0]
rop = model[:, 1:4]
ustr = model[:, 4]
udip = model[:, 5]
usla = model[:, 6]
is_iso = model[:, 7] == 1


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

res = aniso1d_impedance_sens(
                periods_s=Periods, # (nper,)
                h_m=h, # (nl,)
                rop=rop, # (nl,3)
                ustr_deg=ustr, # (nl,)
                udip_deg=udip, # (nl,)
                usla_deg=usla, # (nl,)
                compute_sens=False,
                )
Z = res['Z'] # (nper,2,2)
P, _ = compute_pt(Z)

Z = Z.reshape((np.shape(Z)[0],4))
P = P.reshape((np.shape(Z)[0],4))


# === Loop over periods for impedances ===
# Z = []
# for per in Periods:
#     print()
#     z, _, _, _, _ = mt1d_aniso(ani_flag[:NLayer], h[:NLayer], al[:NLayer], at[:NLayer],
#                blt[:NLayer], per)
#     z_flat = z.flatten()
#     Z.append(z_flat)
# Z = np.array(Z)

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
            line = f'{periods[iper]:14.5f} ' + real_imag
            f.write(line + '\n')




if ImpPlt:

    Imp = interlaced


    pltargs['pltsize'] = [16., 16.]
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
            per = periods[iper]
            # rhoa = np.ravel([[fac*np.abs(tmp)**2 for tmp in Z[iper,0::2]]])
            # phas = np.ravel([[deg*dphase(tmp) for tmp in Z[iper,1::2]]])
            # print('rhoa:', rhoa[iper, :])
            # print('phas:', phas[iper, :])
            rhoa_phas = ''.join(
                [f'{float(rhoa[iper,ii]):14.5e} {float(phas[iper,ii]):12.2f}' for ii in range(4)])
            line = f'{per:14.5f} ' + rhoa_phas
            f.write(''.join(line) + '\n')

if RhoPlt:

    freqs = (1./periods).reshape(-1, 1)
    rhoa, phas = calc_rhoa_phas(freq=freqs, Z=Z)

    pltargs['pltsize'] = [16., 16.]
    fig, ax = plt.subplots(2, 2, figsize=pltargs['pltsize'])
    fig.suptitle(pltargs['suptitle'], fontsize=pltargs['fontsizes'][2])

    data = np.zeros((len(Periods), 3))
    data[:, 0] = Periods[:]

    data[:, 1] = rhoa[:, 1]
    data[:, 2] = rhoa[:, 2]
    pltargs['title'] = 'Rho xy/yx'
    pltargs['legend'] = [r'$\rho_{a, xy}$', r'$\rho_{a, xy}$']
    pltargs['yscale'] = 'log'
    pltargs['ylimits'] = [] #[1.e1, 1.e3]
    pltargs['ylabel'] = r'$\rho_a$  [$\Omega$ m]'
    viz.plot_rhophas(thisaxis=ax[0, 0], data=data, **pltargs)

    data[:, 1] = phas[:, 1]+90.
    data[:, 2] = phas[:, 2]-90.
    pltargs['title'] = 'Phas xy/yx'
    pltargs['legend'] = [r'$\phi_{xy}$', r'$\phi_{yx}$']
    pltargs['yscale'] = 'linear'
    pltargs['ylimits'] = [-180., 180.]
    pltargs['ylabel'] = r'$\phi$ [$^\circ$]'
    viz.plot_rhophas(thisaxis=ax[1, 0], data=data, **pltargs)

    data[:, 1] = rhoa[:, 0]
    data[:, 2] = rhoa[:, 3]
    pltargs['title'] = 'Rho xx/yy'
    pltargs['legend'] = [r'$\rho_{a, xx}$', r'$\rho_{a, yy}$']
    pltargs['yscale'] = 'log'
    pltargs['ylimits'] = [] #[1.e-12, 1.e1]
    pltargs['ylabel'] = r'$\rho_a$  [$\Omega$ m]'
    viz.plot_rhophas(thisaxis=ax[0, 1], data=data, **pltargs)

    data[:, 1] = phas[:, 0]-90.
    data[:, 2] = phas[:, 3]+90.
    pltargs['title'] = 'Phas xx/yy'
    pltargs['legend'] = [r'$\phi_{xx}$', r'$\phi_{yy}$']
    pltargs['yscale'] = 'linear'
    pltargs['ylimits'] =  [-180., 180.] #[]
    pltargs['ylabel'] = r'$\phi$ [$^\circ$]'
    viz.plot_rhophas(thisaxis=ax[1, 1], data=data, **pltargs)

    for f in PlotFormat:
            plt.savefig(PlotFile+'_rhophas'+f)

if PhTOut:
    freqs = (1./Periods).reshape(-1, 1)
    rhoa, phas = calc_rhoa_phas(freq=freqs, Z=Z)
    with open(RhoFile, 'w') as f:
        line = '#   PERIOD,  Rhoa xx,  Phs xx,  Rhoa xy,  Phs xy,  Rhoa yx,  Phs yx,  Rhoa yy,  Phs yy'
        f.write(line + '\n')
        for iper in np.arange(len(Periods)):
            per = periods[iper]

            # rhoa = np.ravel([[fac*np.abs(tmp)**2 for tmp in Z[iper,0::2]]])
            # phas = np.ravel([[deg*dphase(tmp) for tmp in Z[iper,1::2]]])
            # print('phstens:', P[iper, :])
            phstens = ''.join(
                [f'{float(P[iper,ii]):14.5e} ' for ii in range(4)])
            line = f'{per:14.5f} ' + phstens
            f.write(''.join(line) + '\n')


if PhTPlt:

    freqs = (1./periods).reshape(-1, 1)

    pltargs['pltsize'] = [16., 10.]
    fig, ax = plt.subplots(1, 2, figsize=pltargs['pltsize'])
    fig.suptitle(pltargs['suptitle'], fontsize=pltargs['fontsizes'][2])

    data = np.zeros((len(Periods), 3))
    data[:, 0] = Periods[:]

    data[:, 1] = P[:, 1]
    data[:, 2] = P[:, 2]
    pltargs['title'] = r'Phase Tensor xy/yx'
    pltargs['legend'] = [r'$\Phi_{a, xy}$', r'$\Phi_{a, xy}$']
    pltargs['yscale'] = 'linear'
    pltargs['ylimits'] = [] #[1.e1, 1.e3]
    pltargs['ylabel'] = r'$\Phi$  [-]'
    viz.plot_phastens(thisaxis=ax[0], data=data, **pltargs)

    data[:, 1] = P[:, 0]
    data[:, 2] = P[:, 3]
    pltargs['title'] = r'Phase Tensor xx/yy'
    pltargs['legend'] = [r'$\Phi_{xx}$', r'$\Phi_{yy}$']
    pltargs['yscale'] = 'linear'
    pltargs['ylimits'] = []
    pltargs['ylabel'] = r'$\Phi$ [-]'
    viz.plot_phastens(thisaxis=ax[1], data=data, **pltargs)




    for f in PlotFormat:
            plt.savefig(PlotFile+'_phstens'+f)

    # df= pd.DataFrame.from_dict({item: npz[item] for item in npz.files}, orient='index')

    # if Plot:
    #     fig, axs = plt.subplots(3, 2, figsize=(8, 14), sharex=True)
    #     add_rho(df, comps="xy,yx", ax=axs[0, 0])
    #     add_phase(df, comps="xy,yx", ax=axs[0, 1])
    #     add_rho(df, comps="xx,yy", ax=axs[1, 0])
    #     add_phase(df, comps="xx,yy", ax=axs[1, 1])
    #     add_pt(df, ax=axs[2, 1])
    #     fig.suptitle(station)

    #     # Remove empty axes
    #     for ax in axs:
    #         if not ax.lines and not ax.images and not ax.collections:
    #             fig.delaxes(ax)

    #     for f in PlotFormat:
    #         plt.savefig(WorkDir + station + String_out + f, dpi=600)

    #     plt.show()
