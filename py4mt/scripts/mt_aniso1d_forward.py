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

WorkDir = '/home/vrath/Py4MTX/aniso/'

SumOut = True
ResFile = WorkDir+ 'summary.dat'

ImpOut = True
ImpFile = WorkDir+ 'impedance.dat' #ModFile.replace('.dat', '_impedance.dat')
ImpPlt = True

RhoOut = True
RhoFile = WorkDir+ 'rhophas.dat' #ModFile.replace('.dat', '_rhophas.dat')
RhoPlt = True


if RhoPlt or ImpPlt:      
    pltargs = {
    'pltsize' : [16.,16.],
    'fontsizes' : [18, 20, 24], # axis, label,title
    'm_size' : 8,
    'c_obs' : ['b', 'r'],
    'm_obs' : ['s', 'o' ],
    'l_obs' : 2,
    'c_cal' : ['b', 'r'],
    'm_cal' : ['.', '.' ],
    'l_cal' : 2,
    'nrms'  : [],
    'perlimits' : [], # [1e-3, 1e3],
    'zlimits' : [],
    'title' : 'Anisotropic model A',
    'pltformat' : '.pdf',
    'pltfile' : 'modelA',
    'yscale' : 'linear'
    }

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
    line = '# layer, thick (km)  res_x, res_y, res_z, strike, dip, slant' 
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
interlaced = np.empty((shp[0],2*shp[1]))
interlaced[:, 0::2] = Z.real
interlaced[:, 1::2] = Z.imag
Z = interlaced

if ImpOut:    
    with open(ImpFile, 'w') as f:
        line = '#   PERIOD,  Re Zxx,  Im Zxx,  Re Zxy,  Im Zxy ,  Re Zyx,  Im Zyx,  Re Zyy,  Im Zyy'
        f.write(line + "\n")
        for iper in np.arange(len(Periods)):
            interlaced = "  ".join([f"{tmp.real:.5e} {tmp.imag:.5e}" for tmp in Z[iper,:]])
            line = f"{per:.5e} " + interlaced
            f.write(line + "\n")


fig, ax =  plt.subplots(2,2, figsize=pltargs['pltsize'])
fig.suptitle(pltargs['title'], fontsize=pltargs['fontsizes'][2])


data = np.zeros((len(Periods),3))     
data[:,0] = Periods[:]


data[:,1] = Z[:,0]
data[:,2] = Z[:,1]
pltargs['title']='Zxx'
viz.plot_impedance(thisaxis=ax[0,0], data=data, **pltargs)



data[:,1] = Z[:,2]
data[:,2] = Z[:,3]
pltargs['title']='Zxy'
viz.plot_impedance(thisaxis=ax[0,1], data=data, **pltargs)


data[:,1] = Z[:,4]
data[:,2] = Z[:,5]
pltargs['title']='Zyx'
viz.plot_impedance(thisaxis=ax[1,0], data=data, **pltargs)

data[:,1] = Z[:,6]
data[:,2] = Z[:,7]
pltargs['title']='Zyy'
viz.plot_impedance(thisaxis=ax[1,1], data=data, **pltargs)

# with open(RhoFile, 'w') as f:
#     line = '#   PERIOD,  Rhoa xx,  Phs xx,  Rhoa xy,  Phs xy,  Rhoa yx,  Phs yx,  Rhoa yy,  Phs yy'
#     f.write(line + "\n")
#     for per in Periods:
#         z = z1anis(NLayer, h[:NLayer], al[:NLayer], at[:NLayer],
#                    blt[:NLayer], np.array([per]))[:, :, 0]
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

