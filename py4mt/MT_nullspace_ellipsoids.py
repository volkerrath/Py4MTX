#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
# ---

"""
Reads ModEM model, reads ModEM's Jacobian, does fancy things.

Created on Sun Jan 17 15:09:34 2021

@author: vrath jan 2021

"""

# Import required modules

import os
import sys
from sys import exit as error
# import struct
import time
from datetime import datetime


import numpy as np
import netCDF4 as nc

mypath = ["/home/vrath/Py4MT/py4mt/modules/", "/home/vrath/Py4MT/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)


import jacproc as jac
import modem as mod



import numpy as np
import math as ma
import netCDF4 as nc

import modem as mod
import util as utl

from version import versionstrg

Strng, _ = versionstrg()
now = datetime.now()
print("\n\n"+Strng)
print("Generate Random sets of bodies"+"\n"+"".join("Date " + now.strftime("%m/%d/%Y, %H:%M:%S")))
print("\n\n")




rng = np.random.default_rng()
nan = np.nan  # float('NaN')


rhoair = 1.e+17

total = 0
ModFile_in = r'/home/vrath/work/MT/Annecy/ImageProc/In/ANN20_02_PT_NLCG_016'
ModFile_out = r'/home/vrath/work/MT/Annecy/ImageProc/Out/ANN20_02_PT_NLCG_016_nse'

geocenter = [45.938251, 6.084900]
utm_x, utm_y = utl.proj_latlon_to_utm(geocenter[0], geocenter[1], utm_zone=32631)
utmcenter = [utm_x, utm_y, 0.]

ssamples = 10000


body = [
    'ellipsoid',
    'add',
    0.,
    0.,
    0.,
    3000.,
    1000.,
    2000.,
    1000.,
    0.,
    0.,
    30.]

normalize_err = True
normalize_max = True
calcsens = True

JacFile = r'/home/vrath/work/MT/Jacobians/Maurienne/Maur_PT.jac'
DatFile = r'/home/vrath/work/MT/Jacobians/Maurienne/Maur_PT.dat'
ModFile = r'/home/vrath/work/MT/Jacobians/Maurienne//Maur_PT_R500_NLCG_016.rho'
SnsFile = r'/home/vrath/work/MT/Jacobians/Maurienne/Maur_PT_R500_NLCG_016.sns'

total = 0.


start = time.time()
dx, dy, dz, rho, reference = mod.read_model(ModFile)
elapsed = (time.time() - start)
total = total + elapsed
print(' Used %7.4f s for reading model from %s ' % (elapsed, DatFile))


nb = np.shape(body)

# smoother=['gaussian',0.5]
smoother = ['uniform', 3]
total = 0
start = time.time()

dx, dy, dz, rho, reference = mod.read_model(ModFile_in + '.rho', out=True)
# writeMod(ModFile_out+'.rho', dx, dy, dz, rho,reference,out = True)
elapsed = (time.time() - start)
total = total + elapsed
print(' Used %7.4f s for reading model from %s ' %
      (elapsed, ModFile_in + '.rho'))

air = rho > rhoair / 100.

rho = mod.prepare_model(rho, rhoair=rhoair)

for ibody in range(nb[0]):
    body = bodies[ibody]
    rhonew = insert_body(dx, dy, dz, rho, body, smooth=smoother)
    rhonew[air] = rhoair
    Modout = ModFile_out + '_' + body[0] + \
        str(ibody) + '_' + smoother[0] + '.rho'
    writeMod(Modout, dx, dy, dz, rhonew, reference, out=True)

    elapsed = (time.time() - start)
    print(
        ' Used %7.4f s for processing/writing model to %s ' %
        (elapsed, Modout))
    print('\n')


total = total + elapsed
print(' Total time used:  %f s ' % (total))
