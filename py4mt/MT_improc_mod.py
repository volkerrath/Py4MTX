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
Reads ModEM model, does fancy improc things.

@author: vr july 2020

Created on Tue Jul  7 16:59:01 2020

@author: vrath

"""

# Import required modules

import os
import sys
from sys import exit as error
# import struct
import time


import numpy as np
import math as ma
import netCDF4 as nc

from sys import exit as error
from modules.modem import *

#import readJac, writeJacNC, readDat, writeDatNC, sparsifyJac, readMod, rsvd
rhoair = 1.e+17

total = 0
ModFile_in = r'/home/vrath/work/MT/Annecy/ImageProc/In/ANN20_02_PT_NLCG_016'
ModFile_out = r'/home/vrath/work/MT/Annecy/ImageProc/Out/ANN20_02_PT_NLCG_016_ImProc'

action = 'anidiff'

if action == 'medfilt':
    ksize = 3
    bmode = 'nearest'  # 'reflect'
    maxit = 3
elif action == 'anidiff':
    maxit = 50
    fopt = 1


start = time.time()
dx, dy, dz, rho, reference = readMod(ModFile_in + '.rho', out=True)
writeMod(ModFile_out + '.rho', dx, dy, dz, rho, reference, out=True)
elapsed = (time.time() - start)
total = total + elapsed
print(' Used %7.4f s for reading model from %s ' %
      (elapsed, ModFile_in + '.rho'))

air = rho > rhoair / 100.
# prepare extended area of filter action (air)
rho = prepare_mod(rho, rhoair=rhoair)

start = time.time()
if action == 'medfilt':
    rhonew = medfilt3D(rho, kernel_size=ksize,
                       boundary_mode=bmode, maxiter=maxit)
    rhonew[air] = rhoair
    Modout = ModFile_out + '_mediankernel' + \
        str(kersiz) + '_' + str(maxit) + '.rho'
    writeMod(Modout, dx, dy, dz, rhonew, reference, out=True)
    elapsed = (time.time() - start)
    print(' Used %7.4f s for processing/writing model to %s ' %
          (elapsed, Modout))

elif action == 'anidiff':
    rhonew = anidiff3D(rho, ckappa=20, dgamma=0.24,
                       foption=fopt, maxiter=maxit, Out=True, Plot=True)
    rhonew[air] = rhoair
    Modout = ModFile_out + '_anisodiff' + str(fopt) + '-' + str(maxit) + '.rho'
    writeMod(Modout, dx, dy, dz, rhonew, reference, out=True)
    elapsed = (time.time() - start)
    print(' Used %7.4f s for processing/writing model to %s ' %
          (elapsed, Modout))


total = total + elapsed
print(' Total time used:  %f s ' % (total))
