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
#       jupytext_version: 1.8.0
# ---

"""
Reads ModEM's Jacobian, does fancy things.

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
import math  as ma
import netCDF4 as nc

from scipy.ndimage import \
    gaussian_filter, laplace, convolve, gaussian_gradient_magnitude,median_filter
from scipy.linalg  import norm
from sys import exit as error

from modules.modem import *
from modules.util import *
#import readJac, writeJacNC, readDat, writeDatNC, sparsifyJac, readMod, rsvd
rhoair = 1.e+17


ModFile_in  = r'/home/vrath/work/MT/Annecy/ImageProc/In/ANN20_02_PT_NLCG_016'
ModFile_out = r'/home/vrath/work/MT/Annecy/ImageProc/Out/ANN20_02_PT_NLCG_016_insbody'

bodies = [['ell', 0., 0., 10.,  4., 2., 3.,45., 45., 30.]]
nb     = np.shape(bodies)

smooth    = True   #'reflect'

total = 0
start = time.time()

dx, dy, dz, rho, reference = readMod(ModFile_in+'.rho',out = True)
# writeMod(ModFile_out+'.rho', dx, dy, dz, rho,reference,out = True)
elapsed = (time.time() - start)
total = total + elapsed
print (' Used %7.4f s for reading model from %s ' % (elapsed,ModFile_in+'.rho'))

air = rho > rhoair/100.

start = time.time()

for ibody in range(nb[0]):
    rhonew = insert_body(rho,bodies)
    Modout =ModFile_out+'.rho'
    writeMod(Modout, dx, dy, dz, rhonew,reference,out = True)

    if bodies[ibody,0][0:3] == 'ell':
        rhonew = insert_ellipsoid(rho, size=kernel_size, mode = bmode)
        rhonew[air] = rhoair
        Modout =ModFile_out+'_mediankernel'+str(kersiz)+'_'+bmode+'.rho'
        writeMod(Modout, dx, dy, dz, rhonew,reference,out = True)
        elapsed = (time.time() - start)
        print (' Used %7.4f s for processing/writing model to %s ' % (elapsed,Modout))

    elif bodies[ibody,0][0:3]  == 'box':
        rhonew = in_box(rho, size=kernel_size, mode = bmode)
        rhonew[air] = rhoair
        Modout =ModFile_out+'_shockfilt'+str(maxit)+'.rho'
        writeMod(Modout, dx, dy, dz, rhonew,reference,out = True)
        elapsed = (time.time() - start)
        print (' Used %7.4f s for processing/writing model to %s ' % (elapsed,Modout))


total = total + elapsed
print (' Total time used:  %f s ' % (total))
