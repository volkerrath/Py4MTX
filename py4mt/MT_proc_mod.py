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
#       jupytext_version: 1.7.1
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
import scipy.ndimage as scim
import scipy.signal  as scsi
# from modules.jacproc import *
from modules.modem import *
#import readJac, writeJacNC, readDat, writeDatNC, sparsifyJac, readMod, rsvd
total = 0
rhoair = 1.e+17

ModFile_in  = r'/home/vrath/work/MT/Annecy/ImageProc/In/ANN20_02_PTZ_NLCG_016'
ModFile_out = r'/home/vrath/work/MT/Annecy/ImageProc/Out/ANN20_02_PTZ_NLCG_016_ImProc'




start = time.time()
dx, dy, dz, rho, reference = readMod(ModFile_in+'.rho',out = True)
writeMod(ModFile_out+'.rho', dx, dy, dz, rho,reference,out = True)
elapsed = (time.time() - start)
total = total + elapsed
print (' Used %7.4f s for reading model from %s ' % (elapsed,ModFile_in+'.rho'))

air = rho > rhoair/100.

start = time.time()
for kersiz in [3, 5, 7]:
    rhonew = scsi.medfilt(rho, kernel_size=kersiz)
    rhonew[air] = rhoair
    # print(np.shape(rhonew))
    writeMod(ModFile_out+'_medker'+str(kersiz)+'.rho', dx, dy, dz, rhonew,reference,out = True)
    #rhonew = scim.median_filter(rho, size=kersize)
elapsed = (time.time() - start)
total = total + elapsed
print (' Used %7.4f s for processing/writing model to %s ' % (elapsed,ModFile_out))


total = total + elapsed
print (' Total time used:  %f s ' % (total))

