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
#       jupytext_version: 1.10.2
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
import math as ma
import netCDF4 as nc

# from modules.jacproc import *
from modules.modem import *
#import readJac, writeJacNC, readDat, writeDatNC, sparsifyJac, readMod, rsvd


ModFile_in = r'/home/vrath/work/MT/Annecy/ImageProc/In'
SnsFile_in = r'/home/vrath/work/MT/Annecy/ImageProc/Out/'


rhomt = model.flat[0 * nlyr:1 * nlyr]
delmt = model.flat[6 * nlyr:7 * nlyr - 1]

mt1dfwd(freq, sig, d, inmod='r', out="imp"):

Fileout = file.replace(InStrng, OutStrng) + '_MTdata.npz'
np.savez_compressed(OutDatDir + Fileout,
                    header=Header,
                    Nlyr=Nlyr,
                    mt_num=site_num,
                    mt_model=site_model,
                    mt_error=site_error,
                    mt_data=site_data,
                    mt_y=site_y, mt_x=site_x, mt_z=site_dem)

start = time.time()
dx, dy, dz, rho, reference = readMod(ModFile)
elapsed = (time.time() - start)
total = total + elapsed
print(' Used %7.4f s for reading model from %s ' % (elapsed, DatFile))


start = time.time()
elapsed = (time.time() - start)
total = total + elapsed
print(' Used %7.4f s for reading model from %s ' % (elapsed, DatFile))


start = time.time()
elapsed = (time.time() - start)
total = total + elapsed
print(' Used %7.4f s for reading model from %s ' % (elapsed, DatFile))


total = total + elapsed
print(' Total time used:  %f s ' % (total))
