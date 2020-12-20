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
# import struct
import time

import numpy as np
import math  as ma
import netCDF4 as nc

from scipy.interpolate import interp2d
from scipy.interpolate import Rbf
import pyresample as prs
import matplotlib.pyplot as plt




# import mtpy.modeling.modem as modem
# from modules.ModEM import readDat, writeDatNC
from modules.ModEM import * 

DatFile = r'./work/MauTopoDense.dat'


Site, Comp, Data = readDat(DatFile)

start = time.time()
name, ext = os.path.splitext(DatFile)
NCFile = name+'_dat.nc'
writeDatNC(NCFile, Data, Site, Comp) 
elapsed = (time.time()- start)
print (' Used %7.4f s for writing data to %s ' % (elapsed,NCFile))


print('Available Periods:')
print(np.unique(np.sort(Data[:,0])))

print('Available Components:')
print(np.unique(np.sort(Comp)))

print('Available Sites:')
print(np.unique(np.sort(Site)))

Periods = [1.05400e-03, 1.11030e-02, 1.16999e-01 ]
Components = ['ZXXI', 'ZXXR', 'ZXYI', 'ZXYR', 'ZYXI', 'ZYXR', 'ZYYI','ZYYR' ]
pp = np.log10(Data[:,0])
for p in np.log10(Periods):
    i1 = np.isclose(pp, p, rtol=1e-03, atol=1e-12, equal_nan=False)
    d1 = Data[i1,:]
    c1 = Comp[i1]
    for c in Components:
        i2 = (c1 == c)
        tmp=d1[i2,:]
        print(np.shape(tmp))
        lat = tmp[:,1]
        lon = tmp[:,2]
        val = tmp[:,6]
        f = interp2d(lon, lat, val, kind="cubic", bounds_error=False)
