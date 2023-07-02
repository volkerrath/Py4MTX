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
#       jupytext_version: 1.11.3
# ---

"""
Reads ModEM model and covariance files, fix border (padding zones).

@author: vr jun 2023


"""

import os
import sys
from sys import exit as error
import time
from datetime import datetime
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import scipy.ndimage as spn
import scipy.linalg as spl
PY4MT_ROOT = os.environ["PY4MT_ROOT"]
mypath = [PY4MT_ROOT+"/py4mt/modules/", PY4MT_ROOT+"/py4mt/scripts/"]

for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)

import modem as mod
import util as  utl
from version import versionstrg


rng = np.random.default_rng()
nan = np.nan  # float("NaN")
version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")

PY4MT_DATA = os.environ["PY4MT_DATA"]


"""
0           treated analogous to air/ocean, i.e., no smoothing at boundaries
2           with smoothing at boundaries
"""
fixed_zone = "2"
# fixed_zone = "0"

"""
Border 
"""
method = "border"
border = 5
"""
Distance
"""
method = "distance"
distance = 25000.


# ModFile_in = PY4MT_DATA +"/test/test.rho"
# CovFile_in = PY4MT_DATA +"/test/test.cov"
# CovFile_out = PY4MT_DATA +"/test/test_fix"+str(border)+".cov"
ModFile_in = PY4MT_DATA +"/Peru/Tacna/TAC8/TAC_300.rho"
CovFile_in = PY4MT_DATA +"/Peru/Tacna/TAC8/TAC6_04.cov"
DatFile_in = PY4MT_DATA +"/Peru/Tacna/TAC8/TAC6_Z.dat"

if "bord" in method.lower():
    CovFile_out = PY4MT_DATA+"/Peru/Tacna/TAC8/TAC_04_border"+str(border)+"_fixed"+str(fixed_zone)+".cov"
else:
    CovFile_out = PY4MT_DATA+"/Peru/Tacna/TAC8/TAC_04_mindist"+str(distance/1000)+"km_fixed"+str(fixed_zone)+".cov"

start = time.time()

dx, dy, dz, rho, reference, _ = mod.read_model(ModFile_in, out=True)
# write_model(ModFile_out+'.rho', dx, dy, dz, rho,reference,out = True)
elapsed = time.time() - start
print("Used %7.4f s for reading model from %s " % (elapsed, ModFile_in))
modsize = np.shape(rho)

start = time.time()

if "dist" in method.lower():
    x = np.append(0., np.cumsum(dx)) + reference[0]
    xc =0.5*(x[0:len(x)-1]+x[1:len(x)]) 
    y = np.append(0., np.cumsum(dy)) + reference[1]
    yc =0.5*(y[0:len(y)-1]+y[1:len(y)]) 
    cellcent = [xc, yc]
    
    print(len(xc),len(yc))
    Site , _, Data, _ = mod.read_data(DatFile_in, out=True)
    
    xs = []
    ys = []
    for idt in range(0, np.size(Site)):            
        ss = Site[idt]
        if idt == 0:
            site = Site[idt]
            xs.append(Data[idt,3])
            ys.append(Data[idt,4])
        elif ss != site:
            site = Site[idt]
            xs.append(Data[idt,3])
            ys.append(Data[idt,4])
            
    sitepos = [xs, ys]
    
    lines_out = mod.proc_covar(CovFile_in, 
                               CovFile_out, 
                               method=method, fixdist=distance, 
                               cellcent=cellcent, sitepos=sitepos, 
                               out=True)

if "bord" in method.lower():

    lines_out = mod.proc_covar(CovFile_in, 
                               CovFile_out, 
                               method = method, fixed=fixed_zone, border=border, 
                               out=True)

elapsed = time.time() - start
print(" Used %7.4f for reading/processing/writing covar:"  % (elapsed))
