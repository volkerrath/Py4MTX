#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reads ModEM'smodel files, does cellwise stats on them.

@author: vr +sb Nov 2024

"""
import os
import sys
from sys import exit as error
import time
from datetime import datetime
import warnings

import numpy as np
import netCDF4 as nc
import scipy.ndimage as spn
import scipy.linalg as spl

#import vtk
#import pyvista as pv
#import PVGeo as pvg


PY4MT_ROOT = os.environ["PY4MT_ROOT"]
mypath = [PY4MT_ROOT+"/py4mt/modules/", PY4MT_ROOT+"/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)


import modem as mod
import util as utl
from version import versionstrg


rng = np.random.default_rng()
blank = 1.e-30 # np.nan
rhoair = 1.e17


version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")

PY4MT_DATA =os.environ["PY4MT_DATA"]

# PY4MT_DATA = fullpath you like..

blank =


Models = [
    PY4MT_DATA+
    ]

ModFileAvg  = PY4MT_DATA+"AvgFile.rho"
ModFileVar  = PY4MT_DATA+"VarFile.rho"

#total = 0
#start = time.time()

imod = -1
for f in Models:
    imod += 1
    dx, dy, dz, rho, ref, trans = mod.read_mod(file=f, modext=".rho", trans="LOGE", blank=1.e-30, out=True):
    dims = np.shape(rho)
    aircells = np.where(rho>rhoair)

    print(ModFile_in + ".rho"+", shape is",dims)
    rtmp = rho.ravel()
    print(np.shape(rtmp))
    if imod==0:

        ModEns = rtmp
    else:
        ModEns = np.vstack((ModEns, rtmp))


ModAvg = np.mean(ModEns, axis=1).reshape(dims)
ModVar = np.var(ModEns, axis =1).reshape(dims)


if "mod" in OutFormat.lower():
    # for modem_readable files

    mod.write_mod(ModFileAvg, modext="_avg.rho",
                    dx=dx, dy=dy, dz=dz, mval=ModAvg,
                    reference=ref, mvalair=blank, aircells=aircells, header="Model log-average"
    print("Averages (ModEM format) written to "+ModFileAvg)

    mod.write_mod(ModFileVar, modext="_var.rho",
                    dx=dx, dy=dy, dz=dz, mval=np.sqrt(ModVar),
                    reference=ref, mvalair=blank, aircells=aircells, header="Model log-std"
    print("Variances (ModEM format) written to "+ModFileAvg)

#if "ubc" in OutFormat.lower():
    #elev = -ref[2]
    #refubc =  [MOrig[0], MOrig[1], elev]
    #mod.write_ubc(ModFileAvg, modext="_ubc.avg", mshext="_ubc.msh",
                    #dx=dx, dy=dy, dz=dz, mval=vol, reference=refubc, mvalair=Blank, aircells=aircells, header="Model log-average")
    #print(" Cell volumes (UBC format) written to "+VolFile)

if "rlm" in OutFormat.lower():
    mod.write_rlm(ModFileAvg, modext="_avg.rlm",
                    dx=dx, dy=dy, dz=dz, mval=ModAvg, reference=ref, mvalair=blank, aircells=aircells, comment="Model log-average")
    print(" Averages (CGG format) written to "+ModFileAvg)
    mod.write_rlm(ModFileAvg, modext="_avg.rlm",
                    dx=dx, dy=dy, dz=dz, mval=np.sqrt(ModVar), reference=ref, mvalair=blank, aircells=aircells, comment="Model log-std")
    print(" Variances (CGG format) written to "+ModFileAvg)


#ModVar = np.zeros()
# write_model(ModFile_out+'.rho', dx, dy, dz, rho,reference,out = True)
