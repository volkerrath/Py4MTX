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
#       jupytext_version: 1.11.2
# ---

"""
Reads ModEM model, does fancy improc things.

@author: vr july 2020

@author: vrath  Feb 2021

"""

# Import required modules

import os
import sys
from sys import exit as error
import time
from datetime import datetime
import warnings

import numpy as np
import math as ma
import netCDF4 as nc


import vtk
import pyvista as pv
import PVGeo as pvg

mypath = ["/home/vrath/Py4MT/py4mt/modules/", "/home/vrath/Py4MT/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)

import modem as mod
from version import versionstrg

Strng, _ = versionstrg()
now = datetime.now()
print("\n\n"+Strng)
print("Image processing on model"+"\n"+"".join("Date " + now.strftime("%m/%d/%Y, %H:%M:%S")))
print("\n\n")


warnings.simplefilter(action="ignore", category=FutureWarning)

rhoair = 1.e17

total = 0
ModFile_in = r"/home/vrath/work/MT/Annecy/ImageProc/In/ANN20_02_PT_NLCG_016"
ModFile_out = r"/home/vrath/work/MT/Annecy/ImageProc/Out/ANN20_02_PT_NLCG_016_ImProc"

action = "anidiff"

if action == "medfilt":
    ksize = 3
    bmode = "nearest"  # 'reflect'
    maxit = 3
elif action == "anidiff":
    maxit = 50
    fopt = 1


start = time.time()
dx, dy, dz, rho, reference = mod.readMod(ModFile_in + ".rho", out=True)
mod.writeMod(ModFile_out + ".rho", dx, dy, dz, rho, reference, out=True)
elapsed = time.time() - start
total = total + elapsed
print(" Used %7.4f s for reading model from %s "
      % (elapsed, ModFile_in + ".rho"))

air = rho > rhoair / 100.
# prepare extended area of filter action (air)
rho = mod.prepare_mod(rho, rhoair=rhoair)

start = time.time()
if action == "medfilt":
    rhonew = mod.medfilt3D(rho,
                           kernel_size=ksize,
                           boundary_mode=bmode, maxiter=maxit)
    rhonew[air] = rhoair
    Modout = ModFile_out+"_mediankernel"+str(ksize)+"_"+str(maxit)+".rho"
    mod.writeMod(Modout, dx, dy, dz, rhonew, reference, out=True)
    elapsed = time.time() - start
    print(
        " Used %7.4f s for processing/writing model to %s "
        % (elapsed, Modout))

elif action == "anidiff":
    rhonew = mod.anidiff3D(
        rho,
        ckappa=20, dgamma=0.24, foption=fopt, maxiter=maxit,
        Out=True, Plot=True)
    rhonew[air] = rhoair
    Modout = ModFile_out + "_anisodiff" + str(fopt) + "-" + str(maxit) + ".rho"
    mod.writeMod(Modout, dx, dy, dz, rhonew, reference, out=True)
    elapsed = time.time() - start
    print(" Used %7.4f s for processing/writing model to %s "
          % (elapsed, Modout))


total = total + elapsed
print(" Total time used:  %f s " % (total))
