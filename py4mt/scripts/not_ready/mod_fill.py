#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 17:17:16 2023

@author: vrath
"""
import os
import sys
import numpy as np

from sys import exit as error
# import struct
import time
from datetime import datetime
import warnings


PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]
PY4MTX_DATA = os.environ["PY4MTX_DATA"]

mypath = [PY4MTX_ROOT+"/modules/", PY4MTX_ROOT+"/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)


import jacproc as jac
import modem as mod
from version import versionstrg
import util as utl

version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")

# # FOGO
# rhosea = 0.333
# rhoair = 1.e10
# FillVal = 100.
# InMod =  "/home/vrath/MT_Data/Fogo/FOG_best"
# OutMod =  "/home/vrath/MT_Data/Fogo/FOG_prior"+str(int(FillVal))

# UBI
rhosea = 0.3
rhoair = 1.e17
FillVal = 100.
InMod =  "/home/vrath/MT_Data/Fogo/FOG_best"
OutMod =  "/home/vrath/MT_Data/Fogo/FOG_prior"+str(int(FillVal))

print("\n\nTransforming ModEM model file:" )
print(InMod)



start = time.perf_counter()
dx, dy, dz, rho, refmod, _ = mod.read_mod(InMod, ".rho",trans="linear")
dims = np.shape(rho)
elapsed = time.perf_counter() - start
print(" Used %7.4f s for reading MOD model from %s " % (elapsed, InMod))
   
allcells = np.where(rho)
aircells = np.where(np.isclose(rho, np.ones_like(rho)*rhoair, rtol=1e-04))
seacells = np.where(np.isclose(rho, np.ones_like(rho)*rhosea, rtol=1e-04))

print(np.shape(aircells))
rho_new = rho.copy()
rho_new[allcells] = FillVal
rho_new[seacells] = rhosea
rho_new[aircells] = rhoair

# rho_new[np.where(np.isclose(rho, rhoair, rtol=1e-05))] = rhoair

print("\n\nTransformed ModEM model file:" )
print(OutMod)

mod.write_mod(OutMod, modext=".rho",trans = "LOGE",
              dx=dx, dy=dy, dz=dz, mval=rho_new,
              reference=refmod, mvalair=rhoair, aircells=aircells, header="")

