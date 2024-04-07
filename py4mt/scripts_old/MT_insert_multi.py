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
Reads ModEM model, reads ModEM"s Jacobian, does fancy things.

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
import numpy.linalg as npl
import scipy.linalg as scl
import scipy.sparse as scs
import netCDF4 as nc

JACOPYAN_DATA = os.environ["JACOPYAN_DATA"]
JACOPYAN_ROOT = os.environ["JACOPYAN_ROOT"]

mypath = [JACOPYAN_ROOT+"/modules/", JACOPYAN_ROOT+"/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)


import jacproc as jac
import modem as mod
import modem as mod
import util as utl

from version import versionstrg

rng = np.random.default_rng()
nan = np.nan  # float("NaN")
blank = 1.e-30 # np.nan
rhoair = 1.e17

version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")




total = 0
ModDir_in = JACOPYAN_DATA + "/Peru/Misti/"
ModDir_out = ModDir_in + "/results_shuttle/"

ModFile_in = ModDir_in + "Misti10_best"    
ModFile_out = ModFile_in
ModFormat = "mod rlm" # "ubc"   
ModOrig = [-16.277300, -71.444397]# Misti


SVDFile = ModDir_in +"Misti_best_Z5_nerr_sp-8"


ModOutSingle = True


if not os.path.isdir(ModDir_out):
    print("File: %s does not exist, but will be created" % ModDir_out)
    os.mkdir(ModDir_out)
    
    
padding = [10, 10,   10, 10,   0, 20]
bodymask = [3, 3, 5]
bodyval = 0.2
flip = "alt"

# regular perturbed model (like checkerboard) 
# model_set = 1
# method =   ["regular", [1, 1,   1, 1,   1, 1], [4, 4, 6]]

# random perturbed grid 
model_set = 10 # should be more
method = [
    ["random", 25, [1, 1,   1, 1,   1, 1], "uniform", [3, 3, 5], 6]
       ]


total = 0.
start = time.perf_counter()
dx, dy, dz, base_model, refmod, _ = mod.read_mod(ModFile_in, ".rho",trans="log10")
mdims = np.shape(base_model)
aircells = np.where(base_model>np.log10(rhoair/10.))
jacmask = jac.set_airmask(rho=base_model, aircells=aircells, blank=np.log10(blank), flat = False, out=True)
jacflat = jacmask.flatten(order="F")
elapsed = time.perf_counter() - start
total = total + elapsed
print(" Used %7.4f s for reading model from %s "
      % (elapsed, ModFile_in + ".rho"))

start = time.perf_counter()
print("Reading Jacobian SVD from "+SVDFile)
SVD = np.load(SVDFile) 
U = SVD["U"]
S = SVD["S"]
print(np.shape(U), np.shape(U))
elapsed = time.perf_counter() - start
print(" Used %7.4f s for reading Jacobian/data from %s" % (elapsed, SVDFile))
total = total + elapsed
print("\n")




for ibody in range(model_set):
    
    model = base_model.copy()
    templ = mod.distribute_bodies_ijk(model=model, method=method)
    new_model = mod.insert_body_ijk(rho_in=model, template=templ, perturb=bodyval, bodymask=bodymask) 
    new_model[aircells] = rhoair
    
    ModFile = ModDir_out+ModFile_out+"_"+str(ibody)+"+perturbed.rho"
    Header = "# "+ModFile
    
    rho_proj = jac.project_model(m=model, U=U, tst_sample=new_model, nsamp=1)
    
    if ibody = 0:
        nix = 0
    else:
        nix = 0
        

# mod.write_mod_npz(file=None, 
                    # dx=None, dy=None, dz=None, mval=None, reference=None,
                    # compressed=True, trans="LINEAR", 
                    # aircells=None, mvalair=1.e17, blank=1.e-30, header="", 
                    # out=True):


total = total + elapsed
print(" Total time used:  %f s " % (total))
