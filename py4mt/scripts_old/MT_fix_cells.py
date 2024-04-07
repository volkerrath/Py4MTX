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

fixrho = "prior"
if "val" in fixrho.lower():
    rhofix = 300.
    fixmod = [fixrho, rhofix]
else:
    modfix= False
    fixmod = [fixrho]

"""
Border 
"""
method = ["border", 5]
"""
Distance
"""
method = ["distance", 100000.]


ModFile_in = PY4MT_DATA +"/Peru/Tacna/TAC8/TAC_300.rho"
CovFile_in = PY4MT_DATA +"/Peru/Tacna/TAC8/TAC6_04.cov"
DatFile_in = PY4MT_DATA +"/Peru/Tacna/TAC8/TAC6_Z.dat"

if "bord" in method.lower():
    CovFile_out = PY4MT_DATA+"/Peru/Tacna/TAC8/TAC_04_border"+str(border)+"_fixed_prior.cov"
    if modfix:
        ModFile_out=PY4MT_DATA+"/Peru/Tacna/TAC8/TAC_04_border"+str(border)+"_fixed"+str(round(rhofix))+"Ohmm.rho"
else:
    CovFile_out = PY4MT_DATA+"/Peru/Tacna/TAC8/TAC_04_mindist"+str(distance/1000)+"km_fixed"+str(fixed_zone)+".cov"
    if modfix:
        ModFile_out=PY4MT_DATA+"/Peru/Tacna/TAC8/TAC_04_mindist"+str(distance/1000)+"km_fixed"+str(round(rhofix))+"Ohmm.rho"



start = time.time()

if "dist" in method[0].lower():

    lines_out = mod.fix_cells(covfile_i=CovFile_in, covfile_o=CovFile_out, 
                              modfile_i=ModFile_in, modfile_o=ModFile_out, 
                              datfile_i=DatFile_in, 
                              method=method, fixed=fixed_zone, fixmod=fixmod,
                              out=True)

if "bord" in method[0].lower():

    lines_out = mod.fix_cells(covfile_i=CovFile_in, covfile_o=CovFile_out, 
                              modfile_i=ModFile_in,
                              method = method, fixed=fixed_zone, fixmod=fixmod,
                              out=True)

elapsed = time.time() - start
print(" Used %7.4f for reading/processing/writing covar:"  % (elapsed))
