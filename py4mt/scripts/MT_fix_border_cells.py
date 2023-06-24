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
import netCDF4 as nc
import scipy.ndimage as spn
import scipy.linalg as spl
PY4MT_ROOT = os.environ["PY4MT_ROOT"]
mypath = [PY4MT_ROOT+"/py4mt/modules/", PY4MT_ROOT+"/py4mt/scripts/"]

for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)

import modules
import modem as mod
import util as utl
from version import versionstrg


rng = numpy.random.default_rng()
nan = numpy.nan  # float("NaN")
version, _ = versionstrg()
titstrng = util.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")

PY4MT_DATA = os.environ["PY4MT_DATA"]


air = 0
ocean = 9
fixed = 3
border = 5

ModFile_in = PY4MT_DATA +"/test/test.rho"
CovFile_in = PY4MT_DATA +"/test/test.cov"
CovFile_out = PY4MT_DATA +"/test/test_fix"+str(border)+".cov""

start = time.time()

dx, dy, dz, rho, reference = mod.readMod(ModFile_in + ".rho", out=True)
# writeMod(ModFile_out+'.rho', dx, dy, dz, rho,reference,out = True)
elapsed = time.time() - start
total = total + elapsed
print(" Used %7.4f s for reading model from %s "
      % (elapsed, ModFile_in + ".rho"))
modsize = np.shape(rho)



total = total + elapsed
print(" Total time used:  %f s " % (total))
