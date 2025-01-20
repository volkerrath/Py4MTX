#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 12:36:43 2024

@author: vrath
"""


import os
import sys
import numpy as np

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [PY4MTX_ROOT+"/py4mt/modules/", PY4MTX_ROOT+"/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)


import modem as mod
import util as utl

from version import versionstrg

# rng = np.random.default_rng()
# nan = np.nan  # float("NaN")
# blank = 1.e-30 # np.nan
# rhoair = 1.e17

version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")



total = 0
ModDir_in = PY4MTX_DATA + "/Peru/Misti/"
ModDir_in = "/home/vrath/Py4MTX/work/Misti_alphas/"
# /home/vrath/MT_Data/Peru/Misti/
ModDir_out = ModDir_in

ModFile_in = ModDir_in + "Mis_50"
ModFile_out = ModFile_in



if not os.path.isdir(ModDir_out):
    print("File: %s does not exist, but will be created" % ModDir_out)
    os.mkdir(ModDir_out)

dx, dy, dz, base_model, refmod, _ = mod.read_mod(ModFile_in, ".rho",trans="log10")

_, depth = mod.set_mesh(d=dz)

beg_lin = np.array([100., 0.1,0.1])
end_lin = np.array([25000., 0.6,0.6])

ax, ay = mod.generate_alphas(dz, beg_lin=beg_lin, end_lin=end_lin)

print(ax)
