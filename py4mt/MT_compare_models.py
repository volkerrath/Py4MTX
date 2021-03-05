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
#       jupytext_version: 1.9.1
# ---

"""
Created on Fri Sep 11 15:41:25 2020

@author: vrath
"""
# Import required modules

import os
import sys
from sys import exit as error
# import struct
import time
from datetime import datetime
import warnings

import numpy as np

mypath = ["/home/vrath/Py4MT/py4mt/modules/",
          "/home/vrath/Py4MT/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)

import modules
import modules.util as utl
import modules.modem as mod
import modules.inv
from version import versionstrg

Strng, _ = versionstrg()
now = datetime.now()
print("\n\n"+Strng)
print("Nullspace Shuttle"+"\n"+"".join("Date " + now.strftime("%m/%d/%Y, %H:%M:%S")))
print("\n\n")

warnings.simplefilter(action="ignore", category=FutureWarning)


Data_tmp = '.TMP'

M_In1 = r'/home/vrath/work/MaurienneJCN/MaurPrior.rho'
M_In2 = r'/home/vrath/work/MaurienneJCN/MaurPrior.rho'

M_Out_Dif = 'CompareMod_Diff.rho'
M_Out_Crg = 'CompareMod_CrossGrad.rho'


dx, dy, dz, R1, ref = mod.read_model(M_In1, out=True)
dx, dy, dz, R2, ref = mod.read_model(M_In2, out=True)

r1 = np.log10(R1)
r2 = np.log10(R2)

rdiff = r2 - r1

mod.write_model(M_Out_Dif, dx=dx, dy=dy, dz=dz, rho=rdiff,
         reference=ref, trans="LINEAR", out=True)

# g1x,g1y,g1z = np.gradient(r1)
# ng1 = np.sqrt(g1x*g1x + g1y*g1y + g1z*g1z)

# g2x,g2y,g2z = np.gradient(r2)
# ng2 = np.sqrt(g2x*g1x + g2y*g1y + g2z*g2z)

# print(np.shape(ng2))



# Rcg = inv.crossgrad(dx, dy, dz, R1, R2)

# writeMod(MCrg, dx=dx, dy=dy, dz=dz, rho=Rcrg,
#          reference=ref, trans="LINEAR", out=True)
