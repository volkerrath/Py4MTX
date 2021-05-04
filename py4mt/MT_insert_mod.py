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
Reads ModEM'smodel file, adds perturbations.

Ellipsoids or boxes

@author: vr jan 2021


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

import vtk
import pyvista as pv
import PVGeo as pvg
import omf
import omfvista as ov
import gdal

mypath = ["/home/vrath/Py4MT/py4mt/modules/",
          "/home/vrath/Py4MT/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)

import modules
import modem as mod
import util as utl
from version import versionstrg

Strng, _ = versionstrg()
now = datetime.now()
print("\n\n"+Strng)
print("Insert bodies into model"+"\n"+"".join("Date " + now.strftime("%m/%d/%Y, %H:%M:%S")))
print("\n\n")


warnings.simplefilter(action="ignore", category=FutureWarning)


rhoair = 1.e17

ModFile_in = r"/home/vrath/work/MT/Annecy/ImageProc/In/ANN20_02_PT_NLCG_016"
ModFile_out = r"/home/vrath/work/MT/Annecy/ImageProc/Out/ANN20_02_PT_NLCG_016_insert"

geocenter = [45.938251, 6.084900]
utm_x, utm_y = utl.proj_latlon_to_utm(geocenter[0], geocenter[1], utm_zone=32631)
utmcenter = [utm_x, utm_y, 0.0]

ell = ["ell", "rep", 0., 0., 0., 3000., 10000., 2000., 1000., 0., 0., 30.]
box = ["box", "rep", 0., 0., 0., 1000., 2000., 1000., 1000., 0., 0., 30.]
bodies = [ell, box]
nb = np.shape(bodies)
# smoother=['gaussian',0.5]
smoother = ["uniform", 3]
total = 0
start = time.time()

dx, dy, dz, rho, reference = mod.readMod(ModFile_in + ".rho", out=True)
# writeMod(ModFile_out+'.rho', dx, dy, dz, rho,reference,out = True)
elapsed = time.time() - start
total = total + elapsed
print(" Used %7.4f s for reading model from %s "
      % (elapsed, ModFile_in + ".rho"))

air = rho > rhoair / 100.

rho = mod.prepare_mod(rho, rhoair=rhoair)

for ibody in range(nb[0]):
    body = bodies[ibody]
    rhonew = mod.insert_body(dx, dy, dz, rho, body, smooth=smoother)
    rhonew[air] = rhoair
    Modout = ModFile_out+"_"+body[0]+str(ibody)+"_"+smoother[0]+".rho"
    mod.writeMod(Modout, dx, dy, dz, rhonew, reference, out=True)

    elapsed = time.time() - start
    print(" Used %7.4f s for processing/writing model to %s"
          % (elapsed, Modout))
    print("\n")


total = total + elapsed
print(" Total time used:  %f s " % (total))
