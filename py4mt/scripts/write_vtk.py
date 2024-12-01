#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 18:34:38 2024

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
import numpy.linalg as npl
import scipy.linalg as scl
import scipy.sparse as scs
import netCDF4 as nc

import pyevtk.hl as vtx

JACOPYAN_DATA = os.environ["JACOPYAN_DATA"]
JACOPYAN_ROOT = os.environ["JACOPYAN_ROOT"]

mypath = [JACOPYAN_ROOT+"/modules/", JACOPYAN_ROOT+"/scripts/"]
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



rng = np.random.default_rng()
blank =  np.nan
rhoair = 1.e17
rhoair = np.log10(rhoair)

# Ubinas Ubinas Ubinas Ubinas Ubinas Ubinas Ubinas Ubinas Ubinas Ubinas
WorkDir = JACOPYAN_DATA+"/Peru/Ubinas/"
# WorkDir = "/home/vrath/UBI38_JAC/"

  
files = ["Ubi38_ZssPT_Alpha02_NLCG_023"]
nfiles = len(files)

PadMask = [10, 10, 10, 10, 0, 32] 
Scale = 1000.  # 1 = m 1000.=km
DstMask = [-15., 15.,   -15., 15.,  0., 15.]

mask = PadMask



fcount =0
for ifile in np.arange(nfiles):
    infile = WorkDir+files[ifile] +".rho"
    outfile = infile+"_tovtk"


    dx, dy, dz, rho, refmod, _ = mod.read_mod(infile, ".rho",trans="log10")
    
    aircells = np.where(rho>rhoair-1.0)
    
    x, y, z = mod.cells3d(dx, dy, dz)
    
    
    x =  x + refmod[0]
    y =  y + refmod[1]
    z =  z + refmod[2]
        
    x, y, z  = Scale*x, Scale*y, Scale*z
    
    vals = rho.copy()
    vals = np.swapaxes(vals, 0, 1)
    
       
    x, y, z, rho = mod.mask_mesh(x, y, z, mod=vals, mask=mask)
    
    print(np.shape(rho))
    comments = [ "", "" ]
    f= vtx.gridToVTK(outfile, x, y, -z, cellData = {"rho" : rho})
    print("model written to ", f)   


    infile = WorkDir+files[ifile]+".dat"
    outfile = infile.replace(".dat","_tovtk.dat")
    
    site, _, data, _ = mod.read_data(Datfile=infile)
    

    x =  data[:, 3]
    y =  data[:, 4]
    z =  data[:, 5]    
 
    x =  x + refmod[0]
    y =  y + refmod[1]
    z =  z + refmod[2]
     
    y, x, z  = Scale*x, Scale*y, -Scale*z
     
     
       
    sites, siteindex = np.unique(site, return_index=True)
    x = x[siteindex]
    y = y[siteindex]
    z = z[siteindex]

    sitenum= np.array([ii for ii in np.arange(len(z))])    

  
   
    comments = [ "", "" ]
    f= vtx.pointsToVTK(outfile, x, y, z, data = {"sites" : sites})
    print("sites written to ", f)   