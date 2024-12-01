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
Reads ModEM's Jacobian, does fancy things.

@author: vrath   Feb 2021

"""

# Import required modules


import os
import sys

# import struct
import time
from datetime import datetime
import warnings
from sys import exit as error

import gc


import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl
import scipy.sparse as scs
import netCDF4 as nc

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [PY4MTX_ROOT+"/modules/", PY4MTX_ROOT+"/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)

import util as utl
from version import versionstrg
import modem as mod
import jacproc as jac

version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")


gc.enable()

rng = np.random.default_rng()
nan = np.nan


SparseThresh = 1.e-8
Sparse = SparseThresh > 0

"""
Errorscale should be set to false if Jacobian originates from the ModEM3DJE.x version
"""
ErrorScale = True 
Scale = 1.


# WorkDir = PY4MTX_DATA+"/Annecy/Jacobians/"
# JFiles = [WorkDir+"ANN_P.jac",WorkDir+"ANN_T.jac", WorkDir+"ANN_Z.jac", WorkDir+"ANN_ZPT.jac"]
# MFile = WorkDir + "ANN_best.rho"

# WorkDir = PY4MTX_DATA+"/NewJacTest/"
# JFiles = [WorkDir+"NewJacTest_P.jac",WorkDir+"NewJacTest_T.jac",WorkDir+"NewJacTest_Z.jac"]
# MFile = WorkDir + "JacTest.rho"

#WorkDir = PY4MTX_DATA+"/Peru/Ubinas/"
#JFiles = [WorkDir+"UBI9_ZPTss.jac"] # WorkDir+"SABA8_Z.jac",]
#MFile = WorkDir + "UBI9_best"

WorkDir = "/home/vrath/Ubaye/"
# WorkDir = PY4MTX_DATA+"/Ubaye/"
JFiles = [WorkDir+"Ubaye26_Z.jac", WorkDir+"Ubaye26_P.jac",  WorkDir+"Ubaye26_T.jac"]

MFile = WorkDir + "Ub26_ZPT_T200_NLCG_014"



if not WorkDir.endswith("/"):
    WorkDir = WorkDir+"/"
nF = len(JFiles)

total = 0.0
start = time.perf_counter()
dx, dy, dz, rho, reference, _ = mod.read_mod(MFile, trans="linear")
dims = np.shape(rho)

rhoair = 1.e17
rhosea = 0.3

aircells = np.where(rho > rhoair/10)
#seacells = np.where(np.isclose((rho, rhosea*np.ones_like(rho)), [1.e-8,1.e-8]))
seacells = np.where(rho == rhosea)
airmask = jac.set_airmask(rho=rho, aircells=aircells, flat = False, out=True)

#blank = 1.e-30 #np.nan
elapsed = time.perf_counter() - start
total = total + elapsed

print(" Used %7.4f s for reading model from %s " % (elapsed, MFile))


# only testing 
# name, ext = os.path.splitext(MFile)
# TSTFile = name 
# Head = "#  original model"
# mod.write_mod(TSTFile, ModExt="_0_MaskTest.rho", trans="LOGE",
#                   dx=dx, dy=dy, dz=dz, mval=rho,
#                   reference=reference, mvalair=rhoair, aircells=aircells, header=Head)
# rhotest = airmask.reshape(dims)*rho
# mod.write_mod(TSTFile, ModExt="_1_MaskTest.rho", trans="LOGE",
#                   dx=dx, dy=dy, dz=dz, mval=rhotest,
#                   reference=reference, mvalair=rhoair, aircells=aircells, header=Head)


for f in np.arange(nF):
    nstr = ""
    name, ext = os.path.splitext(JFiles[f])
    start = time.perf_counter()
    DFile = JFiles[f].replace(".jac", "_jac.dat")
    print("\nReading Data from "+DFile)
    Data, Site, Freq, Comp, DTyp, Head = mod.read_data_jac(DFile)
    elapsed = time.perf_counter() - start
    print(" Used %7.4f s for reading Data from %s " % (elapsed, DFile))
    total = total + elapsed
   
    # print(np.unique(DTyp))
    start = time.perf_counter()
    print("Reading Jacobian from "+JFiles[f])
    Jac, Info = mod.read_jac(JFiles[f])
    elapsed = time.perf_counter() - start
    print(" Used %7.4f s for reading Jacobian from %s " % (elapsed, JFiles[f]))
    total = total + elapsed
    

    if np.shape(Jac)[0]!=np.shape(Data)[0]:
        print(np.shape(Jac),np.shape(Data))
        error(" Dimensions of Jacobian and data do not match! Exit.")
        
    mx = np.nanmax(Jac)
    mn = np.nanmin(Jac)
    print(JFiles[f]+" raw minimum/maximum masked Jacobian value is "+str(mn)+"/"+str(mx)) 
   

    if ErrorScale:
        nstr = nstr+"_nerr"
        start = time.perf_counter()
        dsh = np.shape(Data)
        err = np.reshape(Data[:, 7], (dsh[0], 1))
        Jac = jac.normalize_jac(Jac, err)
        
        mx = np.nanmax(Jac)
        mn = np.nanmin(Jac)
        print(JFiles[f]+" scaled minimum/maximum masked Jacobian value is "+str(mn)+"/"+str(mx)) 
        
        
        elapsed = time.perf_counter() - start
        print(" Used %7.4f s for normalizing Jacobian with data error from %s " %
              (elapsed, DFile))
        start = time.perf_counter()



    sstr = "_full"
    if SparseThresh > 0.:
        
        # airmask = airmask.flatten(order="F")
        sstr = "_sp"+str(round(np.log10(SparseThresh)))
        start = time.perf_counter()
        for idt in np.arange(np.shape(Jac)[0]):
            JJ = Jac[idt,:].reshape(dims,order="F")
            Jac[idt,:] = (airmask*JJ).flatten(order="F")
                  
         
        Scale = np.nanmax(np.abs(Jac))
        Jac, Scale = jac.sparsify_jac(Jac, scalval=Scale, sparse_thresh=SparseThresh)
        mx = np.nanmax(np.abs(Jac.todense()))
        mn = np.nanmin(np.abs(Jac.todense()))
        print(JFiles[f]+" sparse minimum/maximum masked Jacobian value is "+str(mn)+"/"+str(mx)) 

        elapsed = time.perf_counter() - start
        total = total + elapsed
        print(" Used %7.4f s for sparsifying Jacobian %s " %
              (elapsed, JFiles[f]))



    name = name+nstr+sstr
    start = time.perf_counter()

    np.savez_compressed(name + "_info.npz", Freq=Freq, Data=Data, Site=Site, Comp=Comp,
                        Info=Info, DTyp=DTyp, Scale=Scale, allow_pickle=True)
    if Sparse:
        scs.save_npz(name + "_jac.npz", matrix=Jac)  # , compressed=True)
    else:
        np.savez_compressed(name +"_jac.npz", Jac)
    elapsed = time.perf_counter() - start
    total = total + elapsed
    print(" Used %7.4f s for writing Jacobian and info to %s " % (elapsed, name))
    Jac = None
    JJ = None
    #del Jac
