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
from sys import exit as error
# import struct
import time
from datetime import datetime
import warnings
import gc

import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl
import scipy.sparse as scs
import netCDF4 as nc


mypath = ["/home/vrath/Py4MT/py4mt/modules/", "/home/vrath/Py4MT/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)


import jacproc as jac
import modem as mod
from version import versionstrg

Strng, _ = versionstrg()
now = datetime.now()
print("\n\n"+Strng)
print("Calculate Sensitivity "+"\n"+"".join("Date " + now.strftime("%m/%d/%Y, %H:%M:%S")))
print("\n\n")


PY4MT_DATA = os.environ["PY4MT_DATA"]


gc.enable()

rng = np.random.default_rng()
blank = np.nan

sparsify = True
sparse_thresh = 1.e-6

# WorkDir = r"/home/vrath/work/MT_Data/Ubaye/UB22_jac_best/"
# MFile   = WorkDir +r"Ub22_ZoffPT_02_NLCG_014.rho"
# MPad=[12, 12 , 12, 12, 0, 36]

# # JFiles = [WorkDir+r"Ub22_Zoff.jac", ]
# # DFiles = [WorkDir+r"Ub22_Zoff.dat", ]

# # JFiles = [WorkDir+r"Ub22_P.jac", ]
# # DFiles = [WorkDir+r"Ub22_P.dat", ]

# # JFiles = [WorkDir+r"Ub22_T.jac", ]
# # DFiles = [WorkDir+r"Ub22_T.dat", ]

# JFiles = [WorkDir+r"Ub22_T.jac", WorkDir+r"Ub22_P.jac", WorkDir+r"Ub22_Zoff.jac", ]
# DFiles = [WorkDir+r"Ub22_T.dat", WorkDir+r"Ub22_P.dat", WorkDir+r"Ub22_Zoff.dat", ]


# KRAFLA case
WorkDir = PY4MT_DATA+"/NewJacobians/"
MFile   = WorkDir +"Ub22.rho"
MPad=[13, 13 , 13, 13, 0, 36]
JFiles = []
DFiles = []
files = os.listdir(WorkDir)
for entry in files:
    # print(entry)
    if entry.endswith(".jac"):
        JFiles.append(entry)
        name, ext = os.path.splitext(entry)
        DFiles.append(name+"_jac.dat")

NN = np.size(JFiles)
if np.size(DFiles) != np.size(JFiles):
    error("Data file number not equal Jac file number! Exit.")
nF = np.size(DFiles)


total = 0.0


start = time.time()
dx, dy, dz, rho, reference, _, vcell = mod.read_model(MFile, trans="linear", volumes=True)
dims = np.shape(rho)

rhoair = 1.e17
aircells = np.where(rho>rhoair/10)

TSTFile = WorkDir+"Krafla0_MaskTest.rho"
mod.write_model(TSTFile, dx, dy, dz, rho, reference, trans="LOGE", mvalair=blank, aircells=aircells)


jacmask = jac.set_mask(rho=rho, pad=MPad, blank= blank, flat = True, out=True)
jdims= np.shape(jacmask)
j0 = jacmask.reshape(dims)
j0[aircells] = blank
jacmask = j0.reshape(jdims)

rhotest = jacmask.reshape(dims)*rho
TSTFile = WorkDir+"Krafla1_MaskTest.rho"
mod.write_model(TSTFile, dx, dy, dz, rhotest, reference, trans="LOGE", mvalair=blank, aircells=aircells)

elapsed = time.time() - start
total = total + elapsed
print(" Used %7.4f s for reading model from %s " % (elapsed, MFile))


snsValT = np.zeros(jdims)
snsValP = np.zeros(jdims)
snsValZ = np.zeros(jdims)

for f in np.arange(nF):

    name, ext = os.path.splitext(JFiles[f])
    start =time.time()
    print("\nReading Data from "+DFiles[f])
    Data, Site, Freq, Comp, Head = mod.read_data_jac(WorkDir+DFiles[f])
    elapsed = time.time() - start
    print(" Used %7.4f s for reading Data from %s " % (elapsed, DFiles[f]))
    total = total + elapsed

    start = time.time()
    print("Reading Jacobian from "+JFiles[f])
    Jac, Info = mod.read_jac(WorkDir+JFiles[f])
    elapsed = time.time() - start
    print(" Used %7.4f s for reading Jacobian from %s " % (elapsed, JFiles[f]))
    total = total + elapsed


    start = time.time()
    dsh = np.shape(Data)
    err = np.reshape(Data[:, 7], (dsh[0], 1))
    Jac = jac.normalize_jac(Jac, err)
    elapsed = time.time() - start
    print(" Used %7.4f s for normalizing Jacobian from %s " % (elapsed, JFiles[f]))

    mx = np.nanmax(np.abs(Jac))
    mn = np.nanmin(np.abs(Jac))
    print(JFiles[f]+" minimum/maximum Jacobian value is "+str(mn)+"/"+str(mx))
    mx = np.nanmax(np.abs(Jac*jacmask))
    print(JFiles[f]+" minimum/maximum masked Jacobian value is "+str(mn)+"/"+str(mx))
    print(JFiles[f]+" number of elements in masked Jacobian is "+str(np.count_nonzero(~np.isfinite(Jac))))
    # print( np.count_nonzero(~np.isnan(jacmask))*np.shape(Jac)[0])

    """
    Full_Impedance              = 1
    Off_Diagonal_Impedance      = 2
    Full_Vertical_Components    = 3
    Full_Interstation_TF        = 4
    Off_Diagonal_Rho_Phase      = 5
    Phase_Tensor                = 6
    """

    I1 =Info[:,1]
    tmpJac = Jac[np.where(I1 == 1),:]*jacmask
    print(np.shape(tmpJac))
    print(np.shape(jacmask))

    tmpValZ = np.nansum(np.power(tmpJac, 2), axis=1)
    # print(np.shape(tmpValZ))
    snsValZ = snsValZ + tmpValZ
    maxValZ = np.sqrt(np.nanmean(tmpValZ))
    print(" Z Maximum value is "+str(maxValZ))

    tmpJac = Jac[np.where(I1 == 3),:]*jacmask
    tmpValT = np.nansum(np.power(tmpJac, 2), axis=1)
    snsValT = snsValT + tmpValT
    maxValT = np.sqrt(np.nanmean(tmpValT))
    print(" T Maximum value is "+str(maxValT))

    tmpJac = Jac[np.where(I1 == 6),:]*jacmask
    tmpValP = np.nansum(np.power(tmpJac, 2), axis=1)
    snsValP = snsValP + tmpValP
    maxValP = np.sqrt(np.nanmean(tmpValP))
    print(" P Maximum value is "+str(maxValP))

maxValZ = np.sqrt(np.nanmax([snsValZ]))
print(" Merged Z Maximum value is "+str(maxValZ))

maxValT = np.sqrt(np.nanmax([snsValT]))
print(" Merged T Maximum value is "+str(maxValT))

maxValP = np.sqrt(np.nanmax([snsValP]))
print(" Merged P Maximum value is "+str(maxValP))

snsVal = snsValZ+snsValT+snsValP
maxVal = np.sqrt(np.nanmax([snsVal]))
print(" Merged ZTP Maximum value is "+str(maxVal))

start = time.time()

"""
Raw sensitivities
"""
SNSFile = WorkDir+"Krafla1_nerr_C.sns"
tmpS = np.sqrt(snsVal)
S = np.reshape(tmpS, dims)# S = np.reshape(tmpS, dims, order="F")
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans="LOGE", rhoair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

SNSFile = WorkDir+"Krafla1_Z_nerr_C.sns"
tmpS = np.sqrt(snsValZ)
S = np.reshape(tmpS, dims)# S = np.reshape(tmpS, dims, order="F")
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans="LOGE", rhoair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

SNSFile = WorkDir+"Krafla1_T_nerr_C.sns"
tmpS = np.sqrt(snsValT)
S = np.reshape(tmpS, dims)# S = np.reshape(tmpS, dims, order="F")
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans="LOGE", rhoair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

SNSFile = WorkDir+"Krafla1_P_nerr_C.sns"
tmpS = np.sqrt(snsValP)
S = np.reshape(tmpS, dims)# S = np.reshape(tmpS, dims, order="F")
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans="LOGE", rhoair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

"""
Max normalized sensitivities
"""
SNSFile = WorkDir+"Krafla1_nerr_nmax_C.sns"
tmpS = np.sqrt(snsVal)
S = np.reshape(tmpS, dims)# S = np.reshape(tmpS, dims, order="F")/maxVal
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans="LOGE", rhoair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

SNSFile = WorkDir+"Krafla1_Z_nerr_nmax_C.sns"
tmpS = np.sqrt(snsValZ)
S = np.reshape(tmpS, dims)# S = np.reshape(tmpS, dims, order="F")/maxValZ
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans="LOGE", rhoair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

SNSFile = WorkDir+"Krafla1_T_nerr_nmax_C.sns"
tmpS = np.sqrt(snsValT)
S = np.reshape(tmpS, dims)# S = np.reshape(tmpS, dims, order="F")/maxValT
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans="LOGE", rhoair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

SNSFile = WorkDir+"Krafla1_P_nerr_nmax_C.sns"
tmpS = np.sqrt(snsValP)
S = np.reshape(tmpS, dims)# S = np.reshape(tmpS, dims, order="F")/maxValP
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans="LOGE", rhoair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

"""
Max and vol normalized sensitivities
"""
SNSFile = WorkDir+"Krafla1_nerr_nmax_nvol_C.sns"
tmpS = np.sqrt(snsVal)
S = np.reshape(tmpS, dims)# S = np.reshape(tmpS, dims, order="F")/(maxVal*vcell)
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans="LOGE", rhoair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

SNSFile = WorkDir+"Krafla1_Z_nerr_nmax_nvol_C.sns"
tmpS = np.sqrt(snsValZ)
S = np.reshape(tmpS, dims)# S = np.reshape(tmpS, dims, order="F")/(maxValZ*vcell)
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans="LOGE", rhoair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

SNSFile = WorkDir+"Krafla1_T_nerr_nmax_nvol_C.sns"
tmpS = np.sqrt(snsValT)
S = np.reshape(tmpS, dims)# S = np.reshape(tmpS, dims, order="F")/(maxValT*vcell)
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans="LOGE", rhoair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

SNSFile = WorkDir+"Krafla1_P_nerr_nmax_nvol_C.sns"
tmpS = np.sqrt(snsValP)
S = np.reshape(tmpS, dims)# S = np.reshape(tmpS, dims, order="F")/(maxValP*vcell)
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans="LOGE", rhoair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)


elapsed = time.time() - start
print(" Used %7.4f s for writing sensitivities " % (elapsed))
