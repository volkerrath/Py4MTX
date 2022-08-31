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
sparse_thresh = 1.e-7
normalize_jacmax = "local"
if "loc" in normalize_jacmax.lower():
    jacmax ="_max"
else:
    jacmax ="_maxtotal"

outform = "LINEAR"
outform = outform.upper()

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


# Ubaye case
WorkDir = PY4MT_DATA+"/NewJacobians/"
WorkName = "Ub22Jac"
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
vcell = vcell.flatten().reshape((1,-1))

rhoair = 1.e17
aircells = np.where(rho>rhoair/10)

# TSTFile = WorkDir+WorkName+"0_MaskTest.rho"
# mod.write_model(TSTFile, dx, dy, dz, rho, reference, trans="LINEAR", mvalair=blank, aircells=aircells)


jacmask = jac.set_mask(rho=rho, pad=MPad, blank= blank, flat = True, out=True)
jdims= np.shape(jacmask)
j0 = jacmask.reshape(dims)
j0[aircells] = blank
jacmask = j0.reshape(jdims)

# rhotest = jacmask.reshape(dims)*rho
# TSTFile = WorkDir+WorkName+"1_MaskTest.rho"
# mod.write_model(TSTFile, dx, dy, dz, rhotest, reference, trans="LINEAR", mvalair=blank, aircells=aircells)

elapsed = time.time() - start
total = total + elapsed
print(" Used %7.4f s for reading model from %s " % (elapsed, MFile))


snsValT = np.zeros(jdims)
snsValP = np.zeros(jdims)
snsValZ = np.zeros(jdims)
snsValZoff = np.zeros(jdims)
snsValZtot = np.zeros(jdims)

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
    err = np.reshape(Data[:, 5], (dsh[0], 1))
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
    tmpValZ = np.nansum(np.power(tmpJac, 2), axis=1)
    snsValZtot = snsValZtot + tmpValZ
    # maxValZ = np.nanmax(np.sqrt(tmpValZoff))
    # print(" Z Maximum value is "+str(maxValZ))


    tmpJac = Jac[np.where(I1 == 2),:]*jacmask
    tmpValZ = np.nansum(np.power(tmpJac, 2), axis=1)
    snsValZoff = snsValZoff + tmpValZ
    # maxValZ = np.nanmax(np.sqrt(tmpValZtot))
    # print(" Z Maximum value is "+str(maxValZ))

    tmpJac = Jac[np.where(I1 == 3),:]*jacmask
    tmpValT = np.nansum(np.power(tmpJac, 2), axis=1)
    snsValT = snsValT + tmpValT
    # maxValT = np.nanmax(np.sqrt(tmpValT))
    # print(" T Maximum value is "+str(maxValT))

    tmpJac = Jac[np.where(I1 == 6),:]*jacmask
    tmpValP = np.nansum(np.power(tmpJac, 2), axis=1)
    snsValP = snsValP + tmpValP
    # maxValP = np.nanmax(np.sqrt(tmpValP))
    # print(" P Maximum value is "+str(maxValP))

# maxValZ = np.nanmax([snsValZ])
maxValZtot = np.nanmax(np.sqrt(snsValZtot))
print(" Merged Z Maximum value is "+str(maxValZtot))
# maxValZ = np.nanmax([snsValZ])
maxValZoff = np.nanmax(np.sqrt(snsValZoff))
print(" Merged Z Maximum value is "+str(maxValZoff))

# maxValT = np.nanmax([snsValT])
maxValT = np.nanmax(np.sqrt(snsValT))
print(" Merged T Maximum value is "+str(maxValT))

# maxValP = np.nanmax([snsValP])
maxValP = np.nanmax(np.sqrt(snsValP))
print(" Merged P Maximum value is "+str(maxValP))

snsVal = snsValZtot+snsValZoff+snsValT+snsValP
# maxVal = np.nanmax([snsVal])
maxVal = np.nanmax(np.sqrt(snsVal))
print(" Merged ZTP Maximum value is "+str(maxVal))

start = time.time()

"""
Raw sensitivities
"""
SNSFile = WorkDir+WorkName+"_raw.sns"
tmpS = snsVal
S = np.reshape(tmpS, dims)# S = np.reshape(tmpS, dims, order="F")
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

SNSFile = WorkDir+WorkName+"_Ztot_raw.sns"
tmpS = snsValZtot
S = np.reshape(tmpS, dims)# S = np.reshape(tmpS, dims, order="F")
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

SNSFile = WorkDir+WorkName+"_Zoff_raw.sns"
tmpS = snsValZoff
S = np.reshape(tmpS, dims)# S = np.reshape(tmpS, dims, order="F")
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

SNSFile = WorkDir+WorkName+"_T_raw.sns"
tmpS = snsValT
S = np.reshape(tmpS, dims)# S = np.reshape(tmpS, dims, order="F")
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

SNSFile = WorkDir+WorkName+"_P_raw.sns"
tmpS = snsValP
S = np.reshape(tmpS, dims)# S = np.reshape(tmpS, dims, order="F")
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)



"""
Squareroot Raw sensitivities
"""
SNSFile = WorkDir+WorkName+"_sqrtraw.sns"
tmpS = np.sqrt(snsVal)
S = np.reshape(tmpS, dims)# S = np.reshape(tmpS, dims, order="F")
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

SNSFile = WorkDir+WorkName+"_Ztot_sqrtraw.sns"
tmpS = np.sqrt(snsValZtot)
S = np.reshape(tmpS, dims)# S = np.reshape(tmpS, dims, order="F")
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

SNSFile = WorkDir+WorkName+"_Zoff_sqrtraw.sns"
tmpS = np.sqrt(snsValZoff)
S = np.reshape(tmpS, dims)# S = np.reshape(tmpS, dims, order="F")
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

SNSFile = WorkDir+WorkName+"_T_sqrtraw.sns"
tmpS = np.sqrt(snsValT)
S = np.reshape(tmpS, dims)# S = np.reshape(tmpS, dims, order="F")
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

SNSFile = WorkDir+WorkName+"_P_sqrtraw.sns"
tmpS = np.sqrt(snsValP)
S = np.reshape(tmpS, dims)# S = np.reshape(tmpS, dims, order="F")
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

"""
vol normalized sensitivities
"""
SNSFile = WorkDir+WorkName+"_vol.sns"
tmpS = np.sqrt(snsVal)/vcell[:]
S = np.reshape(tmpS, dims)# S = np.reshape(tmpS, dims, order="F")/(maxVal*vcell)
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

SNSFile = WorkDir+WorkName+"_Ztot_vol.sns"
tmpS = np.sqrt(snsValZtot)/vcell[:]
S = np.reshape(tmpS, dims)# S = np.reshape(tmpS, dims, order="F")/(maxValZ*vcell)
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

SNSFile = WorkDir+WorkName+"_Zoff_vol.sns"
tmpS = np.sqrt(snsValZoff)/vcell[:]
S = np.reshape(tmpS, dims)# S = np.reshape(tmpS, dims, order="F")/(maxValZ*vcell)
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

SNSFile = WorkDir+WorkName+"_T_vol.sns"
tmpS = np.sqrt(snsValT)/vcell[:]
S = np.reshape(tmpS, dims)
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

SNSFile = WorkDir+WorkName+"_P_vol.sns"
tmpS = np.sqrt(snsValP)/vcell[:]
S = np.reshape(tmpS, dims)# S = np.reshape(tmpS, dims, order="F")/(maxValP*vcell)
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

"""
Max normalized sensitivities
"""
SNSFile = WorkDir+WorkName+"_max.sns"
tmpS = np.sqrt(snsVal)/(maxVal)
S = np.reshape(tmpS, dims)# S = np.reshape(tmpS, dims, order="F")/(maxVal)
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

SNSFile = WorkDir+WorkName+"_Ztot_"+jacmax+".sns"
if "loc" in normalize_jacmax.lower():
    tmpS = np.sqrt(snsValZtot)/(maxValZtot)
else:
    tmpS = np.sqrt(snsValZtot)/(maxVal)
S = np.reshape(tmpS, dims)# S = np.reshape(tmpS, dims, order="F")/(maxValZ)
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

SNSFile = WorkDir+WorkName+"_Zoff_"+jacmax+".sns"
if "loc" in normalize_jacmax.lower():
    tmpS = np.sqrt(snsValZoff)/(maxValZoff)
else:
    tmpS = np.sqrt(snsValZoff)/(maxVal)
S = np.reshape(tmpS, dims)# S = np.reshape(tmpS, dims, order="F")/(maxValZ)
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

SNSFile = WorkDir+WorkName+"_T_"+jacmax+".sns"
if "loc" in normalize_jacmax.lower():
    tmpS = np.sqrt(snsValT)/(maxValT)
else:
    tmpS = np.sqrt(snsValT)/(maxVal)
S = np.reshape(tmpS, dims)
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

SNSFile = WorkDir+WorkName+"_P_"+jacmax+".sns"
if "loc" in normalize_jacmax.lower():
    tmpS = np.sqrt(snsValP)/(maxValP)
else:
    tmpS = np.sqrt(snsValP)/(maxVal)
S = np.reshape(tmpS, dims)# S = np.reshape(tmpS, dims, order="F")/(maxValP)
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)


"""
Max and vol normalized sensitivities
"""
SNSFile = WorkDir+WorkName+"_vol_max.sns"
tmpS = np.sqrt(snsVal)/(maxVal*vcell)
S = np.reshape(tmpS, dims)# S = np.reshape(tmpS, dims, order="F")/(maxVal*vcell)
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

SNSFile = WorkDir+WorkName+"_Ztot_vol_"+jacmax+".sns"
if "loc" in normalize_jacmax.lower():
    tmpS = np.sqrt(snsValZtot)/(maxValZtot*vcell)
else:
    tmpS = np.sqrt(snsValZtot)/(maxVal)
S = np.reshape(tmpS, dims)# S = np.reshape(tmpS, dims, order="F")/(maxValZ*vcell)
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

SNSFile = WorkDir+WorkName+"_Zoff_vol_"+jacmax+".sns"
if "loc" in normalize_jacmax.lower():
    tmpS = np.sqrt(snsValZoff)/(maxValZoff*vcell)
else:
    tmpS = np.sqrt(snsValZoff)/(maxVal)
S = np.reshape(tmpS, dims)# S = np.reshape(tmpS, dims, order="F")/(maxValZ*vcell)
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

SNSFile = WorkDir+WorkName+"_T_vol_"+jacmax+".sns"
if "loc" in normalize_jacmax.lower():
    tmpS = np.sqrt(snsValT)/(maxValT*vcell)
else:
    tmpS = np.sqrt(snsValT)/(maxVal)
S = np.reshape(tmpS, dims)
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

SNSFile = WorkDir+WorkName+"_P_vol_"+jacmax+".sns"
if "loc" in normalize_jacmax.lower():
    tmpS = np.sqrt(snsValP)/(maxValP*vcell)
else:
    tmpS = np.sqrt(snsValP)/(maxVal)
S = np.reshape(tmpS, dims)# S = np.reshape(tmpS, dims, order="F")/(maxValP*vcell)
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)


elapsed = time.time() - start
print(" Used %7.4f s for writing sensitivities " % (elapsed))
