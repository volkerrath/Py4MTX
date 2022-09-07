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
normalize_jacmax = "total"
if "loc" in normalize_jacmax.lower():
    jacmax ="maxlocal"
else:
    jacmax ="maxtotal"

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


# # Ubaye case
# WorkDir = PY4MT_DATA+"/NewJacobians/Ubaye/"
# WorkName = "Ub22Jac"
# MFile   = WorkDir +"Ub22.rho"
# MPad=[13, 13 , 13, 13, 0, 36]

# Annecy case
WorkDir = PY4MT_DATA+"/NewJacobians/Annecy/work/"
WorkName = "Ann25Jac"
MFile   = WorkDir +"Ann25.rho"
MPad=[22, 22 , 22, 22, 0, 15]

# # Maurienne case
# WorkDir = PY4MT_DATA+"/NewJacobians/Maurienne/E10/"
# WorkName = "MauJac"
# MFile   = WorkDir +"Maur15_500_PTZ_E10_NLCG_016.rho"
# # MFile   = WorkDir +"Maur15_500_PTZ_E03_NLCG_026.rho"
# MPad=[14, 14 , 14, 14, 0, 15]


JFiles = []
DFiles = []
files = os.listdir(WorkDir)
for entry in files:
    # print(entry)
    if entry.endswith(".jac"):
        JFiles.append(entry)
        name, ext = os.path.splitext(entry)
        DFiles.append(name+"_jac.dat")

JFiles = sorted(JFiles)
DFiles = sorted(DFiles)


NN = np.size(JFiles)
if np.size(DFiles) != np.size(JFiles):
    error("Data file number not equal Jac file number! Exit.")
nF = np.size(DFiles)
print(JFiles)
print(DFiles)

total = 0.0


start = time.time()
dx, dy, dz, rho, reference, _, vcell = mod.read_model(MFile, trans="linear", volumes=True)
dims = np.shape(rho)

rhoair = 1.e17
aircells = np.where(rho>rhoair/10)


TSTFile = WorkDir+WorkName+"0_MaskTest.rho"
mod.write_model(TSTFile, dx, dy, dz, rho, reference, trans="LINEAR", mvalair=blank, aircells=aircells)


jacmask = jac.set_mask(rho=rho, pad=MPad, blank= blank, flat = False, out=True)
jdims= np.shape(jacmask)
j0 = jacmask.reshape(dims)
j0[aircells] = blank
jacmask = j0.reshape(jdims)

rhotest = jacmask.reshape(dims)*rho
TSTFile = WorkDir+WorkName+"1_MaskTest.rho"
mod.write_model(TSTFile, dx, dy, dz, rhotest, reference, trans="LINEAR", mvalair=blank, aircells=aircells)

elapsed = time.time() - start
total = total + elapsed
print(" Used %7.4f s for reading model from %s " % (elapsed, MFile))

sdims = np.size(rho)
snsValT = np.zeros(sdims)
snsValP = np.zeros(sdims)
snsValZ = np.zeros(sdims)
snsValZoff = np.zeros(sdims)
snsValZtot = np.zeros(sdims)


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
    # print(np.amin(err), np.amax(err))
    Jac = jac.normalize_jac(Jac, err)
    elapsed = time.time() - start
    print(" Used %7.4f s for normalizing Jacobian from %s " % (elapsed, JFiles[f]))

    mx = np.nanmax(np.abs(Jac))
    mn = np.nanmin(np.abs(Jac))
    jm = jacmask.flatten(order="F")
    print(JFiles[f]+" minimum/maximum Jacobian value is "+str(mn)+"/"+str(mx))
    mx = np.nanmax(np.abs(Jac*jm))
    mn = np.nanmin(np.abs(Jac*jm))
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

    Component =Info[:,1]

    noZtot = False
    tmpJac = Jac[np.where(Component == 1),:]
    if np.size(tmpJac)== 0:
        noZtot = True
    tmpValZ = np.nansum(np.power(tmpJac, 2), axis=1)
    snsValZtot = snsValZtot + tmpValZ

    tmpJac = Jac[np.where(Component == 2),:]
    noZoff = False
    if np.size(tmpJac)== 0:
        noZoff = True
    tmpValZ = np.nansum(np.power(tmpJac, 2), axis=1)
    snsValZoff = snsValZoff + tmpValZ

    tmpJac = Jac[np.where(Component == 3),:]
    noT = False
    if np.size(tmpJac)== 0:
        noT = True
    tmpValT = np.nansum(np.power(tmpJac, 2), axis=1)
    snsValT = snsValT + tmpValT

    tmpJac = Jac[np.where(Component == 6),:]
    noP = False
    if np.size(tmpJac)== 0:
        noP = True
    tmpValP = np.nansum(np.power(tmpJac, 2), axis=1)
    snsValP = snsValP + tmpValP


maxValZtot = np.nanmax(np.sqrt(snsValZtot*jm))
print(" Merged Z Maximum value is "+str(maxValZtot))
snsValZtot = np.reshape(snsValZtot, dims, order="F")

maxValZoff = np.nanmax(np.sqrt(snsValZoff*jm))
print(" Merged Z Maximum value is "+str(maxValZoff))
snsValZoff = np.reshape(snsValZoff, dims, order="F")

maxValT = np.nanmax(np.sqrt(snsValT*jm))
print(" Merged T Maximum value is "+str(maxValT))
snsValT = np.reshape(snsValT, dims, order="F")

maxValP = np.nanmax(np.sqrt(snsValP*jm))
print(" Merged P Maximum value is "+str(maxValP))
snsValP = np.reshape(snsValP, dims, order="F")

snsVal = snsValZtot+snsValZoff+snsValT+snsValP
maxVal = np.nanmax(snsVal)
print(" Merged ZTP Maximum value is "+str(maxVal))

start = time.time()

"""
Raw sensitivities
"""
SNSFile = WorkDir+WorkName+"_raw.sns"
S = snsVal
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

SNSFile = WorkDir+WorkName+"_Ztot_raw.sns"
S = snsValZtot
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

SNSFile = WorkDir+WorkName+"_Zoff_raw.sns"
S = snsValZoff
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

SNSFile = WorkDir+WorkName+"_T_raw.sns"
S = snsValT
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

SNSFile = WorkDir+WorkName+"_P_raw.sns"
S = snsValP
mod.write_model(SNSFile, dx,dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

# """
# Squareroot Raw sensitivities
# """
SNSFile = WorkDir+WorkName+"_sqrtraw.sns"
S = snsVal
S = np.sqrt(S)
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

SNSFile = WorkDir+WorkName+"_Ztot_sqrtraw.sns"
S = snsValZtot
S = np.sqrt(S)
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

SNSFile = WorkDir+WorkName+"_Zoff_sqrtraw.sns"
S = snsValZoff
S = np.sqrt(S)
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

SNSFile = WorkDir+WorkName+"_T_sqrtraw.sns"
S = snsValT
S = np.sqrt(S)
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

SNSFile = WorkDir+WorkName+"_P_sqrtraw.sns"
S = snsValP
S = np.sqrt(S)
mod.write_model(SNSFile, dx,dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)


# """
# vol normalized sensitivities
# """
SNSFile = WorkDir+WorkName+"_vol.sns"
S = snsVal
S = np.sqrt(S)/vcell
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

SNSFile = WorkDir+WorkName+"_Ztot_vol.sns"
S = snsValZtot
S = np.sqrt(S)/vcell
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

SNSFile = WorkDir+WorkName+"_Zoff_vol.sns"
S = snsValZoff
S = np.sqrt(S)/vcell
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

SNSFile = WorkDir+WorkName+"_T_vol.sns"
S = snsValT
S = np.sqrt(S)/vcell
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

SNSFile = WorkDir+WorkName+"_P_vol.sns"
S = snsValP
S = np.sqrt(S)/vcell
mod.write_model(SNSFile, dx,dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

# """
# Max normalized sensitivities
# """
SNSFile = WorkDir+WorkName+"_max.sns"
S = snsVal
S = np.sqrt(S)/maxVal
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

if noZtot and ("loc" in normalize_jacmax.lower()):
    print(" No Ztot local maximum values available.")

else:
    SNSFile = WorkDir+WorkName+"_Ztot_"+jacmax+".sns"
    S = snsValZtot
    scale =maxVal
    if "loc" in normalize_jacmax.lower():
        scale = maxValZtot
    S = np.sqrt(S)/scale
    mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
    print(" Sensitivities written to "+SNSFile)

if noZoff and ("loc" in normalize_jacmax.lower()):
    print(" No Zoff local maximum values available.")
else:
    SNSFile = WorkDir+WorkName+"_Zoff_"+jacmax+".sns"
    S = snsValZoff
    scale =maxVal
    if "loc" in normalize_jacmax.lower():
        scale = maxValZoff
    S = np.sqrt(S)/scale
    mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
    print(" Sensitivities written to "+SNSFile)

if noT and ("loc" in normalize_jacmax.lower()):
    print(" No T local maximum values available.")
else:
    SNSFile = WorkDir+WorkName+"_T_"+jacmax+".sns"
    S = snsValT
    scale =maxVal
    if "loc" in normalize_jacmax.lower():
        scale = maxValT
    S = np.sqrt(S)/scale
    mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
    print(" Sensitivities written to "+SNSFile)

if noT and ("loc" in normalize_jacmax.lower()) :
    print(" No P local maximum values available.")
else:
    SNSFile = WorkDir+WorkName+"_P_"+jacmax+".sns"
    S = snsValP
    scale =maxVal
    if "loc" in normalize_jacmax.lower():
        scale = maxValP
    S = np.sqrt(S)/scale
    mod.write_model(SNSFile, dx,dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
    print(" Sensitivities written to "+SNSFile)

# """
# Max and vol normalized sensitivities
# """
SNSFile = WorkDir+WorkName+"_max_vol.sns"
S = snsVal
S = np.sqrt(S)/(maxVal*vcell)
mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
print(" Sensitivities written to "+SNSFile)

if noZtot and ("loc" in normalize_jacmax.lower()):
    print(" No Ztot local maximum values available.")
else:
    SNSFile = WorkDir+WorkName+"_Ztot_"+jacmax+"_vol.sns"
    S = snsValZtot
    scale =maxVal
    if "loc" in normalize_jacmax.lower():
        scale = maxValZtot
    S = np.sqrt(S)/(scale*vcell)
    mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
    print(" Sensitivities written to "+SNSFile)

if noZoff and ("loc" in normalize_jacmax.lower()):
    print(" No Zoff local maximum values available.")
else:
    SNSFile = WorkDir+WorkName+"_Zoff_"+jacmax+"_vol.sns"
    S = snsValZoff
    scale =maxVal
    if "loc" in normalize_jacmax.lower():
        scale = maxValZoff
    S = np.sqrt(S)/(scale*vcell)
    mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
    print(" Sensitivities written to "+SNSFile)

if noT and ("loc" in normalize_jacmax.lower()):
    print(" No T local maximum values available.")
else:
    SNSFile = WorkDir+WorkName+"_T_"+jacmax+"_vol.sns"
    S = snsValT
    scale =maxVal
    if "loc" in normalize_jacmax.lower():
        scale = maxValT
    S = np.sqrt(S)/(scale*vcell)
    mod.write_model(SNSFile, dx, dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
    print(" Sensitivities written to "+SNSFile)

if noP and ("loc" in normalize_jacmax.lower()) :
    print(" No P local maximum values available.")
else:
    SNSFile = WorkDir+WorkName+"_P_"+jacmax+"_vol.sns"
    S = snsValP
    scale =maxVal
    if "loc" in normalize_jacmax.lower():
        scale = maxValP
    S = np.sqrt(S)/(scale*vcell)
    mod.write_model(SNSFile, dx,dy, dz, S, reference, trans=outform, mvalair=rhoair, aircells=aircells)
    print(" Sensitivities written to "+SNSFile)



elapsed = time.time() - start
print(" Used %7.4f s for writing sensitivities " % (elapsed))
