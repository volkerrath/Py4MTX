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


import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl
import scipy.sparse as scs
import netCDF4 as nc


# import vtk
# import pyvista as pv
# import PVGeo as pvg


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



rng = np.random.default_rng()
nan = np.nan

normalize_err = True
normalize_max = False
normalize_vol = True

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
WorkDir = r"/media/vrath/BlackOne/MT_Data/Krafla/Krafla1/"
MFile   = WorkDir +r"Krafla.rho"
MPad=[15, 15 , 15, 15, 0, 36]
JFiles = []
DFiles = []
files = os.listdir(WorkDir)
for entry in files:
    # print(entry)
    if entry.endswith(".jac"):
        JFiles.append(entry)
        name, ext = os.path.splitext(entry)
        DFiles.append(name+".dat")

NN = np.size(JFiles)
if np.size(DFiles) != np.size(JFiles):
    error("Data file number not equal Jac file number! Exit.")
nF = np.size(DFiles)


total = 0.0


start = time.time()
dx, dy, dz, rho, reference, _, vcell = mod.read_model(MFile, trans="linear", volumes=True)
dims = np.shape(rho)

resair = 1.e17
aircells = np.where(rho>resair/100)

jacmask = jac.set_mask(rho=rho, pad=MPad, flat = True, out=True)
jdims= np.shape(jacmask)
j0 = jacmask.reshape(dims)
j0[aircells] = nan
jacmask = j0.reshape(jdims)


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

    nstr = ""
    if normalize_err:
        nstr = nstr+"_nerr"
        start = time.time()
        dsh = np.shape(Data)
        err = np.reshape(Data[:, 7], (dsh[0], 1))
        Jac = jac.normalize_jac(Jac, err)
        elapsed = time.time() - start
        print(" Used %7.4f s for normalizing Jacobian from %s " % (elapsed, JFiles[f]))

    mx = np.nanmax(np.abs(Jac))
    print(JFiles[f]+" maximum Jacobian value is "+str(mx))
    Jac = Jac*jacmask
    mx = np.nanmax(np.abs(Jac))
    print(JFiles[f]+" maximum masked Jacobian value is "+str(mx))
    print(JFiles[f]+" number of elements in maskedJacobian is "+str( np.count_nonzero(~np.isnan(Jac))))
    print( np.count_nonzero(~np.isnan(jacmask))*np.shape(Jac)[0])

    """
    Full_Impedance              = 1
    Off_Diagonal_Impedance      = 2
    Full_Vertical_Components    = 3
    Full_Interstation_TF        = 4
    Off_Diagonal_Rho_Phase      = 5
    Phase_Tensor                = 6
    """

    I1 =Info[:,1]
    tmpJac = Jac[np.where(I1 == 1),:]
    print(np.shape(tmpJac))
    tmpValZ = np.nansum(np.power(tmpJac, 2), axis=1)
    print(np.shape(tmpValZ))
    tmpValZ
    snsValZ = snsValZ + tmpValZ
    maxValZ = np.sqrt(np.nanmean(tmpValZ))
    print(" Z Maximum value is "+str(maxValZ))

    tmpJac = Jac[np.where(I1 == 3),:]
    tmpValT = np.nansum(np.power(tmpJac, 2), axis=1)
    snsValT = snsValT + tmpValT
    maxValT = np.sqrt(np.nanmean(tmpValT))
    print(" T Maximum value is "+str(maxValT))

    tmpJac = Jac[np.where(I1 == 6),:]
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





# for f in np.arange(nF):

#     name, ext = os.path.splitext(JFiles[f])
#     start =time.time()
#     print("\nReading Data from "+DFiles[f])
#     Data, Site, Freq, Comp, Head = mod.read_data_jac(DFiles[f])
#     elapsed = time.time() - start
#     print(" Used %7.4f s for reading Data from %s " % (elapsed, DFiles[f]))
#     total = total + elapsed

#     start = time.time()
#     print("Reading Jacobian from "+JFiles[f])
#     Jac, Info = mod.read_jac(JFiles[f])
#     elapsed = time.time() - start
#     print(" Used %7.4f s for reading Jacobian from %s " % (elapsed, JFiles[f]))
#     total = total + elapsed

#     nstr = ""
#     if normalize_err:
#         nstr = nstr+"_nerr"
#         start = time.time()
#         dsh = np.shape(Data)
#         err = np.reshape(Data[:, 7], (dsh[0], 1))
#         mx0 = np.nanmax(np.abs(Jac))
#         Jac = jac.normalize_jac(Jac, err)
#         elapsed = time.time() - start
#         print(" Used %7.4f s for normalizing Jacobian from %s " % (elapsed, JFiles[f]))

#     mx = np.nanmax(np.abs(Jac*jacmask))
#     mxLst.append(mx)
#     mxVal = np.amax([mxVal,mx])

#     NPZFile = name+"_info.npz"
#     np.savez(NPZFile, Info=Info,  Data=Data, Site=Site, Comp=Comp, MaxVal=mxVal)


#     if normalize_max:
#         nstr = nstr+"_max"
#         start = time.time()
#         Jac = jac.normalize_jac(Jac,[mx])
#         elapsed = time.time() - start
#         total = total + elapsed
#         print(" Max value is %7.4f, before was %7.4f" % (mx, mx0))


#     sstr=""
#     if sparsify:
#         sstr="_sp"+str(round(np.log10(sparse_thresh)))
#         start = time.time()
#         Jacs, _ = jac.sparsify_jac(Jac,sparse_thresh=sparse_thresh)
#         elapsed = time.time() - start
#         total = total + elapsed
#         print(" Used %7.4f s for sparsifying Jacobian %s " % (elapsed, JFiles[f]))
#         NPZFile = name+nstr+sstr+"_jacs.npz"
#         scs.save_npz(NPZFile, Jacs)
#         NPZFile = name+nstr+sstr+"_dats.npz"
#         np.savez(NPZFile, Data=Data, Site=Site, Comp=Comp)
#         elapsed = time.time() - start
#         total = total + elapsed
#         print(" Used %7.4f s for writing sparsified Jacobian to %s " % (elapsed, name+nstr+sstr))
#         if f==0:
#             JacStack = Jacs.copy()
#         else:
#             JacStack = scs.vstack([JacStack, Jacs.copy()])




#     start = time.time()
#     NCFile = name + nstr+".nc"
#     mod.write_jac_ncd(NCFile, Jac, Data, Site, Comp)
#     elapsed = time.time() - start
#     total = total + elapsed
#     print(" Used %7.4f s for writing Jacobian to %s " % (elapsed, NCFile))


#     start = time.time()
#     S0, Smax = jac.calculate_sens(Jac*jacmask, normalize=False, small=1.0e-12)
#     print(" Max sensitivity value is %7.4f " % (Smax))
#     S1 = S0.reshape(np.shape(rho))
#     if normalize_vol:
#         Sz = np.divide(S1,vcell)
#     print(" Max normalized sensitivity value is %7.4f " % (np.nanmax(abs(Sz))))

#     elapsed = time.time() - start
#     total = total + elapsed
#     print(" Used %7.4f s for calculating Sensitivity from Jacobian  %s " % (elapsed, JFiles[f]))
# #


#     start = time.time()
#     SNSFile = name+nstr+".sns"
#     S = np.reshape(Sz, dims, order="F")
#     mod.write_model(SNSFile, dx, dy, dz, S, reference, trans="linear", air=aircells)
#     elapsed = time.time() - start
#     total = total + elapsed
#     print(" Used %7.4f s for writing Sensitivity from Jacobian  %s " % (elapsed, JFiles[f]))


# NPZFile = name+nstr+sstr+"_JacStack_jacs.npz"

# scs.save_npz(NPZFile, Jacs)
# NPZFile = name+nstr+sstr+"_JacStack_dats.npz"
# np.savez(NPZFile, Data=Data, Site=Site, Comp=Comp)
# elapsed = time.time() - start
# total = total + elapsed
# print(" Used %7.4f s for writing sparsified Jacobian to %s " % (elapsed, name+nstr+sstr+"_JacStack"))

# print("\n\nUsed %7.4f s for processing Jacobian." % (total))
