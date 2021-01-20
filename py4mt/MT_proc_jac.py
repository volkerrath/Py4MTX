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
Reads ModEM's Jacobian, does fancy things.

@author: vr july 2020

Created on Tue Jul  7 16:59:01 2020

@author: vrath

"""

# Import required modules

import os
import sys
# import struct
import time

import numpy as np
import math as ma
import netCDF4 as nc

from modules.jacproc import *
from modules.modem import *
#import readJac, writeJacNC, readDat, writeDatNC, sparsifyJac, readMod, rsvd

normalize_err = True
normalize_max = True
calcsens = True

JacFile = r'/home/vrath/work/MT/Jacobians/Maurienne/Maur_PT.jac'
DatFile = r'/home/vrath/work/MT/Jacobians/Maurienne/Maur_PT.dat'
ModFile = r'/home/vrath/work/MT/Jacobians/Maurienne//Maur_PT_R500_NLCG_016.rho'
SnsFile = r'/home/vrath/work/MT/Jacobians/Maurienne/Maur_PT_R500_NLCG_016.sns'

total = 0.


start = time.time()
dx, dy, dz, rho, reference = readMod(ModFile)
elapsed = (time.time() - start)
total = total + elapsed
print(' Used %7.4f s for reading model from %s ' % (elapsed, DatFile))


start = time.time()
Site, Comp, Data, Head = readDat(DatFile)
elapsed = (time.time() - start)
total = total + elapsed
print(' Used %7.4f s for reading data from %s ' % (elapsed, DatFile))


start = time.time()
name, ext = os.path.splitext(DatFile)
NCFile = name + '_dat.nc'
writeDatNC(NCFile, Data, Site, Comp)
elapsed = (time.time() - start)
total = total + elapsed
print(' Used %7.4f s for writing data to %s ' % (elapsed, NCFile))


start = time.time()
Jac = readJac(JacFile)
elapsed = (time.time() - start)
total = total + elapsed
print(' Used %7.4f s for reading Jacobian from %s ' % (elapsed, JacFile))

# print(np.shape(Data))
# print(np.shape(Jac))


if normalize_err:
    start = time.time()
    dsh = np.shape(Data)
    err = np.reshape(Data[:, 7], (dsh[0], 1))
    Jac = normalizeJac(Jac, err)
    elapsed = (time.time() - start)
    total = total + elapsed
    print(
        ' Used %7.4f s for normalizing Jacobian from %s ' %
        (elapsed, JacFile))


if calcsens:
    start = time.time()
    Sens, Sens_max = calculateSens(Jac, normalize=True)
    elapsed = (time.time() - start)
    total = total + elapsed
    print(' Used %7.4f s for caculating sensitivity from %s ' %
          (elapsed, JacFile))
    sns = np.reshape(Sens, rho.shape)
    print(np.shape(sns))
    writeMod(SnsFile, dx, dy, dz, sns, reference, trans='LOG10', out=True)


start = time.time()
name, ext = os.path.splitext(JacFile)
NCFile = name + '_jac.nc'
writeJacNC(NCFile, Jac, Data, Site, Comp)
elapsed = (time.time() - start)
total = total + elapsed
print(' Used %7.4f s for writing Jacobian to %s ' % (elapsed, NCFile))


start = time.time()
Js = sparsifyJac(Jac)
elapsed = (time.time() - start)
total = total + elapsed
print(' Used %7.4f s for sparsifying Jacobian from %s ' % (elapsed, JacFile))

mu = 0.
sigma = 0.5
r = rho.flat
nproj = 1000

for rank in [50, 100, 200, 400, 1000]:
    start = time.time()
    U, S, Vt = rsvd(Jac.T, rank, n_oversamples=0, n_subspace_iters=0)
    elapsed = (time.time() - start)
    print(' Used %7.4f s for calculating k = %i  SVD from %s ' %
          (elapsed, rank, JacFile))
    # print(U.shape)
    # print(S.shape)
    # print(Vt.shape)
    s = time.time()
    m = r + np.random.normal(mu, sigma, size=np.shape(r))
    t = (time.time() - s)
    print(' Used %7.4f s for generating m  ' % (t))

    s = time.time()
    for proj in range(nproj):
        p = projectMod(m, U)

    t = (time.time() - s)
    print(' Used %7.4f s for %i projections' % (t, nproj))

total = total + elapsed
print(' Total time used:  %f s ' % (total))
