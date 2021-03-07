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
#       jupytext_version: 1.10.2
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


import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl
import scipy.sparse as scp
import netCDF4 as nc


import vtk
import pyvista as pv
import PVGeo as pvg


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
print("Nullspace Shuttle"+"\n"+"".join("Date " + now.strftime("%m/%d/%Y, %H:%M:%S")))
print("\n\n")

rng = np.random.default_rng()
nan = np.nan

JFile = r"/home/vrath/Py4MT/py4mt/data/ANN21_Jacobian/Ann21_Prior100_T-T3.jac"
DFile = r"/home/vrath/Py4MT/py4mt/data/ANN21_Jacobian/Ann21_Prior100_T_NLCG_033.dat"
MFile = r"/home/vrath/Py4MT/py4mt/data/ANN21_Jacobian/Ann21_Prior100_T_NLCG_033.rho"
SFile = r"/home/vrath/Py4MT/py4mt/data/ANN21_Jacobian/Ann21_Prior100_T-Z3.sns"

NSamples = 1000
NBodies  = 32
x_bounds = [-3000., 3000.]
y_bounds = [-3000., 3000.]
z_bounds = [-300., 3000.]
rad_bounds = [100.,1000.]
res_bounds = [-0.3, 0.3]



total = 0.0
start = time.time()
dx, dy, dz, rho, reference = mod.read_model(MFile)
elapsed = time.time() - start
total = total + elapsed
print(" Used %7.4f s for reading model from %s " % (elapsed, DFile))


start = time.time()
Site, Comp, Data, Head = mod.read_data(DFile)
elapsed = time.time() - start
total = total + elapsed
print(" Used %7.4f s for reading data from %s " % (elapsed, DFile))

start = time.time()
Jac = mod.read_jac(JFile)
elapsed = time.time() - start
total = total + elapsed
print(" Used %7.4f s for reading Jacobian from %s " % (elapsed, JFile))

start = time.time()
dsh = np.shape(Data)
err = np.reshape(Data[:, 7], (dsh[0], 1))
Jac = jac.normalize_jac(Jac, err)
elapsed = time.time() - start
total = total + elapsed
print(" Used %7.4f s for normalizing Jacobian from %s " % (elapsed, JFile))

start = time.time()
Js = jac.sparsify_jac(Jac, sparse_thresh=1.e-4)
elapsed = time.time() - start
total = total + elapsed
print(" Used %7.4f s for sparsifying Jacobian from %s " % (elapsed, JFile))

mu = 0.0
sigma = 0.5
r = rho.flat
nproj = 1000

rank = 500
start = time.time()
U, S, Vt = jac.rsvd(Jac.T, rank, n_oversamples=0, n_subspace_iters=0)
elapsed = time.time() - start
print(
    "Used %7.4f s for calculating k = %i SVD from %s " % (elapsed, rank, JFile)
)

D = U@scp.diags(S[:])@Vt - Jac.T
x_op = np.random.normal(size=np.shape(D)[1])
n_op = npl.norm(D@x_op)/npl.norm(x_op)
j_op = npl.norm(Jac.T@x_op)/npl.norm(x_op)
print(" Op-norm J_k = "+str(n_op)+", explains "
      +str(100. - n_op*100./j_op)+"% of variations")


# m_avg = 0.
# v_avg = 0.
# s = time.time()
for isample in np.arange(NSamples):

    body = [
    "ellipsoid",
    "add",
    0.,
    0.,
    0.,
    3000.,
    1000.,
    2000.,
    1000.,
    0.,
    0.,
    30.]

# m = r + np.random.normal(mu, sigma, size=np.shape(r))
#     t = time.time() - s
#     print(" Used %7.4f s for generating m  " % (t))

#     s = time.time()
#     for proj in range(nproj):
#         p = jac.projectMod(m, U)

#     t = time.time() - s
#     print(" Used %7.4f s for %i projections" % (t, nproj))

# total = total + elapsed
# print(" Total time used:  %f s " % (total))
