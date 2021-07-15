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
print("Merge & Sparsify "+"\n"+"".join("Date " + now.strftime("%m/%d/%Y, %H:%M:%S")))
print("\n\n")



rng = np.random.default_rng()
nan = np.nan

normalize_err = True
normalize_max = True
sparse_first= True


MFile = r"/home/vrath/work/MT/Jacobians/Annecy2021/ANN25a_bestjac/Ann25_best_ZPT.rho"

JFile = [r"/home/vrath/work/MT/Jacobians/Annecy2021/ANN25a_bestjac/Ann25_T.jac",
         r"/home/vrath/work/MT/Jacobians/Annecy2021/ANN25a_bestjac/Ann25_P.jac",]


DFile = [r"/home/vrath/work/MT/Jacobians/Annecy2021/ANN25a_bestjac/Ann25_best_T.dat",
         r"/home/vrath/work/MT/Jacobians/Annecy2021/ANN25a_bestjac/Ann25_best_P.dat",]

SFile = [r"/home/vrath/work/MT/Jacobians/Annecy2021/ANN25a_bestjac/Ann25_best_T.sns",
         r"/home/vrath/work/MT/Jacobians/Annecy2021/ANN25a_bestjac/Ann25_best_P.sns",]



total = 0.0


start = time.time()
dx, dy, dz, rho, reference = mod.read_model(MFile, trans="log10")
elapsed = time.time() - start
total = total + elapsed
print(" Used %7.4f s for reading model from %s " % (elapsed, DFile))


if np.size(DFile) != np.size(JFile):
    error("Data file number not equal Jac file number! Exit.")
nF = np.size(DFile)

for f in np.arange(nF):
    start = time.time()
    print("Reading Data from "+JFile[f])
    Data, Site, Freq, Comp, Head = mod.read_data_jac(DFile[f])
    elapsed = time.time() - start
    print(" Used %7.4f s for reading Data from %s " % (elapsed, JFile))
    total = total + elapsed

    start = time.time()
    print("Reading Jacobian from "+JFile[f])
    Jac = mod.read_jac(JFile[f])
    elapsed = time.time() - start
    print(" Used %7.4f s for reading Jacobian from %s " % (elapsed, JFile))
    total = total + elapsed


# start = time.time()
# Jac = mod.read_jac(JFile)
# elapsed = time.time() - start
# total = total + elapsed
# print(" Used %7.4f s for reading Jacobian from %s " % (elapsed, JFile))

# # print(np.shape(Data))
# # print(np.shape(Jac))


# if normalize_err:
#     start = time.time()
#     dsh = np.shape(Data)
#     err = np.reshape(Data[:, 7], (dsh[0], 1))
#     Jac = jac.normalize_jac(Jac, err)
#     elapsed = time.time() - start
#     total = total + elapsed
#     print(" Used %7.4f s for normalizing Jacobian from %s " % (elapsed, JFile))


# start = time.time()
# name, ext = os.path.splitext(JFile)
# NCFile = name + "_jac.nc"
# mod.write_jac_ncd(NCFile, Jac, Data, Site, Comp)
# elapsed = time.time() - start
# total = total + elapsed
# print(" Used %7.4f s for writing Jacobian to %s " % (elapsed, NCFile))


# start = time.time()
# Js = jac.sparsify_jac(Jac,sparse_thresh=1.e-6)
# elapsed = time.time() - start
# total = total + elapsed
# print(" Used %7.4f s for sparsifying Jacobian from %s " % (elapsed, JFile))

# mu = 0.0
# sigma = 0.5
# r = rho.flat
# nproj = 1000

# rank_results = []
# for rank in [100, 200, 300, 400, 500, 1000]:
#     start = time.time()
#     U, S, Vt = jac.rsvd(Jac.T, rank, n_oversamples=0, n_subspace_iters=0)
#     elapsed = time.time() - start
#     print(
#         "Used %7.4f s for calculating k = %i SVD from %s " % (elapsed, rank, JFile)
#     )

#     D = U@scp.diags(S[:])@Vt - Jac.T

#     x_op = np.random.normal(size=np.shape(D)[1])
#     n_op = npl.norm(D@x_op)/npl.norm(x_op)
#     j_op = npl.norm(Jac.T@x_op)/npl.norm(x_op)
#     print(" Op-norm J_k = "+str(n_op)+", explains "+str(100. - n_op*100./j_op)+"% of J_full")

#     kk= [rank, n_op, j_op, 100. - n_op*100./j_op]

#     rank_results.append(kk)

# Fileout = r"Rank_Results.npz"
# np.savez_compressed(Fileout,
#                     rank_results=rank_results
# )

# rank = 500
# thresh_results = []
# for thresh in [1.e-2, 1.e-4, 1.e-6, 1.e-8]:
#     start = time.time()

#     Js = jac.sparsify_jac(Jac,sparse_thresh=thresh)

#     U, S, Vt = jac.rsvd(Js.T, rank, n_oversamples=0, n_subspace_iters=0)
#     elapsed = time.time() - start
#     print(
#         "Used %7.4f s for thresg = %g SVD from %s " % (elapsed, thresh, JFile)
#     )

#     D = U@scp.diags(S[:])@Vt - Js.T

#     x_op = np.random.normal(size=np.shape(D)[1])
#     n_op = npl.norm(D@x_op)/npl.norm(x_op)
#     j_op = npl.norm(Js.T@x_op)/npl.norm(x_op)
#     print(" Op-norm J_thresh = "+str(n_op)+", explains "+str(100. - n_op*100./j_op)+"% of J_full")

#     kk= [rank, n_op, j_op, 100. - n_op*100./j_op]

#     thresh_results.append(kk)

# Fileout = r"Sparse_Results.npz"
# np.savez_compressed(Fileout,
#                     thresh_results=thresh_results
# )



# # for rank in [50, 100, 200, 400, 1000]:
# #     start = time.time()
# #     U, S, Vt = jac.rsvd(Jac.T, rank, n_oversamples=0, n_subspace_iters=0)
# #     elapsed = time.time() - start
# #     print(
# #         " Used %7.4f s for calculating k = %i  SVD from %s " % (elapsed, rank, JFile)
# #     )
# #     # print(U.shape)
# #     # print(S.shape)
# #     # print(Vt.shape)
# #     s = time.time()
# #     m = r + np.random.normal(mu, sigma, size=np.shape(r))
# #     t = time.time() - s
# #     print(" Used %7.4f s for generating m  " % (t))

# #     s = time.time()
# #     for proj in range(nproj):
# #         p = jac.projectMod(m, U)

# #     t = time.time() - s
# #     print(" Used %7.4f s for %i projections" % (t, nproj))

# # total = total + elapsed
# # print(" Total time used:  %f s " % (total))
