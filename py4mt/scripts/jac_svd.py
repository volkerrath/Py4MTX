#!/usr/bin/env python3

"""
Reads ModEM's Jacobian, does fancy things.

@author: vrath   Feb 2021

"""

# Import required modules

import os
# from https://stackoverflow.com/questions/73391779/setting-number-of-threads-in-python
nthreads = 8  # tinney  62
os.environ["OMP_NUM_THREADS"] = str(nthreads)
os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
os.environ["MKL_NUM_THREADS"] = str(nthreads)

import sys

# import struct
import time
from datetime import datetime
import warnings
import gc

import jax.numpy as nj
import jax.scipy as sj

import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl
import scipy.sparse as scs
import netCDF4 as nc

# from sklearn.utils.extmath import randomized_svd
from numba import njit

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [PY4MTX_ROOT+"/modules/", PY4MTX_ROOT+"/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)


import jacproc as jac
import modem as mod
import util as utl

from version import versionstrg


# RunParallel = False
# if RunParallel:
#     pth = PY4MTX_ROOT+"/external/PyParSVD/pyparsvd/"
#     if pth not in sys.path:
#         sys.path.insert(0,pth)




version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")

rng = np.random.default_rng()
nan = np.nan

# KRAFLA case
# WorkDir = "/media/vrath/BlackOne/MT_Data/Krafla/Krafla1/"
# MFile   = WorkDir +r"Krafla"

# Annecy case
# WorkDir = "/home/vrath/MT_Data/Annecy/Jacobians/"
# MFile = WorkDir+"UBI9_best"
# MOrig = [45.941551, 6.079800] # ANN


# Ubinas case
# WorkDir =  PY4MTX_DATA+
WorkDir = "/home/vrath/UBI38_JAC/"
Orig = [-16.345800 -70.908249] # UBI
JacName = "Ubi38_ZPT_nerr_sp-8"
MFile = WorkDir + "Ubi38_ZssPT_Alpha02_NLCG_023"

# # Misti case
# WorkDir =  PY4MTX_DATA+"/Peru/Misti/"
# MFile = WorkDir+"Misti10_best"
# JFile = WorkDir+"Misti_best_Z5_nerr_sp-8"
# MOrig = [-16.277300, -71.444397]


JFile = WorkDir+JacName
OutName = "_run_subsit"
# NumSingular = [ 100, 200, 300, 400, 500, 1000]
NumSingular = [ 500]
OverSample =  [2]
SubspaceIt = [0]



total = 0.0
start =time.perf_counter()
print("\nReading Data from "+JFile)

Jac = scs.load_npz(JFile +"_jac.npz")
Dat = np.load(JFile +"_info.npz", allow_pickle=True)

Freq = Dat["Freq"]
Comp = Dat["Comp"]
Site = Dat["Site"]
DTyp = Dat["DTyp"]
Data = Dat["Data"]
Scale = Dat["Scale"]
Info = Dat["Info"]

elapsed = time.perf_counter() - start
total = total + elapsed
print(" Used %7.4f s for reading Jacobian from %s " % (elapsed, JFile))

nsingval = NumSingular[0]
noversmp = OverSample[0]
nsubspit = SubspaceIt[0]

info = []
for noversmp in OverSample:
    for nsubspit in SubspaceIt:
        for nsingval in NumSingular:

            start = time.perf_counter()
            U, S, Vt = jac.rsvd(Jac.T,
                                rank=nsingval,
                                n_oversamples=noversmp*nsingval,
                                n_subspace_iters=nsubspit)
            elapsed = time.perf_counter() - start
            print("Used %7.4f s for calculating k = %i SVD " % (elapsed, nsingval))
            print("Oversamplinng factor =  ", str(noversmp))
            print("Subspace iterations  =  ", str(nsubspit))

            D = U@scs.diags(S[:])@Vt - Jac.T
            x_op = np.random.default_rng().normal(size=np.shape(D)[1])
            n_op = npl.norm(D@x_op)/npl.norm(x_op)
            j_op = npl.norm(Jac.T@x_op)/npl.norm(x_op)
            perc = 100. - n_op*100./j_op
            tmp = [nsingval, noversmp, nsubspit, perc, elapsed]
            info.append(tmp)

            File = JFile+"_SVD_k"+str(nsingval)\
                    +"_o"+str(noversmp)\
                    +"_s"+str(nsubspit)\
                    +"_"+str(np.around(perc,1))\
                    +"percent.npz"
            np.savez_compressed(File, U=U, S=S, V=Vt, Nop=perc)


            # start = time.perf_counter()
            # U, S, Vt = randomized_svd(M=Jac,
            #                     n_components=nsingval,
            #                     n_oversamples=noversmp*nsingval)
            # elapsed = time.perf_counter() - start
            # print("Used %7.4f s for calculating k = %i SVD " % (elapsed, nsingval))
            # print("Oversamplinng factor =  ", str(noversmp))
            # print("Subspace iterations  =  ", str(nsubspit))

            # D = U@scs.diags(S[:])@Vt - Jac.T
            # x_op = np.random.default_rng().normal(size=np.shape(D)[1])
            # n_op = npl.norm(D@x_op)/npl.norm(x_op)
            # j_op = npl.norm(Jac.T@x_op)/npl.norm(x_op)
            # perc = 100. - n_op*100./j_op
            # tmp = [nsingval, noversmp, nsubspit, perc, elapsed]
            # info.append(tmp)
            # print(" Op-norm J_k = "+str(n_op)+", explains "
            #     +str(perc)+"% of variations")
            # print("")

            # File = JFile+"_SVD_k"+str(nsingval)\
            #         +"_o"+str(noversmp)\
            #         +"_s"+str(nsubspit)\
            #         +"_"+str(np.around(perc,1))\
            #         +"percent.npz"

            # np.savez_compressed(File, U=U, S=S, V=Vt, Nop=perc)

np.savetxt(JFile+OutName+".dat",  np.vstack(info),
                fmt="%6i, %6i, %6i, %4.6g, %4.6g")
