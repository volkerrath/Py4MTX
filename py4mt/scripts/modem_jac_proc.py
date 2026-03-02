#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process ModEM Jacobian matrices.

Reads raw Jacobian files, optionally normalises by data error,
applies air masking, sparsifies, and saves in compressed format.

@author: vrath
"""

import os
import sys
import inspect
import time

import numpy as np
import scipy.sparse as scs

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [PY4MTX_ROOT + "/py4mt/modules/", PY4MTX_ROOT + "/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)

import util as utl
from version import versionstrg
import modem as mod
import jac_proc as jac

version, _ = versionstrg()
titstrng = utl.print_title(
    version=version, fname=inspect.getfile(inspect.currentframe()), out=False
)
print(titstrng + "\n\n")

rng = np.random.default_rng()
nan = np.nan

# =============================================================================
#  Configuration
# =============================================================================
SparseThresh = 1.0e-6
Sparse = SparseThresh > 0

# Set to False if Jacobian originates from ModEM3DJE.x version
ErrorScale = True
Scale = 1.0

WorkDir = "/home/vrath/ModEM_work/Ub25_ZT_600_PT_jac/"
JFiles = [WorkDir + "Ub25_Z.jac"]
MFile = WorkDir + "Ub_600ZT4_PT_NLCG_009"

if not WorkDir.endswith("/"):
    WorkDir = WorkDir + "/"
nF = len(JFiles)

# =============================================================================
#  Read model
# =============================================================================
total = 0.0
start = time.perf_counter()
dx, dy, dz, rho, reference, _ = mod.read_mod(MFile, trans="linear")
dims = np.shape(rho)

rhoair = 1.0e17
rhosea = 0.3

aircells = np.where(rho > rhoair / 10)
seacells = np.where(rho == rhosea)
airmask = jac.set_airmask(rho=rho, aircells=aircells, flat=False, out=True)

elapsed = time.perf_counter() - start
total += elapsed
print(" Used %7.4f s for reading model from %s " % (elapsed, MFile))

# =============================================================================
#  Process each Jacobian file
# =============================================================================
for f in np.arange(nF):
    nstr = ""
    name, ext = os.path.splitext(JFiles[f])

    # Read data
    start = time.perf_counter()
    DFile = JFiles[f].replace(".jac", "_jac.dat")
    print("\nReading Data from " + DFile)
    Data, Site, Freq, Comp, DTyp, Head = mod.read_data_jac(DFile)
    elapsed = time.perf_counter() - start
    print(" Used %7.4f s for reading Data from %s " % (elapsed, DFile))
    total += elapsed

    # Read Jacobian
    start = time.perf_counter()
    print("Reading Jacobian from " + JFiles[f])
    Jac, Info = mod.read_jac(JFiles[f])
    elapsed = time.perf_counter() - start
    print(" Used %7.4f s for reading Jacobian from %s " % (elapsed, JFiles[f]))
    total += elapsed

    if np.shape(Jac)[0] != np.shape(Data)[0]:
        print(np.shape(Jac), np.shape(Data))
        sys.exit(" Dimensions of Jacobian and data do not match! Exit.")

    mx = np.nanmax(Jac)
    mn = np.nanmin(Jac)
    print(JFiles[f] + " raw min/max Jacobian value is " + str(mn) + "/" + str(mx))

    # Error-normalize
    if ErrorScale:
        nstr += "_nerr"
        start = time.perf_counter()
        dsh = np.shape(Data)
        err = np.reshape(Data[:, 7], (dsh[0], 1))
        Jac = jac.normalize_jac(Jac, err)

        mx = np.nanmax(Jac)
        mn = np.nanmin(Jac)
        print(JFiles[f] + " scaled min/max Jacobian value is " + str(mn) + "/" + str(mx))

        elapsed = time.perf_counter() - start
        print(
            " Used %7.4f s for normalizing Jacobian with data error from %s "
            % (elapsed, DFile)
        )

    # Sparsify
    sstr = "_full"
    if SparseThresh > 0.0:
        sstr = "_sp" + str(round(np.log10(SparseThresh)))
        start = time.perf_counter()

        for idt in np.arange(np.shape(Jac)[0]):
            JJ = Jac[idt, :].reshape(dims, order="F")
            Jac[idt, :] = (airmask * JJ).flatten(order="F")

        Scale = np.nanmax(np.abs(Jac))
        Jac, Scale = jac.sparsify_jac(
            Jac, scalval=Scale, sparse_thresh=SparseThresh
        )
        mx = np.nanmax(np.abs(Jac.todense()))
        mn = np.nanmin(np.abs(Jac.todense()))
        print(JFiles[f] + " sparse min/max Jacobian value is " + str(mn) + "/" + str(mx))

        elapsed = time.perf_counter() - start
        total += elapsed
        print(" Used %7.4f s for sparsifying Jacobian %s " % (elapsed, JFiles[f]))

    # Save
    name = name + nstr + sstr
    start = time.perf_counter()

    np.savez_compressed(
        name + "_info.npz",
        Freq=Freq, Data=Data, Site=Site, Comp=Comp,
        Info=Info, DTyp=DTyp, Scale=Scale, allow_pickle=True,
    )
    if Sparse:
        scs.save_npz(name + "_jac.npz", matrix=Jac)
    else:
        np.savez_compressed(name + "_jac.npz", Jac)

    elapsed = time.perf_counter() - start
    total += elapsed
    print(" Used %7.4f s for writing Jacobian and info to %s " % (elapsed, name))

    Jac = None
