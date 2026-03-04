#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process ModEM Jacobian matrices.

Reads raw Jacobian files, optionally normalises by data error,
applies air masking, sparsifies, and saves in compressed format.

@author: vrath
Cleanup: 4 Mar 2026 by Claude (Anthropic)
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

# =============================================================================
#  Configuration
# =============================================================================
SPARSE_THRESH = 1.0e-6
SPARSE = SPARSE_THRESH > 0

# Set to False if Jacobian originates from ModEM3DJE.x version
ERROR_SCALE = True
SCALE = 1.0

RHOAIR = 1.0e17
RHOSEA = 0.3

WORK_DIR = "/home/vrath/ModEM_work/Ub25_ZT_600_PT_jac/"
J_FILES = [WORK_DIR + "Ub25_Z.jac"]
M_FILE = WORK_DIR + "Ub_600ZT4_PT_NLCG_009"

if not WORK_DIR.endswith("/"):
    WORK_DIR = WORK_DIR + "/"
nF = len(J_FILES)

# =============================================================================
#  Read model
# =============================================================================
total = 0.0
start = time.perf_counter()
dx, dy, dz, rho, reference, _ = mod.read_mod(M_FILE, trans="linear")
dims = np.shape(rho)

aircells = np.where(rho > RHOAIR / 10)
seacells = np.where(rho == RHOSEA)
airmask = jac.set_airmask(rho=rho, aircells=aircells, flat=False, out=True)

elapsed = time.perf_counter() - start
total += elapsed
print(" Used %7.4f s for reading model from %s " % (elapsed, M_FILE))

# =============================================================================
#  Process each Jacobian file
# =============================================================================
for f in np.arange(nF):
    nstr = ""
    name, ext = os.path.splitext(J_FILES[f])

    # Read data
    start = time.perf_counter()
    DFile = J_FILES[f].replace(".jac", "_jac.dat")
    print("\nReading Data from " + DFile)
    Data, Site, Freq, Comp, DTyp, Head = mod.read_data_jac(DFile)
    elapsed = time.perf_counter() - start
    print(" Used %7.4f s for reading Data from %s " % (elapsed, DFile))
    total += elapsed

    # Read Jacobian
    start = time.perf_counter()
    print("Reading Jacobian from " + J_FILES[f])
    Jac, Info = mod.read_jac(J_FILES[f])
    elapsed = time.perf_counter() - start
    print(" Used %7.4f s for reading Jacobian from %s " % (elapsed, J_FILES[f]))
    total += elapsed

    if np.shape(Jac)[0] != np.shape(Data)[0]:
        print(np.shape(Jac), np.shape(Data))
        sys.exit(" Dimensions of Jacobian and data do not match! Exit.")

    mx = np.nanmax(Jac)
    mn = np.nanmin(Jac)
    print(J_FILES[f] + " raw min/max Jacobian value is " + str(mn) + "/" + str(mx))

    # Error-normalize
    if ERROR_SCALE:
        nstr += "_nerr"
        start = time.perf_counter()
        dsh = np.shape(Data)
        err = np.reshape(Data[:, 7], (dsh[0], 1))
        Jac = jac.normalize_jac(Jac, err)

        mx = np.nanmax(Jac)
        mn = np.nanmin(Jac)
        print(J_FILES[f] + " scaled min/max Jacobian value is " + str(mn) + "/" + str(mx))

        elapsed = time.perf_counter() - start
        print(
            " Used %7.4f s for normalizing Jacobian with data error from %s "
            % (elapsed, DFile)
        )

    # Sparsify
    sstr = "_full"
    if SPARSE_THRESH > 0.0:
        sstr = "_sp" + str(round(np.log10(SPARSE_THRESH)))
        start = time.perf_counter()

        for idt in np.arange(np.shape(Jac)[0]):
            JJ = Jac[idt, :].reshape(dims, order="F")
            Jac[idt, :] = (airmask * JJ).flatten(order="F")

        Scale = np.nanmax(np.abs(Jac))
        Jac, Scale = jac.sparsify_jac(
            Jac, scalval=Scale, sparse_thresh=SPARSE_THRESH
        )
        mx = np.nanmax(np.abs(Jac.todense()))
        mn = np.nanmin(np.abs(Jac.todense()))
        print(J_FILES[f] + " sparse min/max Jacobian value is " + str(mn) + "/" + str(mx))

        elapsed = time.perf_counter() - start
        total += elapsed
        print(" Used %7.4f s for sparsifying Jacobian %s " % (elapsed, J_FILES[f]))

    # Save
    name = name + nstr + sstr
    start = time.perf_counter()

    np.savez_compressed(
        name + "_info.npz",
        Freq=Freq, Data=Data, Site=Site, Comp=Comp,
        Info=Info, DTyp=DTyp, Scale=Scale, allow_pickle=True,
    )
    if SPARSE:
        scs.save_npz(name + "_jac.npz", matrix=Jac)
    else:
        np.savez_compressed(name + "_jac.npz", Jac)

    elapsed = time.perf_counter() - start
    total += elapsed
    print(" Used %7.4f s for writing Jacobian and info to %s " % (elapsed, name))

    Jac = None
