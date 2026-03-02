#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute statistics on ModEM Jacobian matrices.

Splits the Jacobian by component, site, and/or frequency band and
prints summary statistics for each subset.

@author: vrath
"""

import os
import sys
import inspect
import time
import gc

import numpy as np
import scipy.sparse as scs

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [PY4MTX_ROOT + "/py4mt/modules/", PY4MTX_ROOT + "/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)

import jac_proc as jac
import modem as mod
from version import versionstrg
import util as utl

version, _ = versionstrg()
titstrng = utl.print_title(
    version=version, fname=inspect.getfile(inspect.currentframe()), out=False
)
print(titstrng + "\n\n")

gc.enable()
rng = np.random.default_rng()
blank = 1.0e-30
rhoair = 1.0e17

# =============================================================================
#  Configuration
# =============================================================================
InpFormat = "sparse"

WorkDir = PY4MTX_DATA + "/Peru/Ubinas/"
if not WorkDir.endswith("/"):
    WorkDir = WorkDir + "/"

MFile = WorkDir + "UBI9_best"
JacName = "UBI9_ZPTss_nerr_sp-8"
JFile = WorkDir + JacName

OutFile = JFile + "_stats.dat"
ofile = open(OutFile, "w")

Splits = ["comp", "site", "freq"]
PerIntervals = [
    [0.0001, 0.001], [0.001, 0.01], [0.01, 0.1], [0.1, 1.0],
    [1.0, 10.0], [10.0, 100.0], [100.0, 1000.0], [1000.0, 10000.0],
]

# =============================================================================
#  Read model
# =============================================================================
total = 0.0
start = time.perf_counter()
dx, dy, dz, rho, refmod, _ = mod.read_mod(MFile, trans="linear", volumes=True)
elapsed = time.perf_counter() - start
total += elapsed
print(" Used %7.4f s for reading model from %s " % (elapsed, MFile))

dims = np.shape(rho)
aircells = np.where(rho > rhoair / 10)
jacmask = jac.set_airmask(rho=rho, aircells=aircells, blank=blank, flat=False, out=True)
jdims = np.shape(jacmask)
j0 = jacmask.reshape(dims)
j0[aircells] = blank
jacmask = j0.reshape(jdims)
jacflat = jacmask.flatten(order="F")

# =============================================================================
#  Read Jacobian
# =============================================================================
start = time.perf_counter()
print("Reading Jacobian from " + JFile)

if "spa" in InpFormat:
    Jac = scs.load_npz(JFile + "_jac.npz")
    tmp = np.load(JFile + "_info.npz", allow_pickle=True)
    Freqs = tmp["Freq"]
    Comps = tmp["Comp"]
    Sites = tmp["Site"]
    Dtype = tmp["DTyp"]
else:
    Jac, tmp = mod.read_jac(JFile + ".jac")
    Data, Site, Freq, Comp, Dtype, Head = mod.read_data_jac(JFile + "_jac.dat")
    dsh = np.shape(Data)
    err = np.reshape(Data[:, 5], (dsh[0], 1))
    Jac = jac.normalize_jac(Jac, err)
    Freqs, Comps, Sites = Freq, Comp, Site

print("Full Jacobian")
ofile.write("Full Jacobian")
jac.print_stats(jac=Jac, jacmask=jacflat, outfile=ofile)
print("\n")

# =============================================================================
#  Compute stats by splits
# =============================================================================
for Split in Splits:

    if "comp" in Split.lower():
        start = time.perf_counter()
        ExistType = np.unique(Dtype)
        for icmp in ExistType:
            indices = np.where(Dtype == icmp)
            JacTmp = Jac[indices]
            print("Component: ", icmp)
            ofile.write("\n Component: " + str(icmp))
            jac.print_stats(jac=JacTmp, jacmask=jacflat, outfile=ofile)
            print()

    if "site" in Split.lower():
        start = time.perf_counter()
        SiteNames = Sites[np.sort(np.unique(Sites, return_index=True)[1])]
        for sit in SiteNames:
            indices = np.where(sit == Sites)
            JacTmp = Jac[indices]
            print("Site: ", sit)
            ofile.write("\n Site: " + sit)
            jac.print_stats(jac=JacTmp, jacmask=jacflat, outfile=ofile)
            print()

    if "freq" in Split.lower():
        start = time.perf_counter()
        for ibnd in np.arange(len(PerIntervals)):
            lowstr = str(1.0 / PerIntervals[ibnd][0]) + "Hz"
            uppstr = str(1.0 / PerIntervals[ibnd][1]) + "Hz"
            indices = np.where(
                (Freqs >= PerIntervals[ibnd][0]) & (Freqs < PerIntervals[ibnd][1])
            )
            JacTmp = Jac[indices]
            if np.shape(JacTmp)[0] > 0:
                print("Freqband: ", lowstr, "to", uppstr)
                ofile.write("\n Freqband: " + lowstr + " to " + uppstr)
                jac.print_stats(jac=JacTmp, jacmask=jacflat, outfile=ofile)
                print()
            else:
                print("Frequency band is empty! Continue.")

print("\nDone!")
ofile.close()
