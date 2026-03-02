#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split or merge ModEM Jacobian matrices.

Merge: combine Jacobians from separate data types (Z, P, T) into one.
Split: separate a merged Jacobian by frequency band, component, data type,
       or site.

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

import jac_proc as jac
import modem as mod
import util as utl
from version import versionstrg

version, _ = versionstrg()
titstrng = utl.print_title(
    version=version, fname=inspect.getfile(inspect.currentframe()), out=False
)
print(titstrng + "\n\n")

# =============================================================================
#  Configuration
# =============================================================================
WorkDir = PY4MTX_DATA + "/Peru/Sababcaya/SABA8_Jac/"
if not WorkDir.endswith("/"):
    WorkDir = WorkDir + "/"

Task = "merge"

# Merge settings
MergedFile = "SABA8_ZPT__nerr_sp-8_merged"
MFiles = [
    WorkDir + "SABA8_Z_nerr_sp-8",
    WorkDir + "SABA8_P_nerr_sp-8",
    WorkDir + "SABA8_T_nerr_sp-8",
]
nF = np.size(MFiles)

# Split settings
SFile = WorkDir + "merged/UBI_ZPT_sp-8_merged"
Split = "dtype  site  freq  comp"

PerIntervals = [
    [0.0001, 0.001],
    [0.001, 0.01],
    [0.01, 0.1],
    [0.1, 1.0],
    [1.0, 10.0],
    [10.0, 100.0],
    [100.0, 1000.0],
    [1000.0, 10000.0],
]

# =============================================================================
#  Merge
# =============================================================================
if "mer" in Task.lower():
    print(" The following files will be merged:")
    print(MFiles)

    for ifile in np.arange(nF):
        start = time.perf_counter()
        print("\nReading Data from " + MFiles[ifile])
        Jac = scs.load_npz(MFiles[ifile] + "_jac.npz")

        tmp = np.load(MFiles[ifile] + "_info.npz", allow_pickle=True)
        Freq = tmp["Freq"]
        Comp = tmp["Comp"]
        Site = tmp["Site"]
        DTyp = tmp["DTyp"]
        Data = tmp["Data"]
        Scale = tmp["Scale"]
        Info = tmp["Info"]

        elapsed = time.perf_counter() - start
        print(" Used %7.4f s for reading Jacobian from %s " % (elapsed, MFiles[ifile]))

        if ifile == 0:
            Jac_merged = Jac
            Data_merged = Data
            Site_merged = Site
            Freq_merged = Freq
            Comp_merged = Comp
            DTyp_merged = DTyp
            Infblk = Info
            Scales = Scale
        else:
            Jac_merged = scs.vstack((Jac_merged, Jac))
            Data_merged = np.vstack((Data_merged, Data))
            Site_merged = np.hstack((Site_merged, Site))
            Freq_merged = np.hstack((Freq_merged, Freq))
            Comp_merged = np.hstack((Comp_merged, Comp))
            DTyp_merged = np.hstack((DTyp_merged, DTyp))
            Infblk = np.vstack((Infblk, Info))
            Scales = np.hstack((Scales, Scale))

        print(ifile, type(Jac_merged), np.shape(Jac_merged))

    start = time.perf_counter()
    np.savez_compressed(
        WorkDir + MergedFile + "_info.npz",
        Freq=Freq_merged, Data=Data_merged, Site=Site_merged,
        Comp=Comp_merged, Info=Infblk, DTyp=DTyp_merged,
        Scale=Scales, allow_pickle=True,
    )
    scs.save_npz(
        WorkDir + MergedFile + "_jac.npz", matrix=Jac_merged, compressed=True
    )
    elapsed = time.perf_counter() - start
    print(" Used %7.4f s for storing Jacobian to %s " % (elapsed, WorkDir + MergedFile))


# =============================================================================
#  Split
# =============================================================================
if "spl" in Task.lower():
    print("\nReading Data from " + SFile)
    start = time.perf_counter()
    Jac = scs.load_npz(SFile + "_jac.npz")
    sparse = scs.issparse(Jac)
    tmp = np.load(SFile + "_info.npz", allow_pickle=True)
    Freq = tmp["Freq"]
    Comp = tmp["Comp"]
    Site = tmp["Site"]
    DTyp = tmp["DTyp"]
    Data = tmp["Data"]
    Scal = tmp["Scale"]
    Info = tmp["Info"]

    tmpinfo = np.reshape(Info, (len(Info), 3))
    freq = tmpinfo[:, 0]
    jcmp = tmpinfo[:, 1]

    # ---- Split by frequency band ----
    if "fre" in Split.lower():
        start = time.perf_counter()
        nF = len(PerIntervals)

        for ibnd in np.arange(nF):
            lowstr = str(1.0 / PerIntervals[ibnd][0]) + "Hz"
            uppstr = str(1.0 / PerIntervals[ibnd][1]) + "Hz"

            indices = np.where(
                (Freq >= PerIntervals[ibnd][0]) & (Freq < PerIntervals[ibnd][1])
            )
            JacTmp = Jac[indices]
            FreqTmp = Freq[indices]
            DataTmp = Data[indices, :]
            SiteTmp = Site[indices]
            CompTmp = Comp[indices]
            InfoTmp = Info[indices]
            DTypTmp = DTyp[indices]
            ScalTmp = Scal

            Name = SFile + "_freqband" + lowstr + "_to_" + uppstr
            np.savez_compressed(
                Name + "_info.npz",
                Freq=FreqTmp, Data=DataTmp, Site=SiteTmp,
                Comp=CompTmp, Info=InfoTmp, DTyp=DTypTmp,
                Scale=ScalTmp, allow_pickle=True,
            )
            if scs.issparse(JacTmp):
                scs.save_npz(Name + "_jac.npz", matrix=JacTmp)
            else:
                np.savez_compressed(Name + "_jac.npz", JacTmp)

        elapsed = time.perf_counter() - start
        print(" Used %7.4f s for splitting into frequency bands " % elapsed)

    # ---- Split by component ----
    if "comp" in Split.lower():
        start = time.perf_counter()
        compstr = [
            "zxy", "zyx", "zxx", "zyy",
            "txr", "tyr", "txi", "txr",
            "ptxy", "ptyx", "ptxx", "ptyy",
        ]

        ExistComp = np.unique(Comp)
        for icmp in ExistComp:
            # BUG FIX: was `=` (assignment), now `==` (comparison)
            indices = np.where(jcmp == icmp)
            JacTmp = Jac[indices]
            FreqTmp = Freq[indices]
            DataTmp = Data[indices, :]
            SiteTmp = Site[indices]
            CompTmp = Comp[indices]
            InfoTmp = Info[indices]
            DTypTmp = DTyp[indices]
            ScalTmp = Scal

            Name = SFile + "_Comp" + compstr[icmp - 1]
            np.savez_compressed(
                Name + "_info.npz",
                Freq=FreqTmp, Data=DataTmp, Site=SiteTmp,
                Comp=CompTmp, Info=InfoTmp, DTyp=DTypTmp,
                Scale=ScalTmp, allow_pickle=True,
            )
            if scs.issparse(JacTmp):
                scs.save_npz(Name + "_jac.npz", matrix=JacTmp)
            else:
                np.savez_compressed(Name + "_jac.npz", JacTmp)

        elapsed = time.perf_counter() - start
        print(" Used %7.4f s for splitting into components " % elapsed)

    # ---- Split by data type ----
    if "dtyp" in Split.lower():
        start = time.perf_counter()
        typestr = ["zfull", "zoff", "tp", "mf", "rpoff", "pt"]

        ExistType = np.unique(DTyp)
        for tnum, ityp in enumerate(ExistType):
            # BUG FIX: was `=` (assignment), now `==` (comparison)
            indices = np.where(jcmp == ityp)
            JacTmp = Jac[indices]
            FreqTmp = Freq[indices]
            DataTmp = Data[indices, :]
            SiteTmp = Site[indices]
            CompTmp = Comp[indices]
            InfoTmp = Info[indices]
            DTypTmp = DTyp[indices]
            ScalTmp = Scal[tnum] if np.ndim(Scal) > 0 else Scal

            Name = SFile + "_DType" + typestr[ityp - 1]
            np.savez_compressed(
                Name + "_info.npz",
                Freq=FreqTmp, Data=DataTmp, Site=SiteTmp,
                Comp=CompTmp, Info=InfoTmp, DTyp=DTypTmp,
                Scale=ScalTmp, allow_pickle=True,
            )
            if scs.issparse(JacTmp):
                scs.save_npz(Name + "_jac.npz", matrix=JacTmp)
            else:
                np.savez_compressed(Name + "_jac.npz", JacTmp)

        elapsed = time.perf_counter() - start
        print(" Used %7.4f s for splitting into data types " % elapsed)

    # ---- Split by site ----
    if "sit" in Split.lower():
        start = time.perf_counter()
        SiteNames = Site[np.sort(np.unique(Site, return_index=True)[1])]

        for sit in SiteNames:
            indices = np.where(sit == Site)
            JacTmp = Jac[indices]
            FreqTmp = Freq[indices]
            DataTmp = Data[indices, :]
            SiteTmp = Site[indices]
            CompTmp = Comp[indices]
            InfoTmp = Info[indices]
            DTypTmp = DTyp[indices]
            ScalTmp = Scal

            Name = SFile + "_" + sit.lower()
            np.savez_compressed(
                Name + "_info.npz",
                Freq=FreqTmp, Data=DataTmp, Site=SiteTmp,
                Comp=CompTmp, Info=InfoTmp, DTyp=DTypTmp,
                Scale=ScalTmp, allow_pickle=True,
            )
            if scs.issparse(JacTmp):
                scs.save_npz(Name + "_jac.npz", matrix=JacTmp)
            else:
                np.savez_compressed(Name + "_jac.npz", JacTmp)

        elapsed = time.perf_counter() - start
        print(" Used %7.4f s for splitting into sites " % elapsed)
