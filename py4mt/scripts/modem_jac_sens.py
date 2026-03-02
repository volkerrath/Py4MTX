#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute sensitivity maps from a ModEM Jacobian.

Reads a processed (error-normalised, sparsified) Jacobian and computes
sensitivity maps split by: total, data type, component, site, and/or
frequency band. Supports output in ModEM, UBC, and RLM/CGG formats.

Sensitivity types: 'raw', 'abs'/'cov' (coverage), 'euc' (Euclidean).
Transforms: 'vol'/'siz' (volume normalisation), 'max', 'sur', 'log'.

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

import jacproc as jac
import modem as mod
from version import versionstrg
import util as utl

version, _ = versionstrg()
titstrng = utl.print_title(
    version=version, fname=inspect.getfile(inspect.currentframe()), out=False
)
print(titstrng + "\n\n")

rng = np.random.default_rng()
Blank = 1.0e-30
Rhoair = 1.0e17

# =============================================================================
#  Configuration
# =============================================================================
InpFormat = "sparse"
OutFormat = "mod rlm"
ModExt = "_sns.rho"

WorkDir = "./"
JName = "Ub25_ZPT_nerr_sp-8"
JFile = WorkDir + JName
MFile = WorkDir + "Ub_600ZT4_PT_NLCG_009"
MOrig = [0.0, 0.0]

SizExtract = True
TopoExtract = True

Splits = "total dtyp site freq comp"
NoReIm = True

NormLocal = True
if (not NormLocal) and ("tot" not in Splits.lower()):
    Splits = "total " + Splits

PerIntervals = [
    [0.0001, 0.001], [0.001, 0.01], [0.01, 0.1], [0.1, 1.0],
    [1.0, 10.0], [10.0, 100.0], [100.0, 1000.0], [1000.0, 10000.0],
]

Type = "cov"
Transform = "siz vol max"

if Transform is None:
    snsstring = Type.lower()
else:
    snsstring = Type.lower() + "_" + Transform.replace(" ", "-").lower()

SensDir = WorkDir + JName + "_sens_" + snsstring + "/"
if not SensDir.endswith("/"):
    SensDir = SensDir + "/"
if not os.path.isdir(SensDir):
    print("Directory: %s does not exist, but will be created" % SensDir)
    os.mkdir(SensDir)


# =============================================================================
#  Helper: write sensitivity in all requested formats
# =============================================================================
def write_sensitivity(filename, S, dx, dy, dz, refmod, aircells,
                      out_format, mod_ext, morig, blank, label=""):
    """Write sensitivity array in requested output formats."""
    header = "# " + filename.replace("_", " | ")

    if "mod" in out_format.lower():
        mod.write_mod(
            filename, modext=mod_ext, dx=dx, dy=dy, dz=dz, mval=S,
            reference=refmod, mvalair=blank, aircells=aircells, header=header,
        )
        print(f" {label} sensitivities (ModEM format) written to {filename}")

    if "ubc" in out_format.lower():
        elev = -refmod[2]
        refubc = [morig[0], morig[1], elev]
        mod.write_ubc(
            filename, modext="_ubc.sns", mshext="_ubc.msh",
            dx=dx, dy=dy, dz=dz, mval=S,
            reference=refubc, mvalair=blank, aircells=aircells, header=header,
        )
        print(f" {label} sensitivities (UBC format) written to {filename}")

    if "rlm" in out_format.lower():
        mod.write_rlm(
            filename, modext="_sns.rlm", dx=dx, dy=dy, dz=dz, mval=S,
            reference=refmod, mvalair=blank, aircells=aircells, comment=header,
        )
        print(f" {label} sensitivities (RLM/CGG format) written to {filename}")


# =============================================================================
#  Read model
# =============================================================================
total = 0.0
start = time.perf_counter()
dx, dy, dz, rho, refmod, _ = mod.read_mod(MFile, trans="linear")
elapsed = time.perf_counter() - start
total += elapsed
print(" Used %7.4f s for reading model from %s " % (elapsed, MFile))

aircells = np.where(rho > Rhoair / 10)

if TopoExtract:
    TopoFile = MFile + ".top"
    xcnt, ycnt, topo = mod.get_topo(
        dx=dx, dy=dy, dz=dz, mval=rho, ref=refmod, mvalair=1.0e17, out=True,
    )
    if os.path.isfile(TopoFile):
        os.remove(TopoFile)
    with open(TopoFile, "a") as f:
        for ii in np.arange(len(dx)):
            for jj in np.arange(len(dy)):
                f.write(f"{xcnt[ii]}, {ycnt[jj]}, {topo[ii, jj]}\n")

if "siz" in Transform.lower():
    for key, val in [("vol", "vol"), ("area", "area"), ("hsiz", "hsiz"), ("vsiz", "vsiz")]:
        if key in Transform.lower():
            siztyp = val
            break
    else:
        siztyp = "vol"
    siz = mod.get_size(dx=dx, dy=dy, dz=dz, mval=rho, how=siztyp, out=True)
else:
    siztyp = "vol"
    siz = np.array([])

if SizExtract:
    SizFile = MFile + ".siz"
    siz = mod.get_size(dx=dx, dy=dy, dz=dz, mval=rho, how=siztyp, out=True)
    Header = "# " + MFile
    write_sensitivity(SizFile, siz, dx, dy, dz, refmod, aircells,
                      OutFormat, "_siz.rho", MOrig, Blank, "Cell volumes")

mdims = np.shape(rho)
aircells = np.where(rho > Rhoair / 10)
jacmask = jac.set_airmask(rho=rho, aircells=aircells, blank=Blank, flat=False, out=True)
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
    Data, Sites, Freqs, Comps, Dtype, Head = mod.read_data_jac(JFile + "_jac.dat")
    dsh = np.shape(Data)
    err = np.reshape(Data[:, 5], (dsh[0], 1))
    Jac = jac.normalize_jac(Jac, err)

elapsed = time.perf_counter() - start
total += elapsed
print(" Used %7.4f s for reading Jacobian/data from %s" % (elapsed, JFile))

print("Full Jacobian")
jac.print_stats(jac=Jac, jacmask=jacflat)

MaxTotal = None

# ---- Total ----
if "tot" in Splits.lower():
    start = time.perf_counter()
    SensTmp = jac.calc_sensitivity(Jac, Type=Type, OutInfo=False)
    if Transform is None:
        SensTot = SensTmp
        MaxTotal = np.amax(SensTot)
    else:
        SensTot, MaxTotal = jac.transform_sensitivity(
            S=SensTmp, Siz=siz, Transform=Transform, OutInfo=False,
        )
    S = SensTot.reshape(mdims, order="F")
    SensFile = SensDir + JName + "_total_" + snsstring
    write_sensitivity(SensFile, S, dx, dy, dz, refmod, aircells,
                      OutFormat, ModExt, MOrig, Blank, "Total")
    elapsed = time.perf_counter() - start
    print(" Used %7.4f s for total sensitivities " % elapsed)

# ---- By data type ----
if "dtyp" in Splits.lower():
    start = time.perf_counter()
    typestr = ["zfull", "zoff", "tp", "mf", "rpoff", "pt"]
    for ityp in np.unique(Dtype):
        JacTmp = Jac[np.where(Dtype == ityp)]
        print("Data type:", ityp)
        jac.print_stats(jac=JacTmp, jacmask=jacflat)
        maxval = None if NormLocal else MaxTotal
        SensTmp = jac.calc_sensitivity(JacTmp, Type=Type, OutInfo=False)
        if Transform is not None:
            SensTmp, _ = jac.transform_sensitivity(
                S=SensTmp, Siz=siz, Transform=Transform, Maxval=maxval, OutInfo=False,
            )
        S = np.reshape(SensTmp, mdims, order="F")
        SensFile = SensDir + JName + "_Dtype_" + typestr[ityp - 1] + "_" + snsstring
        write_sensitivity(SensFile, S, dx, dy, dz, refmod, aircells,
                          OutFormat, ModExt, MOrig, Blank, "Data type")
    elapsed = time.perf_counter() - start
    print(" Used %7.4f s for data type sensitivities " % elapsed)

# ---- By component ----
if "comp" in Splits.lower():
    start = time.perf_counter()
    ExistComp = np.unique(Comps)
    if NoReIm:
        ExistComp = np.unique([c.replace("R", "").replace("I", "") for c in ExistComp])
        Comps_nori = np.array([c.replace("R", "").replace("I", "") for c in Comps])
    else:
        Comps_nori = Comps

    for icmp in ExistComp:
        # BUG FIX: original used `Jac[icmp in Comps]` (boolean scalar), now proper indexing
        JacTmp = Jac[np.where(Comps_nori == icmp)]
        print("Component:", icmp)
        jac.print_stats(jac=JacTmp, jacmask=jacflat)
        maxval = None if NormLocal else MaxTotal
        SensTmp = jac.calc_sensitivity(JacTmp, Type=Type, OutInfo=False)
        if Transform is not None:
            SensTmp, _ = jac.transform_sensitivity(
                S=SensTmp, Siz=siz, Transform=Transform, Maxval=maxval, OutInfo=False,
            )
        S = np.reshape(SensTmp, mdims, order="F")
        SensFile = SensDir + JName + "_" + icmp + "_" + snsstring
        write_sensitivity(SensFile, S, dx, dy, dz, refmod, aircells,
                          OutFormat, ModExt, MOrig, Blank, "Component")
    elapsed = time.perf_counter() - start
    print(" Used %7.4f s for comp sensitivities " % elapsed)

# ---- By site ----
if "site" in Splits.lower():
    start = time.perf_counter()
    SiteNames = Sites[np.sort(np.unique(Sites, return_index=True)[1])]
    for sit in SiteNames:
        JacTmp = Jac[np.where(sit == Sites)]
        print("Site:", sit)
        jac.print_stats(jac=JacTmp, jacmask=jacflat)
        maxval = None if NormLocal else MaxTotal
        SensTmp = jac.calc_sensitivity(JacTmp, Type=Type, OutInfo=False)
        if Transform is not None:
            SensTmp, _ = jac.transform_sensitivity(
                S=SensTmp, Siz=siz, Transform=Transform, Maxval=maxval, OutInfo=False,
            )
        S = np.reshape(SensTmp, mdims, order="F")
        SensFile = SensDir + JName + "_" + sit.lower() + "_" + snsstring
        write_sensitivity(SensFile, S, dx, dy, dz, refmod, aircells,
                          OutFormat, ModExt, MOrig, Blank, "Site")
    elapsed = time.perf_counter() - start
    print(" Used %7.4f s for site sensitivities " % elapsed)

# ---- By frequency band ----
if "freq" in Splits.lower():
    start = time.perf_counter()
    for ibnd in np.arange(len(PerIntervals)):
        lowstr = str(1.0 / PerIntervals[ibnd][0]) + "Hz"
        uppstr = str(1.0 / PerIntervals[ibnd][1]) + "Hz"
        indices = np.where(
            (Freqs >= PerIntervals[ibnd][0]) & (Freqs < PerIntervals[ibnd][1])
        )
        JacTmp = Jac[indices]
        if np.shape(JacTmp)[0] > 0:
            print("Freqband:", lowstr, "to", uppstr)
            jac.print_stats(jac=JacTmp, jacmask=jacflat)
            maxval = None if NormLocal else MaxTotal
            SensTmp = jac.calc_sensitivity(JacTmp, Type=Type, OutInfo=False)
            if Transform is not None:
                SensTmp, _ = jac.transform_sensitivity(
                    S=SensTmp, Siz=siz, Transform=Transform,
                    Maxval=maxval, OutInfo=False,
                )
            S = np.reshape(SensTmp, mdims, order="F")
            SensFile = (SensDir + JName + "_freqband" + lowstr
                        + "_to_" + uppstr + "_" + snsstring)
            write_sensitivity(SensFile, S, dx, dy, dz, refmod, aircells,
                              OutFormat, ModExt, MOrig, Blank, "Frequency band")
        else:
            print("Frequency band is empty! Continue.")
    elapsed = time.perf_counter() - start
    print(" Used %7.4f s for freq sensitivities " % elapsed)
