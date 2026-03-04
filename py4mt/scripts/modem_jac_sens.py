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

import jac_proc as jac
import modem as mod
from version import versionstrg
import util as utl

version, _ = versionstrg()
titstrng = utl.print_title(
    version=version, fname=inspect.getfile(inspect.currentframe()), out=False
)
print(titstrng + "\n\n")

# =============================================================================
#  Configuration
# =============================================================================
BLANK = 1.0e-30
RHOAIR = 1.0e17

INP_FORMAT = "sparse"
OUT_FORMAT = "mod rlm"
MOD_EXT = "_sns.rho"

WORK_DIR = "./"
J_NAME = "Ub25_ZPT_nerr_sp-8"
J_FILE = WORK_DIR + J_NAME
M_FILE = WORK_DIR + "Ub_600ZT4_PT_NLCG_009"
M_ORIG = [0.0, 0.0]

SIZ_EXTRACT = True
TOPO_EXTRACT = True

SPLITS = "total dtyp site freq comp"
NO_REIM = True

NORM_LOCAL = True
if (not NORM_LOCAL) and ("tot" not in SPLITS.lower()):
    SPLITS = "total " + SPLITS

PER_INTERVALS = [
    [0.0001, 0.001], [0.001, 0.01], [0.01, 0.1], [0.1, 1.0],
    [1.0, 10.0], [10.0, 100.0], [100.0, 1000.0], [1000.0, 10000.0],
]

TYPE = "cov"
TRANSFORM = "siz vol max"

if TRANSFORM is None:
    snsstring = TYPE.lower()
else:
    snsstring = TYPE.lower() + "_" + TRANSFORM.replace(" ", "-").lower()

SENS_DIR = WORK_DIR + J_NAME + "_sens_" + snsstring + "/"
if not SENS_DIR.endswith("/"):
    SENS_DIR = SENS_DIR + "/"
if not os.path.isdir(SENS_DIR):
    print("Directory: %s does not exist, but will be created" % SENS_DIR)
    os.mkdir(SENS_DIR)


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
dx, dy, dz, rho, refmod, _ = mod.read_mod(M_FILE, trans="linear")
elapsed = time.perf_counter() - start
total += elapsed
print(" Used %7.4f s for reading model from %s " % (elapsed, M_FILE))

aircells = np.where(rho > RHOAIR / 10)

if TOPO_EXTRACT:
    TopoFile = M_FILE + ".top"
    xcnt, ycnt, topo = mod.get_topo(
        dx=dx, dy=dy, dz=dz, mval=rho, ref=refmod, mvalair=1.0e17, out=True,
    )
    if os.path.isfile(TopoFile):
        os.remove(TopoFile)
    with open(TopoFile, "a") as f:
        for ii in np.arange(len(dx)):
            for jj in np.arange(len(dy)):
                f.write(f"{xcnt[ii]}, {ycnt[jj]}, {topo[ii, jj]}\n")

if "siz" in TRANSFORM.lower():
    for key, val in [("vol", "vol"), ("area", "area"), ("hsiz", "hsiz"), ("vsiz", "vsiz")]:
        if key in TRANSFORM.lower():
            siztyp = val
            break
    else:
        siztyp = "vol"
    siz = mod.get_size(dx=dx, dy=dy, dz=dz, mval=rho, how=siztyp, out=True)
else:
    siztyp = "vol"
    siz = np.array([])

if SIZ_EXTRACT:
    SizFile = M_FILE + ".siz"
    siz = mod.get_size(dx=dx, dy=dy, dz=dz, mval=rho, how=siztyp, out=True)
    Header = "# " + M_FILE
    write_sensitivity(SizFile, siz, dx, dy, dz, refmod, aircells,
                      OUT_FORMAT, "_siz.rho", M_ORIG, BLANK, "Cell volumes")

mdims = np.shape(rho)
aircells = np.where(rho > RHOAIR / 10)
jacmask = jac.set_airmask(rho=rho, aircells=aircells, blank=BLANK, flat=False, out=True)
jacflat = jacmask.flatten(order="F")

# =============================================================================
#  Read Jacobian
# =============================================================================
start = time.perf_counter()
print("Reading Jacobian from " + J_FILE)

if "spa" in INP_FORMAT:
    Jac = scs.load_npz(J_FILE + "_jac.npz")
    tmp = np.load(J_FILE + "_info.npz", allow_pickle=True)
    Freqs = tmp["Freq"]
    Comps = tmp["Comp"]
    Sites = tmp["Site"]
    Dtype = tmp["DTyp"]
else:
    Jac, tmp = mod.read_jac(J_FILE + ".jac")
    Data, Sites, Freqs, Comps, Dtype, Head = mod.read_data_jac(J_FILE + "_jac.dat")
    dsh = np.shape(Data)
    err = np.reshape(Data[:, 5], (dsh[0], 1))
    Jac = jac.normalize_jac(Jac, err)

elapsed = time.perf_counter() - start
total += elapsed
print(" Used %7.4f s for reading Jacobian/data from %s" % (elapsed, J_FILE))

print("Full Jacobian")
jac.print_stats(jac=Jac, jacmask=jacflat)

MaxTotal = None

# ---- Total ----
if "tot" in SPLITS.lower():
    start = time.perf_counter()
    SensTmp = jac.calc_sensitivity(Jac, Type=TYPE, OutInfo=False)
    if TRANSFORM is None:
        SensTot = SensTmp
        MaxTotal = np.amax(SensTot)
    else:
        SensTot, MaxTotal = jac.transform_sensitivity(
            S=SensTmp, Siz=siz, Transform=TRANSFORM, OutInfo=False,
        )
    S = SensTot.reshape(mdims, order="F")
    SensFile = SENS_DIR + J_NAME + "_total_" + snsstring
    write_sensitivity(SensFile, S, dx, dy, dz, refmod, aircells,
                      OUT_FORMAT, MOD_EXT, M_ORIG, BLANK, "Total")
    elapsed = time.perf_counter() - start
    print(" Used %7.4f s for total sensitivities " % elapsed)

# ---- By data type ----
if "dtyp" in SPLITS.lower():
    start = time.perf_counter()
    typestr = ["zfull", "zoff", "tp", "mf", "rpoff", "pt"]
    for ityp in np.unique(Dtype):
        JacTmp = Jac[np.where(Dtype == ityp)]
        print("Data type:", ityp)
        jac.print_stats(jac=JacTmp, jacmask=jacflat)
        maxval = None if NORM_LOCAL else MaxTotal
        SensTmp = jac.calc_sensitivity(JacTmp, Type=TYPE, OutInfo=False)
        if TRANSFORM is not None:
            SensTmp, _ = jac.transform_sensitivity(
                S=SensTmp, Siz=siz, Transform=TRANSFORM, Maxval=maxval, OutInfo=False,
            )
        S = np.reshape(SensTmp, mdims, order="F")
        SensFile = SENS_DIR + J_NAME + "_Dtype_" + typestr[ityp - 1] + "_" + snsstring
        write_sensitivity(SensFile, S, dx, dy, dz, refmod, aircells,
                          OUT_FORMAT, MOD_EXT, M_ORIG, BLANK, "Data type")
    elapsed = time.perf_counter() - start
    print(" Used %7.4f s for data type sensitivities " % elapsed)

# ---- By component ----
if "comp" in SPLITS.lower():
    start = time.perf_counter()
    ExistComp = np.unique(Comps)
    if NO_REIM:
        ExistComp = np.unique([c.replace("R", "").replace("I", "") for c in ExistComp])
        Comps_nori = np.array([c.replace("R", "").replace("I", "") for c in Comps])
    else:
        Comps_nori = Comps

    for icmp in ExistComp:
        JacTmp = Jac[np.where(Comps_nori == icmp)]
        print("Component:", icmp)
        jac.print_stats(jac=JacTmp, jacmask=jacflat)
        maxval = None if NORM_LOCAL else MaxTotal
        SensTmp = jac.calc_sensitivity(JacTmp, Type=TYPE, OutInfo=False)
        if TRANSFORM is not None:
            SensTmp, _ = jac.transform_sensitivity(
                S=SensTmp, Siz=siz, Transform=TRANSFORM, Maxval=maxval, OutInfo=False,
            )
        S = np.reshape(SensTmp, mdims, order="F")
        SensFile = SENS_DIR + J_NAME + "_" + icmp + "_" + snsstring
        write_sensitivity(SensFile, S, dx, dy, dz, refmod, aircells,
                          OUT_FORMAT, MOD_EXT, M_ORIG, BLANK, "Component")
    elapsed = time.perf_counter() - start
    print(" Used %7.4f s for comp sensitivities " % elapsed)

# ---- By site ----
if "site" in SPLITS.lower():
    start = time.perf_counter()
    SiteNames = Sites[np.sort(np.unique(Sites, return_index=True)[1])]
    for sit in SiteNames:
        JacTmp = Jac[np.where(sit == Sites)]
        print("Site:", sit)
        jac.print_stats(jac=JacTmp, jacmask=jacflat)
        maxval = None if NORM_LOCAL else MaxTotal
        SensTmp = jac.calc_sensitivity(JacTmp, Type=TYPE, OutInfo=False)
        if TRANSFORM is not None:
            SensTmp, _ = jac.transform_sensitivity(
                S=SensTmp, Siz=siz, Transform=TRANSFORM, Maxval=maxval, OutInfo=False,
            )
        S = np.reshape(SensTmp, mdims, order="F")
        SensFile = SENS_DIR + J_NAME + "_" + sit.lower() + "_" + snsstring
        write_sensitivity(SensFile, S, dx, dy, dz, refmod, aircells,
                          OUT_FORMAT, MOD_EXT, M_ORIG, BLANK, "Site")
    elapsed = time.perf_counter() - start
    print(" Used %7.4f s for site sensitivities " % elapsed)

# ---- By frequency band ----
if "freq" in SPLITS.lower():
    start = time.perf_counter()
    for ibnd in np.arange(len(PER_INTERVALS)):
        lowstr = str(1.0 / PER_INTERVALS[ibnd][0]) + "Hz"
        uppstr = str(1.0 / PER_INTERVALS[ibnd][1]) + "Hz"
        indices = np.where(
            (Freqs >= PER_INTERVALS[ibnd][0]) & (Freqs < PER_INTERVALS[ibnd][1])
        )
        JacTmp = Jac[indices]
        if np.shape(JacTmp)[0] > 0:
            print("Freqband:", lowstr, "to", uppstr)
            jac.print_stats(jac=JacTmp, jacmask=jacflat)
            maxval = None if NORM_LOCAL else MaxTotal
            SensTmp = jac.calc_sensitivity(JacTmp, Type=TYPE, OutInfo=False)
            if TRANSFORM is not None:
                SensTmp, _ = jac.transform_sensitivity(
                    S=SensTmp, Siz=siz, Transform=TRANSFORM,
                    Maxval=maxval, OutInfo=False,
                )
            S = np.reshape(SensTmp, mdims, order="F")
            SensFile = (SENS_DIR + J_NAME + "_freqband" + lowstr
                        + "_to_" + uppstr + "_" + snsstring)
            write_sensitivity(SensFile, S, dx, dy, dz, refmod, aircells,
                              OUT_FORMAT, MOD_EXT, M_ORIG, BLANK, "Frequency band")
        else:
            print("Frequency band is empty! Continue.")
    elapsed = time.perf_counter() - start
    print(" Used %7.4f s for freq sensitivities " % elapsed)
