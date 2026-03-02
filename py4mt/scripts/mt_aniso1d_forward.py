#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anisotropic 1-D MT forward modelling and plotting.

Computes impedance, apparent resistivity/phase, and phase tensor
for a layered anisotropic conductivity model, writes results to
text files, and generates diagnostic plots.

Author: Volker Rath (DIAS)
Created with the help of ChatGPT (GPT-5 Thinking) on 2026-02-13 (UTC)
"""

import os
import sys
import inspect

import numpy as np
import matplotlib.pyplot as plt

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [PY4MTX_ROOT + "/py4mt/modules/", PY4MTX_ROOT + "/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)

from aniso import aniso1d_impedance_sens
from data_proc import calc_rhoa_phas, compute_pt
import viz
import util as utl
from version import versionstrg

rng = np.random.default_rng()
nan = np.nan
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())

titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng + "\n\n")

# =============================================================================
#  Configuration
# =============================================================================
WorkDir = "/home/vrath/Py4MTX/work/aniso/"
if not os.path.isdir(WorkDir):
    print(" File: %s does not exist, but will be created" % WorkDir)
    os.mkdir(WorkDir)

SumOut = True
ResFile = WorkDir + "summary.dat"

ImpOut = True
ImpFile = WorkDir + "impedance.dat"
ImpPlt = True

RhoOut = True
RhoFile = WorkDir + "rhophas.dat"
RhoPlt = True

PhTOut = True
PhTFile = WorkDir + "phstens.dat"
PhTPlt = True

if RhoPlt or ImpPlt or PhTPlt:
    PlotFormat = [".png"]
    PlotFile = WorkDir + "AnisoTest"
    pltargs = {
        "pltsize": [16., 16.],
        "fontsizes": [18, 20, 24, 18],
        "m_size": 10,
        "c_obs": ["b", "r"],
        "m_obs": ["s", "o"],
        "l_obs": ["-", 2],
        "c_cal": ["b", "r"],
        "m_cal": [".", "."],
        "l_cal": ["-", 2],
        "nrms": [],
        "xlimits": [],
        "ylimits": [],
        "suptitle": "Anisotropic model Rong et al. (2022)",
    }

# =============================================================================
#  Model definition
#  Each row: [thickness_m, rho_x, rho_y, rho_z, strike, dip, slant, is_iso]
#  Last layer thickness = 0 (half-space).
# =============================================================================

# --- Model from Rong et al. (2022), Fig 1 ---
Periods = np.logspace(-2, 4., 41)
Freqs = 1. / Periods
NLayer = 4
Model = [
    [10000.,   1000.,   1000.,   1000.,    0.,  0.,  0., 0],
    [18000.,    200.,   2000.,    200.,   15.,  0.,  0., 0],
    [100000.,  1000.,  10000.,   1000.,  -75.,  0.,  0., 0],
    [0.,        100.,    100.,    100.,    0.,  0.,  0., 0],
]

# --- Alternative: Model A from Pek & Santos (2002) ---
# Periods = np.logspace(-3, 5., 41)
# Freqs = 1./Periods
# NLayer = 4
# Model = [
#     [10000.,  10000.,  10000.,  10000.,   0.,  0.,  0., 0],
#     [18000.,    200.,  20000.,    200.,  15.,  0.,  0., 0],
#     [100000., 1000.,   2000.,   1000., -75.,  0.,  0., 0],
#     [0.,       100.,    100.,    100.,   0.,  0.,  0., 0],
# ]

model = np.array(Model)
periods = Periods.copy()

h = model[:, 0]
rop = model[:, 1:4]
ustr = model[:, 4]
udip = model[:, 5]
usla = model[:, 6]
is_iso = model[:, 7] == 1

# =============================================================================
#  Write model summary
# =============================================================================
if SumOut:
    with open(ResFile, "w") as f:
        f.write("# Model parameters\n")
        f.write("# layer, thick (m)  res_x, res_y, res_z, strike, dip, slant\n")
        for layer in range(NLayer):
            pars = f"{layer:5d} {h[layer]:12.4f}"
            rops = f"   {rop[layer, 0]:12.4f} {rop[layer, 1]:12.4f} {rop[layer, 2]:12.4f} "
            angs = f"   {ustr[layer]:12.4f} {udip[layer]:12.4f} {usla[layer]:12.0f} "
            f.write(pars + rops + angs + "\n")

# =============================================================================
#  Compute impedances
# =============================================================================
res = aniso1d_impedance_sens(
    periods_s=Periods,
    h_m=h,
    rop=rop,
    ustr_deg=ustr,
    udip_deg=udip,
    usla_deg=usla,
    compute_sens=False,
)
Z = res["Z"]                                  # (nper, 2, 2)
P, _ = compute_pt(Z)                          # (nper, 2, 2)

nper = Z.shape[0]
Z = Z.reshape((nper, 4))
P = P.reshape((nper, 4))

# Interlace Re/Im columns: [Re_Zxx, Im_Zxx, Re_Zxy, Im_Zxy, ...]
shp = Z.shape
interlaced = np.empty((shp[0], 2 * shp[1]))
interlaced[:, 0::2] = Z.real
interlaced[:, 1::2] = Z.imag

# =============================================================================
#  Impedance output
# =============================================================================
if ImpOut:
    Imp = interlaced
    with open(ImpFile, "w") as f:
        f.write("#   PERIOD,  Re Zxx,  Im Zxx,  Re Zxy,  Im Zxy,"
                "  Re Zyx,  Im Zyx,  Re Zyy,  Im Zyy\n")
        for iper in range(len(Periods)):
            vals = "  ".join([f"{Imp[iper, k]:14.5e}" for k in range(Imp.shape[1])])
            f.write(f"{periods[iper]:14.5f}  {vals}\n")

if ImpPlt:
    Imp = interlaced
    pltargs["pltsize"] = [16., 16.]
    fig, ax = plt.subplots(2, 2, figsize=pltargs["pltsize"])
    fig.suptitle(pltargs["suptitle"], fontsize=pltargs["fontsizes"][2])

    pltargs["yscale"] = "linear"
    pltargs["ylabel"] = r"impedance [$\Omega$]"
    pltargs["legend"] = ["real", "imag"]

    data = np.zeros((len(Periods), 3))
    data[:, 0] = Periods[:]

    for idx, (col_re, col_im, title) in enumerate([
        (0, 1, "Zxx"), (2, 3, "Zxy"), (4, 5, "Zyx"), (6, 7, "Zyy"),
    ]):
        data[:, 1] = Imp[:, col_re]
        data[:, 2] = Imp[:, col_im]
        pltargs["title"] = title
        viz.plot_impedance(thisaxis=ax[idx // 2, idx % 2], data=data, **pltargs)

    for fmt in PlotFormat:
        plt.savefig(PlotFile + "_imped" + fmt)

# =============================================================================
#  Apparent resistivity / phase output
# =============================================================================
if RhoOut or RhoPlt:
    freqs = (1. / Periods).reshape(-1, 1)
    rhoa, phas = calc_rhoa_phas(freq=freqs, Z=Z)
    rhoa = np.asarray(rhoa).reshape((nper, -1))
    phas = np.asarray(phas).reshape((nper, -1))

if RhoOut:
    with open(RhoFile, "w") as f:
        f.write("#   PERIOD,  Rhoa xx,  Phs xx,  Rhoa xy,  Phs xy,"
                "  Rhoa yx,  Phs yx,  Rhoa yy,  Phs yy\n")
        for iper in range(len(Periods)):
            rhoa_phas = "".join(
                [f"{rhoa[iper, ii]:14.5e} {phas[iper, ii]:12.2f}" for ii in range(4)]
            )
            f.write(f"{periods[iper]:14.5f} {rhoa_phas}\n")

if RhoPlt:
    pltargs["pltsize"] = [16., 16.]
    fig, ax = plt.subplots(2, 2, figsize=pltargs["pltsize"])
    fig.suptitle(pltargs["suptitle"], fontsize=pltargs["fontsizes"][2])

    data = np.zeros((len(Periods), 3))
    data[:, 0] = Periods[:]

    data[:, 1] = rhoa[:, 1]
    data[:, 2] = rhoa[:, 2]
    pltargs["title"] = "Rho xy/yx"
    pltargs["legend"] = [r"$\rho_{a, xy}$", r"$\rho_{a, yx}$"]
    pltargs["yscale"] = "log"
    pltargs["ylimits"] = []
    pltargs["ylabel"] = r"$\rho_a$  [$\Omega$ m]"
    viz.plot_rhophas(thisaxis=ax[0, 0], data=data, **pltargs)

    data[:, 1] = phas[:, 1] + 90.
    data[:, 2] = phas[:, 2] - 90.
    pltargs["title"] = "Phas xy/yx"
    pltargs["legend"] = [r"$\phi_{xy}$", r"$\phi_{yx}$"]
    pltargs["yscale"] = "linear"
    pltargs["ylimits"] = [-180., 180.]
    pltargs["ylabel"] = r"$\phi$ [$^\circ$]"
    viz.plot_rhophas(thisaxis=ax[1, 0], data=data, **pltargs)

    data[:, 1] = rhoa[:, 0]
    data[:, 2] = rhoa[:, 3]
    pltargs["title"] = "Rho xx/yy"
    pltargs["legend"] = [r"$\rho_{a, xx}$", r"$\rho_{a, yy}$"]
    pltargs["yscale"] = "log"
    pltargs["ylimits"] = []
    pltargs["ylabel"] = r"$\rho_a$  [$\Omega$ m]"
    viz.plot_rhophas(thisaxis=ax[0, 1], data=data, **pltargs)

    data[:, 1] = phas[:, 0] - 90.
    data[:, 2] = phas[:, 3] + 90.
    pltargs["title"] = "Phas xx/yy"
    pltargs["legend"] = [r"$\phi_{xx}$", r"$\phi_{yy}$"]
    pltargs["yscale"] = "linear"
    pltargs["ylimits"] = [-180., 180.]
    pltargs["ylabel"] = r"$\phi$ [$^\circ$]"
    viz.plot_rhophas(thisaxis=ax[1, 1], data=data, **pltargs)

    for fmt in PlotFormat:
        plt.savefig(PlotFile + "_rhophas" + fmt)

# =============================================================================
#  Phase tensor output
# =============================================================================
if PhTOut:
    with open(PhTFile, "w") as f:
        f.write("#   PERIOD,  PT_xx,  PT_xy,  PT_yx,  PT_yy\n")
        for iper in range(len(Periods)):
            phstens = "".join(
                [f"{float(P[iper, ii]):14.5e} " for ii in range(4)]
            )
            f.write(f"{periods[iper]:14.5f} {phstens}\n")

if PhTPlt:
    pltargs["pltsize"] = [16., 8.]
    fig, ax = plt.subplots(1, 2, figsize=pltargs["pltsize"])
    fig.suptitle(pltargs["suptitle"], fontsize=pltargs["fontsizes"][2])

    data = np.zeros((len(Periods), 3))
    data[:, 0] = Periods[:]

    data[:, 1] = P[:, 1]
    data[:, 2] = P[:, 2]
    pltargs["title"] = r"Phase Tensor xy/yx"
    pltargs["legend"] = [r"$\Phi_{xy}$", r"$\Phi_{yx}$"]
    pltargs["yscale"] = "linear"
    pltargs["ylimits"] = []
    pltargs["ylabel"] = r"$\Phi$  [-]"
    viz.plot_phastens(thisaxis=ax[0], data=data, **pltargs)

    data[:, 1] = P[:, 0]
    data[:, 2] = P[:, 3]
    pltargs["title"] = r"Phase Tensor xx/yy"
    pltargs["legend"] = [r"$\Phi_{xx}$", r"$\Phi_{yy}$"]
    pltargs["yscale"] = "linear"
    pltargs["ylimits"] = []
    pltargs["ylabel"] = r"$\Phi$ [-]"
    viz.plot_phastens(thisaxis=ax[1], data=data, **pltargs)

    for fmt in PlotFormat:
        plt.savefig(PlotFile + "_phstens" + fmt)
