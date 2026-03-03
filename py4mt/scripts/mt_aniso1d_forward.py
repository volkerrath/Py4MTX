#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anisotropic 1-D MT forward modelling and plotting.

Computes impedance, apparent resistivity/phase, and phase tensor
for a layered anisotropic conductivity model, writes results to
text files, and generates diagnostic plots.

@author:    Volker Rath (DIAS)
@project:   py4mt — Python for Magnetotellurics
@created:   2026-02-13 with the help of ChatGPT (GPT-5 Thinking)
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
WORK_DIR = "/home/vrath/Py4MTX/work/aniso/"
if not os.path.isdir(WORK_DIR):
    print(" File: %s does not exist, but will be created" % WORK_DIR)
    os.mkdir(WORK_DIR)

SUM_OUT = True
RES_FILE = WORK_DIR + "summary.dat"

IMP_OUT = True
IMP_FILE = WORK_DIR + "impedance.dat"
IMP_PLT = True

RHO_OUT = True
RHO_FILE = WORK_DIR + "rhophas.dat"
RHO_PLT = True

PHT_OUT = True
PHT_FILE = WORK_DIR + "phstens.dat"
PHT_PLT = True

if RHO_PLT or IMP_PLT or PHT_PLT:
    PLOT_FORMAT = [".png"]
    PLOT_FILE = WORK_DIR + "AnisoTest"
    PLTARGS = {
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
PERIODS = np.logspace(-2, 4., 41)
FREQS = 1. / PERIODS
N_LAYER = 4
MODEL = [
    [10000.,   1000.,   1000.,   1000.,    0.,  0.,  0., 0],
    [18000.,    200.,   2000.,    200.,   15.,  0.,  0., 0],
    [100000.,  1000.,  10000.,   1000.,  -75.,  0.,  0., 0],
    [0.,        100.,    100.,    100.,    0.,  0.,  0., 0],
]

# --- Alternative: Model A from Pek & Santos (2002) ---
# PERIODS = np.logspace(-3, 5., 41)
# FREQS = 1. / PERIODS
# N_LAYER = 4
# MODEL = [
#     [10000.,  10000.,  10000.,  10000.,   0.,  0.,  0., 0],
#     [18000.,    200.,  20000.,    200.,  15.,  0.,  0., 0],
#     [100000., 1000.,   2000.,   1000., -75.,  0.,  0., 0],
#     [0.,       100.,    100.,    100.,   0.,  0.,  0., 0],
# ]

model = np.array(MODEL)
periods = PERIODS.copy()

h = model[:, 0]
rop = model[:, 1:4]
ustr = model[:, 4]
udip = model[:, 5]
usla = model[:, 6]
is_iso = model[:, 7] == 1

# =============================================================================
#  Write model summary
# =============================================================================
if SUM_OUT:
    with open(RES_FILE, "w") as f:
        f.write("# Model parameters\n")
        f.write("# layer, thick (m)  res_x, res_y, res_z, strike, dip, slant\n")
        for layer in range(N_LAYER):
            pars = f"{layer:5d} {h[layer]:12.4f}"
            rops = f"   {rop[layer, 0]:12.4f} {rop[layer, 1]:12.4f} {rop[layer, 2]:12.4f} "
            angs = f"   {ustr[layer]:12.4f} {udip[layer]:12.4f} {usla[layer]:12.0f} "
            f.write(pars + rops + angs + "\n")

# =============================================================================
#  Compute impedances
# =============================================================================
res = aniso1d_impedance_sens(
    periods_s=PERIODS,
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
if IMP_OUT:
    Imp = interlaced
    with open(IMP_FILE, "w") as f:
        f.write("#   PERIOD,  Re Zxx,  Im Zxx,  Re Zxy,  Im Zxy,"
                "  Re Zyx,  Im Zyx,  Re Zyy,  Im Zyy\n")
        for iper in range(len(PERIODS)):
            vals = "  ".join([f"{Imp[iper, k]:14.5e}" for k in range(Imp.shape[1])])
            f.write(f"{periods[iper]:14.5f}  {vals}\n")

if IMP_PLT:
    Imp = interlaced
    PLTARGS["pltsize"] = [16., 16.]
    fig, ax = plt.subplots(2, 2, figsize=PLTARGS["pltsize"])
    fig.suptitle(PLTARGS["suptitle"], fontsize=PLTARGS["fontsizes"][2])

    PLTARGS["yscale"] = "linear"
    PLTARGS["ylabel"] = r"impedance [$\Omega$]"
    PLTARGS["legend"] = ["real", "imag"]

    data = np.zeros((len(PERIODS), 3))
    data[:, 0] = PERIODS[:]

    for idx, (col_re, col_im, title) in enumerate([
        (0, 1, "Zxx"), (2, 3, "Zxy"), (4, 5, "Zyx"), (6, 7, "Zyy"),
    ]):
        data[:, 1] = Imp[:, col_re]
        data[:, 2] = Imp[:, col_im]
        PLTARGS["title"] = title
        viz.plot_impedance(thisaxis=ax[idx // 2, idx % 2], data=data, **PLTARGS)

    for fmt in PLOT_FORMAT:
        plt.savefig(PLOT_FILE + "_imped" + fmt)

# =============================================================================
#  Apparent resistivity / phase output
# =============================================================================
if RHO_OUT or RHO_PLT:
    freqs = (1. / PERIODS).reshape(-1, 1)
    rhoa, phas = calc_rhoa_phas(freq=freqs, Z=Z)
    rhoa = np.asarray(rhoa).reshape((nper, -1))
    phas = np.asarray(phas).reshape((nper, -1))

if RHO_OUT:
    with open(RHO_FILE, "w") as f:
        f.write("#   PERIOD,  Rhoa xx,  Phs xx,  Rhoa xy,  Phs xy,"
                "  Rhoa yx,  Phs yx,  Rhoa yy,  Phs yy\n")
        for iper in range(len(PERIODS)):
            rhoa_phas = "".join(
                [f"{rhoa[iper, ii]:14.5e} {phas[iper, ii]:12.2f}" for ii in range(4)]
            )
            f.write(f"{periods[iper]:14.5f} {rhoa_phas}\n")

if RHO_PLT:
    PLTARGS["pltsize"] = [16., 16.]
    fig, ax = plt.subplots(2, 2, figsize=PLTARGS["pltsize"])
    fig.suptitle(PLTARGS["suptitle"], fontsize=PLTARGS["fontsizes"][2])

    data = np.zeros((len(PERIODS), 3))
    data[:, 0] = PERIODS[:]

    data[:, 1] = rhoa[:, 1]
    data[:, 2] = rhoa[:, 2]
    PLTARGS["title"] = "Rho xy/yx"
    PLTARGS["legend"] = [r"$\rho_{a, xy}$", r"$\rho_{a, yx}$"]
    PLTARGS["yscale"] = "log"
    PLTARGS["ylimits"] = []
    PLTARGS["ylabel"] = r"$\rho_a$  [$\Omega$ m]"
    viz.plot_rhophas(thisaxis=ax[0, 0], data=data, **PLTARGS)

    data[:, 1] = phas[:, 1] + 90.
    data[:, 2] = phas[:, 2] - 90.
    PLTARGS["title"] = "Phas xy/yx"
    PLTARGS["legend"] = [r"$\phi_{xy}$", r"$\phi_{yx}$"]
    PLTARGS["yscale"] = "linear"
    PLTARGS["ylimits"] = [-180., 180.]
    PLTARGS["ylabel"] = r"$\phi$ [$^\circ$]"
    viz.plot_rhophas(thisaxis=ax[1, 0], data=data, **PLTARGS)

    data[:, 1] = rhoa[:, 0]
    data[:, 2] = rhoa[:, 3]
    PLTARGS["title"] = "Rho xx/yy"
    PLTARGS["legend"] = [r"$\rho_{a, xx}$", r"$\rho_{a, yy}$"]
    PLTARGS["yscale"] = "log"
    PLTARGS["ylimits"] = []
    PLTARGS["ylabel"] = r"$\rho_a$  [$\Omega$ m]"
    viz.plot_rhophas(thisaxis=ax[0, 1], data=data, **PLTARGS)

    data[:, 1] = phas[:, 0] - 90.
    data[:, 2] = phas[:, 3] + 90.
    PLTARGS["title"] = "Phas xx/yy"
    PLTARGS["legend"] = [r"$\phi_{xx}$", r"$\phi_{yy}$"]
    PLTARGS["yscale"] = "linear"
    PLTARGS["ylimits"] = [-180., 180.]
    PLTARGS["ylabel"] = r"$\phi$ [$^\circ$]"
    viz.plot_rhophas(thisaxis=ax[1, 1], data=data, **PLTARGS)

    for fmt in PLOT_FORMAT:
        plt.savefig(PLOT_FILE + "_rhophas" + fmt)

# =============================================================================
#  Phase tensor output
# =============================================================================
if PHT_OUT:
    with open(PHT_FILE, "w") as f:
        f.write("#   PERIOD,  PT_xx,  PT_xy,  PT_yx,  PT_yy\n")
        for iper in range(len(PERIODS)):
            phstens = "".join(
                [f"{float(P[iper, ii]):14.5e} " for ii in range(4)]
            )
            f.write(f"{periods[iper]:14.5f} {phstens}\n")

if PHT_PLT:
    PLTARGS["pltsize"] = [16., 8.]
    fig, ax = plt.subplots(1, 2, figsize=PLTARGS["pltsize"])
    fig.suptitle(PLTARGS["suptitle"], fontsize=PLTARGS["fontsizes"][2])

    data = np.zeros((len(PERIODS), 3))
    data[:, 0] = PERIODS[:]

    data[:, 1] = P[:, 1]
    data[:, 2] = P[:, 2]
    PLTARGS["title"] = r"Phase Tensor xy/yx"
    PLTARGS["legend"] = [r"$\Phi_{xy}$", r"$\Phi_{yx}$"]
    PLTARGS["yscale"] = "linear"
    PLTARGS["ylimits"] = []
    PLTARGS["ylabel"] = r"$\Phi$  [-]"
    viz.plot_phastens(thisaxis=ax[0], data=data, **PLTARGS)

    data[:, 1] = P[:, 0]
    data[:, 2] = P[:, 3]
    PLTARGS["title"] = r"Phase Tensor xx/yy"
    PLTARGS["legend"] = [r"$\Phi_{xx}$", r"$\Phi_{yy}$"]
    PLTARGS["yscale"] = "linear"
    PLTARGS["ylimits"] = []
    PLTARGS["ylabel"] = r"$\Phi$ [-]"
    viz.plot_phastens(thisaxis=ax[1], data=data, **PLTARGS)

    for fmt in PLOT_FORMAT:
        plt.savefig(PLOT_FILE + "_phstens" + fmt)
