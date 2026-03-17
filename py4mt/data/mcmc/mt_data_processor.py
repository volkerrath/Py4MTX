#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-process MT station EDI files.

Reads EDI files, computes derived quantities, optionally interpolates /
sets errors / rotates, then exports in multiple formats (EDI, NPZ, HDF, MAT).
A collection NPZ file with all stations is saved at the end.

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

mypath = [PY4MTX_ROOT + "/py4mt/modules/",
          PY4MTX_ROOT + "/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)

import util as utl
from data_viz import add_phase, add_rho, add_tipper, add_pt
from data_proc import (
    get_edi_list,
    load_edi, save_edi, save_ncd, save_hdf, save_npz, save_mat,
    save_list_of_dicts_npz, dataframe_from_edi,
    interpolate_data, set_errors, estimate_errors, rotate_data,
    compute_pt, compute_zdet, compute_zssq, compute_rhophas,
)
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
WORK_DIR = PY4MTX_ROOT + "py4mt/data/edi/mcmc/"
if not os.path.isdir(WORK_DIR):
    print(" File: %s does not exist, but will be created" % WORK_DIR)
    os.mkdir(WORK_DIR)

DATA_DIR = WORK_DIR # + "/proc/"
if not os.path.isdir(DATA_DIR):
    print(" File: %s does not exist, but will be created" % DATA_DIR)
    os.mkdir(DATA_DIR)

EDI_DIR = WORK_DIR # + "/orig/"
edi_files = get_edi_list(EDI_DIR, fullpath=True)
ns = np.size(edi_files)

OUT_FILES = "edi, npz, hdf, mat"

STAT_FILE = True

PLOT = False
if PLOT:
    PLTARGS = {"show_errors": True}
    PLOT_FORMAT = [".png", ".pdf"]

NAME_STR = "_proc"
COLL_NAME = "Ann_MCMC"

SET_ERRORS = False
ERRORS = {
    "Zerr": [0.1, 0.1, 0.1, 0.1],
    "Terr": [0.03, 0.03, 0.03, 0.03],
    "PTerr": [0.1, 0.1, 0.1, 0.1],
}

PHAS_TENS = True
INVARS = True

INTERPOLATE = False
if INTERPOLATE:
    FREQ_PER_DEC = 6
    INT_METHOD = [None, FREQ_PER_DEC]

ESTIMATE_ERRORS = False
if ESTIMATE_ERRORS:
    sys.exit("Work in progress! Exit")
    SPREAD = 2.0  # *std-dev
    ERR_METHOD = ["gcvspline", SPREAD]

ROTATE = False
if ROTATE:
    ANGLE = 0.0
    DEC_DEG = True

# =============================================================================
#  Processing loop
# =============================================================================
all_data = []
for edi in edi_files:
    print("\n\nFound edi file: ", edi)

    data_dict = load_edi(edi, drop_invalid_periods=True)

    station = data_dict["station"]
    Z = data_dict["Z"]
    Zerr = data_dict["Z_err"]
    T = data_dict["T"]
    Terr = data_dict["T_err"]

    # --- Task block ---

    if PHAS_TENS:
        P, Perr = compute_pt(Z, Zerr)
        data_dict["P"] = P
        data_dict["P_err"] = Perr

    if INVARS:
        Zdet, Zdeterr = compute_zdet(Z, Zerr)
        data_dict["Zdet"] = Zdet
        data_dict["Zdet_err"] = Zdeterr
        Zssq, Zssqerr = compute_zssq(Z, Zerr)
        data_dict["Zssq"] = Zssq
        data_dict["Zssq_err"] = Zssqerr

    if ESTIMATE_ERRORS:
        data_dict = estimate_errors(data_dict=data_dict, method=ERR_METHOD)

    if SET_ERRORS:
        data_dict = set_errors(data_dict=data_dict, errors=ERRORS)

    if INTERPOLATE:
        data_dict = interpolate_data(data_dict=data_dict, method=INT_METHOD)

    if ROTATE:
        data_dict = rotate_data(data_dict=data_dict, angle=ANGLE)

    # --- Refresh apparent resistivity/phase after any Z modification ---
    if data_dict.get("freq") is not None and data_dict.get("Z") is not None:
        _ek = str(data_dict.get("err_kind", "var")).strip().lower()
        _ek = "std" if _ek.startswith("std") else "var"
        rho, phi, rho_err, phi_err = compute_rhophas(
            freq=np.asarray(data_dict["freq"]),
            Z=np.asarray(data_dict["Z"]),
            Z_err=(
                np.asarray(data_dict["Z_err"])
                if data_dict.get("Z_err") is not None
                else None
            ),
            err_kind=_ek,
            err_method="analytic",
        )
        data_dict["rho"] = rho
        data_dict["phi"] = phi
        if rho_err is not None:
            data_dict["rho_err"] = rho_err
        if phi_err is not None:
            data_dict["phi_err"] = phi_err

    all_data.append(data_dict)

    statname = station
    if STAT_FILE:
        nam, ext = os.path.splitext(os.path.basename(edi))
        statname = nam

    if "edi" in OUT_FILES.lower():
        save_edi(
            **data_dict,
            path=DATA_DIR + statname + NAME_STR + ".edi",
        )
        print("Wrote file: ", DATA_DIR + statname + NAME_STR + ".edi")

    if "ncd" in OUT_FILES.lower():
        save_ncd(
            **data_dict,
            path=DATA_DIR + statname + NAME_STR + ".ncd",
        )
        print("Wrote file: ", DATA_DIR + statname + NAME_STR + ".ncd")

    if "hdf" in OUT_FILES.lower():
        save_hdf(
            **data_dict,
            path=DATA_DIR + statname + NAME_STR + ".hdf",
        )
        print("Wrote file: ", DATA_DIR + statname + NAME_STR + ".hdf")

    if "mat" in OUT_FILES.lower():
        save_mat(
            **data_dict,
            path=DATA_DIR + statname + NAME_STR + ".mat",
        )
        print("Wrote file: ", DATA_DIR + statname + NAME_STR + ".mat")

    if "npz" in OUT_FILES.lower():
        save_npz(
            **data_dict,
            path=DATA_DIR + statname + NAME_STR + ".npz",
        )
        print("Wrote file: ", DATA_DIR + statname + NAME_STR + ".npz")

    if PLOT:
        fig, axs = plt.subplots(3, 2, figsize=(8, 14), sharex=True)
        df_rp = dataframe_from_edi(
            data_dict, include_tipper=False, include_pt=False
        )
        add_rho(df_rp, comps="xy,yx", ax=axs[0, 0], **PLTARGS)
        add_phase(df_rp, comps="xy,yx", ax=axs[0, 1], **PLTARGS)
        add_rho(df_rp, comps="xx,yy", ax=axs[1, 0], **PLTARGS)
        add_phase(df_rp, comps="xx,yy", ax=axs[1, 1], **PLTARGS)
        add_tipper(data_dict, ax=axs[2, 0], **PLTARGS)
        add_pt(data_dict, ax=axs[2, 1], **PLTARGS)
        fig.suptitle(statname + NAME_STR.replace("_", " | "))

        for ax in axs.flat:
            if not ax.lines and not ax.images and not ax.collections:
                fig.delaxes(ax)

        fig.tight_layout(rect=[0, 0, 1, 0.97])

        for f in PLOT_FORMAT:
            plt.savefig(WORK_DIR + statname + NAME_STR + f, dpi=600)

        plt.show()

save_list_of_dicts_npz(
    records=all_data,
    path=DATA_DIR + COLL_NAME + NAME_STR + "_collection.npz",
)
