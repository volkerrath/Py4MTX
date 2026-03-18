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
@modified:  2026-03-16 — freq_order, D+/rho+ test (DPLUS), add_rhoplus plot; Claude Sonnet 4.6 (Anthropic)
@modified:  2026-03-18 — add_noise option in SET_ERRORS (mode='fix' only); Claude Sonnet 4.6 (Anthropic)
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
from data_viz import add_phase, add_rho, add_rhoplus, add_tipper, add_pt
from data_proc import (
    get_edi_list,
    load_edi, save_edi, save_ncd, save_hdf, save_npz, save_mat,
    save_list_of_dicts_npz, dataframe_from_edi,
    interpolate_data, set_errors, estimate_errors, rotate_data,
    compute_pt, compute_zdet, compute_zssq, compute_rhophas, compute_rhoplus,
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
WORK_DIR = PY4MTX_ROOT + "py4mt/data/edi/"
# WORK_DIR = "/home/vrath/MT_Data/waldim/edi_synth_iso/"
if not os.path.isdir(WORK_DIR):
   sys.exit(" File: %s does not exist, but will be created" % WORK_DIR)


DATA_DIR_IN = WORK_DIR + "/orig/"
DATA_DIR_OUT = WORK_DIR + "/proc/"

NAME_STR = "_test_proc"
COLL_NAME = "TEST"

OUT_FILES = "edi, npz"
STAT_FILE = True
STAT_UPPER = True


DROP_INVALID = True
FREQ_ORDER = "dec"   # "inc", "dec", or "keep"


PHAS_TENS = True
RHOPHAS = True
INVARS = True
DPLUS = False   # D+/rho+ test (Parker 1980; also known as dplus in Cordell mtcode)

ROTATE = False
if ROTATE:
    ANGLE = 0.0

DEC_DEG = True

SET_ERRORS = True
if SET_ERRORS:
    # Passed as **err_pars to set_errors(data_dict, mode=..., **err_pars)
    # Z_rel    : [xx, xy, yx, yy] relative levels
    # Z_rel_mode: "ij"  → σ_ij = Z_rel_ij * |Z_ij(ω)|
    #             "ij*ii" → σ_ij = Z_rel_ij * sqrt(|Z_ii|*|Z_ij|) off-diag
    # T_abs    : [Tx, Ty] absolute (constant) tipper error
    # PT_abs   : [xx, xy, yx, yy] absolute (constant) phase-tensor error
    # add_noise: if True, perturb data by N(0, σ) — only valid with mode='fix'
    ADD_NOISE = False
    err_pars = {
        "Z_rel":      [0.1, 0.1, 0.1, 0.1],
        "Z_rel_mode": "ij",          # "ij" or "ij*ii"
        "T_abs":      [0.03, 0.03],
        "PT_abs":     [0.1, 0.1, 0.1, 0.1],
        "mode":       "fix",         # "floor"
        "add_noise":  ADD_NOISE,
        "random_state": rng,         # module-level Generator; set seed for reproducibility
    }


INTERPOLATE = False
if INTERPOLATE:
    # Passed as **interp_pars to interpolate_data(data_dict, **interp_pars)
    interp_pars = {
        "freq_per_dec":   6,
        "interp_method":  "gcvspline",
    }

ESTIMATE_ERRORS = False
if ESTIMATE_ERRORS:
    sys.exit("Work in progress! Exit")
    SPREAD = 2.0  # *std-dev
    ERR_METHOD = ["gcvspline", SPREAD]

PLOT = True
if PLOT:
    PLOT_DIR = WORK_DIR +"/plots/"
    PLTARGS = {
        "show_errors": True,
        "xlim": None,       # period limits (s), e.g. (1e-3, 1e3); None = auto
        "ylim": None,       # y-axis limits; None = auto
    }
    PLOT_FORMAT = [".pdf"]

if not os.path.isdir(PLOT_DIR):
    print(" File: %s does not exist, but will be created" % PLOT_DIR)
    os.mkdir(PLOT_DIR)
if not os.path.isdir(DATA_DIR_OUT):
    print(" File: %s does not exist, but will be created" % DATA_DIR_OUT)
    os.mkdir(DATA_DIR_OUT)


edi_files = get_edi_list(DATA_DIR_IN, fullpath=True)
ns = np.size(edi_files)
# =============================================================================
#  Processing loop
# =============================================================================
all_data = []
for edi in edi_files:
    print("\n\nFound edi file: ", edi)

    data_dict = load_edi(edi, drop_invalid_periods=DROP_INVALID, freq_order=FREQ_ORDER)

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
        data_dict = set_errors(data_dict, **err_pars)


    if INTERPOLATE:
        data_dict = interpolate_data(data_dict, **interp_pars)

    if ROTATE:
        data_dict = rotate_data(data_dict, angle=ANGLE)

    # --- Refresh apparent resistivity/phase after any Z modification ---
    if RHOPHAS:
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

    if DPLUS:
        # D+/rho+ test on Zxy, Zyx, and Zdet (if available)
        _freq = np.asarray(data_dict["freq"])
        _dplus_results = {}
        for _comp, _zi, _zj in [("xy", 0, 1), ("yx", 1, 0)]:
            _ze = Zerr[:, _zi, _zj] if Zerr is not None else None
            _rho_plus, _rho_a, _ok = compute_rhoplus(_freq, Z[:, _zi, _zj], _ze)
            _dplus_results[_comp] = {
                "rho_plus": _rho_plus, "rho_a": _rho_a, "pass": _ok
            }
            _nviol = int((~_ok).sum())
            print(f"  D+ {_comp.upper()}: {_nviol}/{len(_ok)} violations")
        if data_dict.get("Zdet") is not None:
            _ze = data_dict.get("Zdet_err")
            _rho_plus, _rho_a, _ok = compute_rhoplus(_freq, data_dict["Zdet"], _ze)
            _dplus_results["det"] = {
                "rho_plus": _rho_plus, "rho_a": _rho_a, "pass": _ok
            }
            print(f"  D+ DET: {int((~_ok).sum())}/{len(_ok)} violations")
        data_dict["dplus"] = _dplus_results


    all_data.append(data_dict)

    statname = station
    if STAT_FILE:
        nam, ext = os.path.splitext(os.path.basename(edi))
        statname = nam

    if STAT_UPPER:
        statname =  statname.upper()
        data_dict["station"] = statname

    if "edi" in OUT_FILES.lower():
        save_edi(
            **data_dict,
            path=DATA_DIR_OUT + statname + NAME_STR + ".edi",
        )
        print("Wrote file: ", DATA_DIR_OUT + statname + NAME_STR + ".edi")

    if "ncd" in OUT_FILES.lower():
        save_ncd(
            **data_dict,
            path=DATA_DIR_OUT + statname + NAME_STR + ".ncd",
        )
        print("Wrote file: ", DATA_DIR_OUT + statname + NAME_STR + ".ncd")

    if "hdf" in OUT_FILES.lower():
        save_hdf(
            **data_dict,
            path=DATA_DIR_OUT + statname + NAME_STR + ".hdf",
        )
        print("Wrote file: ", DATA_DIR_OUT + statname + NAME_STR + ".hdf")

    if "mat" in OUT_FILES.lower():
        save_mat(
            **data_dict,
            path=DATA_DIR_OUT + statname + NAME_STR + ".mat",
        )
        print("Wrote file: ", DATA_DIR_OUT + statname + NAME_STR + ".mat")

    if "npz" in OUT_FILES.lower():
        save_npz(
            **data_dict,
            path=DATA_DIR_OUT + statname + NAME_STR + ".npz",
        )
        print("Wrote file: ", DATA_DIR_OUT + statname + NAME_STR + ".npz")

    if PLOT:
        nrows =  3
        fig, axs = plt.subplots(nrows, 2, figsize=(
            8, 14 + 4 * (nrows - 3)), sharex=True)

        fig.suptitle(statname + NAME_STR.replace("_", " | "))

        df_rp = dataframe_from_edi(
            data_dict, include_tipper=False, include_pt=False
        )
        PLTARGS["xlim"] = (1.e-4, 1.e+3)
        PLTARGS["ylim"] = (1.e-4, 1.e+4)
        add_rho(df_rp, comps="xy,yx", ax=axs[0, 0], **PLTARGS)
        PLTARGS["ylim"] = (-180., +180.)
        add_phase(df_rp, comps="xy,yx", ax=axs[0, 1], **PLTARGS)
        PLTARGS["ylim"] = (1.e-4, 1.e+4)
        add_rho(df_rp, comps="xx,yy", ax=axs[1, 0], **PLTARGS)
        PLTARGS["ylim"] = (-180., +180.)
        add_phase(df_rp, comps="xx,yy", ax=axs[1, 1], **PLTARGS)
        PLTARGS["ylim"] = (1., +1.)
        add_tipper(data_dict, ax=axs[2, 0], **PLTARGS)
        PLTARGS["ylim"] = (1., +1.)
        add_pt(data_dict, ax=axs[2, 1], **PLTARGS)

        for ax in axs.flat:
            if not ax.lines and not ax.images and not ax.collections:
                fig.delaxes(ax)

        fig.tight_layout(rect=[0, 0, 1, 0.97])

        for f in PLOT_FORMAT:
            plt.savefig(PLOT_DIR + statname + NAME_STR + f, dpi=600)

        plt.show()

save_list_of_dicts_npz(
    records=all_data,
    path=DATA_DIR_OUT + COLL_NAME + NAME_STR + "_collection.npz",
)
