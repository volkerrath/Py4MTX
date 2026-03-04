#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

"""
Generate pseudo data for forward modelling studies.

Creates a rectangular grid of MT station locations and writes one EDI file
per site using an existing EDI template.

@author:    sb & vr  July 2020
@project:   py4mt — Python for Magnetotellurics
@inversion: ModEM
"""

import os
import sys
import inspect

import numpy as np

mypath = ["/home/vrath/Py4MT/py4mt/modules/",
          "/home/vrath/Py4MT/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)

import util as utl
import modem as mod

from data_proc import read_edi, save_edi, save_npz
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

# Define the path to your EDI-template:
EDI_TEMPLATE = r"/home/vrath/work/MT_Data/Krafla/Templates/template8.edi"
print(" Edifile template read from: %s" % EDI_TEMPLATE)

# Define the path and appended string for saved EDI-files:
OUT_DIR = r"/home/vrath/work/MT_Data/Krafla/EDI/"
print(" Edifiles written to: %s" % OUT_DIR)
if not os.path.isdir(OUT_DIR):
    print(" File: %s does not exist, but will be created" % OUT_DIR)
    os.mkdir(OUT_DIR)

OUT_NAME = "Krafla_"
OUT_FILES = "edi, npz"

EDI_GEN = "rect regular"
# EDI_GEN = "rect random"

NAME_STR = EDI_GEN.replace(" ", "_")

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

# =============================================================================
#  Site grid definition
# =============================================================================
if "rect" in EDI_GEN.lower():

    # Krafla  65.711deg, -16.778deg
    LAT_LIMITS = (65.67, 65.75000)
    LON_LIMITS = (-16.90000, -16.483333)
    CENTER_LATLON = [np.mean(LAT_LIMITS), np.mean(LON_LIMITS)]

    DX = DY = 1000

    epsg = utl.get_utm_zone(
        latitude=CENTER_LATLON[0], longitude=CENTER_LATLON[1]
    )
    UTM_X_LIMITS, UTM_Y_LIMITS = utl.project_latlon_to_utm(
        latitude=LAT_LIMITS, longitude=LON_LIMITS, utm_zone=epsg[0]
    )

    UTM_DIST_X = np.abs(UTM_X_LIMITS[1] - UTM_X_LIMITS[0])
    UTM_DIST_Y = np.abs(UTM_Y_LIMITS[1] - UTM_Y_LIMITS[0])

    N_X = np.ceil(UTM_DIST_X / DX) + 1
    if N_X % 2 == 1:
        N_X = N_X + 1
    N_Y = np.ceil(UTM_DIST_Y / DY) + 1
    if N_Y % 2 == 1:
        N_Y = N_Y + 1

# =============================================================================
#  No changes required after this line!
# =============================================================================

# Construct list of EDI-files:
if "rect" in EDI_GEN.lower():

    epsg = utl.get_utm_zone(
        latitude=CENTER_LATLON[0], longitude=CENTER_LATLON[1]
    )
    UTM_CENTER = utl.project_latlon_to_utm(
        latitude=CENTER_LATLON[0],
        longitude=CENTER_LATLON[1],
        utm_zone=epsg[0],
    )

    X = DX * np.arange(N_X)
    X_CENTER = 0.5 * np.abs(X[0] - X[-1])
    X = X + UTM_CENTER[0] - X_CENTER

    Y = DY * np.arange(N_Y)
    Y_CENTER = 0.5 * np.abs(Y[0] - Y[-1])
    Y = Y + UTM_CENTER[1] - Y_CENTER

    GRID_X, GRID_Y = np.meshgrid(X, Y, indexing="xy")
    Lat, Lon = utl.project_utm_to_latlon(
        utm_x=GRID_X, utm_y=GRID_Y, utm_zone=epsg[0]
    )
    Lat = Lat.flat
    Lon = Lon.flat

    for nn in range(np.size(Lat)):
        nnstr = str(nn)
        print(nnstr)

        # Create an MT object
        file_in = EDI_TEMPLATE
        edi_dict = edi_read  edi_dict = load_edi(edi, drop_invalid_periods=True)

        station = edi_dict["station"]

        Z = edi_dict["Z"]
        Zerr = edi_dict["Z_err"]
        T = edi_dict["T"]
        Terr = edi_dict["T_err"]

        # --- Task block ---

        if PHAS_TENS:
            P, Perr = compute_pt(Z, Zerr)
            edi_dict["P"] = P
            edi_dict["P_err"] = Perr

        if INVARS:
            Zdet, Zdeterr = compute_zdet(Z, Zerr)
            edi_dict["Zdet"] = Zdet
            edi_dict["Zdet_err"] = Zdeterr
            Zssq, Zssqerr = compute_zssq(Z, Zerr)
            edi_dict["Zssq"] = Zssq
            edi_dict["Zssq_err"] = Zssqerr

        if SET_ERRORS:
            edi_dict = set_errors(edi_dict=edi_dict, errors=ERRORS)

        if INTERPOLATE:
            edi_dict = interpolate_data(edi_dict=edi_dict, method=INT_METHOD)

        # --- Refresh apparent resistivity/phase after any Z modification ---
        if edi_dict.get("freq") is not None and edi_dict.get("Z") is not None:
            _ek = str(edi_dict.get("err_kind", "var")).strip().lower()
            _ek = "std" if _ek.startswith("std") else "var"
            rho, phi, rho_err, phi_err = compute_rhophas(
                freq=np.asarray(edi_dict["freq"]),
                Z=np.asarray(edi_dict["Z"]),
                Z_err=(
                    np.asarray(edi_dict["Z_err"])
                    if edi_dict.get("Z_err") is not None
                    else None
                ),
                err_kind=_ek,
                err_method="analytic",
            )
            edi_dict["rho"] = rho
            edi_dict["phi"] = phi
            if rho_err is not None:
                edi_dict["rho_err"] = rho_err
            if phi_err is not None:
                edi_dict["phi_err"] = phi_err

        statname = OUT_NAME+f"{nn:03d}"
        edi_dict["station"] = statname

        edi_dict["lat_deg"] = Lat[nn]
        edi_dict["lon_deg"] = Lon[nn]

        edi_dict["info"] =  EDI_GEN +"|"+str(LAT_LIMITS)+str(LON_LIMITS)+"|"+ str{DX}+", "+str(DY)


        if "edi" in OUT_FILES.lower():
            _ = save_edi(
                path=OUT_DIR + statname + NAME_STR + ".edi",
                edi=edi_dict,
            )
            print("Wrote file: ", OUT_DIR + statname + NAME_STR + ".edi")

        if "npz" in OUT_FILES.lower():
        _ = save_npz(
            path=OUT_DIR + statname + NAME_STR + ".npz",
            data_dict=edi_dict,
        )
        print("Wrote file: ", OUT_DIR + statname + NAME_STR + ".npz")


