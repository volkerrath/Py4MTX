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
from mtpy.core.mt import MT
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
EDI_OUT_DIR = r"/home/vrath/work/MT_Data/Krafla/EDI/"
print(" Edifiles written to: %s" % EDI_OUT_DIR)
if not os.path.isdir(EDI_OUT_DIR):
    print(" File: %s does not exist, but will be created" % EDI_OUT_DIR)
    os.mkdir(EDI_OUT_DIR)

OUT_NAME = "Krafla_"

EDI_GEN = "rect regular"
# EDI_GEN = "rect random"

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
        mt_tmp = MT(file_in)

        mt_tmp.lat = Lat[nn]
        mt_tmp.lon = Lon[nn]
        mt_tmp.station = OUT_NAME + nnstr

        file_out = OUT_NAME + nnstr + ".edi"

        print("\n Generating " + EDI_OUT_DIR + file_out)
        print(
            " site %s at :  % 10.6f % 10.6f"
            % (mt_tmp.station, mt_tmp.lat, mt_tmp.lon)
        )

        # Write a new edi file:
        print("Writing data to " + EDI_OUT_DIR + file_out)
        mt_tmp.write_mt_file(
            save_dir=EDI_OUT_DIR,
            fn_basename=file_out,
            file_type="edi",
            longitude_format="LONG",
            latlon_format="dd",
        )
