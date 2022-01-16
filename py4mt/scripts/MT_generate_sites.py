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
#       format_version: "1.5"
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

"""
generate pseudo dat for forward modelling studies

@author: sb & vr July 2020

"""

# Import required modules
import os
import sys
from sys import exit as error
import time
from datetime import datetime
import warnings
import csv


import numpy as np

mypath = ["/home/vrath/Py4MT/py4mt/modules/",
          "/home/vrath/Py4MT/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)
# Import required modules

import util as utl
import modem as mod
from mtpy.core.mt import MT

from version import versionstrg

Strng, _ = versionstrg()
now = datetime.now()
print("\n\n"+Strng)
print("Generate sites on a mesh (various methods)"+"\n"+"".join("Date " + now.strftime("%m/%d/%Y, %H:%M:%S")))
print("\n\n")


edi_gen = "center" # rect

if "rect" in edi_gen.lower():

    # # generate site list
    # LonLimits = ( 6.275, 6.39)
    # nLon = 31
    # LatLimits = (45.37,45.46)
    # nLat = 31
    # LonLimits = (-25.5600, -25.2250)
    # nLon = 36
    # LatLimits = ( 37.6700,  37.8550)
    # nLat = 36
    LonLimits = (-16.90000, -16.483333)
    nLon = 11
    LatLimits = ( 65.66666666,  65.75000)
    nLat = 11

if "center" in edi_gen.lower():
    # Krafla  65.711°, -16.778°
    LatLimits = ( 65.67,  65.75000)
    LonLimits = (-16.90000, -16.483333)
    CenterLatLon = [65.771, -16.778]
    Dx = Dy = 1000

    epsg = utl.get_utm_zone(latitude=CenterLatLon[0], longitude=CenterLatLon[1])
    UTMxLimits, UTMyLimits= utl.project_latlon_to_utm(latitude=LatLimits,
                                       longitude=LonLimits,
                                       utm_zone=epsg[0])
    UTMDistx =np.abs(UTMxLimits[1]-UTMxLimits[0])
    UTMDisty =np.abs(UTMyLimits[1]-UTMyLimits[0])
    nX= np.ceil(UTMDistx/Dx)+1
    if nX % 2 == 1:
        nX=nX+1
    nY= np.ceil(UTMDisty/Dy)+1
    if nY % 2 == 1:
        nY=nY+1


if "readcsv" in edi_gen.lower():
    # edi_gen = "readcsv"
    # # read site list
    # edi_file = r"/home/vrath/AEM_Limerick/Limerick_pilot.csv"
    edi_file = ""

if "readmod" in edi_gen.lower():
    error("Option "+edi_gen.lower()+"not yet implemeted! Exit.")
    # # read site list
    # mod_file = r"/home/vrath/AEM_Limerick/Limerick_pilot_etopo.rho"
    # nx_bnd = 14
    # ny_bnd = 14
    # centerlatlon = ()
    # centermod = ()
    mod_file = ""
    nx_bnd = 0
    ny_bnd = 0


# Define the path to your EDI-template:

edi_template = r"/home/vrath/work/MT_Data/Krafla/Templates/template8.edi"
print(" Edifile template read from: %s" % edi_template)


# Define the path and appended string for saved EDI-files:

edi_out_dir = r"/home/vrath/work/MT_Data/Krafla/EDI/"
print(" Edifiles written to: %s" % edi_out_dir)
if not os.path.isdir(edi_out_dir):
    print(" File: %s does not exist, but will be created" % edi_out_dir)
    os.mkdir(edi_out_dir)

OutName = "Krafla_"


# No changes required after this line!


# Construct list of EDI-files:

if "rect" in edi_gen.lower():
    # generate site list
    Lat, Lon = utl.gen_grid_latlon(LatLimits, nLat, LonLimits, nLon)
    nn = -1
    for latval in Lat:
        nn = nn + 1
        nnstr = str(nn)
        mm = -1
        print(nnstr)
        for lonval in Lon:
            mm = mm + 1
            mmstr = str(mm)
            print(mmstr)

    # # Create an MT object

            file_in = edi_template
            mt_tmp = MT(file_in)

            mt_tmp.lat = Lat[nn]
            mt_tmp.lon = Lon[mm]
            mt_tmp.station = OutName + nnstr + "_" + mmstr

            file_out = OutName + nnstr + "_" + mmstr + ".edi"

            print("\n Generating " + edi_out_dir + file_out)
            print(
                " site %s at :  % 10.6f % 10.6f" %
                (mt_tmp.station, mt_tmp.lat, mt_tmp.lon))

    #  Write a new edi file:

            print("Writing data to " + edi_out_dir + file_out)
            mt_tmp.write_mt_file(
                save_dir=edi_out_dir,
                fn_basename=file_out,
                file_type="edi",
                longitude_format="LONG",
                latlon_format="dd"
            )

if "center" in edi_gen.lower():

    epsg = utl.get_utm_zone(latitude=CenterLatLon[0], longitude=CenterLatLon[1])
    UTMCenter = utl.project_latlon_to_utm(latitude=CenterLatLon[0],
                                       longitude=CenterLatLon[1],
                                       utm_zone=epsg[0])

    X = Dx*np.arange(nX)
    XCenter= 0.5*np.abs((X[0]-X[-1]))
    X=X+UTMCenter[0]-XCenter
    # print(X)

    Y = Dy*np.arange(nY)
    YCenter = 0.5*np.abs((Y[0]-Y[-1]))
    Y=Y+UTMCenter[1]-YCenter
    # print(Y)

    GridX, GridY = np.meshgrid(X, Y,indexing='xy')
    Lat, Lon = utl.project_utm_to_latlon(utm_x=GridX, utm_y=GridY, utm_zone=epsg[0])
    Lat = Lat.flat
    Lon = Lon.flat

    for nn in range(np.size(Lat)):
        nnstr = str(nn)
        print(nnstr)

    # # Create an MT object

        file_in = edi_template
        mt_tmp = MT(file_in)

        mt_tmp.lat = Lat[nn]
        mt_tmp.lon = Lon[nn]
        mt_tmp.station = OutName + nnstr

        file_out = OutName + nnstr + ".edi"

        print("\n Generating " + edi_out_dir + file_out)
        print(
            " site %s at :  % 10.6f % 10.6f" %
            (mt_tmp.station, mt_tmp.lat, mt_tmp.lon))

#  Write a new edi file:

        print("Writing data to " + edi_out_dir + file_out)
        mt_tmp.write_mt_file(
            save_dir=edi_out_dir,
            fn_basename=file_out,
            file_type="edi",
            longitude_format="LONG",
            latlon_format="dd"
        )


if "readcsv" in edi_gen.lower():
    # read site list
    Site = []
    Data = []
    with open(edi_file) as ef:
        for line in ef:
            print(line)
            d = line.split(",")
            Site.append([d[0]])
            Data.append([float(d[1]), float(d[2]), float(d[3])])

    Site = [item for sublist in Site for item in sublist]
    Site = np.asarray(Site, dtype=object)
    Data = np.asarray(Data)

    Lon = Data[:, 0]
    Lat = Data[:, 1]
    Elev = Data[:, 2]

    # Enter loop:
    nn = -1
    for place in Site:
        # Create an MT object
        nn = nn + 1
        file_in = edi_template
        mt_tmp = MT(file_in)

        mt_tmp.lat = Lat[nn]
        mt_tmp.lon = Lon[nn]
        mt_tmp.station = place

        file_out = OutName + "_" + place + ".edi"

        print("\n Generating " + edi_out_dir + file_out)
        print(" site %s at :  % 10.6f % 10.6f" %
              (mt_tmp.station, mt_tmp.lat, mt_tmp.lon))

    # # Write a new edi file:

        print("Writing data to " + edi_out_dir + file_out)
        mt_tmp.write_mt_file(
            save_dir=edi_out_dir,
            fn_basename=file_out,
            file_type="edi",
            longitude_format="LONG",
            latlon_format="dd"
        )


if "mod" in edi_gen.lower():

    dx, dy, dz, rho, reference, trans = mod.read_model(mod_file)

    nx = np.shape(dx)[0]
    x = np.hstack((0, np.cumsum(dx)))
    xreference = 0.5 * (x[0] + x[nx])
    x = x - xreference
    xc = 0.5 * (x[1:nx + 1] + x[0:nx])

    ny = np.shape(dy)[0]
    y = np.hstack((0, np.cumsum(dy)))
    yreference = 0.5 * (y[0] + y[ny])
    y = y - yreference
    yc = 0.5 * (y[1:ny + 1] + y[0:ny])

    for ii in np.arange(nx_bnd + 1, nx - nx_bnd + 1):

        for jj in np.arange(ny_bnd + 1, ny - ny_bnd + 1):

            Site = "_" + str(ii) + "_" + str(jj)

        for place in Site:
            # Create an MT object
            nn = nn + 1
            file_in = edi_template
            mt_tmp = MT(file_in)

            mt_tmp.lat = Lat[nn]
            mt_tmp.lon = Lon[nn]
            mt_tmp.station = place

            file_out = OutName + "_" + place + ".edi"

            print("\n Generating " + edi_out_dir + file_out)
            print(" site %s at :  % 10.6f % 10.6f" %
                  (mt_tmp.station, mt_tmp.lat, mt_tmp.lon))

        # # Write a new edi file:

            print("Writing data to " + edi_out_dir + file_out)
            mt_tmp.write_mt_file(
                save_dir=edi_out_dir,
                fn_basename=file_out,
                file_type="edi",
                longitude_format="LONG",
                latlon_format="dd"
            )
