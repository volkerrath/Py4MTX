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

'''
generate pseudo dat for forward modelling studies

@author: sb & vr July 2020

'''

# Import required modules
import os
import sys

import time
from datetime import datetime
import warnings
import csv
import inspect


import numpy as np

mypath = ['/home/vrath/Py4MT/py4mt/modules/',
          '/home/vrath/Py4MT/py4mt/scripts/']
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)
# Import required modules

import util as utl
import modem as mod
from mtpy.core.mt import MT

from version import versionstrg



rng = np.random.default_rng()
nan = np.nan  # float('NaN')
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())

titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng+'\n\n')

# Define the path to your EDI-template:

edi_template = r'/home/vrath/work/MT_Data/Krafla/Templates/template8.edi'
print(' Edifile template read from: %s' % edi_template)


# Define the path and appended string for saved EDI-files:

edi_out_dir = r'/home/vrath/work/MT_Data/Krafla/EDI/'
print(' Edifiles written to: %s' % edi_out_dir)
if not os.path.isdir(edi_out_dir):
    print(' File: %s does not exist, but will be created' % edi_out_dir)
    os.mkdir(edi_out_dir)

OutName = 'Krafla_'

edi_gen = 'rect regular' # rect
# edi_gen = 'rect random' # rect

if 'rect' in edi_gen.lower():

    # Krafla  65.711°, -16.778°
    LatLimits = ( 65.67,  65.75000)
    LonLimits = (-16.90000, -16.483333)
    # CenterLatLon = [65.771, -16.778]
    CenterLatLon =[np.mean(LatLimits),np.mean(LonLimits)]
    
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




# No changes required after this line!


# Construct list of EDI-files:

if 'rect' in edi_gen.lower():
    # generate site list

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

        file_out = OutName + nnstr + '.edi'

        print('\n Generating ' + edi_out_dir + file_out)
        print(
            ' site %s at :  % 10.6f % 10.6f' %
            (mt_tmp.station, mt_tmp.lat, mt_tmp.lon))

#  Write a new edi file:

        print('Writing data to ' + edi_out_dir + file_out)
        mt_tmp.write_mt_file(
            save_dir=edi_out_dir,
            fn_basename=file_out,
            file_type='edi',
            longitude_format='LONG',
            latlon_format='dd'
        )

