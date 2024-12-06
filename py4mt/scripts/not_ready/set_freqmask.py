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
This script constructs a list of edifiles in a given directory, and produces
plots for all of them.

@author: sb & vr oct 2019

"""

# Import required modules

import os
import sys
import numpy as np

from mtpy.core.mt import MT
import mtpy.core.mt as mt

import matplotlib as mpl
import matplotlib.pyplot as plt


PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [PY4MTX_ROOT+"/py4mt/modules/", PY4MTX_ROOT+"/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)

import util
from version import versionstrg

version, _ = versionstrg()
titstrng = util.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")


DecDeg = True
String_out = ""

# Define the path to your EDI-files:
   
EdiDir_in = r"/home/vrath/rjmcmc_mt/work/edi/"
print(" Edifiles read from: %s" % EdiDir_in)
EdiDir_out =  r"/home/vrath/rjmcmc_mt/work/edi/masked/"
print(" Plots written to: %s" % EdiDir_out)
if not os.path.isdir(EdiDir_out):
    print(" File: %s does not exist, but will be created" % EdiDir_out)
    os.mkdir(EdiDir_out)
SearchStrng = ".edi"

PerLimits = (0.01, 1.)  # AMT
DecDeg = True
String_out = ""

# Probably No changes required after this line!

# Construct list of EDI-files:


edi_files = []
files = os.listdir(EdiDir_in)
for entry in files:
    # print(entry)
    if entry.endswith(".edi") and not entry.startswith("."):
        edi_files.append(entry)
edi_files = sorted(edi_files)

pmin, pmax = PerLimits

for filename in edi_files:
    name, ext = os.path.splitext(filename)
    file_i = EdiDir_in + filename
    
    
    mt_obj = MT()
    mt_obj.read(file_i)
    
    lat = mt_obj.station_metadata.location.latitude
    lon = mt_obj.station_metadata.location.longitude
    elev = mt_obj.station_metadata.location.elevation
    print(" site %s at :  % 10.6f % 10.6f % 8.1f" % (name, lat, lon, elev ))

    # pz = mt_obj.Z.period
   
    z = mt_obj.Z
    t = mt_obj.Tipper 
    per = z.period
    
    mask= np.where((per>=pmin) & (per<=pmax))
    new_periods = per[mask]
    if np.shape(new_periods)[0]==0:
        print("No data in period range!")
        continue
    
    new_mt_obj = mt_obj.interpolate(new_periods, inplace=False)
   
 
    # Write a new edi file (as before)
    file_out =EdiDir_out+name+String_out+ext
    print(" Writing data to "+file_out)
    if DecDeg:
        mt_obj.write(file_out, latlon_format='dd')
    else:
        mt_obj.write(file_out)

    
 
