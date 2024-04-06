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
from mtpy.core.z import Z, Tipper

import matplotlib as mpl
import matplotlib.pyplot as plt

PY4MT_ROOT = os.environ["PY4MT_ROOT"]
mypath = [PY4MT_ROOT+"/py4mt/modules/", PY4MT_ROOT+"/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)

import util
from version import versionstrg

PY4MT_DATA = os.environ["PY4MT_DATA"]

version, _ = versionstrg()
titstrng = util.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")

# Graphical paramter. Determine the plot formats produced,
# and the required resolution:

PerLimits = (0.001, 10.)  # AMT

# Define the path to your EDI-files:
    

    
EdiDir_in = r"/home/vrath/rjmcmc_mt/work/edi/"
print(" Edifiles read from: %s" % EdiDir_in)
EdiDir_out =  r"/home/vrath/rjmcmc_mt/work/edi/masked/"
print(" Plots written to: %s" % EdiDir_out)
if not os.path.isdir(EdiDir_out):
    print(" File: %s does not exist, but will be created" % EdiDir_out)
    os.mkdir(EdiDir_out)


# Probably No changes required after this line!

# Construct list of EDI-files:


edi_files = []
files = os.listdir(EdiDir_in)
for entry in files:
    # print(entry)
    if entry.endswith(".edi") and not entry.startswith("."):
        edi_files.append(entry)
edi_files = sorted(edi_files)


for filename in edi_files:
    name, ext = os.path.splitext(filename)
    file_i = EdiDir_in + filename
    mt = MT()
    mt.read(file_i)
    
    lat = mt.station_metadata.location.latitude
    lon = mt.station_metadata.location.longitude
    elev = mt.station_metadata.location.elevation
    
    print(" site %s at :  % 10.6f % 10.6f" % (name, lat, lon))


    f = mt.Z.freq
    fs = np.shape(f)

    z = mt.Z.z
    t = mt.Tipper.tipper


    fmin = 1./PerLimits[1]
    fmax = 1./PerLimits[0]



    mask = (f>=fmin) & (f<=fmax)


    new_Z_obj = Z(z_array=mt_obj.Z.z[mask],
                  z_err_array=mt_obj.Z.z_err[mask],
                  freq=mt_obj.Z.freq[mask])
    new_Tipper_obj = Tipper(tipper_array=mt_obj.Tipper.tipper[mask],
                            tipper_err_array=mt_obj.Tipper.tipper_err[mask],
                            freq = mt_obj.Tipper.freq[mask])

    # Write a new edi file (as before)
    file_out = name +"_masked"+".edi"
    print("Writing data to " + EdiDir_out + file_out)
    mt.write_mt_file(save_dir=EdiDir_out,
    fn_basename= file_out,
    file_type="edi",
    new_Z_obj=new_Z_obj,
    new_Tipper_obj=new_Tipper_obj,
    longitude_format='LONG',
    latlon_format='dd')
