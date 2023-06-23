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

# Graphical paramter. Determine the plot formats produced,
# and the required resolution:

PerLimits = (0.001, 10.)  # AMT

# Define the path to your EDI-files:
# edi_in_dir = r"/home/vrath/RRV_work/edi_work/Edited/"
edi_in_dir = r"/home/vrath/BHP/edi/"
# r"/home/vrath/MauTopo/MauTopo500_edi/"
# r"/home/vrath/RRV_work/edifiles_in/"
# edi_in_dir =  r"/home/vrath/RRV_work/edifiles_r1500m_bbmt/"
print(" Edifiles read from: %s" % edi_in_dir)

# Define theedi_in_dir path for saving  plots:


edi_out_dir =  r"/home/vrath/BHP/edi_masked/"
print(" Plots written to: %s" % edi_out_dir)
if not os.path.isdir(edi_out_dir):
    print(" File: %s does not exist, but will be created" % edi_out_dir)
    os.mkdir(edi_out_dir)

# No changes required after this line!

# Construct list of EDI-files:


edi_files = []
files = os.listdir(edi_in_dir)
for entry in files:
    # print(entry)
    if entry.endswith(".edi") and not entry.startswith("."):
        edi_files.append(entry)
edi_files = sorted(edi_files)


for filename in edi_files:
    name, ext = os.path.splitext(filename)
    file_i = edi_in_dir + filename
    mt_obj = MT(file_i)
    print(" site %s at :  % 10.6f % 10.6f" % (name, mt_obj.lat, mt_obj.lon))


    f = mt_obj.Z.freq
    fs = np.shape(f)

    z = mt_obj.Z.z
    t = mt_obj.Tipper.tipper


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
    print("Writing data to " + edi_out_dir + file_out)
    mt_obj.write_mt_file(save_dir=edi_out_dir,
    fn_basename= file_out,
    file_type="edi",
    new_Z_obj=new_Z_obj,
    new_Tipper_obj=new_Tipper_obj,
    longitude_format='LONG',
    latlon_format='dd')
