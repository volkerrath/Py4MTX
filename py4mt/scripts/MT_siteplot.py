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

PlotFmt = [".png"]
dpi = 400
PdfC = True
if not ".pdf" in PlotFmt:
    PdfC = False
    print("No PDF catalog because no pdf output!")
PdfCName  = "Opf2023_data.pdf"


# What should be plotted?
# 1 = yx and xy; 2 = all 4 components
# 3 = off diagonal + determinant

plot_z = 3
no_err = False
strng="_Z"+str(plot_z)
# Plot tipper?
# "y" or "n", followed by "r","i", or "ri", for real part, imaginary part, or both, respectively.
plot_t = "n" #yri" #""yri"
print (plot_t[0])
if plot_t[0]=="y":
    strng = strng+"T"+plot_t[1:]
# Plot phase tensor?
# "y" or "n"
plot_p = "y"
if plot_p=="y":
    strng = strng+"P"

strng="_data"


PerLimits = (0.0001, 1.)  # AMT
# PerLimits = (0.001,100000.) #BBMT
# PerLimits = (0.00003,10000.) #AMT+BBMT
RhoLimits = (10., 10000.)
PhiLimits = (-180., 180.)
Tiplimits = (-.5, 0.5)
# Define the path to your EDI-files:
# edi_in_dir = r"/home/vrath/RRV_work/edi_work/Edited/"
edi_in_dir = r"/home/vrath/work/MT_Data/Opf/2023/edi/"
# r"/home/vrath/MauTopo/MauTopo500_edi/"
# r"/home/vrath/RRV_work/edifiles_in/"
# edi_in_dir =  r"/home/vrath/RRV_work/edifiles_r1500m_bbmt/"
print(" Edifiles read from: %s" % edi_in_dir)

# Define the edi_in_dir path for saving  plots:


plots_dir =  r"/home/vrath/work/MT_Data/Opf/2023/plots/"
print(" Plots written to: %s" % plots_dir)
if not os.path.isdir(plots_dir):
    print(" File: %s does not exist, but will be created" % plots_dir)
    os.mkdir(plots_dir)

# No changes required after this line!

# Construct list of EDI-files:


edi_files = []
files = os.listdir(edi_in_dir)
for entry in files:
    # print(entry)
    if entry.endswith(".edi") and not entry.startswith("."):
        edi_files.append(entry)
edi_files = sorted(edi_files)

if PdfC:
    pdf_list= []
    for filename in edi_files:
        name, ext = os.path.splitext(filename)
        pdf_list.append(plots_dir+name+strng+".pdf")
# Create an MT object

for filename in edi_files:
    name, ext = os.path.splitext(filename)
    file_i = edi_in_dir + filename
    mt_obj   = MT(file_i)
    
    print(" site %s at :  % 10.6f % 10.6f" % (name, mt_obj.lat, mt_obj.lon))


    f = mt_obj.Z.freq
    fs = np.shape(f)

    z = mt_obj.Z.z
    t = mt_obj.Tipper.tipper
    print(np.shape(t))
    print()
    
    DatLimits = PerLimits
    fmin = 1./DatLimits[1]
    fmax = 1./DatLimits[0]

    for ii in np.arange(fs[0]):

        if (abs(z[ii,:,:]).any()>1.e30):    z[ii,:,:] = 1.e30
        if (abs(z[ii,:,:]).any()<1.e-30):   z[ii,:,:] = 1.e-30
        if (abs(t[ii,:]).any()>1.e30):      t[ii,:] = 1.e30
        if (abs(t[ii,:]).any()<1.e-30):     t[ii,:] = 1.e-30


    if no_err is True:
        # mt_obj.Z.z_err = 0.0001*np.ones_like( freq_list = mt_obj.Z.freqnp.real(mt_obj.Z.z))
        # mt_obj.Tipper.tipper_err = 0.0001*np.ones_like(np.real(mt_obj.Tipper.tipper))
        mt_obj.Z.z_err = 0.001 * np.real(mt_obj.Z.z)
        mt_obj.Tipper.tipper_err = 0.001 * np.real(mt_obj.Tipper.tipper)

    plot_obj = mt_obj.plot_mt_response(plot_num=plot_z,
                                       plot_tipper=plot_t,
                                       plot_pt=plot_p,
                                       x_limits=PerLimits,
                                       # res_limits=RhoLimits,
                                       # phase_limits=PhiLimits,
                                       # tipper_limits=Tiplimits,
                                       fig_dpi=400,
                                       xy_ls="",yx_ls="", det_ls="",
                                       # ellipse_colorby="skew",
                                       # ellipse_range = [-10.,10.,2.]
                                       )

# Finally save figure

    for F in PlotFmt:
          plot_obj.save_plot(plots_dir+name+strng+F)

# if PdfC:
#     util.make_pdf_catalog(plots_dir, PdfList=pdf_list, FileName=plots_dir+PdfCName)
