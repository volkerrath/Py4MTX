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
# ---

"""

This script produces a site list containing site names,
coordinates and elevations, e. g., for WALDIM analysis.

@author: sb & vr dec 2019
"""

# Import required modules

import os
from sys import exit as error
from mtpy.core.mt import MT
from mtpy.analysis.geometry import dimensionality
from mtpy.imaging.phase_tensor_maps import PlotPhaseTensorMaps

import numpy as np


# Define the path to your EDI-files and for the list produced
edi_dir = r"/home/vrath/Limerick2022/3D/edi_all/"
print(" Edifiles read from: %s" % edi_dir)

OutFmt = ".png"
OutDPI = 600
plot_freq = [1000., 100., 10., 1., 0.1, 0.01, 0.001]
Survey = "Limerick2022"
savepath = r"/home/vrath/Limerick2022/reports/Plots/"
"""
         ("Diverging", [
            "PiYG", "PRGn", "BrBG", "PuOr", "RdGy", "RdBu",
            "RdYlBu", "RdYlGn", "Spectral", "coolwarm", "bwr", "seismic"]),
"""
# gets edi file names as a list
edi_files = [os.path.join(edi_dir,ff) for ff in os.listdir(edi_dir) \
if ff.endswith(".edi")]
ns = np.size(edi_files)
if ns ==0:
    error("No edi files found in "+edi_dir+"! Exit.")
else:
    print(str(ns)+" edi files found")

for pf in plot_freq:

    ptmap = PlotPhaseTensorMaps(fn_list = edi_files,
            plot_freq = pf , # frequency to plot
            fig_size = (4,3), # x, y dimensions of figure
            xpad = 0.02, ypad = 0.02, # pad around stations
            plot_tipper = "yr", # "y" + "r" and/or "i" to plot real and/or imaginary
            edgecolor="k", # a matplotlib colour or None for no borders
            lw=0.5, # linewidth for the ellipses
            minorticks_on=True, # whether or not to turn on minor ticks
            ellipse_colorby="skew", # "phimin", "phimax", or "skew"
            ellipse_range = [-12,12,2], # [min,max,step]
            ellipse_size=0.005, # scaling factor for the ellipses
            arrow_size=0.025,
            arrow_head_width=0.001, # scaling for arrows (head width)
            arrow_head_length=0.001, # scaling for arrows (head length)
            ellipse_cmap="seismic", # matplotlib colormap
            # station_dict={"id":(5,7)} ,
            cb_dict = {"position":
            [1.05,0.2,0.02,0.3]}, # colorbar position [x,y,dx,dy]
            font_size=8    , background_image="/home/vrath/GSM.tif"

    )
    # save the plot
    if pf >= 1:
        ff = str(int(pf))+"Hz"
    else:
        ff = str(int(1./pf))+"s"

    image_fn = Survey+"_PhaseTensorMap_"+ff+OutFmt
    print(image_fn)
    ptmap.save_figure(os.path.join(savepath,image_fn), fig_dpi=OutDPI)
