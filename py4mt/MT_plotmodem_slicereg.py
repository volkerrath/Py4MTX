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
#       jupytext_version: 1.4.2
# ---

"""
Created on Wed Jan 29 15:45:32 2020

@author: vrath
"""

import os
from mtpy.modeling.modem import PlotSlices

# Graphical paramter. Determine the plot formats produced, 
# and the required resolution: 

plot_pdf=True
plot_png=True
plot_eps=False

dpi = 400

ModelPath = r'./Fogo/'
plots_dir = model_dir 


model_fn = os.path.join(ModelPath,'Modular_MPI_NLCG_004.rho')
data_fn = os.path.join(ModelPath,'ModEM_Data.dat')



plot_obj = PlotSlices(model_fn=model_fn, data_fn=data_fn,
                  climits = [0,2], # log10(colour limits)
                  cmap='bwr_r', # colour map
                  plot_stations=True, # True/False - show station locations or not
                  station_id=[5,8], # indices (start,finish) of station label to plot
                  ew_limits=[-220,220], # east-west limits, if not provided will auto calculate from data
                  ns_limits=[-170,170], # north-south limits, if not provided will auto calculate from data
                  font_size=6, # font size on plots
                  fig_size=(6,3), # figure size
                  plot_yn='n', # whether to load interactive plotting
                  fig_dpi = dpi # change to your preferred file resolution
)
plot_obj.PlotPath = plots_dir
plot_obj.export_slices(plane='N-E', # options are 'N-Z', 'E-Z', and 'N-E'
                   indexlist=[32], # slice indices to plot
                   station_buffer=20e3, # distance threshold for plotting stations on vertical slice
                   save=True,
)
# Finally save figure

# if plot_png:
#     plot_obj.save_plot(os.path.join(plots_dir,name+".png"),file_format='png',fig_dpi=dpi)
# if plot_pdf:
#     plot_obj.save_plot(os.path.join(plots_dir,name+".pdf"),file_format='pdf',fig_dpi=dpi)
# if plot_eps:
#     plot_obj.save_plot(os.path.join(plots_dir,name+".eps"),file_format='eps',fig_dpi=dpi)
