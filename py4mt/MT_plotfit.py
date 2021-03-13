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
#       jupytext_version: 1.10.3
# ---

"""
Created on Sun Jun  7 20:38:34 2020

@author: vrath
"""

import mtpy.modeling.modem as modem

# Graphical paramter. Determine the plot formats produced,
# and the required resolution:

plot_pdf = True
plot_png = True
plot_eps = False

dpi = 400

# PerLimits = (0.0001,10.) #AMT
# PerLimits = (0.001,100000.) #BBMT
PerLimits = (0.0003, 10000.)  # AMT+BBMT
RhoLimits = (10., 30000.)
PhiLimits = (-10., 100.)


dfn = r"/home/vrath/RRV_work/lofreq_sscor/data_file.dat"
rfn = r"/home/vrath/RRV_work/lofreq_sscor/NewRRV_Lfreq_sscor_NLCG_091.dat"

plot_obj = modem.PlotResponse(data_fn=dfn, resp_fn=rfn)

# plot only the TE and TM modes

plot_obj.plot_z = False
plot_obj.plot_component = 2
plot_obj.plot_style = 1
plot_obj.phase_limits = PhiLimits
plot_obj.res_limits = RhoLimits
plot_obj.period_limits = PerLimits
plot_obj.plot_tipper = False

plot_obj.redraw_plot()

if plot_png:
    plot_obj.save_plot(
        os.path.join(
            plots_dir,
            name + ".png"),
        file_format='png',
        fig_dpi=dpi)
if plot_pdf:
    plot_obj.save_plot(
        os.path.join(
            plots_dir,
            name + ".pdf"),
        file_format='pdf',
        fig_dpi=dpi)
if plot_eps:
    plot_obj.save_plot(
        os.path.join(
            plots_dir,
            name + ".eps"),
        file_format='eps',
        fig_dpi=dpi)
