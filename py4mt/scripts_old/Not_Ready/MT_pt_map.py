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
#       jupytext_version: 1.5.2
# ---

"""
Created on Fri Nov 22 07:29:58 2013

@author: Alison Kirkby

plots phase tensor ellipses as a map for a given frequency
"""

import os
import os.path as op

os.chdir(r'/home/geothest/mtpy4') # change to your path to mtpy installation to ensure correct version is used
import mtpy.imaging.phase_tensor_maps as pptmaps

# directory containing edis
edipath = r"/home/geothest/Desktop/WEST_TIMOR/WT8_2/interp_dist_ss/"

# whether or not to save the figure to file
save = True

# full path to file to save to
savepath = r"/home/geothest/Desktop/WEST_TIMOR/WT8_2/Plots/"


# frequency to plot
plot_freq = 100

# value to color ellipses by, options are phimin,phimax,skew
colorby='skew'
colorby='phimax'
ellipse_range = [-4,4]
ellipse_range = [10,70]

image_fn = 'phase_tensor_map%1is_'%(float(1./plot_freq))+colorby+'.png'

# gets edi file names as a list
elst = [op.join(edipath,f) for f in os.listdir(edipath) if f.endswith('.edi')]




m = pptmaps.PlotPhaseTensorMaps(fn_list = elst,
                                plot_freq = plot_freq ,
                                fig_size=(6,4),
                                ftol = .1,
                                xpad = 0.02,
                                plot_tipper = 'n',
                                edgecolor='k',
                                lw=0.01,
                                minorticks_on=False,
                                font_size = 2,
                                ellipse_colorby=colorby,
                                ellipse_range = ellipse_range,
                                ellipse_size=0.02,
                                arrow_lw=0.0001,
                                arrow_head_width=0.00001,
                                arrow_head_length=0.00001,
                                arrow_size=0.03,
                                arrow_threshold = 1.5,
#                                ellipse_cmap='mt_seg_bl2wh2rd'
                                station_dict={'id':(3,9)}
                                )


if save:
    m.save_figure(op.join(savepath,image_fn), fig_dpi=500) # change to your preferred file resolution
