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
#       jupytext_version: 1.11.1
# ---


"""
This script constructs a list of edifiles in a given directory, and produces
plots for all of them.

@author: sb & vr oct 2019

"""

# Import required modules

import os
#from mtpy.core.mt import MT
from modules.phase_tensor_pseudosection import PlotPhaseTensorPseudoSection


# Graphical paramter. Determine the plot formats produced,
# and the required resolution:

plot_pdf = True
plot_png = True
plot_eps = False
dpi = 600
fsiz = 8
lwid = 0.1
stretch = (15000, 200)
prefix_remove = 'ARR'
plot_name = 'RRV_BBMT_PhaseTensorSection_Skew'

# colorby:          - colour by phimin, phimax, skew, skew_seg
# ellipse_range     - 3 numbers, the 3rd indicates interval, e.g. [-12,12,3]
# set color limits  - default 0,90 for phimin or max, [-12,12] for skew.
#
# If plotting skew_seg need to provide ellipse_dic, e.g:
#                   ellipse_dict={'ellipse_colorby':'skew_seg',
#                                 'ellipse_range':[-12, 12, 3]}


edict = {'ellipse_colorby': 'skew',
         'ellipse_range': [-12, 12, 3]}

# edict =  {'ellipse_colorby':'phimin',
#           'ellipse_range':[0.,90.]}


# Plot tipper?
# plot tipper       - 'n'/'y' + 'ri/r/i' means real+imag
plot_t = 'n'


# Define the path to your EDI-files:
edi_in_dir = r'/home/vrath/RRV_work/edi_work/'
print(' Edifiles read from: %s' % edi_in_dir)

# Define the path for saving  plots:
plots_dir = edi_in_dir
# plots_dir = r'/home/vrath/RRV_work/edifiles_in/'
print(' Plots written to: %s' % plots_dir)
if not os.path.isdir(plots_dir):
    print(' File: %s does not exist, but will be created' % plots_dir)
    os.mkdir(plots_dir)

# No changes required after this line!

# Construct list of EDI-files:


edi_files = []
files = os.listdir(edi_in_dir)
for entry in files:
    if entry.endswith('.edi') and not entry.startswith(prefix_remove):
        full = os.path.join(edi_in_dir, entry)
        edi_files.append(full)


#  print(edi_files)
#  create a plot object

plot_obj = PlotPhaseTensorPseudoSection(fn_list=edi_files,
                                        ellipse_dict=edict,
                                        plot_tipper=plot_t,
                                        )

# update parameters (tweak for your dataset)

plot_obj.fig_size = (15, 8)

plot_obj.linedir = 'ns'
plot_obj.xstretch = stretch[0]
plot_obj.ystretch = stretch[1]
plot_obj.ylimits = (.001, 10000.)

plot_obj.lw = lwid
plot_obj.font_size = fsiz + 2
plot_obj.station_id = (0, 34)

plot_obj.ellipse_size = 36

plot_obj.plot_title = 'Rainy River Transect - BBMT'
plot_obj.cb_orientation = 'vertical'
plot_obj.plot_reference = True

plot_obj.plot()


# Finally save figure

if plot_png:
    plot_obj.save_figure(
        os.path.join(
            plots_dir,
            plot_name + ".png"),
        file_format='png',
        fig_dpi=dpi)
if plot_pdf:
    plot_obj.save_figure(
        os.path.join(
            plots_dir,
            plot_name + ".pdf"),
        file_format='pdf',
        fig_dpi=dpi)
if plot_eps:
    plot_obj.save_figure(
        os.path.join(
            plots_dir,
            plot_name + ".eps"),
        file_format='eps',
        fig_dpi=dpi)
