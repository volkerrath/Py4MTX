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

"""
This script constructs a list of edifiles in a given directory, and produces
plots for all of them.

@author: sb & vr oct 2019

"""

# Import required modules

import os
import numpy as np
from mtpy.core.mt import MT


PY4MT_ROOT = os.environ["PY4MT_ROOT"]
mypath = [PY4MT_ROOT+"/py4mt/modules/", PY4MT_ROOT+"/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)


# Graphical paramter. Determine the plot formats produced,
# and the required resolution:

plot_pdf = True
plot_png = True
plot_eps = False

dpi = 400

# What should be plotted?
# 1 = yx and xy; 2 = all 4 components
# 3 = off diagonal + determinant

plot_z = 2
no_err = True
# Plot tipper?
# 'y' or 'n', followed by 'r','i', or 'ri', for real part, imaginary part, or both, respectively.

plot_t = 'yri'  # 'yri'

# Plot phase tensor?
# 'y' or 'n'

plot_p = 'y'


PerLimits = (0.0001, 1.)  # AMT
# PerLimits = (0.001,100000.) #BBMT
# PerLimits = (0.00003,10000.) #AMT+BBMT
RhoLimits = (0.1, 10000.)
PhiLimits = (-180., 180.)
Tiplimits = (-.5, 0.5)
# Define the path to your EDI-files:
# edi_in_dir = r'/home/vrath/RRV_work/edi_work/Edited/'
edi_in_dir = r'/home/vrath/Desktop/MauTopo/MauEdi/'
# r'/home/vrath/MauTopo/MauTopo500_edi/'
# r'/home/vrath/RRV_work/edifiles_in/'
# edi_in_dir =  r'/home/vrath/RRV_work/edifiles_r1500m_bbmt/'
print(' Edifiles read from: %s' % edi_in_dir)

# Define the path for saving  plots:


plots_dir = edi_in_dir + 'data_plots/'
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
    # print(entry)
    if entry.endswith('.edi') and not entry.startswith('.'):
        edi_files.append(entry)

# Enter loop for plotting


for filename in edi_files:
    print('\n \n \n reading data from ' + filename)
    name, ext = os.path.splitext(filename)

# Create an MT object

    file_i = edi_in_dir + filename
    mt_obj = MT(file_i)
    print(' site %s at :  % 10.6f % 10.6f' % (name, mt_obj.lat, mt_obj.lon))

    if no_err is True:
        # mt_obj.Z.z_err = 0.0001*np.ones_like(np.real(mt_obj.Z.z))
        # mt_obj.Tipper.tipper_err = 0.0001*np.ones_like(np.real(mt_obj.Tipper.tipper))
        mt_obj.Z.z_err = 0.0001 * np.real(mt_obj.Z.z)
        mt_obj.Tipper.tipper_err = 0.0001 * np.real(mt_obj.Tipper.tipper)

    plot_obj = mt_obj.plot_mt_response(plot_num=plot_z,
                                       plot_tipper=plot_t,
                                       plot_pt=plot_p,
                                       x_limits=PerLimits,
                                       res_limits=RhoLimits,
                                       phase_limits=PhiLimits,
                                       tipper_limits=Tiplimits,
                                       no_err=True
                                       )

# Finally save figure

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
