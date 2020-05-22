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
from mtpy.core.mt import MT

# Graphical paramter. Determine the plot formats produced, 
# and the required resolution: 

plot_pdf=True
plot_png=True
plot_eps=False

dpi = 400

# What should be plotted?
# 1 = yx and xy; 2 = all 4 components
# 3 = off diagonal + determinant

plot_z = 3

# Plot tipper?
# 'y' or 'n', followed by 'r','i', or 'ri', for real part, imaginary part, or both, respectively.

plot_t = 'n'

# Plot phase tensor?
# 'y' or 'n'

plot_p  = 'y'


# PerLimits = (0.0001,10.) #AMT
# PerLimits = (0.001,10000.) #BBMT
PerLimits = (0.0001,1000.) #AMT+BBMT
RhoLimits = (0.1 ,100000.)
PhiLimits = (-10.,100.)

# Define the path to your EDI-files:
edi_in_dir =  r'/home/vrath/RRV_work/edifiles_out1/'
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


edi_files=[]
files= os.listdir(edi_in_dir) 
for entry in files:
   # print(entry)
   if entry.endswith('.edi') and not entry.startswith('.'):
            edi_files.append(entry)

# Enter loop for plotting


for filename in edi_files :
    print('\n \n \n reading data from '+filename)
    name, ext = os.path.splitext(filename)

# Create an MT object 

    file_i = edi_in_dir+filename
    mt_obj = MT(file_i)
    print(' site %s at :  % 10.6f % 10.6f' % (name, mt_obj.lat, mt_obj.lon))
    plot_obj = mt_obj.plot_mt_response(plot_num=plot_z,
                                     plot_tipper = plot_t,
                                     plot_pt = plot_p,
                                     x_limits = PerLimits,
                                     res_limits=RhoLimits, 
                                     phase_limits=PhiLimits
    )

# Finally save figure

    if plot_png:
        plot_obj.save_plot(os.path.join(plots_dir,name+".png"),file_format='png',fig_dpi=dpi)
    if plot_pdf:
        plot_obj.save_plot(os.path.join(plots_dir,name+".pdf"),file_format='pdf',fig_dpi=dpi)
    if plot_eps:
        plot_obj.save_plot(os.path.join(plots_dir,name+".eps"),file_format='eps',fig_dpi=dpi)

        

  
