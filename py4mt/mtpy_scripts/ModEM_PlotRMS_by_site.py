#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 04 13:13:29 2016

@author: Alison Kirkby

Plot root-mean-square misfit (RMS across all periods) at each site


"""
import matplotlib.pyplot as plt
from mtpy.modeling.modem import Residual
import os.path as op
import os
os.chdir(r'./')
#from mtpy.imaging.plot_response import PlotResponse

#### Inputs ####
wd = r'/home/vrath/WT6C/WT6_33/'
savepath = wd  # r'/home/vrath/WT6C/'
filestem = 'WT6_33_Zoffd_3_02_300_NLCG_006'

# datafn = op.join(wd,'WT6Z_I4_Zfull_centered.dat')
# respfn = op.join(wd,filestem+'.dat')


# read residual file into a residual object
residObj = Residual(residual_fn=op.join(wd, filestem + '.res'))
residObj.read_residual_file()
residObj.get_rms()


# get some parameters as attributes
lat, lon, east, north, rel_east, rel_north, rms, station = [
    residObj.rms_array[key] for key in [
        'lat', 'lon', 'east', 'north', 'rel_east', 'rel_north', 'rms', 'station']]

# create the figure
plt.figure()
plt.scatter(east, north, c=rms, cmap='bwr')
for i in range(len(station)):
    plt.text(east[i], north[i], station[i], fontsize=8)

plt.colorbar()
plt.clim(1, 4)
