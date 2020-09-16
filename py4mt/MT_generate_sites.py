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
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

"""
generate pseudo dat for forward modelling studies

@author: sb & vr July 2020

"""

# Import required modules

import os
from mtpy.core.mt import MT
import numpy as np


# Define the path to your EDI-files:

edi_template = r'/home/vrath/Py4MT/py4mt/M/Template.edi'
print(' Edifile template read from: %s' % edi_template)

# Define the path and appended string for saved EDI-files:

edi_out_dir=r'/home/vrath/Py4MT/py4mt/M/MauTopo_dense/'
print(' Edifiles written to: %s' % edi_out_dir)
if not os.path.isdir(edi_out_dir):
    print(' File: %s does not exist, but will be created' % edi_out_dir)
    os.mkdir(edi_out_dir)

OutName = ''

# Construct list of EDI-files:

small = 0.000001
LonLimits = ( 6.275, 6.39)
nLon = 31
LonStep  = (LonLimits[1] - LonLimits[0])/nLon
Lon = np.arange(LonLimits[0],LonLimits[1]+small,LonStep)

LatLimits = (45.37,45.46)
nLat = 31
LatStep  = (LatLimits[1] - LatLimits[0])/nLat
Lat = np.arange(LatLimits[0],LatLimits[1]+small,LatStep)



# No changes required after this line!

# Enter loop:
nn = -1
for latval in Lat:
    nn=nn+1
    nnstr = str(nn)
    mm = -1
    print(nnstr)
    for lonval in Lon:
        mm=mm+1
        mmstr = str(mm)
        print(mmstr)


# # Create an MT object 

        file_in = edi_template
        mt_tmp = MT(file_in)
    
        mt_tmp.lat = Lat[nn]
        mt_tmp.lon = Lon[mm]
        mt_tmp.station = OutName+nnstr+'_'+mmstr
    
        file_out = OutName+nnstr+'_'+mmstr+'.edi'
    
        print('\n Generating '+edi_out_dir+file_out)
        print(' site %s at :  % 10.6f % 10.6f' % (mt_tmp.station, mt_tmp.lat, mt_tmp.lon))

# # Write a new edi file:

        print('Writing data to '+edi_out_dir+file_out)
        mt_tmp.write_mt_file(
            save_dir=edi_out_dir,
            fn_basename=file_out,
            file_type='edi',
            longitude_format='LONG',
            latlon_format='dd'
            )

  
