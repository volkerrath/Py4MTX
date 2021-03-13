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

'''

This script produces a site list containing site names,
coordinates and elevations, e. g., for WALDIM analysis.

@author: sb & vr dec 2019
'''

# Import required modules

import os
import csv
from mtpy.core.mt import MT
import numpy as np

dialect = 'unix'

# Define the path to your EDI-files and for the list produced


#edi_in_dir       = '/home/vrath/WT7_Svet_edited/set1/'
edi_in_dir = '/home/vrath/RainyRiverTranssect/RRV/hfreq_amt/'
print(' Edifiles read from: %s' % edi_in_dir)

# Define the path and append string for resampled data:

edi_out_dir = edi_in_dir + 'coords'
print(' Edifiles written to: %s' % edi_out_dir)
if not os.path.isdir(edi_out_dir):
    print(' File: %s does not exist, but will be created' % edi_out_dir)
    os.mkdir(edi_out_dir)

out_string = ''
# No changes required after this line!

# Construct list of edi-files:

edi_files = []
files = os.listdir(edi_in_dir)
for entry in files:
    if entry.endswith('.edi') and not entry.startswith('.'):
        edi_files.append(entry)
# Loop over edifiles:

for filename in edi_files:
    print('reading data from: ' + filename)
    name, ext = os.path.splitext(filename)
    file_i = edi_in_dir + filename

# Create MT object

    mt_obj = MT(file_i)
    Z = mt_obj.Z
    T = mt_obj.Tipper
    lon = mt_obj.lon
    lat = mt_obj.lat
    elev = mt_obj.elev
    east = mt_obj.east
    north = mt_obj.north
    sname = mt_obj.station
    print(' site %s at :  % 10.6f % 10.6f' % (sname, mt_obj.lat, mt_obj.lon))
    file_out = name + out_string + ext

    mt_obj.write_mt_file(save_dir=edi_out_dir,
                         fn_basename=file_out,
                         file_type='edi',  # edi or xml format
                         new_Z_obj=Z,  # provide a z object to update the data
                         new_Tipper_obj=T,  # provide a tipper object to update the data
                         longitude_format='LONG',  # write longitudes as 'LON' or 'LONG'
                         # write as decimal degrees (any other input
                         latlon_format='dd'
                         # will write as degrees minutes seconds
                         )

    print('Writing data to ' + edi_out_dir + file_out)
