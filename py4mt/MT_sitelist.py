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
delim = ','
whatfor = 'nix'

# Define the path to your EDI-files and for the list produced
# edi_dir = r'/home/vrath/RRV_work/edi_work/BBMT/'
edi_dir = r'/home/vrath/Work/MAUR9VR/orig/edis_AB1_dist_ss_dist_just/'
# r'/home/vrath/Desktop/MauTopo/MauEdi/'
# r'/media/vrath/MT/Ireland/Northwest_CarboniferousBasin/MT_DATA/EDI/'
print(' Edifiles read from: %s' % edi_dir)
csv_file =edi_dir+'Sitelist.csv'
print('Writing data to CSV file: '+csv_file)


# No changes required after this line!

# Construct list of edi-files:

edi_files=[]
files= os.listdir(edi_dir) 
for entry in files:
   # print(entry)
   if entry.endswith('.edi') and not entry.startswith('.'):
            edi_files.append(entry)
ns =  np.size(edi_files)

# Outputfile (e. g., for WALDIM analysis)

with open(csv_file, 'w') as f:
    sitelist = csv.writer(f, delimiter=delim)
    if whatfor != 'waldim':
        sitelist.writerow(['Sitename', 'Latitude', 'Longitude'])
        sitelist.writerow([ns, ' ', ' '])

# Loop over edifiles:

    for filename in edi_files :
        print('reading data from: '+filename)
        name, ext = os.path.splitext(filename)
        file_i = edi_dir+filename

# Create MT object

        mt_obj = MT(file_i)
        lon = mt_obj.lon
        lat = mt_obj.lat
        elev = mt_obj.elev
        east = mt_obj.east
        north = mt_obj.north
        # sitename = mt_obj.station
        sitelist.writerow([name, lat, lon, elev])
