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
#       jupytext_version: 1.11.3
# ---

'''

This script produces a site list containing site names,
coordinates and elevations, e. g., for WALDIM analysis.

@author: sb & vr dec 2019
'''

# Import required modules

import os
import sys

import csv
import inspect

import numpy as np




PY4MTX_ROOT = os.environ['PY4MTX_ROOT']
PY4MTX_DATA = os.environ['PY4MTX_DATA']

mypath = [PY4MTX_ROOT+'/py4mt/modules/', PY4MTX_ROOT+'/py4mt/scripts/']
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)

import util as utl
from version import versionstrg
from data_proc import load_edi, save_edi, save_npz


PY4MTX_DATA = os.environ['PY4MTX_DATA']

version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=inspect.getfile(inspect.currentframe()), out=False)
print(titstrng+'\n\n')

Coords = 'utm'
# EPSG =  32631
EPSG =   None

dialect = 'unix'
delim = ','
whatfor = 'femtic'
whatfor = 'wal'
# whatfor = 'kml'
if  'wal' in whatfor:
    delim = ' '
    Coords= 'latlon'


# Define the path to your EDI-files and for the list produced
# PY4MTX_DATA = '/home/vrath/Py4MTX/work/results_ploting/'
WorkDir = '/home/vrath/MT_Data/waldim/'
EdiDir =  WorkDir + '/edi_jc/'
# EdiDir =  WorkDir + '/edi_noss/'
# EdiDir =  WorkDir + '/edi_eps/'

print(' Edifiles read from: %s' % EdiDir)

if  'wal' in whatfor.lower():
    CSVFile = EdiDir + 'Sitelist_waldim.txt'
elif 'fem' in whatfor.lower():
    CSVFile = EdiDir + 'Sitelist_femtic.txt'
else:
    CSVFile = EdiDir + 'Sitelist.txt'
print('Writing data to file: ' + CSVFile)

# No changes required after this line!

# Construct list of edi-files:

edi_files = []
files = os.listdir(EdiDir)
for entry in files:
    # print(entry)
    if entry.endswith('.edi') and not entry.startswith('.'):
        edi_files.append(entry)
ns = np.size(edi_files)
edi_files = sorted(edi_files)

# Outputfile (e. g., for WALDIM analysis)

with open(CSVFile, 'w') as f:

    sitelist = csv.writer(f, delimiter=delim)
    if 'wal' in whatfor.lower():
        sitelist.writerow(['Sitename', 'Latitude', 'Longitude'])
        sitelist.writerow([ns, ' ', ' '])

# Loop over edifiles:
    sitenum=-1

    for filename in edi_files:
        print('reading data from: ' + filename)
        name, ext = os.path.splitext(filename)
        file_i = EdiDir + filename
        sitenum = sitenum + 1
        edi_dict = load_edi(file_i, drop_invalid_periods=True)

        station = edi_dict['station']


        lat = edi_dict['lat']
        lon = edi_dict['lon']
        elev = edi_dict['elev']

        if 'utm' in Coords.lower():
            if EPSG is not None:
                easting, northing =  utl.proj_latlon_to_utm(latitude=lat, longitude=lon, utm_zone=EPSG)
            else:
                EPSG = utl.get_utm_zone(lat=lat, lon=lon)
                sys.exit('make sitelist: utm required, but no EPSG given!. Exit.')


        # sitename = mt_obj.station
        if 'wal' in whatfor:
            sitelist.writerow([name, lat, lon])
        elif 'fem' in whatfor:
            sitelist.writerow([name, lat, lon, elev, sitenum])
        else:
            sitelist.writerow([name, lat, lon, elev])
