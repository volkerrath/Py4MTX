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
# ---

'''
 
This script produces a site list containing site names, 
coordinates and elevations, e. g., for WALDIM analysis.

@author: sb & vr dec 2019
'''

# Import required modules

import os
from mtpy.core.mt import MT
from mtpy.analysis.geometry import dimensionality
import numpy as np


# Define the path to your EDI-files and for the list produced
edi_dir = r'/home/vrath/RRV_work/edi_work/'
print(' Edifiles read from: %s' % edi_dir)


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
# use the phase tensor to determine which frequencies are 1D/2D/3D
    dim = dimensionality(z_object=mt_obj.Z,
                     skew_threshold=5,          # threshold in skew angle (degrees) to determine if data are 3d
                     eccentricity_threshold=0.1 # threshold in phase ellipse eccentricity to determine if data are 2d (vs 1d)
                     )

    print(dim)
