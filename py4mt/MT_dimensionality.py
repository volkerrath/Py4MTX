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

edi_files = []
files = os.listdir(edi_dir)
for entry in files:
    # print(entry)
    if entry.endswith('.edi') and not entry.startswith('.'):
        edi_files.append(entry)
ns = np.size(edi_files)

# Outputfile (e. g., for WALDIM analysis)


# Loop over edifiles:
n3d = 0
n2d = 0
n1d = 0
nel = 0
sit = 0
for filename in edi_files:
    sit = sit + 1
    print('reading data from: ' + filename)
    name, ext = os.path.splitext(filename)
    file_i = edi_dir + filename

# Create MT object

    mt_obj = MT(file_i)
    lon = mt_obj.lon
    lat = mt_obj.lat
    elev = mt_obj.elev
    east = mt_obj.east
    north = mt_obj.north
# use the phase tensor to determine which frequencies are 1D/2D/3D
    dim = dimensionality(z_object=mt_obj.Z,
                         skew_threshold=5,
                         # threshold in skew angle (degrees) to determine if
                         # data are 3d
                         # threshold in phase ellipse eccentricity to determine
                         # if data are 2d (vs 1d)
                         eccentricity_threshold=0.1
                         )

    nel = nel + np.size(dim)
    n1d = n1d + sum(map(lambda x: x == 1, dim))
    n2d = n2d + sum(map(lambda x: x == 2, dim))
    n3d = n3d + sum(map(lambda x: x == 3, dim))


print('number of sites = ' + str(sit))
print('total number of elements = ' + str(nel))
print('  number of undetermined elements = ' +
      str(nel - n1d - n2d - n3d) + '\n')
print('  number of 1-D elements = ' + str(n1d) +
      '  (' + str(round(100 * n1d / nel)) + '%)')
print('  number of 2-D elements = ' + str(n2d) +
      '  (' + str(round(100 * n2d / nel)) + '%)')
print('  number of 3-D elements = ' + str(n3d) +
      '  (' + str(round(100 * n3d / nel)) + '%)')
