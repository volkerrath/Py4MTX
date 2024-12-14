#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

This script produces a site list containing site names,
coordinates and elevations, e. g., for WALDIM analysis.

@author: sb & vr dec 2019
"""

# Import required modules

import os
import sys
from sys import exit as error
import re

import numpy as np

from mtpy.core.mt import MT
# import mtpy.core.mt as mt

PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]
mypath = [PY4MTX_ROOT+"/py4mt/modules/", PY4MTX_ROOT+"/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)

PY4MTX_DATA = os.environ["PY4MTX_DATA"]


import util as utl
import mtproc as mtp
from version import versionstrg


version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")

PY4MTX_DATA =  "/home/vrath/MT_Data/"
WorkDir = PY4MTX_DATA+"/Ubaye_best/"

LatLonFmt = "dd"
LONG=False


EdiDir_in = WorkDir+"/edis/"
print(" Edifiles read from: %s" % EdiDir_in)
EdiDir_out = WorkDir+"/edis_dd/"
if not os.path.isdir(EdiDir_out):
    print("File: %s does not exist, but will be created" % EdiDir_out)
    os.mkdir(EdiDir_out)

# Construct list of edi-files:
edi_files = mtp.get_edi_list(EdiDir_in, fullpath=False)
ns = np.size(edi_files)

for filename in edi_files:
    print("\n Reading data from " + EdiDir_in + filename)
    name, ext = os.path.splitext(filename)

# Create an MT object

    file = EdiDir_in + filename
    mt_obj = MT()
    mt_obj.read(file)


    file_out = EdiDir_out+os.path.basename(file)
    print(" Writing data to " + file_out)
    if LONG:
        mt_obj.write(file_out, latlon_format=LatLonFmt, longitude_format="LONG")
    else:
        mt_obj.write(file_out, latlon_format=LatLonFmt)


# for file in edi_files:
#     print("reading data from: " + file)
#     # name, ext = os.path.splitext(file)
#     filein = file
#     filout = EdiDir_out+os.path.basename(file)
#     print("writing data to: " + filout)
#     fo = open(filout, "w")
#     with open(filein) as fi:
#         for line in fi:
#             if ("lat" in line.lower()):
#                 parts = re.split("[^\d\w]+", line)
#                 parts[3]= ".".join((parts[3],parts[4]))
#                 # print(line)
#                 lat = float(parts[1]) + float(parts[2])/60 + float(parts[3])/(60*60)
#                 if digits!=0:
#                     lat = round(lat,digits)
#                 line= (parts[0]+"="+str(lat)+"\n")
#                 fo.write(line)
#             elif ("lon" in line.lower()):
#                 parts = re.split("[^\d\w]+", line)
#                 parts[3]= ".".join((parts[3],parts[4]))
#                 lon = float(parts[1]) + float(parts[2])/60 + float(parts[3])/(60*60)
#                 if digits!=0:
#                     lon = round(lon,6)
#                 line= (parts[0]+"="+str(lon)+"\n")
#                 fo.write(line)
#             else:
#                 fo.write(line)


print("Done")
