#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

This script produces a site list containing site names,
coordinates and elevations, e. g., for WALDIM analysis.

@author: sb & vr dec 2019
"""

# Import required modules

import os
import re
import numpy as np

edi_in_dir = r"/home/vrath/work/MT_Data/Opf/2023/zips/GLW-ROT/"
print(" Edifiles read from: %s" % edi_in_dir)
edi_out_dir = r"/home/vrath/work/MT_Data/Opf/2023/edi_new/"
if not os.path.isdir(edi_out_dir):
    print("File: %s does not exist, but will be created" % edi_out_dir)
    os.mkdir(edi_out_dir)

# Construct list of edi-files:

edi_files = []
files = os.listdir(edi_in_dir)
for entry in files:
    # print(entry)
    if entry.endswith(".edi") and not entry.startswith("."):
        edi_files.append(edi_in_dir+entry)
ns = np.size(edi_files)
if ns ==0:
    error("No edi files found in "+edi_in_dir+"! Exit.")


for file in edi_files:
    print("reading data from: " + file)
    # name, ext = os.path.splitext(file)
    filein = file
    filout = edi_out_dir+os.path.basename(file)
    print("writing data to: " + filout)
    fo = open(filout, "w")
    with open(filein) as fi:
        for line in fi:
            if ("lat" in line.lower()):
                parts = re.split('[^\d\w]+', line)
                parts[3]= ".".join((parts[3],parts[4]))
                # print(line)
                lat = float(parts[1]) + float(parts[2])/60 + float(parts[3])/(60*60)
                line= (parts[0]+"="+str(lat)+"\n")           
                fo.write(line)
            elif ("lon" in line.lower()):
                parts = re.split('[^\d\w]+', line)
                parts[3]= ".".join((parts[3],parts[4]))
                lon = float(parts[1]) + float(parts[2])/60 + float(parts[3])/(60*60)
                line= (parts[0]+"="+str(lon)+"\n")
                fo.write(line)
            else:
                fo.write(line)
            

    # print('Done')
