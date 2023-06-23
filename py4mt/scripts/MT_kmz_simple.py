#!/usr/bin/env python3

"""
@author: sb & vr oct 2019
"""

# Import required modules

import os
import sys
import csv

mypath = ["/home/vrath/Py4MT/py4mt/modules/", "/home/vrath/Py4MT/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)


import util
import numpy as np

import simplekml
# from mtpy.core import mt, transfer_function
import mtpy.core
from mtpy.core.z import Z, Tipper
from mtpy.core.mt import MT
import matplotlib.pyplot as plt

# Determine what is added to the KML-tags:
# plots_1 =  site plots produced by MT_siteplot.py, with an
# additional string strng_1 added to the EDI basename.
# plots_2 =  plots produced by MT_mcmcplot.py or  from
# other sources, with an additional string strn_2 added to the EDI basename.


plots_1 = False
strng_1 = "_data"

plots_2 = False
strng_2 = "_edi_imp_rjmcmc"


kml = False
kmz = True
# Define the path to your EDI-files

edi_dir = "/home/vrath/MT_Data/Opf/2023/edi_plus/"
print(" Edifiles read from: %s" % edi_dir)

kml_dir = r"/home/vrath/MT_Data/Opf/2023/"
kml_file = r"OldOpf"

plots_dir =   r"/home/vrath/MT_Data/Opf/2023/plots/"    
print(" Plots read from: %s" % plots_dir)



kml_dir = r"/home/vrath/MT_Data/Opf/2023/"
kml_file = r"OldOpf"
site_icolor = simplekml.Color.blue
site_rcolor = simplekml.Color.blue



icon_dir = r"/home/vrath/GoogleEarth/icons/"
site_icon =  icon_dir + "placemark_circle.png"
site_icolor = simplekml.Color.red
site_tcolor = simplekml.Color.white  # "#555500" #
site_tscale = 1.  # scale the text
site_iscale = 1.

# No changes required after this line!

# Construct list of EDI-files:
edi_files = []
files = os.listdir(edi_dir)
for entry in files:
    # print(entry)
    if entry.endswith(".edi") and not entry.startswith("."):
        edi_files.append(entry)
edi_files = sorted(edi_files)

    
# Open kml object:
 
if not os.path.isdir(kml_dir):
    print("File: %s does not exist, but will be created" % kml_dir)
    os.mkdir(kml_dir)
    
kml = simplekml.Kml(open=1)

site_iref = kml.addfile(site_icon)

# Loop over sites
    #print("reading data from "+filename)
    
for edi_name in edi_files:
    name, ext = os.path.splitext(edi_name)
    file_i = edi_dir + edi_name    
    
    mt_obj   = MT(file_i)
    lat = mt_obj.lat
    lon = mt_obj.lon
    elev = mt_obj.elev
    print(" site %s at :  % 10.6f % 10.6f" % (name, lat, lon))
       
    site = kml.newpoint(name=edi_name)
    site.coords = [(lon, lat, elev)]
#  Now add the plots to tag:
    normal = True
    if normal:
        site.style.labelstyle.color = site_tcolor
        site.style.labelstyle.scale = site_tscale
        site.style.iconstyle.icon.href = site_iref
        site.style.iconstyle.scale = site_iscale
        site.style.iconstyle.color = site_icolor
        description = " data from site : "+edi_name

    if plots_1:
        nam_1 = name + strng_1
        print(nam_1)
        srcfile_1 = kml.addfile(plots_dir + nam_1 + '.png')
        description_1 = (
            '<img width="800" align="left" src="' + srcfile_1 + '"/>'
        )
        description = description + description_1

    if plots_2:
        nam_2 = name + strng_2
        print(nam_2)
        srcfile_2 = kml.addfile(plots_dir + nam_2 + '.png')
        description_2 = (
            '<img width="900" align="left" src="' + srcfile_2 + '"/>'
        )
        description = description + description_2

    site.description = description


kml_outfile = kml_dir + kml_file

# # Save raw kml file:

if kml:
     kml.save(kml_outfile + ".kml")

# Compressed kmz file:
if kmz:
    kml.savekmz(kml_outfile + ".kmz")

print("Done. "+str(len(edi_files))+" files added.")
print("kml/z written to " + kml_file)
