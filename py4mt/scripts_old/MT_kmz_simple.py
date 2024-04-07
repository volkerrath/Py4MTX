#!/usr/bin/env python3

"""
@author: sb & vr oct 2019
"""

# Import required modules

import os
import sys
import csv

import numpy as np
from datetime import datetime


import simplekml
# from mtpy.core import mt, transfer_function
import mtpy.core
from mtpy.core.z import Z, Tipper
from mtpy.core.mt import MT
import matplotlib.pyplot as plt

mypath = ["/home/vrath/Py4MT/py4mt/modules/", "/home/vrath/Py4MT/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)


import util as utl
from version import versionstrg

PY4MT_ROOT = os.environ["PY4MT_ROOT"]
PY4MT_DATA = os.environ["PY4MT_DATA"]

version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")

cm = 1./2.54  # centimeters to inches

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


# Define the path to your EDI-files:
EdiDir = r"/home/vrath/rjmcmc_mt/work/edi/"
print(" Edifiles read from: %s" % EdiDir)
    
# Define the  path for saving  plots:
PltDir = r"/home/vrath/rjmcmc_mt/work/plots/"
print(" Plots written to: %s" % PltDir)
if not os.path.isdir(PltDir):
    print(" File: %s does not exist, but will be created" % PltDir)
    os.mkdir(PltDir)


KmlDir = r"/home/vrath/rjmcmc_mt/work/kml/"
KmlFile = r"ANN_v2.kml"
site_icolor = simplekml.Color.blue
site_rcolor = simplekml.Color.blue



icon_dir = PY4MT_ROOT+ "/py4mt/share/icons/"
site_icon =  icon_dir + "placemark_circle.png"
site_icolor = simplekml.Color.red
site_tcolor = simplekml.Color.white  # "#555500" #
site_tscale = 1.  # scale the text
site_iscale = 1.

# No changes required after this line!

# Construct list of EDI-files:
edi_files = []
files = os.listdir(EdiDir)
for entry in files:
    # print(entry)
    if entry.endswith(".edi") and not entry.startswith("."):
        edi_files.append(entry)
edi_files = sorted(edi_files)

    
# Open kml object:
 
if not os.path.isdir(KmlDir):
    print("File: %s does not exist, but will be created" % KmlDir)
    os.mkdir(KmlDir)
    
kml = simplekml.Kml(open=1)

site_iref = kml.addfile(site_icon)

# Loop over sites
    #print("reading data from "+filename)
    
for edi_name in edi_files:
    name, ext = os.path.splitext(edi_name)
    file_i = EdiDir + edi_name    
    mt_obj = MT()
    mt_obj.read(file_i)

    lat = mt_obj.station_metadata.location.latitude
    lon = mt_obj.station_metadata.location.longitude
    elev = mt_obj.station_metadata.location.elevation
    print(" site %s at :  % 10.6f % 10.6f % 8.1f" % (name, lat, lon, elev ))

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
        srcfile_1 = kml.addfile(PltDir + nam_1 + '.png')
        description_1 = (
            '<img width="800" align="left" src="' + srcfile_1 + '"/>'
        )
        description = description + description_1

    if plots_2:
        nam_2 = name + strng_2
        print(nam_2)
        srcfile_2 = kml.addfile(PltDir + nam_2 + '.png')
        description_2 = (
            '<img width="900" align="left" src="' + srcfile_2 + '"/>'
        )
        description = description + description_2

    site.description = description


kml_outfile = KmlDir + KmlFile

# # Save raw kml file:

if kml:
     kml.save(kml_outfile + ".kml")

# Compressed kmz file:
if kmz:
    kml.savekmz(kml_outfile + ".kmz")

print("Done. "+str(len(edi_files))+" files added.")
print("kml/z written to " + KmlFile)
