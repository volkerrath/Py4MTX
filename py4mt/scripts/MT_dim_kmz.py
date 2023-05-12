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
#       format_version: "1.5"
#       jupytext_version: 1.11.3
# ---

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


import numpy

import simplekml
from mtpy.core.mt import MT

# Define the path to your EDI-files

# edi_dir = r"/home/vrath/MT_Data/Naser/Limerick2023/mt/reprocessed_quality/reprocessed_bad/"
# edi_dir = r"/home/vrath/MT_Data/Naser/Limerick2023/mt/reprocessed_quality/reprocessed_good/"
# edi_dir = r"/home/vrath/MT_Data/Naser/Limerick2023/mt/reprocessed_quality/reprocessed_ugly/"
edi_dir = r"/home/vrath/MT_Data/Peru/Tacna/edi/"
print(" Edifiles read from: %s" % edi_dir)


# open file and read the content in a list
places_file = edi_dir + "Sitelist.csv"

tmp = []
with open(places_file, "r") as f:
    place_list = csv.reader(f, delimiter=" ")
    for site in place_list:
        tmp.append(site)
tmp = tmp[2:]


places = []
for row in tmp:
    for ii in range(1,3):
        row[ii] = float(row[ii])
    places.append(row)

# print(places)
# Define the path for saving  kml files
kml = False
kmz = True
kml_dir = edi_dir
kml_file = r"Tacna2023_dim"

icon_dir = r"/home/vrath/GoogleEarth/icons/"
site_icon =  icon_dir + "placemark_circle.png"

site_tcolor = simplekml.Color.white  # "#555500" #
site_tscale = 1.2  # scale the text
site_iscale = 1.5

site_icolor_none = simplekml.Color.yellow
site_rcolor_none= simplekml.Color.yellow
site_icolor_1d = simplekml.Color.blue
site_rcolor_1d = simplekml.Color.blue
site_icolor_2d = simplekml.Color.green
site_rcolor_2d = simplekml.Color.green
site_icolor_3d = simplekml.Color.red
site_rcolor_3d = simplekml.Color.red
# simplekml.Color.rgb(0, 0, 255)
# "ffff0000"


# No changes required after this line!
# Construct list of EDI-files:


# Open kml object:

kml = simplekml.Kml(open=1)

site_iref = kml.addfile(site_icon)

# Determine unique freqs.

freqs = []
for site in places:    
    nam = site[0]
    freqs = []
    with open(edi_dir+nam+"_dim.dat", "r") as f:
        tmp = csv.reader(f)

        for site in tmp:
            site = site[0].split()            
            freqs.append(float(site[0]))
            
freqs = numpy.unique(freqs)

nam = []
lat = []
lon = []
sit = []
for site in places:
    print(site[0])
    nam.append(site[0])
    lat.append(site[1])
    lon.append(site[2])

    frq = []
    dim = []
    with open(edi_dir+site[0]+"_dim.dat", "r") as f:
        tmp = csv.reader(f)
        for site in tmp:
            site = site[0].split() 
            frq.append(float(site[0]))
            dim.append(int(site[1]))
            
        dim = numpy.asarray(dim) 
        frq = numpy.asarray(frq)    
        lst = numpy.vstack((frq, dim))
        
    sit.append(lst)
         
ifr = 0
for f in freqs:
    Nams = []
    Lats = []
    Lons = []
    Dims = []
    

    ff = numpy.log10(f)
    ifr = ifr + 1
    
    freq_strng = "Freq"+str(ifr)
    freqfolder = kml.newfolder(name=freq_strng)

    ns = len(nam)
    for isit in numpy.arange(ns):
        sf, sd = sit[isit]
        nf = numpy.shape(sf)[0]
    
        for ii in numpy.arange(nf):
            fs = numpy.log10(sf[ii])
            # print(ff,fs)
            if numpy.isclose(ff, fs, rtol=1e-2, atol=0.):
                # print("found  ", ff, fs)
                Nams.append(nam[isit])
                Lats.append(lat[isit])
                Lons.append(lon[isit])
                Dims.append(int(sd[ii]))
                
        # F.append([f, Nams, Lats, Lons, Dims])
 
    nsites =len(Nams)
    # print (nsites)
    for ii in numpy.arange(nsites):
        site = freqfolder.newpoint()
        site.coords = [(Lons[ii], Lats[ii], 0.)]

        if Dims[ii]==0:
            site.style.labelstyle.color = site_tcolor
            site.style.labelstyle.scale = site_tscale
            site.style.iconstyle.icon.href = site_icon
            site.style.iconstyle.scale = site_iscale
            site.style.iconstyle.color = site_icolor_none
            site.description ="undetermined"
        if Dims[ii]==1:
            site.style.labelstyle.color = site_tcolor
            site.style.labelstyle.scale = site_tscale
            site.style.iconstyle.icon.href = site_icon
            site.style.iconstyle.scale = site_iscale
            site.style.iconstyle.color = site_icolor_1d
            site.description ="1-D"
        if Dims[ii]==2:
            site.style.labelstyle.color = site_tcolor
            site.style.labelstyle.scale = site_tscale
            site.style.iconstyle.icon.href = site_icon
            site.style.iconstyle.scale = site_iscale
            site.style.iconstyle.color = site_icolor_2d
            site.description ="2-D"
        if Dims[ii]==3:
            site.style.labelstyle.color = site_tcolor
            site.style.labelstyle.scale = site_tscale
            site.style.iconstyle.icon.href = site_icon
            site.style.iconstyle.scale = site_iscale
            site.style.iconstyle.color = site_icolor_3d
            site.description ="3-D"               
 
    
kml_outfile = kml_dir + kml_file

# Save raw kml file:

if kml:
    kml.save(kml_outfile + ".kml")

# Compressed kmz file:
if kmz:
    kml.savekmz(kml_outfile + ".kmz")

print("Done. kml/z written to " + kml_outfile)
