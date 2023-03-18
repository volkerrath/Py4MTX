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



import simplekml
from mtpy.core.mt import MT

# Determine what is added to the KML-tags:
# plots_1 =  site plots produced by MT_siteplot.py, with an
# additional string strng_1 added to the EDI basename.
# plots_2 =  plots produced by MT_mcmcplot.py or  from
# other sources, with an additional string strn_2 added to the EDI basename.

plots_1 = True
strng_1 = "_data"

plots_2 = True
strng_2 = "_edi_imp_rjmcmc"

repeat = False
repeat_string ="XXXXX"# "R"

reproc = False
reproc_string ="XXXXX"# "N"

kml = False
kmz = True
# Define the path to your EDI-files

# edi_dir = r"/home/vrath/MT_Data/Naser/Limerick2023/mt/reprocessed_quality/reprocessed_bad/"
# edi_dir = r"/home/vrath/MT_Data/Naser/Limerick2023/mt/reprocessed_quality/reprocessed_good/"
# edi_dir = r"/home/vrath/MT_Data/Naser/Limerick2023/mt/reprocessed_quality/reprocessed_ugly/"
edi_dir = r"/home/vrath/MT_Data/Naser/Limerick2023/mt/reprocessed_final/"
print(" Edifiles read from: %s" % edi_dir)

if plots_1 or plots_2:
    # plots_dir =   r"/home/vrath/MT_Data/Naser/Limerick2023/mt/reprocessed_quality/reprocessed_bad/"
    # plots_dir =   r"/home/vrath/MT_Data/Naser/Limerick2023/mt/reprocessed_quality/reprocessed_good/"
    #plots_dir =   r"/home/vrath/MT_Data/Naser/Limerick2023/mt/reprocessed_quality/reprocessed_ugly/"
    plots_dir =   r"/home/vrath/MT_Data/Naser/Limerick2023/mt/reprocessed_plots_3km/"
    print(" Plots read from: %s" % plots_dir)


# Determine which geographical info is added to the KML-tags:
# define empty list
places = []

# open file and read the content in a list
places_file = edi_dir + "Sitelist.csv"
# r"/home/vrath/Py4MT/py4mt/M/FWD/Sitelist.csv"
# r"/home/vrath/WestTimor/places.csv"
with open(places_file, "r") as f:
    placelist = csv.reader(f, delimiter=" ")
    for row in placelist:
        print(row)

# Define the path for saving  kml files


kml_dir = r"/home/vrath/MT_Data/Naser/Limerick2023/mt/"
# kml_file = r"Limerick_reprocessed_good"
# site_icolor = simplekml.Color.green
# site_rcolor = simplekml.Color.green
# kml_file = r"Limerick_reprocessed_ugly"
# site_icolor = simplekml.Color.yellow
# site_rcolor = simplekml.Color.yellow
# kml_file = r"Limerick_reprocessed_bad"
# site_icolor = simplekml.Color.red
# site_rcolor = simplekml.Color.red
kml_file = r"Limerick_reprocessed_3km"
site_icolor = simplekml.Color.blue
site_rcolor = simplekml.Color.blue



icon_dir = r"/home/vrath/GoogleEarth/icons/"
site_icon =  icon_dir + "placemark_circle.png"
site_icon_rept =  icon_dir + "placemark_circle.png"
site_icon_repr =  icon_dir + "placemark_circle.png"

site_tcolor = simplekml.Color.white  # "#555500" #
site_tscale = 1.2  # scale the text

site_iscale = 1.5
# site_icolor = simplekml.Color.blue
# site_rcolor = simplekml.Color.blue

site_icolor_rept = simplekml.Color.yellow
site_rcolor_rept = simplekml.Color.yellow

site_icolor_repr = simplekml.Color.red
site_rcolor_repr = simplekml.Color.red
# simplekml.Color.rgb(0, 0, 255)
# "ffff0000"


# No changes required after this line!

# Construct list of EDI-files:


edi_files = []
files = os.listdir(edi_dir)
for entry in files:
   # print(entry)
    if entry.endswith(".edi") and not entry.startswith("."):
        edi_files.append(entry)


# Open kml object:

kml = simplekml.Kml(open=1)

site_iref = kml.addfile(site_icon)
site_iref_rept = kml.addfile(site_icon_rept)
site_iref_repr = kml.addfile(site_icon_repr)

# Loop over edifiles

for filename in edi_files:
    #print("reading data from "+filename)
    name, ext = os.path.splitext(filename)
    file_i = edi_dir + filename

# Create MT object

    mt_obj = MT(file_i)
    lon = mt_obj.lon
    lat = mt_obj.lat
    hgt = mt_obj.elev
    full_name = mt_obj.station
    # nam = full_name[3:]
    # print(full_name, nam,plots_dir+full_name+".png")
    # print(full_name)
    description = ("")

    site = kml.newpoint(name=name)
    site.coords = [(lon, lat, hgt)]
#  Now add the plots to tag:
    normal = (not repeat_string in name[-1]) and (not reproc_string in name[-1])
    if normal:
        site.style.labelstyle.color = site_tcolor
        site.style.labelstyle.scale = site_tscale
        site.style.iconstyle.icon.href = site_iref
        site.style.iconstyle.scale = site_iscale
        site.style.iconstyle.color = site_icolor
        description = name

    if repeat:
        if name.endswith(repeat_string):
            site.style.iconstyle.icon.href = site_iref_rept
            site.style.labelstyle.color = site_rcolor_rept
            site.style.iconstyle.color = site_icolor_rept
            site.style.iconstyle.scale = site_iscale
            site.style.labelstyle.scale = site_tscale
            site.style.balloonstyle.textcolor = site_rcolor
            print(name+" is repeated site")
            description = name + " - repeated site"

    if reproc:
        if name.endswith(reproc_string):
            site.style.iconstyle.icon.href = site_iref_repr
            site.style.labelstyle.color = site_rcolor_repr
            site.style.iconstyle.color = site_icolor_repr
            site.style.iconstyle.scale = site_iscale
            site.style.labelstyle.scale = site_tscale
            print(name+" is reprocessed site")
            description = name + " - reprocessed site"

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

# Save raw kml file:

if kml:
    kml.save(kml_outfile + ".kml")

# Compressed kmz file:
if kmz:
    kml.savekmz(kml_outfile + ".kmz")

print("Done. "+str(len(edi_files))+" files in list.")
print("kml/z written to " + kml_file)
