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
#       jupytext_version: 1.9.1
# ---

'''
@author: sb & vr oct 2019
'''

# Import required modules

import os
import csv
import simplekml 
from mtpy.core.mt import MT

# Determine what is added to the KML-tags:
# plots_1 =  site plots produced by MT_siteplot.py, with an 
# additional string strng_1 added to the EDI basename.
# plots_2 =  plots produced by MT_mcmcplot.py or  from
# other sources, with an additional string strn_2 added to the EDI basename.

plots_1 = True
strng_1 = '' #'_data'

plots_2 = True
strng_2 ='_fwdres500'

repeats = False
repeat_string = 'r'

bads = False
bad_string = 'x'

# Define the path to your EDI-files

# edi_dir = r'/home/vrath/WestTimor/WT8C_edi/'
edi_dir = r'/home/vrath/Desktop/MauTopo/MauEdi/'
#'/home/vrath/Py4MT/py4mt/M/MauTopo_fwd/'
# r'/media/vrath/MT/Ireland/Donegal/Donegal_EDIs_3DGridEdited/'
print(' Edifiles read from: %s' % edi_dir)

if plots_1 or plots_2:
    plots_dir= edi_dir+'data_plots/'
    # r'/media/vrath/MT/Ireland/Northwest_CarboniferousBasin/MT_DATA/EDI/data_plots/'
    # r'/home/vrath/WestTimor/WT8C_plots/' #edi_dir #'NEW_plots_bbmt_roi_edit/'
    print(' Plots read from: %s' % plots_dir)


# Determine which geographical info is added to the KML-tags:
# define empty list
places = []

# open file and read the content in a list
places_file = edi_dir+'Sitelist.csv'
#r'/home/vrath/Py4MT/py4mt/M/FWD/Sitelist.csv'
# r'/home/vrath/WestTimor/places.csv'
with open(places_file, 'r') as f:
    placelist = csv.reader(f, delimiter=' ')
    for row in placelist:
        print(row)

# Define the path for saving  kml files

kml_dir = ''
kml_file = edi_dir+'MauSites'


site_icon   = 'icons/star.png'
site_icon_rept  = 'icons/donut.png'
site_icon_bad  = 'icons/triangle.png'

site_tcolor = simplekml.Color.white ###'#555500' #
site_tscale = 0.8  # scale the text 

site_iscale = 1.
site_icolor = simplekml.Color.blue
site_rcolor = simplekml.Color.blue 

site_icolor_rept = simplekml.Color.green
site_rcolor_rept = simplekml.Color.green 

site_icolor_bad = simplekml.Color.red
site_rcolor_bad = simplekml.Color.red 
   # simplekml.Color.rgb(0, 0, 255)
    # 'ffff0000'


# No changes required after this line!

# Construct list of EDI-files:


edi_files=[]
files= os.listdir(edi_dir) 
for entry in files:
   # print(entry)
   if entry.endswith('.edi') and not entry.startswith('.'):
            edi_files.append(entry)


# Open kml object:

kml = simplekml.Kml(open=1)

site_iref           = kml.addfile(site_icon)
site_iref_rept      =  kml.addfile(site_icon_rept)
site_iref_bad       =  kml.addfile(site_icon_bad)

# Loop over edifiles

for filename in edi_files :
    #print('reading data from '+filename)
    name, ext = os.path.splitext(filename)
    file_i = edi_dir+filename

# Create MT object

    mt_obj = MT(file_i)
    lon = mt_obj.lon
    lat = mt_obj.lat
    hgt = mt_obj.elev
    full_name = mt_obj.station
    # nam = full_name[3:] 
    nam = full_name
    # print(full_name, nam,plots_dir+full_name+'.png')
    # print(full_name)
    description = ('')

#  Now add the plots to tag:

    if plots_1:
        nam_1 = name+strng_1
        print(nam_1)
        srcfile_1 = kml.addfile(plots_dir+nam_1+'.png')
        description_1 = (
            '<img width="800" align="left" src="'+srcfile_1+'"/>'
            )
        description = description+description_1
        
    if plots_2:
        nam__2 = name+strng_2
        print(nam__2)
        srcfile_2 = kml.addfile(plots_dir+nam__2+'.png')
        description_2 = (
            '<img width="900" align="left" src="'+srcfile_2+'"/>'
            )
        description = description+description_2
    


    site = kml.newpoint(name=nam) 
    site.coords = [(lon,lat,hgt)]
    site.style.labelstyle.color = site_tcolor 
    site.style.labelstyle.scale = site_tscale
    site.style.iconstyle.icon.href = site_iref
    site.style.iconstyle.scale = site_iscale
    site.style.iconstyle.color = site_icolor
    
    if repeats:
        if repeat_string in full_name:
            site.style.iconstyle.icon.href = site_iref_rept
            site.style.labelstyle.color = site_rcolor_rept
            site.style.iconstyle.color  = site_icolor_rept
            site.style.balloonstyle.text = 'repeated site'
            site.style.balloonstyle.textcolor = site_rcolor
        #print(nam, mt_obj.lat, mt_obj.lon, hgt)
            site.description = description+'  - repeated site'
    
    if bads:
        if bad_string in full_name:
            site.style.iconstyle.icon.href = site_iref_bad
            site.style.labelstyle.color = site_rcolor_bad
            site.style.iconstyle.color  = site_icolor_bad
            site.style.balloonstyle.text = 'bad site'
            site.style.balloonstyle.textcolor = site_rcolor
        #print(nam, mt_obj.lat, mt_obj.lon, hgt)
            site.description = description+'  - repeated site'

    site.description = description

    

kml_outfile = kml_dir+kml_file 

# Save raw kml file:    

kml.save(kml_outfile+'.kml')

# Compressed kmz file:  

kml.savekmz(kml_outfile+'.kmz')

print('kml/z written to '+kml_file)
