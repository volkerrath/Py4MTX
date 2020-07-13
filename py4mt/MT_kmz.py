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
@author: sb & vr oct 2019
'''

# Import required modules

import os
import csv
import simplekml 
from mtpy.core.mt import MT

# Determine what is added to the KML-tags:
# plotsDA =  site plots produced by MT_siteplot.py, with an 
# additional string strngDA added to the EDI basename.
# plotsRJ =  MCMC plots produced by MT_mcmcplot.py, with an 
# additional string strnRJ added to the EDI basename.

plotsDA = True
strngDA = '' #'_data'

plotsRJ = False
strngRJ ='_rjmcmc'

# Define the path to your EDI-files

# edi_dir = r'/home/vrath/WestTimor/WT8C_edi/'
edi_dir = r'/media/vrath/MT/Ireland/Donegal/Donegal_EDIs_3DGridEdited/'
print(' Edifiles read from: %s' % edi_dir)

if plotsDA or plotsRJ:
    plots_dir= edi_dir+'data_plots/'
    # r'/media/vrath/MT/Ireland/Northwest_CarboniferousBasin/MT_DATA/EDI/data_plots/'
    # r'/home/vrath/WestTimor/WT8C_plots/' #edi_dir #'NEW_plots_bbmt_roi_edit/'
    print(' Plots read from: %s' % plots_dir)


# Determine which geographical info is added to the KML-tags:
# define empty list
places = []

# open file and read the content in a list
places_file = r'/media/vrath/MT/Ireland/Northwest_CarboniferousBasin/MT_DATA/EDI/Sitelist.csv'
# r'/home/vrath/WestTimor/places.csv'
with open(places_file, 'r') as f:
    placelist = csv.reader(f, delimiter=' ')
    for row in placelist:
        print(row)

# Define the path for saving  kml files

kml_dir = ''
kml_file = edi_dir+'Donegal'

site_icon = 'icons/triangle.png'
site_tcolor = simplekml.Color.white ###'#555500' #
site_tscale = 0.8  # scale the text 

site_iscale = 1.
site_icolor = simplekml.Color.red 
site_rcolor = simplekml.Color.blue 
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

site_iref   = kml.addfile(site_icon)
#iref_alt1 =  kml.addfile('star.png')
#iref_alt2 =  kml.addfile('donut.png')

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

    if plotsDA:
        nam_DA = name+strngDA
        print(nam_DA)
        srcfileDA = kml.addfile(plots_dir+nam_DA+'.png')
        descriptionDA = (
            '<img width="800" align="left" src="'+srcfileDA+'"/>'
            )
        description = description+descriptionDA
        
    if plotsRJ:
        nam_RJ = name+strngRJ
        print(nam_RJ)
        srcfileRJ = kml.addfile(plots_dir+nam_RJ+'.png')
        descriptionRJ = (
            '<img width="900" align="left" src="'+srcfileRJ+'"/>'
            )
        description = description+descriptionRJ
    


    site = kml.newpoint(name=nam) 
    site.coords = [(lon,lat,hgt)]
    site.style.labelstyle.color = site_tcolor 
    site.style.labelstyle.scale = site_tscale
    site.style.iconstyle.icon.href = site_iref
    site.style.iconstyle.scale = site_iscale
    site.style.iconstyle.color = site_icolor
    
    if full_name[-1:] =='R':
        site.style.labelstyle.color =site_rcolor
        site.style.balloonstyle.text = 'repeated site'
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
