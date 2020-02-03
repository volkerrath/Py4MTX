# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.1
# ---

'''
@author: sb & vr oct 2019
'''

# Import required modules

import os
import simplekml 
from mtpy.core.mt import MT

# Deermine what is added to the KML-tags:

# plotsDA =  site plots produced by MT_siteplot.py, with an 
# additional string strngDA added to the EDI basename.
# plotsRJ =  MCMC plots produced by MT_mcmcplot.py, with an 
# additional string strnRJ added to the EDI basename.

plotsDA = True
strngDA ='_data'

plotsRJ = True
strngRJ ='_rjmcmc'

# Define the path to your EDI-files

edi_dir = 'NEW_edifiles_bbmt_roi_edit/'
print(' Edifiles read from: %s' % edi_dir)

if plotsDA or plotsRJ:
    plots_dir='NEW_plots_bbmt_roi_edit/'
    print(' Plots read from: %s' % plots_dir)


# Define the path for saving  kml files

kml_dir = './'
kml_file = 'NEW_Timor_bbmt_edit'

icon = 'icons/triangle.png'
tcolor = simplekml.Color.white ###'#555500' #
tscale = 0.8  # scale the text 

iscale = 1.
icolor = simplekml.Color.red 
rcolor = simplekml.Color.blue 
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
iref   = kml.addfile(icon)
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
    nam = full_name[3:] 
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
    site.style.labelstyle.color = tcolor 
    site.style.labelstyle.scale = tscale
    site.style.iconstyle.icon.href = iref
    site.style.iconstyle.scale = iscale
    site.style.iconstyle.color = icolor
    
    if full_name[-1:] =='R':
        site.style.labelstyle.color =rcolor
        site.style.balloonstyle.text = 'repeated site'
        site.style.balloonstyle.textcolor = rcolor
    #print(nam, mt_obj.lat, mt_obj.lon, hgt)
        site.description = description+'  - repeated site'

    site.description = description

kml_outfile = kml_dir+kml_file 

# Save raw kml file:    

kml.save(kml_outfile+'.kml')

# Compressed kmz file:  

kml.savekmz(kml_outfile+'.kmz')

print('kml/z written to '+kml_file)
