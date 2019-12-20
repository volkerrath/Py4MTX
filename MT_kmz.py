# -*- coding: utf-8 -*-
'''
@author: sb & vr oct 2019
'''

# import required modules
import os
import simplekml 
from mtpy.core.mt import MT


plotsDA = True
strngDA ='MT0_'
plotsRJ = False
strngRJ ='RJ10_'
 
# Define the path to your edi files
edi_dir = '/home/vrath/Timor/edifiles_bbmt_roi/'
print(' Edifiles read from: %s' % edi_dir)
edi_files=[]
files= os.listdir(edi_dir) 
for entry in files:
   # print(entry)
   if entry.endswith('.edi') and not entry.endswith('.'):
            edi_files.append(entry)



# Define the path to corresponding plot files
if plotsDA or plotsRJ:
    plots_dir='plots_bbmt_roi/'
    print(' Plots read from: %s' % plots_dir)

# Define the path for saving  kml files
kml_dir = './'
kml_file = 'Timor_bbmt_roi'

# now open kml object
kml = simplekml.Kml(open=1)

tcolor = simplekml.Color.white ###'#555500' #
tscale = 0.8  # scale the text 
iref   = kml.addfile('icons/triangle.png')
#iref_alt1 =  kml.addfile('star.png')
#iref_alt2 =  kml.addfile('donut.png')
iscale = 1.
icolor = simplekml.Color.red 
rcolor = simplekml.Color.blue 
# simplekml.Color.rgb(0, 0, 255)
# 'ffff0000'
# loop over edifiles

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

    if plotsDA:
        nam_DA = strngDA+nam
        print(nam_DA)
        srcfileDA = kml.addfile(plots_dir+nam_DA+'.png')
        descriptionDA = (
            '<img width="800" align="left" src="'+srcfileDA+'"/>'
            )
        description = description+descriptionDA
        
    if plotsRJ:
        nam_RJ = strngRJ+nam
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
# save kml file    
kml.save(kml_outfile+'.kml')
# save compressed kmz file   
kml.savekmz(kml_outfile+'.kmz')
print('kml/z written to '+kml_file)