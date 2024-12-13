#!/usr/bin/env python3
"""
@author: sb & vr oct 2019
"""

# Import required modules

import os
import sys
import csv
import numpy
import simplekml

from mtpy.core.mt import MT

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [PY4MTX_ROOT+"/py4mt/modules/", PY4MTX_ROOT+"/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)


import util
from version import versionstrg

version, _ = versionstrg()
titstrng = util.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")

PY4MTX_DATA =  "/home/vrath/MT_Data/"
WorkDir = PY4MTX_DATA+"/Ubaye_best/"
EdiDir = WorkDir+"/edis/"
print(" Edifiles read from: %s" % EdiDir)
# open file and read the content in a list
SiteFile = EdiDir + "Sitelist.dat"

KmlDir =  WorkDir
KmlFile = "Ubaye_all"

PltDir = WorkDir+"/plots/"


# print(places)
# Define the path for saving  kml files
kml = False
kmz = True

AddImages = "model" # "strikes" # "data", "strikes", "model"
if len(AddImages)>0:
    PltDir = WorkDir+"/plots/"
    ImageWidth= 800



AddSpecial = False
if AddSpecial:
    SpcDir = WorkDir
    SpecialDat = SpcDir+"Special.dat"
    specials = []
    with open(SpecialDat) as file:
        for line in file:
            tmp = line.split(",")
            tmp[0] = float(tmp[0])
            tmp[1] = float(tmp[1])
            tmp[2] = tmp[2].strip()
            tmp[3] = tmp[3].strip()
            tmp[4] = float(tmp[4])
            tmp[5] = tmp[5].strip()
            specials.append(tmp)

    print("specials:",specials)






icon_dir = PY4MTX_ROOT + "/py4mt/share/icons/"
site_icon =  icon_dir + "placemark_circle.png"

site_tcolor = simplekml.Color.white  # "#555500" #
site_tscale = 1.2  # scale the text
site_iscale = 1.5


site_icolor = simplekml.Color.red
site_rcolor = simplekml.Color.red


# No changes required after this line!
# Construct list of EDI-files:


# Open kml object:

kml = simplekml.Kml(open=1)

site_iref = kml.addfile(site_icon)

# Determine unique sites.

Nams = []
Lats = []
Lons = []

with open(SiteFile) as file:
    for line in file:
        sit =line.split(" ")
        Nams.append(sit[0])
        Lats.append(float(sit[1]))
        Lons.append(float(sit[2]))


nsites =len(Nams)
# print (nsites)
for ii in numpy.arange(nsites):

    site =kml.newpoint(name=Nams[ii])
    site.coords = [(Lons[ii], Lats[ii], 0.)]

    site.style.labelstyle.color = site_tcolor
    site.style.labelstyle.scale = site_tscale
    site.style.iconstyle.icon.href = site_icon
    site.style.iconstyle.scale = site_iscale
    site.style.iconstyle.color = site_icolor
    site.description = Nams[ii]



    if len(AddImages)>0:

        if "dat" in AddImages:
            sf = Nams[ii]+"_data"
            d_plot = PltDir+sf+".png"
            if os.path.exists(d_plot)==True:
                src= kml.addfile(d_plot)
                imstring ='<img width="'+str(ImageWidth)+'" align="left" src="' + src + '"/>'
                # imstring = '<img width="1200" align="left" src="' + src + '"/>'
                site.description = (imstring)
            else:
                print(d_plot+ " does not exist!")

        if "str" in AddImages:
            sf = Nams[ii]+"_strikes"
            d_plot = PltDir+sf+".png"
            if os.path.exists(d_plot)==True:
                src= kml.addfile(d_plot)
                imstring ='<img width="'+str(ImageWidth)+'" align="left" src="' + src + '"/>'
                # imstring = '<img width="1200" align="left" src="' + src + '"/>'
                site.description = (imstring)
            else:
                print(d_plot+ " does not exist!")

        if "mod" in AddImages:
            sf = Nams[ii]+"_model"
            d_plot = PltDir+sf+".png"
            if os.path.exists(d_plot)==True:
                src= kml.addfile(d_plot)
                imstring ='<img width="'+str(ImageWidth)+'" align="left" src="' + src + '"/>'
                # imstring = '<img width="1200" align="left" src="' + src + '"/>'
                site.description = (imstring)
            else:
                print(d_plot+ " does not exist!")



kml_outfile = KmlDir + KmlFile + AddImages
kml.savekmz(kml_outfile + ".kmz")

print("Done. kml/z written to " + kml_outfile)
