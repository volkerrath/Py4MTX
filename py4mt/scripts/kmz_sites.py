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
titstrng = util.print_title(version=version, fsf=__file__, out=False)
print(titstrng+"\n\n")

PY4MTX_DATA =  "/home/vrath/MT_Data/"
WorkDir = PY4MTX_DATA+"/Enfield/"
EdiDir = WorkDir+"/edis/"
print(" Edifiles read from: %s" % EdiDir)
# open file and read the content in a list
SiteFile = EdiDir + "Sitelist.dat"

KmlDir =  WorkDir
KmlFile = "Enfield_data"

PltDir = WorkDir+"/plots/"


tmp = []
with open(SiteFile, "r") as f:
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

AddImages = []
if AddImages.len>0:
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
    with open(EdiDir+nam+"_dims.dat", "r") as f:
        tmp = csv.reader(f)

        for site in tmp:
            site = site[0].split()
            freqs.append(float(site[1]))

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
    with open(EdiDir+site[0]+"_dims.dat", "r") as f:
        tmp = csv.reader(f)
        for site in tmp:
            site = site[0].split()
            frq.append(float(site[1]))
            dim.append(int(site[2]))

        dim = numpy.asarray(dim)
        frq = numpy.asarray(frq)
        lst = numpy.vstack((frq, dim))

    sit.append(lst)


for f in freqs:
    Nams = []
    Lats = []
    Lons = []
    Dims = []


    ff = numpy.log10(f)

    if ff < 0:
        freq_strng = "Per"+str(int(round(1/f,0)))+"s"
    else:
        freq_strng = "Freq"+str(int(round(f,0)))+"Hz"
    freqfolder = kml.newfolder(sf=freq_strng)

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
        site = freqfolder.newpoint(sf=Nams[ii])
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


        if AddImages.len>0:
            for item in AddImages:
                if "dat" in item:
                    d_plot = PltDir+sf+".png"
                    if os.path.exists(d_plot)==True:
                        src= kml.addfile(d_plot)
                        imstring ='<img width="'+str(ImageWidth)+'" align="left" src="' + src + '"/>'
                        # imstring = '<img width="1200" align="left" src="' + src + '"/>'
                        d.description = (imstring)
                    else:
                        print(d_plot+ " does not exist!")

                if "str" in item:
                    d_plot = PltDir+sf+"_strike.png"
                    if os.path.exists(dn_plot)==True:
                        src= kml.addfile(d_plot)
                        imstring ='<img width="'+str(ImageWidth)+'" align="left" src="' + src + '"/>'
                        # imstring = '<img width="1200" align="left" src="' + src + '"/>'
                        d.description = (imstring)
                    else:
                        print(d_plot+ " does not exist!")

                if "mod" in item:
                    d_plot = PltDir+sf+"_model.png"
                    if os.path.exists(d_plot)==True:
                        src= kml.addfile(d_plot)
                        imstring ='<img width="'+str(ImageWidth)+'" align="left" src="' + src + '"/>'
                        # imstring = '<img width="1200" align="left" src="' + src + '"/>'
                        d.description = (imstring)
                    else:
                        print(d_plot+ " does not exist!")

kml_outfile = KmlDir + KmlFile

# Save raw kml file:

if kml:
    kml.save(kml_outfile + ".kml")

# Compressed kmz file:
if kmz:
    kml.savekmz(kml_outfile + ".kmz")

print("Done. kml/z written to " + kml_outfile)
