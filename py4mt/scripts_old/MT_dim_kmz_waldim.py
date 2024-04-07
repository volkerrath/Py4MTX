#!/usr/bin/env python3
"""
Plots WALDIM output as KMZ-file

@author: sb & vr may 2023


Martí A, Queralt P, Ledo, J (2009)
WALDIM: A code for the dimensionality analysis of magnetotelluric data 
using the rotational invariants of the magnetotelluric tensor
Computers & Geosciences  , Vol. 35, 2295-2303

Martí A, Queralt P, Ledo J, Farquharson C (2010)
Dimensionality imprint of electrical anisotropy in magnetotelluric responses
Physics of the Earth and Planetary Interiors, 182, 139-151.

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





# Define the path to your files

# dim_dir = r"/home/vrath/MT_Data/Naser/Limerick2023/mt/reprocessed_quality/reprocessed_bad/"
# dim_dir = r"/home/vrath/MT_Data/Naser/Limerick2023/mt/reprocessed_quality/reprocessed_good/"
# dim_dir = r"/home/vrath/MT_Data/Naser/Limerick2023/mt/reprocessed_quality/reprocessed_ugly/"
dim_dir = r"/home/vrath/MT_Data/Peru/Tacna/edi/"
print(" Edifiles read from: %s" % dim_dir)

dim_file = "Tac2023_WALDIM.dat"
# open file and read the content in a list
places_file = dim_dir + dim_file
# # Define the path for saving  kml files
kml = False
kmz = True
kml_dir = dim_dir
kml_file = r"Tacna2023_WALDIM"
Class3 = True 
Class3 = False

read=[]
with open(places_file, "r") as f:
    place_list = csv.reader(f)
    
    for site in place_list:
        tmp= site[0].split()[:6]            
        read.append(tmp)
read = read[1:]

data=[]
for line in read:
        line[1] = float(line[1])
        line[2] = float(line[2])
        line[3] = float(line[3])
        line[4] = float(line[4])
        line[5] = int(line[5])
        print(line)
        data.append(line)
data =  numpy.asarray(data, dtype="object")
ndt = numpy.shape(data)

freqs = numpy.unique(data[:,3])
print("freqs")
print(freqs)

icon_dir = dim_dir+"/icons/"# "/home/vrath/GoogleEarth/icons/"
site_icon =  icon_dir + "placemark_circle.png"

site_tcolor = simplekml.Color.white  # "#555500" #
site_tscale = 1.2  # scale the text
site_iscale = 1.5

if Class3:
# for only 3 classes
    site_icolor_none = simplekml.Color.white
    site_icolor_1d = simplekml.Color.blue
    site_icolor_2d = simplekml.Color.green
    site_icolor_3d = simplekml.Color.red

else:
    from matplotlib import cm, colors
    cols = cm.get_cmap('rainbow', 9)
    # dimcolors = [(1., 1., 1., 1.)]
    dimcolors = ["ffffff"]
    for c in range(cols.N):
        rgba = cols(c)
        # dimcolors.append(rgba) 
        hexo = colors.rgb2hex(rgba)[1:]
        dimcolors.append(hexo)
    
    desc =[
    "0: UNDETERMINED",
    "1: 1D",
    "2: 2D",
    "3: 3D/2D only twist",
    "4: 3D/2D general",
    "5: 3D",
    "6: 3D/2D with diagonal regional tensor",
    "7: 3D/2D or 3D/1D indistinguishable",
    "8: Anisotropy hint 1: homogeneous anisotropic medium",
    "9: Anisotropy hint 2: anisotropic body within a 2D medium"
        ]
    

# Open kml object:

kml = simplekml.Kml(open=1)

site_iref = kml.addfile(site_icon)
      

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
        
    freqfolder = kml.newfolder(name=freq_strng)

    for line in numpy.arange(ndt[0]):
        fs = numpy.log10(data[line,3])
        if numpy.isclose(ff, fs, rtol=1e-2, atol=0.):
            Nams.append(data[line,0])
            Lats.append(data[line,2])
            Lons.append(data[line,1])
            Dims.append(data[line,5]) 
  
    nsites =len(Nams)
    # print (nsites)
    for ii in numpy.arange(nsites):
        site = freqfolder.newpoint(name=Nams[ii])
        site.coords = [(Lons[ii], Lats[ii], 0.)]
        
        site.style.labelstyle.color = site_tcolor
        site.style.labelstyle.scale = site_tscale
        site.style.iconstyle.icon.href = site_icon
        site.style.iconstyle.scale = site_iscale
        
        if Class3:
            if Dims[ii]==0:
                site.style.iconstyle.color = site_icolor_none
                site.description ="undetermined"
            if Dims[ii]==1:
                site.style.iconstyle.color = site_icolor_1d
                site.description ="1-D"
            if Dims[ii]==2:
                site.style.iconstyle.color = site_icolor_2d
                site.description ="2-D"
            if Dims[ii]>2:
                site.style.iconstyle.color = site_icolor_3d
                site.description ="3-D"             
        else:
            # print(Dims[ii], desc[Dims[ii]], dimcolors[Dims[ii]])
            site.style.iconstyle.color = simplekml.Color.hex(dimcolors[Dims[ii]]) 
            #str(dimcolors[Dims[ii]])
            # print(simplekml.Color.hex(dimcolors[Dims[ii]]))
            site.description = desc[Dims[ii]]
 
if Class3:
    kml_outfile = kml_dir + kml_file+"_3"
else:
    loncenter=numpy.mean(Lons)
    latcenter=numpy.mean(Lats)
    site = kml.newpoint(name="Legend")
    leg_icon =  icon_dir + "star.png"
    site.coords = [(loncenter, latcenter, 0.)]
    site.style.iconstyle.icon.href = leg_icon
    site.style.iconstyle.color =  simplekml.Color.yellow
    site.style.iconstyle.scale = 2.
    site.style.labelstyle.color = simplekml.Color.yellow
    site.style.labelstyle.scale = 1.8
    srcfile = kml.addfile(dim_dir+"DimColorScheme.png")
    site.description = ('<img width="800" align="left" src="' + srcfile + '"/>')
    kml_outfile = kml_dir + kml_file+"_9"
# Save raw kml file:


# Compressed kmz file:
if kmz:
    kml.savekmz(kml_outfile + ".kmz")

print("Done. kml/z written to " + kml_outfile)
