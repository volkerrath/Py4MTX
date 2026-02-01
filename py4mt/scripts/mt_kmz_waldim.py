#!/usr/bin/env python3
'''
Plots WALDIM output as KMZ-file

@author: sb & vr may 2023


Martí A, Queralt P, Ledo, J (2009)
WALDIM: A code for the dimensionality analysis of magnetotelluric data
using the rotational invariants of the magnetotelluric tensor
Computers & Geosciences  , Vol. 35, 2295-2303

Martí A, Queralt P, Ledo J, Farquharson C (2010)
Dimensionality imprint of electrical anisotropy in magnetotelluric responses
Physics of the Earth and Planetary Interiors, 182, 139-151.

'''

# Import required modules

import os
import sys
import csv
import inspect


import numpy
import simplekml

PY4MTX_DATA = os.environ['PY4MTX_DATA']
PY4MTX_ROOT = os.environ['PY4MTX_ROOT']


mypath = [PY4MTX_ROOT+'/py4mt/modules/', PY4MTX_ROOT+'/py4mt/scripts/']
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)


import util as utl
from version import versionstrg

version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=inspect.getfile(inspect.currentframe()), out=False)
print(titstrng+'\n\n')


# Define the path to your EDI-files

#
WorkDir = '/home/vrath/MT_Data/waldim/'
EdiDir =  WorkDir + '/edi_eps/'
# EdiDir =  WorkDir + '/edi_noss/'
# EdiDir =  WorkDir + '/edi_don/'

DimDir = WorkDir
print(' WALdim results read from: %s' % DimDir)



UseFreqs = False
if UseFreqs:
    DimFile = EdiDir+'ANN_DIM_0.30.dat'
    KmlFile = 'ANN_FREQ'
else:
    DimFile = EdiDir+'ANN_BANDCLASS_0.30.dat'
    KmlFile = 'ANN_BAND_30'
#Class3 = True
Class3 = False

# # Define the path for saving  kml files
KmlDir =  EdiDir
kml = False
kmz = True


icon_dir = PY4MTX_ROOT + '/py4mt/share/icons/'
site_icon =  icon_dir + 'placemark_circle.png'

site_tcolor = simplekml.Color.white  # '#555500' #
site_tscale = 1.  # scale the text
site_iscale = 1.5

if Class3:
# for only 3 classes
    site_icolor_none = simplekml.Color.white
    site_icolor_1d = simplekml.Color.blue
    site_icolor_2d = simplekml.Color.green
    site_icolor_3d = simplekml.Color.red

else:
    from matplotlib import colormaps, colors
    # from matplotlib import cm, colors
    # cols = cm.get_cmap('rainbow', 9)
    cols = colormaps['rainbow'].resampled(9)
    # dimcolors = [(1., 1., 1., 1.)]
    dimcolors = ['ffffff']
    for c in range(cols.N):
        rgba = cols(c)
        # dimcolors.append(rgba)
        hexo = colors.rgb2hex(rgba)[1:]
        dimcolors.append(hexo)

    desc =[
    '0: UNDETERMINED',
    '1: 1D',
    '2: 2D',
    '3: 3D/2D only twist',
    '4: 3D/2D general',
    '5: 3D',
    '6: 3D/2D with diagonal regional tensor',
    '7: 3D/2D or 3D/1D indistinguishable',
    '8: Anisotropy hint 1: homogeneous anisotropic medium',
    '9: Anisotropy hint 2: anisotropic body within a 2D medium'
        ]


# Open kml object:

kml = simplekml.Kml(open=1)

site_iref = kml.addfile(site_icon)

if UseFreqs:
    read=[]
    with open(DimFile, 'r') as f:
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
            # print(repr(line))
            data.append(line)
    data =  numpy.asarray(data, dtype='object')
    ndt = numpy.shape(data)

    freqs = numpy.unique(data[:,3])
    print('freqs')
    print(freqs)


    for f in freqs:
        Nams = []
        Lats = []
        Lons = []
        Dims = []

        ff = numpy.log10(f)
        if ff < 0:
            freq_strng = 'Per'+str(int(round(1/f,0)))+'s'
        else:
            freq_strng = 'Freq'+str(int(round(f,0)))+'Hz'

        freqfolder = kml.newfolder(name=freq_strng)

        for line in numpy.arange(ndt[0]):
            fs = numpy.log10(data[line,3])
            if numpy.isclose(ff, fs, rtol=1e-2, atol=0.):
                Nams.append(data[line,0])
                Lons.append(data[line,1])
                Lats.append(data[line,2])
                Dims.append(data[line,5])

        nsites =len(Nams)
        # print ('lat\n',Lats)
        # print ('lon\n',Lons)
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
                    site.description ='undetermined'
                if Dims[ii]==1:
                    site.style.iconstyle.color = site_icolor_1d
                    site.description ='1-D'
                if Dims[ii]==2:
                    site.style.iconstyle.color = site_icolor_2d
                    site.description ='2-D'
                if Dims[ii]>2:
                    site.style.iconstyle.color = site_icolor_3d
                    site.description ='3-D'
            else:
                # print(Dims[ii], desc[Dims[ii]], dimcolors[Dims[ii]])
                site.style.iconstyle.color = simplekml.Color.hex(dimcolors[Dims[ii]])
                #str(dimcolors[Dims[ii]])
                # print(simplekml.Color.hex(dimcolors[Dims[ii]]))
                site.description = desc[Dims[ii]]

else:

    read=[]
    with open(DimFile, 'r') as f:
        place_list = csv.reader(f)

        for site in place_list:
            tmp= site[0].split()[:8]
            # print(tmp)
            read.append(tmp)
    read = read[1:]
# Site   Longitude        Latitude   BAND       Tmin        Tmax nper  DIM
    data=[]
    for line in read:
            line[1] = float(line[1]) # lon
            line[2] = float(line[2]) # lat
            line[3] = int(line[3])   # band
            line[4] = float(line[4]) # per min
            line[5] = float(line[5]) # per max
            line[6] = int(line[7])   # dim
            # print(repr(line))
            data.append(line)
    data =  numpy.asarray(data, dtype='object')
    ndt = numpy.shape(data)

    bands = numpy.unique(data[:,3])
    print('bands')
    print(bands)


    for bnd in bands:
        bnd_name = 'Band'+str(bnd)

        Nams = []
        Lats = []
        Lons = []
        Bnds = []
        Dims = []
        Tmin = []
        Tmax = []



        for line in numpy.arange(ndt[0]):
            if bnd==data[line,3]:
                Nams.append(data[line,0])
                Lons.append(data[line,1])
                Lats.append(data[line,2])
                Bnds.append(data[line,3])
                Tmin.append(data[line,4])
                Tmax.append(data[line,5])
                Dims.append(data[line,6])

        nsites =len(Nams)

        bnd_strg = ('Band: '+str(bnd) +' periods '
                    +str(Tmin[0])+'-' +str(Tmax[0])+' s')
        bndfolder = kml.newfolder(name=bnd_strg)
        # print ('lat\n',Lats)
        # print ('lon\n',Lons)
        for ii in numpy.arange(nsites):
            site = bndfolder.newpoint(name=Nams[ii])
            site.coords = [(Lons[ii], Lats[ii], 0.)]

            site.style.labelstyle.color = site_tcolor
            site.style.labelstyle.scale = site_tscale
            site.style.iconstyle.icon.href = site_icon
            site.style.iconstyle.scale = site_iscale

            if Class3:
                if Dims[ii]==0:
                    site.style.iconstyle.color = site_icolor_none
                    site.description ='undetermined'
                if Dims[ii]==1:
                    site.style.iconstyle.color = site_icolor_1d
                    site.description ='1-D'
                if Dims[ii]==2:
                    site.style.iconstyle.color = site_icolor_2d
                    site.description ='2-D'
                if Dims[ii]>2:
                    site.style.iconstyle.color = site_icolor_3d
                    site.description ='3-D'
            else:
                # print(Dims[ii], desc[Dims[ii]], dimcolors[Dims[ii]])
                site.style.iconstyle.color = simplekml.Color.hex(dimcolors[Dims[ii]])
                #str(dimcolors[Dims[ii]])
                # print(simplekml.Color.hex(dimcolors[Dims[ii]]))
                site.description = desc[Dims[ii]]


if Class3:
    kml_outfile = KmlDir + KmlFile+'_CLASS3'
else:
    loncenter=numpy.mean(Lons)
    latcenter=numpy.mean(Lats)
    site = kml.newpoint(name='Legend')
    leg_icon =  icon_dir + 'star.png'
    site.coords = [(loncenter, latcenter, 0.)]
    site.style.iconstyle.icon.href = leg_icon
    site.style.iconstyle.color =  simplekml.Color.yellow
    site.style.iconstyle.scale = site_iscale*1.5
    site.style.labelstyle.color = simplekml.Color.yellow
    site.style.labelstyle.scale =site_tscale*1.2
    srcfile = kml.addfile(PY4MTX_ROOT + '/py4mt/share/DimColorScheme.png')
    site.description = f"<img width='300' align='left' src='{srcfile}'/>"
    #site.description = ('<img width='800' align='left' src='' + srcfile + ''/>')
    kml_outfile = KmlDir + KmlFile+'_CLASS9'




# Compressed kmz file:
if kmz:
    kml.savekmz(kml_outfile + '.kmz')

print('Done. kml/z written to ' + kml_outfile)
