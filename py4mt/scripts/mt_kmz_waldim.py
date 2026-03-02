#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot WALDIM dimensionality analysis results as a KMZ file.

Reads WALDIM output (per-frequency or per-band classification) and
creates a Google Earth KMZ with colour-coded site markers indicating
the dimensionality class (1-D, 2-D, 3-D, anisotropic, etc.).

Supports a 3-class scheme (1-D/2-D/3-D) or the full 10-class WALDIM
classification with rainbow colour coding.

References:
    Marti, Queralt & Ledo (2009), Computers & Geosciences, 35, 2295-2303.
    Marti, Queralt, Ledo & Farquharson (2010), PEPI, 182, 139-151.

@author: sb & vr may 2023
"""

import os
import sys
import csv
import inspect

import numpy as np
import simplekml

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [PY4MTX_ROOT + "/py4mt/modules/", PY4MTX_ROOT + "/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)

import util as utl
from version import versionstrg

version, _ = versionstrg()
titstrng = utl.print_title(
    version=version, fname=inspect.getfile(inspect.currentframe()), out=False
)
print(titstrng + "\n\n")

# =============================================================================
#  Configuration
# =============================================================================
WorkDir = "/home/vrath/MT_Data/waldim/"
EdiDir = WorkDir + "/edi_eps/"

DimDir = WorkDir
print(" WALdim results read from: %s" % DimDir)

UseFreqs = False
if UseFreqs:
    DimFile = EdiDir + "ANN_DIM_0.30.dat"
    KmlFile = "ANN_FREQ"
else:
    DimFile = EdiDir + "ANN_BANDCLASS_0.30.dat"
    KmlFile = "ANN_BAND_30"

Class3 = False

KmlDir = EdiDir
SaveKml = False
SaveKmz = True

icon_dir = PY4MTX_ROOT + "/py4mt/share/icons/"
site_icon = icon_dir + "placemark_circle.png"

site_tcolor = simplekml.Color.white
site_tscale = 1.
site_iscale = 1.5

if Class3:
    site_icolor_none = simplekml.Color.white
    site_icolor_1d = simplekml.Color.blue
    site_icolor_2d = simplekml.Color.green
    site_icolor_3d = simplekml.Color.red
else:
    from matplotlib import colormaps, colors

    cols = colormaps["rainbow"].resampled(9)
    dimcolors = ["ffffff"]
    for c in range(cols.N):
        rgba = cols(c)
        hexo = colors.rgb2hex(rgba)[1:]
        dimcolors.append(hexo)

    desc = [
        "0: UNDETERMINED",
        "1: 1D",
        "2: 2D",
        "3: 3D/2D only twist",
        "4: 3D/2D general",
        "5: 3D",
        "6: 3D/2D with diagonal regional tensor",
        "7: 3D/2D or 3D/1D indistinguishable",
        "8: Anisotropy hint 1: homogeneous anisotropic medium",
        "9: Anisotropy hint 2: anisotropic body within a 2D medium",
    ]


# =============================================================================
#  Helper: assign colour to a placemark given dimensionality
# =============================================================================
def _style_site(site_pt, dim_val):
    """Apply dimensionality colour to a KML placemark."""
    site_pt.style.labelstyle.color = site_tcolor
    site_pt.style.labelstyle.scale = site_tscale
    site_pt.style.iconstyle.icon.href = site_icon
    site_pt.style.iconstyle.scale = site_iscale

    if Class3:
        if dim_val == 0:
            site_pt.style.iconstyle.color = site_icolor_none
            site_pt.description = "undetermined"
        elif dim_val == 1:
            site_pt.style.iconstyle.color = site_icolor_1d
            site_pt.description = "1-D"
        elif dim_val == 2:
            site_pt.style.iconstyle.color = site_icolor_2d
            site_pt.description = "2-D"
        else:
            site_pt.style.iconstyle.color = site_icolor_3d
            site_pt.description = "3-D"
    else:
        site_pt.style.iconstyle.color = simplekml.Color.hex(dimcolors[dim_val])
        site_pt.description = desc[dim_val]


# =============================================================================
#  Build KML
# =============================================================================
kml_obj = simplekml.Kml(open=1)
kml_obj.addfile(site_icon)

if UseFreqs:
    # --- Per-frequency mode ---
    read = []
    with open(DimFile, "r") as f:
        place_list = csv.reader(f)
        for row in place_list:
            tmp = row[0].split()[:6]
            read.append(tmp)
    read = read[1:]

    data = []
    for line in read:
        line[1] = float(line[1])
        line[2] = float(line[2])
        line[3] = float(line[3])
        line[4] = float(line[4])
        line[5] = int(line[5])
        data.append(line)
    data = np.asarray(data, dtype="object")
    ndt = np.shape(data)

    freqs = np.unique(data[:, 3])
    print("freqs:", freqs)

    for freq in freqs:
        ff = np.log10(freq)
        if ff < 0:
            freq_strng = "Per" + str(int(round(1 / freq, 0))) + "s"
        else:
            freq_strng = "Freq" + str(int(round(freq, 0))) + "Hz"

        freqfolder = kml_obj.newfolder(name=freq_strng)

        for idx in np.arange(ndt[0]):
            fs = np.log10(data[idx, 3])
            if np.isclose(ff, fs, rtol=1e-2, atol=0.):
                pt = freqfolder.newpoint(name=data[idx, 0])
                pt.coords = [(data[idx, 1], data[idx, 2], 0.)]
                _style_site(pt, data[idx, 5])

    Lons = data[:, 1]
    Lats = data[:, 2]

else:
    # --- Per-band mode ---
    # Columns: Site, Longitude, Latitude, BAND, Tmin, Tmax, nper, DIM
    read = []
    with open(DimFile, "r") as f:
        place_list = csv.reader(f)
        for row in place_list:
            tmp = row[0].split()[:8]
            read.append(tmp)
    read = read[1:]

    data = []
    for line in read:
        line[1] = float(line[1])   # lon
        line[2] = float(line[2])   # lat
        line[3] = int(line[3])     # band
        line[4] = float(line[4])   # per min
        line[5] = float(line[5])   # per max
        line[6] = int(line[7])     # dim (from column 7)
        data.append(line)
    data = np.asarray(data, dtype="object")
    ndt = np.shape(data)

    bands = np.unique(data[:, 3])
    print("bands:", bands)

    for bnd in bands:
        Nams, Lats, Lons, Dims, Tmin, Tmax = [], [], [], [], [], []

        for idx in np.arange(ndt[0]):
            if bnd == data[idx, 3]:
                Nams.append(data[idx, 0])
                Lons.append(data[idx, 1])
                Lats.append(data[idx, 2])
                Tmin.append(data[idx, 4])
                Tmax.append(data[idx, 5])
                Dims.append(data[idx, 6])

        bnd_strg = (
            "Band: " + str(bnd) + " periods "
            + str(Tmin[0]) + "-" + str(Tmax[0]) + " s"
        )
        bndfolder = kml_obj.newfolder(name=bnd_strg)

        for ii in range(len(Nams)):
            pt = bndfolder.newpoint(name=Nams[ii])
            pt.coords = [(Lons[ii], Lats[ii], 0.)]
            _style_site(pt, Dims[ii])

# =============================================================================
#  Legend and save
# =============================================================================
if Class3:
    kml_outfile = KmlDir + KmlFile + "_CLASS3"
else:
    loncenter = np.mean(Lons)
    latcenter = np.mean(Lats)
    legend = kml_obj.newpoint(name="Legend")
    leg_icon = icon_dir + "star.png"
    legend.coords = [(loncenter, latcenter, 0.)]
    legend.style.iconstyle.icon.href = leg_icon
    legend.style.iconstyle.color = simplekml.Color.yellow
    legend.style.iconstyle.scale = site_iscale * 1.5
    legend.style.labelstyle.color = simplekml.Color.yellow
    legend.style.labelstyle.scale = site_tscale * 1.2
    srcfile = kml_obj.addfile(PY4MTX_ROOT + "/py4mt/share/DimColorScheme.png")
    legend.description = f"<img width='300' align='left' src='{srcfile}'/>"
    kml_outfile = KmlDir + KmlFile + "_CLASS9"

if SaveKml:
    kml_obj.save(kml_outfile + ".kml")

if SaveKmz:
    kml_obj.savekmz(kml_outfile + ".kmz")

print("Done. kml/z written to " + kml_outfile)
