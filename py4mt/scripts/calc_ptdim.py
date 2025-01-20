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

This script produces a site list containing site names,
coordinates and elevations, e. g., for WALDIM analysis.

@author: sb & vr dec 2019
"""

# Import required modules

import os
import sys
from sys import exit as error
import csv



from mtpy  import MT, MTCollection, MTData
# from mtpy.core.transfer_function import z_analysis
# mtpy.core.transfer_function.z.Z
# mtpy.core.transfer_function.z_analysis
from mtpy.core.transfer_function.z import Z
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

cm = 1/2.54  # centimeters in inches



PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [PY4MTX_ROOT+"/py4mt/modules/", PY4MTX_ROOT+"/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)


import util as utl
import mtproc as mtp
from version import versionstrg


version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")

PY4MTX_DATA =  "/home/vrath/MT_Data/"
WorkDir = PY4MTX_DATA+"/Enfield/"
# Define the path to your EDI-files:
EdiDir = WorkDir
print(" Edifiles read from: %s" % EdiDir)

DimFile = EdiDir+"Dimensions.dat"

# No changes required after this line!

# Construct list of edi-files:
edi_files = mtp.get_edi_list(EdiDir)
ns = len(edi_files)
print(ns, " edi-files in list.")

# Loop over edifiles:
n3d = 0
n2d = 0
n1d = 0
nel = 0
sit = -1

# fig  = plt.figure()
# fig.set_figwidth(16*cm)

dimlist = []
for filename in edi_files:
    sit = sit + 1
    print("reading data from: " + filename)
    name, ext = os.path.splitext(filename)
    file_i = filename

# Create MT object
    mt_obj = MT()
    mt_obj.read(file_i)

    site = mt_obj.station_metadata.id
    lat = mt_obj.station_metadata.location.latitude
    lon = mt_obj.station_metadata.location.longitude
    elev = mt_obj.station_metadata.location.elevation

    Z = mt_obj.Z
    per = Z.period

    print(" site %s at :  % 10.6f % 10.6f % 8.1f" % (name, lat, lon, elev ))

# use the phase tensor to determine which frequencies are 1D/2D/3D
    dim = Z.estimate_dimensionality(
                         skew_threshold=3,
                         # threshold in skew angle (degrees) to determine if
                         # data are 3d
                         # threshold in phase ellipse eccentricity to determine
                         # if data are 2d (vs 1d)
                         eccentricity_threshold=0.1
                         )

    tmp = [(site, lat, lon, elev, per[ind], dim[ind]) for ind in np.arange(len(dim))]
    np.savetxt(fname = name+"_dims.dat", X=tmp, delimiter='\t', fmt="%s")
    print("Dimension writen to", name+"_dims.dat")
    if sit == 0:
        dims = tmp
    else:
        dims = np.vstack((dims, tmp))
    # print(np.shape(dims))

    print("dimensionality:")
    nel_site = np.size(dim)
    n1d_site = sum(map(lambda x: x == 1, dim))
    n2d_site = sum(map(lambda x: x == 2, dim))
    n3d_site = sum(map(lambda x: x == 3, dim))
    print("  number of undetermined elements = " +
          str(nel_site - n1d_site - n2d_site- n3d_site) + "\n")
    print("  number of 1-D elements = " + str(n1d_site) +
          "  (" + str(round(100 * n1d_site / nel_site)) + "%)")
    print("  number of 2-D elements = " + str(n2d_site) +
          "  (" + str(round(100 * n2d_site / nel_site)) + "%)")
    print("  number of 3-D elements = " + str(n3d_site) +
          "  (" + str(round(100 * n3d_site / nel_site)) + "%)")


    _, sitn = os.path.split(name)
    dimlist.append([sitn, nel_site, n1d_site, n2d_site, n3d_site])

    nel = nel + np.size(dim)
    n1d = n1d + n1d_site
    n2d = n2d + n2d_site
    n3d = n3d + n3d_site





print("\n\n\n")
print("number of sites = " + str(sit))
print("total number of elements = " + str(nel))
print("  number of undetermined elements = " +
      str(nel - n1d - n2d - n3d) + "\n")
print("  number of 1-D elements = " + str(n1d) +
      "  (" + str(round(100 * n1d / nel)) + "%)")
print("  number of 2-D elements = " + str(n2d) +
      "  (" + str(round(100 * n2d / nel)) + "%)")
print("  number of 3-D elements = " + str(n3d) +
      "  (" + str(round(100 * n3d / nel)) + "%)")

np.savetxt(fname = EdiDir+"All_dims.dat", X=dims, delimiter='\t', fmt="%s")

dimlist.append(["all_sites", nel,
                round(100*n1d/nel),
                round(100*n2d/nel),
                round(100*n3d/nel)])

with open(DimFile, "w") as f:
    sites = csv.writer(f, delimiter = " ")
    sites.writerow(["Sitename", "Ntot", "N1d%", "N2d%", "N3d%"])
    sites.writerow([ns, " ", " "])

    for item in dimlist:
        sites.writerow(item)

    print('Done')
