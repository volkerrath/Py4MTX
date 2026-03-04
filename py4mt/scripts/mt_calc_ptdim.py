#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate phase-tensor dimensionality for each MT station.

Reads EDI files and uses the impedance tensor's phase-tensor
analysis to classify each frequency as 1-D, 2-D, or 3-D.
Writes per-site and combined dimensionality tables.

@author: sb & vr dec 2019
Cleanup: 4 Mar 2026 by Claude (Anthropic)
"""

# Import required modules

import os
import sys
import csv
import inspect

import numpy as np
from mtpy import MT

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [PY4MTX_ROOT + "/py4mt/modules/", PY4MTX_ROOT + "/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)

import util as utl
import data_proc as mtp
from version import versionstrg

version, _ = versionstrg()
titstrng = utl.print_title(
    version=version, fname=inspect.getfile(inspect.currentframe()), out=False
)
print(titstrng + "\n\n")

# =============================================================================
#  Configuration
# =============================================================================
WORK_DIR = PY4MTX_DATA + "/France/annecy_2025_dist/edi_files/"
EDI_DIR = WORK_DIR
print(" Edifiles read from: %s" % EDI_DIR)

DIM_FILE = EDI_DIR + "Dimensions.dat"

SKEW_THRESHOLD = 3.0
ECCENTRICITY_THRESHOLD = 0.1

# No changes required after this line!

# =============================================================================
#  Construct list of EDI files
# =============================================================================
edi_files = mtp.get_edi_list(EDI_DIR)
ns = len(edi_files)
print(ns, " edi-files in list.")

# =============================================================================
#  Loop over EDI files
# =============================================================================
n3d = 0
n2d = 0
n1d = 0
nel = 0

dims = None
dimlist = []

for sit, filename in enumerate(edi_files):
    print("reading data from: " + filename)
    name, ext = os.path.splitext(filename)

    # Create MT object
    mt_obj = MT()
    mt_obj.read(filename)

    site = mt_obj.station_metadata.id
    lat = mt_obj.station_metadata.location.latitude
    lon = mt_obj.station_metadata.location.longitude
    elev = mt_obj.station_metadata.location.elevation

    z_obj = mt_obj.Z
    per = z_obj.period

    print(" site %s at :  % 10.6f % 10.6f % 8.1f" % (name, lat, lon, elev))

    # Use the phase tensor to determine which frequencies are 1D/2D/3D
    dim = z_obj.estimate_dimensionality(
        skew_threshold=SKEW_THRESHOLD,
        eccentricity_threshold=ECCENTRICITY_THRESHOLD,
    )

    tmp = [
        (site, lat, lon, elev, per[ind], dim[ind])
        for ind in np.arange(len(dim))
    ]
    np.savetxt(fname=name + "_dims.dat", X=tmp, delimiter="\t", fmt="%s")
    print("Dimension written to", name + "_dims.dat")

    if dims is None:
        dims = np.array(tmp, dtype=object)
    else:
        dims = np.vstack((dims, tmp))

    print("dimensionality:")
    nel_site = np.size(dim)
    n1d_site = np.sum(dim == 1)
    n2d_site = np.sum(dim == 2)
    n3d_site = np.sum(dim == 3)
    n_undet = nel_site - n1d_site - n2d_site - n3d_site
    print("  number of undetermined elements = %d" % n_undet)
    print("  number of 1-D elements = %d  (%d%%)" % (n1d_site, round(100 * n1d_site / nel_site)))
    print("  number of 2-D elements = %d  (%d%%)" % (n2d_site, round(100 * n2d_site / nel_site)))
    print("  number of 3-D elements = %d  (%d%%)" % (n3d_site, round(100 * n3d_site / nel_site)))

    _, sitn = os.path.split(name)
    dimlist.append([sitn, nel_site, n1d_site, n2d_site, n3d_site])

    nel += np.size(dim)
    n1d += n1d_site
    n2d += n2d_site
    n3d += n3d_site

# =============================================================================
#  Summary
# =============================================================================
print("\n\n\n")
print("number of sites = %d" % (sit + 1))
print("total number of elements = %d" % nel)
print("  number of undetermined elements = %d\n" % (nel - n1d - n2d - n3d))
print("  number of 1-D elements = %d  (%d%%)" % (n1d, round(100 * n1d / nel)))
print("  number of 2-D elements = %d  (%d%%)" % (n2d, round(100 * n2d / nel)))
print("  number of 3-D elements = %d  (%d%%)" % (n3d, round(100 * n3d / nel)))

np.savetxt(fname=EDI_DIR + "All_dims.dat", X=dims, delimiter="\t", fmt="%s")

dimlist.append([
    "all_sites", nel,
    round(100 * n1d / nel),
    round(100 * n2d / nel),
    round(100 * n3d / nel),
])

with open(DIM_FILE, "w") as f:
    sites = csv.writer(f, delimiter=" ")
    sites.writerow(["Sitename", "Ntot", "N1d%", "N2d%", "N3d%"])
    sites.writerow([ns, " ", " "])
    for item in dimlist:
        sites.writerow(item)

print("Done")
