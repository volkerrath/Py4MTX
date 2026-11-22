#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Produce a site list (name, coordinates, elevation) from EDI files.

Output formats (WHAT_FOR) are tailored for WALDIM, FEMTIC, KML/general use,
or a tab-separated UTM list in km with depth ("femticpy").

@author: sb & vr dec 2019
Cleanup: 4 Mar 2026 by Claude (Anthropic)
Modified: 2026-05-23 — use utm_zone_from_latlon / latlon_to_utm_zn after
    reading each EDI; FEMTIC output now includes easting and northing after
    sitenum; Claude Sonnet 4.6 (Anthropic)
Modified: 2026-11-20 -- sites may lie in different UTM zones: by default ONE
    zone is chosen for the whole survey (UTM_ZONE_MODE = "common", zone of
    the mean position, util.utm_common_zone) and every site is projected
    into it, so that eastings/northings share one grid.  The old behaviour
    (own zone per site, which mixes grids when the survey crosses a zone
    boundary) is UTM_ZONE_MODE = "per_site".  New: report of the zones the
    sites belong to and of the grid scale range, warning when the scale
    varies by more than UTM_MAX_SCALE_SPREAD, and a "#" comment line with the
    zone at the top of the FEMTIC/CSV list (WRITE_ZONE_HEADER; all py4mt
    readers skip "#" lines).  All EDIs are read first, the file is written
    afterwards.  Claude Sonnet 5.5 (Anthropic)
Modified: 2026-11-22 -- WHAT_FOR option "femticpy" (called "utm_py" in
    mt_make_sitelist_more.py) merged in: tab-separated list written to
    WORK_DIR + "Sitelist_UTM.csv" with name, easting [km], northing [km]
    (2 decimals) and depth = -elev; no "#" lines.  The format is now resolved
    once from WHAT_FOR (FORMAT: wal / femtic_site / kml / femticpy / other)
    and used for the zone step, the file name and the rows; the file name is
    printed after it is known.  A missing ELEV stops "femticpy" before
    anything is written.  The FEMTIC site list option is now called
    "femtic_site" (was "fem"/"femtic", default is "femtic_site"); options are
    matched by their exact names, so "fem" and "femtic" now count as unknown
    (same seven-column file, with a console note).  Common-zone logic of
    2026-11-20 unchanged.  Claude Sonnet 5.5 (Anthropic)

Provenance (changes of 2026-11-20 and 2026-11-22)
-------------------------------------------------
Author    : Claude (Claude Sonnet 5.5, Anthropic)
Generated : 2026-11-20 (common UTM zone), 2026-11-22 (WHAT_FOR "femticpy")
Notice    : AI-generated code; review and test it before production use.
"""

import os
import sys

from pathlib import Path
import csv
import inspect

import numpy as np

PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]
PY4MTX_DATA = os.environ["PY4MTX_DATA"]

for _base in [PY4MTX_ROOT + "/py4mt/modules/"]:
    for _p in [Path(_base), *Path(_base).rglob("*")]:
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

import util as utl
from version import versionstrg
from data_proc import load_edi

version, _ = versionstrg()
titstrng = utl.print_title(
    version=version, fname=inspect.getfile(inspect.currentframe()), out=False
)
print(titstrng + "\n\n")

# =============================================================================
#  Configuration
# =============================================================================
#: Column delimiter of the "femtic_site", "kml" and other lists ("wal" uses a
#: blank, "femticpy" a tab).
DELIM = ","

#: Output format of the site list (matched as a lower-case substring):
#:   "wal"          WALDIM: blank-separated; first line "Sitename Latitude
#:                  Longitude", second line the number of sites, then one
#:                  line per site: name, lat, lon.  No UTM coordinates.
#:   "femtic_site"  DELIM-separated, written to EDI_DIR + "site.dat": name,
#:                  lat, lon, elev, sitenum, easting [m], northing [m].
#:                  First lines are "#" comments with the UTM zone
#:                  (WRITE_ZONE_HEADER).  Read by the FEMTIC plot scripts.
#:   "femticpy"     tab-separated, NO header or "#" lines, written to
#:                  WORK_DIR + "Sitelist_UTM.csv": name, easting [km],
#:                  northing [km] (both rounded to 2 decimals = 10 m),
#:                  depth = -elev [m].
#:   "kml"          the same seven columns as "femtic_site".
#:   other          anything else (this includes the former names "fem",
#:                  "femtic" and "utm_py"): the same seven columns as
#:                  "femtic_site", with a note on the console.
WHAT_FOR = "femtic_site"


#: How the UTM zone is chosen.
#:   "common"   one zone for ALL sites (zone of the mean position, or
#:              UTM_ZONE_OVERRIDE).  Sites on the other side of a zone
#:              boundary are projected into that zone as well (extended
#:              UTM), so all eastings/northings are in one grid.  Use this
#:              for any inversion/mesh.
#:   "per_site" each site in its own zone (old behaviour).  Coordinates of
#:              sites in different zones can NOT be used together.
UTM_ZONE_MODE = "common"

#: UTM zone override - None = derive from the sites.  Set to an integer
#: (e.g. 19) to force this zone for all sites (the hemisphere follows the
#: mean latitude in "common" mode, each site's latitude in "per_site" mode).
UTM_ZONE_OVERRIDE = None

#: Warn if the UTM grid scale factor varies across the sites by more than
#: this (relative; 1e-3 = 1 m per km).  Far outside the zone's own 6 degrees
#: the grid distorts distances; for such surveys a local projection is better.
UTM_MAX_SCALE_SPREAD = 1.0e-3

#: Write a "# utm_zone=..." comment line as first line of the FEMTIC / CSV
#: list.  All py4mt readers skip "#" lines; set False for other programs.
WRITE_ZONE_HEADER = True

#WORK_DIR = "/home/vrath/Py4MTX/py4mt/data/rto/ubinas//"
# WORK_DIR = "/home/vrath/Py4MTX/py4mt/data/rto/misti//"
WORK_DIR = "/media/vrath/LargeBack/Ensembles/sabancaya2026/l-curve/"
# WORK_DIR = "/home/vrath/ACTIVE/mt_processor/2022_25/"
#WORK_DIR = "/home/vrath/ACTIVE/mt_processor/edi_files/"
EDI_DIR = WORK_DIR + "/edi_files/"

print(" Edifiles read from: %s" % EDI_DIR)

CSV_FILE = EDI_DIR + "site.dat"

# =============================================================================
#  Output format and file name (derived from WHAT_FOR)
# =============================================================================
_wf = WHAT_FOR.lower()
if "wal" in _wf:
    FORMAT = "wal"
elif "femticpy" in _wf:
    FORMAT = "femticpy"
elif "femtic_site" in _wf:
    FORMAT = "femtic_site"
elif "kml" in _wf:
    FORMAT = "kml"
else:
    FORMAT = "other"
    print(" WHAT_FOR = %r not recognised (known: wal, femtic_site, femticpy,"
          " kml): writing the generic seven-column list (as for"
          " 'femtic_site')." % (WHAT_FOR,))

if FORMAT == "femticpy":
    CSV_FILE = WORK_DIR + "Sitelist_UTM.csv"
print("Writing data to file: " + CSV_FILE)

# =============================================================================
#  Build file list and read the sites
# =============================================================================
if UTM_ZONE_MODE not in ("common", "per_site"):
    sys.exit(f"UTM_ZONE_MODE must be 'common' or 'per_site', got {UTM_ZONE_MODE!r}")

edi_files = sorted([
    entry for entry in os.listdir(EDI_DIR)
    if entry.endswith(".edi") and not entry.startswith(".")
])
ns = len(edi_files)

sites = []
for sitenum, filename in enumerate(edi_files):
    print("reading data from: " + filename)

    name, _ = os.path.splitext(filename)
    edi_dict = load_edi(EDI_DIR + filename, drop_invalid_periods=True)
    lat = edi_dict["lat"]
    lon = edi_dict["lon"]
    if lat is None or lon is None or not (np.isfinite(lat) and np.isfinite(lon)):
        sys.exit(f"{filename}: no valid LAT/LONG in the EDI file (lat={lat!r}, lon={lon!r})")
    sites.append(dict(name=name, sitenum=sitenum, lat=float(lat),
                      lon=float(lon), elev=edi_dict["elev"]))

if FORMAT == "femticpy":
    # depth = -elev is written, so every site needs a valid elevation
    no_elev = [st["name"] for st in sites
               if st["elev"] is None or not np.isfinite(st["elev"])]
    if no_elev:
        sys.exit("WHAT_FOR = 'femticpy' writes depth = -elev, but these EDI "
                 "files have no valid ELEV: " + ", ".join(no_elev))

# =============================================================================
#  UTM zone(s) and coordinates
# =============================================================================
if FORMAT != "wal":
    for st in sites:
        st["zone_native"], st["north_native"] = utl.utm_zone_from_latlon(
            st["lat"], st["lon"])

    if UTM_ZONE_MODE == "common":
        zone_c, north_c = utl.utm_common_zone(
            [st["lat"] for st in sites], [st["lon"] for st in sites],
            override=UTM_ZONE_OVERRIDE)
    for st in sites:
        if UTM_ZONE_MODE == "common":
            st["zone"], st["north"] = zone_c, north_c
        else:
            st["zone"], st["north"] = utl.utm_zone_from_latlon(
                st["lat"], st["lon"], override=UTM_ZONE_OVERRIDE)
        e, n = utl.latlon_to_utm_zn(st["lat"], st["lon"], st["zone"], st["north"])
        st["easting"], st["northing"] = np.around(e, 1), np.around(n, 1)
        st["k"] = utl.utm_scale_factor(st["lat"], st["lon"], st["zone"])
        st["dcm"] = float((st["lon"] - utl.utm_central_meridian(st["zone"])
                           + 180.0) % 360.0 - 180.0)

    print()
    print(" %-14s %-8s %-8s %12s %13s %9s %8s"
          % ("site", "zone", "native", "easting", "northing", "dlon_cm", "scale"))
    for st in sites:
        zs = "%d%s" % (st["zone"], "N" if st["north"] else "S")
        zn = "%d%s" % (st["zone_native"], "N" if st["north_native"] else "S")
        print(" %-14s %-8s %-8s %12.1f %13.1f %+9.2f %8.5f"
              % (st["name"][:14], zs, zn, st["easting"], st["northing"],
                 st["dcm"], st["k"]))

    # ---- summary
    native = {}
    for st in sites:
        key = "%d%s" % (st["zone_native"], "N" if st["north_native"] else "S")
        native[key] = native.get(key, 0) + 1
    used = sorted({(st["zone"], st["north"]) for st in sites})
    used_s = ", ".join("%d%s" % (z, "N" if nh else "S") for z, nh in used)
    ks = np.array([st["k"] for st in sites])
    spread = float(ks.max() - ks.min())
    n_out = sum(1 for st in sites
                if (st["zone"], st["north"]) != (st["zone_native"], st["north_native"]))

    print()
    print(" UTM zone(s) used          : %s  (mode %s)" % (used_s, UTM_ZONE_MODE))
    print(" zones the sites belong to : %s"
          % ", ".join("%s (%d)" % (k_, v) for k_, v in sorted(native.items())))
    if len(native) > 1:
        if UTM_ZONE_MODE == "common":
            print(" survey crosses a zone boundary; %d of %d sites lie outside the zone"
                  " used (extended UTM)." % (n_out, ns))
        else:
            print(" survey crosses a zone boundary (sites in %d different zones)."
                  % len(native))
    print(" grid scale factor         : %.5f - %.5f  (spread %.0f ppm = %.2f m/km)"
          % (ks.min(), ks.max(), spread * 1e6, spread * 1e3))
    print(" largest distance from the central meridian: %.2f deg"
          % max(abs(st["dcm"]) for st in sites))
    if UTM_ZONE_MODE == "per_site" and len(used) > 1:
        print(" WARNING: sites are in different UTM zones (mode 'per_site'); their"
              " easting/northing are in different grids\n"
              "          and must not be used together. Use UTM_ZONE_MODE = 'common'.")
    if spread > UTM_MAX_SCALE_SPREAD:
        print(" WARNING: the UTM grid scale varies by %.0f ppm across the sites"
              " (limit %.0f ppm): distances between\n"
              "          far apart sites are distorted by up to %.2f m/km. Sites this"
              " far from the central meridian are better\n"
              "          placed with a local projection (e.g. transverse Mercator"
              " centred on the survey)."
              % (spread * 1e6, UTM_MAX_SCALE_SPREAD * 1e6, spread * 1e3))
    print()

# =============================================================================
#  Write the site list
# =============================================================================
with open(CSV_FILE, "w") as f:

    # csv.writer bakes in its delimiter at construction time, so it is chosen
    # here, before the writer is created.
    if FORMAT == "wal":
        sitelist = csv.writer(f, delimiter=" ")
        sitelist.writerow(["Sitename", "Latitude", "Longitude"])
        sitelist.writerow([ns, " ", " "])
    elif FORMAT == "femticpy":
        # tab-separated, no header and no "#" lines (WRITE_ZONE_HEADER is not
        # used): the zone is reported on the console only.
        sitelist = csv.writer(f, delimiter="\t")
    else:
        sitelist = csv.writer(f, delimiter=DELIM)
        if WRITE_ZONE_HEADER:
            if UTM_ZONE_MODE == "common" or len(used) == 1:
                sitelist.writerow(
                    ["# utm_zone=%d hemisphere=%s mode=%s central_meridian=%.1f"
                     % (used[0][0], "N" if used[0][1] else "S", UTM_ZONE_MODE,
                        utl.utm_central_meridian(used[0][0]))])
            else:
                sitelist.writerow(
                    ["# utm_zone=per_site zones=%s mode=%s" % (
                        "/".join("%d%s" % (z, "N" if nh else "S") for z, nh in used),
                        UTM_ZONE_MODE)])
            if UTM_ZONE_MODE == "common":
                sitelist.writerow(
                    ["# easting/northing [m] WGS84 UTM; sites outside the zone are"
                     " projected into it (extended UTM)"])
            else:
                sitelist.writerow(
                    ["# WARNING: easting/northing of sites in different zones are"
                     " in different grids"])

    for st in sites:
        if FORMAT == "wal":
            sitelist.writerow([st["name"], st["lat"], st["lon"]])
        elif FORMAT == "femticpy":
            # km with 2 decimals (10 m); depth below sea level = -elev
            sitelist.writerow(
                [st["name"],
                 float(np.around(st["easting"] * 0.001, 2)),
                 float(np.around(st["northing"] * 0.001, 2)),
                 0.0 - float(st["elev"])])
        else:
            # "femtic_site", "kml" and other: seven columns
            sitelist.writerow(
                [st["name"], st["lat"], st["lon"], st["elev"], st["sitenum"],
                 st["easting"], st["northing"]])
