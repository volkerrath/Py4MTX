#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_plot_sitemap.py
======================

Site map of a FEMTIC run, or of a whole ensemble, on a DEM (Copernicus) background:
every site is an icon (marker) coloured by its normalized RMS misfit (nRMS), for
frequency bands and data components. For an ensemble the colour is a statistic
of the member nRMS at the site. Profile lines and other polylines (faults, coast
lines, ...) can be drawn on the maps.

What it does
------------
1. Reads one results_iter<N>.h5 from WORK_DIR (ITERATION = "best", "last" or
   a number; as in femtic_ens_misfit.py / femtic_mod_misfit.py) with
   femtic_misfit.resolve_run, and the site list site.dat (name, lat, lon,
   elev, sitenum, easting, northing) from WORK_DIR/templates (TEMPLATES_DIR;
   if absent there, WORK_DIR itself and ../templates, next to WORK_DIR). With
   several members (see "Ensemble") every member is read on its own.
2. Assigns the site ids of the results file to the sites of site.dat, by
   position (SITE_MATCH = "coords"; the results file holds x, y in metres
   relative to the model origin, site.dat the UTM easting / northing) or by
   number (SITE_MATCH = "id"; site_id = sitenum + SITE_ID_OFFSET). All data
   types of one site (MT, VTF, ...) count for the site; HTF and NMT rows have no
   station position and are only counted in the global values. Two stations at
   one place (several ids at one position and several site.dat entries there)
   get one entry each, in the order of the numbers.
3. nRMS for every combination of a frequency band (BANDS) and a component
   selection (COMPONENTS), per site and for the whole data set ("global"):

       nRMS = sqrt( mean( ((obs - cal) / err)**2 ) )

   real and imaginary parts separately (the imaginary part is left out for
   real-valued data types), non-finite values and err <= 0 skipped; this is
   the nRMS of femtic_misfit.site_nrms, and the global value for all bands
   and all components is femtic_misfit.data_rms.
4. Basemap: the extent comes from the user parameters (MAP_EXTENT_MODE: the
   sites plus a margin, a lon / lat box, or a centre and a size in km). The
   DEM is the Copernicus DEM GLO-30 (DEM_SOURCE = "copernicus30", the default;
   "copernicus90" for large maps): its 1 x 1 degree GeoTIFF tiles are
   downloaded into DEM_CACHE_DIR if they are not there yet and read with
   rasterio. Other sources: the AWS Terrain Tiles ("skadi"), tiles of your
   own ("tiles"), one file ("file": GeoTIFF with rasterio, .npz with lon,
   lat, z, or an .hgt tile), or none ("none"). It is shown as a shaded relief.
5. One figure per page (PAGE_BY): the panels are the maps of the selections,
   the sites drawn as icons coloured by their nRMS (colour map CMAP_NRMS,
   limits NRMS_VMIN / NRMS_VMAX, one colour scale for all pages; values
   beyond the limits get the end colours and arrows on the colour bar). Sites
   without data in a selection are drawn as open icons.
6. Tables: <OUTPUT_BASENAME>_nrms.csv and _n.csv (nRMS and number of values,
   one row per site, one column per selection) and _summary.txt (global nRMS
   and the spread of the site values per selection); for an ensemble also
   _members.csv (one row per member) and _ens_stats.csv (mean, median, std,
   min, max over the members per selection and site); for profiles
   _profiles.csv and _profile_sites.csv (see "Profiles and lines").

Ensemble
--------
WORK_DIR = the root of an ensemble: <WORK_DIR>/templates/site.dat and one
directory per member with results_iter*.h5. If RESULTS_PATH (default WORK_DIR)
holds no results file, its sub-directories that do are the members; RUN_PATHS
(run directories, results files or globs, relative to WORK_DIR) names them
explicitly. Every member gets its own site assignment and nRMS, against its own
observed data (as in its .cnv file; the RMS1 of the ensemble mean is not
computed). The value of a site is ENS_STAT over the members: "mean", "median"
(default), "min", "max", "std" (sample std; own colour map CMAP_SPREAD, linear
scale from 0) or a percentile "pNN" ("p90"). MEMBER_NRMS_MAX leaves out members
with a larger nRMS of the iteration. A member that cannot be read is skipped
with a warning (and listed in _members.csv); with one member left or only one
found, the output is that of a single run.

Profiles and lines
------------------
PROFILES (cross-section lines) and LINES (faults, coast lines, caldera rims,
...) are lists of dicts (the keys are listed at the settings and in the readme)
that are drawn on every map. Positions are given in `coords`: "lonlat" (default),
"latlon", "utm" (easting, northing in m, in the UTM zone of site.dat, which is
found by comparing the easting / northing of the sites with their lat / lon) or
"model" (x, y in m of the FEMTIC model, as site_x, site_y of the results file;
needs the sites assigned by position). Lines are given by points or read from a
file: text (GMT multi-segment, two columns), GeoJSON, KML or KMZ (the last three
always lon, lat). A profile is given by points, a file, start and end, or
centre, azimuth and length_km; ticks_km marks the distance along it, width_km
gives it a swath: the sites within width / 2 of the profile (not beyond its ends)
get a ring on the map and are listed with their distance along and across the
profile and their nRMS in _profile_sites.csv. Distances are flat-earth km on a
sphere (the scale of the map), about 0.3 % less east-west than on the WGS84
ellipsoid at mid latitudes; use profiles of up to a few 100 km.
The end points may carry their label: start=(x, y, "A"), end=(x, y, "A'");
PROFILE_END_MARKER adds a symbol there. Profiles and lines are cut at the map
box made smaller by OVERLAY_CUT_MARGIN_KM (default 1 km), so that both ends of a
cut profile are inside the map; the tick distances and the tables still refer to
the whole profile (None = not cut). The map box itself does not depend on them
(MAP_INCLUDE_PROFILES = False): what goes beyond it is cut at it.

Selections
----------
BANDS      "all"      all frequencies only
           "decades"  all, and bands of BAND_PER_DECADE per decade (period or
                      frequency, BAND_AXIS), with at least BAND_MIN_N data
                      rows; the bands of femtic_misfit (after Baba, 2023)
           list       ["all", (0.01, 1.0), (1.0, 100.0)]: limits in s
                      (BAND_AXIS = "period") or Hz ("frequency"), lower limit
                      included, upper excluded; bands with fewer than
                      BAND_MIN_N rows are dropped
COMPONENTS list of selections: "all", a group of COMPONENT_GROUPS ("Z", "Zoff",
           "Zdiag", "T", ...), a component name ("Zxy"; any case, as written
           in the results file), or several joined by "+" ("Zxy+Zyx").
DATATYPES  None = all data types, or a list of names, e.g. ["MT", "VTF"].

DEM source
----------
DEM_SOURCE selects the elevation model of the relief:

  "copernicus30"  Copernicus DEM GLO-30 (the default): global, 1 arc second
                  (30 m; coarser in longitude at high latitudes), a surface
                  model (it includes buildings and vegetation) with heights
                  above the EGM2008 geoid. Cloud Optimized GeoTIFF tiles of
                  the open data bucket copernicus-dem-30m on AWS (no key, no
                  login), one file per 1 x 1 degree: https://copernicus-dem-30m
                  .s3.amazonaws.com/Copernicus_DSM_COG_10_N65_00_W017_00_DEM/
                  Copernicus_DSM_COG_10_N65_00_W017_00_DEM.tif. There are no
                  tiles over the open ocean (the maps get DEM_FILL_MISSING
                  there), and a few tiles of some countries are not public.
  "copernicus90"  GLO-90 (3 arc seconds), bucket copernicus-dem-90m: for maps
                  of a few 100 km, where GLO-30 needs more tiles than
                  DEM_MAX_TILES.
  "skadi"         The AWS Terrain Tiles (Mapzen), .hgt tiles of 1 arc second:
                  a composite of several elevation data sets, not one model
                  (the SRTM is the main source between about 60 N and 56 S,
                  other sources are used at high latitudes, and coarse sources
                  are interpolated). The former "srtm" setting. No rasterio.
  "tiles"         Your own tiles: DEM_URL_TEMPLATE = a URL or a path (relative
                  paths: from WORK_DIR) with the placeholders {name} (N65W017),
                  {NS}, {lat}, {EW}, {lon}; GeoTIFF (.tif) or .hgt tiles.
                  A path is used as it is (nothing is copied to the cache).
  "file"          One file (DEM_FILE): GeoTIFF, .npz or an .hgt tile.
  "none"          No relief.

The tiles of a URL are downloaded once into DEM_CACHE_DIR and never again
(tiles can also be put there by hand); DEM_DOWNLOAD = False uses only those.
DEM_URL_TEMPLATE replaces the URL of the source (a mirror of the bucket).
GeoTIFF files are read with rasterio, only the part inside the map, averaged
down to about DEM_MAX_PIXELS pixels per side; a raster in a projected
coordinate system (a national DEM, UTM) is warped to lon / lat first. The credit
line of the maps (DEM_CREDIT = "auto") is the notice that the Copernicus
licence asks for, or a note on the AWS Terrain Tiles; check the terms of any
other source you use.

Copernicus GLO-30 from Python (rasterio needed; the tiles of the map are
downloaded once into DEM_CACHE_DIR):

    import femtic_plot_sitemap as sm
    cfg = sm.settings("/data/krafla/krafla_rto_26", DEM_SOURCE="copernicus30",
                      DEM_CACHE_DIR="/data/dem_cache", DEM_STYLE="terrain")
    sm.run(cfg)

Usage
-----
    python femtic_plot_sitemap.py                  # settings below
    python femtic_plot_sitemap.py -w /data/run01   # another WORK_DIR
    python femtic_plot_sitemap.py --dem copernicus30   # Copernicus DEM GLO-30 (the default)
    python femtic_plot_sitemap.py --dem copernicus90   # GLO-90, for large maps
    python femtic_plot_sitemap.py --dem none       # no DEM (no download)
    python femtic_plot_sitemap.py -i last          # last iteration
    python femtic_plot_sitemap.py -w /data/ens     # ensemble root: median of the members
    python femtic_plot_sitemap.py -w /data/ens --stat std   # spread of the members
    python femtic_plot_sitemap.py --runs "kra*/" --stat p90 # members by glob, 90th percentile
    python femtic_plot_sitemap.py --profile "A-A'" -16.82 65.70 -16.74 65.74   # profile line
    python femtic_plot_sitemap.py --line gis/faults.txt    # polylines (text, GeoJSON, KML)

All relative paths of the settings refer to WORK_DIR. Return code 0 on
success, 1 if the data could not be read or nothing could be plotted, 2 for a
wrong usage (WORK_DIR missing).

@author vrath
@project py4mt
@created 2026-11-21
@modified 2026-11-22

AI-GENERATED CODE NOTICE
-------------------------
Author      : Claude Sonnet 5.5 (Anthropic)
Generated   : 2026-11-21
Notice      : This code is AI-generated. Review and test it before any
              production use. The ensemble reading (2026-11-21, later the same
              day) was tested with synthetic ensembles (statistics against
              numpy and a brute-force nRMS per member and site) and with a
              pseudo-ensemble of three real Krafla iterations (1, 7, 8 as three
              members); it has NOT been run on a real RTO ensemble. The
              single-run part was tested with synthetic results files
              (the HDF5 reader replaced by a stand-in; h5py was not
              available) and with synthetic tiles, and run on a real FEMTIC
              results file (Krafla RTO member, iteration 8, read with a
              libhdf5 reader written for the test; the global nRMS agrees
              with the RMS of the .cnv file, and the 100 site ids agree with
              the site.dat by position and by number), but without a DEM
              there. The download from the server has NOT been run (no
              network where it was written; the URL pattern is the one of
              the AWS documentation), nor has the GeoTIFF reader.
              The profiles and lines (2026-11-22) were tested with synthetic
              data (UTM transform against a numerical meridian arc and the
              100 real Krafla site.dat positions; projection of the sites
              against a plain loop; readers with hand-made text, GeoJSON,
              KML and KMZ files) and run on the real Krafla results; the
              readers have NOT been tried on files from GIS programs.
              The GeoTIFF tiles of the Copernicus DEM (2026-11-22, later the
              same day) are read with rasterio, which was NOT installed where
              this was written: the code was tested with a stand-in module that
              imitates the few rasterio calls it makes (open, transform, read
              of a window, WarpedVRT), not with rasterio itself, and the
              download from the AWS bucket has NOT been run (the file names
              are those of the bucket documentation and of listings of it),
              and the warping of a raster in a projected system has not been
              tried on a real file.

Reference
---------
Copernicus GLO-30 Digital Elevation Model, data record of OpenTopography:
doi:10.5069/G9028PQB (data: ESA, DLR, Airbus Defence and Space; the notice of
the licence is the default credit line of the maps).

Farr, T. G., Rosen, P. A., Caro, E., Crippen, R., Duren, R., Hensley, S.,
Kobrick, M., Paller, M., Rodriguez, E., Roth, L., Seal, D., Shaffer, S.,
Shimada, J., Umland, J., Werner, M., Oskin, M., Burbank, D., Alsdorf, D.
(2007): The Shuttle Radar Topography Mission. Reviews of Geophysics 45,
RG2004. doi:10.1029/2005RG000183

Changelog
---------
2026-11-21  Claude Sonnet 5.5 (Anthropic), AI-generated, review before use:
            created. Per-site nRMS for bands x components on an SRTM shaded
            relief, DEM download from the AWS Terrain Tiles with a cache.
            site.dat is searched in TEMPLATES_DIR, WORK_DIR and next to
            WORK_DIR (ensemble layout); ids at one position are given the
            site.dat entries at that position in the order of their numbers
            (two stations at one place in the Krafla data).
2026-11-21  Claude Sonnet 5.5 (Anthropic), AI-generated, review before use:
            a full ensemble is read. WORK_DIR = ensemble root (members and
            templates/) or RUN_PATHS; every member is read, matched to the
            sites and evaluated on its own; ENS_STAT (mean, median, min, max,
            std, percentile) over the members per site and selection;
            MEMBER_NRMS_MAX; unreadable members are skipped; new tables
            _members.csv and _ens_stats.csv; ensemble summary, titles ({M},
            {stat}), colour bar label and a sequential scale (CMAP_SPREAD) for
            the std; command line --runs, --stat. Output of a single run
            unchanged.
2026-11-22  Claude Sonnet 5.5 (Anthropic), AI-generated, review before use:
            profile lines and other polylines on the maps. PROFILES (points,
            file, start + end, or centre + azimuth + length_km; swath with the
            sites in it, ticks with distances, end labels) and LINES (points,
            segments, text / GMT, GeoJSON, KML, KMZ; polygons with fill),
            coordinates lonlat, latlon, utm (own Kruger series, zone from
            site.dat) or model (frame and shift of the site assignment);
            legend with the labelled items (LEGEND_LOC); the map extent can
            include the profiles (MAP_INCLUDE_PROFILES); new tables
            _profiles.csv and _profile_sites.csv; command line --profile,
            --line, --no-overlays. Without PROFILES and LINES the output is
            unchanged.
2026-11-22  Claude Sonnet 5.5 (Anthropic), AI-generated, review before use:
            the default DEM is the Copernicus DEM GLO-30 instead of the AWS
            Terrain Tiles. DEM_SOURCE "copernicus30" (default), "copernicus90",
            "skadi" (the former "srtm", now named for what the tiles are:
            "srtm" is refused with a hint), "tiles" (own URL or path pattern,
            GeoTIFF or .hgt), "file", "none". GeoTIFF tiles and files are read
            with rasterio, windowed and averaged down to DEM_MAX_PIXELS; a
            raster in a projected system is warped to lon / lat; tiles are
            merged (also with different pixel sizes; ocean tiles are filled
            over the map and a margin of three pixels only); credit line of the maps
            = the licence notice, wrapped to the width of the panel;
            DEM_URL_TEMPLATE is now None = the URL of the source;
            DEM_CACHE_DIR default "dem_cache" (was "srtm_cache"; tiles of the
            old cache can be copied there).
2026-11-22  Claude Sonnet 5.5 (Anthropic), later: example lines for the
            Copernicus GLO-30 in the CONFIG block, the Usage and the readme
            (cache, colour relief, offline, own tile names, one GeoTIFF).
2026-11-22  Claude Sonnet 5.5 (Anthropic), AI-generated, review before use:
            CBAR_LENGTH, CBAR_WIDTH [cm] set the size of the colour bar (the
            length with the extension triangles; the figure is laid out without
            drawing and the bar made again until it has the size; CBAR_SHRINK
            as before without them; CBAR_SHRINK is checked, 0 < s <= 1).
            Profiles and lines are cut at the map box made smaller by
            OVERLAY_CUT_MARGIN_KM (default 1 km; None = not cut): both ends of
            a cut profile are inside the map, the end labels stand at the cut
            ends, the swath is flat there, rings and ticks follow, the ticks
            keep the distances of the whole profile (tables unchanged; message
            with the km shown). End labels in the start / end keys,
            start=(x, y, "A"), end=(x, y, "A'"); PROFILE_END_MARKER marks the
            ends with a symbol. Tests on synthetic data only.
2026-11-22  Claude Sonnet 5.5 (Anthropic), AI-generated, review before use:
            MAP_INCLUDE_PROFILES now defaults to False: the map box (MAP_EXTENT_MODE
            "sites": sites + margin) is not enlarged by profiles that go beyond it;
            they are cut at that box (OVERLAY_CUT_MARGIN_KM). True restores the
            earlier behaviour (box large enough for the profiles).
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import gzip
import http.client
import io
import math
import os
import re
import shutil
import sys
import inspect
import urllib.error
import urllib.parse
import urllib.request
import warnings
import zipfile
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# py4mt environment: make the modules importable (femtic_misfit.py, femtic.py)
# ---------------------------------------------------------------------------
PY4MTX_ROOT = os.environ.get("PY4MTX_ROOT", "")
if PY4MTX_ROOT:
    for _pth in (os.path.join(PY4MTX_ROOT, "py4mt", "modules"),
                 os.path.join(PY4MTX_ROOT, "py4mt", "scripts")):
        if os.path.isdir(_pth) and _pth not in sys.path:
            sys.path.insert(0, _pth)

import femtic_misfit as fm  # noqa: E402
import util as utl
from version import versionstrg

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())
titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng + "\n\n")

# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------
# --- working directory and input ---
# Every relative path below is taken from WORK_DIR (an absolute path, or one
# that starts with ~, is used as given). None or "" = the current directory.
# WORK_DIR: Optional[str] = "./"
# RESULTS_PATH: str = "."               # run directory (searched for ITER_PATTERN) or a results file
WORK_DIR: Optional[str] = r"/media/vrath/LargeBack/Ensembles/krafla2026/krafla_ens/"
RESULTS_PATH: str = WORK_DIR              # run directory (searched for ITER_PATTERN) or a results file


ITERATION: Union[str, int] = "best"   # "best" (smallest nRMS), "last", or a number
ITER_PATTERN: str = "results_iter*.h5"
TEMPLATES_DIR: str = "templates"      # sub-directory of WORK_DIR with site.dat
SITE_DAT: Optional[str] = "auto"      # "auto": TEMPLATES_DIR, WORK_DIR, then ../<TEMPLATES_DIR>; or a path
SITE_MATCH: str = "auto"              # "auto", "coords" (by position) or "id" (sitenum + SITE_ID_OFFSET)
SITE_MATCH_TOL: float = 15.0          # [m] largest distance for a match by position
SITE_ID_OFFSET: int = 0               # site_id in the results file = sitenum + SITE_ID_OFFSET
DATATYPES: Optional[Sequence[str]] = None   # None = all, or e.g. ["MT", "VTF"]

# --- ensemble ---
# WORK_DIR = the root of an ensemble (member directories and templates/): if
# RESULTS_PATH holds no results file, but sub-directories do, they are the
# members. RUN_PATHS names the members explicitly. With one member only, the
# ensemble settings have no effect.
RUN_PATHS: Optional[Sequence[str]] =  ["kra*/"] # None   # run directories / results files / globs, e.g. ["kra*/"]
ENS_STAT: str = "median"              # per site over the members: "mean", "median", "min", "max",
                                      # "std" (spread) or "pNN" (percentile, e.g. "p90")
MEMBER_NRMS_MAX: Optional[float] = None   # members with a larger nRMS of the iteration are left out

# --- nRMS selections ---
BANDS: Union[str, Sequence[Any]] = "decades"   # "all", "decades", or ["all", (lo, hi), ...]
BAND_AXIS: str = "period"             # "period" [s] or "frequency" [Hz]
BAND_PER_DECADE: int = 1              # for "decades"
BAND_MIN_N: int = 20                  # bands with fewer data rows are dropped
COMPONENTS: Sequence[str] = ["all", "Zoff", "Zdiag", "T"]
COMPONENT_GROUPS: Dict[str, Sequence[str]] = {
    "Z": ("Zxx", "Zxy", "Zyx", "Zyy"),
    "Zoff": ("Zxy", "Zyx"),
    "Zdiag": ("Zxx", "Zyy"),
    "T": ("Tzx", "Tzy"),
    "rho": ("rhoXX", "rhoXY", "rhoYX", "rhoYY"),
    "phs": ("phsXX", "phsXY", "phsYX", "phsYY"),
    "PT": ("PTxx", "PTxy", "PTyx", "PTyy"),
}

# --- basemap: extent and coordinates ---
MAP_EXTENT_MODE: str = "sites"        # "sites" (sites + margin), "box" or "center"
MAP_MARGIN_FRAC: float = 0.15         # "sites": margin as a fraction of the extent of the sites
MAP_MARGIN_MIN_KM: float = 2.0        # "sites": but at least this [km]
MAP_BOX: Optional[Tuple[float, float, float, float]] = None   # "box": (lon_min, lon_max, lat_min, lat_max) [deg]
MAP_CENTER: Optional[Tuple[float, float]] = None              # "center": (lon, lat) [deg]
MAP_SIZE_KM: Tuple[float, float] = (50.0, 50.0)               # "center": (width, height) [km]
DISPLAY_COORDS: str = "latlon"        # "latlon" (degrees) or "km" (km from the centre of the map)
MAP_GRID: bool = False
SCALE_BAR: bool = True
SCALE_BAR_KM: Optional[float] = None  # None = a round length of about 1/5 of the width

# --- DEM ---
# The relief is the hillshade of a digital elevation model. DEM_SOURCE:
#   "copernicus30"  Copernicus DEM GLO-30 (30 m, a surface model, heights above EGM2008): the
#                   Cloud Optimized GeoTIFF tiles of the AWS open data bucket copernicus-dem-30m,
#                   downloaded into DEM_CACHE_DIR and read with rasterio
#   "copernicus90"  Copernicus DEM GLO-90 (90 m; bucket copernicus-dem-90m): large maps
#   "skadi"         AWS Terrain Tiles (Mapzen): a composite of several DEMs in .hgt tiles
#                   (no rasterio needed)
#   "tiles"         tiles of your own: DEM_URL_TEMPLATE = URL or path pattern, GeoTIFF or .hgt
#   "file"          one DEM file, DEM_FILE: GeoTIFF (rasterio), .npz (lon, lat, z) or one .hgt tile
#   "none"          no relief
# Examples for the Copernicus DEM GLO-30 (copy the lines to the settings below):
#   DEM_SOURCE = "copernicus30"                      # the default: download once into DEM_CACHE_DIR
#   DEM_CACHE_DIR = "/data/dem_cache"                # a cache shared by all runs and maps
#   DEM_STYLE = "terrain"; DEM_CMAP = "gist_earth"   # colour + shading instead of the grey hillshade
#   DEM_MAX_PIXELS = 2400                            # finer relief of a small map (default 1600)
#   DEM_DOWNLOAD = False                             # offline: only the tiles that are in DEM_CACHE_DIR,
#                                                    # e.g. Copernicus_DSM_COG_10_N65_00_W017_00_DEM.tif
#   DEM_SOURCE = "tiles"
#   DEM_URL_TEMPLATE = "/data/cop30/Copernicus_DSM_COG_10_{NS}{lat:02d}_00_{EW}{lon:03d}_00_DEM.tif"   # own folder / names
#   DEM_SOURCE = "file"; DEM_FILE = "/data/krafla_cop30.tif"   # one GeoTIFF (e.g. cut out with gdal_translate)
DEM_SOURCE: str = "copernicus30"
DEM_URL_TEMPLATE: Optional[str] = None   # None = the tiles of DEM_SOURCE; else a mirror, or the pattern of "tiles":
                                         # placeholders {name} N65W017, {NS}, {lat}, {EW}, {lon}; a path (relative: to WORK_DIR) is no download
DEM_DOWNLOAD: bool = True             # False: use only the tiles in DEM_CACHE_DIR
DEM_CACHE_DIR: str = "dem_cache"      # downloaded tiles are kept here (give an absolute path to share it between runs)
DEM_FILE: Optional[str] = None        # "file": GeoTIFF (needs rasterio), .npz (lon, lat, z) or one .hgt tile
DEM_TIMEOUT: float = 60.0             # [s] per tile
DEM_MAX_TILES: int = 16               # more tiles than this are refused (too large a map)
DEM_MAX_PIXELS: int = 1600            # the DEM is thinned out to at most this many pixels per side
DEM_FILL_MISSING: Optional[float] = 0.0   # elevation of tiles the server does not have (ocean); None = no value
DEM_STRICT: bool = False              # True: a failed download stops the script (else: map without relief)
DEM_STYLE: str = "terrain"          # "hillshade" (grey) or "terrain" (colour + shading, DEM_CMAP)
DEM_CMAP: str = "gist_earth"
HILLSHADE_AZIMUTH: float = 315.0
HILLSHADE_ALTITUDE: float = 45.0
HILLSHADE_VERT_EXAG: float = 1.0
BASEMAP_CMAP: str = "Greys"           # "hillshade": colours of the shading
BASEMAP_ALPHA: float = 0.9
DEM_CREDIT: Optional[str] = None    # text at the lower right of every map: "auto", None or text

# --- icons ---
SITE_MARKER: str = "o"
SITE_SIZE: float = 80.0               # [pt^2]
SITE_EDGECOLOR: str = "black"
SITE_LW: float = 0.8
SITE_NODATA_EDGECOLOR: str = "0.35"   # open icons: sites without data in the selection
SITE_LABELS: bool = False             # site names next to the icons
LABEL_FONTSIZE: float = 6.0
CMAP_NRMS: str = "jet"            # diverging colour map, centred on NRMS_REFERENCE
# CMAP_NRMS: str = "RdBu_r"             # diverging colour map, centred on NRMS_REFERENCE
CMAP_SPREAD: str = "YlOrRd"           # ENS_STAT = "std": sequential colour map, linear scale from 0
NRMS_REFERENCE: Optional[float] = 1.0   # target nRMS = centre colour; None = linear scale
NRMS_VMIN: Optional[float] = 0.6     # colour limits; None = from the data ("std": 0)
NRMS_VMAX: Optional[float] = 1.8

# --- profiles and polylines ---
# Profile lines (cross-section lines) and other polylines / polygons (faults, coast
# lines, caldera rims, ...) on the maps; lists of dicts, see the readme.
#   profile: name, one of points | file | start + end | center + azimuth + length_km,
#            coords, width_km (swath), ticks_km, tick_labels, labels, label, color, lw, ls
#            start / end: (a, b) or (a, b, "A") = with the label ("marker") of that end; without
#            it labels = ("A", "A'") or, if there is none, the two halves of the name "A-A'"
#   line:    label, one of file | points | segments, coords, closed, fill, alpha,
#            color, lw, ls, halo
#   coords:  "lonlat" (default), "latlon", "utm" (easting, northing of the site.dat zone)
#            or "model" (x, y in m of the FEMTIC model); GeoJSON and KML are always lon, lat
# PROFILES: Sequence[dict] = []         # e.g. [dict(name="A-A'", start=(-16.80, 65.68, "A"), end=(-16.70, 65.72, "A'"), width_km=4.0, ticks_km=5.0)]
#                                       # UTM: coords="utm", start=(405000, 7291800, "A"), end=(424000, 7287800, "A'")
PROFILES: Sequence[dict] = [
    dict(name="A-A'", start=(405000, 7291800, "A"), end=(424000, 7287800, "A'"), coords="utm", 
         width_km=1.0,) # ticks_km=5.0)
    ]                       
LINES: Sequence[dict] = []            # e.g. [dict(label="Faults", file="gis/faults.txt", color="tab:red")]
PROFILE_COLOR: str = "black"
PROFILE_LW: float = 1.8
PROFILE_FONTSIZE: float = 9.0         # end labels (the distance labels are 2 pt smaller)
PROFILE_SWATH_ALPHA: float = 0.15     # shading of the swath (width_km of the profile)
PROFILE_MARK_SITES: bool = True       # open circles around the sites in the swath
PROFILE_END_MARKER: Optional[str] = None   # symbol ("o", "s", "D", "^", ...) at the two ends of every profile
                                      # (the ends of the cut part, see OVERLAY_CUT_MARGIN_KM); None = none
LINE_COLOR: str = "tab:green"
LINE_LW: float = 1.2
LINE_HALO: bool = True                # white outline under the lines
MAP_INCLUDE_PROFILES: bool = False    # MAP_EXTENT_MODE "sites": False = the map box does not depend on the profiles:
                                      # what goes beyond it is cut at it (OVERLAY_CUT_MARGIN_KM); True = the box is
                                      # made large enough for the profiles
OVERLAY_STRICT: bool = False          # True: a profile / line that cannot be made stops the script
OVERLAY_CUT_MARGIN_KM: Optional[float] = 1.0   # profiles and lines are cut at the map box, made smaller by this
                                      # margin [km] on every side, so that the ends of a cut profile (and its
                                      # labels) are inside the map; None = not cut (the map edge clips them)
LEGEND_LOC: Optional[str] = "upper left"   # legend (no-data icon, labelled profiles and lines); None = none

# --- figures and output ---
PAGE_BY: str = "band"                 # one figure per "band" (panels: components), per "component"
                                      # (panels: bands), per "none" (one map each), or "all" (one figure)
PANEL_COLS: int = 2                   # panels per row
PANEL_SIZE: Tuple[float, float] = (4.8, 4.4)   # [inch] of one panel
TITLE: Optional[str] = "auto"         # "auto", None / "" (none) or text with {iter}, {nrms}, {M}, {stat}
CBAR_LOCATION: Optional[str] = None   # None / "right", "left", "top", "bottom"
CBAR_SHRINK: float = 0.7              # length of the colour bar: fraction (0 .. 1] of the panels along it
CBAR_LENGTH: Optional[float] = None   # [cm] length of the colour bar; None = CBAR_SHRINK x the panels
CBAR_WIDTH: Optional[float] = None    # [cm] width (thickness); None = 1/20 of the length
TICK_FONTSIZE: float = 8.0
DPI: int = 200
SHOW: bool = False
OUTPUT_DIR: str = "sitemap"           # relative to WORK_DIR
OUTPUT_BASENAME: str = "femtic_sitemap"
OUTPUT_FORMATS: Sequence[str] = ["png", "pdf"]

# ---------------------------------------------------------------------------
# End of the configuration
# ---------------------------------------------------------------------------
_SETTING_NAMES = tuple(k for k in list(globals())
                       if k.isupper() and not k.startswith("_") and k != "PY4MTX_ROOT")

_TAG = "femtic_plot_sitemap:"
_EARTH_RADIUS_M = 6371008.8
_KM_PER_DEG = _EARTH_RADIUS_M * math.pi / 180.0 / 1000.0     # one degree of latitude [km]
_HGT_EXTENSIONS = (".hgt", ".hgt.gz", ".hgt.zip", ".SRTMGL1.hgt.zip", ".SRTMGL3.hgt.zip")
_RASTER_EXTENSIONS = (".tif", ".tiff")
_DEM_SOURCES = ("copernicus30", "copernicus90", "skadi", "tiles", "file", "none")
_COP_NOTICE = ("produced using Copernicus WorldDEM-{n} \u00a9 DLR e.V. 2010-2014 and \u00a9 Airbus "
               "Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union "
               "and ESA; all rights reserved")
# tiles of the DEM_SOURCEs: URL pattern ({name}, {NS}, {lat}, {EW}, {lon}) and the credit line of the maps
_DEM_PRESETS: Dict[str, dict] = {
    "copernicus30": dict(
        template=("https://copernicus-dem-30m.s3.amazonaws.com/"
                  "Copernicus_DSM_COG_10_{NS}{lat:02d}_00_{EW}{lon:03d}_00_DEM/"
                  "Copernicus_DSM_COG_10_{NS}{lat:02d}_00_{EW}{lon:03d}_00_DEM.tif"),
        credit="Relief: " + _COP_NOTICE.format(n=30)),
    "copernicus90": dict(
        template=("https://copernicus-dem-90m.s3.amazonaws.com/"
                  "Copernicus_DSM_COG_30_{NS}{lat:02d}_00_{EW}{lon:03d}_00_DEM/"
                  "Copernicus_DSM_COG_30_{NS}{lat:02d}_00_{EW}{lon:03d}_00_DEM.tif"),
        credit="Relief: " + _COP_NOTICE.format(n=90)),
    "skadi": dict(
        template="https://s3.amazonaws.com/elevation-tiles-prod/skadi/{NS}{lat:02d}/{name}.hgt.gz",
        credit="Relief: AWS Terrain Tiles (Mapzen), a composite of several elevation data sets"),
}
_PAGE_BY = ("band", "component", "none", "all")
_CBAR_LOCATIONS = (None, "right", "left", "top", "bottom")


def _say(msg: str) -> None:
    print(f"{_TAG} {msg}")


# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------
def resolve_work_dir(work_dir: Optional[str]) -> str:
    """Absolute WORK_DIR; None or "" = the current directory; ~ is expanded."""
    return os.path.abspath(os.path.expanduser(str(work_dir or ".")))


def anchor(path: str, work_dir: str) -> str:
    """`path` with ~ expanded; a relative path is joined to `work_dir`."""
    p = os.path.expanduser(str(path))
    return p if os.path.isabs(p) else os.path.join(work_dir, p)


def settings(work_dir: Optional[str] = None, **overrides: Any) -> Dict[str, Any]:
    """
    The settings of this script as a dict (uppercase names), with `overrides`
    applied and the paths anchored at WORK_DIR (`work_dir`, else the setting).
    A name that is no setting is an error (typo).
    """
    cfg = {k: globals()[k] for k in _SETTING_NAMES}
    for k, v in overrides.items():
        if k not in cfg:
            raise KeyError(f"{_TAG} unknown setting {k!r}")
        cfg[k] = v
    wd = resolve_work_dir(cfg["WORK_DIR"] if work_dir is None else work_dir)
    cfg["WORK_DIR"] = wd
    for key in ("RESULTS_PATH", "TEMPLATES_DIR", "DEM_CACHE_DIR", "OUTPUT_DIR"):
        cfg[key] = anchor(cfg[key], wd)
    for key in ("DEM_FILE",):
        if cfg[key]:
            cfg[key] = anchor(cfg[key], wd)
    if cfg["DEM_URL_TEMPLATE"] and not is_url(cfg["DEM_URL_TEMPLATE"]):
        cfg["DEM_URL_TEMPLATE"] = anchor(cfg["DEM_URL_TEMPLATE"], wd)        # path pattern
    if cfg["SITE_DAT"] and cfg["SITE_DAT"] != "auto":
        cfg["SITE_DAT"] = anchor(cfg["SITE_DAT"], wd)
    runs = cfg["RUN_PATHS"]
    if runs:
        cfg["RUN_PATHS"] = [anchor(p, wd) for p in ([runs] if isinstance(runs, str) else runs)]
    else:
        cfg["RUN_PATHS"] = None
    for key in ("PROFILES", "LINES"):                 # copies, with the files anchored
        items = cfg[key]
        items = [items] if isinstance(items, dict) else (list(items) if items else [])
        cfg[key] = [dict(it, file=anchor(it["file"], wd)) if isinstance(it, dict) and it.get("file")
                    else (dict(it) if isinstance(it, dict) else it) for it in items]
    return cfg


def check_settings(cfg: Dict[str, Any]) -> None:
    """Raise ValueError for a setting that cannot work (before any data is read)."""
    def one_of(key: str, allowed: Sequence[Any]) -> None:
        if cfg[key] not in allowed:
            raise ValueError(f"{key} = {cfg[key]!r}: must be one of {list(allowed)}")
    one_of("SITE_MATCH", ("auto", "coords", "id"))
    one_of("BAND_AXIS", ("period", "frequency"))
    one_of("MAP_EXTENT_MODE", ("sites", "box", "center"))
    one_of("DISPLAY_COORDS", ("latlon", "km"))
    if cfg["DEM_SOURCE"] == "srtm":
        raise ValueError("DEM_SOURCE = 'srtm' does not exist any more: use 'copernicus30' (the "
                         "default), 'copernicus90', or 'skadi' (the AWS Terrain Tiles that 'srtm' "
                         "meant)")
    one_of("DEM_SOURCE", _DEM_SOURCES)
    one_of("DEM_STYLE", ("hillshade", "terrain"))
    one_of("PAGE_BY", _PAGE_BY)
    one_of("CBAR_LOCATION", _CBAR_LOCATIONS)
    for key in ("CBAR_LENGTH", "CBAR_WIDTH"):
        v = cfg[key]
        if v is not None and not (isinstance(v, (int, float, np.integer, np.floating))
                                  and not isinstance(v, bool) and np.isfinite(v) and v > 0):
            raise ValueError(f"{key} = {v!r}: must be None or a number > 0 (cm)")
    em = cfg["PROFILE_END_MARKER"]
    if em is not None:
        try:
            from matplotlib.markers import MarkerStyle
            MarkerStyle(em)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"PROFILE_END_MARKER = {em!r} is no matplotlib marker: {exc}") from None
    mg = cfg["OVERLAY_CUT_MARGIN_KM"]
    if mg is not None and not (isinstance(mg, (int, float, np.integer, np.floating))
                               and not isinstance(mg, bool) and np.isfinite(mg) and mg >= 0):
        raise ValueError(f"OVERLAY_CUT_MARGIN_KM = {mg!r}: must be None or a number >= 0 (km)")
    sh = cfg["CBAR_SHRINK"]
    if not (isinstance(sh, (int, float, np.integer, np.floating)) and not isinstance(sh, bool)
            and 0.0 < sh <= 1.0):
        raise ValueError(f"CBAR_SHRINK = {sh!r}: must be a number in (0, 1]")
    if cfg["MAP_EXTENT_MODE"] == "box":
        b = cfg["MAP_BOX"]
        if b is None or len(b) != 4 or not (b[0] < b[1] and b[2] < b[3]):
            raise ValueError("MAP_BOX must be (lon_min, lon_max, lat_min, lat_max) "
                             "with lon_min < lon_max and lat_min < lat_max")
    if cfg["MAP_EXTENT_MODE"] == "center":
        c, s = cfg["MAP_CENTER"], cfg["MAP_SIZE_KM"]
        if c is None or len(c) != 2:
            raise ValueError("MAP_CENTER must be (lon, lat) for MAP_EXTENT_MODE = \"center\"")
        if len(s) != 2 or not (s[0] > 0 and s[1] > 0):
            raise ValueError("MAP_SIZE_KM must be (width, height) in km, both > 0")
    if cfg["DEM_SOURCE"] == "file" and not cfg["DEM_FILE"]:
        raise ValueError("DEM_SOURCE = \"file\" needs DEM_FILE")
    if cfg["DEM_SOURCE"] == "tiles" and not cfg["DEM_URL_TEMPLATE"]:
        raise ValueError("DEM_SOURCE = \"tiles\" needs DEM_URL_TEMPLATE")
    if cfg["DEM_SOURCE"] in ("copernicus30", "copernicus90", "skadi", "tiles"):
        tile_kind(dem_template(cfg["DEM_SOURCE"], cfg["DEM_URL_TEMPLATE"]))      # ValueError
    if int(cfg["PANEL_COLS"]) < 1:
        raise ValueError("PANEL_COLS must be >= 1")
    if int(cfg["DEM_MAX_PIXELS"]) < 2:
        raise ValueError("DEM_MAX_PIXELS must be >= 2")
    parse_stat(cfg["ENS_STAT"])
    if cfg["MEMBER_NRMS_MAX"] is not None and not float(cfg["MEMBER_NRMS_MAX"]) > 0:
        raise ValueError("MEMBER_NRMS_MAX must be > 0 (or None)")
    one_of("LEGEND_LOC", _LEGEND_LOCS)
    check_overlay_specs(cfg)


# ---------------------------------------------------------------------------
# Sites: site.dat and the assignment of the site ids
# ---------------------------------------------------------------------------
def read_site_dat(path: str) -> List[Dict[str, Any]]:
    """
    Sites of a site.dat of mt_make_sitelist.py: columns name, lat, lon, elev,
    sitenum, easting, northing (comma or white-space separated; lines that
    do not fit, and comments, are skipped). Returns a list of dicts with these
    keys. ValueError if there is no site, or lat / lon are out of range.
    """
    out: List[Dict[str, Any]] = []
    with open(path, "r", errors="replace") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            p = [x.strip() for x in (ln.split(",") if "," in ln else ln.split())]
            if len(p) < 7:
                continue
            try:
                out.append(dict(name=p[0], lat=float(p[1]), lon=float(p[2]),
                                elev=float(p[3]), sitenum=int(float(p[4])),
                                easting=float(p[5]), northing=float(p[6])))
            except ValueError:
                continue
    if not out:
        raise ValueError(f"no site (name, lat, lon, elev, sitenum, easting, northing) in {path}")
    lat = np.array([s["lat"] for s in out])
    lon = np.array([s["lon"] for s in out])
    if np.any(np.abs(lat) > 90.0) or np.any(np.abs(lon) > 180.0):
        raise ValueError(f"{path}: lat / lon out of range (columns name, lat, lon, ... "
                         "expected, in degrees)")
    return out


def site_search_dirs(templates_dir: str, work_dir: str) -> List[str]:
    """
    Directories searched for site.dat: TEMPLATES_DIR (inside WORK_DIR), WORK_DIR
    itself, and a directory of the same name next to WORK_DIR (the layout of an
    ensemble: <ensemble>/templates and <ensemble>/<member>).
    """
    wd = os.path.abspath(work_dir)
    sibling = os.path.join(os.path.dirname(wd), os.path.basename(os.path.normpath(templates_dir)))
    out: List[str] = []
    for d in (templates_dir, wd, sibling):
        if d not in out:
            out.append(d)
    return out


def find_site_dat(spec: Optional[str], search_dirs: Sequence[str]) -> Optional[str]:
    """spec "auto": the first site.dat in `search_dirs`; a path: as given; None / "": None."""
    if not spec:
        return None
    if spec != "auto":
        return spec if os.path.isfile(spec) else None
    for d in search_dirs:
        p = os.path.join(d, "site.dat")
        if os.path.isfile(p):
            return p
    return None


def spread_colocated(info: Dict[int, dict], id_to_site: Dict[int, int],
                     sites: Sequence[Dict[str, Any]], tol: float) -> int:
    """
    Several site ids at exactly one position (in the results file) were all given
    the same nearest site.dat entry. If site.dat has several entries within `tol`
    of that one (two stations at one place), the ids, in ascending order, get
    these entries in order of their sitenum, skipping entries that other ids use;
    ids left over stay on the nearest entry. Changes `id_to_site` in place and
    returns the number of ids moved.
    """
    groups: Dict[Tuple[float, float], List[int]] = {}
    for sid_, p in info.items():
        if sid_ in id_to_site:
            groups.setdefault((round(p["x"], 2), round(p["y"], 2)), []).append(sid_)
    moved = 0
    for ids in groups.values():
        ids = sorted(ids)
        j0 = id_to_site[ids[0]]
        if len(ids) < 2 or any(id_to_site[i] != j0 for i in ids):
            continue
        e0, n0 = float(sites[j0]["easting"]), float(sites[j0]["northing"])
        used = {j for i, j in id_to_site.items() if i not in ids}
        cand = sorted((k for k, s_ in enumerate(sites)
                       if k not in used and math.hypot(float(s_["easting"]) - e0,
                                                       float(s_["northing"]) - n0) <= tol),
                      key=lambda k: (int(sites[k]["sitenum"]), k))
        for i, k in zip(ids, cand):
            if id_to_site[i] != k:
                id_to_site[i] = k
                moved += 1
    return moved


def assign_sites(ds: dict, sites: Sequence[Dict[str, Any]], *, match: str = "auto",
                 tol: float = 15.0, offset: int = 0) -> Tuple[np.ndarray, dict]:
    """
    Index of the site.dat site (position in `sites`) for every row of `ds`,
    -1 = not assigned.

    match "coords": by position (femtic_misfit.match_sites_by_coords, using the
    first row of each site id; data types without a station position, HTF and
    NMT, are never assigned); "id": site_id = sitenum + offset; "auto": by
    position, by id only if no row has a position. Several site ids at one
    place (MT and VTF) get the same site, unless site.dat has several entries
    there (two stations at one place): then the ids, in ascending order, get
    these entries in order of their sitenum (spread_colocated).

    Returns (site_index, stats) with stats = n_ids, n_hit, n_sites, n_rows,
    n_unassigned_rows, mode, id_to_site ({site id: index}), unmatched (ids),
    and for "coords" also frame, shift, dmax, n_spread (ids moved by
    spread_colocated).
    """
    sid = np.asarray(ds["site_id"]).astype(np.int64)
    dn = np.asarray(ds["datatype_name"])
    x = np.asarray(ds["site_x"], float)
    y = np.asarray(ds["site_y"], float)
    station = ~np.isin(dn, fm._NO_STATION_POSITION)        # rows that belong to one station
    has_pos = station & np.isfinite(x) & np.isfinite(y)

    ids_u = np.unique(sid[station])
    info: Dict[int, dict] = {}
    if has_pos.any():
        u, first = np.unique(sid[has_pos], return_index=True)
        rows = np.flatnonzero(has_pos)[first]
        info = {int(s): dict(x=float(x[r]), y=float(y[r])) for s, r in zip(u, rows)}

    mode = match
    if mode == "auto":
        mode = "coords" if info else "id"
    stats: Dict[str, Any] = dict(mode=mode, frame="", shift=(float("nan"), float("nan")),
                                 dmax=float("nan"))
    id_to_site: Dict[int, int] = {}
    if mode == "coords":
        sites_xy = [(str(j), int(s["sitenum"]), float(s["easting"]), float(s["northing"]))
                    for j, s in enumerate(sites)]
        names, st = fm.match_sites_by_coords(info, sites_xy, float(tol))
        id_to_site = {int(k): int(v) for k, v in names.items()}
        stats.update(frame=st["frame"], shift=st["shift"], dmax=st["dmax"],
                     n_spread=spread_colocated(info, id_to_site, sites, float(tol)))
    else:
        by_num: Dict[int, int] = {}
        for j, s in enumerate(sites):
            by_num.setdefault(int(s["sitenum"]) + int(offset), j)
        id_to_site = {int(s): by_num[int(s)] for s in ids_u if int(s) in by_num}

    uniq, inv = np.unique(sid, return_inverse=True)
    lut = np.array([id_to_site.get(int(u_), -1) for u_ in uniq], dtype=np.int64)
    site_index = lut[inv]
    site_index[~station] = -1
    stats.update(
        n_ids=int(ids_u.size), n_hit=int(sum(1 for s in ids_u if int(s) in id_to_site)),
        n_sites=len(set(id_to_site.values())), n_rows=int(sid.size),
        n_unassigned_rows=int(np.count_nonzero(site_index < 0)),
        id_to_site=id_to_site,
        unmatched=[int(s) for s in ids_u if int(s) not in id_to_site])
    return site_index, stats


# ---------------------------------------------------------------------------
# nRMS: residuals, selections, reduction
# ---------------------------------------------------------------------------
def row_residuals(ds: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Error-normalized residuals (obs - cal) / err of the real and the imaginary
    parts, one value per row; NaN where not usable (non-finite value, err <= 0,
    and the imaginary part of real-valued data types). The same arithmetic as
    femtic_misfit.site_nrms (calculated response = ref_re / ref_im).
    """
    n = np.asarray(ds["freq"]).size
    cpx = ~np.isin(ds["datatype_name"], fm.REAL_ONLY_TYPES)
    out = []
    for ko, kr, ke, msk in (("obs_re", "ref_re", "err_re", np.ones(n, bool)),
                            ("obs_im", "ref_im", "err_im", cpx)):
        obs, ref, err = (np.asarray(ds[k], float) for k in (ko, kr, ke))
        ok = msk & np.isfinite(err) & (err > 0.0) & np.isfinite(obs) & np.isfinite(ref)
        r = np.full(n, np.nan)
        r[ok] = (obs[ok] - ref[ok]) / err[ok]
        out.append(r)
    return out[0], out[1]


def _num(x: float) -> str:
    """Number for a label: 1000 (not 1e+03), 0.1, 3.16."""
    return f"{int(round(x))}" if x >= 1.0 and abs(x - round(x)) < 1e-9 * x else f"{x:.3g}"


def decade_bands(freq: np.ndarray, *, axis: str = "period", per_decade: int = 1,
                 min_n: int = 1) -> List[Tuple[str, np.ndarray]]:
    """
    [("all bands", mask), (label, mask), ...]: bands of `per_decade` per decade,
    anchored at whole decades, from short to long period (for axis "frequency":
    decreasing frequency); bands with fewer than `min_n` rows are dropped. The
    bands of femtic_misfit.select_bands, with labels like "1 - 10 s".
    """
    freq = np.asarray(freq, float)
    bidx = fm.band_index(freq, per_decade=per_decade, axis=axis)
    unit = "s" if axis == "period" else "Hz"
    out: List[Tuple[str, np.ndarray]] = [("all bands", np.ones(freq.size, bool))]
    ks = np.unique(bidx)
    if axis == "frequency":
        ks = ks[::-1]
    for k in ks:
        m = bidx == k
        if np.count_nonzero(m) >= int(min_n):
            lo, hi = 10.0 ** (int(k) / per_decade), 10.0 ** ((int(k) + 1) / per_decade)
            out.append((f"{_num(lo)} - {_num(hi)} {unit}", m))
    return out


def band_selections(freq: np.ndarray, spec: Union[str, Sequence[Any]], *, axis: str = "period",
                    per_decade: int = 1, min_n: int = 1) -> List[Tuple[str, np.ndarray]]:
    """
    [(label, row mask), ...] of the frequency bands. spec "all"; "decades" (all,
    then the bands of femtic_misfit.select_bands); or a list of "all" and
    (lo, hi) pairs, in s for axis "period", in Hz for "frequency" (lo <= x < hi).
    Bands with fewer than `min_n` rows are dropped ("all" is kept).
    """
    freq = np.asarray(freq, float)
    n = freq.size
    all_sel = ("all bands", np.ones(n, bool))
    if isinstance(spec, str):
        s = spec.strip().lower()
        if s == "all":
            return [all_sel]
        if s in ("decades", "auto"):
            return decade_bands(freq, axis=axis, per_decade=int(per_decade), min_n=int(min_n))
        raise ValueError(f"BANDS = {spec!r}: \"all\", \"decades\" or a list")
    x = 1.0 / freq if axis == "period" else freq
    unit = "s" if axis == "period" else "Hz"
    out: List[Tuple[str, np.ndarray]] = []
    for item in spec:
        if isinstance(item, str):
            if item.strip().lower() != "all":
                raise ValueError(f"BANDS: {item!r} is not \"all\" or a (lo, hi) pair")
            out.append(all_sel)
            continue
        lo, hi = float(item[0]), float(item[1])
        if not (0.0 < lo < hi):
            raise ValueError(f"BANDS: ({lo}, {hi}) needs 0 < lo < hi")
        m = (x >= lo * (1.0 - 1e-9)) & (x < hi * (1.0 - 1e-9))
        label = f"{_num(lo)} - {_num(hi)} {unit}"
        if np.count_nonzero(m) < int(min_n):
            _say(f"band {label}: {int(np.count_nonzero(m))} data rows (BAND_MIN_N = "
                 f"{int(min_n)}), dropped.")
            continue
        out.append((label, m))
    return out


def component_selections(comp_names: np.ndarray, specs: Sequence[str],
                         groups: Dict[str, Sequence[str]]) -> List[Tuple[str, np.ndarray]]:
    """
    [(label, row mask), ...]. A spec is "all", a group of `groups`, a component
    name, or several joined by "+"; names and groups are not case sensitive.
    Selections that match no row are dropped (with a message).
    """
    names = np.char.lower(np.asarray(comp_names, dtype=str))
    glow = {str(k).lower(): [str(w).lower() for w in v] for k, v in groups.items()}
    out: List[Tuple[str, np.ndarray]] = []
    for spec in specs:
        sp = str(spec).strip()
        if sp.lower() == "all":
            out.append(("all", np.ones(names.size, bool)))
            continue
        m = np.zeros(names.size, bool)
        for tok in [t.strip() for t in sp.split("+") if t.strip()]:
            m |= np.isin(names, glow.get(tok.lower(), [tok.lower()]))
        if not m.any():
            _say(f"component selection {sp!r}: no data row, dropped.")
            continue
        out.append((sp, m))
    return out


def nrms_pooled(r_re: np.ndarray, r_im: np.ndarray, mask: np.ndarray) -> Tuple[int, float]:
    """(number of values, nRMS) of the residuals of the rows in `mask`."""
    r = np.concatenate([r_re[mask], r_im[mask]])
    r = r[np.isfinite(r)]
    if r.size == 0:
        return 0, float("nan")
    return int(r.size), float(np.sqrt(np.mean(r * r)))


def nrms_per_site(r_re: np.ndarray, r_im: np.ndarray, mask: np.ndarray,
                  site_index: np.ndarray, n_sites: int) -> Tuple[np.ndarray, np.ndarray]:
    """(number of values, nRMS) per site index; NaN nRMS where a site has no value."""
    rows = mask & (site_index >= 0)
    s2 = np.zeros(n_sites)
    cnt = np.zeros(n_sites, dtype=np.int64)
    for r in (r_re, r_im):
        ok = rows & np.isfinite(r)
        s2 += np.bincount(site_index[ok], weights=r[ok] ** 2, minlength=n_sites)
        cnt += np.bincount(site_index[ok], minlength=n_sites)
    nr = np.full(n_sites, np.nan)
    has = cnt > 0
    nr[has] = np.sqrt(s2[has] / cnt[has])
    return cnt, nr


def compute_nrms(ds: dict, site_index: np.ndarray, n_sites: int,
                 bands: Sequence[Tuple[str, np.ndarray]],
                 comps: Sequence[Tuple[str, np.ndarray]], *,
                 base: Optional[np.ndarray] = None) -> List[dict]:
    """
    One dict per (band, component selection) that has data: ib, ic, band, comp,
    n and nrms (global: all rows of the selection, assigned to a site or not),
    n_site and nrms_site (arrays per site index), n_unassigned (values of rows
    that belong to no site). `base` = rows to use at all (data types).
    """
    r_re, r_im = row_residuals(ds)
    if base is None:
        base = np.ones(r_re.size, bool)
    out: List[dict] = []
    for ib, (bl, bm) in enumerate(bands):
        for ic, (cl, cm) in enumerate(comps):
            m = bm & cm & base
            n_all, nr_all = nrms_pooled(r_re, r_im, m)
            if n_all == 0:
                _say(f"selection {bl} | {cl}: no usable value, skipped.")
                continue
            n_s, nr_s = nrms_per_site(r_re, r_im, m, site_index, n_sites)
            out.append(dict(ib=ib, ic=ic, band=bl, comp=cl, n=n_all, nrms=nr_all,
                            n_site=n_s, nrms_site=nr_s,
                            n_unassigned=int(n_all - int(n_s.sum()))))
    return out


# ---------------------------------------------------------------------------
# Ensemble: members and statistics over the members
# ---------------------------------------------------------------------------
class DataTypeError(ValueError):
    """No row of the requested data types (DATATYPES) in a results file."""


_STAT_WORDS = {"mean": "mean", "median": "median", "min": "minimum", "max": "maximum",
               "std": "std"}


def parse_stat(stat: str) -> Tuple[str, Optional[float]]:
    """
    ENS_STAT -> (kind, q): kind "mean", "median", "min", "max", "std" or
    "percentile" (q = 0..100 for "pNN", e.g. "p90"; else None). ValueError for
    anything else.
    """
    s = str(stat).strip().lower()
    if s in _STAT_WORDS:
        return s, None
    m = re.fullmatch(r"p(\d+(?:\.\d+)?)", s)
    if m and 0.0 <= float(m.group(1)) <= 100.0:
        return "percentile", float(m.group(1))
    raise ValueError(f"ENS_STAT = {stat!r}: must be one of mean, median, min, max, std, or "
                     "pNN (a percentile, e.g. p90)")


def stat_words(stat: str) -> str:
    """The statistic in words: "median", "std", "percentile 90", ..."""
    kind, q = parse_stat(stat)
    return f"percentile {q:g}" if kind == "percentile" else _STAT_WORDS[kind]


def ens_statistic(x: Any, stat: str, axis: int = 0) -> np.ndarray:
    """
    Statistic of `x` over `axis` (the members), ignoring NaN; NaN where there is
    no value (std: fewer than two; it is the sample std, ddof = 1).
    """
    kind, q = parse_stat(stat)
    a = np.asarray(x, float)
    a = np.where(np.isfinite(a), a, np.nan)
    cnt = np.sum(np.isfinite(a), axis=axis)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if kind == "mean":
            out = np.nanmean(a, axis=axis)
        elif kind == "median":
            out = np.nanmedian(a, axis=axis)
        elif kind == "min":
            out = np.nanmin(a, axis=axis)
        elif kind == "max":
            out = np.nanmax(a, axis=axis)
        elif kind == "std":
            out = np.nanstd(a, axis=axis, ddof=1)
        else:
            out = np.nanpercentile(a, q, axis=axis)
    return np.where(cnt >= (2 if kind == "std" else 1), out, np.nan)


def member_paths(cfg: Dict[str, Any]) -> Tuple[List[str], str]:
    """
    (paths of the members, where they came from): RUN_PATHS (globs expanded); else
    the sub-directories of RESULTS_PATH that contain ITER_PATTERN files, if
    RESULTS_PATH itself contains none; else RESULTS_PATH (one member).
    """
    runs = cfg.get("RUN_PATHS")
    if runs:
        return fm.expand_paths([runs] if isinstance(runs, str) else list(runs)), "RUN_PATHS"
    top, pat = os.path.normpath(cfg["RESULTS_PATH"]), cfg["ITER_PATTERN"]
    if os.path.isdir(top) and not glob.glob(os.path.join(glob.escape(top), pat)):
        subs = [d for d in sorted(glob.glob(os.path.join(glob.escape(top), "*")))
                if os.path.isdir(d) and glob.glob(os.path.join(glob.escape(d), pat))]
        if subs:
            return subs, "sub-directories of RESULTS_PATH"
    return [top], "RESULTS_PATH"


def member_names(paths: Sequence[str]) -> List[str]:
    """Short unique names: the directory name (of the file, for a results file); repeats get #2, #3, ..."""
    out: List[str] = []
    count: Dict[str, int] = {}
    for p in paths:
        q = os.path.normpath(p)
        if os.path.isfile(q):
            q = os.path.dirname(os.path.abspath(q))
        nm = os.path.basename(q) or q
        count[nm] = count.get(nm, 0) + 1
        out.append(nm if count[nm] == 1 else f"{nm}#{count[nm]}")
    return out


def apply_datatypes(ds: dict, datatypes: Optional[Sequence[str]]) -> Tuple[dict, np.ndarray]:
    """
    (dataset, base): the rows of the data types `datatypes` (None = all). DataTypeError
    if there is none.
    """
    base = np.ones(int(np.asarray(ds["freq"]).size), bool)
    if datatypes:
        base = np.isin(ds["datatype_name"], list(datatypes))
        if not base.any():
            raise DataTypeError(f"no row of the data types {list(datatypes)} "
                                f"(found: {sorted(set(map(str, ds['datatype_name'])))}).")
        ds = {k: (v[base] if isinstance(v, np.ndarray) and v.shape[:1] == base.shape else v)
              for k, v in ds.items() if k not in ("cal_re_m", "cal_im_m")}
        base = np.ones(int(np.asarray(ds["freq"]).size), bool)
    return ds, base


def load_member(path: str, cfg: Dict[str, Any]) -> dict:
    """
    Read one member: dict(res, ds, base, n_rows) with the resolve_run result `res`
    (without its data), the dataset of the member (own observed data) restricted
    to DATATYPES, and the number of rows of the file. OSError, ValueError, KeyError
    (DataTypeError) if it cannot be read.
    """
    res = fm.resolve_run(path, iteration=cfg["ITERATION"], pattern=cfg["ITER_PATTERN"])
    ds = fm.build_dataset([res["data"]])
    n_rows = int(np.asarray(ds["freq"]).size)
    res = {k: v for k, v in res.items() if k != "data"}
    ds, base = apply_datatypes(ds, cfg["DATATYPES"])
    return dict(res=res, ds=ds, base=base, n_rows=n_rows)


def member_selections(lm: dict, site_index: np.ndarray, n_sites: int,
                      cfg: Dict[str, Any]) -> List[dict]:
    """compute_nrms of a loaded member for the BANDS and COMPONENTS of `cfg` (ValueError for a bad setting)."""
    ds = lm["ds"]
    bands = band_selections(ds["freq"], cfg["BANDS"], axis=cfg["BAND_AXIS"],
                            per_decade=cfg["BAND_PER_DECADE"], min_n=cfg["BAND_MIN_N"])
    comps = component_selections(ds["component_name"], cfg["COMPONENTS"], cfg["COMPONENT_GROUPS"])
    return compute_nrms(ds, site_index, n_sites, bands, comps, base=lm["base"])


@contextlib.contextmanager
def _print_once(seen: set):
    """Lines printed in the block appear once: not again if they are in `seen` (which is updated)."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            yield
    finally:
        for ln in buf.getvalue().splitlines():
            if ln not in seen:
                seen.add(ln)
                print(ln)


def aggregate_members(members: Sequence[dict], stat: str, n_sites: int) -> List[dict]:
    """
    Selections of an ensemble: one dict per (band, components) found in any member,
    in the keys of compute_nrms. nrms_site = the statistic `stat` over the members
    of their nRMS at the site, nrms = the same for the global nRMS of the members;
    n, n_site, n_unassigned = median of the members (rounded). Further keys: M
    (members with a value), G (global nRMS per member), X (members x sites), stat,
    sub (second line of the panel title). `members`: dicts with "sels".
    """
    bands: List[str] = []
    comps: List[str] = []
    keys: List[Tuple[str, str]] = []
    lookup: List[Dict[Tuple[str, str], dict]] = []
    for m in members:
        d: Dict[Tuple[str, str], dict] = {}
        for s in m["sels"]:
            if s["band"] not in bands:
                bands.append(s["band"])
            if s["comp"] not in comps:
                comps.append(s["comp"])
            if (s["band"], s["comp"]) not in keys:
                keys.append((s["band"], s["comp"]))
            d[(s["band"], s["comp"])] = s
        lookup.append(d)
    keys.sort(key=lambda k: (bands.index(k[0]), comps.index(k[1])))
    kind, _ = parse_stat(stat)
    nm = len(members)
    out: List[dict] = []
    for key in keys:
        X = np.full((nm, n_sites), np.nan)
        N = np.full((nm, n_sites), np.nan)
        G = np.full(nm, np.nan)
        nG = np.full(nm, np.nan)
        nU = np.full(nm, np.nan)
        for k, d in enumerate(lookup):
            s = d.get(key)
            if s is None:
                continue
            X[k] = s["nrms_site"]
            N[k] = np.where(s["n_site"] > 0, s["n_site"], np.nan)
            G[k], nG[k], nU[k] = s["nrms"], s["n"], s["n_unassigned"]
        n_have = int(np.sum(np.isfinite(G)))
        g = float(ens_statistic(G, stat))
        n_med = int(round(float(ens_statistic(nG, "median"))))
        what = "std of global nRMS" if kind == "std" else "global nRMS"
        words = "" if kind == "std" else f"{stat_words(stat)}, "
        out.append(dict(
            ib=bands.index(key[0]), ic=comps.index(key[1]), band=key[0], comp=key[1],
            n=n_med, nrms=g,
            n_site=np.nan_to_num(np.rint(ens_statistic(N, "median"))).astype(np.int64),
            nrms_site=ens_statistic(X, stat),
            n_unassigned=int(round(float(np.nan_to_num(ens_statistic(nU, "median"))))),
            M=n_have, G=G, X=X, stat=stat,
            sub=f"{what} = {g:.2f} ({words}M = {n_have}), n = {n_med}"))
    return out


def iteration_text(iters: Sequence[int]) -> str:
    """"8" for one iteration, "5-8" for a range (smallest-largest)."""
    lo, hi = int(min(iters)), int(max(iters))
    return str(lo) if lo == hi else f"{lo}-{hi}"


# ---------------------------------------------------------------------------
# DEM: tiles, download, mosaic
# ---------------------------------------------------------------------------
def tile_name(lat0: int, lon0: int) -> str:
    """Name of the 1 x 1 degree tile with the lower left corner (lat0, lon0): N37W105."""
    return (f"{'N' if lat0 >= 0 else 'S'}{abs(int(lat0)):02d}"
            f"{'E' if lon0 >= 0 else 'W'}{abs(int(lon0)):03d}")


def tiles_for_extent(lon_min: float, lon_max: float, lat_min: float,
                     lat_max: float) -> List[Tuple[int, int]]:
    """(lat0, lon0) of the lower left corners of the tiles that cover the box."""
    if not (-90.0 <= lat_min <= lat_max <= 90.0 and -180.0 <= lon_min <= lon_max <= 180.0):
        raise ValueError(f"extent lon {lon_min:g}..{lon_max:g}, lat {lat_min:g}..{lat_max:g} "
                         "is outside -180..180, -90..90")
    i0 = int(math.floor(lat_min))
    i1 = max(int(math.floor(lat_max - 1e-9)), i0)
    j0 = int(math.floor(lon_min))
    j1 = max(int(math.floor(lon_max - 1e-9)), j0)
    return [(i, j) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)]


def tile_url(template: str, lat0: int, lon0: int) -> str:
    """URL of a tile; the placeholders of `template` are {name}, {NS}, {lat}, {EW}, {lon}."""
    return template.format(name=tile_name(lat0, lon0), NS="N" if lat0 >= 0 else "S",
                           lat=abs(int(lat0)), EW="E" if lon0 >= 0 else "W",
                           lon=abs(int(lon0)))


def parse_hgt(raw: bytes, label: str = "tile") -> np.ndarray:
    """
    Elevation [m] of an SRTM .hgt tile: 16-bit signed big-endian, n x n samples
    (3601 for 1 arc second, 1201 for 3 arc seconds), the first row at the north
    edge; the void value -32768 becomes NaN. float32 array (n, n).
    """
    if len(raw) % 2:
        raise ValueError(f"{label}: odd number of bytes ({len(raw)}), not a .hgt tile")
    n2 = len(raw) // 2
    n = int(round(math.sqrt(n2)))
    if n < 2 or n * n != n2:
        raise ValueError(f"{label}: {len(raw)} bytes are not a square 16-bit .hgt tile "
                         "(3601 x 3601 or 1201 x 1201 samples expected)")
    z = np.frombuffer(raw, dtype=">i2").reshape(n, n).astype(np.float32)
    z[z == -32768.0] = np.nan
    return z


def read_hgt_file(path: str) -> np.ndarray:
    """Tile from a .hgt, .hgt.gz or .hgt.zip file (the first .hgt in the archive)."""
    low = path.lower()
    if low.endswith(".gz"):
        with gzip.open(path, "rb") as fh:
            raw = fh.read()
    elif low.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            member = [m for m in zf.namelist() if m.lower().endswith(".hgt")]
            if not member:
                raise ValueError(f"{path}: no .hgt file in the archive")
            raw = zf.read(member[0])
    else:
        with open(path, "rb") as fh:
            raw = fh.read()
    return parse_hgt(raw, os.path.basename(path))


def fetch_tile(lat0: int, lon0: int, *, cache_dir: str, template: str, download: bool,
               timeout: float) -> Tuple[Optional[np.ndarray], str]:
    """
    One tile, from `cache_dir` if it is there, else downloaded (and kept in
    `cache_dir`). Returns (elevation or None, status); status is "cache",
    "downloaded", "missing" (HTTP 403 / 404: the server has no such tile, as for
    the ocean), "absent" (not in the cache, download off) or "failed (reason)".
    """
    name = tile_name(lat0, lon0)
    for ext in _HGT_EXTENSIONS:
        p = os.path.join(cache_dir, name + ext)
        if os.path.isfile(p):
            try:
                return read_hgt_file(p), "cache"
            except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
                _say(f"WARNING: cannot read {p} ({exc}); "
                     + ("loading it again." if download else "ignored."))
    if not download:
        return None, "absent"
    url = tile_url(template, lat0, lon0)
    req = urllib.request.Request(url, headers={"User-Agent": "py4mt-femtic_plot_sitemap"})
    try:
        with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 404):
            return None, "missing"
        return None, f"failed (HTTP {exc.code})"
    except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
        return None, f"failed ({exc})"
    gz = raw[:2] == b"\x1f\x8b"
    try:
        z = parse_hgt(gzip.decompress(raw) if gz else raw, name)
    except (OSError, EOFError, ValueError) as exc:
        return None, f"failed (bad data from {url}: {exc})"
    try:
        os.makedirs(cache_dir, exist_ok=True)
        dest = os.path.join(cache_dir, name + (".hgt.gz" if gz else ".hgt"))
        with open(dest + ".part", "wb") as fh:
            fh.write(raw)
        os.replace(dest + ".part", dest)
    except OSError as exc:
        _say(f"WARNING: tile {name} not kept in {cache_dir} ({exc}).")
    return z, "downloaded"


def mosaic_tiles(tiles: Dict[Tuple[int, int], Optional[np.ndarray]], *,
                 fill: Optional[float] = 0.0,
                 sea: Iterable[Tuple[int, int]] = ()) -> Optional[dict]:
    """
    One DEM from tiles {(lat0, lon0): elevation or None}. The tiles share their
    edge row / column with the neighbours (n samples = n - 1 intervals per
    degree); tiles with more samples than the coarsest one are thinned out to
    it. A missing tile in `sea` is set to `fill` (None = NaN), other missing
    tiles are NaN. Returns dict(lon, lat, z): lon increasing, lat decreasing
    (row 0 = north), z[lat, lon] float32 (NaN = void); None without any tile.
    """
    present = {k: v for k, v in tiles.items() if v is not None}
    if not present:
        return None
    nmin = min(v.shape[0] for v in present.values())
    red: Dict[Tuple[int, int], np.ndarray] = {}
    for k, v in present.items():
        if (v.shape[0] - 1) % (nmin - 1):
            raise ValueError(f"tile {tile_name(*k)} has {v.shape[0]} samples, not a multiple of "
                             f"the {nmin} of the coarsest tile")
        s = (v.shape[0] - 1) // (nmin - 1)
        red[k] = v[::s, ::s]
    m = nmin - 1
    lats = [k[0] for k in tiles]
    lons = [k[1] for k in tiles]
    i0, i1, j0, j1 = min(lats), max(lats), min(lons), max(lons)
    ny, nx = (i1 - i0 + 1) * m + 1, (j1 - j0 + 1) * m + 1
    z = np.full((ny, nx), np.nan, dtype=np.float32)
    if fill is not None:
        for k in sea:
            if k in tiles and tiles[k] is None:
                r0, c0 = (i1 - k[0]) * m, (k[1] - j0) * m
                z[r0:r0 + nmin, c0:c0 + nmin] = np.float32(fill)
    for (la, lo), v in red.items():
        r0, c0 = (i1 - la) * m, (lo - j0) * m
        blk = z[r0:r0 + nmin, c0:c0 + nmin]
        z[r0:r0 + nmin, c0:c0 + nmin] = np.where(np.isfinite(v), v, blk)
    lon = j0 + np.arange(nx) / float(m)
    lat = (i1 + 1) - np.arange(ny) / float(m)
    return dict(lon=lon, lat=lat, z=z)


def _tile_report(status: Dict[Tuple[int, int], str], *, fill: Optional[float],
                 strict: bool) -> List[Tuple[int, int]]:
    """
    Messages on the tiles that could not be had (warning, or RuntimeError if `strict`), and
    on the ones that the server does not have; returns these (the open ocean).
    """
    bad = [k for k, st in status.items() if st.startswith("failed") or st == "absent"]
    if bad:
        txt = ("DEM tiles not available: "
               + ", ".join(f"{tile_name(*k)} ({status[k]})" for k in bad))
        if strict:
            raise RuntimeError(txt)
        _say("WARNING: " + txt + " -- no relief there.")
    sea = [k for k, st in status.items() if st == "missing"]
    if sea:
        _say(f"{len(sea)} tile(s) not on the server (ocean?): "
             + (f"set to {fill:g} m." if fill is not None else "no value."))
    return sea


def load_dem_hgt(lon_min: float, lon_max: float, lat_min: float, lat_max: float, *,
                 cache_dir: str, template: str, download: bool = True, timeout: float = 60.0,
                 max_tiles: int = 16, fill: Optional[float] = 0.0,
                 strict: bool = False) -> Optional[dict]:
    """
    DEM of the box from tiles in the .hgt format (see fetch_tile, mosaic_tiles). None if
    no tile could be had. strict: a tile that failed to download (or is absent
    with download off) is an error, else a message and NaN there.
    """
    tiles = tiles_for_extent(lon_min, lon_max, lat_min, lat_max)
    if len(tiles) > int(max_tiles):
        raise ValueError(f"the map needs {len(tiles)} DEM tiles (DEM_MAX_TILES = {int(max_tiles)}); "
                         "make the map smaller or raise DEM_MAX_TILES")
    got: Dict[Tuple[int, int], Optional[np.ndarray]] = {}
    status: Dict[Tuple[int, int], str] = {}
    for la, lo in tiles:
        z, st = fetch_tile(la, lo, cache_dir=cache_dir, template=template,
                           download=download, timeout=timeout)
        got[(la, lo)], status[(la, lo)] = z, st
        _say(f"DEM tile {tile_name(la, lo)}: {st}.")
    sea = _tile_report(status, fill=fill, strict=strict)
    return mosaic_tiles(got, fill=fill, sea=sea)


# --- DEM tiles as GeoTIFF files (rasterio) ---
def is_url(text: str) -> bool:
    """True for an http:// or https:// address (else: a path)."""
    return str(text).lower().startswith(("http://", "https://"))


def dem_template(source: str, template: Optional[str]) -> str:
    """Tile pattern of the DEM source: `template` if given, else that of the preset of `source`."""
    if template:
        return str(template)
    if source in _DEM_PRESETS:
        return _DEM_PRESETS[source]["template"]
    raise ValueError(f"DEM_SOURCE = {source!r} needs DEM_URL_TEMPLATE")


def tile_kind(template: str) -> str:
    """
    "raster" (GeoTIFF tiles, read with rasterio) or "hgt" (.hgt, .hgt.gz, .hgt.zip tiles),
    from the end of the tile name of `template`. ValueError for an unusable pattern
    (a placeholder that does not exist, another file type).
    """
    try:
        name = tile_url(template, 65, -17)
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(f"DEM_URL_TEMPLATE {template!r}: {exc!r}; the placeholders are {{name}}, "
                         "{NS}, {lat}, {EW}, {lon}") from exc
    path = urllib.parse.urlsplit(name).path if is_url(name) else name
    low = path.lower()
    if low.endswith(_RASTER_EXTENSIONS):
        return "raster"
    if low.endswith((".hgt", ".hgt.gz", ".hgt.zip")):
        return "hgt"
    raise ValueError(f"DEM_URL_TEMPLATE {template!r}: the tiles have to be GeoTIFF (.tif, .tiff) "
                     "or .hgt (.hgt, .hgt.gz, .hgt.zip) files")


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def fetch_raster_tile(lat0: int, lon0: int, *, cache_dir: str, template: str, download: bool,
                      timeout: float, refresh: bool = False) -> Tuple[Optional[str], str]:
    """
    The file of one raster tile. A path pattern is used as it is; a URL is looked up in
    `cache_dir` (the file name is the last part of the URL), else downloaded into it.
    Returns (path or None, status); status is "local", "cache", "downloaded", "missing"
    (HTTP 403 / 404: the server has no such tile, as for the open ocean), "absent" (not in
    the cache and download off, or a local file that is not there) or "failed (reason)".
    `refresh`: ignore a file in the cache and load it again.
    """
    url = tile_url(template, lat0, lon0)
    if not is_url(url):
        path = os.path.expanduser(url)
        return (path, "local") if os.path.isfile(path) else (None, "absent")
    dest = os.path.join(cache_dir, os.path.basename(urllib.parse.urlsplit(url).path))
    if not refresh and os.path.isfile(dest):
        return dest, "cache"
    if not download:
        return None, "absent"
    req = urllib.request.Request(url, headers={"User-Agent": "py4mt-femtic_plot_sitemap"})
    try:
        os.makedirs(cache_dir, exist_ok=True)
        with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
            with open(dest + ".part", "wb") as fh:
                shutil.copyfileobj(resp, fh, 1 << 20)
        os.replace(dest + ".part", dest)
    except urllib.error.HTTPError as exc:
        _remove(dest + ".part")
        return None, ("missing" if exc.code in (403, 404) else f"failed (HTTP {exc.code})")
    except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
        _remove(dest + ".part")
        return None, f"failed ({exc})"
    return dest, "downloaded"


def _raster_errors() -> tuple:
    """The exceptions of reading a damaged or unusable raster file."""
    errs: tuple = (OSError, ValueError, RuntimeError)
    try:
        from rasterio.errors import RasterioError
        errs += (RasterioError,)
    except ImportError:
        pass
    return errs


def read_raster_window(path: str, lon_min: float, lon_max: float, lat_min: float,
                       lat_max: float, *, max_pixels: Optional[int] = None) -> Optional[dict]:
    """
    The part of a GeoTIFF that covers the box (plus a margin of two output pixels), read with
    rasterio: dict(lon, lat, z) as mosaic_tiles (pixel centres; lon increasing, lat
    decreasing; NaN = no data). The raster has to be north-up; a raster in a projected
    coordinate system is warped to lon / lat (WGS 84) first. With `max_pixels` the window is
    averaged down to at most about that many pixels per side. None if the raster does not
    cover the box. ImportError without rasterio.
    """
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.windows import Window
    except ImportError as exc:
        raise ImportError("reading a GeoTIFF needs rasterio (not installed): install it (conda "
                          "install -c conda-forge rasterio), or use DEM_SOURCE = \"skadi\" or an "
                          ".npz file") from exc
    with contextlib.ExitStack() as stack:
        ds = stack.enter_context(rasterio.open(path))
        if ds.crs is not None and not ds.crs.is_geographic:
            from rasterio.vrt import WarpedVRT
            ds = stack.enter_context(WarpedVRT(ds, crs="EPSG:4326", resampling=Resampling.bilinear))
        tr = ds.transform
        if tr.b != 0.0 or tr.d != 0.0 or not (tr.a > 0.0 > tr.e):
            raise ValueError(f"{path}: the raster is rotated or not north-up")
        width, height = int(ds.width), int(ds.height)

        def span(lo: float, hi: float, origin: float, step: float, n: int) -> Tuple[int, int]:
            a, b = sorted(((lo - origin) / step, (hi - origin) / step))
            return max(int(math.floor(a)), 0), min(int(math.ceil(b)), n)

        c0, c1 = span(lon_min, lon_max, tr.c, tr.a, width)
        r0, r1 = span(lat_max, lat_min, tr.f, tr.e, height)
        if c1 - c0 < 1 or r1 - r0 < 1:
            return None
        step = 1
        if max_pixels:
            step = max(1, int(math.ceil(max(c1 - c0, r1 - r0) / float(max_pixels))))
        c0, c1 = max(c0 - 2 * step, 0), min(c1 + 2 * step, width)
        r0, r1 = max(r0 - 2 * step, 0), min(r1 + 2 * step, height)
        ncol, nrow = c1 - c0, r1 - r0
        out_w, out_h = int(math.ceil(ncol / step)), int(math.ceil(nrow / step))
        if out_w < 2 or out_h < 2:
            return None
        z = np.array(ds.read(1, window=Window(c0, r0, ncol, nrow), out_shape=(out_h, out_w),
                             resampling=Resampling.average if step > 1 else Resampling.nearest),
                     dtype=np.float32)
        nodata = ds.nodata
        if nodata is not None and np.isfinite(nodata):
            z[z == np.float32(nodata)] = np.nan
    dx, dy = tr.a * ncol / out_w, tr.e * nrow / out_h
    lon = tr.c + c0 * tr.a + dx * (np.arange(out_w) + 0.5)
    lat = tr.f + r0 * tr.e + dy * (np.arange(out_h) + 0.5)
    return dict(lon=lon, lat=lat, z=z)


def merge_dems(parts: Sequence[dict], *, fill_boxes: Sequence[Tuple[float, float, float, float]] = (),
               fill: Optional[float] = 0.0) -> Optional[dict]:
    """
    One DEM from windows dict(lon, lat, z) of the same area (pixel centres, regular spacing;
    the first valid value wins where they overlap). The grid has the coarsest spacing of the
    windows (they are sampled by the nearest pixel). `fill_boxes` (lon_min, lon_max, lat_min,
    lat_max) are given the value `fill` where the windows have no value (None = NaN); they
    widen the grid in whole pixels, so that the pixels of the windows stay where they are.
    None without a window.
    """
    if not parts:
        return None
    ext = [dem_extent(p) for p in parts]
    step = [((p["lon"][-1] - p["lon"][0]) / (p["lon"].size - 1),
             (p["lat"][0] - p["lat"][-1]) / (p["lat"].size - 1)) for p in parts]
    dlon = max(st[0] for st in step)
    dlat = max(st[1] for st in step)
    boxes = [tuple(float(v) for v in b) for b in fill_boxes]
    west, east = min(e[0] for e in ext), max(e[1] for e in ext)
    south, north = min(e[2] for e in ext), max(e[3] for e in ext)
    for b in boxes:                  # a box widens the grid in whole pixels (the pixels of the windows stay)
        west -= dlon * max(0, math.ceil((west - b[0]) / dlon - 1e-6))
        east += dlon * max(0, math.ceil((b[1] - east) / dlon - 1e-6))
        south -= dlat * max(0, math.ceil((south - b[2]) / dlat - 1e-6))
        north += dlat * max(0, math.ceil((b[3] - north) / dlat - 1e-6))
    nx = max(2, int(round((east - west) / dlon)))
    ny = max(2, int(round((north - south) / dlat)))
    lon = west + dlon * (np.arange(nx) + 0.5)
    lat = north - dlat * (np.arange(ny) + 0.5)
    z = np.full((ny, nx), np.nan, dtype=np.float32)
    for p, (w, e, s_, n), (px, py) in zip(parts, ext, step):
        jj = np.nonzero((lon >= w) & (lon < e))[0]
        ii = np.nonzero((lat > s_) & (lat <= n))[0]
        if jj.size == 0 or ii.size == 0:
            continue
        sj = np.clip(np.floor((lon[jj] - w) / px).astype(int), 0, p["lon"].size - 1)
        si = np.clip(np.floor((n - lat[ii]) / py).astype(int), 0, p["lat"].size - 1)
        cur = z[np.ix_(ii, jj)]
        z[np.ix_(ii, jj)] = np.where(np.isfinite(cur), cur, p["z"][np.ix_(si, sj)])
    if fill is not None:
        for w, e, s_, n in boxes:
            jj = np.nonzero((lon >= w) & (lon < e))[0]
            ii = np.nonzero((lat > s_) & (lat <= n))[0]
            if jj.size and ii.size:
                cur = z[np.ix_(ii, jj)]
                z[np.ix_(ii, jj)] = np.where(np.isfinite(cur), cur, np.float32(fill))
    return dict(lon=lon, lat=lat, z=z)


def load_dem_raster(lon_min: float, lon_max: float, lat_min: float, lat_max: float, *,
                    cache_dir: str, template: str, download: bool = True, timeout: float = 60.0,
                    max_tiles: int = 16, fill: Optional[float] = 0.0, strict: bool = False,
                    max_pixels: Optional[int] = None) -> Optional[dict]:
    """
    DEM of the box from GeoTIFF tiles (see fetch_raster_tile, read_raster_window,
    merge_dems); a tile that the server does not have (the ocean) is set to `fill`. None if
    no tile could be had. strict: a tile that failed to download or to read (or is absent) is
    an error, else a message and NaN there.
    """
    tiles = tiles_for_extent(lon_min, lon_max, lat_min, lat_max)
    if len(tiles) > int(max_tiles):
        raise ValueError(f"the map needs {len(tiles)} DEM tiles (DEM_MAX_TILES = {int(max_tiles)}); "
                         "make the map smaller or raise DEM_MAX_TILES")
    errs = _raster_errors()
    parts: List[dict] = []
    status: Dict[Tuple[int, int], str] = {}
    for la, lo in tiles:
        kw = dict(cache_dir=cache_dir, template=template, download=download, timeout=timeout)
        path, st = fetch_raster_tile(la, lo, **kw)
        retry = path is not None and st == "cache" and download
        while path is not None:
            try:
                win = read_raster_window(path, lon_min, lon_max, lat_min, lat_max,
                                         max_pixels=max_pixels)
            except errs as exc:
                if retry:
                    _say(f"WARNING: cannot read {path} ({exc}); loading it again.")
                    retry = False
                    path, st = fetch_raster_tile(la, lo, refresh=True, **kw)
                    continue
                path, st = None, f"failed (cannot read {path}: {exc})"
                break
            if win is not None:
                parts.append(win)
            break
        status[(la, lo)] = st
        _say(f"DEM tile {tile_name(la, lo)}: {st}.")
    sea = _tile_report(status, fill=fill, strict=strict)
    boxes = [(lo, lo + 1.0, la, la + 1.0) for la, lo in sea]
    if not parts:
        return None
    if boxes:                          # no larger than the map and the margin of the windows (< 3 pixels)
        pad_x = 3.0 * max((p["lon"][-1] - p["lon"][0]) / (p["lon"].size - 1) for p in parts)
        pad_y = 3.0 * max((p["lat"][0] - p["lat"][-1]) / (p["lat"].size - 1) for p in parts)
        boxes = [(max(b[0], lon_min - pad_x), min(b[1], lon_max + pad_x),
                  max(b[2], lat_min - pad_y), min(b[3], lat_max + pad_y)) for b in boxes]
        boxes = [b for b in boxes if b[0] < b[1] and b[2] < b[3]]
    return merge_dems(parts, fill_boxes=boxes, fill=fill)


def _tile_corner(path: str) -> Tuple[int, int]:
    """(lat0, lon0) from a file name like N37W105.hgt.gz."""
    m = re.match(r"([NS])(\d{2})([EW])(\d{3})", os.path.basename(path), re.IGNORECASE)
    if not m:
        raise ValueError(f"{path}: the name has to start with the tile name (N37W105...)")
    lat0 = int(m.group(2)) * (1 if m.group(1).upper() == "N" else -1)
    lon0 = int(m.group(4)) * (1 if m.group(3).upper() == "E" else -1)
    return lat0, lon0


def _orient(lon: np.ndarray, lat: np.ndarray, z: np.ndarray) -> dict:
    """lon increasing, lat decreasing (row 0 = north)."""
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    z = np.asarray(z, dtype=np.float32)
    if z.shape != (lat.size, lon.size):
        raise ValueError(f"DEM: z has the shape {z.shape}, lat / lon give {(lat.size, lon.size)}")
    if lon[0] > lon[-1]:
        lon, z = lon[::-1], z[:, ::-1]
    if lat[0] < lat[-1]:
        lat, z = lat[::-1], z[::-1, :]
    return dict(lon=lon, lat=lat, z=np.ascontiguousarray(z))


def load_dem_file(path: str, box: Optional[Tuple[float, float, float, float]] = None,
                  max_pixels: Optional[int] = None) -> dict:
    """
    DEM from a file: .npz (arrays lon, lat, z with z[lat, lon]), GeoTIFF (.tif /
    .tiff; needs rasterio; only the part inside `box` = (lon_min, lon_max, lat_min, lat_max)
    is read, averaged down to about `max_pixels` per side; see read_raster_window) or one
    .hgt tile (.hgt, .hgt.gz, .hgt.zip, named like N37W105...). Returns dict(lon, lat, z)
    as mosaic_tiles does.
    """
    low = path.lower()
    if low.endswith(".npz"):
        with np.load(path) as d:
            return _orient(d["lon"], d["lat"], d["z"])
    if low.endswith(_RASTER_EXTENSIONS):
        b = (-180.0, 180.0, -90.0, 90.0) if box is None else box
        dem = read_raster_window(path, b[0], b[1], b[2], b[3], max_pixels=max_pixels)
        if dem is None:
            raise ValueError(f"{path}: the raster does not cover the map")
        return dem
    if low.endswith((".hgt", ".hgt.gz", ".hgt.zip")):
        la, lo = _tile_corner(path)
        dem = mosaic_tiles({(la, lo): read_hgt_file(path)})
        assert dem is not None
        return dem
    raise ValueError(f"{path}: DEM file type not known (.npz, .tif, .tiff, .hgt, .hgt.gz, .hgt.zip)")


def crop_dem(dem: dict, lon_min: float, lon_max: float, lat_min: float, lat_max: float) -> dict:
    """The part of `dem` that covers the box (one more pixel at each side)."""
    lon, lat, z = dem["lon"], dem["lat"], dem["z"]
    j0 = max(int(np.searchsorted(lon, lon_min, side="left")) - 1, 0)
    j1 = min(int(np.searchsorted(lon, lon_max, side="right")) + 1, lon.size)
    i0 = max(int(np.searchsorted(-lat, -lat_max, side="left")) - 1, 0)
    i1 = min(int(np.searchsorted(-lat, -lat_min, side="right")) + 1, lat.size)
    if i1 - i0 < 2 or j1 - j0 < 2:
        raise ValueError("the DEM does not cover the map "
                         f"(lon {lon_min:g}..{lon_max:g}, lat {lat_min:g}..{lat_max:g})")
    return dict(lon=lon[j0:j1], lat=lat[i0:i1], z=z[i0:i1, j0:j1])


def decimate_dem(dem: dict, max_pixels: int) -> dict:
    """Every s-th pixel, s the smallest integer that gives at most `max_pixels` per side."""
    ny, nx = dem["z"].shape
    s = max(1, int(math.ceil(max(ny, nx) / float(max_pixels))))
    if s == 1:
        return dem
    return dict(lon=dem["lon"][::s], lat=dem["lat"][::s], z=dem["z"][::s, ::s])


def dem_extent(dem: dict) -> Tuple[float, float, float, float]:
    """(lon_west, lon_east, lat_south, lat_north) of the pixel edges."""
    lon, lat = dem["lon"], dem["lat"]
    dlon = (lon[-1] - lon[0]) / (lon.size - 1)
    dlat = (lat[0] - lat[-1]) / (lat.size - 1)
    return (float(lon[0] - dlon / 2), float(lon[-1] + dlon / 2),
            float(lat[-1] - dlat / 2), float(lat[0] + dlat / 2))


def shade_dem(dem: dict, *, style: str = "hillshade", azdeg: float = 315.0,
              altdeg: float = 45.0, vert_exag: float = 1.0,
              cmap: str = "gist_earth") -> dict:
    """
    Shaded relief of `dem` (row 0 = north) for imshow: dict(kind, img) with
    kind "gray" (img: intensity 0..1, NaN = no value) or "rgba" (style
    "terrain": the elevation in colour `cmap`, shaded). The pixel size in m is
    taken from the pixel spacing in degrees at the mean latitude.
    """
    import matplotlib
    from matplotlib.colors import LightSource
    z = np.asarray(dem["z"], float)
    fin = np.isfinite(z)
    zf = np.where(fin, z, float(np.mean(z[fin])) if fin.any() else 0.0)
    lon, lat = dem["lon"], dem["lat"]
    dlon = abs((lon[-1] - lon[0]) / (lon.size - 1))
    dlat = abs((lat[0] - lat[-1]) / (lat.size - 1))
    lat0 = float(np.mean(lat))
    dx = _EARTH_RADIUS_M * math.radians(dlon) * math.cos(math.radians(lat0))
    dy = _EARTH_RADIUS_M * math.radians(dlat)
    ls = LightSource(azdeg=float(azdeg), altdeg=float(altdeg))
    if style == "hillshade":
        img = ls.hillshade(zf, vert_exag=float(vert_exag), dx=dx, dy=dy)
        return dict(kind="gray", img=np.where(fin, img, np.nan))
    if style == "terrain":
        vmin = float(np.min(z[fin])) if fin.any() else 0.0
        vmax = float(np.max(z[fin])) if fin.any() else 1.0
        if vmax <= vmin:
            vmax = vmin + 1.0
        rgba = ls.shade(zf, cmap=matplotlib.colormaps[cmap], blend_mode="soft",
                        vert_exag=float(vert_exag), dx=dx, dy=dy, vmin=vmin, vmax=vmax)
        rgba[~fin, 3] = 0.0
        return dict(kind="rgba", img=rgba)
    raise ValueError(f"DEM_STYLE = {style!r}: \"hillshade\" or \"terrain\"")


# ---------------------------------------------------------------------------
# Map extent and coordinates
# ---------------------------------------------------------------------------
def map_extent(lon: np.ndarray, lat: np.ndarray, *, mode: str = "sites",
               box: Optional[Sequence[float]] = None,
               center: Optional[Sequence[float]] = None,
               size_km: Sequence[float] = (50.0, 50.0), margin_frac: float = 0.15,
               margin_min_km: float = 2.0) -> Tuple[float, float, float, float]:
    """
    (lon_min, lon_max, lat_min, lat_max) of the map. mode "sites": the box of the
    sites (`lon`, `lat`) plus a margin of `margin_frac` of its size on each side,
    at least `margin_min_km`; "box": `box` as given; "center": `size_km` (width,
    height) around `center` = (lon, lat).
    """
    if mode == "box":
        if box is None or len(box) != 4:
            raise ValueError("MAP_BOX = (lon_min, lon_max, lat_min, lat_max) needed")
        return float(box[0]), float(box[1]), float(box[2]), float(box[3])
    if mode == "center":
        if center is None or len(center) != 2:
            raise ValueError("MAP_CENTER = (lon, lat) needed")
        lon_c, lat_c = float(center[0]), float(center[1])
        kx = _KM_PER_DEG * math.cos(math.radians(lat_c))
        hw, hh = 0.5 * float(size_km[0]) / kx, 0.5 * float(size_km[1]) / _KM_PER_DEG
        return lon_c - hw, lon_c + hw, lat_c - hh, lat_c + hh
    if mode != "sites":
        raise ValueError(f"MAP_EXTENT_MODE = {mode!r}: \"sites\", \"box\" or \"center\"")
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    if lon.size == 0:
        raise ValueError("no site to take the map extent from")
    lat_c = float(np.mean(lat))
    kx = _KM_PER_DEG * math.cos(math.radians(lat_c))
    span_lon, span_lat = float(lon.max() - lon.min()), float(lat.max() - lat.min())
    mlon = max(float(margin_frac) * span_lon, float(margin_min_km) / kx)
    mlat = max(float(margin_frac) * span_lat, float(margin_min_km) / _KM_PER_DEG)
    return (float(lon.min()) - mlon, float(lon.max()) + mlon,
            float(lat.min()) - mlat, float(lat.max()) + mlat)


def make_frame(display: str, lon0: float, lat0: float) -> dict:
    """Plot coordinates: "latlon" (degrees) or "km" (from (lon0, lat0), local scale)."""
    if display not in ("latlon", "km"):
        raise ValueError(f"DISPLAY_COORDS = {display!r}: \"latlon\" or \"km\"")
    c = math.cos(math.radians(lat0))
    return dict(display=display, lon0=float(lon0), lat0=float(lat0),
                kx=_KM_PER_DEG * c, ky=_KM_PER_DEG,
                aspect=(1.0 / c if display == "latlon" else 1.0))


def to_frame(frame: dict, lon: Any, lat: Any) -> Tuple[np.ndarray, np.ndarray]:
    """(x, y) plot coordinates of (lon, lat) in degrees."""
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    if frame["display"] == "latlon":
        return lon, lat
    return (lon - frame["lon0"]) * frame["kx"], (lat - frame["lat0"]) * frame["ky"]


# ---------------------------------------------------------------------------
# Overlays: profile lines and other polylines
# ---------------------------------------------------------------------------
class OverlayError(ValueError):
    """A profile or line that cannot be used (file, coordinates, geometry)."""


_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_UTM_K0 = 0.9996
_UTM_TOL_M = 25.0                  # site.dat easting / northing must agree with lat / lon to this (median)
_COORDS = ("lonlat", "latlon", "utm", "model")
_PROFILE_KEYS = ("name", "points", "file", "start", "end", "center", "azimuth", "length_km",
                 "coords", "width_km", "ticks_km", "tick_labels", "labels", "label", "color",
                 "lw", "ls")
_LINE_KEYS = ("label", "file", "points", "segments", "coords", "closed", "fill", "alpha",
              "color", "lw", "ls", "halo")
_LEGEND_LOCS = (None, "best", "upper right", "upper left", "lower left", "lower right", "right",
                "center left", "center right", "lower center", "upper center", "center")
_MAX_TICKS = 2000
_PROFILE_TICK_FRAC = 0.012         # length of a tick: this fraction of the width of the map
_END_TOL_KM = 0.01                 # a site this close beyond the end of a profile still counts (10 m)


def _krueger(n: float) -> Tuple[tuple, tuple, tuple]:
    """Series coefficients (alpha, beta, delta) of the Krueger transverse Mercator, to n**4."""
    a = (n / 2 - 2 * n**2 / 3 + 5 * n**3 / 16 + 41 * n**4 / 180,
         13 * n**2 / 48 - 3 * n**3 / 5 + 557 * n**4 / 1440,
         61 * n**3 / 240 - 103 * n**4 / 140,
         49561 * n**4 / 161280)
    b = (n / 2 - 2 * n**2 / 3 + 37 * n**3 / 96 - n**4 / 360,
         n**2 / 48 + n**3 / 15 - 437 * n**4 / 1440,
         17 * n**3 / 480 - 37 * n**4 / 840,
         4397 * n**4 / 161280)
    d = (2 * n - 2 * n**2 / 3 - 2 * n**3 + 116 * n**4 / 45,
         7 * n**2 / 3 - 8 * n**3 / 5 - 227 * n**4 / 45,
         56 * n**3 / 15 - 136 * n**4 / 35,
         4279 * n**4 / 630)
    return a, b, d


def utm_forward(lat: Any, lon: Any, zone: int, north: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    UTM (easting, northing) in m of (lat, lon) in degrees, WGS84, for `zone`
    (central meridian 6 * zone - 183 deg); `north` False = southern hemisphere
    (false northing 10000 km). Kruger series to 4th order (sub-millimetre inside
    the zone and some degrees beyond), no external library.
    """
    n = _WGS84_F / (2.0 - _WGS84_F)
    big_a = _WGS84_A / (1.0 + n) * (1.0 + n**2 / 4.0 + n**4 / 64.0)
    al = _krueger(n)[0]
    e = 2.0 * math.sqrt(n) / (1.0 + n)
    phi = np.radians(np.asarray(lat, float))
    lam = np.radians(np.asarray(lon, float) - (6.0 * int(zone) - 183.0))
    t = np.sinh(np.arctanh(np.sin(phi)) - e * np.arctanh(e * np.sin(phi)))
    xi0 = np.arctan2(t, np.cos(lam))
    eta0 = np.arcsinh(np.sin(lam) / np.hypot(t, np.cos(lam)))
    xi, eta = xi0.copy(), eta0.copy()
    for j, aj in enumerate(al, start=1):
        xi = xi + aj * np.sin(2 * j * xi0) * np.cosh(2 * j * eta0)
        eta = eta + aj * np.cos(2 * j * xi0) * np.sinh(2 * j * eta0)
    return (500000.0 + _UTM_K0 * big_a * eta,
            (0.0 if north else 1.0e7) + _UTM_K0 * big_a * xi)


def utm_inverse(east: Any, north_m: Any, zone: int, north: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """(lon, lat) in degrees of the UTM (easting, northing) in m; inverse of utm_forward."""
    n = _WGS84_F / (2.0 - _WGS84_F)
    big_a = _WGS84_A / (1.0 + n) * (1.0 + n**2 / 4.0 + n**4 / 64.0)
    _, be, de = _krueger(n)
    xi0 = (np.asarray(north_m, float) - (0.0 if north else 1.0e7)) / (_UTM_K0 * big_a)
    eta0 = (np.asarray(east, float) - 500000.0) / (_UTM_K0 * big_a)
    xi, eta = xi0.copy(), eta0.copy()
    for j, bj in enumerate(be, start=1):
        xi = xi - bj * np.sin(2 * j * xi0) * np.cosh(2 * j * eta0)
        eta = eta - bj * np.cos(2 * j * xi0) * np.sinh(2 * j * eta0)
    chi = np.arcsin(np.sin(xi) / np.cosh(eta))
    phi = chi.copy()
    for j, dj in enumerate(de, start=1):
        phi = phi + dj * np.sin(2 * j * chi)
    lam = np.arctan2(np.sinh(eta), np.cos(xi))
    return np.degrees(lam) + (6.0 * int(zone) - 183.0), np.degrees(phi)


def utm_zone_of(lon: float) -> int:
    """UTM zone (1 .. 60) of a longitude in degrees (no Norway / Svalbard exceptions)."""
    return int(math.floor((float(lon) + 180.0) / 6.0)) % 60 + 1


def make_geo(sites: Sequence[Dict[str, Any]], st: Optional[dict] = None) -> dict:
    """
    Georeferencing of the overlays: dict(zone, north, resid, frame, shift).
    zone / north = the UTM zone and hemisphere in which the easting / northing
    of the site.dat sites are the UTM coordinates of their lat / lon (the one
    with the smallest median distance, `resid` in m), zone None if no zone fits
    within 25 m (then "utm" and "model" coordinates cannot be used). frame /
    shift = those of the assignment of the site ids by position (assign_sites
    stats `st`: "x=north, y=east" or "x=east, y=north", and the shift (sx, sy)
    in m that turns model coordinates into UTM), "" and NaN otherwise.
    """
    geo: Dict[str, Any] = dict(zone=None, north=True, resid=float("nan"), frame="",
                               shift=(float("nan"), float("nan")))
    if st:
        sh = st.get("shift", geo["shift"])
        geo["frame"] = str(st.get("frame", "") or "")
        geo["shift"] = (float(sh[0]), float(sh[1]))
    if not sites:
        return geo
    lat = np.array([s["lat"] for s in sites], float)
    lon = np.array([s["lon"] for s in sites], float)
    east = np.array([s["easting"] for s in sites], float)
    north = np.array([s["northing"] for s in sites], float)
    ok = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(east) & np.isfinite(north)
    if not ok.any():
        return geo
    lat, lon, east, north = lat[ok], lon[ok], east[ok], north[ok]
    zones = set()
    for lo in (lon.min(), lon.max(), lon.mean()):
        z0 = utm_zone_of(lo)
        for dz in (-1, 0, 1):
            zones.add((z0 - 1 + dz) % 60 + 1)
    best = None
    for z in sorted(zones):
        for nh in (True, False):
            e, n = utm_forward(lat, lon, z, nh)
            r = float(np.median(np.hypot(e - east, n - north)))
            if best is None or r < best[0]:
                best = (r, z, nh)
    geo["resid"] = best[0]
    if best[0] <= _UTM_TOL_M:
        geo["zone"], geo["north"] = best[1], best[2]
    return geo


def to_lonlat(x: Any, y: Any, coords: str, geo: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    (lon, lat) in degrees of the coordinates (x, y) given in `coords`: "lonlat"
    (lon, lat), "latlon" (lat, lon), "utm" (easting, northing in m, zone of
    site.dat) or "model" (x, y in m of the FEMTIC model, as site_x, site_y of
    the results file). OverlayError if the coordinates cannot be converted or
    are not finite or not on the globe.
    """
    if coords not in _COORDS:
        raise OverlayError(f"coords = {coords!r}: one of {list(_COORDS)}")
    x = np.atleast_1d(np.asarray(x, float))
    y = np.atleast_1d(np.asarray(y, float))
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise OverlayError("coordinates are not finite")
    if coords == "lonlat":
        lon, lat = x, y
    elif coords == "latlon":
        lon, lat = y, x
    else:
        if geo.get("zone") is None:
            res = geo.get("resid", float("nan"))
            raise OverlayError(
                f"coords = {coords!r} needs the UTM easting / northing of site.dat; they do not "
                "agree with its lat / lon in any zone"
                + (f" (median distance {res:.0f} m)" if np.isfinite(res) else ""))
        if coords == "utm":
            east, north = x, y
        else:
            sx, sy = geo.get("shift", (float("nan"), float("nan")))
            fr = geo.get("frame", "")
            if not (np.isfinite(sx) and np.isfinite(sy)) or fr not in ("x=north, y=east",
                                                                       "x=east, y=north"):
                raise OverlayError("coords = \"model\" needs the sites assigned by position "
                                   "(SITE_MATCH = \"coords\" or \"auto\" with site coordinates)")
            if fr == "x=north, y=east":
                north, east = x + sx, y + sy
            else:
                east, north = x + sx, y + sy
        lon, lat = utm_inverse(east, north, geo["zone"], geo["north"])
    if np.any(np.abs(lat) > 90.0) or np.any(np.abs(lon) > 180.0):
        raise OverlayError(f"lon / lat out of range with coords = {coords!r} "
                           "(lonlat and latlon mixed up?)")
    return lon, lat


def local_km(lon: Any, lat: Any, lon0: float, lat0: float) -> Tuple[np.ndarray, np.ndarray]:
    """(east, north) in km of (lon, lat) from (lon0, lat0), flat earth at the latitude lat0."""
    kx = _KM_PER_DEG * math.cos(math.radians(lat0))
    return ((np.asarray(lon, float) - lon0) * kx, (np.asarray(lat, float) - lat0) * _KM_PER_DEG)


def local_lonlat(x: Any, y: Any, lon0: float, lat0: float) -> Tuple[np.ndarray, np.ndarray]:
    """(lon, lat) of the local (east, north) in km from (lon0, lat0); inverse of local_km."""
    kx = _KM_PER_DEG * math.cos(math.radians(lat0))
    return lon0 + np.asarray(x, float) / kx, lat0 + np.asarray(y, float) / _KM_PER_DEG


# --- readers: text (GMT multi-segment), GeoJSON, KML / KMZ ---
def read_text_polylines(path: str) -> List[dict]:
    """
    Polylines of a text file: two numbers per line (the first two columns; comma,
    semicolon or white space), segments separated by a line that starts with ">"
    (GMT multi-segment) or by a blank line; "#" and "%" start a comment, lines
    that are no numbers are skipped. Returns [dict(xy=(N, 2) array, closed)];
    closed if the first and the last point are equal (N >= 4). Segments of
    fewer than 2 points are dropped.
    """
    parts: List[dict] = []
    cur: List[Tuple[float, float]] = []

    def flush() -> None:
        if len(cur) >= 2:
            xy = np.array(cur, float)
            parts.append(dict(xy=xy, closed=bool(len(cur) >= 4 and np.array_equal(xy[0], xy[-1]))))
        cur.clear()

    with open(path, "r", errors="replace") as fh:
        for ln in fh:
            s = ln.strip()
            if not s or s.startswith(">"):
                flush()
                continue
            if s[0] in "#%":
                continue
            tok = s.replace(",", " ").replace(";", " ").split()
            try:
                a, b = float(tok[0]), float(tok[1])
            except (ValueError, IndexError):
                continue
            if math.isfinite(a) and math.isfinite(b):
                cur.append((a, b))
    flush()
    return parts


def read_geojson_polylines(path: str) -> List[dict]:
    """
    Lines and polygon rings of a GeoJSON file (FeatureCollection, Feature, a
    geometry or a GeometryCollection): LineString, MultiLineString, Polygon,
    MultiPolygon; points are ignored. Coordinates are lon, lat. Returns
    [dict(xy, closed)] as read_text_polylines (rings are closed).
    """
    import json
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        obj = json.load(fh)
    parts: List[dict] = []

    def add(coords: Any, closed: bool) -> None:
        try:
            xy = np.array([[float(c[0]), float(c[1])] for c in coords], float)
        except (TypeError, ValueError, IndexError):
            return
        if xy.shape[0] >= 2 and np.all(np.isfinite(xy)):
            parts.append(dict(xy=xy, closed=closed))

    def walk(g: Any) -> None:
        if not isinstance(g, dict):
            return
        t = g.get("type")
        if t == "FeatureCollection":
            for f in g.get("features") or []:
                walk(f)
        elif t == "Feature":
            walk(g.get("geometry"))
        elif t == "GeometryCollection":
            for h in g.get("geometries") or []:
                walk(h)
        elif t == "LineString":
            add(g.get("coordinates") or [], False)
        elif t == "MultiLineString":
            for c in g.get("coordinates") or []:
                add(c, False)
        elif t == "Polygon":
            for r in g.get("coordinates") or []:
                add(r, True)
        elif t == "MultiPolygon":
            for p in g.get("coordinates") or []:
                for r in p:
                    add(r, True)

    walk(obj)
    return parts


def read_kml_polylines(path: str) -> List[dict]:
    """
    LineString and LinearRing (polygon boundaries; closed) elements of a KML or
    KMZ file (the first .kml of the archive, doc.kml preferred). Coordinates are
    lon, lat[, alt]. Returns [dict(xy, closed)] as read_text_polylines.
    """
    import xml.etree.ElementTree as ET
    try:
        if path.lower().endswith(".kmz"):
            with zipfile.ZipFile(path) as zf:
                kml = [n for n in zf.namelist() if n.lower().endswith(".kml")]
                if not kml:
                    raise ValueError("no .kml in the archive")
                data = zf.read("doc.kml" if "doc.kml" in kml else kml[0])
            root = ET.fromstring(data)
        else:
            root = ET.parse(path).getroot()
    except (ET.ParseError, zipfile.BadZipFile) as exc:
        raise ValueError(f"not a readable KML / KMZ file: {exc}") from exc
    parts: List[dict] = []
    for el in root.iter():
        tag = el.tag.rsplit("}", 1)[-1] if isinstance(el.tag, str) else ""
        if tag not in ("LineString", "LinearRing"):
            continue
        for c in el:
            if isinstance(c.tag, str) and c.tag.rsplit("}", 1)[-1] == "coordinates" and c.text:
                pts = []
                for tok in c.text.split():
                    v = tok.split(",")
                    try:
                        pts.append((float(v[0]), float(v[1])))
                    except (ValueError, IndexError):
                        continue
                xy = np.array(pts, float)
                if xy.shape[0] >= 2 and np.all(np.isfinite(xy)):
                    parts.append(dict(xy=xy, closed=(tag == "LinearRing")))
    return parts


def read_geometry_file(path: str) -> Tuple[List[dict], bool]:
    """
    (parts, geographic) of a polyline file by its extension: .geojson / .json,
    .kml / .kmz (coordinates always lon, lat: geographic = True), anything else is
    text (GMT multi-segment; the coordinates are those of `coords`). OSError and
    ValueError for a file that cannot be read.
    """
    low = str(path).lower()
    if low.endswith((".geojson", ".json")):
        return read_geojson_polylines(path), True
    if low.endswith((".kml", ".kmz")):
        return read_kml_polylines(path), True
    return read_text_polylines(path), False


# --- profiles and lines ---
def _as_points(pts: Any, what: str) -> np.ndarray:
    """(N, 2) float array of a sequence of pairs; OverlayError if it is none."""
    try:
        a = np.array([[float(p[0]), float(p[1])] for p in pts], float)
    except (TypeError, ValueError, IndexError) as exc:
        raise OverlayError(f"{what}: a sequence of (a, b) pairs is needed") from exc
    if a.ndim != 2 or a.shape[1] != 2:
        raise OverlayError(f"{what}: a sequence of (a, b) pairs is needed")
    return a


def _read_parts(spec: dict, geo: dict) -> List[dict]:
    """Parts [dict(lon, lat, closed)] of the file of `spec`; coords as given, or lon / lat."""
    path = str(spec["file"])
    try:
        raw, geographic = read_geometry_file(path)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise OverlayError(f"{path}: {exc}") from exc
    if not raw:
        raise OverlayError(f"{path}: no polyline found")
    coords = "lonlat" if geographic else spec.get("coords", "lonlat")
    out = []
    for p in raw:
        try:
            lon, lat = to_lonlat(p["xy"][:, 0], p["xy"][:, 1], coords, geo)
        except OverlayError as exc:
            raise OverlayError(f"{path}: {exc}") from exc
        out.append(dict(lon=lon, lat=lat, closed=bool(p["closed"])))
    return out


def _end_point(p: Any, what: str) -> Tuple[Tuple[float, float], Optional[str]]:
    """((a, b), label) of an end point of a profile: (a, b), or (a, b, "A") with the label of that
    end (a third entry that is no text is ignored). OverlayError for anything else."""
    try:
        label = None
        if len(p) >= 3 and isinstance(p[2], str) and p[2].strip():
            label = p[2].strip()
        return (float(p[0]), float(p[1])), label
    except (TypeError, ValueError, IndexError) as exc:
        raise OverlayError(f"{what}: (a, b) or (a, b, \"label\") is needed") from exc


def default_end_labels(name: str) -> Tuple[str, str]:
    """End labels of a profile from its name: "A-A'" -> ("A", "A'"), "P1" -> ("P1", "P1'")."""
    m = re.match(r"^\s*([^-]+?)\s*-\s*([^-]+?)\s*$", str(name))
    return (m.group(1), m.group(2)) if m else (str(name), str(name) + "'")


def prepare_profile(spec: dict, geo: dict, index: int = 0) -> dict:
    """
    One profile from its spec (see PROFILES): the vertices in lon / lat and in
    km (flat earth around their mean position, origin lon0 / lat0), cumulative
    distance cum, length, the swath polygons (lon / lat) if width_km > 0, and
    the style entries of the spec. OverlayError if the geometry cannot be made.
    """
    name = str(spec.get("name") or f"P{index + 1}")
    coords = spec.get("coords", "lonlat")
    end_labels: Tuple[Optional[str], Optional[str]] = (None, None)
    try:
        if "points" in spec:
            a = _as_points(spec["points"], "points")
            lon, lat = to_lonlat(a[:, 0], a[:, 1], coords, geo)
            source = "points"
        elif "file" in spec and spec["file"]:
            parts = _read_parts(spec, geo)
            if len(parts) > 1:
                _say(f"WARNING: profile {name!r}: {spec['file']} holds {len(parts)} polylines; "
                     "the first is used.")
            lon, lat = parts[0]["lon"], parts[0]["lat"]
            source = str(spec["file"])
        elif "start" in spec:
            (p0, l0), (p1, l1) = _end_point(spec["start"], "start"), _end_point(spec["end"], "end")
            a = _as_points([p0, p1], "start / end")
            end_labels = (l0, l1)
            lon, lat = to_lonlat(a[:, 0], a[:, 1], coords, geo)
            source = "start, end"
        else:
            c = _as_points([spec["center"]], "center")
            c_lon, c_lat = to_lonlat(c[:, 0], c[:, 1], coords, geo)
            half = 0.5 * float(spec["length_km"])
            az = math.radians(float(spec["azimuth"]))
            kx = _KM_PER_DEG * math.cos(math.radians(float(c_lat[0])))
            lon = float(c_lon[0]) + np.array([-half, half]) * math.sin(az) / kx
            lat = float(c_lat[0]) + np.array([-half, half]) * math.cos(az) / _KM_PER_DEG
            source = "center, azimuth, length_km"
    except KeyError as exc:
        raise OverlayError(f"missing {exc}") from exc
    lon0, lat0 = float(np.mean(lon)), float(np.mean(lat))
    x, y = local_km(lon, lat, lon0, lat0)
    keep = np.ones(x.size, bool)                        # no zero-length segments
    keep[1:] = np.hypot(np.diff(x), np.diff(y)) > 1e-6
    lon, lat, x, y = lon[keep], lat[keep], x[keep], y[keep]
    if x.size < 2:
        raise OverlayError("a profile needs two different points")
    seg = np.hypot(np.diff(x), np.diff(y))
    cum = np.r_[0.0, np.cumsum(seg)]
    width = float(spec.get("width_km") or 0.0)
    ticks = spec.get("ticks_km")
    ticks = float(ticks) if ticks else None
    if ticks is not None and cum[-1] / ticks > _MAX_TICKS:
        raise OverlayError(f"ticks_km = {ticks:g} gives more than {_MAX_TICKS} ticks on "
                           f"{cum[-1]:.1f} km")
    lab = spec.get("labels")
    if lab is False:
        labels = None
    else:                                  # a label in start / end, else labels, else from the name
        base = default_end_labels(name) if lab is None else (str(lab[0]), str(lab[1]))
        labels = (end_labels[0] or base[0], end_labels[1] or base[1])
    prof = dict(kind="profile", name=name, source=source, labels=labels, lon=lon, lat=lat,
                x=x, y=y, lon0=lon0, lat0=lat0, cum=cum, length=float(cum[-1]), width=width,
                ticks_km=ticks, tick_labels=bool(spec.get("tick_labels", True)),
                color=spec.get("color"), lw=spec.get("lw"), ls=spec.get("ls", "-"),
                label=spec.get("label"))
    prof["swath"] = profile_swath(prof) if width > 0 else []
    return prof


def profile_swath(prof: dict, n_circle: int = 24) -> List[np.ndarray]:
    """
    The swath of a profile as (N, 2) lon / lat polygons that all have the same
    orientation: one rectangle of the width per segment, and a disc at every
    inner vertex (rounded corners); together they are the set of points within
    width / 2 of the profile, without end caps.
    """
    w2 = 0.5 * float(prof["width"])
    x, y = prof["x"], prof["y"]
    polys: List[np.ndarray] = []
    for i in range(x.size - 1):
        d = np.array([x[i + 1] - x[i], y[i + 1] - y[i]])
        nrm = np.array([-d[1], d[0]]) / np.hypot(*d) * w2
        a, b = np.array([x[i], y[i]]), np.array([x[i + 1], y[i + 1]])
        polys.append(np.array([a - nrm, b - nrm, b + nrm, a + nrm]))
    ang = np.linspace(0.0, 2.0 * np.pi, n_circle, endpoint=False)
    circ = np.c_[np.cos(ang), np.sin(ang)] * w2
    for i in range(1, x.size - 1):
        polys.append(circ + np.array([x[i], y[i]]))
    out = []
    for p in polys:
        lo, la = local_lonlat(p[:, 0], p[:, 1], prof["lon0"], prof["lat0"])
        out.append(np.c_[lo, la])
    return out


def project_on_profile(prof: dict, lon: Any, lat: Any) -> dict:
    """
    Position of points relative to a profile: along (km from the start, along the
    polyline), offset (km, signed distance to the polyline, positive to the right
    of the direction start -> end), dist (|offset|) and inside (within width / 2
    of the polyline, and not beyond the start or the end of it; all False
    without a width). Flat earth around the profile (local_km).
    """
    x, y = local_km(lon, lat, prof["lon0"], prof["lat0"])
    px, py = prof["x"], prof["y"]
    ax_, ay_ = px[:-1, None], py[:-1, None]
    dx, dy = (px[1:] - px[:-1])[:, None], (py[1:] - py[:-1])[:, None]
    l2 = dx * dx + dy * dy
    vx, vy = x[None, :] - ax_, y[None, :] - ay_
    t_raw = (vx * dx + vy * dy) / l2
    t = np.clip(t_raw, 0.0, 1.0)
    dist = np.hypot(vx - t * dx, vy - t * dy)
    k = np.argmin(dist, axis=0)
    cols = np.arange(x.size)
    dist_k, t_k, traw_k = dist[k, cols], t[k, cols], t_raw[k, cols]
    seg_len = np.sqrt(l2[:, 0])
    along = prof["cum"][k] + t_k * seg_len[k]
    cross = dx[k, 0] * vy[k, cols] - dy[k, 0] * vx[k, cols]
    offset = np.where(cross > 0.0, -dist_k, dist_k)
    if prof["width"] > 0:
        nseg = px.size - 1
        past = (traw_k - np.clip(traw_k, 0.0, 1.0)) * seg_len[k]       # km beyond the segment
        beyond = ((k == 0) & (past < -_END_TOL_KM)) | ((k == nseg - 1) & (past > _END_TOL_KM))
        inside = (dist_k <= 0.5 * prof["width"] * (1.0 + 1e-9)) & ~beyond
    else:
        inside = np.zeros(x.size, bool)
    return dict(along=along, offset=offset, dist=dist_k, inside=inside)


def project_profiles(profiles: Sequence[dict], lon: np.ndarray, lat: np.ndarray) -> List[dict]:
    """
    The profiles with the projection of the sites (lon, lat): proj (project_on_profile),
    order (indices of the sites inside the swath, by increasing along) and ring (their
    lon, lat).
    """
    out = []
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    for p in profiles:
        pr = project_on_profile(p, lon, lat)
        idx = np.flatnonzero(pr["inside"])
        idx = idx[np.argsort(pr["along"][idx], kind="stable")]
        out.append(dict(p, proj=pr, order=idx, ring=(lon[idx], lat[idx])))
    return out


def profile_tick_points(prof: dict, spacing: float) -> dict:
    """
    Positions s = 0, spacing, 2 * spacing, ... <= length on a profile: s (km), x, y (km,
    local frame of the profile) and the unit direction ux, uy of the segment.
    """
    s = np.minimum(np.arange(int(math.floor(prof["length"] / float(spacing) + 1e-9)) + 1)
                   * float(spacing), prof["length"])
    nseg = prof["x"].size - 1
    i = np.clip(np.searchsorted(prof["cum"], s, side="right") - 1, 0, nseg - 1)
    dx = prof["x"][i + 1] - prof["x"][i]
    dy = prof["y"][i + 1] - prof["y"][i]
    ln = np.hypot(dx, dy)
    ux, uy = dx / ln, dy / ln
    t = np.minimum(s - prof["cum"][i], ln)
    return dict(s=s, x=prof["x"][i] + t * ux, y=prof["y"][i] + t * uy, ux=ux, uy=uy)


def prepare_line(spec: dict, geo: dict) -> dict:
    """
    One line (polylines and polygons) from its spec (see LINES): parts =
    [dict(lon, lat, closed)] and the style entries. OverlayError if the geometry
    cannot be made.
    """
    coords = spec.get("coords", "lonlat")
    if spec.get("file"):
        parts = _read_parts(spec, geo)
    else:
        if "points" in spec:
            groups = [spec["points"]]
        else:
            groups = list(spec["segments"])
        parts = []
        for g in groups:
            a = _as_points(g, "points")
            if a.shape[0] < 2:
                raise OverlayError("a polyline needs two points")
            lon, lat = to_lonlat(a[:, 0], a[:, 1], coords, geo)
            parts.append(dict(lon=lon, lat=lat, closed=False))
    if spec.get("closed") is not None:
        for p in parts:
            p["closed"] = bool(spec["closed"])
    return dict(kind="line", label=spec.get("label"), parts=parts, color=spec.get("color"),
                lw=spec.get("lw"), ls=spec.get("ls", "-"), fill=spec.get("fill"),
                alpha=spec.get("alpha"), halo=spec.get("halo"),
                n_vertices=int(sum(p["lon"].size for p in parts)))


def prepare_overlays(cfg: Dict[str, Any], geo: dict) -> Tuple[List[dict], List[dict]]:
    """
    (profiles, lines) of the settings PROFILES and LINES. An item that cannot be made
    is skipped with a warning (OverlayError for the first one if OVERLAY_STRICT).
    """
    strict = bool(cfg.get("OVERLAY_STRICT"))
    profiles: List[dict] = []
    lines: List[dict] = []
    for i, spec in enumerate(cfg.get("PROFILES") or []):
        what = f"profile {str(spec.get('name') or 'P' + str(i + 1))!r}"
        try:
            profiles.append(prepare_profile(spec, geo, i))
        except OverlayError as exc:
            if strict:
                raise OverlayError(f"{what}: {exc}") from exc
            _say(f"WARNING: {what} skipped: {exc}")
    for i, spec in enumerate(cfg.get("LINES") or []):
        what = f"line {str(spec.get('label') or spec.get('file') or '#' + str(i + 1))!r}"
        try:
            lines.append(prepare_line(spec, geo))
        except OverlayError as exc:
            if strict:
                raise OverlayError(f"{what}: {exc}") from exc
            _say(f"WARNING: {what} skipped: {exc}")
    return profiles, lines


def check_overlay_specs(cfg: Dict[str, Any]) -> None:
    """ValueError for PROFILES / LINES entries that cannot work (keys, options, numbers)."""
    def number(spec: dict, key: str, what: str, sign: str = "pos") -> None:
        v = spec.get(key)
        if v is None:
            return
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"{what}: {key} = {v!r} is not a number") from None
        need = {"pos": "> 0", "nonneg": ">= 0", "any": "finite"}[sign]
        bad = (not math.isfinite(f) or (sign == "pos" and f <= 0.0)
               or (sign == "nonneg" and f < 0.0))
        if bad:
            raise ValueError(f"{what}: {key} = {v!r} must be {need}")

    for key, allowed in (("PROFILES", _PROFILE_KEYS), ("LINES", _LINE_KEYS)):
        specs = cfg.get(key) or []
        if not isinstance(specs, (list, tuple)):
            raise ValueError(f"{key} must be a list of dicts")
        for i, spec in enumerate(specs):
            what = f"{key}[{i}]"
            if not isinstance(spec, dict):
                raise ValueError(f"{what} must be a dict")
            bad = [k for k in spec if k not in allowed]
            if bad:
                raise ValueError(f"{what}: unknown key {bad[0]!r}; allowed: {list(allowed)}")
            if spec.get("coords", "lonlat") not in _COORDS:
                raise ValueError(f"{what}: coords = {spec.get('coords')!r} must be one of "
                                 f"{list(_COORDS)}")
            number(spec, "lw", what)
            if key == "PROFILES":
                modes = [m for m, on in (
                    ("points", "points" in spec),
                    ("file", bool(spec.get("file"))),
                    ("start + end", "start" in spec or "end" in spec),
                    ("center + azimuth + length_km",
                     any(k in spec for k in ("center", "azimuth", "length_km")))) if on]
                if len(modes) != 1:
                    raise ValueError(f"{what}: exactly one of points, file, start + end, center + "
                                     f"azimuth + length_km is needed (found {modes or 'none'})")
                if modes[0] == "start + end" and not ("start" in spec and "end" in spec):
                    raise ValueError(f"{what}: start and end are both needed")
                if modes[0].startswith("center") and not all(
                        k in spec for k in ("center", "azimuth", "length_km")):
                    raise ValueError(f"{what}: center, azimuth and length_km are all needed")
                number(spec, "length_km", what)
                number(spec, "azimuth", what, "any")
                number(spec, "width_km", what, "nonneg")
                number(spec, "ticks_km", what)
                lab = spec.get("labels")
                if lab is not None and lab is not False and not (
                        isinstance(lab, (list, tuple)) and len(lab) == 2):
                    raise ValueError(f"{what}: labels must be False or two texts (start, end)")
            else:
                modes = [m for m in ("file", "points", "segments") if spec.get(m)]
                if len(modes) != 1:
                    raise ValueError(f"{what}: exactly one of file, points, segments is needed "
                                     f"(found {modes or 'none'})")
                number(spec, "alpha", what, "nonneg")
                if spec.get("alpha") is not None and float(spec["alpha"]) > 1.0:
                    raise ValueError(f"{what}: alpha = {spec['alpha']!r} must be <= 1")


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
def make_norm(values: np.ndarray, *, vmin: Optional[float], vmax: Optional[float],
              reference: Optional[float]):
    """
    (norm, extend) of the nRMS colour scale. With a `reference`, a two-slope scale
    centred on it (limits from the data, symmetric around the reference but not
    below 0, unless given); without, a linear one. `extend` = "min", "max",
    "both" or "neither" for values beyond the limits (colour bar arrows).
    """
    from matplotlib.colors import Normalize, TwoSlopeNorm
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    vlo = float(v.min()) if v.size else 0.0
    vhi = float(v.max()) if v.size else 1.0
    if reference is None:
        lo = vlo if vmin is None else float(vmin)
        hi = vhi if vmax is None else float(vmax)
        if not hi > lo:
            hi = lo + 1.0
        norm = Normalize(vmin=lo, vmax=hi)
    else:
        ref = float(reference)
        spread = max(abs(vhi - ref), abs(ref - vlo), 0.1)
        lo = max(ref - spread, 0.0) if vmin is None else float(vmin)
        hi = ref + spread if vmax is None else float(vmax)
        if lo < ref < hi:
            norm = TwoSlopeNorm(vcenter=ref, vmin=lo, vmax=hi)
        else:
            _say(f"WARNING: NRMS_REFERENCE = {ref:g} is not between the colour limits "
                 f"{lo:g} and {hi:g}; linear colour scale.")
            if not hi > lo:
                hi = lo + 1.0
            norm = Normalize(vmin=lo, vmax=hi)
    below = bool(v.size and vlo < lo)
    above = bool(v.size and vhi > hi)
    extend = "both" if below and above else "min" if below else "max" if above else "neither"
    return norm, extend


def _nice_length(target: float) -> float:
    """Largest of 1, 2, 5 x 10^k that is <= target."""
    if not target > 0:
        return 1.0
    mag = 10.0 ** math.floor(math.log10(target))
    for f in (5.0, 2.0, 1.0):
        if f * mag <= target * (1.0 + 1e-12):
            return f * mag
    return mag


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9.+-]+", "_", str(s)).strip("_") or "x"


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def _overlay_style(item: dict, cfg: Dict[str, Any]) -> Tuple[str, float]:
    """(colour, line width) of a profile or line: its own entries, else the settings."""
    if item["kind"] == "profile":
        return (item.get("color") or cfg.get("PROFILE_COLOR", "black"),
                float(item.get("lw") or cfg.get("PROFILE_LW", 1.8)))
    return (item.get("color") or cfg.get("LINE_COLOR", "tab:green"),
            float(item.get("lw") or cfg.get("LINE_LW", 1.2)))


def clip_polyline(x: Any, y: Any, box: Sequence[float], cum: Optional[Any] = None) -> List[dict]:
    """
    The parts of the polyline (x, y) inside the rectangle box = (xa, xb, ya, yb)
    (edges included), as a list of dict(x, y, s0, s1): the vertices of the part
    and the distance along the polyline (in the units of x and y, or `cum`, the
    cumulative lengths of the vertices) at its first and last point. A polyline
    that leaves and enters again gives several parts; a part that touches the
    rectangle in a point is dropped. Every segment is clipped with the Liang-
    Barsky algorithm, the clipped segments that follow each other are joined.
    """
    xa, xb, ya, yb = (float(v) for v in box)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    cum = (np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))] if cum is None
           else np.asarray(cum, float))
    done: List[dict] = []
    cur: Optional[List[Tuple[float, float, float]]] = None

    def close() -> None:
        if cur is not None and len(cur) >= 2 and cur[-1][2] - cur[0][2] > 1e-9:
            a = np.array(cur)
            done.append(dict(x=a[:, 0], y=a[:, 1], s0=float(a[0, 2]), s1=float(a[-1, 2])))

    for i in range(x.size - 1):
        x0, y0, dx, dy = x[i], y[i], x[i + 1] - x[i], y[i + 1] - y[i]
        t0, t1, ok = 0.0, 1.0, True
        for p, q in ((-dx, x0 - xa), (dx, xb - x0), (-dy, y0 - ya), (dy, yb - y0)):
            if p == 0.0:
                if q < 0.0:
                    ok = False
                    break
                continue
            r = q / p
            if p < 0.0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
            if t0 > t1:
                ok = False
                break
        if not ok:
            close()
            cur = None
            continue
        ln = cum[i + 1] - cum[i]
        a = (x0 + t0 * dx, y0 + t0 * dy, cum[i] + t0 * ln)
        b = (x0 + t1 * dx, y0 + t1 * dy, cum[i] + t1 * ln)
        if cur is None or t0 != 0.0:                      # a new part (else: goes on from the vertex)
            close()
            cur = [a]
        if abs(b[2] - cur[-1][2]) > 1e-12:                # no repeated points
            cur.append(b)
        if t1 < 1.0:                                      # leaves the rectangle in this segment
            close()
            cur = None
    close()
    return done


def clip_polygon(x: Any, y: Any, box: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """The polygon (x, y) cut at the rectangle box = (xa, xb, ya, yb) (Sutherland-Hodgman);
    empty arrays if nothing is left."""
    xa, xb, ya, yb = (float(v) for v in box)
    pts = list(zip(np.asarray(x, float), np.asarray(y, float)))
    edges = ((lambda p: p[0] >= xa, lambda a, b: (xa, a[1] + (b[1] - a[1]) * (xa - a[0]) / (b[0] - a[0]))),
             (lambda p: p[0] <= xb, lambda a, b: (xb, a[1] + (b[1] - a[1]) * (xb - a[0]) / (b[0] - a[0]))),
             (lambda p: p[1] >= ya, lambda a, b: (a[0] + (b[0] - a[0]) * (ya - a[1]) / (b[1] - a[1]), ya)),
             (lambda p: p[1] <= yb, lambda a, b: (a[0] + (b[0] - a[0]) * (yb - a[1]) / (b[1] - a[1]), yb)))
    for inside, cross in edges:
        if not pts:
            break
        out = []
        prev = pts[-1]
        for cur in pts:
            if inside(cur):
                if not inside(prev):
                    out.append(cross(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(cross(prev, cur))
            prev = cur
        pts = out
    if len(pts) < 3:
        return np.empty(0), np.empty(0)
    a = np.array(pts, float)
    return a[:, 0], a[:, 1]


def cut_box_lonlat(bbox: Sequence[float], margin_km: float) -> Tuple[float, float, float, float]:
    """(lon_min, lon_max, lat_min, lat_max) of the map `bbox` made smaller by `margin_km` on every
    side. OverlayError if the margin leaves nothing."""
    dlat = float(margin_km) / _KM_PER_DEG
    dlon = float(margin_km) / (_KM_PER_DEG * math.cos(math.radians(0.5 * (bbox[2] + bbox[3]))))
    box = (bbox[0] + dlon, bbox[1] - dlon, bbox[2] + dlat, bbox[3] - dlat)
    if not (box[0] < box[1] and box[2] < box[3]):
        raise OverlayError(f"OVERLAY_CUT_MARGIN_KM = {margin_km:g} km is larger than the map allows")
    return box


def cut_profile(prof: dict, bbox: Optional[Sequence[float]], margin_km: Optional[float]) -> dict:
    """
    The profile `prof` cut at the map `bbox` (lon_min, lon_max, lat_min, lat_max) made smaller by
    `margin_km` on every side: a copy with `pieces` = [dict(x, y, lon, lat, s0, s1)] (the parts
    inside, x, y in km as prof["x"], s0 / s1 the distance from the start of the profile at their
    ends, so the distances of the ticks stay those of the whole profile), `cut` (True if
    anything was cut off) and, if it was cut, the swath of the parts (flat at the cut).
    Without `margin_km` or `bbox` the profile is not cut. OverlayError if the margin leaves
    no map.
    """
    whole = dict(x=prof["x"], y=prof["y"], lon=prof["lon"], lat=prof["lat"], s0=0.0,
                 s1=prof["length"])
    if margin_km is None or bbox is None:
        return dict(prof, pieces=[whole], cut=False)
    lo0, lo1, la0, la1 = cut_box_lonlat(bbox, margin_km)
    xa, ya = local_km(lo0, la0, prof["lon0"], prof["lat0"])
    xb, yb = local_km(lo1, la1, prof["lon0"], prof["lat0"])
    pieces = clip_polyline(prof["x"], prof["y"], (float(xa), float(xb), float(ya), float(yb)),
                           cum=prof["cum"])
    for pc in pieces:
        pc["lon"], pc["lat"] = local_lonlat(pc["x"], pc["y"], prof["lon0"], prof["lat0"])
    if (len(pieces) == 1 and pieces[0]["s0"] <= 1e-9
            and pieces[0]["s1"] >= prof["length"] - 1e-9):
        return dict(prof, pieces=[whole], cut=False)
    out = dict(prof, pieces=pieces, cut=True)
    if prof["width"] > 0:
        out["swath"] = [poly for pc in pieces for poly in profile_swath(
            dict(x=pc["x"], y=pc["y"], width=prof["width"], lon0=prof["lon0"], lat0=prof["lat0"]))]
    return out


def cut_line(ln: dict, bbox: Optional[Sequence[float]], margin_km: Optional[float]) -> dict:
    """
    The line `ln` (polylines and polygons) cut at the map `bbox` made smaller by `margin_km`: a
    copy with `drawn` = [dict(lon, lat, closed, fill = (lon, lat) of the polygon cut at the box
    or None)], the polylines to draw. A closed part is cut as its outline, its fill as a polygon.
    Without `margin_km` or `bbox` the parts are returned as they are.
    """
    if margin_km is None or bbox is None:
        return dict(ln, drawn=[dict(lon=p["lon"], lat=p["lat"], closed=p["closed"], fill=None,
                                    cut=False) for p in ln["parts"]])
    box = cut_box_lonlat(bbox, margin_km)
    rect = (box[0], box[1], box[2], box[3])
    drawn: List[dict] = []
    for part in ln["parts"]:
        lon, lat = np.asarray(part["lon"], float), np.asarray(part["lat"], float)
        if part["closed"]:
            lon, lat = np.r_[lon, lon[0]], np.r_[lat, lat[0]]
        pieces = clip_polyline(lon, lat, rect)
        inside = (len(pieces) == 1 and pieces[0]["x"].size == lon.size
                  and np.allclose(pieces[0]["x"], lon) and np.allclose(pieces[0]["y"], lat))
        fill = None
        if ln.get("fill") and part["closed"]:
            fx, fy = clip_polygon(part["lon"], part["lat"], rect)
            fill = (fx, fy)
        if inside:
            drawn.append(dict(lon=part["lon"], lat=part["lat"], closed=part["closed"], fill=fill,
                              cut=False))
            continue
        for k, pc in enumerate(pieces):
            drawn.append(dict(lon=pc["x"], lat=pc["y"], closed=False,
                              fill=fill if k == 0 else None, cut=True))
        if not pieces and fill is not None and fill[0].size:
            drawn.append(dict(lon=np.empty(0), lat=np.empty(0), closed=False, fill=fill, cut=True))
    return dict(ln, drawn=drawn)


def cut_report(profiles: Sequence[dict], lines: Sequence[dict], bbox: Sequence[float],
               margin_km: Optional[float]) -> List[str]:
    """Messages about what the cut at the map box takes off: one per profile (the km kept of
    its length) and per line that is cut."""
    if margin_km is None:
        return []
    out: List[str] = []
    for pr in profiles:
        c = cut_profile(pr, bbox, margin_km)
        if c["cut"]:
            kept = sum(pc["s1"] - pc["s0"] for pc in c["pieces"])
            out.append(f"profile {pr['name']!r} is cut at the map box (margin {float(margin_km):g} km): "
                       f"{kept:.1f} of {pr['length']:.1f} km are shown.")
    for ln in lines:
        c = cut_line(ln, bbox, margin_km)
        if any(d["cut"] for d in c["drawn"]):
            out.append(f"line {str(ln.get('label') or 'line')!r} is cut at the map box "
                       f"(margin {float(margin_km):g} km).")
    return out


def _cut_margin(cfg: Dict[str, Any], bbox: Optional[Sequence[float]]) -> Optional[float]:
    """OVERLAY_CUT_MARGIN_KM if the map box can have it (else a message, said once, and None)."""
    m = cfg.get("OVERLAY_CUT_MARGIN_KM")
    if m is None or bbox is None:
        return None
    try:
        cut_box_lonlat(bbox, m)
    except OverlayError as exc:
        _say_once(f"WARNING: {exc}: profiles and lines are not cut.")
        return None
    return float(m)


def draw_overlays(ax, *, frame: dict, overlays: Optional[dict], cfg: Dict[str, Any],
                  bbox: Optional[Sequence[float]] = None) -> None:
    """
    Draws the lines and the profiles on the map `ax` (limits already set): lines
    (and polygon fills) below the sites, profiles with swath, ticks, distance
    labels, end labels and rings around the sites in the swath. `overlays` =
    dict(profiles=[...], lines=[...]) as made by prepare_overlays / project_profiles.
    With `bbox` (the map box, lon / lat) and OVERLAY_CUT_MARGIN_KM they are cut at the
    box made smaller by the margin (cut_profile, cut_line): the end labels stand at the
    ends of the cut profile, the ticks keep the distances of the whole profile.
    """
    if not overlays:
        return
    from matplotlib import patheffects
    from matplotlib.collections import LineCollection
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path
    x0, x1 = ax.get_xlim()
    kx_plot = frame["kx"] if frame["display"] == "latlon" else 1.0
    map_km = abs(x1 - x0) * kx_plot                       # width of the map [km]
    margin = _cut_margin(cfg, bbox)

    for ln0 in overlays.get("lines") or []:
        ln = cut_line(ln0, bbox, margin)
        color, lw = _overlay_style(ln, cfg)
        halo = cfg.get("LINE_HALO", True) if ln.get("halo") is None else bool(ln["halo"])
        pe = [patheffects.withStroke(linewidth=lw + 1.6, foreground="white")] if halo else []
        alpha = 0.25 if ln.get("alpha") is None else float(ln["alpha"])
        for part in ln["drawn"]:
            px, py = to_frame(frame, part["lon"], part["lat"])
            if part["closed"]:
                px, py = np.r_[px, px[0]], np.r_[py, py[0]]
            if ln.get("fill") and part["closed"]:
                ax.fill(px, py, facecolor=ln["fill"], edgecolor="none", alpha=alpha, zorder=2.5)
            elif part["fill"] is not None and part["fill"][0].size:     # the polygon cut at the box
                fx, fy = to_frame(frame, part["fill"][0], part["fill"][1])
                ax.fill(fx, fy, facecolor=ln["fill"], edgecolor="none", alpha=alpha, zorder=2.5)
            if px.size:
                ax.plot(px, py, color=color, lw=lw, ls=ln.get("ls") or "-", zorder=3,
                        path_effects=pe, solid_joinstyle="round")

    fs = float(cfg.get("PROFILE_FONTSIZE", 9.0))
    for pr0 in overlays.get("profiles") or []:
        pr = cut_profile(pr0, bbox, margin)
        if not pr["pieces"]:
            continue                                       # nothing of it inside the map
        color, lw = _overlay_style(pr, cfg)
        pe = [patheffects.withStroke(linewidth=lw + 1.6, foreground="white")]
        halo_t = [patheffects.withStroke(linewidth=2.0, foreground="white")]
        if pr.get("swath"):
            verts: List[np.ndarray] = []
            codes: List[int] = []
            for poly in pr["swath"]:
                qx, qy = to_frame(frame, poly[:, 0], poly[:, 1])
                verts.append(np.c_[np.r_[qx, qx[0]], np.r_[qy, qy[0]]])
                codes += [Path.MOVETO] + [Path.LINETO] * (len(qx) - 1) + [Path.CLOSEPOLY]
            ax.add_patch(PathPatch(Path(np.vstack(verts), codes), facecolor=color,
                                   edgecolor="none", alpha=float(cfg.get("PROFILE_SWATH_ALPHA", 0.15)),
                                   zorder=2))
        for pc in pr["pieces"]:
            px, py = to_frame(frame, pc["lon"], pc["lat"])
            ax.plot(px, py, color=color, lw=lw, ls=pr.get("ls") or "-", zorder=4, path_effects=pe,
                    solid_capstyle="butt", solid_joinstyle="round")
        tp = profile_tick_points(pr, pr["ticks_km"]) if pr.get("ticks_km") else None
        if tp is not None and pr["cut"]:                   # only the ticks on the parts that are left
            keep = np.zeros(tp["s"].size, bool)
            for pc in pr["pieces"]:
                keep |= (tp["s"] >= pc["s0"] - 1e-6) & (tp["s"] <= pc["s1"] + 1e-6)
            tp = {k: v[keep] for k, v in tp.items()}
        if tp is not None and tp["s"].size:
            h = 0.5 * _PROFILE_TICK_FRAC * map_km
            a_lon, a_lat = local_lonlat(tp["x"] + h * tp["uy"], tp["y"] - h * tp["ux"],
                                        pr["lon0"], pr["lat0"])
            b_lon, b_lat = local_lonlat(tp["x"] - h * tp["uy"], tp["y"] + h * tp["ux"],
                                        pr["lon0"], pr["lat0"])
            ax1, ay1 = to_frame(frame, a_lon, a_lat)
            bx1, by1 = to_frame(frame, b_lon, b_lat)
            segs = [[(ax1[k], ay1[k]), (bx1[k], by1[k])] for k in range(ax1.size)]
            ax.add_collection(LineCollection(segs, colors=color, linewidths=max(lw * 0.8, 0.8),
                                             zorder=4, path_effects=pe))
            if pr.get("tick_labels", True):
                c_lon, c_lat = local_lonlat(tp["x"], tp["y"], pr["lon0"], pr["lat0"])
                cx, cy = to_frame(frame, c_lon, c_lat)
                for k in range(cx.size):
                    ax.annotate(f"{tp['s'][k]:g}", (cx[k], cy[k]),
                                xytext=(-8.0 * tp["uy"][k], 8.0 * tp["ux"][k]),
                                textcoords="offset points", ha="center", va="center",
                                fontsize=fs - 2.0, color=color, zorder=6.5, path_effects=halo_t,
                                annotation_clip=True)
        first, last = pr["pieces"][0], pr["pieces"][-1]          # the ends of what is shown
        fx0, fy0 = to_frame(frame, first["lon"][:1], first["lat"][:1])
        fx1, fy1 = to_frame(frame, last["lon"][-1:], last["lat"][-1:])
        if cfg.get("PROFILE_END_MARKER"):
            ax.plot([fx0[0], fx1[0]], [fy0[0], fy1[0]], ls="", marker=cfg["PROFILE_END_MARKER"],
                    ms=6.0, mfc=color, mec="white", mew=0.8, zorder=6)
        if pr.get("labels"):
            ends = ((fx0[0], fy0[0], -1.0, pr["labels"][0],
                     first["x"][1] - first["x"][0], first["y"][1] - first["y"][0]),
                    (fx1[0], fy1[0], 1.0, pr["labels"][1],
                     last["x"][-1] - last["x"][-2], last["y"][-1] - last["y"][-2]))
            for vx, vy, sgn, txt, dxk, dyk in ends:
                nrm = math.hypot(dxk, dyk) or 1.0
                ax.annotate(txt, (vx, vy), xytext=(sgn * 10.0 * dxk / nrm, sgn * 10.0 * dyk / nrm),
                            textcoords="offset points", ha="center", va="center", fontsize=fs,
                            fontweight="bold", color=color, zorder=6.5, path_effects=halo_t)
        ring = pr.get("ring")
        if cfg.get("PROFILE_MARK_SITES", True) and ring is not None and len(ring[0]):
            rlon, rlat = np.asarray(ring[0], float), np.asarray(ring[1], float)
            if pr["cut"] and pr.get("proj") is not None and pr.get("order") is not None:
                al = np.asarray(pr["proj"]["along"])[np.asarray(pr["order"], int)]
                keep = np.zeros(al.size, bool)                  # only the sites along the parts left
                for pc in pr["pieces"]:
                    keep |= (al >= pc["s0"] - 1e-6) & (al <= pc["s1"] + 1e-6)
                rlon, rlat = rlon[keep], rlat[keep]
            if rlon.size:
                rx, ry = to_frame(frame, rlon, rlat)
                ax.scatter(rx, ry, s=1.9 * float(cfg["SITE_SIZE"]), marker="o", facecolors="none",
                           edgecolors=color, linewidths=1.0, zorder=4.5)


def overlay_handles(overlays: Optional[dict], cfg: Dict[str, Any]) -> list:
    """Legend handles of the profiles and lines that have a label (profiles first)."""
    from matplotlib.lines import Line2D
    out = []
    if not overlays:
        return out
    for kind in ("profiles", "lines"):
        for it in overlays.get(kind) or []:
            if it.get("label"):
                color, lw = _overlay_style(it, cfg)
                out.append(Line2D([], [], color=color, lw=lw, ls=it.get("ls") or "-",
                                  label=str(it["label"])))
    return out


def overlay_outside(items: Sequence[dict], bbox: Sequence[float]) -> List[str]:
    """Names of the profiles / lines whose bounding box does not touch the map `bbox`."""
    out = []
    for it in items:
        if it["kind"] == "profile":
            lon, lat, name = it["lon"], it["lat"], it["name"]
        else:
            lon = np.concatenate([p["lon"] for p in it["parts"]])
            lat = np.concatenate([p["lat"] for p in it["parts"]])
            name = str(it.get("label") or "line")
        if (lon.max() < bbox[0] or lon.min() > bbox[1] or lat.max() < bbox[2]
                or lat.min() > bbox[3]):
            out.append(name)
    return out


def draw_panel(ax, *, frame: dict, bbox: Tuple[float, float, float, float], relief: Optional[dict],
               relief_extent: Optional[Tuple[float, float, float, float]],
               lon: np.ndarray, lat: np.ndarray, values: np.ndarray,
               names: Optional[Sequence[str]], cmap, norm, cfg: Dict[str, Any],
               credit: Optional[str] = None, overlays: Optional[dict] = None):
    """
    One map: relief, then the sites as icons coloured by `values` (NaN = open
    icon, no data), and the profiles and lines of `overlays` (draw_overlays).
    Returns the scatter of the coloured icons (None if there is none). bbox =
    (lon_min, lon_max, lat_min, lat_max) of the map.
    """
    from matplotlib import patheffects
    from matplotlib.ticker import MaxNLocator
    halo = [patheffects.withStroke(linewidth=1.5, foreground="white")]
    x0, x1 = to_frame(frame, [bbox[0], bbox[1]], [bbox[2], bbox[2]])[0]
    y0, y1 = to_frame(frame, [bbox[0], bbox[0]], [bbox[2], bbox[3]])[1]
    if relief is not None and relief_extent is not None:
        ex0, ex1 = to_frame(frame, [relief_extent[0], relief_extent[1]],
                            [relief_extent[2], relief_extent[2]])[0]
        ey0, ey1 = to_frame(frame, [relief_extent[0], relief_extent[0]],
                            [relief_extent[2], relief_extent[3]])[1]
        if relief["kind"] == "gray":
            ax.imshow(np.ma.masked_invalid(relief["img"]), extent=(ex0, ex1, ey0, ey1),
                      origin="upper", cmap=cfg["BASEMAP_CMAP"], vmin=0.0, vmax=1.0,
                      alpha=float(cfg["BASEMAP_ALPHA"]), interpolation="bilinear", zorder=1)
        else:
            ax.imshow(relief["img"], extent=(ex0, ex1, ey0, ey1), origin="upper",
                      alpha=float(cfg["BASEMAP_ALPHA"]), interpolation="bilinear", zorder=1)
    x, y = to_frame(frame, lon, lat)
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    sc = None
    kw = dict(marker=cfg["SITE_MARKER"], s=float(cfg["SITE_SIZE"]), zorder=5)
    if (~ok).any():
        ax.scatter(x[~ok], y[~ok], facecolors="none", edgecolors=cfg["SITE_NODATA_EDGECOLOR"],
                   linewidths=float(cfg["SITE_LW"]), **kw)
    if ok.any():
        order = np.argsort(v[ok])                      # the largest nRMS on top
        sc = ax.scatter(x[ok][order], y[ok][order], c=v[ok][order], cmap=cmap, norm=norm,
                        edgecolors=cfg["SITE_EDGECOLOR"], linewidths=float(cfg["SITE_LW"]), **kw)
    if cfg["SITE_LABELS"] and names is not None:
        for xi, yi, nm in zip(x, y, names):
            ax.annotate(str(nm), (xi, yi), xytext=(4, 4), textcoords="offset points",
                        fontsize=float(cfg["LABEL_FONTSIZE"]), zorder=6, path_effects=halo)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect(frame["aspect"])
    draw_overlays(ax, frame=frame, overlays=overlays, cfg=cfg, bbox=bbox)
    if cfg["MAP_GRID"]:
        ax.grid(True, color="0.5", lw=0.4, alpha=0.6)
    fs = float(cfg["TICK_FONTSIZE"])
    ax.tick_params(labelsize=fs)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.ticklabel_format(useOffset=False, style="plain")
    if frame["display"] == "latlon":
        ax.set_xlabel("Longitude [deg]", fontsize=fs)
        ax.set_ylabel("Latitude [deg]", fontsize=fs)
    else:
        ax.set_xlabel("East [km]", fontsize=fs)
        ax.set_ylabel("North [km]", fontsize=fs)
    if cfg["SCALE_BAR"]:
        width_km = abs(x1 - x0) * (frame["kx"] if frame["display"] == "latlon" else 1.0)
        length_km = (float(cfg["SCALE_BAR_KM"]) if cfg["SCALE_BAR_KM"]
                     else _nice_length(0.2 * width_km))
        length = length_km / (frame["kx"] if frame["display"] == "latlon" else 1.0)
        xa, ya = x0 + 0.05 * (x1 - x0), y0 + 0.05 * (y1 - y0)
        ax.plot([xa, xa + length], [ya, ya], color="black", lw=2.0,
                solid_capstyle="butt", zorder=7, path_effects=[patheffects.withStroke(
                    linewidth=4.0, foreground="white")])
        ax.text(xa + 0.5 * length, ya + 0.012 * (y1 - y0), f"{length_km:g} km", ha="center",
                va="bottom", fontsize=fs - 1, zorder=7, path_effects=halo)
    if credit:
        w_in = ax.figure.get_figwidth() * ax.get_position().width
        ax.text(0.99, 0.01, wrap_credit(credit, w_in, 5.0), transform=ax.transAxes, ha="right",
                va="bottom", multialignment="right", fontsize=5.0, color="0.15", zorder=7,
                path_effects=halo)
    return sc


def wrap_credit(text: str, width_in: float, fontsize: float) -> str:
    """`text` broken into lines that fit (about) `width_in` inch at `fontsize` pt."""
    import textwrap
    n = max(20, int(0.9 * float(width_in) * 72.0 / (0.55 * float(fontsize))))
    return "\n".join(textwrap.wrap(str(text), width=n)) if len(str(text)) > n else str(text)


def build_pages(sels: Sequence[dict], page_by: str) -> List[Tuple[str, List[dict]]]:
    """[(file tag, [selections on the page]), ...] in the order of `sels`."""
    if page_by == "all":
        return [("all", list(sels))] if sels else []
    pages: Dict[str, List[dict]] = {}
    for s in sels:
        if page_by == "none":
            tag = f"b{s['ib']:02d}_c{s['ic']:02d}"
        elif page_by == "band":
            tag = f"b{s['ib']:02d}_{_slug(s['band'])}"
        elif page_by == "component":
            tag = f"c{s['ic']:02d}_{_slug(s['comp'])}"
        else:
            raise ValueError(f"PAGE_BY = {page_by!r}: one of {list(_PAGE_BY)}")
        pages.setdefault(tag, []).append(s)
    return list(pages.items())


def make_title(spec: Optional[str], *, iteration: int, nrms: float,
               ens: Optional[dict] = None) -> str:
    """
    Figure title: "auto", None / "" (none) or text with {iter}, {nrms}, {M} and
    {stat}. `ens` (an ensemble) = dict(M, stat, iters): number of members, ENS_STAT,
    iterations of the members; then {iter} is "8" or "5-8", {nrms} the statistic
    of the nRMS of the members, {M} their number and {stat} the statistic in
    words (for a single run: M = 1, stat = "").
    """
    if spec is None or spec is False or spec == "":
        return ""
    if ens is None:
        if spec == "auto":
            return f"FEMTIC site nRMS (iteration {iteration}, global nRMS = {nrms:.2f})"
        return str(spec).format(iter=iteration, nrms=nrms, M=1, stat="")
    it = iteration_text(ens["iters"])
    words = stat_words(ens["stat"])
    if spec == "auto":
        what = "std of global nRMS" if parse_stat(ens["stat"])[0] == "std" else "global nRMS"
        word = "iteration" if "-" not in it else "iterations"
        return (f"FEMTIC site nRMS, {words} of {ens['M']} members "
                f"({word} {it}, {what} = {nrms:.2f})")
    return str(spec).format(iter=it, nrms=nrms, M=ens["M"], stat=words)


_CBAR_ASPECT = 20.0                # matplotlib's default: length / width of a colour bar
_CBAR_TOL = 0.01                   # the colour bar has the asked size within 1 %
_SAID: set = set()                 # messages that are said once (per run)


def _say_once(msg: str) -> None:
    if msg not in _SAID:
        _SAID.add(msg)
        _say(msg)


def _layout_pass(fig) -> None:
    """One layout pass of the figure without drawing it (the positions are then final)."""
    eng = fig.get_layout_engine() if hasattr(fig, "get_layout_engine") else None
    if eng is not None and hasattr(eng, "execute"):
        eng.execute(fig)
    else:                                          # older matplotlib: layout happens at draw time
        fig.canvas.draw()


def _cbar_size_in(cb, vertical: bool) -> Tuple[float, float]:
    """
    (length, width) in inch of the colour bar as it is drawn, after a layout pass.
    The length is that of the whole bar with its extension triangles (`extend`);
    the axes of the bar itself (the coloured rectangle) are shorter by that, which
    matplotlib's locator of the bar applies at draw time, so it is applied here.
    """
    ax = cb.ax
    full = ax.get_position(original=True)          # the slot: the bar with its extensions
    loc = ax.get_axes_locator()
    pos = full
    if loc is not None:
        try:
            renderer = ax.figure.canvas.get_renderer()
        except Exception:                          # a canvas without get_renderer
            renderer = None
        pos = loc(ax, renderer)
    ax.apply_aspect(pos)
    body = ax.get_position()
    fw, fh = ax.figure.get_size_inches()
    if vertical:
        ext = full.height / pos.height if pos.height > 0.0 else 1.0
        return body.height * ext * fh, body.width * fw
    ext = full.width / pos.width if pos.width > 0.0 else 1.0
    return body.width * ext * fw, body.height * fh


def add_colorbar(fig, mappable, axes, *, location: str, shrink: float, extend: str,
                 label: str, length_cm: Optional[float] = None,
                 width_cm: Optional[float] = None):
    """
    The colour bar of the panels `axes`. Without `length_cm` and `width_cm` it is
    matplotlib's: the length is `shrink` x the length of the panels along the bar
    and the width 1/20 of its length. `length_cm` / `width_cm` [cm] fix the size
    that is drawn (the length with the extension triangles of `extend`, the width of
    the coloured bar): the figure is laid out (without drawing) and the bar made again
    with the length (`shrink`) and the width (`aspect` = length / width) that this
    pass shows are needed, repeated (up to 6 passes) until it is within 1 %. `width_cm` alone keeps the
    length, `length_cm` alone the proportions (1/20). A length that the panels
    cannot hold is reduced to the length of the panels, with a message. The size
    holds for the layout as it is when this is called (set the figure title first).
    """
    kw = dict(location=location, extend=extend, label=label)
    cb = fig.colorbar(mappable, ax=axes, shrink=shrink, **kw)
    if length_cm is None and width_cm is None:
        return cb
    vertical = location in ("right", "left")
    want_l = None if length_cm is None else float(length_cm) / 2.54
    want_w = None if width_cm is None else float(width_cm) / 2.54
    aspect = _CBAR_ASPECT
    frac = 0.5 if want_w is not None else 0.15      # the width may be more than 15 % of the panels
    clipped = False
    keep_l = None                                   # no CBAR_LENGTH: the length of the plain bar
    for it in range(6):
        _layout_pass(fig)
        have_l, have_w = _cbar_size_in(cb, vertical)
        if want_l is None and keep_l is None:
            keep_l = have_l
        goal_l = have_l if clipped else (keep_l if want_l is None else want_l)
        goal_w = want_w if want_w is not None else goal_l / aspect
        if abs(have_l - goal_l) <= _CBAR_TOL * goal_l and abs(have_w - goal_w) <= _CBAR_TOL * goal_w:
            break
        new_shrink = shrink * goal_l / have_l
        if new_shrink > 1.0:                        # longer than the panels: all of their length
            new_shrink, clipped = 1.0, True
        if want_w is not None:
            aspect = have_l * new_shrink / shrink / want_w
        shrink = new_shrink
        cb.remove()
        cb = fig.colorbar(mappable, ax=axes, shrink=shrink, aspect=aspect, fraction=frac, **kw)
    else:
        _say(f"the colour bar has {have_l * 2.54:.2f} x {have_w * 2.54:.2f} cm, not the asked "
             f"size (CBAR_LENGTH, CBAR_WIDTH).")
    if clipped:
        msg = (f"CBAR_LENGTH = {length_cm:g} cm is longer than the panels "
               f"({have_l * 2.54:.1f} cm): the colour bar is as long as the panels.")
        _say_once(msg)
    return cb


def plot_page(page: Sequence[dict], *, site_lon: np.ndarray, site_lat: np.ndarray,
              site_names: Sequence[str], frame: dict,
              bbox: Tuple[float, float, float, float], relief: Optional[dict],
              relief_extent: Optional[Tuple[float, float, float, float]], norm, extend: str,
              cfg: Dict[str, Any], title: str, credit: Optional[str],
              cbar_label: str = "nRMS", overlays: Optional[dict] = None):
    """
    One figure with the maps of the selections of `page`; returns the figure. The
    second line of a panel title is s["sub"] if the selection has one (ensemble),
    else the global nRMS and n. `overlays` (profiles, lines) are drawn in every
    panel; the legend (LEGEND_LOC, first panel) holds the no-data icon and the
    profiles and lines that have a label.
    """
    import matplotlib.pyplot as plt
    n = len(page)
    nc = min(int(cfg["PANEL_COLS"]), n)
    nr = int(math.ceil(n / nc))
    pw, ph = cfg["PANEL_SIZE"]
    fig, axs = plt.subplots(nr, nc, figsize=(pw * nc + 1.0, ph * nr + (0.5 if title else 0.0)),
                            squeeze=False, constrained_layout=True)
    cmap = plt.get_cmap(cfg["CMAP_NRMS"])
    used = []
    any_open = False
    for k, s in enumerate(page):
        ax = axs[k // nc][k % nc]
        used.append(ax)
        vals = s["nrms_site"]
        any_open = any_open or bool(np.isnan(vals).any())
        draw_panel(ax, frame=frame, bbox=bbox, relief=relief, relief_extent=relief_extent,
                   lon=site_lon, lat=site_lat, values=vals, names=site_names, cmap=cmap,
                   norm=norm, cfg=cfg, credit=credit, overlays=overlays)
        sub = s.get("sub") or f"global nRMS = {s['nrms']:.2f}, n = {s['n']}"
        ax.set_title(f"{s['band']} | {s['comp']}\n{sub}", fontsize=float(cfg["TICK_FONTSIZE"]) + 1)
    for k in range(n, nr * nc):
        axs[k // nc][k % nc].set_visible(False)
    if title:                                      # before the colour bar: its size is fitted to the layout
        fig.suptitle(title, fontsize=float(cfg["TICK_FONTSIZE"]) + 3)
    loc = cfg["CBAR_LOCATION"] or "right"
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = add_colorbar(fig, sm, used, location=loc, shrink=float(cfg["CBAR_SHRINK"]),
                      extend=extend, label=cbar_label, length_cm=cfg["CBAR_LENGTH"],
                      width_cm=cfg["CBAR_WIDTH"])
    ref = cfg["NRMS_REFERENCE"]
    if ref is not None and norm.vmin <= ref <= norm.vmax:
        (cb.ax.axhline if loc in ("right", "left") else cb.ax.axvline)(
            ref, color="black", lw=0.8, ls="--")
    handles = []
    if any_open:
        from matplotlib.lines import Line2D
        handles.append(Line2D([], [], ls="", marker=cfg["SITE_MARKER"], mfc="none",
                              mec=cfg["SITE_NODATA_EDGECOLOR"], label="no data"))
    handles += overlay_handles(overlays, cfg)
    leg_loc = cfg.get("LEGEND_LOC", "upper left")
    if handles and leg_loc:
        used[0].legend(handles=handles, loc=leg_loc, fontsize=float(cfg["TICK_FONTSIZE"]) - 1,
                       framealpha=0.8)
    return fig


def save_figure(fig, stem: str, formats: Sequence[str], dpi: int) -> List[str]:
    """Write `fig` to stem.<fmt> for every format; returns the file names."""
    out = []
    for fmt in ([formats] if isinstance(formats, str) else formats):
        p = f"{stem}.{str(fmt).lstrip('.')}"
        fig.savefig(p, dpi=int(dpi))
        out.append(p)
    return out


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def _fmt(v: float, nd: int = 4) -> str:
    return "" if not np.isfinite(v) else f"{v:.{nd}f}"


def write_site_tables(stem: str, rows: Sequence[dict], sels: Sequence[dict],
                      used: np.ndarray) -> List[str]:
    """
    <stem>_nrms.csv and <stem>_n.csv: one row per site (name, lat, lon, site
    ids), one column per selection ("band | components"): nRMS, and the number
    of values. `rows` = the sites of site.dat (name, lat, lon, ids), `used` =
    their indices (the sites with data).
    """
    paths = []
    head = ["site", "lat", "lon", "site_ids"] + [f"{s['band']} | {s['comp']}" for s in sels]
    for suffix, key, fmt in (("nrms", "nrms_site", _fmt), ("n", "n_site", lambda v: str(int(v)))):
        p = f"{stem}_{suffix}.csv"
        with open(p, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(head)
            for j in used:
                r = rows[j]
                w.writerow([r["name"], f"{r['lat']:.6f}", f"{r['lon']:.6f}",
                            " ".join(str(i) for i in r.get("ids", []))]
                           + [fmt(s[key][j]) for s in sels])
        paths.append(p)
    return paths


def write_members_table(stem: str, members: Sequence[dict], sels: Sequence[dict]) -> str:
    """
    <stem>_members.csv: one row per member (in the order found): member, path
    (the results file read), iteration, nrms (of the iteration), used (1 / 0), note
    (why it is not used), then the global nRMS of the member for every selection
    of `sels` ("band | comp"). `members`: dicts with name, path, res (None if not
    read), keep, note, sels.
    """
    p = f"{stem}_members.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["member", "path", "iteration", "nrms", "used", "note"]
                   + [f"{s['band']} | {s['comp']}" for s in sels])
        for m in members:
            res = m.get("res")
            look = {(t["band"], t["comp"]): t["nrms"] for t in m.get("sels", [])}
            w.writerow([m["name"], res["path"] if res else m["path"],
                        res["iter"] if res else "", _fmt(res["rms"]) if res else "",
                        int(bool(m.get("keep"))), m.get("note", "")]
                       + [_fmt(look.get((s["band"], s["comp"]), float("nan"))) for s in sels])
    return p


def write_ens_stats(stem: str, rows: Sequence[dict], sels: Sequence[dict],
                    used: np.ndarray) -> str:
    """
    <stem>_ens_stats.csv: for every selection and site with data, the number of
    members M and the mean, median, std (ddof = 1), min and max of the member nRMS.
    `sels` = aggregate_members result (key X), `rows`, `used` as in write_site_tables.
    """
    p = f"{stem}_ens_stats.csv"
    kinds = ("mean", "median", "std", "min", "max")
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["band", "components", "site", "lat", "lon", "M"] + list(kinds))
        for s in sels:
            cnt = np.sum(np.isfinite(s["X"]), axis=0)
            vals = {k: ens_statistic(s["X"], k) for k in kinds}
            for j in used:
                if cnt[j] == 0:
                    continue
                r = rows[j]
                w.writerow([s["band"], s["comp"], r["name"], f"{r['lat']:.6f}", f"{r['lon']:.6f}",
                            int(cnt[j])] + [_fmt(vals[k][j]) for k in kinds])
    return p


def overlay_summary(profiles: Sequence[dict], lines: Sequence[dict]) -> List[str]:
    """Text lines about the profiles (length, swath, sites in it) and lines, for the summary."""
    out = []
    for p in profiles:
        txt = f"profile {p['name']}: {p['length']:.1f} km, {p['lon'].size} vertices"
        if p["width"] > 0:
            n = int(p["proj"]["inside"].sum()) if "proj" in p else 0
            txt += f", swath {p['width']:g} km wide with {n} sites"
        out.append(txt)
    for ln in lines:
        nm = ln.get("label") or "(no label)"
        out.append(f"line {nm}: {len(ln['parts'])} parts, {ln['n_vertices']} vertices")
    return out


def write_profile_tables(stem: str, profiles: Sequence[dict], rows: Sequence[dict],
                         used: np.ndarray, sels: Sequence[dict]) -> List[str]:
    """
    <stem>_profiles.csv: one row per profile (name, vertices, length_km, width_km,
    n_sites in the swath, lon / lat of the start and the end, source).
    <stem>_profile_sites.csv (profiles with a swath): the sites in the swath, by
    profile and increasing distance: profile, site, lat, lon, along_km (from the
    start), offset_km (positive to the right of the direction start -> end) and the
    nRMS of every selection of `sels`. `profiles` = project_profiles result, `rows`
    = the sites of site.dat (name, lat, lon), `used` = their indices behind the
    projection.
    """
    paths = []
    p1 = f"{stem}_profiles.csv"
    with open(p1, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["profile", "vertices", "length_km", "width_km", "n_sites", "lon_start",
                    "lat_start", "lon_end", "lat_end", "source"])
        for p in profiles:
            w.writerow([p["name"], p["lon"].size, f"{p['length']:.3f}", f"{p['width']:g}",
                        len(p["order"]), f"{p['lon'][0]:.6f}", f"{p['lat'][0]:.6f}",
                        f"{p['lon'][-1]:.6f}", f"{p['lat'][-1]:.6f}", p["source"]])
    paths.append(p1)
    if any(p["width"] > 0 for p in profiles):
        p2 = f"{stem}_profile_sites.csv"
        with open(p2, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["profile", "site", "lat", "lon", "along_km", "offset_km"]
                       + [f"{s['band']} | {s['comp']}" for s in sels])
            for p in profiles:
                for k in p["order"]:
                    j = int(used[k])
                    r = rows[j]
                    w.writerow([p["name"], r["name"], f"{r['lat']:.6f}", f"{r['lon']:.6f}",
                                f"{p['proj']['along'][k]:.3f}", f"{p['proj']['offset'][k]:.3f}"]
                               + [_fmt(s["nrms_site"][j]) for s in sels])
        paths.append(p2)
    return paths


def format_summary(sels: Sequence[dict], *, header: Sequence[str]) -> str:
    """
    Plain-text table: global nRMS and the spread of the site values per selection.
    For an ensemble (selections of aggregate_members) with the number of members M
    and the smallest and largest global nRMS of a member (m_min, m_max).
    """
    ens = bool(sels) and "G" in sels[0]
    cols = ("band", "components", "n", "nRMS") + (("M", "m_min", "m_max") if ens else ()) \
        + ("sites", "min", "median", "max", "n_unassigned")
    body = []
    for s in sels:
        v = s["nrms_site"][np.isfinite(s["nrms_site"])]
        row = [s["band"], s["comp"], str(s["n"]), _fmt(s["nrms"], 3)]
        if ens:
            g = s["G"][np.isfinite(s["G"])]
            row += [str(s["M"]), _fmt(float(g.min()), 3) if g.size else "",
                    _fmt(float(g.max()), 3) if g.size else ""]
        row += [str(v.size), _fmt(float(v.min()), 3) if v.size else "",
                _fmt(float(np.median(v)), 3) if v.size else "",
                _fmt(float(v.max()), 3) if v.size else "", str(s["n_unassigned"])]
        body.append(tuple(row))
    w = [max(len(c), *(len(r[i]) for r in body)) if body else len(c) for i, c in enumerate(cols)]
    line = "-" * (sum(w) + 2 * (len(w) - 1))
    fmtrow = lambda r: "  ".join(                                    # noqa: E731
        (str(x).ljust(w[i]) if i < 2 else str(x).rjust(w[i])) for i, x in enumerate(r))
    out = list(header) + ["", "=" * len(line), fmtrow(cols), line]
    out += [fmtrow(r) for r in body]
    out += [line,
            "n = number of values (real and imaginary parts), global = all rows of the "
            "selection;",
            "sites, min, median, max = nRMS of the sites; n_unassigned = values of rows "
            "that belong to no site."]
    if ens:
        stat = stat_words(sels[0]["stat"])
        out += [f"nRMS and the nRMS of the sites = {stat} over the members (M = members with a "
                "value; n, n_unassigned = median of the members);",
                "m_min, m_max = smallest and largest global nRMS of a member."]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run(cfg: Dict[str, Any]) -> int:
    """
    Run with the settings `cfg` (see settings()). Return code: 0 success, 1 data
    problem (nothing to plot), 2 usage (WORK_DIR missing).
    """
    wd = cfg["WORK_DIR"]
    if not os.path.isdir(wd):
        print(f"{_TAG} WORK_DIR is not a directory: {wd}", file=sys.stderr)
        return 2
    try:
        check_settings(cfg)
    except ValueError as exc:
        print(f"{_TAG} {exc}", file=sys.stderr)
        return 2
    import matplotlib
    if not cfg["SHOW"]:
        matplotlib.use("Agg")
    _say(f"WORK_DIR = {wd}")

    # --- members: the first one that can be read ---
    paths, how = member_paths(cfg)
    if not paths:
        print(f"{_TAG} RUN_PATHS matches no path: {list(cfg['RUN_PATHS'] or [])}", file=sys.stderr)
        return 1
    n_paths = len(paths)
    names = member_names(paths)
    if n_paths > 1:
        _say(f"ensemble of {n_paths} members ({how}); statistic over the members: "
             f"{stat_words(cfg['ENS_STAT'])}.")
    failed: Dict[int, str] = {}              # member index -> why it is not read
    first, k0, err0 = None, 0, None
    for k, pth in enumerate(paths):
        try:
            first = load_member(pth, cfg)
            k0 = k
            break
        except (OSError, ValueError, KeyError) as exc:       # DataTypeError is a ValueError
            if err0 is None:
                err0 = exc
            failed[k] = str(exc)
            if n_paths > 1:
                _say(f"WARNING: member {names[k]} skipped: {exc}")
    if first is None:
        if isinstance(err0, DataTypeError):
            print(f"{_TAG} {err0}", file=sys.stderr)
        else:
            print(f"{_TAG} cannot read the results{' of any member' if n_paths > 1 else ''}: "
                  f"{err0}", file=sys.stderr)
        return 1
    res, ds, n_rows = first["res"], first["ds"], first["n_rows"]
    _say(f"{res['path']}: iteration {res['iter']}, {n_rows} data rows, nRMS = {res['rms']:.3f}.")

    # --- sites ---
    site_dirs = site_search_dirs(cfg["TEMPLATES_DIR"], wd)
    site_path = find_site_dat(cfg["SITE_DAT"], site_dirs)
    if site_path is None:
        print(f"{_TAG} no site.dat found (SITE_DAT = {cfg['SITE_DAT']!r}; looked in "
              f"{', '.join(site_dirs)}).", file=sys.stderr)
        return 1
    try:
        sites = read_site_dat(site_path)
    except (OSError, ValueError) as exc:
        print(f"{_TAG} cannot read site.dat: {exc}", file=sys.stderr)
        return 1
    site_index, st = assign_sites(ds, sites, match=cfg["SITE_MATCH"],
                                  tol=cfg["SITE_MATCH_TOL"], offset=cfg["SITE_ID_OFFSET"])
    _say(f"{len(sites)} sites in {site_path}; assigned by {st['mode']}: {st['n_hit']} of "
         f"{st['n_ids']} site ids ({st['n_sites']} sites)"
         + (f"; frame {st['frame']}, shift {st['shift'][0]:.1f}, {st['shift'][1]:.1f} m, "
            f"largest distance {st['dmax']:.1f} m" if st["mode"] == "coords" else "") + ".")
    if st.get("n_spread"):
        _say(f"{st['n_spread']} site ids share a position with another id; they were given the "
             "site.dat entries at that position in the order of their numbers.")
    if st["unmatched"]:
        _say(f"WARNING: no site for the site ids {st['unmatched'][:10]}"
             f"{' ...' if len(st['unmatched']) > 10 else ''}; their data count in the global "
             "values only.")
    if st["n_hit"] == 0:
        print(f"{_TAG} no site id could be assigned to a site of {site_path}; "
              "check SITE_MATCH, SITE_MATCH_TOL, SITE_ID_OFFSET and the site.dat.", file=sys.stderr)
        return 1
    if st["n_ids"] and st["n_hit"] < 0.5 * st["n_ids"]:
        _say("WARNING: fewer than half of the site ids assigned -- is this the site.dat of the "
             "data set?")

    # --- nRMS of every member ---
    seen: set = set()                        # messages of the selections: once
    try:
        with _print_once(seen):
            sels0 = member_selections(first, site_index, len(sites), cfg)
    except ValueError as exc:
        print(f"{_TAG} {exc}", file=sys.stderr)
        return 1
    members = [dict(k=k0, name=names[k0], path=paths[k0], res=res, n_rows=n_rows, sels=sels0,
                    keep=True, note="", used=np.unique(site_index[site_index >= 0]),
                    id_to_site=st["id_to_site"])]
    for k in range(k0 + 1, n_paths):
        try:
            lm = load_member(paths[k], cfg)
            idx, st_k = assign_sites(lm["ds"], sites, match=cfg["SITE_MATCH"],
                                     tol=cfg["SITE_MATCH_TOL"], offset=cfg["SITE_ID_OFFSET"])
            if st_k["n_hit"] == 0:
                raise ValueError("no site id could be assigned to a site of the site.dat")
            with _print_once(seen):
                sels_k = member_selections(lm, idx, len(sites), cfg)
        except (OSError, ValueError, KeyError) as exc:
            failed[k] = str(exc)
            _say(f"WARNING: member {names[k]} skipped: {exc}")
            continue
        if (st_k["n_hit"], st_k["n_ids"]) != (st["n_hit"], st["n_ids"]):
            _say(f"WARNING: member {names[k]}: {st_k['n_hit']} of {st_k['n_ids']} site ids "
                 f"assigned ({st['n_hit']} of {st['n_ids']} in {names[k0]}).")
        members.append(dict(k=k, name=names[k], path=paths[k], res=lm["res"],
                            n_rows=lm["n_rows"], sels=sels_k, keep=True, note="",
                            used=np.unique(idx[idx >= 0]), id_to_site=st_k["id_to_site"]))
        if n_paths <= 20 or len(members) % 10 == 0 or k == n_paths - 1:
            _say(f"member {names[k]} ({k + 1} of {n_paths}): iteration {lm['res']['iter']}, "
                 f"nRMS = {lm['res']['rms']:.3f}.")

    # --- which members count ---
    mmax = cfg["MEMBER_NRMS_MAX"]
    if mmax is not None:
        for m in members:
            if not m["res"]["rms"] <= float(mmax):
                m["keep"] = False
                m["note"] = f"nRMS above MEMBER_NRMS_MAX = {float(mmax):g}"
        out_ = [m["name"] for m in members if not m["keep"]]
        if out_:
            _say(f"{len(out_)} members with nRMS above MEMBER_NRMS_MAX = {float(mmax):g} left "
                 f"out: {', '.join(out_[:10])}{' ...' if len(out_) > 10 else ''}.")
    kept = [m for m in members if m["keep"]]
    if not kept:
        print(f"{_TAG} no member left: the nRMS of all {len(members)} members is above "
              f"MEMBER_NRMS_MAX = {float(mmax):g}.", file=sys.stderr)
        return 1
    ens = len(kept) > 1
    if n_paths > 1:
        _say(f"{len(kept)} of {n_paths} members used"
             + ("" if ens else " (plotted as a single run)") + ".")
    used = np.unique(np.concatenate([m["used"] for m in kept]))
    ids_of: Dict[int, set] = {}              # site.dat entry -> its site ids in the members
    for m in kept:
        for sid_, j in m["id_to_site"].items():
            ids_of.setdefault(j, set()).add(sid_)
    site_rows = [dict(name=s["name"], lat=s["lat"], lon=s["lon"], ids=sorted(ids_of.get(j, ())))
                 for j, s in enumerate(sites)]
    kind = parse_stat(cfg["ENS_STAT"])[0]
    if ens:
        sels = aggregate_members(kept, cfg["ENS_STAT"], len(sites))
    else:
        res = kept[0]["res"]
        sels = kept[0]["sels"]
    plot_sels = [s for s in sels if np.isfinite(s["nrms_site"][used]).any()]
    if not plot_sels:
        print(f"{_TAG} no selection (band, components) has data at a site.", file=sys.stderr)
        return 1
    for s in sels:
        if s not in plot_sels:
            _say(f"selection {s['band']} | {s['comp']}: no data at a site, no map.")

    # --- profiles and lines ---
    lon_s = np.array([site_rows[j]["lon"] for j in used])
    lat_s = np.array([site_rows[j]["lat"] for j in used])
    profiles: List[dict] = []
    lines: List[dict] = []
    if cfg["PROFILES"] or cfg["LINES"]:
        try:
            profiles, lines = prepare_overlays(cfg, make_geo(sites, st))
        except OverlayError as exc:
            print(f"{_TAG} {exc}", file=sys.stderr)
            return 1
        profiles = project_profiles(profiles, lon_s, lat_s)
    overlays = dict(profiles=profiles, lines=lines) if (profiles or lines) else None

    # --- output: tables ---
    out_dir = cfg["OUTPUT_DIR"]
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, cfg["OUTPUT_BASENAME"])
    files = write_site_tables(stem, site_rows, sels, used)
    if profiles:
        files += write_profile_tables(stem, profiles, site_rows, used, sels)
    if ens:
        rms_k = np.array([m["res"]["rms"] for m in kept], float)
        its = [int(m["res"]["iter"]) for m in kept]
        n_drop = len(members) - len(kept)
        header = [f"femtic_plot_sitemap.py: nRMS of the sites, ensemble of {len(kept)} members "
                  f"({how})",
                  f"statistic over the members: {stat_words(cfg['ENS_STAT'])}; {n_paths} members "
                  f"found, {len(kept)} used, {n_drop} above MEMBER_NRMS_MAX, {len(failed)} "
                  "not readable",
                  f"iterations {iteration_text(its)}; nRMS of the iteration: min "
                  f"{np.nanmin(rms_k):.4f}, median {np.nanmedian(rms_k):.4f}, max "
                  f"{np.nanmax(rms_k):.4f}",
                  f"{len(used)} sites with data of {len(sites)} in {site_path}; "
                  f"site ids assigned by {st['mode']}"]
    else:
        header = [f"femtic_plot_sitemap.py: nRMS of the sites ({res['path']})",
                  f"iteration {res['iter']}, {kept[0]['n_rows']} data rows, "
                  f"nRMS of the iteration = {res['rms']:.4f}",
                  f"{len(used)} sites with data of {len(sites)} in {site_path}; "
                  f"site ids assigned by {st['mode']}"]
    if n_paths > 1:
        every = sorted([dict(m) for m in members] +
                       [dict(k=k, name=names[k], path=paths[k], res=None, sels=[], keep=False,
                             note=msg) for k, msg in failed.items()], key=lambda m: m["k"])
        files.append(write_members_table(stem, every, sels))
    if ens:
        files.append(write_ens_stats(stem, site_rows, sels, used))
    header += overlay_summary(profiles, lines)
    txt = format_summary(sels, header=header)
    with open(stem + "_summary.txt", "w") as fh:
        fh.write(txt)
    files.append(stem + "_summary.txt")
    print(txt)

    # --- basemap ---
    ext_lon, ext_lat = lon_s, lat_s
    if cfg["MAP_INCLUDE_PROFILES"] and profiles:
        ext_lon = np.concatenate([lon_s] + [p["lon"] for p in profiles])
        ext_lat = np.concatenate([lat_s] + [p["lat"] for p in profiles])
    try:
        bbox = map_extent(ext_lon, ext_lat, mode=cfg["MAP_EXTENT_MODE"], box=cfg["MAP_BOX"],
                          center=cfg["MAP_CENTER"], size_km=cfg["MAP_SIZE_KM"],
                          margin_frac=cfg["MAP_MARGIN_FRAC"],
                          margin_min_km=cfg["MAP_MARGIN_MIN_KM"])
    except ValueError as exc:
        print(f"{_TAG} {exc}", file=sys.stderr)
        return 1
    inside = ((lon_s >= bbox[0]) & (lon_s <= bbox[1]) & (lat_s >= bbox[2]) & (lat_s <= bbox[3]))
    if not inside.all():
        _say(f"WARNING: {int(np.count_nonzero(~inside))} of {inside.size} sites lie outside the "
             "map (not shown; see the tables).")
    out_of_map = overlay_outside(list(profiles) + list(lines), bbox)
    if out_of_map:
        _say(f"WARNING: outside the map (not shown): {', '.join(out_of_map)}.")
    try:
        for msg in cut_report(profiles, lines, bbox, cfg["OVERLAY_CUT_MARGIN_KM"]):
            _say(msg)
    except OverlayError as exc:
        _say(f"WARNING: {exc}: profiles and lines are not cut.")
    frame = make_frame(cfg["DISPLAY_COORDS"], 0.5 * (bbox[0] + bbox[1]), 0.5 * (bbox[2] + bbox[3]))
    relief = None
    relief_extent = None
    credit = None
    if cfg["DEM_SOURCE"] != "none":
        try:
            if cfg["DEM_SOURCE"] == "file":
                dem = load_dem_file(cfg["DEM_FILE"], bbox, cfg["DEM_MAX_PIXELS"])
            else:
                template = dem_template(cfg["DEM_SOURCE"], cfg["DEM_URL_TEMPLATE"])
                tkw = dict(cache_dir=cfg["DEM_CACHE_DIR"], template=template,
                           download=cfg["DEM_DOWNLOAD"], timeout=cfg["DEM_TIMEOUT"],
                           max_tiles=cfg["DEM_MAX_TILES"], fill=cfg["DEM_FILL_MISSING"],
                           strict=cfg["DEM_STRICT"])
                if tile_kind(template) == "raster":
                    dem = load_dem_raster(*bbox, max_pixels=cfg["DEM_MAX_PIXELS"], **tkw)
                else:
                    dem = load_dem_hgt(*bbox, **tkw)
            if dem is not None:
                dem = decimate_dem(crop_dem(dem, *bbox), cfg["DEM_MAX_PIXELS"])
                relief = shade_dem(dem, style=cfg["DEM_STYLE"], azdeg=cfg["HILLSHADE_AZIMUTH"],
                                   altdeg=cfg["HILLSHADE_ALTITUDE"],
                                   vert_exag=cfg["HILLSHADE_VERT_EXAG"], cmap=cfg["DEM_CMAP"])
                relief_extent = dem_extent(dem)
                _say(f"DEM: {dem['z'].shape[1]} x {dem['z'].shape[0]} pixels, "
                     f"{np.nanmin(dem['z']):.0f} .. {np.nanmax(dem['z']):.0f} m.")
            else:
                _say("WARNING: no DEM -- the maps have no relief.")
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            if cfg["DEM_STRICT"]:
                print(f"{_TAG} DEM: {exc}", file=sys.stderr)
                return 1
            _say(f"WARNING: DEM: {exc} -- the maps have no relief.")
    if relief is not None:
        credit = cfg["DEM_CREDIT"]
        if credit == "auto":
            credit = (_DEM_PRESETS[cfg["DEM_SOURCE"]]["credit"]
                      if cfg["DEM_SOURCE"] in _DEM_PRESETS and not cfg["DEM_URL_TEMPLATE"]
                      else None)

    # --- maps ---
    spread = ens and kind == "std"           # spread of the members: sequential scale from 0
    pcfg = dict(cfg, CMAP_NRMS=cfg["CMAP_SPREAD"], NRMS_REFERENCE=None) if spread else cfg
    vmin = 0.0 if spread and cfg["NRMS_VMIN"] is None else cfg["NRMS_VMIN"]
    allv = np.concatenate([s["nrms_site"][used] for s in plot_sels])
    norm, extend = make_norm(allv, vmin=vmin, vmax=cfg["NRMS_VMAX"],
                             reference=pcfg["NRMS_REFERENCE"])
    if ens:
        n_k = len(kept)
        title = make_title(cfg["TITLE"], iteration=its[0],
                           nrms=float(ens_statistic(rms_k, cfg["ENS_STAT"])),
                           ens=dict(M=n_k, stat=cfg["ENS_STAT"], iters=its))
        cbar_label = (f"std of nRMS ({n_k} members)" if spread
                      else f"nRMS ({stat_words(cfg['ENS_STAT'])} of {n_k} members)")
    else:
        title = make_title(cfg["TITLE"], iteration=int(res["iter"]), nrms=float(res["rms"]))
        cbar_label = "nRMS"
    import matplotlib.pyplot as plt
    figs = []
    for tag, page in build_pages(plot_sels, cfg["PAGE_BY"]):
        page_sels = [dict(s, nrms_site=s["nrms_site"][used]) for s in page]
        fig = plot_page(page_sels, site_lon=lon_s, site_lat=lat_s,
                        site_names=[site_rows[j]["name"] for j in used], frame=frame, bbox=bbox,
                        relief=relief, relief_extent=relief_extent, norm=norm, extend=extend,
                        cfg=pcfg, title=title, credit=credit, cbar_label=cbar_label,
                        overlays=overlays)
        files += save_figure(fig, f"{stem}_{tag}", cfg["OUTPUT_FORMATS"], cfg["DPI"])
        figs.append(fig)
    _say(f"{len(files)} files in {out_dir}:")
    for p in files:
        print("   " + os.path.basename(p))
    if cfg["SHOW"]:
        plt.show()
    for fig in figs:
        plt.close(fig)
    return 0


def main(argv: Sequence[str]) -> int:
    """Command line: -w, -i, --dem, --runs, --stat, --profile, --line, --no-overlays."""
    ap = argparse.ArgumentParser(
        prog="femtic_plot_sitemap.py",
        description="Site map of a FEMTIC run or ensemble: nRMS per site, band and component on "
                    "a shaded relief (Copernicus DEM by default). All paths are taken relative to "
                    "WORK_DIR.")
    ap.add_argument("-w", "--work-dir", default=None, metavar="DIR",
                    help="base of all relative paths (replaces WORK_DIR)")
    ap.add_argument("-i", "--iteration", default=None, metavar="N",
                    help="\"best\", \"last\" or a number (replaces ITERATION)")
    ap.add_argument("--dem", default=None, choices=_DEM_SOURCES,
                    help="DEM source (replaces DEM_SOURCE)")
    ap.add_argument("--runs", nargs="+", default=None, metavar="PATH",
                    help="members of the ensemble: run directories, results files or globs "
                         "(replaces RUN_PATHS)")
    ap.add_argument("--stat", default=None, metavar="STAT",
                    help="statistic over the members: mean, median, min, max, std or pNN "
                         "(replaces ENS_STAT)")
    ap.add_argument("--profile", nargs=5, action="append", default=None,
                    metavar=("NAME", "LON1", "LAT1", "LON2", "LAT2"),
                    help="a profile line from (LON1, LAT1) to (LON2, LAT2), in degrees; can be "
                         "repeated (added to PROFILES)")
    ap.add_argument("--line", action="append", default=None, metavar="FILE",
                    help="a polyline file (text / GMT with lon lat columns, GeoJSON, KML, KMZ); "
                         "can be repeated (added to LINES)")
    ap.add_argument("--no-overlays", action="store_true",
                    help="no profiles and lines (empties PROFILES and LINES)")
    args = ap.parse_args(list(argv))
    over: Dict[str, Any] = {}
    if args.no_overlays:
        if args.profile or args.line:
            ap.error("--no-overlays cannot be combined with --profile or --line")
        over["PROFILES"] = []
        over["LINES"] = []
    if args.profile:
        new = []
        for name, *num in args.profile:
            try:
                lon1, lat1, lon2, lat2 = (float(v) for v in num)
            except ValueError:
                ap.error("--profile: LON1 LAT1 LON2 LAT2 must be numbers")
            new.append(dict(name=name, start=(lon1, lat1), end=(lon2, lat2)))
        over["PROFILES"] = list(PROFILES) + new
    if args.line:
        over["LINES"] = list(LINES) + [dict(file=f, label=os.path.basename(f)) for f in args.line]
    if args.iteration is not None:
        if args.iteration in ("best", "last"):
            over["ITERATION"] = args.iteration
        else:
            try:
                over["ITERATION"] = int(args.iteration)
            except ValueError:
                ap.error("--iteration: \"best\", \"last\" or an integer")
    if args.dem is not None:
        over["DEM_SOURCE"] = args.dem
    if args.runs is not None:
        over["RUN_PATHS"] = args.runs
    if args.stat is not None:
        try:
            parse_stat(args.stat)
        except ValueError as exc:
            ap.error(str(exc))
        over["ENS_STAT"] = args.stat
    return run(settings(args.work_dir, **over))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
