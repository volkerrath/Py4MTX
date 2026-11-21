#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_plot_sitemap.py
======================

Site map of a FEMTIC run, or of a whole ensemble, on a DEM (SRTM) background:
every site is an icon (marker) coloured by its normalized RMS misfit (nRMS), for
frequency bands and data components. For an ensemble the colour is a statistic
of the member nRMS at the site.

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
   DEM is made of SRTM-format tiles (1 x 1 degree, 16-bit .hgt) that are
   downloaded into DEM_CACHE_DIR if they are not there yet (DEM_SOURCE =
   "srtm", the default), or is read from a file (DEM_SOURCE = "file":
   GeoTIFF with rasterio, or .npz with lon, lat, z), or left out ("none").
   It is shown as a shaded relief.
5. One figure per page (PAGE_BY): the panels are the maps of the selections,
   the sites drawn as icons coloured by their nRMS (colour map CMAP_NRMS,
   limits NRMS_VMIN / NRMS_VMAX, one colour scale for all pages; values
   beyond the limits get the end colours and arrows on the colour bar). Sites
   without data in a selection are drawn as open icons.
6. Tables: <OUTPUT_BASENAME>_nrms.csv and _n.csv (nRMS and number of values,
   one row per site, one column per selection) and _summary.txt (global nRMS
   and the spread of the site values per selection); for an ensemble also
   _members.csv (one row per member) and _ens_stats.csv (mean, median, std,
   min, max over the members per selection and site).

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
The default URL (DEM_URL_TEMPLATE) is the "skadi" tile set of the AWS Terrain
Tiles (public bucket elevation-tiles-prod, no key, no login):
https://s3.amazonaws.com/elevation-tiles-prod/skadi/N37/N37W105.hgt.gz. The
tiles are in the SRTM .hgt format, but they are a composite: SRTM is the main
source between about 56 S and 60 N, other sources are used elsewhere (high
latitudes). The original NASA / USGS distribution of SRTM needs an Earthdata
login. Tiles of other servers work if DEM_URL_TEMPLATE is changed (the
placeholders are {name}, {NS}, {lat}, {EW}, {lon}); a tile that exists in
DEM_CACHE_DIR (.hgt, .hgt.gz, .hgt.zip) is never downloaded again, so tiles
can also be put there by hand. Check the attribution of the sources (the
default credit line is DEM_CREDIT).

Usage
-----
    python femtic_plot_sitemap.py                  # settings below
    python femtic_plot_sitemap.py -w /data/run01   # another WORK_DIR
    python femtic_plot_sitemap.py --dem none       # no DEM (no download)
    python femtic_plot_sitemap.py -i last          # last iteration
    python femtic_plot_sitemap.py -w /data/ens     # ensemble root: median of the members
    python femtic_plot_sitemap.py -w /data/ens --stat std   # spread of the members
    python femtic_plot_sitemap.py --runs "kra*/" --stat p90 # members by glob, 90th percentile

All relative paths of the settings refer to WORK_DIR. Return code 0 on
success, 1 if the data could not be read or nothing could be plotted, 2 for a
wrong usage (WORK_DIR missing).

@author vrath
@project py4mt
@created 2026-11-21
@modified 2026-11-21

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

Reference
---------
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
import sys
import urllib.error
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

# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------
# --- working directory and input ---
# Every relative path below is taken from WORK_DIR (an absolute path, or one
# that starts with ~, is used as given). None or "" = the current directory.
WORK_DIR: Optional[str] = "./"
RESULTS_PATH: str = "."               # run directory (searched for ITER_PATTERN) or a results file
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
RUN_PATHS: Optional[Sequence[str]] = None   # run directories / results files / globs, e.g. ["kra*/"]
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
DEM_SOURCE: str = "srtm"              # "srtm" (tiles; cache, download), "file" (DEM_FILE), "none"
DEM_URL_TEMPLATE: str = ("https://s3.amazonaws.com/elevation-tiles-prod/skadi/"
                         "{NS}{lat:02d}/{name}.hgt.gz")
DEM_DOWNLOAD: bool = True             # False: use only the tiles in DEM_CACHE_DIR
DEM_CACHE_DIR: str = "srtm_cache"     # tiles are kept here (give an absolute path to share it between runs)
DEM_FILE: Optional[str] = None        # "file": GeoTIFF (needs rasterio), .npz (lon, lat, z) or one .hgt tile
DEM_TIMEOUT: float = 60.0             # [s] per tile
DEM_MAX_TILES: int = 16               # more tiles than this are refused (too large a map)
DEM_MAX_PIXELS: int = 1600            # the DEM is thinned out to at most this many pixels per side
DEM_FILL_MISSING: Optional[float] = 0.0   # elevation of tiles the server does not have (ocean); None = no value
DEM_STRICT: bool = False              # True: a failed download stops the script (else: map without relief)
DEM_STYLE: str = "hillshade"          # "hillshade" (grey) or "terrain" (colour + shading, DEM_CMAP)
DEM_CMAP: str = "gist_earth"
HILLSHADE_AZIMUTH: float = 315.0
HILLSHADE_ALTITUDE: float = 45.0
HILLSHADE_VERT_EXAG: float = 1.0
BASEMAP_CMAP: str = "Greys"           # "hillshade": colours of the shading
BASEMAP_ALPHA: float = 0.9
DEM_CREDIT: Optional[str] = "auto"    # text at the lower right of every map: "auto", None or text

# --- icons ---
SITE_MARKER: str = "^"
SITE_SIZE: float = 80.0               # [pt^2]
SITE_EDGECOLOR: str = "black"
SITE_LW: float = 0.8
SITE_NODATA_EDGECOLOR: str = "0.35"   # open icons: sites without data in the selection
SITE_LABELS: bool = False             # site names next to the icons
LABEL_FONTSIZE: float = 6.0
CMAP_NRMS: str = "RdBu_r"             # diverging colour map, centred on NRMS_REFERENCE
CMAP_SPREAD: str = "YlOrRd"           # ENS_STAT = "std": sequential colour map, linear scale from 0
NRMS_REFERENCE: Optional[float] = 1.0   # target nRMS = centre colour; None = linear scale
NRMS_VMIN: Optional[float] = None     # colour limits; None = from the data ("std": 0)
NRMS_VMAX: Optional[float] = None

# --- figures and output ---
PAGE_BY: str = "band"                 # one figure per "band" (panels: components), per "component"
                                      # (panels: bands), per "none" (one map each), or "all" (one figure)
PANEL_COLS: int = 2                   # panels per row
PANEL_SIZE: Tuple[float, float] = (4.8, 4.4)   # [inch] of one panel
TITLE: Optional[str] = "auto"         # "auto", None / "" (none) or text with {iter}, {nrms}, {M}, {stat}
CBAR_LOCATION: Optional[str] = None   # None / "right", "left", "top", "bottom"
CBAR_SHRINK: float = 0.8
TICK_FONTSIZE: float = 8.0
DPI: int = 200
SHOW: bool = False
OUTPUT_DIR: str = "sitemap"           # relative to WORK_DIR
OUTPUT_BASENAME: str = "femtic_sitemap"
OUTPUT_FORMATS: Sequence[str] = ["png"]

# ---------------------------------------------------------------------------
# End of the configuration
# ---------------------------------------------------------------------------
_SETTING_NAMES = tuple(k for k in list(globals())
                       if k.isupper() and not k.startswith("_") and k != "PY4MTX_ROOT")

_TAG = "femtic_plot_sitemap:"
_EARTH_RADIUS_M = 6371008.8
_KM_PER_DEG = _EARTH_RADIUS_M * math.pi / 180.0 / 1000.0     # one degree of latitude [km]
_HGT_EXTENSIONS = (".hgt", ".hgt.gz", ".hgt.zip", ".SRTMGL1.hgt.zip", ".SRTMGL3.hgt.zip")
_AWS_SKADI_PREFIX = "https://s3.amazonaws.com/elevation-tiles-prod/skadi/"
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
    if cfg["SITE_DAT"] and cfg["SITE_DAT"] != "auto":
        cfg["SITE_DAT"] = anchor(cfg["SITE_DAT"], wd)
    runs = cfg["RUN_PATHS"]
    if runs:
        cfg["RUN_PATHS"] = [anchor(p, wd) for p in ([runs] if isinstance(runs, str) else runs)]
    else:
        cfg["RUN_PATHS"] = None
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
    one_of("DEM_SOURCE", ("srtm", "file", "none"))
    one_of("DEM_STYLE", ("hillshade", "terrain"))
    one_of("PAGE_BY", _PAGE_BY)
    one_of("CBAR_LOCATION", _CBAR_LOCATIONS)
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
    if int(cfg["PANEL_COLS"]) < 1:
        raise ValueError("PANEL_COLS must be >= 1")
    if int(cfg["DEM_MAX_PIXELS"]) < 2:
        raise ValueError("DEM_MAX_PIXELS must be >= 2")
    parse_stat(cfg["ENS_STAT"])
    if cfg["MEMBER_NRMS_MAX"] is not None and not float(cfg["MEMBER_NRMS_MAX"]) > 0:
        raise ValueError("MEMBER_NRMS_MAX must be > 0 (or None)")


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


def load_dem_srtm(lon_min: float, lon_max: float, lat_min: float, lat_max: float, *,
                  cache_dir: str, template: str, download: bool = True, timeout: float = 60.0,
                  max_tiles: int = 16, fill: Optional[float] = 0.0,
                  strict: bool = False) -> Optional[dict]:
    """
    DEM of the box from SRTM-format tiles (see fetch_tile, mosaic_tiles). None if
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
    return mosaic_tiles(got, fill=fill, sea=sea)


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


def load_dem_file(path: str) -> dict:
    """
    DEM from a file: .npz (arrays lon, lat, z with z[lat, lon]), GeoTIFF (.tif /
    .tiff; needs rasterio; north-up, geographic coordinates) or one SRTM tile
    (.hgt, .hgt.gz, .hgt.zip, named like N37W105...). Returns dict(lon, lat, z)
    as mosaic_tiles does.
    """
    low = path.lower()
    if low.endswith(".npz"):
        with np.load(path) as d:
            return _orient(d["lon"], d["lat"], d["z"])
    if low.endswith((".tif", ".tiff")):
        try:
            import rasterio
        except ImportError as exc:
            raise ImportError("reading a GeoTIFF needs rasterio (not installed); use an .npz "
                              "file, SRTM tiles, or DEM_SOURCE = \"srtm\"") from exc
        with rasterio.open(path) as src:
            z = src.read(1).astype(np.float32)
            tr = src.transform
            if src.nodata is not None:
                z[z == np.float32(src.nodata)] = np.nan
            if src.crs is not None and not src.crs.is_geographic:
                raise ValueError(f"{path}: the coordinates are not geographic (lon / lat)")
            if tr.b != 0.0 or tr.d != 0.0:
                raise ValueError(f"{path}: rotated raster")
            lon = tr.c + tr.a * (np.arange(z.shape[1]) + 0.5)
            lat = tr.f + tr.e * (np.arange(z.shape[0]) + 0.5)
        return _orient(lon, lat, z)
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
def draw_panel(ax, *, frame: dict, bbox: Tuple[float, float, float, float], relief: Optional[dict],
               relief_extent: Optional[Tuple[float, float, float, float]],
               lon: np.ndarray, lat: np.ndarray, values: np.ndarray,
               names: Optional[Sequence[str]], cmap, norm, cfg: Dict[str, Any],
               credit: Optional[str] = None):
    """
    One map: relief, then the sites as icons coloured by `values` (NaN = open
    icon, no data). Returns the scatter of the coloured icons (None if there is
    none). bbox = (lon_min, lon_max, lat_min, lat_max) of the map.
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
        ax.text(0.99, 0.01, credit, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=5.0, color="0.15", zorder=7, path_effects=halo)
    return sc


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


def plot_page(page: Sequence[dict], *, site_lon: np.ndarray, site_lat: np.ndarray,
              site_names: Sequence[str], frame: dict,
              bbox: Tuple[float, float, float, float], relief: Optional[dict],
              relief_extent: Optional[Tuple[float, float, float, float]], norm, extend: str,
              cfg: Dict[str, Any], title: str, credit: Optional[str],
              cbar_label: str = "nRMS"):
    """
    One figure with the maps of the selections of `page`; returns the figure. The
    second line of a panel title is s["sub"] if the selection has one (ensemble),
    else the global nRMS and n.
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
                   norm=norm, cfg=cfg, credit=credit)
        sub = s.get("sub") or f"global nRMS = {s['nrms']:.2f}, n = {s['n']}"
        ax.set_title(f"{s['band']} | {s['comp']}\n{sub}", fontsize=float(cfg["TICK_FONTSIZE"]) + 1)
    for k in range(n, nr * nc):
        axs[k // nc][k % nc].set_visible(False)
    loc = cfg["CBAR_LOCATION"] or "right"
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=used, location=loc, shrink=float(cfg["CBAR_SHRINK"]),
                      extend=extend, label=cbar_label)
    ref = cfg["NRMS_REFERENCE"]
    if ref is not None and norm.vmin <= ref <= norm.vmax:
        (cb.ax.axhline if loc in ("right", "left") else cb.ax.axvline)(
            ref, color="black", lw=0.8, ls="--")
    if any_open:
        from matplotlib.lines import Line2D
        used[0].legend(handles=[Line2D([], [], ls="", marker=cfg["SITE_MARKER"], mfc="none",
                                       mec=cfg["SITE_NODATA_EDGECOLOR"], label="no data")],
                       loc="upper left", fontsize=float(cfg["TICK_FONTSIZE"]) - 1,
                       framealpha=0.8)
    if title:
        fig.suptitle(title, fontsize=float(cfg["TICK_FONTSIZE"]) + 3)
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

    # --- output: tables ---
    out_dir = cfg["OUTPUT_DIR"]
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, cfg["OUTPUT_BASENAME"])
    files = write_site_tables(stem, site_rows, sels, used)
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
    txt = format_summary(sels, header=header)
    with open(stem + "_summary.txt", "w") as fh:
        fh.write(txt)
    files.append(stem + "_summary.txt")
    print(txt)

    # --- basemap ---
    lon_s = np.array([site_rows[j]["lon"] for j in used])
    lat_s = np.array([site_rows[j]["lat"] for j in used])
    try:
        bbox = map_extent(lon_s, lat_s, mode=cfg["MAP_EXTENT_MODE"], box=cfg["MAP_BOX"],
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
    frame = make_frame(cfg["DISPLAY_COORDS"], 0.5 * (bbox[0] + bbox[1]), 0.5 * (bbox[2] + bbox[3]))
    relief = None
    relief_extent = None
    credit = None
    if cfg["DEM_SOURCE"] != "none":
        try:
            if cfg["DEM_SOURCE"] == "srtm":
                dem = load_dem_srtm(*bbox, cache_dir=cfg["DEM_CACHE_DIR"],
                                    template=cfg["DEM_URL_TEMPLATE"],
                                    download=cfg["DEM_DOWNLOAD"], timeout=cfg["DEM_TIMEOUT"],
                                    max_tiles=cfg["DEM_MAX_TILES"], fill=cfg["DEM_FILL_MISSING"],
                                    strict=cfg["DEM_STRICT"])
            else:
                dem = load_dem_file(cfg["DEM_FILE"])
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
            credit = ("Relief: SRTM-format AWS terrain tiles (SRTM courtesy of USGS)"
                      if cfg["DEM_SOURCE"] == "srtm"
                      and str(cfg["DEM_URL_TEMPLATE"]).startswith(_AWS_SKADI_PREFIX) else None)

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
                        cfg=pcfg, title=title, credit=credit, cbar_label=cbar_label)
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
    """Command line: -w / --work-dir, -i / --iteration, --dem, --runs, --stat."""
    ap = argparse.ArgumentParser(
        prog="femtic_plot_sitemap.py",
        description="Site map of a FEMTIC run or ensemble: nRMS per site, band and component on "
                    "an SRTM shaded relief. All paths are taken relative to WORK_DIR.")
    ap.add_argument("-w", "--work-dir", default=None, metavar="DIR",
                    help="base of all relative paths (replaces WORK_DIR)")
    ap.add_argument("-i", "--iteration", default=None, metavar="N",
                    help="\"best\", \"last\" or a number (replaces ITERATION)")
    ap.add_argument("--dem", default=None, choices=("srtm", "file", "none"),
                    help="DEM source (replaces DEM_SOURCE)")
    ap.add_argument("--runs", nargs="+", default=None, metavar="PATH",
                    help="members of the ensemble: run directories, results files or globs "
                         "(replaces RUN_PATHS)")
    ap.add_argument("--stat", default=None, metavar="STAT",
                    help="statistic over the members: mean, median, min, max, std or pNN "
                         "(replaces ENS_STAT)")
    args = ap.parse_args(list(argv))
    over: Dict[str, Any] = {}
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
