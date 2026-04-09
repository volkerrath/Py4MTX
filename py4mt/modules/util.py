#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
util.py
=======
General-purpose utility functions for the Py4MT package.

Provides helpers for:
- HDF5 workspace save/load (MATLAB-like)
- Module introspection and runtime environment detection
- Coordinate projections (pyproj-based: WGS84 ↔ UTM ↔ ITM ↔ Gauss-Kruger)
- File and string manipulation
- Archive unpacking and packing (zip, tar, gz, bz2, xz)
- Sequential script/command queue runner with glob expansion and logging
- Grid generation (lat/lon and UTM)
- Geometry (point-in-polygon, projection onto lines)
- Numerical utilities (KL divergence, L-curve corner, DCT, curvature)
- Miscellaneous (PDF catalog generation, symlink/copy helpers)
- Petrophysical resistivity & permeability models (Archie, Simandoux,
  dual-porosity, RGPZ, Hashin-Shtrikman; brine conductivity via Sen & Goode)

Author: Volker Rath (DIAS)
Created: 2020-11-01
Modified: 2026-03-25 — added section headers; docstrings for undocumented functions; get_percentile verbose parameter; cleanup; Claude Sonnet 4.6 (Anthropic)
Modified: 2026-03-26 — added unpack_compressed(), pack_compressed(), run_queue(); Claude Sonnet 4.6 (Anthropic)
Modified: 2026-04-02 — merged resistivity_models.py (petrophysical models); Claude Sonnet 4.6 (Anthropic)
"""

from __future__ import annotations

import os
import sys
import ast
import fnmatch
import inspect
import math
import pathlib
import shutil
import bz2
import gzip
import lzma
import tarfile
import zipfile
import h5py
import json
import re

import numpy as np

import pyproj
from pyproj import CRS, database, Transformer
from scipy.fftpack import dct, idct

from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import List, Any, Dict, Optional, Iterable
from pathlib import Path


# ---------------------------------------------------------------------------
# Path and sys.path helpers
# ---------------------------------------------------------------------------

def add_tree(path: str):
    """Recursively add all subdirectories of *path* to ``sys.path``."""
    base = Path(path)
    for p in base.rglob("*"):
        if p.is_dir():
            sys.path.append(str(p))

def _is_hdf5_compatible(value: Any) -> bool:
    """Return True if the object can be stored directly in HDF5."""
    return (
        isinstance(value, (np.ndarray, np.number, str, bytes, float, int))
    )


def _is_json_compatible(value: Any) -> bool:
    """Return True if the object can be JSON-serialized."""
    try:
        json.dumps(value)
        return True
    except Exception:
        return False


def _filter_namespace(ns: Dict[str, Any]) -> Dict[str, Any]:
    """Filter out private names, modules, and callables."""
    out = {}
    for k, v in ns.items():
        if k.startswith("_"):
            continue
        if isinstance(v, ModuleType):
            continue
        if callable(v):
            continue
        out[k] = v
    return out



# ---------------------------------------------------------------------------
# HDF5 workspace persistence
# ---------------------------------------------------------------------------

def save_workspace_hdf5(filename: str = "workspace.h5",
                        namespace: Dict[str, Any] = None) -> None:
    """
    Save a filtered namespace to an HDF5 file.

    Parameters
    ----------
    filename : str
        Output HDF5 file.
    namespace : dict, optional
        Namespace to save (defaults to globals()).
    """
    if namespace is None:
        namespace = globals()

    data = _filter_namespace(namespace)

    with h5py.File(filename, "w") as h5:
        for key, value in data.items():
            if _is_hdf5_compatible(value):
                h5.create_dataset(key, data=value)
            elif _is_json_compatible(value):
                h5.create_dataset(key, data=json.dumps(value))
            else:
                print(f"[workspace_hdf5] Skipping non-serializable: {key}")


def load_workspace_hdf5(filename: str = "workspace.h5",
                        namespace: Dict[str, Any] = None) -> None:
    """
    Load variables from an HDF5 file into a namespace.

    Parameters
    ----------
    filename : str
        Input HDF5 file.
    namespace : dict, optional
        Namespace to update (defaults to globals()).
    """
    if namespace is None:
        namespace = globals()

    with h5py.File(filename, "r") as h5:
        for key in h5.keys():
            raw = h5[key][()]
            # Try JSON decode
            if isinstance(raw, (bytes, str)):
                try:
                    namespace[key] = json.loads(raw)
                    continue
                except Exception:
                    pass
            namespace[key] = raw



# ---------------------------------------------------------------------------
# Module introspection and runtime environment
# ---------------------------------------------------------------------------

def list_module_callables(module: ModuleType, public_only: bool = False) -> List[str]:
    """
    Return a list of callable objects defined in a module.

    Parameters
    ----------
    module : ModuleType
        The module to inspect.
    public_only : bool, optional
        If True, exclude names starting with '_'.

    Returns
    -------
    List[str]
        Sorted list of callable names defined in the module.
    """
    callables = [
        name for name, obj in inspect.getmembers(module)
        if callable(obj) and obj.__module__ == module.__name__
    ]

    if public_only:
        callables = [name for name in callables if not name.startswith("_")]

    return sorted(callables)


def running_in_notebook() -> bool:
    """
    Return True if running inside a Jupyter notebook / JupyterLab / qtconsole kernel.
    Return False for plain Python, scripts, and most terminals.
    """
    try:
        from IPython import get_ipython  # type: ignore
        ip = get_ipython()
        if ip is None:
            return False
        # Kernel is usually present in notebook/lab/qtconsole
        return "IPKernelApp" in ip.config
    except Exception:
        return False


def runtime_env() -> str:
    """
    Detect the current Python runtime environment.

    Returns
    -------
    str
        One of 'spyder', 'jupyter', 'ipython-terminal', 'ipython-<name>', or 'python'.
    """
    # Spyder
    if os.environ.get("SPYDER_KERNEL") == "True" or "spyder_kernels" in sys.modules:
        return "spyder"

    # Jupyter / notebook-like
    try:
        from IPython import get_ipython  # type: ignore
        ip = get_ipython()
        if ip is None:
            return "python"
        name = ip.__class__.__name__
        if name == "ZMQInteractiveShell":
            return "jupyter"          # notebook/lab/qtconsole
        if name == "TerminalInteractiveShell":
            return "ipython-terminal"
        return f"ipython-{name}"
    except Exception:
        return "python"



# ---------------------------------------------------------------------------
# Miscellaneous small helpers
# ---------------------------------------------------------------------------

def stop(s: str = ''):
    '''
    Simple stopping utility.

    Parameters
    ----------
    s : str, optional
        Additional string for output. The default is ''.

    Returns
    -------
    None.
    '''
    sys.exit('Execution stopped. ' + s)


def dd(lat, lon):
    """Convert DMS strings (``"DD:MM:SS"``) or plain floats to decimal degrees.

    Parameters
    ----------
    lat, lon : str or float
        Latitude and longitude in ``"DD:MM:SS"`` format or already as float.

    Returns
    -------
    lat_dd, lon_dd : float
        Decimal degrees.
    """
    if ":" in lat:
        deg, minute, sec = (float(p) for p in lat.split(":")[:3])
        lat_dd = deg + minute / 60.0 + sec / 3600.0
    else:
        lat_dd = lat

    if ":" in lon:
        deg, minute, sec = (float(p) for p in lon.split(":")[:3])
        lon_dd = deg + minute / 60.0 + sec / 3600.0
    else:
        lon_dd = lon

    return lat_dd, lon_dd


def nan_like(a: np.ndarray, dtype: type | None = None) -> np.ndarray:
    """
    Create an array of NaN values with the same shape as `a`.

    Parameters
    ----------
    a : np.ndarray
        Reference array.
    dtype : type, optional
        Desired dtype. If None, uses dtype of `a`.

    Returns
    -------
    arr : np.ndarray
        Array of NaN values with same shape and dtype.
    """
    # Default to float if dtype of a cannot represent NaN
    if dtype is None:
        dtype = a.dtype
        if np.issubdtype(dtype, np.integer):
            dtype = float

    return np.full_like(a, np.nan, dtype=dtype)


def dictget(d, *k):
    '''Get the values corresponding to the given keys in the provided dict.'''
    return [d[i] for i in k]

def parse_ast(filename):
    """Parse a Python source file and return its AST."""
    with open(filename, 'rt') as file:
        return ast.parse(file.read(), filename=filename)

def check_env(envar='CONDA_PREFIX', action='error'):
    '''
    Check if environment variable exists

    Parameters
    ----------
    envar : strng, optional
        The default is ['CONDA_PREFIX'].

    Returns
    -------
    None.

    '''
    act_env = os.environ[envar]
    if len(act_env)>0:
        print('\n\n')
        print('Active conda Environment  is:  ' + act_env)
        print('\n\n')
    else:
        if 'err' in action.lower():
            sys.exit('Environment '+ act_env+'is not activated! Exit.')


def find_functions(body):
    """Yield ``ast.FunctionDef`` nodes from an AST body."""
    return (f for f in body if isinstance(f, ast.FunctionDef))


def list_functions(filename):
    '''
    Generate list of functions in module.

    author: VR 3/21
    '''

    print(filename)
    tree = parse_ast(filename)
    for func in find_functions(tree.body):
        print('  %s' % func.name)


# ---------------------------------------------------------------------------
# File and string utilities
# ---------------------------------------------------------------------------

def get_filelist(searchstr=['*'], searchpath='./', sortedlist =True, fullpath=False):
    '''
    Generate filelist from path and unix wildcard list.

    author: VR 3/20

    last change 4/23
    '''

    filelist = fnmatch.filter(os.listdir(searchpath), '*')
    print('\n ')
    print(filelist)
    for sstr in searchstr:
        filelist = fnmatch.filter(filelist, sstr)

    filelist = [os.path.basename(f) for f in filelist]

    if sortedlist:
        filelist = sorted(filelist)
        print(filelist)
    if fullpath:
       filelist = [os.path.join(searchpath,filelist[ii]) for ii in range(len(filelist))]

    print(filelist)
    return filelist


def get_percentile(nsig=1, verbose=True):
    """Return lower/upper normal-distribution percentiles for ±*nsig* sigma.

    Parameters
    ----------
    nsig : float
        Number of standard deviations (default 1).
    verbose : bool
        Print coverage and percentile values when ``True`` (default).

    Returns
    -------
    lower, upper, coverage : float
    """
    import scipy.stats as st
    lower = st.norm.cdf(-nsig)
    upper = st.norm.cdf( nsig)
    coverage = upper - lower
    if verbose:
        print('Coverage', round(coverage), 'percentiles=', lower, upper)
    return lower, upper, coverage


# ---------------------------------------------------------------------------
# Coordinate projections (pyproj)
# ---------------------------------------------------------------------------

def get_utm_zone(lat=None, lon=None):
    '''
    Get utm-zone from lat and lon

    Typical usage:

        epsg = get_utm_epsg(lon, lat)
        crs = CRS.from_epsg(epsg)
        transformer = Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True)
        x, y = transformer.transform(lon, lat)

    For more accuracy, cross‑border work, or points near zone boundaries,
    consider querying local/national projected CRSs instead.

    Parameters
    ----------
    lat : float
        latitude . The default is None.
    lon : float
        longitude. The default is None.


    Returns
    -------
    epsg : int
        EPSG value for utm-zone.

    Raises
    ------
        ValueError if invalid EPSG.


    vr 10/25 + copilot

    '''

    # normalize longitude to (-180, 180]
    lon = ((lon + 180) % 360) - 180
    if abs(lat) > 84.0:
        raise ValueError("UTM undefined for |lat| > 84 degrees")
    zone = int((math.floor((lon + 180) / 6) % 60) + 1)
    base = 32600 if lat >= 0 else 32700
    epsg = base + zone
    # validate
    CRS.from_epsg(epsg)  # will raise if invalid
    return epsg

def get_local_crs(lon=None, lat=None, buffer_deg=1.0, max_results= 10):
    """
    Query pyproj database for projected CRSs overlapping a bbox around the point.
    Returns a list of dicts: [{'epsg': int, 'name': str, 'extent': (west,south,east,north)}...]
    buffer_deg is the half-width/half-height of the bbox in degrees (approx).

    vr 10/25 + copilot
    """
    # build zone of interest (west, south, east, north)
    west = lon - buffer_deg
    east = lon + buffer_deg
    south = lat - buffer_deg
    north = lat + buffer_deg

    # query CRSs from the pyproj database (authority = 'EPSG')
    info_list = database.query_crs_info(auth_name='EPSG', bbox=(west, south, east, north))
    results = []
    for info in info_list[:max_results]:
        # info is a pyproj.database.CRSInfo object with attributes: name, code, auth_name, area_of_use, bbox
        try:
            epsg_code = int(info.code)
            crs = CRS.from_epsg(epsg_code)
            # keep only projected CRSs (not geographic)
            if crs.is_projected:
                results.append({
                    'epsg': epsg_code,
                    'name': info.name,
                    'area_of_use': info.area_of_use,
                    'bbox': info.bbox
                })
        except Exception:
            continue
    return results

def proj_latlon_to_utm(latitude, longitude, utm_zone=32629):
    '''
    Transform latlon to UTM, using pyproj Transformer.
    Look for other EPSG at https://epsg.io/

    VR 04/21, updated to Transformer API
    '''
    transformer = Transformer.from_crs('epsg:4326', f'epsg:{utm_zone}', always_xy=True)
    utm_x, utm_y = transformer.transform(longitude, latitude)

    return utm_x, utm_y

def proj_utm_to_latlon(utm_x, utm_y, utm_zone=32629):
    '''
    Transform UTM to latlon, using pyproj Transformer.
    Look for other EPSG at https://epsg.io/

    VR 04/21, updated to Transformer API
    '''
    transformer = Transformer.from_crs(f'epsg:{utm_zone}', 'epsg:4326', always_xy=True)
    longitude, latitude = transformer.transform(utm_x, utm_y)
    return latitude, longitude


def proj_latlon_to_itm(longitude, latitude):
    '''
    Transform latlon to ITM, using pyproj Transformer.
    Look for other EPSG at https://epsg.io/

    VR 04/21, updated to Transformer API
    '''
    transformer = Transformer.from_crs('epsg:4326', 'epsg:2157', always_xy=True)
    itm_x, itm_y = transformer.transform(longitude, latitude)
    return itm_x, itm_y


def proj_itm_to_latlon(itm_x, itm_y):
    '''
    Transform ITM to latlon, using pyproj Transformer.
    Look for other EPSG at https://epsg.io/

    VR 04/21, updated to Transformer API
    '''
    transformer = Transformer.from_crs('epsg:2157', 'epsg:4326', always_xy=True)
    longitude, latitude = transformer.transform(itm_x, itm_y)
    return latitude, longitude


def proj_itm_to_utm(itm_x, itm_y, utm_zone=32629):
    '''
    Transform ITM to UTM, using pyproj Transformer.
    Look for other EPSG at https://epsg.io/

    VR 04/21, updated to Transformer API
    '''
    transformer = Transformer.from_crs('epsg:2157', f'epsg:{utm_zone}', always_xy=True)
    utm_x, utm_y = transformer.transform(itm_x, itm_y)
    return utm_x, utm_y


def proj_utm_to_itm(utm_x, utm_y, utm_zone=32629):
    '''
    Transform UTM to ITM, using pyproj Transformer.
    Look for other EPSG at https://epsg.io/

    VR 04/21, updated to Transformer API
    '''
    transformer = Transformer.from_crs(f'epsg:{utm_zone}', 'epsg:2157', always_xy=True)
    itm_x, itm_y = transformer.transform(utm_x, utm_y)
    return itm_x, itm_y

def project_wgs_to_geoid(lat, lon, alt, geoid=3855 ):
    '''
    transform ellipsoid heigth to geoid, using pyproj
    Look for other EPSG at https://epsg.io/

    VR 09/21

    '''

    geoidtrans =pyproj.crs.CompoundCRS(name='WGS 84 + EGM2008 height', components=[4979, geoid])
    wgs = pyproj.Transformer.from_crs(
            pyproj.CRS(4979), geoidtrans, always_xy=True)
    lat, lon, elev = wgs.transform(lat, lon, alt)

    return lat, lon, elev

def project_utm_to_geoid(utm_x, utm_y, utm_z, utm_zone=32629, geoid=3855):
    '''
    transform ellipsoid heigth to geoid, using pyproj
    Look for other EPSG at https://epsg.io/

    VR 09/21

    '''

    geoidtrans =pyproj.crs.CompoundCRS(name='UTM + EGM2008 height', components=[utm_zone, geoid])
    utm = pyproj.Transformer.from_crs(
            pyproj.CRS(utm_zone), geoidtrans, always_xy=True)
    utm_x, utm_y, elev = utm.transform(utm_x, utm_y, utm_z)

    return utm_x, utm_y, elev

def project_gk_to_latlon(gk_x, gk_y, gk_zone=5684):
    '''
    Transform Gauss-Kruger to latlon, using pyproj Transformer.
    Look for other EPSG at https://epsg.io/

    VR 04/21, updated to Transformer API
    '''
    transformer = Transformer.from_crs(f'epsg:{gk_zone}', 'epsg:4326', always_xy=True)
    longitude, latitude = transformer.transform(gk_x, gk_y)
    return latitude, longitude

def splitall(path):
    """Split *path* into all of its components.

    Examples
    --------
    >>> splitall("/a/b/c")
    ['/', 'a', 'b', 'c']
    """
    allparts = []
    while True:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def get_files(SearchString=None, SearchDirectory='.'):
    '''
    FileList = get_files(Filterstring) produces a list
    of files from a searchstring (allows wildcards)

    VR 11/20
    '''
    FileList = fnmatch.filter(os.listdir(SearchDirectory), SearchString)

    return FileList


def unique(seq, out=False):
    '''
    Find unique elements in list/array, preserving order.

    Parameters
    ----------
    seq : list or array-like
        Input sequence.
    out : bool, optional
        If True, print unique elements. The default is False.

    Returns
    -------
    unique_list : list
        List of unique elements in order of first appearance.

    VR 9/20
    '''
    unique_list = []
    for x in seq:
        if x not in unique_list:
            unique_list.append(x)
    # print list
    if out:
        for x in unique_list:
            print(x)

    return unique_list

def bytes2human(n):
    '''
    http://code.activestate.com/recipes/578019
    >>> bytes2human(10000)
    '9.8K'
    >>> bytes2human(100001221)
    '95.4M'
    '''
    symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = {}
    for i, s in enumerate(symbols):
        prefix[s] = 1 << (i + 1) * 10
    for s in reversed(symbols):
        if abs(n) >= prefix[s]:
            value = float(n) / prefix[s]
            return '%.1f%s' % (value, s)
    return '%sB' % n

def strcount(keyword=None, fname=None):
    '''
    count occurences of keyword in file
     Parameters
    ----------
    keywords : TYPE, optional
        DESCRIPTION. The default is None.
    fname : TYPE, optional
        DESCRIPTION. The default is None.

    VR 9/20
    '''
    with open(fname, 'r') as fin:
        return sum([1 for line in fin if keyword in line])
    # sum([1 for line in fin if keyword not in line])


def strdelete(keyword=None, fname_in=None, fname_out=None, out=True):
    '''
    delete lines containing on of the keywords in list

    Parameters
    ----------
    keywords : TYPE, optional
        DESCRIPTION. The default is None.
    fname_in : TYPE, optional
        DESCRIPTION. The default is None.
    fname_out : TYPE, optional
        DESCRIPTION. The default is None.
    out : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    VR 9/20
    '''
    nn = strcount(keyword, fname_in)

    if out:
        print(str(nn) + ' occurances of <' + keyword + '> in ' + fname_in)

    # if fname_out  is None: fname_out= fname_in
    with open(fname_in, 'r') as fin, open(fname_out, 'w') as fou:
        for line in fin:
            if keyword not in line:
                fou.write(line)


def strreplace(key_in=None, key_out=None, fname_in=None, fname_out=None):
    '''
    replaces key_in in keywords by key_out

    Parameters
    ----------
    key_in : TYPE, optional
        DESCRIPTION. The default is None.
    key_out : TYPE, optional
        DESCRIPTION. The default is None.
    fname_in : TYPE, optional
        DESCRIPTION. The default is None.
    fname_out : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    VR 9/20

    '''
    if key_in is None:
        sys.exit('strreplace: input key.missing!')

    if key_out is None:
        sys.exit('strreplace: output key.missing!')

    if fname_in is None:
        sys.exit('strreplace: input file not given!')

    if fname_out is None:
        fname_out = fname_in
        print('strreplace: warning outpu file overwrites input file!')

    with open(fname_in, 'r') as fin, open(fname_out, 'w') as fou:
        for line in fin:
            fou.write(line.replace(key_in, key_out))



# ---------------------------------------------------------------------------
# Archive unpacking
# ---------------------------------------------------------------------------

def unpack_compressed(directories, *, recurse=False, remove_archive=False,
                      verbose=True):
    """Unpack all compressed files found in one or more directories.

    Supported formats: .zip, .tar, .tar.gz / .tgz, .tar.bz2 / .tbz2,
    .tar.xz / .txz, and single-file .gz / .bz2 / .xz.

    Parameters
    ----------
    directories : str | Path | list[str | Path]
        One directory or a list of directories to scan.
    recurse : bool, optional
        If True, also scan sub-directories recursively. Default False.
    remove_archive : bool, optional
        If True, delete each archive after successful extraction. Default False.
    verbose : bool, optional
        Print progress messages. Default True.

    Returns
    -------
    list[Path]
        Paths of all successfully unpacked archives.

    VR 2026-03-26, Claude Sonnet 4.6 (Anthropic)
    """
    if isinstance(directories, (str, Path)):
        directories = [directories]

    def _is_compressed(p):
        try:
            if zipfile.is_zipfile(p):
                return True
            if tarfile.is_tarfile(p):
                return True
        except Exception:
            pass
        s = "".join(p.suffixes).lower()
        return s.endswith((".gz", ".bz2", ".xz"))

    def _extract(p):
        dest = p.parent
        s = "".join(p.suffixes).lower()
        try:
            if zipfile.is_zipfile(p):
                with zipfile.ZipFile(p) as zf:
                    zf.extractall(dest)
            elif tarfile.is_tarfile(p):
                with tarfile.open(p) as tf:
                    tf.extractall(dest)
            elif s.endswith(".gz"):
                out = dest / p.stem
                with gzip.open(p, "rb") as fi, open(out, "wb") as fo:
                    fo.write(fi.read())
            elif s.endswith(".bz2"):
                out = dest / p.stem
                with bz2.open(p, "rb") as fi, open(out, "wb") as fo:
                    fo.write(fi.read())
            elif s.endswith(".xz"):
                out = dest / p.stem
                with lzma.open(p, "rb") as fi, open(out, "wb") as fo:
                    fo.write(fi.read())
            else:
                return False
            return True
        except Exception as exc:
            print(f"  [WARN] could not unpack {p.name}: {exc}")
            return False

    unpacked = []
    for d in directories:
        d = Path(d)
        if not d.is_dir():
            print(f"  [WARN] not a directory, skipping: {d}")
            continue
        pattern = "**/*" if recurse else "*"
        candidates = sorted(f for f in d.glob(pattern) if f.is_file())
        for f in candidates:
            try:
                is_comp = _is_compressed(f)
            except Exception:
                continue
            if not is_comp:
                continue
            if verbose:
                print(f"  Unpacking {f.name} -> {f.parent}/")
            ok = _extract(f)
            if ok:
                unpacked.append(f)
                if remove_archive:
                    f.unlink()
                    if verbose:
                        print(f"    Removed {f.name}")
    if verbose:
        print(f"Unpacked {len(unpacked)} archive(s).")
    return unpacked


def pack_compressed(directories, method="zip", *, outdir=None, archive_name=None,
                    recurse=False, remove_source=False, verbose=True):
    """Pack one or more directories into a single compressed archive.

    Each directory in *directories* produces one archive named after that
    directory (unless *archive_name* overrides this for a single-directory
    call).  Archives are written to *outdir* (default: parent of each source
    directory).

    Supported methods
    -----------------
    ``"zip"``        — ZIP archive (.zip)
    ``"tar"``        — uncompressed tar (.tar)
    ``"tgz"``        — gzip-compressed tar (.tar.gz)
    ``"tbz2"``       — bzip2-compressed tar (.tar.bz2)
    ``"txz"``        — xz-compressed tar (.tar.xz)

    Parameters
    ----------
    directories : str | Path | list[str | Path]
        One directory or a list of directories to archive.
    method : str, optional
        Compression method; one of ``"zip"``, ``"tar"``, ``"tgz"``,
        ``"tbz2"``, ``"txz"``.  Default ``"zip"``.
    outdir : str | Path | None, optional
        Directory where archives are written.  If None, each archive is
        placed next to its source directory. Default None.
    archive_name : str | None, optional
        Override the archive stem for a single-directory call.  Ignored when
        *directories* contains more than one entry. Default None.
    recurse : bool, optional
        If True, include sub-directories recursively.  If False, only files
        directly inside the directory are packed. Default False.
    remove_source : bool, optional
        If True, delete the source directory after successful packing.
        Default False.
    verbose : bool, optional
        Print progress messages. Default True.

    Returns
    -------
    list[Path]
        Paths of all successfully created archives.

    VR 2026-03-26, Claude Sonnet 4.6 (Anthropic)
    """
    _METHODS = {
        "zip":  (".zip",    None),
        "tar":  (".tar",    ""),
        "tgz":  (".tar.gz", "gz"),
        "tbz2": (".tar.bz2","bz2"),
        "txz":  (".tar.xz", "xz"),
    }
    method = method.lower().strip()
    if method not in _METHODS:
        raise ValueError(
            f"pack_compressed: unknown method {method!r}. "
            f"Choose one of: {list(_METHODS)}")

    suffix, tar_mode = _METHODS[method]

    if isinstance(directories, (str, Path)):
        directories = [directories]

    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

    created = []
    for d in directories:
        d = Path(d).resolve()
        if not d.is_dir():
            print(f"  [WARN] not a directory, skipping: {d}")
            continue

        stem = archive_name if (archive_name and len(directories) == 1) else d.name
        dest_dir = outdir if outdir is not None else d.parent
        archive_path = dest_dir / (stem + suffix)

        # Collect files
        pattern = "**/*" if recurse else "*"
        files = sorted(f for f in d.glob(pattern) if f.is_file())
        if not files:
            print(f"  [WARN] no files found in {d}, skipping.")
            continue

        if verbose:
            print(f"  Packing {d.name}/ -> {archive_path.name} "
                  f"({len(files)} file(s))")
        try:
            if method == "zip":
                with zipfile.ZipFile(archive_path, "w",
                                     compression=zipfile.ZIP_DEFLATED) as zf:
                    for f in files:
                        zf.write(f, f.relative_to(d.parent))
            else:
                mode = "w:" + tar_mode if tar_mode else "w"
                with tarfile.open(archive_path, mode) as tf:
                    for f in files:
                        tf.add(f, arcname=f.relative_to(d.parent))

            created.append(archive_path)
            if remove_source:
                shutil.rmtree(d)
                if verbose:
                    print(f"    Removed source {d.name}/")
        except Exception as exc:
            print(f"  [WARN] could not pack {d.name}: {exc}")

    if verbose:
        print(f"Packed {len(created)} archive(s).")
    return created




def run_queue(scripts, *, mode="strict", logfile=None, verbose=True):
    """Run a sequence of scripts or shell commands sequentially.

    Entries in *scripts* may be literal paths/commands or glob patterns.
    Glob patterns are expanded and sorted before execution.  Each entry is
    executed via ``subprocess`` (bash) and its stdout/stderr are streamed
    live and optionally written to a timestamped log file.

    Parameters
    ----------
    scripts : str | list[str]
        One item or a list of items; each item is either:

        - a literal path to a shell script (``"./run_step1.sh"``)
        - a shell command string (``"python process.py --site A01"``)
        - a glob pattern (``"./stage2_*.sh"``, ``"/jobs/step?.sh"``)

        Glob patterns are expanded relative to the current working directory
        and sorted lexicographically before any other entries are appended.
    mode : {"strict", "lenient"}, optional
        ``"strict"``  — stop immediately if any script returns a non-zero
        exit code (default).
        ``"lenient"`` — log failures and continue with remaining scripts.
    logfile : str | Path | None, optional
        Path to the log file.  If None, a timestamped name is generated
        automatically (``run_queue_YYYYMMDD_HHMMSS.log``).  Pass ``False``
        to disable logging entirely. Default None.
    verbose : bool, optional
        If True, print log lines to stdout in addition to the log file.
        Default True.

    Returns
    -------
    dict
        Summary with keys:

        ``"resolved"``  — list[str] of scripts after glob expansion
        ``"ok"``        — list[str] of scripts that exited with code 0
        ``"failed"``    — list[tuple[str, int]] of (script, exit_code) pairs
        ``"logfile"``   — Path | None — path to the log file

    Raises
    ------
    RuntimeError
        In ``"strict"`` mode, raised after the first failing script.

    VR 2026-03-26, Claude Sonnet 4.6 (Anthropic)
    """
    import subprocess
    import glob as _glob
    from datetime import datetime

    if isinstance(scripts, str):
        scripts = [scripts]

    # ---- glob expansion ------------------------------------------------
    resolved = []
    for entry in scripts:
        if any(c in entry for c in ("*", "?", "[")):
            matches = sorted(_glob.glob(entry))
            if not matches:
                _rq_log(f"[WARN] glob matched nothing: {entry}",
                        fh=None, verbose=verbose)
            else:
                resolved.extend(matches)
        else:
            resolved.append(entry)

    # ---- log file setup ------------------------------------------------
    fh = None
    logpath = None
    if logfile is not False:
        if logfile is None:
            logpath = Path(f"run_queue_{datetime.now():%Y%m%d_%H%M%S}.log")
        else:
            logpath = Path(logfile)
        fh = open(logpath, "w", buffering=1)

    def _log(msg):
        _rq_log(msg, fh=fh, verbose=verbose)

    _log(f"run_queue started. mode={mode}, scripts={len(resolved)}")
    for s in resolved:
        _log(f"  {s}")

    # ---- execution loop ------------------------------------------------
    ok, failed = [], []
    for script in resolved:
        _log(f">>> Starting: {script}")
        try:
            proc = subprocess.Popen(
                ["bash", script] if Path(script).exists() else script,
                shell=not Path(script).exists(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                _log(line.rstrip())
            proc.wait()
            rc = proc.returncode
        except Exception as exc:
            _log(f"[WARN] could not launch {script!r}: {exc}")
            failed.append((script, -1))
            if mode == "strict":
                if fh:
                    fh.close()
                raise RuntimeError(
                    f"run_queue: launch failed for {script!r}") from exc
            continue

        if rc == 0:
            _log(f">>> Finished OK: {script}")
            ok.append(script)
        else:
            _log(f"!!! {script} exited with code {rc}")
            failed.append((script, rc))
            if mode == "strict":
                _log("Stopping execution due to error.")
                if fh:
                    fh.close()
                raise RuntimeError(
                    f"run_queue: {script!r} exited with code {rc}")

    _log(f"run_queue finished. ok={len(ok)}, failed={len(failed)}")
    if fh:
        fh.close()

    return {"resolved": resolved, "ok": ok, "failed": failed,
            "logfile": logpath}


def _rq_log(msg, *, fh=None, verbose=True):
    """Internal timestamped logger for :func:`run_queue`."""
    from datetime import datetime
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} - {msg}"
    if verbose:
        print(line)
    if fh is not None:
        fh.write(line + "\n")


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

def gen_grid_latlon(
        LatLimits=None,
        nLat=None,
        LonLimits=None,
        nLon=None,
        out=True):
    '''
     Generates equidistant 1-d grids in latLong.

     VR 11/20
    '''
    small = 0.000001
# LonLimits = ( 6.275, 6.39)
# nLon = 31
    LonStep = (LonLimits[1] - LonLimits[0]) / nLon
    Lon = np.arange(LonLimits[0], LonLimits[1] + small, LonStep)

# LatLimits = (45.37,45.46)
# nLat = 31
    LatStep = (LatLimits[1] - LatLimits[0]) / nLat
    Lat = np.arange(LatLimits[0], LatLimits[1] + small, LatStep)

    return Lat, Lon


def gen_grid_utm(XLimits=None, nX=None, YLimits=None, nY=None, out=True):
    '''
     Generates equidistant 1-d grids in m.

     VR 11/20
    '''

    small = 0.000001
# LonLimits = ( 6.275, 6.39)
# nLon = 31
    XStep = (XLimits[1] - XLimits[0]) / nX
    X = np.arange(XLimits[0], XLimits[1] + small, XStep)

# LatLimits = (45.37,45.46)
# nLat = 31
    YStep = (YLimits[1] - YLimits[0]) / nY
    Y = np.arange(YLimits[0], YLimits[1] + small, YStep)

    return X, Y



# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def choose_data_poly(Data=None, PolyPoints=None, Out=True):
    '''
     Chooses polygon area from data set, given
     PolyPoints = [[X1 Y1,...[XN YN]]. First and last points will
     be connected for closure.

     VR 11/20
    '''
    if Data.size == 0:
        sys.exit('No Data given!')
    if not PolyPoints:
        sys.exit('No Rectangle given!')

    Ddims = np.shape(Data)
    if Out:
        print('data matrix input: ' + str(Ddims))

    Poly = []
    for row in np.arange(Ddims[0] - 1):
        if point_inside_polygon(Data[row, 1], Data[row, 1], PolyPoints):
            Poly.append(Data[row, :])

    Poly = np.asarray(Poly, dtype=float)
    if Out:
        Ddims = np.shape(Poly)
        print('data matrix output: ' + str(Ddims))

    return Poly

# potentially faster:
# from shapely.geometry import Point
# from shapely.geometry.polygon import Polygon

# lons_lats_vect = np.column_stack((lons_vect, lats_vect)) # Reshape coordinates
# polygon = Polygon(lons_lats_vect) # create polygon
# point = Point(y,x) # create point
# print(polygon.contains(point)) # check if polygon contains point
# print(point.within(polygon)) # check if a point is in the polygon

# @jit(nopython=True)


def point_inside_polygon(x, y, poly):
    '''
    Determine if a point is inside a given polygon or not, where
    the Polygon is given as a list of (x,y) pairs.
    Returns True  when point (x,y) ins inside polygon poly, False otherwise

    '''
    # @jit(nopython=True)
    n = len(poly)
    inside = False

    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def choose_data_rect(Data=None, Corners=None, Out=True):
    '''
     Chooses rectangular area from aempy data set, giveb
     the left lower and right uper corners in m as [minX maxX minY maxY]

    '''
    if Data.size == 0:
        sys.exit('No Data given!')
    if not Corners:
        sys.exit('No Rectangle given!')

    Ddims = np.shape(Data)
    if Out:
        print('data matrix input: ' + str(Ddims))
    Rect = []
    for row in np.arange(Ddims[0] - 1):
        if (Data[row, 1] > Corners[0] and Data[row, 1] < Corners[1] and
                Data[row, 2] > Corners[2] and Data[row, 2] < Corners[3]):
            Rect.append(Data[row, :])
    Rect = np.asarray(Rect, dtype=float)
    if Out:
        Ddims = np.shape(Rect)
        print('data matrix output: ' + str(Ddims))

    return Rect


def proj_to_line(x, y, line):
    '''
    Projects a point onto a line, where line is represented by two arbitrary
    points. as an array
    '''
#    http://www.vcskicks.com/code-snippet/point-projection.php
#    private Point Project(Point line1, Point line2, Point toProject)
# {
#    double m = (double)(line2.Y - line1.Y) / (line2.X - line1.X);
#    double b = (double)line1.Y - (m * line1.X);
#
#    double x = (m * toProject.Y + toProject.X - m * b) / (m * m + 1);
#    double y = (m * m * toProject.Y + m * toProject.X + b) / (m * m + 1);
#
#    return new Point((int)x, (int)y);
# }
    x1 = line[0, 0]
    x2 = line[1, 0]
    y1 = line[0, 1]
    y2 = line[1, 1]
    m = (y2 - y1) / (x2 - x1)
    b = y1 - (m * x1)

    xn = (m * y + x - m * b) / (m * m + 1.)
    yn = (m * m * y + m * x + b) / (m * m + 1.)

    return xn, yn


def gen_searchgrid(Points=None,
                   XLimits=None, dX=None, YLimits=None, dY=None, Out=False):
    '''
    Generate equidistant grid for searching (in m).

    VR 02/21
    '''
    small = 0.1

    datax = Points[:, 0]
    datay = Points[:, 1]
    nD = np.shape(Points)[0]

    X = np.arange(np.min(XLimits), np.max(XLimits) + small, dX)
    nXc = np.shape(X)[0]-1
    Y = np.arange(np.min(YLimits), np.max(YLimits)+ small, dY)
    nYc = np.shape(Y)[0]-1
    if Out:
        print('Mesh size: '+str(nXc)+'X'+str(nYc)
              +'\nCell sizes: '+str(dX)+'X'+str(dY)
              +'\nNuber of data = '+str(nD))


    p = np.zeros((nXc, nYc), dtype=object)
    # print(np.shape(p))


    xp= np.digitize(datax, X, right=False)
    yp= np.digitize(datay, Y, right=False)

    for ix in np.arange (nXc):
        incol = np.where(xp == ix)[0]
        for iy in np.arange (nYc):
            rlist = incol[np.where(yp[incol] == iy)[0]]
            p[ix,iy]=rlist

            if Out:
               print('mesh cell: '+str(ix)+' '+str(iy))

    return p



# ---------------------------------------------------------------------------
# Numerical utilities
# ---------------------------------------------------------------------------

def KLD(P=np.array([]), Q=np.array([]), epsilon= 1.e-8):
    '''
    Calculates Kullback-Leibler distance

    Parameters
    ----------
    P, Q: np.array
        pdfs
    epsilon : TYPE
        Epsilon is used here to avoid conditional code for
        checking that neither P nor Q is equal to 0.

    Returns
    -------

    distance: float
        KL distance


    '''
    if P.size * Q.size==0:
        sys.exit('KLD: P or Q not defined! Exit.')

    # You may want to instead make copies to avoid changing the np arrays.
    PP = P.copy()+epsilon
    QQ = Q.copy()+epsilon

    distance = np.sum(PP*np.log(PP/QQ))

    return distance

def dctn(x, normused='ortho'):
    '''
    Discrete cosine transform (fwd)
    https://stackoverflow.com/questions/13904851/use-pythons-scipy-dct-ii-to-do-2d-or-nd-dct
    '''
    for i in range(x.ndim):
        x = dct(x, axis=i, norm=normused)
    return x

def idctn(x, normused='ortho'):
    '''
    Discrete cosine transform (inv)
    https://stackoverflow.com/questions/13904851/use-pythons-scipy-dct-ii-to-do-2d-or-nd-dct
    '''
    for i in range(x.ndim):
        x = idct(x, axis=i, norm=normused)
    return x

def fractrans(m=None, x=None , a=0.5):
    '''
    Caklculate fractional derivative of m.

    VR Apr 2021
    '''
    import differint as df

    if m  is None or x  is None:
        sys.exit('No vector for diff given! Exit.')

    if np.size(m) != np.size(x):
        sys.exit('Vectors m and x have different length! Exit.')

    x0 = x[0]
    x1 = x[-1]
    npnts = np.size(x)
    mm = df.differint(a, m, x0, x1, npnts)

    return mm


def calc_lc_corner(dnorm=np.array([]), mnorm=np.array([])):
    '''
    Calculates corner of thhe L-curve.

    Parameters
    ----------
    dnorm                   data norm
    mnorm                   Generalized inverse times J^T

    Returns
    -------
    lcc_val                 value of gcv function)

    see:

        Per Christian Hansen:
        Discrete Inverse Problems: Insight and Algorithms
        SIAM, Philadelphia, 2010

        Per Christian Hansen:
        The L-Curve and its Use in the Numerical Treatment of Inverse Problems
        In: P. Johnston ,Computational Inverse Problems in Electrocardiology
        WIT Press, 2001
        119-142

        Per Christian Hansen:
        Rank Deficient and Discrete Ill-Posed Problems
        SIAM, Philadelphia, 1998

    VR June 2022
    '''
    if (np.size(dnorm) == 0) or (np.size(mnorm) == 0):
        sys.exit('calc_lcc: parameters missing! Exit.')

    lcurvature = curvature(np.log(dnorm), np.log(mnorm))

    indexmax = np.argmax(lcurvature)

    return indexmax

def curvature(x_data, y_data):
    '''
    Calculates curvature for all interior points
    on a curve whose coordinates are provided
    Used for l-curve corner estimation.
    Input:
        - x_data: list of n x-coordinates
        - y_data: list of n y-coordinates
    Output:
        - curvature: list of n-2 curvature values

    originally written by Hunter Ratliff on 2019-02-03
    '''
    curvature = []
    for i in range(1, len(x_data)-1):
        R = circumradius(x_data[i-1:i+2], y_data[i-1:i+2])
        if (R == 0):
            print('Failed: points are either collinear or not distinct')
            return 0
        curvature.append(1/R)
    return curvature


def circumradius(xvals, yvals):
    '''
    Calculates the circumradius for three 2D points

    originally written by Hunter Ratliff on 2019-02-03
    '''
    x1, x2, x3, y1, y2, y3 = xvals[0], xvals[1], xvals[2], yvals[0], yvals[1], yvals[2]
    den = 2.*((x2-x1)*(y3-y2)-(y2-y1)*(x3-x2))
    num = ((((x2-x1)**2) + ((y2-y1)**2))
           * (((x3-x2)**2)+((y3-y2)**2))
           * (((x1-x3)**2)+((y1-y3)**2)))**(0.5)
    if (den == 0.):
        print('Failed: points are either collinear or not distinct')
        return 0.
    R = abs(num/den)

    return R


def circumcenter(xvals, yvals):
    '''
    Calculates the circumcenter for three 2D points

    originally written by Hunter Ratliff on 2019-02-03
    '''
    x1, x2, x3, y1, y2, y3 = xvals[0], xvals[1], xvals[2], yvals[0], yvals[1], yvals[2]
    A = 0.5*((x2-x1)*(y3-y2)-(y2-y1)*(x3-x2))
    if (A == 0):
        print('Failed: points are either collinear or not distinct')
        return 0
    xnum = ((y3 - y1)*(y2 - y1)*(y3 - y2)) - \
        ((x2**2 - x1**2)*(y3 - y2)) + ((x3**2 - x2**2)*(y2 - y1))
    x = xnum/(-4*A)
    y = (-1*(x2 - x1)/(y2 - y1))*(x-0.5*(x1 + x2)) + 0.5*(y1 + y2)
    return x, y

def calc_resnorm(data_obs=None, data_calc=None, data_std=None, p=2):
    '''
    Calculate the p-norm of the residuals.

    VR Jan 2021

    '''
    if data_std is None:
        data_std = np.ones(np.shape(data_obs))

    resid = (data_obs - data_calc) / data_std

    rnormp = np.power(resid, p)
    rnorm = np.sum(rnormp)
    #    return {'rnorm':rnorm, 'resid':resid }
    return rnorm, resid


def calc_rms(dcalc=None, dobs=None, Wd=1.0):
    '''
    Calculate the NRMS ans SRMS.

    VR Jan 2021

    '''
    sizedat = np.shape(dcalc)
    nd = sizedat[0]
    rscal = Wd * (dobs - dcalc).T
    print(sizedat,nd)
    # normalized root mean square error
    nrms = np.sqrt(np.sum(np.power(abs(rscal), 2)) / (nd - 1))

    # sum squared scaled symmetric error
    serr = 2.0 * nd * np.abs(rscal) / (abs(dobs.T) + abs(dcalc.T))
    ssq = np.sum(np.power(serr, 2))
    # print(ssq)
    srms = 100.0 * np.sqrt(ssq / nd)

    return nrms, srms

def nearly_equal(a, b, sig_fig=6):
    """Return ``True`` if *a* and *b* agree to *sig_fig* significant figures."""
    return (a==b or int(a*10**sig_fig) == int(b*10**sig_fig))



# ---------------------------------------------------------------------------
# Rotation matrices (right-handed, active rotations)
# ---------------------------------------------------------------------------

def rot_z(angle_deg):
    """3×3 rotation matrix about the Z axis by *angle_deg* degrees."""
    t = np.radians(angle_deg)
    return np.array([[ np.cos(t), -np.sin(t), 0.0],
                     [ np.sin(t),  np.cos(t), 0.0],
                     [ 0.0,         0.0,      1.0]])

def rot_x(angle_deg):
    """3×3 rotation matrix about the X axis by *angle_deg* degrees."""
    t = np.radians(angle_deg)
    return np.array([[1.0, 0.0,         0.0       ],
                     [0.0, np.cos(t), -np.sin(t)],
                     [0.0, np.sin(t),  np.cos(t)]])

def rot_y(angle_deg):
    """3×3 rotation matrix about the Y axis by *angle_deg* degrees."""
    t = np.radians(angle_deg)
    return np.array([[ np.cos(t), 0.0, np.sin(t)],
                     [ 0.0,       1.0, 0.0      ],
                     [-np.sin(t), 0.0, np.cos(t)]])

def rot_full(T, angle_deg_x, angle_deg_y, angle_deg_z):
    """Apply combined X/Y/Z rotation to tensor *T*.

    Rotation order: R = rot_z @ rot_y @ rot_x; result = R @ T @ R.T.
    """
    T0 = T.copy()
    # Combined rotation: rot_ = rot_z @ rot_y @ rot_x
    rot = rot_z(angle_deg_z) @ rot_y(angle_deg_y) @ rot_x(angle_deg_x)

    # rotate tensor
    T_rot = rot @ T0 @ rot.T
    return T_rot



# ---------------------------------------------------------------------------
# System / OS helpers
# ---------------------------------------------------------------------------

def make_pdf_catalog(workdir='./', pdflist= None, filename=None):
    '''
    Make pdf catalog from site-plot(

    Parameters
    ----------
    Workdir : string
        Working directory.
    Filename : string
        Filename. Files to be appended must begin with this string.

    Returns
    -------
    None.

    '''
    # sys.exit('not in 3.9! Exit')

    import fitz

    catalog = fitz.open()

    for pdf in pdflist:
        with fitz.open(pdf) as mfile:
            catalog.insert_pdf(mfile)

    catalog.save(filename, garbage=4, clean = True, deflate=True)
    catalog.close()

    print('\n'+str(np.size(pdflist))+' files collected to '+filename)

def print_title(version='0.99.99', fname='', form='%m/%d/%Y, %H:%M:%S', out=True):
    '''
    Print version, calling file name, and modification date.
    '''

    import os.path
    from datetime import datetime

    title = ''

    if len(version)==0:
        print('No version string given! Not printed to title.')
        tstr = ''
    else:
       ndat = '\n'+''.join('Date ' + datetime.now().strftime(form))
       tstr =  'Py4MT Version '+version+ndat+ '\n'

    if len(fname)==0:
        print('No calling filenane given! Not printed to title.')
        fstr = ''
    else:
        fnam = os.path.basename(fname)
        mdat = datetime.fromtimestamp((os.path.getmtime(fname))).strftime(form)
        fstr = fnam+', modified '+mdat+'\n'
        fstr = fstr + fname

    title = tstr+ fstr

    if out:
        print(title)

    return title

def symlink(src: str, dst: str) -> None:
    """
    Create a symbolic link with force option (like `ln -sf`).

    Parameters
    ----------
    src : str
        Path to the source file or directory.
    dst : str
        Path to the destination symlink.

    Notes
    -----
    - If `dst` already exists (file, symlink, or directory), it will be removed.
    - Directories are removed recursively if non-empty.
    - This function mimics the Unix `ln -sf` behavior.

    Author: Volker Rath (DIAS)
    Copilot (version) and date: Copilot v1.0, 2025-12-01
    """
    try:
        dst_path = pathlib.Path(dst)

        # Remove existing destination if present
        if dst_path.exists() or dst_path.is_symlink():
            if dst_path.is_dir() and not dst_path.is_symlink():
                shutil.rmtree(dst_path)  # remove non-empty directory
            else:
                dst_path.unlink()  # remove file or symlink

        os.symlink(src, dst)
        print(f"Symlink created: {dst} → {src}")

    except Exception as e:
        raise RuntimeError(f"Failed to create symlink {dst} → {src}: {e}") from e

def filecopy(src: str, dst: str) -> None:
    """
    Copy a file or directory with force option (like `cp -f`).

    Parameters
    ----------
    src : str
        Path to the source file or directory.
    dst : str
        Path to the destination file or directory.

    Notes
    -----
    - If `dst` already exists, it will be removed before copying.
    - Directories are copied recursively.
    - Metadata (timestamps, permissions) are preserved for files.
    """
    try:
        src_path = pathlib.Path(src)
        dst_path = pathlib.Path(dst)

        if not src_path.exists():
            raise FileNotFoundError(f"Source {src} does not exist")

        # Remove existing destination
        if dst_path.exists():
            if dst_path.is_dir():
                shutil.rmtree(dst_path)
            else:
                dst_path.unlink()

        if src_path.is_dir():
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)

        print(f"Copied {src} → {dst}")

    except Exception as e:
        raise RuntimeError(f"Failed to copy {src} → {dst}: {e}") from e


def extract_archive(
    src: str,
    dst: str = ".",
    *,
    fmt: str = "auto",
    out: bool = True,
) -> list[str]:
    """
    Extract a ZIP or gzip-compressed TAR archive (tgz / tar.gz) to a directory.

    Parameters
    ----------
    src : str
        Path to the archive file.  Recognised extensions:

        - ``.zip``              → ZIP (via :mod:`zipfile`)
        - ``.tgz``, ``.tar.gz`` → gzip-compressed TAR (via :mod:`tarfile`)

    dst : str, optional
        Destination directory.  Created if it does not exist.
        Default is the current working directory (``"."``).
    fmt : {"auto", "zip", "tgz"}, optional
        Override automatic format detection.

        - ``"auto"`` (default): detect from the file name.
        - ``"zip"``: force ZIP mode.
        - ``"tgz"``: force TAR/gzip mode.

    out : bool, optional
        If True (default), print a summary line after extraction.

    Returns
    -------
    members : list of str
        Sorted list of paths that were extracted (relative to *dst*).

    Raises
    ------
    ValueError
        If the archive format cannot be determined or is unsupported.
    FileNotFoundError
        If *src* does not exist.
    tarfile.TarError / zipfile.BadZipFile
        On corrupt or invalid archive data.

    Examples
    --------
    >>> extract_archive("results.zip", dst="./output")
    >>> extract_archive("data.tgz",   dst="./output")
    >>> extract_archive("data.tar.gz", dst="./output", fmt="tgz")
    """
    src_path = pathlib.Path(src)
    dst_path = pathlib.Path(dst)

    if not src_path.exists():
        raise FileNotFoundError(f"Archive not found: {src}")

    dst_path.mkdir(parents=True, exist_ok=True)

    # ---- format detection ---------------------------------------------------
    name_lower = src_path.name.lower()
    if fmt == "auto":
        if name_lower.endswith(".zip"):
            fmt = "zip"
        elif name_lower.endswith(".tgz") or name_lower.endswith(".tar.gz"):
            fmt = "tgz"
        else:
            raise ValueError(
                f"Cannot determine archive format from '{src_path.name}'. "
                "Pass fmt='zip' or fmt='tgz' explicitly."
            )

    # ---- extraction ---------------------------------------------------------
    members: list[str] = []

    if fmt == "zip":
        with zipfile.ZipFile(src_path, "r") as zf:
            zf.extractall(dst_path)
            members = sorted(zf.namelist())

    elif fmt == "tgz":
        with tarfile.open(src_path, "r:gz") as tf:
            tf.extractall(dst_path)
            members = sorted(m.name for m in tf.getmembers())

    else:
        raise ValueError(f"Unsupported fmt='{fmt}'. Use 'zip', 'tgz', or 'auto'.")

    if out:
        print(f"Extracted {len(members)} item(s) from '{src}' → '{dst}'")

    return members


def dict_to_namespace(d):
    """
    Convert a dictionary into a SimpleNamespace with attribute access.

    Parameters
    ----------
    d : dict
        Dictionary whose keys become attributes.

    Returns
    -------
    SimpleNamespace
        Object with attribute-style access to dictionary entries.

    Notes
    -----
    - Keys must be valid Python identifiers.
    - Values are stored as-is.
    """
    for key in d:
        if not key.isidentifier():
            raise ValueError(f"Invalid key for attribute access: {key}")
    return SimpleNamespace(**d)



# ---------------------------------------------------------------------------
# 1-D MT forward modelling
# ---------------------------------------------------------------------------

def mt1dfwd(
    freq: np.ndarray,
    sig: np.ndarray,
    d: np.ndarray,
    inmod: str = "r",
    out: str = "imp",
    magfield: str = "b",
):
    """Compute 1D MT forward response for a layered Earth."""
    mu0 = 4.0 * np.pi * 1.0e-7
    sig = np.array(sig, dtype=float)
    freq = np.array(freq, dtype=float)
    d = np.array(d, dtype=float)

    if inmod.lower().startswith("r"):
        sig = 1.0 / sig

    if sig.ndim > 1:
        raise ValueError("sig must be 1D.")

    nlay = sig.size
    Z = np.zeros_like(freq, dtype=complex)
    w = 2.0 * np.pi * freq

    for ifr, omega in enumerate(w):
        imp = np.empty(nlay, dtype=complex)
        imp[-1] = np.sqrt(1j * omega * mu0 / sig[-1])

        for layer in range(nlay - 2, -1, -1):
            sl = sig[layer]
            dl = d[layer]
            dj = np.sqrt(1j * omega * mu0 * sl)
            wj = dj / sl
            ej = np.exp(-2.0 * dl * dj)
            impb = imp[layer + 1]
            rj = (wj - impb) / (wj + impb)
            reff = rj * ej
            imp[layer] = wj * ((1.0 - reff) / (1.0 + reff))

        Z[ifr] = imp[0]

    if out.lower() == "imp":
        return Z / mu0 if magfield.lower() == "b" else Z

    absZ = np.abs(Z)
    rhoa = (absZ**2) / (mu0 * w)
    phase = np.rad2deg(np.angle(Z))

    if out.lower() == "rho":
        return rhoa, phase
    return Z, rhoa, phase


def wait1d(periods: np.ndarray, thick: np.ndarray, res: np.ndarray):
    """Alternative 1D MT forward modelling implementation (legacy)."""
    mu = 4.0 * np.pi * 1.0e-7
    omega = 2.0 * np.pi / periods

    cond = 1.0 / np.asarray(res, dtype=float)
    nlay = cond.size

    spn = np.size(periods)
    Z = np.zeros(spn, dtype=complex)

    for idx, w in enumerate(omega):
        prop_const = np.sqrt(1j * mu * cond[-1] * w)
        C = np.zeros(nlay, dtype=complex)
        C[-1] = 1.0 / prop_const
        if len(thick) > 0:
            for k in reversed(range(nlay - 1)):
                prop_layer = np.sqrt(1j * w * mu * cond[k])
                k1 = (C[k + 1] * prop_layer + np.tanh(prop_layer * thick[k]))
                k2 = ((C[k + 1] * prop_layer * np.tanh(prop_layer * thick[k])) + 1.0)
                C[k] = (1.0 / prop_layer) * (k1 / k2)
        Z[idx] = 1j * w * mu * C[0]

    rhoa = (np.abs(Z) ** 2) / (mu * omega)
    phi = np.angle(Z, deg=True)
    return rhoa, phi, np.real(Z), np.imag(Z)


# ===========================================================================
# Petrophysical resistivity & permeability models
# ===========================================================================
# Merged from resistivity_models.py (VR 2026-04-02; Claude Sonnet 4.6 (Anthropic))
#
# References
# ----------
# Archie, G.E. (1942) Trans. AIME, 146, 54-62.
# Simandoux, P. (1963) Rev. Inst. Fr. Pétrole, 18 (suppl.), 193-220.
# Warren, J.E. & Root, P.J. (1963) SPE-J, 3(3), 245-255.
# Glover, P.W.J., Zadjali, I.I. & Frew, K.A. (2006) Geophysics, 71(4), F49-F60.
# Glover, P.W.J. (2010) Solid Earth, 1, 85-91.
# Sen, P.N. & Goode, P.A. (1992) Geophysics, 57(1), 89-96.
# Hashin, Z. & Shtrikman, S. (1962) J. Appl. Phys., 33(10), 3125-3131.
# Berryman, J.G. (1995) Rock Physics & Phase Relations (AGU Ref. Shelf 3).
# ===========================================================================

# ---------------------------------------------------------------------------
# Brine conductivity / resistivity (salinity + temperature → σw, Rw)
# ---------------------------------------------------------------------------

def brine_conductivity_sen_goode(
    salinity_ppm: float,
    temp_c: float,
) -> float:
    """
    NaCl brine electrical conductivity via Sen & Goode (1992).

    This is the standard petrophysical formula for formation-water
    conductivity, derived from a polynomial fit to experimental NaCl
    solution data.  It is physically more rigorous than Arps/Hilchie
    and is valid over the full range of formation-brine conditions:

        T  :   0 – 300 °C
        C  :   0 – 300 g/L  (≈ 0 – 300,000 ppm)

    Governing equation (Sen & Goode 1992, eq. 1)
    ---------------------------------------------
    σ_w(T, C) = C · [5.6 + 0.27·T - 1.5×10⁻⁴·T²]
                  - C^1.5 · [2.36 + 0.099·T] / (1 + 0.214·C^0.5)

    where
        σ_w  conductivity in S/m
        C    NaCl concentration in mol/L  (molarity)
        T    temperature in °C

    Salinity conversion
    -------------------
    The function accepts salinity in ppm (mg NaCl per kg solution) and
    converts internally:

        C [mol/L] ≈ salinity_ppm × ρ_brine / (58,443 × 10⁶)

    Brine density ρ_brine is estimated from the Batzle & Wang (1992)
    approximation for NaCl solutions:

        ρ_brine [kg/m³] = ρ_water(T) + 0.668·S + 0.44·S²
                          + 1e-6·S·(300·P - 2400·S + T·(80 + 3T - 3300·S))

    where S is salinity in g/cm³ and P is pressure in MPa.
    At typical reservoir pressures (< 50 MPa) and moderate salinities,
    the pressure correction is small; this function uses P = 0 MPa
    (surface / low-pressure approximation).  For high-pressure reservoirs
    supply the corrected density via :func:`brine_conductivity_sen_goode_mol`
    directly.

    Parameters
    ----------
    salinity_ppm : float
        NaCl equivalent salinity in mg/kg (ppm by mass).
        Range: 1 000 – 300 000 ppm.
    temp_c : float
        Temperature in °C.  Range: 0 – 300 °C.

    Returns
    -------
    float
        Brine conductivity σ_w in S/m.

    See Also
    --------
    brine_conductivity_sen_goode_mol : same model but accepts molarity directly.
    brine_resistivity_sen_goode      : returns Rw = 1/σ_w in Ω·m.
    brine_resistivity                : legacy Arps/Hilchie approximation.

    References
    ----------
    Sen, P.N. & Goode, P.A. (1992) Geophysics 57(1), 89-96.
    Batzle, M. & Wang, Z. (1992) Geophysics 57(11), 1396-1408.

    Examples
    --------
    >>> sw = brine_conductivity_sen_goode(30_000, 60)
    >>> print(f"σ_w = {sw:.4f} S/m")
    >>> print(f"Rw  = {1/sw:.4f} Ω·m")
    """
    if salinity_ppm <= 0:
        raise ValueError("salinity_ppm must be positive")
    if not 0 <= temp_c <= 300:
        raise ValueError(f"temp_c must be in [0, 300] °C for Sen & Goode, got {temp_c}")
    if salinity_ppm > 350_000:
        raise ValueError(f"salinity_ppm={salinity_ppm} exceeds NaCl solubility limit (~360 g/kg)")

    # --- Brine density (Batzle & Wang 1992, P = 0 MPa approximation) ---
    S_frac   = salinity_ppm * 1e-6              # mass fraction (kg NaCl / kg solution)
    T        = temp_c
    # Pure water density polynomial (valid 0–350 °C, Batzle & Wang eq. 27a)
    rho_w = (1.0
             + 1e-6 * (-80.0*T - 3.3*T**2 + 0.00175*T**3
                       + 489.0*300*0         # pressure term zeroed
                       - 2*T*300*0
                       + 0.016*T**2*0))
    # Simpler pure-water density fit good to ±0.5 % for 0–200 °C:
    rho_w = (999.842594
              + 6.793952e-2 * T
              - 9.095290e-3 * T**2
              + 1.001685e-4 * T**3
              - 1.120083e-6 * T**4
              + 6.536332e-9 * T**5) / 1000.0  # → g/cm³

    S_gcm3   = S_frac                           # ≈ g NaCl / cm³ solution at low salinity
    rho_b    = rho_w + 0.668*S_gcm3 + 0.44*S_gcm3**2   # g/cm³  (Batzle & Wang eq. 27b, P=0)

    # --- Molarity C [mol/L] ---
    # C = (mass fraction × density [g/mL]) / molar mass [g/mol]
    M_NaCl   = 58.443                           # g/mol
    C        = (S_frac * rho_b * 1000.0) / M_NaCl   # mol/L

    return brine_conductivity_sen_goode_mol(C, temp_c)


def brine_conductivity_sen_goode_mol(
    C: float,
    temp_c: float,
) -> float:
    """
    Sen & Goode (1992) conductivity formula in its native molarity form.

    Parameters
    ----------
    C : float
        NaCl molarity in mol/L.  Range: 0 – ~5.5 mol/L (saturation).
    temp_c : float
        Temperature in °C.  Range: 0 – 300 °C.

    Returns
    -------
    float
        Brine conductivity σ_w in S/m.
    """
    if C < 0:
        raise ValueError(f"Molarity C must be ≥ 0, got {C}")
    if C == 0:
        return 0.0
    if not 0 <= temp_c <= 300:
        raise ValueError(f"temp_c must be in [0, 300] °C, got {temp_c}")

    T  = temp_c
    # Sen & Goode (1992) eq. 1
    sigma = (C * (5.6 + 0.27*T - 1.5e-4*T**2)
             - C**1.5 * (2.36 + 0.099*T) / (1.0 + 0.214*C**0.5))
    return max(sigma, 0.0)


def brine_resistivity_sen_goode(
    salinity_ppm: float,
    temp_c: float,
) -> float:
    """
    NaCl brine resistivity Rw via Sen & Goode (1992).

    Convenience wrapper: returns 1 / σ_w from
    :func:`brine_conductivity_sen_goode`.

    Parameters
    ----------
    salinity_ppm : float
        NaCl equivalent salinity in ppm (mg/kg).
    temp_c : float
        Temperature in °C.

    Returns
    -------
    float
        Brine resistivity Rw in Ω·m.

    Examples
    --------
    >>> rw = brine_resistivity_sen_goode(30_000, 60)
    >>> print(f"Rw = {rw:.4f} Ω·m")
    """
    sigma = brine_conductivity_sen_goode(salinity_ppm, temp_c)
    if sigma == 0.0:
        return math.inf
    return 1.0 / sigma


def brine_resistivity(salinity_ppm: float, temp_c: float) -> float:
    """
    Estimate brine resistivity Rw — **Arps/Hilchie empirical correlation**.

    Kept for backward compatibility and quick estimates.  For new work
    prefer :func:`brine_resistivity_sen_goode`, which is physically
    grounded and valid to 300 °C / 300 g/L.

    Arps/Hilchie equations
    ----------------------
        Rw(25 °C) = 0.0123 + 3647.5 / salinity_ppm^0.955
        Rw(T)     = Rw(25°C) × (25 + 21.5) / (T + 21.5)

    Valid range: ~1 000 – 200 000 ppm, 15 – 150 °C.

    Parameters
    ----------
    salinity_ppm : float
        NaCl equivalent salinity in ppm (mg/kg).
    temp_c : float
        Temperature in °C.

    Returns
    -------
    float
        Brine resistivity Rw in Ω·m.
    """
    if salinity_ppm <= 0:
        raise ValueError("salinity_ppm must be positive")
    if temp_c < 0:
        raise ValueError("temp_c must be ≥ 0 °C")

    rw_25 = 0.0123 + 3647.5 / (salinity_ppm ** 0.955)
    rw_t  = rw_25 * (25.0 + 21.5) / (temp_c + 21.5)
    return rw_t


# ---------------------------------------------------------------------------
# 1. Archie model (clean formation)
# ---------------------------------------------------------------------------

@dataclass
class ArchieResult:
    """Results from the Archie model."""
    Rw:  float          # brine resistivity, Ω·m
    F:   float          # formation factor  F = a / φ^m
    Ro:  float          # 100 % water-saturated resistivity, Ω·m
    Rt:  float          # true formation resistivity, Ω·m
    RI:  float          # resistivity index  RI = Rt / Ro  (= Sw^-n)
    Sw:  float          # water saturation (input echo)


def archie(
    phi:          float,
    Sw:           float,
    Rw:           float,
    m:            float = 2.0,
    n:            float = 2.0,
    a:            float = 1.0,
) -> ArchieResult:
    """
    Archie (1942) clean-formation resistivity model.

    Governing equations
    -------------------
    Formation factor :  F  = a / φ^m          (Archie's first law)
    100 % Sw resisivity: Ro = F · Rw
    True resistivity :  Rt = Ro / Sw^n        (Archie's second law)
    Resistivity index:  RI = Rt / Ro = Sw^-n

    Parameters
    ----------
    phi : float
        Total (connected) porosity, fraction  0 < φ ≤ 1.
    Sw : float
        Water saturation, fraction  0 < Sw ≤ 1.
    Rw : float
        Brine resistivity in Ω·m.  Use :func:`brine_resistivity` to
        compute from salinity + temperature.
    m : float
        Cementation exponent (default 2.0).
        Typical ranges: 1.3–1.7 (fractures/vugs), 1.8–2.2 (clastics),
        2.0–3.0 (carbonates).
    n : float
        Saturation exponent (default 2.0).  Usually 1.5–2.5.
    a : float
        Tortuosity factor (default 1.0).  Glover (2010) argues a must
        equal 1 to be physically consistent; use 1.0 unless fitting
        legacy datasets that require a ≠ 1.

    Returns
    -------
    ArchieResult
        Named tuple with Rw, F, Ro, Rt, RI, Sw.

    Raises
    ------
    ValueError
        If any parameter is outside its physical range.

    Examples
    --------
    >>> rw = brine_resistivity(30_000, 60)
    >>> r  = archie(phi=0.20, Sw=0.80, Rw=rw, m=2.0, n=2.0)
    >>> print(f"Rt = {r.Rt:.3f} Ω·m")
    """
    if not 0 < phi <= 1:
        raise ValueError(f"phi must be in (0, 1], got {phi}")
    if not 0 < Sw <= 1:
        raise ValueError(f"Sw must be in (0, 1], got {Sw}")
    if Rw <= 0:
        raise ValueError(f"Rw must be positive, got {Rw}")
    if m <= 0 or n <= 0 or a <= 0:
        raise ValueError("m, n, a must all be positive")

    F  = a / (phi ** m)
    Ro = F * Rw
    Rt = Ro / (Sw ** n)
    RI = Rt / Ro

    return ArchieResult(Rw=Rw, F=F, Ro=Ro, Rt=Rt, RI=RI, Sw=Sw)


# ---------------------------------------------------------------------------
# 2. Simandoux model (shaly sand)
# ---------------------------------------------------------------------------

@dataclass
class SimandouxResult:
    """Results from the Simandoux model."""
    Rw:   float     # brine resistivity, Ω·m
    F:    float     # clean-sand formation factor (Archie)
    Ro:   float     # 100 % Sw resistivity (shaly), Ω·m
    Rt:   float     # true resistivity, Ω·m
    RI:   float     # resistivity index Rt / Ro
    Sw:   float     # water saturation (input echo)
    Vsh:  float     # shale volume (input echo)


def simandoux(
    phi:          float,
    Sw:           float,
    Rw:           float,
    Vsh:          float,
    Rsh:          float,
    m:            float = 2.0,
    n:            float = 2.0,
    a:            float = 1.0,
) -> SimandouxResult:
    """
    Simandoux (1963) shaly-sand resistivity model.

    Adds a parallel conduction path through dispersed clay to the
    Archie clean-sand framework.

    Governing equation
    ------------------
    Conductivity form (solves for Rt):

        1/Rt = (φ^m · Sw^n) / (a · Rw)  +  Vsh · Sw / Rsh

    At Sw = 1 this reduces to:

        1/Ro = φ^m / (a · Rw)  +  Vsh / Rsh

    which equals pure Archie when Vsh → 0.

    Parameters
    ----------
    phi : float
        Total porosity, fraction  0 < φ ≤ 1.
    Sw : float
        Water saturation, fraction  0 < Sw ≤ 1.
    Rw : float
        Brine resistivity in Ω·m.
    Vsh : float
        Shale (clay) volume fraction  0 ≤ Vsh < 1.
    Rsh : float
        Shale resistivity in Ω·m (measured from pure shale baseline).
    m : float
        Cementation exponent for the clean sand (default 2.0).
    n : float
        Saturation exponent (default 2.0).
    a : float
        Tortuosity factor (default 1.0).

    Returns
    -------
    SimandouxResult

    Notes
    -----
    The model is most reliable for Vsh < 0.5.  For highly shaly
    formations consider Indonesia, Waxman-Smits, or Dual-Water models
    which account for bound-water conductivity more rigorously.

    Examples
    --------
    >>> rw = brine_resistivity(30_000, 60)
    >>> r  = simandoux(phi=0.20, Sw=0.70, Rw=rw, Vsh=0.15, Rsh=3.0)
    >>> print(f"Rt = {r.Rt:.3f} Ω·m")
    """
    if not 0 < phi <= 1:
        raise ValueError(f"phi must be in (0, 1], got {phi}")
    if not 0 < Sw <= 1:
        raise ValueError(f"Sw must be in (0, 1], got {Sw}")
    if not 0 <= Vsh < 1:
        raise ValueError(f"Vsh must be in [0, 1), got {Vsh}")
    if Rw <= 0 or Rsh <= 0:
        raise ValueError("Rw and Rsh must be positive")

    F       = a / (phi ** m)
    C_sand  = (phi ** m * Sw ** n) / (a * Rw)
    C_shale = Vsh * Sw / Rsh
    Ct      = C_sand + C_shale
    Rt      = 1.0 / Ct

    # Ro = Rt at Sw = 1
    Ct_ro = phi ** m / (a * Rw) + Vsh / Rsh
    Ro    = 1.0 / Ct_ro
    RI    = Rt / Ro

    return SimandouxResult(Rw=Rw, F=F, Ro=Ro, Rt=Rt, RI=RI, Sw=Sw, Vsh=Vsh)


# ---------------------------------------------------------------------------
# 3. Dual-porosity / fractured model
# ---------------------------------------------------------------------------

@dataclass
class DualPorosityResult:
    """Results from the dual-porosity (matrix + fracture) model."""
    Rw:        float   # brine resistivity, Ω·m
    Rt_matrix: float   # matrix resistivity (Archie), Ω·m
    Rt_frac:   float   # fracture network resistivity (Archie), Ω·m
    Rt:        float   # combined true resistivity (parallel), Ω·m
    Ro_matrix: float   # matrix Ro (Sw=1), Ω·m
    Ro_frac:   float   # fracture Ro (Sw=1), Ω·m
    Ro:        float   # combined Ro (parallel), Ω·m
    RI:        float   # resistivity index Rt / Ro
    F_matrix:  float   # matrix formation factor
    F_frac:    float   # fracture formation factor
    Sw_matrix: float   # matrix water saturation (input echo)
    Sw_frac:   float   # fracture water saturation (input echo)
    phi_matrix: float  # matrix porosity (input echo)
    phi_frac:  float   # fracture porosity (input echo)


def dual_porosity(
    phi_matrix:  float,
    Sw_matrix:   float,
    Rw:          float,
    phi_frac:    float,
    Sw_frac:     float   = 1.0,
    m_matrix:    float   = 2.0,
    n_matrix:    float   = 2.0,
    m_frac:      float   = 1.3,
    n_frac:      float   = 1.5,
    a_matrix:    float   = 1.0,
    a_frac:      float   = 1.0,
) -> DualPorosityResult:
    """
    Dual-porosity (Warren-Root) resistivity model for fractured formations.

    The matrix and fracture networks are treated as two independent
    Archie conductors arranged in **electrical parallel**:

        1/Rt = 1/Rt_matrix + 1/Rt_frac

    Each end-member follows its own Archie law with independent
    cementation and saturation exponents.

    Parameters
    ----------
    phi_matrix : float
        Matrix (intergranular) porosity, fraction.  Typically 0.05–0.35.
    Sw_matrix : float
        Water saturation in the matrix, fraction.
    Rw : float
        Brine resistivity in Ω·m (same fluid assumed in both systems).
    phi_frac : float
        Fracture porosity, fraction.  Typically 0.001–0.02.
    Sw_frac : float
        Water saturation in the fracture network (default 1.0 — fractures
        are commonly fully brine-saturated even when matrix has hydrocarbons).
    m_matrix : float
        Matrix cementation exponent (default 2.0).
    n_matrix : float
        Matrix saturation exponent (default 2.0).
    m_frac : float
        Fracture cementation exponent (default 1.3).
        Open fractures range 1.0–1.5; partially healed fractures 1.5–2.0.
    n_frac : float
        Fracture saturation exponent (default 1.5).
    a_matrix : float
        Tortuosity factor for the matrix (default 1.0).
    a_frac : float
        Tortuosity factor for the fracture network (default 1.0).

    Returns
    -------
    DualPorosityResult

    Notes
    -----
    Fractures dramatically lower the bulk resistivity when they are
    brine-saturated (Sw_frac = 1), because even tiny fracture porosity
    (~0.5 %) creates a very low-resistivity parallel path (F_frac is
    small for m_frac ≈ 1.3).

    For a triple-porosity system (matrix + fractures + vugs) call this
    function twice: first combine matrix + fractures, then combine the
    result with a vug Archie term.

    Examples
    --------
    >>> rw = brine_resistivity(30_000, 60)
    >>> r  = dual_porosity(phi_matrix=0.20, Sw_matrix=0.70, Rw=rw,
    ...                    phi_frac=0.005, Sw_frac=1.0)
    >>> print(f"Rt = {r.Rt:.3f} Ω·m  (matrix alone: {r.Rt_matrix:.2f} Ω·m)")
    """
    if not 0 < phi_matrix <= 1:
        raise ValueError(f"phi_matrix must be in (0,1], got {phi_matrix}")
    if not 0 < phi_frac <= 1:
        raise ValueError(f"phi_frac must be in (0,1], got {phi_frac}")
    if not 0 < Sw_matrix <= 1:
        raise ValueError(f"Sw_matrix must be in (0,1], got {Sw_matrix}")
    if not 0 < Sw_frac <= 1:
        raise ValueError(f"Sw_frac must be in (0,1], got {Sw_frac}")
    if Rw <= 0:
        raise ValueError("Rw must be positive")

    # --- Matrix (Archie)
    F_m      = a_matrix / (phi_matrix ** m_matrix)
    Ro_m     = F_m * Rw
    Rt_m     = Ro_m / (Sw_matrix ** n_matrix)

    # --- Fracture network (Archie with lower m)
    F_f      = a_frac / (phi_frac ** m_frac)
    Ro_f     = F_f * Rw
    Rt_f     = Ro_f / (Sw_frac ** n_frac)

    # --- Parallel combination
    Rt       = 1.0 / (1.0 / Rt_m + 1.0 / Rt_f)
    Ro       = 1.0 / (1.0 / Ro_m + 1.0 / Ro_f)
    RI       = Rt / Ro

    return DualPorosityResult(
        Rw        = Rw,
        Rt_matrix = Rt_m,
        Rt_frac   = Rt_f,
        Rt        = Rt,
        Ro_matrix = Ro_m,
        Ro_frac   = Ro_f,
        Ro        = Ro,
        RI        = RI,
        F_matrix  = F_m,
        F_frac    = F_f,
        Sw_matrix = Sw_matrix,
        Sw_frac   = Sw_frac,
        phi_matrix = phi_matrix,
        phi_frac  = phi_frac,
    )


# ---------------------------------------------------------------------------
# Convenience: invert for Sw from measured Rt
# ---------------------------------------------------------------------------

def solve_Sw_archie(
    Rt:   float,
    phi:  float,
    Rw:   float,
    m:    float = 2.0,
    n:    float = 2.0,
    a:    float = 1.0,
) -> float:
    """
    Invert Archie's law to recover Sw from a measured Rt.

    Sw = (a · Rw / (φ^m · Rt))^(1/n)

    Parameters
    ----------
    Rt : float
        Measured true resistivity in Ω·m.
    phi, Rw, m, n, a : float
        Same as in :func:`archie`.

    Returns
    -------
    float
        Water saturation Sw, clipped to [0, 1].
    """
    F   = a / (phi ** m)
    Ro  = F * Rw
    Sw  = (Ro / Rt) ** (1.0 / n)
    return max(0.0, min(1.0, Sw))


def solve_Sw_simandoux(
    Rt:   float,
    phi:  float,
    Rw:   float,
    Vsh:  float,
    Rsh:  float,
    m:    float = 2.0,
    n:    float = 2.0,
    a:    float = 1.0,
    tol:  float = 1e-8,
    maxiter: int = 200,
) -> float:
    """
    Invert the Simandoux equation for Sw using Newton-Raphson iteration.

    Solves:

        f(Sw) = φ^m · Sw^n / (a · Rw)  +  Vsh · Sw / Rsh  -  1/Rt = 0

    Parameters
    ----------
    Rt : float
        Measured true resistivity in Ω·m.
    phi, Rw, Vsh, Rsh, m, n, a : float
        Same as in :func:`simandoux`.
    tol : float
        Convergence tolerance on |f(Sw)| (default 1e-8).
    maxiter : int
        Maximum Newton iterations (default 200).

    Returns
    -------
    float
        Water saturation Sw in [0, 1].

    Raises
    ------
    RuntimeError
        If Newton-Raphson does not converge within `maxiter` iterations.
    """
    A  = phi ** m / (a * Rw)   # coefficient for Sw^n term
    B  = Vsh / Rsh             # coefficient for Sw term
    C  = 1.0 / Rt

    Sw = 0.5  # initial guess
    for _ in range(maxiter):
        f   = A * Sw ** n + B * Sw - C
        df  = A * n * Sw ** (n - 1) + B
        dSw = -f / df
        Sw  = Sw + dSw
        Sw  = max(1e-6, min(1.0, Sw))
        if abs(f) < tol:
            return Sw

    raise RuntimeError(
        f"solve_Sw_simandoux did not converge after {maxiter} iterations "
        f"(last Sw={Sw:.4f}, residual={abs(f):.2e})"
    )


# ---------------------------------------------------------------------------
# 4. RGPZ permeability model (Glover et al. 2006)
# ---------------------------------------------------------------------------

#: 1 Darcy in m²  (exact SI conversion)
_DARCY_m2 = 9.869_233e-13

#: Packing constant for quasi-spherical grains (= 8/3)
PACKING_SPHERICAL: float = 8.0 / 3.0


@dataclass
class RGPZResult:
    """Results from the RGPZ permeability model."""
    k_m2:        float   # permeability in m²
    k_mD:        float   # permeability in milliDarcy
    F:           float   # formation factor  φ^-m
    S:           float   # connectivity  φ^m  (= 1/F)
    tortuosity:  float   # electrical tortuosity  T = F · φ
    d_geom_um:   float   # grain diameter used (µm, input echo)
    phi:         float   # porosity (input echo)
    m:           float   # cementation exponent (input echo)
    a_pack:      float   # packing constant (input echo)


def rgpz(
    phi:       float,
    d_geom_um: float,
    m:         float = 2.0,
    a_pack:    float = PACKING_SPHERICAL,
) -> RGPZResult:
    """
    RGPZ permeability model (Revil, Glover, Pezard & Zamora 1999/2006).

    Derives permeability from porosity, cementation exponent, and grain
    size — quantities accessible from resistivity logs plus a grain-size
    measurement (sieve, laser diffraction, or image analysis).

    Governing equation
    ------------------

        k = d̄² · φ^(3m) / (4 · a · m²)          [m²]

    where

        φ^(3m) = φ^m · φ^m · φ^m = S³   (connectivity cubed)

    Equivalently, using the formation factor F = φ^(-m):

        k = d̄² / (4 · a · m² · F³)

    Key derived quantities
    ----------------------
    Formation factor  F = φ^(-m)   (Archie's first law, a = 1)
    Connectivity      S = φ^m  = 1/F
    Tortuosity        T = F · φ

    Parameters
    ----------
    phi : float
        Connected porosity, fraction  0 < φ ≤ 1.
    d_geom_um : float
        **Geometric mean** grain diameter in µm.  Glover et al. (2006)
        demonstrated that the geometric mean gives significantly better
        predictions than the arithmetic mean.  For unimodal, roughly
        log-normal grain size distributions convert from sieve data via:
            d_geom = exp(mean(ln(d_i) · f_i))
        where d_i are sieve midpoints and f_i the mass fractions.
    m : float
        Cementation exponent (default 2.0).  The same m used in Archie's
        formation-factor law F = φ^(-m).  Can be derived from resistivity
        and porosity logs, or from MICP.
    a_pack : float
        Packing constant (default 8/3 ≈ 2.667 for quasi-spherical grains,
        following Glover et al. 2006).  For angular grains or other
        geometries this may be fitted to core data.

    Returns
    -------
    RGPZResult
        Dataclass containing k in m² and mD, plus F, S, tortuosity, and
        echoed inputs.

    Raises
    ------
    ValueError
        If any parameter is outside its physical range.

    Notes
    -----
    The model inherently accounts for dead-end and unconnected porosity
    through the φ^(3m) term, which is why it outperforms Kozeny-Carman
    (which uses φ³/(1-φ)²) for tight and heterogeneous rocks.

    The Kozeny-Carman equation for comparison:

        k_KC = d̄² · φ³ / (72 · τ · (1 - φ)²)

    where τ is tortuosity (≈ 2.5 for random sphere packs).

    Unit conversion: 1 mD = 9.869 × 10⁻¹⁶ m²

    Examples
    --------
    >>> r = rgpz(phi=0.20, d_geom_um=150.0, m=2.0)
    >>> print(f"k = {r.k_mD:.2f} mD")

    >>> # Derive m from a resistivity + porosity measurement:
    >>> import math
    >>> F_measured = 28.5          # from Rt / Rw at Sw = 1
    >>> phi_core   = 0.18
    >>> m_fit      = -math.log(F_measured) / math.log(phi_core)
    >>> r2 = rgpz(phi=phi_core, d_geom_um=120.0, m=m_fit)
    >>> print(f"m = {m_fit:.3f},  k = {r2.k_mD:.2f} mD")
    """
    if not 0 < phi <= 1:
        raise ValueError(f"phi must be in (0, 1], got {phi}")
    if d_geom_um <= 0:
        raise ValueError(f"d_geom_um must be positive, got {d_geom_um}")
    if m <= 0:
        raise ValueError(f"m must be positive, got {m}")
    if a_pack <= 0:
        raise ValueError(f"a_pack must be positive, got {a_pack}")

    d_m   = d_geom_um * 1e-6          # µm → m
    k_m2  = (d_m ** 2) * (phi ** (3 * m)) / (4.0 * a_pack * m ** 2)
    k_mD  = k_m2 / _DARCY_m2 * 1000.0

    F          = phi ** (-m)
    S          = phi ** m              # connectivity = 1/F
    tortuosity = F * phi               # T = F·φ

    return RGPZResult(
        k_m2       = k_m2,
        k_mD       = k_mD,
        F          = F,
        S          = S,
        tortuosity = tortuosity,
        d_geom_um  = d_geom_um,
        phi        = phi,
        m          = m,
        a_pack     = a_pack,
    )


def rgpz_from_formation_factor(
    F:         float,
    phi:       float,
    d_geom_um: float,
    a_pack:    float = PACKING_SPHERICAL,
) -> RGPZResult:
    """
    RGPZ permeability using a *measured* formation factor F directly.

    This avoids fitting m explicitly: it derives m = -ln(F)/ln(φ) and
    then calls :func:`rgpz`.  Useful when F is read directly from a
    resistivity log (F = Rt/Rw at Sw = 1) and core porosity is known.

    Parameters
    ----------
    F : float
        Measured formation factor F = Ro / Rw  (dimensionless, > 1).
    phi : float
        Porosity, fraction.
    d_geom_um : float
        Geometric mean grain diameter in µm.
    a_pack : float
        Packing constant (default 8/3).

    Returns
    -------
    RGPZResult
        Same as :func:`rgpz`.

    Examples
    --------
    >>> r = rgpz_from_formation_factor(F=28.5, phi=0.18, d_geom_um=120.0)
    >>> print(f"m = {r.m:.3f},  k = {r.k_mD:.2f} mD")
    """
    if F <= 1:
        raise ValueError(f"Formation factor F must be > 1, got {F}")
    if not 0 < phi < 1:
        raise ValueError(f"phi must be in (0, 1), got {phi}")

    m_derived = -math.log(F) / math.log(phi)
    return rgpz(phi=phi, d_geom_um=d_geom_um, m=m_derived, a_pack=a_pack)


def kozeny_carman(
    phi:       float,
    d_geom_um: float,
    tau:       float = 2.5,
) -> float:
    """
    Kozeny-Carman permeability for reference / comparison with RGPZ.

        k = d̄² · φ³ / (72 · τ · (1 - φ)²)     [mD]

    Parameters
    ----------
    phi : float
        Porosity, fraction.
    d_geom_um : float
        Grain diameter in µm.
    tau : float
        Tortuosity (default 2.5, typical for random sphere packs).

    Returns
    -------
    float
        Permeability in milliDarcy.
    """
    d_m  = d_geom_um * 1e-6
    k_m2 = (d_m ** 2) * phi ** 3 / (72.0 * tau * (1.0 - phi) ** 2)
    return k_m2 / _DARCY_m2 * 1000.0




# ---------------------------------------------------------------------------
# 5. Hashin-Shtrikman bounds on effective conductivity
# ---------------------------------------------------------------------------

@dataclass
class HSBoundsResult:
    """
    Hashin-Shtrikman upper and lower conductivity bounds for a two-phase
    composite, plus the HS mixing-formula estimate.

    Naming convention (conductivity space → resistivity space):
        sigma_upper / R_lower  — HS upper bound on σ  = lower bound on R
        sigma_lower / R_upper  — HS lower bound on σ  = upper bound on R
    """
    sigma_upper:  float   # HS upper bound on σ_eff  [S/m]
    sigma_lower:  float   # HS lower bound on σ_eff  [S/m]
    sigma_hs_mix: float   # HS mixing formula (geometric mean)  [S/m]
    R_upper:      float   # highest resistivity bound (= 1/sigma_lower)  [Ω·m]
    R_lower:      float   # lowest  resistivity bound (= 1/sigma_upper)  [Ω·m]
    R_hs_mix:     float   # HS mixing resistivity  [Ω·m]
    R_parallel:   float   # Voigt / parallel bound  [Ω·m]
    R_series:     float   # Reuss / series bound  [Ω·m]
    sigma_1:      float   # phase-1 conductivity (input echo)  [S/m]
    sigma_2:      float   # phase-2 conductivity (input echo)  [S/m]
    f1:           float   # volume fraction of phase 1 (input echo)
    f2:           float   # volume fraction of phase 2 (input echo)


def hashin_shtrikman_two_phase(
    sigma_1: float,
    sigma_2: float,
    f1:      float,
) -> HSBoundsResult:
    """
    Hashin-Shtrikman (1962) conductivity bounds for a two-phase composite.

    Returns the HS upper bound, HS lower bound, and a geometric mixing
    estimate, together with the classical Voigt (parallel) and Reuss
    (series) arithmetic bounds for comparison.

    Background
    ----------
    Hashin & Shtrikman (1962) derived the tightest possible bounds on the
    effective conductivity σ_eff of a statistically isotropic two-phase
    composite, given only the phase conductivities and volume fractions —
    without any assumption about microgeometry.

    For a composite with phase conductivities σ₁ ≤ σ₂ and volume
    fractions f₁ + f₂ = 1:

    HS lower bound (phase 2 = matrix, phase 1 = inclusions):

        σ_HS⁻ = σ₁ + f₂ / [1/(σ₂ - σ₁) + f₁/(3σ₁)]

    HS upper bound (phase 1 = matrix, phase 2 = inclusions):

        σ_HS⁺ = σ₂ + f₁ / [1/(σ₁ - σ₂) + f₂/(3σ₂)]

    The HS bounds are always tighter than the Voigt-Reuss (parallel-series)
    bounds:

        σ_Reuss  ≤  σ_HS⁻  ≤  σ_eff  ≤  σ_HS⁺  ≤  σ_Voigt

    Petrophysical context
    ---------------------
    For a rock with two phases (e.g. pore fluid and mineral matrix):

        Phase 1 = pore fluid  (higher σ, e.g. brine:  ~5 S/m)
        Phase 2 = mineral     (lower σ,  e.g. quartz: ~1e-9 S/m)
        f₁      = porosity φ

    The HS bounds then bracket the effective formation conductivity for
    any pore geometry.  Archie's law should fall inside the HS bounds; if
    it does not, the cementation exponent m is likely inconsistent with
    an isotropic composite assumption.

    HS mixing formula
    -----------------
    Glover, Hole & Pous (2000) proposed a self-consistent HS mixing
    formula that interpolates between bounds using the volume fractions
    directly:

        σ_mix = σ₁^f₁ · σ₂^f₂      (geometric mean)

    This is equivalent to the HS mixing model for the special case of
    equal volume fractions and gives a single-valued estimate inside
    the HS envelope.

    Parameters
    ----------
    sigma_1 : float
        Conductivity of phase 1 in S/m  (e.g. pore fluid).
    sigma_2 : float
        Conductivity of phase 2 in S/m  (e.g. mineral matrix).
    f1 : float
        Volume fraction of phase 1 (0 < f1 < 1).
        f2 is computed as 1 - f1.

    Returns
    -------
    HSBoundsResult
        Upper bound, lower bound, mixing estimate, Voigt and Reuss bounds —
        all in both S/m (conductivity) and Ω·m (resistivity).

    Raises
    ------
    ValueError
        If conductivities are non-positive or volume fractions are out of range.

    Notes
    -----
    The formula requires σ₁ ≠ σ₂.  If the two phases have the same
    conductivity, σ_eff equals that conductivity regardless of geometry
    and the function returns that value for all bounds.

    For N > 2 phases use :func:`hashin_shtrikman_n_phase`.

    Examples
    --------
    >>> # Brine-saturated sandstone: φ = 0.20, σ_fluid = 5 S/m, σ_grain ≈ 0
    >>> r = hashin_shtrikman_two_phase(sigma_1=5.0, sigma_2=1e-9, f1=0.20)
    >>> print(f"HS+  = {r.R_upper:.4f} Ω·m")
    >>> print(f"HS-  = {r.R_lower:.6f} Ω·m")

    >>> # Cross-check: Archie Rt should lie between HS- and HS+
    >>> rw = 1/5.0                    # 0.20 Ω·m
    >>> ar = archie(phi=0.20, Sw=1.0, Rw=rw, m=2.0, n=2.0)
    >>> assert r.R_lower <= ar.Rt <= r.R_upper, "Archie outside HS bounds!"
    """
    if sigma_1 <= 0 or sigma_2 <= 0:
        raise ValueError("Both phase conductivities must be positive (S/m)")
    if not 0.0 < f1 < 1.0:
        raise ValueError(f"f1 must be strictly in (0, 1), got {f1}")

    f2 = 1.0 - f1

    # --- Voigt (parallel) and Reuss (series) bounds ---
    sigma_voigt = f1 * sigma_1 + f2 * sigma_2
    sigma_reuss = 1.0 / (f1 / sigma_1 + f2 / sigma_2)

    if math.isclose(sigma_1, sigma_2, rel_tol=1e-10):
        # Degenerate: both phases identical
        s = sigma_1
        return HSBoundsResult(
            sigma_upper=s, sigma_lower=s, sigma_hs_mix=s,
            R_upper=1/s, R_lower=1/s, R_hs_mix=1/s,
            R_parallel=1/s, R_series=1/s,
            sigma_1=sigma_1, sigma_2=sigma_2, f1=f1, f2=f2,
        )

    # Ensure σ_lo ≤ σ_hi for consistent bound labelling
    s_lo, s_hi = min(sigma_1, sigma_2), max(sigma_1, sigma_2)
    f_lo = f2 if sigma_1 > sigma_2 else f1   # volume fraction of the LOW-σ phase
    f_hi = 1.0 - f_lo

    # --- HS lower bound  (low-σ phase is the matrix) ---
    # σ_HS⁻ = σ_lo + f_hi / [1/(σ_hi - σ_lo) + f_lo/(3·σ_lo)]
    hs_lower = s_lo + f_hi / (1.0 / (s_hi - s_lo) + f_lo / (3.0 * s_lo))

    # --- HS upper bound  (high-σ phase is the matrix) ---
    # σ_HS⁺ = σ_hi + f_lo / [1/(σ_lo - σ_hi) + f_hi/(3·σ_hi)]
    hs_upper = s_hi + f_lo / (1.0 / (s_lo - s_hi) + f_hi / (3.0 * s_hi))

    # --- HS geometric mixing (Glover et al. 2000) ---
    hs_mix = (sigma_1 ** f1) * (sigma_2 ** f2)

    def safe_R(s: float) -> float:
        return 1.0 / s if s > 0 else math.inf

    # HS+ is the upper bound on conductivity → lower bound on resistivity
    # HS- is the lower bound on conductivity → upper bound on resistivity
    return HSBoundsResult(
        sigma_upper  = hs_upper,
        sigma_lower  = hs_lower,
        sigma_hs_mix = hs_mix,
        R_upper      = safe_R(hs_lower),   # highest R ← lowest σ
        R_lower      = safe_R(hs_upper),   # lowest  R ← highest σ
        R_hs_mix     = safe_R(hs_mix),
        R_parallel   = safe_R(sigma_voigt),
        R_series     = safe_R(sigma_reuss),
        sigma_1      = sigma_1,
        sigma_2      = sigma_2,
        f1           = f1,
        f2           = f2,
    )


@dataclass
class HSNPhaseResult:
    """
    Results from the N-phase Hashin-Shtrikman generalisation.

    Naming convention (conductivity space → resistivity space):
        sigma_upper / R_lower  — HS upper bound on σ  = lower bound on R
        sigma_lower / R_upper  — HS lower bound on σ  = upper bound on R
    """
    sigma_upper:  float          # HS upper bound on σ_eff  [S/m]
    sigma_lower:  float          # HS lower bound on σ_eff  [S/m]
    R_upper:      float          # highest R bound (= 1/sigma_lower)  [Ω·m]
    R_lower:      float          # lowest  R bound (= 1/sigma_upper)  [Ω·m]
    R_parallel:   float          # Voigt bound     [Ω·m]
    R_series:     float          # Reuss bound     [Ω·m]
    sigmas:       list[float]    # input conductivities (echo)
    fractions:    list[float]    # input volume fractions (echo)


def hashin_shtrikman_n_phase(
    sigmas:    list[float],
    fractions: list[float],
) -> HSNPhaseResult:
    """
    Generalised Hashin-Shtrikman bounds for an N-phase isotropic composite.

    Uses the Berryman (1995) / Milton (1981) N-phase HS formulation:

        σ_HS± = reference phase conductivity + correction sum

    The HS bounds are computed by choosing the reference phase σ_ref
    as the minimum conductivity phase (lower bound) or the maximum
    conductivity phase (upper bound):

        σ_HS = σ_ref  +  [Σᵢ fᵢ / (1/(σᵢ - σ_ref) + 1/(d·σ_ref))]⁻¹

    where d = 3 for three-dimensional composites (isotropic), and the
    sum excludes the reference phase itself (its contribution vanishes).

    Parameters
    ----------
    sigmas : list[float]
        Conductivity of each phase in S/m.  Length N ≥ 2.
    fractions : list[float]
        Volume fraction of each phase.  Must sum to 1.  Length N.

    Returns
    -------
    HSNPhaseResult

    Raises
    ------
    ValueError
        If lengths differ, fractions do not sum to 1, or any value is
        out of physical range.

    Examples
    --------
    >>> # Three-phase: brine pores + quartz + clay
    >>> r = hashin_shtrikman_n_phase(
    ...     sigmas    = [5.0,   1e-9, 0.05],
    ...     fractions = [0.20,  0.70, 0.10],
    ... )
    >>> print(f"HS+ = {r.R_upper:.4f} Ω·m,  HS- = {r.R_lower:.6f} Ω·m")
    """
    if len(sigmas) != len(fractions):
        raise ValueError("sigmas and fractions must have the same length")
    if len(sigmas) < 2:
        raise ValueError("Need at least 2 phases")
    if any(s <= 0 for s in sigmas):
        raise ValueError("All conductivities must be positive")
    if any(f < 0 for f in fractions):
        raise ValueError("All volume fractions must be ≥ 0")

    total_f = sum(fractions)
    if not math.isclose(total_f, 1.0, rel_tol=1e-6):
        raise ValueError(f"Volume fractions must sum to 1, got {total_f:.6f}")

    d = 3  # spatial dimension (isotropic 3-D)

    def _hs_bound(sigma_ref: float) -> float:
        """HS bound with a given reference-phase conductivity."""
        # Σ fᵢ / (1/(σᵢ - σ_ref) + 1/(d·σ_ref))
        # = Σ fᵢ · (σᵢ - σ_ref) · d·σ_ref / (d·σ_ref + σᵢ - σ_ref)
        numerator = 0.0
        for s_i, f_i in zip(sigmas, fractions):
            if math.isclose(s_i, sigma_ref, rel_tol=1e-12):
                continue   # reference phase: term is 0/0 → 0
            numerator += f_i / (1.0 / (s_i - sigma_ref) + 1.0 / (d * sigma_ref))
        return sigma_ref + numerator

    sigma_lower = _hs_bound(min(sigmas))
    sigma_upper = _hs_bound(max(sigmas))

    # For very high conductivity contrasts (many decades) the polynomial
    # N-phase HS formula can return a slightly negative sigma_upper due to
    # floating-point cancellation.  Clamp to zero (physically: the true
    # upper bound approaches the Voigt value, which is always valid).
    sigma_lower = max(sigma_lower, 0.0)
    sigma_upper = max(sigma_upper, 0.0)

    # Voigt and Reuss
    sigma_voigt = sum(f * s for f, s in zip(fractions, sigmas))
    sigma_reuss = 1.0 / sum(f / s for f, s in zip(fractions, sigmas))

    def safe_R(s: float) -> float:
        return 1.0 / s if s > 0 else math.inf

    return HSNPhaseResult(
        sigma_upper = sigma_upper,
        sigma_lower = sigma_lower,
        R_upper     = safe_R(sigma_lower),   # highest R ← lowest σ
        R_lower     = safe_R(sigma_upper),   # lowest  R ← highest σ
        R_parallel  = safe_R(sigma_voigt),
        R_series    = safe_R(sigma_reuss),
        sigmas      = list(sigmas),
        fractions   = list(fractions),
    )


