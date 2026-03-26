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
- Grid generation (lat/lon and UTM)
- Geometry (point-in-polygon, projection onto lines)
- Numerical utilities (KL divergence, L-curve corner, DCT, curvature)
- Miscellaneous (PDF catalog generation, symlink/copy helpers)

Author: Volker Rath (DIAS)
Created: 2020-11-01
Modified: 2026-03-25 — added section headers; docstrings for undocumented functions; get_percentile verbose parameter; cleanup; Claude Sonnet 4.6 (Anthropic)
Modified: 2026-03-26 — added unpack_compressed(), pack_compressed(); Claude Sonnet 4.6 (Anthropic)
"""

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

import numpy as np

import pyproj
from pyproj import CRS, database, Transformer
from scipy.fftpack import dct, idct

from types import ModuleType, SimpleNamespace
from typing import List, Any, Dict
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
        print('strreplace: warning output file overwrites input file!')

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
