#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dem.py

Digital elevation models (DEM) for maps and sections: tile names, download and
cache, reading, mosaic, crop, thinning out and shaded relief.

Moved out of scripts/femtic/femtic_plot_sitemap.py so that any script can use
them. All functions are stateless: every option is an argument, nothing is read
from script settings. A DEM is a dict(lon, lat, z): lon increasing, lat
decreasing (row 0 = north), z[lat, lon] in m (float32, NaN = no value).

Sources (DEM_SOURCES, DEM_PRESETS)
----------------------------------
"copernicus30"  Copernicus DEM GLO-30, GeoTIFF tiles of the AWS open data bucket
                copernicus-dem-30m (rasterio needed)
"copernicus90"  Copernicus DEM GLO-90 (copernicus-dem-90m; rasterio needed)
"skadi"         AWS Terrain Tiles (Mapzen), .hgt tiles (no rasterio needed)
"tiles"         own tiles: a URL or path pattern, GeoTIFF or .hgt
"file"          one file: GeoTIFF (rasterio), .npz (lon, lat, z) or one .hgt tile
"none"          no DEM

Main functions
--------------
load_dem()      DEM of a box from any source (cropped, thinned out)
dem_credit()    the credit line of a source
shade_dem()     hillshade / coloured relief for imshow
dem_extent()    imshow extent of a DEM

Messages go to `say` (a function of one string; default: print with the prefix
"dem:"). Some error texts name the settings DEM_MAX_TILES, DEM_URL_TEMPLATE and
DEM_STYLE, which the scripts use for the arguments max_tiles, url_template and
style.

rasterio is imported only when a GeoTIFF is read, matplotlib only by
shade_dem(). (modules/topo.py is an older, separate SRTM helper; it is not used
here.)

Provenance
----------
Author      : Claude (Claude Sonnet 5.5, Anthropic)
Generated   : 2026-11-27
Notice      : This code is AI-generated. Review and test it before any
              production use.

Changelog
---------
2026-11-27  Claude Sonnet 5.5 (Anthropic), AI-generated, review before use:
            created. The DEM functions of femtic_plot_sitemap.py (written
            2026-11-21, Claude) moved here; changes: explicit `say` and
            `user_agent` arguments; _raster_errors() and _tile_report() are now
            raster_errors() and tile_report(); the constants have lost their
            leading underscore; load_dem() and dem_credit() are new (the DEM
            steps of the script's run()). Tested on synthetic files only, with
            a stand-in for rasterio and without network access.

@author vrath
@project py4mt
@created 2026-11-27
@modified 2026-11-27
"""
from __future__ import annotations

import contextlib
import gzip
import http.client
import math
import os
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EARTH_RADIUS_M = 6371008.8                     # mean Earth radius [m]
HGT_EXTENSIONS = (".hgt", ".hgt.gz", ".hgt.zip", ".SRTMGL1.hgt.zip", ".SRTMGL3.hgt.zip")
RASTER_EXTENSIONS = (".tif", ".tiff")
DEM_SOURCES = ("copernicus30", "copernicus90", "skadi", "tiles", "file", "none")
DEM_STYLES = ("hillshade", "terrain")
USER_AGENT = "py4mt-dem"                       # of the tile downloads
COPERNICUS_NOTICE = ("produced using Copernicus WorldDEM-{n} \u00a9 DLR e.V. 2010-2014 and \u00a9 Airbus "
                     "Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union "
                     "and ESA; all rights reserved")
# tiles of the sources: URL pattern ({name}, {NS}, {lat}, {EW}, {lon}) and the credit line of the maps
DEM_PRESETS: Dict[str, dict] = {
    "copernicus30": dict(
        template=("https://copernicus-dem-30m.s3.amazonaws.com/"
                  "Copernicus_DSM_COG_10_{NS}{lat:02d}_00_{EW}{lon:03d}_00_DEM/"
                  "Copernicus_DSM_COG_10_{NS}{lat:02d}_00_{EW}{lon:03d}_00_DEM.tif"),
        credit="Relief: " + COPERNICUS_NOTICE.format(n=30)),
    "copernicus90": dict(
        template=("https://copernicus-dem-90m.s3.amazonaws.com/"
                  "Copernicus_DSM_COG_30_{NS}{lat:02d}_00_{EW}{lon:03d}_00_DEM/"
                  "Copernicus_DSM_COG_30_{NS}{lat:02d}_00_{EW}{lon:03d}_00_DEM.tif"),
        credit="Relief: " + COPERNICUS_NOTICE.format(n=90)),
    "skadi": dict(
        template="https://s3.amazonaws.com/elevation-tiles-prod/skadi/{NS}{lat:02d}/{name}.hgt.gz",
        credit="Relief: AWS Terrain Tiles (Mapzen), a composite of several elevation data sets"),
}

Say = Callable[[str], None]


def say_default(msg: str) -> None:
    """Print a message of this module."""
    print(f"dem: {msg}")


# ---------------------------------------------------------------------------
# Tiles in the .hgt format: names, download, mosaic
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
               timeout: float, user_agent: str = USER_AGENT,
               say: Say = say_default) -> Tuple[Optional[np.ndarray], str]:
    """
    One tile, from `cache_dir` if it is there, else downloaded (and kept in
    `cache_dir`). Returns (elevation or None, status); status is "cache",
    "downloaded", "missing" (HTTP 403 / 404: the server has no such tile, as for
    the ocean), "absent" (not in the cache, download off) or "failed (reason)".
    """
    name = tile_name(lat0, lon0)
    for ext in HGT_EXTENSIONS:
        p = os.path.join(cache_dir, name + ext)
        if os.path.isfile(p):
            try:
                return read_hgt_file(p), "cache"
            except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
                say(f"WARNING: cannot read {p} ({exc}); "
                     + ("loading it again." if download else "ignored."))
    if not download:
        return None, "absent"
    url = tile_url(template, lat0, lon0)
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
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
        say(f"WARNING: tile {name} not kept in {cache_dir} ({exc}).")
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


def tile_report(status: Dict[Tuple[int, int], str], *, fill: Optional[float],
                strict: bool, say: Say = say_default) -> List[Tuple[int, int]]:
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
        say("WARNING: " + txt + " -- no relief there.")
    sea = [k for k, st in status.items() if st == "missing"]
    if sea:
        say(f"{len(sea)} tile(s) not on the server (ocean?): "
             + (f"set to {fill:g} m." if fill is not None else "no value."))
    return sea


def load_dem_hgt(lon_min: float, lon_max: float, lat_min: float, lat_max: float, *,
                 cache_dir: str, template: str, download: bool = True, timeout: float = 60.0,
                 max_tiles: int = 16, fill: Optional[float] = 0.0,
                 strict: bool = False, user_agent: str = USER_AGENT,
                 say: Say = say_default) -> Optional[dict]:
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
                           download=download, timeout=timeout,
                           user_agent=user_agent, say=say)
        got[(la, lo)], status[(la, lo)] = z, st
        say(f"DEM tile {tile_name(la, lo)}: {st}.")
    sea = tile_report(status, fill=fill, strict=strict, say=say)
    return mosaic_tiles(got, fill=fill, sea=sea)


# ---------------------------------------------------------------------------
# Tiles as GeoTIFF files (rasterio)
# ---------------------------------------------------------------------------
def is_url(text: str) -> bool:
    """True for an http:// or https:// address (else: a path)."""
    return str(text).lower().startswith(("http://", "https://"))


def dem_template(source: str, template: Optional[str]) -> str:
    """Tile pattern of the DEM source: `template` if given, else that of the preset of `source`."""
    if template:
        return str(template)
    if source in DEM_PRESETS:
        return DEM_PRESETS[source]["template"]
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
    if low.endswith(RASTER_EXTENSIONS):
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
                      timeout: float, refresh: bool = False,
                      user_agent: str = USER_AGENT) -> Tuple[Optional[str], str]:
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
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
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


def raster_errors() -> tuple:
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
                    max_pixels: Optional[int] = None, user_agent: str = USER_AGENT,
                    say: Say = say_default) -> Optional[dict]:
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
    errs = raster_errors()
    parts: List[dict] = []
    status: Dict[Tuple[int, int], str] = {}
    for la, lo in tiles:
        kw = dict(cache_dir=cache_dir, template=template, download=download, timeout=timeout,
                  user_agent=user_agent)
        path, st = fetch_raster_tile(la, lo, **kw)
        retry = path is not None and st == "cache" and download
        while path is not None:
            try:
                win = read_raster_window(path, lon_min, lon_max, lat_min, lat_max,
                                         max_pixels=max_pixels)
            except errs as exc:
                if retry:
                    say(f"WARNING: cannot read {path} ({exc}); loading it again.")
                    retry = False
                    path, st = fetch_raster_tile(la, lo, refresh=True, **kw)
                    continue
                path, st = None, f"failed (cannot read {path}: {exc})"
                break
            if win is not None:
                parts.append(win)
            break
        status[(la, lo)] = st
        say(f"DEM tile {tile_name(la, lo)}: {st}.")
    sea = tile_report(status, fill=fill, strict=strict, say=say)
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
    if low.endswith(RASTER_EXTENSIONS):
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
    dx = EARTH_RADIUS_M * math.radians(dlon) * math.cos(math.radians(lat0))
    dy = EARTH_RADIUS_M * math.radians(dlat)
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
# One call for all sources
# ---------------------------------------------------------------------------
def load_dem(source: str, box: Tuple[float, float, float, float], *,
             file: Optional[str] = None, url_template: Optional[str] = None,
             cache_dir: str = "dem_cache", download: bool = True, timeout: float = 60.0,
             max_tiles: int = 16, fill: Optional[float] = 0.0, strict: bool = False,
             max_pixels: int = 1600, user_agent: str = USER_AGENT,
             say: Say = say_default) -> Optional[dict]:
    """
    The DEM of a box, from any source, cropped to the box and thinned out.

    Parameters
    ----------
    source : str
        One of DEM_SOURCES.
    box : tuple
        (lon_min, lon_max, lat_min, lat_max) in degrees.
    file : str, optional
        The DEM file of source "file" (see load_dem_file).
    url_template : str, optional
        Tile pattern (URL or path); None = that of the preset of `source`. Needed
        for source "tiles".
    cache_dir, download, timeout, max_tiles, fill, strict
        Passed to load_dem_raster / load_dem_hgt.
    max_pixels : int
        At most this many pixels per side.
    user_agent : str
        Of the downloads.
    say : callable
        Receives the messages.

    Returns
    -------
    dict or None
        dict(lon, lat, z); None for source "none" or if no tile could be had.

    Raises
    ------
    ValueError, OSError, RuntimeError, ImportError
        Unusable source or file, DEM not covering the box, failed tiles with
        `strict`, no rasterio for a GeoTIFF.
    """
    if source not in DEM_SOURCES:
        raise ValueError(f"DEM source {source!r}: must be one of {list(DEM_SOURCES)}")
    if source == "none":
        return None
    if source == "file":
        if not file:
            raise ValueError("DEM source \"file\" needs a file name")
        dem = load_dem_file(file, box, max_pixels)
    else:
        template = dem_template(source, url_template)
        kw = dict(cache_dir=cache_dir, template=template, download=download, timeout=timeout,
                  max_tiles=max_tiles, fill=fill, strict=strict, user_agent=user_agent, say=say)
        if tile_kind(template) == "raster":
            dem = load_dem_raster(*box, max_pixels=max_pixels, **kw)
        else:
            dem = load_dem_hgt(*box, **kw)
    if dem is None:
        return None
    return decimate_dem(crop_dem(dem, *box), max_pixels)


def dem_credit(source: str, url_template: Optional[str] = None,
               credit: Optional[str] = "auto") -> Optional[str]:
    """
    The credit line of a map: `credit` as given, except "auto" = the line of the
    preset of `source` (None for other sources and with an own `url_template`).
    """
    if credit != "auto":
        return credit
    if source in DEM_PRESETS and not url_template:
        return DEM_PRESETS[source]["credit"]
    return None
