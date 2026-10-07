#!/usr/bin/env python3
"""
modem_femtic_interp.py

Volume-aware interpolation between a ModEM model (regular hexahedral
.rho grid) and a FEMTIC model (tetrahedral mesh + region-based
resistivity block). Both directions are supported:

    DIRECTION = "modem2femtic"
        ModEM .rho model  ->  FEMTIC resistivity_block (free regions only).
        Every free FEMTIC region receives the volume-weighted mean of the
        ModEM cells it overlaps.

    DIRECTION = "femtic2modem"
        FEMTIC mesh + resistivity block  ->  ModEM .rho model on the grid
        of a ModEM template model. Every ModEM cell receives the
        volume-weighted mean of the FEMTIC tetrahedra it overlaps.

@author     vrath
@project    py4mt
@created    2026-10-04
@modified   2026-10-05

Method (what "volume-aware" means here)
---------------------------------------
Exact tetrahedron/hexahedron intersection volumes are expensive and
fragile. Instead, every tetrahedron is covered by a set of equal-weight
sample points (randomly shifted Sobol points mapped uniformly into the
tetrahedron), each carrying the weight  V_tet / N_points. A point's
weight is added to the ModEM cell that contains it (femtic2modem) or the
point's ModEM cell value is added to the FEMTIC region owning the
tetrahedron (modem2femtic). Hence

    value(target) = sum_p w_p * f(source(p)) / sum_p w_p

is a Monte-Carlo estimate of the exact volume-weighted average, and
sum_p w_p over all points equals the tetrahedral volume exactly.
N_points per tetrahedron adapts to the ratio of tetrahedron volume to the
smallest ModEM cell volume near it, so coarse tetrahedra over fine cells
(and the reverse) are resolved. Averaging is done in log10(rho) by
default (geometric mean), alternatively in conductivity or resistivity.

Air is handled explicitly: ModEM cells with rho >= AIR_RHO_THRESHOLD and
FEMTIC air (region 0) are excluded from the averages, so they never
contaminate a neighbouring subsurface value; fixed FEMTIC regions are
never overwritten.

Coordinates
-----------
Internally everything is in the ModEM order (north, east, z-down).
FEMTIC mesh.dat nodes are assumed stored as (northing, easting, z_down)
(MESH_COLUMNS = "ne", the py4mt convention); set "en" if your mesh.dat
holds (easting, northing, z). The ModEM grid is placed in the FEMTIC frame
by matching model centres (COORD_MODE = "centre") or by using the ModEM
reference corner directly (COORD_MODE = "reference"), plus the explicit
OFFSET_* shifts. CHECK the printed bounding boxes before trusting a run.

Georeferencing (COORD_MODE = "georef")
--------------------------------------
Instead of hand-set offsets, the relative position of the two models is
computed from their UTM origins:

    FEMTIC origin (UTM of the mesh centre), FEMTIC_ORIGIN_METHOD =
        "box"          midpoint of the site bounding box from site.dat
                       (canonical, as in femtic_gst_prep.py)
        "calibration"  sites with known lat/lon (or UTM) AND model-local
                       position in observe.dat (x = east, y = north):
                       origin = mean(UTM_site - model_xy)
        "manual"       FEMTIC_ORIGIN_UTM = (easting, northing)
    ModEM origin (UTM of the ModEM data-frame origin, i.e. where the
        data-file X/Y = 0), MODEM_ORIGIN_METHOD =
        "data"         from the ModEM data file: per site lat/lon and
                       model X (north), Y (east): origin = mean(UTM_site -
                       (Y, X)); the scatter over sites is printed
        "manual"       MODEM_ORIGIN_LATLON = (lat, lon)
    The ModEM grid is then placed with its stored reference corner in the
    data frame, shifted by (UTM_modem - UTM_femtic). Both origins use the
    same UTM zone (UTM_ZONE, or derived from the site mean).

Provenance
----------
Author    : Claude (Anthropic)
Generated : 2026-10-04
Notice    : This code is AI-generated. Review and test it before any
            production use. Verified with ast.parse() and a synthetic-
            data test (see modem_femtic_interp_readme.md); NOT yet run
            against real ModEM / FEMTIC files.

Changelog
---------
2026-10-04  Claude Sonnet 5.5 (Anthropic): created.
2026-10-05  Claude Sonnet 5.5 (Anthropic): added COORD_MODE="georef":
            FEMTIC origin from site.dat bounding box / observe.dat
            calibration sites, ModEM origin from the data file lat/lon
            and X/Y; built-in UTM conversion (latlon_to_utm).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence, Tuple

import numpy as np

# ===========================================================================
# Configuration (script level; library functions below take explicit args)
# ===========================================================================

DIRECTION = "modem2femtic"          # "modem2femtic" or "femtic2modem"

# --- ModEM side ---
MODEM_MODEL = "model_in"            # base name of ModEM model (source for
                                    #   modem2femtic; grid TEMPLATE for
                                    #   femtic2modem)
MODEM_EXT = ".rho"
MODEM_OUT = "model_from_femtic"     # femtic2modem: output base name
MODEM_OUT_TRANS = "LOGE"            # trans written to the ModEM file
MODEM_USE_REFERENCE_Z = False       # add reference[2] to the ModEM z edges

# --- FEMTIC side ---
FEMTIC_MESH = "mesh.dat"
FEMTIC_BLOCK = "resistivity_block_iter0.dat"   # template (modem2femtic) /
                                               #   source (femtic2modem)
FEMTIC_OUT = "resistivity_block_from_modem.dat"  # modem2femtic: output
MESH_COLUMNS = "ne"                 # "ne": (north, east, z); "en": swap
OCEAN = None                        # None = auto-infer, True/False = force
INCLUDE_OCEAN = True                # femtic2modem: ocean elements count as
                                    #   conductive cells (True) or not
CLIP_TO_BOUNDS = True               # modem2femtic: clip to region rho_lo/hi

# --- placement of the ModEM grid in the FEMTIC frame ---
COORD_MODE = "centre"               # "centre", "reference" or "georef"
OFFSET_N = 0.0                      # shift of the ModEM grid [m]
OFFSET_E = 0.0
OFFSET_Z = 0.0

# --- georeferencing (used when COORD_MODE = "georef") ---
UTM_ZONE = None                     # None = derive from site mean lat/lon
UTM_NORTHERN = None                 # None = derive from site mean latitude
FEMTIC_ORIGIN_METHOD = "box"        # "box" | "calibration" | "manual"
SITE_DAT = "site.dat"               # box: CSV name,lat,lon,elev,num,E,N
OBSERVE_DAT = "observe.dat"         # calibration: model-local site x/y [km]
CALIBRATION_SITES = []              # calibration: [{"site": 3,
                                    #   "crs": "latlon"|"utm",
                                    #   "coords": [lon, lat] | [E, N]}, ...]
FEMTIC_ORIGIN_UTM = None            # manual: (easting, northing) [m]
MODEM_ORIGIN_METHOD = "data"        # "data" | "manual"
MODEM_DATA = "modem_data"           # base name of ModEM data file
MODEM_DATA_EXT = ".dat"
MODEM_ORIGIN_LATLON = None          # manual: (lat, lon) of data-frame origin

# --- averaging ---
AVERAGING = "log10"                 # "log10", "sigma", or "rho"
AIR_RHO_THRESHOLD = 1.0e10          # rho >= this is treated as air [Ohm.m]
MIN_COVERAGE = 0.10                 # min fraction of a target's volume that
                                    #   must be covered by valid source
                                    #   volume, else the template value is
                                    #   kept
CENTRE_FALLBACK = True              # femtic2modem: poorly covered cells get
                                    #   the value of the tetrahedron that
                                    #   contains the cell centre

# --- sampling ---
OVERSAMPLE = 2.0                    # sample points per (tet / min cell) ratio
N_MIN = 4                           # min points per tetrahedron
N_MAX = 4096                        # max points per tetrahedron
MAX_BATCH_POINTS = 2_000_000        # memory cap per processing batch
CHUNK_TETS = 100_000
SEED = 0

DIAGNOSTICS_NPZ = None              # e.g. "interp_diag.npz" (coverage etc.)


# ===========================================================================
# Library functions
# ===========================================================================

def _fwd(rho: np.ndarray, mode: str) -> np.ndarray:
    """Transform resistivity [Ohm.m] to the averaging variable."""
    m = str(mode).lower()
    if m == "log10":
        return np.log10(rho)
    if m == "sigma":
        return 1.0 / rho
    if m == "rho":
        return np.asarray(rho, dtype=float)
    raise ValueError(f"unknown averaging mode {mode!r}; use log10|sigma|rho")


def _inv(val: np.ndarray, mode: str) -> np.ndarray:
    """Inverse of :func:`_fwd`."""
    m = str(mode).lower()
    if m == "log10":
        return np.power(10.0, val)
    if m == "sigma":
        return 1.0 / val
    if m == "rho":
        return np.asarray(val, dtype=float)
    raise ValueError(f"unknown averaging mode {mode!r}; use log10|sigma|rho")


def tet_volumes_np(nodes: np.ndarray, conn: np.ndarray) -> np.ndarray:
    """Absolute tetrahedron volumes, V = |det([b-a, c-a, d-a])| / 6."""
    v = nodes[conn]
    a, b, c, d = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
    return np.abs((np.cross(b - a, c - a) * (d - a)).sum(axis=1)) / 6.0


class ModemGrid:
    """ModEM grid placed in the (north, east, z-down) frame of a FEMTIC mesh.

    Parameters
    ----------
    dx, dy, dz : ndarray
        ModEM cell sizes in north, east and depth direction [m].
    reference : sequence of 3 floats
        ModEM reference (corner) as returned by modem.read_mod.
    coord_mode : {"centre", "reference"}
        "centre": the ModEM model centre is placed at the FEMTIC origin
        (plus offsets). "reference": ModEM reference coordinates are used
        as they are (plus offsets).
    offset : sequence of 3 floats
        Extra shift (north, east, z) applied to the grid [m].
    use_reference_z : bool
        If True, reference[2] is added to the z edges.
    """

    def __init__(self, dx, dy, dz, reference, *, coord_mode="centre",
                 offset=(0.0, 0.0, 0.0), use_reference_z=False):
        self.d = [np.asarray(dx, float).ravel(),
                  np.asarray(dy, float).ravel(),
                  np.asarray(dz, float).ravel()]
        ref = [float(r) for r in list(reference)[:3]]
        mode = str(coord_mode).lower()
        if mode not in ("centre", "center", "reference"):
            raise ValueError("coord_mode must be 'centre' or 'reference' "
                             "(georef is resolved by run())")
        e = []
        for ax in range(3):
            edges = np.concatenate([[0.0], np.cumsum(self.d[ax])])
            if ax < 2:
                edges = edges + ref[ax]
                if mode.startswith("cen"):
                    edges = edges - 0.5 * (edges[0] + edges[-1])
            else:
                if use_reference_z:
                    edges = edges + ref[2]
            e.append(edges + float(offset[ax]))
        self.e = e
        self.shape = tuple(len(x) for x in self.d)

    def bbox(self) -> np.ndarray:
        """(3, 2) array of [min, max] per axis."""
        return np.array([[ed[0], ed[-1]] for ed in self.e])

    def cell_volumes(self) -> np.ndarray:
        """Cell volume array, shape (nx, ny, nz)."""
        return (self.d[0][:, None, None] * self.d[1][None, :, None]
                * self.d[2][None, None, :])

    def locate(self, pts: np.ndarray) -> np.ndarray:
        """Flat (C-order) cell index per point, -1 if outside the grid."""
        idx = []
        ok = np.ones(len(pts), dtype=bool)
        for ax in range(3):
            i = np.searchsorted(self.e[ax], pts[:, ax], side="right") - 1
            ok &= (i >= 0) & (i < self.shape[ax])
            idx.append(np.clip(i, 0, self.shape[ax] - 1))
        flat = (idx[0] * self.shape[1] + idx[1]) * self.shape[2] + idx[2]
        return np.where(ok, flat, -1)

    def local_cell_volume(self, pts: np.ndarray) -> np.ndarray:
        """Volume of the nearest (clamped) cell for each point."""
        v = np.ones(len(pts))
        for ax in range(3):
            i = np.searchsorted(self.e[ax], pts[:, ax], side="right") - 1
            i = np.clip(i, 0, self.shape[ax] - 1)
            v = v * self.d[ax][i]
        return v


def iter_tet_samples(nodes: np.ndarray, conn: np.ndarray, grid: ModemGrid, *,
                     oversample: float = 2.0, n_min: int = 4,
                     n_max: int = 4096, chunk: int = 100_000,
                     max_batch_points: int = 2_000_000,
                     seed: int = 0, stats: Optional[dict] = None
                     ) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield equal-weight sample points inside tetrahedra.

    Yields
    ------
    tid : int ndarray (m,)   tetrahedron index of each point
    pts : float ndarray (m, 3)   point coordinates (same frame as nodes)
    w   : float ndarray (m,)   weight = V_tet / N_points of its tetrahedron

    Notes
    -----
    N_points per tetrahedron is a power of two, chosen as
    ceil(oversample * V_tet / V_cell_min), clipped to [n_min, n_max], where
    V_cell_min is the smallest grid-cell volume found at the centroid and
    the four vertices. Points are randomly shifted (Cranley-Patterson)
    Sobol points mapped uniformly into the tetrahedron via sorted
    uniforms. `stats` (if given) accumulates 'n_points', 'n_capped'.
    """
    from scipy.stats import qmc

    rng = np.random.default_rng(seed)
    vol = tet_volumes_np(nodes, conn)
    sob_cache: Dict[int, np.ndarray] = {}
    nelem = conn.shape[0]

    def sobol(lv: int) -> np.ndarray:
        if lv not in sob_cache:
            s = qmc.Sobol(d=3, scramble=True, seed=seed + lv)
            sob_cache[lv] = s.random_base2(m=lv)
        return sob_cache[lv]

    for start in range(0, nelem, chunk):
        idx = np.arange(start, min(start + chunk, nelem))
        v = nodes[conn[idx]]                                # (n, 4, 3)
        probes = np.concatenate([v.mean(axis=1)[:, None, :], v], axis=1)
        cmin = np.min(
            [grid.local_cell_volume(probes[:, k, :]) for k in range(5)],
            axis=0)
        ratio = vol[idx] / np.maximum(cmin, 1e-30)
        want = np.ceil(oversample * ratio)
        npts = np.clip(want, n_min, n_max)
        level = np.ceil(np.log2(npts)).astype(int)
        if stats is not None:
            stats["n_capped"] = stats.get("n_capped", 0) + int(
                np.sum(want > n_max))

        for lv in np.unique(level):
            sel = np.where(level == lv)[0]
            n_p = 2 ** int(lv)
            u0 = sobol(int(lv))
            bs = max(1, max_batch_points // n_p)
            for b0 in range(0, sel.size, bs):
                sb = sel[b0:b0 + bs]
                shifts = rng.random((sb.size, 1, 3))
                u = (u0[None, :, :] + shifts) % 1.0
                s = np.sort(u, axis=2)
                bary = np.stack([s[..., 0], s[..., 1] - s[..., 0],
                                 s[..., 2] - s[..., 1], 1.0 - s[..., 2]],
                                axis=-1)                    # (t, n, 4)
                pts = np.einsum("tnk,tkd->tnd", bary, v[sb])
                w = np.repeat((vol[idx[sb]] / n_p)[:, None], n_p, axis=1)
                tid = np.repeat(idx[sb][:, None], n_p, axis=1)
                if stats is not None:
                    stats["n_points"] = stats.get("n_points", 0) + tid.size
                yield tid.ravel(), pts.reshape(-1, 3), w.ravel()


def gather_at_points(pts: np.ndarray, nodes: np.ndarray, conn: np.ndarray,
                     values: np.ndarray, k: int = 16) -> np.ndarray:
    """Value of the containing tetrahedron for each point (NaN if none).

    Candidates are the k tetrahedra with nearest centroids; containment is
    tested with barycentric coordinates (tolerance 1e-10).
    """
    from scipy.spatial import cKDTree

    v = nodes[conn]
    v0 = v[:, 0]
    T = np.stack([v[:, 1] - v0, v[:, 2] - v0, v[:, 3] - v0], axis=2)
    det = np.linalg.det(T)
    good = np.abs(det) > 1e-30
    Tinv = np.zeros_like(T)
    Tinv[good] = np.linalg.inv(T[good])
    tree = cKDTree(v.mean(axis=1))
    kk = min(int(k), conn.shape[0])
    _, nb = tree.query(pts, k=kk)
    nb = nb.reshape(len(pts), kk)
    lam = np.einsum("qkij,qkj->qki", Tinv[nb], pts[:, None, :] - v0[nb])
    tol = 1e-10
    inside = good[nb] & np.all(lam >= -tol, axis=2) & (lam.sum(axis=2) <= 1 + tol)
    found = inside.any(axis=1)
    first = inside.argmax(axis=1)
    res = np.full(len(pts), np.nan)
    sel = nb[np.arange(len(pts)), first]
    res[found] = values[sel[found]]
    return res


def modem_to_regions(grid: ModemGrid, rho: np.ndarray, nodes: np.ndarray,
                     conn: np.ndarray, elem_region: np.ndarray, nreg: int, *,
                     averaging: str = "log10",
                     air_threshold: float = 1.0e10,
                     oversample: float = 2.0, n_min: int = 4,
                     n_max: int = 4096, chunk: int = 100_000,
                     max_batch_points: int = 2_000_000, seed: int = 0
                     ) -> dict:
    """Volume-weighted mean of ModEM cells per FEMTIC region.

    Parameters
    ----------
    grid : ModemGrid
    rho : ndarray (nx, ny, nz)   ModEM resistivity, linear [Ohm.m].
    nodes, conn : mesh in the grid's (north, east, z) frame.
    elem_region : int ndarray (nelem,)   region index of each tetrahedron.
    nreg : int   number of regions.

    Returns
    -------
    dict with
        rho      (nreg,) region resistivity [Ohm.m], NaN if no valid sample
        coverage (nreg,) valid sampled volume / total region volume
        volume   (nreg,) total region volume [m^3]
        mean_f   float   global volume-weighted mean of the averaging
                         variable over all used samples (diagnostic)
        stats    dict    sampling statistics
    """
    f = _fwd(rho, averaging).ravel()
    valid = ((rho < air_threshold) & (rho > 0)).ravel() & np.isfinite(f)
    s_wf = np.zeros(nreg)
    s_w = np.zeros(nreg)
    s_all = np.zeros(nreg)
    stats: dict = {}
    for tid, pts, w in iter_tet_samples(
            nodes, conn, grid, oversample=oversample, n_min=n_min,
            n_max=n_max, chunk=chunk, max_batch_points=max_batch_points,
            seed=seed, stats=stats):
        reg = elem_region[tid]
        flat = grid.locate(pts)
        use = flat >= 0
        use[use] = valid[flat[use]]
        s_all += np.bincount(reg, weights=w, minlength=nreg)
        s_w += np.bincount(reg[use], weights=w[use], minlength=nreg)
        s_wf += np.bincount(reg[use], weights=w[use] * f[flat[use]],
                            minlength=nreg)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_f = np.where(s_w > 0, s_wf / s_w, np.nan)
        cov = np.where(s_all > 0, s_w / s_all, 0.0)
    out_rho = np.full(nreg, np.nan)
    ok = np.isfinite(mean_f)
    out_rho[ok] = _inv(mean_f[ok], averaging)
    return dict(rho=out_rho, coverage=cov, volume=s_all,
                mean_f=float(s_wf.sum() / s_w.sum()) if s_w.sum() > 0 else np.nan,
                weighted_mean_regions=float(
                    np.nansum(mean_f * s_w) / s_w.sum()) if s_w.sum() > 0 else np.nan,
                stats=stats)


def elements_to_modem(grid: ModemGrid, nodes: np.ndarray, conn: np.ndarray,
                      elem_rho: np.ndarray, elem_mask: np.ndarray,
                      rho_template: np.ndarray, *,
                      averaging: str = "log10",
                      air_threshold: float = 1.0e10,
                      min_coverage: float = 0.10,
                      centre_fallback: bool = True,
                      oversample: float = 2.0, n_min: int = 4,
                      n_max: int = 4096, chunk: int = 100_000,
                      max_batch_points: int = 2_000_000, seed: int = 0
                      ) -> dict:
    """Volume-weighted mean of FEMTIC tetrahedra per ModEM cell.

    Parameters
    ----------
    elem_rho : (nelem,) element resistivity [Ohm.m].
    elem_mask : (nelem,) bool, True for elements that take part
        (air and, optionally, ocean excluded by the caller).
    rho_template : (nx, ny, nz) ModEM template; cells that are air in the
        template are never changed, and cells without enough source
        coverage keep their template value.

    Returns
    -------
    dict with
        rho       (nx, ny, nz) new model, linear [Ohm.m]
        coverage  (nx, ny, nz) sampled source volume / cell volume
        n_updated, n_fallback, n_kept, n_air : int
        mean_f_source, mean_f_target : float   diagnostics (see Notes)
        stats     dict

    Notes
    -----
    mean_f_source is the volume-weighted mean of the averaging variable
    over all sampled source volume inside the grid; mean_f_target is the
    same quantity recomputed from the new cells, weighted with the sampled
    volume of each cell. They agree to round-off by construction -- a
    check on the bookkeeping, not on the interpolation error.
    """
    shape = grid.shape
    ncell = int(np.prod(shape))
    sel = np.asarray(elem_mask, dtype=bool) & (elem_rho > 0) \
        & (elem_rho < air_threshold) & np.isfinite(elem_rho)
    conn_s = conn[sel]
    f_s = _fwd(elem_rho[sel], averaging)

    s_wf = np.zeros(ncell)
    s_w = np.zeros(ncell)
    stats: dict = {}
    for tid, pts, w in iter_tet_samples(
            nodes, conn_s, grid, oversample=oversample, n_min=n_min,
            n_max=n_max, chunk=chunk, max_batch_points=max_batch_points,
            seed=seed, stats=stats):
        flat = grid.locate(pts)
        ins = flat >= 0
        s_w += np.bincount(flat[ins], weights=w[ins], minlength=ncell)
        s_wf += np.bincount(flat[ins], weights=w[ins] * f_s[tid[ins]],
                            minlength=ncell)

    cvol = grid.cell_volumes().ravel()
    cov = s_w / cvol
    tmpl = np.asarray(rho_template, dtype=float).ravel()
    is_air = tmpl >= air_threshold
    accepted = (cov >= min_coverage) & (s_w > 0) & ~is_air

    new = tmpl.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_f = np.where(s_w > 0, s_wf / s_w, np.nan)
    new[accepted] = _inv(mean_f[accepted], averaging)

    n_fb = 0
    if centre_fallback and conn_s.shape[0] > 0:
        todo = np.where(~accepted & ~is_air)[0]
        if todo.size:
            ijk = np.unravel_index(todo, shape)
            ctr = np.stack(
                [0.5 * (grid.e[ax][ijk[ax]] + grid.e[ax][ijk[ax] + 1])
                 for ax in range(3)], axis=1)
            got = gather_at_points(ctr, nodes, conn_s, f_s)
            hit = np.isfinite(got)
            new[todo[hit]] = _inv(got[hit], averaging)
            n_fb = int(hit.sum())

    fsrc = float(s_wf.sum() / s_w.sum()) if s_w.sum() > 0 else np.nan
    wt = s_w[accepted]
    ftgt = float((mean_f[accepted] * wt).sum() / wt.sum()) if wt.sum() > 0 else np.nan
    return dict(
        rho=new.reshape(shape), coverage=cov.reshape(shape),
        n_updated=int(accepted.sum()), n_fallback=n_fb,
        n_kept=int((~accepted).sum()) - n_fb - int((is_air & ~accepted).sum()),
        n_air=int(is_air.sum()),
        mean_f_source=fsrc, mean_f_target=ftgt, stats=stats)


# ===========================================================================
# Georeferencing (UTM origins of the two models)
# ===========================================================================

_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563


def utm_zone_from_latlon(lat: float, lon: float) -> Tuple[int, bool]:
    """Standard UTM zone number and hemisphere (Norway/Svalbard exceptions
    are ignored; set UTM_ZONE explicitly there)."""
    zone = int(np.floor((float(lon) + 180.0) / 6.0)) % 60 + 1
    return zone, float(lat) >= 0.0


def latlon_to_utm(lat, lon, zone: int, northern: bool
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """WGS84 geographic -> UTM (easting, northing) [m] for a given zone.

    Kruger n-series (6th order), accuracy well below 1 mm within the zone.
    Vectorised; `lat`, `lon` in decimal degrees.
    """
    lat = np.radians(np.asarray(lat, dtype=float))
    dlon = np.radians(np.asarray(lon, dtype=float) - (6.0 * zone - 183.0))
    k0, e0, n0 = 0.9996, 500000.0, (0.0 if northern else 10000000.0)
    f = _WGS84_F
    n = f / (2.0 - f)
    big_a = _WGS84_A / (1.0 + n) * (1.0 + n**2 / 4.0 + n**4 / 64.0
                                    + n**6 / 256.0)
    alpha = (
        n / 2 - 2 * n**2 / 3 + 5 * n**3 / 16 + 41 * n**4 / 180
        - 127 * n**5 / 288 + 7891 * n**6 / 37800,
        13 * n**2 / 48 - 3 * n**3 / 5 + 557 * n**4 / 1440
        + 281 * n**5 / 630 - 1983433 * n**6 / 1935360,
        61 * n**3 / 240 - 103 * n**4 / 140 + 15061 * n**5 / 26880
        + 167603 * n**6 / 181440,
        49561 * n**4 / 161280 - 179 * n**5 / 168 + 6601661 * n**6 / 7257600,
        34729 * n**5 / 80640 - 3418889 * n**6 / 1995840,
        212378941 * n**6 / 319334400)
    t = np.sinh(np.arctanh(np.sin(lat))
                - (2.0 * np.sqrt(n) / (1.0 + n))
                * np.arctanh(2.0 * np.sqrt(n) / (1.0 + n) * np.sin(lat)))
    xi = np.arctan2(t, np.cos(dlon))
    eta = np.arctanh(np.sin(dlon) / np.sqrt(1.0 + t * t))
    xs, es = xi.copy(), eta.copy()
    for j, a in enumerate(alpha, start=1):
        xs = xs + a * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
        es = es + a * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
    return e0 + k0 * big_a * es, n0 + k0 * big_a * xs


def origin_from_bbox(easting: np.ndarray, northing: np.ndarray
                     ) -> Tuple[float, float]:
    """Bounding-box midpoint (UTM origin of the FEMTIC mesh centre)."""
    e = np.asarray(easting, float)
    n = np.asarray(northing, float)
    return (0.5 * float(e.min() + e.max()), 0.5 * float(n.min() + n.max()))


def origin_from_pairs(site_en: np.ndarray, local_en: np.ndarray
                      ) -> Tuple[float, float, np.ndarray]:
    """UTM origin from sites with known UTM and local (east, north) [m].

    origin = mean(site_utm - local). Returns (E0, N0, residuals (n, 2)),
    residuals = individual implied origin minus mean.
    """
    d = np.asarray(site_en, float) - np.asarray(local_en, float)
    m = d.mean(axis=0)
    return float(m[0]), float(m[1]), d - m


def _fmt_zone(zone: int, northern: bool) -> str:
    return "%d%s" % (zone, "N" if northern else "S")


# ===========================================================================
# Script-level helpers (I/O, reporting)
# ===========================================================================

def _swap_mesh_columns(nodes: np.ndarray, mesh_columns: str) -> np.ndarray:
    """Return nodes in (north, east, z) order."""
    mc = str(mesh_columns).lower()
    if mc == "ne":
        return nodes
    if mc == "en":
        return nodes[:, [1, 0, 2]]
    raise ValueError("MESH_COLUMNS must be 'ne' or 'en'")


def _read_modem_sites(mod) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                    np.ndarray]:
    """Unique ModEM sites: lat, lon, X (north) [m], Y (east) [m]."""
    site, _, data, _ = mod.read_data(Datfile=MODEM_DATA,
                                     modext=MODEM_DATA_EXT, out=False)
    names = np.asarray(site, dtype=str)
    _, first = np.unique(names, return_index=True)
    first = np.sort(first)
    return data[first, 1], data[first, 2], data[first, 3], data[first, 4]


def _georef_offsets(fem, mod) -> Tuple[float, float]:
    """Compute (OFFSET_N, OFFSET_E) of the ModEM data frame in the FEMTIC
    frame from the two UTM origins; prints a summary table."""
    meth = str(FEMTIC_ORIGIN_METHOD).lower()
    mmeth = str(MODEM_ORIGIN_METHOD).lower()
    print("  Georeferencing (UTM)")
    print("  " + "-" * 66)

    # --- gather raw site information (zone needs a lat/lon hint) ---
    rows = None
    if meth == "box":
        rows = fem.read_site_dat(SITE_DAT)
        if not rows:
            raise ValueError("site.dat %r contains no sites" % SITE_DAT)
    m_lat = m_lon = m_x = m_y = None
    if mmeth == "data":
        m_lat, m_lon, m_x, m_y = _read_modem_sites(mod)

    hint = None
    if rows is not None:
        hint = (float(np.mean([r["lat"] for r in rows])),
                float(np.mean([r["lon"] for r in rows])))
    elif m_lat is not None:
        hint = (float(np.mean(m_lat)), float(np.mean(m_lon)))
    elif MODEM_ORIGIN_LATLON is not None:
        hint = (float(MODEM_ORIGIN_LATLON[0]), float(MODEM_ORIGIN_LATLON[1]))
    if UTM_ZONE is not None:
        zone = int(UTM_ZONE)
        north = bool(UTM_NORTHERN) if UTM_NORTHERN is not None else (
            True if hint is None else hint[0] >= 0.0)
    else:
        if hint is None:
            raise ValueError("cannot derive the UTM zone: set UTM_ZONE")
        zone, north = utm_zone_from_latlon(hint[0], hint[1])
        if UTM_NORTHERN is not None:
            north = bool(UTM_NORTHERN)
    print("  UTM zone                : %s" % _fmt_zone(zone, north))

    # --- FEMTIC origin ---
    if meth == "box":
        e = np.array([r["easting"] for r in rows])
        n = np.array([r["northing"] for r in rows])
        f_e, f_n = origin_from_bbox(e, n)
        print("  FEMTIC origin (box, %d sites in %s)" % (len(rows), SITE_DAT))
    elif meth == "calibration":
        if not CALIBRATION_SITES:
            raise ValueError("CALIBRATION_SITES is empty")
        s_en, l_en = [], []
        for c in CALIBRATION_SITES:
            if "x_km" in c and "y_km" in c:
                x_m, y_m = 1e3 * float(c["x_km"]), 1e3 * float(c["y_km"])
            else:
                x_m, y_m = fem.read_site_position(OBSERVE_DAT, int(c["site"]))
            if str(c["crs"]) == "latlon":
                lon_d, lat_d = c["coords"]
                ee, nn = latlon_to_utm(lat_d, lon_d, zone, north)
            elif str(c["crs"]) == "utm":
                ee, nn = float(c["coords"][0]), float(c["coords"][1])
            else:
                raise ValueError("calibration crs must be 'latlon' or 'utm'")
            s_en.append([float(ee), float(nn)])
            l_en.append([x_m, y_m])         # observe.dat: x = east, y = north
        f_e, f_n, res = origin_from_pairs(np.array(s_en), np.array(l_en))
        print("  FEMTIC origin (calibration, %d sites; max residual "
              "%.1f m)" % (len(s_en), float(np.abs(res).max())))
    elif meth == "manual":
        if FEMTIC_ORIGIN_UTM is None:
            raise ValueError("FEMTIC_ORIGIN_UTM must be set for 'manual'")
        f_e, f_n = float(FEMTIC_ORIGIN_UTM[0]), float(FEMTIC_ORIGIN_UTM[1])
        print("  FEMTIC origin (manual)")
    else:
        raise ValueError("FEMTIC_ORIGIN_METHOD must be box|calibration|"
                         "manual")
    print("    E %.2f  N %.2f" % (f_e, f_n))

    # --- ModEM origin (data-frame origin: X = Y = 0) ---
    if mmeth == "data":
        se, sn = latlon_to_utm(m_lat, m_lon, zone, north)
        # ModEM data: X = north, Y = east
        m_e, m_n, res = origin_from_pairs(
            np.c_[se, sn], np.c_[m_y, m_x])
        print("  ModEM origin (data file %s%s, %d sites; max residual "
              "%.1f m)" % (MODEM_DATA, MODEM_DATA_EXT, m_lat.size,
                           float(np.abs(res).max())))
    elif mmeth == "manual":
        if MODEM_ORIGIN_LATLON is None:
            raise ValueError("MODEM_ORIGIN_LATLON must be set for 'manual'")
        m_e, m_n = latlon_to_utm(MODEM_ORIGIN_LATLON[0],
                                 MODEM_ORIGIN_LATLON[1], zone, north)
        m_e, m_n = float(m_e), float(m_n)
        print("  ModEM origin (manual lat/lon)")
    else:
        raise ValueError("MODEM_ORIGIN_METHOD must be data|manual")
    print("    E %.2f  N %.2f" % (m_e, m_n))

    off_n, off_e = m_n - f_n, m_e - f_e
    print("  ModEM frame offset in FEMTIC frame: N %.1f m  E %.1f m"
          % (off_n, off_e))
    print()
    return off_n, off_e


def _print_bboxes(grid: ModemGrid, nodes: np.ndarray) -> None:
    gb = grid.bbox()
    mb = np.array([[nodes[:, a].min(), nodes[:, a].max()] for a in range(3)])
    names = ("north", "east", "z")
    print("  Frame check (north, east, z-down) [m]:")
    print("  %-6s  %28s  %28s" % ("axis", "ModEM grid (placed)", "FEMTIC mesh"))
    print("  " + "-" * 66)
    for a in range(3):
        print("  %-6s  %13.1f .. %-12.1f  %13.1f .. %-12.1f"
              % (names[a], gb[a, 0], gb[a, 1], mb[a, 0], mb[a, 1]))
    ov = [max(0.0, min(gb[a, 1], mb[a, 1]) - max(gb[a, 0], mb[a, 0]))
          for a in range(3)]
    print("  overlap extent: N %.1f  E %.1f  Z %.1f m" % tuple(ov))
    if min(ov) <= 0.0:
        print("  *** WARNING: grid and mesh do not overlap -- check "
              "COORD_MODE / OFFSET_* / MESH_COLUMNS! ***")


def _fmt_ratio(a: float, b: float) -> str:
    return "n/a" if b == 0 else "%.1f%%" % (100.0 * a / b)


def run() -> None:
    """Run the configured interpolation (uses the module-level config)."""
    import femtic as fem
    import modem as mod

    t0 = time.time()
    print("modem_femtic_interp: DIRECTION=%s  AVERAGING=%s"
          % (DIRECTION, AVERAGING))

    dx, dy, dz, rho, ref, _ = mod.read_mod(
        file=MODEM_MODEL, modext=MODEM_EXT, trans="LINEAR", out=True)
    if str(COORD_MODE).lower() == "georef":
        off_n, off_e = _georef_offsets(fem, mod)
        grid = ModemGrid(dx, dy, dz, ref, coord_mode="reference",
                         offset=(off_n + OFFSET_N, off_e + OFFSET_E,
                                 OFFSET_Z),
                         use_reference_z=MODEM_USE_REFERENCE_Z)
    else:
        grid = ModemGrid(dx, dy, dz, ref, coord_mode=COORD_MODE,
                         offset=(OFFSET_N, OFFSET_E, OFFSET_Z),
                         use_reference_z=MODEM_USE_REFERENCE_Z)

    nodes, conn = fem.read_femtic_mesh(FEMTIC_MESH)
    nodes = _swap_mesh_columns(np.asarray(nodes, float), MESH_COLUMNS)
    st = fem._read_resistivity_block_struct(
        FEMTIC_BLOCK, model_trans="none", ocean=OCEAN, out=True)
    elem_region = np.asarray(st["elem_region"], dtype=np.int64)
    nreg = int(st["nreg"])
    if elem_region.size != conn.shape[0]:
        raise ValueError("mesh (%d elements) and resistivity block (%d) "
                         "disagree" % (conn.shape[0], elem_region.size))
    _print_bboxes(grid, nodes)

    kw = dict(averaging=AVERAGING, air_threshold=AIR_RHO_THRESHOLD,
              oversample=OVERSAMPLE, n_min=N_MIN, n_max=N_MAX, chunk=CHUNK_TETS,
              max_batch_points=MAX_BATCH_POINTS, seed=SEED)

    if DIRECTION == "modem2femtic":
        res = modem_to_regions(grid, rho, nodes, conn, elem_region, nreg, **kw)
        free_idx = np.asarray(st["free_idx"], dtype=np.int64)
        region_rho = np.asarray(st["region_rho"], dtype=float)
        lo = np.asarray(st["region_lo"], dtype=float)
        hi = np.asarray(st["region_hi"], dtype=float)

        new_free = res["rho"][free_idx].copy()
        cov = res["coverage"][free_idx]
        bad = ~np.isfinite(new_free) | (cov < MIN_COVERAGE)
        new_free[bad] = region_rho[free_idx][bad]
        n_clip = 0
        if CLIP_TO_BOUNDS:
            okb = (lo[free_idx] > 0) & (hi[free_idx] > lo[free_idx])
            clipped = np.where(okb, np.clip(new_free, lo[free_idx], hi[free_idx]),
                               new_free)
            n_clip = int(np.sum(clipped != new_free))
            new_free = clipped

        ocean_present = bool(st["meta"]["ocean_present"])
        air_rho = float(region_rho[0])
        ocean_rho = float(region_rho[1]) if (ocean_present and nreg > 1) else 0.25
        fem.insert_model(template=FEMTIC_BLOCK, model=np.log10(new_free),
                         model_file=FEMTIC_OUT, out=True, ocean=OCEAN,
                         air_rho=air_rho, ocean_rho=ocean_rho)

        vol = res["volume"][free_idx]
        print("\n  Summary (free regions)")
        print("  " + "-" * 60)
        print("  regions free / updated / kept (template): %d / %d / %d"
              % (free_idx.size, int((~bad).sum()), int(bad.sum())))
        print("  clipped to region bounds              : %d" % n_clip)
        print("  coverage (volume) min / median        : %.3f / %.3f"
              % (cov.min() if cov.size else np.nan,
                 np.median(cov) if cov.size else np.nan))
        print("  volume kept at template value         : %s"
              % _fmt_ratio(float(vol[bad].sum()), float(vol.sum())))
        print("  sample points / capped tets           : %d / %d"
              % (res["stats"].get("n_points", 0),
                 res["stats"].get("n_capped", 0)))
        print("  mean %s source (sampled) / regions: %.5g / %.5g"
              % (AVERAGING, res["mean_f"], res["weighted_mean_regions"]))
        print("  written: %s" % FEMTIC_OUT)
        diag = dict(region_rho=res["rho"], coverage=res["coverage"],
                    volume=res["volume"], free_idx=free_idx)

    elif DIRECTION == "femtic2modem":
        region_rho = np.asarray(st["region_rho"], dtype=float)
        elem_rho = region_rho[elem_region]
        mask = elem_region != 0                         # air always excluded
        if not INCLUDE_OCEAN and st["meta"]["ocean_present"]:
            mask &= elem_region != 1
        res = elements_to_modem(
            grid, nodes, conn, elem_rho, mask, rho,
            min_coverage=MIN_COVERAGE, centre_fallback=CENTRE_FALLBACK, **{
                k: v for k, v in kw.items()})
        new = np.array(res["rho"], dtype=float)
        mod.write_mod(file=MODEM_OUT, modext=MODEM_EXT, dx=dx, dy=dy, dz=dz,
                      mval=new, reference=ref, trans=MODEM_OUT_TRANS,
                      aircells=None, blank=1.0e-30,
                      header="# ModEM model interpolated from FEMTIC "
                             "(volume-weighted, modem_femtic_interp.py)",
                      out=True)
        ncell = int(np.prod(grid.shape))
        print("\n  Summary (ModEM cells)")
        print("  " + "-" * 60)
        print("  cells total / air (kept)            : %d / %d"
              % (ncell, res["n_air"]))
        print("  updated from FEMTIC volume          : %d" % res["n_updated"])
        print("  updated from cell-centre fallback   : %d" % res["n_fallback"])
        print("  kept at template value              : %d" % res["n_kept"])
        print("  sample points / capped tets         : %d / %d"
              % (res["stats"].get("n_points", 0),
                 res["stats"].get("n_capped", 0)))
        print("  mean %s source / target (sampled): %.5g / %.5g"
              % (AVERAGING, res["mean_f_source"], res["mean_f_target"]))
        print("  written: %s%s" % (MODEM_OUT, MODEM_EXT))
        diag = dict(rho=res["rho"], coverage=res["coverage"])
    else:
        raise ValueError("DIRECTION must be 'modem2femtic' or 'femtic2modem'")

    if DIAGNOSTICS_NPZ:
        np.savez_compressed(DIAGNOSTICS_NPZ, **diag)
        print("  diagnostics: %s" % DIAGNOSTICS_NPZ)
    print("  elapsed: %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    run()
