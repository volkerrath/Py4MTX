#!/usr/bin/env python3
"""
femtic_modem_trans.py

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
@modified   2026-10-07

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

Observed data (optional, TRANSFORM_DATA = True)
-----------------------------------------------
The same script can also transform the observed data: FEMTIC observe.dat
(MT, VTF, PT blocks) <-> ModEM data file (Full_Impedance,
Full_Vertical_Components, Phase_Tensor; Off_Diagonal_Impedance read with
MISSING_ERR), in both directions. Units (Ohm / [V/m]/[T] / [mV/km]/[nT]),
FT sign, period/frequency, error merging and the site frame shift (same
placement as the model grid, COORD_MODE) are handled; see the readme.
FEMTIC conventions (Z in Ohm, exp(-i omega t), header "name x y z" with
x = east, tipper sign) are assumed, not verified against real runs.

Provenance
----------
Author    : Claude (Anthropic)
Generated : 2026-10-04
Notice    : This code is AI-generated. Review and test it before any
            production use. Verified with ast.parse() and a synthetic-
            data test (see femtic_modem_trans_readme.md); NOT yet run
            against real ModEM / FEMTIC files.

Changelog
---------
2026-10-04  Claude Sonnet 5.5 (Anthropic): created.
2026-10-05  Claude Sonnet 5.5 (Anthropic): added COORD_MODE="georef":
            FEMTIC origin from site.dat bounding box / observe.dat
            calibration sites, ModEM origin from the data file lat/lon
            and X/Y; built-in UTM conversion (latlon_to_utm).
2026-10-07  Claude Sonnet 5.5 (Anthropic): renamed modem_femtic_interp.py
            -> femtic_modem_trans.py (and the readme accordingly); the
            module name no longer implies model interpolation only, since
            data transformation is to be added. No functional change.
2026-10-07  Claude Sonnet 5.5 (Anthropic): optional observed-data
            transformation (TRANSFORM_DATA): read_modem_data,
            write_modem_data, read/write_femtic_observe, _run_data,
            utm_to_latlon; run() split into _run_model() / _run_data().
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
OBSERVE_DAT = "observe.dat"         # calibration: model-local site x/y [km];
                                    #   also the FEMTIC data INPUT of the
                                    #   data transformation (see below)
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

# --- what to run ---
TRANSFORM_MODEL = True              # model interpolation (as before)
TRANSFORM_DATA = False              # optional: transform observed data

# --- data transformation (used when TRANSFORM_DATA = True) ---
# Direction: None = follow DIRECTION (modem2femtic: ModEM data file ->
# FEMTIC observe.dat; femtic2modem: FEMTIC observe.dat -> ModEM data file);
# or set "modem2femtic" / "femtic2modem" explicitly.
DATA_DIRECTION = None
# Inputs: femtic2modem reads OBSERVE_DAT, modem2femtic reads
# MODEM_DATA + MODEM_DATA_EXT (both defined above).
OBSERVE_OUT = "observe_from_modem.dat"          # modem2femtic: output file
MODEM_DATA_OUT = "modem_data_from_femtic"       # femtic2modem: output base
DATA_TYPES = ("MT", "VTF", "PT")    # FEMTIC block types to transform
FEMTIC_SITE_XY = "en"               # observe.dat site header (x, y): "en" =
                                    #   (east, north) [m]; "ne" = swapped
DATA_Z_OFFSET = 0.0                 # z_femtic = Z_modem + DATA_Z_OFFSET [m]
                                    #   (both z-down); site z is carried over
# ModEM output conventions (femtic2modem)
MODEM_Z_UNITS_OUT = "[mV/km]/[nT]"  # "Ohm" | "[V/m]/[T]" | "[mV/km]/[nT]"
MODEM_FT_OUT = "exp(-i\\omega t)"    # or "exp(+i\\omega t)" (conjugates Z, T)
MODEM_ERR_COMBINE = "max"           # one ModEM error from FEMTIC re/im
                                    #   errors: "max" | "mean" | "min" |
                                    #   "re" | "im"
MODEM_DATA_COMMENT = "femtic_modem_trans.py"
# modem2femtic
OBSERVE_PREAMBLE = None             # optional comment line(s) before MT block
MISSING_ERR = None                  # component absent in a ModEM block (e.g.
                                    #   ZXX/ZYY of Off_Diagonal_Impedance, or
                                    #   a period lacking a component): None =
                                    #   drop that frequency/block; a number =
                                    #   write value 0 with this error


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


def utm_to_latlon(easting, northing, zone: int, northern: bool
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """UTM (easting, northing) [m] -> WGS84 (lat, lon) [deg]; inverse of
    :func:`latlon_to_utm` (Kruger n-series, Karney 2011; vectorised)."""
    e = np.asarray(easting, dtype=float)
    nn = np.asarray(northing, dtype=float)
    k0, e0, n0 = 0.9996, 500000.0, (0.0 if northern else 10000000.0)
    f = _WGS84_F
    n = f / (2.0 - f)
    big_a = _WGS84_A / (1.0 + n) * (1.0 + n**2 / 4.0 + n**4 / 64.0
                                    + n**6 / 256.0)
    beta = (
        n / 2 - 2 * n**2 / 3 + 37 * n**3 / 96 - n**4 / 360
        - 81 * n**5 / 512 + 96199 * n**6 / 604800,
        n**2 / 48 + n**3 / 15 - 437 * n**4 / 1440 + 46 * n**5 / 105
        - 1118711 * n**6 / 3870720,
        17 * n**3 / 480 - 37 * n**4 / 840 - 209 * n**5 / 4480
        + 5569 * n**6 / 90720,
        4397 * n**4 / 161280 - 11 * n**5 / 504 - 830251 * n**6 / 7257600,
        4583 * n**5 / 161280 - 108847 * n**6 / 3991680,
        20648693 * n**6 / 638668800)
    delta = (
        2 * n - 2 * n**2 / 3 - 2 * n**3 + 116 * n**4 / 45
        + 26 * n**5 / 45 - 2854 * n**6 / 675,
        7 * n**2 / 3 - 8 * n**3 / 5 - 227 * n**4 / 45
        + 2704 * n**5 / 315 + 2323 * n**6 / 945,
        56 * n**3 / 15 - 136 * n**4 / 35 - 1262 * n**5 / 105
        + 73814 * n**6 / 2835,
        4279 * n**4 / 630 - 332 * n**5 / 35 - 399572 * n**6 / 14175,
        4174 * n**5 / 315 - 144838 * n**6 / 6237,
        601676 * n**6 / 22275)
    xi_p = (nn - n0) / (k0 * big_a)
    eta_p = (e - e0) / (k0 * big_a)
    xi, eta = xi_p.copy(), eta_p.copy()
    for j, b in enumerate(beta, start=1):
        xi = xi - b * np.sin(2 * j * xi_p) * np.cosh(2 * j * eta_p)
        eta = eta - b * np.cos(2 * j * xi_p) * np.sinh(2 * j * eta_p)
    chi = np.arcsin(np.sin(xi) / np.cosh(eta))
    phi = chi.copy()
    for j, d in enumerate(delta, start=1):
        phi = phi + d * np.sin(2 * j * chi)
    lon = (6.0 * zone - 183.0) + np.degrees(np.arctan2(np.sinh(eta),
                                                        np.cos(xi)))
    return np.degrees(phi), lon


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


def _georef_full(fem, mod) -> dict:
    """Compute the placement of the ModEM data frame in the FEMTIC frame
    from the two UTM origins; prints a summary table.

    Returns a dict with off_n, off_e (ModEM data frame in the FEMTIC frame
    [m]), the UTM origins f_e, f_n (FEMTIC) and m_e, m_n (ModEM data
    frame), the UTM zone and hemisphere flag."""
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
    return dict(off_n=off_n, off_e=off_e, f_e=f_e, f_n=f_n, m_e=m_e,
                m_n=m_n, zone=zone, north=north)


def _georef_offsets(fem, mod) -> Tuple[float, float]:
    """(OFFSET_N, OFFSET_E) of the ModEM data frame in the FEMTIC frame."""
    g = _georef_full(fem, mod)
    return g["off_n"], g["off_e"]


# ===========================================================================
# Observed-data transformation (optional, TRANSFORM_DATA = True)
# ===========================================================================
#
# Neutral in-memory form (FEMTIC layout, SI units, e^{-i omega t}):
#   blocks[type] = [ dict(name, freq (nf,) Hz ascending,
#                         data (nf, ncol), err (nf, ncol)), ... ]
#   MT: ncol = 8 (Zxx_re, Zxx_im, ..., Zyy_im) [Ohm]; VTF: ncol = 4
#   (Tzx_re, Tzx_im, Tzy_re, Tzy_im); PT: ncol = 4 (Pxx, Pxy, Pyx, Pyy).

_MU0 = 4.0e-7 * np.pi
_MODEM_BLOCK_TYPES = {              # normalised ModEM DataType -> (type, comps)
    "fullimpedance": ("MT", ("ZXX", "ZXY", "ZYX", "ZYY")),
    "offdiagonalimpedance": ("MT", ("ZXX", "ZXY", "ZYX", "ZYY")),
    "fullverticalcomponents": ("VTF", ("TX", "TY")),
    "phasetensor": ("PT", ("PTXX", "PTXY", "PTYX", "PTYY")),
}
_MODEM_OUT_NAMES = {"MT": "Full_Impedance", "VTF": "Full_Vertical_Components",
                    "PT": "Phase_Tensor"}


def _norm_key(text: str) -> str:
    return str(text).strip().lower().replace("_", "").replace(" ", "")


def _z_unit_factor(label: str) -> float:
    """Multiplier converting a ModEM impedance unit label to SI Ohm."""
    key = _norm_key(label)
    table = {"ohm": 1.0, "[ohm]": 1.0, "[v/m]/[t]": _MU0,
             "[mv/km]/[nt]": _MU0 * 1.0e3}
    if key not in table:
        raise ValueError("unknown ModEM impedance unit %r (expected Ohm, "
                         "[V/m]/[T] or [mV/km]/[nT])" % label)
    return table[key]


def _ft_sign(text: str) -> int:
    """-1 for exp(-i omega t), +1 for exp(+i omega t)."""
    t = str(text)
    if "+" in t:
        return 1
    if "-" not in t:
        print("  WARNING: cannot parse the ModEM FT convention %r; "
              "assuming exp(-i\\omega t)" % t)
    return -1


def _pkey(period: float) -> float:
    """Period key tolerant to formatting noise (9 significant digits)."""
    return float("%.9g" % period)


def _combine_err(ere: np.ndarray, eim: np.ndarray, how: str) -> np.ndarray:
    h = str(how).lower()
    if h == "max":
        return np.maximum(ere, eim)
    if h == "min":
        return np.minimum(ere, eim)
    if h == "mean":
        return 0.5 * (ere + eim)
    if h == "re":
        return ere
    if h == "im":
        return eim
    raise ValueError("MODEM_ERR_COMBINE must be max|min|mean|re|im")


def read_modem_data(path: str) -> dict:
    """Read a ModEM data file (Full_Impedance, Off_Diagonal_Impedance,
    Full_Vertical_Components, Phase_Tensor blocks).

    Values are converted to the neutral form described above: impedances
    to SI Ohm (unit label of the block header), complex data conjugated if
    the block is exp(+i omega t), one ModEM error copied to the real and
    imaginary error column. Blocks of other DataTypes are skipped; a
    period missing a component is dropped, or filled with value 0 and
    error MISSING_ERR if that is set.

    Returns dict(blocks, pos, origin, skipped, n_dropped); pos[site] =
    (lat, lon, X north, Y east, Z down) in the ModEM data frame [m].
    """
    groups, hdr, rows = [], [], None
    with open(path, "r") as fh:
        for ln in fh:
            t = ln.strip()
            if not t or t.startswith("#"):
                continue
            if t.startswith(">"):
                if rows is not None:
                    groups.append((hdr, rows))
                    hdr, rows = [], None
                hdr.append(t[1:].strip())
                continue
            if rows is None:
                rows = []
            rows.append(t.split())
    if rows is not None:
        groups.append((hdr, rows))
    if not groups:
        raise ValueError("no data blocks found in %s" % path)

    acc: Dict[Tuple[str, str], Dict[float, dict]] = {}
    ncols: Dict[str, int] = {}
    pos: Dict[str, tuple] = {}
    skipped, origin = [], None
    for hdr, rows in groups:
        name = hdr[0] if hdr else "?"
        key = _norm_key(name)
        if key not in _MODEM_BLOCK_TYPES:
            skipped.append(name)
            continue
        ftype, comps = _MODEM_BLOCK_TYPES[key]
        sgn = _ft_sign(hdr[1]) if len(hdr) > 1 else -1
        fz = 1.0
        if ftype == "MT":
            fz = _z_unit_factor(hdr[2]) if len(hdr) > 2 else 1.0
        if len(hdr) > 3:
            try:
                if abs(float(hdr[3].split()[0])) > 1.0e-6:
                    print("  WARNING: block %s has orientation %s deg; "
                          "NOT applied" % (name, hdr[3]))
            except ValueError:
                pass
        if len(hdr) > 4 and origin is None:
            try:
                tk = hdr[4].split()
                origin = (float(tk[0]), float(tk[1]))
            except (ValueError, IndexError):
                pass
        ncols[ftype] = 8 if ftype == "MT" else 4
        for tk in rows:
            comp = tk[7].upper()
            if comp not in comps:
                continue
            site = tk[1]
            pos.setdefault(site, (float(tk[2]), float(tk[3]), float(tk[4]),
                                  float(tk[5]), float(tk[6])))
            if ftype == "PT":                    # real: Period..Comp Val Err
                val, err = float(tk[8]), float(tk[9])
            else:                                # Re Im Err
                val = complex(float(tk[8]), float(tk[9]))
                if sgn > 0:
                    val = val.conjugate()
                val *= fz
                err = float(tk[10]) * fz
            d = acc.setdefault((ftype, site), {}).setdefault(
                _pkey(float(tk[0])), {})
            d[comp] = (val, err)

    blocks: Dict[str, list] = {}
    n_dropped = 0
    for (ftype, site), per in acc.items():
        comps = [v[1] for v in _MODEM_BLOCK_TYPES.values() if v[0] == ftype][0]
        fr, dat, er = [], [], []
        for pk in sorted(per, reverse=True):          # period desc = f asc
            d = per[pk]
            if len(d) < len(comps) and MISSING_ERR is None:
                n_dropped += 1
                continue
            vrow, erow = [], []
            for c in comps:
                v, e = d.get(c, (0.0, MISSING_ERR))
                if ftype == "PT":
                    vrow.append(float(np.real(v)))
                    erow.append(float(e))
                else:
                    vrow += [float(np.real(v)), float(np.imag(v))]
                    erow += [float(e), float(e)]
            fr.append(1.0 / pk)
            dat.append(vrow)
            er.append(erow)
        if fr:
            blocks.setdefault(ftype, []).append(dict(
                name=site, freq=np.array(fr), data=np.array(dat),
                err=np.array(er)))
    return dict(blocks=blocks, pos=pos, origin=origin, skipped=skipped,
                n_dropped=n_dropped)


def write_modem_data(path: str, blocks: dict, sites: dict, *,
                     z_units: str, ft: str, err_combine: str,
                     comment: str, origin_latlon: Tuple[float, float]) -> dict:
    """Write a ModEM data file from the neutral form.

    sites[name] = (lat, lon, X north, Y east, Z down). Impedances are
    divided by the unit factor of z_units, conjugated (Z and tipper) for
    exp(+i omega t); FEMTIC real/imag errors are merged to one ModEM
    error (err_combine; PT: the single error). Rows are ordered by period
    (ascending) then site. Returns row/zero-error counts."""
    sgn = _ft_sign(ft)
    fz = _z_unit_factor(z_units)
    stats = dict(rows=0, zero_err=0)
    out = []
    for ftype in ("MT", "VTF", "PT"):
        if ftype not in blocks:
            continue
        comps = [v[1] for v in _MODEM_BLOCK_TYPES.values()
                 if v[0] == ftype][0]
        ncomp = len(comps)
        recs: Dict[float, list] = {}
        names = []
        for b in blocks[ftype]:
            if b["name"] not in sites:
                raise KeyError("no position for site %r" % b["name"])
            names.append(b["name"])
            lat, lon, X, Y, Z = sites[b["name"]]
            for i, f in enumerate(b["freq"]):
                T = 1.0 / float(f)
                dat, er = b["data"][i], b["err"][i]
                if ftype == "PT":
                    vals = [(dat[k], None, er[k]) for k in range(ncomp)]
                else:
                    zr, zi = dat[0::2], dat[1::2]
                    e = _combine_err(er[0::2], er[1::2], err_combine)
                    z = zr + 1j * zi
                    if sgn > 0:
                        z = np.conj(z)
                    if ftype == "MT":
                        z, e = z / fz, e / fz
                    vals = [(z[k].real, z[k].imag, e[k])
                            for k in range(ncomp)]
                recs.setdefault(_pkey(T), []).append(
                    (b["name"], T, lat, lon, X, Y, Z, vals))
        out.append("# %s" % comment)
        out.append("# Period(s) Code GG_Lat GG_Lon X(m) Y(m) Z(m) "
                   "Component Real Imag Error")
        out.append("> %s" % _MODEM_OUT_NAMES[ftype])
        out.append("> %s" % ft)
        out.append("> %s" % (z_units if ftype == "MT" else "[]"))
        out.append("> 0.00")
        out.append("> %.6f %.6f" % origin_latlon)
        out.append("> %d %d" % (len(recs), len(set(names))))
        for pk in sorted(recs):
            for (nm, T, lat, lon, X, Y, Z, vals) in recs[pk]:
                for k, (re_, im_, e_) in enumerate(vals):
                    if not e_ > 0.0:
                        stats["zero_err"] += 1
                    head = "%14.8E %-14s %12.6f %12.6f %12.1f %12.1f %12.1f" \
                           % (T, nm, lat, lon, X, Y, Z)
                    if ftype == "PT":
                        out.append("%s %-6s %14.6E %14.6E"
                                   % (head, comps[k], re_, e_))
                    else:
                        out.append("%s %-6s %14.6E %14.6E %14.6E"
                                   % (head, comps[k], re_, im_, e_))
                    stats["rows"] += 1
    Path(path).write_text("\n".join(out) + "\n", encoding="utf-8")
    return stats


def read_femtic_observe(fem, path: str, types: Sequence[str]) -> dict:
    """Read FEMTIC observe.dat into the neutral form via
    femtic.read_observe_dat. pos[name] = (north, east, z_down) [m] in the
    FEMTIC frame (site header x, y interpreted via FEMTIC_SITE_XY)."""
    parsed = fem.read_observe_dat(path, compute_mt_derived=False)
    xy = str(FEMTIC_SITE_XY).lower()
    if xy not in ("en", "ne"):
        raise ValueError("FEMTIC_SITE_XY must be 'en' or 'ne'")
    blocks: Dict[str, list] = {}
    pos: Dict[str, tuple] = {}
    for blk in parsed["blocks"]:
        t = str(blk["obs_type"])
        if t not in types:
            continue
        for s in blk["sites"]:
            tk = s["site_header_tokens"]
            x, y, z = float(tk[1]), float(tk[2]), float(tk[3])
            e, n = (x, y) if xy == "en" else (y, x)
            pos.setdefault(tk[0], (n, e, z))
            blocks.setdefault(t, []).append(dict(
                name=tk[0], freq=np.asarray(s["freq"], float),
                data=np.asarray(s["data"], float),
                err=np.asarray(s["error"], float)))
    return dict(blocks=blocks, pos=pos)


def write_femtic_observe(fem, path: str, blocks: dict, pos: dict) -> None:
    """Write the neutral form as FEMTIC observe.dat (femtic.write_observe_dat).
    pos[name] = (north, east, z_down) [m] in the FEMTIC frame."""
    xy = str(FEMTIC_SITE_XY).lower()
    pre = OBSERVE_PREAMBLE
    if pre is None:
        pre_lines = []
    elif isinstance(pre, str):
        pre_lines = pre.split("\n")
    else:
        pre_lines = list(pre)
    parsed = dict(preamble_lines=[p if p.endswith("\n") else p + "\n"
                                  for p in pre_lines],
                  blocks=[], end_line="END\n", tail_lines=[])
    for t in ("MT", "VTF", "PT"):
        if t not in blocks:
            continue
        sites = []
        for b in blocks[t]:
            n, e, z = pos[b["name"]]
            x, y = (e, n) if xy == "en" else (n, e)
            nf = len(b["freq"])
            sites.append(dict(
                site_header_line="%s %.3f %.3f %.3f\n" % (b["name"], x, y, z),
                nfreq=nf, nfreq_line="%d\n" % nf, freq=b["freq"],
                data=b["data"], error=b["err"], extras=[]))
        parsed["blocks"].append(dict(
            obs_type=t, header_line="%s    %d\n" % (t, len(sites)),
            sites=sites))
    fem.write_observe_dat(parsed, path)


def _data_frame_shift(fem, mod):
    """(shift_n, shift_e, georef): FEMTIC position = ModEM data-frame
    position + shift. Mirrors the grid placement of the model step so
    sites and model stay mutually consistent. georef is the dict of
    _georef_full (COORD_MODE = "georef") or None."""
    mode = str(COORD_MODE).lower()
    if mode == "georef":
        g = _georef_full(fem, mod)
        return g["off_n"] + OFFSET_N, g["off_e"] + OFFSET_E, g
    dx, dy, dz, _, ref, _ = mod.read_mod(file=MODEM_MODEL, modext=MODEM_EXT,
                                         trans="LINEAR", out=False)
    grid = ModemGrid(dx, dy, dz, ref, coord_mode="reference")
    if mode.startswith("cen"):
        c = [0.5 * (grid.e[a][0] + grid.e[a][-1]) for a in (0, 1)]
    elif mode == "reference":
        c = [0.0, 0.0]
    else:
        raise ValueError("COORD_MODE must be centre|reference|georef")
    return OFFSET_N - c[0], OFFSET_E - c[1], None


def _run_data() -> None:
    """Optional observed-data transformation (module-level config)."""
    import femtic as fem
    import modem as mod

    direction = DATA_DIRECTION or DIRECTION
    types = tuple(str(t).upper() for t in DATA_TYPES)
    print("  Observed-data transformation: %s  types=%s"
          % (direction, ",".join(types)))
    shift_n, shift_e, geo = _data_frame_shift(fem, mod)
    print("  data-frame shift (FEMTIC = ModEM + shift): N %.1f  E %.1f  "
          "z %.1f m" % (shift_n, shift_e, DATA_Z_OFFSET))

    if direction == "modem2femtic":
        src = "%s%s" % (MODEM_DATA, MODEM_DATA_EXT)
        ds = read_modem_data(src)
        blocks = {t: v for t, v in ds["blocks"].items() if t in types}
        if not blocks:
            raise ValueError("no usable %s data in %s (skipped blocks: %s)"
                             % ("/".join(types), src, ds["skipped"]))
        pos = {nm: (p[2] + shift_n, p[3] + shift_e, p[4] + DATA_Z_OFFSET)
               for nm, p in ds["pos"].items()}
        write_femtic_observe(fem, OBSERVE_OUT, blocks, pos)
        print("  read %s: %s" % (src, ", ".join(
            "%s %d sites" % (t, len(v)) for t, v in blocks.items())))
        if ds["skipped"]:
            print("  skipped DataTypes: %s" % ", ".join(ds["skipped"]))
        if ds["n_dropped"]:
            print("  periods dropped (component missing, MISSING_ERR unset)"
                  ": %d" % ds["n_dropped"])
        print("  written: %s" % OBSERVE_OUT)
    elif direction == "femtic2modem":
        ds = read_femtic_observe(fem, OBSERVE_DAT, types)
        if not ds["blocks"]:
            raise ValueError("no %s blocks in %s" % ("/".join(types),
                                                     OBSERVE_DAT))
        rows = None
        if geo is None and os.path.isfile(SITE_DAT):
            rows = {r["name"]: r for r in fem.read_site_dat(SITE_DAT)}
        sites, nolatlon = {}, 0
        for nm, (n, e, z) in ds["pos"].items():
            X, Y, Z = n - shift_n, e - shift_e, z - DATA_Z_OFFSET
            if geo is not None:
                lat, lon = utm_to_latlon(geo["m_e"] + Y, geo["m_n"] + X,
                                         geo["zone"], geo["north"])
                lat, lon = float(lat), float(lon)
            elif rows is not None and nm in rows:
                lat, lon = rows[nm]["lat"], rows[nm]["lon"]
            else:
                lat = lon = 0.0
                nolatlon += 1
            sites[nm] = (lat, lon, X, Y, Z)
        if nolatlon:
            print("  WARNING: %d site(s) without lat/lon (set COORD_MODE="
                  "'georef' or provide SITE_DAT); written as 0.0" % nolatlon)
        if geo is not None:
            o = utm_to_latlon(geo["m_e"], geo["m_n"], geo["zone"],
                              geo["north"])
            origin = (float(o[0]), float(o[1]))
        elif MODEM_ORIGIN_LATLON is not None:
            origin = (float(MODEM_ORIGIN_LATLON[0]),
                      float(MODEM_ORIGIN_LATLON[1]))
        else:
            origin = (0.0, 0.0)
        out = MODEM_DATA_OUT + MODEM_DATA_EXT
        st = write_modem_data(
            out, ds["blocks"], sites, z_units=MODEM_Z_UNITS_OUT,
            ft=MODEM_FT_OUT, err_combine=MODEM_ERR_COMBINE,
            comment=MODEM_DATA_COMMENT, origin_latlon=origin)
        print("  read %s: %s" % (OBSERVE_DAT, ", ".join(
            "%s %d sites" % (t, len(v)) for t, v in ds["blocks"].items())))
        print("  ModEM rows written: %d (non-positive errors: %d)"
              % (st["rows"], st["zero_err"]))
        print("  written: %s" % out)
    else:
        raise ValueError("DATA_DIRECTION/DIRECTION must be 'modem2femtic' "
                         "or 'femtic2modem'")


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


def _run_model() -> None:
    """Model interpolation (uses the module-level config)."""
    import femtic as fem
    import modem as mod

    t0 = time.time()
    print("femtic_modem_trans: DIRECTION=%s  AVERAGING=%s"
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
                             "(volume-weighted, femtic_modem_trans.py)",
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


def run() -> None:
    """Run the configured steps: model interpolation (TRANSFORM_MODEL) and
    optionally observed-data transformation (TRANSFORM_DATA)."""
    if not (TRANSFORM_MODEL or TRANSFORM_DATA):
        print("nothing to do: TRANSFORM_MODEL and TRANSFORM_DATA are False")
        return
    if TRANSFORM_MODEL:
        _run_model()
    if TRANSFORM_DATA:
        t0 = time.time()
        _run_data()
        print("  elapsed (data): %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    run()
