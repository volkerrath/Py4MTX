#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
remapping.py
=====================
Conservative (cell-volume-weighted) remapping of the pipeline's three
native MT/seismic sources onto ONE common regular UTM-km grid, as a
sibling to interpolate.py's RBF/kriging/IDW/nearest scheme -- built
specifically to remove a bias interpolate.py cannot avoid: every one of
its four methods treats each native "point" (a ModEM cell center, a
FEMTIC tetrahedron centroid, a seismic-grid node) as an unweighted
sample. ModEM meshes are far finer near the surface/stations than at
depth, FEMTIC tetrahedra vary in size by orders of magnitude across a
single mesh, and none of that size variation is visible to a point-
cloud interpolator -- a cluster of many small cells in one corner of the
model outweighs a single huge cell elsewhere purely because it
contributes more points, not because it represents more physical
volume. This script instead performs a genuine conservative regrid:
every native CELL's full volume (not just its center) is intersected
against the target grid's cells, so a target cell's value is a true
volume-weighted average of whatever native volume actually falls inside
it, regardless of how finely or coarsely that native volume happened to
be discretised.

Pipeline
--------
precompute.py  →  remapping.py  →  {SITE_PREFIX}_remap.npz  →  cluster.py

This sits alongside (NOT instead of) interpolate.py: both read
precompute.py's outputs and both feed cluster.py, via different
container formats (interpolate.py → NetCDF, this script → NPZ) that
cluster.py's own INPUT_KIND setting picks between (see that script's
own header). Point interpolate.py's inputs and this script's at the
same precompute.py outputs and compare the two clusterings directly if
you want to see how much the cell-size bias actually matters for a
given site.

Backends: REMAP_BACKEND = "medcoupling" | "esmpy"
------------------------------------------------------
Every native source is first turned into an explicit volumetric cell
mesh in pure NumPy (see "Native sources" below) -- nodes + connectivity
+ one value + one volume per cell -- independently of which remapping
library actually performs the conservative intersection. That common
"cells" representation is then handed to one of two backend adapters:

  "medcoupling" (default) -- MEDCoupling's MEDCouplingRemapper with the
    "P0P0" (cell-to-cell, conservative) intersection type. This is the
    library's core, long-standing use case (arbitrary-shaped source and
    target cells, exact geometric intersection, any mix of hexahedral/
    tetrahedral cells) and the only backend this script uses for
    "femtic_cells" sources -- see below. Requires `pip install
    medcoupling` (part of the SALOME platform's standalone MEDCoupling
    wheels) or a system install with its Python bindings on
    `sys.path`.
  "esmpy" -- ESMF's Python interface, implemented here ONLY for
    "modem_cells" sources (a genuinely rectilinear grid remapped onto
    the target grid, itself always rectilinear too) via `esmpy.Grid` +
    conservative `esmpy.Regrid`, ESMF's own long-established
    structured-to-structured conservative-remap machinery -- the same
    code path climate/ocean/land models use for SCRIP-style regridding,
    and squarely its best-supported, most mature use case. "seis_grid"
    (curvilinear -- would need ESMF's curvilinear-Grid corner handling)
    and "femtic_cells" (tetrahedral -- would need ESMF's considerably
    less mature/uniformly-available 3-D unstructured `Mesh` volumetric-
    element support) are NOT attempted under this backend:
    REMAP_BACKEND="esmpy" raises NotImplementedError for both rather
    than silently guessing at APIs whose availability/correctness
    varies by ESMF build/version -- set REMAP_BACKEND="medcoupling"
    (globally, or via that one VARIABLE_SOURCES entry's own "backend"
    override) for those instead.

Both backends are imported lazily (inside the function that needs
them), exactly like interpolate.py's `pykrige` import -- neither is a
hard dependency of this module for the other backend's sources.

Native sources
--------------
Reuses precompute.py's own outputs, same directory (NC_DIR) as
interpolate.py, but reads each one differently -- as a full CELL
geometry with a real volume, not a point:

  "modem_cells"  -- precompute.py's modem_submesh_points.nc (Part A),
    read as the FULL structured product grid it actually is (every
    cell present as a row, including excluded/air cells kept as
    valid=0 NaN rows -- see the "Excluded elements" convention in
    precompute.py's own docs). Genuine axis-aligned cell edges are
    reconstructed from the sorted, unique easting/northing/depth values
    (`cell_edges_from_centers()`) -- exact for a rectilinear mesh like
    ModEM's, not an approximation, EXCEPT at the two outermost cells
    along each axis, whose true outer edge isn't recoverable from
    centers alone: those are given the same width as their single
    interior neighbor (MODEM_EDGE_EXTRAPOLATION="mirror", the only
    option currently implemented) -- flagged here as a real, if usually
    small, source of edge-volume error rather than silently assumed
    exact.
  "femtic_cells" -- precompute.py Part C's femtic_mesh_utm.nc (full
    tetrahedral node/connectivity, UTM METRES) + femtic_submesh_points.nc
    (one resistivity value per element, same row order) together give
    the genuine tetrahedral source mesh needed for a real geometric
    conservative remap -- unlike interpolate.py's "femtic_points" kind,
    which only ever sees each element's centroid. The row-order match
    between the two files is NOT assumed silently: a random sample of
    valid elements' mesh-derived centroids is checked against
    femtic_submesh_points.nc's own stored centroids
    (FEMTIC_CENTROID_CHECK_TOL_KM) before proceeding.
  "seis_grid"    -- one precompute.py Part B output ({SITE_PREFIX}_vp.nc
    etc.), a gridded (depth, row, col) cube with 2-D utm_easting/
    utm_northing aux coords -- i.e. a CURVILINEAR grid, not necessarily
    axis-aligned with UTM easting/northing. Cell corners are
    reconstructed from cell centers (`_corners_from_centers_2d()`:
    each interior corner = mean of its four surrounding centers,
    boundary corners linearly extrapolated one half-cell past the
    outermost centers -- the standard "corners from centers"
    construction used whenever a source only ships cell centers, e.g.
    for SCRIP/ESMF curvilinear grids built from model output that only
    records center lat/lon). Vertical cell thickness comes from the
    same `cell_edges_from_centers()` used for ModEM's depth axis. This
    source kind is usually the LEAST affected by the cell-size bias
    interpolate.py has (tomography grids are typically fairly regular
    already) but is remapped the same conservative way for consistency
    with "modem_cells"/"femtic_cells" in the same run, and because nothing
    about the corner-reconstruction/hexahedral-cell machinery below
    cares whether the grid happens to already be regular.

Every source kind is reduced to ONE common in-memory shape before
either backend ever sees it -- `build_source_cells()` returns a dict
with `nodes` (n, 3), `conn` (ncell, 8) for "HEX8" cell types or (ncell,
4) for "TET4", `value`/`valid`/`volume` (ncell,) each -- via
`hex_mesh_from_corner_grid()` (shared by "modem_cells"' rectilinear
corners and "seis_grid"'s curvilinear ones -- the connectivity/volume
logic is identical either way, only the corner coordinates differ) or
directly from FEMTIC's own tetrahedral connectivity. Cell volumes for
both HEX8 and TET4 cells are computed geometrically
(`hexahedron_volume()`/`tetrahedron_volume()`, both simple centroid- or
vertex-anchored tetrahedral decompositions -- exact for TET4, and exact
for HEX8 as long as each face is planar, which is always true for
"modem_cells"' rectilinear boxes and true to within the mild curvature
"seis_grid" corner-reconstruction introduces) rather than trusted from
any one source file, so every source's total remapped volume can be
cross-checked against its own reported sum as a sanity check
(printed at the end of each variable's remap).

Target grid
-----------
Always a single freshly-built, genuinely regular UTM-km grid (GRID_*_KM
below -- same shape/meaning as interpolate.py's own joint-grid settings,
auto-bounded to the tightest common overlap of every active source's
own extent unless given explicit bounds), turned into its own HEX8 cell
mesh via the same `hex_mesh_from_corner_grid()` used for "modem_cells"/
"seis_grid" sources. Unlike interpolate.py, this script does NOT offer
a "seismic" (reuse one source's own native grid) target-grid mode --
every native source here already has to be turned into an explicit
cell mesh regardless of its own resolution, so there's no shortcut
available the way there is for point-cloud interpolation, and
supporting an arbitrary curvilinear TARGET (rather than just letting a
curvilinear grid be a SOURCE, which is already handled) would roughly
double the backend-adapter code for a case none of this project's sites
currently need.

Coverage-fraction masking (replaces MASK_TO_CONVEX_HULL)
----------------------------------------------------------
interpolate.py masks unconstrained extrapolation past a variable's own
data footprint via a separate Delaunay convex-hull check
(MASK_TO_CONVEX_HULL). A conservative remap doesn't need a second,
independent geometric check for this -- the remap itself already knows
exactly what fraction of each target cell's volume was actually covered
by valid source cells (computed here via the "ones-field trick": a
constant-1 source field, of nature ExtensiveConservation/its esmpy
equivalent, remapped the same way, divided by the target cell's own
volume). Target cells with a covered fraction below
MIN_COVERAGE_FRACTION are set to NaN -- a strictly more principled mask
than a convex hull, since it directly reflects how much real volume
actually informs that cell's value rather than just whether the cell's
CENTER happens to fall inside the source's outer hull.
APPLY_ROI_MASK/ROI_VERTICES_KM (optional, additional) work identically
to interpolate.py's own -- a single shared rectangular region applied
to every variable's remapped grid after the fact.

Output
------
{SITE_PREFIX}_remap.npz (NC_DIR, override with OUTPUT_FILE) -- an NPZ
archive rather than NetCDF, since this script's per-variable coverage
diagnostics and metadata are simpler to keep as a flat set of arrays +
one JSON-encoded metadata string than to force into NetCDF's
attribute-string conventions. Contains: `depth`/`northing`/`easting`
(the target grid's 1-D cell-center axes, km), one `{key}` array per
REMAP_VARS entry (float32, dims (depth, northing, easting), NaN where
MIN_COVERAGE_FRACTION/ROI masked it out), one `{key}_coverage` array
(float32, covered-volume fraction in [0, 1]) per entry, and a
`meta_json` 0-d string array (`json.loads()` it back) recording
`dim_names`, `remap_vars`, `units`/`label` per variable, `remap_backend`,
`min_coverage_fraction`, and the grid/ROI settings used -- everything
cluster.py needs to read it back and treat it exactly like a "joint"-
mode interpolate.py output (see that script's own header for how
INPUT_KIND="remap" is wired in).

Dependencies
------------
    numpy, xarray
plus EITHER `medcoupling` OR `esmpy` (REMAP_BACKEND), each imported
lazily and each optional with respect to the other.

Authors: Svetlana Byrdina (SMB) & Volker Rath (DIAS)
AI-assisted development: Claude (Anthropic), 2026-09-06.
License: GNU General Public License v3 (GPL-3.0-or-later).
AI-generated code — review before use in production.
"""

import json

import numpy as np
import xarray as xr
from matplotlib.path import Path as MplPath

import tomomt

# =====================================================================
# USER SETTINGS
# =====================================================================

# --- Site selector ---
# Must match SITE_PREFIX in precompute.py.
SITE_PREFIX = "saba"
# SITE_PREFIX = "tacna"  # TACNA

# --- Input/output directory ---
NC_DIR = "../precompute/saba/"   # must match precompute.py's OUTPUT_DIR /
                                  # interpolate.py's NC_DIR; {SITE_PREFIX}_remap.npz
                                  # is written here too.

# --- Variable registry ---
# Same idea as interpolate.py's VARIABLE_SOURCES, but every entry is
# read as a CELL (with real geometry/volume), not a point -- see the
# "Native sources" section in the module docstring for what each kind
# expects to find in NC_DIR. `backend` optionally overrides
# REMAP_BACKEND for that one entry (e.g. force "medcoupling" for a
# "femtic_cells" source regardless of the global default).
VARIABLE_SOURCES = {
    "rho":  dict(kind="modem_cells", file="modem_submesh_points.nc",
                 value_var="resistivity", label="log10 resistivity",
                 units="log10(Ohm.m)"),
    "cond": dict(kind="modem_cells", file="modem_submesh_points.nc",
                 value_var="resistivity", label="conductivity", units=None,
                 derive="conductivity_from_rho"),
    "sens": dict(kind="modem_cells", file="modem_submesh_points.nc",
                 value_var="sensitivity", label="sensitivity",
                 units="(as stored)"),
    # FEMTIC's tetrahedral mesh -- see precompute.py Part C. Node/
    # connectivity variable names below MUST match whatever
    # precompute.py's Part C actually wrote into femtic_mesh_utm.nc;
    # adjust FEMTIC_MESH_NODE_VARS/FEMTIC_MESH_CONN_VAR below if they
    # differ (not shown to this script -- kept as an explicit, flagged
    # setting rather than guessed).
    "rho_femtic": dict(kind="femtic_cells", points_file="femtic_submesh_points.nc",
                        mesh_file="femtic_mesh_utm.nc", value_var="resistivity",
                        label="log10 resistivity (FEMTIC)", units="log10(Ohm.m)",
                        backend="medcoupling"),
    "vp":   dict(kind="seis_grid", file=f"{SITE_PREFIX}_vp.nc",   var="data", label="Vp",      units="km/s"),
    "vs":   dict(kind="seis_grid", file=f"{SITE_PREFIX}_vs.nc",   var="data", label="Vs",      units="km/s"),
    "vps":  dict(kind="seis_grid", file=f"{SITE_PREFIX}_vps.nc",  var="data", label="Vp/Vs",   units=""),
    "dens": dict(kind="seis_grid", file=f"{SITE_PREFIX}_dens.nc", var="data", label="Density", units="(as stored)"),
}

# Variable names expected inside femtic_mesh_utm.nc (precompute.py Part
# C's own output) -- node coordinates (UTM metres) and 0-based
# tetrahedral connectivity. See VARIABLE_SOURCES["rho_femtic"] comment.
FEMTIC_MESH_NODE_VARS = ("node_x", "node_y", "node_z")
FEMTIC_MESH_CONN_VAR = "connectivity"

# Maximum allowed mismatch (km) between femtic_mesh_utm.nc's own
# element centroids (computed from its nodes/connectivity) and
# femtic_submesh_points.nc's stored centroids, checked on a random
# sample of valid elements -- catches a silent row-order mismatch
# between the two files rather than assuming they agree.
FEMTIC_CENTROID_CHECK_TOL_KM = 0.05
FEMTIC_CENTROID_CHECK_N = 200
FEMTIC_CENTROID_CHECK_SEED = 0

# --- Which variables to prepare (remap onto the target grid) ---
REMAP_VARS = ["rho", "vps", "dens"]

# --- Use conductivity instead of resistivity, optionally ---
# Same swap as interpolate.py's USE_CONDUCTIVITY; no effect if "rho"
# isn't in REMAP_VARS.
USE_CONDUCTIVITY = False

# --- Remapping backend ---
# "medcoupling" (default, recommended) | "esmpy". See module docstring
# for what each one actually does and its "femtic_cells" limitation.
# Per-source overrides live in VARIABLE_SOURCES[key]["backend"].
REMAP_BACKEND = "medcoupling"

# --- esmpy conservative-regrid order (REMAP_BACKEND == "esmpy" only) ---
# "conserve" (1st-order, always monotone) | "conserve2nd" (2nd-order,
# can slightly overshoot at sharp gradients but is more accurate on
# smooth fields).
ESMPY_CONSERVE_METHOD = "conserve"

# --- Target grid (always a freshly-built regular UTM-km grid) ---
# min/max = None auto-computes from the intersection (tightest common
# overlap) of every active variable's own cell-mesh extent, same
# philosophy as interpolate.py's GRID_*_KM. step here is the TARGET
# CELL SIZE (not a sample spacing) -- this is the grid whose per-cell
# volumes the conservative remap actually integrates against.
GRID_EASTING_KM  = dict(min=None, max=None, step=2.0)
GRID_NORTHING_KM = dict(min=None, max=None, step=2.0)
GRID_DEPTH_KM    = dict(min=-4, max=36, step=1.0)

# --- ModEM edge-cell extrapolation ---
# Only "mirror" is implemented (outermost cell along each axis gets the
# same width as its single interior neighbor) -- see module docstring.
MODEM_EDGE_EXTRAPOLATION = "mirror"

# --- Coverage-fraction masking (replaces interpolate.py's
# MASK_TO_CONVEX_HULL -- see module docstring) ---
MIN_COVERAGE_FRACTION = 0.5   # target cells covered less than this
                              # fraction by valid source volume become NaN

# --- Region of interest (optional, additional; same convention as
# interpolate.py's APPLY_ROI_MASK/ROI_VERTICES_KM) ---
APPLY_ROI_MASK = False
ROI_VERTICES_KM = [
    (330.0, 7990.0),
    (430.0, 7990.0),
    (430.0, 8080.0),
    (330.0, 8080.0),
]
ROI_DEPTH_MIN_KM = None
ROI_DEPTH_MAX_KM = None

# --- Output ---
# None = auto ("{SITE_PREFIX}_remap.npz" in NC_DIR).
OUTPUT_FILE = None

# =====================================================================
# END USER SETTINGS
# =====================================================================


def ncpath(name):
    """Join a bare filename onto NC_DIR."""
    return tomomt.resolve_path(NC_DIR, name)


# ------------------------------------------------------------------
# Resistivity <-> conductivity
# ------------------------------------------------------------------
# Intentionally duplicated from interpolate.py rather than moved into
# tomomt.py: a small, self-contained, pure function, and this script
# deliberately avoids introducing a new cross-script dependency into
# interpolate.py itself (which this bundle doesn't otherwise touch) --
# see README_tomomt.md's own stated policy of leaving small per-script
# glue alone rather than forcing everything through the shared module.
def resistivity_to_conductivity(values, units):
    """Invert a resistivity field to conductivity -- see
    interpolate.py's identical function for the full docstring."""
    u = (units or "").lower()
    if "log10" in u:
        return -values, "log10(S/m)"
    if u.startswith("ln("):
        return -values, "ln(S/m)"
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = np.where(values != 0, 1.0 / values, np.nan)
    return cond.astype(np.float64), "S/m"


# ------------------------------------------------------------------
# Rectilinear / curvilinear cell-edge and cell-corner reconstruction
# ------------------------------------------------------------------
def cell_edges_from_centers(centers):
    """
    Reconstruct 1-D cell edges (length n+1) from n sorted, unique cell-
    CENTER coordinates, by taking midpoints between consecutive centers
    and mirroring the outermost gap's half-width to close off each end
    (MODEM_EDGE_EXTRAPOLATION="mirror" -- see module docstring). Exact
    for every interior edge on a genuinely rectilinear axis; the two
    outer edges are the one part of this that's an assumption, not a
    fact recoverable from centers alone.

    Returns
    -------
    edges  : ndarray, shape (n+1,)
    widths : ndarray, shape (n,) -- np.diff(edges)
    """
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError(
            "cell_edges_from_centers needs >= 2 unique cell-center "
            f"coordinates along this axis to infer cell widths; got "
            f"{centers.size}."
        )
    if np.any(np.diff(centers) <= 0):
        raise ValueError(
            "cell_edges_from_centers requires strictly increasing, "
            "unique centers (sort/np.unique them first)."
        )
    mids = 0.5 * (centers[:-1] + centers[1:])
    first = 2.0 * centers[0] - mids[0]
    last = 2.0 * centers[-1] - mids[-1]
    edges = np.concatenate([[first], mids, [last]])
    widths = np.diff(edges)
    return edges, widths


def _corners_from_centers_2d(c2d):
    """
    One component (easting OR northing) of curvilinear grid CORNER
    coordinates, shape (nrow+1, ncol+1), from cell-CENTER values c2d
    (nrow, ncol): every interior corner = mean of its four surrounding
    cell centers; boundary corners are linearly extrapolated one
    half-cell past the outermost centers (same "mirror" logic as
    cell_edges_from_centers(), in 2-D) -- the standard construction used
    whenever a curvilinear source only ships cell centers.
    """
    c2d = np.asarray(c2d, dtype=np.float64)
    if c2d.ndim != 2 or c2d.shape[0] < 2 or c2d.shape[1] < 2:
        raise ValueError(
            f"_corners_from_centers_2d needs a >= 2x2 grid of centers; "
            f"got shape {c2d.shape}."
        )
    nrow, ncol = c2d.shape
    padded = np.empty((nrow + 2, ncol + 2), dtype=np.float64)
    padded[1:-1, 1:-1] = c2d
    padded[0, 1:-1] = 2.0 * c2d[0, :] - c2d[1, :]
    padded[-1, 1:-1] = 2.0 * c2d[-1, :] - c2d[-2, :]
    padded[1:-1, 0] = 2.0 * c2d[:, 0] - c2d[:, 1]
    padded[1:-1, -1] = 2.0 * c2d[:, -1] - c2d[:, -2]
    padded[0, 0] = 2.0 * padded[0, 1] - padded[0, 2]
    padded[0, -1] = 2.0 * padded[0, -2] - padded[0, -3]
    padded[-1, 0] = 2.0 * padded[-1, 1] - padded[-1, 2]
    padded[-1, -1] = 2.0 * padded[-1, -2] - padded[-1, -3]
    corners = 0.25 * (
        padded[:-1, :-1] + padded[:-1, 1:] + padded[1:, :-1] + padded[1:, 1:]
    )
    return corners


# ------------------------------------------------------------------
# Explicit volumetric cell meshes (the common shape every backend sees)
# ------------------------------------------------------------------
def hexahedron_volume(cell_nodes):
    """
    Volume of each hexahedron, cell_nodes shape (ncell, 8, 3) in the
    VTK/ESMF HEX8 node order (0-3 = bottom face, 4-7 = top face, both
    CCW, node i+4 above node i). Computed by splitting the hex into 6
    pyramids (apex = the 8-node centroid, base = each quad face) and
    each pyramid into 2 tetrahedra sharing the face diagonal (a, c) --
    correct for any hexahedron that is star-shaped from its own
    centroid, which every cell this script builds is (exact for a
    planar-faced box; still correct to within the mild curvature
    `_corners_from_centers_2d()` can introduce for "seis_grid" cells).
    """
    n = cell_nodes
    centroid = n.mean(axis=1)
    faces = [
        (0, 1, 2, 3), (4, 5, 6, 7),
        (0, 1, 5, 4), (1, 2, 6, 5),
        (2, 3, 7, 6), (3, 0, 4, 7),
    ]

    def _tet_vol(p, a, b, c):
        return np.abs(np.einsum("ij,ij->i", a - p, np.cross(b - p, c - p))) / 6.0

    vol = np.zeros(n.shape[0], dtype=np.float64)
    for (a, b, c, d) in faces:
        va, vb, vc, vd = n[:, a], n[:, b], n[:, c], n[:, d]
        vol += _tet_vol(centroid, va, vb, vc)
        vol += _tet_vol(centroid, va, vc, vd)
    return vol


def tetrahedron_volume(cell_nodes):
    """Volume of each tetrahedron, cell_nodes shape (ncell, 4, 3):
    |det([v1-v0, v2-v0, v3-v0])| / 6."""
    v0, v1, v2, v3 = (cell_nodes[:, 0], cell_nodes[:, 1],
                      cell_nodes[:, 2], cell_nodes[:, 3])
    return np.abs(np.einsum(
        "ij,ij->i", v1 - v0, np.cross(v2 - v0, v3 - v0)
    )) / 6.0


def hex_mesh_from_corner_grid(Xc, Yc, Zc):
    """
    Build an explicit HEX8 volumetric mesh from a structured corner
    lattice Xc, Yc, Zc (each shape (nk+1, nj+1, ni+1), axes ordered
    depth-like/k, northing-or-row-like/j, easting-or-col-like/i) --
    works identically for a genuinely rectilinear grid ("modem_cells" /
    the target grid, corners built via np.meshgrid of 1-D edge arrays)
    and a curvilinear one ("seis_grid", corners built from
    _corners_from_centers_2d() broadcast across depth edges): the
    node/connectivity/volume logic doesn't know or care which.

    Returns
    -------
    nodes       : ndarray, shape (nnode, 3)
    conn        : ndarray, shape (ncell, 8), 0-based, VTK/ESMF HEX8 order
    cell_volume : ndarray, shape (ncell,)

    Cell ordering is k-major, then j, then i (matches
    `np.meshgrid(..., indexing="ij")` raveled in C order) -- callers
    must flatten their own per-cell value/valid arrays the same way.
    """
    nk1, nj1, ni1 = Xc.shape
    if Yc.shape != Xc.shape or Zc.shape != Xc.shape:
        raise ValueError("Xc, Yc, Zc must all have the same shape.")
    nk, nj, ni = nk1 - 1, nj1 - 1, ni1 - 1
    if nk < 1 or nj < 1 or ni < 1:
        raise ValueError(f"Corner grid too small: shape {Xc.shape}.")
    nodes = np.column_stack([Xc.ravel(), Yc.ravel(), Zc.ravel()])

    def _nid(k, j, i):
        return (k * nj1 + j) * ni1 + i

    kk, jj, ii = np.meshgrid(
        np.arange(nk), np.arange(nj), np.arange(ni), indexing="ij",
    )
    kk = kk.ravel(); jj = jj.ravel(); ii = ii.ravel()

    conn = np.column_stack([
        _nid(kk,     jj,     ii),
        _nid(kk,     jj,     ii + 1),
        _nid(kk,     jj + 1, ii + 1),
        _nid(kk,     jj + 1, ii),
        _nid(kk + 1, jj,     ii),
        _nid(kk + 1, jj,     ii + 1),
        _nid(kk + 1, jj + 1, ii + 1),
        _nid(kk + 1, jj + 1, ii),
    ]).astype(np.int64)

    cell_volume = hexahedron_volume(nodes[conn])
    return nodes, conn, cell_volume


# ------------------------------------------------------------------
# Native source loaders -> {"cell_type", "nodes", "conn", "value",
#                           "valid", "volume", "units"}
# ------------------------------------------------------------------
def load_modem_cells(key, src):
    """
    Read precompute.py's modem_submesh_points.nc (Part A) as the FULL
    structured product grid it actually is, and build its HEX8 cell
    mesh. See module docstring's "modem_cells" entry for the edge-
    reconstruction caveat.
    """
    path = ncpath(src["file"])
    ds = xr.open_dataset(path)
    value_var = src["value_var"]
    if value_var not in ds.data_vars:
        ds.close()
        raise KeyError(
            f"{path!r} has no variable {value_var!r} -- if this is "
            f"'sensitivity', re-run precompute.py with USE_SENSITIVITY "
            f"= True first."
        )
    e = ds["easting"].values.astype(np.float64)
    n = ds["northing"].values.astype(np.float64)
    d = ds["depth"].values.astype(np.float64)
    v = ds[value_var].values.astype(np.float64)
    valid = ds["valid"].values.astype(bool) & np.isfinite(v)
    units = ds[value_var].attrs.get("units", "")
    ds.close()

    e_axis = np.unique(e)
    n_axis = np.unique(n)
    d_axis = np.unique(d)
    nx, ny, nz = len(e_axis), len(n_axis), len(d_axis)
    if nx * ny * nz != len(e):
        raise ValueError(
            f"{src['file']!r} does not look like a full structured "
            f"product grid (unique axis sizes {nx} x {ny} x {nz} = "
            f"{nx * ny * nz} != {len(e)} rows). remapping.py needs "
            f"precompute.py's modem_submesh_points.nc written with "
            f"excluded cells KEPT as NaN/valid=0 rows, not dropped -- "
            f"interpolate.py's load_point_table() works fine either way "
            f"since it discards invalid rows anyway; this script cannot "
            f"reconstruct cell geometry without the full product grid."
        )
    if MODEM_EDGE_EXTRAPOLATION != "mirror":
        raise ValueError(
            f"MODEM_EDGE_EXTRAPOLATION={MODEM_EDGE_EXTRAPOLATION!r} -- "
            f"only 'mirror' is implemented."
        )

    ie = np.searchsorted(e_axis, e)
    in_ = np.searchsorted(n_axis, n)
    id_ = np.searchsorted(d_axis, d)
    value_cube = np.full((nx, ny, nz), np.nan, dtype=np.float64)
    valid_cube = np.zeros((nx, ny, nz), dtype=bool)
    value_cube[ie, in_, id_] = v
    valid_cube[ie, in_, id_] = valid

    if src.get("derive") == "conductivity_from_rho":
        value_cube, units = resistivity_to_conductivity(value_cube, units)

    e_edges, _ = cell_edges_from_centers(e_axis)
    n_edges, _ = cell_edges_from_centers(n_axis)
    d_edges, _ = cell_edges_from_centers(d_axis)

    # (nx, ny, nz) = (easting, northing, depth) -> (depth, northing,
    # easting) so cell order matches hex_mesh_from_corner_grid()'s own
    # k(depth)-major, j(northing), i(easting) convention.
    Zc, Yc, Xc = np.meshgrid(d_edges, n_edges, e_edges, indexing="ij")
    nodes, conn, cell_volume = hex_mesh_from_corner_grid(Xc, Yc, Zc)

    value_flat = np.transpose(value_cube, (2, 1, 0)).ravel()
    valid_flat = np.transpose(valid_cube, (2, 1, 0)).ravel()

    return dict(
        cell_type="HEX8", nodes=nodes, conn=conn,
        value=value_flat, valid=valid_flat, volume=cell_volume, units=units,
        # Only a genuinely rectilinear source (this one) can feed the
        # esmpy backend -- see remap_esmpy()'s NotImplementedError for
        # every other kind.
        rectilinear_axes=dict(
            e_edges=e_edges, n_edges=n_edges, d_edges=d_edges,
            value_cube=value_cube, valid_cube=valid_cube,
        ),
    )


def load_femtic_cells(key, src):
    """
    Build the genuine tetrahedral FEMTIC source mesh from
    femtic_mesh_utm.nc (precompute.py Part C, UTM METRES) + the
    matching resistivity value from femtic_submesh_points.nc. See
    module docstring's "femtic_cells" entry.
    """
    mesh_path = ncpath(src["mesh_file"])
    pts_path = ncpath(src["points_file"])

    mesh_ds = xr.open_dataset(mesh_path)
    try:
        nx_v, ny_v, nz_v = FEMTIC_MESH_NODE_VARS
        for v in (nx_v, ny_v, nz_v, FEMTIC_MESH_CONN_VAR):
            if v not in mesh_ds.variables:
                raise KeyError(
                    f"{mesh_path!r} has no variable {v!r} -- adjust "
                    f"FEMTIC_MESH_NODE_VARS/FEMTIC_MESH_CONN_VAR to match "
                    f"precompute.py Part C's actual femtic_mesh_utm.nc "
                    f"schema (found: {list(mesh_ds.variables)})."
                )
        nodes_m = np.column_stack([
            mesh_ds[nx_v].values, mesh_ds[ny_v].values, mesh_ds[nz_v].values,
        ]).astype(np.float64)
        conn = mesh_ds[FEMTIC_MESH_CONN_VAR].values.astype(np.int64)
    finally:
        mesh_ds.close()
    if conn.ndim != 2 or conn.shape[1] != 4:
        raise ValueError(
            f"{FEMTIC_MESH_CONN_VAR!r} in {mesh_path!r} must have shape "
            f"(nelem, 4) (tetrahedral connectivity); got {conn.shape}."
        )

    pts_ds = xr.open_dataset(pts_path)
    value_var = src["value_var"]
    if value_var not in pts_ds.data_vars:
        pts_ds.close()
        raise KeyError(f"{pts_path!r} has no variable {value_var!r}.")
    v = pts_ds[value_var].values.astype(np.float64)
    valid = pts_ds["valid"].values.astype(bool) & np.isfinite(v)
    e_km = pts_ds["easting"].values.astype(np.float64)
    n_km = pts_ds["northing"].values.astype(np.float64)
    d_km = pts_ds["depth"].values.astype(np.float64)
    units = pts_ds[value_var].attrs.get("units", "")
    pts_ds.close()

    if conn.shape[0] != len(v):
        raise ValueError(
            f"{src['mesh_file']!r} has {conn.shape[0]} elements but "
            f"{src['points_file']!r} has {len(v)} rows -- these two "
            f"precompute.py Part C outputs are expected to share one row "
            f"per FEMTIC element, in the same order. Re-run precompute.py."
        )

    nodes_km = nodes_m / 1000.0
    coords = nodes_km[conn]                 # (nelem, 4, 3)
    centroid_km = coords.mean(axis=1)       # (nelem, 3) = [E, N, depth]

    check_idx = np.where(valid)[0]
    if check_idx.size:
        rng = np.random.default_rng(FEMTIC_CENTROID_CHECK_SEED)
        sample = rng.choice(
            check_idx, size=min(FEMTIC_CENTROID_CHECK_N, check_idx.size),
            replace=False,
        )
        d_east = np.abs(centroid_km[sample, 0] - e_km[sample])
        d_north = np.abs(centroid_km[sample, 1] - n_km[sample])
        d_depth = np.abs(centroid_km[sample, 2] - d_km[sample])
        worst = max(d_east.max(), d_north.max(), d_depth.max())
        if worst > FEMTIC_CENTROID_CHECK_TOL_KM:
            raise ValueError(
                f"{src['mesh_file']!r} element centroids do not match "
                f"{src['points_file']!r}'s own stored centroids (max "
                f"mismatch E={d_east.max():.3f}, N={d_north.max():.3f}, "
                f"D={d_depth.max():.3f} km, tol="
                f"{FEMTIC_CENTROID_CHECK_TOL_KM} km) -- the two files "
                f"appear to use a different element order; remapping.py "
                f"cannot safely assume row i in one corresponds to "
                f"element i in the other."
            )
        print(
            f"    Centroid cross-check ({sample.size} elements): "
            f"max mismatch {worst * 1000:.1f} m (tol "
            f"{FEMTIC_CENTROID_CHECK_TOL_KM * 1000:.0f} m) -- OK"
        )

    if src.get("derive") == "conductivity_from_rho":
        v, units = resistivity_to_conductivity(v, units)

    volume = tetrahedron_volume(coords)

    return dict(
        cell_type="TET4", nodes=nodes_km, conn=conn,
        value=v, valid=valid, volume=volume, units=units,
    )


def load_seis_cells(key, src):
    """
    Build the curvilinear HEX8 cell mesh for one precompute.py Part B
    seismic tomography grid. See module docstring's "seis_grid" entry.
    """
    ds = xr.open_dataset(ncpath(src["file"]))
    da = ds[src["var"]]
    dim_depth, dim_row, dim_col = da.dims
    depth = ds[dim_depth].values.astype(np.float64)
    e2d_km = ds["utm_easting"].values.astype(np.float64) / 1e3
    n2d_km = ds["utm_northing"].values.astype(np.float64) / 1e3
    value_cube = da.values.astype(np.float64)   # (depth, row, col)
    units = da.attrs.get("units", "")
    ds.close()

    valid_cube = np.isfinite(value_cube)

    if src.get("derive") == "conductivity_from_rho":
        value_cube, units = resistivity_to_conductivity(value_cube, units)

    e_corners = _corners_from_centers_2d(e2d_km)   # (nrow+1, ncol+1)
    n_corners = _corners_from_centers_2d(n2d_km)
    d_edges, _ = cell_edges_from_centers(depth)

    nk1 = d_edges.size
    nj1, ni1 = e_corners.shape
    Xc = np.broadcast_to(e_corners[None, :, :], (nk1, nj1, ni1))
    Yc = np.broadcast_to(n_corners[None, :, :], (nk1, nj1, ni1))
    Zc = np.broadcast_to(d_edges[:, None, None], (nk1, nj1, ni1))
    nodes, conn, cell_volume = hex_mesh_from_corner_grid(
        np.array(Xc), np.array(Yc), np.array(Zc),
    )

    # value_cube is already (depth, row, col) = (k, j, i) order -- no
    # transpose needed, unlike load_modem_cells().
    value_flat = value_cube.ravel()
    valid_flat = valid_cube.ravel()

    return dict(
        cell_type="HEX8", nodes=nodes, conn=conn,
        value=value_flat, valid=valid_flat, volume=cell_volume, units=units,
    )


def build_source_cells(key):
    """Dispatch to the right native-cell loader for VARIABLE_SOURCES[key]."""
    src = VARIABLE_SOURCES[key]
    if src["kind"] == "modem_cells":
        return load_modem_cells(key, src)
    elif src["kind"] == "femtic_cells":
        return load_femtic_cells(key, src)
    elif src["kind"] == "seis_grid":
        return load_seis_cells(key, src)
    else:
        raise ValueError(f"Unknown VARIABLE_SOURCES kind {src['kind']!r} for {key!r}")


# ------------------------------------------------------------------
# Target grid
# ------------------------------------------------------------------
def _cells_bbox(cells):
    """(e_lo, e_hi), (n_lo, n_hi), (d_lo, d_hi) spanned by a cell mesh's
    VALID cells only (nodes referenced only by invalid cells are
    ignored, matching interpolate.py's own philosophy of never letting
    excluded/invalid material widen a variable's reported footprint)."""
    valid_conn = cells["conn"][cells["valid"]]
    if valid_conn.size == 0:
        raise RuntimeError("Cell mesh has no valid cells at all.")
    node_ids = np.unique(valid_conn)
    pts = cells["nodes"][node_ids]
    return (
        (float(pts[:, 0].min()), float(pts[:, 0].max())),
        (float(pts[:, 1].min()), float(pts[:, 1].max())),
        (float(pts[:, 2].min()), float(pts[:, 2].max())),
    )


def build_target_grid(active_vars, source_cells):
    """
    Build the target HEX8 cell mesh -- a fresh, genuinely regular UTM-km
    grid (GRID_*_KM), auto-bounded to the tightest common overlap of
    every active variable's own valid-cell footprint unless given
    explicit bounds. See module docstring's "Target grid" section.
    """
    bboxes = [_cells_bbox(source_cells[k]) for k in active_vars]

    def _grid_axis(spec, axis_index, name):
        auto_lo = max(b[axis_index][0] for b in bboxes)
        auto_hi = min(b[axis_index][1] for b in bboxes)
        lo = spec["min"] if spec["min"] is not None else auto_lo
        hi = spec["max"] if spec["max"] is not None else auto_hi
        if hi <= lo:
            raise RuntimeError(
                f"Target grid {name} range is empty ({lo} .. {hi}) -- the "
                f"selected REMAP_VARS don't overlap along this axis. "
                f"Check GRID_{name.upper()}_KM or REMAP_VARS."
            )
        edges = np.arange(lo, hi + spec["step"] / 2.0, spec["step"])
        centers = 0.5 * (edges[:-1] + edges[1:])
        print(
            f"  {name}: {edges[0]:.2f} .. {edges[-1]:.2f} km, "
            f"step {spec['step']} km ({len(centers)} cells)"
        )
        return edges, centers

    e_edges, e_axis = _grid_axis(GRID_EASTING_KM, 0, "easting")
    n_edges, n_axis = _grid_axis(GRID_NORTHING_KM, 1, "northing")
    d_edges, d_axis = _grid_axis(GRID_DEPTH_KM, 2, "depth")

    Zc, Yc, Xc = np.meshgrid(d_edges, n_edges, e_edges, indexing="ij")
    nodes, conn, cell_volume = hex_mesh_from_corner_grid(Xc, Yc, Zc)
    grid_shape = (len(d_axis), len(n_axis), len(e_axis))  # (depth, northing, easting)

    return dict(
        dim_names=("depth", "northing", "easting"),
        coords={"depth": d_axis, "northing": n_axis, "easting": e_axis},
        nodes=nodes, conn=conn, volume=cell_volume, grid_shape=grid_shape,
        cell_type="HEX8",
        rectilinear_axes=dict(e_edges=e_edges, n_edges=n_edges, d_edges=d_edges),
        _mc_mesh=None,   # lazily built/cached by get_target_mc_mesh()
    )


def get_target_mc_mesh(target):
    """Build (once) and cache the target grid's MEDCouplingUMesh."""
    if target["_mc_mesh"] is None:
        target["_mc_mesh"] = _mc_umesh(
            target["nodes"], target["conn"], target["cell_type"], "target",
        )
    return target["_mc_mesh"]


# ------------------------------------------------------------------
# Backend: MEDCoupling (all source kinds)
# ------------------------------------------------------------------
def _mc_umesh(nodes, conn, cell_type, name):
    """
    Build a MEDCouplingUMesh (mesh dim 3) from an explicit node/
    connectivity array, via the documented insertNextCell() loop (safe
    and unambiguous; a vectorised setConnectivity() call is possible for
    very large meshes but is skipped here in favour of the API usage
    every MEDCoupling tutorial documents).
    """
    import medcoupling as mc

    norm = mc.NORM_HEXA8 if cell_type == "HEX8" else mc.NORM_TETRA4
    mesh = mc.MEDCouplingUMesh(name, 3)
    mesh.setCoords(mc.DataArrayDouble(np.ascontiguousarray(nodes, dtype=np.float64)))
    mesh.allocateCells(conn.shape[0])
    for row in conn:
        mesh.insertNextCell(norm, row.tolist())
    mesh.finishInsertingCells()
    mesh.checkConsistencyLight()
    return mesh


def remap_medcoupling(cells, target):
    """
    Conservative P0P0 remap of one source's valid cells onto the
    (cached) target MEDCouplingUMesh in target["_mc_mesh"].

    Nature choice (see module docstring / MEDCoupling docs):
      - value field: IntensiveMaximum -- W_ij = Vol(Ti ∩ Sj) /
        sum_j Vol(Ti ∩ Sj), i.e. a true average over only the volume
        that actually overlaps Ti (an intensive quantity like
        resistivity/Vp should not be diluted by an Ti's uncovered
        remainder).
      - coverage field: source values set to each cell's own VOLUME,
        nature=ExtensiveConservation -- W_ij = Vol(Ti ∩ Sj) / Vol(Sj),
        so transferring "volume-valued" cells gives
        sum_j Vol(Ti ∩ Sj) = the true covered volume of Ti directly;
        dividing by Ti's own volume then gives the covered FRACTION.

    Returns (value_out, coverage_out), both shape (target ncell,), in
    the same cell order as target["conn"].
    """
    import medcoupling as mc

    valid = cells["valid"]
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise RuntimeError("No valid source cells to remap.")

    src_mesh = _mc_umesh(cells["nodes"], cells["conn"][valid], cells["cell_type"], "src")
    tgt_mesh = get_target_mc_mesh(target)

    rem = mc.MEDCouplingRemapper()
    n_pairs = rem.prepare(src_mesh, tgt_mesh, "P0P0")
    if n_pairs == 0:
        print("    WARNING: MEDCouplingRemapper found no source/target "
              "cell overlap at all -- check the two meshes actually cover "
              "the same region.")

    value_field = mc.MEDCouplingFieldDouble(mc.ON_CELLS, mc.ONE_TIME)
    value_field.setMesh(src_mesh)
    value_field.setArray(mc.DataArrayDouble(cells["value"][valid].astype(np.float64).copy()))
    value_field.setNature(mc.IntensiveMaximum)
    value_out_field = rem.transferField(value_field, float("nan"))
    value_out = np.asarray(value_out_field.getArray().toNumPyArray(), dtype=np.float64).ravel()

    volume_field = mc.MEDCouplingFieldDouble(mc.ON_CELLS, mc.ONE_TIME)
    volume_field.setMesh(src_mesh)
    volume_field.setArray(mc.DataArrayDouble(cells["volume"][valid].astype(np.float64).copy()))
    volume_field.setNature(mc.ExtensiveConservation)
    covered_volume_field = rem.transferField(volume_field, 0.0)
    covered_volume = np.asarray(
        covered_volume_field.getArray().toNumPyArray(), dtype=np.float64
    ).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        coverage_out = np.where(
            target["volume"] > 0, covered_volume / target["volume"], 0.0,
        )
    coverage_out = np.clip(coverage_out, 0.0, 1.0)
    return value_out, coverage_out


# ------------------------------------------------------------------
# Backend: esmpy (RECTILINEAR sources only -- "modem_cells" -> target)
# ------------------------------------------------------------------
def _build_esmpy_grid(e_edges, n_edges, d_edges, valid_cube=None):
    """
    Build a 3-D Cartesian esmpy.Grid with both CENTER_VCENTER and
    CORNER_VFACE coordinates set, following the local-bounds slicing
    pattern from ESMPy's own `grid_create_from_coordinates_3d()`
    tutorial helper (correct for both serial and parallel runs; this
    script is only ever exercised serially, but the slicing costs
    nothing extra). valid_cube, if given (nx, ny, nz) bool, is written
    into a GridItem.MASK (1 = excluded from regridding) at CENTER_VCENTER.
    """
    import esmpy

    nx, ny, nz = len(e_edges) - 1, len(n_edges) - 1, len(d_edges) - 1
    ec = 0.5 * (e_edges[:-1] + e_edges[1:])
    nc = 0.5 * (n_edges[:-1] + n_edges[1:])
    dc = 0.5 * (d_edges[:-1] + d_edges[1:])

    grid = esmpy.Grid(
        np.array([nx, ny, nz], dtype=np.int32),
        staggerloc=[esmpy.StaggerLoc.CENTER_VCENTER],
        coord_sys=esmpy.CoordSys.CART,
    )
    grid.add_coords([esmpy.StaggerLoc.CORNER_VFACE])

    lbc = grid.lower_bounds[esmpy.StaggerLoc.CENTER_VCENTER]
    ubc = grid.upper_bounds[esmpy.StaggerLoc.CENTER_VCENTER]
    for dim, arr in enumerate((ec, nc, dc)):
        g = grid.get_coords(dim)
        par = arr[lbc[dim]:ubc[dim]]
        shape = [1, 1, 1]
        shape[dim] = par.size
        g[...] = par.reshape(shape)

    lbv = grid.lower_bounds[esmpy.StaggerLoc.CORNER_VFACE]
    ubv = grid.upper_bounds[esmpy.StaggerLoc.CORNER_VFACE]
    for dim, arr in enumerate((e_edges, n_edges, d_edges)):
        g = grid.get_coords(dim, staggerloc=esmpy.StaggerLoc.CORNER_VFACE)
        par = arr[lbv[dim]:ubv[dim]]
        shape = [1, 1, 1]
        shape[dim] = par.size
        g[...] = par.reshape(shape)

    if valid_cube is not None:
        grid.add_item(esmpy.GridItem.MASK, staggerloc=esmpy.StaggerLoc.CENTER_VCENTER)
        mask = grid.get_item(esmpy.GridItem.MASK, staggerloc=esmpy.StaggerLoc.CENTER_VCENTER)
        mask_local = (~valid_cube[lbc[0]:ubc[0], lbc[1]:ubc[1], lbc[2]:ubc[2]]).astype(np.int32)
        mask[...] = mask_local

    return grid, (nx, ny, nz), (lbc, ubc)


def remap_esmpy(cells, target):
    """
    Conservative regrid via esmpy.Regrid(regrid_method=CONSERVE[_2ND])
    -- ONLY for a source that carries a "rectilinear_axes" entry (i.e.
    "modem_cells" -- see module docstring's honest limitation on why
    "femtic_cells"/"seis_grid" aren't attempted here).

    ESMF's own CONSERVE weights are W_ij = Vol(Ti ∩ Sj) / Vol(Ti) (see
    ESMPy's docs: "weights ... are the area of intersection of each
    source cell with the destination cell divided by the area of the
    destination cell") -- i.e. an EXTENSIVE-style weight that dilutes a
    partially-covered destination cell rather than averaging only its
    covered part. Recovered here with the same "regrid a ones field
    too" trick as the MEDCoupling backend: regridding a constant-1
    source field with these exact weights gives the covered FRACTION
    directly (bounded in [0, 1] already, no extra volume bookkeeping
    needed), and dividing the raw regridded value by that same fraction
    turns ESMF's diluted result into the proper covered-volume average.
    """
    if "rectilinear_axes" not in cells:
        raise NotImplementedError(
            "REMAP_BACKEND='esmpy' only implements rectilinear-grid "
            "sources ('modem_cells'); this source's own cell geometry "
            "is unstructured/curvilinear -- set REMAP_BACKEND="
            "'medcoupling' (globally, or via this VARIABLE_SOURCES "
            "entry's own 'backend' override) instead. See module "
            "docstring."
        )
    import esmpy

    ax = cells["rectilinear_axes"]
    src_grid, src_shape, _ = _build_esmpy_grid(
        ax["e_edges"], ax["n_edges"], ax["d_edges"], valid_cube=ax["valid_cube"],
    )
    tax = target["rectilinear_axes"]
    dst_grid, dst_shape, (dlb, dub) = _build_esmpy_grid(
        tax["e_edges"], tax["n_edges"], tax["d_edges"], valid_cube=None,
    )

    srcfield = esmpy.Field(src_grid, name="value", staggerloc=esmpy.StaggerLoc.CENTER_VCENTER)
    dstfield = esmpy.Field(dst_grid, name="value_out", staggerloc=esmpy.StaggerLoc.CENTER_VCENTER)
    onesfield = esmpy.Field(src_grid, name="ones", staggerloc=esmpy.StaggerLoc.CENTER_VCENTER)
    onesdst = esmpy.Field(dst_grid, name="ones_out", staggerloc=esmpy.StaggerLoc.CENTER_VCENTER)

    slbc = src_grid.lower_bounds[esmpy.StaggerLoc.CENTER_VCENTER]
    subc = src_grid.upper_bounds[esmpy.StaggerLoc.CENTER_VCENTER]
    value_local = ax["value_cube"][slbc[0]:subc[0], slbc[1]:subc[1], slbc[2]:subc[2]]
    srcfield.data[...] = np.where(np.isfinite(value_local), value_local, 0.0)
    onesfield.data[...] = 1.0

    method = {
        "conserve": esmpy.RegridMethod.CONSERVE,
        "conserve2nd": esmpy.RegridMethod.CONSERVE_2ND,
    }[ESMPY_CONSERVE_METHOD]

    regridder = esmpy.Regrid(
        srcfield, dstfield, regrid_method=method,
        unmapped_action=esmpy.UnmappedAction.IGNORE,
        src_mask_values=np.array([1], dtype=np.int32),
    )
    dstfield = regridder(srcfield, dstfield)
    val_raw = np.array(dstfield.data, dtype=np.float64)

    onesdst = regridder(onesfield, onesdst)
    coverage_local = np.clip(np.array(onesdst.data, dtype=np.float64), 0.0, 1.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        val_local = np.where(coverage_local > 0, val_raw / coverage_local, np.nan)

    # Scatter this rank's local block back into a full (nx, ny, nz)
    # array -- trivial (whole array) for the serial case this script
    # actually runs in; kept general via lower/upper bounds regardless.
    dlbc = dst_grid.lower_bounds[esmpy.StaggerLoc.CENTER_VCENTER]
    dubc = dst_grid.upper_bounds[esmpy.StaggerLoc.CENTER_VCENTER]
    val_full = np.full(dst_shape, np.nan, dtype=np.float64)
    cov_full = np.zeros(dst_shape, dtype=np.float64)
    val_full[dlbc[0]:dubc[0], dlbc[1]:dubc[1], dlbc[2]:dubc[2]] = val_local
    cov_full[dlbc[0]:dubc[0], dlbc[1]:dubc[1], dlbc[2]:dubc[2]] = coverage_local

    # (nx, ny, nz) = (easting, northing, depth) -> flatten in the same
    # k(depth)-major, j(northing), i(easting) order the target mesh's
    # own cell order uses (see build_target_grid()/hex_mesh_from_corner_grid()).
    value_out = np.transpose(val_full, (2, 1, 0)).ravel()
    coverage_out = np.transpose(cov_full, (2, 1, 0)).ravel()
    return value_out, coverage_out


# ------------------------------------------------------------------
# Dispatch + region-of-interest mask (identical convention to
# interpolate.py's outside_roi())
# ------------------------------------------------------------------
def outside_roi(centers_3d, vertices, depth_min, depth_max):
    """
    Boolean mask, True where target-grid CELL CENTERS (easting_km,
    northing_km, depth_km) fall outside the RoI box. Same convention as
    interpolate.py's outside_roi(), operating on cell centers here
    since the remap itself (not a separate hull check) already
    determined which cells have real source coverage.
    """
    e_axis, n_axis, d_axis = centers_3d
    Ngrid, Egrid = np.meshgrid(n_axis, e_axis, indexing="ij")  # (nn, ne)
    poly = MplPath(np.asarray(vertices, dtype=np.float64))
    outside_2d = ~poly.contains_points(np.column_stack([Egrid.ravel(), Ngrid.ravel()]))
    outside_2d = outside_2d.reshape(Egrid.shape)  # (nn, ne)
    outside = np.broadcast_to(outside_2d[None, :, :], (len(d_axis), len(n_axis), len(e_axis))).copy()
    if depth_min is not None:
        outside |= (d_axis[:, None, None] < depth_min)
    if depth_max is not None:
        outside |= (d_axis[:, None, None] > depth_max)
    return outside


def remap_variable(key, cells, target, backend):
    """Dispatch to the configured (or per-source-overridden) backend."""
    if backend == "medcoupling":
        return remap_medcoupling(cells, target)
    elif backend == "esmpy":
        return remap_esmpy(cells, target)
    else:
        raise ValueError(f"REMAP_BACKEND must be 'medcoupling' or 'esmpy', got {backend!r}.")


# ==================================================================
# Resolve active variables (apply USE_CONDUCTIVITY swap)
# ==================================================================
active_vars = list(REMAP_VARS)
if USE_CONDUCTIVITY:
    active_vars = ["cond" if v == "rho" else v for v in active_vars]
    if active_vars != list(REMAP_VARS):
        print(f"USE_CONDUCTIVITY=True -- remapping {active_vars} "
              f"instead of {REMAP_VARS}")

print(f"Remapping (conservative): {active_vars}  "
      f"(default backend={REMAP_BACKEND!r})")

# ==================================================================
# Load native source cells
# ==================================================================
source_cells = {}
for key in active_vars:
    if key not in VARIABLE_SOURCES:
        raise KeyError(
            f"REMAP_VARS entry {key!r} has no matching VARIABLE_SOURCES "
            f"registration. Known keys: {list(VARIABLE_SOURCES)}"
        )
    src = VARIABLE_SOURCES[key]
    print(f"  Loading {key!r} ({src['kind']}) …")
    cells = build_source_cells(key)
    n_valid = int(cells["valid"].sum())
    n_total = cells["valid"].size
    total_volume = float(cells["volume"][cells["valid"]].sum())
    print(
        f"    {n_valid} / {n_total} valid {cells['cell_type']} cells "
        f"({100.0 * n_valid / n_total:.1f}%), total volume "
        f"{total_volume:,.1f} km^3, units={cells['units']!r}"
    )
    source_cells[key] = cells

# ==================================================================
# Build the target grid
# ==================================================================
print("\nBuilding target grid …")
target = build_target_grid(active_vars, source_cells)
grid_shape = target["grid_shape"]
n_total = int(np.prod(grid_shape))
print(f"Target grid dims {target['dim_names']}, shape {grid_shape} "
      f"({n_total} cells total)")

if APPLY_ROI_MASK:
    roi_outside = outside_roi(
        (target["coords"]["easting"], target["coords"]["northing"],
         target["coords"]["depth"]),
        ROI_VERTICES_KM, ROI_DEPTH_MIN_KM, ROI_DEPTH_MAX_KM,
    )
    print(
        f"\nRoI mask: {int(roi_outside.sum())} / {roi_outside.size} target-grid "
        f"cells fall outside the RoI box and will be set to NaN "
        f"(applies identically to every variable)"
    )
else:
    roi_outside = None

# ==================================================================
# Conservative remap, one variable at a time
# ==================================================================
remapped = {}
coverage = {}
resolved_units = {}
for key in active_vars:
    src = VARIABLE_SOURCES[key]
    backend = src.get("backend", REMAP_BACKEND)
    print(f"\nRemapping {key!r} onto the target grid (backend={backend!r}) …")
    value_flat, coverage_flat = remap_variable(key, source_cells[key], target, backend)

    value_flat = np.where(coverage_flat >= MIN_COVERAGE_FRACTION, value_flat, np.nan)
    n_covered = int(np.sum(coverage_flat >= MIN_COVERAGE_FRACTION))
    print(
        f"    {n_covered} / {n_total} target cells covered >= "
        f"{MIN_COVERAGE_FRACTION:.0%} (MIN_COVERAGE_FRACTION); "
        f"mean coverage over those: "
        f"{coverage_flat[coverage_flat >= MIN_COVERAGE_FRACTION].mean():.3f}"
        if n_covered else "    0 target cells reached MIN_COVERAGE_FRACTION -- "
                            "check REMAP_VARS/GRID_*_KM/the source data itself."
    )

    value_grid = value_flat.reshape(grid_shape)
    coverage_grid = coverage_flat.reshape(grid_shape)
    if APPLY_ROI_MASK:
        value_grid = np.where(roi_outside, np.nan, value_grid)

    remapped[key] = value_grid.astype(np.float32)
    coverage[key] = coverage_grid.astype(np.float32)
    resolved_units[key] = source_cells[key]["units"]

    # Volume-conservation sanity print: total remapped "mass" (value *
    # covered volume) vs the source's own total (value * cell volume),
    # restricted to source cells that fall within the target's own
    # bounding box -- won't match exactly (target cells straddling the
    # RoI/coverage threshold are excluded on one side only), but should
    # be in the right ballpark; a wildly different number is a red flag
    # worth investigating before trusting this variable's clustering
    # input.
    finite = np.isfinite(value_grid)
    if np.any(finite):
        remapped_mass = float(np.sum(
            value_grid[finite].astype(np.float64) *
            (coverage_grid[finite].astype(np.float64) * target["volume"].reshape(grid_shape)[finite])
        ))
        print(f"    Sanity check -- sum(value * covered_volume) over kept "
              f"cells: {remapped_mass:,.3f} ({key} units * km^3)")

# ==================================================================
# Save
# ==================================================================
meta = dict(
    dim_names=list(target["dim_names"]),
    remap_vars=active_vars,
    units={k: (resolved_units[k] or "") for k in active_vars},
    label={k: VARIABLE_SOURCES[k]["label"] for k in active_vars},
    remap_backend={k: VARIABLE_SOURCES[k].get("backend", REMAP_BACKEND) for k in active_vars},
    esmpy_conserve_method=ESMPY_CONSERVE_METHOD,
    min_coverage_fraction=MIN_COVERAGE_FRACTION,
    use_conductivity=USE_CONDUCTIVITY,
    grid_easting_km=GRID_EASTING_KM,
    grid_northing_km=GRID_NORTHING_KM,
    grid_depth_km=GRID_DEPTH_KM,
    modem_edge_extrapolation=MODEM_EDGE_EXTRAPOLATION,
    roi_applied=APPLY_ROI_MASK,
    roi_vertices_km=ROI_VERTICES_KM if APPLY_ROI_MASK else None,
    roi_depth_range_km=[ROI_DEPTH_MIN_KM, ROI_DEPTH_MAX_KM] if APPLY_ROI_MASK else None,
    description=(
        f"{', '.join(active_vars)} conservatively remapped onto a "
        f"jointly-defined regular UTM-km grid -- see remapping.py."
    ),
)

npz_payload = {
    "depth": target["coords"]["depth"].astype(np.float64),
    "northing": target["coords"]["northing"].astype(np.float64),
    "easting": target["coords"]["easting"].astype(np.float64),
    "meta_json": json.dumps(meta),
}
for key in active_vars:
    npz_payload[key] = remapped[key]
    npz_payload[f"{key}_coverage"] = coverage[key]

output_file = OUTPUT_FILE or f"{SITE_PREFIX}_remap.npz"
out_path = ncpath(output_file)
np.savez_compressed(out_path, **npz_payload)
print(f"\nSaved: {out_path}")
print("\nDone.")
