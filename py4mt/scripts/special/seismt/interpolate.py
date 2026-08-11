#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
interpolate.py
=====================
Loads MT resistivity/conductivity/sensitivity + seismic tomography
properties (Vp, Vs, Vp/Vs, density), each from its own NATIVE grid
produced by precompute.py, and interpolates every requested one
onto ONE common target grid — either a freshly-defined regular UTM-km
grid ("joint" mode) or the seismic tomography's own native (depth, row,
col) grid, reused as-is ("seismic" mode) — via RBF, ordinary kriging, or
inverse-distance weighting (IDW).

Site-specific settings (which are few here, mainly the SITE_PREFIX used
to find precompute.py's outputs) are kept as capitalized constants below;
where a setting differs between sites, the currently-active value is live
and any other site's value is kept as a labeled, commented-out
alternative (e.g. "# TACNA") right next to it.

This is the interpolation-only half of what used to be
cluster_rbf.py / cluster_kriging.py / cluster_som.py:
those three scripts each duplicated grid-building + interpolation +
clustering + plotting, differing only in interpolation method (RBF vs.
kriging) or clustering method (fuzzy c-means vs. SOM). Grid-building and
interpolation now live HERE, once, with the method picked by
INTERP_METHOD; cluster.py reads this script's output netCDF
(INTERP_FILE) and no longer builds any grid or interpolates anything
itself — see that script's own header for the clustering half.

Pipeline
--------
precompute.py  →  interpolate.py  →  {SITE_PREFIX}_interp_<method>.nc
                                                        ↓
                                          cluster.py  →  {SITE_PREFIX}_clusters.nc + figures

Two kinds of native source (VARIABLE_SOURCES below)
-----------------------------------------------------
  "modem_points" — a flat point table (precompute.py's
    modem_submesh_points.nc, Part A): already (easting_km, northing_km,
    depth_km, value) rows at full native ModEM resolution.
  "seis_grid"    — a gridded (depth, row, col) cube (precompute.py
    Part B's {SITE_PREFIX}_vp.nc / {SITE_PREFIX}_vs.nc / {SITE_PREFIX}_vps.nc / {SITE_PREFIX}_dens.nc),
    with 2-D utm_easting/utm_northing aux coords — flattened, with the
    horizontal coords broadcast across every depth level, into a point
    cloud by load_seis_grid_points() below. "row"/"col" are whatever
    that source's own second/third dimension is actually called (lat/
    lon, y/x, ...) — never assumed/hard-coded, always read from the
    file itself.

Target grid: TARGET_GRID = "joint" | "seismic"
------------------------------------------------
  "joint"    — build a fresh, genuinely regular UTM-km grid (explicit
    GRID_*_KM bounds, or auto = the tightest common overlap of every
    active variable's own extent), independent of any source's own
    resolution. Output dims: (depth, northing, easting).
  "seismic"  — skip building a new grid; reuse one seis_grid source's
    own native (depth, row, col) grid as the interpolation target
    instead (SEISMIC_MESH_VAR below picks which one). Useful to avoid a
    second resampling step for whichever seismic variable you treat as
    the "reference" grid, and to keep cluster output on that model's own
    native resolution rather than an independently-chosen one. Output
    dims: (depth, row, col), with 2-D utm_easting_km/utm_northing_km aux
    coords carried over from the reference source (Not a regular grid in
    UTM space — cluster.py's plotting handles this with
    pcolormesh(shading="nearest") using those 2-D coords, rather than
    imshow's regular-grid extent=).

Interpolation methods: INTERP_METHOD = "rbf" | "kriging" | "idw" | "nearest"
--------------------------------------------------------------------------------
  "rbf"     — scipy.interpolate.RBFInterpolator, fit independently per
    variable on its own native point cloud (RBF_* settings).
  "kriging" — pykrige.ok3d.OrdinaryKriging3D, ordinary kriging with a
    per-variable variogram fit (KRIGING_* settings). Requires
    `pip install pykrige`; native point clouds are randomly subsampled
    to KRIGING_MAX_POINTS first (variogram estimation is O(n^2)).
  "idw"     — simple inverse-distance weighting over each query point's
    IDW_NEIGHBORS nearest native points (scipy.spatial.cKDTree), weight
    = 1 / distance**IDW_POWER. No extra dependency, cheap, and exact at
    source points (no shape parameter or variogram to fit) but purely
    local — no smooth extrapolation behaviour of its own past its
    neighbor set the way RBF/kriging have.
  "nearest" — nearest-neighbor: each target-grid point simply takes the
    value of its single closest native point (scipy.spatial.cKDTree,
    k=1). No extra dependency, the cheapest of the four, and useful as a
    quick sanity baseline (or deliberately blocky classification-style
    map) — produces piecewise-constant Voronoi-cell output rather than
    any smooth field, unlike the other three methods.

All four extrapolate/interpolate past a variable's own data footprint
with no natural cutoff; MASK_TO_CONVEX_HULL (default True) nulls out
target-grid points outside each variable's own 3-D convex hull
(scipy.spatial.Delaunay) so that footprint is respected regardless of
method. APPLY_ROI_MASK optionally additionally restricts every variable
to one shared rectangular region (see "Region of interest" below).

Gradient (optional, COMPUTE_GRADIENT)
----------------------------------------
If enabled, the spatial gradient of each GRADIENT_VARS entry is computed
on the SAME grid the field was just interpolated onto, and stored
alongside it in INTERP_FILE as four extra data variables per variable:
"{key}_grad_easting", "{key}_grad_northing", "{key}_grad_depth" (partial
derivatives, value-units per km) and "{key}_grad_mag" (their combined
3-D magnitude). GRADIENT_METHOD == "spline" fits a 1-D
scipy.interpolate.UnivariateSpline along each grid axis through the
already-interpolated values and differentiates it analytically
(_derivative_along_axis()) — "finite_difference" uses plain
numpy.gradient instead.

For TARGET_GRID == "joint" this is a plain per-axis partial derivative,
since the grid is genuinely regular in UTM easting/northing/depth. For
TARGET_GRID == "seismic" the grid is regular in (row, col) index space
but NOT in UTM space (see build_seismic_target_grid()), so a raw
row/col-axis derivative would not be a physical easting/northing
gradient: the row/col-space derivatives are instead converted to true
easting/northing derivatives via the local 2x2 Jacobian of the
(row, col) -> (easting, northing) mapping (itself differentiated the
same way, from the grid's own utm_easting_km/utm_northing_km 2-D
coordinates) — see the "seismic" branch below _derivative_along_axis().
Gradients are NOT added to INTERP_FILE's own interp_vars attribute, so
they're never picked up by cluster.py's default (None ->
"everything") CLUSTER_VARS; add "{key}_grad_mag" etc. to CLUSTER_VARS
explicitly there to cluster on them too.

Dependencies
------------
    numpy, xarray, scipy (RBFInterpolator, Delaunay, cKDTree)
plus pykrige (only if INTERP_METHOD == "kriging"; imported lazily so it
is not a hard dependency for the other two methods).

Authors: Svetlana Byrdina (SMB) & Volker Rath (DIAS)
AI-assisted development: Claude (Anthropic). Split out of
cluster_rbf.py / cluster_kriging.py / cluster_som.py
into its own interpolation-only script (grid-building + RBF/kriging/IDW
interpolation, with an option to reuse a seismic source's own native
grid instead of building a new joint one), decoupled from the
clustering step: Claude (Anthropic), 2026-08-02.
License: GNU General Public License v3 (GPL-3.0-or-later).
AI-generated code — review before use in production.
"""

import os
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.interpolate import RBFInterpolator
from scipy.spatial import Delaunay, cKDTree
from matplotlib.path import Path as MplPath

# Cap BLAS/OpenMP thread counts used by scipy's linear-algebra-heavy
# interpolators (RBF/kriging); set before any heavy computation, has no
# effect on numpy/scipy internals once they've already spun up their own
# thread pools on first use. Adjust or remove as suits your machine.
N_THREADS = "32"
# N_THREADS = "8"  # TACNA
os.environ["OMP_NUM_THREADS"] = N_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = N_THREADS
os.environ["MKL_NUM_THREADS"] = N_THREADS

# =====================================================================
# USER SETTINGS
# =====================================================================

# --- Site selector ---
# Must match SITE_PREFIX in precompute.py — used below to find its
# {SITE_PREFIX}_vp.nc / {SITE_PREFIX}_vs.nc / {SITE_PREFIX}_vps.nc /
# {SITE_PREFIX}_dens.nc outputs and to name this script's own default
# output file.
SITE_PREFIX = "saba"
# SITE_PREFIX = "tacna"  # TACNA

# --- Input/output directories ---
NC_DIR = "../precompute/saba/"   # must match OUTPUT_DIR in precompute.py;
                             # {SITE_PREFIX}_interp_<method>.nc is written
                             # here too, alongside its inputs.

# --- Variable registry ---
# Two kinds of source — see module docstring. Every variable becomes a
# point cloud (load_variable_points()) before interpolation onto the
# target grid, so sources on different native grids/resolutions can be
# combined freely.
VARIABLE_SOURCES = {
    "rho":  dict(kind="modem_points", file="modem_submesh_points.nc",
                 value_var="resistivity", label="log10 resistivity",
                 units="log10(Ohm.m)"),
    "cond": dict(kind="modem_points", file="modem_submesh_points.nc",
                 value_var="resistivity", label="conductivity", units=None,
                 derive="conductivity_from_rho"),
    "sens": dict(kind="modem_points", file="modem_submesh_points.nc",
                 value_var="sensitivity", label="sensitivity",
                 units="(as stored)"),
    "vp":   dict(kind="seis_grid", file=f"{SITE_PREFIX}_vp.nc",   var="data", label="Vp",      units="km/s"),
    "vs":   dict(kind="seis_grid", file=f"{SITE_PREFIX}_vs.nc",   var="data", label="Vs",      units="km/s"),
    "vps":  dict(kind="seis_grid", file=f"{SITE_PREFIX}_vps.nc",  var="data", label="Vp/Vs",   units=""),
    "dens": dict(kind="seis_grid", file=f"{SITE_PREFIX}_dens.nc", var="data", label="Density", units="(as stored)"),
}

# --- Which variables to prepare (interpolate onto the target grid) ---
# Everything listed here gets loaded, interpolated, and written into
# INTERP_FILE. cluster.py's own CLUSTER_VARS then picks any subset
# of what's available here for a given clustering run — so you can
# interpolate a superset once and cluster on different subsets without
# re-running this script, as long as you don't need a variable that
# isn't listed here.
INTERP_VARS = ["rho", "vps", "dens"]

# --- Use conductivity instead of resistivity, optionally ---
# If True, any "rho" entry in INTERP_VARS is swapped for "cond"
# (conductivity, derived from the same modem_submesh_points.nc points) at
# load time. No effect if "rho" isn't in INTERP_VARS (and don't put both
# "rho" and "cond" in INTERP_VARS directly — same information, just
# inverted).
USE_CONDUCTIVITY = False

# --- Target grid ---
# "joint"    — build a fresh regular UTM-km grid from GRID_*_KM below.
# "seismic"  — reuse SEISMIC_MESH_VAR's own native (depth, row, col)
#              grid as-is; GRID_*_KM below is then ignored.
TARGET_GRID = "joint"

# --- Joint regular interpolation grid (TARGET_GRID == "joint") ---
# min/max = None auto-computes from the intersection (tightest common
# overlap) of every active variable's own point-cloud extent, so the
# grid never reaches further than every selected variable actually
# covers. Override with explicit numbers for a fixed, reproducible grid
# regardless of which variables happen to be selected.
GRID_EASTING_KM  = dict(min=None, max=None, step=2.0)
GRID_NORTHING_KM = dict(min=None, max=None, step=2.0)
GRID_DEPTH_KM    = dict(min=-4, max=36, step=1.0)

# --- Seismic reference grid (TARGET_GRID == "seismic") ---
# Which VARIABLE_SOURCES entry's native grid to reuse as the
# interpolation target — must be a "seis_grid" kind entry.
SEISMIC_MESH_VAR = "vps"

# --- Interpolation method ---
INTERP_METHOD = "nearest"   # "rbf" | "kriging" | "idw" | "nearest"

# --- RBF settings (INTERP_METHOD == "rbf") ---
RBF_KERNEL = "linear"        # "linear" | "thin_plate_spline" | "cubic" |
                              # "quintic" | "multiquadric" |
                              # "inverse_multiquadric" | "gaussian" | …
                              # "multiquadric"/"inverse_multiquadric"/
                              # "gaussian" additionally require RBF_EPSILON.
RBF_EPSILON = None           # shape parameter; required for the kernels
                              # noted above, unused by "linear"/
                              # "thin_plate_spline"/"cubic"/"quintic".
RBF_SMOOTHING = 0.0          # 0 = exact interpolation; > 0 = smoothing
RBF_NEIGHBORS = 26           # use only the N nearest source points per
                              # query point (fast, local); None = exact
                              # global RBF using every source point.
RBF_DEGREE = None            # polynomial term degree; None = kernel default

# --- Kriging settings (INTERP_METHOD == "kriging") ---
KRIGING_VARIOGRAM_MODEL = "spherical"   # "linear" | "power" | "gaussian" |
                                          # "spherical" | "exponential" | ...
KRIGING_NLAGS = 6             # number of bins used when fitting the variogram
KRIGING_WEIGHT = True         # weight closer lags more when fitting the variogram
KRIGING_N_CLOSEST_POINTS = 50 # use only the N nearest source points per query
                              # point (fast, local); requires KRIGING_BACKEND
                              # "loop" or "C" — "vectorized" always ignores it
                              # (always global).
KRIGING_BACKEND = "loop"      # "vectorized" | "loop" | "C" (falls back to
                              # "loop" automatically if "C" isn't supported)
KRIGING_MAX_POINTS = 6000     # pykrige's variogram estimation is O(n^2) in
                              # the source point count; native point clouds
                              # larger than this are randomly subsampled
                              # (KRIGING_RANDOM_SEED) down to this many
                              # points BEFORE kriging. None = never subsample.
# KRIGING_MAX_POINTS = 10000   # TACNA
KRIGING_RANDOM_SEED = 42

# --- IDW settings (INTERP_METHOD == "idw") ---
IDW_POWER = 2.0                # distance exponent; higher = more local
IDW_NEIGHBORS = 12             # use only the N nearest source points per
                              # query point (scipy.spatial.cKDTree); None
                              # = use every source point (slow for large
                              # native point clouds, e.g. the full
                              # modem_submesh_points.nc table).

# --- Mask grid points outside each variable's own data footprint ---
# All three interpolation methods extrapolate smoothly forever past the
# convex hull of their source points; MASK_TO_CONVEX_HULL clips that
# back to NaN outside each variable's own 3-D convex hull, so nothing
# downstream is driven by unconstrained extrapolation in corners only
# some variables actually cover. Uses scipy.spatial.Delaunay per
# variable — can be slow for very large point clouds; set False to skip.
MASK_TO_CONVEX_HULL = True

# --- Region of interest (RoI) mask ---
# An explicit rectangular box, independent of GRID_EASTING_KM/
# GRID_NORTHING_KM (which set the *grid extent/resolution*, TARGET_GRID
# == "joint" only) and of MASK_TO_CONVEX_HULL (which is per-variable,
# driven by each variable's own data footprint). This is a single, joint
# box applied identically to every variable after interpolation: any
# target-grid cell whose (easting, northing) falls outside the box is set
# to NaN, regardless of depth. Given as the box's four corners, in order
# (closed automatically — do not repeat the first point at the end), so a
# rotated (non-axis-aligned) box works too. Set APPLY_ROI_MASK = False to
# disable.
APPLY_ROI_MASK = False
ROI_VERTICES_KM = [
    (330.0, 7990.0),
    (430.0, 7990.0),
    (430.0, 8080.0),
    (330.0, 8080.0),
]
# Optional depth restriction (km, positive down) — applied together with
# the horizontal box above; either bound may be None to leave it open.
ROI_DEPTH_MIN_KM = None
ROI_DEPTH_MAX_KM = None

# --- Gradient of the interpolated field(s) (optional) ---
# If True, compute the spatial gradient of each GRADIENT_VARS entry on
# the SAME grid as the interpolated field itself, and store it in
# INTERP_FILE alongside it (see "Gradient" in the module docstring).
COMPUTE_GRADIENT = True

# Which already-interpolated variables to differentiate. None = every
# entry in active_vars (i.e. everything just interpolated); must be a
# subset of that.
GRADIENT_VARS = None

# "spline" — fit a 1-D spline (scipy.interpolate.UnivariateSpline) along
#   each grid axis through the already-interpolated values (holding the
#   other axes fixed) and differentiate it analytically. Smoother than a
#   raw finite-difference gradient, and a natural fit here since the
#   field being differentiated is itself already the output of an
#   interpolation step rather than raw noisy data.
# "finite_difference" — plain numpy.gradient along each axis; cheaper,
#   no extra fitting, but less smooth and more sensitive to any residual
#   small-scale wiggles in the interpolated field.
GRADIENT_METHOD = "spline"
GRADIENT_SPLINE_ORDER = 3          # k for UnivariateSpline (3 = cubic)
GRADIENT_SPLINE_SMOOTHING = 0.0    # s for UnivariateSpline; 0 = spline
                                    # passes exactly through every finite
                                    # point on that line (pure
                                    # differentiating spline, no extra
                                    # smoothing beyond what interpolation
                                    # already did); raise for a smoother
                                    # (less exact) fit if the interpolated
                                    # field itself is noisy.
GRADIENT_MIN_POINTS = None         # minimum finite points required along
                                    # a line to fit a spline to it; None =
                                    # GRADIENT_SPLINE_ORDER + 1 (the
                                    # minimum scipy itself requires).
                                    # Lines with fewer finite points come
                                    # back all-NaN, same as a fully-masked
                                    # line.

# --- Output ---
# None = auto ("{SITE_PREFIX}_interp_<INTERP_METHOD>.nc" in NC_DIR), so a "rbf"
# and a "kriging" (and an "idw") run in the same directory never
# overwrite each other. Set explicitly to pin down a specific filename
# regardless of INTERP_METHOD.
OUTPUT_FILE = None

# =====================================================================
# END USER SETTINGS
# =====================================================================


def ncpath(name):
    """Join a bare NetCDF filename onto NC_DIR."""
    return os.path.join(NC_DIR, name)


def safe_to_netcdf(obj, path):
    """
    Write a Dataset/DataArray to NetCDF, overwriting any existing file at
    `path` even if it's read-only — e.g. left over from an earlier run —
    which otherwise makes xarray's own to_netcdf() raise PermissionError
    instead of just overwriting it. Removes the stale file first (fixing
    its permissions first if needed), then writes normally.
    """
    p = Path(path)
    if p.exists():
        try:
            p.unlink()
        except PermissionError:
            os.chmod(p, 0o644)
            p.unlink()
    obj.to_netcdf(path)


# ------------------------------------------------------------------
# Resistivity <-> conductivity
# ------------------------------------------------------------------
def resistivity_to_conductivity(values, units):
    """
    Invert a resistivity field to conductivity, matching whichever
    transform precompute.py's Part A actually applied
    (OUTPUT_TRANSFORM = "LOG10" / "LOGE" / "LINEAR") — read from the
    source variable's own `units` attribute (e.g. "log10(Ohm.m)",
    "ln(Ohm.m)", "Ohm.m") rather than hard-coded here.

    log10(sigma) = -log10(rho); ln(sigma) = -ln(rho); sigma = 1/rho.
    """
    u = (units or "").lower()
    if "log10" in u:
        return -values, "log10(S/m)"
    if u.startswith("ln("):
        return -values, "ln(S/m)"
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = np.where(values != 0, 1.0 / values, np.nan)
    return cond.astype(np.float64), "S/m"


# ------------------------------------------------------------------
# Native point-cloud loaders
# ------------------------------------------------------------------
def load_modem_points(file, value_var):
    """
    Load one variable from precompute.py's modem_submesh_points.nc
    (Part A's flat, full-native-ModEM-resolution point table).

    Returns
    -------
    points : ndarray, shape (n, 3) — [easting_km, northing_km, depth_km]
    values : ndarray, shape (n,)
    units  : str — the source variable's own `units` attribute
    """
    path = ncpath(file)
    ds = xr.open_dataset(path)
    if value_var not in ds.data_vars:
        ds.close()
        raise KeyError(
            f"{path!r} has no variable {value_var!r} — if this is "
            f"'sensitivity', re-run precompute.py with "
            f"USE_SENSITIVITY = True first."
        )
    valid = ds["valid"].values.astype(bool) & np.isfinite(ds[value_var].values)
    points = np.column_stack([
        ds["easting"].values[valid],
        ds["northing"].values[valid],
        ds["depth"].values[valid],
    ]).astype(np.float64)
    values = ds[value_var].values[valid].astype(np.float64)
    units = ds[value_var].attrs.get("units", "")
    ds.close()
    return points, values, units


def load_seis_grid_points(file, var):
    """
    Load one variable from a precompute.py Part B output
    ({SITE_PREFIX}_vp.nc etc.) — a gridded (depth, row, col) cube with 2-D
    utm_easting/utm_northing aux coords (metres) — and flatten it into a
    point cloud, broadcasting the horizontal coords across every depth
    level.

    Returns
    -------
    points, values, units — same shapes/meaning as load_modem_points().
    """
    ds = xr.open_dataset(ncpath(file))
    da = ds[var]
    depth = ds["depth"].values
    e2d_km = ds["utm_easting"].values / 1e3
    n2d_km = ds["utm_northing"].values / 1e3
    nz, nrow, ncol = da.shape
    e3d = np.broadcast_to(e2d_km[None, :, :], (nz, nrow, ncol))
    n3d = np.broadcast_to(n2d_km[None, :, :], (nz, nrow, ncol))
    d3d = np.broadcast_to(depth[:, None, None], (nz, nrow, ncol))
    vals3d = da.values.astype(np.float64)
    valid = np.isfinite(vals3d)
    points = np.column_stack(
        [e3d[valid], n3d[valid], d3d[valid]]
    ).astype(np.float64)
    values = vals3d[valid]
    units = da.attrs.get("units", "")
    ds.close()
    return points, values, units


def load_variable_points(key):
    """
    Dispatch to the right native loader for VARIABLE_SOURCES[key], then
    apply any registered `derive` transform (e.g. conductivity from
    resistivity).
    """
    src = VARIABLE_SOURCES[key]
    if src["kind"] == "modem_points":
        points, values, units = load_modem_points(src["file"], src["value_var"])
    elif src["kind"] == "seis_grid":
        points, values, units = load_seis_grid_points(src["file"], src["var"])
    else:
        raise ValueError(f"Unknown VARIABLE_SOURCES kind {src['kind']!r} for {key!r}")

    if src.get("derive") == "conductivity_from_rho":
        values, units = resistivity_to_conductivity(values, units)

    return points, values, units


# ------------------------------------------------------------------
# Target grid
# ------------------------------------------------------------------
def build_joint_grid(active_vars, native_points):
    """
    Build a fresh, genuinely regular UTM-km grid (TARGET_GRID ==
    "joint") from GRID_EASTING_KM/GRID_NORTHING_KM/GRID_DEPTH_KM,
    auto-bounded (where min/max is None) to the tightest common overlap
    of every active variable's own point-cloud extent.

    Returns a dict with grid_points (m, 3) = [easting, northing, depth]
    plus the axes and output shape, in the (depth, northing, easting)
    dim order used everywhere else in this pipeline.
    """
    def _auto_bounds(axis_index):
        lo = max(native_points[k][:, axis_index].min() for k in active_vars)
        hi = min(native_points[k][:, axis_index].max() for k in active_vars)
        return float(lo), float(hi)

    def _grid_axis(spec, axis_index, name):
        auto_lo, auto_hi = _auto_bounds(axis_index)
        lo = spec["min"] if spec["min"] is not None else auto_lo
        hi = spec["max"] if spec["max"] is not None else auto_hi
        if hi <= lo:
            raise RuntimeError(
                f"Joint grid {name} range is empty ({lo} .. {hi}) — the "
                f"selected INTERP_VARS don't overlap along this axis. "
                f"Check GRID_{name.upper()}_KM or INTERP_VARS."
            )
        axis = np.arange(lo, hi + spec["step"] / 2.0, spec["step"])
        print(
            f"  {name}: {lo:.2f} .. {hi:.2f} km, step {spec['step']} km "
            f"({len(axis)} points)"
        )
        return axis

    e_axis = _grid_axis(GRID_EASTING_KM, 0, "easting")
    n_axis = _grid_axis(GRID_NORTHING_KM, 1, "northing")
    d_axis = _grid_axis(GRID_DEPTH_KM, 2, "depth")

    Egrid, Ngrid, Dgrid = np.meshgrid(e_axis, n_axis, d_axis, indexing="ij")  # (ne, nn, nd)
    grid_points = np.column_stack([Egrid.ravel(), Ngrid.ravel(), Dgrid.ravel()])
    grid_shape = (len(d_axis), len(n_axis), len(e_axis))  # (depth, northing, easting)

    return dict(
        mode="joint", dim_names=("depth", "northing", "easting"),
        coords={"depth": d_axis, "northing": n_axis, "easting": e_axis},
        grid_points=grid_points, grid_shape=grid_shape,
        # marker used by the reshape step below: (ne, nn, nd) -> transpose
        # to (nd, nn, ne), matching Egrid/Ngrid/Dgrid's own shape.
        _reshape_from=Egrid.shape, _transpose=(2, 1, 0),
    )


def build_seismic_target_grid(ref_key):
    """
    Reuse VARIABLE_SOURCES[ref_key]'s own native (depth, row, col) grid
    as the interpolation target (TARGET_GRID == "seismic"), instead of
    building a new one. "row"/"col" are read from that source's own
    DataArray dims — never assumed to be "lat"/"lon" — so this works
    regardless of what precompute.py's Part B happens to call them.
    """
    src = VARIABLE_SOURCES[ref_key]
    if src.get("kind") != "seis_grid":
        raise ValueError(
            f"SEISMIC_MESH_VAR={ref_key!r} must be a 'seis_grid' "
            f"VARIABLE_SOURCES entry, got kind={src.get('kind')!r}."
        )
    ds = xr.open_dataset(ncpath(src["file"]))
    da = ds[src["var"]]
    dim_depth, dim_row, dim_col = da.dims
    depth = ds[dim_depth].values
    row_coord = ds[dim_row].values
    col_coord = ds[dim_col].values
    e2d_km = ds["utm_easting"].values / 1e3
    n2d_km = ds["utm_northing"].values / 1e3
    ds.close()

    nz, nrow, ncol = len(depth), len(row_coord), len(col_coord)
    e3d = np.broadcast_to(e2d_km[None, :, :], (nz, nrow, ncol))
    n3d = np.broadcast_to(n2d_km[None, :, :], (nz, nrow, ncol))
    d3d = np.broadcast_to(depth[:, None, None], (nz, nrow, ncol))
    grid_points = np.column_stack([e3d.ravel(), n3d.ravel(), d3d.ravel()])
    grid_shape = (nz, nrow, ncol)  # already (depth, row, col) — no transpose needed

    print(
        f"  Reusing {ref_key!r}'s native grid ({src['file']}): "
        f"dims {da.dims}, shape {grid_shape}"
    )

    return dict(
        mode="seismic", dim_names=(dim_depth, dim_row, dim_col), ref_key=ref_key,
        coords={dim_depth: depth, dim_row: row_coord, dim_col: col_coord},
        utm_easting_km=e2d_km, utm_northing_km=n2d_km,
        grid_points=grid_points, grid_shape=grid_shape,
        _reshape_from=grid_shape, _transpose=None,
    )


# ------------------------------------------------------------------
# Interpolation methods
# ------------------------------------------------------------------
def rbf_interpolate_to_grid(points, values, grid_points):
    """Fit scipy's RBFInterpolator on (points, values) and evaluate it
    at grid_points. Uses the RBF_* settings above."""
    rbf = RBFInterpolator(
        points, values,
        kernel=RBF_KERNEL, epsilon=RBF_EPSILON, smoothing=RBF_SMOOTHING,
        neighbors=RBF_NEIGHBORS, degree=RBF_DEGREE,
    )
    return rbf(grid_points)


def subsample_for_kriging(points, values, max_points, seed, label):
    """
    Randomly subsample (points, values) down to max_points rows if they
    exceed it — pykrige's variogram estimation is O(n^2) in the source
    point count. No-op if max_points is None or already small enough.
    """
    n = len(points)
    if max_points is None or n <= max_points:
        return points, values
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    print(f"    Subsampled {label!r}: {max_points} / {n} points for kriging "
          f"(KRIGING_MAX_POINTS)")
    return points[idx], values[idx]


def kriging_interpolate_to_grid(points, values, grid_points):
    """
    Fit pykrige's OrdinaryKriging3D on (points, values) and evaluate it
    at grid_points. Uses the KRIGING_* settings above. Returns just the
    kriged values (drops the kriging-variance field pykrige also
    returns — not currently used downstream).
    """
    from pykrige.ok3d import OrdinaryKriging3D  # lazy import — only needed here

    backend = KRIGING_BACKEND
    exec_kwargs = {}
    if KRIGING_N_CLOSEST_POINTS is not None:
        if backend == "vectorized":
            print(
                "    WARNING: KRIGING_N_CLOSEST_POINTS is set but "
                "KRIGING_BACKEND='vectorized' ignores it (always global) — "
                "set KRIGING_BACKEND to 'loop' or 'C' for local kriging."
            )
        else:
            exec_kwargs["n_closest_points"] = KRIGING_N_CLOSEST_POINTS

    ok3d = OrdinaryKriging3D(
        points[:, 0], points[:, 1], points[:, 2], values,
        variogram_model=KRIGING_VARIOGRAM_MODEL, nlags=KRIGING_NLAGS,
        weight=KRIGING_WEIGHT, exact_values=True,
    )
    if backend == "C":
        try:
            kvalues, _sigma_sq = ok3d.execute(
                "points", grid_points[:, 0], grid_points[:, 1], grid_points[:, 2],
                backend="C", **exec_kwargs,
            )
        except Exception as exc:
            print(
                f"    WARNING: KRIGING_BACKEND='C' failed ({type(exc).__name__}: "
                f"{exc}) — retrying with backend='loop'."
            )
            kvalues, _sigma_sq = ok3d.execute(
                "points", grid_points[:, 0], grid_points[:, 1], grid_points[:, 2],
                backend="loop", **exec_kwargs,
            )
    else:
        kvalues, _sigma_sq = ok3d.execute(
            "points", grid_points[:, 0], grid_points[:, 1], grid_points[:, 2],
            backend=backend, **exec_kwargs,
        )
    return np.asarray(kvalues)


def idw_interpolate_to_grid(points, values, grid_points, power=2.0, neighbors=12):
    """
    Inverse-distance weighting: each grid point's value is a
    distance-weighted average of its `neighbors` nearest native points
    (scipy.spatial.cKDTree), weight = 1 / distance**power. Exact at
    source points (distance 0 gets full weight, handled explicitly
    rather than via a small-epsilon fudge). neighbors=None uses every
    native point (slow for large point clouds).
    """
    tree = cKDTree(points)
    k = len(points) if neighbors is None else min(neighbors, len(points))
    dist, idx = tree.query(grid_points, k=k)
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]

    exact = dist[:, 0] <= 1e-9
    dist_safe = np.where(dist <= 1e-9, np.inf, dist)  # zero out that neighbor's weight instead
    w = 1.0 / dist_safe ** power
    vals = values[idx]
    out = np.sum(w * vals, axis=1) / np.sum(w, axis=1)
    out[exact] = vals[exact, 0]
    return out


def nearest_interpolate_to_grid(points, values, grid_points):
    """
    Nearest-neighbor interpolation: each target-grid point simply takes
    the value of its single closest native point (scipy.spatial.cKDTree,
    k=1) — piecewise-constant (Voronoi-cell) output, no smoothing or
    shape/variogram fit of any kind.
    """
    tree = cKDTree(points)
    _dist, idx = tree.query(grid_points, k=1)
    return values[idx]


def interpolate_to_grid(points, values, grid_points, label):
    """Dispatch to the configured INTERP_METHOD."""
    if INTERP_METHOD == "rbf":
        print(f"\nRBF-interpolating {label!r} onto the target grid "
              f"(kernel={RBF_KERNEL!r}, neighbors={RBF_NEIGHBORS}) …")
        return rbf_interpolate_to_grid(points, values, grid_points), points
    elif INTERP_METHOD == "kriging":
        print(f"\nKriging {label!r} onto the target grid "
              f"(variogram={KRIGING_VARIOGRAM_MODEL!r}, "
              f"n_closest_points={KRIGING_N_CLOSEST_POINTS}) …")
        kpoints, kvalues = subsample_for_kriging(
            points, values, KRIGING_MAX_POINTS, KRIGING_RANDOM_SEED, label,
        )
        return kriging_interpolate_to_grid(kpoints, kvalues, grid_points), kpoints
    elif INTERP_METHOD == "idw":
        print(f"\nIDW-interpolating {label!r} onto the target grid "
              f"(power={IDW_POWER}, neighbors={IDW_NEIGHBORS}) …")
        return idw_interpolate_to_grid(
            points, values, grid_points, power=IDW_POWER, neighbors=IDW_NEIGHBORS,
        ), points
    elif INTERP_METHOD == "nearest":
        print(f"\nNearest-neighbor-interpolating {label!r} onto the target grid …")
        return nearest_interpolate_to_grid(points, values, grid_points), points
    else:
        raise ValueError(
            f"INTERP_METHOD must be 'rbf', 'kriging', 'idw', or 'nearest', got {INTERP_METHOD!r}."
        )


def outside_convex_hull(points, query_points):
    """Boolean mask, True where query_points fall OUTSIDE the 3-D convex
    hull of points — used to null out interpolation's unconstrained
    extrapolation beyond a variable's own data footprint."""
    hull = Delaunay(points)
    return hull.find_simplex(query_points) < 0


def outside_roi(query_points, vertices, depth_min, depth_max):
    """
    Boolean mask, True where query_points (n, 3) = [easting_km,
    northing_km, depth_km] fall OUTSIDE the RoI box — a single,
    variable-independent region applied identically to every variable's
    interpolated grid (see APPLY_ROI_MASK/ROI_VERTICES_KM above).
    """
    poly = MplPath(np.asarray(vertices, dtype=np.float64))
    outside = ~poly.contains_points(query_points[:, :2])
    if depth_min is not None:
        outside |= query_points[:, 2] < depth_min
    if depth_max is not None:
        outside |= query_points[:, 2] > depth_max
    return outside


# ------------------------------------------------------------------
# Gradient of the interpolated field(s)
# ------------------------------------------------------------------
def _derivative_along_axis(field, coord, axis, method=GRADIENT_METHOD,
                            order=GRADIENT_SPLINE_ORDER,
                            smoothing=GRADIENT_SPLINE_SMOOTHING,
                            min_points=GRADIENT_MIN_POINTS):
    """
    Partial derivative of `field` (any ndim) along `axis`, with respect
    to the 1-D physical coordinate `coord` (length == field.shape[axis]),
    holding every other axis fixed. Works with either an increasing or a
    decreasing coord (flips internally, un-flips the result — e.g. a
    descending latitude axis). NaN in `field` propagates: any line with
    fewer than `min_points` finite values comes back all-NaN; where a
    line does have enough finite points, the derivative is only ever
    evaluated at those finite locations (still NaN elsewhere on that
    same line), so the derivative field stays aligned with the input
    field's own masking.

    method="finite_difference" is a thin wrapper around numpy.gradient
    (fast, no extra dependency beyond numpy). method="spline" fits
    scipy.interpolate.UnivariateSpline(x, y, k=order, s=smoothing) per
    line and evaluates its analytic derivative — slower (one fit per
    line orthogonal to `axis`) but smoother, and the natural choice here
    since `field` is itself already the output of an interpolation step.
    """
    field = np.asarray(field, dtype=np.float64)
    coord = np.asarray(coord, dtype=np.float64)
    if min_points is None:
        min_points = order + 1

    flip = coord[0] > coord[-1]
    if flip:
        coord = coord[::-1]
        field = np.flip(field, axis=axis)

    if method == "finite_difference":
        with np.errstate(invalid="ignore"):
            deriv = np.gradient(field, coord, axis=axis, edge_order=1)
    elif method == "spline":
        from scipy.interpolate import UnivariateSpline

        moved = np.moveaxis(field, axis, -1)   # (..., n_along)
        out_shape = moved.shape
        n_along = out_shape[-1]
        flat = moved.reshape(-1, n_along)
        deriv_flat = np.full_like(flat, np.nan)

        for i in range(flat.shape[0]):
            line = flat[i]
            finite = np.isfinite(line)
            n_finite = int(finite.sum())
            if n_finite < min_points:
                continue
            x = coord[finite]
            y = line[finite]
            k = min(order, n_finite - 1)
            try:
                spl = UnivariateSpline(x, y, k=k, s=smoothing)
            except Exception:
                continue
            deriv_flat[i, finite] = spl.derivative()(x)

        deriv = np.moveaxis(deriv_flat.reshape(out_shape), -1, axis)
    else:
        raise ValueError(
            f"GRADIENT_METHOD must be 'spline' or 'finite_difference', got {method!r}."
        )

    if flip:
        deriv = np.flip(deriv, axis=axis)
    return deriv.astype(np.float32)


def compute_gradient_joint(field, e_axis, n_axis, d_axis):
    """
    Gradient of `field` (dims depth, northing, easting) on a genuinely
    regular UTM-km "joint" target grid — a plain per-axis partial
    derivative in each direction, since grid axes here ARE the physical
    easting/northing/depth axes.
    """
    grad_easting = _derivative_along_axis(field, e_axis, axis=2)
    grad_northing = _derivative_along_axis(field, n_axis, axis=1)
    grad_depth = _derivative_along_axis(field, d_axis, axis=0)
    return grad_easting, grad_northing, grad_depth


def compute_gradient_seismic(field, d_axis, row_coord, col_coord,
                              utm_easting_2d, utm_northing_2d):
    """
    Gradient of `field` (dims depth, row, col) on a reused "seismic"
    target grid, which is regular in (row, col) INDEX space but not in
    UTM space (row/col are e.g. lat/lon, not easting/northing). A raw
    row/col-axis derivative of `field` is therefore a derivative with
    respect to the wrong coordinates; converted to true
    easting/northing derivatives via the local 2x2 Jacobian of the
    (row, col) -> (easting, northing) mapping, itself differentiated the
    same way from the grid's own utm_easting_2d/utm_northing_2d:

        dV/drow = dV/dE * dE/drow + dV/dN * dN/drow
        dV/dcol = dV/dE * dE/dcol + dV/dN * dN/dcol

    solved for (dV/dE, dV/dN) at every (row, col) location (broadcast
    across depth, since the row/col -> UTM mapping here doesn't itself
    vary with depth — see build_seismic_target_grid()).
    """
    dV_drow = _derivative_along_axis(field, row_coord, axis=1)
    dV_dcol = _derivative_along_axis(field, col_coord, axis=2)

    a = _derivative_along_axis(utm_easting_2d, row_coord, axis=0)   # dE/drow, (row, col)
    c_ = _derivative_along_axis(utm_easting_2d, col_coord, axis=1)  # dE/dcol, (row, col)
    b = _derivative_along_axis(utm_northing_2d, row_coord, axis=0)  # dN/drow, (row, col)
    d = _derivative_along_axis(utm_northing_2d, col_coord, axis=1)  # dN/dcol, (row, col)

    det = a * d - b * c_  # det(J^T), (row, col)
    degenerate = np.broadcast_to(np.abs(det) < 1e-9, dV_drow.shape)

    with np.errstate(divide="ignore", invalid="ignore"):
        grad_easting = (d * dV_drow - b * dV_dcol) / det
        grad_northing = (-c_ * dV_drow + a * dV_dcol) / det
    grad_easting = np.where(degenerate, np.nan, grad_easting).astype(np.float32)
    grad_northing = np.where(degenerate, np.nan, grad_northing).astype(np.float32)

    grad_depth = _derivative_along_axis(field, d_axis, axis=0)
    return grad_easting, grad_northing, grad_depth


# ==================================================================
# Resolve active variables (apply USE_CONDUCTIVITY swap)
# ==================================================================
active_vars = list(INTERP_VARS)
if USE_CONDUCTIVITY:
    active_vars = ["cond" if v == "rho" else v for v in active_vars]
    if active_vars != list(INTERP_VARS):
        print(f"USE_CONDUCTIVITY=True — interpolating {active_vars} "
              f"instead of {INTERP_VARS}")

print(f"Interpolating: {active_vars}  (method={INTERP_METHOD!r}, target={TARGET_GRID!r})")

# ==================================================================
# Load native point clouds
# ==================================================================
native_points = {}
native_values = {}
resolved_units = {}
for key in active_vars:
    if key not in VARIABLE_SOURCES:
        raise KeyError(
            f"INTERP_VARS entry {key!r} has no matching VARIABLE_SOURCES "
            f"registration. Known keys: {list(VARIABLE_SOURCES)}"
        )
    src = VARIABLE_SOURCES[key]
    print(f"  Loading {key!r} ({src['kind']}) from {src['file']} …")
    pts, vals, units = load_variable_points(key)
    native_points[key] = pts
    native_values[key] = vals
    resolved_units[key] = units
    print(
        f"    {len(vals)} native points — "
        f"E [{pts[:, 0].min():.1f}, {pts[:, 0].max():.1f}], "
        f"N [{pts[:, 1].min():.1f}, {pts[:, 1].max():.1f}], "
        f"D [{pts[:, 2].min():.1f}, {pts[:, 2].max():.1f}] km"
    )

# ==================================================================
# Build the target grid
# ==================================================================
print(f"\nBuilding target grid (TARGET_GRID={TARGET_GRID!r}) …")
if TARGET_GRID == "joint":
    target = build_joint_grid(active_vars, native_points)
elif TARGET_GRID == "seismic":
    target = build_seismic_target_grid(SEISMIC_MESH_VAR)
else:
    raise ValueError(f"TARGET_GRID must be 'joint' or 'seismic', got {TARGET_GRID!r}.")

grid_points = target["grid_points"]
grid_shape = target["grid_shape"]
n_total = int(np.prod(grid_shape))
print(f"Target grid dims {target['dim_names']}, shape {grid_shape} "
      f"({n_total} cells total)")

if APPLY_ROI_MASK:
    roi_outside = outside_roi(grid_points, ROI_VERTICES_KM, ROI_DEPTH_MIN_KM, ROI_DEPTH_MAX_KM)
    print(
        f"\nRoI mask: {int(roi_outside.sum())} / {roi_outside.size} target-grid "
        f"points fall outside the RoI box and will be set to NaN "
        f"(applies identically to every variable)"
    )
else:
    roi_outside = None

# ==================================================================
# Interpolate each variable onto the target grid
# ==================================================================
interpolated = {}
for key in active_vars:
    interp_vals, used_points = interpolate_to_grid(
        native_points[key], native_values[key], grid_points, key,
    )
    if MASK_TO_CONVEX_HULL:
        outside = outside_convex_hull(used_points, grid_points)
        interp_vals = np.where(outside, np.nan, interp_vals)
        print(
            f"    Masked {int(outside.sum())} / {outside.size} target-grid "
            f"points outside {key!r}'s own convex hull"
        )
    if APPLY_ROI_MASK:
        interp_vals = np.where(roi_outside, np.nan, interp_vals)

    reshaped = interp_vals.reshape(target["_reshape_from"])
    if target["_transpose"] is not None:
        reshaped = np.transpose(reshaped, target["_transpose"])
    interpolated[key] = reshaped.astype(np.float32)

# ==================================================================
# Gradient of the interpolated field(s) (optional)
# ==================================================================
gradients = {}
if COMPUTE_GRADIENT:
    grad_vars = list(GRADIENT_VARS) if GRADIENT_VARS is not None else list(active_vars)
    missing = [v for v in grad_vars if v not in interpolated]
    if missing:
        raise KeyError(
            f"GRADIENT_VARS {missing} not among the just-interpolated "
            f"variables {active_vars} — GRADIENT_VARS must be a subset of "
            f"INTERP_VARS (after any USE_CONDUCTIVITY swap)."
        )
    print(
        f"\nComputing gradients ({GRADIENT_METHOD!r}) for: {grad_vars} "
        f"(this fits one spline per grid line per axis per variable if "
        f"GRADIENT_METHOD='spline' — can take a while on a large grid) …"
    )
    for key in grad_vars:
        field = interpolated[key].astype(np.float64)
        if target["mode"] == "joint":
            ge, gn, gd = compute_gradient_joint(
                field, target["coords"]["easting"], target["coords"]["northing"],
                target["coords"]["depth"],
            )
        else:
            dim_depth, dim_row, dim_col = target["dim_names"]
            ge, gn, gd = compute_gradient_seismic(
                field, target["coords"][dim_depth],
                target["coords"][dim_row], target["coords"][dim_col],
                target["utm_easting_km"], target["utm_northing_km"],
            )
        gmag = np.sqrt(ge.astype(np.float64) ** 2 + gn.astype(np.float64) ** 2
                        + gd.astype(np.float64) ** 2).astype(np.float32)
        gradients[f"{key}_grad_easting"] = ge
        gradients[f"{key}_grad_northing"] = gn
        gradients[f"{key}_grad_depth"] = gd
        gradients[f"{key}_grad_mag"] = gmag
        print(f"    {key!r}: done")

# ==================================================================
# Save
# ==================================================================
dim_names = target["dim_names"]
data_vars = {}
for key in active_vars:
    src = VARIABLE_SOURCES[key]
    data_vars[key] = (
        dim_names, interpolated[key],
        {"long_name": src["label"], "units": resolved_units[key] or ""},
    )
for grad_name, grad_field in gradients.items():
    base_key, comp = grad_name.rsplit("_grad_", 1)   # e.g. "rho_grad_easting" -> ("rho", "easting")
    base_units = resolved_units[base_key] or "1"
    base_label = VARIABLE_SOURCES[base_key]["label"]
    if comp == "mag":
        grad_units = f"({base_units})/km"
        long_name = f"3-D gradient magnitude of {base_label} ({GRADIENT_METHOD})"
    else:
        grad_units = f"({base_units})/km"
        long_name = f"Partial derivative of {base_label} with respect to {comp} ({GRADIENT_METHOD})"
    data_vars[grad_name] = (dim_names, grad_field, {"long_name": long_name, "units": grad_units})

coords = {name: (name, vals) for name, vals in target["coords"].items()}
if target["mode"] == "seismic":
    dim_row, dim_col = dim_names[1], dim_names[2]
    coords["utm_easting_km"] = ((dim_row, dim_col), target["utm_easting_km"])
    coords["utm_northing_km"] = ((dim_row, dim_col), target["utm_northing_km"])

attrs = {
    "description": (
        f"{', '.join(active_vars)} interpolated (method={INTERP_METHOD}) onto "
        f"a {'jointly-defined regular UTM-km' if target['mode'] == 'joint' else 'reused seismic-tomography native'} "
        f"grid — see interpolate.py."
    ),
    "target_grid_mode": target["mode"],
    "dim_names": ", ".join(dim_names),
    "interp_method": INTERP_METHOD,
    "interp_vars": ", ".join(active_vars),
    "use_conductivity": str(USE_CONDUCTIVITY),
    "mask_to_convex_hull": str(MASK_TO_CONVEX_HULL),
    "roi_applied": str(APPLY_ROI_MASK),
    "roi_vertices_km": str(ROI_VERTICES_KM) if APPLY_ROI_MASK else "",
    "roi_depth_range_km": f"{ROI_DEPTH_MIN_KM}, {ROI_DEPTH_MAX_KM}" if APPLY_ROI_MASK else "",
}
if target["mode"] == "seismic":
    attrs["seismic_mesh_var"] = target["ref_key"]
attrs["gradient_computed"] = str(COMPUTE_GRADIENT)
if COMPUTE_GRADIENT:
    attrs.update(
        gradient_vars=", ".join(grad_vars),
        gradient_method=GRADIENT_METHOD,
        gradient_spline_order=str(GRADIENT_SPLINE_ORDER),
        gradient_spline_smoothing=GRADIENT_SPLINE_SMOOTHING,
    )
if INTERP_METHOD == "rbf":
    attrs.update(rbf_kernel=RBF_KERNEL, rbf_neighbors=str(RBF_NEIGHBORS),
                 rbf_smoothing=RBF_SMOOTHING)
elif INTERP_METHOD == "kriging":
    attrs.update(kriging_variogram_model=KRIGING_VARIOGRAM_MODEL,
                 kriging_n_closest_points=str(KRIGING_N_CLOSEST_POINTS),
                 kriging_max_points=str(KRIGING_MAX_POINTS))
elif INTERP_METHOD == "idw":
    attrs.update(idw_power=IDW_POWER, idw_neighbors=str(IDW_NEIGHBORS))

out_ds = xr.Dataset(data_vars, coords=coords, attrs=attrs)

output_file = OUTPUT_FILE or f"{SITE_PREFIX}_interp_{INTERP_METHOD}.nc"
out_nc = ncpath(output_file)
safe_to_netcdf(out_ds, out_nc)
print(f"\nSaved: {out_nc}")
print("\nDone.")
