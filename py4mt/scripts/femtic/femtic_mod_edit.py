#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_mod_edit.py — Apply arithmetic operations to a FEMTIC resistivity model
in log10 space, then rewrite the model file.

Workflow
--------
    (1) Read resistivity_block_iterX.dat → free log10(ρ) vector.
    (2) Apply one of the available operations (see OPERATION below).
    (3) Write the modified model to a new (or the same) file.

Available operations (OPERATION key)
--------------------------------------
    "fill"          Set every free region to a single constant log10(ρ) value
                    (OP_FILL_VALUE).  Simplest possible initialisation.
    "mean"          Replace every free region with the mean   of log10(ρ).
    "wmean"         Replace every free region with the inverse-volume-weighted
                    mean of log10(ρ).  Requires MESH_FILE.  Large coarse cells
                    (deep background) contribute less than small fine cells
                    (near-surface target zone).
    "median"        Replace every free region with the median of log10(ρ).
    "clip"          Clamp log10(ρ) to [OP_CLIP_MIN, OP_CLIP_MAX].
    "shift"         Add a scalar offset: log10(ρ) += OP_SHIFT_VALUE.
    "standardise"   Map to zero-mean / unit-variance in log10 space.
    "smooth"        Spatial smoothing in log10 space.  Requires MESH_FILE.
                    The kernel is selected by OP_SMOOTH_MODE:
                      "physical"    Gaussian with global σ = OP_SMOOTH_SIGMA
                                    (original behaviour).
                      "knn_uniform" Flat average over the K nearest neighbours;
                                    reach adapts to local mesh density.
                      "knn_gauss"   Per-region Gaussian over K neighbours with
                                    σ_i = OP_SMOOTH_KNN_SIGMA_FRAC × d_{i,K}.
                    Memory is O(n_free × K) — fixed and predictable.
    "ellipsoid"     Modify free regions whose centroid falls inside one or
                    more rotated ellipsoids.  Requires MESH_FILE.
                    Bodies are defined by the list OP_ELLIPSOID_BODIES; each
                    entry is a dict with keys: mode, value, center, axes,
                    angles (ZYX degrees).  Bodies are applied in order; later
                    bodies overwrite earlier ones where masks overlap.
                    Mask computed via fem.ellipsoid_mask().
    "brick"         Same as "ellipsoid" but with a rotated rectangular prism
                    (box) geometry.  Bodies are defined by OP_BRICK_BODIES;
                    each entry has the same keys as an ellipsoid body but
                    axes = [a, b, c] are half-extents (metres), not semi-axes.
                    The rotation uses the same ZYX convention.
                    Mask computed via fem.brick_mask().
    "null"          No-op: the input model is passed through unchanged and
                    no output file is written.  Use this to inspect the
                    input model with the slice plotter without modifying
                    anything.

Note: multiplicative scaling (×factor) is not offered because multiplying
log10(ρ) by a constant has no clean physical interpretation.

New operations are easy to add — extend the ``_OPERATIONS`` dict below.

Plotting
--------
    When PLOT = True the script calls ``plot_model_slices`` after writing,
    which produces a single Matplotlib figure with a configurable list of
    axis-parallel slices:
      - horizontal map slices  (z = const)
      - NS curtain slices      (y = const, displayed as x vs z)
      - EW curtain slices      (x = const, displayed as y vs z)
    Air is transparent (NaN → no colour); ocean/lake cells are rendered in
    a flat colour (OCEAN_COLOR).  Axis limits, colormap, and colour range
    are controlled by PLOT_* parameters.

Provenance
----------
    2026-04-30  vrath / Claude Sonnet 4.6   Created, modelled on
                femtic_gst_prep.py.  Uses fem.read_model / fem.insert_model
                for I/O so that fixed regions (air, ocean, flag==1) are handled
                transparently and the template metadata (bounds, flag, n) is
                preserved unchanged.
    2026-04-30  vrath / Claude Sonnet 4.6   Added "smooth" operation:
                Gaussian-weighted region-centroid smoothing in log10 space.
    2026-04-30  vrath / Claude Sonnet 4.6   Added "fill" operation:
                constant-value initialisation for all free regions.
    2026-04-30  vrath / Claude Sonnet 4.6   Added "ellipsoid" operation:
                local replace/add inside a rotated ellipsoid defined by
                centroid, semi-axes, and ZYX rotation angles.
    2026-05-03  vrath / Claude Sonnet 4.6   Added "brick" operation:
                rotated rectangular prism (box) with ZYX rotation, same
                mode/value/center/angles convention as ellipsoid.
                Both operations now accept a list of bodies (OP_ELLIPSOID_BODIES,
                OP_BRICK_BODIES); bodies are applied sequentially, later ones
                overwrite earlier where masks overlap.
    2026-05-03  vrath / Claude Sonnet 4.6   Added plotting section:
                axis-parallel map/curtain slices via femtic_viz;
                air transparent, ocean flat colour, shared colormap/clim.
    2026-04-30  vrath / Claude Sonnet 4.6   Added "wmean" operation:
                inverse-volume-weighted mean in log10 space.
                Refactored _build_region_centroids into _build_region_geometry
                (returns centroids + volumes in one element loop).
    2026-05-04  vrath / Claude Sonnet 4.6   Added "null" operation: no-op
                pass-through; skips write step and plots MODEL_IN directly —
                use to inspect the input model without any modification.
    2026-05-13  vrath / Claude Sonnet 4.6   Harmonised plotting config block
                with femtic_mod_plot.py: unified variable names, comments,
                and PLOT_FILE rename guard.
    2026-05-27  vrath / Claude Sonnet 4.6 (Anthropic)
                Removed local plot_model_slices / _plot_slice_panel /
                _slice_geometry / _intersect_tet_plane_general /
                _axis_slice_params / _plane_basis / _strike_dip_to_normal
                helpers (~430 lines); step (4) now delegates to
                fviz.plot_model_slices for consistency across all scripts.
    2026-06-08  vrath / Claude Sonnet 4.6 (Anthropic)
                Removed broken SITE_NUMBER / OBSERVE_FILE fallback branch
                (OBSERVE_FILE was never defined); SITE_DAT is the sole site
                source. obs_coords_only kwarg hardened to False.
    2026-06-08  vrath / Claude Sonnet 4.6 (Anthropic)
                Added mesh-adaptive smoothing modes to "smooth" operation via
                new OP_SMOOTH_MODE config variable: "physical" preserves the
                original global-σ Gaussian; "knn_uniform" averages the K
                nearest neighbours with flat weights; "knn_gauss" uses a
                per-region Gaussian with σ_i = OP_SMOOTH_KNN_SIGMA_FRAC ×
                d_{i,K} (distance to K-th neighbour).  Both knn variants
                require SciPy.  New config vars: OP_SMOOTH_MODE,
                OP_SMOOTH_KNN_SIGMA_FRAC.
    2026-06-08  vrath / Claude Sonnet 4.6 (Anthropic)
                Refactored geometry helpers into femtic.py: removed
                _tet_volumes, _build_region_geometry, _rotation_matrix_zyx,
                _local_coords, _ellipsoid_mask, _brick_mask.  Call sites now
                delegate to fem.build_region_geometry(), fem.ellipsoid_mask(),
                fem.brick_mask().  _smooth_body_boundary and _apply_bodies
                remain local (workflow machinery, reference OUT global).

@author: vrath
"""

import os
import sys
import inspect
import numpy as np

# ---------------------------------------------------------------------------
# Py4MTX-specific settings and imports
# ---------------------------------------------------------------------------
PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [PY4MTX_ROOT + "/py4mt/modules/", PY4MTX_ROOT + "/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)

from version import versionstrg
import util as utl
import femtic as fem

try:
    import femtic_viz as fviz
except ImportError:
    fviz = None

version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())
titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng + "\n\n")

# ===========================================================================
# Configuration
# ===========================================================================

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORK_DIR = r"/home/vrath/Py4MTX/py4mt/data/rto/misti/ensemble/templates/"
WORK_DIR = r"/home/vrath/Py4MTX/py4mt/data/rto/misti/PrepRun8/"
#: Template / source resistivity block (also used as format template by
#: insert_model to preserve header, bounds and flag columns).
# MODEL_IN  = WORK_DIR + "resistivity_block_iter0.dat"
MODEL_IN  =WORK_DIR + "resistivity_block_iter15.dat"
#: Mesh file — required for "smooth" and "ellipsoid"; ignored otherwise.
MESH_FILE = WORK_DIR + "mesh.dat"

#: Output file.  Set to MODEL_IN to overwrite in-place (be careful!).
MODEL_OUT = WORK_DIR + "reference_i15_smooth3000.dat"

# ---------------------------------------------------------------------------
# Ocean / fixed-region handling
# ---------------------------------------------------------------------------
#: None → auto-infer from region 1 heuristic (rho ≤ 1 Ω·m AND flag==1).
#: True / False → force ocean-present / ocean-absent.
OCEAN = None

AIR_RHO   = 1.0e9   # Ω·m written for region 0 (air)
OCEAN_RHO = 0.25    # Ω·m written for region 1 when treated as ocean

# ---------------------------------------------------------------------------
# Operation to apply
# ---------------------------------------------------------------------------
#: One of: "fill" | "mean" | "wmean" | "median" | "clip" | "shift"
#:         | "standardise" | "smooth" | "ellipsoid" | "brick" | "null"
OPERATION = "smooth"
# OPERATION = "wmean"
# OPERATION = "median"
# OPERATION = "mean"
#OPERATION = "ellipsoid"

if OPERATION != "null":
    MODEL_OUT = MODEL_OUT.replace("edited", OPERATION)
print(MODEL_OUT)

# Parameters used by specific operations (ignored when not applicable):
OP_FILL_VALUE   = 2.0    # log10(100 Ω·m) — used by "fill"
OP_CLIP_MIN     = 0.0    # log10(1 Ω·m)   — used by "clip"
OP_CLIP_MAX     = 4.0    # log10(10 kΩ·m) — used by "clip"
OP_SHIFT_VALUE  = 0.5    # added to every log10(ρ) — used by "shift"

#: Smoothing kernel mode — used by "smooth".
#:
#:   "physical"    Gaussian with a fixed global length scale σ = OP_SMOOTH_SIGMA.
#:                 Weight: w_ij = exp(−‖c_i−c_j‖² / 2σ²).
#:                 Reach is the same everywhere in physical space, so fine-mesh
#:                 zones receive many neighbours inside the kernel while coarse
#:                 zones receive few.  This is the original behaviour.
#:
#:   "knn_uniform" Flat average over the K nearest neighbours.
#:                 Weight: w_ij = 1/K for all j in KNN(i).
#:                 Reach adapts to local mesh density: fine regions are smoothed
#:                 over a small physical footprint, coarse regions over a large
#:                 one — the same number of cells is always blended.
#:
#:   "knn_gauss"   Gaussian over the K nearest neighbours, but with a per-region
#:                 σ_i = OP_SMOOTH_KNN_SIGMA_FRAC × d_{i,K} where d_{i,K} is the
#:                 distance to region i's K-th nearest neighbour.
#:                 Weight: w_ij = exp(−‖c_i−c_j‖² / 2σ_i²).
#:                 Like "knn_uniform" the reach is mesh-adaptive, but the kernel
#:                 decays smoothly within the neighbourhood rather than cutting
#:                 off abruptly.
#:
#: "knn_uniform" and "knn_gauss" require SciPy; they raise RuntimeError if it
#: is absent.  "physical" falls back to a chunked dense path when SciPy is
#: unavailable.
OP_SMOOTH_MODE    = "physical"   # "physical" | "knn_uniform" | "knn_gauss"

#: Gaussian smoothing length σ in metres — used by mode "physical" only.
#: Controls the decay of the Gaussian weight with distance.  A good first
#: guess is 1–2× the typical element edge length in the target depth range.
OP_SMOOTH_SIGMA   = 3000.0  # metres

#: Number of nearest neighbours — used by all three modes.
#:   "physical"    : neighbours beyond K get effectively zero weight if their
#:                   distance >> σ; K is a memory/speed cap (start: 50–200).
#:   "knn_uniform" : exactly K neighbours are averaged.
#:   "knn_gauss"   : exactly K neighbours enter the per-region Gaussian.
#: Memory scales as n_free × K × 8 bytes (predictable, no variable-length
#: lists).
OP_SMOOTH_K       = 100     # nearest neighbours

#: Per-region σ fraction — used by mode "knn_gauss" only.
#: σ_i = OP_SMOOTH_KNN_SIGMA_FRAC × d_{i,K}  (distance to the K-th neighbour).
#: Fraction = 0.5  → weights ≈ 0.13 at the neighbourhood edge (steeper decay).
#: Fraction = 1.0  → weights ≈ 0.61 at the edge (gentler, closer to uniform).
#: Values > 1 approach uniform weighting.
OP_SMOOTH_KNN_SIGMA_FRAC = 0.5

#: Maximum RAM (GiB) for the chunked dense fallback — used by "smooth" mode
#: "physical" when SciPy is unavailable.
OP_SMOOTH_MAX_GB  = 4.0     # GiB

# ---------------------------------------------------------------------------
# Ellipsoid bodies — used by "ellipsoid" only
# ---------------------------------------------------------------------------
#: List of ellipsoid body dicts, applied in order (later bodies win on overlap).
#: Each dict must contain:
#:   mode            : "replace" | "add"
#:   value           : float  log10(Ω·m) — absolute if replace, signed offset if add
#:   center          : [x, y, z]  metres, z positive-down
#:   axes            : [a, b, c]  semi-axes in metres, all > 0
#:   angles          : [α, β, γ]  ZYX rotation in degrees (yaw, pitch, roll)
#:   boundary_smooth : optional dict — smooth the body boundary after insertion
#:     sigma  : Gaussian length scale in metres (blend width per pass)
#:     passes : number of smoothing passes (transition zone depth ≈ passes × σ)
OP_ELLIPSOID_BODIES = [
    # dict(mode="replace", value=0.0,
    #      center=[0.0, 0.0, 5000.0],
    #      axes=[10000.0, 10000.0, 5000.0],
    #      angles=[0.0, 0.0, 0.0]),
    # With boundary smoothing:
    dict(mode="replace", value=0.0,
         center=[0.0, 0.0, 5000.0],
         axes=[10000.0, 10000.0, 5000.0],
         angles=[0.0, 0.0, 0.0],
         boundary_smooth=dict(sigma=1000., passes=3)),
]

# ---------------------------------------------------------------------------
# Brick bodies — used by "brick" only
# ---------------------------------------------------------------------------
#: List of brick (rotated rectangular prism) body dicts, applied in order.
#: Same keys as ellipsoid bodies; axes = [a, b, c] are half-extents (metres).
#: The box test in the rotated local frame is |x'| ≤ a, |y'| ≤ b, |z'| ≤ c.
#: Optional boundary_smooth key: dict(sigma=…, passes=…) — see ellipsoid docs.
OP_BRICK_BODIES = [
    dict(mode="replace", value=0.0,
         center=[0.0, 0.0, 5000.0],
         axes=[10000.0, 8000.0, 4000.0],
         angles=[0.0, 0.0, 0.0]),
    # With boundary smoothing:
    # dict(mode="add", value=1.0,
    #      center=[0.0, 0.0, 15000.0],
    #      axes=[5000.0, 5000.0, 5000.0],
    #      angles=[45.0, 0.0, 0.0],
    #      boundary_smooth=dict(sigma=2000., passes=2)),
]

# ---------------------------------------------------------------------------
# Verbose output
# ---------------------------------------------------------------------------
OUT = True

# ---------------------------------------------------------------------------
# Plotting — requires femtic_viz and Matplotlib
# ---------------------------------------------------------------------------
#: Set True to plot slices of the *output* model after writing.
PLOT = True

#: Output file path — None → interactive show().
PLOT_FILE = WORK_DIR + "resistivity_block_edited.pdf"
if OPERATION != "null":
    PLOT_FILE = PLOT_FILE.replace("edited", OPERATION)
print(PLOT_FILE)

#: Figure DPI for saved file.
PLOT_DPI = 600

#: Matplotlib colormap name.
PLOT_CMAP = "turbo_r"

#: Colour limits [log10(ρ_min), log10(ρ_max)] — None = auto.
PLOT_CLIM = [0.0, 4.0]      # log10(Ω·m)

#: Flat colour for ocean / lake cells.  None → use colormap.
PLOT_OCEAN_COLOR = "lightgrey"

#: Axes facecolor for air / background.  None = figure default.
PLOT_AIR_BGCOLOR = None

#: Slice specification — a list of dicts, one per panel (left to right).
#:
#: Slices use exact tetrahedron-plane intersection — no selection slab,
#: no dw, no dz.  The plane is infinitely thin; every tetrahedron that
#: straddles it contributes an exact triangle or quadrilateral polygon.
#:
#: Each dict must contain:
#:   kind   : "map"   — horizontal slice at z = z0
#:            "ns"    — N-S vertical section at x = x0   (y vs depth)
#:            "ew"    — E-W vertical section at y = y0   (x vs depth)
#:            "plane" — arbitrary plane by strike / dip / point
#:   z0     : (map   only)  depth in metres
#:   x0     : (ns    only)  easting — plain float = model-local metres
#:   y0     : (ew    only)  northing — plain float = model-local metres
#:   point  : (plane only)  [x, y, z] any point on the plane (metres)
#:   strike : (plane only)  clockwise from North, degrees (0=N, 90=E)
#:   dip    : (plane only)  downward inclination from horizontal, degrees
#:   xlim   : [xmin, xmax] — easting or along-strike axis limit
#:   ylim   : [ymin, ymax] — northing or down-dip axis limit
#:   zlim   : [zmin, zmax] — depth axis limit (ns/ew panels)
#:   title  : optional string override
#:
#: Per-panel xlim/ylim/zlim override the global PLOT_XLIM/PLOT_YLIM/PLOT_ZLIM.
PLOT_SLICES = [
    dict(kind="map",   z0=5000.0),
    dict(kind="map",   z0=15000.0),
    dict(kind="ns",    x0=0.0),
    dict(kind="ew",    y0=0.0),
    # dict(kind="plane", point=[0., 0., 5000.], strike=45., dip=60.),
]

#: Global axis limits in model-local metres — used for panels that do not
#: specify their own.  None → auto (inferred from data extent).
PLOT_XLIM = [-20000., 20000.]   # [xmin, xmax] metres — easting
PLOT_YLIM = [-20000., 20000.]   # [ymin, ymax] metres — northing
PLOT_ZLIM = [  -6000., 15000.]  # [zmin, zmax] metres — depth (z positive-down)

#: True → depth axis in km; False → metres.
DEPTH_KM = True

#: True → horizontal axes in km (model/utm modes); False → metres.
HORIZ_KM = True

#: Equal aspect ratio on map and curtain panels (model/utm coords only).
PLOT_EQUAL_ASPECT = True

#: Panel height in cm.  Width auto-computed from axis limits when PLOT_EQUAL_ASPECT.
PLOT_PANEL_HEIGHT = 16.0   # cm

#: Grid layout.  None → 1 row / len(PLOT_SLICES) columns.
PLOT_NROWS = None
PLOT_NCOLS = None

# ---------------------------------------------------------------------------
# Geographic / UTM origin of the mesh centre
# ---------------------------------------------------------------------------
#: Fallback values.  When ORIGIN_METHOD is "box" or "average" these are
#: overwritten at runtime from site.dat and may be left as None.
UTM_ORIGIN_LAT = None
UTM_ORIGIN_LON = None
UTM_ORIGIN_E   = None
UTM_ORIGIN_N   = None
UTM_ZONE_OVERRIDE = None

#: Method to estimate origin from site.dat: None | "box" | "average".
ORIGIN_METHOD = "box"

# ---------------------------------------------------------------------------
# Display coordinate system
# ---------------------------------------------------------------------------
DISPLAY_COORDS = "model"   # "model" | "utm" | "latlon"

# ---------------------------------------------------------------------------
# Site overlay
# ---------------------------------------------------------------------------
SITE_DAT    = WORK_DIR + "site.dat"   # set to None to disable
SITE_NAMES  = None                    # None = all sites


PLOT_SITES_MAPS   = True
PLOT_SITES_SLICES = False
PROJECTION_DIST   = 5000.   # m

SITE_MARKER        = dict(marker="v", color="black", ms=8, zorder=10, label=None)
SITE_MARKER_SLICES = None

MAP_MARKERS = []



# ===========================================================================
# Helper: operation registry
# ===========================================================================

def _op_fill(m: np.ndarray) -> np.ndarray:
    """Set every free region to a constant log10(ρ) value (OP_FILL_VALUE).

    Air (region 0), ocean (region 1 when active), and any region with
    flag == 1 are *not* part of the free vector and are therefore untouched
    regardless of this operation — fem.insert_model enforces their canonical
    values (AIR_RHO, OCEAN_RHO) independently.
    """
    return np.full_like(m, float(OP_FILL_VALUE))


def _op_mean(m: np.ndarray) -> np.ndarray:
    """Replace every element with the mean log10(ρ) of free regions."""
    return np.full_like(m, np.mean(m))


def _op_median(m: np.ndarray) -> np.ndarray:
    """Replace every element with the median log10(ρ) of free regions."""
    return np.full_like(m, np.median(m))


def _op_clip(m: np.ndarray) -> np.ndarray:
    """Clamp log10(ρ) to [OP_CLIP_MIN, OP_CLIP_MAX]."""
    return np.clip(m, OP_CLIP_MIN, OP_CLIP_MAX)


def _op_shift(m: np.ndarray) -> np.ndarray:
    """Add a scalar offset in log10 space."""
    return m + OP_SHIFT_VALUE


def _op_standardise(m: np.ndarray) -> np.ndarray:
    """Map to zero-mean / unit-variance in log10 space.

    Note: the result may lie outside physical resistivity bounds.
    Combine with "clip" afterwards if needed.
    """
    mu  = np.mean(m)
    sig = np.std(m)
    if sig == 0.0:
        return np.zeros_like(m)
    return (m - mu) / sig


# Module-level context dicts — populated in the main block before dispatch.
# Defined here so operation functions can reference them as globals without
# triggering NameError or linter warnings.
_wmean_ctx:     dict = {}
_smooth_ctx:    dict = {}
_ellipsoid_ctx: dict = {}
_brick_ctx:     dict = {}


def _op_wmean(m: np.ndarray) -> np.ndarray:
    """Replace every free region with the inverse-volume-weighted mean log10(ρ).

    Regions are weighted by 1/V so that large coarse cells (deep background)
    contribute less than small fine cells (near-surface target zone).  This
    is usually a better homogeneous starting point than the arithmetic mean
    when the mesh has strongly varying cell sizes.

    Weight for region k:  w_k = 1 / V_k   (V_k = total volume of region k).

    Regions with V = 0 (no elements assigned) receive zero weight and do not
    contribute to the average; their output value is set to the weighted mean
    of the rest.  A warning is printed if any such region exists.
    """
    vol = _wmean_ctx["volumes"]    # (n_free,)  m³

    zero_vol = vol == 0.0
    if np.any(zero_vol):
        print(f"  wmean: WARNING — {int(zero_vol.sum())} free region(s) have V=0; excluded.")

    w = np.where(zero_vol, 0.0, 1.0 / np.where(zero_vol, 1.0, vol))
    w_sum = w.sum()
    if w_sum == 0.0:
        raise RuntimeError("wmean: all region volumes are zero — cannot compute weighted mean.")

    mean_val = float((w @ m) / w_sum)
    return np.full_like(m, mean_val)


def _op_smooth(m: np.ndarray) -> np.ndarray:
    """Mesh-adaptive or physical-distance smoothing of log10(ρ) across free regions.

    The kernel is selected by ``OP_SMOOTH_MODE``:

    "physical"
        Classic Gaussian with a fixed global length scale σ = OP_SMOOTH_SIGMA:

            w_ij = exp(−‖c_i − c_j‖² / 2σ²)

        Reach is the same everywhere in physical space, so fine-mesh zones
        receive many in-kernel neighbours while coarse zones receive few.
        Falls back to a chunked dense path when SciPy is unavailable.

    "knn_uniform"
        Flat average over the K nearest neighbours:

            w_ij = 1 / K   for all j ∈ KNN(i)

        The physical reach adapts to local mesh density: fine regions are
        smoothed over a small footprint, coarse regions over a large one —
        always blending the same number of cells.  Requires SciPy.

    "knn_gauss"
        Gaussian over the K nearest neighbours with a per-region σ:

            σ_i  = OP_SMOOTH_KNN_SIGMA_FRAC × d_{i,K}
            w_ij = exp(−‖c_i − c_j‖² / 2σ_i²)

        where d_{i,K} is the distance from region i to its K-th nearest
        neighbour.  Reach is mesh-adaptive like "knn_uniform" but the
        kernel decays smoothly within the neighbourhood rather than cutting
        off sharply.  Degenerate regions where d_{i,K} = 0 fall back to
        uniform weights (count reported at end).  Requires SciPy.

    All modes use ``cKDTree.query(k=K)`` which returns a fixed-shape
    (n, K) distance/index array — memory is exactly n × K × 8 bytes,
    predictable and safe regardless of mode.
    """
    ctr  = _smooth_ctx["centroids"]   # (n_free, 3)
    K    = min(int(OP_SMOOTH_K), len(m))
    mode = str(OP_SMOOTH_MODE).strip().lower()

    _valid_modes = {"physical", "knn_uniform", "knn_gauss"}
    if mode not in _valid_modes:
        raise ValueError(
            f"smooth: unknown OP_SMOOTH_MODE={mode!r}. "
            f"Choose one of: {sorted(_valid_modes)}."
        )

    n = len(m)

    # ------------------------------------------------------------------
    # SciPy fast path — covers all three modes
    # ------------------------------------------------------------------
    try:
        from scipy.spatial import cKDTree as _cKDTree
        tree = _cKDTree(ctr)
        # dist: (n, K)  idx: (n, K) — dense, fixed shape, safe
        dist, idx = tree.query(ctr, k=K, workers=-1)
        m_nbr = m[idx]                           # (n, K)  neighbour values

        if mode == "physical":
            two_s2 = 2.0 * _smooth_ctx["sigma"] ** 2
            d2 = dist ** 2                       # (n, K)
            W  = np.exp(-d2 / two_s2)           # (n, K)

        elif mode == "knn_uniform":
            W = np.ones((n, K), dtype=float)     # (n, K) — flat

        else:  # "knn_gauss"
            frac = float(OP_SMOOTH_KNN_SIGMA_FRAC)
            # d_{i,K} = dist[:, -1] (distance to K-th neighbour, 0-based)
            d_kmax = dist[:, -1]                 # (n,)
            sigma_i = frac * d_kmax              # (n,)  per-region σ

            # Degenerate guard: regions where all K neighbours coincide
            # (d_{i,K} == 0) receive uniform weights instead.
            degenerate = sigma_i == 0.0
            n_degen = int(degenerate.sum())

            two_s2_i = 2.0 * np.where(degenerate, 1.0, sigma_i) ** 2  # (n,)
            d2 = dist ** 2                       # (n, K)
            W  = np.exp(-d2 / two_s2_i[:, np.newaxis])   # (n, K)

            # Overwrite degenerate rows with uniform weights
            if n_degen:
                W[degenerate, :] = 1.0
                print(f"  smooth (knn_gauss): {n_degen} degenerate region(s) "
                      f"(d_K=0) fell back to uniform weights.")

        return np.einsum("ij,ij->i", W, m_nbr) / W.sum(axis=1)

    except ImportError:
        if mode != "physical":
            raise RuntimeError(
                f"smooth mode '{mode}' requires SciPy (scipy.spatial.cKDTree). "
                "Install SciPy or use OP_SMOOTH_MODE='physical'."
            )

    # ------------------------------------------------------------------
    # Fallback for mode "physical": chunked dense path, no SciPy needed
    # ------------------------------------------------------------------
    sigma  = _smooth_ctx["sigma"]
    two_s2 = 2.0 * sigma * sigma
    max_bytes  = int(OP_SMOOTH_MAX_GB * 1024 ** 3)
    chunk_size = max(1, max_bytes // (n * 8))
    chunk_size = min(chunk_size, n)

    w_sum  = np.zeros(n, dtype=float)
    wm_sum = np.zeros(n, dtype=float)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        blk = ctr[start:end]                     # (chunk, 3)
        d2  = (
            np.sum(blk ** 2, axis=1, keepdims=True)
            + np.sum(ctr ** 2, axis=1)[np.newaxis, :]
            - 2.0 * (blk @ ctr.T)
        )
        d2 = np.maximum(d2, 0.0)
        W  = np.exp(-d2 / two_s2)               # (chunk, n) — all neighbours
        wm_sum[start:end] = W @ m
        w_sum [start:end] = W.sum(axis=1)

    return wm_sum / w_sum


def _ellipsoid_mask(centroids: np.ndarray, center: list, axes: list,
                    angles_deg: list) -> np.ndarray:
    """True for centroids inside the rotated ellipsoid.  Delegates to fem.ellipsoid_mask."""
    return fem.ellipsoid_mask(
        centroids, center=center, axes=axes, angles_deg=angles_deg, convention="zyx"
    )


def _brick_mask(centroids: np.ndarray, center: list, axes: list,
                angles_deg: list) -> np.ndarray:
    """True for centroids inside the rotated rectangular prism.  Delegates to fem.brick_mask."""
    return fem.brick_mask(
        centroids, center=center, axes=axes, angles_deg=angles_deg, convention="zyx"
    )


def _smooth_body_boundary(m: np.ndarray, centroids: np.ndarray,
                          inside: np.ndarray,
                          sigma: float, passes: int) -> np.ndarray:
    """Smooth the boundary of a body by iterative Gaussian blending.

    On each pass, every region in the transition band (one shell of exterior
    neighbours + the adjacent interior shell) is replaced by the Gaussian-
    weighted average of its neighbours within 4σ.  The zone widens by one
    shell per pass.

    Memory strategy
    ---------------
    The key constraint: with 125 k regions, ANY query_ball_point call that
    touches a large fraction of centroids (e.g. all interior regions) will
    materialise enormous Python lists and segfault.  The solution is to work
    exclusively from the **exterior side**:

    1.  Build a small KD-tree on interior centroids only (tree_int).
    2.  Find exterior regions near the body by a **radius query on the
        exterior tree** bounded by the body's bounding sphere — a tiny set.
    3.  For each near-exterior region, ask tree_int for the nearest interior
        neighbour; if it is within radius, that exterior region is in the band.
        This is an O(n_near_exterior × log n_interior) fixed-k query — no
        variable-length lists, no large allocations.
    4.  The active zone (near-exterior + adjacent interior shell) is built
        entirely from small sets.  All subsequent queries are local.

    Parameters
    ----------
    m         : (n_free,) log10(ρ) vector (body already applied)
    centroids : (n_free, 3) region centroids
    inside    : (n_free,) bool — body interior mask
    sigma     : Gaussian length scale in metres
    passes    : number of smoothing passes

    Returns
    -------
    m_out : (n_free,) smoothed log10(ρ) vector
    """
    try:
        from scipy.spatial import cKDTree as _cKDTree
    except ImportError:
        print("  boundary_smooth: SciPy not available — skipping.")
        return m

    m_out  = m.copy()
    two_s2 = 2.0 * sigma ** 2
    radius = 4.0 * sigma

    interior_idx = np.where( inside)[0]   # global indices into centroids
    exterior_idx = np.where(~inside)[0]

    if len(interior_idx) == 0:
        return m_out

    int_ctr = centroids[interior_idx]     # (n_int, 3)
    ext_ctr = centroids[exterior_idx]     # (n_ext, 3)

    # ── Build trees on each partition — both are safe-sized ────────────────
    tree_int = _cKDTree(int_ctr)          # queried by near-exterior points
    tree_ext = _cKDTree(ext_ctr)          # queried by bounding-sphere filter

    # ── Bounding sphere of interior centroids ───────────────────────────────
    body_centre = int_ctr.mean(axis=0)
    body_radius = float(np.max(np.linalg.norm(int_ctr - body_centre, axis=1)))

    # ── Find exterior regions within body_radius + radius of body centre ────
    # This is the only query on ext_ctr; its result is a small index list.
    near_ext_local = tree_ext.query_ball_point(
        body_centre, r=body_radius + radius, workers=1
    )                                     # list of local indices into ext_ctr
    near_ext_local = np.asarray(near_ext_local, dtype=int)

    if len(near_ext_local) == 0:
        return m_out

    near_ext_ctr = ext_ctr[near_ext_local]   # (n_near_ext, 3) — small

    # ── Identify exterior band: near-exterior regions within radius of any
    #    interior region.  Use fixed-k=1 query (nearest interior neighbour). ──
    dist_to_int, _ = tree_int.query(near_ext_ctr, k=1, workers=-1)
    in_band_local  = near_ext_local[dist_to_int <= radius]   # local ext indices
    ext_band_global = exterior_idx[in_band_local]            # global indices

    # ── Identify interior boundary shell: interior regions within radius of
    #    any exterior band region.  Query tree_int with ext_band centroids. ──
    if len(ext_band_global) == 0:
        return m_out

    ext_band_ctr = centroids[ext_band_global]
    dist_int_to_band, int_shell_local = tree_int.query(
        ext_band_ctr, k=min(64, len(interior_idx)), workers=-1
    )
    int_shell_set = set()
    for row, drow in zip(int_shell_local, dist_int_to_band):
        for j, d in zip(row, drow):
            if d <= radius:
                int_shell_set.add(int(interior_idx[j]))
    int_shell_global = np.array(sorted(int_shell_set), dtype=int)

    # ── Initial active zone and outer shell for iteration ──────────────────
    active_global  = np.concatenate([int_shell_global, ext_band_global])
    outer_shell_global = ext_band_global   # exterior band becomes outer shell

    for p in range(passes):
        if len(active_global) == 0:
            break

        active_ctr = centroids[active_global]   # small — safe for local tree
        tree_local = _cKDTree(active_ctr)

        # query_ball_point on the small active zone — safe
        nbr_lists    = tree_local.query_ball_point(active_ctr, r=radius, workers=1)
        m_active_old = m_out[active_global].copy()

        for i, nb_local in enumerate(nbr_lists):
            nb_local = np.asarray(nb_local, dtype=int)
            d2 = np.sum((active_ctr[nb_local] - active_ctr[i]) ** 2, axis=1)
            w  = np.exp(-d2 / two_s2)
            m_out[active_global[i]] = float(
                (w @ m_active_old[nb_local]) / max(w.sum(), 1e-30)
            )

        # ── Expand: find next exterior shell beyond current outer shell ────
        if p < passes - 1:
            # Query tree_ext for exterior neighbours of current outer shell
            outer_ctr = centroids[outer_shell_global]
            new_ext_local_lists = tree_ext.query_ball_point(
                outer_ctr, r=radius, workers=1
            )
            new_ext_set = set()
            for nb_list in new_ext_local_lists:
                new_ext_set.update(nb_list)
            # Remove already-active exterior regions
            active_set = set(active_global.tolist())
            new_ext_global = np.array(
                [exterior_idx[j] for j in sorted(new_ext_set)
                 if exterior_idx[j] not in active_set],
                dtype=int
            )
            if len(new_ext_global) == 0:
                break
            active_global      = np.concatenate([active_global, new_ext_global])
            outer_shell_global = new_ext_global

    return m_out


def _apply_bodies(m: np.ndarray, centroids: np.ndarray,
                  bodies: list, mask_fn, op_name: str) -> np.ndarray:
    """Apply a list of bodies to the free log10(ρ) vector.

    Each body dict must contain: mode, value, center, axes, angles.
    Bodies are applied in order; later entries overwrite earlier ones where
    masks overlap — allowing layered construction of complex structures.

    Optional per-body boundary smoothing is controlled by the key
    ``boundary_smooth``, a dict with:
        sigma  : Gaussian length scale in metres (blend width per pass)
        passes : number of smoothing passes (transition zone depth)

    Example::

        dict(mode="replace", value=0.0,
             center=[0., 0., 5000.], axes=[10000., 10000., 5000.],
             angles=[0., 0., 0.],
             boundary_smooth=dict(sigma=3000., passes=3))

    Parameters
    ----------
    m          : (n_free,) free log10(ρ) vector
    centroids  : (n_free, 3) region centroids
    bodies     : list of body dicts
    mask_fn    : callable(centroids, center, axes, angles) → bool (n,)
    op_name    : string used in diagnostic output ("ellipsoid" or "brick")
    """
    _valid_modes = {"replace", "add"}
    m_new = m.copy()
    total_modified = np.zeros(len(m), dtype=bool)

    for k, body in enumerate(bodies):
        mode   = str(body.get("mode",   "replace")).strip().lower()
        value  = float(body.get("value",  0.0))
        center = list(body.get("center", [0., 0., 0.]))
        axes   = list(body.get("axes",   [1., 1., 1.]))
        angles = list(body.get("angles", [0., 0., 0.]))
        bsmooth = body.get("boundary_smooth", None)

        if mode not in _valid_modes:
            raise ValueError(
                f"{op_name} body {k}: mode must be 'replace' or 'add', got {mode!r}.")

        inside = mask_fn(centroids, center, axes, angles)
        n_in   = int(inside.sum())
        if n_in == 0:
            print(f"  {op_name} body {k}: WARNING — no free regions inside body "
                  f"(center={center}, axes={axes}).")

        if mode == "replace":
            m_new[inside] = value
        else:  # "add"
            m_new[inside] += value

        # Optional boundary smoothing — applied immediately after this body,
        # before the next body overwrites anything.
        if bsmooth is not None and n_in > 0:
            bs_sigma  = float(bsmooth.get("sigma",  5000.0))
            bs_passes = int(bsmooth.get("passes", 2))
            if OUT:
                print(f"  {op_name} body {k}: boundary smoothing "
                      f"σ={bs_sigma:.0f} m, {bs_passes} pass(es) ...")
            m_new = _smooth_body_boundary(
                m_new, centroids, inside, sigma=bs_sigma, passes=bs_passes
            )

        total_modified |= inside
        if OUT:
            print(f"  {op_name} body {k}: {n_in} regions modified "
                  f"(mode='{mode}', value={value:+.3f}, "
                  f"center={center}, axes={axes}, angles={angles}).")

    if OUT:
        print(f"  {op_name}: {int(total_modified.sum())} of {len(m)} free regions "
              f"modified in total across {len(bodies)} body/bodies.")
    return m_new


def _op_ellipsoid(m: np.ndarray) -> np.ndarray:
    """Modify free-region log10(ρ) inside one or more rotated ellipsoids.

    Body definitions are read from ``_ellipsoid_ctx["bodies"]``.
    """
    return _apply_bodies(
        m, _ellipsoid_ctx["centroids"],
        _ellipsoid_ctx["bodies"],
        _ellipsoid_mask, "ellipsoid",
    )


def _op_brick(m: np.ndarray) -> np.ndarray:
    """Modify free-region log10(ρ) inside one or more rotated rectangular prisms.

    Brick geometry: a box of half-extents [a, b, c] centred at `center` and
    rotated by ZYX angles (yaw, pitch, roll).  The mask test in the local
    frame is: |x'| ≤ a AND |y'| ≤ b AND |z'| ≤ c.

    Body definitions are read from ``_brick_ctx["bodies"]``.
    """
    return _apply_bodies(
        m, _brick_ctx["centroids"],
        _brick_ctx["bodies"],
        _brick_mask, "brick",
    )


def _op_null(m: np.ndarray) -> np.ndarray:
    """No-op: return the input model unchanged.

    No output file is written when OPERATION == "null"; the main block
    skips the write step and goes directly to plotting.  Use this to
    visualise the input model with the slice plotter without modifying
    the block file.
    """
    return m.copy()


_OPERATIONS: dict = {
    "fill":        _op_fill,
    "mean":        _op_mean,
    "wmean":       _op_wmean,
    "median":      _op_median,
    "clip":        _op_clip,
    "shift":       _op_shift,
    "standardise": _op_standardise,
    "smooth":      _op_smooth,
    "ellipsoid":   _op_ellipsoid,
    "brick":       _op_brick,
    "null":        _op_null,
}


# ===========================================================================
# Main
# ===========================================================================

if OPERATION not in _OPERATIONS:
    sys.exit(
        f"Unknown OPERATION={OPERATION!r}. "
        f"Choose one of: {list(_OPERATIONS.keys())}."
    )

# --- (1) Read free log10(ρ) vector ----------------------------------------
print(f"Reading model: {MODEL_IN}")
log_m = fem.read_model(
    model_file=MODEL_IN,
    model_trans="log10",
    ocean=OCEAN,
    out=OUT,
)
print(f"  free parameters: {log_m.size}")
print(f"  log10(ρ) range before: [{log_m.min():.3f}, {log_m.max():.3f}]")
print()

# --- (1b) Build mesh-dependent contexts if needed -------------------------
_NEEDS_MESH = {"smooth", "ellipsoid", "brick", "wmean"}

if OPERATION in _NEEDS_MESH:
    if not os.path.isfile(MESH_FILE):
        sys.exit(f"{OPERATION}: MESH_FILE not found: {MESH_FILE}")

    print(f"Reading mesh: {MESH_FILE}")
    nodes, conn = fem.read_femtic_mesh(MESH_FILE)
    print(f"  nodes={nodes.shape[0]}, elements={conn.shape[0]}")

    _struct = fem._read_resistivity_block_struct(
        MODEL_IN, model_trans="log10", ocean=OCEAN, out=False
    )
    elem_region = _struct["elem_region"]
    free_idx    = _struct["free_idx"]

    region_ctr, region_vol = fem.build_region_geometry(nodes, conn, elem_region, free_idx)
    print(f"  region geometry: {len(free_idx)} free regions, "
          f"total volume={region_vol.sum():.3e} m³")
    print()

    if OPERATION == "wmean":
        _wmean_ctx["volumes"] = region_vol
        print(f"  wmean context ready: "
              f"V range [{region_vol.min():.3e}, {region_vol.max():.3e}] m³, "
              f"ratio={region_vol.max()/max(region_vol.min(),1e-30):.1e}")

    if OPERATION == "smooth":
        _smooth_ctx["centroids"] = region_ctr
        _smooth_ctx["sigma"]     = float(OP_SMOOTH_SIGMA)
        _n_mem_mb = int(len(free_idx) * min(int(OP_SMOOTH_K), len(free_idx)) * 8 / 1024**2)
        _mode_str = str(OP_SMOOTH_MODE).strip().lower()
        if _mode_str == "physical":
            _mode_info = f"σ={OP_SMOOTH_SIGMA:.0f} m"
        elif _mode_str == "knn_uniform":
            _mode_info = "uniform weights"
        else:  # knn_gauss
            _mode_info = f"per-region σ = {OP_SMOOTH_KNN_SIGMA_FRAC} × d_K"
        print(f"  smooth context ready: mode='{_mode_str}', K={OP_SMOOTH_K}, "
              f"{_mode_info}, estimated peak memory ~{_n_mem_mb} MB")

    if OPERATION == "ellipsoid":
        _ellipsoid_ctx["centroids"] = region_ctr
        _ellipsoid_ctx["bodies"]    = list(OP_ELLIPSOID_BODIES)
        print(f"  ellipsoid context ready: {len(OP_ELLIPSOID_BODIES)} body/bodies.")

    if OPERATION == "brick":
        _brick_ctx["centroids"] = region_ctr
        _brick_ctx["bodies"]    = list(OP_BRICK_BODIES)
        print(f"  brick context ready: {len(OP_BRICK_BODIES)} body/bodies.")
    print()

# --- (2) Apply operation ---------------------------------------------------
apply_fn = _OPERATIONS[OPERATION]
log_m_new = apply_fn(log_m)

print(f"Operation '{OPERATION}' applied.")
print(f"  log10(ρ) range after:  [{log_m_new.min():.3f}, {log_m_new.max():.3f}]")
print()

# --- (3) Write modified model ---------------------------------------------
if OPERATION == "null":
    print("Operation 'null': no output file written (input model displayed as-is).")
else:
    print(f"Writing model: {MODEL_OUT}")
    fem.insert_model(
        template=MODEL_IN,
        model=log_m_new,
        model_file=MODEL_OUT,
        ocean=OCEAN,
        air_rho=AIR_RHO,
        ocean_rho=OCEAN_RHO,
        out=OUT,
    )
    print("Done.")

# --- (3b) Derive UTM zone and site positions (needed for plot) -------------
if PLOT:
    # Origin estimation
    if ORIGIN_METHOD is not None and SITE_DAT is not None and os.path.isfile(SITE_DAT):
        import numpy as _np
        _sdat = fem.read_site_dat(SITE_DAT)
        if _sdat:
            _Es = _np.array([d["easting"]  for d in _sdat])
            _Ns = _np.array([d["northing"] for d in _sdat])
            if ORIGIN_METHOD == "box":
                UTM_ORIGIN_E = 0.5 * (_Es.min() + _Es.max())
                UTM_ORIGIN_N = 0.5 * (_Ns.min() + _Ns.max())
            else:
                UTM_ORIGIN_E = float(_Es.mean())
                UTM_ORIGIN_N = float(_Ns.mean())
            _lats = _np.array([d["lat"] for d in _sdat])
            _lons = _np.array([d["lon"] for d in _sdat])
            _z0, _n0 = utl.utm_zone_from_latlon(
                float(_lats.mean()), float(_lons.mean()), override=UTM_ZONE_OVERRIDE)
            UTM_ORIGIN_LAT, UTM_ORIGIN_LON = utl.utm_to_latlon_zn(
                UTM_ORIGIN_E, UTM_ORIGIN_N, _z0, _n0)
    UTM_ZONE, UTM_NORTHERN = utl.utm_zone_from_latlon(
        UTM_ORIGIN_LAT, UTM_ORIGIN_LON, override=UTM_ZONE_OVERRIDE)

    site_xys = []
    if SITE_DAT is not None and os.path.isfile(SITE_DAT):
        for row in fem.read_site_dat(SITE_DAT, site_names=SITE_NAMES):
            sx_m, sy_m = fem.utm_to_model(row["easting"], row["northing"],
                                          UTM_ORIGIN_E, UTM_ORIGIN_N)
            site_xys.append((row["name"], sx_m, sy_m, float(row.get("elev", 0.0))))

# --- (4) Plot slices of output model ---------------------------------------
if PLOT:
    if fviz is None:
        print("  PLOT: femtic_viz not available — skipping slice plot.")
    else:
        _slices_resolved = fem.resolve_slice_positions(
            PLOT_SLICES, UTM_ZONE, UTM_NORTHERN,
            UTM_ORIGIN_E, UTM_ORIGIN_N,
            UTM_ORIGIN_LAT, UTM_ORIGIN_LON,
            verbose=OUT,
        )
        fviz.plot_model_slices(
            model_file         = MODEL_IN if OPERATION == "null" else MODEL_OUT,
            mesh_file          = MESH_FILE,
            slices             = _slices_resolved,
            cmap               = PLOT_CMAP,
            clim               = PLOT_CLIM,
            xlim               = PLOT_XLIM,
            ylim               = PLOT_YLIM,
            zlim               = PLOT_ZLIM,
            ocean_color        = PLOT_OCEAN_COLOR,
            ocean_value        = OCEAN_RHO,
            air_bgcolor        = PLOT_AIR_BGCOLOR,
            site_xys           = site_xys,
            obs_coords_only    = False,
            sites_in_maps      = PLOT_SITES_MAPS,
            sites_in_slices    = PLOT_SITES_SLICES,
            site_marker        = SITE_MARKER,
            site_marker_slices = SITE_MARKER_SLICES,
            map_markers        = MAP_MARKERS,
            projection_dist    = PROJECTION_DIST,
            display_coords     = DISPLAY_COORDS,
            utm_origin_e       = UTM_ORIGIN_E,
            utm_origin_n       = UTM_ORIGIN_N,
            utm_zone           = UTM_ZONE,
            utm_northern       = UTM_NORTHERN,
            utm_to_latlon_fn   = utl.utm_to_latlon_zn,
            latlon_to_model_fn = fem.latlon_to_model,
            depth_km           = DEPTH_KM,
            horiz_km           = HORIZ_KM,
            equal_aspect       = PLOT_EQUAL_ASPECT,
            panel_height       = PLOT_PANEL_HEIGHT / 2.54,
            nrows              = PLOT_NROWS,
            ncols              = PLOT_NCOLS,
            plot_file          = PLOT_FILE,
            dpi                = PLOT_DPI,
            out                = OUT,
        )
