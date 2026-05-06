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
    "smooth"        Gaussian-weighted spatial smoothing in log10 space.
                    Requires MESH_FILE.  Each free region is replaced by the
                    Gaussian-weighted average over its K nearest neighbours
                    (K = OP_SMOOTH_K), with weights = exp(-d²/(2σ²)).
                    Memory is O(n_free × K) — fixed and predictable.
    "ellipsoid"     Modify free regions whose centroid falls inside one or
                    more rotated ellipsoids.  Requires MESH_FILE.
                    Bodies are defined by the list OP_ELLIPSOID_BODIES; each
                    entry is a dict with keys: mode, value, center, axes,
                    angles (ZYX degrees).  Bodies are applied in order; later
                    bodies overwrite earlier ones where masks overlap.
    "brick"         Same as "ellipsoid" but with a rotated rectangular prism
                    (box) geometry.  Bodies are defined by OP_BRICK_BODIES;
                    each entry has the same keys as an ellipsoid body but
                    axes = [a, b, c] are half-extents (metres), not semi-axes.
                    The rotation uses the same ZYX convention.

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
WORK_DIR = r"/home/vrath/Py4MTX/work/"

#: Template / source resistivity block (also used as format template by
#: insert_model to preserve header, bounds and flag columns).
# MODEL_IN  = WORK_DIR + "resistivity_block_iter0.dat"
MODEL_IN  =WORK_DIR + "resistivity_block_ellipsoid.dat"
#: Mesh file — required for "smooth" and "ellipsoid"; ignored otherwise.
MESH_FILE = WORK_DIR + "mesh.dat"

#: Output file.  Set to MODEL_IN to overwrite in-place (be careful!).
MODEL_OUT = WORK_DIR + "resistivity_block_edit.dat"

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
#:         | "standardise" | "smooth" | "ellipsoid" | "brick"
# OPERATION = "mean"
# OPERATION = "wmean"
# OPERATION = "median"
# OPERATION = "mean"
# OPERATION = "ellipsoid"
OPERATION = "smooth"
MODEL_OUT = MODEL_OUT.replace("edit", OPERATION)
print(MODEL_OUT)

# Parameters used by specific operations (ignored when not applicable):
OP_FILL_VALUE   = 2.0    # log10(100 Ω·m) — used by "fill"
OP_CLIP_MIN     = 0.0    # log10(1 Ω·m)   — used by "clip"
OP_CLIP_MAX     = 4.0    # log10(10 kΩ·m) — used by "clip"
OP_SHIFT_VALUE  = 0.5    # added to every log10(ρ) — used by "shift"

#: Gaussian smoothing length σ in metres — used by "smooth".
#: Controls the decay of the Gaussian weight with distance.  A good first
#: guess is 1–2× the typical element edge length in the target depth range.
OP_SMOOTH_SIGMA   = 5000.0  # metres

#: Number of nearest neighbours considered per region — used by "smooth".
#: Memory scales as n_free × K × 8 bytes (predictable, no variable-length
#: lists).  Regions beyond the K-th neighbour get effectively zero weight
#: if their distance >> σ.  Start with K = 50–200; increase if σ is large
#: relative to the typical inter-region spacing.
OP_SMOOTH_K       = 100     # nearest neighbours

#: Maximum RAM (GiB) for the chunked dense fallback — used by "smooth"
#: when SciPy is unavailable.
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
#:   angles          : [α, β, γ]  ZYX rotation in degrees (yaw/strike, pitch/dip, roll)
#:   boundary_smooth : optional dict — smooth the body boundary after insertion
#:     sigma  : Gaussian length scale in metres (blend width per pass)
#:     passes : number of smoothing passes (transition zone depth ≈ passes × σ)
OP_ELLIPSOID_BODIES = [
    dict(mode="replace", value=0.5,
         center=[0.0, 0.0, 5000.0],
         axes=[10000.0, 5000.0, 2500.0],
         angles=[20.0, 30.0, 0.0]),
    # With boundary smoothing:
    #dict(mode="replace", value=0.0,
         #center=[0.0, 0.0, 5000.0],
         #axes=[10000.0, 10000.0, 5000.0],
         #angles=[0.0, 0.0, 0.0],
         #boundary_smooth=dict(sigma=1000., passes=3)),
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

#: File to save the figure (None → interactive show).
PLOT_FILE = WORK_DIR + "resistivity_block_edited.pdf"
PLOT_FILE = PLOT_FILE.replace("edited", OPERATION)
print(PLOT_FILE)

#: Figure DPI.
PLOT_DPI = 600

#: Matplotlib colormap name.
PLOT_CMAP = "turbo_r"

#: Colour limits [log10(rho_min), log10(rho_max)] — None = auto per panel.
PLOT_CLIM = [0.0, 4.0]      # log10(Ohm·m)

#: Colour for ocean / lake cells (flat, not mapped through colormap).
#: Must match OCEAN_RHO set above.  None → use colormap.
PLOT_OCEAN_COLOR = "lightgrey"

#: Colour for "no-data" / air cells (NaN → transparent by default).
#: Setting this to a string (e.g. "whitesmoke") fills the axes background;
#: None leaves the axes background as the figure facecolor.
PLOT_AIR_BGCOLOR = None

#: Slice specification — a list of dicts, one per panel (left to right).
#:
#: Slices use exact tetrahedron-plane intersection — no selection slab,
#: no dw, no dz.  The plane is infinitely thin; every tetrahedron that
#: straddles it contributes an exact triangle or quadrilateral polygon.
#:
#: Each dict must contain:
#:   kind   : "map"   — horizontal slice at z = z0
#:            "ns"    — N-S vertical section at x = x0
#:            "ew"    — E-W vertical section at y = y0
#:            "plane" — arbitrary plane by strike / dip / point
#:   z0     : (map   only)  depth in metres
#:   x0     : (ns    only)  easting in metres
#:   y0     : (ew    only)  northing in metres
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

#: Global axis limits — used for panels that do not specify their own.
#: None → auto (inferred from data extent).
PLOT_XLIM = [-20000., 20000.] #  None   # [xmin, xmax] metres — easting
PLOT_YLIM = [-20000., 20000.] #  None   # [ymin, ymax] metres — northing
PLOT_ZLIM = [ -6000., 15000.] #  None    # [zmin, zmax] metres — depth (z positive-down)



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


def _tet_volumes(nodes: np.ndarray, conn: np.ndarray) -> np.ndarray:
    """Compute the signed volume of each tetrahedron (vectorised).

    V = |det([b-a, c-a, d-a])| / 6  for vertices a, b, c, d.

    Returns
    -------
    vol : (nelem,)  absolute volumes in the same units as the node coordinates³.
    """
    v = nodes[conn]           # (nelem, 4, 3)
    a, b, c, d = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
    bma = b - a;  cma = c - a;  dma = d - a
    # Scalar triple product row-wise
    cross = np.cross(bma, cma)            # (nelem, 3)
    det   = (cross * dma).sum(axis=1)     # (nelem,)
    return np.abs(det) / 6.0


def _build_region_geometry(
    nodes: np.ndarray,
    conn: np.ndarray,
    elem_region: np.ndarray,
    free_idx: np.ndarray,
) -> tuple:
    """Compute centroid and total volume for each *free* region in one pass.

    Parameters
    ----------
    nodes       : (nn, 3)
    conn        : (nelem, 4)  0-based node indices
    elem_region : (nelem,)    region index for each element
    free_idx    : (n_free,)   region indices of free regions (in order)

    Returns
    -------
    region_ctr : (n_free, 3)   centroid [x, y, z] for each free region
    region_vol : (n_free,)     total volume (m³) for each free region
    """
    elem_ctr = nodes[conn].mean(axis=1)        # (nelem, 3)
    elem_vol = _tet_volumes(nodes, conn)        # (nelem,)

    n_free = len(free_idx)
    region_ctr = np.empty((n_free, 3), dtype=float)
    region_vol = np.empty(n_free,      dtype=float)

    for k, ireg in enumerate(free_idx):
        sel = elem_region == ireg
        if not np.any(sel):
            region_ctr[k] = 0.0
            region_vol[k] = 0.0
        else:
            vols = elem_vol[sel]
            region_vol[k] = vols.sum()
            # volume-weighted centroid within each region
            region_ctr[k] = (vols[:, np.newaxis] * elem_ctr[sel]).sum(axis=0) / region_vol[k]

    return region_ctr, region_vol


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
    """Gaussian-weighted spatial smoothing of log10(ρ) across free regions.

    Each region's new value is the normalised Gaussian-weighted sum over
    its K nearest neighbours (by region centroid distance):

        m̃_i = Σ_{j ∈ KNN(i)}  w_ij · m_j  /  Σ w_ij
        w_ij = exp(−‖c_i − c_j‖² / (2 σ²))

    Implementation strategy
    -----------------------
    *   ``query_ball_point`` (variable-length lists) segfaults at ~125 k
        regions because the combined neighbour lists exhaust RAM during the
        Python loop before any computation completes.

    *   ``cKDTree.query(k=K)`` returns a **fixed-shape** (n, K) array of
        distances and indices.  Memory is exactly n × K × 8 bytes — fully
        predictable and safe.  For K = 100 and n = 125 k that is ~100 MB.

    *   Vectorised computation: weights W (n, K), dot products via einsum —
        no Python loop over regions.

    If SciPy is unavailable the code falls back to a chunked dense path
    that stays within ``OP_SMOOTH_MAX_GB`` gigabytes.
    """
    ctr    = _smooth_ctx["centroids"]   # (n_free, 3)
    sigma  = _smooth_ctx["sigma"]
    K      = int(OP_SMOOTH_K)
    two_s2 = 2.0 * sigma * sigma
    n      = len(m)

    K = min(K, n)   # cannot request more neighbours than points

    # ------------------------------------------------------------------
    # Fast path: SciPy cKDTree with fixed-k query  (no variable lists)
    # ------------------------------------------------------------------
    try:
        from scipy.spatial import cKDTree as _cKDTree
        tree = _cKDTree(ctr)
        # dist: (n, K)  idx: (n, K) — both dense, fixed shape, safe
        dist, idx = tree.query(ctr, k=K, workers=-1)
        d2 = dist ** 2                           # (n, K)
        W  = np.exp(-d2 / two_s2)               # (n, K)
        # Gather m values for each neighbour: m_nbr[i, j] = m[idx[i, j]]
        m_nbr = m[idx]                           # (n, K)
        return np.einsum("ij,ij->i", W, m_nbr) / W.sum(axis=1)

    except ImportError:
        pass   # fall through to chunked dense path

    # ------------------------------------------------------------------
    # Fallback: chunked dense path capped at OP_SMOOTH_MAX_GB
    # ------------------------------------------------------------------
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


def _rotation_matrix_zyx(angles_deg: list) -> np.ndarray:
    """Build an intrinsic ZYX rotation matrix from [yaw, pitch, roll] in degrees.

    Applies rotations in order: Z (yaw α), then Y (pitch β), then X (roll γ).
    The resulting matrix R satisfies:  p_local = R.T @ (p - center).
    """
    a, b, g = (np.deg2rad(x) for x in angles_deg)
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cg, sg = np.cos(g), np.sin(g)
    Rz = np.array([[ ca, -sa, 0.],
                   [ sa,  ca, 0.],
                   [ 0.,  0., 1.]])
    Ry = np.array([[ cb,  0., sb],
                   [ 0.,  1., 0.],
                   [-sb,  0., cb]])
    Rx = np.array([[1.,  0.,  0.],
                   [0.,  cg, -sg],
                   [0.,  sg,  cg]])
    return Rz @ Ry @ Rx   # shape (3, 3)


def _local_coords(centroids: np.ndarray, center: list, angles_deg: list) -> np.ndarray:
    """Map centroids into body-local frame via ZYX rotation about center.

    Returns
    -------
    local : (n, 3)  coordinates in rotated frame
    """
    c = np.asarray(center, dtype=float)
    R = _rotation_matrix_zyx(angles_deg)
    return (centroids - c[np.newaxis, :]) @ R   # (n, 3)


def _ellipsoid_mask(centroids: np.ndarray, center: list, axes: list,
                    angles_deg: list) -> np.ndarray:
    """True for centroids inside the rotated ellipsoid.

    Quadratic form in local frame: (x'/a)²+(y'/b)²+(z'/c)² ≤ 1
    """
    ax = np.asarray(axes, dtype=float)
    if np.any(ax <= 0.):
        raise ValueError(f"ellipsoid axes must all be > 0, got {axes}.")
    local = _local_coords(centroids, center, angles_deg)
    q = (local[:, 0] / ax[0])**2 + (local[:, 1] / ax[1])**2 + (local[:, 2] / ax[2])**2
    return q <= 1.0


def _brick_mask(centroids: np.ndarray, center: list, axes: list,
                angles_deg: list) -> np.ndarray:
    """True for centroids inside the rotated rectangular prism (brick).

    Box test in local frame: |x'| ≤ a  AND  |y'| ≤ b  AND  |z'| ≤ c
    axes = [a, b, c] are half-extents in metres (all > 0).
    """
    ax = np.asarray(axes, dtype=float)
    if np.any(ax <= 0.):
        raise ValueError(f"brick axes must all be > 0, got {axes}.")
    local = _local_coords(centroids, center, angles_deg)
    return (np.abs(local[:, 0]) <= ax[0]) &            (np.abs(local[:, 1]) <= ax[1]) &            (np.abs(local[:, 2]) <= ax[2])


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
}


# ===========================================================================
# Plotting helpers
# ===========================================================================

# Each tetrahedron has 4 faces; face_opp[i] = face opposite vertex i
_FACE_OPP = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]


def _plane_basis(normal: np.ndarray):
    """Compute two orthonormal in-plane axes for a given unit normal.

    Returns
    -------
    u, v : (3,) unit vectors spanning the plane, with v chosen to point
           as close to +z (down) as possible for curtain-like sections.
    """
    n = normal / np.linalg.norm(normal)
    # Choose a reference vector not parallel to n
    ref = np.array([0., 0., 1.]) if abs(n[2]) < 0.9 else np.array([1., 0., 0.])
    u = np.cross(n, ref);  u /= np.linalg.norm(u)
    v = np.cross(n, u);    v /= np.linalg.norm(v)
    # Flip v so it has a positive z-component where possible (depth increases
    # downward in the plot), mirroring the convention of the axis-aligned slices.
    if v[2] < 0:
        v = -v;  u = -u
    return u, v


def _strike_dip_to_normal(strike_deg: float, dip_deg: float) -> np.ndarray:
    """Convert geological strike and dip to a unit plane normal.

    Convention (right-hand rule, z positive-down):
      - strike is measured clockwise from North (= +y axis).
      - dip is the maximum downward inclination from horizontal, measured
        perpendicular to strike toward the right-hand side of the strike
        direction.
      - A horizontal plane has dip = 0; a vertical plane has dip = 90.

    Parameters
    ----------
    strike_deg : float  Strike angle in degrees (0 = N, 90 = E, 180 = S …)
    dip_deg    : float  Dip angle in degrees (0 = horizontal, 90 = vertical)

    Returns
    -------
    normal : (3,) unit vector perpendicular to the plane, pointing upward
             (negative z component for dipping planes in a z-down frame).
    """
    s = np.deg2rad(strike_deg)
    d = np.deg2rad(dip_deg)
    # Strike direction (horizontal, clockwise from N/+y):  (+sin s, +cos s, 0)
    # Dip direction (perpendicular to strike, horizontal): (+cos s, -sin s, 0)
    # Down-dip unit vector (tilted by dip angle):
    #   dip_vec = cos(d)*(dip_horiz) + sin(d)*(0,0,1)
    # Normal = strike_vec × dip_vec  (right-hand rule)
    strike_vec = np.array([ np.sin(s),  np.cos(s), 0.0])
    dip_horiz  = np.array([ np.cos(s), -np.sin(s), 0.0])
    dip_vec    = np.cos(d) * dip_horiz + np.sin(d) * np.array([0., 0., 1.])
    normal     = np.cross(strike_vec, dip_vec)
    return normal / np.linalg.norm(normal)


def _intersect_tet_plane_general(tet_nodes: np.ndarray,
                                  normal: np.ndarray,
                                  point: np.ndarray,
                                  u: np.ndarray,
                                  v: np.ndarray):
    """Compute the exact polygon where an arbitrary plane intersects a tetrahedron.

    The plane is defined by a point and a unit normal.  Intersection points
    are projected onto the two in-plane basis vectors u, v to give 2-D
    coordinates suitable for plotting.

    Parameters
    ----------
    tet_nodes : (4, 3)  node coordinates
    normal    : (3,)    unit plane normal
    point     : (3,)    any point on the plane
    u, v      : (3,)    orthonormal in-plane basis vectors

    Returns
    -------
    poly2d : (k, 2) array of (u, v) coordinates, or None if no intersection.
    """
    d = (tet_nodes - point[np.newaxis, :]) @ normal   # signed distances, (4,)

    pts2d = []
    edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    for i, j in edges:
        di, dj = d[i], d[j]
        if (di < 0) == (dj < 0):
            continue
        if di == dj:
            continue
        t = di / (di - dj)
        pt3d = tet_nodes[i] + t * (tet_nodes[j] - tet_nodes[i])
        pts2d.append([float((pt3d - point) @ u),
                      float((pt3d - point) @ v)])

    if len(pts2d) < 3:
        return None

    pts = np.array(pts2d, dtype=float)
    c   = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[np.argsort(angles)]


def _slice_geometry(nodes: np.ndarray, conn: np.ndarray,
                    rho_elem: np.ndarray,
                    normal: np.ndarray,
                    point: np.ndarray,
                    u: np.ndarray,
                    v: np.ndarray):
    """Compute exact plane-tetrahedron intersection polygons for one slice.

    Works for any plane orientation (axis-aligned or arbitrary strike/dip).

    Parameters
    ----------
    nodes    : (nn, 3)    node coordinates
    conn     : (nelem, 4) 0-based connectivity
    rho_elem : (nelem,)   element resistivity (NaN = air)
    normal   : (3,)       unit plane normal
    point    : (3,)       any point on the plane
    u, v     : (3,)       orthonormal in-plane basis vectors

    Returns
    -------
    polys : list of (k, 2) arrays  — intersection polygons in (u, v) space
    vals  : list of float          — resistivity for each polygon
    """
    # Pre-filter: signed distance of all nodes; element straddles plane iff
    # min(d) ≤ 0 ≤ max(d)
    node_d   = (nodes - point[np.newaxis, :]) @ normal   # (nn,)
    tet_d    = node_d[conn]                               # (nelem, 4)
    straddles = (tet_d.min(axis=1) <= 0.) & (tet_d.max(axis=1) >= 0.)

    polys = []
    vals  = []
    for ie in np.where(straddles)[0]:
        poly = _intersect_tet_plane_general(
            nodes[conn[ie]], normal, point, u, v
        )
        if poly is not None:
            polys.append(poly)
            vals.append(float(rho_elem[ie]))

    return polys, vals


def _axis_slice_params(axis: int, coord0: float):
    """Return (normal, point, u, v, xlabel, ylabel, invert_v) for an axis-aligned slice."""
    n = np.zeros(3);  n[axis] = 1.0
    p = np.zeros(3);  p[axis] = coord0
    u, v = _plane_basis(n)
    labels = (["x (m)", "y (m)", "z (m, +down)"],
              ["y (m)", "x (m)", "z (m, +down)"])  # placeholder
    # Override u/v to match previous axis-aligned conventions exactly:
    if axis == 2:   # map: u=x, v=y, no depth inversion
        u = np.array([1., 0., 0.])
        v = np.array([0., 1., 0.])
        return n, p, u, v, "x (m)", "y (m)", False
    elif axis == 0:  # NS: u=y, v=z
        u = np.array([0., 1., 0.])
        v = np.array([0., 0., 1.])
        return n, p, u, v, "y (m)", "z (m, positive down)", True
    else:            # EW: u=x, v=z
        u = np.array([1., 0., 0.])
        v = np.array([0., 0., 1.])
        return n, p, u, v, "x (m)", "z (m, positive down)", True


def _plot_slice_panel(ax, polys: list, vals: list, *,
                      cmap_obj, norm, ocean_color, ocean_value,
                      invert_v: bool = False):
    """Render exact plane-intersection polygons onto ax.

    Parameters
    ----------
    ax          : Matplotlib axes
    polys       : list of (k, 2) vertex arrays (from _slice_geometry)
    vals        : list of float resistivity per polygon (NaN = air)
    cmap_obj    : colormap with set_bad(alpha=0) already set
    norm        : Matplotlib Normalize
    ocean_color : flat colour for ocean polygons; None → colormap
    ocean_value : sentinel rho identifying ocean (Ohm·m)
    invert_v    : invert vertical axis for depth sections

    Returns
    -------
    mappable : PolyCollection or None
    """
    from matplotlib.collections import PolyCollection

    if not polys:
        return None

    vals_arr = np.asarray(vals, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lvals = np.log10(vals_arr)

    is_air = ~np.isfinite(lvals)
    if ocean_color is not None and ocean_value is not None:
        is_ocean = np.isclose(vals_arr, float(ocean_value), rtol=1e-6, atol=0.0)
    else:
        is_ocean = np.zeros(len(vals_arr), dtype=bool)
    earth = ~is_air & ~is_ocean

    polys_arr  = np.asarray(polys, dtype=object)
    mappable   = None

    # Earth polygons — coloured by log10(rho)
    earth_polys = [polys[i] for i in np.where(earth)[0]]
    if earth_polys:
        pc = PolyCollection(earth_polys,
                            array=lvals[earth],
                            cmap=cmap_obj, norm=norm,
                            linewidths=0, rasterized=True)
        ax.add_collection(pc)
        mappable = pc

    # Ocean polygons — flat colour
    if is_ocean.any() and ocean_color is not None:
        oc_polys = [polys[i] for i in np.where(is_ocean)[0]]
        oc = PolyCollection(oc_polys,
                            facecolor=ocean_color, linewidths=0,
                            zorder=3, rasterized=True)
        ax.add_collection(oc)

    # Air polygons — omitted (transparent background shows through)

    ax.autoscale_view()
    if invert_v:
        ax.invert_yaxis()

    return mappable


def plot_model_slices(
    model_file: str,
    mesh_file: str,
    slices: list,
    *,
    cmap: str = "turbo_r",
    clim=None,
    xlim=None,
    ylim=None,
    zlim=None,
    ocean_color="lightgrey",
    ocean_value: float = OCEAN_RHO,
    air_bgcolor="white",
    plot_file=None,
    dpi: int = 200,
    out: bool = True,
):
    """Produce a multi-panel figure of axis-parallel model slices.

    Each slice is computed by intersecting all tetrahedra with the slice
    plane — exact geometry, no Delaunay re-triangulation, no bridging
    artefacts, correct topography regardless of mesh coarseness.

    For map slices (z = const) the intersection polygons are in (x, y).
    For NS curtains (x = const) they are in (y, z); for EW (y = const)
    in (x, z).  Depth axis is positive-down and inverted for curtains.

    Parameters
    ----------
    model_file : resistivity block file (typically MODEL_OUT)
    mesh_file  : mesh file (same mesh.dat used during inversion)
    slices     : list of slice-spec dicts (see PLOT_SLICES config)
    cmap       : Matplotlib colormap name
    clim       : [log10_min, log10_max]; None = auto from finite model values
    xlim, ylim, zlim : global axis limits (metres); per-panel keys override
    ocean_color: flat colour for ocean polygons; None → colormap
    ocean_value: sentinel rho (Ohm·m) identifying ocean; must match OCEAN_RHO
    air_bgcolor: axes facecolor for transparent air pixels (default "white")
    plot_file  : output path; None = interactive show()
    dpi        : figure DPI for saved file
    out        : verbose progress
    """
    if fviz is None:
        print("  plot_model_slices: femtic_viz not available — skipping.")
        return

    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import matplotlib.cm as mcm
    except ImportError:
        print("  plot_model_slices: Matplotlib not available — skipping.")
        return

    # ── load model ──────────────────────────────────────────────────────────
    if out:
        print(f"  plot: reading model {os.path.basename(model_file)}")
    mesh  = fviz.read_femtic_mesh(mesh_file)
    block = fviz.read_resistivity_block(model_file)
    rho_elem = fviz.map_regions_to_element_rho(
        block.region_of_elem, block.region_rho
    )
    rho_plot = fviz.prepare_rho_for_plotting(
        rho_elem,
        air_is_nan=True,
        ocean_value=float(ocean_value),
        region_of_elem=block.region_of_elem,
    )
    nodes = mesh.nodes   # (nn, 3)
    conn  = mesh.conn    # (nelem, 4)
    if out:
        print(f"  plot: {len(slices)} panels, exact plane-intersection method")

    # ── colormap: NaN (air) → transparent ──────────────────────────────────
    cmap_obj = mcm.get_cmap(cmap).copy()
    cmap_obj.set_bad(alpha=0.0)

    # ── colour normalisation ────────────────────────────────────────────────
    if clim is not None:
        norm = mcolors.Normalize(vmin=float(clim[0]), vmax=float(clim[1]))
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            _lall = np.log10(rho_plot[np.isfinite(rho_plot)])
        _lall = _lall[np.isfinite(_lall)]
        norm = mcolors.Normalize(vmin=float(np.nanmin(_lall)),
                                 vmax=float(np.nanmax(_lall)))

    # ── figure layout ────────────────────────────────────────────────────────
    ncols = len(slices)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5),
                             constrained_layout=True)
    if ncols == 1:
        axes = [axes]
    fig.set_facecolor("white")

    # ── render each panel ───────────────────────────────────────────────────
    for ax, spec in zip(axes, slices):
        kind  = str(spec.get("kind", "map")).lower()
        title = spec.get("title", None)
        _xlim = spec.get("xlim", xlim)
        _ylim = spec.get("ylim", ylim)
        _zlim = spec.get("zlim", zlim)

        if air_bgcolor is not None:
            ax.set_facecolor(air_bgcolor)

        if kind == "map":
            z0 = float(spec.get("z0", 5000.0))
            if out: print(f"    map slice z={z0:.0f} m ...")
            normal, point, u, v, xlabel, ylabel, inv = _axis_slice_params(2, z0)
            polys, vals = _slice_geometry(nodes, conn, rho_plot, normal, point, u, v)
            mappable = _plot_slice_panel(ax, polys, vals,
                cmap_obj=cmap_obj, norm=norm,
                ocean_color=ocean_color, ocean_value=ocean_value, invert_v=inv)
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
            ax.set_aspect("equal")
            if _xlim is not None: ax.set_xlim(_xlim)
            if _ylim is not None: ax.set_ylim(_ylim)
            if title is None: title = f"z = {z0/1000:.1f} km"

        elif kind == "ns":
            x0 = float(spec.get("x0", 0.0))
            if out: print(f"    NS slice x={x0:.0f} m ...")
            normal, point, u, v, xlabel, ylabel, inv = _axis_slice_params(0, x0)
            polys, vals = _slice_geometry(nodes, conn, rho_plot, normal, point, u, v)
            mappable = _plot_slice_panel(ax, polys, vals,
                cmap_obj=cmap_obj, norm=norm,
                ocean_color=ocean_color, ocean_value=ocean_value, invert_v=inv)
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
            if _ylim is not None: ax.set_xlim(_ylim)
            if _zlim is not None: ax.set_ylim([_zlim[1], _zlim[0]])
            if title is None: title = f"N-S  x = {x0/1000:.1f} km"

        elif kind == "ew":
            y0 = float(spec.get("y0", 0.0))
            if out: print(f"    EW slice y={y0:.0f} m ...")
            normal, point, u, v, xlabel, ylabel, inv = _axis_slice_params(1, y0)
            polys, vals = _slice_geometry(nodes, conn, rho_plot, normal, point, u, v)
            mappable = _plot_slice_panel(ax, polys, vals,
                cmap_obj=cmap_obj, norm=norm,
                ocean_color=ocean_color, ocean_value=ocean_value, invert_v=inv)
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
            if _xlim is not None: ax.set_xlim(_xlim)
            if _zlim is not None: ax.set_ylim([_zlim[1], _zlim[0]])
            if title is None: title = f"E-W  y = {y0/1000:.1f} km"

        elif kind == "plane":
            # Arbitrary plane defined by strike / dip / point.
            # point : [x, y, z] any point on the plane (metres, z positive-down)
            # strike: clockwise from North (degrees); 0 = N, 90 = E
            # dip   : downward inclination from horizontal (degrees); 0 = flat, 90 = vertical
            _pt     = np.asarray(spec.get("point",  [0., 0., 0.]), dtype=float)
            _strike = float(spec.get("strike", 0.0))
            _dip    = float(spec.get("dip",    90.0))
            if out: print(f"    plane slice strike={_strike:.0f}° dip={_dip:.0f}° "
                          f"through ({_pt[0]:.0f}, {_pt[1]:.0f}, {_pt[2]:.0f}) m ...")
            normal = _strike_dip_to_normal(_strike, _dip)
            u, v   = _plane_basis(normal)
            polys, vals = _slice_geometry(nodes, conn, rho_plot, normal, _pt, u, v)
            mappable = _plot_slice_panel(ax, polys, vals,
                cmap_obj=cmap_obj, norm=norm,
                ocean_color=ocean_color, ocean_value=ocean_value,
                invert_v=True)
            ax.set_xlabel("along-strike (m)")
            ax.set_ylabel("down-dip (m)")
            # Apply limits in (u, v) space if given
            if _xlim is not None: ax.set_xlim(_xlim)
            if _ylim is not None: ax.set_ylim(_ylim)
            if title is None:
                title = f"Plane  str={_strike:.0f}°  dip={_dip:.0f}°"

        else:
            ax.set_visible(False)
            print(f"  plot: unknown slice kind {kind!r} — skipped.")
            continue

        if mappable is not None:
            cb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("log₁₀(ρ / Ω·m)", fontsize=8)
            cb.ax.tick_params(labelsize=7)

        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)

    fig.suptitle(f"Model: {os.path.basename(model_file)}", fontsize=10)

    if plot_file is not None:
        fig.savefig(plot_file, dpi=dpi, bbox_inches="tight")
        if out:
            print(f"  plot: saved → {plot_file}")
    else:
        plt.show()


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

    region_ctr, region_vol = _build_region_geometry(nodes, conn, elem_region, free_idx)
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
        print(f"  smooth context ready: σ={OP_SMOOTH_SIGMA:.0f} m, K={OP_SMOOTH_K}, "
              f"estimated peak memory ~{_n_mem_mb} MB")

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

# --- (4) Plot slices of output model ---------------------------------------
if PLOT:
    plot_model_slices(
        model_file  = MODEL_OUT,
        mesh_file   = MESH_FILE,
        slices      = PLOT_SLICES,
        cmap        = PLOT_CMAP,
        clim        = PLOT_CLIM,
        xlim        = PLOT_XLIM,
        ylim        = PLOT_YLIM,
        zlim        = PLOT_ZLIM,
        ocean_color = PLOT_OCEAN_COLOR,
        ocean_value = OCEAN_RHO,
        air_bgcolor = PLOT_AIR_BGCOLOR,
        plot_file   = PLOT_FILE,
        dpi         = PLOT_DPI,
        out         = OUT,
    )
