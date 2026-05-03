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
MODEL_IN  = WORK_DIR + "resistivity_block_iter0.dat"

#: Mesh file — required for "smooth" and "ellipsoid"; ignored otherwise.
MESH_FILE = WORK_DIR + "mesh.dat"

#: Output file.  Set to MODEL_IN to overwrite in-place (be careful!).
MODEL_OUT = WORK_DIR + "resistivity_block_smooth.dat"

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
OPERATION = "smooth"

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
#:   mode   : "replace" | "add"
#:   value  : float  log10(Ω·m) — absolute if replace, signed offset if add
#:   center : [x, y, z]  metres, z positive-down
#:   axes   : [a, b, c]  semi-axes in metres, all > 0
#:   angles : [α, β, γ]  ZYX rotation in degrees (yaw, pitch, roll)
OP_ELLIPSOID_BODIES = [
    dict(mode="replace", value=0.0,
         center=[0.0, 0.0, 5000.0],
         axes=[10000.0, 10000.0, 5000.0],
         angles=[0.0, 0.0, 0.0]),
    # add more bodies here if needed, e.g.:
    # dict(mode="add", value=-1.0,
    #      center=[5000.0, 0.0, 8000.0],
    #      axes=[3000.0, 3000.0, 3000.0],
    #      angles=[30.0, 0.0, 0.0]),
]

# ---------------------------------------------------------------------------
# Brick bodies — used by "brick" only
# ---------------------------------------------------------------------------
#: List of brick (rotated rectangular prism) body dicts, applied in order.
#: Same keys as ellipsoid bodies; axes = [a, b, c] are half-extents (metres).
#: The box test in the rotated local frame is |x'| ≤ a, |y'| ≤ b, |z'| ≤ c.
OP_BRICK_BODIES = [
    dict(mode="replace", value=0.0,
         center=[0.0, 0.0, 5000.0],
         axes=[10000.0, 8000.0, 4000.0],
         angles=[0.0, 0.0, 0.0]),
    # dict(mode="add", value=1.0,
    #      center=[0.0, 0.0, 15000.0],
    #      axes=[5000.0, 5000.0, 5000.0],
    #      angles=[45.0, 0.0, 0.0]),
]

# ---------------------------------------------------------------------------
# Verbose output
# ---------------------------------------------------------------------------
OUT = True

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


_wmean_ctx: dict = {}


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


def _apply_bodies(m: np.ndarray, centroids: np.ndarray,
                  bodies: list, mask_fn, op_name: str) -> np.ndarray:
    """Apply a list of bodies to the free log10(ρ) vector.

    Each body dict must contain: mode, value, center, axes, angles.
    Bodies are applied in order; later entries overwrite earlier ones where
    masks overlap — allowing layered construction of complex structures.

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
#: Module-level dicts populated here; consumed by mesh-dependent operations.
_smooth_ctx:    dict = {}
_ellipsoid_ctx: dict = {}
_brick_ctx:     dict = {}
_wmean_ctx:     dict = {}

_NEEDS_MESH = {"smooth", "ellipsoid", "brick", "wmean"}

if OPERATION in _NEEDS_MESH:
    if not os.path.isfile(MESH_FILE):
        sys.exit(f"{OPERATION}: MESH_FILE not found: {MESH_FILE}")

    print(f"Reading mesh: {MESH_FILE}")
    nodes, conn = fem.read_femtic_mesh(MESH_FILE)
    print(f"  nodes={nodes.shape[0]}, elements={conn.shape[0]}")

    # Element→region mapping and free_idx — shared by all mesh operations.
    _struct = fem._read_resistivity_block_struct(
        MODEL_IN, model_trans="log10", ocean=OCEAN, out=False
    )
    elem_region = _struct["elem_region"]   # (nelem,)
    free_idx    = _struct["free_idx"]      # (n_free,) region indices of free regions

    # Single pass: centroids + volumes for all free regions.
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
