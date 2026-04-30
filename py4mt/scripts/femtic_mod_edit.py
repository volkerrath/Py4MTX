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
    "median"        Replace every free region with the median of log10(ρ).
    "clip"          Clamp log10(ρ) to [OP_CLIP_MIN, OP_CLIP_MAX].
    "shift"         Add a scalar offset: log10(ρ) += OP_SHIFT_VALUE.
    "standardise"   Map to zero-mean / unit-variance in log10 space.
    "smooth"        Gaussian-weighted spatial smoothing in log10 space.
                    Requires MESH_FILE.  Each free region is replaced by the
                    distance-weighted average of all free-region values,
                    where weights = exp(-d² / (2 σ²)) and σ = OP_SMOOTH_SIGMA.
    "ellipsoid"     Modify free regions whose centroid falls inside a
                    rotated ellipsoid.  Requires MESH_FILE.
                    OP_ELLIPSOID_MODE controls the modification:
                      "replace" — set inside regions to OP_ELLIPSOID_VALUE.
                      "add"     — add OP_ELLIPSOID_VALUE to inside regions.
                    Geometry is defined by OP_ELLIPSOID_CENTER (x,y,z in m),
                    OP_ELLIPSOID_AXES (a,b,c semi-axes in m), and
                    OP_ELLIPSOID_ANGLES (rotation angles in degrees, ZYX
                    convention).

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
#: One of: "fill" | "mean" | "median" | "clip" | "shift" | "standardise"
#:         | "smooth" | "ellipsoid"
# OPERATION = "mean"
OPERATION = "smooth"


# Parameters used by specific operations (ignored when not applicable):
OP_FILL_VALUE   = 2.0    # log10(100 Ω·m) — used by "fill"
OP_CLIP_MIN     = 0.0    # log10(1 Ω·m)   — used by "clip"
OP_CLIP_MAX     = 4.0    # log10(10 kΩ·m) — used by "clip"
OP_SHIFT_VALUE  = 0.5    # added to every log10(ρ) — used by "shift"

#: Gaussian smoothing length σ in metres — used by "smooth".
#: Controls the spatial correlation length: neighbours within ~2σ contribute
#: most of the weight.  A good first guess is 1–2× the typical element size
#: in the target depth range.
OP_SMOOTH_SIGMA   = 5000.0  # metres

#: Distance cutoff as a multiple of σ — used by "smooth".
#: Neighbours beyond this radius get zero weight (Gaussian < exp(-cutoff²/2)).
#: 4σ → weight < 0.03 %; 5σ → < 0.0004 %.  Larger values = more accurate but
#: more neighbours per region and slower.
OP_SMOOTH_CUTOFF  = 4.0     # multiples of σ

#: Maximum RAM (GiB) for the fallback chunked-dense path — used by "smooth"
#: when SciPy is unavailable.  Each chunk is (chunk_size × n_free) float64.
OP_SMOOTH_MAX_GB  = 4.0     # GiB

# ---------------------------------------------------------------------------
# Ellipsoid parameters — used by "ellipsoid" only
# ---------------------------------------------------------------------------
#: Modification mode inside the ellipsoid:
#:   "replace" — overwrite inside-region values with OP_ELLIPSOID_VALUE.
#:   "add"     — add OP_ELLIPSOID_VALUE to existing inside-region values
#:               (positive → more resistive; negative → more conductive).
OP_ELLIPSOID_MODE   = "replace"

#: Value applied inside the ellipsoid (log10 Ω·m).
#:   "replace" mode: absolute resistivity, e.g. 0.0 = 1 Ω·m (conductor).
#:   "add"     mode: signed offset,        e.g. 1.0 = one decade more resistive.
OP_ELLIPSOID_VALUE  = 0.0    # log10(Ω·m)

#: Ellipsoid centre [x, y, z] in model coordinates (metres, z positive-down).
OP_ELLIPSOID_CENTER = [0.0, 0.0, 5000.0]

#: Ellipsoid semi-axes [a, b, c] in metres (must be > 0).
#: a = half-extent along rotated X, b = Y, c = Z (before rotation).
OP_ELLIPSOID_AXES   = [10000.0, 10000.0, 5000.0]

#: Rotation angles [α, β, γ] in degrees — intrinsic ZYX (yaw, pitch, roll).
#: [0, 0, 0] = axis-aligned ellipsoid.
OP_ELLIPSOID_ANGLES = [0.0, 0.0, 0.0]

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


def _build_region_centroids(
    nodes: np.ndarray,
    conn: np.ndarray,
    elem_region: np.ndarray,
    free_idx: np.ndarray,
) -> np.ndarray:
    """Compute the volume-weighted centroid for each *free* region.

    Each free region may contain many elements.  The region centroid is the
    mean of all element centroids belonging to that region (unweighted by
    element volume — acceptable for smoothly varying meshes; volume-weighting
    can be added if needed).

    Parameters
    ----------
    nodes : (nn, 3)
    conn  : (nelem, 4)  0-based node indices
    elem_region : (nelem,)  region index for each element
    free_idx : (n_free,)  region indices of free regions (in order)

    Returns
    -------
    region_ctr : (n_free, 3)  centroid [x, y, z] for each free region
    """
    # Element centroids — vectorised, shape (nelem, 3)
    elem_ctr = nodes[conn].mean(axis=1)

    region_ctr = np.empty((len(free_idx), 3), dtype=float)
    for k, ireg in enumerate(free_idx):
        sel = elem_region == ireg
        if not np.any(sel):
            # Region present in block but no elements assigned — use origin as
            # a safe fallback; its weight will be small or zero in practice.
            region_ctr[k] = 0.0
        else:
            region_ctr[k] = elem_ctr[sel].mean(axis=0)
    return region_ctr


def _op_smooth(m: np.ndarray) -> np.ndarray:
    """Gaussian-weighted spatial smoothing of log10(ρ) across free regions.

    Uses precomputed region centroids stored in ``_smooth_ctx``.  Each
    region's new value is the normalised Gaussian-weighted sum:

        m̃_i = Σ_j  w_ij · m_j  /  Σ_j w_ij
        w_ij = exp(−‖c_i − c_j‖² / (2 σ²))

    Memory-efficient implementation
    --------------------------------
    The naive dense approach allocates an (n, n, 3) array of differences
    (~350 GiB for n = 125 k).  Instead:

    1.  A distance cutoff of ``OP_SMOOTH_CUTOFF * σ`` is applied.
        At 4σ the Gaussian weight is exp(-8) ≈ 0.03 %, negligible.
        Neighbours within the cutoff radius are found with a cKDTree
        query_ball_point call — O(n · n_neighbours) memory.

    2.  Weighted accumulation is done row-by-row in a plain Python loop
        so peak memory at any instant is O(n_neighbours_max), not O(n²).

    If SciPy is unavailable the code falls back to a chunked dense path
    that stays within ``OP_SMOOTH_MAX_GB`` gigabytes.
    """
    ctr   = _smooth_ctx["centroids"]     # (n_free, 3)
    sigma = _smooth_ctx["sigma"]
    cutoff_factor = float(OP_SMOOTH_CUTOFF)
    radius = cutoff_factor * sigma

    n = len(m)
    m_out   = np.empty(n, dtype=float)
    two_s2  = 2.0 * sigma * sigma

    # ------------------------------------------------------------------
    # Fast path: SciPy cKDTree
    # ------------------------------------------------------------------
    try:
        from scipy.spatial import cKDTree as _cKDTree
        tree = _cKDTree(ctr)
        # Returns list of lists — variable-length neighbour sets
        idx_lists = tree.query_ball_point(ctr, r=radius)
        for i, nbrs in enumerate(idx_lists):
            nbrs = np.asarray(nbrs, dtype=int)
            d2   = np.sum((ctr[nbrs] - ctr[i]) ** 2, axis=1)
            w    = np.exp(-d2 / two_s2)
            m_out[i] = (w @ m[nbrs]) / w.sum()
        return m_out

    except ImportError:
        pass   # fall through to chunked dense path

    # ------------------------------------------------------------------
    # Fallback: chunked dense path capped at OP_SMOOTH_MAX_GB
    # ------------------------------------------------------------------
    max_bytes  = int(OP_SMOOTH_MAX_GB * 1024 ** 3)
    chunk_size = max(1, max_bytes // (n * 8))   # rows per chunk
    chunk_size = min(chunk_size, n)

    w_sum  = np.zeros(n, dtype=float)
    wm_sum = np.zeros(n, dtype=float)

    for start in range(0, n, chunk_size):
        end  = min(start + chunk_size, n)
        blk  = ctr[start:end]           # (chunk, 3)
        # d2[i, j] = ||blk[i] - ctr[j]||^2 — shape (chunk, n)
        d2   = (
            np.sum(blk ** 2, axis=1, keepdims=True)
            + np.sum(ctr ** 2, axis=1)[np.newaxis, :]
            - 2.0 * (blk @ ctr.T)
        )
        d2   = np.maximum(d2, 0.0)      # numerical safety
        mask = d2 <= radius ** 2
        W    = np.where(mask, np.exp(-d2 / two_s2), 0.0)
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


def _ellipsoid_mask(centroids: np.ndarray, center: list, axes: list,
                    angles_deg: list) -> np.ndarray:
    """Return boolean mask: True for centroids inside the rotated ellipsoid.

    Parameters
    ----------
    centroids : (n, 3)
    center    : [cx, cy, cz] in metres
    axes      : [a, b, c] semi-axes in metres (> 0)
    angles_deg: [α, β, γ] ZYX rotation angles in degrees

    Returns
    -------
    inside : (n,) bool
    """
    c  = np.asarray(center, dtype=float)
    ax = np.asarray(axes,   dtype=float)
    if np.any(ax <= 0.):
        raise ValueError("ellipsoid axes must all be > 0.")
    R = _rotation_matrix_zyx(angles_deg)
    # Transform points to ellipsoid-local frame: local_i = R^T (p_i - c)
    local = (centroids - c[np.newaxis, :]) @ R   # (n, 3)
    q = (local[:, 0] / ax[0])**2 + (local[:, 1] / ax[1])**2 + (local[:, 2] / ax[2])**2
    return q <= 1.0


def _op_ellipsoid(m: np.ndarray) -> np.ndarray:
    """Modify free-region log10(ρ) values inside a rotated ellipsoid.

    Uses geometry and mode stored in ``_ellipsoid_ctx`` (populated before
    dispatch).  Only regions whose centroid falls inside the ellipsoid are
    affected; all others are returned unchanged.

    Modes
    -----
    "replace"  m[inside] = OP_ELLIPSOID_VALUE
    "add"      m[inside] += OP_ELLIPSOID_VALUE
    """
    ctr    = _ellipsoid_ctx["centroids"]   # (n_free, 3)
    center = _ellipsoid_ctx["center"]
    axes   = _ellipsoid_ctx["axes"]
    angles = _ellipsoid_ctx["angles"]
    mode   = _ellipsoid_ctx["mode"]
    value  = _ellipsoid_ctx["value"]

    inside = _ellipsoid_mask(ctr, center, axes, angles)
    n_inside = int(inside.sum())
    if n_inside == 0:
        print("  ellipsoid: WARNING — no free regions found inside ellipsoid.")

    m_new = m.copy()
    if mode == "replace":
        m_new[inside] = float(value)
    elif mode == "add":
        m_new[inside] += float(value)
    else:
        raise ValueError(f"OP_ELLIPSOID_MODE must be 'replace' or 'add', got {mode!r}.")

    if OUT:
        print(f"  ellipsoid: {n_inside} of {len(m)} free regions modified "
              f"(mode='{mode}', value={value:+.3f}).")
    return m_new


_OPERATIONS: dict = {
    "fill":        _op_fill,
    "mean":        _op_mean,
    "median":      _op_median,
    "clip":        _op_clip,
    "shift":       _op_shift,
    "standardise": _op_standardise,
    "smooth":      _op_smooth,
    "ellipsoid":   _op_ellipsoid,
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
#: Module-level dicts populated here; consumed by _op_smooth / _op_ellipsoid.
_smooth_ctx:    dict = {}
_ellipsoid_ctx: dict = {}

_NEEDS_MESH = {"smooth", "ellipsoid"}

if OPERATION in _NEEDS_MESH:
    if not os.path.isfile(MESH_FILE):
        sys.exit(f"{OPERATION}: MESH_FILE not found: {MESH_FILE}")

    print(f"Reading mesh: {MESH_FILE}")
    nodes, conn = fem.read_femtic_mesh(MESH_FILE)
    print(f"  nodes={nodes.shape[0]}, elements={conn.shape[0]}")

    # Element→region mapping and free_idx — shared by both operations.
    _struct = fem._read_resistivity_block_struct(
        MODEL_IN, model_trans="log10", ocean=OCEAN, out=False
    )
    elem_region = _struct["elem_region"]   # (nelem,)
    free_idx    = _struct["free_idx"]      # (n_free,) region indices of free regions

    region_ctr = _build_region_centroids(nodes, conn, elem_region, free_idx)
    print(f"  region centroids: {region_ctr.shape[0]} free regions")
    print()

    if OPERATION == "smooth":
        _smooth_ctx["centroids"] = region_ctr
        _smooth_ctx["sigma"]     = float(OP_SMOOTH_SIGMA)
        print(f"  smooth context ready: σ={OP_SMOOTH_SIGMA:.0f} m, "
              f"cutoff={OP_SMOOTH_CUTOFF}σ={OP_SMOOTH_CUTOFF*OP_SMOOTH_SIGMA:.0f} m")

    if OPERATION == "ellipsoid":
        _mode = str(OP_ELLIPSOID_MODE).strip().lower()
        if _mode not in {"replace", "add"}:
            sys.exit(f"OP_ELLIPSOID_MODE must be 'replace' or 'add', got {_mode!r}.")
        _ellipsoid_ctx["centroids"] = region_ctr
        _ellipsoid_ctx["center"]    = list(OP_ELLIPSOID_CENTER)
        _ellipsoid_ctx["axes"]      = list(OP_ELLIPSOID_AXES)
        _ellipsoid_ctx["angles"]    = list(OP_ELLIPSOID_ANGLES)
        _ellipsoid_ctx["mode"]      = _mode
        _ellipsoid_ctx["value"]     = float(OP_ELLIPSOID_VALUE)
        print(f"  ellipsoid context ready: mode='{_mode}', value={OP_ELLIPSOID_VALUE:+.3f}")
        print(f"    center={OP_ELLIPSOID_CENTER}, axes={OP_ELLIPSOID_AXES}, "
              f"angles={OP_ELLIPSOID_ANGLES}")
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
