"""
femtic_cutaway.py
=================
Python translation of the FEMTIC resistivity-cutaway tool (main.cpp +
MeshData*.cpp + ResistivityBlock.cpp + Util.cpp).

Original C++ code © 2021 Yoshiya Usui, MIT License.
Python translation adds:
  • NumPy-based mesh/resistivity data structures (vectorised element loop)
  • NPZ output (``resistivity_GMT_iter<N>.npz``) alongside the original
    GMT text format (``resistivity_GMT_iter<N>.dat``)

Usage
-----
    python femtic_cutaway.py <param_file> [--mesh mesh.dat] [--resdir .]

NPZ output arrays
-----------------
The saved ``.npz`` file contains one record per polygon (element
cross-section), stored as ragged arrays encoded with an offset scheme:

  ``log10_rho``    float64 (N,)   – log10(resistivity) per polygon
  ``is_negative``  bool    (N,)   – True when the raw block rho was <0
                                    (sentinel value; clamped to 1e20 in
                                    log10_rho)
  ``coords_flat``  float64 (M,2)  – concatenated (X_km, Y_km) coordinates
  ``offsets``      int64   (N+1,) – coords_flat[offsets[i]:offsets[i+1]]
                                    is the open coordinate ring of polygon i
                                    (first point is NOT repeated)

  Provenance scalars (for reproducibility):
  ``center_km``       float64 (3,)  – cut-plane centre [X, Y, Z] in km
  ``rotation_deg``    float64 scalar
  ``plane_type``      int64   scalar  (0 = ZX, 1 = XY)
  ``iter_num``        int64   scalar
  ``element_type``    int64   scalar  (0 = Tetra, 1 = Brick, 2 = NCHexa)

Note – GMT writer vs NPZ ring convention
-----------------------------------------
``write_gmt`` repeats the first vertex to close each polygon ring (GMT
``-M`` / ``>`` segment convention).  ``write_npz`` stores open rings
(first vertex NOT repeated) to avoid redundancy; callers must close
rings themselves when needed.
"""

from __future__ import annotations

import math
import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants (from CommonParameters.h)
# ---------------------------------------------------------------------------
DEG2RAD = math.pi / 180.0

# Signed-distance epsilon for the degenerate (edge-in-plane) case
EPS_INTERSECT = 1.0e-12

# Epsilon for near-duplicate point removal (improvement #3)
EPS_DEDUP = 1.0e-6   # metres; points closer than this are considered equal

# Edge-to-local-node tables (from MeshData*Element.cpp constructors)
# Tetra: 6 edges
TETRA_EDGE2NODE: np.ndarray = np.array([
    [0, 1], [0, 2], [0, 3], [1, 2], [3, 1], [2, 3]
], dtype=np.intp)

# Brick / NonConformingHexa: 12 edges (same table for both)
HEXA_EDGE2NODE: np.ndarray = np.array([
    [0, 1], [3, 2], [4, 5], [7, 6],
    [0, 3], [4, 7], [1, 2], [5, 6],
    [0, 4], [1, 5], [3, 7], [2, 6],
], dtype=np.intp)

# Plane-type constants (from main.cpp)
ZX_PLANE = 0
XY_PLANE = 1

# Element-type constants
TETRA = 0
BRICK = 1
NONCONFORMING_HEXA = 2


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MeshData:
    """
    Raw mesh arrays read from mesh.dat.

    Attributes
    ----------
    mesh_type : str
        One of ``"TETRA"``, ``"HEXA"``, ``"DHEXA"``.
    node_xyz : ndarray, shape (n_node, 3)
        X, Y, Z coordinates of every node (metres).
    elem_nodes : ndarray, shape (n_elem, nodes_per_elem), dtype int
        Global node indices for every element.
    """
    mesh_type: str
    node_xyz: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    elem_nodes: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.intp))


@dataclass
class ResistivityBlock:
    """
    Resistivity block data read from ``resistivity_block_iter<N>.dat``.

    Attributes
    ----------
    elem2block : ndarray (n_elem,), int
        Block index for every mesh element.
    rho : ndarray (n_block,), float64
        Resistivity [Ω·m] per block.  Negative values are sentinels for
        the air layer (treated as 1 × 10²⁰ Ω·m downstream).
    fixed : ndarray (n_block,), bool
        True where the resistivity is held fixed during inversion.
    is_negative : ndarray (n_block,), bool
        True where ``rho`` is negative (air-layer sentinel).  Kept
        separately so callers can mask rather than inspect the raw sign.
        (improvement #2)
    """
    elem2block:  np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.intp))
    rho:         np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    fixed:       np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    is_negative: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))


# ---------------------------------------------------------------------------
# File readers  (improvement #9: read whole file then split, faster I/O)
# ---------------------------------------------------------------------------

def _tokens(path: str) -> List[str]:
    """Read *path* and return all whitespace-split tokens as a flat list."""
    with open(path, "r") as fh:
        return fh.read().split()


class _TokenStream:
    """Thin wrapper around a pre-loaded token list with a cursor."""
    __slots__ = ("_toks", "_pos")

    def __init__(self, tokens: List[str]) -> None:
        self._toks = tokens
        self._pos = 0

    def next(self) -> str:
        tok = self._toks[self._pos]
        self._pos += 1
        return tok

    def next_int(self) -> int:
        return int(self.next())

    def next_float(self) -> float:
        return float(self.next())

    def skip(self, n: int) -> None:
        self._pos += n


# ---------------------------------------------------------------------------
# Mesh readers
# ---------------------------------------------------------------------------

def read_mesh(path: str = "mesh.dat") -> MeshData:
    """
    Read *path* (default ``mesh.dat``) and return a :class:`MeshData`.

    The file format is auto-detected from the header keyword
    (``TETRA``, ``HEXA``, ``DHEXA``).
    """
    ts = _TokenStream(_tokens(path))
    header = ts.next().upper()
    if header.startswith("TETRA"):
        return _read_mesh_tetra(ts)
    elif header.startswith("DHEXA"):
        return _read_mesh_nonconforming_hexa(ts)
    elif header.startswith("HEXA"):
        return _read_mesh_brick(ts)
    else:
        raise ValueError(f"Unknown mesh type in {path!r}: {header!r}")


def _read_nodes(ts: _TokenStream, n_node: int) -> np.ndarray:
    """Read *n_node* lines of ``index x y z`` and return (n_node, 3) array."""
    xyz = np.empty((n_node, 3), dtype=np.float64)
    for i in range(n_node):
        ts.skip(1)                  # index (ignored)
        xyz[i, 0] = ts.next_float()
        xyz[i, 1] = ts.next_float()
        xyz[i, 2] = ts.next_float()
    return xyz


def _read_mesh_tetra(ts: _TokenStream) -> MeshData:
    n_node = ts.next_int()
    xyz = _read_nodes(ts, n_node)

    n_elem = ts.next_int()
    nodes = np.empty((n_elem, 4), dtype=np.intp)
    for i in range(n_elem):
        ts.skip(1 + 4)              # elem index + 4 neighbour IDs
        for j in range(4):
            nodes[i, j] = ts.next_int()

    # Boundary planes (6 × variable length) — skip
    for _ in range(6):
        n = ts.next_int()
        ts.skip(n * 2)              # elemID + faceID per entry

    # Land surface — skip
    n_land = ts.next_int()
    ts.skip(n_land * 2)

    return MeshData(mesh_type="TETRA", node_xyz=xyz, elem_nodes=nodes)


def _read_mesh_brick(ts: _TokenStream) -> MeshData:
    """
    Read a HEXA (conforming brick) mesh.

    The ``old_format`` flag for the resistivity-block file (improvement #1)
    is a separate concern handled in :func:`read_resistivity_block`; it does
    not affect mesh parsing.
    """
    nX = ts.next_int()
    nY = ts.next_int()
    nZ = ts.next_int()
    ts.skip(1)                      # n_air_layer

    n_node = (nX + 1) * (nY + 1) * (nZ + 1)
    xyz = _read_nodes(ts, n_node)

    n_elem = nX * nY * nZ
    nodes = np.empty((n_elem, 8), dtype=np.intp)
    for i in range(n_elem):
        ts.skip(1 + 6)              # elem index + 6 neighbour IDs
        for j in range(8):
            nodes[i, j] = ts.next_int()

    # Boundary planes — skip (elemID + 4 nodeIDs per entry)
    for _ in range(6):
        n = ts.next_int()
        ts.skip(n * 5)

    return MeshData(mesh_type="HEXA", node_xyz=xyz, elem_nodes=nodes)


def _read_mesh_nonconforming_hexa(ts: _TokenStream) -> MeshData:
    n_node = ts.next_int()
    xyz = _read_nodes(ts, n_node)

    n_elem = ts.next_int()
    nodes = np.empty((n_elem, 8), dtype=np.intp)
    for i in range(n_elem):
        ts.skip(1)                  # elem index
        for j in range(8):
            nodes[i, j] = ts.next_int()
        # Variable-count neighbour lists — skip
        for _ in range(6):
            n_face = ts.next_int()
            ts.skip(n_face)

    # Boundary planes — skip
    for _ in range(6):
        n = ts.next_int()
        ts.skip(n * 2)

    # Land surface — skip
    n_land = ts.next_int()
    ts.skip(n_land * 2)

    return MeshData(mesh_type="DHEXA", node_xyz=xyz, elem_nodes=nodes)


# ---------------------------------------------------------------------------
# Resistivity block reader  (improvement #1: old_format support)
# ---------------------------------------------------------------------------

def read_resistivity_block(
    iter_num: int,
    resdir: str = ".",
    old_format: bool = False,
) -> ResistivityBlock:
    """
    Read ``<resdir>/resistivity_block_iter<iter_num>.dat``.

    Parameters
    ----------
    iter_num : int
        Inversion iteration number.
    resdir : str
        Directory that contains the resistivity-block files.
        Defaults to the current working directory.  (improvement #7)
    old_format : bool
        If True, parse the ``#ifdef _OLD`` layout: each block line has
        only 3 tokens (``index  rho  ifix``).  If False (default), the
        new layout with 6 tokens is expected:
        ``index  rho  d d d  ifix``.  (improvement #1)
    """
    import os
    path = os.path.join(resdir, f"resistivity_block_iter{iter_num}.dat")
    ts = _TokenStream(_tokens(path))

    n_elem = ts.next_int()
    n_blk  = ts.next_int()

    elem2block = np.empty(n_elem, dtype=np.intp)
    for i in range(n_elem):
        ts.skip(1)                  # index
        elem2block[i] = ts.next_int()

    rho         = np.empty(n_blk, dtype=np.float64)
    fixed       = np.zeros(n_blk, dtype=bool)
    is_negative = np.zeros(n_blk, dtype=bool)

    for i in range(n_blk):
        ts.skip(1)                  # index
        rho[i] = ts.next_float()
        if old_format:
            # #ifdef _OLD layout: ``index  rho  ifix``
            ifix = ts.next_int()
        else:
            # new layout: ``index  rho  d d d  ifix``
            ts.skip(3)
            ifix = ts.next_int()
        fixed[i]       = ifix == 1
        is_negative[i] = rho[i] < 0

    if not fixed[0]:
        raise RuntimeError("Resistivity block 0 must be fixed (air layer).")

    return ResistivityBlock(
        elem2block=elem2block,
        rho=rho,
        fixed=fixed,
        is_negative=is_negative,
    )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def calc_normal_vector(plane_type: int, rotation_angle_rad: float) -> np.ndarray:
    """Return the unit normal to the cutting plane (shape ``(3,)``)."""
    if plane_type == ZX_PLANE:
        # Y-axis rotated by –angle in the XY plane
        a = -rotation_angle_rad
        return np.array([-math.sin(a), math.cos(a), 0.0])
    elif plane_type == XY_PLANE:
        return np.array([0.0, 0.0, -1.0])
    else:
        raise ValueError(f"Unknown plane type: {plane_type}")


def _project_batch(
    pts3d: np.ndarray,
    plane_type: int,
    center: np.ndarray,
    normal: np.ndarray,
    rotation_angle_rad: float,
) -> np.ndarray:
    """
    Project an (M, 3) array of 3-D points to (M, 2) 2-D plot coordinates
    (metres).  Vectorised over all points simultaneously.
    """
    if plane_type == ZX_PLANE:
        x2 = normal[1] * (pts3d[:, 0] - center[0]) - normal[0] * (pts3d[:, 1] - center[1])
        y2 = pts3d[:, 2]
    elif plane_type == XY_PLANE:
        a  = -rotation_angle_rad
        vx = pts3d[:, 1]
        vy = pts3d[:, 0]
        x2 = vx * math.cos(a) - vy * math.sin(a)
        y2 = vx * math.sin(a) + vy * math.cos(a)
    else:
        raise ValueError(f"Unknown plane type: {plane_type}")
    return np.stack([x2, y2], axis=1)


def _delete_near_duplicate_points(pts: np.ndarray) -> np.ndarray:
    """
    Remove near-duplicate 2-D points using an epsilon threshold
    (``EPS_DEDUP``).

    Improvement #3: replaces the original exact float-equality deduplication
    (C++ ``std::unique`` on sorted pairs) with a rounded-grid approach so
    that nearly-coincident intersection points from degenerate edges are
    merged correctly.  In the limit ``EPS_DEDUP → 0`` the behaviour
    matches the C++ original exactly.
    """
    if len(pts) == 0:
        return pts
    scale   = 1.0 / EPS_DEDUP
    snapped = np.round(pts * scale).astype(np.int64)
    # Lexicographic sort (matches C++ sort order on pairs)
    order   = np.lexsort((snapped[:, 1], snapped[:, 0]))
    snapped = snapped[order]
    pts     = pts[order]
    # Keep only the first occurrence of each snapped key
    if len(snapped) == 1:
        return pts
    diff = np.any(snapped[1:] != snapped[:-1], axis=1)
    keep = np.concatenate([[True], diff])
    return pts[keep]


def _reorder_points(pts: np.ndarray) -> np.ndarray:
    """
    Sort 2-D points by angle around their centroid.

    Note: this produces a valid (non-self-intersecting) polygon only for
    **convex** cross-sections.  Non-convex element slices may yield
    self-intersecting rings — the same limitation exists in the original
    C++ ``reorderPoints``.  (improvement #10)
    """
    if len(pts) == 0:
        return pts
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    return pts[np.argsort(angles)]


# ---------------------------------------------------------------------------
# Vectorised cutaway  (improvement #4)
# ---------------------------------------------------------------------------

def make_cutaway(
    mesh: MeshData,
    res_block: ResistivityBlock,
    plane_type: int,
    center_m: np.ndarray,
    rotation_angle_rad: float,
    excluded_blocks: List[int],
) -> Tuple[List[np.ndarray], List[float], List[bool]]:
    """
    Compute resistivity cross-section polygons for a cutting plane.

    The inner edge-intersection loop is fully vectorised with NumPy:
    all edges of all non-excluded elements are gathered in one index
    operation, signed distances are computed in one dot-product, and
    boolean masks select crossing / degenerate edges without any Python
    loop over edges.  Per-element polygon assembly (deduplication,
    angular reordering) is still done element-by-element because the
    number of surviving intersection points varies per element.
    (improvement #4)

    Parameters
    ----------
    mesh : MeshData
    res_block : ResistivityBlock
    plane_type : int
        ``ZX_PLANE`` or ``XY_PLANE``.
    center_m : ndarray (3,)
        Centre of the cutting plane in **metres**.
    rotation_angle_rad : float
    excluded_blocks : list of int
        Block IDs whose elements are skipped entirely.

    Returns
    -------
    polygons : list of (K, 2) float64 arrays
        Coordinates in **kilometres**, open rings (first point not repeated).
    log10_rho_list : list of float
        log10(resistivity [Ω·m]) per polygon; negative-sentinel blocks
        contribute log10(1 × 10²⁰).
    is_negative_list : list of bool
        True when the underlying block had rho < 0 (sentinel).
        (improvement #2)
    """
    normal       = calc_normal_vector(plane_type, rotation_angle_rad)
    excluded_set = set(excluded_blocks)
    edge_table   = TETRA_EDGE2NODE if mesh.mesh_type == "TETRA" else HEXA_EDGE2NODE

    # node_xyz already lives in MeshData (improvement #5)
    node_xyz = mesh.node_xyz           # (n_node, 3)
    n_elem   = mesh.elem_nodes.shape[0]

    # ---- Build element mask -----------------------------------------------
    block_ids = res_block.elem2block   # (n_elem,)
    elem_mask = np.ones(n_elem, dtype=bool)
    for bid in excluded_set:
        elem_mask &= (block_ids != bid)
    active_elems = np.where(elem_mask)[0]
    if len(active_elems) == 0:
        return [], [], []

    # ---- Gather edge endpoint coordinates for all active elements ---------
    local0 = edge_table[:, 0]          # (n_edges,)
    local1 = edge_table[:, 1]

    elem_node_matrix = mesh.elem_nodes[active_elems]   # (n_active, nodes_per_elem)
    gnid0 = elem_node_matrix[:, local0]                # (n_active, n_edges)
    gnid1 = elem_node_matrix[:, local1]

    c0 = node_xyz[gnid0]               # (n_active, n_edges, 3)
    c1 = node_xyz[gnid1]

    # ---- Signed distances from the plane ----------------------------------
    ip0 = (c0 - center_m) @ normal     # (n_active, n_edges)
    ip1 = (c1 - center_m) @ normal

    # ---- Classify edges ---------------------------------------------------
    same_side = ip0 * ip1 > 0          # no intersection
    in_plane  = (np.abs(ip0) + np.abs(ip1)) <= EPS_INTERSECT   # degenerate
    crossing  = ~same_side             # includes in_plane

    # ---- Interpolated intersection points ---------------------------------
    s      = np.abs(ip0) + np.abs(ip1)
    s_safe = np.where(s > EPS_INTERSECT, s, 1.0)   # avoid division by zero
    ratio  = np.abs(ip0) / s_safe                  # (n_active, n_edges)
    p_cross = c0 + ratio[:, :, np.newaxis] * (c1 - c0)   # (n_active, n_edges, 3)

    # ---- Per-element polygon assembly -------------------------------------
    polygons:         List[np.ndarray] = []
    log10_rho_list:   List[float]      = []
    is_negative_list: List[bool]       = []

    for k, iElem in enumerate(active_elems):
        cross_mask = crossing[k]       # (n_edges,) bool
        deg_mask   = in_plane[k]
        reg_mask   = cross_mask & ~deg_mask

        pts3d_parts = []
        if reg_mask.any():
            pts3d_parts.append(p_cross[k][reg_mask])    # one point per edge
        if deg_mask.any():
            # Both endpoints lie in the plane — add both.
            # C++ original (bug: coord0[0] used instead of coord1[0] for p2.X):
            #   p2 = np.array([c0[0], c1[1], c1[2]])   # c0[0] should be c1[0]
            pts3d_parts.append(c0[k][deg_mask])
            pts3d_parts.append(c1[k][deg_mask])

        if not pts3d_parts:
            continue

        pts3d = np.concatenate(pts3d_parts, axis=0)   # (M, 3)
        if len(pts3d) < 3:
            continue

        pts2d = _project_batch(pts3d, plane_type, center_m, normal, rotation_angle_rad)
        pts2d = _delete_near_duplicate_points(pts2d)
        if len(pts2d) < 3:
            continue
        pts2d = _reorder_points(pts2d)
        pts2d *= 1e-3                  # metres → kilometres

        bid     = int(block_ids[iElem])
        neg     = bool(res_block.is_negative[bid])
        rho_eff = 1.0e20 if neg else float(res_block.rho[bid])
        log10_rho = math.log10(rho_eff)

        polygons.append(pts2d)
        log10_rho_list.append(log10_rho)
        is_negative_list.append(neg)

    return polygons, log10_rho_list, is_negative_list


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_gmt(
    path: str,
    polygons: List[np.ndarray],
    log10_rho_list: List[float],
) -> None:
    """
    Write the classic GMT multi-segment text output.

    Each polygon is written as a ``> -Z <log10_rho>`` segment.  The first
    vertex is repeated at the end of each segment to close the ring
    (GMT convention).

    Note: the NPZ writer stores open rings (first vertex NOT repeated).
    See module docstring for the ring-closure convention difference.
    (improvement #12)
    """
    with open(path, "w") as fh:
        for poly, lr in zip(polygons, log10_rho_list):
            fh.write(f"> -Z {lr:15.6e}\n")
            for xy in poly:
                fh.write(f"{xy[0]:15.6e}{xy[1]:15.6e}\n")
            # Repeat first vertex to close the ring (GMT convention)
            fh.write(f"{poly[0, 0]:15.6e}{poly[0, 1]:15.6e}\n")
    print(f"Wrote GMT file: {path}")


def write_npz(
    path: str,
    polygons: List[np.ndarray],
    log10_rho_list: List[float],
    is_negative_list: List[bool],
    params: Optional[dict] = None,
) -> None:
    """
    Write a compressed NPZ file.

    Arrays
    ------
    log10_rho    float64 (N,)   – log10(Ω·m) per polygon
    is_negative  bool    (N,)   – True when raw rho was < 0 (sentinel)
    coords_flat  float64 (M,2)  – all vertices concatenated (km)
    offsets      int64   (N+1,) – CSR-style offsets; polygon i occupies
                                   coords_flat[offsets[i]:offsets[i+1]]
                                   (open rings: first point NOT repeated)

    Provenance scalars (from *params* dict, improvement #8):
    center_km       float64 (3,)
    rotation_deg    float64 scalar
    plane_type      int64   scalar
    iter_num        int64   scalar
    element_type    int64   scalar
    """
    n             = len(polygons)
    log10_rho_arr = np.array(log10_rho_list,  dtype=np.float64)
    is_neg_arr    = np.array(is_negative_list, dtype=bool)

    offsets = np.zeros(n + 1, dtype=np.int64)
    for i, poly in enumerate(polygons):
        offsets[i + 1] = offsets[i] + len(poly)
    total       = int(offsets[-1])
    coords_flat = np.empty((total, 2), dtype=np.float64)
    for i, poly in enumerate(polygons):
        coords_flat[offsets[i]:offsets[i + 1]] = poly

    save_kwargs: dict = dict(
        log10_rho   = log10_rho_arr,
        is_negative = is_neg_arr,
        coords_flat = coords_flat,
        offsets     = offsets,
    )

    # Embed provenance scalars so the file is self-describing (improvement #8)
    if params is not None:
        save_kwargs["center_km"]    = np.asarray(params.get("center_km",    [0, 0, 0]), dtype=np.float64)
        save_kwargs["rotation_deg"] = np.float64( params.get("rotation_deg", 0.0))
        save_kwargs["plane_type"]   = np.int64(   params.get("plane_type",   -1))
        save_kwargs["iter_num"]     = np.int64(   params.get("iter_num",     -1))
        save_kwargs["element_type"] = np.int64(   params.get("element_type", -1))

    np.savez_compressed(path, **save_kwargs)
    print(f"Wrote NPZ file: {path}.npz  ({n} polygons, {total} vertices)")


# ---------------------------------------------------------------------------
# Parameter-file reader
# ---------------------------------------------------------------------------

def read_param_file(path: str) -> dict:
    """
    Parse a FEMTIC cutaway parameter file.

    Returns a dict with keys:
    ``element_type``, ``iter_num``, ``plane_type``, ``center_km``
    (ndarray), ``rotation_deg``, ``excluded_blocks``.
    """
    ts = _TokenStream(_tokens(path))

    element_type = ts.next_int()
    iter_num     = ts.next_int()
    plane_type   = ts.next_int()
    cx, cy, cz   = ts.next_float(), ts.next_float(), ts.next_float()
    center_km    = np.array([cx, cy, cz])
    rotation_deg = ts.next_float()
    n_excl       = ts.next_int()
    excluded     = [ts.next_int() for _ in range(n_excl)]

    etype_names = {TETRA: "Tetra", BRICK: "Brick", NONCONFORMING_HEXA: "Nonconforming Hexa"}
    ptype_names = {ZX_PLANE: "ZX plane", XY_PLANE: "XY plane"}
    print(f"Element type      : {etype_names.get(element_type, element_type)}")
    print(f"Iteration number  : {iter_num}")
    print(f"Plane type        : {ptype_names.get(plane_type, plane_type)}")
    print(f"Center coord (km) : X={cx}, Y={cy}, Z={cz}")
    print(f"Rotation angle    : {rotation_deg} deg")
    print(f"Excluded blocks   : {excluded}")

    return dict(
        element_type    = element_type,
        iter_num        = iter_num,
        plane_type      = plane_type,
        center_km       = center_km,
        rotation_deg    = rotation_deg,
        excluded_blocks = excluded,
    )


# ---------------------------------------------------------------------------
# Entry point  (improvements #6, #7: mesh_path and resdir wired through)
# ---------------------------------------------------------------------------

def run(
    param_file: str,
    mesh_path: str = "mesh.dat",
    resdir: str = ".",
    old_format: bool = False,
) -> None:
    """
    Run the full cutaway pipeline.

    Parameters
    ----------
    param_file : str
        Path to the FEMTIC parameter file.
    mesh_path : str
        Path to the mesh file (default: ``"mesh.dat"`` in the current
        working directory).  (improvement #6)
    resdir : str
        Directory containing ``resistivity_block_iter<N>.dat`` files.
        (improvements #6, #7)
    old_format : bool
        Pass True to use the ``#ifdef _OLD`` resistivity-block format.
        (improvement #1)
    """
    params = read_param_file(param_file)

    center_m     = params["center_km"] * 1000.0
    rotation_rad = params["rotation_deg"] * DEG2RAD

    print(f"Reading {mesh_path} …")
    mesh = read_mesh(mesh_path)
    print(f"  mesh type: {mesh.mesh_type}, "
          f"nodes: {len(mesh.node_xyz)}, elements: {len(mesh.elem_nodes)}")

    rblk_name = f"resistivity_block_iter{params['iter_num']}.dat"
    print(f"Reading {resdir}/{rblk_name} …")
    res_block = read_resistivity_block(
        params["iter_num"], resdir=resdir, old_format=old_format)
    print(f"  {len(res_block.rho)} blocks, "
          f"{int(res_block.fixed.sum())} fixed, "
          f"{int(res_block.is_negative.sum())} negative-sentinel")

    print("Computing cutaway …")
    polygons, log10_rho_list, is_negative_list = make_cutaway(
        mesh, res_block,
        params["plane_type"], center_m, rotation_rad,
        params["excluded_blocks"],
    )
    print(f"  {len(polygons)} polygons found")

    stem = f"resistivity_GMT_iter{params['iter_num']}"
    write_gmt(stem + ".dat", polygons, log10_rho_list)
    write_npz(stem, polygons, log10_rho_list, is_negative_list, params=params)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FEMTIC resistivity cutaway tool (Python port)")
    parser.add_argument("param_file",
                        help="Parameter file (same format as C++ version)")
    parser.add_argument("--mesh", default="mesh.dat", metavar="PATH",
                        help="Path to mesh.dat (default: mesh.dat)")
    parser.add_argument("--resdir", default=".", metavar="DIR",
                        help="Directory containing resistivity_block_iter*.dat "
                             "(default: current directory)")
    parser.add_argument("--old-format", action="store_true",
                        help="Use the legacy (#ifdef _OLD) resistivity-block format")
    args = parser.parse_args()
    run(args.param_file,
        mesh_path  = args.mesh,
        resdir     = args.resdir,
        old_format = args.old_format)


if __name__ == "__main__":
    main()
