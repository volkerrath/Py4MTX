"""
femtic_cutaway.py
=================
Python translation of the FEMTIC resistivity-cutaway tool (main.cpp +
MeshData*.cpp + ResistivityBlock.cpp + Util.cpp).

Original C++ code © 2021 Yoshiya Usui, MIT License.
Python translation adds:
  • NumPy-based mesh/resistivity data structures
  • NPZ output (``resistivity_GMT_iter<N>.npz``) in addition to the
    original GMT text format (``resistivity_GMT_iter<N>.dat``)

Usage
-----
    python femtic_cutaway.py <param_file>

NPZ output arrays
-----------------
The saved ``.npz`` file contains one record per polygon (element
cross-section), stored as ragged arrays encoded with an offset scheme:

  ``log10_rho``   float64 (N,)   – log10(resistivity) per polygon
  ``coords_flat`` float64 (M,2)  – concatenated (X_km, Z_km) coordinates
  ``offsets``     int64   (N+1,) – coords_flat[offsets[i]:offsets[i+1]]
                                    is the coordinate ring of polygon i

The first coordinate of each ring is NOT repeated (the ring is implicitly
closed). Use ``np.savez_compressed`` so the file is space-efficient.
"""

from __future__ import annotations

import sys
import math
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants (from CommonParameters.h)
# ---------------------------------------------------------------------------
DEG2RAD = math.pi / 180.0
EPS_INTERSECT = 1.0e-12

# Edge-to-local-node tables (from MeshData*Element.cpp constructors)
# Tetra: 6 edges, each has 2 node indices in the local 4-node element
TETRA_EDGE2NODE: List[Tuple[int, int]] = [
    (0, 1), (0, 2), (0, 3), (1, 2), (3, 1), (2, 3)
]

# Brick / NonConformingHexa: 12 edges, same table for both mesh types
HEXA_EDGE2NODE: List[Tuple[int, int]] = [
    (0, 1), (3, 2), (4, 5), (7, 6),
    (0, 3), (4, 7), (1, 2), (5, 6),
    (0, 4), (1, 5), (3, 7), (2, 6),
]

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
    """Holds raw mesh arrays read from mesh.dat."""
    mesh_type: str                          # "TETRA", "HEXA", "DHEXA"
    node_x: np.ndarray = field(default_factory=lambda: np.empty(0))
    node_y: np.ndarray = field(default_factory=lambda: np.empty(0))
    node_z: np.ndarray = field(default_factory=lambda: np.empty(0))
    elem_nodes: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=int))
    # elem_nodes shape: (n_elem, nodes_per_elem)


@dataclass
class ResistivityBlock:
    """Holds resistivity block data read from resistivity_block_iter<N>.dat."""
    elem2block: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    rho: np.ndarray = field(default_factory=lambda: np.empty(0))
    fixed: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))


# ---------------------------------------------------------------------------
# File readers
# ---------------------------------------------------------------------------

def _tokenize(path: str):
    """Generator that yields whitespace-split tokens from a text file."""
    with open(path, "r") as fh:
        for line in fh:
            for tok in line.split():
                yield tok


def read_mesh(path: str = "mesh.dat") -> MeshData:
    """Read mesh.dat and return a MeshData instance."""
    toks = _tokenize(path)

    header = next(toks)
    mesh_type = header[:5].upper()   # "TETRA", "HEXA " -> "HEXA", "DHEXA"
    mesh_type = mesh_type.rstrip()

    if mesh_type.startswith("TETRA"):
        return _read_mesh_tetra(toks)
    elif mesh_type.startswith("DHEXA"):
        return _read_mesh_nonconforming_hexa(toks)
    elif mesh_type.startswith("HEXA"):
        return _read_mesh_brick(toks)
    else:
        raise ValueError(f"Unknown mesh type in mesh.dat: {header!r}")


def _read_mesh_tetra(toks) -> MeshData:
    n_node = int(next(toks))
    xs, ys, zs = np.empty(n_node), np.empty(n_node), np.empty(n_node)
    for i in range(n_node):
        _idum = int(next(toks))
        xs[i] = float(next(toks))
        ys[i] = float(next(toks))
        zs[i] = float(next(toks))

    n_elem = int(next(toks))
    nodes = np.empty((n_elem, 4), dtype=int)
    for i in range(n_elem):
        _idum = int(next(toks))
        # 4 neighbour IDs (skip)
        for _ in range(4):
            next(toks)
        # 4 node IDs
        for j in range(4):
            nodes[i, j] = int(next(toks))

    # Boundary planes (skip — not needed for cutaway)
    for _plane in range(6):
        n_on_plane = int(next(toks))
        for _ in range(n_on_plane):
            next(toks); next(toks)  # elemID, faceID

    # Land surface (skip)
    n_land = int(next(toks))
    for _ in range(n_land):
        next(toks); next(toks)

    return MeshData(mesh_type="TETRA", node_x=xs, node_y=ys, node_z=zs, elem_nodes=nodes)


def _read_mesh_brick(toks) -> MeshData:
    nX = int(next(toks))
    nY = int(next(toks))
    nZ = int(next(toks))
    _n_air = int(next(toks))

    n_node = (nX + 1) * (nY + 1) * (nZ + 1)
    xs, ys, zs = np.empty(n_node), np.empty(n_node), np.empty(n_node)
    for i in range(n_node):
        _idum = int(next(toks))
        xs[i] = float(next(toks))
        ys[i] = float(next(toks))
        zs[i] = float(next(toks))

    n_elem = nX * nY * nZ
    nodes = np.empty((n_elem, 8), dtype=int)
    for i in range(n_elem):
        _idum = int(next(toks))
        # 6 neighbour IDs (skip)
        for _ in range(6):
            next(toks)
        # 8 node IDs
        for j in range(8):
            nodes[i, j] = int(next(toks))

    # Boundary planes (skip)
    for _plane in range(6):
        n_on_plane = int(next(toks))
        for _ in range(n_on_plane):
            next(toks)           # elemID
            for _ in range(4):
                next(toks)       # 4 nodeIDs

    return MeshData(mesh_type="HEXA", node_x=xs, node_y=ys, node_z=zs, elem_nodes=nodes)


def _read_mesh_nonconforming_hexa(toks) -> MeshData:
    n_node = int(next(toks))
    xs, ys, zs = np.empty(n_node), np.empty(n_node), np.empty(n_node)
    for i in range(n_node):
        _idum = int(next(toks))
        xs[i] = float(next(toks))
        ys[i] = float(next(toks))
        zs[i] = float(next(toks))

    n_elem = int(next(toks))
    nodes = np.empty((n_elem, 8), dtype=int)
    for i in range(n_elem):
        _idum = int(next(toks))
        # 8 node IDs
        for j in range(8):
            nodes[i, j] = int(next(toks))
        # Variable-count neighbour lists (skip)
        for _ in range(6):
            n_face = int(next(toks))
            for _ in range(n_face):
                next(toks)

    # Boundary planes (skip)
    for _plane in range(6):
        n_on_plane = int(next(toks))
        for _ in range(n_on_plane):
            next(toks); next(toks)

    # Land surface (skip)
    n_land = int(next(toks))
    for _ in range(n_land):
        next(toks); next(toks)

    return MeshData(mesh_type="DHEXA", node_x=xs, node_y=ys, node_z=zs, elem_nodes=nodes)


def read_resistivity_block(iter_num: int) -> ResistivityBlock:
    """Read resistivity_block_iter<N>.dat."""
    path = f"resistivity_block_iter{iter_num}.dat"
    toks = _tokenize(path)

    n_elem = int(next(toks))
    n_blk = int(next(toks))

    elem2block = np.empty(n_elem, dtype=int)
    for i in range(n_elem):
        _idum = int(next(toks))
        elem2block[i] = int(next(toks))

    rho = np.empty(n_blk)
    fixed = np.zeros(n_blk, dtype=bool)
    for i in range(n_blk):
        _idum = int(next(toks))
        rho[i] = float(next(toks))
        # skip three dummy doubles (added in non-_OLD format)
        try:
            _d1 = float(next(toks))
            _d2 = float(next(toks))
            _d3 = float(next(toks))
            ifix = int(next(toks))
        except (StopIteration, ValueError):
            ifix = 0
        fixed[i] = ifix == 1

    # Guard: block 0 must be air (fixed)
    if not fixed[0]:
        raise RuntimeError("Resistivity block 0 must be fixed (air layer).")

    return ResistivityBlock(elem2block=elem2block, rho=rho, fixed=fixed)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def calc_normal_vector(plane_type: int, rotation_angle_rad: float) -> np.ndarray:
    """Return the unit normal to the cutting plane."""
    if plane_type == ZX_PLANE:
        # Original Y-axis normal, rotated by -angle in XY plane
        a = -rotation_angle_rad
        return np.array([math.cos(a) * 0.0 - math.sin(a) * 1.0,
                         math.sin(a) * 0.0 + math.cos(a) * 1.0,
                         0.0])
    elif plane_type == XY_PLANE:
        return np.array([0.0, 0.0, -1.0])
    else:
        raise ValueError(f"Unknown plane type: {plane_type}")


def calc_intersect_point(
    c0: np.ndarray,
    c1: np.ndarray,
    center: np.ndarray,
    normal: np.ndarray,
) -> List[np.ndarray]:
    """
    Return 0, 1, or 2 intersection points of edge (c0,c1) with the plane
    defined by (center, normal).  Matches C++ calcIntersectPoint exactly.
    """
    v0 = c0 - center
    v1 = c1 - center
    ip0 = float(np.dot(v0, normal))
    ip1 = float(np.dot(v1, normal))

    if ip0 * ip1 > 0:           # both on same side → no intersection
        return []

    s = abs(ip0) + abs(ip1)
    if s > EPS_INTERSECT:
        ratio = abs(ip0) / s
        return [c0 + ratio * (c1 - c0)]
    else:
        # Edge lies in the plane (degenerate) — return both endpoints.
        # C++ original (bug: coord0[0] used instead of coord1[0] for p2.X):
        #   p1 = np.array([c0[0], c0[1], c0[2]])
        #   p2 = np.array([c0[0], c1[1], c1[2]])   # c0[0] should be c1[0]
        p1 = np.array([c0[0], c0[1], c0[2]])
        p2 = np.array([c1[0], c1[1], c1[2]])
        return [p1, p2]


def project_to_2d(
    p3d: np.ndarray,
    plane_type: int,
    center: np.ndarray,
    normal: np.ndarray,
    rotation_angle_rad: float,
) -> np.ndarray:
    """Project a 3-D intersection point to 2-D plot coordinates (metres)."""
    if plane_type == ZX_PLANE:
        x2 = normal[1] * (p3d[0] - center[0]) - normal[0] * (p3d[1] - center[1])
        y2 = p3d[2]
    elif plane_type == XY_PLANE:
        vx = p3d[1]
        vy = p3d[0]
        a = -rotation_angle_rad
        x2 = vx * math.cos(a) - vy * math.sin(a)
        y2 = vx * math.sin(a) + vy * math.cos(a)
    else:
        raise ValueError(f"Unknown plane type: {plane_type}")
    return np.array([x2, y2])


def delete_same_points(pts: np.ndarray) -> np.ndarray:
    """Remove duplicate 2-D points (matches C++ deleteSamePoints)."""
    if len(pts) == 0:
        return pts
    # Round-trip through set of tuples for exact duplicate removal
    seen = {}
    for p in pts:
        key = (p[0], p[1])
        seen[key] = p
    unique = sorted(seen.keys())   # sort as C++ does
    return np.array([list(k) for k in unique])


def reorder_points(pts: np.ndarray) -> np.ndarray:
    """Sort 2-D points by angle around their centroid (convex hull order)."""
    if len(pts) == 0:
        return pts
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    order = np.argsort(angles)
    return pts[order]


# ---------------------------------------------------------------------------
# Core cutaway logic
# ---------------------------------------------------------------------------

def make_cutaway(
    mesh: MeshData,
    res_block: ResistivityBlock,
    plane_type: int,
    center_m: np.ndarray,
    rotation_angle_rad: float,
    excluded_blocks: List[int],
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Compute the cross-section polygons.

    Returns
    -------
    polygons : list of (K, 2) float arrays, coordinates in **kilometres**
    log10_rho_list : list of floats, log10(resistivity) per polygon
    """
    normal = calc_normal_vector(plane_type, rotation_angle_rad)
    excluded_set = set(excluded_blocks)

    # Choose edge table based on mesh type
    if mesh.mesh_type == "TETRA":
        edge_table = TETRA_EDGE2NODE
    else:
        edge_table = HEXA_EDGE2NODE

    node_xyz = np.stack([mesh.node_x, mesh.node_y, mesh.node_z], axis=1)  # (N, 3)
    n_elem = mesh.elem_nodes.shape[0]

    polygons: List[np.ndarray] = []
    log10_rho_list: List[float] = []

    for iElem in range(n_elem):
        block_id = int(res_block.elem2block[iElem])
        if block_id in excluded_set:
            continue

        pts2d: List[np.ndarray] = []
        for local0, local1 in edge_table:
            nid0 = mesh.elem_nodes[iElem, local0]
            nid1 = mesh.elem_nodes[iElem, local1]
            c0 = node_xyz[nid0]
            c1 = node_xyz[nid1]
            for p3d in calc_intersect_point(c0, c1, center_m, normal):
                pts2d.append(project_to_2d(p3d, plane_type, center_m, normal, rotation_angle_rad))

        if len(pts2d) < 3:
            continue

        arr = np.array(pts2d)
        arr = delete_same_points(arr)
        if len(arr) < 3:
            continue
        arr = reorder_points(arr)
        arr *= 1e-3   # metres → kilometres

        rho_val = float(res_block.rho[block_id])
        if rho_val < 0:
            rho_val = 1.0e20
        log10_rho = math.log10(rho_val)

        polygons.append(arr)
        log10_rho_list.append(log10_rho)

    return polygons, log10_rho_list


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_gmt(
    path: str,
    polygons: List[np.ndarray],
    log10_rho_list: List[float],
) -> None:
    """Write the classic GMT text output (``resistivity_GMT_iter<N>.dat``)."""
    with open(path, "w") as fh:
        for poly, lr in zip(polygons, log10_rho_list):
            fh.write(f"> -Z {lr:15.6e}\n")
            for xy in poly:
                fh.write(f"{xy[0]:15.6e}{xy[1]:15.6e}\n")
            # Close the ring (repeat first point)
            fh.write(f"{poly[0, 0]:15.6e}{poly[0, 1]:15.6e}\n")
    print(f"Wrote GMT file: {path}")


def write_npz(
    path: str,
    polygons: List[np.ndarray],
    log10_rho_list: List[float],
) -> None:
    """
    Write compressed NPZ output.

    Arrays saved
    ------------
    log10_rho  : (N,)   – log10(Ω·m) per polygon
    coords_flat: (M, 2) – all polygon vertices concatenated
    offsets    : (N+1,) – offsets into coords_flat; polygon i occupies
                          coords_flat[offsets[i]:offsets[i+1]]
                          (ring is open, i.e. first point NOT repeated)
    """
    n = len(polygons)
    log10_rho_arr = np.array(log10_rho_list, dtype=np.float64)

    offsets = np.zeros(n + 1, dtype=np.int64)
    for i, poly in enumerate(polygons):
        offsets[i + 1] = offsets[i] + len(poly)

    total = int(offsets[-1])
    coords_flat = np.empty((total, 2), dtype=np.float64)
    for i, poly in enumerate(polygons):
        coords_flat[offsets[i]:offsets[i + 1]] = poly

    np.savez_compressed(
        path,
        log10_rho=log10_rho_arr,
        coords_flat=coords_flat,
        offsets=offsets,
    )
    print(f"Wrote NPZ file: {path}.npz  ({n} polygons, {total} vertices)")


# ---------------------------------------------------------------------------
# Parameter-file reader
# ---------------------------------------------------------------------------

def read_param_file(path: str):
    """
    Parse the parameter file.  Returns a dict with keys:
        element_type, iter_num, plane_type, center_km, rotation_deg,
        excluded_blocks
    """
    toks = _tokenize(path)

    element_type = int(next(toks))
    iter_num = int(next(toks))
    plane_type = int(next(toks))

    cx = float(next(toks))
    cy = float(next(toks))
    cz = float(next(toks))
    center_km = np.array([cx, cy, cz])

    rotation_deg = float(next(toks))

    n_excluded = int(next(toks))
    excluded = [int(next(toks)) for _ in range(n_excluded)]

    etype_names = {TETRA: "Tetra", BRICK: "Brick", NONCONFORMING_HEXA: "Nonconforming Hexa"}
    ptype_names = {ZX_PLANE: "ZX plane", XY_PLANE: "XY plane"}

    print(f"Element type      : {etype_names.get(element_type, element_type)}")
    print(f"Iteration number  : {iter_num}")
    print(f"Plane type        : {ptype_names.get(plane_type, plane_type)}")
    print(f"Center coord (km) : X={cx}, Y={cy}, Z={cz}")
    print(f"Rotation angle    : {rotation_deg} deg")
    print(f"Excluded blocks   : {excluded}")

    return dict(
        element_type=element_type,
        iter_num=iter_num,
        plane_type=plane_type,
        center_km=center_km,
        rotation_deg=rotation_deg,
        excluded_blocks=excluded,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(param_file: str) -> None:
    params = read_param_file(param_file)

    center_m = params["center_km"] * 1000.0          # km → m
    rotation_rad = params["rotation_deg"] * DEG2RAD

    print("Reading mesh.dat …")
    mesh = read_mesh("mesh.dat")
    print(f"  mesh type: {mesh.mesh_type}, "
          f"nodes: {len(mesh.node_x)}, elements: {len(mesh.elem_nodes)}")

    print(f"Reading resistivity_block_iter{params['iter_num']}.dat …")
    res_block = read_resistivity_block(params["iter_num"])
    print(f"  {len(res_block.rho)} blocks, "
          f"{int(res_block.fixed.sum())} fixed")

    print("Computing cutaway …")
    polygons, log10_rho_list = make_cutaway(
        mesh, res_block,
        params["plane_type"], center_m, rotation_rad,
        params["excluded_blocks"],
    )
    print(f"  {len(polygons)} polygons found")

    stem = f"resistivity_GMT_iter{params['iter_num']}"
    write_gmt(stem + ".dat", polygons, log10_rho_list)
    write_npz(stem, polygons, log10_rho_list)


def main():
    parser = argparse.ArgumentParser(
        description="FEMTIC resistivity cutaway tool (Python port)")
    parser.add_argument("param_file", help="Parameter file (same format as C++ version)")
    args = parser.parse_args()
    run(args.param_file)


if __name__ == "__main__":
    main()
