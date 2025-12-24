#!/usr/bin/env python3
"""femtic_viz_new.py

Visualisation utilities for FEMTIC resistivity models.

This module focuses on two complementary workflows:

1. **Direct-from-FEMTIC files (no intermediate NPZ)**:
   - ``mesh.dat`` + ``resistivity_block_iterX.dat`` → NumPy arrays / PyVista grid
   - Matplotlib map slices (XY at depth) and curtain slices (profile vs depth)
   - PyVista sampling on explicit slice surfaces

2. **Convenience wrappers for NPZ-based workflows** (optional):
   - Uses functions from ``femtic.py`` if you already work with NPZ models.

The direct workflow is usually preferred for quick inspection because it avoids
writing any temporary model files.

Author: Volker Rath (DIAS)
Created with the help of ChatGPT (GPT-5 Thinking) on 2025-12-23
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union, Literal

import numpy as np

# Optional scientific helpers
try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:  # pragma: no cover
    cKDTree = None  # type: ignore

# Optional plotting / 3-D visualisation
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore

try:
    import pyvista as pv  # type: ignore
except Exception:  # pragma: no cover
    pv = None  # type: ignore


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class FemticMesh:
    """Container for a FEMTIC tetrahedral mesh.

    Parameters
    ----------
    nodes
        Node coordinates of shape ``(nn, 3)`` as ``[x, y, z]``.
    conn
        Tetra connectivity of shape ``(nelem, 4)`` with **0-based** node indices.
    """

    nodes: np.ndarray
    conn: np.ndarray


@dataclass(frozen=True)
class FemticResistivityBlock:
    """Container for a FEMTIC resistivity block.

    Notes
    -----
    FEMTIC block files usually contain:

    - a mapping ``region_of_elem`` (length ``nelem``) that assigns each element
      to a region index (often 0-based in the file, but not guaranteed)
    - region properties (one line per region), including a nominal resistivity

    The first two region entries often have special meaning in workflows:

    - ``region_rho[0]``: air
    - ``region_rho[1]``: ocean

    This module does *not* enforce any special treatment automatically; use
    :func:`prepare_rho_for_plotting`.
    """

    region_of_elem: np.ndarray
    region_rho: np.ndarray
    region_rho_lower: Optional[np.ndarray] = None
    region_rho_upper: Optional[np.ndarray] = None
    region_n: Optional[np.ndarray] = None
    region_flag: Optional[np.ndarray] = None


# =============================================================================
# FEMTIC parsing helpers (direct, no NPZ)
# =============================================================================

def read_femtic_mesh(mesh_path: Union[str, Path]) -> FemticMesh:
    """Read a FEMTIC ``mesh.dat`` file (TETRA) into NumPy arrays.

    Parameters
    ----------
    mesh_path
        Path to the FEMTIC mesh file (usually ``mesh.dat``).

    Returns
    -------
    mesh
        Parsed mesh (nodes and connectivity). Connectivity is returned as
        **0-based row indices** into the returned ``nodes`` array.

    Notes
    -----
    FEMTIC mesh files store node coordinates as::

        node_id  x  y  z

    and element lines often contain additional indices before the four vertex
    node IDs (e.g. edge indices). In such cases, the **last four** integer
    fields are the tetra vertex node IDs. This reader therefore uses
    ``parts[-4:]`` for connectivity.
    """
    mesh_path = Path(mesh_path)

    with mesh_path.open("r", errors="ignore") as f:
        header = f.readline().strip()
        if header.upper() != "TETRA":
            raise ValueError(f"Unsupported mesh type {header!r}; expected 'TETRA'.")

        nn_line = f.readline().split()
        if not nn_line:
            raise ValueError("Missing node count after 'TETRA' header.")
        nn = int(nn_line[0])

        node_ids = np.empty(nn, dtype=int)
        nodes = np.empty((nn, 3), dtype=float)
        for i in range(nn):
            parts = f.readline().split()
            if len(parts) < 4:
                raise ValueError(f"Malformed node line {i+1}: {parts!r}")
            node_ids[i] = int(parts[0])
            nodes[i, 0] = float(parts[1])
            nodes[i, 1] = float(parts[2])
            nodes[i, 2] = float(parts[3])

        id2row = {int(nid): i for i, nid in enumerate(node_ids.tolist())}

        ne_line = f.readline().split()
        if not ne_line:
            raise ValueError("Missing element count after nodes.")
        nelem = int(ne_line[0])

        conn = np.empty((nelem, 4), dtype=int)

        def _map_node(nid: int) -> int:
            """Map a FEMTIC node ID to row index."""
            try:
                return id2row[nid]
            except KeyError as e:
                # Conservative fallbacks for off-by-one variants
                if (nid - 1) in id2row:
                    return id2row[nid - 1]
                if (nid + 1) in id2row:
                    return id2row[nid + 1]
                raise e

        for i in range(nelem):
            parts = f.readline().split()
            if len(parts) < 5:
                raise ValueError(f"Malformed element line {i+1}: {parts!r}")
            nids = [int(parts[-4]), int(parts[-3]), int(parts[-2]), int(parts[-1])]
            conn[i, :] = [_map_node(n) for n in nids]

    return FemticMesh(nodes=nodes, conn=conn)

def read_resistivity_block(block_path: Union[str, Path]) -> FemticResistivityBlock:
    """Read a FEMTIC ``resistivity_block_iterX.dat`` file.

    Parameters
    ----------
    block_path
        Path to the resistivity block file.

    Returns
    -------
    block
        Parsed resistivity block (element→region and region properties).

    Notes
    -----
    FEMTIC's exact block format can vary slightly across versions. This reader
    follows the common structure:

    - First line: ``nelem nreg`` (integers)
    - Next ``nelem`` lines: ``ielem region`` (integers)
    - Next ``nreg`` lines: ``ireg rho rho_min rho_max n flag``

    If your file deviates, consider using the robust reader in ``femtic.py`` and
    pass the arrays to :func:`unstructured_grid_from_arrays`.
    """
    block_path = Path(block_path)

    with block_path.open("r", errors="ignore") as f:
        first = f.readline().split()
        if len(first) < 2:
            raise ValueError("Block file must start with two integers: nelem nreg.")
        nelem = int(first[0])
        nreg = int(first[1])

        region_of_elem = np.empty(nelem, dtype=int)
        for i in range(nelem):
            parts = f.readline().split()
            if len(parts) < 2:
                raise ValueError(f"Malformed element→region line {i+1}: {parts!r}")
            region_of_elem[i] = int(parts[1])

        region_rho = np.empty(nreg, dtype=float)
        region_rho_lower = np.empty(nreg, dtype=float)
        region_rho_upper = np.empty(nreg, dtype=float)
        region_n = np.empty(nreg, dtype=float)
        region_flag = np.empty(nreg, dtype=int)

        for i in range(nreg):
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF while reading region lines.")
            parts = line.split()
            if len(parts) < 6:
                raise ValueError(f"Region line has too few columns: {line!r}")
            _ireg = int(parts[0])
            region_rho[i] = float(parts[1])
            region_rho_lower[i] = float(parts[2])
            region_rho_upper[i] = float(parts[3])
            region_n[i] = float(parts[4])
            region_flag[i] = int(parts[5])

    return FemticResistivityBlock(
        region_of_elem=region_of_elem,
        region_rho=region_rho,
        region_rho_lower=region_rho_lower,
        region_rho_upper=region_rho_upper,
        region_n=region_n,
        region_flag=region_flag,
    )


def map_regions_to_element_rho(
    region_of_elem: np.ndarray,
    region_rho: np.ndarray,
    *,
    region_index_base: Literal["auto", "0", "1"] = "auto",
) -> np.ndarray:
    """Map region resistivities onto elements.

    Parameters
    ----------
    region_of_elem
        Region index for each element, shape ``(nelem,)``.
    region_rho
        Resistivity per region, shape ``(nreg,)``.
    region_index_base
        How region indices are stored in ``region_of_elem``.

        - ``"auto"``: infer 0-based vs 1-based from min/max values.
        - ``"0"``: treat as 0-based indices.
        - ``"1"``: treat as 1-based indices.

    Returns
    -------
    rho_elem
        Element-wise resistivity of shape ``(nelem,)``.
    """
    region_of_elem = np.asarray(region_of_elem).astype(int, copy=False)
    region_rho = np.asarray(region_rho, dtype=float)

    if region_index_base == "auto":
        rmin = int(region_of_elem.min())
        rmax = int(region_of_elem.max())
        if rmin >= 1 and rmax == len(region_rho):
            base = 1
        else:
            base = 0
    elif region_index_base == "0":
        base = 0
    else:
        base = 1

    idx = region_of_elem - base
    if idx.min() < 0 or idx.max() >= len(region_rho):
        raise ValueError(
            "Region indices are out of bounds after applying base "
            f"{base}: min={idx.min()}, max={idx.max()}, nreg={len(region_rho)}"
        )
    return region_rho[idx]


def prepare_rho_for_plotting(
    rho_elem: np.ndarray,
    *,
    air_is_nan: bool = True,
    ocean_value: Optional[float] = 1.0e-10,
    air_region_index: int = 0,
    ocean_region_index: int = 1,
    region_of_elem: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Prepare element-wise resistivity for plotting.

    Parameters
    ----------
    rho_elem
        Element-wise resistivity values, shape ``(nelem,)``.
    air_is_nan
        If True, set all elements belonging to air region to NaN.
    ocean_value
        If not None, set all ocean-region elements to this value (Ohm·m).
        Set to None to keep original values.
    air_region_index, ocean_region_index
        Indices identifying air / ocean *in the region numbering*.
    region_of_elem
        Region index for each element. If provided, air/ocean modifications
        are applied by matching region indices. If omitted, no special
        handling is applied.

    Returns
    -------
    rho_plot
        Modified copy of resistivity values.
    """
    rho_plot = np.asarray(rho_elem, dtype=float).copy()

    if region_of_elem is None:
        return rho_plot

    region_of_elem = np.asarray(region_of_elem).astype(int, copy=False)

    if air_is_nan:
        rho_plot[region_of_elem == air_region_index] = np.nan
    if ocean_value is not None:
        rho_plot[region_of_elem == ocean_region_index] = float(ocean_value)
    return rho_plot


# =============================================================================
# PyVista grids (direct, no NPZ)
# =============================================================================

def element_centroids(mesh: FemticMesh) -> np.ndarray:
    """Compute tetrahedron centroids.

    Parameters
    ----------
    mesh
        FEMTIC mesh.

    Returns
    -------
    centroids
        Array of shape ``(nelem, 3)`` with ``[x, y, z]`` centroids.
    """
    return mesh.nodes[mesh.conn].mean(axis=1)


def unstructured_grid_from_arrays(
    nodes: np.ndarray,
    conn: np.ndarray,
    *,
    rho_elem: Optional[np.ndarray] = None,
    region_of_elem: Optional[np.ndarray] = None,
    add_log10: bool = True,
) -> "pv.UnstructuredGrid":
    """Build a PyVista unstructured grid from arrays.

    Parameters
    ----------
    nodes
        Node coordinates of shape ``(nn, 3)``.
    conn
        Tetra connectivity of shape ``(nelem, 4)`` (0-based).
    rho_elem
        Optional element-wise resistivity, shape ``(nelem,)``.
    region_of_elem
        Optional element→region mapping, shape ``(nelem,)`` (0- or 1-based).
        This will be attached as cell-data under the name ``"region"``.
    add_log10
        If True and ``rho_elem`` is provided, also attach ``"log10_resistivity"``.

    Returns
    -------
    grid
        PyVista unstructured grid.

    Raises
    ------
    ImportError
        If PyVista is not available.
    """
    if pv is None:  # pragma: no cover
        raise ImportError("pyvista is required for unstructured grid creation.")

    nodes = np.asarray(nodes, dtype=float)
    conn = np.asarray(conn, dtype=int)

    nelem = conn.shape[0]
    cells = np.hstack([np.full((nelem, 1), 4, dtype=np.int64), conn.astype(np.int64)]).ravel()
    celltypes = np.full(nelem, pv.CellType.TETRA, dtype=np.uint8)

    grid = pv.UnstructuredGrid(cells, celltypes, nodes)

    if rho_elem is not None:
        rho_elem = np.asarray(rho_elem, dtype=float)
        if rho_elem.shape[0] != nelem:
            raise ValueError(f"rho_elem length mismatch: {rho_elem.shape[0]} vs nelem={nelem}")
        grid.cell_data["resistivity"] = rho_elem
        if add_log10:
            with np.errstate(divide="ignore", invalid="ignore"):
                grid.cell_data["log10_resistivity"] = np.log10(rho_elem)

    if region_of_elem is not None:
        region_of_elem = np.asarray(region_of_elem).astype(int, copy=False)
        if region_of_elem.shape[0] != nelem:
            raise ValueError(
                f"region_of_elem length mismatch: {region_of_elem.shape[0]} vs nelem={nelem}"
            )
        grid.cell_data["region"] = region_of_elem

    return grid


def unstructured_grid_from_femtic(
    mesh_path: Union[str, Path],
    block_path: Union[str, Path],
    *,
    region_index_base: Literal["auto", "0", "1"] = "auto",
    apply_plotting_conventions: bool = True,
    air_is_nan: bool = True,
    ocean_value: Optional[float] = 1.0e-10,
    air_region_index: int = 0,
    ocean_region_index: int = 1,
) -> "pv.UnstructuredGrid":
    """Create a PyVista grid directly from FEMTIC files (no NPZ).

    Parameters
    ----------
    mesh_path, block_path
        Paths to ``mesh.dat`` and ``resistivity_block_iterX.dat``.
    region_index_base
        See :func:`map_regions_to_element_rho`.
    apply_plotting_conventions
        If True, apply :func:`prepare_rho_for_plotting` using air/ocean region indices.
    air_is_nan, ocean_value, air_region_index, ocean_region_index
        Passed to :func:`prepare_rho_for_plotting`.

    Returns
    -------
    grid
        PyVista unstructured grid with cell-data:

        - ``resistivity``
        - ``log10_resistivity`` (if possible)
        - ``region``
    """
    mesh = read_femtic_mesh(mesh_path)
    block = read_resistivity_block(block_path)

    rho_elem = map_regions_to_element_rho(
        block.region_of_elem, block.region_rho, region_index_base=region_index_base
    )

    if apply_plotting_conventions:
        rho_elem = prepare_rho_for_plotting(
            rho_elem,
            air_is_nan=air_is_nan,
            ocean_value=ocean_value,
            air_region_index=air_region_index,
            ocean_region_index=ocean_region_index,
            region_of_elem=block.region_of_elem,
        )

    return unstructured_grid_from_arrays(
        mesh.nodes,
        mesh.conn,
        rho_elem=rho_elem,
        region_of_elem=block.region_of_elem,
        add_log10=True,
    )


# =============================================================================
# Slice surfaces (PyVista sampling)
# =============================================================================

def sample_polyline(
    polyline_xy: np.ndarray,
    *,
    n: int = 501,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample a polyline (XY) and return points and cumulative distance.

    Parameters
    ----------
    polyline_xy
        Polyline vertices, shape ``(m, 2)``.
    n
        Number of resampled points.

    Returns
    -------
    pts
        Resampled polyline points, shape ``(n, 2)``.
    s
        Along-profile distance (same length as ``pts``), starting at 0.
    """
    polyline_xy = np.asarray(polyline_xy, dtype=float)
    if polyline_xy.ndim != 2 or polyline_xy.shape[1] != 2:
        raise ValueError("polyline_xy must have shape (m, 2).")
    if polyline_xy.shape[0] < 2:
        raise ValueError("polyline_xy must contain at least two points.")

    seg = np.diff(polyline_xy, axis=0)
    seglen = np.sqrt((seg ** 2).sum(axis=1))
    s0 = np.hstack([[0.0], np.cumsum(seglen)])
    total = s0[-1]
    if total <= 0:
        raise ValueError("Polyline has zero length.")
    s = np.linspace(0.0, total, int(n))

    x = np.interp(s, s0, polyline_xy[:, 0])
    y = np.interp(s, s0, polyline_xy[:, 1])
    return np.column_stack([x, y]), s


def build_curtain_surface(
    polyline_xy: np.ndarray,
    *,
    zmin: float,
    zmax: float,
    nz: int = 201,
    ns: int = 501,
) -> "pv.StructuredGrid":
    """Create a vertical curtain surface as a PyVista StructuredGrid.

    Parameters
    ----------
    polyline_xy
        Polyline in XY, shape ``(m, 2)``.
    zmin, zmax
        Vertical bounds (same units as mesh z).
    nz, ns
        Grid resolution in z and along profile.

    Returns
    -------
    grid
        PyVista StructuredGrid surface.

    Raises
    ------
    ImportError
        If PyVista is not available.
    """
    if pv is None:  # pragma: no cover
        raise ImportError("pyvista is required for curtain surface construction.")

    poly_s, s = sample_polyline(polyline_xy, n=ns)
    z = np.linspace(zmin, zmax, int(nz))

    X = np.repeat(poly_s[:, 0][:, None], z.size, axis=1)
    Y = np.repeat(poly_s[:, 1][:, None], z.size, axis=1)
    Z = np.repeat(z[None, :], poly_s.shape[0], axis=0)

    grid = pv.StructuredGrid(X, Y, Z)
    grid.point_data["s"] = np.repeat(s[:, None], z.size, axis=1).ravel(order="F")
    grid.point_data["z"] = Z.ravel(order="F")
    return grid


def build_map_surface(
    x: np.ndarray,
    y: np.ndarray,
    *,
    z0: float,
) -> "pv.StructuredGrid":
    """Create an XY map surface at constant z as a PyVista StructuredGrid.

    Parameters
    ----------
    x, y
        1-D arrays defining the grid coordinates. The mesh is ``len(y) x len(x)``.
    z0
        Constant z value for the surface.

    Returns
    -------
    grid
        PyVista StructuredGrid.

    Raises
    ------
    ImportError
        If PyVista is not available.
    """
    if pv is None:  # pragma: no cover
        raise ImportError("pyvista is required for map surface construction.")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X, Y = np.meshgrid(x, y, indexing="xy")
    Z = np.full_like(X, float(z0))

    grid = pv.StructuredGrid(X, Y, Z)
    grid.point_data["x"] = X.ravel(order="F")
    grid.point_data["y"] = Y.ravel(order="F")
    grid.point_data["z"] = Z.ravel(order="F")
    return grid


def sample_grid_on_surface(
    grid: "pv.UnstructuredGrid",
    surface: "pv.StructuredGrid",
    *,
    scalar: str = "log10_resistivity",
) -> "pv.StructuredGrid":
    """Sample an unstructured grid on a structured surface.

    Parameters
    ----------
    grid
        Unstructured FEMTIC grid.
    surface
        Structured surface grid (curtain or map).
    scalar
        Name of the scalar in ``grid.cell_data`` (or ``grid.point_data``) to sample.

    Returns
    -------
    sampled
        StructuredGrid with sampled scalar in ``point_data``.

    Notes
    -----
    PyVista will convert cell-data to point samples on the fly.
    """
    if pv is None:  # pragma: no cover
        raise ImportError("pyvista is required for sampling.")

    if scalar not in grid.cell_data and scalar not in grid.point_data:
        raise KeyError(f"Scalar {scalar!r} not found in grid (cell_data/point_data).")

    return surface.sample(grid)


# =============================================================================
# Matplotlib slices (with/without interpolation)
# =============================================================================

def _require_mpl() -> Any:
    """Internal helper: ensure Matplotlib is available."""
    if plt is None:  # pragma: no cover
        raise ImportError("matplotlib is required for Matplotlib plotting functions.")
    return plt



def _apply_log10(values: np.ndarray, *, log10: bool) -> Tuple[np.ndarray, str]:
    """Apply optional base-10 logarithm to values for plotting.

    Parameters
    ----------
    values
        Input values.
    log10
        If True, compute ``log10(values)`` with non-positive values mapped to NaN.

    Returns
    -------
    plotted
        Values to plot.
    label
        Suggested colourbar label.
    """
    v = np.asarray(values, dtype=float)
    if not log10:
        return v, "resistivity"
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log10(v)
    return out, "log10(resistivity)"


def map_slice_points_from_cells(
    mesh: FemticMesh,
    rho_elem: np.ndarray,
    *,
    z0: float,
    dz: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract point samples for an XY map slice from cell centroids.

    Parameters
    ----------
    mesh
        FEMTIC mesh.
    rho_elem
        Element-wise resistivity, shape ``(nelem,)``.
    z0
        Target slice depth (same z convention as in the mesh).
    dz
        Half-thickness of the depth window. Cells with centroid z in
        ``[z0 - dz, z0 + dz]`` are selected.

    Returns
    -------
    x, y, rho
        Arrays of equal length with centroid coordinates and resistivity.

    Notes
    -----
    These are *samples* (centroid picks), not a true geometric intersection of
    the slice plane with each tetrahedron. For fast inspection, this is often
    sufficient; for exact patch geometry you would need tetra/plane intersection.
    """
    rho_elem = np.asarray(rho_elem, dtype=float)
    ctr = element_centroids(mesh)
    sel = np.isfinite(rho_elem) & (ctr[:, 2] >= z0 - dz) & (ctr[:, 2] <= z0 + dz)
    return ctr[sel, 0], ctr[sel, 1], rho_elem[sel]


def curtain_points_from_cells(
    mesh: FemticMesh,
    rho_elem: np.ndarray,
    polyline_xy: np.ndarray,
    *,
    width: float = 500.0,
    n_profile: int = 1001,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract point samples for a curtain section from nearby cell centroids.

    Parameters
    ----------
    mesh
        FEMTIC mesh.
    rho_elem
        Element-wise resistivity, shape ``(nelem,)``.
    polyline_xy
        Profile polyline in XY, shape ``(m, 2)``.
    width
        Maximum lateral distance from profile (same units as x/y). Only cell
        centroids within this distance are kept.
    n_profile
        Number of points used to resample the polyline for nearest-distance
        mapping to the along-profile coordinate ``s``.

    Returns
    -------
    s, z, rho
        Arrays of equal length, where ``s`` is distance along the profile and
        ``z`` is the centroid z-coordinate.

    Raises
    ------
    ImportError
        If SciPy is not available (uses :class:`scipy.spatial.cKDTree`).
    """
    if cKDTree is None:  # pragma: no cover
        raise ImportError("scipy is required for curtain point extraction (cKDTree).")

    rho_elem = np.asarray(rho_elem, dtype=float)
    ctr = element_centroids(mesh)

    prof_xy, prof_s = sample_polyline(polyline_xy, n=n_profile)
    tree = cKDTree(prof_xy)
    dist, idx = tree.query(ctr[:, :2], k=1)

    sel = np.isfinite(rho_elem) & (dist <= float(width))
    return prof_s[idx[sel]], ctr[sel, 2], rho_elem[sel]


def _triangulation_mask(
    x: np.ndarray,
    y: np.ndarray,
    triangles: np.ndarray,
    *,
    max_edge: Optional[float] = None,
    max_area: Optional[float] = None,
) -> np.ndarray:
    """Compute a boolean mask for triangles based on simple geometric criteria.

    Parameters
    ----------
    x, y
        Point coordinates.
    triangles
        Triangle connectivity, shape ``(ntri, 3)``.
    max_edge
        If not None, mask triangles where *any* edge length exceeds this threshold.
    max_area
        If not None, mask triangles where absolute area exceeds this threshold.

    Returns
    -------
    mask
        Boolean mask of length ``ntri`` (True means triangle is masked).

    Notes
    -----
    For point-based sections, Delaunay triangulation can create long triangles
    across gaps. ``max_edge`` is a practical way to suppress such artefacts.
    """
    mask = np.zeros(triangles.shape[0], dtype=bool)

    if max_edge is None and max_area is None:
        return mask

    tri = triangles
    x0, y0 = x[tri[:, 0]], y[tri[:, 0]]
    x1, y1 = x[tri[:, 1]], y[tri[:, 1]]
    x2, y2 = x[tri[:, 2]], y[tri[:, 2]]

    if max_edge is not None:
        e01 = np.hypot(x1 - x0, y1 - y0)
        e12 = np.hypot(x2 - x1, y2 - y1)
        e20 = np.hypot(x0 - x2, y0 - y2)
        mask |= (e01 > max_edge) | (e12 > max_edge) | (e20 > max_edge)

    if max_area is not None:
        area2 = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
        area = 0.5 * np.abs(area2)
        mask |= area > max_area

    return mask


def plot_points_matplotlib(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    *,
    mode: Literal["scatter", "tri"] = "tri",
    log10: bool = True,
    ax: Optional[Any] = None,
    s: int = 6,
    mask_max_edge: Optional[float] = None,
    mask_max_area: Optional[float] = None,
    title: Optional[str] = None,
    xlabel: str = "x",
    ylabel: str = "y",
    invert_yaxis: bool = False,
) -> Any:
    """Plot scattered samples as either markers or filled triangle patches.

    Parameters
    ----------
    x, y
        Coordinates of samples.
    values
        Scalar values at samples.
    mode
        Plot style:

        - ``"scatter"``: marker plot (no connectivity).
        - ``"tri"``: Delaunay triangulation + ``tripcolor`` (patch-like).
    log10
        If True, colour by ``log10(values)``.
    ax
        Axes to draw on. If None, create a new figure/axes.
    s
        Marker size for ``mode="scatter"``.
    mask_max_edge, mask_max_area
        Optional masking criteria for ``mode="tri"`` to suppress long/bridging
        triangles and/or very large triangles.
    title
        Optional plot title.
    xlabel, ylabel
        Axis labels.
    invert_yaxis
        If True, invert the y-axis (useful for depth sections).

    Returns
    -------
    ax
        Matplotlib axes.

    Raises
    ------
    ImportError
        If Matplotlib is not available.
    """
    plt_ = _require_mpl()

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    v_plot, label = _apply_log10(np.asarray(values, dtype=float), log10=log10)

    if ax is None:
        _, ax = plt_.subplots()

    if mode == "scatter":
        sc = ax.scatter(x, y, c=v_plot, s=int(s))
        plt_.colorbar(sc, ax=ax, label=label)
    elif mode == "tri":
        import matplotlib.tri as mtri  # local import (optional dependency)

        tri = mtri.Triangulation(x, y)
        mask = _triangulation_mask(
            x, y, tri.triangles, max_edge=mask_max_edge, max_area=mask_max_area
        )
        if np.any(mask):
            tri.set_mask(mask)

        pc = ax.tripcolor(tri, v_plot, shading="flat")
        plt_.colorbar(pc, ax=ax, label=label)
    else:
        raise ValueError(f"Unsupported mode: {mode!r}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if invert_yaxis:
        ax.invert_yaxis()
    return ax


def map_slice_grid_idw(
    mesh: FemticMesh,
    rho_elem: np.ndarray,
    *,
    z0: float,
    dz: float = 50.0,
    nx: int = 301,
    ny: int = 301,
    k: int = 8,
    power: float = 2.0,
    pad: float = 0.0,
    max_dist: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a regular (x, y) map grid at depth using inverse-distance weighting.

    Parameters
    ----------
    mesh
        FEMTIC mesh.
    rho_elem
        Element-wise resistivity.
    z0, dz
        Slice window definition (see :func:`map_slice_points_from_cells`).
    nx, ny
        Output grid size.
    k
        Number of nearest neighbours for IDW.
    power
        IDW power.
    pad
        Optional padding added to the (x, y) bounds before gridding.
    max_dist
        If not None, grid points where all neighbours are farther than this are set to NaN.

    Returns
    -------
    xg, yg, V
        ``xg`` and ``yg`` are 1-D coordinate vectors, and ``V`` is a 2-D array of
        shape ``(ny, nx)`` suitable for :func:`matplotlib.axes.Axes.pcolormesh`.

    Raises
    ------
    ImportError
        If SciPy is not available.
    """
    if cKDTree is None:  # pragma: no cover
        raise ImportError("scipy is required for IDW map grids (cKDTree).")

    x, y, rho = map_slice_points_from_cells(mesh, rho_elem, z0=z0, dz=dz)
    ok = np.isfinite(rho)
    x = x[ok]
    y = y[ok]
    rho = rho[ok]

    if x.size < 3:
        raise ValueError("Too few points in slice to build an IDW grid.")

    xmin, xmax = float(x.min()) - pad, float(x.max()) + pad
    ymin, ymax = float(y.min()) - pad, float(y.max()) + pad

    xg = np.linspace(xmin, xmax, int(nx))
    yg = np.linspace(ymin, ymax, int(ny))
    X, Y = np.meshgrid(xg, yg, indexing="xy")

    tree = cKDTree(np.column_stack([x, y]))
    dist, idx = tree.query(np.column_stack([X.ravel(), Y.ravel()]), k=int(k))

    dist = np.asarray(dist, dtype=float)
    idx = np.asarray(idx, dtype=int)
    if dist.ndim == 1:
        dist = dist[:, None]
        idx = idx[:, None]

    if max_dist is not None:
        mask_far = np.all(dist > float(max_dist), axis=1)
    else:
        mask_far = np.zeros(dist.shape[0], dtype=bool)

    with np.errstate(divide="ignore", invalid="ignore"):
        w = 1.0 / np.maximum(dist, 1.0e-12) ** float(power)
    wsum = np.sum(w, axis=1)
    V = np.sum(w * rho[idx], axis=1) / np.maximum(wsum, 1.0e-30)
    V[mask_far] = np.nan

    V = V.reshape((yg.size, xg.size))
    return xg, yg, V


def plot_map_grid_matplotlib(
    x: np.ndarray,
    y: np.ndarray,
    V: np.ndarray,
    *,
    log10: bool = True,
    ax: Optional[Any] = None,
    cmap: Optional[str] = None,
    title: Optional[str] = None,
) -> Any:
    """Plot a regular (x, y) grid produced by :func:`map_slice_grid_idw`.

    Parameters
    ----------
    x, y
        1-D grid coordinates.
    V
        2-D values of shape ``(ny, nx)``.
    log10
        If True, display ``log10(V)``.
    ax
        Axes to draw on. If None, create a new figure/axes.
    cmap
        Optional Matplotlib colormap name (default: Matplotlib default).
    title
        Optional plot title.

    Returns
    -------
    ax
        Matplotlib axes.
    """
    plt_ = _require_mpl()

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    V = np.asarray(V, dtype=float)

    Vp, label = _apply_log10(V, log10=log10)

    if ax is None:
        _, ax = plt_.subplots()

    im = ax.pcolormesh(x, y, Vp, shading="auto", cmap=cmap)
    plt_.colorbar(im, ax=ax, label=label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    if title is not None:
        ax.set_title(title)
    return ax



def map_slice_from_cells(
    mesh: FemticMesh,
    rho_elem: np.ndarray,
    *,
    z0: float,
    dz: float = 50.0,
    mode: Literal["scatter", "tri", "grid"] = "tri",
    log10: bool = True,
    ax: Optional[Any] = None,
    s: int = 8,
    mask_max_edge: Optional[float] = None,
    mask_max_area: Optional[float] = None,
    grid_nx: int = 301,
    grid_ny: int = 301,
    grid_k: int = 8,
    grid_power: float = 2.0,
    grid_pad: float = 0.0,
    grid_max_dist: Optional[float] = None,
    cmap: Optional[str] = None,
) -> Any:
    """Plot an XY map slice from cell-centroid samples.

    Parameters
    ----------
    mesh
        FEMTIC mesh.
    rho_elem
        Element-wise resistivity, shape ``(nelem,)``.
    z0
        Target slice depth.
    dz
        Half-thickness of the depth window. Cells with centroid z in
        ``[z0 - dz, z0 + dz]`` are selected.
    mode
        Plot style:

        - ``"scatter"``: marker plot (fast, no connectivity).
        - ``"tri"``: Delaunay triangulation + ``tripcolor`` (patch-like).
        - ``"grid"``: regular grid via IDW + ``pcolormesh`` (requires SciPy).

    log10
        If True, colour by log10(resistivity).
    ax
        Matplotlib axes to plot into. If None, a new figure/axes is created.
    s
        Marker size for ``mode="scatter"``.
    mask_max_edge, mask_max_area
        Masking criteria for ``mode="tri"`` to avoid long triangles bridging gaps.
        See :func:`plot_points_matplotlib`.
    grid_nx, grid_ny, grid_k, grid_power, grid_pad, grid_max_dist
        Parameters for ``mode="grid"``. See :func:`map_slice_grid_idw`.
    cmap
        Optional Matplotlib colormap name for ``mode="grid"``.

    Returns
    -------
    ax
        Matplotlib axes.

    Notes
    -----
    The input data are cell-centroid samples in a depth window. This is not a
    geometric slice through the tetrahedra; it is designed for quick inspection.
    """
    if mode in ("scatter", "tri"):
        x, y, rho = map_slice_points_from_cells(mesh, rho_elem, z0=z0, dz=dz)
        title = f"Map slice at z={z0} ± {dz} ({mode})"
        ax = plot_points_matplotlib(
            x,
            y,
            rho,
            mode=("scatter" if mode == "scatter" else "tri"),
            log10=log10,
            ax=ax,
            s=s,
            mask_max_edge=mask_max_edge,
            mask_max_area=mask_max_area,
            title=title,
            xlabel="x",
            ylabel="y",
            invert_yaxis=False,
        )
        ax.set_aspect("equal")
        return ax

    if mode == "grid":
        xg, yg, V = map_slice_grid_idw(
            mesh,
            rho_elem,
            z0=z0,
            dz=dz,
            nx=grid_nx,
            ny=grid_ny,
            k=grid_k,
            power=grid_power,
            pad=grid_pad,
            max_dist=grid_max_dist,
        )
        title = f"Map slice at z={z0} ± {dz} (IDW grid)"
        ax = plot_map_grid_matplotlib(xg, yg, V, log10=log10, ax=ax, cmap=cmap, title=title)
        return ax

    raise ValueError(f"Unsupported mode: {mode!r}")


def map_slice_scatter_from_cells(
    mesh: FemticMesh,
    rho_elem: np.ndarray,
    *,
    z0: float,
    dz: float = 50.0,
    log10: bool = True,
    ax: Optional[Any] = None,
    s: int = 8,
) -> Any:
    """Backward-compatible wrapper for map slice scatter.

    This calls :func:`map_slice_from_cells` with ``mode="scatter"``.
    """
    return map_slice_from_cells(
        mesh,
        rho_elem,
        z0=z0,
        dz=dz,
        mode="scatter",
        log10=log10,
        ax=ax,
        s=s,
    )




def curtain_from_cells(
    mesh: FemticMesh,
    rho_elem: np.ndarray,
    polyline_xy: np.ndarray,
    *,
    width: float = 500.0,
    n_profile: int = 1001,
    mode: Literal["scatter", "tri", "grid"] = "tri",
    log10: bool = True,
    ax: Optional[Any] = None,
    s: int = 6,
    mask_max_edge: Optional[float] = None,
    mask_max_area: Optional[float] = None,
    grid_zmin: Optional[float] = None,
    grid_zmax: Optional[float] = None,
    grid_nz: int = 201,
    grid_ns: int = 501,
    grid_k: int = 8,
    grid_power: float = 2.0,
    grid_max_dist: Optional[float] = None,
    cmap: Optional[str] = None,
) -> Any:
    """Plot a curtain section along a polyline from cell-centroid samples.

    Parameters
    ----------
    mesh
        FEMTIC mesh.
    rho_elem
        Element-wise resistivity, shape ``(nelem,)``.
    polyline_xy
        Profile polyline in XY, shape ``(m, 2)``.
    width
        Maximum lateral distance from profile (same units as x/y) for point sampling.
    n_profile
        Resampling density for mapping centroids to along-profile distance ``s``.
    mode
        Plot style:

        - ``"scatter"``: marker plot (no connectivity).
        - ``"tri"``: Delaunay triangulation in (s, z) + ``tripcolor`` (patch-like).
        - ``"grid"``: regular (s, z) grid via IDW + ``pcolormesh`` (requires SciPy).

    log10
        If True, colour by log10(resistivity).
    ax
        Matplotlib axes to plot into. If None, a new figure/axes is created.
    s
        Marker size for ``mode="scatter"``.
    mask_max_edge, mask_max_area
        Masking criteria for ``mode="tri"`` to avoid long triangles bridging gaps.
        In curtain coordinates, ``mask_max_edge`` is measured in the same units as
        ``s`` and ``z``.
    grid_zmin, grid_zmax
        Depth range for ``mode="grid"``. If omitted, it is inferred from the sampled
        point cloud.
    grid_nz, grid_ns, grid_k, grid_power, grid_max_dist
        Parameters for ``mode="grid"``. See :func:`curtain_grid_idw`.
    cmap
        Optional Matplotlib colormap name for ``mode="grid"``.

    Returns
    -------
    ax
        Matplotlib axes.

    Notes
    -----
    - ``mode="tri"`` produces the "coloured patches" look using triangulated points.
      It does not compute exact tetra/plane intersection polygons.
    - If your section has gaps, consider setting ``mask_max_edge`` (e.g. a few times
      the median point spacing in ``s``) to suppress bridging triangles.
    """
    if mode in ("scatter", "tri"):
        scoord, z, rho = curtain_points_from_cells(
            mesh,
            rho_elem,
            polyline_xy,
            width=width,
            n_profile=n_profile,
        )
        title = f"Curtain section (width ≤ {width}) ({mode})"
        ax = plot_points_matplotlib(
            scoord,
            z,
            rho,
            mode=("scatter" if mode == "scatter" else "tri"),
            log10=log10,
            ax=ax,
            s=s,
            mask_max_edge=mask_max_edge,
            mask_max_area=mask_max_area,
            title=title,
            xlabel="s (along profile)",
            ylabel="z",
            invert_yaxis=True,
        )
        return ax

    if mode == "grid":
        # Infer depth range if needed from sampled points (fast, avoids extra args).
        if grid_zmin is None or grid_zmax is None:
            scoord, z, rho = curtain_points_from_cells(
                mesh,
                rho_elem,
                polyline_xy,
                width=width,
                n_profile=n_profile,
            )
            if grid_zmin is None:
                grid_zmin = float(np.nanmin(z))
            if grid_zmax is None:
                grid_zmax = float(np.nanmax(z))

        s1, z1, V = curtain_grid_idw(
            mesh,
            rho_elem,
            polyline_xy,
            zmin=float(grid_zmin),
            zmax=float(grid_zmax),
            nz=grid_nz,
            ns=grid_ns,
            k=grid_k,
            power=grid_power,
            max_dist=grid_max_dist,
        )
        title = "Curtain slice (IDW grid)"
        ax = plot_curtain_matplotlib(s1, z1, V, log10=log10, ax=ax, cmap=cmap)
        ax.set_title(title)
        return ax

    raise ValueError(f"Unsupported mode: {mode!r}")


def curtain_scatter_from_cells(
    mesh: FemticMesh,
    rho_elem: np.ndarray,
    polyline_xy: np.ndarray,
    *,
    width: float = 500.0,
    n_profile: int = 1001,
    log10: bool = True,
    ax: Optional[Any] = None,
    s: int = 6,
) -> Any:
    """Backward-compatible wrapper for curtain scatter.

    This calls :func:`curtain_from_cells` with ``mode="scatter"``.
    """
    return curtain_from_cells(
        mesh,
        rho_elem,
        polyline_xy,
        width=width,
        n_profile=n_profile,
        mode="scatter",
        log10=log10,
        ax=ax,
        s=s,
    )



def curtain_grid_idw(
    mesh: FemticMesh,
    rho_elem: np.ndarray,
    polyline_xy: np.ndarray,
    *,
    zmin: float,
    zmax: float,
    nz: int = 201,
    ns: int = 501,
    k: int = 8,
    power: float = 2.0,
    max_dist: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a regular (s, z) curtain grid using inverse-distance weighting.

    Parameters
    ----------
    mesh
        FEMTIC mesh.
    rho_elem
        Element-wise resistivity, shape ``(nelem,)``.
    polyline_xy
        Profile polyline in XY, shape ``(m, 2)``.
    zmin, zmax, nz, ns
        Curtain grid definition.
    k
        Number of nearest neighbours used for IDW.
    power
        IDW power.
    max_dist
        Optional maximum neighbour distance; if given, query points with all
        neighbours farther than this are set to NaN.

    Returns
    -------
    s
        1-D along-profile coordinates, shape ``(ns,)``.
    z
        1-D z coordinates, shape ``(nz,)``.
    V
        2-D array of sampled values, shape ``(nz, ns)``.

    Raises
    ------
    ImportError
        If SciPy is not available.
    """
    if cKDTree is None:  # pragma: no cover
        raise ImportError("scipy is required for IDW curtain grids (cKDTree).")

    rho_elem = np.asarray(rho_elem, dtype=float)
    ctr = element_centroids(mesh)
    ok = np.isfinite(rho_elem)
    pts = ctr[ok, :]
    vals = rho_elem[ok]

    prof_xy, s = sample_polyline(polyline_xy, n=ns)
    z = np.linspace(zmin, zmax, int(nz))

    X = np.repeat(prof_xy[:, 0][:, None], z.size, axis=1)
    Y = np.repeat(prof_xy[:, 1][:, None], z.size, axis=1)
    Z = np.repeat(z[None, :], prof_xy.shape[0], axis=0)
    Q = np.column_stack([X.ravel(order="C"), Y.ravel(order="C"), Z.ravel(order="C")])

    tree = cKDTree(pts)
    dist, idx = tree.query(Q, k=int(k))

    dist = np.asarray(dist, dtype=float)
    idx = np.asarray(idx, dtype=int)

    if dist.ndim == 1:
        dist = dist[:, None]
        idx = idx[:, None]

    if max_dist is not None:
        mask_far = np.all(dist > float(max_dist), axis=1)
    else:
        mask_far = np.zeros(dist.shape[0], dtype=bool)

    with np.errstate(divide="ignore", invalid="ignore"):
        w = 1.0 / np.maximum(dist, 1.0e-12) ** float(power)
    wsum = np.sum(w, axis=1)
    V = np.sum(w * vals[idx], axis=1) / np.maximum(wsum, 1.0e-30)
    V[mask_far] = np.nan

    V = V.reshape((ns, z.size)).T  # (nz, ns)
    return s, z, V


def plot_curtain_matplotlib(
    s: np.ndarray,
    z: np.ndarray,
    V: np.ndarray,
    *,
    log10: bool = True,
    ax: Optional[Any] = None,
    cmap: Optional[str] = None,
) -> Any:
    """Plot a regular curtain grid produced by :func:`curtain_grid_idw`.

    Parameters
    ----------
    s, z
        1-D grid coordinates.
    V
        2-D values of shape ``(nz, ns)``.
    log10
        If True, display ``log10(V)``.
    ax
        Axes to draw on. If None, create a new figure/axes.
    cmap
        Optional Matplotlib colormap name (default: Matplotlib default).

    Returns
    -------
    ax
        Matplotlib axes.
    """
    plt_ = _require_mpl()

    s = np.asarray(s, dtype=float)
    z = np.asarray(z, dtype=float)
    V = np.asarray(V, dtype=float)

    if log10:
        with np.errstate(divide="ignore", invalid="ignore"):
            Vp = np.log10(V)
        label = "log10(resistivity)"
    else:
        Vp = V
        label = "resistivity"

    if ax is None:
        _, ax = plt_.subplots()

    im = ax.pcolormesh(s, z, Vp, shading="auto", cmap=cmap)
    plt_.colorbar(im, ax=ax, label=label)
    ax.set_xlabel("s (along profile)")
    ax.set_ylabel("z")
    ax.set_title("Curtain slice (IDW)")
    ax.invert_yaxis()
    return ax


# =============================================================================
# Optional NPZ wrappers (use femtic.py if present)
# =============================================================================

def npz_to_unstructured_grid(
    npz_path: Union[str, Path],
) -> "pv.UnstructuredGrid":
    """Load an NPZ model as a PyVista unstructured grid using femtic.py.

    Parameters
    ----------
    npz_path
        Path to NPZ produced by FEMTIC conversion tools.

    Returns
    -------
    grid
        PyVista grid.

    Raises
    ------
    ImportError
        If ``femtic.py`` cannot be imported or PyVista is unavailable.
    """
    if pv is None:  # pragma: no cover
        raise ImportError("pyvista is required for NPZ→grid conversion.")

    try:
        import femtic  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("Could not import femtic.py for NPZ helpers.") from e

    return femtic.npz_to_unstructured_grid(str(npz_path))


# =============================================================================
# CLI
# =============================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command line entry point.

    This CLI is intentionally small; it exists mainly for quick inspection.

    Examples
    --------
    Export VTU directly from FEMTIC files::

        python femtic_viz_new.py export-vtu --mesh mesh.dat --block resistivity_block_iter0.dat --out model.vtu

    Curtain section with patch-like plotting (triangulated points)::

        python femtic_viz_new.py curtain --mesh mesh.dat --block resistivity_block_iter0.dat --polyline profile.csv --mode tri --mask-max-edge 500

    Notes
    -----
    The ``profile.csv`` is expected to have two columns: ``x,y`` without header.
    """
    import argparse

    p = argparse.ArgumentParser(prog="femtic_viz", description="FEMTIC visualisation helpers.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_vtu = sub.add_parser("export-vtu", help="Export a VTU from mesh+block (no NPZ).")
    p_vtu.add_argument("--mesh", required=True)
    p_vtu.add_argument("--block", required=True)
    p_vtu.add_argument("--out", required=True)

    # Flexible Matplotlib section commands
    p_map = sub.add_parser("map", help="Matplotlib map slice (scatter/tri/IDW grid).")
    p_map.add_argument("--mesh", required=True)
    p_map.add_argument("--block", required=True)
    p_map.add_argument("--z0", type=float, required=True)
    p_map.add_argument("--dz", type=float, default=50.0)
    p_map.add_argument("--mode", choices=["scatter", "tri", "grid"], default="tri")
    p_map.add_argument("--mask-max-edge", type=float, default=None)
    p_map.add_argument("--mask-max-area", type=float, default=None)
    p_map.add_argument("--grid-nx", type=int, default=301)
    p_map.add_argument("--grid-ny", type=int, default=301)
    p_map.add_argument("--grid-k", type=int, default=8)
    p_map.add_argument("--grid-power", type=float, default=2.0)
    p_map.add_argument("--grid-pad", type=float, default=0.0)
    p_map.add_argument("--grid-max-dist", type=float, default=None)
    p_map.add_argument("--out", default=None)

    p_cur = sub.add_parser("curtain", help="Matplotlib curtain section (scatter/tri/IDW grid).")
    p_cur.add_argument("--mesh", required=True)
    p_cur.add_argument("--block", required=True)
    p_cur.add_argument("--polyline", required=True, help="CSV with two columns: x,y")
    p_cur.add_argument("--width", type=float, default=500.0)
    p_cur.add_argument("--mode", choices=["scatter", "tri", "grid"], default="tri")
    p_cur.add_argument("--mask-max-edge", type=float, default=None)
    p_cur.add_argument("--mask-max-area", type=float, default=None)
    p_cur.add_argument("--zmin", type=float, default=None, help="Only used for --mode grid (optional).")
    p_cur.add_argument("--zmax", type=float, default=None, help="Only used for --mode grid (optional).")
    p_cur.add_argument("--nz", type=int, default=201)
    p_cur.add_argument("--ns", type=int, default=501)
    p_cur.add_argument("--k", type=int, default=8)
    p_cur.add_argument("--power", type=float, default=2.0)
    p_cur.add_argument("--grid-max-dist", type=float, default=None)
    p_cur.add_argument("--out", default=None)

    # Backward-compatible (older) commands
    p_map_sc = sub.add_parser("map-scatter", help="(Deprecated) scatter map slice (no gridding).")
    p_map_sc.add_argument("--mesh", required=True)
    p_map_sc.add_argument("--block", required=True)
    p_map_sc.add_argument("--z0", type=float, required=True)
    p_map_sc.add_argument("--dz", type=float, default=50.0)
    p_map_sc.add_argument("--out", default=None)

    p_cur_sc = sub.add_parser("curtain-scatter", help="(Deprecated) scatter curtain (no gridding).")
    p_cur_sc.add_argument("--mesh", required=True)
    p_cur_sc.add_argument("--block", required=True)
    p_cur_sc.add_argument("--polyline", required=True, help="CSV with two columns: x,y")
    p_cur_sc.add_argument("--width", type=float, default=500.0)
    p_cur_sc.add_argument("--out", default=None)

    p_idw = sub.add_parser("curtain-idw", help="Curtain grid via IDW and plot (regular grid).")
    p_idw.add_argument("--mesh", required=True)
    p_idw.add_argument("--block", required=True)
    p_idw.add_argument("--polyline", required=True, help="CSV with two columns: x,y")
    p_idw.add_argument("--zmin", type=float, required=True)
    p_idw.add_argument("--zmax", type=float, required=True)
    p_idw.add_argument("--nz", type=int, default=201)
    p_idw.add_argument("--ns", type=int, default=501)
    p_idw.add_argument("--k", type=int, default=8)
    p_idw.add_argument("--power", type=float, default=2.0)
    p_idw.add_argument("--out", default=None)

    args = p.parse_args(list(argv) if argv is not None else None)

    if args.cmd == "export-vtu":
        grid = unstructured_grid_from_femtic(args.mesh, args.block)
        grid.save(args.out)
        return 0

    if args.cmd == "map":
        mesh = read_femtic_mesh(args.mesh)
        block = read_resistivity_block(args.block)
        rho = map_regions_to_element_rho(block.region_of_elem, block.region_rho)
        rho = prepare_rho_for_plotting(rho, region_of_elem=block.region_of_elem)
        map_slice_from_cells(
            mesh,
            rho,
            z0=args.z0,
            dz=args.dz,
            mode=args.mode,
            mask_max_edge=args.mask_max_edge,
            mask_max_area=args.mask_max_area,
            grid_nx=args.grid_nx,
            grid_ny=args.grid_ny,
            grid_k=args.grid_k,
            grid_power=args.grid_power,
            grid_pad=args.grid_pad,
            grid_max_dist=args.grid_max_dist,
        )
        if args.out:
            _require_mpl().savefig(args.out, dpi=200, bbox_inches="tight")
        else:
            _require_mpl().show()
        return 0

    if args.cmd == "curtain":
        mesh = read_femtic_mesh(args.mesh)
        block = read_resistivity_block(args.block)
        rho = map_regions_to_element_rho(block.region_of_elem, block.region_rho)
        rho = prepare_rho_for_plotting(rho, region_of_elem=block.region_of_elem)
        poly = np.loadtxt(args.polyline, delimiter=",", dtype=float)
        curtain_from_cells(
            mesh,
            rho,
            poly,
            width=args.width,
            mode=args.mode,
            mask_max_edge=args.mask_max_edge,
            mask_max_area=args.mask_max_area,
            grid_zmin=args.zmin,
            grid_zmax=args.zmax,
            grid_nz=args.nz,
            grid_ns=args.ns,
            grid_k=args.k,
            grid_power=args.power,
            grid_max_dist=args.grid_max_dist,
        )
        if args.out:
            _require_mpl().savefig(args.out, dpi=200, bbox_inches="tight")
        else:
            _require_mpl().show()
        return 0

    if args.cmd == "map-scatter":
        mesh = read_femtic_mesh(args.mesh)
        block = read_resistivity_block(args.block)
        rho = map_regions_to_element_rho(block.region_of_elem, block.region_rho)
        rho = prepare_rho_for_plotting(rho, region_of_elem=block.region_of_elem)
        map_slice_scatter_from_cells(mesh, rho, z0=args.z0, dz=args.dz)
        if args.out:
            _require_mpl().savefig(args.out, dpi=200, bbox_inches="tight")
        else:
            _require_mpl().show()
        return 0

    if args.cmd == "curtain-scatter":
        mesh = read_femtic_mesh(args.mesh)
        block = read_resistivity_block(args.block)
        rho = map_regions_to_element_rho(block.region_of_elem, block.region_rho)
        rho = prepare_rho_for_plotting(rho, region_of_elem=block.region_of_elem)
        poly = np.loadtxt(args.polyline, delimiter=",", dtype=float)
        curtain_scatter_from_cells(mesh, rho, poly, width=args.width)
        if args.out:
            _require_mpl().savefig(args.out, dpi=200, bbox_inches="tight")
        else:
            _require_mpl().show()
        return 0

    if args.cmd == "curtain-idw":
        mesh = read_femtic_mesh(args.mesh)
        block = read_resistivity_block(args.block)
        rho = map_regions_to_element_rho(block.region_of_elem, block.region_rho)
        rho = prepare_rho_for_plotting(rho, region_of_elem=block.region_of_elem)
        poly = np.loadtxt(args.polyline, delimiter=",", dtype=float)
        s, z, V = curtain_grid_idw(
            mesh,
            rho,
            poly,
            zmin=args.zmin,
            zmax=args.zmax,
            nz=args.nz,
            ns=args.ns,
            k=args.k,
            power=args.power,
        )
        plot_curtain_matplotlib(s, z, V, log10=True)
        if args.out:
            _require_mpl().savefig(args.out, dpi=200, bbox_inches="tight")
        else:
            _require_mpl().show()
        return 0

    raise RuntimeError("Unhandled command.")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
