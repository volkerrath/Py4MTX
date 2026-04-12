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

Provenance:
    2025-12-23  vrath   Created (with ChatGPT GPT-5 Thinking).
    2026-03-24  Claude  Added plot_data_ensemble and plot_model_ensemble
                        for RTO ensemble diagnostic plots.
    2026-03-29  Claude  Added n_sites parameter to plot_data_ensemble for
                        random site sub-sampling per row.
                        Fixed ocean_value default: 1e-10 → 3e-1 Ohm.m in
                        prepare_rho_for_plotting and
                        unstructured_grid_from_femtic.
                        Added ocean_color parameter (default 'lightgrey') to
                        plot_points_matplotlib, plot_map_grid_matplotlib,
                        plot_curtain_matplotlib, map_slice_from_cells,
                        curtain_from_cells, and plot_model_ensemble.
    2026-03-30  Claude  Fixed plot_data_ensemble: replaced non-existent
                        fem.read_observe() with fem.read_observe_dat();
                        added _observe_to_site_list() helper to flatten the
                        nested blocks→sites structure and build Z/Z_err/T/P
                        complex arrays for data_viz.datadict_to_plot_df;
                        plotter now iterates per-site rather than passing the
                        whole file dict; n_sites sub-sampling now operates on
                        site index list rather than dict keys.
    2026-03-30  Claude  Fixed rho_a unit mismatch in _observe_to_site_list:
                        FEMTIC observe.dat Z is in SI Ohm; data_viz assumes
                        mV/km/nT (field units). Rewrote helper to use
                        fem.observe_to_site_viz_list() as authoritative reader,
                        then scales Z by 1/(mu0*1e3) to field units before
                        passing to datadict_to_plot_df. Also simplified
                        plot_data_ensemble: removed redundant read_observe_dat
                        pre-call; _observe_to_site_list now takes file path
                        directly.
    2026-03-30  Claude  Split show_errors into show_errors_orig and
                        show_errors_pert in plot_data_ensemble (default both
                        False); raw template errors are misleading at long
                        periods, perturbed files carry reset relative errors.
    2026-04-02  Claude  Added xlim/ylim/zlim parameters to plot_model_ensemble.
                        Map slices: xlim/ylim clip easting/northing axes.
                        Curtain slices: ylim clips profile-distance axis,
                        zlim clips the depth axis. Per-slice dict keys
                        override the function-level defaults.
    2026-04-03  Claude  Added alpha_orig/alpha_pert to plot_data_ensemble
                        (opacity per curve type; defaults 1.0/0.6).
                        Added mesh_lines/mesh_lw/mesh_color to
                        plot_model_ensemble, map_slice_from_cells,
                        curtain_from_cells, and plot_points_matplotlib;
                        triplot overlay drawn after tripcolor when enabled.
    2026-04-12  Claude  Added comp_markers to plot_data_ensemble: different
                        marker symbols for diagonal (ii), off-diagonal (ij),
                        and invariant components.  Default dict maps 'ii'->'o',
                        'ij'->'s', 'inv'->'^'; pass None to disable markers.
                        Added error_style_orig / error_style_pert parameters
                        ('shade' / 'bar' / 'both') for independent control of
                        error rendering on original vs. perturbed curves; no
                        shared fallback variable.
                        Markers applied post-hoc (snapshot ax.lines before
                        plotter call, patch new Line2D objects after).
                        Fixed legend: each component gets legend=True on its
                        first occurrence; final ax.legend() deduplicates.
                        plot_data_ensemble redesigned for multi-panel layouts:
                        'what' now accepts a list of panel types (e.g.
                        ['rho', 'phase', 'tipper']); 'comps' may be a single
                        string (shared) or a per-panel list; returned axs
                        shape is (n_samples, n_panels).
    2026-04-12  Claude  Added perlims / rholims / phslims / vtflims / ptlims
                        parameters to plot_data_ensemble for optional axis-
                        limit control.  perlims sets ax.set_xlim on every
                        panel; the remaining four set ax.set_ylim on their
                        respective panel type only.  Applied post-hoc after
                        each cell is filled so underlying plotters cannot
                        override them.
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
    ocean_value: Optional[float] = 3.0e-1,
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
        Default is 0.3 Ohm·m (typical seawater resistivity).
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
    ocean_value: Optional[float] = 3.0e-1,
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
    ocean_color: Optional[str] = "lightgrey",
    ocean_value: Optional[float] = 3.0e-1,
    mesh_lines: bool = False,
    mesh_lw: float = 0.3,
    mesh_color: str = "k",
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
    ocean_color : str or None, optional
        If not None, cells whose value (before log10) equals ``ocean_value``
        are coloured with this flat colour (default ``'lightgrey'``), keeping
        them visually distinct from the resistivity colour scale.
        Set to None to let ocean cells follow the normal colormap.
    ocean_value : float, optional
        Resistivity value (Ohm·m) used to identify ocean cells.
        Should match the value passed to :func:`prepare_rho_for_plotting`
        (default 0.3 Ohm·m).
    mesh_lines : bool, optional
        If ``True`` and ``mode="tri"``, overlay the triangulation edges
        (``ax.triplot``) on top of the filled patches.  Default ``False``.
    mesh_lw : float, optional
        Line width for mesh edge overlay (default 0.3).
    mesh_color : str, optional
        Colour for mesh edge overlay (default ``'k'``).

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
    values = np.asarray(values, dtype=float)

    # Identify ocean cells before log10 transformation
    _ocean_mask: Optional[np.ndarray] = None
    if ocean_color is not None and ocean_value is not None:
        _ocean_mask = np.isclose(values, float(ocean_value), rtol=1e-6, atol=0.0)

    v_plot, label = _apply_log10(values, log10=log10)

    if ax is None:
        _, ax = plt_.subplots()

    if mode == "scatter":
        sc = ax.scatter(x, y, c=v_plot, s=int(s))
        plt_.colorbar(sc, ax=ax, label=label)
        # Overlay ocean cells in flat colour
        if _ocean_mask is not None and np.any(_ocean_mask):
            ax.scatter(x[_ocean_mask], y[_ocean_mask],
                       c=ocean_color, s=int(s), zorder=sc.get_zorder() + 1)
    elif mode == "tri":
        import matplotlib.tri as mtri  # local import (optional dependency)

        tri = mtri.Triangulation(x, y)
        mask = _triangulation_mask(
            x, y, tri.triangles, max_edge=mask_max_edge, max_area=mask_max_area
        )
        # Also mask triangles whose centroid falls in the ocean
        if _ocean_mask is not None and np.any(_ocean_mask):
            ocean_tri_mask = _ocean_mask[tri.triangles].any(axis=1)
            mask = mask | ocean_tri_mask if np.any(mask) else ocean_tri_mask
        if np.any(mask):
            tri.set_mask(mask)

        pc = ax.tripcolor(tri, v_plot, shading="flat")
        plt_.colorbar(pc, ax=ax, label=label)

        # Optional mesh edge overlay
        if mesh_lines:
            ax.triplot(tri, color=mesh_color, lw=mesh_lw, zorder=pc.get_zorder() + 0.5)

        # Draw ocean triangles as flat-coloured patches on top
        if _ocean_mask is not None and np.any(_ocean_mask):
            tri_ocean = mtri.Triangulation(x, y)
            ocean_only_mask = ~_ocean_mask[tri_ocean.triangles].all(axis=1)
            tri_ocean.set_mask(ocean_only_mask)
            ax.tripcolor(tri_ocean, np.ones(len(x)),
                         shading="flat", cmap=None,
                         facecolor=ocean_color, edgecolors="none",
                         zorder=pc.get_zorder() + 1)
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
    ocean_color: Optional[str] = "lightgrey",
    ocean_value: Optional[float] = 3.0e-1,
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
    ocean_color : str or None, optional
        Flat colour for ocean cells (default ``'lightgrey'``).
        Ocean cells are identified by ``ocean_value`` before log10.
        Set to None to let them follow the colormap.
    ocean_value : float, optional
        Resistivity (Ohm·m) marking ocean cells (default 0.3).

    Returns
    -------
    ax
        Matplotlib axes.
    """
    plt_ = _require_mpl()

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    V = np.asarray(V, dtype=float)

    # Mask ocean cells so they are rendered separately
    _ocean_2d: Optional[np.ndarray] = None
    if ocean_color is not None and ocean_value is not None:
        _ocean_2d = np.isclose(V, float(ocean_value), rtol=1e-6, atol=0.0)
        V = np.where(_ocean_2d, np.nan, V)

    Vp, label = _apply_log10(V, log10=log10)

    if ax is None:
        _, ax = plt_.subplots()

    import matplotlib.colors as mcolors  # local import
    _cmap_obj = plt_.get_cmap(cmap) if cmap else plt_.get_cmap()
    _cmap_obj = _cmap_obj.copy()
    if ocean_color is not None:
        _cmap_obj.set_bad(color=ocean_color)

    im = ax.pcolormesh(x, y, Vp, shading="auto", cmap=_cmap_obj)
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
    ocean_color: Optional[str] = "lightgrey",
    ocean_value: Optional[float] = 3.0e-1,
    mesh_lines: bool = False,
    mesh_lw: float = 0.3,
    mesh_color: str = "k",
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
        Optional Matplotlib colormap name.
    ocean_color : str or None, optional
        Flat colour for ocean cells (default ``'lightgrey'``).
        Forwarded to :func:`plot_points_matplotlib` or
        :func:`plot_map_grid_matplotlib`.  Set to None to use the colormap.
    ocean_value : float, optional
        Resistivity (Ohm·m) identifying ocean cells (default 0.3).

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
            ocean_color=ocean_color,
            ocean_value=ocean_value,
            mesh_lines=mesh_lines,
            mesh_lw=mesh_lw,
            mesh_color=mesh_color,
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
        ax = plot_map_grid_matplotlib(
            xg, yg, V, log10=log10, ax=ax, cmap=cmap, title=title,
            ocean_color=ocean_color, ocean_value=ocean_value,
        )
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
    ocean_color: Optional[str] = "lightgrey",
    ocean_value: Optional[float] = 3.0e-1,
    mesh_lines: bool = False,
    mesh_lw: float = 0.3,
    mesh_color: str = "k",
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
        Optional Matplotlib colormap name.
    ocean_color : str or None, optional
        Flat colour for ocean cells (default ``'lightgrey'``).
        Forwarded to :func:`plot_points_matplotlib` or
        :func:`plot_curtain_matplotlib`.  Set to None to use the colormap.
    ocean_value : float, optional
        Resistivity (Ohm·m) identifying ocean cells (default 0.3).

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
            ocean_color=ocean_color,
            ocean_value=ocean_value,
            mesh_lines=mesh_lines,
            mesh_lw=mesh_lw,
            mesh_color=mesh_color,
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
        ax = plot_curtain_matplotlib(
            s1, z1, V, log10=log10, ax=ax, cmap=cmap,
            ocean_color=ocean_color, ocean_value=ocean_value,
        )
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
    ocean_color: Optional[str] = "lightgrey",
    ocean_value: Optional[float] = 3.0e-1,
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
    ocean_color : str or None, optional
        Flat colour for ocean cells (default ``'lightgrey'``).
        Ocean cells are identified by ``ocean_value`` before log10.
        Set to None to let them follow the colormap.
    ocean_value : float, optional
        Resistivity (Ohm·m) marking ocean cells (default 0.3).

    Returns
    -------
    ax
        Matplotlib axes.
    """
    plt_ = _require_mpl()

    s = np.asarray(s, dtype=float)
    z = np.asarray(z, dtype=float)
    V = np.asarray(V, dtype=float)

    # Mask ocean cells before log10 so they render in ocean_color
    if ocean_color is not None and ocean_value is not None:
        _ocean_2d = np.isclose(V, float(ocean_value), rtol=1e-6, atol=0.0)
        V = np.where(_ocean_2d, np.nan, V)

    if log10:
        with np.errstate(divide="ignore", invalid="ignore"):
            Vp = np.log10(V)
        label = "log10(resistivity)"
    else:
        Vp = V
        label = "resistivity"

    if ax is None:
        _, ax = plt_.subplots()

    _cmap_obj = plt_.get_cmap(cmap) if cmap else plt_.get_cmap()
    _cmap_obj = _cmap_obj.copy()
    if ocean_color is not None:
        _cmap_obj.set_bad(color=ocean_color)

    im = ax.pcolormesh(s, z, Vp, shading="auto", cmap=_cmap_obj)
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
# RTO ensemble diagnostic plots
# =============================================================================

_MU0 = 4.0e-7 * np.pi          # H/m
_Z_SI_TO_MT = 1.0 / (_MU0 * 1.0e3)   # SI Ohm → mV/km/nT

# ---------------------------------------------------------------------------
# Component classification for marker differentiation
# ---------------------------------------------------------------------------

#: Default marker symbols per component class.
#: Keys: ``'ii'`` (diagonal: xx, yy), ``'ij'`` (off-diagonal: xy, yx),
#: ``'inv'`` (invariants: det, bahr, ssq, …).
#: Pass ``comp_markers=None`` to :func:`plot_data_ensemble` to disable markers.
DEFAULT_COMP_MARKERS: dict = {
    "ii":  "o",   # diagonal — circles
    "ij":  "s",   # off-diagonal — squares
    "inv": "^",   # invariants — triangles-up
}

# Set of recognised diagonal component labels (lowercase).
_DIAG_COMPS = {"xx", "yy"}
# Set of recognised off-diagonal component labels (lowercase).
_OFFDIAG_COMPS = {"xy", "yx"}


def _comp_class(comp: str) -> str:
    """Return the class label (``'ii'``, ``'ij'``, or ``'inv'``) for *comp*.

    Parameters
    ----------
    comp : str
        Component name, e.g. ``'xx'``, ``'xy'``, ``'zx'``, ``'det'``.
        Comparison is case-insensitive.
    """
    c = comp.strip().lower()
    if c in _DIAG_COMPS:
        return "ii"
    if c in _OFFDIAG_COMPS:
        return "ij"
    return "inv"


def _observe_to_site_list(
    observe_path: Union[str, Path],
    fem_mod: Any,
) -> list:
    """Return a list of per-site dicts from a FEMTIC ``observe.dat`` file,
    ready for passing to ``data_viz.add_*`` plotters.

    Uses :func:`femtic.observe_to_site_viz_list` as the single authoritative
    reader so that apparent-resistivity computation uses the correct FEMTIC
    unit convention (Z in SI Ω; ``rho_a = |Z|² / (μ₀ω)``).

    The returned dicts contain ``Z`` scaled to MT field units (mV/km/nT) so
    that :func:`data_viz.datadict_to_plot_df` — which assumes field-unit Z
    and applies ``rho = |Z|² × μ₀ × 10⁶ / ω`` — produces the right answer.
    For VTF sites ``T`` is built from the raw data columns; for PT sites ``P``
    is built similarly.

    Parameters
    ----------
    observe_path : str or Path
        Path to ``observe.dat``.
    fem_mod : module
        The already-imported ``femtic`` module.

    Returns
    -------
    list of dict
        One dict per site.
    """
    # observe_to_site_viz_list returns Z in SI Ohm and pre-computed rhoa/phase_deg.
    raw_sites = fem_mod.observe_to_site_viz_list(
        observe_path,
        obs_type="MT",
        add_rhoa_phase=True,
        mc_n=0,            # skip MC errors for speed in diagnostic plots
    )

    result = []
    for s in raw_sites:
        d: dict = {
            "freq":    s["freq"],
            "obs_type": s["obs_type"],
            "name":    s.get("name", ""),
        }

        # Convert Z from SI Ohm to mV/km/nT so datadict_to_plot_df gives
        # correct rho_a values (it assumes field-unit Z).
        Z_si = np.asarray(s["Z"], dtype=np.complex128)   # (nfreq, 2, 2)
        d["Z"] = Z_si * _Z_SI_TO_MT

        Zerr = s.get("Zerr")
        if Zerr is not None:
            d["Z_err"] = np.asarray(Zerr, dtype=np.complex128) * _Z_SI_TO_MT

        result.append(d)

    # Also read VTF and PT sites via the raw parser (observe_to_site_viz_list
    # only returns MT sites in the current call above).
    parsed = fem_mod.read_observe_dat(str(observe_path))
    for s in fem_mod.sites_as_dict_list(parsed):
        obs = str(s.get("obs_type", "")).upper()
        if obs == "MT":
            continue   # already handled above
        d = {"freq": np.asarray(s["freq"], dtype=float), "obs_type": obs,
             "name": s.get("site_header_tokens", [""])[0]}
        raw = np.asarray(s["data"], dtype=float)
        err = np.asarray(s["error"], dtype=float)
        if obs == "VTF" and raw.shape[1] >= 4:
            T = np.empty((raw.shape[0], 2), dtype=np.complex128)
            T[:, 0] = raw[:, 0] + 1j * raw[:, 1]
            T[:, 1] = raw[:, 2] + 1j * raw[:, 3]
            Terr = np.empty_like(T)
            Terr[:, 0] = err[:, 0] + 1j * err[:, 1]
            Terr[:, 1] = err[:, 2] + 1j * err[:, 3]
            d["T"] = T
            d["T_err"] = Terr
        elif obs == "PT" and raw.shape[1] >= 4:
            d["P"] = raw[:, :4].reshape(-1, 2, 2)
            d["P_err"] = err[:, :4].reshape(-1, 2, 2)
        result.append(d)

    return result


def plot_data_ensemble(
    orig_file: Union[str, Path],
    ens_files: Sequence[Union[str, Path]],
    sample_indices: Sequence[int],
    *,
    what: Union[str, Sequence[str]] = "rho",
    comps: Union[str, Sequence[str]] = "xy,yx",
    show_errors: bool = False,
    show_errors_orig: Optional[bool] = None,
    show_errors_pert: Optional[bool] = None,
    error_style_orig: str = "shade",
    error_style_pert: str = "shade",
    n_sites: Optional[int] = None,
    alpha_orig: float = 1.0,
    alpha_pert: float = 0.6,
    comp_markers: Optional[dict] = None,
    markersize: float = 4.0,
    markevery: Optional[int] = None,
    perlims: Optional[Tuple[float, float]] = None,
    rholims: Optional[Tuple[float, float]] = None,
    phslims: Optional[Tuple[float, float]] = None,
    vtflims: Optional[Tuple[float, float]] = None,
    ptlims: Optional[Tuple[float, float]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    fig: Optional[Any] = None,
    axs: Optional[Any] = None,
    out: bool = True,
) -> Tuple[Any, np.ndarray]:
    """Joint plot of original and perturbed MT data for a fixed list of samples.

    Produces a 2-D grid of axes: **rows** = ensemble members (one per entry in
    *sample_indices*); **columns** = plot panels (one per entry in *what*).
    Within each cell the original curves (solid) and the perturbed curves
    (dashed) for a randomly drawn subset of MT sites are overlaid on the same
    axes.

    Follows the ``data_viz`` philosophy: if *fig* / *axs* are provided they
    are used directly; otherwise a new figure is created.  ``(fig, axs)`` is
    always returned so the caller can annotate or save.

    Parameters
    ----------
    orig_file : str or Path
        Path to the reference ``observe.dat`` (template / original data).
    ens_files : sequence of str or Path
        Paths to the perturbed ``observe.dat`` files, one per ensemble member.
    sample_indices : sequence of int
        Indices into ``ens_files`` that should be plotted (e.g. ``[0, 3, 7]``).
    what : str or sequence of str, optional
        Which MT quantity (or quantities) to plot.  Each entry must be one of
        ``'rho'``, ``'phase'``, ``'tipper'``, ``'pt'``.  A plain string is
        treated as a single-panel request.  A list produces one column per
        entry, e.g. ``['rho', 'phase']`` gives a two-column layout with
        apparent resistivity on the left and phase on the right.
        Default is ``'rho'``.
    comps : str or sequence of str, optional
        Impedance components for ``'rho'`` / ``'phase'`` columns.
        A single string (e.g. ``'xy,yx'``) is used for every ``'rho'`` /
        ``'phase'`` column.  A sequence of the same length as *what* provides
        per-column control; use ``None`` or ``''`` for ``'tipper'`` /
        ``'pt'`` columns (the value is ignored for those plotters anyway).
        Default is ``'xy,yx'``.
    show_errors : bool, optional
        Shorthand for both ``show_errors_orig`` and ``show_errors_pert``.
        Default ``False``.
    show_errors_orig : bool or None, optional
        If ``True``, draw error envelopes on the **original** curves.
        Overrides ``show_errors``.  Default ``None`` (falls back to
        ``show_errors``).
    show_errors_pert : bool or None, optional
        If ``True``, draw error envelopes on the **perturbed** curves.
        Overrides ``show_errors``.  Default ``None``.
    error_style_orig : str, optional
        Error rendering for the **original** curves: ``'shade'`` (default),
        ``'bar'``, or ``'both'``.
    error_style_pert : str, optional
        Error rendering for the **perturbed** curves: ``'shade'`` (default),
        ``'bar'``, or ``'both'``.
    n_sites : int or None, optional
        Number of MT sites drawn (without replacement) per row.  The same
        random subset is used across all columns in a row.  ``None`` = all
        sites.
    alpha_orig : float, optional
        Opacity for the **original** curves (default ``1.0``).
    alpha_pert : float, optional
        Opacity for the **perturbed** curves (default ``0.6``).
    comp_markers : dict or None, optional
        Marker symbols keyed by component class (``'ii'``, ``'ij'``,
        ``'inv'``).  ``None`` uses :data:`DEFAULT_COMP_MARKERS`; ``{}``
        disables markers.  Partial dicts are merged with the defaults.
    markersize : float, optional
        Marker size in points (default ``4.0``).
    markevery : int or None, optional
        Mark every *N*-th period; ``None`` = every period.
    perlims : (float, float) or None, optional
        Period-axis limits ``(T_min, T_max)`` in seconds applied to every
        panel via ``ax.set_xlim``.  ``None`` = Matplotlib auto-scaling.
    rholims : (float, float) or None, optional
        y-axis limits for ``'rho'`` panels (apparent resistivity, Ω·m or
        log₁₀(Ω·m) depending on the plotter).  ``None`` = auto.
    phslims : (float, float) or None, optional
        y-axis limits for ``'phase'`` panels (degrees).  ``None`` = auto.
    vtflims : (float, float) or None, optional
        y-axis limits for ``'tipper'`` panels.  ``None`` = auto.
    ptlims : (float, float) or None, optional
        y-axis limits for ``'pt'`` panels.  ``None`` = auto.
    figsize : (float, float) or None, optional
        Figure size.  Auto-sized if ``None``:
        width ≈ 5 × n_panels, height ≈ 3 × n_samples.
    fig : matplotlib.figure.Figure or None, optional
        Pre-existing figure.  If ``None`` a new one is created.
    axs : array-like of Axes or None, optional
        Pre-allocated axes of shape ``(n_samples, n_panels)`` (or ravelled
        equivalent).  If ``None``, subplots are created automatically.
    out : bool, optional
        Print progress messages.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : numpy.ndarray of matplotlib.axes.Axes
        Shape ``(n_samples, n_panels)``.  When *what* is a single string the
        shape is ``(n_samples, 1)`` — index with ``axs[:, 0]`` if you need the
        1-D array of rows.

    Notes
    -----
    **Component markers** are applied post-hoc (snapshot ``ax.lines`` before
    each plotter call, patch the new ``Line2D`` objects after) to avoid a
    ``TypeError`` from any hardcoded ``marker=`` argument inside ``data_viz``.

    **Error styles** ``'bar'`` / ``'both'`` require the ``data_viz`` plotter
    to accept an ``error_style`` kwarg; otherwise falls back to ``'shade'``
    with a one-time ``warnings.warn``.

    Data files are read via :func:`femtic.observe_to_site_viz_list` (through
    ``_observe_to_site_list``).  FEMTIC stores Z in SI Ω; the helper converts
    to mV/km/nT before passing to ``data_viz.datadict_to_plot_df``.
    """
    plt_ = _require_mpl()

    try:
        import femtic as fem  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("femtic.py is required for plot_data_ensemble.") from e

    try:
        import data_viz  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("data_viz.py is required for plot_data_ensemble.") from e

    _plotter_map = {
        "rho":    data_viz.add_rho,
        "phase":  data_viz.add_phase,
        "tipper": data_viz.add_tipper,
        "pt":     data_viz.add_pt,
    }

    # ------------------------------------------------------------------
    # Normalise 'what' → list of panel names; validate each entry.
    # ------------------------------------------------------------------
    if isinstance(what, str):
        what_list = [what]
    else:
        what_list = list(what)
    for w in what_list:
        if str(w).lower() not in _plotter_map:
            raise ValueError(
                f"plot_data_ensemble: unknown panel type {w!r}. "
                f"Choose from {list(_plotter_map)}."
            )
    n_panels = len(what_list)

    # ------------------------------------------------------------------
    # Normalise 'comps' → list of length n_panels.
    # For tipper/pt entries the value is ignored but must be present.
    # ------------------------------------------------------------------
    _needs_comps = {"rho", "phase"}
    if isinstance(comps, str):
        comps_list = [comps] * n_panels
    else:
        comps_list = list(comps)
        if len(comps_list) != n_panels:
            raise ValueError(
                f"plot_data_ensemble: 'comps' has {len(comps_list)} entries "
                f"but 'what' has {n_panels}."
            )

    n_samples = len(sample_indices)
    if n_samples == 0:
        raise ValueError("plot_data_ensemble: sample_indices is empty.")

    # ------------------------------------------------------------------
    # Figure / axes setup.
    # axs_arr shape: (n_samples, n_panels)
    # ------------------------------------------------------------------
    if figsize is None:
        figsize = (5 * n_panels, 3 * n_samples)

    if fig is None or axs is None:
        fig, _axs = plt_.subplots(
            n_samples, n_panels,
            figsize=figsize,
            squeeze=False,
        )
        axs_arr = _axs                       # shape (n_samples, n_panels)
    else:
        axs_arr = np.asarray(axs).reshape(n_samples, n_panels)

    # ------------------------------------------------------------------
    # Shared flags.
    # ------------------------------------------------------------------
    _show_orig = show_errors if show_errors_orig is None else show_errors_orig
    _show_pert = show_errors if show_errors_pert is None else show_errors_pert

    _valid_styles = {"shade", "bar", "both"}
    _style_orig = error_style_orig if error_style_orig in _valid_styles else "shade"
    _style_pert = error_style_pert if error_style_pert in _valid_styles else "shade"

    # ------------------------------------------------------------------
    # Marker dict resolution.
    # ------------------------------------------------------------------
    _use_markers = True
    if comp_markers is None:
        _markers = dict(DEFAULT_COMP_MARKERS)
    elif len(comp_markers) == 0:
        _use_markers = False
        _markers = {}
    else:
        _markers = {**DEFAULT_COMP_MARKERS, **comp_markers}

    # ------------------------------------------------------------------
    # Per-panel component lists (empty list → tipper/pt, no per-comp loop).
    # ------------------------------------------------------------------
    _panel_comp_lists = []
    for w, c in zip(what_list, comps_list):
        if str(w).lower() in _needs_comps and c:
            _panel_comp_lists.append([s.strip() for s in c.split(",")])
        else:
            _panel_comp_lists.append([])

    # ------------------------------------------------------------------
    # Per-panel y-axis limit lookup.
    # Maps panel type → the caller-supplied limit (or None = auto).
    # ------------------------------------------------------------------
    _ylim_map = {
        "rho":    rholims,
        "phase":  phslims,
        "tipper": vtflims,
        "pt":     ptlims,
    }

    # ------------------------------------------------------------------
    # error_style probe (once per panel plotter, shared warning state).
    # ------------------------------------------------------------------
    _es_warned: dict = {}   # plotter_name → bool

    def _error_style_kw(plotter: Any, style: str) -> dict:
        if style == "shade":
            return {}
        pname = getattr(plotter, "__name__", str(plotter))
        try:
            import inspect as _insp
            if "error_style" in _insp.signature(plotter).parameters:
                return {"error_style": style}
            if not _es_warned.get(pname, False):
                import warnings
                warnings.warn(
                    f"data_viz.{pname} does not accept 'error_style'; "
                    f"falling back to 'shade'.",
                    stacklevel=5,
                )
                _es_warned[pname] = True
        except Exception:
            pass
        return {}

    # ------------------------------------------------------------------
    # Post-hoc marker helpers.
    # ------------------------------------------------------------------
    def _apply_markers(ax: Any, lines_before: list, marker: str) -> None:
        for line in ax.lines:
            if line not in lines_before:
                line.set_marker(marker)
                line.set_markersize(markersize)
                if markevery is not None:
                    line.set_markevery(markevery)

    def _marker_for(comp: str) -> str:
        if not _use_markers:
            return "none"
        return _markers.get(_comp_class(comp), "none")

    # ------------------------------------------------------------------
    # Helper: plot one panel (one (sample_row, panel_col) cell).
    # ------------------------------------------------------------------
    def _plot_panel(
        ax: Any,
        plotter: Any,
        comp_list: list,
        sites: list,
        site_idx: list,
        show_err: bool,
        style: str,
        alpha: float,
        linestyle: str,
        add_to_legend: bool,        # True only for the original curves
        legend_done: set,
    ) -> None:
        """Draw all sites onto *ax* for one panel."""

        def _need_legend(key: str) -> bool:
            if not add_to_legend or key in legend_done:
                return False
            legend_done.add(key)
            return True

        es_kw = _error_style_kw(plotter, style)

        if comp_list:
            for si in site_idx:
                if si >= len(sites):
                    continue
                site = sites[si]
                for comp in comp_list:
                    lb = list(ax.lines)
                    plotter(site, ax=ax, show_errors=show_err,
                            legend=_need_legend(comp),
                            alpha=alpha, linestyle=linestyle,
                            comps=comp, **es_kw)
                    _apply_markers(ax, lb, _marker_for(comp))
        else:
            # tipper / pt: one call per site
            for si in site_idx:
                if si >= len(sites):
                    continue
                site = sites[si]
                lb = list(ax.lines)
                plotter(site, ax=ax, show_errors=show_err,
                        legend=_need_legend("_panel_"),
                        alpha=alpha, linestyle=linestyle, **es_kw)
                _apply_markers(ax, lb, _marker_for("inv"))

    # ------------------------------------------------------------------
    # Read original sites once (shared across all rows and columns).
    # ------------------------------------------------------------------
    orig_sites = _observe_to_site_list(orig_file, fem)
    if out:
        print(f"plot_data_ensemble: original read from {orig_file} "
              f"({len(orig_sites)} sites)")

    _rng = np.random.default_rng()
    n_total = len(orig_sites)
    if n_sites is not None and n_total > 0:
        _n = min(n_sites, n_total)
        _site_idx = sorted(_rng.choice(n_total, size=_n, replace=False).tolist())
    else:
        _site_idx = list(range(n_total))

    # ------------------------------------------------------------------
    # Main loop: rows = samples, columns = panels.
    # ------------------------------------------------------------------
    for row, idx in enumerate(sample_indices):

        # Read perturbed sites once per row (shared across columns).
        pert_sites = _observe_to_site_list(ens_files[idx], fem)

        for col, (w, plotter) in enumerate(
            zip(what_list, [_plotter_map[str(w).lower()] for w in what_list])
        ):
            ax = axs_arr[row, col]
            comp_list = _panel_comp_lists[col]
            _legend_done: set = set()

            # original curves (solid, alpha_orig)
            _plot_panel(ax, plotter, comp_list,
                        orig_sites, _site_idx,
                        _show_orig, _style_orig, alpha_orig,
                        linestyle="-",
                        add_to_legend=True,
                        legend_done=_legend_done)

            # perturbed curves (dashed, alpha_pert, no legend entries)
            _plot_panel(ax, plotter, comp_list,
                        pert_sites, _site_idx,
                        _show_pert, _style_pert, alpha_pert,
                        linestyle="--",
                        add_to_legend=False,
                        legend_done=_legend_done)

            # Deduplicated legend (removes duplicates from internal calls).
            handles, labels = ax.get_legend_handles_labels()
            seen: dict = {}
            for h, l in zip(handles, labels):
                if l not in seen:
                    seen[l] = h
            if seen:
                ax.legend(seen.values(), seen.keys(), fontsize=8)

            # Axis limits — applied post-hoc so plotters can't override them.
            if perlims is not None:
                ax.set_xlim(perlims)
            _yl = _ylim_map.get(str(w).lower())
            if _yl is not None:
                ax.set_ylim(_yl)

            # Column header on first row only.
            if row == 0:
                ax.set_title(f"{w.capitalize()} — sample {idx}", fontsize=9)
            else:
                ax.set_title(f"Sample {idx}", fontsize=9)

        if out:
            print(f"  sample {idx} done ({len(_site_idx)} sites, "
                  f"{n_panels} panel(s))")

    fig.tight_layout()
    return fig, axs_arr


def plot_model_ensemble(
    orig_mod_file: Union[str, Path],
    ens_mod_files: Sequence[Union[str, Path]],
    mesh_file: Union[str, Path],
    sample_indices: Sequence[int],
    slices: Sequence[dict],
    *,
    mode: Literal["scatter", "tri", "grid"] = "tri",
    log10: bool = True,
    cmap: Optional[str] = "jet_r",
    clim: Optional[Tuple[float, float]] = None,
    ocean_color: Optional[str] = "lightgrey",
    ocean_value: Optional[float] = 3.0e-1,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    zlim: Optional[Tuple[float, float]] = None,
    mesh_lines: bool = False,
    mesh_lw: float = 0.3,
    mesh_color: str = "k",
    figsize: Optional[Tuple[float, float]] = None,
    fig: Optional[Any] = None,
    axs: Optional[Any] = None,
    out: bool = True,
) -> Tuple[Any, np.ndarray]:
    """Joint plot of original and perturbed resistivity models.

    Produces a grid of Matplotlib axes: **rows** = 2 × number of selected
    samples (original row then perturbed row per block); **columns** = number
    of slices.  Original and perturbed models are placed directly above/below
    each other for easy visual comparison across all requested slices.

    Follows the ``data_viz`` philosophy: if *fig* / *axs* are provided they
    are used directly; otherwise a new figure is created.  ``(fig, axs)`` is
    always returned.

    Parameters
    ----------
    orig_mod_file : str or Path
        Path to the reference (template) ``resistivity_block_iterX.dat``.
    ens_mod_files : sequence of str or Path
        Paths to the perturbed resistivity block files, one per ensemble member.
    mesh_file : str or Path
        Path to the shared ``mesh.dat``.
    sample_indices : sequence of int
        Indices into ``ens_mod_files`` to visualise (e.g. ``[0, 2, 5]``).
    slices : sequence of dict
        Between 1 and 5 slice descriptors.  Each dict **must** have a
        ``'type'`` key (``'map'`` or ``'curtain'``), plus the keyword
        arguments forwarded to :func:`map_slice_from_cells` or
        :func:`curtain_from_cells`.

        Map-slice example::

            {'type': 'map', 'z0': -1000, 'dz': 50}

        Curtain-slice example::

            {'type': 'curtain',
             'polyline': np.array([[x0, y0], [x1, y1]]),
             'width': 500}

    mode : {'scatter', 'tri', 'grid'}, optional
        Slice rendering mode forwarded to the slicer functions.  Default
        is ``'tri'``.
    log10 : bool, optional
        Plot log₁₀(ρ) if ``True`` (default).
    cmap : str or None, optional
        Matplotlib colormap (default ``'jet_r'``).
    clim : (float, float) or None, optional
        Colour limits ``(vmin, vmax)`` in log₁₀(Ω·m).
        If ``None``, limits are derived from the original model.
    xlim : (float, float) or None, optional
        x-axis limits applied to **map** slices (easting range, metres).
        If ``None``, Matplotlib auto-scales.  Individual slice dicts may
        also carry an ``'xlim'`` key to override per-slice.
    ylim : (float, float) or None, optional
        y-axis limits applied to **map** slices (northing range, metres).
        For **curtain** slices this sets the along-profile distance axis
        (horizontal axis).  Individual slice dicts may carry a ``'ylim'`` key.
    zlim : (float, float) or None, optional
        Depth-axis limits applied to **curtain** slices (vertical axis, metres,
        negative-down convention).  Individual slice dicts may carry a
        ``'zlim'`` key.
    mesh_lines : bool, optional
        If ``True``, overlay the Delaunay triangulation edges (``ax.triplot``)
        on top of the filled patches in ``mode='tri'`` panels.  Useful for
        inspecting mesh resolution in ensemble diagnostics.  Default ``False``.
        Per-slice dicts may carry a ``'mesh_lines'`` key to override per-slice.
    mesh_lw : float, optional
        Line width for the mesh edge overlay (default ``0.3``).
    mesh_color : str, optional
        Colour for the mesh edge overlay (default ``'k'``).
    ocean_color : str or None, optional
        Flat colour for ocean cells across all panels (default ``'lightgrey'``).
        Forwarded to the slicer functions.  Set to None to use the colormap.
    ocean_value : float, optional
        Resistivity (Ohm·m) identifying ocean cells (default 0.3).
    figsize : (float, float) or None, optional
        Figure size in inches.  If ``None``, a sensible default is chosen.
    fig : matplotlib.figure.Figure or None, optional
        Pre-existing figure.
    axs : array-like of Axes or None, optional
        Pre-allocated axes of shape
        ``(2 * len(sample_indices), len(slices))``.
        If ``None``, subplots are created automatically.
    out : bool, optional
        If ``True``, print progress messages.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : numpy.ndarray of matplotlib.axes.Axes
        Shape ``(2 * len(sample_indices), len(slices))``.

    Notes
    -----
    The mesh is read once and shared across all samples and slices.
    Colour limits are optionally set globally from the original model so that
    all panels are directly comparable.
    """
    plt_ = _require_mpl()

    n_samples = len(sample_indices)
    n_slices = len(slices)
    if n_samples == 0:
        raise ValueError("plot_model_ensemble: sample_indices is empty.")
    if not (1 <= n_slices <= 5):
        raise ValueError("plot_model_ensemble: slices must have 1–5 entries.")

    n_rows = 2 * n_samples
    if figsize is None:
        figsize = (4 * n_slices, 3 * n_rows)

    if fig is None or axs is None:
        fig, _axs = plt_.subplots(n_rows, n_slices, figsize=figsize, squeeze=False)
        axs_arr = _axs
    else:
        axs_arr = np.asarray(axs)
        if axs_arr.shape != (n_rows, n_slices):
            raise ValueError(
                f"plot_model_ensemble: axs shape {axs_arr.shape} != "
                f"expected ({n_rows}, {n_slices})."
            )

    # Read mesh once — shared by all runs
    mesh = read_femtic_mesh(mesh_file)
    if out:
        print(f"plot_model_ensemble: mesh read from {mesh_file}")

    # Read original block and build element-wise rho
    orig_block = read_resistivity_block(orig_mod_file)
    rho_orig = map_regions_to_element_rho(
        orig_block.region_of_elem, orig_block.region_rho
    )
    rho_orig = prepare_rho_for_plotting(
        rho_orig, region_of_elem=orig_block.region_of_elem,
        ocean_value=ocean_value,
    )

    # Derive colour limits from original model if not supplied
    if clim is None:
        vals = np.log10(rho_orig[np.isfinite(rho_orig) & (rho_orig > 0)])
        clim = (float(vals.min()), float(vals.max())) if vals.size > 0 else (0.0, 4.0)
    vmin, vmax = clim

    def _draw_slice(ax: Any, rho: np.ndarray, slc_spec: dict, title: str = "") -> None:
        """Dispatch one slice descriptor to the appropriate slicer and draw on ax."""
        slc = dict(slc_spec)          # copy — we pop 'type' and axis-limit overrides
        slc_type = slc.pop("type", "map").lower()
        # Per-slice axis-limit overrides take priority over function-level defaults
        slc_xlim = slc.pop("xlim", xlim)
        slc_ylim = slc.pop("ylim", ylim)
        slc_zlim = slc.pop("zlim", zlim)
        slc.setdefault("mode", mode)
        slc.setdefault("log10", log10)
        slc.setdefault("cmap", cmap)
        slc.setdefault("ocean_color", ocean_color)
        slc.setdefault("ocean_value", ocean_value)
        slc.setdefault("mesh_lines", mesh_lines)
        slc.setdefault("mesh_lw", mesh_lw)
        slc.setdefault("mesh_color", mesh_color)

        if "map" in slc_type:
            map_slice_from_cells(mesh, rho, ax=ax, **slc)
            # map slices: x = easting, y = northing
            if slc_xlim is not None:
                ax.set_xlim(slc_xlim)
            if slc_ylim is not None:
                ax.set_ylim(slc_ylim)
        else:
            poly = slc.pop("polyline")
            curtain_from_cells(mesh, rho, poly, ax=ax, **slc)
            # curtain slices: horizontal = along-profile distance, vertical = depth
            if slc_ylim is not None:
                ax.set_xlim(slc_ylim)   # ylim controls the profile-distance axis
            if slc_zlim is not None:
                ax.set_ylim(slc_zlim)   # zlim controls the depth axis

        # Apply shared colour limits after plotting
        for img in ax.get_images():
            img.set_clim(vmin, vmax)
        for coll in ax.collections:
            coll.set_clim(vmin, vmax)

        if title:
            ax.set_title(title, fontsize=8)

    for block_row, idx in enumerate(sample_indices):
        pert_block = read_resistivity_block(ens_mod_files[idx])
        rho_pert = map_regions_to_element_rho(
            pert_block.region_of_elem, pert_block.region_rho
        )
        rho_pert = prepare_rho_for_plotting(
            rho_pert, region_of_elem=pert_block.region_of_elem,
            ocean_value=ocean_value,
        )

        for col, slc_spec in enumerate(slices):
            row_orig = 2 * block_row
            row_pert = 2 * block_row + 1

            # Column header (first block only)
            col_label = f"slice {col}" if block_row == 0 else ""

            ax_orig = axs_arr[row_orig, col]
            _draw_slice(ax_orig, rho_orig, slc_spec, title=col_label)
            if col == 0:
                ax_orig.set_ylabel(f"orig\n(sample {idx})", fontsize=8)

            ax_pert = axs_arr[row_pert, col]
            _draw_slice(ax_pert, rho_pert, slc_spec)
            if col == 0:
                ax_pert.set_ylabel(f"pert {idx}", fontsize=8)

        if out:
            print(f"  sample {idx} done")

    fig.tight_layout()
    return fig, axs_arr


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
