
"""
femtic_viz.py

Visualization utilities for FEMTIC meshes using pure Matplotlib and PyVista.

This module offers:
- Robust FEMTIC TETRA mesh loader (nodes, elements, regions) and optional resistivity mapping.
- Pure Matplotlib plotting:
    * 3D scatter of nodes or element centroids.
    * 2D slice scatter (axis-aligned plane) with thickness tolerance.
- PyVista plotting:
    * Conversion to a tetrahedral UnstructuredGrid with cell data (region, resistivity).
    * 3D shaded view and axis-aligned slicing planes.

The functions are designed to accept a single mesh dictionary:
    mesh = {
        "nodes": (N,3) float ndarray,
        "elements": (M,4) int ndarray,  # 0-based node indices for tets
        "regions": (M,) int ndarray,    # region/block ID for each element
        # optional:
        "resistivity": (R,) float ndarray   # region_id (1-based) -> rho[region_id-1]
    }

Example
-------
>>> from femtic_viz import load_femtic_mesh, plot_scatter_3d_matplotlib, slice_scatter_matplotlib
>>> mesh = load_femtic_mesh("mesh.dat", "resistivity_block_iter0.dat")
>>> fig, ax = plot_scatter_3d_matplotlib(mesh, mode="centroids", color_by="resistivity")
>>> fig, ax = slice_scatter_matplotlib(mesh, axis="z", value=0.0, tol=50.0, mode="centroids", color_by="resistivity")

PyVista usage
-------------
>>> import pyvista as pv
>>> from femtic_viz import to_pyvista_unstructured, pyvista_plot_3d, pyvista_slice_axis
>>> grid = to_pyvista_unstructured(mesh)             # build tetra grid with cell data
>>> plotter = pyvista_plot_3d(grid, scalars="resistivity")
>>> sl_plotter = pyvista_slice_axis(grid, axis="z", value=0.0)  # interactive

Notes
-----
- Matplotlib functions require only numpy + matplotlib.
- PyVista functions require pyvista >= 0.43 and vtk; they are guarded so the rest of the module still works if PyVista is not installed.

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-11-09
"""

from typing import Dict, Any, Optional, Tuple, Sequence
import numpy as np

# ------------------------------
# Loading and basic helpers
# ------------------------------

def load_femtic_mesh(mesh_file: str, resistivity_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Load a FEMTIC TETRA mesh and optional resistivity mapping.

    The supported on-disk format is:
        Line 1: "TETRA"
        Line 2: <n_nodes>
        Next n_nodes lines: "<node_id> <x> <y> <z>"
            - node_id is 0- or 1-based (this routine stores as 0-based internally).
        Next line: <n_elements>
        Next n_elements lines: "<elem_id> ... n1 n2 n3 n4"
            - last four integers on the line are the 1-based node IDs for the tet.
            - additional integers between elem_id and the four nodes are ignored (e.g., neighbors).
        Remaining lines (optional):
            - Possibly a single integer header.
            - Then many lines "<elem_id> <region_id>" giving element-wise regions.

    Parameters
    ----------
    mesh_file : str
        Path to mesh.dat.
    resistivity_file : str, optional
        Path to resistivity_block_iterX.dat. If provided, the last numeric
        value on each line is taken as the resistivity (Ohm·m). Region IDs
        are assumed 1-based; index i corresponds to region_id == i.

    Returns
    -------
    mesh : dict
        Dictionary with keys: "nodes", "elements", "regions", and optionally "resistivity".

    Raises
    ------
    AssertionError
        If the header is not TETRA.
    ValueError
        If element lines do not contain sufficient integer tokens.
    """
    with open(mesh_file, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    assert lines[0].upper().startswith("TETRA"), "Unsupported mesh type; only 'TETRA' is supported."

    pos = 1
    n_nodes = int(lines[pos].split()[0]); pos += 1

    # Node block
    # Accept both 0-based and 1-based IDs; store into array by given index.
    nodes = np.zeros((n_nodes, 3), dtype=float)
    zero_based_seen = False
    one_based_seen = False
    for _ in range(n_nodes):
        parts = lines[pos].split()
        idx = int(parts[0])
        if idx == 0:
            zero_based_seen = True
        if idx == 1:
            one_based_seen = True
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        # If IDs are 1-based, place at idx-1; if 0-based, place at idx.
        target = idx if zero_based_seen and not one_based_seen else (idx - 1)
        nodes[target, 0] = x; nodes[target, 1] = y; nodes[target, 2] = z
        pos += 1

    # Element block
    n_elems = int(lines[pos].split()[0]); pos += 1
    elements = np.zeros((n_elems, 4), dtype=int)
    for e in range(n_elems):
        ints = []
        for tok in lines[pos].split():
            try:
                ints.append(int(tok))
            except ValueError:
                # ignore floats/other tokens
                pass
        if len(ints) < 9:
            raise ValueError(f"Unexpected element line: '{{lines[pos]}}'")
        # Pattern: [elem_id, possibly neighbors/ids..., n1, n2, n3, n4]
        conn_1based = ints[-4:]
        elements[e, :] = np.array(conn_1based, dtype=int) - 1  # store 0-based
        pos += 1

    # Regions (element-wise)
    regions = np.zeros(n_elems, dtype=int)
    if pos < len(lines):
        # Optional single integer header
        maybe_hdr = lines[pos].split()
        if len(maybe_hdr) == 1:
            pos += 1
        # Remaining lines: "<elem_id> <region_id>"
        for k in range(pos, len(lines)):
            parts = lines[k].split()
            if len(parts) < 2:
                continue
            try:
                ei = int(parts[0]); ri = int(parts[-1])
            except ValueError:
                continue
            if 0 <= ei < n_elems:
                regions[ei] = ri

    mesh: Dict[str, Any] = {"nodes": nodes, "elements": elements, "regions": regions}

    # Optional resistivity mapping
    if resistivity_file is not None:
        rvals = []
        with open(resistivity_file, "r") as rf:
            for ln in rf:
                s = ln.strip()
                if not s or s.startswith("#") or s.startswith("!"):
                    continue
                toks = s.split()
                # Append the last numeric token; skip non-numerics
                for tok in reversed(toks):
                    try:
                        rvals.append(float(tok))
                        break
                    except ValueError:
                        continue
        mesh["resistivity"] = np.asarray(rvals, dtype=float)

    return mesh


def element_centroids(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """
    Compute tetrahedral element centroids by averaging vertex coordinates.

    Parameters
    ----------
    nodes : (N, 3) ndarray
        Node coordinates.
    elements : (M, 4) ndarray
        0-based node indices per tetra element.

    Returns
    -------
    centroids : (M, 3) ndarray
        Centroid coordinates (x, y, z) for each element.
    """
    return nodes[elements].mean(axis=1)


def map_regions_to_resistivity(regions: np.ndarray, resistivity: Optional[np.ndarray]) -> np.ndarray:
    """
    Map element-wise region IDs to resistivity, if available.

    Parameters
    ----------
    regions : (M,) ndarray of int
        Region IDs per element (often 0..K or 1..K; this function accepts either).
    resistivity : (R,) ndarray of float or None
        Resistivity values where index i corresponds to region_id == i+1 (1-based).

    Returns
    -------
    vals : (M,) ndarray
        Resistivity mapped to each element if available, else the region IDs as float.
    """
    if resistivity is None or resistivity.size == 0:
        return regions.astype(float)
    idx = np.clip(regions.astype(int) - 1, 0, len(resistivity) - 1)
    return resistivity[idx]

# ---------------------------------
# Matplotlib plotting (pure)
# ---------------------------------

def plot_scatter_3d_matplotlib(
    mesh: Dict[str, Any],
    mode: str = "centroids",
    color_by: str = "region",
    point_size: float = 6.0,
    alpha: float = 0.9,
    show: bool = True,
):
    """
    3D scatter using Matplotlib (mplot3d) of nodes or centroids.

    Parameters
    ----------
    mesh : dict
        FEMTIC mesh dict as returned by load_femtic_mesh().
    mode : {'centroids', 'nodes'}, default 'centroids'
        Data to visualize as points.
    color_by : {'region', 'resistivity'}, default 'region'
        Coloring variable: region IDs or region-mapped resistivity.
    point_size : float, default 6.0
        Scatter marker size.
    alpha : float, default 0.9
        Marker transparency.
    show : bool, default True
        Whether to call plt.show().

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes._subplots.Axes3DSubplot
        Figure and 3D axes.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    nodes = mesh["nodes"]; elements = mesh["elements"]; regions = mesh["regions"]
    if mode == "centroids":
        xyz = element_centroids(nodes, elements)
        scalars = map_regions_to_resistivity(regions, mesh.get("resistivity")) if color_by == "resistivity" else regions.astype(float)
        title = f"Elements ({color_by})"
    elif mode == "nodes":
        xyz = nodes
        region_nodes = np.full(nodes.shape[0], np.nan)
        for ei, tet in enumerate(elements):
            for v in tet:
                if np.isnan(region_nodes[v]):
                    region_nodes[v] = regions[ei]
        if color_by == "resistivity" and "resistivity" in mesh:
            scalars = map_regions_to_resistivity(np.nan_to_num(region_nodes, nan=1).astype(int), mesh["resistivity"])
        else:
            scalars = np.nan_to_num(region_nodes, nan=0.0)
        title = f"Nodes (~{color_by})"
    else:
        raise ValueError("mode must be 'centroids' or 'nodes'")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=point_size, alpha=alpha, c=scalars)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, label=color_by)
    if show:
        plt.show()
    return fig, ax


def slice_scatter_matplotlib(
    mesh: Dict[str, Any],
    axis: str = "z",
    value: float = 0.0,
    tol: float = 0.0,
    mode: str = "centroids",
    color_by: str = "region",
    point_size: float = 6.0,
    alpha: float = 0.9,
    show: bool = True,
):
    """
    2D scatter slice through an axis-aligned plane using Matplotlib.

    Parameters
    ----------
    mesh : dict
        FEMTIC mesh dict.
    axis : {'x','y','z'}, default 'z'
        Axis normal to the slicing plane.
    value : float, default 0.0
        Coordinate value for the slicing plane (axis=value).
    tol : float, default 0.0
        Half-thickness to include points with |coord - value| <= tol.
    mode : {'centroids','nodes'}, default 'centroids'
        Which points to slice and plot.
    color_by : {'region','resistivity'}, default 'region'
        Coloring variable.
    point_size : float, default 6.0
        Marker size.
    alpha : float, default 0.9
        Marker transparency.
    show : bool, default True
        Whether to call plt.show().

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        Figure and Axes for the 2D plot.
    """
    import matplotlib.pyplot as plt

    axis_map = {"x": 0, "y": 1, "z": 2}
    if axis not in axis_map:
        raise ValueError("axis must be one of 'x','y','z'")
    axid = axis_map[axis]

    nodes = mesh["nodes"]; elements = mesh["elements"]; regions = mesh["regions"]
    if mode == "centroids":
        xyz = element_centroids(nodes, elements)
        scalars = map_regions_to_resistivity(regions, mesh.get("resistivity")) if color_by == "resistivity" else regions.astype(float)
        label = f"Elements ({color_by})"
    elif mode == "nodes":
        xyz = nodes
        region_nodes = np.full(nodes.shape[0], np.nan)
        for ei, tet in enumerate(elements):
            for v in tet:
                if np.isnan(region_nodes[v]):
                    region_nodes[v] = regions[ei]
        if color_by == "resistivity" and "resistivity" in mesh:
            scalars = map_regions_to_resistivity(np.nan_to_num(region_nodes, nan=1).astype(int), mesh["resistivity"])
        else:
            scalars = np.nan_to_num(region_nodes, nan=0.0)
        label = f"Nodes (~{color_by})"
    else:
        raise ValueError("mode must be 'centroids' or 'nodes'")

    mask = np.abs(xyz[:, axid] - value) <= tol
    pts = xyz[mask]; c = scalars[mask]

    fig, ax = plt.subplots()
    if pts.size == 0:
        ax.set_title(f"No points within |{axis}-{value}| <= {tol}")
        if show:
            plt.show()
        return fig, ax

    other = [i for i in (0, 1, 2) if i != axid]
    sc = ax.scatter(pts[:, other[0]], pts[:, other[1]], s=point_size, alpha=alpha, c=c)
    ax.set_xlabel(["X", "Y", "Z"][other[0]])
    ax.set_ylabel(["X", "Y", "Z"][other[1]])
    ax.set_title(f"{label} on {axis}={value} (±{tol})")
    fig.colorbar(sc, ax=ax, label=color_by)
    if show:
        plt.show()
    return fig, ax

# ------------------------------
# PyVista utilities (optional)
# ------------------------------

def _require_pyvista():
    """
    Import and return pyvista.

    Returns
    -------
    pv : module
        The imported pyvista module.

    Raises
    ------
    ImportError
        If PyVista (and its VTK dependency) is not installed.
    """
    try:
        import pyvista as pv  # type: ignore
    except Exception as exc:
        raise ImportError("PyVista functions require 'pyvista' (and VTK). Please install them first.") from exc
    return pv


def to_pyvista_unstructured(mesh: Dict[str, Any]):
    """
    Convert a FEMTIC tetra mesh into a PyVista UnstructuredGrid with cell data.

    Parameters
    ----------
    mesh : dict
        FEMTIC mesh dict (nodes, elements, regions, optional resistivity).

    Returns
    -------
    grid : pyvista.UnstructuredGrid
        Tetrahedral unstructured grid. Cell data includes:
        - 'region'       : int region ID per cell.
        - 'resistivity'  : float rho per cell (if available).
    """
    pv = _require_pyvista()
    nodes = mesh["nodes"]; elements = mesh["elements"]; regions = mesh["regions"]

    # VTK expects: for each cell -> [npts, i0, i1, i2, i3]
    n_cells = elements.shape[0]
    npts = 4
    cells = np.hstack([np.full((n_cells, 1), npts, dtype=np.int64), elements.astype(np.int64)]).ravel(order="C")
    celltypes = np.full(n_cells, pv.CellType.TETRA, dtype=np.uint8)

    grid = pv.UnstructuredGrid(cells, celltypes, nodes)
    grid.cell_data["region"] = regions.astype(np.int32)
    if "resistivity" in mesh and mesh["resistivity"].size > 0:
        rho = map_regions_to_resistivity(regions, mesh["resistivity"])
        grid.cell_data["resistivity"] = rho.astype(float)
    return grid


def pyvista_plot_3d(grid, scalars: str = "region", notebook: bool = False):
    """
    Create a PyVista Plotter and add the unstructured grid with shading.

    Parameters
    ----------
    grid : pyvista.UnstructuredGrid
        Unstructured mesh grid (tetra).
    scalars : str, default 'region'
        Name of the cell data array to color by ('region' or 'resistivity').
    notebook : bool, default False
        If True, returns a NotebookPlotter; otherwise a standard Plotter.

    Returns
    -------
    plotter : pyvista.BasePlotter
        Plotter with the mesh added. Call plotter.show() to render.
    """
    pv = _require_pyvista()
    plotter = pv.Plotter(notebook=notebook)
    plotter.add_mesh(grid, scalars=scalars, show_edges=False)
    plotter.add_axes()
    plotter.show_grid()
    plotter.enable_eye_dome_lighting()
    plotter.camera_position = "xy"
    return plotter


def pyvista_slice_axis(grid, axis: str = "z", value: float = 0.0, thickness: Optional[float] = None, scalars: str = "resistivity"):
    """
    Slice the unstructured grid with an axis-aligned plane in PyVista.

    Parameters
    ----------
    grid : pyvista.UnstructuredGrid
        Tetrahedral grid.
    axis : {'x','y','z'}, default 'z'
        Axis normal to the slice plane.
    value : float, default 0.0
        Coordinate of the slicing plane (axis=value).
    thickness : float, optional
        If provided, a band-threshold of ±thickness around the plane is applied before slicing.
    scalars : str, default 'resistivity'
        Cell data scalars to color by ('region' also valid).

    Returns
    -------
    plotter : pyvista.BasePlotter
        Plotter with the slice added, ready to show().
    """
    pv = _require_pyvista()
    normal_map = {'x': (1.0, 0.0, 0.0), 'y': (0.0, 1.0, 0.0), 'z': (0.0, 0.0, 1.0)}
    if axis not in normal_map:
        raise ValueError("axis must be one of 'x','y','z'")
    normal = normal_map[axis]

    # Optional band
    src = grid
    if thickness is not None and thickness > 0:
        bounds = list(grid.bounds)
        idx = {'x': 0, 'y': 2, 'z': 4}[axis]
        bounds[idx]   = value - thickness
        bounds[idx+1] = value + thickness
        src = grid.clip_box(bounds, invert=False)

    sl = src.slice(normal=normal, origin=(value if axis=='x' else 0.0,
                                          value if axis=='y' else 0.0,
                                          value if axis=='z' else 0.0))
    plotter = pv.Plotter()
    plotter.add_mesh(sl, scalars=scalars, show_edges=False)
    plotter.add_axes(); plotter.show_grid()
    plotter.camera_position = "xy" if axis == "z" else ("xz" if axis == "y" else "yz")
    return plotter
