"""femtic_viz_new.py

Visualisation utilities for FEMTIC resistivity models.

This module focuses on two complementary workflows:

1. **Direct-from-FEMTIC files (no intermediate NPZ)**:
   - ``mesh.dat`` + ``resistivity_block_iterX.dat`` -> NumPy arrays / PyVista grid
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
                        Fixed ocean_value default: 1e-10 -> 3e-1 Ohm.m in
                        prepare_rho_for_plotting and
                        unstructured_grid_from_femtic.
                        Added ocean_color parameter (default 'lightgrey') to
                        plot_points_matplotlib, plot_map_grid_matplotlib,
                        plot_curtain_matplotlib, map_slice_from_cells,
                        curtain_from_cells, and plot_model_ensemble.
    2026-03-30  Claude  Fixed plot_data_ensemble: replaced non-existent
                        fem.read_observe() with fem.read_observe_dat();
                        added _observe_to_site_list() helper to flatten the
                        nested blocks->sites structure and build Z/Z_err/T/P
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
    2026-05-13  Claude  Added plot_model_3d: PyVista 3-D renderer with
                        axis-aligned x/y/z plane slices, arbitrary oblique
                        plane slices, and iso-surfaces of any cell-data
                        scalar.  Outputs interactive HTML or static screenshot.
    2026-05-26  Claude Sonnet 4.6 (Anthropic)
                        plot_model_3d: added vtu_file parameter -- saves the
                        full unstructured grid (cell-centred) as .vtu/.vtk
                        for ParaView / Zenodo before any rendering.
                        plot_file=*.vtu/.vtk accepted directly (skips
                        plotter).  Added ImportError fallback for HTML export
                        when trame_vtk absent (warns, saves .png instead).
                        Added import math, import os to module imports.
                        Moved plot_model_slices (exact tet-plane intersection,
                        all inner helpers) and plot_borehole_logs from
                        femtic_mod_plot.py into this module; all formerly-
                        implicit config globals are now explicit keyword
                        parameters.  femtic_mod_plot.py reduced to config +
                        thin call wrappers.
    2026-05-27  vrath / Claude Sonnet 4.6 (Anthropic)
                        plot_model_3d: added xlim/ylim/zlim; VTU export and
                        PyVista scene clipped to the same spatial box as the
                        2-D slice panels (pv.clip_box before grid.save and
                        rendering).
                        plot_model_slices: added alpha_file / alpha_mode for
                        polygon-level fading or blanking driven by a second
                        block file (log10 values, <0 = suppress).
    2026-05-31  vrath / Claude Sonnet 4.6 (Anthropic)
                        plot_model_slices, plot_ensemble_slices: added
                        per-panel ``invert_x`` key in slice-spec dicts for
                        ns / ew / plane panels.  Calls ``ax.invert_xaxis()``
                        after rendering, enabling left-right flip for direct
                        comparison with sections from other software.
                        Added highest-point lat/lon diagnostic print.
    2026-05-31  vrath / Claude Sonnet 4.6 (Anthropic)
                        plot_model_slices: actually removed the broken
                        _outline_curtain_top calls from the ns and ew
                        panel branches (the 2026-05-31 changelog entry
                        above claimed removal but the calls were still
                        present).  _outline_curtain_top drew one vertex
                        per polygon connected in x-order, producing dense
                        zigzag lines that appeared as vertical stripes
                        across the full depth of curtain panels.
                        The function definition is retained for reference
                        but is no longer called.
    2026-05-31  vrath / Claude Sonnet 4.6 (Anthropic)
                        read_femtic_mesh: swapped node coordinate columns on
                        read: FEMTIC stores nodes as (id, x=northing,
                        y=easting, z-down); femtic_viz.py uses geographic
                        convention ([:,0]=easting, [:,1]=northing).
                        Confirmed from femtic_make_cutaway_imp.py which
                        uses node_xyz[:,0] as horizontal axis of EW
                        sections (= northing in FEMTIC) and [:,1] for
                        map horizontal axis (= easting).
                        read_femtic_mesh: fixed node and element readers to
                        use the file index (parts[0]) rather than the
                        sequential loop counter, matching femtic.py's
                        read_femtic_mesh.  Same root cause as the
                        resistivity-block bug: if nodes or elements are
                        not stored in strict sequential order the previous
                        code silently assigned coordinates / connectivity
                        to the wrong indices.  _map_node helper removed
                        (no longer needed).
    2026-06-03  Claude Sonnet 4.6 (Anthropic)
                        plot_borehole_logs: (1) z_top="surface" auto-detects
                        the mesh surface elevation at the borehole (x, y)
                        location using a KD-tree nearest-node search on the
                        minimum node-z per x/y column.  (2) Position and
                        elevation appended to each legend / title label
                        (x_m, y_m, z_top_m).  (3) X-axis switched to
                        log-scale (ax.set_xscale("log")); rho is now plotted
                        in Ohm*m rather than log10(Ohm*m), so clim is in
                        Ohm*m too (e.g. [1.0, 1e4]).  BOREHOLE_XLIM in
                        femtic_mod_plot.py updated accordingly.
                        (4) legend shows lat/lon when spec dict contains
                        "lat"/"lon" keys (replaces raw x/y model-local metres
                        in the annotation line).  (5) Per-spec line style:
                        any Matplotlib Line2D kwargs ("color", "ls", "lw",
                        "marker", "alpha", …) placed in the spec dict override
                        the global borehole_style for that trace only.
    2026-06-03  Claude Sonnet 4.6 (Anthropic)
                        plot_borehole_logs: added npz_file parameter for NPZ
                        export of all sampled depth/rho arrays.  Arrays stored
                        as depth_<name> and rho_<name>; JSON header string
                        (model file, mesh file, timestamp, per-borehole meta)
                        stored as scalar string array "header".  npz_file
                        defaults to plot_file with .npz extension when None;
                        falls back to "borehole_logs.npz" when plot_file is
                        also None.  Config vars BOREHOLE_NPZ / BOREHOLE_NPZ_FILE
                        added to femtic_mod_plot_slice.py.
                        Added _sample_borehole_logs() private helper (shared
                        by plot_borehole_logs and plot_model_slices).
                        plot_model_slices: added borehole_sites,
                        borehole_style, borehole_clim, borehole_shared,
                        borehole_resolve_xy parameters; when borehole_sites is
                        non-empty, borehole panels are appended as extra
                        columns to the right of the slice grid in the same
                        figure (one column per borehole when
                        borehole_shared=False, one shared column when True).
                        Fixed _sample_borehole_logs: CRS-tagged x/y positions
                        ("utm", "latlon") now correctly converted jointly
                        via fem.latlon_to_model / fem.utm_to_model rather than
                        the single-axis resolve_pos_x/y (which held the other
                        coordinate at the mesh origin).  lat/lon for the legend
                        now auto-inferred from the CRS tag: latlon → lon from
                        x, lat from y; utm → back-converted via
                        utl.utm_to_latlon_zn.  Explicit "lat"/"lon" spec keys
                        still override the auto-inferred values.
                        plot_borehole_logs: added utm_zone, utm_northern,
                        utm_origin_e, utm_origin_n parameters (passed through
                        to _sample_borehole_logs for CRS conversion without
                        a resolver function).
                        read_resistivity_block: fixed element->region and
                        region->rho assignment to use the index from the
                        file (parts[0]) rather than the sequential loop
                        counter.  When elements or regions are not stored
                        in strict order the previous code silently assigned
                        resistivities to the wrong regions, producing
                        scrambled / uniformly-conductive model plots.
    2026-06-04  vrath / Claude Sonnet 4.6 (Anthropic)
                        Script-level split only: ``femtic_mod_plot_slice.py``
                        drives ``plot_model_slices``; new
                        ``femtic_mod_plot_bh.py`` drives
                        ``plot_borehole_logs``.  Both functions and
                        ``_sample_borehole_logs`` remain in this module.
    2026-06-04  vrath / Claude Sonnet 4.6 (Anthropic)
                        plot_borehole_logs: added ``markers`` parameter for
                        free annotations (arrows + text) on borehole depth
                        axes.  Each marker dict accepts ``depth`` [m] (arrow
                        tip, required), ``rho`` [Ohm*m] (tip x-position),
                        ``rho_text`` [Ohm*m] (text x-position, display units),
                        ``depth_text`` [m] (text y-position, display units),
                        ``text``, ``borehole`` (name or list for targeting),
                        ``arrowprops``, and any ``ax.annotate`` kwargs.
                        Both text positions in natural axis units (Ohm*m / m)
                        so values can be read directly off the axes.
                        Added ``legend_fontsize`` parameter (default 9) that
                        controls the shared-mode legend and per-panel title
                        font size; tick labels scale with it.
    2026-06-07  Claude Sonnet 4.6 (Anthropic)
                        plot_model_ensemble / _draw_slice: replaced ``'type'``
                        key dispatch with ``'kind'`` (accepting ``'map'``,
                        ``'ns'``, ``'ew'``, ``'curtain'``); ``'type'`` still
                        accepted as legacy fallback.  ``'ns'``/``'ew'`` now
                        build the curtain polyline from ``x0``/``y0`` and mesh
                        bounding-box extents, matching the ``plot_model_slices``
                        slice-spec format.  Routing keys ``kind``, ``type``,
                        ``x0``, ``y0``, ``invert_x``, ``title`` are stripped
                        before forwarding to slicer functions.  Docstring
                        updated accordingly.
    2026-06-26  vrath / Claude Sonnet 4.6 (Anthropic)
                        plot_model_slices: added ``tick_fontsize`` (default 7)
                        and ``label_fontsize`` (default 8) parameters;
                        replaced all hardcoded fontsize literals in tick
                        labels, axis labels, panel titles, colourbar label,
                        inline borehole columns, and suptitle.
                        plot_borehole_logs: added ``tick_fontsize`` and
                        ``label_fontsize`` parameters (both default ``None``
                        → derived from ``legend_fontsize``); replaced
                        hardcoded tick / xlabel / ylabel literals."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
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

        nodes = np.empty((nn, 3), dtype=float)
        for _ in range(nn):
            parts = f.readline().split()
            if len(parts) < 4:
                raise ValueError(f"Malformed node line: {parts!r}")
            idx = int(parts[0])
            if not (0 <= idx < nn):
                raise ValueError(f"Node index {idx} out of range 0..{nn-1}.")
            # FEMTIC mesh stores nodes as (index, x=northing, y=easting, z=down).
            # Remap to geographic convention used throughout femtic_viz.py:
            #   nodes[:,0] = easting, nodes[:,1] = northing, nodes[:,2] = z-down.
            nodes[idx, 0] = float(parts[2])   # easting  ← FEMTIC column y
            nodes[idx, 1] = float(parts[1])   # northing ← FEMTIC column x
            nodes[idx, 2] = float(parts[3])   # z positive-down (unchanged)

        id2row = {i: i for i in range(nn)}   # identity: file index == row index

        ne_line = f.readline().split()
        if not ne_line:
            raise ValueError("Missing element count after nodes.")
        nelem = int(ne_line[0])

        conn = np.empty((nelem, 4), dtype=int)

        for _ in range(nelem):
            parts = f.readline().split()
            if len(parts) < 5:
                raise ValueError(f"Malformed element line: {parts!r}")
            ie   = int(parts[0])
            if not (0 <= ie < nelem):
                raise ValueError(f"Element index {ie} out of range 0..{nelem-1}.")
            nids = [int(parts[-4]), int(parts[-3]), int(parts[-2]), int(parts[-1])]
            conn[ie, :] = nids   # node IDs == row indices after index-based node read

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
        Parsed resistivity block (element->region and region properties).

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
        for _ in range(nelem):
            parts = f.readline().split()
            if len(parts) < 2:
                raise ValueError(f"Malformed element->region line: {parts!r}")
            ie   = int(parts[0])
            ireg = int(parts[1])
            if not (0 <= ie < nelem):
                raise ValueError(f"Element index {ie} out of range 0..{nelem-1}.")
            region_of_elem[ie] = ireg

        region_rho       = np.empty(nreg, dtype=float)
        region_rho_lower = np.empty(nreg, dtype=float)
        region_rho_upper = np.empty(nreg, dtype=float)
        region_n         = np.empty(nreg, dtype=float)
        region_flag      = np.empty(nreg, dtype=int)

        for _ in range(nreg):
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF while reading region lines.")
            parts = line.split()
            if len(parts) < 6:
                raise ValueError(f"Region line has too few columns: {line!r}")
            ireg = int(parts[0])
            if not (0 <= ireg < nreg):
                raise ValueError(f"Region index {ireg} out of range 0..{nreg-1}.")
            region_rho[ireg]       = float(parts[1])
            region_rho_lower[ireg] = float(parts[2])
            region_rho_upper[ireg] = float(parts[3])
            region_n[ireg]         = float(parts[4])
            region_flag[ireg]      = int(parts[5])

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
        If not None, set all ocean-region elements to this value (Ohm*m).
        Default is 0.3 Ohm*m (typical seawater resistivity).
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
        Optional element->region mapping, shape ``(nelem,)`` (0- or 1-based).
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
        Target slice depth in the **FEMTIC model frame** (z positive
        downward).  A surface slice at 500 m elevation above datum would be
        requested as ``z0 = -500``.  To convert from geodetic elevation:
        ``z0 = -elev_m``.
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
        ``z`` is the centroid z-coordinate in the **FEMTIC model frame**
        (z positive downward).  To display depth on a conventional plot
        (positive depth axis pointing down), use ``z`` directly as the
        vertical axis with an inverted y-axis.  To display elevation (positive
        up), plot ``-z`` instead.

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
        Resistivity value (Ohm*m) used to identify ocean cells.
        Should match the value passed to :func:`prepare_rho_for_plotting`
        (default 0.3 Ohm*m).
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
        Resistivity (Ohm*m) marking ocean cells (default 0.3).

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
        Resistivity (Ohm*m) identifying ocean cells (default 0.3).

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
        title = f"Map slice at z={z0} +/- {dz} ({mode})"
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
        title = f"Map slice at z={z0} +/- {dz} (IDW grid)"
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
        Resistivity (Ohm*m) identifying ocean cells (default 0.3).

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
        title = f"Curtain section (width <= {width}) ({mode})"
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
        Resistivity (Ohm*m) marking ocean cells (default 0.3).

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
        raise ImportError("pyvista is required for NPZ->grid conversion.")

    try:
        import femtic  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("Could not import femtic.py for NPZ helpers.") from e

    return femtic.npz_to_unstructured_grid(str(npz_path))


# =============================================================================
# RTO ensemble diagnostic plots
# =============================================================================

_MU0 = 4.0e-7 * np.pi          # H/m
_Z_SI_TO_MT = 1.0 / (_MU0 * 1.0e3)   # SI Ohm -> mV/km/nT

# ---------------------------------------------------------------------------
# Component classification for marker differentiation
# ---------------------------------------------------------------------------

#: Default marker symbols per component class.
#: Keys: ``'ii'`` (diagonal: xx, yy), ``'ij'`` (off-diagonal: xy, yx),
#: ``'inv'`` (invariants: det, bahr, ssq, ...).
#: Pass ``comp_markers=None`` to :func:`plot_data_ensemble` to disable markers.
DEFAULT_COMP_MARKERS: dict = {
    "ii":  "o",   # diagonal -- circles
    "ij":  "s",   # off-diagonal -- squares
    "inv": "^",   # invariants -- triangles-up
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
    unit convention (Z in SI Ohm; ``rho_a = |Z|^2 / (mu0omega)``).

    The returned dicts contain ``Z`` scaled to MT field units (mV/km/nT) so
    that :func:`data_viz.datadict_to_plot_df` -- which assumes field-unit Z
    and applies ``rho = |Z|^2 x mu0 x 10^6 / omega`` -- produces the right answer.
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
    tick_fontsize: int = 8,
    label_fontsize: int = 9,
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
        y-axis limits for ``'rho'`` panels (apparent resistivity, Ohm*m or
        log10(Ohm*m) depending on the plotter).  ``None`` = auto.
    phslims : (float, float) or None, optional
        y-axis limits for ``'phase'`` panels (degrees).  ``None`` = auto.
    vtflims : (float, float) or None, optional
        y-axis limits for ``'tipper'`` panels.  ``None`` = auto.
    ptlims : (float, float) or None, optional
        y-axis limits for ``'pt'`` panels.  ``None`` = auto.
    figsize : (float, float) or None, optional
        Figure size.  Auto-sized if ``None``:
        width ~ 5 x n_panels, height ~ 3 x n_samples.
    fig : matplotlib.figure.Figure or None, optional
        Pre-existing figure.  If ``None`` a new one is created.
    axs : array-like of Axes or None, optional
        Pre-allocated axes of shape ``(n_samples, n_panels)`` (or ravelled
        equivalent).  If ``None``, subplots are created automatically.
    tick_fontsize : int, optional
        Font size for axis tick labels.  Default ``8``.
    label_fontsize : int, optional
        Font size for panel titles and legends.  Default ``9``.
    out : bool, optional
        Print progress messages.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : numpy.ndarray of matplotlib.axes.Axes
        Shape ``(n_samples, n_panels)``.  When *what* is a single string the
        shape is ``(n_samples, 1)`` -- index with ``axs[:, 0]`` if you need the
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
    ``_observe_to_site_list``).  FEMTIC stores Z in SI Ohm; the helper converts
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
    # Normalise 'what' -> list of panel names; validate each entry.
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
    # Normalise 'comps' -> list of length n_panels.
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
    # Per-panel component lists (empty list -> tipper/pt, no per-comp loop).
    # ------------------------------------------------------------------
    _panel_comp_lists = []
    for w, c in zip(what_list, comps_list):
        if str(w).lower() in _needs_comps and c:
            _panel_comp_lists.append([s.strip() for s in c.split(",")])
        else:
            _panel_comp_lists.append([])

    # ------------------------------------------------------------------
    # Per-panel y-axis limit lookup.
    # Maps panel type -> the caller-supplied limit (or None = auto).
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
    _es_warned: dict = {}   # plotter_name -> bool

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
                ax.legend(seen.values(), seen.keys(), fontsize=tick_fontsize)

            # Axis limits -- applied post-hoc so plotters can't override them.
            if perlims is not None:
                ax.set_xlim(perlims)
            _yl = _ylim_map.get(str(w).lower())
            if _yl is not None:
                ax.set_ylim(_yl)

            # Column header on first row only.
            if row == 0:
                ax.set_title(f"{w.capitalize()} -- sample {idx}", fontsize=label_fontsize)
            else:
                ax.set_title(f"Sample {idx}", fontsize=label_fontsize)

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
    tick_fontsize: int = 8,
    label_fontsize: int = 8,
    out: bool = True,
) -> Tuple[Any, np.ndarray]:
    """Joint plot of original and perturbed resistivity models.

    Produces a grid of Matplotlib axes: **rows** = 2 x number of selected
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
        Between 1 and 5 slice descriptors.  Each dict must have a ``'kind'``
        key (preferred) or legacy ``'type'`` key identifying the panel type:

        - ``'map'``     – horizontal depth slice; position key: ``z0`` (m, positive-down).
        - ``'ns'``      – N-S vertical section;   position key: ``x0`` (easting, m).
        - ``'ew'``      – E-W vertical section;   position key: ``y0`` (northing, m).
        - ``'curtain'`` – arbitrary profile;       position key: ``'polyline'`` array.

        Examples::

            dict(kind='map', z0=5000.)
            dict(kind='ns',  x0=0.)
            dict(kind='ew',  y0=0.)
            {'type': 'curtain',
             'polyline': np.array([[x0, y0], [x1, y1]]),
             'width': 500}

    mode : {'scatter', 'tri', 'grid'}, optional
        Slice rendering mode forwarded to the slicer functions.  Default
        is ``'tri'``.
    log10 : bool, optional
        Plot log10(rho) if ``True`` (default).
    cmap : str or None, optional
        Matplotlib colormap (default ``'jet_r'``).
    clim : (float, float) or None, optional
        Colour limits ``(vmin, vmax)`` in log10(Ohm*m).
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
        Resistivity (Ohm*m) identifying ocean cells (default 0.3).
    figsize : (float, float) or None, optional
        Figure size in inches.  If ``None``, a sensible default is chosen.
    fig : matplotlib.figure.Figure or None, optional
        Pre-existing figure.
    axs : array-like of Axes or None, optional
        Pre-allocated axes of shape
        ``(2 * len(sample_indices), len(slices))``.
        If ``None``, subplots are created automatically.
    tick_fontsize : int, optional
        Font size for axis tick labels.  Default ``8``.
    label_fontsize : int, optional
        Font size for panel titles and row labels.  Default ``8``.
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
        raise ValueError("plot_model_ensemble: slices must have 1-5 entries.")

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

    # Read mesh once -- shared by all runs
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
        """Dispatch one slice descriptor to the appropriate slicer and draw on ax.

        Accepts both the new ``'kind'`` key (``'map'``, ``'ns'``, ``'ew'``,
        ``'curtain'``) and the legacy ``'type'`` key for backward compatibility.

        Map slices use ``z0`` (depth, metres, positive-down).
        NS slices use ``x0`` (easting, metres).
        EW slices use ``y0`` (northing, metres).
        Curtain slices use a ``'polyline'`` key (shape (m, 2) array).
        """
        slc = dict(slc_spec)  # shallow copy — we pop routing keys before forwarding

        # Accept both 'kind' (new) and 'type' (legacy), with 'kind' taking priority.
        slc_kind = slc.pop("kind", None)
        if slc_kind is None:
            slc_kind = slc.pop("type", "map")
        else:
            slc.pop("type", None)   # discard legacy key if both were present
        slc_kind = slc_kind.lower()

        # Normalise reversed-direction aliases before dispatch.
        if slc_kind in ("sn", "we"):
            slc_kind = "ns" if slc_kind == "sn" else "ew"
            slc["invert_x"] = not slc.get("invert_x", False)

        # Per-slice axis-limit overrides (consumed here, not forwarded to slicer).
        slc_xlim = slc.pop("xlim", xlim)
        slc_ylim = slc.pop("ylim", ylim)
        slc_zlim = slc.pop("zlim", zlim)

        # Strip routing/geometry keys that are not accepted by the slicer functions.
        slc.pop("invert_x", None)
        slc.pop("title", None)

        # Inject shared plot defaults (caller may override per-slice via slc_spec).
        slc.setdefault("mode", mode)
        slc.setdefault("log10", log10)
        slc.setdefault("cmap", cmap)
        slc.setdefault("ocean_color", ocean_color)
        slc.setdefault("ocean_value", ocean_value)
        slc.setdefault("mesh_lines", mesh_lines)
        slc.setdefault("mesh_lw", mesh_lw)
        slc.setdefault("mesh_color", mesh_color)

        if "map" in slc_kind:
            z0 = slc.pop("z0", 0.0)
            map_slice_from_cells(mesh, rho, ax=ax, z0=z0, **slc)
            if slc_xlim is not None:
                ax.set_xlim(slc_xlim)
            if slc_ylim is not None:
                ax.set_ylim(slc_ylim)

        elif slc_kind in ("ns", "ew"):
            # Build a 2-point polyline spanning the mesh bounding box.
            # NS section: constant easting x0, profile runs N–S (y varies).
            # EW section: constant northing y0, profile runs E–W (x varies).
            node_xyz = mesh.nodes   # shape (nnodes, 3): [easting, northing, depth]
            x_min, x_max = float(node_xyz[:, 0].min()), float(node_xyz[:, 0].max())
            y_min, y_max = float(node_xyz[:, 1].min()), float(node_xyz[:, 1].max())
            # Remove position keys before forwarding to curtain_from_cells.
            slc.pop("x0", None)
            slc.pop("y0", None)
            if slc_kind == "ns":
                x0 = float(slc_spec.get("x0", 0.0))
                poly = np.array([[x0, y_min], [x0, y_max]])
            else:  # "ew"
                y0 = float(slc_spec.get("y0", 0.0))
                poly = np.array([[x_min, y0], [x_max, y0]])
            curtain_from_cells(mesh, rho, poly, ax=ax, **slc)
            # Curtain: horizontal axis = along-profile distance, vertical = depth.
            if slc_ylim is not None:
                ax.set_xlim(slc_ylim)
            if slc_zlim is not None:
                ax.set_ylim(slc_zlim)

        else:
            # Legacy 'curtain' or 'plane' kinds: expect a 'polyline' key in slc_spec.
            poly = slc.pop("polyline", None)
            slc.pop("x0", None); slc.pop("y0", None); slc.pop("z0", None)
            if poly is None:
                ax.set_visible(False)
                return
            curtain_from_cells(mesh, rho, poly, ax=ax, **slc)
            if slc_ylim is not None:
                ax.set_xlim(slc_ylim)
            if slc_zlim is not None:
                ax.set_ylim(slc_zlim)

        # Apply shared colour limits after plotting.
        for img in ax.get_images():
            img.set_clim(vmin, vmax)
        for coll in ax.collections:
            coll.set_clim(vmin, vmax)

        if title:
            ax.set_title(title, fontsize=label_fontsize)

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
                ax_orig.set_ylabel(f"orig\n(sample {idx})", fontsize=label_fontsize)

            ax_pert = axs_arr[row_pert, col]
            _draw_slice(ax_pert, rho_pert, slc_spec)
            if col == 0:
                ax_pert.set_ylabel(f"pert {idx}", fontsize=label_fontsize)

        if out:
            print(f"  sample {idx} done")

    fig.tight_layout()
    return fig, axs_arr


# =============================================================================
# 3-D interactive / static model plot  (PyVista)
# =============================================================================

def plot_model_3d(
    mesh_file: Union[str, Path],
    block_file: Union[str, Path],
    *,
    # --- scalar field ---
    scalar: str = "log10_resistivity",
    clim: Optional[Tuple[float, float]] = None,
    cmap: str = "turbo_r",
    # --- orthogonal axis-aligned slices ---
    slice_x: Optional[Sequence[float]] = None,
    slice_y: Optional[Sequence[float]] = None,
    slice_z: Optional[Sequence[float]] = None,
    # --- arbitrary plane slices ---
    slice_planes: Optional[Sequence[dict]] = None,
    # --- iso-surfaces ---
    isovalues: Optional[Sequence[float]] = None,
    iso_opacity: float = 0.4,
    iso_cmap: str = "turbo_r",
    # --- display options ---
    show_edges: bool = False,
    edge_color: str = "k",
    edge_lw: float = 0.3,
    background: str = "white",
    window_size: Tuple[int, int] = (1600, 900),
    # --- ocean / air ---
    ocean_value: Optional[float] = 3.0e-1,
    air_region_index: int = 0,
    ocean_region_index: int = 1,
    # --- spatial clipping ---
    xlim: Optional[Sequence[float]] = None,
    ylim: Optional[Sequence[float]] = None,
    zlim: Optional[Sequence[float]] = None,
    # --- output ---
    plot_file: Optional[Union[str, Path]] = None,
    vtu_file: Optional[Union[str, Path]] = None,
    screenshot_scale: int = 2,
    out: bool = True,
) -> Optional["pv.Plotter"]:
    """Render a FEMTIC resistivity model in 3-D using PyVista.

    Produces axis-aligned slices (any combination of x, y, z planes),
    arbitrary oblique planes, and iso-surfaces of ``log10_resistivity``
    (or any other cell-data scalar).

    Parameters
    ----------
    mesh_file, block_file
        Paths to ``mesh.dat`` and ``resistivity_block_iterX.dat``.
    scalar
        Cell-data scalar to display.  ``"log10_resistivity"`` (default) or
        ``"resistivity"``.  Must be present on the PyVista grid -- both are
        attached by :func:`unstructured_grid_from_femtic`.
    clim
        Colour limits ``(vmin, vmax)`` for the scalar.
        ``None`` -> PyVista auto.
    cmap
        Matplotlib / PyVista colormap name for slices (default ``"turbo_r"``).
    slice_x
        List of x-positions (model-local metres) at which to cut YZ planes.
        ``None`` or empty -> no x-slices.
    slice_y
        List of y-positions (model-local metres) at which to cut XZ planes.
        ``None`` or empty -> no y-slices.
    slice_z
        List of z-positions (model-local metres, z positive-down) at which
        to cut XY planes.  ``None`` or empty -> no z-slices.
    slice_planes
        List of dicts for arbitrary oblique planes.  Each dict may contain:

        - ``"origin"`` : ``[x, y, z]`` -- point on the plane (model-local m).
          Defaults to mesh centre.
        - ``"normal"`` : ``[nx, ny, nz]`` -- plane normal vector.
          Defaults to ``[0, 1, 0]`` (YZ plane).

        Example::

            slice_planes = [
                dict(origin=[0., 0., 10000.], normal=[1., 1., 0.]),
            ]

    isovalues
        Scalar values at which to draw iso-surfaces.  For
        ``scalar="log10_resistivity"`` these are in log10(Ohm*m) -- e.g.
        ``[1.0, 2.0, 3.0]`` for 10 / 100 / 1000 Ohm*m boundaries.
        ``None`` or empty -> no iso-surfaces.
    iso_opacity
        Opacity for iso-surfaces (0 = transparent, 1 = opaque).
    iso_cmap
        Colormap for iso-surface colouring (defaults to same as ``cmap``).
    show_edges
        Overlay mesh edges on slices (slow for large meshes).
    edge_color, edge_lw
        Colour and line-width for mesh edges when ``show_edges=True``.
    background
        Background colour string (default ``"white"``).
    window_size
        Plotter window resolution ``(width, height)`` in pixels.
    ocean_value
        Resistivity (Ohm*m) used to sentinel-mark ocean cells in
        :func:`prepare_rho_for_plotting`.
    air_region_index, ocean_region_index
        Region indices for air and ocean masking.
    xlim, ylim, zlim
        Optional spatial clipping bounds in model-local metres
        ``[min, max]``.  When set, both the VTU export and the PyVista
        scene are clipped to the box defined by these limits (using
        ``pv.UnstructuredGrid.clip_box``).  Air / padding cells outside
        the box are removed, keeping the exported volume consistent with
        the 2-D slice panels (which use the same ``PLOT_XLIM`` /
        ``PLOT_YLIM`` / ``PLOT_ZLIM`` limits).  ``None`` = no clipping
        along that axis (full mesh extent used).
    plot_file
        Output path for the rendered view.  Recognised extensions:

        - ``.vtu`` / ``.vtk``  -> VTK unstructured-grid file of the full 3-D
          volume (no rendering; same as setting ``vtu_file``).
        - ``.html``  -> interactive WebGL scene (requires ``pyvista[jupyter]``
          / ``trame_vtk``; falls back to ``.png`` if unavailable).
        - ``.png`` / ``.jpg`` / ``.svg`` -> static screenshot.
        - ``None``  -> open an interactive PyVista window.

    vtu_file
        If given, the full unstructured grid (all cell-data arrays, including
        ``resistivity`` and ``log10_resistivity``) is saved as a VTK XML
        unstructured-grid (``.vtu``) or legacy VTK (``.vtk``) file **in
        addition to** any ``plot_file`` output.  Suitable for ParaView,
        Zenodo data deposits, or publication supplementary material.
        ``None`` -> no grid export.

    screenshot_scale
        Anti-aliasing scale factor for screenshot modes (default 2 = 2x resolution).
    out
        Print progress messages when ``True``.

    Returns
    -------
    pl : pv.Plotter or None
        The PyVista Plotter object (already closed / exported when ``plot_file``
        is set).  Returns ``None`` when PyVista is unavailable.

    Raises
    ------
    ImportError
        If PyVista is not importable and ``plot_file`` is not ``None`` (i.e. when
        an output file was requested; interactive use silently returns ``None``).
    """
    if pv is None:
        if plot_file is not None:
            raise ImportError(
                "pyvista is required for plot_model_3d.  "
                "Install with:  conda install -c conda-forge pyvista"
            )
        if out:
            print("  plot_model_3d: pyvista not available -- skipped.")
        return None

    plot_file = Path(plot_file) if plot_file is not None else None

    # -- Build grid ------------------------------------------------------------
    if out:
        print(f"  3-D: building PyVista grid from {Path(block_file).name} ...")
    grid = unstructured_grid_from_femtic(
        mesh_file,
        block_file,
        air_is_nan=True,
        ocean_value=ocean_value,
        air_region_index=air_region_index,
        ocean_region_index=ocean_region_index,
    )
    if scalar not in grid.cell_data:
        raise KeyError(
            f"Scalar {scalar!r} not found in grid.  "
            f"Available: {list(grid.cell_data.keys())}"
        )

    # -- Spatial clipping (cell-data grid, before save and rendering) ----------
    if any(lim is not None for lim in (xlim, ylim, zlim)):
        _gb = list(grid.bounds)   # [xmin,xmax, ymin,ymax, zmin,zmax]
        if xlim is not None:
            _gb[0], _gb[1] = float(xlim[0]), float(xlim[1])
        if ylim is not None:
            _gb[2], _gb[3] = float(ylim[0]), float(ylim[1])
        if zlim is not None:
            _gb[4], _gb[5] = float(zlim[0]), float(zlim[1])
        if out:
            print(f"  3-D: clipping to box x=[{_gb[0]:.0f},{_gb[1]:.0f}] "
                  f"y=[{_gb[2]:.0f},{_gb[3]:.0f}] z=[{_gb[4]:.0f},{_gb[5]:.0f}] m")
        grid = grid.clip_box(_gb, invert=False)
        if out:
            print(f"  3-D: {grid.n_cells} cells after clipping")

    # -- Optional VTK export (full 3-D volume, cell-centred) -------------------
    _vtu_suffixes = {".vtu", ".vtk"}
    vtu_file = Path(vtu_file) if vtu_file is not None else None
    if vtu_file is not None:
        grid.save(str(vtu_file))
        if out:
            print(f"  3-D: VTK grid saved -> {vtu_file}")

    if plot_file is not None and plot_file.suffix.lower() in _vtu_suffixes:
        if vtu_file is None or vtu_file.resolve() != plot_file.resolve():
            grid.save(str(plot_file))
            if out:
                print(f"  3-D: VTK grid saved -> {plot_file}")
        return None
    grid = grid.cell_data_to_point_data()

    # -- Plotter ---------------------------------------------------------------
    off_screen = plot_file is not None
    pl = pv.Plotter(
        off_screen=off_screen,
        window_size=list(window_size),
    )
    pl.background_color = background

    scalar_bar_args = dict(
        title=scalar.replace("_", " "),
        n_labels=5,
        fmt="%.1f",
        vertical=True,
    )

    _slice_kwargs = dict(
        scalars=scalar,
        cmap=cmap,
        clim=list(clim) if clim is not None else None,
        show_edges=show_edges,
        edge_color=edge_color,
        line_width=float(edge_lw),
        scalar_bar_args=scalar_bar_args,
    )

    # -- Axis-aligned slices ---------------------------------------------------
    n_slices = 0

    def _add_slice(origin, normal):
        """Add one plane slice to the plotter."""
        nonlocal n_slices
        try:
            slc = grid.slice(normal=normal, origin=origin)
            if slc.n_points > 0:
                pl.add_mesh(slc, **_slice_kwargs)
                n_slices += 1
        except Exception as exc:
            if out:
                print(f"    3-D: slice at origin={origin} normal={normal} failed: {exc}")

    bounds = grid.bounds   # (xmin, xmax, ymin, ymax, zmin, zmax)
    cx = 0.5 * (bounds[0] + bounds[1])
    cy = 0.5 * (bounds[2] + bounds[3])
    cz = 0.5 * (bounds[4] + bounds[5])

    for xpos in (slice_x or []):
        if out:
            print(f"    3-D: YZ slice at x = {xpos:.0f} m ...")
        _add_slice([float(xpos), cy, cz], [1, 0, 0])

    for ypos in (slice_y or []):
        if out:
            print(f"    3-D: XZ slice at y = {ypos:.0f} m ...")
        _add_slice([cx, float(ypos), cz], [0, 1, 0])

    for zpos in (slice_z or []):
        if out:
            print(f"    3-D: XY slice at z = {zpos:.0f} m ...")
        _add_slice([cx, cy, float(zpos)], [0, 0, 1])

    # -- Arbitrary planes ------------------------------------------------------
    for spec in (slice_planes or []):
        origin = spec.get("origin", [cx, cy, cz])
        normal = spec.get("normal", [0, 1, 0])
        if out:
            print(f"    3-D: oblique slice origin={origin} normal={normal} ...")
        _add_slice(origin, normal)

    # -- Iso-surfaces ----------------------------------------------------------
    for ival in (isovalues or []):
        if out:
            print(f"    3-D: iso-surface {scalar} = {ival:.2f} ...")
        try:
            iso = grid.contour([float(ival)], scalars=scalar)
            if iso.n_points > 0:
                pl.add_mesh(
                    iso,
                    scalars=scalar,
                    cmap=iso_cmap,
                    clim=list(clim) if clim is not None else None,
                    opacity=float(iso_opacity),
                    show_scalar_bar=False,
                )
        except Exception as exc:
            if out:
                print(f"    3-D: iso-surface {ival} failed: {exc}")

    if n_slices == 0 and not isovalues:
        if out:
            print("  3-D: no slices or iso-surfaces defined -- adding orthogonal default.")
        slc = grid.slice_orthogonal()
        for s in slc:
            pl.add_mesh(s, **_slice_kwargs)

    pl.add_axes()

    # -- Output ----------------------------------------------------------------
    if plot_file is None:
        pl.show()
        return pl

    suffix = plot_file.suffix.lower()
    if suffix == ".html":
        try:
            pl.export_html(str(plot_file))
            if out:
                print(f"  3-D: interactive HTML saved -> {plot_file}")
        except ImportError:
            import warnings
            fallback = plot_file.with_suffix(".png")
            warnings.warn(
                "trame_vtk not available (pyvista[jupyter] not installed); "
                f"falling back to screenshot -> {fallback}",
                ImportWarning,
                stacklevel=2,
            )
            pl.screenshot(str(fallback), scale=screenshot_scale)
            if out:
                print(f"  3-D: screenshot saved -> {fallback}")
    else:
        pl.screenshot(str(plot_file), scale=screenshot_scale)
        if out:
            print(f"  3-D: screenshot saved -> {plot_file}")

    pl.close()
    return pl


# =============================================================================
# 2-D slice figure  (exact tetrahedron-plane intersection)
# =============================================================================

# =============================================================================
# Private helper: borehole sampling (shared by plot_model_slices and
# plot_borehole_logs so the mesh is loaded only once when boreholes are
# embedded in the slice figure).
# =============================================================================

def _sample_borehole_logs(
    nodes: np.ndarray,
    conn: np.ndarray,
    rho_plot: np.ndarray,
    borehole_sites: list,
    *,
    resolve_xy_fn=None,
    utm_zone: int = 1,
    utm_northern: bool = True,
    utm_origin_e: float = 0.0,
    utm_origin_n: float = 0.0,
    out: bool = True,
) -> list:
    """Sample resistivity along vertical boreholes and return log dicts.

    Parameters
    ----------
    nodes, conn, rho_plot
        Mesh arrays as returned by ``read_femtic_mesh`` /
        ``prepare_rho_for_plotting``.
    borehole_sites
        List of spec dicts (see ``plot_borehole_logs`` for full key docs).
        ``"x"`` and ``"y"`` accept:

        - plain ``float``          → model-local metres
        - ``(value, "model")``     → model-local metres
        - ``(E_m, "utm")``         → UTM easting / northing (both must be utm)
        - ``(lon, "latlon")``      → geographic lon for x, lat for y
          (both must carry the same "latlon" tag; joint conversion via
          ``femtic.latlon_to_model``)

        The CRS tags on ``"x"`` and ``"y"`` **must match**; mixing tags
        across the two keys raises ``ValueError``.

        When ``"lat"`` / ``"lon"`` keys are explicitly in the spec they take
        priority for the legend.  When the CRS is ``"latlon"`` or ``"utm"``
        the geographic position is inferred automatically for the legend
        (lon from ``"x"``, lat from ``"y"`` for latlon; back-converted from
        UTM for utm).
    resolve_xy_fn
        Optional ``(spec) -> (x_m, y_m)`` override.  When provided it takes
        complete precedence over the built-in CRS logic; ``"lat"``/``"lon"``
        spec keys are still used for the legend if present.
    utm_zone, utm_northern
        UTM zone and hemisphere — required for ``"utm"`` and ``"latlon"``
        conversions.
    utm_origin_e, utm_origin_n
        UTM coordinates of the mesh centre [m] — required for ``"utm"`` and
        ``"latlon"`` conversions.
    out
        Print progress.

    Returns
    -------
    list of dicts, one per borehole, with keys:
        name, pos_str, x_m, y_m, z_top, z_bot, dz, lat, lon,
        depths (ndarray), rho (ndarray), trace_style (dict).
    """
    try:
        import femtic as _fem
    except ImportError:
        raise ImportError("_sample_borehole_logs: femtic module not available.")

    _LINE2D_KEYS = frozenset({
        "color", "c", "ls", "linestyle", "lw", "linewidth",
        "marker", "markersize", "ms", "markeredgecolor", "mec",
        "markeredgewidth", "mew", "markerfacecolor", "mfc",
        "alpha", "zorder", "solid_capstyle", "solid_joinstyle",
        "dash_capstyle", "dash_joinstyle",
    })

    _node_xy   = nodes[:, :2]
    _node_z    = nodes[:, 2]
    _kdtree_xy = None

    def _surface_z_at(x_m: float, y_m: float, k: int = 8) -> float:
        nonlocal _kdtree_xy
        if _kdtree_xy is None:
            if cKDTree is None:
                raise ImportError(
                    "scipy.spatial.cKDTree is required for z_top='surface'.")
            _kdtree_xy = cKDTree(_node_xy)
        _, idxs = _kdtree_xy.query([x_m, y_m], k=k)
        return float(_node_z[idxs].min())

    def _parse_crs(raw):
        """Return (value, crs_str)."""
        if isinstance(raw, (int, float)):
            return float(raw), "model"
        try:
            val, crs = raw
            return val, str(crs).lower().strip()
        except (TypeError, ValueError):
            raise ValueError(
                f"Borehole position {raw!r} must be a scalar or "
                f"(value, 'crs') tuple where crs is 'model'/'utm'/'latlon'.")

    logs = []
    for spec in borehole_sites:
        name = spec.get("name", "?")

        # ---- resolve (x_m, y_m) and infer legend lat/lon --------------------
        leg_lat = spec.get("lat")   # explicit override always wins
        leg_lon = spec.get("lon")

        if resolve_xy_fn is not None:
            # Caller-supplied converter — use it as-is.
            x_m, y_m = resolve_xy_fn(spec)
            # If no explicit lat/lon provided we leave them None
            # (model-local metres shown in legend).
        else:
            x_raw = spec["x"]
            y_raw = spec["y"]
            x_val, x_crs = _parse_crs(x_raw)
            y_val, y_crs = _parse_crs(y_raw)

            if x_crs != y_crs:
                raise ValueError(
                    f"Borehole {name!r}: x CRS ({x_crs!r}) and y CRS "
                    f"({y_crs!r}) must match.")

            crs = x_crs

            if crs == "model":
                x_m, y_m = float(x_val), float(y_val)
                # leg_lat/lon stay as explicit spec values (or None)

            elif crs == "utm":
                # x_val = UTM easting, y_val = UTM northing
                x_m, y_m = _fem.utm_to_model(
                    float(x_val), float(y_val),
                    utm_origin_e, utm_origin_n)
                if leg_lat is None or leg_lon is None:
                    # Back-convert to lat/lon for legend
                    try:
                        import util as _utl
                        _lat, _lon = _utl.utm_to_latlon_zn(
                            float(x_val), float(y_val),
                            utm_zone, utm_northern)
                        leg_lat, leg_lon = _lat, _lon
                    except Exception:
                        pass   # legend falls back to model-local

            elif crs == "latlon":
                # x_val = longitude, y_val = latitude
                lon_v, lat_v = float(x_val), float(y_val)
                x_m, y_m = _fem.latlon_to_model(
                    lat_v, lon_v,
                    utm_zone, utm_northern,
                    utm_origin_e, utm_origin_n)
                if leg_lat is None:
                    leg_lat = lat_v
                if leg_lon is None:
                    leg_lon = lon_v
            else:
                raise ValueError(
                    f"Borehole {name!r}: unknown CRS {crs!r}. "
                    f"Choose 'model', 'utm', or 'latlon'.")

        # ---- z_top ----------------------------------------------------------
        z_top_raw = spec.get("z_top", 0.0)
        if isinstance(z_top_raw, str) and z_top_raw.strip().lower() == "surface":
            z_top = _surface_z_at(x_m, y_m)
            if out:
                print(f"  borehole {name!r}: z_top='surface' resolved to "
                      f"{z_top:.1f} m (mesh node minimum near borehole)")
        else:
            z_top = float(z_top_raw)

        z_bot = float(spec.get("z_bot", 20000.0))
        dz    = float(spec.get("dz", 200.0))
        if out:
            print(f"  borehole {name!r}  x={x_m:.0f} m  y={y_m:.0f} m  "
                  f"z=[{z_top:.0f}..{z_bot:.0f}]  dz={dz:.0f} m")

        depths, rho = _fem.extract_borehole_log(
            nodes, conn, rho_plot, x_m, y_m, z_top, z_bot, dz, out=out)

        # ---- legend annotation ----------------------------------------------
        elev_m = -z_top   # z positive-down → elevation positive-up
        if leg_lat is not None and leg_lon is not None:
            pos_str = (f"lat={float(leg_lat):.4f}°, lon={float(leg_lon):.4f}°, "
                       f"elev={elev_m:+.0f} m")
        else:
            pos_str = f"x={x_m:.0f} m, y={y_m:.0f} m, elev={elev_m:+.0f} m"

        trace_style = {k: v for k, v in spec.items() if k in _LINE2D_KEYS}

        logs.append(dict(
            name=name, pos_str=pos_str,
            x_m=x_m, y_m=y_m, z_top=z_top, z_bot=z_bot, dz=dz,
            lat=leg_lat, lon=leg_lon,
            depths=depths, rho=rho,
            trace_style=trace_style,
        ))
    return logs


def plot_model_slices(
    model_file: Union[str, Path],
    mesh_file: Union[str, Path],
    slices: list,
    *,
    cmap: str = "turbo_r",
    clim=None,
    xlim=None,
    ylim=None,
    zlim=None,
    ocean_color: Optional[str] = "lightgrey",
    ocean_value: float = 0.25,
    air_color: Optional[str] = "white",
    air_bgcolor: Optional[str] = None,
    site_xys: Optional[list] = None,
    obs_coords_only: bool = False,
    projection_dist: Optional[float] = None,
    sites_in_maps: bool = True,
    sites_in_slices: bool = False,
    site_marker: Optional[dict] = None,
    site_marker_slices: Optional[dict] = None,
    map_markers: Optional[list] = None,
    display_coords: str = "model",
    utm_origin_e: float = 0.0,
    utm_origin_n: float = 0.0,
    utm_zone: int = 1,
    utm_northern: bool = True,
    utm_to_latlon_fn=None,
    latlon_to_model_fn=None,
    plot_file=None,
    dpi: int = 200,
    equal_aspect: bool = True,
    depth_km: bool = False,
    horiz_km: bool = False,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    panel_height: float = 5.0,
    panel_width: Optional[float] = None,
    figsize=None,
    alpha_file=None,
    alpha_mode: str = "fade",
    alpha_blank_thresh: float = 0.0,
    mesh_outline: bool = True,
    mesh_outline_color: str = "0.35",
    borehole_sites: Optional[list] = None,
    borehole_style: Optional[dict] = None,
    borehole_clim=None,
    borehole_shared: bool = True,
    borehole_resolve_xy=None,
    tick_fontsize: int = 7,
    label_fontsize: int = 8,
    nrms_annotation: Optional[dict] = None,
    out: bool = True,
):
    """Produce a multi-panel figure of axis-parallel FEMTIC model slices.

    Uses exact tetrahedron-plane intersection -- no selection slab, no dw.
    Every tetrahedron straddling the plane contributes an exact triangle or
    quadrilateral polygon.

    Parameters
    ----------
    model_file, mesh_file
        Resistivity block and mesh.dat paths.
    slices
        List of slice-spec dicts with model-local positions (pre-processed
        by ``fem.resolve_slice_positions``).  Each dict must contain
        ``"kind"`` (``"map"``, ``"ns"``, ``"ew"``, ``"plane"``) and the
        corresponding position keys (``z0``, ``x0``, ``y0``,
        ``point`` / ``strike`` / ``dip``).
    cmap
        Matplotlib colormap name.
    clim
        ``[log10_min, log10_max]``; ``None`` = auto from data.
    xlim, ylim, zlim
        Global axis limits (model-local m); per-panel ``"xlim"`` etc.
        override these.
    ocean_color
        Flat colour for ocean polygon overlay; ``None`` -> use colormap.
    ocean_value
        Ohm*m sentinel for ocean cells (must match inversion setup).
    air_color
        Flat colour for air (above-ground) polygon overlay.  Every element
        whose resistivity exceeds 1e8 Ohm*m is treated as air and rendered
        in this colour (default ``"white"``).  ``None`` -> air polygons are
        not drawn (old behaviour -- leaves blank gaps).
    air_bgcolor
        Axes facecolor for the air region; ``None`` = figure default.
    site_xys
        List of ``(label, x_m, y_m, elev_m)`` tuples in model-local
        metres.  ``None`` / empty -> no markers.
    obs_coords_only
        ``True`` when *site_xys* come from ``observe.dat`` (model-local
        only); suppresses ``"utm"`` / ``"latlon"`` display for markers.
    projection_dist
        Maximum distance [m] from a curtain/plane for a site to appear on
        that panel.  ``None`` = all sites on all panels.
    sites_in_maps, sites_in_slices
        Enable site markers on map and curtain/plane panels respectively.
    site_marker
        Matplotlib ``plot`` kwargs for map-panel site markers.
        Default: ``dict(marker="v", color="black", ms=4, zorder=10)``.
    site_marker_slices
        Matplotlib ``plot`` kwargs for curtain/plane site markers.
        Default: ``dict(marker="o", color="black", ms=4, zorder=10)``.
    map_markers
        List of extra point-marker dicts overlaid on map panels only.
        Each dict: ``"latlon"`` ([lat, lon]), ``"marker"``, ``"color"``,
        ``"ms"``, ``"name"`` (legend label or ``None``).
    display_coords
        ``"model"`` (model-local m), ``"utm"`` (absolute UTM km), or
        ``"latlon"`` (decimal degrees).
    utm_origin_e, utm_origin_n
        UTM easting / northing of the mesh centre [m].
    utm_zone, utm_northern
        UTM zone number and hemisphere flag for coordinate formatting.
    utm_to_latlon_fn
        Callable ``(E_m, N_m, zone, northern) -> (lat, lon)`` used for
        lat/lon tick formatting.  ``None`` -> no lat/lon formatter.
    latlon_to_model_fn
        Callable ``(lat, lon, zone, northern, origin_e, origin_n)
        -> (x_m, y_m)`` used to place ``map_markers``.  ``None`` -> markers
        skipped when ``display_coords != "model"``.
    plot_file
        Save path; ``None`` = interactive ``show()``.
    dpi
        Saved-figure DPI.
    equal_aspect
        Call ``ax.set_aspect("equal")`` on map/ns/ew panels when both axes
        carry the same physical scale.
    depth_km
        Show depth axis in km on curtain and plane panels.
    horiz_km
        Show horizontal axes in km when ``display_coords="model"``.
    nrows, ncols
        Subplot grid shape; ``None`` = 1 x n_panels.  Surplus cells hidden.
    panel_height
        Row height in inches.
    panel_width
        Fixed column width in inches; ``None`` = auto from aspect ratio.
    figsize
        Explicit ``[width, height]`` in inches; overrides auto sizing.
    alpha_file
        Optional path to a second ``resistivity_block_iterX.dat`` file with
        the **same mesh and region structure** as ``model_file``.  Its
        resistivity values are interpreted as **log10 weights** (the raw
        region_rho values are treated directly as log10 numbers; values
        < 0 suppress a polygon, values >= 0 keep it fully visible).
        Typical use: pass a misfit or sensitivity block whose values have
        been stored in log10 form so that cells with poor data coverage or
        high misfit are faded or removed.  ``None`` -> no alpha modulation.
    alpha_mode
        How ``alpha_file`` values drive polygon visibility:

        - ``"fade"``  -- polygon alpha = ``clip(log10_val / alpha_blank_thresh, 0, 1)``
          when ``alpha_blank_thresh < 0``; polygons with ``log10_val >= 0`` are
          fully opaque, polygons with ``log10_val < alpha_blank_thresh`` are
          fully transparent.  Intermediate values produce proportional fading.
        - ``"blank"`` -- hard threshold: polygons with ``log10_val < alpha_blank_thresh``
          are omitted entirely; all others are fully opaque.

        Default ``"fade"``.
    alpha_blank_thresh
        Log10 threshold (<= 0) below which polygons are blanked / fully faded.
        Default ``0.0`` (any negative log10 value triggers suppression).
    mesh_outline
        If ``True`` (default), draw a thin convex-hull outline around the
        intersection polygons on **map** panels and a top-edge polyline on
        **ns** / **ew** curtain panels.  This makes the data footprint visible
        and distinguishes "above topography / outside mesh" from coloured data.
        Requires SciPy (``ConvexHull``); silently skipped when absent.
    mesh_outline_color
        Matplotlib colour string for the outline (default ``"0.35"``).
    borehole_sites
        Optional list of borehole spec dicts (same format as accepted by
        ``plot_borehole_logs``).  When non-empty, borehole panels are
        appended as extra columns to the **right** of the slice grid inside
        the same figure.

        ``borehole_shared=True``  → one extra column, all traces overlaid.
        ``borehole_shared=False`` → one extra column per borehole.

        The borehole column(s) share the depth y-axis scale with the
        leftmost curtain / plane panel present (if any); when only map
        panels are present the y-axis is an independent depth axis.
        ``None`` / empty list → no borehole columns.
    borehole_style
        Baseline Matplotlib ``Line2D`` kwargs for all borehole traces.
        Default: ``lw=1.2, marker="none"``.
    borehole_clim
        ``[rho_min, rho_max]`` in Ohm*m for the borehole x-axis (log scale).
        ``None`` = auto.
    borehole_shared
        ``True``  → all borehole traces on one extra column.
        ``False`` → one extra column per borehole.
    borehole_resolve_xy
        Optional ``(spec) -> (x_m, y_m)`` CRS converter.
        ``None`` = x/y already model-local floats.
    tick_fontsize
        Font size for axis tick labels (main panels and colourbar).
        Default 7.
    label_fontsize
        Font size for axis labels, panel titles, colourbar label, and
        suptitle.  Default 8.
    nrms_annotation
        Optional dict with keys ``nrms`` (float), ``alpha`` (float), and
        ``panel`` (int or None).  When set, a text box reading
        ``nRMS = <value>  (α = <alpha>)`` is placed in the lower-left
        corner of the panel at index ``panel`` (default: 0).
    out
        Print progress messages.
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.collections import PolyCollection
    except ImportError:
        print("  plot_model_slices: Matplotlib not available -- skipping.")
        return

    _sm  = site_marker        or dict(marker="v", color="black", ms=4, zorder=10)
    _sms = site_marker_slices or dict(marker="o", color="black", ms=4, zorder=10)

    # -- inner geometry helpers ------------------------------------------------

    def _axis_slice_params(axis, val):
        normals = [np.array([1., 0., 0.]),
                   np.array([0., 1., 0.]),
                   np.array([0., 0., 1.])]
        pt = np.zeros(3)
        pt[axis] = val
        n = normals[axis]
        ref = np.array([0., 0., 1.]) if axis != 2 else np.array([1., 0., 0.])
        u = np.cross(n, ref); u /= np.linalg.norm(u)
        v = np.cross(n, u);   v /= np.linalg.norm(v)
        return n, pt, u, v, False

    def _tet_plane_intersection(verts, normal, d):
        dots = verts @ normal - d
        pos  = dots >= 0
        if pos.all() or (~pos).all():
            return []
        pts = []
        for i in range(4):
            for j in range(i + 1, 4):
                if pos[i] != pos[j]:
                    t = dots[i] / (dots[i] - dots[j])
                    pts.append(verts[i] + t * (verts[j] - verts[i]))
        c   = np.mean(pts, axis=0)
        u2d = np.cross(normal,
                       np.array([0., 0., 1.]) if abs(normal[2]) < 0.9
                       else np.array([1., 0., 0.]))
        if np.linalg.norm(u2d) < 1e-12:
            return pts
        u2d /= np.linalg.norm(u2d)
        v2d  = np.cross(normal, u2d)
        angles = [np.arctan2((p - c) @ v2d, (p - c) @ u2d) for p in pts]
        return [pts[k] for k in np.argsort(angles)]

    def _slice_geometry(nodes, conn, rho_arr, normal, point, u_ax, v_ax):
        d = float(normal @ point)
        verts_all = nodes[conn]
        polys, vals, eidx = [], [], []
        for k, verts in enumerate(verts_all):
            pts3d = _tet_plane_intersection(verts, normal, d)
            if not pts3d:
                continue
            polys.append([(float(p @ u_ax), float(p @ v_ax)) for p in pts3d])
            with np.errstate(divide="ignore", invalid="ignore"):
                vals.append(math.log10(rho_arr[k]) if rho_arr[k] > 0
                            else float("nan"))
            eidx.append(k)
        return polys, np.asarray(vals, dtype=float), np.asarray(eidx, dtype=int)

    def _compute_poly_alphas(eidx, alpha_vals, mode, thresh):
        """Return per-polygon alpha array (float, 0-1) or None."""
        if alpha_vals is None or len(eidx) == 0:
            return None
        w = alpha_vals[eidx]          # log10 values at intersecting elements
        if mode == "blank":
            return (w >= thresh).astype(float)
        else:  # "fade"
            if thresh >= 0.0:
                # thresh=0 -> anything <0 fully transparent, >=0 fully opaque
                return np.clip(w / min(thresh, -1e-9), 0.0, 1.0) if thresh < 0 \
                       else (w >= 0.0).astype(float)
            a = np.clip(w / thresh, 0.0, 1.0)   # thresh < 0
            return a

    def _outline_convex_hull(ax, polys_display, color, zorder=5):
        """Draw a thin convex-hull outline around all polygon vertices.

        Used on map panels to delimit the data footprint so that blank
        corners (above topography or outside mesh) are visually distinct
        from coloured data.  Silently skipped when SciPy is absent or
        fewer than 3 unique points exist.
        """
        if not polys_display:
            return
        try:
            from scipy.spatial import ConvexHull  # type: ignore
        except ImportError:
            return
        pts = np.array([pt for poly in polys_display for pt in poly])
        if len(pts) < 3:
            return
        try:
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
            hull_closed = np.vstack([hull_pts, hull_pts[0]])
            ax.plot(hull_closed[:, 0], hull_closed[:, 1],
                    color=color, lw=0.6, zorder=zorder,
                    linestyle="-", solid_capstyle="round")
        except Exception:
            return

    def _outline_curtain_top(ax, polys_display, color, zorder=5):
        """Draw a line along the shallowest (minimum-y) polygon vertices per
        x-bin on curtain panels to show the topographic surface cut.

        *polys_display* are 2-D polygons in (horiz, depth) display coords
        where depth increases downward (positive y).  The top edge is the
        minimum-y vertex in each polygon.  A sorted scatter of those minima
        approximates the topographic surface on the curtain.
        """
        if not polys_display:
            return
        # Collect (x, min_depth) per polygon
        top_pts = []
        for poly in polys_display:
            arr = np.array(poly)
            # depth axis is y (index 1); min depth = shallowest (smallest y)
            idx = int(np.argmin(arr[:, 1]))
            top_pts.append(arr[idx])
        if not top_pts:
            return
        top_pts = np.array(top_pts)
        order = np.argsort(top_pts[:, 0])
        tp = top_pts[order]
        ax.plot(tp[:, 0], tp[:, 1],
                color=color, lw=0.6, zorder=zorder,
                linestyle="-", solid_capstyle="round")

    def _plot_slice_panel(ax, polys, vals, *, cmap_obj, norm,
                          ocean_color, ocean_value, air_color, invert_v,
                          poly_alphas=None):
        if not polys:
            return None
        with np.errstate(divide="ignore", invalid="ignore"):
            ov_log = math.log10(ocean_value) if ocean_value > 0 else float("nan")
        # Air: rho ~ 1e9 (flag=1, fixed at AIR_RHO); detect by threshold.
        # vals are log10(rho); log10(1e8) = 8 is a safe threshold between
        # any real rock resistivity and air.
        is_air   = vals > 8.0
        is_ocean = ~is_air & np.isfinite(vals) & np.isclose(vals, ov_log, atol=0.05)
        is_data  = ~is_air & ~is_ocean & np.isfinite(vals)

        # Apply alpha blanking: treat alpha=0 polygons as air (skip them)
        if poly_alphas is not None:
            _blanked = poly_alphas <= 0.0
            is_data  = is_data  & ~_blanked
            is_ocean = is_ocean & ~_blanked

        mappable = None
        if is_data.any():
            data_idx = np.where(is_data)[0]
            if poly_alphas is not None:
                # Render each unique alpha level as a separate PolyCollection
                _alphas = poly_alphas[data_idx]
                _unique = np.unique(np.round(_alphas, 3))
                for _a in _unique:
                    _mask = np.isclose(_alphas, _a, atol=5e-4)
                    pc = PolyCollection(
                        [polys[i] for i in data_idx[_mask]],
                        array=vals[data_idx[_mask]], cmap=cmap_obj, norm=norm,
                        linewidths=0, zorder=2, rasterized=True, alpha=float(_a))
                    ax.add_collection(pc)
                    if mappable is None:
                        mappable = pc
            else:
                pc = PolyCollection(
                    [polys[i] for i in data_idx],
                    array=vals[is_data], cmap=cmap_obj, norm=norm,
                    linewidths=0, zorder=2, rasterized=True)
                ax.add_collection(pc)
                mappable = pc
        if is_ocean.any() and ocean_color is not None:
            oc_idx = np.where(is_ocean)[0]
            _oc_alpha = float(np.mean(poly_alphas[oc_idx])) \
                        if poly_alphas is not None else 1.0
            oc = PolyCollection(
                [polys[i] for i in oc_idx],
                facecolor=ocean_color, linewidths=0, zorder=3,
                rasterized=True, alpha=_oc_alpha)
            ax.add_collection(oc)
        if is_air.any() and air_color is not None:
            air_idx = np.where(is_air)[0]
            ac = PolyCollection(
                [polys[i] for i in air_idx],
                facecolor=air_color, linewidths=0, zorder=4,
                rasterized=True)
            ax.add_collection(ac)
        ax.autoscale_view()
        if invert_v:
            ax.invert_yaxis()
        return mappable

    def _strike_dip_to_normal(strike_deg, dip_deg):
        s, d = math.radians(strike_deg), math.radians(dip_deg)
        return np.array([-math.sin(d) * math.sin(s),
                          math.sin(d) * math.cos(s),
                         -math.cos(d)])

    def _plane_basis(normal):
        ref = (np.array([0., 1., 0.]) if abs(normal[1]) < 0.9
               else np.array([1., 0., 0.]))
        u = np.cross(normal, ref); u /= np.linalg.norm(u)
        v = np.cross(u, normal);   v /= np.linalg.norm(v)
        return u, v

    # -- display offset / scale ------------------------------------------------
    _disp = ("model" if (obs_coords_only and display_coords in ("utm", "latlon"))
             else display_coords)
    if obs_coords_only and display_coords in ("utm", "latlon") and out:
        print("  Note: site positions from observe.dat; "
              f"display_coords={display_coords!r} ignored for site markers.")
    dE = utm_origin_e if _disp in ("utm", "latlon") else 0.0
    dN = utm_origin_n if _disp in ("utm", "latlon") else 0.0
    if _disp == "utm":
        sc, sfx = 1e-3, " [UTM km]"
    elif _disp == "model" and horiz_km:
        sc, sfx = 1e-3, " [km]"
    elif _disp == "latlon":
        sc, sfx = 1.0, " [ deg]"
    else:
        sc, sfx = 1.0, " [m]"

    # lat/lon tick formatters
    _fmt_x = _fmt_y = None
    if _disp == "latlon" and utm_to_latlon_fn is not None:
        import matplotlib.ticker as mticker
        def _lon_fmt(val, _pos):
            _, lon = utm_to_latlon_fn(val, utm_origin_n, utm_zone, utm_northern)
            return f"{lon:.3f}"
        def _lat_fmt(val, _pos):
            lat, _ = utm_to_latlon_fn(utm_origin_e, val, utm_zone, utm_northern)
            return f"{lat:.3f}"
        _fmt_x = mticker.FuncFormatter(_lon_fmt)
        _fmt_y = mticker.FuncFormatter(_lat_fmt)

    _horiz_km_eff = (sc == 1e-3)
    _do_equal = (equal_aspect
                 and _disp in ("model", "utm")
                 and _horiz_km_eff == depth_km)

    # -- load model ------------------------------------------------------------
    if out:
        print(f"  plot: reading model {os.path.basename(str(model_file))}")
    mesh    = read_femtic_mesh(mesh_file)
    block   = read_resistivity_block(model_file)
    rho_elem = map_regions_to_element_rho(block.region_of_elem, block.region_rho)
    # rho_plot: NaN for air, ocean sentinel replaced -- used for colormap norm only.
    # _slice_geometry receives rho_elem (raw values) so every element is
    # intersected; air elements (rho ~ 1e9) are then rendered white in
    # _plot_slice_panel, not skipped.
    rho_plot = prepare_rho_for_plotting(
        rho_elem, air_is_nan=True, ocean_value=float(ocean_value),
        region_of_elem=block.region_of_elem)
    nodes = mesh.nodes
    conn  = mesh.conn

    _non_air_mask     = block.region_of_elem != 0
    _non_air_node_idx = np.unique(conn[_non_air_mask])
    _non_air_nodes    = nodes[_non_air_node_idx]
    z_surf = float(_non_air_nodes[:, 2].min())
    if out:
        _hp_idx = int(np.argmin(_non_air_nodes[:, 2]))
        _hp_x, _hp_y, _hp_z = _non_air_nodes[_hp_idx]
        print(f"  plot: mesh highest point (non-air): "
              f"elev = {-_hp_z:.1f} m  "
              f"model-local x={_hp_x:.1f} m  y={_hp_y:.1f} m")
        if utm_to_latlon_fn is not None:
            try:
                _hp_lat, _hp_lon = utm_to_latlon_fn(
                    utm_origin_e + _hp_x, utm_origin_n + _hp_y,
                    utm_zone, utm_northern)
                print(f"  plot: mesh highest point (non-air): "
                      f"lat={_hp_lat:.6f} deg  lon={_hp_lon:.6f} deg")
            except Exception:
                pass
        # Print mesh-centre coordinates in all available systems
        print(f"  plot: mesh centre  UTM E={utm_origin_e:.1f} m  "
              f"N={utm_origin_n:.1f} m  zone {utm_zone}"
              f"{'N' if utm_northern else 'S'}")
        if utm_to_latlon_fn is not None:
            try:
                _clat, _clon = utm_to_latlon_fn(
                    utm_origin_e, utm_origin_n, utm_zone, utm_northern)
                print(f"  plot: mesh centre  lat={_clat:.6f} deg  lon={_clon:.6f} deg")
            except Exception:
                pass
        print(f"  plot: {len(slices)} panel(s), exact plane-intersection method")

    # -- optional alpha / blanking file ----------------------------------------
    _alpha_vals: Optional[np.ndarray] = None   # per-element log10 weights
    if alpha_file is not None:
        if out:
            print(f"  plot: reading alpha file {os.path.basename(str(alpha_file))}")
        _ablk = read_resistivity_block(alpha_file)
        # region_rho values are treated directly as log10 weights
        _alpha_vals = map_regions_to_element_rho(
            _ablk.region_of_elem, _ablk.region_rho)
        if _alpha_vals.shape[0] != conn.shape[0]:
            raise ValueError(
                f"alpha_file element count ({_alpha_vals.shape[0]}) "
                f"!= mesh element count ({conn.shape[0]})"
            )

    # -- colormap / normalisation ----------------------------------------------
    cmap_obj = matplotlib.colormaps[cmap].copy()
    cmap_obj.set_bad(alpha=0.0)
    if clim is not None:
        norm = mcolors.Normalize(vmin=float(clim[0]), vmax=float(clim[1]))
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            _lall = np.log10(rho_plot[np.isfinite(rho_plot)])
        _lall = _lall[np.isfinite(_lall)]
        norm  = mcolors.Normalize(vmin=float(_lall.min()),
                                  vmax=float(_lall.max()))

    # -- sample boreholes (before figure, so depth range is known) ---------------
    _bh_logs: list = []
    _bh_sites = borehole_sites or []
    if _bh_sites:
        if out:
            print(f"  plot: sampling {len(_bh_sites)} borehole(s) ...")
        _bh_base = dict(lw=1.2, marker="none")
        if borehole_style:
            _bh_base.update(borehole_style)
        _bh_logs = _sample_borehole_logs(
            nodes, conn, rho_plot, _bh_sites,
            resolve_xy_fn=borehole_resolve_xy,
            utm_zone=utm_zone,
            utm_northern=utm_northern,
            utm_origin_e=utm_origin_e,
            utm_origin_n=utm_origin_n,
            out=out)
        # Apply baseline style (per-spec keys already in trace_style from helper)
        for _bl in _bh_logs:
            _ts = dict(_bh_base)
            _ts.update(_bl["trace_style"])
            _bl["trace_style"] = _ts

    # -- figure layout ---------------------------------------------------------
    # Borehole panels occupy extra columns to the right of the slice grid.
    # When borehole_shared=True  → 1 extra column (all traces overlaid).
    # When borehole_shared=False → 1 extra column per borehole.
    n_panels  = len(slices)
    _dz_sc    = 1e-3 if depth_km else 1.0
    _nrows    = int(nrows) if nrows is not None else 1
    _ncols    = int(ncols) if ncols is not None else n_panels
    if _nrows * _ncols < n_panels:
        raise ValueError(
            f"plot_model_slices: grid {_nrows}x{_ncols} = {_nrows * _ncols} "
            f"cells < {n_panels} slices -- increase nrows/ncols.")

    _n_bh_cols = (1 if (borehole_shared or len(_bh_logs) == 1)
                  else len(_bh_logs)) if _bh_logs else 0
    _total_cols = _ncols + _n_bh_cols

    if figsize is not None:
        _fig_w, _fig_h = float(figsize[0]), float(figsize[1])
    else:
        _panel_h = float(panel_height)
        if panel_width is not None:
            _col_widths = [float(panel_width)] * _ncols
        elif _do_equal:
            _pw = []
            for spec in slices:
                kind = spec.get("kind", "map")
                if kind in ("sn", "we"):
                    kind = "ns" if kind == "sn" else "ew"
                _xl = spec.get("xlim", xlim)
                _yl = spec.get("ylim", ylim)
                _zl = spec.get("zlim", zlim)
                if kind == "map":
                    hspan = (_xl[1]-_xl[0])*sc if _xl else _panel_h*200
                    vspan = (_yl[1]-_yl[0])*sc if _yl else _panel_h*200
                elif kind == "ns":
                    hspan = (_yl[1]-_yl[0])*sc    if _yl else _panel_h*200
                    vspan = (_zl[1]-_zl[0])*_dz_sc if _zl else _panel_h*200
                elif kind == "ew":
                    hspan = (_xl[1]-_xl[0])*sc    if _xl else _panel_h*200
                    vspan = (_zl[1]-_zl[0])*_dz_sc if _zl else _panel_h*200
                else:
                    hspan = vspan = 1.0
                ratio = hspan / vspan if vspan > 0 else 1.0
                _pw.append(_panel_h * ratio)
            _pw += [_panel_h] * (_nrows * _ncols - len(_pw))
            _col_widths = [max(_pw[c::_ncols]) for c in range(_ncols)]
        else:
            _col_widths = [_panel_h] * _ncols
        # Borehole columns: narrower than a square slice panel
        _bh_col_w = _panel_h * 0.7
        _fig_w = sum(_col_widths) + _n_bh_cols * _bh_col_w
        _fig_h = _panel_h * _nrows

    # Build figure with gridspec so borehole columns can be narrower
    import matplotlib.gridspec as gridspec
    if _n_bh_cols > 0 and figsize is None:
        _slice_widths = _col_widths + [_bh_col_w] * _n_bh_cols
        fig = plt.figure(figsize=(_fig_w, _fig_h))
        gs  = gridspec.GridSpec(_nrows, _total_cols,
                                width_ratios=_slice_widths,
                                figure=fig)
        _slice_axes = [fig.add_subplot(gs[r, c])
                       for r in range(_nrows) for c in range(_ncols)]
        _bh_axes    = [fig.add_subplot(gs[0, _ncols + bc])
                       for bc in range(_n_bh_cols)]
        # Span all rows for borehole columns
        if _nrows > 1:
            for bc in range(_n_bh_cols):
                _bh_axes[bc].remove()
                _bh_axes[bc] = fig.add_subplot(gs[:, _ncols + bc])
    else:
        fig = plt.figure(figsize=(_fig_w, _fig_h))
        gs  = gridspec.GridSpec(_nrows, _total_cols, figure=fig)
        _slice_axes = [fig.add_subplot(gs[r, c])
                       for r in range(_nrows) for c in range(_ncols)]
        _bh_axes    = [fig.add_subplot(gs[:, _ncols + bc])
                       for bc in range(_n_bh_cols)]

    # Hide surplus slice cells
    for ax in _slice_axes[n_panels:]:
        ax.set_visible(False)
    axes = _slice_axes[:n_panels]
    if air_bgcolor is not None:
        for ax in axes:
            ax.set_facecolor(air_bgcolor)

    _site_xys = site_xys or []

    # -- render each panel -----------------------------------------------------
    for ax, spec in zip(axes, slices):
        kind  = spec.get("kind", "map")
        title = spec.get("title", None)
        _xlim = spec.get("xlim", xlim)
        _ylim = spec.get("ylim", ylim)
        _zlim = spec.get("zlim", zlim)
        mappable = None

        # Normalise reversed-direction aliases before dispatch.
        if kind in ("sn", "we"):
            kind = "ns" if kind == "sn" else "ew"
            spec = dict(spec, kind=kind,
                        invert_x=not spec.get("invert_x", False))

        if kind == "map":
            z0 = float(spec.get("z0", 0.0))
            if out:
                print(f"    map slice z={z0:.0f} m ...")
            normal = np.array([0., 0., 1.])
            point  = np.array([0., 0., z0])
            u_ax   = np.array([1., 0., 0.])
            v_ax   = np.array([0., 1., 0.])
            polys, vals, eidx = _slice_geometry(nodes, conn, rho_elem,
                                          normal, point, u_ax, v_ax)
            polys_d = [[((px + dE)*sc, (py + dN)*sc) for px, py in poly]
                       for poly in polys]
            _pa = _compute_poly_alphas(eidx, _alpha_vals, alpha_mode,
                                       alpha_blank_thresh)
            mappable = _plot_slice_panel(ax, polys_d, vals,
                                         cmap_obj=cmap_obj, norm=norm,
                                         ocean_color=ocean_color,
                                         air_color=air_color, ocean_value=ocean_value, invert_v=False,
                                         poly_alphas=_pa)
            if mesh_outline and polys_d:
                _outline_convex_hull(ax, polys_d, mesh_outline_color)
            ax.set_xlabel(f"x (easting){sfx}", fontsize=label_fontsize)
            ax.set_ylabel(f"y (northing){sfx}", fontsize=label_fontsize)
            if _xlim is not None:
                ax.set_xlim([(v + dE)*sc for v in _xlim])
            if _ylim is not None:
                ax.set_ylim([(v + dN)*sc for v in _ylim])
            if _fmt_x is not None:
                ax.xaxis.set_major_formatter(_fmt_x)
            if _fmt_y is not None:
                ax.yaxis.set_major_formatter(_fmt_y)
            if _do_equal:
                ax.set_aspect("equal", adjustable="box")
            if title is None:
                title = f"Map  z = {z0/1000:.1f} km"
            for sn, sx_m, sy_m, _elev in (_site_xys if sites_in_maps else []):
                mk = dict(_sm); mk.setdefault("label", f"Site {sn}")
                ax.plot((sx_m + dE)*sc, (sy_m + dN)*sc, linestyle="none", **mk)
            for _mm in (map_markers or []):
                _lat, _lon = _mm["latlon"]
                if latlon_to_model_fn is not None:
                    _mx_m, _my_m = latlon_to_model_fn(
                        _lat, _lon, utm_zone, utm_northern,
                        utm_origin_e, utm_origin_n)
                else:
                    _mx_m, _my_m = 0.0, 0.0
                _mk = dict(marker=_mm.get("marker", "+"),
                           color=_mm.get("color", "black"),
                           ms=_mm.get("ms", 8),
                           zorder=_mm.get("zorder", 11),
                           label=_mm.get("name", None))
                _mk.update({k: v for k, v in _mm.items()
                            if k not in ("latlon","marker","color","ms","zorder","name")})
                ax.plot((_mx_m + dE)*sc, (_my_m + dN)*sc, linestyle="none", **_mk)

        elif kind == "ns":
            x0 = float(spec.get("x0", 0.0))
            _invert_x = bool(spec.get("invert_x", False))
            if out:
                print(f"    NS slice x={x0:.0f} m ...")
            normal, point, u_ax, v_ax, inv = _axis_slice_params(0, x0)
            polys, vals, eidx = _slice_geometry(nodes, conn, rho_elem,
                                          normal, point, u_ax, v_ax)
            polys_d = [[((py + dN)*sc, -pz*_dz_sc) for py, pz in poly]
                       for poly in polys]
            _pa = _compute_poly_alphas(eidx, _alpha_vals, alpha_mode,
                                       alpha_blank_thresh)
            mappable = _plot_slice_panel(ax, polys_d, vals,
                                         cmap_obj=cmap_obj, norm=norm,
                                         ocean_color=ocean_color,
                                         air_color=air_color, ocean_value=ocean_value, invert_v=inv,
                                         poly_alphas=_pa)
            ax.set_xlabel(f"y (northing){sfx}", fontsize=label_fontsize)
            ax.set_ylabel("depth (km)" if depth_km else "depth (m)", fontsize=label_fontsize)
            if _ylim is not None:
                ax.set_xlim([(v + dN)*sc for v in _ylim])
            if _zlim is not None:
                ax.set_ylim([_zlim[1]*_dz_sc, _zlim[0]*_dz_sc])
            if _invert_x:
                ax.invert_xaxis()
            if _fmt_y is not None:
                ax.xaxis.set_major_formatter(_fmt_y)
            if _do_equal:
                ax.set_aspect("equal", adjustable="box")
            if title is None:
                title = f"N-S  easting = {(x0 + utm_origin_e)/1000:.1f} km"
            _lbl_l, _lbl_r = ("N", "S") if _invert_x else ("S", "N")
            ax.text(0.02, 0.98, _lbl_l, transform=ax.transAxes,
                    ha="left", va="top", fontsize=label_fontsize, fontweight="bold",
                    clip_on=False, zorder=10)
            ax.text(0.98, 0.98, _lbl_r, transform=ax.transAxes,
                    ha="right", va="top", fontsize=label_fontsize, fontweight="bold",
                    clip_on=False, zorder=10)
            if sites_in_slices:
                for sn, sx_m, sy_m, _elev in _site_xys:
                    if projection_dist is not None and abs(sx_m - x0) > projection_dist:
                        continue
                    mk = dict(_sms); mk.setdefault("label", f"Site {sn}")
                    ax.plot((sy_m + dN)*sc, -_elev*_dz_sc, linestyle="none", **mk)

        elif kind == "ew":
            y0 = float(spec.get("y0", 0.0))
            _invert_x = bool(spec.get("invert_x", False))
            if out:
                print(f"    EW slice y={y0:.0f} m ...")
            normal, point, u_ax, v_ax, inv = _axis_slice_params(1, y0)
            polys, vals, eidx = _slice_geometry(nodes, conn, rho_elem,
                                          normal, point, u_ax, v_ax)
            polys_d = [[((px + dE)*sc, -pz*_dz_sc) for px, pz in poly]
                       for poly in polys]
            _pa = _compute_poly_alphas(eidx, _alpha_vals, alpha_mode,
                                       alpha_blank_thresh)
            mappable = _plot_slice_panel(ax, polys_d, vals,
                                         cmap_obj=cmap_obj, norm=norm,
                                         ocean_color=ocean_color,
                                         air_color=air_color, ocean_value=ocean_value, invert_v=inv,
                                         poly_alphas=_pa)
            ax.set_xlabel(f"x (easting){sfx}", fontsize=label_fontsize)
            ax.set_ylabel("depth (km)" if depth_km else "depth (m)", fontsize=label_fontsize)
            if _xlim is not None:
                ax.set_xlim([(v + dE)*sc for v in _xlim])
            if _zlim is not None:
                ax.set_ylim([_zlim[1]*_dz_sc, _zlim[0]*_dz_sc])
            if _invert_x:
                ax.invert_xaxis()
            if _fmt_x is not None:
                ax.xaxis.set_major_formatter(_fmt_x)
            if _do_equal:
                ax.set_aspect("equal", adjustable="box")
            if title is None:
                title = f"E-W  northing = {(y0 + utm_origin_n)/1000:.1f} km"
            _lbl_l, _lbl_r = ("E", "W") if _invert_x else ("W", "E")
            ax.text(0.02, 0.98, _lbl_l, transform=ax.transAxes,
                    ha="left", va="top", fontsize=label_fontsize, fontweight="bold",
                    clip_on=False, zorder=10)
            ax.text(0.98, 0.98, _lbl_r, transform=ax.transAxes,
                    ha="right", va="top", fontsize=label_fontsize, fontweight="bold",
                    clip_on=False, zorder=10)
            if sites_in_slices:
                for sn, sx_m, sy_m, _elev in _site_xys:
                    if projection_dist is not None and abs(sy_m - y0) > projection_dist:
                        continue
                    mk = dict(_sms); mk.setdefault("label", f"Site {sn}")
                    ax.plot((sx_m + dE)*sc, -_elev*_dz_sc, linestyle="none", **mk)

        elif kind == "plane":
            _pt     = np.asarray(spec.get("point", [0., 0., 0.]), dtype=float)
            _strike = float(spec.get("strike", 0.0))
            _dip    = float(spec.get("dip", 90.0))
            _invert_x = bool(spec.get("invert_x", False))
            if out:
                print(f"    plane slice strike={_strike:.0f} deg dip={_dip:.0f} deg ...")
            normal = _strike_dip_to_normal(_strike, _dip)
            u_ax, v_ax = _plane_basis(normal)
            polys, vals, eidx = _slice_geometry(nodes, conn, rho_elem,
                                          normal, _pt, u_ax, v_ax)
            _pa = _compute_poly_alphas(eidx, _alpha_vals, alpha_mode,
                                       alpha_blank_thresh)
            mappable = _plot_slice_panel(ax, polys, vals,
                                         cmap_obj=cmap_obj, norm=norm,
                                         ocean_color=ocean_color,
                                         air_color=air_color, ocean_value=ocean_value, invert_v=True,
                                         poly_alphas=_pa)
            ax.set_xlabel("along-strike (m)", fontsize=label_fontsize)
            ax.set_ylabel("down-dip (km)" if depth_km else "down-dip (m)", fontsize=label_fontsize)
            if _xlim is not None:
                ax.set_xlim(_xlim)
            if _ylim is not None:
                ax.set_ylim(_ylim)
            if _invert_x:
                ax.invert_xaxis()
            if title is None:
                title = f"Plane  str={_strike:.0f} deg  dip={_dip:.0f} deg"
            if sites_in_slices:
                for sn, sx_m, sy_m, _elev in _site_xys:
                    site_xyz  = np.array([sx_m, sy_m, -_elev]) - _pt
                    perp_dist = abs(float(np.dot(site_xyz, normal)))
                    if projection_dist is not None and perp_dist > projection_dist:
                        continue
                    u_coord = float(np.dot(site_xyz, u_ax))
                    mk = dict(_sms); mk.setdefault("label", f"Site {sn}")
                    ax.plot(u_coord, -_elev*_dz_sc, linestyle="none", **mk)

        else:
            ax.set_visible(False)
            print(f"  plot: unknown slice kind {kind!r} -- skipped.")
            continue

        if mappable is not None:
            cb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("log10(rho / Ohm*m)", fontsize=label_fontsize)
            cb.ax.tick_params(labelsize=tick_fontsize)

        ax.set_title(title, fontsize=label_fontsize + 1)
        ax.tick_params(labelsize=tick_fontsize)
        _show_legend = (
            (_site_xys and (
                (sites_in_maps  and kind == "map") or
                (sites_in_slices and kind in ("ns", "ew", "plane"))
            )) or
            (map_markers and kind == "map" and
             any(m.get("name") for m in map_markers))
        )
        if _show_legend:
            ax.legend(fontsize=tick_fontsize, loc="lower right")

    # -- nRMS annotation -------------------------------------------------------
    if nrms_annotation is not None:
        _ann_panel = nrms_annotation.get("panel") or 0
        _ann_panel = min(_ann_panel, len(axes) - 1)
        _ann_ax    = axes[_ann_panel]
        _ann_nrms  = nrms_annotation["nrms"]
        _ann_alpha = nrms_annotation.get("alpha")
        _ann_txt   = (f"nRMS = {_ann_nrms:.4f}\n\u03b1 = {_ann_alpha:.4g}"
                      if _ann_alpha is not None
                      else f"nRMS = {_ann_nrms:.4f}")
        _ann_ax.text(
            0.02, 0.03, _ann_txt,
            transform=_ann_ax.transAxes,
            ha="left", va="bottom",
            fontsize=label_fontsize,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="0.6", alpha=0.85),
            clip_on=False, zorder=10,
        )

    # -- render borehole columns -----------------------------------------------
    if _bh_logs and _bh_axes:
        # Find the first curtain/plane axes for y-axis sharing (depth scale).
        _curtain_ax = None
        for _ax_s, _sp in zip(axes, slices):
            if _sp.get("kind", "map") in ("ns", "ew", "plane"):
                _curtain_ax = _ax_s
                break

        if borehole_shared:
            # All boreholes on one axes
            _bh_ax = _bh_axes[0]
            for _bl in _bh_logs:
                _depth_plot = _bl["depths"] / 1000.0 if depth_km else _bl["depths"]
                _rho_v = np.where(np.isfinite(_bl["rho"]) & (_bl["rho"] > 0),
                                  _bl["rho"], np.nan)
                _lbl = f"{_bl['name']}\n{_bl['pos_str']}"
                _bh_ax.plot(_rho_v, _depth_plot, label=_lbl, **_bl["trace_style"])
            _bh_ax.set_xscale("log")
            _bh_ax.set_xlabel("resistivity\n(Ohm·m)", fontsize=label_fontsize)
            _bh_ax.set_ylabel(
                "depth (km)" if depth_km else "depth (m)", fontsize=label_fontsize)
            _bh_ax.invert_yaxis()
            if borehole_clim is not None:
                _bh_ax.set_xlim(borehole_clim)
            _bh_ax.tick_params(labelsize=tick_fontsize)
            _bh_ax.legend(fontsize=tick_fontsize, loc="lower right")
            _bh_ax.set_title("Boreholes", fontsize=label_fontsize + 1)
            _bh_ax.grid(True, which="both", axis="x", lw=0.3, alpha=0.5)
            _bh_ax.grid(True, which="major", axis="y", lw=0.3, alpha=0.5)
            if _curtain_ax is not None:
                _curtain_ax.get_shared_y_axes().join(_curtain_ax, _bh_ax)
        else:
            # One column per borehole
            for _bi, (_bh_ax, _bl) in enumerate(zip(_bh_axes, _bh_logs)):
                _depth_plot = _bl["depths"] / 1000.0 if depth_km else _bl["depths"]
                _rho_v = np.where(np.isfinite(_bl["rho"]) & (_bl["rho"] > 0),
                                  _bl["rho"], np.nan)
                _bh_ax.plot(_rho_v, _depth_plot, **_bl["trace_style"])
                _bh_ax.set_xscale("log")
                _bh_ax.set_xlabel("resistivity\n(Ohm·m)", fontsize=label_fontsize)
                if _bi == 0:
                    _bh_ax.set_ylabel(
                        "depth (km)" if depth_km else "depth (m)", fontsize=label_fontsize)
                _bh_ax.invert_yaxis()
                if borehole_clim is not None:
                    _bh_ax.set_xlim(borehole_clim)
                _bh_ax.tick_params(labelsize=tick_fontsize)
                _bh_ax.set_title(f"{_bl['name']}\n{_bl['pos_str']}", fontsize=label_fontsize)
                _bh_ax.grid(True, which="both", axis="x", lw=0.3, alpha=0.5)
                _bh_ax.grid(True, which="major", axis="y", lw=0.3, alpha=0.5)
                if _curtain_ax is not None and _bi == 0:
                    _curtain_ax.get_shared_y_axes().join(_curtain_ax, _bh_ax)
                elif _bi > 0:
                    _bh_axes[0].get_shared_y_axes().join(_bh_axes[0], _bh_ax)

    fig.suptitle(f"Model: {os.path.basename(str(model_file))}", fontsize=label_fontsize + 2)
    fig.tight_layout()
    if plot_file is not None:
        fig.savefig(plot_file, dpi=dpi, bbox_inches="tight")
        if out:
            print(f"  plot: saved -> {plot_file}")
    else:
        plt.show()


# =============================================================================
# 1-D borehole resistivity log
# =============================================================================

def plot_borehole_logs(
    model_file: Union[str, Path],
    mesh_file: Union[str, Path],
    borehole_sites: list,
    *,
    resolve_xy_fn=None,
    utm_zone: int = 1,
    utm_northern: bool = True,
    utm_origin_e: float = 0.0,
    utm_origin_n: float = 0.0,
    ocean_value: float = 0.25,
    clim=None,
    borehole_style: Optional[dict] = None,
    shared: bool = True,
    markers: Optional[list] = None,
    legend_fontsize: int = 9,
    tick_fontsize: Optional[int] = None,
    label_fontsize: Optional[int] = None,
    npz_file=None,
    plot_file=None,
    dpi: int = 200,
    out: bool = True,
):
    """Produce a 1-D rho vs depth figure for a list of boreholes (log x-axis).

    Resistivity is sampled at regular depth intervals using
    ``fem.extract_borehole_log`` (point-in-element, exact barycentric test).
    Air / out-of-mesh levels appear as gaps (NaN).

    The x-axis is logarithmic and shows rho in Ohm*m.  The y-axis shows depth
    in km increasing downward from z_top.

    Sampled data are always exported to an NPZ file before any plotting.  The
    NPZ contains one ``depth_<name>`` and one ``rho_<name>`` array per
    borehole, plus a scalar string array ``header`` with a JSON metadata
    block (model file, mesh file, creation timestamp, per-borehole geometry).

    Parameters
    ----------
    model_file, mesh_file
        Resistivity block and mesh.dat paths.
    borehole_sites
        List of borehole spec dicts.  Each dict must contain:

        ``"name"`` (str), ``"x"`` and ``"y"`` (model-local m or as returned
        by *resolve_xy_fn*), ``"z_top"``, ``"z_bot"``, ``"dz"`` (all in
        model-local metres, z positive-down).

        **Optional spec keys:**

        ``"z_top"`` may be the string ``"surface"`` (case-insensitive).  In
        that case z_top is auto-detected as the minimum mesh node z at the
        borehole (x, y) location using a KD-tree nearest-column search.

        ``"lat"`` and ``"lon"`` (floats, decimal degrees) — when present, the
        legend / panel title shows geographic coordinates instead of model-local
        x/y metres.

        Any Matplotlib ``Line2D`` keyword (``"color"``, ``"ls"``,
        ``"linestyle"``, ``"lw"``, ``"linewidth"``, ``"marker"``, ``"alpha"``,
        ``"zorder"``, …) placed directly in the spec dict overrides the global
        *borehole_style* for that trace only.

        The legend / panel title is always annotated with position and surface
        elevation so that the figure is self-documenting.
    resolve_xy_fn
        Optional callable ``(spec) -> (x_m, y_m)`` that fully overrides the
        built-in CRS conversion.  When ``None`` the helper converts CRS-tagged
        ``"x"``/``"y"`` values directly using *utm_zone* / *utm_origin_e* /
        *utm_origin_n*.
    utm_zone, utm_northern
        UTM zone number and hemisphere flag.  Required when any spec uses
        ``(value, "utm")`` or ``(value, "latlon")`` CRS tags and
        *resolve_xy_fn* is ``None``.
    utm_origin_e, utm_origin_n
        UTM easting / northing of the mesh centre [m].  Required for the
        same CRS tag cases.
    ocean_value
        Ohm*m sentinel for ocean cells.
    clim
        ``[rho_min, rho_max]`` in Ohm*m for the x-axis; ``None`` = auto.
        Example: ``[1.0, 1e4]``.
    borehole_style
        Matplotlib line kwargs applied as the baseline style for every trace.
        Per-spec keys in ``borehole_sites`` override these for individual
        traces.  Default baseline: ``lw=1.2, marker="none"``.
    shared
        ``True`` -> all boreholes on one axes; ``False`` -> one panel each.
    markers
        Optional list of free-annotation dicts placed on the depth axis after
        all traces are drawn.  Each dict may contain:

        ``"depth"``  (float, **required**) — depth in **metres** (z-down) at
        which the annotation is placed.  Converted to km internally.

        ``"rho"``  (float, optional) — x-position of the arrow tip in Ohm·m.
        When omitted the annotation is placed at the left edge of the x-axis.

        ``"text"``  (str, optional, default ``""``) — annotation string.

        ``"borehole"``  (str or list of str, optional) — borehole ``"name"``
        value(s) this marker applies to.  When omitted (or ``None``) the
        marker is placed on **all** panels (or on the shared axes when
        ``shared=True``).

        ``"rho_text"``  (float, optional) — x-position of the annotation text
        in Ohm·m (same log-scale units as the x-axis).  When omitted the text
        is placed at ``rho * 3`` (roughly half a decade to the right on a log
        axis), or at ``xlim[0] * 3`` when ``"rho"`` is also absent.

        ``"depth_text"``  (float, optional) — y-position of the annotation
        text in **metres** (z-down), converted to km internally.  When omitted
        the text is placed 300 m above the tip
        (i.e. ``depth - 300``).

        Both positions are in the natural display units of the axes — Ohm·m
        for x and metres (stored as km) for y — so you can read the values
        directly off the plot axes when choosing them.

        Any remaining keys are forwarded verbatim to ``ax.annotate`` as
        keyword arguments (e.g. ``arrowprops``, ``color``, ``fontsize``,
        ``ha``, ``va``, ``fontweight``, …).  When ``arrowprops`` is absent a
        default thin black arrow is drawn.

        Example::

            markers = [
                dict(depth=1500., rho=10., text="conductor",
                     rho_text=30., depth_text=1200.,
                     borehole="borehole1",
                     color="red", fontsize=8, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color="red", lw=1.2)),
                dict(depth=3000., rho_text=500., depth_text=2600.,
                     text="resistive basement", color="navy"),
            ]

    legend_fontsize
        Font size for the borehole legend (shared mode) and panel titles
        (per-panel mode).  Default ``9``.
    tick_fontsize
        Font size for axis tick labels.  ``None`` → ``legend_fontsize - 2``
        (clamped to at least 6).
    label_fontsize
        Font size for axis labels and standalone titles.  ``None`` →
        ``legend_fontsize``.
    npz_file
        Path for the NPZ data export.

        * explicit path  → saved there.
        * ``None``       → derived from *plot_file* by replacing its extension
          with ``.npz``; if *plot_file* is also ``None`` the file is written
          as ``"borehole_logs.npz"`` in the current working directory.

        Set to ``False`` to suppress NPZ export entirely.
    plot_file
        Save path for the figure; ``None`` = interactive ``show()``.
    dpi
        Figure DPI.
    out
        Print progress messages.
    """
    import json
    import datetime

    # Keys in a spec dict that are Matplotlib Line2D kwargs (not borehole meta).
    if not borehole_sites:
        if out:
            print("  plot_borehole_logs: borehole_sites is empty -- skipping.")
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  plot_borehole_logs: Matplotlib not available -- skipping.")
        return

    _tick_fs  = tick_fontsize  if tick_fontsize  is not None else max(legend_fontsize - 2, 6)
    _label_fs = label_fontsize if label_fontsize is not None else legend_fontsize

    if out:
        print(f"  boreholes: reading model {os.path.basename(str(model_file))}")
    mesh     = read_femtic_mesh(mesh_file)
    block    = read_resistivity_block(model_file)
    rho_elem = map_regions_to_element_rho(block.region_of_elem, block.region_rho)
    rho_plot = prepare_rho_for_plotting(
        rho_elem, air_is_nan=True, ocean_value=float(ocean_value),
        region_of_elem=block.region_of_elem)
    nodes = mesh.nodes
    conn  = mesh.conn

    # Baseline style — per-spec keys from _sample_borehole_logs override below.
    base_style = dict(lw=1.2, marker="none")
    if borehole_style:
        base_style.update(borehole_style)

    logs = _sample_borehole_logs(
        nodes, conn, rho_plot, borehole_sites,
        resolve_xy_fn=resolve_xy_fn,
        utm_zone=utm_zone,
        utm_northern=utm_northern,
        utm_origin_e=utm_origin_e,
        utm_origin_n=utm_origin_n,
        out=out)

    # Apply baseline style (trace_style from helper holds only per-spec overrides)
    for log in logs:
        _ts = dict(base_style)
        _ts.update(log["trace_style"])
        log["trace_style"] = _ts

    # -------------------------------------------------------------------------
    # NPZ export
    # -------------------------------------------------------------------------
    if npz_file is not False:
        # Resolve output path
        if npz_file is None:
            if plot_file is not None:
                _stem = os.path.splitext(str(plot_file))[0]
                _npz_path = _stem + ".npz"
            else:
                _npz_path = "borehole_logs.npz"
        else:
            _npz_path = str(npz_file)

        # Build JSON header
        _bh_meta = []
        for log in logs:
            _entry = dict(
                name   = log["name"],
                x_m    = log["x_m"],
                y_m    = log["y_m"],
                z_top  = log["z_top"],
                z_bot  = log["z_bot"],
                dz     = log["dz"],
                n_levels = int(log["depths"].size),
                depth_key = f"depth_{log['name']}",
                rho_key   = f"rho_{log['name']}",
            )
            if log["lat"] is not None:
                _entry["lat"] = float(log["lat"])
            if log["lon"] is not None:
                _entry["lon"] = float(log["lon"])
            _bh_meta.append(_entry)

        _header_dict = dict(
            created      = datetime.datetime.now().isoformat(timespec="seconds"),
            model_file   = str(model_file),
            mesh_file    = str(mesh_file),
            ocean_value  = ocean_value,
            depth_unit   = "m (z positive-down from datum)",
            rho_unit     = "Ohm*m (NaN = air / outside mesh)",
            boreholes    = _bh_meta,
        )
        _header_json = json.dumps(_header_dict, indent=2)

        # Pack arrays: depth_<name>, rho_<name> for each borehole + header
        _arrays: dict = {"header": np.array(_header_json)}
        for log in logs:
            _safe = log["name"].replace(" ", "_")
            _arrays[f"depth_{_safe}"] = log["depths"]
            _arrays[f"rho_{_safe}"]   = log["rho"]

        np.savez(_npz_path, **_arrays)
        if out:
            print(f"  boreholes: NPZ saved -> {_npz_path}")

    # -------------------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------------------
    n = len(logs)
    if shared:
        fig, ax_single = plt.subplots(figsize=(6, 8))
        ax_arr = [ax_single] * n
    else:
        fig, axs = plt.subplots(1, n, figsize=(4 * n, 8), sharey=True)
        ax_arr = list(axs) if n > 1 else [axs]

    # Map borehole name -> axes for marker targeting
    name_to_ax: dict = {}
    for i, (log, ax) in enumerate(zip(logs, ax_arr)):
        depth_km = log["depths"] / 1000.0
        rho_vals = log["rho"]
        # Replace non-positive / NaN so log-scale gaps render cleanly.
        rho_plot_vals = np.where(
            np.isfinite(rho_vals) & (rho_vals > 0), rho_vals, np.nan)

        legend_label = f"{log['name']}\n{log['pos_str']}"
        ax.plot(rho_plot_vals, depth_km, label=legend_label,
                **log["trace_style"])
        ax.set_xscale("log")
        ax.set_xlabel("resistivity (Ohm·m)", fontsize=_label_fs)
        ax.set_ylabel("depth (km)", fontsize=_label_fs)
        ax.invert_yaxis()
        if clim is not None:
            ax.set_xlim(clim)
        if not shared:
            ax.set_title(f"{log['name']}\n{log['pos_str']}",
                         fontsize=legend_fontsize)
            ax.tick_params(labelsize=_tick_fs)
        name_to_ax[log["name"]] = ax

    if shared:
        ax_arr[0].legend(fontsize=legend_fontsize, loc="lower right")
        ax_arr[0].set_title("Borehole resistivity logs",
                            fontsize=legend_fontsize + 1)
    for ax in set(ax_arr):
        ax.tick_params(labelsize=_tick_fs)
        ax.grid(True, which="both", axis="x", lw=0.4, alpha=0.5)
        ax.grid(True, which="major", axis="y", lw=0.4, alpha=0.5)

    # -------------------------------------------------------------------------
    # Free markers (arrows + text)
    # -------------------------------------------------------------------------
    _DEFAULT_ARROWPROPS = dict(arrowstyle="->", color="black", lw=0.9)

    if markers:
        for mk in markers:
            mk = dict(mk)  # shallow copy — don't mutate caller's dict

            # --- required: arrow tip depth in metres -> km
            depth_m     = float(mk.pop("depth"))
            depth_km_mk = depth_m / 1000.0

            # --- optional: arrow tip rho [Ohm*m]; None -> left x-limit edge
            rho_tip = mk.pop("rho", None)

            # --- optional: text position — both in display units
            #     rho_text  : Ohm*m (x-axis)
            #     depth_text: metres z-down (converted to km)
            rho_text_spec   = mk.pop("rho_text",   None)
            depth_text_spec = mk.pop("depth_text", None)

            # --- optional: text label
            text = mk.pop("text", "")

            # --- optional: borehole targeting
            bh_target = mk.pop("borehole", None)
            if bh_target is None:
                target_axes = list(set(ax_arr))
            elif isinstance(bh_target, str):
                target_axes = ([name_to_ax[bh_target]]
                               if bh_target in name_to_ax else [])
            else:
                target_axes = [name_to_ax[b] for b in bh_target
                               if b in name_to_ax]

            # --- arrowprops: pop from mk or use default
            arrowprops = mk.pop("arrowprops", dict(_DEFAULT_ARROWPROPS))

            # Remaining keys forwarded to ax.annotate (color, fontsize, …)
            annotate_kw = mk

            for ax in target_axes:
                # Arrow tip x
                xlim_cur = ax.get_xlim()
                x_tip = float(rho_tip) if rho_tip is not None else xlim_cur[0]

                # Text position x: explicit Ohm*m, else ~half decade right
                x_text = (float(rho_text_spec) if rho_text_spec is not None
                          else x_tip * 3.0)

                # Text position y: explicit metres->km, else 300 m above tip
                y_text = (float(depth_text_spec) / 1000.0
                          if depth_text_spec is not None
                          else depth_km_mk - 0.3)

                ax.annotate(
                    text,
                    xy=(x_tip, depth_km_mk),
                    xytext=(x_text, y_text),
                    arrowprops=arrowprops,
                    **annotate_kw,
                )

    fig.suptitle(f"Model: {os.path.basename(str(model_file))}", fontsize=legend_fontsize + 1)
    fig.tight_layout()
    if plot_file is not None:
        fig.savefig(plot_file, dpi=dpi, bbox_inches="tight")
        if out:
            print(f"  boreholes: figure saved -> {plot_file}")
    else:
        plt.show()


# =============================================================================
# Ensemble slice plot  (exact tet-plane intersection, same as femtic_mod_plot)
# =============================================================================

def plot_ensemble_slices(
    member_files: list,
    mesh_file: Union[str, Path],
    slices: list,
    *,
    labels: Optional[list] = None,
    stat_rows: Sequence[str] = ("mean", "std"),
    cmap: str = "turbo_r",
    clim: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    zlim: Optional[Tuple[float, float]] = None,
    ocean_color: Optional[str] = "lightgrey",
    ocean_value: float = 0.25,
    air_bgcolor: Optional[str] = None,
    plot_file: Optional[Union[str, Path]] = None,
    per_member_file: bool = False,
    dpi: int = 200,
    tick_fontsize: int = 6,
    label_fontsize: int = 7,
    out: bool = True,
) -> None:
    """Produce a joint ensemble figure using exact tetrahedron-plane intersection.

    Layout: one row per ensemble member, followed by optional statistical
    summary rows (mean, std, median of log10(rho) across all members).
    Columns correspond to the entries of *slices* -- the same slice-spec list
    used by ``femtic_mod_plot.plot_model_slices``.

    The mesh is parsed **once** and slice polygon geometry is precomputed
    **once** per slice position; only the per-element resistivity vector is
    swapped for each member, making the function efficient for large meshes.

    Parameters
    ----------
    member_files
        List of paths to ``resistivity_block_iterX.dat`` files, one per member.
    mesh_file
        Path to the shared ``mesh.dat``.
    slices
        List of slice-spec dicts with all positions already in model-local metres
        (pre-process with ``femtic_mod_plot.resolve_slices`` when using CRS-tagged
        positions).  Each dict must contain at least ``"kind"`` and the
        corresponding position key (``z0`` / ``x0`` / ``y0`` / ``point``).
        Optional per-panel ``xlim`` / ``ylim`` / ``zlim`` / ``title`` keys are
        honoured.
    labels
        Row label strings, one per member.  ``None`` -> "Member 0", "Member 1", ...
    stat_rows
        Stat-summary rows appended after all member rows.  Any subset of
        ``"mean"``, ``"std"``, ``"median"`` in any order.  Pass ``()`` for no
        stat rows.
    cmap
        Matplotlib colormap name for member and mean/median rows.
    clim
        ``[vmin, vmax]`` in log10(Ohm*m).  ``None`` -> auto from ensemble range.
    xlim, ylim, zlim
        Global axis limits in model-local metres; per-panel keys override.
    ocean_color
        Flat colour for ocean/lake cells.  ``None`` -> use colormap.
    ocean_value
        Resistivity sentinel (Ohm*m) identifying ocean cells (default 0.25 Ohm*m).
    air_bgcolor
        Axes facecolor shown through transparent air cells.  ``None`` = figure default.
    plot_file
        Joint figure output path.  ``None`` -> interactive ``plt.show()``.
    per_member_file
        If ``True`` and *plot_file* is set, also save one single-row figure per
        member.  File names are derived from *plot_file* by inserting
        ``_memberN`` before the extension.
    dpi
        Saved-figure DPI.
    tick_fontsize
        Font size for axis tick labels and colourbar ticks.  Default ``6``.
    label_fontsize
        Font size for axis labels, panel titles, row labels, colourbar labels,
        and suptitle.  Default ``7``.
    out
        Print progress messages.

    Notes
    -----
    The ``"std"`` row is rendered on a separate sequential colormap (``cividis``)
    anchored at zero -- because standard deviation is always non-negative and
    has a different physical meaning from resistivity.  Mean and median rows
    share the main colormap / clim.
    """
    import math
    import os

    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import matplotlib.cm as mcm
        from matplotlib.collections import PolyCollection
    except ImportError:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for plot_ensemble_slices."
        )

    member_files = list(member_files)
    if not member_files:
        if out:
            print("  plot_ensemble_slices: member_files is empty -- skipping.")
        return

    n_members = len(member_files)
    n_slices  = len(slices)
    stat_rows = list(stat_rows or [])
    n_rows    = n_members + len(stat_rows)

    if labels is None:
        labels = [f"Member {i}" for i in range(n_members)]
    else:
        labels = list(labels)
        if len(labels) < n_members:
            labels += [f"Member {i}" for i in range(len(labels), n_members)]

    # -- geometry helpers -----------------------------------------------------

    def _tet_plane_intersection(verts, normal, d):
        dots = verts @ normal - d
        pos  = dots >= 0
        if pos.all() or (~pos).all():
            return []
        pts = []
        for i in range(4):
            for j in range(i + 1, 4):
                if pos[i] != pos[j]:
                    t = dots[i] / (dots[i] - dots[j])
                    pts.append(verts[i] + t * (verts[j] - verts[i]))
        c   = np.mean(pts, axis=0)
        u2d = np.cross(normal,
                       np.array([0., 0., 1.]) if abs(normal[2]) < 0.9
                       else np.array([1., 0., 0.]))
        if np.linalg.norm(u2d) < 1e-12:
            return pts
        u2d /= np.linalg.norm(u2d)
        v2d  = np.cross(normal, u2d)
        angles = [np.arctan2((p - c) @ v2d, (p - c) @ u2d) for p in pts]
        return [pts[k] for k in np.argsort(angles)]

    def _axis_slice_params(axis, val):
        normals = [np.array([1., 0., 0.]),
                   np.array([0., 1., 0.]),
                   np.array([0., 0., 1.])]
        inv = [True, True, False]
        pt  = np.zeros(3); pt[axis] = val
        n   = normals[axis]
        ref = np.array([0., 0., 1.]) if axis != 2 else np.array([1., 0., 0.])
        u   = np.cross(n, ref); u /= np.linalg.norm(u)
        v   = np.cross(n, u);   v /= np.linalg.norm(v)
        return n, pt, u, v, inv[axis]

    def _slice_geometry_indices(nodes, conn, normal, point, u_ax, v_ax):
        d         = float(normal @ point)
        verts_all = nodes[conn]
        polys, elem_idx = [], []
        for k, verts in enumerate(verts_all):
            pts3d = _tet_plane_intersection(verts, normal, d)
            if not pts3d:
                continue
            polys.append([(float(p @ u_ax), float(p @ v_ax)) for p in pts3d])
            elem_idx.append(k)
        return polys, np.asarray(elem_idx, dtype=int)

    def _strike_dip_to_normal(strike_deg, dip_deg):
        s = math.radians(strike_deg)
        d = math.radians(dip_deg)
        return np.array([-math.sin(d) * math.sin(s),
                          math.sin(d) * math.cos(s),
                         -math.cos(d)])

    def _plane_basis(normal):
        ref = np.array([0., 1., 0.]) if abs(normal[1]) < 0.9 \
              else np.array([1., 0., 0.])
        u = np.cross(normal, ref); u /= np.linalg.norm(u)
        v = np.cross(u, normal);   v /= np.linalg.norm(v)
        return u, v

    def _plot_panel(fig_ref, ax, polys, vals, *, cmap_obj, norm,
                    oc_color, oc_value, invert_v, row_label, col,
                    cb_label, show_ylabel):
        if not polys:
            return
        with np.errstate(divide="ignore", invalid="ignore"):
            ov_log = math.log10(oc_value) if oc_value > 0 else float("nan")
        is_ocean = np.isclose(vals, ov_log, atol=0.05)
        is_air   = ~np.isfinite(vals)
        is_data  = ~is_ocean & ~is_air
        mappable = None
        if is_data.any():
            pc = PolyCollection(
                [polys[k] for k in np.where(is_data)[0]],
                array=vals[is_data], cmap=cmap_obj, norm=norm,
                linewidths=0, zorder=2, rasterized=True)
            ax.add_collection(pc)
            mappable = pc
        if is_ocean.any() and oc_color is not None:
            oc = PolyCollection(
                [polys[k] for k in np.where(is_ocean)[0]],
                facecolor=oc_color, linewidths=0, zorder=3, rasterized=True)
            ax.add_collection(oc)
        ax.autoscale_view()
        if invert_v:
            ax.invert_yaxis()
        if show_ylabel:
            ax.set_ylabel(row_label, fontsize=label_fontsize, labelpad=4)
        if mappable is not None:
            cb = fig_ref.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(cb_label, fontsize=label_fontsize)
            cb.ax.tick_params(labelsize=tick_fontsize)

    # -- load mesh -------------------------------------------------------------
    if out:
        print(f"  ensemble: reading mesh {Path(mesh_file).name}")
    mesh  = read_femtic_mesh(mesh_file)
    nodes = mesh.nodes
    conn  = mesh.conn

    # -- precompute slice geometry once ----------------------------------------
    if out:
        print(f"  ensemble: precomputing geometry for {n_slices} slice(s) ...")
    slice_geom = []
    for spec in slices:
        kind       = spec.get("kind", "map")
        _xlim      = spec.get("xlim", xlim)
        _ylim      = spec.get("ylim", ylim)
        _zlim      = spec.get("zlim", zlim)
        title_tmpl = spec.get("title", None)

        # Normalise reversed-direction aliases before dispatch.
        if kind in ("sn", "we"):
            kind = "ns" if kind == "sn" else "ew"
            spec = dict(spec, kind=kind,
                        invert_x=not spec.get("invert_x", False))

        if kind == "map":
            z0     = float(spec.get("z0", 0.0))
            normal = np.array([0., 0., 1.])
            point  = np.array([0., 0., z0])
            u_ax   = np.array([1., 0., 0.])
            v_ax   = np.array([0., 1., 0.])
            polys, eidx = _slice_geometry_indices(nodes, conn, normal, point, u_ax, v_ax)
            title_tmpl  = title_tmpl or f"Map  z = {z0/1000:.1f} km"
            slice_geom.append(dict(polys=polys, eidx=eidx, invert_v=False,
                                   invert_x=bool(spec.get("invert_x", False)),
                                   xlabel="x (easting) [m]", ylabel="y (northing) [m]",
                                   xlim=_xlim, ylim=_ylim, zlim=None, title=title_tmpl))

        elif kind == "ns":
            x0 = float(spec.get("x0", 0.0))
            normal, point, u_ax, v_ax, inv = _axis_slice_params(0, x0)
            polys, eidx = _slice_geometry_indices(nodes, conn, normal, point, u_ax, v_ax)
            title_tmpl  = title_tmpl or f"N-S  x = {x0/1000:.1f} km"
            slice_geom.append(dict(polys=polys, eidx=eidx, invert_v=inv,
                                   invert_x=bool(spec.get("invert_x", False)),
                                   xlabel="y (northing) [m]", ylabel="depth [m]",
                                   xlim=_ylim, ylim=None, zlim=_zlim, title=title_tmpl))

        elif kind == "ew":
            y0 = float(spec.get("y0", 0.0))
            normal, point, u_ax, v_ax, inv = _axis_slice_params(1, y0)
            polys, eidx = _slice_geometry_indices(nodes, conn, normal, point, u_ax, v_ax)
            title_tmpl  = title_tmpl or f"E-W  y = {y0/1000:.1f} km"
            slice_geom.append(dict(polys=polys, eidx=eidx, invert_v=inv,
                                   invert_x=bool(spec.get("invert_x", False)),
                                   xlabel="x (easting) [m]", ylabel="depth [m]",
                                   xlim=_xlim, ylim=None, zlim=_zlim, title=title_tmpl))

        elif kind == "plane":
            _pt     = np.asarray(spec.get("point",  [0., 0., 0.]), dtype=float)
            _strike = float(spec.get("strike", 0.0))
            _dip    = float(spec.get("dip",    90.0))
            normal  = _strike_dip_to_normal(_strike, _dip)
            u_ax, v_ax = _plane_basis(normal)
            polys, eidx = _slice_geometry_indices(nodes, conn, normal, _pt, u_ax, v_ax)
            title_tmpl  = title_tmpl or f"Plane  str={_strike:.0f} deg  dip={_dip:.0f} deg"
            slice_geom.append(dict(polys=polys, eidx=eidx, invert_v=True,
                                   invert_x=bool(spec.get("invert_x", False)),
                                   xlabel="along-strike [m]", ylabel="down-dip [m]",
                                   xlim=_xlim, ylim=_ylim, zlim=None, title=title_tmpl))
        else:
            if out:
                print(f"  ensemble: unknown slice kind {kind!r} -- skipped.")
            slice_geom.append(None)

    # -- load all member resistivities -----------------------------------------
    if out:
        print(f"  ensemble: loading {n_members} member(s) ...")
    n_elem    = conn.shape[0]
    log_stack = np.full((n_members, n_elem), np.nan, dtype=float)
    rho_plots: list = []

    for i, mf in enumerate(member_files):
        if out:
            print(f"    [{i + 1}/{n_members}] {Path(mf).name}")
        block    = read_resistivity_block(mf)
        rho_elem = map_regions_to_element_rho(block.region_of_elem, block.region_rho)
        rho_plot = prepare_rho_for_plotting(
            rho_elem, air_is_nan=True,
            ocean_value=float(ocean_value),
            region_of_elem=block.region_of_elem,
        )
        rho_plots.append(rho_plot)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_stack[i] = np.where(rho_plot > 0, np.log10(rho_plot), np.nan)

    # -- stat arrays (log10 space) ---------------------------------------------
    stat_arrays: dict = {}
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if "mean"   in stat_rows:
            stat_arrays["mean"]   = np.nanmean(log_stack,   axis=0)
        if "std"    in stat_rows:
            stat_arrays["std"]    = np.nanstd(log_stack,    axis=0, ddof=1)
        if "median" in stat_rows:
            stat_arrays["median"] = np.nanmedian(log_stack, axis=0)

    # -- colormap / normalisation ----------------------------------------------
    def _get_cmap(name):
        obj = (mcm.colormaps[name].copy() if hasattr(mcm, "colormaps")
               else mcm.get_cmap(name).copy())
        obj.set_bad(alpha=0.0)
        return obj

    cmap_obj = _get_cmap(cmap)

    if clim is not None:
        norm = mcolors.Normalize(vmin=float(clim[0]), vmax=float(clim[1]))
    else:
        _all = log_stack[np.isfinite(log_stack)]
        norm = mcolors.Normalize(vmin=float(np.nanmin(_all)),
                                 vmax=float(np.nanmax(_all)))

    std_norm = None
    std_cmap = _get_cmap("cividis") if "std" in stat_arrays else cmap_obj
    if "std" in stat_arrays:
        _sv = stat_arrays["std"]
        _sv = _sv[np.isfinite(_sv)]
        if _sv.size > 0:
            std_norm = mcolors.Normalize(vmin=0.0, vmax=float(np.nanmax(_sv)))

    # -- helper: render one full row -------------------------------------------
    def _render_row(fig_ref, axes_row, log_elem, row_label, *, use_std=False):
        _norm     = std_norm if use_std else norm
        _cmap_obj = std_cmap if use_std else cmap_obj
        _cb_label = "std  log10(rho)" if use_std else "log10(rho / Ohm*m)"

        for col, sg in enumerate(slice_geom):
            ax = axes_row[col]
            if air_bgcolor is not None:
                ax.set_facecolor(air_bgcolor)
            if sg is None:
                ax.set_visible(False)
                continue

            panel_vals = log_elem[sg["eidx"]]
            _plot_panel(
                fig_ref, ax, sg["polys"], panel_vals,
                cmap_obj=_cmap_obj, norm=_norm,
                oc_color=ocean_color, oc_value=ocean_value,
                invert_v=sg["invert_v"],
                row_label=row_label, col=col,
                cb_label=_cb_label, show_ylabel=(col == 0),
            )
            ax.set_xlabel(sg["xlabel"], fontsize=label_fontsize)
            ax.tick_params(labelsize=tick_fontsize)

            _xl = sg["xlim"]; _yl = sg["ylim"]; _zl = sg["zlim"]
            if _xl is not None: ax.set_xlim(_xl)
            if _yl is not None: ax.set_ylim(_yl)
            if _zl is not None: ax.set_ylim([_zl[1], _zl[0]])
            if sg.get("invert_x", False):
                ax.invert_xaxis()

    # -- build joint figure ----------------------------------------------------
    if out:
        print(f"  ensemble: building joint figure "
              f"({n_rows} rows x {n_slices} cols) ...")
    fig, axes = plt.subplots(
        n_rows, n_slices,
        figsize=(4.5 * n_slices, 3.5 * n_rows),
        squeeze=False,
    )
    for col, sg in enumerate(slice_geom):
        if sg is not None:
            axes[0, col].set_title(sg["title"], fontsize=label_fontsize + 2)

    for row, (mf, lbl, rho_plot) in enumerate(zip(member_files, labels, rho_plots)):
        with np.errstate(divide="ignore", invalid="ignore"):
            log_elem = np.where(rho_plot > 0, np.log10(rho_plot), np.nan)
        _render_row(fig, axes[row], log_elem, lbl)

    for s_row, stat_name in enumerate(stat_rows):
        row = n_members + s_row
        arr = stat_arrays.get(stat_name)
        if arr is None:
            for col in range(n_slices):
                axes[row, col].set_visible(False)
            continue
        _render_row(fig, axes[row], arr,
                    stat_name.capitalize(), use_std=(stat_name == "std"))

    fig.suptitle(f"Ensemble  ({n_members} members)", fontsize=label_fontsize + 4)
    fig.tight_layout()

    if plot_file is not None:
        fig.savefig(str(plot_file), dpi=dpi, bbox_inches="tight")
        if out:
            print(f"  ensemble: joint figure saved -> {plot_file}")
    else:
        plt.show()

    # -- per-member figures ----------------------------------------------------
    if per_member_file and plot_file is not None:
        _stem, _ext = os.path.splitext(str(plot_file))
        for i, (mf, lbl, rho_plot) in enumerate(zip(member_files, labels, rho_plots)):
            fig_m, axes_m = plt.subplots(
                1, n_slices,
                figsize=(4.5 * n_slices, 3.5),
                squeeze=False,
            )
            for col, sg in enumerate(slice_geom):
                if sg is not None:
                    axes_m[0, col].set_title(sg["title"], fontsize=label_fontsize + 2)
            with np.errstate(divide="ignore", invalid="ignore"):
                log_elem = np.where(rho_plot > 0, np.log10(rho_plot), np.nan)
            _render_row(fig_m, axes_m[0], log_elem, lbl)
            fig_m.suptitle(
                f"Ensemble member {i}  --  {Path(mf).name}", fontsize=label_fontsize + 3
            )
            fig_m.tight_layout()
            per_path = f"{_stem}_member{i}{_ext}"
            fig_m.savefig(per_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig_m)
            if out:
                print(f"  ensemble: member {i} saved -> {per_path}")

    plt.close(fig)
    if out:
        print("  ensemble: done.")


