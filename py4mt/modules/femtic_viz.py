#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_viz.py

Visualisation utilities for FEMTIC meshes and resistivity models.

This module is designed to work with the NPZ products created by
:mod:`femtic` (for example by a mesh+model export helper). It provides:

- Lightweight Matplotlib map slices of log10-resistivity at a depth window.
- Lightweight Matplotlib vertical "curtain" slices along an arbitrary XY polyline.
- Optional PyVista structured grids for the same curtain slices.

Air / ocean conventions
-----------------------
FEMTIC block files typically encode two special "regions":

- Region/index 0: air
- Region/index 1: ocean

For plotting it is usually desirable to hide air and optionally force the
ocean to a fixed conductive value. Helpers in this module therefore allow:

- air -> NaN (transparent / blank)
- ocean -> either kept as-is or forced to a user value

Coordinate conventions
----------------------
This module assumes the FEMTIC mesh coordinates use **z positive downward**
(depth). That is the convention used in the associated plotting functions
(e.g., a curtain plot uses z increasing downward).

Dependencies
------------
Only NumPy is required for importing this module.

Matplotlib and SciPy are imported lazily inside plotting/interpolation
functions. PyVista is entirely optional; functions that require PyVista
import it lazily and raise a clear ImportError if unavailable.

Author: Volker Rath (DIAS)
Created with the help of ChatGPT (GPT-5 Thinking) on 2025-12-23 (UTC)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:  # pragma: no cover
    import pyvista as pv


Interp3D = Literal["idw", "nearest"]
Interp2D = Literal["idw", "nearest"]
RhoSpace = Literal["log10", "linear"]


# -----------------------------------------------------------------------------
# Small data containers
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class NPZModel:
    """Minimal FEMTIC NPZ container used by plotting and interpolation.

    Attributes
    ----------
    centroids : ndarray, shape (n_cells, 3)
        Element centroids (x, y, z) in model units.
    log10_rho : ndarray, shape (n_cells,)
        Per-cell log10-resistivity. This may include air/ocean cells depending
        on how the NPZ was produced.
    region : ndarray, shape (n_cells,), dtype int
        Per-cell region indices. If absent in the NPZ, this array is filled
        with -1 (unknown), and air/ocean masking cannot be applied reliably.
    """

    centroids: np.ndarray
    log10_rho: np.ndarray
    region: np.ndarray


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------

def load_npz_model(npz_path: str | Path) -> NPZModel:
    """Load the minimum arrays needed for visualisation from a FEMTIC NPZ.

    Parameters
    ----------
    npz_path : str or pathlib.Path
        Path to an NPZ created from a FEMTIC mesh/model export. The NPZ must
        contain at least:
            - ``centroid`` or ``centroids``: (n_cells, 3)
            - ``log10_rho`` or ``log10_resistivity``: (n_cells,)

        If available, a region array is also read from one of:
            - ``region``, ``regions``, ``reg``, ``elem_region``

    Returns
    -------
    model : NPZModel
        Loaded model container.

    Raises
    ------
    FileNotFoundError
        If the NPZ file does not exist.
    KeyError
        If required arrays are missing.

    Notes
    -----
    This function intentionally keeps the contract small to remain compatible
    across NPZ flavours. If you want a full PyVista grid from an NPZ, use
    :func:`femtic.npz_to_unstructured_grid`.
    """
    npz_path = Path(npz_path)
    if not npz_path.is_file():
        raise FileNotFoundError(str(npz_path))

    with np.load(npz_path) as d:
        keys = set(d.files)

        # centroids
        if "centroid" in keys:
            centroids = d["centroid"]
        elif "centroids" in keys:
            centroids = d["centroids"]
        else:
            raise KeyError("NPZ must contain 'centroid' or 'centroids' array.")

        # log10 rho
        if "log10_rho" in keys:
            log10_rho = d["log10_rho"]
        elif "log10_resistivity" in keys:
            log10_rho = d["log10_resistivity"]
        elif "log10_res" in keys:
            log10_rho = d["log10_res"]
        else:
            raise KeyError(
                "NPZ must contain 'log10_rho' or 'log10_resistivity' array."
            )

        # region (optional)
        region_keys = ["region", "regions", "reg", "elem_region", "element_region"]
        region = None
        for rk in region_keys:
            if rk in keys:
                region = d[rk]
                break
        if region is None:
            region = -np.ones(len(log10_rho), dtype=int)

    centroids = np.asarray(centroids, dtype=float)
    log10_rho = np.asarray(log10_rho, dtype=float).reshape(-1)
    region = np.asarray(region, dtype=int).reshape(-1)

    if centroids.ndim != 2 or centroids.shape[1] != 3:
        raise ValueError(f"centroids must have shape (n_cells,3); got {centroids.shape}.")
    if log10_rho.shape[0] != centroids.shape[0]:
        raise ValueError("log10_rho length does not match centroids.")
    if region.shape[0] != centroids.shape[0]:
        raise ValueError("region length does not match centroids.")

    return NPZModel(centroids=centroids, log10_rho=log10_rho, region=region)


# -----------------------------------------------------------------------------
# Air / ocean handling
# -----------------------------------------------------------------------------

def prepare_log10_rho_for_plotting(
    log10_rho: ArrayLike,
    region: ArrayLike | None = None,
    *,
    mask_air: bool = True,
    ocean_log10_rho: float | None = None,
) -> np.ndarray:
    """Prepare per-cell log10-resistivity for plotting.

    Parameters
    ----------
    log10_rho : array_like, shape (n_cells,)
        Per-cell log10-resistivity.
    region : array_like or None, shape (n_cells,), optional
        Region index per cell. If provided and ``mask_air=True``, region 0
        values are set to NaN. If ``ocean_log10_rho`` is provided, region 1 is
        set to that value.
    mask_air : bool, optional
        If True (default), set air (region 0) to NaN. If region is None or
        contains only -1 (unknown), no masking is applied.
    ocean_log10_rho : float or None, optional
        If not None, force ocean (region 1) to this log10-resistivity value,
        e.g. ``np.log10(1e-10) = -10``. If None, leave ocean unchanged.

    Returns
    -------
    out : ndarray, shape (n_cells,)
        Prepared array for plotting.
    """
    out = np.asarray(log10_rho, dtype=float).reshape(-1).copy()
    if region is None:
        return out

    reg = np.asarray(region, dtype=int).reshape(-1)
    if reg.shape[0] != out.shape[0]:
        raise ValueError("region length does not match log10_rho.")

    # Unknown region encoding: do nothing
    if np.all(reg < 0):
        return out

    if mask_air:
        out[reg == 0] = np.nan

    if ocean_log10_rho is not None:
        out[reg == 1] = float(ocean_log10_rho)

    return out


def log10_to_rho(log10_rho: ArrayLike) -> np.ndarray:
    """Convert log10-resistivity to linear resistivity.

    Parameters
    ----------
    log10_rho : array_like
        log10-resistivity values.

    Returns
    -------
    rho : ndarray
        Linear resistivity (Ohm·m), computed as ``10**log10_rho``.
    """
    return np.power(10.0, np.asarray(log10_rho, dtype=float))


# -----------------------------------------------------------------------------
# Geometry helpers (polyline sampling)
# -----------------------------------------------------------------------------

def sample_polyline_xy(
    polyline_xy: ArrayLike,
    *,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a polyline in XY into equally spaced arc-length positions.

    Parameters
    ----------
    polyline_xy : array_like, shape (n_vertices, 2)
        Polyline vertices as (x, y).
    n_samples : int
        Number of equally spaced samples along the polyline.

    Returns
    -------
    xy : ndarray, shape (n_samples, 2)
        Sampled XY points along the polyline.
    s : ndarray, shape (n_samples,)
        Cumulative distance (arc length) from the first vertex to each sample.

    Notes
    -----
    The returned ``s`` can be used as the horizontal axis for curtain plots.
    """
    pts = np.asarray(polyline_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"polyline_xy must have shape (n,2); got {pts.shape}.")
    if pts.shape[0] < 2:
        raise ValueError("polyline_xy must contain at least 2 vertices.")
    if n_samples < 2:
        raise ValueError("n_samples must be >= 2.")

    seg = pts[1:] - pts[:-1]
    seglen = np.sqrt(np.sum(seg * seg, axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    total = float(cum[-1])
    if total <= 0.0:
        raise ValueError("polyline length is zero.")

    s = np.linspace(0.0, total, n_samples)
    xy = np.empty((n_samples, 2), dtype=float)

    # For each s, find containing segment and interpolate
    j = 0
    for i, si in enumerate(s):
        while j < len(seglen) - 1 and si > cum[j + 1]:
            j += 1
        t = (si - cum[j]) / seglen[j] if seglen[j] > 0 else 0.0
        xy[i] = pts[j] * (1.0 - t) + pts[j + 1] * t

    return xy, s


# -----------------------------------------------------------------------------
# Interpolation (KDTree-based)
# -----------------------------------------------------------------------------

def _require_scipy() -> None:
    """Raise an ImportError with a clear message if SciPy is missing."""
    try:
        import scipy  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "This function requires SciPy (scipy.spatial.cKDTree). "
            "Install SciPy or choose an interpolation method that does not require it."
        ) from exc


def _idw_from_knn(d: np.ndarray, v: np.ndarray, power: float) -> np.ndarray:
    """Compute inverse-distance weighted values from kNN distances.

    Parameters
    ----------
    d : ndarray, shape (..., k)
        Distances to k neighbors. Distances may include zeros.
    v : ndarray, shape (..., k)
        Values at the k neighbors.
    power : float
        Weight exponent. Common choice is 2.

    Returns
    -------
    out : ndarray, shape (...)
        Interpolated values.

    Notes
    -----
    If any distance is exactly zero, the corresponding neighbor value is
    returned (no averaging).
    """
    d = np.asarray(d, dtype=float)
    v = np.asarray(v, dtype=float)

    # If any exact hit: take the first exact-hit value (stable behaviour).
    hit = d == 0.0
    if np.any(hit):
        # Choose first hit along last axis
        idx = np.argmax(hit, axis=-1)
        # fancy take
        out = np.take_along_axis(v, idx[..., None], axis=-1)[..., 0]
        return out

    w = 1.0 / np.power(d, float(power))
    wsum = np.sum(w, axis=-1)
    # avoid divide-by-zero (should not happen if d>0)
    wsum = np.where(wsum > 0.0, wsum, np.nan)
    return np.sum(w * v, axis=-1) / wsum


def interpolate_points_3d(
    xyz_data: np.ndarray,
    values: np.ndarray,
    xyz_query: np.ndarray,
    *,
    method: Interp3D = "idw",
    k: int = 8,
    power: float = 2.0,
) -> np.ndarray:
    """Interpolate scattered 3-D data to query points using kNN.

    Parameters
    ----------
    xyz_data : ndarray, shape (n, 3)
        Data point coordinates.
    values : ndarray, shape (n,)
        Values at data points.
    xyz_query : ndarray, shape (m, 3)
        Query point coordinates.
    method : {"idw", "nearest"}, optional
        Interpolation method.
    k : int, optional
        Number of neighbors used for "idw". Ignored for "nearest".
    power : float, optional
        IDW power exponent.

    Returns
    -------
    out : ndarray, shape (m,)
        Interpolated values at query points.

    Notes
    -----
    Uses SciPy's :class:`scipy.spatial.cKDTree`. For large grids, this is
    significantly faster than per-point brute force.
    """
    _require_scipy()
    from scipy.spatial import cKDTree  # type: ignore

    xyz_data = np.asarray(xyz_data, dtype=float)
    values = np.asarray(values, dtype=float).reshape(-1)
    xyz_query = np.asarray(xyz_query, dtype=float)

    if xyz_data.ndim != 2 or xyz_data.shape[1] != 3:
        raise ValueError("xyz_data must have shape (n, 3).")
    if xyz_query.ndim != 2 or xyz_query.shape[1] != 3:
        raise ValueError("xyz_query must have shape (m, 3).")
    if xyz_data.shape[0] != values.shape[0]:
        raise ValueError("values length must match xyz_data length.")

    tree = cKDTree(xyz_data)

    method = method.lower()
    if method == "nearest":
        d, idx = tree.query(xyz_query, k=1)
        return values[idx]

    if method == "idw":
        kk = int(max(1, k))
        d, idx = tree.query(xyz_query, k=kk)
        # Ensure 2-D (..., k)
        if kk == 1:
            return values[idx]
        v = values[idx]
        return _idw_from_knn(d, v, power=float(power))

    raise ValueError("method must be 'idw' or 'nearest'.")


def interpolate_points_2d(
    xy_data: np.ndarray,
    values: np.ndarray,
    xy_query: np.ndarray,
    *,
    method: Interp2D = "idw",
    k: int = 8,
    power: float = 2.0,
) -> np.ndarray:
    """Interpolate scattered 2-D data to query points using kNN.

    Parameters
    ----------
    xy_data : ndarray, shape (n, 2)
        Data point coordinates.
    values : ndarray, shape (n,)
        Values at data points.
    xy_query : ndarray, shape (m, 2)
        Query point coordinates.
    method : {"idw", "nearest"}, optional
        Interpolation method.
    k : int, optional
        Number of neighbors for "idw". Ignored for "nearest".
    power : float, optional
        IDW power exponent.

    Returns
    -------
    out : ndarray, shape (m,)
        Interpolated values at query points.
    """
    _require_scipy()
    from scipy.spatial import cKDTree  # type: ignore

    xy_data = np.asarray(xy_data, dtype=float)
    values = np.asarray(values, dtype=float).reshape(-1)
    xy_query = np.asarray(xy_query, dtype=float)

    if xy_data.ndim != 2 or xy_data.shape[1] != 2:
        raise ValueError("xy_data must have shape (n, 2).")
    if xy_query.ndim != 2 or xy_query.shape[1] != 2:
        raise ValueError("xy_query must have shape (m, 2).")
    if xy_data.shape[0] != values.shape[0]:
        raise ValueError("values length must match xy_data length.")

    tree = cKDTree(xy_data)

    method = method.lower()
    if method == "nearest":
        d, idx = tree.query(xy_query, k=1)
        return values[idx]

    if method == "idw":
        kk = int(max(1, k))
        d, idx = tree.query(xy_query, k=kk)
        if kk == 1:
            return values[idx]
        v = values[idx]
        return _idw_from_knn(d, v, power=float(power))

    raise ValueError("method must be 'idw' or 'nearest'.")


# -----------------------------------------------------------------------------
# Curtain slices
# -----------------------------------------------------------------------------

def curtain_slice_from_npz(
    npz_path: str | Path,
    *,
    polyline_xy: ArrayLike,
    zmin: float,
    zmax: float,
    nz: int = 201,
    ns: int = 301,
    interp: Interp3D = "idw",
    k: int = 8,
    power: float = 2.0,
    mask_air: bool = True,
    ocean_log10_rho: float | None = None,
    value_space: RhoSpace = "log10",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a vertical curtain slice from a FEMTIC NPZ model.

    Parameters
    ----------
    npz_path : str or pathlib.Path
        Input NPZ containing centroids and log10-resistivity.
    polyline_xy : array_like, shape (n_vertices, 2)
        Curtain trace polyline in the XY plane.
    zmin, zmax : float
        Depth range (z positive downward) sampled uniformly.
    nz : int, optional
        Number of depth samples (default 201).
    ns : int, optional
        Number of samples along the polyline arc length (default 301).
    interp : {"idw", "nearest"}, optional
        Interpolation method in 3-D based on centroid data.
    k : int, optional
        Neighbors for IDW.
    power : float, optional
        IDW power exponent.
    mask_air : bool, optional
        If True, set air region to NaN before interpolation.
    ocean_log10_rho : float or None, optional
        If not None, force ocean region to this value (log10 units).
    value_space : {"log10", "linear"}, optional
        Output value space. If "linear", returns linear resistivity (Ohm·m).

    Returns
    -------
    S : ndarray, shape (ns,)
        Arc-length coordinate along the polyline (same units as XY).
    Z : ndarray, shape (nz,)
        Depth samples (z positive downward).
    V : ndarray, shape (nz, ns)
        Interpolated slice values. Rows correspond to Z, columns to S.

    Notes
    -----
    For performance, interpolation uses a single KDTree built on the centroid
    coordinates and queries all (ns*nz) points at once.
    """
    if nz < 2 or ns < 2:
        raise ValueError("nz and ns must be >= 2.")
    if zmax <= zmin:
        raise ValueError("Require zmax > zmin.")

    model = load_npz_model(npz_path)
    log10_plot = prepare_log10_rho_for_plotting(
        model.log10_rho,
        model.region,
        mask_air=mask_air,
        ocean_log10_rho=ocean_log10_rho,
    )

    xy, S = sample_polyline_xy(polyline_xy, n_samples=ns)
    Z = np.linspace(float(zmin), float(zmax), int(nz))

    # Build full query grid (ns*nz, 3)
    Xq = np.repeat(xy[:, 0][None, :], nz, axis=0)
    Yq = np.repeat(xy[:, 1][None, :], nz, axis=0)
    Zq = np.repeat(Z[:, None], ns, axis=1)
    xyz_query = np.column_stack([Xq.ravel(), Yq.ravel(), Zq.ravel()])

    vals = interpolate_points_3d(
        model.centroids,
        log10_plot,
        xyz_query,
        method=interp,
        k=k,
        power=power,
    ).reshape(nz, ns)

    if value_space == "linear":
        vals = log10_to_rho(vals)

    return S, Z, vals


def plot_curtain_matplotlib(
    S: ArrayLike,
    Z: ArrayLike,
    V: ArrayLike,
    *,
    ax=None,
    title: str | None = None,
    cbar_label: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | None = None,
) -> object:
    """Plot a curtain slice with Matplotlib.

    Parameters
    ----------
    S : array_like, shape (ns,)
        Arc-length coordinates.
    Z : array_like, shape (nz,)
        Depth coordinates (z positive downward).
    V : array_like, shape (nz, ns)
        Slice values (log10 or linear). NaNs are rendered transparent.
    ax : matplotlib Axes or None, optional
        If None, a new figure+axes are created.
    title : str or None, optional
        Plot title.
    cbar_label : str or None, optional
        Colorbar label.
    vmin, vmax : float or None, optional
        Color scale bounds.
    cmap : str or None, optional
        Matplotlib colormap name. If None, Matplotlib default is used.

    Returns
    -------
    ax : matplotlib Axes
        The axes used for plotting.

    Notes
    -----
    The plot uses ``pcolormesh`` with ``shading='auto'``.
    """
    import matplotlib.pyplot as plt  # lazy

    S = np.asarray(S, dtype=float).reshape(-1)
    Z = np.asarray(Z, dtype=float).reshape(-1)
    V = np.asarray(V, dtype=float)

    if V.shape != (Z.size, S.size):
        raise ValueError(f"V must have shape (nz, ns)=({Z.size},{S.size}); got {V.shape}.")

    if ax is None:
        _, ax = plt.subplots()

    mesh = ax.pcolormesh(S, Z, V, shading="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xlabel("Distance along polyline")
    ax.set_ylabel("Depth (z positive down)")
    ax.invert_yaxis()  # visually: depth increases downward on the page
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle=":")

    cbar = plt.colorbar(mesh, ax=ax)
    if cbar_label:
        cbar.set_label(cbar_label)

    return ax


def femtic_curtain_from_npz_matplotlib(
    npz_path: str | Path,
    *,
    polyline_xy: ArrayLike,
    zmin: float,
    zmax: float,
    nz: int = 201,
    ns: int = 301,
    interp: Interp3D = "idw",
    k: int = 8,
    power: float = 2.0,
    mask_air: bool = True,
    ocean_log10_rho: float | None = None,
    value_space: RhoSpace = "log10",
    title: str | None = None,
) -> object:
    """Convenience wrapper: compute and plot a curtain slice from an NPZ.

    Parameters
    ----------
    npz_path : str or pathlib.Path
        Input NPZ model.
    polyline_xy : array_like
        Curtain trace polyline XY vertices.
    zmin, zmax : float
        Depth range.
    nz, ns : int
        Sampling of depth and arc length.
    interp, k, power : see :func:`curtain_slice_from_npz`.
    mask_air, ocean_log10_rho : see :func:`prepare_log10_rho_for_plotting`.
    value_space : {"log10", "linear"}
        Output value space.
    title : str or None
        Plot title.

    Returns
    -------
    ax : matplotlib Axes
        Axes with the plot.
    """
    S, Z, V = curtain_slice_from_npz(
        npz_path,
        polyline_xy=polyline_xy,
        zmin=zmin,
        zmax=zmax,
        nz=nz,
        ns=ns,
        interp=interp,
        k=k,
        power=power,
        mask_air=mask_air,
        ocean_log10_rho=ocean_log10_rho,
        value_space=value_space,
    )
    cbar_label = "log10(resistivity)" if value_space == "log10" else "resistivity (Ohm·m)"
    return plot_curtain_matplotlib(S, Z, V, title=title, cbar_label=cbar_label)


# -----------------------------------------------------------------------------
# Map slices (horizontal)
# -----------------------------------------------------------------------------

def map_slice_from_npz(
    npz_path: str | Path,
    *,
    zmin: float,
    zmax: float,
    nx: int = 301,
    ny: int = 301,
    bounds: tuple[float, float, float, float] | None = None,
    interp: Interp2D = "idw",
    k: int = 8,
    power: float = 2.0,
    mask_air: bool = True,
    ocean_log10_rho: float | None = None,
    value_space: RhoSpace = "log10",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a horizontal map slice by averaging over a depth window.

    Parameters
    ----------
    npz_path : str or pathlib.Path
        Input NPZ model.
    zmin, zmax : float
        Depth window (z positive down). Cells with centroid z in [zmin, zmax]
        are selected.
    nx, ny : int, optional
        Grid size in x and y for the output map (default 301×301).
    bounds : (xmin, xmax, ymin, ymax) or None, optional
        Bounds of the output grid. If None, bounds are derived from the selected
        cell centroids.
    interp : {"idw", "nearest"}, optional
        Interpolation method in 2-D based on XY centroids.
    k : int, optional
        Neighbors for IDW.
    power : float, optional
        IDW power exponent.
    mask_air, ocean_log10_rho : see :func:`prepare_log10_rho_for_plotting`.
    value_space : {"log10", "linear"}, optional
        Output value space.

    Returns
    -------
    X : ndarray, shape (ny, nx)
        X grid (meshgrid output).
    Y : ndarray, shape (ny, nx)
        Y grid.
    V : ndarray, shape (ny, nx)
        Interpolated slice values on the grid.

    Notes
    -----
    Within the selected depth window, this function performs a simple
    per-cell averaging in case multiple cells project to similar XY positions.
    It then interpolates the resulting scattered XY data to a regular grid.
    """
    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must be >= 2.")
    if zmax <= zmin:
        raise ValueError("Require zmax > zmin.")

    model = load_npz_model(npz_path)
    log10_plot = prepare_log10_rho_for_plotting(
        model.log10_rho,
        model.region,
        mask_air=mask_air,
        ocean_log10_rho=ocean_log10_rho,
    )

    z = model.centroids[:, 2]
    sel = (z >= float(zmin)) & (z <= float(zmax)) & np.isfinite(log10_plot)
    if not np.any(sel):
        raise ValueError("No cells found in the requested depth window (after masking).")

    xy = model.centroids[sel, :2]
    vv = log10_plot[sel]

    if bounds is None:
        xmin, ymin = np.min(xy, axis=0)
        xmax, ymax = np.max(xy, axis=0)
    else:
        xmin, xmax, ymin, ymax = map(float, bounds)

    x = np.linspace(xmin, xmax, int(nx))
    y = np.linspace(ymin, ymax, int(ny))
    X, Y = np.meshgrid(x, y)

    vq = interpolate_points_2d(
        xy,
        vv,
        np.column_stack([X.ravel(), Y.ravel()]),
        method=interp,
        k=k,
        power=power,
    ).reshape(ny, nx)

    if value_space == "linear":
        vq = log10_to_rho(vq)

    return X, Y, vq


def plot_map_slice_matplotlib(
    X: ArrayLike,
    Y: ArrayLike,
    V: ArrayLike,
    *,
    ax=None,
    title: str | None = None,
    cbar_label: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | None = None,
    equal_aspect: bool = True,
) -> object:
    """Plot a map slice on an XY grid with Matplotlib.

    Parameters
    ----------
    X, Y : array_like, shape (ny, nx)
        Grid coordinates (as returned by :func:`map_slice_from_npz`).
    V : array_like, shape (ny, nx)
        Slice values.
    ax : matplotlib Axes or None, optional
        If None, creates a new figure+axes.
    title : str or None
        Plot title.
    cbar_label : str or None
        Colorbar label.
    vmin, vmax : float or None
        Color scale bounds.
    cmap : str or None
        Matplotlib colormap name.
    equal_aspect : bool, optional
        If True (default), set equal XY aspect ratio.

    Returns
    -------
    ax : matplotlib Axes
        Axes used for plotting.
    """
    import matplotlib.pyplot as plt  # lazy

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    V = np.asarray(V, dtype=float)
    if X.shape != Y.shape or X.shape != V.shape:
        raise ValueError("X, Y, and V must have identical shapes (ny, nx).")

    if ax is None:
        _, ax = plt.subplots()

    mesh = ax.pcolormesh(X, Y, V, shading="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle=":")

    cbar = plt.colorbar(mesh, ax=ax)
    if cbar_label:
        cbar.set_label(cbar_label)

    return ax


def femtic_map_slice_from_npz_matplotlib(
    npz_path: str | Path,
    *,
    zmin: float,
    zmax: float,
    nx: int = 301,
    ny: int = 301,
    bounds: tuple[float, float, float, float] | None = None,
    interp: Interp2D = "idw",
    k: int = 8,
    power: float = 2.0,
    mask_air: bool = True,
    ocean_log10_rho: float | None = None,
    value_space: RhoSpace = "log10",
    title: str | None = None,
) -> object:
    """Convenience wrapper: compute and plot a map slice from an NPZ model."""
    X, Y, V = map_slice_from_npz(
        npz_path,
        zmin=zmin,
        zmax=zmax,
        nx=nx,
        ny=ny,
        bounds=bounds,
        interp=interp,
        k=k,
        power=power,
        mask_air=mask_air,
        ocean_log10_rho=ocean_log10_rho,
        value_space=value_space,
    )
    cbar_label = "log10(resistivity)" if value_space == "log10" else "resistivity (Ohm·m)"
    return plot_map_slice_matplotlib(X, Y, V, title=title, cbar_label=cbar_label)


# -----------------------------------------------------------------------------
# Optional PyVista curtain grids
# -----------------------------------------------------------------------------

def build_curtain_structured_grid_from_npz(
    npz_path: str | Path,
    *,
    polyline_xy: ArrayLike,
    zmin: float,
    zmax: float,
    nz: int = 201,
    ns: int = 301,
    interp: Interp3D = "idw",
    k: int = 8,
    power: float = 2.0,
    mask_air: bool = True,
    ocean_log10_rho: float | None = None,
    value_space: RhoSpace = "log10",
    name: str = "value",
) -> "pv.StructuredGrid":
    """Build a PyVista StructuredGrid for a curtain slice from an NPZ model.

    Parameters
    ----------
    npz_path : str or pathlib.Path
        Input NPZ model.
    polyline_xy, zmin, zmax, nz, ns, interp, k, power :
        Same meaning as in :func:`curtain_slice_from_npz`.
    mask_air, ocean_log10_rho :
        Same meaning as in :func:`prepare_log10_rho_for_plotting`.
    value_space : {"log10", "linear"}
        Value space of the grid point data.
    name : str, optional
        Name of the point-data array attached to the grid (default "value").

    Returns
    -------
    grid : pyvista.StructuredGrid
        Structured grid with point data named ``name``.

    Raises
    ------
    ImportError
        If PyVista is not installed.

    Notes
    -----
    The grid has dimensions (ns, nz, 1). Values are stored as point data
    consistent with the point coordinates.
    """
    try:
        import pyvista as pv  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("PyVista is required for build_curtain_structured_grid_from_npz.") from exc

    S, Z, V = curtain_slice_from_npz(
        npz_path,
        polyline_xy=polyline_xy,
        zmin=zmin,
        zmax=zmax,
        nz=nz,
        ns=ns,
        interp=interp,
        k=k,
        power=power,
        mask_air=mask_air,
        ocean_log10_rho=ocean_log10_rho,
        value_space=value_space,
    )

    # Build coordinates arrays for StructuredGrid: shape (ns, nz, 1)
    xy, _ = sample_polyline_xy(polyline_xy, n_samples=ns)
    x = np.repeat(xy[:, 0][:, None], nz, axis=1)[:, :, None]
    y = np.repeat(xy[:, 1][:, None], nz, axis=1)[:, :, None]
    z = np.repeat(Z[None, :], ns, axis=0)[:, :, None]

    grid = pv.StructuredGrid(x, y, z)
    # V is (nz, ns) -> point ordering expects (ns, nz, 1) flattened in Fortran order.
    V_p = np.asarray(V, dtype=float).T[:, :, None]  # (ns, nz, 1)
    grid.point_data[name] = V_p.ravel(order="F")
    grid.point_data["s"] = np.repeat(S[:, None], nz, axis=1).ravel(order="F")
    grid.point_data["z"] = z.ravel(order="F")
    return grid


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_xy_pairs(pairs: Sequence[Sequence[float]]) -> np.ndarray:
    """Parse --xy arguments into a (n,2) array."""
    pts = np.asarray(pairs, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("Need at least two --xy pairs, each with two floats.")
    if pts.shape[0] < 2:
        raise ValueError("Need at least two --xy pairs.")
    return pts


# -----------------------------------------------------------------------------
# Native PyVista mesh visualisation (no regular-grid interpolation)
# -----------------------------------------------------------------------------


def native_grid_from_npz(npz_path: str):
    """Build a PyVista :class:`pyvista.UnstructuredGrid` from a FEMTIC NPZ file.

    This function prefers the implementation in :mod:`femtic` if it is importable,
    because that keeps the NPZ-to-grid logic in one place. If :mod:`femtic` is not
    available, a small fallback implementation is used.

    Parameters
    ----------
    npz_path : str
        Path to an NPZ created by FEMTIC (must contain at least ``nodes`` and ``conn``).

    Returns
    -------
    grid : pyvista.UnstructuredGrid
        Unstructured tetrahedral grid with per-cell arrays attached as ``cell_data``
        where possible.

    Notes
    -----
    This path performs **no interpolation**. All slicing/clipping is done on the
    native unstructured tetrahedral mesh.

    Raises
    ------
    ImportError
        If PyVista is not installed.
    KeyError
        If required NPZ keys are missing.
    """
    # Prefer femtic's implementation if present.
    try:
        import femtic  # type: ignore

        return femtic.npz_to_unstructured_grid(npz_path)
    except Exception:
        pass

    try:
        import numpy as np
        import pyvista as pv
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "PyVista (and NumPy) are required for native mesh visualisation."
        ) from e

    data = np.load(npz_path)
    if "nodes" not in data or "conn" not in data:
        raise KeyError("NPZ must contain 'nodes' and 'conn' arrays for native plotting.")

    nodes = np.asarray(data["nodes"], dtype=float)
    conn = np.asarray(data["conn"], dtype=np.int64)
    nelem = int(conn.shape[0])

    if conn.ndim != 2 or conn.shape[1] != 4:
        raise ValueError("Connectivity 'conn' must have shape (nelem, 4) for tetrahedra.")

    cells = np.hstack([np.full((nelem, 1), 4, dtype=np.int64), conn]).ravel()

    try:
        celltypes = np.full(nelem, pv.CellType.TETRA, dtype=np.uint8)  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        celltypes = np.full(nelem, 10, dtype=np.uint8)  # VTK_TETRA = 10

    grid = pv.UnstructuredGrid(cells, celltypes, nodes)

    # Attach any 1-D arrays of length nelem as cell_data.
    for key in getattr(data, "files", []):
        if key in ("nodes", "conn"):
            continue
        arr = np.asarray(data[key])
        if arr.ndim == 1 and arr.shape[0] == nelem:
            name = "region" if key == "region_of_elem" else key
            grid.cell_data[name] = arr

    return grid


def _apply_air_ocean_on_cell_scalar(
    grid,
    *,
    scalar_name: str,
    mask_air: bool = True,
    ocean_log10_rho: float | None = None,
):
    """Apply FEMTIC air/ocean conventions on a grid's scalar array (in-place).

    Parameters
    ----------
    grid : pyvista.UnstructuredGrid
        Grid whose ``cell_data`` contains the scalar.
    scalar_name : str
        Name of the scalar in ``grid.cell_data``.
    mask_air : bool, optional
        If True (default), air cells (region == 0) are removed by extraction.
        If False, air is kept.
    ocean_log10_rho : float or None, optional
        If not None, sets the scalar value of ocean cells (region == 1) to this value.

    Returns
    -------
    grid2 : pyvista.UnstructuredGrid
        Possibly extracted grid. If no extraction is required, returns the input grid.

    Notes
    -----
    This helper assumes FEMTIC's common region convention:

    - region 0: air
    - region 1: ocean

    If the grid does not have a ``region`` cell array, no masking/forcing is applied.
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return grid

    if scalar_name not in grid.cell_data:
        raise KeyError(f"Scalar '{scalar_name}' not found in grid.cell_data.")

    if "region" not in grid.cell_data:
        return grid

    reg = np.asarray(grid.cell_data["region"])
    scal = np.asarray(grid.cell_data[scalar_name])

    if ocean_log10_rho is not None:
        ocean_mask = reg == 1
        if ocean_mask.any():
            scal2 = scal.copy()
            scal2[ocean_mask] = float(ocean_log10_rho)
            grid.cell_data[scalar_name] = scal2

    if mask_air:
        keep = np.where(reg != 0)[0]
        return grid.extract_cells(keep)

    return grid


def native_extract_depth_window(
    npz_path: str,
    *,
    zmin: float,
    zmax: float,
    scalar_name: str = "log10_resistivity",
    mask_air: bool = True,
    ocean_log10_rho: float | None = None,
):
    """Extract cells whose centroid depth lies in a window (native mesh, no interpolation).

    Parameters
    ----------
    npz_path : str
        FEMTIC NPZ path.
    zmin, zmax : float
        Depth window in model coordinates (z positive down).
    scalar_name : str, optional
        Scalar field name in cell_data for plotting (default: ``log10_resistivity``).
    mask_air : bool, optional
        If True (default), removes air cells (region 0) if region information exists.
    ocean_log10_rho : float or None, optional
        If given, forces ocean cells (region 1) to this scalar value (log10 space).

    Returns
    -------
    sub : pyvista.UnstructuredGrid
        Extracted subgrid containing only cells in the depth window.

    Notes
    -----
    This is the most robust way to obtain a "map slice" on an unstructured mesh
    without resampling: extract a depth slab and render it from above.
    """
    try:
        import numpy as np
    except ImportError as e:  # pragma: no cover
        raise ImportError("NumPy is required.") from e

    grid = native_grid_from_npz(npz_path)
    grid = _apply_air_ocean_on_cell_scalar(
        grid, scalar_name=scalar_name, mask_air=mask_air, ocean_log10_rho=ocean_log10_rho
    )

    centers = np.asarray(grid.cell_centers().points)
    idx = np.where((centers[:, 2] >= float(zmin)) & (centers[:, 2] <= float(zmax)))[0]
    return grid.extract_cells(idx)


def native_plane_slice(
    npz_path: str,
    *,
    z0: float,
    scalar_name: str = "log10_resistivity",
    mask_air: bool = True,
    ocean_log10_rho: float | None = None,
):
    """Create a horizontal plane slice at depth ``z0`` (native mesh, no interpolation).

    Parameters
    ----------
    npz_path : str
        FEMTIC NPZ path.
    z0 : float
        Depth of the horizontal slicing plane (z positive down).
    scalar_name : str, optional
        Scalar in cell_data to visualise (default: ``log10_resistivity``).
    mask_air : bool, optional
        If True (default), removes air cells (region 0) if region info exists.
    ocean_log10_rho : float or None, optional
        If given, forces ocean cells (region 1) to this scalar value (log10 space).

    Returns
    -------
    sl : pyvista.PolyData
        Sliced surface polydata.
    """
    grid = native_grid_from_npz(npz_path)
    grid = _apply_air_ocean_on_cell_scalar(
        grid, scalar_name=scalar_name, mask_air=mask_air, ocean_log10_rho=ocean_log10_rho
    )
    sl = grid.slice(normal=(0.0, 0.0, 1.0), origin=(0.0, 0.0, float(z0)))
    return sl


def project_points_to_polyline_xy(
    xy: np.ndarray,
    polyline_xy: np.ndarray,
):
    """Project XY points onto a polyline and compute arc-length coordinate.

    Parameters
    ----------
    xy : ndarray, shape (n, 2)
        XY points to project.
    polyline_xy : ndarray, shape (m, 2)
        Polyline vertices.

    Returns
    -------
    s : ndarray, shape (n,)
        Arc-length coordinate along the polyline of the closest projection.
    d : ndarray, shape (n,)
        Minimum Euclidean distance from each point to the polyline.
    """
    import numpy as np

    pts = np.asarray(xy, dtype=float)
    poly = np.asarray(polyline_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("xy must be (n,2).")
    if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 2:
        raise ValueError("polyline_xy must be (m,2) with m>=2.")

    seg = poly[1:] - poly[:-1]
    seglen = np.sqrt(np.sum(seg**2, axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    # Avoid division by zero for degenerate segments
    seglen2 = np.where(seglen > 0.0, seglen**2, 1.0)

    s_best = np.zeros(pts.shape[0], dtype=float)
    d_best2 = np.full(pts.shape[0], np.inf, dtype=float)

    for i in range(seg.shape[0]):
        a = poly[i]
        v = seg[i]
        # t = ((p-a)·v)/(v·v) clamped
        w = pts - a
        t = (w[:, 0] * v[0] + w[:, 1] * v[1]) / seglen2[i]
        t = np.clip(t, 0.0, 1.0)
        proj = a + t[:, None] * v[None, :]
        d2 = np.sum((pts - proj) ** 2, axis=1)

        upd = d2 < d_best2
        if np.any(upd):
            d_best2[upd] = d2[upd]
            s_best[upd] = cum[i] + t[upd] * seglen[i]

    return s_best, np.sqrt(d_best2)


def native_curtain_slices(
    npz_path: str,
    *,
    polyline_xy: ArrayLike,
    zmin: float,
    zmax: float,
    ns: int = 51,
    corridor: float | None = None,
    scalar_name: str = "log10_resistivity",
    mask_air: bool = True,
    ocean_log10_rho: float | None = None,
):
    """Create a MultiBlock of vertical plane slices along a polyline (no interpolation).

    This generates a series of vertical planes approximately perpendicular to the
    polyline direction and slices the **native** unstructured mesh with those planes.
    It is intended for interactive PyVista viewing.

    Parameters
    ----------
    npz_path : str
        FEMTIC NPZ path.
    polyline_xy : array_like, shape (m, 2)
        Polyline vertices in XY.
    zmin, zmax : float
        Depth range to clip the slices to (z positive down).
    ns : int, optional
        Number of slicing planes along the polyline (default: 51).
    corridor : float or None, optional
        If given, first extracts only those cells whose centroid is within this XY
        distance from the polyline. This keeps slices local and fast.
    scalar_name : str, optional
        Cell scalar to visualise (default: ``log10_resistivity``).
    mask_air : bool, optional
        If True (default), removes air cells (region 0) if region info exists.
    ocean_log10_rho : float or None, optional
        If given, forces ocean cells (region 1) to this scalar value (log10 space).

    Returns
    -------
    blocks : pyvista.MultiBlock
        MultiBlock containing one PolyData slice per plane.
    s : ndarray, shape (ns,)
        Arc-length coordinate of each slicing plane along the polyline.

    Notes
    -----
    This is not a "curtain grid". It is a set of true geometric slices through
    the tetrahedral mesh. You may merge blocks (`blocks.combine()`) or inspect
    them individually.
    """
    try:
        import numpy as np
        import pyvista as pv
    except ImportError as e:  # pragma: no cover
        raise ImportError("PyVista and NumPy are required for native curtain slicing.") from e

    poly = np.asarray(polyline_xy, dtype=float)
    xy_samp, s = sample_polyline_xy(poly, n_samples=int(ns))

    grid = native_grid_from_npz(npz_path)
    grid = _apply_air_ocean_on_cell_scalar(
        grid, scalar_name=scalar_name, mask_air=mask_air, ocean_log10_rho=ocean_log10_rho
    )

    if corridor is not None and corridor > 0.0:
        centers = np.asarray(grid.cell_centers().points)
        _, d = project_points_to_polyline_xy(centers[:, :2], poly)
        keep = np.where(d <= float(corridor))[0]
        grid = grid.extract_cells(keep)

    # Depth clipping bounds for clip_box: (xmin, xmax, ymin, ymax, zmin, zmax)
    b = grid.bounds
    clip_bounds = (b[0], b[1], b[2], b[3], float(zmin), float(zmax))

    z0 = 0.5 * (float(zmin) + float(zmax))
    blocks = pv.MultiBlock()

    for i in range(xy_samp.shape[0]):
        # Tangent from neighbors (central difference)
        i0 = max(i - 1, 0)
        i1 = min(i + 1, xy_samp.shape[0] - 1)
        t = xy_samp[i1] - xy_samp[i0]
        nt = float(np.hypot(t[0], t[1]))
        if nt <= 0.0:
            continue

        # Normal is perpendicular to tangent in XY (vertical plane).
        normal = (t[1] / nt, -t[0] / nt, 0.0)
        origin = (float(xy_samp[i, 0]), float(xy_samp[i, 1]), z0)

        sl = grid.slice(normal=normal, origin=origin)
        try:
            sl = sl.clip_box(bounds=clip_bounds, invert=False)
        except Exception:
            # If clip_box is not available for some PyVista/VTK combos, keep un-clipped slice.
            pass

        blocks[f"slice_{i:04d}"] = sl

    return blocks, s


def pv_plot_dataset(
    dataset,
    *,
    scalar_name: str = "log10_resistivity",
    view_xy: bool = False,
    show_edges: bool = False,
):
    """Convenience plotter for PyVista datasets (interactive).

    Parameters
    ----------
    dataset : pyvista.DataSet or pyvista.MultiBlock
        Dataset to plot.
    scalar_name : str, optional
        Scalar name to plot (default: ``log10_resistivity``).
    view_xy : bool, optional
        If True, sets a top-down XY view.
    show_edges : bool, optional
        If True, draws mesh edges.

    Returns
    -------
    None
        This function calls :meth:`pyvista.DataSet.plot`.
    """
    try:
        import pyvista as pv
    except ImportError as e:  # pragma: no cover
        raise ImportError("PyVista is required for pv_plot_dataset.") from e

    kwargs = {"scalars": scalar_name, "show_edges": bool(show_edges)}
    if view_xy:
        kwargs["viewup"] = (0.0, 1.0, 0.0)

    # Many PyVista objects support .plot directly (DataSet and MultiBlock).
    dataset.plot(**kwargs)



def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    This CLI is intentionally small and focuses on quick-look plots.

    Subcommands
    -----------
    curtain
        Curtain slice along a polyline to Matplotlib (regular-grid interpolation).
    map
        Horizontal map slice (depth window) to Matplotlib (regular-grid interpolation).
    pv-plane
        Native PyVista horizontal plane slice at depth z0 (no interpolation).
    pv-window
        Native PyVista depth-window extraction rendered from above (no interpolation).
    pv-curtain
        Native PyVista multi-slice curtain along a polyline (no interpolation).

    Parameters
    ----------
    argv : sequence of str or None, optional
        Argument vector without the program name. If None, uses sys.argv[1:].

    Returns
    -------
    code : int
        Exit status code (0 = success).
    """
    import argparse
    import sys

    p = argparse.ArgumentParser(prog="femtic_viz", description="FEMTIC visualisation helpers.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("curtain", help="Curtain slice from NPZ (Matplotlib).")
    pc.add_argument("--npz", required=True, help="NPZ produced from FEMTIC mesh/model.")
    pc.add_argument("--zmin", type=float, required=True, help="Minimum depth (z positive down).")
    pc.add_argument("--zmax", type=float, required=True, help="Maximum depth (z positive down).")
    pc.add_argument("--nz", type=int, default=201, help="Depth samples.")
    pc.add_argument("--ns", type=int, default=301, help="Samples along polyline.")
    pc.add_argument("--interp", choices=["idw", "nearest"], default="idw", help="Interpolation method.")
    pc.add_argument("--k", type=int, default=8, help="k neighbors for IDW.")
    pc.add_argument("--power", type=float, default=2.0, help="IDW power exponent.")
    pc.add_argument("--ocean-log10", type=float, default=None, help="Force ocean to this log10 resistivity.")
    pc.add_argument("--no-mask-air", action="store_true", help="Do not mask air region to NaN.")
    pc.add_argument("--linear", action="store_true", help="Output linear resistivity (Ohm·m).")
    pc.add_argument("--title", default=None, help="Plot title.")
    pc.add_argument(
        "--xy",
        action="append",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help="Polyline vertex (X Y). Repeat to build polyline.",
        required=True,
    )

    pm = sub.add_parser("map", help="Map slice from NPZ (Matplotlib).")
    pm.add_argument("--npz", required=True, help="NPZ produced from FEMTIC mesh/model.")
    pm.add_argument("--zmin", type=float, required=True, help="Minimum depth (z positive down).")
    pm.add_argument("--zmax", type=float, required=True, help="Maximum depth (z positive down).")
    pm.add_argument("--nx", type=int, default=301, help="Grid samples in X.")
    pm.add_argument("--ny", type=int, default=301, help="Grid samples in Y.")
    pm.add_argument("--interp", choices=["idw", "nearest"], default="idw", help="Interpolation method.")
    pm.add_argument("--k", type=int, default=8, help="k neighbors for IDW.")
    pm.add_argument("--power", type=float, default=2.0, help="IDW power exponent.")
    pm.add_argument("--ocean-log10", type=float, default=None, help="Force ocean to this log10 resistivity.")
    pm.add_argument("--no-mask-air", action="store_true", help="Do not mask air region to NaN.")
    pm.add_argument("--linear", action="store_true", help="Output linear resistivity (Ohm·m).")
    pm.add_argument("--title", default=None, help="Plot title.")
    pm.add_argument("--bounds", nargs=4, type=float, metavar=("XMIN", "XMAX", "YMIN", "YMAX"), default=None)


    # Native PyVista (no-interpolation) subcommands
    pp = sub.add_parser("pv-plane", help="Native PyVista plane slice at depth z0 (no interpolation).")
    pp.add_argument("--npz", required=True, help="NPZ produced from FEMTIC mesh/model.")
    pp.add_argument("--z0", type=float, required=True, help="Slice depth (z positive down).")
    pp.add_argument("--scalar", default="log10_resistivity", help="Cell scalar name to plot.")
    pp.add_argument("--ocean-log10", type=float, default=None, help="Force ocean to this log10 resistivity.")
    pp.add_argument("--no-mask-air", action="store_true", help="Do not remove air cells (region 0).")
    pp.add_argument("--edges", action="store_true", help="Show mesh edges in PyVista view.")

    pw = sub.add_parser("pv-window", help="Native PyVista depth-window extraction (no interpolation).")
    pw.add_argument("--npz", required=True, help="NPZ produced from FEMTIC mesh/model.")
    pw.add_argument("--zmin", type=float, required=True, help="Minimum depth (z positive down).")
    pw.add_argument("--zmax", type=float, required=True, help="Maximum depth (z positive down).")
    pw.add_argument("--scalar", default="log10_resistivity", help="Cell scalar name to plot.")
    pw.add_argument("--ocean-log10", type=float, default=None, help="Force ocean to this log10 resistivity.")
    pw.add_argument("--no-mask-air", action="store_true", help="Do not remove air cells (region 0).")
    pw.add_argument("--no-view-xy", action="store_true", help="Do not force top-down XY view.")
    pw.add_argument("--edges", action="store_true", help="Show mesh edges in PyVista view.")

    pcv = sub.add_parser("pv-curtain", help="Native PyVista curtain as a set of slices (no interpolation).")
    pcv.add_argument("--npz", required=True, help="NPZ produced from FEMTIC mesh/model.")
    pcv.add_argument("--zmin", type=float, required=True, help="Minimum depth (z positive down).")
    pcv.add_argument("--zmax", type=float, required=True, help="Maximum depth (z positive down).")
    pcv.add_argument("--ns", type=int, default=51, help="Number of slicing planes along the polyline.")
    pcv.add_argument("--corridor", type=float, default=None, help="Optional corridor half-width (XY) in model units.")
    pcv.add_argument("--scalar", default="log10_resistivity", help="Cell scalar name to plot.")
    pcv.add_argument("--ocean-log10", type=float, default=None, help="Force ocean to this log10 resistivity.")
    pcv.add_argument("--no-mask-air", action="store_true", help="Do not remove air cells (region 0).")
    pcv.add_argument("--edges", action="store_true", help="Show mesh edges in PyVista view.")
    pcv.add_argument(
        "--xy",
        action="append",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help="Polyline vertex (X Y). Repeat to build polyline.",
        required=True,
    )

    args = p.parse_args(list(argv) if argv is not None else None)

    if args.cmd == "curtain":
        poly = _parse_xy_pairs(args.xy)
        value_space: RhoSpace = "linear" if args.linear else "log10"
        femtic_curtain_from_npz_matplotlib(
            args.npz,
            polyline_xy=poly,
            zmin=args.zmin,
            zmax=args.zmax,
            nz=args.nz,
            ns=args.ns,
            interp=args.interp,
            k=args.k,
            power=args.power,
            mask_air=not args.no_mask_air,
            ocean_log10_rho=args.ocean_log10,
            value_space=value_space,
            title=args.title,
        )
        import matplotlib.pyplot as plt
        plt.show()
        return 0

    if args.cmd == "map":
        value_space = "linear" if args.linear else "log10"
        femtic_map_slice_from_npz_matplotlib(
            args.npz,
            zmin=args.zmin,
            zmax=args.zmax,
            nx=args.nx,
            ny=args.ny,
            bounds=None if args.bounds is None else tuple(args.bounds),
            interp=args.interp,
            k=args.k,
            power=args.power,
            mask_air=not args.no_mask_air,
            ocean_log10_rho=args.ocean_log10,
            value_space=value_space,
            title=args.title,
        )
        import matplotlib.pyplot as plt
        plt.show()
        return 0

    
    if args.cmd == "pv-plane":
        sl = native_plane_slice(
            args.npz,
            z0=args.z0,
            scalar_name=args.scalar,
            mask_air=not args.no_mask_air,
            ocean_log10_rho=args.ocean_log10,
        )
        pv_plot_dataset(sl, scalar_name=args.scalar, view_xy=False, show_edges=args.edges)
        return 0

    if args.cmd == "pv-window":
        subg = native_extract_depth_window(
            args.npz,
            zmin=args.zmin,
            zmax=args.zmax,
            scalar_name=args.scalar,
            mask_air=not args.no_mask_air,
            ocean_log10_rho=args.ocean_log10,
        )
        pv_plot_dataset(
            subg,
            scalar_name=args.scalar,
            view_xy=not args.no_view_xy,
            show_edges=args.edges,
        )
        return 0

    if args.cmd == "pv-curtain":
        poly = _parse_xy_pairs(args.xy)
        blocks, _s = native_curtain_slices(
            args.npz,
            polyline_xy=poly,
            zmin=args.zmin,
            zmax=args.zmax,
            ns=args.ns,
            corridor=args.corridor,
            scalar_name=args.scalar,
            mask_air=not args.no_mask_air,
            ocean_log10_rho=args.ocean_log10,
        )
        pv_plot_dataset(blocks, scalar_name=args.scalar, view_xy=False, show_edges=args.edges)
        return 0

print("Unknown command.", file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
