"""Unified visualization utilities for FEMTIC resistivity models.

This module collects the various visualization helpers that were
previously spread over several modules:

- femtic_resistivity_plotting
- femtic_borehole_viz
- femtic_slice_matplotlib
- femtic_map_slice_matplotlib
- femtic_slice_pyvista
- femtic_map_slice_pyvista

It provides Matplotlib and PyVista helpers for:

- Borehole visualization.
- Vertical curtain slices.
- Horizontal map slices.
- Consistent handling of special resistivity blocks:
  index 0 = air (not plotted / transparent),
  index 1 = ocean (special conductive value).

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-12-09
"""

from __future__ import annotations

# NOTE:
# The contents below are taken from the original visualization modules.
# They are concatenated here so that all FEMTIC plotting utilities can
# be imported from a single module: ``femtic_viz``.



# ===== Begin femtic_resistivity_plotting.py =====


"""Utilities for plotting FEMTIC resistivity blocks with special handling for air and ocean.

This module provides helper functions and example plotting routines that
respect the special meaning of the first two resistivity blocks in
``resistivity_block...dat`` files used by FEMTIC-style meshes:

- Block index 0: air
- Block index 1: ocean

The convention implemented here is:

* Air (index 0) is set to NaN before plotting, so that it becomes
  transparent or white (depending on the plotting backend and settings).
* Ocean (index 1) can optionally be forced to a very conductive value
  (e.g. 1e-10 Ohm·m) so that it is clearly distinguishable in the color
  scale.

The functions are written to be easy to integrate into existing plotting
scripts for both Matplotlib-based map views and PyVista-based curtain or
3D views.

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-12-09
"""


from pathlib import Path
from typing import Iterable, Literal

import numpy as np


def load_resistivity_blocks(path: str | Path) -> np.ndarray:
    """Load a 1D resistivity block vector from a FEMTIC-style file.

    The function reads a text file (e.g. ``resistivity_block_iter0.dat``)
    and returns the values as a one-dimensional NumPy array. It does not
    apply any special handling to air or ocean; this is delegated to
    :func:`prepare_rho_for_plotting`.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the resistivity block file. The file is expected to
        contain one numerical value per line or a simple whitespace-
        separated list of values that :func:`numpy.loadtxt` can parse
        into a one-dimensional array.

    Returns
    -------
    rho : numpy.ndarray of shape (n_blocks,)
        Raw resistivity values per block [Ohm·m] as stored in the file.

    Raises
    ------
    ValueError
        If the loaded array is not one-dimensional.

    Notes
    -----
    The first two entries in the returned array typically have a fixed
    meaning in FEMTIC-style workflows:

    * ``rho[0]``: air
    * ``rho[1]``: ocean

    These entries are not modified here so that the caller has full
    control over how they are treated in different contexts (plotting,
    inversion, etc.).
    """
    path = Path(path)
    rho = np.loadtxt(path, dtype=float)

    if rho.ndim != 1:
        raise ValueError(f"Expected 1D resistivity vector, got shape {rho.shape!r}")

    return rho


def prepare_rho_for_plotting(
    rho: np.ndarray,
    *,
    ocean_value: float | None = 1.0e-10,
    mask_air: bool = True,
) -> np.ndarray:
    """Prepare a resistivity block vector for plotting.

    This function applies the standard FEMTIC-style conventions for air
    and ocean blocks before plotting:

    * Air (index 0) is set to NaN if ``mask_air=True``. Most plotting
      libraries treat NaNs as transparent or ignore them in color
      mapping, which effectively removes air from the plot.
    * Ocean (index 1) is optionally forced to a fixed value,
      ``ocean_value``. This is useful to ensure that ocean appears with
      a clearly defined, very conductive value in the color scale.

    Parameters
    ----------
    rho : numpy.ndarray of shape (n_blocks,)
        Raw resistivity values per block [Ohm·m]. This array is not
        modified in-place; a copy is returned.
    ocean_value : float or None, optional
        If not None, the ocean block (index 1) is set to this value.
        Use e.g. ``1e-10`` for a very conductive ocean. If None, the
        original value in ``rho[1]`` is left unchanged.
    mask_air : bool, optional
        If True (default), air (index 0) is set to NaN. If False, the
        original value in ``rho[0]`` is left unchanged.

    Returns
    -------
    rho_plot : numpy.ndarray of shape (n_blocks,)
        Copy of the input array with modifications applied to indices 0
        and 1 as requested.
    """
    rho_plot = np.asarray(rho, dtype=float).copy()

    if rho_plot.size >= 1 and mask_air:
        rho_plot[0] = np.nan

    if rho_plot.size >= 2 and ocean_value is not None:
        rho_plot[1] = float(ocean_value)

    return rho_plot


def map_blocks_to_cells(
    block_values: np.ndarray,
    block_indices: np.ndarray,
) -> np.ndarray:
    """Map block-wise values to per-cell or per-node values."""
    values = np.asarray(block_values)[np.asarray(block_indices)]
    return values


# ===== End femtic_resistivity_plotting.py =====



# ===== Begin femtic_borehole_viz.py =====


"""
femtic_borehole_viz.py

Plotting utilities for vertical borehole resistivity profiles using pure
Matplotlib.

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-12-07
"""

from typing import Optional, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def load_resistivity_blocks(path: str) -> np.ndarray:
    """Load 1D resistivity block vector from FEMTIC resistivity_block...dat file.

    The first two entries have a special meaning and are usually fixed:

    - index 0: air
    - index 1: ocean

    Returns
    -------
    rho : ndarray, shape (n_blocks,)
        Resistivity per block as loaded from the file (no modifications).

    """
    rho = np.loadtxt(path, dtype=float)
    if rho.ndim != 1:
        raise ValueError(
            f"Expected 1D resistivity vector, got shape {rho.shape!r}"
        )
    return rho


def prepare_rho_for_plotting(
    rho: np.ndarray,
    *,
    ocean_value: float | None = 1.0e-10,
    mask_air: bool = True,
) -> np.ndarray:
    """Modify block resistivities for plotting with special handling of air and ocean.

    - air (index 0) is set to NaN if ``mask_air=True`` so that it becomes
      transparent/white in plots.
    - ocean (index 1) is optionally forced to ``ocean_value`` to give a
      distinct very conductive value.

    Parameters
    ----------
    rho : ndarray, shape (n_blocks,)
        Raw resistivity block values as loaded from file.
    ocean_value : float or None, optional
        If not None, the ocean block (index 1) is set to this value.
        Use e.g. 1e-10 for “very conductive water”.
    mask_air : bool, optional
        If True, air (index 0) is set to NaN for transparency.

    Returns
    -------
    rho_plot : ndarray, shape (n_blocks,)
        Modified resistivity vector for plotting.

    """
    rho_plot = np.asarray(rho, dtype=float).copy()

    if rho_plot.size >= 1 and mask_air:
        # air → transparent (NaN)
        rho_plot[0] = np.nan

    if rho_plot.size >= 2 and ocean_value is not None:
        # ocean → enforce special value
        rho_plot[1] = float(ocean_value)

    return rho_plot


def plot_vertical_profile(
    z: np.ndarray,
    values: np.ndarray,
    *,
    label: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    logx: bool = False,
    z_positive_down: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a single vertical profile values(z).

    Parameters
    ----------
    z : ndarray, shape (n,)
        Depth or elevation samples.
    values : ndarray, shape (n,)
        Corresponding profile values (e.g., resistivity).
    label : str, optional
        Curve label for the legend.
    ax : matplotlib.axes.Axes, optional
        Existing axis to plot into. If None, a new figure and axis are created.
    logx : bool, optional
        If True, use logarithmic scaling on the x-axis.
    z_positive_down : bool, optional
        If True, orient the axis such that depth increases downwards.

    Returns
    -------
    fig, ax : Figure, Axes
        The Matplotlib figure and axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.0, 6.0))
    else:
        fig = ax.figure

    ax.plot(values, z, label=label)
    ax.set_xlabel("Resistivity")
    ax.set_ylabel("Depth z")

    if logx:
        ax.set_xscale("log")

    if z_positive_down:
        if z.size >= 2 and z[1] < z[0]:
            ax.invert_yaxis()
    else:
        if z.size >= 2 and z[1] > z[0]:
            ax.invert_yaxis()

    if label is not None:
        ax.legend()

    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_vertical_profiles(
    z: np.ndarray,
    profiles: Sequence[np.ndarray],
    *,
    labels: Optional[Sequence[str]] = None,
    ax: Optional[plt.Axes] = None,
    logx: bool = False,
    z_positive_down: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot multiple vertical profiles on a shared z-axis.

    Parameters
    ----------
    z : ndarray, shape (n,)
        Depth or elevation samples.
    profiles : sequence of ndarray
        Collection of profiles values(z) to plot.
    labels : sequence of str, optional
        Labels for each profile curve.
    ax : matplotlib.axes.Axes, optional
        Existing axis to plot into. If None, a new figure and axis are created.
    logx : bool, optional
        If True, use logarithmic scaling on the x-axis.
    z_positive_down : bool, optional
        If True, orient the axis such that depth increases downwards.

    Returns
    -------
    fig, ax : Figure, Axes
        The Matplotlib figure and axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 6.5))
    else:
        fig = ax.figure

    for i, prof in enumerate(profiles):
        lab = labels[i] if (labels is not None and i < len(labels)) else None
        ax.plot(prof, z, label=lab)

    ax.set_xlabel("Resistivity")
    ax.set_ylabel("Depth z")

    if logx:
        ax.set_xscale("log")

    if z_positive_down:
        if z.size >= 2 and z[1] < z[0]:
            ax.invert_yaxis()
    else:
        if z.size >= 2 and z[1] > z[0]:
            ax.invert_yaxis()

    if labels is not None:
        ax.legend()

    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig, ax


# ===== End femtic_borehole_viz.py =====



# ===== Begin femtic_slice_matplotlib.py =====


#!/usr/bin/env python3
"""
femtic_slice_matplotlib.py

Vertical FEMTIC resistivity slices (curtains) from NPZ, pure Matplotlib, with
pluggable interpolation (idw / nearest / rbf).

NPZ is expected to contain at least:
    centroid            (nelem, 3)
    log10_resistivity   (nelem,)
    flag                (nelem,)  [optional, 1 = fixed]

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-12-08
"""

from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt


def _build_s(points_xy: np.ndarray) -> np.ndarray:
    """Cumulative arclength along a polyline."""
    points_xy = np.asarray(points_xy, dtype=float)
    diffs = np.diff(points_xy, axis=0)
    seglen = np.sqrt(np.sum(diffs**2, axis=1))
    return np.concatenate([[0.0], np.cumsum(seglen)])


def sample_polyline(points_xy: np.ndarray, ns: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a polyline at ns equally spaced arclength positions."""
    points_xy = np.asarray(points_xy, dtype=float)
    if points_xy.shape[0] < 2:
        raise ValueError("Polyline must contain at least two vertices.")
    s = _build_s(points_xy)
    total = s[-1]
    if total <= 0.0:
        raise ValueError("Polyline length must be positive.")
    S = np.linspace(0.0, total, ns)
    XY = np.empty((ns, 2), dtype=float)
    for i, si in enumerate(S):
        j = np.searchsorted(s, si, side="right") - 1
        j = max(0, min(j, len(s) - 2))
        denom = max(s[j + 1] - s[j], 1e-12)
        t = (si - s[j]) / denom
        XY[i] = (1.0 - t) * points_xy[j] + t * points_xy[j + 1]
    return S, XY


def _idw_point(q: np.ndarray, pts: np.ndarray, vals: np.ndarray,
               power: float = 2.0, eps: float = 1e-6) -> float:
    """Inverse distance weighting in 3D for a single point (log10-space)."""
    d2 = np.sum((pts - q) ** 2, axis=1) + eps**2
    w = 1.0 / (d2 ** (power / 2.0))
    return float(np.sum(w * vals) / np.sum(w))


def curtain_slice_idw(points_xy: np.ndarray, z_samples: np.ndarray,
                      centroids: np.ndarray, vals_log10: np.ndarray,
                      power: float = 2.0, ns: int = 301
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Curtain slice using 3D IDW on log10 values."""
    centroids = np.asarray(centroids, dtype=float)
    vals_log10 = np.asarray(vals_log10, dtype=float)
    Z = np.asarray(z_samples, dtype=float)
    S, XY = sample_polyline(points_xy, ns)
    V = np.empty((Z.size, S.size), dtype=float)
    for j, (x, y) in enumerate(XY):
        for i, z in enumerate(Z):
            V[i, j] = _idw_point(np.array([x, y, z], float),
                                 centroids, vals_log10, power=power)
    return S, Z, V, XY


def curtain_slice_nearest(points_xy: np.ndarray, z_samples: np.ndarray,
                          centroids: np.ndarray, vals_log10: np.ndarray,
                          ns: int = 301
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Curtain slice using nearest neighbour in 3D (no smoothing)."""
    centroids = np.asarray(centroids, dtype=float)
    vals_log10 = np.asarray(vals_log10, dtype=float)
    Z = np.asarray(z_samples, dtype=float)
    S, XY = sample_polyline(points_xy, ns)
    V = np.empty((Z.size, S.size), dtype=float)
    try:
        from scipy.spatial import cKDTree  # type: ignore[attr-defined]
        tree = cKDTree(centroids)
        queries = np.stack(
            [np.repeat(XY[:, 0], Z.size),
             np.repeat(XY[:, 1], Z.size),
             np.tile(Z, XY.shape[0])],
            axis=1,
        )
        _, idx = tree.query(queries)
        V[:] = vals_log10[idx].reshape(Z.size, XY.shape[0], order="F")
    except Exception:
        for j, (x, y) in enumerate(XY):
            for i, z in enumerate(Z):
                q = np.array([x, y, z], float)
                d2 = np.sum((centroids - q) ** 2, axis=1)
                k = int(np.argmin(d2))
                V[i, j] = vals_log10[k]
    return S, Z, V, XY


def curtain_slice_rbf(points_xy: np.ndarray, z_samples: np.ndarray,
                      centroids: np.ndarray, vals_log10: np.ndarray,
                      ns: int = 301
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Curtain slice using 3D RBF interpolation (SciPy)."""
    try:
        from scipy.interpolate import RBFInterpolator  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "RBF interpolation requires scipy.interpolate.RBFInterpolator."
        ) from exc
    centroids = np.asarray(centroids, dtype=float)
    vals_log10 = np.asarray(vals_log10, dtype=float)
    Z = np.asarray(z_samples, dtype=float)
    S, XY = sample_polyline(points_xy, ns)
    rbf = RBFInterpolator(centroids, vals_log10, kernel="thin_plate_spline")
    XX = np.repeat(XY[:, 0][None, :], Z.size, axis=0)
    YY = np.repeat(XY[:, 1][None, :], Z.size, axis=0)
    ZZ = np.repeat(Z[:, None], XY.shape[0], axis=1)
    Q = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
    V_flat = rbf(Q)
    V = V_flat.reshape(Z.size, XY.shape[0])
    return S, Z, V, XY


def curtain_slice(points_xy: np.ndarray, z_samples: np.ndarray,
                  centroids: np.ndarray, vals_log10: np.ndarray,
                  *, interp: str = "idw", power: float = 2.0, ns: int = 301
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch curtain interpolation according to interp method."""
    interp = interp.lower()
    if interp == "idw":
        return curtain_slice_idw(points_xy, z_samples, centroids, vals_log10,
                                 power=power, ns=ns)
    if interp == "nearest":
        return curtain_slice_nearest(points_xy, z_samples, centroids, vals_log10,
                                     ns=ns)
    if interp == "rbf":
        return curtain_slice_rbf(points_xy, z_samples, centroids, vals_log10,
                                 ns=ns)
    raise ValueError(f"Unknown interp '{interp}' (expected idw, nearest, rbf).")


def plot_curtain_matplotlib(S: np.ndarray, Z: np.ndarray, V: np.ndarray,
                            *, logscale: bool = True,
                            z_positive_down: bool = True,
                            cmap: str = "viridis",
                            vmin: float | None = None,
                            vmax: float | None = None,
                            title: str = "Curtain slice"
                            ) -> tuple[plt.Figure, plt.Axes]:
    """Plot a curtain slice using Matplotlib."""
    S = np.asarray(S, float)
    Z = np.asarray(Z, float)
    V = np.asarray(V, float)
    if logscale:
        data = np.log10(np.clip(V, 1e-30, np.inf))
        cbar_label = "log10(ρ) [Ω·m]"
    else:
        data = V
        cbar_label = "ρ [Ω·m]"
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    if z_positive_down:
        extent = [S.min(), S.max(), Z.max(), Z.min()]
        origin = "upper"
    else:
        extent = [S.min(), S.max(), Z.min(), Z.max()]
        origin = "lower"
    im = ax.imshow(
        data,
        extent=extent,
        origin=origin,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Distance along slice S")
    ax.set_ylabel("Depth z")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    return fig, ax


def femtic_slice_from_npz_matplotlib(
    npz_path: str,
    *,
    polyline_xy: np.ndarray | None = None,
    polyline_csv: str | None = None,
    zmin: float,
    zmax: float,
    nz: int = 201,
    ns: int = 301,
    power: float = 2.0,
    interp: str = "idw",
    logscale: bool = True,
    z_positive_down: bool = True,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    out_npz: str | None = None,
    out_csv: str | None = None,
    out_png: str | None = None,
    title: str = "Curtain slice",
) -> None:
    """High-level helper: build & plot curtain from FEMTIC NPZ."""
    import csv

    data = np.load(npz_path)
    if "centroid" not in data or "log10_resistivity" not in data:
        raise KeyError("NPZ must contain 'centroid' and 'log10_resistivity'.")
    centroids = data["centroid"]
    vals_log10 = data["log10_resistivity"]
    if "flag" in data:
        mask = data["flag"] != 1
        centroids = centroids[mask]
        vals_log10 = vals_log10[mask]

    if polyline_csv is not None:
        pts = []
        with open(polyline_csv, "r") as f:
            lines=f.readlines()
            for line in lines:
                xy = np.fromstring(line, dtype=float, sep=',')
                x = float(xy[0])
                y = float(xy[1])
                pts.append([x, y])
        print(pts)
        if len(pts) < 2:
            raise ValueError("Polyline CSV must contain at least two vertices.")
        poly_xy = np.asarray(pts, dtype=float)
    else:
        if polyline_xy is None or polyline_xy.shape[0] < 2:
            raise ValueError("Provide at least two polyline vertices via polyline_xy.")
        poly_xy = np.asarray(polyline_xy, dtype=float)

    Z = np.linspace(zmin, zmax, nz)
    S, Z, V_log10, XY = curtain_slice(
        poly_xy,
        Z,
        centroids,
        vals_log10,
        interp=interp,
        power=power,
        ns=ns,
    )

    if out_npz is not None:
        np.savez_compressed(out_npz, S=S, Z=Z, V_log10=V_log10, XY=XY, interp=interp)
        print("Saved curtain NPZ:", out_npz)

    if out_csv is not None:
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["s", "z", "log10_rho"])
            for j, s in enumerate(S):
                for i, z in enumerate(Z):
                    w.writerow([s, z, V_log10[i, j]])
        print("Saved curtain CSV:", out_csv)

    V_plot = 10.0 ** V_log10
    fig, ax = plot_curtain_matplotlib(
        S,
        Z,
        V_plot,
        logscale=logscale,
        z_positive_down=z_positive_down,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        title=title,
    )

    if out_png is not None:
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        print("Saved PNG:", out_png)
    else:
        try:
            plt.show()
        except Exception:
            pass


def main() -> None:
    """CLI entry point for vertical FEMTIC slices."""
    import argparse

    ap = argparse.ArgumentParser(
        description="Vertical FEMTIC slices along arbitrary XY polylines (Matplotlib)."
    )
    ap.add_argument("--npz", required=True, help="Element NPZ from femtic_mesh_to_npz.py.")
    ap.add_argument("--polyline-csv", help="CSV file with x,y columns for polyline vertices.")
    ap.add_argument(
        "--xy",
        action="append",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help="Polyline vertex (X Y). Repeat to build polyline.",
    )
    ap.add_argument("--zmin", type=float, required=True, help="Minimum depth of slice.")
    ap.add_argument("--zmax", type=float, required=True, help="Maximum depth of slice.")
    ap.add_argument("--nz", type=int, default=201, help="Number of depth samples.")
    ap.add_argument("--ns", type=int, default=301, help="Number of samples along slice.")
    ap.add_argument("--power", type=float, default=2.0, help="IDW power exponent (interp='idw').")
    ap.add_argument(
        "--interp",
        choices=["idw", "nearest", "rbf"],
        default="idw",
        help="Interpolation method.",
    )
    ap.add_argument("--logscale", action="store_true", help="Plot in log10(ρ).")
    ap.add_argument(
        "--z-positive-down",
        action="store_true",
        help="Depth increases downwards in plot.",
    )
    ap.add_argument("--cmap", default="viridis", help="Matplotlib colormap.")
    ap.add_argument("--vmin", type=float, default=None, help="Color scale minimum.")
    ap.add_argument("--vmax", type=float, default=None, help="Color scale maximum.")
    ap.add_argument("--out-npz", default=None, help="Optional output NPZ for curtain.")
    ap.add_argument("--out-csv", default=None, help="Optional output CSV for curtain.")
    ap.add_argument("--out-png", default=None, help="Optional PNG output for the plot.")
    ap.add_argument("--title", default="Curtain slice", help="Plot title.")
    args = ap.parse_args()

    polyline_xy = None
    if args.polyline_csv is None:
        if not args.xy or len(args.xy) < 2:
            raise ValueError(
                "Provide a polyline via --polyline-csv or at least two --xy X Y pairs."
            )
        polyline_xy = np.asarray([[x, y] for x, y in args.xy], dtype=float)

    femtic_slice_from_npz_matplotlib(
        npz_path=args.npz,
        polyline_xy=polyline_xy,
        polyline_csv=args.polyline_csv,
        zmin=args.zmin,
        zmax=args.zmax,
        nz=args.nz,
        ns=args.ns,
        power=args.power,
        interp=args.interp,
        logscale=args.logscale,
        z_positive_down=args.z_positive_down,
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        out_npz=args.out_npz,
        out_csv=args.out_csv,
        out_png=args.out_png,
        title=args.title,
    )


if __name__ == "__main__":
    main()


# ===== End femtic_slice_matplotlib.py =====



# ===== Begin femtic_map_slice_matplotlib.py =====


#!/usr/bin/env python3
"""
femtic_map_slice_matplotlib.py

Horizontal FEMTIC resistivity slices (maps) at fixed depth from NPZ, pure
Matplotlib, with pluggable interpolation (idw / nearest / rbf).

NPZ is expected to contain at least:
    centroid            (nelem, 3)
    log10_resistivity   (nelem,)
    flag                (nelem,)  [optional, 1 = fixed]

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-12-08
"""

from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt


def _select_depth_window(
    centroids: np.ndarray,
    vals_log10: np.ndarray,
    z0: float,
    dz: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select centroids and log10 values within a vertical window around z0."""
    centroids = np.asarray(centroids, dtype=float)
    vals_log10 = np.asarray(vals_log10, dtype=float)
    z = centroids[:, 2]
    half = dz / 2.0
    mask = (z >= z0 - half) & (z <= z0 + half)
    if not np.any(mask):
        raise ValueError("No centroids found in the depth window around z0.")
    pts_xy = centroids[mask, :2]
    vals_sel = vals_log10[mask]
    return pts_xy, vals_sel


def _idw_grid_2d(
    x: np.ndarray,
    y: np.ndarray,
    pts_xy: np.ndarray,
    vals_log10: np.ndarray,
    power: float = 2.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """2D IDW on regular (x, y) grid (log10-space)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    pts_xy = np.asarray(pts_xy, float)
    vals_log10 = np.asarray(vals_log10, float)
    nx = x.size
    ny = y.size
    V = np.empty((ny, nx), float)
    px = pts_xy[:, 0]
    py = pts_xy[:, 1]
    for j in range(ny):
        yj = y[j]
        dy2 = (py - yj) ** 2
        for i in range(nx):
            xi = x[i]
            d2 = (px - xi) ** 2 + dy2 + eps**2
            w = 1.0 / (d2 ** (power / 2.0))
            V[j, i] = float(np.sum(w * vals_log10) / np.sum(w))
    return V


def _nearest_grid_2d(
    x: np.ndarray,
    y: np.ndarray,
    pts_xy: np.ndarray,
    vals_log10: np.ndarray,
) -> np.ndarray:
    """2D nearest neighbour interpolation on regular grid."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    pts_xy = np.asarray(pts_xy, float)
    vals_log10 = np.asarray(vals_log10, float)
    nx = x.size
    ny = y.size
    V = np.empty((ny, nx), float)
    try:
        from scipy.spatial import cKDTree  # type: ignore[attr-defined]
        tree = cKDTree(pts_xy)
        XX, YY = np.meshgrid(x, y)
        Q = np.column_stack([XX.ravel(), YY.ravel()])
        _, idx = tree.query(Q)
        V[:] = vals_log10[idx].reshape(ny, nx)
    except Exception:
        px = pts_xy[:, 0]
        py = pts_xy[:, 1]
        for j in range(ny):
            yj = y[j]
            dy2 = (py - yj) ** 2
            for i in range(nx):
                xi = x[i]
                d2 = (px - xi) ** 2 + dy2
                k = int(np.argmin(d2))
                V[j, i] = vals_log10[k]
    return V


def _rbf_grid_2d(
    x: np.ndarray,
    y: np.ndarray,
    pts_xy: np.ndarray,
    vals_log10: np.ndarray,
) -> np.ndarray:
    """2D RBF interpolation on regular grid (SciPy)."""
    try:
        from scipy.interpolate import RBFInterpolator  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "RBF interpolation requires scipy.interpolate.RBFInterpolator."
        ) from exc
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    pts_xy = np.asarray(pts_xy, float)
    vals_log10 = np.asarray(vals_log10, float)
    rbf = RBFInterpolator(pts_xy, vals_log10, kernel="thin_plate_spline")
    XX, YY = np.meshgrid(x, y)
    Q = np.column_stack([XX.ravel(), YY.ravel()])
    V_flat = rbf(Q)
    return V_flat.reshape(y.size, x.size)


def _grid_2d(
    x: np.ndarray,
    y: np.ndarray,
    pts_xy: np.ndarray,
    vals_log10: np.ndarray,
    *,
    interp: str = "idw",
    power: float = 2.0,
) -> np.ndarray:
    """Dispatch 2D interpolation on a regular grid."""
    interp = interp.lower()
    if interp == "idw":
        return _idw_grid_2d(x, y, pts_xy, vals_log10, power=power)
    if interp == "nearest":
        return _nearest_grid_2d(x, y, pts_xy, vals_log10)
    if interp == "rbf":
        return _rbf_grid_2d(x, y, pts_xy, vals_log10)
    raise ValueError(f"Unknown interp '{interp}' (expected idw, nearest, rbf).")


def plot_map_slice_matplotlib(
    X: np.ndarray,
    Y: np.ndarray,
    V: np.ndarray,
    *,
    logscale: bool = True,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    title: str = "Horizontal slice",
    z_label: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a horizontal map slice using Matplotlib."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    V = np.asarray(V, float)
    if logscale:
        data = np.log10(np.clip(V, 1e-30, np.inf))
        cbar_label = "log10(ρ) [Ω·m]"
    else:
        data = V
        cbar_label = "ρ [Ω·m]"
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    pc = ax.pcolormesh(X, Y, data, shading="auto", cmap=cmap,
                       vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    full_title = title
    if z_label is not None:
        full_title = f"{title} ({z_label})"
    ax.set_title(full_title)
    cbar = fig.colorbar(pc, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    return fig, ax


def femtic_map_slice_from_npz_matplotlib(
    npz_path: str,
    *,
    z0: float,
    dz: float,
    nx: int = 200,
    ny: int = 200,
    xmin: float | None = None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    power: float = 2.0,
    interp: str = "idw",
    logscale: bool = True,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    out_npz: str | None = None,
    out_csv: str | None = None,
    out_png: str | None = None,
    title: str = "Horizontal slice",
) -> None:
    """High-level helper: compute horizontal map slice from FEMTIC NPZ and plot."""
    import csv

    data = np.load(npz_path)
    if "centroid" not in data or "log10_resistivity" not in data:
        raise KeyError("NPZ must contain 'centroid' and 'log10_resistivity'.")
    centroids = data["centroid"]
    vals_log10 = data["log10_resistivity"]
    if "flag" in data:
        mask = data["flag"] != 1
        centroids = centroids[mask]
        vals_log10 = vals_log10[mask]

    pts_xy, vals_sel = _select_depth_window(centroids, vals_log10, z0=z0, dz=dz)

    if xmin is None:
        xmin = float(pts_xy[:, 0].min())
    if xmax is None:
        xmax = float(pts_xy[:, 0].max())
    if ymin is None:
        ymin = float(pts_xy[:, 1].min())
    if ymax is None:
        ymax = float(pts_xy[:, 1].max())

    dx = xmax - xmin
    dy = ymax - ymin
    xmin -= 0.02 * dx
    xmax += 0.02 * dx
    ymin -= 0.02 * dy
    ymax += 0.02 * dy

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    V_log10 = _grid_2d(x, y, pts_xy, vals_sel, interp=interp, power=power)
    X, Y = np.meshgrid(x, y)

    if out_npz is not None:
        np.savez_compressed(out_npz, X=X, Y=Y, V_log10=V_log10, z0=z0, interp=interp)
        print("Saved map slice NPZ:", out_npz)

    if out_csv is not None:
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x", "y", "log10_rho", "z"])
            for j in range(ny):
                for i in range(nx):
                    w.writerow([X[j, i], Y[j, i], V_log10[j, i], z0])
        print("Saved map slice CSV:", out_csv)

    V_plot = 10.0 ** V_log10
    z_label = f"z = {z0:g} m"
    fig, ax = plot_map_slice_matplotlib(
        X,
        Y,
        V_plot,
        logscale=logscale,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        title=title,
        z_label=z_label,
    )

    if out_png is not None:
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        print("Saved PNG:", out_png)
    else:
        try:
            plt.show()
        except Exception:
            pass


def main() -> None:
    """CLI entry point for computing & plotting FEMTIC horizontal slices."""
    import argparse

    ap = argparse.ArgumentParser(
        description="Horizontal FEMTIC resistivity slices at fixed depth (Matplotlib)."
    )
    ap.add_argument("--npz", required=True, help="Element NPZ from femtic_mesh_to_npz.py.")
    ap.add_argument("--z0", type=float, required=True, help="Central depth of slice.")
    ap.add_argument("--dz", type=float, required=True, help="Thickness of vertical window around z0.")
    ap.add_argument("--nx", type=int, default=200, help="Number of grid points in x.")
    ap.add_argument("--ny", type=int, default=200, help="Number of grid points in y.")
    ap.add_argument("--xmin", type=float, default=None, help="Minimum x for grid.")
    ap.add_argument("--xmax", type=float, default=None, help="Maximum x for grid.")
    ap.add_argument("--ymin", type=float, default=None, help="Minimum y for grid.")
    ap.add_argument("--ymax", type=float, default=None, help="Maximum y for grid.")
    ap.add_argument("--power", type=float, default=2.0, help="IDW power exponent (interp='idw').")
    ap.add_argument(
        "--interp",
        choices=["idw", "nearest", "rbf"],
        default="idw",
        help="Interpolation method.",
    )
    ap.add_argument("--logscale", action="store_true", help="Plot in log10(ρ).")
    ap.add_argument("--cmap", default="viridis", help="Matplotlib colormap name.")
    ap.add_argument("--vmin", type=float, default=None, help="Color scale minimum.")
    ap.add_argument("--vmax", type=float, default=None, help="Color scale maximum.")
    ap.add_argument("--out-npz", default=None, help="Optional output NPZ for map slice.")
    ap.add_argument("--out-csv", default=None, help="Optional output CSV for map slice.")
    ap.add_argument("--out-png", default=None, help="Optional PNG output for the plot.")
    ap.add_argument("--title", default="Horizontal slice", help="Plot title.")
    args = ap.parse_args()

    femtic_map_slice_from_npz_matplotlib(
        npz_path=args.npz,
        z0=args.z0,
        dz=args.dz,
        nx=args.nx,
        ny=args.ny,
        xmin=args.xmin,
        xmax=args.xmax,
        ymin=args.ymin,
        ymax=args.ymax,
        power=args.power,
        interp=args.interp,
        logscale=args.logscale,
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        out_npz=args.out_npz,
        out_csv=args.out_csv,
        out_png=args.out_png,
        title=args.title,
    )


if __name__ == "__main__":
    main()


# ===== End femtic_map_slice_matplotlib.py =====



# ===== Begin femtic_slice_pyvista.py =====


#!/usr/bin/env python3
"""
femtic_slice_pyvista.py

Vertical FEMTIC resistivity slices (curtains) from NPZ using PyVista.

This script reuses the interpolation logic from femtic_slice_matplotlib
(curtain_slice) and builds a 2-D PyVista StructuredGrid in (S, z)-space, where:

    S : distance along polyline
    z : depth (positive downwards if requested)

Cell data:
    log10_rho  (log10_resistivity in NPZ)
    rho        (10**log10_rho)

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-12-08
"""

import numpy as np


def build_curtain_grid_from_npz(
    npz_path: str,
    *,
    polyline_xy: np.ndarray | None = None,
    polyline_csv: str | None = None,
    zmin: float,
    zmax: float,
    nz: int = 201,
    ns: int = 301,
    interp: str = "idw",
    power: float = 2.0,
) -> "pyvista.StructuredGrid":
    """Build a PyVista StructuredGrid for a vertical curtain.

    Parameters
    ----------
    npz_path : str
        FEMTIC NPZ created by femtic_mesh_to_npz.py.
    polyline_xy : ndarray, shape (m, 2), optional
        Polyline vertices in XY. Ignored if polyline_csv is given.
    polyline_csv : str, optional
        CSV file with at least 'x,y' in the first two columns.
    zmin, zmax : float
        Depth range of curtain.
    nz : int
        Number of samples in depth.
    ns : int
        Number of samples along polyline.
    interp : {'idw', 'nearest', 'rbf'}
        Interpolation method (same as femtic_slice_matplotlib).
    power : float
        IDW exponent if interp='idw'.

    Returns
    -------
    grid : pyvista.StructuredGrid
        Structured grid in (S, z) coordinates with 'log10_rho' and 'rho'.
    """
    try:
        import pyvista as pv
    except Exception as exc:  # pragma: no cover
        raise ImportError("pyvista is required for curtain plotting.") from exc

    from femtic_slice_matplotlib import curtain_slice  # reuse interpolation

    import csv

    data = np.load(npz_path)
    if "centroid" not in data or "log10_resistivity" not in data:
        raise KeyError("NPZ must contain 'centroid' and 'log10_resistivity'.")

    centroids = data["centroid"]
    vals_log10 = data["log10_resistivity"]
    if "flag" in data:
        mask = data["flag"] != 1
        centroids = centroids[mask]
        vals_log10 = vals_log10[mask]

    if polyline_csv is not None:
        pts = []
        with open(polyline_csv, "r") as f:
            r = csv.reader(f)
            for row in r:
                if not row:
                    continue
                try:
                    x = float(row[0])
                    y = float(row[1])
                except Exception:
                    continue
                pts.append([x, y])
        if len(pts) < 2:
            raise ValueError("Polyline CSV must contain at least two vertices.")
        poly_xy = np.asarray(pts, dtype=float)
    else:
        if polyline_xy is None or polyline_xy.shape[0] < 2:
            raise ValueError("Provide at least two polyline vertices via polyline_xy.")
        poly_xy = np.asarray(polyline_xy, dtype=float)

    Z = np.linspace(zmin, zmax, nz)
    S, Z, V_log10, XY = curtain_slice(
        poly_xy,
        Z,
        centroids,
        vals_log10,
        interp=interp,
        power=power,
        ns=ns,
    )

    # Build StructuredGrid in (S, z) plane, y=0
    S2, Z2 = np.meshgrid(S, Z)  # (nz, ns)
    X = S2
    Y = np.zeros_like(S2)
    Zcoords = Z2

    # Shape to (nz, ns, 1) for StructuredGrid
    X3 = X[:, :, None]
    Y3 = Y[:, :, None]
    Z3 = Zcoords[:, :, None]

    grid = pv.StructuredGrid(X3, Y3, Z3)
    grid["log10_rho"] = V_log10.ravel(order="C")
    grid["rho"] = (10.0 ** V_log10).ravel(order="C")

    # Optionally store polyline locations as field data
    grid.field_data["polyline_xy"] = XY

    return grid


def main() -> None:
    """CLI entry point for PyVista curtain slices."""
    import argparse
    import pyvista as pv

    ap = argparse.ArgumentParser(
        description="Vertical FEMTIC curtain slices in (S, z) using PyVista."
    )
    ap.add_argument("--npz", required=True, help="Element NPZ from femtic_mesh_to_npz.py.")
    ap.add_argument("--polyline-csv", help="CSV file with x,y columns for polyline vertices.")
    ap.add_argument(
        "--xy",
        action="append",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help="Polyline vertex (X Y). Repeat to build polyline.",
    )
    ap.add_argument("--zmin", type=float, required=True, help="Minimum depth of slice.")
    ap.add_argument("--zmax", type=float, required=True, help="Maximum depth of slice.")
    ap.add_argument("--nz", type=int, default=201, help="Number of depth samples.")
    ap.add_argument("--ns", type=int, default=301, help="Number of samples along slice.")
    ap.add_argument(
        "--interp",
        choices=["idw", "nearest", "rbf"],
        default="idw",
        help="Interpolation method.",
    )
    ap.add_argument("--power", type=float, default=2.0, help="IDW power exponent.")
    ap.add_argument("--out-vtk", default=None, help="Optional output .vts/.vtk file.")
    ap.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open interactive PyVista window.",
    )
    args = ap.parse_args()

    polyline_xy = None
    if args.polyline_csv is None:
        if not args.xy or len(args.xy) < 2:
            raise ValueError(
                "Provide a polyline via --polyline-csv or at least two --xy X Y pairs."
            )
        polyline_xy = np.asarray([[x, y] for x, y in args.xy], dtype=float)

    grid = build_curtain_grid_from_npz(
        args.npz,
        polyline_xy=polyline_xy,
        polyline_csv=args.polyline_csv,
        zmin=args.zmin,
        zmax=args.zmax,
        nz=args.nz,
        ns=args.ns,
        interp=args.interp,
        power=args.power,
    )

    if args.out_vtk is not None:
        grid.save(args.out_vtk)
        print("Saved curtain grid to:", args.out_vtk)

    if not args.no_show:
        grid.plot(
            scalars="log10_rho",
            cmap="viridis",
        )


if __name__ == "__main__":  # pragma: no cover
    main()


# ===== End femtic_slice_pyvista.py =====



# ===== Begin femtic_map_slice_pyvista.py =====


#!/usr/bin/env python3
"""
femtic_map_slice_pyvista.py

Horizontal FEMTIC resistivity slices (maps) at fixed depth using PyVista.

This script mirrors femtic_map_slice_matplotlib: it selects centroids in a
depth window around z0, interpolates onto a regular (x, y) grid using one
of:

    interp = 'idw'      (inverse-distance weighting)
    interp = 'nearest'  (nearest neighbour)
    interp = 'rbf'      (Radial Basis Function, SciPy)

and builds a 2-D PyVista StructuredGrid at constant depth z0.

Cell data:
    log10_rho  (log10_resistivity)
    rho        (10**log10_rho)

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-12-08
"""

from typing import Tuple
import numpy as np


def _select_depth_window(
    centroids: np.ndarray,
    vals_log10: np.ndarray,
    z0: float,
    dz: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select centroids and log10 values within a vertical window around z0."""
    centroids = np.asarray(centroids, dtype=float)
    vals_log10 = np.asarray(vals_log10, dtype=float)
    z = centroids[:, 2]
    half = dz / 2.0
    mask = (z >= z0 - half) & (z <= z0 + half)
    if not np.any(mask):
        raise ValueError("No centroids found in the depth window around z0.")
    pts_xy = centroids[mask, :2]
    vals_sel = vals_log10[mask]
    return pts_xy, vals_sel


def _idw_grid_2d(
    x: np.ndarray,
    y: np.ndarray,
    pts_xy: np.ndarray,
    vals_log10: np.ndarray,
    power: float = 2.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """2D IDW on regular (x, y) grid (log10-space)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    pts_xy = np.asarray(pts_xy, float)
    vals_log10 = np.asarray(vals_log10, float)
    nx = x.size
    ny = y.size
    V = np.empty((ny, nx), float)
    px = pts_xy[:, 0]
    py = pts_xy[:, 1]
    for j in range(ny):
        yj = y[j]
        dy2 = (py - yj) ** 2
        for i in range(nx):
            xi = x[i]
            d2 = (px - xi) ** 2 + dy2 + eps**2
            w = 1.0 / (d2 ** (power / 2.0))
            V[j, i] = float(np.sum(w * vals_log10) / np.sum(w))
    return V


def _nearest_grid_2d(
    x: np.ndarray,
    y: np.ndarray,
    pts_xy: np.ndarray,
    vals_log10: np.ndarray,
) -> np.ndarray:
    """2D nearest neighbour interpolation on regular grid."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    pts_xy = np.asarray(pts_xy, float)
    vals_log10 = np.asarray(vals_log10, float)
    nx = x.size
    ny = y.size
    V = np.empty((ny, nx), float)
    try:
        from scipy.spatial import cKDTree  # type: ignore[attr-defined]
        tree = cKDTree(pts_xy)
        XX, YY = np.meshgrid(x, y)
        Q = np.column_stack([XX.ravel(), YY.ravel()])
        _, idx = tree.query(Q)
        V[:] = vals_log10[idx].reshape(ny, nx)
    except Exception:
        px = pts_xy[:, 0]
        py = pts_xy[:, 1]
        for j in range(ny):
            yj = y[j]
            dy2 = (py - yj) ** 2
            for i in range(nx):
                xi = x[i]
                d2 = (px - xi) ** 2 + dy2
                k = int(np.argmin(d2))
                V[j, i] = vals_log10[k]
    return V


def _rbf_grid_2d(
    x: np.ndarray,
    y: np.ndarray,
    pts_xy: np.ndarray,
    vals_log10: np.ndarray,
) -> np.ndarray:
    """2D RBF interpolation on regular grid (SciPy)."""
    try:
        from scipy.interpolate import RBFInterpolator  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "RBF interpolation requires scipy.interpolate.RBFInterpolator."
        ) from exc
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    pts_xy = np.asarray(pts_xy, float)
    vals_log10 = np.asarray(vals_log10, float)
    rbf = RBFInterpolator(pts_xy, vals_log10, kernel="thin_plate_spline")
    XX, YY = np.meshgrid(x, y)
    Q = np.column_stack([XX.ravel(), YY.ravel()])
    V_flat = rbf(Q)
    return V_flat.reshape(y.size, x.size)


def _grid_2d(
    x: np.ndarray,
    y: np.ndarray,
    pts_xy: np.ndarray,
    vals_log10: np.ndarray,
    *,
    interp: str = "idw",
    power: float = 2.0,
) -> np.ndarray:
    """Dispatch 2D interpolation on a regular grid."""
    interp = interp.lower()
    if interp == "idw":
        return _idw_grid_2d(x, y, pts_xy, vals_log10, power=power)
    if interp == "nearest":
        return _nearest_grid_2d(x, y, pts_xy, vals_log10)
    if interp == "rbf":
        return _rbf_grid_2d(x, y, pts_xy, vals_log10)
    raise ValueError(f"Unknown interp '{interp}' (expected idw, nearest, rbf).")


def build_map_grid_from_npz(
    npz_path: str,
    *,
    z0: float,
    dz: float,
    nx: int = 200,
    ny: int = 200,
    xmin: float | None = None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    interp: str = "idw",
    power: float = 2.0,
) -> "pyvista.StructuredGrid":
    """Build a PyVista StructuredGrid for a horizontal slice at depth z0."""
    try:
        import pyvista as pv
    except Exception as exc:  # pragma: no cover
        raise ImportError("pyvista is required for map plotting.") from exc

    data = np.load(npz_path)
    if "centroid" not in data or "log10_resistivity" not in data:
        raise KeyError("NPZ must contain 'centroid' and 'log10_resistivity'.")

    centroids = data["centroid"]
    vals_log10 = data["log10_resistivity"]
    if "flag" in data:
        mask = data["flag"] != 1
        centroids = centroids[mask]
        vals_log10 = vals_log10[mask]

    pts_xy, vals_sel = _select_depth_window(centroids, vals_log10, z0=z0, dz=dz)

    if xmin is None:
        xmin = float(pts_xy[:, 0].min())
    if xmax is None:
        xmax = float(pts_xy[:, 0].max())
    if ymin is None:
        ymin = float(pts_xy[:, 1].min())
    if ymax is None:
        ymax = float(pts_xy[:, 1].max())

    dx = xmax - xmin
    dy = ymax - ymin
    xmin -= 0.02 * dx
    xmax += 0.02 * dx
    ymin -= 0.02 * dy
    ymax += 0.02 * dy

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    V_log10 = _grid_2d(x, y, pts_xy, vals_sel, interp=interp, power=power)
    X, Y = np.meshgrid(x, y)

    # Build StructuredGrid in (x, y, z0)
    Z = np.full_like(X, z0, dtype=float)
    X3 = X[:, :, None]
    Y3 = Y[:, :, None]
    Z3 = Z[:, :, None]

    import pyvista as pv

    grid = pv.StructuredGrid(X3, Y3, Z3)
    grid["log10_rho"] = V_log10.ravel(order="C")
    grid["rho"] = (10.0 ** V_log10).ravel(order="C")
    return grid


def main() -> None:
    """CLI entry point for PyVista horizontal map slices."""
    import argparse
    import pyvista as pv

    ap = argparse.ArgumentParser(
        description="Horizontal FEMTIC resistivity slices at fixed depth using PyVista."
    )
    ap.add_argument("--npz", required=True, help="Element NPZ from femtic_mesh_to_npz.py.")
    ap.add_argument("--z0", type=float, required=True, help="Central depth of slice.")
    ap.add_argument("--dz", type=float, required=True, help="Thickness of vertical window around z0.")
    ap.add_argument("--nx", type=int, default=200, help="Number of grid points in x.")
    ap.add_argument("--ny", type=int, default=200, help="Number of grid points in y.")
    ap.add_argument("--xmin", type=float, default=None, help="Minimum x for grid.")
    ap.add_argument("--xmax", type=float, default=None, help="Maximum x for grid.")
    ap.add_argument("--ymin", type=float, default=None, help="Minimum y for grid.")
    ap.add_argument("--ymax", type=float, default=None, help="Maximum y for grid.")
    ap.add_argument(
        "--interp",
        choices=["idw", "nearest", "rbf"],
        default="idw",
        help="Interpolation method.",
    )
    ap.add_argument("--power", type=float, default=2.0, help="IDW power exponent (interp='idw').")
    ap.add_argument("--out-vtk", default=None, help="Optional output .vts/.vtk file.")
    ap.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open interactive PyVista window.",
    )
    args = ap.parse_args()

    grid = build_map_grid_from_npz(
        args.npz,
        z0=args.z0,
        dz=args.dz,
        nx=args.nx,
        ny=args.ny,
        xmin=args.xmin,
        xmax=args.xmax,
        ymin=args.ymin,
        ymax=args.ymax,
        interp=args.interp,
        power=args.power,
    )

    if args.out_vtk is not None:
        grid.save(args.out_vtk)
        print("Saved map grid to:", args.out_vtk)

    if not args.no_show:
        grid.plot(
            scalars="log10_rho",
            cmap="viridis",
        )


if __name__ == "__main__":  # pragma: no cover
    main()


# ===== End femtic_map_slice_pyvista.py =====

