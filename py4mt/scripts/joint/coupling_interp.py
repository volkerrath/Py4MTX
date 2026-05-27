#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mesh-to-mesh interpolation for joint MT + seismic ADMM inversion.

Self-contained module. Consolidates all interpolation logic used across
the coupling layer:

  - `interpolate_mesh_to_mesh`  — simple point-query mapping
                                  (nearest / IDW / RBF), pure NumPy
  - `MultiscaleResampler`       — KD-tree IDW with Gaussian pre-smoothing
                                  and adjoint operator (gradient chain rule)
  - `build_common_grid`         — regular voxel common grid builder
  - `ModelGrid`                 — model parameter container (dataclass)

Origin of each component:

  `interpolate_mesh_to_mesh` ← crossgrad/interpolation.py
  `MultiscaleResampler`      ← gramian/modules/model_interp.py
                               (entropy/modules/model_interp.py identical)
  `build_common_grid`        ← gramian/modules/model_interp.py
  `ModelGrid`                ← gramian/modules/model_interp.py

Imported by:
  coupling_crossgrad.py  (interpolate_mesh_to_mesh)
  coupling_gramian.py    (MultiscaleResampler, build_common_grid, ModelGrid)
  coupling_entropy.py    (MultiscaleResampler, ModelGrid)

@author:   Volker Rath (DIAS)
@project:  py4mt — Python for Magnetotellurics
@created:  2026-05-12 — VR / Claude Sonnet 4.6 (Anthropic)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


# =============================================================================
# ModelGrid
# =============================================================================

@dataclass
class ModelGrid:
    """Container for a geophysical model parameterisation.

    Parameters
    ----------
    coords : (N, 3) array  — cell centroids [m], z positive-down
    values : (N,) array    — model parameter (log10(Ohm·m) or Vp [km/s])
    name   : str
    """
    coords: np.ndarray
    values: np.ndarray
    name:   str = "model"

    def __post_init__(self):
        self.coords = np.asarray(self.coords, dtype=float)
        self.values = np.asarray(self.values, dtype=float)
        assert self.coords.ndim == 2 and self.coords.shape[1] == 3
        assert self.values.shape == (len(self.coords),)

    @property
    def n(self):
        return len(self.coords)


# =============================================================================
# Common grid builder
# =============================================================================

def build_common_grid(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    *,
    dx: float,
    extent: Optional[list] = None,
    out: bool = True,
) -> np.ndarray:
    """
    Build a regular voxel common grid covering both model extents.

    Parameters
    ----------
    coords_a, coords_b : (N, 3) arrays  — centroids of the two model grids
    dx     : float  — voxel edge length [m]
    extent : [xmin, xmax, ymin, ymax, zmin, zmax] or None (auto)
    out    : bool   — print node count and spacing

    Returns
    -------
    common_coords : (M, 3) array  — voxel centroids
    """
    all_coords = np.vstack([coords_a, coords_b])
    if extent is None:
        lo = all_coords.min(axis=0)
        hi = all_coords.max(axis=0)
    else:
        lo = np.array(extent[0::2], dtype=float)
        hi = np.array(extent[1::2], dtype=float)

    axes = [np.arange(lo[i] + dx / 2, hi[i], dx) for i in range(3)]
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    common_coords = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    if out:
        print(
            f"  build_common_grid: {len(common_coords)} nodes  "
            f"({len(axes[0])}×{len(axes[1])}×{len(axes[2])})  dx={dx/1e3:.1f} km"
        )
    return common_coords


# =============================================================================
# Simple point-query interpolation  (pure NumPy, no adjoint)
# =============================================================================

def _get_cell_centers(mesh: object) -> np.ndarray:
    """Extract cell centres from a mesh object."""
    if hasattr(mesh, "cell_centers"):
        centers = mesh.cell_centers
    elif hasattr(mesh, "get_cell_centers"):
        centers = mesh.get_cell_centers()
    else:
        raise AttributeError(
            "mesh must provide 'cell_centers' or 'get_cell_centers()'"
        )
    centers = np.asarray(centers)
    if centers.ndim != 2:
        raise ValueError("cell_centers must have shape (N, dim)")
    return centers


def _interp_nearest(src_c: np.ndarray, vals: np.ndarray,
                    dst_c: np.ndarray) -> np.ndarray:
    diff = dst_c[:, None, :] - src_c[None, :, :]
    idx  = np.argmin(np.sum(diff ** 2, axis=2), axis=1)
    return vals[idx]


def _interp_idw(src_c: np.ndarray, vals: np.ndarray,
                dst_c: np.ndarray, p: float = 2.0) -> np.ndarray:
    diff = dst_c[:, None, :] - src_c[None, :, :]
    d    = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-15)
    w    = 1.0 / (d ** p)
    return (w @ vals) / (w.sum(axis=1) + 1e-15)


def _interp_rbf(src_c: np.ndarray, vals: np.ndarray,
                dst_c: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    diff = dst_c[:, None, :] - src_c[None, :, :]
    d2   = np.sum(diff ** 2, axis=2)
    w    = np.exp(-d2 / (2.0 * sigma ** 2))
    return (w @ vals) / (w.sum(axis=1) + 1e-15)


def interpolate_mesh_to_mesh(
    src_mesh,
    dst_mesh,
    values: np.ndarray,
    method: Literal["nearest", "idw", "rbf"] = "nearest",
    p: float = 2.0,
    rbf_sigma: float = 1.0,
) -> np.ndarray:
    """
    Interpolate cell-centred values from one mesh to another.

    Simple, pure-NumPy implementation. For large meshes or when a gradient
    chain rule is needed, use `MultiscaleResampler` instead.

    Parameters
    ----------
    src_mesh, dst_mesh : mesh objects
        Must expose ``cell_centers`` (ndarray) or ``get_cell_centers()``.
    values     : ndarray (N_src,)
    method     : {"nearest", "idw", "rbf"}
    p          : float  IDW power exponent (used when method="idw").
    rbf_sigma  : float  Gaussian RBF width in the same units as cell_centers
                        (used when method="rbf").

    Returns
    -------
    ndarray (N_dst,)

    Notes
    -----
    All three methods are O(N_src · N_dst) in memory and time.  For meshes
    with N > ~10 000 cells use `MultiscaleResampler` which builds a KD-tree
    for O(N log N) complexity.
    """
    src_c = _get_cell_centers(src_mesh)
    dst_c = _get_cell_centers(dst_mesh)

    if method == "nearest":
        return _interp_nearest(src_c, values, dst_c)
    elif method == "idw":
        return _interp_idw(src_c, values, dst_c, p=p)
    elif method == "rbf":
        return _interp_rbf(src_c, values, dst_c, sigma=rbf_sigma)
    else:
        raise ValueError(
            f"Unknown interpolation method '{method}'; "
            "choose 'nearest', 'idw', or 'rbf'."
        )


# =============================================================================
# MultiscaleResampler  (KD-tree IDW + Gaussian pre-smoothing + adjoint)
# =============================================================================

class MultiscaleResampler:
    """
    IDW interpolation from a source grid to a target grid with optional
    Gaussian pre-smoothing and a correct adjoint operator.

    This is the preferred interpolator when:
    - Meshes have very different resolutions (use ``sigma`` to balance).
    - A gradient chain rule through the interpolation is required
      (Gramian and MI coupling).
    - N is large (KD-tree gives O(N log N) construction, O(k log N) query).

    The multiscale approach follows Tu & Zhdanov (2021, GJI 226, 1058–1085):
    before computing the Gramian both model vectors are resampled to a shared
    coarse grid using resolution-appropriate Gaussian pre-smoothing sigmas.

    Parameters
    ----------
    source_coords : (N_src, 3)
    target_coords : (N_tgt, 3)
    k        : int    — IDW nearest neighbours
    p        : float  — IDW distance exponent
    sigma    : float  — Gaussian pre-smoothing length scale [m] on source
                        grid; 0 or None = no pre-smoothing
    K_smooth : int    — neighbours for smoothing kernel

    Examples
    --------
    >>> mt_to_g = MultiscaleResampler(mt_coords, common_coords, sigma=3000.)
    >>> u = mt_to_g(m_mt)          # forward:  (N_mt,) → (M,)
    >>> g_mt = mt_to_g.adjoint(gu) # adjoint:  (M,) → (N_mt,)
    """

    def __init__(
        self,
        source_coords: np.ndarray,
        target_coords: np.ndarray,
        *,
        k: int = 8,
        p: float = 2.0,
        sigma: float = 0.0,
        K_smooth: int = 50,
    ):
        from scipy.spatial import cKDTree

        self._n_src = len(source_coords)
        self._n_tgt = len(target_coords)
        self._sigma = float(sigma) if sigma else 0.0

        # IDW weights: source → target
        tree_src  = cKDTree(source_coords)
        dist, idx = tree_src.query(target_coords, k=k, workers=-1)
        dist      = np.maximum(dist, 1e-6)
        w         = dist ** (-p)
        self._w_idw = w / w.sum(axis=1, keepdims=True)   # (N_tgt, k)
        self._idx   = idx                                  # (N_tgt, k)

        # Gaussian pre-smoothing kernel (on source grid)
        if self._sigma > 0.0:
            K_smooth       = min(K_smooth, self._n_src)
            dist_s, idx_s  = tree_src.query(source_coords, k=K_smooth,
                                             workers=-1)
            two_s2         = 2.0 * self._sigma ** 2
            W_s            = np.exp(-(dist_s ** 2) / two_s2)
            self._w_smooth = W_s / W_s.sum(axis=1, keepdims=True)
            self._idx_s    = idx_s
        else:
            self._w_smooth = None
            self._idx_s    = None

    # -- Forward --------------------------------------------------------------

    def _smooth(self, values: np.ndarray) -> np.ndarray:
        """Gaussian pre-smoothing on the source grid."""
        if self._w_smooth is None:
            return values
        return np.einsum("ij,ij->i", self._w_smooth, values[self._idx_s])

    def __call__(self, values: np.ndarray) -> np.ndarray:
        """
        Forward interpolation: (N_src,) → (N_tgt,).

        Applies Gaussian pre-smoothing (if sigma > 0) then IDW.
        """
        return np.einsum("ij,ij->i", self._w_idw,
                         self._smooth(values)[self._idx])

    # -- Adjoint --------------------------------------------------------------

    def _smooth_adjoint(self, g: np.ndarray) -> np.ndarray:
        """Adjoint of Gaussian pre-smoothing."""
        if self._w_smooth is None:
            return g
        out = np.zeros(self._n_src)
        np.add.at(out, self._idx_s, self._w_smooth * g[:, np.newaxis])
        return out

    def adjoint(self, g_tgt: np.ndarray) -> np.ndarray:
        """
        Adjoint interpolation: (N_tgt,) → (N_src,).

        Used in the gradient chain rule:
            ∂Φ/∂m_src = R^T (∂Φ/∂u)
        where R is the (IDW ∘ smooth) operator.
        """
        g_smooth = np.zeros(self._n_src)
        np.add.at(g_smooth, self._idx,
                  self._w_idw * g_tgt[:, np.newaxis])
        return self._smooth_adjoint(g_smooth)
