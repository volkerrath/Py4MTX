#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structural Gramian coupling for joint MT + seismic ADMM inversion.

Self-contained module. Provides:
  - ModelGrid             — model parameter container
  - MultiscaleResampler   — IDW + Gaussian pre-smoothing interpolator
  - build_common_grid     — regular voxel common grid
  - gram_matrix           — 2×2 Gram matrix of two vectors
  - gramian               — determinant of Gram matrix
  - gramian_gradient      — gradient of det(Gram) w.r.t. u and v
  - StructuralGramian     — full coupling class (value / gradient / report)

Inlined from:
  gramian/modules/model_interp.py, gramian/modules/joint_gramian.py
  (entropy/ copies are identical)

References
----------
Zhdanov, M. S., Gribenko, A. V., & Wilson, G. (2012). GRL 39, L09301.
    https://doi.org/10.1029/2012GL051233

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
    out    : bool   — print progress

    Returns
    -------
    common_coords : (M, 3) array
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
# MultiscaleResampler
# =============================================================================

class MultiscaleResampler:
    """
    IDW interpolation from a source grid to a target grid with optional
    Gaussian pre-smoothing (resolution balancing).

    Parameters
    ----------
    source_coords : (N_src, 3)
    target_coords : (N_tgt, 3)
    k        : int    — IDW nearest neighbours
    p        : float  — IDW distance exponent
    sigma    : float  — Gaussian pre-smoothing length scale [m]; 0 = off
    K_smooth : int    — neighbours for smoothing kernel
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
        self._sigma = float(sigma)

        tree_src  = cKDTree(source_coords)
        dist, idx = tree_src.query(target_coords, k=k, workers=-1)
        dist      = np.maximum(dist, 1e-6)
        w         = dist ** (-p)
        self._w_idw = w / w.sum(axis=1, keepdims=True)
        self._idx   = idx

        if self._sigma > 0.0:
            K_smooth      = min(K_smooth, self._n_src)
            dist_s, idx_s = tree_src.query(source_coords, k=K_smooth, workers=-1)
            two_s2        = 2.0 * self._sigma ** 2
            W_s           = np.exp(-(dist_s ** 2) / two_s2)
            self._w_smooth = W_s / W_s.sum(axis=1, keepdims=True)
            self._idx_s    = idx_s
        else:
            self._w_smooth = None
            self._idx_s    = None

    def _smooth(self, values: np.ndarray) -> np.ndarray:
        if self._w_smooth is None:
            return values
        return np.einsum("ij,ij->i", self._w_smooth, values[self._idx_s])

    def __call__(self, values: np.ndarray) -> np.ndarray:
        """Interpolate (N_src,) → (N_tgt,)."""
        return np.einsum("ij,ij->i", self._w_idw, self._smooth(values)[self._idx])

    def _smooth_adjoint(self, g: np.ndarray) -> np.ndarray:
        if self._w_smooth is None:
            return g
        out = np.zeros(self._n_src)
        np.add.at(out, self._idx_s, self._w_smooth * g[:, np.newaxis])
        return out

    def adjoint(self, g_tgt: np.ndarray) -> np.ndarray:
        """Adjoint interpolation (N_tgt,) → (N_src,)."""
        g_smooth = np.zeros(self._n_src)
        np.add.at(g_smooth, self._idx, self._w_idw * g_tgt[:, np.newaxis])
        return self._smooth_adjoint(g_smooth)


# =============================================================================
# Gram matrix utilities
# =============================================================================

def gram_matrix(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """2×2 Gram matrix: G = [[u·u, u·v], [u·v, v·v]]."""
    uu = float(u @ u)
    uv = float(u @ v)
    vv = float(v @ v)
    return np.array([[uu, uv], [uv, vv]], dtype=float)


def gramian(u: np.ndarray, v: np.ndarray) -> float:
    """det(Gram(u, v)) = (u·u)(v·v) − (u·v)²  ≥ 0."""
    return float(np.linalg.det(gram_matrix(u, v)))


def gramian_gradient(u: np.ndarray, v: np.ndarray):
    """
    Gradient of det(Gram(u, v)) w.r.t. u and v.

    ∂det/∂u = 2(v·v)u − 2(u·v)v
    ∂det/∂v = 2(u·u)v − 2(u·v)u

    Returns
    -------
    grad_u, grad_v : ndarray (N,)
    """
    uu = float(u @ u)
    uv = float(u @ v)
    vv = float(v @ v)
    return 2.0 * vv * u - 2.0 * uv * v, 2.0 * uu * v - 2.0 * uv * u


# =============================================================================
# Spatial difference operators
# =============================================================================

def _fd_gradient_magnitude(values: np.ndarray, coords: np.ndarray,
                            k_nn: int = 6) -> np.ndarray:
    """Approximate ‖∇m‖ at each node via K-NN finite differences."""
    from scipy.spatial import cKDTree
    tree       = cKDTree(coords)
    dist, idx  = tree.query(coords, k=k_nn + 1, workers=-1)
    dist       = np.maximum(dist[:, 1:], 1e-6)
    idx        = idx[:, 1:]
    dv         = values[idx] - values[:, np.newaxis]
    return np.sqrt(np.mean((dv / dist) ** 2, axis=1))


def _build_diff_op(mode: str, coords: np.ndarray, k_nn: int = 6):
    if mode == "value":
        return lambda v: v
    elif mode == "gradient":
        return lambda v: _fd_gradient_magnitude(v, coords, k_nn=k_nn)
    elif mode == "laplacian":
        from scipy.spatial import cKDTree
        tree       = cKDTree(coords)
        dist, idx  = tree.query(coords, k=k_nn + 1, workers=-1)
        dist       = np.maximum(dist[:, 1:], 1e-6)
        idx        = idx[:, 1:]
        w          = 1.0 / dist ** 2
        w         /= w.sum(axis=1, keepdims=True)
        return lambda v: np.einsum("ij,ij->i", w, v[idx]) - v
    else:
        raise ValueError(
            f"Unknown Gramian mode '{mode}'; choose 'value', 'gradient', or 'laplacian'."
        )


def _diff_op_vjp(op, x: np.ndarray, g: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """VJP of op w.r.t. x via forward finite differences."""
    fx  = op(x)
    out = np.zeros_like(x)
    for i in range(len(x)):
        xp      = x.copy(); xp[i] += eps
        out[i]  = float(g @ ((op(xp) - fx) / eps))
    return out


# =============================================================================
# StructuralGramian
# =============================================================================

class StructuralGramian:
    """
    Structural Gramian constraint coupling MT and seismic models.

    Objective: β · det(Gram(T(R_mt m_mt), T(R_seis m_seis)))

    Parameters
    ----------
    mt_to_g       : MultiscaleResampler  — MT grid → common grid
    seis_to_g     : MultiscaleResampler  — seismic grid → common grid
    beta          : float                — Gramian weight
    mode          : {"value", "gradient", "laplacian"}
    common_coords : (M, 3) array         — common grid centroids
    k_nn          : int                  — neighbours for FD diff-op

    References
    ----------
    Zhdanov et al. (2012) GRL 39, L09301.
    """

    def __init__(
        self,
        mt_to_g,
        seis_to_g,
        *,
        beta: float = 1.0,
        mode: Literal["value", "gradient", "laplacian"] = "gradient",
        common_coords: np.ndarray,
        k_nn: int = 6,
    ):
        self.mt_to_g   = mt_to_g
        self.seis_to_g = seis_to_g
        self.beta      = float(beta)
        self.mode      = mode
        self._coords   = np.asarray(common_coords, dtype=float)
        self._diff_op  = _build_diff_op(mode, self._coords, k_nn=k_nn)

    def value(self, m_mt: np.ndarray, m_seis: np.ndarray) -> float:
        """β · det(Gram(T(R_mt m_mt), T(R_seis m_seis)))"""
        u = self._diff_op(self.mt_to_g(m_mt))
        v = self._diff_op(self.seis_to_g(m_seis))
        return self.beta * gramian(u, v)

    def gradient(self, m_mt: np.ndarray, m_seis: np.ndarray,
                 eps: float = 1e-5):
        """
        Gradient of Gramian w.r.t. native model vectors.

        Returns
        -------
        grad_mt   : ndarray (N_mt,)
        grad_seis : ndarray (N_seis,)
        """
        u_raw = self.mt_to_g(m_mt)
        v_raw = self.seis_to_g(m_seis)
        u     = self._diff_op(u_raw)
        v     = self._diff_op(v_raw)
        gu, gv = gramian_gradient(u, v)

        if self.mode == "value":
            grad_u_raw = gu
            grad_v_raw = gv
        else:
            grad_u_raw = _diff_op_vjp(self._diff_op, u_raw, gu, eps=eps)
            grad_v_raw = _diff_op_vjp(self._diff_op, v_raw, gv, eps=eps)

        return (
            self.beta * self.mt_to_g.adjoint(grad_u_raw),
            self.beta * self.seis_to_g.adjoint(grad_v_raw),
        )

    def report(self, m_mt: np.ndarray, m_seis: np.ndarray) -> dict:
        """Correlation statistics between the two attribute vectors."""
        u = self._diff_op(self.mt_to_g(m_mt))
        v = self._diff_op(self.seis_to_g(m_seis))
        G    = gram_matrix(u, v)
        det  = float(np.linalg.det(G))
        corr = float(G[0, 1] / np.sqrt(G[0, 0] * G[1, 1] + 1e-30))
        return dict(gramian=det, correlation=corr, gram_matrix=G)
