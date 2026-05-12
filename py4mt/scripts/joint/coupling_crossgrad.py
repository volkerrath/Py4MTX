#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-gradient structural coupling for joint MT + seismic ADMM inversion.

Self-contained module. Provides:
  - GradientMesh              — abstract base class for mesh gradient operators
  - StructuredGridMesh        — finite-difference gradient on regular tensor grid
  - UnstructuredMesh          — wrapper for user-supplied gradient callable
  - interpolate_mesh_to_mesh  — mesh-to-mesh value mapping (nearest / IDW / RBF)
  - compute_cross_gradient    — cross-gradient field on a common/coupling mesh
  - cross_gradient_proximal_term — ADMM proximal shrinkage toward CG null-space

Inlined from:
  crossgrad/cross_gradient.py, crossgrad/mesh_operators.py,
  crossgrad/interpolation.py

@author:   Volker Rath (DIAS)
@project:  py4mt — Python for Magnetotellurics
@created:  2026-05-12 — VR / Claude Sonnet 4.6 (Anthropic)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np


# =============================================================================
# Mesh gradient interface and adapters
# =============================================================================

class GradientMesh(ABC):
    """Abstract base: mesh providing ``grad(model) -> (N, dim)``."""

    @abstractmethod
    def grad(self, m: np.ndarray) -> np.ndarray:
        """
        Gradient of a cell-centred model.

        Parameters
        ----------
        m : ndarray (N,)

        Returns
        -------
        ndarray (N, dim)
        """
        raise NotImplementedError


class StructuredGridMesh(GradientMesh):
    """
    Finite-difference gradient on a regular tensor grid.

    Parameters
    ----------
    shape   : tuple of int    (nx [, ny [, nz]])
    spacing : tuple of float  (dx [, dy [, dz]])
    """

    def __init__(self, shape, spacing):
        self.shape   = shape
        self.spacing = spacing
        self.dim     = len(shape)
        self.N       = int(np.prod(shape))

    def grad(self, m: np.ndarray) -> np.ndarray:
        if m.size != self.N:
            raise ValueError(f"Expected model size {self.N}, got {m.size}")

        m_grid = m.reshape(self.shape)
        grads  = []

        dx = self.spacing[0]
        gx = np.zeros_like(m_grid)
        gx[1:-1, ...] = (m_grid[2:, ...] - m_grid[:-2, ...]) / (2.0 * dx)
        gx[0, ...]    = (m_grid[1, ...]  - m_grid[0, ...])   / dx
        gx[-1, ...]   = (m_grid[-1, ...] - m_grid[-2, ...])  / dx
        grads.append(gx.reshape(-1))

        if self.dim >= 2:
            dy = self.spacing[1]
            gy = np.zeros_like(m_grid)
            gy[:, 1:-1, ...] = (m_grid[:, 2:, ...] - m_grid[:, :-2, ...]) / (2.0 * dy)
            gy[:, 0, ...]    = (m_grid[:, 1, ...]  - m_grid[:, 0, ...])   / dy
            gy[:, -1, ...]   = (m_grid[:, -1, ...] - m_grid[:, -2, ...])  / dy
            grads.append(gy.reshape(-1))

        if self.dim == 3:
            dz = self.spacing[2]
            gz = np.zeros_like(m_grid)
            gz[:, :, 1:-1] = (m_grid[:, :, 2:] - m_grid[:, :, :-2]) / (2.0 * dz)
            gz[:, :, 0]    = (m_grid[:, :, 1]  - m_grid[:, :, 0])   / dz
            gz[:, :, -1]   = (m_grid[:, :, -1] - m_grid[:, :, -2])  / dz
            grads.append(gz.reshape(-1))

        return np.vstack(grads).T   # (N, dim)


class UnstructuredMesh(GradientMesh):
    """
    Unstructured mesh: wraps a user-supplied gradient callable.

    Parameters
    ----------
    grad_operator : callable  ``(N,) -> (N, dim)``
    dim           : int       Spatial dimension.
    N             : int or None  Cell count (for validation).
    """

    def __init__(self, grad_operator, dim: int, N: int | None = None):
        self._grad_operator = grad_operator
        self.dim = dim
        self.N   = N

    def grad(self, m: np.ndarray) -> np.ndarray:
        if self.N is not None and m.size != self.N:
            raise ValueError(f"Expected model size {self.N}, got {m.size}")
        g = self._grad_operator(m)
        if g.ndim != 2 or g.shape[1] != self.dim:
            raise ValueError(
                f"Gradient operator must return (N, {self.dim}), got {g.shape}"
            )
        return g


# =============================================================================
# Mesh-to-mesh interpolation
# =============================================================================

def _get_cell_centers(mesh):
    if hasattr(mesh, "cell_centers"):
        centers = mesh.cell_centers
    elif hasattr(mesh, "get_cell_centers"):
        centers = mesh.get_cell_centers()
    else:
        raise AttributeError("mesh must provide cell_centers or get_cell_centers()")
    centers = np.asarray(centers)
    if centers.ndim != 2:
        raise ValueError("cell_centers must have shape (N, dim)")
    return centers


def _interp_nearest(src_c, vals, dst_c):
    diff = dst_c[:, None, :] - src_c[None, :, :]
    idx  = np.argmin(np.sum(diff ** 2, axis=2), axis=1)
    return vals[idx]


def _interp_idw(src_c, vals, dst_c, p=2.0):
    diff = dst_c[:, None, :] - src_c[None, :, :]
    d    = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-15)
    w    = 1.0 / (d ** p)
    return (w @ vals) / (w.sum(axis=1) + 1e-15)


def _interp_rbf(src_c, vals, dst_c, sigma=1.0):
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

    Parameters
    ----------
    src_mesh, dst_mesh : mesh objects with ``cell_centers`` or
                         ``get_cell_centers()``
    values     : ndarray (N_src,)
    method     : {"nearest", "idw", "rbf"}
    p          : float  IDW power exponent.
    rbf_sigma  : float  Gaussian RBF width [same units as cell_centers].

    Returns
    -------
    ndarray (N_dst,)
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
        raise ValueError(f"Unknown interpolation method '{method}'")


# =============================================================================
# Cross-gradient
# =============================================================================

def _cross_gradient_single_mesh(m_mt, m_sv, mesh):
    """Cross-gradient X = ∇m_mt × ∇m_sv on a single mesh."""
    g_mt = mesh.grad(m_mt)
    g_sv = mesh.grad(m_sv)

    if g_mt.shape != g_sv.shape:
        raise ValueError("MT and seismic gradients must have the same shape")

    if g_mt.shape[1] == 2:
        g_mt = np.c_[g_mt, np.zeros(g_mt.shape[0])]
        g_sv = np.c_[g_sv, np.zeros(g_sv.shape[0])]

    elif g_mt.shape[1] != 3:
        raise ValueError("Gradient must have 2 or 3 components per cell")

    return np.cross(g_mt, g_sv)   # (N, 3)


def compute_cross_gradient(
    m_mt,
    mesh_mt,
    m_sv,
    mesh_sv,
    coupling_mesh=None,
    interp_method: Literal["nearest", "idw", "rbf"] = "nearest",
    interp_power: float = 2.0,
):
    """
    Compute the cross-gradient X = ∇m_mt × ∇m_sv on a coupling mesh.

    When MT and seismic live on different meshes, one or both are
    interpolated to ``coupling_mesh`` before the gradient computation.

    Parameters
    ----------
    m_mt, m_sv       : ndarray   Model vectors on their native meshes.
    mesh_mt, mesh_sv : GradientMesh (or compatible)
    coupling_mesh    : mesh or None  Defaults to ``mesh_sv``.
    interp_method    : {"nearest", "idw", "rbf"}
    interp_power     : float  IDW exponent.

    Returns
    -------
    X             : ndarray (N_c, 3)  Cross-gradient on the coupling mesh.
    coupling_mesh : mesh              The mesh on which X is defined.
    """
    if coupling_mesh is None:
        coupling_mesh = mesh_sv

    if coupling_mesh is mesh_mt:
        m_mt_c = m_mt
        m_sv_c = interpolate_mesh_to_mesh(
            mesh_sv, mesh_mt, m_sv, method=interp_method, p=interp_power
        )
    elif coupling_mesh is mesh_sv:
        m_mt_c = interpolate_mesh_to_mesh(
            mesh_mt, mesh_sv, m_mt, method=interp_method, p=interp_power
        )
        m_sv_c = m_sv
    else:
        m_mt_c = interpolate_mesh_to_mesh(
            mesh_mt, coupling_mesh, m_mt, method=interp_method, p=interp_power
        )
        m_sv_c = interpolate_mesh_to_mesh(
            mesh_sv, coupling_mesh, m_sv, method=interp_method, p=interp_power
        )

    X = _cross_gradient_single_mesh(m_mt_c, m_sv_c, coupling_mesh)
    return X, coupling_mesh


def cross_gradient_proximal_term(
    z: np.ndarray,
    m_target: np.ndarray,
    weight: float,
) -> np.ndarray:
    """
    ADMM-compatible proximal update nudging ``z`` toward ``m_target``.

    Implements:   z ← (z + weight · m_target) / (1 + weight)

    Parameters
    ----------
    z        : ndarray (N,)  Current consensus variable.
    m_target : ndarray (N,)  Target model (e.g. m_mt or m_sv).
    weight   : float         Proximal weight; 0 = no effect.

    Returns
    -------
    ndarray (N,)
    """
    return (z + weight * m_target) / (1.0 + weight)
