#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADMM coupling wrappers for joint MT + seismic tomography inversion.

Each class adapts one joint regularisation strategy to the interface
expected by ``admm_joint_mt_seis`` in ``joint_admm_driver.py``:

    coupling.update_z(m_mt, m_sv, y_mt, y_sv, rho_mt, rho_sv) -> z
    coupling.report(m_mt, m_sv, z)                             -> dict

Four strategies are provided, with two FCM variants:

    FCMCoupling            (fuzzy C-means; z on same mesh as m_mt/m_sv)
    FCMCouplingMesh        (fuzzy C-means; z on a dedicated coupling mesh)
    CrossGradientCoupling  (structural cross-gradient proximal)
    GramianCoupling        (structural Gramian)
    MutualInfoCoupling     (entropy / mutual-information)
    CombinedCoupling       (weighted average of any of the above)

Imports from the four self-contained sibling modules:
    coupling_fcm.py, coupling_crossgrad.py,
    coupling_gramian.py, coupling_entropy.py, coupling_interp.py

@author:   Volker Rath (DIAS)
@project:  py4mt — Python for Magnetotellurics
@created:  2026-05-12 — VR / Claude Sonnet 4.6 (Anthropic)
@modified: 2026-05-12 — imports from consolidated coupling_*.py modules,
           VR / Claude Sonnet 4.6 (Anthropic)
"""

from __future__ import annotations

import numpy as np

from coupling_fcm import (
    update_centroids, update_memberships,
    update_z_fcm, update_z_fcm_on_mesh,
)
from coupling_interp import interpolate_mesh_to_mesh
from coupling_crossgrad import compute_cross_gradient, cross_gradient_proximal_term
from coupling_gramian import StructuralGramian
from coupling_entropy import MutualInformationCoupling


# =============================================================================
# FCMCoupling
# =============================================================================

class FCMCoupling:
    """
    ADMM coupling via Fuzzy C-Means latent petrophysical field.

    ``z`` is updated in closed form by minimising the FCM objective plus
    the ADMM quadratic penalty terms.  Memberships ``U`` and centroids
    ``c`` are refined for ``n_inner`` sweeps per ``update_z`` call.

    Parameters
    ----------
    K       : int    Number of FCM clusters.
    N       : int    Number of model cells.
    beta    : float  FCM coupling weight. Default 1.0.
    q       : float  Fuzziness exponent. Default 2.0.
    w_mt, w_sv : float  Relative ADMM weights (sum to 1). Default 0.5.
    n_inner : int    FCM sweeps per ADMM iteration. Default 2.
    """

    def __init__(self, K, N, *, beta=1.0, q=2.0, w_mt=0.5, w_sv=0.5,
                 n_inner=2):
        self.beta    = float(beta)
        self.q       = float(q)
        self.w_mt    = float(w_mt)
        self.w_sv    = float(w_sv)
        self.n_inner = int(n_inner)

        self.U = np.full((N, K), 1.0 / K)
        self.c = None   # initialised on first call from z range

    def update_z(self, m_mt, m_sv, y_mt, y_sv, rho_mt, rho_sv):
        """Update latent field z, memberships U, and centroids c."""
        z = 0.5 * (m_mt + m_sv)

        if self.c is None:
            K      = self.U.shape[1]
            self.c = np.linspace(z.min(), z.max(), K)

        for _ in range(self.n_inner):
            z      = update_z_fcm(
                m_mt, m_sv, y_mt, y_sv,
                self.U, self.c,
                self.beta, rho_mt, rho_sv, self.q,
                w_mt=self.w_mt, w_sv=self.w_sv,
            )
            self.c = update_centroids(z, self.U, self.q)
            self.U = update_memberships(z, self.c, self.q)

        return z

    def report(self, m_mt, m_sv, z):
        """FCM diagnostic scalars."""
        Um  = self.U ** self.q
        var = float((Um * (z[:, None] - self.c[None, :]) ** 2).sum())
        return dict(
            r_mt=float(np.linalg.norm(m_mt - z)),
            r_sv=float(np.linalg.norm(m_sv - z)),
            fcm_var=var,
        )


# =============================================================================
# CrossGradientCoupling
# =============================================================================

class CrossGradientCoupling:
    """
    ADMM coupling via structural cross-gradient proximal regularisation.

    The consensus variable ``z`` is the weighted ADMM mean followed by a
    cross-gradient proximal shrinkage step.

    Parameters
    ----------
    mesh      : GradientMesh  Mesh for cross-gradient computation.
    cg_weight : float         Proximal weight; 0 = pure consensus mean.
    w_mt, w_sv : float        Consensus mean weights (sum to 1).
    """

    def __init__(self, mesh, *, cg_weight=0.1, w_mt=0.5, w_sv=0.5):
        self.mesh      = mesh
        self.cg_weight = float(cg_weight)
        self.w_mt      = float(w_mt)
        self.w_sv      = float(w_sv)

    def update_z(self, m_mt, m_sv, y_mt, y_sv, rho_mt, rho_sv):
        """Weighted ADMM consensus mean + cross-gradient proximal step."""
        num = (self.w_mt * rho_mt * (m_mt + y_mt / rho_mt)
               + self.w_sv * rho_sv * (m_sv + y_sv / rho_sv))
        den = self.w_mt * rho_mt + self.w_sv * rho_sv
        z   = num / den

        if self.cg_weight > 0.0:
            z = cross_gradient_proximal_term(z, m_mt, self.cg_weight)

        return z

    def report(self, m_mt, m_sv, z):
        """Cross-gradient RMS diagnostic."""
        X, _ = compute_cross_gradient(m_mt, self.mesh, m_sv, self.mesh)
        return dict(cg_rms=float(np.sqrt(np.mean(np.sum(X ** 2, axis=-1)))))


# =============================================================================
# GramianCoupling
# =============================================================================

class GramianCoupling:
    """
    ADMM coupling via structural Gramian (Zhdanov 2012).

    ``z`` is the plain ADMM weighted mean.  The Gramian gradient is
    available via ``.gradient()`` for injection into the physics solvers.

    Parameters
    ----------
    gramian   : StructuralGramian  Configured Gramian instance.
    w_mt, w_sv : float             Consensus mean weights.
    """

    def __init__(self, gramian: StructuralGramian, *, w_mt=0.5, w_sv=0.5):
        self._gramian = gramian
        self.w_mt     = float(w_mt)
        self.w_sv     = float(w_sv)

    def update_z(self, m_mt, m_sv, y_mt, y_sv, rho_mt, rho_sv):
        """Weighted ADMM consensus mean."""
        num = (self.w_mt * rho_mt * (m_mt + y_mt / rho_mt)
               + self.w_sv * rho_sv * (m_sv + y_sv / rho_sv))
        den = self.w_mt * rho_mt + self.w_sv * rho_sv
        return num / den

    def gradient(self, m_mt, m_sv):
        """Gramian gradients on native model grids (for solver injection)."""
        return self._gramian.gradient(m_mt, m_sv)

    def report(self, m_mt, m_sv, z):
        """Gramian diagnostic scalars."""
        diag = self._gramian.report(m_mt, m_sv)
        diag["r_consensus"] = float(np.linalg.norm(0.5 * (m_mt + m_sv) - z))
        return diag


# =============================================================================
# MutualInfoCoupling
# =============================================================================

class MutualInfoCoupling:
    """
    ADMM coupling via mutual-information / entropy coupling.

    ``z`` is the plain ADMM weighted mean.  The MI gradient is available
    via ``.gradient()`` for injection into the physics solvers.

    Parameters
    ----------
    mi_coupling : MutualInformationCoupling  Configured MI instance.
    w_mt, w_sv  : float                      Consensus mean weights.
    """

    def __init__(self, mi_coupling: MutualInformationCoupling, *,
                 w_mt=0.5, w_sv=0.5):
        self._mi  = mi_coupling
        self.w_mt = float(w_mt)
        self.w_sv = float(w_sv)

    def update_z(self, m_mt, m_sv, y_mt, y_sv, rho_mt, rho_sv):
        """Weighted ADMM consensus mean."""
        num = (self.w_mt * rho_mt * (m_mt + y_mt / rho_mt)
               + self.w_sv * rho_sv * (m_sv + y_sv / rho_sv))
        den = self.w_mt * rho_mt + self.w_sv * rho_sv
        return num / den

    def gradient(self, m_mt, m_sv):
        """MI gradients on native model grids (for solver injection)."""
        return self._mi.gradient(m_mt, m_sv)

    def report(self, m_mt, m_sv, z):
        """MI diagnostic scalars."""
        diag = self._mi.report(m_mt, m_sv)
        diag["r_consensus"] = float(np.linalg.norm(0.5 * (m_mt + m_sv) - z))
        return diag


# =============================================================================
# FCMCouplingMesh
# =============================================================================

class FCMCouplingMesh:
    """
    ADMM coupling via FCM with a dedicated coupling mesh.

    ``z`` lives on ``mesh_z`` (which may differ from the MT and seismic
    meshes).  Each ``update_z`` call:

    1. Interpolates ``m_mt``, ``m_sv``, and the duals from their native
       meshes to ``mesh_z``.
    2. Runs ``n_inner`` FCM sweeps on the coupling mesh.
    3. Returns ``z`` on ``mesh_z``; the driver interpolates it back to
       the physics meshes when building the model-update RHS.

    Duals ``y_mt_c`` and ``y_sv_c`` are maintained on ``mesh_z`` as
    instance state between iterations.

    Parameters
    ----------
    K        : int    Number of FCM clusters.
    N_c      : int    Number of cells on the coupling mesh (``mesh_z``).
    mesh_mt  : object MT mesh (must expose ``cell_centers``).
    mesh_sv  : object Seismic mesh.
    mesh_z   : object Coupling mesh on which z is defined.
    beta     : float  FCM coupling weight. Default 1.0.
    q        : float  Fuzziness exponent. Default 2.0.
    w_mt, w_sv : float Relative ADMM weights (sum to 1). Default 0.5.
    n_inner  : int    FCM sweeps per ADMM iteration. Default 2.
    interp_method : {"nearest", "idw", "rbf"}  Interpolation method.
    interp_power  : float  IDW exponent. Default 2.0.
    """

    def __init__(
        self,
        K,
        N_c,
        mesh_mt,
        mesh_sv,
        mesh_z,
        *,
        beta=1.0,
        q=2.0,
        w_mt=0.5,
        w_sv=0.5,
        n_inner=2,
        interp_method="nearest",
        interp_power=2.0,
    ):
        self.mesh_mt  = mesh_mt
        self.mesh_sv  = mesh_sv
        self.mesh_z   = mesh_z
        self.beta     = float(beta)
        self.q        = float(q)
        self.w_mt     = float(w_mt)
        self.w_sv     = float(w_sv)
        self.n_inner  = int(n_inner)
        self.interp_method = interp_method
        self.interp_power  = float(interp_power)

        # FCM state on coupling mesh
        self.U     = np.full((N_c, K), 1.0 / K)
        self.c     = None   # initialised on first update_z call

        # Duals on coupling mesh (carried across iterations)
        self.y_mt_c = np.zeros(N_c)
        self.y_sv_c = np.zeros(N_c)

    def _to_z(self, values, src_mesh):
        return interpolate_mesh_to_mesh(
            src_mesh, self.mesh_z, values,
            method=self.interp_method, p=self.interp_power,
        )

    def update_z(self, m_mt, m_sv, y_mt, y_sv, rho_mt, rho_sv):
        """
        Update latent field z on the coupling mesh.

        Interpolates models and duals to ``mesh_z``, runs FCM, updates
        coupling-mesh duals, and returns ``z`` on ``mesh_z``.

        Parameters
        ----------
        m_mt, m_sv : ndarray  Models on their native meshes.
        y_mt, y_sv : ndarray  Duals on their native meshes (not used
                               directly; coupling-mesh duals are maintained
                               as instance state in ``y_mt_c``/``y_sv_c``).
        rho_mt, rho_sv : float

        Returns
        -------
        z : ndarray (N_c,)  Latent field on ``mesh_z``.
        """
        m_mt_c = self._to_z(m_mt, self.mesh_mt)
        m_sv_c = self._to_z(m_sv, self.mesh_sv)

        if self.c is None:
            z_init = 0.5 * (m_mt_c + m_sv_c)
            K = self.U.shape[1]
            self.c = np.linspace(z_init.min(), z_init.max(), K)

        for _ in range(self.n_inner):
            z = update_z_fcm_on_mesh(
                m_mt_c, m_sv_c,
                self.y_mt_c, self.y_sv_c,
                self.U, self.c,
                self.beta, rho_mt, rho_sv, self.q,
                w_mt=self.w_mt, w_sv=self.w_sv,
            )
            self.c = update_centroids(z, self.U, self.q)
            self.U = update_memberships(z, self.c, self.q)

        # Update coupling-mesh duals
        r_mt_c = m_mt_c - z
        r_sv_c = m_sv_c - z
        self.y_mt_c += rho_mt * r_mt_c
        self.y_sv_c += rho_sv * r_sv_c

        return z

    def report(self, m_mt, m_sv, z):
        """FCM coupling-mesh diagnostic scalars."""
        m_mt_c = self._to_z(m_mt, self.mesh_mt)
        m_sv_c = self._to_z(m_sv, self.mesh_sv)
        Um  = self.U ** self.q
        var = float((Um * (z[:, None] - self.c[None, :]) ** 2).sum())
        return dict(
            r_mt_c=float(np.linalg.norm(m_mt_c - z)),
            r_sv_c=float(np.linalg.norm(m_sv_c - z)),
            fcm_var=var,
        )


# =============================================================================
# CombinedCoupling
# =============================================================================

class CombinedCoupling:
    """
    Weighted sum of multiple coupling strategies.

    ``update_z`` returns the weighted average of each component's ``z``
    update.  ``report`` merges all component diagnostics, prefixing keys
    with ``c{i}_`` to avoid collisions.

    Parameters
    ----------
    components : list   Coupling objects, each exposing ``update_z``.
    weights    : list of float or None  Equal weights if None.

    Examples
    --------
    >>> coupling = CombinedCoupling([
    ...     FCMCoupling(K=3, N=N, beta=1.0),
    ...     CrossGradientCoupling(mesh, cg_weight=0.05),
    ... ])
    """

    def __init__(self, components, *, weights=None):
        self.components = list(components)
        if weights is None:
            n = len(self.components)
            self.weights = [1.0 / n] * n
        else:
            w = np.asarray(weights, dtype=float)
            self.weights = list(w / w.sum())

    def update_z(self, m_mt, m_sv, y_mt, y_sv, rho_mt, rho_sv):
        """Weighted average of component z updates."""
        z = np.zeros_like(m_mt)
        for w, comp in zip(self.weights, self.components):
            z += w * comp.update_z(m_mt, m_sv, y_mt, y_sv, rho_mt, rho_sv)
        return z

    def report(self, m_mt, m_sv, z):
        """Merged diagnostics from all components."""
        out = {}
        for i, comp in enumerate(self.components):
            if hasattr(comp, "report"):
                for k, v in comp.report(m_mt, m_sv, z).items():
                    out[f"c{i}_{k}"] = v
        return out
