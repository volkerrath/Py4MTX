#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADMM coupling wrappers for joint MT + seismic tomography inversion.

Each class in this module adapts one joint regularisation strategy to the
interface expected by ``admm_joint_mt_seis`` in ``joint_admm_driver.py``:

    coupling.update_z(m_mt, m_sv, y_mt, y_sv, rho_mt, rho_sv) -> z
    coupling.report(m_mt, m_sv, z)                             -> dict

Four strategies are provided:

    FCMCoupling
        Fuzzy C-means latent petrophysical field.  ``z`` is updated in
        closed form; memberships and centroids are refined iteratively.

    CrossGradientCoupling
        Structural cross-gradient proximal.  ``z`` is the ADMM consensus
        (weighted mean); the cross-gradient penalty enters via a proximal
        shrinkage step applied after the mean.

    GramianCoupling
        Structural Gramian (Zhdanov 2012).  ``z`` is the ADMM consensus;
        the Gramian gradient is exposed through ``report`` for diagnostic
        use and can be injected into the model solvers by the caller.

    MutualInfoCoupling
        Mutual-information / entropy coupling.  Same consensus-mean
        approach as Gramian; MI gradient available through ``report``.

All four can be composed with ``CombinedCoupling``:

    coupling = CombinedCoupling([
        FCMCoupling(...),
        CrossGradientCoupling(...),
    ])

@author:   Volker Rath (DIAS)
@project:  py4mt — Python for Magnetotellurics
@created:  2026-05-12 — VR / Claude Sonnet 4.6 (Anthropic)
"""

from __future__ import annotations

import numpy as np


# =============================================================================
# FCMCoupling
# =============================================================================

class FCMCoupling:
    """
    ADMM coupling via Fuzzy C-Means latent petrophysical field.

    ``z`` is updated in closed form by minimising the FCM objective plus
    the ADMM quadratic penalty terms.  Memberships ``U`` and centroids
    ``c`` are refined for ``n_inner`` sweeps per ``update_z`` call.

    Imports ``update_centroids``, ``update_memberships`` from
    ``fcm.fcm`` and ``update_z_coupling_mesh`` from ``fcm.latent_field``.

    Parameters
    ----------
    K : int
        Number of FCM clusters.
    N : int
        Number of model cells (size of m_mt / m_sv).
    beta : float, optional
        FCM coupling weight. Default 1.0.
    q : float, optional
        Fuzziness exponent. Default 2.0.
    w_mt, w_sv : float, optional
        Relative ADMM weights for MT and seismic contributions
        (should sum to 1). Default 0.5 each.
    n_inner : int, optional
        Number of FCM sweeps per ADMM iteration. Default 2.
    """

    def __init__(self, K, N, *, beta=1.0, q=2.0, w_mt=0.5, w_sv=0.5,
                 n_inner=2):
        from fcm.fcm import update_centroids, update_memberships
        from fcm.latent_field import update_z_coupling_mesh

        self._update_z_fcm   = update_z_coupling_mesh
        self._update_c       = update_centroids
        self._update_U       = update_memberships

        self.beta    = float(beta)
        self.q       = float(q)
        self.w_mt    = float(w_mt)
        self.w_sv    = float(w_sv)
        self.n_inner = int(n_inner)

        # Initialise FCM state
        self.U = np.full((N, K), 1.0 / K)
        self.c = None   # set on first update_z call from z range

    def update_z(self, m_mt, m_sv, y_mt, y_sv, rho_mt, rho_sv):
        """
        Update latent field z, memberships U, and centroids c.

        Parameters
        ----------
        m_mt, m_sv : ndarray (N,)
        y_mt, y_sv : ndarray (N,)
        rho_mt, rho_sv : float

        Returns
        -------
        z : ndarray (N,)
        """
        z = 0.5 * (m_mt + m_sv)

        if self.c is None:
            K = self.U.shape[1]
            self.c = np.linspace(z.min(), z.max(), K)

        for _ in range(self.n_inner):
            z = self._update_z_fcm(
                m_mt, m_sv, y_mt, y_sv,
                self.U, self.c,
                self.beta, rho_mt, rho_sv, self.q,
                w_mt=self.w_mt, w_sv=self.w_sv,
            )
            self.c = self._update_c(z, self.U, self.q)
            self.U = self._update_U(z, self.c, self.q)

        return z

    def report(self, m_mt, m_sv, z):
        """Return FCM diagnostic scalars."""
        r_mt = float(np.linalg.norm(m_mt - z))
        r_sv = float(np.linalg.norm(m_sv - z))
        # Weighted intra-cluster variance
        Um   = self.U ** self.q
        var  = float((Um * (z[:, None] - self.c[None, :]) ** 2).sum())
        return dict(r_mt=r_mt, r_sv=r_sv, fcm_var=var)


# =============================================================================
# CrossGradientCoupling
# =============================================================================

class CrossGradientCoupling:
    """
    ADMM coupling via structural cross-gradient proximal regularisation.

    The consensus variable ``z`` is the weighted ADMM mean.  A proximal
    shrinkage step towards the cross-gradient null-space is then applied.

    Imports ``compute_cross_gradient``, ``cross_gradient_proximal_term``
    from ``crossgrad.cross_gradient``.

    Parameters
    ----------
    mesh : object
        Mesh providing ``mesh.grad(model)`` for cross-gradient computation.
    cg_weight : float, optional
        Cross-gradient proximal weight. 0 reduces to pure consensus mean.
        Default 0.1.
    w_mt, w_sv : float, optional
        Weights for the consensus mean (should sum to 1). Default 0.5.
    """

    def __init__(self, mesh, *, cg_weight=0.1, w_mt=0.5, w_sv=0.5):
        from crossgrad.cross_gradient import (
            compute_cross_gradient,
            cross_gradient_proximal_term,
        )
        self._compute_cg  = compute_cross_gradient
        self._proximal    = cross_gradient_proximal_term
        self.mesh         = mesh
        self.cg_weight    = float(cg_weight)
        self.w_mt         = float(w_mt)
        self.w_sv         = float(w_sv)

    def update_z(self, m_mt, m_sv, y_mt, y_sv, rho_mt, rho_sv):
        """
        Consensus mean + cross-gradient proximal step.

        Parameters
        ----------
        m_mt, m_sv : ndarray (N,)
        y_mt, y_sv : ndarray (N,)
        rho_mt, rho_sv : float

        Returns
        -------
        z : ndarray (N,)
        """
        # Weighted ADMM consensus mean
        num = self.w_mt * rho_mt * (m_mt + y_mt / rho_mt) \
            + self.w_sv * rho_sv * (m_sv + y_sv / rho_sv)
        den = self.w_mt * rho_mt + self.w_sv * rho_sv
        z   = num / den

        # Cross-gradient proximal shrinkage toward structural consistency
        if self.cg_weight > 0.0:
            z = self._proximal(z, m_mt, self.cg_weight)

        return z

    def report(self, m_mt, m_sv, z):
        """Return cross-gradient magnitude diagnostic."""
        X     = self._compute_cg(m_mt, self.mesh, m_sv, self.mesh)
        cg_rms = float(np.sqrt(np.mean(np.sum(X ** 2, axis=-1))))
        return dict(cg_rms=cg_rms)


# =============================================================================
# GramianCoupling
# =============================================================================

class GramianCoupling:
    """
    ADMM coupling via structural Gramian (Zhdanov 2012).

    The consensus variable ``z`` is the plain ADMM weighted mean.  The
    Gramian constraint enters through the model-update gradients; this
    wrapper exposes gradient information via ``report`` so the caller can
    inject it into the physics solvers if desired.

    Wraps ``StructuralGramian`` from ``gramian.modules.joint_gramian``.

    Parameters
    ----------
    gramian : StructuralGramian
        Configured ``StructuralGramian`` instance (see
        ``gramian/modules/joint_gramian.py``).
    w_mt, w_sv : float, optional
        Weights for the consensus mean. Default 0.5.
    """

    def __init__(self, gramian, *, w_mt=0.5, w_sv=0.5):
        self._gramian = gramian
        self.w_mt     = float(w_mt)
        self.w_sv     = float(w_sv)

    def update_z(self, m_mt, m_sv, y_mt, y_sv, rho_mt, rho_sv):
        """
        Weighted ADMM consensus mean.

        Parameters
        ----------
        m_mt, m_sv : ndarray (N,)
        y_mt, y_sv : ndarray (N,)
        rho_mt, rho_sv : float

        Returns
        -------
        z : ndarray (N,)
        """
        num = self.w_mt * rho_mt * (m_mt + y_mt / rho_mt) \
            + self.w_sv * rho_sv * (m_sv + y_sv / rho_sv)
        den = self.w_mt * rho_mt + self.w_sv * rho_sv
        return num / den

    def gradient(self, m_mt, m_sv):
        """
        Gramian gradients on native model grids.

        Returns
        -------
        grad_mt, grad_sv : ndarray
        """
        return self._gramian.gradient(m_mt, m_sv)

    def report(self, m_mt, m_sv, z):
        """Return Gramian diagnostic scalars."""
        diag = self._gramian.report(m_mt, m_sv)
        diag["r_consensus"] = float(
            np.linalg.norm(0.5 * (m_mt + m_sv) - z)
        )
        return diag


# =============================================================================
# MutualInfoCoupling
# =============================================================================

class MutualInfoCoupling:
    """
    ADMM coupling via mutual-information / entropy coupling.

    The consensus variable ``z`` is the plain ADMM weighted mean.  The
    MI penalty enters through the model-update gradients; this wrapper
    exposes gradient and MI value via ``report``.

    Wraps ``MutualInformationCoupling`` from
    ``entropy.modules.cross_entropy_coupling``.

    Parameters
    ----------
    mi_coupling : MutualInformationCoupling
        Configured ``MutualInformationCoupling`` instance.
    w_mt, w_sv : float, optional
        Weights for the consensus mean. Default 0.5.
    """

    def __init__(self, mi_coupling, *, w_mt=0.5, w_sv=0.5):
        self._mi  = mi_coupling
        self.w_mt = float(w_mt)
        self.w_sv = float(w_sv)

    def update_z(self, m_mt, m_sv, y_mt, y_sv, rho_mt, rho_sv):
        """
        Weighted ADMM consensus mean.

        Parameters
        ----------
        m_mt, m_sv : ndarray (N,)
        y_mt, y_sv : ndarray (N,)
        rho_mt, rho_sv : float

        Returns
        -------
        z : ndarray (N,)
        """
        num = self.w_mt * rho_mt * (m_mt + y_mt / rho_mt) \
            + self.w_sv * rho_sv * (m_sv + y_sv / rho_sv)
        den = self.w_mt * rho_mt + self.w_sv * rho_sv
        return num / den

    def gradient(self, m_mt, m_sv):
        """
        MI gradients on native model grids (negated: maximise MI).

        Returns
        -------
        grad_mt, grad_sv : ndarray
        """
        return self._mi.gradient(m_mt, m_sv)

    def report(self, m_mt, m_sv, z):
        """Return MI diagnostic scalars."""
        diag = self._mi.report(m_mt, m_sv)
        diag["r_consensus"] = float(
            np.linalg.norm(0.5 * (m_mt + m_sv) - z)
        )
        return diag


# =============================================================================
# CombinedCoupling
# =============================================================================

class CombinedCoupling:
    """
    Weighted sum of multiple coupling strategies.

    ``update_z`` returns the weighted average of each component's ``z``
    update.  ``report`` merges all component diagnostics, prefixing keys
    with the component index to avoid collisions.

    Parameters
    ----------
    components : list of coupling objects
        Each must expose ``update_z``; ``report`` is optional.
    weights : list of float or None, optional
        Per-component weights for the ``z`` average.  If ``None``,
        components are weighted equally.

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
        """
        Weighted average of component ``z`` updates.

        Parameters
        ----------
        m_mt, m_sv : ndarray (N,)
        y_mt, y_sv : ndarray (N,)
        rho_mt, rho_sv : float

        Returns
        -------
        z : ndarray (N,)
        """
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
