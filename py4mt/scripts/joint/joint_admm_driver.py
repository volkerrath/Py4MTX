#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADMM driver for joint MT + seismic tomography inversion.

Main ADMM outer loop controlling the coupled MT + seismic inversion
workflow. Handles:
  - MT model update (optionally fixed)
  - Seismic model update (optionally fixed)
  - Latent / consensus variable update via a user-supplied coupling object
  - Dual variable updates
  - Primal/dual residual tracking and stopping criteria

The driver is agnostic with respect to the joint regularisation strategy.
Any coupling object exposing the interface below is accepted:

    coupling.update_z(m_mt, m_sv, y_mt, y_sv, rho_mt, rho_sv)
        → z  (ndarray, same shape as m_mt/m_sv)
    coupling.report(m_mt, m_sv, z)
        → dict of diagnostic scalars   (optional; used when verbose=True)

Coupling classes implementing this interface are in ``joint_coupling.py``:

    FCMCoupling            (fuzzy C-means latent field)
    CrossGradientCoupling  (structural cross-gradient proximal)
    GramianCoupling        (structural Gramian)
    MutualInfoCoupling     (entropy / mutual-information)
    CombinedCoupling       (weighted sum of several of the above)

@author:   Volker Rath (DIAS)
@project:  py4mt — Python for Magnetotellurics
@created:  2026-05-12 with the help of Copilot
@modified: 2026-05-12 — py4mt header, VR / Claude Sonnet 4.6 (Anthropic)
@modified: 2026-05-12 — coupling-agnostic, fix_method option,
           VR / Claude Sonnet 4.6 (Anthropic)
@modified: 2026-05-12 — docstring updated to reference joint_coupling.py,
           VR / Claude Sonnet 4.6 (Anthropic)
"""

from __future__ import annotations

import numpy as np


def admm_joint_mt_seis(
    d_mt,
    d_sv,
    Wd_mt,
    Wd_sv,
    Wm_mt,
    Wm_sv,
    m_mt0,
    m_sv0,
    m_mt_ref,
    m_sv_ref,
    coupling,
    *,
    alpha_mt=1.0,
    alpha_sv=1.0,
    rho_mt=1.0,
    rho_sv=1.0,
    max_outer=50,
    tol_primal=1e-3,
    tol_dual=1e-3,
    fix_method=None,
    verbose=True,
    solve_m_mt=None,
    solve_m_sv=None,
    apply_Gt_mt=None,
    apply_Gt_sv=None,
):
    """
    Joint MT + seismic inversion via ADMM with pluggable coupling.

    The coupling strategy (FCM, cross-gradient, Gramian, mutual information,
    or any combination) is supplied as a ``coupling`` object and is not
    hard-coded into the loop.  Either physics method can optionally be held
    fixed while the other and the consensus variable are updated.

    Parameters
    ----------
    d_mt, d_sv : ndarray
        MT and seismic observed data vectors.
    Wd_mt, Wd_sv : operators
        Data-weighting operators for MT and seismic data.
    Wm_mt, Wm_sv : operators
        Tikhonov regularisation operators for MT and seismic models.
    m_mt0, m_sv0 : ndarray, shape (N,)
        Initial MT and seismic model vectors.
    m_mt_ref, m_sv_ref : ndarray, shape (N,)
        Reference models for Tikhonov regularisation.
    coupling : object
        Coupling strategy object.  Must expose:

        ``coupling.update_z(m_mt, m_sv, y_mt, y_sv, rho_mt, rho_sv) -> z``
            Closed-form or iterative update of the consensus / latent
            variable ``z``.

        ``coupling.report(m_mt, m_sv, z) -> dict``  *(optional)*
            Diagnostic scalars printed when ``verbose=True``.

    alpha_mt, alpha_sv : float, optional
        Regularisation weights for MT and seismic. Default 1.0.
    rho_mt, rho_sv : float, optional
        ADMM penalty parameters. Default 1.0.
    max_outer : int, optional
        Maximum number of ADMM outer iterations. Default 50.
    tol_primal, tol_dual : float, optional
        Stopping tolerances for primal and dual residuals. Default 1e-3.
    fix_method : {None, "mt", "sv"}, optional
        Hold one physics method fixed throughout the ADMM loop:

        ``None``   — both methods updated every iteration (default).
        ``"mt"``   — MT model is kept at ``m_mt0``; only seismic and ``z``
                     are updated.
        ``"sv"``   — seismic model is kept at ``m_sv0``; only MT and ``z``
                     are updated.

    verbose : bool, optional
        Print per-iteration diagnostics. Default True.
    solve_m_mt, solve_m_sv : callable, optional
        Physics-specific linear solvers for MT and seismic model updates.
        Signature: ``solve(rhs, Wd, Wm, alpha, rho) -> m``.
    apply_Gt_mt, apply_Gt_sv : callable, optional
        Adjoint (data-gradient) operators for MT and seismic physics.
        Signature: ``apply_Gt(v) -> ndarray``.

    Returns
    -------
    dict
        Keys:

        ``m_mt``, ``m_sv``
            Final model vectors.
        ``z``
            Final consensus / latent variable.
        ``y_mt``, ``y_sv``
            Final dual variables.
        ``n_iter``
            Number of ADMM iterations executed.
        ``converged``
            ``True`` if stopping tolerances were met before ``max_outer``.

    Raises
    ------
    ValueError
        If ``fix_method`` is not ``None``, ``"mt"``, or ``"sv"``.
    """

    # ------------------------------------------------------------------
    # Validate fix_method
    # ------------------------------------------------------------------
    _valid_fix = {None, "mt", "sv"}
    if fix_method not in _valid_fix:
        raise ValueError(
            f"fix_method must be one of {_valid_fix}, got {fix_method!r}"
        )

    fix_mt = fix_method == "mt"
    fix_sv = fix_method == "sv"

    if verbose and fix_method is not None:
        fixed_label = "MT" if fix_mt else "seismic"
        print(f"[ADMM] fix_method='{fix_method}': {fixed_label} model held fixed.")

    # ------------------------------------------------------------------
    # Initialise models and dual variables
    # ------------------------------------------------------------------
    m_mt = m_mt0.copy()
    m_sv = m_sv0.copy()
    z = 0.5 * (m_mt + m_sv)

    y_mt = np.zeros_like(m_mt)
    y_sv = np.zeros_like(m_sv)

    z_old = z.copy()
    converged = False

    for it in range(max_outer):

        # --------------------------------------------------------------
        # 1) MT model update  (skipped when MT is fixed)
        # --------------------------------------------------------------
        if not fix_mt:
            rhs_mt = (
                apply_Gt_mt(Wd_mt.T @ (Wd_mt @ d_mt))
                + alpha_mt * (Wm_mt.T @ (Wm_mt @ m_mt_ref))
                + rho_mt * (z - y_mt / rho_mt)
            )
            m_mt = solve_m_mt(rhs_mt, Wd_mt, Wm_mt, alpha_mt, rho_mt)

        # --------------------------------------------------------------
        # 2) Seismic model update  (skipped when seismic is fixed)
        # --------------------------------------------------------------
        if not fix_sv:
            rhs_sv = (
                apply_Gt_sv(Wd_sv.T @ (Wd_sv @ d_sv))
                + alpha_sv * (Wm_sv.T @ (Wm_sv @ m_sv_ref))
                + rho_sv * (z - y_sv / rho_sv)
            )
            m_sv = solve_m_sv(rhs_sv, Wd_sv, Wm_sv, alpha_sv, rho_sv)

        # --------------------------------------------------------------
        # 3) Consensus / latent variable update  (via coupling object)
        # --------------------------------------------------------------
        z = coupling.update_z(m_mt, m_sv, y_mt, y_sv, rho_mt, rho_sv)

        # --------------------------------------------------------------
        # 4) Dual variable updates
        # --------------------------------------------------------------
        r_mt = m_mt - z
        r_sv = m_sv - z

        y_mt += rho_mt * r_mt
        y_sv += rho_sv * r_sv

        s_z = rho_mt * (z - z_old)
        z_old = z.copy()

        # --------------------------------------------------------------
        # 5) Diagnostics and stopping criteria
        # --------------------------------------------------------------
        norm_r = np.sqrt(
            np.linalg.norm(r_mt) ** 2 + np.linalg.norm(r_sv) ** 2
        )
        norm_s = np.linalg.norm(s_z)

        if verbose:
            msg = f"iter {it:03d}: ||r||={norm_r:.3e}  ||s||={norm_s:.3e}"
            if hasattr(coupling, "report"):
                diag = coupling.report(m_mt, m_sv, z)
                extras = "  ".join(f"{k}={v:.3e}" for k, v in diag.items())
                msg += f"  {extras}"
            print(msg)

        if norm_r < tol_primal and norm_s < tol_dual:
            converged = True
            break

    return dict(
        m_mt=m_mt,
        m_sv=m_sv,
        z=z,
        y_mt=y_mt,
        y_sv=y_sv,
        n_iter=it + 1,
        converged=converged,
    )
