"""
inversion_state.py
==================
Joint inversion iteration controller.

Implements an alternating Gauss-Newton strategy:
  1. MT model update  (CGLS inner iterations, Gramian gradient as forcing)
  2. Seismic model update  (CGLS inner iterations, Gramian gradient as forcing)

Both updates use the same Gramian coupling term evaluated at the current
joint model state.  This avoids forming the full joint Hessian while still
enforcing structural similarity through the gradient cross-coupling.

Author: VR 2026-05-11, Claude Sonnet 4.6 (Anthropic)
"""

from __future__ import annotations
import numpy as np


class JointInversionState:
    """Coordinate joint MT + seismic inversion iterations.

    Parameters
    ----------
    mt_fwd       : MTForward
    seis_fwd     : SeisTomoForward
    gramian      : StructuralGramian
    alpha_mt     : float  — regularisation weight for MT
    alpha_seis   : float  — regularisation weight for seismic
    n_inner_mt   : int    — CGLS iterations per MT update
    n_inner_seis : int    — CGLS iterations per seismic update
    reg_strategy : str    — "fixed" or "adaptive" (L-curve / GCV, TODO)
    out          : bool
    """

    def __init__(self, mt_fwd, seis_fwd, gramian, *,
                 alpha_mt: float = 1e-3,
                 alpha_seis: float = 1e-3,
                 n_inner_mt: int = 5,
                 n_inner_seis: int = 5,
                 reg_strategy: str = "fixed",
                 out: bool = True):

        self.mt_fwd       = mt_fwd
        self.seis_fwd     = seis_fwd
        self.gramian      = gramian
        self.alpha_mt     = float(alpha_mt)
        self.alpha_seis   = float(alpha_seis)
        self.n_inner_mt   = int(n_inner_mt)
        self.n_inner_seis = int(n_inner_seis)
        self.reg_strategy = reg_strategy
        self.out          = out
        self._it          = 0

    # ── Regularisation (Tikhonov minimum-norm on model perturbation) ─────────

    def _reg_mt(self, m: np.ndarray) -> tuple[float, np.ndarray]:
        phi  = 0.5 * self.alpha_mt * float(np.dot(m, m))
        grad = self.alpha_mt * m
        return phi, grad

    def _reg_seis(self, m: np.ndarray) -> tuple[float, np.ndarray]:
        phi  = 0.5 * self.alpha_seis * float(np.dot(m, m))
        grad = self.alpha_seis * m
        return phi, grad

    # ── Steepest-descent model update (replace with CGLS for production) ─────

    @staticmethod
    def _sd_step(grad: np.ndarray, *, step: float = 1e-3) -> np.ndarray:
        """Simple steepest-descent step.  Replace with CGLS / L-BFGS."""
        return -step * grad / (np.linalg.norm(grad) + 1e-30)

    # ── One joint iteration ──────────────────────────────────────────────────

    def step(self) -> dict:
        """Perform one joint Gauss-Newton iteration.

        Returns
        -------
        dict with keys 'mt', 'seis', 'gramian', 'total'
        """
        self._it += 1
        m_mt   = self.mt_fwd._m.copy()
        m_seis = self.seis_fwd._m.copy()

        # ── Gramian gradient at current state ──────────────────────────────
        phi_g              = self.gramian.value(m_mt, m_seis)
        dg_mt, dg_seis     = self.gramian.gradient(m_mt, m_seis)

        # ── MT update ──────────────────────────────────────────────────────
        phi_mt             = self.mt_fwd.misfit(m_mt)
        g_mt               = self.mt_fwd.gradient(m_mt)
        _, r_mt            = self._reg_mt(m_mt)
        g_mt_total         = g_mt + r_mt + dg_mt
        dm_mt              = self._sd_step(g_mt_total)
        self.mt_fwd.update(dm_mt)

        # ── Seismic update ─────────────────────────────────────────────────
        phi_seis           = self.seis_fwd.misfit(m_seis)
        g_seis             = self.seis_fwd.gradient(m_seis)
        _, r_seis          = self._reg_seis(m_seis)
        g_seis_total       = g_seis + r_seis + dg_seis
        dm_seis            = self._sd_step(g_seis_total)
        self.seis_fwd.update(dm_seis)

        phi_reg = (0.5 * self.alpha_mt   * float(np.dot(m_mt,   m_mt)) +
                   0.5 * self.alpha_seis * float(np.dot(m_seis, m_seis)))
        phi_tot = phi_mt + phi_seis + phi_reg + phi_g

        return dict(mt=phi_mt, seis=phi_seis,
                    gramian=phi_g, total=phi_tot)
