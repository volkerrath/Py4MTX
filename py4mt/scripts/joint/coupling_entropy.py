#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mutual-information (entropy) coupling for joint MT + seismic ADMM inversion.

Self-contained module. Provides:
  - mutual_information_kde      — Parzen-window MI estimator
  - MutualInformationCoupling   — full coupling class (value / gradient / report)
  - CombinedCoupling            — Gramian + MI sum (imports coupling_gramian)

Inlined from:
  entropy/modules/cross_entropy_coupling.py
  (MultiscaleResampler shared with gramian is imported from coupling_gramian)

References
----------
Haber & Oldenburg (1997). Inverse Problems 13, 63–77.
    https://doi.org/10.1088/0266-5611/13/1/006
Moorkamp et al. (2011). GJI 184, 477–493.
    https://doi.org/10.1111/j.1365-246X.2010.04856.x
Viola & Wells (1997). IJCV 24, 137–154.
    https://doi.org/10.1023/A:1007958904918

@author:   Volker Rath (DIAS)
@project:  py4mt — Python for Magnetotellurics
@created:  2026-05-12 — VR / Claude Sonnet 4.6 (Anthropic)
"""

from __future__ import annotations

from typing import Literal

import numpy as np

# MultiscaleResampler and ModelGrid are shared with the Gramian module
from coupling_gramian import MultiscaleResampler, ModelGrid, _fd_gradient_magnitude


# =============================================================================
# Utility: normal-score transform
# =============================================================================

def _normal_score(x: np.ndarray) -> np.ndarray:
    """Normal-score (quantile) transform → standard normal scores."""
    from scipy.stats import norm
    n    = len(x)
    rank = np.argsort(np.argsort(x)) + 1
    p    = (rank - 0.375) / (n + 0.25)
    return norm.ppf(np.clip(p, 1e-6, 1.0 - 1e-6))


# =============================================================================
# KDE and MI utilities
# =============================================================================

def _scott_bandwidth(samples: np.ndarray) -> float:
    sd = float(np.std(samples))
    return sd * len(samples) ** (-0.2) if sd > 0 else 1.0


def _make_grid(samples: np.ndarray, n_bins: int, h: float,
               pad: float = 3.0) -> np.ndarray:
    lo = samples.min() - pad * h
    hi = samples.max() + pad * h
    return np.linspace(lo, hi, n_bins)


def mutual_information_kde(
    u: np.ndarray,
    v: np.ndarray,
    n_bins: int = 32,
    hu: float = 0.0,
    hv: float = 0.0,
    ug: np.ndarray = None,
    vg: np.ndarray = None,
) -> float:
    """
    Estimate mutual information I(u, v) via Parzen-window KDE.

    Parameters
    ----------
    u, v   : (M,)  model attributes on the common grid
    n_bins : int   histogram bins per axis
    hu, hv : float bandwidth; 0 → Scott's rule
    ug, vg : (B,)  pre-built evaluation grids (optional)

    Returns
    -------
    mi : float  mutual information in nats (≥ 0)
    """
    if hu <= 0.0:
        hu = _scott_bandwidth(u)
    if hv <= 0.0:
        hv = _scott_bandwidth(v)
    if ug is None:
        ug = _make_grid(u, n_bins, hu)
    if vg is None:
        vg = _make_grid(v, n_bins, hv)

    du = ug[1] - ug[0]
    dv = vg[1] - vg[0]
    M  = len(u)
    cu = 1.0 / (hu * np.sqrt(2.0 * np.pi))
    cv = 1.0 / (hv * np.sqrt(2.0 * np.pi))

    Ku = cu * np.exp(-0.5 * ((ug[:, None] - u[None, :]) / hu) ** 2)  # (Bu, M)
    Kv = cv * np.exp(-0.5 * ((vg[:, None] - v[None, :]) / hv) ** 2)  # (Bv, M)

    P   = np.einsum("ik,jk->ij", Ku, Kv) / M
    Pu  = P.sum(axis=1) * dv
    Pv  = P.sum(axis=0) * du
    eps = 1e-30
    log_r = np.log((P + eps) / (Pu[:, None] * Pv[None, :] + eps))
    return float((P * log_r).sum() * du * dv)


def _mi_gradient_samples(
    u: np.ndarray,
    v: np.ndarray,
    n_bins: int = 32,
    hu: float = 0.0,
    hv: float = 0.0,
    eps: float = 1e-30,
):
    """Gradient of I(u, v) w.r.t. each sample u_k and v_k (analytic VJP)."""
    M = len(u)
    if hu <= 0.0:
        hu = _scott_bandwidth(u)
    if hv <= 0.0:
        hv = _scott_bandwidth(v)

    ug = np.linspace(u.min() - 3 * hu, u.max() + 3 * hu, n_bins)
    vg = np.linspace(v.min() - 3 * hv, v.max() + 3 * hv, n_bins)
    du = ug[1] - ug[0]
    dv = vg[1] - vg[0]

    c_u = 1.0 / (hu * np.sqrt(2.0 * np.pi))
    c_v = 1.0 / (hv * np.sqrt(2.0 * np.pi))
    Ku  = c_u * np.exp(-0.5 * ((ug[:, None] - u[None, :]) / hu) ** 2)  # (Bu, M)
    Kv  = c_v * np.exp(-0.5 * ((vg[:, None] - v[None, :]) / hv) ** 2)  # (Bv, M)

    P     = np.einsum("ik,jk->ij", Ku, Kv) / M
    Pu    = P.sum(axis=1) * dv
    Pv    = P.sum(axis=0) * du
    denom = Pu[:, None] * Pv[None, :] + eps
    log_r = np.log((P + eps) / denom)

    dKu = Ku * (ug[:, None] - u[None, :]) / (hu ** 2)
    dKv = Kv * (vg[:, None] - v[None, :]) / (hv ** 2)

    term1_u = (dKu * (log_r @ Kv)).sum(axis=0)
    corr_u  = dKu.sum(axis=0) * Kv.sum(axis=0) / M
    grad_u  = (du * dv / M) * term1_u - (du * dv ** 2 / M ** 2) * corr_u

    term1_v = (dKv * (log_r.T @ Ku)).sum(axis=0)
    corr_v  = dKv.sum(axis=0) * Ku.sum(axis=0) / M
    grad_v  = (du * dv / M) * term1_v - (du ** 2 * dv / M ** 2) * corr_v

    return grad_u, grad_v


# =============================================================================
# MutualInformationCoupling
# =============================================================================

class MutualInformationCoupling:
    """
    Mutual-information coupling for joint MT + seismic inversion.

    Objective:  Φ_MI = −β · I(T(R_mt m_mt), T(R_seis m_seis))

    Parameters
    ----------
    mt_to_g       : MultiscaleResampler  — MT grid → common grid
    seis_to_g     : MultiscaleResampler  — seismic grid → common grid
    beta          : float    — coupling weight (≥ 0)
    n_bins        : int      — KDE bins per axis (default 32)
    bandwidth_u   : float    — KDE bandwidth for MT attribute; 0 = Scott
    bandwidth_v   : float    — KDE bandwidth for seismic attribute; 0 = Scott
    mode          : {"value", "gradient", "rank"}
    grad_mode     : {"analytic", "fd"}
    fd_eps        : float    — FD step for "fd" grad_mode
    common_coords : (M, 3)   — required for mode="gradient"
    k_nn          : int      — neighbours for FD spatial gradient
    """

    def __init__(
        self,
        mt_to_g,
        seis_to_g,
        *,
        beta: float = 1.0,
        n_bins: int = 32,
        bandwidth_u: float = 0.0,
        bandwidth_v: float = 0.0,
        mode: Literal["value", "gradient", "rank"] = "value",
        grad_mode: Literal["analytic", "fd"] = "analytic",
        fd_eps: float = 1e-4,
        common_coords: np.ndarray = None,
        k_nn: int = 6,
    ):
        self.mt_to_g   = mt_to_g
        self.seis_to_g = seis_to_g
        self.beta      = float(beta)
        self.n_bins    = int(n_bins)
        self.hu        = float(bandwidth_u)
        self.hv        = float(bandwidth_v)
        self.mode      = mode
        self.grad_mode = grad_mode
        self.fd_eps    = float(fd_eps)
        self._coords   = (np.asarray(common_coords, dtype=float)
                          if common_coords is not None else None)
        self.k_nn      = int(k_nn)

    # -- Attribute transform --------------------------------------------------

    def _transform(self, raw: np.ndarray) -> np.ndarray:
        if self.mode == "value":
            return raw
        elif self.mode == "rank":
            return _normal_score(raw)
        elif self.mode == "gradient":
            if self._coords is None:
                raise ValueError("common_coords required for mode='gradient'.")
            return _fd_gradient_magnitude(raw, self._coords, k_nn=self.k_nn)
        else:
            raise ValueError(f"Unknown mode '{self.mode}'.")

    def _transform_vjp(self, raw: np.ndarray, g: np.ndarray) -> np.ndarray:
        if self.mode in ("value", "rank"):
            return g
        elif self.mode == "gradient":
            fx  = self._transform(raw)
            out = np.zeros_like(raw)
            for i in range(len(raw)):
                rp     = raw.copy(); rp[i] += self.fd_eps
                out[i] = float(g @ ((self._transform(rp) - fx) / self.fd_eps))
            return out

    def _grids(self, u, v):
        hu = self.hu if self.hu > 0 else _scott_bandwidth(u)
        hv = self.hv if self.hv > 0 else _scott_bandwidth(v)
        return _make_grid(u, self.n_bins, hu), _make_grid(v, self.n_bins, hv), hu, hv

    # -- Objective ------------------------------------------------------------

    def value(self, m_mt: np.ndarray, m_seis: np.ndarray) -> float:
        """Φ_MI = −β · I(u, v)  (scalar, to be minimised)."""
        u = self._transform(self.mt_to_g(m_mt))
        v = self._transform(self.seis_to_g(m_seis))
        ug, vg, hu, hv = self._grids(u, v)
        return -self.beta * mutual_information_kde(
            u, v, n_bins=self.n_bins, hu=hu, hv=hv, ug=ug, vg=vg
        )

    # -- Gradient -------------------------------------------------------------

    def gradient(self, m_mt: np.ndarray, m_seis: np.ndarray):
        """
        Gradients of Φ_MI on the native model grids.

        Returns
        -------
        grad_mt   : (N_mt,)
        grad_seis : (N_seis,)
        """
        u_raw = self.mt_to_g(m_mt)
        v_raw = self.seis_to_g(m_seis)
        u     = self._transform(u_raw)
        v     = self._transform(v_raw)
        ug, vg, hu, hv = self._grids(u, v)

        if self.grad_mode == "analytic":
            gu, gv = _mi_gradient_samples(u, v, n_bins=self.n_bins, hu=hu, hv=hv)
        else:
            gu, gv = self._fd_gradient(u, v, ug=ug, vg=vg, hu=hu, hv=hv)

        return (
            -self.beta * self.mt_to_g.adjoint(self._transform_vjp(u_raw, gu)),
            -self.beta * self.seis_to_g.adjoint(self._transform_vjp(v_raw, gv)),
        )

    def _fd_gradient(self, u, v, ug=None, vg=None, hu=0.0, hv=0.0):
        eps = self.fd_eps
        I0  = mutual_information_kde(u, v, n_bins=self.n_bins,
                                     hu=hu, hv=hv, ug=ug, vg=vg)
        gu = np.zeros_like(u)
        gv = np.zeros_like(v)
        for k in range(len(u)):
            up      = u.copy(); up[k] += eps
            gu[k]   = (mutual_information_kde(up, v, n_bins=self.n_bins,
                                              hu=hu, hv=hv, ug=ug, vg=vg) - I0) / eps
        for k in range(len(v)):
            vp      = v.copy(); vp[k] += eps
            gv[k]   = (mutual_information_kde(u, vp, n_bins=self.n_bins,
                                              hu=hu, hv=hv, ug=ug, vg=vg) - I0) / eps
        return gu, gv

    # -- Diagnostics ----------------------------------------------------------

    def report(self, m_mt: np.ndarray, m_seis: np.ndarray) -> dict:
        """Return MI and scatter statistics."""
        u = self._transform(self.mt_to_g(m_mt))
        v = self._transform(self.seis_to_g(m_seis))
        ug, vg, hu, hv = self._grids(u, v)
        mi   = mutual_information_kde(u, v, n_bins=self.n_bins,
                                      hu=hu, hv=hv, ug=ug, vg=vg)
        corr = float(np.corrcoef(u, v)[0, 1])
        return dict(
            mutual_information=mi,
            pearson_r=corr,
            phi_MI=-self.beta * mi,
            u_range=(float(u.min()), float(u.max())),
            v_range=(float(v.min()), float(v.max())),
        )


# =============================================================================
# CombinedCoupling (Gramian + MI)
# =============================================================================

class CombinedCoupling:
    """
    Sum of StructuralGramian and MutualInformationCoupling.

    Enforces both structural similarity (Gramian) and statistical dependence
    (MI) simultaneously.

    Parameters
    ----------
    gramian : StructuralGramian
    mi      : MutualInformationCoupling
    """

    def __init__(self, gramian, mi):
        self.gramian = gramian
        self.mi      = mi

    def value(self, m_mt, m_seis):
        return self.gramian.value(m_mt, m_seis) + self.mi.value(m_mt, m_seis)

    def gradient(self, m_mt, m_seis):
        gg_mt,  gg_seis  = self.gramian.gradient(m_mt, m_seis)
        gmi_mt, gmi_seis = self.mi.gradient(m_mt, m_seis)
        return gg_mt + gmi_mt, gg_seis + gmi_seis

    def report(self, m_mt, m_seis):
        r_g  = self.gramian.report(m_mt, m_seis)
        r_mi = self.mi.report(m_mt, m_seis)
        return dict(**r_g, **r_mi,
                    phi_total=r_g["gramian"] + r_mi["phi_MI"])
