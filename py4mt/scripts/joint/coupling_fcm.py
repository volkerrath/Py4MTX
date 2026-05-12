#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fuzzy C-Means (FCM) coupling for joint MT + seismic ADMM inversion.

Self-contained module. Provides:
  - update_centroids        — FCM centroid update
  - update_memberships      — FCM membership matrix update
  - compute_distances       — squared distances (diagnostics)
  - update_z_fcm            — closed-form latent field z update

Inlined from:
  fcm/fcm.py, fcm/latent_field.py  (crossgrad/ copies are identical)

@author:   Volker Rath (DIAS)
@project:  py4mt — Python for Magnetotellurics
@created:  2026-05-12 — VR / Claude Sonnet 4.6 (Anthropic)
"""

from __future__ import annotations

import numpy as np


# =============================================================================
# FCM clustering
# =============================================================================

def update_centroids(z, U, q):
    """
    Update FCM centroids.

    Parameters
    ----------
    z : ndarray (N,)
        Latent petrophysical field.
    U : ndarray (N, K)
        Membership matrix.
    q : float
        Fuzziness exponent.

    Returns
    -------
    ndarray (K,)  Updated centroids.
    """
    Um  = U ** q
    num = (Um * z[:, None]).sum(axis=0)
    den = Um.sum(axis=0) + 1e-15
    return num / den


def update_memberships(z, c, q):
    """
    Update FCM memberships.

    Parameters
    ----------
    z : ndarray (N,)  Latent field.
    c : ndarray (K,)  Centroids.
    q : float         Fuzziness exponent.

    Returns
    -------
    ndarray (N, K)  Membership matrix with row sums = 1.
    """
    N = z.size
    K = c.size
    U = np.zeros((N, K))

    d2 = (z[:, None] - c[None, :]) ** 2
    d  = np.sqrt(d2 + 1e-15)
    power = 2.0 / (q - 1.0)

    for i in range(N):
        di = d[i]
        if np.any(di < 1e-12):
            u = np.zeros(K)
            u[np.argmin(di)] = 1.0
        else:
            ratios = di[:, None] / di[None, :]
            denom  = (ratios ** power).sum(axis=1)
            u      = 1.0 / denom
        U[i] = u

    return U


def compute_distances(z, c):
    """
    Squared distances between latent field and centroids.

    Parameters
    ----------
    z : ndarray (N,)  Latent field.
    c : ndarray (K,)  Centroids.

    Returns
    -------
    ndarray (N, K)  Squared distances d_{ik}^2.
    """
    return (z[:, None] - c[None, :]) ** 2


# =============================================================================
# Latent field z update (FCM + ADMM)
# =============================================================================

def update_z_fcm(
    m_mt,
    m_sv,
    y_mt,
    y_sv,
    U,
    c,
    beta,
    rho_mt,
    rho_sv,
    q,
    w_mt=0.5,
    w_sv=0.5,
):
    """
    Update the latent petrophysical field z (FCM + ADMM closed form).

    Solves:

        min_z  β Σ_i Σ_k u_ik^q (z_i - c_k)²
             + (ρ_mt/2) ‖m_mt - z + y_mt/ρ_mt‖²
             + (ρ_sv/2) ‖m_sv - z + y_sv/ρ_sv‖²

    Parameters
    ----------
    m_mt, m_sv : ndarray (N,)   Model vectors.
    y_mt, y_sv : ndarray (N,)   Dual variables.
    U          : ndarray (N, K) Membership matrix.
    c          : ndarray (K,)   Centroids.
    beta       : float          FCM coupling weight.
    rho_mt, rho_sv : float      ADMM penalty parameters.
    q          : float          Fuzziness exponent.
    w_mt, w_sv : float          Relative ADMM weights (sum to 1).

    Returns
    -------
    z : ndarray (N,)
    """
    Um = U ** q

    sum_uq    = Um.sum(axis=1)                   # (N,)  Σ_k u_ik^q
    sum_uq_ck = (Um * c[None, :]).sum(axis=1)    # (N,)  Σ_k u_ik^q c_k

    v_mt = m_mt + y_mt / rho_mt
    v_sv = m_sv + y_sv / rho_sv

    num_pen = w_mt * rho_mt * v_mt + w_sv * rho_sv * v_sv
    den_pen = w_mt * rho_mt + w_sv * rho_sv

    num = 2.0 * beta * sum_uq_ck + num_pen
    den = 2.0 * beta * sum_uq    + den_pen + 1e-15

    return num / den
