#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 16:08:28 2025

@author: vrath
"""

import numpy as np
from scipy.sparse.linalg import cg

def solve_m_step(A, Wd, d, L, alpha, rhs_quad, Gk, rho, m0):
    # Linearized least squares with quadratic term: minimize
    # 0.5||Wd(A m - d)||^2 + 0.5*alpha||L(m - m_ref)||^2 + 0.5*rho||Gk m - rhs_quad||^2
    # Normal matrix: H = A^T Wd^T Wd A + alpha L^T L + rho Gk^T Gk
    # RHS: b = A^T Wd^T Wd d + alpha L^T L m_ref + rho Gk^T rhs_quad
    ATWd = A.T @ Wd
    H = ATWd @ (Wd @ A) + alpha*(L.T @ L) + rho*(Gk.T @ Gk)
    b = ATWd @ (Wd @ d) + alpha*(L.T @ (L @ m0)) + rho*(Gk.T @ rhs_quad)
    m, _ = cg(H, b, x0=m0, maxiter=200)
    return m

def prox_indicator_intervals(x, intervals):
    # intervals: list of (lo, hi); elementwise projection to nearest interval
    y = x.copy()
    for i, val in enumerate(x):
        best = None
        for (lo, hi) in intervals:
            if lo <= val <= hi:
                best = val; break
            # else pick nearest bound
            proj = lo if val < lo else hi
            if best is None or abs(proj - val) < abs(best - val):
                best = proj
        y[i] = best
    return y

def joint_admm(A1, d1, A2, d2, G1, G2, L1, L2, alpha1, alpha2,
               psi='indicator', intervals=None, rho=1.0, iters=50):
    # Initialize
    m1 = np.zeros(A1.shape[1]); m2 = np.zeros(A2.shape[1])
    y = np.zeros_like(G1 @ m1); u = np.zeros_like(y)

    for t in range(iters):
        rhs1 = y - u - G2 @ m2
        m1 = solve_m_step(A1, np.eye(A1.shape[0]), d1, L1, alpha1, rhs1, G1, rho, m1)

        rhs2 = y - u - G1 @ m1
        m2 = solve_m_step(A2, np.eye(A2.shape[0]), d2, L2, alpha2, rhs2, G2, rho, m2)

        v = (G1 @ m1 + G2 @ m2) + u
        if psi == 'indicator':
            y = prox_indicator_intervals(v, intervals)
        else:
            # placeholder for other prox (e.g., soft-threshold for L1)
            y = v  # no coupling

        u = u + (G1 @ m1 + G2 @ m2) - y

        # check residuals
        r = (G1 @ m1 + G2 @ m2) - y
        s = rho * (y - v)
        if np.linalg.norm(r) < 1e-3*np.linalg.norm(y) and np.linalg.norm(s) < 1e-3*np.linalg.norm(u):
            break

    return m1, m2
