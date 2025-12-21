#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inverse.py
==========
Numerical helpers for inverse problems and ensemble-based estimation.

This module collects small, self-contained numerical routines that are useful
in geophysical inverse problems and ensemble methods. The focus is on:

- Thresholding and TV-style regularisation (Split Bregman)
- Empirical covariance estimation (including NICE shrinkage)
- Matrix square-roots for SPD matrices (dense and sparse)
- Randomised SVD utilities (Halko et al., 2011)
- Simple spline fitting and bootstrap confidence bands

The functions are written to be robust for both dense NumPy arrays and common
SciPy sparse matrix types.

Author: Volker Rath (DIAS)
Created with the help of ChatGPT (GPT-5 Thinking) on 2025-12-21 (UTC)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import make_smoothing_spline


def soft_thresh(x: np.ndarray, lam: float) -> np.ndarray:
    """Soft-thresholding operator (prox of the ℓ1 norm).

    Parameters
    ----------
    x : numpy.ndarray
        Input array.
    lam : float
        Threshold parameter (must be non-negative).

    Returns
    -------
    numpy.ndarray
        Thresholded array with the same shape as ``x``.
    """
    lam = float(lam)
    if lam < 0.0:
        raise ValueError("lam must be non-negative.")
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


def splitbreg(
    J: sp.spmatrix | np.ndarray,
    y: np.ndarray,
    lam: float,
    D: sp.spmatrix | np.ndarray,
    c: float | np.ndarray = 0.0,
    tol: float = 1e-5,
    maxiter: int = 50,
) -> np.ndarray:
    """Solve a TV-regularised least-squares problem using Split Bregman.

    Minimises::

        0.5 * ||Jx - y||_2^2 + lam * ||Dx - c||_1

    Parameters
    ----------
    J : numpy.ndarray or scipy.sparse.spmatrix
        Forward operator / Jacobian of shape (nd, nm).
    y : numpy.ndarray
        Data vector of shape (nd,) or (nd, 1).
    lam : float
        Regularisation weight.
    D : numpy.ndarray or scipy.sparse.spmatrix
        Difference operator of shape (m, nm).
    c : float or numpy.ndarray, optional
        Offset for Dx (defaults to 0).
    tol : float, optional
        Relative convergence tolerance on x.
    maxiter : int, optional
        Maximum iterations.

    Returns
    -------
    numpy.ndarray
        Solution vector of shape (nm,).
    """
    lam = float(lam)
    if lam <= 0.0:
        raise ValueError("lam must be positive.")

    y = np.asarray(y, dtype=float).reshape(-1)
    c_arr = np.asarray(c, dtype=float)

    J_is_sparse = sp.issparse(J)
    D_is_sparse = sp.issparse(D)

    nd, nm = J.shape
    if y.size != nd:
        raise ValueError(f"y must have length {nd}, got {y.size}.")

    m, nm2 = D.shape
    if nm2 != nm:
        raise ValueError(f"D must have shape (m, {nm}); got {D.shape}.")

    mu = 2.0 * lam

    def JTJ_mv(xv: np.ndarray) -> np.ndarray:
        return (J.T @ (J @ xv)) if J_is_sparse else (J.T @ (J @ xv))

    def DTD_mv(xv: np.ndarray) -> np.ndarray:
        return (D.T @ (D @ xv)) if D_is_sparse else (D.T @ (D @ xv))

    Aop = spla.LinearOperator((nm, nm), matvec=lambda xv: JTJ_mv(xv) + mu * DTD_mv(xv), dtype=float)

    x = np.zeros(nm, dtype=float)
    b = np.zeros(m, dtype=float)
    d = np.zeros(m, dtype=float)

    JTy = (J.T @ y) if J_is_sparse else (J.T @ y)

    x_old = x.copy()

    for _ in range(int(maxiter)):
        rhs = JTy + mu * (D.T @ (d - b + c_arr)) if D_is_sparse else (JTy + mu * (D.T @ (d - b + c_arr)))

        x, info = spla.cg(Aop, rhs, x0=x, rtol=1e-10, atol=0.0, maxiter=10_000)
        if info != 0:
            if nm <= 4000 and not (J_is_sparse or D_is_sparse):
                A_dense = (J.T @ J) + mu * (D.T @ D)
                x = la.solve(A_dense, rhs, assume_a="sym")
            else:
                raise RuntimeError(f"CG did not converge (info={info}).")

        Dx = (D @ x) if D_is_sparse else (D @ x)
        d = soft_thresh(Dx + b - c_arr, lam / mu)
        b = b + Dx - d - c_arr

        denom = max(1e-14, np.linalg.norm(x))
        if np.linalg.norm(x - x_old) / denom < tol:
            break
        x_old = x.copy()

    return x


def calc_covar_simple(
    x: np.ndarray,
    y: Optional[np.ndarray] = None,
    method: str = "fast",
    out: bool = True,
) -> np.ndarray:
    """Compute the empirical cross-covariance between two ensembles.

    Ensembles are stored as rows: shape (Ne, Nvar).

    Returns a matrix of shape (Nx, Ny).
    """
    X = np.asarray(x, dtype=float)
    Y = X if y is None else np.asarray(y, dtype=float)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("x and y must be 2D arrays of shape (Ne, Nvar).")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("x and y must have the same number of samples (rows).")

    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    Ne = X.shape[0]
    if Ne < 2:
        raise ValueError("Need at least 2 samples to estimate covariance.")

    if method == "naive":
        cov = np.zeros((X.shape[1], Y.shape[1]), dtype=float)
        for _ in range(Ne):
            cov += Xc.T @ Yc
        cov /= (Ne - 1)
    else:
        cov = (Xc.T @ Yc) / (Ne - 1)

    if out:
        print(f"Ensemble covariance shape: {cov.shape}")
    return cov


def calc_covar_nice(
    x: np.ndarray,
    y: np.ndarray,
    fac: float,
    out: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NICE covariance shrinkage estimator (Vishny et al., 2024)."""
    X = np.asarray(x, dtype=float)
    Y = np.asarray(y, dtype=float)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("x and y must be 2D arrays of shape (Ne, Nvar).")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("x and y must have the same number of samples (rows).")

    Ne = X.shape[0]
    if Ne < 3:
        raise ValueError("NICE requires at least 3 samples for stable estimation.")

    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    sx = Xc.std(axis=0, ddof=1)
    sy = Yc.std(axis=0, ddof=1)
    sx[sx == 0.0] = np.nan
    sy[sy == 0.0] = np.nan
    Xz = Xc / sx
    Yz = Yc / sy

    CorrXY = (Xz.T @ Yz) / (Ne - 1)

    std_rho = (1.0 - CorrXY**2) / np.sqrt(Ne)
    sig_rho = float(np.sqrt(np.sum(std_rho**2)))

    expo2_candidates = np.array([2, 4, 6, 8], dtype=int)
    expo2 = int(expo2_candidates[-1])
    for e2 in expo2_candidates:
        L = np.abs(CorrXY) ** int(e2)
        Corr_try = L * CorrXY
        if np.linalg.norm(Corr_try - CorrXY, ord="fro") > fac * sig_rho:
            expo2 = int(e2)
            break

    expo1 = expo2 - 2
    rho_exp1 = CorrXY ** expo1
    rho_exp2 = CorrXY ** expo2

    alphas = np.arange(0.1, 1.01, 0.1)
    Corr_prev = CorrXY.copy()
    L_nice = np.ones_like(CorrXY)

    for a in alphas:
        L = (1.0 - a) * rho_exp1 + a * rho_exp2
        Corr_try = L * CorrXY
        if np.linalg.norm(Corr_try - CorrXY, ord="fro") > fac * sig_rho:
            Corr_try = Corr_prev
            break
        Corr_prev = Corr_try
        L_nice = L

    Corr_nice = Corr_try

    cov_nice = np.diag(sx) @ Corr_nice @ np.diag(sy)

    if out:
        print(f"NICE: expo2={expo2}, fac={fac}")

    return cov_nice, Corr_nice, L_nice


def msqrt_sparse(
    M: sp.spmatrix | np.ndarray,
    method: str = "chol",
    smallval: Optional[float] = None,
    nthreads: int = 16,
    k_eigs: Optional[int] = None,
) -> np.ndarray:
    """Compute a matrix factor S such that S @ S.T ≈ M."""
    from threadpoolctl import threadpool_limits

    n, n2 = M.shape
    if n != n2:
        raise ValueError("M must be square.")

    if smallval is not None and smallval != 0.0:
        if sp.issparse(M):
            M = M + smallval * sp.identity(n, format="csc")
        else:
            M = np.asarray(M, dtype=float) + smallval * np.eye(n)

    mth = method.lower()

    if "eigs" in mth:
        if sp.issparse(M):
            k = int(min(max(2, k_eigs or 50), n - 1))
            with threadpool_limits(limits=nthreads):
                evals, evecs = spla.eigsh(M, k=k, which="LA")
            evals = np.clip(evals, 0.0, None)
            return evecs * np.sqrt(evals)
        A = np.asarray(M, dtype=float)
        with threadpool_limits(limits=nthreads):
            evals, evecs = la.eigh(A)
        evals = np.clip(evals, 0.0, None)
        return evecs * np.sqrt(evals)

    if "chol" in mth:
        A = M.toarray() if sp.issparse(M) else np.asarray(M, dtype=float)
        with threadpool_limits(limits=nthreads):
            return la.cholesky(A, lower=True)

    if "splu" in mth:
        A = M.tocsc() if sp.issparse(M) else sp.csc_matrix(np.asarray(M, dtype=float))
        with threadpool_limits(limits=nthreads):
            LU = spla.splu(A, diag_pivot_thresh=0.0)

        if not (np.all(LU.perm_r == np.arange(n)) and np.all(LU.U.diagonal() > 0)):
            raise ValueError("Matrix does not appear SPD under LU decomposition.")
        return (LU.L @ sp.diags(np.sqrt(LU.U.diagonal()))).toarray()

    raise ValueError("Unknown method. Use 'chol', 'eigs', or 'splu'.")


def isspd(A: sp.spmatrix | np.ndarray, *, atol: float = 1e-12) -> bool:
    """Heuristic check for symmetric positive definiteness."""
    if sp.issparse(A):
        if (A - A.T).nnz != 0:
            return False
        try:
            evals = spla.eigsh(A, k=1, which="SA", return_eigenvectors=False)
            return bool(evals[0] > atol)
        except Exception:
            return False

    try:
        la.cholesky(np.asarray(A, dtype=float), lower=True)
        return True
    except Exception:
        return False


def rsvd(
    A: sp.spmatrix | np.ndarray,
    rank: int = 300,
    n_oversamples: int = 20,
    n_subspace_iters: Optional[int] = None,
    return_range: bool = False,
):
    """Randomised SVD (Halko, Martinsson & Tropp, 2011)."""
    n_samples = int(rank) + int(n_oversamples)

    Q = find_range(A, n_samples, n_subspace_iters)
    B = Q.T @ A
    U_tilde, S, Vt = np.linalg.svd(np.asarray(B), full_matrices=False)
    U = Q @ U_tilde

    U = U[:, :rank]
    S = S[:rank]
    Vt = Vt[:rank, :]

    if return_range:
        return U, S, Vt, Q
    return U, S, Vt


def find_range(A: sp.spmatrix | np.ndarray, n_samples: int, n_subspace_iters: Optional[int] = None) -> np.ndarray:
    """Randomised range finder (Algorithm 4.1 in Halko et al.)."""
    rng = np.random.default_rng()
    _, n = A.shape
    O = rng.normal(size=(n, n_samples))
    Y = A @ O
    if n_subspace_iters and n_subspace_iters > 0:
        return subspace_iter(A, Y, int(n_subspace_iters))
    return ortho_basis(Y)


def subspace_iter(A: sp.spmatrix | np.ndarray, Y0: np.ndarray, n_iters: int) -> np.ndarray:
    """Randomised subspace iteration (Algorithm 4.4 in Halko et al.)."""
    Q = ortho_basis(Y0)
    for _ in range(n_iters):
        Z = ortho_basis(A.T @ Q)
        Q = ortho_basis(A @ Z)
    return Q


def ortho_basis(M: np.ndarray) -> np.ndarray:
    """Compute an orthonormal basis for the range of M using QR."""
    Q, _ = np.linalg.qr(np.asarray(M), mode="reduced")
    return Q


def make_spline(x: np.ndarray, y: np.ndarray, lam: float | None = None):
    """Fit a smoothing spline using SciPy and return the spline object."""
    order = np.argsort(x)
    x_sorted = np.asarray(x)[order]
    y_sorted = np.asarray(y)[order]
    return make_smoothing_spline(x_sorted, y_sorted, lam=lam)


def estimate_variance(y_true: np.ndarray, y_fit: np.ndarray) -> float:
    """Estimate residual variance with ddof=1."""
    res = np.asarray(y_true) - np.asarray(y_fit)
    return float(np.var(res, ddof=1))


def bootstrap_confidence_band(
    x: np.ndarray,
    y: np.ndarray,
    lam: float | None = None,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap confidence bands for spline predictions."""
    x = np.asarray(x)
    y = np.asarray(y)
    order = np.argsort(x)
    x_eval = x[order]
    y_sorted = y[order]

    n = x_eval.size
    preds = np.full((n_bootstrap, n), np.nan, dtype=float)
    rng = np.random.default_rng()

    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        x_res = x_eval[idx]
        y_res = y_sorted[idx]

        ord2 = np.argsort(x_res)
        x_rs = x_res[ord2]
        y_rs = y_res[ord2]

        ux, uidx = np.unique(x_rs, return_index=True)
        if ux.size < 4:
            continue
        y_u = y_rs[uidx]

        spline = make_smoothing_spline(ux, y_u, lam=lam)
        preds[i, :] = spline(x_eval)

    alpha = 1.0 - float(ci)
    lower = np.nanpercentile(preds, 100.0 * alpha / 2.0, axis=0)
    upper = np.nanpercentile(preds, 100.0 * (1.0 - alpha / 2.0), axis=0)
    return x_eval, lower, upper
