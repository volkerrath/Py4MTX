#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inverse.py
==========
Numerical helpers for inverse problems, ensemble-based estimation, and
deterministic 1-D anisotropic MT inversion.

This module consolidates two earlier files into a single coherent package:

**Part A — General numerical routines:**

- Thresholding and TV-style regularisation (Split Bregman)
- Empirical covariance estimation (including NICE shrinkage)
- Matrix square-roots for SPD matrices (dense and sparse)
- Randomised SVD utilities (Halko et al., 2011)
- Simple spline fitting and bootstrap confidence bands

**Part B — Deterministic 1-D MT inversion:**

- Model parameterization (rho/sigma domains, minmax/max_anifac sets)
- Impedance and phase-tensor data packing
- Gauss–Newton solver with Tikhonov (ridge) and TSVD options
- Site I/O (NPZ, EDI via ``data_proc``)

The numerical routines are written to be robust for both dense NumPy arrays
and common SciPy sparse matrix types.  The 1-D inversion focuses on pragmatic
script support, not a full-featured production inversion package.

Key design points (1-D inversion)
----------------------------------
- Parameterization per layer:

  - ``h_m``          thickness (optional inversion)
  - ``rho_min``      minimum horizontal resistivity
  - ``rho_max``      maximum horizontal resistivity
  - ``strike_deg``   anisotropy strike

- Two boolean masks may be supplied:

  - ``is_iso`` : layer is isotropic (forces rho_max == rho_min and ignores strike)
  - ``is_fix`` : layer is fixed (all layer parameters are held constant)

- Regularization:

  The interface accepts Tikhonov/TSVD style options for compatibility with
  earlier scripts. The current implementation supports:

  - ``method='tikhonov'`` with a simple *identity* damping (ridge).
  - ``method='tsvd'`` with optional truncation rank.

  More advanced selection modes (GCV/L-curve/ABIC) are accepted but currently
  fall back to the user-provided fixed values.

Author: Volker Rath (DIAS)
Original numerical utilities created with GPT-5 Thinking on 2025-12-21 (UTC).
1-D inversion helpers created with GPT-5 Thinking on 2026-02-13 (UTC).

Changelog
---------
2026-03-02  Cleanup and merge by Claude (Anthropic, Opus 4.6):
    Numerical utilities (formerly standalone inverse.py):
    - Fixed calc_covar_simple "naive" method: loop body was accumulating the
      full matrix product Xc.T @ Yc on every iteration instead of the per-row
      outer product np.outer(Xc[i], Yc[i]).
    - Fixed splitbreg rhs computation: a ternary expression
      ``rhs = JTy + mu * (...) if D_is_sparse else (...)`` bound the whole
      assignment to the conditional, and both branches were identical.
      Replaced with a single unconditional expression.
    - Fixed calc_covar_nice exponentiation: ``CorrXY ** expo1`` can produce
      NaN for negative correlations with non-integer intermediate exponents.
      Changed to ``np.abs(CorrXY) ** expo1`` (and expo2), consistent with
      the existing expo2_candidates loop.
    - Removed redundant sparse/dense conditionals in splitbreg (JTJ_mv,
      DTD_mv, JTy, Dx, rhs): both branches were identical because the @
      operator already dispatches correctly for sparse and dense operands.

    1-D inversion (formerly inv1d.py):
    - Fixed normalize_model: strict rejection of legacy keys ("rop",
      "ustr_deg", etc.) made the documented legacy handler unreachable dead
      code.  The docstring promised legacy support.  Replaced the hard
      rejection with a FutureWarning deprecation so the legacy path works
      again as documented.
    - Fixed _extract_site_data: returned (dd, False) for the wrapped-dict
      case; changed to (dd, True) to match the documented semantics.
    - Fixed _pack_Z_obs: second branch ``elif Ze.shape == Z.real.shape``
      was identical to the first branch ``Ze.shape == Z.shape`` (.real does
      not change shape), making it unreachable dead code.  Removed.
    - Fixed invert_site: phase tensor was computed twice from the same Zcal
      per iteration (once as Pcal for residuals, once as P0 for the
      Jacobian).  Merged into a single P0 computation.
    - Extracted repeated inline ``comp_map`` dict to module-level constant
      ``_COMP_MAP``.

    Merge:
    - Consolidated imports; organized into thematic sections.
    - Fixed PEP 8 spacing throughout.
"""

from __future__ import annotations

import glob
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import make_smoothing_spline

import aniso

try:
    import data_proc  # type: ignore
except Exception:  # pragma: no cover
    data_proc = None


# Component index map used by impedance/phase-tensor packing routines.
_COMP_MAP = {"xx": (0, 0), "xy": (0, 1), "yx": (1, 0), "yy": (1, 1)}


# =============================================================================
# Part A — General numerical routines
# =============================================================================


# -----------------------------------------------------------------------------
# A1. Thresholding and TV-style regularisation (Split Bregman)
# -----------------------------------------------------------------------------


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

    # The @ operator dispatches correctly for both sparse and dense matrices
    Aop = spla.LinearOperator(
        (nm, nm),
        matvec=lambda xv: J.T @ (J @ xv) + mu * (D.T @ (D @ xv)),
        dtype=float,
    )

    x = np.zeros(nm, dtype=float)
    b = np.zeros(m, dtype=float)
    d = np.zeros(m, dtype=float)

    JTy = J.T @ y
    x_old = x.copy()

    for _ in range(int(maxiter)):
        rhs = JTy + mu * (D.T @ (d - b + c_arr))

        x, info = spla.cg(Aop, rhs, x0=x, rtol=1e-10, atol=0.0, maxiter=10_000)
        if info != 0:
            if nm <= 4000 and not (J_is_sparse or D_is_sparse):
                A_dense = (J.T @ J) + mu * (D.T @ D)
                x = la.solve(A_dense, rhs, assume_a="sym")
            else:
                raise RuntimeError(f"CG did not converge (info={info}).")

        Dx = D @ x
        d = soft_thresh(Dx + b - c_arr, lam / mu)
        b = b + Dx - d - c_arr

        denom = max(1e-14, np.linalg.norm(x))
        if np.linalg.norm(x - x_old) / denom < tol:
            break
        x_old = x.copy()

    return x


# -----------------------------------------------------------------------------
# A2. Empirical covariance estimation (including NICE shrinkage)
# -----------------------------------------------------------------------------


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

    Ne = X.shape[0]
    if Ne < 2:
        raise ValueError("Need at least 2 samples to estimate covariance.")

    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)

    if method == "naive":
        # Explicit row-by-row accumulation of outer products
        cov = np.zeros((X.shape[1], Y.shape[1]), dtype=float)
        for i in range(Ne):
            cov += np.outer(Xc[i], Yc[i])
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
    rho_exp1 = np.abs(CorrXY) ** expo1
    rho_exp2 = np.abs(CorrXY) ** expo2

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


# -----------------------------------------------------------------------------
# A3. Matrix square-roots for SPD matrices (dense and sparse)
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# A4. Randomised SVD utilities (Halko et al., 2011)
# -----------------------------------------------------------------------------


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


def find_range(
    A: sp.spmatrix | np.ndarray,
    n_samples: int,
    n_subspace_iters: Optional[int] = None,
) -> np.ndarray:
    """Randomised range finder (Algorithm 4.1 in Halko et al.)."""
    rng = np.random.default_rng()
    _, n = A.shape
    O = rng.normal(size=(n, n_samples))
    Y = A @ O
    if n_subspace_iters and n_subspace_iters > 0:
        return subspace_iter(A, Y, int(n_subspace_iters))
    return ortho_basis(Y)


def subspace_iter(
    A: sp.spmatrix | np.ndarray,
    Y0: np.ndarray,
    n_iters: int,
) -> np.ndarray:
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


# -----------------------------------------------------------------------------
# A5. Spline fitting and bootstrap confidence bands
# -----------------------------------------------------------------------------


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


# =============================================================================
# Part B — Deterministic 1-D MT inversion
# =============================================================================


# -----------------------------------------------------------------------------
# B1. Small infrastructure (filesystem, dict helpers)
# -----------------------------------------------------------------------------


def ensure_dir(path: str | os.PathLike) -> str:
    """Create *path* if it does not exist and return it as a string."""
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p.as_posix()


def glob_inputs(pattern: str) -> List[str]:
    """Return sorted file list matching a glob *pattern*."""
    return sorted(glob.glob(pattern))


def _coerce_object_dict(obj, *, name: str = "data_dict") -> Dict:
    """Coerce *obj* to a plain Python dict.

    When loading NPZ files with ``allow_pickle=True``, nested dictionaries are
    often stored as 0-d object arrays / scalars. This helper unwraps those.
    """
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "item"):
        try:
            v = obj.item()
            if isinstance(v, dict):
                return v
        except Exception:
            pass
    raise TypeError(f"{name} must be a dict (or a 0-d object container holding a dict). Got {type(obj)!r}.")


def _extract_site_data(site: Dict) -> Tuple[Dict, bool]:
    """Return (data, wrapped) from a site container.

    Site container styles (normalized to flat dict)
    -------------------------------
    1) **Flat**: the site dict directly contains keys like ``freq``, ``Z``, ...
    2) **Wrapped**: the site dict contains a nested dict under ``site['data_dict']``
       that holds those keys, alongside metadata keys like ``station``.

    The boolean return value indicates whether the data were wrapped.
    """
    if "data_dict" in site:
        dd = _coerce_object_dict(site["data_dict"], name="site['data_dict']")
        # Legacy wrapper style: drop wrapper and treat nested dict as the site.
        return dd, True
    return site, False


def _store_site_data(site: Dict, data: Dict, wrapped: bool) -> Dict:
    """Return the updated site dict.

    Wrapper-style containers are no longer used. This helper always returns the
    plain site data dict.
    """
    return data


def _as_1d(a: np.ndarray, *, name: str, nl: int) -> np.ndarray:
    """Convert *a* to shape (nl,) float array and validate length."""
    a = np.asarray(a)
    if a.ndim != 1 or a.size != nl:
        raise ValueError(f"{name} must have shape (nl,) with nl={nl}. Got {a.shape}.")
    return a.astype(float, copy=False)


# -----------------------------------------------------------------------------
# B2. Model parameterization and I/O
# -----------------------------------------------------------------------------


def model_from_direct(model_direct: Dict) -> Dict[str, np.ndarray]:
    """Validate and copy a model dict given inline in a script."""
    if "h_m" not in model_direct:
        raise KeyError("model_direct must contain 'h_m'.")
    model = {k: np.asarray(v) for k, v in model_direct.items()}
    return normalize_model(model)


def normalize_model(model: Dict) -> Dict[str, np.ndarray]:
    """Normalize a model dict to the simplified parameterization.

    This project supports two equivalent parameter domains:

    - Resistivity domain:
        ``rho_min``, ``rho_max`` (Ohm·m), ``strike_deg``.
    - Conductivity domain:
        ``sigma_min``, ``sigma_max`` (S/m), ``strike_deg``.

    For convenience, legacy keys are also accepted:

    - ``rop`` (nl,3) principal resistivities. Only the first two principal values
      are used to define ``rho_min`` and ``rho_max``.
    - ``ustr_deg`` (nl,) is mapped to ``strike_deg``.

    The returned dict always contains **both** rho and sigma fields so that
    downstream code (deterministic inversion, plotting) can work independent of
    the chosen parametrization.

    Returns
    -------
    dict
        A *new* dict with consistent numpy arrays.
    """
    if "h_m" not in model:
        raise KeyError("Model must contain 'h_m'.")

    h_m = np.asarray(model["h_m"], dtype=float).ravel()
    nl = h_m.size

    # Warn on legacy keys (still accepted for backwards compatibility).
    _legacy_keys_found = [k for k in ("rop", "ustr_deg", "udip_deg", "usla_deg") if k in model]
    if _legacy_keys_found:
        warnings.warn(
            f"Legacy parameterization key(s) {_legacy_keys_found} are deprecated. "
            "Prefer (rho_min, rho_max, strike_deg) or (sigma_min, sigma_max, strike_deg).",
            FutureWarning,
            stacklevel=2,
        )

    # Reject unit-suffixed keys.
    for k in model.keys():
        kl = str(k).lower()
        if ("_ohmm" in kl) or ("_spm" in kl):
            raise KeyError(
                f"Unit-suffixed key '{k}' is not supported. "
                "Drop units from keys (e.g., use 'rho_min' not 'rho_min_<unit>')."
            )

    out: Dict[str, np.ndarray] = {"h_m": h_m}

    # masks
    is_iso = np.asarray(model.get("is_iso", np.zeros(nl, dtype=bool)), dtype=bool).ravel()
    is_fix = np.asarray(model.get("is_fix", np.zeros(nl, dtype=bool)), dtype=bool).ravel()
    if is_iso.size != nl or is_fix.size != nl:
        raise ValueError("is_iso and is_fix must have the same length as h_m.")
    out["is_iso"] = is_iso
    out["is_fix"] = is_fix

    # Helper: find first available key in a list
    def _pick(keys: Sequence[str]) -> Optional[str]:
        for k in keys:
            if k in model:
                return k
        return None

    # Preferred: explicit rho
    if ("rho_min" in model) and ("rho_max" in model):
        rho_min = _as_1d(model["rho_min"], name="rho_min", nl=nl)
        rho_max = _as_1d(model["rho_max"], name="rho_max", nl=nl)
        strike = _as_1d(model.get("strike_deg", np.zeros(nl)), name="strike_deg", nl=nl)

        # enforce positivity and ordering
        rho_min = np.maximum(rho_min, np.finfo(float).tiny)
        rho_max = np.maximum(rho_max, np.finfo(float).tiny)
        rlo = np.minimum(rho_min, rho_max)
        rhi = np.maximum(rho_min, rho_max)
        rho_min, rho_max = rlo, rhi

        # isotropic layers
        rho_max = np.where(is_iso, rho_min, rho_max)

        sigma_min = 1.0 / rho_max
        sigma_max = 1.0 / rho_min

    # Alternative: explicit sigma
    elif (_pick(("sigma_min", "sig_min")) is not None) and (
        _pick(("sigma_max", "sig_max")) is not None
    ):
        kmin = _pick(("sigma_min", "sig_min"))
        kmax = _pick(("sigma_max", "sig_max"))
        assert kmin is not None and kmax is not None

        sigma_min = _as_1d(model[kmin], name=kmin, nl=nl)
        sigma_max = _as_1d(model[kmax], name=kmax, nl=nl)
        strike = _as_1d(model.get("strike_deg", np.zeros(nl)), name="strike_deg", nl=nl)

        sigma_min = np.maximum(sigma_min, np.finfo(float).tiny)
        sigma_max = np.maximum(sigma_max, np.finfo(float).tiny)
        slo = np.minimum(sigma_min, sigma_max)
        shi = np.maximum(sigma_min, sigma_max)
        sigma_min, sigma_max = slo, shi

        sigma_min = np.where(is_iso, sigma_max, sigma_min)

        # Map to resistivities used by the forward model:
        # rho_min corresponds to max conductivity; rho_max to min conductivity.
        rho_min = 1.0 / sigma_max
        rho_max = 1.0 / sigma_min

    # Legacy: rop + ustr_deg
    elif "rop" in model:
        rop = np.asarray(model["rop"], dtype=float)
        if rop.shape != (nl, 3):
            raise ValueError("Legacy 'rop' must have shape (nl,3).")
        rho_min = np.minimum(rop[:, 0], rop[:, 1])
        rho_max = np.maximum(rop[:, 0], rop[:, 1])
        if "ustr_deg" in model:
            strike = _as_1d(model["ustr_deg"], name="ustr_deg", nl=nl)
        else:
            strike = np.zeros(nl, dtype=float)

        rho_min = np.maximum(rho_min, np.finfo(float).tiny)
        rho_max = np.maximum(rho_max, np.finfo(float).tiny)
        rho_max = np.where(is_iso, rho_min, rho_max)

        sigma_min = 1.0 / rho_max
        sigma_max = 1.0 / rho_min

    else:
        raise KeyError(
            "Model must provide either (rho_min,rho_max) or "
            "(sigma_min,sigma_max) or legacy 'rop'."
        )

    out["rho_min"] = np.asarray(rho_min, dtype=float)
    out["rho_max"] = np.asarray(rho_max, dtype=float)
    out["sigma_min"] = np.asarray(sigma_min, dtype=float)
    out["sigma_max"] = np.asarray(sigma_max, dtype=float)
    out["strike_deg"] = np.asarray(strike, dtype=float)

    # pass through optional label
    if "prior_name" in model:
        out["prior_name"] = np.asarray(model["prior_name"])  # type: ignore

    return out


def save_model_npz(model: Dict, path: str) -> None:
    """Save a model dict to NPZ."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        p.as_posix(),
        **{k: np.asarray(v) for k, v in normalize_model(model).items() if k != "prior_name"},
        prior_name=str(model.get("prior_name", "")),
    )


def load_model_npz(path: str) -> Dict[str, np.ndarray]:
    """Load a model dict from NPZ."""
    with np.load(Path(path).expanduser(), allow_pickle=True) as npz:
        model = {k: npz[k] for k in npz.files}
    return normalize_model(model)


def load_site(path: str) -> Dict:
    """Load site data from .npz or .edi.

    Notes
    -----
    This function delegates to ``data_proc.load_edi`` when available.
    """
    p = Path(path).expanduser()
    if p.suffix.lower() == ".npz":
        with np.load(p, allow_pickle=True) as npz:
            d = {k: npz[k] for k in npz.files}
        # Prefer the nested data_dict if present; drop outer wrapper keys like 'station'.
        if 'data_dict' in d:
            d = _coerce_object_dict(d['data_dict'], name="npz['data_dict']")
        if 'station' not in d:
            d['station'] = p.stem
        return d

    if p.suffix.lower() == ".edi":
        if data_proc is None or not hasattr(data_proc, "load_edi"):
            raise ImportError("data_proc.load_edi not available to read .edi files.")
        d = data_proc.load_edi(p.as_posix())
        if isinstance(d, dict) and "station" not in d:
            d["station"] = p.stem
        return d

    raise ValueError(f"Unsupported file type: {p.suffix}")


# -----------------------------------------------------------------------------
# B3. Phase tensor computation
# -----------------------------------------------------------------------------


def ensure_phase_tensor(site: Dict, *, nsim: int = 200) -> Dict:
    """Ensure that *site* contains phase tensor ``P`` (and optionally ``P_err``).

    Accepts both flat site dicts and wrapped dicts with a nested ``data_dict``.

    This function **always recomputes** ``P`` from the observed impedance using

        P = inv(Re(Z)) @ Im(Z)

    If ``P_err`` is missing and ``Z_err`` is present, an approximate ``P_err``
    is computed by Monte-Carlo propagation.
    """
    data, wrapped = _extract_site_data(site)
    if "Z" not in data:
        raise KeyError("site must contain 'Z' (in the flat dict or in site['data_dict']) to compute 'P'.")

    data2 = dict(data)
    Z = np.asarray(data2["Z"], dtype=np.complex128)
    data2["P"] = _phase_tensor_from_Z(Z, reg=0.0)

    if ("P_err" not in data2) and ("Z_err" in data2):
        try:
            data2["P_err"] = _bootstrap_P_err_from_Z_err(
                Z, np.asarray(data2["Z_err"]), nsim=int(nsim)
            )
        except Exception:
            # best-effort; keep going without P_err
            pass

    return _store_site_data(site, data2, wrapped)


def _pt_solve_reg(A: np.ndarray, B: np.ndarray, reg: float) -> np.ndarray:
    """Solve A X = B with optional small diagonal regularization.

    The regularization uses a scale based on the infinity-norm of A:

        A_reg = A + (reg * max(1, ||A||_inf)) * I

    Parameters
    ----------
    A : ndarray, shape (2,2)
        System matrix.
    B : ndarray, shape (2,2)
        Right-hand side.
    reg : float
        Relative regularization level.

    Returns
    -------
    ndarray, shape (2,2)
        Solution X.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if reg > 0.0:
        scale = float(max(1.0, np.linalg.norm(A, ord=np.inf)))
        A = A + (reg * scale) * np.eye(2, dtype=float)
    try:
        return np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, B, rcond=None)[0]


def _phase_tensor_from_Z(Z: np.ndarray, *, reg: float = 0.0) -> np.ndarray:
    """Compute phase tensor **P = inv(Re(Z)) @ Im(Z)**.

    This project uses a single, fixed convention:

        X = Re(Z), Y = Im(Z), P = X^{-1} Y

    Parameters
    ----------
    Z : ndarray, shape (nper, 2, 2)
        Complex impedance tensor.
    reg : float
        Small relative diagonal regularization when solving X P = Y.

    Returns
    -------
    ndarray, shape (nper, 2, 2)
        Real phase tensor.
    """
    Z = np.asarray(Z, dtype=np.complex128)
    if Z.ndim != 3 or Z.shape[1:] != (2, 2):
        raise ValueError("Z must have shape (nper,2,2).")
    nper = Z.shape[0]
    P = np.empty((nper, 2, 2), dtype=float)
    for i in range(nper):
        X = Z[i].real
        Y = Z[i].imag
        # Solve X * P = Y
        P[i] = _pt_solve_reg(X, Y, float(reg))
    return P


def _bootstrap_P_err_from_Z_err(
    Z: np.ndarray,
    Z_err: np.ndarray,
    *,
    nsim: int = 200,
    random_state: int | None = 0,
) -> np.ndarray:
    """Approximate P_err by Monte Carlo propagation from Z_err.

    Assumes independent Gaussian errors for Re/Im parts of each Z entry.
    """
    Z = np.asarray(Z, dtype=np.complex128)
    Z_err = np.asarray(Z_err, dtype=float)
    if Z_err.shape != Z.shape:
        raise ValueError("Z_err must have the same shape as Z.")

    rng = np.random.default_rng(None if random_state is None else int(random_state))
    nsim = int(nsim)
    Ps = np.empty((nsim, *Z.shape), dtype=float)
    for k in range(nsim):
        nre = rng.standard_normal(Z.shape)
        nim = rng.standard_normal(Z.shape)
        Zs = Z + (nre + 1j * nim) * Z_err
        Ps[k] = _phase_tensor_from_Z(Zs)
    return Ps.std(axis=0, ddof=1)


# -----------------------------------------------------------------------------
# B4. Impedance data packing
# -----------------------------------------------------------------------------


def _pack_Z_obs(
    Z: np.ndarray,
    Z_err: Optional[np.ndarray],
    *,
    comps: Sequence[str] = ("xx", "xy", "yx", "yy"),
    sigma_floor: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pack complex impedance into a real observation vector and sigma."""
    Z = np.asarray(Z, dtype=np.complex128)
    if Z.shape[-2:] != (2, 2):
        raise ValueError("Z must have shape (nper,2,2).")

    idx = [_COMP_MAP[c] for c in comps]

    zr = np.stack([Z[:, i, j].real for i, j in idx], axis=1)
    zi = np.stack([Z[:, i, j].imag for i, j in idx], axis=1)
    y = np.concatenate([zr.reshape(-1), zi.reshape(-1)]).astype(float)

    if Z_err is None:
        sigma = np.ones_like(y)
    else:
        Ze = np.asarray(Z_err)
        if Ze.shape == Z.shape:
            se = np.stack([np.abs(Ze[:, i, j]) for i, j in idx], axis=1)
        else:
            se = np.ones_like(zr)
        sigma = np.concatenate([se.reshape(-1), se.reshape(-1)]).astype(float)

    if sigma_floor > 0:
        sigma = np.maximum(sigma, float(sigma_floor))

    return y, sigma


def _pack_Z_jac(
    dZ: np.ndarray,
    *,
    comps: Sequence[str] = ("xx", "xy", "yx", "yy"),
) -> np.ndarray:
    """Pack complex sensitivity ``dZ`` into a real Jacobian block.

    Parameters
    ----------
    dZ : ndarray
        Complex sensitivity array shape (nper, 2, 2).

    Returns
    -------
    ndarray
        Real Jacobian column vector of length nobs for the selected components.
    """
    dZ = np.asarray(dZ, dtype=np.complex128)
    idx = [_COMP_MAP[c] for c in comps]

    dr = np.stack([dZ[:, i, j].real for i, j in idx], axis=1)
    di = np.stack([dZ[:, i, j].imag for i, j in idx], axis=1)
    return np.concatenate([dr.reshape(-1), di.reshape(-1)]).astype(float)


# -----------------------------------------------------------------------------
# B5. Parameter spec and Gauss–Newton solver
# -----------------------------------------------------------------------------


class ParamSpec:
    """Configuration container for inversion parameter bounds and masks.

    Notes
    -----
    To match the MCMC sampler, deterministic inversion supports both
    *domains* (rho vs sigma) and two *parameter sets*:

    - ``param_domain``:
        - ``"rho"``   : sample/invert in resistivity (Ohm·m)
        - ``"sigma"`` : sample/invert in conductivity (S/m)

    - ``param_set``:
        - ``"minmax"``     : (min, max, strike) per layer
        - ``"max_anifac"`` : (max, anifac, strike) per layer, where
                             ``anifac = sqrt(max/min) >= 1``

    Bounds are expressed in log10 of the chosen domain variable, i.e.
    ``log10_param_bounds`` applies to log10(rho) when ``param_domain="rho"``
    and to log10(sigma) when ``param_domain="sigma"``.
    """

    def __init__(
        self,
        *,
        nl: int,
        fix_h: bool = True,
        sample_last_thickness: bool = False,
        log10_h_bounds: Tuple[float, float] = (0.0, 5.0),
        log10_param_bounds: Tuple[float, float] = (0.0, 5.0),
        log10_anifac_bounds: Tuple[float, float] = (0.0, 2.0),
        strike_bounds_deg: Tuple[float, float] = (-180.0, 180.0),
        param_domain: str = "rho",
        param_set: str = "minmax",
    ) -> None:
        self.nl = int(nl)
        self.fix_h = bool(fix_h)
        self.sample_last_thickness = bool(sample_last_thickness)
        self.log10_h_bounds = tuple(map(float, log10_h_bounds))
        self.log10_param_bounds = tuple(map(float, log10_param_bounds))
        self.log10_anifac_bounds = tuple(map(float, log10_anifac_bounds))
        self.strike_bounds_deg = tuple(map(float, strike_bounds_deg))
        self.param_domain = str(param_domain).lower().strip()
        self.param_set = str(param_set).lower().strip()

    # Backwards-compatible alias used by older scripts
    @property
    def log10_rho_bounds(self) -> Tuple[float, float]:
        return self.log10_param_bounds


def _ridge_solve(J: np.ndarray, r: np.ndarray, lam: float) -> np.ndarray:
    """Solve (J^T J + lam^2 I) dm = J^T r."""
    JTJ = J.T @ J
    n = JTJ.shape[0]
    A = JTJ + (float(lam) ** 2) * np.eye(n)
    b = J.T @ r
    return np.linalg.solve(A, b)


def _periods_from_site(site: Dict) -> np.ndarray:
    """Return periods (s) from a site container.

    Accepts both flat site dicts and wrapped dicts with a nested ``data_dict``.
    """
    data, _wrapped = _extract_site_data(site)
    if "freq" not in data:
        raise KeyError("site must contain 'freq' to compute periods.")
    freq = np.asarray(data["freq"], dtype=float).ravel()
    if np.any(freq <= 0.0):
        raise ValueError("All frequencies must be positive.")
    return 1.0 / freq


def invert_site(
    site: Dict,
    *,
    spec: ParamSpec,
    model0: Dict,
    method: str = "tikhonov",
    lam: float = 1.0,
    lam_select: str = "fixed",
    lam_grid: Optional[np.ndarray] = None,
    lam_ngrid: int = 40,
    lam_min: Optional[float] = None,
    lam_max: Optional[float] = None,
    reg_order: int = 1,
    include_thickness_in_L: bool = False,
    tsvd_k: Optional[int] = None,
    tsvd_select: str = "fixed",
    tsvd_k_min: int = 1,
    tsvd_k_max: Optional[int] = None,
    tsvd_rcond: Optional[float] = 1e-3,
    max_iter: int = 15,
    tol: float = 1e-3,
    step_scale: float = 1.0,
    use_pt: bool = False,
    z_comps: Sequence[str] = ("xx", "xy", "yx", "yy"),
    pt_comps: Sequence[str] = ("xx", "xy", "yx", "yy"),
    sigma_floor_Z: float = 0.0,
    sigma_floor_P: float = 0.0,
) -> Dict:
    """Invert one site using a Gauss–Newton loop.

    This deterministic solver mirrors the sampler's simplified parameter options.

    - ``spec.param_domain`` in {"rho","sigma"}
    - ``spec.param_set`` in {"minmax","max_anifac"}

    If ``use_pt=True`` the inversion includes phase tensor components. Their
    sensitivities are approximated efficiently using a directional derivative
    through the Z sensitivities (no extra forward runs per parameter).

    Returns
    -------
    dict
        Inversion result dictionary suitable for saving with
        :func:`save_inversion_npz`.
    """
    model = normalize_model(model0)
    nl = int(model["h_m"].size)
    if nl != spec.nl:
        raise ValueError("spec.nl must match model0 layer count.")

    domain = str(getattr(spec, "param_domain", "rho")).lower().strip()
    pset = str(getattr(spec, "param_set", "minmax")).lower().strip()
    if domain not in ("rho", "sigma"):
        raise ValueError("spec.param_domain must be 'rho' or 'sigma'.")
    if pset not in ("minmax", "max_anifac"):
        raise ValueError("spec.param_set must be 'minmax' or 'max_anifac'.")

    # data
    site_full = ensure_phase_tensor(site, nsim=200) if use_pt else site
    data, _wrapped = _extract_site_data(site_full)

    periods_s = _periods_from_site(site_full)
    Zobs = np.asarray(data["Z"], dtype=np.complex128)
    Zerr = data.get("Z_err", None)

    y_obs_Z, sigma_Z = _pack_Z_obs(Zobs, Zerr, comps=z_comps, sigma_floor=sigma_floor_Z)

    # optional PT data (phase tensor)
    y_obs_P = None
    sigma_P = None
    if use_pt and ("P" in data):
        Pobs = np.asarray(data["P"], dtype=float)
        Perr = data.get("P_err", None)
        idx = [_COMP_MAP[c] for c in pt_comps]
        y_obs_P = np.stack([Pobs[:, i, j] for i, j in idx], axis=1).reshape(-1)
        if Perr is None:
            sigma_P = np.ones_like(y_obs_P)
        else:
            Perr = np.asarray(Perr)
            if Perr.shape == Pobs.shape:
                sigma_P = np.stack([np.abs(Perr[:, i, j]) for i, j in idx], axis=1).reshape(-1)
            else:
                sigma_P = np.ones_like(y_obs_P)
        if sigma_floor_P > 0:
            sigma_P = np.maximum(sigma_P, float(sigma_floor_P))

    # masks
    is_fix = np.asarray(model.get("is_fix", np.zeros(nl, dtype=bool)), dtype=bool)
    is_iso = np.asarray(model.get("is_iso", np.zeros(nl, dtype=bool)), dtype=bool)

    # thickness is fixed in this simplified deterministic solver
    h_m = model["h_m"].astype(float, copy=True)

    # --- initialize working parameters ---
    ln10 = float(np.log(10.0))

    if domain == "rho":
        rho_min0 = np.asarray(model["rho_min"], dtype=float)
        rho_max0 = np.asarray(model["rho_max"], dtype=float)
        if pset == "minmax":
            p1 = np.log10(np.maximum(rho_min0, np.finfo(float).tiny))
            p2 = np.log10(np.maximum(rho_max0, np.finfo(float).tiny))
        else:
            # p1 = log10(rho_max), p2 = log10(anifac), anifac = sqrt(rho_max/rho_min) >= 1
            p1 = np.log10(np.maximum(rho_max0, np.finfo(float).tiny))
            anifac0 = np.sqrt(np.maximum(rho_max0 / np.maximum(rho_min0, np.finfo(float).tiny), 1.0))
            p2 = np.log10(np.maximum(anifac0, 1.0))
    else:
        sigma_min0 = np.asarray(model["sigma_min"], dtype=float)
        sigma_max0 = np.asarray(model["sigma_max"], dtype=float)
        if pset == "minmax":
            p1 = np.log10(np.maximum(sigma_min0, np.finfo(float).tiny))
            p2 = np.log10(np.maximum(sigma_max0, np.finfo(float).tiny))
        else:
            # p1 = log10(sigma_max), p2 = log10(anifac), anifac = sqrt(sigma_max/sigma_min) >= 1
            p1 = np.log10(np.maximum(sigma_max0, np.finfo(float).tiny))
            anifac0 = np.sqrt(np.maximum(sigma_max0 / np.maximum(sigma_min0, np.finfo(float).tiny), 1.0))
            p2 = np.log10(np.maximum(anifac0, 1.0))

    strike_deg = np.asarray(model["strike_deg"], dtype=float).copy()

    history: List[Dict[str, float]] = []

    # bounds
    plo, phi = spec.log10_param_bounds
    alo, ahi = spec.log10_anifac_bounds
    slo, shi = spec.strike_bounds_deg

    for it in range(int(max_iter)):
        # --- decode parameters to rho_min/rho_max used by forward ---
        if domain == "rho":
            if pset == "minmax":
                rho_min = 10 ** p1
                rho_max = 10 ** p2
                # enforce ordering and isotropy
                rlo = np.minimum(rho_min, rho_max)
                rhi = np.maximum(rho_min, rho_max)
                rho_min, rho_max = rlo, np.where(is_iso, rlo, rhi)
            else:
                rho_max = 10 ** p1
                anifac = 10 ** np.clip(p2, alo, ahi)
                rho_min = rho_max / (anifac ** 2)
                rho_max = np.where(is_iso, rho_min, rho_max)
            sigma_min = 1.0 / rho_max
            sigma_max = 1.0 / rho_min
        else:
            if pset == "minmax":
                sigma_min = 10 ** p1
                sigma_max = 10 ** p2
                slo2 = np.minimum(sigma_min, sigma_max)
                shi2 = np.maximum(sigma_min, sigma_max)
                sigma_min, sigma_max = np.where(is_iso, shi2, slo2), shi2
            else:
                sigma_max = 10 ** p1
                anifac = 10 ** np.clip(p2, alo, ahi)
                sigma_min = sigma_max / (anifac ** 2)
                sigma_min = np.where(is_iso, sigma_max, sigma_min)
            rho_min = 1.0 / sigma_max
            rho_max = 1.0 / sigma_min

        strike_deg = np.clip(strike_deg, slo, shi)

        fwd = aniso.aniso1d_impedance_sens_simple(
            periods_s,
            h_m,
            rho_max,
            rho_min,
            strike_deg,
            compute_sens=True,
        )
        Zcal = fwd["Z"]

        y_cal_Z, _ = _pack_Z_obs(Zcal, None, comps=z_comps, sigma_floor=0.0)

        # Optional PT prediction (also serves as P0 for Jacobian)
        y_cal_P = None
        P0 = None
        if use_pt and (y_obs_P is not None):
            P0 = _phase_tensor_from_Z(Zcal)
            idx = [_COMP_MAP[c] for c in pt_comps]
            y_cal_P = np.stack([P0[:, i, j] for i, j in idx], axis=1).reshape(-1)

        # residuals (whitened)
        rZ = (y_obs_Z - y_cal_Z) / sigma_Z
        if use_pt and (y_obs_P is not None) and (y_cal_P is not None) and (sigma_P is not None):
            rP = (y_obs_P - y_cal_P) / sigma_P
            r = np.concatenate([rZ, rP])
        else:
            r = rZ

        # Build Jacobian for Z (+ optional PT)
        cols: List[np.ndarray] = []
        names: List[Tuple[str, int]] = []

        dZ_drmin_all = np.asarray(fwd["dZ_drho_min"])
        dZ_drmax_all = np.asarray(fwd["dZ_drho_max"])
        dZ_dstr_all = np.asarray(fwd["dZ_dstrike_deg"])

        idx_pt = [_COMP_MAP[c] for c in pt_comps]

        for k in range(nl):
            if bool(is_fix[k]):
                continue

            dZ_drmin = dZ_drmin_all[:, k]
            dZ_drmax = dZ_drmax_all[:, k]
            dZ_dstr = dZ_dstr_all[:, k]

            # ----- p1 column -----
            if domain == "rho":
                if pset == "minmax":
                    dZ_dp1 = dZ_drmin * (rho_min[k] * ln10)
                else:
                    # log10(rho_max): affects both rho_max and rho_min
                    dZ_dp1 = dZ_drmax * (rho_max[k] * ln10) + dZ_drmin * (rho_min[k] * ln10)
                pname1 = "log10_rho_min" if pset == "minmax" else "log10_rho_max"
            else:
                if pset == "minmax":
                    # log10(sigma_min): affects rho_max only (rho_max = 1/sigma_min)
                    dZ_dp1 = dZ_drmax * (-rho_max[k] * ln10)
                else:
                    # log10(sigma_max): affects rho_min and rho_max (both ~ 1/sigma_max)
                    dZ_dp1 = dZ_drmin * (-rho_min[k] * ln10) + dZ_drmax * (-rho_max[k] * ln10)
                pname1 = "log10_sigma_min" if pset == "minmax" else "log10_sigma_max"

            colZ = _pack_Z_jac(dZ_dp1, comps=z_comps) / sigma_Z
            if use_pt and (y_obs_P is not None) and (sigma_P is not None):
                eps = 1e-6
                P1 = _phase_tensor_from_Z(Zcal + eps * dZ_dp1)
                dP = (P1 - P0) / eps
                colP = np.stack([dP[:, i, j] for i, j in idx_pt], axis=1).reshape(-1) / sigma_P
                col = np.concatenate([colZ, colP])
            else:
                col = colZ
            cols.append(col)
            names.append((pname1, k))

            # ----- p2 column (only if anisotropic) -----
            if not bool(is_iso[k]):
                if domain == "rho":
                    if pset == "minmax":
                        dZ_dp2 = dZ_drmax * (rho_max[k] * ln10)
                        pname2 = "log10_rho_max"
                    else:
                        # log10(anifac): affects rho_min only, rho_min ~ anifac^{-2}
                        dZ_dp2 = dZ_drmin * (rho_min[k] * (-2.0 * ln10))
                        pname2 = "log10_anifac"
                else:
                    if pset == "minmax":
                        # log10(sigma_max): affects rho_min only
                        dZ_dp2 = dZ_drmin * (-rho_min[k] * ln10)
                        pname2 = "log10_sigma_max"
                    else:
                        # log10(anifac): affects rho_max only, rho_max ~ anifac^{2}
                        dZ_dp2 = dZ_drmax * (rho_max[k] * (2.0 * ln10))
                        pname2 = "log10_anifac"

                colZ = _pack_Z_jac(dZ_dp2, comps=z_comps) / sigma_Z
                if use_pt and (y_obs_P is not None) and (sigma_P is not None):
                    eps = 1e-6
                    P1 = _phase_tensor_from_Z(Zcal + eps * dZ_dp2)
                    dP = (P1 - P0) / eps
                    colP = np.stack([dP[:, i, j] for i, j in idx_pt], axis=1).reshape(-1) / sigma_P
                    col = np.concatenate([colZ, colP])
                else:
                    col = colZ
                cols.append(col)
                names.append((pname2, k))

                # ----- strike column -----
                colZ = _pack_Z_jac(dZ_dstr, comps=z_comps) / sigma_Z
                if use_pt and (y_obs_P is not None) and (sigma_P is not None):
                    eps = 1e-6
                    P1 = _phase_tensor_from_Z(Zcal + eps * dZ_dstr)
                    dP = (P1 - P0) / eps
                    colP = np.stack([dP[:, i, j] for i, j in idx_pt], axis=1).reshape(-1) / sigma_P
                    col = np.concatenate([colZ, colP])
                else:
                    col = colZ
                cols.append(col)
                names.append(("strike_deg", k))

        if not cols:
            raise RuntimeError("No free parameters (all layers fixed).")

        J = np.stack(cols, axis=1)

        # solve for update
        method_l = str(method).lower()
        if method_l not in ("tikhonov", "tsvd"):
            raise ValueError("method must be 'tikhonov' or 'tsvd'.")

        if method_l == "tikhonov":
            dm = _ridge_solve(J, r, lam=float(lam))
        else:
            U, s, Vt = np.linalg.svd(J, full_matrices=False)
            if tsvd_k is None:
                rcond = float(tsvd_rcond) if tsvd_rcond is not None else 1e-3
                k = int(np.sum(s / s[0] >= rcond))
            else:
                k = int(tsvd_k)
            k = max(1, min(k, s.size))
            dm = (Vt[:k].T @ ((U[:, :k].T @ r) / s[:k]))

        dm = float(step_scale) * dm

        # apply update
        for val, (pname, k) in zip(dm, names):
            if pname in ("log10_rho_min", "log10_rho_max", "log10_sigma_min", "log10_sigma_max"):
                # map to p1/p2 depending on configuration
                if domain == "rho":
                    if pset == "minmax":
                        if pname == "log10_rho_min":
                            p1[k] += val
                        elif pname == "log10_rho_max":
                            p2[k] += val
                    else:
                        if pname == "log10_rho_max":
                            p1[k] += val
                else:
                    if pset == "minmax":
                        if pname == "log10_sigma_min":
                            p1[k] += val
                        elif pname == "log10_sigma_max":
                            p2[k] += val
                    else:
                        if pname == "log10_sigma_max":
                            p1[k] += val
            elif pname == "log10_anifac":
                p2[k] += val
            elif pname == "strike_deg":
                strike_deg[k] += val

        # enforce bounds
        if pset == "minmax":
            p1 = np.clip(p1, plo, phi)
            p2 = np.clip(p2, plo, phi)
        else:
            p1 = np.clip(p1, plo, phi)
            p2 = np.clip(p2, alo, ahi)

        strike_deg = np.clip(strike_deg, slo, shi)

        # convergence metric
        misfit = float(np.sqrt(np.mean(r ** 2)))
        history.append({"iter": float(it), "rms": misfit})

        if misfit < float(tol):
            break

    # final decoded model (reuse decoding logic)
    if domain == "rho":
        if pset == "minmax":
            rho_min = 10 ** p1
            rho_max = 10 ** p2
            rlo = np.minimum(rho_min, rho_max)
            rhi = np.maximum(rho_min, rho_max)
            rho_min, rho_max = rlo, np.where(is_iso, rlo, rhi)
        else:
            rho_max = 10 ** p1
            anifac = 10 ** np.clip(p2, alo, ahi)
            rho_min = rho_max / (anifac ** 2)
            rho_max = np.where(is_iso, rho_min, rho_max)
        sigma_min = 1.0 / rho_max
        sigma_max = 1.0 / rho_min
    else:
        if pset == "minmax":
            sigma_min = 10 ** p1
            sigma_max = 10 ** p2
            slo2 = np.minimum(sigma_min, sigma_max)
            shi2 = np.maximum(sigma_min, sigma_max)
            sigma_min, sigma_max = np.where(is_iso, shi2, slo2), shi2
        else:
            sigma_max = 10 ** p1
            anifac = 10 ** np.clip(p2, alo, ahi)
            sigma_min = sigma_max / (anifac ** 2)
            sigma_min = np.where(is_iso, sigma_max, sigma_min)
        rho_min = 1.0 / sigma_max
        rho_max = 1.0 / sigma_min

    out_model = dict(model)
    out_model["rho_min"] = np.asarray(rho_min, dtype=float)
    out_model["rho_max"] = np.asarray(rho_max, dtype=float)
    out_model["sigma_min"] = np.asarray(sigma_min, dtype=float)
    out_model["sigma_max"] = np.asarray(sigma_max, dtype=float)
    out_model["strike_deg"] = np.asarray(strike_deg, dtype=float)

    out = {
        "station": str(site_full.get("station", data.get("station", ""))),
        "periods_s": periods_s,
        "Z_obs": Zobs,
        "Z_err": Zerr if Zerr is not None else np.array([]),
        "Z_cal": Zcal,
        "model0": model,
        "model": out_model,
        "history": np.array([(h["iter"], h["rms"]) for h in history], dtype=float),
        "param_domain": np.asarray(domain),
        "param_set": np.asarray(pset),
    }
    if use_pt and y_obs_P is not None:
        out["P_obs"] = np.asarray(data["P"], dtype=float)
        if "P_err" in data:
            out["P_err"] = np.asarray(data["P_err"], dtype=float)

    return out


# -----------------------------------------------------------------------------
# B6. Result I/O
# -----------------------------------------------------------------------------


def save_inversion_npz(res: Dict, path: str) -> None:
    """Save an inversion result dictionary to NPZ."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)

    # Flatten nested model dicts to keep NPZ simple.
    flat: Dict[str, np.ndarray] = {}

    def _put(prefix: str, d: Dict) -> None:
        for k, v in d.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                _put(prefix=key + "/", d=v)
            else:
                flat[key] = np.asarray(v)

    _put("", res)

    np.savez(p.as_posix(), **flat)
