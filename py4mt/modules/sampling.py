"""Gaussian sampling with precision matrix Q = R.T @ R.

This module provides tools to sample from multivariate Gaussian distributions
with zero mean and covariance C = (R.T @ R + lambda * I)^{-1}, where R is a
large, typically sparse matrix. The focus is on matrix-free methods that avoid
forming the covariance explicitly and that can exploit sparse linear algebra.

Main ideas
----------
1. Treat Q = R.T @ R (+ lambda * I) as a precision matrix and work with Q
   via matrix-vector products rather than forming C explicitly.
2. Use conjugate gradients (CG) on Q to solve linear systems Q x = b, which
   is the core operation in many simulation algorithms.
3. Optionally use a low-rank approximation based on eigenpairs of Q for
   reduced-rank or smoothed sampling.

The functions in this module are designed to be reasonably general while
remaining explicit about the underlying numerical linear algebra.

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-11-16
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, Dict, Literal

import scipy
import math

import numpy as np
from numpy.random import Generator, default_rng

import scipy
from scipy.spatial.distance import cdist



from scipy.sparse.linalg import LinearOperator, cg, eigsh


def build_rtr_operator(
    R: np.ndarray | "scipy.sparse.spmatrix",
    lam: float = 0.0,
) -> LinearOperator:
    """Create a LinearOperator representing Q = R.T @ R + lam * I.

    Parameters
    ----------
    R : array_like or sparse matrix, shape (m, n)
        Matrix defining the precision Q = R.T @ R. R is typically sparse.
    lam : float, optional
        Diagonal Tikhonov regularisation parameter. If non-zero, the operator
        represents Q = R.T @ R + lam * I, corresponding to covariance
        C = (R.T @ R + lam * I)^{-1}. The default is 0.0.

    Returns
    -------
    Q_op : scipy.sparse.linalg.LinearOperator, shape (n, n)
        Linear operator that applies Q to a vector via matrix-vector products.

    Notes
    -----
    This function avoids forming Q explicitly. The matvec uses two sparse
    matrix-vector products: y = R @ x and z = R.T @ y, plus a possible
    diagonal shift lam * x.

    This is suitable for use with iterative solvers such as conjugate
    gradients (CG).

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-11-16
    """
    R = R  # no copy; caller controls storage
    m, n = R.shape

    def matvec(x: np.ndarray) -> np.ndarray:
        """Matrix-vector product z = Q @ x."""
        y = R @ x
        z = R.T @ y
        if lam != 0.0:
            z = z + lam * x
        return z

    return LinearOperator((n, n), matvec=matvec, rmatvec=matvec, dtype=np.float64)


def make_cg_precision_solver(
    R: np.ndarray | "scipy.sparse.spmatrix",
    lam: float = 0.0,
    tol: float = 1e-8,
    maxiter: Optional[int] = None,
    M: Optional[LinearOperator] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Construct a solver for Q x = b with Q = R.T @ R + lam * I using CG.

    Parameters
    ----------
    R : array_like or sparse matrix, shape (m, n)
        Matrix defining the precision Q = R.T @ R (+ lam * I). R is typically
        sparse and should be such that Q is symmetric positive definite.
    lam : float, optional
        Diagonal Tikhonov regularisation parameter. The default is 0.0.
    tol : float, optional
        Relative tolerance for the conjugate-gradient solver. The default is
        1e-8.
    maxiter : int, optional
        Maximum number of CG iterations. If None, SciPy chooses a default.
        The default is None.
    M : scipy.sparse.linalg.LinearOperator, optional
        Preconditioner for Q. If provided, M should approximate Q^{-1} in
        some sense and be inexpensive to apply. The default is None.

    Returns
    -------
    solve_Q : callable
        Function ``solve_Q(b: np.ndarray) -> np.ndarray`` that returns the
        CG solution x of Q x = b.

    Notes
    -----
    This wrapper hides the SciPy interface and provides a convenient closure
    that can be used inside sampling routines. For badly conditioned systems
    it is recommended to supply an appropriate preconditioner M to improve
    convergence.


    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-11-16
    """
    Q_op = build_rtr_operator(R, lam=lam)

    def solve_Q(b: np.ndarray) -> np.ndarray:
        """Solve Q x = b with conjugate gradients."""
        x, info = cg(Q_op, b, tol=tol, maxiter=maxiter, M=M)
        if info != 0:
            raise RuntimeError(f"CG did not converge, info={info}")
        return x

    return solve_Q


def sample_gaussian_precision_rtr(
    R: np.ndarray | "scipy.sparse.spmatrix",
    n_samples: int = 1,
    lam: float = 0.0,
    solver: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    rng: Optional[Generator] = None,
) -> np.ndarray:
    """Sample from N(0, C) with C = (R.T @ R + lam * I)^{-1}.

    Parameters
    ----------
    R : array_like or sparse matrix, shape (m, n)
        Matrix defining the precision Q = R.T @ R (+ lam * I). R should have
        full column rank (or be regularised via lam > 0) so that Q is
        positive definite.
    n_samples : int, optional
        Number of independent samples to generate. The default is 1.
    lam : float, optional
        Diagonal Tikhonov regularisation parameter. If non-zero, the precision
        is Q = R.T @ R + lam * I and the covariance is
        C = (R.T @ R + lam * I)^{-1}. The default is 0.0.
    solver : callable, optional
        Existing solver for Q x = b. If None, a CG-based solver is created
        via :func:`make_cg_precision_solver`. The default is None.
    rng : numpy.random.Generator, optional
        Random number generator to use. If None, ``default_rng()`` is used.

    Returns
    -------
    samples : ndarray, shape (n_samples, n)
        Array of Gaussian samples. Each row is one draw x ~ N(0, C).

    Notes
    -----
    The sampling algorithm uses the identity

        x = argmin_y ||R y - xi||_2^2

    with xi ~ N(0, I_m), which yields

        x = (R.T @ R + lam * I)^{-1} R.T @ xi,

    and hence Cov(x) = (R.T @ R + lam * I)^{-1}. Each sample requires one
    multiplication by R and R.T and one solve with Q.

    This scheme is closely related to simulation of Gaussian Markov random
    fields using sparse precision matrices.


    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-11-16
    """
    rng = default_rng() if rng is None else rng
    m, n = R.shape

    if solver is None:
        solver = make_cg_precision_solver(R, lam=lam)

    samples = np.empty((n_samples, n), dtype=np.float64)

    for i in range(n_samples):
        xi = rng.standard_normal(size=m)
        b = R.T @ xi
        samples[i, :] = solver(b)

    return samples


def sample_low_rank_from_precision_eigpairs(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    n_samples: int = 1,
    sigma2_residual: float = 0.0,
    rng: Optional[Generator] = None,
) -> np.ndarray:
    """Sample approximately from N(0, Q^{-1}) using low-rank eigendecomposition.

    Parameters
    ----------
    eigvals : ndarray, shape (k,)
        Eigenvalues of the precision matrix Q corresponding to the columns
        of ``eigvecs``. Typically these are the smallest eigenvalues,
        representing the directions of largest variance.
    eigvecs : ndarray, shape (n, k)
        Matrix of eigenvectors of Q. Columns are assumed orthonormal.
    n_samples : int, optional
        Number of independent samples to generate. The default is 1.
    sigma2_residual : float, optional
        Isotropic residual variance added in directions orthogonal to the
        span of ``eigvecs``. If positive, the effective covariance is
        C ≈ V Λ^{-1} V.T + sigma2_residual * I, where V and Λ are given by
        eigvecs and eigvals. The default is 0.0 (pure low-rank covariance).
    rng : numpy.random.Generator, optional
        Random number generator to use. If None, ``default_rng()`` is used.

    Returns
    -------
    samples : ndarray, shape (n_samples, n)
        Array of approximate Gaussian samples. Each row is one draw.

    Notes
    -----
    This function implements

        C_k = V Λ^{-1} V^T

    with eigenpairs (λ_i, v_i). A sample from N(0, C_k) is given by

        x = V Λ^{-1/2} z,  z ~ N(0, I_k).

    If ``sigma2_residual > 0``, an additional isotropic component is added,

        x = V Λ^{-1/2} z_k + sqrt(sigma2_residual) * z_perp,

    where z_perp ~ N(0, I_n). In high dimensions the second term can be
    expensive; often sigma2_residual is kept at zero or used only when n is
    moderate.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-11-16
    """
    rng = default_rng() if rng is None else rng

    eigvals = np.asarray(eigvals, dtype=np.float64)
    eigvecs = np.asarray(eigvecs, dtype=np.float64)

    if eigvals.ndim != 1:
        raise ValueError("eigvals must be a 1D array of eigenvalues.")
    if eigvecs.ndim != 2:
        raise ValueError("eigvecs must be a 2D array of eigenvectors.")
    if eigvecs.shape[1] != eigvals.shape[0]:
        raise ValueError(
            "eigvecs.shape[1] must equal eigvals.shape[0] (one eigenvalue per vector)."
        )

    n, k = eigvecs.shape
    samples = np.empty((n_samples, n), dtype=np.float64)

    inv_sqrt = 1.0 / np.sqrt(eigvals)

    for i in range(n_samples):
        z = rng.standard_normal(size=k)
        scaled = inv_sqrt * z
        x = eigvecs @ scaled

        if sigma2_residual > 0.0:
            z_perp = rng.standard_normal(size=n)
            x = x + np.sqrt(sigma2_residual) * z_perp

        samples[i, :] = x

    return samples


def estimate_low_rank_eigpairs_from_precision(
    Q: "scipy.sparse.spmatrix | LinearOperator",
    k: int,
    which: str = "SM",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute k extremal eigenpairs of a symmetric precision matrix Q.

    Parameters
    ----------
    Q : sparse matrix or LinearOperator, shape (n, n)
        Symmetric positive definite precision matrix. Q can be provided as
        a SciPy sparse matrix or as a LinearOperator with an appropriate
        matvec implementation.
    k : int
        Number of eigenpairs to compute.
    which : {{'LM', 'SM'}}, optional
        Which eigenvalues to compute. 'LM' requests the largest magnitude
        eigenvalues, 'SM' the smallest magnitude ones. For low-rank covariance
        approximations of Q^{-1}, 'SM' is typically appropriate because the
        smallest eigenvalues correspond to the largest variances in the
        covariance. The default is 'SM'.

    Returns
    -------
    eigvals : ndarray, shape (k,)
        Eigenvalues of Q.
    eigvecs : ndarray, shape (n, k)
        Corresponding eigenvectors of Q (columns).

    Notes
    -----
    This is a thin wrapper around :func:`scipy.sparse.linalg.eigsh`. For very
    large problems, more specialised eigen-solvers or problem-specific
    techniques may be required. The resulting eigenpairs can be passed to
    :func:`sample_low_rank_from_precision_eigpairs` to construct a reduced-rank
    Gaussian sampler.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-11-16
    """
    # SciPy's eigsh expects a matrix or LinearOperator; we simply forward it.
    eigvals, eigvecs = eigsh(Q, k=k, which=which)
    return eigvals, eigvecs
