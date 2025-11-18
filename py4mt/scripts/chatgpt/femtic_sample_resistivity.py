"""
femtic_sample_resistivity.py
============================

Sampling utilities to draw resistivity realizations from a Gaussian
random field whose covariance is defined over centroid distances.

Two strategies are provided:

1. Dense exact sampling (Cholesky) for moderate N.
2. Sparse approximate sampling via truncated eigenpairs for large N.

Resistivity is produced by exponentiating a Gaussian field:

    rho = exp(mu_log_rho + x)

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-11-18
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np

from femtic_covariance import build_covariance_dense, build_covariance_sparse


def sample_from_cov_dense(k: np.ndarray, jitter: float = 1.0e-12) -> np.ndarray:
    """
    Draw a Gaussian sample using a dense Cholesky factorization.

    Parameters
    ----------
    k
        Dense covariance matrix of shape ``(N, N)``.
    jitter
        Small diagonal term added if the Cholesky decomposition fails
        on the first attempt.

    Returns
    -------
    x
        One realization from :math:`\\mathcal{N}(0, K)`.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-11-18

    """
    n = k.shape[0]
    z = np.random.normal(size=n).astype(np.float64)

    try:
        l = np.linalg.cholesky(k)
    except np.linalg.LinAlgError:
        k = k.copy()
        k.flat[:: n + 1] += jitter
        l = np.linalg.cholesky(k)

    return l @ z


def sample_from_cov_sparse_trunc_eig(
    k,
    k_rank: int = 256,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Draw an approximate sample from a sparse covariance matrix.

    The method uses a truncated eigen-decomposition (Lanczos) to compute
    the leading ``k_rank`` eigenpairs and then samples in that subspace.

    Parameters
    ----------
    k
        Sparse covariance matrix (SciPy sparse).
    k_rank
        Number of leading eigenpairs to retain. Must satisfy
        ``1 <= k_rank < N`` where ``N`` is the matrix size.
    random_state
        Optional random seed for reproducibility.

    Returns
    -------
    x
        Approximate realization from :math:`\\mathcal{N}(0, K)`.

    Notes
    -----
    Negative eigenvalues caused by numerical errors are clipped to zero.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-11-18

    """
    from scipy.sparse.linalg import eigsh  # type: ignore

    if random_state is not None:
        np.random.seed(random_state)

    n = k.shape[0]
    k_eff = min(max(1, k_rank), max(1, n - 1))

    vals, vecs = eigsh(k, k=k_eff, which="LM")
    vals = np.clip(vals, 0.0, None)

    g = np.random.normal(size=k_eff).astype(np.float64)
    x = (vecs * np.sqrt(vals)) @ g
    return x


def draw_logrho_field(
    centroids: np.ndarray,
    kernel: Literal["matern", "exponential", "gaussian"] = "matern",
    sigma2: float = 1.0,
    ell: float = 1.0,
    nu: float = 1.5,
    nugget: float = 0.0,
    mean_log_rho: float = 0.0,
    strategy: Literal["dense", "sparse"] = "dense",
    radius: Optional[float] = None,
    trunc_k: int = 256,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draw a log-resistivity Gaussian field and convert to resistivity.

    The field is defined by a covariance matrix constructed on the cell
    centroids using the specified kernel. The resulting Gaussian field
    is shifted by ``mean_log_rho`` and exponentiated to obtain positive
    resistivities.

    Parameters
    ----------
    centroids
        Array of shape ``(N, 3)`` with cell centroids.
    kernel
        Covariance family name: ``"matern"``, ``"exponential"``,
        or ``"gaussian"``.
    sigma2
        Marginal variance of the Gaussian field in log-space.
    ell
        Correlation length scale.
    nu
        Matérn smoothness parameter (used for the Matérn kernel).
    nugget
        Diagonal nugget term added to the covariance matrix in log-space.
    mean_log_rho
        Mean of the log-resistivity field.
    strategy
        Sampling strategy: ``"dense"`` (Cholesky) or ``"sparse"``
        (truncated eigenpairs).
    radius
        Neighborhood radius for the sparse covariance. Required when
        ``strategy == "sparse"``.
    trunc_k
        Rank used for truncated-eigen sampling in the sparse case.
    random_state
        Optional random seed for reproducibility.

    Returns
    -------
    rho
        Array of shape ``(N,)`` with resistivity samples (Ohm·m).
    logrho
        Array of shape ``(N,)`` with Gaussian field values
        (log-resistivity).


    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-11-18

    """
    if random_state is not None:
        np.random.seed(random_state)

    if strategy == "dense":
        k = build_covariance_dense(
            centroids,
            kernel=kernel,
            sigma2=sigma2,
            ell=ell,
            nu=nu,
            nugget=nugget,
        )
        x = sample_from_cov_dense(k)
    else:
        if radius is None:
            raise ValueError("radius must be provided for strategy='sparse'.")
        k = build_covariance_sparse(
            centroids,
            radius=radius,
            kernel=kernel,
            sigma2=sigma2,
            ell=ell,
            nu=nu,
            nugget=nugget,
            return_format="csr",
        )
        x = sample_from_cov_sparse_trunc_eig(k, k_rank=trunc_k, random_state=random_state)

    logrho = mean_log_rho + x
    rho = np.exp(logrho, dtype=np.float64)
    return rho, logrho
