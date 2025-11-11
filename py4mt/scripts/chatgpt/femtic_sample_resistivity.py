'''
femtic_sample_resistivity.py
============================

Sampling utilities to draw resistivity realizations from a Gaussian
random field whose covariance is defined over centroid distances.

Two strategies
--------------
1. Dense exact sampling (Cholesky) for moderate N.
2. Sparse approximate sampling via truncated eigenpairs for large N.

Resistivity is produced by exponentiating a Gaussian field:
    rho = exp(mu_log_rho + x)

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-11-11
'''
from __future__ import annotations

from typing import Literal, Optional, Tuple
import numpy as np

from femtic_covariance import build_covariance_dense, build_covariance_sparse


def sample_from_cov_dense(k: np.ndarray, jitter: float = 1.0e-12) -> np.ndarray:
    '''Sample x ~ N(0, K) via Cholesky.'''
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
    '''Approximate x ~ N(0, K) from a sparse K using top-k eigenpairs.'''
    from scipy.sparse.linalg import eigsh
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
    '''Draw a log-resistivity Gaussian field and convert to resistivity.'''
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
