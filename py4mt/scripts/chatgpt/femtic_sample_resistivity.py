
"""
femtic_sample_resistivity.py
============================
Generate a resistivity realization from a centroid-distance covariance.

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-11-10
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Literal, Dict, Tuple

from femtic_covariance import (
    load_centroids_from_txt,
    build_covariance_dense,
    build_covariance_sparse,
)

def sample_from_cov_dense(K: np.ndarray, jitter: float = 1e-12) -> np.ndarray:
    N = K.shape[0]
    z = np.random.normal(size=N).astype(np.float64)
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        K = K.copy()
        K.flat[:: N + 1] += jitter
        L = np.linalg.cholesky(K)
    return L @ z

def sample_from_cov_sparse_trunc_eig(K, k: int = 256, random_state: Optional[int] = None) -> np.ndarray:
    from scipy.sparse.linalg import eigsh
    if random_state is not None:
        np.random.seed(random_state)
    N = K.shape[0]
    k_eff = min(k, N-1) if N>1 else 1
    vals, vecs = eigsh(K, k=k_eff, which='LM')
    vals = np.clip(vals, 0.0, None)
    g = np.random.normal(size=k_eff).astype(np.float64)
    return (vecs * np.sqrt(vals)) @ g

def draw_logrho_field(
    centroids: np.ndarray,
    kernel: Literal['matern','exponential','gaussian'] = 'matern',
    sigma2: float = 1.0,
    ell: float = 1.0,
    nu: float = 1.5,
    nugget: float = 0.0,
    mean_log_rho: float = 0.0,
    strategy: Literal['dense','sparse'] = 'dense',
    radius: Optional[float] = None,
    trunc_k: int = 256,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if random_state is not None:
        np.random.seed(random_state)
    if strategy == 'dense':
        K = build_covariance_dense(centroids, kernel=kernel, sigma2=sigma2, ell=ell, nu=nu, nugget=nugget)
        x = sample_from_cov_dense(K)
    else:
        if radius is None:
            raise ValueError("radius required for sparse strategy")
        K = build_covariance_sparse(centroids, radius=radius, kernel=kernel, sigma2=sigma2, ell=ell, nu=nu, nugget=nugget, return_format='csr')
        x = sample_from_cov_sparse_trunc_eig(K, k=trunc_k, random_state=random_state)
    logrho = mean_log_rho + x
    rho = np.exp(logrho, dtype=np.float64)
    return rho, logrho

def save_npz(path: str, **arrays):
    np.savez(path, **arrays)

# Minimal smoke test (tiny grid) that writes outputs under /mnt/data/
if __name__ == "__main__":
    xs = ys = zs = np.linspace(0, 10, 2)
    X,Y,Z = np.meshgrid(xs,ys,zs, indexing="ij")
    C = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float64)
    rho, logrho = draw_logrho_field(C, kernel='matern', sigma2=0.5, ell=6.0, nu=1.5, mean_log_rho=np.log(100.0), strategy='dense', random_state=7)
    out = "/mnt/data/resistivity_sample_demo.npz"
    save_npz(out, rho=rho, logrho=logrho, centroids=C)
    print(out)
