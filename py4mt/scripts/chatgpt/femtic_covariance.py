''' 
femtic_covariance.py
====================

Centroid-distance spatial covariance builders for FEMTIC meshes.

This module assembles covariance matrices between cell centroids for
unstructured meshes. It supports Matern, Exponential, and Gaussian
kernels and can build dense or sparse (radius-cutoff) matrices.

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-11-11
'''

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple
import math
import numpy as np

try:
    from scipy.spatial.distance import cdist
except Exception:
    cdist = None  # type: ignore

try:
    from scipy.special import gamma, kv
    _HAS_SCIPY_SPECIAL = True
except Exception:
    gamma = None  # type: ignore
    kv = None  # type: ignore
    _HAS_SCIPY_SPECIAL = False


def load_centroids_from_txt(
    path: str,
    cols: Tuple[int, int, int] = (0, 1, 2),
    skiprows: int = 0,
    comments: str = "#",
    encoding: Optional[str] = None,
) -> np.ndarray:
    '''Load centroid coordinates from a whitespace-delimited table.'''
    kwargs: Dict = {"comments": comments}
    if encoding is not None:
        kwargs["encoding"] = encoding
    data = np.loadtxt(path, skiprows=skiprows, **kwargs)
    if data.ndim == 1:
        raise ValueError("Need a 2-D table of centroids, got a single row.")
    x, y, z = data[:, cols[0]], data[:, cols[1]], data[:, cols[2]]
    centroids = np.column_stack((x, y, z)).astype(np.float64, copy=False)
    return centroids


def _pairwise_distances(a: np.ndarray, b: Optional[np.ndarray] = None) -> np.ndarray:
    '''Compute Euclidean pairwise distances between rows of a and b.'''
    if b is None:
        b = a
    if cdist is not None:
        return cdist(a, b, metric="euclidean").astype(np.float64, copy=False)
    a2 = np.sum(a * a, axis=1)[:, None]
    b2 = np.sum(b * b, axis=1)[None, :]
    d2 = np.maximum(a2 + b2 - 2.0 * (a @ b.T), 0.0)
    return np.sqrt(d2, dtype=np.float64)


def _matern_closed_form(d: np.ndarray, ell: float, nu: float) -> np.ndarray:
    '''Closed-form Matern for nu in {0.5, 1.5, 2.5}.'''
    if ell <= 0.0:
        raise ValueError("ell must be > 0")
    r = d / float(ell)
    if np.isclose(nu, 0.5):
        return np.exp(-r, dtype=np.float64)
    if np.isclose(nu, 1.5):
        c = math.sqrt(3.0)
        return (1.0 + c * r) * np.exp(-c * r, dtype=np.float64)
    if np.isclose(nu, 2.5):
        c = math.sqrt(5.0)
        return (1.0 + c * r + (5.0/3.0) * r * r) * np.exp(-c * r, dtype=np.float64)
    raise ValueError("Closed-form Matern supports nu in {0.5, 1.5, 2.5}.")


def matern_kernel(d: np.ndarray, sigma2: float = 1.0, ell: float = 1.0, nu: float = 1.5) -> np.ndarray:
    '''Matern covariance evaluated on a distance matrix.'''
    if sigma2 <= 0.0 or ell <= 0.0 or nu <= 0.0:
        raise ValueError("sigma2, ell, and nu must be > 0")
    if _HAS_SCIPY_SPECIAL:
        r = d / float(ell)
        t = np.sqrt(2.0 * nu, dtype=np.float64) * r
        out = np.empty_like(d, dtype=np.float64)
        mask = t > 0.0
        if np.any(mask):
            coef = (2.0 ** (1.0 - nu)) / float(gamma(nu))
            tm = t[mask]
            out[mask] = coef * (tm ** nu) * kv(nu, tm)
        out[~mask] = 1.0
        return sigma2 * out
    base = _matern_closed_form(d, ell=ell, nu=nu)
    return sigma2 * base


def exponential_kernel(d: np.ndarray, sigma2: float = 1.0, ell: float = 1.0) -> np.ndarray:
    '''Exponential covariance (Matern with nu=0.5).'''
    if sigma2 <= 0.0 or ell <= 0.0:
        raise ValueError("sigma2 and ell must be > 0")
    return sigma2 * np.exp(-d / float(ell), dtype=np.float64)


def gaussian_kernel(d: np.ndarray, sigma2: float = 1.0, ell: float = 1.0) -> np.ndarray:
    '''Gaussian (squared-exponential) covariance.'''
    if sigma2 <= 0.0 or ell <= 0.0:
        raise ValueError("sigma2 and ell must be > 0")
    r2 = (d / float(ell)) ** 2
    return sigma2 * np.exp(-0.5 * r2, dtype=np.float64)


def build_covariance_dense(
    centroids: np.ndarray,
    kernel: Literal["matern", "exponential", "gaussian"] = "matern",
    sigma2: float = 1.0,
    ell: float = 1.0,
    nu: float = 1.5,
    nugget: float = 0.0,
) -> np.ndarray:
    '''Assemble a dense covariance matrix for centroid coordinates.'''
    if centroids.ndim != 2 or centroids.shape[1] != 3:
        raise ValueError("centroids must be an (N, 3) array")
    d = _pairwise_distances(centroids)
    if kernel == "matern":
        k = matern_kernel(d, sigma2=sigma2, ell=ell, nu=nu)
    elif kernel == "exponential":
        k = exponential_kernel(d, sigma2=sigma2, ell=ell)
    elif kernel == "gaussian":
        k = gaussian_kernel(d, sigma2=sigma2, ell=ell)
    else:
        raise ValueError("unknown kernel family")
    if nugget != 0.0:
        k.flat[:: k.shape[0] + 1] += nugget
    return k


def build_covariance_sparse(
    centroids: np.ndarray,
    radius: float,
    kernel: Literal["matern", "exponential", "gaussian"] = "matern",
    sigma2: float = 1.0,
    ell: float = 1.0,
    nu: float = 1.5,
    nugget: float = 0.0,
    return_format: Literal["coo", "csr", "csc"] = "csr",
    leafsize: int = 32,
):
    '''Assemble a sparse covariance matrix keeping entries with D <= radius.'''
    try:
        from scipy import sparse
        from scipy.spatial import cKDTree
    except Exception:
        n = centroids.shape[0]
        rows, cols, vals = [], [], []
        for i in range(n):
            rows.append(i); cols.append(i); vals.append(sigma2 + nugget)
            di = np.linalg.norm(centroids[i] - centroids, axis=1)
            idx = np.flatnonzero((di <= radius) & (di > 0.0))
            from numpy import newaxis as _na
            if kernel == "matern":
                kij = matern_kernel(di[_na, idx], sigma2=sigma2, ell=ell, nu=nu)[0]
            elif kernel == "exponential":
                kij = exponential_kernel(di[_na, idx], sigma2=sigma2, ell=ell)[0]
            else:
                kij = gaussian_kernel(di[_na, idx], sigma2=sigma2, ell=ell)[0]
            for j, v in zip(idx.tolist(), kij.tolist()):
                rows.extend((i, j)); cols.extend((j, i)); vals.extend((v, v))
        from scipy import sparse as _sparse
        k = _sparse.coo_matrix((vals, (rows, cols)), shape=(n, n))
        return getattr(k, "t" + return_format)()

    n = centroids.shape[0]
    tree = cKDTree(centroids, leafsize=leafsize)
    pairs = tree.query_ball_tree(tree, r=radius)
    rows, cols, vals = [], [], []
    for i, neigh in enumerate(pairs):
        rows.append(i); cols.append(i); vals.append(sigma2 + nugget)
        if not neigh:
            continue
        neigh = [j for j in neigh if j != i]
        if not neigh:
            continue
        dij = np.linalg.norm(centroids[i] - centroids[neigh], axis=1)
        from numpy import newaxis as _na
        if kernel == "matern":
            vij = matern_kernel(dij[_na, :], sigma2=sigma2, ell=ell, nu=nu)[0]
        elif kernel == "exponential":
            vij = exponential_kernel(dij[_na, :], sigma2=sigma2, ell=ell)[0]
        else:
            vij = gaussian_kernel(dij[_na, :], sigma2=sigma2, ell=ell)[0]
        rows.extend([i]*len(neigh)); cols.extend(neigh); vals.extend(vij.tolist())
        rows.extend(neigh); cols.extend([i]*len(neigh)); vals.extend(vij.tolist())
    from scipy import sparse
    k = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n))
    return getattr(k, "t" + return_format)()
