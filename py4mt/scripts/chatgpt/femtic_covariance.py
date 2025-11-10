
"""
femtic_covariance.py
====================

Centroid–distance spatial covariance builders for FEMTIC meshes.

This module provides memory-aware routines to assemble covariance matrices
on an unstructured mesh from the distances between cell centroids.
Supported kernels:
- Matérn (\nu = 0.5, 1.5, 2.5 fast closed forms; generic if SciPy is available)
- Exponential (Ornstein–Uhlenbeck; Matérn with \nu=0.5)
- Gaussian (squared exponential / RBF)

Features
--------
- Dense or sparse construction (radius cutoff)
- Chunked distance computation to limit peak memory
- Optional nugget on the diagonal
- Simple loader for centroid coordinates from a whitespace-delimited file

Conventions
-----------
- Coordinates are treated in Euclidean 3D; use whatever z sign you employ
  in the mesh (commonly z positive downward in this project). Distances
  use the standard Euclidean norm.

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-11-10
"""

from __future__ import annotations

import math
from typing import Literal, Optional, Tuple, Dict

import numpy as np

try:
    from scipy.spatial.distance import cdist
except Exception:
    cdist = None  # type: ignore

try:
    from scipy.special import kv, gamma
    _has_scipy_special = True
except Exception:
    kv = None  # type: ignore
    gamma = None  # type: ignore
    _has_scipy_special = False


def load_centroids_from_txt(
    path: str,
    cols: Tuple[int, int, int] = (0, 1, 2),
    skiprows: int = 0,
    comments: str = "#",
    encoding: Optional[str] = None,
) -> np.ndarray:
    """
    Load centroid coordinates from a whitespace-delimited text file.

    Parameters
    ----------
    path : str
        File path to a text file. The file must contain at least three numeric
        columns representing x, y, z (in any units; consistent with lengthscale).
    cols : tuple of int, default (0, 1, 2)
        Zero-based column indices to extract as x, y, z.
    skiprows : int, default 0
        Number of leading rows to skip (e.g., header lines).
    comments : str, default '#'
        Character denoting comment lines to be ignored by NumPy's loader.
    encoding : Optional[str], default None
        Optional file encoding passed to NumPy's loader when applicable.

    Returns
    -------
    centroids : (N, 3) float64 ndarray
        Array of centroid coordinates.

    Notes
    -----
    - This is intentionally simple and permissive. For custom mesh formats,
      precompute centroids separately and pass them directly.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-11-10
    """
    kwargs: Dict = dict(comments=comments)
    if encoding is not None:
        kwargs["encoding"] = encoding
    data = np.loadtxt(path, skiprows=skiprows, **kwargs)
    if data.ndim == 1:
        raise ValueError("File appears to have a single row; need an (N, M) table.")
    x, y, z = (data[:, cols[0]], data[:, cols[1]], data[:, cols[2]])
    centroids = np.column_stack([x, y, z]).astype(np.float64, copy=False)
    return centroids


def _pairwise_distances(A: np.ndarray, B: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute Euclidean pairwise distances between rows of A (and B if provided).

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-11-10
    """
    if B is None:
        B = A
    if cdist is not None:
        return cdist(A, B, metric="euclidean").astype(np.float64, copy=False)
    A2 = np.sum(A * A, axis=1)[:, None]
    B2 = np.sum(B * B, axis=1)[None, :]
    D2 = np.maximum(A2 + B2 - 2.0 * (A @ B.T), 0.0)
    return np.sqrt(D2, dtype=np.float64)


def _matern_closed_form(D: np.ndarray, ell: float, nu: float) -> np.ndarray:
    """Closed-form Matérn for nu in {0.5, 1.5, 2.5}.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-11-10
    """
    if ell <= 0:
        raise ValueError("ell must be > 0")
    r = D / float(ell)
    if np.isclose(nu, 0.5):
        return np.exp(-r, dtype=np.float64)
    elif np.isclose(nu, 1.5):
        c = math.sqrt(3.0)
        return (1.0 + c * r) * np.exp(-c * r, dtype=np.float64)
    elif np.isclose(nu, 2.5):
        c = math.sqrt(5.0)
        return (1.0 + c * r + 5.0 * (r * r) / 3.0) * np.exp(-c * r, dtype=np.float64)
    else:
        raise ValueError("Closed-form Matérn supports only nu in {0.5, 1.5, 2.5}.")


def matern_kernel(D: np.ndarray, sigma2: float = 1.0, ell: float = 1.0, nu: float = 1.5) -> np.ndarray:
    """Matérn covariance kernel.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-11-10
    """
    if sigma2 <= 0 or ell <= 0 or nu <= 0:
        raise ValueError("sigma2, ell, and nu must be > 0")
    try:
        from scipy.special import kv, gamma
        r = D / float(ell)
        t = np.sqrt(2.0 * nu, dtype=np.float64) * r
        K = np.empty_like(D, dtype=np.float64)
        mask = t > 0
        if np.any(mask):
            coef = (2.0 ** (1.0 - nu)) / float(gamma(nu))
            tm = t[mask]
            K[mask] = coef * (tm ** nu) * kv(nu, tm)
        K[~mask] = 1.0
        return sigma2 * K
    except Exception:
        base = _matern_closed_form(D, ell=ell, nu=nu)
        return sigma2 * base


def exponential_kernel(D: np.ndarray, sigma2: float = 1.0, ell: float = 1.0) -> np.ndarray:
    """Exponential covariance kernel (nu=0.5).

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-11-10
    """
    if sigma2 <= 0 or ell <= 0:
        raise ValueError("sigma2 and ell must be > 0")
    return sigma2 * np.exp(-D / float(ell), dtype=np.float64)


def gaussian_kernel(D: np.ndarray, sigma2: float = 1.0, ell: float = 1.0) -> np.ndarray:
    """Gaussian (RBF) covariance kernel.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-11-10
    """
    if sigma2 <= 0 or ell <= 0:
        raise ValueError("sigma2 and ell must be > 0")
    r2 = (D / float(ell)) ** 2
    return sigma2 * np.exp(-0.5 * r2, dtype=np.float64)


def build_covariance_dense(centroids: np.ndarray, kernel: Literal["matern","exponential","gaussian"]="matern",
                           sigma2: float = 1.0, ell: float = 1.0, nu: float = 1.5, nugget: float = 0.0) -> np.ndarray:
    """Dense covariance assembly.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-11-10
    """
    if centroids.ndim != 2 or centroids.shape[1] != 3:
        raise ValueError("centroids must be an (N, 3) array")
    D = _pairwise_distances(centroids)
    if kernel == "matern":
        K = matern_kernel(D, sigma2=sigma2, ell=ell, nu=nu)
    elif kernel == "exponential":
        K = exponential_kernel(D, sigma2=sigma2, ell=ell)
    elif kernel == "gaussian":
        K = gaussian_kernel(D, sigma2=sigma2, ell=ell)
    else:
        raise ValueError("Unknown kernel family")
    if nugget != 0.0:
        K.flat[:: K.shape[0] + 1] += nugget
    return K


def build_covariance_sparse(centroids: np.ndarray, radius: float, kernel: Literal["matern","exponential","gaussian"]="matern",
                            sigma2: float = 1.0, ell: float = 1.0, nu: float = 1.5, nugget: float = 0.0,
                            return_format: Literal["coo","csr","csc"] = "csr", leafsize: int = 32):
    """Sparse covariance within a cutoff radius.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-11-10
    """
    try:
        from scipy import sparse
        from scipy.spatial import cKDTree
    except Exception as e:
        N = centroids.shape[0]
        rows, cols, vals = [], [], []
        for i in range(N):
            rows.append(i); cols.append(i); vals.append(sigma2 + nugget)
            di = np.linalg.norm(centroids[i] - centroids, axis=1)
            idx = np.flatnonzero((di <= radius) & (di > 0))
            if kernel == "matern":
                from numpy import newaxis as _na
                kij = matern_kernel(di[_na, idx], sigma2=sigma2, ell=ell, nu=nu)[0]
            elif kernel == "exponential":
                from numpy import newaxis as _na
                kij = exponential_kernel(di[_na, idx], sigma2=sigma2, ell=ell)[0]
            else:
                from numpy import newaxis as _na
                kij = gaussian_kernel(di[_na, idx], sigma2=sigma2, ell=ell)[0]
            for j, v in zip(idx.tolist(), kij.tolist()):
                rows.append(i); cols.append(j); vals.append(v)
                rows.append(j); cols.append(i); vals.append(v)
        from scipy import sparse as _sparse
        K = _sparse.coo_matrix((vals, (rows, cols)), shape=(N, N))
        return getattr(K, "t" + return_format)()
    N = centroids.shape[0]
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
    K = sparse.coo_matrix((vals, (rows, cols)), shape=(N, N))
    return getattr(K, "t" + return_format)()
