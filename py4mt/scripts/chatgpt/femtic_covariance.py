
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

# Optional SciPy imports for faster distance/covariance when present
try:
    from scipy.spatial.distance import cdist
except Exception:  # pragma: no cover
    cdist = None  # type: ignore

try:
    from scipy.special import kv, gamma
    _has_scipy_special = True
except Exception:  # pragma: no cover
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
        raise ValueError(
            "File appears to have a single row; need an (N, M) table.")
    x, y, z = (data[:, cols[0]], data[:, cols[1]], data[:, cols[2]])
    centroids = np.column_stack([x, y, z]).astype(np.float64, copy=False)
    return centroids


def _pairwise_distances(A: np.ndarray, B: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute Euclidean pairwise distances between rows of A (and B if provided).

    Parameters
    ----------
    A : (N, d) float64 ndarray
        First set of points.
    B : Optional[(M, d) float64 ndarray], default None
        Second set of points. If None, distances are computed among rows of A.

    Returns
    -------
    D : (N, M) or (N, N) float64 ndarray
        Matrix of pairwise Euclidean distances.

    Implementation Details
    ----------------------
    - Uses SciPy's `cdist` when present, else a stable XOR trick with norms.
    - Ensures non-negative distances via `np.maximum` to mitigate tiny negatives
      from floating point subtraction.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-11-10
    """
    if B is None:
        B = A
    if cdist is not None:
        return cdist(A, B, metric="euclidean").astype(np.float64, copy=False)
    # Fallback without SciPy
    A2 = np.sum(A * A, axis=1)[:, None]
    B2 = np.sum(B * B, axis=1)[None, :]
    # (a - b)^2 = a^2 + b^2 - 2 a·b
    D2 = np.maximum(A2 + B2 - 2.0 * (A @ B.T), 0.0)
    return np.sqrt(D2, dtype=np.float64)


def _matern_closed_form(D: np.ndarray, ell: float, nu: float) -> np.ndarray:
    """
    Closed-form Matérn for special \nu values: 0.5, 1.5, 2.5.

    K(r) = f_\nu(r/\ell).

    Parameters
    ----------
    D : ndarray
        Pairwise distances.
    ell : float
        Lengthscale (> 0).
    nu : float
        Smoothness; supported: 0.5, 1.5, 2.5.

    Returns
    -------
    K : ndarray
        Unscaled covariance (variance factor not applied).

    Raises
    ------
    ValueError
        If \nu is not one of the supported closed-form values.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-11-10
    """
    if ell <= 0:
        raise ValueError("ell must be > 0")
    r = D / float(ell)
    if np.isclose(nu, 0.5):
        # exp(-r)
        return np.exp(-r, dtype=np.float64)
    elif np.isclose(nu, 1.5):
        # (1 + sqrt(3) r) * exp(-sqrt(3) r)
        c = math.sqrt(3.0)
        return (1.0 + c * r) * np.exp(-c * r, dtype=np.float64)
    elif np.isclose(nu, 2.5):
        # (1 + sqrt(5) r + 5 r^2 / 3) * exp(-sqrt(5) r)
        c = math.sqrt(5.0)
        return (1.0 + c * r + 5.0 * (r * r) / 3.0) * np.exp(-c * r, dtype=np.float64)
    else:
        raise ValueError(
            "Closed-form Matérn supports only nu in {{0.5, 1.5, 2.5}}.")


def matern_kernel(
    D: np.ndarray,
    sigma2: float = 1.0,
    ell: float = 1.0,
    nu: float = 1.5,
) -> np.ndarray:
    """
    Matérn covariance kernel evaluated on a distance matrix.

    K = sigma^2 * M_\nu(D / ell)

    Parameters
    ----------
    D : (N, M) float64 ndarray
        Pairwise distance matrix.
    sigma2 : float, default 1.0
        Marginal variance (> 0).
    ell : float, default 1.0
        Lengthscale (> 0).
    nu : float, default 1.5
        Smoothness parameter. If SciPy is not installed, supported values are
        0.5, 1.5, 2.5. With SciPy, any positive \nu is supported.

    Returns
    -------
    K : ndarray
        Covariance matrix of shape like D.

    Notes
    -----
    - With SciPy, the general formula is used:
      M_\nu(r) = 2^{{1-\nu}}/\Gamma(\nu) * (\sqrt{{2\nu}} r)^\nu K_\nu(\sqrt{{2\nu}} r)
      with M_\nu(0)=1 by continuity.
    - Without SciPy, fast closed forms are used for common \nu.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-11-10
    """
    if sigma2 <= 0 or ell <= 0 or nu <= 0:
        raise ValueError("sigma2, ell, and nu must be > 0")
    try:
        from scipy.special import kv, gamma
        r = D / float(ell)
        t = np.sqrt(2.0 * nu, dtype=np.float64) * r
        # Handle r=0 limit: M_nu(0) = 1
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
    """
    Exponential covariance kernel (Matérn with \nu=0.5).

    K = sigma^2 * exp(-D / ell)

    Parameters
    ----------
    D : (N, M) float64 ndarray
        Pairwise distance matrix.
    sigma2 : float, default 1.0
        Marginal variance (> 0).
    ell : float, default 1.0
        Lengthscale (> 0).

    Returns
    -------
    K : ndarray
        Covariance matrix of shape like D.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-11-10
    """
    if sigma2 <= 0 or ell <= 0:
        raise ValueError("sigma2 and ell must be > 0")
    return sigma2 * np.exp(-D / float(ell), dtype=np.float64)


def gaussian_kernel(D: np.ndarray, sigma2: float = 1.0, ell: float = 1.0) -> np.ndarray:
    """
    Gaussian (squared-exponential / RBF) covariance kernel.

    K = sigma^2 * exp(-0.5 * (D/ell)^2)

    Parameters
    ----------
    D : (N, M) float64 ndarray
        Pairwise distance matrix.
    sigma2 : float, default 1.0
        Marginal variance (> 0).
    ell : float, default 1.0
        Lengthscale (> 0).

    Returns
    -------
    K : ndarray
        Covariance matrix of shape like D.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-11-10
    """
    if sigma2 <= 0 or ell <= 0:
        raise ValueError("sigma2 and ell must be > 0")
    r2 = (D / float(ell)) ** 2
    return sigma2 * np.exp(-0.5 * r2, dtype=np.float64)


def build_covariance_dense(
    centroids: np.ndarray,
    kernel: Literal["matern", "exponential", "gaussian"] = "matern",
    sigma2: float = 1.0,
    ell: float = 1.0,
    nu: float = 1.5,
    nugget: float = 0.0,
) -> np.ndarray:
    """
    Assemble a *dense* covariance matrix for the given centroids.

    Parameters
    ----------
    centroids : (N, 3) float64 ndarray
        Cell centroids.
    kernel : {{'matern', 'exponential', 'gaussian'}}, default 'matern'
        Covariance family.
    sigma2 : float, default 1.0
        Marginal variance (> 0).
    ell : float, default 1.0
        Lengthscale (> 0).
    nu : float, default 1.5
        Matérn smoothness (used only when kernel='matern').
    nugget : float, default 0.0
        Additive diagonal term (e.g., model/measurement variance).

    Returns
    -------
    K : (N, N) float64 ndarray
        Dense covariance matrix.

    Notes
    -----
    - For large N this will be memory-heavy (N^2). Consider the sparse builder.

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
        K.flat[:: K.shape[0] + 1] += nugget  # fast add to diagonal
    return K


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
    """
    Assemble a *sparse* covariance matrix by keeping entries within a radius.

    Parameters
    ----------
    centroids : (N, 3) float64 ndarray
        Cell centroids.
    radius : float
        Neighborhood radius (same units as centroids). Entries with D <= radius
        are kept; others are dropped.
    kernel : {{'matern', 'exponential', 'gaussian'}}, default 'matern'
        Covariance family.
    sigma2 : float, default 1.0
        Marginal variance (> 0).
    ell : float, default 1.0
        Lengthscale (> 0).
    nu : float, default 1.5
        Matérn smoothness (used only when kernel='matern').
    nugget : float, default 0.0
        Additive diagonal term.
    return_format : {{'coo', 'csr', 'csc'}}, default 'csr'
        Sparse format of the result.
    leafsize : int, default 32
        KDTree leafsize for neighbor queries (when SciPy is available).

    Returns
    -------
    K : scipy.sparse.spmatrix
        Symmetric sparse covariance matrix.

    Implementation Details
    ----------------------
    - Uses SciPy's `cKDTree` if available for fast radius queries; otherwise,
      falls back to a simple gridless O(N^2) radius mask (suitable for modest N).
    - Always includes the diagonal (self-covariance) with nugget.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-11-10
    """
    try:
        from scipy import sparse
        from scipy.spatial import cKDTree
    except Exception as e:  # pragma: no cover
        # Minimal fallback without SciPy: O(N^2) distance check
        N = centroids.shape[0]
        rows, cols, vals = [], [], []
        for i in range(N):
            # self
            rows.append(i)
            cols.append(i)
            vals.append(sigma2 + nugget)
            di = np.linalg.norm(centroids[i] - centroids, axis=1)
            idx = np.flatnonzero((di <= radius) & (di > 0))
            if kernel == "matern":
                from numpy import newaxis as _na
                kij = matern_kernel(
                    di[_na, idx], sigma2=sigma2, ell=ell, nu=nu)[0]
            elif kernel == "exponential":
                from numpy import newaxis as _na
                kij = exponential_kernel(
                    di[_na, idx], sigma2=sigma2, ell=ell)[0]
            else:
                from numpy import newaxis as _na
                kij = gaussian_kernel(di[_na, idx], sigma2=sigma2, ell=ell)[0]
            for j, v in zip(idx.tolist(), kij.tolist()):
                rows.append(i)
                cols.append(j)
                vals.append(v)
                rows.append(j)
                cols.append(i)
                vals.append(v)  # symmetrize
        try:
            from scipy import sparse as _sparse  # might still be present
            K = _sparse.coo_matrix((vals, (rows, cols)), shape=(N, N))
            return getattr(K, "t" + return_format)()
        except Exception:
            # Build a lightweight CSR by hand if SciPy sparse is unavailable
            raise RuntimeError(
                "SciPy sparse not available to materialize the matrix.")
    # Fast path with SciPy
    N = centroids.shape[0]
    tree = cKDTree(centroids, leafsize=leafsize)
    pairs = tree.query_ball_tree(tree, r=radius)
    rows, cols, vals = [], [], []
    for i, neigh in enumerate(pairs):
        # ensure diagonal
        rows.append(i)
        cols.append(i)
        vals.append(sigma2 + nugget)
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
        rows.extend([i]*len(neigh))
        cols.extend(neigh)
        vals.extend(vij.tolist())
        # Symmetrize explicitly
        rows.extend(neigh)
        cols.extend([i]*len(neigh))
        vals.extend(vij.tolist())
    from scipy import sparse
    K = sparse.coo_matrix((vals, (rows, cols)), shape=(N, N))
    return getattr(K, "t" + return_format)()


def save_covariance(K, path: str) -> None:
    """
    Save a covariance matrix to disk.

    Parameters
    ----------
    K : ndarray or scipy.sparse.spmatrix
        Covariance matrix.
    path : str
        Destination. If extension is '.npz', uses NumPy savez for dense arrays,
        or SciPy's sparse save if available for sparse matrices. If '.npy', uses
        np.save for dense arrays.

    Notes
    -----
    - For sparse matrices with '.npz', this will try `scipy.sparse.save_npz`.
      If SciPy is not present, it falls back to saving COO triplets as a dict.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-11-10
    """
    import numpy as _np
    import os as _os
    ext = _os.path.splitext(path)[1].lower()
    if ext not in (".npz", ".npy"):
        raise ValueError("Use .npz (preferred) or .npy extensions for saving.")
    try:
        from scipy import sparse as _sparse
        is_sparse = _sparse.isspmatrix(K)
    except Exception:
        is_sparse = False
    if is_sparse:
        try:
            from scipy.sparse import save_npz as _save_npz
            _save_npz(path, K)  # type: ignore
            return
        except Exception:
            # Fallback: save COO triplets
            K = K.tocoo()
            _np.savez(path, data=K.data, row=K.row, col=K.col,
                      shape=K.shape)  # type: ignore
            return
    # Dense
    if ext == ".npz":
        _np.savez(path, K=K)
    else:
        _np.save(path, K)


def example_usage() -> None:
    """
    Minimal example creating small dense/sparse covariances.

    This function is safe to run on tiny meshes to validate the pipeline.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-11-10
    """
    import numpy as _np
    # toy centroids: 2x2x2 grid
    xs, ys, zs = [_np.linspace(0, 10, 2) for _ in range(3)]
    X, Y, Z = _np.meshgrid(xs, ys, zs, indexing="ij")
    C = _np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(_np.float64)

    Kd = build_covariance_dense(
        C, kernel="matern", sigma2=1.0, ell=5.0, nu=1.5, nugget=1e-6)
    Ks = build_covariance_sparse(
        C, radius=6.0, kernel="gaussian", sigma2=1.0, ell=4.0, nugget=1e-6)

    # Save examples
    save_covariance(Kd, "/mnt/data/example_dense_cov.npz")
    save_covariance(Ks, "/mnt/data/example_sparse_cov.npz")


if __name__ == "__main__":
    # Example run (kept tiny to avoid heavy memory/time)
    example_usage()
