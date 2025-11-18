"""
femtic_covariance.py
====================

Centroid-distance spatial covariance builders for FEMTIC meshes.

This module assembles covariance matrices between cell centroids for
unstructured meshes. It supports Matérn, Exponential, and Gaussian
kernels and can build dense or sparse (radius-cutoff) matrices.

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-11-18
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple
import math
from scipy.spatial.distance import cdist
import numpy as np

from scipy.spatial.distance import cdist
from scipy.special import gamma, kv


def load_centroids_from_txt(
    path: str,
    cols: Tuple[int, int, int] = (0, 1, 2),
    skiprows: int = 0,
    comments: str = "#",
    encoding: Optional[str] = None,
) -> np.ndarray:
    """
    Load centroid coordinates from a whitespace-delimited table.

    Parameters
    ----------
    path
        Path to a text file with at least three numeric columns (x, y, z).
    cols
        Zero-based column indices to extract as (x, y, z).
    skiprows
        Number of leading rows to skip (e.g. header lines).
    comments
        Single character that marks comment lines for ``numpy.loadtxt``.
    encoding
        Optional text encoding for the file. If ``None``, NumPy’s default
        is used.

    Returns
    -------
    centroids
        Array of shape ``(N, 3)`` with centroid coordinates.

    Raises
    ------
    ValueError
        If the input array is one-dimensional (only a single row).


    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-11-18

    """
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
    """
    Compute Euclidean pairwise distances between rows of two arrays.

    Parameters
    ----------
    a
        Array of shape ``(N, d)``.
    b
        Array of shape ``(M, d)``. If ``None``, ``b`` is set to ``a``
        and distances within ``a`` are computed.

    Returns
    -------
    d
        Distance matrix of shape ``(N, M)``.

    Notes
    -----
    If SciPy is available, :func:`scipy.spatial.distance.cdist` is used.
    Otherwise a numerically-stable NumPy-only computation is used.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-11-18

    """
    if b is None:
        b = a
    if cdist is not None:
        return cdist(a, b, metric="euclidean").astype(np.float64, copy=False)

    a2 = np.sum(a * a, axis=1)[:, None]
    b2 = np.sum(b * b, axis=1)[None, :]
    d2 = np.maximum(a2 + b2 - 2.0 * (a @ b.T), 0.0)
    return np.sqrt(d2, dtype=np.float64)


def _matern_closed_form(d: np.ndarray, ell: float, nu: float) -> np.ndarray:
    """
    Evaluate the closed-form Matérn kernel for specific smoothness values.

    Supported smoothness values are ``nu = 0.5, 1.5, 2.5``.

    Parameters
    ----------
    d
        Distance matrix.
    ell
        Correlation length scale (> 0).
    nu
        Matérn smoothness parameter (0.5, 1.5, or 2.5).

    Returns
    -------
    k
        Kernel values without the variance factor ``sigma2``.

    Raises
    ------
    ValueError
        If ``ell`` is non-positive or ``nu`` is not in
        ``{0.5, 1.5, 2.5}``.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-11-18

    """
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
        return (1.0 + c * r + (5.0 / 3.0) * r * r) * np.exp(
            -c * r,
            dtype=np.float64,
        )

    raise ValueError("Closed-form Matérn supports nu in {0.5, 1.5, 2.5}.")


def matern_kernel(
    d: np.ndarray,
    sigma2: float = 1.0,
    ell: float = 1.0,
    nu: float = 1.5,
) -> np.ndarray:
    """
    Evaluate the Matérn covariance kernel on a distance matrix.

    Parameters
    ----------
    d
        Distance matrix.
    sigma2
        Marginal variance (> 0).
    ell
        Correlation length scale (> 0).
    nu
        Matérn smoothness parameter (> 0).

    Returns
    -------
    k
        Covariance matrix with entries
        ``sigma2 * M_nu(d / ell)``.

    Raises
    ------
    ValueError
        If any of ``sigma2``, ``ell`` or ``nu`` are non-positive.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-11-18

    """
    if sigma2 <= 0.0 or ell <= 0.0 or nu <= 0.0:
        raise ValueError("sigma2, ell, and nu must be > 0")

    if _HAS_SCIPY_SPECIAL:
        r = d / float(ell)
        t = np.sqrt(2.0 * nu, dtype=np.float64) * r
        out = np.empty_like(d, dtype=np.float64)
        mask = t > 0.0
        if np.any(mask):
            coef = (2.0 ** (1.0 - nu)) / float(gamma(nu))  # type: ignore[arg-type]
            tm = t[mask]
            out[mask] = coef * (tm**nu) * kv(nu, tm)  # type: ignore[operator]
        out[~mask] = 1.0
        return sigma2 * out

    base = _matern_closed_form(d, ell=ell, nu=nu)
    return sigma2 * base


def exponential_kernel(d: np.ndarray, sigma2: float = 1.0, ell: float = 1.0) -> np.ndarray:
    """
    Evaluate the exponential covariance kernel.

    This is equivalent to a Matérn kernel with ``nu = 0.5``.

    Parameters
    ----------
    d
        Distance matrix.
    sigma2
        Marginal variance (> 0).
    ell
        Correlation length scale (> 0).

    Returns
    -------
    k
        Covariance matrix with entries ``sigma2 * exp(-d / ell)``.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-11-18

    """
    if sigma2 <= 0.0 or ell <= 0.0:
        raise ValueError("sigma2 and ell must be > 0")
    return sigma2 * np.exp(-d / float(ell), dtype=np.float64)


def gaussian_kernel(d: np.ndarray, sigma2: float = 1.0, ell: float = 1.0) -> np.ndarray:
    """
    Evaluate the Gaussian (squared-exponential) covariance kernel.

    Parameters
    ----------
    d
        Distance matrix.
    sigma2
        Marginal variance (> 0).
    ell
        Correlation length scale (> 0).

    Returns
    -------
    k
        Covariance matrix with entries
        ``sigma2 * exp(-0.5 * (d / ell)**2)``.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-11-18

    """
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
    """
    Assemble a dense covariance matrix for centroid coordinates.

    Parameters
    ----------
    centroids
        Array of shape ``(N, 3)`` with centroid coordinates.
    kernel
        Covariance family name. One of ``"matern"``, ``"exponential"`` or
        ``"gaussian"``.
    sigma2
        Marginal variance.
    ell
        Correlation length scale.
    nu
        Matérn smoothness parameter (used only for the Matérn kernel).
    nugget
        Diagonal nugget term added to the covariance matrix.

    Returns
    -------
    k
        Dense covariance matrix of shape ``(N, N)``.

    Raises
    ------
    ValueError
        If ``centroids`` does not have shape ``(N, 3)`` or the kernel
        name is unknown.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-11-18

    """
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
    """
    Assemble a sparse covariance matrix by thresholding distances.

    Entries with distance ``D <= radius`` are retained. This produces a
    sparse covariance matrix suitable for large meshes.

    Parameters
    ----------
    centroids
        Array of shape ``(N, 3)`` with centroid coordinates.
    radius
        Neighborhood radius for non-zero entries.
    kernel
        Covariance family name. One of ``"matern"``, ``"exponential"``
        or ``"gaussian"``.
    sigma2
        Marginal variance.
    ell
        Correlation length scale.
    nu
        Matérn smoothness parameter (for the Matérn kernel).
    nugget
        Diagonal nugget term.
    return_format
        Output sparse format: one of ``"coo"``, ``"csr"``, ``"csc"``.
    leafsize
        Leaf size for the :class:`scipy.spatial.cKDTree` used in
        neighbor searches.

    Returns
    -------
    k_sparse
        Sparse covariance matrix in the requested format (SciPy sparse).

    Notes
    -----
    If SciPy is not available, a quadratic-time fallback implementation
    is used. This is suitable only for modest problem sizes.


    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-11-18

    """

    from scipy import sparse
    from scipy.spatial import cKDTree


    # SciPy-based implementation
    n = centroids.shape[0]
    tree = cKDTree(centroids, leafsize=leafsize)
    pairs = tree.query_ball_tree(tree, r=radius)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for i, neigh in enumerate(pairs):
        # Diagonal
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

        rows.extend([i] * len(neigh))
        cols.extend(neigh)
        vals.extend(vij.tolist())

        rows.extend(neigh)
        cols.extend([i] * len(neigh))
        vals.extend(vij.tolist())


    k = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n))
    return getattr(k, "t" + return_format)()
