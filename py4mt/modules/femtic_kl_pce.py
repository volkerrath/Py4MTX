#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 20:09:20 2025

@author: vrath
"""

"""
femtic_kl_pce.py

Weighted Karhunen-Loève (KL) decomposition + non-intrusive Polynomial Chaos
Expansion (PCE) for log-resistivity fields defined on an unstructured FEMTIC
mesh.

The code assumes:
- You have log-resistivity samples on FEMTIC *cells* (or nodes) as
  rho_samples.shape == (n_cells, n_samples).
- You have cell volumes (or any positive integration weights) as
  volumes.shape == (n_cells,).
- You have input random samples Xi used to generate these fields as
  Xi.shape == (n_samples, M).

KL part
-------
We compute a weighted KL expansion of the centered field:

    X = rho_samples - mean_field[:, None]

using the spatial inner product

    <f, g>_w = sum_i volumes[i] * f[i] * g[i]

The spatial modes phi_k satisfy

    <phi_k, phi_l>_w = delta_kl

and the random coefficients a_k^(s) are given by

    a_k^(s) = <X(:, s), phi_k>_w

PCE part
--------
For each KL coefficient a_k(ξ) we build a scalar PCE

    a_k(ξ) ≈ Σ_j c_{k,j} Ψ_j(ξ)

where Ψ_j are multivariate orthonormal polynomials (Hermite for Gaussian,
Legendre for uniform) defined on the input random vector ξ.

The full KL+PCE surrogate for new ξ is then

    rho(x, ξ) ≈ mean_field(x) + Σ_k phi_k(x) a_k(ξ)

This module provides:
- compute_weighted_kl       : weighted KL on FEMTIC log-rho samples
- total_degree_multiindex   : multi-index generator for PCE
- build_design_matrix       : PCE design matrix Ψ(ξ)
- fit_pce_for_kl_modes      : scalar PCE for each KL mode coefficient
- evaluate_kl_pce_surrogate : evaluate KL+PCE model at new ξ
- KLPCEModel dataclass      : convenience container + model.evaluate(Xi_new)

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5.1 Thinking) on 2025-12-10 (UTC)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import itertools
import math

import numpy as np
from numpy.polynomial.hermite import hermval
from numpy.polynomial.legendre import legval


# ---------------------------------------------------------------------------
# Weighted KL decomposition on FEMTIC mesh
# ---------------------------------------------------------------------------


def compute_weighted_kl(
    rho_samples: np.ndarray,
    volumes: np.ndarray,
    n_modes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a weighted Karhunen-Loève (KL) expansion of log-rho fields on a
    FEMTIC mesh.

    Parameters
    ----------
    rho_samples : ndarray, shape (n_cells, n_samples)
        Log-resistivity samples (e.g. log10(rho)) on FEMTIC cells.
        Each column is one realization.
    volumes : ndarray, shape (n_cells,)
        Positive weights per cell (e.g. cell volumes). These define the
        spatial inner product
            <f, g>_w = sum_i volumes[i] * f[i] * g[i]
        and should be non-negative. Zero-volume cells are allowed but will
        not contribute to the inner product.
    n_modes : int
        Number of KL modes to retain.

    Returns
    -------
    mean_field : ndarray, shape (n_cells,)
        Sample mean of rho_samples over realizations.
    modes : ndarray, shape (n_cells, n_modes)
        Spatial KL modes phi_k on the FEMTIC mesh, orthonormal w.r.t. the
        weighted inner product:
            modes[:, k].T @ (volumes * modes[:, l]) ≈ delta_kl
    eigvals : ndarray, shape (n_modes,)
        Eigenvalues (mode variances) corresponding to each KL mode.
    mode_coeffs : ndarray, shape (n_modes, n_samples)
        KL coefficients a_k^(s) for each mode k and realization s:
            a_k^(s) = <rho_samples[:, s] - mean_field, modes[:, k]>_w

    Notes
    -----
    Implementation uses an SVD-based algorithm on the weighted data matrix:

        X = rho_samples - mean_field[:, None]
        X_w = sqrt(volumes)[:, None] * X
        X_w = U S V^T

    Then eigenvalues are S^2 / (n_samples - 1) and the physical modes are

        phi_k = U_k / sqrt(volumes)

    Finally the mode coefficients are computed explicitly using the weighted
    inner product for robustness:

        a_k^(s) = modes[:, k].T @ (volumes * X[:, s])

    """
    rho_samples = np.asarray(rho_samples, dtype=float)
    volumes = np.asarray(volumes, dtype=float)

    if rho_samples.ndim != 2:
        raise ValueError("rho_samples must be 2D (n_cells, n_samples).")
    if volumes.ndim != 1:
        raise ValueError("volumes must be 1D (n_cells,).")

    n_cells, n_samples = rho_samples.shape
    if volumes.shape[0] != n_cells:
        raise ValueError(
            f"volumes.shape[0] = {volumes.shape[0]} "
            f"does not match rho_samples.shape[0] = {n_cells}."
        )
    if n_modes > min(n_cells, n_samples):
        raise ValueError(
            "n_modes cannot exceed min(n_cells, n_samples); "
            f"got n_modes={n_modes}."
        )

    # Sample mean over realizations (per cell)
    mean_field = rho_samples.mean(axis=1)

    # Centered samples
    X = rho_samples - mean_field[:, None]  # (n_cells, n_samples)

    # Weighted data matrix: multiply each row by sqrt(volume)
    # Guard against zero or negative volumes: treat <=0 as zero.
    vol_clipped = np.clip(volumes, a_min=0.0, a_max=None)
    sqrt_w = np.sqrt(vol_clipped)
    # Avoid division by zero later by replacing 0 with 1 temporarily
    sqrt_w_safe = np.where(sqrt_w > 0.0, sqrt_w, 1.0)

    X_w = sqrt_w_safe[:, None] * X  # (n_cells, n_samples)

    # Economy SVD: X_w = U S Vt
    U, S, Vt = np.linalg.svd(X_w, full_matrices=False)

    # Eigenvalues of covariance operator
    eigvals_full = (S ** 2) / (n_samples - 1)

    # Truncate
    modes_w = U[:, :n_modes]            # weighted modes (orthonormal in Euclidean sense)
    eigvals = eigvals_full[:n_modes]

    # Transform back to physical modes, normalized w.r.t. weighted inner product
    # phi_k = modes_w[:, k] / sqrt_w  (elementwise)
    modes = modes_w / sqrt_w_safe[:, None]

    # Explicit KL coefficients using weighted inner product
    # a_k^(s) = phi_k^T (volumes * X[:, s])
    WX = vol_clipped[:, None] * X           # (n_cells, n_samples)
    mode_coeffs = modes.T @ WX              # (n_modes, n_samples)

    return mean_field, modes, eigvals, mode_coeffs


# ---------------------------------------------------------------------------
# Polynomial Chaos utilities
# ---------------------------------------------------------------------------


def total_degree_multiindex(M: int, p_max: int) -> List[Tuple[int, ...]]:
    """
    Generate all multi-indices alpha in N_0^M with total degree <= p_max.

    Parameters
    ----------
    M : int
        Number of random input variables xi_1, ..., xi_M.
    p_max : int
        Maximum total polynomial degree.

    Returns
    -------
    multiindex : list of tuple[int, ...]
        List of multi-indices alpha = (alpha_1, ..., alpha_M) with
        sum(alpha) <= p_max. The first element is always (0, ..., 0).
    """
    if M <= 0:
        raise ValueError("M must be positive.")
    if p_max < 0:
        raise ValueError("p_max must be non-negative.")

    multiindex: List[Tuple[int, ...]] = []
    for total_deg in range(p_max + 1):
        for alpha in itertools.product(range(total_deg + 1), repeat=M):
            if sum(alpha) == total_deg:
                multiindex.append(alpha)
    return multiindex


# --- 1D orthonormal polynomial evaluation ---------------------------------


def _hermite_orthonormal(x: np.ndarray | float, degree: int) -> np.ndarray | float:
    """
    Orthonormal Hermite polynomial (physicists' Hermite) for N(0, 1) inputs.

    H_n(x) are the physicists' Hermite polynomials (numpy.polynomial.hermite),
    orthogonal w.r.t. exp(-x^2). The orthonormal basis for standard normal
    N(0, 1) is

        psi_n(x) = H_n(x) / sqrt(2^n n! sqrt(pi))

    Parameters
    ----------
    x : float or ndarray
        Evaluation points.
    degree : int
        Polynomial degree n >= 0.

    Returns
    -------
    psi_n : float or ndarray
        Orthonormal Hermite polynomial psi_n(x).
    """
    if degree < 0:
        raise ValueError("degree must be non-negative.")

    # Coefficients for H_n: [0, 0, ..., 1] of length degree+1
    coeffs = np.zeros(degree + 1)
    coeffs[-1] = 1.0
    Hn = hermval(x, coeffs)

    norm_sq = (2.0 ** degree) * math.factorial(degree) * math.sqrt(math.pi)
    norm = math.sqrt(norm_sq)
    return Hn / norm


def _legendre_orthonormal(x: np.ndarray | float, degree: int) -> np.ndarray | float:
    """
    Orthonormal Legendre polynomial for uniform[-1, 1] inputs.

    P_n(x) are the standard Legendre polynomials (numpy.polynomial.legendre),
    orthogonal on [-1, 1] with weight 1. The orthonormal basis is

        psi_n(x) = P_n(x) * sqrt((2n + 1)/2)

    Parameters
    ----------
    x : float or ndarray
        Evaluation points.
    degree : int
        Polynomial degree n >= 0.

    Returns
    -------
    psi_n : float or ndarray
        Orthonormal Legendre polynomial psi_n(x).
    """
    if degree < 0:
        raise ValueError("degree must be non-negative.")

    coeffs = np.zeros(degree + 1)
    coeffs[-1] = 1.0
    Pn = legval(x, coeffs)

    norm = math.sqrt((2 * degree + 1) / 2.0)
    return Pn * norm


def eval_1d_poly(
    x: np.ndarray | float,
    degree: int,
    family: str = "hermite",
) -> np.ndarray | float:
    """
    Evaluate a 1D orthonormal polynomial of given degree and family.

    Parameters
    ----------
    x : float or ndarray
        Evaluation points.
    degree : int
        Polynomial degree >= 0.
    family : {"hermite", "legendre"}
        Polynomial family:
        - "hermite"  : standard normal inputs N(0, 1)
        - "legendre" : uniform inputs on [-1, 1]

    Returns
    -------
    value : float or ndarray
        psi_n(x) for the chosen family and degree.
    """
    if family == "hermite":
        return _hermite_orthonormal(x, degree)
    if family == "legendre":
        return _legendre_orthonormal(x, degree)
    raise ValueError(f"Unknown polynomial family {family!r}.")


def build_design_matrix(
    Xi: np.ndarray,
    multiindex: List[Tuple[int, ...]],
    family: str = "hermite",
) -> np.ndarray:
    """
    Build the PCE design matrix Psi for given input samples and multi-index set.

    Psi[s, k] = Psi_k(Xi[s, :]) where Psi_k is a multivariate orthonormal
    polynomial defined by multiindex[k].

    Parameters
    ----------
    Xi : ndarray, shape (n_samples, M)
        Input random samples. For "hermite", Xi should be i.i.d. N(0, 1).
        For "legendre", Xi should be within [-1, 1]^M.
    multiindex : list of tuple[int, ...]
        Multi-indices alpha defining the multivariate polynomials.
    family : {"hermite", "legendre"}, optional
        Polynomial family (same for all dimensions).

    Returns
    -------
    Psi : ndarray, shape (n_samples, n_basis)
        Design matrix, where n_basis = len(multiindex).

    Notes
    -----
    For clarity this implementation uses nested Python loops. For large
    n_samples and n_basis this can be optimized by caching 1D polynomial
    evaluations per dimension and degree.
    """
    Xi = np.asarray(Xi, dtype=float)
    if Xi.ndim != 2:
        raise ValueError("Xi must be 2D (n_samples, M).")

    n_samples, M = Xi.shape
    n_basis = len(multiindex)
    Psi = np.empty((n_samples, n_basis), dtype=float)

    for s in range(n_samples):
        xi_s = Xi[s, :]
        for k, alpha in enumerate(multiindex):
            # multivariate polynomial = product_m psi_{alpha_m}(xi_m)
            val = 1.0
            for m, deg in enumerate(alpha):
                if deg == 0:
                    # psi_0 = 1 for any family
                    continue
                val *= eval_1d_poly(xi_s[m], deg, family=family)
            Psi[s, k] = val

    return Psi


# ---------------------------------------------------------------------------
# PCE for KL mode coefficients
# ---------------------------------------------------------------------------


def fit_pce_for_kl_modes(
    mode_coeffs: np.ndarray,
    Xi: np.ndarray,
    multiindex: List[Tuple[int, ...]],
    family: str = "hermite",
) -> np.ndarray:
    """
    Fit a scalar PCE for each KL mode coefficient a_k(xi).

    Parameters
    ----------
    mode_coeffs : ndarray, shape (n_modes, n_samples)
        KL coefficients from the weighted KL decomposition.
        mode_coeffs[k, s] = a_k^(s).
    Xi : ndarray, shape (n_samples, M)
        Input random samples Xi[s, :] corresponding to the realizations in
        mode_coeffs.
    multiindex : list of tuple[int, ...]
        Multi-index set defining the multivariate PCE basis.
    family : {"hermite", "legendre"}, optional
        Polynomial family.

    Returns
    -------
    pce_coeffs : ndarray, shape (n_modes, n_basis)
        PCE coefficients for each mode. Row k contains the coefficients
        c_{k, j} for basis function j.

    Notes
    -----
    We solve the least squares problem

        a_k ≈ Psi c_k

    for each mode k, where a_k is shape (n_samples,) and Psi is the
    design matrix of shape (n_samples, n_basis).
    """
    mode_coeffs = np.asarray(mode_coeffs, dtype=float)
    Xi = np.asarray(Xi, dtype=float)

    if mode_coeffs.ndim != 2:
        raise ValueError("mode_coeffs must be 2D (n_modes, n_samples).")
    if Xi.ndim != 2:
        raise ValueError("Xi must be 2D (n_samples, M).")

    n_modes, n_samples = mode_coeffs.shape
    n_samples_Xi, _ = Xi.shape
    if n_samples_Xi != n_samples:
        raise ValueError(
            "Inconsistent number of samples: "
            f"mode_coeffs.shape[1]={n_samples}, Xi.shape[0]={n_samples_Xi}."
        )

    Psi = build_design_matrix(Xi, multiindex, family=family)  # (n_samples, n_basis)
    n_basis = Psi.shape[1]

    pce_coeffs = np.empty((n_modes, n_basis), dtype=float)
    for k in range(n_modes):
        y = mode_coeffs[k, :]
        c_k, *_ = np.linalg.lstsq(Psi, y, rcond=None)
        pce_coeffs[k, :] = c_k

    return pce_coeffs


def evaluate_kl_pce_surrogate(
    mean_field: np.ndarray,
    modes: np.ndarray,
    pce_coeffs: np.ndarray,
    Xi_new: np.ndarray,
    multiindex: List[Tuple[int, ...]],
    family: str = "hermite",
) -> np.ndarray:
    """
    Evaluate the full KL+PCE surrogate at new random inputs Xi_new.

    Parameters
    ----------
    mean_field : ndarray, shape (n_cells,)
        Mean log-rho field on the FEMTIC mesh.
    modes : ndarray, shape (n_cells, n_modes)
        Spatial KL modes phi_k on the FEMTIC mesh.
    pce_coeffs : ndarray, shape (n_modes, n_basis)
        PCE coefficients c_{k, j} for each KL mode a_k(xi).
    Xi_new : ndarray, shape (n_new, M)
        New input random samples where the surrogate should be evaluated.
    multiindex : list of tuple[int, ...]
        The same multi-index set as used in fitting.
    family : {"hermite", "legendre"}, optional
        Polynomial family.

    Returns
    -------
    rho_new : ndarray, shape (n_cells, n_new)
        Approximate log-rho fields for each new input sample.

    Notes
    -----
    For each new sample xi^*, we compute:

        a_k(xi^*) ≈ Σ_j c_{k, j} Psi_j(xi^*)

    and then reconstruct the field:

        rho(x, xi^*) ≈ mean_field(x) + Σ_k phi_k(x) a_k(xi^*).
    """
    mean_field = np.asarray(mean_field, dtype=float)
    modes = np.asarray(modes, dtype=float)
    pce_coeffs = np.asarray(pce_coeffs, dtype=float)
    Xi_new = np.asarray(Xi_new, dtype=float)

    if Xi_new.ndim != 2:
        raise ValueError("Xi_new must be 2D (n_new, M).")

    n_cells, n_modes = modes.shape
    if mean_field.shape[0] != n_cells:
        raise ValueError(
            "mean_field length does not match modes.shape[0]."
        )

    # Evaluate PCE for all modes at once:
    # Psi_new: (n_new, n_basis)
    Psi_new = build_design_matrix(Xi_new, multiindex, family=family)
    # a_new: (n_modes, n_new) = c @ Psi_new^T
    a_new = pce_coeffs @ Psi_new.T

    # Reconstruct fields: rho_new = mean_field[:, None] + modes @ a_new
    rho_new = mean_field[:, None] + modes @ a_new
    return rho_new


# ---------------------------------------------------------------------------
# Dataclass wrapper for convenience
# ---------------------------------------------------------------------------


@dataclass
class KLPCEModel:
    """
    Container for a fitted KL+PCE model on a FEMTIC mesh.

    Attributes
    ----------
    mean_field : ndarray, shape (n_cells,)
        Mean log-rho field on the FEMTIC mesh.
    modes : ndarray, shape (n_cells, n_modes)
        Spatial KL modes, orthonormal with respect to the weighted inner
        product defined by the cell volumes.
    eigvals : ndarray, shape (n_modes,)
        KL eigenvalues (mode variances).
    volumes : ndarray, shape (n_cells,)
        Cell volumes used in the weighted KL.
    multiindex : list of tuple[int, ...]
        Multi-index set defining the multivariate PCE basis.
    pce_coeffs : ndarray, shape (n_modes, n_basis)
        Scalar PCE coefficients for each mode.
    family : {"hermite", "legendre"}
        Polynomial family used in the PCE.
    """

    mean_field: np.ndarray
    modes: np.ndarray
    eigvals: np.ndarray
    volumes: np.ndarray
    multiindex: List[Tuple[int, ...]]
    pce_coeffs: np.ndarray
    family: str = "hermite"

    def evaluate(self, Xi_new: np.ndarray) -> np.ndarray:
        """
        Evaluate the KL+PCE surrogate at new random inputs Xi_new.

        Parameters
        ----------
        Xi_new : ndarray, shape (n_new, M)
            New input random samples.

        Returns
        -------
        rho_new : ndarray, shape (n_cells, n_new)
            Approximate log-rho fields on FEMTIC cells.
        """
        return evaluate_kl_pce_surrogate(
            self.mean_field,
            self.modes,
            self.pce_coeffs,
            Xi_new,
            self.multiindex,
            family=self.family,
        )


def fit_kl_pce_model(
    rho_samples: np.ndarray,
    volumes: np.ndarray,
    Xi: np.ndarray,
    n_modes: int,
    p_max: int,
    family: str = "hermite",
) -> KLPCEModel:
    """
    High-level convenience function: fit a full KL+PCE model in one call.

    Parameters
    ----------
    rho_samples : ndarray, shape (n_cells, n_samples)
        Log-rho fields on FEMTIC cells (each column is one realization).
    volumes : ndarray, shape (n_cells,)
        Cell volumes used as weights in the spatial inner product.
    Xi : ndarray, shape (n_samples, M)
        Input random samples used to generate rho_samples.
    n_modes : int
        Number of KL modes to retain.
    p_max : int
        Maximum total degree for the polynomial chaos basis.
    family : {"hermite", "legendre"}, optional
        Polynomial family for PCE ("hermite" for Gaussian, "legendre" for
        uniform on [-1, 1]).

    Returns
    -------
    model : KLPCEModel
        Fitted model that can be evaluated at new Xi via model.evaluate(Xi_new).

    Example
    -------
    >>> mean_field, modes, eigvals, mode_coeffs = compute_weighted_kl(
    ...     rho_samples, volumes, n_modes=10)
    >>> multiindex = total_degree_multiindex(M=Xi.shape[1], p_max=2)
    >>> pce_coeffs = fit_pce_for_kl_modes(mode_coeffs, Xi, multiindex)
    >>> model = KLPCEModel(mean_field, modes, eigvals, volumes,
    ...                    multiindex, pce_coeffs, family="hermite")
    """
    mean_field, modes, eigvals, mode_coeffs = compute_weighted_kl(
        rho_samples, volumes, n_modes=n_modes
    )
    M = Xi.shape[1]
    multiindex = total_degree_multiindex(M, p_max)
    pce_coeffs = fit_pce_for_kl_modes(mode_coeffs, Xi, multiindex, family=family)

    return KLPCEModel(
        mean_field=mean_field,
        modes=modes,
        eigvals=eigvals,
        volumes=np.asarray(volumes, dtype=float),
        multiindex=multiindex,
        pce_coeffs=pce_coeffs,
        family=family,
    )
