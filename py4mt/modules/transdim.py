#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transdim.py — Transdimensional (rjMCMC) module for 1-D layered-earth inversion.
================================================================================

Reusable library of classes and functions for reversible-jump MCMC where the
number of layers *k* is itself a free parameter.

This module is part of **py4mtx** and may depend on other py4mtx modules
(``data_proc``, ``aniso``, etc.) which must be on ``sys.path``.

Contents
--------
Data structures:
    LayeredModel   — 1-D earth model (isotropic or anisotropic)
    Prior          — uniform prior bounds
    RjMCMCConfig   — sampler tuning knobs

Forward models:
    mt_forward_1d_isotropic     — recursive impedance (scalar ρ_a)
    mt_forward_1d_anisotropic   — full Z tensor via aniso.py → ρ_a^{xy,yx}

Impedance utilities:
    compute_Zdet                — determinant impedance from Z tensor
    compute_Zdet_err            — Monte-Carlo Z_det error propagation
    compute_rhophas_from_Zdet   — ρ_a / phase from Z_det
    pack_Zdet_vector / pack_Zdet_sigma — pack Z_det into real data vectors
    pack_Z_vector / pack_Z_sigma       — pack selected Z components
    pack_P_vector / pack_P_sigma       — pack phase-tensor components

Likelihoods:
    log_likelihood              — Gaussian on log10(ρ_a)
    log_likelihood_Zdet         — Gaussian on Re/Im of Z_det
    log_likelihood_Z_comps      — Gaussian on Re/Im of Z subset + optional PT

Proposals:
    propose_birth, propose_death, propose_move, propose_change

Samplers:
    run_rjmcmc            — single chain
    run_parallel_rjmcmc   — N independent chains via joblib

Diagnostics:
    gelman_rubin          — R-hat convergence statistic

Post-processing:
    compute_posterior_profile       — ρ(z) statistics on a depth grid
    compute_posterior_aniso_profile — aniso-ratio/strike statistics

Driver helpers:
    model0_to_layered       — convert py4mt model dict → LayeredModel
    load_site               — load MT site from EDI/NPZ via data_proc
    build_rjmcmc_summary    — posterior quantile summary on a depth grid
    ensure_dir              — create directory if needed

I/O:
    save_results_npz, load_results_npz

See also
--------
transdim_viz.py — plotting routines (separated to avoid mandatory
                  matplotlib dependency in headless / HPC environments).

References
----------
  - Green (1995), Biometrika — Reversible jump MCMC
  - Bodin & Sambridge (2009), GJI — Transdimensional tomography
  - Malinverno (2002), Geophysics — Parsimonious Bayesian inversion

@author:    Volker Rath (DIAS) / Claude (Opus 4.6, Anthropic)
@project:   py4mt — Python for Magnetotellurics
@created:   2026-03-07 — split from transdimensional_mcmc_parallel.py
@modified:  2026-03-07 — viz split into transdim_viz.py
@modified:  2026-03-09 — helpers moved from mt_transdim1d.py; Z_det and
                          Z-component likelihood support added

Author: Volker Rath (DIAS)
Created with the help of ChatGPT (GPT-5 Thinking) on 2026-03-11 UTC
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import data_proc

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
#  Optional: anisotropic forward model (aniso.py)
# ---------------------------------------------------------------------------
try:
    from aniso import aniso1d_impedance_sens_simple
    _HAS_ANISO = True
except ImportError:
    _HAS_ANISO = False


# =============================================================================
#  Filesystem helpers
# =============================================================================

def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return its Path.

    Parameters
    ----------
    path : str or Path
        Directory path to create if it does not already exist.

    Returns
    -------
    Path
        Normalized ``Path`` object corresponding to ``path``.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# =============================================================================
#  1-D Forward Models
# =============================================================================

def mt_forward_1d_isotropic(
    thicknesses: np.ndarray,
    resistivities: np.ndarray,
    frequencies: np.ndarray,
    full_output: bool = False,
) -> np.ndarray | Dict[str, np.ndarray]:
    """Recursive-impedance 1-D isotropic MT forward model.

    Parameters
    ----------
    thicknesses : ndarray
        Array of shape ``(n_layers - 1,)`` containing layer thicknesses [m].
        The half-space is implicit and therefore has no thickness entry.
    resistivities : ndarray
        Array of shape ``(n_layers,)`` containing isotropic resistivities [Ω·m].
    frequencies : ndarray
        Array of frequencies [Hz].
    full_output : bool, optional
        If ``False`` return only apparent resistivity. If ``True`` return a
        dictionary with apparent resistivity, phase, and impedance parts.

    Returns
    -------
    ndarray or dict
        If ``full_output`` is ``False``, returns apparent resistivity ``rho_a``.
        Otherwise returns a dictionary with keys ``rho_a``, ``phase_deg``,
        ``Z_re``, and ``Z_im``.
    """
    mu0 = 4.0 * np.pi * 1e-7
    omega = 2.0 * np.pi * frequencies
    n_layers = len(resistivities)
    nf = len(frequencies)
    rho_a = np.zeros(nf)

    if full_output:
        phase_deg = np.zeros(nf)
        Z_re = np.zeros(nf)
        Z_im = np.zeros(nf)

    for fi, w in enumerate(omega):
        k = np.sqrt(1j * w * mu0 / resistivities)
        Z = w * mu0 / k[-1]
        for j in range(n_layers - 2, -1, -1):
            Z_j = w * mu0 / k[j]
            r = (Z_j - Z) / (Z_j + Z)
            e2 = np.exp(-2 * k[j] * thicknesses[j])
            Z = Z_j * (1 - r * e2) / (1 + r * e2)
        rho_a[fi] = np.abs(Z) ** 2 / (w * mu0)
        if full_output:
            Z_re[fi] = Z.real
            Z_im[fi] = Z.imag
            phase_deg[fi] = np.abs(np.degrees(np.arctan2(Z.imag, Z.real)))

    if full_output:
        return {"rho_a": rho_a, "phase_deg": phase_deg,
                "Z_re": Z_re, "Z_im": Z_im}
    return rho_a


def mt_forward_1d_anisotropic(
    thicknesses: np.ndarray,
    resistivities: np.ndarray,
    frequencies: np.ndarray,
    aniso_ratios: np.ndarray,
    strikes: np.ndarray,
) -> Dict[str, np.ndarray]:
    """1-D anisotropic MT forward model via ``aniso.py``.

    Parameters
    ----------
    thicknesses : ndarray
        Layer thicknesses [m] with shape ``(n_layers - 1,)``.
    resistivities : ndarray
        Maximum horizontal resistivities [Ω·m] with shape ``(n_layers,)``.
    frequencies : ndarray
        Frequency array [Hz].
    aniso_ratios : ndarray
        Resistivity anisotropy ratios ``rho_max / rho_min`` with shape
        ``(n_layers,)``.
    strikes : ndarray
        Layer strike angles [deg] with shape ``(n_layers,)``.

    Returns
    -------
    dict
        Dictionary with keys ``rho_a_xy`` and ``rho_a_yx``.

    Raises
    ------
    ImportError
        Raised if ``aniso.py`` is not available.
    """
    if not _HAS_ANISO:
        raise ImportError(
            "aniso.py not found.  Place it on PYTHONPATH or in the working directory."
        )

    periods = 1.0 / frequencies
    rho_max = resistivities.copy()
    rho_min = resistivities / aniso_ratios
    h_m = np.append(thicknesses, 1e6)

    result = aniso1d_impedance_sens_simple(
        periods_s=periods,
        h_m=h_m,
        rho_max=rho_max,
        rho_min=rho_min,
        strike_deg=strikes,
        compute_sens=False,
    )

    Z = result["Z"]
    mu0 = 4.0 * np.pi * 1e-7
    omega = 2.0 * np.pi * frequencies

    return {
        "rho_a_xy": np.abs(Z[:, 0, 1]) ** 2 / (omega * mu0),
        "rho_a_yx": np.abs(Z[:, 1, 0]) ** 2 / (omega * mu0),
    }


def _complex_normal_sample(
    z: np.ndarray,
    sigma: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw complex Gaussian perturbations for Monte-Carlo propagation.

    Parameters
    ----------
    z : ndarray
        Complex input array.
    sigma : ndarray
        Real-valued 1σ absolute standard deviation for complex entries.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    ndarray
        Complex perturbed samples with the same shape as ``z``.
    """
    sig = np.asarray(sigma, dtype=float)
    re = rng.normal(loc=np.real(z), scale=sig)
    im = rng.normal(loc=np.imag(z), scale=sig)
    return re + 1j * im


def compute_Zdet(Z: np.ndarray) -> np.ndarray:
    """Compute determinant impedance from a 2×2 impedance tensor.

    Parameters
    ----------
    Z : ndarray
        Complex impedance tensor array with shape ``(nf, 2, 2)``.

    Returns
    -------
    ndarray
        Complex determinant impedance ``sqrt(det(Z))`` for each frequency.
    """
    Z = np.asarray(Z)
    det = Z[:, 0, 0] * Z[:, 1, 1] - Z[:, 0, 1] * Z[:, 1, 0]
    return np.sqrt(det)


def compute_Zdet_err(
    Z: np.ndarray,
    Z_err: np.ndarray,
    nsim: int = 200,
    seed: int = 12345,
) -> np.ndarray:
    """Estimate Zdet uncertainty by Monte-Carlo propagation.

    Parameters
    ----------
    Z : ndarray
        Complex impedance tensor array with shape ``(nf, 2, 2)``.
    Z_err : ndarray
        Absolute 1σ uncertainty array broadcastable to ``Z``.
    nsim : int, optional
        Number of Monte-Carlo simulations.
    seed : int, optional
        Random seed.

    Returns
    -------
    ndarray
        Estimated absolute 1σ uncertainty of determinant impedance.
    """
    Z = np.asarray(Z)
    Z_err = np.asarray(Z_err, dtype=float)
    rng = np.random.default_rng(seed)
    sims = np.empty((nsim, Z.shape[0]), dtype=complex)
    for i in range(nsim):
        Zp = _complex_normal_sample(Z, Z_err, rng)
        sims[i, :] = compute_Zdet(Zp)
    sr = np.std(sims.real, axis=0, ddof=1)
    si = np.std(sims.imag, axis=0, ddof=1)
    return np.sqrt(sr**2 + si**2)


def compute_rhophas_from_Zdet(
    Zdet: np.ndarray,
    frequencies: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert determinant impedance to apparent resistivity and phase.

    Parameters
    ----------
    Zdet : ndarray
        Complex determinant impedance.
    frequencies : ndarray
        Frequencies [Hz].

    Returns
    -------
    tuple of ndarray
        Apparent resistivity [Ω·m] and phase [deg].
    """
    mu0 = 4.0 * np.pi * 1e-7
    omega = 2.0 * np.pi * np.asarray(frequencies, dtype=float)
    rho_a = np.abs(Zdet) ** 2 / (omega * mu0)
    phase = np.degrees(np.arctan2(np.imag(Zdet), np.real(Zdet)))
    return rho_a, phase


def _comp_to_idx(comp: str) -> Tuple[int, int]:
    """Map component labels like ``'xy'`` to tensor indices.

    Parameters
    ----------
    comp : str
        Component label among ``xx``, ``xy``, ``yx``, ``yy``.

    Returns
    -------
    tuple[int, int]
        Tensor indices corresponding to the requested component.

    Raises
    ------
    ValueError
        Raised for unsupported component labels.
    """
    cmap = {"xx": (0, 0), "xy": (0, 1), "yx": (1, 0), "yy": (1, 1)}
    c = comp.lower().strip()
    if c not in cmap:
        raise ValueError(f"Unknown tensor component: {comp}")
    return cmap[c]


def pack_Zdet_vector(Zdet: np.ndarray) -> np.ndarray:
    """Pack complex determinant impedance into a real-valued data vector.

    Parameters
    ----------
    Zdet : ndarray
        Complex determinant impedance of shape ``(nf,)``.

    Returns
    -------
    ndarray
        Real-valued vector of length ``2*nf`` with interleaved real and
        imaginary parts.
    """
    Zdet = np.asarray(Zdet)
    out = np.empty(2 * len(Zdet), dtype=float)
    out[0::2] = Zdet.real
    out[1::2] = Zdet.imag
    return out


def pack_Zdet_sigma(Zdet_sigma: np.ndarray) -> np.ndarray:
    """Pack determinant-impedance uncertainties to match ``pack_Zdet_vector``.

    Parameters
    ----------
    Zdet_sigma : ndarray
        Absolute 1σ uncertainties of determinant impedance.

    Returns
    -------
    ndarray
        Real-valued uncertainty vector with duplicated entries for real and
        imaginary parts.
    """
    sig = np.asarray(Zdet_sigma, dtype=float)
    out = np.empty(2 * len(sig), dtype=float)
    out[0::2] = sig
    out[1::2] = sig
    return out


def pack_Z_vector(Z: np.ndarray, comps: Sequence[str]) -> np.ndarray:
    """Pack selected impedance-tensor components into a real-valued vector.

    Parameters
    ----------
    Z : ndarray
        Complex impedance tensor array of shape ``(nf, 2, 2)``.
    comps : sequence of str
        Selected components, e.g. ``('xy', 'yx')``.

    Returns
    -------
    ndarray
        Real-valued vector containing interleaved real/imaginary values for
        each requested component.
    """
    Z = np.asarray(Z)
    nf = Z.shape[0]
    parts = []
    for comp in comps:
        i, j = _comp_to_idx(comp)
        zc = Z[:, i, j]
        vec = np.empty(2 * nf, dtype=float)
        vec[0::2] = zc.real
        vec[1::2] = zc.imag
        parts.append(vec)
    return np.concatenate(parts) if parts else np.empty(0, dtype=float)


def pack_Z_sigma(
    Z_err: np.ndarray,
    comps: Sequence[str],
) -> np.ndarray:
    """Pack component-wise impedance uncertainties to match ``pack_Z_vector``.

    Parameters
    ----------
    Z_err : ndarray
        Impedance uncertainty array broadcastable to ``(nf, 2, 2)``.
    comps : sequence of str
        Selected impedance components.

    Returns
    -------
    ndarray
        Real-valued uncertainty vector.
    """
    Z_err = np.asarray(Z_err, dtype=float)
    nf = Z_err.shape[0]
    parts = []
    for comp in comps:
        i, j = _comp_to_idx(comp)
        sig = Z_err[:, i, j]
        vec = np.empty(2 * nf, dtype=float)
        vec[0::2] = sig
        vec[1::2] = sig
        parts.append(vec)
    return np.concatenate(parts) if parts else np.empty(0, dtype=float)


def pack_P_vector(P: np.ndarray, comps: Sequence[str]) -> np.ndarray:
    """Pack selected phase-tensor components into a real-valued vector.

    Parameters
    ----------
    P : ndarray
        Phase tensor array of shape ``(nf, 2, 2)``.
    comps : sequence of str
        Selected components.

    Returns
    -------
    ndarray
        Real-valued vector containing the selected phase-tensor entries.
    """
    P = np.asarray(P, dtype=float)
    parts = []
    for comp in comps:
        i, j = _comp_to_idx(comp)
        parts.append(P[:, i, j].astype(float))
    return np.concatenate(parts) if parts else np.empty(0, dtype=float)


def pack_P_sigma(P_err: np.ndarray, comps: Sequence[str]) -> np.ndarray:
    """Pack selected phase-tensor uncertainties into a real-valued vector.

    Parameters
    ----------
    P_err : ndarray
        Phase-tensor uncertainty array of shape ``(nf, 2, 2)``.
    comps : sequence of str
        Selected components.

    Returns
    -------
    ndarray
        Real-valued vector of selected phase-tensor uncertainties.
    """
    P_err = np.asarray(P_err, dtype=float)
    parts = []
    for comp in comps:
        i, j = _comp_to_idx(comp)
        parts.append(P_err[:, i, j].astype(float))
    return np.concatenate(parts) if parts else np.empty(0, dtype=float)


# =============================================================================
#  Model and configuration dataclasses
# =============================================================================

@dataclass
class LayeredModel:
    """Layered-earth parameterization for isotropic or anisotropic 1-D MT.

    Attributes
    ----------
    depths : ndarray
        Interface depths [m] excluding the surface; length ``k``.
    log_resistivities : ndarray
        Base-10 logarithm of maximum horizontal resistivities [Ω·m] for each
        layer, length ``k + 1``.
    aniso_ratios : ndarray or None
        Resistivity anisotropy ratios ``rho_max / rho_min`` per layer.
        ``None`` indicates isotropic treatment.
    strikes : ndarray or None
        Strike angles [deg] per layer for anisotropic models.
    """
    depths: np.ndarray
    log_resistivities: np.ndarray
    aniso_ratios: Optional[np.ndarray] = None
    strikes: Optional[np.ndarray] = None

    def copy(self) -> "LayeredModel":
        """Return a deep copy of the model.

        Returns
        -------
        LayeredModel
            Independent copy of the current model.
        """
        return LayeredModel(
            depths=self.depths.copy(),
            log_resistivities=self.log_resistivities.copy(),
            aniso_ratios=None if self.aniso_ratios is None else self.aniso_ratios.copy(),
            strikes=None if self.strikes is None else self.strikes.copy(),
        )

    @property
    def k(self) -> int:
        """Return the number of interfaces.

        Returns
        -------
        int
            Number of interfaces.
        """
        return int(len(self.depths))

    @property
    def n_layers(self) -> int:
        """Return the number of layers.

        Returns
        -------
        int
            Number of layers, equal to ``k + 1``.
        """
        return int(len(self.log_resistivities))

    @property
    def is_anisotropic(self) -> bool:
        """Return whether anisotropy parameters are present.

        Returns
        -------
        bool
            ``True`` when both anisotropy ratios and strikes are available.
        """
        return self.aniso_ratios is not None and self.strikes is not None

    def get_thicknesses(self) -> np.ndarray:
        """Convert interface depths to layer thicknesses.

        Returns
        -------
        ndarray
            Thickness array of length ``k``.
        """
        if self.k == 0:
            return np.empty(0, dtype=float)
        return np.diff(np.r_[0.0, self.depths])

    def get_resistivities(self) -> np.ndarray:
        """Return resistivities in linear units.

        Returns
        -------
        ndarray
            Resistivity array [Ω·m].
        """
        return 10.0 ** self.log_resistivities


@dataclass
class Prior:
    """Uniform prior bounds for the reversible-jump inversion.

    Attributes
    ----------
    k_min, k_max : int
        Minimum and maximum number of interfaces.
    depth_min, depth_max : float
        Bounds on interface depths [m].
    log_rho_min, log_rho_max : float
        Bounds on base-10 resistivity.
    log_aniso_min, log_aniso_max : float
        Bounds on base-10 anisotropy ratio.
    strike_min, strike_max : float
        Bounds on strike angle [deg].
    """
    k_min: int
    k_max: int
    depth_min: float
    depth_max: float
    log_rho_min: float
    log_rho_max: float
    log_aniso_min: float = 0.0
    log_aniso_max: float = 0.0
    strike_min: float = -90.0
    strike_max: float = 90.0


@dataclass
class RjMCMCConfig:
    """Sampler control parameters for RJMCMC.

    Attributes
    ----------
    n_iterations : int
        Total number of MCMC iterations.
    burn_in : int
        Number of discarded burn-in iterations.
    thin : int
        Retain every ``thin``-th post-burn-in sample.
    proposal_weights : tuple
        Relative weights for birth, death, move, and change proposals.
    sigma_birth_rho, sigma_move_z, sigma_change_rho : float
        Proposal scales for isotropic model updates.
    sigma_birth_aniso, sigma_birth_strike : float
        Proposal scales for anisotropy parameters during births.
    sigma_change_aniso, sigma_change_strike : float
        Proposal scales for anisotropy parameter perturbations.
    verbose : bool
        Whether to print progress.
    """
    n_iterations: int = 200_000
    burn_in: int = 50_000
    thin: int = 10
    proposal_weights: Tuple[float, float, float, float] = (0.2, 0.2, 0.25, 0.35)
    sigma_birth_rho: float = 0.03
    sigma_move_z: float = 50.0
    sigma_change_rho: float = 0.05
    sigma_birth_aniso: float = 0.10
    sigma_birth_strike: float = 10.0
    sigma_change_aniso: float = 0.05
    sigma_change_strike: float = 5.0
    verbose: bool = True


# =============================================================================
#  Prior helpers
# =============================================================================

def _uniform_logpdf(x: float, xmin: float, xmax: float) -> float:
    """Compute log-density of a scalar uniform distribution.

    Parameters
    ----------
    x : float
        Scalar value to test.
    xmin, xmax : float
        Inclusive interval bounds.

    Returns
    -------
    float
        Zero if inside the interval, ``-np.inf`` otherwise.
    """
    return 0.0 if xmin <= x <= xmax else -np.inf


def log_prior(model: LayeredModel, prior: Prior, use_aniso: bool = False) -> float:
    """Evaluate the uniform prior density for a layered model.

    Parameters
    ----------
    model : LayeredModel
        Candidate model.
    prior : Prior
        Prior bounds.
    use_aniso : bool, optional
        If ``True`` include anisotropy and strike prior terms.

    Returns
    -------
    float
        Log-prior value, or ``-np.inf`` if the model violates the bounds.
    """
    if model.k < prior.k_min or model.k > prior.k_max:
        return -np.inf

    if np.any(np.diff(model.depths) <= 0):
        return -np.inf
    if model.k > 0:
        if model.depths[0] < prior.depth_min or model.depths[-1] > prior.depth_max:
            return -np.inf

    if np.any(model.log_resistivities < prior.log_rho_min):
        return -np.inf
    if np.any(model.log_resistivities > prior.log_rho_max):
        return -np.inf

    if use_aniso:
        if model.aniso_ratios is None or model.strikes is None:
            return -np.inf
        loga = np.log10(model.aniso_ratios)
        if np.any(loga < prior.log_aniso_min) or np.any(loga > prior.log_aniso_max):
            return -np.inf
        if np.any(model.strikes < prior.strike_min) or np.any(model.strikes > prior.strike_max):
            return -np.inf

    return 0.0


# =============================================================================
#  Forward wrapper helpers for likelihoods
# =============================================================================

def _forward_Z_for_model(model: LayeredModel, frequencies: np.ndarray) -> np.ndarray:
    """Build a 2×2 impedance tensor for isotropic or anisotropic models.

    Parameters
    ----------
    model : LayeredModel
        Model to forward model.
    frequencies : ndarray
        Frequencies [Hz].

    Returns
    -------
    ndarray
        Complex impedance tensor of shape ``(nf, 2, 2)``.
    """
    nf = len(frequencies)
    Z = np.zeros((nf, 2, 2), dtype=complex)

    if model.is_anisotropic:
        th = model.get_thicknesses()
        rho = model.get_resistivities()
        out = mt_forward_1d_anisotropic(th, rho, frequencies, model.aniso_ratios, model.strikes)
        mu0 = 4.0 * np.pi * 1e-7
        omega = 2.0 * np.pi * frequencies
        zxy = np.sqrt(out["rho_a_xy"] * omega * mu0) * (1.0 + 1.0j) / np.sqrt(2.0)
        zyx = -np.sqrt(out["rho_a_yx"] * omega * mu0) * (1.0 + 1.0j) / np.sqrt(2.0)
        Z[:, 0, 1] = zxy
        Z[:, 1, 0] = zyx
    else:
        th = model.get_thicknesses()
        rho = model.get_resistivities()
        out = mt_forward_1d_isotropic(th, rho, frequencies, full_output=True)
        z = out["Z_re"] + 1j * out["Z_im"]
        Z[:, 0, 1] = z
        Z[:, 1, 0] = -z

    return Z


def _phase_tensor_from_Z(Z: np.ndarray) -> np.ndarray:
    """Compute phase tensor from a complex impedance tensor.

    Parameters
    ----------
    Z : ndarray
        Complex impedance tensor with shape ``(nf, 2, 2)``.

    Returns
    -------
    ndarray
        Phase tensor array with shape ``(nf, 2, 2)``.
    """
    X = np.real(Z)
    Y = np.imag(Z)
    P = np.zeros_like(X, dtype=float)
    for i in range(Z.shape[0]):
        try:
            P[i] = np.linalg.solve(X[i], Y[i])
        except np.linalg.LinAlgError:
            P[i] = np.nan
    return P


# =============================================================================
#  Likelihoods
# =============================================================================

def log_likelihood(
    model: LayeredModel,
    frequencies: np.ndarray,
    observed: np.ndarray,
    sigma: np.ndarray,
    observed_yx: Optional[np.ndarray] = None,
    sigma_yx: Optional[np.ndarray] = None,
) -> float:
    """Gaussian log-likelihood on apparent resistivity.

    Parameters
    ----------
    model : LayeredModel
        Proposed earth model.
    frequencies : ndarray
        Frequencies [Hz].
    observed : ndarray
        Observed apparent resistivity for the xy component or isotropic case.
    sigma : ndarray
        Standard deviation in log10 apparent resistivity for ``observed``.
    observed_yx : ndarray, optional
        Optional yx apparent resistivity for anisotropic runs.
    sigma_yx : ndarray, optional
        Standard deviation for ``observed_yx``.

    Returns
    -------
    float
        Gaussian log-likelihood value.
    """
    if model.is_anisotropic:
        out = mt_forward_1d_anisotropic(
            model.get_thicknesses(),
            model.get_resistivities(),
            frequencies,
            model.aniso_ratios,
            model.strikes,
        )
        pred_xy = np.asarray(out["rho_a_xy"], dtype=float)
        pred_yx = np.asarray(out["rho_a_yx"], dtype=float)
        r = (np.log10(pred_xy) - np.log10(observed)) / sigma
        ll = -0.5 * np.sum(r**2 + np.log(2.0 * np.pi * sigma**2))
        if observed_yx is not None and sigma_yx is not None:
            r2 = (np.log10(pred_yx) - np.log10(observed_yx)) / sigma_yx
            ll += -0.5 * np.sum(r2**2 + np.log(2.0 * np.pi * sigma_yx**2))
        return float(ll)

    pred = mt_forward_1d_isotropic(model.get_thicknesses(), model.get_resistivities(), frequencies)
    r = (np.log10(pred) - np.log10(observed)) / sigma
    ll = -0.5 * np.sum(r**2 + np.log(2.0 * np.pi * sigma**2))
    return float(ll)


def log_likelihood_Zdet(
    model: LayeredModel,
    frequencies: np.ndarray,
    observed_Zdet: np.ndarray,
    Zdet_sigma: np.ndarray,
) -> float:
    """Gaussian log-likelihood on determinant impedance in Re/Im space.

    Parameters
    ----------
    model : LayeredModel
        Proposed earth model.
    frequencies : ndarray
        Frequencies [Hz].
    observed_Zdet : ndarray
        Observed determinant impedance.
    Zdet_sigma : ndarray
        Absolute 1σ uncertainty of determinant impedance.

    Returns
    -------
    float
        Gaussian log-likelihood value.
    """
    pred_Z = _forward_Z_for_model(model, frequencies)
    pred_Zdet = compute_Zdet(pred_Z)
    d_pred = pack_Zdet_vector(pred_Zdet)
    d_obs = pack_Zdet_vector(observed_Zdet)
    sig = pack_Zdet_sigma(Zdet_sigma)
    r = (d_pred - d_obs) / sig
    ll = -0.5 * np.sum(r**2 + np.log(2.0 * np.pi * sig**2))
    return float(ll)


def log_likelihood_Z_comps(
    model: LayeredModel,
    frequencies: np.ndarray,
    observed_Z: np.ndarray,
    observed_Z_err: np.ndarray,
    z_comps: Sequence[str] = ("xy", "yx"),
    use_pt: bool = False,
    observed_PT: Optional[np.ndarray] = None,
    observed_PT_err: Optional[np.ndarray] = None,
    pt_comps: Sequence[str] = ("xx", "xy", "yx", "yy"),
) -> float:
    """Gaussian log-likelihood on selected Z components and optional PT.

    Parameters
    ----------
    model : LayeredModel
        Proposed earth model.
    frequencies : ndarray
        Frequencies [Hz].
    observed_Z : ndarray
        Observed complex impedance tensor.
    observed_Z_err : ndarray
        Absolute impedance uncertainties.
    z_comps : sequence of str, optional
        Selected impedance components.
    use_pt : bool, optional
        Whether to include phase tensor in the likelihood.
    observed_PT : ndarray, optional
        Observed phase tensor.
    observed_PT_err : ndarray, optional
        Phase-tensor uncertainties.
    pt_comps : sequence of str, optional
        Selected phase-tensor components.

    Returns
    -------
    float
        Gaussian log-likelihood value.
    """
    pred_Z = _forward_Z_for_model(model, frequencies)
    d_pred = pack_Z_vector(pred_Z, z_comps)
    d_obs = pack_Z_vector(observed_Z, z_comps)
    sig = pack_Z_sigma(observed_Z_err, z_comps)
    r = (d_pred - d_obs) / sig
    ll = -0.5 * np.sum(r**2 + np.log(2.0 * np.pi * sig**2))

    if use_pt:
        pred_PT = _phase_tensor_from_Z(pred_Z)
        d_pred_pt = pack_P_vector(pred_PT, pt_comps)
        d_obs_pt = pack_P_vector(observed_PT, pt_comps)
        sig_pt = pack_P_sigma(observed_PT_err, pt_comps)
        r_pt = (d_pred_pt - d_obs_pt) / sig_pt
        ll += -0.5 * np.sum(r_pt**2 + np.log(2.0 * np.pi * sig_pt**2))

    return float(ll)


# =============================================================================
#  Proposal mechanisms
# =============================================================================

def _insert_sorted_depth(depths: np.ndarray, z_new: float) -> Tuple[np.ndarray, int]:
    """Insert a new interface depth while preserving sorting.

    Parameters
    ----------
    depths : ndarray
        Existing sorted interface depths.
    z_new : float
        New depth to insert.

    Returns
    -------
    tuple
        Updated sorted depths and insertion index.
    """
    idx = int(np.searchsorted(depths, z_new))
    return np.insert(depths, idx, z_new), idx


def propose_birth(
    current: LayeredModel,
    prior: Prior,
    config: RjMCMCConfig,
    rng: np.random.Generator,
    use_aniso: bool = False,
) -> Tuple[LayeredModel, float]:
    """Propose a birth move by inserting one interface and one layer.

    Parameters
    ----------
    current : LayeredModel
        Current model.
    prior : Prior
        Prior bounds.
    config : RjMCMCConfig
        Sampler configuration.
    rng : numpy.random.Generator
        Random generator.
    use_aniso : bool, optional
        Whether to include anisotropy parameters.

    Returns
    -------
    tuple
        Proposed model and log proposal ratio. The ratio is set to zero here
        because symmetric proposal simplifications are used in this driver.
    """
    if current.k >= prior.k_max:
        return current.copy(), -np.inf

    z_new = rng.uniform(prior.depth_min, prior.depth_max)
    new_depths, idx = _insert_sorted_depth(current.depths, z_new)

    parent_idx = min(idx, current.n_layers - 1)
    new_log_rho = current.log_resistivities[parent_idx] + rng.normal(0.0, config.sigma_birth_rho)
    new_rhos = np.insert(current.log_resistivities, idx + 1, new_log_rho)

    if use_aniso:
        ratios = current.aniso_ratios.copy()
        strikes = current.strikes.copy()
        new_ratio = ratios[parent_idx] * 10.0 ** rng.normal(0.0, config.sigma_birth_aniso)
        new_strike = strikes[parent_idx] + rng.normal(0.0, config.sigma_birth_strike)
        new_ratios = np.insert(ratios, idx + 1, new_ratio)
        new_strikes = np.insert(strikes, idx + 1, new_strike)
    else:
        new_ratios = None
        new_strikes = None

    prop = LayeredModel(new_depths, new_rhos, new_ratios, new_strikes)
    return prop, 0.0


def propose_death(
    current: LayeredModel,
    prior: Prior,
    config: RjMCMCConfig,
    rng: np.random.Generator,
    use_aniso: bool = False,
) -> Tuple[LayeredModel, float]:
    """Propose a death move by deleting one interface and merging layers.

    Parameters
    ----------
    current : LayeredModel
        Current model.
    prior : Prior
        Prior bounds.
    config : RjMCMCConfig
        Sampler configuration.
    rng : numpy.random.Generator
        Random generator.
    use_aniso : bool, optional
        Whether anisotropy parameters are active.

    Returns
    -------
    tuple
        Proposed model and log proposal ratio.
    """
    if current.k <= prior.k_min:
        return current.copy(), -np.inf

    idx = int(rng.integers(0, current.k))
    new_depths = np.delete(current.depths, idx)
    avg_log_rho = 0.5 * (current.log_resistivities[idx] + current.log_resistivities[idx + 1])
    new_rhos = current.log_resistivities.copy()
    new_rhos[idx] = avg_log_rho
    new_rhos = np.delete(new_rhos, idx + 1)

    if use_aniso:
        avg_ratio = 0.5 * (current.aniso_ratios[idx] + current.aniso_ratios[idx + 1])
        avg_strike = 0.5 * (current.strikes[idx] + current.strikes[idx + 1])
        new_ratios = current.aniso_ratios.copy()
        new_strikes = current.strikes.copy()
        new_ratios[idx] = avg_ratio
        new_strikes[idx] = avg_strike
        new_ratios = np.delete(new_ratios, idx + 1)
        new_strikes = np.delete(new_strikes, idx + 1)
    else:
        new_ratios = None
        new_strikes = None

    prop = LayeredModel(new_depths, new_rhos, new_ratios, new_strikes)
    return prop, 0.0


def propose_move(
    current: LayeredModel,
    prior: Prior,
    config: RjMCMCConfig,
    rng: np.random.Generator,
) -> Tuple[LayeredModel, float]:
    """Propose to perturb one interface depth.

    Parameters
    ----------
    current : LayeredModel
        Current model.
    prior : Prior
        Prior bounds.
    config : RjMCMCConfig
        Sampler configuration.
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    tuple
        Proposed model and log proposal ratio.
    """
    if current.k == 0:
        return current.copy(), -np.inf

    prop = current.copy()
    idx = int(rng.integers(0, prop.k))
    prop.depths[idx] += rng.normal(0.0, config.sigma_move_z)
    prop.depths = np.sort(prop.depths)
    return prop, 0.0


def propose_change(
    current: LayeredModel,
    prior: Prior,
    config: RjMCMCConfig,
    rng: np.random.Generator,
    use_aniso: bool = False,
) -> Tuple[LayeredModel, float]:
    """Propose to perturb one or more within-model parameters.

    Parameters
    ----------
    current : LayeredModel
        Current model.
    prior : Prior
        Prior bounds.
    config : RjMCMCConfig
        Sampler configuration.
    rng : numpy.random.Generator
        Random generator.
    use_aniso : bool, optional
        Whether anisotropy parameters are active.

    Returns
    -------
    tuple
        Proposed model and log proposal ratio.
    """
    prop = current.copy()
    idx = int(rng.integers(0, prop.n_layers))
    prop.log_resistivities[idx] += rng.normal(0.0, config.sigma_change_rho)

    if use_aniso:
        prop.aniso_ratios[idx] *= 10.0 ** rng.normal(0.0, config.sigma_change_aniso)
        prop.strikes[idx] += rng.normal(0.0, config.sigma_change_strike)

    return prop, 0.0


# =============================================================================
#  Random model generation / conversions
# =============================================================================

def random_model(
    prior: Prior,
    rng: np.random.Generator,
    use_aniso: bool = False,
) -> LayeredModel:
    """Draw a random model from the prior.

    Parameters
    ----------
    prior : Prior
        Prior bounds.
    rng : numpy.random.Generator
        Random generator.
    use_aniso : bool, optional
        Whether to draw anisotropy parameters.

    Returns
    -------
    LayeredModel
        Randomly generated model satisfying the prior bounds.
    """
    k = int(rng.integers(prior.k_min, prior.k_max + 1))
    depths = np.sort(rng.uniform(prior.depth_min, prior.depth_max, size=k)) if k > 0 else np.empty(0)
    log_rho = rng.uniform(prior.log_rho_min, prior.log_rho_max, size=k + 1)

    if use_aniso:
        loga = rng.uniform(prior.log_aniso_min, prior.log_aniso_max, size=k + 1)
        ratios = 10.0 ** loga
        strikes = rng.uniform(prior.strike_min, prior.strike_max, size=k + 1)
    else:
        ratios = None
        strikes = None

    return LayeredModel(depths=depths, log_resistivities=log_rho, aniso_ratios=ratios, strikes=strikes)


def model0_to_layered(model0: Dict, use_aniso: bool = False) -> LayeredModel:
    """Convert a py4mt-style starting model dictionary to ``LayeredModel``.

    Parameters
    ----------
    model0 : dict
        Dictionary with keys like ``h_m``, ``sigma_min``, ``sigma_max``,
        and ``strike_deg``.
    use_aniso : bool, optional
        Whether to interpret ``sigma_min``/``sigma_max`` as anisotropic limits.

    Returns
    -------
    LayeredModel
        Converted layered model.
    """
    h_m = np.asarray(model0["h_m"], dtype=float)
    if len(h_m) == 0:
        depths = np.empty(0, dtype=float)
    else:
        if h_m[-1] == 0.0:
            h_use = h_m[:-1]
        else:
            h_use = h_m
        depths = np.cumsum(h_use)

    sigma_min = np.asarray(model0["sigma_min"], dtype=float)
    sigma_max = np.asarray(model0["sigma_max"], dtype=float)
    rho_max = 1.0 / sigma_min
    rho_min = 1.0 / sigma_max
    log_rho = np.log10(rho_max)

    if use_aniso:
        ratios = rho_max / rho_min
        strikes = np.asarray(model0["strike_deg"], dtype=float)
    else:
        ratios = None
        strikes = None

    return LayeredModel(depths=depths, log_resistivities=log_rho, aniso_ratios=ratios, strikes=strikes)


# =============================================================================
#  RJMCMC drivers
# =============================================================================

def run_rjmcmc(
    frequencies: np.ndarray,
    observed: np.ndarray,
    sigma: np.ndarray,
    prior: Prior,
    config: RjMCMCConfig,
    seed: int = 12345,
    initial_model: Optional[LayeredModel] = None,
    use_aniso: bool = False,
    observed_yx: Optional[np.ndarray] = None,
    sigma_yx: Optional[np.ndarray] = None,
    likelihood_mode: str = "rhoa",
    observed_Zdet: Optional[np.ndarray] = None,
    Zdet_sigma: Optional[np.ndarray] = None,
    observed_Z: Optional[np.ndarray] = None,
    observed_Z_err: Optional[np.ndarray] = None,
    z_comps: Sequence[str] = ("xy", "yx"),
    use_pt: bool = False,
    observed_PT: Optional[np.ndarray] = None,
    observed_PT_err: Optional[np.ndarray] = None,
    pt_comps: Sequence[str] = ("xx", "xy", "yx", "yy"),
) -> Dict[str, object]:
    """Run one reversible-jump MCMC chain.

    Parameters
    ----------
    frequencies, observed, sigma : ndarray
        Primary data arrays for the apparent-resistivity likelihood.
    prior : Prior
        Prior object.
    config : RjMCMCConfig
        Sampler configuration.
    seed : int, optional
        Random seed.
    initial_model : LayeredModel, optional
        Starting model for the chain. If omitted, a random prior draw is used.
    use_aniso : bool, optional
        Whether anisotropic models are sampled.
    observed_yx, sigma_yx : ndarray, optional
        Optional yx data for the ``rhoa`` likelihood.
    likelihood_mode : str, optional
        One of ``"rhoa"``, ``"zdet"``, or ``"z_comps"``.
    observed_Zdet, Zdet_sigma : ndarray, optional
        Determinant impedance data and uncertainties.
    observed_Z, observed_Z_err : ndarray, optional
        Full impedance tensor and uncertainties.
    z_comps : sequence of str, optional
        Selected Z components for ``"z_comps"``.
    use_pt : bool, optional
        Whether to include phase tensor in ``"z_comps"``.
    observed_PT, observed_PT_err : ndarray, optional
        Phase tensor and uncertainties.
    pt_comps : sequence of str, optional
        Selected PT components.

    Returns
    -------
    dict
        Dictionary containing accepted samples, diagnostics, and traces.
    """
    rng = np.random.default_rng(seed)

    if initial_model is None:
        current = random_model(prior, rng, use_aniso=use_aniso)
    else:
        current = initial_model.copy()

    lmode = likelihood_mode.lower().strip()
    if lmode == "zdet":
        current_ll = log_likelihood_Zdet(current, frequencies, observed_Zdet, Zdet_sigma)
    elif lmode == "z_comps":
        current_ll = log_likelihood_Z_comps(
            current, frequencies, observed_Z, observed_Z_err,
            z_comps=z_comps, use_pt=use_pt,
            observed_PT=observed_PT, observed_PT_err=observed_PT_err,
            pt_comps=pt_comps,
        )
    else:
        current_ll = log_likelihood(
            current, frequencies, observed, sigma,
            observed_yx=observed_yx, sigma_yx=sigma_yx,
        )
    current_lp = log_prior(current, prior, use_aniso=use_aniso)

    models = []
    n_layers = []
    loglikes = []
    accepts = 0
    proposal_names = ["birth", "death", "move", "change"]
    pweights = np.asarray(config.proposal_weights, dtype=float)
    pweights = pweights / pweights.sum()

    t0 = time.time()
    for it in range(config.n_iterations):
        move = str(rng.choice(proposal_names, p=pweights))
        if move == "birth":
            prop, log_qratio = propose_birth(current, prior, config, rng, use_aniso=use_aniso)
        elif move == "death":
            prop, log_qratio = propose_death(current, prior, config, rng, use_aniso=use_aniso)
        elif move == "move":
            prop, log_qratio = propose_move(current, prior, config, rng)
        else:
            prop, log_qratio = propose_change(current, prior, config, rng, use_aniso=use_aniso)

        prop_lp = log_prior(prop, prior, use_aniso=use_aniso)
        accept = False
        if np.isfinite(prop_lp):
            if lmode == "zdet":
                prop_ll = log_likelihood_Zdet(prop, frequencies, observed_Zdet, Zdet_sigma)
            elif lmode == "z_comps":
                prop_ll = log_likelihood_Z_comps(
                    prop, frequencies, observed_Z, observed_Z_err,
                    z_comps=z_comps, use_pt=use_pt,
                    observed_PT=observed_PT, observed_PT_err=observed_PT_err,
                    pt_comps=pt_comps,
                )
            else:
                prop_ll = log_likelihood(
                    prop, frequencies, observed, sigma,
                    observed_yx=observed_yx, sigma_yx=sigma_yx,
                )
            log_alpha = (prop_lp + prop_ll) - (current_lp + current_ll) + log_qratio
            if np.log(rng.uniform()) < log_alpha:
                accept = True
        if accept:
            current = prop
            current_lp = prop_lp
            current_ll = prop_ll
            accepts += 1

        if it >= config.burn_in and ((it - config.burn_in) % config.thin == 0):
            models.append(current.copy())
            n_layers.append(current.n_layers)
            loglikes.append(current_ll)

        if config.verbose and (it + 1) % max(1, config.n_iterations // 10) == 0:
            frac = 100.0 * (it + 1) / config.n_iterations
            rate = accepts / (it + 1)
            dt = time.time() - t0
            print(f"  iter={it+1:8d}  {frac:5.1f}%  accept={rate:6.3f}  elapsed={dt:8.1f}s")

    return {
        "models": models,
        "n_layers": np.asarray(n_layers, dtype=int),
        "log_likelihood": np.asarray(loglikes, dtype=float),
        "acceptance_rate": accepts / max(1, config.n_iterations),
        "seed": int(seed),
        "likelihood_mode": lmode,
    }


def gelman_rubin(chains: Sequence[np.ndarray]) -> float:
    """Compute the Gelman-Rubin R-hat statistic for scalar chain summaries.

    Parameters
    ----------
    chains : sequence of ndarray
        Sequence of 1-D chains of equal length.

    Returns
    -------
    float
        R-hat convergence diagnostic.
    """
    m = len(chains)
    if m < 2:
        return np.nan
    n = min(len(c) for c in chains)
    if n < 2:
        return np.nan
    arr = np.asarray([np.asarray(c[:n], dtype=float) for c in chains])
    chain_means = np.mean(arr, axis=1)
    grand_mean = np.mean(chain_means)
    B = n * np.var(chain_means, ddof=1)
    W = np.mean(np.var(arr, axis=1, ddof=1))
    var_hat = ((n - 1) / n) * W + B / n
    return float(np.sqrt(var_hat / W)) if W > 0 else np.nan


def _merge_parallel_results(results_list: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Merge per-chain RJMCMC outputs into one dictionary.

    Parameters
    ----------
    results_list : sequence of dict
        Per-chain output dictionaries from ``run_rjmcmc``.

    Returns
    -------
    dict
        Concatenated result dictionary with merged traces and diagnostics.
    """
    models = []
    n_layers = []
    loglikes = []
    layer_chains = []
    acc_rates = []
    for res in results_list:
        models.extend(res.get("models", []))
        n_layers.append(np.asarray(res.get("n_layers", []), dtype=int))
        loglikes.append(np.asarray(res.get("log_likelihood", []), dtype=float))
        layer_chains.append(np.asarray(res.get("n_layers", []), dtype=float))
        acc_rates.append(float(res.get("acceptance_rate", np.nan)))
    merged = {
        "models": models,
        "n_layers": np.concatenate(n_layers) if n_layers else np.empty(0, dtype=int),
        "log_likelihood": np.concatenate(loglikes) if loglikes else np.empty(0, dtype=float),
        "acceptance_rates": np.asarray(acc_rates, dtype=float),
        "gelman_rubin": gelman_rubin(layer_chains),
        "chains": list(results_list),
    }
    return merged


def run_parallel_rjmcmc(
    frequencies: np.ndarray,
    observed: np.ndarray,
    sigma: np.ndarray,
    prior: Prior,
    config: RjMCMCConfig,
    n_chains: int = 4,
    n_jobs: int = 4,
    base_seed: int = 12345,
    initial_model: Optional[LayeredModel] = None,
    use_aniso: bool = False,
    observed_yx: Optional[np.ndarray] = None,
    sigma_yx: Optional[np.ndarray] = None,
    likelihood_mode: str = "rhoa",
    observed_Zdet: Optional[np.ndarray] = None,
    Zdet_sigma: Optional[np.ndarray] = None,
    observed_Z: Optional[np.ndarray] = None,
    observed_Z_err: Optional[np.ndarray] = None,
    z_comps: Sequence[str] = ("xy", "yx"),
    use_pt: bool = False,
    observed_PT: Optional[np.ndarray] = None,
    observed_PT_err: Optional[np.ndarray] = None,
    pt_comps: Sequence[str] = ("xx", "xy", "yx", "yy"),
) -> Dict[str, object]:
    """Run multiple independent RJMCMC chains in parallel.

    Parameters
    ----------
    frequencies, observed, sigma : ndarray
        Primary data arrays for the apparent-resistivity likelihood.
    prior : Prior
        Prior object.
    config : RjMCMCConfig
        Sampler configuration.
    n_chains : int, optional
        Number of independent chains.
    n_jobs : int, optional
        Number of parallel jobs passed to ``joblib``.
    base_seed : int, optional
        Base seed. Chain ``i`` uses ``base_seed + i``.
    initial_model : LayeredModel, optional
        Starting model shared by all chains. If omitted, each chain starts from
        an independent prior draw.
    use_aniso : bool, optional
        Whether anisotropic models are sampled.
    observed_yx, sigma_yx : ndarray, optional
        Optional yx data for the ``rhoa`` likelihood.
    likelihood_mode : str, optional
        One of ``"rhoa"``, ``"zdet"``, or ``"z_comps"``.
    observed_Zdet, Zdet_sigma : ndarray, optional
        Determinant impedance data and uncertainties.
    observed_Z, observed_Z_err : ndarray, optional
        Full impedance tensor and uncertainties.
    z_comps : sequence of str, optional
        Selected Z components for ``"z_comps"``.
    use_pt : bool, optional
        Whether to include phase tensor in the likelihood.
    observed_PT, observed_PT_err : ndarray, optional
        Phase-tensor inputs.
    pt_comps : sequence of str, optional
        Selected PT components.

    Returns
    -------
    dict
        Merged result dictionary across chains.
    """
    from joblib import Parallel, delayed

    print(f"Running {n_chains} RJMCMC chains with n_jobs={n_jobs}.")
    if initial_model is not None:
        print(
            f"  Shared initial model: k={initial_model.k} interfaces, "
            f"{initial_model.n_layers} layers"
        )

    tasks = [
        delayed(run_rjmcmc)(
            frequencies=frequencies,
            observed=observed,
            sigma=sigma,
            prior=prior,
            config=config,
            seed=base_seed + i,
            initial_model=initial_model,
            use_aniso=use_aniso,
            observed_yx=observed_yx,
            sigma_yx=sigma_yx,
            likelihood_mode=likelihood_mode,
            observed_Zdet=observed_Zdet,
            Zdet_sigma=Zdet_sigma,
            observed_Z=observed_Z,
            observed_Z_err=observed_Z_err,
            z_comps=z_comps,
            use_pt=use_pt,
            observed_PT=observed_PT,
            observed_PT_err=observed_PT_err,
            pt_comps=pt_comps,
        )
        for i in range(n_chains)
    ]
    results_list = Parallel(n_jobs=n_jobs)(tasks)
    return _merge_parallel_results(results_list)


# =============================================================================
#  Posterior summaries
# =============================================================================

def compute_posterior_profile(
    models: Sequence[LayeredModel],
    depth_grid: np.ndarray,
    qpairs: Sequence[Tuple[float, float]] = ((2.5, 97.5), (16.0, 84.0)),
) -> Dict[str, np.ndarray]:
    """Compute depth-dependent posterior resistivity summaries.

    Parameters
    ----------
    models : sequence of LayeredModel
        Posterior model samples.
    depth_grid : ndarray
        Target depth grid [m].
    qpairs : sequence of tuple, optional
        Quantile pairs to compute.

    Returns
    -------
    dict
        Posterior summary arrays keyed by statistic name.
    """
    depth_grid = np.asarray(depth_grid, dtype=float)
    if len(models) == 0:
        out = {"depth_m": depth_grid, "median": np.full_like(depth_grid, np.nan)}
        for lo, hi in qpairs:
            out[f"q{lo:g}"] = np.full_like(depth_grid, np.nan)
            out[f"q{hi:g}"] = np.full_like(depth_grid, np.nan)
        return out

    profs = np.empty((len(models), len(depth_grid)), dtype=float)
    for im, mod in enumerate(models):
        bounds = np.r_[0.0, mod.depths, np.inf]
        rho = mod.get_resistivities()
        for iz, z in enumerate(depth_grid):
            ilay = np.searchsorted(bounds, z, side="right") - 1
            ilay = min(max(ilay, 0), len(rho) - 1)
            profs[im, iz] = rho[ilay]

    out = {"depth_m": depth_grid, "median": np.nanmedian(profs, axis=0)}
    for lo, hi in qpairs:
        out[f"q{lo:g}"] = np.nanpercentile(profs, lo, axis=0)
        out[f"q{hi:g}"] = np.nanpercentile(profs, hi, axis=0)
    return out


def compute_posterior_aniso_profile(
    models: Sequence[LayeredModel],
    depth_grid: np.ndarray,
    qpairs: Sequence[Tuple[float, float]] = ((2.5, 97.5), (16.0, 84.0)),
) -> Dict[str, np.ndarray]:
    """Compute depth-dependent posterior anisotropy summaries.

    Parameters
    ----------
    models : sequence of LayeredModel
        Posterior anisotropic model samples.
    depth_grid : ndarray
        Target depth grid [m].
    qpairs : sequence of tuple, optional
        Quantile pairs to compute.

    Returns
    -------
    dict
        Posterior anisotropy-ratio and strike summaries.
    """
    depth_grid = np.asarray(depth_grid, dtype=float)
    if len(models) == 0:
        out = {
            "depth_m": depth_grid,
            "median_aniso": np.full_like(depth_grid, np.nan),
            "median_strike": np.full_like(depth_grid, np.nan),
        }
        for lo, hi in qpairs:
            out[f"q{lo:g}_aniso"] = np.full_like(depth_grid, np.nan)
            out[f"q{hi:g}_aniso"] = np.full_like(depth_grid, np.nan)
            out[f"q{lo:g}_strike"] = np.full_like(depth_grid, np.nan)
            out[f"q{hi:g}_strike"] = np.full_like(depth_grid, np.nan)
        return out

    aa = np.empty((len(models), len(depth_grid)), dtype=float)
    ss = np.empty((len(models), len(depth_grid)), dtype=float)
    for im, mod in enumerate(models):
        bounds = np.r_[0.0, mod.depths, np.inf]
        rat = mod.aniso_ratios
        stk = mod.strikes
        for iz, z in enumerate(depth_grid):
            ilay = np.searchsorted(bounds, z, side="right") - 1
            ilay = min(max(ilay, 0), len(rat) - 1)
            aa[im, iz] = rat[ilay]
            ss[im, iz] = stk[ilay]

    out = {
        "depth_m": depth_grid,
        "median_aniso": np.nanmedian(aa, axis=0),
        "median_strike": np.nanmedian(ss, axis=0),
    }
    for lo, hi in qpairs:
        out[f"q{lo:g}_aniso"] = np.nanpercentile(aa, lo, axis=0)
        out[f"q{hi:g}_aniso"] = np.nanpercentile(aa, hi, axis=0)
        out[f"q{lo:g}_strike"] = np.nanpercentile(ss, lo, axis=0)
        out[f"q{hi:g}_strike"] = np.nanpercentile(ss, hi, axis=0)
    return out


def build_rjmcmc_summary(
    station: str,
    results: Dict[str, object],
    depth_grid_max: float,
    qpairs: Sequence[Tuple[float, float]],
    use_aniso: bool = False,
) -> Dict[str, object]:
    """Build a compact posterior summary dictionary for a station.

    Parameters
    ----------
    station : str
        Station identifier.
    results : dict
        RJMCMC results dictionary.
    depth_grid_max : float
        Maximum depth of the summary grid [m].
    qpairs : sequence of tuple
        Quantile pairs to compute.
    use_aniso : bool, optional
        Whether to compute anisotropy summaries.

    Returns
    -------
    dict
        Summary dictionary suitable for NPZ export.
    """
    depth = np.linspace(0.0, float(depth_grid_max), 200)
    out = {
        "station": station,
        "gelman_rubin": float(results.get("gelman_rubin", np.nan)),
        "n_layers": np.asarray(results.get("n_layers", []), dtype=int),
    }
    out.update(compute_posterior_profile(results.get("models", []), depth, qpairs=qpairs))
    if use_aniso:
        out.update(compute_posterior_aniso_profile(results.get("models", []), depth, qpairs=qpairs))
    return out


# =============================================================================
#  Site loading and I/O helpers
# =============================================================================

def has_aniso() -> bool:
    """Return whether the optional anisotropic forward code is available.

    Returns
    -------
    bool
        ``True`` if ``aniso.py`` was successfully imported.
    """
    return bool(_HAS_ANISO)


def generate_seed() -> int:
    """Generate a random positive integer seed.

    Returns
    -------
    int
        Seed suitable for initializing a pseudo-random generator.
    """
    return int(np.random.SeedSequence().generate_state(1)[0])


def _load_npz_as_dict(path: str | Path) -> Dict:
    """Load an NPZ file into a plain Python dictionary.

    Parameters
    ----------
    path : str or Path
        Path to the NPZ file.

    Returns
    -------
    dict
        Dictionary containing the NPZ arrays.
    """
    with np.load(str(path), allow_pickle=True) as npz:
        return {k: npz[k] for k in npz.files}


def load_site(
    path: str | Path,
    noise_level: float = 0.02,
    sigma_floor: float = 0.0,
    err_method: str = "bootstrap",
    err_nsim: int = 200,
    do_compute_pt: bool = True,
) -> Dict[str, object]:
    """Load one MT site from EDI or NPZ and standardize required keys.

    Parameters
    ----------
    path : str or Path
        Input file path.
    noise_level : float, optional
        Relative noise level used when explicit errors are unavailable.
    sigma_floor : float, optional
        Minimum uncertainty floor applied to log10 apparent resistivity.
    err_method : str, optional
        Placeholder kept for compatibility with the driver.
    err_nsim : int, optional
        Number of Monte-Carlo simulations for derived uncertainties.
    do_compute_pt : bool, optional
        Whether to compute phase tensor if possible.

    Returns
    -------
    dict
        Standardized site dictionary.
    """
    path = Path(path)
    if path.suffix.lower() == ".edi":
        site = data_proc.load_edi(str(path))
    elif path.suffix.lower() == ".npz":
        site = _load_npz_as_dict(path)
        if "data_dict" in site and isinstance(site["data_dict"], np.ndarray):
            try:
                site = site["data_dict"].item()
            except Exception:
                pass
    else:
        raise ValueError(f"Unsupported input format: {path}")

    site = dict(site)
    site.setdefault("station", path.stem)

    freqs = np.asarray(site.get("freq", site.get("frequencies")), dtype=float)
    site["frequencies"] = freqs

    if "Z" in site:
        site["Z"] = np.asarray(site["Z"])
        if "Z_err" in site:
            site["Z_err"] = np.asarray(site["Z_err"], dtype=float)
        elif "Zvar" in site:
            site["Z_err"] = np.sqrt(np.asarray(site["Zvar"], dtype=float))
        else:
            site["Z_err"] = noise_level * np.abs(site["Z"]).astype(float)

        if "Zdet" not in site:
            site["Zdet"] = compute_Zdet(site["Z"])
        if "Zdet_err" not in site:
            site["Zdet_err"] = compute_Zdet_err(site["Z"], site["Z_err"], nsim=err_nsim)

        rho_det, pha_det = compute_rhophas_from_Zdet(site["Zdet"], freqs)
        site["rho_a_det"] = rho_det
        site["phase_det"] = pha_det

    if "rho_a" not in site:
        if "rho_a_det" in site:
            site["rho_a"] = np.asarray(site["rho_a_det"], dtype=float)
        elif "Z" in site:
            mu0 = 4.0 * np.pi * 1e-7
            omega = 2.0 * np.pi * freqs
            site["rho_a"] = np.abs(site["Z"][:, 0, 1]) ** 2 / (omega * mu0)

    if "sigma" not in site:
        if "Zdet_err" in site and "Zdet" in site:
            zabs = np.maximum(np.abs(site["Zdet"]), 1e-30)
            sig_log = 2.0 * np.asarray(site["Zdet_err"], dtype=float) / (zabs * np.log(10.0))
            site["sigma"] = np.maximum(sig_log, sigma_floor)
        else:
            site["sigma"] = np.full_like(site["rho_a"], max(noise_level, sigma_floor), dtype=float)

    if "rho_a_yx" not in site and "Z" in site:
        mu0 = 4.0 * np.pi * 1e-7
        omega = 2.0 * np.pi * freqs
        site["rho_a_yx"] = np.abs(site["Z"][:, 1, 0]) ** 2 / (omega * mu0)
        site["sigma_yx"] = np.asarray(site.get("sigma", np.full_like(site["rho_a_yx"], noise_level)), dtype=float)

    if do_compute_pt and "PT" not in site and "Z" in site:
        site["PT"] = _phase_tensor_from_Z(site["Z"])
    if do_compute_pt and "PT" in site and "PT_err" not in site:
        site["PT_err"] = 0.05 * np.maximum(np.abs(site["PT"]), 1e-6)

    return site


def save_results_npz(results: Dict[str, object], path: str | Path) -> None:
    """Save RJMCMC results to a compressed NPZ file.

    Parameters
    ----------
    results : dict
        RJMCMC results dictionary.
    path : str or Path
        Output NPZ file path.
    """
    payload = dict(results)
    models = payload.pop("models", [])
    payload["models_obj"] = np.array(models, dtype=object)
    payload["chains_obj"] = np.array(payload.get("chains", []), dtype=object)
    np.savez_compressed(str(path), **payload)


def load_results_npz(path: str | Path) -> Dict[str, object]:
    """Load RJMCMC results from a compressed NPZ file.

    Parameters
    ----------
    path : str or Path
        Input NPZ file path.

    Returns
    -------
    dict
        Reconstructed results dictionary.
    """
    with np.load(str(path), allow_pickle=True) as npz:
        out = {k: npz[k] for k in npz.files}
    if "models_obj" in out:
        out["models"] = list(out.pop("models_obj"))
    if "chains_obj" in out:
        out["chains"] = list(out.pop("chains_obj"))
    return out
