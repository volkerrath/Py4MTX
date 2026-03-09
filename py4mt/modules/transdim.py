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
    """Create a directory if needed and return its Path."""
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
    thicknesses : (n_layers - 1,)   Layer thicknesses [m].
    resistivities : (n_layers,)     Resistivities [Ω·m].
    frequencies : (n_freq,)         Frequencies [Hz].
    full_output : bool
        If False (default), return only apparent resistivity.
        If True, return a dict with ``rho_a``, ``phase_deg``,
        ``Z_re``, ``Z_im``.

    Returns
    -------
    rho_a : (n_freq,)  when ``full_output=False``
    dict : when ``full_output=True``
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
    thicknesses : (n_layers - 1,)  Layer thicknesses [m].
    resistivities : (n_layers,)    Maximum horizontal resistivity [Ω·m].
    frequencies : (n_freq,)        Frequencies [Hz].
    aniso_ratios : (n_layers,)     ρ_max / ρ_min  (≥ 1; 1 = isotropic).
    strikes : (n_layers,)          Anisotropy strike [deg].

    Returns
    -------
    dict  ``{'rho_a_xy': ..., 'rho_a_yx': ...}``  each (n_freq,).
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


# -- impedance-level forward models -----------------------------------------

def mt_forward_1d_isotropic_impedance(
    thicknesses: np.ndarray,
    resistivities: np.ndarray,
    frequencies: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Isotropic forward model returning the full 2×2 impedance tensor.

    For isotropic media:  Zxx=Zyy=0, Zxy=Z, Zyx=−Z.

    Returns
    -------
    dict with ``Z`` (n_freq, 2, 2) complex, ``rho_a`` (n_freq,),
    ``phase_deg`` (n_freq,).
    """
    mu0 = 4.0 * np.pi * 1e-7
    omega = 2.0 * np.pi * frequencies
    n_layers = len(resistivities)
    nf = len(frequencies)

    Z_tensor = np.zeros((nf, 2, 2), dtype=np.complex128)
    rho_a = np.zeros(nf)
    phase_deg = np.zeros(nf)

    for fi, w in enumerate(omega):
        k = np.sqrt(1j * w * mu0 / resistivities)
        Z_scalar = w * mu0 / k[-1]
        for j in range(n_layers - 2, -1, -1):
            Z_j = w * mu0 / k[j]
            r = (Z_j - Z_scalar) / (Z_j + Z_scalar)
            e2 = np.exp(-2 * k[j] * thicknesses[j])
            Z_scalar = Z_j * (1 - r * e2) / (1 + r * e2)

        Z_tensor[fi, 0, 1] = Z_scalar       # Zxy
        Z_tensor[fi, 1, 0] = -Z_scalar      # Zyx
        rho_a[fi] = np.abs(Z_scalar) ** 2 / (w * mu0)
        phase_deg[fi] = np.abs(np.degrees(np.arctan2(Z_scalar.imag, Z_scalar.real)))

    return {"Z": Z_tensor, "rho_a": rho_a, "phase_deg": phase_deg}


def mt_forward_1d_anisotropic_impedance(
    thicknesses: np.ndarray,
    resistivities: np.ndarray,
    frequencies: np.ndarray,
    aniso_ratios: np.ndarray,
    strikes: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Anisotropic forward model returning the full 2×2 impedance tensor.

    Returns
    -------
    dict with ``Z`` (n_freq, 2, 2) complex, ``rho_a_xy`` (n_freq,),
    ``rho_a_yx`` (n_freq,).
    """
    if not _HAS_ANISO:
        raise ImportError("aniso.py not found.")

    periods = 1.0 / frequencies
    rho_max = resistivities.copy()
    rho_min = resistivities / aniso_ratios
    h_m = np.append(thicknesses, 1e6)

    result = aniso1d_impedance_sens_simple(
        periods_s=periods, h_m=h_m, rho_max=rho_max,
        rho_min=rho_min, strike_deg=strikes, compute_sens=False,
    )

    Z = result["Z"]
    mu0 = 4.0 * np.pi * 1e-7
    omega = 2.0 * np.pi * frequencies

    return {
        "Z": Z,
        "rho_a_xy": np.abs(Z[:, 0, 1]) ** 2 / (omega * mu0),
        "rho_a_yx": np.abs(Z[:, 1, 0]) ** 2 / (omega * mu0),
    }


def mt_forward_1d_isotropic_full(
    thicknesses: np.ndarray,
    resistivities: np.ndarray,
    frequencies: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Convenience wrapper: equivalent to ``mt_forward_1d_isotropic(..., full_output=True)``."""
    return mt_forward_1d_isotropic(thicknesses, resistivities, frequencies,
                                   full_output=True)


# =============================================================================
#  Determinant impedance utilities
# =============================================================================

def compute_Zdet(Z: np.ndarray) -> np.ndarray:
    """Compute the determinant impedance from a stack of 2×2 tensors.

    Z_det = sqrt(Zxx*Zyy - Zxy*Zyx)

    The branch of the square root is chosen so that Im(Z_det) > 0
    (consistent with the MT sign convention for a positive phase).

    Parameters
    ----------
    Z : (n_freq, 2, 2) complex

    Returns
    -------
    Z_det : (n_freq,) complex
    """
    Z = np.asarray(Z, dtype=np.complex128)
    if Z.ndim != 3 or Z.shape[1:] != (2, 2):
        raise ValueError("Z must have shape (n_freq, 2, 2).")
    det = Z[:, 0, 0] * Z[:, 1, 1] - Z[:, 0, 1] * Z[:, 1, 0]
    Zd = np.sqrt(det)
    # Enforce Im(Z_det) > 0  (MT sign convention)
    flip = Zd.imag < 0
    Zd[flip] = -Zd[flip]
    return Zd


def compute_Zdet_err(
    Z: np.ndarray,
    Z_err: np.ndarray,
    *,
    nsim: int = 200,
    random_state: int | None = 0,
) -> np.ndarray:
    """Estimate Z_det uncertainty by Monte-Carlo propagation from Z_err.

    Parameters
    ----------
    Z : (n_freq, 2, 2) complex
    Z_err : (n_freq, 2, 2) float — per-element absolute error.
    nsim : int — number of Monte-Carlo samples.
    random_state : int or None

    Returns
    -------
    Zdet_err : (n_freq,) float — standard deviation of |Z_det|.
    """
    Z = np.asarray(Z, dtype=np.complex128)
    Z_err = np.asarray(Z_err, dtype=float)
    rng = np.random.default_rng(None if random_state is None else int(random_state))

    Zdet_samples = np.empty((nsim, Z.shape[0]), dtype=float)
    for k in range(nsim):
        nre = rng.standard_normal(Z.shape)
        nim = rng.standard_normal(Z.shape)
        Zs = Z + (nre + 1j * nim) * Z_err
        Zdet_samples[k] = np.abs(compute_Zdet(Zs))

    return Zdet_samples.std(axis=0, ddof=1)


def compute_rhophas_from_Zdet(
    Z_det: np.ndarray,
    frequencies: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute apparent resistivity and phase from determinant impedance.

    Parameters
    ----------
    Z_det : (n_freq,) complex
    frequencies : (n_freq,) Hz

    Returns
    -------
    rho_a : (n_freq,) Ω·m
    phase_deg : (n_freq,) degrees (0–90 range)
    """
    mu0 = 4.0 * np.pi * 1e-7
    omega = 2.0 * np.pi * np.asarray(frequencies, dtype=float)
    Z_det = np.asarray(Z_det, dtype=np.complex128)
    rho_a = np.abs(Z_det) ** 2 / (omega * mu0)
    phase_deg = np.abs(np.degrees(np.arctan2(Z_det.imag, Z_det.real)))
    return rho_a, phase_deg


def pack_Zdet_vector(Z_det: np.ndarray) -> np.ndarray:
    """Pack complex Z_det into a real [Re, Im, Re, Im, ...] vector.

    Parameters
    ----------
    Z_det : (n_freq,) complex

    Returns
    -------
    (2 * n_freq,) float — interleaved Re/Im.
    """
    Z_det = np.asarray(Z_det, dtype=np.complex128).ravel()
    n = Z_det.size
    out = np.empty(2 * n, dtype=float)
    out[0::2] = Z_det.real
    out[1::2] = Z_det.imag
    return out


def pack_Zdet_sigma(
    Z_det_err: np.ndarray,
    *,
    sigma_floor: float = 0.0,
) -> np.ndarray:
    """Pack Z_det error into a sigma vector matching :func:`pack_Zdet_vector`.

    Parameters
    ----------
    Z_det_err : (n_freq,) float — absolute error on |Z_det|.
        Applied identically to both Re and Im parts.
    sigma_floor : float

    Returns
    -------
    (2 * n_freq,) float
    """
    Z_det_err = np.asarray(Z_det_err, dtype=float).ravel()
    n = Z_det_err.size
    out = np.empty(2 * n, dtype=float)
    for i in range(n):
        s = float(np.sqrt(Z_det_err[i] ** 2 + sigma_floor ** 2))
        out[2 * i] = s
        out[2 * i + 1] = s
    return out


# =============================================================================
#  Z-component and phase-tensor packing
# =============================================================================

_COMP_TO_IJ: Dict[str, Tuple[int, int]] = {
    "xx": (0, 0), "xy": (0, 1), "yx": (1, 0), "yy": (1, 1),
}


def pack_Z_vector(
    Z: np.ndarray,
    comps: Sequence[str] = ("xx", "xy", "yx", "yy"),
) -> np.ndarray:
    """Pack selected Z components into a real Re/Im vector.

    Ordering: for each frequency, for each component, [Re, Im].

    Parameters
    ----------
    Z : (n_freq, 2, 2) complex
    comps : component labels

    Returns
    -------
    (n_freq * n_comp * 2,) float
    """
    Z = np.asarray(Z, dtype=np.complex128)
    ij = [_COMP_TO_IJ[c.lower().strip()] for c in comps]
    n = Z.shape[0]
    out = np.empty(n * len(ij) * 2, dtype=float)
    k = 0
    for ip in range(n):
        for (i, j) in ij:
            out[k] = Z[ip, i, j].real; k += 1
            out[k] = Z[ip, i, j].imag; k += 1
    return out


def pack_Z_sigma(
    Z_err: np.ndarray,
    comps: Sequence[str] = ("xx", "xy", "yx", "yy"),
    *,
    sigma_floor: float = 0.0,
) -> np.ndarray:
    """Pack Z_err for selected components into a sigma vector."""
    Z_err = np.asarray(Z_err)
    ij = [_COMP_TO_IJ[c.lower().strip()] for c in comps]
    n = Z_err.shape[0]
    out = np.empty(n * len(ij) * 2, dtype=float)
    k = 0
    for ip in range(n):
        for (i, j) in ij:
            s = float(np.abs(Z_err[ip, i, j]))
            s = float(np.sqrt(s * s + sigma_floor * sigma_floor))
            out[k] = s; k += 1
            out[k] = s; k += 1
    return out


def compute_phase_tensor(Z: np.ndarray) -> np.ndarray:
    """Compute the phase tensor from a stack of 2×2 impedance tensors.

    PT = Re(Z)^{-1} @ Im(Z),  computed per frequency.

    Parameters
    ----------
    Z : (n_freq, 2, 2) complex

    Returns
    -------
    PT : (n_freq, 2, 2) real
    """
    nf = Z.shape[0]
    PT = np.zeros((nf, 2, 2), dtype=float)
    for fi in range(nf):
        re = Z[fi].real
        im = Z[fi].imag
        det = re[0, 0] * re[1, 1] - re[0, 1] * re[1, 0]
        if abs(det) < 1e-30:
            continue
        re_inv = np.array([[re[1, 1], -re[0, 1]],
                           [-re[1, 0], re[0, 0]]]) / det
        PT[fi] = re_inv @ im
    return PT


def pack_P_vector(
    P: np.ndarray,
    comps: Sequence[str] = ("xx", "xy", "yx", "yy"),
) -> np.ndarray:
    """Pack phase tensor components into a real data vector."""
    P = np.asarray(P, dtype=float)
    ij = [_COMP_TO_IJ[c.lower().strip()] for c in comps]
    n = P.shape[0]
    out = np.empty(n * len(ij), dtype=float)
    k = 0
    for ip in range(n):
        for (i, j) in ij:
            out[k] = float(P[ip, i, j]); k += 1
    return out


def pack_P_sigma(
    P_err: np.ndarray,
    comps: Sequence[str] = ("xx", "xy", "yx", "yy"),
    *,
    sigma_floor: float = 0.0,
) -> np.ndarray:
    """Pack phase tensor uncertainties into a sigma vector."""
    P_err = np.asarray(P_err, dtype=float)
    ij = [_COMP_TO_IJ[c.lower().strip()] for c in comps]
    n = P_err.shape[0]
    out = np.empty(n * len(ij), dtype=float)
    k = 0
    for ip in range(n):
        for (i, j) in ij:
            s = float(abs(P_err[ip, i, j]))
            out[k] = float(np.sqrt(s * s + sigma_floor * sigma_floor)); k += 1
    return out


def has_aniso() -> bool:
    """Return True if the anisotropic forward model is available."""
    return _HAS_ANISO


# =============================================================================
#  Model representation
# =============================================================================

@dataclass
class LayeredModel:
    """A 1-D layered earth model (isotropic or anisotropic).

    Attributes
    ----------
    interfaces : (k,)        Sorted depths of internal interfaces [m].
    resistivities : (k+1,)   log10(ρ_max) per layer.
    aniso_ratios : (k+1,) or None   ρ_max/ρ_min per layer (≥ 1).
    strikes : (k+1,) or None        Anisotropy strike [deg].
    """
    interfaces: np.ndarray
    resistivities: np.ndarray
    aniso_ratios: Optional[np.ndarray] = None
    strikes: Optional[np.ndarray] = None

    @property
    def k(self) -> int:
        """Number of internal interfaces."""
        return len(self.interfaces)

    @property
    def n_layers(self) -> int:
        return self.k + 1

    @property
    def is_anisotropic(self) -> bool:
        return self.aniso_ratios is not None

    def get_thicknesses(self) -> np.ndarray:
        """Layer thicknesses (all but half-space)."""
        return np.diff(np.concatenate(([0.0], self.interfaces)))

    def get_resistivities(self) -> np.ndarray:
        """Resistivities in linear scale [Ω·m]."""
        return 10.0 ** self.resistivities

    def copy(self) -> "LayeredModel":
        return LayeredModel(
            interfaces=self.interfaces.copy(),
            resistivities=self.resistivities.copy(),
            aniso_ratios=(
                self.aniso_ratios.copy() if self.aniso_ratios is not None else None
            ),
            strikes=self.strikes.copy() if self.strikes is not None else None,
        )


# =============================================================================
#  Prior specification
# =============================================================================

@dataclass
class Prior:
    """Uniform prior bounds for the transdimensional sampler."""
    k_min: int = 1
    k_max: int = 30
    depth_min: float = 1.0
    depth_max: float = 5000.0
    log_rho_min: float = -1.0
    log_rho_max: float = 5.0
    log_aniso_min: float = 0.0
    log_aniso_max: float = 2.0
    strike_min: float = -90.0
    strike_max: float = 90.0

    def log_prior(self, model: LayeredModel) -> float:
        """Evaluate log-prior (0 inside bounds, -inf outside)."""
        if model.k < self.k_min or model.k > self.k_max:
            return -np.inf
        if np.any(model.interfaces < self.depth_min) or \
           np.any(model.interfaces > self.depth_max):
            return -np.inf
        if np.any(model.resistivities < self.log_rho_min) or \
           np.any(model.resistivities > self.log_rho_max):
            return -np.inf
        if model.is_anisotropic:
            log_ar = np.log10(model.aniso_ratios)
            if np.any(log_ar < self.log_aniso_min) or \
               np.any(log_ar > self.log_aniso_max):
                return -np.inf
            if np.any(model.strikes < self.strike_min) or \
               np.any(model.strikes > self.strike_max):
                return -np.inf
        return 0.0


# =============================================================================
#  Likelihood functions
# =============================================================================

def log_likelihood(
    model: LayeredModel,
    frequencies: np.ndarray,
    observed: np.ndarray,
    sigma: np.ndarray,
    use_aniso: bool = False,
    observed_yx: Optional[np.ndarray] = None,
    sigma_yx: Optional[np.ndarray] = None,
) -> float:
    """Gaussian log-likelihood in log10(ρ_a) space."""
    thicknesses = model.get_thicknesses()
    resistivities = model.get_resistivities()

    if use_aniso and model.is_anisotropic:
        result = mt_forward_1d_anisotropic(
            thicknesses, resistivities, frequencies,
            model.aniso_ratios, model.strikes,
        )
        res_xy = (np.log10(result["rho_a_xy"]) - np.log10(observed)) / sigma
        ll = -0.5 * np.sum(res_xy ** 2)
        if observed_yx is not None and sigma_yx is not None:
            res_yx = (np.log10(result["rho_a_yx"]) - np.log10(observed_yx)) / sigma_yx
            ll += -0.5 * np.sum(res_yx ** 2)
        return ll
    else:
        predicted = mt_forward_1d_isotropic(thicknesses, resistivities, frequencies)
        residuals = (np.log10(predicted) - np.log10(observed)) / sigma
        return -0.5 * np.sum(residuals ** 2)


def log_likelihood_Zdet(
    model: LayeredModel,
    frequencies: np.ndarray,
    Zdet_obs: np.ndarray,
    Zdet_sigma: np.ndarray,
) -> float:
    """Gaussian log-likelihood on Re/Im of the determinant impedance Z_det.

    For isotropic 1-D models, Z_det = Z_xy (since Zxx=Zyy=0 and Zyx=-Zxy,
    so det = Zxy² and sqrt(det) = Zxy choosing the right branch).

    Parameters
    ----------
    model : LayeredModel
    frequencies : (nf,) Hz
    Zdet_obs : (nf,) complex — observed Z_det
    Zdet_sigma : (nf,) float — absolute error on |Z_det| (applied to Re & Im)

    Returns
    -------
    float — log-likelihood value
    """
    thicknesses = model.get_thicknesses()
    resistivities = model.get_resistivities()

    pred = mt_forward_1d_isotropic_impedance(thicknesses, resistivities, frequencies)
    Z_pred = pred["Z"]
    Zdet_pred = compute_Zdet(Z_pred)

    res_re = (Zdet_pred.real - Zdet_obs.real) / Zdet_sigma
    res_im = (Zdet_pred.imag - Zdet_obs.imag) / Zdet_sigma
    return -0.5 * (np.sum(res_re ** 2) + np.sum(res_im ** 2))


def log_likelihood_Z_comps(
    model: LayeredModel,
    frequencies: np.ndarray,
    Z_obs: np.ndarray,
    Z_sigma: np.ndarray,
    comps: Sequence[str] = ("xx", "xy", "yx", "yy"),
    *,
    use_pt: bool = False,
    PT_obs: Optional[np.ndarray] = None,
    PT_sigma: Optional[np.ndarray] = None,
    pt_comps: Sequence[str] = ("xx", "xy", "yx", "yy"),
) -> float:
    """Gaussian log-likelihood on selected Z components (Re/Im) + optional PT.

    Designed for the anisotropic case where the full tensor matters.

    Parameters
    ----------
    model : LayeredModel
    frequencies : (nf,) Hz
    Z_obs : (nf, 2, 2) complex
    Z_sigma : (nf, 2, 2) float — per-element absolute error
    comps : Z components to include in the misfit
    use_pt : bool — also include phase tensor residuals
    PT_obs : (nf, 2, 2) float — observed phase tensor
    PT_sigma : (nf, 2, 2) float — PT uncertainties
    pt_comps : PT components to include

    Returns
    -------
    float
    """
    thicknesses = model.get_thicknesses()
    resistivities = model.get_resistivities()

    if model.is_anisotropic:
        pred = mt_forward_1d_anisotropic_impedance(
            thicknesses, resistivities, frequencies,
            model.aniso_ratios, model.strikes)
    else:
        pred = mt_forward_1d_isotropic_impedance(
            thicknesses, resistivities, frequencies)

    Z_pred = pred["Z"]

    y_obs = pack_Z_vector(Z_obs, comps)
    y_pred = pack_Z_vector(Z_pred, comps)
    sig = pack_Z_sigma(Z_sigma, comps)
    res = (y_pred - y_obs) / sig
    ll = -0.5 * np.sum(res ** 2)

    if use_pt and PT_obs is not None and PT_sigma is not None:
        PT_pred = compute_phase_tensor(Z_pred)
        p_obs = pack_P_vector(PT_obs, pt_comps)
        p_pred = pack_P_vector(PT_pred, pt_comps)
        p_sig = pack_P_sigma(PT_sigma, pt_comps)
        res_p = (p_pred - p_obs) / p_sig
        ll += -0.5 * np.sum(res_p ** 2)

    return ll


# =============================================================================
#  Proposal functions
# =============================================================================

def propose_birth(
    model: LayeredModel, prior: Prior,
    sigma_rho: float = 0.5, use_aniso: bool = False,
    sigma_aniso: float = 0.1, sigma_strike: float = 10.0,
) -> Tuple[LayeredModel, float]:
    """Birth proposal — add a new interface at a random depth."""
    new = model.copy()
    z_new = np.random.uniform(prior.depth_min, prior.depth_max)
    idx = np.searchsorted(new.interfaces, z_new)
    rho_old = new.resistivities[idx]

    delta = np.random.normal(0, sigma_rho)
    new.interfaces = np.insert(new.interfaces, idx, z_new)
    new.resistivities = np.delete(new.resistivities, idx)
    new.resistivities = np.insert(
        new.resistivities, idx, [rho_old - delta * 0.5, rho_old + delta * 0.5]
    )

    if use_aniso and new.is_anisotropic:
        ar_old, st_old = new.aniso_ratios[idx], new.strikes[idx]
        d_ar = np.random.normal(0, sigma_aniso)
        d_st = np.random.normal(0, sigma_strike)
        new.aniso_ratios = np.delete(new.aniso_ratios, idx)
        new.aniso_ratios = np.insert(new.aniso_ratios, idx, [
            max(1.0, ar_old * 10 ** (-d_ar * 0.5)),
            max(1.0, ar_old * 10 ** ( d_ar * 0.5)),
        ])
        new.strikes = np.delete(new.strikes, idx)
        new.strikes = np.insert(new.strikes, idx,
                                [st_old - d_st * 0.5, st_old + d_st * 0.5])

    return new, np.log(new.k) - np.log(prior.depth_max - prior.depth_min)


def propose_death(
    model: LayeredModel, prior: Prior, use_aniso: bool = False,
) -> Tuple[LayeredModel, float]:
    """Death proposal — remove a random interface."""
    if model.k < prior.k_min + 1:
        return model.copy(), -np.inf

    new = model.copy()
    idx = np.random.randint(0, new.k)

    rho_merged = 0.5 * (new.resistivities[idx] + new.resistivities[idx + 1])
    new.interfaces = np.delete(new.interfaces, idx)
    new.resistivities = np.delete(new.resistivities, idx)
    new.resistivities = np.delete(new.resistivities, idx)
    new.resistivities = np.insert(new.resistivities, idx, rho_merged)

    if use_aniso and new.is_anisotropic:
        ar_merged = np.sqrt(new.aniso_ratios[idx] * new.aniso_ratios[idx + 1])
        st_merged = 0.5 * (new.strikes[idx] + new.strikes[idx + 1])
        new.aniso_ratios = np.delete(new.aniso_ratios, idx)
        new.aniso_ratios = np.delete(new.aniso_ratios, idx)
        new.aniso_ratios = np.insert(new.aniso_ratios, idx, ar_merged)
        new.strikes = np.delete(new.strikes, idx)
        new.strikes = np.delete(new.strikes, idx)
        new.strikes = np.insert(new.strikes, idx, st_merged)

    return new, np.log(prior.depth_max - prior.depth_min) - np.log(model.k)


def propose_move(
    model: LayeredModel, prior: Prior, sigma_z: float = 50.0,
) -> Tuple[LayeredModel, float]:
    """Move proposal — perturb a random interface depth."""
    if model.k == 0:
        return model.copy(), 0.0
    new = model.copy()
    idx = np.random.randint(0, new.k)
    new.interfaces[idx] += np.random.normal(0, sigma_z)
    new.interfaces.sort()
    return new, 0.0


def propose_change(
    model: LayeredModel, prior: Prior,
    sigma_rho: float = 0.3, use_aniso: bool = False,
    sigma_aniso: float = 0.05, sigma_strike: float = 5.0,
) -> Tuple[LayeredModel, float]:
    """Change proposal — perturb a random layer's properties."""
    new = model.copy()
    if use_aniso and new.is_anisotropic:
        choice = np.random.choice(3)
        idx = np.random.randint(0, new.n_layers)
        if choice == 0:
            new.resistivities[idx] += np.random.normal(0, sigma_rho)
        elif choice == 1:
            log_ar = np.log10(new.aniso_ratios[idx])
            log_ar += np.random.normal(0, sigma_aniso)
            new.aniso_ratios[idx] = max(1.0, 10 ** log_ar)
        else:
            new.strikes[idx] += np.random.normal(0, sigma_strike)
    else:
        idx = np.random.randint(0, new.n_layers)
        new.resistivities[idx] += np.random.normal(0, sigma_rho)
    return new, 0.0


# =============================================================================
#  Sampler configuration
# =============================================================================

@dataclass
class RjMCMCConfig:
    """Tuning knobs for the rjMCMC sampler."""
    n_iterations: int = 200_000
    burn_in: int = 50_000
    proposal_weights: tuple = (0.20, 0.20, 0.25, 0.35)
    sigma_birth_rho: float = 0.1
    sigma_move_z: float = 100.0
    sigma_change_rho: float = 0.15
    thin: int = 10
    verbose: bool = True
    sigma_birth_aniso: float = 0.1
    sigma_birth_strike: float = 10.0
    sigma_change_aniso: float = 0.05
    sigma_change_strike: float = 5.0


# =============================================================================
#  Single-chain rjMCMC
# =============================================================================

def run_rjmcmc(
    frequencies: np.ndarray,
    observed: np.ndarray,
    sigma: np.ndarray,
    prior: Prior,
    config: RjMCMCConfig,
    initial_model: Optional[LayeredModel] = None,
    use_aniso: bool = False,
    observed_yx: Optional[np.ndarray] = None,
    sigma_yx: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    chain_id: int = 0,
    likelihood_mode: str = "rhoa",
    observed_Z: Optional[np.ndarray] = None,
    observed_Z_err: Optional[np.ndarray] = None,
    z_comps: Sequence[str] = ("xy", "yx"),
    observed_Zdet: Optional[np.ndarray] = None,
    Zdet_sigma: Optional[np.ndarray] = None,
    use_pt: bool = False,
    observed_PT: Optional[np.ndarray] = None,
    observed_PT_err: Optional[np.ndarray] = None,
    pt_comps: Sequence[str] = ("xx", "xy", "yx", "yy"),
) -> Dict:
    """Run a single transdimensional MCMC chain.

    Parameters
    ----------
    likelihood_mode : str
        Controls which likelihood function is used:

        ``"rhoa"`` (default)
            Gaussian on log10(ρ_a).  Uses ``observed`` and ``sigma``.

        ``"Zdet"``
            Gaussian on Re/Im of the determinant impedance.
            Requires ``observed_Zdet`` and ``Zdet_sigma``.

        ``"Z_comps"``
            Gaussian on Re/Im of selected Z components, optionally
            including phase tensor.  Requires ``observed_Z`` and
            ``observed_Z_err``.

    Returns
    -------
    dict with ``models``, ``log_likes``, ``n_layers``,
    ``full_ll_trace``, ``acceptance``, ``chain_id``.
    """
    if seed is not None:
        np.random.seed(seed)

    lmode = str(likelihood_mode).lower().strip()

    if initial_model is None:
        k0 = 3
        interfaces = np.sort(
            np.random.uniform(prior.depth_min, prior.depth_max * 0.5, k0))
        resistivities = np.random.uniform(
            prior.log_rho_min + 1, prior.log_rho_max - 1, k0 + 1)
        if use_aniso:
            aniso_ratios = 10 ** np.random.uniform(
                prior.log_aniso_min, min(prior.log_aniso_max, 0.5), k0 + 1)
            strikes = np.random.uniform(prior.strike_min, prior.strike_max, k0 + 1)
            current = LayeredModel(interfaces, resistivities, aniso_ratios, strikes)
        else:
            current = LayeredModel(interfaces, resistivities)
    else:
        current = initial_model.copy()

    # ---- Select likelihood function ----------------------------------------
    def _eval_ll(m):
        if lmode == "zdet":
            return log_likelihood_Zdet(m, frequencies, observed_Zdet, Zdet_sigma)
        elif lmode == "z_comps":
            return log_likelihood_Z_comps(
                m, frequencies, observed_Z, observed_Z_err, comps=z_comps,
                use_pt=use_pt, PT_obs=observed_PT, PT_sigma=observed_PT_err,
                pt_comps=pt_comps)
        else:  # "rhoa"
            return log_likelihood(
                m, frequencies, observed, sigma, use_aniso,
                observed_yx, sigma_yx)

    current_ll = _eval_ll(current)
    current_lp = prior.log_prior(current)

    models: List[LayeredModel] = []
    log_likes: List[float] = []
    n_layers_trace: List[int] = []
    full_ll_trace: List[float] = []
    proposal_names = ["birth", "death", "move", "change"]
    counts = {n: 0 for n in proposal_names}
    accepts = {n: 0 for n in proposal_names}

    weights = np.array(config.proposal_weights, dtype=float)
    weights /= weights.sum()

    if config.verbose:
        print(f"  [Chain {chain_id}] Starting rjMCMC — "
              f"{config.n_iterations:,} iter, burn-in {config.burn_in:,}, "
              f"likelihood={lmode}")

    for it in range(config.n_iterations):
        ptype = np.random.choice(4, p=weights)
        pname = proposal_names[ptype]
        counts[pname] += 1

        if ptype == 0:
            proposed, log_qr = propose_birth(
                current, prior, config.sigma_birth_rho,
                use_aniso, config.sigma_birth_aniso, config.sigma_birth_strike)
        elif ptype == 1:
            proposed, log_qr = propose_death(current, prior, use_aniso)
        elif ptype == 2:
            proposed, log_qr = propose_move(current, prior, config.sigma_move_z)
        else:
            proposed, log_qr = propose_change(
                current, prior, config.sigma_change_rho,
                use_aniso, config.sigma_change_aniso, config.sigma_change_strike)

        proposed_lp = prior.log_prior(proposed)
        if proposed_lp > -np.inf:
            proposed_ll = _eval_ll(proposed)
            log_alpha = ((proposed_ll - current_ll)
                         + (proposed_lp - current_lp) + log_qr)
            if np.log(np.random.uniform()) < log_alpha:
                current = proposed
                current_ll = proposed_ll
                current_lp = proposed_lp
                accepts[pname] += 1

        full_ll_trace.append(current_ll)

        if it >= config.burn_in and (it - config.burn_in) % config.thin == 0:
            models.append(current.copy())
            log_likes.append(current_ll)
            n_layers_trace.append(current.n_layers)

        if config.verbose and (it + 1) % 50_000 == 0:
            rates = {n: (accepts[n] / counts[n] * 100 if counts[n] > 0 else 0)
                     for n in proposal_names}
            print(f"  [Chain {chain_id}] Iter {it+1:>7,} | k={current.n_layers:>2} | "
                  f"LL={current_ll:>10.2f} | "
                  f"Accept: B={rates['birth']:.0f}% D={rates['death']:.0f}% "
                  f"M={rates['move']:.0f}% C={rates['change']:.0f}%")

    acceptance = {n: (accepts[n] / counts[n] if counts[n] > 0 else 0)
                  for n in proposal_names}
    if config.verbose:
        print(f"  [Chain {chain_id}] Done — {len(models):,} posterior samples.")

    return {
        "models": models, "log_likes": np.array(log_likes),
        "n_layers": np.array(n_layers_trace),
        "full_ll_trace": np.array(full_ll_trace),
        "acceptance": acceptance, "chain_id": chain_id,
    }


# =============================================================================
#  Parallel runner
# =============================================================================

def run_parallel_rjmcmc(
    frequencies: np.ndarray,
    observed: np.ndarray,
    sigma: np.ndarray,
    prior: Prior,
    config: RjMCMCConfig,
    n_chains: int = 4,
    n_jobs: int = -1,
    base_seed: int = 42,
    use_aniso: bool = False,
    observed_yx: Optional[np.ndarray] = None,
    sigma_yx: Optional[np.ndarray] = None,
    likelihood_mode: str = "rhoa",
    observed_Z: Optional[np.ndarray] = None,
    observed_Z_err: Optional[np.ndarray] = None,
    z_comps: Sequence[str] = ("xy", "yx"),
    observed_Zdet: Optional[np.ndarray] = None,
    Zdet_sigma: Optional[np.ndarray] = None,
    use_pt: bool = False,
    observed_PT: Optional[np.ndarray] = None,
    observed_PT_err: Optional[np.ndarray] = None,
    pt_comps: Sequence[str] = ("xx", "xy", "yx", "yy"),
) -> Dict:
    """Run *n_chains* independent rjMCMC chains via ``joblib`` and merge."""
    from joblib import Parallel, delayed

    lmode = str(likelihood_mode).lower().strip()

    print("=" * 70)
    print(f"  Parallel rjMCMC — {n_chains} chains, "
          f"{'anisotropic' if use_aniso else 'isotropic'} forward model, "
          f"likelihood={lmode}")
    print("=" * 70)

    t0 = time.time()

    chain_config = RjMCMCConfig(
        **{f.name: getattr(config, f.name)
           for f in config.__dataclass_fields__.values()})
    if n_jobs != 1:
        chain_config.verbose = False

    seeds = [base_seed + i for i in range(n_chains)]

    chain_results = Parallel(n_jobs=n_jobs, verbose=10 if n_jobs != 1 else 0)(
        delayed(run_rjmcmc)(
            frequencies, observed, sigma, prior, chain_config,
            initial_model=None, use_aniso=use_aniso,
            observed_yx=observed_yx, sigma_yx=sigma_yx,
            seed=seeds[i], chain_id=i,
            likelihood_mode=lmode,
            observed_Z=observed_Z, observed_Z_err=observed_Z_err,
            z_comps=z_comps,
            observed_Zdet=observed_Zdet, Zdet_sigma=Zdet_sigma,
            use_pt=use_pt, observed_PT=observed_PT,
            observed_PT_err=observed_PT_err, pt_comps=pt_comps,
        ) for i in range(n_chains)
    )

    elapsed = time.time() - t0
    print(f"\nAll {n_chains} chains completed in {elapsed:.1f}s")

    all_models: List[LayeredModel] = []
    all_log_likes, all_n_layers = [], []
    for cr in chain_results:
        all_models.extend(cr["models"])
        all_log_likes.append(cr["log_likes"])
        all_n_layers.append(cr["n_layers"])

    rhat = gelman_rubin([cr["log_likes"] for cr in chain_results])
    avg_acceptance = {
        name: float(np.mean([cr["acceptance"][name] for cr in chain_results]))
        for name in ["birth", "death", "move", "change"]
    }

    tag = "✓ converged" if rhat < 1.1 else "⚠ may not have converged"
    print(f"\nMerged posterior: {len(all_models):,} samples from {n_chains} chains")
    print(f"Gelman-Rubin R-hat (log-likelihood): {rhat:.4f}  {tag}")
    print("Average acceptance rates:")
    for name, rate in avg_acceptance.items():
        print(f"  {name:>8s}: {rate*100:.1f}%")

    return {
        "models": all_models,
        "log_likes": np.concatenate(all_log_likes),
        "n_layers": np.concatenate(all_n_layers),
        "acceptance": avg_acceptance,
        "chains": chain_results,
        "gelman_rubin": rhat,
        "elapsed_s": elapsed,
        "burn_in": config.burn_in,
    }


# =============================================================================
#  Diagnostics
# =============================================================================

def gelman_rubin(chain_traces: List[np.ndarray]) -> float:
    """Gelman–Rubin R-hat for a scalar trace (values near 1.0 = converged)."""
    m = len(chain_traces)
    if m < 2:
        return float("nan")
    n = min(len(c) for c in chain_traces)
    chains = np.array([c[:n] for c in chain_traces])
    B = n * np.var(chains.mean(axis=1), ddof=1)
    W = np.mean(chains.var(axis=1, ddof=1))
    if W == 0:
        return float("nan")
    return float(np.sqrt(((n - 1) / n) * W + B / n) / np.sqrt(W))


# =============================================================================
#  Posterior profile computation
# =============================================================================

def compute_posterior_profile(
    models: List[LayeredModel], depth_grid: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Resistivity statistics on a regular depth grid."""
    nd, nm = len(depth_grid), len(models)
    rho_ens = np.zeros((nm, nd))
    for i, m in enumerate(models):
        depths = np.concatenate(([0.0], m.interfaces, [depth_grid[-1] * 2]))
        rhos = m.get_resistivities()
        for j, z in enumerate(depth_grid):
            rho_ens[i, j] = rhos[min(np.searchsorted(depths[1:], z), len(rhos) - 1)]
    log_ens = np.log10(rho_ens)
    return {
        "depth": depth_grid,
        "mean": 10 ** np.mean(log_ens, axis=0),
        "median": 10 ** np.median(log_ens, axis=0),
        "p05": 10 ** np.percentile(log_ens, 5, axis=0),
        "p95": 10 ** np.percentile(log_ens, 95, axis=0),
        "ensemble": rho_ens,
    }


def compute_posterior_rhomin_profile(
    models: List[LayeredModel], depth_grid: np.ndarray,
) -> Dict[str, np.ndarray]:
    """ρ_min statistics on a regular depth grid (aniso models only)."""
    nd, nm = len(depth_grid), len(models)
    rho_ens = np.zeros((nm, nd))
    for i, m in enumerate(models):
        depths = np.concatenate(([0.0], m.interfaces, [depth_grid[-1] * 2]))
        rhos = m.get_resistivities()
        for j, z in enumerate(depth_grid):
            idx = min(np.searchsorted(depths[1:], z), len(rhos) - 1)
            if m.is_anisotropic:
                rho_ens[i, j] = rhos[idx] / m.aniso_ratios[idx]
            else:
                rho_ens[i, j] = rhos[idx]
    log_ens = np.log10(rho_ens)
    return {
        "depth": depth_grid,
        "mean": 10 ** np.mean(log_ens, axis=0),
        "median": 10 ** np.median(log_ens, axis=0),
        "p05": 10 ** np.percentile(log_ens, 5, axis=0),
        "p95": 10 ** np.percentile(log_ens, 95, axis=0),
        "ensemble": rho_ens,
    }


def compute_posterior_aniso_profile(
    models: List[LayeredModel], depth_grid: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Anisotropy-ratio and strike statistics on a regular depth grid."""
    nd, nm = len(depth_grid), len(models)
    ar_ens = np.ones((nm, nd))
    st_ens = np.zeros((nm, nd))
    for i, m in enumerate(models):
        if not m.is_anisotropic:
            continue
        depths = np.concatenate(([0.0], m.interfaces, [depth_grid[-1] * 2]))
        for j, z in enumerate(depth_grid):
            idx = min(np.searchsorted(depths[1:], z), len(m.aniso_ratios) - 1)
            ar_ens[i, j] = m.aniso_ratios[idx]
            st_ens[i, j] = m.strikes[idx]
    return {
        "depth": depth_grid,
        "aniso_median": np.median(ar_ens, axis=0),
        "aniso_p05": np.percentile(ar_ens, 5, axis=0),
        "aniso_p95": np.percentile(ar_ens, 95, axis=0),
        "strike_median": np.median(st_ens, axis=0),
        "strike_p05": np.percentile(st_ens, 5, axis=0),
        "strike_p95": np.percentile(st_ens, 95, axis=0),
    }


def compute_posterior_histogram(
    models: List[LayeredModel],
    depth_grid: np.ndarray,
    value_bins: np.ndarray,
    prop: str = "rho",
) -> Dict[str, np.ndarray]:
    """2-D histogram of a layer property vs depth across the posterior."""
    nd = len(depth_grid)
    nb = len(value_bins) - 1
    centres = 0.5 * (value_bins[:-1] + value_bins[1:])
    hist = np.zeros((nd, nb), dtype=np.float64)

    for m in models:
        depths = np.concatenate(([0.0], m.interfaces, [depth_grid[-1] * 2]))
        rhos = m.get_resistivities()
        nl = len(rhos)

        for j, z in enumerate(depth_grid):
            layer_idx = min(np.searchsorted(depths[1:], z), nl - 1)

            if prop == "rho":
                val = np.log10(rhos[layer_idx])
            elif prop == "rho_min":
                if m.is_anisotropic:
                    val = np.log10(rhos[layer_idx] / m.aniso_ratios[layer_idx])
                else:
                    val = np.log10(rhos[layer_idx])
            elif prop == "strike":
                if m.is_anisotropic:
                    val = m.strikes[layer_idx]
                else:
                    val = 0.0
            else:
                raise ValueError(f"Unknown property: {prop!r}")

            b = np.searchsorted(value_bins[1:], val)
            if 0 <= b < nb:
                hist[j, b] += 1.0

    mode_idx = np.argmax(hist, axis=1)
    mode_vals = centres[mode_idx]
    if prop in ("rho", "rho_min"):
        mode_vals = 10 ** mode_vals

    return {
        "hist2d": hist,
        "depth": depth_grid,
        "value_bins": value_bins,
        "value_centres": centres,
        "mode": mode_vals,
    }


def compute_changepoint_frequency(
    models: List[LayeredModel],
    depth_grid: np.ndarray,
) -> np.ndarray:
    """Count how often an interface falls near each depth grid point."""
    freq = np.zeros(len(depth_grid), dtype=np.float64)
    for m in models:
        for zi in m.interfaces:
            idx = np.argmin(np.abs(depth_grid - zi))
            freq[idx] += 1.0
    return freq


# =============================================================================
#  I/O helpers
# =============================================================================

def save_results_npz(results: Dict, path: str | Path) -> None:
    """Persist sampler results to a compressed ``.npz`` archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = len(results["models"])
    ifaces = np.empty(n, dtype=object)
    rhos = np.empty(n, dtype=object)
    anisos = np.empty(n, dtype=object)
    strikes_a = np.empty(n, dtype=object)
    for i, m in enumerate(results["models"]):
        ifaces[i] = m.interfaces
        rhos[i] = m.resistivities
        anisos[i] = m.aniso_ratios if m.aniso_ratios is not None else np.array([])
        strikes_a[i] = m.strikes if m.strikes is not None else np.array([])

    sd = {
        "log_likes": results["log_likes"], "n_layers": results["n_layers"],
        "gelman_rubin": np.float64(results.get("gelman_rubin", np.nan)),
        "elapsed_s": np.float64(results.get("elapsed_s", 0.0)),
        "accept_birth": np.float64(results["acceptance"]["birth"]),
        "accept_death": np.float64(results["acceptance"]["death"]),
        "accept_move": np.float64(results["acceptance"]["move"]),
        "accept_change": np.float64(results["acceptance"]["change"]),
        "interfaces": ifaces, "resistivities": rhos,
        "aniso_ratios": anisos, "strikes": strikes_a,
    }
    if "chains" in results:
        for i, cr in enumerate(results["chains"]):
            sd[f"chain_{i}_log_likes"] = cr["log_likes"]
            sd[f"chain_{i}_n_layers"] = cr["n_layers"]
    np.savez_compressed(str(path), **sd)
    print(f"Results saved to {path}")


def load_results_npz(path: str | Path) -> Dict:
    """Load results previously saved by :func:`save_results_npz`."""
    path = Path(path)
    with np.load(str(path), allow_pickle=True) as npz:
        models = []
        for i in range(len(npz["interfaces"])):
            ar = npz["aniso_ratios"][i] if len(npz["aniso_ratios"][i]) > 0 else None
            st = npz["strikes"][i] if len(npz["strikes"][i]) > 0 else None
            models.append(LayeredModel(npz["interfaces"][i],
                                       npz["resistivities"][i], ar, st))
        result = {
            "models": models,
            "log_likes": npz["log_likes"], "n_layers": npz["n_layers"],
            "acceptance": {k: float(npz[f"accept_{k}"])
                           for k in ["birth", "death", "move", "change"]},
            "gelman_rubin": float(npz["gelman_rubin"]),
            "elapsed_s": float(npz["elapsed_s"]),
        }
        chains, i = [], 0
        while f"chain_{i}_log_likes" in npz:
            chains.append({"log_likes": npz[f"chain_{i}_log_likes"],
                           "n_layers": npz[f"chain_{i}_n_layers"], "chain_id": i})
            i += 1
        if chains:
            result["chains"] = chains
    return result


def generate_seed() -> int:
    """Return a random integer seed."""
    return int(np.random.default_rng().integers(0, 2**31))


# =============================================================================
#  Driver helpers (moved from mt_transdim1d.py)
# =============================================================================

def model0_to_layered(m0: dict, use_aniso: bool) -> LayeredModel:
    """Convert a py4mt model dict to a transdim LayeredModel.

    The dict is expected to have ``h_m``, ``sigma_min``, ``sigma_max``,
    ``strike_deg`` (same schema as mt_aniso1d_sampler.py MODEL0).

    Parameters
    ----------
    m0 : dict
        Model dict with keys ``h_m``, ``sigma_min``, ``sigma_max``,
        ``strike_deg``.
    use_aniso : bool
        If True, populate ``aniso_ratios`` and ``strikes``.

    Returns
    -------
    LayeredModel
    """
    h_m = np.asarray(m0["h_m"], dtype=float)
    sigma_max = np.asarray(m0["sigma_max"], dtype=float)
    sigma_min = np.asarray(m0["sigma_min"], dtype=float)

    tiny = np.finfo(float).tiny
    rho_max = 1.0 / np.maximum(sigma_min, tiny)

    mask = h_m > 0
    thicknesses = h_m[mask]
    interfaces = np.cumsum(thicknesses)

    log_rho = np.log10(rho_max)

    if use_aniso:
        rho_min = 1.0 / np.maximum(sigma_max, tiny)
        aniso_ratios = np.maximum(rho_max / rho_min, 1.0)
        strikes = np.asarray(m0["strike_deg"], dtype=float)
        return LayeredModel(interfaces, log_rho, aniso_ratios, strikes)

    return LayeredModel(interfaces, log_rho)


def load_site(
    path: str | Path,
    *,
    noise_level: float = 0.02,
    sigma_floor: float = 0.0,
    err_method: str = "analytic",
    err_nsim: int = 200,
    do_compute_pt: bool = True,
) -> dict:
    """Load a site from an EDI or NPZ file using ``data_proc`` routines.

    Returns a dict with the keys expected by the transdim sampler and
    QC plots:

        frequencies, rho_a, sigma, station,
        rho_a_yx, sigma_yx   (if anisotropic data present)
        Z, Z_err             (if impedance tensor available)
        Zdet, Zdet_err       (determinant impedance, always when Z present)
        rho_a_det, phase_det (ρ_a and phase from Z_det)
        PT, PT_err           (if phase tensor available)

    Parameters
    ----------
    path : str or Path
    noise_level : float
        Default data uncertainty in log10(ρ_a) space.
    sigma_floor : float
        Minimum uncertainty (added as floor).
    err_method : str
        Error method for ``data_proc.compute_rhophas``.
    err_nsim : int
        Monte-Carlo samples for bootstrap errors.
    do_compute_pt : bool
        Compute phase tensor from Z for QC plots.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".edi":
        dd = data_proc.load_edi(str(path))
    elif ext == ".npz":
        dd = data_proc.load_npz(str(path))
    else:
        raise ValueError(f"Unsupported input format: {ext!r} ({path})")

    # ---- Frequencies ------------------------------------------------------
    if "freq" in dd:
        frequencies = np.asarray(dd["freq"], dtype=float).ravel()
    elif "frequencies" in dd:
        frequencies = np.asarray(dd["frequencies"], dtype=float).ravel()
    elif "period" in dd:
        frequencies = 1.0 / np.asarray(dd["period"], dtype=float).ravel()
    else:
        raise KeyError(f"No 'freq', 'frequencies', or 'period' in {path}")

    # ---- Station name -----------------------------------------------------
    station = dd.get("station", path.stem)
    if isinstance(station, np.ndarray):
        station = str(station.item()) if station.ndim == 0 else str(station)

    # ---- Impedance tensor Z -----------------------------------------------
    Z = dd.get("Z")
    Z_err = dd.get("Z_err")
    err_kind = str(dd.get("err_kind", "var")).strip().lower()
    if err_kind.startswith("std"):
        err_kind = "std"
    else:
        err_kind = "var"

    # ---- Compute apparent resistivity / phase from Z if not present -------
    rho = dd.get("rho")
    rho_err = dd.get("rho_err")

    if rho is None and Z is not None:
        Z_arr = np.asarray(Z, dtype=complex)
        _Ze = np.asarray(Z_err) if Z_err is not None else None
        rho, phi, rho_err, phi_err = data_proc.compute_rhophas(
            freq=frequencies, Z=Z_arr, Z_err=_Ze,
            err_kind=err_kind,
            err_method=err_method,
            nsim=err_nsim,
        )

    # ---- Determinant impedance (always from Z when available) -------------
    Zdet = None
    Zdet_err = None
    rho_a_det = None
    phase_det = None
    if Z is not None:
        Z_arr = np.asarray(Z, dtype=np.complex128)
        Zdet = compute_Zdet(Z_arr)
        rho_a_det, phase_det = compute_rhophas_from_Zdet(Zdet, frequencies)
        if Z_err is not None:
            Zdet_err = compute_Zdet_err(Z_arr, np.asarray(Z_err, dtype=float),
                                        nsim=err_nsim)

    # ---- Extract xy (and optionally yx) apparent resistivity ---------------
    rho_a_yx = None
    if rho is not None:
        rho = np.asarray(rho)
        if rho.ndim == 3 and rho.shape[1:] == (2, 2):
            rho_a_xy = rho[:, 0, 1]
            rho_a_yx = rho[:, 1, 0]
        elif rho.ndim == 1:
            rho_a_xy = rho
        else:
            rho_a_xy = rho.ravel()
    elif "rho_a" in dd:
        rho_a_xy = np.asarray(dd["rho_a"], dtype=float).ravel()
        rho_a_yx = dd.get("rho_a_yx")
        if rho_a_yx is not None:
            rho_a_yx = np.asarray(rho_a_yx, dtype=float).ravel()
    elif "rho_a_xy" in dd:
        rho_a_xy = np.asarray(dd["rho_a_xy"], dtype=float).ravel()
        rho_a_yx = dd.get("rho_a_yx")
        if rho_a_yx is not None:
            rho_a_yx = np.asarray(rho_a_yx, dtype=float).ravel()
    elif rho_a_det is not None:
        rho_a_xy = rho_a_det
    else:
        raise KeyError(
            f"Cannot determine apparent resistivity from {path}.  "
            "Need 'Z', 'rho', 'rho_a', or 'rho_a_xy'."
        )

    # ---- Uncertainties in log10(rho_a) space ------------------------------
    sigma_yx = None
    if rho_err is not None:
        rho_err = np.asarray(rho_err)
        if rho_err.ndim == 3 and rho_err.shape[1:] == (2, 2):
            rho_err_xy = rho_err[:, 0, 1]
            rho_err_yx = rho_err[:, 1, 0]
        elif rho_err.ndim == 1:
            rho_err_xy = rho_err
            rho_err_yx = None
        else:
            rho_err_xy = rho_err.ravel()
            rho_err_yx = None

        with np.errstate(divide="ignore", invalid="ignore"):
            sigma_xy = np.where(
                rho_a_xy > 0,
                rho_err_xy / (rho_a_xy * np.log(10)),
                noise_level,
            )
        sigma_xy = np.maximum(sigma_xy, sigma_floor)
        sigma_xy = np.where(np.isfinite(sigma_xy), sigma_xy, noise_level)

        if rho_err_yx is not None and rho_a_yx is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                sigma_yx = np.where(
                    rho_a_yx > 0,
                    rho_err_yx / (rho_a_yx * np.log(10)),
                    noise_level,
                )
            sigma_yx = np.maximum(sigma_yx, sigma_floor)
            sigma_yx = np.where(np.isfinite(sigma_yx), sigma_yx, noise_level)
    elif "sigma" in dd:
        sigma_xy = np.maximum(
            np.asarray(dd["sigma"], dtype=float).ravel(), sigma_floor)
        sigma_yx = dd.get("sigma_yx")
        if sigma_yx is not None:
            sigma_yx = np.maximum(
                np.asarray(sigma_yx, dtype=float).ravel(), sigma_floor)
    else:
        sigma_xy = np.full(len(frequencies), noise_level)

    # ---- Phase tensor (for QC plots) --------------------------------------
    PT = dd.get("P", dd.get("PT"))
    PT_err = dd.get("P_err", dd.get("PT_err"))
    if PT is None and do_compute_pt and Z is not None:
        Z_arr = np.asarray(Z, dtype=complex)
        _Ze = np.asarray(Z_err) if Z_err is not None else None
        PT, PT_err = data_proc.compute_pt(Z_arr, _Ze, err_kind=err_kind)

    # ---- Build output dict ------------------------------------------------
    result: Dict = {
        "station": station,
        "frequencies": frequencies,
        "rho_a": rho_a_xy,
        "sigma": sigma_xy,
    }

    if rho_a_yx is not None:
        result["rho_a_yx"] = rho_a_yx
    if sigma_yx is not None:
        result["sigma_yx"] = sigma_yx
    if Z is not None:
        result["Z"] = np.asarray(Z, dtype=complex)
    if Z_err is not None:
        result["Z_err"] = np.asarray(Z_err, dtype=float)
    if Zdet is not None:
        result["Zdet"] = Zdet
    if Zdet_err is not None:
        result["Zdet_err"] = Zdet_err
    if rho_a_det is not None:
        result["rho_a_det"] = rho_a_det
    if phase_det is not None:
        result["phase_det"] = phase_det
    if PT is not None:
        result["PT"] = np.asarray(PT, dtype=float)
    if PT_err is not None:
        result["PT_err"] = np.asarray(PT_err, dtype=float)

    return result


def build_rjmcmc_summary(
    station: str,
    results: dict,
    depth_max: float,
    qpairs: tuple,
    use_aniso: bool,
) -> dict:
    """Build a summary dict with posterior statistics and quantiles.

    Parameters
    ----------
    station : str
    results : dict — output of ``run_rjmcmc`` or ``run_parallel_rjmcmc``
    depth_max : float — maximum depth for the profile grid [m]
    qpairs : tuple of (float, float) — quantile pairs (percentiles)
    use_aniso : bool

    Returns
    -------
    dict — serializable summary.
    """
    depth_grid = np.linspace(1, depth_max, 500)
    prof = compute_posterior_profile(results["models"], depth_grid)

    summary: Dict = {
        "station": station,
        "depth_grid": depth_grid,
        "rho_mean": prof["mean"],
        "rho_median": prof["median"],
        "rho_p05": prof["p05"],
        "rho_p95": prof["p95"],
        "n_layers_trace": results["n_layers"],
        "n_layers_median": float(np.median(results["n_layers"])),
        "n_layers_mode": int(np.bincount(results["n_layers"]).argmax()),
        "gelman_rubin": results.get("gelman_rubin", np.nan),
        "elapsed_s": results.get("elapsed_s", 0.0),
        "acceptance": results["acceptance"],
    }

    log_ens = np.log10(prof["ensemble"])
    for qlo, qhi in qpairs:
        summary[f"rho_p{qlo:02d}"] = 10 ** np.percentile(log_ens, qlo, axis=0)
        summary[f"rho_p{qhi:02d}"] = 10 ** np.percentile(log_ens, qhi, axis=0)

    if use_aniso:
        aprof = compute_posterior_aniso_profile(results["models"], depth_grid)
        summary.update(aprof)

    return summary
