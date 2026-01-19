#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mcmc.py
=================

`emcee` interface for 1-D **anisotropic** magnetotelluric (MT) inversion.

This module provides a practical bridge between:

- observed MT transfer functions for **one site** (impedance tensor **Z** and
  optionally the **phase tensor** **P**), typically stored in the
  ``dataproc.load_edi`` dictionary layout, and
- the anisotropic 1-D forward model :func:`aniso1d_impedance_sens`.

Anisotropic 1-D (layered, laterally homogeneous) modelling implies the tipper is
identically zero, so tipper data are intentionally not handled here.

Data model
----------

Likelihood is defined for selected transfer-function components.

- **Impedance** Z: complex, Gaussian errors on real and imaginary parts.
- **Phase tensor** P: real, Gaussian errors on tensor entries.

Error handling
--------------

If the input dictionary provides ``Z_err`` and/or ``P_err`` they are used.

- If ``err_kind == 'var'`` (variance), ``sigma = sqrt(var)``.
- If ``err_kind == 'std'`` (standard deviation), it is used as-is.

If ``use_pt=True`` and P is requested but missing, it can be computed from Z:

    P = inv(Re(Z)) @ Im(Z)

for each period (same definition as in ``dataproc.compute_pt``).

Parameterization
----------------

The sampler supports the public parameterization expected by
:func:`aniso1d_impedance_sens`:

- ``h_m``      : (nl,) layer thicknesses in meters (basement thickness is
  ignored by the forward recursion, but may still be stored)
- ``rop``      : (nl, 3) principal resistivities [Ohm·m]
- ``ustr_deg`` : (nl,) Euler strike angles in degrees
- ``udip_deg`` : (nl,) Euler dip angles in degrees
- ``usla_deg`` : (nl,) Euler slant angles in degrees

Internally, emcee samples an unconstrained vector ``theta``. By default:

- thicknesses and resistivities are sampled in log10-space
- angles are sampled directly in degrees

You can fix any subset of parameters by providing boolean masks.

Example
-------

.. code-block:: python

    import numpy as np
    import emcee

    from dataproc import load_edi
    from mcmc import EmceeAnisoMTSampler

    edi = load_edi("SITE001.edi")

    # Initial anisotropic model (nl layers)
    h_m = np.array([200.0, 800.0, 2000.0, 0.0])
    rop = np.array([
        [100.0, 100.0, 100.0],
        [50.0, 200.0, 500.0],
        [20.0, 80.0, 200.0],
        [5.0, 5.0, 5.0],
    ])
    ustr = np.zeros(h_m.size)
    udip = np.zeros(h_m.size)
    usla = np.zeros(h_m.size)

    # If you want to keep thicknesses fixed, add fix_h=True
    inv = EmceeAnisoMTSampler(
        data_site=edi,
        periods_s=1.0 / edi["freq"],
        h_m0=h_m,
        rop0=rop,
        ustr_deg0=ustr,
        udip_deg0=udip,
        usla_deg0=usla,
        use_pt=True,
        z_comps=("xy", "yx"),
        pt_comps=("xx", "xy", "yx", "yy"),
        sigma_floor_Z=1e-9,
        sigma_floor_P=1e-6,
    )

    ndim = inv.ndim
    nwalkers = 2 * ndim
    p0 = inv.initial_ensemble(nwalkers, rel_scale=0.05)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, inv.log_probability)
    sampler.run_mcmc(p0, 2000, progress=True)

Author: Volker Rath (DIAS)
Created with the help of ChatGPT (GPT-5 Thinking) on 2026-01-19 (UTC)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "phase_tensor_from_Z",
    "pack_site_data",
    "pack_forward_prediction",
    "ParamConfig",
    "PriorBounds",
    "EmceeAnisoMTSampler",
]


# -----------------------------------------------------------------------------
# Forward-model import (robust to file naming in this project)
# -----------------------------------------------------------------------------


def _import_forward() -> Any:
    """Import and return :func:`aniso1d_impedance_sens`.

    The project stores the forward model in ``aniso.py`` (this repository).
    This helper tries a small set of module names to stay robust.

    Returns
    -------
    callable
        The function ``aniso1d_impedance_sens``.

    Raises
    ------
    ImportError
        If the forward function cannot be imported.
    """

    tried: List[str] = []

    for mod_name in ("aniso", "aniso_sens", "aniso_sens.py"):
        try:
            mod = __import__(mod_name, fromlist=["aniso1d_impedance_sens"])
            return getattr(mod, "aniso1d_impedance_sens")
        except Exception as exc:  # pragma: no cover
            tried.append(f"{mod_name}: {exc}")

    raise ImportError(
        "Could not import aniso1d_impedance_sens from known modules. Tried: "
        + "; ".join(tried)
    )


_ANISO_FWD = _import_forward()


# -----------------------------------------------------------------------------
# Array shape helpers
# -----------------------------------------------------------------------------


def _as_2x2_stack(Z: np.ndarray) -> np.ndarray:
    """Ensure impedance array has shape ``(n, 2, 2)``.

    Parameters
    ----------
    Z : ndarray
        Impedance array.

    Returns
    -------
    ndarray
        Complex array with shape ``(n, 2, 2)``.

    Raises
    ------
    ValueError
        If the input cannot be interpreted as ``(n, 2, 2)``.
    """

    Z = np.asarray(Z)
    if Z.ndim != 3 or Z.shape[1:] != (2, 2):
        raise ValueError("Z must have shape (n, 2, 2).")
    return Z.astype(np.complex128, copy=False)


def _as_pt_stack(P: np.ndarray) -> np.ndarray:
    """Ensure phase tensor array has shape ``(n, 2, 2)`` and float dtype.

    Parameters
    ----------
    P : ndarray
        Phase tensor array.

    Returns
    -------
    ndarray
        Float array with shape ``(n, 2, 2)``.

    Raises
    ------
    ValueError
        If the input cannot be interpreted as ``(n, 2, 2)``.
    """

    P = np.asarray(P)
    if P.ndim != 3 or P.shape[1:] != (2, 2):
        raise ValueError("P must have shape (n, 2, 2).")
    return P.astype(float, copy=False)


def _comp_to_ij(comp: str) -> Tuple[int, int]:
    """Map a tensor component label to indices.

    Parameters
    ----------
    comp : str
        One of {"xx", "xy", "yx", "yy"} (case-insensitive).

    Returns
    -------
    (int, int)
        Indices (i, j) into ``A[:, i, j]``.

    Raises
    ------
    ValueError
        If ``comp`` is not recognized.
    """

    c = comp.strip().lower()
    mapping = {"xx": (0, 0), "xy": (0, 1), "yx": (1, 0), "yy": (1, 1)}
    if c not in mapping:
        raise ValueError(f"Unknown component {comp!r}; expected one of {sorted(mapping)}")
    return mapping[c]


def _sigma_from_err(err: Optional[np.ndarray], err_kind: str) -> Optional[np.ndarray]:
    """Convert an error array into 1-sigma standard deviations.

    Parameters
    ----------
    err : ndarray or None
        Error array (either variance or standard deviation).
    err_kind : str
        Either "var" or "std".

    Returns
    -------
    ndarray or None
        Standard-deviation array with same shape as ``err``.

    Raises
    ------
    ValueError
        If ``err_kind`` is unknown.
    """

    if err is None:
        return None

    e = np.asarray(err, dtype=float)

    if err_kind == "var":
        with np.errstate(invalid="ignore"):
            return np.sqrt(e)
    if err_kind == "std":
        return e

    raise ValueError("err_kind must be 'var' or 'std'.")


def phase_tensor_from_Z(Z: np.ndarray) -> np.ndarray:
    """Compute the phase tensor P from impedance Z.

    The phase tensor is defined as:

        P = inv(Re(Z)) @ Im(Z)

    for each period. If Re(Z) is singular for a period, a pseudo-inverse is
    used.

    Parameters
    ----------
    Z : ndarray, shape (n, 2, 2)
        Complex impedance tensor.

    Returns
    -------
    ndarray, shape (n, 2, 2)
        Phase tensor (float).
    """

    Z = _as_2x2_stack(Z)
    n = Z.shape[0]
    P = np.full((n, 2, 2), np.nan, dtype=float)

    for k in range(n):
        X = Z[k].real
        Y = Z[k].imag
        try:
            # Solve X @ P = Y
            P[k] = np.linalg.solve(X, Y)
        except np.linalg.LinAlgError:
            P[k] = np.linalg.pinv(X) @ Y

    return P


# -----------------------------------------------------------------------------
# Packing for likelihood evaluation
# -----------------------------------------------------------------------------


@dataclass
class ChannelSpec:
    """One packed data channel (with a period mask).

    Attributes
    ----------
    kind : str
        Either "Z" or "P".
    part : str
        For kind "Z": "re" or "im". For kind "P": "val".
    ij : tuple of int
        Component indices (i, j).
    mask : ndarray, shape (nper,)
        Boolean mask selecting valid periods.
    label : str
        Human-readable channel label.
    """

    kind: str
    part: str
    ij: Tuple[int, int]
    mask: np.ndarray
    label: str


@dataclass
class PackedData:
    """Packed observation vector and channel definitions.

    Attributes
    ----------
    y : ndarray
        Real-valued observation vector.
    sigma : ndarray
        1-sigma standard deviations for each element of ``y``.
    channels : list of ChannelSpec
        Channel definitions used for packing predictions in the same order.
    meta : dict
        Small metadata dictionary about packing choices.
    """

    y: np.ndarray
    sigma: np.ndarray
    channels: List[ChannelSpec]
    meta: Dict[str, Any]


def pack_site_data(
    data_site: Dict[str, Any],
    *,
    z_comps: Sequence[str] = ("xy", "yx"),
    use_pt: bool = False,
    pt_comps: Sequence[str] = ("xx", "xy", "yx", "yy"),
    compute_pt_if_missing: bool = True,
    sigma_floor_Z: float = 0.0,
    sigma_floor_P: float = 0.0,
) -> PackedData:
    """Pack site transfer functions into a real-valued observation vector.

    Parameters
    ----------
    data_site : dict
        Site dictionary, typically from ``dataproc.load_edi``.

        Required keys:
        - ``freq``: (n,) frequencies in Hz
        - ``Z``: (n,2,2) complex

        Optional keys:
        - ``Z_err``: (n,2,2) float (variance or std)
        - ``P``: (n,2,2) float
        - ``P_err``: (n,2,2) float
        - ``err_kind``: "var" or "std" (default: "var")

    z_comps : sequence of str, default ("xy","yx")
        Impedance components to include.
    use_pt : bool, default False
        If True, include phase tensor components.
    pt_comps : sequence of str, default ("xx","xy","yx","yy")
        Phase tensor components to include.
    compute_pt_if_missing : bool, default True
        If True and ``use_pt=True`` but ``P`` is missing, compute P from Z.
    sigma_floor_Z : float, default 0.0
        Minimum sigma applied to impedance components (real and imag).
    sigma_floor_P : float, default 0.0
        Minimum sigma applied to phase tensor entries.

    Returns
    -------
    PackedData
        Contains concatenated vector ``y``, its ``sigma``, and per-channel masks.

    Raises
    ------
    KeyError
        If required fields are missing.
    ValueError
        For shape mismatches or invalid options.
    """

    if "Z" not in data_site or "freq" not in data_site:
        raise KeyError("data_site must contain 'freq' and 'Z'.")

    err_kind = str(data_site.get("err_kind", "var")).lower()
    if err_kind not in ("var", "std"):
        raise ValueError("data_site['err_kind'] must be 'var' or 'std'.")

    Zobs = _as_2x2_stack(data_site["Z"])
    nper = Zobs.shape[0]

    Zsig = _sigma_from_err(data_site.get("Z_err"), err_kind)
    if Zsig is not None and Zsig.shape != Zobs.shape:
        raise ValueError("Z_err must have the same shape as Z.")

    # Phase tensor (optional)
    pt_source = "none"
    Pobs: Optional[np.ndarray] = None
    Psig: Optional[np.ndarray] = None

    if use_pt:
        if data_site.get("P") is not None:
            Pobs = _as_pt_stack(data_site["P"])
            if Pobs.shape[0] != nper:
                raise ValueError("P must have the same number of samples as Z.")
            pt_source = "edi"
        else:
            if not compute_pt_if_missing:
                raise KeyError(
                    "use_pt=True but data_site['P'] is missing and compute_pt_if_missing=False."
                )
            Pobs = phase_tensor_from_Z(Zobs)
            pt_source = "computed_from_Z"

        Psig = _sigma_from_err(data_site.get("P_err"), err_kind)
        if Psig is not None:
            Psig = np.asarray(Psig, dtype=float)
            if Psig.shape != (nper, 2, 2):
                raise ValueError("P_err must have shape (n,2,2) matching P.")

    y_parts: List[np.ndarray] = []
    s_parts: List[np.ndarray] = []
    channels: List[ChannelSpec] = []

    # Impedance channels
    for comp in z_comps:
        i, j = _comp_to_ij(comp)
        z = Zobs[:, i, j]

        if Zsig is None:
            s_base = np.full(nper, float(sigma_floor_Z), dtype=float)
        else:
            s_base = np.maximum(Zsig[:, i, j], float(sigma_floor_Z))

        # real
        y_re = z.real.astype(float)
        s_re = s_base
        mask_re = np.isfinite(y_re) & np.isfinite(s_re) & (s_re > 0.0)
        y_parts.append(y_re[mask_re])
        s_parts.append(s_re[mask_re])
        channels.append(
            ChannelSpec(kind="Z", part="re", ij=(i, j), mask=mask_re, label=f"Z{comp.lower()}_re")
        )

        # imag
        y_im = z.imag.astype(float)
        s_im = s_base
        mask_im = np.isfinite(y_im) & np.isfinite(s_im) & (s_im > 0.0)
        y_parts.append(y_im[mask_im])
        s_parts.append(s_im[mask_im])
        channels.append(
            ChannelSpec(kind="Z", part="im", ij=(i, j), mask=mask_im, label=f"Z{comp.lower()}_im")
        )

    # Phase tensor channels
    if use_pt:
        assert Pobs is not None
        for comp in pt_comps:
            i, j = _comp_to_ij(comp)
            p = Pobs[:, i, j].astype(float)
            if Psig is None:
                s = np.full(nper, float(sigma_floor_P), dtype=float)
            else:
                s = np.maximum(Psig[:, i, j], float(sigma_floor_P))
            mask = np.isfinite(p) & np.isfinite(s) & (s > 0.0)
            y_parts.append(p[mask])
            s_parts.append(s[mask])
            channels.append(ChannelSpec(kind="P", part="val", ij=(i, j), mask=mask, label=f"P{comp.lower()}"))

    y = (
        np.concatenate([np.asarray(v, dtype=float).ravel() for v in y_parts])
        if y_parts
        else np.zeros(0, dtype=float)
    )
    sigma = (
        np.concatenate([np.asarray(v, dtype=float).ravel() for v in s_parts])
        if s_parts
        else np.zeros(0, dtype=float)
    )

    if y.size != sigma.size:
        raise RuntimeError("Internal packing error: y and sigma size mismatch.")

    if y.size == 0:
        raise ValueError(
            "No valid observations packed. Check component selection, NaNs, and errors. "
            "If you did not provide Z_err/P_err, set sigma_floor_Z and/or sigma_floor_P > 0."
        )

    meta = {
        "n_freq": int(nper),
        "z_comps": tuple([c.lower() for c in z_comps]),
        "use_pt": bool(use_pt),
        "pt_comps": tuple([c.lower() for c in pt_comps]) if use_pt else tuple(),
        "pt_source": pt_source,
        "err_kind": err_kind,
    }

    return PackedData(y=y, sigma=sigma, channels=channels, meta=meta)


def pack_forward_prediction(Zpred: np.ndarray, channels: Sequence[ChannelSpec]) -> np.ndarray:
    """Pack predicted transfer functions into the same real-valued order as observations.

    Parameters
    ----------
    Zpred : ndarray, shape (nper, 2, 2)
        Predicted impedance tensor.
    channels : sequence of ChannelSpec
        Channel definitions returned by :func:`pack_site_data`.

    Returns
    -------
    ndarray
        Concatenated prediction vector with exactly the same length/order as
        the packed observation vector.

    Raises
    ------
    ValueError
        If an unknown channel kind is encountered.
    """

    Zpred = _as_2x2_stack(Zpred)
    Ppred: Optional[np.ndarray] = None

    y_parts: List[np.ndarray] = []

    for ch in channels:
        i, j = ch.ij

        if ch.kind == "Z":
            z = Zpred[:, i, j]
            vals = z.real if ch.part == "re" else z.imag
            y_parts.append(np.asarray(vals, dtype=float)[ch.mask])
        elif ch.kind == "P":
            if Ppred is None:
                Ppred = phase_tensor_from_Z(Zpred)
            vals = Ppred[:, i, j]
            y_parts.append(np.asarray(vals, dtype=float)[ch.mask])
        else:
            raise ValueError(f"Unknown channel kind: {ch.kind!r}")

    return (
        np.concatenate([np.asarray(v, dtype=float).ravel() for v in y_parts])
        if y_parts
        else np.zeros(0, dtype=float)
    )


# -----------------------------------------------------------------------------
# Parameter transform and priors
# -----------------------------------------------------------------------------


def _validate_mask(mask: Optional[np.ndarray], shape: Tuple[int, ...], name: str) -> np.ndarray:
    """Validate a boolean mask or create an all-True mask.

    Parameters
    ----------
    mask : ndarray or None
        Mask array.
    shape : tuple
        Required shape.
    name : str
        Name used in error messages.

    Returns
    -------
    ndarray
        Boolean mask of required shape.

    Raises
    ------
    ValueError
        If mask has incompatible shape.
    """

    if mask is None:
        return np.ones(shape, dtype=bool)
    m = np.asarray(mask, dtype=bool)
    if m.shape != shape:
        raise ValueError(f"{name} mask must have shape {shape}, got {m.shape}.")
    return m


@dataclass
class ParamConfig:
    """Configuration for mapping between structured model parameters and theta.

    Notes
    -----
    The default sampling space uses log10 for positive parameters.

    Attributes
    ----------
    sample_h : ndarray, shape (nl,)
        True for layers whose thickness is sampled.
    sample_rop : ndarray, shape (nl,3)
        True for (layer, principal-axis) resistivities sampled.
    sample_angles : ndarray, shape (nl,)
        True for layers whose Euler angles are sampled.
    include_basement_thickness : bool
        If False (default), the last thickness is kept fixed and not part of theta.
    """

    sample_h: np.ndarray
    sample_rop: np.ndarray
    sample_angles: np.ndarray
    include_basement_thickness: bool = False


class ParameterTransform:
    """Pack/unpack between structured model arrays and emcee's theta vector."""

    def __init__(
        self,
        h_m0: np.ndarray,
        rop0: np.ndarray,
        ustr_deg0: np.ndarray,
        udip_deg0: np.ndarray,
        usla_deg0: np.ndarray,
        *,
        config: Optional[ParamConfig] = None,
    ):
        """Initialize the transform.

        Parameters
        ----------
        h_m0, rop0, ustr_deg0, udip_deg0, usla_deg0 : ndarray
            Reference model (used for fixed parameters and for default masks).
        config : ParamConfig, optional
            Sampling configuration. If None, a default config is created that
            samples all parameters except basement thickness.

        Raises
        ------
        ValueError
            If array shapes are inconsistent.
        """

        self.h0 = np.asarray(h_m0, dtype=float).ravel()
        self.rop0 = np.asarray(rop0, dtype=float)
        self.ustr0 = np.asarray(ustr_deg0, dtype=float).ravel()
        self.udip0 = np.asarray(udip_deg0, dtype=float).ravel()
        self.usla0 = np.asarray(usla_deg0, dtype=float).ravel()

        nl = self.h0.size
        if self.rop0.shape != (nl, 3):
            raise ValueError("rop0 must have shape (nl, 3).")
        if not (self.ustr0.shape == self.udip0.shape == self.usla0.shape == (nl,)):
            raise ValueError("Euler angle arrays must all have shape (nl,).")

        if config is None:
            sample_h = np.ones(nl, dtype=bool)
            sample_h[-1] = False  # basement thickness unused
            sample_rop = np.ones((nl, 3), dtype=bool)
            sample_angles = np.ones(nl, dtype=bool)
            config = ParamConfig(sample_h=sample_h, sample_rop=sample_rop, sample_angles=sample_angles)

        self.config = config

        self.sample_h = _validate_mask(config.sample_h, (nl,), "sample_h")
        self.sample_rop = _validate_mask(config.sample_rop, (nl, 3), "sample_rop")
        self.sample_angles = _validate_mask(config.sample_angles, (nl,), "sample_angles")

        if not config.include_basement_thickness:
            self.sample_h = self.sample_h.copy()
            self.sample_h[-1] = False

        # Pre-compute index slices for packing
        self._idx_h = np.where(self.sample_h)[0]
        self._idx_rop = np.argwhere(self.sample_rop)
        self._idx_ang = np.where(self.sample_angles)[0]

    @property
    def ndim(self) -> int:
        """Return the dimension of the theta vector."""

        n_h = int(self._idx_h.size)
        n_rop = int(self._idx_rop.shape[0])
        n_ang = int(self._idx_ang.size)
        return n_h + n_rop + 3 * n_ang

    def pack_theta(
        self,
        h_m: np.ndarray,
        rop: np.ndarray,
        ustr_deg: np.ndarray,
        udip_deg: np.ndarray,
        usla_deg: np.ndarray,
        *,
        log10_positive: bool = True,
    ) -> np.ndarray:
        """Pack structured parameters into theta.

        Parameters
        ----------
        h_m, rop, ustr_deg, udip_deg, usla_deg : ndarray
            Model parameters.
        log10_positive : bool, default True
            If True, thicknesses and resistivities are packed in log10-space.

        Returns
        -------
        ndarray
            1-D theta vector.
        """

        h = np.asarray(h_m, dtype=float).ravel()
        r = np.asarray(rop, dtype=float)
        us = np.asarray(ustr_deg, dtype=float).ravel()
        ud = np.asarray(udip_deg, dtype=float).ravel()
        ul = np.asarray(usla_deg, dtype=float).ravel()

        pieces: List[np.ndarray] = []

        if self._idx_h.size:
            hh = h[self._idx_h]
            pieces.append(np.log10(hh) if log10_positive else hh)

        if self._idx_rop.size:
            rr = np.array([r[i, j] for i, j in self._idx_rop], dtype=float)
            pieces.append(np.log10(rr) if log10_positive else rr)

        if self._idx_ang.size:
            pieces.append(us[self._idx_ang])
            pieces.append(ud[self._idx_ang])
            pieces.append(ul[self._idx_ang])

        if not pieces:
            return np.zeros(0, dtype=float)

        return np.concatenate([p.astype(float).ravel() for p in pieces])

    def unpack_theta(self, theta: np.ndarray, *, log10_positive: bool = True) -> Dict[str, np.ndarray]:
        """Unpack theta into structured arrays.

        Parameters
        ----------
        theta : ndarray
            1-D parameter vector.
        log10_positive : bool, default True
            If True, thicknesses and resistivities are interpreted in log10-space.

        Returns
        -------
        dict
            Keys: ``h_m``, ``rop``, ``ustr_deg``, ``udip_deg``, ``usla_deg``.

        Raises
        ------
        ValueError
            If theta has wrong length.
        """

        th = np.asarray(theta, dtype=float).ravel()
        if th.size != self.ndim:
            raise ValueError(f"theta has length {th.size}, expected {self.ndim}.")

        h = self.h0.copy()
        rop = self.rop0.copy()
        ustr = self.ustr0.copy()
        udip = self.udip0.copy()
        usla = self.usla0.copy()

        p = 0

        # thickness
        if self._idx_h.size:
            n_h = int(self._idx_h.size)
            vals = th[p : p + n_h]
            p += n_h
            h[self._idx_h] = np.power(10.0, vals) if log10_positive else vals

        # resistivities
        if self._idx_rop.size:
            n_r = int(self._idx_rop.shape[0])
            vals = th[p : p + n_r]
            p += n_r
            vals = np.power(10.0, vals) if log10_positive else vals
            for (i, j), v in zip(self._idx_rop, vals):
                rop[int(i), int(j)] = float(v)

        # angles
        if self._idx_ang.size:
            n_a = int(self._idx_ang.size)
            ustr[self._idx_ang] = th[p : p + n_a]
            p += n_a
            udip[self._idx_ang] = th[p : p + n_a]
            p += n_a
            usla[self._idx_ang] = th[p : p + n_a]
            p += n_a

        return {"h_m": h, "rop": rop, "ustr_deg": ustr, "udip_deg": udip, "usla_deg": usla}


@dataclass
class PriorBounds:
    """Simple box priors for physical parameters."""

    # thickness (meters)
    h_min_m: float = 1e-1
    h_max_m: float = 1e6

    # resistivities (ohm m)
    rho_min: float = 1e-3
    rho_max: float = 1e6

    # angles (degrees)
    strike_min_deg: float = -360.0
    strike_max_deg: float = 360.0
    dip_min_deg: float = -90.0
    dip_max_deg: float = 90.0
    slant_min_deg: float = -180.0
    slant_max_deg: float = 180.0


def _log_box(x: np.ndarray, lo: float, hi: float) -> float:
    """Return log prior for independent box constraints.

    Parameters
    ----------
    x : ndarray
        Values.
    lo, hi : float
        Lower and upper bounds.

    Returns
    -------
    float
        0 if all within bounds else -inf.
    """

    if np.any(~np.isfinite(x)):
        return -np.inf
    if np.any(x < lo) or np.any(x > hi):
        return -np.inf
    return 0.0


# -----------------------------------------------------------------------------
# emcee wrapper
# -----------------------------------------------------------------------------


class EmceeAnisoMTSampler:
    """Build an emcee-compatible log-probability for anisotropic 1-D MT.

    Parameters
    ----------
    data_site : dict
        Observed site data (see :func:`pack_site_data`).
    periods_s : ndarray
        Periods in seconds used for forward modelling. Usually
        ``periods_s = 1.0 / data_site['freq']``.
    h_m0, rop0, ustr_deg0, udip_deg0, usla_deg0 : ndarray
        Initial / reference model.
    z_comps : sequence of str
        Impedance components to include.
    use_pt : bool, default False
        If True, include phase tensor components in the likelihood.
    pt_comps : sequence of str
        Phase tensor components to include.
    compute_pt_if_missing : bool, default True
        If True and P missing in the site dictionary, compute it from Z.
    sigma_floor_Z, sigma_floor_P : float
        Minimum sigma values (applied if errors are absent or too small).
    prior_bounds : PriorBounds, optional
        Simple uniform bounds for physical parameters.
    param_config : ParamConfig, optional
        Which parameters to sample vs keep fixed.
    fix_h : bool, default False
        Convenience option: if True and ``param_config`` is None, layer thicknesses
        are treated as fixed (not sampled).
    h_m_fixed : ndarray, optional
        If provided, these thicknesses are used as fixed reference thicknesses
        (requires ``fix_h=True`` unless you also provide an explicit
        ``param_config``).

    Notes
    -----
    - emcee operates on *theta*; this class provides ``log_probability(theta)``.
    - You can access the structured model for any theta using
      :meth:`unpack_theta`.
    """

    def __init__(
        self,
        *,
        data_site: Dict[str, Any],
        periods_s: np.ndarray,
        h_m0: np.ndarray,
        rop0: np.ndarray,
        ustr_deg0: np.ndarray,
        udip_deg0: np.ndarray,
        usla_deg0: np.ndarray,
        z_comps: Sequence[str] = ("xy", "yx"),
        use_pt: bool = False,
        pt_comps: Sequence[str] = ("xx", "xy", "yx", "yy"),
        compute_pt_if_missing: bool = True,
        sigma_floor_Z: float = 0.0,
        sigma_floor_P: float = 0.0,
        prior_bounds: Optional[PriorBounds] = None,
        param_config: Optional[ParamConfig] = None,
        fix_h: bool = False,
        h_m_fixed: Optional[np.ndarray] = None,
        log10_positive: bool = True,
    ):
        self.data_site = data_site
        self.periods_s = np.asarray(periods_s, dtype=float).ravel()

        if self.periods_s.size == 0 or np.any(self.periods_s <= 0.0):
            raise ValueError("periods_s must be a non-empty array of positive values.")

        self.z_comps = tuple([c.lower() for c in z_comps])
        self.use_pt = bool(use_pt)
        self.pt_comps = tuple([c.lower() for c in pt_comps])
        self.compute_pt_if_missing = bool(compute_pt_if_missing)
        self.log10_positive = bool(log10_positive)

        # Thickness handling
        # ------------------
        # In anisotropic 1-D inversion it is common to keep the layer thicknesses
        # fixed (e.g., from independent constraints) and invert only for
        # resistivities + Euler angles. This is provided as a convenience
        # option. For fine control, pass an explicit ParamConfig instead.
        self.fix_h = bool(fix_h)
        if h_m_fixed is not None and not self.fix_h and param_config is None:
            raise ValueError('h_m_fixed was provided but fix_h=False and param_config=None. '
                             'Either set fix_h=True or provide an explicit ParamConfig.')

        h_ref = np.asarray(h_m_fixed if h_m_fixed is not None else h_m0, dtype=float).ravel()

        if self.fix_h and param_config is None:
            nl = h_ref.size
            sample_h = np.zeros(nl, dtype=bool)
            sample_rop = np.ones((nl, 3), dtype=bool)
            sample_angles = np.ones(nl, dtype=bool)
            param_config = ParamConfig(
                sample_h=sample_h,
                sample_rop=sample_rop,
                sample_angles=sample_angles,
                include_basement_thickness=False,
            )

        self.packed = pack_site_data(
            data_site,
            z_comps=self.z_comps,
            use_pt=self.use_pt,
            pt_comps=self.pt_comps,
            compute_pt_if_missing=self.compute_pt_if_missing,
            sigma_floor_Z=float(sigma_floor_Z),
            sigma_floor_P=float(sigma_floor_P),
        )

        self.transform = ParameterTransform(
            h_m0=h_ref,
            rop0=rop0,
            ustr_deg0=ustr_deg0,
            udip_deg0=udip_deg0,
            usla_deg0=usla_deg0,
            config=param_config,
        )

        self.prior_bounds = prior_bounds if prior_bounds is not None else PriorBounds()

    @property
    def ndim(self) -> int:
        """Dimension of the theta vector."""

        return self.transform.ndim

    def unpack_theta(self, theta: np.ndarray) -> Dict[str, np.ndarray]:
        """Unpack theta into structured model arrays."""

        return self.transform.unpack_theta(theta, log10_positive=self.log10_positive)

    def log_prior(self, theta: np.ndarray) -> float:
        """Evaluate the log prior for theta.

        The prior is a product of uniform (box) constraints in the
        **physical** parameter space.

        Parameters
        ----------
        theta : ndarray
            emcee parameter vector.

        Returns
        -------
        float
            Log prior value (0 or -inf).
        """

        m = self.unpack_theta(theta)
        pb = self.prior_bounds

        lp = 0.0
        lp += _log_box(m["h_m"][self.transform.sample_h], pb.h_min_m, pb.h_max_m)

        rop = m["rop"]
        if np.any(self.transform.sample_rop):
            vals = rop[self.transform.sample_rop]
            lp += _log_box(vals, pb.rho_min, pb.rho_max)

        ang_idx = self.transform.sample_angles
        if np.any(ang_idx):
            lp += _log_box(m["ustr_deg"][ang_idx], pb.strike_min_deg, pb.strike_max_deg)
            lp += _log_box(m["udip_deg"][ang_idx], pb.dip_min_deg, pb.dip_max_deg)
            lp += _log_box(m["usla_deg"][ang_idx], pb.slant_min_deg, pb.slant_max_deg)

        return float(lp)

    def _forward(self, theta: np.ndarray) -> np.ndarray:
        """Compute forward impedance prediction for a theta.

        Parameters
        ----------
        theta : ndarray
            Parameter vector.

        Returns
        -------
        ndarray
            Predicted impedance ``Zpred`` with shape ``(nper, 2, 2)``.
        """

        m = self.unpack_theta(theta)

        res = _ANISO_FWD(
            periods_s=self.periods_s,
            h_m=m["h_m"],
            rop=m["rop"],
            ustr_deg=m["ustr_deg"],
            udip_deg=m["udip_deg"],
            usla_deg=m["usla_deg"],
            compute_sens=False,
        )

        return np.asarray(res.Z, dtype=np.complex128)

    def log_likelihood(self, theta: np.ndarray) -> float:
        """Evaluate Gaussian log-likelihood for the packed observation vector.

        Parameters
        ----------
        theta : ndarray
            Parameter vector.

        Returns
        -------
        float
            Log likelihood.
        """

        Zpred = self._forward(theta)
        yhat = pack_forward_prediction(Zpred, self.packed.channels)

        y = self.packed.y
        sigma = self.packed.sigma

        if yhat.shape != y.shape:
            raise ValueError(
                "Prediction vector shape mismatch; check periods and component selection (or NaN masks)."
            )

        r = (y - yhat) / sigma

        # Standard Gaussian log-likelihood
        ll = -0.5 * np.sum(r * r + np.log(2.0 * np.pi * sigma * sigma))
        return float(ll)

    def log_probability(self, theta: np.ndarray) -> float:
        """Evaluate log posterior = log prior + log likelihood."""

        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)

    def initial_ensemble(
        self,
        nwalkers: int,
        *,
        rel_scale: float = 0.05,
        abs_angle_deg: float = 1.0,
        random_state: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Create a simple initial ensemble around the reference model.

        Parameters
        ----------
        nwalkers : int
            Number of walkers.
        rel_scale : float, default 0.05
            Relative perturbation for log10-positive parameters.
        abs_angle_deg : float, default 1.0
            Absolute perturbation for angles in degrees.
        random_state : numpy.random.Generator, optional
            Random generator.

        Returns
        -------
        ndarray, shape (nwalkers, ndim)
            Initial walker positions.

        Notes
        -----
        This is intentionally simple; you may want to tailor it to your
        problem (e.g. bounds-aware initialization).
        """

        if nwalkers < 2:
            raise ValueError("nwalkers must be >= 2.")

        rng = np.random.default_rng() if random_state is None else random_state

        theta0 = self.transform.pack_theta(
            self.transform.h0,
            self.transform.rop0,
            self.transform.ustr0,
            self.transform.udip0,
            self.transform.usla0,
            log10_positive=self.log10_positive,
        )

        if theta0.size != self.ndim:
            raise RuntimeError("Internal error: theta0 has wrong size.")

        p0 = np.tile(theta0, (nwalkers, 1)).astype(float)

        # Apply perturbations only to free dimensions.
        if self.ndim:
            n_h = int(self.transform._idx_h.size)
            n_r = int(self.transform._idx_rop.shape[0])
            n_a = int(self.transform._idx_ang.size)

            n_pos = n_h + n_r

            if n_pos > 0:
                p0[:, :n_pos] += rng.normal(scale=float(rel_scale), size=(nwalkers, n_pos))

            if n_a > 0:
                for b in range(3):
                    start = n_pos + b * n_a
                    stop = start + n_a
                    p0[:, start:stop] += rng.normal(scale=float(abs_angle_deg), size=(nwalkers, n_a))

        return p0
