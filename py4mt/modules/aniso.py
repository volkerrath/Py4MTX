#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""aniso.py
================

Sensitivity-enabled 1-D anisotropic MT forward modelling (layered Earth).

This module computes the 2×2 surface impedance tensor **Z** for a stack of
horizontally layered, electrically anisotropic media.

Two model parameterizations are supported:

A) **Simplified (recommended for inversion):**

   - ``h_m``          (nl,) layer thicknesses in meters (last entry is basement; ignored)
   - ``rho_max_ohmm`` (nl,) *maximum* horizontal resistivity [Ohm·m]
   - ``rho_min_ohmm`` (nl,) *minimum* horizontal resistivity [Ohm·m]
   - ``strike_deg``   (nl,) anisotropy strike in degrees

   This maps to the legacy conductivity parameters
   ``AL = 1/rho_min`` (maximum conductivity), ``AT = 1/rho_max`` (minimum
   conductivity), and ``BLT = strike``.

B) **Full (legacy):**

   - ``h_m``        (nl,)
   - ``rop``        (nl, 3) principal resistivities [Ohm·m]
   - ``ustr_deg``   (nl,) strike
   - ``udip_deg``   (nl,) dip
   - ``usla_deg``   (nl,) slant

   Converted internally via :func:`cpanis`.

Sensitivities
-------------

If ``compute_sens=True`` the forward functions also return sensitivities
(derivatives) of **Z** with respect to the chosen parameters.

- Analytic derivatives are used for ``AL, AT, BLT``.
- Thickness derivatives ``∂Z/∂h`` use a centered finite difference (stable).
- For the *full* parameterization, the chain rule is applied using finite
  differences of :func:`cpanis`.

Return format
-------------

All public forward functions return a **dict** (not a dataclass) to keep the
API simple for script-style use.

Author: Volker Rath (DIAS)
Created with the help of ChatGPT (GPT-5 Thinking) on 2026-02-08 (UTC)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# --- Constants ----------------------------------------------------------------

_MU0: float = 4.0e-7 * np.pi


# --- Small helpers ------------------------------------------------------------

def dfp(x: np.ndarray | complex | float) -> np.ndarray | complex | float:
    """Regularized hyperbolic-cosine-like function: ``dfp(x) = 1 + exp(-2x)``.

    Parameters
    ----------
    x : array_like or scalar
        Argument.

    Returns
    -------
    array_like or scalar
        ``1 + exp(-2x)`` with broadcasting applied by NumPy.
    """
    return 1.0 + np.exp(-2.0 * x)


def dfm(x: np.ndarray | complex | float) -> np.ndarray | complex | float:
    """Regularized hyperbolic-sine-like function: ``dfm(x) = 1 - exp(-2x)``.

    Parameters
    ----------
    x : array_like or scalar
        Argument.

    Returns
    -------
    array_like or scalar
        ``1 - exp(-2x)`` with broadcasting applied by NumPy.
    """
    return 1.0 - np.exp(-2.0 * x)


def rotz(za: np.ndarray, betrad: float) -> np.ndarray:
    """Rotate a 2×2 impedance-like matrix by an angle (radians).

    Notes
    -----
    The rotation is expressed with ``2*beta`` inside the trigonometric factors,
    which is the convention used by many legacy MT tensor rotation formulas.

    Parameters
    ----------
    za : ndarray, shape (2, 2)
        Input complex matrix.
    betrad : float
        Rotation angle in radians.

    Returns
    -------
    ndarray, shape (2, 2)
        Rotated complex matrix.

    Raises
    ------
    ValueError
        If ``za`` does not have shape (2, 2).
    """
    za = np.asarray(za, dtype=np.complex128)
    if za.shape != (2, 2):
        raise ValueError("za must have shape (2,2).")

    co2 = np.cos(2.0 * betrad)
    si2 = np.sin(2.0 * betrad)

    sum1 = za[0, 0] + za[1, 1]
    sum2 = za[0, 1] + za[1, 0]
    dif1 = za[0, 0] - za[1, 1]
    dif2 = za[0, 1] - za[1, 0]

    zb = np.empty((2, 2), dtype=np.complex128)
    zb[0, 0] = 0.5 * (sum1 + dif1 * co2 + sum2 * si2)
    zb[0, 1] = 0.5 * (dif2 + sum2 * co2 - dif1 * si2)
    zb[1, 0] = 0.5 * (-dif2 + sum2 * co2 - dif1 * si2)
    zb[1, 1] = 0.5 * (sum1 - dif1 * co2 - sum2 * si2)
    return zb


def rotz_stack(dza: np.ndarray, betrad: float) -> np.ndarray:
    """Rotate a stack of 2×2 matrices along the leading axis.

    Parameters
    ----------
    dza : ndarray, shape (..., 2, 2)
        Stack of complex matrices.
    betrad : float
        Rotation angle in radians.

    Returns
    -------
    ndarray, shape (..., 2, 2)
        Rotated stack (new array).

    Raises
    ------
    ValueError
        If ``dza`` does not have trailing shape (2, 2).
    """
    dza = np.asarray(dza, dtype=np.complex128)
    if dza.shape[-2:] != (2, 2):
        raise ValueError("dza must have trailing shape (2,2).")

    out = np.empty_like(dza)
    for idx in np.ndindex(dza.shape[:-2]):
        out[idx] = rotz(dza[idx], betrad)
    return out


# --- Effective-parameter conversion -------------------------------------------

def cpanis(
    rop_ohmm: np.ndarray,
    ustr_deg: np.ndarray,
    udip_deg: np.ndarray,
    usla_deg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute conductivity tensors and effective horizontal anisotropy parameters.

    Parameters
    ----------
    rop_ohmm : ndarray, shape (nl, 3)
        Principal resistivities (Ohm·m). Each row is (rho1, rho2, rho3).
    ustr_deg, udip_deg, usla_deg : ndarray, shape (nl,)
        Euler angles (degrees): strike, dip, slant.

    Returns
    -------
    sg : ndarray, shape (nl, 3, 3)
        Full 3×3 conductivity tensor (S/m) in global coordinates for each layer.
    al : ndarray, shape (nl,)
        Maximum effective horizontal conductivity (S/m).
    at : ndarray, shape (nl,)
        Minimum effective horizontal conductivity (S/m).
    blt_rad : ndarray, shape (nl,)
        Effective horizontal anisotropy strike (radians).

    Notes
    -----
    This is a direct, pragmatic translation of a legacy formulation.

    The simplified parameterization used by :func:`aniso1d_impedance_sens_simple`
    avoids this conversion entirely.
    """
    rop_ohmm = np.asarray(rop_ohmm, dtype=float)
    ustr_deg = np.asarray(ustr_deg, dtype=float)
    udip_deg = np.asarray(udip_deg, dtype=float)
    usla_deg = np.asarray(usla_deg, dtype=float)

    if rop_ohmm.ndim != 2 or rop_ohmm.shape[1] != 3:
        raise ValueError("rop_ohmm must have shape (nl, 3).")
    nl = rop_ohmm.shape[0]
    if not (ustr_deg.shape == udip_deg.shape == usla_deg.shape == (nl,)):
        raise ValueError("ustr_deg, udip_deg, usla_deg must all have shape (nl,).")

    sg = np.zeros((nl, 3, 3), dtype=float)
    al = np.zeros(nl, dtype=float)
    at = np.zeros(nl, dtype=float)
    blt = np.zeros(nl, dtype=float)

    tiny = np.finfo(float).tiny

    for k in range(nl):
        sgp1 = 1.0 / float(rop_ohmm[k, 0])
        sgp2 = 1.0 / float(rop_ohmm[k, 1])
        sgp3 = 1.0 / float(rop_ohmm[k, 2])

        rstr = np.deg2rad(float(ustr_deg[k]))
        rdip = np.deg2rad(float(udip_deg[k]))
        rsla = np.deg2rad(float(usla_deg[k]))

        sps, cps = np.sin(rstr), np.cos(rstr)
        sth, cth = np.sin(rdip), np.cos(rdip)
        sfi, cfi = np.sin(rsla), np.cos(rsla)

        pom1 = sgp1 * cfi * cfi + sgp2 * sfi * sfi
        pom2 = sgp1 * sfi * sfi + sgp2 * cfi * cfi
        pom3 = (sgp1 - sgp2) * sfi * cfi

        c2ps, s2ps = cps * cps, sps * sps
        c2th, s2th = cth * cth, sth * sth
        csps = cps * sps
        csth = cth * sth

        sg[k, 0, 0] = (
            pom1 * c2ps
            + pom2 * s2ps * c2th
            - 2.0 * pom3 * cth * csps
            + sgp3 * s2th * s2ps
        )
        sg[k, 0, 1] = (
            pom1 * csps
            - pom2 * c2th * csps
            + pom3 * cth * (c2ps - s2ps)
            - sgp3 * s2th * csps
        )
        sg[k, 0, 2] = -pom2 * csth * sps + pom3 * sth * cps + sgp3 * csth * sps
        sg[k, 1, 0] = sg[k, 0, 1]
        sg[k, 1, 1] = (
            pom1 * s2ps
            + pom2 * c2ps * c2th
            + 2.0 * pom3 * cth * csps
            + sgp3 * s2th * c2ps
        )
        sg[k, 1, 2] = pom2 * csth * cps + pom3 * sth * sps - sgp3 * csth * cps
        sg[k, 2, 0] = sg[k, 0, 2]
        sg[k, 2, 1] = sg[k, 1, 2]
        sg[k, 2, 2] = pom2 * s2th + sgp3 * c2th

        denom = sg[k, 2, 2]
        if abs(denom) < tiny:
            denom = tiny

        axx = sg[k, 0, 0] - sg[k, 0, 2] * sg[k, 2, 0] / denom
        axy = sg[k, 0, 1] - sg[k, 0, 2] * sg[k, 2, 1] / denom
        ayx = sg[k, 1, 0] - sg[k, 2, 0] * sg[k, 1, 2] / denom
        ayy = sg[k, 1, 1] - sg[k, 1, 2] * sg[k, 2, 1] / denom

        da = np.sqrt((axx - ayy) * (axx - ayy) + 4.0 * axy * ayx)
        al[k] = 0.5 * (axx + ayy + da)
        at[k] = 0.5 * (axx + ayy - da)

        if da >= tiny:
            cos2 = (axx - ayy) / da
            cos2 = np.clip(cos2, -1.0, 1.0)
            blt_val = 0.5 * np.arccos(cos2)
        else:
            blt_val = 0.0

        if axy < 0.0:
            blt_val = -blt_val
        blt[k] = blt_val

    return sg, al, at, blt


def alat_from_rho_strike(
    rho_max_ohmm: np.ndarray,
    rho_min_ohmm: np.ndarray,
    strike_deg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert (rho_max, rho_min, strike) to (AL, AT, BLT).

    Parameters
    ----------
    rho_max_ohmm : ndarray, shape (nl,)
        Maximum horizontal resistivity (Ohm·m).
    rho_min_ohmm : ndarray, shape (nl,)
        Minimum horizontal resistivity (Ohm·m).
    strike_deg : ndarray, shape (nl,)
        Strike angle (degrees).

    Returns
    -------
    al : ndarray, shape (nl,)
        Maximum horizontal conductivity (S/m), i.e. ``1/rho_min``.
    at : ndarray, shape (nl,)
        Minimum horizontal conductivity (S/m), i.e. ``1/rho_max``.
    blt_rad : ndarray, shape (nl,)
        Strike in radians.

    Notes
    -----
    - If any resistivity is non-positive, a ValueError is raised.
    - No ordering is enforced. If you want ``rho_max >= rho_min`` you must
      ensure that before calling.
    """
    rho_max_ohmm = np.asarray(rho_max_ohmm, dtype=float).ravel()
    rho_min_ohmm = np.asarray(rho_min_ohmm, dtype=float).ravel()
    strike_deg = np.asarray(strike_deg, dtype=float).ravel()

    if not (rho_max_ohmm.shape == rho_min_ohmm.shape == strike_deg.shape):
        raise ValueError("rho_max_ohmm, rho_min_ohmm, strike_deg must have the same shape.")
    if np.any(rho_max_ohmm <= 0.0) or np.any(rho_min_ohmm <= 0.0):
        raise ValueError("All resistivities must be > 0.")

    al = 1.0 / rho_min_ohmm
    at = 1.0 / rho_max_ohmm
    blt_rad = np.deg2rad(strike_deg)
    return al, at, blt_rad


# --- Core recursion (Z and sensitivities) -------------------------------------

def _propagate_impedance(
    zbot: np.ndarray,
    dz1: complex,
    dz2: complex,
    ag1: complex,
    ag2: complex,
) -> np.ndarray:
    """Compute the top impedance of a layer given the bottom impedance.

    Parameters
    ----------
    zbot : ndarray, shape (2, 2)
        Bottom impedance in the *current layer strike-aligned frame*.
    dz1, dz2 : complex
        Basement-like impedances of the current layer eigenmodes.
    ag1, ag2 : complex
        Propagation terms (k*h) of the current layer eigenmodes.

    Returns
    -------
    ndarray, shape (2, 2)
        Top impedance for the layer, in the same frame as ``zbot``.
    """
    zbot = np.asarray(zbot, dtype=np.complex128)
    dtzbot = zbot[0, 0] * zbot[1, 1] - zbot[0, 1] * zbot[1, 0]

    denom = (
        dtzbot * dfm(ag1) * dfm(ag2) / (dz1 * dz2)
        + zbot[0, 1] * dfm(ag1) * dfp(ag2) / dz1
        - zbot[1, 0] * dfp(ag1) * dfm(ag2) / dz2
        + dfp(ag1) * dfp(ag2)
    )

    ztop = np.zeros((2, 2), dtype=np.complex128)
    ztop[0, 0] = 4.0 * zbot[0, 0] * np.exp(-ag1 - ag2) / denom
    ztop[1, 1] = 4.0 * zbot[1, 1] * np.exp(-ag1 - ag2) / denom
    ztop[0, 1] = (
        zbot[0, 1] * dfp(ag1) * dfp(ag2)
        - zbot[1, 0] * dfm(ag1) * dfm(ag2) * dz1 / dz2
        + dtzbot * dfp(ag1) * dfm(ag2) / dz2
        + dfm(ag1) * dfp(ag2) * dz1
    ) / denom
    ztop[1, 0] = (
        zbot[1, 0] * dfp(ag1) * dfp(ag2)
        - zbot[0, 1] * dfm(ag1) * dfm(ag2) * dz2 / dz1
        - dtzbot * dfm(ag1) * dfp(ag2) / dz1
        - dfp(ag1) * dfm(ag2) * dz2
    ) / denom
    return ztop


def _propagate_sens_from_zbot(
    dzbot: np.ndarray,
    zbot: np.ndarray,
    ztop: np.ndarray,
    dz1: complex,
    dz2: complex,
    ag1: complex,
    ag2: complex,
) -> np.ndarray:
    """Propagate sensitivity from bottom impedance to top impedance.

    Parameters
    ----------
    dzbot : ndarray, shape (2, 2)
        Sensitivity of the bottom impedance with respect to some parameter.
    zbot : ndarray, shape (2, 2)
        Bottom impedance.
    ztop : ndarray, shape (2, 2)
        Top impedance.
    dz1, dz2, ag1, ag2 : complex
        Layer quantities (see :func:`_propagate_impedance`).

    Returns
    -------
    ndarray, shape (2, 2)
        Sensitivity of the top impedance.
    """
    dzbot = np.asarray(dzbot, dtype=np.complex128)
    zbot = np.asarray(zbot, dtype=np.complex128)
    ztop = np.asarray(ztop, dtype=np.complex128)

    dtzbot = zbot[0, 0] * zbot[1, 1] - zbot[0, 1] * zbot[1, 0]
    zdenom = (
        dtzbot * dfm(ag1) * dfm(ag2) / (dz1 * dz2)
        + zbot[0, 1] * dfm(ag1) * dfp(ag2) / dz1
        - zbot[1, 0] * dfp(ag1) * dfm(ag2) / dz2
        + dfp(ag1) * dfp(ag2)
    )

    ddtzbot = (
        dzbot[0, 0] * zbot[1, 1]
        + zbot[0, 0] * dzbot[1, 1]
        - dzbot[0, 1] * zbot[1, 0]
        - zbot[0, 1] * dzbot[1, 0]
    )
    dzdenom = (
        ddtzbot * dfm(ag1) * dfm(ag2) / (dz1 * dz2)
        + dzbot[0, 1] * dfm(ag1) * dfp(ag2) / dz1
        - dzbot[1, 0] * dfp(ag1) * dfm(ag2) / dz2
    )

    exp_term = np.exp(-ag1 - ag2)
    dn11 = 4.0 * dzbot[0, 0] * exp_term
    dn12 = (
        dzbot[0, 1] * dfp(ag1) * dfp(ag2)
        - dzbot[1, 0] * dfm(ag1) * dfm(ag2) * dz1 / dz2
        + ddtzbot * dfp(ag1) * dfm(ag2) / dz2
    )
    dn21 = (
        dzbot[1, 0] * dfp(ag1) * dfp(ag2)
        - dzbot[0, 1] * dfm(ag1) * dfm(ag2) * dz2 / dz1
        - ddtzbot * dfm(ag1) * dfp(ag2) / dz1
    )
    dn22 = 4.0 * dzbot[1, 1] * exp_term

    dztop = np.empty((2, 2), dtype=np.complex128)
    dztop[0, 0] = (dn11 - ztop[0, 0] * dzdenom) / zdenom
    dztop[0, 1] = (dn12 - ztop[0, 1] * dzdenom) / zdenom
    dztop[1, 0] = (dn21 - ztop[1, 0] * dzdenom) / zdenom
    dztop[1, 1] = (dn22 - ztop[1, 1] * dzdenom) / zdenom
    return dztop


def _dZ_dal_layer(
    zbot: np.ndarray,
    ztop: np.ndarray,
    a1: float,
    dz1: complex,
    dz2: complex,
    ag1: complex,
    ag2: complex,
) -> np.ndarray:
    """Analytic derivative ``∂Z_top/∂AL`` for the *current* layer.

    Parameters
    ----------
    zbot, ztop : ndarray, shape (2, 2)
        Bottom and top impedances (strike-aligned frame).
    a1 : float
        Current layer AL (maximum horizontal conductivity).
    dz1, dz2, ag1, ag2 : complex
        Layer quantities.

    Returns
    -------
    ndarray, shape (2, 2)
        Complex derivative of the top impedance with respect to AL.
    """
    dtzbot = zbot[0, 0] * zbot[1, 1] - zbot[0, 1] * zbot[1, 0]
    zdenom = (
        dtzbot * dfm(ag1) * dfm(ag2) / (dz1 * dz2)
        + zbot[0, 1] * dfm(ag1) * dfp(ag2) / dz1
        - zbot[1, 0] * dfp(ag1) * dfm(ag2) / dz2
        + dfp(ag1) * dfp(ag2)
    )

    dapom = dtzbot * dfm(ag2) / dz2 + zbot[0, 1] * dfp(ag2)
    dbpom = dfp(ag2) - zbot[1, 0] * dfm(ag2) / dz2
    dcpom = zbot[1, 0] * dfp(ag2) - dz2 * dfm(ag2)
    ddpom = dtzbot * dfp(ag2) + dz2 * zbot[0, 1] * dfm(ag2)

    depom = ag1 * dfm(ag1)
    dfpom = (dfm(ag1) + ag1 * dfp(ag1)) / dz1
    dgpom = (dfm(ag1) - ag1 * dfp(ag1)) * dz1

    dzdenom = dfpom * dapom + depom * dbpom
    dn12 = depom * dapom - dgpom * dbpom
    dn21 = depom * dcpom - dfpom * ddpom

    a1_safe = a1 if a1 != 0.0 else np.finfo(float).tiny
    zdenom_safe = zdenom if zdenom != 0 else (np.finfo(float).tiny + 0j)

    dztop = np.empty((2, 2), dtype=np.complex128)
    dztop[0, 0] = -0.5 * ztop[0, 0] * dzdenom / (a1_safe * zdenom_safe)
    dztop[0, 1] = 0.5 * (dn12 - ztop[0, 1] * dzdenom) / (a1_safe * zdenom_safe)
    dztop[1, 0] = 0.5 * (dn21 - ztop[1, 0] * dzdenom) / (a1_safe * zdenom_safe)
    dztop[1, 1] = -0.5 * ztop[1, 1] * dzdenom / (a1_safe * zdenom_safe)
    return dztop


def _dZ_dat_layer(
    zbot: np.ndarray,
    ztop: np.ndarray,
    a2: float,
    dz1: complex,
    dz2: complex,
    ag1: complex,
    ag2: complex,
) -> np.ndarray:
    """Analytic derivative ``∂Z_top/∂AT`` for the *current* layer.

    Parameters
    ----------
    zbot, ztop : ndarray, shape (2, 2)
        Bottom and top impedances.
    a2 : float
        Current layer AT (minimum horizontal conductivity).
    dz1, dz2, ag1, ag2 : complex
        Layer quantities.

    Returns
    -------
    ndarray, shape (2, 2)
        Complex derivative of the top impedance with respect to AT.
    """
    dtzbot = zbot[0, 0] * zbot[1, 1] - zbot[0, 1] * zbot[1, 0]
    zdenom = (
        dtzbot * dfm(ag1) * dfm(ag2) / (dz1 * dz2)
        + zbot[0, 1] * dfm(ag1) * dfp(ag2) / dz1
        - zbot[1, 0] * dfp(ag1) * dfm(ag2) / dz2
        + dfp(ag1) * dfp(ag2)
    )

    dapom = dtzbot * dfm(ag1) / dz1 - zbot[1, 0] * dfp(ag1)
    dbpom = dfp(ag1) + zbot[0, 1] * dfm(ag1) / dz1
    dcpom = zbot[0, 1] * dfp(ag1) + dz1 * dfm(ag1)
    ddpom = dtzbot * dfp(ag1) - dz1 * zbot[1, 0] * dfm(ag1)

    depom = ag2 * dfm(ag2)
    dfpom = (dfm(ag2) + ag2 * dfp(ag2)) / dz2
    dgpom = (dfm(ag2) - ag2 * dfp(ag2)) * dz2

    dzdenom = dfpom * dapom + depom * dbpom
    dn12 = depom * dcpom + dfpom * ddpom
    dn21 = -depom * dapom + dgpom * dbpom

    a2_safe = a2 if a2 != 0.0 else np.finfo(float).tiny
    zdenom_safe = zdenom if zdenom != 0 else (np.finfo(float).tiny + 0j)

    dztop = np.empty((2, 2), dtype=np.complex128)
    dztop[0, 0] = -0.5 * ztop[0, 0] * dzdenom / (a2_safe * zdenom_safe)
    dztop[0, 1] = 0.5 * (dn12 - ztop[0, 1] * dzdenom) / (a2_safe * zdenom_safe)
    dztop[1, 0] = 0.5 * (dn21 - ztop[1, 0] * dzdenom) / (a2_safe * zdenom_safe)
    dztop[1, 1] = -0.5 * ztop[1, 1] * dzdenom / (a2_safe * zdenom_safe)
    return dztop


def _dZ_dblt_layer(
    zbot: np.ndarray,
    ztop: np.ndarray,
    dz1: complex,
    dz2: complex,
    ag1: complex,
    ag2: complex,
) -> np.ndarray:
    """Analytic derivative ``∂Z_top/∂BLT`` (radians) for the current layer."""
    dztop = np.empty((2, 2), dtype=np.complex128)
    dztop[0, 0] = -ztop[0, 1] - ztop[1, 0]
    dztop[0, 1] = ztop[0, 0] - ztop[1, 1]
    dztop[1, 0] = dztop[0, 1]
    dztop[1, 1] = -dztop[0, 0]

    dtzbot = zbot[0, 0] * zbot[1, 1] - zbot[0, 1] * zbot[1, 0]
    zdenom = (
        dtzbot * dfm(ag1) * dfm(ag2) / (dz1 * dz2)
        + zbot[0, 1] * dfm(ag1) * dfp(ag2) / dz1
        - zbot[1, 0] * dfp(ag1) * dfm(ag2) / dz2
        + dfp(ag1) * dfp(ag2)
    )

    exp_term = np.exp(-(ag1 + ag2))
    dzbot_like = np.empty((2, 2), dtype=np.complex128)
    dzbot_like[0, 0] = 4.0 * (zbot[0, 1] + zbot[1, 0]) * exp_term
    dzbot_like[0, 1] = (zbot[0, 0] - zbot[1, 1]) * (dfm(ag1) * dfm(ag2) * dz1 / dz2 - dfp(ag1) * dfp(ag2))
    dzbot_like[1, 0] = (zbot[0, 0] - zbot[1, 1]) * (dfm(ag1) * dfm(ag2) * dz2 / dz1 - dfp(ag1) * dfp(ag2))
    dzbot_like[1, 1] = -4.0 * (zbot[0, 1] + zbot[1, 0]) * exp_term

    dpom = (zbot[0, 0] - zbot[1, 1]) * (dfm(ag1) * dfp(ag2) / dz1 - dfp(ag1) * dfm(ag2) / dz2)

    dztop[0, 0] = dztop[0, 0] + (dzbot_like[0, 0] + dpom * ztop[0, 0]) / zdenom
    dztop[0, 1] = dztop[0, 1] + (dzbot_like[0, 1] + dpom * ztop[0, 1]) / zdenom
    dztop[1, 0] = dztop[1, 0] + (dzbot_like[1, 0] + dpom * ztop[1, 0]) / zdenom
    dztop[1, 1] = dztop[1, 1] + (dzbot_like[1, 1] + dpom * ztop[1, 1]) / zdenom
    return dztop


def _dZ_dh_layer_fdiff(
    zbot: np.ndarray,
    dz1: complex,
    dz2: complex,
    k1: complex,
    k2: complex,
    h_m: float,
    *,
    dh_rel: float = 1e-6,
    dh_abs_m: float = 1e-5,
) -> np.ndarray:
    """Finite-difference derivative ``∂Z_top/∂h_m`` for the current layer.

    Parameters
    ----------
    zbot : ndarray, shape (2, 2)
        Bottom impedance.
    dz1, dz2 : complex
        Layer eigen-impedance factors.
    k1, k2 : complex
        Layer eigen-wavenumbers.
    h_m : float
        Layer thickness in meters.
    dh_rel, dh_abs_m : float
        Relative and absolute perturbation sizes.

    Returns
    -------
    ndarray, shape (2, 2)
        Numerical derivative of the top impedance with respect to thickness.
    """
    h0 = float(h_m)
    dh = max(abs(h0) * dh_rel, dh_abs_m)
    hp = h0 + dh
    hm = max(0.0, h0 - dh)

    ag1p = k1 * hp
    ag2p = k2 * hp
    ag1m = k1 * hm
    ag2m = k2 * hm

    zp = _propagate_impedance(zbot, dz1, dz2, ag1p, ag2p)
    zm = _propagate_impedance(zbot, dz1, dz2, ag1m, ag2m)

    denom = (hp - hm) if hp != hm else dh
    return (zp - zm) / denom


# --- Public forward models ----------------------------------------------------

def _aniso1d_impedance_sens_alat(
    periods_s: np.ndarray,
    h_m: np.ndarray,
    al: np.ndarray,
    at: np.ndarray,
    blt_rad: np.ndarray,
    *,
    compute_sens: bool = True,
    dh_rel: float = 1e-6,
) -> Dict[str, np.ndarray]:
    """Compute 1-D anisotropic impedance and (optionally) sensitivities in (AL,AT,BLT).

    Parameters
    ----------
    periods_s : ndarray, shape (nper,)
        Periods in seconds.
    h_m : ndarray, shape (nl,)
        Layer thicknesses in meters (last entry is basement; ignored).
    al, at, blt_rad : ndarray, shape (nl,)
        Effective horizontal parameters per layer.
    compute_sens : bool
        If True, compute derivatives w.r.t. AL, AT, BLT and h.
    dh_rel : float
        Relative perturbation for the thickness finite difference.

    Returns
    -------
    dict
        Always contains:

        - ``Z``: complex ndarray, shape (nper, 2, 2)

        If ``compute_sens=True`` also contains (complex arrays):

        - ``dZ_dal``:   (nper, nl, 2, 2)
        - ``dZ_dat``:   (nper, nl, 2, 2)
        - ``dZ_dblt``:  (nper, nl, 2, 2)
        - ``dZ_dh_m``:  (nper, nl, 2, 2)

    Notes
    -----
    This is the internal workhorse used by the public wrappers.
    """
    periods_s = np.asarray(periods_s, dtype=float).ravel()
    h_m = np.asarray(h_m, dtype=float).ravel()
    al = np.asarray(al, dtype=float).ravel()
    at = np.asarray(at, dtype=float).ravel()
    blt_rad = np.asarray(blt_rad, dtype=float).ravel()

    nper = periods_s.size
    nl = h_m.size
    if not (al.size == at.size == blt_rad.size == nl):
        raise ValueError("h_m, al, at, blt_rad must all have the same length (nl).")
    if nper == 0:
        raise ValueError("periods_s must be non-empty.")
    if nl == 0:
        raise ValueError("At least one layer (basement) is required.")
    if np.any(periods_s <= 0.0):
        raise ValueError("All periods must be > 0.")

    Z = np.empty((nper, 2, 2), dtype=np.complex128)

    if compute_sens:
        dZ_dal = np.zeros((nper, nl, 2, 2), dtype=np.complex128)
        dZ_dat = np.zeros((nper, nl, 2, 2), dtype=np.complex128)
        dZ_dblt = np.zeros((nper, nl, 2, 2), dtype=np.complex128)
        dZ_dh = np.zeros((nper, nl, 2, 2), dtype=np.complex128)
    else:
        dZ_dal = dZ_dat = dZ_dblt = dZ_dh = None

    k0_prefactor = (1.0 - 1.0j) * 0.002 * np.pi  # legacy scaling

    for ip, per in enumerate(periods_s):
        k0 = k0_prefactor / np.sqrt(10.0 * per)

        # Basement (last layer) in its own strike-aligned frame
        bs = float(blt_rad[-1])
        a1 = float(al[-1])
        a2 = float(at[-1])

        dz1 = k0 / np.sqrt(a1)
        dz2 = k0 / np.sqrt(a2)

        zrot = np.array([[0.0 + 0.0j, dz1], [-dz2, 0.0 + 0.0j]], dtype=np.complex128)

        if compute_sens:
            # Derivatives of the basement impedance w.r.t its AL/AT.
            dZ_dal[ip, -1] = np.array(
                [[0.0 + 0.0j, -0.5 * dz1 / a1], [0.0 + 0.0j, 0.0 + 0.0j]],
                dtype=np.complex128,
            )
            dZ_dat[ip, -1] = np.array(
                [[0.0 + 0.0j, 0.0 + 0.0j], [0.5 * dz2 / a2, 0.0 + 0.0j]],
                dtype=np.complex128,
            )

        bsref = bs

        # Upward recursion through layers nl-2 ... 0
        for il in range(nl - 2, -1, -1):
            a1 = float(al[il])
            a2 = float(at[il])
            bs = float(blt_rad[il])

            # Rotate into current layer strike frame if needed
            if (bs != bsref) and (a1 != a2):
                ang = bs - bsref
                zbot = rotz(zrot, ang)
                if compute_sens:
                    dZ_dal[ip, il + 1 :] = rotz_stack(dZ_dal[ip, il + 1 :], ang)
                    dZ_dat[ip, il + 1 :] = rotz_stack(dZ_dat[ip, il + 1 :], ang)
                    dZ_dblt[ip, il + 1 :] = rotz_stack(dZ_dblt[ip, il + 1 :], ang)
                    dZ_dh[ip, il + 1 :] = rotz_stack(dZ_dh[ip, il + 1 :], ang)
                bsref = bs
            else:
                zbot = zrot.copy()

            k1 = k0 * np.sqrt(a1)
            k2 = k0 * np.sqrt(a2)
            dz1 = k0 / np.sqrt(a1)
            dz2 = k0 / np.sqrt(a2)
            ag1 = k1 * float(h_m[il])
            ag2 = k2 * float(h_m[il])

            ztop = _propagate_impedance(zbot, dz1, dz2, ag1, ag2)

            if compute_sens:
                # Propagate existing layer sensitivities from below.
                for jl in range(il + 1, nl):
                    dZ_dal[ip, jl] = _propagate_sens_from_zbot(dZ_dal[ip, jl], zbot, ztop, dz1, dz2, ag1, ag2)
                    dZ_dat[ip, jl] = _propagate_sens_from_zbot(dZ_dat[ip, jl], zbot, ztop, dz1, dz2, ag1, ag2)
                    dZ_dblt[ip, jl] = _propagate_sens_from_zbot(dZ_dblt[ip, jl], zbot, ztop, dz1, dz2, ag1, ag2)
                    dZ_dh[ip, jl] = _propagate_sens_from_zbot(dZ_dh[ip, jl], zbot, ztop, dz1, dz2, ag1, ag2)

                # Local derivatives for this layer.
                dZ_dal[ip, il] = _dZ_dal_layer(zbot, ztop, a1, dz1, dz2, ag1, ag2)
                dZ_dat[ip, il] = _dZ_dat_layer(zbot, ztop, a2, dz1, dz2, ag1, ag2)
                dZ_dblt[ip, il] = _dZ_dblt_layer(zbot, ztop, dz1, dz2, ag1, ag2)
                dZ_dh[ip, il] = _dZ_dh_layer_fdiff(zbot, dz1, dz2, k1, k2, float(h_m[il]), dh_rel=dh_rel)

            zrot = ztop

        # Rotate final impedance and sensitivities back to global frame
        if bsref != 0.0:
            Z[ip] = rotz(zrot, -bsref)
            if compute_sens:
                dZ_dal[ip] = rotz_stack(dZ_dal[ip], -bsref)
                dZ_dat[ip] = rotz_stack(dZ_dat[ip], -bsref)
                dZ_dblt[ip] = rotz_stack(dZ_dblt[ip], -bsref)
                dZ_dh[ip] = rotz_stack(dZ_dh[ip], -bsref)
        else:
            Z[ip] = zrot

    out: Dict[str, np.ndarray] = {"Z": Z}
    if compute_sens:
        out.update({"dZ_dal": dZ_dal, "dZ_dat": dZ_dat, "dZ_dblt": dZ_dblt, "dZ_dh_m": dZ_dh})
    return out


def aniso1d_impedance_sens_simple(
    periods_s: np.ndarray,
    h_m: np.ndarray,
    rho_max_ohmm: np.ndarray,
    rho_min_ohmm: np.ndarray,
    strike_deg: np.ndarray,
    *,
    compute_sens: bool = True,
    dh_rel: float = 1e-6,
) -> Dict[str, np.ndarray]:
    """Simplified anisotropic 1-D MT forward model (+ sensitivities).

    Parameters
    ----------
    periods_s : ndarray, shape (nper,)
        Periods in seconds.
    h_m : ndarray, shape (nl,)
        Layer thicknesses in meters (last entry is basement; ignored).
    rho_max_ohmm : ndarray, shape (nl,)
        Maximum horizontal resistivity (Ohm·m).
    rho_min_ohmm : ndarray, shape (nl,)
        Minimum horizontal resistivity (Ohm·m).
    strike_deg : ndarray, shape (nl,)
        Anisotropy strike in degrees.
    compute_sens : bool
        If True, also return derivatives w.r.t. ``rho_max_ohmm``,
        ``rho_min_ohmm``, ``strike_deg``, and ``h_m``.
    dh_rel : float
        Relative perturbation for the thickness finite difference.

    Returns
    -------
    dict
        Always contains:

        - ``Z``: complex ndarray, shape (nper, 2, 2)

        If ``compute_sens=True`` also contains:

        - ``dZ_drho_max``:     (nper, nl, 2, 2)
        - ``dZ_drho_min``:     (nper, nl, 2, 2)
        - ``dZ_dstrike_deg``:  (nper, nl, 2, 2)
        - ``dZ_dh_m``:         (nper, nl, 2, 2)

    Notes
    -----
    The mapping used is:

    - ``AL = 1/rho_min``
    - ``AT = 1/rho_max``
    - ``BLT = strike``

    Therefore:

    - ``dAL/drho_min = -1/rho_min^2``
    - ``dAT/drho_max = -1/rho_max^2``
    - ``dBLT/dstrike_deg = pi/180``
    """
    rho_max_ohmm = np.asarray(rho_max_ohmm, dtype=float).ravel()
    rho_min_ohmm = np.asarray(rho_min_ohmm, dtype=float).ravel()
    strike_deg = np.asarray(strike_deg, dtype=float).ravel()
    h_m = np.asarray(h_m, dtype=float).ravel()

    if not (rho_max_ohmm.shape == rho_min_ohmm.shape == strike_deg.shape == h_m.shape):
        raise ValueError("h_m, rho_max_ohmm, rho_min_ohmm, strike_deg must all have the same shape (nl,).")

    al, at, blt_rad = alat_from_rho_strike(rho_max_ohmm, rho_min_ohmm, strike_deg)

    base = _aniso1d_impedance_sens_alat(
        periods_s,
        h_m,
        al,
        at,
        blt_rad,
        compute_sens=compute_sens,
        dh_rel=dh_rel,
    )

    if not compute_sens:
        return {"Z": base["Z"]}

    # Chain rule from (AL, AT, BLT) to (rho_max, rho_min, strike_deg)
    dAL_drho_min = (-1.0 / (rho_min_ohmm**2)).reshape(1, -1, 1, 1)
    dAT_drho_max = (-1.0 / (rho_max_ohmm**2)).reshape(1, -1, 1, 1)
    dBLT_dstrike = (np.pi / 180.0)

    dZ_drho_min = base["dZ_dal"] * dAL_drho_min
    dZ_drho_max = base["dZ_dat"] * dAT_drho_max
    dZ_dstrike = base["dZ_dblt"] * dBLT_dstrike

    out = {
        "Z": base["Z"],
        "dZ_drho_max": dZ_drho_max,
        "dZ_drho_min": dZ_drho_min,
        "dZ_dstrike_deg": dZ_dstrike,
        "dZ_dh_m": base["dZ_dh_m"],
        # Expose the intermediate derivatives for debugging / advanced use.
        "dZ_dal": base["dZ_dal"],
        "dZ_dat": base["dZ_dat"],
        "dZ_dblt": base["dZ_dblt"],
        "al": al,
        "at": at,
        "blt_rad": blt_rad,
    }
    return out




def aniso1d_impedance_sens_simple_sigma(
    periods_s: np.ndarray,
    h_m: np.ndarray,
    sigma_max_Spm: np.ndarray,
    sigma_min_Spm: np.ndarray,
    strike_deg: np.ndarray,
    *,
    compute_sens: bool = True,
    dh_rel: float = 1e-6,
) -> Dict[str, np.ndarray]:
    """Simplified anisotropic 1-D MT forward model (+ sensitivities), conductivity input.

    This is a convenience wrapper around :func:`aniso1d_impedance_sens_simple`
    for workflows that prefer parameterizing *conductivity* instead of
    resistivity.

    Parameters
    ----------
    periods_s : ndarray, shape (nper,)
        Periods in seconds.
    h_m : ndarray, shape (nl,)
        Layer thicknesses in meters.
    sigma_max_Spm : ndarray, shape (nl,)
        Maximum horizontal conductivity (S/m).
        This corresponds to ``1/rho_min`` in the resistivity parameterization.
    sigma_min_Spm : ndarray, shape (nl,)
        Minimum horizontal conductivity (S/m).
        This corresponds to ``1/rho_max`` in the resistivity parameterization.
    strike_deg : ndarray, shape (nl,)
        Strike angle in degrees.
    compute_sens : bool
        If True, also compute sensitivities.
    dh_rel : float
        Relative thickness perturbation used internally for the thickness
        sensitivity finite difference.

    Returns
    -------
    dict
        Same as :func:`aniso1d_impedance_sens_simple`, but the sensitivity keys
        are still expressed w.r.t. resistivities (``dZ_drho_*``). If you need
        sensitivities w.r.t. conductivities, convert using the chain rule:

        - ``rho = 1/sigma``
        - ``dZ/dsigma = dZ/drho * d(1/sigma)/dsigma = -dZ/drho / sigma**2``
    """
    sigma_max_Spm = np.asarray(sigma_max_Spm, dtype=float)
    sigma_min_Spm = np.asarray(sigma_min_Spm, dtype=float)

    # Convert to resistivities for the core implementation.
    tiny = np.finfo(float).tiny
    rho_min_ohmm = 1.0 / np.maximum(sigma_max_Spm, tiny)  # rho_min <-> sigma_max
    rho_max_ohmm = 1.0 / np.maximum(sigma_min_Spm, tiny)  # rho_max <-> sigma_min

    return aniso1d_impedance_sens_simple(
        periods_s=periods_s,
        h_m=h_m,
        rho_max_ohmm=rho_max_ohmm,
        rho_min_ohmm=rho_min_ohmm,
        strike_deg=strike_deg,
        compute_sens=compute_sens,
        dh_rel=dh_rel,
    )

def aniso1d_impedance_sens(
    periods_s: np.ndarray,
    h_m: np.ndarray,
    rop: np.ndarray,
    ustr_deg: np.ndarray,
    udip_deg: np.ndarray,
    usla_deg: np.ndarray,
    *,
    compute_sens: bool = True,
    dh_rel: float = 1e-6,
    fd_rel_rop: float = 1e-6,
    fd_abs_angle_deg: float = 1e-4,
) -> Dict[str, np.ndarray]:
    """Full (legacy) forward model with sensitivities.

    This wrapper is kept for backward compatibility. For inversion workflows,
    prefer :func:`aniso1d_impedance_sens_simple`.

    Parameters
    ----------
    periods_s : ndarray, shape (nper,)
        Periods in seconds.
    h_m : ndarray, shape (nl,)
        Layer thicknesses in meters (last entry is basement).
    rop : ndarray, shape (nl, 3)
        Principal resistivities (Ohm·m).
    ustr_deg, udip_deg, usla_deg : ndarray, shape (nl,)
        Euler angles (degrees): strike, dip, slant.
    compute_sens : bool
        If True, also return derivatives w.r.t. the public parameters.
    dh_rel : float
        Relative step for thickness finite difference.
    fd_rel_rop : float
        Relative step for finite difference derivatives of :func:`cpanis` w.r.t. ``rop``.
    fd_abs_angle_deg : float
        Absolute step (deg) for finite difference derivatives of :func:`cpanis` w.r.t. angles.

    Returns
    -------
    dict
        Always contains ``Z`` (nper,2,2).

        If ``compute_sens=True`` also contains:

        - ``dZ_drop``       (nper, nl, 3, 2, 2)
        - ``dZ_dustr_deg``  (nper, nl, 2, 2)
        - ``dZ_dudip_deg``  (nper, nl, 2, 2)
        - ``dZ_dusla_deg``  (nper, nl, 2, 2)
        - ``dZ_dh_m``       (nper, nl, 2, 2)

        plus intermediate derivatives (``dZ_dal``, ``dZ_dat``, ``dZ_dblt``).
    """
    periods_s = np.asarray(periods_s, dtype=float).ravel()
    h_m = np.asarray(h_m, dtype=float).ravel()
    rop = np.asarray(rop, dtype=float)
    ustr_deg = np.asarray(ustr_deg, dtype=float).ravel()
    udip_deg = np.asarray(udip_deg, dtype=float).ravel()
    usla_deg = np.asarray(usla_deg, dtype=float).ravel()

    if rop.ndim != 2 or rop.shape[1] != 3:
        raise ValueError("rop must have shape (nl, 3).")
    nl = h_m.size
    if rop.shape[0] != nl:
        raise ValueError("rop must have the same number of layers as h_m.")
    if not (ustr_deg.shape == udip_deg.shape == usla_deg.shape == (nl,)):
        raise ValueError("ustr_deg, udip_deg, usla_deg must all have shape (nl,).")

    _sg, al, at, blt_rad = cpanis(rop, ustr_deg, udip_deg, usla_deg)

    base = _aniso1d_impedance_sens_alat(
        periods_s,
        h_m,
        al,
        at,
        blt_rad,
        compute_sens=compute_sens,
        dh_rel=dh_rel,
    )

    if not compute_sens:
        return {"Z": base["Z"]}

    dZ_dal = base["dZ_dal"]
    dZ_dat = base["dZ_dat"]
    dZ_dblt = base["dZ_dblt"]

    nper = periods_s.size

    dZ_drop = np.zeros((nper, nl, 3, 2, 2), dtype=np.complex128)
    dZ_dustr = np.zeros((nper, nl, 2, 2), dtype=np.complex128)
    dZ_dudip = np.zeros((nper, nl, 2, 2), dtype=np.complex128)
    dZ_dusla = np.zeros((nper, nl, 2, 2), dtype=np.complex128)

    def _apply_chain(k: int, dal: float, dat: float, dblt: float) -> np.ndarray:
        return dZ_dal[:, k, :, :] * dal + dZ_dat[:, k, :, :] * dat + dZ_dblt[:, k, :, :] * dblt

    for k in range(nl):
        for j in range(3):
            step = float(fd_rel_rop) * max(abs(float(rop[k, j])), 1.0)
            rop_p = rop.copy()
            rop_p[k, j] = float(rop_p[k, j]) + step
            _sgp, al_p, at_p, blt_p = cpanis(rop_p, ustr_deg, udip_deg, usla_deg)
            dal = (float(al_p[k]) - float(al[k])) / step
            dat = (float(at_p[k]) - float(at[k])) / step
            dblt = (float(blt_p[k]) - float(blt_rad[k])) / step
            dZ_drop[:, k, j, :, :] = _apply_chain(k, dal, dat, dblt)

        for which, arr, out in (
            ("ustr", ustr_deg, dZ_dustr),
            ("udip", udip_deg, dZ_dudip),
            ("usla", usla_deg, dZ_dusla),
        ):
            step = float(fd_abs_angle_deg)
            arr_p = arr.copy()
            arr_p[k] = float(arr_p[k]) + step
            if which == "ustr":
                _sgp, al_p, at_p, blt_p = cpanis(rop, arr_p, udip_deg, usla_deg)
            elif which == "udip":
                _sgp, al_p, at_p, blt_p = cpanis(rop, ustr_deg, arr_p, usla_deg)
            else:
                _sgp, al_p, at_p, blt_p = cpanis(rop, ustr_deg, udip_deg, arr_p)

            dal = (float(al_p[k]) - float(al[k])) / step
            dat = (float(at_p[k]) - float(at[k])) / step
            dblt = (float(blt_p[k]) - float(blt_rad[k])) / step
            out[:, k, :, :] = _apply_chain(k, dal, dat, dblt)

    out = {
        "Z": base["Z"],
        "dZ_drop": dZ_drop,
        "dZ_dustr_deg": dZ_dustr,
        "dZ_dudip_deg": dZ_dudip,
        "dZ_dusla_deg": dZ_dusla,
        "dZ_dh_m": base["dZ_dh_m"],
        # Intermediate derivatives
        "dZ_dal": dZ_dal,
        "dZ_dat": dZ_dat,
        "dZ_dblt": dZ_dblt,
        "al": al,
        "at": at,
        "blt_rad": blt_rad,
    }
    return out


# --- Optional CLI: kept small and focused ------------------------------------

def _parse_periods_arg(s: str) -> np.ndarray:
    """Parse a comma-separated list of periods into a float array."""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("No periods provided.")
    return np.asarray([float(p) for p in parts], dtype=float)


def _load_model_npz(model_path: Path) -> Dict[str, np.ndarray]:
    """Load a model from an NPZ file.

    This loader supports BOTH parameterizations:

    - Simplified keys: ``h_m, rho_min_ohmm, rho_max_ohmm, strike_deg``
    - Full keys: ``h_m, rop, ustr_deg, udip_deg, usla_deg``

    The dict returned contains the keys found.
    """
    with np.load(model_path, allow_pickle=False) as npz:
        keys = set(npz.files)
        out: Dict[str, np.ndarray] = {k: npz[k] for k in npz.files}

    # Normalise common names.
    if "h" in out and "h_m" not in out:
        out["h_m"] = np.asarray(out["h"], dtype=float).ravel()

    # Provide a clearer error if neither parameterization is present.
    has_simple = {"h_m", "rho_min_ohmm", "rho_max_ohmm", "strike_deg"} <= set(out.keys())
    has_full = {"h_m", "rop", "ustr_deg", "udip_deg", "usla_deg"} <= set(out.keys())
    if not (has_simple or has_full):
        raise KeyError(
            "Model NPZ must contain either simplified keys (h_m,rho_min_ohmm,rho_max_ohmm,strike_deg) "
            "or full keys (h_m,rop,ustr_deg,udip_deg,usla_deg). "
            f"Found keys: {sorted(keys)}"
        )

    return {k: np.asarray(v) for k, v in out.items()}


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for a small command-line interface."""
    import argparse

    p = argparse.ArgumentParser(
        prog="aniso.py",
        description="1-D anisotropic MT impedance forward model with optional sensitivities.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    runp = sub.add_parser("run", help="Compute impedance (and optionally sensitivities) from a model NPZ.")
    runp.add_argument("--model", required=True, help="Path to model NPZ.")
    runp.add_argument("--periods", default=None, help="Comma-separated list of periods in seconds.")
    runp.add_argument("--periods-file", default=None, help="Text file with one period per line (seconds).")
    runp.add_argument("--out", required=True, help="Output NPZ path.")
    runp.add_argument("--sens", action="store_true", help="Also compute and store sensitivities.")
    runp.add_argument("--dh-rel", default=1e-6, type=float, help="Relative step for thickness finite difference.")
    runp.add_argument("--quiet", action="store_true", help="Suppress console output.")

    args = p.parse_args(argv)

    if args.cmd != "run":
        p.error(f"Unknown command: {args.cmd}")

    if (args.periods is None) == (args.periods_file is None):
        p.error("Provide exactly one of --periods or --periods-file.")

    model_path = Path(args.model).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    model = _load_model_npz(model_path)

    periods = _parse_periods_arg(args.periods) if args.periods else np.loadtxt(args.periods_file, ndmin=1)
    res = aniso1d_impedance_sens_simple(
         periods_s=periods,
         h_m=np.asarray(model["h_m"], dtype=float).ravel(),
         rho_max_ohmm=np.asarray(model["rho_max_ohmm"], dtype=float).ravel(),
         rho_min_ohmm=np.asarray(model["rho_min_ohmm"], dtype=float).ravel(),
         strike_deg=np.asarray(model["strike_deg"], dtype=float).ravel(),
         compute_sens=bool(args.sens),
         dh_rel=float(args.dh_rel),
     )


    out_dict = {"periods_s": periods}
    out_dict.update(res)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path.as_posix(), **out_dict)

    if not args.quiet:
        print(f"Wrote: {out_path}")
        print(f"  periods: {periods.size}, layers: {np.asarray(model['h_m']).size}")
        print(f"  Z shape: {res['Z'].shape}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
