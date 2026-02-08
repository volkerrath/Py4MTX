#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mcmc.py
================

PyMC driver utilities for simplified anisotropic 1-D MT inversion.

This module is designed to be imported by script-style drivers such as
``mt_aniso1d_sampler.py`` and ``mt_aniso1d_plot.py``.

Compared to older prototypes that inverted full principal resistivities +
Euler angles, this version uses the *simplified* per-layer parameterization
implemented in :func:`aniso.aniso1d_impedance_sens_simple`:

- ``h_m``          (nl,) layer thicknesses in meters (last entry is basement; ignored)
- ``rho_max_ohmm`` (nl,) maximum horizontal resistivity [Ohm·m]
- ``rho_min_ohmm`` (nl,) minimum horizontal resistivity [Ohm·m]
- ``strike_deg``   (nl,) anisotropy strike in degrees

Per-layer flags (optional):

- ``is_iso`` (nl,) boolean: if True, enforce ``rho_max == rho_min`` (strike is irrelevant)
- ``is_fix`` (nl,) boolean: if True, keep this layer fixed at the starting model

Important modelling note
------------------------

Two likelihood implementations are available:

- **Robust black-box** (default): wraps the NumPy forward model with
  PyTensor ``as_op``. This is very stable, but **not differentiable**.
  Use ``DEMetropolisZ``/``Metropolis``.

- **Gradient-enabled** (optional): uses a custom PyTensor ``Op`` whose
  vector–Jacobian products are computed from the analytic impedance
  sensitivities returned by the forward model. This enables NUTS/HMC for
  **impedance** likelihoods and optionally **phase tensor** components (``use_pt=True``).

The deterministic inversion driver (``inv1d.py``) always uses the analytic
sensitivities from the forward model.


Author: Volker Rath (DIAS)
Created with the help of ChatGPT (GPT-5 Thinking) on 2026-02-08 (UTC)
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Optional runtime dependencies (only needed when actually sampling / plotting)
try:  # pragma: no cover
    import arviz as az
except Exception:  # pragma: no cover
    az = None

try:  # pragma: no cover
    import pymc as pm
    import pytensor.tensor as pt
    from pytensor.compile.ops import as_op
    from pytensor.graph.op import Op
except Exception:  # pragma: no cover
    pm = None
    pt = None
    as_op = None
    Op = None


# Local forward model
from aniso import aniso1d_impedance_sens_simple


# -----------------------------------------------------------------------------
# Small filesystem helpers
# -----------------------------------------------------------------------------

def ensure_dir(path: str | os.PathLike) -> str:
    """Ensure a directory exists and return its string path."""
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p.as_posix()


def glob_inputs(pattern: str) -> List[str]:
    """Expand a glob pattern into a sorted list of matching files."""
    from glob import glob

    out = sorted(glob(pattern))
    return out


# -----------------------------------------------------------------------------
# Model I/O and normalization
# -----------------------------------------------------------------------------

def _as_1d(x: np.ndarray, name: str) -> np.ndarray:
    """Convert input to 1-D float array."""
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {x.shape}.")
    return x.astype(float, copy=False)


def model_from_direct(model: Dict) -> Dict[str, np.ndarray]:
    """Create a model dict from an in-script template.

    This is a shallow normalization step: arrays are copied to NumPy ndarrays.
    Use :func:`normalize_model` afterwards.

    Parameters
    ----------
    model : dict
        Model dict (possibly with Python lists).

    Returns
    -------
    dict
        New dict with NumPy arrays.
    """
    out: Dict[str, np.ndarray] = {}
    for k, v in model.items():
        if isinstance(v, (list, tuple)):
            out[k] = np.asarray(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.copy()
        else:
            # keep scalars/strings
            out[k] = v
    return out


def normalize_model(model: Dict) -> Dict[str, np.ndarray]:
    """Normalize a model dict to the simplified parameterization.

    This function makes migration easier by accepting both:

    - new keys: ``rho_min_ohmm``, ``rho_max_ohmm``, ``strike_deg``
    - legacy keys: ``rop`` (nl,3) and ``ustr_deg``

    If the legacy form is used, we map:

    - ``rho_min_ohmm = min(rop[:,0], rop[:,1])``
    - ``rho_max_ohmm = max(rop[:,0], rop[:,1])``
    - ``strike_deg   = ustr_deg``

    The function also ensures that ``is_iso`` and ``is_fix`` exist and have
    length ``nl``.

    Parameters
    ----------
    model : dict
        Model dict.

    Returns
    -------
    dict
        Normalized model dict.
    """
    out = dict(model)

    if "h_m" not in out:
        raise KeyError("Model must contain 'h_m'.")

    h_m = _as_1d(np.asarray(out["h_m"], dtype=float), "h_m")
    nl = h_m.size

    # --- Detect / create simplified resistivity parameters -------------------
    if ("rho_min_ohmm" not in out) or ("rho_max_ohmm" not in out):
        if "rop" in out:
            rop = np.asarray(out["rop"], dtype=float)
            if rop.ndim != 2 or rop.shape[0] != nl or rop.shape[1] < 2:
                raise ValueError("rop must have shape (nl,3) or (nl,>=2).")
            rho1 = rop[:, 0]
            rho2 = rop[:, 1]
            out["rho_min_ohmm"] = np.minimum(rho1, rho2)
            out["rho_max_ohmm"] = np.maximum(rho1, rho2)
        else:
            raise KeyError("Model must contain either (rho_min_ohmm,rho_max_ohmm) or 'rop'.")

    if "strike_deg" not in out:
        if "ustr_deg" in out:
            out["strike_deg"] = np.asarray(out["ustr_deg"], dtype=float)
        else:
            out["strike_deg"] = np.zeros(nl, dtype=float)

    rho_min = _as_1d(np.asarray(out["rho_min_ohmm"], dtype=float), "rho_min_ohmm")
    rho_max = _as_1d(np.asarray(out["rho_max_ohmm"], dtype=float), "rho_max_ohmm")
    strike = _as_1d(np.asarray(out["strike_deg"], dtype=float), "strike_deg")

    if not (rho_min.size == rho_max.size == strike.size == nl):
        raise ValueError("h_m, rho_min_ohmm, rho_max_ohmm, strike_deg must all have length nl.")

    # --- Flags ----------------------------------------------------------------
    if "is_iso" not in out or out["is_iso"] is None:
        out["is_iso"] = np.zeros(nl, dtype=bool)
    if "is_fix" not in out or out["is_fix"] is None:
        out["is_fix"] = np.zeros(nl, dtype=bool)

    is_iso = np.asarray(out["is_iso"], dtype=bool).ravel()
    is_fix = np.asarray(out["is_fix"], dtype=bool).ravel()
    if is_iso.size != nl or is_fix.size != nl:
        raise ValueError("is_iso and is_fix must both have length nl (same as h_m).")

    # Enforce isotropy in the stored model (best-effort)
    rho_max2 = rho_max.copy()
    rho_min2 = rho_min.copy()
    rho_min2[is_iso] = np.minimum(rho_min2[is_iso], rho_max2[is_iso])
    rho_max2[is_iso] = rho_min2[is_iso]

    out.update(
        {
            "h_m": h_m,
            "rho_min_ohmm": rho_min2,
            "rho_max_ohmm": rho_max2,
            "strike_deg": strike,
            "is_iso": is_iso,
            "is_fix": is_fix,
        }
    )

    return out


def save_model_npz(model: Dict, path: str | os.PathLike) -> None:
    """Save a model dict to an NPZ file."""
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    m = normalize_model(model)

    np.savez(
        p.as_posix(),
        h_m=m["h_m"],
        rho_min_ohmm=m["rho_min_ohmm"],
        rho_max_ohmm=m["rho_max_ohmm"],
        strike_deg=m["strike_deg"],
        is_iso=m["is_iso"],
        is_fix=m["is_fix"],
        prior_name=str(m.get("prior_name", "")),
    )


def load_model_npz(path: str | os.PathLike) -> Dict[str, np.ndarray]:
    """Load a model dict from an NPZ file."""
    p = Path(path).expanduser().resolve()
    with np.load(p.as_posix(), allow_pickle=True) as npz:
        model = {k: npz[k] for k in npz.files}
    return normalize_model(model)


# -----------------------------------------------------------------------------
# Site I/O
# -----------------------------------------------------------------------------

def load_site(path: str | os.PathLike) -> Dict:
    """Load one MT site from an ``.edi`` or ``.npz`` file.

    The exact file format is delegated to the user's `data_proc` module when
    reading EDI.

    The returned dict is expected to contain at least:

    - ``station`` (str)
    - ``freq`` (n,) in Hz
    - ``Z`` (n,2,2) complex
    - optionally ``Z_err`` (n,2,2)
    - optionally ``P`` and ``P_err`` for phase tensor

    Parameters
    ----------
    path : str or PathLike
        Input file.

    Returns
    -------
    dict
        Site dictionary.
    """
    p = Path(path).expanduser().resolve()
    ext = p.suffix.lower()

    if ext == ".npz":
        with np.load(p.as_posix(), allow_pickle=True) as npz:
            site = {k: npz[k] for k in npz.files}
        if "station" not in site:
            site["station"] = p.stem
        return site

    if ext == ".edi":
        import data_proc  # provided by the user's code base

        site = data_proc.load_edi(p.as_posix())
        if "station" not in site:
            site["station"] = p.stem
        return site

    raise ValueError(f"Unsupported site format: {p}")


def _call_compute_pt(Z: np.ndarray, Z_err: Optional[np.ndarray], nsim: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Compute phase tensor via the user's `data_proc.compute_pt` if possible."""
    import data_proc

    fn = getattr(data_proc, "compute_pt", None)
    if fn is None:
        raise ImportError("data_proc.compute_pt not found.")

    # Try a few common calling conventions.
    try:
        P, P_err = fn(Z, Z_err, nsim=nsim)
        return P, P_err
    except TypeError:
        pass
    try:
        P, P_err = fn(Z, nsim=nsim)
        return P, P_err
    except TypeError:
        pass
    P, P_err = fn(Z)
    return P, P_err


def ensure_phase_tensor(site: Dict, nsim: int = 200, *, overwrite: bool = True) -> Dict:
    """Ensure that a site dict contains a phase tensor ``P`` (and optionally ``P_err``).

    **Always** recomputes ``P`` from ``Z`` using the fixed convention::

        P = inv(Re(Z)) @ Im(Z)

    If ``Z_err`` is present and ``P_err`` is missing (or ``overwrite=True``), an
    approximate ``P_err`` is estimated by simple Monte-Carlo propagation with
    independent entry errors.

    Parameters
    ----------
    site : dict
        Site dict.
    nsim : int
        Monte-Carlo sample count used for the optional ``P_err`` estimate.
    overwrite : bool
        If True, overwrite existing ``P`` (and ``P_err`` when estimated).

    Returns
    -------
    dict
        A (possibly copied) site dict containing ``P``.
    """
    Z = np.asarray(site.get("Z", None))
    if Z is None:
        raise KeyError("site must contain 'Z' to compute 'P'.")
    Z = np.asarray(Z, dtype=np.complex128)

    out = dict(site)
    if overwrite or ("P" not in out):
        out["P"] = phase_tensor_from_Z(Z, reg=0.0)

    # Error propagation for PT (optional)
    if ("Z_err" in out) and (overwrite or ("P_err" not in out)):
        try:
            Z_err = np.asarray(out["Z_err"], dtype=float)
            if Z_err.shape == Z.shape:
                rng = np.random.default_rng(12345)
                ns = int(max(10, nsim))
                Ps = np.empty((ns,) + out["P"].shape, dtype=float)
                for i in range(ns):
                    noise = rng.standard_normal(Z.shape) + 1j * rng.standard_normal(Z.shape)
                    Zs = Z + noise * Z_err
                    Ps[i] = phase_tensor_from_Z(Zs, reg=0.0)
                out["P_err"] = Ps.std(axis=0, ddof=1)
        except Exception:
            pass

    return out


# -----------------------------------------------------------------------------
# Component selection / data vectors
# -----------------------------------------------------------------------------

_COMP_TO_IJ = {"xx": (0, 0), "xy": (0, 1), "yx": (1, 0), "yy": (1, 1)}


def _parse_comps(comps: Sequence[str]) -> List[Tuple[int, int]]:
    """Parse component labels into (i,j) indices."""
    out: List[Tuple[int, int]] = []
    for c in comps:
        cc = str(c).strip().lower()
        if cc not in _COMP_TO_IJ:
            raise ValueError(f"Unknown component label: {c}")
        out.append(_COMP_TO_IJ[cc])
    return out


def pack_Z_vector(
    Z: np.ndarray,
    comps: Sequence[str] = ("xx", "xy", "yx", "yy"),
) -> np.ndarray:
    """Pack complex impedance tensor Z into a real-valued data vector.

    The packed vector is ordered by period/frequency, then by component,
    and stores **Re** and **Im** parts consecutively.

    Parameters
    ----------
    Z : ndarray, shape (n,2,2)
        Complex impedance tensor.
    comps : sequence of str
        Component labels among {xx,xy,yx,yy}.

    Returns
    -------
    ndarray, shape (n * ncomp * 2,)
        Real-valued data vector.
    """
    Z = np.asarray(Z, dtype=np.complex128)
    if Z.ndim != 3 or Z.shape[1:] != (2, 2):
        raise ValueError("Z must have shape (n,2,2).")

    ij = _parse_comps(comps)
    n = Z.shape[0]
    ncomp = len(ij)

    out = np.empty(n * ncomp * 2, dtype=float)
    k = 0
    for ip in range(n):
        for (i, j) in ij:
            out[k] = float(np.real(Z[ip, i, j])); k += 1
            out[k] = float(np.imag(Z[ip, i, j])); k += 1
    return out


def pack_Z_sigma(
    Z_err: np.ndarray,
    comps: Sequence[str] = ("xx", "xy", "yx", "yy"),
    *,
    sigma_floor: float = 0.0,
) -> np.ndarray:
    """Pack impedance uncertainties into a sigma vector matching :func:`pack_Z_vector`.

    Parameters
    ----------
    Z_err : ndarray, shape (n,2,2)
        Error estimate per Z entry. May be real or complex; magnitudes are used.
    comps : sequence of str
        Component labels.
    sigma_floor : float
        Minimum sigma added in quadrature.

    Returns
    -------
    ndarray
        Sigma vector.
    """
    Z_err = np.asarray(Z_err)
    if Z_err.ndim != 3 or Z_err.shape[1:] != (2, 2):
        raise ValueError("Z_err must have shape (n,2,2).")

    ij = _parse_comps(comps)
    n = Z_err.shape[0]
    ncomp = len(ij)

    out = np.empty(n * ncomp * 2, dtype=float)
    k = 0
    for ip in range(n):
        for (i, j) in ij:
            s = float(np.abs(Z_err[ip, i, j]))
            s = float(np.sqrt(s * s + sigma_floor * sigma_floor))
            out[k] = s; k += 1
            out[k] = s; k += 1
    return out


def pack_P_vector(
    P: np.ndarray,
    comps: Sequence[str] = ("xx", "xy", "yx", "yy"),
) -> np.ndarray:
    """Pack phase tensor components into a real-valued data vector."""
    P = np.asarray(P, dtype=float)
    if P.ndim != 3 or P.shape[1:] != (2, 2):
        raise ValueError("P must have shape (n,2,2).")

    ij = _parse_comps(comps)
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
    if P_err.ndim != 3 or P_err.shape[1:] != (2, 2):
        raise ValueError("P_err must have shape (n,2,2).")

    ij = _parse_comps(comps)
    n = P_err.shape[0]
    out = np.empty(n * len(ij), dtype=float)
    k = 0
    for ip in range(n):
        for (i, j) in ij:
            s = float(abs(P_err[ip, i, j]))
            out[k] = float(np.sqrt(s * s + sigma_floor * sigma_floor)); k += 1
    return out




# -----------------------------------------------------------------------------
# Gradient-enabled forward operator (for NUTS/HMC)
# -----------------------------------------------------------------------------

def _pack_Z_derivs_by_layer(dZ: np.ndarray, comps: Sequence[str]) -> np.ndarray:
    """Pack per-layer impedance derivatives into a matrix.

    Parameters
    ----------
    dZ : ndarray, shape (nper, nl, 2, 2)
        Complex impedance derivatives w.r.t. one per-layer parameter.
    comps : sequence of str
        Components among {xx,xy,yx,yy}.

    Returns
    -------
    ndarray, shape (nl, nper * ncomp * 2)
        Matrix whose row ``l`` is the packed derivative vector for layer ``l``,
        in the same ordering as :func:`pack_Z_vector`.
    """
    dZ = np.asarray(dZ, dtype=np.complex128)
    if dZ.ndim != 4 or dZ.shape[2:] != (2, 2):
        raise ValueError("dZ must have shape (nper,nl,2,2).")
    ij = _parse_comps(comps)
    nper, nl = dZ.shape[0], dZ.shape[1]
    ncomp = len(ij)

    out = np.empty((nl, nper * ncomp * 2), dtype=float)
    k = 0
    for ip in range(nper):
        for (i, j) in ij:
            out[:, k] = np.real(dZ[ip, :, i, j]); k += 1
            out[:, k] = np.imag(dZ[ip, :, i, j]); k += 1
    return out




def _pack_P_derivs_by_layer(dP: np.ndarray, comps: Sequence[str]) -> np.ndarray:
    """Pack per-layer phase tensor derivatives into a matrix.

    Parameters
    ----------
    dP : ndarray, shape (nper, nl, 2, 2)
        Phase tensor derivatives w.r.t. one per-layer parameter.
    comps : sequence of str
        Components among {xx,xy,yx,yy}.

    Returns
    -------
    ndarray, shape (nl, nper * ncomp)
        Matrix whose row ``l`` is the packed derivative vector for layer ``l``,
        in the same ordering as :func:`pack_P_vector`.
    """
    dP = np.asarray(dP, dtype=float)
    if dP.ndim != 4 or dP.shape[2:] != (2, 2):
        raise ValueError("dP must have shape (nper,nl,2,2).")
    ij = _parse_comps(comps)
    nper, nl = dP.shape[0], dP.shape[1]
    out = np.empty((nl, nper * len(ij)), dtype=float)
    k = 0
    for ip in range(nper):
        for (i, j) in ij:
            out[:, k] = dP[ip, :, i, j]; k += 1
    return out


def _pt_solve_reg(A: np.ndarray, B: np.ndarray, reg: float) -> np.ndarray:
    """Solve A X = B with optional small diagonal regularization.

    The regularization uses a scale based on the infinity-norm of A:

        A_reg = A + (reg * max(1, ||A||_inf)) * I

    Parameters
    ----------
    A : ndarray, shape (2,2)
        System matrix.
    B : ndarray, shape (2,2)
        Right-hand side.
    reg : float
        Relative regularization level.

    Returns
    -------
    ndarray, shape (2,2)
        Solution X.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if reg > 0.0:
        scale = float(max(1.0, np.linalg.norm(A, ord=np.inf)))
        A = A + (reg * scale) * np.eye(2, dtype=float)
    try:
        return np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        # Fallback: least squares
        return np.linalg.lstsq(A, B, rcond=None)[0]


def phase_tensor_from_Z(
    Z: np.ndarray,
    *,
    reg: float = 0.0,
) -> np.ndarray:
    """Compute phase tensor **P = inv(Re(Z)) @ Im(Z)**.

    This project now uses a single, fixed convention:

        X = Re(Z),  Y = Im(Z),   P = X^{-1} Y

    Parameters
    ----------
    Z : ndarray, shape (n,2,2)
        Complex impedance tensor.
    reg : float
        Relative diagonal regularization for the inversion (useful for NUTS/HMC
        stability when Re(Z) becomes ill-conditioned).

    Returns
    -------
    ndarray, shape (n,2,2)
        Real phase tensor.
    """
    Z = np.asarray(Z, dtype=np.complex128)
    if Z.ndim != 3 or Z.shape[1:] != (2, 2):
        raise ValueError("Z must have shape (n,2,2).")

    n = Z.shape[0]
    P = np.empty((n, 2, 2), dtype=float)
    for k in range(n):
        X = Z[k].real
        Y = Z[k].imag
        P[k] = _pt_solve_reg(X, Y, reg)
    return P


def phase_tensor_dP_from_dZ(
    Z: np.ndarray,
    dZ: np.ndarray,
    *,
    reg: float = 0.0,
    P: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute phase tensor derivatives dP from impedance derivatives dZ.

    For the fixed convention **P = X^{-1} Y** with X=Re(Z), Y=Im(Z):

        dP = X^{-1} ( dY - dX P )

    Parameters
    ----------
    Z : ndarray, shape (nper,2,2)
        Impedance tensor.
    dZ : ndarray, shape (nper,nl,2,2)
        Impedance derivatives for one per-layer parameter.
    reg : float
        Relative diagonal regularization for the inversion.
    P : ndarray, optional
        If provided, uses this phase tensor instead of recomputing it.

    Returns
    -------
    ndarray, shape (nper,nl,2,2)
        Phase tensor derivatives.
    """
    Z = np.asarray(Z, dtype=np.complex128)
    dZ = np.asarray(dZ, dtype=np.complex128)
    if Z.ndim != 3 or Z.shape[1:] != (2, 2):
        raise ValueError("Z must have shape (nper,2,2).")
    if dZ.ndim != 4 or dZ.shape[0] != Z.shape[0] or dZ.shape[2:] != (2, 2):
        raise ValueError("dZ must have shape (nper,nl,2,2) matching Z.")

    nper, nl = dZ.shape[0], dZ.shape[1]
    if P is None:
        P = phase_tensor_from_Z(Z, reg=reg)
    else:
        P = np.asarray(P, dtype=float)

    dP = np.empty((nper, nl, 2, 2), dtype=float)
    for ip in range(nper):
        X = Z[ip].real
        for l in range(nl):
            dX = dZ[ip, l].real
            dY = dZ[ip, l].imag
            rhs = dY - dX @ P[ip]
            dP[ip, l] = _pt_solve_reg(X, rhs, reg)
    return dP


class _ForwardPackedOp(Op):  # pragma: no cover (depends on pytensor runtime)
    """PyTensor Op: forward prediction y_pred = f(h, rho_max, rho_min, strike).

    This Op is differentiable because its :meth:`grad` returns a companion Op
    that computes vector–Jacobian products using the analytic impedance
    sensitivities from :func:`aniso.aniso1d_impedance_sens_simple`.

    The Op returns a *real* vector:

    - impedance contributions are stacked as Re/Im pairs (see :func:`pack_Z_vector`)
    - if ``use_pt=True`` an additional block contains phase tensor components
      (see :func:`pack_P_vector`)

    Phase tensor convention
    -----------------------
    This code always uses **P = inv(Re(Z)) @ Im(Z)**. The phase tensor is
    recomputed from Z for every likelihood evaluation to keep conventions and
    gradients consistent.

    Notes
    -----
    PT gradients can become numerically unstable if the inverted matrix
    (Re(Z) or Im(Z), depending on definition) approaches singularity. A small
    diagonal regularization ``pt_reg`` can help NUTS/HMC explore the posterior
    more robustly.
    """

    itypes = None  # set in __init__
    otypes = None  # set in __init__

    def __init__(
        self,
        *,
        periods_s: np.ndarray,
        z_comps: Sequence[str],
        use_pt: bool,
        pt_comps: Sequence[str],
                pt_reg: float,
        dh_rel: float,
    ):
        if pt is None or Op is None:
            raise ImportError("PyTensor not available.")
        self.periods_s = np.asarray(periods_s, dtype=float).ravel()
        self.z_comps = tuple(z_comps)
        self.use_pt = bool(use_pt)
        self.pt_comps = tuple(pt_comps)
        self.pt_reg = float(pt_reg)
        self.dh_rel = float(dh_rel)

        # Types: four dvector inputs, one dvector output
        self.itypes = [pt.dvector, pt.dvector, pt.dvector, pt.dvector]
        self.otypes = [pt.dvector]

        # Shared cache for forward + vjp
        self._cache = {"x": None, "y": None, "sens": None}

        self._vjp_op = _ForwardPackedVJPOp(
            periods_s=self.periods_s,
            z_comps=self.z_comps,
            use_pt=self.use_pt,
            pt_comps=self.pt_comps,
                        pt_reg=self.pt_reg,
            dh_rel=self.dh_rel,
            cache=self._cache,
        )

    def perform(self, node, inputs, outputs):
        h_m_v, rho_max_v, rho_min_v, strike_v = inputs
        fres = aniso1d_impedance_sens_simple(
            periods_s=self.periods_s,
            h_m=np.asarray(h_m_v, dtype=float),
            rho_max_ohmm=np.asarray(rho_max_v, dtype=float),
            rho_min_ohmm=np.asarray(rho_min_v, dtype=float),
            strike_deg=np.asarray(strike_v, dtype=float),
            compute_sens=False,
            dh_rel=self.dh_rel,
        )
        Zp = np.asarray(fres["Z"], dtype=np.complex128)
        y_parts = [pack_Z_vector(Zp, comps=self.z_comps).astype(float)]

        if self.use_pt:
            Pp = phase_tensor_from_Z(Zp, reg=self.pt_reg)
            y_parts.append(pack_P_vector(Pp, comps=self.pt_comps).astype(float))

        ypred = np.concatenate(y_parts).astype(float)
        outputs[0][0] = ypred

        # Update cache (no sensitivities here)
        self._cache["x"] = (
            np.asarray(h_m_v, dtype=float).copy(),
            np.asarray(rho_max_v, dtype=float).copy(),
            np.asarray(rho_min_v, dtype=float).copy(),
            np.asarray(strike_v, dtype=float).copy(),
        )
        self._cache["y"] = ypred
        self._cache["sens"] = None

    def grad(self, inputs, g_outputs):
        (g_y,) = g_outputs
        gh, grmax, grmin, gstrike = self._vjp_op(*inputs, g_y)
        return [gh, grmax, grmin, gstrike]


class _ForwardPackedVJPOp(Op):  # pragma: no cover (depends on pytensor runtime)
    """Companion Op computing vector–Jacobian products for _ForwardPackedOp."""

    itypes = None
    otypes = None

    def __init__(
        self,
        *,
        periods_s: np.ndarray,
        z_comps: Sequence[str],
        use_pt: bool,
        pt_comps: Sequence[str],
                pt_reg: float,
        dh_rel: float,
        cache: Optional[Dict] = None,
    ):
        if pt is None or Op is None:
            raise ImportError("PyTensor not available.")
        self.periods_s = np.asarray(periods_s, dtype=float).ravel()
        self.z_comps = tuple(z_comps)
        self.use_pt = bool(use_pt)
        self.pt_comps = tuple(pt_comps)
        self.pt_reg = float(pt_reg)
        self.dh_rel = float(dh_rel)
        self.cache = cache

        # inputs: h, rho_max, rho_min, strike, g_y
        self.itypes = [pt.dvector, pt.dvector, pt.dvector, pt.dvector, pt.dvector]
        # outputs: gradients for each of the 4 vector inputs
        self.otypes = [pt.dvector, pt.dvector, pt.dvector, pt.dvector]

    def perform(self, node, inputs, outputs):
        h_m_v, rho_max_v, rho_min_v, strike_v, g_y = inputs
        g_y = np.asarray(g_y, dtype=float).ravel()

        # Reuse sensitivities if cached and inputs match exactly
        fres = None
        if self.cache is not None and self.cache.get("sens", None) is not None and self.cache.get("x", None) is not None:
            x0 = self.cache["x"]
            if (
                np.array_equal(x0[0], h_m_v)
                and np.array_equal(x0[1], rho_max_v)
                and np.array_equal(x0[2], rho_min_v)
                and np.array_equal(x0[3], strike_v)
            ):
                fres = self.cache["sens"]

        if fres is None:
            fres = aniso1d_impedance_sens_simple(
                periods_s=self.periods_s,
                h_m=np.asarray(h_m_v, dtype=float),
                rho_max_ohmm=np.asarray(rho_max_v, dtype=float),
                rho_min_ohmm=np.asarray(rho_min_v, dtype=float),
                strike_deg=np.asarray(strike_v, dtype=float),
                compute_sens=True,
                dh_rel=self.dh_rel,
            )
            if self.cache is not None:
                self.cache["sens"] = fres
                self.cache["x"] = (
                    np.asarray(h_m_v, dtype=float).copy(),
                    np.asarray(rho_max_v, dtype=float).copy(),
                    np.asarray(rho_min_v, dtype=float).copy(),
                    np.asarray(strike_v, dtype=float).copy(),
                )

        Z = np.asarray(fres["Z"], dtype=np.complex128)
        dZ_dh = np.asarray(fres["dZ_dh_m"], dtype=np.complex128)           # (nper,nl,2,2)
        dZ_drmax = np.asarray(fres["dZ_drho_max"], dtype=np.complex128)    # (nper,nl,2,2)
        dZ_drmin = np.asarray(fres["dZ_drho_min"], dtype=np.complex128)    # (nper,nl,2,2)
        dZ_dstr = np.asarray(fres["dZ_dstrike_deg"], dtype=np.complex128)  # (nper,nl,2,2)

        A_h = _pack_Z_derivs_by_layer(dZ_dh, comps=self.z_comps)
        A_rmax = _pack_Z_derivs_by_layer(dZ_drmax, comps=self.z_comps)
        A_rmin = _pack_Z_derivs_by_layer(dZ_drmin, comps=self.z_comps)
        A_str = _pack_Z_derivs_by_layer(dZ_dstr, comps=self.z_comps)

        # Split upstream gradient into Z / PT blocks
        nper = int(self.periods_s.size)
        nZ = nper * len(_parse_comps(self.z_comps)) * 2
        gZ = g_y[:nZ]

        gh = A_h @ gZ
        grmax = A_rmax @ gZ
        grmin = A_rmin @ gZ
        gstr = A_str @ gZ

        if self.use_pt:
            gP = g_y[nZ:]
            # Compute P and dP/dparam from Z and dZ/dparam
            P = phase_tensor_from_Z(Z, reg=self.pt_reg)

            dP_dh = phase_tensor_dP_from_dZ(Z, dZ_dh, reg=self.pt_reg, P=P)
            dP_drmax = phase_tensor_dP_from_dZ(Z, dZ_drmax, reg=self.pt_reg, P=P)
            dP_drmin = phase_tensor_dP_from_dZ(Z, dZ_drmin, reg=self.pt_reg, P=P)
            dP_dstr = phase_tensor_dP_from_dZ(Z, dZ_dstr, reg=self.pt_reg, P=P)

            B_h = _pack_P_derivs_by_layer(dP_dh, comps=self.pt_comps)
            B_rmax = _pack_P_derivs_by_layer(dP_drmax, comps=self.pt_comps)
            B_rmin = _pack_P_derivs_by_layer(dP_drmin, comps=self.pt_comps)
            B_str = _pack_P_derivs_by_layer(dP_dstr, comps=self.pt_comps)

            gh = gh + (B_h @ gP)
            grmax = grmax + (B_rmax @ gP)
            grmin = grmin + (B_rmin @ gP)
            gstr = gstr + (B_str @ gP)

        outputs[0][0] = np.asarray(gh, dtype=float)
        outputs[1][0] = np.asarray(grmax, dtype=float)
        outputs[2][0] = np.asarray(grmin, dtype=float)
        outputs[3][0] = np.asarray(gstr, dtype=float)

# -----------------------------------------------------------------------------
# Parameter specification
# -----------------------------------------------------------------------------


class ParamSpec:
    """Container describing the inversion parameterization and bounds.

    Notes
    -----
    This is intentionally *not* a dataclass (user preference).

    Parameters
    ----------
    nl : int
        Number of layers.
    fix_h : bool
        If True, thicknesses are not sampled.
    sample_last_thickness : bool
        If True, also sample the last entry of ``h_m`` (usually the basement
        thickness placeholder). Most workflows keep it fixed at 0.
    log10_h_bounds : tuple(float,float)
        Bounds for log10(h_m).
    log10_rho_bounds : tuple(float,float)
        Bounds for log10 resistivities.
    strike_bounds_deg : tuple(float,float)
        Bounds for strike in degrees.
    """

    def __init__(
        self,
        *,
        nl: int,
        fix_h: bool = True,
        sample_last_thickness: bool = False,
        log10_h_bounds: Tuple[float, float] = (0.0, 5.0),
        log10_rho_bounds: Tuple[float, float] = (0.0, 5.0),
        strike_bounds_deg: Tuple[float, float] = (-180.0, 180.0),
    ) -> None:
        self.nl = int(nl)
        self.fix_h = bool(fix_h)
        self.sample_last_thickness = bool(sample_last_thickness)
        self.log10_h_bounds = tuple(float(x) for x in log10_h_bounds)
        self.log10_rho_bounds = tuple(float(x) for x in log10_rho_bounds)
        self.strike_bounds_deg = tuple(float(x) for x in strike_bounds_deg)


# -----------------------------------------------------------------------------
# PyMC model build + sampling
# -----------------------------------------------------------------------------


def build_pymc_model(
    site: Dict,
    *,
    spec: ParamSpec,
    model0: Optional[Dict] = None,
    h_m0: Optional[np.ndarray] = None,
    rho_min0: Optional[np.ndarray] = None,
    rho_max0: Optional[np.ndarray] = None,
    strike_deg0: Optional[np.ndarray] = None,
    is_iso: Optional[np.ndarray] = None,
    is_fix: Optional[np.ndarray] = None,
    use_pt: bool = False,
    z_comps: Sequence[str] = ("xx", "xy", "yx", "yy"),
    pt_comps: Sequence[str] = ("xx", "xy", "yx", "yy"),
    pt_reg: float = 1e-12,
    sigma_floor_Z: float = 0.0,
    sigma_floor_P: float = 0.0,
    enable_grad: bool = False,
    prior_kind: str = "default",
    param_domain: str = "rho",
    dh_rel: float = 1e-6,
) -> Tuple["pm.Model", Dict]:
    """Build a PyMC model for one site using the simplified forward model.

    Parameters
    ----------
    site : dict
        Site dict containing ``freq``, ``Z`` and (optionally) errors.
    spec : ParamSpec
        Parameter specification.
    model0 : dict, optional
        Starting model. If provided, used as defaults for missing arrays.
    h_m0, rho_min0, rho_max0, strike_deg0 : ndarray, optional
        Starting arrays (override `model0`).
    is_iso, is_fix : ndarray, optional
        Per-layer flags (override `model0`).
    use_pt : bool
        If True, include phase tensor components in the likelihood.
    sigma_floor_Z, sigma_floor_P : float
        Error floors added in quadrature.
    enable_grad : bool
        If True, build a gradient-enabled likelihood (supports NUTS/HMC) by
        using a custom PyTensor Op whose gradients are computed from the
        analytic impedance sensitivities.

        Notes:
        - Impedance + optional phase tensor likelihoods are supported.
        - For best stability with NUTS, consider ``spec.fix_h=True`` to avoid
          finite-difference thickness sensitivities.
    prior_kind : str
        One of ``{"default","uniform"}``.

        ``"default"`` uses smooth bounded transforms and an explicit anisotropy
        ratio parameter to encourage near-isotropy while enforcing
        ``rho_max >= rho_min``.

        ``"uniform"`` reproduces the older broad uniform sampling behaviour.
    param_domain : str
        One of ``{"rho","sigma"}``.

        - ``"rho"`` samples resistivities (Ohm·m).
        - ``"sigma"`` samples conductivities (S/m) and converts to resistivities
          for the forward model.
    dh_rel : float
        Relative perturbation used for the thickness derivative inside the
        forward sensitivities (only relevant when sampling/inverting ``h_m``).

    Returns
    -------
    pm.Model
        The PyMC model.
    dict
        Info dict (contains packed observation vectors and component metadata).
    """
    if pm is None or pt is None:
        raise ImportError("PyMC / PyTensor not available in this environment.")
    if enable_grad:
        if Op is None:
            raise ImportError("PyTensor Op not available (needed for enable_grad=True).")
    else:
        if as_op is None:
            raise ImportError("pytensor.as_op not available (needed for enable_grad=False).")

    # Starting model resolution
    if model0 is not None:
        m0 = normalize_model(model0)
    else:
        m0 = None

    freq = np.asarray(site.get("freq", None), dtype=float).ravel()
    if freq.size == 0:
        raise ValueError("site must contain non-empty 'freq'.")
    periods_s = 1.0 / freq

    Z_obs = np.asarray(site.get("Z", None))
    if Z_obs is None:
        raise KeyError("site must contain 'Z'.")
    Z_obs = np.asarray(Z_obs, dtype=np.complex128)

    if use_pt:
        site = ensure_phase_tensor(dict(site), nsim=200, overwrite=True)


    # Observations
    yZ = pack_Z_vector(Z_obs, comps=z_comps)
    if "Z_err" in site:
        sigmaZ = pack_Z_sigma(site["Z_err"], comps=z_comps, sigma_floor=float(sigma_floor_Z))
    else:
        sigmaZ = np.ones_like(yZ) * max(float(sigma_floor_Z), 1.0)

    y_list = [yZ]
    s_list = [sigmaZ]

    if use_pt:
        P_obs = np.asarray(site.get("P", None), dtype=float)
        yP = pack_P_vector(P_obs, comps=pt_comps)
        if "P_err" in site:
            sigmaP = pack_P_sigma(site["P_err"], comps=pt_comps, sigma_floor=float(sigma_floor_P))
        else:
            sigmaP = np.ones_like(yP) * max(float(sigma_floor_P), 1.0)
        y_list.append(yP)
        s_list.append(sigmaP)

    y_obs = np.concatenate(y_list).astype(float)
    sigma = np.concatenate(s_list).astype(float)

    nl = int(spec.nl)
    if m0 is not None:
        h0 = np.asarray(m0["h_m"], dtype=float)
        rmin0 = np.asarray(m0["rho_min_ohmm"], dtype=float)
        rmax0 = np.asarray(m0["rho_max_ohmm"], dtype=float)
        str0 = np.asarray(m0["strike_deg"], dtype=float)
        iso0 = np.asarray(m0["is_iso"], dtype=bool)
        fix0 = np.asarray(m0["is_fix"], dtype=bool)
    else:
        # must be provided
        if h_m0 is None or rho_min0 is None or rho_max0 is None or strike_deg0 is None:
            raise ValueError("Provide model0 or explicit *_0 arrays.")
        h0 = np.asarray(h_m0, dtype=float)
        rmin0 = np.asarray(rho_min0, dtype=float)
        rmax0 = np.asarray(rho_max0, dtype=float)
        str0 = np.asarray(strike_deg0, dtype=float)
        iso0 = np.zeros(nl, dtype=bool) if is_iso is None else np.asarray(is_iso, dtype=bool)
        fix0 = np.zeros(nl, dtype=bool) if is_fix is None else np.asarray(is_fix, dtype=bool)

    if not (h0.size == rmin0.size == rmax0.size == str0.size == iso0.size == fix0.size == nl):
        raise ValueError("Starting model arrays must all have length nl.")

    # Bounds
    lo_r, hi_r = spec.log10_rho_bounds
    lo_s, hi_s = spec.strike_bounds_deg
    lo_h, hi_h = spec.log10_h_bounds

    with pm.Model() as model:
        is_iso_data = pm.MutableData("is_iso", iso0.astype(bool))
        is_fix_data = pm.MutableData("is_fix", fix0.astype(bool))

        
        # ------------------------------------------------------------------
        # Resistivity / conductivity parameterization (ordered pair + soft priors)
        # ------------------------------------------------------------------
        prior_kind_ = str(prior_kind).lower().strip()
        domain_ = str(param_domain).lower().strip()
        if domain_ not in ("rho", "sigma"):
            raise ValueError("param_domain must be 'rho' or 'sigma'.")

        # Bounds are specified in terms of log10(rho) in ParamSpec.
        # For conductivity we use the implied bounds log10(sigma) = -log10(rho).
        if domain_ == "rho":
            lo_a, hi_a = float(lo_r), float(hi_r)  # log10(rho)
        else:
            lo_a, hi_a = float(-hi_r), float(-lo_r)  # log10(sigma)

        max_delta = float(hi_a - lo_a)

        if prior_kind_ == "uniform":
            # Sample two unconstrained log10 fields and then take min/max.
            # This guarantees an ordered pair without hard constraints.
            log10_a1 = pm.Uniform("log10_a1", lower=lo_a, upper=hi_a, shape=(nl,))
            log10_a2 = pm.Uniform("log10_a2", lower=lo_a, upper=hi_a, shape=(nl,))
            log10_low_free = pt.minimum(log10_a1, log10_a2)
            log10_high_free = pt.maximum(log10_a1, log10_a2)

        elif prior_kind_ in ("default", "soft"):
            # NUTS-friendly default priors:
            # - bounded via sigmoid (no hard walls)
            # - anisotropy ratio biased toward 1 (delta near 0)
            low_un = pm.Normal("log10_low_un", mu=0.0, sigma=1.0, shape=(nl,))
            log10_low_free = lo_a + (hi_a - lo_a) * pm.math.sigmoid(low_un)

            # delta in [0, max_delta], with most mass near 0 (prefers isotropy)
            delta_un = pm.Normal("log10_delta_un", mu=-1.0, sigma=1.0, shape=(nl,))
            log10_delta_free = max_delta * pm.math.sigmoid(delta_un)
            log10_high_free = log10_low_free + log10_delta_free

            # Expose delta for diagnostics/plots
            pm.Deterministic("log10_delta", log10_delta_free)

        else:
            raise ValueError(f"Unknown prior_kind: {prior_kind!r} (use 'default' or 'uniform').")

        # Apply isotropy: low == high
        log10_high_free = pt.switch(is_iso_data, log10_low_free, log10_high_free)

        # Apply per-layer fixed flags (in the chosen domain)
        tiny = np.finfo(float).tiny
        if domain_ == "rho":
            log10_low0 = np.log10(np.maximum(rmin0, tiny))
            log10_high0 = np.log10(np.maximum(rmax0, tiny))
        else:
            # sigma_max = 1/rho_min, sigma_min = 1/rho_max
            log10_low0 = np.log10(np.maximum(1.0 / np.maximum(rmax0, tiny), tiny))   # log10(sigma_min)
            log10_high0 = np.log10(np.maximum(1.0 / np.maximum(rmin0, tiny), tiny))  # log10(sigma_max)

        log10_low = pt.switch(is_fix_data, log10_low0, log10_low_free)
        log10_high = pt.switch(is_fix_data, log10_high0, log10_high_free)

        # Map to physical resistivities
        if domain_ == "rho":
            log10_rmin = log10_low
            log10_rmax = log10_high
            rho_min = pm.Deterministic("rho_min_ohmm", 10.0 ** log10_rmin)
            rho_max = pm.Deterministic("rho_max_ohmm", 10.0 ** log10_rmax)
        else:
            # In sigma-domain, keep ordered (sigma_min <= sigma_max) and convert.
            sigma_min = pm.Deterministic("sigma_min_Spm", 10.0 ** log10_low)
            sigma_max = pm.Deterministic("sigma_max_Spm", 10.0 ** log10_high)

            rho_min = pm.Deterministic("rho_min_ohmm", 1.0 / sigma_max)
            rho_max = pm.Deterministic("rho_max_ohmm", 1.0 / sigma_min)

            log10_rmin = -log10_high  # rho_min = 1/sigma_max
            log10_rmax = -log10_low   # rho_max = 1/sigma_min

        # Expose the ordered log10 resistivities explicitly for convenience
        pm.Deterministic("log10_rho_min", log10_rmin)
        pm.Deterministic("log10_rho_max", log10_rmax)


# Strike
        if prior_kind_ in ("default", "soft"):
            strike_un = pm.Normal("strike_deg_un", mu=0.0, sigma=1.0, shape=(nl,))
            strike_free = lo_s + (hi_s - lo_s) * pm.math.sigmoid(strike_un)
        else:
            strike_free = pm.Uniform("strike_deg_free", lower=lo_s, upper=hi_s, shape=(nl,))
        strike = pt.switch(is_fix_data, str0, strike_free)
        strike = pt.switch(is_iso_data, str0, strike)  # strike irrelevant for isotropic layers
        strike = pm.Deterministic("strike_deg", strike)

        # Thickness
        if spec.fix_h:
            h_m = pm.Deterministic("h_m", h0)
        else:
            if prior_kind_ in ("default", "soft"):
                log10_h_un = pm.Normal("log10_h_un", mu=0.0, sigma=1.0, shape=(nl,))
                log10_h_free = lo_h + (hi_h - lo_h) * pm.math.sigmoid(log10_h_un)
            else:
                log10_h_free = pm.Uniform("log10_h_free", lower=lo_h, upper=hi_h, shape=(nl,))
            if not spec.sample_last_thickness:
                # keep the last entry fixed at its starting value
                log10_h_free = pt.set_subtensor(
                    log10_h_free[-1],
                    np.log10(max(h0[-1], np.finfo(float).tiny)),
                )
            log10_h0 = np.log10(np.maximum(h0, np.finfo(float).tiny))
            log10_h = pt.switch(is_fix_data, log10_h0, log10_h_free)
            h_m = pm.Deterministic("h_m", 10.0 ** log10_h)

        # Forward model: return packed observation vector
        z_comps_ = tuple(z_comps)
        pt_comps_ = tuple(pt_comps)
        use_pt_ = bool(use_pt)

        if enable_grad:
            # Differentiable forward for NUTS/HMC (impedance + optional PT)
            if Op is None:
                raise ImportError("PyTensor Op class not available.")
            fop = _ForwardPackedOp(
                periods_s=periods_s,
                z_comps=z_comps_,
                use_pt=use_pt_,
                pt_comps=pt_comps_,
                                pt_reg=float(pt_reg),
                dh_rel=dh_rel,
            )
            y_pred = fop(h_m, rho_max, rho_min, strike)
        else:
            # Robust black-box forward (supports optional PT)
            if as_op is None:
                raise ImportError("pytensor.as_op not available")

            @as_op(itypes=[pt.dvector, pt.dvector, pt.dvector, pt.dvector], otypes=[pt.dvector])
            def forward_packed(h_m_v, rho_max_v, rho_min_v, strike_v):
                fres = aniso1d_impedance_sens_simple(
                    periods_s=periods_s,
                    h_m=np.asarray(h_m_v, dtype=float),
                    rho_max_ohmm=np.asarray(rho_max_v, dtype=float),
                    rho_min_ohmm=np.asarray(rho_min_v, dtype=float),
                    strike_deg=np.asarray(strike_v, dtype=float),
                    compute_sens=False,
                    dh_rel=dh_rel,
                )
                Zp = fres["Z"]
                ypZ = pack_Z_vector(Zp, comps=z_comps_)
                y_parts = [ypZ]

                if use_pt_:
                    Pp = phase_tensor_from_Z(Zp, reg=float(pt_reg))
                    ypP = pack_P_vector(Pp, comps=pt_comps_)
                    y_parts.append(ypP)

                return np.concatenate(y_parts).astype(float)

            y_pred = forward_packed(h_m, rho_max, rho_min, strike)

        # Packed parameter vector (handy for plotting)
        theta = pm.Deterministic(
            "theta",
            pt.concatenate(
                [
                    log10_rmin,
                    log10_rmax,
                    strike,
                    pt.log(h_m) / np.log(10.0),
                ]
            ),
        )

        pm.Normal("y", mu=y_pred, sigma=sigma, observed=y_obs)

    info = {
        "periods_s": periods_s,
        "z_comps": tuple(z_comps),
        "pt_comps": tuple(pt_comps),
        "use_pt": bool(use_pt),
        "pt_reg": float(pt_reg),
        "y_obs": y_obs,
        "sigma": sigma,
    }

    return model, info


def sample_pymc(
    model: "pm.Model",
    *,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    cores: int = 4,
    step_method: str = "demetropolis",
    target_accept: float = 0.85,
    random_seed: Optional[int] = None,
    progressbar: bool = True,
):
    """Run PyMC sampling.

    Parameters
    ----------
    model : pm.Model
        Built model.
    step_method : str
        One of {"demetropolis", "metropolis", "nuts", "hmc"}.

    Returns
    -------
    arviz.InferenceData
        Sampling results.
    """
    if pm is None:
        raise ImportError("PyMC not available.")

    step_method = str(step_method).lower().strip()
    with model:
        if step_method in ("demetropolis", "demetropolisz"):
            step = pm.DEMetropolisZ()
        elif step_method == "metropolis":
            step = pm.Metropolis()
        elif step_method == "nuts":
            step = pm.NUTS(target_accept=float(target_accept))
        elif step_method in ("hmc", "hamiltonian"):
            step = pm.HamiltonianMC()
        else:
            raise ValueError(f"Unknown step_method: {step_method}")

        idata = pm.sample(
            draws=int(draws),
            tune=int(tune),
            chains=int(chains),
            cores=int(cores),
            step=step,
            target_accept=float(target_accept),
            random_seed=random_seed,
            progressbar=bool(progressbar),
            return_inferencedata=True,
        )

    return idata


def save_idata(idata, path: str | os.PathLike) -> None:
    """Save an ArviZ InferenceData to NetCDF."""
    if az is None:
        raise ImportError("arviz not available.")
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(p.as_posix())


# -----------------------------------------------------------------------------
# Summary NPZ for plotting
# -----------------------------------------------------------------------------

def _stack_samples(var) -> np.ndarray:
    """Stack chain/draw dimensions into a single sample axis."""
    if hasattr(var, "stack"):
        v = var.stack(sample=("chain", "draw")).values
        return np.asarray(v)
    # fall back
    return np.asarray(var)


def _quantiles(x: np.ndarray, qs: Sequence[float]) -> np.ndarray:
    """Compute quantiles along axis 0."""
    return np.quantile(x, np.asarray(qs, dtype=float), axis=0)


def build_summary_npz(
    *,
    station: str,
    site: Dict,
    idata,
    spec: ParamSpec,
    model0: Dict,
    info: Dict,
    qpairs: Sequence[Tuple[float, float]] = ((0.1, 0.9), (0.25, 0.75)),
) -> Dict[str, np.ndarray]:
    """Build a compact NPZ summary for plotting.

    The resulting dict contains per-layer envelopes (quantile pairs) for
    ``rho_min_ohmm``, ``rho_max_ohmm`` and ``strike_deg``, plus a depth axis.

    Parameters
    ----------
    station : str
        Station name.
    site : dict
        Site dict.
    idata : arviz.InferenceData
        Sampling output.
    spec : ParamSpec
        Parameter spec.
    model0 : dict
        Starting model.
    info : dict
        Info returned by :func:`build_pymc_model`.
    qpairs : sequence of (float,float)
        Quantile pairs.

    Returns
    -------
    dict
        Serializable summary dict.
    """
    if az is None:
        raise ImportError("arviz not available.")

    m0 = normalize_model(model0)
    h0 = np.asarray(m0["h_m"], dtype=float)
    z = np.r_[0.0, np.cumsum(np.maximum(h0[:-1], 0.0))]

    # Extract posterior samples
    post = idata.posterior

    rmin_s = _stack_samples(post["rho_min_ohmm"])  # (ns, nl)
    rmax_s = _stack_samples(post["rho_max_ohmm"])  # (ns, nl)
    str_s = _stack_samples(post["strike_deg"])     # (ns, nl)
    theta_s = _stack_samples(post["theta"])        # (ns, ntheta)

    # Quantile pairs
    qpairs = tuple((float(a), float(b)) for a, b in qpairs)
    q_all = sorted(set([q for pair in qpairs for q in pair] + [0.5]))

    rmin_q = _quantiles(rmin_s, q_all)  # (nq, nl)
    rmax_q = _quantiles(rmax_s, q_all)
    str_q = _quantiles(str_s, q_all)

    # Theta qpairs for density plots
    theta_qpairs = np.array([[np.quantile(theta_s[:, i], [a, b]) for (a, b) in qpairs] for i in range(theta_s.shape[1])])

    out: Dict[str, np.ndarray] = {
        "station": np.array(str(station)),
        "periods_s": np.asarray(info.get("periods_s"), dtype=float),
        "h_m0": h0,
        "rho_min0": np.asarray(m0["rho_min_ohmm"], dtype=float),
        "rho_max0": np.asarray(m0["rho_max_ohmm"], dtype=float),
        "strike0": np.asarray(m0["strike_deg"], dtype=float),
        "z_m": z,
        "q": np.asarray(q_all, dtype=float),
        "rho_min_q": np.asarray(rmin_q, dtype=float),
        "rho_max_q": np.asarray(rmax_q, dtype=float),
        "strike_q": np.asarray(str_q, dtype=float),
        "qpairs": np.asarray(qpairs, dtype=float),
        "theta_qpairs": np.asarray(theta_qpairs, dtype=float),
    }

    return out


def save_summary_npz(summary: Dict, path: str | os.PathLike) -> None:
    """Save summary dict to NPZ."""
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    # NPZ does not like plain Python strings unless object dtype.
    out = {}
    for k, v in summary.items():
        if isinstance(v, str):
            out[k] = np.array(v)
        else:
            out[k] = v

    np.savez(p.as_posix(), **out)

