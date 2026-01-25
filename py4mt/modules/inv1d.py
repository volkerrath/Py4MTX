#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inv1d.py
========

Deterministic (non-Bayesian) inversion utilities for anisotropic 1-D MT.

This module implements a Gauss-Newton style deterministic inversion for a single
MT site, using the sensitivity-enabled forward model in :mod:`aniso`
(:func:`aniso.aniso1d_impedance_sens`).

At each iteration, we linearize the data functional around the current model:

    y(m + δm) ≈ y(m) + J δθ

where ``θ`` is the internal parameter vector (log-thickness, log-resistivities,
angles). The update ``δθ`` is computed from the weighted residual using one of:

- **TSVD** (truncated SVD) on the weighted Jacobian.
- **Tikhonov** regularization with first- or second-order difference matrices
  along depth (``L``), i.e.

    min || W (r - J δθ) ||^2  +  || λ L δθ ||^2

Layer-freezing is supported by a per-layer flag ``is_fix``. If ``is_fix[i]`` is
True, *all* parameters of that layer are held constant during inversion.

No Bayesian sampling code is used here.

Author: Volker Rath (DIAS)
Created with the help of ChatGPT (GPT-5 Thinking) on 2026-01-25
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

import aniso


# -----------------------------------------------------------------------------
# I/O helpers (kept minimal and explicit; do not import from the sampling module)
# -----------------------------------------------------------------------------

def ensure_dir(path: Union[str, Path]) -> str:
    """
    Ensure a directory exists and return its path as string.

    Parameters
    ----------
    path
        Directory path.

    Returns
    -------
    str
        Normalized directory path.
    """
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p.as_posix()


def glob_inputs(pattern: str) -> List[str]:
    """
    Expand a glob pattern to input files (sorted).

    Parameters
    ----------
    pattern
        Glob pattern, e.g. ``"/path/*.edi"`` or ``"/path/*.npz"``.

    Returns
    -------
    list of str
        Sorted list of matching files.
    """
    import glob
    return sorted([Path(p).expanduser().as_posix() for p in glob.glob(str(pattern))])


def load_site(path: Union[str, Path]) -> Dict[str, object]:
    """
    Load a site dictionary from EDI or NPZ.

    This uses :mod:`data_proc` (provided by the surrounding project environment).
    The site dict must at least provide:

    - ``freq`` : (nper,)
    - ``Z`` : (nper, 2, 2) complex

    Optionally:
    - ``Z_err`` : same shape as Z (std or var)
    - ``P`` and ``P_err`` : (nper,2,2) floats
    - ``err_kind`` : "std" or "var"
    - ``station`` : station name

    Parameters
    ----------
    path
        Input path (.edi or .npz).

    Returns
    -------
    dict
        Site dictionary.
    """
    p = Path(path).expanduser()
    suf = p.suffix.lower()
    if suf == ".edi":
        from data_proc import load_edi
        site = load_edi(p.as_posix())
    elif suf == ".npz":
        from data_proc import load_npz
        site = load_npz(p.as_posix())
    else:
        raise ValueError(f"Unsupported input type: {p}")
    if "station" not in site:
        site["station"] = p.stem
    return dict(site)


def ensure_phase_tensor(site: Mapping[str, object], *, nsim: int = 200) -> Dict[str, object]:
    """
    Ensure phase tensor (P) and its uncertainty are present in the site dict.

    If the site does not contain ``P``, it is computed from ``Z`` using
    :func:`data_proc.compute_pt` with bootstrap uncertainties.

    Parameters
    ----------
    site
        Site dictionary with at least ``Z``.
    nsim
        Number of bootstrap samples for P_err (passed to data_proc.compute_pt).

    Returns
    -------
    dict
        Copy of the input site dict with keys ``P`` and ``P_err`` ensured.
    """
    out = dict(site)
    if out.get("P", None) is not None:
        return out
    from data_proc import compute_pt
    Z = np.asarray(out["Z"])
    Z_err = out.get("Z_err", None)
    err_kind = str(out.get("err_kind", "var"))
    P, P_err = compute_pt(Z, Z_err, err_kind=err_kind, err_method="bootstrap", nsim=int(nsim))
    out["P"] = P
    out["P_err"] = P_err
    return out


def save_model_npz(model: Mapping[str, object], path: Union[str, Path]) -> None:
    """
    Save a model dict to compressed NPZ.

    The structure matches the project convention for ``model0.npz``:

    - prior_name (optional; stored as object array)
    - h_m, rop, ustr_deg, udip_deg, usla_deg, is_iso, is_fix

    Parameters
    ----------
    model
        Model dictionary.
    path
        Output file path (should end with .npz).

    Returns
    -------
    None
    """
    md = normalize_model(model)
    out: Dict[str, object] = {k: md[k] for k in ("h_m", "rop", "ustr_deg", "udip_deg", "usla_deg", "is_iso", "is_fix")}
    if "prior_name" in md:
        out["prior_name"] = np.array(md["prior_name"], dtype=object)
    np.savez_compressed(Path(path).expanduser().as_posix(), **out)


def load_model_npz(path: Union[str, Path]) -> Dict[str, object]:
    """
    Load a model dict from NPZ (project convention).

    Parameters
    ----------
    path
        Path to model NPZ.

    Returns
    -------
    dict
        Normalized model dictionary.
    """
    p = Path(path).expanduser()
    d = dict(np.load(p.as_posix(), allow_pickle=True))
    if "prior_name" in d:
        try:
            d["prior_name"] = str(np.asarray(d["prior_name"]).item())
        except Exception:
            d["prior_name"] = str(d["prior_name"])
    return normalize_model(d)


def model_from_direct(model_direct: Mapping[str, object]) -> Dict[str, object]:
    """
    Prepare an in-script model definition (python dict) for inversion.

    Parameters
    ----------
    model_direct
        Mapping with at least ``rop`` and preferably ``h_m`` and angle arrays.

    Returns
    -------
    dict
        Normalized model dict (numpy arrays).
    """
    return normalize_model(dict(model_direct))


# -----------------------------------------------------------------------------
# Parameterization utilities
# -----------------------------------------------------------------------------

class ParamSpec:
    """
    Parameter specification for deterministic inversion.

    The internal parameter vector θ is constructed from the public model arrays:

    - thicknesses (if not fixed): ``log10(h_m)`` for the first ``nh`` layers
    - resistivities: ``log10(rop)`` flattened (nl*3,)
    - angles: ``[ustr_deg, udip_deg, usla_deg]`` interleaved per layer (nl*3,)

    Parameters
    ----------
    nl
        Number of layers (including basement).
    fix_h
        If True, thicknesses are not inverted (held at model0 values).
    sample_last_thickness
        If False, the basement thickness (last element) is not included in θ.
    log10_h_bounds
        Bounds for log10(thickness in meters) when thickness is inverted.
    log10_rho_bounds
        Bounds for log10(resistivity in Ohm·m).
    ustr_bounds_deg, udip_bounds_deg, usla_bounds_deg
        Bounds (degrees) for Euler angles.
    """
    def __init__(
        self,
        *,
        nl: int,
        fix_h: bool,
        sample_last_thickness: bool,
        log10_h_bounds: Tuple[float, float],
        log10_rho_bounds: Tuple[float, float],
        ustr_bounds_deg: Tuple[float, float],
        udip_bounds_deg: Tuple[float, float],
        usla_bounds_deg: Tuple[float, float],
    ) -> None:
        self.nl = int(nl)
        if self.nl <= 0:
            raise ValueError("nl must be positive.")
        self.fix_h = bool(fix_h)
        self.sample_last_thickness = bool(sample_last_thickness)

        self.log10_h_bounds = (float(log10_h_bounds[0]), float(log10_h_bounds[1]))
        self.log10_rho_bounds = (float(log10_rho_bounds[0]), float(log10_rho_bounds[1]))
        self.ustr_bounds_deg = (float(ustr_bounds_deg[0]), float(ustr_bounds_deg[1]))
        self.udip_bounds_deg = (float(udip_bounds_deg[0]), float(udip_bounds_deg[1]))
        self.usla_bounds_deg = (float(usla_bounds_deg[0]), float(usla_bounds_deg[1]))

    def nh(self) -> int:
        """
        Number of thickness entries included in θ.

        Returns
        -------
        int
            0 if thickness is fixed; else nl (if sampling last thickness) or nl-1.
        """
        if self.fix_h:
            return 0
        return self.nl if self.sample_last_thickness else max(0, self.nl - 1)

    def ndim(self) -> int:
        """
        Total number of parameters in θ.

        Returns
        -------
        int
            Dimension of θ.
        """
        return self.nh() + 3 * self.nl + 3 * self.nl


def normalize_model(model: Mapping[str, object]) -> Dict[str, object]:
    """
    Normalize a model dictionary to the public anisotropic 1-D parameterization.

    Required keys / shapes after normalization:
    - h_m: (nl,)
    - rop: (nl,3)
    - ustr_deg, udip_deg, usla_deg: (nl,)
    - is_iso, is_fix: (nl,)

    Parameters
    ----------
    model
        Mapping containing at least ``rop``; other arrays may be missing or scalar.

    Returns
    -------
    dict
        Normalized model dictionary with numpy arrays.

    Raises
    ------
    ValueError
        If arrays cannot be made consistent.
    """
    md: Dict[str, object] = dict(model)

    # thickness
    h_key = "h_m" if "h_m" in md else ("h" if "h" in md else None)
    if h_key is not None and md.get(h_key, None) is not None:
        h_m = np.asarray(md[h_key], dtype=float).ravel()
    else:
        h_m = None

    # resistivities must exist
    if "rop" not in md or md.get("rop", None) is None:
        raise ValueError("Model must contain 'rop' (principal resistivities).")
    rop = np.asarray(md["rop"], dtype=float)
    if rop.ndim != 2:
        raise ValueError(f"rop must be 2-D, got shape {rop.shape}")

    # infer nl if needed
    if h_m is None:
        if rop.shape[1] == 3:
            nl = int(rop.shape[0])
        elif rop.shape[0] == 3:
            nl = int(rop.shape[1])
        else:
            raise ValueError("Cannot infer nl: rop must have one dimension of length 3.")
        h_m = np.ones(nl, dtype=float)
        h_m[-1] = 0.0
    else:
        nl = int(h_m.size)
        if nl <= 0:
            raise ValueError("h_m must have positive length.")

    # normalize rop to (nl,3)
    if rop.shape == (nl, 3):
        pass
    elif rop.shape == (3, nl):
        rop = rop.T
    else:
        raise ValueError(f"rop must have shape ({nl},3) or (3,{nl}), got {rop.shape}")

    def _norm_ang(key: str) -> np.ndarray:
        v = md.get(key, 0.0)
        a = np.asarray(v, dtype=float)
        if a.ndim == 0:
            return np.full(nl, float(a), dtype=float)
        a = a.ravel()
        if a.shape != (nl,):
            raise ValueError(f"{key} must have shape ({nl},), got {a.shape}")
        return a

    ustr_deg = _norm_ang("ustr_deg")
    udip_deg = _norm_ang("udip_deg")
    usla_deg = _norm_ang("usla_deg")

    def _norm_flag(key: str) -> np.ndarray:
        v = md.get(key, None)
        if v is None:
            return np.zeros(nl, dtype=bool)
        a = np.asarray(v, dtype=bool)
        if a.ndim == 0:
            return np.full(nl, bool(a), dtype=bool)
        a = a.ravel()
        if a.shape != (nl,):
            raise ValueError(f"{key} must have shape ({nl},), got {a.shape}")
        return a

    is_iso = _norm_flag("is_iso")
    is_fix = _norm_flag("is_fix")

    md["h_m"] = np.asarray(h_m, dtype=float)
    md["rop"] = np.asarray(rop, dtype=float)
    md["ustr_deg"] = np.asarray(ustr_deg, dtype=float)
    md["udip_deg"] = np.asarray(udip_deg, dtype=float)
    md["usla_deg"] = np.asarray(usla_deg, dtype=float)
    md["is_iso"] = np.asarray(is_iso, dtype=bool)
    md["is_fix"] = np.asarray(is_fix, dtype=bool)

    if "prior_name" in md and md["prior_name"] is not None:
        md["prior_name"] = str(md["prior_name"])
    return md


def theta_from_model(model: Mapping[str, object], *, spec: ParamSpec) -> np.ndarray:
    """
    Construct θ from a public model dictionary.

    Parameters
    ----------
    model
        Model dictionary (will be normalized).
    spec
        Parameter specification (controls which components are included).

    Returns
    -------
    ndarray
        Theta vector (spec.ndim(),).
    """
    md = normalize_model(model)
    nl = spec.nl

    h_m = np.asarray(md["h_m"], dtype=float)
    rop = np.asarray(md["rop"], dtype=float)
    ustr = np.asarray(md["ustr_deg"], dtype=float)
    udip = np.asarray(md["udip_deg"], dtype=float)
    usla = np.asarray(md["usla_deg"], dtype=float)

    parts: List[np.ndarray] = []

    if not spec.fix_h:
        nh = spec.nh()
        lo, hi = spec.log10_h_bounds
        h_clip = np.clip(h_m[:nh], 10.0 ** lo, 10.0 ** hi)
        parts.append(np.log10(h_clip))

    lo, hi = spec.log10_rho_bounds
    rop_clip = np.clip(rop, 10.0 ** lo, 10.0 ** hi)
    parts.append(np.log10(rop_clip).reshape(-1))

    parts.append(np.vstack([ustr, udip, usla]).T.reshape(-1))

    theta = np.concatenate(parts).astype(np.float64)
    if theta.size != spec.ndim():
        raise ValueError("theta size mismatch to spec.")
    return theta


def model_from_theta(theta: np.ndarray, *, spec: ParamSpec, model0: Mapping[str, object]) -> Dict[str, object]:
    """
    Convert θ to a public model dict, using model0 as base for fixed components.

    Parameters
    ----------
    theta
        Theta vector of length spec.ndim().
    spec
        Parameter specification.
    model0
        Reference model providing fixed values when needed.

    Returns
    -------
    dict
        Normalized model dictionary suitable for :func:`aniso.aniso1d_impedance_sens`.
    """
    md0 = normalize_model(model0)
    nl = spec.nl
    th = np.asarray(theta, dtype=np.float64).ravel()
    if th.size != spec.ndim():
        raise ValueError("theta size mismatch.")

    p = 0

    # thickness
    if spec.fix_h:
        h_m = np.asarray(md0["h_m"], dtype=float).copy()
    else:
        nh = spec.nh()
        log10_h = th[p:p + nh]
        p += nh
        h_m = np.asarray(md0["h_m"], dtype=float).copy()
        h_m[:nh] = 10.0 ** log10_h

    # resistivities
    log10_rho = th[p:p + 3 * nl].reshape((nl, 3))
    p += 3 * nl
    rop = 10.0 ** log10_rho

    # angles
    ang = th[p:].reshape((nl, 3))
    ustr = ang[:, 0].copy()
    udip = ang[:, 1].copy()
    usla = ang[:, 2].copy()

    out = dict(md0)
    out["h_m"] = np.asarray(h_m, dtype=float)
    out["rop"] = np.asarray(rop, dtype=float)
    out["ustr_deg"] = np.asarray(ustr, dtype=float)
    out["udip_deg"] = np.asarray(udip, dtype=float)
    out["usla_deg"] = np.asarray(usla, dtype=float)
    return normalize_model(out)


def clip_theta_to_bounds(theta: np.ndarray, *, spec: ParamSpec) -> np.ndarray:
    """
    Clip θ to hard bounds.

    Parameters
    ----------
    theta
        Theta vector (ndim,).
    spec
        Parameter specification.

    Returns
    -------
    ndarray
        Clipped theta copy.
    """
    th = np.asarray(theta, dtype=np.float64).copy()
    if th.size != spec.ndim():
        raise ValueError("theta size mismatch.")

    p = 0
    nl = spec.nl

    if not spec.fix_h:
        nh = spec.nh()
        lo, hi = spec.log10_h_bounds
        th[p:p + nh] = np.clip(th[p:p + nh], lo, hi)
        p += nh

    lo, hi = spec.log10_rho_bounds
    th[p:p + 3 * nl] = np.clip(th[p:p + 3 * nl], lo, hi)
    p += 3 * nl

    ang = th[p:].reshape((nl, 3))
    loS, hiS = spec.ustr_bounds_deg
    loD, hiD = spec.udip_bounds_deg
    loL, hiL = spec.usla_bounds_deg
    ang[:, 0] = np.clip(ang[:, 0], loS, hiS)
    ang[:, 1] = np.clip(ang[:, 1], loD, hiD)
    ang[:, 2] = np.clip(ang[:, 2], loL, hiL)
    th[p:] = ang.reshape(-1)
    return th


def free_theta_indices(*, spec: ParamSpec, is_fix: Optional[np.ndarray]) -> np.ndarray:
    """
    Compute indices of free (updatable) θ entries given per-layer ``is_fix``.

    Parameters
    ----------
    spec
        Parameter specification.
    is_fix
        Boolean array of shape (nl,). If True, the entire layer is frozen.

    Returns
    -------
    ndarray
        Sorted int indices into θ that are free.
    """
    nl = spec.nl
    if is_fix is None:
        fix = np.zeros(nl, dtype=bool)
    else:
        fix = np.asarray(is_fix, dtype=bool).ravel()
        if fix.shape != (nl,):
            raise ValueError(f"is_fix must have shape ({nl},), got {fix.shape}")

    idx: List[int] = []
    p = 0

    if not spec.fix_h:
        nh = spec.nh()
        for il in range(nh):
            if not bool(fix[il]):
                idx.append(p + il)
        p += nh

    for il in range(nl):
        if bool(fix[il]):
            continue
        base = p + 3 * il
        idx.extend([base, base + 1, base + 2])
    p += 3 * nl

    for il in range(nl):
        if bool(fix[il]):
            continue
        base = p + 3 * il
        idx.extend([base, base + 1, base + 2])

    return np.array(sorted(idx), dtype=np.int64)


# -----------------------------------------------------------------------------
# Data packing and uncertainty handling
# -----------------------------------------------------------------------------

def _parse_err_kind(site: Mapping[str, object]) -> str:
    """
    Parse uncertainty kind from a site dict.

    Parameters
    ----------
    site
        Site mapping possibly containing ``err_kind``.

    Returns
    -------
    str
        "std" or "var" (defaults to "std").
    """
    kind = str(site.get("err_kind", "std")).strip().lower()
    if kind not in ("std", "var"):
        kind = "std"
    return kind


def _err_to_std(err: np.ndarray, err_kind: str) -> np.ndarray:
    """
    Convert an uncertainty array to standard deviations.

    Parameters
    ----------
    err
        Uncertainty array.
    err_kind
        "std" or "var".

    Returns
    -------
    ndarray
        Standard deviation array.
    """
    e = np.asarray(err, dtype=np.float64)
    if err_kind == "var":
        e = np.sqrt(np.maximum(e, 0.0))
    return np.maximum(e, 0.0)


def _comp_indices(comps: Sequence[str]) -> List[Tuple[int, int]]:
    """
    Map component strings ("xx","xy","yx","yy") to tensor indices.

    Parameters
    ----------
    comps
        Component strings.

    Returns
    -------
    list
        List of (i,j) indices.
    """
    m = {"xx": (0, 0), "xy": (0, 1), "yx": (1, 0), "yy": (1, 1)}
    idx: List[Tuple[int, int]] = []
    for c in comps:
        cc = str(c).strip().lower()
        if cc not in m:
            raise ValueError(f"Unknown component '{c}'. Use xx, xy, yx, yy.")
        idx.append(m[cc])
    return idx


def phase_tensor_from_Z(Z: np.ndarray) -> np.ndarray:
    """
    Compute phase tensor P from complex impedance Z.

    Parameters
    ----------
    Z
        Complex impedance array (nper,2,2).

    Returns
    -------
    ndarray
        Phase tensor array (nper,2,2), float64.
    """
    Z = np.asarray(Z)
    X = np.real(Z)
    Y = np.imag(Z)
    nper = Z.shape[0]
    P = np.empty((nper, 2, 2), dtype=np.float64)
    for k in range(nper):
        P[k] = np.linalg.solve(X[k], Y[k])
    return P


def _pack_Z(Z: np.ndarray, comps: Sequence[str]) -> np.ndarray:
    """
    Pack impedance components into 1-D vector [Re, Im].

    Parameters
    ----------
    Z
        Complex impedance array (nper,2,2).
    comps
        Components to include.

    Returns
    -------
    ndarray
        Packed Z vector.
    """
    Z = np.asarray(Z)
    idx = _comp_indices(comps)
    nper = Z.shape[0]
    out = np.empty(nper * len(idx) * 2, dtype=np.float64)
    p = 0
    for k in range(nper):
        for (i, j) in idx:
            z = Z[k, i, j]
            out[p] = float(np.real(z))
            out[p + 1] = float(np.imag(z))
            p += 2
    return out


def _pack_P(P: np.ndarray, comps: Sequence[str]) -> np.ndarray:
    """
    Pack phase tensor components into 1-D vector.

    Parameters
    ----------
    P
        Phase tensor array (nper,2,2).
    comps
        Components to include.

    Returns
    -------
    ndarray
        Packed P vector.
    """
    P = np.asarray(P, dtype=np.float64)
    idx = _comp_indices(comps)
    nper = P.shape[0]
    out = np.empty(nper * len(idx), dtype=np.float64)
    p = 0
    for k in range(nper):
        for (i, j) in idx:
            out[p] = float(P[k, i, j])
            p += 1
    return out


def _pack_Z_std(Z_std: np.ndarray, comps: Sequence[str], floor: float) -> np.ndarray:
    """
    Pack Z standard deviations into same layout as :func:`_pack_Z`.

    Parameters
    ----------
    Z_std
        Standard deviations for Z (nper,2,2), real.
    comps
        Components to include.
    floor
        Minimum sigma.

    Returns
    -------
    ndarray
        Packed sigma vector.
    """
    idx = _comp_indices(comps)
    nper = Z_std.shape[0]
    out = np.empty(nper * len(idx) * 2, dtype=np.float64)
    p = 0
    for k in range(nper):
        for (i, j) in idx:
            s = max(float(Z_std[k, i, j]), float(floor))
            out[p] = s
            out[p + 1] = s
            p += 2
    return out


def _pack_P_std(P_std: np.ndarray, comps: Sequence[str], floor: float) -> np.ndarray:
    """
    Pack P standard deviations into same layout as :func:`_pack_P`.

    Parameters
    ----------
    P_std
        Standard deviations for P (nper,2,2), real.
    comps
        Components to include.
    floor
        Minimum sigma.

    Returns
    -------
    ndarray
        Packed sigma vector.
    """
    idx = _comp_indices(comps)
    nper = P_std.shape[0]
    out = np.empty(nper * len(idx), dtype=np.float64)
    p = 0
    for k in range(nper):
        for (i, j) in idx:
            out[p] = max(float(P_std[k, i, j]), float(floor))
            p += 1
    return out


# -----------------------------------------------------------------------------
# Regularization matrices
# -----------------------------------------------------------------------------

def build_L_difference(*, spec: ParamSpec, order: int = 1, include_thickness: bool = False) -> np.ndarray:
    """
    Build a dense finite-difference regularization matrix L in θ-space.

    Families regularized separately along depth:
    - log10(rop) per principal axis (3 series)
    - ustr, udip, usla (3 series)
    - optionally log10(h) (1 series, only if thickness is inverted)

    Order-1 rows implement:  x[i+1] - x[i]
    Order-2 rows implement:  x[i+2] - 2*x[i+1] + x[i]

    Parameters
    ----------
    spec
        Parameter specification (defines θ layout).
    order
        Difference order: 1 or 2.
    include_thickness
        If True, include differences for log10(thickness) if thickness is inverted.

    Returns
    -------
    ndarray
        L matrix (nreg, ndim).

    Raises
    ------
    ValueError
        If order is not 1 or 2.
    """
    order = int(order)
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2.")

    nl = spec.nl
    ndim = spec.ndim()
    nh = spec.nh()

    def nrows_series(n: int) -> int:
        return max(0, n - 1) if order == 1 else max(0, n - 2)

    nreg = 0
    if include_thickness and (not spec.fix_h):
        nreg += nrows_series(nh)
    nreg += 3 * nrows_series(nl)
    nreg += 3 * nrows_series(nl)

    L = np.zeros((nreg, ndim), dtype=np.float64)

    row = 0
    p = 0

    if not spec.fix_h:
        if include_thickness:
            if order == 1:
                for i in range(nh - 1):
                    L[row, p + i] = -1.0
                    L[row, p + i + 1] = 1.0
                    row += 1
            else:
                for i in range(nh - 2):
                    L[row, p + i] = 1.0
                    L[row, p + i + 1] = -2.0
                    L[row, p + i + 2] = 1.0
                    row += 1
        p += nh

    rop_base = p
    if order == 1:
        for comp in range(3):
            for i in range(nl - 1):
                L[row, rop_base + 3 * i + comp] = -1.0
                L[row, rop_base + 3 * (i + 1) + comp] = 1.0
                row += 1
    else:
        for comp in range(3):
            for i in range(nl - 2):
                L[row, rop_base + 3 * i + comp] = 1.0
                L[row, rop_base + 3 * (i + 1) + comp] = -2.0
                L[row, rop_base + 3 * (i + 2) + comp] = 1.0
                row += 1
    p += 3 * nl

    ang_base = p
    if order == 1:
        for comp in range(3):
            for i in range(nl - 1):
                L[row, ang_base + 3 * i + comp] = -1.0
                L[row, ang_base + 3 * (i + 1) + comp] = 1.0
                row += 1
    else:
        for comp in range(3):
            for i in range(nl - 2):
                L[row, ang_base + 3 * i + comp] = 1.0
                L[row, ang_base + 3 * (i + 1) + comp] = -2.0
                L[row, ang_base + 3 * (i + 2) + comp] = 1.0
                row += 1

    return L


# -----------------------------------------------------------------------------
# Linear solvers for Gauss-Newton updates
# -----------------------------------------------------------------------------

def solve_update_tsvd(*, Jw: np.ndarray, rw: np.ndarray, k: Optional[int] = None, rcond: Optional[float] = 1e-3) -> np.ndarray:
    """
    Solve a least-squares update using truncated SVD.

    Parameters
    ----------
    Jw
        Weighted Jacobian (ndata, npar).
    rw
        Weighted residual (ndata,).
    k
        Keep exactly k singular values (if provided).
    rcond
        If k is None: keep singular values s_i > rcond*s_0.

    Returns
    -------
    ndarray
        Update vector (npar,).
    """
    Jw = np.asarray(Jw, dtype=np.float64)
    rw = np.asarray(rw, dtype=np.float64).ravel()
    if Jw.shape[0] != rw.size:
        raise ValueError("Jw and rw dimension mismatch.")
    U, s, VT = np.linalg.svd(Jw, full_matrices=False)
    if s.size == 0:
        return np.zeros(Jw.shape[1], dtype=np.float64)
    if k is not None:
        kk = int(max(1, min(int(k), s.size)))
    else:
        if rcond is None:
            kk = int(s.size)
        else:
            thr = float(rcond) * float(s[0])
            kk = int(np.sum(s > thr))
            kk = max(1, kk)
    return (VT[:kk, :].T @ ((U[:, :kk].T @ rw) / s[:kk])).astype(np.float64)


def solve_update_tikhonov(*, Jw: np.ndarray, rw: np.ndarray, lam: float, L: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Solve a Tikhonov-regularized update in least-squares sense.

    Parameters
    ----------
    Jw
        Weighted Jacobian (ndata, npar).
    rw
        Weighted residual (ndata,).
    lam
        Regularization strength λ.
    L
        Regularization matrix (nreg, npar). If None, identity is used.

    Returns
    -------
    ndarray
        Update vector (npar,).
    """
    Jw = np.asarray(Jw, dtype=np.float64)
    rw = np.asarray(rw, dtype=np.float64).ravel()
    if Jw.shape[0] != rw.size:
        raise ValueError("Jw and rw dimension mismatch.")
    npar = int(Jw.shape[1])
    if L is None:
        L = np.eye(npar, dtype=np.float64)
    else:
        L = np.asarray(L, dtype=np.float64)
        if L.shape[1] != npar:
            raise ValueError("L must have same number of columns as Jw.")
    A = np.vstack([Jw, float(lam) * L])
    b = np.concatenate([rw, np.zeros(L.shape[0], dtype=np.float64)], axis=0)
    delta, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.asarray(delta, dtype=np.float64).ravel()



# -----------------------------------------------------------------------------
# Parameter-choice utilities (optimal TSVD truncation and Tikhonov λ selection)
# -----------------------------------------------------------------------------

def _safe_log(x: np.ndarray, eps: float = 1e-300) -> np.ndarray:
    """
    Safe natural log that avoids log(0).

    Parameters
    ----------
    x
        Input array.
    eps
        Floor value.

    Returns
    -------
    ndarray
        log(max(x, eps)).
    """
    return np.log(np.maximum(np.asarray(x, dtype=np.float64), float(eps)))


def _curvature_parametric(x: np.ndarray, y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Curvature κ of a parametric curve (x(t), y(t)) sampled on t.

    Uses finite-difference derivatives via np.gradient.

    Parameters
    ----------
    x, y
        Coordinates arrays (n,).
    t
        Parameter array (n,).

    Returns
    -------
    ndarray
        Curvature values (n,), with NaNs at ends possibly.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    t = np.asarray(t, dtype=np.float64).ravel()
    if x.size != y.size or x.size != t.size:
        raise ValueError("x, y, t must have same length.")
    if x.size < 3:
        return np.full_like(x, np.nan)

    dx = np.gradient(x, t)
    dy = np.gradient(y, t)
    ddx = np.gradient(dx, t)
    ddy = np.gradient(dy, t)

    num = np.abs(dx * ddy - dy * ddx)
    den = np.power(dx * dx + dy * dy, 1.5)
    den = np.maximum(den, 1e-300)
    return num / den


def _solve_tikhonov_normal(A: np.ndarray, b: np.ndarray, lam: float, L: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve (A^T A + (λ^2) L^T L) x = A^T b and return diagnostics.

    Parameters
    ----------
    A
        Matrix (m,n).
    b
        Vector (m,).
    lam
        Regularization strength λ.
    L
        Regularization matrix (p,n) or None for identity.

    Returns
    -------
    x, r, Lx, M
        x  : (n,) solution
        r  : (m,) residual (b - A x)
        Lx : (p,) seminorm vector (L x) or x if L is None
        M  : (n,n) normal matrix
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).ravel()
    if A.shape[0] != b.size:
        raise ValueError("A and b dimension mismatch.")

    m, n = A.shape
    if L is None:
        Lm = np.eye(n, dtype=np.float64)
    else:
        Lm = np.asarray(L, dtype=np.float64)
        if Lm.shape[1] != n:
            raise ValueError("L must have same number of columns as A.")
    alpha = float(lam) ** 2
    G = A.T @ A
    LT_L = Lm.T @ Lm
    M = G + alpha * LT_L
    rhs = A.T @ b

    try:
        x = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        x, *_ = np.linalg.lstsq(M, rhs, rcond=None)

    r = b - A @ x
    Lx = (Lm @ x) if (L is not None) else x
    return x, r, Lx, M


def select_tikhonov_lambda(
    *,
    A: np.ndarray,
    b: np.ndarray,
    L: Optional[np.ndarray],
    method: str = "gcv",
    lam_grid: Optional[np.ndarray] = None,
    ngrid: int = 40,
    lam_min: Optional[float] = None,
    lam_max: Optional[float] = None,
) -> Tuple[float, Dict[str, np.ndarray]]:
    """
    Select a Tikhonov regularization strength λ by L-curve, GCV, or ABIC.

    The linearized step is:
        min ||A x - b||^2 + ||λ L x||^2

    Parameters
    ----------
    A
        Weighted Jacobian (m,n).
    b
        Weighted residual (m,).
    L
        Regularization matrix (p,n) or None (identity).
    method
        One of: "lcurve", "gcv", "abic".
    lam_grid
        Explicit grid of λ values to test.
    ngrid
        If lam_grid is None: number of grid points.
    lam_min, lam_max
        Optional explicit λ range endpoints.

    Returns
    -------
    lam_star, diag
        Selected λ and a diagnostics dict with arrays:
        - "lam", "resnorm", "seminorm", plus criterion arrays depending on method.
    """
    method = str(method).strip().lower().replace("-", "")
    if method not in ("lcurve", "gcv", "abic"):
        raise ValueError("method must be one of: 'lcurve', 'gcv', 'abic'.")

    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).ravel()
    if A.shape[0] != b.size:
        raise ValueError("A and b dimension mismatch.")

    m, n = A.shape
    if lam_grid is None:
        # scale range from singular values of A
        try:
            s = np.linalg.svd(A, compute_uv=False)
            smax = float(s[0]) if s.size else 1.0
        except np.linalg.LinAlgError:
            smax = float(np.linalg.norm(A, ord=2)) if np.isfinite(np.linalg.norm(A)) else 1.0
        lo = smax * 1e-6
        hi = smax * 1e2
        if lam_min is not None:
            lo = float(lam_min)
        if lam_max is not None:
            hi = float(lam_max)
        lo = max(lo, 1e-12)
        hi = max(hi, lo * 1.01)
        lam = np.logspace(np.log10(lo), np.log10(hi), int(ngrid))
    else:
        lam = np.asarray(lam_grid, dtype=np.float64).ravel()
        lam = lam[np.isfinite(lam) & (lam > 0)]
        if lam.size < 3:
            raise ValueError("lam_grid must contain at least 3 positive finite values.")
        lam = np.unique(lam)

    resnorm = np.zeros(lam.size, dtype=np.float64)
    seminorm = np.zeros(lam.size, dtype=np.float64)
    gcv = np.full(lam.size, np.nan, dtype=np.float64)
    abic = np.full(lam.size, np.nan, dtype=np.float64)

    # precompute for GCV trace
    G = A.T @ A
    if L is None:
        Lm = np.eye(n, dtype=np.float64)
    else:
        Lm = np.asarray(L, dtype=np.float64)
        if Lm.shape[1] != n:
            raise ValueError("L must have same number of columns as A.")
    LT_L = Lm.T @ Lm
    rhs = A.T @ b

    for i, li in enumerate(lam):
        alpha = float(li) ** 2
        M = G + alpha * LT_L
        try:
            x = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            x, *_ = np.linalg.lstsq(M, rhs, rcond=None)

        r = b - A @ x
        res2 = float(r @ r)
        resnorm[i] = np.sqrt(res2)

        Lx = (Lm @ x) if (L is not None) else x
        sem2 = float(Lx @ Lx)
        seminorm[i] = np.sqrt(sem2)

        if method == "gcv":
            # tr(H) = tr( A^T A (A^T A + α L^T L)^-1 ) = tr( M^-1 G )
            try:
                MinvG = np.linalg.solve(M, G)
            except np.linalg.LinAlgError:
                MinvG, *_ = np.linalg.lstsq(M, G, rcond=None)
            trH = float(np.trace(MinvG))
            den = (float(m) - trH)
            gcv[i] = res2 / max(den * den, 1e-300)

        if method == "abic":
            # A practical ABIC-style score (up to irrelevant constants)
            # using α = λ^2 and L as prior precision operator.
            sign, logdet = np.linalg.slogdet(M)
            if sign <= 0:
                logdet = np.nan
            p = int(Lm.shape[0]) if (L is not None) else int(n)
            abic[i] = (
                float(m) * _safe_log(res2 / max(m, 1), eps=1e-300)
                + float(p) * _safe_log(sem2 / max(p, 1), eps=1e-300)
                + float(logdet)
                - float(n) * _safe_log(alpha, eps=1e-300)
            )

    diag: Dict[str, np.ndarray] = {"lam": lam, "resnorm": resnorm, "seminorm": seminorm}

    if method == "lcurve":
        t = _safe_log(lam)
        x = _safe_log(resnorm)
        y = _safe_log(seminorm)
        curv = _curvature_parametric(x, y, t)
        diag["lcurve_curv"] = curv
        # choose maximum curvature (ignore ends)
        j = int(np.nanargmax(curv))
        lam_star = float(lam[j])
    elif method == "gcv":
        diag["gcv"] = gcv
        j = int(np.nanargmin(gcv))
        lam_star = float(lam[j])
    else:
        diag["abic"] = abic
        j = int(np.nanargmin(abic))
        lam_star = float(lam[j])

    return lam_star, diag


def _tsvd_update_from_svd(U: np.ndarray, s: np.ndarray, VT: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    """
    Compute TSVD solution x_k from SVD(A)=U diag(s) V^T and rhs b.

    Parameters
    ----------
    U, s, VT
        SVD factors of A.
    b
        RHS vector.
    k
        Truncation rank.

    Returns
    -------
    ndarray
        x_k (n,).
    """
    k = int(max(1, min(int(k), s.size)))
    Ub = U.T @ b
    coeff = Ub[:k] / s[:k]
    return (VT[:k, :].T @ coeff).astype(np.float64)


def select_tsvd_k(
    *,
    A: np.ndarray,
    b: np.ndarray,
    method: str = "gcv",
    k_min: int = 1,
    k_max: Optional[int] = None,
) -> Tuple[int, Dict[str, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Select TSVD truncation rank k by GCV or L-curve.

    Parameters
    ----------
    A
        Weighted Jacobian (m,n).
    b
        Weighted residual (m,).
    method
        "gcv" or "lcurve".
    k_min
        Minimum k to consider.
    k_max
        Maximum k to consider (defaults to min(m,n)).

    Returns
    -------
    k_star, diag, svd
        Selected k, diagnostics dict, and SVD tuple (U,s,VT) for reuse.
    """
    method = str(method).strip().lower().replace("-", "")
    if method not in ("gcv", "lcurve"):
        raise ValueError("method must be 'gcv' or 'lcurve'.")

    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).ravel()
    if A.shape[0] != b.size:
        raise ValueError("A and b dimension mismatch.")
    m, n = A.shape

    U, s, VT = np.linalg.svd(A, full_matrices=False)
    r = int(s.size)
    if r == 0:
        return 0, {"k": np.array([], dtype=int)}, (U, s, VT)

    kmin = int(max(1, min(int(k_min), r)))
    kmax = int(r if k_max is None else max(kmin, min(int(k_max), r)))

    ks = np.arange(kmin, kmax + 1, dtype=int)
    Ub = U.T @ b
    Ub2 = Ub * Ub

    # residual norm^2 for truncation k: sum_{i>k} (Ub_i)^2
    # build cumulative sums
    tail = np.cumsum(Ub2[::-1])[::-1]  # tail[i] = sum_{j>=i} Ub2[j]
    res2 = np.array([float(tail[k]) if (k < r) else 0.0 for k in ks], dtype=np.float64)

    # solution norm^2: sum_{i<=k} (Ub_i/s_i)^2
    ratio2 = (Ub[:r] / np.maximum(s, 1e-300)) ** 2
    head = np.cumsum(ratio2)
    sol2 = np.array([float(head[k - 1]) for k in ks], dtype=np.float64)

    resnorm = np.sqrt(res2)
    solnorm = np.sqrt(sol2)

    diag: Dict[str, np.ndarray] = {"k": ks, "resnorm": resnorm, "solnorm": solnorm}

    if method == "gcv":
        gcv = np.zeros(ks.size, dtype=np.float64)
        for i, k in enumerate(ks):
            den = float(m - k)
            gcv[i] = res2[i] / max(den * den, 1e-300)
        diag["gcv"] = gcv
        j = int(np.argmin(gcv))
        k_star = int(ks[j])
    else:
        t = ks.astype(np.float64)
        x = _safe_log(resnorm)
        y = _safe_log(solnorm)
        curv = _curvature_parametric(x, y, t)
        diag["lcurve_curv"] = curv
        j = int(np.nanargmax(curv))
        k_star = int(ks[j])

    return k_star, diag, (U, s, VT)


# -----------------------------------------------------------------------------
# Core forward/linearization and inversion
# -----------------------------------------------------------------------------

def _pack_prediction(*, Z: np.ndarray, use_pt: bool, z_comps: Sequence[str], pt_comps: Sequence[str]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Pack predicted data vector from impedance and optional phase tensor.

    Parameters
    ----------
    Z
        Complex impedance prediction (nper,2,2).
    use_pt
        Whether to include phase tensor.
    z_comps, pt_comps
        Component selections.

    Returns
    -------
    y_pred, P_pred
        Packed predicted vector and predicted phase tensor (or None).
    """
    yZ = _pack_Z(Z, z_comps)
    if not use_pt:
        return yZ, None
    P = phase_tensor_from_Z(Z)
    yP = _pack_P(P, pt_comps)
    return np.concatenate([yZ, yP], axis=0), P


def _pack_observations_and_sigma(
    site: Mapping[str, object],
    *,
    use_pt: bool,
    z_comps: Sequence[str],
    pt_comps: Sequence[str],
    compute_pt_if_missing: bool,
    sigma_floor_Z: float,
    sigma_floor_P: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Build observed data vector y and its sigma vector s.

    Parameters
    ----------
    site
        Site dict containing Z and optional P plus error arrays.
    use_pt
        Include phase tensor components.
    z_comps, pt_comps
        Component selections.
    compute_pt_if_missing
        If True, compute P from Z if missing.
    sigma_floor_Z, sigma_floor_P
        Floors for standard deviations.

    Returns
    -------
    y, s, yZ, P
        Observation vector, sigma vector, packed Z vector, and P (if used).
    """
    if "Z" not in site:
        raise KeyError("site must contain key 'Z'.")
    Z = np.asarray(site["Z"])
    nper = Z.shape[0]
    if Z.shape != (nper, 2, 2):
        raise ValueError(f"Z must have shape (nper,2,2), got {Z.shape}")

    err_kind = _parse_err_kind(site)
    Z_err = site.get("Z_err", None)
    if Z_err is None:
        Z_std = np.ones_like(Z, dtype=np.float64) * float(sigma_floor_Z)
    else:
        Z_std = _err_to_std(np.asarray(Z_err), err_kind)
        if Z_std.shape != Z.shape:
            raise ValueError("Z_err must have same shape as Z.")
        Z_std = np.maximum(Z_std, float(sigma_floor_Z))

    yZ = _pack_Z(Z, z_comps)
    sZ = _pack_Z_std(Z_std, z_comps, float(sigma_floor_Z))

    if not use_pt:
        return yZ, sZ, yZ, None

    if "P" in site and site.get("P", None) is not None:
        P = np.asarray(site["P"], dtype=np.float64)
        if P.shape != (nper, 2, 2):
            raise ValueError("P must have shape (nper,2,2).")
    else:
        if not compute_pt_if_missing:
            raise KeyError("use_pt=True but P missing and compute_pt_if_missing=False.")
        P = phase_tensor_from_Z(Z)

    P_err = site.get("P_err", None)
    if P_err is None:
        P_std = np.ones_like(P, dtype=np.float64) * float(sigma_floor_P)
    else:
        P_std = _err_to_std(np.asarray(P_err), err_kind)
        if P_std.shape != P.shape:
            raise ValueError("P_err must have same shape as P.")
        P_std = np.maximum(P_std, float(sigma_floor_P))

    yP = _pack_P(P, pt_comps)
    sP = _pack_P_std(P_std, pt_comps, float(sigma_floor_P))

    return np.concatenate([yZ, yP], axis=0), np.concatenate([sZ, sP], axis=0), yZ, P


def _pack_dZ_like_y(*, Zp: np.ndarray, dZ: np.ndarray, use_pt: bool, z_comps: Sequence[str], pt_comps: Sequence[str]) -> np.ndarray:
    """
    Pack a complex impedance derivative into the same layout as y.

    Parameters
    ----------
    Zp
        Current predicted impedance (nper,2,2).
    dZ
        Complex derivative dZ/dq (nper,2,2).
    use_pt
        Whether y includes phase tensor components.
    z_comps, pt_comps
        Component selections.

    Returns
    -------
    ndarray
        Packed derivative vector dy/dq.
    """
    nper = Zp.shape[0]
    idxZ = _comp_indices(z_comps)

    dyZ = np.empty(nper * len(idxZ) * 2, dtype=np.float64)
    p = 0
    for k in range(nper):
        for (i, j) in idxZ:
            dz = dZ[k, i, j]
            dyZ[p] = float(np.real(dz))
            dyZ[p + 1] = float(np.imag(dz))
            p += 2

    if not use_pt:
        return dyZ

    Pp = phase_tensor_from_Z(Zp)
    idxP = _comp_indices(pt_comps)
    dyP = np.empty(nper * len(idxP), dtype=np.float64)
    q = 0
    for k in range(nper):
        X = np.real(Zp[k])
        A = np.linalg.inv(X)
        dX = np.real(dZ[k])
        dY = np.imag(dZ[k])
        dP = -A @ dX @ Pp[k] + A @ dY
        for (i, j) in idxP:
            dyP[q] = float(dP[i, j])
            q += 1

    return np.concatenate([dyZ, dyP], axis=0)


def misfit_phi(r: np.ndarray, s: np.ndarray) -> float:
    """
    Weighted least-squares misfit φ = ||diag(1/s) r||^2.

    Parameters
    ----------
    r
        Residual vector.
    s
        Sigma vector.

    Returns
    -------
    float
        Misfit value.
    """
    rr = np.asarray(r, dtype=np.float64).ravel()
    ss = np.asarray(s, dtype=np.float64).ravel()
    if rr.size != ss.size:
        raise ValueError("r and s size mismatch.")
    w = 1.0 / np.maximum(ss, 1e-30)
    rw = w * rr
    return float(np.sum(rw * rw))


def invert_site(
    site: Mapping[str, object],
    *,
    spec: ParamSpec,
    model0: Mapping[str, object],
    method: str = "tikhonov",
    lam: float = 1.0,
    lam_select: str = "fixed",
    lam_grid: Optional[np.ndarray] = None,
    lam_ngrid: int = 40,
    lam_min: Optional[float] = None,
    lam_max: Optional[float] = None,
    reg_order: int = 1,
    include_thickness_in_L: bool = False,
    tsvd_k: Optional[int] = None,
    tsvd_select: str = "fixed",
    tsvd_k_min: int = 1,
    tsvd_k_max: Optional[int] = None,
    tsvd_rcond: Optional[float] = 1e-3,
    max_iter: int = 15,
    tol: float = 1e-3,
    step_scale: float = 1.0,
    use_pt: bool = False,
    z_comps: Sequence[str] = ("xy", "yx"),
    pt_comps: Sequence[str] = ("xx", "xy", "yx", "yy"),
    compute_pt_if_missing: bool = True,
    sigma_floor_Z: float = 0.0,
    sigma_floor_P: float = 0.0,
) -> Dict[str, object]:
    """
    Deterministic Gauss-Newton inversion for a single site.

    Results are saved in a single NPZ per site by :func:`save_inversion_npz`.
    The *final model* is stored with the same keys as ``model0.npz``.

    Parameters
    ----------
    site
        Site dictionary with at least freq, Z.
    spec
        Parameter specification.
    model0
        Starting/reference model.
    method
        "tikhonov" or "tsvd".
    lam
        Regularization strength (Tikhonov).
    reg_order
        1 or 2 for first/second order difference L (Tikhonov).
    include_thickness_in_L
        If True, regularize log10(thickness) when thickness is inverted.
    tsvd_k, tsvd_rcond
        TSVD truncation controls.
    max_iter
        Maximum number of Gauss-Newton iterations.
    tol
        Relative misfit reduction tolerance.
    step_scale
        Initial step length used in a simple backtracking line-search.
    use_pt
        Include phase tensor.
    z_comps, pt_comps
        Component selections.
    compute_pt_if_missing
        Compute P from Z when missing.
    sigma_floor_Z, sigma_floor_P
        Sigma floors.

    Returns
    -------
    dict
        NPZ-ready results.
    """
    method = str(method).strip().lower()
    if method not in ("tikhonov", "tsvd"):
        raise ValueError("method must be 'tikhonov' or 'tsvd'")

    md0 = normalize_model(model0)
    nl = spec.nl
    is_fix = np.asarray(md0.get("is_fix", np.zeros(nl, dtype=bool)), dtype=bool)
    is_iso = np.asarray(md0.get("is_iso", None), dtype=bool) if "is_iso" in md0 else None

    theta = clip_theta_to_bounds(theta_from_model(md0, spec=spec), spec=spec)
    free_idx = free_theta_indices(spec=spec, is_fix=is_fix)

    L_full = None
    L_free = None
    if method == "tikhonov":
        L_full = build_L_difference(spec=spec, order=int(reg_order), include_thickness=bool(include_thickness_in_L))
        if L_full.size and free_idx.size:
            L_free = L_full[:, free_idx]

    phi_hist: List[float] = []
    step_hist: List[float] = []
    lam_hist: List[float] = []
    tsvd_k_hist: List[int] = []
    lam_diag_last: Optional[Dict[str, np.ndarray]] = None
    tsvd_diag_last: Optional[Dict[str, np.ndarray]] = None
    converged = False

    freq = np.asarray(site["freq"], dtype=np.float64).ravel()
    periods_s = 1.0 / freq

    # Build observation vector once (shape defines ndata)
    y_obs, s_obs, _, P_obs = _pack_observations_and_sigma(
        site,
        use_pt=bool(use_pt),
        z_comps=z_comps,
        pt_comps=pt_comps,
        compute_pt_if_missing=bool(compute_pt_if_missing),
        sigma_floor_Z=float(sigma_floor_Z),
        sigma_floor_P=float(sigma_floor_P),
    )

    for _it in range(int(max_iter)):
        # forward + sensitivities
        mcur = model_from_theta(theta, spec=spec, model0=md0)

        fwd = aniso.aniso1d_impedance_sens(
            periods_s,
            np.asarray(mcur["h_m"], dtype=float),
            np.asarray(mcur["rop"], dtype=float),
            np.asarray(mcur["ustr_deg"], dtype=float),
            np.asarray(mcur["udip_deg"], dtype=float),
            np.asarray(mcur["usla_deg"], dtype=float),
            is_iso=None if is_iso is None else np.asarray(is_iso, dtype=bool),
            compute_sens=True,
        )
        Zp = np.asarray(fwd["Z"])
        y_pred, Pp = _pack_prediction(Z=Zp, use_pt=bool(use_pt), z_comps=z_comps, pt_comps=pt_comps)
        r = y_obs - y_pred
        phi = misfit_phi(r, s_obs)
        phi_hist.append(phi)

        if free_idx.size == 0:
            converged = True
            break

        # Jacobian assembly
        ndata = y_obs.size
        J = np.zeros((ndata, spec.ndim()), dtype=np.float64)
        p = 0

        if not spec.fix_h:
            nh = spec.nh()
            dZ_dh = np.asarray(fwd.get("dZ_dh_m"))
            for il in range(nh):
                dy = _pack_dZ_like_y(Zp=Zp, dZ=dZ_dh[:, il, :, :], use_pt=bool(use_pt), z_comps=z_comps, pt_comps=pt_comps)
                J[:, p + il] = dy * (np.log(10.0) * float(mcur["h_m"][il]))
            p += nh

        dZ_drop = np.asarray(fwd.get("dZ_drop"))
        for il in range(nl):
            for ir in range(3):
                dy = _pack_dZ_like_y(Zp=Zp, dZ=dZ_drop[:, il, ir, :, :], use_pt=bool(use_pt), z_comps=z_comps, pt_comps=pt_comps)
                J[:, p + 3 * il + ir] = dy * (np.log(10.0) * float(mcur["rop"][il, ir]))
        p += 3 * nl

        dZ_ustr = np.asarray(fwd.get("dZ_dustr_deg"))
        dZ_udip = np.asarray(fwd.get("dZ_dudip_deg"))
        dZ_usla = np.asarray(fwd.get("dZ_dusla_deg"))
        for il in range(nl):
            J[:, p + 3 * il + 0] = _pack_dZ_like_y(Zp=Zp, dZ=dZ_ustr[:, il, :, :], use_pt=bool(use_pt), z_comps=z_comps, pt_comps=pt_comps)
            J[:, p + 3 * il + 1] = _pack_dZ_like_y(Zp=Zp, dZ=dZ_udip[:, il, :, :], use_pt=bool(use_pt), z_comps=z_comps, pt_comps=pt_comps)
            J[:, p + 3 * il + 2] = _pack_dZ_like_y(Zp=Zp, dZ=dZ_usla[:, il, :, :], use_pt=bool(use_pt), z_comps=z_comps, pt_comps=pt_comps)

        # reduce to free parameters + weighting
        w = 1.0 / np.maximum(s_obs, 1e-30)
        rw = w * r
        Jw = w[:, None] * J[:, free_idx]

        # choose regularization / truncation for this linearized step
        lam_it = float(lam)
        k_it: Optional[int] = None

        if method == "tsvd":
            if (tsvd_select.strip().lower() != "fixed") or (tsvd_k is None):
                k_it, tsvd_diag_last, (U, s, VT) = select_tsvd_k(
                    A=Jw,
                    b=rw,
                    method=str(tsvd_select),
                    k_min=int(tsvd_k_min),
                    k_max=None if tsvd_k_max is None else int(tsvd_k_max),
                )
                delta_free = _tsvd_update_from_svd(U, s, VT, rw, k_it)
            else:
                k_it = int(tsvd_k)
                delta_free = solve_update_tsvd(Jw=Jw, rw=rw, k=k_it, rcond=tsvd_rcond)
            tsvd_k_hist.append(int(k_it))
        else:
            if str(lam_select).strip().lower() != "fixed":
                lam_it, lam_diag_last = select_tikhonov_lambda(
                    A=Jw,
                    b=rw,
                    L=L_free,
                    method=str(lam_select),
                    lam_grid=lam_grid,
                    ngrid=int(lam_ngrid),
                    lam_min=lam_min,
                    lam_max=lam_max,
                )
            delta_free = solve_update_tikhonov(Jw=Jw, rw=rw, lam=float(lam_it), L=L_free)
            lam_hist.append(float(lam_it))

        delta = np.zeros(spec.ndim(), dtype=np.float64)
        delta[free_idx] = delta_free

        # backtracking line-search
        alpha = float(step_scale)
        accepted = False
        for _ls in range(8):
            th_try = clip_theta_to_bounds(theta + alpha * delta, spec=spec)
            if free_idx.size != spec.ndim():
                mask_fixed = np.ones(spec.ndim(), dtype=bool)
                mask_fixed[free_idx] = False
                th_try[mask_fixed] = theta[mask_fixed]

            mtry = model_from_theta(th_try, spec=spec, model0=md0)
            ftry = aniso.aniso1d_impedance_sens(
                periods_s,
                np.asarray(mtry["h_m"], dtype=float),
                np.asarray(mtry["rop"], dtype=float),
                np.asarray(mtry["ustr_deg"], dtype=float),
                np.asarray(mtry["udip_deg"], dtype=float),
                np.asarray(mtry["usla_deg"], dtype=float),
                is_iso=None if is_iso is None else np.asarray(is_iso, dtype=bool),
                compute_sens=False,
            )
            Ztry = np.asarray(ftry["Z"])
            y_try, _ = _pack_prediction(Z=Ztry, use_pt=bool(use_pt), z_comps=z_comps, pt_comps=pt_comps)
            phi_try = misfit_phi(y_obs - y_try, s_obs)
            if phi_try <= phi:
                theta = th_try
                accepted = True
                break
            alpha *= 0.5

        step_hist.append(alpha if accepted else 0.0)

        if not accepted:
            converged = True
            break

        if len(phi_hist) >= 2:
            rel = abs(phi_hist[-2] - phi_hist[-1]) / max(phi_hist[-2], 1e-12)
            if rel < float(tol):
                converged = True
                break

    # final model & prediction
    model_hat = model_from_theta(theta, spec=spec, model0=md0)
    f_final = aniso.aniso1d_impedance_sens(
        periods_s,
        np.asarray(model_hat["h_m"], dtype=float),
        np.asarray(model_hat["rop"], dtype=float),
        np.asarray(model_hat["ustr_deg"], dtype=float),
        np.asarray(model_hat["udip_deg"], dtype=float),
        np.asarray(model_hat["usla_deg"], dtype=float),
        is_iso=None if is_iso is None else np.asarray(is_iso, dtype=bool),
        compute_sens=False,
    )
    Z_pred = np.asarray(f_final["Z"])
    P_pred = phase_tensor_from_Z(Z_pred) if bool(use_pt) else None
    y_pred_final, _ = _pack_prediction(Z=Z_pred, use_pt=bool(use_pt), z_comps=z_comps, pt_comps=pt_comps)

    # assemble result
    res: Dict[str, object] = {}
    res["station"] = str(site.get("station", ""))
    res["method"] = method
    res["converged"] = bool(converged)
    res["niter"] = int(len(phi_hist))
    res["phi_hist"] = np.asarray(phi_hist, dtype=np.float64)
    res["step_hist"] = np.asarray(step_hist, dtype=np.float64)

    # store data + prediction
    res["freq"] = freq
    res["Z"] = np.asarray(site["Z"])
    if "Z_err" in site and site["Z_err"] is not None:
        res["Z_err"] = np.asarray(site["Z_err"])
    res["err_kind"] = str(site.get("err_kind", "std"))

    if bool(use_pt):
        if P_obs is not None:
            res["P"] = np.asarray(P_obs, dtype=np.float64)
        if "P_err" in site and site["P_err"] is not None:
            res["P_err"] = np.asarray(site["P_err"], dtype=np.float64)
        if P_pred is not None:
            res["P_pred"] = np.asarray(P_pred, dtype=np.float64)

    res["Z_pred"] = np.asarray(Z_pred)

    # store inversion parameters
    res["fix_h"] = bool(spec.fix_h)
    res["sample_last_thickness"] = bool(spec.sample_last_thickness)
    res["log10_h_bounds"] = np.asarray(spec.log10_h_bounds, dtype=np.float64)
    res["log10_rho_bounds"] = np.asarray(spec.log10_rho_bounds, dtype=np.float64)
    res["ustr_bounds_deg"] = np.asarray(spec.ustr_bounds_deg, dtype=np.float64)
    res["udip_bounds_deg"] = np.asarray(spec.udip_bounds_deg, dtype=np.float64)
    res["usla_bounds_deg"] = np.asarray(spec.usla_bounds_deg, dtype=np.float64)

    res["use_pt"] = bool(use_pt)
    res["z_comps"] = np.array(list(map(str, z_comps)), dtype=object)
    res["pt_comps"] = np.array(list(map(str, pt_comps)), dtype=object)
    res["compute_pt_if_missing"] = bool(compute_pt_if_missing)
    res["sigma_floor_Z"] = float(sigma_floor_Z)
    res["sigma_floor_P"] = float(sigma_floor_P)

    res["lam"] = float(lam)
    res["lam_select"] = np.array(str(lam_select), dtype=object)
    res["lam_ngrid"] = int(lam_ngrid)
    res["lam_min"] = -1.0 if lam_min is None else float(lam_min)
    res["lam_max"] = -1.0 if lam_max is None else float(lam_max)
    res["lam_hist"] = np.asarray(lam_hist, dtype=np.float64) if len(lam_hist) else np.asarray([], dtype=np.float64)
    if lam_diag_last is not None:
        res["lam_grid_last"] = np.asarray(lam_diag_last.get("lam", []), dtype=np.float64)
        res["lam_resnorm_last"] = np.asarray(lam_diag_last.get("resnorm", []), dtype=np.float64)
        res["lam_seminorm_last"] = np.asarray(lam_diag_last.get("seminorm", []), dtype=np.float64)
        if "gcv" in lam_diag_last:
            res["lam_gcv_last"] = np.asarray(lam_diag_last.get("gcv", []), dtype=np.float64)
        if "abic" in lam_diag_last:
            res["lam_abic_last"] = np.asarray(lam_diag_last.get("abic", []), dtype=np.float64)
        if "lcurve_curv" in lam_diag_last:
            res["lam_lcurve_curv_last"] = np.asarray(lam_diag_last.get("lcurve_curv", []), dtype=np.float64)

    res["reg_order"] = int(reg_order)
    res["include_thickness_in_L"] = bool(include_thickness_in_L)

    res["tsvd_k"] = -1 if tsvd_k is None else int(tsvd_k)
    res["tsvd_select"] = np.array(str(tsvd_select), dtype=object)
    res["tsvd_k_min"] = int(tsvd_k_min)
    res["tsvd_k_max"] = -1 if tsvd_k_max is None else int(tsvd_k_max)
    res["tsvd_k_hist"] = np.asarray(tsvd_k_hist, dtype=np.int64) if len(tsvd_k_hist) else np.asarray([], dtype=np.int64)
    if tsvd_diag_last is not None:
        res["tsvd_k_grid_last"] = np.asarray(tsvd_diag_last.get("k", []), dtype=np.int64)
        res["tsvd_resnorm_last"] = np.asarray(tsvd_diag_last.get("resnorm", []), dtype=np.float64)
        res["tsvd_solnorm_last"] = np.asarray(tsvd_diag_last.get("solnorm", []), dtype=np.float64)
        if "gcv" in tsvd_diag_last:
            res["tsvd_gcv_last"] = np.asarray(tsvd_diag_last.get("gcv", []), dtype=np.float64)
        if "lcurve_curv" in tsvd_diag_last:
            res["tsvd_lcurve_curv_last"] = np.asarray(tsvd_diag_last.get("lcurve_curv", []), dtype=np.float64)

    res["tsvd_rcond"] = -1.0 if tsvd_rcond is None else float(tsvd_rcond)
    res["max_iter"] = int(max_iter)
    res["tol"] = float(tol)
    res["step_scale"] = float(step_scale)

    # packed vectors for diagnostics
    res["y_obs"] = np.asarray(y_obs, dtype=np.float64)
    res["y_pred"] = np.asarray(y_pred_final, dtype=np.float64)
    res["sigma"] = np.asarray(s_obs, dtype=np.float64)
    res["theta_hat"] = np.asarray(theta, dtype=np.float64)
    res["free_idx"] = np.asarray(free_idx, dtype=np.int64)

    # model0 (prefixed) and final model (exact keys)
    if "prior_name" in md0:
        res["model0_prior_name"] = np.array(str(md0["prior_name"]), dtype=object)
        res["prior_name"] = np.array(str(md0["prior_name"]), dtype=object)

    for k in ("h_m", "rop", "ustr_deg", "udip_deg", "usla_deg", "is_iso", "is_fix"):
        res[f"model0_{k}"] = np.asarray(md0[k])
        res[k] = np.asarray(model_hat[k])

    return res


def save_inversion_npz(result: Mapping[str, object], path: Union[str, Path]) -> None:
    """
    Save an inversion result dictionary to compressed NPZ.

    Parameters
    ----------
    result
        Result dict from :func:`invert_site`.
    path
        Output path (.npz).

    Returns
    -------
    None
    """
    p = Path(path).expanduser()
    np.savez_compressed(p.as_posix(), **dict(result))
