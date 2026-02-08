#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""inv1d.py
================

Deterministic 1-D inversion helpers for the *simplified* anisotropic MT model.

This module provides a small Gauss–Newton loop that uses analytic sensitivities
from :func:`aniso.aniso1d_impedance_sens_simple` to build a Jacobian for the
observed impedance (and optional phase tensor) data.

The focus here is pragmatic script support, not a full-featured production
inversion package.

Key design points
-----------------
- Parameterization per layer:

  - ``h_m``          thickness (optional inversion)
  - ``rho_min_ohmm`` minimum horizontal resistivity
  - ``rho_max_ohmm`` maximum horizontal resistivity
  - ``strike_deg``   anisotropy strike

- Two boolean masks may be supplied:

  - ``is_iso`` : layer is isotropic (forces rho_max == rho_min and ignores strike)
  - ``is_fix`` : layer is fixed (all layer parameters are held constant)

- Regularization:

  The interface accepts Tikhonov/TSVD style options for compatibility with
  earlier scripts. The current implementation supports:

  - ``method='tikhonov'`` with a simple *identity* damping (ridge).
  - ``method='tsvd'`` with optional truncation rank.

  More advanced selection modes (GCV/L-curve/ABIC) are accepted but currently
  fall back to the user-provided fixed values.

Author: Volker Rath (DIAS)
Created with the help of ChatGPT (GPT-5 Thinking) on 2026-02-08 (UTC)
"""

from __future__ import annotations

import glob
import inspect
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import aniso

try:
    import data_proc  # type: ignore
except Exception:  # pragma: no cover
    data_proc = None


# -----------------------------------------------------------------------------
# Small infrastructure
# -----------------------------------------------------------------------------


def ensure_dir(path: str | os.PathLike) -> str:
    """Create *path* if it does not exist and return it as a string."""
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p.as_posix()


def glob_inputs(pattern: str) -> List[str]:
    """Return sorted file list matching a glob *pattern*."""
    return sorted(glob.glob(pattern))


def _as_1d(a: np.ndarray, *, name: str, nl: int) -> np.ndarray:
    """Convert *a* to shape (nl,) float array and validate length."""
    a = np.asarray(a)
    if a.ndim != 1 or a.size != nl:
        raise ValueError(f"{name} must have shape (nl,) with nl={nl}. Got {a.shape}.")
    return a.astype(float, copy=False)


def model_from_direct(model_direct: Dict) -> Dict[str, np.ndarray]:
    """Validate and copy a model dict given inline in a script."""
    if "h_m" not in model_direct:
        raise KeyError("model_direct must contain 'h_m'.")
    model = {k: np.asarray(v) for k, v in model_direct.items()}
    return normalize_model(model)


def normalize_model(model: Dict) -> Dict[str, np.ndarray]:
    """Normalize a model dict to the simplified parameterization.

    Accepts either:

    - native simplified keys (rho_min_ohmm, rho_max_ohmm, strike_deg)
    - legacy keys (rop + ustr_deg), which are converted via:
      rho_min = min(rop[:,0], rop[:,1]), rho_max = max(...), strike = ustr_deg

    Returns
    -------
    dict
        A *new* dict with consistent numpy arrays.
    """
    if "h_m" not in model:
        raise KeyError("Model must contain 'h_m'.")

    h_m = np.asarray(model["h_m"], dtype=float).ravel()
    nl = h_m.size

    out: Dict[str, np.ndarray] = {"h_m": h_m}

    # masks
    is_iso = np.asarray(model.get("is_iso", np.zeros(nl, dtype=bool)), dtype=bool).ravel()
    is_fix = np.asarray(model.get("is_fix", np.zeros(nl, dtype=bool)), dtype=bool).ravel()
    if is_iso.size != nl or is_fix.size != nl:
        raise ValueError("is_iso and is_fix must have the same length as h_m.")
    out["is_iso"] = is_iso
    out["is_fix"] = is_fix

    if "rho_min_ohmm" in model and "rho_max_ohmm" in model:
        rho_min = _as_1d(model["rho_min_ohmm"], name="rho_min_ohmm", nl=nl)
        rho_max = _as_1d(model["rho_max_ohmm"], name="rho_max_ohmm", nl=nl)
        strike = _as_1d(model.get("strike_deg", np.zeros(nl)), name="strike_deg", nl=nl)
    elif "rop" in model:
        rop = np.asarray(model["rop"], dtype=float)
        if rop.shape != (nl, 3):
            raise ValueError("Legacy 'rop' must have shape (nl,3).")
        rho_min = np.minimum(rop[:, 0], rop[:, 1])
        rho_max = np.maximum(rop[:, 0], rop[:, 1])
        if "ustr_deg" in model:
            strike = _as_1d(model["ustr_deg"], name="ustr_deg", nl=nl)
        else:
            strike = np.zeros(nl, dtype=float)
    else:
        raise KeyError("Model must have either (rho_min_ohmm,rho_max_ohmm) or legacy 'rop'.")

    # enforce isotropy and positivity
    rho_min = np.maximum(rho_min, np.finfo(float).tiny)
    rho_max = np.maximum(rho_max, np.finfo(float).tiny)
    rho_max = np.where(is_iso, rho_min, rho_max)

    out["rho_min_ohmm"] = rho_min
    out["rho_max_ohmm"] = rho_max
    out["strike_deg"] = strike

    # pass through optional label
    if "prior_name" in model:
        out["prior_name"] = np.asarray(model["prior_name"])  # type: ignore

    return out


def save_model_npz(model: Dict, path: str) -> None:
    """Save a model dict to NPZ."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        p.as_posix(),
        **{k: np.asarray(v) for k, v in normalize_model(model).items() if k != "prior_name"},
        prior_name=str(model.get("prior_name", "")),
    )


def load_model_npz(path: str) -> Dict[str, np.ndarray]:
    """Load a model dict from NPZ."""
    with np.load(Path(path).expanduser(), allow_pickle=True) as npz:
        model = {k: npz[k] for k in npz.files}
    return normalize_model(model)


def load_site(path: str) -> Dict:
    """Load site data from .npz or .edi.

    Notes
    -----
    This function delegates to ``data_proc.load_edi`` when available.
    """
    p = Path(path).expanduser()
    if p.suffix.lower() == ".npz":
        with np.load(p, allow_pickle=True) as npz:
            d = {k: npz[k] for k in npz.files}
        if "station" not in d:
            d["station"] = p.stem
        return d

    if p.suffix.lower() == ".edi":
        if data_proc is None or not hasattr(data_proc, "load_edi"):
            raise ImportError("data_proc.load_edi not available to read .edi files.")
        d = data_proc.load_edi(p.as_posix())
        if isinstance(d, dict) and "station" not in d:
            d["station"] = p.stem
        return d

    raise ValueError(f"Unsupported file type: {p.suffix}")


def ensure_phase_tensor(site: Dict, *, nsim: int = 200) -> Dict:
    """Ensure that *site* contains phase tensor ``P`` (and optionally ``P_err``).

    This function **always recomputes** ``P`` from the observed impedance using

        P = inv(Re(Z)) @ Im(Z)

    If ``P_err`` is missing and ``Z_err`` is present, an approximate ``P_err``
    is computed by Monte-Carlo propagation.
    """
    if "Z" not in site:
        raise KeyError("site must contain 'Z' to compute 'P'.")

    Z = np.asarray(site["Z"], dtype=np.complex128)
    site = dict(site)
    site["P"] = _phase_tensor_from_Z(Z, reg=0.0)

    if ("P_err" not in site) and ("Z_err" in site):
        try:
            site["P_err"] = _bootstrap_P_err_from_Z_err(Z, np.asarray(site["Z_err"]), nsim=int(nsim))
        except Exception:
            # best-effort; keep going without P_err
            pass

    return site


# -----------------------------------------------------------------------------
# Parameter spec
# -----------------------------------------------------------------------------


class ParamSpec:
    """Configuration container for inversion parameter bounds and masks."""

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
        self.log10_h_bounds = tuple(map(float, log10_h_bounds))
        self.log10_rho_bounds = tuple(map(float, log10_rho_bounds))
        self.strike_bounds_deg = tuple(map(float, strike_bounds_deg))


# -----------------------------------------------------------------------------
# Deterministic inversion
# -----------------------------------------------------------------------------


def _periods_from_site(site: Dict) -> np.ndarray:
    """Return periods (s) from a site dict that contains ``freq``."""
    if "freq" not in site:
        raise KeyError("site must contain 'freq'.")
    freq = np.asarray(site["freq"], dtype=float).ravel()
    if np.any(freq <= 0.0):
        raise ValueError("All frequencies must be positive.")
    return 1.0 / freq


def _pack_Z_obs(
    Z: np.ndarray,
    Z_err: Optional[np.ndarray],
    *,
    comps: Sequence[str] = ("xx", "xy", "yx", "yy"),
    sigma_floor: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pack complex impedance into a real observation vector and sigma."""
    Z = np.asarray(Z, dtype=np.complex128)
    if Z.shape[-2:] != (2, 2):
        raise ValueError("Z must have shape (nper,2,2).")

    comp_map = {"xx": (0, 0), "xy": (0, 1), "yx": (1, 0), "yy": (1, 1)}
    idx = [comp_map[c] for c in comps]

    zr = np.stack([Z[:, i, j].real for i, j in idx], axis=1)
    zi = np.stack([Z[:, i, j].imag for i, j in idx], axis=1)
    y = np.concatenate([zr.reshape(-1), zi.reshape(-1)]).astype(float)

    if Z_err is None:
        sigma = np.ones_like(y)
    else:
        Ze = np.asarray(Z_err)
        if Ze.shape == Z.shape:
            se = np.stack([np.abs(Ze[:, i, j]) for i, j in idx], axis=1)
        elif Ze.shape == Z.real.shape:
            se = np.stack([np.abs(Ze[:, i, j]) for i, j in idx], axis=1)
        else:
            se = np.ones_like(zr)
        sigma = np.concatenate([se.reshape(-1), se.reshape(-1)]).astype(float)

    if sigma_floor > 0:
        sigma = np.maximum(sigma, float(sigma_floor))

    return y, sigma


def _pack_Z_jac(
    dZ: np.ndarray,
    *,
    comps: Sequence[str] = ("xx", "xy", "yx", "yy"),
) -> np.ndarray:
    """Pack complex sensitivity ``dZ`` into a real Jacobian block.

    Parameters
    ----------
    dZ : ndarray
        Complex sensitivity array shape (nper, 2, 2).

    Returns
    -------
    ndarray
        Real Jacobian column vector of length nobs for the selected components.
    """
    dZ = np.asarray(dZ, dtype=np.complex128)
    comp_map = {"xx": (0, 0), "xy": (0, 1), "yx": (1, 0), "yy": (1, 1)}
    idx = [comp_map[c] for c in comps]

    dr = np.stack([dZ[:, i, j].real for i, j in idx], axis=1)
    di = np.stack([dZ[:, i, j].imag for i, j in idx], axis=1)
    return np.concatenate([dr.reshape(-1), di.reshape(-1)]).astype(float)




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
        return np.linalg.lstsq(A, B, rcond=None)[0]


def _phase_tensor_from_Z(Z: np.ndarray, *, reg: float = 0.0) -> np.ndarray:
    """Compute phase tensor **P = inv(Re(Z)) @ Im(Z)**.

    This project uses a single, fixed convention:

        X = Re(Z), Y = Im(Z), P = X^{-1} Y

    Parameters
    ----------
    Z : ndarray, shape (nper, 2, 2)
        Complex impedance tensor.
    reg : float
        Small relative diagonal regularization when solving X P = Y.

    Returns
    -------
    ndarray, shape (nper, 2, 2)
        Real phase tensor.
    """
    Z = np.asarray(Z, dtype=np.complex128)
    if Z.ndim != 3 or Z.shape[1:] != (2, 2):
        raise ValueError("Z must have shape (nper,2,2).")
    nper = Z.shape[0]
    P = np.empty((nper, 2, 2), dtype=float)
    for i in range(nper):
        X = Z[i].real
        Y = Z[i].imag
        # Solve X * P = Y
        P[i] = _pt_solve_reg(X, Y, float(reg))
    return P


def _bootstrap_P_err_from_Z_err(
    Z: np.ndarray,
    Z_err: np.ndarray,
    *,
    nsim: int = 200,
    random_state: int | None = 0,
) -> np.ndarray:
    """Approximate P_err by Monte Carlo propagation from Z_err.

    Assumes independent Gaussian errors for Re/Im parts of each Z entry.
    """
    Z = np.asarray(Z, dtype=np.complex128)
    Z_err = np.asarray(Z_err, dtype=float)
    if Z_err.shape != Z.shape:
        raise ValueError("Z_err must have the same shape as Z.")

    rng = np.random.default_rng(None if random_state is None else int(random_state))
    nsim = int(nsim)
    Ps = np.empty((nsim, *Z.shape), dtype=float)
    for k in range(nsim):
        nre = rng.standard_normal(Z.shape)
        nim = rng.standard_normal(Z.shape)
        Zs = Z + (nre + 1j * nim) * Z_err
        Ps[k] = _phase_tensor_from_Z(Zs)
    return Ps.std(axis=0, ddof=1)

def _ridge_solve(J: np.ndarray, r: np.ndarray, lam: float) -> np.ndarray:
    """Solve (J^T J + lam^2 I) dm = J^T r."""
    JTJ = J.T @ J
    n = JTJ.shape[0]
    A = JTJ + (float(lam) ** 2) * np.eye(n)
    b = J.T @ r
    return np.linalg.solve(A, b)


def invert_site(
    site: Dict,
    *,
    spec: ParamSpec,
    model0: Dict,
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
    z_comps: Sequence[str] = ("xx", "xy", "yx", "yy"),
    pt_comps: Sequence[str] = ("xx", "xy", "yx", "yy"),
    sigma_floor_Z: float = 0.0,
    sigma_floor_P: float = 0.0,
) -> Dict:
    """Invert one site using a Gauss–Newton loop.

    If ``use_pt=True`` the inversion includes phase tensor components, the inversion
    augments the data vector by selected phase tensor components. Their
    sensitivities are approximated efficiently using a directional derivative
    through the Z sensitivities (no extra forward runs per parameter).

    Returns
    -------
    dict
        Inversion result dictionary suitable for saving with
        :func:`save_inversion_npz`.
    """
    model = normalize_model(model0)
    nl = int(model["h_m"].size)
    if nl != spec.nl:
        raise ValueError("spec.nl must match model0 layer count.")
    # data
    if use_pt:
        site = ensure_phase_tensor(site, nsim=200)

    periods_s = _periods_from_site(site)
    Zobs = np.asarray(site["Z"], dtype=np.complex128)
    Zerr = site.get("Z_err", None)

    y_obs_Z, sigma_Z = _pack_Z_obs(Zobs, Zerr, comps=z_comps, sigma_floor=sigma_floor_Z)

    # optional PT data (phase tensor)
    y_obs_P = None
    sigma_P = None
    if use_pt and ("P" in site):
        Pobs = np.asarray(site["P"], dtype=float)
        Perr = site.get("P_err", None)
        comp_map = {"xx": (0, 0), "xy": (0, 1), "yx": (1, 0), "yy": (1, 1)}
        idx = [comp_map[c] for c in pt_comps]
        y_obs_P = np.stack([Pobs[:, i, j] for i, j in idx], axis=1).reshape(-1)
        if Perr is None:
            sigma_P = np.ones_like(y_obs_P)
        else:
            Perr = np.asarray(Perr)
            if Perr.shape == Pobs.shape:
                sigma_P = np.stack([np.abs(Perr[:, i, j]) for i, j in idx], axis=1).reshape(-1)
            else:
                sigma_P = np.ones_like(y_obs_P)
        if sigma_floor_P > 0:
            sigma_P = np.maximum(sigma_P, float(sigma_floor_P))

    # working variables (log10 parameters are stable)
    log10_rho_min = np.log10(np.maximum(model["rho_min_ohmm"], np.finfo(float).tiny))
    log10_rho_max = np.log10(np.maximum(model["rho_max_ohmm"], np.finfo(float).tiny))
    strike_deg = model["strike_deg"].astype(float, copy=True)

    # masks
    is_fix = np.asarray(model.get("is_fix", np.zeros(nl, dtype=bool)), dtype=bool)
    is_iso = np.asarray(model.get("is_iso", np.zeros(nl, dtype=bool)), dtype=bool)

    # do not invert last thickness unless requested
    h_m = model["h_m"].astype(float, copy=True)

    history: List[Dict[str, float]] = []

    for it in range(int(max_iter)):
        rho_min = 10 ** log10_rho_min
        rho_max = 10 ** log10_rho_max
        rho_max = np.where(is_iso, rho_min, rho_max)

        fwd = aniso.aniso1d_impedance_sens_simple(
            periods_s=periods_s,
            h_m=h_m,
            rho_max_ohmm=rho_max,
            rho_min_ohmm=rho_min,
            strike_deg=strike_deg,
            compute_sens=True,
        )
        Zcal = fwd["Z"]

        y_cal_Z, _ = _pack_Z_obs(Zcal, None, comps=z_comps, sigma_floor=0.0)

        # Optional PT prediction
        y_cal_P = None
        if use_pt and (y_obs_P is not None):
            Pcal = _phase_tensor_from_Z(Zcal)
            comp_map = {"xx": (0, 0), "xy": (0, 1), "yx": (1, 0), "yy": (1, 1)}
            idx = [comp_map[c] for c in pt_comps]
            y_cal_P = np.stack([Pcal[:, i, j] for i, j in idx], axis=1).reshape(-1)

        # residuals (whitened)
        rZ = (y_obs_Z - y_cal_Z) / sigma_Z
        if use_pt and (y_obs_P is not None) and (y_cal_P is not None) and (sigma_P is not None):
            rP = (y_obs_P - y_cal_P) / sigma_P
            r = np.concatenate([rZ, rP])
        else:
            r = rZ

        # Precompute phase tensor at current prediction once (for directional derivatives)
        P0 = None
        if use_pt and (y_obs_P is not None) and (sigma_P is not None):
            P0 = _phase_tensor_from_Z(Zcal)

        # Build Jacobian for Z (+ optional PT), with parameters per layer:
        # [log10_rho_min(k), log10_rho_max(k), strike_deg(k)] for free layers.

        # [log10_rho_min(k), log10_rho_max(k), strike_deg(k)] for free layers.
        cols: List[np.ndarray] = []
        names: List[Tuple[str, int]] = []

        for k in range(nl):
            if bool(is_fix[k]):
                continue

            # dZ/dlog10(rho) = dZ/drho * drho/dlog10(rho) = dZ/drho * rho * ln(10)
            ln10 = np.log(10.0)

            dZ_drmin = fwd["dZ_drho_min_ohmm"][:, k]
            dZ_drmax = fwd["dZ_drho_max_ohmm"][:, k]
            dZ_dstr = fwd["dZ_dstrike_deg"][:, k]

            colZ = _pack_Z_jac(dZ_drmin * (rho_min[k] * ln10), comps=z_comps) / sigma_Z
            if use_pt and (y_obs_P is not None) and (sigma_P is not None):
                eps = 1e-6
                P1 = _phase_tensor_from_Z(Zcal + eps * dZ_drmin * (rho_min[k] * ln10))
                dP = (P1 - P0) / eps
                comp_map = {"xx": (0, 0), "xy": (0, 1), "yx": (1, 0), "yy": (1, 1)}
                idx = [comp_map[c] for c in pt_comps]
                colP = np.stack([dP[:, i, j] for i, j in idx], axis=1).reshape(-1) / sigma_P
                col = np.concatenate([colZ, colP])
            else:
                col = colZ
            cols.append(col)
            names.append(("log10_rho_min", k))

            if not bool(is_iso[k]):
                colZ = _pack_Z_jac(dZ_drmax * (rho_max[k] * ln10), comps=z_comps) / sigma_Z
                if use_pt and (y_obs_P is not None) and (sigma_P is not None):
                    eps = 1e-6
                    P1 = _phase_tensor_from_Z(Zcal + eps * dZ_drmax * (rho_max[k] * ln10))
                    dP = (P1 - P0) / eps
                    comp_map = {"xx": (0, 0), "xy": (0, 1), "yx": (1, 0), "yy": (1, 1)}
                    idx = [comp_map[c] for c in pt_comps]
                    colP = np.stack([dP[:, i, j] for i, j in idx], axis=1).reshape(-1) / sigma_P
                    col = np.concatenate([colZ, colP])
                else:
                    col = colZ
                cols.append(col)
                names.append(("log10_rho_max", k))

                colZ = _pack_Z_jac(dZ_dstr, comps=z_comps) / sigma_Z
                if use_pt and (y_obs_P is not None) and (sigma_P is not None):
                    eps = 1e-6
                    P1 = _phase_tensor_from_Z(Zcal + eps * dZ_dstr)
                    dP = (P1 - P0) / eps
                    comp_map = {"xx": (0, 0), "xy": (0, 1), "yx": (1, 0), "yy": (1, 1)}
                    idx = [comp_map[c] for c in pt_comps]
                    colP = np.stack([dP[:, i, j] for i, j in idx], axis=1).reshape(-1) / sigma_P
                    col = np.concatenate([colZ, colP])
                else:
                    col = colZ
                cols.append(col)
                names.append(("strike_deg", k))

        if not cols:
            raise RuntimeError("No free parameters (all layers fixed).")

        J = np.stack(cols, axis=1)

        # solve for update
        method_l = str(method).lower()
        if method_l not in ("tikhonov", "tsvd"):
            raise ValueError("method must be 'tikhonov' or 'tsvd'.")

        if method_l == "tikhonov":
            dm = _ridge_solve(J, r, lam=float(lam))
        else:
            # TSVD
            U, s, Vt = np.linalg.svd(J, full_matrices=False)
            if tsvd_k is None:
                # basic rcond truncation
                rcond = float(tsvd_rcond) if tsvd_rcond is not None else 1e-3
                k = int(np.sum(s / s[0] >= rcond))
            else:
                k = int(tsvd_k)
            k = max(1, min(k, s.size))
            dm = (Vt[:k].T @ ((U[:, :k].T @ r) / s[:k]))

        dm = float(step_scale) * dm

        # apply update
        for val, (pname, k) in zip(dm, names):
            if pname == "log10_rho_min":
                log10_rho_min[k] += val
            elif pname == "log10_rho_max":
                log10_rho_max[k] += val
            elif pname == "strike_deg":
                strike_deg[k] += val

        # enforce bounds crudely
        lo, hi = spec.log10_rho_bounds
        log10_rho_min = np.clip(log10_rho_min, lo, hi)
        log10_rho_max = np.clip(log10_rho_max, lo, hi)

        slo, shi = spec.strike_bounds_deg
        strike_deg = np.clip(strike_deg, slo, shi)

        # convergence metric
        misfit = float(np.sqrt(np.mean(r ** 2)))
        history.append({"iter": float(it), "rms": misfit})

        if misfit < float(tol):
            break

    # final model
    rho_min = 10 ** log10_rho_min
    rho_max = 10 ** log10_rho_max
    rho_max = np.where(is_iso, rho_min, rho_max)

    out_model = dict(model)
    out_model["rho_min_ohmm"] = rho_min
    out_model["rho_max_ohmm"] = rho_max
    out_model["strike_deg"] = strike_deg

    out = {
        "station": str(site.get("station", "")),
        "periods_s": periods_s,
        "Z_obs": Zobs,
        "Z_err": Zerr if Zerr is not None else np.array([]),
        "Z_cal": Zcal,
        "model0": model,
        "model": out_model,
        "history": np.array([(h["iter"], h["rms"]) for h in history], dtype=float),
    }
    if use_pt and y_obs_P is not None:
        out["P_obs"] = np.asarray(site["P"], dtype=float)
        if "P_err" in site:
            out["P_err"] = np.asarray(site["P_err"], dtype=float)

    return out


def save_inversion_npz(res: Dict, path: str) -> None:
    """Save an inversion result dictionary to NPZ."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)

    # Flatten nested model dicts to keep NPZ simple.
    flat: Dict[str, np.ndarray] = {}

    def _put(prefix: str, d: Dict) -> None:
        for k, v in d.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                _put(prefix=key + "/", d=v)
            else:
                flat[key] = np.asarray(v)

    _put("", res)

    np.savez(p.as_posix(), **flat)


