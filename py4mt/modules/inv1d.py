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
  - ``rho_min`` minimum horizontal resistivity
  - ``rho_max`` maximum horizontal resistivity
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


def _coerce_object_dict(obj, *, name: str = "data_dict") -> Dict:
    """Coerce *obj* to a plain Python dict.

    When loading NPZ files with ``allow_pickle=True``, nested dictionaries are
    often stored as 0-d object arrays / scalars. This helper unwraps those.
    """
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "item"):
        try:
            v = obj.item()
            if isinstance(v, dict):
                return v
        except Exception:
            pass
    raise TypeError(f"{name} must be a dict (or a 0-d object container holding a dict). Got {type(obj)!r}.")


def _extract_site_data(site: Dict) -> Tuple[Dict, bool]:
    """Return (data, wrapped) from a site container.

    Supported site container styles
    -------------------------------
    1) **Flat**: the site dict directly contains keys like ``freq``, ``Z``, ...
    2) **Wrapped**: the site dict contains a nested dict under ``site['data_dict']``
       that holds those keys, alongside metadata keys like ``station``.

    The boolean return value indicates whether the data were wrapped.
    """
    if "data_dict" in site:
        dd = _coerce_object_dict(site["data_dict"], name="site['data_dict']")
        return dd, True
    return site, False


def _store_site_data(site: Dict, data: Dict, wrapped: bool) -> Dict:
    """Store *data* back into *site* respecting wrapper style and return a copy."""
    if wrapped:
        s = dict(site)
        s["data_dict"] = data
        return s
    return data


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

    This project supports two equivalent parameter domains:

    - Resistivity domain:
        ``rho_min``, ``rho_max`` (Ohm·m), ``strike_deg``.
    - Conductivity domain:
        ``sigma_min``, ``sigma_max`` (S/m), ``strike_deg``.

    For convenience, legacy keys are also accepted:

    - ``rop`` (nl,3) principal resistivities. Only the first two principal values
      are used to define ``rho_min`` and ``rho_max``.
    - ``ustr_deg`` (nl,) is mapped to ``strike_deg``.

    The returned dict always contains **both** rho and sigma fields so that
    downstream code (deterministic inversion, plotting) can work independent of
    the chosen parametrization.

    Returns
    -------
    dict
        A *new* dict with consistent numpy arrays.
    """
    if "h_m" not in model:
        raise KeyError("Model must contain 'h_m'.")

    h_m = np.asarray(model["h_m"], dtype=float).ravel()
    nl = h_m.size

    # Strictly reject legacy parameterizations and unit-suffixed keys.
    for k in ("rop", "ustr_deg", "udip_deg", "usla_deg"):
        if k in model:
            raise KeyError(
                f"Legacy parameterization key '{k}' is not supported. "
                "Use (rho_min,rho_max,strike_deg) or (sigma_min,sigma_max,strike_deg)."
            )
    for k in model.keys():
        kl = str(k).lower()
        if ("_ohmm" in kl) or ("_spm" in kl):
            raise KeyError(
                f"Unit-suffixed key '{k}' is not supported. "
                "Drop units from keys (e.g., use 'rho_min' not 'rho_min_ohmm')."
            )

    out: Dict[str, np.ndarray] = {"h_m": h_m}

    # masks
    is_iso = np.asarray(model.get("is_iso", np.zeros(nl, dtype=bool)), dtype=bool).ravel()
    is_fix = np.asarray(model.get("is_fix", np.zeros(nl, dtype=bool)), dtype=bool).ravel()
    if is_iso.size != nl or is_fix.size != nl:
        raise ValueError("is_iso and is_fix must have the same length as h_m.")
    out["is_iso"] = is_iso
    out["is_fix"] = is_fix

    # Helper: find first available key in a list
    def _pick(keys: Sequence[str]) -> Optional[str]:
        for k in keys:
            if k in model:
                return k
        return None

    # Preferred: explicit rho
    if ("rho_min" in model) and ("rho_max" in model):
        rho_min = _as_1d(model["rho_min"], name="rho_min", nl=nl)
        rho_max = _as_1d(model["rho_max"], name="rho_max", nl=nl)
        strike = _as_1d(model.get("strike_deg", np.zeros(nl)), name="strike_deg", nl=nl)

        # enforce positivity and ordering
        rho_min = np.maximum(rho_min, np.finfo(float).tiny)
        rho_max = np.maximum(rho_max, np.finfo(float).tiny)
        rlo = np.minimum(rho_min, rho_max)
        rhi = np.maximum(rho_min, rho_max)
        rho_min, rho_max = rlo, rhi

        # isotropic layers
        rho_max = np.where(is_iso, rho_min, rho_max)

        sigma_min = 1.0 / rho_max
        sigma_max = 1.0 / rho_min

    # Alternative: explicit sigma
    elif (_pick(("sigma_min", "sig_min")) is not None) and (
        _pick(("sigma_max", "sig_max")) is not None
    ):
        kmin = _pick(("sigma_min", "sig_min"))
        kmax = _pick(("sigma_max", "sig_max"))
        assert kmin is not None and kmax is not None

        sigma_min = _as_1d(model[kmin], name=kmin, nl=nl)
        sigma_max = _as_1d(model[kmax], name=kmax, nl=nl)
        strike = _as_1d(model.get("strike_deg", np.zeros(nl)), name="strike_deg", nl=nl)

        sigma_min = np.maximum(sigma_min, np.finfo(float).tiny)
        sigma_max = np.maximum(sigma_max, np.finfo(float).tiny)
        slo = np.minimum(sigma_min, sigma_max)
        shi = np.maximum(sigma_min, sigma_max)
        sigma_min, sigma_max = slo, shi

        sigma_min = np.where(is_iso, sigma_max, sigma_min)

        # Map to resistivities used by the forward model:
        # rho_min corresponds to max conductivity; rho_max to min conductivity.
        rho_min = 1.0 / sigma_max
        rho_max = 1.0 / sigma_min

    # Legacy: rop + ustr_deg
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

        rho_min = np.maximum(rho_min, np.finfo(float).tiny)
        rho_max = np.maximum(rho_max, np.finfo(float).tiny)
        rho_max = np.where(is_iso, rho_min, rho_max)

        sigma_min = 1.0 / rho_max
        sigma_max = 1.0 / rho_min

    else:
        raise KeyError(
            "Model must provide either (rho_min,rho_max) or "
            "(sigma_min,sigma_max) or legacy 'rop'."
        )

    out["rho_min"] = np.asarray(rho_min, dtype=float)
    out["rho_max"] = np.asarray(rho_max, dtype=float)
    out["sigma_min"] = np.asarray(sigma_min, dtype=float)
    out["sigma_max"] = np.asarray(sigma_max, dtype=float)
    out["strike_deg"] = np.asarray(strike, dtype=float)

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

    Accepts both flat site dicts and wrapped dicts with a nested ``data_dict``.

    This function **always recomputes** ``P`` from the observed impedance using

        P = inv(Re(Z)) @ Im(Z)

    If ``P_err`` is missing and ``Z_err`` is present, an approximate ``P_err``
    is computed by Monte-Carlo propagation.
    """
    data, wrapped = _extract_site_data(site)
    if "Z" not in data:
        raise KeyError("site must contain 'Z' (in the flat dict or in site['data_dict']) to compute 'P'.")

    data2 = dict(data)
    Z = np.asarray(data2["Z"], dtype=np.complex128)
    data2["P"] = _phase_tensor_from_Z(Z, reg=0.0)

    if ("P_err" not in data2) and ("Z_err" in data2):
        try:
            data2["P_err"] = _bootstrap_P_err_from_Z_err(
                Z, np.asarray(data2["Z_err"]), nsim=int(nsim)
            )
        except Exception:
            # best-effort; keep going without P_err
            pass

    return _store_site_data(site, data2, wrapped)


# -----------------------------------------------------------------------------
# Parameter spec
# -----------------------------------------------------------------------------


class ParamSpec:
    """Configuration container for inversion parameter bounds and masks.

    Notes
    -----
    To match the MCMC sampler, deterministic inversion supports both
    *domains* (rho vs sigma) and two *parameter sets*:

    - ``param_domain``:
        - ``"rho"``   : sample/invert in resistivity (Ohm·m)
        - ``"sigma"`` : sample/invert in conductivity (S/m)

    - ``param_set``:
        - ``"minmax"``     : (min, max, strike) per layer
        - ``"max_anifac"`` : (max, anifac, strike) per layer, where
                             ``anifac = sqrt(max/min) >= 1``

    Bounds are expressed in log10 of the chosen domain variable, i.e.
    ``log10_param_bounds`` applies to log10(rho) when ``param_domain="rho"``
    and to log10(sigma) when ``param_domain="sigma"``.
    """

    def __init__(
        self,
        *,
        nl: int,
        fix_h: bool = True,
        sample_last_thickness: bool = False,
        log10_h_bounds: Tuple[float, float] = (0.0, 5.0),
        log10_param_bounds: Tuple[float, float] = (0.0, 5.0),
        log10_anifac_bounds: Tuple[float, float] = (0.0, 2.0),
        strike_bounds_deg: Tuple[float, float] = (-180.0, 180.0),
        param_domain: str = "rho",
        param_set: str = "minmax",
    ) -> None:
        self.nl = int(nl)
        self.fix_h = bool(fix_h)
        self.sample_last_thickness = bool(sample_last_thickness)
        self.log10_h_bounds = tuple(map(float, log10_h_bounds))
        self.log10_param_bounds = tuple(map(float, log10_param_bounds))
        self.log10_anifac_bounds = tuple(map(float, log10_anifac_bounds))
        self.strike_bounds_deg = tuple(map(float, strike_bounds_deg))
        self.param_domain = str(param_domain).lower().strip()
        self.param_set = str(param_set).lower().strip()

    # Backwards-compatible alias used by older scripts
    @property
    def log10_rho_bounds(self) -> Tuple[float, float]:
        return self.log10_param_bounds


# -----------------------------------------------------------------------------
# Deterministic inversion
# -----------------------------------------------------------------------------


def _periods_from_site(site: Dict) -> np.ndarray:
    """Return periods (s) from a site container.

    Accepts both flat site dicts and wrapped dicts with a nested ``data_dict``.
    """
    data, _wrapped = _extract_site_data(site)
    if "freq" not in data:
        raise KeyError("site must contain 'freq' to compute periods.")
    freq = np.asarray(data["freq"], dtype=float).ravel()
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

    This deterministic solver mirrors the sampler's simplified parameter options.

    - ``spec.param_domain`` in {"rho","sigma"}
    - ``spec.param_set`` in {"minmax","max_anifac"}

    If ``use_pt=True`` the inversion includes phase tensor components. Their
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

    domain = str(getattr(spec, "param_domain", "rho")).lower().strip()
    pset = str(getattr(spec, "param_set", "minmax")).lower().strip()
    if domain not in ("rho", "sigma"):
        raise ValueError("spec.param_domain must be 'rho' or 'sigma'.")
    if pset not in ("minmax", "max_anifac"):
        raise ValueError("spec.param_set must be 'minmax' or 'max_anifac'.")

    # data
    site_full = ensure_phase_tensor(site, nsim=200) if use_pt else site
    data, _wrapped = _extract_site_data(site_full)

    periods_s = _periods_from_site(site_full)
    Zobs = np.asarray(data["Z"], dtype=np.complex128)
    Zerr = data.get("Z_err", None)

    y_obs_Z, sigma_Z = _pack_Z_obs(Zobs, Zerr, comps=z_comps, sigma_floor=sigma_floor_Z)

    # optional PT data (phase tensor)
    y_obs_P = None
    sigma_P = None
    if use_pt and ("P" in data):
        Pobs = np.asarray(data["P"], dtype=float)
        Perr = data.get("P_err", None)
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

    # masks
    is_fix = np.asarray(model.get("is_fix", np.zeros(nl, dtype=bool)), dtype=bool)
    is_iso = np.asarray(model.get("is_iso", np.zeros(nl, dtype=bool)), dtype=bool)

    # thickness is fixed in this simplified deterministic solver
    h_m = model["h_m"].astype(float, copy=True)

    # --- initialize working parameters ---
    ln10 = float(np.log(10.0))

    if domain == "rho":
        rho_min0 = np.asarray(model["rho_min"], dtype=float)
        rho_max0 = np.asarray(model["rho_max"], dtype=float)
        if pset == "minmax":
            p1 = np.log10(np.maximum(rho_min0, np.finfo(float).tiny))
            p2 = np.log10(np.maximum(rho_max0, np.finfo(float).tiny))
        else:
            # p1 = log10(rho_max), p2 = log10(anifac), anifac = sqrt(rho_max/rho_min) >= 1
            p1 = np.log10(np.maximum(rho_max0, np.finfo(float).tiny))
            anifac0 = np.sqrt(np.maximum(rho_max0 / np.maximum(rho_min0, np.finfo(float).tiny), 1.0))
            p2 = np.log10(np.maximum(anifac0, 1.0))
    else:
        sigma_min0 = np.asarray(model["sigma_min"], dtype=float)
        sigma_max0 = np.asarray(model["sigma_max"], dtype=float)
        if pset == "minmax":
            p1 = np.log10(np.maximum(sigma_min0, np.finfo(float).tiny))
            p2 = np.log10(np.maximum(sigma_max0, np.finfo(float).tiny))
        else:
            # p1 = log10(sigma_max), p2 = log10(anifac), anifac = sqrt(sigma_max/sigma_min) >= 1
            p1 = np.log10(np.maximum(sigma_max0, np.finfo(float).tiny))
            anifac0 = np.sqrt(np.maximum(sigma_max0 / np.maximum(sigma_min0, np.finfo(float).tiny), 1.0))
            p2 = np.log10(np.maximum(anifac0, 1.0))

    strike_deg = np.asarray(model["strike_deg"], dtype=float).copy()

    history: List[Dict[str, float]] = []

    # bounds
    plo, phi = spec.log10_param_bounds
    alo, ahi = spec.log10_anifac_bounds
    slo, shi = spec.strike_bounds_deg

    for it in range(int(max_iter)):
        # --- decode parameters to rho_min/rho_max used by forward ---
        if domain == "rho":
            if pset == "minmax":
                rho_min = 10 ** p1
                rho_max = 10 ** p2
                # enforce ordering and isotropy
                rlo = np.minimum(rho_min, rho_max)
                rhi = np.maximum(rho_min, rho_max)
                rho_min, rho_max = rlo, np.where(is_iso, rlo, rhi)
            else:
                rho_max = 10 ** p1
                anifac = 10 ** np.clip(p2, alo, ahi)
                rho_min = rho_max / (anifac ** 2)
                rho_max = np.where(is_iso, rho_min, rho_max)
            sigma_min = 1.0 / rho_max
            sigma_max = 1.0 / rho_min
        else:
            if pset == "minmax":
                sigma_min = 10 ** p1
                sigma_max = 10 ** p2
                slo2 = np.minimum(sigma_min, sigma_max)
                shi2 = np.maximum(sigma_min, sigma_max)
                sigma_min, sigma_max = np.where(is_iso, shi2, slo2), shi2
            else:
                sigma_max = 10 ** p1
                anifac = 10 ** np.clip(p2, alo, ahi)
                sigma_min = sigma_max / (anifac ** 2)
                sigma_min = np.where(is_iso, sigma_max, sigma_min)
            rho_min = 1.0 / sigma_max
            rho_max = 1.0 / sigma_min

        strike_deg = np.clip(strike_deg, slo, shi)

        fwd = aniso.aniso1d_impedance_sens_simple(
            periods_s,
            h_m,
            rho_max,
            rho_min,
            strike_deg,
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

        # Build Jacobian for Z (+ optional PT)
        cols: List[np.ndarray] = []
        names: List[Tuple[str, int]] = []

        dZ_drmin_all = np.asarray(fwd["dZ_drho_min"])
        dZ_drmax_all = np.asarray(fwd["dZ_drho_max"])
        dZ_dstr_all = np.asarray(fwd["dZ_dstrike_deg"])

        comp_map = {"xx": (0, 0), "xy": (0, 1), "yx": (1, 0), "yy": (1, 1)}
        idx_pt = [comp_map[c] for c in pt_comps]

        for k in range(nl):
            if bool(is_fix[k]):
                continue

            dZ_drmin = dZ_drmin_all[:, k]
            dZ_drmax = dZ_drmax_all[:, k]
            dZ_dstr = dZ_dstr_all[:, k]

            # ----- p1 column -----
            if domain == "rho":
                if pset == "minmax":
                    dZ_dp1 = dZ_drmin * (rho_min[k] * ln10)
                else:
                    # log10(rho_max): affects both rho_max and rho_min
                    dZ_dp1 = dZ_drmax * (rho_max[k] * ln10) + dZ_drmin * (rho_min[k] * ln10)
                pname1 = "log10_rho_min" if pset == "minmax" else "log10_rho_max"
            else:
                if pset == "minmax":
                    # log10(sigma_min): affects rho_max only (rho_max = 1/sigma_min)
                    dZ_dp1 = dZ_drmax * (-rho_max[k] * ln10)
                else:
                    # log10(sigma_max): affects rho_min and rho_max (both ~ 1/sigma_max)
                    dZ_dp1 = dZ_drmin * (-rho_min[k] * ln10) + dZ_drmax * (-rho_max[k] * ln10)
                pname1 = "log10_sigma_min" if pset == "minmax" else "log10_sigma_max"

            colZ = _pack_Z_jac(dZ_dp1, comps=z_comps) / sigma_Z
            if use_pt and (y_obs_P is not None) and (sigma_P is not None):
                eps = 1e-6
                P1 = _phase_tensor_from_Z(Zcal + eps * dZ_dp1)
                dP = (P1 - P0) / eps
                colP = np.stack([dP[:, i, j] for i, j in idx_pt], axis=1).reshape(-1) / sigma_P
                col = np.concatenate([colZ, colP])
            else:
                col = colZ
            cols.append(col)
            names.append((pname1, k))

            # ----- p2 column (only if anisotropic) -----
            if not bool(is_iso[k]):
                if domain == "rho":
                    if pset == "minmax":
                        dZ_dp2 = dZ_drmax * (rho_max[k] * ln10)
                        pname2 = "log10_rho_max"
                    else:
                        # log10(anifac): affects rho_min only, rho_min ~ anifac^{-2}
                        dZ_dp2 = dZ_drmin * (rho_min[k] * (-2.0 * ln10))
                        pname2 = "log10_anifac"
                else:
                    if pset == "minmax":
                        # log10(sigma_max): affects rho_min only
                        dZ_dp2 = dZ_drmin * (-rho_min[k] * ln10)
                        pname2 = "log10_sigma_max"
                    else:
                        # log10(anifac): affects rho_max only, rho_max ~ anifac^{2}
                        dZ_dp2 = dZ_drmax * (rho_max[k] * (2.0 * ln10))
                        pname2 = "log10_anifac"

                colZ = _pack_Z_jac(dZ_dp2, comps=z_comps) / sigma_Z
                if use_pt and (y_obs_P is not None) and (sigma_P is not None):
                    eps = 1e-6
                    P1 = _phase_tensor_from_Z(Zcal + eps * dZ_dp2)
                    dP = (P1 - P0) / eps
                    colP = np.stack([dP[:, i, j] for i, j in idx_pt], axis=1).reshape(-1) / sigma_P
                    col = np.concatenate([colZ, colP])
                else:
                    col = colZ
                cols.append(col)
                names.append((pname2, k))

                # ----- strike column -----
                colZ = _pack_Z_jac(dZ_dstr, comps=z_comps) / sigma_Z
                if use_pt and (y_obs_P is not None) and (sigma_P is not None):
                    eps = 1e-6
                    P1 = _phase_tensor_from_Z(Zcal + eps * dZ_dstr)
                    dP = (P1 - P0) / eps
                    colP = np.stack([dP[:, i, j] for i, j in idx_pt], axis=1).reshape(-1) / sigma_P
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
            U, s, Vt = np.linalg.svd(J, full_matrices=False)
            if tsvd_k is None:
                rcond = float(tsvd_rcond) if tsvd_rcond is not None else 1e-3
                k = int(np.sum(s / s[0] >= rcond))
            else:
                k = int(tsvd_k)
            k = max(1, min(k, s.size))
            dm = (Vt[:k].T @ ((U[:, :k].T @ r) / s[:k]))

        dm = float(step_scale) * dm

        # apply update
        for val, (pname, k) in zip(dm, names):
            if pname in ("log10_rho_min", "log10_rho_max", "log10_sigma_min", "log10_sigma_max"):
                # map to p1/p2 depending on configuration
                if domain == "rho":
                    if pset == "minmax":
                        if pname == "log10_rho_min":
                            p1[k] += val
                        elif pname == "log10_rho_max":
                            p2[k] += val
                    else:
                        if pname == "log10_rho_max":
                            p1[k] += val
                else:
                    if pset == "minmax":
                        if pname == "log10_sigma_min":
                            p1[k] += val
                        elif pname == "log10_sigma_max":
                            p2[k] += val
                    else:
                        if pname == "log10_sigma_max":
                            p1[k] += val
            elif pname == "log10_anifac":
                p2[k] += val
            elif pname == "strike_deg":
                strike_deg[k] += val

        # enforce bounds
        if pset == "minmax":
            p1 = np.clip(p1, plo, phi)
            p2 = np.clip(p2, plo, phi)
        else:
            p1 = np.clip(p1, plo, phi)
            p2 = np.clip(p2, alo, ahi)

        strike_deg = np.clip(strike_deg, slo, shi)

        # convergence metric
        misfit = float(np.sqrt(np.mean(r ** 2)))
        history.append({"iter": float(it), "rms": misfit})

        if misfit < float(tol):
            break

    # final decoded model (reuse decoding logic)
    if domain == "rho":
        if pset == "minmax":
            rho_min = 10 ** p1
            rho_max = 10 ** p2
            rlo = np.minimum(rho_min, rho_max)
            rhi = np.maximum(rho_min, rho_max)
            rho_min, rho_max = rlo, np.where(is_iso, rlo, rhi)
        else:
            rho_max = 10 ** p1
            anifac = 10 ** np.clip(p2, alo, ahi)
            rho_min = rho_max / (anifac ** 2)
            rho_max = np.where(is_iso, rho_min, rho_max)
        sigma_min = 1.0 / rho_max
        sigma_max = 1.0 / rho_min
    else:
        if pset == "minmax":
            sigma_min = 10 ** p1
            sigma_max = 10 ** p2
            slo2 = np.minimum(sigma_min, sigma_max)
            shi2 = np.maximum(sigma_min, sigma_max)
            sigma_min, sigma_max = np.where(is_iso, shi2, slo2), shi2
        else:
            sigma_max = 10 ** p1
            anifac = 10 ** np.clip(p2, alo, ahi)
            sigma_min = sigma_max / (anifac ** 2)
            sigma_min = np.where(is_iso, sigma_max, sigma_min)
        rho_min = 1.0 / sigma_max
        rho_max = 1.0 / sigma_min

    out_model = dict(model)
    out_model["rho_min"] = np.asarray(rho_min, dtype=float)
    out_model["rho_max"] = np.asarray(rho_max, dtype=float)
    out_model["sigma_min"] = np.asarray(sigma_min, dtype=float)
    out_model["sigma_max"] = np.asarray(sigma_max, dtype=float)
    out_model["strike_deg"] = np.asarray(strike_deg, dtype=float)

    out = {
        "station": str(site_full.get("station", data.get("station", ""))),
        "periods_s": periods_s,
        "Z_obs": Zobs,
        "Z_err": Zerr if Zerr is not None else np.array([]),
        "Z_cal": Zcal,
        "model0": model,
        "model": out_model,
        "history": np.array([(h["iter"], h["rms"]) for h in history], dtype=float),
        "param_domain": np.asarray(domain),
        "param_set": np.asarray(pset),
    }
    if use_pt and y_obs_P is not None:
        out["P_obs"] = np.asarray(data["P"], dtype=float)
        if "P_err" in data:
            out["P_err"] = np.asarray(data["P_err"], dtype=float)

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


