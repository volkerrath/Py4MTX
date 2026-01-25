"""
mcmc.py
=======

PyMC interface for anisotropic 1-D MT inversion (single site, impedance only)
with optional Phase Tensor likelihood.

This module provides a small, self-contained bridge between:

- the anisotropic 1-D forward model / sensitivities in ``aniso.py``:
  :func:`aniso.aniso1d_impedance_sens`
- a PyMC probabilistic model for MCMC sampling (``pymc``)

Design goals
------------
- **No emcee**: sampling is performed with PyMC.
- **Optional gradients**: if enabled, the likelihood Op also provides a gradient
  based on the analytic sensitivities returned by :func:`aniso1d_impedance_sens`.
  This allows using gradient-based samplers (e.g., NUTS) later on, while the
  default remains gradient-free samplers (e.g., DEMetropolisZ).
- **Impedance + Phase Tensor**: use complex impedance components (default ``xy,yx``)
  and optionally add Phase Tensor components (real 2x2).

Data format assumptions
-----------------------
The likelihood expects a *site dictionary* (as commonly produced by MT parsing
utilities) with at least:

- ``freq`` : ndarray (nfreq,)  [Hz]
- ``Z``    : ndarray (nfreq, 2, 2) complex128
- ``Z_err`` (optional) : ndarray same shape, either std-dev or variance
- ``err_kind`` (optional) : "std" or "var" (applies to *_err arrays)

For phase tensor (optional, when ``use_pt=True``):

- ``P`` (optional) : ndarray (nfreq, 2, 2) float64
- ``P_err`` (optional) : ndarray same shape, std-dev or variance

If ``P`` is missing and ``compute_pt_if_missing=True``, it is computed from Z:
    P = inv(Re(Z)) @ Im(Z)

If ``P_err`` is missing, a constant floor ``sigma_floor_P`` is used.

Parameterization
----------------
The forward model uses per-layer principal resistivities and Euler angles:

- ``h_m`` (nl,) thicknesses in meters
- ``rop`` (nl, 3) principal resistivities [Ohm·m]
- ``ustr_deg, udip_deg, usla_deg`` (nl,) angles in degrees
- ``is_iso`` (nl,) optional flag: if True, layer treated as isotropic
- ``is_fix`` (nl,) optional flag: if True, keep all parameters of that layer fixed in sampling

In PyMC we typically sample on transformed variables:

- thicknesses: ``log10(h_m)`` (positivity)
- resistivities: ``log10(rop)``
- angles: degrees (bounded)

The mapping is controlled by :class:`ParamSpec` and :func:`build_pymc_model`.

Author: Volker Rath (DIAS)
Created with the help of ChatGPT (GPT-5 Thinking) on 2026-01-20
"""

from __future__ import annotations
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import os
from pathlib import Path


# -------------------------------------------------------------------------
# Import guards:
# PyMC imports ArviZ, which may import numba. In some environments a mismatch
# between `numba` and `coverage` can raise errors during import (not always as
# ImportError). We patch missing typing aliases in `coverage.types` that numba's
# optional coverage integration expects, and we also remove any partially
# imported `numba`/`arviz` modules from previous attempts.
# -------------------------------------------------------------------------

def _prepare_optional_deps() -> None:
    """Prepare environment so importing PyMC does not fail due to numba/coverage issues."""
    import sys
    from typing import Any

    # Remove partially imported modules from previous failed imports
    for k in list(sys.modules.keys()):
        if k == "numba" or k.startswith("numba.") or k == "arviz" or k.startswith("arviz."):
            del sys.modules[k]

    # Patch coverage.types typing aliases expected by numba
    try:
        import coverage  # type: ignore
    except Exception:
        return
    if not hasattr(coverage, "types"):
        return
    ct = coverage.types  # type: ignore[attr-defined]
    if not hasattr(ct, "Tracer"):
        class Tracer:
            pass
        ct.Tracer = Tracer  # type: ignore[attr-defined]
    for name in ("TTraceFn", "TWarnFn", "TShouldTraceFn", "TShouldStartContextFn", "TFileDisposition"):
        if not hasattr(ct, name):
            setattr(ct, name, Any)  # type: ignore[attr-defined]


_prepare_optional_deps()


# PyMC / PyTensor
import pymc as pm
import pytensor
import pytensor.tensor as pt

# -------------------------------------------------------------------------
# PyTensor configuration:
# Some minimal/container environments lack Python development headers
# (Python.h), which breaks PyTensor's C compilation. Force pure-Python
# execution to keep PyMC usable.
# -------------------------------------------------------------------------
try:
    pytensor.config.cxx = ""  # disable C compilation
    pytensor.config.linker = "py"  # pure-Python linker
    pytensor.config.mode = "FAST_COMPILE"
except Exception:
    pass
from pytensor.graph.op import Op
from pytensor.graph.basic import Apply

# Local forward model
import aniso


ArrayLike = Union[np.ndarray, Sequence[float]]


def _as_1d_float(x: ArrayLike, name: str) -> np.ndarray:
    """
    Convert input to a 1-D float64 numpy array.

    Parameters
    ----------
    x
        Array-like input.
    name
        Variable name used in error messages.

    Returns
    -------
    ndarray
        1-D float64 array.

    Raises
    ------
    ValueError
        If the array cannot be converted to 1-D.
    """
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
    return a


def _parse_err_kind(site: Mapping) -> str:
    """
    Parse the error kind for uncertainty arrays.

    Parameters
    ----------
    site
        Site dictionary possibly containing ``err_kind``.

    Returns
    -------
    str
        Either "std" (standard deviation) or "var" (variance).
    """
    kind = str(site.get("err_kind", "std")).strip().lower()
    if kind not in ("std", "var"):
        kind = "std"
    return kind


def _err_to_std(err: np.ndarray, err_kind: str) -> np.ndarray:
    """
    Convert error array to standard deviation.

    Parameters
    ----------
    err
        Error array (std or var depending on ``err_kind``).
    err_kind
        "std" or "var".

    Returns
    -------
    ndarray
        Standard deviation array (float64, non-negative).
    """
    e = np.asarray(err, dtype=np.float64)
    if err_kind == "var":
        e = np.sqrt(np.maximum(e, 0.0))
    return np.maximum(e, 0.0)


def _comp_indices(comps: Sequence[str]) -> List[Tuple[int, int]]:
    """
    Map component strings to tensor indices.

    Parameters
    ----------
    comps
        Sequence like ("xx","xy","yx","yy").

    Returns
    -------
    list
        List of (i, j) indices for each component.
    """
    m = {"xx": (0, 0), "xy": (0, 1), "yx": (1, 0), "yy": (1, 1)}
    idx = []
    for c in comps:
        cc = c.strip().lower()
        if cc not in m:
            raise ValueError(f"Unknown component '{c}'. Use xx, xy, yx, yy.")
        idx.append(m[cc])
    return idx


def phase_tensor_from_Z(Z: np.ndarray) -> np.ndarray:
    """
    Compute Phase Tensor P from complex impedance Z.

    P is defined as:
        P = inv(Re(Z)) @ Im(Z)

    Parameters
    ----------
    Z
        Complex impedance array of shape (nper, 2, 2).

    Returns
    -------
    ndarray
        Phase tensor array of shape (nper, 2, 2), float64.

    Notes
    -----
    This function does **not** compute uncertainties for P.
    """
    Z = np.asarray(Z)
    X = np.real(Z)
    Y = np.imag(Z)
    nper = Z.shape[0]
    P = np.empty((nper, 2, 2), dtype=np.float64)
    for k in range(nper):
        P[k] = np.linalg.solve(X[k], Y[k])
    return P


def _pack_Z(
    Z: np.ndarray,
    comps: Sequence[str],
) -> np.ndarray:
    """
    Pack selected impedance components into a 1-D vector (real+imag).

    Parameters
    ----------
    Z
        Complex impedance (nper, 2, 2).
    comps
        Components to include, e.g. ("xy","yx").

    Returns
    -------
    ndarray
        Packed vector of length nper * len(comps) * 2:
        [Re(Zc1), Im(Zc1), Re(Zc2), Im(Zc2), ...] per period.
    """
    Z = np.asarray(Z)
    idx = _comp_indices(comps)
    nper = Z.shape[0]
    out = np.empty(nper * len(idx) * 2, dtype=np.float64)
    p = 0
    for k in range(nper):
        for (i, j) in idx:
            z = Z[k, i, j]
            out[p] = np.real(z)
            out[p + 1] = np.imag(z)
            p += 2
    return out


def _pack_P(
    P: np.ndarray,
    comps: Sequence[str],
) -> np.ndarray:
    """
    Pack selected phase tensor components into a 1-D vector.

    Parameters
    ----------
    P
        Phase tensor (nper, 2, 2), real.
    comps
        Components to include, e.g. ("xx","xy","yx","yy").

    Returns
    -------
    ndarray
        Packed vector of length nper * len(comps).
    """
    P = np.asarray(P, dtype=np.float64)
    idx = _comp_indices(comps)
    nper = P.shape[0]
    out = np.empty(nper * len(idx), dtype=np.float64)
    p = 0
    for k in range(nper):
        for (i, j) in idx:
            out[p] = P[k, i, j]
            p += 1
    return out
class ParamSpec:
    """
    Parameter specification for the PyMC inversion.

    Attributes
    ----------
    nl
        Number of layers (including basement layer).
    fix_h
        If True, thicknesses are fixed and not sampled.
    sample_last_thickness
        If True, include the last thickness entry in sampling. In many MT
        parameterizations the basement thickness is unused (often set to 0);
        in that case set this to False.
    log10_h_bounds
        (low, high) bounds for log10 thickness [m] when sampled.
    log10_rho_bounds
        (low, high) bounds for log10 resistivity [Ohm·m].
    ustr_bounds_deg
        Strike bounds in degrees.
    udip_bounds_deg
        Dip bounds in degrees.
    usla_bounds_deg
        Slant bounds in degrees.
    """

    def __init__(
        self,
        nl: int,
        fix_h: bool = False,
        sample_last_thickness: bool = False,
        log10_h_bounds: Tuple[float, float] = (0.0, 5.0),          # 1 m .. 100 km
        log10_rho_bounds: Tuple[float, float] = (-1.0, 6.0),       # 0.1 .. 1e6 Ohm m
        ustr_bounds_deg: Tuple[float, float] = (-180.0, 180.0),
        udip_bounds_deg: Tuple[float, float] = (0.0, 90.0),
        usla_bounds_deg: Tuple[float, float] = (-180.0, 180.0),
    ) -> None:
        """Create a parameter specification (no dataclass).

        Parameters
        ----------
        nl : int
            Number of layers (including basement).
        fix_h : bool, optional
            If True, thicknesses are fixed and not sampled.
        sample_last_thickness : bool, optional
            If True, include the last thickness entry in sampling.
        log10_h_bounds : tuple of float, optional
            Bounds for log10 thickness [m] when sampled.
        log10_rho_bounds : tuple of float, optional
            Bounds for log10 resistivity [Ohm·m].
        ustr_bounds_deg, udip_bounds_deg, usla_bounds_deg : tuple of float, optional
            Bounds for strike/dip/slant angles in degrees.
        """
        self.nl = int(nl)
        self.fix_h = bool(fix_h)
        self.sample_last_thickness = bool(sample_last_thickness)
        self.log10_h_bounds = (float(log10_h_bounds[0]), float(log10_h_bounds[1]))
        self.log10_rho_bounds = (float(log10_rho_bounds[0]), float(log10_rho_bounds[1]))
        self.ustr_bounds_deg = (float(ustr_bounds_deg[0]), float(ustr_bounds_deg[1]))
        self.udip_bounds_deg = (float(udip_bounds_deg[0]), float(udip_bounds_deg[1]))
        self.usla_bounds_deg = (float(usla_bounds_deg[0]), float(usla_bounds_deg[1]))


    def nh(self) -> int:
        """
        Number of thickness parameters that are sampled.

        Returns
        -------
        int
            Number of sampled thickness entries.
        """
        if self.fix_h:
            return 0
        return self.nl if self.sample_last_thickness else max(self.nl - 1, 0)

    def nrho(self) -> int:
        """
        Number of resistivity parameters (flattened).

        Returns
        -------
        int
            nl * 3.
        """
        return self.nl * 3

    def nang(self) -> int:
        """
        Number of angle parameters (flattened).

        Returns
        -------
        int
            nl * 3.
        """
        return self.nl * 3

    def ndim(self) -> int:
        """
        Total parameter dimension.

        Returns
        -------
        int
            Total number of scalar parameters in theta.
        """
        return self.nh() + self.nrho() + self.nang()

def normalize_is_fix(is_fix: Optional[np.ndarray], *, nl: int) -> np.ndarray:
    """Normalize a per-layer fixed flag array.

    Parameters
    ----------
    is_fix : ndarray or None
        If provided, must be array-like with shape (nl,) and boolean-convertible.
        True means the entire layer is fixed (thickness, resistivities, angles).
    nl : int
        Number of layers.

    Returns
    -------
    ndarray
        Boolean array of shape (nl,). If ``is_fix`` is None, returns all-False.
    """
    if is_fix is None:
        return np.zeros(int(nl), dtype=bool)
    arr = np.asarray(is_fix, dtype=bool).ravel()
    if arr.shape != (int(nl),):
        raise ValueError(f"is_fix must have shape ({int(nl)},), got {arr.shape}")
    return arr


def normalize_model(model: Mapping[str, object]) -> Dict[str, np.ndarray]:
    """Normalize a model dictionary for the anisotropic 1-D MT forward model.

    The inversion stack expects a consistent public parameterization per layer:

    - ``h_m``       : (nl,) float, thicknesses [m]
    - ``rop``       : (nl, 3) float, principal resistivities [Ohm·m]
    - ``ustr_deg``  : (nl,) float, strike angles [deg]
    - ``udip_deg``  : (nl,) float, dip angles [deg]
    - ``usla_deg``  : (nl,) float, slant angles [deg]
    - ``is_iso``    : (nl,) bool, isotropic-layer flags
    - ``is_fix``    : (nl,) bool, fixed-layer flags (freeze that layer in sampling)

    The key requirement (important for downstream code) is:

    ``len(is_iso) == len(is_fix) == len(h_m) == nl``.

    This helper also tolerates a common shape mistake for ``rop``:

    - If ``rop`` is given as (3, nl), it is transposed to (nl, 3).

    Parameters
    ----------
    model
        Mapping containing at least ``rop`` and (ideally) ``h_m``.

    Returns
    -------
    dict
        Normalized model dictionary (numpy arrays). Any extra keys from the
        input mapping are preserved.

    Raises
    ------
    ValueError
        If the model arrays cannot be made consistent.
    """
    md: Dict[str, object] = dict(model)

    # --- thickness -------------------------------------------------------
    h_key = "h_m" if "h_m" in md else ("h" if "h" in md else None)
    if h_key is not None and md.get(h_key, None) is not None:
        h_m = np.asarray(md[h_key], dtype=float).ravel()
    else:
        h_m = None

    # --- resistivities (must exist) -------------------------------------
    if "rop" not in md or md.get("rop", None) is None:
        raise ValueError("Model must contain 'rop' (principal resistivities).")
    rop = np.asarray(md["rop"], dtype=float)
    if rop.ndim != 2:
        raise ValueError(f"rop must be 2-D, got shape {rop.shape}")

    # Infer nl if thickness missing
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

    # Normalize rop to (nl, 3)
    if rop.shape == (nl, 3):
        pass
    elif rop.shape == (3, nl):
        rop = rop.T
    elif rop.shape[1] == 3 and rop.shape[0] != nl:
        raise ValueError(f"rop has {rop.shape[0]} layers but h_m has {nl}.")
    elif rop.shape[0] == 3 and rop.shape[1] != nl:
        raise ValueError(f"rop has {rop.shape[1]} layers but h_m has {nl}.")
    else:
        raise ValueError(f"rop must have shape ({nl},3) (or (3,{nl}) to be transposed), got {rop.shape}.")

    # --- angles ----------------------------------------------------------
    def _norm_ang(key: str) -> np.ndarray:
        val = md.get(key, 0.0)
        a = np.asarray(val, dtype=float)
        if a.ndim == 0:
            return np.full(nl, float(a), dtype=float)
        a = a.ravel()
        if a.shape != (nl,):
            raise ValueError(f"{key} must have shape ({nl},), got {a.shape}")
        return a

    ustr_deg = _norm_ang("ustr_deg")
    udip_deg = _norm_ang("udip_deg")
    usla_deg = _norm_ang("usla_deg")

    # --- flags (must match h_m length) ----------------------------------
    def _norm_flag(key: str) -> np.ndarray:
        val = md.get(key, None)
        if val is None:
            return np.zeros(nl, dtype=bool)
        a = np.asarray(val, dtype=bool)
        if a.ndim == 0:
            return np.full(nl, bool(a), dtype=bool)
        a = a.ravel()
        if a.shape != (nl,):
            raise ValueError(f"{key} must have shape ({nl},), got {a.shape}")
        return a

    is_iso = _norm_flag("is_iso")
    is_fix = _norm_flag("is_fix")

    # Update mapping (preserve extras)
    md["h_m"] = np.asarray(h_m, dtype=float)
    md["rop"] = np.asarray(rop, dtype=float)
    md["ustr_deg"] = np.asarray(ustr_deg, dtype=float)
    md["udip_deg"] = np.asarray(udip_deg, dtype=float)
    md["usla_deg"] = np.asarray(usla_deg, dtype=float)
    md["is_iso"] = np.asarray(is_iso, dtype=bool)
    md["is_fix"] = np.asarray(is_fix, dtype=bool)

    return md  # type: ignore[return-value]



def theta_to_model(
    theta: np.ndarray,
    spec: ParamSpec,
    h_m_fixed: Optional[np.ndarray],
    rop_fixed: Optional[np.ndarray],
    ustr_fixed: Optional[np.ndarray],
    udip_fixed: Optional[np.ndarray],
    usla_fixed: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert theta vector into physical model arrays for the forward model.

    Parameters
    ----------
    theta
        1-D parameter vector (float64).
    spec
        Parameter specification (controls sizes and transforms).
    h_m_fixed, rop_fixed, ustr_fixed, udip_fixed, usla_fixed
        Fixed reference arrays used for entries not sampled (or as base).

    Returns
    -------
    h_m, rop, ustr_deg, udip_deg, usla_deg
        Physical arrays with shapes (nl,), (nl,3), (nl,), (nl,), (nl,).

    Notes
    -----
    Thicknesses and resistivities are sampled in log10 space and converted
    back to linear space here.
    """
    theta = _as_1d_float(theta, "theta")
    if theta.size != spec.ndim():
        raise ValueError(f"theta has size {theta.size}, expected {spec.ndim()}")

    nl = spec.nl
    # Initialize from fixed arrays
    if h_m_fixed is None:
        h_m = np.ones(nl, dtype=np.float64)
        h_m[-1] = 0.0
    else:
        h_m = np.asarray(h_m_fixed, dtype=np.float64).copy()
        if h_m.shape != (nl,):
            raise ValueError(f"h_m_fixed must have shape ({nl},), got {h_m.shape}")

    if rop_fixed is None:
        rop = np.ones((nl, 3), dtype=np.float64) * 100.0
    else:
        rop = np.asarray(rop_fixed, dtype=np.float64).copy()
        if rop.shape != (nl, 3):
            raise ValueError(f"rop_fixed must have shape ({nl},3), got {rop.shape}")

    def _ang_init(a_fixed: Optional[np.ndarray], name: str) -> np.ndarray:
        if a_fixed is None:
            return np.zeros(nl, dtype=np.float64)
        a = np.asarray(a_fixed, dtype=np.float64).copy()
        if a.shape != (nl,):
            raise ValueError(f"{name} must have shape ({nl},), got {a.shape}")
        return a

    ustr_deg = _ang_init(ustr_fixed, "ustr_fixed")
    udip_deg = _ang_init(udip_fixed, "udip_fixed")
    usla_deg = _ang_init(usla_fixed, "usla_fixed")

    p = 0

    # Thicknesses (log10)
    if not spec.fix_h:
        nh = spec.nh()
        log10_h = theta[p : p + nh]
        p += nh
        h_m[:nh] = 10.0 ** log10_h
        if not spec.sample_last_thickness and nl > nh:
            # Keep last thickness as provided (often 0 basement)
            pass

    # Resistivities (log10), always sampled unless user passes fixed by changing spec externally
    log10_rop = theta[p : p + spec.nrho()]
    p += spec.nrho()
    rop[:] = (10.0 ** log10_rop).reshape((nl, 3))

    # Angles (degrees)
    ang = theta[p : p + spec.nang()]
    p += spec.nang()
    ang = ang.reshape((nl, 3))
    ustr_deg[:] = ang[:, 0]
    udip_deg[:] = ang[:, 1]
    usla_deg[:] = ang[:, 2]

    return h_m, rop, ustr_deg, udip_deg, usla_deg


class _AnisoMTContext:
    """
    Internal container for data packing and forward/gradient evaluation.

    Parameters
    ----------
    site
        Site dictionary with Z and optional P.
    periods_s
        Period array in seconds.
    spec
        Parameter specification.
    use_pt
        Whether to include Phase Tensor in the likelihood.
    z_comps
        Impedance components to use.
    pt_comps
        Phase tensor components to use.
    compute_pt_if_missing
        Compute P from Z if missing.
    sigma_floor_Z
        Floor added to Z sigmas (real and imag) to avoid zero variance.
    sigma_floor_P
        Floor added to P sigmas to avoid zero variance.
    is_iso
        Optional per-layer isotropic flags passed to the forward model.

    Notes
    -----
    This object precomputes the observed data vector and uncertainty vector.
    """
    def __init__(
        self,
        site: Mapping,
        periods_s: np.ndarray,
        spec: ParamSpec,
        *,
        use_pt: bool,
        z_comps: Sequence[str],
        pt_comps: Sequence[str],
        compute_pt_if_missing: bool,
        sigma_floor_Z: float,
        sigma_floor_P: float,
        is_iso: Optional[np.ndarray],
    ) -> None:
        self.spec = spec
        self.use_pt = bool(use_pt)
        self.z_comps = tuple(z_comps)
        self.pt_comps = tuple(pt_comps)
        self.compute_pt_if_missing = bool(compute_pt_if_missing)
        self.sigma_floor_Z = float(sigma_floor_Z)
        self.sigma_floor_P = float(sigma_floor_P)

        self.periods_s = _as_1d_float(periods_s, "periods_s")
        self.nper = self.periods_s.size

        # Observations
        if "Z" not in site:
            raise KeyError("site must contain key 'Z' with shape (nper,2,2).")
        self.Z_obs = np.asarray(site["Z"])
        if self.Z_obs.shape != (self.nper, 2, 2):
            raise ValueError(f"Z must have shape ({self.nper},2,2), got {self.Z_obs.shape}")

        err_kind = _parse_err_kind(site)

        Z_err = site.get("Z_err", None)
        if Z_err is None:
            Z_std = np.ones_like(self.Z_obs, dtype=np.float64) * self.sigma_floor_Z
        else:
            Z_std = _err_to_std(np.asarray(Z_err), err_kind)
            if Z_std.shape != self.Z_obs.shape:
                raise ValueError("Z_err must have same shape as Z.")
            Z_std = np.maximum(Z_std, self.sigma_floor_Z)

        self.yZ = _pack_Z(self.Z_obs, self.z_comps)
        self.sZ = _pack_Z(Z_std.astype(np.complex128), self.z_comps)  # pack uses real/imag
        # _pack_Z expects complex; for std we pack real/imag the same by using complex std in real part
        # but we constructed complex with std in real part only; fix:
        # Better: rebuild sZ explicitly:
        self.sZ = self._pack_Z_std(Z_std, self.z_comps, self.sigma_floor_Z)

        # Optional phase tensor
        self.P_obs = None
        self.yP = None
        self.sP = None
        if self.use_pt:
            if "P" in site:
                P = np.asarray(site["P"], dtype=np.float64)
                if P.shape != (self.nper, 2, 2):
                    raise ValueError("P must have shape (nper,2,2).")
            else:
                if not self.compute_pt_if_missing:
                    raise KeyError("use_pt=True but site has no 'P' and compute_pt_if_missing=False.")
                P = phase_tensor_from_Z(self.Z_obs)

            P_err = site.get("P_err", None)
            if P_err is None:
                P_std = np.ones_like(P, dtype=np.float64) * self.sigma_floor_P
            else:
                P_std = _err_to_std(np.asarray(P_err), err_kind)
                if P_std.shape != P.shape:
                    raise ValueError("P_err must have same shape as P.")
                P_std = np.maximum(P_std, self.sigma_floor_P)

            self.P_obs = P
            self.yP = _pack_P(P, self.pt_comps)
            self.sP = self._pack_P_std(P_std, self.pt_comps, self.sigma_floor_P)

        # concatenate
        if self.use_pt:
            self.y = np.concatenate([self.yZ, self.yP], axis=0)
            self.s = np.concatenate([self.sZ, self.sP], axis=0)
        else:
            self.y = self.yZ
            self.s = self.sZ

        # Forward-model flags
        if is_iso is None:
            self.is_iso = None
        else:
            is_iso = np.asarray(is_iso).astype(bool)
            if is_iso.shape != (self.spec.nl,):
                raise ValueError(f"is_iso must have shape ({self.spec.nl},), got {is_iso.shape}")
            self.is_iso = is_iso

        # Constants for loglike
        self._log2pi = np.log(2.0 * np.pi)

    @staticmethod
    def _pack_Z_std(Z_std: np.ndarray, comps: Sequence[str], floor: float) -> np.ndarray:
        """
        Pack standard deviations for Z into the same layout as :func:`_pack_Z`.

        Parameters
        ----------
        Z_std
            Standard deviation array for Z (nper,2,2), real.
        comps
            Components list.
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

    @staticmethod
    def _pack_P_std(P_std: np.ndarray, comps: Sequence[str], floor: float) -> np.ndarray:
        """
        Pack standard deviations for P into the same layout as :func:`_pack_P`.

        Parameters
        ----------
        P_std
            Standard deviation array for P (nper,2,2), real.
        comps
            Components list.
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

    def forward(
        self,
        h_m: np.ndarray,
        rop: np.ndarray,
        ustr_deg: np.ndarray,
        udip_deg: np.ndarray,
        usla_deg: np.ndarray,
        *,
        compute_sens: bool,
    ) -> Dict[str, np.ndarray]:
        """
        Run the anisotropic 1-D forward model.

        Parameters
        ----------
        h_m, rop, ustr_deg, udip_deg, usla_deg
            Model arrays.
        compute_sens
            If True, request sensitivities for gradient evaluation.

        Returns
        -------
        dict
            Output dictionary from :func:`aniso.aniso1d_impedance_sens`.
        """
        out = aniso.aniso1d_impedance_sens(
            self.periods_s,
            h_m,
            rop,
            ustr_deg,
            udip_deg,
            usla_deg,
            is_iso=self.is_iso,
            compute_sens=compute_sens,
        )
        return out

    def pack_prediction(self, Z_pred: np.ndarray) -> np.ndarray:
        """
        Pack model prediction into the data vector layout.

        Parameters
        ----------
        Z_pred
            Predicted impedance (nper,2,2).

        Returns
        -------
        ndarray
            Packed prediction vector matching ``self.y``.
        """
        yZp = _pack_Z(Z_pred, self.z_comps)
        if not self.use_pt:
            return yZp

        Pp = phase_tensor_from_Z(Z_pred)
        yPp = _pack_P(Pp, self.pt_comps)
        return np.concatenate([yZp, yPp], axis=0)

    def loglike_from_pred(self, y_pred: np.ndarray) -> float:
        """
        Compute Gaussian log-likelihood given packed predictions.

        Parameters
        ----------
        y_pred
            Packed prediction vector (same length as ``self.y``).

        Returns
        -------
        float
            Log-likelihood value.
        """
        r = self.y - y_pred
        s = self.s
        s2 = s * s
        return float(-0.5 * np.sum((r * r) / s2 + self._log2pi + np.log(s2)))

    def grad_loglike(
        self,
        theta: np.ndarray,
        h_m_fixed: Optional[np.ndarray],
        rop_fixed: Optional[np.ndarray],
        ustr_fixed: Optional[np.ndarray],
        udip_fixed: Optional[np.ndarray],
        usla_fixed: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Compute gradient of the Gaussian log-likelihood w.r.t. theta.

        Parameters
        ----------
        theta
            Current parameter vector.
        h_m_fixed, rop_fixed, ustr_fixed, udip_fixed, usla_fixed
            Fixed reference arrays used in theta->model conversion.

        Returns
        -------
        ndarray
            Gradient vector with same shape as theta (float64).

        Notes
        -----
        - Uses sensitivities from :func:`aniso.aniso1d_impedance_sens`.
        - For Z-part, chain rule converts complex derivatives into derivatives
          of packed real/imag residuals.
        - For PT-part, uses dP = -A dX P + A dY with A = inv(X), X=Re(Z), Y=Im(Z).
        """
        theta = _as_1d_float(theta, "theta")
        h_m, rop, ustr, udip, usla = theta_to_model(
            theta, self.spec, h_m_fixed, rop_fixed, ustr_fixed, udip_fixed, usla_fixed
        )

        out = self.forward(h_m, rop, ustr, udip, usla, compute_sens=True)
        Zp = out["Z"]

        # Residual weights
        y_pred = self.pack_prediction(Zp)
        r = self.y - y_pred
        w = r / (self.s * self.s)  # weights for derivative of pred

        # Precompute per-period P and inv(Re(Z)) for PT gradients
        if self.use_pt:
            X = np.real(Zp)
            Y = np.imag(Zp)
            P = phase_tensor_from_Z(Zp)
            invX = np.empty_like(X)
            for k in range(self.nper):
                invX[k] = np.linalg.inv(X[k])

        # Helper to compute contribution for a single complex dZ array (nper,2,2)
        def add_from_dZ(dZ: np.ndarray, grad_acc: np.ndarray, theta_slice: slice, transform: Optional[Tuple[str, np.ndarray]] = None):
            """
            Add gradient contributions for parameters associated with dZ.

            Parameters
            ----------
            dZ
                Complex derivative (nper,2,2) for a single scalar parameter.
            grad_acc
                Gradient accumulator array (ndim,).
            theta_slice
                Slice in theta corresponding to this parameter (scalar).
            transform
                Optional tuple describing additional scaling due to parameter transform:
                ("log10", x_linear) scales by ln(10)*x_linear.
            """
            # Z-part packing
            idxZ = _comp_indices(self.z_comps)
            p = 0
            g = 0.0
            for k in range(self.nper):
                for (i, j) in idxZ:
                    dz = dZ[k, i, j]
                    # packed order: Re, Im
                    g += w[p] * np.real(dz) + w[p + 1] * np.imag(dz)
                    p += 2

            # PT-part packing
            if self.use_pt:
                idxP = _comp_indices(self.pt_comps)
                # PT weights start after Z block
                pP0 = self.yZ.size
                pp = 0
                for k in range(self.nper):
                    # dX, dY for this parameter
                    dX = np.real(dZ[k])
                    dY = np.imag(dZ[k])
                    A = invX[k]
                    dP = -A @ dX @ P[k] + A @ dY
                    for (i, j) in idxP:
                        g += w[pP0 + pp] * dP[i, j]
                        pp += 1

            if transform is not None:
                kind, xlin = transform
                if kind == "log10":
                    g *= (np.log(10.0) * float(xlin))
            grad_acc[theta_slice] += g

        ndim = self.spec.ndim()
        grad = np.zeros(ndim, dtype=np.float64)

        p = 0

        # Thickness params
        if not self.spec.fix_h:
            nh = self.spec.nh()
            dZ_dh = out["dZ_dh_m"]  # (nper,nl,2,2)
            for j in range(nh):
                # d/dlog10(h) = ln(10)*h * d/dh
                add_from_dZ(dZ_dh[:, j, :, :], grad, slice(p + j, p + j + 1), ("log10", h_m[j]))
            p += nh

        # Resistivity params
        dZ_drop = out["dZ_drop"]  # (nper,nl,3,2,2)
        nrho = self.spec.nrho()
        log10_rop = theta[p : p + nrho].reshape((self.spec.nl, 3))
        # rop linear:
        rop_lin = 10.0 ** log10_rop
        # For each rop scalar
        idx = 0
        for il in range(self.spec.nl):
            for ir in range(3):
                add_from_dZ(dZ_drop[:, il, ir, :, :], grad, slice(p + idx, p + idx + 1), ("log10", rop_lin[il, ir]))
                idx += 1
        p += nrho

        # Angle params (degrees) - already in degrees
        dZ_ustr = out["dZ_dustr_deg"]  # (nper,nl,2,2)
        dZ_udip = out["dZ_dudip_deg"]
        dZ_usla = out["dZ_dusla_deg"]
        idx = 0
        for il in range(self.spec.nl):
            add_from_dZ(dZ_ustr[:, il, :, :], grad, slice(p + idx, p + idx + 1), None)
            idx += 1
            add_from_dZ(dZ_udip[:, il, :, :], grad, slice(p + idx, p + idx + 1), None)
            idx += 1
            add_from_dZ(dZ_usla[:, il, :, :], grad, slice(p + idx, p + idx + 1), None)
            idx += 1

        return grad


class AnisoMTLogLikeOp(Op):
    """
    PyTensor Op computing the Gaussian log-likelihood for anisotropic 1-D MT.

    The Op takes a single input:
        theta : vector (ndim,)

    and returns:
        loglike : scalar

    Parameters
    ----------
    ctx
        Context with packed observations and forward-model settings.
    h_m_fixed, rop_fixed, ustr_fixed, udip_fixed, usla_fixed
        Fixed arrays used by theta->model conversion (for non-sampled entries).
    enable_grad
        If True, :meth:`grad` is implemented (via :class:`AnisoMTLogLikeGradOp`).

    Notes
    -----
    - In ``perform`` we evaluate the forward model without sensitivities to keep
      likelihood evaluation fast. The gradient Op evaluates with sensitivities.
    """
    itypes = [pt.dvector]
    otypes = [pt.dscalar]

    def __init__(
        self,
        ctx: _AnisoMTContext,
        *,
        h_m_fixed: Optional[np.ndarray],
        rop_fixed: Optional[np.ndarray],
        ustr_fixed: Optional[np.ndarray],
        udip_fixed: Optional[np.ndarray],
        usla_fixed: Optional[np.ndarray],
        enable_grad: bool,
    ) -> None:
        self.ctx = ctx
        self.h_m_fixed = None if h_m_fixed is None else np.asarray(h_m_fixed, dtype=np.float64)
        self.rop_fixed = None if rop_fixed is None else np.asarray(rop_fixed, dtype=np.float64)
        self.ustr_fixed = None if ustr_fixed is None else np.asarray(ustr_fixed, dtype=np.float64)
        self.udip_fixed = None if udip_fixed is None else np.asarray(udip_fixed, dtype=np.float64)
        self.usla_fixed = None if usla_fixed is None else np.asarray(usla_fixed, dtype=np.float64)
        self.enable_grad = bool(enable_grad)
        self._grad_op = None

    def make_node(self, theta):
        """
        Create an Apply node for the Op.

        Parameters
        ----------
        theta
            PyTensor vector variable.

        Returns
        -------
        Apply
            PyTensor apply node.
        """
        theta = pt.as_tensor_variable(theta)
        if theta.type.ndim != 1:
            raise TypeError("theta must be a 1-D vector.")
        return Apply(self, [theta], [pt.dscalar()])

    def perform(self, node, inputs, outputs):
        """
        Evaluate log-likelihood at a numeric theta.

        Parameters
        ----------
        node
            PyTensor node (unused).
        inputs
            List containing theta ndarray.
        outputs
            List containing output storage.
        """
        (theta,) = inputs
        h_m, rop, ustr, udip, usla = theta_to_model(
            theta,
            self.ctx.spec,
            self.h_m_fixed,
            self.rop_fixed,
            self.ustr_fixed,
            self.udip_fixed,
            self.usla_fixed,
        )
        out = self.ctx.forward(h_m, rop, ustr, udip, usla, compute_sens=False)
        Zp = out["Z"]
        y_pred = self.ctx.pack_prediction(Zp)
        ll = self.ctx.loglike_from_pred(y_pred)
        outputs[0][0] = np.asarray(ll, dtype=np.float64)

    def grad(self, inputs, g_outputs):
        """
        Gradient of the Op output w.r.t. inputs.

        Parameters
        ----------
        inputs
            List with theta symbolic.
        g_outputs
            List with upstream gradient (scalar).

        Returns
        -------
        list
            List containing the symbolic gradient vector.

        Notes
        -----
        This uses a separate numeric gradient Op to keep the logic clean.
        """
        if not self.enable_grad:
            return [pt.zeros_like(inputs[0])]

        if self._grad_op is None:
            self._grad_op = AnisoMTLogLikeGradOp(
                self.ctx,
                h_m_fixed=self.h_m_fixed,
                rop_fixed=self.rop_fixed,
                ustr_fixed=self.ustr_fixed,
                udip_fixed=self.udip_fixed,
                usla_fixed=self.usla_fixed,
            )
        (theta,) = inputs
        (g_out,) = g_outputs
        return [g_out * self._grad_op(theta)]


class AnisoMTLogLikeGradOp(Op):
    """
    PyTensor Op computing gradient of log-likelihood w.r.t. theta.

    Parameters
    ----------
    ctx
        Shared context.
    h_m_fixed, rop_fixed, ustr_fixed, udip_fixed, usla_fixed
        Fixed reference arrays.
    """
    itypes = [pt.dvector]
    otypes = [pt.dvector]

    def __init__(
        self,
        ctx: _AnisoMTContext,
        *,
        h_m_fixed: Optional[np.ndarray],
        rop_fixed: Optional[np.ndarray],
        ustr_fixed: Optional[np.ndarray],
        udip_fixed: Optional[np.ndarray],
        usla_fixed: Optional[np.ndarray],
    ) -> None:
        self.ctx = ctx
        self.h_m_fixed = h_m_fixed
        self.rop_fixed = rop_fixed
        self.ustr_fixed = ustr_fixed
        self.udip_fixed = udip_fixed
        self.usla_fixed = usla_fixed

    def make_node(self, theta):
        """
        Create apply node.

        Parameters
        ----------
        theta
            1-D vector variable.

        Returns
        -------
        Apply
            Node producing gradient vector.
        """
        theta = pt.as_tensor_variable(theta)
        if theta.type.ndim != 1:
            raise TypeError("theta must be a 1-D vector.")
        return Apply(self, [theta], [pt.dvector()])

    def perform(self, node, inputs, outputs):
        """
        Numeric gradient evaluation.

        Parameters
        ----------
        node
            PyTensor node (unused).
        inputs
            [theta ndarray]
        outputs
            output storage list.
        """
        (theta,) = inputs
        g = self.ctx.grad_loglike(
            theta,
            self.h_m_fixed,
            self.rop_fixed,
            self.ustr_fixed,
            self.udip_fixed,
            self.usla_fixed,
        )
        outputs[0][0] = g.astype(np.float64)


def build_pymc_model(
    site: Mapping,
    *,
    spec: ParamSpec,
    h_m0: Optional[np.ndarray] = None,
    rop0: Optional[np.ndarray] = None,
    ustr_deg0: Optional[np.ndarray] = None,
    udip_deg0: Optional[np.ndarray] = None,
    usla_deg0: Optional[np.ndarray] = None,
    is_iso: Optional[np.ndarray] = None,
    is_fix: Optional[np.ndarray] = None,
    use_pt: bool = False,
    z_comps: Sequence[str] = ("xy", "yx"),
    pt_comps: Sequence[str] = ("xx", "xy", "yx", "yy"),
    compute_pt_if_missing: bool = True,
    sigma_floor_Z: float = 0.0,
    sigma_floor_P: float = 0.0,
    enable_grad: bool = False,
    prior_kind: str = "uniform",
) -> Tuple[pm.Model, Dict[str, object]]:
    """
    Build a PyMC model for anisotropic 1-D MT inversion.

    Parameters
    ----------
    site
        Site dict with keys described in module docstring.
    spec
        Parameter specification (sizes, transforms, bounds).
    h_m0, rop0, ustr_deg0, udip_deg0, usla_deg0
        Reference arrays used to initialize fixed entries. If a quantity is
        sampled, the prior mean/center can be set close to these values by
        choosing ``prior_kind="normal"``.
    is_iso
        Optional (nl,) isotropy flag passed to forward model.
    is_fix
        Optional (nl,) per-layer flag. If True, all parameters of that layer are held fixed
        at their initial values and are not sampled.
    use_pt
        If True, include phase tensor likelihood.
    z_comps, pt_comps
        Components to include.
    compute_pt_if_missing
        If True and ``use_pt`` is True, compute P from Z if missing.
    sigma_floor_Z, sigma_floor_P
        Minimum sigmas to prevent singular likelihood.
    enable_grad
        If True, likelihood Op provides a gradient (enables NUTS later).
    prior_kind
        "uniform" or "normal".
        - uniform: bounded Uniform priors on transformed variables
        - normal: Normal priors around initial values with wide stds, plus hard bounds

    Returns
    -------
    model, info
        PyMC model and an info dict containing:
        - "param_names": list of theta parameter names in order
        - "theta_init": initial theta vector (float64)
        - "loglike_op": the PyTensor Op used for likelihood
        - "context": internal context (for debugging)

    Notes
    -----
    For gradient-based samplers, you must set ``enable_grad=True`` and then
    choose an appropriate step method when calling :func:`sample_pymc`.
    """
    # periods from freq
    if "freq" not in site:
        raise KeyError("site must contain 'freq' [Hz].")
    freq = _as_1d_float(site["freq"], "freq")
    periods_s = 1.0 / freq

    # Context
    ctx = _AnisoMTContext(
        site,
        periods_s,
        spec,
        use_pt=use_pt,
        z_comps=z_comps,
        pt_comps=pt_comps,
        compute_pt_if_missing=compute_pt_if_missing,
        sigma_floor_Z=sigma_floor_Z,
        sigma_floor_P=sigma_floor_P,
        is_iso=is_iso,
    )

    # Per-layer fixed flags (freeze entire layer during sampling)
    is_fix_arr = normalize_is_fix(is_fix, nl=spec.nl)

    # Prepare initial theta from reference arrays
    nl = spec.nl
    if h_m0 is None:
        h_m0 = np.ones(nl, dtype=np.float64)
        h_m0[-1] = 0.0
    if rop0 is None:
        rop0 = np.ones((nl, 3), dtype=np.float64) * 100.0
    if ustr_deg0 is None:
        ustr_deg0 = np.zeros(nl, dtype=np.float64)
    if udip_deg0 is None:
        udip_deg0 = np.zeros(nl, dtype=np.float64)
    if usla_deg0 is None:
        usla_deg0 = np.zeros(nl, dtype=np.float64)

    h_m0 = np.asarray(h_m0, dtype=np.float64)
    rop0 = np.asarray(rop0, dtype=np.float64)
    ustr_deg0 = np.asarray(ustr_deg0, dtype=np.float64)
    udip_deg0 = np.asarray(udip_deg0, dtype=np.float64)
    usla_deg0 = np.asarray(usla_deg0, dtype=np.float64)

    # Build theta_init
    parts = []
    names = []

    if not spec.fix_h:
        nh = spec.nh()
        h_init = np.clip(h_m0[:nh], 10.0 ** spec.log10_h_bounds[0], 10.0 ** spec.log10_h_bounds[1])
        parts.append(np.log10(h_init))
        names += [f"log10_h[{i}]" for i in range(nh)]

    rop_init = np.clip(rop0, 10.0 ** spec.log10_rho_bounds[0], 10.0 ** spec.log10_rho_bounds[1])
    parts.append(np.log10(rop_init).reshape(-1))
    names += [f"log10_rop[{i},{j}]" for i in range(nl) for j in range(3)]

    parts.append(np.vstack([ustr_deg0, udip_deg0, usla_deg0]).T.reshape(-1))
    names += [f"ustr_deg[{i}]" for i in range(nl)]
    names += [f"udip_deg[{i}]" for i in range(nl)]
    names += [f"usla_deg[{i}]" for i in range(nl)]

    theta_init = np.concatenate(parts).astype(np.float64)

    # Likelihood Op
    loglike_op = AnisoMTLogLikeOp(
        ctx,
        h_m_fixed=h_m0,
        rop_fixed=rop0,
        ustr_fixed=ustr_deg0,
        udip_fixed=udip_deg0,
        usla_fixed=usla_deg0,
        enable_grad=enable_grad,
    )

    # Build PyMC model
    with pm.Model() as model:
        parts_full = []

        def _merge_fixed(
            *,
            name: str,
            full_mu: np.ndarray,
            free_idx: np.ndarray,
            bounds: Tuple[float, float],
            prior_kind: str,
            sigma: float,
        ) -> pt.TensorVariable:
            """Create a vector with free entries sampled and fixed entries held constant.

            The returned tensor has the same length as ``full_mu``.
            """
            lo, hi = float(bounds[0]), float(bounds[1])
            mu = np.asarray(full_mu, dtype=np.float64).ravel()
            mu = np.clip(mu, lo, hi)
            n = int(mu.size)
            idx = np.asarray(free_idx, dtype=np.int64).ravel()
            if idx.size == n:
                # nothing fixed
                if prior_kind == "normal":
                    sd = np.ones(n, dtype=np.float64) * float(sigma)
                    x = pm.TruncatedNormal(name, mu=mu, sigma=sd, lower=lo, upper=hi, shape=n)
                else:
                    x = pm.Uniform(name, lower=lo, upper=hi, shape=n)
                return x
            if idx.size == 0:
                x_full = pt.as_tensor_variable(mu)
                pm.Deterministic(name, x_full)
                return x_full

            # Sample only free subset
            mu_free = mu[idx]
            if prior_kind == "normal":
                sd_free = np.ones(idx.size, dtype=np.float64) * float(sigma)
                x_free = pm.TruncatedNormal(
                    f"{name}_free", mu=mu_free, sigma=sd_free, lower=lo, upper=hi, shape=int(idx.size)
                )
            else:
                x_free = pm.Uniform(f"{name}_free", lower=lo, upper=hi, shape=int(idx.size))

            x_full = pt.as_tensor_variable(mu)
            x_full = pt.set_subtensor(x_full[idx], x_free)
            pm.Deterministic(name, x_full)
            return x_full

        # Thickness priors in log10 (per-layer fixing supported)
        if not spec.fix_h:
            nh = spec.nh()
            free_idx_h = np.where(~is_fix_arr[:nh])[0]
            log10_h_full = _merge_fixed(
                name="log10_h",
                full_mu=theta_init[:nh],
                free_idx=free_idx_h,
                bounds=spec.log10_h_bounds,
                prior_kind=prior_kind,
                sigma=1.0,
            )
            parts_full.append(log10_h_full)

        # Resistivity priors in log10 (freeze all 3 principal values per fixed layer)
        nrho = spec.nrho()
        mu_rop = theta_init[spec.nh() : spec.nh() + nrho]
        free_layers = np.where(~is_fix_arr)[0]
        if free_layers.size:
            free_idx_rop = np.concatenate([np.arange(i * 3, i * 3 + 3, dtype=np.int64) for i in free_layers])
        else:
            free_idx_rop = np.array([], dtype=np.int64)
        log10_rop_full = _merge_fixed(
            name="log10_rop",
            full_mu=mu_rop,
            free_idx=free_idx_rop,
            bounds=spec.log10_rho_bounds,
            prior_kind=prior_kind,
            sigma=1.5,
        )
        parts_full.append(log10_rop_full)

        # Angle priors (degrees) (freeze per-layer)
        free_idx_ang = free_layers
        ustr_full = _merge_fixed(
            name="ustr_deg",
            full_mu=ustr_deg0,
            free_idx=free_idx_ang,
            bounds=spec.ustr_bounds_deg,
            prior_kind=prior_kind,
            sigma=60.0,
        )
        parts_full.append(ustr_full)

        udip_full = _merge_fixed(
            name="udip_deg",
            full_mu=udip_deg0,
            free_idx=free_idx_ang,
            bounds=spec.udip_bounds_deg,
            prior_kind=prior_kind,
            sigma=30.0,
        )
        parts_full.append(udip_full)

        usla_full = _merge_fixed(
            name="usla_deg",
            full_mu=usla_deg0,
            free_idx=free_idx_ang,
            bounds=spec.usla_bounds_deg,
            prior_kind=prior_kind,
            sigma=60.0,
        )
        parts_full.append(usla_full)

        # Concatenate into full theta tensor (same order as theta_init/names)
        theta_rv = pt.concatenate([pt.flatten(v) for v in parts_full], axis=0)
        pm.Deterministic("theta", theta_rv)

        # Add log-likelihood as Potential
        ll = loglike_op(theta_rv)
        pm.Potential("loglike", ll)

    info = {
        "param_names": names,
        "theta_init": theta_init,
        "is_fix": is_fix_arr,
        "loglike_op": loglike_op,
        "context": ctx,
        "periods_s": periods_s,
    }
    return model, info


def sample_pymc(
    model: pm.Model,
    *,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 2,
    cores: int = 2,
    step_method: str = "auto",
    target_accept: float = 0.85,
    random_seed: Optional[int] = None,
    progressbar: bool = True,
) -> "arviz.InferenceData":
    """
    Run PyMC sampling for a model created by :func:`build_pymc_model`.

    Parameters
    ----------
    model
        PyMC model.
    draws
        Number of posterior draws.
    tune
        Number of tuning steps.
    chains
        Number of MCMC chains.
    cores
        Number of CPU cores to use.
    step_method
        "auto", "demetropolis", "metropolis", or "nuts".
        - auto: try NUTS if the model appears differentiable; otherwise DEMetropolisZ.
        - demetropolis: DEMetropolisZ (good default for black-box likelihood)
        - metropolis: Metropolis
        - nuts: NUTS (requires gradients; works if likelihood Op implements grad)
    target_accept
        Target acceptance probability (for NUTS).
    random_seed
        Random seed.
    progressbar
        Show progress bar.

    Returns
    -------
    arviz.InferenceData
        Sampling results (requires arviz, which is a PyMC dependency).

    Notes
    -----
    If you built the model with ``enable_grad=False``, using ``step_method="nuts"``
    will likely fail or be very slow because the likelihood has no gradient.
    """
    step_method = str(step_method).lower().strip()

    with model:
        step = None
        if step_method == "demetropolis":
            step = pm.DEMetropolisZ()
        elif step_method == "metropolis":
            step = pm.Metropolis()
        elif step_method == "nuts":
            step = pm.NUTS(target_accept=target_accept)
        elif step_method == "auto":
            # Try NUTS; if it fails, fall back to DEMetropolisZ.
            try:
                step = pm.NUTS(target_accept=target_accept)
            except Exception:
                step = pm.DEMetropolisZ()
        else:
            raise ValueError("step_method must be one of: auto, demetropolis, metropolis, nuts")

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            step=step,
            random_seed=random_seed,
            progressbar=progressbar,
            return_inferencedata=True,
        )

    return idata


# =============================================================================
# Helpers for the script driver (mt_aniso1d_sampler.py)
# =============================================================================

def ensure_dir(path: Union[str, "Path"]) -> str:
    """
    Ensure a directory exists and return its string path.

    Parameters
    ----------
    path : str or Path
        Directory path.

    Returns
    -------
    str
        Directory path as string.
    """
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p.as_posix()


def glob_inputs(pattern: str) -> List[str]:
    """
    Expand a glob pattern into a sorted list of file paths.

    Parameters
    ----------
    pattern : str
        Glob pattern for EDI or NPZ files.

    Returns
    -------
    list of str
        Sorted matching files.
    """
    import glob
    files = glob.glob(os.path.expanduser(pattern))
    files.sort()
    return files


def load_model_npz(path: Union[str, "Path"]) -> Dict[str, np.ndarray]:
    """
    Load a model template stored as an NPZ.

    The NPZ is expected to contain at least:
    - rop : (nl,3)
    and optionally:
    - h_m, ustr_deg, udip_deg, usla_deg, is_iso, is_fix
    """
    p = Path(path).expanduser()
    d = dict(np.load(p, allow_pickle=True))
    for k, v in list(d.items()):
        if isinstance(v, np.ndarray) and v.shape == () and v.dtype == object:
            d[k] = v.item()
    # Normalize to ensure shapes are consistent (notably rop orientation and flag lengths).
    return normalize_model(d)  # type: ignore[return-value]


def save_model_npz(model: Mapping[str, object], path: Union[str, "Path"]) -> None:
    """Save a model dict to NPZ.

    The model is normalized before saving so that the stored NPZ is directly
    usable by the sampler.
    """
    p = Path(path).expanduser()
    md = normalize_model(model)
    np.savez_compressed(p, **dict(md))


def model_from_direct(model_direct: object) -> Dict[str, np.ndarray]:
    """
    Convert an in-code model specification to a model dict.

    Accepted forms:
    - dict with keys 'h_m','rop','ustr_deg','udip_deg','usla_deg' (+ optional 'is_iso')
    - array-like (nl,7) or (nl,8): [h, rop1, rop2, rop3, ustr, udip, usla, (is_iso)]
    """
    if isinstance(model_direct, Mapping):
        md = dict(model_direct)
        # Preserve any extra metadata (e.g. "prior_name") and normalize only the
        # physical model arrays.
        out = dict(md)
        out["h_m"] = np.asarray(md.get("h_m", md.get("h", None)), dtype=float)
        out["rop"] = np.asarray(md["rop"], dtype=float)
        out["ustr_deg"] = np.asarray(md.get("ustr_deg", md.get("ustr", 0.0)), dtype=float)
        out["udip_deg"] = np.asarray(md.get("udip_deg", md.get("udip", 0.0)), dtype=float)
        out["usla_deg"] = np.asarray(md.get("usla_deg", md.get("usla", 0.0)), dtype=float)
        if "is_iso" in md:
            out["is_iso"] = np.asarray(md.get("is_iso"), dtype=bool)
        if "is_fix" in md:
            out["is_fix"] = np.asarray(md.get("is_fix"), dtype=bool)
        return normalize_model(out)

    arr = np.asarray(model_direct)
    if arr.ndim != 2 or arr.shape[1] not in (7, 8, 9):
        raise ValueError("MODEL_DIRECT must be dict or array-like (nl,7/8/9).")
    out = {
        "h_m": np.asarray(arr[:, 0], dtype=float),
        "rop": np.asarray(arr[:, 1:4], dtype=float),
        "ustr_deg": np.asarray(arr[:, 4], dtype=float),
        "udip_deg": np.asarray(arr[:, 5], dtype=float),
        "usla_deg": np.asarray(arr[:, 6], dtype=float),
    }
    if arr.shape[1] >= 8:
        out["is_iso"] = np.asarray(arr[:, 7], dtype=bool)
    if arr.shape[1] == 9:
        out["is_fix"] = np.asarray(arr[:, 8], dtype=bool)
    return normalize_model(out)


def load_site(path: Union[str, "Path"]) -> Dict[str, object]:
    """
    Load a site dict from EDI or NPZ using data_proc.py (no duplicated parsing).
    """
    from data_proc import load_edi, load_npz
    p = Path(path).expanduser()
    if p.suffix.lower() == ".edi":
        site = load_edi(p.as_posix())
    elif p.suffix.lower() == ".npz":
        site = load_npz(p.as_posix())
    else:
        raise ValueError(f"Unsupported input type: {p}")
    if "station" not in site:
        site["station"] = p.stem
    return site


def ensure_phase_tensor(site: Mapping[str, object], *, nsim: int = 200) -> Dict[str, object]:
    """
    Ensure phase tensor P/P_err is present, using data_proc.compute_pt(...).
    """
    from data_proc import compute_pt
    out = dict(site)
    if out.get("P", None) is not None:
        return out
    Z = np.asarray(out["Z"])
    Z_err = out.get("Z_err", None)
    err_kind = str(out.get("err_kind", "var"))
    P, P_err = compute_pt(Z, Z_err, err_kind=err_kind, err_method="bootstrap", nsim=int(nsim))
    out["P"] = P
    out["P_err"] = P_err
    return out


def save_idata(idata: "arviz.InferenceData", path: Union[str, "Path"]) -> None:
    """Save ArviZ InferenceData to NetCDF."""
    p = Path(path).expanduser()
    idata.to_netcdf(p.as_posix())


def posterior_theta_stats(
    idata: "arviz.InferenceData",
    *,
    qpairs: Sequence[Tuple[float, float]] = ((0.1, 0.9),),
) -> Dict[str, np.ndarray]:
    """Compute theta median and quantile bands."""
    arr = np.asarray(idata.posterior["theta"])
    theta = arr.reshape(-1, arr.shape[-1])
    med = np.quantile(theta, 0.5, axis=0)
    qlos, qhis = [], []
    for qlo, qhi in qpairs:
        qlos.append(np.quantile(theta, qlo, axis=0))
        qhis.append(np.quantile(theta, qhi, axis=0))
    return dict(
        theta_med=med,
        theta_qpairs=np.array(list(qpairs), dtype=float),
        theta_qlo=np.array(qlos, dtype=float),
        theta_qhi=np.array(qhis, dtype=float),
    )


def build_summary_npz(
    *,
    station: str,
    site: Mapping[str, object],
    idata: "arviz.InferenceData",
    spec: ParamSpec,
    model0: Mapping[str, np.ndarray],
    info: Mapping[str, object],
    qpairs: Sequence[Tuple[float, float]] = ((0.1, 0.9),),
) -> Dict[str, object]:
    """
    Build a compact summary dict (NPZ-ready).

    Stores Z_obs and Z_pred at the median model. Derived quantities (rho/phase/PT)
    should be computed in plotting via data_proc to avoid duplication.
    """
    from aniso import aniso1d_impedance_sens

    # Make sure the template model is consistent (notably rop orientation and flag lengths).
    model0 = normalize_model(model0)

    s: Dict[str, object] = {}
    s["station"] = station
    s["freq"] = np.asarray(site["freq"], dtype=float)
    s["Z_obs"] = np.asarray(site["Z"])
    s["Z_err"] = site.get("Z_err", None)
    s["err_kind"] = site.get("err_kind", "var")
    if "P" in site:
        s["P_obs"] = site.get("P")
        s["P_err"] = site.get("P_err", None)

    ts = posterior_theta_stats(idata, qpairs=qpairs)
    s.update(ts)

    pn = info.get("param_names", None)
    if pn is not None:
        s["param_names"] = np.array(pn, dtype=object)

    # Convert theta median -> model
    theta_med = ts["theta_med"]
    h_m, rop, ustr, udip, usla = theta_to_model(
        theta_med,
        spec=spec,
        h_m_fixed=model0.get("h_m", None),
        rop_fixed=model0.get("rop", None),
        ustr_fixed=model0.get("ustr_deg", None),
        udip_fixed=model0.get("udip_deg", None),
        usla_fixed=model0.get("usla_deg", None),
    )
    s["h_m_med"] = h_m
    s["rop_med"] = rop
    s["ustr_deg_med"] = ustr
    s["udip_deg_med"] = udip
    s["usla_deg_med"] = usla
    if "is_iso" in model0:
        s["is_iso"] = np.asarray(model0["is_iso"], dtype=bool)
    if "is_fix" in model0:
        s["is_fix"] = np.asarray(model0["is_fix"], dtype=bool)

    # Layer bottom depths for plotting (ensure length nl including basement)
    if h_m is not None and len(h_m) >= 1:
        if float(np.asarray(h_m, dtype=float)[-1]) > 0.0:
            s["z_bot_med"] = np.cumsum(np.asarray(h_m, dtype=float))
        else:
            z_int = np.cumsum(np.asarray(h_m, dtype=float)[:-1])
            if z_int.size == 0:
                s["z_bot_med"] = np.array([max(1.0, float(np.asarray(h_m, dtype=float)[0]))], dtype=float)
            else:
                z_last = float(z_int[-1])
                ext = max(1.0, 0.25 * max(z_last, 1.0))
                s["z_bot_med"] = np.r_[z_int, z_last + ext]

    periods_s = 1.0 / np.asarray(s["freq"], dtype=float)
    fwd = aniso1d_impedance_sens(
        periods_s=periods_s,
        h_m=h_m,
        rop=rop,
        ustr_deg=ustr,
        udip_deg=udip,
        usla_deg=usla,
        is_iso=model0.get("is_iso", None),
        compute_sens=False,
    )
    if isinstance(fwd, Mapping):
        s["Z_pred"] = fwd.get("Z", None)
    else:
        s["Z_pred"] = getattr(fwd, "Z", None)

    return s


def save_summary_npz(summary: Mapping[str, object], path: Union[str, "Path"]) -> None:
    """Save summary dict as compressed NPZ."""
    p = Path(path).expanduser()
    np.savez_compressed(p, **dict(summary))
