#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mt_aniso1d_sampler.py

Script-style PyMC driver for anisotropic 1-D MT inversion (impedance; optional
phase tensor).

This file is intentionally **not a CLI**. Edit the "USER CONFIG" section and run:

    python mt_aniso1d_sampler.py

Uses only the uploaded project files:
- aniso.py : forward model + sensitivities
- mcmc.py  : PyMC model builder + sampler wrappers
- data_proc.py : EDI/NPZ input (and optional phase tensor error propagation)

Outputs
-------
For each input site file, this script writes:
- <outdir>/<station>_pmc.nc : ArviZ InferenceData NetCDF
- <outdir>/<station>_pmc_summary.npz : compact summary (median theta + model + predictions)

Notes
-----
- No VTF/tipper: in 1-D MT the tipper is identically zero.
- Default sampling is gradient-free (DEMetropolisZ). If you enable analytic
  gradients (ENABLE_GRAD=True), you *may* experiment with NUTS.

Author: Volker Rath (DIAS)
Created with the help of ChatGPT (GPT-5 Thinking) on 2026-01-20
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Local project modules (uploaded files)
import data_proc
import mcmc


# =============================================================================
# USER CONFIG
# =============================================================================

# Input data
INPUT_GLOB = "/home/vrath/MT_Data/waldim/edi_jc/*.edi"   # can also point to *.npz
OUTDIR = "/home/vrath/MT_Data/waldim/edi_jc/pmc_out"     # will be created

# Starting model (template)
# Provide a NPZ with at least rop (nl,3); optionally h_m (nl,), angles, is_iso.
MODEL_NPZ = "/home/vrath/MT_Data/waldim/edi_jc/model.npz"
# Optional: define the starting model directly (see mt_aniso1d_forward.py for examples).
# If MODEL_DIRECT is not None, it is converted to the internal arrays and written
# to MODEL_DIRECT_SAVE_PATH (default: MODEL_NPZ), then used as the starting model.
MODEL_DIRECT = None  # dict or (nl,7/8) array-like
MODEL_DIRECT_SAVE_PATH = MODEL_NPZ
MODEL_DIRECT_OVERWRITE = True


# Data choices
USE_PT = True                # include phase tensor likelihood
PT_ERR_NSIM = 200            # bootstrap realisations for PT error propagation
Z_COMPS = ("xy", "yx")        # impedance components used in likelihood
PT_COMPS = ("xx", "xy", "yx", "yy")

# Parameterization
FIX_H = True                 # if True: thicknesses are fixed from model; not sampled
ENABLE_GRAD = False          # if True: likelihood Op provides analytic gradients
PRIOR_KIND = "uniform"       # "uniform" or "normal"

# Likelihood sigma floors (avoid zero variance)
SIGMA_FLOOR_Z = 1e-12
SIGMA_FLOOR_P = 1e-6

# Sampling
STEP_METHOD = "demetropolis"  # "demetropolis" | "metropolis" | "nuts" | "auto"
DRAWS = 3000
TUNE = 1500
CHAINS = 2
CORES = 2
RANDOM_SEED = 20260120
PROGRESS = True

# Post-processing
THIN = 1                      # thin factor for summary medians (>=1)
# Optional posterior credible intervals for theta (percentile pairs).
# Examples: ((10, 90),) or ((16, 84), (5, 95)). Set to None/() to skip.
THETA_QPAIRS = ((10.0, 90.0),)



# =============================================================================
# Helpers
# =============================================================================

@dataclass
class Model0:
    """Container for initial model arrays."""

    h_m: np.ndarray           # (nl,)
    rop: np.ndarray           # (nl,3)
    ustr_deg: np.ndarray      # (nl,)
    udip_deg: np.ndarray      # (nl,)
    usla_deg: np.ndarray      # (nl,)
    is_iso: Optional[np.ndarray] = None  # (nl,) bool


def _ensure_dir(path: str | Path) -> Path:
    """Create directory if missing and return as Path."""
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _glob_inputs(pattern: str) -> List[Path]:
    """Return sorted list of input files matching a glob."""
    pat = str(Path(pattern).expanduser())
    files = sorted(Path().glob(pat) if ("*" in pat or "?" in pat or "[" in pat) else [Path(pat)])
    # The Path().glob(...) above uses current working directory; for absolute patterns use glob from pathlib directly
    if Path(pat).is_absolute():
        files = sorted(Path(pat).parent.glob(Path(pat).name))
    return [f.resolve() for f in files]


def load_model_npz(path: str | Path) -> Model0:
    """Load a starting model from NPZ.

    Expected keys (accepted aliases in parentheses):
    - rop (rop, rho, resistivity): (nl,3)
    - h_m (h, thickness): (nl,) thickness in meters, last entry may be 0 for basement
    - ustr_deg (ustr): (nl,)
    - udip_deg (udip): (nl,)
    - usla_deg (usla): (nl,)
    - is_iso (is_iso): (nl,) bool

    Returns
    -------
    Model0
        Container with arrays (defaults filled if missing).
    """
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"MODEL_NPZ not found: {p}")

    d = np.load(p, allow_pickle=True)

    def _get_first(keys: Sequence[str]) -> Optional[np.ndarray]:
        for k in keys:
            if k in d:
                return np.asarray(d[k])
        return None

    rop = _get_first(["rop", "rho", "resistivity", "rop0"]) 
    if rop is None:
        raise KeyError("MODEL_NPZ must contain 'rop' with shape (nl,3).")
    rop = np.asarray(rop, dtype=np.float64)
    if rop.ndim != 2 or rop.shape[1] != 3:
        raise ValueError(f"rop must have shape (nl,3), got {rop.shape}")
    nl = rop.shape[0]

    h_m = _get_first(["h_m", "h", "thickness", "h_m0"]) 
    if h_m is None:
        # default: unit thickness for nl-1 layers and 0 for basement
        h_m = np.ones(nl, dtype=np.float64)
        h_m[-1] = 0.0
    else:
        h_m = np.asarray(h_m, dtype=np.float64).reshape(-1)
        if h_m.shape != (nl,):
            raise ValueError(f"h_m must have shape ({nl},), got {h_m.shape}")

    def _get_ang(k: str, default: float = 0.0) -> np.ndarray:
        a = _get_first([k, k.replace("_deg", ""), k + "0"]) 
        if a is None:
            return np.ones(nl, dtype=np.float64) * default
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        if a.shape != (nl,):
            raise ValueError(f"{k} must have shape ({nl},), got {a.shape}")
        return a

    ustr_deg = _get_ang("ustr_deg", 0.0)
    udip_deg = _get_ang("udip_deg", 0.0)
    usla_deg = _get_ang("usla_deg", 0.0)

    is_iso = _get_first(["is_iso", "iso", "isotropic"]) 
    if is_iso is not None:
        is_iso = np.asarray(is_iso).astype(bool).reshape(-1)
        if is_iso.shape != (nl,):
            raise ValueError(f"is_iso must have shape ({nl},), got {is_iso.shape}")

    return Model0(h_m=h_m, rop=rop, ustr_deg=ustr_deg, udip_deg=udip_deg, usla_deg=usla_deg, is_iso=is_iso)

def model_from_direct(model) -> Model0:
    """Convert a "direct" model definition (as used in mt_aniso1d_forward.py) into :class:`Model0`.

    Accepted formats
    ----------------
    1) Array-like with shape (nl, 7) or (nl, 8), interpreted as rows:
       ``[h_m, rop_x, rop_y, rop_z, strike_deg, dip_deg, slant_deg, is_iso]``.
       If the 8th column is missing, ``is_iso`` defaults to False.
       ``is_iso`` follows the convention from mt_aniso1d_forward.py: 1 => isotropic.

    2) Dict with keys (case-insensitive, aliases allowed):
       - thickness: ``h_m`` or ``h``
       - resistivity: ``rop`` (or ``rho``), shape (nl,3) or (nl,)
       - angles: ``ustr_deg``/``ustr``, ``udip_deg``/``udip``, ``usla_deg``/``usla``
       - isotropy flag: ``is_iso``

    Returns
    -------
    Model0
        Parsed model with float arrays (and bool ``is_iso`` if provided).
    """

    if isinstance(model, dict):
        # Normalize keys
        keys = {str(k).lower(): k for k in model.keys()}

        def _get(*names, default=None):
            for nm in names:
                if nm.lower() in keys:
                    return model[keys[nm.lower()]]
            return default

        rop = _get("rop", "rho", "resistivity")
        if rop is None:
            raise KeyError("MODEL_DIRECT dict must contain 'rop' (nl,3) or (nl,)")

        rop = np.asarray(rop, dtype=np.float64)
        if rop.ndim == 1:
            rop = np.repeat(rop.reshape(-1, 1), 3, axis=1)
        if rop.ndim != 2 or rop.shape[1] != 3:
            raise ValueError(f"rop must have shape (nl,3) (or (nl,) for isotropic), got {rop.shape}")
        nl = int(rop.shape[0])

        h_m = _get("h_m", "h", "thickness")
        if h_m is None:
            h_m = np.ones(nl, dtype=np.float64)
            h_m[-1] = 0.0
        h_m = np.asarray(h_m, dtype=np.float64).reshape(-1)
        if h_m.shape != (nl,):
            raise ValueError(f"h_m must have shape ({nl},), got {h_m.shape}")

        def _ang(name_deg: str, name: str):
            a = _get(name_deg, name)
            if a is None:
                return np.zeros(nl, dtype=np.float64)
            a = np.asarray(a, dtype=np.float64).reshape(-1)
            if a.shape != (nl,):
                raise ValueError(f"{name_deg} must have shape ({nl},), got {a.shape}")
            return a

        ustr_deg = _ang("ustr_deg", "ustr")
        udip_deg = _ang("udip_deg", "udip")
        usla_deg = _ang("usla_deg", "usla")

        is_iso = _get("is_iso", "iso", "isotropic", default=None)
        if is_iso is not None:
            is_iso = np.asarray(is_iso).astype(bool).reshape(-1)
            if is_iso.shape != (nl,):
                raise ValueError(f"is_iso must have shape ({nl},), got {is_iso.shape}")

        return Model0(
            h_m=h_m,
            rop=rop,
            ustr_deg=ustr_deg,
            udip_deg=udip_deg,
            usla_deg=usla_deg,
            is_iso=is_iso,
        )

    # Array-like model
    arr = np.asarray(model, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 7:
        raise ValueError("MODEL_DIRECT array-like must have shape (nl,7) or (nl,8)")

    h_m = arr[:, 0].astype(np.float64)
    rop = arr[:, 1:4].astype(np.float64)
    ustr_deg = arr[:, 4].astype(np.float64)
    udip_deg = arr[:, 5].astype(np.float64)
    usla_deg = arr[:, 6].astype(np.float64)
    is_iso = None
    if arr.shape[1] >= 8:
        is_iso = (arr[:, 7].astype(np.int64) == 1)

    return Model0(
        h_m=h_m,
        rop=rop,
        ustr_deg=ustr_deg,
        udip_deg=udip_deg,
        usla_deg=usla_deg,
        is_iso=is_iso,
    )


def save_model_npz(model0: Model0, path: str | Path, *, overwrite: bool = True) -> Path:
    """Save a :class:`Model0` to a flat ``.npz`` archive compatible with :func:`load_model_npz`."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and not overwrite:
        return p

    kw = dict(
        h_m=np.asarray(model0.h_m, dtype=np.float64),
        rop=np.asarray(model0.rop, dtype=np.float64),
        ustr_deg=np.asarray(model0.ustr_deg, dtype=np.float64),
        udip_deg=np.asarray(model0.udip_deg, dtype=np.float64),
        usla_deg=np.asarray(model0.usla_deg, dtype=np.float64),
    )
    if model0.is_iso is not None:
        kw["is_iso"] = np.asarray(model0.is_iso).astype(bool)

    np.savez_compressed(p.as_posix(), **kw)
    return p




def load_site(path: Path, *, prefer_spectra: bool = True, err_kind: str = "var") -> Dict:
    """Load a site dict from .edi or .npz.

    Parameters
    ----------
    path
        Input file path.
    prefer_spectra
        Passed to :func:`data_proc.load_edi`.
    err_kind
        "var" or "std". Using "var" is common for FEMTIC-style uncertainty.

    Returns
    -------
    dict
        Site dictionary with at least freq, Z, Z_err, err_kind, and station.
    """
    ext = path.suffix.lower()
    if ext == ".npz":
        d = dict(np.load(path, allow_pickle=True))
        # np.load returns numpy scalars/arrays; station may be a 0-d array
        if "station" in d and isinstance(d["station"], np.ndarray) and d["station"].shape == ():
            d["station"] = str(d["station"].item())
        return d

    if ext == ".edi":
        d = data_proc.load_edi(path.as_posix(), prefer_spectra=prefer_spectra, err_kind=err_kind)
        return d

    raise ValueError(f"Unsupported input extension: {ext}")


def ensure_phase_tensor(site: Dict, *, nsim: int, err_kind: str = "var") -> None:
    """Ensure site dict has P and P_err keys (variance or std according to err_kind).

    This uses data_proc.compute_pt to propagate impedance errors (bootstrap by default).
    """
    if "P" in site and "P_err" in site and site["P"] is not None:
        return

    Z = np.asarray(site["Z"], dtype=np.complex128)
    Z_err = site.get("Z_err", None)
    P, P_err = data_proc.compute_pt(
        Z,
        Z_err,
        err_kind=site.get("err_kind", err_kind),
        err_method="bootstrap" if Z_err is not None else "none",
        nsim=int(nsim),
    )
    site["P"] = P
    site["P_err"] = P_err


def posterior_theta_median(idata, *, thin: int = 1, qpairs=None):
    """Return posterior summaries of the sampled ``theta`` vector.

    Parameters
    ----------
    idata
        ArviZ ``InferenceData`` returned by :func:`mcmc.sample_pymc`. Must contain
        ``idata.posterior['theta']`` with dims (chain, draw, theta_dim_0).
    thin : int, default 1
        Optional thinning applied along the ``draw`` dimension before computing
        summary statistics.
    qpairs : None | tuple | list of tuple, optional
        Percentile pair(s) in percent, e.g. ``(10, 90)`` or ``[(16, 84), (5, 95)]``.
        If provided, the function returns a dict with:
        ``median`` (ntheta,), ``qpairs`` (npairs,2), ``q_low`` (npairs, ntheta),
        ``q_high`` (npairs, ntheta). If not provided, only the median vector is returned
        (backwards compatible with the previous behaviour).

    Returns
    -------
    ndarray or dict
        Median theta vector (if ``qpairs`` is None/empty), otherwise a dict as described above.
    """
    import xarray as xr  # noqa: F401

    if "theta" not in idata.posterior:
        raise KeyError("InferenceData has no 'theta' in posterior.")

    da = idata.posterior["theta"]
    if thin is not None and int(thin) > 1:
        da = da.isel(draw=slice(None, None, int(thin)))

    # Stack chains/draws -> sample dimension
    stacked = da.stack(sample=("chain", "draw"))

    th = np.asarray(stacked.values)
    dims = tuple(getattr(stacked, "dims", ()))
    # Bring to shape (ntheta, nsample)
    if th.ndim == 2:
        if dims and dims[0] == "sample":
            th = th.T
        # else assume (theta_dim, sample)
    else:
        # Try to flatten everything except sample
        if "sample" in dims:
            sample_axis = dims.index("sample")
            th = np.moveaxis(th, sample_axis, -1)
        th = th.reshape((-1, th.shape[-1]))

    med = np.nanmedian(th, axis=1).astype(np.float64).reshape(-1)

    if qpairs is None or (isinstance(qpairs, (list, tuple)) and len(qpairs) == 0):
        return med

    # Normalize qpairs -> list[(qlo, qhi)]
    if isinstance(qpairs, tuple) and len(qpairs) == 2 and not isinstance(qpairs[0], (list, tuple)):
        qpairs_list = [qpairs]
    else:
        qpairs_list = list(qpairs)

    qpairs_norm = []
    q_low = []
    q_high = []
    for qlo, qhi in qpairs_list:
        qlo_f = float(qlo)
        qhi_f = float(qhi)
        if not (0.0 <= qlo_f <= 100.0 and 0.0 <= qhi_f <= 100.0):
            raise ValueError(f"Percentiles must be in [0,100], got {(qlo_f, qhi_f)}")
        if qlo_f >= qhi_f:
            raise ValueError(f"Percentile pair must satisfy qlo < qhi, got {(qlo_f, qhi_f)}")
        qpairs_norm.append((qlo_f, qhi_f))
        q_low.append(np.nanpercentile(th, qlo_f, axis=1))
        q_high.append(np.nanpercentile(th, qhi_f, axis=1))

    return {
        "median": med,
        "qpairs": np.asarray(qpairs_norm, dtype=np.float64),
        "q_low": np.asarray(q_low, dtype=np.float64),
        "q_high": np.asarray(q_high, dtype=np.float64),
    }



# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run PyMC inversion for all inputs defined in USER CONFIG."""

    outdir = _ensure_dir(OUTDIR)
    inputs = _glob_inputs(INPUT_GLOB)
    if not inputs:
        raise FileNotFoundError(f"No files match INPUT_GLOB: {INPUT_GLOB}")

# Starting model: either from NPZ, or from a direct model definition.
if MODEL_DIRECT is not None:
    model0 = model_from_direct(MODEL_DIRECT)
    saved = save_model_npz(model0, MODEL_DIRECT_SAVE_PATH, overwrite=bool(MODEL_DIRECT_OVERWRITE))
    print(f"Model (direct) saved to: {saved}")
else:
    model0 = load_model_npz(MODEL_NPZ)

    nl = int(model0.rop.shape[0])

    # Build parameter specification
    spec = mcmc.ParamSpec(
        nl=nl,
        fix_h=bool(FIX_H),
        sample_last_thickness=False,
    )

    print("\n=== PyMC anisotropic 1-D MT inversion ===")
    print(f"Inputs : {len(inputs)} file(s)")
    print(f"Outdir : {outdir}")
    print(f"Model  : {MODEL_NPZ} (nl={nl})")
    print(f"Use PT : {USE_PT}")
    print(f"Fix h  : {FIX_H}")
    print(f"Grad   : {ENABLE_GRAD}")
    print(f"Step   : {STEP_METHOD}")

    for f in inputs:
        print("\n----------------------------------------")
        print(f"Reading: {f}")
        site = load_site(f, prefer_spectra=True, err_kind="var")

        station = str(site.get("station", f.stem))
        print(f"Station: {station}")

        # Ensure PT if requested
        if USE_PT:
            ensure_phase_tensor(site, nsim=int(PT_ERR_NSIM), err_kind="var")

        # Build PyMC model
        pm_model, info = mcmc.build_pymc_model(
            site,
            spec=spec,
            h_m0=model0.h_m,
            rop0=model0.rop,
            ustr_deg0=model0.ustr_deg,
            udip_deg0=model0.udip_deg,
            usla_deg0=model0.usla_deg,
            is_iso=model0.is_iso,
            use_pt=bool(USE_PT),
            z_comps=Z_COMPS,
            pt_comps=PT_COMPS,
            compute_pt_if_missing=True,
            sigma_floor_Z=float(SIGMA_FLOOR_Z),
            sigma_floor_P=float(SIGMA_FLOOR_P),
            enable_grad=bool(ENABLE_GRAD),
            prior_kind=str(PRIOR_KIND),
        )

        # Sample
        idata = mcmc.sample_pymc(
            pm_model,
            draws=int(DRAWS),
            tune=int(TUNE),
            chains=int(CHAINS),
            cores=int(CORES),
            random_seed=int(RANDOM_SEED),
            progressbar=bool(PROGRESS),
            step_method=str(STEP_METHOD),
        )

        # Save InferenceData
        nc_path = outdir / f"{station}_pmc.nc"
        try:
            idata.to_netcdf(nc_path.as_posix())
            print(f"Saved: {nc_path}")
        except Exception as e:
            print(f"WARNING: could not write netcdf {nc_path}: {e}")

        # Summary NPZ: posterior median model + forward prediction
        theta_stats = posterior_theta_median(idata, thin=int(THIN), qpairs=THETA_QPAIRS)
        if isinstance(theta_stats, dict):
            theta_med = theta_stats["median"]
            theta_qpairs = theta_stats["qpairs"]
            theta_qlo = theta_stats["q_low"]
            theta_qhi = theta_stats["q_high"]
        else:
            theta_med = np.asarray(theta_stats, dtype=np.float64).reshape(-1)
            theta_qpairs = np.zeros((0, 2), dtype=np.float64)
            theta_qlo = np.zeros((0, theta_med.size), dtype=np.float64)
            theta_qhi = np.zeros((0, theta_med.size), dtype=np.float64)

        h_m_med, rop_med, ustr_med, udip_med, usla_med = mcmc.theta_to_model(
            theta_med,
            spec,
            h_m_fixed=model0.h_m,
            rop_fixed=model0.rop,
            ustr_fixed=model0.ustr_deg,
            udip_fixed=model0.udip_deg,
            usla_fixed=model0.usla_deg,
        )

        ctx = info["context"]
        pred = ctx.forward(h_m_med, rop_med, ustr_med, udip_med, usla_med, compute_sens=False)
        Z_pred = np.asarray(pred["Z"], dtype=np.complex128)

        summary_path = outdir / f"{station}_pmc_summary.npz"
        np.savez(
            summary_path.as_posix(),
            station=station,
            freq=np.asarray(site["freq"], dtype=np.float64),
            Z_obs=np.asarray(site["Z"], dtype=np.complex128),
            Z_err=np.asarray(site.get("Z_err")) if site.get("Z_err") is not None else None,
            P_obs=np.asarray(site.get("P")) if site.get("P") is not None else None,
            P_err=np.asarray(site.get("P_err")) if site.get("P_err") is not None else None,
            err_kind=str(site.get("err_kind", "var")),
            z_comps=np.array(Z_COMPS, dtype=object),
            pt_comps=np.array(PT_COMPS, dtype=object),
            theta_med=theta_med,
            theta_qpairs=theta_qpairs,
            theta_qlo=theta_qlo,
            theta_qhi=theta_qhi,
            param_names=np.array(info["param_names"], dtype=object),
            h_m_med=h_m_med,
            rop_med=rop_med,
            ustr_deg_med=ustr_med,
            udip_deg_med=udip_med,
            usla_deg_med=usla_med,
            Z_pred=Z_pred,
            is_iso=model0.is_iso,
        )
        print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
