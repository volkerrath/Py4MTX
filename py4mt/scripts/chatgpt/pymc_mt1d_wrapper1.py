"""
PyMC wrapper for 1-D anisotropic MT forward (mt1d_aniso) with NUTS gradients,
plus loaders for EDI, NPZ, and HDF5 data as used in the EDI project, and a
short CLI for quick inversions.

This module exposes:
- MT1DImpedanceOp: differentiable forward operator.
- build_pymc_model: convenience builder for a PyMC model.
- load_edi / load_npz / load_hdf5: data ingestion helpers that return unified dicts.
- CLI:
      python pymc_mt1d_wrapper.py --data site.edi --niter 1000 --plot
      python pymc_mt1d_wrapper.py --data site_data.npz --site SITE_A
      python pymc_mt1d_wrapper.py --data pack.h5 --site SITE_B --group /edi

Unified return format of loaders
--------------------------------
Each loader returns a dict with at least:
    {
      "periods": np.ndarray shape (nper,), float64  # seconds
      "Z":       np.ndarray shape (nper, 2, 2), complex128
      "sigma":   np.ndarray or float or None  # diagonal noise for stacked Re/Im
      "name":    str  # site or source name if available
    }

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-11-11
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, Optional, Tuple

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pytensor.graph.op import Op

# Optional h5 support without making it mandatory for EDI/NPZ users
try:
    import h5py  # type: ignore
    _H5_AVAILABLE = True
except Exception:
    _H5_AVAILABLE = False

# Import forward + helpers from the uploaded aniso.py
from aniso import mt1d_aniso, prep_aniso  # noqa: F401


# ==========================
# --- Helper functions -----
# ==========================
def _stack_z_components(Z: np.ndarray, comp_mask: np.ndarray) -> np.ndarray:
    """
    Stack selected components of a 2x2 impedance tensor into a 1D vector
    [Zxx, Zxy, Zyx, Zyy], then split into Re/Im and concatenate.

    Parameters
    ----------
    Z : ndarray, shape (2, 2), complex128
        Impedance tensor for a single period.
    comp_mask : ndarray, shape (4,), bool
        Mask selecting which components to keep in the [xx, xy, yx, yy] order.

    Returns
    -------
    y : ndarray, shape (2 * k,), float64
        Real-concatenated vector of selected components (k = comp_mask.sum()).
    """
    vec = np.array([Z[0, 0], Z[0, 1], Z[1, 0], Z[1, 1]], dtype=np.complex128)
    sel = vec[comp_mask]
    return np.concatenate([sel.real, sel.imag]).astype(np.float64)


def _stack_sensitivities(
    dZdal: np.ndarray,
    dZdat: np.ndarray,
    dZdbs: np.ndarray,
    dZdh: np.ndarray,
    comp_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert layerwise sensitivity tensors to stacked Re/Im vectors
    compatible with _stack_z_components.
    """
    nl = dZdal.shape[0]
    blocks = []
    for dZ in (dZdal, dZdat, dZdbs, dZdh):
        cols = []
        for il in range(nl):
            v = np.array(
                [dZ[il, 0, 0], dZ[il, 0, 1], dZ[il, 1, 0], dZ[il, 1, 1]],
                dtype=np.complex128,
            )
            v = v[comp_mask]
            cols.append(np.concatenate([v.real, v.imag]))
        blocks.append(np.column_stack(cols).astype(np.float64))
    return tuple(blocks)  # type: ignore[return-value]


# ==============================
# --- PyTensor custom Ops ------
# ==============================
class MT1DImpedanceOp(Op):
    """
    Forward Op computing impedances and analytic gradient.

    Inputs
    ------
    h, al, at, blt : float64 vectors (nl,)

    Output
    ------
    y : float64 vector (2 * k * nper,)
        Stacked real/imag of selected impedance components across periods.
    """

    itypes = [pt.dvector, pt.dvector, pt.dvector, pt.dvector]
    otypes = [pt.dvector]

    def __init__(self, layani: np.ndarray, periods: np.ndarray, comp_mask: np.ndarray):
        self.layani = np.asarray(layani, dtype=int)
        self.periods = np.asarray(periods, dtype=float)
        self.comp_mask = np.asarray(comp_mask, dtype=bool)
        self.nl = int(self.layani.size)
        self.k = int(self.comp_mask.sum())
        self.nper = int(self.periods.size)
        self.out_len = 2 * self.k * self.nper

    def make_node(self, h, al, at, blt):
        h = pt.as_tensor_variable(h).astype("float64")
        al = pt.as_tensor_variable(al).astype("float64")
        at = pt.as_tensor_variable(at).astype("float64")
        blt = pt.as_tensor_variable(blt).astype("float64")
        return pytensor.graph.Apply(self, [h, al, at, blt], [pt.vector(dtype="float64")])

    def perform(self, node, inputs, outputs):
        h, al, at, blt = [np.asarray(x, dtype=float) for x in inputs]
        y_list = []
        for p in self.periods:
            Z, *_ = mt1d_aniso(self.layani, h, al, at, blt, p)
            y_list.append(_stack_z_components(Z, self.comp_mask))
        outputs[0][0] = np.concatenate(y_list).astype(np.float64)

    def grad(self, inputs, output_grads):
        (gz,) = output_grads  # shape (2*k*nper,)
        return MT1DImpedanceVJP(self.layani, self.periods, self.comp_mask)(*inputs, gz)


class MT1DImpedanceVJP(Op):
    """Vector–Jacobian product for analytic sensitivities from mt1d_aniso."""

    itypes = [pt.dvector, pt.dvector, pt.dvector, pt.dvector, pt.dvector]
    otypes = [pt.dvector, pt.dvector, pt.dvector, pt.dvector]

    def __init__(self, layani: np.ndarray, periods: np.ndarray, comp_mask: np.ndarray):
        self.layani = np.asarray(layani, dtype=int)
        self.periods = np.asarray(periods, dtype=float)
        self.comp_mask = np.asarray(comp_mask, dtype=bool)
        self.nl = int(self.layani.size)
        self.k = int(self.comp_mask.sum())
        self.nper = int(self.periods.size)

    def perform(self, node, inputs, outputs):
        h, al, at, blt, gz = [np.asarray(x, dtype=float) for x in inputs]

        gh = np.zeros(self.nl)
        gal = np.zeros(self.nl)
        gat = np.zeros(self.nl)
        gblt = np.zeros(self.nl)

        offset = 0
        for p in self.periods:
            Z, dZdal, dZdat, dZdbs, dZdh = mt1d_aniso(self.layani, h, al, at, blt, p)
            _ = Z  # unused
            J_al, J_at, J_bs, J_h = _stack_sensitivities(dZdal, dZdat, dZdbs, dZdh, self.comp_mask)
            gzp = gz[offset : offset + 2 * self.k]
            offset += 2 * self.k
            gal += J_al.T @ gzp
            gat += J_at.T @ gzp
            gblt += J_bs.T @ gzp
            gh += J_h.T @ gzp

        outputs[0][0] = gh
        outputs[1][0] = gal
        outputs[2][0] = gat
        outputs[3][0] = gblt


# ==============================
# --- Data loaders -------------
# ==============================
_FLOAT_RE = r"[+-]?(?:(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][+-]?\d+)?)"


def _parse_edi_blocks(text: str) -> Dict[str, np.ndarray]:
    """
    Minimal, tolerant EDI parser for Phoenix/EGI-style sections used in the project.
    """
    txt = text.upper()
    freq_match = re.findall(r">FREQ[^\n]*\n((?:\s*" + _FLOAT_RE + r"\s*)+)", txt)
    if not freq_match:
        freq_match = re.findall(r">FREQUENC(?:Y|IES)[^\n]*\n((?:\s*" + _FLOAT_RE + r"\s*)+)", txt)
    if not freq_match:
        raise ValueError("EDI: could not find FREQ block")
    freq_vals = re.findall(_FLOAT_RE, " ".join(freq_match))
    freq = np.array([float(v.replace("D", "E")) for v in freq_vals], dtype=float)
    periods = 1.0 / freq

    comps = {}
    for comp in ("XX", "XY", "YX", "YY"):
        r_match = re.findall(rf">Z{comp}R[^\n]*\n((?:\s*{_FLOAT_RE}\s*)+)", txt)
        i_match = re.findall(rf">Z{comp}I[^\n]*\n((?:\s*{_FLOAT_RE}\s*)+)", txt)
        if r_match and i_match:
            r_vals = [float(v.replace("D", "E")) for v in re.findall(_FLOAT_RE, " ".join(r_match))]
            i_vals = [float(v.replace("D", "E")) for v in re.findall(_FLOAT_RE, " ".join(i_match))]
            comps[comp] = np.array(r_vals, dtype=float) + 1j * np.array(i_vals, dtype=float)

    if not all(c in comps for c in ("XX", "XY", "YX", "YY")):
        for comp in ("XX", "XY", "YX", "YY"):
            both = re.findall(
                rf">Z{comp}\b[^\n]*\n((?:\s*{_FLOAT_RE}\s+{_FLOAT_RE}\s*)+)",
                txt,
            )
            if both and comp not in comps:
                nums = [float(v.replace("D", "E")) for v in re.findall(_FLOAT_RE, " ".join(both))]
                arr = np.array(nums, dtype=float).reshape(-1, 2)
                comps[comp] = arr[:, 0] + 1j * arr[:, 1]

    nper = periods.size
    for comp in ("XX", "XY", "YX", "YY"):
        if comp not in comps:
            comps[comp] = np.full(nper, np.nan + 1j * np.nan, dtype=np.complex128)
        else:
            if comps[comp].size != nper:
                raise ValueError(f"EDI: component Z{comp} length mismatch with FREQ")

    sigma = None
    for comp in ("XX", "XY", "YX", "YY"):
        var = re.findall(rf">Z{comp}\s*VAR[^\n]*\n((?:\s*{_FLOAT_RE}\s*)+)", txt)
        err = re.findall(rf">Z{comp}\s*ERR[^\n]*\n((?:\s*{_FLOAT_RE}\s*)+)", txt)
        src = var or err
        if src:
            vals = np.array([float(v.replace("D", "E")) for v in re.findall(_FLOAT_RE, " ".join(src))], dtype=float)
            if vals.size == nper:
                if sigma is None:
                    sigma = np.zeros((nper, 8), dtype=float)
                idx = {"XX": 0, "XY": 1, "YX": 2, "YY": 3}[comp]
                sigma[:, 2 * idx] = np.sqrt(np.abs(vals))
                sigma[:, 2 * idx + 1] = np.sqrt(np.abs(vals))

    return {
        "periods": periods.astype(float),
        "Zxx": comps["XX"],
        "Zxy": comps["XY"],
        "Zyx": comps["YX"],
        "Zyy": comps["YY"],
        "sigma": sigma,
    }


def load_edi(path: str, site_name: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Load a single-site EDI text file into unified dict."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    blk = _parse_edi_blocks(text)
    nper = blk["periods"].size
    Z = np.empty((nper, 2, 2), dtype=np.complex128)
    Z[:, 0, 0] = blk["Zxx"]
    Z[:, 0, 1] = blk["Zxy"]
    Z[:, 1, 0] = blk["Zyx"]
    Z[:, 1, 1] = blk["Zyy"]
    sigma = blk.get("sigma")
    return {
        "periods": blk["periods"],
        "Z": Z,
        "sigma": sigma,
        "name": site_name or os.path.splitext(os.path.basename(path))[0],
    }


def load_npz(path: str, site: Optional[str] = None) -> Dict[str, np.ndarray]:
    """
    Load NPZ exported by the EDI project scripts with flexible key heuristics.
    """
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())

    def pick(*names):
        for n in names:
            if n in data:
                return data[n]
        return None

    prefix = ""
    if site is not None:
        for sep in (":", "_"):
            candidate = f"{site}{sep}periods"
            if candidate in data:
                prefix = f"{site}{sep}"
                break

    per = pick(f"{prefix}periods", f"{prefix}per")
    if per is None:
        freq = pick(f"{prefix}freq", f"{prefix}frequency", "freq", "frequency")
        if freq is None:
            raise ValueError("NPZ: no 'periods'/'per' or 'freq' array found")
        per = 1.0 / np.array(freq, dtype=float)
    else:
        per = np.array(per, dtype=float)

    Z = pick(f"{prefix}Z", "Z")
    if Z is None:
        comps = {}
        for comp in ("xx", "xy", "yx", "yy"):
            c = pick(f"{prefix}Z{comp}", f"Z{comp}")
            if c is None:
                r = pick(f"{prefix}Z{comp}r", f"Z{comp}r")
                i = pick(f"{prefix}Z{comp}i", f"Z{comp}i")
                if r is not None and i is not None:
                    c = np.asarray(r) + 1j * np.asarray(i)
            if c is not None:
                comps[comp] = np.asarray(c)
        if not all(k in comps for k in ("xx", "xy", "yx", "yy")):
            raise ValueError("NPZ: could not assemble Z from keys")
        nper = per.size
        Z = np.empty((nper, 2, 2), dtype=np.complex128)
        Z[:, 0, 0] = comps["xx"]
        Z[:, 0, 1] = comps["xy"]
        Z[:, 1, 0] = comps["yx"]
        Z[:, 1, 1] = comps["yy"]
    else:
        Z = np.asarray(Z).astype(np.complex128)

    sigma = pick(f"{prefix}sigma", "sigma")
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float)

    return {
        "periods": per,
        "Z": Z,
        "sigma": sigma,
        "name": site or os.path.splitext(os.path.basename(path))[0],
    }


def load_hdf5(path: str, site: Optional[str] = None, group: str = "/") -> Dict[str, np.ndarray]:
    """
    Load HDF5 exported by the EDI project scripts (heuristic discovery of keys).
    """
    if not _H5_AVAILABLE:
        raise ImportError("h5py is not available; install h5py to read HDF5.")

    import h5py  # type: ignore  # local import for safety

    with h5py.File(path, "r") as f:
        g = f[group] if group in f else f["/"]
        # Determine site
        if site is None:
            subgroups = [k for k, v in g.items() if isinstance(v, h5py.Group)]
            if len(subgroups) == 1:
                site = subgroups[0]
            elif len(subgroups) == 0:
                site = None
            else:
                raise ValueError(f"HDF5: multiple sites found at {group}, specify --site from {subgroups}")

        h = g[site] if site is not None else g

        def pick(*names):
            for n in names:
                if n in h:
                    return h[n][()]
            return None

        per = pick("periods", "per")
        if per is None:
            freq = pick("freq", "frequency")
            if freq is None:
                raise ValueError("HDF5: no 'periods'/'per' or 'freq' in group")
            per = 1.0 / np.array(freq, dtype=float)
        else:
            per = np.array(per, dtype=float)

        if "Z" in h:
            Z = h["Z"][()].astype(np.complex128)
        else:
            comps = {}
            for comp in ("xx", "xy", "yx", "yy"):
                key = f"Z{comp}"
                if key in h:
                    comps[comp] = h[key][()]
            if not all(k in comps for k in ("xx", "xy", "yx", "yy")):
                raise ValueError("HDF5: cannot assemble Z")
            nper = per.size
            Z = np.empty((nper, 2, 2), dtype=np.complex128)
            Z[:, 0, 0] = comps["xx"]
            Z[:, 0, 1] = comps["xy"]
            Z[:, 1, 0] = comps["yx"]
            Z[:, 1, 1] = comps["yy"]

        sigma = h["sigma"][()] if "sigma" in h else None

    return {
        "periods": per,
        "Z": Z,
        "sigma": sigma,
        "name": site or os.path.splitext(os.path.basename(path))[0],
    }


# ==============================
# --- Model builder ------------
# ==============================
def build_pymc_model(
    periods: np.ndarray,
    Z_obs: np.ndarray,
    sigma: float | np.ndarray,
    layani: np.ndarray,
    h_init: np.ndarray,
    al_init: np.ndarray,
    at_init: np.ndarray,
    blt_init: np.ndarray,
    comp_mask: np.ndarray | None = None,
    priors: Optional[dict] = None,
) -> pm.Model:
    """Construct a PyMC model that fits complex impedances with diagonal Gaussian noise."""
    periods = np.asarray(periods, dtype=float)
    Z_obs = np.asarray(Z_obs, dtype=np.complex128)
    layani = np.asarray(layani, dtype=int)

    nper = periods.size
    if comp_mask is None:
        comp_mask = np.array([False, True, True, False], dtype=bool)
    else:
        comp_mask = np.asarray(comp_mask, dtype=bool)
        assert comp_mask.shape == (4,), "comp_mask must be length-4"

    k = int(comp_mask.sum())

    yobs_list = [_stack_z_components(Z_obs[i], comp_mask) for i in range(nper)]
    y_obs = np.concatenate(yobs_list).astype(np.float64)

    def _shape_sigma(sig):
        sig = np.asarray(sig, dtype=float)
        if sig.ndim == 0:
            sig = np.full((nper, 2 * k), float(sig))
        elif sig.ndim == 1:
            assert sig.size == 2 * k, "If 1D, sigma must be length 2*k"
            sig = np.tile(sig[None, :], (nper, 1))
        elif sig.ndim == 2:
            assert sig.shape == (nper, 2 * k), "2D sigma must be (nper, 2*k)"
        else:
            raise ValueError("sigma must be scalar, (2*k,), or (nper, 2*k)")
        return sig

    sigma2d = _shape_sigma(sigma)
    pr = dict(log_al_sd=2.0, log_at_sd=2.0, d_blt_sd=np.pi / 4, log_h_sd=1.5, eps_sigma=0.0)

    h0 = np.asarray(h_init, dtype=float).copy()
    h0[h0 <= 0] = 1e-6
    al0 = np.asarray(al_init, dtype=float)
    at0 = np.asarray(at_init, dtype=float)
    blt0 = np.asarray(blt_init, dtype=float)

    op = MT1DImpedanceOp(layani=layani, periods=periods, comp_mask=comp_mask)

    with pm.Model() as model:
        log_al = pm.Normal("log_al", mu=np.log(al0), sigma=pr["log_al_sd"], shape=layani.size)
        log_at = pm.Normal("log_at", mu=np.log(at0), sigma=pr["log_at_sd"], shape=layani.size)
        log_h = pm.Normal("log_h", mu=np.log(h0), sigma=pr["log_h_sd"], shape=layani.size)
        d_blt = pm.Normal("d_blt", mu=0.0, sigma=pr["d_blt_sd"], shape=layani.size)

        al = pm.Deterministic("al", pt.exp(log_al))
        at = pm.Deterministic("at", pt.exp(log_at))
        h = pm.Deterministic("h", pt.exp(log_h))
        blt = pm.Deterministic("blt", blt0 + d_blt)

        y_pred = op(h, al, at, blt)

        sig_flat = pt.as_tensor_variable(sigma2d.reshape(-1))
        eps = float(pr["eps_sigma"])
        sig_flat = pt.sqrt(pt.square(sig_flat) + eps * eps)

        pm.Normal("y", mu=y_pred, sigma=sig_flat, observed=y_obs)

    return model


# ==============================
# --- Simple CLI ---------------
# ==============================
def _infer_loader(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".edi", ".z3d", ".zxy"):
        return "edi"
    if ext == ".npz":
        return "npz"
    if ext in (".h5", ".hdf5"):
        return "hdf5"
    raise ValueError(f"Unrecognized data extension: {ext}")


def main():
    """CLI for quick test sampling with EDI/NPZ/HDF5 inputs."""
    parser = argparse.ArgumentParser(description="Run PyMC NUTS inversion on anisotropic MT model.")
    parser.add_argument("--data", type=str, required=True, help="Path to EDI / NPZ / HDF5 dataset.")
    parser.add_argument("--site", type=str, default=None, help="Site name (for NPZ/HDF5 multi-site packs).")
    parser.add_argument("--group", type=str, default="/", help="HDF5 group prefix containing sites (default '/').")
    parser.add_argument("--sigma", type=float, default=None, help="Override absolute sigma if dataset has none.")
    parser.add_argument("--mask", type=str, default="offdiag", choices=["offdiag", "all"],
                        help="Impedance components to use: offdiag(Zxy,Zyx) or all four.")
    parser.add_argument("--niter", type=int, default=500, help="Number of posterior draws (tune=niter//2).")
    parser.add_argument("--plot", action="store_true", help="Plot posterior for al/at.")
    args = parser.parse_args()

    which = _infer_loader(args.data)
    if which == "edi":
        loaded = load_edi(args.data, site_name=args.site)
    elif which == "npz":
        loaded = load_npz(args.data, site=args.site)
    else:
        loaded = load_hdf5(args.data, site=args.site, group=args.group)

    periods = loaded["periods"]
    Z_obs = loaded["Z"]
    sigma_loaded = loaded.get("sigma")
    name = loaded.get("name", "site")

    comp_mask = np.array([False, True, True, False]) if args.mask == "offdiag" else np.array([True, True, True, True])

    if sigma_loaded is not None and isinstance(sigma_loaded, np.ndarray):
        k_full = 4
        if sigma_loaded.ndim == 2 and sigma_loaded.shape[1] in (8,):
            sel_idx = np.where(np.array([True, True, True, True]) & comp_mask)[0]
            cols = []
            for idx in sel_idx:
                cols.extend([2 * idx, 2 * idx + 1])
            sigma = sigma_loaded[:, cols]
        else:
            sigma = sigma_loaded
    else:
        sigma = args.sigma if args.sigma is not None else 0.05

    # Default 3-layer starting model (can be edited on the CLI later if needed)
    nl = 3
    layani = np.zeros(nl, dtype=int)
    h_init = np.array([0.5, 1.0, 0.0])
    al_init = np.full(nl, 0.02)
    at_init = np.full(nl, 0.05)
    blt_init = np.zeros(nl)

    model = build_pymc_model(
        periods=periods,
        Z_obs=Z_obs,
        sigma=sigma,
        layani=layani,
        h_init=h_init,
        al_init=al_init,
        at_init=at_init,
        blt_init=blt_init,
        comp_mask=comp_mask,
    )

    print(f"Sampling site '{name}' with {args.niter} draws (tune={max(200, args.niter // 2)}) ...")
    with model:
        idata = pm.sample(draws=args.niter, tune=max(200, args.niter // 2), target_accept=0.9, progressbar=True)

    if args.plot:
        try:
            import matplotlib.pyplot as plt  # type: ignore
            pm.plot_trace(idata, var_names=["al", "at"])
            plt.suptitle(f"Posterior — {name}")
            plt.show()
        except Exception as e:
            print(f"Plotting failed: {e}")


if __name__ == "__main__":
    main()
