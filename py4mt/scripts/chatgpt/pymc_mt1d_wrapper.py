"""
PyMC wrapper for 1-D anisotropic MT forward (mt1d_aniso) with NUTS gradients.

This module exposes:
- MT1DImpedanceOp: differentiable forward operator.
- build_pymc_model: convenience builder for a PyMC model.
- Command-line interface for quick tests:
      python pymc_mt1d_wrapper.py --edi data.edi [--niter 1000] [--plot]

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-11-10
"""

from __future__ import annotations

import argparse
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pytensor.graph.op import Op

# Import your forward + helpers from the uploaded aniso.py
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

    Parameters
    ----------
    dZdal, dZdat, dZdbs, dZdh : ndarray, shape (nl, 2, 2), complex128
        Sensitivity of Z to the respective parameter for a single period.
    comp_mask : ndarray, shape (4,), bool
        Component selection mask [xx, xy, yx, yy].

    Returns
    -------
    (J_al, J_at, J_bs, J_h) : tuple of ndarrays, each shape (2*k, nl), float64.
        Jacobian blocks relating stacked y to increments in each parameter,
        where k = comp_mask.sum().
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
    """
    Vector–Jacobian product for analytic sensitivities from mt1d_aniso.

    Inputs
    ------
    h, al, at, blt : float64 vectors (nl,)
    gz : float64 vector (2*k*nper,)  -- upstream gradient

    Outputs
    -------
    (gh, gal, gat, gblt) : 4 float64 vectors (nl,)
        Gradient contributions with respect to h, al, at, blt.
    """

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
    """
    Construct a PyMC model that fits complex impedances with diagonal Gaussian noise.

    Parameters
    ----------
    periods : ndarray, shape (nper,), float
        Periods [s].
    Z_obs : ndarray, shape (nper, 2, 2), complex128
        Observed impedances.
    sigma : float or ndarray
        Standard deviation(s) of the stacked real/imag vector y. Accepts:
          - scalar: applied to all entries;
          - shape (2*k,): per-component (k = number of selected components);
          - shape (nper, 2*k): per-period per-component.
    layani : ndarray, shape (nl,), int
        Layer anisotropy flags; kept fixed.
    h_init, al_init, at_init, blt_init : ndarray, shape (nl,)
        Initial values; also used to center priors.
    comp_mask : ndarray, shape (4,), bool, optional
        Which components to use, ordered [Zxx, Zxy, Zyx, Zyy].
        Default: only off-diagonals.
    priors : dict, optional
        Override default prior hyperparameters. Keys:
          - "log_al_sd", "log_at_sd" (float, default 2.0)
          - "d_blt_sd" (float rad, default np.pi/4)
          - "log_h_sd" (float, default 1.5)
          - "eps_sigma" (float, jitter added to sigma to avoid 0, default 0)

    Returns
    -------
    model : pm.Model
        A compiled PyMC model ready for sampling with NUTS.
    """
    periods = np.asarray(periods, dtype=float)
    Z_obs = np.asarray(Z_obs, dtype=np.complex128)
    layani = np.asarray(layani, dtype=int)

    nper = periods.size
    nl = layani.size

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
    if priors:
        pr.update(priors)

    h0 = np.asarray(h_init, dtype=float).copy()
    h0[h0 <= 0] = 1e-6
    al0 = np.asarray(al_init, dtype=float)
    at0 = np.asarray(at_init, dtype=float)
    blt0 = np.asarray(blt_init, dtype=float)

    op = MT1DImpedanceOp(layani=layani, periods=periods, comp_mask=comp_mask)

    with pm.Model() as model:
        log_al = pm.Normal("log_al", mu=np.log(al0), sigma=pr["log_al_sd"], shape=nl)
        log_at = pm.Normal("log_at", mu=np.log(at0), sigma=pr["log_at_sd"], shape=nl)
        log_h = pm.Normal("log_h", mu=np.log(h0), sigma=pr["log_h_sd"], shape=nl)
        d_blt = pm.Normal("d_blt", mu=0.0, sigma=pr["d_blt_sd"], shape=nl)

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
def main():
    """
    CLI for quick test sampling.

    Example
    -------
    python pymc_mt1d_wrapper.py --edi myfile.edi --niter 500 --plot
    """
    parser = argparse.ArgumentParser(description="Run PyMC NUTS inversion on anisotropic MT model.")
    parser.add_argument("--edi", type=str, help="EDI file (placeholder for future direct ingestion).")
    parser.add_argument("--niter", type=int, default=500, help="Number of posterior draws (tuning = niter//2).")
    parser.add_argument("--plot", action="store_true", help="Plot posterior for al/at.")
    args = parser.parse_args()

    print("Running demo with synthetic model (CLI stub; EDI parsing to be added).")

    # Simple synthetic dataset for a smoke test
    layani = np.array([0, 0, 0])
    h = np.array([0.5, 1.0, 0.0])
    al = np.array([0.01, 0.02, 0.05])
    at = np.array([0.02, 0.05, 0.1])
    blt = np.zeros(3)
    periods = np.logspace(-1, 2, 12)
    comp_mask = np.array([False, True, True, False])

    Z_obs = np.empty((len(periods), 2, 2), dtype=complex)
    for i, p in enumerate(periods):
        Z, *_ = mt1d_aniso(layani, h, al, at, blt, p)
        Z_obs[i] = Z

    model = build_pymc_model(
        periods=periods,
        Z_obs=Z_obs,
        sigma=0.05,
        layani=layani,
        h_init=h,
        al_init=al,
        at_init=at,
        blt_init=blt,
        comp_mask=comp_mask,
    )

    with model:
        idata = pm.sample(draws=args.niter, tune=max(200, args.niter // 2), target_accept=0.9, progressbar=True)

    if args.plot:
        pm.plot_trace(idata, var_names=["al", "at"])
        plt.show()


if __name__ == "__main__":
    main()
