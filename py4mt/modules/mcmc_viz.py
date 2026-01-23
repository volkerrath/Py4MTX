#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mcmc_viz.py

Axis-based plotting utilities for anisotropic 1-D MT Bayesian inversion results
produced by ``mt_aniso1d_sampler.py`` (PyMC backend).

Important design choice
-----------------------
This module does **not** create figures or subplot layouts.

- The calling script creates figures/axes with ``plt.subplots(...)`` (or any
  other layout method).
- The functions here *only* draw onto the axes you pass in and then return
  those axes.

This keeps plotting code easy to read and easy to reuse (also for students that
are new to Python/matplotlib).

Expected per-site outputs from the sampler
------------------------------------------
1) ``<station>_pmc.nc``          : ArviZ InferenceData (posterior samples).
2) ``<station>_pmc_summary.npz`` : NPZ with observed data + posterior summaries
                                  + predictive quantiles.

Typical usage (sketch)
----------------------
>>> import matplotlib.pyplot as plt
>>> import mcmc_viz as mv
>>> s = mv.load_summary_npz("SITE_pmc_summary.npz")
>>> idata = mv.open_idata("SITE_pmc.nc")
>>> fig, axs = plt.subplots(3, 1, figsize=(6, 10))
>>> mv.plot_theta_trace(axs[0], idata, idx=0, name="theta[0]")
>>> mv.plot_theta_density(axs[1], idata, idx=0, qpairs=s.get("theta_qpairs"))
>>> mv.plot_vertical_resistivity(axs[2], s, comp=0, use_log10=True)
>>> fig.tight_layout()
>>> fig.show()

Author: Volker Rath (DIAS)
Created with the help of ChatGPT (GPT-5 Thinking) on 2026-01-23
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


# Magnetic permeability of vacuum [H/m]
MU0 = 4e-7 * np.pi


# -----------------------------------------------------------------------------
# Loading helpers
# -----------------------------------------------------------------------------


def _unwrap_npz_scalar(value: Any) -> Any:
    """
    Unwrap a NumPy "0-d object array" to a Python object.

    NumPy stores some Python objects (e.g., lists, dicts, None) as object arrays
    when writing NPZ files. When loaded, they often appear as:
        array(obj, dtype=object) with shape ()

    Parameters
    ----------
    value : Any
        Value returned from ``np.load(..., allow_pickle=True)``.

    Returns
    -------
    Any
        A "plain" Python object if ``value`` was a 0-d object array,
        otherwise ``value`` unchanged.
    """
    if isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
        return value.item()
    return value


def load_summary_npz(path: str | Path) -> Dict[str, Any]:
    """
    Load a ``*_pmc_summary.npz`` file into a plain Python dictionary.

    This function:
    - reads the NPZ with ``allow_pickle=True`` (required for Python objects)
    - unwraps 0-d object arrays back to normal Python values

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the summary NPZ file, typically ``<station>_pmc_summary.npz``.

    Returns
    -------
    dict
        Dictionary mapping NPZ keys to values (arrays or Python objects).
    """
    p = Path(path)
    with np.load(p.as_posix(), allow_pickle=True) as npz:
        out = {k: _unwrap_npz_scalar(npz[k]) for k in npz.files}
    return out


def open_idata(netcdf_path: str | Path):
    """
    Open ArviZ InferenceData from the sampler's ``*_pmc.nc`` output.

    Parameters
    ----------
    netcdf_path : str or pathlib.Path
        Path to the NetCDF file created by ArviZ.

    Returns
    -------
    arviz.InferenceData
        Loaded inference data.

    Raises
    ------
    ImportError
        If ArviZ is not installed.
    """
    try:
        import arviz as az  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "ArviZ is required to open InferenceData (*.nc). "
            "Install with: pip install arviz"
        ) from e
    return az.from_netcdf(Path(netcdf_path).as_posix())


def theta_draws(idata, var: str = "theta") -> np.ndarray:
    """
    Extract posterior draws for a vector parameter (default: ``theta``).

    The sampler stores theta as a vector variable with dimensions:
        (chain, draw, n_theta)

    This function stacks chain and draw into one dimension so that the result is:
        (n_samples_total, n_theta)

    Parameters
    ----------
    idata : arviz.InferenceData
        InferenceData containing ``idata.posterior[var]``.
    var : str
        Posterior variable name (default: ``"theta"``).

    Returns
    -------
    ndarray
        Array of shape (n_samples_total, n_theta).

    Raises
    ------
    ValueError
        If the posterior variable cannot be reshaped to a vector.
    """
    post = np.asarray(idata.posterior[var])
    if post.ndim < 3:
        raise ValueError(
            f"Posterior variable '{var}' must be at least 3-D (chain,draw,theta). "
            f"Got shape {post.shape}."
        )
    n_chain = post.shape[0]
    n_draw = post.shape[1]
    n_theta = int(np.prod(post.shape[2:]))
    return post.reshape(n_chain * n_draw, n_theta)


def normalize_qpairs(qpairs: Optional[Sequence[Sequence[float]]]) -> np.ndarray:
    """
    Normalize percentile/quantile pairs to a numeric array.

    The module uses *percentiles* (0..100), e.g.:
        [(10, 90), (20, 80)]

    Parameters
    ----------
    qpairs : sequence of pairs or None
        Each entry must be a length-2 sequence (lo, hi) in percent.

    Returns
    -------
    ndarray
        Array of shape (npairs, 2). Empty array if qpairs is None/empty.

    Raises
    ------
    ValueError
        If percentiles are outside [0, 100].
    """
    if qpairs is None:
        return np.zeros((0, 2), dtype=np.float64)
    q = np.asarray(qpairs, dtype=np.float64).reshape(-1, 2)
    if np.any(q < 0.0) or np.any(q > 100.0):
        raise ValueError("qpairs must be in percent (0..100).")
    return np.sort(q, axis=1)


def get_array(summary: Mapping[str, Any], key: str, dtype=None) -> np.ndarray:
    """
    Convenience getter for arrays from the summary dict.

    Parameters
    ----------
    summary : Mapping
        Summary dict returned by ``load_summary_npz``.
    key : str
        Key to fetch.
    dtype : dtype or None
        If provided, cast the array to this dtype.

    Returns
    -------
    ndarray
        The value as an ndarray, or an empty array if missing / None.
    """
    if key not in summary or summary[key] is None:
        return np.asarray([], dtype=dtype)
    return np.asarray(summary[key], dtype=dtype)


# -----------------------------------------------------------------------------
# MT derived quantities
# -----------------------------------------------------------------------------


def period_from_freq(freq_hz: np.ndarray) -> np.ndarray:
    """
    Convert frequency (Hz) to period (s).

    Parameters
    ----------
    freq_hz : ndarray
        Frequencies in Hz.

    Returns
    -------
    ndarray
        Periods in seconds.
    """
    f = np.asarray(freq_hz, dtype=np.float64)
    return 1.0 / f


def z_to_rho_phase(freq_hz: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert impedance Z(f) to apparent resistivity and phase.

    Parameters
    ----------
    freq_hz : ndarray, shape (nper,)
        Frequencies in Hz.
    Z : ndarray, shape (nper, 2, 2), complex
        Complex impedance tensor.

    Returns
    -------
    rho : ndarray, shape (nper, 2, 2)
        Apparent resistivity (Ohm·m).
    phase_deg : ndarray, shape (nper, 2, 2)
        Phase in degrees.
    """
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    omega = 2.0 * np.pi * freq_hz[:, None, None]
    rho = (np.abs(Z) ** 2) / (MU0 * omega)
    phase_deg = np.degrees(np.arctan2(np.imag(Z), np.real(Z)))
    return rho, phase_deg


# -----------------------------------------------------------------------------
# Layered step plotting helpers
# -----------------------------------------------------------------------------


def step_arrays(z_bot: np.ndarray, v_layer: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build step-plot arrays for a layered model quantity vs depth.

    Parameters
    ----------
    z_bot : ndarray, shape (nl,)
        Layer bottom depths (m), increasing.
    v_layer : ndarray, shape (nl,)
        Layer values.

    Returns
    -------
    z_step : ndarray, shape (2*nl,)
        Depth values for step plotting.
    v_step : ndarray, shape (2*nl,)
        Repeated layer values for step plotting.
    """
    z_bot = np.asarray(z_bot, dtype=np.float64).ravel()
    v_layer = np.asarray(v_layer, dtype=np.float64).ravel()
    if z_bot.size != v_layer.size:
        raise ValueError("z_bot and v_layer must have the same length.")

    nl = z_bot.size
    z_top = np.r_[0.0, z_bot[:-1]]

    z_step = np.empty(2 * nl, dtype=np.float64)
    v_step = np.empty(2 * nl, dtype=np.float64)
    z_step[0::2] = z_top
    z_step[1::2] = z_bot
    v_step[0::2] = v_layer
    v_step[1::2] = v_layer
    return z_step, v_step


def plot_layer_steps(
    ax: plt.Axes,
    z_bot: np.ndarray,
    v_med: np.ndarray,
    v_qlo: Optional[np.ndarray] = None,
    v_qhi: Optional[np.ndarray] = None,
    qpairs: Optional[Sequence[Sequence[float]]] = None,
    alpha: float = 0.12,
    label_intervals: bool = False,
    label_median: str = "median",
    invert_y: bool = True,
) -> plt.Axes:
    """
    Plot a layered step function (median) and optional credible envelopes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on.
    z_bot : ndarray, shape (nl,)
        Layer bottom depths.
    v_med : ndarray, shape (nl,)
        Median layer values.
    v_qlo, v_qhi : ndarray or None
        If given, arrays of shape (npairs, nl) containing lower and upper
        credible bounds for each percentile pair.
    qpairs : sequence of pairs or None
        Percentile pairs in percent, e.g. [(10,90),(20,80)].
        Only used for labeling if ``label_intervals=True``.
    alpha : float
        Transparency for filled envelopes.
    label_intervals : bool
        If True, fill envelopes will be given labels "lo-hi%".
    label_median : str
        Label for the median line.
    invert_y : bool
        If True, invert y-axis (depth increases downward).

    Returns
    -------
    ax : matplotlib.axes.Axes
        The same axis, for chaining.
    """
    z_bot = np.asarray(z_bot, dtype=np.float64).ravel()
    v_med = np.asarray(v_med, dtype=np.float64).ravel()

    if v_qlo is not None and v_qhi is not None:
        qlo = np.asarray(v_qlo, dtype=np.float64)
        qhi = np.asarray(v_qhi, dtype=np.float64)
        if qlo.ndim != 2 or qhi.ndim != 2:
            raise ValueError("v_qlo and v_qhi must have shape (npairs, nl).")
        if qlo.shape != qhi.shape:
            raise ValueError("v_qlo and v_qhi must have the same shape.")
        if qlo.shape[1] != z_bot.size:
            raise ValueError("Second dimension of v_qlo/v_qhi must match len(z_bot).")

        q = normalize_qpairs(qpairs) if qpairs is not None else np.zeros((qlo.shape[0], 2))
        for k in range(qlo.shape[0]):
            z_s, lo_s = step_arrays(z_bot, qlo[k, :])
            _, hi_s = step_arrays(z_bot, qhi[k, :])
            lab = None
            if label_intervals and q.size and k < q.shape[0]:
                lab = f"{q[k,0]:.0f}-{q[k,1]:.0f}%"
            ax.fill_betweenx(z_s, lo_s, hi_s, alpha=alpha, label=lab)

    z_s, v_s = step_arrays(z_bot, v_med)
    ax.plot(v_s, z_s, lw=1.6, label=label_median)
    ax.grid(True, alpha=0.2)
    if invert_y:
        ax.invert_yaxis()
    return ax


# -----------------------------------------------------------------------------
# Component index helpers
# -----------------------------------------------------------------------------


def comp_to_ij(comp: str) -> Tuple[int, int]:
    """
    Map component string to impedance/phase-tensor indices.

    Parameters
    ----------
    comp : str
        One of: 'xx', 'xy', 'yx', 'yy' (case-insensitive).

    Returns
    -------
    (i, j) : tuple of int
        Indices into a (2,2) tensor.
    """
    c = comp.strip().lower()
    mapping = {"xx": (0, 0), "xy": (0, 1), "yx": (1, 0), "yy": (1, 1)}
    if c not in mapping:
        raise ValueError(f"Unknown component '{comp}'. Use one of {list(mapping)}.")
    return mapping[c]


# -----------------------------------------------------------------------------
# Theta-space plots (one parameter per axis)
# -----------------------------------------------------------------------------


def plot_theta_trace(
    ax: plt.Axes,
    idata,
    idx: int,
    name: Optional[str] = None,
    thin: int = 1,
) -> plt.Axes:
    """
    Plot trace lines (all chains) for one theta component onto an axis.
    """
    post = np.asarray(idata.posterior["theta"])
    if post.ndim != 3:
        post = post.reshape(post.shape[0], post.shape[1], -1)

    n_chain, n_draw, n_theta = post.shape
    if idx < 0 or idx >= n_theta:
        raise IndexError(f"idx={idx} out of range (theta has size {n_theta}).")

    x = np.arange(0, n_draw, thin)
    for c in range(n_chain):
        ax.plot(x, post[c, ::thin, idx], lw=0.7, alpha=0.85)

    lab = name or f"theta[{idx}]"
    ax.set_title(lab, fontsize=9)
    ax.set_xlabel("draw")
    ax.set_ylabel(lab)
    ax.grid(True, alpha=0.2)
    return ax


def plot_theta_density(
    ax: plt.Axes,
    idata,
    idx: int,
    qpairs: Optional[Sequence[Sequence[float]]] = None,
    name: Optional[str] = None,
    bins: int = 60,
    show_median: bool = True,
    show_qpairs: bool = True,
) -> plt.Axes:
    """
    Plot a normalized histogram (density) for one theta component.
    """
    draws = theta_draws(idata, var="theta")
    if idx < 0 or idx >= draws.shape[1]:
        raise IndexError(f"idx={idx} out of range (theta has size {draws.shape[1]}).")

    x = draws[:, idx]
    ax.hist(x, bins=bins, density=True, alpha=0.9)

    if show_median:
        ax.axvline(np.percentile(x, 50.0), lw=1.0)

    if show_qpairs and qpairs is not None:
        q = normalize_qpairs(qpairs)
        for k in range(q.shape[0]):
            ax.axvline(np.percentile(x, q[k, 0]), lw=0.9, ls="--", alpha=0.8)
            ax.axvline(np.percentile(x, q[k, 1]), lw=0.9, ls="--", alpha=0.8)

    lab = name or f"theta[{idx}]"
    ax.set_title(lab, fontsize=9)
    ax.grid(True, alpha=0.2)
    return ax


# -----------------------------------------------------------------------------
# Model-space vertical plots (one quantity per axis)
# -----------------------------------------------------------------------------


def plot_depth_resistivity(
    ax: plt.Axes,
    summary: Mapping[str, Any],
    comp: int = 0,
    quantity: str = "resistivity",
    use_log10: bool = True,
    qpairs: Optional[Sequence[Sequence[float]]] = None,
    label_intervals: bool = False,
    legend: bool = False,
    invert_y: bool = True,
    sigma_floor: float = 1e-18,
) -> plt.Axes:
    """
    Plot depth profiles (layer steps) and credible envelopes for one principal component.

    This function is intended for *layered* (piecewise constant) model parameters,
    plotted versus depth-to-layer-bottom.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on.
    summary : mapping
        Summary dict loaded from ``*_pmc_summary.npz``.
        Must contain ``z_bot_med`` plus medians for the chosen quantity.
    comp : int
        Principal component index (0..2).
    quantity : str
        Either ``"resistivity"`` (default) or ``"conductivity"``.
        Conductivity is plotted as ``sigma = 1/rho`` in S/m.
    use_log10 : bool
        If True, plot on a log10 scale in x (i.e., x values are log10 of the quantity).
        If False, plot the quantity on a linear x-axis (but you may still set log-scale
        on the axis outside this function if desired).
    qpairs : sequence of pairs or None
        Credible-interval pairs (in percent) to label envelopes when present.
        If None, uses ``summary.get("model_qpairs")``.
    label_intervals : bool
        If True, add labels like "10-90%" for each filled interval.
    legend : bool
        If True, draw a legend.
    invert_y : bool
        If True, invert y-axis so depth increases downward (positive down).
    sigma_floor : float
        Floor for conductivity to avoid division-by-zero when converting 1/rho.
        Only used for ``quantity="conductivity"``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The same axis, for chaining.
    """
    quantity_n = str(quantity).strip().lower()
    if quantity_n not in {"resistivity", "conductivity"}:
        raise ValueError("quantity must be 'resistivity' or 'conductivity'.")

    z_bot = np.asarray(summary["z_bot_med"], dtype=np.float64).ravel()
    q = qpairs if qpairs is not None else summary.get("model_qpairs", None)

    # Base arrays in resistivity space (Ohm·m)
    rop_med = get_array(summary, "rop_med", dtype=np.float64)
    log10_rop_med = get_array(summary, "log10_rop_med", dtype=np.float64)
    rop_qlo = get_array(summary, "rop_qlo", dtype=np.float64)
    rop_qhi = get_array(summary, "rop_qhi", dtype=np.float64)
    log10_rop_qlo = get_array(summary, "log10_rop_qlo", dtype=np.float64)
    log10_rop_qhi = get_array(summary, "log10_rop_qhi", dtype=np.float64)

    if use_log10:
        if quantity_n == "resistivity":
            if log10_rop_med.size:
                med = np.asarray(log10_rop_med, dtype=np.float64)[:, comp]
            else:
                if not rop_med.size:
                    raise KeyError("summary must contain rop_med or log10_rop_med")
                med = np.log10(np.asarray(rop_med, dtype=np.float64)[:, comp])

            qlo_all = log10_rop_qlo if log10_rop_qlo.size else (np.log10(rop_qlo) if rop_qlo.size else np.array([]))
            qhi_all = log10_rop_qhi if log10_rop_qhi.size else (np.log10(rop_qhi) if rop_qhi.size else np.array([]))
            xlabel = "log10 ρ  [Ohm·m]" if quantity_n == "resistivity" else "log10 σ  [S/m]"
        else:
            # conductivity in log10: log10(sigma)= -log10(rho)
            if log10_rop_med.size:
                med = -np.asarray(log10_rop_med, dtype=np.float64)[:, comp]
            else:
                if not rop_med.size:
                    raise KeyError("summary must contain rop_med or log10_rop_med")
                med = np.log10(np.maximum(sigma_floor, 1.0 / np.asarray(rop_med, dtype=np.float64)[:, comp]))

            # Credible bounds: convert in resistivity space, then transform
            if log10_rop_qlo.size and log10_rop_qhi.size:
                qlo_all = -np.asarray(log10_rop_qhi, dtype=np.float64)  # note swap
                qhi_all = -np.asarray(log10_rop_qlo, dtype=np.float64)
            elif rop_qlo.size and rop_qhi.size:
                qlo_all = np.log10(np.maximum(sigma_floor, 1.0 / np.asarray(rop_qhi, dtype=np.float64)))  # swap
                qhi_all = np.log10(np.maximum(sigma_floor, 1.0 / np.asarray(rop_qlo, dtype=np.float64)))
            else:
                qlo_all = np.array([])
                qhi_all = np.array([])
            xlabel = "log10 σ  [S/m]"
    else:
        if quantity_n == "resistivity":
            if not rop_med.size:
                raise KeyError("summary must contain rop_med for use_log10=False")
            med = np.asarray(rop_med, dtype=np.float64)[:, comp]
            qlo_all = rop_qlo
            qhi_all = rop_qhi
            xlabel = "ρ  [Ohm·m]"
        else:
            if not rop_med.size:
                raise KeyError("summary must contain rop_med for use_log10=False")
            med = np.maximum(sigma_floor, 1.0 / np.asarray(rop_med, dtype=np.float64)[:, comp])
            if rop_qlo.size and rop_qhi.size:
                qlo_all = np.maximum(sigma_floor, 1.0 / np.asarray(rop_qhi, dtype=np.float64))  # swap
                qhi_all = np.maximum(sigma_floor, 1.0 / np.asarray(rop_qlo, dtype=np.float64))
            else:
                qlo_all = np.array([])
                qhi_all = np.array([])
            xlabel = "σ  [S/m]"

    v_qlo = v_qhi = None
    if qlo_all.size and qhi_all.size:
        v_qlo = np.asarray(qlo_all, dtype=np.float64)[:, :, comp]
        v_qhi = np.asarray(qhi_all, dtype=np.float64)[:, :, comp]

    plot_layer_steps(
        ax,
        z_bot,
        med,
        v_qlo=v_qlo,
        v_qhi=v_qhi,
        qpairs=q,
        label_intervals=label_intervals,
        invert_y=invert_y,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("depth [m]")
    ax.set_title(f"component {comp+1}", fontsize=10)
    if legend:
        ax.legend(fontsize=8, loc="best")
    return ax


def plot_vertical_resistivity(
    ax: plt.Axes,
    summary: Mapping[str, Any],
    comp: int = 0,
    use_log10: bool = True,
    qpairs: Optional[Sequence[Sequence[float]]] = None,
    label_intervals: bool = False,
    legend: bool = False,
) -> plt.Axes:
    """
    Deprecated wrapper for backward compatibility.

    Use :func:`plot_depth_resistivity` instead.
    """
    return plot_depth_resistivity(
        ax=ax,
        summary=summary,
        comp=comp,
        quantity="resistivity",
        use_log10=use_log10,
        qpairs=qpairs,
        label_intervals=label_intervals,
        legend=legend,
    )


def plot_depth_angle(
    ax: plt.Axes,
    summary: Mapping[str, Any],
    key: str = "ustr_deg",
    qpairs: Optional[Sequence[Sequence[float]]] = None,
    label_intervals: bool = False,
    legend: bool = False,
    invert_y: bool = True,
) -> plt.Axes:
    """
    Plot depth profiles (layer steps) and credible envelopes for one Euler-angle field.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on.
    summary : mapping
        Summary dict loaded from ``*_pmc_summary.npz``.
        Must contain ``z_bot_med`` and the chosen angle key.
    key : str
        One of: ``"ustr_deg"``, ``"udip_deg"``, ``"usla_deg"`` (or any other
        field stored in the summary with suffixes ``_med/_qlo/_qhi``).
    qpairs, label_intervals, legend : see :func:`plot_depth_resistivity`.
    invert_y : bool
        If True, invert y-axis so depth increases downward (positive down).

    Returns
    -------
    ax : matplotlib.axes.Axes
        The same axis, for chaining.
    """
    z_bot = np.asarray(summary["z_bot_med"], dtype=np.float64).ravel()
    q = qpairs if qpairs is not None else summary.get("model_qpairs", None)

    med = np.asarray(summary[f"{key}_med"], dtype=np.float64).ravel()
    qlo = get_array(summary, f"{key}_qlo", dtype=np.float64)
    qhi = get_array(summary, f"{key}_qhi", dtype=np.float64)

    plot_layer_steps(
        ax,
        z_bot,
        med,
        v_qlo=(qlo if qlo.size else None),
        v_qhi=(qhi if qhi.size else None),
        qpairs=q,
        label_intervals=label_intervals,
        invert_y=invert_y,
    )
    ax.set_xlabel("angle [deg]")
    ax.set_ylabel("depth [m]")
    ax.set_title(key, fontsize=10)
    if legend:
        ax.legend(fontsize=8, loc="best")
    return ax


def plot_vertical_angle(
    ax: plt.Axes,
    summary: Mapping[str, Any],
    key: str = "ustr_deg",
    qpairs: Optional[Sequence[Sequence[float]]] = None,
    label_intervals: bool = False,
    legend: bool = False,
) -> plt.Axes:
    """
    Deprecated wrapper for backward compatibility.

    Use :func:`plot_depth_angle` instead.
    """
    return plot_depth_angle(
        ax=ax,
        summary=summary,
        key=key,
        qpairs=qpairs,
        label_intervals=label_intervals,
        legend=legend,
    )


def plot_depth_thickness(
    ax: plt.Axes,
    summary: Mapping[str, Any],
    qpairs: Optional[Sequence[Sequence[float]]] = None,
    label_intervals: bool = False,
    legend: bool = False,
    invert_y: bool = True,
) -> plt.Axes:
    """
    Plot layer thickness as a depth-referenced step plot with optional credible envelopes.

    Note
    ----
    Thickness is a per-layer quantity; plotting it against depth can still be useful
    for comparing thickness credibility to the depth of interfaces.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on.
    summary : mapping
        Summary dict loaded from ``*_pmc_summary.npz``.
        Must contain ``z_bot_med`` and ``h_m_med`` (unless thickness is fixed).
    qpairs, label_intervals, legend : see :func:`plot_depth_resistivity`.
    invert_y : bool
        If True, invert y-axis so depth increases downward (positive down).

    Returns
    -------
    ax : matplotlib.axes.Axes
        The same axis, for chaining.
    """
    z_bot = np.asarray(summary["z_bot_med"], dtype=np.float64).ravel()
    q = qpairs if qpairs is not None else summary.get("model_qpairs", None)

    if "h_m_med" not in summary:
        raise KeyError("summary does not contain h_m_med (thickness may have been fixed).")

    med = np.asarray(summary["h_m_med"], dtype=np.float64).ravel()
    qlo = get_array(summary, "h_m_qlo", dtype=np.float64)
    qhi = get_array(summary, "h_m_qhi", dtype=np.float64)

    plot_layer_steps(
        ax,
        z_bot,
        med,
        v_qlo=(qlo if qlo.size else None),
        v_qhi=(qhi if qhi.size else None),
        qpairs=q,
        label_intervals=label_intervals,
        invert_y=invert_y,
    )
    ax.set_xlabel("thickness [m]")
    ax.set_ylabel("depth [m]")
    ax.set_title("thickness", fontsize=10)
    if legend:
        ax.legend(fontsize=8, loc="best")
    return ax


def plot_vertical_thickness(
    ax: plt.Axes,
    summary: Mapping[str, Any],
    qpairs: Optional[Sequence[Sequence[float]]] = None,
    label_intervals: bool = False,
    legend: bool = False,
) -> plt.Axes:
    """
    Deprecated wrapper for backward compatibility.

    Use :func:`plot_depth_thickness` instead.
    """
    return plot_depth_thickness(
        ax=ax,
        summary=summary,
        qpairs=qpairs,
        label_intervals=label_intervals,
        legend=legend,
    )


def plot_rho_phase_fit(
    ax_rho: plt.Axes,
    ax_phase: plt.Axes,
    summary: Mapping[str, Any],
    comp: str = "xy",
    rho_quantity: str = "resistivity",
    use_predictive: bool = True,
    qpairs: Optional[Sequence[Sequence[float]]] = None,
    label_intervals: bool = False,
    legend: bool = False,
) -> Tuple[plt.Axes, plt.Axes]:
    """
    Plot apparent resistivity (or apparent conductivity) and phase vs period for one impedance component.
    """
    freq = np.asarray(summary["freq"], dtype=np.float64)
    per = period_from_freq(freq)

    Z_obs = np.asarray(summary["Z_obs"], dtype=np.complex128)
    rho_obs, ph_obs = z_to_rho_phase(freq, Z_obs)

    rho_q = str(rho_quantity).strip().lower()
    if rho_q not in {"resistivity", "conductivity"}:
        raise ValueError("rho_quantity must be 'resistivity' or 'conductivity'.")
    if rho_q == "conductivity":
        rho_obs = np.maximum(1e-18, 1.0 / np.asarray(rho_obs, dtype=np.float64))

    i, j = comp_to_ij(comp)

    # predictive (already computed by sampler)
    rho_med = get_array(summary, "rho_med", dtype=np.float64)
    rho_qlo = get_array(summary, "rho_qlo", dtype=np.float64)
    rho_qhi = get_array(summary, "rho_qhi", dtype=np.float64)
    ph_med = get_array(summary, "phase_deg_med", dtype=np.float64)
    ph_qlo = get_array(summary, "phase_deg_qlo", dtype=np.float64)
    ph_qhi = get_array(summary, "phase_deg_qhi", dtype=np.float64)

    q = qpairs if qpairs is not None else summary.get("pred_qpairs", None)
    if rho_q == "conductivity" and rho_med.size:
        rho_med = np.maximum(1e-18, 1.0 / rho_med)
        if rho_qlo.size and rho_qhi.size:
            # invert and swap bounds
            rho_qlo, rho_qhi = np.maximum(1e-18, 1.0 / rho_qhi), np.maximum(1e-18, 1.0 / rho_qlo)

    qn = normalize_qpairs(q) if q is not None else np.zeros((0, 2), dtype=np.float64)

    if use_predictive and rho_med.size and ph_med.size:
        ax_rho.plot(per, rho_med[:, i, j], lw=1.6, label="pred median")
        ax_phase.plot(per, ph_med[:, i, j], lw=1.6, label="pred median")

        if rho_qlo.size and rho_qhi.size:
            for k in range(rho_qlo.shape[0]):
                lab = None
                if label_intervals and qn.size and k < qn.shape[0]:
                    lab = f"{qn[k,0]:.0f}-{qn[k,1]:.0f}%"
                ax_rho.fill_between(per, rho_qlo[k, :, i, j], rho_qhi[k, :, i, j], alpha=0.12, label=lab)
                ax_phase.fill_between(per, ph_qlo[k, :, i, j], ph_qhi[k, :, i, j], alpha=0.12)

    # observed points
    ax_rho.scatter(per, rho_obs[:, i, j], s=12, label="obs", zorder=3)
    ax_phase.scatter(per, ph_obs[:, i, j], s=12, label="obs", zorder=3)

    ax_rho.set_xscale("log")
    ax_rho.set_yscale("log")
    ax_rho.set_ylabel("ρ [Ohm·m]" if rho_q == "resistivity" else "σ_app [S/m]")
    ax_rho.set_title(f"Z{comp.lower()}", fontsize=10)
    ax_rho.grid(True, which="both", alpha=0.2)

    ax_phase.set_xscale("log")
    ax_phase.set_ylabel("phase [deg]")
    ax_phase.set_xlabel("period [s]")
    ax_phase.set_title(f"Z{comp.lower()}", fontsize=10)
    ax_phase.grid(True, which="both", alpha=0.2)

    if legend:
        ax_rho.legend(fontsize=8, loc="best")
    return ax_rho, ax_phase


def plot_phase_tensor_element(
    ax: plt.Axes,
    summary: Mapping[str, Any],
    comp: str = "xy",
    use_predictive: bool = True,
    qpairs: Optional[Sequence[Sequence[float]]] = None,
    label_intervals: bool = False,
    legend: bool = False,
) -> plt.Axes:
    """
    Plot one phase-tensor element vs period.
    """
    if summary.get("P_obs", None) is None:
        raise ValueError("No P_obs present in summary; cannot plot phase tensor element.")

    freq = np.asarray(summary["freq"], dtype=np.float64)
    per = period_from_freq(freq)
    P_obs = np.asarray(summary["P_obs"], dtype=np.float64)

    i, j = comp_to_ij(comp)

    P_med = get_array(summary, "P_med", dtype=np.float64)
    P_qlo = get_array(summary, "P_qlo", dtype=np.float64)
    P_qhi = get_array(summary, "P_qhi", dtype=np.float64)

    q = qpairs if qpairs is not None else summary.get("pred_qpairs", None)
    if rho_q == "conductivity" and rho_med.size:
        rho_med = np.maximum(1e-18, 1.0 / rho_med)
        if rho_qlo.size and rho_qhi.size:
            # invert and swap bounds
            rho_qlo, rho_qhi = np.maximum(1e-18, 1.0 / rho_qhi), np.maximum(1e-18, 1.0 / rho_qlo)

    qn = normalize_qpairs(q) if q is not None else np.zeros((0, 2), dtype=np.float64)

    if use_predictive and P_med.size:
        ax.plot(per, P_med[:, i, j], lw=1.6, label="pred median")
        if P_qlo.size and P_qhi.size:
            for k in range(P_qlo.shape[0]):
                lab = None
                if label_intervals and qn.size and k < qn.shape[0]:
                    lab = f"{qn[k,0]:.0f}-{qn[k,1]:.0f}%"
                ax.fill_between(per, P_qlo[k, :, i, j], P_qhi[k, :, i, j], alpha=0.12, label=lab)

    ax.scatter(per, P_obs[:, i, j], s=12, label="obs", zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("period [s]")
    ax.set_ylabel("P element")
    ax.set_title(f"P{comp.lower()}", fontsize=10)
    ax.grid(True, which="both", alpha=0.2)

    if legend:
        ax.legend(fontsize=8, loc="best")
    return ax
