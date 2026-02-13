#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mcmc_viz.py
================

Axes-based plotting helpers for the simplified anisotropic 1-D MT inversion.

This module is intentionally *axes based*: all functions draw into an existing
Matplotlib ``Axes`` (or a list/array of axes) and return the modified axes.
The calling script is responsible for figure layout (``plt.subplots`` etc.).

The main goal (requested Feb 2026) is to produce **three-panel vertical
(step) profiles** for the available 3-parameter model sets, with support for:

- plotting a single profile (one model), and/or
- plotting posterior uncertainty as shaded quantile envelopes.

Supported 3-parameter sets
--------------------------

The helper :func:`plot_paramset_threepanel` supports these parameter sets
(``param_set``) and domains (``param_domain``):

**Resistivity domain ("rho")**

- ``"minmax"``: (rho_min, rho_max, strike)
- ``"max_anifac"``: (rho_max, rho_anifac, strike)

**Conductivity domain ("sigma")**

- ``"minmax"``: (sigma_min, sigma_max, strike)
- ``"max_anifac"``: (sigma_max, sigma_anifac, strike)

Notes
-----

*Quantiles:* If you provide an ArviZ ``InferenceData`` (``idata``), quantiles
are computed directly from the posterior samples of deterministic variables
when available. If those variables are not present, the code falls back to
arrays stored in the summary ``*.npz`` (keys like ``rho_min_med``,
``rho_min_qlo``, ``rho_min_qhi``).

*Depth axis:* Thickness arrays are expected to include a final “basement”
layer. The forward model ignores the basement thickness; for plotting, we set
it to 0 m (so the last depth edge equals the total thickness of finite
layers).

Author: Volker Rath (DIAS)
Created with the help of ChatGPT (GPT-5 Thinking) on 2026-02-13 (UTC)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


def open_idata(nc_path: str | Path):
    """Open an ArviZ ``InferenceData`` from a netCDF file.

    Parameters
    ----------
    nc_path : str or pathlib.Path
        Path to the ``*.nc`` file produced by the sampler.

    Returns
    -------
    arviz.InferenceData
        InferenceData loaded with :func:`arviz.from_netcdf`.
    """
    import arviz as az

    return az.from_netcdf(Path(nc_path).expanduser().as_posix())


def load_summary_npz(path: str | Path) -> dict:
    """Load a ``*_pmc_summary.npz`` into a plain ``dict``.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the summary file.

    Returns
    -------
    dict
        Dictionary with all NPZ entries. Small object arrays are unwrapped.
    """
    p = Path(path).expanduser()
    with np.load(p.as_posix(), allow_pickle=True) as npz:
        out = {k: npz[k] for k in npz.files}

    # allow_pickle=True stores object arrays; unwrap small objects
    for k, v in list(out.items()):
        if isinstance(v, np.ndarray) and v.dtype == object and v.size == 1:
            out[k] = v.item()
    return out


def _require_matplotlib():
    """Import Matplotlib lazily.

    Returns
    -------
    module
        ``matplotlib.pyplot``.
    """
    import matplotlib.pyplot as plt

    return plt


def depth_edges_from_h(h_m: np.ndarray, *, ignore_basement: bool = True) -> np.ndarray:
    """Compute depth edges from a thickness vector.

    Parameters
    ----------
    h_m : array_like, shape (nl,)
        Layer thicknesses in meters. Conventionally, the last layer is the
        basement.
    ignore_basement : bool, default True
        If True, the last thickness is set to 0 m for plotting.

    Returns
    -------
    z_edges_m : ndarray, shape (nl,)
        Depth edges (m), starting at 0.
    """
    h = np.asarray(h_m, dtype=float).ravel().copy()
    if h.size < 1:
        raise ValueError("h_m must be non-empty")
    if ignore_basement and h.size >= 1:
        h[-1] = 0.0
    # z_edges has length nl (matching the step vectors)
    z_edges = np.r_[0.0, np.cumsum(h[:-1])]
    return z_edges


def _stack_chain_draw_samples(da) -> np.ndarray:
    """Stack a posterior DataArray over chain and draw.

    Parameters
    ----------
    da : xarray.DataArray
        Posterior variable with dimensions like (chain, draw, layer).

    Returns
    -------
    samples : ndarray, shape (nsamples, nl)
        Stacked samples.
    """
    import xarray as xr

    if not isinstance(da, xr.DataArray):
        raise TypeError("Expected an xarray.DataArray")
    samp = da.stack(sample=("chain", "draw")).transpose("sample", ...).values
    samp = np.asarray(samp, dtype=float)
    if samp.ndim != 2:
        raise ValueError(f"Expected a 2-D stacked array, got shape {samp.shape}")
    return samp


def _posterior_samples_for_var(idata, candidates: Sequence[str]) -> np.ndarray | None:
    """Try to fetch per-layer posterior samples for the first existing variable.

    Parameters
    ----------
    idata : arviz.InferenceData
        InferenceData holding posterior samples.
    candidates : sequence of str
        Candidate variable names to try in ``idata.posterior``.

    Returns
    -------
    samples : ndarray or None
        Array with shape (nsamples, nl) or None if none of the candidates exist.
    """
    if idata is None:
        return None

    post = getattr(idata, "posterior", None)
    if post is None:
        return None

    for name in candidates:
        if name in post:
            return _stack_chain_draw_samples(post[name])
    return None


def _summary_arrays_for_var(summary: Mapping, base: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Get (qlo, med, qhi) arrays for a variable from a summary dictionary.

    The summary convention is:

    - ``{base}_qlo``
    - ``{base}_med``
    - ``{base}_qhi``

    Parameters
    ----------
    summary : mapping
        Summary dictionary.
    base : str
        Base name.

    Returns
    -------
    (qlo, med, qhi) : tuple of ndarray or None
        Arrays with shape (nl,) if present, else None.
    """
    kqlo = f"{base}_qlo"
    kmed = f"{base}_med"
    kqhi = f"{base}_qhi"
    if kqlo in summary and kmed in summary and kqhi in summary:
        return (
            np.asarray(summary[kqlo], dtype=float).ravel(),
            np.asarray(summary[kmed], dtype=float).ravel(),
            np.asarray(summary[kqhi], dtype=float).ravel(),
        )
    return None


def _safe_log10(x: np.ndarray) -> np.ndarray:
    """Compute log10(x) with a tiny floor to avoid ``-inf``.

    Parameters
    ----------
    x : ndarray
        Positive array.

    Returns
    -------
    ndarray
        ``log10(max(x, tiny))``.
    """
    x = np.asarray(x, dtype=float)
    tiny = np.finfo(float).tiny
    return np.log10(np.maximum(x, tiny))


def _fill_betweenx_step(ax, y_edges: np.ndarray, x1: np.ndarray, x2: np.ndarray, *, alpha: float = 0.25):
    """Fill between two step profiles along the depth axis.

    This tries ``ax.fill_betweenx(..., step='post')`` and falls back to a
    manual stair expansion for older Matplotlib versions.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    y_edges : ndarray, shape (nl,)
        Depth edges.
    x1, x2 : ndarray, shape (nl,)
        Lower and upper envelope values.
    alpha : float, default 0.25
        Fill transparency.

    Returns
    -------
    None
    """
    y_edges = np.asarray(y_edges, dtype=float).ravel()
    x1 = np.asarray(x1, dtype=float).ravel()
    x2 = np.asarray(x2, dtype=float).ravel()

    if not (y_edges.size == x1.size == x2.size):
        raise ValueError("y_edges, x1, x2 must have the same length")

    try:
        ax.fill_betweenx(y_edges, x1, x2, alpha=alpha, step="post", linewidth=0)
        return
    except TypeError:
        pass

    # Manual expansion: repeat values to draw rectangular steps
    # y: [z0, z0, z1, z1, ..., z_{n-1}, z_{n-1}]
    y = np.repeat(y_edges, 2)
    # x: [x0, x0, x1, x1, ..., x_{n-1}, x_{n-1}]
    x1s = np.repeat(x1, 2)
    x2s = np.repeat(x2, 2)
    ax.fill_betweenx(y, x1s, x2s, alpha=alpha, linewidth=0)


def plot_vertical_profile(
    ax,
    *,
    h_m: np.ndarray,
    values: np.ndarray,
    label: str,
    use_log10: bool = False,
    xlabel: str | None = None,
    linestyle: str = "-",
):
    """Plot a single step-style vertical profile.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    h_m : ndarray, shape (nl,)
        Thickness array.
    values : ndarray, shape (nl,)
        Per-layer values.
    label : str
        Legend label.
    use_log10 : bool, default False
        If True, plot ``log10(values)``.
    xlabel : str or None
        Optional x-axis label.
    linestyle : str, default "-"
        Line style.

    Returns
    -------
    matplotlib.axes.Axes
        The modified axes.
    """
    z_edges = depth_edges_from_h(h_m)
    v = np.asarray(values, dtype=float).ravel()
    if v.size != z_edges.size:
        raise ValueError(f"values must have length {z_edges.size}, got {v.size}")

    x = _safe_log10(v) if use_log10 else v
    ax.step(x, z_edges, where="post", label=label, linestyle=linestyle)

    ax.invert_yaxis()
    ax.set_ylabel("depth [m]")
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    return ax


def plot_vertical_envelope(
    ax,
    *,
    h_m: np.ndarray,
    qlo: np.ndarray,
    med: np.ndarray,
    qhi: np.ndarray,
    label: str,
    use_log10: bool = False,
    xlabel: str | None = None,
    shade: bool = True,
    show_quantile_lines: bool = False,
):
    """Plot a median step profile with an optional shaded quantile envelope.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    h_m : ndarray, shape (nl,)
        Thickness array.
    qlo, med, qhi : ndarray, shape (nl,)
        Lower quantile, median, upper quantile arrays.
    label : str
        Base label used in the legend.
    use_log10 : bool, default False
        If True, plot in log10 space.
    xlabel : str or None
        Optional x-axis label.
    shade : bool, default True
        If True, draw a translucent band between qlo and qhi.
    show_quantile_lines : bool, default False
        If True, draw dashed qlo/qhi step lines in addition to the band.

    Returns
    -------
    matplotlib.axes.Axes
        The modified axes.
    """
    z_edges = depth_edges_from_h(h_m)
    qlo = np.asarray(qlo, dtype=float).ravel()
    med = np.asarray(med, dtype=float).ravel()
    qhi = np.asarray(qhi, dtype=float).ravel()

    if not (qlo.size == med.size == qhi.size == z_edges.size):
        raise ValueError("qlo/med/qhi must all match the number of layers")

    if use_log10:
        qlo_p = _safe_log10(qlo)
        med_p = _safe_log10(med)
        qhi_p = _safe_log10(qhi)
    else:
        qlo_p, med_p, qhi_p = qlo, med, qhi

    line = ax.step(med_p, z_edges, where="post", label=f"{label} median")

    if shade:
        # Use the median line color for the envelope (explicit color reduces confusion).
        col = line[0].get_color()
        _fill_betweenx_step(ax, z_edges, qlo_p, qhi_p, alpha=0.25)
        for coll in ax.collections[-1:]:
            coll.set_facecolor(col)

    if show_quantile_lines:
        ax.step(qlo_p, z_edges, where="post", linestyle="--", label=f"{label} qlo")
        ax.step(qhi_p, z_edges, where="post", linestyle="--", label=f"{label} qhi")

    ax.invert_yaxis()
    ax.set_ylabel("depth [m]")
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    return ax


def plot_vertical_bands_from_samples(
    ax,
    *,
    h_m: np.ndarray,
    samples: np.ndarray,
    qpairs: Sequence[Sequence[float]] = ((0.1, 0.9),),
    label: str,
    use_log10: bool = False,
    xlabel: str | None = None,
    show_quantile_lines: bool = False,
):
    """Plot median + one or more shaded quantile bands from posterior samples.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    h_m : ndarray, shape (nl,)
        Thickness array.
    samples : ndarray, shape (nsamples, nl)
        Posterior samples for a per-layer parameter.
    qpairs : sequence of (qlo, qhi)
        Quantile pairs (in [0, 1]) to shade. Outer bands are drawn first.
    label : str
        Base label used in the legend.
    use_log10 : bool, default False
        If True, compute and plot quantiles in log10 space.
    xlabel : str or None
        Optional x-axis label.
    show_quantile_lines : bool, default False
        If True, also plot dashed lines for each band edge.

    Returns
    -------
    matplotlib.axes.Axes
        The modified axes.
    """
    z_edges = depth_edges_from_h(h_m)
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError("samples must have shape (nsamples, nl)")
    if samples.shape[1] != z_edges.size:
        raise ValueError("samples second dimension must match number of layers")

    # Work in requested space
    ss = _safe_log10(samples) if use_log10 else samples

    med = np.quantile(ss, 0.5, axis=0)
    line = ax.step(med, z_edges, where="post", label=f"{label} median")
    col = line[0].get_color()

    # Draw outer -> inner so inner remains visible
    qpairs_sorted = sorted([(float(a), float(b)) for a, b in qpairs], key=lambda t: (t[1] - t[0]), reverse=True)
    nband = max(len(qpairs_sorted), 1)

    for i, (qlo, qhi) in enumerate(qpairs_sorted):
        lo = np.quantile(ss, qlo, axis=0)
        hi = np.quantile(ss, qhi, axis=0)
        # Alpha increases towards inner bands
        alpha = 0.15 + 0.25 * (i + 1) / nband
        _fill_betweenx_step(ax, z_edges, lo, hi, alpha=alpha)
        # enforce same color
        for coll in ax.collections[-1:]:
            coll.set_facecolor(col)

        if show_quantile_lines:
            ax.step(lo, z_edges, where="post", linestyle="--", color=col, label=f"{label} q={qlo:.2f}")
            ax.step(hi, z_edges, where="post", linestyle="--", color=col, label=f"{label} q={qhi:.2f}")

    ax.invert_yaxis()
    ax.set_ylabel("depth [m]")
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    return ax


def _param_meta(param_domain: str, param_set: str) -> list[dict]:
    """Return plotting metadata for the requested domain and parameter set.

    Returns a list of 3 dicts with keys:

    - ``name``: canonical internal name (e.g. "rho_min")
    - ``label``: label string
    - ``xlabel``: x-axis label
    - ``use_log10``: whether to plot in log10
    - ``idata_candidates``: variable name candidates in idata.posterior
    - ``summary_base``: base key prefix in summary (e.g. "rho_min")

    Parameters
    ----------
    param_domain : {"rho", "sigma"}
        Resistivity or conductivity domain.
    param_set : {"minmax", "max_anifac"}
        Parameter set.

    Returns
    -------
    list of dict
        Exactly 3 parameter descriptors.
    """
    dom = str(param_domain).lower()
    pset = str(param_set).lower()
    if dom not in {"rho", "sigma"}:
        raise ValueError("param_domain must be 'rho' or 'sigma'")
    if pset not in {"minmax", "max_anifac"}:
        raise ValueError("param_set must be 'minmax' or 'max_anifac'")

    if dom == "rho" and pset == "minmax":
        return [
            {
                "name": "rho_min",
                "label": r"$\rho_{\min}$",
                "xlabel": r"$\log_{10}(\rho_{\min})$ [$\Omega\,m$]",
                "use_log10": True,
                "idata_candidates": ("rho_min",),
                "summary_base": "rho_min",
            },
            {
                "name": "rho_max",
                "label": r"$\rho_{\max}$",
                "xlabel": r"$\log_{10}(\rho_{\max})$ [$\Omega\,m$]",
                "use_log10": True,
                "idata_candidates": ("rho_max",),
                "summary_base": "rho_max",
            },
            {
                "name": "strike",
                "label": "strike",
                "xlabel": "strike [deg]",
                "use_log10": False,
                "idata_candidates": ("strike_deg", "strike"),
                "summary_base": "strike",
            },
        ]

    if dom == "rho" and pset == "max_anifac":
        return [
            {
                "name": "rho_max",
                "label": r"$\rho_{\max}$",
                "xlabel": r"$\log_{10}(\rho_{\max})$ [$\Omega\,m$]",
                "use_log10": True,
                "idata_candidates": ("rho_max",),
                "summary_base": "rho_max",
            },
            {
                "name": "rho_anifac",
                "label": r"$a_\rho=\sqrt{\rho_{\max}/\rho_{\min}}$",
                "xlabel": r"$\log_{10}(a_\rho)$",
                "use_log10": True,
                "idata_candidates": ("rho_anifac", "anifac_rho", "anifac"),
                "summary_base": "rho_anifac",
            },
            {
                "name": "strike",
                "label": "strike",
                "xlabel": "strike [deg]",
                "use_log10": False,
                "idata_candidates": ("strike_deg", "strike"),
                "summary_base": "strike",
            },
        ]

    if dom == "sigma" and pset == "minmax":
        return [
            {
                "name": "sigma_min",
                "label": r"$\sigma_{\min}$",
                "xlabel": r"$\log_{10}(\sigma_{\min})$ [S/m]",
                "use_log10": True,
                "idata_candidates": ("sigma_min", "sig_min"),
                "summary_base": "sigma_min",
            },
            {
                "name": "sigma_max",
                "label": r"$\sigma_{\max}$",
                "xlabel": r"$\log_{10}(\sigma_{\max})$ [S/m]",
                "use_log10": True,
                "idata_candidates": ("sigma_max", "sig_max"),
                "summary_base": "sigma_max",
            },
            {
                "name": "strike",
                "label": "strike",
                "xlabel": "strike [deg]",
                "use_log10": False,
                "idata_candidates": ("strike_deg", "strike"),
                "summary_base": "strike",
            },
        ]

    # dom == "sigma" and pset == "max_anifac"
    return [
        {
            "name": "sigma_max",
            "label": r"$\sigma_{\max}$",
            "xlabel": r"$\log_{10}(\sigma_{\max})$ [S/m]",
            "use_log10": True,
            "idata_candidates": ("sigma_max", "sig_max"),
            "summary_base": "sigma_max",
        },
        {
            "name": "sigma_anifac",
            "label": r"$a_\sigma=\sqrt{\sigma_{\max}/\sigma_{\min}}$",
            "xlabel": r"$\log_{10}(a_\sigma)$",
            "use_log10": True,
            "idata_candidates": ("sigma_anifac", "anifac_sigma", "anifac"),
            "summary_base": "sigma_anifac",
        },
        {
            "name": "strike",
            "label": "strike",
            "xlabel": "strike [deg]",
            "use_log10": False,
            "idata_candidates": ("strike_deg", "strike"),
            "summary_base": "strike",
        },
    ]


def _derive_anifac_from_minmax(values_max: np.ndarray, values_min: np.ndarray) -> np.ndarray:
    """Derive anisotropy factor samples/arrays from min/max values.

    The convention used in the simplified model is:

    - Resistivity anisotropy factor: ``a_rho = sqrt(rho_max / rho_min)``
    - Conductivity anisotropy factor: ``a_sigma = sqrt(sigma_max / sigma_min)``

    Parameters
    ----------
    values_max, values_min : ndarray
        Arrays with identical shapes. Can be per-layer vectors (nl,) or
        stacked samples (nsamples, nl).

    Returns
    -------
    ndarray
        ``sqrt(max/min)`` with a tiny floor on the denominator.
    """
    vmax = np.asarray(values_max, dtype=float)
    vmin = np.asarray(values_min, dtype=float)
    tiny = np.finfo(float).tiny
    return np.sqrt(np.maximum(vmax, tiny) / np.maximum(vmin, tiny))


def _get_h_m_from_summary_or_idata(summary: Mapping | None, idata) -> np.ndarray:
    """Get the layer thickness vector from either summary or idata.

    Parameters
    ----------
    summary : mapping or None
        Summary dictionary (preferred).
    idata : arviz.InferenceData
        InferenceData (fallback).

    Returns
    -------
    h_m : ndarray
        Thickness vector.
    """
    if summary is not None:
        if "h_m0" in summary:
            return np.asarray(summary["h_m0"], dtype=float).ravel()
        if "h_m" in summary:
            return np.asarray(summary["h_m"], dtype=float).ravel()

    if idata is not None:
        post = getattr(idata, "posterior", None)
        if post is not None and "h_m" in post:
            # take median thickness if stored as posterior var
            samp = _stack_chain_draw_samples(post["h_m"])
            return np.quantile(samp, 0.5, axis=0)

    raise KeyError("Could not find thickness vector (expected 'h_m0' in summary).")


def plot_paramset_threepanel(
    axs,
    *,
    summary: Mapping | None = None,
    idata=None,
    param_domain: str = "rho",
    param_set: str = "minmax",
    qpairs: Sequence[Sequence[float]] = ((0.1, 0.9),),
    prefer_idata: bool = True,
    show_quantile_lines: bool = False,
    overlay_single: Mapping[str, np.ndarray] | None = None,
):
    """Plot a three-panel vertical profile figure for one of the parameter sets.

    Parameters
    ----------
    axs : sequence of matplotlib.axes.Axes
        Three axes (top-to-bottom) typically created by:

        ``fig, axs = plt.subplots(3, 1, figsize=(8, 8))``

    summary : mapping or None
        Summary dictionary loaded by :func:`load_summary_npz`. Used as fallback
        source of (qlo, med, qhi) arrays.
    idata : arviz.InferenceData or None
        InferenceData loaded by :func:`open_idata`. If present and
        ``prefer_idata=True``, posterior samples are used to compute quantile
        bands for each parameter (supports multiple ``qpairs``).
    param_domain : {"rho", "sigma"}, default "rho"
        Resistivity or conductivity domain.
    param_set : {"minmax", "max_anifac"}, default "minmax"
        Which 3-parameter set to plot.
    qpairs : sequence of (qlo, qhi), default ((0.1, 0.9),)
        Quantile pairs for bands (only used when drawing from samples).
    prefer_idata : bool, default True
        Prefer deriving bands from ``idata`` when available.
    show_quantile_lines : bool, default False
        If True, also show band edge lines.
    overlay_single : mapping or None
        Optional single model to overlay. Expected to contain keys matching the
        selected parameter set (e.g. ``rho_min``/``rho_max``/``strike`` or
        ``rho_max``/``rho_anifac``/``strike``) and a thickness vector
        ``h_m`` or ``h_m0``.

    Returns
    -------
    axs
        The input axes, after plotting.
    """
    axs = list(axs)
    if len(axs) != 3:
        raise ValueError("axs must be a sequence of 3 axes")

    h_m = _get_h_m_from_summary_or_idata(summary, idata)

    metas = _param_meta(param_domain, param_set)

    for ax, meta in zip(axs, metas):
        name = meta["name"]
        label = meta["label"]
        xlabel = meta["xlabel"]
        use_log10 = bool(meta["use_log10"])

        # 1) Use posterior samples if possible
        samples = None
        if prefer_idata and idata is not None:
            samples = _posterior_samples_for_var(idata, meta["idata_candidates"])

            # If conductivity variables are not stored, try to derive them from
            # resistivity variables (sigma = 1/rho). This keeps the
            # "conductivity plotting" option usable even if only rho was saved.
            if samples is None and param_domain.lower() == "sigma" and name in {"sigma_min", "sigma_max"}:
                if name == "sigma_min":
                    # sigma_min = 1 / rho_max
                    r = _posterior_samples_for_var(idata, ("rho_max",))
                else:
                    # sigma_max = 1 / rho_min
                    r = _posterior_samples_for_var(idata, ("rho_min",))
                if r is not None:
                    tiny = np.finfo(float).tiny
                    samples = 1.0 / np.maximum(r, tiny)

            # If an explicit anifac variable is not available, try to derive it
            # from min/max variables present in the posterior.
            if samples is None and meta["name"].endswith("anifac"):
                if param_domain.lower() == "rho":
                    smax = _posterior_samples_for_var(idata, ("rho_max",))
                    smin = _posterior_samples_for_var(idata, ("rho_min",))
                else:
                    smax = _posterior_samples_for_var(idata, ("sigma_max", "sig_max"))
                    smin = _posterior_samples_for_var(idata, ("sigma_min", "sig_min"))
                if smax is not None and smin is not None:
                    samples = _derive_anifac_from_minmax(smax, smin)

        if samples is not None:
            plot_vertical_bands_from_samples(
                ax,
                h_m=h_m,
                samples=samples,
                qpairs=qpairs,
                label=label,
                use_log10=use_log10,
                xlabel=xlabel,
                show_quantile_lines=show_quantile_lines,
            )
        else:
            # 2) Fall back to summary arrays
            if summary is None:
                raise KeyError(
                    f"No posterior samples for '{name}' and no summary provided to fall back on."
                )
            trip = _summary_arrays_for_var(summary, meta["summary_base"])

            # Derive conductivity arrays from resistivity arrays if needed
            if trip is None and param_domain.lower() == "sigma" and meta["name"] in {"sigma_min", "sigma_max"}:
                if meta["name"] == "sigma_min":
                    # sigma_min = 1 / rho_max
                    rr = _summary_arrays_for_var(summary, "rho_max")
                else:
                    # sigma_max = 1 / rho_min
                    rr = _summary_arrays_for_var(summary, "rho_min")
                if rr is not None:
                    tiny = np.finfo(float).tiny
                    qlo = 1.0 / np.maximum(rr[0], tiny)
                    med = 1.0 / np.maximum(rr[1], tiny)
                    qhi = 1.0 / np.maximum(rr[2], tiny)
                    trip = (qlo, med, qhi)

            # Derive anifac from min/max arrays if needed
            if trip is None and meta["name"].endswith("anifac"):
                if param_domain.lower() == "rho":
                    tmax = _summary_arrays_for_var(summary, "rho_max")
                    tmin = _summary_arrays_for_var(summary, "rho_min")
                else:
                    tmax = _summary_arrays_for_var(summary, "sigma_max")
                    tmin = _summary_arrays_for_var(summary, "sigma_min")
                if tmax is not None and tmin is not None:
                    qlo = _derive_anifac_from_minmax(tmax[0], tmin[0])
                    med = _derive_anifac_from_minmax(tmax[1], tmin[1])
                    qhi = _derive_anifac_from_minmax(tmax[2], tmin[2])
                    trip = (qlo, med, qhi)

            if trip is None:
                raise KeyError(
                    f"Summary does not contain qlo/med/qhi for '{meta['summary_base']}'."
                )

            qlo, med, qhi = trip
            plot_vertical_envelope(
                ax,
                h_m=h_m,
                qlo=qlo,
                med=med,
                qhi=qhi,
                label=label,
                use_log10=use_log10,
                xlabel=xlabel,
                shade=True,
                show_quantile_lines=show_quantile_lines,
            )

        # Optional overlay of a single profile
        if overlay_single is not None:
            if "h_m" in overlay_single:
                h_m_ov = np.asarray(overlay_single["h_m"], dtype=float).ravel()
            elif "h_m0" in overlay_single:
                h_m_ov = np.asarray(overlay_single["h_m0"], dtype=float).ravel()
            else:
                h_m_ov = h_m

            # Accept either canonical key or summary-like key
            ov_val = None
            for k in (name, f"{name}_deg"):
                if k in overlay_single:
                    ov_val = np.asarray(overlay_single[k], dtype=float).ravel()
                    break

            # Derive overlay anifac from overlay min/max if not present
            if ov_val is None and name.endswith("anifac"):
                if param_domain.lower() == "rho":
                    vmax = overlay_single.get("rho_max")
                    vmin = overlay_single.get("rho_min")
                else:
                    vmax = overlay_single.get("sigma_max")
                    vmin = overlay_single.get("sigma_min")
                if vmax is not None and vmin is not None:
                    ov_val = _derive_anifac_from_minmax(vmax, vmin).ravel()
            if ov_val is not None:
                z_edges = depth_edges_from_h(h_m_ov)
                x = _safe_log10(ov_val) if use_log10 else ov_val
                ax.step(x, z_edges, where="post", linestyle=":", label=f"{label} (single)")
                ax.legend()

    return axs
