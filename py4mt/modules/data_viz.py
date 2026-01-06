#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ediviz.py
=================
Matplotlib helpers for magnetotelluric (MT) transfer-function plots.

This module is designed to work with :class:`pandas.DataFrame` objects
created from EDI data (for example using :mod:`ediproc` + :mod:`edidat`).
All plotting functions accept an optional Matplotlib ``Axes``; if ``ax`` is
None, a new figure/axes pair is created.

Each plotter can optionally visualise 1-sigma uncertainties as
semi-transparent ``fill_between`` envelopes, assuming that the input
dataframe contains matching ``*_err`` columns.

Expected column naming
----------------------

The following columns are recognised (all optional except ``freq``):

- Frequencies:
    - ``freq`` : frequency [Hz] (required)
    - ``period`` : period [s] is convenient but not required – it is
      recomputed from ``freq`` if missing.

- Apparent resistivity and phase:
    - ``rho_xx``, ``rho_xy``, ``rho_yx``, ``rho_yy`` [Ω·m]
    - ``phi_xx``, ``phi_xy``, ``phi_yx``, ``phi_yy`` [deg]
    - and optional error columns with suffix ``_err``:
      ``rho_xy_err``, ``phi_xy_err``, ...

- Tipper:
    - ``Tx_re``, ``Tx_im``, ``Ty_re``, ``Ty_im``
    - optional uncertainties:
      ``Tx_re_err``, ``Tx_im_err``, ``Ty_re_err``, ``Ty_im_err``

- Phase tensor:
    - ``ptxx_re``, ``ptxy_re``, ``ptyx_re``, ``ptyy_re``
    - optional uncertainties:
      ``ptxx_re_err``, ``ptxy_re_err``, ``ptyx_re_err``, ``ptyy_re_err``

All plotters share the arguments ``show_errors``, ``error_suffix`` and
``error_alpha`` to control the error visualisation.

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-11-20
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parse_comps(comp_str: Optional[str]) -> List[str]:
    """Parse a comma-separated list of impedance component labels.

    Parameters
    ----------
    comp_str : str or None
        Comma-separated list of labels (for example ``"xy,yx"``). Valid
        entries are ``"xx"``, ``"xy"``, ``"yx"``, ``"yy"`` (case-insensitive).
        If None or empty, a sensible default ``["xy", "yx"]`` is used.

    Returns
    -------
    list of str
        Normalised component labels in lower case, filtered to valid ones.
        If no valid labels are found, the default ``["xy", "yx"]`` is
        returned.
    """
    allowed = {"xx", "xy", "yx", "yy"}
    if not comp_str:
        return ["xy", "yx"]
    comps = [c.strip().lower() for c in comp_str.split(",") if c.strip()]
    comps = [c for c in comps if c in allowed]
    return comps or ["xy", "yx"]


def _maybe_ax(ax=None):
    """Ensure a Matplotlib axes object is available.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        Existing axes instance or None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure associated with the axes.
    ax : matplotlib.axes.Axes
        The (possibly newly created) axes instance.
    created_new : bool
        True if a new figure/axes was created, False otherwise.
    """
    if ax is None:
        fig, ax = plt.subplots()
        return fig, ax, True
    return ax.figure, ax, False


def _period_from_df(df: pd.DataFrame) -> np.ndarray:
    """Compute or retrieve period array (seconds) from a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe with a ``"freq"`` column in Hz and optionally a
        ``"period"`` column.

    Returns
    -------
    numpy.ndarray
        1-D array of periods [s].
    """
    if "period" in df.columns:
        return df["period"].to_numpy()
    freq = df["freq"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        period = np.where(freq > 0.0, 1.0 / freq, np.nan)
    return period


def _maybe_fill_between(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    err: Optional[np.ndarray],
    *,
    alpha: float = 0.25,
) -> None:
    """Draw a symmetric error band ``[y - err, y + err]`` if errors are provided.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    x : array_like
        Abscissa values (for example periods).
    y : array_like
        Central curve values.
    err : array_like or None
        1-sigma error values. If None, nothing is drawn.
    alpha : float, optional
        Opacity of the filled band. Default is 0.25.
    """
    if err is None:
        return
    y = np.asarray(y, dtype=float)
    err = np.asarray(err, dtype=float)
    if y.shape != err.shape:
        return
    lower = y - err
    upper = y + err
    ax.fill_between(x, lower, upper, alpha=alpha, linewidth=0)


def add_rho(
    df: pd.DataFrame,
    comps: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    legend: bool = True,
    *,
    show_errors: bool = False,
    error_suffix: str = "_err",
    error_alpha: float = 0.25,
    **line_kw,
) -> plt.Axes:
    """Add apparent resistivity curves to an axes.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing at least the columns ``"freq"`` and some of
        ``"rho_xx"``, ``"rho_xy"``, ``"rho_yx"``, ``"rho_yy"`` in Ω·m.
    comps : str, optional
        Comma-separated list of components to plot (subset of
        ``"xx,xy,yx,yy"``). If None, the default is ``"xy,yx"``.
    ax : matplotlib.axes.Axes, optional
        Target axes. If None, a new figure and axes are created.
    legend : bool, optional
        Whether to display a legend (default is True).
    show_errors : bool, optional
        If True, look for columns named
        ``rho_<comp><error_suffix>`` (for example ``rho_xy_err``) and draw
        a semi-transparent error envelope.
    error_suffix : str, optional
        Suffix appended to the base column name when searching for error
        columns. Default is ``"_err"``.
    error_alpha : float, optional
        Opacity of the error envelopes (0–1). Default is 0.25.
    **line_kw :
        Additional keyword arguments forwarded to ``ax.loglog``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes instance with the curves added.
    """
    fig, ax, _ = _maybe_ax(ax)
    period = _period_from_df(df)
    comps_list = _parse_comps(comps)

    for c in comps_list:
        col = f"rho_{c}"
        if col not in df.columns:
            continue
        y = df[col].to_numpy()
        ax.loglog(
            period,
            y,
            marker="o",
            linestyle="-",
            label=f"$\\rho_a,{c.upper()}$",
            **line_kw,
        )
        if show_errors:
            err_col = f"{col}{error_suffix}"
            if err_col in df.columns:
                err = df[err_col].to_numpy()
                _maybe_fill_between(ax, period, y, err, alpha=error_alpha)

    ax.invert_xaxis()
    ax.set_xlabel("Period (s)")
    ax.set_ylabel("Apparent resistivity $\\rho_a$ ($\\Omega\\,\\mathrm{m}$)")
    ax.grid(True, linestyle=":")
    if legend:
        ax.legend()
    return ax


def add_phase(
    df: pd.DataFrame,
    comps: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    legend: bool = True,
    *,
    show_errors: bool = False,
    error_suffix: str = "_err",
    error_alpha: float = 0.25,
    **line_kw,
) -> plt.Axes:
    """Add impedance phase curves to an axes.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing at least ``"freq"`` and some of ``"phi_xx"``,
        ``"phi_xy"``, ``"phi_yx"``, ``"phi_yy"`` in degrees.
    comps : str, optional
        Comma-separated list of components to plot (subset of
        ``"xx,xy,yx,yy"``). If None, the default is ``"xy,yx"``.
    ax : matplotlib.axes.Axes, optional
        Target axes. If None, a new figure and axes are created.
    legend : bool, optional
        Whether to display a legend (default is True).
    show_errors : bool, optional
        If True, look for columns named
        ``phi_<comp><error_suffix>`` (for example ``phi_xy_err``) and draw
        a semi-transparent error envelope.
    error_suffix : str, optional
        Suffix appended to the base column name when searching for error
        columns. Default is ``"_err"``.
    error_alpha : float, optional
        Opacity of the error envelopes (0–1). Default is 0.25.
    **line_kw :
        Additional keyword arguments forwarded to ``ax.semilogx``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes instance with the curves added.
    """
    fig, ax, _ = _maybe_ax(ax)
    period = _period_from_df(df)
    comps_list = _parse_comps(comps)

    for c in comps_list:
        col = f"phi_{c}"
        if col not in df.columns:
            continue
        y = df[col].to_numpy()
        ax.semilogx(
            period,
            y,
            marker="o",
            linestyle="-",
            label=f"$\\phi,{c.upper()}$",
            **line_kw,
        )
        if show_errors:
            err_col = f"{col}{error_suffix}"
            if err_col in df.columns:
                err = df[err_col].to_numpy()
                _maybe_fill_between(ax, period, y, err, alpha=error_alpha)

    ax.invert_xaxis()
    ax.set_xlabel("Period (s)")
    ax.set_ylabel("Phase $\\phi$ (deg)")
    ax.grid(True, linestyle=":")
    if legend:
        ax.legend()
    return ax


def add_tipper(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    legend: bool = True,
    *,
    show_errors: bool = False,
    error_suffix: str = "_err",
    error_alpha: float = 0.25,
    **line_kw,
) -> plt.Axes:
    """Add tipper components (Tx, Ty) to an axes.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing at least ``"freq"`` and some of
        ``"Tx_re"``, ``"Tx_im"``, ``"Ty_re"``, ``"Ty_im"``.
    ax : matplotlib.axes.Axes, optional
        Target axes. If None, a new figure and axes are created.
    legend : bool, optional
        Whether to display a legend (default is True).
    show_errors : bool, optional
        If True, look for columns like ``Tx_re_err``, ``Tx_im_err``,
        ``Ty_re_err``, ``Ty_im_err`` and draw symmetric error envelopes.
    error_suffix : str, optional
        Suffix appended to the base column name when searching for error
        columns. Default is ``"_err"``.
    error_alpha : float, optional
        Opacity of the error envelopes (0–1). Default is 0.25.
    **line_kw :
        Additional keyword arguments forwarded to ``ax.semilogx``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes instance with the curves added.
    """
    fig, ax, _ = _maybe_ax(ax)
    period = _period_from_df(df)

    specs = [
        ("Tx_re", "o", "-", "Re(Tx)"),
        ("Tx_im", "^", "--", "Im(Tx)"),
        ("Ty_re", "s", "-", "Re(Ty)"),
        ("Ty_im", "d", "--", "Im(Ty)"),
    ]

    for col, marker, linestyle, label in specs:
        if col not in df.columns:
            continue
        y = df[col].to_numpy()
        ax.semilogx(
            period,
            y,
            marker=marker,
            linestyle=linestyle,
            label=label,
            **line_kw,
        )
        if show_errors:
            err_col = f"{col}{error_suffix}"
            if err_col in df.columns:
                err = df[err_col].to_numpy()
                _maybe_fill_between(ax, period, y, err, alpha=error_alpha)

    ax.invert_xaxis()
    ax.set_xlabel("Period (s)")
    ax.set_ylabel("Tipper (dimensionless)")
    ax.grid(True, linestyle=":")
    if legend:
        ax.legend()
    return ax


def add_pt(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    legend: bool = True,
    *,
    show_errors: bool = False,
    error_suffix: str = "_err",
    error_alpha: float = 0.25,
    **line_kw,
) -> plt.Axes:
    """Add phase tensor components to an axes.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing at least ``"freq"`` and some of
        ``"ptxx_re"``, ``"ptxy_re"``, ``"ptyx_re"``, ``"ptyy_re"``.
    ax : matplotlib.axes.Axes, optional
        Target axes. If None, a new figure and axes are created.
    legend : bool, optional
        Whether to display a legend (default is True).
    show_errors : bool, optional
        If True, look for columns such as ``ptxx_re_err`` and draw
        symmetric error envelopes.
    error_suffix : str, optional
        Suffix appended to the base column name when searching for error
        columns. Default is ``"_err"``.
    error_alpha : float, optional
        Opacity of the error envelopes (0–1). Default is 0.25.
    **line_kw :
        Additional keyword arguments forwarded to ``ax.semilogx``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes instance with the curves added.
    """
    fig, ax, _ = _maybe_ax(ax)
    period = _period_from_df(df)

    comps = [
        ("ptxx_re", "o", "-", "PTxx"),
        ("ptxy_re", "^", "-", "PTxy"),
        ("ptyx_re", "s", "-", "PTyx"),
        ("ptyy_re", "d", "-", "PTyy"),
    ]

    for col, marker, linestyle, label in comps:
        if col not in df.columns:
            continue
        y = df[col].to_numpy()
        ax.semilogx(
            period,
            y,
            marker=marker,
            linestyle=linestyle,
            label=label,
            **line_kw,
        )
        if show_errors:
            err_col = f"{col}{error_suffix}"
            if err_col in df.columns:
                err = df[err_col].to_numpy()
                _maybe_fill_between(ax, period, y, err, alpha=error_alpha)

    ax.invert_xaxis()
    ax.set_xlabel("Period (s)")
    ax.set_ylabel("Phase tensor (dimensionless)")
    ax.grid(True, linestyle=":")
    if legend:
        ax.legend()
    return ax


def plot_gridx(data_list: List[Optional[dict]], nrows: int, ncols: int, figsize=(10, 6)):
    """
    Plot multiple datasets in a subplot grid, removing empty axes.

    Parameters
    ----------
    data_list : list of dict
        Each dict may contain:
        - 'type': str, one of {'line', 'scatter', 'image', 'hist'}
        - 'x': array-like, x-values (for line/scatter/hist)
        - 'y': array-like, y-values (for line/scatter)
        - 'style': str, optional matplotlib style (line only)
        - 'img': 2D array, for image plots
        - 'bins': int, optional number of bins (hist)
    nrows : int
        Number of subplot rows.
    ncols : int
        Number of subplot columns.
    figsize : tuple
        Figure size.


    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : list of matplotlib.axes.Axes
        List of active axes (after removing empties).

    Usage:

        import numpy as np
        from plot_grid import plot_grid

        x = np.linspace(0, 2*np.pi, 100)
        img = np.random.rand(10, 10)

        datasets = [
            {"type": "line", "x": x, "y": np.sin(x), "style": "b-"},
            {"type": "scatter", "x": x, "y": np.cos(x), "c": "r"},
            {"type": "image", "img": img, "cmap": "plasma"},
            {"type": "hist", "x": np.random.randn(1000), "bins": 30, "color": "g"},
            None,  # empty slot
            None   # empty slot
        ]

        fig, axes = plot_grid(datasets, nrows=2, ncols=3, figsize=(12, 6))
        fig.show()

    Author: Volker Rath (DIAS)
    Generated by Copilot v1.0
    Date: 2025-11-26
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for ax, data in zip(axes, data_list):
        if data is None or "type" not in data:
            fig.delaxes(ax)
            continue

        dtype = data["type"]

        if dtype == "line" and "x" in data and "y" in data:
            style = data.get("style", "-")
            ax.plot(data["x"], data["y"], style)

        elif dtype == "scatter" and "x" in data and "y" in data:
            ax.scatter(data["x"], data["y"], c=data.get("c", "b"), marker=data.get("marker", "o"))

        elif dtype == "image" and "img" in data:
            ax.imshow(data["img"], cmap=data.get("cmap", "viridis"), aspect="auto")

        elif dtype == "hist" and "x" in data:
            bins = data.get("bins", 20)
            ax.hist(data["x"], bins=bins, color=data.get("color", "gray"), alpha=0.7)

        else:
            fig.delaxes(ax)

    fig.tight_layout()
    return fig, [ax for ax in fig.axes]



def plot_grid(data_list: List[Optional[dict]], nrows: int, ncols: int, figsize=(10, 6)):
    """
    Plot multiple datasets in a subplot grid, removing empty axes.

    Parameters
    ----------
    data_list : list of dict
        Each dict should contain keys:
        - 'x': array-like, x-values
        - 'y': array-like, y-values
        - 'style': str, optional matplotlib style (e.g. 'r--')
    nrows : int
        Number of subplot rows.
    ncols : int
        Number of subplot columns.
    figsize : tuple
        Figure size.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : list of matplotlib.axes.Axes
        List of active axes (after removing empties).

    Usage:

        import numpy as np
        from plot_grid import plot_grid

        x = np.linspace(0, 2*np.pi, 100)
        datasets = [
            {"x": x, "y": np.sin(x), "style": "b-"},
            {"x": x, "y": np.cos(x), "style": "r--"},
            None,  # empty slot
            {"x": x, "y": np.exp(-x/3), "style": "g-"},
            None,
            None
        ]
        fig, axes = plot_grid(datasets, nrows=2, ncols=3, figsize=(12, 6))
        fig.show()


    Author: Volker Rath (DIAS)
    Generated by Copilot v1.0
    Date: 2025-11-26
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for ax, data in zip(axes, data_list):
        if data is not None and "x" in data and "y" in data:
            style = data.get("style", "-")
            ax.plot(data["x"], data["y"], style)
        else:
            # mark empty axes for removal
            fig.delaxes(ax)

    fig.tight_layout()
    return fig, [ax for ax in fig.axes]
