#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ediviz.py
=================
Matplotlib helpers to add MT (magnetotelluric) plots into an existing figure layout.

This module reuses computations from :mod:`edi_viz` (dataframe building and
derived columns) and focuses on composable plotting primitives that accept an
optional Matplotlib ``Axes``. If ``ax`` is None, a new figure/axes is created.

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-11-12 18:35:56 UTC
"""

from __future__ import annotations

from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parse_comps(comp_str: Optional[str]) -> List[str]:
    """Validate a comma-separated component list ('xx,xy,yx,yy').

    Returns ['xy','yx'] if input is empty or invalid.
    """
    allowed = {'xx','xy','yx','yy'}
    if not comp_str:
        return ['xy','yx']
    out = [c.strip().lower() for c in comp_str.split(',') if c.strip()]
    out = [c for c in out if c in allowed]
    return out or ['xy','yx']

def _maybe_ax(ax=None):
    """Return (fig, ax, created_new). Create new figure if ax is None."""
    if ax is None:
        fig, ax = plt.subplots()
        return fig, ax, True
    return ax.figure, ax, False

def add_rho(df, comps: Optional[str] = None, ax: Optional[plt.Axes] = None,
            legend: bool = True, **line_kw) -> plt.Axes:
    """Add apparent resistivity curves to an axes.

    This helper plots apparent resistivity as a function of period on a
    log-log scale.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data frame containing at least ``freq`` and ``rho_*`` columns.
    comps : str, optional
        Comma-separated list of tensor components (``"xx,xy,yx,yy"``).
        If None, a sensible default subset is used.
    ax : matplotlib.axes.Axes, optional
        Target axes. If None, a new figure and axes are created.
    legend : bool, optional
        If True, draw a legend (default is True).
    **line_kw :
        Additional keyword arguments forwarded to Matplotlib ``loglog``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes instance with the curves added.
    """
    fig, ax, _ = _maybe_ax(ax)
    comps_list = _parse_comps(comps)
    period = 1.0 / df['freq'].to_numpy()
    for c in comps_list:
        ax.loglog(period, df[f'rho_{c}'].to_numpy(), marker='o', linestyle='-', label=f'$\\rho_a\\,{c.upper()}$', **line_kw)
    ax.invert_xaxis(); ax.set_xlabel('Period (s)'); ax.set_ylabel('Apparent resistivity $\\rho_a$ ($\\Omega\\,\\mathrm{m}$)')
    if legend: ax.legend()
    ax.grid(True, linestyle=":")
    return ax

def add_phase(df, comps: Optional[str] = None, ax: Optional[plt.Axes] = None,
              legend: bool = True, **line_kw) -> plt.Axes:
    """Add impedance phase curves to an axes.

    This helper plots impedance phase as a function of period on a
    semi-logarithmic scale (log-period, linear phase).

    Parameters
    ----------
    df : pandas.DataFrame
        Input data frame containing at least ``freq`` and ``phi_*`` columns.
    comps : str, optional
        Comma-separated list of tensor components (``"xx,xy,yx,yy"``).
        If None, a sensible default subset is used.
    ax : matplotlib.axes.Axes, optional
        Target axes. If None, a new figure and axes are created.
    legend : bool, optional
        If True, draw a legend (default is True).
    **line_kw :
        Additional keyword arguments forwarded to Matplotlib ``semilogx``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes instance with the curves added.
    """
    fig, ax, _ = _maybe_ax(ax)
    comps_list = _parse_comps(comps)
    period = 1.0 / df['freq'].to_numpy()
    for c in comps_list:
        ax.semilogx(period, df[f'phi_{c}'].to_numpy(), marker='o', linestyle='-', label=f'$\\phi\\,{c.upper()}$', **line_kw)
    ax.invert_xaxis(); ax.set_xlabel('Period (s)'); ax.set_ylabel('Phase $\\phi$ (deg)')
    ax.grid(True, linestyle=":")
    if legend: ax.legend()
    return ax

def add_tipper(df, ax: Optional[plt.Axes] = None, legend: bool = True, **line_kw) -> plt.Axes:
    """Add tipper (Tx, Ty) real/imaginary components to an axes.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data frame containing at least ``freq`` and tipper columns
        ``Tx_re``, ``Tx_im``, ``Ty_re``, ``Ty_im``.
    ax : matplotlib.axes.Axes, optional
        Target axes. If None, a new figure and axes are created.
    legend : bool, optional
        If True, draw a legend (default is True).
    **line_kw :
        Additional keyword arguments forwarded to Matplotlib ``semilogx``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes instance with the curves added.
    """
    fig, ax, _ = _maybe_ax(ax)
    period = 1.0 / df['freq'].to_numpy()
    ax.semilogx(period, df['Tx_re'].to_numpy(), marker='o', linestyle='-', label='Re(Tx)', **line_kw)
    ax.semilogx(period, df['Tx_im'].to_numpy(), marker='^', linestyle='--', label='Im(Tx)', **line_kw)
    ax.semilogx(period, df['Ty_re'].to_numpy(), marker='s', linestyle='-', label='Re(Ty)', **line_kw)
    ax.semilogx(period, df['Ty_im'].to_numpy(), marker='d', linestyle='--', label='Im(Ty)', **line_kw)
    ax.invert_xaxis(); ax.set_xlabel('Period (s)'); ax.set_ylabel('Tipper (dimensionless)')
    ax.grid(True, linestyle=":")
    if legend: ax.legend()
    return ax

def add_pt(df, ax: Optional[plt.Axes] = None, legend: bool = True, **line_kw) -> plt.Axes:
    """Add Phase Tensor components (PTxx, PTxy, PTyx, PTyy) to an axes.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data frame containing at least ``freq`` and PT columns
        ``ptxx_re``, ``ptxy_re``, ``ptyx_re``, ``ptyy_re``.
    ax : matplotlib.axes.Axes, optional
        Target axes. If None, a new figure and axes are created.
    legend : bool, optional
        If True, draw a legend (default is True).
    **line_kw :
        Additional keyword arguments forwarded to Matplotlib ``semilogx``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes instance with the curves added.
    """
    fig, ax, _ = _maybe_ax(ax)
    period = 1.0 / df['freq'].to_numpy()
    ax.semilogx(period, df['ptxx_re'].to_numpy(), marker='o', linestyle='-', label='PTxx', **line_kw)
    ax.semilogx(period, df['ptxy_re'].to_numpy(), marker='^', linestyle='-', label='PTxy', **line_kw)
    ax.semilogx(period, df['ptyx_re'].to_numpy(), marker='s', linestyle='-', label='PTyx', **line_kw)
    ax.semilogx(period, df['ptyy_re'].to_numpy(), marker='d', linestyle='-', label='PTyy', **line_kw)
    ax.invert_xaxis(); ax.set_xlabel('Period (s)'); ax.set_ylabel('Phase Tensor (dimensionless)')
    ax.grid(True, linestyle=":")
    if legend: ax.legend()
    return ax
