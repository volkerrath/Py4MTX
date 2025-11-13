#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mt_dataviz.py
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
    """Add apparent resistivity curves to an axes (period on X, log-log)."""
    fig, ax, _ = _maybe_ax(ax)
    comps_list = _parse_comps(comps)
    period = 1.0 / df['freq'].to_numpy()
    for c in comps_list:
        ax.loglog(period, df[f'rho_{c}'].to_numpy(), marker='o', linestyle='-', label=f'ρa {c.upper()}', **line_kw)
    ax.invert_xaxis(); ax.set_xlabel('Period (s)'); ax.set_ylabel('Apparent resistivity (Ω·m)')
    if legend: ax.legend()
    return ax

def add_phase(df, comps: Optional[str] = None, ax: Optional[plt.Axes] = None,
              legend: bool = True, **line_kw) -> plt.Axes:
    """Add impedance phase curves to an axes (period on X, semilogx)."""
    fig, ax, _ = _maybe_ax(ax)
    comps_list = _parse_comps(comps)
    period = 1.0 / df['freq'].to_numpy()
    for c in comps_list:
        ax.semilogx(period, df[f'phi_{c}'].to_numpy(), marker='o', linestyle='-', label=f'φ {c.upper()}', **line_kw)
    ax.invert_xaxis(); ax.set_xlabel('Period (s)'); ax.set_ylabel('Phase (deg)')
    if legend: ax.legend()
    return ax

def add_tipper(df, ax: Optional[plt.Axes] = None, legend: bool = True, **line_kw) -> plt.Axes:
    """Add tipper (Tx, Ty) real/imag components to an axes (period on X)."""
    fig, ax, _ = _maybe_ax(ax)
    period = 1.0 / df['freq'].to_numpy()
    ax.semilogx(period, df['Tx_re'].to_numpy(), marker='o', linestyle='-', label='Re(Tx)', **line_kw)
    ax.semilogx(period, df['Tx_im'].to_numpy(), marker='^', linestyle='--', label='Im(Tx)', **line_kw)
    ax.semilogx(period, df['Ty_re'].to_numpy(), marker='s', linestyle='-', label='Re(Ty)', **line_kw)
    ax.semilogx(period, df['Ty_im'].to_numpy(), marker='d', linestyle='--', label='Im(Ty)', **line_kw)
    ax.invert_xaxis(); ax.set_xlabel('Period (s)'); ax.set_ylabel('Tipper (dimensionless)')
    if legend: ax.legend()
    return ax

def add_pt(df, ax: Optional[plt.Axes] = None, legend: bool = True, **line_kw) -> plt.Axes:
    """Add Phase Tensor components (PTxx, PTxy, PTyx, PTyy) to an axes."""
    fig, ax, _ = _maybe_ax(ax)
    period = 1.0 / df['freq'].to_numpy()
    ax.semilogx(period, df['ptxx_re'].to_numpy(), marker='o', linestyle='-', label='PTxx', **line_kw)
    ax.semilogx(period, df['ptxy_re'].to_numpy(), marker='^', linestyle='-', label='PTxy', **line_kw)
    ax.semilogx(period, df['ptyx_re'].to_numpy(), marker='s', linestyle='-', label='PTyx', **line_kw)
    ax.semilogx(period, df['ptyy_re'].to_numpy(), marker='d', linestyle='-', label='PTyy', **line_kw)
    ax.invert_xaxis(); ax.set_xlabel('Period (s)'); ax.set_ylabel('Phase Tensor (dimensionless)')
    if legend: ax.legend()
    return ax
