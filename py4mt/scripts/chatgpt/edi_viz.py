#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
edi_viz.py
=================
Plotting utilities for Magnetotelluric EDI transfer functions (Z, T, PT).

This module provides small, dependency-light helpers to visualize:
- Apparent resistivity (\u03C1_a) for Zxx/Zxy/Zyx/Zyy
- Phase (\u03C6) for Zxx/Zxy/Zyx/Zyy
- Tipper (Tx, Ty) real/imag parts
- Phase Tensor components (PTxx, PTxy, PTyx, PTyy)

Public API
----------
- dataframe_from_arrays(freq, Z, T=None)
- plot_rho(df, station, outdir, comps=None)
- plot_phase(df, station, outdir, comps=None)
- plot_tipper(df, station, outdir)
- plot_pt(df, station, outdir)

Notes
-----
- Period is defined as 1/f. X-axis is logarithmic with decreasing period to the right.
- Files are saved as PNG in the chosen output directory.

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-11-12 18:22:48 UTC
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


MU0: float = 4e-7 * np.pi


def _ensure_plot_dir(plot_dir: Path | str) -> Path:
    """Create and return a plot output directory.

    Parameters
    ----------
    plot_dir : str or Path
        Path to the directory where plots should be written.

    Returns
    -------
    pathlib.Path
        Existing directory path. The directory is created if it does not exist.
    """
    d = Path(plot_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _parse_comps(comp_str: Optional[str]) -> List[str]:
    """Parse a component list string like "xy,yx" into a validated list.

    Parameters
    ----------
    comp_str : str or None
        Comma-separated list containing any of: 'xx, xy, yx, yy'.

    Returns
    -------
    list of str
        Validated list (defaults to ['xy','yx'] if empty or invalid).
    """
    allowed = {"xx", "xy", "yx", "yy"}
    if not comp_str:
        return ["xy", "yx"]
    comps = [c.strip().lower() for c in comp_str.split(",") if c.strip()]
    comps = [c for c in comps if c in allowed]
    return comps or ["xy", "yx"]


def dataframe_from_arrays(freq: np.ndarray, Z: np.ndarray, T: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Build a pandas DataFrame with columns matching `edi_processor` CSV schema.

    Parameters
    ----------
    freq : ndarray, shape (n,)
        Frequencies in Hz.
    Z : ndarray, shape (n,2,2), complex128
        Impedance tensor per frequency.
    T : ndarray or None, shape (n,1,2), complex128, optional
        Tipper per frequency. If None, zeroed columns are used.

    Returns
    -------
    pandas.DataFrame
        Columns: freq, Zxx_re/_im, Zxy_re/_im, Zyx_re/_im, Zyy_re/_im,
                 Tx_re/_im, Ty_re/_im, plus derived rho/phi and PT entries.
    """
    n = freq.shape[0]
    df = pd.DataFrame({
        'freq': freq,
        'Zxx_re': Z[:,0,0].real, 'Zxx_im': Z[:,0,0].imag,
        'Zxy_re': Z[:,0,1].real, 'Zxy_im': Z[:,0,1].imag,
        'Zyx_re': Z[:,1,0].real, 'Zyx_im': Z[:,1,0].imag,
        'Zyy_re': Z[:,1,1].real, 'Zyy_im': Z[:,1,1].imag,
    })
    if T is None:
        df['Tx_re'] = 0.0; df['Tx_im'] = 0.0
        df['Ty_re'] = 0.0; df['Ty_im'] = 0.0
    else:
        df['Tx_re'] = T[:,0,0].real; df['Tx_im'] = T[:,0,0].imag
        df['Ty_re'] = T[:,0,1].real; df['Ty_im'] = T[:,0,1].imag

    # Derived quantities
    period = 1.0 / np.asarray(freq, dtype=float)
    omega_mu = 2*np.pi*freq*MU0

    for lab, zr, zi in (
        ('xx', df['Zxx_re'].to_numpy(), df['Zxx_im'].to_numpy()),
        ('xy', df['Zxy_re'].to_numpy(), df['Zxy_im'].to_numpy()),
        ('yx', df['Zyx_re'].to_numpy(), df['Zyx_im'].to_numpy()),
        ('yy', df['Zyy_re'].to_numpy(), df['Zyy_im'].to_numpy()),
    ):
        Zc = zr + 1j*zi
        rho = np.abs(Zc)**2 / omega_mu
        phi = np.degrees(np.arctan2(zi, zr))
        df[f'rho_{{lab}}'] = rho
        df[f'phi_{{lab}}'] = phi

    # Phase Tensor
    PT = np.zeros((n,2,2), dtype=float)
    for i in range(n):
        X = np.array([[df['Zxx_re'].iat[i], df['Zxy_re'].iat[i]],
                      [df['Zyx_re'].iat[i], df['Zyy_re'].iat[i]]], dtype=float)
        Y = np.array([[df['Zxx_im'].iat[i], df['Zxy_im'].iat[i]],
                      [df['Zyx_im'].iat[i], df['Zyy_im'].iat[i]]], dtype=float)
        try:
            Xinv = np.linalg.inv(X)
        except np.linalg.LinAlgError:
            Xinv = np.linalg.pinv(X)
        PT[i] = Y @ Xinv
    df['ptxx_re'] = PT[:,0,0]; df['ptxy_re'] = PT[:,0,1]
    df['ptyx_re'] = PT[:,1,0]; df['ptyy_re'] = PT[:,1,1]

    return df


def plot_rho(df: pd.DataFrame, station: str, outdir: Path | str, comps: Optional[str] = None) -> str:
    """Plot apparent resistivity for selected Z components vs period (log scale).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame created by :func:`dataframe_from_arrays`.
    station : str
        Station identifier for titles/filenames.
    outdir : str or Path
        Output directory to write the PNG file.
    comps : str, optional
        Comma-separated components among 'xx,xy,yx,yy'. Default 'xy,yx'.

    Returns
    -------
    str
        Path of the saved PNG.
    """
    outdir = _ensure_plot_dir(outdir)
    comps_list = _parse_comps(comps)
    period = 1.0 / df['freq'].to_numpy()

    fig = plt.figure()
    for c in comps_list:
        plt.loglog(period, df[f'rho_{{c}}'].to_numpy(), marker='o', linestyle='-', label=f'\u03C1_a {{c.upper()}}')
    plt.gca().invert_xaxis()
    plt.xlabel('Period (s)')
    plt.ylabel('Apparent resistivity (\u03A9·m)')
    plt.title(f'{{station}} — Apparent Resistivity')
    plt.legend()
    out = Path(outdir) / f"{{station}}_rho.png"
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return str(out)


def plot_phase(df: pd.DataFrame, station: str, outdir: Path | str, comps: Optional[str] = None) -> str:
    """Plot impedance phase for selected components vs period (log scale).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame created by :func:`dataframe_from_arrays`.
    station : str
        Station identifier.
    outdir : str or Path
        Output directory to write the PNG file.
    comps : str, optional
        Comma-separated components among 'xx,xy,yx,yy'. Default 'xy,yx'.

    Returns
    -------
    str
        Path of the saved PNG.
    """
    outdir = _ensure_plot_dir(outdir)
    comps_list = _parse_comps(comps)
    period = 1.0 / df['freq'].to_numpy()

    fig = plt.figure()
    for c in comps_list:
        plt.semilogx(period, df[f'phi_{{c}}'].to_numpy(), marker='o', linestyle='-', label=f'\u03C6 {{c.upper()}}')
    plt.gca().invert_xaxis()
    plt.xlabel('Period (s)')
    plt.ylabel('Phase (deg)')
    plt.title(f'{{station}} — Impedance Phase')
    plt.legend()
    out = Path(outdir) / f"{{station}}_phase.png"
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return str(out)


def plot_tipper(df: pd.DataFrame, station: str, outdir: Path | str) -> str:
    """Plot real and imaginary parts of tipper (Tx, Ty) vs period.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame created by :func:`dataframe_from_arrays`.
    station : str
        Station identifier.
    outdir : str or Path
        Output directory to write the PNG file.

    Returns
    -------
    str
        Path of the saved PNG.
    """
    outdir = _ensure_plot_dir(outdir)
    period = 1.0 / df['freq'].to_numpy()
    tx_re = df['Tx_re'].to_numpy(); tx_im = df['Tx_im'].to_numpy()
    ty_re = df['Ty_re'].to_numpy(); ty_im = df['Ty_im'].to_numpy()

    fig = plt.figure()
    plt.semilogx(period, tx_re, marker='o', linestyle='-', label='Re(Tx)')
    plt.semilogx(period, tx_im, marker='^', linestyle='--', label='Im(Tx)')
    plt.semilogx(period, ty_re, marker='s', linestyle='-', label='Re(Ty)')
    plt.semilogx(period, ty_im, marker='d', linestyle='--', label='Im(Ty)')
    plt.gca().invert_xaxis()
    plt.xlabel('Period (s)')
    plt.ylabel('Tipper (dimensionless)')
    plt.title(f'{{station}} — Tipper')
    plt.legend()
    out = Path(outdir) / f"{{station}}_tipper.png"
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return str(out)


def plot_pt(df: pd.DataFrame, station: str, outdir: Path | str) -> str:
    """Plot Phase Tensor components (\u03A6) vs period.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame created by :func:`dataframe_from_arrays` (must include 'pt??_re').
    station : str
        Station identifier.
    outdir : str or Path
        Output directory to write the PNG file.

    Returns
    -------
    str
        Path of the saved PNG.
    """
    outdir = _ensure_plot_dir(outdir)
    period = 1.0 / df['freq'].to_numpy()
    ptxx = df['ptxx_re'].to_numpy()
    ptxy = df['ptxy_re'].to_numpy()
    ptyx = df['ptyx_re'].to_numpy()
    ptyy = df['ptyy_re'].to_numpy()

    fig = plt.figure()
    plt.semilogx(period, ptxx, marker='o', linestyle='-', label='PTxx')
    plt.semilogx(period, ptxy, marker='^', linestyle='-', label='PTxy')
    plt.semilogx(period, ptyx, marker='s', linestyle='-', label='PTyx')
    plt.semilogx(period, ptyy, marker='d', linestyle='-', label='PTyy')
    plt.gca().invert_xaxis()
    plt.xlabel('Period (s)')
    plt.ylabel('Phase Tensor entries (dimensionless)')
    plt.title(f'{{station}} — Phase Tensor \u03A6 components')
    plt.legend()
    out = Path(outdir) / f"{{station}}_PT_components.png"
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return str(out)
