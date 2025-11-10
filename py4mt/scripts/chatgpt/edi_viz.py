#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_edi_viz.py
=================
Plotting utilities for EDI transfer-function outputs produced by `edi_processor.py`.

Functions
---------
- _ensure_plot_dir(plot_dir)
- _parse_comps(comp_str)
- plot_rho(df, station, outdir, comps=None)
- plot_phase(df, station, outdir, comps=None)
- plot_tipper(df, station, outdir)
- plot_pt(df, station, outdir)

Notes
-----
- Uses matplotlib with the non-interactive Agg backend for headless environments.
- Plots follow the project constraint: matplotlib only, no seaborn, a single chart
  per call, and no explicit color specifications unless requested.

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-11-10 20:33:04 UTC
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

__all__ = [
    "_ensure_plot_dir",
    "_parse_comps",
    "plot_rho",
    "plot_phase",
    "plot_tipper",
    "plot_pt",
]


def _ensure_plot_dir(plot_dir: Path) -> Path:
    """Create the plot directory if needed and return an absolute path.

    Parameters
    ----------
    plot_dir : pathlib.Path
        Target folder for plots.

    Returns
    -------
    pathlib.Path
        Absolute, existing directory.
    """
    p = Path(plot_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def _parse_comps(comp_str: str) -> List[str]:
    """Parse and validate a comma-separated list of Z components.

    Parameters
    ----------
    comp_str : str
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


def plot_rho(df, station: str, outdir: Path, comps: str | Iterable[str] | None = None) -> None:
    """Plot apparent resistivity vs period for selected components; one PNG per component.

    Parameters
    ----------
    df : pandas.DataFrame
        Wide dataframe from `edi_processor.py` (must contain 'freq_Hz' and 'rho_*' columns).
    station : str
        Station name for titling and filenames.
    outdir : pathlib.Path
        Output folder for PNGs (created if needed).
    comps : str or iterable of str, optional
        Comma list or iterable subset of {'xx','xy','yx','yy'}.
    """
    if isinstance(comps, str):
        comps = _parse_comps(comps)
    elif comps is None:
        comps = ["xy", "yx"]
    period = 1.0 / df["freq_Hz"].to_numpy()
    outdir = _ensure_plot_dir(outdir)
    for c in comps:
        key = f"rho_{c}"
        if key not in df.columns:
            continue
        fig = plt.figure()
        plt.loglog(period, df[key].to_numpy(), marker="o", linestyle="-")
        plt.gca().invert_xaxis()
        plt.xlabel("Period (s)")
        plt.ylabel(f"Apparent resistivity ρ_{c} (Ω·m)")
        plt.title(f"{station} — ρ_{c}")
        fig.savefig(outdir / f"{station}_rho_{c}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_phase(df, station: str, outdir: Path, comps: str | Iterable[str] | None = None) -> None:
    """Plot phase (deg) vs period for selected components; one PNG per component.

    Parameters
    ----------
    df : pandas.DataFrame
        Wide dataframe containing 'freq_Hz' and 'phi_*_deg' columns.
    station : str
        Station name for titling and filenames.
    outdir : pathlib.Path
        Output folder for PNGs (created if needed).
    comps : str or iterable of str, optional
        Comma list or iterable subset of {'xx','xy','yx','yy'}.
    """
    if isinstance(comps, str):
        comps = _parse_comps(comps)
    elif comps is None:
        comps = ["xy", "yx"]
    period = 1.0 / df["freq_Hz"].to_numpy()
    outdir = _ensure_plot_dir(outdir)
    for c in comps:
        key = f"phi_{c}_deg"
        if key not in df.columns:
            continue
        fig = plt.figure()
        plt.semilogx(period, df[key].to_numpy(), marker="o", linestyle="-")
        plt.gca().invert_xaxis()
        plt.xlabel("Period (s)")
        plt.ylabel(f"Phase φ_{c} (deg)")
        plt.title(f"{station} — φ_{c}")
        fig.savefig(outdir / f"{station}_phi_{c}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_tipper(df, station: str, outdir: Path) -> None:
    """Plot tipper magnitude and argument vs period (two PNGs).

    Parameters
    ----------
    df : pandas.DataFrame
        Wide dataframe containing 'freq_Hz', and tipper columns 'tx_*', 'ty_*' if present.
    station : str
        Station name for titling and filenames.
    outdir : pathlib.Path
        Output folder for PNGs (created if needed).
    """
    outdir = _ensure_plot_dir(outdir)
    if "tx_re" not in df.columns or "ty_re" not in df.columns:
        return
    period = 1.0 / df["freq_Hz"].to_numpy()
    tx = (df["tx_re"].to_numpy() + 1j * df["tx_im"].to_numpy())
    ty = (df["ty_re"].to_numpy() + 1j * df["ty_im"].to_numpy())

    fig = plt.figure()
    plt.semilogx(period, np.abs(tx), marker="o", linestyle="-", label="|Tx|")
    plt.semilogx(period, np.abs(ty), marker="^", linestyle="-", label="|Ty|")
    plt.gca().invert_xaxis()
    plt.xlabel("Period (s)")
    plt.ylabel("Tipper magnitude")
    plt.title(f"{station} — |T|")
    plt.legend()
    fig.savefig(outdir / f"{station}_tipper_amp.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure()
    plt.semilogx(period, np.degrees(np.angle(tx)), marker="o", linestyle="-", label="arg(Tx)")
    plt.semilogx(period, np.degrees(np.angle(ty)), marker="^", linestyle="-", label="arg(Ty)")
    plt.gca().invert_xaxis()
    plt.xlabel("Period (s)")
    plt.ylabel("Tipper arg (deg)")
    plt.title(f"{station} — arg(T)")
    plt.legend()
    fig.savefig(outdir / f"{station}_tipper_arg.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pt(df, station: str, outdir: Path) -> None:
    """Plot Phase Tensor components (PTxx, PTxy, PTyx, PTyy) vs period as a single PNG.

    Parameters
    ----------
    df : pandas.DataFrame
        Wide dataframe containing 'freq_Hz' and PT entries 'ptxx_re', 'ptxy_re', 'ptyx_re', 'ptyy_re'.
    station : str
        Station name for titling and filenames.
    outdir : pathlib.Path
        Output folder for PNGs (created if needed).
    """
    outdir = _ensure_plot_dir(outdir)
    required = {"ptxx_re", "ptxy_re", "ptyx_re", "ptyy_re"}
    if not required.issubset(df.columns):
        return
    period = 1.0 / df["freq_Hz"].to_numpy()
    ptxx = df["ptxx_re"].to_numpy()
    ptxy = df["ptxy_re"].to_numpy()
    ptyx = df["ptyx_re"].to_numpy()
    ptyy = df["ptyy_re"].to_numpy()

    fig = plt.figure()
    plt.semilogx(period, ptxx, marker="o", linestyle="-", label="PTxx")
    plt.semilogx(period, ptxy, marker="^", linestyle="-", label="PTxy")
    plt.semilogx(period, ptyx, marker="s", linestyle="-", label="PTyx")
    plt.semilogx(period, ptyy, marker="d", linestyle="-", label="PTyy")
    plt.gca().invert_xaxis()
    plt.xlabel("Period (s)")
    plt.ylabel("Phase Tensor entries (dimensionless)")
    plt.title(f"{station} — Phase Tensor Φ components")
    plt.legend()
    fig.savefig(outdir / f"{station}_PT_components.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
