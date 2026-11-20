#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

data_viz.py
=================
Matplotlib helpers for magnetotelluric (MT) transfer-function plots.

This module is designed to work with :class:`pandas.DataFrame` objects
created from EDI data (for example using :func:`data_proc.dataframe_from_edi`).
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


datadict support
---------------

In addition to :class:`pandas.DataFrame`, the main plotters can accept an
"data_dict" (``Mapping[str, Any]``) with the common keys produced by the
EDI parsers in this project:

- ``freq`` : (nf,) frequency array [Hz] (required)
- ``Z`` : (nf, 2, 2) complex impedance tensor
- ``Z_err`` : (nf, 2, 2) complex or real 1-sigma errors (optional)
- ``T`` : (nf, 2) complex tipper (Tx, Ty)
- ``T_err`` : (nf, 2) complex or real 1-sigma errors (optional)
- ``P`` : (nf, 2, 2) real phase tensor (optional)
- ``P_err`` : (nf, 2, 2) real 1-sigma errors (optional)

When an data_dict is provided, it is converted internally into a plotting
DataFrame with the expected flat column names (``rho_xy``, ``phi_xy``,
``Tx_re``, ``ptxx_re``, ...). Metadata (station, location, rotation, etc.)
is preserved in ``df.attrs``.

Author: Volker Rath (DIAS)
Created with the help of ChatGPT (GPT-5 Thinking) on 2026-01-11
Modified: 2026-03-16 — add_rhoplus (D+/rho+ test plot), MT unit fix (mV/km/nT) for rho_a; Claude Sonnet 4.6 (Anthropic)
Modified: 2026-03-17 — xlim/ylim parameters for all plotters (add_rho, add_rhoplus, add_phase, add_tipper, add_pt); Claude Sonnet 4.6 (Anthropic)
Modified: 2026-03-30 — pop linestyle from **line_kw in add_rho/add_phase/add_tipper/add_pt (TypeError fix); add linestyle override to add_tipper/add_pt; remove bogus required-key guard from add_tipper/add_pt (checked "Z" key but tipper needs "T"; also broke DataFrame input); Claude Sonnet 4.6 (Anthropic)
Modified: 2026-04-12 — remove hardcoded marker="o" from add_rho, add_phase, add_rhoplus; pop marker from **line_kw (default None = no marker) so callers control markers without TypeError; add marker override to add_tipper and add_pt (uniform override, same pattern as existing linestyle override); Claude Sonnet 4.6 (Anthropic)
Modified: 2026-11-20 -- add_phase: where a phase curve leaves the +-180 deg window and re-enters on the other side, the solid line and the error band are no longer drawn straight through the jump (long vertical streak); the curve is broken there and the two points are connected by one dotted line (wrap_mode="direct", default) or by two short dotted stubs through the window edge (wrap_mode="short"). The style of the connection is a dict of Line2D keywords (wrap_style, module default PHASE_WRAP_STYLE; False = no connection; wrap_mode="off" = old behaviour). Marker-only calls (linestyle "none") get no connection. "ls" is accepted as alias of "linestyle". Claude Sonnet 5.5 (Anthropic); AI-generated, review before production use

Provenance (changes of 2026-11-20)
----------------------------------
Author    : Claude (Claude Sonnet 5.5, Anthropic)
Generated : 2026-11-20
Notice    : AI-generated code; review and test it before production use.
"""

from __future__ import annotations

from typing import Any, List, Optional
from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Physical constant
MU0 = 4.0e-7 * np.pi  # magnetic permeability of free space [H/m]

#: Default style of the connection drawn by add_phase where a phase curve
#: leaves the +-180 deg window and re-enters on the other side: a dict of
#: matplotlib Line2D keywords.  Whatever is not given here (or in the
#: ``wrap_style`` argument of add_phase) is taken from the solid curve
#: (colour, linewidth, alpha, zorder).  Change it for all later calls with
#: ``data_viz.PHASE_WRAP_STYLE.update(linestyle="--", alpha=0.5)``.
PHASE_WRAP_STYLE: dict = {"linestyle": ":"}


def _as_array(x: Any) -> Optional[np.ndarray]:
    """Convert a list/tuple/ndarray to ndarray; return None for unsupported types."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (list, tuple)):
        return np.asarray(x)
    return None


def _zcomp_name(i: int, j: int) -> str:
    """Map tensor indices to component labels ('xx','xy','yx','yy')."""
    return ("x" if i == 0 else "y") + ("x" if j == 0 else "y")


def _split_complex_1d(
    cols: dict[str, np.ndarray],
    base: str,
    arr: np.ndarray,
    *,
    suffix_re: str = "_re",
    suffix_im: str = "_im",
) -> None:
    """Store a 1D complex array as base_re and base_im."""
    cols[base + suffix_re] = np.asarray(arr.real)
    cols[base + suffix_im] = np.asarray(arr.imag)


def _complex_sigma_abs(err: np.ndarray) -> np.ndarray:
    """
    Convert an error array to an absolute sigma for |z|.

    - complex: sigma_abs = hypot(re, im)
    - real:    sigma_abs = abs(err)
    """
    err = np.asarray(err)
    if np.iscomplexobj(err):
        return np.hypot(err.real, err.imag)
    return np.abs(err)


def datadict_to_plot_df(datadict: Mapping[str, Any]) -> pd.DataFrame:
    """
    Convert an EDI dict (freq, Z, T, P plus optional *_err) to a plotting DataFrame.

    The returned DataFrame contains:

    - ``freq`` and ``period``
    - ``rho_<comp>`` and ``phi_<comp>`` derived from ``Z`` (if present)
    - tipper columns ``Tx_re``, ``Tx_im``, ``Ty_re``, ``Ty_im`` derived from ``T`` (if present)
    - phase tensor columns ``ptxx_re``, ``ptxy_re``, ``ptyx_re``, ``ptyy_re`` derived from ``P`` (if present)

    When ``Z_err`` is present and has compatible shape, approximate 1-sigma
    errors are added as ``rho_<comp>_err`` and ``phi_<comp>_err``.

    Metadata keys (station, location, rotation, etc.) are copied into ``df.attrs``.
    """
    freq = _as_array(datadict.get("freq"))
    if freq is None or np.asarray(freq).ndim != 1:
        raise ValueError("datadict['freq'] must be a 1D array.")
    freq = np.asarray(freq, dtype=float)
    n = int(freq.size)

    with np.errstate(divide="ignore", invalid="ignore"):
        period = np.where(freq > 0.0, 1.0 / freq, np.nan)
    omega = 2.0 * np.pi * freq

    cols: dict[str, np.ndarray] = {"freq": freq, "period": period}

    # Pass-through if already present as 1D arrays of length n
    for k, v in datadict.items():
        if k in cols:
            continue
        arr = _as_array(v)
        if arr is not None and np.asarray(arr).ndim == 1 and int(np.asarray(arr).size) == n:
            # do not overwrite derived columns later unless explicitly derived
            cols.setdefault(k, np.asarray(arr))

    # --- Z: accept (n,2,2) or (n,4)
    Z = _as_array(datadict.get("Z"))
    if Z is not None:
        Z = np.asarray(Z)
        if Z.shape[0] != n:
            raise ValueError(f"datadict['Z'] first dimension must be nf={n}; got {Z.shape}.")
        if Z.ndim == 2 and Z.shape[1] == 4:
            Z = Z.reshape(n, 2, 2)
        if Z.ndim != 3 or Z.shape[1:] != (2, 2):
            raise ValueError(f"datadict['Z'] expected shape (nf,2,2) or (nf,4); got {Z.shape}.")
        if not np.iscomplexobj(Z):
            Z = Z.astype(np.complex128)

        for i in range(2):
            for j in range(2):
                comp = _zcomp_name(i, j)
                z = Z[:, i, j]
                # Derived rho/phi (do not force if already present in cols)
                zabs = np.abs(z)
                rho = (zabs ** 2) * MU0 * 1.0e6 / omega
                phi = np.degrees(np.arctan2(z.imag, z.real))
                cols.setdefault(f"rho_{comp}", rho)
                cols.setdefault(f"phi_{comp}", phi)

        # optional error propagation
        Zerr = _as_array(datadict.get("Z_err"))
        if Zerr is not None:
            Zerr = np.asarray(Zerr)
            if Zerr.shape[0] == n:
                if Zerr.ndim == 2 and Zerr.shape[1] == 4:
                    Zerr = Zerr.reshape(n, 2, 2)
                if Zerr.ndim == 3 and Zerr.shape[1:] == (2, 2):
                    for i in range(2):
                        for j in range(2):
                            comp = _zcomp_name(i, j)
                            z = Z[:, i, j]
                            zabs = np.abs(z)
                            sig_abs = _complex_sigma_abs(Zerr[:, i, j])

                            cols.setdefault(
                                f"rho_{comp}_err",
                                (2.0 * zabs * sig_abs) * MU0 * 1.0e6 / omega,
                            )

                            with np.errstate(divide="ignore", invalid="ignore"):
                                sig_phi = (sig_abs / np.where(zabs == 0.0, np.nan, zabs)) * (
                                    180.0 / np.pi
                                )
                            cols.setdefault(f"phi_{comp}_err", sig_phi)

    # --- T: accept (n,2) or (n,2,1) or (n,1,2)
    T = _as_array(datadict.get("T"))
    if T is not None:
        T = np.asarray(T)
        if T.shape[0] != n:
            raise ValueError(f"datadict['T'] first dimension must be nf={n}; got {T.shape}.")
        if T.ndim == 3:
            if T.shape[1:] == (2, 1):
                T = T[:, :, 0]
            elif T.shape[1:] == (1, 2):
                T = T[:, 0, :]
        if T.ndim != 2 or T.shape[1] != 2:
            raise ValueError(f"datadict['T'] expected shape (nf,2); got {T.shape}.")
        if not np.iscomplexobj(T):
            T = T.astype(np.complex128)

        _split_complex_1d(cols, "Tx", T[:, 0])
        _split_complex_1d(cols, "Ty", T[:, 1])

        Terr = _as_array(datadict.get("T_err"))
        if Terr is not None:
            Terr = np.asarray(Terr)
            if Terr.shape[0] == n:
                if Terr.ndim == 3:
                    if Terr.shape[1:] == (2, 1):
                        Terr = Terr[:, :, 0]
                    elif Terr.shape[1:] == (1, 2):
                        Terr = Terr[:, 0, :]
                if Terr.ndim == 2 and Terr.shape[1] == 2:
                    if not np.iscomplexobj(Terr):
                        Terr = Terr.astype(np.complex128)
                    cols["Tx_re_err"] = np.asarray(Terr[:, 0].real)
                    cols["Tx_im_err"] = np.asarray(Terr[:, 0].imag)
                    cols["Ty_re_err"] = np.asarray(Terr[:, 1].real)
                    cols["Ty_im_err"] = np.asarray(Terr[:, 1].imag)

    # --- P: accept (n,2,2) or (n,4)
    P = _as_array(datadict.get("P"))
    if P is not None:
        P = np.asarray(P)
        if P.shape[0] != n:
            raise ValueError(f"datadict['P'] first dimension must be nf={n}; got {P.shape}.")
        if P.ndim == 2 and P.shape[1] == 4:
            P = P.reshape(n, 2, 2)
        if P.ndim != 3 or P.shape[1:] != (2, 2):
            raise ValueError(f"datadict['P'] expected shape (nf,2,2) or (nf,4); got {P.shape}.")
        cols.setdefault("ptxx_re", np.asarray(P[:, 0, 0], dtype=float))
        cols.setdefault("ptxy_re", np.asarray(P[:, 0, 1], dtype=float))
        cols.setdefault("ptyx_re", np.asarray(P[:, 1, 0], dtype=float))
        cols.setdefault("ptyy_re", np.asarray(P[:, 1, 1], dtype=float))

        Perr = _as_array(datadict.get("P_err"))
        if Perr is not None:
            Perr = np.asarray(Perr)
            if Perr.shape[0] == n:
                if Perr.ndim == 2 and Perr.shape[1] == 4:
                    Perr = Perr.reshape(n, 2, 2)
                if Perr.ndim == 3 and Perr.shape[1:] == (2, 2):
                    cols.setdefault("ptxx_re_err", np.asarray(Perr[:, 0, 0], dtype=float))
                    cols.setdefault("ptxy_re_err", np.asarray(Perr[:, 0, 1], dtype=float))
                    cols.setdefault("ptyx_re_err", np.asarray(Perr[:, 1, 0], dtype=float))
                    cols.setdefault("ptyy_re_err", np.asarray(Perr[:, 1, 1], dtype=float))

    df = pd.DataFrame(cols)

    # Copy common metadata into attrs
    for meta_k in (
        "rot",
        "err_kind",
        "header_raw",
        "source_kind",
        "station",
        "lat_deg",
        "lon_deg",
        "elev_m",
        "lat",
        "lon",
        "elev",
    ):
        if meta_k in datadict:
            v = datadict[meta_k]
            df.attrs[meta_k] = v if isinstance(v, (str, int, float, bool, type(None), np.number)) else repr(v)

    return df


def _ensure_df(data: pd.DataFrame | Mapping[str, Any]) -> pd.DataFrame:
    """Accept a DataFrame or EDI dict; always return a plotting DataFrame."""
    if isinstance(data, pd.DataFrame):
        return data
    return datadict_to_plot_df(data)


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
    data : pandas.DataFrame or Mapping[str, Any]
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
    breaks: Optional[np.ndarray] = None,
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
    breaks : array_like of int, optional
        Indices ``k`` after which the band is interrupted (between point
        ``k`` and ``k+1``), e.g. where a phase curve wraps around +-180 deg.
        The band is then drawn in separate pieces.
    """
    if err is None:
        return
    y = np.asarray(y, dtype=float)
    err = np.asarray(err, dtype=float)
    if y.shape != err.shape:
        return
    lower = y - err
    upper = y + err
    if breaks is not None and len(breaks) > 0:
        x = np.asarray(x, dtype=float)
        cut = np.asarray(breaks, dtype=int) + 1
        for xs, lo_, up_ in zip(np.split(x, cut), np.split(lower, cut),
                                np.split(upper, cut)):
            ax.fill_between(xs, lo_, up_, alpha=alpha, linewidth=0)
        return
    ax.fill_between(x, lower, upper, alpha=alpha, linewidth=0)


def _apply_limits(
    ax: plt.Axes,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
) -> None:
    """Apply optional x/y axis limits to *ax*.

    Both arguments accept a 2-tuple ``(lo, hi)``.  Either element may be
    ``None`` to leave that bound at the Matplotlib-computed default.
    No-op when the argument itself is ``None``.
    """
    if xlim is not None:
        lo, hi = xlim
        cur = list(ax.get_xlim())
        if lo is not None:
            cur[0] = lo
        if hi is not None:
            cur[1] = hi
        ax.set_xlim(cur)
    if ylim is not None:
        lo, hi = ylim
        cur = list(ax.get_ylim())
        if lo is not None:
            cur[0] = lo
        if hi is not None:
            cur[1] = hi
        ax.set_ylim(cur)


def add_rho(
    data: pd.DataFrame | Mapping[str, Any],
    comps: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    legend: bool = True,
    *,
    show_errors: bool = False,
    error_suffix: str = "_err",
    error_alpha: float = 0.25,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    **line_kw,
) -> plt.Axes:
    """Add apparent resistivity curves to an axes.

    Parameters
    ----------
    data : pandas.DataFrame or Mapping[str, Any]
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
    df = _ensure_df(data)
    fig, ax, _ = _maybe_ax(ax)
    period = _period_from_df(df)
    comps_list = _parse_comps(comps)

    # Allow callers to override linestyle and marker via **line_kw without
    # collision.  Default linestyle is "-"; default marker is None (no marker)
    # so that upstream callers (e.g. femtic_viz.plot_data_ensemble) can apply
    # per-component markers via post-hoc Line2D patching.
    _ls = line_kw.pop("linestyle", "-")
    _mk = line_kw.pop("marker", None)

    for c in comps_list:
        col = f"rho_{c}"
        if col not in df.columns:
            continue
        y = df[col].to_numpy()
        _plot_kw = dict(linestyle=_ls, label=f"$\\rho_a,{c.upper()}$", **line_kw)
        if _mk is not None:
            _plot_kw["marker"] = _mk
        ax.loglog(period, y, **_plot_kw)
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
    _apply_limits(ax, xlim=xlim, ylim=ylim)
    return ax


def add_rhoplus(
    data: pd.DataFrame | Mapping[str, Any],
    comps: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    legend: bool = True,
    *,
    n_lambda_per_freq: int = 4,
    violation_marker: str = "x",
    violation_color: str = "red",
    violation_size: float = 120.0,
    show_errors: bool = False,
    error_suffix: str = "_err",
    error_alpha: float = 0.25,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    **line_kw,
) -> plt.Axes:
    """Add D⁺ / rho-plus curves and violation markers to a log–log axes.

    For each requested impedance component the function plots:

    * the measured apparent resistivity ρ_a (solid line with markers), and
    * the D⁺ upper bound ρ⁺ (dashed line, same colour), computed via
      :func:`data_proc.compute_rhoplus`.

    Periods where ρ_a > ρ⁺ (i.e. the 1-D consistency test fails) are
    highlighted with a scatter overlay of crosses in ``violation_color``.

    The D⁺ test is the necessary and sufficient condition for a scalar
    impedance dataset to be compatible with *any* 1-D conductivity profile
    (Parker 1980; Parker & Whaler 1981). A violation at one or more periods
    proves that a 1-D model cannot explain the data.  The test is also known
    as **dplus** in Cordell's mtcode
    (https://github.com/darcycordell/mtcode).

    Parameters
    ----------
    data : pandas.DataFrame or Mapping[str, Any]
        Either a tidy plotting DataFrame (from
        :func:`data_proc.dataframe_from_edi`) or a raw EDI dict with at
        least keys ``"freq"`` and ``"Z"``.  When a raw dict is passed the
        conversion is done internally.
    comps : str, optional
        Comma-separated impedance components to test, e.g. ``"xy,yx"``.
        Defaults to ``"xy,yx"``.
    ax : matplotlib.axes.Axes, optional
        Target axes.  If None a new figure/axes pair is created.
    legend : bool, optional
        Whether to show a legend (default ``True``).
    n_lambda_per_freq : int, optional
        Density of the D⁺ λ grid; passed to
        :func:`data_proc.compute_rhoplus`. Default is 4.
    violation_marker : str, optional
        Matplotlib marker style for violation points. Default ``"x"``.
    violation_color : str, optional
        Colour for violation markers. Default ``"red"``.
    violation_size : float, optional
        Size of violation markers. Default 120.
    show_errors : bool, optional
        If True and an ``"Z_err"`` / ``rho_<comp>_err`` column is present,
        draw a ±1σ envelope around ρ_a.
    error_suffix : str, optional
        Column-name suffix for error columns. Default ``"_err"``.
    error_alpha : float, optional
        Opacity of the error envelopes (0–1). Default 0.25.
    **line_kw :
        Extra keyword arguments forwarded to ``ax.loglog`` for the ρ_a
        line; the ρ⁺ line uses the same keywords with ``linestyle="--"``
        and no marker.

    Returns
    -------
    matplotlib.axes.Axes
        The axes instance with the curves added.

    Notes
    -----
    :func:`data_proc.compute_rhoplus` must be importable. The function
    lazily imports it so that ``data_viz`` does not hard-depend on
    ``data_proc`` at module load time.

    References
    ----------
    Parker, R. L. (1980). The inverse problem of electromagnetic induction:
    existence and construction of solutions based on incomplete data.
    *J. Geophys. Res.*, 85, 4421–4428.

    Parker, R. L., & Whaler, K. A. (1981). Numerical methods for establishing
    solutions to the inverse problem of electromagnetic induction.
    *J. Geophys. Res.*, 86, 9574–9584.

    Parker, R. L., & Booker, J. R. (1996). Optimal one-dimensional inversion
    and bounding of magnetotelluric apparent resistivity and phase
    measurements. *Phys. Earth Planet. Inter.*, 98, 269–282.

    Cordell, D., Lee, B., & Unsworth, M. J. (2022). mtcode.
    doi:10.5281/zenodo.6784201 https://github.com/darcycordell/mtcode

    Examples
    --------
    >>> import data_proc, data_viz, matplotlib.pyplot as plt
    >>> site = data_proc.load_edi("SITE.edi")
    >>> ax = data_viz.add_rhoplus(site, comps="xy,yx")
    >>> plt.show()
    """
    try:
        from data_proc import compute_rhoplus as _compute_rhoplus
    except ImportError as exc:
        raise ImportError(
            "add_rhoplus requires data_proc.compute_rhoplus. "
            "Make sure data_proc.py is on the Python path."
        ) from exc

    # Accept raw EDI dict or DataFrame
    raw_dict: Optional[Mapping[str, Any]] = None
    if not isinstance(data, pd.DataFrame):
        raw_dict = data  # keep the original dict for Z/Z_err access
    df = _ensure_df(data)

    fig, ax, _ = _maybe_ax(ax)
    period = _period_from_df(df)
    comps_list = _parse_comps(comps)

    # Pop marker from line_kw so callers can supply it (or leave it absent for
    # no marker).  The ρ⁺ line always uses no marker regardless.
    _mk = line_kw.pop("marker", None)

    # Colour cycle — reuse current axes prop_cycle so colours align with
    # any previously plotted curves on the same axes.
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colours = [p["color"] for p in prop_cycle]

    for idx, c in enumerate(comps_list):
        colour = colours[idx % len(colours)]

        # --- retrieve Z_scalar and optional Z_scalar_err ---
        freq_arr = df["freq"].to_numpy()

        # Prefer raw array from the dict when available (avoids rounding in df)
        if raw_dict is not None and "Z" in raw_dict:
            Z = np.asarray(raw_dict["Z"], dtype=np.complex128)
            comp_map = {"xx": (0, 0), "xy": (0, 1), "yx": (1, 0), "yy": (1, 1)}
            i, j = comp_map[c]
            z_scalar = Z[:, i, j]
            z_err = None
            if raw_dict.get("Z_err") is not None:
                Ze = np.asarray(raw_dict["Z_err"], dtype=float)
                if Ze.shape == Z.shape:
                    z_err = Ze[:, i, j]
        else:
            # Reconstruct from DataFrame columns (Re+Im or rho/phi — use rho)
            # We need the complex Z; fall back to rho_a-only mode (no Z_err)
            rho_col = f"rho_{c}"
            phi_col = f"phi_{c}"
            if rho_col not in df.columns:
                continue
            rho = df[rho_col].to_numpy()
            phi = df[phi_col].to_numpy() if phi_col in df.columns else np.full_like(rho, 45.0)
            omega = 2.0 * np.pi * freq_arr
            absZ = np.sqrt(rho * omega / (MU0 * 1.0e6))
            z_scalar = absZ * np.exp(1j * np.radians(phi))
            z_err = None
            err_col = f"{rho_col}{error_suffix}"
            if show_errors and err_col in df.columns:
                # crude sigma_Z from sigma_rho: sigma_Z ~ sigma_rho * mu0*omega / (2*absZ)
                sig_rho = df[err_col].to_numpy()
                with np.errstate(invalid="ignore"):
                    z_err = sig_rho * omega / (2.0 * np.where(absZ > 0, absZ, np.nan) * MU0 * 1.0e6)

        # --- run D+ ---
        rho_plus, rho_a, pass_test = _compute_rhoplus(
            freq_arr,
            z_scalar,
            z_err if show_errors else None,
            n_lambda_per_freq=n_lambda_per_freq,
        )

        # compute_rhoplus sorts by freq internally; period here is already
        # in the same order as df, so sort rho_a/rho_plus back to df order
        sort_idx = np.argsort(freq_arr)
        unsort = np.argsort(sort_idx)
        rho_plus = rho_plus[unsort]
        rho_a_plot = rho_a[unsort]
        pass_plot = pass_test[unsort]

        label_a = f"$\\rho_a^{{\\,{c.upper()}}}$"
        label_p = f"$\\rho^+_{{{c.upper()}}}$"

        # ρ_a — solid line
        _rha_kw = dict(linestyle="-", color=colour, label=label_a, **line_kw)
        if _mk is not None:
            _rha_kw["marker"] = _mk
        ax.loglog(period, rho_a_plot, **_rha_kw)

        # ρ⁺ — dashed, same colour, no marker
        lkw = {k: v for k, v in line_kw.items() if k not in ("marker", "linestyle")}
        ax.loglog(
            period,
            rho_plus,
            marker="",
            linestyle="--",
            color=colour,
            label=label_p,
            **lkw,
        )

        # violation markers
        viol_mask = ~pass_plot
        if viol_mask.any():
            ax.scatter(
                period[viol_mask],
                rho_a_plot[viol_mask],
                marker=violation_marker,
                color=violation_color,
                s=violation_size,
                zorder=5,
                label=f"violation ({c.upper()})" if idx == 0 else "_nolegend_",
            )

        # optional ρ_a error envelope
        if show_errors:
            err_col = f"rho_{c}{error_suffix}"
            if err_col in df.columns:
                err = df[err_col].to_numpy()
                _maybe_fill_between(ax, period, rho_a_plot, err, alpha=error_alpha)

    ax.invert_xaxis()
    ax.set_xlabel("Period (s)")
    ax.set_ylabel("Apparent resistivity ($\\Omega\\,\\mathrm{m}$)")
    ax.grid(True, linestyle=":")
    if legend:
        ax.legend()
    _apply_limits(ax, xlim=xlim, ylim=ylim)
    return ax



def _phase_wrap_index(x: np.ndarray, y: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Indices ``k`` where a phase curve wraps between point ``k`` and ``k+1``.

    A wrap is a jump of more than half the window width ``hi - lo`` (180 deg
    for the usual +-180 deg window) between two neighbouring points in the
    order in which they are plotted.  Points which are not finite (or whose
    period is not positive) are never part of a wrap: the line is
    interrupted there by matplotlib anyway.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.size < 2:
        return np.zeros(0, dtype=int)
    ok = np.isfinite(x) & (x > 0.0) & np.isfinite(y)
    with np.errstate(invalid="ignore"):
        jump = np.abs(np.diff(y)) > 0.5 * (hi - lo)
    return np.flatnonzero(jump & ok[:-1] & ok[1:])


def _phase_wrap_segments(
    x: np.ndarray, y: np.ndarray, k: np.ndarray, lo: float, hi: float,
    mode: str = "short",
) -> tuple[np.ndarray, np.ndarray]:
    """Connection lines for the wraps ``k``, as one NaN-separated polyline.

    ``mode="short"``: the curve is continued the short way, through the
    window edge: from point ``k`` to the edge it leaves at (``hi`` for a jump
    downwards, ``lo`` for one upwards) and from the opposite edge to point
    ``k+1``.  Both stubs meet at the same period, which is found by linear
    interpolation in log(period) of the unwrapped curve.
    ``mode="direct"``: one straight line from point ``k`` to point ``k+1``.
    """
    width = hi - lo
    xs: list = []
    ys: list = []
    for i in np.asarray(k, dtype=int):
        x0, x1, y0, y1 = float(x[i]), float(x[i + 1]), float(y[i]), float(y[i + 1])
        if mode == "direct":
            xs += [x0, x1, np.nan]
            ys += [y0, y1, np.nan]
            continue
        if y1 < y0:                                  # leaves at the top
            edge_out, edge_in, y1u = hi, lo, y1 + width
        else:                                        # leaves at the bottom
            edge_out, edge_in, y1u = lo, hi, y1 - width
        den = y1u - y0
        f = 0.5 if den == 0.0 else float(np.clip((edge_out - y0) / den, 0.0, 1.0))
        xc = float(np.exp(np.log(x0) + f * (np.log(x1) - np.log(x0))))
        xs += [x0, xc, np.nan, xc, x1, np.nan]
        ys += [y0, edge_out, np.nan, edge_in, y1, np.nan]
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _linestyle_is_none(ls: Any) -> bool:
    """True if a matplotlib linestyle means "no line"."""
    return isinstance(ls, str) and ls.strip().lower() in ("none", "")


def add_phase(
    data: pd.DataFrame | Mapping[str, Any],
    comps: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    legend: bool = True,
    *,
    show_errors: bool = False,
    error_suffix: str = "_err",
    error_alpha: float = 0.25,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    wrap_mode: str = "direct",
    wrap_style: Optional[Mapping[str, Any] | bool] = None,
    wrap_range: tuple = (-180.0, 180.0),
    **line_kw,
) -> plt.Axes:
    """Add impedance phase curves to an axes.

    Parameters
    ----------
    data : pandas.DataFrame or Mapping[str, Any]
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
    wrap_mode : {"short", "direct", "off"}, optional
        What to do where the phase of a curve leaves the window ``wrap_range``
        (+-180 deg) and re-enters on the other side, i.e. where two
        neighbouring points differ by more than half the window width.
        ``"direct"`` (default): the solid line and the error band are
        interrupted, and the two points are joined by one straight line in
        ``wrap_style`` (dotted by default).
        ``"short"``: interrupted as above, and the curve is continued the
        short way through the window edge with two short stubs in
        ``wrap_style``; much less ink if the phase jitters around +-180 deg.
        ``"off"``: nothing special, the line runs straight through the jump.
        Curves without a line (``linestyle="none"``, markers only) are never
        connected.
    wrap_style : dict or False, optional
        Matplotlib Line2D keywords of the connection, e.g.
        ``{"linestyle": (0, (1, 3)), "alpha": 0.5, "color": "0.4"}``.  They are
        applied on top of the module default ``PHASE_WRAP_STYLE`` (dotted),
        so only the differences need to be given.  Colour, linewidth, alpha
        and zorder not given are those of the solid curve.  ``False`` draws no
        connection (the curve is just interrupted).  None (default) = the
        module default.
    wrap_range : (float, float), optional
        Lower and upper edge of the phase window; default ``(-180, 180)``.
        Use ``(0, 360)`` for phases given in that range.
    **line_kw :
        Additional keyword arguments forwarded to ``ax.semilogx``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes instance with the curves added.

    Notes
    -----
    A jump of more than half the window is taken as a wrap, never as a real
    change of the phase through the middle of the window (the usual phase
    unwrapping assumption).  The error band is interrupted at a wrap, but not
    continued on the other side of the window edge.
    """
    df = _ensure_df(data)
    fig, ax, _ = _maybe_ax(ax)
    period = _period_from_df(df)
    comps_list = _parse_comps(comps)

    # Allow callers to override linestyle and marker via **line_kw without
    # collision.  Default linestyle is "-"; default marker is None (no marker).
    _ls_alias = line_kw.pop("ls", None)
    _ls = line_kw.pop("linestyle", "-" if _ls_alias is None else _ls_alias)
    _mk = line_kw.pop("marker", None)

    _wmode = str(wrap_mode).lower()
    if _wmode not in ("short", "direct", "off"):
        raise ValueError(
            f"add_phase: wrap_mode must be 'short', 'direct' or 'off', got {wrap_mode!r}.")
    _lo, _hi = float(wrap_range[0]), float(wrap_range[1])
    if not _hi > _lo:
        raise ValueError(f"add_phase: wrap_range must be (lo, hi) with hi > lo, got {wrap_range!r}.")
    if wrap_style is False:
        _wstyle = None
    elif wrap_style is None or wrap_style is True:
        _wstyle = dict(PHASE_WRAP_STYLE)
    else:
        _wstyle = {**PHASE_WRAP_STYLE, **dict(wrap_style)}
    _lw = line_kw.get("linewidth", line_kw.get("lw"))
    _has_line = not _linestyle_is_none(_ls) and not (_lw is not None and float(_lw) == 0.0)

    for c in comps_list:
        col = f"phi_{c}"
        if col not in df.columns:
            continue
        y = df[col].to_numpy()
        kwrap = (_phase_wrap_index(period, y, _lo, _hi) if _wmode != "off"
                 else np.zeros(0, dtype=int))
        xp, yp = period, y
        if kwrap.size and _has_line:
            # interrupt the solid line at the wraps (NaN = gap)
            xp = np.insert(np.asarray(period, dtype=float), kwrap + 1, np.nan)
            yp = np.insert(np.asarray(y, dtype=float), kwrap + 1, np.nan)
        _plot_kw = dict(linestyle=_ls, label=f"$\\phi,{c.upper()}$", **line_kw)
        if _mk is not None:
            _plot_kw["marker"] = _mk
        main = ax.semilogx(xp, yp, **_plot_kw)[0]
        if kwrap.size and _has_line and _wstyle is not None:
            xs, ys = _phase_wrap_segments(period, y, kwrap, _lo, _hi, mode=_wmode)
            ax.plot(xs, ys, **{
                "color": main.get_color(), "linewidth": main.get_linewidth(),
                "alpha": main.get_alpha(), "zorder": main.get_zorder(),
                **_wstyle, "label": "_nolegend_"})
        if show_errors:
            err_col = f"{col}{error_suffix}"
            if err_col in df.columns:
                err = df[err_col].to_numpy()
                _maybe_fill_between(ax, period, y, err, alpha=error_alpha,
                                    breaks=kwrap if kwrap.size else None)

    ax.invert_xaxis()
    ax.set_xlabel("Period (s)")
    ax.set_ylabel("Phase $\\phi$ (deg)")
    ax.grid(True, linestyle=":")
    if legend:
        ax.legend()
    _apply_limits(ax, xlim=xlim, ylim=ylim)
    return ax


def add_tipper(
    data: pd.DataFrame | Mapping[str, Any],
    ax: Optional[plt.Axes] = None,
    legend: bool = True,
    *,
    show_errors: bool = False,
    error_suffix: str = "_err",
    error_alpha: float = 0.25,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    **line_kw,
) -> plt.Axes:
    """Add tipper components (Tx, Ty) to an axes.

    Parameters
    ----------
    data : pandas.DataFrame or Mapping[str, Any]
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

    df = _ensure_df(data)

    fig, ax, _ = _maybe_ax(ax)
    period = _period_from_df(df)

    # If the caller supplies linestyle= or marker= (e.g. for perturbed ensemble
    # curves), apply them uniformly to all components instead of per-component
    # defaults.  Pop here to avoid a duplicate-keyword error.
    _ls_override = line_kw.pop("linestyle", None)
    _mk_override = line_kw.pop("marker", None)

    specs = [
        ("Tx_re", "o", "-",  "Re(Tx)"),
        ("Tx_im", "^", "--", "Im(Tx)"),
        ("Ty_re", "s", "-",  "Re(Ty)"),
        ("Ty_im", "d", "--", "Im(Ty)"),
    ]

    for col, marker, linestyle, label in specs:
        if col not in df.columns:
            continue
        y = df[col].to_numpy()
        ax.semilogx(
            period,
            y,
            marker=_mk_override if _mk_override is not None else marker,
            linestyle=_ls_override if _ls_override is not None else linestyle,
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
    _apply_limits(ax, xlim=xlim, ylim=ylim)
    return ax


def add_pt(
    data: pd.DataFrame | Mapping[str, Any],
    ax: Optional[plt.Axes] = None,
    legend: bool = True,
    *,
    show_errors: bool = False,
    error_suffix: str = "_err",
    error_alpha: float = 0.25,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
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
    df = _ensure_df(data)

    fig, ax, _ = _maybe_ax(ax)
    period = _period_from_df(df)

    # Allow a uniform linestyle or marker override via **line_kw (e.g. "--"
    # or a single marker symbol for perturbed ensemble curves) without
    # colliding with per-component defaults.
    _ls_override = line_kw.pop("linestyle", None)
    _mk_override = line_kw.pop("marker", None)

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
            marker=_mk_override if _mk_override is not None else marker,
            linestyle=_ls_override if _ls_override is not None else linestyle,
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
    _apply_limits(ax, xlim=xlim, ylim=ylim)
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
        from data_viz import plot_gridx, plot_grid

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
        from data_viz import plot_gridx, plot_grid

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
