#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_crossplot.py

Crossplots (scatter) of observed vs. calculated data from a FEMTIC
results_iterN.h5 file, one panel per data type and component, with the
log10 frequency encoded as point colour.

@author   vrath
@project  py4mt
@created  2026-09-18
@modified 2026-09-18

Provenance
----------
Author      : Claude Sonnet 5 (Anthropic)
Generated   : 2026-09-18
Notice      : This code is AI-generated. Review and test it before any
              production use.

What it does
------------
- Reads the /data group of results_iterN.h5 via
  femtic.read_results_data() (fields re_val/im_val = observed,
  cal_re/cal_im = calculated, re_err/im_err = observed errors).
- Groups rows by datatype_name and component_name (see
  femtic.RESULTS_H5_DATATYPE_NAMES / RESULTS_H5_COMPONENT_NAMES).
- Complex-valued data types (MT, HTF, VTF, NMT, NMT2, ...) get one Re and
  one Im panel per component; real-valued types (APP_RES_AND_PHS, PT, ...)
  get a single panel per component.
- One figure per data type is written to OUT_DIR, in every format listed
  in PLOT_FORMATS. A statistics table (n, normalised RMS per panel) is
  printed and written to OUT_DIR as well.

Usage
-----
    python femtic_crossplot.py                      # uses RESULTS_FILES below
    python femtic_crossplot.py results_iter12.h5    # file(s) from command line

Notes
-----
- Rows whose observed or calculated value (or frequency) is not finite are
  dropped per panel; the number dropped is reported.
- Apparent-resistivity components (names starting with "rho") are drawn on
  log-log axes if LOG_RHO is True and all values are positive.
- nRMS = sqrt(mean(((obs - calc) / err)**2)) over rows with finite,
  positive error.

Changelog
---------
2026-09-18  Claude Sonnet 5 (Anthropic): initial version.
"""
from __future__ import annotations

import math
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# py4mt environment: make the modules importable (femtic.py, ensembles.py)
# ---------------------------------------------------------------------------
PY4MTX_ROOT = os.environ.get("PY4MTX_ROOT", "")
if PY4MTX_ROOT:
    for _pth in (os.path.join(PY4MTX_ROOT, "py4mt", "modules"),
                 os.path.join(PY4MTX_ROOT, "py4mt", "scripts")):
        if os.path.isdir(_pth) and _pth not in sys.path:
            sys.path.insert(0, _pth)

import femtic as fem  # noqa: E402

# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------
RESULTS_FILES: List[str] = ["results_iter10.h5"]   # overridden by argv[1:]
OUT_DIR: str = "./crossplots/"
OUT_PREFIX: Optional[str] = None      # None -> stem of the results file
PLOT_FORMATS: Tuple[str, ...] = (".png", ".pdf")
PLOT_DPI: int = 300

# --- selection (None = everything) ---
DATATYPES: Optional[Sequence[str]] = None     # e.g. ["MT", "VTF"]
COMPONENTS: Optional[Sequence[str]] = None    # e.g. ["Zxy", "Zyx"]

# --- layout ---
NCOLS: int = 4                # panels per row
PANEL_SIZE: float = 3.3       # inches per panel edge
EQUAL_ASPECT: bool = True

# --- appearance ---
CMAP: str = "turbo"
COLOR_RANGE: str = "global"   # "global": same log10(f) range on all figures
                              # "panel": each panel scaled to its own data
MARKER_SIZE: float = 14.0
ALPHA: float = 0.85
LOG_RHO: bool = True          # log-log axes for rho* components
SHOW_ERRORBARS: bool = False  # horizontal error bars (observed errors)
SHOW: bool = False            # interactive display after saving
WRITE_STATS: bool = True


# ---------------------------------------------------------------------------
# Library-style helpers (stateless, explicit parameters)
# ---------------------------------------------------------------------------
REAL_ONLY_TYPES = ("APP_RES_AND_PHS", "NMT2_APP_RES_AND_PHS", "PT")


def _safe_name(s: str) -> str:
    """Return s with everything but [A-Za-z0-9_.-] replaced by '_'."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def build_panels(
    data: dict,
    *,
    datatypes: Optional[Sequence[str]] = None,
    components: Optional[Sequence[str]] = None,
) -> Dict[str, List[dict]]:
    """
    Split the /data rows of a results file into crossplot panels.

    Parameters
    ----------
    data : dict
        Return value of femtic.read_results_data() (with decode_labels=True).
    datatypes, components : sequence of str, optional
        Restrict to these data type / component names.

    Returns
    -------
    dict
        {datatype_name: [panel, ...]}, each panel a dict with keys
        datatype, component, part ("re"/"im"/"real"), label, freq, obs,
        cal, err, n_dropped.
    """
    rows = data["rows"]
    if rows.size == 0:
        raise ValueError("build_panels: results file contains no data rows.")
    if "datatype_name" not in data:
        raise KeyError("build_panels: decoded labels missing "
                       "(read_results_data(decode_labels=True) required).")

    dt_all = data["datatype_name"]
    cn_all = data["component_name"]

    # data types in order of first appearance
    _, first = np.unique(dt_all, return_index=True)
    dtype_order = [dt_all[i] for i in np.sort(first)]

    panels: Dict[str, List[dict]] = {}
    for dname in dtype_order:
        if datatypes is not None and dname not in datatypes:
            continue
        m_dt = dt_all == dname
        codes = np.unique(rows["component"][m_dt])
        for code in codes:
            m = m_dt & (rows["component"] == code)
            cname = str(cn_all[m][0])
            if components is not None and cname not in components:
                continue
            sub = rows[m]
            freq = np.asarray(sub["freq"], dtype=float)

            parts = [("re", sub["re_val"], sub["cal_re"], sub["re_err"])]
            im_obs = np.asarray(sub["im_val"], dtype=float)
            im_cal = np.asarray(sub["cal_im"], dtype=float)
            has_im = (dname not in REAL_ONLY_TYPES) and (
                np.any(np.nan_to_num(im_obs) != 0.0)
                or np.any(np.nan_to_num(im_cal) != 0.0)
            )
            if has_im:
                parts.append(("im", sub["im_val"], sub["cal_im"], sub["im_err"]))

            for part, obs, cal, err in parts:
                obs = np.asarray(obs, dtype=float)
                cal = np.asarray(cal, dtype=float)
                err = np.asarray(err, dtype=float)
                ok = (np.isfinite(obs) & np.isfinite(cal)
                      & np.isfinite(freq) & (freq > 0.0))
                if not has_im:
                    label = cname
                    part_tag = "real"
                else:
                    label = f"{'Re' if part == 're' else 'Im'} {cname}"
                    part_tag = part
                panels.setdefault(dname, []).append(dict(
                    datatype=dname,
                    component=cname,
                    part=part_tag,
                    label=label,
                    freq=freq[ok],
                    obs=obs[ok],
                    cal=cal[ok],
                    err=err[ok],
                    n_dropped=int(np.count_nonzero(~ok)),
                ))
    return panels


def panel_stats(panel: dict) -> dict:
    """Return n, nRMS (error-normalised) and plain RMS for one panel."""
    obs, cal, err = panel["obs"], panel["cal"], panel["err"]
    n = obs.size
    res = obs - cal
    rms = float(np.sqrt(np.mean(res**2))) if n else float("nan")
    ok = np.isfinite(err) & (err > 0.0)
    nrms = (float(np.sqrt(np.mean((res[ok] / err[ok]) ** 2)))
            if np.any(ok) else float("nan"))
    return dict(n=n, n_dropped=panel["n_dropped"], rms=rms, nrms=nrms)


def _axis_limits(x: np.ndarray, y: np.ndarray, *, log: bool) -> Tuple[float, float]:
    """Common (square) axis limits with a small margin."""
    v = np.concatenate([x, y])
    if log:
        lo, hi = float(np.min(v)), float(np.max(v))
        if lo == hi:
            return lo / 2.0, hi * 2.0
        f = (hi / lo) ** 0.05
        return lo / f, hi * f
    lo, hi = float(np.min(v)), float(np.max(v))
    if lo == hi:
        d = abs(lo) * 0.1 if lo != 0.0 else 1.0
        return lo - d, hi + d
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def plot_datatype(
    dname: str,
    panels: Sequence[dict],
    *,
    iter_num: int,
    ncols: int,
    panel_size: float,
    cmap: str,
    color_range: Optional[Tuple[float, float]],
    marker_size: float,
    alpha: float,
    log_rho: bool,
    show_errorbars: bool,
    equal_aspect: bool,
):
    """
    Draw one figure holding all panels of a data type.

    Parameters
    ----------
    color_range : (vmin, vmax) of log10(f), or None to scale per panel
        (in the latter case the colourbar follows the first panel).

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    n = len(panels)
    nc = max(1, min(ncols, n))
    nr = int(math.ceil(n / nc))
    fig, axes = plt.subplots(
        nr, nc, figsize=(panel_size * nc + 1.2, panel_size * nr + 0.6),
        squeeze=False, constrained_layout=True,
    )
    axes_flat = axes.ravel()

    sm_norm = None
    for k, p in enumerate(panels):
        ax = axes_flat[k]
        x, y, err = p["obs"], p["cal"], p["err"]
        if x.size == 0:
            ax.text(0.5, 0.5, "no valid data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(p["label"], fontsize=10)
            continue

        logf = np.log10(p["freq"])
        if color_range is not None:
            norm = Normalize(vmin=color_range[0], vmax=color_range[1])
        else:
            norm = Normalize(vmin=float(logf.min()), vmax=float(logf.max()))
        if sm_norm is None:
            sm_norm = norm

        use_log = (log_rho and p["component"].lower().startswith("rho")
                   and bool(np.all(x > 0.0)) and bool(np.all(y > 0.0)))

        if show_errorbars:
            xerr = np.where(np.isfinite(err) & (err > 0.0), err, 0.0)
            ax.errorbar(x, y, xerr=xerr, fmt="none", ecolor="0.75",
                        elinewidth=0.5, zorder=1)
        ax.scatter(x, y, c=logf, cmap=cmap, norm=norm, s=marker_size,
                   alpha=alpha, edgecolors="none", zorder=2)

        if use_log:
            ax.set_xscale("log")
            ax.set_yscale("log")
        lo, hi = _axis_limits(x, y, log=use_log)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.plot([lo, hi], [lo, hi], color="k", lw=0.8, zorder=3)
        if equal_aspect:
            ax.set_aspect("equal", adjustable="box")

        st = panel_stats(p)
        ax.text(0.04, 0.96, f"n={st['n']}\nnRMS={st['nrms']:.2f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.8))
        ax.set_title(p["label"], fontsize=10)
        ax.set_xlabel("observed", fontsize=9)
        ax.set_ylabel("calculated", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, which="major", lw=0.3, alpha=0.5)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    if sm_norm is not None:
        sm = cm.ScalarMappable(norm=sm_norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=list(axes_flat[:n]), shrink=0.85, pad=0.02)
        cb.set_label("log10 frequency (Hz)")

    fig.suptitle(f"{dname}: observed vs. calculated, iteration {iter_num}",
                 fontsize=12)
    return fig


def global_logf_range(panels: Dict[str, List[dict]]) -> Optional[Tuple[float, float]]:
    """Min/max of log10(f) over every panel, or None if there is no data."""
    fs = [p["freq"] for lst in panels.values() for p in lst if p["freq"].size]
    if not fs:
        return None
    f = np.concatenate(fs)
    return float(np.log10(f.min())), float(np.log10(f.max()))


def format_stats_table(panels: Dict[str, List[dict]]) -> str:
    """Plain-ASCII statistics table for all panels."""
    hdr = (f"{'datatype':<22} {'component':<8} {'part':<5} "
           f"{'n':>7} {'dropped':>8} {'rms':>12} {'nrms':>8}")
    sep = "-" * len(hdr)
    lines = ["=" * len(hdr), hdr, sep]
    for dname, lst in panels.items():
        for p in lst:
            s = panel_stats(p)
            lines.append(
                f"{dname:<22} {p['component']:<8} {p['part']:<5} "
                f"{s['n']:>7d} {s['n_dropped']:>8d} "
                f"{s['rms']:>12.4e} {s['nrms']:>8.3f}"
            )
    lines.append("=" * len(hdr))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Script driver
# ---------------------------------------------------------------------------
def main(argv: Sequence[str]) -> int:
    import matplotlib
    if not SHOW:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = list(argv) if argv else list(RESULTS_FILES)
    os.makedirs(OUT_DIR, exist_ok=True)

    for h5_path in files:
        if not os.path.isfile(h5_path):
            print(f"femtic_crossplot: {h5_path} not found, skipping.")
            continue

        data = fem.read_results_data(h5_path, out=True)
        if data["nRows"] == 0:
            print(f"femtic_crossplot: {h5_path}: no data rows, skipping.")
            continue

        panels = build_panels(data, datatypes=DATATYPES, components=COMPONENTS)
        if not panels:
            print(f"femtic_crossplot: {h5_path}: nothing left after "
                  f"DATATYPES/COMPONENTS selection.")
            continue

        prefix = OUT_PREFIX or os.path.splitext(os.path.basename(h5_path))[0]
        crange = global_logf_range(panels) if COLOR_RANGE == "global" else None

        for dname, lst in panels.items():
            fig = plot_datatype(
                dname, lst,
                iter_num=data["iterNum"], ncols=NCOLS, panel_size=PANEL_SIZE,
                cmap=CMAP, color_range=crange, marker_size=MARKER_SIZE,
                alpha=ALPHA, log_rho=LOG_RHO, show_errorbars=SHOW_ERRORBARS,
                equal_aspect=EQUAL_ASPECT,
            )
            for ext in PLOT_FORMATS:
                out = os.path.join(OUT_DIR, f"{prefix}_{_safe_name(dname)}{ext}")
                fig.savefig(out, dpi=PLOT_DPI)
                print(f"femtic_crossplot: wrote {out}")
            if SHOW:
                plt.show()
            plt.close(fig)

        table = format_stats_table(panels)
        print(table)
        if WRITE_STATS:
            out = os.path.join(OUT_DIR, f"{prefix}_stats.txt")
            with open(out, "w", encoding="ascii") as fh:
                fh.write(f"# {h5_path}  iteration {data['iterNum']}\n")
                fh.write(table + "\n")
            print(f"femtic_crossplot: wrote {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
