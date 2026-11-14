#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_distort_stat.py
======================

Statistics and histograms of the galvanic-distortion matrices of a FEMTIC
ensemble (ens family): for every accepted member the distortion file of its
best iteration, distortion_iter<N>.dat, is read; the distortion matrices of all
members and sites are collected, plotted as histograms and summarised.

@author vrath
@project py4mt
@created 2026-11-13
@modified 2026-11-14

AI-GENERATED CODE NOTICE
------------------------
Author    : Claude Sonnet 5.5 (Anthropic)
Generated : 2026-11-13
This module is AI-generated. It was tested only with a synthetic ensemble (see
femtic_distort_stat_readme.md), not with a real FEMTIC ensemble. Review before
production or publication use.

Members and iterations
----------------------
The members are chosen as in femtic_ens_post.py (same settings ENSEMBLE_DIR,
ENSEMBLE_NAME, CNV_FILE, NRMS_MAX, ITER_FALLBACK): every sub-directory of
ENSEMBLE_DIR named ENSEMBLE_NAME* is a candidate; its best iteration N is the
iteration > 0 with the smallest nRMS in the convergence file (ties: the lower
iteration; iteration 0 is the starting model and is not eligible); members with
nRMS > NRMS_MAX are left out. If distortion_iter<N>.dat does not exist,
ITER_FALLBACK = True takes the iteration > 0 with the smallest nRMS that has
the file (and nRMS <= NRMS_MAX), as femtic_ens_post.py does for the block
files. The members are therefore the same as in the other ens scripts.

Quantities
----------
FEMTIC estimates the difference matrix C' (type
ESTIMATE_DISTORTION_MATRIX_DIFFERENCE) and writes it to distortion_iter<N>.dat;
the distortion matrix is C = I + C' (fem.read_distortion_file). QUANTITIES:

    "xx", "xy", "yx", "yy"   elements of C (DISTORTION_KIND = "full") or of C'
                             (DISTORTION_KIND = "prime"; the diagonal then
                             is C - 1)
    "condition"              sigma_max / sigma_min of C
    "determinant"            det(C)
    "gain"                   mean singular value of C = (sigma_max+sigma_min)/2

The last three are always those of the full matrix C.

Output (in OUT_DIR)
-------------------
    <OUTPUT_BASENAME>.<fmt>            histograms of all members and sites
                                       pooled (one panel per quantity)
    <OUTPUT_BASENAME>_site<ID>.<fmt>   one figure per site of PER_SITE: the
                                       members of that site (best-nRMS member
                                       marked)
                                       <fmt> = every format of OUTPUT_FORMATS
                                       (png, pdf, jpg, svg, eps, tif, ...)
    STATS_CSV                          n, mean, std and PERCENTILES of every
                                       quantity, pooled ("ALL") and per site
    RESULTS_NPZ                        members, iterations, nRMS, site IDs and
                                       the matrices C (members x sites x 2 x 2,
                                       NaN where a member has no such site)
    femtic_distort_stat_summary.md     the user-set parameters (in ENSEMBLE_DIR)

The distortion matrices of the best-fit member (the accepted member with the
smallest nRMS; ties: the first by name) are printed for every site: elements of
C (or C'), condition number and determinant (PRINT_BEST).

PLOT_MEMBERS limits the figures to one member or a short list (e.g. "best", a
directory name, a pattern, or a number); the statistics always cover all
accepted members.

The pooled histograms mix the spread of the distortion between sites with the
spread between members; the per-site figures and statistics show the
uncertainty of the distortion of one site.

Limits
------
Only distortion files of type ESTIMATE_DISTORTION_MATRIX_DIFFERENCE are
understood (the type for which fem.read_distortion_file gives C = C' + I). The
files of the gains-only and the gains-and-rotations types hold other parameters
and are NOT implemented. If results_iter<N>.h5 is next to a member's files (and
h5py is available) the /distortion type of the first such member is checked and
the run stops for any other type.

Usage
-----
Edit the configuration block below and run

    python femtic_distort_stat.py

Requires
--------
numpy, matplotlib; femtic.py, util.py, version.py (py4mt, found via
PY4MTX_ROOT); h5py only for the optional type check.

Provenance
----------
    2026-11-13  Claude Sonnet 5.5 (Anthropic)
                Created. Collects distortion_iter<best>.dat of the accepted
                members (selection as in femtic_ens_post.py), pooled and
                per-site histograms of the elements, condition number,
                determinant and gain of C, statistics CSV and results NPZ.
                AI-generated; review before production use.
    2026-11-13b Claude Sonnet 5.5 (Anthropic)
                OUTPUT_FORMATS: one string or a list, any format matplotlib
                writes (png, pdf, jpg, svg, eps, tif, ...); names are
                normalised and checked before the ensemble is read; new
                JPEG_QUALITY; text stays text in pdf / eps / ps.
                AI-generated; review before production use.
    2026-11-13c Claude Sonnet 5.5 (Anthropic)
                New PRINT_BEST: table of the distortion matrices of the
                best-fit member (smallest nRMS) for every site.
                AI-generated; review before production use.
    2026-11-14  Claude Sonnet 5.5 (Anthropic)
                New PLOT_MEMBERS: figures for one member or a short list of
                members ("best", names, patterns, numbers); the file names
                carry the members' names; PER_SITE needs two or more members.
                AI-generated; review before production use.
"""

from __future__ import annotations

import csv
import fnmatch
import glob
import inspect
import os
import re
import sys
import textwrap
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

for _base in [PY4MTX_ROOT + "/py4mt/modules/"]:
    for _p in [Path(_base), *Path(_base).rglob("*")]:
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

import femtic as fem
import util as utl
from version import versionstrg

rng = np.random.default_rng()
nan = np.nan
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())
titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng + "\n\n")

# =============================================================================
#  Configuration
# =============================================================================

# --- ensemble (same settings as femtic_ens_post.py) ---------------------------
ENSEMBLE_DIR = r"/media/vrath/LargeBack/Ensembles/annecy2026/RTO/ann_rto_orig-err/"
ENSEMBLE_NAME = "ann_rto_orig-err_"

#: Convergence file in each member directory: one name or a list of candidates.
CNV_FILE = ["femtic.cnv", "FEMTIC-DABIC.cnv"]

#: Members whose best nRMS is larger are left out.
NRMS_MAX = 1.5

#: If distortion_iter<best>.dat does not exist: True = use the best iteration
#: that has the file (nRMS <= NRMS_MAX), False = skip the member.
ITER_FALLBACK = True

#: Distortion file in each member directory; {iter} = iteration number.
DISTORTION_FILE = "distortion_iter{iter}.dat"

#: results_iter<N>.h5 of the member, only to check the type of the distortion
#: parameters (None = no check; skipped without h5py or file).
RESULTS_H5_FILE = "results_iter{iter}.h5"

#: Site list CSV (mt_make_sitelist.py format) for site names in titles and in
#: the CSV; None = site IDs only. A name inside each member directory is not
#: searched; give one path.
SITE_DAT = None

# --- quantities ---------------------------------------------------------------
#: "full": elements of C = I + C'; "prime": elements of C' (see above).
DISTORTION_KIND = "full"

#: Any of "xx", "xy", "yx", "yy", "condition", "determinant", "gain".
QUANTITIES = ["xx", "xy", "yx", "yy"]

#: Percentiles of the statistics table / CSV (2.3, 15.9, 84.1, 97.7 = the
#: 2-sigma and 1-sigma bounds, as in the other ens scripts).
PERCENTILES = [2.3, 15.9, 50.0, 84.1, 97.7]

# --- histograms ---------------------------------------------------------------
#: Number of bins, or any numpy rule ("auto", "fd", "sturges", ...), or a
#: sequence of bin edges.
BINS = 20

#: True = density (area 1), False = counts.
DENSITY = False

#: Range of the histograms: None (each panel its own), or a dict
#: {quantity: (lo, hi)}, e.g. {"xx": (0.5, 1.5)}.
RANGE = None

#: Mark the median (solid) and the 1-sigma percentiles (dashed) in every panel.
SHOW_PERCENTILES = True

#: Mark the value without distortion (C = I: 1 for xx, yy, condition,
#: determinant, gain; 0 for xy, yx, and for all elements of C' with "prime").
SHOW_REFERENCE = True

#: Per-site figures: None (none), "all", or a list of site IDs and / or names.
PER_SITE = None

#: In the per-site figures: mark the value of the member with the smallest nRMS.
MARK_BEST = True

#: Members shown in the figures: None (or "all") = all accepted members; or one
#: selector or a short list of selectors:
#:   "best"           the best-fit member (smallest nRMS)
#:   "ann_..._017"    a member by its directory name
#:   "ann_*_01?"      a pattern (* ? [..]) matched against the directory names
#:   17 or "17"       the member whose name ends in the number 17 (_017, 17, ...)
#: e.g. PLOT_MEMBERS = "best"  or  ["best", 17, "ann_rto_orig-err_023"].
#: Only the figures use the selection: the statistics (CSV, NPZ, printed table)
#: and the best-fit table cover all accepted members. The figure files get the
#: names of the members added (femtic_distort_stat_<member>.png). The histograms
#: of one member show its sites; PER_SITE figures need two or more members.
PLOT_MEMBERS = None

#: Print the distortion matrices of the best-fit member (the accepted member with
#: the smallest nRMS) for every site: elements of C (or C'), cond(C), det(C).
PRINT_BEST = True

# --- figure / output ------------------------------------------------------------
#: Size in inches of one panel (axes and labels); the figure is made from it, and
#: the title and caption are added to the height. Make it larger for large fonts
#: (3.6 x 2.9 holds font sizes up to 12; matplotlib then warns "Tight layout not
#: applied" for panels that are too small).
FIGSIZE_PANEL = (3.6, 2.9)
#: Resolution in dpi of the raster formats (png, jpg, tif, webp); pdf, svg, eps
#: and ps are vector formats and ignore it.
DPI = 400
TICK_FONTSIZE = 12
LABEL_FONTSIZE_AXES = 12
HIST_COLOR = "#2a78d6"          # bars (single series)
INK = "#0b0b0b"                 # text
INK_SECONDARY = "#52514e"       # reference / percentile lines, secondary text
SURFACE = "#fcfcfb"             # figure and axes background, bar gaps
GRID_COLOR = "#dcdbd6"

#: Figure title: "auto"; None, False or "" = none; text (placeholders
#: {n_members}, {n_sites}, {site}, {id}); or a dict {"pooled": spec,
#: "site": spec, "default": spec}.
TITLE = ""

SHOW = False

#: Output directory (made if missing) and names.
OUT_DIR = os.path.join(ENSEMBLE_DIR, "results") + os.sep
OUTPUT_BASENAME = "femtic_distort_stat"
#: Output formats of the figures, one file per format: a list, e.g.
#: ["png", "pdf", "jpg"], or one string ("pdf", or "pdf, jpg"). Any format that
#: matplotlib writes: raster png, jpg (jpeg), tif (tiff), webp; vector pdf, svg,
#: eps, ps. Case and a leading dot do not matter (".PDF" = "pdf").
OUTPUT_FORMATS = ["png"]
#: Quality of jpg / jpeg files, 1 - 100 (95 = high quality; higher values make
#: large files with little gain).
JPEG_QUALITY = 95
STATS_CSV = "femtic_distort_stat.csv"
RESULTS_NPZ = "femtic_distort_stat.npz"

# =============================================================================
#  Functions (explicit parameters; the configuration is supplied in main())
# =============================================================================

_ELEMENTS = {"xx": (0, 0), "xy": (0, 1), "yx": (1, 0), "yy": (1, 1)}
_DERIVED = ("condition", "determinant", "gain")


def find_member_dirs(ensemble_dir: str, ensemble_name: str) -> List[str]:
    """Sub-directories of ``ensemble_dir`` named ``ensemble_name*``, sorted by name."""
    pat = os.path.join(glob.escape(str(ensemble_dir)), glob.escape(ensemble_name) + "*")
    return sorted(d for d in glob.glob(pat) if os.path.isdir(d))


def best_iteration(rows: Sequence[dict]) -> Tuple[Optional[int], Optional[float], List[Tuple[int, float]]]:
    """Best iteration of a convergence history (``fem.read_cnv`` rows).

    Returns ``(iteration, nRMS, candidates)``: the iteration > 0 with the
    smallest nRMS (ties: the lower iteration), and the list of all
    ``(iteration, nRMS)`` pairs with iteration > 0. ``(None, None, [])`` if
    there is no iteration > 0.
    """
    cands = [(int(r["Iter"]), float(r["RMS"])) for r in rows if int(r["Iter"]) > 0]
    if not cands:
        return None, None, []
    it, rms = min(cands, key=lambda c: (c[1], c[0]))
    return it, rms, cands


def scan_members(ensemble_dir: str, ensemble_name: str, cnv_file, nrms_max: float,
                 distortion_tpl: str, fallback: bool = True
                 ) -> Tuple[List[dict], List[dict]]:
    """Choose the members and their iterations, as femtic_ens_post.py does.

    Returns ``(members, rejected)``. Every member is a dict ``label, dir, iter,
    nrms, file, status`` (status ``accepted`` or ``accepted_fallback``); every
    rejected directory a dict ``label, reason, detail`` with reason one of
    ``no_cnv_file``, ``cnv_unreadable``, ``cnv_no_iterations``,
    ``rejected_nrms``, ``no_distortion_file``.
    """
    members: List[dict] = []
    rejected: List[dict] = []
    for d in find_member_dirs(ensemble_dir, ensemble_name):
        label = os.path.basename(os.path.normpath(d))
        cnv = fem.find_cnv_file(d, cnv_file)
        if cnv is None:
            rejected.append(dict(label=label, reason="no_cnv_file",
                                 detail="none of " + ", ".join(fem.cnv_names(cnv_file))))
            continue
        try:
            rows = fem.read_cnv(str(cnv))["rows"]
        except (ValueError, OSError) as exc:
            rejected.append(dict(label=label, reason="cnv_unreadable", detail=str(exc)))
            continue
        it, rms, cands = best_iteration(rows)
        if it is None:
            rejected.append(dict(label=label, reason="cnv_no_iterations",
                                 detail=f"{Path(cnv).name}: no iteration > 0"))
            continue
        if rms > nrms_max:
            rejected.append(dict(label=label, reason="rejected_nrms",
                                 detail=f"best nRMS {rms:.4f} (iter {it}) > NRMS_MAX {nrms_max}"))
            continue
        f = os.path.join(d, distortion_tpl.format(iter=it))
        status = "accepted"
        if not os.path.isfile(f):
            avail = [(r2, i2) for i2, r2 in cands
                     if os.path.isfile(os.path.join(d, distortion_tpl.format(iter=i2)))]
            if fallback and avail:
                r2, i2 = min(avail)                       # lowest nRMS, ties: lower iteration
                if r2 <= nrms_max:
                    it, rms, status = i2, r2, "accepted_fallback"
                    f = os.path.join(d, distortion_tpl.format(iter=it))
                else:
                    rejected.append(dict(
                        label=label, reason="no_distortion_file",
                        detail=f"{Path(f).name} missing; best available iter {i2} has "
                               f"nRMS {r2:.4f} > NRMS_MAX {nrms_max}"))
                    continue
            else:
                rejected.append(dict(
                    label=label, reason="no_distortion_file",
                    detail=f"{Path(f).name} missing"
                           + ("" if fallback else " (ITER_FALLBACK is False)")))
                continue
        members.append(dict(label=label, dir=d, iter=it, nrms=rms, file=f, status=status))
    return members, rejected


def read_distortion(path: str) -> Tuple[List[int], np.ndarray]:
    """Site IDs (file order) and the matrices C = C' + I of one distortion file."""
    c, _cp = fem.read_distortion_file(str(path))
    lines = Path(path).read_text().strip().splitlines()
    ids = [int(ln.split()[0]) for ln in lines[1:] if ln.split()]
    if len(ids) != len(c):
        raise RuntimeError(f"{path}: {len(ids)} site IDs but {len(c)} matrices.")
    return ids, np.asarray(c, dtype=float)


def load_ensemble(members: List[dict]) -> Tuple[List[dict], List[int], np.ndarray, List[dict]]:
    """Read the distortion file of every member.

    Returns ``(members, site_ids, C, failed)``: the members that could be read,
    the sorted union of their site IDs, the array ``C`` of shape
    ``(n_members, n_sites, 2, 2)`` (NaN where a member lacks a site) and the
    members that could not be read (dict ``label, reason, detail``).
    """
    ok: List[dict] = []
    failed: List[dict] = []
    data: List[Tuple[List[int], np.ndarray]] = []
    for m in members:
        try:
            ids, C = read_distortion(m["file"])
        except Exception as exc:  # unreadable or malformed file: skip the member
            failed.append(dict(label=m["label"], reason="distortion_unreadable",
                               detail=f"{Path(m['file']).name}: {exc}"))
            continue
        if len(set(ids)) != len(ids):
            warnings.warn(f"{m['file']}: duplicate site IDs; the last row of each is used.")
        ok.append(m)
        data.append((ids, C))
    site_ids = sorted({i for ids, _ in data for i in ids})
    pos = {sid: k for k, sid in enumerate(site_ids)}
    arr = np.full((len(ok), len(site_ids), 2, 2), np.nan)
    for im, (ids, C) in enumerate(data):
        for row, sid in enumerate(ids):
            arr[im, pos[sid]] = C[row]
    return ok, site_ids, arr, failed


def check_distortion_type(h5_path) -> Optional[str]:
    """Stop if ``results_iter<N>.h5`` says the distortion is not a matrix difference.

    Returns a short note (str) if the check could not be made, else None.
    Raises RuntimeError for any /distortion type other than
    ESTIMATE_DISTORTION_MATRIX_DIFFERENCE (1).
    """
    p = Path(h5_path)
    if not p.is_file():
        return f"{p.name} not found"
    try:
        res = fem.read_results_distortion(str(p), out=False)
    except ImportError:
        return "h5py not available"
    except Exception as exc:
        return f"{p.name} not readable ({exc})"
    if res is None:
        return f"{p.name} has no /distortion group"
    if int(res["type"]) != 1:
        raise RuntimeError(
            f"{p.name}: distortion type is {res['type_name']}; only "
            f"ESTIMATE_DISTORTION_MATRIX_DIFFERENCE is implemented here.")
    return None


def quantity_values(C: np.ndarray, key: str, kind: str = "full") -> np.ndarray:
    """Values of one quantity for an array of matrices ``C[..., 2, 2]``.

    ``key``: ``"xx" "xy" "yx" "yy"`` (elements of C, or of C' for
    ``kind="prime"``), ``"condition"``, ``"determinant"``, ``"gain"`` (always of
    the full C). NaN stays NaN; the condition number of a singular matrix is inf.
    """
    if kind not in ("full", "prime"):
        raise ValueError(f"DISTORTION_KIND must be 'full' or 'prime', got {kind!r}")
    if key in _ELEMENTS:
        i, j = _ELEMENTS[key]
        v = np.array(C[..., i, j], dtype=float)
        if kind == "prime" and i == j:
            v = v - 1.0
        return v
    if key not in _DERIVED:
        raise ValueError(f"unknown quantity {key!r}; use {list(_ELEMENTS) + list(_DERIVED)}")
    a, b, c, d = C[..., 0, 0], C[..., 0, 1], C[..., 1, 0], C[..., 1, 1]
    det = a * d - b * c
    if key == "determinant":
        return det
    s1 = a * a + b * b + c * c + d * d               # sum of the squared singular values
    root = np.sqrt(np.maximum(s1 * s1 - 4.0 * det * det, 0.0))
    smax = np.sqrt((s1 + root) / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        smin = np.abs(det) / smax                    # sigma_max * sigma_min = |det|
        if key == "condition":
            return smax / smin
    return 0.5 * (smax + smin)


def reference_value(key: str, kind: str = "full") -> float:
    """Value of a quantity for no distortion (C = I)."""
    if key in _ELEMENTS:
        i, j = _ELEMENTS[key]
        return 1.0 if (i == j and kind == "full") else 0.0
    return 1.0


def quantity_label(key: str, kind: str = "full") -> str:
    """Axis label (mathtext) of a quantity."""
    c = "C" if kind == "full" else "C'"
    if key in _ELEMENTS:
        return rf"${c}_{{{key}}}$"
    return {"condition": r"cond($C$) = $\sigma_{max}/\sigma_{min}$",
            "determinant": r"det($C$)",
            "gain": r"mean singular value of $C$"}[key]


def describe(values: np.ndarray, percentiles: Sequence[float]) -> dict:
    """n, mean, std (ddof = 1) and percentiles of the finite values."""
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    out = {"n": int(v.size), "mean": nan, "std": nan}
    for q in percentiles:
        out[f"p{q:g}"] = nan
    if v.size:
        out["mean"] = float(v.mean())
        out["std"] = float(v.std(ddof=1)) if v.size > 1 else nan
        for q, val in zip(percentiles, np.percentile(v, list(percentiles))):
            out[f"p{q:g}"] = float(val)
    return out


def stats_table(C: np.ndarray, site_ids: Sequence[int], names: Dict[int, str],
                quantities: Sequence[str], kind: str, percentiles: Sequence[float]
                ) -> List[dict]:
    """Statistics of every quantity: pooled (site ``"ALL"``) and per site."""
    rows: List[dict] = []
    for q in quantities:
        vals = quantity_values(C, q, kind)                      # (members, sites)
        rows.append(dict(site_id="ALL", site_name="", quantity=q, **describe(vals, percentiles)))
    for k, sid in enumerate(site_ids):
        for q in quantities:
            vals = quantity_values(C[:, k], q, kind)
            rows.append(dict(site_id=sid, site_name=names.get(sid, ""), quantity=q,
                             **describe(vals, percentiles)))
    return rows


def write_stats_csv(path: Path, rows: List[dict], percentiles: Sequence[float]) -> None:
    cols = ["site_id", "site_name", "quantity", "n", "mean", "std"] + [f"p{q:g}" for q in percentiles]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="ascii", errors="replace") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] if c in ("site_id", "site_name", "quantity", "n")
                        else f"{r[c]:.6g}" for c in cols])


def format_pooled_table(rows: List[dict], percentiles: Sequence[float]) -> str:
    """ASCII table of the pooled statistics."""
    pc = [f"p{q:g}" for q in percentiles]
    head = f"{'quantity':<12s} {'n':>7s} {'mean':>9s} {'std':>9s} " + " ".join(f"{c:>9s}" for c in pc)
    line = "-" * len(head)
    out = ["=" * len(head), head, line]
    for r in rows:
        if r["site_id"] != "ALL":
            continue
        out.append(f"{r['quantity']:<12s} {r['n']:7d} {r['mean']:9.4f} {r['std']:9.4f} "
                   + " ".join(f"{r[c]:9.4f}" for c in pc))
    out.append("=" * len(head))
    return "\n".join(out)


def format_best_table(member: dict, C_best: np.ndarray, site_ids: Sequence[int],
                      names: Dict[int, str], kind: str = "full") -> str:
    """ASCII table of the distortion matrices of one member (the best-fit member).

    ``member`` is a member dict of :func:`scan_members`; ``C_best`` has shape
    ``(n_sites, 2, 2)`` (the full matrices C, NaN for a site the member does not
    have). Columns: site ID, name (only if names are known), the four elements
    of C (``kind = "full"``) or C' (``"prime"``), cond(C) and det(C); a
    negative determinant is marked ``det<0``.
    """
    cp = "C" if kind == "full" else "C'"
    head_txt = (f"Distortion of the best-fit member {member['label']} (iteration {member['iter']}, "
                f"nRMS {member['nrms']:.4f}"
                + (", ITER_FALLBACK" if member.get("status") == "accepted_fallback" else "")
                + f"): elements of {cp}" + (" = I + C'" if kind == "full" else ""))
    show_names = any(names.get(sid) for sid in site_ids)
    el = [quantity_values(C_best, q, kind) for q in ("xx", "xy", "yx", "yy")]
    cond = quantity_values(C_best, "condition")
    det = quantity_values(C_best, "determinant")
    nw = max([4] + [len(str(names.get(sid, ""))) for sid in site_ids]) if show_names else 0
    cols = (f"{'site':>8s} " + (f"{'name':<{nw}s} " if show_names else "")
            + " ".join(f"{cp + q:>9s}" for q in ("xx", "xy", "yx", "yy"))
            + f" {'cond(C)':>9s} {'det(C)':>9s}")
    out = [head_txt, "=" * len(cols), cols, "-" * len(cols)]
    for k, sid in enumerate(site_ids):
        left = f"{sid:>8d} " + (f"{str(names.get(sid, '')):<{nw}s} " if show_names else "")
        if not np.isfinite(C_best[k]).all():
            out.append(left + "(site not in the distortion file of this member)")
            continue
        out.append(left + " ".join(f"{e[k]:9.4f}" for e in el)
                   + f" {cond[k]:9.3f} {det[k]:9.4f}" + ("  det<0" if det[k] < 0 else ""))
    out.append("=" * len(cols))
    return "\n".join(out)


def select_members(members: Sequence[dict], spec) -> List[int]:
    """Indices into ``members`` chosen by the PLOT_MEMBERS setting.

    ``spec``: None or ``"all"`` (all members), or one selector or a list of
    selectors: ``"best"`` (smallest nRMS; ties: the first), a directory name, a
    pattern with ``* ? [..]`` (matched against the directory names), or a number
    (``17`` or ``"17"``: the member whose name ends in that number). The result
    keeps the order of the selectors and has no repeats. A selector that matches
    nothing, or a number that matches several members, raises ``ValueError``.
    """
    n = len(members)
    labels = [m["label"] for m in members]
    if spec is None or (isinstance(spec, str) and spec.strip().lower() in ("", "all")):
        return list(range(n))
    items = list(spec) if isinstance(spec, (list, tuple, set, np.ndarray)) else [spec]
    if not items:
        raise ValueError("PLOT_MEMBERS: empty list (use None for all members)")

    def number_hits(num: int) -> List[int]:
        hits = []
        for i, lab in enumerate(labels):
            mm = re.search(r"(\d+)$", lab)
            if mm and int(mm.group(1)) == num:
                hits.append(i)
        return hits

    def available() -> str:
        txt = ", ".join(labels[:8])
        return txt + (f", ... ({n} members)" if n > 8 else "")

    out: List[int] = []
    for it in items:
        if isinstance(it, (bool, np.bool_)) or not isinstance(it, (str, int, np.integer)):
            raise ValueError(f"PLOT_MEMBERS: unusable selector {it!r}; use 'best', a name, "
                             "a pattern or a number")
        if isinstance(it, (int, np.integer)):
            hits = number_hits(int(it))
            if len(hits) > 1:
                raise ValueError(f"PLOT_MEMBERS: number {it} matches {len(hits)} members "
                                 f"({', '.join(labels[i] for i in hits[:5])}); use the full name")
        else:
            sel = it.strip()
            if sel.lower() == "best":
                hits = [int(np.argmin([m["nrms"] for m in members]))]
            elif sel.lower() == "all":
                hits = list(range(n))
            elif sel in labels:
                hits = [labels.index(sel)]
            elif any(c in sel for c in "*?["):
                hits = [i for i, lab in enumerate(labels) if fnmatch.fnmatchcase(lab, sel)]
            elif sel.isdigit():
                hits = number_hits(int(sel))
                if len(hits) > 1:
                    raise ValueError(f"PLOT_MEMBERS: number {sel} matches {len(hits)} members "
                                     f"({', '.join(labels[i] for i in hits[:5])}); use the full name")
            else:
                hits = []
        if not hits:
            raise ValueError(f"PLOT_MEMBERS: {it!r} matches no member; available: {available()}")
        out.extend(i for i in hits if i not in out)
    return out


def members_tag(labels: Sequence[str], maxlen: int = 60) -> str:
    """File-name part for a set of members: the names joined by "+", or "<k>members" if long."""
    tag = "+".join(re.sub(r"[^A-Za-z0-9_.-]", "_", lab) for lab in labels)
    return tag if len(tag) <= maxlen else f"{len(labels)}members"


def _panel(ax, vals: np.ndarray, key: str, kind: str, *, bins, density: bool,
           value_range, show_percentiles: bool, show_reference: bool, best_value,
           color: str, ink: str, ink2: str, surface: str, grid: str,
           tick_fs: int, label_fs: int) -> None:
    """One histogram panel."""
    v = np.asarray(vals, dtype=float).ravel()
    v = v[np.isfinite(v)]
    ax.set_facecolor(surface)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(ink2)
    ax.tick_params(labelsize=tick_fs, colors=ink2)
    ax.grid(axis="y", color=grid, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(quantity_label(key, kind), fontsize=label_fs, color=ink)
    ax.set_ylabel("density" if density else "count", fontsize=label_fs, color=ink)
    if not density:
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))     # counts: whole numbers
    if v.size == 0:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center",
                color=ink2, fontsize=label_fs)
        return
    rng_ = value_range if value_range is not None else None
    ax.hist(v, bins=bins, range=rng_, density=density, color=color, edgecolor=surface,
            linewidth=0.8)
    if rng_ is not None:
        ax.set_xlim(rng_)                            # the requested range, without margins
    x0, x1 = ax.get_xlim()
    pct = np.percentile(v, [15.9, 50.0, 84.1])
    if show_percentiles:
        ax.axvline(pct[1], color=ink2, lw=1.4)
        for q in (pct[0], pct[2]):
            ax.axvline(q, color=ink2, lw=1.0, ls="--")
    if show_reference:
        ref = reference_value(key, kind)
        if x0 <= ref <= x1:
            ax.axvline(ref, color=ink, lw=1.0, ls=":")
    if best_value is not None and np.isfinite(best_value):
        ax.plot([best_value], [0.0], marker="v", ms=7, mfc=ink, mec=surface, mew=0.8,
                transform=ax.get_xaxis_transform(), clip_on=False, ls="none")
    ax.set_xlim(x0, x1)
    ax.set_title(f"n = {v.size},  median = {pct[1]:.3g}", loc="right", fontsize=tick_fs,
                 color=ink2, pad=4)


def _wrap_parts(parts: Sequence[str], nchar: int) -> str:
    """Join ``parts`` with "; ", starting a new line instead of passing ``nchar`` characters."""
    lines: List[str] = []
    cur = ""
    for part in parts:
        add = part if not cur else cur + "; " + part
        if cur and len(add) > nchar:
            lines.append(cur + ";")
            cur = part
        else:
            cur = add
    if cur:
        lines.append(cur)
    out: List[str] = []
    for ln in lines:                                  # a single long part is wrapped at blanks
        out.extend(textwrap.wrap(ln, width=nchar, break_long_words=False) or [""])
    return "\n".join(out)


def _chars_per_line(width_in: float, fontsize: float) -> int:
    """Number of characters of the given font size that fit safely into ``width_in`` inches."""
    return max(10, int(0.92 * width_in * 72.0 / (0.58 * fontsize)))


def plot_histograms(data: Dict[str, np.ndarray], *, kind: str = "full", bins=30,
                    density: bool = False, ranges=None, show_percentiles: bool = True,
                    show_reference: bool = True, best_values: Optional[Dict[str, float]] = None,
                    title: str = "", panel_size: Tuple[float, float] = (3.6, 2.9),
                    color: str = "#2a78d6", ink: str = "#0b0b0b", ink2: str = "#52514e",
                    surface: str = "#fcfcfb", grid: str = "#dcdbd6", tick_fs: int = 8,
                    label_fs: int = 9):
    """Figure with one histogram panel per quantity of ``data`` (key -> values).

    The title and the caption that names the line styles are wrapped to the
    width of the figure, which is made from the size of one panel; the room for
    them is added to the height.
    """
    import matplotlib.pyplot as plt
    keys = list(data)
    n = len(keys)
    ncol = 2 if n <= 4 else 3
    nrow = int(np.ceil(n / ncol))
    width = panel_size[0] * ncol

    # one caption names the line styles (instead of a label in every panel)
    parts = []
    if show_percentiles:
        parts.append("solid line = median, dashed = 15.9 / 84.1 percentile")
    if show_reference:
        parts.append("dotted = no distortion" + (" (C = I)" if kind == "full" else " (C' = 0)"))
    if best_values:
        parts.append("triangle = best-nRMS member")
    caption = _wrap_parts(parts, _chars_per_line(width, tick_fs))
    title_txt = "\n".join(textwrap.wrap(str(title), width=_chars_per_line(width, label_fs + 1),
                                        break_long_words=False)) if title else ""
    cap_h = (caption.count("\n") + 1) * 1.3 * tick_fs / 72.0 + 0.15 if caption else 0.0
    ttl_h = (title_txt.count("\n") + 1) * 1.3 * (label_fs + 1) / 72.0 + 0.1 if title_txt else 0.0
    height = panel_size[1] * nrow + cap_h + ttl_h

    fig, axs = plt.subplots(nrow, ncol, squeeze=False, facecolor=surface, figsize=(width, height))
    for i, key in enumerate(keys):
        ax = axs[i // ncol, i % ncol]
        rg = None if ranges is None else ranges.get(key)
        _panel(ax, data[key], key, kind, bins=bins, density=density, value_range=rg,
               show_percentiles=show_percentiles, show_reference=show_reference,
               best_value=None if best_values is None else best_values.get(key),
               color=color, ink=ink, ink2=ink2, surface=surface, grid=grid,
               tick_fs=tick_fs, label_fs=label_fs)
    for j in range(n, nrow * ncol):
        axs[j // ncol, j % ncol].set_visible(False)
    if title_txt:
        fig.suptitle(title_txt, fontsize=label_fs + 1, color=ink)
    if caption:
        fig.text(0.5, 0.08 / height, caption, ha="center", va="bottom", fontsize=tick_fs,
                 color=ink2, linespacing=1.2)
    fig.tight_layout(rect=(0, cap_h / height, 1, 1.0 - ttl_h / height))
    return fig


def select_sites(site_ids: Sequence[int], names: Dict[int, str], spec) -> List[int]:
    """Indices into ``site_ids`` for the per-site figures (None, "all", IDs / names)."""
    if spec is None or spec is False:
        return []
    if isinstance(spec, str) and spec.strip().lower() == "all":
        return list(range(len(site_ids)))
    items = [spec] if isinstance(spec, (str, int, np.integer)) else list(spec)
    out: List[int] = []
    for it in items:
        hit = [k for k, sid in enumerate(site_ids)
               if str(sid) == str(it) or names.get(sid, None) == str(it)]
        if not hit:
            warnings.warn(f"PER_SITE: site {it!r} is not in the ensemble.")
        out.extend(k for k in hit if k not in out)
    return out


def normalize_formats(spec, supported: Optional[Sequence[str]] = None) -> List[str]:
    """File extensions from the OUTPUT_FORMATS setting.

    ``spec`` is a list / tuple of names or one string; a string may hold several
    names separated by commas, semicolons or blanks (``"pdf, jpg"``). Names are
    lower-cased, a leading dot is removed, repeated names are dropped (the order
    is kept). ``supported`` (e.g. the formats matplotlib writes) restricts the
    allowed names; a name outside it raises ``ValueError`` that lists the
    allowed ones. An empty result raises ``ValueError`` too.
    """
    if spec is None:
        items: List[str] = []
    elif isinstance(spec, (str, os.PathLike)):
        items = [os.fspath(spec)]
    else:
        items = [os.fspath(x) if isinstance(x, os.PathLike) else str(x) for x in spec]
    out: List[str] = []
    for item in items:
        for name in re.split(r"[,;\s]+", item):
            f = name.strip().lstrip(".").lower()
            if not f:
                continue
            if supported is not None and f not in supported:
                raise ValueError(f"OUTPUT_FORMATS: unknown format {name!r}; "
                                 f"use any of {sorted(supported)}")
            if f not in out:
                out.append(f)
    if not out:
        raise ValueError("OUTPUT_FORMATS: no format given (e.g. ['png', 'pdf'])")
    return out


def _save_figure(fig, stem: Path, formats: Sequence[str], dpi: int,
                 jpeg_quality: int = 95) -> List[str]:
    """Save ``fig`` as ``<stem>.<fmt>`` for every format; returns the file names.

    ``dpi`` applies to the raster formats; ``jpeg_quality`` to jpg / jpeg. In
    pdf / eps / ps the text is kept as text (fonts embedded as Type 42).
    """
    import matplotlib as mpl
    saved = []
    for fmt in formats:
        p = Path(f"{stem}.{fmt}")
        p.parent.mkdir(parents=True, exist_ok=True)
        kw = {}
        if fmt in ("jpg", "jpeg"):
            kw["pil_kwargs"] = {"quality": int(jpeg_quality)}
        with mpl.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
            fig.savefig(p, dpi=dpi, facecolor=fig.get_facecolor(), **kw)
        saved.append(str(p))
    return saved


# =============================================================================
#  Run
# =============================================================================


def main() -> int:
    import matplotlib.pyplot as plt
    if not SHOW:
        plt.switch_backend("Agg")

    for q in QUANTITIES:
        if q not in _ELEMENTS and q not in _DERIVED:
            raise ValueError(f"QUANTITIES: unknown quantity {q!r}; "
                             f"use {list(_ELEMENTS) + list(_DERIVED)}")
    from matplotlib.backend_bases import FigureCanvasBase
    formats = normalize_formats(OUTPUT_FORMATS, FigureCanvasBase.get_supported_filetypes())
    if not 1 <= int(JPEG_QUALITY) <= 100:
        raise ValueError(f"JPEG_QUALITY must be between 1 and 100, got {JPEG_QUALITY!r}")
    out_dir = Path(OUT_DIR)

    # ---- members -------------------------------------------------------------
    members, rejected = scan_members(ENSEMBLE_DIR, ENSEMBLE_NAME, CNV_FILE, NRMS_MAX,
                                     DISTORTION_FILE, ITER_FALLBACK)
    n_dirs = len(find_member_dirs(ENSEMBLE_DIR, ENSEMBLE_NAME))
    print(f"[femtic_distort_stat] {n_dirs} directory/ies matching '{ENSEMBLE_NAME}*' in "
          f"{ENSEMBLE_DIR}: {len(members)} accepted")
    if not members:
        raise RuntimeError("no member accepted - check ENSEMBLE_DIR / ENSEMBLE_NAME / "
                           "NRMS_MAX / DISTORTION_FILE.\n" + "\n".join(
                               f"  {r['label']}: {r['reason']} - {r['detail']}" for r in rejected[:20]))

    # type of the distortion parameters: first member that has its results file
    if RESULTS_H5_FILE is not None:
        checked, first_note = False, None
        for m in members:
            h5 = os.path.join(m["dir"], RESULTS_H5_FILE.format(iter=m["iter"]))
            note = check_distortion_type(h5)
            if note is None:
                checked = True                                   # type is fine
                break
            first_note = first_note or note
            if note.startswith("h5py"):
                break                                            # no member can be checked
        if not checked:
            print("[femtic_distort_stat] note: type of the distortion parameters "
                  f"not checked ({first_note})")

    members, site_ids, C, failed = load_ensemble(members)
    rejected = rejected + failed
    if not members:
        raise RuntimeError("no distortion file could be read.")

    sel = select_members(members, PLOT_MEMBERS)                    # members shown in the figures

    n_fb = sum(1 for m in members if m["status"] == "accepted_fallback")
    print(f"[femtic_distort_stat] {len(members)} member(s), {len(site_ids)} site(s); "
          f"iteration used: {min(m['iter'] for m in members)} - {max(m['iter'] for m in members)}"
          + (f"; {n_fb} member(s) with ITER_FALLBACK" if n_fb else ""))
    if rejected:
        by_reason: Dict[str, int] = {}
        for r in rejected:
            by_reason[r["reason"]] = by_reason.get(r["reason"], 0) + 1
        print("[femtic_distort_stat] not used: "
              + ", ".join(f"{v} {k}" for k, v in sorted(by_reason.items())))
        for r in rejected[:10]:
            print(f"    {r['label']}: {r['detail']}")
        if len(rejected) > 10:
            print(f"    ... and {len(rejected) - 10} more")
    n_partial = int(np.sum(np.isnan(C[..., 0, 0]).any(axis=0)))
    if n_partial:
        warnings.warn(f"{n_partial} site(s) are not in every member's distortion file; "
                      "their n is smaller than the number of members.")

    names: Dict[int, str] = {}
    if SITE_DAT:
        names = {int(r["sitenum"]): str(r["name"]) for r in fem.read_site_dat(SITE_DAT)}

    # ---- statistics ----------------------------------------------------------
    rows = stats_table(C, site_ids, names, QUANTITIES, DISTORTION_KIND, PERCENTILES)
    print(format_pooled_table(rows, PERCENTILES))
    ibest = int(np.argmin([m["nrms"] for m in members]))          # best-fit member
    if PRINT_BEST:
        print(format_best_table(members[ibest], C[ibest], site_ids, names, DISTORTION_KIND))
    f_csv = out_dir / STATS_CSV
    write_stats_csv(f_csv, rows, PERCENTILES)
    print(f"[femtic_distort_stat] statistics: {f_csv}")

    f_npz = out_dir / RESULTS_NPZ
    np.savez(f_npz, members=np.array([m["label"] for m in members]),
             iterations=np.array([m["iter"] for m in members]),
             nrms=np.array([m["nrms"] for m in members]),
             site_ids=np.array(site_ids), C=C, kind=np.array(DISTORTION_KIND),
             quantities=np.array(QUANTITIES))
    print(f"[femtic_distort_stat] results: {f_npz}")

    # ---- figures -------------------------------------------------------------
    subset = len(sel) < len(members)
    mp = [members[i] for i in sel]                                 # members in the figures
    Cp = C[sel]
    labels_p = [m["label"] for m in mp]
    if not subset:
        who = f"{len(mp)} members"
    elif len(mp) == 1:
        who = f"member {labels_p[0]}"
    else:
        who = f"{len(mp)} selected members"
    base = OUTPUT_BASENAME + (f"_{members_tag(labels_p)}" if subset else "")
    if subset:
        print(f"[femtic_distort_stat] figures for {len(mp)} of {len(members)} member(s): "
              + ", ".join(labels_p[:10]) + (" ..." if len(labels_p) > 10 else ""))
    pooled = {q: quantity_values(Cp, q, DISTORTION_KIND) for q in QUANTITIES}
    common = dict(kind=DISTORTION_KIND, bins=BINS, density=DENSITY, ranges=RANGE,
                  show_percentiles=SHOW_PERCENTILES, show_reference=SHOW_REFERENCE,
                  panel_size=FIGSIZE_PANEL, color=HIST_COLOR, ink=INK, ink2=INK_SECONDARY,
                  surface=SURFACE, grid=GRID_COLOR, tick_fs=TICK_FONTSIZE,
                  label_fs=LABEL_FONTSIZE_AXES)
    kind_txt = "C = I + C'" if DISTORTION_KIND == "full" else "C'"
    auto = (f"Distortion matrices of the ensemble ({kind_txt}): {who}, "
            f"{len(site_ids)} sites, best iteration of each")
    fields = dict(n_members=len(mp), n_sites=len(site_ids),
                  members=", ".join(labels_p) if subset else "all")
    ttl = utl.resolve_title(TITLE, key="pooled", auto=auto, name="TITLE", **fields)
    saved: List[str] = []
    fig = plot_histograms(pooled, title=ttl, **common)
    saved += _save_figure(fig, out_dir / base, formats, DPI, JPEG_QUALITY)
    if SHOW:
        plt.show()
    plt.close(fig)

    sites_sel = select_sites(site_ids, names, PER_SITE)
    if sites_sel and len(mp) < 2:
        print("[femtic_distort_stat] PER_SITE skipped: one member gives one value per site "
              "(select two or more members)")
        sites_sel = []
    mark = bool(MARK_BEST) and len(mp) > 1                         # marker needs a distribution
    ibest_p = int(np.argmin([m["nrms"] for m in mp]))
    for k in sites_sel:
        sid = site_ids[k]
        data = {q: quantity_values(Cp[:, k], q, DISTORTION_KIND) for q in QUANTITIES}
        best = ({q: float(quantity_values(Cp[ibest_p, k], q, DISTORTION_KIND)) for q in QUANTITIES}
                if mark else None)
        nm = names.get(sid, "")
        site_txt = f"site {nm} ({sid})" if nm else f"site {sid}"
        auto_s = (f"Distortion matrix of {site_txt} ({kind_txt}): {who}"
                  + (f"; marker = best member {mp[ibest_p]['label']} "
                     f"(nRMS {mp[ibest_p]['nrms']:.3g})" if mark else ""))
        ttl_s = utl.resolve_title(TITLE, key="site", auto=auto_s, name="TITLE",
                                  site=nm or sid, id=sid, **fields)
        fig = plot_histograms(data, title=ttl_s, best_values=best, **common)
        tag = re.sub(r"[^A-Za-z0-9_.-]", "_", str(sid))
        saved += _save_figure(fig, out_dir / f"{base}_site{tag}", formats, DPI, JPEG_QUALITY)
        if SHOW:
            plt.show()
        plt.close(fig)
    print(f"[femtic_distort_stat] {len(saved)} figure file(s) ({', '.join(formats)}):")
    for s in saved:
        print("   " + s)

    utl.write_param_summary(fname, out_dir=str(ENSEMBLE_DIR))
    return 0


if __name__ == "__main__":
    sys.exit(main())
