#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Part of the DIAS MT/seismic joint-imaging pipeline (tacnancaya / Tacna).
# Developed with AI assistance (Claude, Anthropic); review before use.
"""
crossplots.py — Simple 2-D and 3-D cross-plots of property pairs/triples
on the joint grid produced by ``interpolate.py``.

New pipeline stage, parallel to ``structure.py``/``plot_joint.py``/
``cluster.py``:

    precompute.py -> interpolate.py -> {SITE_PREFIX}_interp_<method>.nc
                                                     |
                                        crossplots.py -> figures

Reads the same ``INTERP_FILE`` NetCDF as the rest of the plotting
pipeline and is equally agnostic to which interpolation method produced
it. Unlike ``structure.py`` (which compares fields' *gradients* to
diagnose structural coupling), this script plots raw grid-point *values*
of two or three fields against each other -- the standard first-look
exploratory technique for spotting petrophysical relationships, clusters,
or outliers between co-located properties, independent of any assumed
functional relationship or the structural-coupling machinery in
``structure.py``/``cluster.py``.

Two plot kinds
--------------
  * Pairs (``CROSSPLOT_PAIRS``): 2-D scatter (or, for very dense point
    clouds, hexbin density) of field_a vs. field_b, every valid grid
    point pooled across depth/northing/easting. Optionally colour-coded
    by a third field (or by depth). Annotated with Pearson r and
    Spearman rho and the point count n -- both computed on whatever
    representation is actually plotted (see "Log fields" below), not
    silently on the raw linear values.
  * Triples (``CROSSPLOT_TRIPLES``): 3-D scatter of field_a/field_b/
    field_c, optionally colour-coded by a fourth field (or depth).
    Annotated with the three pairwise Pearson r's among the triple.

Log fields
----------
Fields in ``LOG_FIELDS`` are plotted (and correlated) as `log10(field)`
rather than the linear value, with the axis labelled accordingly --
same `log10(rho)` convention, and the same "compute on the plotted
representation, not the linear one" principle, used for the gradients in
``structure.py``/``plot_joint.py``/``plot_modem_mesh.py``. Non-positive
values in a log field are masked (flagged in the console log if any are
dropped), not silently clipped.

Point count and subsampling
----------------------------
Joint grids can have far more points than are useful to render
individually. If the number of valid (finite, in-range) points for a
plot exceeds ``MAX_POINTS``, a reproducible random subsample (seeded by
``RANDOM_SEED``) is drawn for *display* only -- the annotated statistics
(Pearson/Spearman/pairwise r) are always computed from the *full* valid
point set, not the subsample, so the numbers on the plot don't depend on
how many points happened to get drawn. ``MAX_POINTS`` has no
literature- or pipeline-verified "right" value for this project's grids;
flagged as a free parameter rather than tuned/guessed at.

Known limitation
-----------------
No depth/region filtering beyond the optional ``DEPTH_RANGE`` cutoff is
implemented (e.g. no polygon/mask clipping to a study area) -- every
valid grid point in ``INTERP_FILE`` is pooled. If a future need arises to
restrict cross-plots to a sub-volume, that's a straightforward filter to
add on top of the flattened arrays built by ``_prepare_pair()``/
``_prepare_triple()``.

``_resolve_interp_file()``, ``_derive_interp_tag()``, ``load_joint_grid()``,
``get_field()``, ``_title_suffix()``, ``_group_label()``, and
``_maybe_show()`` are thin per-script wrappers around the shared
implementations in ``tomomt.py`` (formerly duplicated verbatim here and
in ``structure.py``; now a single source of truth) -- each wrapper just
supplies this script's own SITE_PREFIX/INTERP_FILE/SHOW_PLOTS/etc.
globals, the same pattern ``plot_joint.py`` already used for ``tomomt.py``
(née ``plotpy.py``)'s plotting primitives. See ``tomomt.py``'s module
docstring for the full list of what moved there and, deliberately, what
didn't.

Authors: Svetlana Byrdina (SMB), Volker Rath (DIAS)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the '3d' projection
from scipy import stats

import tomomt

# ---------------------------------------------------------------------------
# Site / interpolated-grid selection
# ---------------------------------------------------------------------------

SITE_PREFIX = "tacna"  # "tacna" or "tacna" -- must match precompute.py/interpolate.py

# Explicit path, or None to auto-pick the newest {SITE_PREFIX}_interp_*.nc
# in the current directory (mirrors structure.py/plot_joint.py behaviour).
INTERP_FILE = "/home/sbyrd/python_scripts/tomomt_peru/precompute/tacna/tacna_interp_nearest.nc" #  None

# ---------------------------------------------------------------------------
# What to plot
# ---------------------------------------------------------------------------
# Pairs: (field_a, field_b, color_by, label). field_a/field_b must be
# variable names present in INTERP_FILE (e.g. "rho", "vp", "vs", "vpvs",
# "dens"). color_by is None (uniform colour), "depth" (colour by depth
# km), or another field name. label=None falls back to "field_a-field_b"
# in titles/filenames.
CROSSPLOT_PAIRS = [
    ("rho", "vp", "depth", None),
    ("rho", "dens", "depth", None),
    ("vp", "dens", None, None),
]

# Triples: (field_a, field_b, field_c, color_by, label). Same color_by/
# label conventions as CROSSPLOT_PAIRS.
CROSSPLOT_TRIPLES = [
    ("rho", "vp", "dens", "depth", None),
]

# Fields plotted (and correlated) as log10(field) rather than linear --
# kept in sync with the log10(rho) convention used throughout
# structure.py/plot_joint.py/plot_modem_mesh.py.
LOG_FIELDS = {"rho"}

# Optional (min_km, max_km) depth cutoff applied before plotting/stats;
# None = full grid (safe default, matching the project's policy of not
# guessing at an unverified geographic/depth subset).
DEPTH_RANGE = None

# ---------------------------------------------------------------------------
# Point count / subsampling
# ---------------------------------------------------------------------------

MAX_POINTS = 20000  # display-only subsample cap; UNVERIFIED default, tune to your grid size
RANDOM_SEED = 0     # reproducible subsampling

# ---------------------------------------------------------------------------
# 2-D pair plot styling
# ---------------------------------------------------------------------------

PLOT_KIND_2D = "scatter"  # "scatter" or "hexbin" (density, better for very dense clouds)
HEXBIN_GRIDSIZE = 40      # only used if PLOT_KIND_2D == "hexbin"
HEXBIN_CMAP = "magma"

MARKER_SIZE = 6
MARKER_ALPHA = 0.35   # translucent: point clouds usually overlap heavily
MARKER_COLOR = "steelblue"  # used when color_by is None

ANNOTATE_STATS_2D = True  # Pearson r, Spearman rho, n

# ---------------------------------------------------------------------------
# 3-D triple plot styling
# ---------------------------------------------------------------------------

MARKER_SIZE_3D = 6
MARKER_ALPHA_3D = 0.35
MARKER_COLOR_3D = "steelblue"

ANNOTATE_STATS_3D = True  # pairwise Pearson r for all 3 pairs in the triple
VIEW_ELEV, VIEW_AZIM = 20, -60  # matplotlib 3-D default-ish view angle

# ---------------------------------------------------------------------------
# Shared styling
# ---------------------------------------------------------------------------

CMAP_COLORBY = "viridis"  # colormap when color_by is set (pairs and triples)

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

OUTPUT_DIR = "."
FIG_DPI = 200

# List of matplotlib-supported output extensions, e.g. ["png"] or
# ["png", "pdf"]. save_paris() saves once per format and returns a list
# of paths -- call sites use saved.extend(...), not .append(...).
PLOT_FORMATS = ["png"]

# Also display each figure on screen. Only actually shown if
# matplotlib.is_interactive() is also True (Spyder's Qt/inline backends,
# etc.) -- safe to leave on when running headless on the DIAS HPC cluster
# or from a terminal; same SHOW_PLOTS/_maybe_show() convention as
# plot_joint.py/structure.py.
SHOW_PLOTS = False

# Derived at runtime by main() from INTERP_FILE's name; placeholder here
# only so the module has a defined value if a plotting function is ever
# called without going through main() first.
INTERP_TAG = None


# ===========================================================================
# Interpolation-tag derivation / grid loading (mirrors structure.py)
# ===========================================================================

# ===========================================================================
# Thin wrappers around tomomt's shared helpers, each supplying this
# script's own globals -- same pattern plot_joint.py already uses for
# plotpy (now tomomt).
# ===========================================================================

def _derive_interp_tag(interp_file):
    return tomomt.derive_interp_tag(interp_file)


def _resolve_interp_file():
    return tomomt.resolve_interp_file(SITE_PREFIX, INTERP_FILE)


def _title_suffix():
    return tomomt.title_suffix(SITE_PREFIX, INTERP_TAG)


def _group_label(fields, label):
    return tomomt.group_label(fields, label)


def _maybe_show(fig):
    tomomt.maybe_show(SHOW_PLOTS)


# load_joint_grid() here is kept to the same "joint"-only grid mode as
# structure.py for consistency (one grid convention across the
# pipeline's diagnostic stages), not because cross-plots strictly need
# the coordinate-exact gradients that make it a requirement there; see
# tomomt.load_joint_grid()'s docstring.
load_joint_grid = tomomt.load_joint_grid
get_field = tomomt.get_field


# ===========================================================================
# Flattening / masking / log-transform
# ===========================================================================

def _flat_field(ds, field):
    """
    Return (values_flat, depth_flat, axis_label) for `field`, log10-
    transformed (with non-positive values masked to NaN) if field is in
    LOG_FIELDS, else linear. depth_flat is the depth (km) of each grid
    point, broadcast to the same flattened shape, for color_by="depth"
    and DEPTH_RANGE filtering.
    """
    values, depth, northing, easting = get_field(ds, field)
    depth3d = np.broadcast_to(depth[:, None, None], values.shape)

    if field in LOG_FIELDS:
        n_nonpositive = int(np.sum(np.isfinite(values) & (values <= 0)))
        if n_nonpositive:
            print(f"[crossplots] {field}: masking {n_nonpositive} non-positive "
                  f"value(s) before log10 transform.")
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.where(values > 0, np.log10(values), np.nan)
        label = f"log10({field})"
    else:
        label = field

    return values.ravel(), depth3d.ravel(), label


def _subsample(n_valid, max_points, seed):
    """
    Indices (into a length-n_valid array) to display, subsampled without
    replacement if n_valid > max_points, else all of them. Reproducible
    via `seed`. Returns None if no subsampling is needed (caller should
    then use all points).
    """
    if max_points is None or n_valid <= max_points:
        return None
    rng = np.random.default_rng(seed)
    return rng.choice(n_valid, size=max_points, replace=False)


def _color_array(ds, color_by, mask, depth_flat):
    """Return the (masked) colour array and its axis label for color_by, or (None, None)."""
    if color_by is None:
        return None, None
    if color_by == "depth":
        return depth_flat[mask], "depth (km)"
    c_flat, _, c_label = _flat_field(ds, color_by)
    return c_flat[mask], c_label


# ===========================================================================
# Pair (2-D) cross-plots
# ===========================================================================

def _prepare_pair(ds, field_a, field_b, color_by):
    xa, depth_flat, label_a = _flat_field(ds, field_a)
    xb, _, label_b = _flat_field(ds, field_b)

    mask = np.isfinite(xa) & np.isfinite(xb)
    if DEPTH_RANGE is not None:
        lo, hi = DEPTH_RANGE
        mask &= (depth_flat >= lo) & (depth_flat <= hi)

    x, y, depth_m = xa[mask], xb[mask], depth_flat[mask]
    color, color_label = _color_array(ds, color_by, mask, depth_flat)

    return dict(x=x, y=y, label_a=label_a, label_b=label_b,
                color=color, color_label=color_label, n=len(x))


def _pearson_spearman(x, y):
    if len(x) < 2:
        return np.nan, np.nan
    r_p = stats.pearsonr(x, y).statistic
    r_s = stats.spearmanr(x, y).statistic
    return r_p, r_s


def plot_pair(ds, field_a, field_b, color_by, group_label):
    """
    2-D cross-plot (scatter or hexbin, see PLOT_KIND_2D) of field_a vs.
    field_b's grid-point values, optionally colour-coded, with Pearson r
    / Spearman rho / n annotated (computed on the full valid point set,
    not the display subsample -- see module docstring).
    """
    d = _prepare_pair(ds, field_a, field_b, color_by)
    x, y = d["x"], d["y"]

    fig, ax = plt.subplots(figsize=(6.5, 6))

    if PLOT_KIND_2D == "hexbin":
        hb = ax.hexbin(x, y, gridsize=HEXBIN_GRIDSIZE, cmap=HEXBIN_CMAP,
                       mincnt=1, zorder=5)
        cb = fig.colorbar(hb, ax=ax, shrink=0.85)
        cb.set_label("point count")
    else:
        idx = _subsample(d["n"], MAX_POINTS, RANDOM_SEED)
        xs, ys = (x[idx], y[idx]) if idx is not None else (x, y)
        if d["color"] is not None:
            cs = d["color"][idx] if idx is not None else d["color"]
            sc = ax.scatter(xs, ys, c=cs, cmap=CMAP_COLORBY, s=MARKER_SIZE,
                            alpha=MARKER_ALPHA, linewidths=0, zorder=5)
            cb = fig.colorbar(sc, ax=ax, shrink=0.85)
            cb.set_label(d["color_label"])
        else:
            ax.scatter(xs, ys, c=MARKER_COLOR, s=MARKER_SIZE,
                      alpha=MARKER_ALPHA, linewidths=0, zorder=5)
        if idx is not None:
            ax.text(0.02, 0.02, f"showing {len(idx)}/{d['n']} points",
                   transform=ax.transAxes, ha="left", va="bottom",
                   fontsize=8, zorder=20)

    if ANNOTATE_STATS_2D:
        r_p, r_s = _pearson_spearman(x, y)
        ax.text(0.02, 0.98, f"r = {r_p:.3f}\n$\\rho_s$ = {r_s:.3f}\nn = {d['n']}",
               transform=ax.transAxes, ha="left", va="top", fontsize=9,
               bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85),
               zorder=20)

    ax.set_xlabel(d["label_a"])
    ax.set_ylabel(d["label_b"])

    suffix = ", ".join(_title_suffix())
    ax.set_title(f"{group_label} cross-plot ({suffix})")

    fig.tight_layout()

    tag = f"_by_{color_by}" if color_by else ""
    stem = f"{SITE_PREFIX}_{group_label}_{INTERP_TAG}_crossplot2d{tag}"
    return fig, stem


# ===========================================================================
# Triple (3-D) cross-plots
# ===========================================================================

def _prepare_triple(ds, field_a, field_b, field_c, color_by):
    xa, depth_flat, label_a = _flat_field(ds, field_a)
    xb, _, label_b = _flat_field(ds, field_b)
    xc, _, label_c = _flat_field(ds, field_c)

    mask = np.isfinite(xa) & np.isfinite(xb) & np.isfinite(xc)
    if DEPTH_RANGE is not None:
        lo, hi = DEPTH_RANGE
        mask &= (depth_flat >= lo) & (depth_flat <= hi)

    x, y, z = xa[mask], xb[mask], xc[mask]
    color, color_label = _color_array(ds, color_by, mask, depth_flat)

    return dict(x=x, y=y, z=z, label_a=label_a, label_b=label_b, label_c=label_c,
                color=color, color_label=color_label, n=len(x))


def plot_triple(ds, field_a, field_b, field_c, color_by, group_label):
    """
    3-D cross-plot of field_a/field_b/field_c's grid-point values,
    optionally colour-coded, with the three pairwise Pearson r's among
    the triple annotated (full valid point set, not the display
    subsample).
    """
    d = _prepare_triple(ds, field_a, field_b, field_c, color_by)
    x, y, z = d["x"], d["y"], d["z"]

    idx = _subsample(d["n"], MAX_POINTS, RANDOM_SEED)
    xs, ys, zs = (x[idx], y[idx], z[idx]) if idx is not None else (x, y, z)

    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(projection="3d")
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

    if d["color"] is not None:
        cs = d["color"][idx] if idx is not None else d["color"]
        sc = ax.scatter(xs, ys, zs, c=cs, cmap=CMAP_COLORBY, s=MARKER_SIZE_3D,
                        alpha=MARKER_ALPHA_3D, linewidths=0)
        cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.1)
        cb.set_label(d["color_label"])
    else:
        ax.scatter(xs, ys, zs, c=MARKER_COLOR_3D, s=MARKER_SIZE_3D,
                  alpha=MARKER_ALPHA_3D, linewidths=0)

    ax.set_xlabel(d["label_a"])
    ax.set_ylabel(d["label_b"])
    ax.set_zlabel(d["label_c"])

    if ANNOTATE_STATS_3D:
        r_ab, _ = _pearson_spearman(x, y)
        r_ac, _ = _pearson_spearman(x, z)
        r_bc, _ = _pearson_spearman(y, z)
        stats_txt = (
            f"r({field_a},{field_b}) = {r_ab:.3f}\n"
            f"r({field_a},{field_c}) = {r_ac:.3f}\n"
            f"r({field_b},{field_c}) = {r_bc:.3f}\n"
            f"n = {d['n']}"
        )
        if idx is not None:
            stats_txt += f"\nshowing {len(idx)}/{d['n']} points"
        # Bottom-left, not top-left: the 3-D axes' title placement varies
        # with VIEW_ELEV/VIEW_AZIM and can otherwise run into a top-corner
        # annotation.
        fig.text(0.02, 0.02, stats_txt, ha="left", va="bottom", fontsize=9,
                 bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85))
    elif idx is not None:
        fig.text(0.02, 0.02, f"showing {len(idx)}/{d['n']} points",
                 ha="left", va="bottom", fontsize=8)

    suffix = ", ".join(_title_suffix())
    ax.set_title(f"{group_label} cross-plot ({suffix})")

    fig.tight_layout()

    tag = f"_by_{color_by}" if color_by else ""
    stem = f"{SITE_PREFIX}_{group_label}_{INTERP_TAG}_crossplot3d{tag}"
    return fig, stem


# ===========================================================================
# Delivery helpers (mirrors structure.py)
# ===========================================================================

def save_paris(fig, stem, outdir):
    """
    Save `fig` once per format in PLOT_FORMATS, each with its mtime set
    to now in Europe/Paris local time. Shows the figure first if
    SHOW_PLOTS (see _maybe_show). Returns a list of the saved paths --
    call sites use ``saved.extend(save_paris(...))``, not ``.append(...)``.
    """
    _maybe_show(fig)
    return tomomt.save_paris(fig, stem, outdir, PLOT_FORMATS, FIG_DPI)


def zip_outputs(paths, project_name="crossplots"):
    return tomomt.zip_outputs(paths, project_name, OUTPUT_DIR)


# ===========================================================================
# Main
# ===========================================================================

def main():
    global INTERP_TAG

    interp_file = _resolve_interp_file()
    INTERP_TAG = _derive_interp_tag(interp_file)

    ds = load_joint_grid(interp_file)
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    saved = []

    for field_a, field_b, color_by, label in CROSSPLOT_PAIRS:
        group = _group_label((field_a, field_b), label)
        print(f"[crossplots] {group}: 2-D cross-plot...")
        fig, stem = plot_pair(ds, field_a, field_b, color_by, group)
        saved.extend(save_paris(fig, stem, outdir))

    for field_a, field_b, field_c, color_by, label in CROSSPLOT_TRIPLES:
        group = _group_label((field_a, field_b, field_c), label)
        print(f"[crossplots] {group}: 3-D cross-plot...")
        fig, stem = plot_triple(ds, field_a, field_b, field_c, color_by, group)
        saved.extend(save_paris(fig, stem, outdir))

    ds.close()

    if saved:
        zpath = zip_outputs(saved, project_name=f"{SITE_PREFIX}_crossplots")
        print(f"[crossplots] wrote {len(saved)} figure(s) -> {zpath}")
    else:
        print("[crossplots] nothing to save (CROSSPLOT_PAIRS and CROSSPLOT_TRIPLES both empty).")


if __name__ == "__main__":
    main()
