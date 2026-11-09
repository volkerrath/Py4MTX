#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_ens_plot_misfit.py -- nRMS histogram of a FEMTIC ensemble

Reads the results .npz written by femtic_ens_post.py and draws the histogram
of the per-member best nRMS: accepted members (including those taken from the
fallback iteration) and rejected ones stacked, the acceptance threshold
(NRMS_MAX) as a dashed line, the median of the accepted members. The table of
all scanned members is stored in the .npz (<prefix>_nrms_table); for results
written before that was the case, the CSV of femtic_ens_post.py (NRMS_CSV) can
be given instead. With a random subsample (ENS_SUBSAMPLE_N in
femtic_ens_post.py) the members that were accepted but not drawn count as
accepted, as before.

Companion scripts: femtic_ens_post.py (calculation), femtic_ens_plot_slices.py
(slice figures), femtic_ens_plot_bh.py (borehole profiles).

Provenance
----------
Author      : Claude (Claude Sonnet 5.5, Anthropic)
Generated   : 2026-11-08
Origin      : the nRMS histogram of femtic_ens_post.py (_plot_nrms_hist,
              MOD_NRMS_HIST / MOD_NRMS_BINS); the drawing is now
              femtic_viz.plot_nrms_histogram(), the table I/O
              femtic.read_nrms_table().
Changes     : in the old code the font-size argument of the title was lost
              inside a comment, so the title always had Matplotlib's default
              size; that stays the default, MOD_NRMS_TITLE_FONTSIZE sets it.
              Everything else looks as before.
AI-generated code -- review before production use.

2026-11-09  Claude Sonnet 5.5 (Anthropic)
              New MOD_TITLE: figure title ("auto" = the ensemble prefix, as
              before; None / False / "" = no title; a string with the
              placeholder {prefix}). With the default the figure is unchanged.
              AI-generated; review before production use.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

for _base in [PY4MTX_ROOT + "/py4mt/modules/"]:
    for _p in [Path(_base), *Path(_base).rglob("*")]:
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

import femtic as fem
import femtic_viz as fviz
import util as utl
from version import versionstrg

version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng + "\n\n")

# ===========================================================================
# USER SECTION -- all user-set parameters below are UPPERCASE
# ===========================================================================
# ---------------------------------------------------------------------------
# Input: results of femtic_ens_post.py
# ---------------------------------------------------------------------------
ENSEMBLE_DIR = r"/media/vrath/LargeBack/Ensembles/annecy2026/RTO/ann_rto_orig-err/"

#: Prefix of femtic_ens_post.py's ENSEMBLE_PREFIX (including the suffix it adds
#: for ENS_MODEL_SOURCE = "perturbed" or for a random subsample). It names the
#: results file and the figure.
ENSEMBLE_PREFIX = "annecy_rto_orig-err"

#: Where femtic_ens_post.py wrote its results; the figure goes here too.
RESULTS_DIR = os.path.join(ENSEMBLE_DIR, "results") + os.sep

#: The results .npz (same name rule as femtic_ens_post.py).
ENSEMBLE_RESULTS = RESULTS_DIR + ENSEMBLE_PREFIX+ "_results.npz"
#.upper() + "_results256.npz"

#: Key prefix inside the .npz. None = detect (the file's only ensemble).
NPZ_PREFIX = None

#: Fallback for results files without the nRMS table: the CSV written by
#: femtic_ens_post.py (NRMS_CSV), e.g. RESULTS_DIR + ENSEMBLE_PREFIX +
#: "_nrms.csv". None = no fallback. Used only if the .npz holds no table.
NRMS_CSV = None

#: Acceptance threshold marked in the plot. None = the value stored in the .npz
#: (femtic_ens_post.py NRMS_MAX); if the file has none, no line is drawn.
NRMS_MAX = None

# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------
#: Bins: an int, or any rule numpy.histogram_bin_edges accepts ("auto", "fd",
#: "sturges", ...).
MOD_NRMS_BINS = "auto"

#: Extension-less output path; one file per MOD_PLOT_FORMAT entry. None =
#: RESULTS_DIR/<prefix>_nrms_hist.
MOD_NRMS_HIST_FILE = None

# ---------------------------------------------------------------------------
# Figure output format
# ---------------------------------------------------------------------------
#: One or more file extensions. Every QC / MOD_STATS figure is saved once
#: per entry (slice geometry resolved once and reused; the figure itself
#: is still rebuilt per format, since plot_model_slices doesn't expose a
#: way to re-save an already-built figure), so e.g. ["pdf", "jpg"] writes
#: both a vector version for print and a raster preview in one run. A bare
#: string ("pdf") also works and is treated as a single-entry list.
#: MOD_QC_FILE / MOD_STATS_DIR filenames below are built without an
#: extension; _plot_slice() appends each MOD_PLOT_FORMAT entry in turn.
#: Supported values (Matplotlib Agg-backend savefig formats):
#:   "pdf"  -- vector, multi-page-safe, default; best for print/publication
#:   "svg"  -- vector, editable in Illustrator/Inkscape
#:   "eps"  -- vector, legacy PostScript (no transparency support)
#:   "png"  -- raster, lossless, transparency-capable; good for slides/web
#:   "jpg" / "jpeg" -- raster, lossy, no transparency; small file size
#:   "tif" / "tiff"  -- raster, lossless, common in publication workflows
#:   "webp" -- raster, modern lossy/lossless web format
#: Raster formats (png/jpg/tif/webp) are rendered at MOD_DPI; vector
#: formats (pdf/svg/eps) are resolution-independent and MOD_DPI is ignored
#: except for any embedded raster elements (e.g. rasterized markers).
MOD_PLOT_FORMAT = ["pdf",] # "jpg"]

#: Normalised to a list regardless of whether a bare string or a list was
#: set above.
_MOD_PLOT_FORMATS = (
    [MOD_PLOT_FORMAT]
    if isinstance(MOD_PLOT_FORMAT, str)
    else list(MOD_PLOT_FORMAT)
)

MOD_DPI = 400  # figure DPI
#: Axis annotation font sizes, and the title size (None = Matplotlib's
#: default, as in the old femtic_ens_post.py).
MOD_TICK_FONTSIZE = 16
MOD_LABEL_FONTSIZE = 16
MOD_NRMS_TITLE_FONTSIZE = None
#: Figure title. "auto" = the ensemble prefix (as before); None, False or "" =
#: no title; a string is used as given ({prefix} = ENSEMBLE_PREFIX as stored in
#: the results file; text with other braces is used as it stands).
MOD_TITLE = "auto"

#: When True (default) and this script is running inside Spyder, the figure is
#: also displayed inline via plt.show(). Has no effect outside Spyder.
MOD_SHOW_IN_SPYDER = True

# ---------------------------------------------------------------------------
# Verbose output
# ---------------------------------------------------------------------------
OUT = True

_IN_SPYDER = utl.runtime_env() == "spyder"
_SHOW_PLOTS = MOD_SHOW_IN_SPYDER and _IN_SPYDER

# ===========================================================================
# Main
# ===========================================================================
try:
    res = fem.read_ensemble_results(ENSEMBLE_RESULTS, NPZ_PREFIX, out=OUT)
except (FileNotFoundError, KeyError, ValueError) as _exc:
    sys.exit(f"femtic_ens_plot_misfit.py: {_exc}")

P = res["prefix"]
rows = res["nrms_table"]
if rows is None:
    if NRMS_CSV and os.path.isfile(NRMS_CSV):
        rows = fem.read_nrms_table(NRMS_CSV)
        print(f"  nRMS table read from {NRMS_CSV}")
    else:
        # last resort: the accepted members the ensemble itself holds
        rows = [
            (os.path.basename(os.path.dirname(m[0])), m[1], m[2], "accepted")
            for m in res["model_list"]
        ]
        print(
            "  the results file holds no nRMS table (written by an older "
            "femtic_ens_post.py) - only the accepted members are shown. Set "
            "NRMS_CSV to get the rejected ones too."
        )

_nrms_max = NRMS_MAX if NRMS_MAX is not None else res["nrms_max"]
_base = MOD_NRMS_HIST_FILE or os.path.join(RESULTS_DIR, P + "_nrms_hist")
os.makedirs(os.path.dirname(_base) or ".", exist_ok=True)

def _figure_title():
    """Figure title from MOD_TITLE: a string, or None for no title."""
    if MOD_TITLE is None or MOD_TITLE is False:
        return None
    if MOD_TITLE is True:
        return P
    _t = str(MOD_TITLE)
    if _t.strip() == "":
        return None
    if _t.strip().lower() == "auto":
        return P
    try:
        return _t.format(prefix=P)
    except (KeyError, IndexError, ValueError, AttributeError) as _exc:
        print(f"  MOD_TITLE: cannot fill in {_t!r} ({_exc!r}) - used as it stands.")
        return _t


fviz.plot_nrms_histogram(
    rows,
    _base,
    nrms_max=_nrms_max,
    bins=MOD_NRMS_BINS,
    title=_figure_title(),
    tick_fontsize=MOD_TICK_FONTSIZE,
    label_fontsize=MOD_LABEL_FONTSIZE,
    title_fontsize=MOD_NRMS_TITLE_FONTSIZE,
    dpi=MOD_DPI,
    formats=_MOD_PLOT_FORMATS,
    show=_SHOW_PLOTS,
    out=OUT,
)

print("\nfemtic_ens_plot_misfit.py complete.")

# ---------------------------------------------------------------------------
# Parameter summary
# ---------------------------------------------------------------------------
_summary_path = utl.write_param_summary(__file__, out_dir=RESULTS_DIR)
