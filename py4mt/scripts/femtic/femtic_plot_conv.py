#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_plot_conv.py — Plot any femtic.cnv field(s) vs. iteration/row index
for one or more FEMTIC inversion run directories.

Generalises femtic_lcurve_plot.py's single alpha/roughness/misfit-vs-nRMS
scatter and femtic_ens_repair.py's/femtic_ens_post.py's per-run
best-nRMS-only scan into a full convergence-history line plot over
*any* column femtic.cnv happens to carry (Iter#, Retrial#, Alpha, Beta,
Damp, Roughness, Distortion, Misfit, RMS, ObjFunc, or the ABIC/
cross-gradient extension columns LmdCG/CrossGra/MupdateMean/ABIC) —
selected purely by name in PLOT_FIELDS, with no version/column-count
switch anywhere in this script.

@author: vrath

Provenance:
    2026-09-07  Claude Sonnet 5 (Anthropic)
                Created. Uses fem.read_cnv(source, columnar=True) and
                reads exclusively from its "data" dict (the optional,
                columnar=True-only entry: column name -> full 1-D array
                over the run's convergence history) for every quantity
                plotted here, rather than re-deriving arrays by looping
                over "rows" by hand -- this script's whole purpose is
                showing entire per-iteration columns, which is exactly
                what the columnar output was added for. One or more run
                directories are supported (RUN_DIRS explicit list, or a
                WORK_DIR + SEARCH_STRNG glob when RUN_DIRS is None); a
                single directory is just a one-element case of the same
                path. Retrials (multiple rows sharing one Iter#) are
                plotted against a running row index (0, 1, 2, ...), not
                Iter# itself, so every row is visible rather than several
                rows overplotting at the same x position.
"""

import os
import sys
from pathlib import Path
import inspect

import numpy as np
import matplotlib.pyplot as plt

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

for _base in [PY4MTX_ROOT + "/py4mt/modules/"]:
    for _p in [Path(_base), *Path(_base).rglob("*")]:
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

import femtic as fem
import util as utl
from version import versionstrg

version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())
titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng + "\n\n")

# ===========================================================================
# Configuration
# ===========================================================================

# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------
WORK_DIR = r"/media/vrath/LargeBack/Ensembles/annecy2026/lcurve/"

#: Explicit list of run directories (relative to WORK_DIR, or absolute).
#: A single-element list plots just one run. Set to None to instead
#: discover directories via WORK_DIR + SEARCH_STRNG (glob), matching
#: femtic_lcurve_plot.py's convention.
RUN_DIRS = None

#: Glob pattern used to discover run directories when RUN_DIRS is None.
SEARCH_STRNG = "ann_reg*"

#: One label per RUN_DIRS entry (legend / console messages); None = use
#: each directory's basename. Ignored (basenames used) when RUN_DIRS is
#: None, since the glob order/count isn't known ahead of time.
RUN_LABELS = None

# ---------------------------------------------------------------------------
# Fields to plot
# ---------------------------------------------------------------------------
#: Canonical femtic.cnv column names to plot, one stacked subplot per
#: entry, in this order. Available canonical names (only those actually
#: present in a given run's header will have data): "Iter", "Retrial",
#: "Alpha", "Beta", "Damp", "Roughness", "Distortion", "Misfit", "RMS",
#: "ObjFunc", and — for ABIC/cross-gradient runs — "LmdCG", "CrossGra",
#: "MupdateMean", "ABIC". Matched case-insensitively against each run's
#: own columns (see _match_field below), so "rms"/"Rms"/"RMS" all work.
PLOT_FIELDS = ["RMS", "Roughness", "Misfit"]

#: Subset of PLOT_FIELDS to render with a log10 y-axis.
PLOT_LOG_FIELDS = ["Misfit"]

#: Optional dashed horizontal reference line per field, e.g. an nRMS
#: acceptance threshold: {"RMS": 1.0}. Fields not listed get no line.
PLOT_THRESHOLDS = {"RMS": 1.0}

# ---------------------------------------------------------------------------
# Plot appearance
# ---------------------------------------------------------------------------
PLOT_TITLE = "FEMTIC convergence history"
#: x-axis label. Rows are plotted at a running index (0, 1, 2, ...) over
#: the convergence history so retrials (several rows sharing one Iter#)
#: are all visible, rather than Iter# itself, which would overplot them.
PLOT_XLABEL = "Row index (Iter# / Retrial# sequence)"

FONTSIZE = 10
LEGEND_FONTSIZE = 8
LINEWIDTH = 1.2
MARKERSIZE = 3
MARKER = "o"

#: Per-panel size in cm (width, height); converted to inches internally.
PANEL_SIZE_CM = (14.0, 4.0)

#: Output file base path (no extension).
PLOT_NAME = WORK_DIR + "femtic_conv"
#: One or more file extensions to save.
PLOT_FORMAT = ["pdf", "jpg"]
PLOT_DPI = 200

#: Also save the raw per-run, per-field arrays actually plotted to an
#: .npz file (keys "<label>__<field>"); None = skip.
CONV_DATA_FILE = WORK_DIR + "femtic_conv_data.npz"

#: When running inside Spyder, also display the figure inline via
#: plt.show() in addition to saving it to disk. No effect outside Spyder.
SHOW_IN_SPYDER = True

# ---------------------------------------------------------------------------
# Verbose output
# ---------------------------------------------------------------------------
OUT = True


# ===========================================================================
# Helpers
# ===========================================================================
def _match_field(requested: str, available: dict) -> "str | None":
    """Case-insensitive lookup of `requested` among `available`'s keys
    (a run's fem.read_cnv(...)['data'] dict). Returns the actual key to
    use, or None if no case-insensitive match exists.
    """
    _low = requested.strip().lower()
    for key in available:
        if key.lower() == _low:
            return key
    return None


# ===========================================================================
# (1) Resolve run directories and labels
# ===========================================================================
if RUN_DIRS is not None:
    _dirs = [d if os.path.isabs(d) else os.path.join(WORK_DIR, d)
             for d in RUN_DIRS]
    _labels = (list(RUN_LABELS) if RUN_LABELS is not None
               else [os.path.basename(os.path.normpath(d)) for d in _dirs])
    if len(_labels) != len(_dirs):
        sys.exit(f"RUN_LABELS has {len(_labels)} entries but RUN_DIRS has "
                  f"{len(_dirs)} — they must match.")
else:
    _dirs = utl.get_filelist(
        searchstr=[SEARCH_STRNG], searchpath=WORK_DIR,
        sortedlist=True, fullpath=True,
    )
    _labels = [os.path.basename(os.path.normpath(d)) for d in _dirs]

print(f"Found {len(_dirs)} run director{'y' if len(_dirs) == 1 else 'ies'}.")

# ===========================================================================
# (2) Read femtic.cnv for every run (columnar=True: use the "data" dict
#     for every quantity below, since this script's whole job is showing
#     full per-iteration columns, not single last-row scalars).
# ===========================================================================
_run_data = {}   # label -> cnv["data"] dict
for _d, _label in zip(_dirs, _labels):
    if not os.path.isdir(_d):
        print(f"  {_label}: {_d} is not a directory — skipped.")
        continue
    try:
        _cnv = fem.read_cnv(_d, columnar=True)
    except ValueError as _e:
        print(f"  {_label}: {_e}")
        continue
    if not _cnv["rows"]:
        print(f"  {_label}: femtic.cnv is empty — skipped.")
        continue
    _run_data[_label] = _cnv["data"]
    print(f"  {_label}: {len(_cnv['rows'])} rows, "
          f"columns = {list(_cnv['columns'])}")

if not _run_data:
    sys.exit("No usable femtic.cnv found in any run directory. Exit.")

# ===========================================================================
# (3) Resolve requested fields per run (case-insensitive), collect data
#     to plot and to save.
# ===========================================================================
_panel_series = {field: {} for field in PLOT_FIELDS}   # field -> {label: array}
_save_dict = {}

for _label, _data in _run_data.items():
    for _field in PLOT_FIELDS:
        _key = _match_field(_field, _data)
        if _key is None:
            print(f"  {_label}: field '{_field}' not present in this run's "
                  f"femtic.cnv (columns = {list(_data)}) — skipped for "
                  f"this run.")
            continue
        _panel_series[_field][_label] = _data[_key]
        _save_dict[f"{_label}__{_field}"] = _data[_key]

_active_fields = [f for f in PLOT_FIELDS if _panel_series[f]]
if not _active_fields:
    sys.exit("None of PLOT_FIELDS were found in any run's femtic.cnv. Exit.")
if len(_active_fields) < len(PLOT_FIELDS):
    _missing = [f for f in PLOT_FIELDS if f not in _active_fields]
    print(f"\nWARNING: field(s) {_missing} not found in ANY run — "
          f"omitted from the figure entirely.")

if CONV_DATA_FILE is not None:
    np.savez_compressed(CONV_DATA_FILE, **_save_dict)
    print(f"\nSaved raw per-run field arrays to {CONV_DATA_FILE}")

# ===========================================================================
# (4) Plot — one stacked subplot per active field, shared x-axis, one
#     consistently-coloured line per run.
# ===========================================================================
cm = 1 / 2.54
_panel_w_in, _panel_h_in = PANEL_SIZE_CM[0] * cm, PANEL_SIZE_CM[1] * cm
_n_panels = len(_active_fields)

fig, axes = plt.subplots(
    _n_panels, 1, sharex=True,
    figsize=(_panel_w_in, _panel_h_in * _n_panels),
)
if _n_panels == 1:
    axes = [axes]

_color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
_run_color = {label: _color_cycle[i % len(_color_cycle)]
              for i, label in enumerate(_run_data)}

for ax, field in zip(axes, _active_fields):
    for label, values in _panel_series[field].items():
        ax.plot(np.arange(len(values)), values,
                color=_run_color[label], marker=MARKER,
                markersize=MARKERSIZE, linewidth=LINEWIDTH,
                label=label)
    if field in PLOT_LOG_FIELDS:
        ax.set_yscale("log")
    if field in PLOT_THRESHOLDS:
        _thr = PLOT_THRESHOLDS[field]
        ax.axhline(_thr, color="black", linestyle="dashed", linewidth=1.0)
        ax.text(0.995, _thr, f"{field}_MAX={_thr:g}",
                transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=LEGEND_FONTSIZE)
    ax.set_ylabel(field, fontsize=FONTSIZE)
    ax.tick_params(labelsize=FONTSIZE - 1)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel(PLOT_XLABEL, fontsize=FONTSIZE)
fig.suptitle(PLOT_TITLE, fontsize=FONTSIZE + 1)

_handles, _leg_labels = axes[0].get_legend_handles_labels()
fig.legend(_handles, _leg_labels, loc="lower center",
           ncol=min(len(_leg_labels), 4), fontsize=LEGEND_FONTSIZE,
           bbox_to_anchor=(0.5, -0.02 / _n_panels))

fig.tight_layout(rect=[0, 0.02, 1, 0.98])

_IN_SPYDER = (utl.runtime_env() == "spyder")
_SHOW_PLOTS = SHOW_IN_SPYDER and _IN_SPYDER

_formats = [PLOT_FORMAT] if isinstance(PLOT_FORMAT, str) else list(PLOT_FORMAT)
for _fmt in _formats:
    _out_file = f"{PLOT_NAME}.{_fmt}"
    fig.savefig(_out_file, dpi=PLOT_DPI,
                transparent=(_fmt in ("png", "jpg", "jpeg")))
    print(f"Saved {_out_file}")

if _SHOW_PLOTS:
    plt.show()
plt.close(fig)


# ---------------------------------------------------------------------------
# Parameter summary
# ---------------------------------------------------------------------------
utl.write_param_summary(fname)
