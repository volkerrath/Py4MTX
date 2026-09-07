# femtic_plot_conv.py

Plot any `femtic.cnv` field(s) vs. iteration/row index, for one or more
FEMTIC inversion run directories.

---

## Purpose

`femtic_lcurve_plot.py` plots one fixed pair (roughness vs. misfit/nRMS,
last iteration only, one point per run). `femtic_ens_post.py` /
`femtic_ens_repair.py` only ever look at the single best- or
final-iteration nRMS per run. `femtic_plot_conv.py` fills the remaining
gap: the **full convergence history**, for **any** column `femtic.cnv`
happens to carry, overlaid across **one or more** run directories.

Column selection is purely by name (`PLOT_FIELDS`) — there is no
version/column-count switch anywhere in this script. Every field is read
straight from `fem.read_cnv(source, columnar=True)`'s `"data"` dict (the
optional, `columnar=True`-only entry mapping each column name to its
full 1-D array over the run's convergence history), which is exactly
what that option was added for.

---

## Workflow

```
RUN_DIRS (explicit list)  or  WORK_DIR + SEARCH_STRNG (glob)
        │
        ▼  fem.read_cnv(run_dir, columnar=True)   — per run
   run_data[label] = cnv["data"]   (column name -> full array)
        │
        ▼  case-insensitive match of each PLOT_FIELDS entry
   panel_series[field][label] = array   (skipped per-run if that run's
                                          header doesn't have the field,
                                          e.g. Beta/Distortion absent)
        │
        ▼  one stacked subplot per field, shared x-axis (row index),
           one consistently-coloured line per run
PLOT_NAME.<fmt>   (one file per PLOT_FORMAT entry)
femtic_conv_data.npz   (raw per-run, per-field arrays actually plotted)
```

---

## Column identification

Column positions are read from each run's own `femtic.cnv` header row via
`fem.read_cnv()` (case-insensitive substring match, e.g. `"rough"` ->
`"Roughness"`, `"rms"` -> `"RMS"`), so this is robust to FEMTIC version
and to whether the `Beta`/`Distortion` (or ABIC/cross-gradient
`LmdCG`/`CrossGra`/`MupdateMean`/`ABIC`) columns are present. `PLOT_FIELDS`
entries are then matched to a run's actual column names case-insensitively
(`"rms"`, `"Rms"`, `"RMS"` all resolve the same way) — see
`femtic_readme.md`'s `read_cnv()` changelog for the full column-matching
design.

If a requested field is missing from a particular run's header (e.g.
`"Distortion"` requested but that run didn't invert distortion
parameters), that run is simply omitted from that field's panel, with a
console message — it does not stop the rest of the plot. If a field is
missing from **every** run, that panel is dropped from the figure
entirely, with a warning.

---

## Retrials and the x-axis

FEMTIC's `femtic.cnv` can contain more than one row per `Iter#` (a
retrial after a rejected step, `Retrial# > 0`). Plotting against `Iter#`
directly would overplot those rows at the same x position. This script
instead plots every row against a running index (`0, 1, 2, ...`) over the
convergence history, so retrials remain individually visible; the x-axis
label reflects this (`PLOT_XLABEL`, default `"Row index (Iter# / Retrial#
sequence)"`).

---

## Configuration reference

### Run discovery

| Variable | Default | Description |
|---|---|---|
| `WORK_DIR` | — | Base directory |
| `RUN_DIRS` | `None` | Explicit list of run directories (relative to `WORK_DIR`, or absolute). A single-element list plots just one run. `None` → discover via `WORK_DIR` + `SEARCH_STRNG` glob instead. |
| `SEARCH_STRNG` | `"ann_reg*"` | Glob pattern used when `RUN_DIRS is None`, matching `femtic_lcurve_plot.py`'s convention. |
| `RUN_LABELS` | `None` | One label per `RUN_DIRS` entry (legend / console messages); `None` = each directory's basename. Ignored (basenames used) when `RUN_DIRS is None`. |

### Fields

| Variable | Default | Description |
|---|---|---|
| `PLOT_FIELDS` | `["RMS", "Roughness", "Misfit"]` | Canonical `femtic.cnv` column names to plot, one stacked subplot per entry, in this order. Available canonical names: `"Iter"`, `"Retrial"`, `"Alpha"`, `"Beta"`, `"Damp"`, `"Roughness"`, `"Distortion"`, `"Misfit"`, `"RMS"`, `"ObjFunc"`, plus (ABIC/cross-gradient runs) `"LmdCG"`, `"CrossGra"`, `"MupdateMean"`, `"ABIC"`. Matched case-insensitively. |
| `PLOT_LOG_FIELDS` | `["Misfit"]` | Subset of `PLOT_FIELDS` rendered with a log10 y-axis. |
| `PLOT_THRESHOLDS` | `{"RMS": 1.0}` | Optional dashed horizontal reference line per field, e.g. an nRMS acceptance threshold. Fields not listed get no line. |

### Plot appearance

| Variable | Default | Description |
|---|---|---|
| `PLOT_TITLE` | `"FEMTIC convergence history"` | Figure super-title |
| `PLOT_XLABEL` | `"Row index (Iter# / Retrial# sequence)"` | x-axis label (bottom panel only) |
| `FONTSIZE` | `10` | Axis label / tick font size |
| `LEGEND_FONTSIZE` | `8` | Legend and threshold-label font size |
| `LINEWIDTH` | `1.2` | Line width |
| `MARKERSIZE` | `3` | Marker size |
| `MARKER` | `"o"` | Marker style |
| `PANEL_SIZE_CM` | `(14.0, 4.0)` | Per-panel `(width, height)` in cm; converted to inches internally |

### Output

| Variable | Default | Description |
|---|---|---|
| `PLOT_NAME` | `WORK_DIR + "femtic_conv"` | Output file base path (no extension) |
| `PLOT_FORMAT` | `["pdf", "jpg"]` | One or more file extensions to save |
| `PLOT_DPI` | `200` | Raster format DPI |
| `CONV_DATA_FILE` | `WORK_DIR + "femtic_conv_data.npz"` | Raw per-run, per-field arrays actually plotted (keys `"<label>__<field>"`); `None` = skip |
| `SHOW_IN_SPYDER` | `True` | When running inside Spyder, also display the figure inline via `plt.show()`, in addition to saving to disk. No effect outside Spyder. |

### Verbose output

| Variable | Default | Description |
|---|---|---|
| `OUT` | `True` | Print progress messages throughout |

---

## Output files

- `<PLOT_NAME>.<fmt>` — one convergence figure per `PLOT_FORMAT` entry.
- `<CONV_DATA_FILE>` — `.npz` with one array per `"<label>__<field>"` key actually plotted (skipped if `CONV_DATA_FILE is None`).
- `femtic_plot_conv_summary.md` — user-set (UPPERCASE) parameters, script path, and run date/time (via `utl.write_param_summary()`).

---

## Quick start

```python
# Single run:
RUN_DIRS = ["my_inversion_run"]

# Several runs overlaid:
RUN_DIRS = ["run_a", "run_b", "run_c"]
RUN_LABELS = ["A", "B", "C"]

# Or, discover automatically:
RUN_DIRS = None
SEARCH_STRNG = "misti_gst_suzuki_rnd*"

PLOT_FIELDS = ["RMS", "Roughness", "Alpha"]
PLOT_LOG_FIELDS = []
PLOT_THRESHOLDS = {"RMS": 1.4}
```

```
python femtic_plot_conv.py
```

---

## Relationship to other scripts

- **`femtic_lcurve_plot.py`** — one point per run (final iteration only),
  roughness vs. misfit/nRMS, annotated by α; for regularisation-parameter
  studies. `femtic_plot_conv.py` instead shows the whole convergence
  history of any field(s), for iteration-by-iteration diagnosis of a
  single run or a handful of runs.
- **`femtic_ens_post.py`** / **`femtic_ens_repair.py`** — only the
  best/final-iteration nRMS scalar per run, used to decide
  accepted/rejected status across a whole ensemble. `femtic_plot_conv.py`
  does not filter or classify runs at all — it just plots whatever
  `PLOT_FIELDS` says, for whichever `RUN_DIRS` say, unconditionally.
- **`femtic.py`** — `read_cnv()` supplies both the per-row `"rows"` list
  (used by the other scripts above) and the whole-column `"data"` dict
  (`columnar=True`) that this script relies on exclusively.

---

## Dependencies

| Module | Source | Purpose |
|---|---|---|
| `numpy` | PyPI | Array operations, `.npz` save |
| `matplotlib` | PyPI | Plotting |
| `femtic` | Py4MTX | `read_cnv()` |
| `util` | Py4MTX | `get_filelist`, `print_title`, `runtime_env`, `write_param_summary` |
| `version` | Py4MTX | Version string for title banner |

---

## Provenance

| Date | Author | Note |
|---|---|---|
| 2026-09-07 | Claude Sonnet 5 (Anthropic) | Created. |

Author: Volker Rath (DIAS)
