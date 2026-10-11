# `crossplots.py` — Simple 2-D/3-D Cross-Plots of Property Pairs and Triples

New pipeline stage, parallel to `structure.py`/`plot_joint.py`/`cluster.py`:

```
precompute.py -> interpolate.py -> {SITE_PREFIX}_interp_<method>.nc
                                                 |
                                    crossplots.py -> figures
```

It reads the same `INTERP_FILE` NetCDF as the rest of the plotting
pipeline and is equally agnostic to which interpolation method
(`krig`/`rbf`/`idw`/`nn`) produced it.

**How this differs from `structure.py`:** `structure.py` compares two or
three fields' *gradients* to diagnose structural coupling (cross-gradient,
cosine similarity, Gramian, ...). `crossplots.py` plots raw grid-point
*values* of two or three fields against each other, with no
gradient/structural machinery at all — the standard first-look
exploratory technique for spotting petrophysical relationships, clusters,
or outliers between co-located properties (e.g. "does low resistivity
line up with low Vp here?"), independent of the assumptions either the
cross-gradient family or `cluster.py`'s clustering methods make.

## What it plots

- **Pairs** (`CROSSPLOT_PAIRS`) — 2-D scatter (or hexbin density, for very
  dense point clouds) of field_a vs. field_b, every valid grid point
  pooled across depth/northing/easting. Optionally colour-coded by a
  third field or by depth. Annotated with Pearson *r*, Spearman *ρ*, and
  the point count *n*.
- **Triples** (`CROSSPLOT_TRIPLES`) — 3-D scatter of field_a/field_b/
  field_c, optionally colour-coded by a fourth field or by depth.
  Annotated with the three pairwise Pearson *r*'s among the triple.

Both are "simple" cross-plots deliberately: no fitted trend line, no
petrophysical model, no clustering — those already live in `cluster.py`
(clustering) and, for structural rather than value comparisons, in
`structure.py`.

## Log fields and correlation statistics

Fields in `LOG_FIELDS` (default `{"rho"}`, matching the `log10(rho)`
convention used throughout the pipeline) are plotted, labelled, *and
correlated* as `log10(field)` rather than the linear value — statistics
are always computed on whatever representation is actually on the axis,
never silently on the raw linear values underneath a log-labelled axis.
Non-positive values in a log field are masked (and the count logged to
the console), not clipped.

## Point count and subsampling

Joint grids can have far more points than are useful to render
individually. If a plot's valid point count exceeds `MAX_POINTS`
(default `20000`), a reproducible random subsample (seeded by
`RANDOM_SEED`) is drawn **for display only** — the annotated statistics
are always computed from the *full* valid point set, so the numbers on
the plot don't change with how many points happened to get drawn. A
"showing N/M points" note appears on the figure whenever subsampling
kicked in. `MAX_POINTS` has no literature- or pipeline-verified "right"
value for this project's grids; it's a flagged free parameter, not a
tuned default — check it against your actual grid sizes.

## Configuration

- `SITE_PREFIX` — your site's short code (e.g. `"site_a"`), must match `interpolate.py`.
- `INTERP_FILE` — explicit path, or `None` to auto-pick the newest
  `{SITE_PREFIX}_interp_*.nc` in the current directory.
- `CROSSPLOT_PAIRS` — list of `(field_a, field_b, color_by, label)`
  tuples. `color_by` is `None` (uniform colour), `"depth"`, or another
  field name. `label=None` falls back to `"field_a-field_b"`.
- `CROSSPLOT_TRIPLES` — list of `(field_a, field_b, field_c, color_by,
  label)` tuples, same `color_by`/`label` conventions.
- `LOG_FIELDS` — fields plotted/correlated as `log10(field)`; default
  `{"rho"}`.
- `DEPTH_RANGE` — optional `(min_km, max_km)` cutoff applied before
  plotting/stats; `None` (default) uses the full grid, matching the
  project's safe-default policy for unverified geographic/depth subsets.
- `MAX_POINTS` / `RANDOM_SEED` — display subsample cap and its seed; see
  above.
- `PLOT_KIND_2D` — `"scatter"` (default) or `"hexbin"` (point-count
  density, better once a scatter gets too dense to read); `HEXBIN_GRIDSIZE`
  / `HEXBIN_CMAP` control the hexbin variant.
- `MARKER_SIZE` / `MARKER_ALPHA` / `MARKER_COLOR` and the `_3D` variants
  — point styling for the uncoloured case; `CMAP_COLORBY` is used
  whenever `color_by` is set (pairs and triples alike).
- `ANNOTATE_STATS_2D` / `ANNOTATE_STATS_3D` — toggle the correlation
  text boxes.
- `VIEW_ELEV` / `VIEW_AZIM` — initial 3-D view angle for triples (static
  PNG/PDF output, not an interactive rotatable plot).
- `PLOT_FORMATS` — list of matplotlib-supported output extensions, e.g.
  `["png"]` or `["png", "pdf"]`.
- `SHOW_PLOTS` — also display each figure on screen; only actually shows
  if `matplotlib.is_interactive()` is also `True` (Spyder's Qt/inline
  backends), so it's safe to leave on when running headless on the DIAS
  HPC cluster or from a terminal — same `SHOW_PLOTS`/`_maybe_show()`
  convention as `plot_joint.py`/`structure.py`. Defaults `False`.

## Grid support: `"joint"` only

Like `structure.py`, this script requires `INTERP_FILE` to be a
`TARGET_GRID="joint"` file (1-D depth/northing/easting coordinates), and
raises a clear `ValueError` otherwise. Cross-plots don't strictly need
the coordinate-exact gradients that make `structure.py` "joint"-only, but
this script is kept to the same grid convention for consistency across
the pipeline's diagnostic stages, and because it isn't written to
flatten a `"seismic"`-mode file's 2-D auxiliary coordinates correctly.

## Known limitation

No spatial filtering beyond the optional `DEPTH_RANGE` cutoff is
implemented — e.g. no polygon/mask clipping to a study sub-area. Every
valid grid point in `INTERP_FILE` (within `DEPTH_RANGE`, if set) is
pooled. Adding a spatial mask would be a straightforward filter on top
of the flattened arrays built by `_prepare_pair()`/`_prepare_triple()`,
not attempted here since no specific sub-area was requested.

## A note on the code itself

`crossplots.py` duplicates (rather than imports) `_resolve_interp_file()`,
`_derive_interp_tag()`, `load_joint_grid()`, and `get_field()` from
`structure.py`. That's intentional, matching this project's existing
per-script self-containment pattern — each plotting script in the
pipeline (`structure.py`, `plot_joint.py`, `cluster.py`, ...) owns its
own copy of this small grid-loading boilerplate; only `tomomt.py` and
`modem.py` are actual shared modules. Not introducing a new shared
import here keeps this script consistent with that pattern rather than
being a one-off exception.

## Output

Files named:
- `{SITE_PREFIX}_{group}_{INTERP_TAG}_crossplot2d[_by_<color_by>].<ext>`
  for pairs,
- `{SITE_PREFIX}_{group}_{INTERP_TAG}_crossplot3d[_by_<color_by>].<ext>`
  for triples,

where `group` is `field_a-field_b` (or `field_a-field_b-field_c`) unless
overridden by `label`, and `<ext>` is emitted once per entry in
`PLOT_FORMATS`. Titled with the field group and `[SITE_PREFIX,
INTERP_TAG]` — same suffix convention as the rest of the pipeline. All
outputs for a run are bundled into
`{SITE_PREFIX}_crossplots_YYYYMMDD_HHMM.zip` (Europe/Paris timestamp,
both the zip name and each member's internal mtime), matching the rest
of the pipeline's delivery convention.

## Verification performed

`ast.parse` was run on the delivered file. Functional smoke tests were
run against a synthetic joint-grid dataset (not real project data):
`log10()` transform and axis labelling for `LOG_FIELDS`, colour-by-depth
and colour-by-field paths, the uncoloured path, `hexbin` mode, the 3-D
triple plot (including the pairwise-*r* annotation, moved to the
bottom-left corner after an initial layout check showed it overlapping
the title in some view angles), forced subsampling with a small
`MAX_POINTS`, and `DEPTH_RANGE` filtering (confirmed it strictly reduces
the point count relative to the unfiltered case).
