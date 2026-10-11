# `tomomt.py` — Shared Helper Module (renamed from `plotpy.py`)

`plotpy.py` is renamed to `tomomt.py`. Same module, broadened scope: a
second consolidation pass pulled genuinely duplicated helper functions
out of `cluster.py`, `crossplots.py`, `interpolate.py`, `plot_joint.py`,
`precompute.py`, and `structure.py` — some duplicated with plotting
primitives already in `plotpy.py`'s original scope, some duplicated with
each other and new to this module. `tomomt.py` is now imported by all six
of those scripts.

`plotpy.py` still exists as a **backward-compatible shim** (`from
tomomt import *`, plus the handful of underscore-prefixed names a star
import skips) — its own docstring says it's also used by
`{SITE_PREFIX}_plot_seis.py`, `{SITE_PREFIX}_plot_modem_image.py`, and
`{SITE_PREFIX}_plot_modem_mesh.py`, which weren't in this bundle. Deleting
`plotpy.py` outright would have silently broken those three on their
next run; the shim means they keep working unmodified. **When those
three scripts are next touched, update them to `import tomomt` directly
and delete `plotpy.py`** — it isn't meant to be a permanent second name
for the same module.

## What moved into `tomomt.py`, and from where

| Helper(s) | Previously duplicated in | Notes |
|---|---|---|
| `safe_to_netcdf`, `safe_open_w` | `cluster.py`, `interpolate.py`, `precompute.py` | Byte-for-byte identical bodies (the read-only-stale-file recovery logic). `safe_open_w` was only in `cluster.py`, moved alongside `safe_to_netcdf` as the same pattern rather than left as a one-off. |
| `resolve_path(directory, name)` | `cluster.py` (`ncpath`), `interpolate.py` (`ncpath`), `plot_joint.py` (`ncpath`), `precompute.py` (`outpath`) | Same one-line join-onto-a-configured-directory logic under two different names. Each script keeps a 1-line local `ncpath`/`outpath` wrapper supplying its own directory. |
| `derive_interp_tag`, `INTERP_METHOD_TAG_MAP` | `structure.py`, `crossplots.py` (simple regex, no normalisation), `plot_joint.py` (richer: normalises several spellings, warns on unrecognised input) | **Deliberate behavior upgrade**, not a silent change — see below. |
| `resolve_interp_file` | `structure.py`, `crossplots.py` | Identical. |
| `title_suffix`, `group_label` | `structure.py`, `crossplots.py` | Identical one-liners. |
| `maybe_show` | `plot_joint.py` (no-arg, reads a `SHOW_PLOTS` global), `structure.py`/`crossplots.py` (`fig` arg accepted but never used in the body — `plt.show()` shows whatever's open regardless) | Unified as `maybe_show(show_plots)`, explicit arg, no unused `fig` parameter. |
| `save_fig` | `cluster.py`, `plot_joint.py` | Identical (loops `PLOT_FORMATS`, `dpi`, `bbox_inches="tight"`, prints `"  Saved: ..."`). Distinct from `save_paris` below — different delivery convention, kept as two separate functions. |
| `save_paris`, `zip_outputs` | `structure.py`, `crossplots.py` | Identical (Paris-timestamped save + zip, used for delivery). |
| `load_joint_grid`, `get_field` | `structure.py`, `crossplots.py` | Identical (only the script name embedded in one error message differed; genericised). |
| `map_panel_size_in`, `build_map_figure` | `cluster.py`, `plot_joint.py` (`create_map_figure`, twice — once as the full function, once again inline inside `plot_joint.py`'s own `create_map_figure_pair`) | Found **three** copies of the same two-line width/height arithmetic, not two — see below. |

## One deliberate behavior change: `INTERP_TAG` derivation

Three independent implementations existed:
- `structure.py`/`crossplots.py`: a plain regex capturing whatever
  followed `_interp_` in the filename, verbatim, falling back to
  `"unknown"` if the pattern wasn't found.
- `plot_joint.py`: normalises several spellings (`"kriging"`/`"krige"`/
  `"krig"` → `"krig"`, `"nearest_neighbour"`/`"nearest_neighbor"`/
  `"nearest"`/`"nn"` → `"nn"`, etc.) and prints a warning rather than
  silently falling back on an unrecognised method string.

`tomomt.derive_interp_tag()` adopts `plot_joint.py`'s richer version as
the single canonical implementation for all three scripts. This means
`structure.py`'s and `crossplots.py`'s `INTERP_TAG` — and therefore their
output filenames — now normalise e.g. `"kriging"` to `"krig"` the same
way `plot_joint.py`'s always did, instead of keeping the literal
regex-captured substring. Flagged here rather than changed silently;
worth knowing if you have scripts or notes referencing the old,
un-normalised tag in a filename.

While porting this, the original three-way duplication was checked
carefully: `plot_joint.py`'s own `_INTERP_METHOD_TAG_MAP` included
`"krige"` as a spelling variant that an earlier draft of the
consolidated version omitted — caught and fixed before delivery (see
"Verification performed" below).

## What deliberately stayed OUT — per-script by design, not oversight

- **`draw_basemap`** (`cluster.py` / `plot_joint.py`) — ~90% identical
  bodies, but each closes over that script's own loaded topo/bathymetry
  arrays and style globals. Genericising it would mean a large explicit
  parameter list for a function that's realistically more likely to
  diverge further (different basemap needs per script) than converge.
  Left alone; flagged here if you want to revisit it later.
- **`_region()` / `_colorbar_settings()`** (`cluster.py` / `plot_joint.py`)
  — thin config-dict builders closing over each script's own USER
  SETTINGS names, which differ slightly (`nticks=n_labels` vs.
  `nticks=COLORBAR_NTICKS`). Exactly the kind of per-script glue
  `plotpy.py`'s own original design section was already meant to leave
  alone — its docstring explicitly named `draw_basemap` and friends as
  intentionally *not* shared, and this consolidation followed that same
  logic for the config-dict builders too.
- **`_title_suffix()`** — kept per-script: `structure.py`/`crossplots.py`
  return a list (`[site, tag]`, joined by the caller with `", "`),
  `plot_joint.py` returns a pre-formatted string (`" [site, tag]"`).
  Different call-site contracts for a one-line function; not worth
  forcing a shared signature over for two behaviourally-identical-info
  but differently-shaped one-liners.
- **The gradient machinery in `interpolate.py`** (spline- or
  finite-difference-based, with a seismic-grid Jacobian correction) vs.
  **`structure.py`'s `gradient_components()`** (plain `numpy.gradient`,
  joint-grid only) — similar in spirit (both compute spatial gradients),
  materially different in capability. Not the same function wearing two
  names; left as two functions.
- **`clipped_markers`/`clipped_labels`/`draw_north_arrow`/
  `add_latlon_ticks` "duplicates" in `plot_joint.py`** — these looked
  like redundant reimplementations at first grep (same function names as
  in `plotpy.py`), but turned out to already be thin wrappers injecting
  `plot_joint.py`'s own `_region()`/`ARROW_STYLE`/etc. into calls to the
  shared versions — i.e. already correctly using the shared module, not
  duplicating its logic. No change needed there.

## Verification performed

- `ast.parse` on every edited file (`tomomt.py`, `cluster.py`,
  `crossplots.py`, `interpolate.py`, `plot_joint.py`, `precompute.py`,
  `structure.py`, `plotpy.py`).
- `tomomt.py` imported standalone (pyproj/xarray stubbed — neither is
  installed in the environment this was built in) and every new
  function spot-checked: `derive_interp_tag` against several filename
  variants (including the recovered `"krige"` case), `resolve_path`,
  `title_suffix`, `group_label`, `maybe_show` (confirmed it's a no-op
  under a non-interactive backend even when `show_plots=True`),
  `map_panel_size_in`/`build_map_figure` (arithmetic checked against
  hand-computed expected values).
- `structure.py` and `crossplots.py`: full smoke test against a
  synthetic joint-grid dataset (2D/3D cross-plots, cross-gradient/
  cosine/Gramian maps, `save_paris`, `zip_outputs`) — re-run a second
  time after `tomomt.py` had two more rounds of edits, to catch any
  regression from the later changes.
- `cluster.py`, `plot_joint.py`, `interpolate.py`, `precompute.py`:
  these are linear, Spyder-style scripts that read real NetCDF/ModEM
  data and execute top-to-bottom at import time (no `if __name__ ==
  "__main__":` guard), so a full import-based test isn't practical
  without real project data. Instead, each edited function
  (`ncpath`/`outpath`, `safe_to_netcdf`, `save_fig`, `_maybe_show`,
  `create_map_figure`/`create_map_figure_pair`, `_derive_interp_tag`)
  was extracted and exercised directly against `tomomt` with the same
  globals it would see in the real script (`NC_DIR`, `PLOT_DIR`,
  `PLOT_FORMATS`, `FIG_WIDTH`, region bounds, etc.), including
  `safe_to_netcdf`'s actual read-only-file recovery path (created a
  chmod'd-read-only stale file, confirmed it gets overwritten).
- `plotpy.py`'s backward-compatibility shim: imported standalone,
  spot-checked that the functions the three unseen scripts
  (`{SITE_PREFIX}_plot_seis.py` etc.) would plausibly need are all present as
  `plotpy.<name>`.
- Diffed every edited file against its pre-refactor version: `cluster.py`
  (67 diff lines), `interpolate.py` (21), `plot_joint.py` (108),
  `precompute.py` (22), `structure.py` (137) — all reviewed by eye to
  confirm every change was either an import-line edit, a `plotpy.` →
  `tomomt.` rename, or a function body replaced by a call into `tomomt`
  with the same arguments. `modem.py` was not touched (no duplication
  was found in or against it) — confirmed unchanged.

## What's unchanged

Every script's own settings, output filenames, figure content, and
public function signatures are the same as before **except** the one
`INTERP_TAG` normalisation change in `structure.py`/`crossplots.py`
described above. Nothing in `modem.py` was touched.
