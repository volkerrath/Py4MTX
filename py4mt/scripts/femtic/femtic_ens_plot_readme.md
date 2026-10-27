# femtic_ens_plot.py

Ensemble slice plot for a set of FEMTIC inversion runs, with optional
borehole resistivity logs.

---

## Purpose

`femtic_ens_plot.py` discovers converged ensemble members exactly the way
`femtic_ens_post.py` does — scanning `ENSEMBLE_DIR` for `ENSEMBLE_NAME*`
sub-directories and filtering on `femtic.cnv` / `NRMS_MAX` — and produces:

1. **[default, `PER_MEMBER_PLOT=True`]** Per converged member, two
   `fviz.plot_model_slices()` figures in their own sub-directory
   `WORK_DIR/<label>/`, using the same slice geometry and `PLOT_*`
   parameters as `femtic_mod_plot.py`:
   - `iter0.<ext>` — the perturbed prior model
     (`resistivity_block_iter0.dat`), colormap `PLOT_CMAP_ITER0`,
     colour scale `PLOT_CLIM_ITER0`
   - `best.<ext>` — the best-fit model
     (`resistivity_block_iter{numit}.dat`, `numit` from `femtic.cnv`,
     with an `nRMS = ...` annotation on the figure), colormap
     `PLOT_CMAP_BEST`, colour scale `PLOT_CLIM_BEST`

   One file pair per entry in `PLOT_FORMAT` (e.g. `["pdf", "jpg"]`
   writes both a vector and a raster version — the figure is rebuilt
   once per format). When `"pdf"` is among the formats,
   `PER_MEMBER_PDF_CATALOG_MODE` selects which per-member pdf pages
   are *also* collected, in plot order, into one multi-page catalog
   PDF via `matplotlib.backends.backend_pdf.PdfPages`, written to
   `PER_MEMBER_CATALOG_FILE` — `"none"` (nothing), `"iter0"` (prior
   pages only), `"best"` (best-fit pages only), or `"both"` (default;
   both, interlaced per member: iter0_A, best_A, iter0_B, best_B, ...).
   This is in addition to, not instead of, the individual per-member
   files.

2. Optionally, a borehole resistivity log figure (point-in-element sampling,
   drawn by `fviz.plot_borehole_logs()`: resistivity on a log axis, depth in
   km pointing down), sampled from the first converged member's best-fit
   model.

3. Optionally, a **reference model** (`REF_FILE`) shown on both kinds of
   figure: iso-resistivity lines on the slice panels, a dashed profile per
   borehole. See "Reference model" below.

4. Optionally (**`RESULTS_NPZ`**), the same kind of figures from the results
   `.npz` of `femtic_ens_post.py` instead of the member directories -- "NPZ
   mode", see "Working from the results .npz" below.

The slice figures can be made publication-ready with **equal axes or a
defined vertical exaggeration** (`PLOT_EQUAL_ASPECT`, `PLOT_VERT_EXAG`), and
with the font, tick-label, title and figure-size options of
`femtic_ens_post.py` (`PLOT_TICK_FONTSIZE`, ...). See "Display and layout".

> **Removed:** the former joint multi-row ensemble figure
> (`PLOT_JOINT=True`, via `fviz.plot_ensemble_slices()`) has been removed
> entirely — see Changelog, 2026-08-25. It was never the default path
> (`PER_MEMBER_PLOT` always was) and its call carried a long-standing
> keyword mismatch against `plot_ensemble_slices()`'s current signature.

---

## Workflow

```
ORIGIN_METHOD + SITE_DAT  →  UTM_ORIGIN_E/N/LAT/LON (bounding-box midpoint)
                          →  UTM zone auto-derived from origin lat/lon

ENSEMBLE_DIR + ENSEMBLE_NAME + NRMS_MAX
        |
        v  utl.get_filelist() → per-dir femtic.cnv → numit, nRMS
   model_list: [{label, dir, numit, nrms, iter0_file, best_file}, ...]
   (converged members only — same set femtic_ens_post.py includes)
        |
        v  km → m unit conversion, then fem.resolve_slice_positions(PLOT_SLICES)
   slice positions in model-local metres
        |
        v  fem.read_site_dat(SITE_DAT)  [or fem.read_site_position fallback]
   site_xys: (name, x_m, y_m, elev_m) per site
        |
        v  fviz.plot_model_slices(...)  × 2 per member × len(PLOT_FORMAT) [PER_MEMBER_PLOT]
WORK_DIR/<label>/iter0.<ext>  +  WORK_DIR/<label>/best.<ext>   (per converged member)
        |                                    ["pdf" in PLOT_FORMAT, PER_MEMBER_PDF_CATALOG_MODE != "none"]
        v  PdfPages(PER_MEMBER_CATALOG_FILE)
one multi-page pdf catalog, all member pdf figures in plot order
        |                                    [PLOT_BOREHOLE = True]
        v  fviz.plot_borehole_logs(...)
borehole PDF / interactive window
```

---

## Key configuration variables

### Paths
| Variable | Description |
|---|---|
| `WORK_DIR` | Base directory for all relative paths |
| `MESH_FILE` | `mesh.dat` |
| `OBSERVE_FILE` | `observe.dat` (fallback site source) |
| `SITE_DAT` | `site.dat` CSV from `mt_make_sitelist.py` |

### Ensemble input — converged-member discovery
| Variable | Description |
|---|---|
| `ENSEMBLE_DIR` | Directory containing one sub-directory per member |
| `ENSEMBLE_NAME` | Member sub-directories matched via `"<ENSEMBLE_NAME>*"` |
| `CNV_FILE` | Convergence file name in each member directory (default `"femtic.cnv"`; a list gives candidates, first existing wins) -- keep equal to `femtic_ens_post.py`'s value |
| `NRMS_MAX` | Max accepted nRMS from `femtic.cnv` — keep equal to `femtic_ens_post.py`'s value |
| `ENS_LABELS` | Labels for member plots/filenames; `None` → directory basenames |

nRMS and the iteration number are read from each member's `femtic.cnv` via `fem.read_cnv()`, which parses column positions from the file's own header row (case-insensitive substring match) — no FEMTIC-version setting needed.

### Working from the results .npz (NPZ mode)

With `RESULTS_NPZ` set to the results file written by `femtic_ens_post.py`
(`<RESULTS_DIR>/<ENSEMBLE_PREFIX.upper()>_results.npz`; the file its
`PLOT_ONLY` mode reads) the script does **not** scan the member directories
and does not read any cnv or member block file. Every figure is a block file
written from a vector stored in the file (`femtic.insert_model`, template =
the best member's region structure) and drawn by the same code as in the
normal mode, so slices, layout, `PLOT_*` styling, `PLOT_VERT_EXAG` /
`PLOT_EQUAL_ASPECT`, fonts, sites and the reference model act unchanged.

| Variable | Default | Description |
|---|---|---|
| `RESULTS_NPZ` | `None` | `None` = normal mode (scan `ENSEMBLE_DIR`). Path = NPZ mode. |
| `RESULTS_PREFIX` | `None` | Key prefix in the file (`ens_post`'s `ENSEMBLE_PREFIX`); `None` = the only ensemble in the file. |
| `RESULTS_TEMPLATE` | `None` | Block file for header / regions / flags. `None` = the best member's file recorded in the `.npz` (must still exist); otherwise any block file of this ensemble (same mesh and free regions, e.g. the starting model). Only its structure is used. |
| `RESULTS_OUT_DIR` | `None` | Output directory; `None` = `ens_plot/` next to the results file. Holds `<label>/best.<ext>`, `<prefix>_<key>.<ext>`, `blocks/` and the pdf catalog. |
| `RESULTS_MEMBERS` | `"best"` | Best-fit figures of: `"none"`, `"best"` (lowest nRMS), `"all"`, or a list of 0-based indices and/or labels, e.g. `[0, 7, "ann_03"]`. Labels = `ENS_LABELS` or the member directory names recorded in the file. Figures carry the `nRMS = ...` annotation as in the normal mode. |
| `RESULTS_STATS` | `["avg","med","err","mad"]` | Statistic figures, by key: `avg`, `med`, `var`, `err`, `mad`, `p<level>` (e.g. `p50`, `p2_3`), `qdiff_<lo>_<hi>`, `var_boot`, `err_boot`, `sens_<kind>`, `simrc_coef`, `simrc_corr` -- whatever the file holds; keys without data are skipped with a message. |
| `RESULTS_STATS_STYLE` | spread stats `clim=[0, 0.3]` | Per-key `dict(cmap=, clim=, label=)`. Value statistics (`avg`, `med`, `p*`) default to `PLOT_CMAP_BEST` / `PLOT_CLIM_BEST` / `"log10(rho / Ohm*m)"`; all others to `PLOT_CMAP_BEST`, auto limits, the key as label. Sensitivity and SimRC statistics are log10 (the `cv_*` ones linear). |
| `RESULTS_BOREHOLE_MODEL` | `"best"` | Model of the borehole figure (`PLOT_BOREHOLE`): `"best"`, a member index, `"avg"`, `"med"` or `"p<level>"`. Validated before any figure is made. (Normal mode: first converged member.) |

Notes:

- **Not available in NPZ mode:** the perturbed-prior (`iter0`) figures -- the
  file holds the inverted models only (a note is printed). `PER_MEMBER_PLOT`
  does not apply; `PER_MEMBER_PDF_CATALOG_MODE` `"best"` / `"both"` collects the
  member figures drawn here into `<RESULTS_OUT_DIR>/<name of PER_MEMBER_CATALOG_FILE>`.
  `NRMS_MAX` is not applied (the file holds the accepted members only).
- **Reference model:** `REF_SLICES` `"slices"` / `"best"` put it on the member
  figures, `"slices"` also on the statistic figures; on those the lines are drawn
  at resistivity decades of `PLOT_CLIM_BEST`, not at decades of the plotted
  statistic.
- **Air / ocean:** the block files of this mode mark air and ocean by exact
  sentinel values, so a *free* region with a resistivity near `OCEAN_RHO` is
  drawn with its colour. In the normal mode `fviz.plot_model_slices` treats
  every cell within 0.05 (log10) of `OCEAN_RHO` as ocean, so such cells show up
  grey there. Otherwise the member figures of both modes are the same.
- Still needed: `MESH_FILE`, `SITE_DAT` / `OBSERVE_FILE` (origin, sites),
  `REF_FILE` if used, and the template.

```python
RESULTS_NPZ = "/data/ens/results/ANNECY_RTO_results.npz"
RESULTS_MEMBERS = "best"
RESULTS_STATS = ["avg", "err", "sens_mean_an"]
PLOT_VERT_EXAG = 2.0
PLOT_FORMAT = ["pdf"]
```

### Per-member plots (default)
| Variable | Description |
|---|---|
| `PER_MEMBER_PLOT` | If `True` (default), plot iter0 + best-fit figures for every converged member |
| `PLOT_FORMAT` | Output format(s), e.g. `"pdf"` or `["pdf", "jpg"]` — normalised to `_PLOT_FORMATS` |
| `PLOT_CMAP_ITER0` | Matplotlib colormap name for iter0 (prior) plots |
| `PLOT_CMAP_BEST` | Matplotlib colormap name for best-fit plots |
| `PLOT_CLIM_ITER0` | Colour limits `[log10(ρ_min), log10(ρ_max)]` for iter0 (prior) plots; `None` = auto |
| `PLOT_CLIM_BEST` | Colour limits for best-fit plots; `None` = auto |
| `PER_MEMBER_PDF_CATALOG_MODE` | `"none"` / `"iter0"` / `"best"` / `"both"` (default) — which per-member pdf pages, if `"pdf"` is among `_PLOT_FORMATS`, go into the catalog; `"both"` interlaces per member (iter0_A, best_A, iter0_B, best_B, ...) |
| `PER_MEMBER_CATALOG_FILE` | Output path for the multi-page pdf catalog |

### Origin estimation
| Variable | Description |
|---|---|
| `ORIGIN_METHOD` | `None` / `"box"` / `"average"` — how to derive origin from `SITE_DAT` |
| `UTM_ORIGIN_LAT/LON` | Fallback geographic origin (used when `ORIGIN_METHOD=None`) |
| `UTM_ORIGIN_E/N` | Fallback UTM origin in metres |
| `UTM_ZONE_OVERRIDE` | Force a specific UTM zone number; `None` = auto |

### Display and layout
| Variable | Description |
|---|---|
| `DISPLAY_COORDS` | `"model"` / `"utm"` / `"latlon"` |
| `DEPTH_KM` | `True` → depth axis in km |
| `HORIZ_KM` | `True` → horizontal axes in km |
| `PLOT_EQUAL_ASPECT` | `True`: map panels 1:1 and section panels (ns / ew / plane) at true scale (vertical exaggeration 1) — for every `DISPLAY_COORDS`, including `"latlon"`, because the aspect is computed from the real axis units, not from the tick labels. `False`: free aspect (each panel fills its box) |
| `PLOT_VERT_EXAG` | `None` (default) = decided by `PLOT_EQUAL_ASPECT` (`True` → 1.0, `False` → free). A number VE > 0 fixes the section aspect exactly, whatever `PLOT_EQUAL_ASPECT` says: 1.0 = equal axes, 2.0 = vertical scale twice the horizontal, 0.5 = compressed. Map panels are not affected. Set `PLOT_XLIM/YLIM/ZLIM` (or the slice's own `xlim`/`zlim`) so the panel width follows from real extents |
| `PLOT_PANEL_HEIGHT` | Panel height in cm |
| `PLOT_FIGSIZE` | `[width, height]` of the whole figure in cm; `None` = from the panel sizes. Overrides `PLOT_PANEL_HEIGHT` / `PLOT_PANEL_WIDTH` when set |
| `PLOT_TICK_FONTSIZE` | Tick-label size in points, also colourbar ticks (default 7 = fviz's default) |
| `PLOT_LABEL_FONTSIZE` | Axis-label / panel-title / colourbar-label size (default 8) |
| `PLOT_TITLE_FONTSIZE` | Figure-title size; `None` = `PLOT_LABEL_FONTSIZE + 6` |
| `PLOT_TICK_DECIMALS` | Decimal digits of the axis tick labels (depth, easting/northing and lat/lon alike); `None` = fviz's own formatting |
| `PLOT_AIR_COLOR` | Colour of air cells / the area above the topography (default `"white"`) |
| `PLOT_PANEL_WIDTH` | Panel width in cm; `None` (default) = auto per-column width from each panel's aspect ratio (needs `PLOT_EQUAL_ASPECT=True`). When set, overrides that auto-width with one fixed value for every column — takes precedence over `PLOT_PANEL_HEIGHT`-driven sizing, mirroring `femtic_ens_post.py`'s `MOD_PANEL_WIDTH` |
| `PLOT_NROWS/NCOLS` | Grid layout (`None` = auto) |

### Slice geometry
| Variable | Description |
|---|---|
| `PLOT_SLICES` | List of slice dicts; `kind` = `"map"` / `"ns"` / `"ew"` / `"plane"`; all model-local lengths in **km** |
| `PLOT_XLIM/YLIM/ZLIM` | Axis limits in model-local **km** |

### Equal axes and vertical exaggeration

`PLOT_EQUAL_ASPECT` and `PLOT_VERT_EXAG` combine as follows (`vert_exag` is
what is handed to `fviz.plot_model_slices`):

| `PLOT_VERT_EXAG` | `PLOT_EQUAL_ASPECT` | maps | sections (ns / ew / plane) |
|---|---|---|---|
| `None` | `True` | 1:1 | true scale (VE = 1) |
| `None` | `False` | free | free |
| `2.0` | `True` | 1:1 | VE = 2 |
| `2.0` | `False` | free | VE = 2 |

The panel width is derived from the real extents (×VE for the depth span),
so sections come out as wide as they should. With
`DISPLAY_COORDS="latlon"` the horizontal axis is plotted in UTM metres under
lat/lon tick labels, so VE is exact there too.

> **Behaviour change (2026-10-27).** Before, `PLOT_EQUAL_ASPECT=True` had no
> effect with `DISPLAY_COORDS="latlon"` (the check needed model/utm display),
> and the file defined `PLOT_EQUAL_ASPECT` twice (the second definition
> silently won). It is now one setting that acts for every display mode. For
> the old free-aspect look in latlon display set `PLOT_EQUAL_ASPECT=False`.
> Model display with `True` is unchanged (checked pixel-identical on a
> synthetic model).

### Reference model

Compare every member with one model, e.g. a deterministic inversion, the
starting model or a published model.

| Variable | Description |
|---|---|
| `REF_FILE` | Resistivity block file on the **same mesh** as `MESH_FILE`; `None` (default) = off. Its region structure may differ from the members'. A missing file switches the reference off with a warning; a file with another element count stops the run with an error naming both counts |
| `REF_SLICES` | `"slices"` (default): iso-resistivity lines on every panel of the iter0 and best-fit figures; `"best"`: best-fit figures only; `"none"`: not on slices |
| `REF_BOREHOLE` | `True` (default): dashed reference profile per borehole, in the colour of that borehole's trace |
| `REF_LEVELS` | Contour levels, log10(Ω·m), e.g. `[1.0, 2.0, 3.0]`; `None` = the integer decades inside the colour limits of the figure being drawn |
| `REF_STYLE` | `LineCollection` keywords of the slice lines (default `dict(colors="k", linewidths=0.8)`); `linestyles="--"` etc. allowed |
| `REF_LABEL` | Legend entry (lower right of each slice panel, and in the borehole legend); `None` = none |

The lines are the exact boundaries between slice cells that lie on different
sides of a level, so they follow the mesh and show no interpolation
artefacts; air and ocean cells of the reference are ignored.

### Borehole logs

| Variable | Description |
|---|---|
| `PLOT_BOREHOLE` | `True` → draw the figure after the slice plots |
| `BOREHOLE_SITES` | List of dicts: `name`, `x`, `y`, `z_top`, `z_bot`, `dz` (km; `x`/`y` may be CRS-tagged). `z_top` may be `"surface"` (minimum mesh-node depth at that position — with an air layer in the mesh that is the top of the air, so give a number to start at the ground). Line2D keywords (`color`, `ls`, `lw`, …) in a spec style that trace |
| `BOREHOLE_FILE` | Figure path (extension sets the format); `None` = interactive window |
| `BOREHOLE_STYLE` | Baseline Line2D style of all traces |
| `BOREHOLE_XLIM` | x range as `[log10 min, log10 max]` (converted to Ω·m for the plot); `None` = auto |
| `BOREHOLE_SHARED` | `True` = one axes for all boreholes; `False` = one panel each |
| `BOREHOLE_NPZ` | `True` (default): also write the sampled profiles to `<BOREHOLE_FILE stem>.npz` (`depth_<name>`, `rho_<name>`, `reference_rho_<name>` with a reference) |

### Site overlay
| Variable | Description |
|---|---|
| `SITE_NAMES` | Site filter; `None` = all |
| `PLOT_SITES_MAPS` | Show sites on map panels |
| `PLOT_SITES_SLICES` | Show sites on curtain panels |
| `PROJECTION_DIST` | Max distance (**km**) from slice plane for curtain projection |
| `SITE_MARKER` | Marker style dict for map panels |
| `SITE_MARKER_SLICES` | Marker style for curtain panels (`None` → same as `SITE_MARKER`) |
| `MAP_MARKERS` | Additional map markers (known features, etc.); `pos` in model-local **km** or CRS-tagged. An entry with `"is_model_centre": True` (instead of `"pos"`) overrides the model-centre marker's style rather than plotting as a regular marker. |
| `SHOW_MODEL_CENTRE` | `True` (default): marks the model origin on `"map"` panels whenever `DISPLAY_COORDS` is `"utm"`/`"latlon"` (no-op for `"model"`). Default style: black `"+"`, `ms=10`; no legend entry. Override via `MAP_MARKERS` as above; set `False` to force off. |

---

## Units

All model-local length parameters the user sets — `PLOT_SLICES` (`z0`,
plain `x0`/`y0`, `point`), `PLOT_XLIM`/`PLOT_YLIM`/`PLOT_ZLIM`,
`PROJECTION_DIST`, `MAP_MARKERS` positions, and `BOREHOLE_SITES`
(`x`/`y`/`z_top`/`z_bot`/`dz`) — are entered in **kilometres**. A unit
conversion block, run once immediately after the configuration section,
converts everything to model-local metres before `fem.resolve_slice_positions()`
or `fviz.plot_model_slices()`/`plot_ensemble_slices()` see it; those
functions and `femtic_ens_post.py`'s own convention are unaffected and
still work in metres internally.

CRS-tagged position specs — `(value, "utm")` in absolute UTM metres, or
`(value, "latlon")` in decimal degrees — are absolute coordinates, not
model-local lengths, and are **not** affected by the km conversion. For a
`"plane"` slice's `point`, this applies per-axis: `z` is always
model-local and always converted from km, while `x`/`y` are converted
only under an implicit or explicit `"model"` CRS.

---

## Slice specification (`PLOT_SLICES`)

Each entry is a dict with `kind` and the matching position key:

```python
dict(kind="map",   z0=5.0)                           # horizontal map at 5 km depth
dict(kind="ew",    y0=(-16.35, "latlon"))             # E-W section at lat −16.35°
dict(kind="ns",    x0=(300000., "utm"))               # N-S section at UTM easting (m, absolute)
dict(kind="plane", point=[0, 0, 5.0], strike=45, dip=60)   # x,y,z in model-local km
```

Position values accept:
- plain `float` → model-local **km**
- `(value, "utm")` → UTM metres, absolute (easting for `x0`, northing for `y0`)
- `(value, "latlon")` → decimal degrees, absolute (longitude for `x0`, latitude for `y0`)

---

## Changes from previous version

- **2026-10-27b (Claude Sonnet 5.5, Anthropic):** Can work from the results
  `.npz` of `femtic_ens_post.py` (the file its new `PLOT_ONLY` mode reads):
  new `RESULTS_NPZ` (`None` = unchanged normal mode), `RESULTS_PREFIX`,
  `RESULTS_TEMPLATE`, `RESULTS_OUT_DIR`, `RESULTS_MEMBERS`, `RESULTS_STATS`,
  `RESULTS_STATS_STYLE`, `RESULTS_BOREHOLE_MODEL`; see "Working from the
  results .npz". Figures are drawn from stored vectors by the same
  `_plot_member_slice()` (new keywords `stats_panel`, `cbar_label`,
  `ref_clim`), with the same slices, styling, vertical exaggeration and
  reference model. The iter0 figures cannot be made from a results file;
  `NRMS_MAX` is not applied there. The normal mode is unchanged (member and
  borehole output identical to the 2026-10-27 version on a synthetic
  ensemble). AI-generated code -- review before production use.

- **2026-10-27 (Claude Sonnet 5.5, Anthropic):** Publication-quality slices
  and an optional reference model, built on the new `femtic_viz.py` of the
  same date. (1) Equal axes or a defined vertical exaggeration: new
  `PLOT_VERT_EXAG`; `PLOT_EQUAL_ASPECT` now acts for every `DISPLAY_COORDS`
  (see "Equal axes and vertical exaggeration" for the one behaviour change)
  and is defined once. (2) New options passed through to
  `fviz.plot_model_slices`: `PLOT_FIGSIZE`, `PLOT_TICK_FONTSIZE`,
  `PLOT_LABEL_FONTSIZE`, `PLOT_TITLE_FONTSIZE`, `PLOT_TICK_DECIMALS`,
  `PLOT_AIR_COLOR` (defaults reproduce the previous figures). (3) Reference
  model: `REF_FILE`, `REF_SLICES`, `REF_BOREHOLE`, `REF_LEVELS`,
  `REF_STYLE`, `REF_LABEL`. (4) The local `plot_borehole_logs()` is replaced
  by `fviz.plot_borehole_logs()`: log resistivity axis, depth in km pointing
  down (the local version inverted the depth axis once per borehole, so it
  pointed up for an even number of boreholes), `z_top="surface"`, per-spec
  line styles, NPZ export (`BOREHOLE_NPZ`). `_km_to_m_point()` accepts `[x, y]`
  points. AI-generated code — review before production use.

- 2026-10-11: the local borehole-log helper reads the mesh with `fem.read_femtic_mesh(path)` (returns `nodes, conn`; the reader now lives only in `femtic.py`, with the optional `swap_xy` switch / `femtic.MESH_SWAP_XY`) instead of the removed `fviz.read_femtic_mesh`.

- `ESTIMATE_ORIGIN` / `CALIBRATION_SITES` / `UPDATE_CONFIG` replaced by
  `ORIGIN_METHOD` (`None` | `"box"` | `"average"`).  Origin estimation
  now runs **before** UTM zone derivation, fixing a `TypeError` when
  `UTM_ORIGIN_LAT/LON` are `None`.
- Local coordinate helpers removed; `fem.*` and `utl.*` called directly.
- `site_xys` tuples now carry elevation (`elev_m`) as fourth element.
- `plot_ensemble_slices` call extended with `site_xys`, `utm_origin_*`,
  `utm_zone`, `utm_northern`, `utm_to_latlon_fn`, `latlon_to_model_fn`,
  `display_coords`, `depth_km`, `horiz_km`, `equal_aspect`,
  `panel_height`, `nrows`, `ncols`, `projection_dist`,
  `sites_in_maps`, `sites_in_slices`, `site_marker_slices`,
  `map_markers`, `obs_coords_only` kwargs.
- Added `DEPTH_KM`, `HORIZ_KM`, `PLOT_EQUAL_ASPECT`, `PLOT_PANEL_HEIGHT`,
  `PLOT_NROWS`, `PLOT_NCOLS`, `PLOT_SITES_MAPS`, `PLOT_SITES_SLICES`,
  `SITE_MARKER_SLICES`, `MAP_MARKERS`, `DISPLAY_COORDS` config vars.
- **2026-08-13 (Claude Sonnet 5, Anthropic):** Added `femtic_ens_plot_summary.md`
  output at end of run: user-set (UPPERCASE) parameters, script path, and
  run date/time.
- **2026-08-14 (Claude Sonnet 5, Anthropic):** Replaced `ENS_DIRS` /
  `BLOCK_PATTERN` / `ENS_ITER` with `ENSEMBLE_DIR` / `ENSEMBLE_NAME` /
  `NRMS_MAX` / `FEMTIC` — converged members are now discovered the same
  way `femtic_ens_post.py` does, via `femtic.cnv`. Default behaviour is
  now two per-member figures (`PER_MEMBER_PLOT=True`): perturbed prior
  (`iter0`) and best-fit (`iterX`, `numit` from `femtic.cnv`), each via
  `fviz.plot_model_slices()`. The former joint multi-row figure is kept
  as an optional extra (`PLOT_JOINT=False` by default); its
  `plot_ensemble_slices()` call still has the previously flagged keyword
  mismatch and will raise `TypeError` if enabled — unresolved, out of
  scope for this change. Added `PER_MEMBER_PLOT`, `PER_MEMBER_FORMAT`,
  `PLOT_JOINT` config vars.
- **2026-08-15 (Claude Sonnet 5, Anthropic):** `PER_MEMBER_FORMAT`
  replaced by `PLOT_FORMAT`, which now accepts a list (e.g.
  `["pdf", "jpg"]`, normalised to `_PLOT_FORMATS`), mirroring
  `MOD_PLOT_FORMAT` in `femtic_ens_post.py`. Per-member files moved into
  their own sub-directory `WORK_DIR/<label>/` as `iter0.<ext>` /
  `best.<ext>` (previously the flat `<label>_iter0.<ext>` naming in
  `WORK_DIR`). Added `PER_MEMBER_PDF_CATALOG` (default `True`) and
  `PER_MEMBER_CATALOG_FILE`: when `"pdf"` is among `_PLOT_FORMATS`,
  every per-member pdf figure is additionally combined into one
  multi-page catalog via `matplotlib.backends.backend_pdf.PdfPages`.
- **2026-08-16 (Claude Sonnet 5, Anthropic):** Reduced peak memory for
  large ensembles: the pdf catalog is now opened once before the
  per-member loop and each figure is written via `PdfPages.savefig()`
  and released (`del` + `gc.collect()`) immediately after it's built,
  instead of accumulating every member's `Figure` in a list and writing
  them all at the end — bounds peak memory to roughly one member's
  figures rather than the whole ensemble's.
- **2026-08-20 (Claude Sonnet 5, Anthropic):** All user-facing length
  parameters — `PLOT_SLICES` (`z0`, plain `x0`/`y0`, `point`),
  `PLOT_XLIM`/`PLOT_YLIM`/`PLOT_ZLIM`, `PROJECTION_DIST`, `MAP_MARKERS`
  positions, `BOREHOLE_SITES` (`x`/`y`/`z_top`/`z_bot`/`dz`) — are now
  entered in **kilometres** instead of metres. A new unit-conversion
  block, run once right after the config section, converts everything
  to model-local metres before `fem.resolve_slice_positions()` or
  `fviz.plot_model_slices()`/`plot_ensemble_slices()` see it;
  `femtic.py`/`femtic_viz.py` and `femtic_ens_post.py`'s own convention
  are unchanged (still metres internally). CRS-tagged specs — `(value,
  "utm")` / `(value, "latlon")` — are absolute coordinates and are left
  unchanged by the conversion. This also fixes a pre-existing bug: the
  active `PLOT_SLICES` block had `z0` values and `PLOT_XLIM/YLIM/ZLIM`
  commented as `# km` but were actually consumed as metres, so map
  slices were plotted at 0–25 **m** depth instead of 0–25 km, and the
  map panels' extent was clipped to a ±25 m box instead of ±25 km —
  producing flat, near-uniform panels zoomed in on a single point
  rather than the full model domain. `PROJECTION_DIST` and the
  `BOREHOLE_SITES` example were also updated from metres to km.
- **2026-08-20 (Claude Sonnet 5, Anthropic):** (1) `PER_MEMBER_PDF_CATALOG`
  (bool) replaced by `PER_MEMBER_PDF_CATALOG_MODE` (`"none"` / `"iter0"`
  / `"best"` / `"both"`, default `"both"`), selecting which per-member
  pdf pages go into the catalog — none, prior-only, best-only, or both
  interlaced per member in plot order (iter0_A, best_A, iter0_B, best_B,
  ...); `"both"` reproduces the previous `PER_MEMBER_PDF_CATALOG=True`
  behaviour. Validated against an allowed-value list at load time.
  (2) `PLOT_CLIM` split into `PLOT_CLIM_ITER0` and `PLOT_CLIM_BEST` so
  the perturbed-prior and best-fit model plots can use distinct
  log₁₀(ρ) colour scales, since the prior's resistivity range often
  differs substantially from the inverted result. The joint ensemble
  figure (`PLOT_JOINT`), which only plots best-fit models, now uses
  `PLOT_CLIM_BEST`.
  (3) `PLOT_CMAP` likewise split into `PLOT_CMAP_ITER0` and
  `PLOT_CMAP_BEST`, mirroring (2), so the colormap itself — not just
  the colour limits — can differ between the prior and best-fit plots.
  The joint ensemble figure uses `PLOT_CMAP_BEST`.
- **2026-08-23 (Claude Sonnet 5, Anthropic):** Fixed per-member pdf
  files being much larger than the catalog pdf they feed into. The
  matplotlib indexed-colour PDF workaround added on 2026-08-20
  (`pdf.compression=0`) was being applied pre-emptively to every
  per-member save, while the catalog page for the same figure was
  written by a second, independent `PdfPages.savefig()` call outside
  that override — so the catalog kept the default compressed size and
  the per-member file did not. `pdf.compression` now stays at its
  default (compressed) for both writes; disabling it is only tried
  reactively, on an actual `IndexError` from the specific save that
  hit it, so only an affected figure pays the larger-file cost. The
  catalog-append call is now guarded the same way, since it renders
  the figure independently and can in principle hit the same bug even
  when the per-member save already succeeded.
- **2026-08-25 (Claude Sonnet 5, Anthropic):** Removed `PLOT_JOINT` and
  the joint multi-row ensemble figure entirely (the `fviz.plot_
  ensemble_slices()` call in the former step (6b) of the workflow),
  along with its now-unused config vars `ENS_STAT_ROWS`, `PLOT_ENS_
  FILE`, `ENS_PER_MEMBER` — rather than fixing the long-standing
  keyword mismatch against `plot_ensemble_slices()`'s current
  signature, since `PER_MEMBER_PLOT` was always the only path actually
  exercised. `PLOT_CMAP_BEST`/`PLOT_CLIM_BEST` docs updated to drop
  the now-obsolete "and the joint figure" mentions; `ENS_FILES` is
  kept (still feeds the borehole step, renumbered (7)). Also added
  `PLOT_PANEL_WIDTH` (cm; `None` = auto from aspect ratio), mirroring
  `femtic_ens_post.py`'s `MOD_PANEL_WIDTH`: when set it overrides
  `plot_model_slices()`'s per-column auto-width with one fixed value,
  taking precedence over `PLOT_PANEL_HEIGHT`-driven sizing; `None`
  (default, unchanged prior behaviour) leaves `PLOT_PANEL_HEIGHT`
  alone driving the aspect-ratio auto-width. Wired into the single
  shared `_kwargs` dict in `_plot_member_slice()`, so both the iter0
  and best-fit per-member figures pick it up.
- **2026-09-06 (Claude Sonnet 5, Anthropic):** Added `SHOW_MODEL_CENTRE`
  (default `True`): marks the model origin on `"map"` panels whenever
  `DISPLAY_COORDS` is `"utm"`/`"latlon"`, via `fviz.plot_model_slices`'s
  new `show_model_centre` parameter. Override style with a `MAP_MARKERS`
  entry carrying `"is_model_centre": True`.
- **2026-09-07 (Claude Sonnet 5, Anthropic):** Removed the `FEMTIC`
  config variable and its `"4.3"`/`"5."` version-string switch on the
  nRMS column index (6 vs 8): that switch only matched one specific
  column layout, and silently misread nRMS (e.g. reading the Distortion
  or Misfit column instead) for any run whose actual column count
  didn't match the assumed FEMTIC-version pairing. Now uses
  `fem.read_cnv()`, matching `femtic_ens_post.py`'s `get_nrms()`-based
  fix of 2026-09-02 — column positions are read from `femtic.cnv`'s own
  header row every time, independent of version or Beta/Distortion
  presence.
- **2026-10-08 (Claude Sonnet 5.5, Anthropic):** Convergence file name is now configurable via `CNV_FILE` (default `"femtic.cnv"`; list of candidates allowed), resolved per member with `fem.resolve_cnv_path()`. AI-generated; review before production use.
