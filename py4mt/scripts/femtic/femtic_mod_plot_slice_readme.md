# femtic_mod_plot_slice.py — README

**Purpose:** 2-D slice panels (map, curtain, plane) for a FEMTIC tetrahedral
resistivity model, with optional embedded borehole columns.

One of the "mod" scripts (plots of single models; the "ens" scripts are the
quality-control and ensemble summary plots). It draws one model, or one figure
per model for a list / glob / run directories, see "Several models (batch)".

Sister scripts:
- [`femtic_mod_plot_bh.py`](femtic_mod_plot_bh_readme.md) — standalone 1-D ρ(z) borehole log figures.
- [`femtic_mod_plot_3d.py`](femtic_mod_plot_3d_readme.md) — PyVista 3-D rendering and VTK/VTU export.

---

## What this script does

```
mesh.dat + resistivity_block_iterX.dat
        │
        └─(5)─► fviz.plot_model_slices(...)   [2-D map/curtain/plane panels]
                 → PDF / PNG / interactive
                 (optional embedded borehole columns when BOREHOLE_IN_SLICE=True)
```

All geometry lives in `femtic_viz.py`; no geometry code lives here.

---

## Execution steps

| Step | What happens |
|---|---|
| (0) | Resolve `MODEL_FILE` / `MODEL_ITER` to the list of models (`femtic.resolve_model_files`); stop if there is none |
| (1) | Optionally estimate UTM mesh-origin from `SITE_DAT` bounding-box or mean |
| (2) | Derive UTM zone from finalised `UTM_ORIGIN_LAT` / `UTM_ORIGIN_LON` |
| (3) | Resolve `PLOT_SLICES` positions to model-local metres (CRS conversion) |
| (4) | Read site positions from `SITE_DAT` or `OBSERVE_FILE` for marker overlay |
| (5) | Plot 2-D slice panels via `fviz.plot_model_slices_robust` (one figure per model; optional catalog PDF) |

---

## Several models (batch)

`MODEL_FILE` is, as before, one resistivity block file and `PLOT_FILE` its
figure (the output is identical to the earlier version). It can also be

- a list of block files,
- a glob pattern (`"ens/ens_*/resistivity_block_iter0.dat"`),
- run directories: one directory, a list, or a glob such as `"ens/ens_*"`.
  `MODEL_ITER` then chooses the block file of each run:

| `MODEL_ITER` | Block file |
|---|---|
| `"last"` (default) | the iteration of the last row of `femtic.cnv` (the best-fit model of a converged run, as `femtic_ens_post.py` takes it) |
| `"best"` | the iteration with the lowest nRMS in `femtic.cnv` |
| `N` or `"iterN"` | `resistivity_block_iterN.dat`; `0` = starting model (for an RTO member the perturbed prior) |
| a list, e.g. `[0, "last"]` | one figure per entry and run, grouped by run |

`MODEL_ITER` does not apply to block files given as files. With more than one
resulting model there is **one figure per model**, `PLOT_FILE` is not used,
and

| Setting | Default | Meaning |
|---|---|---|
| `BATCH_OUT_DIR` | `None` | directory of the figures; `None` = the model's own directory |
| `BATCH_FORMATS` | `None` | e.g. `("pdf", "png")`; `None` = the extension of `PLOT_FILE` |
| `BATCH_CATALOG` | `None` | multi-page PDF, one page per model: `"auto"` (= `catalog_slices.pdf` in `BATCH_OUT_DIR` or the common parent directory) or a path; needs `"pdf"` among the formats |
| `BATCH_TITLE` | `"{label} - iteration {iter}"` | figure title; `{label}`, `{iter}`, `{tag}`, `{nrms}` |

Files are named `<label>_<tag>_slices.<ext>` (label = run directory, tag =
`iterN`), e.g. `ens_003_iter0_slices.pdf`. The nRMS (and alpha) of that
iteration is read from `femtic.cnv` of the run directory and annotated on the
figure (in the single-model case from `CNV_FILE`, as before). The same
mesh (`MESH_FILE`), origin, slices and colour settings are used for all models.

Example: prior and best-fit figure of every member of an RTO ensemble, with a
catalog:

    WORK_DIR   = "/data/ens/"                 # contains mesh.dat
    MODEL_FILE = WORK_DIR + "ens_*"
    MODEL_ITER = [0, "last"]
    BATCH_CATALOG = "auto"

---

## Slice coordinate system

Every horizontal slice position (`x0`, `y0`, `point` horizontal component)
accepts a coordinate-system tag:

| Value | Meaning |
|---|---|
| plain `float` | model-local metres — origin at mesh centre (backward-compatible) |
| `(value, "utm")` | UTM metres in the mesh UTM zone |
| `(value, "latlon")` | decimal degrees (longitude for easting, latitude for northing) |

Depth `z0` is always model-local metres (z positive-down); no geographic
conversion applies.

Conversion chain: `lat/lon → UTM → model-local`.

---

## 2-D slice panels (`PLOT_SLICES`)

### Panel kinds

| `kind` | Position key | Description |
|---|---|---|
| `"map"` | `z0` | Horizontal slice at depth `z0` |
| `"ns"` | `x0` | N-S vertical section at easting `x0` |
| `"ew"` | `y0` | E-W vertical section at northing `y0` |
| `"plane"` | `point`, `strike`, `dip` | Arbitrary oblique plane |
| `"profile"` | `p1`, `p2`, `z_top`, `z_bot` | Vertical fence section between two endpoints; strike auto-derived |

### `kind="profile"` keys

| Key | Type | Default | Description |
|---|---|---|---|
| `p1`, `p2` | position spec | required | Endpoint positions. Each accepts a bare 2-element list (model-local metres), `([x, y], "model")`, `([E, N], "utm")`, or `([lon, lat], "latlon")`. A 3-element `[x, y, z]` form is also accepted (z ignored). |
| `z_top` | float | `0.0` | Shallowest depth of the panel [m, z-positive-down] |
| `z_bot` | float | `20000.0` | Deepest depth of the panel [m, z-positive-down] |

The resolver converts a `"profile"` spec into a `kind="plane"` dict with `dip=90` before passing it to `fviz.plot_model_slices`. The horizontal axis of the panel runs along-strike from *p1* to *p2*; use `invert_x=True` to reverse the sense.

### Plane slices (`kind="plane"`)

`point` is any point on the plane (`[x, y, z]` model-local m, or
`([x, y, z], "model")`, `([E, N, z], "utm")`, `([lon, lat, z], "latlon")`;
`z` = depth in m, positive down). `strike` is the azimuth of the strike line,
degrees clockwise from north, **for every dip**; `dip` is the angle below the
horizontal, `0` .. `180`, towards `strike + 90` (`strike=325., dip=70.` dips
70 deg towards N55E). `dip = 90` is a vertical section (horizontal axis =
along-strike distance in model km, vertical axis = depth). For other dips the
axes are the true in-plane distances from `point`: horizontal = along strike,
vertical = down dip (positive downwards), in km if `DEPTH_KM` else m, with
true aspect where the other sections get a fixed aspect (`PLOT_EQUAL_ASPECT`, or a
vertical exaggeration set). The default extent is the plane cut with the box
`PLOT_XLIM` x `PLOT_YLIM` x `PLOT_ZLIM` (mesh extent where one is `None`).
Before 2026-11-09 dipping planes came out sheared and mislabelled.

### Per-panel optional keys

| Key | Description |
|---|---|
| `xlim` | `[xmin, xmax]`: easting axis limit; for a `"plane"`: along-strike range in m relative to `point` |
| `ylim` | `[ymin, ymax]`: northing axis limit; for a `"plane"` with `dip != 90`: down-dip range in m relative to `point` |
| `zlim` | `[zmin, zmax]` — depth axis limit (ns / ew / plane panels) |
| `invert_x` | `True` → flip horizontal axis left-to-right (default `False`) |
| `title` | Optional string override for panel title |

Per-panel `xlim`/`ylim`/`zlim` override the global `PLOT_XLIM`/`PLOT_YLIM`/`PLOT_ZLIM`.

### Example

```python
PLOT_SLICES = [
    dict(kind="map", z0=-4000.0),
    dict(kind="ns",  x0=(-70.90, "latlon")),
    dict(kind="ew",  y0=(-16.40, "latlon")),
    dict(kind="plane",
         point=([-71.5, -16.4, 5000.0], "latlon"),
         strike=45., dip=70.,
         invert_x=True),
    dict(kind="profile",
         p1=([-71.536, -16.197], "latlon"),
         p2=([-71.406, -16.299], "latlon"),
         z_top=0.0, z_bot=25000.0,
         title="NW-SE profile"),
]
```

---

## Polylines on the slice panels (`PLOT_POLYLINES`)

`PLOT_POLYLINES` is a dict `{name: entry}` of lines (faults, profiles,
boundaries, borehole traces, ...) drawn on the slice panels. `{}` (default)
draws nothing and leaves the figures unchanged. Lengths are in **metres**
(like `PLOT_SLICES`); the dict is passed to
`fviz.plot_model_slices(polylines=...)` with the other shared options, so it
applies to every figure of a batch run.

```python
PLOT_POLYLINES = {
    "fault F1": dict(
        points=[[-4000., -3000., 500.], [0., 0., 2500.], [4000., 3000., 4500.]],
        style=dict(color="white", linewidth=2.0), max_dist=1500.),
    "profile": dict(points=([[-70.95, -16.40], [-70.85, -16.30]], "latlon"),
                    kinds="map", style=dict(color="k", linestyle=":")),
}
```

| Key of an entry | Default | Description |
|---|---|---|
| `points` | required | `[[x, y], ...]` or `[[x, y, z], ...]`, at least 2 rows, model-local m; z = depth, positive down (missing z = 0). Or CRS-tagged `(pts, "model" \| "utm" \| "latlon")`: `"utm"` = `[E, N(, z)]` in UTM m, `"latlon"` = `[lon, lat(, z)]` in degrees (longitude first); z is m in every case. A bare point list instead of the entry dict is shorthand. |
| `style` | 1.5 pt solid, colour cycled | matplotlib `Line2D` keywords (`color`, `linewidth`, `linestyle`, `marker`, `alpha`, `zorder`, ...). |
| `label` | the dict key | Legend text (lower right of each panel); `None` = no entry. |
| `kinds` | all panels | `"map"`, `"ns"`, `"ew"`, `"plane"` (also the `"profile"` panels), `"vertical"` (= ns + ew + plane) or a list of these. |
| `max_dist` | `PROJECTION_DIST` | m; vertical sections only: the parts of the polyline within `max_dist` of the section plane are projected into it (slab, like the site markers). `0` = only parts lying in the section plane. If `PROJECTION_DIST` is `None` the whole polyline is projected. |
| `z_range` | none | `[zmin, zmax]` m; maps only: draw only on maps whose `z0` lies in the range. Default: plan view on every map. |

Maps show the whole polyline in plan view. The polylines never change the axis
limits; parts outside the panel are clipped by the axes.

**Intersection line of two slices.** `fviz.slice_intersection_polyline(a, b,
xlim=..., ylim=..., zlim=...)` returns the line in which two slices meet
(clipped to the box and to the extents of the slices) as an `(n, 3)` array in
model-local metres, which is what `points` takes;
`fviz.slice_intersection_polylines(slices, ...)` makes ready entries for many
pairs. For example the line of the map at 5 km depth and the N-S section
through the origin:

```python
_p = fviz.slice_intersection_polyline(
    dict(kind="map", z0=5000.0), dict(kind="ns", x0=0.0),
    xlim=PLOT_XLIM, ylim=PLOT_YLIM, zlim=PLOT_ZLIM)
PLOT_POLYLINES["map 5 km x ns 0"] = dict(
    points=_p, max_dist=0.0, z_range=[5000., 5000.])
```

`max_dist=0.0` draws the line only on the panels that contain it, `z_range`
only on the map at that depth. There is no automatic switch that draws the
intersections of all `PLOT_SLICES`. Slices with a CRS-tagged position or an
unresolved `"profile"` must be resolved to model-local metres first
(`slices_resolved` in the script).

Needs the `femtic_viz.py` of 2026-11-11.

---

## Embedded borehole columns (`BOREHOLE_IN_SLICE`)

When `BOREHOLE_IN_SLICE = True` and `BOREHOLE_SITES` is non-empty, borehole
ρ(z) panels are appended as extra columns to the right of the slice grid
inside the **same figure**.  The borehole depth axis is linked to the leftmost
curtain/plane panel for synchronised zoom/pan.

For a standalone borehole figure (higher DPI, separate PDF), use
`femtic_mod_plot_bh.py` instead.

### Borehole spec dict keys

| Key | Type | Required | Description |
|---|---|---|---|
| `"name"` | str | yes | Label in legend / panel title |
| `"x"` | float or `(v, "crs")` | yes | Easting: float = model-local m; `(lon, "latlon")` = longitude [°]; `(E_m, "utm")` = UTM easting [m] |
| `"y"` | float or `(v, "crs")` | yes | Northing: float = model-local m; `(lat, "latlon")` = latitude [°]; `(N_m, "utm")` = UTM northing [m] |
| `"z_top"` | float or `"surface"` | no (def. 0) | Start depth [m, z-down]. `"surface"` → auto from mesh nodes (requires scipy) |
| `"z_bot"` | float | no (def. 20000) | End depth [m, z-down] |
| `"dz"` | float | no (def. 200) | Sampling interval [m] |
| `"lat"` | float | no | Override legend latitude [°] (auto-inferred for `"latlon"` / `"utm"` CRS) |
| `"lon"` | float | no | Override legend longitude [°] |
| `"color"`, `"ls"`, `"lw"`, `"marker"`, `"alpha"`, … | any | no | Matplotlib `Line2D` kwargs — override `BOREHOLE_STYLE` for this trace |

### Borehole configuration

| Variable | Default | Description |
|---|---|---|
| `BOREHOLE_SITES` | `[]` | List of borehole spec dicts |
| `BOREHOLE_STYLE` | `dict(lw=1.2, marker="none")` | Baseline line style; per-spec keys override |
| `BOREHOLE_XLIM` | `[1., 1e4]` | x-axis limits [Ω·m, log scale]; `None` = auto |
| `BOREHOLE_SHARED` | `True` | `True` = one shared column; `False` = one column per borehole |
| `BOREHOLE_IN_SLICE` | `True` | Embed borehole columns inside the slice figure |

---

## Configuration reference

### Paths

| Variable | Description |
|---|---|
| `WORK_DIR` | Working directory prefix for all paths |
| `MODEL_FILE` | `resistivity_block_iterX.dat` to display; or a list / glob / run directories (batch, see above) |
| `MODEL_ITER`, `BATCH_*` | Batch settings, see "Several models (batch)" |
| `MESH_FILE` | `mesh.dat` |
| `MESH_SWAP_XY` | `True` | Swap of the first two node-coordinate columns when the mesh is read (`femtic.read_femtic_mesh`). `mesh.dat` holds x = northing, y = easting; the plots, sites and slice positions use x = easting, y = northing, so `True` (default) gives the right orientation. `False` draws the model transposed (NNE structures appear as ENE) and is only for meshes already written as x = easting, y = northing; `None` = leave femtic's module default (no swap). |
| `OBSERVE_FILE` | `observe.dat` — fallback site source |
| `SITE_DAT` | `mt_make_sitelist.py` CSV (`name,lat,lon,elev,sitenum,easting,northing`); `None` to disable |

### Ocean / air

| Variable | Default | Description |
|---|---|---|
| `OCEAN` | `None` | `None` = auto-infer; `True`/`False` = force |
| `AIR_RHO` | `1e9` | Air sentinel Ω·m (region 0) |
| `OCEAN_RHO` | `0.25` | Ocean sentinel Ω·m (region 1) |

### UTM / CRS

| Variable | Default | Description |
|---|---|---|
| `UTM_ORIGIN_LAT` | `None` | Mesh-centre latitude [°] |
| `UTM_ORIGIN_LON` | `None` | Mesh-centre longitude [°] |
| `UTM_ORIGIN_E` | `None` | Mesh-centre UTM easting [m] |
| `UTM_ORIGIN_N` | `None` | Mesh-centre UTM northing [m] |
| `UTM_ZONE_OVERRIDE` | `None` | Force UTM zone number; `None` = auto |
| `ORIGIN_METHOD` | `"box"` | `None` / `"box"` / `"average"` — origin estimation from `SITE_DAT` |
| `DISPLAY_COORDS` | `"model"` | `"model"` / `"utm"` / `"latlon"` |

### Site overlay

| Variable | Default | Description |
|---|---|---|
| `SITE_NAMES` | `None` | Site names from `SITE_DAT`; `None` = all |
| `SITE_NUMBER` | `[5,6,7]` | Fallback: site numbers from `observe.dat` |
| `PLOT_SITES_MAPS` | `True` | Show markers on map panels |
| `PLOT_SITES_SLICES` | `True` | Show markers on curtain / plane panels |
| `PROJECTION_DIST` | `1000.` | Max distance [m] from slice plane for site to appear |
| `SITE_MARKER` | `dict(marker="v",…)` | Marker style for map panels |
| `SITE_MARKER_SLICES` | `dict(marker="v",…)` | Marker style for curtain / plane panels |
| `MAP_MARKERS` | `[]` | Extra lat/lon point markers on map panels only. An entry with `"is_model_centre": True` (instead of `"latlon"`) overrides the model-centre marker's style rather than plotting as a regular marker. |
| `SHOW_MODEL_CENTRE` | `True` | Marks the model origin on `"map"` panels whenever `DISPLAY_COORDS` is `"utm"`/`"latlon"` (no-op for `"model"`). Default style: black `"+"`, `ms=10`; no legend entry. Override style via `MAP_MARKERS` as above; set `False` to force off. |

### Plotting

| Variable | Default | Description |
|---|---|---|
| `PLOT_FILE` | `*_slices.pdf` | Output path; `None` = interactive |
| `PLOT_DPI` | `600` | Saved-figure DPI |
| `PLOT_CMAP` | `"turbo_r"` | Matplotlib colormap |
| `PLOT_CLIM` | `[0., 3.]` | `[log10_min, log10_max]` in log10(Ω·m); `None` = auto |
| `PLOT_OCEAN_COLOR` | `"lightgrey"` | Flat colour for ocean cells; `None` = colormap |
| `PLOT_AIR_COLOR` | `"whitesmoke"` | Flat colour for air polygons; `None` = blank |
| `PLOT_AIR_BGCOLOR` | `None` | Axes facecolor for air/background |
| `TICK_FONTSIZE` | `7` | Font size for axis tick labels and colourbar ticks |
| `LABEL_FONTSIZE` | `8` | Font size for axis labels, panel titles, and colourbar label |

### Alpha / blanking

| Variable | Default | Description |
|---|---|---|
| `ALPHA_FILE` | `None` | Second block file whose `region_rho` values are log10 weights |
| `ALPHA_MODE` | `"fade"` | `"fade"` = proportional transparency; `"blank"` = hard cutoff |
| `ALPHA_BLANK_THRESH` | `0.0` | log10 threshold below which polygons are suppressed |

### Axis limits and aspect

| Variable | Default | Description |
|---|---|---|
| `PLOT_XLIM` | `[-15000., 15000.]` | Global easting limits [m] |
| `PLOT_YLIM` | `[-15000., 15000.]` | Global northing limits [m] |
| `PLOT_ZLIM` | `[-6000., 15000.]` | Global depth limits [m] |
| `PLOT_EQUAL_ASPECT` | `True` | Equal-aspect on map / curtain panels |
| `DEPTH_KM` | `True` | Depth axis in km on curtain / plane panels |
| `HORIZ_KM` | `True` | Horizontal axes in km when `DISPLAY_COORDS="model"` |

### Figure layout

| Variable | Default | Description |
|---|---|---|
| `PLOT_NROWS` | `None` | Subplot rows; `None` = 1 |
| `PLOT_NCOLS` | `None` | Subplot columns; `None` = `len(PLOT_SLICES)` |
| `PLOT_PANEL_HEIGHT` | `16.0` | Panel height [cm] |
| `PLOT_PANEL_WIDTH` | `None` | Fixed panel width [cm]; `None` = auto |
| `PLOT_FIGSIZE` | `None` | Full figure size `[w, h]` [cm]; overrides auto |

---

## Dependencies

| Package | Role |
|---|---|
| `femtic` (Py4MTX) | `resolve_model_files`, `resolve_slice_positions`, `read_site_dat`, `read_site_position`, `utm_to_model`, `latlon_to_model` |
| `femtic_viz` (Py4MTX) | `plot_model_slices`, `plot_model_slices_robust`, `append_catalog_page` |
| `util` (Py4MTX) | `utm_zone_from_latlon`, `utm_to_latlon_zn`, `print_title` |
| `matplotlib` | 2-D rendering (via `femtic_viz`) |
| `scipy` | Only for `z_top="surface"` KD-tree lookup in embedded boreholes |
| `numpy` | Array operations |

---

## Provenance

| Date | Author | Note |
|---|---|---|
| 2026-05-06 | vrath / Claude Sonnet 4.6 | Created as part of `femtic_mod_plot.py` |
| 2026-05-06 | vrath / Claude Sonnet 4.6 | Lat/lon and UTM slice positions; auto UTM zone |
| 2026-05-13 | vrath / Claude Sonnet 4.6 | Config block harmonised with `femtic_mod_edit.py` |
| 2026-05-16 | vrath / Claude Sonnet 4.6 | Borehole step added; `BOREHOLE_*` config block |
| 2026-05-23 | vrath / Claude Sonnet 4.6 | Geographic helpers moved to `util.py`; model-local helpers to `femtic.py` |
| 2026-05-23 | vrath / Claude Sonnet 4.6 | `SITE_DAT` in `mt_make_sitelist.py` CSV format |
| 2026-05-24 | vrath / Claude Sonnet 4.6 | Ensemble step removed (→ `snippets.py`); borehole renumbered step 6 |
| 2026-05-25 | vrath / Claude Sonnet 4.6 | `PLOT_SITES_MAPS`/`_SLICES`, `PROJECTION_DIST`, `DEPTH_KM`, `HORIZ_KM`, `PLOT_NROWS`/`NCOLS`, panel sizing in cm |
| 2026-05-26 | Claude Sonnet 4.6 | `plot_model_slices`/`plot_borehole_logs` moved into `femtic_viz.py`; `ALPHA_FILE`/`ALPHA_MODE`/`ALPHA_BLANK_THRESH` added |
| 2026-05-27 | vrath / Claude Sonnet 4.6 | `PLOT_XLIM/YLIM/ZLIM` passed to slice panels |
| 2026-05-31 | vrath / Claude Sonnet 4.6 | `invert_x` per-panel key; origin estimation before UTM zone derivation |
| 2026-06-03 | Claude Sonnet 4.6 | **Split** from `femtic_mod_plot.py` → `femtic_mod_plot_slice.py` + `femtic_mod_plot_3d.py`. `BOREHOLE_IN_SLICE`: embedded borehole columns via `plot_model_slices(borehole_sites=...)`; `BOREHOLE_XLIM` now in Ω·m; `z_top="surface"` supported; lat/lon legend; per-trace line-style keys; `BOREHOLE_NPZ` added |
| 2026-06-04 | vrath / Claude Sonnet 4.6 | **Split** from `femtic_mod_plot_slice.py`: standalone borehole step (step 6) and `PLOT_BOREHOLE` flag moved to new `femtic_mod_plot_bh.py`. `BOREHOLE_IN_SLICE` retained here for embedded columns only |
| 2026-06-19 | Claude Sonnet 4.6 | `kind="profile"` added to `PLOT_SLICES`: vertical fence section defined by two endpoint positions (`p1`, `p2`) each accepting model-local / UTM / latlon CRS tags; `strike` derived from p1→p2 azimuth; `dip` fixed at 90°. New helper `resolve_pos_two_point_profile()` in `femtic.py`; `resolve_slice_positions()` extended. |
| 2026-06-26 | vrath / Claude Sonnet 4.6 | Added `TICK_FONTSIZE` (default `7`) and `LABEL_FONTSIZE` (default `8`) config constants; forwarded to `plot_model_slices` as `tick_fontsize` / `label_fontsize`. |
| 2026-08-13 | Claude Sonnet 5 (Anthropic) | Added `femtic_mod_plot_slice_summary.md` output at end of run: writes user-set (UPPERCASE) parameters, script path, and run date/time. |
| 2026-09-06 | Claude Sonnet 5 (Anthropic) | Added `SHOW_MODEL_CENTRE` (default `True`): marks the model origin on `"map"` panels whenever `DISPLAY_COORDS` is `"utm"`/`"latlon"`, via `fviz.plot_model_slices`'s new `show_model_centre` parameter. Override style with a `MAP_MARKERS` entry carrying `"is_model_centre": True`. |
| 2026-10-28 | Claude Sonnet 5.5 (Anthropic) | The parameter summary `femtic_mod_plot_slice_summary.md` is now written to `WORK_DIR` (the working directory set in the script) instead of next to the script: `utl.write_param_summary(fname, out_dir=WORK_DIR)`. Needs the `util.py` of the same date. AI-generated; review before production use. |
| 2026-11-08 | Claude Sonnet 5.5 (Anthropic) | **Several models in one run:** `MODEL_FILE` may be a list, a glob or run directories (`MODEL_ITER` selects the iteration), giving one figure per model plus an optional catalog PDF (`BATCH_*`); new `femtic.resolve_model_files`, `fviz.plot_model_slices_robust`, `fviz.append_catalog_page`. One model behaves exactly as before (checked: the figure is pixel-identical). Takes over the per-member figures of the former `femtic_ens_plot.py`. AI-generated; review before production use. |
| 2026-11-09 | Claude Sonnet 5.5 (Anthropic) | Dipping planes (`kind="plane"`, `dip != 90`) fixed in `femtic_viz.plot_model_slices`: `strike` is the azimuth for every dip, axes are along-strike / down-dip distance from `point`, default extent = plane cut with the `PLOT_XLIM` x `PLOT_YLIM` x `PLOT_ZLIM` box, per-slice `xlim` / `ylim` = along-strike / down-dip range (m, relative to `point`). The CRS-tagged `point` (`"model"` / `"utm"` / `"latlon"`) was already supported by this script. Needs the `femtic_viz.py` of the same date. |
| 2026-11-10 | Claude Sonnet 5.5 (Anthropic) | New `MESH_SWAP_XY` (default `True`), applied before the mesh is read. Slice maps were drawn transposed (x = easting and y = northing exchanged; a NNE structure appeared as ENE). Cause: since 2026-10-11 `femtic.read_femtic_mesh` does not swap the node columns by default, but `mesh.dat` holds x = northing, y = easting. `None` keeps femtic's default, `False` forces no swap. Only the orientation changes. Checked on a synthetic mesh with a N30E block, not on real data. AI-generated; review before production use. |
| 2026-11-11 | Claude Sonnet 5.5 (Anthropic) | New `PLOT_POLYLINES = {name: entry}` (metres; `{}` = none): lines drawn on the slice panels. Maps: plan view, optional `z_range`; ns / ew / plane / profile sections: the parts within `max_dist` (default `PROJECTION_DIST`) of the section plane, projected into it. Points model-local m or CRS-tagged (`"utm"` metres, `"latlon"` degrees, longitude first); `style`, `label`, `kinds`. Passed to `fviz.plot_model_slices(polylines=...)` in the shared options (every figure of a batch). The intersection line of two slices is available as `fviz.slice_intersection_polyline` / `slice_intersection_polylines` (example in the section above); it is not drawn automatically. With `PLOT_POLYLINES = {}` the figures are unchanged. Needs the `femtic_viz.py` of the same date. Tested on a synthetic mesh only. AI-generated; review before production use. |
