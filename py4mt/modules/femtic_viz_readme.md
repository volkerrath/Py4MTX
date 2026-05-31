# femtic_viz_new.py

Visualisation utilities for FEMTIC resistivity models, with **direct procedures that operate on FEMTIC files**
(`mesh.dat` + `resistivity_block_iterX.dat`) **without creating an intermediate NPZ**.

This is a restarted / cleaned version of the previously concatenated `femtic_viz.py`.

---

## Installation / requirements

- `numpy`
- `matplotlib` (optional; required for Matplotlib plotting helpers)
- `scipy` (optional; required for KDTree-based scatter curtain and IDW curtain gridding)
- `pyvista` (optional; required for VTU/VTK export and PyVista sampling)

---

## Quick start

All functionality is accessed programmatically — `femtic_viz` is a module,
not a script.  Import it and call the relevant function directly.

### Export a VTK grid for ParaView

```python
import femtic_viz as fviz

grid = fviz.unstructured_grid_from_femtic("mesh.dat", "resistivity_block_iter0.dat")
grid.save("model.vtu")
```

### Matplotlib map slice (centroid-sampled)

```python
mesh = fviz.read_femtic_mesh("mesh.dat")
block = fviz.read_resistivity_block("resistivity_block_iter0.dat")
rho = fviz.map_regions_to_element_rho(block.region_of_elem, block.region_rho)

ax = fviz.map_slice_from_cells(mesh, rho, z0=-1000, dz=50, mode="tri", mask_max_edge=500)
```

`mode` options: `"tri"` (patch-like, default), `"scatter"` (markers), `"grid"` (IDW; requires `scipy`).

### Exact-intersection slice figure (from `femtic_mod_plot.py`)

```python
fviz.plot_model_slices(
    model_file="resistivity_block_iter10.dat",
    mesh_file="mesh.dat",
    slices=slices_resolved,          # output of fem.resolve_slice_positions
    cmap="turbo_r", clim=[0., 4.],
    display_coords="utm",
    utm_origin_e=229047., utm_origin_n=8184127.,
    utm_zone=19, utm_northern=False,
    plot_file="model_slices.pdf", dpi=300,
)
```

### 3-D PyVista render + VTK export

```python
fviz.plot_model_3d(
    mesh_file="mesh.dat",
    block_file="resistivity_block_iter10.dat",
    slice_x=[0.], slice_y=[0.], slice_z=[5000., 15000.],
    isovalues=[1., 2., 3.],
    plot_file="model_3d.png",
    vtu_file="model.vtu",            # ParaView / Zenodo export
)
```

---

## PyVista sampling on explicit surfaces

If you prefer "true" sampling (no IDW), use:

- `build_curtain_surface(polyline_xy, zmin, zmax, nz, ns)`
- `build_map_surface(x, y, z0)`
- `sample_grid_on_surface(unstructured_grid, surface, scalar="log10_resistivity")`

Example:

```python
import numpy as np
from femtic_viz_new import unstructured_grid_from_femtic, build_curtain_surface, sample_grid_on_surface

grid = unstructured_grid_from_femtic("mesh.dat", "resistivity_block_iter0.dat")
poly = np.loadtxt("profile.csv", delimiter=",")

surf = build_curtain_surface(poly, zmin=0, zmax=-5000, nz=201, ns=501)
sampled = sample_grid_on_surface(grid, surf, scalar="log10_resistivity")

# sampled.point_data now contains the sampled scalar and "s", "z".
```

---

## RTO ensemble diagnostic plots

Two high-level helpers produce joint plots of original vs. perturbed
data / models for a fixed list of ensemble members.  Both follow the
**`data_viz` philosophy**: an optional `ax` (or `axs`) can be passed in;
if `None`, the function creates its own figure.  `(fig, axs)` is always
returned so the caller can further annotate or save.

These functions are called from `femtic_rto_prep.py` but can also be used
independently in notebooks or post-processing scripts.

---

### `plot_data_ensemble`

```python
fig, axs = plot_data_ensemble(
    orig_file,                   # path to template observe.dat
    ens_files,                   # list of perturbed observe.dat paths (one per member)
    sample_indices,              # list of int — which members to plot
    what="rho",                  # str or list: 'rho', 'phase', 'tipper', 'pt'
                                 # e.g. ['rho', 'phase'] → two-column layout
    comps="xy,yx",               # str (shared) or list of same length as what
    show_errors=False,
    show_errors_orig=None,       # override for original curves (None → show_errors)
    show_errors_pert=None,       # override for perturbed curves
    error_style_orig="shade",    # 'shade' | 'bar' | 'both' for original curves
    error_style_pert="shade",    # 'shade' | 'bar' | 'both' for perturbed curves
    n_sites=None,                # MT sites drawn per row; None = all sites
    alpha_orig=1.0,
    alpha_pert=0.6,
    comp_markers=None,           # dict of marker per class; None = DEFAULT_COMP_MARKERS
    markersize=4.0,
    markevery=None,
    figsize=None,                # auto: (5 × n_panels, 3 × n_samples)
    fig=None,
    axs=None,                    # pre-existing axes, shape (n_samples, n_panels)
    out=True,
)
# Returns: fig, axs  — axs.shape = (n_samples, n_panels)
```

**Layout:** rows = number of selected samples; columns = number of entries in
*what*.  Each cell shows the original (solid) and perturbed (dashed) curves for
the randomly drawn site subset, overlaid on the same axes.

Typical multi-panel usage:

```python
fig, axs = fviz.plot_data_ensemble(
    orig_file, ens_files, VIZ_SAMPLES,
    what=["rho", "phase", "tipper"],
    comps="xy,yx",          # shared for rho and phase; ignored for tipper
)
# axs.shape == (len(VIZ_SAMPLES), 3)
```

For per-column component control (e.g. off-diagonal columns only for ρ_a,
all four for phase):

```python
fig, axs = fviz.plot_data_ensemble(
    orig_file, ens_files, VIZ_SAMPLES,
    what=["rho", "phase"],
    comps=["xy,yx", "xx,xy,yx,yy"],
)
```

`show_errors` is a shared shorthand (default `False`).  Use `show_errors_orig`
and `show_errors_pert` for independent control — the template `observe.dat`
typically carries raw measured uncertainties that can be very large at long
periods (dead-band noise), while the perturbed files contain reset relative
errors (compact ±σ bands).  The recommended setting for RTO diagnostics is
therefore `show_errors_orig=False, show_errors_pert=True`.

**Error rendering style** is set independently per curve with `error_style_orig`
and `error_style_pert` (both default to `'shade'`).  Three modes are available:

| `error_style_orig` / `error_style_pert` | Rendering |
|---|---|
| `'shade'` | Semi-transparent `fill_between` band (default). |
| `'bar'` | Discrete `errorbar` caps at each period. |
| `'both'` | Shade *and* bar simultaneously. |

`'bar'` and `'both'` require that the underlying `data_viz` plotter accepts an
`error_style` kwarg; if not, `plot_data_ensemble` falls back to `'shade'`
with a one-time `warnings.warn`.

**Component markers** assign distinct Matplotlib marker symbols to three
component classes, making diagonal, off-diagonal, and invariant components
visually distinguishable at a glance even when many sites are overlaid.

| Class | Components | Default marker |
|---|---|---|
| `'ii'` | `xx`, `yy` (diagonal) | `'o'` (circles) |
| `'ij'` | `xy`, `yx` (off-diagonal) | `'s'` (squares) |
| `'inv'` | invariants (`det`, `bahr`, …), tipper, PT | `'^'` (triangles up) |

The mapping is controlled by `comp_markers` (a partial or full dict merged with
`DEFAULT_COMP_MARKERS`).  Pass `comp_markers=None` to use the defaults; pass
`comp_markers={}` to disable markers entirely (lines only).  `markersize`
(default `4.0` pt) and `markevery` (default `None` = every period) are
applied post-hoc: because `data_viz` plotters hardcode their own `marker=`
argument inside `ax.loglog`, forwarding `marker` via `**line_kw` would raise
a `TypeError`.  Instead, `plot_data_ensemble` snapshots `ax.lines` before each
plotter call and patches the marker properties on the new `Line2D` objects
after the call — fully non-invasive across all `data_viz` versions.

Data files are read via the internal `_observe_to_site_list()` helper,
which calls `femtic.observe_to_site_viz_list()` — the authoritative FEMTIC
reader. FEMTIC's `observe.dat` stores impedance in **SI Ω**; because
`data_viz.datadict_to_plot_df` assumes **mV/km/nT** (MT field units),
`_observe_to_site_list` scales Z by `1/(μ₀ × 10³)` before passing to the
plotter.  VTF and PT sites are handled via the raw parser path.  The plotter
is called once per site so all sites are overlaid on the same axes within
each row.

---

### `plot_model_ensemble`

```python
fig, axs = plot_model_ensemble(
    orig_mod_file,       # path to template resistivity block
    ens_mod_files,       # list of perturbed block paths (one per member)
    mesh_file,           # path to shared mesh.dat
    sample_indices,      # list of int — which members to plot
    slices,              # list of 1-5 slice dicts (see below)
    mode="tri",          # 'tri' | 'scatter' | 'grid'
    log10=True,
    cmap="jet_r",
    clim=None,           # (vmin, vmax) in log10(Ohm.m); auto if None
    xlim=None,           # easting limits for map slices (m); None = auto
    ylim=None,           # northing / profile-distance limits (m); None = auto
    zlim=None,           # depth limits for curtain slices (m); None = auto
    mesh_lines=False,    # overlay triangulation edges on filled patches
    mesh_lw=0.3,         # mesh edge line width (pt)
    mesh_color="k",      # mesh edge colour
    figsize=None,
    fig=None,
    axs=None,            # pre-existing axes, shape (2*len(sample_indices), len(slices))
    out=True,
)
```

**Layout:** rows = 2 x number of selected samples (original row + perturbed
row per block); columns = number of slices.  The original and perturbed models
sit directly above/below each other for easy visual comparison.

The mesh is read once and shared across all samples and slices.  Colour limits
are set globally from the original model (or from `clim`) so all panels are
directly comparable.

Each entry in `slices` is a dict with `'type': 'map'` or `'type': 'curtain'`
and the keyword arguments forwarded to `map_slice_from_cells` /
`curtain_from_cells`:

```python
slices = [
    {"type": "map",     "z0": -500,  "dz": 50},
    {"type": "map",     "z0": -2000, "dz": 50},
    {"type": "curtain",
     "polyline": np.array([[0., 0.], [10000., 0.]]),
     "width": 500},
]
```

---

## Air / ocean conventions

Many FEMTIC workflows treat:

- region 0 as **air**
- region 1 as **ocean**

By default, `unstructured_grid_from_femtic()` and CLI plotting apply:

- air → `NaN` (transparent/blank in most plots)
- ocean → `0.3` Ohm·m (typical seawater resistivity)

In Matplotlib plots (`map_slice_from_cells`, `curtain_from_cells`, and the
low-level `plot_points_matplotlib`, `plot_map_grid_matplotlib`,
`plot_curtain_matplotlib`) ocean cells are additionally rendered in a flat
**light-grey** colour so they remain visually distinct from the resistivity
colour scale regardless of the chosen colormap and colour limits.

You can control this via `prepare_rho_for_plotting()` and via the two
keyword arguments available on all Matplotlib plotting helpers:

| Parameter      | Default        | Description                                              |
|----------------|----------------|----------------------------------------------------------|
| `ocean_value`  | `3.0e-1`       | Resistivity (Ohm·m) used to tag ocean cells.            |
| `ocean_color`  | `'lightgrey'`  | Flat Matplotlib colour for ocean cells.  `None` = use colormap. |

To suppress the flat-colour treatment entirely, pass `ocean_color=None`.
To change the colour, pass any valid Matplotlib colour string, e.g.
`ocean_color='#b0c4de'` (light steel blue).

You can also control this via `prepare_rho_for_plotting()` or by setting
`apply_plotting_conventions=False`.


---

## 2-D slice figure (`plot_model_slices`)

Produces a multi-panel Matplotlib figure of axis-parallel model slices using
**exact tetrahedron-plane intersection** — no selection slab, no `dw` parameter.
Every tetrahedron straddling the cutting plane contributes an exact triangle or
quadrilateral polygon.  Moved here from `femtic_mod_plot.py`.

```python
fviz.plot_model_slices(
    model_file = "resistivity_block_iter10.dat",
    mesh_file  = "mesh.dat",
    slices     = slices_resolved,   # pre-resolved by fem.resolve_slice_positions

    # colouring
    cmap  = "turbo_r",
    clim  = [0., 4.],               # log10(Ω·m); None = auto

    # axis limits (model-local m); per-panel keys override these
    xlim  = [-20000., 20000.],
    ylim  = [-20000., 20000.],
    zlim  = [     0.,  20000.],

    # display coordinate system
    display_coords   = "utm",        # "model" | "utm" | "latlon"
    utm_origin_e     = 229047.0,     # mesh-centre UTM easting [m]
    utm_origin_n     = 8184127.0,    # mesh-centre UTM northing [m]
    utm_zone         = 19,
    utm_northern     = False,
    utm_to_latlon_fn = utl.utm_to_latlon_zn,    # for latlon tick formatting
    latlon_to_model_fn = fem.latlon_to_model,   # for map_markers placement

    # site overlay
    site_xys          = site_xys,   # [(label, x_m, y_m, elev_m), ...]
    sites_in_maps     = True,
    sites_in_slices   = True,
    site_marker       = dict(marker="v", color="black", ms=4, zorder=10),
    site_marker_slices= dict(marker="o", color="black", ms=4, zorder=10),
    projection_dist   = 3000.,      # m; None = all sites on all panels

    # extra map markers (lat/lon)
    map_markers = [
        dict(latlon=[-16.35, -70.90], marker="*", color="red",
             ms=10, name="Summit"),
    ],

    # figure layout
    depth_km     = True,
    horiz_km     = True,
    equal_aspect = True,
    nrows        = 2,
    ncols        = 2,
    panel_height = 8.0 / 2.54,   # pass in inches (divide cm by 2.54)

    plot_file = "model_slices.pdf",
    dpi       = 300,
)
```

### Parameters summary

| Parameter | Default | Description |
|---|---|---|
| `model_file`, `mesh_file` | — | Resistivity block and `mesh.dat` |
| `slices` | — | Pre-resolved slice-spec list (output of `fem.resolve_slice_positions`) |
| `cmap` | `"turbo_r"` | Matplotlib colormap |
| `clim` | `None` | `[vmin, vmax]` log₁₀(Ω·m); `None` = auto |
| `xlim`, `ylim`, `zlim` | `None` | Global axis limits (model-local m) |
| `ocean_color` | `"lightgrey"` | Flat colour for ocean polygons; `None` = colormap |
| `ocean_value` | `0.25` | Ω·m sentinel for ocean cells |
| `air_bgcolor` | `None` | Axes facecolor for air region |
| `site_xys` | `None` | `[(label, x_m, y_m, elev_m), …]` in model-local m |
| `obs_coords_only` | `False` | Sites from observe.dat only (suppresses UTM/latlon display) |
| `projection_dist` | `None` | Max distance [m] from curtain for site to appear |
| `sites_in_maps` | `True` | Site markers on map panels |
| `sites_in_slices` | `False` | Site markers on curtain/plane panels |
| `site_marker` | `dict(marker="v", …)` | Matplotlib kwargs for map markers |
| `site_marker_slices` | `dict(marker="o", …)` | Matplotlib kwargs for curtain markers |
| `map_markers` | `None` | Extra markers (lat/lon dicts) on map panels |
| `display_coords` | `"model"` | `"model"` / `"utm"` / `"latlon"` |
| `utm_origin_e`, `utm_origin_n` | `0.0` | Mesh-centre UTM [m] |
| `utm_zone`, `utm_northern` | `1`, `True` | UTM zone and hemisphere |
| `utm_to_latlon_fn` | `None` | Callable for lat/lon tick formatting |
| `latlon_to_model_fn` | `None` | Callable for `map_markers` placement |
| `plot_file` | `None` | Save path; `None` = `plt.show()` |
| `dpi` | `200` | Saved-figure DPI |
| `equal_aspect` | `True` | Equal aspect on map/ns/ew panels (when scales match) |
| `depth_km` | `False` | Depth axis in km on curtain/plane panels |
| `horiz_km` | `False` | Horizontal axis in km in `"model"` mode |
| `nrows`, `ncols` | `None` | Grid shape; surplus cells hidden |
| `panel_height` | `5.0` | Row height **in inches** |
| `panel_width` | `None` | Column width in inches; `None` = auto from aspect |
| `figsize` | `None` | `[width, height]` in inches; overrides auto sizing |

Per-panel `invert_x` key in each slice dict (applies to `ns`, `ew`, `plane`
kinds only): when `True`, calls `ax.invert_xaxis()` after rendering so the
horizontal axis reads right-to-left.  Use for comparison with sections from
other software that uses the opposite orientation convention.  Default `False`;
has no effect on `map` panels.

---

## 1-D borehole log (`plot_borehole_logs`)

Samples the resistivity model along vertical boreholes using point-in-element
search (`fem.extract_borehole_log`, exact barycentric test) and plots
log₁₀(ρ) vs depth.  Moved here from `femtic_mod_plot.py`.

```python
fviz.plot_borehole_logs(
    model_file = "resistivity_block_iter10.dat",
    mesh_file  = "mesh.dat",
    borehole_sites = [
        dict(name="BH-centre", x=0.0,  y=0.0,
             z_top=0., z_bot=20000., dz=200.),
    ],
    resolve_xy_fn = _resolve_borehole_xy,  # converts CRS-tagged x/y to model-local m
    ocean_value   = 0.25,
    clim          = [0., 4.],
    shared        = True,
    plot_file     = "boreholes.pdf",
    dpi           = 300,
)
```

`resolve_xy_fn` is a script-level closure (defined in `femtic_mod_plot.py`)
that calls `fem.resolve_pos_x` / `fem.resolve_pos_y` with the current mesh
origin and UTM zone.  When `x` / `y` are already plain floats in model-local
metres, pass `resolve_xy_fn=None`.

### Parameters summary

| Parameter | Default | Description |
|---|---|---|
| `model_file`, `mesh_file` | — | Resistivity block and `mesh.dat` |
| `borehole_sites` | — | List of spec dicts (`name`, `x`, `y`, `z_top`, `z_bot`, `dz`) |
| `resolve_xy_fn` | `None` | `(spec) → (x_m, y_m)` CRS converter; `None` = `x`/`y` are model-local floats |
| `ocean_value` | `0.25` | Ω·m sentinel for ocean cells |
| `clim` | `None` | x-axis limits `[log10_min, log10_max]`; `None` = auto |
| `borehole_style` | `None` | Matplotlib line kwargs (default `lw=1.2, marker="none"`) |
| `shared` | `True` | `True` = all traces on one axes; `False` = one panel each |
| `plot_file` | `None` | Save path; `None` = `plt.show()` |
| `dpi` | `200` | Saved-figure DPI |

---

## 3-D model plot (`plot_model_3d`)

High-level function for interactive or static 3-D rendering of a FEMTIC
resistivity model directly from `mesh.dat` + `resistivity_block_iterX.dat`.
Requires PyVista (`conda install -c conda-forge pyvista`).

```python
fviz.plot_model_3d(
    mesh_file   = "mesh.dat",
    block_file  = "resistivity_block_iter10.dat",

    # scalar & colouring
    scalar      = "log10_resistivity",  # or "resistivity"
    clim        = [0., 4.],             # log10(Ω·m)
    cmap        = "turbo_r",

    # axis-aligned slices (model-local metres, z positive-down)
    slice_x     = [0.],                 # one YZ plane
    slice_y     = [0.],                 # one XZ plane
    slice_z     = [5000., 15000.],      # two horizontal maps

    # oblique planes
    slice_planes = [
        dict(origin=[0., 0., 8000.], normal=[1., 1., 0.]),
    ],

    # iso-surfaces
    isovalues   = [1., 2., 3.],        # 10 / 100 / 1000 Ω·m boundaries
    iso_opacity = 0.35,

    # output — .html = interactive WebGL, .png = screenshot, None = live window
    plot_file   = "model_3d.html",
)
```

### Parameters summary

| Parameter | Default | Description |
|---|---|---|
| `scalar` | `"log10_resistivity"` | Cell-data scalar to display |
| `clim` | `None` | Colour limits; `None` = PyVista auto |
| `cmap` | `"turbo_r"` | Colormap for slices and iso-surfaces |
| `slice_x` | `None` | x-positions of YZ cutting planes (m) |
| `slice_y` | `None` | y-positions of XZ cutting planes (m) |
| `slice_z` | `None` | z-positions of XY cutting planes (m) |
| `slice_planes` | `None` | List of `dict(origin, normal)` for oblique planes |
| `isovalues` | `None` | Iso-surface levels in scalar units |
| `iso_opacity` | `0.4` | Iso-surface opacity (0–1) |
| `iso_cmap` | same as `cmap` | Colormap for iso-surfaces |
| `show_edges` | `False` | Overlay mesh edges on slices (slow for large grids) |
| `background` | `"white"` | Scene background colour |
| `window_size` | `[1600, 900]` | Window / screenshot resolution in pixels |
| `ocean_value` | `0.3` | Ω·m sentinel for ocean cells |
| `plot_file` | `None` | `.vtu`/`.vtk` → VTK grid (no render); `.html` → interactive WebGL (requires `pyvista[jupyter]`; falls back to `.png`); `.png`/`.jpg` → screenshot; `None` → live window |
| `vtu_file` | `None` | Separate VTK grid export (`.vtu` recommended) written before rendering; cell-centred. Suitable for ParaView / Zenodo. |
| `screenshot_scale` | `2` | Anti-aliasing scale for screenshot output |

If no slices or iso-surfaces are defined, a default orthogonal triple (one
XY, YZ, and XZ plane through the mesh centre) is added automatically.

---

## Ensemble slice plot (`plot_ensemble_slices`)

Joint figure of all ensemble members using the same exact tetrahedron-plane
intersection as `femtic_mod_plot.plot_model_slices`.  One row per member,
columns = slices.  Optional statistical summary rows appended at the bottom.

```python
fviz.plot_ensemble_slices(
    member_files = [
        "ensemble/rto_0/resistivity_block_iter10.dat",
        "ensemble/rto_1/resistivity_block_iter10.dat",
        # …
    ],
    mesh_file  = "mesh.dat",
    slices     = [                        # model-local metres
        dict(kind="map", z0=5000.),
        dict(kind="map", z0=15000.),
        dict(kind="ns",  x0=0.),
        dict(kind="ew",  y0=0.),
    ],
    labels     = ["RTO-0", "RTO-1"],     # None → "Member 0", …
    stat_rows  = ["mean", "std"],        # rows appended after members
    cmap       = "turbo_r",
    clim       = [0., 4.],              # log10(Ω·m); None = auto
    xlim       = [-20000., 20000.],
    ylim       = [-20000., 20000.],
    zlim       = [-6000., 15000.],
    ocean_value    = 0.25,
    per_member_file = True,             # also save _member0.pdf, _member1.pdf …
    plot_file  = "ensemble_slices.pdf",
    dpi        = 300,
)
```

### Parameters summary

| Parameter | Default | Description |
|---|---|---|
| `member_files` | — | List of resistivity block paths, one per member |
| `mesh_file` | — | Shared `mesh.dat` |
| `slices` | — | Slice-spec list in model-local metres (kinds: `"map"`, `"ns"`, `"ew"`, `"plane"`); per-panel `invert_x=True` flips horizontal axis on curtain/plane panels |
| `labels` | `None` | Row label per member; `None` → "Member 0", … |
| `stat_rows` | `("mean", "std")` | Stat rows after member rows; subset of `"mean"`, `"std"`, `"median"` |
| `cmap` | `"turbo_r"` | Colormap for member / mean / median rows |
| `clim` | `None` | `[vmin, vmax]` log₁₀(Ω·m); `None` = auto from ensemble |
| `xlim`, `ylim`, `zlim` | `None` | Global axis limits in model-local metres |
| `ocean_color` | `"lightgrey"` | Flat colour for ocean cells |
| `ocean_value` | `0.25` | Ω·m sentinel for ocean cells |
| `air_bgcolor` | `None` | Axes facecolor for air / background |
| `plot_file` | `None` | Joint figure path; `None` → interactive window |
| `per_member_file` | `False` | Save `_memberN` figures alongside the joint figure |
| `dpi` | `200` | Saved-figure DPI |

The `"std"` row is rendered on a separate `cividis` colormap anchored at zero;
mean and median share the main colormap / clim.  NaN (air, missing) cells are
excluded from all statistics.

---

## Provenance

| Date       | Author | Change                                                      |
|------------|--------|-------------------------------------------------------------|
| 2025-12-23 | vrath  | Created (with ChatGPT GPT-5 Thinking).                      |
| 2026-03-24 | Claude | Added `plot_data_ensemble` and `plot_model_ensemble`.        |
| 2026-03-29 | Claude | Added `n_sites` to `plot_data_ensemble` (random site draw). |
|            |        | Fixed `ocean_value` default: `1e-10` → `3e-1` Ohm·m.       |
|            |        | Added `ocean_color` (`'lightgrey'`) to all Matplotlib       |
|            |        | plotting helpers for flat-colour ocean rendering.           |
| 2026-03-30 | Claude | Fixed `plot_data_ensemble`: replaced non-existent           |
|            |        | `fem.read_observe()` with `fem.read_observe_dat()`; added   |
|            |        | `_observe_to_site_list()` to convert the nested             |
|            |        | blocks→sites structure and build `Z`/`Z_err`/`T`/`P`       |
|            |        | complex arrays; plotter now iterates per-site.              |
|            |        | Fixed ρ_a unit mismatch: FEMTIC Z is in SI Ω but            |
|            |        | `data_viz` expects mV/km/nT; `_observe_to_site_list` now   |
|            |        | uses `fem.observe_to_site_viz_list()` and scales Z by       |
|            |        | `1/(μ₀×10³)` before passing to `datadict_to_plot_df`.      |
|            |        | Split `show_errors` into `show_errors_orig` / `show_errors_pert` |
|            |        | (both default `False`); raw template errors are noisy at    |
|            |        | long periods; perturbed files carry reset relative errors.  |
| 2026-04-02 | Claude | Added `xlim`, `ylim`, `zlim` to `plot_model_ensemble`.      |
|            |        | Map slices: `xlim`/`ylim` clip easting/northing axes.       |
|            |        | Curtain slices: `ylim` clips profile-distance axis,         |
|            |        | `zlim` clips depth axis. Per-slice dict keys override       |
|            |        | function-level defaults.                                    |
| 2026-04-03 | Claude | Added `alpha_orig`/`alpha_pert` to `plot_data_ensemble`.    |
|            |        | Added `mesh_lines`/`mesh_lw`/`mesh_color` to                |
|            |        | `plot_model_ensemble`, `map_slice_from_cells`,              |
|            |        | `curtain_from_cells`, and `plot_points_matplotlib`.         |
|            |        | `triplot` overlay drawn after `tripcolor` when enabled.     |
| 2026-04-12 | Claude | Added `comp_markers` / `markersize` / `markevery` to        |
|            |        | `plot_data_ensemble`: distinct marker symbols per component  |
|            |        | class (`'ii'` circles, `'ij'` squares, `'inv'` triangles). |
|            |        | Added `error_style_orig` / `error_style_pert`               |
|            |        | (`'shade'` / `'bar'` / `'both'`); no shared fallback param. |
|            |        | Added module-level `DEFAULT_COMP_MARKERS` dict and          |
|            |        | `_comp_class()` helper for component classification.        |
|            |        | Fixed legend: all components appear once per row.           |
|            |        | Multi-panel redesign: `what` accepts a list of panel types  |
|            |        | (e.g. `['rho', 'phase', 'tipper']`); `comps` may be a      |
|            |        | single string or per-panel list; returned `axs` shape is   |
|            |        | `(n_samples, n_panels)`.                                    |
| 2026-05-13 | Claude | Added `plot_model_3d`: PyVista 3-D renderer with axis-aligned |
|            |        | x/y/z plane slices, arbitrary oblique planes, and iso-      |
|            |        | surfaces of any cell-data scalar. Outputs interactive HTML  |
|            |        | (WebGL) or static screenshot. Graceful skip when PyVista    |
|            |        | is absent. Called from `femtic_mod_plot.py` step 5.         |
| 2026-05-13 | Claude | Added `plot_ensemble_slices`: joint member × slice figure   |
|            |        | using exact tet-plane intersection. Mesh and geometry       |
|            |        | precomputed once; member resistivities swapped per row.     |
|            |        | Optional mean/std/median stat rows; std on separate         |
|            |        | sequential colormap. Called from `femtic_mod_plot.py`       |
|            |        | step 6 and from `femtic_rto_prep.py` / `femtic_gst_prep.py`.|
| 2026-05-26 | Claude Sonnet 4.6 | Moved `plot_model_slices` (exact tet-plane intersection, all |
|            |        | inner geometry helpers: `_tet_plane_intersection`, `_slice_geometry`, `_plot_slice_panel`, `_strike_dip_to_normal`, `_plane_basis`) and `plot_borehole_logs` from `femtic_mod_plot.py` into this module. All formerly-implicit config globals (`DISPLAY_COORDS`, `UTM_ORIGIN_*`, `UTM_ZONE`, `SITE_MARKER`, etc.) are now explicit keyword parameters. Added `import math`, `import os`. |
|            |        | `plot_model_3d`: added `vtu_file` parameter (cell-centred `.vtu`/`.vtk` export for ParaView / Zenodo); `plot_file=*.vtu/.vtk` accepted directly. Added `ImportError` fallback for HTML export when `trame_vtk` absent. |
| 2026-05-31 | vrath / Claude Sonnet 4.6 (Anthropic) | `plot_model_slices`, `plot_ensemble_slices`: added per-panel `invert_x` key in slice-spec dicts. When `True` on an `ns`, `ew`, or `plane` panel, calls `ax.invert_xaxis()` after rendering (after axis limits are applied) to flip the horizontal axis left-to-right. Enables direct comparison with sections from other software that uses the opposite orientation convention. Default `False`; no effect on `map` panels. In `plot_ensemble_slices`, stored in `slice_geom` dicts so the geometry precomputation step is unaffected. |

Author: Volker Rath (DIAS)
