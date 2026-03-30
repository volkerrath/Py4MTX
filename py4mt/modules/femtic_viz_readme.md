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

## Quick start: direct from FEMTIC files (no NPZ)

> Deprecated CLI aliases kept for compatibility: `map-scatter`, `curtain-scatter`.


### 1) Create a PyVista grid and export VTU

```bash
python femtic_viz_new.py export-vtu --mesh mesh.dat --block resistivity_block_iter0.dat --out model.vtu
```

Programmatically:

```python
from femtic_viz_new import unstructured_grid_from_femtic

grid = unstructured_grid_from_femtic("mesh.dat", "resistivity_block_iter0.dat")
grid.save("model.vtu")
```

### 2) Map slice (Matplotlib): **patches** (`tri`), scatter, or regular grid

Default is **patch-like** plotting via Delaunay triangulation in the slice plane:

```bash
python femtic_viz_new.py map --mesh mesh.dat --block resistivity_block_iter0.dat --z0 -1000 --dz 50 --mode tri
```

Useful options:

- `--mode scatter` : markers (no connectivity)
- `--mode tri` : coloured patches from triangulated points (**default**)
- `--mode grid` : IDW to regular grid + `pcolormesh` (requires `scipy`)
- `--mask-max-edge` : suppress long/bridging triangles in `tri` mode

Programmatically:

```python
from femtic_viz_new import map_slice_from_cells

ax = map_slice_from_cells(mesh, rho, z0=-1000, dz=50, mode="tri", mask_max_edge=500)
```

### 3) Curtain slice (Matplotlib): **patches** (`tri`), scatter, or regular grid

Prepare a CSV polyline `profile.csv` with two columns `x,y` (no header).

Patch-like plotting in `(s, z)` from triangulated points:

```bash
python femtic_viz_new.py curtain --mesh mesh.dat --block resistivity_block_iter0.dat --polyline profile.csv --width 500 --mode tri --mask-max-edge 500
```

Other modes:

- `--mode scatter` : markers (no connectivity)
- `--mode grid` : IDW to regular `(s, z)` grid + `pcolormesh` (requires `scipy`)
  - optional: `--zmin ... --zmax ...` (otherwise inferred from points)

Programmatically:

```python
import numpy as np
from femtic_viz_new import curtain_from_cells

poly = np.loadtxt("profile.csv", delimiter=",")
ax = curtain_from_cells(mesh, rho, poly, width=500, mode="tri", mask_max_edge=500)
```

Notes:

- `tri` mode gives the "coloured patches" look, but it is still based on centroid samples
  (not an exact tetra/plane intersection).
- If you see triangles "bridging" gaps, increase `--mask-max-edge`.


### 4) Curtain slice on a regular (s–z) grid (IDW)

```bash
python femtic_viz_new.py curtain-idw --mesh mesh.dat --block resistivity_block_iter0.dat --polyline profile.csv --zmin 0 --zmax -5000
```

Programmatically:

```python
from femtic_viz_new import curtain_grid_idw, plot_curtain_matplotlib

s, z, V = curtain_grid_idw(mesh, rho, poly, zmin=0, zmax=-5000, nz=201, ns=501, k=8, power=2.0)
ax = plot_curtain_matplotlib(s, z, V, log10=True)
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
    orig_file,           # path to template observe.dat
    ens_files,           # list of perturbed observe.dat paths (one per member)
    sample_indices,      # list of int — which members to plot
    comps="xy,yx",       # impedance components (ignored for tipper / pt)
    what="rho",          # 'rho' | 'phase' | 'tipper' | 'pt'
    show_errors=False,   # shorthand for both curves; default False
    show_errors_orig=None,  # override for original curves only (None → show_errors)
    show_errors_pert=None,  # override for perturbed curves only (None → show_errors)
    figsize=None,        # (width, height) in inches; auto if None
    fig=None,            # pre-existing Figure
    axs=None,            # pre-existing axes, shape (len(sample_indices),)
    out=True,
)
```

**Layout:** one subplot row per selected sample.  Within each row the original
curve is drawn **solid** and the perturbed curve **dashed** on the same axes,
so differences are immediately visible.

`show_errors` is a shared shorthand (default `False`).  Use `show_errors_orig`
and `show_errors_pert` for independent control — the template `observe.dat`
typically carries raw measured uncertainties that can be very large at long
periods (dead-band noise), while the perturbed files contain reset relative
errors (compact ±σ bands).  The recommended setting for RTO diagnostics is
therefore `show_errors_orig=False, show_errors_pert=True`.

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

Author: Volker Rath (DIAS)
