# README_femticviz.md

Visualisation helpers for **FEMTIC** meshes and resistivity models.

This README documents the cleaned `femtic_viz` module (the version generated here as
`femtic_viz_clean.py`). It is designed to be compatible with the NPZ products created
by `femtic.py`, especially the element-based NPZ written by
`femtic.save_element_npz_with_mesh_and_regions(...)`.

---

## What this module does

- **Curtain slices** (vertical sections) along an arbitrary XY polyline
  - Matplotlib plot helper
  - Optional PyVista `StructuredGrid` builder (if you want 3‑D/interactive plotting)
- **Map slices** (horizontal/depth-window slices) on a regular XY grid
  - Matplotlib plot helper
- **Air / ocean handling** for plotting:
  - *Air* (region 0) can be masked to `NaN` (transparent/blank)
  - *Ocean* (region 1) can be kept as-is or forced to a user-selected value

Conventions: **z is positive downward** (depth). This matches typical FEMTIC usage.

---

## Dependencies

Required:
- Python ≥ 3.10
- NumPy

Optional (only needed for specific features):
- SciPy (recommended): fast kNN interpolation via `scipy.spatial.cKDTree`
- Matplotlib: plotting
- PyVista: building `StructuredGrid` output (curtain grids)

The module is written so it can be imported without SciPy/Matplotlib/PyVista; those
imports happen lazily inside the relevant functions.

---

## Input data format (NPZ)

The visualisation routines expect an NPZ file with at least:

- `centroid` **or** `centroids` : shape `(n_cells, 3)`  
  Cell centroids (x, y, z) with **z positive down**.
- `log10_resistivity` **or** `log10_rho` : shape `(n_cells,)`  
  Per-cell log10 resistivity.

Optional (recommended):

- `region` (or one of: `regions`, `reg`, `elem_region`, `element_region`) : `(n_cells,)`  
  Region index per cell. Used for masking air and optionally forcing ocean.

Air/ocean convention used by the helpers (only applied if `region` is available):
- `region == 0` → air
- `region == 1` → ocean

### Producing a compatible NPZ with `femtic.py`

In `femtic.py`, the typical route is:

1. Read a FEMTIC mesh (`nodes`, `conn`) and resistivity block (`block`)
2. Build element arrays (centroids, per-cell log10 resistivity, region index, …)
3. Save everything as NPZ

Functions involved (names from `femtic.py`):
- `build_element_arrays(...)`
- `save_element_npz_with_mesh_and_regions(...)`

---

## Quick start

### 1) Install / place the module

If you generated `femtic_viz_clean.py`, you can either:

- Rename it to `femtic_viz.py` and put it next to `femtic.py`, **or**
- Keep the filename and import it explicitly.

Example:
```bash
cp femtic_viz_clean.py femtic_viz.py
```

### 2) Curtain plot (Matplotlib) from NPZ

Command-line example (repeat `--xy` to define the polyline):

```bash
python femtic_viz.py curtain   --npz model_with_mesh.npz   --zmin 0 --zmax 6000   --nz 251 --ns 401   --interp idw --k 8 --power 2.0   --xy 450000 8200000   --xy 470000 8200000   --xy 490000 8210000
```

Notes:
- `--interp nearest` is faster (no IDW), but less smooth.
- Use `--linear` to plot **Ohm·m** instead of log10(Ohm·m).
- Use `--ocean-log10 -10` (example) to force ocean to a chosen log10 resistivity.
- Add `--no-mask-air` if you want air values to remain visible.

### 3) Map slice (Matplotlib) from NPZ

```bash
python femtic_viz.py map   --npz model_with_mesh.npz   --zmin 1000 --zmax 1500   --nx 401 --ny 401   --interp idw --k 8 --power 2.0   --bounds 430000 510000 8180000 8230000
```

`--zmin/--zmax` define a *depth window*; the map uses cells whose centroid z lies
in that window, then interpolates onto a regular XY grid.

---

## Python API overview

Typical usage pattern:

```python
from femtic_viz import (
    curtain_slice_from_npz,
    plot_curtain_matplotlib,
    map_slice_from_npz,
    plot_map_slice_matplotlib,
)

curt = curtain_slice_from_npz(
    "model_with_mesh.npz",
    polyline_xy=[[0, 0], [10000, 0]],
    zmin=0, zmax=6000,
    nz=201, ns=301,
    interp="idw", k=8, power=2.0,
    mask_air=True,
    ocean_log10_rho=None,
)

fig, ax = plot_curtain_matplotlib(curt)

m = map_slice_from_npz(
    "model_with_mesh.npz",
    zmin=1000, zmax=1500,
    nx=301, ny=301,
    bounds=None,        # or (xmin, xmax, ymin, ymax)
    interp="idw", k=8, power=2.0,
)
fig2, ax2 = plot_map_slice_matplotlib(m)
```

### PyVista structured curtain grid

If PyVista is installed:

```python
from femtic_viz import build_curtain_structured_grid_from_npz

grid = build_curtain_structured_grid_from_npz(
    "model_with_mesh.npz",
    polyline_xy=[[0, 0], [10000, 0]],
    zmin=0, zmax=6000,
    nz=201, ns=301,
    interp="idw",
)
# grid is a pyvista.StructuredGrid
```

---

## Interpolation notes (performance)

- The IDW path uses **k-nearest neighbors** via `scipy.spatial.cKDTree` and is
  suitable for large meshes (100k–millions of cells).
- `k` controls smoothness/cost; typical values: 6–16.
- `power` controls how quickly weights decay with distance; typical: 1.5–3.0.
- If SciPy is not installed, only the `"nearest"` interpolation is available.

---

## Troubleshooting

- **ImportError for SciPy/Matplotlib/PyVista**  
  Only install what you need. The module imports optional dependencies lazily.

- **Blank plot / nothing visible**  
  Check that your `zmin/zmax` window overlaps your model depth range and that
  you didn’t mask everything (e.g., plotting only air).

- **Wrong “up/down” direction**  
  The module assumes **z positive down**. If your NPZ uses z positive up,
  flip sign before exporting or when constructing `centroid`.

---

## File list

- `femtic_viz.py` (or `femtic_viz_clean.py`): visualisation module
- `README_femticviz.md`: this document



---

## Native PyVista (no interpolation)

If your NPZ was produced by `femtic.py` and contains `nodes` + `conn`, you can
plot **directly on the unstructured tetrahedral mesh** without resampling to any
regular grid.

The updated module provides these CLI subcommands:

### Plane slice at depth z0

```bash
python femtic_viz.py pv-plane \
  --npz model_with_mesh.npz \
  --z0 1200 \
  --scalar log10_resistivity
```

### Depth window (slab) rendered from above

```bash
python femtic_viz.py pv-window \
  --npz model_with_mesh.npz \
  --zmin 1000 --zmax 1500 \
  --scalar log10_resistivity
```

### Curtain as a set of true mesh slices along a polyline

```bash
python femtic_viz.py pv-curtain \
  --npz model_with_mesh.npz \
  --zmin 0 --zmax 6000 \
  --ns 51 --corridor 500 \
  --scalar log10_resistivity \
  --xy 450000 8200000 \
  --xy 470000 8200000 \
  --xy 490000 8210000
```

Notes:
- `--corridor` first extracts only cells close to the polyline (faster and local).
- `--no-mask-air` keeps air cells (region 0) if present.
- `--ocean-log10 VALUE` forces ocean cells (region 1) to a chosen value.

