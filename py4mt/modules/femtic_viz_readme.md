# femtic_viz_new.py

Visualisation utilities for FEMTIC resistivity models, with **direct procedures that operate on FEMTIC files**
(``mesh.dat`` + ``resistivity_block_iterX.dat``) **without creating an intermediate NPZ**.

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

- `tri` mode gives the “coloured patches” look, but it is still based on centroid samples
  (not an exact tetra/plane intersection).
- If you see triangles “bridging” gaps, increase `--mask-max-edge`.


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

If you prefer “true” sampling (no IDW), use:

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

## Air / ocean conventions

Many FEMTIC workflows treat:

- region 0 as **air**
- region 1 as **ocean**

By default, `unstructured_grid_from_femtic()` and CLI plotting apply:

- air → `NaN` (transparent/blank in many plots)
- ocean → `1e-10` Ohm·m

You can control this via `prepare_rho_for_plotting()` or by setting
`apply_plotting_conventions=False`.

---

Author: Volker Rath (DIAS)  
Created with the help of ChatGPT (GPT-5 Thinking) on 2025-12-23
