# femtic_viz — Unified visualization utilities for FEMTIC resistivity models

`femtic_viz.py` collects the various visualization helpers that were
previously spread over several standalone scripts:

- `femtic_resistivity_plotting.py`
- `femtic_borehole_viz.py`
- `femtic_slice_matplotlib.py`
- `femtic_map_slice_matplotlib.py`
- `femtic_slice_pyvista.py`
- `femtic_map_slice_pyvista.py`

The goal is that **all FEMTIC plotting utilities can be imported from a single
module**.

It provides:

- Matplotlib helpers (1D & 2D)
- PyVista helpers (2D structured grids in VTK space)
- Consistent handling of special resistivity blocks (air, ocean)
- A common NPZ model format for slices


## 1. Scope and conventions

The module assumes a **FEMTIC-style unstructured resistivity model**:

- A tetrahedral mesh (`mesh.dat` etc., handled elsewhere).
- A 1D **block resistivity vector** stored in text files such as  
  `resistivity_block_iter0.dat`.
- A preprocessed **NPZ file** with per-element information, typically
  produced by `femtic_mesh_to_npz.py`:

  - `centroid` — `(nelem, 3)` array of cell centroids `[x, y, z]`
  - `log10_resistivity` — `(nelem,)` array with `log10(ρ [Ω·m])`
  - `flag` — optional `(nelem,)` integer array with `1` for “fixed” cells  
    (these are automatically excluded from interpolation in the slice
    routines).

### Air and ocean blocks

For the **block resistivity vector** (as read from
`resistivity_block...dat`), the first two entries have a special meaning:

- index `0` — air  
- index `1` — ocean

`femtic_viz` provides helpers that:

- set **air → NaN** for plotting (transparent/white)
- optionally enforce a **fixed ocean resistivity** (e.g. `1e-10 Ω·m`)

These conventions are *only* applied when you explicitly call the helper
functions documented below, so they do not interfere with inversion or
other processing.


## 2. Dependencies

Core:

- `numpy`
- `matplotlib`

Optional (recommended):

- `scipy`  
  - used for `scipy.spatial.cKDTree` (fast nearest-neighbour)
  - and `scipy.interpolate.RBFInterpolator` (RBF interpolation)
- `pyvista` (and optionally `pyvistaqt` / a plotting backend)  
  for interactive curtain / map slices and VTK export.

If SciPy is not available:

- RBF-based methods (`interp="rbf"`) will raise an `ImportError`.
- Nearest-neighbour methods automatically fall back to a **pure NumPy**
  (but slower) implementation.


## 3. Resistivity block helpers (air/ocean handling)

### 3.1 Load a block vector

```python
from femtic_viz import load_resistivity_blocks

rho = load_resistivity_blocks("resistivity_block_iter0.dat")
# rho.shape == (n_blocks,)
# rho[0] = air, rho[1] = ocean
```

The function expects a simple text file with one number per line (or
whitespace-separated values) that `numpy.loadtxt` can parse into a 1D
array. A `ValueError` is raised if the loaded array is not 1D.


### 3.2 Prepare resistivities for plotting

```python
from femtic_viz import prepare_rho_for_plotting

rho_plot = prepare_rho_for_plotting(
    rho,
    ocean_value=1.0e-10,  # None → keep original rho[1]
    mask_air=True,        # False → keep original rho[0]
)
```

Behaviour:

- if `mask_air=True` and `rho.size >= 1`:
  - `rho_plot[0] = NaN` (air → transparent/white in plots)
- if `ocean_value is not None` and `rho.size >= 2`:
  - `rho_plot[1] = float(ocean_value)` (enforce special ocean value)

The input `rho` is **not** modified in-place; a copy is returned.


### 3.3 Map block values to per-cell values

```python
from femtic_viz import map_blocks_to_cells

# block_indices: e.g. per-element block indices from FEMTIC mesh
values_on_cells = map_blocks_to_cells(rho_plot, block_indices)
```

This is a simple helper that applies `values = block_values[block_indices]`
(with basic sanity handling via `np.asarray`).


## 4. Borehole visualization (Matplotlib, 1D)

The borehole utilities are designed for vertical profiles of resistivity
(or any scalar) vs. depth.

### 4.1 Single vertical profile

```python
import numpy as np
from femtic_viz import plot_vertical_profile

z = np.linspace(0, 3000, 151)      # depth or elevation samples
rho_profile = 10 ** np.random.randn(z.size)

fig, ax = plot_vertical_profile(
    z,
    rho_profile,
    label="BH-01",
    logx=True,             # log-scale in resistivity (x-axis)
    z_positive_down=True,  # typical geophysical convention
)
fig.savefig("borehole_BH01.png", dpi=200)
```

- If `z_positive_down=True`, the function **inverts the y-axis** if
  needed so that increasing `z` corresponds to increasing depth.
- If multiple profiles share the same `z`, you can use multiple calls or
  `plot_vertical_profiles` below.


### 4.2 Multiple vertical profiles on one axis

```python
import numpy as np
from femtic_viz import plot_vertical_profiles

z = np.linspace(0, 3000, 151)
profiles = [
    10 ** np.random.randn(z.size),
    10 ** np.random.randn(z.size),
]
labels = ["BH-01", "BH-02"]

fig, ax = plot_vertical_profiles(
    z,
    profiles,
    labels=labels,
    logx=True,
    z_positive_down=True,
)
fig.savefig("boreholes_combined.png", dpi=200)
```

Both functions are pure Matplotlib wrappers; they return `(fig, ax)` so
you can further customize axes (titles, annotations, etc.).


## 5. Vertical curtain slices (NPZ → Matplotlib)

The vertical slice routines work directly on the **NPZ element file**
created by `femtic_mesh_to_npz.py` and follow these steps:

1. Construct a polyline in the `(x, y)` plane.
2. Sample this polyline at `ns` equally spaced arclength positions → `S`.
3. For each sampled `(x, y)` and each depth `z`, interpolate
   `log10_resistivity` from cell centroids in 3D.
4. Convert to `ρ` if required and plot as an image (`S` vs. `z`).


### 5.1 Building the curtain (low-level API)

The actual interpolation is handled by:

- `sample_polyline(points_xy, ns)`
- `curtain_slice_idw(...)`
- `curtain_slice_nearest(...)`
- `curtain_slice_rbf(...)`
- `curtain_slice(...)` (dispatcher: `"idw"`, `"nearest"`, `"rbf"`)

For typical use you do not need to call these directly, but they are
available for more specialized workflows (e.g. embedding into external
plotting frameworks).


### 5.2 High-level helper: `femtic_slice_from_npz_matplotlib`

```python
import numpy as np
from femtic_viz import femtic_slice_from_npz_matplotlib

# Option 1: polyline from code
polyline_xy = np.array([
    [  0.0,   0.0],
    [500.0, 100.0],
    [800.0, 400.0],
])

femtic_slice_from_npz_matplotlib(
    npz_path="femtic_model.npz",
    polyline_xy=polyline_xy,
    polyline_csv=None,     # or provide CSV instead
    zmin=0.0,
    zmax=3000.0,
    nz=201,
    ns=301,
    power=2.0,
    interp="idw",          # "idw", "nearest", or "rbf"
    logscale=True,         # plot log10(ρ) by default
    z_positive_down=True,
    cmap="viridis",
    vmin=None,
    vmax=None,
    out_npz="curtain.npz", # optional; stores S, Z, V_log10, XY
    out_csv="curtain.csv", # optional; long table with (s, z, log10_rho)
    out_png="curtain.png", # optional; uses Matplotlib figure
    title="Curtain slice",
)
```

Alternatively, you can provide `polyline_csv="polyline.csv"` with simple
`x,y` lines; if present, it overrides `polyline_xy`.

Cells with `flag == 1` in the NPZ are automatically excluded from
interpolation (masked out).


### 5.3 Curtain plotting function

If you already have sampled values (e.g. from your own interpolation),
you can call the plotting function directly:

```python
from femtic_viz import plot_curtain_matplotlib

fig, ax = plot_curtain_matplotlib(
    S,              # 1D arclength coordinate
    Z,              # 1D depths
    V,              # 2D resistivity (ρ) array, shape (nz, ns)
    logscale=True,
    z_positive_down=True,
    cmap="viridis",
    vmin=None,
    vmax=None,
    title="Curtain slice",
)
```


## 6. Horizontal map slices (NPZ → Matplotlib)

Horizontal slices are constructed by:

1. Selecting all centroids within a depth window around `z0`:
   `z0 - dz/2 ≤ z ≤ z0 + dz/2`.
2. Interpolating `log10_resistivity` onto a regular `(x, y)` grid using
   IDW / nearest / RBF.
3. Converting to `ρ` as needed and plotting a colored map.


### 6.1 High-level helper: `femtic_map_slice_from_npz_matplotlib`

```python
from femtic_viz import femtic_map_slice_from_npz_matplotlib

femtic_map_slice_from_npz_matplotlib(
    npz_path="femtic_model.npz",
    z0=1500.0,      # target depth [m]
    dz=200.0,       # vertical window thickness
    nx=200,
    ny=200,
    xmin=None,      # None → auto from centroids
    xmax=None,
    ymin=None,
    ymax=None,
    power=2.0,
    interp="idw",   # "idw", "nearest", or "rbf"
    logscale=True,
    cmap="viridis",
    vmin=None,
    vmax=None,
    out_npz="map_slice.npz",  # optional
    out_csv="map_slice.csv",  # optional
    out_png="map_slice.png",  # optional
    title="Horizontal slice",
)
```

Internally, this calls:

- `_select_depth_window(...)`
- `_grid_2d(...)` with the chosen interpolation method
- `plot_map_slice_matplotlib(...)`

You can also call `plot_map_slice_matplotlib(X, Y, V, ...)` directly if
you have precomputed grids.


## 7. PyVista grids (curtains & maps → VTK)

For integration into 3D visualization pipelines or Paraview, `femtic_viz`
provides **PyVista StructuredGrid builders** that mirror the Matplotlib
slice utilities.


### 7.1 Vertical curtain grid

```python
import numpy as np
from femtic_viz import build_curtain_grid_from_npz

polyline_xy = np.array([
    [  0.0,   0.0],
    [500.0, 100.0],
    [800.0, 400.0],
])

grid = build_curtain_grid_from_npz(
    npz_path="femtic_model.npz",
    polyline_xy=polyline_xy,  # or polyline_csv="polyline.csv"
    polyline_csv=None,
    zmin=0.0,
    zmax=3000.0,
    nz=201,
    ns=301,
    interp="idw",
    power=2.0,
)

# cell data:
#   "log10_rho"  — log10(ρ [Ω·m])
#   "rho"        — ρ [Ω·m]

grid.save("curtain.vts")  # Paraview-readable

# quick look
grid.plot(scalars="log10_rho", cmap="viridis")
```

The grid lives in a 2D `(S, z)` plane with `y = 0`. The original polyline
coordinates `(x, y)` in the model space are stored as `grid.field_data["polyline_xy"]`.


### 7.2 Horizontal map grid

```python
from femtic_viz import build_map_grid_from_npz

grid = build_map_grid_from_npz(
    npz_path="femtic_model.npz",
    z0=1500.0,
    dz=200.0,
    nx=200,
    ny=200,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    interp="idw",
    power=2.0,
)

grid.save("map_slice_z1500.vts")
grid.plot(scalars="log10_rho", cmap="viridis")
```

Here the grid is a 2D `(x, y)` plane at constant depth `z0`. The
log-resistivity and resistivity fields are attached as cell data, as
above.


## 8. Notes and recommended workflow

A typical workflow in the FEMTIC context might be:

1. **Generate NPZ** from FEMTIC mesh and resistivities (using
   `femtic_mesh_to_npz.py` or an equivalent tool).
2. Use `femtic_viz` for visualization:

   - 1D **borehole-style profiles** from any vertical samples.
   - 2D **vertical curtains** following arbitrary polylines.
   - 2D **horizontal maps** at chosen depths.
   - Optional **VTK/PyVista exports** (`.vts`/`.vtk`) for Paraview or
     more complex 3D scenes.

3. Use the **air/ocean helpers** for consistent treatment of boundary
   regions in all plots (mask air, enforce special ocean conductivities).

The module is designed to be conservative and explicit: most high-level
functions simply wrap core numerical building blocks, so that you can
easily extract pieces and re-use them in your own scripts or Jupyter
notebooks without changing your overall FEMTIC workflow.
