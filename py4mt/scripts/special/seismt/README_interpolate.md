# `saba_interpolate.py` — Interpolation onto a Common Grid

Loads MT resistivity/conductivity/sensitivity + seismic tomography
properties (Vp, Vs, Vp/Vs, density), each from its own native grid
produced by `saba_precompute.py`, and interpolates every requested one
onto one common target grid via RBF, ordinary kriging, inverse-distance
weighting (IDW), or nearest-neighbor.

This is the interpolation half of the pipeline; see `README_cluster.md`
for `saba_cluster.py`, which reads this script's output and clusters it.

Authors: Svetlana Byrdina (SMB) & Volker Rath (DIAS)
AI-assisted development: Claude (Anthropic)
License: GNU General Public License v3 (GPL-3.0-or-later)
AI-generated code — review before use in production.

---

## Pipeline

```
saba_precompute.py  →  saba_interpolate.py  →  saba_interp_<method>.nc  →  saba_cluster.py  →  saba_clusters.nc + figures
```

`saba_interpolate.py` does not build or interpolate onto a specific
target unconditionally — it reads every variable in `INTERP_VARS`, each
from its own native grid, and interpolates each one independently onto
whichever target grid `TARGET_GRID` selects. It has no knowledge of
clustering; `saba_cluster.py` reads its output (`INTERP_FILE`) but
otherwise builds no grid and interpolates nothing itself.

---

## Native sources (`VARIABLE_SOURCES`)

Two kinds of native source, each with its own loader
(`load_variable_points()` dispatches):

- **`"modem_points"`** — a flat point table (`saba_precompute.py`'s
  `modem_submesh_points.nc`, Part A): already `(easting_km, northing_km,
  depth_km, value)` rows at full native ModEM resolution. Used for MT
  resistivity/conductivity/sensitivity. `saba_precompute.py` already
  masks both air cells (`AIR_RHO_THRESHOLD`, resistivity >= 1e10 Ω·m)
  and near-zero sentinel/placeholder values some `.rho`/`.sns` files use
  for invalid cells (`ILLEGAL_LOW_THRESHOLD`, e.g. a 1e-32 no-data flag)
  to NaN before this table is written, so `saba_interpolate.py` never
  sees either kind of illegal value as if it were real data.
- **`"seis_grid"`** — a gridded `(depth, row, col)` cube
  (`saba_precompute.py` Part B's `saba_vp.nc` / `saba_vs.nc` /
  `saba_vps.nc` / `saba_dens.nc`), with 2-D `utm_easting`/`utm_northing`
  aux coords — flattened, with the horizontal coords broadcast across
  every depth level, into a point cloud by `load_seis_grid_points()`.
  "row"/"col" are read from the source file's own dimension names —
  never assumed to be lat/lon — so this works regardless of what
  `saba_precompute.py`'s Part B actually calls them.

`INTERP_VARS` picks which of these get loaded, interpolated, and
written into `INTERP_FILE`. It's worth interpolating a superset once:
`saba_cluster.py`'s own `CLUSTER_VARS` can pick any subset of what's
available in `INTERP_FILE` for a given clustering run, without
re-running this script, as long as it doesn't need a variable that
isn't in `INTERP_VARS`.

### Resistivity vs. conductivity

`resistivity_to_conductivity()` inverts the resistivity values loaded
from `modem_submesh_points.nc`, reading *that variable's own* `units`
attribute (`"log10(Ohm.m)"`, `"ln(Ohm.m)"`, or `"Ohm.m"`) to apply the
matching inversion, so it keeps working regardless of
`saba_precompute.py`'s `OUTPUT_TRANSFORM`. `USE_CONDUCTIVITY` swaps any
`"rho"` in `INTERP_VARS` for `"cond"` at load time.

---

## Target grid (`TARGET_GRID`)

- **`"joint"`** (default) — a freshly-built, genuinely regular UTM-km
  grid (`GRID_EASTING_KM`/`GRID_NORTHING_KM`/`GRID_DEPTH_KM`), auto-
  bounded to the tightest common overlap of every selected variable's
  own extent unless you set explicit bounds. Output dims:
  `(depth, northing, easting)`.
- **`"seismic"`** — skip building a new grid; reuse one seismic
  tomography variable's own native `(depth, row, col)` grid as-is
  (`SEISMIC_MESH_VAR` picks which one, e.g. `"vps"`). Avoids a second
  resampling step for that variable and keeps everything on that
  model's own native resolution. Output dims: `(depth, row, col)`, with
  2-D `utm_easting_km`/`utm_northing_km` aux coords carried over from
  the reference source — **not** a regular grid in UTM space, so
  `saba_cluster.py`'s plotting handles this with
  `pcolormesh(shading="nearest")` against those 2-D coords, rather than
  `imshow`'s regular-grid `extent=`.

---

## Interpolation method (`INTERP_METHOD`)

| Method       | Function                                              | Extra dependency | Notes |
|--------------|--------------------------------------------------------|-------------------|-------|
| `"rbf"`      | `scipy.interpolate.RBFInterpolator`                     | — | `RBF_*` settings; `RBF_NEIGHBORS` keeps it local/fast |
| `"kriging"`  | `pykrige.ok3d.OrdinaryKriging3D`                        | `pykrige` | `KRIGING_*` settings; `KRIGING_MAX_POINTS` subsamples first (variogram fit is O(n²)) |
| `"idw"`      | inverse-distance weighting (`scipy.spatial.cKDTree`)    | — | `IDW_POWER`/`IDW_NEIGHBORS`; exact at source points, purely local |
| `"nearest"`  | nearest-neighbor (`scipy.spatial.cKDTree`, k=1)          | — | No fitting/shape parameter at all; piecewise-constant (Voronoi-cell) output, cheapest of the four — useful as a quick sanity baseline or deliberately blocky classification-style map, not a smooth field |

All four extrapolate/interpolate past a variable's own data footprint
with no natural cutoff. `MASK_TO_CONVEX_HULL` (default `True`) nulls out
target-grid points outside each variable's own 3-D convex hull
(`outside_convex_hull()`, `scipy.spatial.Delaunay`). `APPLY_ROI_MASK`/
`ROI_VERTICES_KM` (+ optional `ROI_DEPTH_MIN_KM`/`ROI_DEPTH_MAX_KM`)
additionally restrict every variable to one shared rectangular region,
applied identically after interpolation.

Output: `saba_interp_<INTERP_METHOD>.nc` in `NC_DIR` (override with
`OUTPUT_FILE`) — grid coords/dims (varying by `TARGET_GRID`), one data
variable per `INTERP_VARS` entry (with `units`/`long_name` attrs), and
attributes recording `target_grid_mode`, `dim_names`, `interp_method`,
`interp_vars`, and the method-specific settings used — everything
`saba_cluster.py` needs to read it back correctly.

---

## Gradient (optional, `COMPUTE_GRADIENT`)

If enabled, computes the spatial gradient of each `GRADIENT_VARS` entry
on the same grid the field was just interpolated onto, and stores it in
`saba_interp_<method>.nc` alongside it as four extra data variables per
variable: `{key}_grad_easting`, `{key}_grad_northing`, `{key}_grad_depth`
(value-units per km) and `{key}_grad_mag` (their combined 3-D
magnitude).

`GRADIENT_METHOD`:

- **`"spline"`** (default) — fits a `scipy.interpolate.UnivariateSpline`
  along each grid axis, through the already-interpolated values, and
  differentiates it analytically. Smoother than a raw finite-difference
  gradient — a natural fit here since the field being differentiated is
  itself already the output of an interpolation step, not raw noisy
  data. `GRADIENT_SPLINE_ORDER`/`GRADIENT_SPLINE_SMOOTHING` control the
  fit (`s=0`, the default, passes exactly through every finite point on
  that line). Lines with fewer than `GRADIENT_MIN_POINTS` finite values
  come back all-NaN.
- **`"finite_difference"`** — plain `numpy.gradient`; cheaper, no
  fitting, but less smooth.

For `TARGET_GRID == "joint"` this is a plain per-axis partial
derivative, since the grid is genuinely regular in UTM
easting/northing/depth. For `TARGET_GRID == "seismic"` the grid is
regular in `(row, col)` index space but **not** in UTM space, so a raw
row/col-axis derivative would not be a physical easting/northing
gradient: those row/col-space derivatives are instead converted via the
local 2×2 Jacobian of the `(row, col) -> (easting, northing)` mapping
(itself differentiated the same way, from the grid's own
`utm_easting_km`/`utm_northing_km`), so the reported gradient is a true
spatial derivative either way. Verified against synthetic linear test
fields on both a regular and a curvilinear grid (exact recovery, edge
effects only at the grid boundary).

Gradients are **not** added to `interp_vars`, so `saba_cluster.py`'s
default (`CLUSTER_VARS = None`, "cluster on everything") never picks
them up automatically — add `"{key}_grad_mag"` etc. to `CLUSTER_VARS`
explicitly there if you want to cluster on gradients too.

---

## Coordinate convention

**UTM Zone 19S (EPSG:32719)**, distances in km, depth in km positive
down. A `"joint"`-mode `saba_interp_<method>.nc` has a genuine,
uniformly-spaced regular `(depth, northing, easting)` grid; a
`"seismic"`-mode one inherits whatever native `(depth, row, col)`
resolution the reference seismic-tomography variable has, with 2-D
`utm_easting_km`/`utm_northing_km` aux coordinates rather than 1-D
regular axes.

## Dependencies

```
numpy, xarray, scipy
```
(`scipy.interpolate.RBFInterpolator`/`scipy.spatial.cKDTree` for
RBF/IDW/nearest, `scipy.spatial.Delaunay` for convex-hull masking,
`scipy.interpolate.UnivariateSpline` for gradients). `pykrige` is only
needed if `INTERP_METHOD = "kriging"` (imported lazily, not a hard
dependency otherwise).

## Typical run

```bash
python3 saba_precompute.py      # must be run first (or already have been)
python3 saba_interpolate.py     # loads native point clouds, builds/reuses the
                                 # target grid, interpolates (RBF/kriging/IDW/
                                 # nearest), optionally computes gradients,
                                 # writes saba_interp_<method>.nc
```

Re-run whenever `INTERP_VARS`, `USE_CONDUCTIVITY`, `TARGET_GRID`, the
`GRID_*_KM`/`SEISMIC_MESH_VAR` settings, `INTERP_METHOD` or its
`RBF_*`/`KRIGING_*`/`IDW_*` settings, `MASK_TO_CONVEX_HULL`,
`APPLY_ROI_MASK`/`ROI_*`, `COMPUTE_GRADIENT`/`GRADIENT_*`, or
`saba_precompute.py`'s own output change.
