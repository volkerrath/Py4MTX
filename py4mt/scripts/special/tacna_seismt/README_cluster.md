# Tacna Fuzzy Clustering / SOM Pipeline

Interpolation of MT resistivity (or conductivity) + seismic tomography
properties (Vp, Vs, Vp/Vs, density) onto one common grid, followed by
fuzzy c-means or self-organizing-map (SOM) clustering, with cluster maps
using the same basemap engine and styling conventions as the MT and
seismic plot pipelines (`README_mt.md`, `README_seis.md`).

Authors: Svetlana Byrdina (SMB) & Volker Rath (DIAS)
AI-assisted development: Claude (Anthropic)
License: GNU General Public License v3 (GPL-3.0-or-later)
AI-generated code — review before use in production.

---

## Pipeline (reorganized 2026-08-02)

```
tacna_precompute.py  →  tacna_interpolate.py  →  tacna_interp_<method>.nc  →  tacna_cluster.py  →  tacna_clusters.nc + figures
```

This replaces the earlier `tacna_cluster_rbf.py` / `tacna_cluster_kriging.py`
/ `tacna_cluster_som.py` trio, which each duplicated grid-building +
interpolation + clustering + plotting end to end, differing only in
interpolation method (RBF vs. kriging) or clustering method (fuzzy
c-means vs. SOM). Interpolation and clustering are now two separate
scripts:

- **`tacna_interpolate.py`** loads every needed variable from its own
  native grid, builds (or reuses) a target grid, and interpolates onto
  it via RBF, ordinary kriging, or inverse-distance weighting (IDW) —
  `INTERP_METHOD`. Writes `tacna_interp_<method>.nc`.
- **`tacna_cluster.py`** reads that file — no grid-building or
  interpolation of its own — and clusters via fuzzy c-means or a SOM —
  `CLUSTERING_METHOD` — then plots depth-slice maps.

Because the two are decoupled, `tacna_cluster.py` is completely
agnostic to which interpolation method or target-grid choice produced
its input; point `INTERP_FILE` at whichever `tacna_interpolate.py` run
you want to cluster. You can also interpolate a superset of variables
once and cluster on different `CLUSTER_VARS` subsets without re-running
`tacna_interpolate.py`.

`plotpy.py` must sit alongside `tacna_cluster.py` — the shared plotting
helper module also used by the MT and seismic plot scripts.

---

## 1. `tacna_interpolate.py`

Reads the variables in `INTERP_VARS`, each from its own native grid:

- MT resistivity/conductivity/sensitivity come from Part A's
  `modem_submesh_points.nc` — the full native ModEM mesh, already
  flattened to a point table (`easting_km`, `northing_km`, `depth_km`,
  value) by `tacna_precompute.py` (see `README_mt.md`).
- Vp/Vs/Vp-Vs-ratio/density come from Part B's `tacna_vp.nc` /
  `tacna_vs.nc` / `tacna_vps.nc` / `tacna_dens.nc` — gridded
  `(depth, row, col)` cubes with 2-D `utm_easting`/`utm_northing` aux
  coords (see `README_seis.md`).

Every selected variable is flattened into a point cloud
(`load_variable_points()`), then interpolated onto ONE common **target
grid** (`TARGET_GRID`):

- **`"joint"`** (default) — a freshly-built, genuinely regular UTM-km
  grid (`GRID_EASTING_KM`/`GRID_NORTHING_KM`/`GRID_DEPTH_KM`), auto-
  bounded to the tightest common overlap of every selected variable's
  own extent unless you set explicit bounds.
- **`"seismic"`** — skip building a new grid; reuse one seismic
  tomography variable's own native `(depth, row, col)` grid as-is
  (`SEISMIC_MESH_VAR` picks which one, e.g. `"vps"`). Avoids a second
  resampling step for that variable and keeps everything on that
  model's own native resolution. "row"/"col" are read from the source
  file's own dimension names — not assumed to be lat/lon — so this
  works regardless of what `tacna_precompute.py`'s Part B calls them.
  The output is **not** a regular grid in UTM space, so `tacna_cluster.py`
  plots it with `pcolormesh(shading="nearest")` against the 2-D
  `utm_easting_km`/`utm_northing_km` coordinates carried over into
  `tacna_interp_<method>.nc`, rather than `imshow`'s `extent=`.

Interpolation method (`INTERP_METHOD`):

| Method     | Function                        | Extra dependency | Notes |
|------------|----------------------------------|-------------------|-------|
| `"rbf"`    | `scipy.interpolate.RBFInterpolator` | — | `RBF_*` settings; `RBF_NEIGHBORS` keeps it local/fast |
| `"kriging"`| `pykrige.ok3d.OrdinaryKriging3D`     | `pykrige` | `KRIGING_*` settings; `KRIGING_MAX_POINTS` subsamples first (variogram fit is O(n²)) |
| `"idw"`    | inverse-distance weighting (`scipy.spatial.cKDTree`) | — | `IDW_POWER`/`IDW_NEIGHBORS`; exact at source points, purely local |

All three extrapolate past a variable's own data footprint with no
natural cutoff. `MASK_TO_CONVEX_HULL` (default `True`) nulls out
target-grid points outside each variable's own 3-D convex hull
(`outside_convex_hull()`, `scipy.spatial.Delaunay`). `APPLY_ROI_MASK`/
`ROI_VERTICES_KM` (+ optional `ROI_DEPTH_MIN_KM`/`ROI_DEPTH_MAX_KM`)
additionally restrict every variable to one shared rectangular region,
applied identically after interpolation — the same mechanism as the
earlier per-cluster-script version, just applied once here instead of
per clustering run.

Output: `tacna_interp_<INTERP_METHOD>.nc` in `NC_DIR` (override with
`OUTPUT_FILE`) — grid coords/dims (varying by `TARGET_GRID`), one
data variable per `INTERP_VARS` entry (with `units`/`long_name`
attrs), and attributes recording `target_grid_mode`, `dim_names`,
`interp_method`, `interp_vars`, and the method-specific settings used —
everything `tacna_cluster.py` needs to read it back correctly.

### Resistivity vs. conductivity

`resistivity_to_conductivity()` inverts the resistivity values loaded
from `modem_submesh_points.nc`, reading *that variable's own* `units`
attribute (`"log10(Ohm.m)"`, `"ln(Ohm.m)"`, or `"Ohm.m"`) to apply the
matching inversion, so it keeps working regardless of
`tacna_precompute.py`'s `OUTPUT_TRANSFORM`. `USE_CONDUCTIVITY` swaps any
`"rho"` in `INTERP_VARS` for `"cond"` at load time.

---

## 2. `tacna_cluster.py`

Reads `INTERP_FILE` (a bare filename under `NC_DIR`, e.g.
`"tacna_interp_rbf.nc"`), picks `CLUSTER_VARS` (`None` = every variable
in that file), builds one feature table, drops any grid cell with a NaN
in a selected variable, optionally standardizes (`STANDARDIZE`, z-score)
each feature, then weights (`CLUSTER_WEIGHTS`, `sqrt(weight)` per
feature — see "Per-variable weighting" below), and clusters via
`CLUSTERING_METHOD`:

- **`"fcm"`** — fuzzy c-means (Bezdek, 1981), self-contained NumPy
  implementation, `N_CLUSTERS` discrete classes. Reports the fuzzy
  partition coefficient (FPC) as a quick quality check.
- **`"som"`** — self-organizing map (Kohonen, 1982), self-contained
  NumPy implementation, `SOM_ROWS x SOM_COLS` neurons; every point is
  labelled with its best-matching unit (BMU) over the **full** neuron
  grid (`SOM_ROWS*SOM_COLS` classes, not collapsed to `N_CLUSTERS`),
  colored with a topological colormap (`som_grid_colormap()`) so
  visually similar map colors reflect genuinely similar feature-space
  neighbors. Reports mean quantization error and topographic error.

Writes:

| Output file                    | Contents                                   |
|----------------------------------|---------------------------------------------|
| `tacna_clusters.nc`              | Hard label + membership/quantization-error on the grid read from `INTERP_FILE` |
| `tacna_cluster_centers.csv`      | Cluster/class centers in raw (physical) units, point counts, fractions, and the `weight` row used |
| `clusters_{depth}km_tacna.{ext}` | Plain cluster map, one per `PLOT_DEPTHS_KM` entry, one file per `PLOT_FORMATS` entry |
| `clusters_{depth}km_tacna_annotated.{ext}` | The same map, additionally annotated with seismicity/MT-sites/volcanoes/cities — produced in parallel, not instead of, the plain map; toggle with `SHOW_SPECIFIC_PLOT` |

Both `tacna_clusters.nc` and `tacna_cluster_centers.csv` are written into
`NC_DIR`, alongside `INTERP_FILE`.

Plotting reuses the same topography/hillshade/ocean-fill basemap and
deterministic equal-scale panel layout as `tacna_plot_seis.py`
(`plotpy.build_panel_figure`). The cluster overlay itself switches on
`target_grid_mode` (read from `INTERP_FILE`'s attributes): `imshow` with
`extent=` for a regular `"joint"` grid, `pcolormesh(shading="nearest")`
against the reused `"seismic"` grid's own 2-D UTM coordinates otherwise
— see `tacna_interpolate.py`'s "Target grid" section above.

### Per-variable weighting

Unchanged from the earlier version: `CLUSTER_WEIGHTS` lets individual
variables count more or less toward cluster/BMU assignment than a plain
(unweighted) Euclidean distance would give them. Each standardized
feature is scaled by `sqrt(weight)` before clustering — equivalent to
the weighted Euclidean distance `d² = Σⱼ weightⱼ · (xⱼ − cⱼ)²` — and the
resulting centers are divided back by the same `sqrt(weight)` afterward,
before undoing standardization, so `tacna_cluster_centers.csv` and
`tacna_clusters.nc` always report centers in true physical units. Every
variable not listed in `CLUSTER_WEIGHTS` defaults to `1.0` (no effect).

### Specific (annotated) cluster maps

Unchanged: a second map per `PLOT_DEPTHS_KM` entry, in addition to
(never instead of) the plain cluster map. Reuses `_draw_cluster_overlay()`
on the same basemap, then layers `tacna_plot_seis.py`-style feature
markers/labels via `draw_specific_features()` — same CSVs, on/off
switches, and marker/label style dicts as `tacna_plot_seis.py`.

---

## Coordinate convention

Same as the MT and seismic pipelines: **UTM Zone 19S (EPSG:32719)**,
distances in km, depth in km positive down. A `"joint"`-mode
`tacna_interp_<method>.nc` / `tacna_clusters.nc` has a genuine, uniformly
-spaced regular `(depth, northing, easting)` grid; a `"seismic"`-mode one
inherits whatever native `(depth, row, col)` resolution the reference
seismic-tomography variable has, with 2-D `utm_easting_km`/
`utm_northing_km` aux coordinates rather than 1-D regular axes.

## Dependencies

```
numpy, xarray, pandas, matplotlib, pyproj, scipy
```
(`scipy.interpolate.RBFInterpolator`/`scipy.spatial.cKDTree` for
RBF/IDW, `scipy.spatial.Delaunay` for convex-hull masking; `pandas` for
the feature CSVs used by the specific/annotated cluster maps) plus the
local `plotpy.py` helper module. `pykrige` is only needed if
`tacna_interpolate.py`'s `INTERP_METHOD = "kriging"` (imported lazily,
not a hard dependency otherwise). No `scikit-fuzzy`/`MiniSom`
dependency — both clustering implementations are self-contained.

## Typical run

```bash
python3 tacna_precompute.py     # must be run first (or already have been)
python3 tacna_interpolate.py    # loads native point clouds, builds/reuses the
                                 # target grid, interpolates (RBF/kriging/IDW),
                                 # writes tacna_interp_<method>.nc
python3 tacna_cluster.py        # reads tacna_interp_<method>.nc, clusters
                                 # (fcm/som), writes tacna_clusters.nc /
                                 # tacna_cluster_centers.csv and the maps
```

Re-run `tacna_interpolate.py` whenever `INTERP_VARS`, `USE_CONDUCTIVITY`,
`TARGET_GRID`, the `GRID_*_KM`/`SEISMIC_MESH_VAR` settings,
`INTERP_METHOD` or its `RBF_*`/`KRIGING_*`/`IDW_*` settings,
`MASK_TO_CONVEX_HULL`, `APPLY_ROI_MASK`/`ROI_*`, or `tacna_precompute.py`'s
own output change. Re-run `tacna_cluster.py` (only — no need to re-run
`tacna_interpolate.py`) whenever `CLUSTER_VARS`, `CLUSTER_WEIGHTS`,
`STANDARDIZE`, `CLUSTERING_METHOD` or its `N_CLUSTERS`/`FUZZINESS`/
`SOM_*` settings, or any plotting setting change.
