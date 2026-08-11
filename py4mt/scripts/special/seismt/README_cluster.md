# `saba_cluster.py` — Fuzzy c-means / SOM Clustering

Clusters MT resistivity/conductivity + seismic tomography properties
(Vp, Vs, Vp/Vs, density) that have already been interpolated onto a
common grid by `saba_interpolate.py`, via fuzzy c-means or a
self-organizing map (SOM), and plots depth-slice maps using the same
basemap engine and styling conventions as the MT and seismic plot
pipelines (`README_mt.md`, `README_seis.md`).

This is the clustering half of the pipeline; see `README_interpolate.md`
for `saba_interpolate.py`, which produces this script's input.

Authors: Svetlana Byrdina (SMB) & Volker Rath (DIAS)
AI-assisted development: Claude (Anthropic)
License: GNU General Public License v3 (GPL-3.0-or-later)
AI-generated code — review before use in production.

---

## Pipeline

```
saba_precompute.py  →  saba_interpolate.py  →  saba_interp_<method>.nc  →  saba_cluster.py  →  saba_clusters.nc + figures
```

`saba_cluster.py` reads `INTERP_FILE` — no grid-building or
interpolation of its own — and is therefore completely agnostic to
which interpolation method (`rbf`/`kriging`/`idw`/`nearest`) or target-
grid choice (`joint`/`seismic`) produced it. Point `INTERP_FILE` at
whichever `saba_interpolate.py` run you want to cluster. You can also
interpolate a superset of variables once and cluster on different
`CLUSTER_VARS` subsets without re-running `saba_interpolate.py`.

`plotpy.py` must sit alongside `saba_cluster.py` — the shared plotting
helper module also used by the MT and seismic plot scripts.

---

## What this script does

1. Reads `INTERP_FILE` (`saba_interpolate.py`'s output) — grid
   coords/dims, per-variable interpolated fields and their units, and
   enough attributes (`target_grid_mode`, `interp_vars`) to know how to
   plot it.
2. Picks `CLUSTER_VARS`, a subset of whatever variables are actually in
   `INTERP_FILE` (`None` = every one of them) — so you can interpolate a
   superset once and cluster on different subsets without re-running
   `saba_interpolate.py`.
3. Builds one feature table, drops any grid cell with a NaN in a
   selected variable, optionally standardizes (`STANDARDIZE`, z-score)
   each feature, then weights (`CLUSTER_WEIGHTS`).
4. Clusters via `CLUSTERING_METHOD`:
   - **`"fcm"`** — fuzzy c-means (Bezdek, 1981), self-contained NumPy
     implementation, `N_CLUSTERS` discrete classes. Reports the fuzzy
     partition coefficient (FPC) as a quick quality check.
   - **`"som"`** — self-organizing map (Kohonen, 1982), self-contained
     NumPy implementation, `SOM_ROWS x SOM_COLS` neurons; every point is
     labelled with its best-matching unit (BMU) over the **full**
     neuron grid (`SOM_ROWS*SOM_COLS` classes, not collapsed to
     `N_CLUSTERS`), colored with a topological colormap
     (`som_grid_colormap()`) so visually similar map colors reflect
     genuinely similar feature-space neighbors. Reports mean
     quantization error and topographic error.
5. Reconstructs the hard label + membership/quantization-error back onto
   the grid and saves `saba_clusters.nc` / `saba_cluster_centers.csv`.
6. Plots horizontal cluster maps at `PLOT_DEPTHS_KM`. The cluster
   overlay switches on `target_grid_mode` (read from `INTERP_FILE`'s
   attributes): `imshow(extent=...)` for a regular `"joint"` grid,
   `pcolormesh(shading="nearest")` against the reused `"seismic"`
   grid's own 2-D `utm_easting_km`/`utm_northing_km` coordinates
   otherwise.

Writes:

| Output file                    | Contents                                   |
|----------------------------------|---------------------------------------------|
| `saba_clusters.nc`               | Hard label + membership/quantization-error on the grid read from `INTERP_FILE` |
| `saba_cluster_centers.csv`       | Cluster/class centers in raw (physical) units, point counts, fractions, and the `weight` row used |
| `clusters_{depth}km_saba.{ext}`  | Plain cluster map, one per `PLOT_DEPTHS_KM` entry, one file per `PLOT_FORMATS` entry |
| `clusters_{depth}km_saba_annotated.{ext}` | The same map, additionally annotated with seismicity/MT-sites/volcanoes/cities — produced in parallel, not instead of, the plain map; toggle with `SHOW_SPECIFIC_PLOT` |

Both `saba_clusters.nc` and `saba_cluster_centers.csv` are written into
`NC_DIR`, alongside `INTERP_FILE`.

---

## Per-variable weighting

`CLUSTER_WEIGHTS` lets individual variables count more or less toward
cluster/BMU assignment than a plain (unweighted) Euclidean distance
would give them. Each standardized feature is scaled by
`sqrt(weight)` before clustering — equivalent to the weighted Euclidean
distance `d² = Σⱼ weightⱼ · (xⱼ − cⱼ)²` — and the resulting centers are
divided back by the same `sqrt(weight)` afterward, before undoing
standardization, so `saba_cluster_centers.csv` and `saba_clusters.nc`
always report centers in true physical units. Every variable not listed
in `CLUSTER_WEIGHTS` defaults to `1.0` (no effect).

## Specific (annotated) cluster maps

A second map per `PLOT_DEPTHS_KM` entry, in addition to (never instead
of) the plain cluster map. Reuses `_draw_cluster_overlay()` on the same
basemap, then layers `saba_plot_seis.py`-style feature markers/labels
via `draw_specific_features()` — same CSVs, on/off switches, and
marker/label style dicts as `saba_plot_seis.py`.

---

## Coordinate convention

**UTM Zone 19S (EPSG:32719)**, distances in km, depth in km positive
down. A `"joint"`-mode `saba_clusters.nc` has a genuine,
uniformly-spaced regular `(depth, northing, easting)` grid; a
`"seismic"`-mode one inherits whatever native `(depth, row, col)`
resolution the reference seismic-tomography variable has, with 2-D
`utm_easting_km`/`utm_northing_km` aux coordinates rather than 1-D
regular axes — see `README_interpolate.md`.

## Dependencies

```
numpy, xarray, pandas, matplotlib
```
plus the local `plotpy.py` helper module. No `scikit-fuzzy`/`MiniSom`
dependency — both clustering implementations are self-contained.

## Typical run

```bash
python3 saba_precompute.py      # must be run first (or already have been)
python3 saba_interpolate.py     # see README_interpolate.md
python3 saba_cluster.py         # reads saba_interp_<method>.nc, clusters
                                 # (fcm/som), writes saba_clusters.nc /
                                 # saba_cluster_centers.csv and the maps
```

Re-run `saba_cluster.py` (no need to re-run `saba_interpolate.py`)
whenever `CLUSTER_VARS`, `CLUSTER_WEIGHTS`, `STANDARDIZE`,
`CLUSTERING_METHOD` or its `N_CLUSTERS`/`FUZZINESS`/`SOM_*` settings, or
any plotting setting change. Re-run `saba_interpolate.py` first (see
`README_interpolate.md`) whenever the interpolated grid itself needs to
change.
