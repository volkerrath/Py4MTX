# Tacna Fuzzy Clustering Pipeline

Fuzzy c-means clustering of MT resistivity (or conductivity) + seismic
tomography properties (Vp, Vs, Vp/Vs, density), each loaded from its own
native grid and RBF-interpolated onto one jointly-defined regular UTM
Zone 19S (EPSG:32719) grid, with cluster maps using the same basemap
engine and styling conventions as the MT and seismic plot pipelines
(`README_mt.md`, `README_seis.md`).

Authors: Svetlana Byrdina (SMB) & Volker Rath (DIAS)
AI-assisted development: Claude (Anthropic)
License: GNU General Public License v3 (GPL-3.0-or-later)
AI-generated code — review before use in production.

---

## Pipeline

```
tacna_precompute.py  →  tacna_cluster.py  →  tacna_clusters.nc + figures
```

`tacna_cluster.py` reads directly from `tacna_precompute.py`'s output
directory, but — unlike an earlier version of this pipeline — the
sources it reads are each on their **own native grid**, not a shared one:

- MT resistivity/conductivity/sensitivity come from Part A's
  `modem_submesh_points.nc` — the full native ModEM mesh, already
  flattened to a point table (`easting_km`, `northing_km`, `depth_km`,
  value) by `tacna_precompute.py` (see `README_mt.md`).
- Vp/Vs/Vp-Vs-ratio/density come from Part B's `tacna_vp.nc` /
  `tacna_vs.nc` / `tacna_vps.nc` / `tacna_dens.nc` — gridded
  `(depth, lat, lon)` cubes with 2-D `utm_easting`/`utm_northing` aux
  coords (see `README_seis.md`).

`tacna_cluster.py` flattens whichever of these it needs into point
clouds, then **RBF-interpolates each one independently onto one jointly-
defined regular grid** (`GRID_EASTING_KM`/`GRID_NORTHING_KM`/
`GRID_DEPTH_KM`) before clustering. This grid is defined here, not tied
to either source's own resolution or extent.

### Where the interpolation moved (changed from an earlier version)

Earlier versions of this pipeline resampled MT resistivity onto the
seismic tomography's own grid inside `tacna_precompute.py`
(`modem_rho_on_seisgrid*.nc`), and `tacna_cluster.py` just loaded
already-aligned files. That resampling step now lives in
`tacna_cluster.py` instead, and uses RBF interpolation
(`scipy.interpolate.RBFInterpolator`) rather than trilinear resampling
onto someone else's grid — see "RBF interpolation" below. This means
variables no longer need to already share a grid before this script can
combine them, and the target grid's resolution/extent is a free choice
rather than inherited from the seismic tomography model.

**`plotpy.py`** must sit alongside the script — the same shared plotting
helper module used by all three plot scripts (map/figure layout,
hillshading, north arrow, colorbar). `tacna_cluster.py` reuses
`plotpy.build_panel_figure` for the same deterministic, equal-scale map
layout as `tacna_plot_seis.py` / `tacna_plot_modem_image.py`.

### 1. `tacna_cluster.py`

Reads the variables in `CLUSTER_VARS`, RBF-interpolates each onto the
joint grid, clusters them with fuzzy c-means, and writes:

| Output file                    | Contents                                   |
|----------------------------------|---------------------------------------------|
| `tacna_clusters.nc`              | Hard cluster label + membership ("confidence") on the joint regular grid — dims `(depth, northing, easting)`, genuinely regular UTM-km, unlike any of the source grids |
| `tacna_cluster_centers.csv`      | Cluster centers in raw (physical) units, point counts, fractions |
| `clusters_{depth}km_tacna.{ext}` | One map per entry in `PLOT_DEPTHS_KM`, one file per `PLOT_FORMATS` entry |

Both `tacna_clusters.nc` and `tacna_cluster_centers.csv` are written into
`NC_DIR`, alongside their inputs.

**What happens, step by step:**

1. **Load each variable as a native point cloud.** `VARIABLE_SOURCES`
   registers two kinds of source:
   - `"modem_points"` — read straight from `modem_submesh_points.nc`
     (`load_modem_points()`), already a point table; just filtered to
     valid (finite) rows.
   - `"seis_grid"` — read from a Part B cube (`load_seis_grid_points()`),
     flattened into a point cloud by broadcasting the 2-D
     `utm_easting`/`utm_northing` coords across every depth level.
2. **Build the joint regular grid** (`GRID_EASTING_KM`/
   `GRID_NORTHING_KM`/`GRID_DEPTH_KM`) — explicit bounds, or auto = the
   tightest common overlap of every loaded variable's own extent, so the
   grid never reaches into a region only some variables cover.
3. **RBF-interpolate** each variable onto that grid independently
   (`rbf_interpolate_to_grid()`, `scipy.interpolate.RBFInterpolator`),
   then (if `MASK_TO_CONVEX_HULL`) null out grid points that fall outside
   that variable's own 3-D convex hull (`outside_convex_hull()`,
   `scipy.spatial.Delaunay`) — RBF otherwise extrapolates smoothly
   forever past its source points, which would let an under-constrained
   corner quietly influence the clustering.
4. **Flatten + drop NaNs.** All (now grid-aligned) variables are stacked
   into one feature table; any grid cell with a NaN in *any* selected
   variable — outside one variable's convex hull, outside the ModEM crop,
   etc. — is dropped before clustering.
5. **Standardize** (z-score, `STANDARDIZE`) — on by default, since
   resistivity/conductivity/Vp-Vs-ratio/density live on very different
   numeric scales and units, and Euclidean distance would otherwise be
   dominated by whichever variable has the largest raw range.
6. **Fuzzy c-means** — a self-contained (NumPy-only) implementation of
   the standard Bezdek (1981) algorithm; see `fuzzy_cmeans()` in the
   script. Reports the fuzzy partition coefficient (FPC, 1/`N_CLUSTERS`
   = maximally fuzzy, 1 = fully crisp) as a quick quality check, printed
   alongside a table of cluster centers (back-transformed to raw
   physical units) and sizes.
7. **Reconstruct + save.** The hard label (`argmax` membership) and its
   membership value are written back onto the joint grid (`-1`/`NaN`
   where a cell was dropped in step 4) and saved to `tacna_clusters.nc`.
8. **Plot.** One horizontal map per `PLOT_DEPTHS_KM` entry (nearest
   available depth level on the joint grid is used), with the same
   topography/hillshade/ocean-fill basemap as `tacna_plot_seis.py`, a
   discrete colorbar (one tick per cluster), and an optional north arrow
   / lon-lat tick overlay.

**Key settings:**

- `NC_DIR` (default `"../precompute/"`) — must match `OUTPUT_DIR` in
  `tacna_precompute.py`; this is where `tacna_clusters.nc` and
  `tacna_cluster_centers.csv` are written too.
- `PLOT_DIR` (default `"../plots_cluster/"`) — where cluster map figures
  are written (created automatically if it doesn't exist).
- `VARIABLE_SOURCES` — the registry of everything *available* to
  cluster on: `rho` (log₁₀ resistivity), `cond` (conductivity, derived —
  see below), `sens` (sensitivity — requires `tacna_precompute.py`'s
  `USE_SENSITIVITY = True`), `vp`, `vs`, `vps` (Vp/Vs ratio), `dens`
  (density). Add new entries here as new co-registered properties become
  available; each needs a `kind` (`"modem_points"` or `"seis_grid"`), a
  source file, a value variable name, a label, and units.
- `CLUSTER_VARS` (default `["rho", "vps", "dens"]`) — which
  `VARIABLE_SOURCES` keys actually get clustered. Deliberately starting
  simple; add `"vp"`/`"vs"`/`"sens"` (already registered, just unused)
  with a one-line change once you're ready.
- `USE_CONDUCTIVITY` (default `True`) — if `True`, any `"rho"` entry in
  `CLUSTER_VARS` is swapped for `"cond"` at load time, so you can flip
  between resistivity- and conductivity-based clustering without editing
  `CLUSTER_VARS` itself. See "Resistivity vs. conductivity" below. Don't
  put both `"rho"` and `"cond"` in `CLUSTER_VARS` directly — same
  information, just inverted, so clustering on both is redundant rather
  than genuinely adding a feature.
- `STANDARDIZE` (default `True`) — z-score each feature before
  clustering; see step 5 above.
- `GRID_EASTING_KM` / `GRID_NORTHING_KM` / `GRID_DEPTH_KM` — each a
  `dict(min=…, max=…, step=…)` defining the joint regular grid every
  variable gets RBF-interpolated onto. `min`/`max = None` (the default)
  auto-computes the tightest common overlap of every *currently selected*
  variable's own extent — so the grid adapts automatically as
  `CLUSTER_VARS` changes. Set explicit numbers instead for a fixed,
  reproducible grid regardless of which variables are selected (e.g. to
  keep grid cells identical across separate runs with different
  `CLUSTER_VARS`, so `tacna_clusters.nc` outputs stay directly
  comparable).
- `RBF_KERNEL` (default `"linear"`) — passed straight to
  `scipy.interpolate.RBFInterpolator`; also accepts `"thin_plate_spline"`,
  `"cubic"`, `"quintic"`, `"multiquadric"`, `"inverse_multiquadric"`,
  `"gaussian"`, etc. The last three additionally require `RBF_EPSILON`
  (shape parameter) — unused by the others.
- `RBF_SMOOTHING` (default `0.0`) — `0` = exact interpolation at every
  source point; `> 0` trades exactness for a smoother fitted surface
  (useful if the source data itself is noisy).
- `RBF_NEIGHBORS` (default `50`) — use only the nearest N source points
  per query point (fast, local, scales to large point clouds like the
  full `modem_submesh_points.nc` table); `None` = exact global RBF using
  every source point at once (more accurate, but can be slow/memory-heavy
  for large point clouds).
- `RBF_DEGREE` — polynomial term degree augmenting the RBF fit; `None`
  uses the kernel's own default degree.
- `MASK_TO_CONVEX_HULL` (default `True`) — null out joint-grid points
  that fall outside a variable's own 3-D convex hull, rather than keeping
  RBF's unconstrained extrapolation there. Uses `scipy.spatial.Delaunay`
  per variable — can be slow for very large point clouds; set `False` to
  skip (keeps RBF's raw extrapolation everywhere on the joint grid
  instead, which will bias the clustering in poorly-constrained corners).
- `N_CLUSTERS` (default `4`), `FUZZINESS` (`m`, default `2.0` —
  conventional FCM default), `MAX_ITER` (`300`), `TOL` (`1e-5`,
  convergence threshold on max membership change between iterations),
  `RANDOM_SEED` (`42`, for the initial membership matrix).
- `PLOT_DEPTHS_KM` (default `[1.0, 5.0, 9.0]`) — depths to render cluster
  maps at; each is snapped to the nearest available depth level on the
  joint grid.
- `CLUSTER_CMAP` (default `"tab10"`, a qualitative colormap — one colour
  per cluster, not a continuous scale) / `CLUSTER_ALPHA` (default
  `0.80`).
- `SHOW_TOPO_BASEMAP`, `HS_AZIMUTH`/`HS_ALTITUDE`/`HS_SIGMA`,
  `TOPO_VMIN`/`TOPO_VMAX`, `OCEAN_COLOR` — same meaning/defaults as the
  seismic/ModEM plot scripts' basemap settings.
- `MAP_XLIM`/`MAP_YLIM` (default `None` — auto from the joint grid's own
  easting/northing extent), `REGION_MARGIN_KM`, `FIG_WIDTH` (cm, map
  panel width — height is derived, equal-scale by construction, same as
  the other plot scripts).
- `AXES_UNITS` (`"km"`/`"latlon"`), `AXES_KM_COMMA`, `LATLON_NTICKS`/
  `LATLON_DECIMALS` — same meaning as the other plot scripts.
- `SHOW_COLORBAR`, `COLORBAR_POSITION`/`COLORBAR_SIZE`/`COLORBAR_ASPECT`/
  `COLORBAR_PAD`/`COLORBAR_LABEL_SIZE`/`COLORBAR_TICK_SIZE` — same
  meaning as the other plot scripts' colorbar settings (see
  `README_seis.md` for the full explanation of the deterministic
  panel/colorbar layout these feed into).
- `SHOW_NORTH_ARROW`, `ARROW_LON`/`ARROW_LAT`/`ARROW_LEN_KM`,
  `ARROW_STYLE`/`ARROW_LABEL_STYLE` — same as the other plot scripts.
- `AXIS_LABEL_SIZE`/`AXIS_TICK_SIZE`/`AXIS_TITLE_SIZE`,
  `ANNOTATION_TEXT`/`ANNOTATION_POS`/`ANNOTATION_STYLE` — same as the
  other plot scripts.

### Resistivity vs. conductivity

`resistivity_to_conductivity()` inverts the resistivity values loaded
from `modem_submesh_points.nc`, reading *that variable's own* `units`
attribute (`"log10(Ohm.m)"`, `"ln(Ohm.m)"`, or `"Ohm.m"`) to apply the
matching inversion — `-log10(ρ)`, `-ln(ρ)`, or `1/ρ` respectively —
rather than assuming a fixed transform. This means it keeps working
correctly regardless of whatever `OUTPUT_TRANSFORM` Part A of
`tacna_precompute.py` used, without needing to duplicate/track that
setting here. The resulting units (`log10(S/m)`, `ln(S/m)`, or `S/m`)
are resolved at load time and used automatically in the printed cluster
table, `tacna_clusters.nc`'s attributes, and plot titles. The inversion
happens on the raw point-cloud values, before RBF interpolation.

### RBF interpolation onto the joint grid

`rbf_interpolate_to_grid()` fits a `scipy.interpolate.RBFInterpolator`
independently for each `CLUSTER_VARS` entry, on that variable's own
native point cloud, then evaluates it at every point of the joint
regular grid (`GRID_EASTING_KM`/`GRID_NORTHING_KM`/`GRID_DEPTH_KM`).
Because each variable is interpolated separately, they don't need to
share a native grid, resolution, or even coverage — the joint grid is
the one place they're all guaranteed to line up. `RBF_NEIGHBORS` (default
`50`) keeps this tractable for the full native-resolution ModEM point
cloud by using a local (K-nearest) RBF fit rather than a single global
fit over every source point at once.

RBF interpolants extrapolate smoothly with no natural cutoff, which
means a query point far from any source data still gets *some* value —
just an increasingly unreliable one. `MASK_TO_CONVEX_HULL` (default
`True`) guards against this by nulling out joint-grid points that fall
outside each variable's own 3-D convex hull (`outside_convex_hull()`,
via `scipy.spatial.Delaunay`), so the clustering only ever sees
genuinely interpolated (not extrapolated) values for each variable.

### Fuzzy c-means implementation

`fuzzy_cmeans(X, n_clusters, m, max_iter, tol, seed)` is a small,
self-contained (NumPy-only) implementation of the standard Bezdek (1981)
algorithm — deliberately not a dependency on `scikit-fuzzy`, to keep
this in line with the rest of the pipeline's minimal-dependency style
(the RBF/convex-hull work above already pulls in `scipy`, which the
pipeline uses elsewhere too). Distances are computed with a per-cluster
loop (not a full `(n_samples, n_clusters, n_features)` broadcast) to
keep memory bounded for large point counts. Unit-tested against
synthetic, well-separated Gaussian blobs (recovers all cluster centers
correctly, FPC ≈ 0.99).

---

## Coordinate convention

Same as the MT and seismic pipelines: **UTM Zone 19S (EPSG:32719)**,
distances in km, depth in km positive down. Unlike either source
pipeline's own native grid, `tacna_clusters.nc`'s `(depth, northing,
easting)` grid is a genuine, uniformly-spaced regular grid — defined
entirely by `GRID_EASTING_KM`/`GRID_NORTHING_KM`/`GRID_DEPTH_KM` here,
not inherited from any source file's own resolution.

## Dependencies

```
numpy, xarray, matplotlib, pyproj, scipy
```
(`scipy.interpolate.RBFInterpolator` for the interpolation,
`scipy.spatial.Delaunay` for convex-hull masking) plus the local
`plotpy.py` helper module. No `scikit-fuzzy` dependency — the fuzzy
c-means implementation is self-contained.

## Typical run

```bash
python3 tacna_precompute.py   # must be run first (or already have been) —
                               # tacna_cluster.py reads its output directly
python3 tacna_cluster.py      # loads native point clouds, RBF-interpolates
                               # onto the joint grid, clusters, writes
                               # tacna_clusters.nc / tacna_cluster_centers.csv
                               # and the maps
```

Run `tacna_cluster.py` again whenever `CLUSTER_VARS`, `USE_CONDUCTIVITY`,
`STANDARDIZE`, the `GRID_*_KM` settings, the `RBF_*` settings,
`MASK_TO_CONVEX_HULL`, `N_CLUSTERS`, `FUZZINESS`, or `RANDOM_SEED` change,
or whenever `tacna_precompute.py` has been re-run with different data.
Changing only plotting settings (colours, `PLOT_DEPTHS_KM`, basemap
styling, colorbar, annotations) still requires re-running the whole
script — clustering and plotting aren't split into separate steps here,
unlike the precompute/plot separation in the other two pipelines.
