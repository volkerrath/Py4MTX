# `interpolate.py` — Interpolation onto a Common Grid

Loads MT resistivity/conductivity/sensitivity + seismic tomography
properties (Vp, Vs, Vp/Vs, density), each from its own native grid
produced by `precompute.py`, and interpolates every requested one
onto one common target grid via RBF, ordinary kriging, inverse-distance
weighting (IDW), or nearest-neighbor.

This is the interpolation half of the pipeline; see `README_cluster.md`
for `cluster.py`, which reads this script's output and clusters it.

Authors: Svetlana Byrdina (SMB) & Volker Rath (DIAS)
AI-assisted development: Claude (Anthropic)
License: GNU General Public License v3 (GPL-3.0-or-later)
AI-generated code — review before use in production.

---

## Pipeline

```
precompute.py  →  interpolate.py  →  {SITE_PREFIX}_interp_<method>.nc  →  cluster.py  →  {SITE_PREFIX}_clusters.nc + figures
```

`interpolate.py` does not build or interpolate onto a specific
target unconditionally — it reads every variable in `INTERP_VARS`, each
from its own native grid, and interpolates each one independently onto
whichever target grid `TARGET_GRID` selects. It has no knowledge of
clustering; `cluster.py` reads its output (`INTERP_FILE`) but
otherwise builds no grid and interpolates nothing itself.

---

## Native sources (`VARIABLE_SOURCES`)

Three kinds of native source, each with its own loader
(`load_variable_points()` dispatches):

- **`"modem_points"`** — a flat point table (`precompute.py`'s
  `modem_submesh_points.nc`, Part A): already `(easting_km, northing_km,
  depth_km, value)` rows at full native ModEM resolution. Used for MT
  resistivity/conductivity/sensitivity. `precompute.py` already
  masks both air cells (`AIR_RHO_THRESHOLD`, resistivity >= 1e10 Ω·m)
  and near-zero sentinel/placeholder values some `.rho`/`.sns` files use
  for invalid cells (`ILLEGAL_LOW_THRESHOLD`, e.g. a 1e-32 no-data flag)
  to NaN before this table is written, so `interpolate.py` never
  sees either kind of illegal value as if it were real data.
- **`"seis_grid"`** — a gridded `(depth, row, col)` cube
  (`precompute.py` Part B's `{SITE_PREFIX}_vp.nc` / `{SITE_PREFIX}_vs.nc` /
  `{SITE_PREFIX}_vps.nc` / `{SITE_PREFIX}_dens.nc`), with 2-D `utm_easting`/`utm_northing`
  aux coords — flattened, with the horizontal coords broadcast across
  every depth level, into a point cloud by `load_seis_grid_points()`.
  "row"/"col" are read from the source file's own dimension names —
  never assumed to be lat/lon — so this works regardless of what
  `precompute.py`'s Part B actually calls them.
- **`"femtic_points"`** — a FEMTIC tetrahedral mesh (`mesh.dat`) +
  resistivity block (`resistivity_block_iterX.dat`), read directly via
  `femtic.py` — no `precompute.py` step of its own, since FEMTIC's own
  output files are already the native source. Each mesh element's
  centroid + `log10(resistivity)` becomes one point, via
  `load_femtic_points()`. See "FEMTIC meshes" below.

`INTERP_VARS` picks which of these get loaded, interpolated, and
written into `INTERP_FILE`. It's worth interpolating a superset once:
`cluster.py`'s own `CLUSTER_VARS` can pick any subset of what's
available in `INTERP_FILE` for a given clustering run, without
re-running this script, as long as it doesn't need a variable that
isn't in `INTERP_VARS`.

### Resistivity vs. conductivity

`resistivity_to_conductivity()` inverts the resistivity values loaded
from `modem_submesh_points.nc`, reading *that variable's own* `units`
attribute (`"log10(Ohm.m)"`, `"ln(Ohm.m)"`, or `"Ohm.m"`) to apply the
matching inversion, so it keeps working regardless of
`precompute.py`'s `OUTPUT_TRANSFORM`. `USE_CONDUCTIVITY` swaps any
`"rho"` in `INTERP_VARS` for `"cond"` at load time. (`derive`-based
transforms like this one aren't currently wired up for
`"femtic_points"` entries, since FEMTIC's own resistivity convention is
already `log10(Ohm.m)`, same as `modem_points`' `"rho"` — nothing to
invert differently there yet.)

### FEMTIC meshes

`load_femtic_points()` reads a FEMTIC tetrahedral mesh + resistivity
block directly (via `femtic.read_femtic_mesh()` /
`femtic.read_resistivity_block()` / `femtic.build_element_arrays()`)
and converts every free (non-fixed) element's centroid into a
`(easting_km, northing_km, depth_km)` point with its
`log10(resistivity)` value — the same shape/units contract as
`load_modem_points()`, so it plugs into `build_joint_grid()`'s
auto-bounds computation and the RBF/kriging/IDW/nearest interpolators
exactly like a ModEM or seismic source.

**Coordinate conversion.** FEMTIC's `mesh.dat` stores node coordinates
in *model-local metres*: x/y are UTM easting/northing offset by the
mesh's own local origin (axes aligned with UTM east/north, no rotation
— `femtic.py`'s own `utm_to_model()`), and z is positive *downward* in
metres (`femtic.py`'s documented convention, already matching this
pipeline's own depth-positive-down convention — no sign flip needed,
only the unit conversion to km). `FEMTIC_ORIGIN_E_M`/`FEMTIC_ORIGIN_N_M`
(UTM **metres**) are therefore **required** for any `"femtic_points"`
entry — there's no safe "auto" origin to fall back to; get them from the
FEMTIC run's own setup, or from `femtic.estimate_utm_origin()` against
known calibration site positions. `FEMTIC_DEPTH_OFFSET_KM` (default
`0.0`) is available if the mesh's own z=0 datum doesn't already match
this project's z=0 reference.

**Region exclusion.** By default (`FEMTIC_INCLUDE_FIXED = False`),
air (region 0), any explicitly flagged-fixed region (`flag == 1`), and
region 1 if auto-inferred (or forced via `FEMTIC_OCEAN`) to be ocean are
all excluded before building the point cloud — mirroring
`femtic.read_model()`'s own semantics exactly (including: a `flag == 1`
region stays excluded even if `FEMTIC_OCEAN = False` is set explicitly),
just applied per mesh *element* via `region_of_elem` rather than
`read_model()`'s per-*region* output, since this pipeline needs one
point per element. Set `FEMTIC_INCLUDE_FIXED = True` to keep every
element instead.

**A ModEM and a FEMTIC resistivity side by side.** Since
`"femtic_points"` is just another native source feeding the same
point-cloud → interpolation pipeline, a ModEM-derived `"rho"` and a
FEMTIC-derived `"rho_femtic"` (or whatever key you give it) can both be
declared in `VARIABLE_SOURCES` and interpolated onto the *same* joint
grid in the same run — directly comparable, e.g. as a two-inversion-code
cross-check via `structure.py`'s cross-gradient/cosine-similarity/
Gramian diagnostics between the two.

**Dependency note.** `femtic.py` imports `ensembles.py` unconditionally
at module level (for roughness/prior-covariance tools this pipeline
doesn't use) — `ensembles.py` must be importable on `sys.path` for
`load_femtic_points()` to work, same as `pykrige` must be installed for
`INTERP_METHOD = "kriging"`. Neither is a hard dependency of
`interpolate.py` itself; both are only imported lazily, inside the
function that actually needs them, the first time a
`"femtic_points"`/`"kriging"` entry is actually used.

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
  `cluster.py`'s plotting handles this with
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

Output: `{SITE_PREFIX}_interp_<INTERP_METHOD>.nc` in `NC_DIR` (override with
`OUTPUT_FILE`) — grid coords/dims (varying by `TARGET_GRID`), one data
variable per `INTERP_VARS` entry (with `units`/`long_name` attrs), and
attributes recording `target_grid_mode`, `dim_names`, `interp_method`,
`interp_vars`, and the method-specific settings used — everything
`cluster.py` needs to read it back correctly.

---

## Gradient (optional, `COMPUTE_GRADIENT`)

If enabled, computes the spatial gradient of each `GRADIENT_VARS` entry
on the same grid the field was just interpolated onto, and stores it in
`{SITE_PREFIX}_interp_<method>.nc` alongside it as four extra data variables per
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

Gradients are **not** added to `interp_vars`, so `cluster.py`'s
default (`CLUSTER_VARS = None`, "cluster on everything") never picks
them up automatically — add `"{key}_grad_mag"` etc. to `CLUSTER_VARS`
explicitly there if you want to cluster on gradients too.

---

## Coordinate convention

**UTM Zone 19S (EPSG:32719)**, distances in km, depth in km positive
down. A `"joint"`-mode `{SITE_PREFIX}_interp_<method>.nc` has a genuine,
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
python3 precompute.py      # must be run first (or already have been)
python3 interpolate.py     # loads native point clouds, builds/reuses the
                                 # target grid, interpolates (RBF/kriging/IDW/
                                 # nearest), optionally computes gradients,
                                 # writes {SITE_PREFIX}_interp_<method>.nc
```

Re-run whenever `INTERP_VARS`, `USE_CONDUCTIVITY`, `TARGET_GRID`, the
`GRID_*_KM`/`SEISMIC_MESH_VAR` settings, `INTERP_METHOD` or its
`RBF_*`/`KRIGING_*`/`IDW_*` settings, `MASK_TO_CONVEX_HULL`,
`APPLY_ROI_MASK`/`ROI_*`, `COMPUTE_GRADIENT`/`GRADIENT_*`, or
`precompute.py`'s own output change.
