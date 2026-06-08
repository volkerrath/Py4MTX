# femtic_mod_edit.py

Edit a FEMTIC resistivity block file — set values, smooth, perturb,
clip, or null selected regions — and optionally plot the result.

---

## Purpose

`femtic_mod_edit.py` reads a `resistivity_block_iterX.dat`, applies one
of several algebraic operations to the free-parameter regions, writes the
modified block, and optionally plots slices of the result via
`fviz.plot_model_slices`.

Typical use cases:

- resetting the starting model before a re-inversion,
- clamping extreme resistivity values after convergence,
- applying a smooth Gaussian perturbation for sensitivity testing,
- nulling selected regions to isolate anomalies.

---

## Operations (`OPERATION`)

| Value | Effect |
|---|---|
| `"fill"` | Set all free regions to `OP_FILL_VALUE` (log₁₀ Ω·m) |
| `"smooth"` | Spatial smoothing in log₁₀ space (kernel set by `OP_SMOOTH_MODE`) |
| `"perturb"` | Add Gaussian noise with std `OP_PERTURB_STD` |
| `"clip"` | Clamp log₁₀(ρ) to `[OP_CLIP_MIN, OP_CLIP_MAX]` |
| `"null"` | No-op — reads and rewrites the block unchanged (useful for plot-only) |

---

## Workflow

```
MODEL_IN  +  MESH_FILE
        |
        v  fviz.read_femtic_mesh / fviz.read_resistivity_block
   mesh geometry  +  region arrays
        |
        v  apply OPERATION
   modified log₁₀(ρ) vector
        |
        v  fem.insert_model → write MODEL_OUT
        |                                    [PLOT = True]
        v  fem.resolve_slice_positions(PLOT_SLICES)
        v  fviz.plot_model_slices(...)
slice figure saved / shown
```

---

## Key configuration variables

### Paths
| Variable | Description |
|---|---|
| `WORK_DIR` | Base directory |
| `MESH_FILE` | `mesh.dat` |
| `MODEL_IN` | Input resistivity block |
| `MODEL_OUT` | Output resistivity block |
| `SITE_DAT` | `site.dat` CSV (`None` to disable) |

### Operation
| Variable | Description |
|---|---|
| `OPERATION` | One of `"fill"`, `"smooth"`, `"perturb"`, `"clip"`, `"null"` |
| `OP_FILL_VALUE` | Target log₁₀(ρ) for `"fill"` |
| `OP_SMOOTH_MODE` | Smoothing kernel: `"physical"` (global-σ Gaussian, original) \| `"knn_uniform"` (flat K-NN average) \| `"knn_gauss"` (per-region Gaussian) |
| `OP_SMOOTH_SIGMA` | Global Gaussian σ in metres — `"physical"` mode only |
| `OP_SMOOTH_K` | Number of nearest neighbours (all modes) |
| `OP_SMOOTH_KNN_SIGMA_FRAC` | Per-region σ fraction for `"knn_gauss"`: σ_i = frac × d_{i,K} |
| `OP_PERTURB_STD` | Noise standard deviation (log₁₀ Ω·m) for `"perturb"` |
| `OP_CLIP_MIN/MAX` | log₁₀(ρ) bounds for `"clip"` |

### Origin estimation
| Variable | Description |
|---|---|
| `ORIGIN_METHOD` | `None` / `"box"` / `"average"` |
| `UTM_ORIGIN_LAT/LON/E/N` | Fallback origin (used when `ORIGIN_METHOD=None`) |
| `UTM_ZONE_OVERRIDE` | Force a UTM zone; `None` = auto |

### Plot
| Variable | Description |
|---|---|
| `PLOT` | `True` to produce a slice figure after editing |
| `PLOT_FILE` | Output path; `None` = interactive |
| `PLOT_CMAP` | Colormap (default `"turbo_r"`) |
| `PLOT_CLIM` | `[vmin, vmax]` in log₁₀(Ω·m) |
| `PLOT_SLICES` | Slice dicts — same format as `femtic_mod_plot.py` |
| `PLOT_XLIM/YLIM/ZLIM` | Axis limits in model-local metres |
| `DEPTH_KM` | `True` → depth axis in km |
| `HORIZ_KM` | `True` → horizontal axes in km |
| `PLOT_EQUAL_ASPECT` | Equal aspect ratio |
| `PLOT_PANEL_HEIGHT` | Panel height in cm |
| `DISPLAY_COORDS` | `"model"` / `"utm"` / `"latlon"` |

### Site overlay
| Variable | Description |
|---|---|
| `SITE_DAT` | Site list CSV; `None` to disable |
| `SITE_NAMES` | Filter; `None` = all |
| `PLOT_SITES_MAPS` | Show sites on map panels |
| `PLOT_SITES_SLICES` | Show sites on curtain panels |
| `PROJECTION_DIST` | Max distance (m) for curtain projection |
| `SITE_MARKER` | Marker style dict |
| `SITE_MARKER_SLICES` | Marker style for curtains |
| `MAP_MARKERS` | Additional map markers |

---

## Geometry helpers

The geometry primitives used by `"smooth"`, `"wmean"`, `"ellipsoid"`, and
`"brick"` all live in `femtic.py` and are called via the `fem.*` namespace:

| femtic.py function | Role in mod_edit |
|---|---|
| `fem.tet_volumes()` | Called by `fem.build_region_geometry()` |
| `fem.build_region_geometry()` | Builds `region_ctr` / `region_vol` for smooth, wmean |
| `fem.ellipsoid_mask()` | Containment test for `"ellipsoid"` bodies |
| `fem.brick_mask()` | Containment test for `"brick"` bodies |

`_smooth_body_boundary` and `_apply_bodies` remain local — they reference the
`OUT` config global and are specific to the body-editing workflow.

---

## Changes from previous version

- `plot_model_slices` call now includes `depth_km`, `horiz_km`,
  `equal_aspect`, `panel_height`, `nrows`, `ncols`, `site_xys`,
  `utm_origin_*`, `utm_zone`, `utm_northern`, `display_coords`,
  `obs_coords_only`, `sites_in_maps`, `sites_in_slices`,
  `site_marker_slices`, `map_markers`, `projection_dist` kwargs.
- `PLOT_SLICES` is now resolved via `fem.resolve_slice_positions`
  (supporting `"utm"` / `"latlon"` CRS tagging) before the call.
- Added `DEPTH_KM`, `HORIZ_KM`, `PLOT_EQUAL_ASPECT`, `PLOT_PANEL_HEIGHT`,
  `ORIGIN_METHOD`, `DISPLAY_COORDS`, `SITE_DAT`, `SITE_NAMES`,
  `PLOT_SITES_*`, `SITE_MARKER_SLICES`, `MAP_MARKERS`,
  `PROJECTION_DIST`, `UTM_ORIGIN_*`, `UTM_ZONE_OVERRIDE` config vars.

## 2026-06-08 (smoothing modes)

- Added `OP_SMOOTH_MODE` (`"physical"` | `"knn_uniform"` | `"knn_gauss"`).
  `"physical"` preserves the original global-σ Gaussian.  `"knn_uniform"`
  and `"knn_gauss"` are mesh-adaptive: the smoothing footprint tracks local
  cell size rather than a fixed physical distance.
- New config vars: `OP_SMOOTH_MODE` (default `"physical"`),
  `OP_SMOOTH_KNN_SIGMA_FRAC` (default `0.5`).

## 2026-06-08 (geometry refactor)

- Removed private helpers `_tet_volumes`, `_build_region_geometry`,
  `_rotation_matrix_zyx`, `_local_coords`, `_ellipsoid_mask`, `_brick_mask`.
  All geometry primitives now live in `femtic.py` and are called via
  `fem.build_region_geometry()`, `fem.ellipsoid_mask()`, `fem.brick_mask()`.
- `_smooth_body_boundary` and `_apply_bodies` remain local.
