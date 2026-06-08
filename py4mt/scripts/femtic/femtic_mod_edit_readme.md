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
| `"smooth"` | Gaussian-kernel smoothing with `OP_SMOOTH_SIGMA` |
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
| `OP_SMOOTH_SIGMA` | Gaussian sigma (model-local metres) for `"smooth"` |
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

## 2026-06-08

- Removed `OBSERVE_FILE` and `SITE_NUMBER`: the `observe.dat` fallback
  branch was broken (`OBSERVE_FILE` was never defined). `SITE_DAT` is
  now the sole site source.
