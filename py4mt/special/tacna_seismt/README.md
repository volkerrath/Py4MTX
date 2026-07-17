# Tacna Geophysical Imaging Pipeline

Python/matplotlib pipeline for visualizing seismic tomography (Vp, Vs,
Vp/Vs) and ModEM 3-D magnetotelluric (MT) resistivity inversion results
for the Tacna region, southern Peru. Produces UTM-referenced basemaps,
horizontal depth slices, and arbitrary vertical cross-sections for both
datasets, with a shared basemap/styling engine and no GMT/PyGMT
dependency.

**Authors:** Svetlana Byrdina (SMB) & Volker Rath (DIAS)
**AI-assisted development:** Claude (Anthropic)
**License:** GNU General Public License v3 (GPL-3.0-or-later)

> AI-generated code — review before use in production.

---

## Contents

| File                          | Role                                                          |
|--------------------------------|----------------------------------------------------------------|
| `tacna_precompute_seis.py`     | Reads Chevrot Vp/Vs velocity models + topo/bathymetry, writes UTM-km NetCDF |
| `tacna_plot_seis.py`           | Depth-slice maps and vertical sections for Vp/Vs/Vp-Vs ratio  |
| `tacna_precompute_modem.py`    | Reads a ModEM `.rho`/`.dat`/`.sns` inversion result, writes UTM-km NetCDF |
| `tacna_plot_modem.py`          | Depth-slice maps and vertical sections for log₁₀(ρ)           |
| `modem.py`                     | Shared ModEM file I/O helper library (readers, `get_topo`, `cells3d`, NetCDF export) |

Each `precompute` script is a one-time (or as-needed) heavy-lifting step;
each `plot` script is a fast, repeatable visualization step that only
reads the precomputed NetCDF files. GMT/PyGMT originals are retained
alongside these with a `_gmt` suffix where applicable.

---

## Requirements

```
numpy, scipy, xarray, pandas, matplotlib, pyproj
rioxarray + rasterio + the `elevation` package (eio CLI)   — seis precompute, TOPO_SOURCE="elevation" (default)
netCDF4                                                     — modem.py NetCDF export
```

`tacna_precompute_seis.py` can avoid `rioxarray`/`rasterio` entirely by
setting `TOPO_SOURCE = "etopo"` and pointing `ETOPO_PATH` at a local
ETOPO1/ETOPO2022 NetCDF file — useful if the geospatial stack
(`rasterio`/`gdal`/`libjxl`) is broken in your environment, since that
backend only needs `xarray`.

---

## Coordinate conventions

- All NetCDF outputs and plots use **UTM Zone 19S (EPSG:32719)**, in
  **kilometres**.
- Seismic velocity depth: km, positive down, referenced to the source
  model's own depth coordinate (assumed to already be true depth below
  sea level).
- ModEM depth: km, positive down, referenced to **sea level** — computed
  from the `.rho` file's own reference elevation (`reference[2]`), not
  the mesh's arbitrary top face. See "Known issues" below; this took a
  few iterations to get right.
- ModEM air cells (ρ ≈ 10¹⁷ Ω·m) are masked to `NaN` rather than clipped,
  using `AIR_LOG10_RHO_THRESHOLD`, independent of the display colour
  scale (`CMIN_RHO`/`CMAX_RHO`).

---

## Typical workflow

```bash
# 1. One-time (or as-needed) precompute step
python tacna_precompute_seis.py
python tacna_precompute_modem.py

# 2. Fast, iterable plotting
python tacna_plot_seis.py
python tacna_plot_modem.py
```

Re-run the relevant `precompute` script whenever you change: geographic
crop bounds (`TAR_LON`/`TAR_LAT`), depth range (`DEPTH_RANGE`/`TRIM_PAD`),
or the source model files themselves. Everything else (styling, colour
scales, profile endpoints for display cropping, marker styles, `xlim`,
colourmaps) lives in the `plot` scripts and re-runs in seconds.

### Adding a vertical cross-section

Both plot scripts share a `VSLICES` list — one dict per profile:

```python
VSLICES = [
    dict(
        name     = "profile_CD",
        p1       = [-70.476, -18.255],   # lon, lat (or UTM km, see coord)
        p2       = [-69.499, -17.048],
        coord    = "latlon",
        zmin_km  = -8.0,   # negative enough to reach above sea level
        zmax_km  = 60.0,
        npts     = 200,
        nz       = 150,
        swath_km = 10.0,   # half-width for projecting seismicity/MT sites
        xlim     = None,   # optional [xmin, xmax] to crop the *displayed*
                            # view only, no recompute needed
    ),
]
```

`zmin_km` must be negative enough to include any real topography/seismicity
above sea level in your area (e.g. a volcanic edifice) — the depth-sampling
grid and the seismicity/MT-site filters are both driven by this value.

---

## Colourmap customization

Both plot scripts' colourmap settings (`CMAP_VP`/`CMAP_VS`/`CMAP_VPS` in
`tacna_plot_seis.py`, `CMAP_RHO` in `tacna_plot_modem.py`) accept, in
addition to a matplotlib built-in name:

- **A GMT `.cpt` file** — parsed directly, preserving the file's own
  (possibly non-uniform) colour-stop spacing, e.g.:
  ```python
  CMAP_VP = "./cpt/viridisr_vp.cpt"
  ```
  Useful for reproducing the exact original palette from a GMT-based
  figure for direct visual comparison, rather than a same-ish matplotlib
  named stand-in.
- **A plain `.txt`/`.csv` file** of RGB(A) rows (0–255 or 0–1,
  comma- or whitespace-separated, one colour per line) — built into an
  evenly-spaced `ListedColormap`. Useful for reusing an exact palette
  exported from another tool.
- Anything else is resolved via `plt.get_cmap()` as before — matplotlib
  built-ins, or a name registered by a third-party package
  (`cmcrameri`, `cmocean`, etc.) if you've imported it elsewhere in the
  process.

Implemented by the `load_colormap()` helper (duplicated identically in
both plot scripts, consistent with how other shared utilities are
handled here). Known gaps in the `.cpt` parser: no support for GMT
categorical/patterned fills, `HSV`-mode colour models, or cyclic/hinge
headers — numeric RGB triples and `R/G/B`/hex/grey shorthand are handled.

### Exporting a colourmap back to `.cpt`

The reverse direction is also available — `export_colormap_to_cpt()`
(same function in both scripts) writes whatever colourmap is actually in
use (a matplotlib built-in, or something already imported via
`load_colormap`) out to a GMT-format `.cpt` file, stretched over the same
`CMIN`/`CMAX` range used for display, so it's directly usable in GMT for
the same data. Off by default:

```python
# tacna_plot_modem.py
EXPORT_CPT = True
EXPORT_CPT_PATH = "modem_rho_cmap.cpt"
EXPORT_CPT_NSTEPS = 32   # number of colour segments written

# tacna_plot_seis.py — one file per variable (each has its own range)
EXPORT_CPT = True
EXPORT_CPT_PATHS = {"vs": "seis_vs_cmap.cpt", "vp": "seis_vp_cmap.cpt",
                    "vps": "seis_vps_cmap.cpt"}
```

Round-tripped through `load_colormap()` again, the exported file
reproduces the original colours to within ~1% (limited by 8-bit RGB
quantization and `EXPORT_CPT_NSTEPS` resolution — raise it for finer
fidelity).

---

## Sensitivity-based shading & blanking (ModEM only)

If a `.sns` sensitivity/resolution file exists alongside the `.rho`
model (same grid format, read with the same `mdm.read_mod()` reader),
`tacna_precompute_modem.py` will pick it up and carry it through the
exact same trim/crop steps as the resistivity model, writing
`modem_sens_utm.nc` and `modem_sens_utm_{D}km.nc` alongside the
resistivity outputs. Controlled by, in the precompute script:

```python
USE_SENSITIVITY = True
SENS_FILE = MODEL_FILE      # base name; same as MODEL_FILE by convention
SENS_EXT  = ".sns"
SENS_TRANSFORM = "LOG10"    # sensitivity often spans many orders of magnitude
```

`tacna_plot_modem.py` then uses it for both the horizontal slices and
the vertical sections:

```python
USE_SENSITIVITY = True
SENS_BLANK_THRESHOLD = None   # hard cutoff: cells below this -> NaN (removed)
SENS_SHADE_RANGE = None       # e.g. (-2.0, 0.0): smooth fade from fully
                               # shaded at/below the first value to
                               # unshaded at/above the second
SENS_SHADE_COLOR = "white"
SENS_SHADE_MAX_ALPHA = 0.85
```

Both can be used together, separately, or not at all. Missing sensitivity
data (`NaN` — outside the `.sns` file's own coverage) is always treated as
fully shaded/blanked: there's no basis for treating "we don't know" as
"well resolved."

**Important ordering note:** the section's own surface line
(`surf_depth`, used to position the topography line, MT-site depths, and
the seismicity-in-air filter) is derived from the air/rock mask *before*
sensitivity blanking is applied — a well-resolved physical boundary
shouldn't shift just because sensitivity coverage happens to be patchy
near the surface. If you modify this code, keep that ordering.

---

## Known issues & design notes

A running log of non-obvious bugs found and fixed during development —
useful context if behavior looks wrong after modifying the scripts.

- **Region-crop / profile mismatch.** `TAR_LON`/`TAR_LAT` (velocity
  subset and ModEM `CROP_TO_REGION` bounds) must fully contain every
  `VSLICES` profile endpoint. The two are defined independently in
  different files/sections and can silently drift apart — this caused a
  real "white gap at both edges of the section" bug in both pipelines.
  Widen `TAR_LON`/`TAR_LAT` (and `MAP_LON`/`MAP_LAT` for topo/bath) if you
  add a profile that reaches further.

- **ModEM depth axis vs. topography reference.** `get_topo()` computes
  surface elevation as `cumsum(dz) + reference[2]`, anchoring to sea
  level. The model's own depth axis must add that same `reference[2]`
  offset (`build_depth_axis_km(dz, ref_z=reference[2] + z_trim_offset_m)`)
  — otherwise the depth axis and the topography are anchored to two
  different zero points, and no interpolated depth can ever come out
  negative (above sea level) regardless of `zmin_km`. Manifested as: MT
  sites/seismicity impossible to place above sea level, and a large
  spurious "no data" gap between the (wrong) topo line and where the
  colour data actually started.

- **ModEM `DEPTH_RANGE`/crop must also allow negative depths at the
  source.** Fixing the plot-script depth filter isn't enough if the
  precompute step already cropped out above-sea-level coverage
  (`DEPTH_RANGE = [0, 100]` in the seis precompute had the same issue).
  `xr.Dataset.sel(slice(...))` clips gracefully to whatever the source
  actually covers, so widening the requested range is always safe — check
  the printed actual depth range to confirm real coverage was gained.

- **Air-cell masking must be independent of the display colour scale.**
  Clipping to `[CMIN_RHO, CMAX_RHO]` and then NaN-ing both ends
  conflates "too resistive to display nicely" with "actually air" — real
  volcanic edifice rock is often genuinely >10⁴ Ω·m. Use a dedicated
  `AIR_LOG10_RHO_THRESHOLD` (comfortably between realistic rock and the
  true air value ~10¹⁷ Ω·m) instead.

- **Sensitivity blanking must not move the surface line.** See the
  ordering note in the sensitivity section above — computing
  `surf_depth` after sensitivity blanking (instead of before) would let
  patchy `.sns` coverage distort the topography line, MT-site depths, and
  the seismicity-in-air filter.

- **MT sites don't sit at a fixed z=0.** Interpolate their depth from the
  section's own surface at each site's along-profile position, not a
  hardcoded 0 — real topography varies along a profile.

- **Seismicity can appear "in the air" after a swath projection.** With a
  wide `swath_km`, an event's true (off-profile) position can have
  different local relief than the profile line itself. Events computed
  shallower than the local section surface are dropped from that
  particular figure rather than drawn floating above ground.

- **`pcolormesh` + `alpha<1` + vector output (PDF) shows seams** between
  adjacent cells from anti-aliasing. Don't fix this with
  `set_rasterized(True)` — rasterized artists render **upside-down**
  under matplotlib's PDF/PS backends when combined with
  `ax.invert_yaxis()` (used for the depth axis), a known matplotlib bug.
  Use `shading="gouraud"` instead (smooth per-vertex interpolation, no
  discrete cell polygons, no rasterization needed). The sensitivity-shade
  overlay on vertical sections reuses this same `pcolormesh`/`gouraud`
  approach (with a per-cell alpha array) rather than `imshow`, to stay on
  the identical grid/orientation as the main data.

- **Custom "pin" marker for MT sites.** A plain `marker="v"` is centred
  on the data point, so the tip doesn't actually mark the true location.
  `MT_PIN_MARKER` is a custom `matplotlib.path.Path` triangle with its
  apex at `(0, 0)` so the marker's tip — not its centroid — lands exactly
  on the site coordinate, in both map and section views.

---

## Output files

**Seismic (`tacna_precompute_seis.py`):**
`tacna_topo_utm.nc`, `tacna_bath_utm.nc`, `tacna_pvelocity_subset.nc`,
`tacna_vp.nc`, `tacna_vs.nc`, `tacna_vps.nc`, per-depth
`tacna_{vp,vs,vps}_utm_{tag}.nc`

**ModEM (`tacna_precompute_modem.py`):**
`modem_model_utm.nc`, `modem_topo_utm.nc`, `modem_sites_utm.nc`,
`modem_rho_utm_{D}km.nc` (one per `DEPTH_SLICES_KM` entry), plus, only if
a `.sns` file was found: `modem_sens_utm.nc` and
`modem_sens_utm_{D}km.nc`

**Figures:** `{seis,modem}_map_{var}_{depth}.{png,pdf,jpg}` (per
`PLOT_FORMATS`), `{seis,modem}_section_{profile_name}_tacna.{png,pdf,jpg}`
