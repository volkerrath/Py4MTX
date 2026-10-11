# `plot_interp.py` — Plotting for `interpolate.py` Output

Depth-slice maps and vertical cross-sections of any variable stored in
an `interpolate.py` output file (`{SITE_PREFIX}_interp_{METHOD}.nc`).
Shares the same basemap engine, colorbar placement, feature overlays,
marker/label styles, `VSLICES` schema, and USER SETTINGS layout as
`plot_seis.py` and `plot_femtic_mesh.py`.

Authors: Svetlana Byrdina (SMB) & Volker Rath (DIAS)
AI-assisted development: Claude (Anthropic), 2026-08.
License: GNU General Public License v3 (GPL-3.0-or-later).

---

## Typical run

```bash
python3 plot_interp.py
```

Writes one figure per `(variable, depth)` combination into `PLOT_DIR`,
then one figure per `(variable, profile)` for each entry in `VSLICES`.

---

## Data format

The script reads the 3-D regular UTM-km grid written by `interpolate.py`:

```
dims   : (depth, northing, easting)
coords : depth    [km, positive down]
         northing [km, UTM]
         easting  [km, UTM]
vars   : rho_femtic, vps, dens, *_grad_easting, *_grad_northing,
         *_grad_depth, *_grad_mag   (depends on interpolate.py settings)
```

Any subset of these can be requested; the script exits cleanly if a
requested variable is not present.

---

## USER SETTINGS

### File / directory

| Parameter | Default | Description |
|---|---|---|
| `SITE_PREFIX` | `"ubinas"` | Site tag, matches precompute.py |
| `INTERP_METHOD` | `"nearest"` | Method tag from interpolate.py filename |
| `INTERP_FILE` | `None` | Full path override; `None` = auto-build |
| `NC_DIR` | `"../precompute/"` | NetCDF input directory |
| `NC_DIR_TOPO` | `None` | Topo/bath directory; `None` = same as `NC_DIR` |
| `PLOT_DIR` | `"../plots_interp/"` | Output directory |
| `PLOT_FORMATS` | `[".pdf", ".jpg"]` | Output formats |
| `PLOT_DPI` | `600` | Raster DPI |

### Variables

**`PLOT_VARS`** — list of variable names to plot (must exist in the
NetCDF).  Example: `["rho_femtic", "vps", "rho_femtic_grad_mag"]`.

**`CMAP_SETTINGS`** — dict keyed by variable name, each entry:

```python
CMAP_SETTINGS = {
    "rho_femtic": dict(
        cmap  = "RdBu_r",      # matplotlib name or .cpt path
        cmin  = 0.0,
        cmax  = 3.0,
        label = "log₁₀ ρ (Ω·m)",
        alpha = 0.35,          # data transparency (0 = opaque)
    ),
    ...
}
```

Variables not listed fall back to `CMAP_DEFAULT` (viridis, auto-range
from finite data, `long_name` attribute as label).

### Depth slices

| Parameter | Description |
|---|---|
| `DEPTH_SLICES_KM` | List of depths (km). Matched to nearest grid level. |
| `ZMIN_SEISM` / `ZMAX_SEISM` | Per-depth seismicity window (km). `None` = show all. |

### Map region

| Parameter | Description |
|---|---|
| `REGION_SOURCE` | `"data"` = interpolated grid extent; `"topo"` = topo grid |
| `REGION_MARGIN_KM` | Extra margin added to the auto-region |
| `MAP_XLIM` / `MAP_YLIM` | Explicit override (UTM km). `None` = auto. |

All other map settings (`SHOW_TOPO_BASEMAP`, `AXES_UNITS`, colorbar
geometry, feature on/off switches, marker/label styles, north arrow,
annotation) are identical in name and meaning to `plot_seis.py` and
`plot_femtic_mesh.py` — see `README_seis.md` for the full explanation.

### Vertical slices

Same `VSLICES` schema as the other scripts:

```python
VSLICES = [
    dict(
        name     = "profile AA'",
        p1       = [-70.48, -18.245],   # lon, lat  (coord="latlon")
        p2       = [-69.580, -17.135],
        coord    = "latlon",            # or "utm" for UTM km endpoints
        zmin_km  = -8.0,               # top of section (negative = above sea level)
        zmax_km  = 40.0,
        npts     = 200,                # horizontal sample points
        swath_km = 10.0,               # seismicity projection half-width
        # xlim   = [0, 120],           # optional: crop displayed x-axis
        # ylim   = [-5, 35],           # optional: crop displayed depth-axis
    ),
]
```

Per-variable colour-scale overrides for sections go in `VSLICE_CMAP_OVERRIDES`
(same dict structure as `CMAP_SETTINGS`).

Vertical exaggeration, x-axis mode (`"distance"` / `"utm"`), topo fill,
and VE label position are the same parameters as in `plot_seis.py`.

### Output filenames

```
{var}_interp_{INTERP_METHOD}_{tag}_{SITE_PREFIX}.{ext}      ← depth slice
{var}_interp_{INTERP_METHOD}_section_{name}_{SITE_PREFIX}.{ext}  ← section
```

where `tag` is the actual depth level used (e.g. `6km`).

---

## Relationship to other scripts

| Script | Input | Rendering |
|---|---|---|
| `plot_seis.py` | `{prefix}_vp/vs/vps_utm_{D}km.nc` | `imshow` on regular lat/lon grid |
| `plot_femtic_mesh.py` | `femtic_mesh_utm.nc` (precompute output) | `PolyCollection` of exact mesh polygons |
| `plot_modem_mesh.py` | `modem_*_utm.nc` | `pcolormesh` on tensor cell edges |
| **`plot_interp.py`** | `{prefix}_interp_{method}.nc` | `imshow` on regular UTM-km grid |

`plot_interp.py` is the natural companion to `interpolate.py`: it plots
whatever the interpolation step produced, on a common UTM-km grid, with
the same basemap and feature layers used for all other model views.  Use
it to compare, e.g., `rho_femtic` from the interpolated FEMTIC model
directly against `vps` from the seismic tomography, at the same depths
and on the same coordinate frame.

---

## Dependencies

```
numpy, matplotlib, xarray, pandas, scipy
```

plus the local `tomomt.py` module (same path as the script).
