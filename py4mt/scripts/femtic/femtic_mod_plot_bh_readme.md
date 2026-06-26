# femtic_mod_plot_bh.py — README

**Purpose:** Standalone 1-D ρ(z) borehole resistivity logs for a FEMTIC
tetrahedral resistivity model.

Sister scripts:
- [`femtic_mod_plot_slice.py`](femtic_mod_plot_slice_readme.md) — 2-D map / curtain / plane slice panels.
- [`femtic_mod_plot_3d.py`](femtic_mod_plot_3d_readme.md) — PyVista 3-D rendering and VTK/VTU export.

---

## What this script does

```
mesh.dat + resistivity_block_iterX.dat
        │
        └─(3)─► fviz.plot_borehole_logs(...)  [1-D ρ(z) traces, log x-axis]
                 → PDF / NPZ / interactive
```

All sampling and plotting logic lives in `femtic_viz.py`; no geometry code
lives here.

---

## Execution steps

| Step | What happens |
|---|---|
| (1) | Optionally estimate UTM mesh-origin from `SITE_DAT` bounding-box or mean |
| (2) | Derive UTM zone from finalised `UTM_ORIGIN_LAT` / `UTM_ORIGIN_LON` |
| (3) | Sample boreholes and plot via `fviz.plot_borehole_logs` |

---

## Borehole position coordinate systems

Each spec dict `"x"` / `"y"` field accepts:

| Value | Meaning |
|---|---|
| plain `float` | model-local metres — origin at mesh centre |
| `(value, "utm")` | UTM metres in the mesh UTM zone |
| `(value, "latlon")` | decimal degrees (longitude for x, latitude for y) |

`"x"` and `"y"` must carry the **same** CRS tag.  For `"latlon"` and `"utm"`
the legend shows geographic coordinates automatically — no separate `"lat"`/`"lon"`
key needed.

Conversion chain: `lat/lon → UTM → model-local`.

---

## Borehole spec dict keys

| Key | Type | Required | Description |
|---|---|---|---|
| `"name"` | str | yes | Label in legend / panel title |
| `"x"` | float or `(v, "crs")` | yes | Easting: float = model-local m; `(lon, "latlon")` = longitude [°]; `(E_m, "utm")` = UTM easting [m] |
| `"y"` | float or `(v, "crs")` | yes | Northing: float = model-local m; `(lat, "latlon")` = latitude [°]; `(N_m, "utm")` = UTM northing [m] |
| `"z_top"` | float or `"surface"` | no (def. 0) | Start depth [m, z-down]. `"surface"` → auto from mesh nodes (requires scipy) |
| `"z_bot"` | float | no (def. 20000) | End depth [m, z-down] |
| `"dz"` | float | no (def. 200) | Sampling interval [m] |
| `"lat"` | float | no | Override legend latitude [°] (auto-inferred for `"latlon"` / `"utm"` CRS) |
| `"lon"` | float | no | Override legend longitude [°] (auto-inferred for `"latlon"` / `"utm"` CRS) |
| `"color"`, `"ls"`, `"lw"`, `"marker"`, `"alpha"`, … | any | no | Matplotlib `Line2D` kwargs — override `BOREHOLE_STYLE` for this trace |

---

## Example

```python
BOREHOLE_SITES = [
    # Geographic coordinates — legend auto-shows lat/lon
    dict(name="BH-centre",
         x=(-70.868, "latlon"), y=(-16.363, "latlon"),
         z_top="surface", z_bot=20000., dz=200.,
         color="steelblue", ls="-"),

    # UTM — legend back-converts to lat/lon automatically
    dict(name="BH-north",
         x=(229047., "utm"), y=(8190000., "utm"),
         z_top=0., z_bot=15000., dz=100.,
         color="firebrick", ls="--"),

    # Model-local — legend shows x/y in metres
    dict(name="BH-origin", x=0.0, y=0.0,
         z_top=0., z_bot=10000., dz=250.,
         color="seagreen", ls="-."),
]
```

---

## Configuration reference

### Paths

| Variable | Description |
|---|---|
| `WORK_DIR` | Working directory prefix for all paths |
| `MODEL_FILE` | `resistivity_block_iterX.dat` to sample |
| `MESH_FILE` | `mesh.dat` |
| `SITE_DAT` | `mt_make_sitelist.py` CSV (`name,lat,lon,elev,sitenum,easting,northing`); used for origin estimation only; `None` to disable |

### Ocean / air

| Variable | Default | Description |
|---|---|---|
| `AIR_RHO` | `1e9` | Air sentinel Ω·m (region 0) |
| `OCEAN_RHO` | `0.25` | Ocean sentinel Ω·m (region 1) |

### UTM / CRS

| Variable | Default | Description |
|---|---|---|
| `UTM_ORIGIN_LAT` | `None` | Mesh-centre latitude [°] |
| `UTM_ORIGIN_LON` | `None` | Mesh-centre longitude [°] |
| `UTM_ORIGIN_E` | `None` | Mesh-centre UTM easting [m] |
| `UTM_ORIGIN_N` | `None` | Mesh-centre UTM northing [m] |
| `UTM_ZONE_OVERRIDE` | `None` | Force UTM zone number; `None` = auto |
| `ORIGIN_METHOD` | `"box"` | `None` / `"box"` / `"average"` — origin estimation from `SITE_DAT` |

### Borehole

| Variable | Default | Description |
|---|---|---|
| `BOREHOLE_FILE` | `*_boreholes.pdf` | Output path for figure; `None` = interactive |
| `BOREHOLE_SITES` | `[]` | List of borehole spec dicts (see above) |
| `BOREHOLE_STYLE` | `dict(lw=1.2, marker="none")` | Baseline line style; per-spec keys override |
| `BOREHOLE_XLIM` | `[1., 1e4]` | x-axis limits [Ω·m, log scale]; `None` = auto |
| `BOREHOLE_SHARED` | `True` | `True` = all boreholes on one shared axes; `False` = one panel per borehole |
| `BOREHOLE_MARKERS` | `[]` | List of free-annotation dicts (see below) |
| `LEGEND_FONTSIZE` | `9` | Legend / panel-title font size; tick labels = `LEGEND_FONTSIZE - 2` |
| `TICK_FONTSIZE` | `None` | Tick-label font size; `None` → `LEGEND_FONTSIZE - 2` (≥ 6) |
| `LABEL_FONTSIZE` | `None` | Axis-label font size; `None` → `LEGEND_FONTSIZE` |
| `BOREHOLE_NPZ` | `True` | NPZ export: `True` = auto path (same stem as `BOREHOLE_FILE`, `.npz`); `False` = skip; explicit path = save there |
| `PLOT_DPI` | `600` | Saved-figure DPI |

---

## Free markers (`BOREHOLE_MARKERS`)

`BOREHOLE_MARKERS` is a list of dicts that add annotated arrows to the depth
panels after all traces are drawn.  Each dict may contain:

| Key | Type | Required | Description |
|---|---|---|---|
| `"depth"` | float | **yes** | Arrow tip depth [m, z-down] |
| `"rho"` | float | no | Arrow tip x-position [Ohm·m]; defaults to left x-axis edge |
| `"rho_text"` | float | no | Text x-position [Ohm·m] — same units as x-axis; defaults to `rho * 3` |
| `"depth_text"` | float | no | Text y-position [m, z-down] — same units as depth; defaults to `depth - 300 m` |
| `"text"` | str | no | Annotation text (default `""`) |
| `"borehole"` | str or list of str | no | Target borehole name(s); `None` or absent = all panels |
| `"arrowprops"` | dict | no | Forwarded to `ax.annotate`; default `dict(arrowstyle="->", color="black", lw=0.9)` |
| `"color"`, `"fontsize"`, `"fontweight"`, `"ha"`, `"va"`, `"zorder"`, … | any | no | Any remaining keys forwarded verbatim to `ax.annotate` |

Both `"rho_text"` and `"depth_text"` are in the **natural display units of
the axes** (Ohm·m and metres respectively), so you can read values directly
off the plot when deciding where to place the text.

### Example

```python
BOREHOLE_MARKERS = [
    dict(depth=1500., rho=10.,
         rho_text=30., depth_text=1200.,
         text="conductor",
         borehole="borehole1",
         color="red", fontsize=8, fontweight="bold",
         arrowprops=dict(arrowstyle="->", color="red", lw=1.2)),
    dict(depth=3200., rho_text=500., depth_text=2900.,
         text="resistive basement", color="navy", fontsize=8),
]
```

---

## NPZ export (`BOREHOLE_NPZ`)

Sampled depth and resistivity arrays are always written to an NPZ file before
plotting (unless `BOREHOLE_NPZ = False`).

| Array key | Shape | Contents |
|---|---|---|
| `header` | scalar str | JSON metadata (model file, mesh file, timestamp, per-borehole geometry) |
| `depth_<name>` | `(n_levels,)` | Depth samples [m, z positive-down] |
| `rho_<name>` | `(n_levels,)` | Resistivity [Ω·m]; `NaN` where outside mesh or air |

`<name>` is the borehole `"name"` key with spaces replaced by underscores.

**Reading the NPZ:**

```python
import numpy as np, json

d   = np.load("borehole_logs.npz", allow_pickle=False)
hdr = json.loads(str(d["header"]))
for bh in hdr["boreholes"]:
    depth = d[bh["depth_key"]]   # m, z-down
    rho   = d[bh["rho_key"]]     # Ohm*m
    print(bh["name"], depth.shape)
```

---

## Dependencies

| Package | Role |
|---|---|
| `femtic` (Py4MTX) | `read_site_dat`, `extract_borehole_log`, `latlon_to_model`, `utm_to_model` |
| `femtic_viz` (Py4MTX) | `plot_borehole_logs` |
| `util` (Py4MTX) | `utm_zone_from_latlon`, `utm_to_latlon_zn`, `print_title` |
| `matplotlib` | 2-D rendering (via `femtic_viz`) |
| `scipy` | Only for `z_top="surface"` KD-tree lookup |
| `numpy` | Array operations |

---

## Provenance

| Date | Author | Note |
|---|---|---|
| 2026-05-16 | vrath / Claude Sonnet 4.6 | Borehole step created inside `femtic_mod_plot.py` |
| 2026-06-03 | Claude Sonnet 4.6 | Carried into `femtic_mod_plot_slice.py` after script split from `femtic_mod_plot.py`. `BOREHOLE_XLIM` in Ω·m; `z_top="surface"`; lat/lon legend; per-trace line-style keys; `BOREHOLE_NPZ` |
| 2026-06-04 | vrath / Claude Sonnet 4.6 | **Split** from `femtic_mod_plot_slice.py` into this dedicated script. `PLOT_BOREHOLE` flag removed (script is the flag). `BOREHOLE_IN_SLICE` removed (handled in `femtic_mod_plot_slice.py`). UTM origin preamble retained so script runs independently |
| 2026-06-04 | vrath / Claude Sonnet 4.6 | Added `BOREHOLE_MARKERS` (free arrow + text annotations) and `LEGEND_FONTSIZE`; forwarded to `fviz.plot_borehole_logs` |
| 2026-06-26 | vrath / Claude Sonnet 4.6 | Added `TICK_FONTSIZE` and `LABEL_FONTSIZE` (both default `None` → derived from `LEGEND_FONTSIZE`); forwarded to `fviz.plot_borehole_logs` as `tick_fontsize` / `label_fontsize`. |
