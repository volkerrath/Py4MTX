# femtic_mod_plot_3d.py — README

**Purpose:** PyVista 3-D rendering and VTK/VTU export for a FEMTIC
tetrahedral resistivity model.

Sister script: [`femtic_mod_plot_slice.py`](femtic_mod_plot_slice_readme.md) —
2-D slice panels and borehole logs from the same files.

---

## What this script does

```
mesh.dat + resistivity_block_iterX.dat
        │
        └─(3)─► fviz.plot_model_3d(...)
                 ├─► .vtu / .vtk    [ParaView / Zenodo export, optional]
                 ├─► .png / .jpg    [static screenshot]
                 ├─► .html          [interactive WebGL, if trame_vtk present]
                 └─► interactive    [PyVista window, if None]
```

The 3-D scene is constructed in `femtic_viz.plot_model_3d` and can include:

- **Axis-aligned slice planes** (XY map, YZ N-S, XZ E-W) at any number of depths / positions.
- **Arbitrary oblique planes** defined by an origin point and a normal vector.
- **Iso-surfaces** of the selected scalar (log10(ρ) or ρ directly).
- **Spatial clipping** to `PLOT_XLIM`/`PLOT_YLIM`/`PLOT_ZLIM` before export and rendering.

Requires [PyVista](https://docs.pyvista.org/)
(`conda install -c conda-forge pyvista`).

---

## Execution steps

| Step | What happens |
|---|---|
| (1) | Optionally estimate UTM mesh-origin from `SITE_DAT` bounding-box or mean |
| (2) | Derive UTM zone from finalised `UTM_ORIGIN_LAT` / `UTM_ORIGIN_LON` |
| (3) | Render 3-D scene and/or export VTU grid via `fviz.plot_model_3d` |

---

## Configuration reference

### Paths

| Variable | Description |
|---|---|
| `WORK_DIR` | Working directory prefix for all paths |
| `MODEL_FILE` | `resistivity_block_iterX.dat` to display |
| `MESH_FILE` | `mesh.dat` |
| `SITE_DAT` | `mt_make_sitelist.py` CSV — used only for origin estimation; `None` to disable |

### Ocean / air

| Variable | Default | Description |
|---|---|---|
| `OCEAN` | `None` | `None` = auto-infer; `True`/`False` = force |
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

### Spatial clip box

Both the VTU grid export and the PyVista scene are clipped to this box
before any rendering.

| Variable | Default | Description |
|---|---|---|
| `PLOT_XLIM` | `[-15000., 15000.]` | Easting limits [m] |
| `PLOT_YLIM` | `[-15000., 15000.]` | Northing limits [m] |
| `PLOT_ZLIM` | `[-6000., 15000.]` | Depth limits [m] (z positive-down) |

### Output files

| Variable | Default | Description |
|---|---|---|
| `PLOT3D_FILE` | `*_3d.png` | Rendered output — `.png`, `.jpg`, `.html`, `.vtu/.vtk`, or `None` (interactive window) |
| `PLOT3D_VTU_FILE` | `*.vtu` | Separate VTK/VTU export for ParaView; `None` = skip |
| `PLOT_DPI` | `600` | DPI for screenshot files |

Output format is determined by extension:

| Extension | Result |
|---|---|
| `.png`, `.jpg` | Static screenshot (offscreen render) |
| `.html` | Interactive WebGL scene (requires `trame_vtk`; falls back to `.png`) |
| `.vtu`, `.vtk` | VTK unstructured grid written directly, no render window opened |
| `None` | Interactive PyVista render window (requires a display) |

### Scalar and colouring

| Variable | Default | Description |
|---|---|---|
| `PLOT3D_SCALAR` | `"log10_resistivity"` | Scalar to colour slices by: `"log10_resistivity"` or `"resistivity"` |
| `PLOT3D_CLIM` | `[0., 3.]` | `[vmin, vmax]` for the scalar; `None` = PyVista auto |
| `PLOT3D_CMAP` | `"turbo_r"` | Matplotlib / PyVista colormap name |

### Axis-aligned slices

| Variable | Default | Description |
|---|---|---|
| `PLOT3D_SLICE_X` | `[0.0]` | YZ cutting planes — N-S sections at these easting values |
| `PLOT3D_SLICE_Y` | `[0.0]` | XZ cutting planes — E-W sections at these northing values |
| `PLOT3D_SLICE_Z` | `[5000., 15000.]` | XY cutting planes — horizontal maps at these depths |

All positions are model-local metres (z positive-down).
Empty list or `None` → no slices along that axis.

### Oblique plane slices

```python
PLOT3D_SLICE_PLANES = [
    dict(origin=[0., 0., 8000.], normal=[1., 1., 0.]),   # NE-trending vertical
    dict(origin=[5000., 0., 0.], normal=[0., 0., 1.]),   # horizontal at z=0
]
```

| Key | Type | Description |
|---|---|---|
| `"origin"` | `[x, y, z]` | Any point on the plane (model-local m) |
| `"normal"` | `[nx, ny, nz]` | Plane normal vector (need not be unit length) |

Empty list or `None` → no oblique slices.

### Iso-surfaces

| Variable | Default | Description |
|---|---|---|
| `PLOT3D_ISOVALUES` | `[1., 2., 3.]` | Iso-surface levels in `PLOT3D_SCALAR` units |
| `PLOT3D_ISO_OPACITY` | `0.35` | Opacity (0 = transparent, 1 = solid) |

For `log10_resistivity`: `1.0 = 10 Ω·m`, `2.0 = 100 Ω·m`, `3.0 = 1000 Ω·m`.
Empty list or `None` → no iso-surfaces.

### Window

| Variable | Default | Description |
|---|---|---|
| `PLOT3D_WINDOW_SIZE` | `[1600, 900]` | Render window size in pixels `[width, height]` |

---

## VTU export for ParaView

Set `PLOT3D_VTU_FILE` to a path ending in `.vtu` to export the full clipped
unstructured grid as VTK XML.  This file can be opened directly in ParaView
for interactive slicing, iso-surfacing, and publication-quality rendering,
and is suitable for Zenodo data deposits.

```python
PLOT3D_VTU_FILE = WORK_DIR + "resistivity_block_iter17.vtu"
```

Setting `PLOT3D_FILE` to a `.vtu` / `.vtk` path is equivalent to VTU-only
mode: the grid is written but no PyVista render window is opened.

---

## Dependencies

| Package | Role |
|---|---|
| `femtic` (Py4MTX) | `read_site_dat` (origin estimation) |
| `femtic_viz` (Py4MTX) | `plot_model_3d`, `read_femtic_mesh`, `read_resistivity_block` |
| `util` (Py4MTX) | `utm_zone_from_latlon`, `utm_to_latlon_zn`, `print_title` |
| `pyvista` | 3-D rendering and VTU export |
| `numpy` | Array operations |
| `trame_vtk` | Optional — required for HTML export only |

---

## Provenance

| Date | Author | Note |
|---|---|---|
| 2026-05-13 | vrath / Claude Sonnet 4.6 | Created as step (5) inside `femtic_mod_plot.py`; `PLOT3D` config block |
| 2026-05-26 | Claude Sonnet 4.6 | `plot_model_3d` moved into `femtic_viz.py`; `PLOT3D_VTU_FILE` added; `PLOT3D_FILE` default changed to `.png` |
| 2026-05-27 | vrath / Claude Sonnet 4.6 | `PLOT_XLIM/YLIM/ZLIM` passed for spatial clipping of VTU export and scene |
| 2026-05-31 | vrath / Claude Sonnet 4.6 | Origin estimation before UTM zone derivation; hard-coded `UTM_ORIGIN_*` set to `None` |
| 2026-06-03 | Claude Sonnet 4.6 | **Split** from `femtic_mod_plot.py` → `femtic_mod_plot_3d.py` (this script) + `femtic_mod_plot_slice.py` |
