# femtic_mod_plot_3d.py - README

PyVista 3-D rendering and VTK/VTU export of single FEMTIC resistivity models.

Settings are named as in the other plot scripts (`MOD_*`, lengths in km), and
several runs can be given, as in `femtic_mod_misfit.py`: every entry is a model
of its own and gets its own scene and grid file. The scene itself is built by
`femtic_viz.plot_model_3d()`.

Sister scripts:
[`femtic_mod_plot_slice.py`](femtic_mod_plot_slice_readme.md),
[`femtic_mod_plot_bh.py`](femtic_mod_plot_bh_readme.md).

Requires [PyVista](https://docs.pyvista.org/)
(`conda install -c conda-forge pyvista`).

Paths, colours and region of interest default to the values of
`femtic_ens_plot_slices.py` (2026-11-26).

AI-generated documentation (Claude Sonnet 5.5, Anthropic, 2026-11-26); review
before use. The 2026-11-26 version was run with a stand-in for PyVista only; no
scene has been rendered with it.

---

## Usage

```
python femtic_mod_plot_3d.py                          # RUN_PATHS
python femtic_mod_plot_3d.py kra_rto_0 kra_rto_1      # two run directories
python femtic_mod_plot_3d.py "kra*/"                  # quote globs
python femtic_mod_plot_3d.py -w /data/ann ann_rto_orig-err_0/resistivity_block_iter10.dat
```

## Runs

| Setting | Default | Meaning |
|---|---|---|
| `WORK_DIR` | `.../krafla2026/krafla_fine/krafla_ens_fine/` | base of relative paths; `None` / `""` = current directory |
| `RUN_PATHS` | `["kra*/"]` | run directories, block files, `results_iterN.h5` files or globs |
| `ITERATION` | `"best"` | for run directories: `"best"`, `"last"`, `N`, or a list |
| `CNV_FILE` | `["femtic.cnv", "FEMTIC-DABIC.cnv"]` | convergence file; first name that exists |
| `MOD_MESH` | `"templates/mesh.dat"` | per run: run directory first, then `WORK_DIR` |
| `MOD_MESH_SWAP_XY` | `True` | x = easting, y = northing |
| `MOD_OCEAN` | `None` | `None` = infer from the block file |
| `MOD_OCEAN_RHO`, `MOD_AIR_RHO` | `0.25`, `1e9` | region 1 gets `MOD_OCEAN_RHO` only if it is the ocean |

Return code: `0` all models found were done; `1` one failed, none was found,
PyVista is missing, or no output is selected; `2` `WORK_DIR` missing.

## Output

| Setting | Default | Meaning |
|---|---|---|
| `MOD_PLOT_FORMAT` | `["png"]` | `"png"`, `"jpg"`, `"tif"`, `"bmp"`, `"html"` (needs `trame_vtk`, else png); `None` = interactive window per model; `[]` = no rendering |
| `MOD_3D_VTU` | `"vtu"` | grid file: `"vtu"`, `"vtk"`, `None` |
| `OUT_DIR`, `OUT_PREFIX`, `OUT_PREFIX_DIR` | `None`, `None`, `True` | as in the slice script |
| `MOD_3D_FILE` | `None` | explicit path without extension (one model only) |

Files are `<dir>/[<OUT_PREFIX>_]<label>_<tag>_3d.<ext>`. Every image format
builds and renders the scene again. The parameter summary
`femtic_mod_plot_3d_summary.md` is written to `WORK_DIR`.

## Scene

Coordinates are model-local **km**; z is positive down.

| Setting | Default | Meaning |
|---|---|---|
| `MOD_XLIM`, `MOD_YLIM`, `MOD_ZLIM` | `None` | clip box (grid file and scene); `None` = none |
| `MOD_ROI_AUTO`, `MOD_ROI_PAD_XY`, `MOD_ROI_ZLIM` | `True`, `2.0`, `[-1, 6]` | clip box from the sites of each run (x / y: site box + padding; z as given); without sites the limits above |
| `MOD_SITE_DAT` | `"templates/site.dat"` | site list, used by `MOD_ROI_AUTO` only |
| `MOD_ORIGIN_METHOD`, `MOD_UTM_ORIGIN_LAT/LON/E/N` (E / N in UTM km), `MOD_UTM_ZONE_OVERRIDE` | `"box"`, `None`, `None` | origin of the mesh centre, as in the slice script |
| `MOD_3D_SCALAR` | `"log10_resistivity"` | or `"resistivity"` |
| `MOD_CLIM` | `[0, 4]` | log10(Ohm.m), also for `"resistivity"`; `None` = auto |
| `MOD_CMAP` | `"jet_r"` | colormap |
| `MOD_3D_SLICE_X` | `[0.0]` | N-S sections at these eastings |
| `MOD_3D_SLICE_Y` | `[0.0]` | E-W sections at these northings |
| `MOD_3D_SLICE_Z` | `[1.0, 3.0]` | maps at these depths |
| `MOD_3D_SLICE_PLANES` | `[]` | `dict(origin=[x, y, z], normal=[nx, ny, nz])` |
| `MOD_3D_ISOVALUES` | `[1, 2, 3]` | log10(Ohm.m) |
| `MOD_3D_ISO_OPACITY` | `0.35` | |
| `MOD_3D_WINDOW_SIZE` | `[1600, 900]` | pixels |
| `MOD_3D_SCREENSHOT_SCALE` | `2` | screenshot size = window size x scale |

## Migration from the earlier version (2026-11-26)

| Old | New |
|---|---|
| `MODEL_FILE` | `RUN_PATHS`, `ITERATION` |
| `MESH_FILE` | `MOD_MESH` |
| `OCEAN`, `AIR_RHO`, `OCEAN_RHO` | `MOD_OCEAN`, `MOD_AIR_RHO`, `MOD_OCEAN_RHO` |
| `PLOT_XLIM/YLIM/ZLIM` (m) | `MOD_XLIM/YLIM/ZLIM` (km) |
| `PLOT3D_FILE` | `MOD_PLOT_FORMAT`, `MOD_3D_FILE` |
| `PLOT3D_VTU_FILE` | `MOD_3D_VTU` |
| `PLOT3D_CLIM`, `PLOT3D_CMAP` | `MOD_CLIM`, `MOD_CMAP` |
| `PLOT3D_SLICE_*` (m) | `MOD_3D_SLICE_*` (km) |
| `PLOT3D_<name>` | `MOD_3D_<name>` |
| `PLOT_DPI` (unused) | `MOD_3D_SCREENSHOT_SCALE` |
| `SITE_DAT`, `UTM_*`, `ORIGIN_METHOD` | `MOD_SITE_DAT`, `MOD_UTM_*`, `MOD_ORIGIN_METHOD` (for the region of interest) |

## Dependencies

`femtic.py`, `femtic_viz.py`, `util.py` (Py4MTX), `numpy`, `pyvista`;
`trame_vtk` for HTML.

---

## Provenance

| Date | Author | Note |
|---|---|---|
| 2026-05-13 | vrath / Claude Sonnet 4.6 | Created as step (5) inside `femtic_mod_plot.py`; `PLOT3D` config block |
| 2026-05-26 | Claude Sonnet 4.6 | `plot_model_3d` moved into `femtic_viz.py`; `PLOT3D_VTU_FILE` added; `PLOT3D_FILE` default changed to `.png` |
| 2026-05-27 | vrath / Claude Sonnet 4.6 | `PLOT_XLIM/YLIM/ZLIM` passed for spatial clipping of VTU export and scene |
| 2026-05-31 | vrath / Claude Sonnet 4.6 | Origin estimation before UTM zone derivation; hard-coded `UTM_ORIGIN_*` set to `None` |
| 2026-06-03 | Claude Sonnet 4.6 | **Split** from `femtic_mod_plot.py` → `femtic_mod_plot_3d.py` (this script) + `femtic_mod_plot_slice.py` |
| 2026-08-13 | Claude Sonnet 5 (Anthropic) | Added `femtic_mod_plot_3d_summary.md` output at end of run: writes user-set (UPPERCASE) parameters, script path, and run date/time |
| 2026-10-28 | Claude Sonnet 5.5 (Anthropic) | The parameter summary `femtic_mod_plot_3d_summary.md` is now written to `WORK_DIR` (the working directory set in the script) instead of next to the script: `utl.write_param_summary(fname, out_dir=WORK_DIR)`. Needs the `util.py` of the same date. AI-generated; review before production use. |
| 2026-11-26 | Claude Sonnet 5.5 (Anthropic) | Settings named as in the other plot scripts (`MOD_*`, km) and several runs as in `femtic_mod_misfit.py` (`RUN_PATHS`, command line, `-w`, `ITERATION`); `OUT_DIR` / `OUT_PREFIX`, `MOD_PLOT_FORMAT`, `MOD_3D_VTU`, `MOD_3D_FILE`. UTM origin and `SITE_DAT` settings removed (unused for the scene). Fixes: `MOD_MESH_SWAP_XY` (default `True`) now sets the x/y orientation; region 1 gets the ocean resistivity only if the model has an ocean. This readme was rewritten. Run with a PyVista stand-in only. AI-generated; review before production use. |
| 2026-11-26c | Claude Sonnet 5.5 (Anthropic) | Defaults as in `femtic_ens_plot_slices.py` (`WORK_DIR`, `RUN_PATHS = ["kra*/"]`, mesh in `templates/`, `jet_r` [0, 4], no fixed clip box); new `MOD_ROI_*`, `MOD_SITE_DAT`, `MOD_UTM_*`, `MOD_ORIGIN_METHOD`; `MOD_3D_SLICE_Z` [1, 3] km. Run with a PyVista stand-in only. AI-generated; review before production use. |
| 2026-11-26d | Claude Sonnet 5.5 (Anthropic) | All user settings are km, UTM coordinates too: `MOD_UTM_ORIGIN_E / _N` are UTM km (before: m). A value that looks like metres stops with an error (`fem.KmUnitError`); smaller values given in metres cannot be recognised and are read as km. Needs `femtic.py` of the same date. Tested on synthetic data only: same output as before for the same positions. AI-generated; review before production use. |
