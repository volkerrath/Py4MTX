# Parameter summary: femtic_gst_prep

- **Script path:** `/home/vrath/Py4MTX/py4mt/scripts/femtic/femtic_gst_prep.py`
- **Run date/time:** 2026-09-05 10:20:19

## User-set parameters

| Parameter | Value |
|---|---|
| `COPY_LIST` | `['resistivity_block_iter0.dat']` |
| `DAT_ALPHA_ORIG` | `0.5` |
| `DAT_ALPHA_PERT` | `1.0` |
| `DAT_COMPS` | `['xy,yx', 'xy,yx']` |
| `DAT_COMP_MARKERS` | `None` |
| `DAT_ERROR_STYLE_ORIG` | `'bar'` |
| `DAT_ERROR_STYLE_PERT` | `'shade'` |
| `DAT_MARKERSIZE` | `3.0` |
| `DAT_MARKEVERY` | `2` |
| `DAT_PERLIMS` | `(0.0001, 10000.0)` |
| `DAT_PHSLIMS` | `(-180.0, 180.0)` |
| `DAT_PTLIMS` | `None` |
| `DAT_RHOLIMS` | `None` |
| `DAT_SHOW_ERRORS_ORIG` | `False` |
| `DAT_SHOW_ERRORS_PERT` | `True` |
| `DAT_VTFLIMS` | `(-1.0, 1.0)` |
| `DAT_WHAT` | `['rho', 'phase']` |
| `ENSEMBLE_DIR` | `'/media/vrath/LargeBack/Ensembles/annecy2026/ensemble_gst_4/'` |
| `ENSEMBLE_NAME` | `'annecy_rnd_4_'` |
| `ENS_CLIM` | `[0.0, 4.0]` |
| `ENS_CMAP` | `'turbo_r'` |
| `ENS_LABEL_FONTSIZE` | `8` |
| `ENS_OCEAN_COLOR` | `'lightgrey'` |
| `ENS_PER_MEMBER` | `False` |
| `ENS_PLOT_DPI` | `300` |
| `ENS_PLOT_FILE` | `'/media/vrath/LargeBack/Ensembles/annecy2026/ensemble_gst_4//plots/gst_ensemble_slices.pdf'` |
| `ENS_SLICES` | `[{'kind': 'map', 'z0': 0.0}, {'kind': 'map', 'z0': 1.0}, {'kind': 'ns', 'x0': 0.0}, {'kind': 'ew', 'y0': 0.0}]` |
| `ENS_STAT_ROWS` | `['mean', 'std']` |
| `ENS_TICK_FONTSIZE` | `8` |
| `ENS_XLIM` | `[-15.0, 15.0]` |
| `ENS_YLIM` | `[-15.0, 15.0]` |
| `ENS_ZLIM` | `[-1.0, 15.0]` |
| `ERRORS` | `[[0.25, 0.1, 0.1, 0.25, 0.25, 0.1, 0.1, 0.25], [0.05, 0.05, 0.05, 0.05], [0.5, 0.2, 0.2, 0.5]]` |
| `FROM_TO` | `array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63])` |
| `LINK_LIST` | `['control.dat', 'mesh.dat', 'referencemodel.dat', 'observe.dat', 'distortion_iter0.dat', 'site.dat', 'run_femtic_dias.sh', 'run_femtic_kraken.sh']` |
| `MOD_AIR_BGCOLOR` | `None` |
| `MOD_AIR_COLOR` | `'whitesmoke'` |
| `MOD_AIR_RHO` | `1000000000.0` |
| `MOD_ALPHA_BLANK_THRESH` | `0.0` |
| `MOD_ALPHA_FILE` | `None` |
| `MOD_ALPHA_MODE` | `'fade'` |
| `MOD_CLIM` | `[0.0, 4.0]` |
| `MOD_CMAP` | `'turbo_r'` |
| `MOD_DEPTH_KM` | `True` |
| `MOD_DISPLAY_COORDS` | `'latlon'` |
| `MOD_DPI` | `200` |
| `MOD_EQUAL_ASPECT` | `True` |
| `MOD_FIGSIZE` | `None` |
| `MOD_HORIZ_KM` | `True` |
| `MOD_LABEL_FONTSIZE` | `8` |
| `MOD_LOG_RHO_MAX` | `4.0` |
| `MOD_LOG_RHO_MIN` | `0.0` |
| `MOD_MAP_MARKERS` | `[]` |
| `MOD_MESH` | `'/media/vrath/LargeBack/Ensembles/annecy2026/ensemble_gst_4//templates/mesh.dat'` |
| `MOD_NCOLS` | `2` |
| `MOD_NROWS` | `3` |
| `MOD_N_PP` | `100` |
| `MOD_OCEAN` | `None` |
| `MOD_OCEAN_COLOR` | `'lightgrey'` |
| `MOD_OCEAN_RHO` | `0.25` |
| `MOD_ORIGIN_METHOD` | `'box'` |
| `MOD_OUTPUT_TARGET` | `'both'` |
| `MOD_PANEL_HEIGHT` | `18.0` |
| `MOD_PANEL_WIDTH` | `None` |
| `MOD_PILOT_POINTS_FILE` | `None` |
| `MOD_PLOT_SITES_MAPS` | `True` |
| `MOD_PLOT_SITES_SLICES` | `True` |
| `MOD_PP_BBOX` | `None` |
| `MOD_PP_COORDS` | `None` |
| `MOD_PP_EXTREMA_K` | `33` |
| `MOD_PP_EXTREMA_WHICH` | `'both'` |
| `MOD_PP_MODE` | `'random'` |
| `MOD_PP_ROI` | `None` |
| `MOD_PP_VALUE_DELTA` | `1.0` |
| `MOD_PP_VALUE_MODE` | `'reference'` |
| `MOD_PROJECTION_DIST` | `1.0` |
| `MOD_REF` | `'/media/vrath/LargeBack/Ensembles/annecy2026/ensemble_gst_4//templates/referencemodel.dat'` |
| `MOD_REFERENCE_FILE` | `'referencemodel.dat'` |
| `MOD_REF_BASE` | `'referencemodel.dat'` |
| `MOD_RESISTIVITY_FILE` | `'resistivity_block_iter0.dat'` |
| `MOD_SAVE_PILOT_POINTS` | `True` |
| `MOD_SHOW_IN_SPYDER` | `True` |
| `MOD_SITE_DAT` | `'/media/vrath/LargeBack/Ensembles/annecy2026/ensemble_gst_4//templates/site.dat'` |
| `MOD_SITE_MARKER` | `{'marker': 'v', 'color': 'black', 'ms': 4, 'zorder': 10, 'label': None}` |
| `MOD_SITE_MARKER_SLICES` | `{'marker': 'v', 'color': 'black', 'ms': 4, 'zorder': 10, 'label': None}` |
| `MOD_SITE_NAMES` | `None` |
| `MOD_SITE_NUMBER` | `None` |
| `MOD_SLICES` | `[{'kind': 'map', 'z0': 0.0}, {'kind': 'map', 'z0': 1.0}, {'kind': 'ns', 'x0': 0.0}, {'kind': 'ew', 'y0': 0.0}]` |
| `MOD_TICK_FONTSIZE` | `7` |
| `MOD_UTM_ORIGIN_E` | `None` |
| `MOD_UTM_ORIGIN_LAT` | `None` |
| `MOD_UTM_ORIGIN_LON` | `None` |
| `MOD_UTM_ORIGIN_N` | `None` |
| `MOD_UTM_ZONE_OVERRIDE` | `None` |
| `MOD_VARIO_ANGLES` | `None` |
| `MOD_VARIO_MODEL` | `'Spherical'` |
| `MOD_VARIO_NUGGET` | `0.01` |
| `MOD_VARIO_RANGE` | `(8.0, 4.0)` |
| `MOD_VARIO_SILL` | `0.5` |
| `MOD_XLIM` | `[-15.0, 15.0]` |
| `MOD_YLIM` | `[-15.0, 15.0]` |
| `MOD_ZLIM` | `[-1.0, 15.0]` |
| `N_SAMPLES` | `64` |
| `N_THREADS` | `'32'` |
| `OUT` | `True` |
| `PERTURB_DAT` | `False` |
| `PERTURB_MOD` | `True` |
| `PLOT_DATA` | `False` |
| `PLOT_DIR` | `'/media/vrath/LargeBack/Ensembles/annecy2026/ensemble_gst_4//plots/'` |
| `PLOT_MODEL` | `True` |
| `PLOT_ONLY` | `False` |
| `PLOT_SLICES_ENS` | `False` |
| `PLOT_SLICES_QC` | `False` |
| `PLOT_STR` | `''` |
| `PY4MTX_DATA` | `'/home/vrath/work/MT_Data/'` |
| `PY4MTX_ROOT` | `'/home/vrath/Py4MTX/'` |
| `RANDOM_SEED` | `None` |
| `RELATIVE_LINKS` | `True` |
| `RESET_ERRORS` | `True` |
| `TEMPLATES` | `'/media/vrath/LargeBack/Ensembles/annecy2026/ensemble_gst_4//templates/'` |
| `VIZ_N_SAMPLES` | `64` |
| `VIZ_N_SITES` | `10` |
| `VIZ_SAMPLES` | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]` |
