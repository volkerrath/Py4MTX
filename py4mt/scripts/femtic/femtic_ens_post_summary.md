# Parameter summary: femtic_ens_post

- **Script path:** `/home/vrath/Py4MTX/py4mt/scripts/femtic/femtic_ens_post.py`
- **Run date/time:** 2026-09-25 19:16:42

## User-set parameters

| Parameter | Value |
|---|---|
| `BOOTSTRAP_N` | `500` |
| `BOOTSTRAP_SEED` | `None` |
| `BOOTSTRAP_VAR` | `True` |
| `COMPUTE_COV` | `True` |
| `COMPUTE_SENS` | `True` |
| `COMPUTE_SIMRC` | `True` |
| `COV_METHOD` | `'low_rank'` |
| `ENSEMBLE_DIR` | `'/media/vrath/LargeBack/Ensembles/annecy2026/ensemble_gst_2/'` |
| `ENSEMBLE_NAME` | `'annecy_rnd_2_'` |
| `ENSEMBLE_PREFIX` | `'annecy_rnd_2'` |
| `ENSEMBLE_RESULTS` | `'/media/vrath/LargeBack/Ensembles/annecy2026/ensemble_gst_2/ANNECY_RND_2_results.npz'` |
| `MOD_AIR_BGCOLOR` | `None` |
| `MOD_AIR_COLOR` | `'whitesmoke'` |
| `MOD_AIR_RHO` | `1000000000.0` |
| `MOD_ALPHA_FADE_WIDTH` | `1.0` |
| `MOD_ALPHA_MODE` | `'blank'` |
| `MOD_ALPHA_QC` | `True` |
| `MOD_ALPHA_SOURCE` | `'sens_mean_an > -3.'` |
| `MOD_CLIM` | `[0.0, 4.0]` |
| `MOD_CMAP` | `'jet_r'` |
| `MOD_DEPTH_KM` | `True` |
| `MOD_DISPLAY_COORDS` | `'latlon'` |
| `MOD_DPI` | `400` |
| `MOD_EQUAL_ASPECT` | `True` |
| `MOD_FIGSIZE` | `None` |
| `MOD_HORIZ_KM` | `True` |
| `MOD_LABEL_FONTSIZE` | `16` |
| `MOD_MAP_MARKERS` | `[]` |
| `MOD_MESH` | `'/media/vrath/LargeBack/Ensembles/annecy2026/ensemble_gst_2/templates/mesh.dat'` |
| `MOD_NCOLS` | `3` |
| `MOD_NROWS` | `4` |
| `MOD_OCEAN` | `None` |
| `MOD_OCEAN_COLOR` | `'lightgrey'` |
| `MOD_OCEAN_RHO` | `0.25` |
| `MOD_ORIGIN_METHOD` | `'box'` |
| `MOD_PANEL_HEIGHT` | `18.0` |
| `MOD_PANEL_WIDTH` | `None` |
| `MOD_PLOT_FORMAT` | `['pdf', 'jpg']` |
| `MOD_PLOT_SITES_MAPS` | `True` |
| `MOD_PLOT_SITES_SLICES` | `False` |
| `MOD_PROJECTION_DIST` | `1.0` |
| `MOD_QC` | `True` |
| `MOD_QC_FILE` | `'/media/vrath/LargeBack/Ensembles/annecy2026/ensemble_gst_2/annecy_rnd_2_best'` |
| `MOD_ROI_AUTO` | `True` |
| `MOD_ROI_PAD_XY` | `4.0` |
| `MOD_ROI_ZLIM` | `[-1.0, 7.0]` |
| `MOD_SHOW_IN_SPYDER` | `True` |
| `MOD_SHOW_MODEL_CENTRE` | `True` |
| `MOD_SITE_DAT` | `'/media/vrath/LargeBack/Ensembles/annecy2026/ensemble_gst_2/templates/site.dat'` |
| `MOD_SITE_MARKER` | `{'marker': 'v', 'color': 'black', 'ms': 8, 'zorder': 10, 'label': None}` |
| `MOD_SITE_MARKER_SLICES` | `None` |
| `MOD_SITE_NAMES` | `None` |
| `MOD_SITE_NUMBER` | `None` |
| `MOD_SLICES` | `[{'kind': 'map', 'z0': -0.25}, {'kind': 'map', 'z0': 0.0}, {'kind': 'map', 'z0': 0.5}, {'kind': 'map', 'z0': 1.0}, {'kind': 'map', 'z0': 2.0}, {'kind': 'map', 'z0': 2.5}, {'kind': 'map', 'z0': 3.0}, {'kind': 'map', 'z0': 4.0}, {'kind': 'map', 'z0': 5.0}, {'kind': 'ns', 'x0': 0.0}, {'kind': 'ew', 'y0 ... (truncated)` |
| `MOD_STATS` | `True` |
| `MOD_STATS_DIR` | `'/media/vrath/LargeBack/Ensembles/annecy2026/ensemble_gst_2//stats_plots/'` |
| `MOD_STATS_STYLE` | `{'var': {'clim': [0.0, 0.3], 'label': 'var log10(rho)'}, 'err': {'clim': [0.0, 0.3], 'label': 'std log10(rho)'}, 'mad': {'clim': [0.0, 0.3], 'label': 'MAD log10(rho)'}, 'qdiff_15_9_84_1': {'clim': [0.0, 0.5], 'label': 'P84.1 - P15.9  log10(rho)'}, 'qdiff_2_3_97_7': {'clim': [0.0, 0.5], 'label': 'P97 ... (truncated)` |
| `MOD_STATS_WHAT` | `['avg', 'med', 'err', 'mad', 'p2_3', 'p15_9', 'p50', 'p84_1', 'p97_7', 'qdiff_15_9_84_1', 'qdiff_2_3_97_7', 'err_boot', 'sens_mean_raw', 'sens_mean_na', 'sens_cv_na', 'sens_mean_an', 'sens_cv_an', 'simrc_corr']` |
| `MOD_TICK_DECIMALS` | `2` |
| `MOD_TICK_FONTSIZE` | `16` |
| `MOD_UTM_ORIGIN_E` | `None` |
| `MOD_UTM_ORIGIN_LAT` | `None` |
| `MOD_UTM_ORIGIN_LON` | `None` |
| `MOD_UTM_ORIGIN_N` | `None` |
| `MOD_UTM_ZONE_OVERRIDE` | `None` |
| `MOD_XLIM` | `[-7.3315499999999885, 7.331550000000046]` |
| `MOD_YLIM` | `[-9.639599999999628, 9.63960000000056]` |
| `MOD_ZLIM` | `[-1.0, 7.0]` |
| `NRMS_MAX` | `1.5` |
| `OUT` | `True` |
| `P` | `'annecy_rnd_2'` |
| `PERCENTILES` | `[2.3, 15.9, 50.0, 84.1, 97.7]` |
| `PY4MTX_DATA` | `'/home/vrath/work/MT_Data/'` |
| `PY4MTX_ROOT` | `'/home/vrath/Py4MTX/'` |
| `QDIFF_PAIRS` | `[(15.9, 84.1), (2.3, 97.7)]` |
| `SENS_ERROR_KEY` | `'data_errors'` |
| `SENS_H5_GROUP` | `None` |
| `SENS_H5_PATTERN` | `'results_iter{numit}.h5'` |
| `SENS_JACOBIAN_KEY` | `'jacobian'` |
| `SENS_KIND` | `'volume_normalised'` |
| `SIMRC_CHUNK` | `200` |
| `SIMRC_DATA_KIND` | `None` |
| `SIMRC_RESULT_FILES` | `{'imp': 'result_MT.txt', 'vtf': 'result_VTF.txt'}` |
| `SIMRC_SITE_FILE` | `'/media/vrath/LargeBack/Ensembles/annecy2026/ensemble_gst_2/templates/site.dat'` |
| `SPARSE_THRESH` | `1e-08` |
| `SPARSIFY` | `True` |
