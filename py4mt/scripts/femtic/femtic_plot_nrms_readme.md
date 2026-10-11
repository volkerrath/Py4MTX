# femtic_plot_nrms.py

Plot per-site normalised RMS (nRMS) data misfit from a FEMTIC run as
colour-coded glyphs on a map, optionally over an SRTM-derived hillshade
basemap. Modeled on `femtic_plot_distort.py` -- same CONFIG-block style,
`model`/`utm`/`latlon` coordinate choice, SRTM basemap, and multi-format
output (`OUTPUT_BASENAME` + `OUTPUT_FORMATS`).

---

## Data flow

1. **Which iteration?** The best (smallest-nRMS) iteration is identified
   from `femtic.cnv` via `fem.get_nrms(FEMTIC_DIR)` -- this is FEMTIC's
   own authoritative whole-run nRMS trace (one row per iteration); nothing
   in this script recomputes or second-guesses it.
2. **Per-site breakdown for that iteration.** `femtic.cnv` has no per-site
   information at all, only one row per iteration, so the actual per-site
   residuals come from `results_iter<N>.h5` (`N` = the iteration selected in
   step 1), read via `femtic.read_results_data`.
3. **Per-site nRMS:**

   ```
   nRMS_site = sqrt( mean_i[ ((obs_i - cal_i) / err_i)**2 ] )
   ```

   pooled over every frequency, component, and data type
   (`STATION_TYPES`) recorded for that site in the HDF5 file. Real and
   imaginary parts count separately; the imaginary part is skipped for
   the real-valued types `APP_RES_AND_PHS`, `NMT2_APP_RES_AND_PHS`, `PT`.

A well-fit site should sit near `nRMS = 1`; the default colour map
(`RdBu_r`) is diverging and centred there via
`matplotlib.colors.TwoSlopeNorm`. Marker size scales with `nRMS` by
default (`GLYPH_SIZE_MODE = "scaled"`). An optional `NRMS_FLAG_THRESHOLD`
draws a highlighted ring around outlier sites. The plot title shows the
selected iteration and its overall nRMS from `femtic.cnv` alongside the
per-site map.

**This script went through two wrong assumptions before landing here** --
worth knowing since it explains some of the defensive checks below:
- First version recomputed per-site nRMS from `result_MT.txt`-style
  files, with no tie to a specific iteration.
- Corrected to "nRMS comes from `femtic.cnv`" -- but `femtic.cnv` is a
  whole-run, per-*iteration* trace with no per-site breakdown at all, so
  that alone can't drive a per-site map.
- Final, confirmed design (this one): `femtic.cnv` picks *which
  iteration*; `results_iter<N>.h5` supplies the per-site residuals *for*
  that iteration.

---

## Installation / requirements

- `numpy`, `matplotlib` (required)
- `h5py` (required, by `femtic.read_results_data`)
- `rasterio`, and either the `elevation` package (auto-download, requires
  GDAL + internet) or a pre-downloaded GeoTIFF via `SRTM_LOCAL_FILE`
  (only required when `BASEMAP_ENABLE = True`)
- `femtic.py`, `util.py` (py4mt) importable on `PYTHONPATH` / same directory

A missing/failed SRTM basemap only disables the basemap (with a warning)
-- it never blocks the glyph plot.

---

## Accuracy note -- please read

**Migration, 2026-10-11.** The script originally read per-station-type
groups of `data_iter<N>.h5` through a separately supplied
`femtic_hdf5_readers.py`, which is no longer part of the code base. It now
uses `femtic.read_results_data` (flat structured rows of a
`results_iter<N>.h5`: `datatype`, `site_id`, `re_val`/`im_val`,
`re_err`/`im_err`, `cal_re`/`cal_im`; older standalone `data_iter<N>.h5`
files are accepted by that reader too). The pooling arithmetic was
re-checked on synthetic rows against hand-computed values; **the file
reading was not run against a real FEMTIC file** (h5py was not
installable where this was done).

- `_resolve_station_ids` is applied **per data type** and assumes the
  results file's `site_id` and `site.dat`'s `sitenum` use the same integer
  convention. It requires *every* unique `site_id` of that type to match a
  `sitenum` before accepting a mapping (tried as-is, then with a `+1`
  offset); a partial overlap is deliberately **not** accepted. A data type
  that cannot be mapped is skipped with a warning (the old code raised);
  if no type can be mapped the script raises. Note that `femtic_data_misfit.py`
  documents that VTF may carry its own `site_id` range (e.g. 1001...) --
  such types are skipped here rather than mis-assigned.
- Per-site nRMS values that look implausible most likely point at the
  `site_id` numbering, not at the arithmetic.

---

## Quick start

Edit the `CONFIG` block at the top of the script and run:

```bash
python femtic_plot_nrms.py
```

Or call it programmatically:

```python
import femtic_plot_nrms as fpn

fpn.plot_nrms_glyphs(
    femtic_dir=".",
    results_iter_template="results_iter{iter}.h5",
    site_dat="site.dat",
    display_coords="latlon",
)
```

---

## Inputs

| Input | Role |
|---|---|
| `FEMTIC_DIR` | Directory containing `femtic.cnv` and the `results_iter<N>.h5` files. |
| `RESULTS_ITER_TEMPLATE` | Filename template (relative to `FEMTIC_DIR`) for the per-iteration HDF5 results file (renamed from `DATA_ITER_TEMPLATE`, default now `results_iter{iter}.h5`); must contain the literal `{iter}` placeholder. |
| `STATION_TYPES` | `None` (pool every data type present, e.g. `MT` + `VTF` + `PT`; names as in `fem.RESULTS_H5_DATATYPE_NAMES`) or a list to restrict to specific types. |
| `SITE_DAT` | Sitelist CSV (`mt_make_sitelist.py` format), read via `fem.read_site_dat` for coordinates and matched to the results file's `site_id` (see accuracy note above). |

---

## Key CONFIG parameters

| Parameter | Meaning |
|---|---|
| `DISPLAY_COORDS` | `"model"` (model-local km), `"utm"` (absolute UTM km), or `"latlon"` (decimal degrees). |
| `UTM_ZONE`, `UTM_NORTHERN`, `UTM_ORIGIN_E/N` | Same role as in `femtic_plot_distort.py`. |
| `ERR_MIN` | `(obs, cal, err)` triples with `err <= ERR_MIN` are excluded from the misfit sum. |
| `GLYPH_SIZE_MODE` | `"fixed"` or `"scaled"` (size grows with `nRMS`, between `GLYPH_SIZE_MIN`/`MAX`). |
| `CMAP_NRMS`, `NRMS_REFERENCE` | Diverging colormap and its centre value (default `1.0`). `NRMS_VMIN`/`NRMS_VMAX` override the auto symmetric range. |
| `NRMS_FLAG_THRESHOLD` | `None` (off) or a cutoff above which sites get a highlighted ring and a legend entry. |
| `TITLE_SHOW_ITERATION` | Appends `"(iteration N, overall nRMS = X)"` (from `femtic.cnv`) to the plot title. |
| `BASEMAP_ENABLE`, `SRTM_PRODUCT`, `SRTM_LOCAL_FILE` | SRTM hillshade basemap controls; identical to `femtic_plot_distort.py`. |
| `OUTPUT_BASENAME`, `OUTPUT_FORMATS` | Output file stem and a list of formats, e.g. `["png"]` or `["pdf", "jpg"]`. |

---

## Verification performed

*(2026-10-11: items below about `femtic_hdf5_readers` stubs, `attrs["iteration"]`
and `DATA_ITER_TEMPLATE` describe the earlier reader and are kept as history.
After the migration only the pooling arithmetic and the `site_id` matching
were re-tested, on synthetic rows; see the accuracy note.)*

- `ast.parse()` syntax check.
- **Iteration selection**: a synthetic `femtic.cnv` with four iterations
  (nRMS `3.5, 2.1, 1.05, 1.4`) correctly selected iteration 2 (the
  smallest) via `fem.get_nrms`, and that iteration number was correctly
  substituted into `DATA_ITER_TEMPLATE` to build the HDF5 path actually
  requested (checked with an assertion inside the test's `read_data_hdf5`
  stub, so a wrong iteration number would have failed the test outright,
  not just produced a wrong plot).
- **Per-site nRMS arithmetic**: built synthetic `MT` (8-component, 2
  records/site) and `VTF` (4-component, 1 record/site) station-type
  blocks with known random `cal`/`err`/`obs` values (one deliberately
  grossly-misfit site among three), computed each site's expected pooled
  nRMS independently in the test script (plain numpy), and confirmed
  `_load_nrms_per_site` reproduced those exact values to full float
  precision (`0.657295`, `4.999105`, `1.056280`) when pooling both types.
- **`_resolve_station_ids`**: tested a full direct match, a full
  0-based-needs-`+1` match (correctly warned and corrected), and a
  deliberately partial-overlap case (`[1, 2, 99]` against valid sitenums
  `{1, 2, 3}`, which partially "matches" both as-is and with `+1`) --
  confirmed this correctly **raises** rather than silently accepting a
  coincidental partial match. This test caught and fixed a real bug in an
  earlier draft (an "any overlap" check that would have silently
  mis-mapped a genuinely 0-based station_id array without applying the
  `+1` correction, because 2 of 3 IDs happened to coincide with valid
  sitenums as-is).
- **Iteration-attribute cross-check**: a stubbed `data_iter2.h5` with a
  deliberately mismatched `attrs["iteration"] = 99` correctly triggered
  the mismatch warning while still proceeding with the file as given.
- The above used a hand-written stub standing in for
  `femtic_hdf5_readers.py` (matching its real `FemticStationData`/
  `FemticData` field names) because `h5py` could not be installed in the
  environment this script was written in (no network access) -- so what's
  verified is this script's own logic (iteration selection, residual
  pooling, station_id matching, plotting), **not** `femtic_hdf5_readers.py`
  reading a real `.h5` file, which remains exactly as unverified as that
  module's own readme already states.
- The SRTM download/reprojection path (`elevation` + `rasterio`) also
  could not be exercised (no network access) -- untested end-to-end, as
  with `femtic_plot_distort.py`.

---

## Provenance

| Date | Author | Change |
|---|---|---|
| 2026-09-13 | Claude Sonnet 5 (Anthropic) | Created (v1): per-site nRMS glyph map computed directly from `result_XXX.txt` residuals -- **superseded**, no tie to a specific FEMTIC iteration. |
| 2026-10-11 | Claude Sonnet 5.5 (Anthropic) | Migrated HDF5 input from the absent `femtic_hdf5_readers.read_data_hdf5` (`data_iter<N>.h5`) to `femtic.read_results_data` (`results_iter<N>.h5`); `DATA_ITER_TEMPLATE` -> `RESULTS_ITER_TEMPLATE`; imaginary parts now pooled (skipped for real-valued types); unmappable data types are skipped with a warning. Arithmetic checked on synthetic rows; file reading untested (no h5py). |
| 2026-09-13 | Claude Sonnet 5 (Anthropic) | Rewritten (v2): best iteration now selected from `femtic.cnv` via `fem.get_nrms`; per-site residuals read from that iteration's `data_iter<N>.h5` via `femtic_hdf5_readers.read_data_hdf5`. Added `_resolve_station_ids` (station_id/sitenum matching with a 0-based/1-based fallback, full-containment check only) and `TITLE_SHOW_ITERATION`. Verified against a hand-written `femtic_hdf5_readers` stub (see "Verification performed"); found and fixed a partial-overlap matching bug during that testing. AI-generated; the real `femtic_hdf5_readers.py` HDF5 schema itself remains unverified against a live file (inherited from that module's own readme). |

Author: Volker Rath (DIAS)
