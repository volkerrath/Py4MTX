# femtic_plot_sitemap.py

Site map of a FEMTIC run, or of a whole ensemble, on a DEM background. Every
site is an icon coloured by its normalized RMS misfit (nRMS), for frequency bands
and data components, or by other measures of data fit or the strength of its
galvanic distortion (see "Measures and distortion strength"). For an ensemble (`WORK_DIR` = the ensemble root with the
member directories and `templates/`) the colour is a statistic of the member
nRMS at the site (median by default; mean, min, max, a percentile or the std).
The DEM is the Copernicus DEM GLO-30 by default (GeoTIFF tiles that are
downloaded once and cached, read with `rasterio`); GLO-90, the AWS Terrain Tiles,
tiles of your own, a DEM file or no DEM are the alternatives. Profile lines (cross-section
lines, with swath, distance ticks and the sites in the swath) and other polylines
or polygons (faults, coast lines, caldera rims, ...) can be drawn on the maps.

Same layout as the other FEMTIC scripts: a CONFIG block of UPPERCASE settings
at the top of the script, `WORK_DIR` as the base of all relative paths,
`-w` / `-i` / `--dem` / `--runs` / `--stat` / `--measures` / `--profile` / `--line` on the command line, `settings()` /
`run()` for use from other code. For a plot of the one nRMS per site (all bands, all components) see
also `femtic_plot_nrms.py`; this script adds the bands x components selections,
the tables and the DEM sources.

Needs in `modules/`: `femtic_misfit.py` (with the modules it imports),
`misfit_measures.py`, `dem.py` (the DEM routines, see `dem_readme.md`),
`util.py` and `version.py`.

---

## Data flow

1. **Results.** `results_iter<N>.h5` from `WORK_DIR` (`RESULTS_PATH`,
   `ITERATION` = `"best"`, `"last"` or a number), found and read with
   `femtic_misfit.resolve_run` / `femtic_misfit.build_dataset`. If `RESULTS_PATH`
   holds no results file, but sub-directories do, these are the members of an
   ensemble (or name them with `RUN_PATHS`, see "Ensemble"); every member is read
   on its own, with its own observed data.
2. **Sites.** `site.dat` (name, lat, lon, elev, sitenum, easting, northing; comma
   or blank separated, `#` comments) from `WORK_DIR/templates` (`TEMPLATES_DIR`);
   if it is not there from `WORK_DIR` itself, then from a `templates` directory
   next to `WORK_DIR` (the layout of an ensemble: `<ens>/templates` and
   `<ens>/<member>`), or the path in `SITE_DAT`.
3. **Assignment of the site ids to the sites of `site.dat`.**
   - `SITE_MATCH = "coords"`: by position, with `femtic_misfit.match_sites_by_coords`.
     The results file holds x (north), y (east) in metres relative to the model
     origin, `site.dat` the UTM easting / northing; the shift between the two is
     found from the data (`SITE_MATCH_TOL` = largest distance of a match, in km).
   - `SITE_MATCH = "id"`: `site_id = sitenum + SITE_ID_OFFSET`.
   - `"auto"` (default): by position if the results file has positions, else by id.
   - All data types of one site (MT, VTF, ...) count for the site. HTF and NMT
     rows have no station position; they only count in the global values.
   - Several ids at one position and several `site.dat` entries there (two
     stations at one place) get one entry each, in the order of the numbers.
     Several ids and one entry (MT and VTF of one station) share the entry.
4. **nRMS** for every combination of a frequency band and a component selection:

   ```
   nRMS = sqrt( mean( ((obs - cal) / err)**2 ) )
   ```

   Real and imaginary parts count separately (the imaginary part is left out
   for real-valued data types: `APP_RES_AND_PHS`, `NMT2_APP_RES_AND_PHS`, `PT`);
   non-finite values and `err <= 0` are skipped. This is the nRMS of
   `femtic_misfit.site_nrms`; the global value for all bands and all components
   is `femtic_misfit.data_rms`, i.e. the RMS of the `.cnv` file of FEMTIC.
   "Global" in the tables is the value of all rows of the selection, also of rows
   that belong to no site; "n" is the number of values (real and imaginary parts).
5. **Ensemble** (more than one member): steps 3 and 4 are done for every member
   (the sites are matched member by member, by position or number, so the site ids
   may differ between members); the nRMS of a site is the statistic `ENS_STAT`
   over the members that have a value there, and the global nRMS of a selection
   is the same statistic over the global nRMS of the members.
6. **Profiles and lines** (if `PROFILES` / `LINES` are set, see "Profiles and
   lines"): converted to lon / lat, the sites projected on every profile.
7. **Basemap** (see "DEM source" below) and **figures**: one figure per page
   (`PAGE_BY`), one panel per selection, one colour scale for all pages; the
   lines and profiles are drawn on every panel.
8. **Tables**: nRMS and number of values per site and selection, a summary and,
   for an ensemble, one table of the members and one of the statistics; for the
   profiles one table of the profiles and one of the sites in their swath.

---

## Quick start

```
python femtic_plot_sitemap.py -w /data/krafla/krafla_rto_26          # settings in the script
python femtic_plot_sitemap.py -w /data/krafla/krafla_rto_26 --dem none   # without DEM, no download
python femtic_plot_sitemap.py -w /data/krafla/krafla_rto_26 --dem copernicus30   # Copernicus GLO-30 (the default)
python femtic_plot_sitemap.py -w /data/krafla/krafla_rto_26 --dem copernicus90   # GLO-90, for large maps
python femtic_plot_sitemap.py -w /data/krafla/krafla_rto_26 --dem skadi  # AWS Terrain Tiles, no rasterio needed
python femtic_plot_sitemap.py -w /data/krafla/krafla_rto_26 -i last

# other measures of fit, and the distortion strength
python femtic_plot_sitemap.py -w /data/krafla/krafla_rto_26 --measures nrms r_mae mean
python femtic_plot_sitemap.py -w /data/krafla/krafla_rto_26 --measures distortion

# a whole ensemble: WORK_DIR = the root with krafla_rto_NN/ and templates/
python femtic_plot_sitemap.py -w /data/krafla --dem none                  # median of all members
python femtic_plot_sitemap.py -w /data/krafla --stat std                  # spread of the members
python femtic_plot_sitemap.py -w /data/krafla --stat p90 --runs "krafla_rto_1*"

# profiles and polylines (lon lat); PROFILES / LINES in the script give more options
python femtic_plot_sitemap.py -w /data/krafla/krafla_rto_26 --profile "A-A'" -16.83 65.69 -16.73 65.73
python femtic_plot_sitemap.py -w /data/krafla/krafla_rto_26 --line gis/faults.txt --line gis/rim.geojson
python femtic_plot_sitemap.py -w /data/krafla/krafla_rto_26 --no-overlays   # the settings are ignored
```

or set `WORK_DIR` and the other settings in the CONFIG block, or from Python:

```python
import femtic_plot_sitemap as sm
cfg = sm.settings("/data/krafla/krafla_rto_26", BANDS="all", COMPONENTS=["all"], NRMS_VMAX=3.0)
sm.run(cfg)            # 0 = ok, 1 = data problem, 2 = wrong usage
```

Return codes: 0 success, 1 data could not be read or nothing could be plotted,
2 `WORK_DIR` missing or a setting that cannot work.

For the Krafla ensemble (`krafla_ens/krafla_rto_26`, site ids = sitenum + 1) the
defaults are enough: the positions are matched (100 of 100 sites, largest distance
0.7 m); `SITE_MATCH = "id"` would need `SITE_ID_OFFSET = 1`.

---

## Ensemble

`WORK_DIR` is the root of the ensemble, as made by the RTO scripts:

```
<ens>/templates/site.dat
<ens>/krafla_rto_01/results_iter*.h5
<ens>/krafla_rto_02/results_iter*.h5
...
```

With `RUN_PATHS = None` the sub-directories of `RESULTS_PATH` (default: `WORK_DIR`)
that contain `ITER_PATTERN` files are the members, in alphabetical order
(`templates/`, the output directory and the like have none and are skipped). With
`RUN_PATHS` (run directories, results files or globs, relative to `WORK_DIR`, e.g.
`["krafla_rto_*"]`; all entries together form one ensemble, in the order given) the
members are named explicitly. One member is a single run: the ensemble settings
have no effect then.

| Setting | Default | Meaning |
|---|---|---|
| `RUN_PATHS` | `None` | members: run directories, results files or globs; `None` = see above |
| `ENS_STAT` | `"median"` | statistic over the members at every site: `"mean"`, `"median"`, `"min"`, `"max"`, `"std"` (sample std, `ddof = 1`) or a percentile `"pNN"` (`"p90"`; linear interpolation as `numpy.percentile`) |
| `MEMBER_NRMS_MAX` | `None` | members whose nRMS of the iteration is larger are left out (and listed in `_members.csv`) |
| `CMAP_SPREAD` | `"YlOrRd"` | colour map of `ENS_STAT = "std"` |

- `ITERATION` applies to every member (`"best"` = the best iteration of each
  member). A member that cannot be read (or has no such iteration, no rows of the
  `DATATYPES`, no site that matches) is skipped with a warning and listed with the
  reason in `_members.csv`; only if no member can be read the run stops (return
  code 1).
- The nRMS of a member is that against its **own** observed data (the perturbed
  data of an RTO member), as in the `.cnv` file of the member, not against the
  mean of the ensemble; the RMS1 of the ensemble mean (as in `femtic_ens_misfit.py`)
  is not computed here.
- A selection (band, components) that exists in only some members is computed from
  those (`M` in the summary and in `_ens_stats.csv` gives the number); a site
  without data in a member is left out of the statistic for that member. `n` and
  `n_unassigned` are the median over the members.
- Level statistics (`mean`, `median`, `min`, `max`, `pNN`) use the colour scale of
  a single run: diverging, centred on `NRMS_REFERENCE`. `"std"` uses a linear
  sequential scale from 0 (`NRMS_VMIN` = `None` gives 0) with `CMAP_SPREAD`, no
  reference line. `NRMS_VMIN` / `NRMS_VMAX` always refer to the plotted values.
- The auto title reads "FEMTIC site nRMS, median of 20 members (iterations 5-8,
  global nRMS = 1.39)"; `TITLE` may contain `{iter}`, `{nrms}`, `{M}` (members)
  and `{stat}` (the statistic in words). The second line of a panel and the colour
  bar label name the statistic and the number of members.
- The site ids listed in the tables are those of all members used.
- The messages of the selections ("no data row, dropped") are printed once, not
  once per member; every member gets a progress line (every 10th for more than 20
  members).

---

## Selections

| Setting | Values |
|---|---|
| `BANDS` | `"all"` all frequencies only; `"decades"` all, and bands of `BAND_PER_DECADE` per decade (period or frequency, `BAND_AXIS`) with at least `BAND_MIN_N` data rows (the bands of `femtic_misfit`, after Baba, 2023); or a list `["all", (0.01, 1.0), (1.0, 100.0)]` with limits in s (`"period"`) or Hz (`"frequency"`), lower limit included, upper excluded |
| `COMPONENTS` | list of selections: `"all"`, a group of `COMPONENT_GROUPS` (`Z`, `Zoff`, `Zdiag`, `T`, `rho`, `phs`, `PT`), a component name (`"Zxy"`, any case, as written in the results file) or several joined by `+` (`"Zxy+Zyx"`); a selection without data is dropped with a message (the Krafla data have no VTF, so `"T"` is dropped) |
| `DATATYPES` | `None` = all data types, or a list of names, e.g. `["MT", "VTF"]` |

---

## Measures and distortion strength

`SITE_MEASURES` (command line `--measures`) is one name or a list; every entry
gets its own maps, tables and a section of the summary. Default: `"nrms"`.

```python
SITE_MEASURES = ["nrms", "r_mae", "distortion"]
MEASURE_SCALES = {"distortion": dict(vmax=0.5), "r_mae": dict(vmax=2.5)}
```

### Fit measures

Computed per site and for the whole data set from the error-normalized residuals
`r = (obs - cal) / err` of the rows of a selection (band x components), real and
imaginary parts together (the same values that enter the nRMS;
`modules/misfit_measures.py`, as in the misfit scripts). "Expected" = value for
Gaussian errors of the stated size.

| Name (alias) | Definition | Expected | Default colour scale |
|---|---|---|---|
| `"nrms"` (`rms`) | `sqrt(mean(r^2))` | 1 | `CMAP_NRMS` about `NRMS_REFERENCE`, `NRMS_VMIN` / `NRMS_VMAX` |
| `"r_mae"` (`rmae`) | `sqrt(pi/2) mean(\|r\|)` | 1 | as nrms |
| `"r_med"` (`rmed`) | `median(\|r\|) / 0.67449` | 1 | as nrms |
| `"std"` | standard deviation of `r` (1/n) | 1 | as nrms |
| `"q95"` | 95th percentile of `\|r\|` | 1.96 | `CMAP_NRMS` about 1.96, limits from the data |
| `"mean"` (`bias`) | `mean(r)` | 0 | `"RdBu_r"` about 0, symmetric |
| `"skew"` | skewness | 0 | `"RdBu_r"` about 0, symmetric |
| `"kurt"` | excess kurtosis | 0 | `"RdBu_r"` about 0, symmetric |

- With few values per site (narrow bands, `"T"`) `q95`, `skew` and `kurt` are very
  uncertain; `skew` needs 3 values, `kurt` 4 (else the icon is open).
- For the signed measures the largest absolute values are drawn on top, and the
  panel title has 3 decimals.
- `MEASURE_SCALES = {measure: dict(cmap=, reference=, vmin=, vmax=)}` replaces any
  of the four (`reference = None` = linear scale; `vmin` / `vmax` `None` = from
  the data). An entry for `"nrms"` changes the nRMS map only.
- `ENS_STAT = "std"` (spread over the members): all measures use `CMAP_SPREAD`,
  linear from 0, and `MEASURE_SCALES` is not used. nRMS behaves as before:
  `NRMS_VMIN` / `NRMS_VMAX` apply to the spread too, so set both to `None` for
  such maps (limits meant for the nRMS itself make a small spread one flat colour).

### Distortion strength

`"distortion"` maps one number per site (no bands or components):

```
|C - I| = sqrt(C'xx^2 + C'xy^2 + C'yx^2 + C'yy^2),   C' = C - I
```

C is the galvanic distortion matrix (`Z_obs = C Z_model`); FEMTIC with the type
`ESTIMATE_DISTORTION_MATRIX_DIFFERENCE` estimates C'. 0 means no distortion was
estimated at the site.

How to read it:

- The Frobenius norm is a summary chosen here (`femtic_distcorr.py` ranks sites by
  the same norm). It does not separate gain, twist and shear; the elements of C'
  of a single run are in `_distortion.csv`.
- The estimate depends on how strongly the inversion penalizes the distortion
  (the trade-off parameter of the distortion term; see the reference below), so
  compare runs with the same settings only.

Where C' comes from, in this order:

1. the `/distortion` group of the results file (needs `h5py`); its type is checked:
   type "no distortion" and the gain / rotation types give a warning and no map;
2. without a readable group, the text file `DISTORTION_FILE`
   (`"distortion_iter{iter}.dat"`) next to the results file. **Its type cannot be
   checked** (the gain / rotation types use the same columns with another
   meaning): for such runs set `DISTORTION_FILE = None`. The console and the
   summary say which source was used.

| Setting | Default | Meaning |
|---|---|---|
| `SITE_MEASURES` | `"nrms"` | names of the table above and / or `"distortion"` (`"dist"`) |
| `DISTORTION_FILE` | `"distortion_iter{iter}.dat"` | text file next to the results file; `None` = `/distortion` group only |
| `MEASURE_SCALES` | `{}` | colour scales per measure |

- The rows are matched to the sites by their site ids (those of the data rows; also
  when `DATATYPES` leaves out MT). Ids without a site are named in a warning;
  several ids of one site give the mean of their strengths. Only sites with data
  in the selected data types are mapped.
- A matrix held fixed in the inversion is included with its value (`fixed` /
  `n_fixed` in the table).
- Panel title: single run, the median of the sites; ensemble, the statistic
  `ENS_STAT` over the members of the site median of each member. The value of a
  site is `ENS_STAT` over the members that have distortion for it.
- If no member has distortion, the other measures are still mapped; with
  `SITE_MEASURES = "distortion"` alone the return code is 1.
- The auto title keeps the nRMS of the run; `TITLE` may contain `{measure}`.

Reference: Usui, Y., Ogawa, Y., Aizawa, K., Kanda, W., Hashimoto, T., Koyama, T.,
Yamaya, Y., Kagiyama, T. (2017): Three-dimensional resistivity structure of Asama
Volcano revealed by data-space magnetotelluric inversion using unstructured
tetrahedral elements. Geophysical Journal International 208(3), 1359-1372.
https://doi.org/10.1093/gji/ggw459 (inversion with distortion matrices, penalized
with a trade-off parameter; the norm used for the maps is not from this paper).

---

## Settings

Everything in the CONFIG block; relative paths refer to `WORK_DIR`. The tables
list the defaults; the CONFIG block of your copy may hold other run values
(for example `SITE_LABELS`, `DPI`, `OUTPUT_FORMATS`, the colour map and limits).
Sizes of figures are in **cm**, sizes of text, lines and symbols in pt.

**Input**

| Setting | Default | Meaning |
|---|---|---|
| `WORK_DIR` | `"./"` | base of all relative paths |
| `RESULTS_PATH` | `"."` | run directory (searched for `ITER_PATTERN`) or a results file |
| `ITERATION` | `"best"` | `"best"` (smallest nRMS), `"last"` or a number |
| `ITER_PATTERN` | `"results_iter*.h5"` | |
| `TEMPLATES_DIR` | `"templates"` | directory with `site.dat` |
| `SITE_DAT` | `"auto"` | `"auto"` (see Data flow) or a path |
| `SITE_MATCH` | `"auto"` | `"auto"`, `"coords"`, `"id"` |
| `SITE_MATCH_TOL` | `0.015` | [km] largest distance of a match by position (0.015 = 15 m); 1 or more is refused |
| `SITE_ID_OFFSET` | `0` | `site_id = sitenum + SITE_ID_OFFSET` for `"id"` |
| `DATATYPES` | `None` | see Selections |

**nRMS selections**: `BANDS`, `BAND_AXIS`, `BAND_PER_DECADE`, `BAND_MIN_N`,
`COMPONENTS`, `COMPONENT_GROUPS` (see Selections).

**Map**

| Setting | Default | Meaning |
|---|---|---|
| `MAP_EXTENT_MODE` | `"sites"` | `"sites"` (sites plus a margin), `"box"` or `"center"` |
| `MAP_MARGIN_FRAC`, `MAP_MARGIN_MIN_KM` | `0.15`, `2.0` | margin of `"sites"`: fraction of the extent of the sites, at least this [km] |
| `MAP_BOX` | `None` | `"box"`: `(lon_min, lon_max, lat_min, lat_max)` in degrees |
| `MAP_CENTER`, `MAP_SIZE_KM` | `None`, `(50, 50)` | `"center"`: `(lon, lat)` in degrees and `(width, height)` in km |
| `DISPLAY_COORDS` | `"latlon"` | `"latlon"` (degrees, aspect 1/cos(lat)) or `"km"` (km east / north of the centre of the map) |
| `MAP_GRID`, `SCALE_BAR`, `SCALE_BAR_KM` | `False`, `True`, `None` | grid, scale bar (`None` = a round length of about 1/5 of the width) |

**DEM**

| Setting | Default | Meaning |
|---|---|---|
| `DEM_SOURCE` | `"copernicus30"` | `"copernicus30"`, `"copernicus90"`, `"skadi"` (tiles: cache, download), `"tiles"` (`DEM_URL_TEMPLATE`), `"file"` (`DEM_FILE`), `"none"`; `"srtm"` is an error (it meant `"skadi"`) |
| `DEM_URL_TEMPLATE` | `None` | `None` = the tiles of `DEM_SOURCE`; else a mirror of them, or the tiles of `"tiles"`: a URL or a path (relative: from `WORK_DIR`) with the placeholders `{name}`, `{NS}`, `{lat}`, `{EW}`, `{lon}`; `.tif` / `.tiff` (rasterio) or `.hgt`, `.hgt.gz`, `.hgt.zip` |
| `DEM_DOWNLOAD` | `True` | `False`: only the tiles in `DEM_CACHE_DIR` |
| `DEM_CACHE_DIR` | `"dem_cache"` | tiles are kept here (an absolute path shares the cache between runs) |
| `DEM_FILE` | `None` | GeoTIFF (needs `rasterio`), `.npz` (`lon`, `lat`, `z`) or one `.hgt` tile |
| `DEM_TIMEOUT`, `DEM_MAX_TILES`, `DEM_MAX_PIXELS` | `60`, `16`, `1600` | s per tile; more tiles are refused; the DEM is averaged down to about this many pixels per side |
| `DEM_FILL_MISSING` | `0.0` | elevation of tiles the server does not have (ocean); `None` = no value |
| `DEM_STRICT` | `False` | `True`: a failed download, an unreadable tile or a missing `rasterio` stops the script (else a warning and a map without relief) |
| `DEM_STYLE`, `DEM_CMAP` | `"hillshade"`, `"gist_earth"` | grey shading, or colour + shading |
| `HILLSHADE_AZIMUTH`, `HILLSHADE_ALTITUDE`, `HILLSHADE_VERT_EXAG` | `315`, `45`, `1.0` | light direction and height [deg], vertical exaggeration |
| `BASEMAP_CMAP`, `BASEMAP_ALPHA` | `"Greys"`, `0.9` | |
| `DEM_CREDIT` | `"auto"` | text at the lower right of every map (wrapped to the width of the panel); `"auto"` = the notice of the source (none with `DEM_URL_TEMPLATE`, `"tiles"` or `"file"`); `None` = none |

**Icons and colours**

| Setting | Default | Meaning |
|---|---|---|
| `SITE_MARKER`, `SITE_SIZE` | `"^"`, `80` | |
| `SITE_EDGECOLOR`, `SITE_LW` | `"black"`, `0.8` | |
| `SITE_NODATA_EDGECOLOR` | `"0.35"` | open icons: sites without data in the selection |
| `SITE_LABELS` | `False` | site names (font: `LABEL_FONTSIZE`, see "Fonts") |
| `CMAP_NRMS` | `"RdBu_r"` | diverging, centred on `NRMS_REFERENCE` |
| `NRMS_REFERENCE` | `1.0` | the centre colour (and a dashed line on the colour bar); `None` = linear scale |
| `CMAP_SPREAD` | `"YlOrRd"` | colour map of `ENS_STAT = "std"` (sequential, from 0) |
| `NRMS_VMIN`, `NRMS_VMAX` | `None` | colour limits; `None` = from the data (symmetric around the reference, not below 0; `"std"`: 0 and the largest value). Values beyond the limits get the end colours and an arrow on the colour bar |

One colour scale is used for all pages. If a few outliers (short periods, off-diagonal
components) stretch it, set `NRMS_VMAX` (for the Krafla data 3.0 shows the structure).

**Figures and output**

| Setting | Default | Meaning |
|---|---|---|
| `PAGE_BY` | `"band"` | `"band"` (panels = components), `"component"` (panels = bands) or `"all"` |
| `PANEL_COLS`, `PANEL_SIZE` | `2`, `(12.2, 11.2)` | panels per row; width, height of one panel in **cm** (until 2026-11-26: inch, `(4.8, 4.4)`); both below 6 cm: a note |
| `TITLE` | `"auto"` | `"auto"`, `None` / `""` (none) or text with `{iter}`, `{nrms}`, `{M}`, `{stat}` |
| `CBAR_LOCATION`, `CBAR_SHRINK` | `None` (right), `0.8` | colour bar at `"right"` (default), `"left"`, `"top"`, `"bottom"`; its length as a fraction (0 .. 1] of the panels along it |
| `CBAR_LENGTH`, `CBAR_WIDTH` | `None`, `None` | size of the colour bar in **cm**. `CBAR_LENGTH` = the whole length with the extension triangles (`extend`; the coloured part is shorter by 5 % per triangle); wins over `CBAR_SHRINK`; `None` = `CBAR_SHRINK` x the panels. `CBAR_WIDTH` = the thickness; `None` = 1/20 of the length (matplotlib's). Either alone is fine. A length that the panels cannot hold is reduced to the length of the panels, with a message. See "Size of the colour bar" |
| `DPI`, `SHOW` | `200`, `False` | |
| `OUTPUT_DIR`, `OUTPUT_BASENAME`, `OUTPUT_FORMATS` | `"sitemap"`, `"femtic_sitemap"`, `["png"]` | |

**Fonts**

| Setting | Default | Meaning |
|---|---|---|
| `FONT_FAMILY` | `None` | `None` = matplotlib's; `"serif"`, `"sans-serif"`, `"monospace"`, the name of an installed font, or a list in fall-back order such as `["Arial", "DejaVu Sans"]`. A name that matplotlib does not find gives a warning, and another font is used |
| `FONT_RC` | `None` | further matplotlib rcParams for the text, e.g. `{"mathtext.fontset": "stix"}` or `{"font.style": "italic"}`; an unknown key or an invalid value is refused |
| `TICK_FONTSIZE` | `8.0` | tick labels of the maps; base of the sizes that are `None` |
| `AXIS_LABEL_FONTSIZE` | `None` | axis titles; `None` = `TICK_FONTSIZE` |
| `TITLE_FONTSIZE`, `TITLE_FONTWEIGHT` | `None`, `"normal"` | figure title; `None` = `TICK_FONTSIZE + 3`. A title wider than the figure is broken into lines |
| `PANEL_TITLE_FONTSIZE` | `None` | panel titles; `None` = `TICK_FONTSIZE + 1` |
| `LABEL_FONTSIZE`, `LABEL_FONTWEIGHT` | `6.0`, `"normal"` | site names |
| `LEGEND_FONTSIZE` | `None` | legend; `None` = `TICK_FONTSIZE - 1` |
| `CBAR_LABEL_FONTSIZE`, `CBAR_TICK_FONTSIZE` | `None`, `None` | colour bar title and numbers; `None` = matplotlib's |
| `SCALEBAR_FONTSIZE` | `None` | scale bar; `None` = `TICK_FONTSIZE - 1` |
| `CREDIT_FONTSIZE` | `5.0` | DEM credit line |
| `MAP_HALO` | `False` | outline around the site names, the scale bar and the DEM credit (until 2026-11-27 always drawn) |
| `MAP_HALO_COLOR`, `MAP_HALO_WIDTH` | `"white"`, `1.5` | its colour and width [pt] (scale bar: 1 pt more) |
| `PROFILE_FONTSIZE`, `PROFILE_TICK_FONTSIZE` | `9.0`, `None` | texts of the profiles, see "Profiles and lines" |

---

## Size of the colour bar

`CBAR_LENGTH` and `CBAR_WIDTH` [cm] fix what is drawn. The figure is laid out once without
drawing, the colour bar is made again with the length and the aspect ratio that this shows
are needed, and measured again (up to 6 passes; the size is held within 1 %, and a message says
if it is not reached). The title is set before, so the size is that of the finished figure,
also when the figure is saved.

- `CBAR_LENGTH` is the whole bar, with the extension triangles of a colour scale that has
  `NRMS_VMIN` / `NRMS_VMAX` inside the data; the coloured part alone is about 5 % (one
  triangle) or 9 % (two) shorter. It is cut to the panels if it is too long.
- `CBAR_WIDTH` is the thickness of the coloured part; without it, 1/20 of the length.
- Neither given: matplotlib's colour bar with `CBAR_SHRINK`, as before (no fit passes; the
  title is created before the bar. Compared once with the earlier script: the axes of the
  page have the same positions).

---

## Profiles and lines

`PROFILES` (cross-section lines) and `LINES` (faults, coast lines, caldera rims,
...) are lists of dicts in the CONFIG block; they are drawn on every map, on top
of the DEM and below the site icons.

```python
PROFILES = [
    dict(name="A-A'", start=(-16.83, 65.69, "A"), end=(-16.73, 65.73, "A'"),
         width_km=4.0, ticks_km=2.0, label="Section A"),
    dict(name="B-B'", center=(-16.78, 65.71), azimuth=90.0, length_km=6.0,
         width_km=1.0, ticks_km=1.0, color="tab:red"),
    dict(name="C-C'", file="gis/profile_c.txt", coords="utm", width_km=2.0),
    dict(name="D-D'", coords="utm", start=(405.0, 7291.8, "D"), end=(424.0, 7287.8, "D'"),
         width_km=4.0, ticks_km=5.0),                      # UTM km of the site.dat zone
]
LINES = [
    dict(label="Faults", file="gis/faults.txt", color="tab:red", ls="--"),
    dict(label="Caldera rim", file="gis/rim.geojson", color="tab:orange",
         fill="orange", alpha=0.15),
    dict(label="Coast", points=[(-16.9, 65.6), (-16.7, 65.65), (-16.5, 65.7)]),
]
```

**Profile keys** (exactly one way to give the geometry)

| Key | Meaning |
|---|---|
| `name` | name; the end labels follow from it (`"A-A'"` gives `A` and `A'`; a name without a hyphen `"P1"` gives `P1` and `P1'`); default `P1`, `P2`, ... |
| `points` | `[(a, b), ...]`, at least two different points (a polyline with bends is allowed) |
| `file` | text, GeoJSON, KML or KMZ file (see below); several parts in the file: the first one is the profile, with a warning |
| `start`, `end` | the two end points `(a, b)`, or `(a, b, "A")` with the label ("marker") of that end: `start=(x, y, "A")`, `end=(x, y, "A'")`. A third entry that is no text is ignored |
| `center`, `azimuth`, `length_km` | all three: centre `(a, b)`, direction in degrees clockwise from north (in the local km frame of the profile), length [km] |
| `coords` | `"lonlat"` (default), `"latlon"`, `"utm"` or `"model"` for `points`, `start`, `end`, `center` and a text `file` (see below) |
| `width_km` | width of the swath [km]; `0` / not given = no swath |
| `ticks_km` | spacing of the distance ticks along the profile [km] (a tick at 0, at every multiple, none at the end unless it is a multiple); at most 2000 ticks |
| `tick_labels` | `True` (default): the distance is written at every tick |
| `labels` | `False` = no end labels at all (also not the ones in `start` / `end`), or two texts `("A", "A'")`. The label of an end is, in this order: the text in `start` / `end`, `labels`, the half of `name` (see above) |
| `label` | legend text; without it the profile is not in the legend |
| style keys | `color`, `lw`, `ls`, `line_alpha`, `zorder`, `halo`, ... : see "Style" below |

**Line keys** (exactly one of `file`, `points`, `segments`)

| Key | Meaning |
|---|---|
| `label` | legend text; without it the line is not in the legend |
| `file` | text, GeoJSON, KML or KMZ file |
| `points` | one polyline `[(a, b), ...]` |
| `segments` | a list of polylines `[[(a, b), ...], ...]` |
| `coords` | as for the profiles (not for GeoJSON, KML, KMZ) |
| `closed` | `True` / `False` overrides the closing of all parts; default: a ring in a GeoJSON polygon or a KML `LinearRing`, or a text segment with equal first and last point (4 or more points) |
| `fill`, `alpha` | colour of the fill of closed parts and its opacity (default `LINE_FILL_ALPHA`); no fill without `fill` |
| style keys | `color`, `lw`, `ls`, `line_alpha`, `zorder`, `halo`, ... : see "Style" below |

**Style**

Every entry may set its own style. A key that is not given (or is `None`) takes
the setting of the CONFIG block, so one change there restyles all entries
without a key of their own. Aliases: `linewidth` = `lw`, `linestyle` = `ls`,
`ms` = `markersize`. A value that cannot work (colour, line style, symbol,
size, opacity, font weight) is refused before anything is read.

| Key (profile and line) | Setting `PROFILE_...` / `LINE_...` | Meaning |
|---|---|---|
| `color` | `COLOR` | colour |
| `lw` | `LW` | line width [pt] |
| `ls` | `LS` | `"-"`, `"--"`, `"-."`, `":"` or `(offset, (on, off, ...))` |
| `line_alpha` | `ALPHA` | opacity of the line |
| `zorder` | `ZORDER` | drawing order; the sites are at 5 |
| `halo`, `halo_color`, `halo_width` | `HALO`, `HALO_COLOR`, `HALO_WIDTH` | outline under the line (profiles: also under ticks and texts); width = how much wider than the line [pt] |
| `marker`, `markersize` | `MARKER`, `MARKER_SIZE` | symbol at every vertex; `None` = none |

| Key (profile only) | Setting | Meaning |
|---|---|---|
| `end_marker`, `end_marker_size` | `PROFILE_END_MARKER`, `PROFILE_END_MARKER_SIZE` | symbol at the two ends |
| `fontsize`, `fontweight` | `PROFILE_FONTSIZE`, `PROFILE_FONTWEIGHT` | end labels |
| `text_color` | `PROFILE_TEXT_COLOR` | end and distance labels; `None` = line colour |
| `label_offset` | `PROFILE_LABEL_OFFSET` | distance of an end label from the end [pt] |
| `tick_fontsize` | `PROFILE_TICK_FONTSIZE` | distance labels; `None` = end label size - 2 |
| `tick_length` | `PROFILE_TICK_LENGTH` | tick length as a fraction of the map width; `None` = 0.012 |
| `tick_lw` | `PROFILE_TICK_LW` | tick line width; `None` = 0.8 x `lw`, at least 0.8 |
| `swath_color`, `swath_alpha` | `PROFILE_SWATH_COLOR`, `PROFILE_SWATH_ALPHA` | swath; colour `None` = line colour |
| `mark_sites` | `PROFILE_MARK_SITES` | rings around the sites in the swath |
| `ring_size`, `ring_lw` | `PROFILE_RING_SIZE`, `PROFILE_RING_LW` | size of the rings (units of `SITE_SIZE`) and their line width [pt] |

```python
PROFILES = [
    dict(name="Vuache fault",
         start=(6.030869, 45.962166),      # (longitude, latitude)
         end=(6.115936, 45.910655),
         coords="lonlat", labels=False, label="Vuache fault",
         color="magenta", lw=3.0, ls="--", halo_width=2.0, zorder=5.5),   # on top of the sites
]
LINES = [dict(label="Faults", file="gis/faults.txt", color="tab:red", lw=2.0,
              marker="o", markersize=3.0)]
```

**Coordinates (`coords`)**

| Value | Meaning |
|---|---|
| `"lonlat"` | longitude, latitude in degrees (default) |
| `"latlon"` | latitude, longitude in degrees |
| `"utm"` | easting, northing in the UTM zone of `site.dat`: **km** when typed in the settings (`points`, `start`, `end`, `center`, `segments`), m in a text `file`. The zone and the hemisphere are found by comparing the easting / northing of the sites with their lat / lon (median distance of the best zone, at most 25 m; for the 100 Krafla sites 0.05 m). The transform is a Krueger series of 4th order on WGS84 (no `pyproj` needed) |
| `"model"` | x, y of the FEMTIC model: **km** when typed in the settings, m in a text `file` (as `site_x`, `site_y` of the results file); needs the sites assigned by position (`SITE_MATCH = "coords"`, or `"auto"` with site coordinates): the shift and the frame found there are used |

Check the order of a pair: `"latlon"` means (latitude, longitude) here. In
`femtic_ens_plot_slices.py` the tag `"latlon"` stands for [lon, lat], so a point
list copied from there needs `coords="lonlat"`.

**Units.** All geographic settings are in km, UTM coordinates too; coordinates read from
files (`site.dat`, the results file, text files of a profile or line) stay in
metres, as GIS programs and FEMTIC write them. A setting that looks like metres
(`"utm"`: easting above 2000 or northing above 20000; `"model"`: above 1000) is
refused; smaller values given in metres cannot be recognised and are read as km.

GeoJSON, KML and KMZ files are always lon, lat (as the formats say); `coords` is
ignored for them. Coordinates that are not on the globe (lat beyond 90, lon beyond
180) are refused: the usual cause is `lonlat` and `latlon` mixed up.

**Files**

- Text: two numbers per line (the first two columns; separated by spaces, commas
  or semicolons); a line that starts with `>` (GMT multi-segment) or a blank line
  ends a segment; `#` and `%` start comments; lines that are no numbers are
  skipped; segments of one point are dropped.
- GeoJSON (`.geojson`, `.json`): FeatureCollection, Feature, a geometry or a
  GeometryCollection; LineString, MultiLineString, Polygon (all rings, also the
  holes) and MultiPolygon are drawn, points are ignored.
- KML / KMZ (`.kml`, `.kmz`; in a KMZ the `doc.kml`, else the first `.kml`):
  every LineString and LinearRing (also the ones in a Polygon).
- Shapefiles are not read; convert them to GeoJSON (`ogr2ogr -f GeoJSON`).

**Swath and the sites**

With `width_km > 0` the swath is the strip of this width around the profile
(a polyline with bends: rectangles along the segments, round joins at the inner
vertices, no caps at the ends). A site is in the swath if its distance to the
profile is at most `width_km / 2` and its projection is not beyond the ends of the
profile (10 m tolerance). The sites in the swath are marked by a ring on the map
(`PROFILE_MARK_SITES`) and listed in `_profile_sites.csv`:

| Column | Meaning |
|---|---|
| `along_km` | distance along the profile from its start to the foot of the site |
| `offset_km` | signed distance across the profile; positive to the **right** of the direction start -> end |
| nRMS | one column per selection (as in `_nrms.csv`), empty without data |

Distances are flat-earth km on a sphere (R = 6371.0088 km; the scale of the map)
around the mean position of the profile; east-west they are about 0.3 % shorter
than on the WGS84 ellipsoid at mid latitudes. Use profiles of up to a few 100 km.

**Cut at the map box (`OVERLAY_CUT_MARGIN_KM`)**

Profiles and lines are cut at the map box (the extent of `MAP_EXTENT_MODE`, e.g. `MAP_BOX`),
made smaller by `OVERLAY_CUT_MARGIN_KM` (default **1 km**) on every side, so that both
ends of a cut profile lie inside the map, at least the margin from its edge (`None` = not cut;
the edge of the map hides what is outside, as before). For a profile cut this way:

- the line, the swath (flat at the cut) and the rings stop there; the end labels, and the
  end markers of `PROFILE_END_MARKER`, stand at the ends of what is shown;
- the ticks and their distances stay those of the whole profile, counted from its real start
  (a map that starts 7 km along the profile shows the ticks 10, 15, ...);
- a profile that leaves the box and comes back is shown in parts; one that is outside is not drawn
  (the warning names it); one that is inside is drawn as before;
- `_profiles.csv` and `_profile_sites.csv` still describe the whole profile and all its sites
  (also the ones outside the map); the console says how much was left:
  `profile 'A-A'' is cut at the map box (margin 1 km): 14.2 of 19.4 km are shown.`

Lines are cut in the same way (the outline of a polygon as a polyline, its fill as a
polygon). A margin larger than half the map is a warning, and nothing is cut. With
`MAP_EXTENT_MODE = "sites"` the map box is the box of the sites plus the map margin
(`MAP_MARGIN_FRAC`, `MAP_MARGIN_MIN_KM`), and **it does not depend on the profiles and lines**
(`MAP_INCLUDE_PROFILES = False`, the default): what goes beyond the box is cut at the box
made smaller by `OVERLAY_CUT_MARGIN_KM`. With `MAP_INCLUDE_PROFILES = True` the box is made
large enough for the profiles instead (a profile is then cut only if the map margin is
smaller than `OVERLAY_CUT_MARGIN_KM`); lines never change the box. The cut always refers to
the box that is drawn, whatever `MAP_EXTENT_MODE` is.

**Errors**

- A key that does not exist, a geometry given in two ways, a number that cannot
  be (a negative width, an azimuth that is no number), `coords` or `LEGEND_LOC`
  not in the list: `ValueError` from `settings()` (return code 2 on the command
  line), before anything is read.
- A problem with the data (a file that cannot be read or has no line, coordinates
  out of range, `"utm"` or `"model"` without a zone or frame, fewer than two
  different points): a warning and the profile or line is left out; with
  `OVERLAY_STRICT = True` the script stops with return code 1 before any output.
- A profile or line that lies outside the map is not drawn; the warning names it
  and gives its range of longitude and latitude and that of the map. If it would
  be on the map with the two exchanged, the message says so: check `coords`.

**Settings**

| Setting | Default | Meaning |
|---|---|---|
| `PROFILES`, `LINES` | `[]`, `[]` | lists of dicts, see above |
| `PROFILE_COLOR`, `PROFILE_LW`, `PROFILE_LS`, `PROFILE_ALPHA` | `"black"`, `1.8`, `"-"`, `1.0` | profile line |
| `PROFILE_ZORDER` | `4.0` | the sites are at 5 |
| `PROFILE_HALO`, `PROFILE_HALO_COLOR`, `PROFILE_HALO_WIDTH` | `False`, `"white"`, `1.6` | outline under line, ticks and texts |
| `PROFILE_MARKER`, `PROFILE_MARKER_SIZE` | `None`, `4.0` | symbol at every vertex |
| `PROFILE_END_MARKER`, `PROFILE_END_MARKER_SIZE` | `None`, `6.0` | a matplotlib marker (`"o"`, `"s"`, `"D"`, `"^"`, ...) at the two ends of every profile (of the part that is shown); `None` = none |
| `PROFILE_FONTSIZE`, `PROFILE_FONTWEIGHT` | `9.0`, `"bold"` | end labels |
| `PROFILE_TEXT_COLOR`, `PROFILE_LABEL_OFFSET` | `None`, `10.0` | colour of the labels; offset of the end labels [pt] |
| `PROFILE_TICK_FONTSIZE`, `PROFILE_TICK_LENGTH`, `PROFILE_TICK_LW` | `None`, `None`, `None` | distance labels and ticks, see "Style" |
| `PROFILE_SWATH_COLOR`, `PROFILE_SWATH_ALPHA` | `None`, `0.15` | swath (one patch, so the overlap at the bends is not darker) |
| `PROFILE_MARK_SITES` | `False` | rings around the sites in the swath |
| `PROFILE_RING_SIZE`, `PROFILE_RING_LW` | `1.9`, `1.0` | these rings |
| `LINE_COLOR`, `LINE_LW`, `LINE_LS`, `LINE_ALPHA` | `"tab:green"`, `1.2`, `"-"`, `1.0` | lines |
| `LINE_ZORDER` | `3.0` | the sites are at 5 |
| `LINE_HALO`, `LINE_HALO_COLOR`, `LINE_HALO_WIDTH` | `False`, `"white"`, `1.6` | outline under the lines |
| `LINE_MARKER`, `LINE_MARKER_SIZE`, `LINE_FILL_ALPHA` | `None`, `4.0`, `0.25` | symbol at every vertex; opacity of a fill |
| `OVERLAY_CUT_MARGIN_KM` | `1.0` | profiles and lines are cut at the map box made smaller by this [km] on every side; `None` = not cut; see "Cut at the map box" |
| `MAP_INCLUDE_PROFILES` | `False` | `MAP_EXTENT_MODE = "sites"`: `False` = the box of the map does not depend on the profiles, they are cut at it (`OVERLAY_CUT_MARGIN_KM`); `True` = the box includes the profiles (with the margin) |
| `OVERLAY_STRICT` | `False` | `True`: a profile or line that cannot be made stops the script |
| `LEGEND_LOC` | `"upper left"` | legend (icon "no data" and the labelled profiles and lines) in the first panel of a page: a matplotlib `loc` string, `None` = no legend |

Command line: `--profile NAME LON1 LAT1 LON2 LAT2` (repeatable; lon / lat) and
`--line FILE` (repeatable; the label is the file name) are added to `PROFILES`
and `LINES` of the script; `--no-overlays` ignores both lists (and cannot be
combined with the two options).

---

## DEM source

The DEM routines (tiles, download and cache, readers, mosaic, shading, credit)
are in `modules/dem.py` and can be used by other scripts; see
`modules/dem_readme.md`. The settings below are those of this script.

| `DEM_SOURCE` | Data | Tiles | Needs |
|---|---|---|---|
| `"copernicus30"` (default) | Copernicus DEM GLO-30, 1 arc second (30 m) | GeoTIFF (COG), bucket `copernicus-dem-30m` | `rasterio` |
| `"copernicus90"` | Copernicus DEM GLO-90, 3 arc seconds (90 m) | GeoTIFF (COG), bucket `copernicus-dem-90m` | `rasterio` |
| `"skadi"` | AWS Terrain Tiles (Mapzen), a composite | `.hgt.gz` | numpy only |
| `"tiles"` | your own | `.tif` or `.hgt` files, URL or path pattern | `rasterio` for `.tif` |
| `"file"` | one file | `.tif`, `.npz`, `.hgt` | `rasterio` for `.tif` |

- **Copernicus DEM** (ESA, based on TanDEM-X; the open data on AWS need no key and
  no login): 1 x 1 degree tiles,
  `https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N65_00_W017_00_DEM/Copernicus_DSM_COG_10_N65_00_W017_00_DEM.tif`
  (GLO-90: `copernicus-dem-90m`, `Copernicus_DSM_COG_30_...`). It is a **surface
  model** (buildings, vegetation) with heights above the **EGM2008 geoid**; for a
  hillshade this does not matter, but do not take the heights as ellipsoidal or as
  a bare-earth terrain. The GLO-30 tiles have fewer pixels in longitude at high
  latitudes (e.g. 1800 at 60 - 70 N); the script takes the pixel size from the
  file. There are **no tiles over the open ocean** (the server answers 403 / 404:
  the map gets `DEM_FILL_MISSING` there), and a few tiles of some countries are
  not public in GLO-30 (they are reported as failed or missing; use GLO-90 there).
- **The credit** the licence asks for is the default `DEM_CREDIT`: "produced using
  Copernicus WorldDEM-30 (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space
  GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights
  reserved" (WorldDEM-90 for GLO-90). Check the licence for your use.
- **"skadi"** (the former `"srtm"`): the AWS Terrain Tiles,
  `https://s3.amazonaws.com/elevation-tiles-prod/skadi/N37/N37W105.hgt.gz`, in the SRTM `.hgt`
  format. They are a **composite** of several elevation data sets (SRTM between
  about 60 N and 56 S, other sources at high latitudes, e.g. Iceland), not one model;
  the tile provider asks for attribution of the underlying sources (see
  https://registry.opendata.aws/terrain-tiles). Use it when `rasterio` is not
  installed.
- **How the tiles are read**: a tile of a URL is downloaded whole into
  `DEM_CACHE_DIR` (a `.part` file, then renamed; the file name is the last part of
  the URL) and never downloaded again; a GeoTIFF tile is then read with `rasterio`,
  only the window inside the map (plus a margin of two pixels), averaged down to
  about `DEM_MAX_PIXELS` pixels per side. A GLO-30 tile is roughly 4 - 40 MB. A
  cached file that cannot be read is downloaded again once. A path pattern
  (`"tiles"`, or `DEM_URL_TEMPLATE` with a path) is used as it is, nothing is copied.
  A raster in a projected coordinate system (a national DEM in UTM) is warped to
  lon / lat first (`WarpedVRT`); a rotated raster is refused.
- **Windows of several tiles** are put on one grid with the coarsest pixel size of
  the windows (nearest pixel); the ocean tiles are added as `DEM_FILL_MISSING`
  over the map and a margin of three pixels, not over the whole tile.
- A missing tile (HTTP 403 / 404 = ocean) is filled with `DEM_FILL_MISSING`; any
  other failure, an unreadable tile or no `rasterio` gives a warning and a map
  without relief (`DEM_STRICT = True`: an error). If all tiles are missing there is
  no DEM (a warning), so a wrong URL cannot give a flat sea-level map.
- The relief is a hillshade (`matplotlib.colors.LightSource`, spacing in metres from
  the pixel size in degrees); the map is limited to `DEM_MAX_TILES` tiles. Extents
  that cross the antimeridian are not supported.

### Examples: Copernicus GLO-30

In the CONFIG block of the script (`rasterio` has to be installed; the tiles of the
map are downloaded once; a map inside one degree square needs one tile, e.g. N65 W017):

```python
DEM_SOURCE = "copernicus30"                      # the default
DEM_CACHE_DIR = "/data/dem_cache"                # a cache shared by all runs and maps
DEM_STYLE = "terrain"; DEM_CMAP = "gist_earth"   # colour + shading instead of the grey hillshade
DEM_MAX_PIXELS = 2400                            # finer relief of a small map (default 1600)
```

Offline, with tiles that are already on the disk. Put the files under their original
names into `DEM_CACHE_DIR` (e.g. `Copernicus_DSM_COG_10_N65_00_W017_00_DEM.tif`) and stop the
download:

```python
DEM_SOURCE = "copernicus30"
DEM_DOWNLOAD = False
DEM_CACHE_DIR = "/data/dem_cache"
```

Tiles under other names or in another folder (a path pattern is used as it is, nothing
is copied; `{NS}{lat:02d}` is N65, `{EW}{lon:03d}` is W017):

```python
DEM_SOURCE = "tiles"
DEM_URL_TEMPLATE = "/data/cop30/Copernicus_DSM_COG_10_{NS}{lat:02d}_00_{EW}{lon:03d}_00_DEM.tif"
```

A mirror of the bucket, same file names: `DEM_SOURCE = "copernicus30"` with
`DEM_URL_TEMPLATE = "https://mirror.example.org/cop30/{name}.tif"`.

One GeoTIFF that you cut out yourself (e.g. `gdal_translate -projwin -17.3 65.9 -16.2 65.3 in.vrt krafla_cop30.tif`):

```python
DEM_SOURCE = "file"
DEM_FILE = "/data/krafla_cop30.tif"
```

From Python, without editing the script:

```python
import femtic_plot_sitemap as sm
cfg = sm.settings("/data/krafla/krafla_rto_26", DEM_SOURCE="copernicus30",
                  DEM_CACHE_DIR="/data/dem_cache", DEM_STYLE="terrain")
sm.run(cfg)
```

Sources: Registry of Open Data on AWS, "Copernicus Digital Elevation Model (DEM)"
(https://registry.opendata.aws/copernicus-dem/); "Terrain Tiles"
(https://registry.opendata.aws/terrain-tiles; Mapzen, "Terrain Tiles v1.1 Update",
the S3 path `elevation-tiles-prod/skadi/{N|S}{y}/{N|S}{y}{E|W}{x}.hgt.gz`).

References:

Copernicus GLO-30 Digital Elevation Model, data record of OpenTopography:
doi:10.5069/G9028PQB (data: ESA, DLR, Airbus Defence and Space).

Farr, T. G., Rosen, P. A., Caro, E., Crippen, R., Duren, R., Hensley, S., Kobrick,
M., Paller, M., Rodriguez, E., Roth, L., Seal, D., Shaffer, S., Shimada, J.,
Umland, J., Werner, M., Oskin, M., Burbank, D., Alsdorf, D. (2007): The Shuttle
Radar Topography Mission. Reviews of Geophysics 45, RG2004.
doi:10.1029/2005RG000183

---

## Output

In `OUTPUT_DIR` (default `WORK_DIR/sitemap`):

| File | Content |
|---|---|
| `<OUTPUT_BASENAME>_<tag>.<fmt>` | figure of a page: `_b00_all_bands`, `_b01_0.001_-_0.01_s`, ... for `PAGE_BY = "band"` |
| `<OUTPUT_BASENAME>_nrms.csv` | nRMS per site (rows: name, lat, lon, site ids; columns: `band \| components`); for an ensemble the statistic `ENS_STAT` over the members |
| `<OUTPUT_BASENAME>_n.csv` | number of values per site and selection (ensemble: median of the members) |
| `<OUTPUT_BASENAME>_summary.txt` | per selection: n, global nRMS, number of sites, min / median / max of the site values, values of rows without a site; ensemble: also the number of members `M` and the smallest and largest global nRMS of a member (`m_min`, `m_max`), and a header with the members found / used / above the limit / not readable and the iterations |
| `<OUTPUT_BASENAME>_members.csv` | (more than one member found) one row per member in the order found: `member`, `path` (the results file), `iteration`, `nrms` (of the iteration), `used` (1 / 0), `note` (why not used), then the global nRMS of the member for every selection |
| `<OUTPUT_BASENAME>_ens_stats.csv` | (ensemble) for every selection and site: `M`, mean, median, std, min, max of the member nRMS |
| `<OUTPUT_BASENAME>_profiles.csv` | (profiles) one row per profile: `profile`, `vertices`, `length_km`, `width_km`, `n_sites` (in the swath), `lon_start`, `lat_start`, `lon_end`, `lat_end`, `source` (how the profile was given) |
| `<OUTPUT_BASENAME>_profile_sites.csv` | (profiles with a swath) the sites in the swath, sorted by `along_km` for every profile: `profile`, `site`, `lat`, `lon`, `along_km`, `offset_km`, then the nRMS for every selection |

Further files with other `SITE_MEASURES` (`<m>` = the measure):

| File | Content |
|---|---|
| `<OUTPUT_BASENAME>_<m>_<tag>.<fmt>` | maps of the fit measure `<m>` (the nRMS maps have no `<m>` in the name) |
| `<OUTPUT_BASENAME>_<m>.csv`, `_<m>_ens_stats.csv` | as `_nrms.csv` and `_ens_stats.csv` |
| `<OUTPUT_BASENAME>_distortion.<fmt>`, `_distortion.csv` | distortion map; table with `site`, `lat`, `lon`, `site_ids`, `distortion` and, single run, `cp_xx`, `cp_xy`, `cp_yx`, `cp_yy`, `fixed`; ensemble, `M`, `mean`, `median`, `std`, `min`, `max`, `n_fixed` |

`_nrms.csv`, `_n.csv` and the nRMS part of the summary are written whatever
`SITE_MEASURES` is; `_members.csv` lists the nRMS only. `_profile_sites.csv` gets
extra columns `<m> | band | components` and `distortion`.

Only sites with data are listed; sites of `site.dat` without data in the selection
are drawn as open icons (legend "no data"). The summary lists the profiles (length,
vertices, swath and number of sites in it) and the lines (parts, vertices).

Example (`_summary.txt`, Krafla RTO member, iteration 8; MT only; the ensemble
summary has the extra columns described above):

```
iteration 8, 8608 data rows, nRMS of the iteration = 1.3870
100 sites with data of 100 in .../templates/site.dat; site ids assigned by coords

band            components      n   nRMS  sites    min  median    max  n_unassigned
-----------------------------------------------------------------------------------
all bands       all         17216  1.387    100  0.989   1.321  2.019             0
all bands       Zoff         8608  1.577    100  0.940   1.456  2.388             0
all bands       Zdiag        8608  1.167    100  0.825   1.159  1.565             0
0.001 - 0.01 s  all           552  2.089     69  0.441   1.763  4.059             0
0.001 - 0.01 s  Zoff          276  2.693     69  0.548   2.280  5.702             0
```

---

## Verification performed

**2026-11-28** (halos off, measures, distortion strength): new test file
`test_femtic_plot_sitemap_measures.py` (90 tests) and the 816 tests of
`test_femtic_plot_sitemap.py` (two changed for the halo defaults); all pass on a
copy of the script with the run settings of the CONFIG block set back to their
defaults, on synthetic data. Every fit measure per site and global was compared
with a plain loop over the rows and independent formulas (`numpy`,
`scipy.stats`), the distortion strength with `numpy.linalg.norm`, the ensemble
statistics with `numpy`. Maps of a made-up ensemble were looked at. **Not
tested**: a real results file; the reading of a real `/distortion` group (`h5py`
not installed: stand-in with the documented fields) and a real
`distortion_iter<N>.dat`.

**2026-11-27** (`dem.py`, fonts, styles, cm sizes): the test file has 36 new tests
(own styles, fonts and panel size, the hints for overlays outside the map, the
module `dem.py`). All 816 tests pass on a copy of the script in which the run
settings of the CONFIG block that the tests depend on were set back to their
defaults (`WORK_DIR`, `RESULTS_PATH`, `RUN_PATHS`, `COMPONENTS`, `PROFILES`,
`PROFILE_MARK_SITES = True`, `DEM_STYLE`, `DEM_CREDIT`, `CMAP_NRMS`, `NRMS_VMIN`,
`NRMS_VMAX`, `CBAR_SHRINK`, `SITE_LABELS = False`, `OUTPUT_FORMATS = ["png"]`);
with other run values some tests fail. Maps with the Vuache profile, own styles
and other fonts were drawn from made-up data and looked at. `h5py`, `pyproj` and
`rasterio` were replaced by test stand-ins (not installed, no network). The
numbers below are those of the earlier versions.

What was checked, and how. `h5py` and `rasterio` were not available and there was
no network where the script was written.

- **Unit and run tests**: `scripts/general/test_femtic_plot_sitemap.py`, 779
  tests (about 75 s; the whole `scripts/general` suite: 1291 passed, 2 skipped, 112 s), on synthetic results files (the HDF5 reader replaced by a
  stand-in that returns the same structure) and synthetic tiles: names and URLs
  of tiles, the `.hgt` reader (both sizes, void values, the overlapping edge row
  and column), the mosaic and the cache, the download with a mocked `urlopen`
  (success, 403 / 404, other errors, `.part` files), DEM files, crop and
  thinning, hillshade, map extent and frames, colour scale, assignment of the
  sites (by position, by id, order, tolerance, HTF / NMT, co-located stations),
  bands and components against the engine, nRMS against a brute-force loop over
  all rows, drawing, pages, tables, settings, return codes and command line.
  Ensemble: the statistics against `numpy` (with missing values), the discovery of
  the members, every `ENS_STAT` through `run` against a brute-force nRMS of every
  member and site, the tables, `MEMBER_NRMS_MAX` (also: one member left, none left),
  unreadable members (also the first), members with other site ids, without a
  site or without a selection, messages printed once, titles, colour scales and
  colours of the icons, settings and command line.
  Profiles and lines: the UTM transform against an independent numerical meridian
  arc (agreement about 5e-11 m) and against the 100 real Krafla `site.dat` rows;
  the zone search, `"model"` coordinates in both frames; the readers (text, GeoJSON,
  KML, KMZ, with comments, holes, broken files); the projection of the sites on
  straight and bent profiles against a plain loop (along, offset, sign, ends,
  round joins); the swath (one patch, same pixel value in the overlap); ticks;
  the specs (every error path), drawing (z-order, styles, labels, rings,
  legend), the tables, the summary, the extent, `run` and the command line.
  DEM sources (2026-11-22): names, URLs and the notice of every source; the tile
  kind from the extension; the download into the cache (whole tile, `.part` file,
  403 / 404, other errors, a broken cached file loaded once more); the window
  reader against an analytic surface (window and margin, averaging to
  `DEM_MAX_PIXELS`, nodata, flipped axes, projected rasters through a stand-in
  `WarpedVRT`, rotated and south-up rasters refused); `merge_dems` (pixel sizes,
  first valid value wins, ocean boxes in whole pixels, so that the land pixels do
  not move); `load_dem_raster` (missing tiles, ocean fill, all tiles missing),
  files, settings, credit line and `run` with every source. `rasterio` is a
  stand-in in `sys.modules` (pickled rasters, a fake `urlopen` server).
- **Mutation testing**: about 70 hand-made changes of the main logic (comparison
  signs, constants, indices, ordering, dropped checks) and 106 of the ensemble
  code; every one that changes the behaviour is caught by a test (two of the 106
  are equivalent: a one-value std is NaN anyway, and all-missing columns are NaN
  anyway). For the profiles and lines 211 changes (UTM series and signs, zone
  search, readers, profile geometry, swath, projection, ticks, specs, drawing,
  tables, `run`, command line): 207 are caught; the last four have no effect on
  the result (the `%` comment test of the text reader, because a line that starts
  with `%` is no number anyway; the candidate zones taken from the extremes of
  the longitudes, which the mean +- 1 zone covers; the `min` in the tick position,
  which only removes a rounding error; `not overlays` against `overlays is None`
  with an empty dict). The 24 that survived the first round led to new tests.
  For the DEM code (2026-11-22) about 209 changes (tile names, URLs, kind of tile,
  cache and download, window and margin, averaging, nodata, warping, merging,
  ocean boxes, clipping, `load_dem_file`, settings, credit line, command line):
  about 205 are caught; the four that survive do not change the result (two
  latitude arguments of the window reader that only matter for rotated rasters,
  the order of two lookups in `load_dem_file`, a clip of the ocean boxes that the
  merge repeats, and `fill=None` in the merge with no box). Two real defects
  were found this way and are fixed: ocean boxes were not clipped to the map when
  no fill value was given, and boxes that were not a whole number of pixels wide
  shifted the land pixels.
- **Real data**: the script was run on the Krafla RTO member `krafla_rto_26`
  (iteration 8, 8608 data rows, 100 sites, 22 frequencies, MT only) with the real
  `templates/site.dat`; the HDF5 file was read with a libhdf5 reader written for the
  test (through `ctypes`), not with `h5py`. The global nRMS for all bands and
  components is 1.387, the RMS of iteration 8 in the `.cnv` file (1.3870); the
  values of iterations 1 and 7 also agree with the `.cnv` file (2.0337, 1.3906).
  The 100 site ids are matched by position (shift 7288933.8, 418059.5 m, largest
  distance 0.7 m) and agree with the matching by number (`SITE_ID_OFFSET = 1`).
  Two pairs of ids sit at the same position (ids 59 / 89 and 86 / 88): they get
  the two `site.dat` entries each. In this run no DEM was used (no network).
- **Real data, ensemble**: only one real member was available, so a pseudo-ensemble
  was made of its iterations 1, 7 and 8 (three member directories and one
  `templates/` with the real `site.dat`). The three members are found, read and
  matched (100 of 100 sites each); their nRMS of the iteration (2.034, 1.391, 1.387)
  agrees with the `.cnv` file, and the median and std maps and the tables were
  produced. This tests the reading, the aggregation and the figures on real
  data, not the statistics of a real RTO ensemble.
- **Real data, profiles**: on the Krafla member two profiles of the same line,
  one given in lon / lat and one in UTM (zone 28 N found from `site.dat`, median
  distance 0.05 m), gave the same 36 sites in a 2 km swath; a profile given by
  centre, azimuth and length gave 14 sites in a 1 km swath; a polygon (text and
  GeoJSON) and a two-part fault file were drawn with `DISPLAY_COORDS` `"km"` and
  `"latlon"`, and the figures were looked at. The files were made for the test.
- **Colour bar size and the cut of profiles and lines** (2026-11-22): tests on synthetic figures
  (`TestColorbarSize`: sizes at the four locations with length, width, both, and a length that is
  too long; extension triangles; with and without a title; no leftover bar; no fit pass without
  a size) and on synthetic geometry
  (`TestClipPolyline` against dense sampling of polylines, `TestClipPolygon` against a grid
  count, `TestCutBox`, `TestCutProfile`, `TestCutLine`, `TestCutDrawing`, `TestCutRun`;
  `TestProfileEndLabels`, `TestProfileEndMarker`). 116 mutations of the new code (clipping,
  cut box, cut of profiles and lines, drawing, end labels, validation, colour bar fit and
  its measure): 108 are caught by the tests; the 8 that are not are equivalent (redundant
  guards, e.g. the length check of a clipped part, a zero-length guard that no profile can
  reach, or early returns that only save time). The colour bar sizes were checked on
  matplotlib 3.11.2 only (the layout code of the colour bar differs between versions).
  Not looked at with real data.
- **Not tested**: the readers on files that come from GIS programs (QGIS, Google
  Earth, `ogr2ogr`; only hand-made files); `rasterio` itself (not installed where
  this was written: only the stand-in), so the reading of a real GeoTIFF / COG and
  the warping of a raster in a projected system (`WarpedVRT`) have not run; the
  download from the AWS buckets (no network; the file names are those of the bucket
  documentation); and a real DEM in the real-data run. The default `"copernicus30"`
  needs `rasterio` on your machine; `"skadi"` does not. The Krafla area (65.7 N) is
  north of the SRTM coverage, so the "skadi" tiles there come from the other sources
  of the composite; the Copernicus DEM covers it.

---

## Limitations

- One iteration per member and run; use `-i` and `OUTPUT_DIR` for several.
- The member nRMS is against the member's own data; there is no RMS1 of the
  ensemble mean, and no per-site test against the ensemble mean.
- The members are read one after the other (no parallel reading); every member is
  read completely before its nRMS is computed, then dropped from memory.
- Extents that cross the antimeridian are not supported (also not by the
  overlays).
- Profiles are straight lines or polylines in a flat-earth km frame, not great
  circles or geodesics; distances differ from the ellipsoid by about 0.3 %
  east-west. The sites are projected on the nearest segment, and the swath has
  no end caps.
- No shapefiles; KML is read with `xml.etree` (no protection against hostile
  XML: do not read KML files of unknown origin); KML styles, names and
  altitudes are ignored.
- A profile in `"model"` coordinates needs the sites assigned by position; with
  `SITE_MATCH = "id"` use `"lonlat"` or `"utm"`.
- `modules/topo.py` has its own DEM download (USGS); it is not used here.
- The Copernicus DEM is a surface model with EGM2008 heights; GLO-30 has no tiles
  over the open ocean and lacks a few countries (use GLO-90 or your own DEM there).
- Where tiles of different pixel size meet (the latitude zones of GLO-30, or windows
  that are thinned differently) the grid has the coarsest pixel size, and the
  outermost pixel rows outside the map can be empty; the map itself is not affected.
- Only whole tiles are downloaded (no remote windows of the COG), so a map of one
  small area still costs a tile of about 4 - 40 MB the first time.
- The colour scale is common to all pages and has no robust automatic limit; set
  `NRMS_VMAX` if outliers dominate.
- Distortion: only the matrix-difference type; the type of a text file is not
  checked; one scalar per site (no map of the single elements, gain or rotation).

---

## Provenance

| Date | Author | Change |
|---|---|---|
| 2026-11-21 | Claude Sonnet 5.5 (Anthropic) | Created. Per-site nRMS for bands x components as coloured icons on a shaded relief from SRTM-format tiles (download with a cache, or a DEM file, or none); tables and summary; `site.dat` searched in `TEMPLATES_DIR`, `WORK_DIR` and next to `WORK_DIR`; ids at one position get the co-located `site.dat` entries in the order of their numbers. AI-generated; review before use. Checked on synthetic data and on the Krafla RTO member (see "Verification performed"); the DEM download and the GeoTIFF reader were not run. |
| 2026-11-21 | Claude Sonnet 5.5 (Anthropic) | A full ensemble is read: `WORK_DIR` = ensemble root (members and `templates/`), or `RUN_PATHS`; `ENS_STAT` (mean, median, min, max, std, percentile) over the members for every site and selection; `MEMBER_NRMS_MAX`; unreadable members are skipped; new tables `_members.csv`, `_ens_stats.csv`; ensemble summary, titles, colour bar label and a sequential scale for the std; `--runs`, `--stat`. Single-run output unchanged. AI-generated; review before use. Checked on synthetic ensembles and on a pseudo-ensemble of three real Krafla iterations. |
| 2026-11-22 | Claude Sonnet 5.5 (Anthropic) | Profile lines and other polylines: `PROFILES` (points, file, start + end, centre + azimuth + length; swath, ticks, end labels, the sites in the swath with `along_km` / `offset_km`) and `LINES` (points, segments, text / GMT, GeoJSON, KML, KMZ; polygons with fill); coordinates lonlat, latlon, utm (own Krueger series, zone found from `site.dat`) and model; legend (`LEGEND_LOC`); map extent with the profiles; new tables `_profiles.csv`, `_profile_sites.csv`; `--profile`, `--line`, `--no-overlays`. Without `PROFILES` and `LINES` the output is unchanged. AI-generated; review before use. Checked on synthetic data (417 tests) and on the real Krafla results; the readers were not tried on files from GIS programs. |
| 2026-11-22 | Claude Sonnet 5.5 (Anthropic) | DEM: default Copernicus DEM GLO-30 (`DEM_SOURCE` `"copernicus30"`, also `"copernicus90"`, `"tiles"` with own URL or path pattern, GeoTIFF or `.hgt`); GeoTIFF / COG read with `rasterio` (window, averaging to `DEM_MAX_PIXELS`, warping of projected rasters), tiles merged, ocean tiles filled; the AWS Terrain Tiles are now `"skadi"` and `"srtm"` is refused; `DEM_URL_TEMPLATE` default None, `DEM_CACHE_DIR` default `dem_cache`; credit line = the licence notice; example lines for the GLO-30 (cache, colour relief, offline, own tile names, one GeoTIFF, Python). AI-generated; review before use. Checked with 596 tests and about 209 mutations on a `rasterio` stand-in; `rasterio`, the real GeoTIFF reading, the warping and the AWS download were not run. |
| 2026-11-22 | Claude Sonnet 5.5 (Anthropic) | `CBAR_LENGTH`, `CBAR_WIDTH` [cm]: the size of the colour bar (the figure title is now set before the bar). Profiles and lines are cut at the map box made smaller by `OVERLAY_CUT_MARGIN_KM` (default 1 km): both ends of a cut profile are inside the map, end labels, swath, rings and ticks follow (ticks keep the distances of the whole profile). End labels in `start` / `end`: `start=(x, y, "A")`, `end=(x, y, "A'")`; `PROFILE_END_MARKER`. AI-generated, review before use. Checked with 779 tests and 116 mutations on synthetic data; not run on real results. |
| 2026-11-22 | Claude Sonnet 5.5 (Anthropic) | `MAP_INCLUDE_PROFILES` now defaults to `False`: the map box (`"sites"`: sites + margin) is not enlarged by profiles that go beyond it; they are cut at that box (`OVERLAY_CUT_MARGIN_KM`). `True` gives the earlier behaviour. This changes the extent of maps with profiles that went beyond the sites. AI-generated, review before use. |
| 2026-11-26 | Claude Sonnet 5.5 (Anthropic) | All user settings are km: `SITE_MATCH_TOL` (`0.015`; before `15.0` m) and `"utm"` / `"model"` positions of `PROFILES` / `LINES` typed in the settings (before: m; the default profile is the same one written in km). Personal settings in metres must be converted; values that look like metres are refused. Coordinates read from files stay in m. `--profile` (degrees) is unchanged. Tested on synthetic data only. AI-generated; review before production use. |
| 2026-11-27 | Claude Sonnet 5.5 (Anthropic) | (1) The DEM routines moved to the new module `modules/dem.py`. (2) Fonts are user settings (`FONT_FAMILY`, `FONT_RC`, sizes and weights). (3) Profiles and lines: the "Vuache fault" entry was not drawn because its (longitude, latitude) pairs were read with `coords="latlon"`; it is now `"lonlat"`, with `labels=False` and a legend label. Style settings and per-entry keys for profiles and lines; the warning for an overlay outside the map gives the ranges and a hint on swapped coordinates. (4) `PANEL_SIZE` is in cm (before: inch), default `(12.2, 11.2)`. AI-generated; review before use. Tested on synthetic data only. |
| 2026-11-28 | Claude Sonnet 5.5 (Anthropic) | Halos off by default (`PROFILE_HALO`, `LINE_HALO` `False`; new `MAP_HALO` `False` for site names, scale bar and DEM credit). New `SITE_MEASURES`, `DISTORTION_FILE`, `MEASURE_SCALES`, `--measures`: maps and tables of the fit measures (`misfit_measures.py`) and of the galvanic distortion strength `\|C - I\|`. With `SHOW = False` every figure is closed once saved. AI-generated, review before use. Checked with synthetic data only (see "Verification performed"). |

Author: Volker Rath (DIAS)
