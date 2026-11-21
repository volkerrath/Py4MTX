# femtic_plot_sitemap.py

Site map of a FEMTIC run, or of a whole ensemble, on a DEM background. Every
site is an icon coloured by its normalized RMS misfit (nRMS), for frequency bands
and data components. For an ensemble (`WORK_DIR` = the ensemble root with the
member directories and `templates/`) the colour is a statistic of the member
nRMS at the site (median by default; mean, min, max, a percentile or the std).
The DEM is made of SRTM-format tiles that are downloaded by default (and
cached); a DEM file or no DEM are the alternatives.

Same layout as the other FEMTIC scripts: a CONFIG block of UPPERCASE settings
at the top of the script, `WORK_DIR` as the base of all relative paths,
`-w` / `-i` / `--dem` / `--runs` / `--stat` on the command line, `settings()` /
`run()` for use from other code. For a plot of the one nRMS per site (all bands, all components) see
also `femtic_plot_nrms.py`; this script adds the bands x components selections,
the tables and the DEM download.

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
     found from the data (`SITE_MATCH_TOL` = largest distance of a match, in m).
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
6. **Basemap** (see "DEM source" below) and **figures**: one figure per page
   (`PAGE_BY`), one panel per selection, one colour scale for all pages.
7. **Tables**: nRMS and number of values per site and selection, a summary and,
   for an ensemble, one table of the members and one of the statistics.

---

## Quick start

```
python femtic_plot_sitemap.py -w /data/krafla/krafla_rto_26          # settings in the script
python femtic_plot_sitemap.py -w /data/krafla/krafla_rto_26 --dem none   # without DEM, no download
python femtic_plot_sitemap.py -w /data/krafla/krafla_rto_26 -i last

# a whole ensemble: WORK_DIR = the root with krafla_rto_NN/ and templates/
python femtic_plot_sitemap.py -w /data/krafla --dem none                  # median of all members
python femtic_plot_sitemap.py -w /data/krafla --stat std                  # spread of the members
python femtic_plot_sitemap.py -w /data/krafla --stat p90 --runs "krafla_rto_1*"
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

## Settings

Everything in the CONFIG block; relative paths refer to `WORK_DIR`.

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
| `SITE_MATCH_TOL` | `15.0` | [m] largest distance of a match by position |
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
| `DEM_SOURCE` | `"srtm"` | `"srtm"` (tiles: cache, download), `"file"` (`DEM_FILE`), `"none"` |
| `DEM_URL_TEMPLATE` | AWS skadi | placeholders `{name}`, `{NS}`, `{lat}`, `{EW}`, `{lon}` |
| `DEM_DOWNLOAD` | `True` | `False`: only the tiles in `DEM_CACHE_DIR` |
| `DEM_CACHE_DIR` | `"srtm_cache"` | tiles are kept here (an absolute path shares the cache between runs) |
| `DEM_FILE` | `None` | GeoTIFF (needs `rasterio`), `.npz` (`lon`, `lat`, `z`) or one `.hgt` tile |
| `DEM_TIMEOUT`, `DEM_MAX_TILES`, `DEM_MAX_PIXELS` | `60`, `16`, `1600` | s per tile; more tiles are refused; the DEM is thinned to at most this many pixels per side |
| `DEM_FILL_MISSING` | `0.0` | elevation of tiles the server does not have (ocean); `None` = no value |
| `DEM_STRICT` | `False` | `True`: a failed download stops the script (else a warning and a map without relief) |
| `DEM_STYLE`, `DEM_CMAP` | `"hillshade"`, `"gist_earth"` | grey shading, or colour + shading |
| `HILLSHADE_AZIMUTH`, `HILLSHADE_ALTITUDE`, `HILLSHADE_VERT_EXAG` | `315`, `45`, `1.0` | light direction and height [deg], vertical exaggeration |
| `BASEMAP_CMAP`, `BASEMAP_ALPHA` | `"Greys"`, `0.9` | |
| `DEM_CREDIT` | `"auto"` | text at the lower right of every map; `None` = none |

**Icons and colours**

| Setting | Default | Meaning |
|---|---|---|
| `SITE_MARKER`, `SITE_SIZE` | `"^"`, `80` | |
| `SITE_EDGECOLOR`, `SITE_LW` | `"black"`, `0.8` | |
| `SITE_NODATA_EDGECOLOR` | `"0.35"` | open icons: sites without data in the selection |
| `SITE_LABELS`, `LABEL_FONTSIZE` | `False`, `6` | site names |
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
| `PANEL_COLS`, `PANEL_SIZE` | `2`, `(4.8, 4.4)` | panels per row, inch |
| `TITLE` | `"auto"` | `"auto"`, `None` / `""` (none) or text with `{iter}`, `{nrms}`, `{M}`, `{stat}` |
| `CBAR_LOCATION`, `CBAR_SHRINK` | `None` (right), `0.8` | |
| `TICK_FONTSIZE`, `DPI`, `SHOW` | `8`, `200`, `False` | |
| `OUTPUT_DIR`, `OUTPUT_BASENAME`, `OUTPUT_FORMATS` | `"sitemap"`, `"femtic_sitemap"`, `["png"]` | |

---

## DEM source

- Default: the "skadi" tile set of the AWS Terrain Tiles (public bucket
  `elevation-tiles-prod`, no key, no login),
  `https://s3.amazonaws.com/elevation-tiles-prod/skadi/N37/N37W105.hgt.gz`.
  The tiles are 1 x 1 degree, in the SRTM `.hgt` format (16-bit big-endian,
  3601 x 3601 or 1201 x 1201 samples, first row at the northern edge, void
  -32768), gzipped.
- **They are a composite**, not pure SRTM: SRTM is the main source between about
  56 S and 60 N; other sources are used elsewhere (high latitudes, e.g. Iceland).
  The original NASA / USGS distribution of SRTM needs an Earthdata login (not
  done here). The tile provider asks for attribution of the underlying sources
  when the tiles are used for analysis or display (see
  https://aws.amazon.com/public-datasets/terrain/): check it before publishing a
  figure. The default credit line is "Relief: SRTM-format AWS terrain tiles (SRTM
  courtesy of USGS)" (`DEM_CREDIT`).
- Tiles are downloaded once into `DEM_CACHE_DIR` (`.part` file, then renamed).
  A tile that is there (`.hgt`, `.hgt.gz`, `.hgt.zip`) is never downloaded again;
  tiles of another server can be put there by hand or by `DEM_URL_TEMPLATE`.
- A missing tile (HTTP 403 / 404 = ocean) is filled with `DEM_FILL_MISSING`; any
  other failure gives a warning and a map without relief (`DEM_STRICT = True`:
  an error).
- The relief is a hillshade (`matplotlib.colors.LightSource`, spacing in metres from
  the pixel size in degrees); the map is limited to `DEM_MAX_TILES` tiles and the
  DEM is thinned to `DEM_MAX_PIXELS`. Extents that cross the antimeridian are not
  supported.

Sources for the tile set: Registry of Open Data on AWS, "Terrain Tiles"
(https://registry.opendata.aws/terrain-tiles); Mapzen, "Terrain Tiles v1.1 Update"
(https://www.mapzen.com/blog/terrain-tiles-v1.1/; the S3 path
`elevation-tiles-prod/skadi/{N|S}{y}/{N|S}{y}{E|W}{x}.hgt.gz`).

Reference for SRTM:

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

Only sites with data are listed; sites of `site.dat` without data in the selection
are drawn as open icons (legend "no data").

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

What was checked, and how. `h5py` and `rasterio` were not available and there was
no network where the script was written.

- **Unit and run tests**: `scripts/general/test_femtic_plot_sitemap.py`, 265
  tests (about 30 s), on synthetic results files (the HDF5 reader replaced by a
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
- **Mutation testing**: about 70 hand-made changes of the main logic (comparison
  signs, constants, indices, ordering, dropped checks) and 106 of the ensemble
  code; every one that changes the behaviour is caught by a test (two of the 106
  are equivalent: a one-value std is NaN anyway, and all-missing columns are NaN
  anyway). The ones that survived the first rounds led to new tests.
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
- **Not tested**: the download from the real server (the sandbox proxy refused
  s3.amazonaws.com; the URL pattern is the one of the AWS documentation), the
  GeoTIFF reader (needs `rasterio`), and a real DEM in the real-data run. The
  Krafla area (65.7 N) is north of the SRTM coverage, so the tiles there come from
  the other sources of the composite.

---

## Limitations

- One iteration per member and run; use `-i` and `OUTPUT_DIR` for several.
- The member nRMS is against the member's own data; there is no RMS1 of the
  ensemble mean, and no per-site test against the ensemble mean.
- The members are read one after the other (no parallel reading); every member is
  read completely before its nRMS is computed, then dropped from memory.
- Extents that cross the antimeridian are not supported.
- `modules/topo.py` has its own DEM download (USGS); it is not used here.
- The colour scale is common to all pages and has no robust automatic limit; set
  `NRMS_VMAX` if outliers dominate.

---

## Provenance

| Date | Author | Change |
|---|---|---|
| 2026-11-21 | Claude Sonnet 5.5 (Anthropic) | Created. Per-site nRMS for bands x components as coloured icons on a shaded relief from SRTM-format tiles (download with a cache, or a DEM file, or none); tables and summary; `site.dat` searched in `TEMPLATES_DIR`, `WORK_DIR` and next to `WORK_DIR`; ids at one position get the co-located `site.dat` entries in the order of their numbers. AI-generated; review before use. Checked on synthetic data and on the Krafla RTO member (see "Verification performed"); the DEM download and the GeoTIFF reader were not run. |
| 2026-11-21 | Claude Sonnet 5.5 (Anthropic) | A full ensemble is read: `WORK_DIR` = ensemble root (members and `templates/`), or `RUN_PATHS`; `ENS_STAT` (mean, median, min, max, std, percentile) over the members for every site and selection; `MEMBER_NRMS_MAX`; unreadable members are skipped; new tables `_members.csv`, `_ens_stats.csv`; ensemble summary, titles, colour bar label and a sequential scale for the std; `--runs`, `--stat`. Single-run output unchanged. AI-generated; review before use. Checked on synthetic ensembles and on a pseudo-ensemble of three real Krafla iterations. |

Author: Volker Rath (DIAS)
