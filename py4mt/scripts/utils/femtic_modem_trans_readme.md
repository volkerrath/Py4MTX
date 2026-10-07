# femtic_modem_trans.py

Volume-aware interpolation between a ModEM model (`.rho`) and a FEMTIC model
(`mesh.dat` + `resistivity_block_iterN.dat`), in both directions; optionally
also transformation of the observed data (FEMTIC `observe.dat` <-> ModEM data
file), in both directions.

- Author: Claude (Anthropic), generated 2026-10-04 for Volker Rath (py4mt).
- Status: AI-generated. Tested on synthetic data only (see "Tests"); NOT yet
  run on real ModEM/FEMTIC files. Review before production use.
- Requires: numpy, scipy (`scipy.stats.qmc`, `scipy.spatial`), plus py4mt
  `modem.py` and `femtic.py` (imported lazily inside `run()`).

## Usage

Edit the UPPERCASE config block at the top of the script, then:

    python femtic_modem_trans.py

| DIRECTION      | Source                         | Result                                             |
|----------------|--------------------------------|----------------------------------------------------|
| `modem2femtic` | ModEM `.rho` model             | New resistivity block: each *free* region gets the volume-weighted mean of the ModEM cells it overlaps |
| `femtic2modem` | FEMTIC mesh + block            | New ModEM model on the grid of `MODEM_MODEL` (template): each cell gets the volume-weighted mean of the tetrahedra it overlaps |

## Method

Each tetrahedron is covered by N equal-weight points (randomly shifted Sobol
points mapped uniformly into the tet), weight `V_tet / N`. N is a power of
two adapted to `V_tet / V_cell_min` (`OVERSAMPLE`, `N_MIN`, `N_MAX`).
Weighted sums per target (cell or region) give the volume-weighted mean of
log10(rho) (default), sigma, or rho (`AVERAGING`). Properties:

- Weights of all points sum to the exact tet volume (conservative).
- Air (ModEM rho >= `AIR_RHO_THRESHOLD`, FEMTIC region 0) is excluded from
  averages. Template air cells and fixed FEMTIC regions are never changed.
- Targets whose valid coverage is below `MIN_COVERAGE` keep the template
  value (femtic2modem: optionally the value of the tet containing the cell
  centre, `CENTRE_FALLBACK`).
- modem2femtic: results are clipped to region bounds (`CLIP_TO_BOUNDS`).
- The estimate is Monte-Carlo (error ~ 1/sqrt(N points)). Raise `OVERSAMPLE`
  / `N_MAX` for small regions; fix `SEED` for reproducibility.

## Coordinates -- check before trusting a run

Internal frame: (north, east, z-down) = ModEM order. `MESH_COLUMNS="ne"`
assumes mesh nodes are (northing, easting, z); use `"en"` otherwise.
`COORD_MODE="centre"` puts the ModEM model centre at the FEMTIC origin;
`"reference"` uses the ModEM reference corner. `OFFSET_N/E/Z` add shifts.
The script prints grid vs. mesh bounding boxes and warns if they do not
overlap. A wrong frame shows up as low coverage.

## Georeferencing (COORD_MODE = "georef")

Instead of hand-set offsets, the placement of the ModEM grid in the FEMTIC
frame is computed from the UTM origins of both models (same UTM zone;
`UTM_ZONE` / `UTM_NORTHERN`, else derived from the mean site lat/lon).

FEMTIC origin (UTM of the mesh centre), `FEMTIC_ORIGIN_METHOD`:

| Method          | Input                                                              |
|-----------------|--------------------------------------------------------------------|
| `box` (default) | `SITE_DAT` (name,lat,lon,elev,num,E,N): midpoint of the site bounding box, as in femtic_gst_prep |
| `calibration`   | `CALIBRATION_SITES` (lat/lon or UTM) + model-local x/y [km] in `OBSERVE_DAT` (x = east, y = north); origin = mean(UTM_site - local), residuals printed |
| `manual`        | `FEMTIC_ORIGIN_UTM = (easting, northing)`                          |

ModEM origin (UTM of the data-frame origin X = Y = 0), `MODEM_ORIGIN_METHOD`:

| Method          | Input                                                              |
|-----------------|--------------------------------------------------------------------|
| `data` (default)| `MODEM_DATA + MODEM_DATA_EXT`: per-site lat/lon and X (north), Y (east); origin = mean(UTM_site - (Y, X)); scatter over sites printed |
| `manual`        | `MODEM_ORIGIN_LATLON = (lat, lon)`                                 |

The ModEM grid is placed with its stored reference corner in the data
frame and shifted by (UTM_modem - UTM_femtic) in north/east; `OFFSET_*`
are added on top. This assumes the `.rho` reference corner is expressed in
the same frame as the data-file X/Y (check the printed bounding boxes).
A large residual in the origin tables points to wrong coordinates, a
wrong zone, or swapped X/Y. The UTM conversion is built in
(`latlon_to_utm`, WGS84, Kruger series, sub-mm; no pyproj/util needed).

## Observed-data transformation (optional, `TRANSFORM_DATA = True`)

Off by default. `TRANSFORM_MODEL` / `TRANSFORM_DATA` select the steps; both
may run in one call. Direction: `DATA_DIRECTION`, or `DIRECTION` if None.

| Direction      | Input                                | Output                         |
|----------------|--------------------------------------|--------------------------------|
| `modem2femtic` | `MODEM_DATA + MODEM_DATA_EXT`        | `OBSERVE_OUT` (observe.dat)    |
| `femtic2modem` | `OBSERVE_DAT`                        | `MODEM_DATA_OUT + MODEM_DATA_EXT` |

Block types (`DATA_TYPES`):

| FEMTIC | ModEM DataType             | Components                         |
|--------|----------------------------|------------------------------------|
| MT     | Full_Impedance (Off_Diagonal_Impedance read, see below) | ZXX ZXY ZYX ZYY |
| VTF    | Full_Vertical_Components   | TX TY                              |
| PT     | Phase_Tensor (real)        | PTXX PTXY PTYX PTYY                |

Other ModEM DataTypes (Off_Diagonal_Rho_Phase, Full_Interstation_TF) are
skipped and reported. ModEM data file reading is done by a self-contained
block-aware parser in this script (`read_modem_data`); FEMTIC files go
through `femtic.read_observe_dat` / `femtic.write_observe_dat`.

Conversions (neutral form: FEMTIC layout, SI Ohm, exp(-i omega t)):

- Units: Z_Ohm = Z * mu0 * 1e3 for `[mV/km]/[nT]`, Z * mu0 for `[V/m]/[T]`,
  Z for `Ohm` (read from the block header; written as `MODEM_Z_UNITS_OUT`).
  Tipper and phase tensor are dimensionless.
- FT convention: read from the block header; exp(+i omega t) data are
  conjugated (Z and tipper) on input. Output follows `MODEM_FT_OUT`.
- Period/frequency: f = 1/T; FEMTIC frequencies are written ascending.
- Errors: ModEM has one error per datum, FEMTIC one for Re and one for Im.
  modem2femtic copies it to both; femtic2modem merges them with
  `MODEM_ERR_COMBINE` (`max` default, `min`, `mean`, `re`, `im`). PT: single
  error. ModEM errors on impedances are scaled by the unit factor.
- Missing components (e.g. ZXX/ZYY of Off_Diagonal_Impedance): periods are
  dropped (count printed) unless `MISSING_ERR` is a number, in which case
  value 0 with that error (SI) is written.
- Site positions: ModEM X = north, Y = east, Z down (data frame) ->
  FEMTIC header `name x y z` with (x, y) = (east, north) for
  `FEMTIC_SITE_XY = "en"` (or swapped for "ne"), z down; FEMTIC = ModEM +
  shift, with the shift taken from the same placement logic as the model
  grid (`COORD_MODE`: centre / reference / georef, plus `OFFSET_N/E`), so
  sites and model stay consistent; `DATA_Z_OFFSET` is added to z.
- femtic2modem lat/lon: from the UTM origins when `COORD_MODE = "georef"`
  (inverse UTM, `utm_to_latlon`), else from `SITE_DAT` by site name, else
  0.0 with a warning. The data-header origin is the ModEM origin (georef) or
  `MODEM_ORIGIN_LATLON`. Note: georef with `MODEM_ORIGIN_METHOD = "data"`
  needs the ModEM data file to exist; use "manual" for femtic2modem if not.
- Not applied: block orientation angle of the ModEM header (warning if
  non-zero); no rotation, no distortion handling.

## Outputs

modem2femtic: `FEMTIC_OUT` (block written via `femtic.insert_model`).
femtic2modem: `MODEM_OUT + MODEM_EXT` (written with `MODEM_OUT_TRANS`).
Optional `DIAGNOSTICS_NPZ` stores coverage (and region/cell values).
Data step: see the data section above.

## Tests (synthetic, 2026-10-04)

- constant model reproduced exactly in both directions;
- aligned tets/cells round trip: max relative error 5e-15;
- air template cells preserved; partial overlap keeps template values;
- UTM conversion checked against the numerically integrated meridian arc
  (central meridian) and the Snyder series off the meridian (< 1e-5 m);
- georef: `box`, `calibration` and `manual` placements run end to end; box
  and calibration reproduce a hand-set `reference` + offsets run exactly;
- full file-I/O run through real `modem.py`/`femtic.py` (numba, netCDF4 and
  ensembles stubbed): region values converge to an independent fine-voxel
  reference as `OVERSAMPLE` grows.

## Tests of the data step (synthetic, 2026-10-07)

- FEMTIC -> ModEM -> FEMTIC round trip (MT, VTF, PT; 3 sites, 5 frequencies,
  georef/manual): frequencies, data (rtol 1e-5), positions (< 0.01 m) agree;
  errors equal max(Re, Im) of the originals as expected;
- same round trip with exp(+i omega t) and `Ohm` units;
- `[mV/km]/[nT]` output checked against the closed form Z/(mu0*1e3);
- site lat/lon checked against an independent inverse-UTM of the expected
  UTM position; frame shift checked for centre/reference modes;
- Off_Diagonal_Impedance: dropped without / filled with `MISSING_ERR`.

## Known limitations

- Data step: FEMTIC conventions assumed, NOT verified against a real FEMTIC
  run: observe.dat Z in Ohm with exp(-i omega t); site header `name x y z`
  in metres (z down), x = east; VTF columns (Tzx_re, Tzx_im, Tzy_re,
  Tzy_im) with the same tipper sign as ModEM; PT columns Pxx Pxy Pyx Pyy.
  Check one site against a known dataset before production use. Never run
  on real ModEM data files yet.

- Isotropic regions only (v5 anisotropic blocks: first component is used by
  the reader).
- Region-based FEMTIC models: modem2femtic gives one value per region.
- Ocean handling in modem2femtic relies on the block's own region flags.

## Changelog

- 2026-10-04  Claude Sonnet 5.5 (Anthropic): created script and readme.
- 2026-10-05  Claude Sonnet 5.5 (Anthropic): added `COORD_MODE = "georef"`
  (FEMTIC origin from site.dat box / observe.dat calibration / manual;
  ModEM origin from data-file lat/lon + X/Y / manual; built-in UTM).
- 2026-10-07  Claude Sonnet 5.5 (Anthropic): renamed `modem_femtic_interp.py`
  to `femtic_modem_trans.py` (readme likewise). No functional change.
- 2026-10-07  Claude Sonnet 5.5 (Anthropic): optional observed-data
  transformation (`TRANSFORM_DATA`): ModEM data reader and writer, FEMTIC
  observe.dat <-> ModEM data file in both directions (MT, VTF, PT), unit/FT
  conversion, frame shift and georeferenced site lat/lon, `utm_to_latlon`;
  `run()` now runs `_run_model()` and/or `_run_data()`.
