# modem_femtic_interp.py

Volume-aware interpolation between a ModEM model (`.rho`) and a FEMTIC model
(`mesh.dat` + `resistivity_block_iterN.dat`), in both directions.

- Author: Claude (Anthropic), generated 2026-10-04 for Volker Rath (py4mt).
- Status: AI-generated. Tested on synthetic data only (see "Tests"); NOT yet
  run on real ModEM/FEMTIC files. Review before production use.
- Requires: numpy, scipy (`scipy.stats.qmc`, `scipy.spatial`), plus py4mt
  `modem.py` and `femtic.py` (imported lazily inside `run()`).

## Usage

Edit the UPPERCASE config block at the top of the script, then:

    python modem_femtic_interp.py

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

## Outputs

modem2femtic: `FEMTIC_OUT` (block written via `femtic.insert_model`).
femtic2modem: `MODEM_OUT + MODEM_EXT` (written with `MODEM_OUT_TRANS`).
Optional `DIAGNOSTICS_NPZ` stores coverage (and region/cell values).

## Tests (synthetic, 2026-10-04)

- constant model reproduced exactly in both directions;
- aligned tets/cells round trip: max relative error 5e-15;
- air template cells preserved; partial overlap keeps template values;
- full file-I/O run through real `modem.py`/`femtic.py` (numba, netCDF4 and
  ensembles stubbed): region values converge to an independent fine-voxel
  reference as `OVERSAMPLE` grows.

## Known limitations

- Isotropic regions only (v5 anisotropic blocks: first component is used by
  the reader).
- Region-based FEMTIC models: modem2femtic gives one value per region.
- Ocean handling in modem2femtic relies on the block's own region flags.

## Changelog

- 2026-10-04  Claude Sonnet 5.5 (Anthropic): created script and readme.
