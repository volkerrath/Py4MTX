# README\_femtic.md

Module `femtic.py` — FEMTIC-specific I/O, model conversion, and format utilities.

---

## Module boundaries

| Module | Responsibility |
|---|---|
| **`femtic.py`** (this file) | FEMTIC file I/O, model read/write, mesh parsing, NPZ/VTK/NetCDF conversion |
| **`ensembles.py`** | Roughness/precision matrix tools, Gaussian sampling, ensemble generation, EOF/PCE |

`femtic.py` imports the shared matrix/roughness tools (`get_roughness`,
`make_prior_cov`, `matrix_reduce`, `check_sparse_matrix`, `save_spilu`,
`load_spilu`, and sparse pruning helpers) directly from `ensembles.py` so
they remain available as `femtic.<name>()` for backward compatibility.
Ensemble generation (`generate_directories`, `generate_model_ensemble`, etc.)
and all sampling helpers live exclusively in `ensembles.py`.

---

## Overview

`femtic.py` provides:

- **Data I/O** — read and modify `observe.dat` files (impedance, VTF, phase
  tensor), including Gaussian perturbation of data for RTO ensembles.
- **Distortion I/O** — read FEMTIC distortion files and decompose into C and
  C′ matrices.
- **Resistivity-block model workflow** — a clean 3-step read → NPZ → modify →
  write pipeline for `resistivity_block_iterX.dat`.
- **Mesh I/O** — parse FEMTIC `mesh.dat` tetrahedral meshes.
- **NPZ ↔ VTK / VTU** — convert NPZ model files for ParaView / PyVista.
- **NPZ ↔ NetCDF / HDF5** — CF-compliant and HDF5 export/import.
- **Model-local coordinate helpers** — UTM ↔ model-local conversions,
  CRS-tagged position resolution for slice specifications.
- **Site-list and mesh-centre helpers** — read `mt_make_sitelist.py` CSV and
  `observe.dat` site positions; estimate the mesh-centre UTM origin from
  calibration sites or a bounding-box midpoint.
- **Borehole sampling** — point-in-tetrahedron search for 1-D ρ(z) extraction
  along vertical boreholes (`_point_in_tet`, `extract_borehole_log`).
- **Model cell summary** — count air / ocean / other-fixed / free-parameter
  elements in a `resistivity_block_iterXX.dat` (`summarise_model_file`).
- **Data summary** — report sites, frequencies, data-vector size per
  observation type from an `observe.dat` (`summarise_observe_dat`).
- **CLI interface** — subcommand-style command-line usage for batch conversion.

---

## Resistivity-block model workflow (NPZ-based)

### Fixed regions and sampling compatibility

The NPZ stores:

| Field         | Description                                          |
|---------------|------------------------------------------------------|
| `fixed_mask`  | Boolean per region — `True` if the region is frozen. |
| `free_idx`    | Indices of regions that may change during sampling.  |
| `model_free`  | Free-vector in `model_trans` space (default log₁₀ρ). |

Fixed regions are defined as: region 0 (air) always fixed; any region with
`flag == 1` fixed; region 1 treated as ocean-fixed when `ocean_present` is true.

### 1) Read model → NPZ

```python
from femtic import read_model_to_npz

read_model_to_npz(
    model_file="resistivity_block_iter0.dat",
    npz_file="model_iter0.npz",
    model_trans="log10",
)
```

### 2a) Modify by adding Gaussian noise (log₁₀-space)

```python
from femtic import modify_model_npz

modify_model_npz(
    npz_in="model_iter0.npz",
    npz_out="model_iter0_noisy.npz",
    method="add_noise",
    add_sigma=0.05,
)
```

### 2b) Modify by precision-matrix sampling

```python
modify_model_npz(
    npz_in="model_iter0.npz",
    npz_out="model_iter0_prior_draw.npz",
    method="precision_sample",
    roughness="roughening_matrix.out",
    lam_mode="scaled_median_diag",
    lam_alpha=1e-5,
    solver_method="cg",
    scale=1.0,
    add_to_current=True,
)
```

### 2c) Precision sampling with preconditioning (recommended)

`solver_method="cg"` uses iterative solves of `Qx=b`.  Preconditioning options:

| `precond`    | Notes                                                     |
|--------------|-----------------------------------------------------------|
| `"jacobi"`   | Does **not** form `Q` explicitly; safest default.         |
| `"ilu"`      | Requires sparse `R`; builds `Q = R^T R` as sparse.       |
| `"amg"`      | Requires `pyamg`; builds sparse `Q`.                      |
| `"identity"` | No-op; useful for debugging.                              |
| `None`       | Unpreconditioned CG.                                      |

```python
modify_model_npz(
    npz_in="model_iter0.npz",
    npz_out="model_iter0_prior_draw_pcg.npz",
    method="precision_sample",
    roughness="roughening_matrix.out",
    lam_mode="scaled_median_diag",
    lam_alpha=1e-5,
    solver_method="cg",
    precond="jacobi",
    scale=1.0,
)
```

### 3) Write FEMTIC model from NPZ

```python
from femtic import write_model_from_npz

write_model_from_npz(
    npz_file="model_iter0_prior_draw.npz",
    model_file="resistivity_block_iter0_new.dat",
    also_write_npz="resistivity_block_iter0_new.npz",
)
```

---

## Command-line interface

```bash
python femtic.py femtic-to-npz \
    --mesh mesh.dat \
    --rho-block resistivity_block_iter0.dat \
    --out-npz femtic_model.npz

python femtic.py npz-to-vtk \
    --npz femtic_model.npz \
    --out-vtu model.vtu \
    --out-legacy model.vtk

python femtic.py npz-to-femtic \
    --npz femtic_model.npz \
    --mesh-out mesh_reconstructed.dat \
    --rho-block-out resistivity_block_iter0_reconstructed.dat

python femtic.py edi-to-observe SITE01.edi SITE02.edi \
    --xy-csv positions.csv \
    --out observe.dat
```

---

## Key data-handling functions

| Function              | Purpose                                                  |
|-----------------------|----------------------------------------------------------|
| `read_observe_dat()`  | Parse FEMTIC `observe.dat` into a nested dict (blocks → sites). |
| `summarise_observe_dat()` | Report sites / frequencies / data-vector size per obs-type block. |
| `summarise_model_file()` | Count air / ocean / other-fixed / free-parameter elements in a resistivity block. |
| `sites_as_dict_list()` | Flatten parsed observe.dat into a list of per-site dicts.      |
| `write_observe_dat()` | Write a parsed (and possibly modified) structure back to disk.  |
| `edi_list_to_observe_dat()` | Convert a list of `data_proc.load_edi` dicts to `observe.dat`. |
| `observe_to_site_viz_list()` | Read observe.dat and return per-site dicts with Z, rhoa, phase. |
| `modify_data()`       | Add Gaussian perturbations to observation data.                  |
| `insert_model()`      | Write sampled log₁₀ρ into a resistivity block file.     |
| `read_distortion_file()` | Read FEMTIC galvanic distortion file.                |
| `read_resistivity_block()` | Parse resistivity block file → dict of arrays.     |
| `tet_volumes()` | Vectorised tetrahedral volume via scalar triple product (nelem,). |
| `build_region_geometry()` | Volume-weighted centroid and total volume for each free region. |
| `ellipsoid_mask()` | Boolean mask for points inside a rotated ellipsoid (ZYX or SDS convention). |
| `brick_mask()` | Boolean mask for points inside a rotated rectangular prism. |
| `read_site_position()` | Return (x_m, y_m) model-local position for a site from observe.dat. |
| `read_site_dat()`     | Parse a `mt_make_sitelist.py` CSV (name, lat, lon, elev, sitenum, easting, northing). |
| `estimate_utm_origin()` | Estimate mesh-centre UTM coordinates from calibration sites or bounding-box midpoint. |
| `extract_borehole_log()` | Sample ρ along a vertical borehole by point-in-element search. |

---

## EDI → observe.dat conversion

### Programmatic usage

```python
import data_proc as dp
import femtic as fem

# 1. Load EDI files
edi_files = ["SITE01.edi", "SITE02.edi", "SITE03.edi"]
edis = [dp.load_edi(f, manufacturer="metronix", err_kind="var") for f in edi_files]

# 2. Attach model-frame XY (metres, from your projection step)
positions = {"SITE01": (1000.0, 2000.0), "SITE02": (3000.0, 4000.0), "SITE03": (5000.0, 6000.0)}
for edi in edis:
    x, y = positions[edi["station"]]
    edi["x_m"] = x
    edi["y_m"] = y

# 3. Write observe.dat — z handled automatically (elev_m negated)
fem.edi_list_to_observe_dat(edis, "observe.dat", preamble="# generated by edi_list_to_observe_dat\n")
```

**Unit conversion** applied internally:

```
Z_SI [Ω] = Z_MT [mV km⁻¹ nT⁻¹] × μ₀ × 10³     (μ₀ = 4π × 10⁻⁷ H m⁻¹)
```

**z-convention** applied automatically:

```
z_femtic = -elev_m        # EDI z-up (positive = above datum) → FEMTIC z-down
```

Sites with `elev_m = None` are placed at `z = 0.0` with a `UserWarning`.
Sites with no finite Z values are skipped with a `UserWarning`.

### CLI usage

```bash
# Minimal: positions from CSV (columns: station,x_m,y_m — no header)
python femtic.py edi-to-observe SITE01.edi SITE02.edi \
    --xy-csv positions.csv \
    --out observe.dat

# Phoenix instruments (FT sign-convention correction applied automatically)
python femtic.py edi-to-observe *.edi \
    --xy-csv positions.csv \
    --manufacturer phoenix \
    --out observe.dat

# All options
python femtic.py edi-to-observe SITE*.edi \
    --xy-csv positions.csv \
    --manufacturer metronix \
    --err-kind var \
    --freq-order inc \
    --preamble "# MT survey — converted from EDI\n# Project: Example" \
    --out observe.dat
```

**`positions.csv`** format (no header line required; non-numeric rows skipped):

```
SITE01,1000.0,2000.0
SITE02,3000.0,4000.0
SITE03,5000.0,6000.0
```

---

## Site-list, observe.dat, and mesh-centre helpers

These functions (Section 6b) were moved from `femtic_mod_plot.py` to make
them reusable across scripts without copying code.

### `read_site_position(observe_file, site_number)`

Scans `observe.dat` for a site-header line matching *site_number* and returns
`(x_m, y_m)` in model-local metres (converted from the km stored in the file).

### `read_site_dat(path, site_names=None)`

Parses the comma-separated sitelist produced by `mt_make_sitelist.py`
(`WHAT_FOR="femtic"`).  Column order (no header, `#` = comment):

```
name, lat, lon, elev, sitenum, easting, northing
```

Returns a list of dicts with keys `name`, `lat`, `lon`, `elev`, `sitenum`,
`easting`, `northing`.  Pass `site_names` (str or list of str) to filter by
name; `None` = return all rows.

### `estimate_utm_origin(calibration_sites, observe_file, zone, northern, *, site_dat=None, out=True)`

Two methods, selected automatically:

**Bounding-box centre** (default when `calibration_sites` is empty): reads
all UTM coordinates from `site_dat` and returns the midpoint of the bounding
box — identical to the femticPY convention.  No `observe.dat` required.

**Calibration-site pairs** (when `calibration_sites` is non-empty): each
entry supplies a site number, its CRS (`"latlon"` or `"utm"`), and its
geographic coordinates.  Model-local positions are read from `observe.dat`
(or from inline `x_km`/`y_km` keys).  Returns the least-squares mean of the
implied origins; per-site residuals are printed when `out=True`.

```python
import femtic as fem

# Bounding-box from sitelist
origin_E, origin_N = fem.estimate_utm_origin(
    [], "observe.dat", zone=19, northern=False,
    site_dat="site.dat",
)

# Calibration-site pairs
origin_E, origin_N = fem.estimate_utm_origin(
    [
        dict(site=1,  crs="latlon", coords=[-71.500, -16.380]),
        dict(site=10, crs="utm",    coords=[224500., 8179300.]),
    ],
    "observe.dat", zone=19, northern=False,
)
```

### `_point_in_tet(p, verts)` / `extract_borehole_log(...)`

`_point_in_tet` tests point containment in a tetrahedron via barycentric
coordinates (tolerance 10⁻¹⁰ on each test).

`extract_borehole_log` samples element resistivity along a vertical borehole:

```python
depths, rho = fem.extract_borehole_log(
    nodes, conn, rho_elem,
    x_m=0.0, y_m=0.0,
    z_top=0.0, z_bot=20000.0, dz=200.0,
)
```

A lateral bounding-box pre-filter reduces the per-depth element test to a
small candidate set; exact barycentric containment is checked only for those
candidates.  Levels outside the mesh are returned as NaN.

---

## Mesh geometry helpers

These functions operate on the raw `(nodes, conn)` arrays returned by
`read_femtic_mesh()` and are used internally by `femtic_mod_edit.py` and
`ensembles.py`.

### `tet_volumes(nodes, conn)`

Vectorised tetrahedral volume computation using the scalar triple product:

```
V_i = |det([b-a, c-a, d-a])_i| / 6
```

Returns a `(nelem,)` array of absolute volumes in metres³ (assuming node
coordinates are in metres).

### `build_region_geometry(nodes, conn, elem_region, free_idx)`

For each free region (identified by `free_idx`), computes:

- the **volume-weighted centroid** `[x, y, z]`
- the **total volume** (sum of element volumes assigned to that region)

Returns `(region_ctr, region_vol)` arrays of shape `(n_free, 3)` and
`(n_free,)`.  Regions with no assigned elements receive centroid `[0, 0, 0]`
and volume `0`.  Used by `femtic_mod_edit.py` to set up the smooth and wmean
operations.

### `ellipsoid_mask(centroids, *, center, axes, angles_deg, convention)`

Returns a boolean `(n,)` mask for points inside a rotated ellipsoid.

```python
mask = fem.ellipsoid_mask(
    centroids,
    center=[0., 0., 5000.],
    axes=[10000., 10000., 5000.],
    angles_deg=[0., 0., 0.],
    convention="zyx",   # or "sds"
)
```

### `brick_mask(centroids, *, center, axes, angles_deg, convention)`

Boolean `(n,)` mask for points inside a rotated rectangular prism.
`axes = [a, b, c]` are half-extents; the containment test in the local frame
is `|x'| ≤ a AND |y'| ≤ b AND |z'| ≤ c`.  Same signature and `convention`
parameter as `ellipsoid_mask()`.

```python
mask = fem.brick_mask(
    centroids,
    center=[0., 0., 5000.],
    axes=[10000., 8000., 4000.],
    angles_deg=[30., 0., 0.],
    convention="zyx",
)
```

---

## z-convention (z positive downward)

FEMTIC uses a right-handed coordinate system with **z increasing downward** (depth),
consistent with `mesh.dat` node coordinates.  All observe.dat site headers store
`(name, x, y, z)` in this frame: surface stations have **negative** z.

| Quantity | Convention | Sign at surface |
|---|---|---|
| EDI `ELEV=` field | geodetic (z-up) | positive |
| `observe.dat` site header z | FEMTIC (z-down) | **negative** |
| `xyz[2]` from `_site_header_to_meta` | FEMTIC (z-down) | **negative** |
| `elev_m` from `_site_header_to_meta` | geodetic (z-up) | positive |

`observe_to_site_viz_list` returns both `xyz` (FEMTIC z-down) and `elev_m`
(geodetic, positive-up) in every site dict for convenience.

`write_observe_dat` emits a `UserWarning` if any site header has a positive z
value, which almost certainly indicates a missing negation upstream.

---

## Dependencies

| Package             | Role                                          |
|---------------------|-----------------------------------------------|
| `numpy`             | Core array operations.                        |
| `scipy`             | Sparse matrices, solvers.                     |
| `joblib` (optional) | Kept for backward compatibility.              |

Optional for visualisation and export:

| Package    | Role                         |
|------------|------------------------------|
| `pyvista`  | VTK/VTU mesh export.         |
| `netCDF4`  | NetCDF export.               |
| `xarray`   | Higher-level NetCDF support. |

---

## Related modules and scripts

| File                  | Purpose                                                              |
|-----------------------|----------------------------------------------------------------------|
| **`ensembles.py`**    | Roughness/precision tools, sampling, ensemble generation, EOF / PCE. |
| `femtic_viz.py`       | Matplotlib and PyVista visualisation.                                |
| `femtic_mod_plot.py`  | Read and plot slice panels of a FEMTIC model; calls `fem.*` helpers. |
| `snippets.py`         | Optional code blocks for `femtic_mod_plot.py` (e.g. ensemble plot).  |
| `util.py`             | General-purpose helpers.                                             |
| `femtic_rto_rough.py` | Extract roughness matrix from FEMTIC.                               |
| `femtic_rto_prior.py` | Build prior covariance proxy.                                       |
| `femtic_rto_prep.py`  | Generate RTO ensemble.                                              |
| `femtic_summarize_model_cells.py` | CLI: cell-count summary for `resistivity_block_iterXX.dat`. Delegates to `summarise_model_file()`; self-contained fallback when `femtic.py` is not on the path. |
| `femtic_summarize_observe_dat.py` | CLI: data-content summary for `observe.dat`. Delegates to `summarise_observe_dat()`; self-contained fallback when `femtic.py` is not on the path. |

---

## Version / provenance

Updated: 2026-06-22

### Changelog (2026-06-22)
- **v5 (anisotropic) resistivity-block format support** added throughout.
  The v4 format (isotropic-only, 6 columns per region line) is fully
  backward-compatible; format is auto-detected at file-open time with no
  change to any call sites.

  New helpers:
  - `_detect_block_format(line)` — sniffs a region line and returns `"v4"` or
    `"v5"` based on whether the second token is a small integer anisotropy
    type (0/1/2) followed by a float.
  - `_parse_region_line_v4(line)` — renamed core of the original
    `_parse_region_line`; returns the 6-tuple `(ireg, rho, lo, hi, n, flag)`.
  - `_parse_region_line_v5(line)` — returns a full dict for v5 region lines,
    covering all three anisotropy sub-types mirroring the C++ reader logic:
    - type 0 (ISO, 6 cols): `ireg, 0, rho, rho_lo, rho_hi, flag`
    - type 1 (TI, 12 cols): adds `rhoXX, rhoYY, strike, dip, rho_lo, rho_hi,
      fix_rhoXX, fix_rhoYY, fix_strike, fix_dip`
    - type 2 (GA, 16 cols): additionally `rhoZZ, slant, fix_rhoZZ, fix_slant`
  - `_format_region_line_v5(d)` — inverse formatter for all three sub-types.
  - Module-level constants `_ANISO_ISO = 0`, `_ANISO_TI = 1`, `_ANISO_GA = 2`.

  Updated functions (all backward-compatible):
  - `_parse_region_line(line, fmt="v4")` — now dispatches on `fmt`; v5 returns
    the same 6-tuple with `rho = rhoXX` and `n = 1.0`.
  - `_infer_ocean_present(line, fmt="v4")` — passes `fmt` through.
  - `read_model()` — auto-detects format; prints `fmt=` in info line;
    otherwise identical interface and return type.
  - `summarise_model_file()` — auto-detects format.
  - `insert_model()` — for v5 files, preserves all per-block anisotropic
    fields verbatim for fixed blocks; for free anisotropic (TI/GA) blocks,
    scales rhoYY and rhoZZ by the same ratio as rhoXX so relative anisotropy
    is maintained.
  - `_read_resistivity_block_struct()` — auto-detects; stores `fmt` and
    `region_lines_raw` in the returned struct to enable lossless v5 round-trips.
  - `_write_resistivity_block_struct()` — uses `region_lines_raw` to
    reconstruct v5 lines; falls back to v4 writer when `fmt == "v4"`.
  - `read_resistivity_block()` — auto-detects; for v5 files returns extra keys
    `fmt`, `region_aniso_type`, `region_rhoYY`, `region_rhoZZ`,
    `region_strike`, `region_dip`, `region_slant` (all shape `(nreg,)`).
  - `write_resistivity_block()` — parameter `fmt` renamed `float_fmt`; new
    keyword-only parameter `block_fmt="v4"|"v5"` selects the output format,
    plus optional `region_rhoYY`, `region_rhoZZ`, `region_aniso_type`,
    `region_strike`, `region_dip`, `region_slant`, `region_fix_flags`.

### Changelog (2026-06-10)
- Added `summarise_model_file(path, *, ocean=None, out=True)` — counts air,
  ocean, other-fixed, and free-parameter elements in a
  `resistivity_block_iterXX.dat`.  Reuses the existing `_parse_region_line`
  and `_infer_ocean_present` infrastructure; prints a formatted table when
  `out=True`; companion `_print_model_summary()` formats the table from a
  cached dict.  Inserted between `read_model` and `insert_model`.
- Added `summarise_observe_dat(path_or_parsed, *, out=True)` — reports sites,
  frequencies, data values per obs-type block (MT, VTF, PT) and global totals
  from an `observe.dat`.  Accepts a filesystem path or an already-parsed dict
  (avoids re-parsing when `read_observe_dat` has already been called); calls
  `read_observe_dat` with `compute_mt_derived=False, bootstrap_n=0` for speed.
  Companion `_print_observe_summary()` formats the table.  Inserted before
  `write_observe_dat`.
- Both functions are also exposed as standalone CLI scripts:
  `femtic_summarize_model_cells.py` and `femtic_summarize_observe_dat.py`.
- Updated: Overview, Key data-handling functions table, Related modules table.

### Changelog (2026-06-08)
- Promoted three geometry helpers from `femtic_mod_edit.py` into the
  mesh/geometry section (placed between `build_element_arrays` and
  `_rotation_matrix_zyx`):
  - `tet_volumes(nodes, conn)` — vectorised tetrahedral volume via scalar
    triple product; replaces the private `_tet_volumes` in mod_edit.
  - `build_region_geometry(nodes, conn, elem_region, free_idx)` — computes
    volume-weighted centroid and total volume for each free region; replaces
    the private `_build_region_geometry` in mod_edit.
  - `brick_mask(centroids, *, center, axes, angles_deg, convention)` —
    rotated rectangular prism containment test; parallel to `ellipsoid_mask()`
    with the same keyword-only signature and `convention` parameter.
- `femtic_mod_edit.py` now delegates to these functions; its private
  `_tet_volumes`, `_build_region_geometry`, `_rotation_matrix_zyx`,
  `_local_coords`, `_ellipsoid_mask`, and `_brick_mask` have been removed.
- Updated: Overview, Key data-handling functions table.

### Changelog (2026-05-24)
- Added Section 6b: site-list, observe.dat, and mesh-centre helpers moved
  from `femtic_mod_plot.py`: `read_site_position()`, `read_site_dat()`,
  `estimate_utm_origin()`, `_point_in_tet()`, `extract_borehole_log()`.
- `read_site_dat()` parses the `mt_make_sitelist.py` CSV format
  (name, lat, lon, elev, sitenum, easting, northing); replaces the old
  whitespace-format `site.dat` reader.
- `estimate_utm_origin()` kwarg renamed `sitelist_file` → `site_dat`;
  bounding-box path now calls `read_site_dat()` directly.
- Updated: Overview, Key data-handling functions table, Related modules table.

### Changelog (2026-04-13)
- z-convention documented and enforced consistently across all observe.dat
  interfaces: `_site_header_to_meta` now returns `elev_m = -xyz[2]`;
  `observe_to_site_viz_list` exposes `elev_m` in every site dict;
  `write_observe_dat` emits a `UserWarning` for positive site-header z values.
- Added `edi_list_to_observe_dat()`: converts a list of `data_proc.load_edi`
  dicts to a FEMTIC `observe.dat`, handling unit conversion (mV/km/nT → SI Ω),
  z-sign negation, missing coordinates, and all-NaN site skipping.
- Added `_edi_Z_to_observe_row()` private helper for per-frequency flat-row
  packing with correct var/std error handling.
- Added `_Z_MT_TO_SI` module constant (μ₀ × 10³).
- Added `edi-to-observe` CLI subcommand with `--xy-csv`, `--manufacturer`,
  `--err-kind`, `--freq-order`, and `--preamble` options.

### Changelog (2026-04-11)
- Section 2 (matrix/roughness tools) replaced by `from ensembles import ...`.
  All implementations now live in `ensembles.py`; `femtic.py` re-exports them
  for backward compatibility. `check_sparse_matrix` moved to `ensembles.py`.
- Module docstring updated to reflect FEMTIC-specific I/O focus.

### Changelog (2026-07-17) — scipy.sparse: matrix → array API
- Migrated from legacy `scipy.sparse` matrix classes to the array-equivalent
  API: removed unused `isspmatrix` import (`issparse` covers both matrix and
  array sparse types); updated the `scipy.sparse.spmatrix` type hint to
  `scipy.sparse.sparray` in `sample_precision_gaussian_gmrf`. No functional
  change.

Author: Volker Rath (DIAS)
