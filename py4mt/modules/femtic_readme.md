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
- **Isotropic → anisotropic block conversion** — convert a v4 (isotropic-only)
  `resistivity_block_iterX.dat` to v5 (anisotropic) format, optionally
  marking selected regions transverse-isotropic or generally anisotropic
  (`convert_resistivity_block_to_anisotropic`).
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

# Pure v4 -> v5 format conversion (all regions stay isotropic)
python femtic.py iso-to-aniso-block \
    --in resistivity_block_iter0.dat \
    --out resistivity_block_iter0_aniso.dat

# Also mark selected regions transverse-isotropic (rhoXX = rhoYY, angles fixed at 0)
python femtic.py iso-to-aniso-block \
    --in resistivity_block_iter0.dat \
    --out resistivity_block_iter0_aniso.dat \
    --aniso-regions 257 258 259 \
    --aniso-type 1

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
| `convert_resistivity_block_to_anisotropic()` | Convert a v4 (isotropic) resistivity block to v5 (anisotropic) format; optionally mark regions TI/GA. |
| `read_distortion_file()` | Read FEMTIC galvanic distortion file.                |
| `read_resistivity_block()` | Parse resistivity block file → dict of arrays.     |
| `tet_volumes()` | Vectorised tetrahedral volume via scalar triple product (nelem,). |
| `build_region_geometry()` | Volume-weighted centroid and total volume for each free region. |
| `read_h5_block_volumes()` | Volume of every resistivity block, summed from the mesh stored in `results_iterX.h5` (index space of `/model/sensitivity/*`); checks it against `raw / volume_normalised`. |
| `ellipsoid_mask()` | Boolean mask for points inside a rotated ellipsoid (ZYX or SDS convention). |
| `brick_mask()` | Boolean mask for points inside a rotated rectangular prism. |
| `read_site_position()` | Return (x_m, y_m) model-local position for a site from observe.dat (4-6 token station lines). |
| `read_site_dat()`     | Parse a `mt_make_sitelist.py` CSV (name, lat, lon, elev, sitenum, easting, northing). |
| `estimate_utm_origin()` | Estimate mesh-centre UTM coordinates from calibration sites or bounding-box midpoint. |
| `extract_borehole_log()` | Sample ρ along a vertical borehole by point-in-element search. |

---

## observe.dat station-line layouts (OWNER_ELEMENT -1, dummy VTF)

Since 2026-10-22 the observe.dat reader accepts the station lines FEMTIC
uses when `control.dat` has `OWNER_ELEMENT -1` and/or `ELEC_FIELD -1`
(manual v4.3, slides 36-45), including the dummy-VTF layout written by
`femtic_add_dummy_vtf.py` (manual slides 55-58):

```
MT, PT :  ID  H-ID  [owner]  [E-type]  X_km  Y_km
VTF    :  ID  H-ID  [owner]            X_km  Y_km
```

`owner`: 0 = downward, 1 = upward element. `E-type`: 0 = horizontal,
1 = tangential electric field (not for VTF). A dummy VTF station has
`nfreq = 0` and no data rows.

| Tokens | Read as                                   |
|--------|-------------------------------------------|
| 4      | `ID H-ID X Y` (as before)                 |
| 5      | `ID H-ID owner X Y`                       |
| 6      | `ID H-ID owner E-type X Y` (MT, PT only)  |
| 4, `H-ID` not an integer | legacy `name x_m y_m z_m` (written by `edi_list_to_observe_dat`) |

A 5-token MT/PT line could also be `ID H-ID E-type X Y` (only
`ELEC_FIELD -1`). It is read as owner, unless the first VTF station of the
file has 4 tokens (then there is no owner column), or the caller says so.

```python
d = fem.read_observe_dat("observe.dat")                      # automatic
d = fem.read_observe_dat("observe.dat", elec_column=True,    # 5 tokens = E-type only,
                         owner_column=False)                 # no VTF block to tell
site = d["blocks"][0]["sites"][0]
site["header"]    # {'layout': 'femtic', 'name': '1', 'hid': 1001, 'owner': 0,
                  #  'elec': None, 'x_km': -2.671, 'y_km': 19.713, ...}
```

- The keywords `site_header_layout` (`"auto"`, `"femtic"`, `"legacy"`),
  `owner_column` and `elec_column` (`True`, `False`, `None` = detect) are
  accepted by `read_observe_dat`, `modify_data`, `convert_observe_dat`,
  `summarise_observe_dat` and `observe_to_site_viz_list`
  (`read_site_position` takes the last two).
- Station lines and `nfreq` lines are written back verbatim, so
  `modify_data` (and therefore `ensembles.generate_data_ensemble`) keeps the
  owner / E-type columns and the dummy stations.
- `observe_to_site_viz_list` skips stations with 0 frequencies
  (`skip_empty=True`, default) and adds `hid`, `owner`, `elec_type`, `x_km`,
  `y_km` to every entry.
- `summarise_observe_dat` reports `n_sites_empty` per block.
- Not supported: the block keywords `APP_RES_AND_PHS` and `HTF`; the reader
  stops with a message naming the block.
- `modify_data_fcn` (legacy, single block type) only recognises 5- and
  6-token station lines; use `modify_data` for files with a VTF block.

Tests: `python -I femtic_observe_io_test.py` (22 cases).

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
Sites without any usable frequency are skipped with a `UserWarning`.

### Invalid data (`invalid="drop"` / `"mark"`)

FEMTIC has **no marker value** for invalid data.  It uses a datum (each of the 8
real numbers Re/Im Zxx..Zyy of a frequency; likewise for VTF, PT, ...) only if its
error is `> 0`; the manual says "If SE is negative, the corresponding component is
not used in the inversion" (slides of the data formats).  The value itself is not
tested, and C++ stream input does not read `nan`/`inf` text, so non-finite numbers
must never be written.  (Checked in the public FEMTIC source, `src/ObservedDataStationMT.cpp`,
commit d858036 of 2026-10-08: residual, data ID and sensitivity row exist only for
`SD > 0`; check this in your own tree.)

`edi_list_to_observe_dat(..., invalid=..., invalid_se=-1.0, invalid_sentinel=1e30)`
offers two strategies:

| `invalid` | What is written |
|-----------|-----------------|
| `"drop"` (default; as before) | A frequency in which a value or an error of any component is not finite is **left out**.  Errors that are exactly 0 (a component that was invalid at all frequencies and zeroed by `load_edi`) are written as 0, which FEMTIC does not use either. |
| `"mark"` | The frequency **stays**.  A component with a non-finite value or error is written as value `0.0` with error `invalid_se`; a component with an error `<= 0` keeps its value and gets `invalid_se`.  The valid components of that frequency are kept. |

A frequency in which nothing is usable is left out in both cases, and a site without
any usable frequency is skipped.  Z or Z_err above `invalid_sentinel` (the 1e32 marker
of `save_edi`, kept as a finite number by `load_edi(drop_invalid_periods=False)`) count
as non-finite.  The returned dicts have `n_dropped` and `n_marked`.

With `load_edi(drop_invalid_periods=True)` (default) invalid periods are removed when
the EDI is read, so the strategies only matter for data that still contain invalid
values at this stage, or for zeroed components.

`mark` keeps more data; `drop` gives the smaller, simpler file.  The readers accept
negative errors: `read_observe_dat` / `observe_to_site_viz_list` do not perturb a
component with error `<= 0` in their bootstrap / Monte Carlo (numpy raises for a
negative scale; fixed 2026-11-20), `modify_data` leaves such a component alone, and
`summarise_observe_dat` reports `n_data_used` (errors `> 0`).

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

Scans `observe.dat` for a station line whose ID matches *site_number* and
returns `(x_m, y_m)` in model-local metres (converted from the km stored in
the file). Station lines may have an owner and/or E-type column
(see "observe.dat station-line layouts"); the first match wins, so for an
MT station and a dummy VTF station at the same place the MT line is used.
Before 2026-10-22 a station line with an owner column returned the owner
type as x.

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

### `read_femtic_mesh(mesh_path, swap_xy=None)`

The package's only mesh reader (`femtic_viz.py` wraps it). Returns
`(nodes, conn)`: node coordinates `[x, y, z]` (z positive down) exactly as
stored in `mesh.dat`, and 0-based tetra connectivity. `swap_xy=True` swaps
the first two columns after reading, for meshes stored as x = northing,
y = easting (so that `nodes[:, 0]` is the easting); `None` (default) uses
the module-level `MESH_SWAP_XY` (`False`), which a script can set once to
change the behaviour of every reader and plotter in the run. Element lines
need at least 5 columns (id + 4 node ids).

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

## Ensemble results, reference model and borehole sampling (Section 6c)

Added 2026-11-08 when `femtic_ens_post.py` was split into a calculation script
and the plot scripts `femtic_ens_plot_slices.py` and `femtic_ens_plot_bh.py`
(the nRMS histogram is the method `"nrms"` of `femtic_data_misfit.py`). These
are the pieces the scripts share. Nothing
here draws (drawing is in `femtic_viz.py`) and nothing needs the ensemble
member directories except where a block file is named.

### Reading the results file and the nRMS table

| Function | Purpose |
|---|---|
| `read_ensemble_results(path, prefix=None, *, percentiles=None, out=True)` | Read the results `.npz` of `femtic_ens_post.py` into a dict: `ens`, `pert` (perturbed models of the same members or `None`), `model_source`, `avg`/`var`/`err`/`med`/`mad`, `prc`/`prc_levels`, `qdiff`, `var_boot`/`err_boot`/`var_boot_se`, `sens` (`{suffix: vector}`), `sens_kinds`/`sens_suffixes` (the kinds the `sens` keys hold and their key suffixes, or `None` for files from before 2026-11-15), `sens_ranges` (`{kind: {stat: (min, max)}}`, the native range of each stored statistic, or `None` for files from before 2026-11-15 (c)), `simrc_coef`/`simrc_corr`, `diff_prior`/`prior`/`diff_prior_info` (the signed log10 difference model(ens) - model(prior), the prior itself and `[stat, source kind, text]`, or `None` for files from before 2026-11-15 (d) or when no prior could be used), `model_list`, `best_index`/`best_member`/`best_file`/`best_iter`/`best_nrms`, `nrms_table`, `nrms_max`, `perturbed_file`, `subsample` (`idx`, `n_accepted`, `seed`, or `None`). `prefix=None` detects the file's only ensemble. Statistics missing from the file are derived from the stored members. |
| `ensemble_stat_map(res)` | `(stat_map, value_keys)`: every plottable statistic `key -> (vector, description)` (avg, var, err, med, mad, `p<level>`, `qdiff_*`, bootstrap, `sens_*`, `simrc_*`, `diff_prior` = signed difference to the prior model and `abs_diff_prior` = its absolute value, made here) and the keys on the resistivity scale (usable as borehole lines). |
| `resolve_sens_stat(name, available, *, stored_kinds=None, stored_suffixes=None, kind="auto")` | `(key, note)`: the stored key that a `sens_*` panel / blanking name stands for, for the wanted sensitivity kind (`"auto"`, `"raw"`, `"volume_normalised"`; see `normalise_sens_kind`, `strip_sens_suffix`). `key` is `None` if nothing fits or the file holds only the other kind. Maps the dropped names `sens_<stat>_an` / `_na` too (see the 2026-11-15 (c) changelog). |
| `resolve_results_template(res, template=None, *, ocean=None)` | Block file supplying header / region structure for the panels: the best member's own file, or `template`. Returns `(path, from_vector)`; `from_vector` is `True` when the "best" panel must be rebuilt from the stored best vector. |
| `write_nrms_table(path, rows)` / `read_nrms_table(path)` | CSV of the per-member best nRMS (member, iteration, nRMS, status), sorted by nRMS. |

### Units, slices and the alpha mask

| Function | Purpose |
|---|---|
| `km_to_m`, `lim_km_to_m`, `slices_km_to_m`, `plane_spec_km_to_m`, `plane_point_km_to_m`, `polylines_km_to_m`, `markers_km_to_m` | `MOD_*` settings are in model-local km, `femtic_viz` wants metres; these convert scalars, limit pairs, `MOD_SLICES` lists and `kind="plane"` entries. The plane `point` may be model-local km (plain list or `([x, y, z], "model")`), `([E, N, z], "utm")` (E / N in UTM metres) or `([lon, lat, z], "latlon")` (degrees, longitude first); `z` is always km, depth positive down. The tagged UTM / latlon forms need the keyword arguments `zone`, `northern`, `origin_E`, `origin_N` of `plane_spec_km_to_m` (2026-11-09). `polylines_km_to_m(polylines)` converts the `MOD_POLYLINES` dict `{name: entry}` of `femtic_ens_plot_slices.py` for `fviz.plot_model_slices(polylines=...)`: `points` (model-local km, `z` = depth in km) are multiplied by 1000 (`x`, `y`, `z`), for `(pts, "utm")` only `z` (E / N stay UTM metres) and for `(pts, "latlon")` only `z` (degrees stay, longitude first); `max_dist` and `z_range` are km; `style` / `label` / `kinds` pass through; the input is not modified; `None` / `{}` -> `None`; `ValueError` / `TypeError` for malformed entries (2026-11-11). `markers_km_to_m(markers)` converts the `MOD_MAP_MARKERS` list of `femtic_ens_plot_slices.py` for `fviz.plot_model_slices(map_markers=...)`: `depth`, `depth_err` and `max_dist` of an entry are km (-> m); `latlon`, `marker`, `color`, `ms`, `name`, `kinds`, `is_model_centre` and all other keys pass through; the input is not modified; `None` / `[]` -> `[]`; `TypeError` for a non-list or a non-dict entry (2026-11-15). |
| `parse_alpha_source(expr)` | `"sens_median > -3."` -> `(name, op, value)`; regex, never `eval`. |
| `alpha_from_condition(arr, op, thr, mode, width)` | Per-free-parameter alpha in [0, 1] (`"blank"` hard cut, `"fade"` linear over `width`). |
| `write_alpha_block(alpha, template, path, ocean=None, out=True)` | Block file whose raw values are the alpha values (read by `plot_model_slices(alpha_mode="direct")`). |
| `STATS_AIR_RHO`, `STATS_AIR_LOG10_THRESH`, `STATS_OCEAN_RHO` | Sentinels (`1e30`, `29.0`, `1e-30`) written for air / ocean in statistics block files, so that a statistic near the ordinary air / ocean values is not painted as air / ocean. |
| `DEFAULT_PERCENTILES` | `(2.3, 15.9, 50.0, 84.1, 97.7)`, used when a file stores no levels. |

### Origin, sites, reference model, region of interest

| Function / class | Purpose |
|---|---|
| `resolve_origin_and_sites(*, site_dat, site_names, site_number, observe_file, origin_*, zone_override, origin_method, plot_sites)` | UTM origin of the model and the site list in model-local coordinates; same logic as `femtic_gst_prep.py` / `femtic_rto_prep.py` (incl. the `observe.dat` fallback). Returns `utm_e, utm_n, utm_lat, utm_lon, utm_zone, utm_northern, site_xys, obs_coords_only`. |
| `ReferenceModel(path, ocean=None, ocean_rho=...)` | Optional reference model on the ensemble's mesh. `active()`, `disable(why)`, `levels(clim, levels)`, `slice_kwargs(block_file, ...)` (iso-resistivity lines for `plot_model_slices`), `element_log10(n_elem)` (values per element for borehole lines). A missing file or another element count switches it off with a message. |
| `roi_free_mask(block_file, n_free, mesh_file, *, xlim_km, ylim_km, zlim_km, ocean=None)` | Mask of free parameters whose centroid lies in the ROI box (for the sensitivity ROI diagnostic / normalisation). `None` with a warning if the geometry cannot be read. |

### Borehole sampling

`sample_boreholes()` is the numerical half of the borehole figures; the
drawing is `femtic_viz.plan_borehole_figures()` / `draw_borehole_figures()`.

| Function / class | Purpose |
|---|---|
| `BoreholeMesh(...)` | Mesh arrays for the point search, read once. `column(x, y)`, `locate(x, y, depths, cand=None)` (barycentric point-in-tet search, once per borehole), `surface_m(...)`. |
| `expand_site_templates(specs, site_dat, utm_e, *, out=True)` | Expand `MOD_BH` entries with `sites=` (all / named / `n_random` with `seed`) to one entry per site. |
| `sample_boreholes(specs, *, mesh, struct, ens, stat_map, value_keys, best_i, percentiles, utm, cs, pert=None, ref_el=None, ...)` | Profiles of all statistics, members, best member, reference and heat maps along the boreholes (and of `pert` the same way). Returns `dict(profiles, npz, hdr, npz_pert, hdr_pert)`. |
| `borehole_resolve_xy`, `borehole_pos_str`, `borehole_expand_what`, `borehole_values`, `borehole_global_xlim`, `borehole_edges`, `borehole_heatmap`, `borehole_stat_vec`, `borehole_second_source`, `borehole_npz_entry`, `BOREHOLE_SIGMA_Q` | Building blocks of `sample_boreholes()`. |
| `heat_scale_defaults`, `heat_scale_check`, `heat_scale_read`, `heat_scale_write`, `heat_scale_init` | Heat-map colour-scale settings, and saving / reusing them as a JSON file across runs (same format as `femtic_ens_plot.py`). |

### Models of a single-model script

| Function | Purpose |
|---|---|
| `resolve_model_files(spec, iteration="last", *, cnv_name=None, block_name="resistivity_block_iter{n}.dat", out=True)` | `spec` = a block file, a run directory, a glob, or a list of them; for run directories `iteration` is `"last"` (last row of the convergence file), `"best"` (lowest nRMS), an integer / `"iterN"` (0 = starting model), or a list of those. Returns one dict per model: `file`, `dir`, `label` (unique), `tag` (`"iter21"`), `iter`, `nrms`, `alpha` (from the `.cnv` row of that iteration, or `None`), `selector`. Entries without a block file are skipped with a message. Used by `femtic_mod_plot_slice.py` and `femtic_mod_plot_bh.py`. |

### Directory scan report

| Function | Purpose |
|---|---|
| `build_scan_report(not_accepted, fallback, matches, n_scanned, n_accepted, ensemble_dir, ensemble_name, nrms_max, index_range=None)` | Markdown section accounting for every scanned directory and the reason each non-accepted one was left out, plus the directory-numbering check. |
| `append_scan_report(report_md, summary_path, script_file)` | Append it to the parameter-summary file (fallback: `<script>_scan_report.md`). Used by `femtic_ens_post.py` and `femtic_ens_repair.py` (which had its own copy of these helpers before 2026-11-08). |

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
| `femtic_ens_post.py`  | Ensemble statistics, sensitivity, covariance; writes the results `.npz` read by `read_ensemble_results()`. |
| `femtic_ens_plot_slices.py`, `femtic_ens_plot_bh.py`, `femtic_data_misfit.py` (method `"nrms"`) | Slice, borehole and nRMS-histogram figures from that `.npz`; use Section 6c and the new `femtic_viz.py` functions. |
| `femtic_mod_plot_slice.py`, `femtic_mod_plot_bh.py` | Single-model figures; with a list / glob / run directories, one figure per model (`resolve_model_files()`). |
| `femtic_mod_plot.py`  | Read and plot slice panels of a FEMTIC model; calls `fem.*` helpers. |
| `snippets.py`         | Optional code blocks for `femtic_mod_plot.py` (e.g. ensemble plot).  |
| `util.py`             | General-purpose helpers.                                             |
| `femtic_rto_rough.py` | Extract roughness matrix from FEMTIC.                               |
| `femtic_rto_prior.py` | Build prior covariance proxy.                                       |
| `femtic_rto_prep.py`  | Generate RTO ensemble.                                              |
| `femtic_summarize_model.py` | CLI: problem-size and model-size summary for `resistivity_block_iterXX.dat` (tetrahedral and hexahedral meshes). Parses the files itself; does not import `femtic.py`. Renamed from `femtic_summarize_model_cells.py`. |
| `femtic_summarize_observe_dat.py` | CLI: data-content summary for `observe.dat`. Delegates to `summarise_observe_dat()`; self-contained fallback when `femtic.py` is not on the path. |

---

## Version / provenance

Updated: 2026-11-20

### Changelog (2026-11-20, b) --- invalid data in `observe.dat`: `drop` or `mark`
- `edi_list_to_observe_dat(..., invalid="drop"|"mark", invalid_se=-1.0,
  invalid_sentinel=1e30)`.  FEMTIC uses a datum only if its error is `> 0` (negative
  error = "not used", manual), so a component can be switched off without leaving out
  the frequency: `"mark"` writes it with `invalid_se` (value 0.0 if it was not
  finite); `"drop"` (default, as before) leaves the frequency out.  Non-finite
  *errors* now also drop the frequency (before, `nan` could be written, which FEMTIC
  cannot read).  Z / Z_err above `invalid_sentinel` (1e32 markers) are invalid; before,
  they were written as huge valid numbers.  The returned dicts have `n_dropped` and
  `n_marked`.
- `_bootstrap_mt_derived` and `_mt_rhoa_phase_from_Z` (used by `read_observe_dat` and
  `observe_to_site_viz_list`) passed the error as `scale` to numpy, which raises for a
  negative value: they now do not perturb a component with an error `<= 0` or
  non-finite.
- `summarise_observe_dat` reports `n_data_used` (errors `> 0`) next to `n_data_total`.
- `mt_data_processor.py`: `OBS_INVALID`, `OBS_INVALID_SE`.
- Tests: `test_data_proc_freq_ft.py`, class `TestObserveInvalid` (drop, mark, zero and
  negative errors, 1e32 markers, custom marker, validation, readers, `modify_data`);
  also run end to end through the script with injected invalid data.  AI-generated
  (Claude Sonnet 5.5, Anthropic); review before production use.

### Changelog (2026-11-20) --- two errors in `edi_list_to_observe_dat()` fixed
- **Error scale**: for site dicts with `err_kind="var"` (the default of
  `data_proc.load_edi`) the impedance variances were converted to SI with the unit
  factor `c = mu0 * 1e3 = 1.2566e-3` instead of `c**2`, so the standard deviations
  written to `observe.dat` were too large by `1/sqrt(c)` = 28.2.  The `"std"` case
  was correct.  Now variances are scaled with `c**2`, standard deviations with `c`.
- **Imaginary errors**: real-valued `Z_err` was cast to complex, so the errors of
  Im(Z) were written as 0.  Real errors stay real; the same value is used for the
  real and the imaginary part.
- Checked against the processed data: maximal relative deviation of the data and the
  errors in `observe.dat` about 5e-9 (tests in `test_data_proc_freq_ft.py`, class
  `TestObserve`: var and std, Im error not 0, dummy-VTF layout).
- **`observe.dat` files made earlier with this function have wrong errors (28.2 times
  too large for var-kind dicts, Im errors 0 for real-valued errors); write them again.**
- `mt_data_processor.py` now writes `observe.dat` directly (token `observe` in
  `OUT_FILES`, optionally with the dummy VTF stations).  AI-generated (Claude
  Sonnet 5.5, Anthropic); review before production use.

### Changelog (2026-11-15) --- block volumes from `results_iterX.h5`
- New `read_h5_block_volumes(h5_path, *, check_sensitivity=True, rtol=1e-4,
  out=True)` (Section 6a, after `read_results_model`). FEMTIC does not write
  the volume of a resistivity block, but `results_iterX.h5` holds the mesh
  (`/model/mesh/node_coords`, `/model/mesh/elem_nodes`) and the
  `/model/element_block_map`, so the function sums `tet_volumes()` per block.
  Returns a dict: `volume` (`(nBlocks,)`, m^3, in the index space of the
  `/model/sensitivity/*` datasets, fixed blocks included, 0 for a block without
  elements), `n_elem`, `n_blocks`, `n_empty`, `n_unassigned` (elements with a
  block index outside `0..nBlocks-1`, not counted) and `check`.
- `check`: when both `/model/sensitivity/raw` and `.../volume_normalised` are
  present, `raw / volume_normalised` is compared with the mesh volumes for all
  blocks where both are positive; the dict holds `n_compared`, `max_rel_dev`
  (`max |raw/vn/vol - 1|`), `ratio_median` and `ok` (`max_rel_dev <= rtol`).
  A failed check only prints a WARNING. A constant `ratio_median != 1` points to
  a unit difference between the two; a varying ratio to an element-block map that
  is not in the index space of the sensitivity datasets (e.g. a permuted
  `blockID`).
- Only tetrahedral meshes (first four columns of `elem_nodes` are the
  vertices, 0-based). `ValueError` for inconsistent mesh arrays (wrong shape,
  node indices out of range, element counts that differ), `KeyError` if the mesh
  or the block map is missing, `FileNotFoundError` for a missing file.
- Used by `femtic_ens_post.py` (`SENS_STORE_VOLUMES`, `.npz` key
  `<P>_cell_volume`), which maps the result to the free parameters with the same
  function as the sensitivities. No existing function changed.
- Tested against an in-memory `h5py` stand-in only (`h5py` was not installed
  where this was written): consistent file, unit mismatch, missing sensitivity,
  datasets of different length, bad node indices, bad element count, 3-column
  connectivity, missing file, and inside a full run of `femtic_ens_post.py` on a
  synthetic ensemble. Not yet run against a real `results_iterX.h5`: the
  `check` line it prints is the test on real data. AI-generated (Claude
  Sonnet 5.5, Anthropic); review before production use.

### Changelog (2026-11-15, b) --- raw or volume-normalised sensitivity in the plot scripts
- `femtic_ens_post.py` (`SENS_KIND = "both"`) stores the sensitivity statistics
  of both kinds, with the key suffix `_cum` (raw) / `_vn` (volume_normalised);
  a single kind keeps the plain keys. The plot side needs to know which keys a
  file holds and to turn a name like `sens_mean` into the right key:
- `read_ensemble_results()` also returns `sens_kinds` (list of
  `"raw"` / `"volume_normalised"`) and `sens_suffixes` (`["_cum", "_vn"]`, or
  `[""]` for one kind), read from `<P>_sensitivity_kinds` /
  `<P>_sensitivity_key_suffixes`; both are `None` for results files from before
  this date. `ensemble_stat_map()` names the kind in the description of a
  suffixed `sens_*` key. Everything else it returns is as before.
- New: `resolve_sens_stat(name, available, *, stored_kinds, stored_suffixes,
  kind="auto")` -> `(key, note)`. An available name is used as it is (a plain
  name of a single-kind file is checked against `kind`; a file that does not
  record its kind is used as it is, with a note if a kind other than `"auto"`
  was asked for); a plain name that exists only with suffixes takes the suffix
  of `kind` (`"auto"` = `SENS_KIND_DEFAULT_BOTH` = volume_normalised); anything
  else gives `key=None`, with a note when the reason is a kind the file does
  not hold. Helpers: `normalise_sens_kind()` (`None` -> `"auto"`,
  `"volume_normalized"` accepted, `ValueError` otherwise),
  `strip_sens_suffix()`, constants `SENS_KIND_SUFFIX`, `SENS_KIND_SPELLING`,
  `SENS_KIND_DEFAULT_BOTH`.
- Used by `femtic_ens_plot_slices.py` (`MOD_SENS_KIND`). Tested on synthetic
  results only (single-kind raw / volume_normalised, both, file without the
  kind keys, subsample file): resolution rules, mismatch refusals, and the plot
  script's panel list / blanking / names with the drawing stubbed out. AI-generated
  (Claude Sonnet 5.5, Anthropic); review before production use.

### Changelog (2026-11-15, c) --- one normalisation, ranges of the sensitivity statistics
- `femtic_ens_post.py` now takes the ensemble statistics of the native
  sensitivity and divides each (`mean`, `median`, `min`, `max`, `std`) by **its
  own max** over the free blocks just before storing it (log10, each peaks at
  0); `cv = std / mean` stays linear. The keys are `sens_{mean,median,min,max,
  std,cv}` (+ `_cum` / `_vn` for a file with both kinds). The two versions
  `sens_*_na` / `sens_*_an` and `sens_mean_raw` are gone.
- `read_ensemble_results()` returns `sens_ranges = {kind: {stat: (min, max)}}`,
  the native [min, max] over the free blocks of each statistic (read from
  `<P>_sensitivity_native_range` / `<P>_sensitivity_range_stats`; `NaN` where
  there is none); native value of a stored key = `10**key * max`. `None` for
  results files from before this date. (The structure with `scale`,
  `member_max` and `member_rows` of the intermediate delivery earlier the
  same day, `py4mt_femtic_sens_norm_ranges_20261115.zip`, is replaced; files
  written by that version are read with `sens_ranges = None` semantics only
  where the new keys are missing.)
- `resolve_sens_stat()` maps the dropped names: `sens_<stat>_an[_cum|_vn]`
  asked of a new file is read as the plain statistic (note: same statistic,
  divided by its own max), `_na` gives `None` with a note; a new plain name
  asked of an OLD file is read from its `_an` key (note: there all statistics
  were divided by the max of the mean). `ensemble_stat_map()` descriptions name
  the kind of a suffixed key. `parse_alpha_source()`'s error example is now
  `'sens_median > -3.'`.
- Needs `femtic_ens_post.py` and `femtic_ens_plot_slices.py` of the same date.
  Tested on synthetic results only (old-format files written by the previous
  post script included). AI-generated (Claude Sonnet 5.5, Anthropic); review
  before production use.

### Changelog (2026-11-15, g) --- difference to the prior model
- `femtic_ens_post.py` of the same date (its changelog (d)) stores
  `<P>_diff_prior` = model(ens) - model(prior) in log10(Ohm.m), **signed**,
  with `<P>_prior` and `<P>_diff_prior_info`.
  `read_ensemble_results()` returns them as `res["diff_prior"]`,
  `res["prior"]`, `res["diff_prior_info"]` (`None` for older files, and for a
  `diff_prior` whose length is not the number of free parameters, which is
  ignored). `ensemble_stat_map()` offers two plot names: `"diff_prior"` (the
  stored array) and `"abs_diff_prior"` (`abs()` of it: the post script stores
  no absolute value). Neither is a resistivity-scale key. The description
  names the statistic and the prior, e.g. "log10 mean(ens) - prior
  (reference), signed". `select_ensemble_source()` drops all three when the
  perturbed models alone are drawn (the difference belongs to the converged
  ensemble). Nothing else changed; files without the keys read as before.
- Needs `femtic_ens_post.py` and `femtic_ens_plot_slices.py` of the same date
  for the keys and the styles. Tested on synthetic results (reader, stat map,
  source swap, a file with a wrong-length key, a real drawing run);
  not on your data. AI-generated (Claude Sonnet 5.5, Anthropic); review before
  production use.

### Changelog (2026-11-15, f) --- point markers in km
- New `markers_km_to_m(markers)` (Section 6c, after `polylines_km_to_m`): the
  km scripts keep the `depth`, `depth_err` and `max_dist` of a `MOD_MAP_MARKERS`
  entry in km; `femtic_viz.plot_model_slices(map_markers=...)` wants metres
  (it now draws markers with a depth on the vertical sections too, see its
  readme, "Point markers"). The function multiplies those three keys by 1000
  and passes everything else through; the input is not modified.

### Changelog (2026-11-15, e) --- `alpha_outside` fades
- Documentation only. The `kind="profile"` entry takes one more key that passes
  through `resolve_slice_positions()`: `fade_km` (km, default 2.0). Outside the
  stretch between its two defining points the fault's opacity now falls
  linearly to `alpha_outside` over `fade_km` instead of stepping to it (like
  `alpha_mode="fade"`); `0` = the abrupt step of (d). Handled by `femtic_viz.py`
  of the same date, see its readme, "Limit to the defining points". No code
  change in `femtic.py`.

### Changelog (2026-11-15, d) --- fault limited to its defining points
- Documentation only. The two keys `limit_km` and `alpha_outside` of a
  `kind="profile"` slice entry (fault from two surface points) pass through
  `resolve_slice_positions()` like `overlay` and `panel`; the resolved plane
  spec already carries the end points as `p1_model` / `p2_model`, which
  `femtic_viz.py` (same date) uses to draw the part of the fault between the
  points as given and the rest fainter. See the `femtic_viz.py` readme, "Fault
  planes". No code change in `femtic.py`.

### Changelog (2026-11-12) --- fault plane from two surface points
- `resolve_pos_two_point_profile(..., dip=90.0)`: new keyword `dip`. The
  default (90) gives the vertical fence section exactly as before. With
  `0 < dip < 180`, `dip != 90`, the two end points define a **dipping plane**
  (a fault from two surface points and a dip): `strike` = azimuth p1 -> p2,
  `point` = midpoint of p1-p2 at depth `z_top`, the plane dips towards
  `strike + 90` (to the right of p1 -> p2; `dip = 180 - d` dips `d` degrees
  to the left), and the returned spec carries the down-dip range `ylim =
  [0, (z_bot - z_top) / sin(dip)]` (m, relative to `point`) instead of
  `zlim`. `dip` outside `(0, 180)` raises `ValueError`.
- `resolve_slice_positions()`: a `kind="profile"` spec may carry `dip` (it
  was ignored before; default 90 = as before); other keys such as `overlay`
  (see the `femtic_viz.py` readme, "Fault planes") pass through. The verbose
  output names the dip direction of a dipping profile.
- Slice specs may also carry `panel=False` (passes through unchanged): with
  `overlay` the fault trace is then drawn in all the other slices without a
  fault panel of its own (`femtic_viz.py`, "Fault planes"; works in
  `plot_model_slices` and `plot_ensemble_slices`).
- Nothing else changed. Needs the `femtic_viz.py` of the same date for the
  overlay. Tests: `test_femtic_fault_plane.py`. AI-generated (Claude Sonnet 5.5, Anthropic); review before production use.

### Changelog (2026-11-11) --- polylines for the slice plots
- New `polylines_km_to_m()` (Section 6c, next to `plane_spec_km_to_m()`):
  converts the km `MOD_POLYLINES` dict of `femtic_ens_plot_slices.py` to the
  metre dict of `femtic_viz.plot_model_slices(polylines=...)`. Model-local
  points: x, y, z km -> m; `(pts, "utm")`: only z km -> m (E, N are UTM
  metres); `(pts, "latlon")`: only z km -> m (degrees stay); `max_dist` and
  `z_range` km -> m. The UTM / latlon -> model-local conversion itself is done
  by `plot_model_slices`, which knows the zone and the origin.
- No existing function was changed. Needs the `femtic_viz.py` of the same
  date (see its readme, "Polylines" and "Intersection of two slices").
  AI-generated (Claude Sonnet 5.5, Anthropic); review before production use.

### Changelog (2026-11-09) --- plane `point` in latlon / utm / model
- `plane_spec_km_to_m()` accepts a CRS-tagged `point`: `([lon, lat, z],
  "latlon")`, `([E, N, z], "utm")` (UTM metres) or `([x, y, z], "model")`
  (km), z in km and depth positive down; the new helper
  `plane_point_km_to_m()` does the conversion. The tagged UTM / latlon forms
  need the new keyword arguments `zone`, `northern`, `origin_E`, `origin_N`
  (a ValueError says so when they are missing). Plain `[x, y(, z)]` lists
  and all other behaviour are unchanged.
- `plane_spec_km_to_m()` also converts an optional spec `ylim` (down-dip
  range of a dipping plane, km relative to `point`); the default along-strike
  range of a vertical plane now uses the resolved point.
- Needs the `femtic_viz.py` of the same date for dipping planes (see its
  readme). AI-generated (Claude Sonnet 5.5, Anthropic); review before
  production use.

### Changelog (2026-11-08) --- ensemble results, reference model, borehole sampling
- New Section 6c (code moved out of `femtic_ens_post.py` and generalised so
  the plot scripts can share it): `read_ensemble_results`,
  `ensemble_stat_map`, `resolve_results_template`, `write_nrms_table`,
  `read_nrms_table`, the km/m helpers, the alpha helpers
  (`parse_alpha_source`, `alpha_from_condition`, `write_alpha_block`),
  `resolve_origin_and_sites`, `ReferenceModel`, `roi_free_mask`,
  `BoreholeMesh`, `expand_site_templates`, `sample_boreholes` with the
  `borehole_*` and `heat_scale_*` helpers, and `build_scan_report` /
  `append_scan_report`. Constants `STATS_AIR_RHO`, `STATS_AIR_LOG10_THRESH`,
  `STATS_OCEAN_RHO`, `DEFAULT_PERCENTILES`.
- `read_ensemble_results` also returns `nrms_table`, `nrms_max`,
  `perturbed_file` and `subsample` (the random-draw bookkeeping written by
  `femtic_ens_post.py` with `ENS_SUBSAMPLE_N`).
- No existing function was changed.
- Later the same day: new `resolve_model_files()` (section "Models of a
  single-model script" above), for the batch option of
  `femtic_mod_plot_slice.py` / `femtic_mod_plot_bh.py`. The plot scripts that
  read the results file are now `femtic_ens_plot_slices.py`,
  `femtic_ens_plot_bh.py` and, for the nRMS histogram, `femtic_data_misfit.py`
  (a separate `femtic_ens_plot_misfit.py` was merged into it).

### Changelog (2026-10-28)
- Documentation only, no change to `femtic.py`: in "Related modules and
  scripts", the CLI summary script is now listed as
  `femtic_summarize_model.py` (renamed from
  `femtic_summarize_model_cells.py` by Claude Sonnet 5.5 (Anthropic),
  2026-10-28, via the interim name `summarize_model.py`). The old row text (delegates to
  `summarise_model_file()`, self-contained fallback) no longer described the
  script, which parses the files itself and does not import `femtic.py`.
  The older changelog entries below keep the former name, as it was then.

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

### Changelog (2026-09-02) --- header-driven femtic.cnv parsing
- Added `read_cnv(source)`: reads a FEMTIC convergence file (`femtic.cnv`)
  by parsing its own header row (e.g. `Iter#  Retrial#  Alpha  Beta  Damp
  Roughness  Distortion  Misfit  RMS  ObjFunc`) into a name -> column-index
  map, instead of assuming fixed column positions. The `Beta`/`Distortion`
  columns only appear when distortion parameters are being inverted, which
  shifts every column after them (`RMS` at index 6 without distortion,
  index 8 with it) independently of the FEMTIC version number -- a
  4.3-with-distortion run has the same column layout as a typical 5.x run,
  not the same layout as plain 4.3.
- `get_nrms()` rewritten to use `read_cnv()` internally. It previously
  hardcoded `misft=nline[7]`, `nrmse=nline[8]` unconditionally (correct
  only for the with-distortion layout, silently wrong for
  without-distortion runs). Return values and the empty-file fallback
  (`(-1, 1e32)` with a printed message) are unchanged; a missing
  `femtic.cnv` still raises `FileNotFoundError` as before.
- Also removed a stray leftover `print(nline)` debug statement inside the
  old parsing loop, dropped along with the loop it lived in.
- `femtic_ens_post.py`'s inline best-iteration scan (previously its own
  `"4.3"`/`"5."` version-string switch on `nrms_col`, with the same
  distortion-blind bug) now calls `fem.read_cnv()` too -- see
  `femtic_ens_post_readme.md`'s matching changelog entry.

### Changelog (2026-09-02b) --- read_cnv columnar option
- Added `columnar` parameter to `read_cnv()` (default `False`, backward
  compatible -- return dict keys are unchanged unless requested). When
  `columnar=True`, an additional `"data"` entry is returned: a dict mapping
  each header column name to a 1-D `np.ndarray` of that column's values
  over the whole convergence history (row order preserved), e.g.
  `cnv["data"]["RMS"]` for the full nRMS trace, without having to loop
  over `"rows"` and pull one field out of each dict by hand -- convenient
  for convergence-curve plotting and similar whole-column access.

### Changelog (2026-09-07) --- read_cnv: case-insensitive, substring column matching

- `read_cnv()`'s header-row detection and column-name mapping previously
  required an exact, case-sensitive token match (header-row detection:
  literal `"RMS"` in the split tokens; column naming: the raw header
  token verbatim, e.g. `"Roughness"`). Both now use a case-insensitive
  **substring** test against a canonical pattern list
  (`_CNV_CANONICAL_PATTERNS`, e.g. `"rough"` -> `"Roughness"`, `"rms"` ->
  `"RMS"`, `"iter"` -> `"Iter"`, ...; also covers the ABIC/cross-gradient
  columns `LmdCG`/`CrossGra`/`MupdateMean`/`ABIC`). Header-row detection
  now requires both an "iter"-like and an "rms"-like token (either case),
  rather than the literal token `"RMS"`. A header token that matches no
  known pattern is kept verbatim (trailing `#` stripped), so a genuinely
  new future column is still captured rather than silently dropped. This
  guards against a header spelled or capitalised differently in a future
  FEMTIC version (e.g. `"roughness"` lower-case, or `"Rough"`
  abbreviated) silently breaking column identification the same way the
  original hardcoded-index bugs did.
- Prompted by three still-unmigrated callers (`femtic_ens_plot.py`,
  `femtic_ens_repair.py`, `femtic_lcurve_plot.py`) each retaining their
  own pre-`read_cnv()` hardcoded/version-switch or column-count column
  lookup, which is exactly the kind of column mismatch `read_cnv()` was
  built to eliminate. All three now call `fem.read_cnv()` directly and
  had their `FEMTIC`/`DISTORTION` config variables removed; see their
  own READMEs' 2026-09-07 changelog entries.

### Changelog (2026-09-08) --- sensitivity-ensemble support functions

Two new functions added to support `femtic_ens_post.py`'s new
`COMPUTE_SENS`/`COMPUTE_SIMRC` sensitivity diagnostics, which mitigate the
ambiguity that low RTO/GST ensemble spread can reflect either genuine
data control or a regularisation-collapsed null-space cell.

- Added `read_h5_sensitivity(h5_path, group=None, cumsens_key=..., 
  jacobian_key=..., error_key=..., normalize=False, out=True)`: reads or
  derives a per-free-parameter cumulative sensitivity vector from a
  `model_iterX.h5` file. Tries a precomputed 1-D `cumsens_key` dataset
  first; falls back to summing `|Jacobian|` columns from a 2-D
  `jacobian_key` dataset, optionally weighted by a 1-D `error_key`
  per-datum error vector (cf. Christiansen & Auken, 2012, *Geophysics*
  77, WB171, doi:10.1190/geo2011-0393.1; Oldenburg & Li, 1999,
  *Geophysics* 64, 403-416, doi:10.1190/1.1444545). **Schema is
  provisional** -- no `model_iterX.h5` writer exists elsewhere in the
  codebase yet, so every dataset name is a configurable argument rather
  than hardcoded; adjust once the actual FEMTIC HDF5 export settles (see
  "HDF5 archive schema compatibility with FEMTIC v5" -- open item).
- Added `collect_forward_response(result_files, site_file,
  data_kind="rhophas", out=True)`: flattens one ensemble member's
  calculated ("cal") forward response across whichever `result_XXX.txt`
  files exist for it (MT/VTF/PT) into a single 1-D vector plus a
  `(data_type, n_values)` manifest, via `get_femtic_data()`. Used to
  build a per-member forward-response ensemble for the "SimRC"
  ensemble-native sensitivity measure (Bobe, Keller & Van De Vijver,
  2021, *Geophysical Prospecting*, doi:10.1111/1365-2478.13068), which
  needs no Jacobian at all -- it correlates the model ensemble directly
  against the forward-response ensemble.

Both are consumed by `femtic_ens_post.py`'s main scan loop; see
`femtic_ens_post_readme.md`'s matching 2026-09-08 entry for the full
pipeline (aggregation across members, the combined
`flag_null_space` diagnostic, `.npz` output keys, and `MOD_STATS`
plotting).

### Changelog (2026-09-11) --- iso-to-anisotropic resistivity-block conversion

Added by Claude Sonnet 5 (Anthropic), 2026-09-11. AI-generated code --
please review before production use.

- Added `convert_resistivity_block_to_anisotropic(model_file_in,
  model_file_out, *, aniso_regions=None, aniso_type=_ANISO_TI,
  rhoYY_ratio=1.0, rhoZZ_ratio=1.0, strike=0.0, dip=0.0, slant=0.0,
  fix_strike=True, fix_dip=True, fix_slant=True,
  fix_rho_from_flag=True, out=True)`: converts a v4 (isotropic-only)
  `resistivity_block_iterX.dat` into v5 (anisotropic) format, for use
  with anisotropy-capable FEMTIC builds. Reuses the existing
  `_parse_region_line_v4()` / `_format_region_line_v5()` / `_ANISO_*`
  infrastructure from the 2026-06-22 v5-format-support entry above
  rather than duplicating parsing/formatting logic.
  - Default (`aniso_regions=None`): pure format conversion -- every
    region is written as `aniso_type=0` (v5-isotropic) with the same
    resistivity, bounds, and fix flag as the input; the model is
    physically unchanged.
  - `aniso_regions=[...]`: additionally marks the given 0-based region
    indices as transverse-isotropic (`aniso_type=1`, default) or
    general anisotropy (`aniso_type=2`), using `rhoYY = rhoXX *
    rhoYY_ratio` (and, for general anisotropy, `rhoZZ = rhoXX *
    rhoZZ_ratio`) and the given rotation angles. With the defaults
    (`*_ratio=1.0`, angles `0.0`), marked regions stay numerically
    isotropic but are flagged as anisotropic cell types -- matching how
    the boundary regions in the shipped `FemticAniso.zip` anisotropic
    example are marked.
  - Raises `ValueError` on an already-v5 input file, an unsupported
    `aniso_type`, or an out-of-range region index in `aniso_regions`.
  - Also exposed as CLI subcommand `iso-to-aniso-block` (`--in`,
    `--out`, `--aniso-regions`, `--aniso-type`, `--rhoyy-ratio`,
    `--rhozz-ratio`, `--strike`, `--dip`, `--slant`, `--free-strike`,
    `--free-dip`, `--free-slant`).
  - Inserted immediately after `insert_model()`.
- **Verification**: run against the `Isotropic/resistivity_block_iter0.dat`
  / `Anisotropic/resistivity_block_iter0.dat` example pair shipped in
  `FemticAniso.zip` (32 boundary regions marked TI, matching the
  example). A field-by-field comparison of the parsed v5 structures
  showed the converted file is identical to the shipped anisotropic
  example, except for the air region's resistivity (1e9 in the
  isotropic source vs. 1e10 in the shipped anisotropic example) --
  an incidental difference specific to that demo file, not a format
  rule, so it is not reproduced automatically.
  - **Caveat**: the v5 region-line layout was reverse-engineered from
    the `FemticAniso.zip` example files and
    `Manual_Of_FEMTIC_Anisotropic.pdf` slides (not from FEMTIC's C++
    source), per the 2026-06-22 entry above. Please double-check
    against a fresh FEMTIC-anisotropic run before relying on it in
    production.
- Updated: `Key data-handling functions` table, `Overview`,
  `Command-line interface` section.

### Changelog (2026-09-13) --- results_iterX.h5 readers

Added by Claude Sonnet 5 (Anthropic), 2026-09-13, as the Python side of
merging FEMTIC's C++ `model_iterX.h5` + `data_iterX.h5` HDF5 output into one
`results_iterX.h5` per iteration (as sibling `/model`, `/data`, `/distortion`
groups; `/data` rows also gained `cal_re`/`cal_im` calculated-response
fields alongside the existing observed `re_val`/`im_val`/`re_err`/`im_err`).
This resolves the "HDF5 archive schema compatibility with FEMTIC v5" open
item that `read_h5_sensitivity()`'s 2026-09-08 entry (above) flagged: that
function's schema was provisional because no writer existed yet at the time.

- Added `read_results_model(source, *, out=True)`: reads the `/model` group
  -- metadata, `element_block_map`, `blocks`, mesh node/element arrays, and
  the two sensitivity datasets (`raw`, `volume_normalised`) when present.
- Added `read_results_data(source, *, out=True, decode_labels=True)`: reads
  the `/data` group's compound row array, plus (when `decode_labels=True`,
  the default) human-readable `datatype_name`/`component_name` string
  arrays derived from the new `RESULTS_H5_DATATYPE_NAMES`/
  `RESULTS_H5_COMPONENT_NAMES` lookup tables.
- Added `read_results_distortion(source, *, out=True)`: reads the
  `/distortion` group (site IDs, `param1..param4`, `isFixed`, and a decoded
  `type_name`); returns `None` when the run had no distortion estimation
  (there is no pre-2026-09-13 equivalent for this group).
- Added `read_results_hdf5(h5_path, *, out=True)`: the umbrella reader --
  opens the file once and returns `{"model":..., "data":..., "distortion":...}`,
  each `None` if that group is absent.
- All four transparently accept either a path or an already-open
  `h5py.File`/`h5py.Group` (the umbrella reader opens once and passes the
  handle to each sub-reader), and all four fall back to the older,
  standalone `model_iterX.h5`/`data_iterX.h5` layout (root-level content, no
  `/distortion`) when pointed at a pre-2026-09-13 file -- **except** that
  `read_results_data(decode_labels=True)` (the default) raises `KeyError` on
  such a file, since `cal_re`/`cal_im` cannot be reconstructed after the
  fact; pass `decode_labels=False` for observed-data-only access to old
  files.
- Updated `read_h5_sensitivity()` (2026-09-08 entry, above) now that the
  schema it called provisional is settled: `cumsens_key` now defaults to
  `"raw"` and is looked up automatically under `model/sensitivity` (in
  addition to the explicit `group` and the file root) even with
  `group=None`, so `read_h5_sensitivity("results_iter7.h5")` with no other
  arguments works against FEMTIC's own output; `error_key` now defaults to
  `"data_errors"` (plural, matching `jacobian.h5`'s real dataset name --
  the previous default, `"data_error"` singular, never matched any real
  file).
- **Verification**: h5py was not installable in the authoring environment
  (no network access), so this was tested against a minimal in-memory
  mock of h5py's `File`/`Group`/`Dataset`/`attrs` API surface rather than a
  real HDF5 file. Covered: the new combined-file schema end-to-end (model +
  data + distortion, including label decoding); both legacy fallback
  shapes; the `decode_labels=True` `KeyError` on old data files; and
  `read_h5_sensitivity()`'s new automatic `model/sensitivity` fallback plus
  the `jacobian.h5`-with-error-weighting path. One real bug was caught this
  way and fixed before it shipped: a legacy `data_iterX.h5` has a top-level
  DATASET literally named `data` (not a group), which the naive "does
  `h5["data"]` exist" check would have silently mis-detected as the new
  `/data` GROUP, only to fail confusingly several lines later when the
  reader tried to index into it like one -- `_h5_find_group()`'s group
  lookup now explicitly checks `isinstance(..., h5py.Group)` to rule this
  out. **Not yet re-verified against a real FEMTIC-written file on disk** --
  run `read_results_hdf5()` against an actual run's output as a final
  check; it prints a one-line summary of everything it found.
- Updated: module docstring changelog, `Overview` bullet list.

Author: Volker Rath (DIAS)

### Changelog (2026-10-08) --- configurable convergence file name
- `read_cnv(source, *, columnar=False, filename="femtic.cnv")` and
  `get_nrms(directory, filename="femtic.cnv")` accept a different file name
  (or a list/tuple of candidates; first existing wins). New helper
  `resolve_cnv_path(directory, filename)`. Defaults unchanged. AI-generated
  (Claude Sonnet 5.5, Anthropic); review before production use.

### Changelog (2026-11-11c) --- model source of the stand-alone borehole scripts
- New `ENS_MODEL_SOURCES = ("auto", "converged", "perturbed", "both")` and
  `select_ensemble_source(res, source="auto", *, out=True)`: applies the setting
  `ENS_MODEL_SOURCE` of `femtic_ens_plot_bh.py` / `femtic_ens_plot_bh_SV.py` to
  the dict of `read_ensemble_results()`. `"auto"` (or `None`) returns it
  unchanged; `"converged"` drops the perturbed models of a `"both"` file;
  `"perturbed"` makes them the ensemble (`avg`, `var`, `err`, `med`, `mad`,
  `prc` recomputed; `qdiff`, bootstrap, `sens` and SimRC dropped;
  `res["source_swapped"] = True`); `"both"` needs a `"both"` file. A choice the
  file cannot give raises a `ValueError` that says how the file was made. The
  input dict is not changed.
- Tests: `test_femtic_ens_source.py`, 17 cases. AI-generated (Claude Sonnet
  5.5, Anthropic); review before production use.

### Changelog (2026-11-11b) --- one way to find and read the convergence file
- New `CNV_DEFAULT_NAMES = ("femtic.cnv", "FEMTIC-DABIC.cnv")` and
  `cnv_names(names=None)`: a setting that is one name, a list of names or
  `None` (= the defaults) becomes a list (empty and repeated entries dropped).
- New `find_cnv_file(directory, names=None)` returns the first existing file
  of the list as a `Path`, or `None`; if no name exists exactly, the names are
  tried again ignoring upper / lower case. An absolute name is used as it is.
  `resolve_cnv_path()` returns that file, or the path of the first name.
- `read_cnv`, `get_nrms` and `resolve_model_files` (`cnv_name`) use the same
  finder; with no name given they now also find `FEMTIC-DABIC.cnv` (before:
  `femtic.cnv` only). A directory without any of the files gives a
  `FileNotFoundError` that names the candidates.
- New `read_cnv_last(source, filename=None)`: last row of the file as a dict
  (`row["Iter"]`, `row["RMS"]`, `row["Alpha"]`, ...), or `None` if there is no
  such file, no usable header or no data row.
- Scripts: all `femtic_*` scripts that read the file now have the setting
  `CNV_FILE = ["femtic.cnv", "FEMTIC-DABIC.cnv"]`. `femtic_ens_decomp.py`,
  `femtic_mod_math.py` and `femtic_mod_plot_slice.py` read the last row by
  column name instead of the fixed positions 0 / 8 (which fitted only the
  layout with distortion columns); `femtic_mod_plot_slice.py` no longer has
  `DISTORTION`. Readers catch `(ValueError, OSError)`.
- `CNV_DEFAULT_NAME` is kept (= first of the list).
- Tests: `test_femtic_cnv.py`, 19 cases. AI-generated (Claude Sonnet 5.5,
  Anthropic); review before production use.

### Changelog (2026-10-11) --- single mesh reader, optional x/y swap

`read_femtic_mesh()` is now the only mesh reader in the package: the copy
in `femtic_viz.py` was removed (that module wraps this one). New optional
`swap_xy` argument and module-level default `MESH_SWAP_XY = False`; the
swap that `femtic_viz` applied unconditionally since 2026-05-31 is now off
unless requested. Element lines require >= 5 columns (was 9). AI-generated
code -- review before production use.

### Changelog (2026-10-22) --- observe.dat station lines with owner / E-type columns, dummy VTF

Problem: `read_observe_dat` stopped with `expected 4 tokens, got 5` on an
observe.dat written for `OWNER_ELEMENT -1` (station line
`ID H-ID owner X Y`, dummy VTF stations with 0 frequencies, as written by
`femtic_add_dummy_vtf.py`).

- `read_observe_dat` parses station lines of 4, 5 or 6 tokens with the new
  helper `_parse_site_header_tokens` (columns per manual v4.3 slides 36-45).
  New keywords `site_header_layout`, `owner_column`, `elec_column`
  (automatic by default); parsed fields in `site["header"]`; the verbatim
  line is still in `site["site_header_line"]`.
- Zero-frequency (dummy) stations are accepted by the reader, writer,
  `modify_data`, `convert_observe_dat` and `summarise_observe_dat`
  (new `n_sites_empty`); `observe_to_site_viz_list` skips them by default.
- Fixed (existing behaviour, not only for the new layout):
  `_site_header_to_meta` read the tokens of a FEMTIC `ID H-ID X Y` line as
  `name x y z`, so `xyz` held `[H-ID, X_km, Y_km]`. For FEMTIC lines it now
  gives `xyz = [X*1000, Y*1000, nan]` and `elev_m = None`; the legacy
  `name x y z` layout is unchanged. Nothing in `ensembles.py`,
  `femtic_viz.py` or `femtic_rto_prep.py` uses `xyz` / `elev_m`.
- Fixed: `write_observe_dat` warned "positive z" for every FEMTIC station
  line with Y > 0 (4th token = Y). The check now applies to the legacy layout
  only.
- Fixed: `read_site_position` took tokens 3 and 4 as X and Y, so with an
  owner column it returned the owner type as x. It now reads the layout.
  (`femtic_rto_prep.py` uses it for the site position in the QC plots.)
- Clear error for the block keywords `APP_RES_AND_PHS` and `HTF` (still not
  supported by the reader).
- `modify_data_fcn` (legacy): station-line detection widened to 4-6 tokens.
- `femtic_viz._observe_to_site_list`: skips VTF/PT stations with 0
  frequencies.
- Unchanged: `ensembles.py`, `femtic_rto_prep.py`. `generate_data_ensemble`
  goes through `modify_data`, which keeps the station lines.
- Open point: `edi_list_to_observe_dat` writes the legacy `name x y z`
  station line, which does not match the manual's `ID H-ID X Y`. The reader
  accepts both; check whether FEMTIC itself accepts files written by that
  function.
- The E-type column is placed after the owner column, following manual
  slide 36. The manual examples for the dummy-VTF case (slides 56, 58) show
  only the owner column.

Tests: `femtic_observe_io_test.py`, 22 cases, all pass (they use the new
features, so none passes on the previous `femtic.py`). AI-generated (Claude Sonnet 5.5, Anthropic); review
before production use.
