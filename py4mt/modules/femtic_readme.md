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
| `sites_as_dict_list()` | Flatten parsed observe.dat into a list of per-site dicts.      |
| `write_observe_dat()` | Write a parsed (and possibly modified) structure back to disk.  |
| `edi_list_to_observe_dat()` | Convert a list of `data_proc.load_edi` dicts to `observe.dat`. |
| `observe_to_site_viz_list()` | Read observe.dat and return per-site dicts with Z, rhoa, phase. |
| `modify_data()`       | Add Gaussian perturbations to observation data.                  |
| `insert_model()`      | Write sampled log₁₀ρ into a resistivity block file.     |
| `read_distortion_file()` | Read FEMTIC galvanic distortion file.                |
| `read_resistivity_block()` | Parse resistivity block file → dict of arrays.     |

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
| `util.py`             | General-purpose helpers.                                             |
| `femtic_rto_rough.py` | Extract roughness matrix from FEMTIC.                               |
| `femtic_rto_prior.py` | Build prior covariance proxy.                                       |
| `femtic_rto_prep.py`  | Generate RTO ensemble.                                              |

---

## Version / provenance

Updated: 2026-04-13

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

Author: Volker Rath (DIAS)
