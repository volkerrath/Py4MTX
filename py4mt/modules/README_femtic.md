# FEMTIC Python utilities (`femtic.py`)

This repository contains a single-file Python helper module **`femtic.py`** that bundles
common tasks around **FEMTIC** model/data handling, roughness/precision matrices, Gaussian
sampling, and **mesh/model format conversions**.

It is intended to be:

- **Importable** as a normal Python module (no code runs on import)
- **CLI-callable** for the most common format conversions

> If you also use ensemble generation / EOF / KL / PCE tools, see **`ReadMe_ensembles.md`**.

---

## What `femtic.py` provides

### 1) FEMTIC observe.dat perturbation helpers

Core routines for perturbing a FEMTIC-style `observe.dat` **in-place**:

- `modify_data(...)` — perturb observations by drawing random noise and applying it
- `get_nrms(...)` — compute N-RMS between two observation files
- `get_femtic_sites(...)`, `get_femtic_data(...)` — light helpers for site/data access

### 2) Resistivity model helpers

- `read_model(...)`, `modify_model(...)`, `insert_model(...)` — read/perturb/insert
  resistivity-block style models (region-based)

### 3) Roughness / precision matrix helpers and Gaussian sampling

- `get_roughness(...)` — read a FEMTIC roughness/roughening-matrix file
- `make_prior_cov(...)` — build prior covariance proxies from roughness matrices
- `make_precision_solver(...)` — build a solver for systems involving **Q = RᵀR (+ λI)**
- `sample_rtr_full_rank(...)`, `sample_rtr_low_rank(...)`, `sample_prior_from_roughness(...)`
  — sample Gaussian vectors consistent with the roughness/precision model

### 4) Mesh/model format conversion: FEMTIC ↔ NPZ ↔ VTK/VTU

- `read_femtic_mesh(...)` / `write_femtic_mesh(...)`
- `read_resistivity_block(...)` / `write_resistivity_block(...)`
- `mesh_and_block_to_npz(...)` — `mesh.dat` + `resistivity_block_iterX.dat` → compact NPZ
- `npz_to_femtic(...)` — NPZ → `mesh.dat` + `resistivity_block_iterX.dat`
- `save_vtk_from_npz(...)` — NPZ → VTU (and optional legacy VTK), for PyVista/ParaView
- Optional container formats:
  - `npz_to_hdf5(...)` / `hdf5_to_npz(...)`
  - `npz_to_netcdf(...)` / `netcdf_to_npz(...)`

---

## Requirements

Minimum:

- Python 3.10+
- `numpy`
- `scipy`

Optional (only needed if you use the corresponding functionality):

- `pyvista` (and VTK) for VTU/VTK export helpers
- `h5py` for HDF5 conversions
- `netCDF4` or `xarray` backend support for NetCDF conversions (depending on your environment)

---

## CLI usage (format conversions)

`femtic.py` includes a small CLI with subcommands.

### Convert FEMTIC mesh + resistivity block → NPZ

```bash
python femtic.py femtic-to-npz \
  --mesh mesh.dat \
  --rho-block resistivity_block_iter0.dat \
  --out-npz model_iter0.npz
```

### Convert NPZ → VTU (and optional legacy VTK)

```bash
python femtic.py npz-to-vtk \
  --npz model_iter0.npz \
  --out-vtu model_iter0.vtu \
  --out-legacy model_iter0.vtk
```

You can choose which cell scalar to export (defaults to `log10_resistivity`):

```bash
python femtic.py npz-to-vtk --npz model_iter0.npz --out-vtu model_iter0.vtu --scalar log10_resistivity
```

### Convert FEMTIC-style HDF5 / NetCDF → NPZ

```bash
python femtic.py hdf5-to-npz --hdf5 model.h5 --out-npz model.npz --group femtic_model
python femtic.py netcdf-to-npz --netcdf model.nc --out-npz model.npz
```

### Convert NPZ → FEMTIC mesh.dat + resistivity_block

```bash
python femtic.py npz-to-femtic \
  --npz model_iter0.npz \
  --mesh-out mesh_out.dat \
  --rho-block-out resistivity_block_iter0_out.dat
```

---

## Python usage examples

### Mesh + resistivity block → NPZ (library usage)

```python
import femtic as fem

fem.mesh_and_block_to_npz(
    mesh_path="mesh.dat",
    rho_block_path="resistivity_block_iter0.dat",
    out_npz="model_iter0.npz",
)
```

### NPZ → VTU (for PyVista/ParaView)

```python
import femtic as fem

fem.save_vtk_from_npz(
    npz_path="model_iter0.npz",
    out_vtu="model_iter0.vtu",
    out_legacy=None,  # or "model_iter0.vtk"
)
```

### Perturb an `observe.dat` in-place

```python
import femtic as fem

# Add N(0,1) noise scaled by the file's error columns (see function docstring)
fem.modify_data(
    template_file="observe.dat",
    draw_from=("normal", 0.0, 1.0),
    method="add",
    out=True,
)
```

---

## Notes and conventions

- Many routines follow FEMTIC naming conventions such as:
  - `observe.dat`
  - `mesh.dat`
  - `resistivity_block_iterX.dat`
  - `roughening_matrix_*.dat`
- **Air / ocean regions:** if you also plot models, keep in mind FEMTIC commonly uses
  special regions for air/ocean (your visualization module may treat them specially).

---

## License / attribution

If you redistribute, please keep the module header and docstrings intact.

Author: Volker Rath (DIAS)  
Created by ChatGPT (GPT-5 Thinking) on 2025-12-21 (UTC)
