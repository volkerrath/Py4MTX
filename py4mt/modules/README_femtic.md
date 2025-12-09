# FEMTIC utilities (`femtic.py`)

This module collects utilities used in your FEMTIC-based workflows:

- ensemble directory generation and file management,
- roughness / precision-matrix helpers,
- Gaussian sampling with Q = R.T @ R + λ I,
- a sampler for covariance-driven resistivity fields on FEMTIC meshes,
- and NPZ exporters that can be further converted to HDF5 / NetCDF.

Only a small subset is exposed via the command-line interface for now.

## 1. Requirements

Core dependencies (already used elsewhere in the module):

- `numpy`
- `scipy`
- `joblib`

Optional dependencies for specific features:

- `h5py` – required for the `npz-to-hdf5` sub-command / :func:`npz_to_hdf5`.
- `netCDF4` – required for the `npz-to-netcdf` sub-command / :func:`npz_to_netcdf`.

## 2. Command-line interface

The recommended way to call the small set of high-level tools is via

```bash
python femtic.py <sub-command> [options]
```

The following sub-commands are currently implemented.

### 2.1 Sample a covariance-driven resistivity field

```bash
python femtic.py sample-field \\
  --mesh mesh.dat \\
  --kernel matern \\
  --ell 500 \\
  --sigma2 0.5 \\
  --nu 1.5 \\
  --nugget 1e-6 \\
  --mean 4.60517019 \\
  --strategy sparse \\
  --radius 1250 \\
  --trunc_k 1024 \\
  --seed 1234 \\
  --out rho_sample_on_mesh.npz
```

This uses :func:`sampler1` to draw a log-resistivity field on the FEMTIC
TETRA mesh specified by `mesh.dat`. The NPZ file contains at least

- `rho` – sampled resistivity,
- `logrho` – log-resistivity,
- `centroids` – element centroids,
- `tet_ids` – element indices.

### 2.2 Export NPZ → HDF5

```bash
python femtic.py npz-to-hdf5 \\
  --npz femtic_model.npz \\
  --out-hdf5 femtic_model.h5 \\
  --group femtic_model
```

This calls :func:`npz_to_hdf5`, which writes each key in the NPZ as one
dataset in the group `femtic_model` of `femtic_model.h5`.

### 2.3 Export NPZ → NetCDF4

```bash
python femtic.py npz-to-netcdf \\
  --npz femtic_model.npz \\
  --out-nc femtic_model.nc
```

This calls :func:`npz_to_netcdf`, which converts standard FEMTIC NPZ
layouts into a NetCDF4 file with named dimensions (`node`, `cell`,
`region`, …) where possible.

## 3. Notes

- The CLI is intentionally minimal; most lower-level functions are
  meant to be imported and scripted from Python.
- The HDF5 / NetCDF converters are agnostic of how the NPZ was created,
  as long as the expected FEMTIC keys are present.

Author: Volker Rath (DIAS)  
Created by ChatGPT (GPT-5 Thinking) on 2025-12-09
