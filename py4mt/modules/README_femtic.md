# ReadMe_femtic.md

`femtic.py` is a unified utility module for FEMTIC-related workflows, including:

- reading/writing FEMTIC **mesh** and **resistivity block** files
- conversion to/from compact **NPZ** model files
- optional export to **VTK/VTU** via PyVista
- assorted helpers for FEMTIC outputs (roughness matrices, results parsing, etc.)

---

## What’s in this snapshot

### Cleanups
- Removed a duplicate import (`from pathlib import Path`) and minor style cleanups.

### No behavioural changes required for your new \(\lambda\) rule
The new \(\lambda\) selection rule was implemented in **`ensembles.py`**, where the precision
solver lives. `femtic.py` remains focused on file formats and conversions.

---

## Command line interface

`femtic.py` provides a subcommand-style CLI.

### mesh.dat + resistivity_block → NPZ

```bash
python femtic.py femtic-to-npz --mesh mesh.dat --rho-block resistivity_block_iter0.dat --out-npz model.npz
```

### NPZ → VTU (PyVista)

```bash
python femtic.py npz-to-vtk --npz model.npz --out-vtu model.vtu
```

### NPZ → FEMTIC files

```bash
python femtic.py npz-to-femtic --npz model.npz --mesh-out mesh_out.dat --rho-block-out resistivity_block_out.dat
```

---

## Python usage examples

### Convert FEMTIC files to NPZ

```python
from femtic import mesh_and_block_to_npz

mesh_and_block_to_npz("mesh.dat", "resistivity_block_iter0.dat", "model.npz")
```

### Load NPZ as a PyVista grid and export

```python
from femtic import save_vtk_from_npz

save_vtk_from_npz("model.npz", "model.vtu", out_legacy="model.vtk")
```

---

## Dependencies

- `numpy`
- `scipy`
- Optional:
  - `pyvista` (for VTK/VTU export)
  - `h5py` (for HDF5 conversion helpers)
  - `netCDF4` (for NetCDF conversion helpers)

---

## Version / provenance

This README corresponds to the `femtic.py` found alongside it in this project snapshot.

- Updated: 2026-01-02 (UTC)

