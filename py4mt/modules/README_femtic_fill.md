# femtic.py — FEMTIC utilities (NPZ workflow + ellipsoid modifier)

This module provides a compact command-line workflow to convert FEMTIC model files to a self-contained NPZ representation, apply model modifications in NPZ space, and write results back to FEMTIC-format files.

This README documents the **updated version** that includes an **ellipsoid fill** modifier with two angle conventions:

- Euler angles with order **ZYX** (`--angle-convention zyx`)
- **Strike/Dip/Slant** (`--angle-convention sds`)

---

## Quick start

### 1) Convert FEMTIC model to NPZ

```bash
python femtic_updated_sds.py femtic-to-npz \
  --mesh mesh.dat \
  --rho-block resistivity_block_iter0.dat \
  --out-npz model_elem.npz
```

### 2) Fill an oriented ellipsoid (centroid-in-ellipsoid test)

**Fill value as resistivity (Ohm·m):**
```bash
python femtic_updated_sds.py npz-ellipsoid-fill model_elem.npz model_filled.npz \
  --center 0 0 -2000 \
  --axes 5000 3000 1500 \
  --angles 20 0 0 \
  --angle-convention zyx \
  --fill 10 --fill-space rho
```

**Fill value as log10(resistivity):**
```bash
python femtic_updated_sds.py npz-ellipsoid-fill model_elem.npz model_filled.npz \
  --center 0 0 -2000 \
  --axes 5000 3000 1500 \
  --angles 1.0 0.0 0.0 \
  --fill 1.0 --fill-space log10
```

**Strike/Dip/Slant convention:**
```bash
python femtic_updated_sds.py npz-ellipsoid-fill model_elem.npz model_filled.npz \
  --center 0 0 -2000 \
  --axes 5000 3000 1500 \
  --angles 30 45 10 \
  --angle-convention sds \
  --fill 2.0 --fill-space log10
```

### 3) Write NPZ back to FEMTIC format

```bash
python femtic_updated_sds.py npz-to-femtic \
  --npz model_filled.npz \
  --mesh-out mesh_out.dat \
  --rho-block-out resistivity_block_out.dat
```

---

## Commands

### `femtic-to-npz`

Convert `mesh.dat` + `resistivity_block_iterX.dat` to an NPZ file containing:

- mesh nodes and tetra connectivity
- region metadata (rho, bounds, flags, counts)
- element arrays (centroids, region per element, log10 resistivity, etc.)

Typical usage:
```bash
python femtic_updated_sds.py femtic-to-npz --mesh mesh.dat --rho-block resistivity_block_iter0.dat --out-npz model_elem.npz
```

---

### `npz-ellipsoid-fill`

Modify an NPZ model by filling an oriented ellipsoid using **cell/element centroids**.

**Important behavior**
- The ellipsoid selection is performed on **element centroids**.
- Elements inside the ellipsoid are **reassigned to a new region** with the requested resistivity.
- By default, **fixed regions are preserved**:
  - region 0 (air)
  - any region with `region_flag == 1`
  - (ocean handling depends on your model convention; if your workflow uses a fixed ocean region, keep it flagged fixed)

Usage:
```bash
python femtic_updated_sds.py npz-ellipsoid-fill IN_NPZ OUT_NPZ \
  --center CX CY CZ \
  --axes A B C \
  --angles ANG1 ANG2 ANG3 \
  --fill VALUE \
  --fill-space {rho,log10} \
  --angle-convention {zyx,sds}
```

**Arguments**
- `--center CX CY CZ` : ellipsoid center in model coordinates
- `--axes A B C` : semi-axes (must be positive)
- `--angles ANG1 ANG2 ANG3` :
  - if `--angle-convention zyx`: Euler angles (deg), applied as **Rz @ Ry @ Rx**
  - if `--angle-convention sds`: **strike/dip/slant** (deg), applied as **Rz(strike) @ Rx(dip) @ Rz(slant)**
- `--fill-space` :
  - `rho` means `--fill` is in Ohm·m
  - `log10` means `--fill` is log10(Ohm·m)
- `--fill` : value to insert (interpreted according to `--fill-space`)

**Notes on angle conventions**
- The implementation assumes **right-handed rotations** about the model axes.
- “Strike” is often defined in geoscience as clockwise-from-North; if your model x/y axes do not match that convention, you may need to convert:
  - a common mapping is `strike_math = 90 - strike_geo` (depends on axis definitions).

---

### `npz-to-femtic`

Write FEMTIC files from an NPZ model:
- `mesh_out.dat`
- `resistivity_block_out.dat`

Usage:
```bash
python femtic_updated_sds.py npz-to-femtic --npz model_filled.npz --mesh-out mesh_out.dat --rho-block-out resistivity_block_out.dat
```

---

## Practical tips

- **Check coordinate system**: ensure your ellipsoid center/axes are in the same coordinate system as FEMTIC mesh nodes.
- **Use centroids**: thin targets might require slightly larger axes because the test is centroid-based.
- **Preserve fixed regions**: keep air/ocean (and other fixed blocks) flagged fixed to avoid accidental edits.
- **Versioning**: keep NPZ snapshots for reproducibility when running ensembles/sampling.

---

## Files

- `femtic_updated_sds.py` : updated module including ellipsoid fill + strike/dip/slant option
- `model_elem.npz` : typical NPZ intermediate file

