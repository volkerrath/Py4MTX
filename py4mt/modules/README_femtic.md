# README_femtic.md

Utilities for working with **FEMTIC** mesh/model/data files in Python.

This module bundles helpers for:

- Reading / writing **FEMTIC resistivity blocks** (`resistivity_block_iter*.dat`)
- Converting between FEMTIC text formats and compact **NPZ** model files
- Exporting models to **VTK/VTU** (PyVista)
- Ensemble / covariance / sampling utilities used in FEMTIC-style workflows

> This README reflects the updated behaviour where the **ocean block (region 1) is optional**
> and where **additional fixed blocks** may be present.

---

## Files and conventions

### Resistivity block: `resistivity_block_iterX.dat`

Typical structure:

1. Header line:  
   `nelem  nreg`
2. `nelem` lines mapping elements → region indices
3. `nreg` lines defining region parameters (at least 6 columns)

A common region-line convention is:

```
ireg  rho  rho_lower  rho_upper  n  flag
```

where `flag` typically indicates whether a region is **fixed** (`flag == 1`) or **free**
(`flag == 0`).

Common special regions:

- **Region 0**: *air* (often fixed, very resistive)
- **Region 1**: *ocean* (often fixed, very conductive) — **may be absent**
- Other regions may also be fixed (e.g., prescribed background blocks), not only air/ocean.

---

## Fixed blocks and optional ocean (important)

### `read_model(...)`

```python
m = read_model("resistivity_block_iter0.dat", model_trans="log10")
```

Default behaviour (`include_fixed=False`, `ocean=None`):

- always skips **region 0 (air)**,
- skips **any region with `flag == 1`** (fixed blocks),
- additionally skips **region 1** *if treated as ocean* (auto-inferred unless you override).

So the returned vector is the **free** model vector (by default `log10(rho)`) for all
regions that are not fixed by the rules above.

Control parameters:

- `ocean=None` *(default)*: infer whether region 1 is ocean (conservative heuristic)
- `ocean=True`: force “treat region 1 as ocean” (skip it as fixed + optionally enforce `ocean_rho` on writing)
- `ocean=False`: force “do not treat region 1 as ocean”
  - Note: if region 1 has `flag == 1`, it is still skipped because it is fixed.
- `include_fixed=True`: return **all** regions, including air and any fixed blocks

Ocean detection heuristic (conservative):

- region 1 has `flag == 1` (fixed), and
- `rho <= 1 Ωm` (very conductive; typical seawater ~0.25 Ωm)

### `insert_model(...)`

```python
m_free = read_model("resistivity_block_iter0.dat")  # free only (log10)
m_free2 = m_free + 0.1                              # example update

insert_model(
    template="resistivity_block_iter0.dat",
    model=m_free2,
    model_file="resistivity_block_iter0.new",
)
```

Default behaviour:

- **Region 0 (air)** is always written as fixed (`air_rho`, default `1e9 Ωm`)
- **Any region with `flag == 1`** is preserved from the template (fixed blocks)
- **Region 1 (ocean)** is written as fixed **only if treated as ocean** (auto-inferred unless overridden)
- all remaining (free) regions are written from `model` (interpreted as **log10(ρ)**)

Metadata columns from the template (bounds, `n`, `flag`) are preserved.

Control parameters:

- `ocean=None/True/False`: same meaning as for `read_model`
- `air_rho`: resistivity enforced for region 0
- `ocean_rho`: resistivity enforced for region 1 **if treated as ocean**

---

## Quick recipes

### Case A: file has air + ocean fixed blocks, plus additional fixed regions

```python
m = read_model("resistivity_block_iter0.dat")   # returns only free regions
insert_model("resistivity_block_iter0.dat", m, model_file="rho_out.dat")
```

### Case B: file has air only (no ocean region)

If region 1 is actually your first free region, just use defaults (the code will *not*
treat it as ocean unless it looks like a fixed conductive ocean block):

```python
m = read_model("resistivity_block_iter0.dat")   # region 1 included if free
insert_model("resistivity_block_iter0.dat", m, model_file="rho_out.dat")
```

### Case C: ocean inference is wrong (force behaviour)

```python
# Force ocean handling:
m = read_model("resistivity_block_iter0.dat", ocean=True)
insert_model("resistivity_block_iter0.dat", m, ocean=True, model_file="rho_ocean.dat")

# Force no-ocean special-casing:
m = read_model("resistivity_block_iter0.dat", ocean=False)
insert_model("resistivity_block_iter0.dat", m, ocean=False, model_file="rho_no_ocean.dat")
```

### Return full region vector (including fixed regions)

```python
rho_all = read_model("resistivity_block_iter0.dat", model_trans="none", include_fixed=True)
```

---

## Notes / troubleshooting

- If `insert_model` complains about a size mismatch, your template likely contains
  additional fixed regions (`flag == 1`). Use `read_model(...)` on the **same template**
  to generate a consistent free vector length.
- If ocean inference mis-classifies region 1 (e.g., a very conductive non-ocean fixed block),
  pass `ocean=True/False` explicitly.

---

## License / attribution

Author: Volker Rath (DIAS)  
Created/updated with assistance from ChatGPT.
