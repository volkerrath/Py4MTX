# mt_data_processor.py

Batch MT station processing script.

## Provenance

| Field | Value |
|-------|-------|
| Script | `mt_data_processor.py` |
| Author | Volker Rath (DIAS), with ChatGPT (GPT-5 Thinking), 2026-02-13 |
| Part of | **py4mt** — Python for Magnetotellurics |
| README generated | 3 March 2026 by Claude (Anthropic), from cleaned source |
| Modified | 2026-03-07 — unified save_xxx(**data_dict, path=...) calling convention |
| Modified | 2026-03-16 — freq_order, D+/rho+ test (DPLUS), add_rhoplus plot; Claude Sonnet 4.6 (Anthropic) |

## Purpose

Reads all `.edi` files from an input directory, computes derived quantities
(phase tensor, impedance invariants, apparent resistivity/phase), optionally
sets errors, estimates errors, interpolates, or rotates, then exports each
station in one or more formats. A collection NPZ containing all stations is
saved at the end.

## Workflow

1. **Read** all `.edi` files from `EDI_DIR` via `data_proc.load_edi()`.
2. **Compute** derived quantities (phase tensor, impedance invariants,
   apparent resistivity/phase).
3. **Optionally** set errors, estimate errors, interpolate, or rotate data.
4. **Export** each station in one or more formats (EDI, NPZ, HDF, MAT).
5. **Save** a collection NPZ containing all stations.

## Configuration constants

| Constant | Default | Description |
|----------|---------|-------------|
| `WORK_DIR` | *(set per project)* | Root working directory |
| `EDI_DIR` | `WORK_DIR + "/orig/"` | Input EDI directory |
| `DATA_DIR` | `WORK_DIR + "/proc/"` | Output directory (created if missing) |
| `OUT_FILES` | `"edi, npz, hdf, mat"` | Comma-separated output formats |
| `NAME_STR` | `"_proc"` | Suffix appended to output filenames |
| `COLL_NAME` | `"ANN_DJ_aniso"` | Name for the collection NPZ |
| `FREQ_ORDER` | `"inc"` | Frequency order passed to `load_edi()`: `"inc"`, `"dec"`, or `"keep"` |

### Processing switches

| Constant | Default | Description |
|----------|---------|-------------|
| `PHAS_TENS` | `True` | Compute phase tensor via `compute_pt()` |
| `INVARS` | `True` | Compute Zdet and Zssq invariants |
| `DPLUS` | `True` | D+/rho+ test via `compute_rhoplus()` on Zxy, Zyx, Zdet; prints violation counts; stores results in `data_dict["dplus"]` |
| `SET_ERRORS` | `False` | Override errors with fixed relative values |
| `ESTIMATE_ERRORS` | `False` | Estimate errors from data (work in progress) |
| `INTERPOLATE` | `False` | Interpolate to uniform frequency sampling |
| `ROTATE` | `False` | Rotate impedance/tipper by a fixed angle |
| `PLOT` | `False` | Generate per-station diagnostic plots |
| `STAT_FILE` | `True` | Use EDI filename (not station header) for output names |

### Error configuration (when `SET_ERRORS = True`)

```python
ERRORS = {
    "Zerr":  [0.1, 0.1, 0.1, 0.1],      # relative Z errors (xx, xy, yx, yy)
    "Terr":  [0.03, 0.03, 0.03, 0.03],   # relative tipper errors
    "PTerr": [0.1, 0.1, 0.1, 0.1],       # relative phase tensor errors
}
```

### Interpolation (when `INTERPOLATE = True`)

```python
FREQ_PER_DEC = 6
INT_METHOD = [None, FREQ_PER_DEC]
```

### Rotation (when `ROTATE = True`)

```python
ANGLE = 0.0
DEC_DEG = True
```

## Output formats

For each station, the script can write: EDI (classical table-style),
NPZ (NumPy compressed archive), HDF (HDF5), MAT (MATLAB `.mat`), and
NCD (NetCDF, if added to `OUT_FILES`).

All save functions are called with the `**data_dict` splatting convention
(`save_xxx(**data_dict, path="...")`), where `path` is keyword-only.

After all stations are processed, a single collection NPZ is written via
`save_list_of_dicts_npz()`.

## Diagnostic plots (when `PLOT = True`)

A 3×2 subplot figure is generated per station (4×2 when `DPLUS = True`) showing:

| Row | Left | Right |
|-----|------|-------|
| 1 | ρ_a (xy, yx) | Phase (xy, yx) |
| 2 | ρ_a (xx, yy) | Phase (xx, yy) |
| 3 *(DPLUS only)* | D+/rho+ test (xy, yx) | *(hidden)* |
| 3 (or 4) | Tipper | Phase tensor |

Empty axes are removed. Plots are saved in all formats listed in `PLOT_FORMAT`
(default: PNG and PDF at 600 DPI).

## Dependencies

`numpy`, `matplotlib`; py4mt: `data_proc` (`load_edi`, `compute_rhoplus`, …), `data_viz` (`add_rhoplus`, …), `util`, `version`.
