# mt_data_processor.py

Batch MT station processing script.

## Provenance

| Field | Value |
|-------|-------|
| Script | `mt_data_processor.py` |
| Author | Volker Rath (DIAS) |
| Part of | **py4mt** — Python for Magnetotellurics |
| README generated | 2 March 2026 by Claude (Anthropic), from author-supplied documentation |

---

## Workflow

1. **Read** all `.edi` files from `EdiDir` via `data_proc.load_edi()`
2. **Compute** derived quantities (phase tensor, impedance invariants,
   apparent resistivity/phase)
3. **Optionally** set errors, estimate errors, interpolate, or rotate data
4. **Export** each station in one or more formats (EDI, NPZ, HDF, MAT)
5. **Save** a collection NPZ containing all stations

---

## Configuration

All processing options are controlled by variables near the top of the script.
The main switches and their defaults:

| Variable | Default | Description |
|----------|---------|-------------|
| `WorkDir` | *(set per project)* | Root working directory |
| `EdiDir` | `WorkDir + '/orig/'` | Input EDI directory |
| `DataDir` | `WorkDir + '/proc/'` | Output directory (created if missing) |
| `OutFiles` | `'edi, npz, hdf, mat'` | Comma-separated output formats |
| `NameStr` | `'_proc'` | Suffix appended to output filenames |
| `CollName` | `'ANN_DJ_aniso'` | Name for the collection NPZ |

### Processing switches

| Switch | Default | Description |
|--------|---------|-------------|
| `PhasTens` | `True` | Compute phase tensor via `compute_pt()` |
| `Invars` | `True` | Compute Zdet and Zssq invariants |
| `SetErrors` | `False` | Override errors with fixed relative values |
| `EstimateErrors` | `False` | Estimate errors from data (work in progress) |
| `Interpolate` | `False` | Interpolate to uniform frequency sampling |
| `Rotate` | `False` | Rotate impedance/tipper by a fixed angle |
| `Plot` | `False` | Generate per-station diagnostic plots |
| `StatFile` | `True` | Use EDI filename (not station header) for output names |

### Error configuration (when `SetErrors = True`)

```python
Errors = {
    'Zerr':  [0.1, 0.1, 0.1, 0.1],      # relative Z errors (xx, xy, yx, yy)
    'Terr':  [0.03, 0.03, 0.03, 0.03],   # relative tipper errors
    'PTerr': [0.1, 0.1, 0.1, 0.1],       # relative phase tensor errors
}
```

### Interpolation (when `Interpolate = True`)

```python
FreqPerDec = 6
IntMethod = [None, FreqPerDec]
```

### Rotation (when `Rotate = True`)

```python
Angle = 0.0       # rotation angle
DecDeg = True      # angle is in decimal degrees
```

---

## Output formats

For each station, the script can write:

- **EDI** — classical table-style EDI via `save_edi()`
- **NPZ** — NumPy compressed archive via `save_npz()`
- **HDF** — HDF5 via `save_hdf()`
- **MAT** — MATLAB `.mat` via `save_mat(include_raw=True)`
- **NCD** — NetCDF via `save_ncd()` (if added to `OutFiles`)

After all stations are processed, a single collection NPZ is written via
`save_list_of_dicts_npz()`.

---

## Diagnostic plots (when `Plot = True`)

A 3×2 subplot figure is generated per station showing:

| Row | Left | Right |
|-----|------|-------|
| 1 | ρ_a (xy, yx) | Phase (xy, yx) |
| 2 | ρ_a (xx, yy) | Phase (xx, yy) |
| 3 | Tipper | Phase tensor |

Empty axes are removed. Plots are saved in all formats listed in `PlotFormat`
(default: PNG and PDF at 600 DPI).

---

## Dependencies

- `data_proc` — EDI I/O, processing, and export functions
- `data_viz` — plotting helpers (`add_rho`, `add_phase`, `add_tipper`, `add_pt`)
- `util` — `print_title()` and general helpers
- `version` — `versionstrg()` for Py4MT version string
- Standard scientific Python: `numpy`, `matplotlib`

---

## Typical usage

```bash
# Edit WorkDir, EdiDir, and processing switches at the top of the script, then:
python mt_data_processor.py
```
