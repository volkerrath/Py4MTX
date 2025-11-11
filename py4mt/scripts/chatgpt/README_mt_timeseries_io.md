# MT Time-Series I/O (Metronix & Phoenix) with TF Deconvolution

This package provides a single Python module, **`mt_timeseries_io.py`**, that can:
- Read **Metronix ADU-08/07** `.ats` channels with their paired XML.
- Read **Phoenix MTU (5/5A/5C)** int32 channels with metadata from `.xml`/`.log`.
- Load **instrument transfer functions (TFs)** for coils/electrodes from Metronix XML/CSV and Phoenix `.cal/.coi/.csv`.
- **Deconvolve** to physical units (H → nT, E → mV/km) with Tikhonov regularization.
- Export to **NPZ / HDF5 / CSV**, plus a small JSON sidecar for metadata.
- Provide a quick-look **plot** (optional) and accept simple **YAML configs** (optional).

> Scope: The readers are made to be robust and dependency-light; vendor formats vary in the wild.
> If your files use different XML tags/units, adjust the few regex patterns in the readers.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy pandas h5py
# Optionals (recommended):
pip install matplotlib pyyaml obspy
```

Python ≥ 3.9 is recommended.

---

## CLI Usage

```bash
python mt_timeseries_io.py --vendor metronix --in Hx.ats --xml Hx.xml \
  --tf MFS-06e.xml --out out/Hx --export npz h5 csv --deconv

python mt_timeseries_io.py --vendor phoenix --in EX.raw --xml EX.xml \
  --tf phoenix_sensor.cal --out out/Ex --export npz --deconv --dipole-km 0.1
```

**Key options**
- `--vendor {metronix|phoenix}` — choose reader
- `--in PATH` — channel file (`.ats` for Metronix; int32 raw for Phoenix)
- `--xml PATH` — paired XML/LOG for metadata (autodetected if omitted)
- `--tf PATH` — sensor TF file (Metronix XML/CSV, Phoenix `.cal/.coi/.csv`)
- `--deconv` — perform frequency-domain deconvolution to physical units
- `--export` — any of: `npz h5 csv` (multiple allowed)
- `--dipole-km` — E-field dipole length in km (if you want mV/km from Volts)
- `--reg-eps` — regularization epsilon (default `1e-3`)

---

## Programmatic API (quick tour)

```python
from mt_timeseries_io import (
    read_metronix_ats, read_phoenix_mtu,
    load_tf_metronix_sensor, load_tf_phoenix_sensor,
    counts_to_volts, deconvolve_sensor, quick_plot
)

ts = read_metronix_ats("Hx.ats", "Hx.xml")
ts_v = counts_to_volts(ts)
tf = load_tf_metronix_sensor("MFS-06e.xml")
ts_phys = deconvolve_sensor(ts_v, tf)  # -> H in nT by default for Hx/Hy/Hz
```

`quick_plot(ts)` will plot if `matplotlib` is installed.

---

## Supported Inputs & Metadata

### Metronix ADU-08/07
- **Data**: `.ats` (little-endian int32)
- **Header/metadata**: paired `.xml` (same basename)  
  Parsed fields include: sampling rate, ADC bits, ADC Vpp, preamp gain, start time,
  orientation (Hx/Hy/Hz/Ex/Ey), sensor model, site coordinates (if present).

### Phoenix MTU 5/5A/5C
- **Data**: int32 channel file (varies by export)
- **Header/metadata**: paired `.xml` or `.log` (autodetected)  
  Parsed fields (when present): sampling rate, ADC bits, ADC range, preamp gain,
  start time, orientation, sensor model, coordinates.

> If sampling rate is missing, an error is raised.

---

## Instrument Transfer Functions (TF)

- **Metronix**: sensor XML (typical coil) or CSV with columns `f_Hz, amp, phase_deg`  
- **Phoenix**: `.cal/.coi/.csv` lines formatted as `freq, amplitude, phase_deg`  

The code expects **amplitude linear** and **phase in degrees**. The complex response is
`H = amp * exp(i * phase_deg * pi/180)`. If the TF is in **V/(nT·Hz)**, the deconvolver
detects `units` string like `"V/(nT*Hz)"` in `tf.meta["units"]` and multiplies by `2πf`
internally to get **V/nT** before inversion.

---

## Deconvolution & Units

Let `V(f) = H(f) * X(f)`. We estimate `X(f)` with a regularized inverse:
`X ≈ V * H* / (|H|^2 + ε^2)`, `ε = reg_eps * max|H|`.  
Output units by default:
- **H-channels** (`Hx`,`Hy`,`Hz`): **nT**
- **E-channels** (`Ex`,`Ey`): **mV/km** (if you provide `--dipole-km` or set `ts.meta.notes["dipole_km"]`).

---

## Exports

- **NPZ**: `{basename}.npz` with `data`, `fs`, `units`; plus `{basename}.json` sidecar with metadata
- **HDF5**: dataset `/data` with attributes (`fs`, `units`, etc.)
- **CSV**: one column `data` preceded by a few header lines starting with `#`

---

## Examples

See the `examples/` folder:
- `example_run.sh` shows CLI calls for both vendors.
- `example_sensor_tf.csv` is a minimal TF file (3 frequency points) usable for quick tests.

```bash
bash examples/example_run.sh
```

---

## Notes & Caveats

- Vendor XML/LOG formats have variants. If a field is not found, the code keeps data
  in **counts** or **Volts** and warns; supply missing values or adapt the regex.
- For **E-fields**, ensure your **dipole length** is set to obtain mV/km (either CLI `--dipole-km` or `ts.meta.notes["dipole_km"] = ...`).

---

## License

MIT (or adapt to your project).

---

**Authors & provenance**  
- Author: Volker Rath (DIAS)  
- Created by ChatGPT (GPT-5 Thinking) on 2025-11-11
