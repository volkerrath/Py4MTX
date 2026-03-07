# README — `modem_improc.py`

Provenance: cleaned/debugged with Claude (Anthropic), Mar 2026

## Purpose

Apply spatial image-processing filters (Gaussian smoothing, median
filter, or anisotropic diffusion) to a ModEM 3-D resistivity model.

---


## Usage

```bash
export PY4MTX_ROOT=/path/to/Py4MTX
export PY4MTX_DATA=/path/to/data
python modem_improc.py
```

Edit `MODFILE_IN`, `MODFILE_OUT`, and `ACTION` at the top of the script.
Valid actions: `"smooth"`, `"med"`, `"anidiff"`.
