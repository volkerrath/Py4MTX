# dataproc.py

`dataproc.py` provides lightweight utilities to **read, process, and write**
magnetotelluric (MT) transfer functions, focusing on EDI.

This README documents the updated drop-in module `dataproc_updated.py`.

---

## Main entry points

- `load_edi(path, prefer_spectra=True, err_kind="var", ...)`
  - Phoenix **SPECTRA** EDIs: parses `>SPECTRA` and reconstructs Z and T
  - Classical EDIs: reads `>FREQ`, Z blocks, optional variance blocks, optional T blocks

- `compute_pt(Z, Z_err=None, err_kind="var", nsim=200, ...)`
  - Computes phase tensor Φ and optionally propagates Z uncertainties via Monte-Carlo.

- `save_edi(path, edi, add_pt_blocks=True, ...)`
  - Writes a **classical table-style** EDI from an in-memory dictionary.

- `dataframe_from_edi(edi, ...)`
  - Creates a tidy `pandas.DataFrame` (optionally includes rhoa/phase/tipper/PT columns).

---

## Minimal usage

```python
from dataproc_updated import load_edi, compute_pt, save_edi

edi = load_edi("SITE001.edi", prefer_spectra=True, err_kind="var")

P, P_err = compute_pt(edi["Z"], edi.get("Z_err"), err_kind=edi.get("err_kind", "var"))
edi["P"] = P
edi["P_err"] = P_err

save_edi("SITE001_out.edi", edi, add_pt_blocks=True)
```

---

## Error conventions

- `err_kind="var"` means `*_err` arrays are **variances**
- `err_kind="std"` means `*_err` arrays are **standard deviations**

When writing an EDI, variance blocks are written; if your in-memory errors are
standard deviations, they are squared before export.

---

## Dependencies

Required:
- Python 3.10+
- `numpy`, `scipy`, `pandas`

Optional:
- `xarray` for `save_ncd`
- `tables` for `save_hdf`

---

Author: Volker Rath (DIAS)  
Created by ChatGPT (GPT-5 Thinking) on 2025-12-21 (UTC)
