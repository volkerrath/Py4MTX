# README — `modem_improc.py`

Provenance: cleaned/debugged with Claude (Anthropic), Mar 2026

## Purpose

Apply spatial image-processing filters (Gaussian smoothing, median
filter, or anisotropic diffusion) to a ModEM 3-D resistivity model.

---

## Bugs found and fixed

### Round 1 (original script → first cleanup)

1. **`read_model` → `read_mod`** — The module function is `read_mod`.

2. **`write_model` → `write_mod`** — Same issue.

3. **`prepare_mod` → `prepare_model`** — The module function is
   `prepare_model`.

4. **`generic_gaussian` / `generic_uniform` do not exist** — Replaced
   with `scipy.ndimage.gaussian_filter` and `uniform_filter`.

5. **Double `.rho` extension** — `read_mod` already appends `.rho`.

6. **Missing imports: `util` and `inspect`** — Both were used but never
   imported.

7. **Smooth loop did not iterate** — The loop stored results in
   `rhonew` but never fed them back into the next iteration.  Fixed by
   updating `rho_tmp` in place.

8. **`ksize` referenced in "smooth" branch** — `ksize` is only defined
   when `action == "med"` → `NameError`.  Replaced with a filename
   using the actual smoothing parameters.

9. **Wrong keyword `rho=`** — `write_mod` expects `mval=`.

10. **`Plot=True` passed to `anidiff3D`** — Not a valid keyword.
    Removed.

11. **Positional args to `write_mod`** — Order did not match the
    function signature.  Switched to explicit keywords.

### Round 2 (revised script → this version)

No new bugs found.  Applied UPPERCASE convention to all user
parameters and added provenance note.

---

## Cleanup / style

| What | Detail |
|------|--------|
| UPPERCASE user parameters | `RHOAIR`, `MODFILE_IN`, `MODFILE_OUT`, `ACTION`, `FTYPE`, `SIGMA`, `ORDER`, `BMODE`, `MAXIT`, `KSIZE`, `FOPT` |
| Provenance note | Added to module docstring |
| Removed unused imports | `gc` (original) |
| f-string formatting | Throughout |
| PEP 8 | Consistent spacing, keyword arguments in all `mod.*` calls |
| `modext=""` on output | Output filenames already include `.rho` |

---

## Usage

```bash
export PY4MTX_ROOT=/path/to/Py4MTX
export PY4MTX_DATA=/path/to/data
python modem_improc.py
```

Edit `MODFILE_IN`, `MODFILE_OUT`, and `ACTION` at the top of the script.
Valid actions: `"smooth"`, `"med"`, `"anidiff"`.
