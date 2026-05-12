# README — `modem_compare.py`

Provenance: cleaned/debugged with Claude (Anthropic), Mar 2026

## Purpose

Compare two ModEM 3-D resistivity models by computing the
log10-resistivity difference and the cross-gradient.

---

## Bugs found and fixed

### Round 1 (original script → first cleanup)

1. **`read_model` → `read_mod`** — The module function is `read_mod`, not
   `read_model`.  Every call would crash with `AttributeError`.

2. **`write_model` → `write_mod`** — Same issue.

3. **Wrong number of return values** — `read_mod` returns six values
   `(dx, dy, dz, mval, reference, trans)`, but the original unpacked
   only five.

4. **Wrong keyword `rho=`** — `write_mod` expects `mval=`.

5. **Double `.rho` extension** — Paths already ended in `.rho`, but
   `read_mod` appends `.rho` again by default.

6. **Bug in cross-gradient norms** — `ng2` computation mixed gradients
   from both models (`g2x*g1x` instead of `g2x**2`).

### Round 2 (revised script → this version)

7. **Missing `import os`** — `os.environ` was used but `os` was not
   imported → `NameError` at startup.

8. **Missing `import inspect`** — `inspect.getfile(...)` was used but
   `inspect` was not imported → `NameError` at startup.

9. **`model_out_crg` commented out but referenced** — The variable was
   commented with `#` in the user-parameters block, but the live
   cross-gradient section at the bottom used it → `NameError`.
   Uncommented the variable definition.

10. **Wrong `crossgrad` call signature** — The script called
    `inv.crossgrad(dx, dy, dz, rho1, rho2)`, but the function in
    `modem.py` is `crossgrad(m1, m2, mesh, Out)` and it returns
    **two** values `(cgm, cgnm)`.  Fixed to
    `mod.crossgrad(m1=rho1, m2=rho2, mesh=[dx, dy, dz])` and
    unpacked both return values.

---

## Cleanup / style

| What | Detail |
|------|--------|
| UPPERCASE user parameters | `MODEL_IN1`, `MODEL_IN2`, `MODEL_OUT_DIFF`, `MODEL_OUT_CRG` |
| Provenance note | Added to module docstring |
| Removed unused imports | `time`, `struct` (original) |
| f-string formatting | Throughout |
| PEP 8 | Consistent spacing and keyword arguments |

---

## Usage

```bash
export PY4MTX_ROOT=/path/to/Py4MTX
export PY4MTX_DATA=/path/to/data
python modem_compare.py
```

Edit `MODEL_IN1`, `MODEL_IN2`, and the output stems at the top of the
script.  All input paths should be file **stems** without the `.rho`
extension.
