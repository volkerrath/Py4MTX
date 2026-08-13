# femtic_summarize_model_cells.py

Report the Jacobian parameter dimension for one or more FEMTIC inversion
iterations, combining free resistivity regions and free distortion
parameters.

---

## Purpose

Quickly inspect the size of the inverse problem for a given iteration
without running a full inversion. For each iteration the script reads:

- `resistivity_block_iterXX.dat` — one ρ value per region, and
- `distortion_iterXX.dat` — four C-matrix values per site (if present),

and reports the composition of the model (air / ocean / other-fixed / free
regions), the number of free distortion parameters, and the resulting
total Jacobian column count (`n_rho + n_distortion`).

The script is fully self-contained (Python standard library only —
`os`, `re`, `sys`, `pathlib`). It does **not** import or delegate to
`femtic.py`; there is no `femtic.summarise_model_file()` /
`femtic._print_model_summary()` dependency.

---

## Usage

```bash
# Single resistivity_block file (its paired distortion file, if any,
# is picked up automatically from the same directory)
python femtic_summarize_model_cells.py resistivity_block_iter0.dat

# Glob (shell expands)
python femtic_summarize_model_cells.py resistivity_block_iter*.dat

# Scan a directory for all resistivity_block_iter*.dat / distortion_iter*.dat
python femtic_summarize_model_cells.py /path/to/run/

# No arguments -> scans the current directory
python femtic_summarize_model_cells.py
```

Arguments are parsed directly from `sys.argv[1:]` (no `argparse`, no
`--ocean` or other flags). Each argument may be a directory (scanned for
`resistivity_block_iter*.dat` and `distortion_iter*.dat`) or an individual
file path; multiple arguments may be mixed freely. Files are paired by the
iteration number embedded in their name (`iter<N>`, via a regex, e.g.
`resistivity_block_iter3.dat` pairs with `distortion_iter3.dat`).

The ocean region (region 1) is always auto-inferred via the heuristic
`flag == 1 AND rho <= 1 Ω·m` — there is currently no way to override this
from the CLI. (The underlying `parse_resistivity_block()` function does
accept an `ocean: bool | None` keyword for programmatic overriding — see
below — it's just not exposed as a command-line option.)

---

## Programmatic usage

The script's own functions can be imported and called directly; there is
no separate `femtic.py`-based API.

```python
from femtic_summarize_model_cells import parse_resistivity_block, parse_distortion

rho  = parse_resistivity_block("resistivity_block_iter0.dat")   # dict, see below
dist = parse_distortion("distortion_iter0.dat")                  # dict, see below

n_total = rho["n_rho"] + dist["n_distortion"]
```

`parse_resistivity_block(path, *, ocean=None)` — pass `ocean=True` or
`ocean=False` to force the ocean-region interpretation instead of using
the `flag == 1 AND rho <= 1 Ω·m` heuristic.

---

## Output format

For each iteration, `print_summary()` prints a fixed-width report, e.g.:

```
  ----------------------------------------------------
  Iteration       : 0
  Total elements  :    254,016  (tetrahedra)
  Total regions   :        312  (resistivity blocks)
  Ocean inferred  : yes  (rho = 0.3 Ohm.m)
  ----------------------------------------------------
  Air regions                       1
  Ocean regions                     1
  Other fixed regions               0
  Free rho regions                310  <- n_rho
  Free sites (distort.)           120  x 4 values
  Distortion params                480  <- n_dist
  ======================================================
  Jacobian columns                 790  = n_rho + n_dist
  ----------------------------------------------------

  1 iteration(s) processed.
```

If no matching `distortion_iter<N>.dat` file is found for an iteration,
the "Distortion params" row instead prints `n/a (no distortion_iterNN.dat)`
and `n_total` is just `n_rho`. Parse errors for an individual iteration are
caught, reported to stderr as `[error] iter <N>: ...`, and do not abort the
run — remaining iterations are still processed.

---

## Region conventions

| Region | Category    | Condition |
|---|---|---|
| 0     | Air          | Always treated as fixed |
| 1     | Ocean        | Fixed when `flag == 1` **and** `rho <= 1 Ω·m` (heuristic, region-1-only) |
| 2 …   | Other fixed  | `flag == 1` |
| 2 …   | Free (`n_rho`) | `flag == 0` |

The ocean heuristic is only ever applied to region 1; region 0 is always
air regardless of its flag/ρ values.

---

## File formats expected

**`resistivity_block_iterXX.dat`:**
```
nelem  nreg
ielem  iregion          <- nelem lines, element -> region index (skipped, not parsed in detail)
ireg  rho  ...  ...  ...  flag   <- nreg lines; only columns 0 (ireg), 1 (rho), 5 (flag) are read
```

**`distortion_iterXX.dat`:**
```
nsites
...  ...  ...  ...  ...  flag   <- one line per site; only column 5 (flag) is read
                                    flag == 0 -> free site (contributes 4 distortion params)
```

---

## Return dict keys

**`parse_resistivity_block()`:**

| Key | Type | Description |
|---|---|---|
| `nelem` | int | Total number of mesh elements |
| `nreg` | int | Number of regions |
| `n_air` | int | Always `1` (region 0) |
| `n_ocean` | int | `1` if an ocean region was identified, else `0` |
| `n_other_fixed` | int | Count of other fixed (`flag == 1`) regions, excluding air/ocean |
| `n_rho` | int | Count of free (`flag == 0`) regions — the resistivity contribution to the Jacobian column count |
| `ocean_present` | bool | Whether region 1 was treated as ocean |
| `ocean_rho` | float \| None | Resistivity of region 1 (Ω·m) if ocean, else `None` |

**`parse_distortion()`:**

| Key | Type | Description |
|---|---|---|
| `nsites` | int | Total number of sites (first line of the file) |
| `n_free_sites` | int | Sites with `flag == 0` |
| `n_fixed_sites` | int | Sites with `flag != 0` |
| `n_distortion` | int | `n_free_sites * 4` — the distortion contribution to the Jacobian column count |

---

## Parameter summary output

At the start of `main()`, after resolving the CLI arguments, the script
writes `femtic_summarize_model_cells_summary.md` (next to the script
itself) via a small, self-contained `_write_param_summary()` helper —
stdlib only, no dependency on `util.py`/`femtic.py`. It records the
resolved `paths` argument list, the script's absolute path, and the run
date/time.

---

## Dependencies

None beyond the Python standard library (`os`, `re`, `sys`, `pathlib`).
There is no optional `femtic.py` delegation — the parsing logic here is
the only implementation.

---

## Provenance

| Date | Author | Note |
|---|---|---|
| 2026-06-08 | Claude Sonnet 4.6 (Anthropic) | Created (standalone). |
| 2026-08-13 | Claude Sonnet 5 (Anthropic) | Added `femtic_summarize_model_cells_summary.md` output after CLI argument parsing: writes the resolved `paths` argument, script path, and run date/time via a self-contained, stdlib-only helper. |
| 2026-08-13 | Claude Sonnet 5 (Anthropic) | Rewrote this README to match the script's actual current implementation: removed references to a `femtic.py`-delegated `summarise_model_file()`/`_print_model_summary()` API, an `argparse`-based CLI, and a `--ocean` flag, none of which exist in the installed script. Documented the real `sys.argv`-based CLI, the real `parse_resistivity_block()`/`parse_distortion()` return dicts, and the real fixed-width `print_summary()` output (including the distortion/Jacobian-column reporting, which the previous README did not mention at all). |
