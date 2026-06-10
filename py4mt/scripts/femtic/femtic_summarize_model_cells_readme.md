# femtic_summarize_model_cells.py

Summarise the number of air, ocean, other-fixed, and free-parameter cells
in one or more FEMTIC `resistivity_block_iterXX.dat` model files.

---

## Purpose

Quickly inspect the element composition of a FEMTIC resistivity block without
running a full inversion.  Useful for:

- verifying that the ocean region was correctly identified before a run,
- checking the number of free parameters before choosing regularisation,
- comparing cell counts across iterations or ensemble members.

When `femtic.py` is importable the script delegates to
`femtic.summarise_model_file()` and `femtic._print_model_summary()`.
Otherwise it falls back to a self-contained parser with no external
dependencies.

---

## Usage

```bash
# Single file
python femtic_summarize_model_cells.py resistivity_block_iter0.dat

# Glob (shell expands)
python femtic_summarize_model_cells.py resistivity_block_iter*.dat

# Scan a directory for all resistivity_block_iter*.dat
python femtic_summarize_model_cells.py /path/to/run/

# Force ocean / no-ocean interpretation of region 1
python femtic_summarize_model_cells.py resistivity_block_iter0.dat --ocean yes
python femtic_summarize_model_cells.py resistivity_block_iter0.dat --ocean no
```

Default (no `--ocean` flag): auto-infer from the heuristic `flag==1 AND rho ≤ 1 Ω·m`.

---

## Programmatic usage (via `femtic.py`)

```python
import femtic as fem

s = fem.summarise_model_file("resistivity_block_iter0.dat")
# s["n_params"]   → number of free-parameter elements
# s["n_ocean"]    → number of ocean elements
# s["ocean_present"] → bool

# Re-print the table from a cached dict
fem._print_model_summary(s)
```

`summarise_model_file` accepts `ocean=True/False` to override the heuristic.

---

## Output format

```
────────────────────────────────────────────────────────────
  File          : resistivity_block_iter0.dat
  Total cells   :    254 016   (312 regions)
  Ocean inferred: yes (rho=0.3 Ω·m)
  ┌─────────────────────────────────────────┐
  │  Air cells        :     42 880          │
  │  Ocean cells      :     18 432          │
  │  Other fixed      :          0          │
  │  Parameters (free):    192 704          │
  │─────────────────────────────────────────│
  │  Check sum        :    254 016          │
  └─────────────────────────────────────────┘
```

A `*** WARNING ***` line is printed if the check sum does not equal `nelem`
(indicates a parsing or region-assignment problem).

---

## Region conventions

| Region | Category | Condition |
|---|---|---|
| 0 | Air | Always fixed |
| 1 | Ocean | Fixed when `flag==1` AND `rho ≤ 1 Ω·m` (heuristic) |
| 2 … | Other fixed | `flag == 1` |
| 2 … | Parameters | `flag == 0` (free) |

The ocean heuristic can be overridden with `--ocean yes/no` (CLI) or
`ocean=True/False` (Python API).

---

## File format expected

```
nelem  nreg
ielem  iregion      ← nelem lines, 0-based element → region index
ireg  rho  rho_lo  rho_hi  n  flag   ← nreg lines
```

---

## Return dict keys (`summarise_model_file`)

| Key | Type | Description |
|---|---|---|
| `file` | str | Filename (basename) |
| `nelem` | int | Total number of mesh elements |
| `nreg` | int | Number of regions |
| `n_air` | int | Elements assigned to region 0 |
| `n_ocean` | int | Elements assigned to region 1 (when ocean present) |
| `n_other_fixed` | int | Elements in other fixed regions |
| `n_params` | int | Elements in free (inversion) regions |
| `ocean_present` | bool | Whether region 1 was treated as ocean |
| `ocean_rho` | float \| None | Resistivity of region 1 (Ω·m) if ocean |
| `region_rho` | list[float] | Resistivity per region |
| `region_flag` | list[int] | Flag per region (0 = free, 1 = fixed) |

---

## Dependencies

None beyond the Python standard library (uses only `pathlib`, `collections`).
When `femtic.py` is on the path, its `_parse_region_line` and
`_infer_ocean_present` are used instead of the fallback implementations.

---

## Provenance

| Date | Author | Note |
|---|---|---|
| 2026-06-08 | Claude Sonnet 4.6 (Anthropic) | Created (standalone). |
| 2026-06-10 | Claude Sonnet 4.6 (Anthropic) | Refactored: delegates to `femtic.summarise_model_file` / `femtic._print_model_summary` when `femtic.py` is on the path; self-contained fallback kept.  Added `--ocean` CLI flag; `argparse` replaces raw `sys.argv`. |
