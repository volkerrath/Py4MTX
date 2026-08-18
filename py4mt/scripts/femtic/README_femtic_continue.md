# femtic_continue.py

Prepares one or more FEMTIC run directories for **continuation** from the
last completed iteration, by rewriting the `ITERATION` block in
`control.dat`.

## What it does

For every directory in `DIR_LIST`:

1. **Detects the last completed iteration** by scanning the directory for
   per-iteration output files (`ITER_FILE_GLOB`) and extracting the
   iteration number from each filename (`ITER_NUMBER_REGEX`). The highest
   number found is taken as the restart point.
2. **Backs up the template link.** `control.dat` is normally a symlink
   into a shared template directory. That symlink is preserved, unchanged,
   as `control.orig` in the same directory — so the pristine template
   reference is never lost. If `control.orig` already exists it is left
   alone (unless `OVERWRITE_ORIG = True`).
3. **Rewrites `control.dat`** as a plain (non-symlink) file: the line
   following the `ITERATION` keyword,

   ```
   ITERATION
   0 30
   ```

   becomes

   ```
   ITERATION
   12 30
   ```

   where `12` is the detected last iteration and `30` is `MAX_ITERATIONS`
   (a user parameter — always written, whether or not it matches the
   template's original value). Indentation, spacing, and any trailing
   inline comment on that line are preserved.

Directories that don't exist, don't contain `control.dat`, or have no
matching iteration files are skipped with a message; nothing else in
`DIR_LIST` is affected.

## Usage

Edit the **USER SECTION** at the top of the script, then run it directly
(no CLI arguments — following Py4MTX convention):

```bash
python3 femtic_continue.py
```

### USER SECTION parameters

| Parameter               | Purpose                                                                 | Default                        |
|--------------------------|--------------------------------------------------------------------------|---------------------------------|
| `DIR_LIST`               | List of run directories to process                                       | *(must be filled in)*          |
| `CONTROL_FILENAME`       | Name of the control file to adapt                                        | `"control.dat"`                |
| `ORIG_FILENAME`          | Name used for the preserved template link                                | `"control.orig"`               |
| `ITER_FILE_GLOB`         | Glob pattern (per directory) matching per-iteration output files         | `"resistivity_iter*.dat"`      |
| `ITER_NUMBER_REGEX`      | Regex with one capture group extracting the iteration number from a filename matched by `ITER_FILE_GLOB` | `r"iter0*(\d+)"` |
| `MAX_ITERATIONS`         | Second number in the `ITERATION` block (max iterations to run to); always written | `30`                            |
| `OVERWRITE_ORIG`         | Recreate `control.orig` even if it already exists                        | `False`                        |
| `DRY_RUN`                | Print planned actions without touching any files                        | `False`                        |
| `VERBOSE`                | Print per-directory progress                                             | `True`                         |

**Important:** `ITER_FILE_GLOB` / `ITER_NUMBER_REGEX` must match the
actual per-iteration output naming used in the FEMTIC source tree that
produced the run — this can differ between `femtic_v4_src`,
`femtic_v5_src`, and the `femtic_dabic_v1.x_src` variants. Check/adjust
before running on a new tree.

## Safety notes

- Re-running the script on an already-processed directory is safe: it
  will re-detect the (possibly higher) last iteration, but will **not**
  overwrite `control.orig` unless `OVERWRITE_ORIG = True`, so the original
  template link is never lost.
- Use `DRY_RUN = True` first to confirm detected iteration numbers before
  writing anything.
- `control.dat` stops being a symlink after processing (it becomes a
  regular file with the updated `ITERATION` block). To restore the
  pristine template, remove `control.dat` and re-link it from
  `control.orig`.

## Changelog

| Date       | Description                                   |
|------------|------------------------------------------------|
| 2026-08-18 | Initial version                                 |
| 2026-08-18 | Made max-iterations count a user parameter (`MAX_ITERATIONS`), always applied instead of an optional override |

---
Author: Volker Rath (DIAS) with Claude
