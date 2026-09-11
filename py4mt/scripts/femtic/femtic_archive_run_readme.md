# mt_archive_run

Clean and optionally archive (currently FEMTIC-style) iteration directories.

This utility keeps selected low- and high-iteration files, preserves protected files, deletes the rest (optionally), and creates a compressed archive (`.zip`, `.tgz`, `.tar.gz`) of the retained files.

---

## ✨ Features

- Select files by iteration pattern (default: `_iter(\d+)`), matched
  regardless of extension -- `model_iter12.dat` and `model_iter12.h5`
  are both treated as iteration files
- Keep:
  - lowest `keep_n_low` iterations
  - highest `keep_n_high` iterations
  - selection is done **per containing directory**, so running with
    `--recursive` over several run subdirectories (an "ensemble")
    keeps each run's own lowest/highest iterations, even if runs
    don't share the same iteration count
- Never delete protected files:
  - by substring tokens (e.g. `obs`, `mesh`)
  - by suffix (e.g. `.log`, `.cnv`)
  - by exact filename (e.g. `mesh.h5`, `rough.h5`, `jac.h5`)
- Case-insensitive matching
- Safe by default (`dry-run`)
- Optional deletion (`--delete`)
- Optional compression:
  - `.zip`
  - `.tgz` / `.tar.gz`
- Optional inclusion of leading directory in archive

---

## 📂 Typical Use Case

FEMTIC / MT inversion runs often produce many files like:

```
model_iter0.dat
model_iter0.h5
model_iter1.dat
model_iter1.h5
...
model_iter100.dat
model_iter100.h5
log.txt
mesh.cnv
mesh.h5
rough.h5
jac.h5
```

This tool allows you to:

- keep only selected iterations (e.g. first + last)
- retain important metadata/log files
- clean the directory
- archive the result

---

## 🚀 Usage

### Basic (dry-run)
```
python mt_archive_run.py .
```

### Keep lowest + highest iteration
```
python mt_archive_run.py . \
    --keep-n-low 1 \
    --keep-n-high 1
```

### Actually delete unwanted files
```
python mt_archive_run.py . --delete
```

### Recursive mode
```
python mt_archive_run.py results --recursive
```

---

## 📦 Compression

### Create archive (recommended)
```
python mt_archive_run.py run \
    --delete \
    --compress run_cleaned.tgz
```

### Supported formats
- `.zip`
- `.tgz`
- `.tar.gz`

---

## 📁 Archive Structure

### Default (includes leading directory)
```
python mt_archive_run.py run --compress out.tgz
```

Archive contents:
```
run/
  model_iter100.dat
  mesh.cnv
  log.txt
```

### Without leading directory
```
python mt_archive_run.py run --compress out.tgz --no-root
```

Archive contents:
```
model_iter100.dat
mesh.cnv
log.txt
```

---

## 🔒 Protected Files

### By default

#### Tokens (substring match, case-insensitive)
```
obs, ref, mesh, iter0, control
```

#### Suffixes (full filename match)
```
.log, .sh, .cnv
```

#### Exact filenames (case-insensitive, not extension/suffix matching)
```
mesh.h5, rough.h5, jac.h5
```

Examples:
```
mesh.cnv          → protected (suffix match)
mesh.h5           → protected (exact filename match)
rough.h5          → protected (exact filename match)
jac.h5             → protected (exact filename match)
run.log           → protected (suffix match)
observations.dat  → protected (token match)
model_iter42.h5   → NOT protected -- treated as an iteration file,
                     same as model_iter42.dat
```

Note: unlike earlier versions of this script, `.h5` is **not** a
blanket-protected suffix any more. Only the three named HDF5 files
above are protected; any other `*_iterN.h5` file is subject to the
same keep-lowest/keep-highest iteration logic as the `.dat` files.

---

## 🗂️ Running From Above an Ensemble of Runs

Yes -- this can be run from a parent directory that sits above several
FEMTIC run subdirectories, e.g.:

```
ensemble/
  run_001/  (model_iter0..iterN.dat/.h5, mesh.h5, rough.h5, jac.h5, run.log, ...)
  run_002/  (model_iter0..iterM.dat/.h5, mesh.h5, rough.h5, jac.h5, run.log, ...)
  ...
```

**`--recursive` is required in this case.** Without it, the script
only lists files directly inside the directory you pass it (a plain
glob, not a directory walk) -- and the `ensemble/` directory itself
has no `iterN` files, only its `run_XXX/` subdirectories do. Without
`--recursive`, running on `ensemble/` would simply find nothing to
clean.

```
python mt_archive_run.py ensemble --recursive --keep-n-low 1 --keep-n-high 1 \
    --delete --compress ensemble_cleaned.tgz
```

Iteration selection is applied **per containing directory**, not
pooled across the whole tree, so `run_001` and `run_002` each keep
their own lowest/highest iterations even if one run went to iteration
50 and another stopped at iteration 12. Protected files (`mesh.h5`,
`rough.h5`, `jac.h5`, logs, observation files, etc.) are preserved in
every run subdirectory independently. The resulting archive mirrors
the retained directory tree (`ensemble/run_001/...`,
`ensemble/run_002/...`), so nothing from different runs gets mixed
together.

### Worked example

Given:

```
ensemble/
  run_001/
    model_iter0.dat  model_iter0.h5   (protected: iter0 is a protected token)
    model_iter1.dat  model_iter1.h5
    model_iter2.dat  model_iter2.h5
    model_iter3.dat  model_iter3.h5   <- highest iteration in this run
    mesh.h5  rough.h5  jac.h5  run.log  observations.dat
  run_002/
    model_iter0.dat  model_iter0.h5   (protected: iter0 is a protected token)
    model_iter1.dat  model_iter1.h5   <- highest iteration in this run
    mesh.h5  rough.h5  jac.h5  run.log
```

Running:

```
python mt_archive_run.py ensemble --recursive --keep-n-low 1 --keep-n-high 1 \
    --delete --compress ensemble_cleaned.tgz
```

prints, per run subdirectory:

```
Kept iterations in ensemble/run_001: [1, 3]
Kept iterations in ensemble/run_002: [1]
```

and deletes `model_iter2.*` from `run_001` (not present in `run_002`
at all), while keeping `model_iter1.*`/`model_iter3.*` in `run_001`
and `model_iter1.*` in `run_002` -- plus the always-protected
`iter0.*`, `mesh.h5`, `rough.h5`, `jac.h5`, `run.log`, and
`observations.dat` files in both. `ensemble_cleaned.tgz` then contains
the retained files under their original `run_001/...`/`run_002/...`
paths.

Caveats:
- This groups strictly by *immediate containing directory*. If a
  single run's files are themselves split across further
  subdirectories you don't want treated as separate "runs" (unlikely
  in a normal FEMTIC layout), each of those subdirectories gets its
  own independent keep-lowest/keep-highest selection.
- Always do a dry run (the default, i.e. without `--delete`) first
  and check the printed `Kept iterations in <dir>: [...]` lines for
  every run subdirectory before adding `--delete`.

## ⚙️ Options

| Option | Description |
|------|-------------|
| `directory` | Directory to process (default: `.`) |
| `--keep-n-low` | Number of lowest iterations to keep |
| `--keep-n-high` | Number of highest iterations to keep |
| `--recursive` | Scan subdirectories |
| `--delete` | Actually delete files (otherwise dry-run) |
| `--compress` | Output archive path |
| `--no-root` | Do not include leading directory in archive |

---

## 🧠 Notes

- Matching is **case-insensitive**
- Only files matching the iteration pattern are filtered
- Protected files are **never deleted**
- Archive paths are **relative** (portable)
- Default behavior is **safe (no deletion)**

---

## 🔧 Example Workflow

```
# preview
python mt_archive_run.py run

# clean + archive
python mt_archive_run.py run \
    --keep-n-low 1 \
    --keep-n-high 2 \
    --delete \
    --compress run_final.tgz
```

---

## 📜 License / Provenance

Author: Volker Rath (DIAS)  
Created with the help of ChatGPT (GPT-5 Thinking) on 2026-04-07

| Date | Author | Change |
|---|---|---|
| 2026-04-07 | vrath / ChatGPT (GPT-5 Thinking) | Created. |
| 2026-08-13 | Claude Sonnet 5 (Anthropic) | Added `femtic_archive_run_summary.md` output after CLI argument parsing: writes the resolved parameters (`directory`, `keep_n_low`, `keep_n_high`, `recursive`, `delete`, `compress`, `no_root`), script path, and run date/time via a self-contained, stdlib-only helper. |
| 2026-09-11 | Claude Sonnet 5 (Anthropic) | HDF5 support: removed the blanket `.h5` suffix protection and replaced it with exact-filename protection for `mesh.h5`, `rough.h5`, `jac.h5`; `iterX.h5` is now matched by the iteration regex like `iterX.dat`. Fixed iteration keep-selection to be grouped per containing directory instead of pooled globally, so `--recursive` gives correct results when run above an ensemble of run subdirectories with differing iteration counts. |

**Note on AI assistance:** this script and README were produced with the
help of AI tools (ChatGPT and Claude, see table above) and have not
been independently verified for production use. Review the file
selection/deletion logic before running with `--delete` on data that
cannot be regenerated.
