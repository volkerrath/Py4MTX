# mt_archive_run

Clean and optionally archive (currently FEMTIC-style) iteration directories.

This utility keeps selected low- and high-iteration files, preserves protected files, deletes the rest (optionally), and creates a compressed archive (`.zip`, `.tgz`, `.tar.gz`) of the retained files.

---

## ✨ Features

- Select files by iteration pattern (default: `_iter(\d+)`), matched
  regardless of extension -- `model_iter12.dat` and `model_iter12.h5`
  are both treated as iteration files
- Keep:
  - lowest `keep_n_low` iterations (0 is allowed -- keep none from
    the low end)
  - highest `keep_n_high` iterations (must be >= 1 -- you always keep
    at least the highest iteration)
  - selection is done **per containing directory**, so running with
    `--recursive` over several run subdirectories (an "ensemble")
    keeps each run's own lowest/highest iterations, even if runs
    don't share the same iteration count
- Never delete protected files:
  - by substring tokens (e.g. `obs`, `mesh`)
  - by suffix (e.g. `.log`, `.cnv`)
  - by exact filename (e.g. `mesh.h5`, `jacobian.h5`, `rough.h5`)
- Case-insensitive matching
- Safe by default (`dry-run`): without `--delete`, no source file is
  ever touched -- this applies independently of `--compress`, so a
  real archive can be written while every source file is left as-is
- Optional deletion (`--delete`) of the source files, entirely
  independent of archiving
- Optional compression:
  - `.zip`
  - `.tgz` / `.tar.gz`
  - always writes a real, on-disk archive containing only the
    kept-file selection, whether or not `--delete` is also given
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
jacobian.h5
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

### Keep only the highest iteration (drop the low end entirely)
```
python mt_archive_run.py . \
    --keep-n-low 0 \
    --keep-n-high 1
```
`--keep-n-low 0` is allowed and keeps nothing from the low end of the
matched iterations. `iter0` is still kept regardless, though -- it's
protected by the default `iter0` token independently of
`--keep-n-low`/`--keep-n-high` (see Protected Files below). If you
truly need to discard `iter0` too, drop `iter0` from
`protected_tokens` (no `--protected-tokens` CLI flag is exposed yet;
call `mt_archive_run()` directly, or ask for the flag to be added).
`--keep-n-high` must always be >= 1 -- you always keep at least the
highest iteration.

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

`--compress` and `--delete` are independent: `--compress` always
writes a real archive containing the kept-file selection, whether or
not `--delete` is also given. Use `--compress` alone for a genuine,
already-smaller archive with every source file left exactly as it
was; add `--delete` if you also want the unwanted iterations removed
from the source directories.

### Create archive without touching the source files
```
python mt_archive_run.py run --compress run_cleaned.tgz
```
This writes a real, already-reduced `run_cleaned.tgz` containing only
the kept iterations plus protected files. `run/` itself is left
completely unchanged -- every file, including the ones not in the
archive, is still there.

### Create archive and clean up the source directory
```
python mt_archive_run.py run \
    --delete \
    --compress run_cleaned.tgz
```
Same archive contents as above, but the unwanted iteration files are
also deleted from `run/` afterwards.

### Supported formats
- `.zip`
- `.tgz`
- `.tar.gz`

### ⚠️ Symlinks: use `.tgz`/`.tar.gz`, not `.zip`

If your run directories contain symlinks (common if you link in
shared mesh/model files, ensemble members, or scratch-space outputs),
**prefer `.tgz`/`.tar.gz` over `.zip`**:

- `.tgz`/`.tar.gz` (via Python's `tarfile`) preserve symlinks as real
  symlink entries, including broken/dangling ones -- nothing is
  followed or duplicated.
- `.zip` (via Python's `zipfile`) has no concept of a symlink: a valid
  symlink is dereferenced and stored as a full duplicate copy of its
  target's content (silently, and possibly bloating the archive if the
  same target is linked from many places), and a broken/dangling
  symlink cannot be stored at all. The script prints a warning
  whenever `--compress` targets a `.zip` file, and skips (with a
  printed message) any file it can't add rather than aborting the
  whole archive -- but the safer option is simply to use `.tgz` when
  links matter.

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
mesh.h5, jacobian.h5, rough.h5
```

Examples:
```
mesh.cnv          → protected (suffix match)
mesh.h5           → protected (exact filename match)
rough.h5          → protected (exact filename match)
jacobian.h5       → protected (exact filename match)
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

## 🚫 Excluded and ✅ Always-Included Subdirectories

`--exclude-dirs` only applies when `--recursive` is set -- a
non-recursive run never looks inside subdirectories in the first
place. `--always-include-dirs` gets one extra check that runs
unconditionally (see below), but finding it at deeper nesting or
under more than one distinct parent still needs `--recursive`.

#### `--exclude-dirs` (default: `plots`)

Any subdirectory whose name (case-insensitive) matches one of these,
at any depth, is skipped entirely -- not scanned, not deleted, not
archived. Useful for plot/figure output that has nothing to do with
the inversion state and doesn't need to travel with the archive.

```
python mt_archive_run.py run --recursive --exclude-dirs plots figures
```

#### `--always-include-dirs` (default: `templates python`)

Any subdirectory whose name (case-insensitive) matches one of these,
at any depth under `--recursive`, is **not scanned at all** --
exactly like `--exclude-dirs`, no file inside it is checked against
the iteration pattern or the protection rules, and nothing inside it
is ever deleted. Unlike `--exclude-dirs`, though, the whole directory
is located and added to the compressed archive **untouched**, as a
single recursive directory add (preserving its internal structure,
permissions, and symlinks -- with `.tgz`/`.tar.gz`; see the symlink
note under Compression for `.zip`). Useful for template files or
accompanying Python scripts that belong with the run but are not run
output and shouldn't be subject to per-file iteration logic at all.

**This one extra check runs even without `--recursive`:** the script
always looks for `--always-include-dirs` names as *immediate
children of the archive's base directory* -- the common parent when
several directories or a wildcard are given, or the directory itself
otherwise. This matches the usual ensemble layout, where `templates/`
and `python/` sit as siblings of `run_1`, `run_2`, ... directly under
one ensemble folder:

```
ensemble_x/
  run_1/  run_2/  ...
  templates/
  python/
```

```
python mt_archive_run.py ensemble_x/run_* --keep-n-low 1 --keep-n-high 1 \
    --compress ensemble_x_cleaned.tgz
```

Even though this targets the `run_*` directories directly and never
passes `--recursive`, `ensemble_x/templates` and `ensemble_x/python`
are still found (as `ensemble_x` is the runs' common parent) and
added to the archive untouched. `--recursive` is still needed if your
always-include directories sit deeper than one level down, or your
matched directories don't share a single common parent.

```
python mt_archive_run.py run --recursive --always-include-dirs templates python scripts
```

Example directory:
```
run/
  model_iter0.dat ... model_iter50.dat
  plots/            <- excluded entirely (--exclude-dirs)
    misfit_iter30.png
  templates/        <- not scanned, archived untouched (--always-include-dirs)
    control_template_iter0.dat
  python/           <- not scanned, archived untouched (--always-include-dirs)
    postprocess.py
```

Here `plots/misfit_iter30.png` never appears anywhere (not scanned,
not deleted, not archived). `templates/` and `python/` are likewise
never scanned -- `control_template_iter0.dat` is never evaluated
against the iteration pattern, so it's neither "kept" nor "deleted" in
the usual sense -- but both directories are added to the archive in
their entirety, exactly as they are on disk.

---

## 🗂️ Running From Above an Ensemble of Runs

Yes -- this can be run from a parent directory that sits above several
FEMTIC run subdirectories, e.g.:

```
ensemble/
  run_001/  (model_iter0..iterN.dat/.h5, mesh.h5, rough.h5, jacobian.h5, run.log, ...)
  run_002/  (model_iter0..iterM.dat/.h5, mesh.h5, rough.h5, jacobian.h5, run.log, ...)
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
`rough.h5`, `jacobian.h5`, logs, observation files, etc.) are preserved in
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
    mesh.h5  rough.h5  jacobian.h5  run.log  observations.dat
  run_002/
    model_iter0.dat  model_iter0.h5   (protected: iter0 is a protected token)
    model_iter1.dat  model_iter1.h5   <- highest iteration in this run
    mesh.h5  rough.h5  jacobian.h5  run.log
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
`iter0.*`, `mesh.h5`, `rough.h5`, `jacobian.h5`, `run.log`, and
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

### Alternative: Target the Individual Run Directories Directly (Wildcards)

`--recursive` walks down from a shared parent to *find* the run
directories. If you already know which ones you want, you can instead
pass them -- or a wildcard pattern matching them -- straight to the
`directory` argument, with no `--recursive` needed:

```
python mt_archive_run.py ensemble/run_* --keep-n-low 1 --keep-n-high 1 \
    --delete --compress ensemble_cleaned.tgz
```

This gives the identical result as the `--recursive` example above:
each matched directory (`ensemble/run_001`, `ensemble/run_002`, ...)
gets its own independent keep-lowest/keep-highest selection, and the
archive nests each run under its own name (`ensemble/run_001/...`,
`ensemble/run_002/...`) using the matched directories' common parent
as the base. Unlike `--recursive`, it never looks inside
`ensemble/other_stuff/` or any other sibling directory the wildcard
doesn't match -- so it's a good fit when the ensemble folder contains
things alongside the runs that you don't want scanned at all.

The `directory` argument accepts one or more paths, and more than one
wildcard pattern can be given at once (e.g. `run_* backup_run_*`).
If your shell already expands the pattern (the normal case in bash),
that's all you need. If you'd rather this script do the expansion
itself -- e.g. to avoid a "no matches" error from an empty shell glob,
or on a shell that doesn't expand wildcards -- quote the pattern:

```
python mt_archive_run.py "ensemble/run_*" --keep-n-low 1 --keep-n-high 1
```

Both forms behave identically. A pattern (quoted or not) that matches
nothing prints a warning and is skipped rather than aborting the run;
a non-directory match (a stray file caught by a loose pattern) is
skipped the same way.

## ⚙️ Options

| Option | Description |
|------|-------------|
| `directory` | One or more directories to process, or wildcard pattern(s) matching several (e.g. `ensemble/run_*`); quote a pattern to have this script expand it instead of the shell (default: `.`) |
| `--keep-n-low` | Number of lowest iterations to keep. May be 0 (keep none from the low end). Default: 1 |
| `--keep-n-high` | Number of highest iterations to keep. Must be >= 1. Default: 1 |
| `--recursive` | Scan subdirectories |
| `--delete` | Actually remove the unwanted iteration files from the source directories (otherwise dry run -- nothing on disk is touched). Independent of `--compress`. |
| `--compress` | Output archive path (`.zip`, `.tgz`, `.tar.gz`). Always writes a real archive with the kept-file selection, whether or not `--delete` is given. |
| `--no-root` | Do not include leading directory in archive |
| `--exclude-dirs` | Subdirectory names (case-insensitive) to skip entirely when `--recursive` is set: not scanned, not deleted, not archived. Default: `plots` |
| `--always-include-dirs` | Subdirectory names (case-insensitive) that are never scanned, but added to the compressed archive untouched, as whole directories. Always checked as immediate children of the archive's base directory (no `--recursive` needed for that); `--recursive` is needed for deeper nesting or more than one distinct parent. Default: `templates python` |

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
| 2026-09-11 | Claude Sonnet 5 (Anthropic) | Added `--exclude-dirs` (default: `plots`) to prune whole subdirectories from the `--recursive` walk (not scanned, not deleted, not archived), and `--always-include-dirs` (default: `templates python`) to always fully protect and include whole subdirectories in the archive. Added a printed warning when `--compress` targets a `.zip` file, since `zipfile` dereferences symlinks (duplicating target content) and fails outright on broken symlinks, whereas `.tgz`/`.tar.gz` (`tarfile`) preserves symlinks -- including broken ones -- as real symlink entries; zip writing now also skips an unstorable file with a message instead of aborting the whole archive. |
| 2026-09-11 | Claude Sonnet 5 (Anthropic) | Corrected `--always-include-dirs`: these directories are now never scanned at all (like `--exclude-dirs`), rather than scanned file-by-file and marked protected; each is located as a whole and added to the archive untouched via a recursive directory add. |
| 2026-09-15 | Claude Sonnet 5 (Anthropic) | Iteration matching was already extension-agnostic (any `_iterN` file, `.dat`, `.h5`, or otherwise, is treated the same) -- no code change needed there. Changed the default protected filenames from `mesh.h5`, `rough.h5`, `jac.h5` to `mesh.h5`, `jacobian.h5`, `rough.h5` (`jac.h5` is no longer protected by default). Dry runs (i.e. whenever `--delete` is not given) now print an explicit "Files that will be kept" list, independent of whether `--compress` is also given. |
| 2026-09-15 | Claude Sonnet 5 (Anthropic) (same-day follow-up) | Added support for running directly on top of individual ensemble run directories via wildcards (e.g. `ensemble/run_*`), instead of only via `--recursive` from a shared parent. `directory` now accepts one or more paths/patterns; unmatched or non-directory patterns are skipped with a warning rather than aborting. Each matched directory keeps its own keep-lowest/keep-highest selection, and archive paths are computed relative to the matched directories' common parent, matching `--recursive`'s output structure. |
| 2026-09-15 | Claude Sonnet 5 (Anthropic) (third same-day follow-up) | Decoupled `--compress` from `--delete`. Previously, `--compress` without `--delete` only printed a "WOULD ADD" preview and never wrote a real archive file. `--compress` now always writes a real, on-disk archive containing the kept-file selection regardless of `--delete` -- so `--compress` alone gives a genuine, already-smaller archive with every source file left untouched. `--delete` continues to control only whether the unwanted iteration files are removed from the source directories. |
| 2026-09-15 | Claude Sonnet 5 (Anthropic) (fourth same-day follow-up) | `--always-include-dirs` names are now also looked for as immediate children of the archive's base directory (the common parent when several directories/wildcards are given, or the directory itself otherwise), via an always-on, non-recursive check -- not just when `--recursive` is set. Matches the usual ensemble layout (`run_1`, `run_2`, ..., `templates/`, `python/` as siblings under one ensemble folder): targeting the run directories directly via a wildcard, e.g. `ensemble_x/run_*`, now still picks up `ensemble_x/templates` and `ensemble_x/python` with no `--recursive` needed. `--recursive` is still required for always-include directories nested deeper than one level, or spread under more than one distinct parent. |
| 2026-09-15 | Claude Sonnet 5 (Anthropic) (fifth same-day follow-up) | `--keep-n-low` may now be `0` ("keep none of the matched iteration files from the low end"), instead of rejecting anything below `1`. `--keep-n-high` still must be `>= 1` -- you always keep at least the highest iteration -- so this is an asymmetric relaxation. `iter0` is unaffected either way: it's kept via its own `iter0` protected-token match, independent of `--keep-n-low`/`--keep-n-high`. Also fixed a latent slicing bug this change would otherwise have exposed: Python's `list[-0:]` is the *whole* list (since `-0 == 0`), not an empty one, so a `keep_n_high == 0` could never have correctly meant "keep none from the high end" without an explicit guard -- added defensively even though the CLI no longer allows `keep_n_high == 0` to reach it. |

**Note on AI assistance:** this script and README were produced with the
help of AI tools (ChatGPT and Claude, see table above) and have not
been independently verified for production use. Review the file
selection/deletion logic before running with `--delete` on data that
cannot be regenerated.
