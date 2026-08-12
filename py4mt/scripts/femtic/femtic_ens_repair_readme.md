# femtic_ens_repair.py

Convergence diagnostic and repair for a FEMTIC ensemble.

Scans a FEMTIC ensemble directory tree (same `ENSEMBLE_DIR`/`ENSEMBLE_NAME`
convention as `femtic_ens_post.py`, `femtic_rto_prep.py`,
`femtic_gst_prep.py`) and, unlike `femtic_ens_post.py`, reports on
**every** scanned directory — not just converged ("accepted") members:

1. **Convergence diagnostic** — a bar chart or binned histogram of
   per-member nRMS (from `femtic.cnv`), colour-coded by status
   (`accepted` / `rejected_nrms` / `missing_cnv` / `missing_model`), via
   `fviz.plot_convergence_bar()` / `fviz.plot_convergence_histogram()`
   (see `femtic_viz_readme.md`).

2. **REPAIR procedure** — for every non-converged directory, draws 2
   distinct converged members at random, averages their log₁₀(ρ) models
   element-wise, and writes that mean as a fresh starting model into a
   **new sibling directory** (`<original>_restart` by default). The
   original non-converged directory is left completely untouched under
   its original name — REPAIR only ever creates, it never renames or
   deletes.

This functionality was originally prototyped inside `femtic_ens_post.py`,
then split out into this standalone script once the design settled, so
`femtic_ens_post.py` stays focused on its own scope (summary statistics,
covariance, QC/statistics slice plots over converged members only).

---

## Workflow

```
Ensemble sub-directories  (rto_*, gst_*, member_*, …)
          │
          ▼
  Scan ALL directories  (not just accepted)
          │
          ├──▶ conv_list: label, dir, nrms, status
          │      status ∈ {accepted, rejected_nrms,
          │                missing_cnv, missing_model}
          │
          ├──▶  MOD_CONV = True
          │        → nRMS bar chart / histogram, every scanned dir
          │
          └──▶  MOD_REPAIR = True
                   → for each non-accepted dir:
                       pick 2 random accepted members
                       repaired = mean(log10ρ_1, log10ρ_2)
                       copytree(<dir>, <dir>_restart, symlinks=True)
                       overwrite <dir>_restart/resistivity_block_iter0.dat
                       (original <dir> left untouched)
```

---

## Configuration parameters

### Ensemble input

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ENSEMBLE_DIR` | str | — | Root directory containing ensemble sub-directories. |
| `ENSEMBLE_NAME` | str | — | Glob matched against sub-directory names. Only actual directories among the matches are scanned (`os.path.isdir()` checked in step (1)); a stray file that happens to match the glob is skipped. |
| `ENSEMBLE_PREFIX` | str | — | Prefix for default file/figure names. |
| `NRMS_MAX` | float | `1.5` | Members whose final nRMS exceeds this value are tagged `"rejected_nrms"`. |
| `FEMTIC` | str | `"5.0"` | FEMTIC version string; controls which `femtic.cnv` column holds nRMS (`"4.3"` → column 6, `"5.x"` → column 8). |

### Ocean / air handling (REPAIR only)

Must match the values used by the FEMTIC inversion that produced the
ensemble; passed straight through to `fem.insert_model` when writing each
repaired starting model.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `MOD_OCEAN` | bool / None | `None` | `None` = auto-infer; `True`/`False` forces ocean-present/absent. |
| `MOD_AIR_RHO` | float | `1.0e9` | Ω·m sentinel for air cells (region 0). |
| `MOD_OCEAN_RHO` | float | `0.25` | Ω·m sentinel for ocean cells (region 1). |

### Convergence diagnostic

| Parameter | Type | Default | Description |
|---|---|---|---|
| `MOD_CONV` | bool | `True` | Enable the convergence diagnostic. |
| `MOD_CONV_FILE` | str | `<ENSEMBLE_DIR><prefix>_convergence` | Extension-less output base path; one file per `MOD_PLOT_FORMAT` entry. |
| `MOD_CONV_PER_MEMBER` | bool | `False` | `True` → one-bar-per-member chart (`fviz.plot_convergence_bar`); `False` (default) → binned histogram (`fviz.plot_convergence_histogram`). |
| `MOD_CONV_BINS` | int / `"auto"` | `"auto"` | Histogram mode only. Number of equal-width nRMS bins spanning the *accepted* members' range only, or `"auto"` to use `MOD_CONV_NBINS_AUTO`. |
| `MOD_CONV_NBINS_AUTO` | int | `15` | Bin count used when `MOD_CONV_BINS="auto"`. |
| `MOD_CONV_SHOW_MISSING` | bool | `False` | Histogram mode only. Show the aggregate "missing" bar for members with no usable nRMS. `False` (default) omits it — those members are still counted in the console log and figure title, just not drawn. |
| `MOD_CONV_HORIZONTAL` | bool | `True` | Per-member mode only. Horizontal vs. vertical bars. |
| `MOD_CONV_LOG` | bool | `False` | Log-scale the count axis (histogram mode) or the nRMS axis (per-member mode). |

Both rendering modes always: sort/bin by nRMS, draw a dashed line at
`NRMS_MAX`, and lump rejected members into one aggregate `"rejected"` bar
(histogram mode) rather than letting a single badly-diverged run stretch
the axis. See `femtic_viz_readme.md` for the full parameter reference of
both underlying functions.

### REPAIR

| Parameter | Type | Default | Description |
|---|---|---|---|
| `MOD_REPAIR` | bool | `False` | Enable the REPAIR procedure. Off by default — an explicit opt-in, since it writes new directories/files to disk. |
| `MOD_REPAIR_SUFFIX` | str | `"_restart"` | Appended to the directory's basename to form the new sibling directory, e.g. `rto_017` → `rto_017_restart`. The original `rto_017` directory is left exactly as it was found. |
| `MOD_REPAIR_MODEL_NAME` | str | `"resistivity_block_iter0.dat"` | Filename inside the copied `_restart` directory that gets overwritten with the repaired starting model. `"iter0"` matches FEMTIC's own starting-model naming convention, and the GST/RTO prep scripts' own `COPY_LIST` entry of the same name — so this file already exists as a real, physically-copied file right after the directory copy, and REPAIR simply replaces its contents. |
| `MOD_REPAIR_SEED` | int / None | `None` | RNG seed for the random member-pair draws. `None` = fresh entropy each run; int = reproducible. |
| `MOD_REPAIR_MIN_MEMBERS` | int | `2` | Minimum converged members required to draw a distinct pair from. REPAIR is skipped entirely (with a console warning) if fewer are available. |

**What REPAIR does, precisely, for each non-converged directory `<dir>`
(`status != "accepted"`):**

1. If `<dir>_restart` already exists, skip it (idempotent — safe to
   re-run; `<dir>` is left untouched either way).
2. If fewer than `MOD_REPAIR_MIN_MEMBERS` converged members exist across
   the whole ensemble, skip REPAIR entirely.
3. Draw 2 distinct converged members at random (uniform, without
   replacement, independently per non-converged directory — different
   `<dir>`s can draw overlapping or disjoint pairs).
4. Compute the element-wise mean of their **log₁₀(ρ)** models — the same
   space used throughout this codebase (RTO perturbations,
   `femtic_ens_post.py`'s mean/var/median/MAD).
5. `shutil.copytree(<dir>, <dir>_restart, symlinks=True)` — copies the
   **entire** original directory. `symlinks=True` preserves symlinked
   entries (the `LINK_LIST` files: `control.dat`, `mesh.dat`,
   `referencemodel.dat`, `distortion_iter0.dat`, `site.dat`, run
   scripts, …) as symlinks to their original shared targets, rather than
   following and duplicating them; `COPY_LIST` entries (`observe.dat`,
   `resistivity_block_iter0.dat`, …) come along as real, independent
   files, exactly as they were in the source.
6. Overwrite `MOD_REPAIR_MODEL_NAME` inside the **copy** with the
   averaged model, using the lowest-nRMS converged member's file as the
   format template (preserves header, bounds, and flag columns — same
   convention as `femtic_ens_post.py`'s `MOD_QC`/`MOD_STATS` blocks). A
   defensive check unlinks the target first if it's unexpectedly a
   symlink, so the write can never land on a shared template file.
7. `<dir>` itself is only ever *read* (by `shutil.copytree`) — never
   modified, renamed, or deleted.

Because the whole directory is copied first, `<dir>_restart` ends up
containing everything `<dir>` had — `femtic.cnv`, whatever partial
results existed, logs, plus all the `LINK_LIST`/`COPY_LIST` run
machinery — with only the starting model swapped out. It is a
ready-to-run restart directory, not just a bare model file.

### Plot output

| Parameter | Type | Default | Description |
|---|---|---|---|
| `MOD_PLOT_FORMAT` | str / list of str | `["pdf", "jpg"]` | One or more save formats; see `femtic_ens_post.py`'s `MOD_PLOT_FORMAT` docstring for the full supported list. |
| `MOD_DPI` | int | `200` | Figure DPI (raster formats only). |
| `MOD_TICK_FONTSIZE` / `MOD_LABEL_FONTSIZE` | int | `8` / `9` | Axis tick / axis-label & title font sizes. |
| `MOD_SHOW_IN_SPYDER` | bool | `True` | When running inside Spyder, also display every saved figure inline via `plt.show()`. No effect outside Spyder. |

### Verbose output

| Parameter | Type | Default | Description |
|---|---|---|---|
| `OUT` | bool | `True` | Print progress messages throughout. |

---

## Output files

- `<MOD_CONV_FILE>.<fmt>` — one convergence figure per `MOD_PLOT_FORMAT` entry (e.g. `..._convergence.pdf`, `..._convergence.jpg`).
- `<dir>_restart/` — one complete copy of `<dir>` per repaired non-converged directory, when `MOD_REPAIR=True`, with `MOD_REPAIR_MODEL_NAME` overwritten by the repaired starting model.

This script does **not** write a `.npz` results file — that remains
`femtic_ens_post.py`'s job, over converged members only.

---

## Quick start

```python
# Edit ENSEMBLE_DIR, ENSEMBLE_NAME, ENSEMBLE_PREFIX, NRMS_MAX, FEMTIC
# in the USER SECTION, then:
python femtic_ens_repair.py
```

Convergence diagnostic only (default `MOD_REPAIR=False`): just review
`<MOD_CONV_FILE>.pdf` to see which directories converged, which were
rejected on nRMS, and which are missing files entirely.

To also repair non-converged directories, set `MOD_REPAIR = True` and
re-run. Each repaired directory comes out ready to restart FEMTIC in
directly (`mesh.dat`/`control.dat`/etc. symlinks and `observe.dat` all
present via the copy, only the starting model swapped). Re-running again
is safe — already-repaired `_restart` directories are detected and
skipped.

---

## Relationship to other scripts

- **`femtic_ens_post.py`** — converged-members-only postprocessing
  (statistics, covariance, `.npz`, QC/statistics slice plots). Does not
  see or report on non-converged directories at all.
- **`femtic_viz.py`** — supplies `plot_convergence_bar()` and
  `plot_convergence_histogram()`, used by this script's convergence
  diagnostic.
- **`femtic_rto_prep.py`** / **`femtic_gst_prep.py`** — the ensemble
  *generation* scripts; REPAIR's `shutil.copytree(symlinks=True)` step
  relies on their `LINK_LIST`/`COPY_LIST` convention having already set
  each source directory up correctly (symlinked shared files vs. real
  per-member files) — REPAIR just carries that same split through into
  the `_restart` copy, it doesn't reconstruct it from scratch.

---

## Changelog

| Date | Author | Change |
|---|---|---|
| 2026-08-12 | Claude Sonnet 5 (Anthropic) | Created. Split out of `femtic_ens_post.py`, where a convergence diagnostic and a REPAIR procedure had been prototyped across several iterations earlier the same day (see `femtic_ens_post_readme.md`'s changelog for that history). Step (1)'s directory scan reuses `femtic_ens_post.py`'s scan loop, including its `os.path.isdir(d)` guard against non-directory glob matches. Step (2) (convergence diagnostic) defaults to the binned histogram with the aggregate `"missing"` bar turned **off** (`MOD_CONV_SHOW_MISSING=False`) at the user's request. Step (3) (REPAIR) creates a **new sibling** `_restart` directory per non-converged member rather than renaming the original in place, at the user's explicit request, so the original failed-run directory is preserved unmodified under its original name; REPAIR is skipped (with a console warning) if a target `_restart` directory already exists, or if fewer than `MOD_REPAIR_MIN_MEMBERS` converged members are available. The repaired model is the element-wise mean of 2 distinct converged members' **log₁₀(ρ)** models, matching this codebase's existing log10-space averaging convention. |
| 2026-08-12 | Claude Sonnet 5 (Anthropic) | REPAIR now `shutil.copytree()`'s the **entire** original directory to the `_restart` copy (`symlinks=True`, so `LINK_LIST` entries — `control.dat`, `mesh.dat`, `referencemodel.dat`, `distortion_iter0.dat`, `site.dat`, run scripts — stay symlinks to the shared template rather than being followed/duplicated) before overwriting `MOD_REPAIR_MODEL_NAME` with the repaired model, instead of creating a bare directory with just that one file. `resistivity_block_iter0.dat` is itself a `COPY_LIST` entry, so it already exists as a real file (not a symlink) in the fresh copy; REPAIR simply replaces its contents. Added a defensive `os.path.islink()` guard before the overwrite regardless, mirroring `femtic.py`'s own `insert_model` symlink-hazard guard. `_restart` directories produced by REPAIR are now ready to restart FEMTIC in directly. |

Author: Volker Rath (DIAS)
