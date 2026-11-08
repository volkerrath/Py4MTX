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
| `ENSEMBLE_INDEX_RANGE` | tuple or None | `None` | Expected member numbers `(first, last)`, inclusive, e.g. `(0, 255)` for 256 members named `<ENSEMBLE_NAME>0` ... `<ENSEMBLE_NAME>255`. Used only by the "Directory numbering" part of the scan report in the parameter summary. `None` = the range found in the names (cannot show a missing first or last member). |
| `CNV_FILE` | str or list | `"femtic.cnv"` | Convergence file name in each member directory; a list gives candidate names (first existing wins). Directories without it are tagged `"missing_cnv"`. |
| `NRMS_MAX` | float | `1.5` | Members whose **best** nRMS (smallest over all iterations > 0, not the last iteration) exceeds this value are tagged `"rejected_nrms"`. |
| `ITER_FALLBACK` | bool | `True` | If the best-nRMS iteration `N` has no `resistivity_block_iter<N>.dat`, use the best iteration that does have one (usually `N-1`); `NRMS_MAX` is then applied to that iteration's nRMS. `False` = tag the member `"missing_model"` instead. Same meaning as in `femtic_ens_post.py` -- keep the two equal. |

nRMS and the iteration number are read from each member's `femtic.cnv` via `fem.read_cnv()`, which parses column positions from the file's own header row (case-insensitive substring match) — no FEMTIC-version setting needed.

**Member selection** (`select_member_iteration()`) follows `femtic_ens_post.py` exactly, so both scripts accept the same members:

1. Candidates are all `femtic.cnv` rows with `Iter > 0`; the one with the smallest nRMS wins (ties: lowest iteration). FEMTIC does not guarantee a monotonic nRMS decrease, so the last row is not necessarily the best.
2. Best nRMS > `NRMS_MAX` -> `"rejected_nrms"` (the nRMS shown in the diagnostic is that best value).
3. Block file of that iteration missing and `ITER_FALLBACK=True` -> fall back as described above; the member still counts as `"accepted"`.
4. No usable block file -> `"missing_model"`. No iteration > 0 in the cnv file -> `"missing_cnv"`.

The one difference left: `femtic_ens_post.py` with `ENS_MODEL_SOURCE = "perturbed"` or `"both"` additionally skips members lacking `ENS_PERTURBED_FILE` (`referencemodel.dat`). That is not a convergence criterion, so this script does not check it; it can therefore count a few more converged members than `femtic_ens_post.py`.

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
   convention as the slice settings of `femtic_ens_plot_slices.py`, formerly the `MOD_QC`/`MOD_STATS` blocks of `femtic_ens_post.py`). A
   defensive check unlinks the target first if it's unexpectedly a
   symlink, so the write can never land on a shared template file.
7. `<dir>` itself is only ever *read* (by `shutil.copytree`) — never
   modified, renamed, or deleted.

Because the whole directory is copied first, `<dir>_restart` ends up
containing everything `<dir>` had — `femtic.cnv`, whatever partial
results existed, logs, plus all the `LINK_LIST`/`COPY_LIST` run
machinery — with only the starting model swapped out. It is a
ready-to-run restart directory, not just a bare model file.

### Directory scan report (in the summary file)

The section appended to `femtic_ens_repair_summary.md` accounts for every
scanned directory (scanned = accepted + not accepted; a warning line appears if
that does not add up) and gives a **reason** and a **detail** text for each one
that is not in the ensemble:

| Reason | Status in the convergence plot | Meaning |
|---|---|---|
| `no_cnv_file` | `missing_cnv` | none of the `CNV_FILE` names exists in the directory |
| `cnv_unreadable` | `missing_cnv` | the convergence file has no usable header row |
| `cnv_no_iterations` | `missing_cnv` | the convergence file has no data rows or no iteration > 0 |
| `rejected_nrms` | `rejected_nrms` | best nRMS (iterations > 0) above `NRMS_MAX` |
| `no_model_file` | `missing_model` | no `resistivity_block_iter<N>.dat` for the best iteration and no usable fallback (the detail names the iteration and why no fallback was possible) |

Further tables list the counts per reason, the members accepted via
`ITER_FALLBACK` (best iteration -> iteration used) and glob matches that are not
directories. This also explains differences to `femtic_ens_post.py`, whose own
section additionally has `no_perturbed_file` (a member that converged but lacks
`ENS_PERTURBED_FILE`; not a convergence criterion, so not checked here). If the
summary file cannot be appended to, the report goes to
`femtic_ens_repair_scan_report.md` instead. Plain ASCII only.

**Directory numbering.** The report always gives the number of glob matches next
to the number of directories scanned (they differ when a match is a file or a
broken symbolic link). It then reads the number that follows `ENSEMBLE_NAME` in
every matching name and compares it with the expected range
(`ENSEMBLE_INDEX_RANGE`, default: the range found in the names). It lists, if
present:

- numbers with **no entry at all** (nothing starts with `<ENSEMBLE_NAME><number>`: never created, deleted, renamed, or elsewhere) -- this is how a "255 scanned, 256 expected" difference is traced to a number;
- numbers matched **only by a non-directory** (file, broken symbolic link);
- numbers matched by **more than one name** (e.g. `X` and `X_restart`);
- names with a number outside the expected range, and names without a number.

Zero-padded numbers are fine (`007` = 7). With the default (inferred) range a
missing first or last member cannot be seen; set `ENSEMBLE_INDEX_RANGE = (0, 255)`
for a known ensemble size.

### Plot output

| Parameter | Type | Default | Description |
|---|---|---|---|
| `MOD_PLOT_FORMAT` | str / list of str | `["pdf", "jpg"]` | One or more save formats; see `femtic_ens_plot_slices.py`'s `MOD_PLOT_FORMAT` comment for the full supported list. |
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
- `femtic_ens_repair_summary.md` (in `ENSEMBLE_DIR`) - the UPPERCASE settings, script path and run date/time, followed by a **"Directory scan report"** section: why each scanned directory that is not in the ensemble was left out (see below).
- `<dir>_restart/` — one complete copy of `<dir>` per repaired non-converged directory, when `MOD_REPAIR=True`, with `MOD_REPAIR_MODEL_NAME` overwritten by the repaired starting model.

This script does **not** write a `.npz` results file — that remains
`femtic_ens_post.py`'s job, over converged members only.

---

## Quick start

```python
# Edit ENSEMBLE_DIR, ENSEMBLE_NAME, ENSEMBLE_PREFIX, NRMS_MAX
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
| 2026-08-13 | Claude Sonnet 5 (Anthropic) | Added `femtic_ens_repair_summary.md` output at end of run: writes user-set (UPPERCASE) parameters, script path, and run date/time. |
| 2026-09-07 | Claude Sonnet 5 (Anthropic) | Removed the `FEMTIC` config variable and its `"4.3"`/`"5."` version-string switch on the nRMS column index (6 vs 8): that switch only matched one specific column layout, and silently misread nRMS (e.g. reading the Distortion or Misfit column instead) for any run whose actual column count didn't match the assumed FEMTIC-version pairing. Now uses `fem.read_cnv()`, matching `femtic_ens_post.py`'s `get_nrms()`-based fix of 2026-09-02 — column positions are read from `femtic.cnv`'s own header row every time, independent of version or Beta/Distortion presence. |
| 2026-10-08 | Claude Sonnet 5.5 (Anthropic) | New config `CNV_FILE` (default `"femtic.cnv"`, single name or candidate list) replaces the hard-coded `femtic.cnv` in step (1)'s scan loop; uses `fem.resolve_cnv_path()`. AI-generated; review before production use. |
| 2026-10-28 | Claude Sonnet 5.5 (Anthropic) | The parameter summary `femtic_ens_repair_summary.md` is now written to `ENSEMBLE_DIR` (the working directory set in the script) instead of next to the script: `utl.write_param_summary(fname, out_dir=ENSEMBLE_DIR)`. Needs the `util.py` of the same date. AI-generated; review before production use. |
| 2026-11-06 | Claude Sonnet 5.5 (Anthropic) | **Fix: far too many members were reported as non-converged** (78 of 256 in one ensemble where `femtic_ens_post.py` correctly accepted 253 of 256). Step (1) judged each member by the **last** row of its `.cnv` file; `femtic_ens_post.py` uses the **best-nRMS iteration (iter > 0)** and falls back to the best iteration that has a block file when the best one has none (`ITER_FALLBACK`). A run whose last iteration drifted above `NRMS_MAX`, or whose last iteration wrote no block file, was wrongly tagged `"rejected_nrms"` / `"missing_model"` and, with `MOD_REPAIR=True`, would have received a spurious `_restart` copy. New function `select_member_iteration()` implements the `femtic_ens_post.py` rule; new config `ITER_FALLBACK` (default `True`). A member accepted via the fallback keeps status `"accepted"`; for `"rejected_nrms"` members the diagnostic now shows the best nRMS, not the last. Tested on a synthetic 21-directory ensemble (last-iteration drift, missing last block file, truly rejected, no cnv, no block files, fallback above `NRMS_MAX`, alternative cnv name, stray file): old logic 11 accepted, new logic 17, identical to a verbatim port of `femtic_ens_post.py`'s selection for every member; REPAIR created exactly the 4 expected `_restart` directories and left the originals bit-identical. Not tested on the real ensemble. AI-generated; review before production use. |
| 2026-11-06b | Claude Sonnet 5.5 (Anthropic) | **Directory scan report** appended to `femtic_ens_repair_summary.md`: reason (`no_cnv_file`, `cnv_unreadable`, `cnv_no_iterations`, `rejected_nrms`, `no_model_file`) and detail for every scanned directory that is not in the ensemble, counts per reason, members accepted via `ITER_FALLBACK`, ignored non-directory glob matches. The `conv_list` entries gain `reason` / `detail` keys (the plot functions only receive the label / nRMS / status lists, so they are unaffected). Uses the path returned by `utl.write_param_summary()`; falls back to `femtic_ens_repair_scan_report.md`. Same helpers as in `femtic_ens_post.py`. Tested on a synthetic 25-directory ensemble with the real `write_param_summary`. AI-generated; review before production use. |
| 2026-11-06c | Claude Sonnet 5.5 (Anthropic) | Scan report: the number of glob matches is always shown next to the directories scanned (they differ for a file or broken symbolic link), and a new **"Directory numbering"** part reads the number after `ENSEMBLE_NAME` in each name and lists numbers with no entry at all, numbers matched only by a non-directory, numbers matched by several names (e.g. `X` and `X_restart`) and unnumbered names (prompted by a report of 255 directories scanned for an ensemble numbered 0..255). New optional `ENSEMBLE_INDEX_RANGE = (first, last)`, default `None` = inferred from the names. Same helpers as in `femtic_ens_post.py`. Tested on a synthetic ensemble with a missing number, a broken symbolic link, an `_restart` duplicate and a stray file. AI-generated; review before production use. |

Author: Volker Rath (DIAS)
| 2026-11-08 | Claude Sonnet 5.5 (Anthropic) | The scan-report helpers (`_scan_reason_text`, `_index_check`, building and appending the "Directory scan report") were copies of the ones in `femtic_ens_post.py`; both scripts now use the shared `femtic.build_scan_report` / `femtic.append_scan_report` (script shortened by about 200 lines). No change in the report or the settings. `femtic_ens_post.py` itself no longer draws figures (see `femtic_ens_plot_slices.py`, `femtic_ens_plot_bh.py`, `femtic_ens_plot_misfit.py`). |
