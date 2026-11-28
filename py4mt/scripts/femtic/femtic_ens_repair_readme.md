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

3. **RESTART** (optional, `MOD_RESTART`) - for every run directory with a
   usable iteration, points `control.dat` at the **last completed
   iteration**, so FEMTIC continues from there instead of from iteration 0.
   Only `control.dat` is changed; the original is first saved as
   `control.orig`. See the RESTART section below.

This functionality was originally prototyped inside `femtic_ens_post.py`,
then split out into this standalone script once the design settled, so
`femtic_ens_post.py` stays focused on its own scope (summary statistics,
covariance, QC/statistics slice plots over converged members only).

---

## Workflow

```
Ensemble sub-directories  (rto_*, gst_*, member_*, ...)
          |
          v
  Scan ALL directories  (not just accepted)
          |
          +--> conv_list: label, dir, nrms, status
          |      status in {accepted, rejected_nrms,
          |                 missing_cnv, missing_model}
          |
          +-->  MOD_CONV = True
          |        -> nRMS bar chart / histogram, every scanned dir
          |
          +-->  MOD_REPAIR = True
          |        -> for each non-accepted dir:
          |            pick 2 random accepted members
          |            repaired = mean(log10(rho_1), log10(rho_2))
          |            copytree(<dir>, <dir>_restart, symlinks=True)
          |            overwrite <dir>_restart/resistivity_block_iter0.dat
          |            (original <dir> left untouched)
          |
          +-->  MOD_RESTART = True          (runs after REPAIR)
                   -> for each scanned dir (not *_restart):
                       N = last iteration with a complete model
                       control.dat (link or file) -> copy to control.orig
                       control.dat := same file, initial iteration = N
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
| `CNV_FILE` | str or list | `["femtic.cnv", "FEMTIC-DABIC.cnv"]` | Convergence file name in each member directory; a list gives candidate names (first existing wins). Directories without it are tagged `"missing_cnv"`. |
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

### RESTART

Continue every run where it stopped, by changing only its `control.dat`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `MOD_RESTART` | bool | `False` | Enable the RESTART step. Off by default, since it rewrites `control.dat` in every run directory. |
| `RESTART_FROM` | `"last"` / `"best"` | `"last"` | `"last"` = highest usable iteration (continue where the run stopped); `"best"` = the usable iteration with the lowest nRMS (ties: lowest iteration). |
| `RESTART_UNCONDITIONAL` | bool | `False` | `True` = take the iteration from the convergence file alone (its last iteration > 0, or with `RESTART_FROM = "best"` the lowest nRMS), **without** checking that the model files exist and without falling back to an earlier iteration. Missing files only give a warning in the report. See below. |
| `RESTART_CONTROL_FILE` | str | `"control.dat"` | Control file in each run directory. |
| `RESTART_ORIG_NAME` | str | `"control.orig"` | Name under which the original content is saved. |
| `RESTART_KEY` | str | `"ITERATION"` | Keyword of the control file that holds the initial iteration number. |
| `RESTART_VALUE_INDEX` | int | `0` | Which value token after `RESTART_KEY` is the initial iteration (`0` = the first, the `0` in `ITERATION 0 30`). |
| `RESTART_MAXITER_INDEX` | int / None | `1` | Which value token is the maximum iteration (the `30`); only used to warn when it is `<=` the start iteration, since such a run would not iterate. `None` = no check. |
| `RESTART_DRY_RUN` | bool | `False` | `True` = print and report what would be changed in every directory, write nothing. |

**Which iteration.** Only iterations `> 0` listed in the convergence file
count, and only those with a complete model: `resistivity_block_iter<N>.dat`,
plus `distortion_iter<N>.dat` if the directory has a `distortion_iter0.dat`.
If the last iteration of the convergence file lacks its files, the highest
iteration that has them is used, and the report says so. This is **not** the
`NRMS_MAX` / best-nRMS rule of the scan: a run that is not "accepted" is
restarted as well (that is usually the one that needs it).

**Unconditional (`RESTART_UNCONDITIONAL = True`).** The checks on the model files
are dropped: the iteration is the last one listed in the convergence file
(`"last"`), or the one with the lowest nRMS (`"best"`), whatever its nRMS and
whether or not its files exist. A directory is then never skipped or moved to an
earlier iteration because of missing files; if the files of the chosen iteration
are missing, the note says `WARNING unconditional: <file> not found` and
`control.dat` is written anyway (FEMTIC will fail on a missing
`resistivity_block_iter<N>.dat`, so look at those warnings before submitting).
Still left alone: directories without any iteration `> 0` in the convergence file
(there is nothing to restart from), directories whose control file is missing or
unusable (`FAILED`), and `*_restart` directories of REPAIR.

**What is written, per directory.**

1. `control.dat` is read (a symbolic link is followed). If `control.orig` does
   not exist yet, the **content** is copied to it. An existing `control.orig` is
   never overwritten, so a second RESTART (or a restart from a later iteration)
   keeps the true original.
2. `control.dat` is rewritten as a **regular file** that is identical except for
   the one value token (`RESTART_KEY`, value `RESTART_VALUE_INDEX`), set to the
   iteration found. It is written to a temporary file and renamed over
   `control.dat`; if `control.dat` was a link, the rename replaces the link only,
   so the **shared template is never written**. Line endings, comments, indentation
   and everything after the value are kept byte for byte; the file keeps its
   permissions plus owner-write (a read-only template gives a writable copy).
3. Nothing else in the directory is touched. If the value already equals the
   iteration, nothing is written (`unchanged`).

Control file rules (as in `femtic_lcurve_prep.py`): whitespace-separated tokens,
a value may follow its keyword on the same or a later line, a comment is a whole
line starting with `#`, reading stops at `END`, a keyword may appear once. A
directory whose control file is missing, a broken link, lacks the keyword, or has
a non-integer value is reported as `FAILED` and left untouched; the others carry on.

**Directories that are skipped:** no usable convergence rows or no iteration with a
complete model; and every directory whose name ends with `MOD_REPAIR_SUFFIX`
(`*_restart`, made by REPAIR), which is meant to start afresh from its new
iteration-0 model.

**Together with REPAIR.** RESTART runs after REPAIR, so a `_restart` copy made in the
same run still has the original `control.dat`. If REPAIR copies a directory that
went through RESTART in an earlier run, it puts the original control file back in
the copy (from `control.orig`, as a regular file) and does not carry
`control.orig` along.

**Check the keyword before the first real run.** The keyword and value position
(`ITERATION`, first value) are the defaults of this script; they could not be
checked against FEMTIC's own `control.dat` parser while it was written. Run once with
`RESTART_DRY_RUN = True` and compare the `old -> new` column with a real `control.dat`
of yours; change `RESTART_KEY` / `RESTART_VALUE_INDEX` if your FEMTIC version spells it
differently. To undo a RESTART, copy `control.orig` back to `control.dat`.

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
- `<dir>/control.orig` and the rewritten `<dir>/control.dat` - in every restarted run directory, when `MOD_RESTART=True` (and `RESTART_DRY_RUN=False`).
- A **"RESTART report"** section at the end of `femtic_ens_repair_summary.md`: one row per scanned directory (result, iteration, old -> new value, note). If the summary file cannot be appended to, it goes to `femtic_ens_repair_restart_report.md` in `ENSEMBLE_DIR`.

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

To continue the runs where they stopped instead, set `MOD_RESTART = True`
(first with `RESTART_DRY_RUN = True` to see the `old -> new` values). Every
run directory gets `control.orig` (the original `control.dat`) and a
`control.dat` that starts at its last completed iteration. Re-running is safe
here too: a directory already at that iteration is reported `unchanged`, and
`control.orig` is never overwritten.

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
| 2026-11-08 | Claude Sonnet 5.5 (Anthropic) | The scan-report helpers (`_scan_reason_text`, `_index_check`, building and appending the "Directory scan report") were copies of the ones in `femtic_ens_post.py`; both scripts now use the shared `femtic.build_scan_report` / `femtic.append_scan_report` (script shortened by about 200 lines). No change in the report or the settings. `femtic_ens_post.py` itself no longer draws figures (see `femtic_ens_plot_slices.py`, `femtic_ens_plot_bh.py`, `femtic_data_misfit.py`). |
| 2026-11-11 | Claude Sonnet 5.5 (Anthropic) | The member directories are searched for the convergence file with `fem.find_cnv_file()` (names of `CNV_FILE` tried in order, then ignoring case); the scan report is unchanged. Needs the `femtic.py` of the same date. AI-generated; review before production use. |
| 2026-11-24 | Claude Sonnet 5.5 (Anthropic) | **New optional step (4), `MOD_RESTART`**: points `control.dat` of every run directory at its last completed iteration, so FEMTIC continues there instead of from iteration 0. The content of the original `control.dat` (a link is followed) is copied to `control.orig`, which is never overwritten; `control.dat` is then replaced by a regular file in which only the initial-iteration value (`RESTART_KEY`, `RESTART_VALUE_INDEX`) is changed, so a shared template is never written. New settings `RESTART_FROM` (`"last"` / `"best"`), `RESTART_CONTROL_FILE`, `RESTART_ORIG_NAME`, `RESTART_KEY`, `RESTART_VALUE_INDEX`, `RESTART_MAXITER_INDEX`, `RESTART_DRY_RUN`. The iteration needs a `resistivity_block_iter<N>.dat` (and a `distortion_iter<N>.dat` when the directory has a `distortion_iter0.dat`); not the `NRMS_MAX` rule. Directories ending in `MOD_REPAIR_SUFFIX` are skipped; REPAIR restores the original control file in a `_restart` copy of a directory that was restarted before. A RESTART report is appended to `femtic_ens_repair_summary.md`. Tested on a synthetic 13-directory ensemble (link and regular `control.dat`, CRLF, read-only file and template, drifted last iteration, missing last block / distortion file, broken link, missing keyword, low maximum iteration, multi-line value, REPAIR before and after RESTART, repeated runs, dry run): the template and all other files are bit-identical afterwards, with RESTART off the script behaves as before. **Not tested against FEMTIC itself; the keyword `ITERATION` and the position of the initial-iteration value are the defaults of this script and were not checked against FEMTIC's `control.dat` parser** -- run once with `RESTART_DRY_RUN = True` first. AI-generated; review before production use. |
| 2026-11-24b | Claude Sonnet 5.5 (Anthropic) | New **`RESTART_UNCONDITIONAL`** (default `False`): the restart iteration is taken from the convergence file alone (last iteration > 0, or lowest nRMS with `RESTART_FROM = "best"`), with no check for `resistivity_block_iter<N>.dat` / `distortion_iter<N>.dat` and no fallback to an earlier iteration; missing files give a `WARNING unconditional` note and the directory is still written. `select_restart_iteration()` gets the argument `check_files`. Default behaviour unchanged: previous and new version give identical directory trees and console output on a synthetic 13-directory ensemble (REPAIR + RESTART on). Unconditional mode tested on the same ensemble (last iteration without block file, without distortion file, directory with no block file at all, dry run writes nothing, template and all other files bit-identical). AI-generated; review before production use. |
