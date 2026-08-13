#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_ens_repair.py — Convergence diagnostic and repair for a FEMTIC ensemble

Scans a FEMTIC ensemble directory tree (same ENSEMBLE_DIR/ENSEMBLE_NAME
convention as femtic_ens_post.py, femtic_rto_prep.py, femtic_gst_prep.py)
and reports on every scanned directory, not just converged ("accepted")
members:

1. **Convergence diagnostic** -- a bar chart or binned histogram of
   per-member nRMS (from femtic.cnv), colour-coded by status ("accepted",
   "rejected_nrms", "missing_cnv", "missing_model"), via
   fviz.plot_convergence_bar() / fviz.plot_convergence_histogram()
   (femtic_viz.py).

2. **REPAIR procedure** -- for every non-converged directory, copies the
   ENTIRE original directory to a **new sibling directory**
   (``<original>_restart`` by default; symlinks such as the LINK_LIST
   entries from femtic_rto_prep.py/femtic_gst_prep.py -- control.dat,
   mesh.dat, referencemodel.dat, distortion_iter0.dat, site.dat, run
   scripts -- are preserved as symlinks to their original shared
   targets, not followed/duplicated), then draws 2 distinct converged
   ("accepted") members at random, averages their log10(rho) models
   element-wise, and overwrites MOD_REPAIR_MODEL_NAME in the *copy*
   with that mean as a fresh starting model. The original non-converged
   directory is only ever read (via shutil.copytree) and is otherwise
   left completely untouched under its original name, so the failed
   run's own files (femtic.cnv, whatever model file it did produce,
   logs, etc.) remain available for inspection. Because the whole
   directory is copied first, the new ``_restart`` directory already
   has everything else FEMTIC needs to run there (mesh, control files,
   site data, run scripts, ...) -- it is a ready-to-run restart
   directory, not just a bare starting model.

This was originally prototyped as an addition to femtic_ens_post.py, then
split out into this standalone script once the design settled, so
femtic_ens_post.py can stay focused on its own scope (summary statistics,
covariance, QC/statistics slice plots over converged members only).

Author: Claude Sonnet 5 (Anthropic)
Created 2026-08-12, at the request of Volker Rath (DIAS).

Provenance:
    2026-08-12  Claude Sonnet 5 (Anthropic)
                Created. Split out of femtic_ens_post.py, where a
                convergence diagnostic and a REPAIR procedure had been
                prototyped across several iterations earlier the same
                day (see femtic_ens_post_readme.md changelog for that
                history). Step (1)'s directory scan reuses
                femtic_ens_post.py's scan loop verbatim, including its
                os.path.isdir(d) guard against non-directory glob
                matches. Step (2) (convergence diagnostic) defaults to
                the binned histogram (fviz.plot_convergence_histogram)
                with the aggregate "missing" bar turned OFF
                (MOD_CONV_SHOW_MISSING=False) at the user's request --
                unusable directories (missing femtic.cnv/model file)
                are still counted in the console scan log and the
                figure title, just not drawn as a bar. Step (3)
                (REPAIR) creates a NEW sibling "_restart" directory
                per non-converged member rather than renaming the
                original in place, at the user's explicit request, so
                the original failed-run directory is preserved
                unmodified under its original name; REPAIR is skipped
                entirely (with a console warning) if a target
                "_restart" directory already exists, or if fewer than
                MOD_REPAIR_MIN_MEMBERS converged members are available
                to draw a pair from. The repaired model is the
                element-wise mean of 2 distinct converged members' log10
                (rho) models, matching this codebase's existing
                log10-space averaging convention (RTO perturbations,
                ensemble mean/var/median/MAD in femtic_ens_post.py all
                operate in log10(rho) space).
    2026-08-12  Claude Sonnet 5 (Anthropic)
                REPAIR now shutil.copytree()'s the ENTIRE original
                directory to the "_restart" copy (symlinks=True, so
                LINK_LIST entries stay symlinks to the shared template
                rather than being followed/duplicated) before
                overwriting MOD_REPAIR_MODEL_NAME with the repaired
                model, instead of creating a bare directory containing
                only that one file. Matches the user-supplied GST
                COPY_LIST (observe.dat, resistivity_block_iter0.dat --
                real files) / LINK_LIST (control.dat, mesh.dat,
                referencemodel.dat, distortion_iter0.dat, site.dat, run
                scripts -- symlinks) convention: resistivity_block_
                iter0.dat is itself a COPY_LIST entry, so it already
                exists as a real, physically-copied file (not a
                symlink) in the fresh copy, and REPAIR simply replaces
                its contents. Added a defensive os.path.islink() guard
                before the overwrite anyway (unlink first if somehow a
                symlink), mirroring femtic.py's own insert_model
                symlink-hazard guard, in case a differently-configured
                ensemble ever puts the starting model in LINK_LIST
                instead. The "_restart" directory is consequently now a
                ready-to-run restart directory (mesh/control/site/run
                scripts all present via the copy), not just a bare
                starting model as in the first version of this script.
    2026-08-13  Claude Sonnet 5 (Anthropic)
                Added femtic_ens_repair_summary.md output at end of run:
                writes user-set (UPPERCASE) parameters, script path, and
                run date/time via utl.write_param_summary().
"""
from __future__ import annotations

import os
import sys
import shutil
from pathlib import Path

import numpy as np

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

for _base in [PY4MTX_ROOT + "/py4mt/modules/"]:
    for _p in [Path(_base), *Path(_base).rglob("*")]:
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

import femtic as fem
import util as utl
from version import versionstrg

try:
    import femtic_viz as fviz
except ImportError:
    fviz = None


version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng + "\n\n")

# ===========================================================================
# USER SECTION -- all user-set parameters below are UPPERCASE
# ===========================================================================
FEMTIC = "5.0"  # "4.3"

# ---------------------------------------------------------------------------
# Ensemble input
# ---------------------------------------------------------------------------
ENSEMBLE_DIR    = r"/media/vrath/LargeBack/Ensembles/misti_gst_ensembles/"
ENSEMBLE_NAME   = "misti_gst_suzuki_rnd"
ENSEMBLE_PREFIX = "misti_gst_suzuki_rnd"

#: Maximum normalised RMS accepted from femtic.cnv.
NRMS_MAX = 1.5

# ---------------------------------------------------------------------------
# Ocean / air handling (REPAIR only -- must match the values used by the
# FEMTIC inversion that produced the ensemble; passed straight through to
# fem.insert_model when writing each repaired starting model)
# ---------------------------------------------------------------------------
MOD_OCEAN     = None    # None = auto-infer; True/False forces ocean-present/absent
MOD_AIR_RHO   = 1.0e9   # Ω·m  (region 0)
MOD_OCEAN_RHO = 0.25    # Ω·m  (region 1 when treated as ocean)

# ---------------------------------------------------------------------------
# Convergence diagnostic — nRMS bar chart / histogram (all scanned dirs)
# ---------------------------------------------------------------------------
#: Set True to plot the nRMS convergence diagnostic. Covers *every*
#: directory scanned in step (1), including ones rejected for
#: nRMS > NRMS_MAX or missing femtic.cnv/model files, so it doubles as a
#: scan-run sanity check independent of MOD_REPAIR.
MOD_CONV      = True
#: Extension-less base path; one file written per MOD_PLOT_FORMAT entry.
MOD_CONV_FILE = ENSEMBLE_DIR + ENSEMBLE_PREFIX + "_convergence"

#: Set True for the one-bar-per-member chart (fviz.plot_convergence_bar)
#: instead of the default binned histogram
#: (fviz.plot_convergence_histogram) — reads better for small ensembles
#: where every directory name matters; ignores MOD_CONV_BINS/_NBINS_AUTO.
MOD_CONV_PER_MEMBER = False

# --- histogram mode (MOD_CONV_PER_MEMBER=False) -----------------------------
#: Number of equal-width bins spanning the *accepted* members' nRMS range
#: only (rejected/missing members never stretch the bin axis), or "auto"
#: to use MOD_CONV_NBINS_AUTO bins. Rejected members (nRMS > NRMS_MAX) are
#: always lumped into a single "rejected" bar rather than binned by value.
MOD_CONV_BINS       = "auto"
MOD_CONV_NBINS_AUTO = 15
#: Show the aggregate "missing" bar (members with no usable nRMS -- missing
#: femtic.cnv or missing model file). False omits it entirely; those
#: members are still counted in the printed scan log and the figure title,
#: just not drawn as a bar.
MOD_CONV_SHOW_MISSING = False

# --- per-member mode (MOD_CONV_PER_MEMBER=True) -----------------------------
#: Horizontal bars (one row per member) vs. vertical.
MOD_CONV_HORIZONTAL = True

#: Log-scale the count axis (histogram mode) or the nRMS axis (per-member
#: mode).
MOD_CONV_LOG = False

# ---------------------------------------------------------------------------
# REPAIR — seed a restart directory for non-converged members
# ---------------------------------------------------------------------------
#: Set True to run the REPAIR procedure (step (3), after the scan and the
#: convergence diagnostic). For every scanned directory that did NOT end
#: up "accepted" (status "rejected_nrms" / "missing_cnv" /
#: "missing_model"), REPAIR:
#:   1. copies the ENTIRE original directory to a new sibling directory
#:      (original name + MOD_REPAIR_SUFFIX) -- symlinks (e.g. the
#:      LINK_LIST entries from femtic_rto_prep.py/femtic_gst_prep.py:
#:      control.dat, mesh.dat, referencemodel.dat, distortion_iter0.dat,
#:      site.dat, run scripts) are preserved as symlinks to the same
#:      shared targets, not followed/duplicated;
#:   2. then overwrites MOD_REPAIR_MODEL_NAME in the *copy* with a fresh
#:      starting model: the element-wise mean (in log10(rho) space) of 2
#:      distinct converged ("accepted") members, chosen at random.
#: The original non-converged directory is left completely untouched
#: under its original name -- REPAIR never renames, deletes, or writes
#: into it; it only reads from it (via shutil.copytree) and writes into
#: the new "_restart" copy. Off by default since it writes new
#: directories/files to disk -- an explicit opt-in, not a side effect of
#: a plotting run.
MOD_REPAIR = False

#: Appended to the directory's basename to form the new sibling directory,
#: e.g. "rto_017" -> "rto_017_restart". The original "rto_017" directory
#: is left exactly as it was found.
MOD_REPAIR_SUFFIX = "_restart"

#: Filename inside the copied "_restart" directory that gets overwritten
#: with the repaired starting model. "iter0" matches FEMTIC's own
#: starting-model naming convention, and matches the "resistivity_block_
#: iter0.dat" entry in the GST/RTO prep scripts' own COPY_LIST -- i.e.
#: this file already exists (as a real, physically-copied file, not a
#: symlink) right after the directory copy, and REPAIR simply replaces
#: its contents with the new averaged starting model.
MOD_REPAIR_MODEL_NAME = "resistivity_block_iter0.dat"

#: Independent RNG seed for the random member-pair draws (same pattern as
#: BOOTSTRAP_SEED in femtic_ens_post.py): None = fresh OS entropy each run
#: (a different repaired model every time you re-run with
#: MOD_REPAIR=True); int = reproducible draws.
MOD_REPAIR_SEED = None

#: Minimum number of converged members required to draw a distinct pair
#: from. REPAIR is skipped entirely (with a warning) if fewer are
#: available -- there's nothing sensible to average.
MOD_REPAIR_MIN_MEMBERS = 2

# ---------------------------------------------------------------------------
# Plot output
# ---------------------------------------------------------------------------
#: One or more file extensions; see femtic_ens_post.py's MOD_PLOT_FORMAT
#: docstring for the full list of supported values and the raster/vector
#: distinction. A bare string ("pdf") also works.
MOD_PLOT_FORMAT = ["pdf", "jpg"]
_MOD_PLOT_FORMATS = (
    [MOD_PLOT_FORMAT] if isinstance(MOD_PLOT_FORMAT, str) else list(MOD_PLOT_FORMAT)
)

MOD_DPI            = 200   # figure DPI (raster formats only)
MOD_TICK_FONTSIZE  = 8
MOD_LABEL_FONTSIZE = 9

#: When True (default) and this script is running inside Spyder, every
#: saved figure is also displayed inline (Spyder's Plots pane) via
#: plt.show(), in addition to being written to disk. No effect outside
#: Spyder.
MOD_SHOW_IN_SPYDER = True

# ---------------------------------------------------------------------------
# Verbose output
# ---------------------------------------------------------------------------
OUT = True

_IN_SPYDER = (utl.runtime_env() == "spyder")
_SHOW_PLOTS = MOD_SHOW_IN_SPYDER and _IN_SPYDER
if _IN_SPYDER:
    print(f"Detected Spyder — inline figure display "
          f"{'enabled' if _SHOW_PLOTS else 'disabled (MOD_SHOW_IN_SPYDER=False)'}.\n")

# ===========================================================================
# Main
# ===========================================================================

# --- (1) Scan ensemble directories ----------------------------------------
# Identical to femtic_ens_post.py's step (1), plus per-directory
# bookkeeping (conv_list) needed for the convergence diagnostic and REPAIR
# below. os.path.isdir(d) is checked first since ENSEMBLE_NAME+"*" is a
# glob-style match that could in principle return a non-directory match
# (e.g. a stray file sharing the prefix) -- such matches are not ensemble
# run directories and are skipped immediately.
dir_list = utl.get_filelist(
    searchstr=[ENSEMBLE_NAME + "*"],
    searchpath=ENSEMBLE_DIR,
    fullpath=True,
)
print(f"Found {len(dir_list)} sub-directory/ies matching '{ENSEMBLE_NAME}'.")

model_list  = []          # list of [block_file, n_iter, nRMS] (accepted only)
model_count = 0
ens_matrix  = None        # will become (n_members, n_free) float64 (accepted only)

#: One entry per scanned *directory* (accepted *and* rejected/unusable).
#: dict(label, dir, nrms, status) with status one of "accepted",
#: "rejected_nrms", "missing_cnv", "missing_model".
conv_list = []

for d in dir_list:
    if not os.path.isdir(d):
        print(f"\n  {d}: not a directory — skipped (not an ensemble run).")
        continue

    print(f"\n  Inversion run: {d}")
    _label = os.path.basename(os.path.normpath(d))
    cnv_file = os.path.join(d, "femtic.cnv")
    if not os.path.isfile(cnv_file):
        print(f"    femtic.cnv not found — skipped.")
        conv_list.append(dict(label=_label, dir=d, nrms=None, status="missing_cnv"))
        continue

    with open(cnv_file) as _fh:
        cnv = _fh.readlines()
    info = cnv[-1].split()
    if "4.3" in FEMTIC:
        numit = int(info[0])
        nrms  = float(info[6])
    elif "5." in FEMTIC:
        numit = int(info[0])
        nrms  = float(info[8])
    else:
        sys.exit("FEMTIC version " + __file__ + ": does not exist! Exit.")

    if nrms > NRMS_MAX:
        print(f"    nRMS={nrms:.4f} > NRMS_MAX={NRMS_MAX} — skipped.")
        conv_list.append(dict(label=_label, dir=d, nrms=nrms, status="rejected_nrms"))
        continue

    mod_file = os.path.join(d, f"resistivity_block_iter{numit}.dat")
    if not os.path.isfile(mod_file):
        print(f"    {mod_file} not found — skipped.")
        conv_list.append(dict(label=_label, dir=d, nrms=nrms, status="missing_model"))
        continue

    print(f"    iter={numit}  nRMS={nrms:.4f}  {mod_file}")
    model_list.append([mod_file, numit, nrms])
    conv_list.append(dict(label=_label, dir=d, nrms=nrms, status="accepted"))

    log_m = fem.read_model(model_file=mod_file, model_trans="log10", out=OUT)

    if ens_matrix is None:
        ens_matrix = log_m[np.newaxis, :]           # (1, n_free)
    else:
        ens_matrix = np.vstack((ens_matrix, log_m))  # (k, n_free)

    model_count += 1

n_members = model_count
print(f"\nConverged members: {n_members}  /  "
      f"non-converged or unusable: {len(conv_list) - n_members}  /  "
      f"scanned total: {len(conv_list)}")

# --- (2) Convergence diagnostic — nRMS bar chart / histogram --------------
if MOD_CONV and fviz is not None and conv_list:
    for _fmt in _MOD_PLOT_FORMATS:
        _conv_fmt_file = f"{MOD_CONV_FILE}.{_fmt}"
        if MOD_CONV_PER_MEMBER:
            fviz.plot_convergence_bar(
                labels          = [c["label"] for c in conv_list],
                nrms            = [c["nrms"] for c in conv_list],
                status          = [c["status"] for c in conv_list],
                threshold       = NRMS_MAX,
                threshold_label = "NRMS_MAX",
                sort_by         = "nrms",
                horizontal      = MOD_CONV_HORIZONTAL,
                log_x           = MOD_CONV_LOG,
                tick_fontsize   = MOD_TICK_FONTSIZE,
                label_fontsize  = MOD_LABEL_FONTSIZE,
                plot_file       = _conv_fmt_file,
                dpi             = MOD_DPI,
                show            = _SHOW_PLOTS,
                out             = OUT,
            )
        else:
            if not any(c["status"] == "accepted" for c in conv_list):
                print("\n  convergence: no accepted members — "
                      "skipping histogram (nothing to bin; all scanned "
                      "dirs are rejected/missing/unusable).")
                continue
            fviz.plot_convergence_histogram(
                nrms             = [c["nrms"] for c in conv_list],
                status           = [c["status"] for c in conv_list],
                threshold        = NRMS_MAX,
                threshold_label  = "NRMS_MAX",
                bins             = MOD_CONV_BINS,
                n_bins_auto      = MOD_CONV_NBINS_AUTO,
                show_missing_bar = MOD_CONV_SHOW_MISSING,
                log_y            = MOD_CONV_LOG,
                tick_fontsize    = MOD_TICK_FONTSIZE,
                label_fontsize   = MOD_LABEL_FONTSIZE,
                plot_file        = _conv_fmt_file,
                dpi              = MOD_DPI,
                show             = _SHOW_PLOTS,
                out              = OUT,
            )
elif MOD_CONV and fviz is None:
    print("\nMOD_CONV=True but femtic_viz not available — skipping convergence plot.")

# --- (3) REPAIR — seed a restart directory for non-converged members ------
if MOD_REPAIR:
    _non_converged = [c for c in conv_list if c["status"] != "accepted"]
    if not _non_converged:
        print("\nREPAIR: no non-converged directories — nothing to repair.")
    elif n_members < MOD_REPAIR_MIN_MEMBERS:
        print(f"\nREPAIR: only {n_members} converged member(s) available "
              f"(need >= {MOD_REPAIR_MIN_MEMBERS}) — skipping repair.")
    else:
        _seed_txt = MOD_REPAIR_SEED if MOD_REPAIR_SEED is not None else "(fresh entropy)"
        print(f"\nREPAIR: {len(_non_converged)} non-converged directory/ies, "
              f"drawing from {n_members} converged member(s) (seed={_seed_txt}) …")
        _repair_rng = np.random.default_rng(MOD_REPAIR_SEED)
        # Template = lowest-nRMS converged member (preserves header / flag
        # columns), same convention as femtic_ens_post.py's MOD_QC/MOD_STATS.
        _best_repair_template = min(model_list, key=lambda x: x[2])[0]
        _n_repaired = 0
        for c in _non_converged:
            _src_dir = c["dir"]
            if _src_dir is None:
                print(f"  REPAIR: {c['label']}: no directory path recorded — skipped.")
                continue

            _dst_dir = _src_dir.rstrip(os.sep) + MOD_REPAIR_SUFFIX
            if os.path.exists(_dst_dir):
                print(f"  REPAIR: {os.path.basename(_dst_dir)} already exists — "
                      f"skipped (already repaired?). Original directory "
                      f"{c['label']} left untouched either way.")
                continue

            _i1, _i2 = _repair_rng.choice(n_members, size=2, replace=False)
            _mod1, _mod2 = model_list[_i1][0], model_list[_i2][0]
            _repaired_log10 = 0.5 * (ens_matrix[_i1] + ens_matrix[_i2])

            # Copy the entire original directory to the new "_restart"
            # sibling first -- symlinks=True preserves LINK_LIST entries
            # (mesh.dat, control.dat, etc.) as symlinks to their original
            # shared targets rather than following/duplicating them;
            # COPY_LIST entries (observe.dat, resistivity_block_iter0.dat,
            # ...) come along as real, independent files, exactly as they
            # were in the source. The source directory itself is only
            # ever read here, never modified.
            try:
                shutil.copytree(_src_dir, _dst_dir, symlinks=True)
            except (shutil.Error, OSError) as _e:
                print(f"  REPAIR: {c['label']}: copytree failed ({_e}) — skipped.")
                continue

            _out_model = os.path.join(_dst_dir, MOD_REPAIR_MODEL_NAME)
            # Defensive: MOD_REPAIR_MODEL_NAME is expected to be a real,
            # physically-copied file (COPY_LIST, not LINK_LIST) both in
            # the source and therefore in the fresh copy -- but if it
            # ever turns out to be a symlink (e.g. a differently-
            # configured ensemble), unlink it before writing so we never
            # write through a symlink back into a shared template file
            # (mirrors femtic.py's own insert_model symlink guard).
            if os.path.islink(_out_model):
                os.unlink(_out_model)
            fem.insert_model(
                template   = _best_repair_template,
                model      = _repaired_log10,
                model_file = _out_model,
                ocean      = MOD_OCEAN,
                air_rho    = MOD_AIR_RHO,
                ocean_rho  = MOD_OCEAN_RHO,
                out        = OUT,
            )
            _nrms_txt = f"{c['nrms']:.4f}" if c["nrms"] is not None else "n/a"
            print(f"  REPAIR: {c['label']} ({c['status']}, nRMS={_nrms_txt}) "
                  f"-- original left untouched; copied to "
                  f"{os.path.basename(_dst_dir)}/, then "
                  f"{MOD_REPAIR_MODEL_NAME} replaced with the mean of "
                  f"{os.path.basename(os.path.dirname(_mod1))} & "
                  f"{os.path.basename(os.path.dirname(_mod2))})")
            _n_repaired += 1
        print(f"REPAIR: {_n_repaired}/{len(_non_converged)} "
              f"non-converged directory/ies repaired.")

print("\nfemtic_ens_repair.py complete.")


# ---------------------------------------------------------------------------
# Parameter summary
# ---------------------------------------------------------------------------
utl.write_param_summary(__file__)
