#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean and optionally archive a FEMTIC run directory.

Author: Volker Rath (DIAS)
Created with the help of ChatGPT (GPT-5 Thinking) on 2026-04-07

Provenance:
    2026-04-07  vrath / ChatGPT (GPT-5 Thinking)  Created.
    2026-08-13  Claude Sonnet 5 (Anthropic)
                Added femtic_archive_run_summary.md output after CLI
                argument parsing: writes the resolved argparse parameters,
                script path, and run date/time via a self-contained
                _write_param_summary() helper (stdlib only, no external
                py4mt dependency).
    2026-09-11  Claude Sonnet 5 (Anthropic)
                HDF5 support: removed the previous blanket ".h5" suffix
                protection (which silently protected ALL .h5 files,
                including iterX.h5) and replaced it with an explicit
                protected_filenames set (mesh.h5, rough.h5, jac.h5,
                exact-name, case-insensitive). iterX.h5 files are now
                matched by the iteration regex exactly like iterX.dat
                files. Also fixed iteration selection to be grouped per
                containing directory (instead of pooled globally across
                all recursively found files), so the script gives
                correct results when run with --recursive from above an
                ensemble of run subdirectories that do not all share the
                same iteration count.
    2026-09-11  Claude Sonnet 5 (Anthropic)  (same-day follow-up)
                Added --exclude-dirs (default: plots) to prune whole
                subdirectories from the --recursive walk entirely (not
                scanned, not deleted, not archived), and
                --always-include-dirs (default: templates, python) to
                treat whole subdirectories as always protected and
                always fully included in the compressed archive,
                regardless of iteration matching. Added a printed
                warning when --compress targets a .zip file: Python's
                zipfile dereferences symlinks (duplicating target
                content) and raises an error outright on a broken
                symlink, whereas .tgz/.tar.gz (tarfile) preserves
                symlinks -- including broken ones -- as real symlink
                entries. Recommended in the warning and in the README.
    2026-09-11  Claude Sonnet 5 (Anthropic)  (second same-day follow-up)
                Corrected always_include_dirs semantics: these
                directories (default: templates, python) are now
                pruned from the per-file scan exactly like
                exclude_dirs -- never walked for iteration matching or
                protection checks -- but are located as whole
                directories and added to the archive untouched via a
                recursive directory add, instead of being scanned
                file-by-file and marked individually protected.
    2026-09-15  Claude Sonnet 5 (Anthropic)
                Confirmed/kept the existing extension-agnostic iteration
                matching (pattern is matched against the filename only,
                so model_iterN.dat, model_iterN.h5, and any other
                extension are all treated identically as iteration
                files -- no code change was needed here, this was
                already the behavior). Updated the default
                protected_filenames set from ("mesh.h5", "rough.h5",
                "jac.h5") to ("mesh.h5", "jacobian.h5", "rough.h5") per
                request -- jac.h5 is no longer protected by default; if
                your runs still use that name, pass it explicitly via
                protected_filenames (no CLI flag exposes this yet).
                Added an explicit "Files that will be kept" listing,
                printed during a dry run (i.e. whenever --delete is not
                given) regardless of whether --compress is also given,
                so a plain dry run now shows the full kept-file list
                and not just the per-directory "Kept iterations" line.
    2026-09-15  Claude Sonnet 5 (Anthropic)  (same-day follow-up)
                Added support for running directly on top of several
                individual ensemble run directories via shell wildcards
                (e.g. ``ensemble/run_*``), instead of only via
                --recursive from a shared parent. The ``directory``
                positional argument now accepts one or more paths
                and/or glob patterns; patterns are expanded internally
                with the stdlib ``glob`` module too, so a quoted
                pattern (e.g. ``"ensemble/run_*"``, needed if you don't
                want your shell to expand it first, or on Windows)
                works the same as shell-expanded arguments. Each
                matched directory keeps its own independent
                keep-lowest/keep-highest iteration selection (same
                per-containing-directory grouping already used by
                --recursive), and the archive's relative paths are
                computed from the matched directories' common parent,
                so an archive built from ``run_001 run_002`` looks the
                same as one built with --recursive over their shared
                parent.
    2026-09-15  Claude Sonnet 5 (Anthropic)  (second same-day follow-up)
                Decoupled --compress from --delete: previously,
                --compress without --delete only printed a "WOULD ADD"
                preview and never wrote an actual archive file, so a
                real, on-disk .tgz/.zip required also deleting the
                source files via --delete. --compress now always
                writes a real archive containing the kept-file
                selection, regardless of --delete -- so --compress
                alone gives a genuine, already-smaller archive while
                leaving every source file untouched. --delete
                continues to control only whether the unwanted
                iteration files are removed from the source
                directories, entirely independent of archiving.
    2026-09-15  Claude Sonnet 5 (Anthropic)  (third same-day follow-up)
                always_include_dirs (default: templates, python) are
                now also located as immediate children of the
                archive's base directory -- the common parent when
                several directories/wildcards are given, or the
                directory itself otherwise -- via a single
                non-recursive listing that runs unconditionally, not
                just when --recursive is set. This matches the usual
                ensemble layout (run_1, run_2, ..., templates/,
                python/ all as siblings under one ensemble folder):
                targeting the run directories directly, e.g. via
                ``ensemble_x/run_*``, now still picks up
                ``ensemble_x/templates`` and ``ensemble_x/python`` and
                adds them to the archive untouched, with no
                --recursive needed. --recursive is still required to
                find always_include_dirs at deeper nesting, or under
                more than one distinct parent. Deduplicated so this
                doesn't double-report/double-add a directory already
                found via a recursive walk.
    2026-09-15  Claude Sonnet 5 (Anthropic)  (fourth same-day follow-up)
                keep_n_low may now be 0 ("keep none of the matched
                iteration files from the low end"), rather than
                rejecting any value below 1 outright -- e.g.
                keep_n_low=0, keep_n_high=1 keeps only the single
                highest iteration and discards everything else matched
                (iter0 remains kept regardless, via its separate
                iter0-token protection). keep_n_high still must be >=
                1 -- there is always at least the highest iteration
                worth keeping -- so this is an asymmetric relaxation,
                not a blanket one. Also fixed a latent bug this change
                would otherwise have exposed in
                _select_kept_iterations: Python's list[-0:] evaluates
                to list[0:] (the whole list, since -0 == 0), not an
                empty slice, so keep_n_high == 0 could never have been
                sliced directly to mean "keep none from the high end"
                -- now handled with an explicit guard instead (kept
                even though keep_n_high == 0 no longer reaches this
                function via the CLI/mt_archive_run validation, as a
                defensive correctness fix).

    NOTE: This script was produced with AI assistance (see provenance
    log above). It has not been independently verified for production
    use -- review the logic (especially file selection/deletion) before
    running with --delete on data you cannot regenerate.
"""

from __future__ import annotations

import argparse
import glob as glob_module
import re
import tarfile
import zipfile
from pathlib import Path
import os
import sys

# PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

# for _base in [PY4MTX_ROOT + "/py4mt/modules/"]:
    #for _p in [Path(_base), *Path(_base).rglob("*")]:
        #if _p.is_dir() and str(_p) not in sys.path:
            #sys.path.insert(0, str(_p))

from typing import Iterable


def _write_param_summary(fname: str, params: dict) -> str:
    """Write a Markdown summary of this script's CLI parameters.

    Self-contained (stdlib only) so this script keeps no dependency on
    the py4mt module stack. Writes ``<script_dir>/<script_stem>_summary.md``
    with the script path, run date/time, and the resolved CLI arguments.
    """
    from datetime import datetime

    script_path = os.path.abspath(fname)
    stem = os.path.splitext(os.path.basename(fname))[0]
    out_path = os.path.join(os.path.dirname(script_path), stem + "_summary.md")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Parameter summary: " + stem, "",
        "- **Script path:** `" + script_path + "`",
        "- **Run date/time:** " + now, "",
        "## User-set parameters", "",
        "| Parameter | Value |", "|---|---|",
    ]
    for k in sorted(params):
        vstr = repr(params[k]).replace("|", "\\|").replace("\n", " ")
        lines.append("| `" + k + "` | `" + vstr + "` |")

    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("[write_param_summary] Wrote parameter summary to " + out_path)
    return out_path


def _normalize_protected_tokens(protected_tokens: Iterable[str]) -> tuple[str, ...]:
    return tuple(str(token).lower() for token in protected_tokens)


def _normalize_protected_suffixes(
    protected_suffixes: Iterable[str],
) -> tuple[str, ...]:
    return tuple(str(suffix).lower() for suffix in protected_suffixes)


def _normalize_protected_filenames(
    protected_filenames: Iterable[str],
) -> tuple[str, ...]:
    return tuple(str(name).lower() for name in protected_filenames)


def _expand_directory_args(patterns: Iterable[str]) -> list[Path]:
    """Turn CLI directory arguments into a de-duplicated list of directory
    Paths, expanding any glob wildcards (``*``, ``?``, ``[...]``) that
    survived shell expansion -- e.g. because the pattern was quoted, or
    because the shell in use (e.g. on Windows) doesn't expand globs
    itself. Plain literal paths (no wildcard, already a real directory)
    pass through unchanged. Non-directory matches (stray files caught by
    a loose pattern) and patterns that match nothing are reported and
    skipped rather than aborting the whole run.
    """
    resolved: list[Path] = []
    for pat in patterns:
        matches = sorted(glob_module.glob(pat))
        if matches:
            for m in matches:
                p = Path(m)
                if p.is_dir():
                    resolved.append(p)
                else:
                    print(f"Skipping non-directory match for {pat!r}: {p}")
        else:
            p = Path(pat)
            if p.is_dir():
                resolved.append(p)
            else:
                print(f"WARNING: no directory matches {pat!r} -- skipping")

    seen: set[Path] = set()
    unique: list[Path] = []
    for p in resolved:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def _common_base_dir(directories: list[Path]) -> Path:
    """Pick the base directory that archive paths are computed relative
    to. For a single directory this is that directory itself (unchanged
    behavior). For several directories (e.g. matched by a wildcard
    pattern over multiple ensemble runs), it's their common parent, so
    the archive nests each run under its own name exactly as a
    --recursive run over that shared parent would.
    """
    if len(directories) == 1:
        return directories[0]
    try:
        return Path(os.path.commonpath([str(d) for d in directories]))
    except ValueError:
        # e.g. paths on different drives on Windows -- fall back to cwd
        return Path(".")


def _collect_files(
    directory: Path, recursive: bool, prune_dirs: tuple[str, ...] = ()
) -> list[Path]:
    """List files for iteration/protection scanning. ``prune_dirs`` names
    (case-insensitive) are never descended into -- used both for
    exclude_dirs (dropped entirely) and always_include_dirs (handled
    wholesale elsewhere, so also kept out of this per-file scan)."""
    if not recursive:
        files = [p for p in directory.glob("*") if p.is_file()]
        return sorted(files)

    files: list[Path] = []
    for root, dirnames, filenames in os.walk(directory):
        dirnames[:] = [d for d in dirnames if d.lower() not in prune_dirs]
        for fname in filenames:
            files.append(Path(root) / fname)
    return sorted(files)


def _find_named_dirs(
    directory: Path, dir_names: tuple[str, ...], exclude_dirs: tuple[str, ...]
) -> list[Path]:
    """Locate subdirectories (at any depth) whose name (case-insensitive)
    is in dir_names, without descending further into a match once found
    (its contents are handled wholesale, not scanned file-by-file), and
    without descending into anything in exclude_dirs either."""
    found: list[Path] = []
    for root, dirnames, _filenames in os.walk(directory):
        dirnames[:] = [d for d in dirnames if d.lower() not in exclude_dirs]
        matched = [d for d in dirnames if d.lower() in dir_names]
        for d in matched:
            found.append(Path(root) / d)
        # Don't walk into a matched dir looking for nested matches or
        # iteration files -- its contents are archived untouched.
        dirnames[:] = [d for d in dirnames if d.lower() not in dir_names]
    return sorted(found)


def _is_protected(
    filepath: Path,
    protected_tokens: tuple[str, ...],
    protected_suffixes: tuple[str, ...],
    protected_filenames: tuple[str, ...] = (),
) -> bool:
    name_l = filepath.name.lower()
    if name_l in protected_filenames:
        return True
    if any(name_l.endswith(s) for s in protected_suffixes):
        return True
    if any(t in name_l for t in protected_tokens):
        return True
    return False


def _extract_iter_value(filepath: Path, regex: re.Pattern[str]) -> int | None:
    m = regex.search(filepath.name)
    return int(m.group(1)) if m else None


def _select_kept_iterations(
    iter_values: Iterable[int],
    keep_n_low: int,
    keep_n_high: int,
) -> set[int]:
    """Pick which iteration numbers survive the keep-lowest/keep-highest
    rule. ``keep_n_low`` may be 0 ("keep none from the low end");
    ``mt_archive_run`` enforces ``keep_n_high >= 1`` before this is
    called, so the ``keep_n_high == 0`` case shouldn't normally arise
    here -- but it's still handled correctly (as empty) rather than
    silently wrong, since Python's ``list[-0:]`` is ``list[0:]`` (the
    *whole* list, as ``-0 == 0``), not an empty slice, if sliced
    directly without this guard.
    """
    unique = sorted(set(iter_values))
    low_part = unique[:keep_n_low] if keep_n_low > 0 else []
    high_part = unique[-keep_n_high:] if keep_n_high > 0 else []
    return set(low_part + high_part)


def _print_kept_files(
    final_keep: Iterable[Path], always_include_paths: Iterable[Path] = ()
) -> None:
    """Print the full list of files that will be kept (not deleted).

    Called during a dry run so the user can see exactly what survives
    the keep-lowest/keep-highest + protection rules, independent of
    whether --compress is also given.
    """
    print("\nFiles that will be kept (dry-run, nothing deleted):")
    for f in sorted(set(final_keep)):
        print(f"  KEEP {f}")
    for d in sorted(set(always_include_paths)):
        print(f"  KEEP (always-included, whole directory) {d}/")


def _arcname(filepath: Path, base_dir: Path, include_root: bool) -> Path:
    rel = filepath.relative_to(base_dir)
    return (base_dir.name / rel) if include_root else rel


def _write_zip(
    archive_path: Path, files, base_dir: Path, include_root: bool,
    extra_dirs: Iterable[Path] = (),
):
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            try:
                zf.write(f, arcname=_arcname(f, base_dir, include_root))
            except OSError as e:
                # Most commonly a broken/dangling symlink -- zipfile has
                # no way to store it as a link, so it tries to follow it
                # and fails. Skip it rather than aborting the archive.
                print(f"  SKIPPED (zip cannot store this, e.g. broken symlink): {f}: {e}")
        for d in extra_dirs:
            for f in sorted(p for p in d.rglob("*") if p.is_file()):
                try:
                    zf.write(f, arcname=_arcname(f, base_dir, include_root))
                except OSError as e:
                    print(f"  SKIPPED (zip cannot store this, e.g. broken symlink): {f}: {e}")


def _write_tgz(
    archive_path: Path, files, base_dir: Path, include_root: bool,
    extra_dirs: Iterable[Path] = (),
):
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tf:
        for f in files:
            tf.add(f, arcname=_arcname(f, base_dir, include_root))
        for d in extra_dirs:
            # recursive=True adds the whole tree untouched, preserving
            # symlinks (tarfile does not dereference by default).
            tf.add(d, arcname=_arcname(d, base_dir, include_root), recursive=True)


def _compress(
    compress_path, files, base_dir: Path, include_root: bool,
    extra_dirs: Iterable[Path] = (), deleted_source: bool = False,
):
    """Write the compressed archive from the kept-file selection.

    This always writes a real archive when ``compress_path`` is given --
    independent of whether the caller also deleted the unwanted files
    from the source directories (``deleted_source``, i.e. ``--delete``
    was passed). The archive only ever contains the kept-file
    selection either way, so --compress alone (no --delete) gives a
    real, smaller archive while leaving every source file untouched.
    """
    if compress_path is None:
        return

    archive_path = Path(compress_path)
    name = archive_path.name.lower()
    files = sorted(set(files))
    extra_dirs = sorted(set(extra_dirs))

    print(f"\nCompression target: {archive_path}")

    if name.endswith(".zip"):
        print(
            "WARNING: .zip does not preserve symlinks. Python's zipfile "
            "dereferences symlinks (each linked file is stored as a full "
            "duplicate copy of its target's content, not as a link) and "
            "raises an error outright on a broken/dangling symlink. If "
            "this run directory contains symlinks you want kept as links "
            "(and not just followed or failing on broken ones), use "
            "--compress with a .tgz or .tar.gz path instead -- tarfile "
            "preserves symlinks, including broken ones, as real symlink "
            "entries."
        )

    if name.endswith(".zip"):
        _write_zip(archive_path, files, base_dir, include_root, extra_dirs)
    elif name.endswith(".tgz") or name.endswith(".tar.gz"):
        _write_tgz(archive_path, files, base_dir, include_root, extra_dirs)
    else:
        raise ValueError("Unsupported archive format")

    if deleted_source:
        print(f"Archive written: {archive_path}")
    else:
        print(
            f"Archive written: {archive_path} "
            "(source files left untouched -- pass --delete to also "
            "remove the unwanted iterations from disk)"
        )


def mt_archive_run(
    directory: str | Path | Iterable[str | Path],
    pattern: str = r"_iter(\d+)",
    protected_tokens: Iterable[str] = ("obs", "ref", "mesh", "iter0", "control", "rough"),
    protected_suffixes: Iterable[str] = (".sh", ".cnv"),
    protected_filenames: Iterable[str] = ("mesh.h5", "jacobian.h5", "rough.h5"),
    exclude_dirs: Iterable[str] = ("plots",),
    always_include_dirs: Iterable[str] = ("templates", "python"),
    keep_n_low: int = 1,
    keep_n_high: int = 1,
    recursive: bool = False,
    dry_run: bool = True,
    compress_path: str | Path | None = None,
    include_root: bool = True,
) -> None:
    """Clean (and optionally archive) one or more FEMTIC run directories.

    ``directory`` accepts a single path, or an iterable of paths --
    typically several individual ensemble run directories matched by a
    shell wildcard (e.g. ``run_*``) and passed straight through by
    argparse. Each directory keeps its own independent
    keep-lowest/keep-highest selection (see below), the same as
    --recursive over their shared parent would give; this is the
    lighter-weight alternative when you already know which run
    directories you want and don't need a full recursive walk to find
    them.

    Iteration files are matched by ``pattern`` regardless of extension,
    so e.g. ``model_iter12.dat`` and ``model_iter12.h5`` are both treated
    as iteration files unless they also match a protected token, suffix,
    or exact filename.

    ``keep_n_low`` may be 0, meaning "keep none of the matched
    iteration files from the low end" -- e.g. ``keep_n_low=0,
    keep_n_high=1`` keeps only the single highest iteration and
    discards every other matched iteration file. ``keep_n_high`` must
    always be >= 1 (there is always at least one iteration -- the
    highest -- worth keeping). This does NOT affect iteration 0's
    separate protection via the ``iter0`` protected token (default
    ``protected_tokens``): an ``iter0`` file is kept regardless of
    ``keep_n_low``/``keep_n_high``, since it's matched as a protected
    file before iteration selection ever runs.

    When ``recursive`` is True (e.g. run from a directory that sits
    above several FEMTIC run subdirectories -- an "ensemble" layout),
    the keep-lowest/keep-highest iteration selection is applied
    separately per containing directory, not pooled across the whole
    tree. This matters because different runs in an ensemble need not
    share the same iteration count: pooling iteration numbers globally
    would apply one run's iteration range to all the others.

    ``exclude_dirs`` names subdirectories (matched case-insensitively,
    at any depth under ``directory``) that are skipped entirely during
    a recursive walk: not scanned, not deleted, not archived -- as if
    they did not exist. Use this for things like plot output that has
    nothing to do with the inversion state and need not be preserved.

    ``always_include_dirs`` names subdirectories (matched the same way)
    that are never scanned for iteration files or protection checks --
    exactly like exclude_dirs while walking -- but, unlike exclude_dirs,
    each such directory is located as a whole and added to the archive
    untouched (its entire contents, as-is, via a recursive directory
    add) rather than being dropped. Use this for template files or
    accompanying Python scripts that belong with the run but aren't run
    output and don't need per-file iteration logic applied to them.

    Both are only meaningful when ``recursive=True`` for finding
    matches *inside* the given directories or at deeper nesting.
    ``always_include_dirs`` also gets one extra, always-on check: its
    names are looked for as immediate children of the archive's base
    directory (the common parent when several directories/wildcards
    are given, or the directory itself otherwise) -- e.g. templates/
    and python/ sitting alongside run_1, run_2, ... directly under an
    ensemble folder -- so that a normal ensemble layout gets its
    always-include directories archived even when you target the
    individual run directories directly (with or without wildcards)
    rather than running --recursive over their shared parent.

    ``dry_run`` (set to False by ``--delete`` on the CLI) controls only
    whether the unwanted iteration files are actually removed from the
    source directories. It has no effect on ``compress_path``: if a
    compress target is given, a real archive is written either way,
    containing exactly the same kept-file selection. So
    ``compress_path`` set with ``dry_run=True`` (the default, i.e. no
    ``--delete``) gives you a genuine, already-smaller archive while
    leaving every source file exactly as it was.
    """

    if keep_n_low < 0 or keep_n_high < 1:
        raise ValueError(
            "keep_n_low must be >= 0 and keep_n_high must be >= 1 "
            "(you always need at least the highest iteration)"
        )

    if isinstance(directory, (str, Path)):
        directories = [Path(directory)]
    else:
        directories = [Path(d) for d in directory]

    if not directories:
        print("No directories to process.")
        return

    base_dir = _common_base_dir(directories)
    if len(directories) > 1:
        print(f"Processing {len(directories)} directories (archive base: {base_dir}):")
        for d in directories:
            print(f"  {d}")

    regex = re.compile(pattern, re.IGNORECASE)

    ptok = _normalize_protected_tokens(protected_tokens)
    psuf = _normalize_protected_suffixes(protected_suffixes)
    pfnm = _normalize_protected_filenames(protected_filenames)
    excl_dirs = tuple(str(d).lower() for d in exclude_dirs)
    always_dirs = tuple(str(d).lower() for d in always_include_dirs)

    # always_include_dirs are pruned from the per-file scan just like
    # exclude_dirs -- they are never scanned -- but are located
    # separately below and archived as whole directories, untouched.
    always_include_paths: list[Path] = []
    files: list[Path] = []
    for d in directories:
        if recursive:
            always_include_paths.extend(_find_named_dirs(d, always_dirs, excl_dirs))
        files.extend(_collect_files(d, recursive, excl_dirs + always_dirs))

    # Also look for always_include_dirs as immediate children of the
    # archive's base directory -- e.g. templates/ and python/ sitting
    # alongside run_1, run_2, ... directly under an ensemble folder.
    # This is a single non-recursive listing, so it always runs (no
    # --recursive needed) as long as it wouldn't just re-find something
    # already picked up above (a directory itself in `directories`, or
    # already found by the recursive walk).
    directory_set = {d.resolve() for d in directories}
    already_found = {p.resolve() for p in always_include_paths}
    if base_dir.is_dir():
        for p in sorted(base_dir.iterdir()):
            if not p.is_dir():
                continue
            name_l = p.name.lower()
            if name_l in excl_dirs or name_l not in always_dirs:
                continue
            rp = p.resolve()
            if rp in directory_set or rp in already_found:
                continue
            always_include_paths.append(p)
            already_found.add(rp)

    always_include_paths = sorted(set(always_include_paths))

    matched = []
    protected = []

    for f in files:
        if _is_protected(f, ptok, psuf, pfnm):
            protected.append(f)
            continue
        it = _extract_iter_value(f, regex)
        if it is not None:
            matched.append((f, it))

    protected = sorted(set(protected))
    protected_set = set(protected)

    if not matched:
        print("No iteration files found.")
        if dry_run:
            _print_kept_files(protected, always_include_paths)
        _compress(
            compress_path, protected, base_dir, include_root,
            always_include_paths, deleted_source=not dry_run,
        )
        return

    # Group iteration files by their containing directory so that each
    # run subdirectory gets its own keep-lowest/keep-highest selection.
    groups: dict[Path, list[tuple[Path, int]]] = {}
    for f, it in matched:
        groups.setdefault(f.parent, []).append((f, it))

    to_keep: list[Path] = []
    to_delete: list[Path] = []

    for group_dir in sorted(groups):
        group_matched = groups[group_dir]
        kept_iters = _select_kept_iterations(
            (it for _, it in group_matched), keep_n_low, keep_n_high
        )
        print(f"Kept iterations in {group_dir}: {sorted(kept_iters)}")
        for f, it in group_matched:
            if it in kept_iters:
                to_keep.append(f)
            else:
                to_delete.append(f)

    for d in always_include_paths:
        print(f"Always-included (untouched, not scanned): {d}/")

    final_keep = sorted(set(to_keep).union(protected_set))

    if dry_run:
        _print_kept_files(final_keep, always_include_paths)

    if not dry_run:
        for f in to_delete:
            try:
                f.unlink()
            except Exception as e:
                print(f"FAILED {f}: {e}")

    _compress(
        compress_path, final_keep, base_dir, include_root,
        always_include_paths, deleted_source=not dry_run,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory", nargs="*", default=["."],
        help="One or more directories to process, or wildcard patterns "
             "matching several at once (e.g. 'ensemble/run_*') -- run "
             "directly on top of individual ensemble runs without "
             "needing --recursive over their shared parent. Quote a "
             "pattern if you want this script (rather than your shell) "
             "to expand it. Default: current directory.",
    )
    parser.add_argument(
        "--keep-n-low", type=int, default=1,
        help="Number of lowest matched iterations to keep. 0 keeps "
             "none from this end (the always-protected iter0 file is "
             "unaffected -- it's kept regardless, via the iter0 "
             "protected token). Default: 1.",
    )
    parser.add_argument(
        "--keep-n-high", type=int, default=1,
        help="Number of highest matched iterations to keep. Must be "
             ">= 1 -- you always need at least the highest iteration. "
             "Default: 1.",
    )
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--delete", action="store_true",
        help="Actually remove the unwanted iteration files from the "
             "source directories. Independent of --compress: omit "
             "this to leave every source file untouched (a dry run "
             "for deletion), even while writing a real archive.",
    )
    parser.add_argument(
        "--compress", default=None,
        help="Archive path (.zip, .tgz, .tar.gz). Always writes a real "
             "archive containing the kept-file selection, whether or "
             "not --delete is also given -- so --compress alone gives "
             "a genuine, already-smaller archive with the source "
             "directories left exactly as they were.",
    )
    parser.add_argument("--no-root", action="store_true",
                        help="Do NOT include leading directory in archive")
    parser.add_argument(
        "--exclude-dirs", nargs="+", default=["plots"],
        help="Subdirectory names (case-insensitive) to skip entirely "
             "when --recursive is set: not scanned, not deleted, not "
             "archived. Default: plots",
    )
    parser.add_argument(
        "--always-include-dirs", nargs="+", default=["templates", "python"],
        help="Subdirectory names (case-insensitive) to always treat as "
             "fully protected and always include in full in the "
             "compressed archive, when --recursive is set. "
             "Default: templates python",
    )

    args = parser.parse_args()

    _write_param_summary(__file__, vars(args))

    directories = _expand_directory_args(args.directory)
    if not directories:
        parser.error("No directory matched any of: " + ", ".join(args.directory))

    mt_archive_run(
        directory=directories,
        keep_n_low=args.keep_n_low,
        keep_n_high=args.keep_n_high,
        recursive=args.recursive,
        dry_run=not args.delete,
        compress_path=args.compress,
        include_root=not args.no_root,
        exclude_dirs=args.exclude_dirs,
        always_include_dirs=args.always_include_dirs,
    )


if __name__ == "__main__":
    main()
