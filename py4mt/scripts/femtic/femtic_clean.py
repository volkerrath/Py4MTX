#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_clean.py -- delete all files in a list of directories except a list
of protected files.

For every directory in DIRS (names or glob patterns) all files are removed
except those whose name matches one of the protected patterns in FILES.
Nothing is deleted unless --delete is given: the default is a dry run that
prints, per directory, how many files would be kept and deleted, and how
much space would be freed.

Usage
-----
Edit the USER SECTION below, or pass everything on the command line (what
is given there replaces the corresponding USER SECTION entry):

    # dry run (default): only report
    python femtic_clean.py --dirs './fem*' --files a.dat b.h5 'x*iter*.dat'

    # really delete
    python femtic_clean.py --dirs './fem*' --files a.dat b.h5 'x*iter*.dat' --delete

    # key=value form; items separated by spaces and/or commas, brackets and
    # quotes are optional
    python femtic_clean.py dirs='./fem*' files=['a.dat', 'b.h5', 'x*iter*.dat']
    python femtic_clean.py dirs=./fem* files=a.dat,b.h5,'x*iter*.dat' --delete

Quote glob patterns (single quotes) so that the shell does not expand them
before the script sees them; the script expands DIRS itself.

Matching rules
--------------
- DIRS: directory names or glob patterns (shell wildcards). Relative
  patterns are taken relative to WORK_DIR (default: the current directory).
  Matches that are not directories are ignored.
- FILES: shell-style wildcards (* ? [seq]), case-sensitive. A pattern without
  "/" is matched against the file name only; a pattern with "/" is matched
  against the path relative to the cleaned directory (POSIX separators),
  e.g. "plots/*".
- Only files directly inside each directory are considered. With
  --recursive (RECURSIVE) subdirectories are searched too; subdirectories
  themselves are never removed. Symbolic links to directories are never
  followed or removed. Symbolic links to files (also broken ones) count as
  files: the link is removed, never its target.
- With --recursive, a listed directory that lies inside another listed
  directory is cleaned as part of the outer one (not scanned twice).
- This script itself is always protected.

Safety
------
- Dry run unless --delete (DELETE = True) is given.
- Refuses to run without any protected pattern, and (with --delete) if a
  protected pattern matches no file in any of the directories (usually a
  typo). --force overrides both.
- Refuses to clean "/" or the home directory.
- Only the standard library is used, so the script also runs on a login node
  without the Py4MTX environment.

Parameter summary
-----------------
At the end of a run femtic_clean_summary.md (user-set parameters, script
path, run date/time and a result table) is written to WORK_DIR.

@author: vrath
@project: py4mt
@created: 2026-10-28
@modified: 2026-10-28

AI-generated code -- author: Claude, Anthropic.
Date generated: 2026-10-28.
Review before use in production; the script deletes files.

Provenance
----------
    2026-10-28  Claude Sonnet 5.5 (Anthropic)
                Created. Dry run by default; --delete to act; protected
                patterns by file name or relative path; optional recursion;
                key=value command-line form; self-contained summary writer
                (stdlib only) that puts femtic_clean_summary.md into
                WORK_DIR.
"""
from __future__ import annotations

import argparse
import fnmatch
import glob
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ===========================================================================
# USER SECTION -- edit here; every user parameter is in CAPITALS.
# Command-line arguments, if given, replace the corresponding entry.
# ===========================================================================

# Base for relative DIRS patterns, and the directory that receives
# femtic_clean_summary.md.
WORK_DIR = "./"

# Directories to clean: names or glob patterns, e.g. ["./fem*"].
DIRS: List[str] = []

# Protected files (kept): patterns, e.g. ["control.dat", "mesh.dat",
# "observe.dat", "*iter0*.dat", "results_iter*.h5"].
FILES: List[str] = []

RECURSIVE = False   # True: also clean files in subdirectories
DELETE = False      # True: really delete (False = dry run)
VERBOSE = False     # True: list every file with its KEEP / DELETE decision
FORCE = False       # True: skip the empty-protect-list and unmatched-pattern checks

# ===========================================================================
# END OF USER SECTION
# ===========================================================================

SEP = "-" * 78
DSEP = "=" * 78
_KEYS = {"dirs": "--dirs", "files": "--files"}


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _normalize_argv(argv: List[str]) -> List[str]:
    """Rewrite ``dirs=...`` / ``files=[...]`` into ``--dirs ...`` / ``--files ...``.

    A key=value group consists of the ``key=value`` token and all following
    tokens up to the next flag (starting with "-") or the next key=value
    token, so that ``files=['a.dat', 'b.h5']`` works even though the shell
    splits it at the blanks. Within a group, a leading "[" (on the first
    token) and the matching trailing "]" (on the last token) are removed,
    items are split at commas, and quotes around items are stripped. Bracket
    characters inside patterns (``x[0-9].dat``) are left alone unless the
    group itself starts with "[". Tokens that are not part of a key=value
    group are passed through unchanged.
    """
    out: List[str] = []
    group: Optional[Tuple[str, List[str]]] = None

    def flush() -> None:
        nonlocal group
        if group is None:
            return
        flag, toks = group
        if toks and toks[0].lstrip().startswith("["):
            toks[0] = toks[0].lstrip()[1:]
            toks[-1] = toks[-1].rstrip()
            if toks[-1].endswith("]"):
                toks[-1] = toks[-1][:-1]
        items = []
        for tok in toks:
            for piece in tok.split(","):
                piece = piece.strip().strip("'\"").strip()
                if piece:
                    items.append(piece)
        out.append(flag)
        out.extend(items)
        group = None

    for tok in argv:
        key, sep, val = tok.partition("=")
        if sep and key in _KEYS:
            flush()
            group = (_KEYS[key], [val])
        elif tok.startswith("-"):
            flush()
            out.append(tok)
        elif group is not None:
            group[1].append(tok)
        else:
            out.append(tok)
    flush()
    return out


# ---------------------------------------------------------------------------
# Directories and files
# ---------------------------------------------------------------------------

def expand_dirs(patterns: List[str], *, base: str = ".") -> Tuple[List[str], List[str]]:
    """Expand directory names / glob patterns into existing directories.

    Relative patterns refer to ``base``. Returns ``(dirs, unmatched)``:
    directories in the order of the patterns (sorted within a pattern, no
    duplicates, duplicates being detected via the real path), and the
    patterns that matched no directory.
    """
    dirs: List[str] = []
    seen = set()
    unmatched: List[str] = []
    for pat in patterns:
        full = os.path.expanduser(pat)
        if not os.path.isabs(full):
            full = os.path.join(os.path.expanduser(base), full)
        hits = [h for h in sorted(glob.glob(full)) if os.path.isdir(h)]
        if not hits:
            unmatched.append(pat)
        for h in hits:
            key = os.path.realpath(h)
            if key not in seen:
                seen.add(key)
                dirs.append(os.path.normpath(h))
    return dirs, unmatched


def match_protected(rel: str, patterns: List[str]) -> List[str]:
    """Return the protected patterns that match a file.

    ``rel`` is the path relative to the cleaned directory (POSIX separators).
    A pattern without "/" is matched against the file name, one with "/"
    against ``rel``. Matching is case-sensitive.
    """
    name = rel.rsplit("/", 1)[-1]
    hits = []
    for pat in patterns:
        target = rel if "/" in pat else name
        if fnmatch.fnmatchcase(target, pat):
            hits.append(pat)
    return hits


def scan_dir(root: str, *, recursive: bool = False) -> Tuple[List[Tuple[str, str]], int, List[str]]:
    """List the files of a directory.

    Returns ``(files, n_subdirs, errors)``. ``files`` holds
    ``(path, rel)`` tuples, ``rel`` being relative to ``root`` with "/"
    separators. Directories, and symbolic links to directories, are not
    files: with ``recursive`` the real subdirectories are entered, otherwise
    they are only counted in ``n_subdirs``. Symbolic links are never
    followed.
    """
    files: List[Tuple[str, str]] = []
    errors: List[str] = []
    n_sub = 0

    def walk(path: str, prefix: str) -> None:
        nonlocal n_sub
        try:
            with os.scandir(path) as it:
                entries = sorted(it, key=lambda e: e.name)
        except OSError as exc:
            errors.append("%s: %s" % (path, exc))
            return
        for e in entries:
            rel = prefix + e.name
            try:
                is_real_dir = e.is_dir(follow_symlinks=False)
                is_any_dir = is_real_dir or e.is_dir(follow_symlinks=True)
            except OSError:
                is_real_dir = is_any_dir = False
            if is_any_dir:
                n_sub += 1
                if recursive and is_real_dir:
                    walk(e.path, rel + "/")
            else:
                files.append((e.path, rel))

    walk(root, "")
    return files, n_sub, errors


def plan_dir(root: str, protect: List[str], *, recursive: bool = False,
             self_path: Optional[str] = None) -> Dict:
    """Decide for every file of ``root`` whether it is kept or deleted.

    Returns a dict with ``dir``, ``keep`` (list of ``(path, rel, patterns)``),
    ``delete`` (list of ``(path, rel, size)``), ``n_subdirs``, ``errors``,
    ``hits`` (protected pattern -> number of files matched) and
    ``n_self`` (files kept because they are this script).
    """
    files, n_sub, errors = scan_dir(root, recursive=recursive)
    keep, delete = [], []
    hits: Dict[str, int] = {p: 0 for p in protect}
    n_self = 0
    for path, rel in files:
        pats = match_protected(rel, protect)
        for p in pats:
            hits[p] += 1
        if self_path is not None and os.path.realpath(path) == self_path:
            n_self += 1
            keep.append((path, rel, ["<this script>"]))
        elif pats:
            keep.append((path, rel, pats))
        else:
            try:
                size = os.lstat(path).st_size
            except OSError:
                size = 0
            delete.append((path, rel, size))
    return {"dir": root, "keep": keep, "delete": delete, "n_subdirs": n_sub,
            "errors": errors, "hits": hits, "n_self": n_self}


def delete_planned(plans: List[Dict]) -> Tuple[int, int, List[str]]:
    """Remove the files marked for deletion in ``plans``.

    Returns ``(n_deleted, n_bytes, errors)``. A file that has vanished in the
    meantime is not an error. Symbolic links are removed themselves, never
    their targets.
    """
    n_del, n_bytes = 0, 0
    errors: List[str] = []
    for plan in plans:
        n_dir = 0
        for path, _rel, size in plan["delete"]:
            try:
                os.unlink(path)
            except FileNotFoundError:
                continue
            except OSError as exc:
                errors.append("%s: %s" % (path, exc))
                continue
            n_dir += 1
            n_bytes += size
        n_del += n_dir
        print("  deleted %6d file(s) in %s" % (n_dir, plan["dir"]))
    return n_del, n_bytes, errors


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def fmt_size(nbytes: float) -> str:
    """Human-readable size (binary prefixes)."""
    size = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return ("%d %s" % (size, unit)) if unit == "B" else ("%.1f %s" % (size, unit))
        size /= 1024.0
    return "%.1f TiB" % size


def print_plan(plans: List[Dict], protect: List[str], *, delete: bool,
               recursive: bool, verbose: bool) -> Dict:
    """Print the per-directory table, the protected-pattern hit counts and
    (verbose) every decision. Returns the totals."""
    print(DSEP)
    print(" femtic_clean: %s" % ("DELETE" if delete else "DRY RUN (nothing is deleted; use --delete)"))
    print(DSEP)
    print(" Protected patterns : %s" % (", ".join(protect) if protect else "(none)"))
    print(" Subdirectories     : %s" % ("searched (recursive)" if recursive else "not searched"))
    print(SEP)

    width = max([len(p["dir"]) for p in plans] + [9])
    head = " %-*s %8s %8s %8s %12s %8s" % (width, "directory", "files", "keep",
                                           "delete", "size", "subdirs")
    print(head)
    print(SEP)
    tot = {"files": 0, "keep": 0, "delete": 0, "bytes": 0, "self": 0}
    hits: Dict[str, int] = {p: 0 for p in protect}
    for plan in plans:
        n_keep, n_del = len(plan["keep"]), len(plan["delete"])
        n_bytes = sum(s for _p, _r, s in plan["delete"])
        print(" %-*s %8d %8d %8d %12s %8d" % (width, plan["dir"], n_keep + n_del,
                                              n_keep, n_del, fmt_size(n_bytes),
                                              plan["n_subdirs"]))
        tot["files"] += n_keep + n_del
        tot["keep"] += n_keep
        tot["delete"] += n_del
        tot["bytes"] += n_bytes
        tot["self"] += plan["n_self"]
        for p, n in plan["hits"].items():
            hits[p] += n
        for err in plan["errors"]:
            print("   [warn] cannot read %s" % err)
    print(SEP)
    print(" %-*s %8d %8d %8d %12s" % (width, "total", tot["files"], tot["keep"],
                                      tot["delete"], fmt_size(tot["bytes"])))
    print(SEP)

    if protect:
        pw = max(len(p) for p in protect)
        print(" Files matched by each protected pattern (all directories):")
        for p in protect:
            note = "   <-- matches nothing" if hits[p] == 0 else ""
            print("   %-*s %8d%s" % (pw, p, hits[p], note))
        print(SEP)
    if tot["self"]:
        print(" This script is in a cleaned directory and was kept.")

    if verbose:
        for plan in plans:
            print(" %s" % plan["dir"])
            for _path, rel, pats in plan["keep"]:
                print("   KEEP   %s   [%s]" % (rel, ", ".join(pats)))
            for _path, rel, size in plan["delete"]:
                print("   DELETE %s   (%s)" % (rel, fmt_size(size)))
        print(SEP)
    tot["hits"] = hits
    return tot


# ---------------------------------------------------------------------------
# Parameter summary (self-contained, stdlib only)
# ---------------------------------------------------------------------------

def write_param_summary(fname: str, params: Dict, results: Dict, *,
                        out_dir: Optional[str] = None) -> Optional[str]:
    """Write ``femtic_clean_summary.md`` (parameters, script path, run date/time,
    results).

    The file goes to ``out_dir`` (created if missing); if that fails the
    script directory is tried. Failing to write the summary never raises --
    the files may already have been deleted by then.
    """
    script_path = os.path.abspath(fname)
    stem = os.path.splitext(os.path.basename(fname))[0]
    name = stem + "_summary.md"
    targets = []
    if out_dir:
        targets.append(os.path.join(os.path.abspath(os.path.expanduser(str(out_dir))), name))
    targets.append(os.path.join(os.path.dirname(script_path), name))

    def row(k: str, v) -> str:
        vstr = repr(v).replace("|", "\\|").replace("\n", " ")
        if len(vstr) > 300:
            vstr = vstr[:300] + " ... (truncated)"
        return "| `" + k + "` | `" + vstr + "` |"

    lines = [
        "# Parameter summary: " + stem, "",
        "- **Script path:** `" + script_path + "`",
        "- **Run date/time:** " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "",
        "## User-set parameters", "",
        "| Parameter | Value |", "|---|---|",
    ]
    lines += [row(k, params[k]) for k in sorted(params)]
    lines += ["", "## Result", "", "| Item | Value |", "|---|---|"]
    lines += [row(k, results[k]) for k in results]
    text = "\n".join(lines) + "\n"

    for n, target in enumerate(targets):
        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "w") as fh:
                fh.write(text)
            print("[write_param_summary] Wrote parameter summary to " + target)
            return target
        except OSError as exc:
            more = ("; trying " + targets[n + 1] + " instead.") if n + 1 < len(targets) else "."
            print("[write_param_summary] Cannot write to " + target + " (" + str(exc) + ")" + more)
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    """Command-line entry point. Returns the exit code (0 ok, 1 errors while
    deleting, 2 refused / usage error)."""
    argv = _normalize_argv(list(sys.argv[1:] if argv is None else argv))
    ap = argparse.ArgumentParser(
        prog="femtic_clean.py",
        description="Delete all files in the given directories except the "
                    "protected ones. Dry run unless --delete is given.",
        epilog="Also accepted: dirs='./fem*' files=['a.dat', 'b.h5', 'x*iter*.dat']",
    )
    ap.add_argument("--dirs", nargs="+", metavar="DIR", default=None,
                    help="directories or glob patterns to clean (replaces DIRS)")
    ap.add_argument("--files", nargs="+", metavar="PATTERN", default=None,
                    help="protected file patterns, e.g. a.dat 'x*iter*.dat' (replaces FILES)")
    ap.add_argument("--work-dir", default=None, metavar="DIR",
                    help="base of relative --dirs and location of the summary (replaces WORK_DIR)")
    ap.add_argument("--recursive", action="store_true", help="also clean subdirectories")
    ap.add_argument("--delete", action="store_true", help="really delete (default: dry run)")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="list every file with its decision")
    ap.add_argument("--force", action="store_true",
                    help="allow an empty protect list and unmatched protected patterns")
    args = ap.parse_args(argv)

    work_dir = args.work_dir if args.work_dir is not None else WORK_DIR
    dir_pats = list(args.dirs if args.dirs is not None else DIRS)
    protect = [p[2:] if p.startswith("./") else p
               for p in (args.files if args.files is not None else FILES)]
    recursive = RECURSIVE or args.recursive
    delete = DELETE or args.delete
    verbose = VERBOSE or args.verbose
    force = FORCE or args.force

    dirs, unmatched = expand_dirs(dir_pats, base=work_dir)
    for pat in unmatched:
        print("[warn] no directory matches %r" % pat, file=sys.stderr)
    if not dirs:
        print("[error] no directory to clean: give --dirs (or set DIRS).", file=sys.stderr)
        return 2

    if recursive:
        # A listed directory inside another listed one is covered by the
        # outer one; scanning it twice would double-count its files.
        real = [os.path.realpath(d) for d in dirs]
        kept_dirs = []
        for d, r in zip(dirs, real):
            outer = [dirs[k] for k, o in enumerate(real) if r != o and r.startswith(o + os.sep)]
            if outer:
                print("[info] %s lies inside %s and is cleaned as part of it." % (d, outer[0]))
            else:
                kept_dirs.append(d)
        dirs = kept_dirs

    forbidden = {os.path.realpath(os.sep), os.path.realpath(os.path.expanduser("~"))}
    for d in dirs:
        if os.path.realpath(d) in forbidden:
            print("[error] refusing to clean %s (root or home directory)." % d, file=sys.stderr)
            return 2

    if not protect and not force:
        print("[error] no protected files given: this would delete EVERYTHING in the "
              "directories. Give --files (or set FILES), or --force if that is intended.",
              file=sys.stderr)
        return 2

    self_path = os.path.realpath(__file__)
    plans = [plan_dir(d, protect, recursive=recursive, self_path=self_path) for d in dirs]
    tot = print_plan(plans, protect, delete=delete, recursive=recursive, verbose=verbose)

    dead = [p for p, n in tot["hits"].items() if n == 0]
    if dead and delete and not force:
        print("[error] refusing to delete: protected pattern(s) match no file in any "
              "directory (typo?): %s. Check them, or use --force." % ", ".join(dead),
              file=sys.stderr)
        return 2
    if dead:
        print("[warn] protected pattern(s) matching nothing: %s" % ", ".join(dead),
              file=sys.stderr)

    n_deleted, bytes_deleted, errors = 0, 0, []
    for plan in plans:
        errors += plan["errors"]
    if delete:
        print(" Deleting ...")
        n_deleted, bytes_deleted, derrors = delete_planned(plans)
        errors += derrors
        for err in derrors:
            print("[error] cannot delete %s" % err, file=sys.stderr)
        print(SEP)
        print(" %d file(s) deleted, %s freed, %d kept, %d error(s)."
              % (n_deleted, fmt_size(bytes_deleted), tot["keep"], len(errors)))
    else:
        print(" Would delete %d of %d file(s), freeing %s. Nothing was changed."
              % (tot["delete"], tot["files"], fmt_size(tot["bytes"])))
    print(DSEP)

    write_param_summary(
        __file__,
        {"WORK_DIR": work_dir, "DIRS": dir_pats, "FILES": protect,
         "RECURSIVE": recursive, "DELETE": delete, "VERBOSE": verbose, "FORCE": force},
        {"mode": "delete" if delete else "dry run",
         "directories": dirs,
         "files seen": tot["files"],
         "files kept": tot["keep"],
         "files deleted" if delete else "files that would be deleted":
             n_deleted if delete else tot["delete"],
         "bytes freed" if delete else "bytes that would be freed":
             bytes_deleted if delete else tot["bytes"],
         "errors": len(errors)},
        out_dir=work_dir,
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
