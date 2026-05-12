#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean and optionally archive a FEMTIC run directory.

Author: Volker Rath (DIAS)
Created with the help of ChatGPT (GPT-5 Thinking) on 2026-04-07
"""

from __future__ import annotations

import argparse
import re
import tarfile
import zipfile
from pathlib import Path
import os
import sys

PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

for _base in [PY4MTX_ROOT + "/py4mt/modules/"]:
    for _p in [Path(_base), *Path(_base).rglob("*")]:
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

from typing import Iterable


def _normalize_protected_tokens(protected_tokens: Iterable[str]) -> tuple[str, ...]:
    return tuple(str(token).lower() for token in protected_tokens)


def _normalize_protected_suffixes(
    protected_suffixes: Iterable[str],
) -> tuple[str, ...]:
    return tuple(str(suffix).lower() for suffix in protected_suffixes)


def _collect_files(directory: Path, recursive: bool) -> list[Path]:
    if recursive:
        files = [p for p in directory.rglob("*") if p.is_file()]
    else:
        files = [p for p in directory.glob("*") if p.is_file()]
    return sorted(files)


def _is_protected(
    filepath: Path,
    protected_tokens: tuple[str, ...],
    protected_suffixes: tuple[str, ...],
) -> bool:
    name_l = filepath.name.lower()
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
    unique = sorted(set(iter_values))
    return set(unique[:keep_n_low] + unique[-keep_n_high:])


def _arcname(filepath: Path, base_dir: Path, include_root: bool) -> Path:
    rel = filepath.relative_to(base_dir)
    return (base_dir.name / rel) if include_root else rel


def _write_zip(archive_path: Path, files, base_dir: Path, include_root: bool):
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=_arcname(f, base_dir, include_root))


def _write_tgz(archive_path: Path, files, base_dir: Path, include_root: bool):
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tf:
        for f in files:
            tf.add(f, arcname=_arcname(f, base_dir, include_root))


def _compress(compress_path, files, base_dir: Path, dry_run: bool, include_root: bool):
    if compress_path is None:
        return

    archive_path = Path(compress_path)
    name = archive_path.name.lower()
    files = sorted(set(files))

    print(f"\nCompression target: {archive_path}")

    if dry_run:
        for f in files:
            print(f"  WOULD ADD {f}")
        return

    if name.endswith(".zip"):
        _write_zip(archive_path, files, base_dir, include_root)
    elif name.endswith(".tgz") or name.endswith(".tar.gz"):
        _write_tgz(archive_path, files, base_dir, include_root)
    else:
        raise ValueError("Unsupported archive format")

    print(f"Archive written: {archive_path}")


def mt_archive_run(
    directory: str | Path,
    pattern: str = r"_iter(\d+)",
    protected_tokens: Iterable[str] = ("obs", "ref", "mesh", "iter0", "control"),
    protected_suffixes: Iterable[str] = (".log", ".sh", ".cnv"),
    keep_n_low: int = 1,
    keep_n_high: int = 1,
    recursive: bool = False,
    dry_run: bool = True,
    compress_path: str | Path | None = None,
    include_root: bool = True,
) -> None:

    if keep_n_low < 1 or keep_n_high < 1:
        raise ValueError("keep_n_low and keep_n_high must be >= 1")

    directory = Path(directory)
    regex = re.compile(pattern, re.IGNORECASE)

    ptok = _normalize_protected_tokens(protected_tokens)
    psuf = _normalize_protected_suffixes(protected_suffixes)

    files = _collect_files(directory, recursive)

    matched = []
    protected = []

    for f in files:
        if _is_protected(f, ptok, psuf):
            protected.append(f)
            continue
        it = _extract_iter_value(f, regex)
        if it is not None:
            matched.append((f, it))

    protected = sorted(set(protected))
    protected_set = set(protected)

    if not matched:
        print("No iteration files found.")
        _compress(compress_path, protected, directory, dry_run, include_root)
        return

    kept_iters = _select_kept_iterations(
        (it for _, it in matched), keep_n_low, keep_n_high
    )

    to_keep = [f for f, it in matched if it in kept_iters]
    to_delete = [f for f, it in matched if it not in kept_iters]

    final_keep = sorted(set(to_keep).union(protected_set))

    print(f"Kept iterations: {sorted(kept_iters)}")

    if not dry_run:
        for f in to_delete:
            try:
                f.unlink()
            except Exception as e:
                print(f"FAILED {f}: {e}")

    _compress(compress_path, final_keep, directory, dry_run, include_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", default=".")
    parser.add_argument("--keep-n-low", type=int, default=1)
    parser.add_argument("--keep-n-high", type=int, default=1)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--delete", action="store_true")
    parser.add_argument("--compress", default=None)
    parser.add_argument("--no-root", action="store_true",
                        help="Do NOT include leading directory in archive")

    args = parser.parse_args()

    mt_archive_run(
        directory=args.directory,
        keep_n_low=args.keep_n_low,
        keep_n_high=args.keep_n_high,
        recursive=args.recursive,
        dry_run=not args.delete,
        compress_path=args.compress,
        include_root=not args.no_root,
    )


if __name__ == "__main__":
    main()
