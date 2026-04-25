#!/usr/bin/env python3
"""
Find third-party Python packages required by a collection of Python scripts.

Scans recursively, excludes hidden folders by default, and allows additional
excluded folder names.

Author: Volker Rath (DIAS)
Created with the help of ChatGPT (GPT-5 Thinking) on 2026-04-25
"""

from __future__ import annotations

import argparse
import ast
import importlib.metadata as md
import sys
import sysconfig
from pathlib import Path


DEFAULT_EXCLUDED_DIRS = {
    "__pycache__",
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "build",
    "dist",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
}


def is_excluded_path(path: Path, root: Path, excluded_dirs: set[str]) -> bool:
    """
    Return True if a path is inside an excluded or hidden directory.

    Parameters
    ----------
    path : Path
        File or directory path to test.
    root : Path
        Root search directory.
    excluded_dirs : set of str
        Directory names to exclude.

    Returns
    -------
    bool
        True if the path should be skipped.
    """
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        rel_parts = path.parts

    for part in rel_parts[:-1]:
        if part in excluded_dirs:
            return True
        if part.startswith("."):
            return True

    return False


def stdlib_names() -> set[str]:
    """
    Return known Python standard-library module names.

    Returns
    -------
    set of str
        Names of standard-library modules.
    """
    names = set(getattr(sys, "stdlib_module_names", set()))

    stdlib = Path(sysconfig.get_paths()["stdlib"])
    if stdlib.exists():
        for item in stdlib.iterdir():
            if item.suffix == ".py":
                names.add(item.stem)
            elif item.is_dir():
                names.add(item.name)

    return names


def top_level_imports(root: Path, excluded_dirs: set[str]) -> set[str]:
    """
    Recursively collect top-level imports from Python files.

    Parameters
    ----------
    root : Path
        Root directory to scan.
    excluded_dirs : set of str
        Directory names to exclude.

    Returns
    -------
    set of str
        Top-level imported module names.
    """
    found: set[str] = set()

    for pyfile in root.rglob("*.py"):
        if is_excluded_path(pyfile, root, excluded_dirs):
            continue

        try:
            tree = ast.parse(pyfile.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Warning: skipped {pyfile}: {exc}", file=sys.stderr)
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    found.add(alias.name.split(".")[0])

            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    found.add(node.module.split(".")[0])

    return found


def local_modules(root: Path, excluded_dirs: set[str]) -> set[str]:
    """
    Recursively collect local module and package names.

    Parameters
    ----------
    root : Path
        Root directory to scan.
    excluded_dirs : set of str
        Directory names to exclude.

    Returns
    -------
    set of str
        Names of local Python modules and packages.
    """
    local: set[str] = set()

    for pyfile in root.rglob("*.py"):
        if is_excluded_path(pyfile, root, excluded_dirs):
            continue
        local.add(pyfile.stem)

    for init_file in root.rglob("__init__.py"):
        if is_excluded_path(init_file, root, excluded_dirs):
            continue
        local.add(init_file.parent.name)

    return local


def resolve_distributions(import_names: set[str]) -> tuple[list[str], list[str]]:
    """
    Map import names to installed distribution package names.

    Parameters
    ----------
    import_names : set of str
        Top-level import names.

    Returns
    -------
    tuple[list[str], list[str]]
        Resolved package names and unresolved import names.
    """
    package_map = md.packages_distributions()

    packages: list[str] = []
    unresolved: list[str] = []

    for name in sorted(import_names):
        dists = package_map.get(name)
        if dists:
            packages.extend(dists)
        else:
            unresolved.append(name)

    return sorted(set(packages), key=str.lower), unresolved


def main() -> None:
    """
    Command-line entry point.

    Returns
    -------
    None
        Writes detected requirements to stdout and optionally to a file.
    """
    parser = argparse.ArgumentParser(
        description="Find required third-party packages from Python imports."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Root directory to scan recursively.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Additional folder names to exclude.",
    )
    parser.add_argument(
        "--no-default-excludes",
        action="store_true",
        help="Do not use the built-in default exclusion list.",
    )
    parser.add_argument(
        "--write",
        type=Path,
        help="Write detected package names to this requirements file.",
    )

    args = parser.parse_args()

    root = args.path.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    excluded_dirs = set(args.exclude)
    if not args.no_default_excludes:
        excluded_dirs |= DEFAULT_EXCLUDED_DIRS

    imports = top_level_imports(root, excluded_dirs)
    stdlib = stdlib_names()
    local = local_modules(root, excluded_dirs)

    third_party_imports = imports - stdlib - local
    packages, unresolved = resolve_distributions(third_party_imports)

    print("# Detected packages")
    for package in packages:
        print(package)

    if unresolved:
        print("\n# Unresolved imports, check manually")
        for name in unresolved:
            print(name)

    if args.write:
        args.write.write_text("\n".join(packages) + "\n", encoding="utf-8")
        print(f"\nWrote {args.write}")


if __name__ == "__main__":
    main()
