#!/usr/bin/env python3
"""
Find third-party Python packages required by a collection of scripts.

Usage
-----
python find_requirements.py /path/to/project
python find_requirements.py /path/to/project --write requirements_detected.txt
"""

from __future__ import annotations

import argparse
import ast
import importlib.metadata as md
import os
import sys
import sysconfig
from pathlib import Path


def stdlib_names() -> set[str]:
    names = set(getattr(sys, "stdlib_module_names", set()))
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    for p in stdlib.iterdir():
        if p.suffix == ".py":
            names.add(p.stem)
        elif p.is_dir():
            names.add(p.name)
    return names


def top_level_imports(path: Path) -> set[str]:
    found: set[str] = set()
    for pyfile in path.rglob("*.py"):
        if any(part in {".git", ".venv", "venv", "__pycache__", "build", "dist"} for part in pyfile.parts):
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


def local_modules(path: Path) -> set[str]:
    local = set()
    for pyfile in path.rglob("*.py"):
        local.add(pyfile.stem)
    for init in path.rglob("__init__.py"):
        local.add(init.parent.name)
    return local


def import_to_distribution_map() -> dict[str, list[str]]:
    return md.packages_distributions()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()

    imports = top_level_imports(args.path)
    stdlib = stdlib_names()
    local = local_modules(args.path)
    pkgmap = import_to_distribution_map()

    third_party_imports = sorted(imports - stdlib - local)

    packages = []
    unresolved = []

    for name in third_party_imports:
        dists = pkgmap.get(name)
        if dists:
            packages.extend(dists)
        else:
            unresolved.append(name)

    packages = sorted(set(packages), key=str.lower)

    print("# Detected packages")
    for pkg in packages:
        print(pkg)

    if unresolved:
        print("\n# Unresolved imports, check manually")
        for name in unresolved:
            print(name)

    if args.write:
        args.write.write_text("\n".join(packages) + "\n", encoding="utf-8")
        print(f"\nWrote {args.write}")


if __name__ == "__main__":
    main()
