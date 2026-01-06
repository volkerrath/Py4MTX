"""
registry/cli.py
Author: Volker Rath (DIAS)
Copilot (version) and date: 2026-01-06

Command-line interface for the registry toolkit.
Provides introspection, registry building, and documentation generation.
"""

import argparse
import importlib
from pathlib import Path
import json

from .introspection import introspect_module
from .registry import ModuleRegistry
from .docgen import write_markdown_docs


def main():
    parser = argparse.ArgumentParser(
        prog="module-registry",
        description="Provenance-aware module introspection and documentation generator."
    )

    subparsers = parser.add_subparsers(dest="command")

    # -------------------------------
    # introspect
    # -------------------------------
    p_introspect = subparsers.add_parser(
        "introspect",
        help="Extract callable metadata from a module."
    )
    p_introspect.add_argument("module", help="Module to introspect (e.g., mypackage.mymodule)")
    p_introspect.add_argument("--json", help="Write output to JSON file", default=None)

    # -------------------------------
    # build-registry
    # -------------------------------
    p_registry = subparsers.add_parser(
        "build-registry",
        help="Build a registry from one or more modules and export JSON."
    )
    p_registry.add_argument("modules", nargs="+", help="Modules to introspect")
    p_registry.add_argument("--out", required=True, help="Output JSON file")

    # -------------------------------
    # generate-docs
    # -------------------------------
    p_docs = subparsers.add_parser(
        "generate-docs",
        help="Generate Markdown documentation for modules."
    )
    p_docs.add_argument("modules", nargs="+", help="Modules to document")
    p_docs.add_argument("--outdir", required=True, help="Output directory for Markdown files")

    args = parser.parse_args()

    # Dispatch
    if args.command == "introspect":
        _cmd_introspect(args)

    elif args.command == "build-registry":
        _cmd_build_registry(args)

    elif args.command == "generate-docs":
        _cmd_generate_docs(args)

    else:
        parser.print_help()


# ============================================================
# Command implementations
# ============================================================

def _cmd_introspect(args):
    module = importlib.import_module(args.module)
    entries = introspect_module(module)

    if args.json:
        Path(args.json).write_text(json.dumps(entries, indent=2))
    else:
        print(json.dumps(entries, indent=2))


def _cmd_build_registry(args):
    registry = ModuleRegistry()

    for modname in args.modules:
        module = importlib.import_module(modname)
        entries = introspect_module(module)
        registry.add_entries(entries)

    Path(args.out).write_text(json.dumps(registry._entries, indent=2))


def _cmd_generate_docs(args):
    registry = ModuleRegistry()

    for modname in args.modules:
        module = importlib.import_module(modname)
        entries = introspect_module(module)
        registry.add_entries(entries)

    write_markdown_docs(registry._entries, Path(args.outdir))
