"""
registry/introspection.py
Author: Volker Rath (DIAS)
Copilot (version) and date: 2026-01-06

Introspection utilities for building provenance-aware module registries.
"""

import inspect
import importlib
from types import ModuleType
from typing import Dict, Any, List


def introspect_module(module: ModuleType) -> List[Dict[str, Any]]:
    """
    Extract callable metadata from a module.

    Returns
    -------
    List[Dict[str, Any]]
        One metadata dictionary per callable.
    """
    results = []

    for name, obj in inspect.getmembers(module):
        if not callable(obj):
            continue
        if obj.__module__ != module.__name__:
            continue

        entry = {
            "module": module.__name__,
            "name": name,
            "type": _callable_type(obj),
            "signature": str(inspect.signature(obj)) if inspect.isfunction(obj) else None,
            "doc": inspect.getdoc(obj) or "",
            "provenance": _extract_provenance(obj),
        }

        results.append(entry)

    return results


def _callable_type(obj):
    if inspect.isfunction(obj):
        return "function"
    if inspect.isclass(obj):
        return "class"
    return "callable"


def _extract_provenance(obj):
    """
    Extract provenance from the top of the docstring.

    Expected format:
    Author: ...
    Copilot: ...
    Date: ...
    """
    doc = inspect.getdoc(obj) or ""
    provenance = {}

    for line in doc.splitlines():
        if line.startswith("Author:"):
            provenance["author"] = line.replace("Author:", "").strip()
        if line.startswith("Copilot"):
            provenance["copilot"] = line.strip()
        if line.startswith("Date:"):
            provenance["date"] = line.replace("Date:", "").strip()

    return provenance
