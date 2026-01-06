"""
registry/registry.py
Author: Volker Rath (DIAS)
Copilot (version) and date: 2026-01-06

A provenance-aware registry for scientific modules.
"""

from typing import Dict, List, Any


class ModuleRegistry:
    def __init__(self):
        self._entries: List[Dict[str, Any]] = []

    def add_entries(self, entries: List[Dict[str, Any]]):
        self._entries.extend(entries)

    def filter(self, **criteria) -> List[Dict[str, Any]]:
        """
        Query registry by arbitrary metadata fields.
        Example:
            registry.filter(type="function", module="my.mesh")
        """
        results = self._entries
        for key, value in criteria.items():
            results = [e for e in results if e.get(key) == value]
        return results

    def to_markdown(self) -> str:
        """
        Produce a Markdown table of all callables.
        """
        lines = ["| Module | Name | Type | Signature |", "|---|---|---|---|"]
        for e in self._entries:
            sig = e["signature"] or ""
            lines.append(f"| {e['module']} | {e['name']} | {e['type']} | `{sig}` |")
        return "\n".join(lines)
