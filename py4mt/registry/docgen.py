"""
registry/docgen.py
Author: Volker Rath (DIAS)
Copilot (version) and date: 2026-01-06

Documentation generation utilities.
"""

from typing import List, Dict, Any
from pathlib import Path


def write_markdown_docs(entries: List[Dict[str, Any]], outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    modules = sorted(set(e["module"] for e in entries))

    for module in modules:
        subset = [e for e in entries if e["module"] == module]
        md = _module_to_markdown(module, subset)
        (outdir / f"{module}.md").write_text(md)


def _module_to_markdown(module: str, entries: List[Dict[str, Any]]) -> str:
    lines = [f"# Module `{module}`", ""]

    for e in entries:
        lines.append(f"## `{e['name']}` ({e['type']})")
        if e["signature"]:
            lines.append(f"**Signature:** `{e['signature']}`")
        if e["doc"]:
            lines.append("\n### Docstring\n")
            lines.append(e["doc"])
        if e["provenance"]:
            lines.append("\n### Provenance\n")
            for k, v in e["provenance"].items():
                lines.append(f"- **{k.capitalize()}**: {v}")
        lines.append("\n---\n")

    return "\n".join(lines)
