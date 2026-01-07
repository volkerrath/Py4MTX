# registry

A lightweight, provenance‑aware introspection and documentation toolkit for scientific Python modules.
`registry` automatically discovers callables, extracts metadata, captures provenance blocks, and generates clean, navigable documentation artifacts (Markdown, JSON, dashboards, etc.).

It is designed for scientific codebases that value **traceability**, **reproducibility**, and **modular architecture**.

---

## ✨ Features

- **Module introspection**
  - Detects functions, classes, and other callables
  - Captures signatures, docstrings, and provenance metadata
  - Filters out imported callables for clean module‑local inventories

- **Provenance extraction**
  - Reads structured metadata from docstrings:
    - `Author:`
    - `Copilot (version) and date:`
    - `Date:`
  - Ensures traceability across scientific workflows

- **Registry system**
  - Stores callable metadata in a queryable structure
  - Supports filtering by module, type, tags, provenance fields, etc.
  - Exportable to Markdown, JSON, or other formats

- **Documentation generation**
  - Auto‑generates Markdown pages per module
  - Produces clean, human‑readable summaries of callables
  - Embeds provenance blocks directly into documentation

- **Extensible architecture**
  - Add tags (e.g., “mesh”, “RBF”, “GIS”)
  - Add dependency graphs or AST‑based call maps
  - Integrate with Sphinx, MkDocs, or dashboard frameworks

---

## 📦 Installation

```bash
pip install registry
