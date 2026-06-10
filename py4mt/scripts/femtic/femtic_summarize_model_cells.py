#!/usr/bin/env python3
"""femtic_summarize_model_cells.py

Summarise the number of air, ocean, other-fixed, and free-parameter cells
in one or more FEMTIC ``resistivity_block_iterXX.dat`` model files.

Usage
-----
    python femtic_summarize_model_cells.py resistivity_block_iter*.dat
    python femtic_summarize_model_cells.py resistivity_block_iter00.dat resistivity_block_iter10.dat
    python femtic_summarize_model_cells.py .          # scan current directory

When ``femtic.py`` is importable from the Python path the script delegates
to :func:`femtic.summarise_model_file` and :func:`femtic._print_model_summary`.
Otherwise it falls back to a self-contained parser that replicates the same
logic without any external dependencies.

File format expected
--------------------
Line 1:   nelem  nreg
Lines 2…nelem+1:  ielem  iregion   (0-based element → region mapping)
Lines nelem+2…:   ireg  rho  rho_lo  rho_hi  n  flag  (one per region)

Region conventions
------------------
- region 0        → AIR   (always fixed)
- region 1        → OCEAN (fixed when flag==1 AND rho ≤ 1 Ω·m; heuristic)
- remaining fixed → extra fixed blocks (flag==1)
- remaining free  → inversion PARAMETERS

Provenance
----------
    2026-06-08  Claude Sonnet 4.6 (Anthropic)   Created (standalone).
    2026-06-10  Claude Sonnet 4.6 (Anthropic)   Refactored: delegates to
                femtic.summarise_model_file / femtic._print_model_summary
                when femtic.py is on the path; self-contained fallback kept.
"""
from __future__ import annotations

import sys
import collections
from pathlib import Path


# ---------------------------------------------------------------------------
# Try to import from femtic.py; fall back to self-contained implementation
# ---------------------------------------------------------------------------

try:
    import femtic as _fem
    _summarise  = _fem.summarise_model_file
    _print_summ = _fem._print_model_summary
    _FEMTIC_AVAILABLE = True
except ImportError:
    _FEMTIC_AVAILABLE = False

    # ------------------------------------------------------------------
    # Self-contained fallback (no femtic.py required)
    # ------------------------------------------------------------------

    def _parse_region_line(line: str):
        parts = line.split()
        if len(parts) < 6:
            raise ValueError(f"Invalid region line (need ≥6 columns): {line!r}")
        return int(parts[0]), float(parts[1]), int(parts[5])   # ireg, rho, flag

    def _infer_ocean(region1_line: str) -> bool:
        _, rho, flag = _parse_region_line(region1_line)
        return (flag == 1) and (rho <= 1.0)

    def _summarise(path, *, ocean=None, out=True) -> dict:
        model_path = Path(path)
        with model_path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        idx = 0
        hdr = lines[idx].split(); idx += 1
        if len(hdr) < 2:
            raise ValueError(f"Invalid resistivity block header: {hdr!r}")
        nelem = int(hdr[0])
        nreg  = int(hdr[1])

        elem_region: list[int] = []
        for _ in range(nelem):
            parts = lines[idx].split(); idx += 1
            elem_region.append(int(parts[1]))

        region_rho:   list[float] = []
        region_flag:  list[int]   = []
        region_lines: list[str]   = []
        for i in range(nreg):
            line = lines[idx]; idx += 1
            ireg, rho, flag = _parse_region_line(line)
            region_rho.append(rho)
            region_flag.append(flag)
            region_lines.append(line)

        ocean_present = False
        ocean_rho_val = None
        if nreg > 1:
            ocean_present = _infer_ocean(region_lines[1]) if ocean is None else bool(ocean)
            if ocean_present:
                ocean_rho_val = region_rho[1]

        fixed = [False] * nreg
        fixed[0] = True
        for i in range(nreg):
            if region_flag[i] == 1:
                fixed[i] = True
        if ocean_present and nreg > 1:
            fixed[1] = True

        reg_counts = collections.Counter(elem_region)
        n_air         = reg_counts.get(0, 0)
        n_ocean       = reg_counts.get(1, 0) if ocean_present else 0
        n_other_fixed = sum(
            reg_counts.get(i, 0)
            for i in range(nreg)
            if fixed[i] and i != 0 and not (i == 1 and ocean_present)
        )
        n_params = sum(
            reg_counts.get(i, 0) for i in range(nreg) if not fixed[i]
        )

        s = {
            "file":          model_path.name,
            "nelem":         nelem,
            "nreg":          nreg,
            "n_air":         n_air,
            "n_ocean":       n_ocean,
            "n_other_fixed": n_other_fixed,
            "n_params":      n_params,
            "ocean_present": ocean_present,
            "ocean_rho":     ocean_rho_val,
            "region_rho":    region_rho,
            "region_flag":   region_flag,
        }
        if out:
            _print_summ(s)
        return s

    def _print_summ(s: dict) -> None:
        n_check = s["n_air"] + s["n_ocean"] + s["n_other_fixed"] + s["n_params"]
        ocean_str = (
            f"yes (rho={s['ocean_rho']:.4g} Ω·m)" if s["ocean_present"] else "no"
        )
        print(f"\n{'─' * 60}")
        print(f"  File          : {s['file']}")
        print(f"  Total cells   : {s['nelem']:>10,d}   ({s['nreg']} regions)")
        print(f"  Ocean inferred: {ocean_str}")
        print(f"  ┌─────────────────────────────────────────┐")
        print(f"  │  Air cells        : {s['n_air']:>10,d}            │")
        print(f"  │  Ocean cells      : {s['n_ocean']:>10,d}            │")
        print(f"  │  Other fixed      : {s['n_other_fixed']:>10,d}            │")
        print(f"  │  Parameters (free): {s['n_params']:>10,d}            │")
        print(f"  │─────────────────────────────────────────│")
        print(f"  │  Check sum        : {n_check:>10,d}            │")
        print(f"  └─────────────────────────────────────────┘")
        if n_check != s["nelem"]:
            print(f"  *** WARNING: check sum {n_check} ≠ nelem {s['nelem']} ***")


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def collect_paths(args: list[str]) -> list[Path]:
    paths: list[Path] = []
    for a in args:
        p = Path(a)
        if p.is_dir():
            found = sorted(p.glob("resistivity_block_iter*.dat"))
            if not found:
                print(f"[warn] No resistivity_block_iter*.dat found in {p}", file=sys.stderr)
            paths.extend(found)
        elif "*" in a or "?" in a:
            paths.extend(sorted(Path(".").glob(a)))
        else:
            paths.append(p)
    return paths


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description="Summarise cell counts in FEMTIC resistivity_block_iterXX.dat files."
    )
    ap.add_argument(
        "paths", nargs="*", default=["."],
        metavar="PATH",
        help="Files, glob patterns, or directories to scan (default: current dir).",
    )
    ap.add_argument(
        "--ocean", choices=["auto", "yes", "no"], default="auto",
        help="Force ocean interpretation of region 1 (default: auto-infer).",
    )
    args = ap.parse_args()

    ocean_flag: bool | None = {"auto": None, "yes": True, "no": False}[args.ocean]

    if _FEMTIC_AVAILABLE:
        print("[info] Using femtic.summarise_model_file", file=sys.stderr)
    else:
        print("[info] femtic.py not found — using built-in fallback parser", file=sys.stderr)

    paths = collect_paths(args.paths)
    if not paths:
        print("No model files found.  Pass paths or a directory.", file=sys.stderr)
        sys.exit(1)

    n_ok = 0
    for p in paths:
        if not p.exists():
            print(f"[warn] File not found: {p}", file=sys.stderr)
            continue
        try:
            _summarise(p, ocean=ocean_flag, out=True)
            n_ok += 1
        except Exception as exc:
            print(f"[error] {p.name}: {exc}", file=sys.stderr)

    print(f"\n{'─' * 60}")
    print(f"  {n_ok}/{len(paths)} file(s) processed.")


if __name__ == "__main__":
    main()
