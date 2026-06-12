#!/usr/bin/env python3
"""summarize_model_cells.py

Report the Jacobian parameter dimension for one or more FEMTIC inversion
iterations, combining resistivity and distortion unknowns.

For each iteration the script reads:
    resistivity_block_iterXX.dat  -- one rho value per free region
    distortion_iterXX.dat         -- four C-matrix values per free site

and prints:
    n_rho        free resistivity regions   (flag != 1, excl. air/ocean)
    n_distortion free distortion parameters (4 x sites with flag == 0)
    n_total      n_rho + n_distortion       -> Jacobian column count

Usage
-----
    python summarize_model_cells.py /path/to/run/
    python summarize_model_cells.py resistivity_block_iter*.dat
    python summarize_model_cells.py        # current directory

AI-generated code -- author: Claude, Anthropic.
Date generated: 2026-06-12.
Review before use in production.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Resistivity block parser
# ---------------------------------------------------------------------------

def _parse_region_line(line: str):
    parts = line.split()
    if len(parts) < 6:
        raise ValueError(f"Bad region line: {line!r}")
    return int(parts[0]), float(parts[1]), int(parts[5])   # ireg, rho, flag


def _infer_ocean(line: str) -> bool:
    _, rho, flag = _parse_region_line(line)
    return (flag == 1) and (rho <= 1.0)


def parse_resistivity_block(path: Path, *, ocean: bool | None = None) -> dict:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    idx = 0
    nelem, nreg = map(int, lines[idx].split()[:2]); idx += 1
    idx += nelem   # skip element->region mapping

    region_rho, region_flag, region_lines = [], [], []
    for i in range(nreg):
        ireg, rho, flag = _parse_region_line(lines[idx]); idx += 1
        if ireg != i:
            raise ValueError(f"Region index mismatch: expected {i}, got {ireg}.")
        region_rho.append(rho)
        region_flag.append(flag)
        region_lines.append(lines[idx - 1])

    ocean_present = False
    if nreg > 1:
        ocean_present = _infer_ocean(region_lines[1]) if ocean is None else bool(ocean)

    fixed = [rf == 1 for rf in region_flag]
    fixed[0] = True
    if ocean_present and nreg > 1:
        fixed[1] = True

    n_ocean_set = {1} if ocean_present else set()
    return {
        "nelem":         nelem,
        "nreg":          nreg,
        "n_air":         1,
        "n_ocean":       int(ocean_present),
        "n_other_fixed": sum(1 for i in range(nreg) if fixed[i] and i not in ({0} | n_ocean_set)),
        "n_rho":         sum(1 for i in range(nreg) if not fixed[i]),
        "ocean_present": ocean_present,
        "ocean_rho":     region_rho[1] if ocean_present else None,
    }


# ---------------------------------------------------------------------------
# Distortion file parser
# ---------------------------------------------------------------------------

def parse_distortion(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    nsites = int(lines[0].strip())
    n_free = n_fixed = 0
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 6:
            continue
        if int(parts[5]) == 0:
            n_free += 1
        else:
            n_fixed += 1
    return {
        "nsites":        nsites,
        "n_free_sites":  n_free,
        "n_fixed_sites": n_fixed,
        "n_distortion":  n_free * 4,
    }


# ---------------------------------------------------------------------------
# Pair up files by iteration number
# ---------------------------------------------------------------------------

def _iter_number(path: Path) -> int | None:
    m = re.search(r"iter(\d+)", path.name)
    return int(m.group(1)) if m else None


def collect_pairs(args: list[str]) -> list[tuple[int, Path, Path | None]]:
    rho_paths:  dict[int, Path] = {}
    dist_paths: dict[int, Path] = {}
    for a in args:
        p = Path(a)
        if p.is_dir():
            for f in sorted(p.glob("resistivity_block_iter*.dat")):
                n = _iter_number(f)
                if n is not None:
                    rho_paths[n] = f
            for f in sorted(p.glob("distortion_iter*.dat")):
                n = _iter_number(f)
                if n is not None:
                    dist_paths[n] = f
        else:
            n = _iter_number(p)
            if n is None:
                print(f"[warn] Cannot parse iteration number from {p.name}", file=sys.stderr)
                continue
            if "resistivity_block" in p.name:
                rho_paths[n] = p
            elif "distortion" in p.name:
                dist_paths[n] = p
    return [(i, rho_paths[i], dist_paths.get(i)) for i in sorted(rho_paths)]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_summary(it: int, rho: dict, dist: dict | None) -> None:
    n_dist  = dist["n_distortion"] if dist else 0
    n_total = rho["n_rho"] + n_dist

    ocean_str = (
        f"yes  (rho = {rho['ocean_rho']:.4g} Ohm.m)" if rho["ocean_present"] else "no"
    )

    SEP  = "  " + "-" * 52
    DSEP = "  " + "=" * 52
    LW, RW = 26, 8   # label width, value width

    def row(label, value, note=""):
        v = f"{value:>{RW},d}" if isinstance(value, int) else f"{value:>{RW}}"
        return f"  {label:<{LW}}  {v}  {note}".rstrip()

    print(f"\n{SEP}")
    print(f"  Iteration       : {it}")
    print(f"  Total elements  : {rho['nelem']:>10,d}  (tetrahedra)")
    print(f"  Total regions   : {rho['nreg']:>10,d}  (resistivity blocks)")
    print(f"  Ocean inferred  : {ocean_str}")
    print(SEP)
    print(row("Air regions",          rho["n_air"]))
    print(row("Ocean regions",        rho["n_ocean"]))
    print(row("Other fixed regions",  rho["n_other_fixed"]))
    print(row("Free rho regions",     rho["n_rho"],           "<- n_rho"))
    if dist:
        print(row("Free sites (distort.)",dist["n_free_sites"],  f"x 4 values"))
        print(row("Distortion params",    dist["n_distortion"],  "<- n_dist"))
    else:
        print(row("Distortion params",    "n/a",
                  f"(no distortion_iter{it:02d}.dat)"))
    print(DSEP)
    print(row("Jacobian columns",     n_total,                "= n_rho + n_dist"))
    print(SEP)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    args = sys.argv[1:] or ["."]
    pairs = collect_pairs(args)

    if not pairs:
        print("No resistivity_block_iter*.dat files found.", file=sys.stderr)
        sys.exit(1)

    for it, rho_path, dist_path in pairs:
        try:
            rho  = parse_resistivity_block(rho_path)
            dist = parse_distortion(dist_path) if dist_path else None
            print_summary(it, rho, dist)
        except Exception as exc:
            print(f"[error] iter {it}: {exc}", file=sys.stderr)

    print(f"\n  {len(pairs)} iteration(s) processed.")


if __name__ == "__main__":
    main()
