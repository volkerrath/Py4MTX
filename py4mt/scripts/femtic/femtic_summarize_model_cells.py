#!/usr/bin/env python3
"""femtic_summarize_model_cells.py

Report the Jacobian parameter dimension and the physical size of the model
for one or more FEMTIC inversion iterations, combining resistivity and
distortion unknowns.

For each iteration the script reads:
    resistivity_block_iterXX.dat  -- one rho value per free region
    distortion_iterXX.dat         -- four C-matrix values per free site
    mesh.dat                      -- node coordinates, for the model size
                                     (read once per directory; optional)

and prints:
    n_rho         free resistivity regions   (flag != 1, excl. air/ocean)
    n_distortion  free distortion parameters (4 x sites with flag == 0)
    n_total       n_rho + n_distortion       -> Jacobian column count
    model size    x, y, z range of the mesh, e.g. x = [-100, 100] km

Usage
-----
Edit the USER SECTION below, or pass paths on the command line (they
replace PATHS):

    python femtic_summarize_model_cells.py /path/to/run/
    python femtic_summarize_model_cells.py resistivity_block_iter*.dat
    python femtic_summarize_model_cells.py        # uses PATHS

The Py4MTX environment must be set: PY4MTX_ROOT and PY4MTX_DATA are read
from the environment, and <PY4MTX_ROOT>/py4mt/modules/ and
<PY4MTX_ROOT>/py4mt/scripts/ are put on sys.path (version.py and util.py
supply the title line printed at start-up).

Optional LaTeX table: set LATEX_FILE in the USER SECTION to the name of a
.tex file. It gets one booktabs row per iteration (elements, regions, n_rho,
n_dist, Jacobian columns and, if mesh.dat was read, the x, y, z range).
LATEX_STANDALONE = True writes a complete document instead of only the
table environment.

AI-generated code -- author: Claude, Anthropic.
Date generated: 2026-06-12.
Review before use in production.

Provenance:
    2026-06-12  Claude, Anthropic  Created.
    2026-08-13  Claude Sonnet 5 (Anthropic)
                Added femtic_summarize_model_cells_summary.md output after
                CLI argument parsing: writes the resolved parameters,
                script path, and run date/time via a self-contained
                _write_param_summary() helper (stdlib only).
    2026-10-24  Claude Sonnet 5.5 (Anthropic)
                Added the physical model size (x, y, z range of the mesh
                nodes, in km by default) to the per-iteration summary,
                read from mesh.dat by the new parse_mesh_extent() (stdlib
                only, one read per directory). Added a USER SECTION with
                all user parameters in capitals (PATHS, MESH_FILE,
                SIZE_UNIT, SIZE_DECIMALS, OCEAN); the ocean override that
                was only a function keyword is now reachable from there.
                The parameter summary file now records all of them.
    2026-10-24  Claude Sonnet 5.5 (Anthropic)
                Added the usual Py4MTX header: PY4MTX_DATA / PY4MTX_ROOT
                from the environment, py4mt/modules and py4mt/scripts put
                on sys.path, version/util title line. The parsing code
                itself is unchanged and still uses the standard library
                only; femtic.py is not imported.
    2026-10-24  Claude Sonnet 5.5 (Anthropic)
                Added the optional LaTeX table (LATEX_FILE, LATEX_CAPTION,
                LATEX_LABEL, LATEX_STANDALONE in the USER SECTION; builder
                latex_table()). The printed summary is unchanged.
"""
from __future__ import annotations

import inspect
import math
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Py4MTX-specific settings and imports
# ---------------------------------------------------------------------------
PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [PY4MTX_ROOT + "/py4mt/modules/", PY4MTX_ROOT + "/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)

from version import versionstrg
import util as utl

version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())
titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng + "\n\n")


# ===========================================================================
# USER SECTION -- edit here; every user parameter is in CAPITALS
# ===========================================================================

# Directories and/or files to scan. Command-line arguments, if given,
# replace this list.
PATHS = ["."]

# Mesh file read for the model size. Looked up in the directory of each
# resistivity_block file (an absolute path is used as it is). None = do not
# print the model-size lines.
MESH_FILE = "mesh.dat"

# Unit of the model-size lines, "km" or "m". Mesh coordinates are taken
# to be in metres (FEMTIC convention).
SIZE_UNIT = "km"

# Decimals of the model-size values (trailing zeros are dropped).
SIZE_DECIMALS = 3

# Region 1 as ocean: None = infer (flag == 1 and rho <= 1 Ohm.m),
# True = ocean, False = no ocean.
OCEAN = None

# Optional LaTeX table of the summaries. None = off; otherwise the name of
# the .tex file to write (a relative path refers to the current directory).
LATEX_FILE = None

# Caption and label of the table, used as they are (write LaTeX, e.g.
# r"Problem size per iteration."). None = default caption / no label.
LATEX_CAPTION = None
LATEX_LABEL = None

# False = only the table environment, to be \input into a paper (needs the
# booktabs package); True = a complete document that can be compiled as is.
LATEX_STANDALONE = False

# ===========================================================================
# END OF USER SECTION
# ===========================================================================


def _write_param_summary(fname: str, params: dict) -> str:
    """Write a Markdown summary of this script's user parameters.

    Self-contained (stdlib only). Writes
    ``<script_dir>/<script_stem>_summary.md`` with the script path, run
    date/time, and the resolved user parameters.
    """
    from datetime import datetime

    script_path = os.path.abspath(fname)
    stem = os.path.splitext(os.path.basename(fname))[0]
    out_path = os.path.join(os.path.dirname(script_path), stem + "_summary.md")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Parameter summary: " + stem, "",
        "- **Script path:** `" + script_path + "`",
        "- **Run date/time:** " + now, "",
        "## User-set parameters", "",
        "| Parameter | Value |", "|---|---|",
    ]
    for k in sorted(params):
        vstr = repr(params[k]).replace("|", "\\|").replace("\n", " ")
        lines.append("| `" + k + "` | `" + vstr + "` |")

    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("[write_param_summary] Wrote parameter summary to " + out_path)
    return out_path


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
# Mesh parser (model size)
# ---------------------------------------------------------------------------

def _is_number(tok: str) -> bool:
    try:
        float(tok)
    except ValueError:
        return False
    return True


def _next_nonblank(fh):
    """Return the next non-blank line of the open file, or None at EOF."""
    for line in fh:
        if line.strip():
            return line
    return None


def parse_mesh_extent(path: Path) -> dict:
    """Bounding box of the node coordinates in a FEMTIC ``mesh.dat``.

    Layout assumed: an optional mesh-type line (``TETRA``, ``HEXA``, ...),
    a line whose first token is the number of nodes, then one line
    ``id  x  y  z`` per node (metres, z positive down). Everything after
    the node block (elements) is not read.

    Returns a dict with ``mesh_type`` (str or None), ``nnode`` (int),
    ``min`` and ``max`` (3-tuples x, y, z in the units of the file).
    Raises ValueError if the file does not follow this layout.
    """
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        first = _next_nonblank(fh)
        if first is None:
            raise ValueError("empty mesh file")
        toks = first.split()
        mesh_type = None
        if not _is_number(toks[0]):
            mesh_type = toks[0].upper()
            second = _next_nonblank(fh)
            if second is None:
                raise ValueError(f"no node count after mesh type {mesh_type!r}")
            toks = second.split()
        try:
            nnode = int(toks[0])
        except ValueError:
            raise ValueError(f"cannot read the node count from {toks!r}") from None
        if nnode <= 0:
            raise ValueError(f"node count {nnode} is not positive")

        xmin = ymin = zmin = math.inf
        xmax = ymax = zmax = -math.inf
        n = 0
        while n < nnode:
            line = fh.readline()
            if not line:
                raise ValueError(f"end of file after {n} of {nnode} nodes")
            p = line.split()
            if not p:
                continue
            if len(p) < 4:
                raise ValueError(f"bad node line {line.strip()!r}")
            try:
                x, y, z = float(p[1]), float(p[2]), float(p[3])
            except ValueError:
                raise ValueError(f"bad node line {line.strip()!r}") from None
            if x < xmin: xmin = x
            if x > xmax: xmax = x
            if y < ymin: ymin = y
            if y > ymax: ymax = y
            if z < zmin: zmin = z
            if z > zmax: zmax = z
            n += 1

    return {
        "mesh_type": mesh_type,
        "nnode":     nnode,
        "min":       (xmin, ymin, zmin),
        "max":       (xmax, ymax, zmax),
    }


_MESH_CACHE: dict = {}


def get_mesh_extent(rho_path: Path):
    """Return ``(extent, message)`` for the mesh next to *rho_path*.

    ``extent`` is the dict of :func:`parse_mesh_extent` or None. ``message``
    says why it is None (file missing / unreadable); it is None as well
    when MESH_FILE is None (model-size lines switched off). The mesh is
    read once per file and cached.
    """
    if MESH_FILE is None:
        return None, None
    mesh_path = rho_path.parent / MESH_FILE
    key = str(mesh_path.resolve())
    if key not in _MESH_CACHE:
        if not mesh_path.is_file():
            _MESH_CACHE[key] = (None, f"{MESH_FILE} not found in {rho_path.parent}")
        else:
            try:
                _MESH_CACHE[key] = (parse_mesh_extent(mesh_path), None)
            except Exception as exc:
                _MESH_CACHE[key] = (None, f"{mesh_path.name}: {exc}")
    return _MESH_CACHE[key]


def _fmt_num(v: float, decimals: int) -> str:
    s = f"{v:.{decimals}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s in ("", "-0"):
        s = "0"
    return s


def size_lines(extent: dict, unit: str = "km", decimals: int = 3) -> list[str]:
    """Format the model size as three lines ``x = [lo, hi] unit (size ...)``."""
    scale = 1.0e-3 if unit == "km" else 1.0
    rows = [
        (name, lo * scale, hi * scale)
        for name, lo, hi in zip("xyz", extent["min"], extent["max"])
    ]
    w = max(len(_fmt_num(v, decimals)) for r in rows for v in r[1:])
    lines = []
    for name, lo, hi in rows:
        lines.append(
            f"{name} = [{_fmt_num(lo, decimals):>{w}}, {_fmt_num(hi, decimals):>{w}}] "
            f"{unit}   (size {_fmt_num(hi - lo, decimals)} {unit})"
        )
    return lines


# ---------------------------------------------------------------------------
# LaTeX table (optional: LATEX_FILE)
# ---------------------------------------------------------------------------

def _tex_int(n: int) -> str:
    """Integer with a thin space as thousands separator (no package needed)."""
    return f"{n:,d}".replace(",", r"\,")


def _tex_range(lo: float, hi: float, scale: float, decimals: int) -> str:
    """``$[lo,\\,hi]$`` in math mode, so that a minus sign is a real minus."""
    return "$[" + _fmt_num(lo * scale, decimals) + r",\," + _fmt_num(hi * scale, decimals) + "]$"


def latex_table(rows: list[dict], unit: str = "km", decimals: int = 3,
                caption: str | None = None, label: str | None = None,
                standalone: bool = False) -> str:
    """LaTeX source (booktabs) of the summary: one table row per iteration.

    *rows* are dicts with the keys ``it``, ``nelem``, ``nreg``, ``n_rho``,
    ``n_dist`` (None if there is no distortion file), ``n_total`` and
    ``mesh`` (the dict of :func:`parse_mesh_extent`, or None). The x, y, z
    range columns appear if at least one row has a mesh. *caption* and
    *label* are used as they are (they are LaTeX). With *standalone* the
    result is a complete document, otherwise only the table environment.
    """
    from datetime import datetime

    scale = 1.0e-3 if unit == "km" else 1.0
    with_size = any(r["mesh"] is not None for r in rows)

    head = ["Iter.", "Elements", "Regions", r"$n_\rho$", r"$n_\mathrm{dist}$",
            r"Jac.\ columns"]
    colspec = "rrrrrr"
    if with_size:
        head += [f"${a}$ ({unit})" for a in "xyz"]
        colspec += "ccc"

    if caption is None:
        caption = (r"Parameter dimension of the FEMTIC inversion per iteration: "
                   r"mesh elements, resistivity regions, free resistivity "
                   r"parameters $n_\rho$, free distortion parameters "
                   r"$n_\mathrm{dist}$, and Jacobian columns "
                   r"$n_\rho + n_\mathrm{dist}$.")
        if with_size:
            caption += (r" The model size is the bounding box of the mesh "
                        r"nodes, including the air layer ($z$ positive down).")

    body = [r"\begin{table}[htbp]", r"  \centering", r"  \small",
            r"  \caption{" + caption + "}"]
    if label:
        body.append(r"  \label{" + label + "}")
    body += [r"  \begin{tabular}{" + colspec + "}", r"    \toprule",
             "    " + " & ".join(head) + r" \\", r"    \midrule"]
    for r in rows:
        cells = [str(r["it"]), _tex_int(r["nelem"]), _tex_int(r["nreg"]),
                 _tex_int(r["n_rho"]),
                 "--" if r["n_dist"] is None else _tex_int(r["n_dist"]),
                 _tex_int(r["n_total"])]
        if with_size:
            m = r["mesh"]
            if m is None:
                cells += ["--"] * 3
            else:
                cells += [_tex_range(lo, hi, scale, decimals)
                          for lo, hi in zip(m["min"], m["max"])]
        body.append("    " + " & ".join(cells) + r" \\")
    body += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    note = ["% Generated by femtic_summarize_model_cells.py on " + stamp]
    if standalone:
        out = [r"\documentclass{article}",
               r"\usepackage[a4paper,margin=1.5cm]{geometry}",
               r"\usepackage{booktabs}", r"\begin{document}", ""] + note + [""] \
              + body + ["", r"\end{document}"]
    else:
        out = note + [r"% Needs \usepackage{booktabs} in the preamble."] + body
    return "\n".join(out) + "\n"


def _write_latex(text: str, path: str) -> None:
    """Write the LaTeX source; a failure is reported but does not stop the run."""
    try:
        Path(path).write_text(text, encoding="utf-8")
    except OSError as exc:
        print(f"[error] cannot write LaTeX table to {path}: {exc}", file=sys.stderr)
        return
    print(f"[latex] Wrote LaTeX table to {Path(path).resolve()}")


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

def print_summary(it: int, rho: dict, dist: dict | None,
                  mesh: dict | None = None, mesh_msg: str | None = None) -> None:
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
    if mesh is not None:
        lines = size_lines(mesh, SIZE_UNIT, SIZE_DECIMALS)
        print(f"  Model size      : {lines[0]}")
        for ln in lines[1:]:
            print(f"                    {ln}")
        print(f"                    (mesh bounding box incl. air, {mesh['nnode']:,d} nodes;"
              f" z positive down)")
    elif mesh_msg:
        print(f"  Model size      : n/a ({mesh_msg})")
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
    if SIZE_UNIT not in ("km", "m"):
        print(f"[error] SIZE_UNIT must be 'km' or 'm', got {SIZE_UNIT!r}", file=sys.stderr)
        sys.exit(2)

    args = sys.argv[1:] or list(PATHS)

    _write_param_summary(__file__, {
        "PATHS":         args,
        "MESH_FILE":     MESH_FILE,
        "SIZE_UNIT":     SIZE_UNIT,
        "SIZE_DECIMALS": SIZE_DECIMALS,
        "OCEAN":         OCEAN,
        "LATEX_FILE":       LATEX_FILE,
        "LATEX_CAPTION":    LATEX_CAPTION,
        "LATEX_LABEL":      LATEX_LABEL,
        "LATEX_STANDALONE": LATEX_STANDALONE,
    })

    pairs = collect_pairs(args)

    if not pairs:
        print("No resistivity_block_iter*.dat files found.", file=sys.stderr)
        sys.exit(1)

    latex_rows: list[dict] = []
    for it, rho_path, dist_path in pairs:
        try:
            rho  = parse_resistivity_block(rho_path, ocean=OCEAN)
            dist = parse_distortion(dist_path) if dist_path else None
            mesh, mesh_msg = get_mesh_extent(rho_path)
            print_summary(it, rho, dist, mesh, mesh_msg)
            latex_rows.append({
                "it":      it,
                "nelem":   rho["nelem"],
                "nreg":    rho["nreg"],
                "n_rho":   rho["n_rho"],
                "n_dist":  dist["n_distortion"] if dist else None,
                "n_total": rho["n_rho"] + (dist["n_distortion"] if dist else 0),
                "mesh":    mesh,
            })
        except Exception as exc:
            print(f"[error] iter {it}: {exc}", file=sys.stderr)

    print(f"\n  {len(pairs)} iteration(s) processed.")

    if LATEX_FILE is not None:
        if latex_rows:
            _write_latex(latex_table(latex_rows, SIZE_UNIT, SIZE_DECIMALS,
                                     LATEX_CAPTION, LATEX_LABEL, LATEX_STANDALONE),
                         LATEX_FILE)
        else:
            print("[warn] no iteration summarised -- no LaTeX table written",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
