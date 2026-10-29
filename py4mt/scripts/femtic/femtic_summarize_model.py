#!/usr/bin/env python3
"""femtic_summarize_model.py

Report the Jacobian parameter dimension and the physical size of the model
for one or more FEMTIC inversion iterations, combining resistivity and
distortion unknowns. Works for tetrahedral and hexahedral meshes; the
hexahedra may be deformed (no regular grid is assumed anywhere).

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
    cell type     tetrahedra / hexahedra, from the type line of mesh.dat

Optional (CELL_STATS = True, needs numpy): the element block of mesh.dat is
read as well, and the script prints the total model volume and the smallest,
median and largest cell volume. Hexahedron volumes are exact for deformed
cells (trilinear map, 2 x 2 x 2 Gauss quadrature, which is exact for the
volume integral, also for non-planar faces). For hexahedra it also says
whether they are all axis-aligned boxes or (partly) deformed.

Usage
-----
Edit the USER SECTION below, or pass paths on the command line (they
replace PATHS):

    python femtic_summarize_model.py /path/to/run/
    python femtic_summarize_model.py resistivity_block_iter*.dat
    python femtic_summarize_model.py        # uses PATHS

The Py4MTX environment must be set: PY4MTX_ROOT and PY4MTX_DATA are read
from the environment, and <PY4MTX_ROOT>/py4mt/modules/ and
<PY4MTX_ROOT>/py4mt/scripts/ are put on sys.path (version.py and util.py
supply the title line printed at start-up).

Optional LaTeX table: set LATEX_FILE in the USER SECTION to the name of a
.tex file. It gets one booktabs row per iteration (elements, regions, n_rho,
n_dist, Jacobian columns and, if mesh.dat was read, the x, y, z range).
LATEX_STANDALONE = True writes a complete document instead of only the
table environment.

Output file names: OUT_FILE in the USER SECTION is a prefix put in front of
the names of the two files the script writes, the parameter summary
(<OUT_FILE><script name>_summary.md, next to the script) and the LaTeX table
(<OUT_FILE><LATEX_FILE>; a directory in LATEX_FILE is kept).

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
    2026-10-28  Claude Sonnet 5.5 (Anthropic)
                Renamed from femtic_summarize_model_cells.py to
                summarize_model.py (the parameter summary and the default
                LATEX_FILE name follow the script name).
                Generalised from tetrahedral to hexahedral meshes, which
                may be deformed. The element label is no longer fixed to
                "tetrahedra": it comes from the type line of mesh.dat
                (TETRA -> tetrahedra, any type containing HEX -> hexahedra,
                otherwise "cell type unknown"). parse_mesh_extent() now also
                returns cell_type and the element count of the mesh, and
                requires integer node ids, so that a mesh.dat that does not
                have an "id x y z" node block gives an n/a message instead
                of a wrong size. A different element count in mesh.dat and
                resistivity_block is reported as a warning. New optional
                CELL_STATS / HEX_NODE_ORDER: parse_mesh_cells() reads the
                element block and gives total and min/median/max cell
                volume (tetrahedra and hexahedra, exact for deformed
                hexahedra) and the regular/deformed classification of
                hexahedra; failed checks (node ids out of range, repeated
                vertices, volumes of mixed sign or zero) are reported
                instead of printing numbers. The TETRA layout follows
                read_femtic_mesh() in femtic.py; the hexahedral layouts are
                ASSUMED (same node block; vertex ids are the last 8 tokens
                of an element line, as for TETRA) and are NOT verified
                against real FEMTIC files. Regular (axis-aligned) and
                deformed hexahedra and the TETRA output were tested on
                synthetic meshes only. Cosmetic: removed a stray "3" from
                a banner line.
    2026-10-28  Claude Sonnet 5.5 (Anthropic)
                Renamed from summarize_model.py to femtic_summarize_model.py
                (the femtic_ prefix of the other scripts in this directory).
                New user parameter OUT_FILE: a prefix for the names of the
                parameter summary .md file and of the LaTeX .tex file (a
                path separator in it is refused). It replaces the hard-coded
                "Ubinas_" in the default LATEX_FILE, which is now the script
                name + ".tex"; OUT_FILE is set to "Ubinas_" so that the
                .tex name keeps its prefix, and the .md file gets it too.
                The parameter summary records OUT_FILE.
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
SIZE_DECIMALS = 0

# Also read the element block of mesh.dat (needs numpy) and print the total
# model volume, the min / median / max cell volume and, for hexahedra,
# whether the cells are regular boxes or deformed. False = only the cell
# type and the bounding box (standard library only, fast on large meshes).
CELL_STATS = False

# Order of the 8 vertex ids of a hexahedron in mesh.dat (only used with
# CELL_STATS). "vtk": 0-1-2-3 around the first face, 4-5-6-7 above/below
# them in the same sense (VTK_HEXAHEDRON). "tensor": lexicographic, node
# index = i + 2*j + 4*k for the corner (i, j, k) in {0, 1}^3 (x fastest).
# Not verified for FEMTIC; if the check "volumes of one sign" fails the
# script says so, then try the other value.
HEX_NODE_ORDER = "vtk"

# Region 1 as ocean: None = infer (flag == 1 and rho <= 1 Ohm.m),
# True = ocean, False = no ocean.
OCEAN = None

# Prefix put in front of the names of the files this script writes: the
# parameter summary (<OUT_FILE><script name>_summary.md, next to the script)
# and the LaTeX table (<OUT_FILE><LATEX_FILE>; a directory part of LATEX_FILE
# is kept, the prefix goes before the file name). "" or None = no prefix.
# It is a file-name prefix only: it must not contain "/" or "\".
OUT_FILE = "Ubinas_"

# Optional LaTeX table of the summaries. None = off; otherwise the name of
# the .tex file to write (a relative path refers to the current directory);
# OUT_FILE is put in front of the file name.
LATEX_FILE = Path(fname).stem + ".tex"

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


def _out_prefix() -> str:
    """OUT_FILE as a string (None = no prefix)."""
    return "" if OUT_FILE is None else str(OUT_FILE)


def _prefixed(path, prefix: str) -> str:
    """*path* with *prefix* put in front of its file name (directory kept)."""
    p = Path(path)
    return str(p.parent / (prefix + p.name))


def _write_param_summary(fname: str, params: dict, prefix: str = "") -> str:
    """Write a Markdown summary of this script's user parameters.

    Self-contained (stdlib only). Writes
    ``<script_dir>/<prefix><script_stem>_summary.md`` with the script path,
    run date/time, and the resolved user parameters.
    """
    from datetime import datetime

    script_path = os.path.abspath(fname)
    stem = os.path.splitext(os.path.basename(fname))[0]
    out_path = os.path.join(os.path.dirname(script_path),
                            prefix + stem + "_summary.md")
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


# Cell types known from the type line of mesh.dat:
#   key      -> (plural label, vertices per element)
_CELL_TYPES = {
    "tetra": ("tetrahedra", 4),
    "hexa":  ("hexahedra",  8),
}


def cell_type_from_header(token: str | None) -> str | None:
    """Cell type key for the type line of ``mesh.dat``.

    ``TETRA`` -> ``"tetra"``; any type containing ``HEX`` (``HEXA`` and
    other hexahedral variants, deformed or not) -> ``"hexa"``. ``None`` for
    a missing or unknown type line.
    """
    if not token:
        return None
    up = token.upper()
    if up.startswith("TET"):
        return "tetra"
    if "HEX" in up:
        return "hexa"
    return None


def cell_label(cell_type: str | None) -> str:
    """Plural name of the cells for the printed summary."""
    if cell_type in _CELL_TYPES:
        return _CELL_TYPES[cell_type][0]
    return "cell type unknown"


def parse_mesh_extent(path: Path) -> dict:
    """Bounding box of the node coordinates in a FEMTIC ``mesh.dat``.

    Layout assumed: an optional mesh-type line (``TETRA``, ``HEXA``, ...),
    a line whose first token is the number of nodes, then one line
    ``id  x  y  z`` per node (metres, z positive down; the id must be an
    integer). The bounding box needs no regular grid, so it is valid for
    deformed hexahedral meshes too. After the node block only the element
    count (first token of the next line) is read, if it is an integer; the
    elements themselves are not read (see :func:`parse_mesh_cells`).

    Returns a dict with ``mesh_type`` (str or None), ``cell_type``
    (``"tetra"``, ``"hexa"`` or None, see :func:`cell_type_from_header`),
    ``nnode`` (int), ``nelem`` (element count stated in the file, int or
    None), ``min`` and ``max`` (3-tuples x, y, z in the units of the file).
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
                int(p[0])
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

        nelem = None
        nel_line = _next_nonblank(fh)
        if nel_line is not None:
            tok = nel_line.split()[0]
            if tok.lstrip("+-").isdigit():
                nelem = int(tok)

    return {
        "mesh_type": mesh_type,
        "cell_type": cell_type_from_header(mesh_type),
        "nnode":     nnode,
        "nelem":     nelem,
        "min":       (xmin, ymin, zmin),
        "max":       (xmax, ymax, zmax),
    }


# ---------------------------------------------------------------------------
# Cell statistics (optional: CELL_STATS, needs numpy)
# ---------------------------------------------------------------------------

# Hexahedron vertex order -> tensor order (index = i + 2*j + 4*k, x fastest).
# VTK vertices 0..7 are the corners (0,0,0) (1,0,0) (1,1,0) (0,1,0)
# (0,0,1) (1,0,1) (1,1,1) (0,1,1). Entry t gives the file vertex that sits at
# tensor corner t.
_HEX_TO_TENSOR = {
    "tensor": (0, 1, 2, 3, 4, 5, 6, 7),
    "vtk":    (0, 1, 3, 2, 4, 5, 7, 6),
}


def tetra_volumes(pts):
    """Signed volumes of tetrahedra; *pts* has shape (n, 4, 3)."""
    import numpy as np

    a = pts[:, 1] - pts[:, 0]
    b = pts[:, 2] - pts[:, 0]
    c = pts[:, 3] - pts[:, 0]
    return np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0


def hexa_volumes(pts, order: str = "vtk"):
    """Signed volumes of hexahedra; *pts* has shape (n, 8, 3).

    Each cell is the trilinear image of the unit cube, so its corners may sit
    anywhere (deformed cells, non-planar faces). The volume is the integral
    of det(J) over the cube. det(J) has degree 2 in each of the three cell
    coordinates, so 2 x 2 x 2 Gauss quadrature gives the volume exactly.
    The sign is positive for a right-handed corner order in the coordinate
    system of the nodes (FEMTIC has z down, so valid cells may be negative);
    only a mixture of signs points to a wrong vertex order.
    """
    import numpy as np

    if order not in _HEX_TO_TENSOR:
        raise ValueError(f"HEX_NODE_ORDER must be 'vtk' or 'tensor', got {order!r}")
    p = pts[:, list(_HEX_TO_TENSOR[order]), :]
    t = np.arange(8)
    s = np.stack([(t & 1) * 2 - 1, ((t >> 1) & 1) * 2 - 1, ((t >> 2) & 1) * 2 - 1],
                 axis=1).astype(float)                  # corner signs, (8, 3)
    g = 1.0 / math.sqrt(3.0)
    vol = np.zeros(p.shape[0])
    for xi in (-g, g):
        for eta in (-g, g):
            for zeta in (-g, g):
                fx = 1.0 + s[:, 0] * xi
                fy = 1.0 + s[:, 1] * eta
                fz = 1.0 + s[:, 2] * zeta
                dn = np.stack([s[:, 0] * fy * fz,
                               s[:, 1] * fx * fz,
                               s[:, 2] * fx * fy], axis=1) / 8.0   # (8, 3)
                jac = np.einsum("nta,tb->nab", p, dn)   # d x_a / d xi_b
                vol += np.linalg.det(jac)
    return vol


def _read_cells(path: Path, ext: dict, hex_order: str) -> dict:
    """Read the element block of *path* and compute the cell statistics.

    *ext* is the result of :func:`parse_mesh_extent` for the same file.
    Layout assumed: after the node block a line with the element count, then
    one line per element whose LAST ``nv`` tokens are the vertex ids
    (``nv`` = 4 for TETRA, 8 for hexahedra); anything before them (element
    id, neighbour ids) is ignored. This is how ``read_femtic_mesh()`` in
    femtic.py reads TETRA; for the hexahedral types it is an assumption.
    Raises ValueError if the file contradicts it.
    """
    import numpy as np

    ctype = ext["cell_type"]
    if ctype is None:
        raise ValueError("cell type unknown (the type line of mesh.dat is "
                         "missing, or neither TETRA nor a hexahedral type)")
    if hex_order not in _HEX_TO_TENSOR:
        raise ValueError(f"HEX_NODE_ORDER must be 'vtk' or 'tensor', got {hex_order!r}")
    nv = _CELL_TYPES[ctype][1]
    nnode = ext["nnode"]

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        _next_nonblank(fh)                  # type line (there is one: cell_type is set)
        _next_nonblank(fh)                  # node count
        ids = np.empty(nnode, dtype=np.int64)
        xyz = np.empty((nnode, 3))
        n = 0
        while n < nnode:
            line = fh.readline()
            if not line:
                raise ValueError(f"end of file after {n} of {nnode} nodes")
            p = line.split()
            if not p:
                continue
            ids[n] = int(p[0])
            xyz[n] = (float(p[1]), float(p[2]), float(p[3]))
            n += 1

        nel_line = _next_nonblank(fh)
        if nel_line is None:
            raise ValueError("no element count after the node block")
        try:
            nelem = int(nel_line.split()[0])
        except ValueError:
            raise ValueError(
                f"cannot read the element count from {nel_line.strip()!r}") from None
        if nelem <= 0:
            raise ValueError(f"element count {nelem} is not positive")

        conn = np.empty((nelem, nv), dtype=np.int64)
        n = 0
        while n < nelem:
            line = fh.readline()
            if not line:
                raise ValueError(f"end of file after {n} of {nelem} elements")
            p = line.split()
            if not p:
                continue
            if len(p) < nv + 1:
                raise ValueError(f"element line has {len(p)} columns, need the "
                                 f"element id and {nv} vertex ids: {line.strip()!r}")
            try:
                conn[n] = [int(tok) for tok in p[-nv:]]
            except ValueError:
                raise ValueError(f"bad element line {line.strip()!r}") from None
            n += 1

    # node ids -> rows of xyz
    if np.array_equal(ids, np.arange(nnode)):
        if conn.min() < 0 or conn.max() >= nnode:
            raise ValueError(f"a vertex id in the element block is outside "
                             f"0..{nnode - 1} (are the vertex ids really the "
                             f"last {nv} columns of an element line?)")
    else:
        order = np.argsort(ids, kind="stable")
        sid = ids[order]
        if np.any(sid[1:] == sid[:-1]):
            raise ValueError("duplicate node ids in the node block")
        pos = np.minimum(np.searchsorted(sid, conn), nnode - 1)
        if np.any(sid[pos] != conn):
            raise ValueError("a vertex id in the element block is not a node id "
                             f"(are the vertex ids really the last {nv} columns "
                             "of an element line?)")
        conn = order[pos]

    warnings: list[str] = []
    srt = np.sort(conn, axis=1)
    n_rep = int(np.count_nonzero((srt[:, 1:] == srt[:, :-1]).any(axis=1)))
    if n_rep:
        warnings.append(f"{n_rep:,d} of {nelem:,d} cells repeat a vertex id "
                        f"(degenerate cells)")

    vol = np.empty(nelem)
    bbv = np.empty(nelem)                   # volume of the bounding box of each cell
    step = 200_000                          # chunks keep the memory use low
    for a in range(0, nelem, step):
        b = min(a + step, nelem)
        pts = xyz[conn[a:b]]                # (m, nv, 3)
        if ctype == "tetra":
            vol[a:b] = tetra_volumes(pts)
        else:
            vol[a:b] = hexa_volumes(pts, hex_order)
        bbv[a:b] = np.prod(pts.max(axis=1) - pts.min(axis=1), axis=1)

    n_deformed = None
    if ctype == "hexa":
        n_pos = int(np.count_nonzero(vol > 0))
        n_neg = int(np.count_nonzero(vol < 0))
        n_tiny = int(np.count_nonzero(np.abs(vol) <= 1.0e-9 * bbv))
        hint = (f"the vertex order is probably not HEX_NODE_ORDER = "
                f"{hex_order!r}; try the other value")
        if n_pos and n_neg:
            raise ValueError(f"hexahedron volumes have both signs "
                             f"({n_pos:,d} positive, {n_neg:,d} negative): {hint}")
        if n_tiny > 0.01 * nelem:
            raise ValueError(f"{n_tiny:,d} of {nelem:,d} hexahedra have "
                             f"(almost) zero volume: {hint}")
        if n_tiny:
            warnings.append(f"{n_tiny:,d} hexahedra have (almost) zero volume")
        # An axis-aligned box fills its bounding box; any shear, rotation or
        # warp of the corners leaves a part of it empty.
        regular = (bbv > 0) & (np.abs(vol) >= (1.0 - 1.0e-6) * bbv)
        n_deformed = int(nelem - np.count_nonzero(regular))

    av = np.abs(vol)
    return {
        "nelem":         nelem,
        "volume_total":  float(av.sum()),
        "volume_min":    float(av.min()),
        "volume_median": float(np.median(av)),
        "volume_max":    float(av.max()),
        "n_deformed":    n_deformed,
        "warnings":      warnings,
    }


def parse_mesh_cells(path: Path, *, hex_order: str = "vtk") -> dict:
    """:func:`parse_mesh_extent` plus the cell statistics of the mesh.

    Returns the dict of :func:`parse_mesh_extent` with two more keys:
    ``cells`` (dict with ``nelem``, ``volume_total``, ``volume_min``,
    ``volume_median``, ``volume_max`` in m^3 -- absolute values --,
    ``n_deformed`` (hexahedra that are not axis-aligned boxes; None for
    tetrahedra) and ``warnings`` (list of str)) and ``cells_msg`` (why
    ``cells`` is None). A failing cell statistic does not lose the bounding
    box; only a failing node block raises ValueError.
    """
    ext = parse_mesh_extent(path)
    ext["cells"] = None
    ext["cells_msg"] = None
    try:
        ext["cells"] = _read_cells(path, ext, hex_order)
    except ImportError:
        ext["cells_msg"] = "numpy is not available"
    except Exception as exc:
        ext["cells_msg"] = str(exc)
    return ext


_MESH_CACHE: dict = {}


def get_mesh_extent(rho_path: Path):
    """Return ``(extent, message)`` for the mesh next to *rho_path*.

    ``extent`` is the dict of :func:`parse_mesh_extent` (of
    :func:`parse_mesh_cells` if CELL_STATS is True) or None. ``message``
    says why it is None (file missing / unreadable); it is None as well
    when MESH_FILE is None (model-size lines switched off). The mesh is
    read once per file and cached.
    """
    if MESH_FILE is None:
        return None, None
    mesh_path = rho_path.parent / MESH_FILE
    key = (str(mesh_path.resolve()), bool(CELL_STATS), HEX_NODE_ORDER)
    if key not in _MESH_CACHE:
        if not mesh_path.is_file():
            _MESH_CACHE[key] = (None, f"{MESH_FILE} not found in {rho_path.parent}")
        else:
            try:
                if CELL_STATS:
                    info = parse_mesh_cells(mesh_path, hex_order=HEX_NODE_ORDER)
                else:
                    info = parse_mesh_extent(mesh_path)
                _MESH_CACHE[key] = (info, None)
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
    note = ["% Generated by " + Path(fname).name + " on " + stamp]
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
    label = cell_label(mesh["cell_type"] if mesh is not None else None)
    print(f"  Total elements  : {rho['nelem']:>10,d}  ({label})")
    print(f"  Total regions   : {rho['nreg']:>10,d}  (resistivity blocks)")
    print(f"  Ocean inferred  : {ocean_str}")
    if mesh is not None:
        lines = size_lines(mesh, SIZE_UNIT, SIZE_DECIMALS)
        print(f"  Model size      : {lines[0]}")
        for ln in lines[1:]:
            print(f"                    {ln}")
        print(f"                    (mesh bounding box incl. air, {mesh['nnode']:,d} nodes;"
              f" z positive down)")
        if mesh.get("nelem") is not None and mesh["nelem"] != rho["nelem"]:
            note = ""
            if mesh["cell_type"] != "tetra":
                note = " (the element block layout of this mesh type is assumed)"
            print(f"  [warn] {MESH_FILE} states {mesh['nelem']:,d} elements, "
                  f"the resistivity block {rho['nelem']:,d}{note}")
        cells = mesh.get("cells")
        if cells is not None:
            u = SIZE_UNIT
            f3 = (1.0e-3 if u == "km" else 1.0) ** 3        # m^3 -> unit^3
            kind = label
            if cells["n_deformed"] is not None:
                if cells["n_deformed"] == 0:
                    kind += ", all axis-aligned boxes"
                else:
                    kind += (f", {cells['n_deformed']:,d} of {cells['nelem']:,d} "
                             f"are not axis-aligned boxes (deformed or rotated)")
            print(f"  Cell type       : {kind}")
            print(f"  Model volume    : {cells['volume_total'] * f3:.4g} {u}^3"
                  f"  (sum of the cell volumes, incl. air)")
            print(f"  Cell volume     : min {cells['volume_min'] * f3:.4g}"
                  f"  median {cells['volume_median'] * f3:.4g}"
                  f"  max {cells['volume_max'] * f3:.4g} {u}^3")
            for w in cells["warnings"]:
                print(f"  [warn] {w}")
        elif mesh.get("cells_msg"):
            print(f"  Cell statistics : n/a ({mesh['cells_msg']})")
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
    if HEX_NODE_ORDER not in _HEX_TO_TENSOR:
        print(f"[error] HEX_NODE_ORDER must be 'vtk' or 'tensor', got "
              f"{HEX_NODE_ORDER!r}", file=sys.stderr)
        sys.exit(2)
    prefix = _out_prefix()
    if "/" in prefix or "\\" in prefix or os.sep in prefix:
        print(f"[error] OUT_FILE is a file-name prefix and must not contain a "
              f"path separator, got {OUT_FILE!r}", file=sys.stderr)
        sys.exit(2)

    args = sys.argv[1:] or list(PATHS)

    _write_param_summary(__file__, {
        "PATHS":          args,
        "MESH_FILE":      MESH_FILE,
        "SIZE_UNIT":      SIZE_UNIT,
        "SIZE_DECIMALS":  SIZE_DECIMALS,
        "CELL_STATS":     CELL_STATS,
        "HEX_NODE_ORDER": HEX_NODE_ORDER,
        "OCEAN":          OCEAN,
        "OUT_FILE":       OUT_FILE,
        "LATEX_FILE":       LATEX_FILE,
        "LATEX_CAPTION":    LATEX_CAPTION,
        "LATEX_LABEL":      LATEX_LABEL,
        "LATEX_STANDALONE": LATEX_STANDALONE,
    }, prefix=prefix)

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
                         _prefixed(LATEX_FILE, prefix))
        else:
            print("[warn] no iteration summarised -- no LaTeX table written",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
