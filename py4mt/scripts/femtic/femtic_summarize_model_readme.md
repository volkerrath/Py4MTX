# femtic_summarize_model.py

Report the Jacobian parameter dimension and the physical size of the model
for one or more FEMTIC inversion iterations, combining free resistivity
regions and free distortion parameters. Works for tetrahedral and for
hexahedral meshes; the hexahedra may be deformed.

Renamed from `femtic_summarize_model_cells.py` on 2026-10-28.

---

## Purpose

Quickly inspect the size of the inverse problem for a given iteration
without running a full inversion. For each iteration the script reads:

- `resistivity_block_iterXX.dat` - one rho value per region,
- `distortion_iterXX.dat` - four C-matrix values per site (if present),
- `mesh.dat` - node coordinates, used for the model size and the cell type
  (if present); with `CELL_STATS = True` also the element block,

and reports the composition of the model (air / ocean / other-fixed / free
regions), the number of free distortion parameters, the resulting total
Jacobian column count (`n_rho + n_distortion`), the physical extent of the
model, e.g. `x = [-100, 100] km`, and the cell type (tetrahedra or
hexahedra). With `CELL_STATS = True` it also reports the model volume, the
smallest / median / largest cell volume and, for hexahedra, whether the cells
are axis-aligned boxes or deformed.

The parsing code uses the Python standard library only (numpy is needed
only for `CELL_STATS = True`, and is imported at that point) and does **not**
import or delegate to `femtic.py`. The script does start with the usual
Py4MTX header (see "Py4MTX environment" below), which needs the environment
variables `PY4MTX_ROOT` and `PY4MTX_DATA`.

---

## Usage

All user parameters are in the `USER SECTION` at the top of the script
(see below). Command-line arguments, if given, replace `PATHS`.

```bash
# Single resistivity_block file (its paired distortion file, if any,
# is picked up automatically from the same directory)
python femtic_summarize_model.py resistivity_block_iter0.dat

# Glob (shell expands)
python femtic_summarize_model.py resistivity_block_iter*.dat

# Scan a directory for all resistivity_block_iter*.dat / distortion_iter*.dat
python femtic_summarize_model.py /path/to/run/

# No arguments -> uses PATHS from the user section (default: ".")
python femtic_summarize_model.py
```

Arguments are read directly from `sys.argv[1:]` (no `argparse`, no flags).
Each argument may be a directory (scanned for `resistivity_block_iter*.dat`
and `distortion_iter*.dat`) or an individual file path; multiple arguments
may be mixed freely. Files are paired by the iteration number embedded in
their name (`iter<N>`, e.g. `resistivity_block_iter3.dat` pairs with
`distortion_iter3.dat`).

### User section

| Parameter | Default | Meaning |
|---|---|---|
| `PATHS` | `["."]` | Directories and/or files to scan. Replaced by command-line arguments. |
| `MESH_FILE` | `"mesh.dat"` | Mesh file read for the model size. Looked up in the directory of each `resistivity_block` file (an absolute path is used as it is). `None` = do not print the model-size lines. |
| `SIZE_UNIT` | `"km"` | Unit of the model-size lines, `"km"` or `"m"`. Mesh coordinates are taken to be in metres. Volumes are printed in this unit cubed. Any other value stops the script with an error. |
| `SIZE_DECIMALS` | `0` | Decimals of the model-size values (trailing zeros are dropped). |
| `CELL_STATS` | `False` | `True` = also read the element block and print model volume, cell volumes and the regular/deformed classification (needs numpy). `False` = only cell type and bounding box (standard library, fast: 0.5 s for a 500,000-cell mesh). |
| `HEX_NODE_ORDER` | `"vtk"` | Order of the 8 vertex ids of a hexahedron in `mesh.dat`; only used with `CELL_STATS`. `"vtk"`: 0-1-2-3 around one face, 4-5-6-7 above/below them in the same sense (VTK_HEXAHEDRON). `"tensor"`: index = i + 2 j + 4 k for the corner (i, j, k), x fastest. Any other value stops the script with an error. |
| `OCEAN` | `None` | Region 1 as ocean: `None` = infer (`flag == 1` and `rho <= 1 Ohm.m`), `True` = ocean, `False` = no ocean. |
| `OUT_FILE` | `"Ubinas_"` | Prefix put in front of the names of the two files the script writes: the parameter summary (`<OUT_FILE>femtic_summarize_model_summary.md`, next to the script) and the LaTeX table (`<OUT_FILE><LATEX_FILE>`). For a `LATEX_FILE` with a directory part the prefix goes before the file name only (`sub/model.tex` with `"run7_"` gives `sub/run7_model.tex`). `""` or `None` = no prefix. A `/` or `\` in it stops the script with an error. The default is the prefix of the script as delivered (the earlier default `LATEX_FILE` had it hard-coded). |
| `LATEX_FILE` | `Path(fname).stem + ".tex"` | Name of a `.tex` file to write the summaries to as a LaTeX table (relative paths refer to the current directory); `OUT_FILE` is put in front of the file name. The default is the script name, so with the default `OUT_FILE` the file is `Ubinas_femtic_summarize_model.tex`. `None` = no LaTeX output. |
| `LATEX_CAPTION` | `None` | Caption of the table, used as it is (write LaTeX). `None` = default caption. |
| `LATEX_LABEL` | `None` | Label of the table, e.g. `"tab:model_cells"`. `None` = no `\label`. |
| `LATEX_STANDALONE` | `False` | `False` = only the table environment (for `\input`); `True` = a complete document that compiles as it is. |

---

## Py4MTX environment

At start-up the script runs the usual Py4MTX header, before the USER SECTION:

```python
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
```

- Both environment variables must be set. If one is missing the script stops
  at once with `KeyError: 'PY4MTX_ROOT'` (or `'PY4MTX_DATA'`).
  `PY4MTX_DATA` belongs to the standard header and is not used by this script.
- `version.py` and `util.py` must be in `py4mt/modules/` or `py4mt/scripts/`;
  they only supply the title line.
- `femtic.py` is not imported.
- Importing the module (see "Programmatic usage") runs the header as well, so
  it needs the same environment and prints the title line.

---

## Cell type

The cell type comes from the type line of `mesh.dat` (its first line):

| Type line | Cell type | Label in the summary | Vertices per element |
|---|---|---|---|
| `TETRA` | tetra | `tetrahedra` | 4 |
| any type containing `HEX` (`HEXA`, and other hexahedral variants, deformed or not) | hexa | `hexahedra` | 8 |
| missing, or anything else | unknown | `cell type unknown` | - |

The label appears after "Total elements". Before 2026-10-28 it was always
`(tetrahedra)`; with no mesh file it is now `(cell type unknown)`.

---

## Model size

The model size is the bounding box of all node coordinates in `mesh.dat`,
one range per axis, plus the extent (max - min):

```
x = [-100, 100] km   (size 200 km)
```

Notes:

- The box includes the air layer, so the z range starts above the surface.
- z is positive down (FEMTIC convention), so air nodes have negative z.
- Coordinates are assumed to be in metres; `SIZE_UNIT = "km"` divides by 1000.
- Only the nodes are used, never grid spacings, so the box is valid for
  deformed hexahedral meshes as well. For a sheared or topography-following
  mesh it is larger than the nominal model: a shear of 0.25 over z in
  [-50, 100] km widens x from [-100, 100] to [-112, 125] km.
- The mesh is read once per directory and cached, so a run with many
  iterations does not re-read it.
- A missing or unreadable mesh is not fatal: the size line shows
  `n/a (<reason>)` and the Jacobian summary is printed as usual.
- After the node block, the element count of `mesh.dat` (first token of the
  next line, if it is an integer) is compared with the element count of the
  resistivity block. A difference prints `[warn] mesh.dat states N elements,
  the resistivity block M`. For the hexahedral types the message adds that
  the element block layout is assumed.

---

## Cell statistics (CELL_STATS = True)

With `CELL_STATS = True` the element block of `mesh.dat` is read as well
(numpy needed) and three lines are added:

```
  Cell type       : hexahedra, 48 of 48 are not axis-aligned boxes (deformed or rotated)
  Model volume    : 5.94e+06 km^3  (sum of the cell volumes, incl. air)
  Cell volume     : min 1.215e+05  median 1.237e+05  max 1.26e+05 km^3
```

- **Volumes.** Tetrahedra: the triple product / 6. Hexahedra: each cell is
  the trilinear image of the unit cube, so the corners may sit anywhere, and
  the volume is the integral of det(J) over the cube, computed with
  2 x 2 x 2 Gauss quadrature. That rule is exact for it (det(J) has degree 2
  in each cell coordinate), also for cells with non-planar faces. Absolute
  values are summed; the volume includes the air cells.
- **Regular or deformed.** An axis-aligned box fills its bounding box; any
  shear, rotation or warp of the corners does not. A hexahedron counts as
  "not an axis-aligned box" when its volume is less than
  `(1 - 1e-6)` times the volume of its bounding box. The line says
  `all axis-aligned boxes` if no cell qualifies. A rotated but otherwise
  regular grid is also reported as "deformed or rotated".
- **Vertex order.** `HEX_NODE_ORDER` tells the script which corner is which.
  The check: the signed volumes of all hexahedra must have one sign (the sign
  itself depends on the handedness of the coordinate system, and FEMTIC has
  z down, so either is fine), and not more than 1 % of the cells may have
  (almost) zero volume. If this fails, the cell statistics line reads
  `Cell statistics : n/a (... try the other value)` and the bounding box is
  still printed. Fewer than 1 % degenerate cells only give a `[warn]` line.
- **Other checks** (each gives `Cell statistics : n/a (<reason>)`): element
  block shorter than stated, a vertex id that is not a node id, an element
  line with too few columns, an unknown cell type. Cells that repeat a vertex
  id give a `[warn]` line.
- **Node ids** may be 0-based or 1-based or any other unique integers; they
  are mapped to node rows.
- **Memory and time.** The element block is processed in chunks of 200,000
  cells. A synthetic mesh of 500,000 deformed hexahedra took 4 s (without
  `CELL_STATS`: 0.5 s).

---

## LaTeX table

Set `LATEX_FILE` in the user section (e.g. `LATEX_FILE = "model_cells.tex"`)
and the summaries are also written as one LaTeX table, after all iterations
have been processed. The printed summary does not change.

- One row per iteration: iteration number, elements, regions, `n_rho`,
  `n_dist`, Jacobian columns (`n_rho + n_dist`).
- If `mesh.dat` was read for at least one iteration, three more columns give
  the x, y, z range in `SIZE_UNIT` (same values as the printed summary).
  Without any mesh these columns are left out.
- `--` marks a missing value (no distortion file, or no mesh for that
  iteration). In that case `n_dist` is `--` and the Jacobian columns are
  `n_rho` alone, as in the printed summary.
- Thousands are separated by a thin space (`254\,016`), so no number package
  is needed. The table uses `booktabs`; with `LATEX_STANDALONE = False` the
  file has only the `table` environment (add `\usepackage{booktabs}` to your
  paper and `\input{model_cells.tex}`), with `True` it is a complete
  `article` document.
- `LATEX_CAPTION` and `LATEX_LABEL` are used exactly as given, so write them
  as LaTeX. The default caption explains the columns.
- With nine columns the table is too wide for one column of a two-column
  layout; use `table*` or wrap the `tabular` in `\resizebox{\columnwidth}{!}{...}`.
- The cell statistics are not part of the table.
- The file name gets the `OUT_FILE` prefix: `LATEX_FILE = "model_cells.tex"`
  with `OUT_FILE = "Ubinas_"` is written as `Ubinas_model_cells.tex`.
- The first comment line of the file names the script that wrote it.
- If the file cannot be written, an error is printed and the run is not
  stopped.

Example of the generated `tabular` (synthetic test run, not a real inversion;
iteration 1 has no distortion file):

```
\begin{tabular}{rrrrrrccc}
  \toprule
  Iter. & Elements & Regions & $n_\rho$ & $n_\mathrm{dist}$ & Jac.\ columns & $x$ (km) & $y$ (km) & $z$ (km) \\
  \midrule
  0 & 25 & 6 & 4 & 12 & 16 & $[-100,\,100]$ & $[-100,\,100]$ & $[-50,\,100]$ \\
  1 & 25 & 6 & 4 & -- & 4 & $[-100,\,100]$ & $[-100,\,100]$ & $[-50,\,100]$ \\
  \bottomrule
\end{tabular}
```

---

## Programmatic usage

The script's own functions can be imported and called directly (the Py4MTX
environment must be set, see above).

```python
from pathlib import Path
from femtic_summarize_model import (
    parse_resistivity_block, parse_distortion, parse_mesh_extent,
    parse_mesh_cells, size_lines,
)

rho  = parse_resistivity_block(Path("resistivity_block_iter0.dat"))
dist = parse_distortion(Path("distortion_iter0.dat"))
mesh = parse_mesh_cells(Path("mesh.dat"), hex_order="vtk")   # or parse_mesh_extent

n_total = rho["n_rho"] + dist["n_distortion"]
for line in size_lines(mesh, unit="km", decimals=3):
    print(line)
if mesh["cells"] is not None:
    print(mesh["cells"]["volume_total"])        # m^3
else:
    print(mesh["cells_msg"])
```

- `parse_resistivity_block(path, *, ocean=None)` - pass `ocean=True` or
  `ocean=False` to force the ocean-region interpretation instead of using
  the `flag == 1 AND rho <= 1 Ohm.m` heuristic.
- `parse_mesh_extent(path)` - reads the node block of `mesh.dat` and the
  element count; the elements are not read.
- `parse_mesh_cells(path, *, hex_order="vtk")` - `parse_mesh_extent()` plus
  the cell statistics under the keys `cells` and `cells_msg` (numpy needed).
- `cell_type_from_header(token)` / `cell_label(cell_type)` - the type line to
  `"tetra"` / `"hexa"` / `None`, and the label printed in the summary.
- `tetra_volumes(pts)` / `hexa_volumes(pts, order="vtk")` - signed volumes for
  corner arrays of shape (n, 4, 3) and (n, 8, 3).
- `get_mesh_extent(rho_path)` - what `main()` uses: finds `MESH_FILE` next to
  a resistivity_block file, caches the result, and returns
  `(extent, None)` or `(None, message)`; the extent comes from
  `parse_mesh_cells()` if `CELL_STATS` is True.
- `size_lines(extent, unit="km", decimals=3)` - the formatted x / y / z lines.
- `latex_table(rows, unit="km", decimals=3, caption=None, label=None, standalone=False)` -
  the LaTeX source as a string. `rows` is a list of dicts with the keys `it`,
  `nelem`, `nreg`, `n_rho`, `n_dist` (`None` if there is no distortion file),
  `n_total` and `mesh` (a `parse_mesh_extent()` dict or `None`).

---

## Output format

For each iteration, `print_summary()` prints a fixed-width report. The first
example was produced from a synthetic test run (random mesh, x and y in
[-100, 100] km, z in [-50, 100] km), not from a real inversion:

```
  ----------------------------------------------------
  Iteration       : 0
  Total elements  :         25  (tetrahedra)
  Total regions   :          6  (resistivity blocks)
  Ocean inferred  : yes  (rho = 0.25 Ohm.m)
  Model size      : x = [-100,  100] km   (size 200 km)
                    y = [-100,  100] km   (size 200 km)
                    z = [ -50,  100] km   (size 150 km)
                    (mesh bounding box incl. air, 68 nodes; z positive down)
  ----------------------------------------------------
  Air regions                        1
  Ocean regions                      1
  Other fixed regions                0
  Free rho regions                   4  <- n_rho
  Free sites (distort.)              3  x 4 values
  Distortion params                 12  <- n_dist
  ====================================================
  Jacobian columns                  16  = n_rho + n_dist
  ----------------------------------------------------

  1 iteration(s) processed.
```

A synthetic deformed hexahedral mesh (4 x 4 x 3 cells, sheared and warped;
`SIZE_DECIMALS = 0`, `CELL_STATS = True`):

```
  ----------------------------------------------------
  Iteration       : 0
  Total elements  :         48  (hexahedra)
  Total regions   :          6  (resistivity blocks)
  Ocean inferred  : yes  (rho = 0.25 Ohm.m)
  Model size      : x = [-112,  125] km   (size 238 km)
                    y = [-103,  103] km   (size 206 km)
                    z = [ -52,  100] km   (size 152 km)
                    (mesh bounding box incl. air, 100 nodes; z positive down)
  Cell type       : hexahedra, 48 of 48 are not axis-aligned boxes (deformed or rotated)
  Model volume    : 5.94e+06 km^3  (sum of the cell volumes, incl. air)
  Cell volume     : min 1.215e+05  median 1.237e+05  max 1.26e+05 km^3
  ----------------------------------------------------
  Air regions                        1
  ...
```

The lower and upper limits are padded to a common width so that the three
lines align.

Special cases:

- No matching `distortion_iter<N>.dat`: the "Distortion params" row prints
  `n/a (no distortion_iterNN.dat)` and `n_total` is just `n_rho`.
- `mesh.dat` not found: `Model size      : n/a (mesh.dat not found in <dir>)`,
  and the element label is `(cell type unknown)`.
- `mesh.dat` unreadable or truncated: `Model size      : n/a (mesh.dat: <reason>)`.
  This includes a node block that is not `id x y z` with an integer id.
- `MESH_FILE = None`: no model-size lines at all.
- `CELL_STATS = True` but the cells cannot be evaluated: the bounding box is
  printed and `Cell statistics : n/a (<reason>)` follows it.
- Parse errors in the resistivity or distortion file of one iteration are
  caught, reported to stderr as `[error] iter <N>: ...`, and do not abort
  the run; the remaining iterations are still processed.

---

## Region conventions

| Region | Category | Condition |
|---|---|---|
| 0 | Air | Always treated as fixed |
| 1 | Ocean | Fixed when `flag == 1` **and** `rho <= 1 Ohm.m` (heuristic, region 1 only; override with `OCEAN`) |
| 2 ... | Other fixed | `flag == 1` |
| 2 ... | Free (`n_rho`) | `flag == 0` |

---

## File formats expected

**`resistivity_block_iterXX.dat`:**
```
nelem  nreg
ielem  iregion                   <- nelem lines, element -> region index (skipped)
ireg  rho  lo  hi  n  flag       <- nreg lines; only columns 0 (ireg), 1 (rho), 5 (flag) are read
```

**`distortion_iterXX.dat`:**
```
nsites
id  d1  d2  d3  d4  flag         <- one line per site; only column 5 (flag) is read
                                    flag == 0 -> free site (contributes 4 distortion params)
```

**`mesh.dat`:**
```
TETRA | HEXA | ...               <- optional type line (any non-numeric first token)
nnode
id  x  y  z                      <- nnode lines, metres, z positive down; id an integer
nelem                            <- element count (read; compared with the resistivity block)
...                              <- nelem element lines
```

Element lines (only read with `CELL_STATS = True`): the LAST 4 (TETRA) or
8 (hexahedral) tokens are the vertex ids; whatever precedes them (element id,
neighbour ids) is ignored.

What is verified and what is not:

- **TETRA** - the node block, the element count line and the "vertex ids are
  the last 4 tokens" rule follow `read_femtic_mesh()` in `femtic.py`
  (written by `write_femtic_mesh()` as `ie -1 -1 -1 -1 n1 n2 n3 n4`).
- **Hexahedral types** - NOT verified against real FEMTIC files. The script
  assumes the same node block, an element count line after it, and the
  vertex ids in the last 8 tokens of an element line. FEMTIC lists three
  mesh types (tetrahedral, hexahedral brick, non-conforming deformed
  hexahedral); the file layout of the hexahedral ones could not be checked
  here. The node-block checks (integer ids, 4 columns) make a different
  layout give `n/a` instead of a wrong size, and the cell statistics check
  vertex ids, vertex order and the element count. A real hexahedral
  `mesh.dat` should be run once with `CELL_STATS = True` and the result
  compared with the known model size and cell sizes.
- **Vertex order of hexahedra** - `"vtk"` and `"tensor"` are the two
  conventions offered; which one FEMTIC uses is not known to this script.
  Other orders are reported as a failed check, not guessed.

The mesh is expected in the same directory as the `resistivity_block` files.
Blank lines are skipped.

---

## Return dict keys

**`parse_resistivity_block()`:**

| Key | Type | Description |
|---|---|---|
| `nelem` | int | Total number of mesh elements |
| `nreg` | int | Number of regions |
| `n_air` | int | Always `1` (region 0) |
| `n_ocean` | int | `1` if an ocean region was identified, else `0` |
| `n_other_fixed` | int | Count of other fixed (`flag == 1`) regions, excluding air/ocean |
| `n_rho` | int | Count of free (`flag == 0`) regions - the resistivity contribution to the Jacobian column count |
| `ocean_present` | bool | Whether region 1 was treated as ocean |
| `ocean_rho` | float or None | Resistivity of region 1 (Ohm.m) if ocean, else `None` |

**`parse_distortion()`:**

| Key | Type | Description |
|---|---|---|
| `nsites` | int | Total number of sites (first line of the file) |
| `n_free_sites` | int | Sites with `flag == 0` |
| `n_fixed_sites` | int | Sites with `flag != 0` |
| `n_distortion` | int | `n_free_sites * 4` - the distortion contribution to the Jacobian column count |

**`parse_mesh_extent()`:**

| Key | Type | Description |
|---|---|---|
| `mesh_type` | str or None | Type line of the file (e.g. `TETRA`), `None` if the file has none |
| `cell_type` | str or None | `"tetra"`, `"hexa"`, or `None` (see "Cell type") |
| `nnode` | int | Number of nodes |
| `nelem` | int or None | Element count stated in the file after the node block, `None` if that line is not an integer |
| `min` | tuple of 3 floats | (xmin, ymin, zmin) in metres |
| `max` | tuple of 3 floats | (xmax, ymax, zmax) in metres |

**`parse_mesh_cells()`:** the keys of `parse_mesh_extent()` plus

| Key | Type | Description |
|---|---|---|
| `cells` | dict or None | `None` if the statistics could not be computed |
| `cells_msg` | str or None | Why `cells` is `None` |

and `cells` has

| Key | Type | Description |
|---|---|---|
| `nelem` | int | Number of elements read |
| `volume_total` | float | Sum of the absolute cell volumes, m^3 |
| `volume_min`, `volume_median`, `volume_max` | float | Cell volumes, m^3 |
| `n_deformed` | int or None | Hexahedra that are not axis-aligned boxes; `None` for tetrahedra |
| `warnings` | list of str | Degenerate cells (repeated vertex id, almost zero volume) |

---

## Parameter summary output

At the start of `main()`, after resolving the arguments, the script writes
`<OUT_FILE>femtic_summarize_model_summary.md` (next to the script itself; with
the default `OUT_FILE` that is `Ubinas_femtic_summarize_model_summary.md`; the
name follows the script name) via a small, self-contained
`_write_param_summary()` helper (stdlib only). It records the script's
absolute path, the run date/time, and all user parameters (`PATHS` as
resolved, `MESH_FILE`, `SIZE_UNIT`, `SIZE_DECIMALS`, `CELL_STATS`,
`HEX_NODE_ORDER`, `OCEAN`, `OUT_FILE`, `LATEX_FILE`, `LATEX_CAPTION`,
`LATEX_LABEL`, `LATEX_STANDALONE`). The model size and the cell statistics
are printed to the terminal only; they are not written to this file.

---

## Dependencies

- Python standard library for all parsing and output (`inspect`, `math`,
  `os`, `re`, `sys`, `pathlib`, and `datetime` inside the summary helper and the LaTeX builder).
- numpy, only for `CELL_STATS = True` (imported when the cell statistics are
  computed; without numpy the run continues with
  `Cell statistics : n/a (numpy is not available)`).
- To compile the LaTeX output: any LaTeX engine and the `booktabs` package.
- Py4MTX header: the environment variables `PY4MTX_ROOT` and `PY4MTX_DATA`,
  and the Py4MTX modules `version` and `util` (title line only).

---

## Tests

Run on synthetic meshes only (4 x 4 x 3 cells, 48 hexahedra or 288
tetrahedra, plus one of 500,000 hexahedra for timing), with stub `version` /
`util` modules (the real `util.py` needs h5py, which was not available), not
on a real FEMTIC run. Checked:

- TETRA output identical to the previous script (apart from the file names
  that follow the rename); the volume of the tetrahedra equals the box volume.
- Regular hexahedra: volume equals the box volume, no cell classified as
  deformed.
- Deformed hexahedra: a pure shear keeps the volume; an affine map gives
  |det A| times the box volume; a nonlinear warp (non-planar faces) agrees
  cell by cell with an independent calculation (parametric subdivision into
  tetrahedra, max relative difference 7e-8); a single twisted cell gives the
  analytic value 1.15 of its mean height.
- Vertex order: a tensor-ordered file is read correctly with
  `HEX_NODE_ORDER = "tensor"` and reported as a failed check with `"vtk"`
  (and the other way round); a mirrored mesh (opposite handedness) gives the
  same volumes; 1-based node ids work.
- Error paths: element count mismatch, no mesh, no type line, a layout that
  is not a node block, a truncated element block, a vertex id out of range,
  a wrong `HEX_NODE_ORDER` value, and the LaTeX and parameter summary files.
- `OUT_FILE`: the default (`Ubinas_`), an empty string and `None` (no prefix),
  a prefix on a `LATEX_FILE` with a directory part, `LATEX_FILE = None`
  (no .tex, the .md is still written), and a `/` or `\` in the prefix (exit
  code 2, nothing written). 49 checks in all, 40 of them for the hexahedral
  generalisation above.

---

## Provenance

| Date | Author | Note |
|---|---|---|
| 2026-06-08 | Claude Sonnet 4.6 (Anthropic) | Created (standalone). |
| 2026-08-13 | Claude Sonnet 5 (Anthropic) | Added `femtic_summarize_model_cells_summary.md` output after CLI argument parsing: writes the resolved `paths` argument, script path, and run date/time via a self-contained, stdlib-only helper. |
| 2026-08-13 | Claude Sonnet 5 (Anthropic) | Rewrote this README to match the script's actual implementation: documented the real `sys.argv`-based CLI, the real `parse_resistivity_block()` / `parse_distortion()` return dicts, and the real fixed-width `print_summary()` output. |
| 2026-10-24 | Claude Sonnet 5.5 (Anthropic) | Added the physical model size (x, y, z range from `mesh.dat`, km by default) to the per-iteration summary via `parse_mesh_extent()`, `get_mesh_extent()` and `size_lines()`. Added a USER SECTION (`PATHS`, `MESH_FILE`, `SIZE_UNIT`, `SIZE_DECIMALS`, `OCEAN`); the ocean override is now reachable from there. The parameter summary file records all of them. README rewritten in plain ASCII. Tested on synthetic meshes only (TETRA, header-less, HEXA header, missing, truncated; km and m). |
| 2026-10-24 | Claude Sonnet 5.5 (Anthropic) | Added the usual Py4MTX header (`PY4MTX_DATA` / `PY4MTX_ROOT` from the environment, `py4mt/modules` and `py4mt/scripts` on `sys.path`, `version` / `util` title line). Parsing code unchanged. Tested with stub `version` / `util` modules, not the real ones. |
| 2026-10-24 | Claude Sonnet 5.5 (Anthropic) | Added the optional LaTeX table: user parameters `LATEX_FILE`, `LATEX_CAPTION`, `LATEX_LABEL`, `LATEX_STANDALONE` and the builder `latex_table()`; one row per iteration, with the x / y / z range columns when a mesh was read. Compiled with pdflatex (standalone and `\input` use); printed output unchanged. |
| 2026-10-28 | Claude Sonnet 5.5 (Anthropic) | Renamed from `femtic_summarize_model_cells.py` to `summarize_model.py` (and this readme from `femtic_summarize_model_cells_readme.md`); the parameter summary file and the default `LATEX_FILE` name follow the script name. Generalised to hexahedral meshes, which may be deformed: element label from the type line of `mesh.dat` (`tetrahedra` / `hexahedra` / `cell type unknown`); `parse_mesh_extent()` returns `cell_type` and `nelem` and requires integer node ids; element count of `mesh.dat` compared with the resistivity block; new optional `CELL_STATS` / `HEX_NODE_ORDER` and `parse_mesh_cells()` (model volume, min / median / max cell volume, exact for deformed hexahedra, and the regular/deformed classification) with checks that report instead of printing wrong numbers. The hexahedral file layouts and the vertex order are ASSUMED and not verified against real FEMTIC files. README: default of `SIZE_DECIMALS` corrected to 0 and of `LATEX_FILE` to the value in the script (both differed from the script), new sections "Cell type", "Cell statistics" and "Tests". Tested on synthetic meshes only. |
| 2026-10-28 | Claude Sonnet 5.5 (Anthropic) | Renamed `summarize_model.py` to `femtic_summarize_model.py` (and this readme to `femtic_summarize_model_readme.md`), the `femtic_` prefix of the other scripts in the directory. New user parameter `OUT_FILE`: prefix for the names of the parameter summary `.md` and the LaTeX `.tex` file (a path separator is refused); the default `LATEX_FILE` is now the script name + `.tex` without the hard-coded `Ubinas_`, which moved into `OUT_FILE`, so the `.tex` name keeps it and the `.md` file gets it as well. `OUT_FILE` is recorded in the parameter summary. Tested on synthetic data only. |

AI-generated documentation (Claude, Anthropic; generated 2026-10-28). Review
before production use.
