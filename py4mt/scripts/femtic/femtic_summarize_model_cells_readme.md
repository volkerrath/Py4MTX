# femtic_summarize_model_cells.py

Report the Jacobian parameter dimension and the physical size of the model
for one or more FEMTIC inversion iterations, combining free resistivity
regions and free distortion parameters.

---

## Purpose

Quickly inspect the size of the inverse problem for a given iteration
without running a full inversion. For each iteration the script reads:

- `resistivity_block_iterXX.dat` - one rho value per region,
- `distortion_iterXX.dat` - four C-matrix values per site (if present),
- `mesh.dat` - node coordinates, used for the model size (if present),

and reports the composition of the model (air / ocean / other-fixed / free
regions), the number of free distortion parameters, the resulting total
Jacobian column count (`n_rho + n_distortion`), and the physical extent of
the model, e.g. `x = [-100, 100] km`.

The parsing code uses the Python standard library only and does **not**
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
python femtic_summarize_model_cells.py resistivity_block_iter0.dat

# Glob (shell expands)
python femtic_summarize_model_cells.py resistivity_block_iter*.dat

# Scan a directory for all resistivity_block_iter*.dat / distortion_iter*.dat
python femtic_summarize_model_cells.py /path/to/run/

# No arguments -> uses PATHS from the user section (default: ".")
python femtic_summarize_model_cells.py
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
| `SIZE_UNIT` | `"km"` | Unit of the model-size lines, `"km"` or `"m"`. Mesh coordinates are taken to be in metres. Any other value stops the script with an error. |
| `SIZE_DECIMALS` | `3` | Decimals of the model-size values (trailing zeros are dropped). |
| `OCEAN` | `None` | Region 1 as ocean: `None` = infer (`flag == 1` and `rho <= 1 Ohm.m`), `True` = ocean, `False` = no ocean. |

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
- The mesh is read once per directory and cached, so a run with many
  iterations does not re-read it.
- A missing or unreadable mesh is not fatal: the size line shows
  `n/a (<reason>)` and the Jacobian summary is printed as usual.

---

## Programmatic usage

The script's own functions can be imported and called directly (the Py4MTX
environment must be set, see above).

```python
from pathlib import Path
from femtic_summarize_model_cells import (
    parse_resistivity_block, parse_distortion, parse_mesh_extent, size_lines,
)

rho  = parse_resistivity_block(Path("resistivity_block_iter0.dat"))
dist = parse_distortion(Path("distortion_iter0.dat"))
mesh = parse_mesh_extent(Path("mesh.dat"))

n_total = rho["n_rho"] + dist["n_distortion"]
for line in size_lines(mesh, unit="km", decimals=3):
    print(line)
```

- `parse_resistivity_block(path, *, ocean=None)` - pass `ocean=True` or
  `ocean=False` to force the ocean-region interpretation instead of using
  the `flag == 1 AND rho <= 1 Ohm.m` heuristic.
- `parse_mesh_extent(path)` - reads the node block of `mesh.dat` only; the
  elements are not read.
- `get_mesh_extent(rho_path)` - what `main()` uses: finds `MESH_FILE` next to
  a resistivity_block file, caches the result, and returns
  `(extent, None)` or `(None, message)`.
- `size_lines(extent, unit="km", decimals=3)` - the formatted x / y / z lines.

---

## Output format

For each iteration, `print_summary()` prints a fixed-width report. The
example below was produced from a synthetic test run (random mesh,
x and y in [-100, 100] km, z in [-50, 100] km), not from a real inversion:

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

The lower and upper limits are padded to a common width so that the three
lines align.

Special cases:

- No matching `distortion_iter<N>.dat`: the "Distortion params" row prints
  `n/a (no distortion_iterNN.dat)` and `n_total` is just `n_rho`.
- `mesh.dat` not found: `Model size      : n/a (mesh.dat not found in <dir>)`.
- `mesh.dat` unreadable or truncated: `Model size      : n/a (mesh.dat: <reason>)`.
- `MESH_FILE = None`: no model-size lines at all.
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

**`mesh.dat`** (the part that is read):
```
TETRA                            <- optional type line (any non-numeric first token)
nnode
id  x  y  z                      <- nnode lines, metres, z positive down
...                              <- element block, not read
```

The mesh is expected in the same directory as the `resistivity_block` files.
Blank lines are skipped. The TETRA node layout was checked against
`read_femtic_mesh()` in `femtic.py`. The node block of the hexahedral and
non-conforming hexahedral layouts was **not** verified against real files;
the script reads them with the same node-block assumption.

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
| `nnode` | int | Number of nodes |
| `min` | tuple of 3 floats | (xmin, ymin, zmin) in metres |
| `max` | tuple of 3 floats | (xmax, ymax, zmax) in metres |

---

## Parameter summary output

At the start of `main()`, after resolving the arguments, the script writes
`femtic_summarize_model_cells_summary.md` (next to the script itself) via a
small, self-contained `_write_param_summary()` helper (stdlib only). It
records the script's absolute path, the run date/time, and all user
parameters (`PATHS` as resolved, `MESH_FILE`, `SIZE_UNIT`, `SIZE_DECIMALS`,
`OCEAN`). The model size itself is printed to the terminal only; it is not
written to this file.

---

## Dependencies

- Python standard library for all parsing and output (`inspect`, `math`,
  `os`, `re`, `sys`, `pathlib`, and `datetime` inside the summary helper).
- Py4MTX header: the environment variables `PY4MTX_ROOT` and `PY4MTX_DATA`,
  and the Py4MTX modules `version` and `util` (title line only).

---

## Provenance

| Date | Author | Note |
|---|---|---|
| 2026-06-08 | Claude Sonnet 4.6 (Anthropic) | Created (standalone). |
| 2026-08-13 | Claude Sonnet 5 (Anthropic) | Added `femtic_summarize_model_cells_summary.md` output after CLI argument parsing: writes the resolved `paths` argument, script path, and run date/time via a self-contained, stdlib-only helper. |
| 2026-08-13 | Claude Sonnet 5 (Anthropic) | Rewrote this README to match the script's actual implementation: documented the real `sys.argv`-based CLI, the real `parse_resistivity_block()` / `parse_distortion()` return dicts, and the real fixed-width `print_summary()` output. |
| 2026-10-24 | Claude Sonnet 5.5 (Anthropic) | Added the physical model size (x, y, z range from `mesh.dat`, km by default) to the per-iteration summary via `parse_mesh_extent()`, `get_mesh_extent()` and `size_lines()`. Added a USER SECTION (`PATHS`, `MESH_FILE`, `SIZE_UNIT`, `SIZE_DECIMALS`, `OCEAN`); the ocean override is now reachable from there. The parameter summary file records all of them. README rewritten in plain ASCII. Tested on synthetic meshes only (TETRA, header-less, HEXA header, missing, truncated; km and m). |
| 2026-10-24 | Claude Sonnet 5.5 (Anthropic) | Added the usual Py4MTX header (`PY4MTX_DATA` / `PY4MTX_ROOT` from the environment, `py4mt/modules` and `py4mt/scripts` on `sys.path`, `version` / `util` title line). Parsing code unchanged. Tested with stub `version` / `util` modules, not the real ones. |

AI-generated documentation (Claude, Anthropic; generated 2026-10-24). Review
before production use.
