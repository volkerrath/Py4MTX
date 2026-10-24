# femtic_summarize_observe_dat.py

Summarise the data content of one or more FEMTIC `observe.dat` files:
number of sites, frequencies per site, data values per frequency, and
overall data-vector size, broken down by observation type (MT, VTF, PT).

---

## Purpose

Quickly inspect the size and structure of an observation file without
running a full inversion. Useful for:

- confirming the data-vector dimension before setting up an inversion,
- comparing site/frequency counts between original and perturbed ensemble members,
- checking which observation types are present in a file.

The file is parsed by `femtic.summarise_observe_dat()` when `femtic.py` is
importable (normally the case once the Py4MTX paths are set). The table is
always printed by the script itself, in plain ASCII. If `femtic.py` cannot
be imported, the script falls back to a self-contained minimal parser.

---

## Usage

All user parameters are in the `USER SECTION` at the top of the script
(see below). Command-line paths, if given, replace `PATHS`.

```bash
# Single file
python femtic_summarize_observe_dat.py observe.dat

# Glob across ensemble members
python femtic_summarize_observe_dat.py ensemble_*/observe.dat

# Scan a directory for observe.dat
python femtic_summarize_observe_dat.py /path/to/run/

# No arguments -> uses PATHS from the user section (default: ".")
python femtic_summarize_observe_dat.py
```

When scanning a directory the script looks for a file literally named
`observe.dat`. To match other names use an explicit glob pattern.

### User section

| Parameter | Default | Meaning |
|---|---|---|
| `PATHS` | `["."]` | `observe.dat` files, glob patterns, or directories to scan. Replaced by command-line arguments. |

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
- `femtic.py` is imported inside a `try` block: with the paths set it is
  found, and its parser is used. `femtic.py` itself imports `ensembles.py`
  when it is loaded, so `ensembles.py` (and its dependencies) must be
  importable too.
- If the import fails for any reason, the script continues with the built-in
  fallback parser and says why on stderr, e.g.
  `[info] femtic.py not importable (No module named 'femtic') -- using built-in fallback parser`.

### Fallback parser

The fallback gives the same table for plain files, but it only recognises
station lines with exactly 4 tokens. Station lines with the owner-element
and/or E-type columns (5 or 6 tokens) are skipped by the fallback and the
counts will then be wrong or empty. Use the `femtic.py` route for such files.

---

## Programmatic usage (via `femtic.py`)

```python
import femtic as fem

# From a path - parses the file (fast: no MT-derived quantities computed)
s = fem.summarise_observe_dat("observe.dat", out=False)

# From an already-parsed dict (zero extra I/O cost)
parsed = fem.read_observe_dat("observe.dat")
s = fem.summarise_observe_dat(parsed, out=False)

# Access results
print(s["n_data_total"])     # total data-vector length
print(s["n_sites_total"])    # total number of sites across all blocks

for b in s["blocks"]:
    print(b["obs_type"], b["n_sites"], b["n_data_total"])
```

`fem._print_observe_summary(s)` (and `out=True`) print a table with Unicode
box-drawing characters. This script does not use it; it prints its own ASCII
table (`_print_summ()`) from the same summary dict.

---

## Output format

The example below was produced from a synthetic `observe.dat` (3 MT sites with
4, 4 and 3 frequencies; 3 VTF stations with 4, 4 and 0 frequencies), not from a
real survey:

```
  ----------------------------------------------------
  File          : observe.dat
  ----------------------------------------------------
  Obs type     Sites  d/freq   Freq tot   Data tot
  ----------------------------------------------------
  MT               3       8         11         88
              nfreq range: 3-4
  VTF              3       4          8         32
              nfreq range: 0-4
  ====================================================
  Total            6                 19        120
  ----------------------------------------------------
  Dummy sites   : VTF 1 (0 frequencies)

  1/1 file(s) processed.
```

If sites within a block have different numbers of frequencies, an additional
line shows the range (`nfreq_min-nfreq_max`) under that block row. If there
are dummy stations (0 frequencies), a final line lists them per type.

Column meanings:

| Column | Description |
|---|---|
| Obs type | Observation type: `MT`, `VTF`, or `PT` |
| Sites | Number of sites in this block (dummy stations included) |
| d/freq | Data values per frequency per site (MT=8, VTF=4, PT=4) |
| Freq tot | Sum of `nfreq` across all sites in this block |
| Data tot | `Freq tot x d/freq` - contribution to the data vector |

---

## Observation types and data lengths

| Type | d/freq | Components |
|---|---|---|
| `MT` | 8 | Zxx_re, Zxx_im, Zxy_re, Zxy_im, Zyx_re, Zyx_im, Zyy_re, Zyy_im |
| `VTF` | 4 | c0, c1, c2, c3 |
| `PT` | 4 | c0, c1, c2, c3 |

---

## Return dict keys (`summarise_observe_dat`)

| Key | Type | Description |
|---|---|---|
| `path` | str | Source path (or `"<in-memory>"` for dict input) |
| `blocks` | list[dict] | One entry per observation-type block (see below) |
| `n_sites_total` | int | Total sites across all blocks |
| `n_freq_total` | int | Total frequency rows across all blocks |
| `n_data_total` | int | Total data-vector length |

Each entry in `blocks`:

| Key | Type | Description |
|---|---|---|
| `obs_type` | str | `"MT"`, `"VTF"`, or `"PT"` |
| `n_sites` | int | Number of sites in this block |
| `n_sites_empty` | int | Stations with 0 frequencies (dummy stations) |
| `dat_length` | int | Data values per frequency per site |
| `n_freq_per_site` | list[int] | `nfreq` for each site |
| `n_freq_total` | int | Sum of `n_freq_per_site` |
| `n_data_total` | int | `n_freq_total x dat_length` |

---

## Parameter summary output

At the start of `main()`, after resolving the paths, the script writes
`femtic_summarize_observe_dat_summary.md` (next to the script itself) via a
small, self-contained `_write_param_summary()` helper (stdlib only). It
records the script's absolute path, the run date/time, and the user
parameter `PATHS` as resolved.

---

## Dependencies

- Python standard library for the fallback parser and all output
  (`argparse`, `inspect`, `os`, `sys`, `pathlib`, and `datetime` inside the
  summary helper).
- Py4MTX header: the environment variables `PY4MTX_ROOT` and `PY4MTX_DATA`,
  and the Py4MTX modules `version` and `util` (title line only).
- Optional: `femtic.py` (and, through it, `ensembles.py`, numpy, scipy,
  joblib) for the robust parser. Without it the script uses the fallback
  parser described above.

---

## Provenance

| Date | Author | Note |
|---|---|---|
| 2026-06-10 | Claude Sonnet 4.6 (Anthropic) | Created. |
| 2026-08-13 | Claude Sonnet 5 (Anthropic) | Added `femtic_summarize_observe_dat_summary.md` output after CLI argument parsing: writes the resolved `paths` argument, script path, and run date/time via a self-contained, stdlib-only helper. |
| 2026-10-24 | Claude Sonnet 5.5 (Anthropic) | Added the usual Py4MTX header (`PY4MTX_DATA` / `PY4MTX_ROOT` from the environment, `py4mt/modules` and `py4mt/scripts` on `sys.path`, `version` / `util` title line). With `femtic.py` now found, the table is printed by the script in plain ASCII instead of `femtic._print_observe_summary()` (Unicode box characters); parsing stays with `femtic.py`. Added a USER SECTION (`PATHS`); the fallback message now gives the import error; the fallback also counts dummy stations. README rewritten in plain ASCII with a real example. Tested with the real `femtic.py` and stub `version` / `util` / `ensembles` modules on a synthetic file; both parser routes gave identical tables. |

AI-generated documentation (Claude, Anthropic; generated 2026-10-24). Review
before production use.
