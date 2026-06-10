# femtic_summarize_observe_dat.py

Summarise the data content of one or more FEMTIC `observe.dat` files:
number of sites, frequencies per site, data values per frequency, and
overall data-vector size, broken down by observation type (MT, VTF, PT).

---

## Purpose

Quickly inspect the size and structure of an observation file without
running a full inversion.  Useful for:

- confirming the data-vector dimension before setting up an inversion,
- comparing site/frequency counts between original and perturbed ensemble members,
- checking which observation types are present in a file.

When `femtic.py` is importable the script delegates to
`femtic.summarise_observe_dat()` and `femtic._print_observe_summary()`.
Otherwise it falls back to a self-contained minimal parser with no external
dependencies.

---

## Usage

```bash
# Single file
python femtic_summarize_observe_dat.py observe.dat

# Glob across ensemble members
python femtic_summarize_observe_dat.py ensemble_*/observe.dat

# Scan a directory for observe.dat
python femtic_summarize_observe_dat.py /path/to/run/
```

When scanning a directory the script looks for a file literally named
`observe.dat`.  To match other names use an explicit glob pattern.

---

## Programmatic usage (via `femtic.py`)

```python
import femtic as fem

# From a path — parses the file (fast: no MT-derived quantities computed)
s = fem.summarise_observe_dat("observe.dat")

# From an already-parsed dict (zero extra I/O cost)
parsed = fem.read_observe_dat("observe.dat")
s = fem.summarise_observe_dat(parsed)

# Access results
print(s["n_data_total"])     # total data-vector length
print(s["n_sites_total"])    # total number of sites across all blocks

for b in s["blocks"]:
    print(b["obs_type"], b["n_sites"], b["n_data_total"])

# Re-print the table from a cached dict
fem._print_observe_summary(s)
```

---

## Output format

```
────────────────────────────────────────────────────────────
  File          : observe.dat
  ┌──────────┬────────┬────────┬──────────┬──────────┐
  │ Obs type │ Sites  │ d/freq │ Freq tot │ Data tot │
  ├──────────┼────────┼────────┼──────────┼──────────┤
  │ MT       │     42 │      8 │     1 680 │   13 440 │
  │ VTF      │     42 │      4 │     1 680 │    6 720 │
  ├──────────┴────────┴────────┼──────────┼──────────┤
  │ Total                      │     3 360 │   20 160 │
  └────────────────────────────┴──────────┴──────────┘
  Sites total   :       42
```

If sites within a block have different numbers of frequencies, an additional
line shows the range (`nfreq_min–nfreq_max`) under that block row.

Column meanings:

| Column | Description |
|---|---|
| Obs type | Observation type: `MT`, `VTF`, or `PT` |
| Sites | Number of sites in this block |
| d/freq | Data values per frequency per site (MT=8, VTF=4, PT=4) |
| Freq tot | Sum of `nfreq` across all sites in this block |
| Data tot | `Freq tot × d/freq` — contribution to the data vector |

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
| `dat_length` | int | Data values per frequency per site |
| `n_freq_per_site` | list[int] | `nfreq` for each site |
| `n_freq_total` | int | Sum of `n_freq_per_site` |
| `n_data_total` | int | `n_freq_total × dat_length` |

---

## Dependencies

None beyond the Python standard library (uses only `pathlib`).
When `femtic.py` is on the path, `read_observe_dat` is used for robust
parsing; the fallback uses a minimal line-by-line parser.

---

## Provenance

| Date | Author | Note |
|---|---|---|
| 2026-06-10 | Claude Sonnet 4.6 (Anthropic) | Created. |
