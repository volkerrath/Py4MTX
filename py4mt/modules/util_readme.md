# util.py — General-purpose utilities for Py4MT

`util.py` is a collection of general-purpose helper functions used across the
Py4MT package. It covers workspace persistence, coordinate transforms, file
manipulation, grid generation, geometry, and numerical utilities.

---

## HDF5 workspace save/load

MATLAB-like workspace persistence using HDF5. Numeric arrays, scalars,
strings, and JSON-serializable objects are stored; non-serializable objects
are skipped with a warning.

- `save_workspace_hdf5(filename, namespace=None)` — save filtered namespace
- `load_workspace_hdf5(filename, namespace=None)` — restore into namespace

---

## Module introspection

- `list_module_callables(module, public_only=False)` — list callable objects
  defined in a module
- `running_in_notebook()` — detect Jupyter notebook environment
- `runtime_env()` — detect runtime (spyder, jupyter, ipython, python)
- `list_functions(filename)` — print function names from a Python source file

---

## Coordinate projections (pyproj)

All projection functions use the modern `pyproj.Transformer` API.

| Function | Conversion |
|----------|------------|
| `get_utm_zone(lat, lon)` | Get EPSG code for UTM zone |
| `get_local_crs(lon, lat)` | Query projected CRSs near a point |
| `proj_latlon_to_utm` | WGS84 → UTM |
| `proj_utm_to_latlon` | UTM → WGS84 |
| `proj_latlon_to_itm` | WGS84 → Irish Transverse Mercator |
| `proj_itm_to_latlon` | ITM → WGS84 |
| `proj_itm_to_utm` | ITM → UTM |
| `proj_utm_to_itm` | UTM → ITM |
| `project_wgs_to_geoid` | Ellipsoid height → geoid (EGM2008) |
| `project_utm_to_geoid` | UTM + ellipsoid → geoid |
| `project_gk_to_latlon` | Gauss-Krüger → WGS84 |

---

## File and string utilities

- `get_filelist(searchstr, searchpath)` — glob-like file listing
- `get_files(SearchString, SearchDirectory)` — simple file filter
- `strcount(keyword, fname)` — count keyword occurrences in a file
- `strdelete(keyword, fname_in, fname_out)` — delete lines containing keyword
- `strreplace(key_in, key_out, fname_in, fname_out)` — find-and-replace in file
- `symlink(src, dst)` — create symlink (`ln -sf`)
- `filecopy(src, dst)` — copy file/directory (`cp -f`)
- `make_pdf_catalog(workdir, pdflist, filename)` — merge PDFs into one catalog

---

## Archive unpacking

- `unpack_compressed(directories, *, recurse=False, remove_archive=False, verbose=True)` — unpack all compressed archives found in one or more directories

Supported formats: `.zip`, `.tar`, `.tar.gz` / `.tgz`, `.tar.bz2` / `.tbz2`, `.tar.xz` / `.txz`, and single-file `.gz`, `.bz2`, `.xz`.  
Multi-file archives (zip/tar) are extracted into the same directory as the archive; single-file compressed files are decompressed in-place (`.gz` → stem, etc.).  
`recurse=True` walks sub-directories. `remove_archive=True` deletes each archive after successful extraction.  
Returns a `list[Path]` of successfully processed archives.

```python
# Unpack everything in two directories
unpack_compressed(["/data/raw", "/data/aux"])

# Recursive, clean up afterwards
unpack_compressed("/data/raw", recurse=True, remove_archive=True)
```

---

## Grid generation

- `gen_grid_latlon(LatLimits, nLat, LonLimits, nLon)` — equidistant lat/lon grid
- `gen_grid_utm(XLimits, nX, YLimits, nY)` — equidistant metric grid
- `gen_searchgrid(Points, XLimits, dX, YLimits, dY)` — bin points into a 2-D grid

---

## Geometry

- `point_inside_polygon(x, y, poly)` — point-in-polygon test
- `choose_data_poly(Data, PolyPoints)` — select data inside a polygon
- `choose_data_rect(Data, Corners)` — select data inside a rectangle
- `proj_to_line(x, y, line)` — project a point onto a line

---

## Numerical utilities

- `KLD(P, Q)` — Kullback-Leibler divergence
- `calc_lc_corner(dnorm, mnorm)` — L-curve corner estimation
- `curvature(x_data, y_data)` / `circumradius` / `circumcenter` — curvature helpers
- `calc_resnorm(data_obs, data_calc, data_std, p)` — residual p-norm
- `calc_rms(dcalc, dobs, Wd)` — NRMS and SRMS
- `dctn(x)` / `idctn(x)` — N-D discrete cosine transform
- `fractrans(m, x, a)` — fractional derivative (requires `differint`)
- `nan_like(a)` — NaN-filled array matching shape of `a`

---

## Rotation matrices

- `rot_x(angle_deg)`, `rot_y(angle_deg)`, `rot_z(angle_deg)` — single-axis rotation
- `rot_full(T, angle_deg_x, angle_deg_y, angle_deg_z)` — combined rotation of a tensor

---

## Other helpers

- `dd(lat, lon)` — DMS string to decimal degrees
- `stop(s)` — exit with message
- `unique(seq)` — unique elements preserving order
- `bytes2human(n)` — human-readable byte sizes
- `nearly_equal(a, b, sig_fig)` — approximate float comparison
- `check_env(envar)` — verify conda environment is active
- `dict_to_namespace(d)` — dict → `SimpleNamespace`
- `print_title(version, fname)` — print version and file info
- `splitall(path)` — split a path into all its components
- `dictget(d, *keys)` — multi-key dict lookup

---

## FT sign-convention correction (`ft_convention.py`)

A companion module — **not part of `util.py`** — for correcting the
Fourier-transform sign convention of MT transfer functions loaded from
instruments with the e⁺ⁱωᵗ convention (Phoenix MTU series).

`data_proc.load_edi` handles this automatically via the `manufacturer`
parameter.  `ft_convention.py` provides the same logic as standalone
functions for post-hoc correction of already-loaded dicts.

| Function | Description |
|----------|-------------|
| `correct_ft_convention(data_dict, *, from_convention, to_convention)` | In-place convention correction with full bookkeeping |
| `apply_conjugation(data_dict)` | Low-level conjugation of Z, T, and P |
| `is_corrected(data_dict)` | Returns `True` if dict is already in standard e⁻ⁱωᵗ convention |
| `correct_batch(sites, *, from_convention, to_convention)` | Apply correction to a list of site dicts |

Constants `CONV_STANDARD = "e-iwt"`, `CONV_PHOENIX = "e+iwt"`,
`PHOENIX_MANUFACTURERS`, `STANDARD_MANUFACTURERS` are exported for
use in scripts.

```python
import ft_convention

# Correct a Phoenix EDI that was loaded without the manufacturer flag
site = data_proc.load_edi("SITE_PHX.edi")        # loaded as Metronix → wrong Im
ft_convention.correct_ft_convention(site, from_convention="e+iwt")
print(site["ft_convention"])  # "e+iwt_corrected"

# Check before acting
if not ft_convention.is_corrected(site):
    ft_convention.correct_ft_convention(site, from_convention="e+iwt")

# Preferred: pass manufacturer at load time — no post-hoc fix needed
site = data_proc.load_edi("SITE_PHX.edi", manufacturer="phoenix")
```

---

Author: Volker Rath (DIAS)
Modified: 2026-03-25 — added ft_convention.py section; Claude Sonnet 4.6 (Anthropic)  
Modified: 2026-03-26 — added unpack_compressed() section; Claude Sonnet 4.6 (Anthropic)
