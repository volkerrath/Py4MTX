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

Author: Volker Rath (DIAS)
