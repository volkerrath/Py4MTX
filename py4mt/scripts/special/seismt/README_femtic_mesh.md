# `plot_femtic_mesh.py` — Exact-Mesh Plotting for a FEMTIC MT Inversion

FEMTIC counterpart to `plot_modem_mesh.py` (ModEM's exact-mesh plotting
script): depth-slice maps and vertical cross-sections of log₁₀(ρ) from a
FEMTIC 3-D MT inversion, sharing the same basemap engine, styling
conventions, and USER SETTINGS layout as the rest of this pipeline's
plot scripts (`README_mt.md`, `README_seis.md`).

Reads `mesh.dat` + `resistivity_block_iterX.dat` directly via
`femtic.py` — no precompute step of its own, matching
`interpolate.py`'s `"femtic_points"` source kind (see
`README_interpolate.md`'s "FEMTIC meshes" section, which this script's
`FEMTIC_DIR`/`FEMTIC_ORIGIN_E_M`/`FEMTIC_ORIGIN_N_M`/
`FEMTIC_DEPTH_OFFSET_KM`/`FEMTIC_INCLUDE_FIXED`/`FEMTIC_OCEAN` settings
are deliberately named identically to, so the same values apply in both
places).

Authors: Svetlana Byrdina (SMB) & Volker Rath (DIAS)
AI-assisted development: Claude (Anthropic)
License: GNU General Public License v3 (GPL-3.0-or-later)
AI-generated code — review before use in production.

---

## Pipeline

```
mesh.dat + resistivity_block_iterX.dat  →  plot_femtic_mesh.py  →  figures
```

No `interpolate.py`/`INTERP_FILE` step — this script reads the FEMTIC
mesh directly, the same way `plot_modem_mesh.py` reads ModEM's mesh
directly rather than going through `interpolate.py`.

`tomomt.py` and `femtic.py` must both sit alongside this script.
`femtic.py` itself imports `ensembles.py` unconditionally at module
level (for roughness/prior-covariance tools this script doesn't use) —
`ensembles.py` must be importable on `sys.path` for this script to run
at all, the same way `pykrige` must be installed for `interpolate.py`'s
`INTERP_METHOD="kriging"`.

---

## Exact mesh rendering — how, and why it's different from `plot_modem_mesh.py`

FEMTIC meshes are unstructured tetrahedra, not a structured (even if
non-uniform) tensor grid the way ModEM's is. There's no "row of cells"
or "grid edge" the way ModEM's `modem_grid_edges_utm.nc` provides, so
`plot_modem_mesh.py`'s approach — intersect the profile with the mesh's
own grid *lines* to find which cell columns it crosses, then
`pcolormesh` each column's cells against their real edges — has no
tetrahedral equivalent.

The exact-geometry analog used here instead: **every depth-slice map and
vertical section is built by intersecting each FEMTIC element with the
requested plane** — horizontal, at a given depth, for maps; vertical,
containing the profile line, for sections — via `plane_intersect_tet()`,
and rendering the resulting polygon (always a triangle or quadrilateral,
never anything else — see that function's docstring for why) coloured by
that one element's own resistivity, using a matplotlib `PolyCollection`.
Every polygon in every figure is therefore an unmodified value from one
specific real 3-D tetrahedron — nothing is resampled onto a pixel grid
or blended across element boundaries, the same "exact, unblended cut"
standard `plot_modem_mesh.py` holds ModEM output to, adapted to a
different mesh topology.

**Verification performed before this script was used to produce any
figure:** `plane_intersect_tet()` was stress-tested against 2000
randomised non-degenerate tetrahedra — every result checked for the
correct point count (3 or 4), distinct (non-duplicate) points, positive
polygon area, and a single consistent winding direction (i.e. genuinely
convex, non-self-intersecting). The full script was then run end-to-end
against a synthetic FEMTIC mesh (elements at multiple depths and
regions, including air/ocean/fixed) with synthetic topo/bathymetry and
feature CSVs, producing depth-slice maps and a vertical section that
were visually inspected for a correctly-rendered basemap, correctly
masked air/ocean/fixed regions, correctly coloured/positioned
resistivity polygons, and correctly projected feature overlays. One
degenerate edge case was deliberately probed and understood along the
way: a profile line running *exactly* along the diagonal of a
diagonally-symmetric synthetic mesh landed several vertices exactly on
the cutting plane, which the `eps`-tolerance boundary handling correctly
(if unhelpfully, for that specific symmetric case) excludes — a
probability-zero coincidence for any real mesh and an arbitrary profile,
confirmed by re-running with a slightly off-diagonal profile, which
recovered the expected polygons immediately.

## What's intentionally not replicated from `plot_modem_mesh.py`

Flagged here rather than silently omitted or fabricated:

- **No sensitivity-based shading/blanking** (`USE_SENSITIVITY`,
  `SENS_BLANK_THRESHOLD`, `SENS_ALPHA_RANGE` in `plot_modem_mesh.py`).
  `femtic.py`'s element-loading functions (`build_element_arrays()`)
  don't expose a per-element sensitivity/resolution field the way
  ModEM's `.sns` file does via `precompute.py`. If FEMTIC per-element
  sensitivity becomes available some other way, this would be the place
  to wire it in, mirroring `plot_modem_mesh.py`'s machinery.
- **No isoline overlay** (`ISO_LINES_MAP`/`ISO_LINES_VSLICE` elsewhere
  in this pipeline). Contour lines need a continuous field to trace, and
  this script's piecewise-constant-per-element data has no natural one
  without an extra interpolation step onto a temporary regular grid —
  which `plot_modem_mesh.py`'s own isoline docstring already flags as an
  approximation *even for ModEM's regular mesh*. Rather than stack a
  second, larger approximation on top of an already-approximate feature,
  this script omits isolines entirely for now.
- **No exact air/ocean topography staircase on vertical sections**
  (unlike `plot_modem_mesh.py`'s `surf_depth`, derived exactly from the
  mesh's own air cells). Building that here would need, for every
  along-profile bin, the shallowest depth at which the cutting plane
  still intersects a non-air element — a genuine extra piece of
  geometry, not attempted in this version. Sections instead show a
  reused DEM line (same `NC_TOPO_SEIS` grid as the map basemap) purely
  as a visual reference, matching `plot_seis.py`'s simpler topo-line
  convention rather than `plot_modem_mesh.py`'s exact staircase.
- **MT sites via a plain CSV** (`CSV_MT_SITES`), matching
  `plot_seis.py`'s convention, rather than a `precompute.py`-produced
  NetCDF the way `plot_modem_mesh.py`/`plot_modem_image.py` read
  `modem_sites_utm.nc` — FEMTIC has no precompute step of its own to
  produce one.

## Configuration

Most settings (colorbar, axis fonts, map units, feature overlay
CSVs/switches, marker/label styles, `VSLICES` schema, annotation) are
identical in name and meaning to `plot_modem_mesh.py`/`plot_seis.py` —
see those scripts' own READMEs for the full explanation of each; only
what's specific to this script is covered here.

- **`SITE_PREFIX`** — unlike the FEMTIC mesh itself (selected via
  `MESH_FILE`/`BLOCK_FILE`, not by site), this only affects the
  seismic-pipeline topo/bathymetry grid reused for the basemap
  (`NC_TOPO_SEIS`/`NC_BATH`) and this script's own output filenames —
  same caveat as `plot_modem_mesh.py`'s identical setting.
- **`FEMTIC_DIR`/`MESH_FILE`/`BLOCK_FILE`** — where the FEMTIC mesh and
  resistivity block live. Not the same directory as `NC_DIR` (which is
  precompute.py's ModEM/seismic output) in general.
- **`FEMTIC_ORIGIN_E_M`/`FEMTIC_ORIGIN_N_M`** — UTM **metres** of the
  FEMTIC mesh's own local-coordinate origin. **Required** — there is no
  safe default to guess here (same as `interpolate.py`'s identical
  setting). Get it from the FEMTIC run's own setup, or from
  `femtic.estimate_utm_origin()` against known calibration site
  positions.
- **`FEMTIC_DEPTH_OFFSET_KM`** — depth-axis calibration if the mesh's
  own z=0 datum isn't exactly this project's z=0 reference. Default
  `0.0` (assume they already match).
- **`FEMTIC_INCLUDE_FIXED`/`FEMTIC_OCEAN`** — which regions to exclude
  before plotting; mirrors `femtic.read_model()`'s own semantics exactly
  (air always excluded, any `flag==1` region always excluded even if you
  override ocean detection, region 1 excluded if inferred/forced as
  ocean) — same as `interpolate.py`'s identical settings.
- **`DEPTH_SLICES_KM`** — depth-slice maps are computed fresh from the
  mesh at each requested depth (no precomputed per-depth file the way
  ModEM's `modem_rho_utm_{D}km.nc` is).
- **`MAP_XLIM`/`MAP_YLIM`** — default `None` (auto from the mesh's own
  free-element extent), the same safe-default policy this pipeline uses
  everywhere else for a site/mesh with no verified framing yet
  (`structure.py`/`crossplots.py`'s identical convention) — set your own
  once a framing has been chosen.
- **`PLOT_FILENAME_SUFFIX`** — default `"_fem"`, so output is told apart
  at a glance from `plot_modem_image.py`'s (`_img`) and
  `plot_modem_mesh.py`'s (`_msh`) ModEM-derived resistivity figures.

## On-screen display (`SHOW_PLOTS`)

Same convention as every other plot script in this pipeline: every
figure is always written to disk via `save_fig()` regardless of this
setting; `SHOW_PLOTS` (default `False`) additionally controls whether
it's *also* popped up on screen, guarded by `_maybe_show()` so it only
calls `plt.show()` when `matplotlib.is_interactive()` is genuinely true
— safe to leave on even if the script happens to run somewhere headless.

## Coordinate convention

**UTM Zone 19S (EPSG:32719)**, distances in km, depth in km positive
down — matching FEMTIC's own documented z-positive-downward convention
exactly, so no sign flip is needed anywhere in this script, only the
metres↔km unit conversion and the UTM-origin offset (see
`load_femtic_mesh()`'s docstring).

## Dependencies

```
numpy, matplotlib, xarray, pandas, pyproj, scipy
```
plus the local `tomomt.py` and `femtic.py` modules (and, transitively,
`ensembles.py` — see above).

## Typical run

```bash
python3 plot_femtic_mesh.py   # reads mesh.dat + resistivity_block_iterX.dat
                               # directly, writes depth-slice maps + VSLICES
                               # sections (PLOT_FORMATS)
```

Re-run whenever the FEMTIC inversion result (`BLOCK_FILE`) or mesh
itself changes, or whenever `DEPTH_SLICES_KM`/`VSLICES`/any plotting
setting changes — there's no separate precompute step to re-run first,
unlike the ModEM/seismic scripts.
