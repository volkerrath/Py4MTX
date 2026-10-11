# `structure.py` — Structural-Coupling Diagnostics Between Field Groups

Renamed from `cross_gradient.py` (same position in the pipeline, same
input file). Scope broadened from a single measure (cross-gradient) to
several structural-coupling diagnostics from the joint-inversion
literature, all computed on the same joint grid so they're directly
comparable panel-to-panel:

```
precompute.py -> interpolate.py -> {SITE_PREFIX}_interp_<method>.nc
                                                 |
                                    structure.py -> figures
```

It reads the same `INTERP_FILE` NetCDF as `plot_joint.py`/`cluster.py` and
is equally agnostic to which interpolation method (`krig`/`rbf`/`idw`/`nn`)
produced it.

## Methods implemented

All measures below are derived from the same real-coordinate 3-D
gradients (`gradient_components()`, `numpy.gradient` on the grid's own
non-uniform coordinate arrays — same standard as the rest of the
pipeline), so they share one grid, one mask, and one set of NaN cells,
and are all plottable on the same mesh.

Let `g1 = grad(m1)`, `g2 = grad(m2)` be two fields' local gradient
vectors, θ the angle between them.

| Measure | Formula | Range | Reference |
|---|---|---|---|
| Cross-gradient magnitude | `\|t\| = \|g1 x g2\| = \|g1\|\|g2\| sinθ` | `[0, ∞)` | Gallardo & Meju (2003, 2004) |
| Normalised cross-gradient | `\|t\| / (\|g1\|\|g2\|) = sinθ` | `[0, 1]` | Gallardo & Meju (2003, 2004) |
| Cosine similarity | `cosθ = (g1·g2) / (\|g1\|\|g2\|)` | `[-1, 1]` | Molodtsov, Troyan, Roslov & Zerilli (2013) |
| Squared cosine similarity | `cos²θ` | `[0, 1]` | Shi, Yu, Zhao, Zhang & Yang (2018) |
| Gramian determinant | `det(Gram(g_1..g_N))`, `Gram_ij = g_i·g_j` | `[0, ∞)` | Zhdanov, Gribenko & Wilson (2012) |
| Normalised Gramian | `det(Gram) / Π\|g_i\|²` | `[0, 1]` (Hadamard's inequality) | Zhdanov, Gribenko & Wilson (2012) |
| Windowed mutual information *(optional, off by default)* | local histogram MI of field values, nats | `[0, ∞)` | inspired by Mandolesi & Jones (2014) |

**Built-in consistency check:** since `sin²θ + cos²θ = 1` for the same
angle, `cosine2 == 1 - crossgrad_norm**2` pointwise (verified numerically
to ~1e-16 for this project's synthetic test grid).

**Gramian N=2 vs. cross-gradient:** by Lagrange's identity,
`det(Gram(g1,g2)) = \|g1\|²\|g2\|² - (g1·g2)² = \|g1 x g2\|²` — algebraically
identical to the squared cross-gradient magnitude (verified numerically
to ~1e-20). The Gramian is only genuinely novel at N=3, where
`det(Gram(g1,g2,g3))` is the squared scalar triple product: it vanishes
wherever the three gradient vectors are *coplanar*, not just wherever any
two of them are parallel. `GRAMIAN_GROUPS` accepts 2- or 3-field tuples;
N>3 raises `ValueError` rather than guessing at a general N×N cofactor
expansion.

**Windowed mutual information** is different in kind from everything
above: it compares the two fields' *values* (not gradients) in a local
sliding window via histogram-based density estimation, so it can register
correlation even where both fields are locally smooth. Mandolesi & Jones
(2014) used a *global*, whole-model mutual information as a joint-inversion
objective, not a spatial diagnostic map — the windowed/local version here
is a diagnostic-mapping choice made for this pipeline, not something taken
from that paper. `MI_WINDOW` and `MI_BINS` are free parameters with no
literature- or pipeline-verified default for this project's grids; they
are flagged as such in the code rather than tuned/guessed at, and the
feature defaults to `ENABLE_MUTUAL_INFO = False`. It is only implemented
for depth-slice maps, not sections (see "Known limitations"), and only
for pairs (no 3-way generalisation attempted).

## Further measures considered but not implemented

Flagged rather than added speculatively, since each needs a design choice
this script's config would otherwise have to invent silently:

- **Joint total variation (JTV)** — Haber & Holtzman Gazit (2013), *Surv.
  Geophys.*, 34, 675–695, doi:[10.1007/s10712-013-9232-4](https://doi.org/10.1007/s10712-013-9232-4).
  An inversion *regulariser* (penalises `Σ √(|grad m1|² + |grad m2|²)`),
  not naturally a per-point diagnostic map the way the measures above are.
- **Variation of information (VI)** — introduced for this application in
  Moorkamp (2021), "Joint inversion of gravity and magnetotelluric data
  from the Ernest Henry IOCG deposit with a variation of information
  constraint," *First International Meeting for Applied Geoscience &
  Energy*, SEG, pp. 1711–1715 (a conference proceedings paper; **no DOI
  could be confirmed for it**, flagged rather than guessed at), with the
  full derivation (including `dVI/dm`) given in Moorkamp (2022),
  "Deciphering the state of the lower crust and upper mantle with
  multi-physics inversion," *Geophysical Research Letters*, 49(9),
  e2021GL096336, doi:[10.1029/2021GL096336](https://doi.org/10.1029/2021GL096336)
  — DOI confirmed against AGU/Wiley and independently cross-referenced
  in three other papers' bibliographies.
- **Joint entropy-based coupling** — referenced in passing in some of the
  above literature (e.g. via Zhdanov et al.) but a precise, verifiable
  primary citation was not confirmed here either; flagged rather than
  cited speculatively. Worth chasing down before adopting.

Both VI and joint entropy are, like MI, global/statistical measures over
the whole model rather than gradient-based local ones, and would need the
same kind of windowing choice flagged for MI above to become a spatial
diagnostic.

**References**
- Gallardo, L. A., and M. A. Meju (2003), Characterization of
  heterogeneous near-surface materials by joint 2D inversion of DC
  resistivity and seismic data, *Geophys. Res. Lett.*, 30(13), 1658,
  doi:[10.1029/2003GL017370](https://doi.org/10.1029/2003GL017370).
- Gallardo, L. A., and M. A. Meju (2004), Joint two-dimensional DC
  resistivity and seismic travel time inversion with cross-gradients
  constraints, *J. Geophys. Res.-Solid Earth*, 109, B03311,
  doi:[10.1029/2003JB002716](https://doi.org/10.1029/2003JB002716).
- Molodtsov, D. M., V. N. Troyan, Y. V. Roslov, and A. Zerilli (2013),
  Joint inversion of seismic traveltimes and magnetotelluric data with a
  directed structural constraint, *Geophysical Prospecting*, 61(6),
  1218–1228, doi:[10.1111/1365-2478.12060](https://doi.org/10.1111/1365-2478.12060).
- Shi, B., P. Yu, C. Zhao, L. Zhang, and H. Yang (2018), Linear
  correlation constrained joint inversion using squared cosine similarity
  of regional residual model vectors, *Geophysical Journal
  International*, 215(2), 1291–1307,
  doi:[10.1093/gji/ggy336](https://doi.org/10.1093/gji/ggy336).
- Zhdanov, M. S., A. V. Gribenko, and G. Wilson (2012), Generalized joint
  inversion of multimodal geophysical data using Gramian constraints,
  *Geophysical Research Letters*, 39(9), L09301,
  doi:[10.1029/2012GL051233](https://doi.org/10.1029/2012GL051233).
- Mandolesi, E., and A. G. Jones (2014), Magnetotelluric inversion based
  on mutual information, *Geophysical Journal International*, 199(1),
  242–252, doi:[10.1093/gji/ggu258](https://doi.org/10.1093/gji/ggu258).
- Moorkamp, M. (2022), Deciphering the state of the lower crust and upper
  mantle with multi-physics inversion, *Geophysical Research Letters*,
  49(9), e2021GL096336,
  doi:[10.1029/2021GL096336](https://doi.org/10.1029/2021GL096336).
  *(Not one of the implemented measures above — cited under "Further
  measures considered but not implemented" for the variation-of-
  information approach.)*

All DOIs above were checked against publisher/aggregator records before
inclusion; none are guessed.

## Configuration

- `SITE_PREFIX` — your site's short code (e.g. `"site_a"`), must match `interpolate.py`.
- `INTERP_FILE` — explicit path, or `None` to auto-pick the newest
  `{SITE_PREFIX}_interp_*.nc` in the current directory.
- `STRUCTURE_PAIRS` — list of `(field_a, field_b, label)` tuples for the
  pairwise measures (cross-gradient, cosine, cosine², optional MI).
  `field_a`/`field_b` must be variable names present in `INTERP_FILE`
  (e.g. `"rho"`, `"vp"`, `"vs"`, `"vpvs"`, `"dens"`). `label=None` falls
  back to `"{field_a}-{field_b}"` in titles/filenames.
- `GRAMIAN_GROUPS` — list of `(fields, label)` tuples, `fields` a 2- or
  3-tuple of variable names, for the Gramian determinant. Defaults mirror
  `STRUCTURE_PAIRS`'s pairs (N=2, mainly a cross-check against the
  cross-gradient); an example N=3 group is left commented out — which
  three fields belong together is a scientific choice specific to this
  project, not something to default silently.
- `LOG_FIELDS` — fields gradiented (and, for MI, compared) in log10 space
  rather than linear, kept in sync with the `log10(rho)` convention used
  throughout `plot_joint.py`/`plot_modem_mesh.py`. Defaults to `{"rho"}`.
- `ENABLE_CROSS_GRADIENT` / `PLOT_CROSSGRAD_NORM`, `ENABLE_COSINE`,
  `ENABLE_COSINE2`, `ENABLE_GRAMIAN` / `PLOT_GRAMIAN_NORM`,
  `ENABLE_MUTUAL_INFO` (+ `MI_WINDOW`, `MI_BINS`) — per-measure on/off
  switches; see table above. `ENABLE_MUTUAL_INFO` defaults `False` (see
  "Windowed mutual information" above).
- `PLOT_MAPS` / `MAP_DEPTHS` / `MAP_XLIM` / `MAP_YLIM` — depth-slice map
  controls; `MAP_DEPTHS=None` plots every depth level in the grid.
  `MAP_XLIM`/`MAP_YLIM` default to `None` (auto from the grid), matching
  the project's safe-default policy for un-verified geographic bounds.
- `PLOT_SECTIONS` / `VSLICES` / `VE` — vertical-section controls, same
  `(name, (lon1, lat1), (lon2, lat2), n_samples)` schema as
  `plot_seis.py`/`plot_joint.py`. **`VSLICES` ships empty** — profile
  endpoints are project- and site-specific and were not guessed at; see
  "Known limitations" below before populating it. Not available for
  `mutual_info` regardless (map-only, see above).
- `CMAP_*`/`CMIN_*`/`CMAX_*` — one triple per quantity (`_CG`, `_CG_NORM`,
  `_COS`, `_COS2`, `_GRAM`, `_GRAM_NORM`, `_MI`); bounded quantities
  (`sinθ`, `cos²θ`, normalised Gramian) default to fixed `[0,1]`, `cosθ`
  to fixed `[-1,1]`, unbounded ones (`|t|`, Gramian determinant, MI) to
  per-panel auto-scaling.
- `ISO_LINES_MAP` / `ISO_LINES_VSLICE` / `ISO_LEVELS` — independent
  isoline toggles for maps and sections, following the same
  per-script-independent pattern as the other plot scripts.
- `PLOT_FORMATS` — list of matplotlib-supported output extensions, e.g.
  `["png"]` or `["png", "pdf"]`. `save_paris()` saves once per format and
  returns a list of paths.
- `SHOW_PLOTS` — also display each figure on screen, in addition to
  saving it. Only actually shows if `matplotlib.is_interactive()` is
  also `True` (e.g. Spyder's Qt/inline backends); safe to leave on when
  running headless on the DIAS HPC cluster or from a terminal, since an
  unconditional `plt.show()` would otherwise block/error there — same
  fix already applied in `plot_joint.py` via `SHOW_PLOTS`/`_maybe_show()`,
  mirrored here. Defaults `False`.

## Grid support: `"joint"` only

`interpolate.py`'s `TARGET_GRID` can be `"joint"` (a genuinely regular
UTM-km grid) or `"seismic"` (the reused seismic-tomography native grid,
structured only in row/col index space, not orthogonal in UTM space).

**`structure.py` only supports `TARGET_GRID="joint"`.** A
coordinate-exact 3-D gradient — the standard this pipeline holds gradients
to everywhere else, via `numpy.gradient` on real, non-uniform coordinate
arrays — cannot be computed from `"seismic"`-mode index-space finite
differences without assuming near-orthogonality between grid rows/columns
and the UTM axes. Rather than silently approximating (and rather than
guessing at the seismic grid's local warp), `load_joint_grid()` raises a
clear `ValueError` if `INTERP_FILE` isn't a `"joint"`-mode grid. Re-run
`interpolate.py` with `TARGET_GRID="joint"` to produce a compatible file.

## Three known integration seams (by design, not oversight)

1. **Basemap/marker/label styling.** This script does its own minimal
   `pcolormesh` plotting rather than calling into `tomomt.py`'s
   `build_panel_figure`/`finish_panel_colorbar`/`draw_north_arrow`/
   `add_latlon_ticks`/`clipped_markers`/`clipped_labels`, since their exact
   current signatures weren't available to verify against here. Swapping
   these in to match `plot_joint.py`'s house style exactly is a
   straightforward follow-up — flagged rather than guessed at.
2. **Section profile projection.** `_profile_line()` is a deliberate stub
   that raises `NotImplementedError`: vertical sections need the same
   lon/lat -> UTM-km transformer `plot_seis.py`/`plot_joint.py` already
   use, and re-deriving that transform independently risked a subtle
   mismatch. Wire in the shared transformer there before populating
   `VSLICES` and enabling `PLOT_SECTIONS`. Depth-slice maps (`PLOT_MAPS`)
   do not depend on this and work as shipped.
3. **Windowed mutual information is map-only.** No along-profile version
   is wired in; a small follow-up once seam (2) is resolved, not
   attempted here.

## Verification performed

Before delivery, the pairwise/Gramian identities noted above were checked
numerically against a synthetic joint-grid dataset (not against real
project data): `cos²θ + sin²θ = 1` to ~1e-16, and
`det(Gram_{N=2}) = \|t\|²` to ~1e-20 — both consistent with the exact
algebraic identities they're derived from. `ast.parse` and a
definition-before-use scan were also run on the delivered file.

## Output

Files named
`{SITE_PREFIX}_{group}_{tag}_{INTERP_TAG}_map_z<depth>km.<ext>` (maps,
`tag` one of `crossgrad_mag`/`crossgrad_norm`/`cosine`/`cosine2`/
`gramian`/`gramian_norm`/`mi`) or `..._sec_<name>.<ext>` (sections, MI
excluded), titled with the field group, the quantity plotted, and
`[SITE_PREFIX, INTERP_TAG]` — same suffix convention as `plot_joint.py`.
`group` is `field_a-field_b` (or the 2-/3-field Gramian group, dash-joined)
when `label=None`. `<ext>` is once per entry in `PLOT_FORMATS` (`png` by
default). All outputs for a run are bundled into
`{SITE_PREFIX}_structure_YYYYMMDD_HHMM.zip` (Europe/Paris timestamp, both
the zip name and each member's internal mtime), matching the rest of the
pipeline's delivery convention.
