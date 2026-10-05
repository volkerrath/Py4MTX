# femtic_plot_distort.py

Plot FEMTIC static (frequency-independent) galvanic distortion matrices as
distortion ellipses on a map, optionally over an SRTM-derived hillshade
basemap.

---

## Concept

Each site's real 2x2 distortion matrix `C` is applied to the unit circle;
the resulting ellipse encodes both the axis ratio (anisotropy/gain) and the
rotation of the distortion in a single glyph. An undistorted site
(`C = identity`) plots as a perfect circle. Ellipses are colour-coded by a
scalar severity metric so problem sites stand out at a glance.

This follows the visualisation approach discussed for FEMTIC's joint
conductivity + full-distortion-matrix inversion (see
Avdeeva et al., 2015, `doi:10.1093/gji/ggv144`) and the eigenvalue/SVD
treatment of the 2x2 distortion tensor in Lilley (2016,
`doi:10.1071/EG14093`), from which the condition-number severity metric is
taken.

Optionally (`GB_DECOMP_ENABLE = True`), each site's `C` is additionally
decomposed into the classic Groom & Bailey (1989, `doi:10.1029/JB094iB02p01913`)
twist / shear / gain / anisotropy parameters, via a numerical least-squares
fit to `C = g * T(twist) * S(shear) * diag(1+s, 1-s)`, and plotted as a 2x2
grid of per-site maps (one panel per parameter).

---

## Installation / requirements

- `numpy`, `matplotlib` (required)
- `rasterio`, and either the `elevation` package (auto-download, requires
  GDAL + internet) or a pre-downloaded GeoTIFF via `SRTM_LOCAL_FILE`
  (only required when `BASEMAP_ENABLE = True`)
- `femtic.py`, `util.py` (py4mt) importable on `PYTHONPATH` / same directory

If `rasterio` or `elevation` are missing, or the SRTM download fails (no
network, GDAL/`eio` not configured), the basemap step is skipped with a
warning -- the ellipse plot itself is never blocked by a missing basemap.

---

## Quick start

Edit the `CONFIG` block at the top of the script and run:

```bash
python femtic_plot_distort.py
```

Or call it programmatically:

```python
import femtic_plot_distort as fpd

fpd.plot_distortion_ellipses(
    distortion_file="distortion_iter7.dat",
    site_dat="site.dat",
    display_coords="latlon",
)
```

---

## Inputs

| File | Format |
|---|---|
| `DISTORTION_FILE` | FEMTIC static distortion file, read via `fem.read_distortion_file`: header line (site count) then one row per site: `sitenum c00 c01 c10 c11`. |
| `SITE_DAT` | Sitelist CSV (as produced by `mt_make_sitelist.py`), read via `fem.read_site_dat`: `name, lat, lon, elev, sitenum, easting, northing`. |

Sites are matched between the two files by `sitenum` (re-parsed directly
from `DISTORTION_FILE`, since `fem.read_distortion_file` itself does not
return the site-ID column). Unmatched site numbers are skipped with a
warning rather than aborting the run.

---

## Key CONFIG parameters

| Parameter | Meaning |
|---|---|
| `DISPLAY_COORDS` | `"model"` (model-local km), `"utm"` (absolute UTM km), or `"latlon"` (decimal degrees). |
| `UTM_ZONE`, `UTM_NORTHERN` | UTM zone / hemisphere, used for `"utm"`/`"model"` display and for reprojecting the SRTM basemap. |
| `UTM_ORIGIN_E`, `UTM_ORIGIN_N` | UTM coordinates of the model origin, used only for `"model"` display. |
| `DISTORTION_KIND` | `"full"` plots `C = C' + I` (identity distortion -> unit circle); `"prime"` plots `C'` (deviation from identity) directly. |
| `ELLIPSE_SCALE` | `"auto"` (fraction of median nearest-neighbour site spacing) or an explicit float in plot units. |
| `COLOR_BY` | `"condition"` (sigma_max/sigma_min, Lilley 2016), `"determinant"` (`|det C|`), or `"gain"` (mean singular value). |
| `BASEMAP_ENABLE`, `SRTM_PRODUCT`, `SRTM_LOCAL_FILE` | SRTM hillshade basemap controls; set `SRTM_LOCAL_FILE` to a pre-downloaded GeoTIFF on offline/HPC nodes. |
| `GB_DECOMP_ENABLE` | `False` (default) / `True` -- adds a second figure: a 2x2 panel of Groom-Bailey twist/shear/gain/anisotropy maps. |
| `OUTPUT_BASENAME`, `OUTPUT_FORMATS` | Output file stem and a list of formats to save, e.g. `["png"]` or `["pdf", "jpg"]`. Each format is saved as its own file; the GB panel (if enabled) is saved as `f"{OUTPUT_BASENAME}{GB_BASENAME_SUFFIX}.{fmt}"`. |

---

## Groom-Bailey decomposition panel

`C = g * T(twist) * S(shear) * diag(1+s, 1-s)` is fit to each site's
distortion matrix via bounded, multi-start `scipy.optimize.least_squares`
(`twist, shear` in `(-45, 45]` degrees, `s` in `(-1, 1)`, `g > 0`) --
algebraically the same T/S/A parameterisation Groom & Bailey (1989) define
via `tan(twist)`/`tan(shear)`, just written directly in bounded angles for
a better-conditioned fit. This is a 4-parameter fit to `C`'s 4 independent
entries, so an exact reconstruction is expected whenever one exists.

**Known limitation -- negative-determinant sites.** This parameterisation
can only produce `det(C) >= 0` (`det(T)=1`, `det(S)=cos(2*shear)>=0` in the
fitted range, `det(A)=1-s**2>=0`), so a real distortion matrix with
`det(C) < 0` has no exact Groom-Bailey fit. This is a known property of the
classic GB model, not a bug in this script -- see Lilley (2016), section
"Negative determinants: the situation with distortion matrices, and
Groom-Bailey analysis". `_gb_decompose` detects this (and any fit that
fails to converge to a good reconstruction) and returns `None` for that
site; such sites are plotted as a grey `"x"` in all four panels and
excluded from each panel's colour scale, with a warning giving the count.
Use the (always-defined, SVD-based) ellipse/condition-number plot for
those sites instead.

---

## Verification performed

- `ast.parse()` syntax check.
- Smoke-tested against synthetic distortion/site files (stub `femtic`/
  `util` modules, `BASEMAP_ENABLE = False`) for all three `DISPLAY_COORDS`
  values, both `DISTORTION_KIND` options, and all three `COLOR_BY` metrics.
  An identity-matrix test site correctly rendered as a perfect circle at
  the minimum of the colour scale; a strongly distorted test site rendered
  as a visibly elongated, rotated ellipse at the maximum.
- The SRTM download/reprojection path (`elevation` + `rasterio`) could
  **not** be exercised in the environment this script was written in (no
  network access) -- it is implemented per the documented `elevation`/
  `rasterio` APIs but is untested end-to-end. Treat it as the least-proven
  part of this script until run once against a real DEM fetch.
- `_gb_decompose`'s `T(twist) @ S(shear) @ A(s)` parameterisation was
  verified numerically: 200 randomly generated positive-determinant 2x2
  matrices were reconstructed via the multi-start least-squares fit to a
  maximum element-wise error of `~9e-8`, with zero fit failures. This
  confirms the fit is mathematically well-posed and correctly implemented
  against its own model; it has **not** been cross-checked against an
  independent Groom-Bailey implementation or real MT data. The
  negative-determinant flagging path was smoke-tested with a synthetic
  `det(C) = -1` site and correctly excluded from the fit/colour scale.
- Multi-format output (`OUTPUT_FORMATS = ["pdf", "jpg", "png"]`) and the
  GB panel were smoke-tested together against the same synthetic
  distortion/site files; all six expected files (2 figures x 3 formats)
  were written successfully.

---

## References

- Avdeeva, A., Moorkamp, M., Avdeev, D., Jegen, M., Miensopust, M. (2015).
  Three-dimensional inversion of magnetotelluric impedance tensor data and
  full distortion matrix. *Geophysical Journal International*, 202(1),
  464-481. `doi:10.1093/gji/ggv144`
- Groom, R. W., Bailey, R. C. (1989). Decomposition of magnetotelluric
  impedance tensors in the presence of local three-dimensional galvanic
  distortion. *Journal of Geophysical Research: Solid Earth*, 94(B2),
  1913-1925. `doi:10.1029/JB094iB02p01913`
- Lilley, F. E. M. (2016). The distortion tensor of magnetotellurics: a
  tutorial on some properties. *Exploration Geophysics*, 47(2), 85-99.
  `doi:10.1071/EG14093`

---

## Provenance

| Date | Author | Change |
|---|---|---|
| 2026-09-13 | Claude Sonnet 5 (Anthropic) | Created: `plot_distortion_ellipses()` -- distortion-ellipse map plot with optional SRTM hillshade basemap, `model`/`utm`/`latlon` coordinate choice, `full`/`prime` distortion-kind choice, and `condition`/`determinant`/`gain` severity colouring. AI-generated; not verified against a real FEMTIC run (see "Verification performed" above). |
| 2026-09-13 | Claude Sonnet 5 (Anthropic) | Added optional Groom-Bailey decomposition panel (`GB_DECOMP_ENABLE`, `_gb_decompose`, `_plot_gb_panel`) -- 2x2 grid of twist/shear/gain/anisotropy maps, with negative-determinant sites flagged and excluded per Lilley (2016). Replaced single `OUTPUT_FILE` with `OUTPUT_BASENAME` + `OUTPUT_FORMATS` list (`_savefig_multi`), so both figures can be saved in one or more formats (e.g. `["pdf", "jpg"]`) in one run. |

Author: Volker Rath (DIAS)
