# `plot_joint.py` — Depth-Slice Maps & Sections of the Interpolated Grid

Plots every field on the common grid produced by `interpolate.py` —
MT resistivity/conductivity/sensitivity and/or seismic tomography
Vp/Vs/Vp-Vs-ratio/density, whichever were interpolated — **including
their spatial gradients** — as depth-slice maps and (where the grid is
regular) vertical cross-sections. This is the plotting counterpart to
`cluster.py`: both read `interpolate.py`'s output (`INTERP_FILE`) and
are completely agnostic to how that grid was built or which
interpolation method produced it. Where `cluster.py` clusters the
fields and plots the resulting class labels, `plot_joint.py` plots
each field's own value (and gradient magnitude/components) directly —
no clustering.

```
precompute.py → interpolate.py → {SITE_PREFIX}_interp_<method>.nc
                                                     ↓
                                    plot_joint.py → figures
```

See `README_interpolate.md` and `README_cluster.md` for the upstream
stages; basemap, colorbar, marker/label style, CSV feature paths,
`VSLICES` schema, and isoline settings are carried over unchanged from
`plot_seis.py`/`cluster.py` — see those scripts' own READMEs for the
full rationale behind each one. Only what's specific to plotting an
arbitrary set of interpolated fields is covered here.

## Site selector

`SITE_PREFIX` ("tacna" or "saba") must match `SITE_PREFIX` in
`precompute.py`/`interpolate.py`. Default is `"saba"` — the complete,
verified configuration for this script. Several Tacna-only geographic
settings (`PROFILE_CD_LON/LAT`, `PROFILE_2_LON/LAT`, `VSLICES`
endpoints, `ARROW_LON/LAT`, `VOLC_LABEL_IDX`) are carried over from
`plot_seis.py` and have **not** been re-verified for Sabancaya —
flagged inline in the script rather than guessed at. `MAP_XLIM`/
`MAP_YLIM` default to `None` (auto from the grid), which is safe for
either site.

## Two grid modes (see `interpolate.py`'s `TARGET_GRID`)

- **`"joint"`** — a genuinely regular UTM-km (depth, northing,
  easting) grid. Depth-slice maps are drawn with `imshow(extent=...)`;
  vertical cross-sections (`VSLICES`, same schema as `plot_seis.py`)
  are supported, sampled via a 3-D `RegularGridInterpolator` directly
  in UTM space.
- **`"seismic"`** — a reused seismic-tomography native
  (depth, row, col) grid, not regular in UTM space (2-D
  `utm_easting_km`/`utm_northing_km` aux coords instead). Depth-slice
  maps are drawn with `pcolormesh(shading="nearest")`, matching
  `cluster.py`'s own rendering. Vertical cross-sections are **not**
  supported in this mode — `VSLICES` is silently skipped with a
  printed warning. Build `INTERP_FILE` with `TARGET_GRID="joint"`
  instead if you need sections.

## Gradients (`PLOT_GRADIENT`)

If `interpolate.py`'s `COMPUTE_GRADIENT` was `True`, `INTERP_FILE`
also carries, for each gradient variable, up to four extra data
variables: `{key}_grad_easting`, `{key}_grad_northing`,
`{key}_grad_depth` (partial derivatives) and `{key}_grad_mag` (their
combined 3-D magnitude). `PLOT_GRADIENT` here controls whether this
script *also* produces gradient maps/sections for those fields.
`GRADIENT_COMPONENTS` picks which of `"mag"`/`"easting"`/`"northing"`/
`"depth"` to plot — `"mag"` only by default. A field with no matching
`{key}_grad_*` in `INTERP_FILE` is silently skipped.

### Standard layout: field + `|∇field|` paired (`PAIR_GRADIENT_WITH_FIELD`)

Whenever a field's gradient magnitude was plotted (`PLOT_GRADIENT=True`,
`"mag"` in `GRADIENT_COMPONENTS`, and that field has one), it's drawn
in the **same figure** as the field itself, each keeping its own
colour scale/colorbar (`VAR_STYLE`/`GRAD_STYLE` respectively) — rather
than as two separate files. This is the default
(`PAIR_GRADIENT_WITH_FIELD = True`); set it `False` to go back to one
file per panel. The two panels are arranged side by side (1×2) or
stacked (2×1) automatically, based on the aspect ratio a *single*
panel would have on its own (`_build_dual_panel_figure`) — a
wide/landscape panel stacks vertically so the combined figure doesn't
end up implausibly wide; a tall/portrait panel goes side by side.
Falls back to a single-panel figure automatically wherever there's
nothing to pair with. Any other gradient component you also request
(e.g. `"easting"`) is **not** paired — it remains its own separate
single-panel figure with a `_grad<component>` filename suffix.

## Colour scales (`VAR_STYLE`)

Unlike `plot_seis.py`/`plot_dens.py` — each hard-coded to one or three
known fields — the set of fields actually present in `INTERP_FILE` is
only known at run time (whatever `interpolate.py`'s `INTERP_VARS`
happened to be). `VAR_STYLE` is a dict keyed by field name, giving
each a colormap and a `(cmin, cmax)` pair; a field not listed there
falls back to `DEFAULT_CMAP` and auto-scaling. Gradient
magnitude/component panels always auto-scale unless given an explicit
override in `GRAD_STYLE`.

## Output filenames

Every figure is tagged with **area** (`SITE_PREFIX`) and
**interpolation method** (`INTERP_TAG`), so files from different sites
or different interpolation runs never collide or get overwritten:

| Content | Filename pattern |
|---|---|
| Depth-slice map | `{field}_joint_{depth}km_{SITE_PREFIX}_{INTERP_TAG}.{ext}` |
| Depth-slice map, gradient (unpaired) | `{field}_joint_{depth}km_{SITE_PREFIX}_{INTERP_TAG}_grad[component].{ext}` |
| Vertical section | `{field}_section_{profile_name}_{SITE_PREFIX}_{INTERP_TAG}.{ext}` |
| Vertical section, gradient (unpaired) | `{field}_section_{profile_name}_{SITE_PREFIX}_{INTERP_TAG}_grad[component].{ext}` |

`[component]` is empty for `"mag"` and the component name otherwise
(e.g. `_gradeasting`). Paired field+`|∇field|` figures use the plain
(non-`_grad`) stem, since both panels live in the one file.

`INTERP_TAG` is derived automatically from `INTERP_FILE`'s own
`{SITE_PREFIX}_interp_<method>.nc` naming — it is **not** a separate
user setting. Recognised method strings and their tags:

| `INTERP_FILE` contains | `INTERP_TAG` |
|---|---|
| `kriging` / `krige` / `krig` | `krig` |
| `rbf` | `rbf` |
| `idw` | `idw` |
| `nearest` / `nearest_neighbor(s)` / `nn` | `nn` |

If none of these match (e.g. a custom `interpolate.py` `OUTPUT_FILE`
naming), the script prints a warning and falls back to using the raw
method substring verbatim in filenames, rather than guessing.

`PLOT_FORMATS` (default `[".pdf", ".jpg"]`) and `PLOT_DPI` control
which formats/resolution each figure is saved in; every stem above is
written once per entry in `PLOT_FORMATS`.

## On-screen display (`SHOW_PLOTS`)

Every figure is always written to disk via `save_fig()` regardless of
this setting. `SHOW_PLOTS` (default `False`) additionally controls
whether it's *also* popped up on screen. Previously the script called
`plt.show()` unconditionally after every figure, which only behaves
reasonably in an environment that keeps a live GUI event loop open
between calls (Spyder's own console); the same script run from a plain
terminal, a batch job, or on the DIAS cluster would either block on a
display that never advances, or error outright with no display at all.
`_maybe_show()` now guards every call site: it only invokes `plt.show()`
when `SHOW_PLOTS=True` **and** `matplotlib.is_interactive()` is true, so
headless/batch runs work unchanged by default, and turning `SHOW_PLOTS`
on stays safe even if the script happens to run somewhere without a
display. Set `SHOW_PLOTS = True` when working interactively in Spyder
(or another GUI/inline backend) if you want figures displayed as they're
produced.

## Plot titles

Every panel title also carries the same area + interpolation-method
tag as the filename, appended as a `[SITE_PREFIX, INTERP_TAG]` suffix,
e.g. `rho at 5.0 km [tacna, krig]` or `|∇vp| — profile AA' [saba, rbf]`
— so a figure is self-identifying even once separated from its
filename (e.g. pasted into a slide or PDF report).

## Dependencies

`numpy`, `xarray`, `pandas`, `matplotlib`, `scipy`
(`RegularGridInterpolator`, only used for `"joint"`-mode vertical
sections), plus the local `plotpy.py` helper module.

---
Authors: Svetlana Byrdina (SMB) & Volker Rath (DIAS).
AI-assisted development: Claude (Anthropic).
License: GNU General Public License v3 (GPL-3.0-or-later).
