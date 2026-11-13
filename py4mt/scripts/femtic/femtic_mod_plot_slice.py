#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_mod_plot_slice.py — 2-D slice panels for a FEMTIC resistivity model.

Produces one figure with horizontal map, N-S curtain, E-W curtain, and/or
arbitrary plane slices using exact tetrahedron-plane intersection
(``fviz.plot_model_slices``).

Sister scripts:
  ``femtic_mod_plot_bh.py``  — 1-D ρ(z) borehole log figures.
  ``femtic_mod_plot_3d.py``  — PyVista 3-D rendering and VTK/VTU export.

Family: the "mod" scripts draw SINGLE models (this script, mod_plot_bh,
mod_plot_3d, with mod_math / mod_edit for model arithmetic and editing); the
"ens" scripts (femtic_ens_post, femtic_ens_plot_slices, femtic_ens_plot_bh,
and the quality-control script femtic_data_misfit) work on the ensemble.

One model or several (batch)
----------------------------
MODEL_FILE is, as before, ONE resistivity block file; PLOT_FILE is its figure.
It can also be

    a list of block files,
    a glob pattern  ("ens/ens_*/resistivity_block_iter0.dat"),
    run directories (a directory, a list, or a glob such as "ens/ens_*"):
        the block file is chosen by MODEL_ITER,
            "last" -- the iteration of the last row of femtic.cnv,
            "best" -- the iteration with the lowest nRMS in femtic.cnv,
            N      -- resistivity_block_iterN.dat (0 = starting model; for
                      an RTO member the perturbed prior),
            or a list such as [0, "last"] (one figure each).

With more than one resulting model there is ONE FIGURE PER MODEL, named
<label>_<tag>_slices.<ext> (label = run directory, tag = "iterN") in
BATCH_OUT_DIR (None = the model's own directory); the nRMS of that iteration
(from femtic.cnv) is annotated, and, with BATCH_CATALOG, all figures are also
collected in one multi-page PDF. PLOT_FILE is not used then. This takes over
the per-member "iter0 / best" figures of the former femtic_ens_plot.py.

Slice positions can be given in three equivalent systems
---------------------------------------------------------
Every horizontal slice position key (``x0``, ``y0``, and the horizontal
components of ``point``) can optionally carry a coordinate-system tag by
replacing the plain scalar with a two-element tuple:

    (value, "crs")

where ``crs`` is one of:

    "model"   model-local metres, origin at mesh centre (default; a bare
              float is treated as "model" — fully backward-compatible)
    "utm"     UTM metres in the same zone as the mesh origin (zone number
              is auto-derived from UTM_ORIGIN_LAT / UTM_ORIGIN_LON)
    "latlon"  geographic decimal degrees (longitude for x0 / NS slices,
              latitude for y0 / EW slices)

The conversion chain is always:

    lat/lon ──► UTM(m) ──► model-local(m)
    UTM(m)            ──► model-local(m)
    model-local(m)    ──► (no-op)

Depth (z0) is always in model-local metres; no geographic conversion applies.

Examples (inside PLOT_SLICES)
------------------------------
    # Plain float — model-local metres (unchanged from previous version):
    dict(kind="map",  z0=5000.0)
    dict(kind="ns",   x0=0.0)

    # UTM easting for the NS curtain:
    dict(kind="ns",   x0=(229047.0, "utm"))

    # Geographic longitude for the NS curtain:
    dict(kind="ns",   x0=(-71.537, "latlon"))

    # Geographic latitude for the EW curtain:
    dict(kind="ew",   y0=(-16.409, "latlon"))

    # Arbitrary plane through a geographic point:
    dict(kind="plane",
         point=([-71.5, -16.4, 5000.0], "latlon"),
         strike=45., dip=70.)

    # Two-point vertical profile (fence section) — strike auto-derived:
    dict(kind="profile",
         p1=([-71.50, -16.30], "latlon"),  # start endpoint [lon, lat]
         p2=([-70.90, -16.40], "latlon"),  # end   endpoint [lon, lat]
         z_top=0.0, z_bot=20000.0)         # depth extent [m, z-down]

UTM zone derivation
--------------------
    Zone number is computed from ``UTM_ORIGIN_LON`` (standard 6° bands,
    ignoring Norway / Svalbard exceptions).  Override with
    ``UTM_ZONE_OVERRIDE`` (positive integer) when needed.

Display coordinate system
--------------------------
    DISPLAY_COORDS = "model"  — axis ticks in model-local metres (default)
    DISPLAY_COORDS = "utm"    — axis ticks in absolute UTM metres
    DISPLAY_COORDS = "latlon" — axis ticks in decimal degrees

Provenance
----------
    2026-05-06  vrath / Claude Sonnet 4.6   Created, modelled on
                femtic_mod_edit.py plotting section.
    2026-05-06  vrath / Claude Sonnet 4.6   Added lat/lon and UTM slice-
                position input; pure-Python UTM forward projection;
                auto-derived UTM zone from mesh origin coordinates.
    2026-05-06  vrath / Claude Sonnet 4.6   Added estimate_utm_origin:
                least-squares mesh-centre estimation from N calibration
                sites with known model-local and geographic coordinates.
    2026-05-13  vrath / Claude Sonnet 4.6   Harmonised plotting config block
                with femtic_mod_edit.py: unified variable names, comments,
                and section header.
    2026-05-16  vrath / Claude Sonnet 4.6   Added borehole resistivity log
                step: BOREHOLE_* config block; CRS tagging on x/y positions.
    2026-05-23  vrath / Claude Sonnet 4.6   Moved pure geographic helpers to
                util.py; model-local helpers to femtic.py.  Script-level
                wrappers delegate to both.  SITE_NUMBER accepts list.
    2026-05-23  vrath / Claude Sonnet 4.6   SITE_DAT in mt_make_sitelist.py
                CSV format; read_site_dat() rewritten.
    2026-05-23  vrath / Claude Sonnet 4.6   PLOT_EQUAL_ASPECT config flag.
    2026-05-24  vrath / Claude Sonnet 4.6   Removed ENS_* block (moved to
                snippets.py).
    2026-05-24  vrath / Claude Sonnet 4.6   Moved read_site_position(),
                read_site_dat(), estimate_utm_origin(), _point_in_tet(),
                extract_borehole_log() to femtic.py.
    2026-05-25  vrath / Claude Sonnet 4.6   Added PLOT_SITES_MAPS /
                PLOT_SITES_SLICES, PROJECTION_DIST, SITE_MARKER_SLICES,
                DEPTH_KM, HORIZ_KM, PLOT_NROWS/NCOLS, PLOT_PANEL_HEIGHT/
                WIDTH/FIGSIZE.  UTM display in km.  Cmap deprecation fix.
    2026-05-26  Claude Sonnet 4.6 (Anthropic)
                Moved plot_model_slices / plot_borehole_logs into
                femtic_viz.py.  Added ALPHA_FILE / ALPHA_MODE /
                ALPHA_BLANK_THRESH config vars.
    2026-05-27  vrath / Claude Sonnet 4.6 (Anthropic)
                Passed PLOT_XLIM/YLIM/ZLIM to slice panels.
    2026-05-31  vrath / Claude Sonnet 4.6 (Anthropic)
                Added invert_x per-panel key in PLOT_SLICES.
                Origin estimation now runs before UTM zone derivation.
                Hard-coded UTM_ORIGIN_* set to None (derived at runtime).
    2026-06-03  Claude Sonnet 4.6 (Anthropic)
                Split from femtic_mod_plot.py into femtic_mod_plot_slice.py
                (2-D slices + boreholes) and femtic_mod_plot_3d.py (PyVista
                3-D rendering).  Borehole: BOREHOLE_XLIM now in Ohm*m;
                z_top="surface" supported; lat/lon legend annotation and
                per-trace line-style keys in BOREHOLE_SITES.
                BOREHOLE_NPZ config var for NPZ data export.
                BOREHOLE_IN_SLICE config var: when True, borehole panels are
                embedded inside the slice figure as extra columns to the right
                via plot_model_slices(borehole_sites=...).
                _resolve_borehole_xy removed: CRS conversion for boreholes now
                handled natively in _sample_borehole_logs via utm_zone /
                utm_origin_e/n parameters (no longer needed in script).
                Standalone plot_borehole_logs call updated to pass utm_zone /
                utm_northern / utm_origin_e / utm_origin_n directly.
    2026-06-04  vrath / Claude Sonnet 4.6 (Anthropic)
                Borehole config block (BOREHOLE_*, PLOT_BOREHOLE) and step (6)
                moved to new sister script femtic_mod_plot_bh.py.
                plot_model_slices call: borehole_* parameters removed.
    2026-06-19  Claude Sonnet 4.6 (Anthropic)
                Added kind="profile" to PLOT_SLICES: a vertical fence section
                defined by two endpoint positions (p1, p2) each accepting
                model-local / UTM / latlon CRS tags; strike is derived from
                the p1→p2 azimuth; dip is fixed at 90.  New helper
                resolve_pos_two_point_profile() added to femtic.py;
                resolve_slice_positions() extended with the "profile" branch.
    2026-08-13  Claude Sonnet 5 (Anthropic)
                Added femtic_mod_plot_slice_summary.md output at end of
                run: writes user-set (UPPERCASE) parameters, script path,
                and run date/time via utl.write_param_summary().
    2026-09-06  Claude Sonnet 5 (Anthropic)
                Added SHOW_MODEL_CENTRE (default True): marks the model
                origin on "map" panels whenever DISPLAY_COORDS is
                "utm"/"latlon" via fviz.plot_model_slices'
                show_model_centre parameter; override style with a
                MAP_MARKERS entry carrying "is_model_centre": True.
    2026-10-28  Claude Sonnet 5.5 (Anthropic)
                The parameter summary femtic_mod_plot_slice_summary.md is
                now written to the working directory set in this script
                (WORK_DIR) instead of next to the script:
                utl.write_param_summary(fname, out_dir=WORK_DIR). Needs
                the util.py of the same date (new out_dir argument).
                AI-generated; review before production use.
    2026-11-08  Claude Sonnet 5.5 (Anthropic)
                Several models in one run: MODEL_FILE may be a list, a glob
                or run directories (MODEL_ITER selects the iteration; see the
                docstring), giving one figure per model plus an optional
                catalog PDF (BATCH_* settings; femtic.resolve_model_files,
                fviz.plot_model_slices_robust, fviz.append_catalog_page).
                One model behaves exactly as before. Takes over the
                per-member "single" figures of the former femtic_ens_plot.py.
                AI-generated; review before production use.
    2026-11-09  Claude Sonnet 5.5 (Anthropic)
                Dipping planes (kind="plane", dip != 90) fixed in
                fviz.plot_model_slices: strike is the geographic azimuth
                for every dip (dip towards strike + 90), the axes are the
                along-strike and down-dip distance from `point` [m], the
                default extent is the plane cut with the PLOT_XLIM x
                PLOT_YLIM x PLOT_ZLIM box; per-slice xlim = along-strike and
                ylim = down-dip range [m] relative to `point`. `point` may
                be model-local m, ([x, y, z], "model"), ([E, N, z], "utm")
                or ([lon, lat, z], "latlon") as before (resolved by
                fem.resolve_slice_positions). Needs the femtic_viz.py of
                the same date; vertical sections and other kinds unchanged.
                AI-generated; review before production use.

@author: vrath
    2026-11-10  Claude Sonnet 5.5 (Anthropic)
                Slice maps were transposed (x = easting and y = northing
                exchanged: a NNE structure was drawn ENE). Cause: since
                2026-10-11 femtic.read_femtic_mesh no longer swaps the
                node columns by default, but mesh.dat holds x = northing,
                y = easting, while femtic_viz, the sites from site.dat and
                the slice positions use a geographic frame. New setting
                MESH_SWAP_XY (default True) sets fem.MESH_SWAP_XY before the
                mesh is read; None leaves femtic's default (no swap), False
                forces it. Only the orientation changes; values, colours and
                the slice geometry are as before. Checked on a synthetic
                mesh with a N30E block, not on real data.
                AI-generated; review before production use.

    2026-11-11  Claude Sonnet 5.5 (Anthropic)
                Polylines. New PLOT_POLYLINES = {name: entry} (metres; {} =
                none): lines (faults, profiles, boundaries, ...) drawn on the
                slice panels. Maps show the polyline in plan view (optional
                z_range); vertical sections (ns, ew, plane, profile) show the
                parts within max_dist of the section plane, projected into it.
                Points may be model-local m or CRS-tagged ("utm" metres,
                "latlon" degrees, longitude first). Passed to
                fviz.plot_model_slices(polylines=...) with the other
                shared options. The intersection line of two slices can be
                computed with fviz.slice_intersection_polyline (comment under
                PLOT_POLYLINES). Needs the femtic_viz.py of the same date.
                With PLOT_POLYLINES = {} the figures are unchanged. Tested on
                a synthetic mesh only.
                AI-generated; review before production use.

    2026-11-11b Claude Sonnet 5.5 (Anthropic)
                Titles and colour bar position. FIGURE_TITLE (and
                BATCH_TITLE) now take "auto" (block-file name; None keeps
                that meaning here), "" / False (none, default), text with
                placeholders ({label}, {tag}, {iter}, {nrms}, {alpha}, {file},
                {key}) or a dict per model (resolve_title in femtic_viz.py);
                new PANEL_TITLES (default True) and the colour bar settings
                CBAR_LOCATION ("panel" = one per panel, as before; "right" /
                "left" / "top" / "bottom" = one joint bar), CBAR_SHRINK,
                CBAR_THICKNESS [cm], passed to fviz.plot_model_slices as
                panel_titles, cbar_location, cbar_shrink, cbar_thickness. The
                defaults leave the figures unchanged (a title given through
                BATCH_TITLE is now filled in by the same function; "{nrms}"
                still shows 3 decimals). Needs the femtic_viz.py of the same
                date.
                AI-generated; review before production use.

    2026-11-11c Claude Sonnet 5.5 (Anthropic)
                Convergence file from a list of names. CNV_FILE is now a name
                or a list of candidate names (default ["femtic.cnv",
                "FEMTIC-DABIC.cnv"]; the first one that exists is used, names
                are also matched ignoring case; a full path still works),
                no longer WORK_DIR + "femtic.cnv". It is passed on to
                fem.resolve_model_files() (cnv_name) and, for the nRMS /
                alpha annotation of a single model, searched first in the
                directory of the model file, then in WORK_DIR. The last row
                is read by column name (fem.read_cnv_last()); the setting
                DISTORTION and the guess from the number of columns (10 =
                distortion) are removed, since the header row decides. Needs
                the femtic.py of the same date.
                AI-generated; review before production use.

    2026-11-12  Claude Sonnet 5.5 (Anthropic)
                Fault from two surface points. kind="profile" takes an
                optional `dip` (default 90 = vertical fence as before): with
                dip != 90 it is the dipping plane through p1 -> p2 (dips to
                the right of p1 -> p2). Any slice entry may carry `overlay`
                (True or Line2D keywords, + label): that plane is also drawn
                on every other panel (trace on maps, dipping line on
                sections). Documentation block and commented example under
                PLOT_SLICES. Needs femtic.py and femtic_viz.py of the same
                date. Entries without `dip` / `overlay`: figures unchanged.
                Tested on a synthetic mesh only.
                AI-generated; review before production use.

    2026-11-12  Claude Sonnet 5.5 (Anthropic)
                Fault trace without a fault panel: `panel=False` on an
                entry with `overlay` makes it no panel, it only draws its
                line on all the other panels. Documentation block and
                commented example under PLOT_SLICES extended. Needs
                femtic_viz.py of the same date. Tested on a synthetic mesh
                only.
                AI-generated; review before production use.
"""

import os
import sys
from pathlib import Path
import gc
import inspect
import numpy as np

# ---------------------------------------------------------------------------
# Py4MTX-specific settings and imports
# ---------------------------------------------------------------------------
PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

for _base in [PY4MTX_ROOT + "/py4mt/modules/"]:
    for _p in [Path(_base), *Path(_base).rglob("*")]:
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

from version import versionstrg
import util as utl
import femtic as fem

try:
    import femtic_viz as fviz
except ImportError:
    fviz = None

version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())
titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng + "\n\n")

# ===========================================================================
# Configuration
# ===========================================================================

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORK_DIR = r"/home/vrath/MT_Data/Ubinas/ubinas_10_LC/1_L2/"
#: Resistivity block to display (any iteration).
MODEL_FILE = WORK_DIR + "resistivity_block_iter21.dat"
PLOT_FILE = WORK_DIR + "UbiSlice_alpha_10.pdf"


# WORK_DIR = r"/home/vrath/MT_Data/Ubinas/ubinas_10_LC/05_L2/"
# #: Resistivity block to display (any iteration).
# MODEL_FILE = WORK_DIR + "resistivity_block_iter26.dat"
# PLOT_FILE = WORK_DIR + "UbiSlice_alpha_05.pdf"

# WORK_DIR = r"/home/vrath/MT_Data/Ubinas/ubinas_10_LC/105_L2/"
# #: Resistivity block to display (any iteration).
# MODEL_FILE = WORK_DIR + "resistivity_block_iter13.dat"
# PLOT_FILE = WORK_DIR + "UbiSlice_alpha_15.pdf"

# WORK_DIR = r"/home/vrath/MT_Data/Ubinas/ubinas_10_LC/2_L2/"
# #: Resistivity block to display (any iteration).
# MODEL_FILE = WORK_DIR + "resistivity_block_iter20.dat"
# PLOT_FILE = WORK_DIR + "UbiSlice_alpha_20.pdf"


# ---------------------------------------------------------------------------
# Several models in one run (batch) - optional
# ---------------------------------------------------------------------------
#: MODEL_FILE above may also be a list of block files, a glob pattern, or run
#: directories (a directory, a list, or a glob such as WORK_DIR + "../ens_*").
#: For run directories MODEL_ITER chooses the block file:
#:   "last" - iteration of the last row of femtic.cnv (best-fit model of a
#:            converged run)
#:   "best" - iteration with the lowest nRMS in femtic.cnv
#:   N      - resistivity_block_iterN.dat (0 = starting model / perturbed prior)
#:   a list - e.g. [0, "last"]: one figure per entry and run
#: MODEL_ITER does not apply to block files given as files.
MODEL_ITER = "last"

#: With more than one model: one figure per model, named
#: <label>_<tag>_slices.<ext> (label = run directory, tag = "iterN").
#: Directory for the figures; None = the model's own directory.
BATCH_OUT_DIR = None
#: Figure formats (without dot), e.g. ("pdf", "png"); None = the extension of
#: PLOT_FILE.
BATCH_FORMATS = None
#: Multi-page PDF with one page per model: None = none; "auto" =
#: <BATCH_OUT_DIR or the common parent directory>/catalog_slices.pdf; or a
#: path. Needs "pdf" among the formats.
BATCH_CATALOG = None
#: Figure title template of the batch; {label}, {iter}, {tag}, {nrms} (and
#: {alpha}, {file}) are filled in, as for FIGURE_TITLE (which takes its place
#: when BATCH_TITLE is None or "").
BATCH_TITLE = "{label} - iteration {iter}"

#: Mesh file — always required for plotting.
MESH_FILE = WORK_DIR + "mesh.dat"


#: x/y column swap applied when the mesh is read (femtic.read_femtic_mesh).
#: FEMTIC stores mesh.dat nodes as x = northing, y = easting, z down; the
#: plotting code (femtic_viz, sites from site.dat, slice positions) works in
#: a geographic frame (x = easting, y = northing). True (default) swaps the
#: two columns on read, which gives that frame: maps, curtains and sites then
#: line up, and strikes come out with the right bearing. False draws the
#: model transposed (a N30E structure appears as N60E, x/y exchanged); it is
#: only for meshes that were already written as x = easting, y = northing.
#: None = leave femtic's module default (False) untouched.
MESH_SWAP_XY = True
if MESH_SWAP_XY is not None:
    fem.MESH_SWAP_XY = bool(MESH_SWAP_XY)

#: observe.dat — used as fallback site-overlay source when SITE_DAT is None
#: and SITE_NUMBER is not None.
OBSERVE_FILE = WORK_DIR + "observe.dat"

#: Site list produced by mt_make_sitelist.py (WHAT_FOR="femtic").
#: Format (comma-separated, no header):
#:   name, lat, lon, elev, sitenum, easting, northing
#: Easting/northing are UTM metres; model-local x/y is derived via
#: fem.utm_to_model using the mesh-centre origin.
#: Set to None to fall back to the observe.dat / SITE_NUMBER path.
SITE_DAT = WORK_DIR + "site.dat"   # set to None to disable
#: Name of the FEMTIC convergence file: one name or a list of candidate names
#: (the first one that exists is used; names are also matched ignoring upper /
#: lower case). Searched in each run directory (MODEL_FILE = run directories)
#: and, for one model file, in the directory of that file, then in WORK_DIR.
#: A full path is also accepted. Same setting as in all femtic_* scripts.
CNV_FILE = ["femtic.cnv", "FEMTIC-DABIC.cnv"]

# ---------------------------------------------------------------------------
# Verbose output
# ---------------------------------------------------------------------------
OUT = True

# ---------------------------------------------------------------------------
# Ocean / air handling (must match the inversion setup)
# ---------------------------------------------------------------------------
#: None → auto-infer from region 1 heuristic (ρ ≤ 1 Ω·m AND flag==1).
#: True / False → force ocean-present / ocean-absent.
OCEAN = None

AIR_RHO   = 1.0e9   # Ω·m  (region 0)
OCEAN_RHO = 0.25    # Ω·m  (region 1 when treated as ocean)

# ---------------------------------------------------------------------------
# Geographic / UTM origin of the mesh centre
# ---------------------------------------------------------------------------
#: Geographic coordinates (WGS-84) of the FEMTIC mesh origin.
UTM_ORIGIN_LAT = None   # decimal degrees, positive = North
UTM_ORIGIN_LON = None   # decimal degrees, positive = East

#: UTM coordinates of the mesh origin in metres.
UTM_ORIGIN_E = None   # easting  [m]
UTM_ORIGIN_N = None   # northing [m]

#: Override the auto-derived UTM zone number.  None = auto from origin lat/lon.
UTM_ZONE_OVERRIDE = None

# ---------------------------------------------------------------------------
# Display coordinate system
# ---------------------------------------------------------------------------
#: "model"  — axis ticks in model-local metres (origin = 0, default)
#: "utm"    — axis ticks in absolute UTM metres
#: "latlon" — axis ticks in decimal degrees (lon for easting, lat for northing)
DISPLAY_COORDS = "model"

# ---------------------------------------------------------------------------
# Site overlay
# ---------------------------------------------------------------------------
#: Primary source: site names to overlay from SITE_DAT.
#: May be a single string or a list of strings matching the "name" column.
#: Set to None to overlay all sites in the file.
SITE_NAMES = None   # e.g. ["MT01", "MT05", "MT12"]  or None = all sites

#: Fallback source (used when SITE_DAT is None): site number(s) to
#: extract from observe.dat (integer, 1-based).
#: May be a single int or a list of ints.  Set to None to skip site overlay.
SITE_NUMBER = [5, 6, 7]

#: Show site markers on horizontal map panels.
PLOT_SITES_MAPS = False
#: Show site markers on vertical curtain (ns/ew) and plane panels.
PLOT_SITES_SLICES = False

#: Maximum distance [m] from a vertical slice plane for a site to be plotted
#: on that panel.  None = plot all sites on every panel regardless of distance.
PROJECTION_DIST = 1000.   # e.g. 5000.0  (5 km)

#: Marker style for map panels (inverted triangle = MT convention).
SITE_MARKER = dict(marker="v", color="black", ms=4, zorder=10,
                   label=None)   # label filled in automatically

#: Marker style for curtain and plane panels.
SITE_MARKER_SLICES = dict(marker="v", color="black", ms=4, zorder=10,
                           label=None)

# ---------------------------------------------------------------------------
# Additional map markers
# ---------------------------------------------------------------------------
#: Arbitrary point markers overlaid on map panels only.
#: Each entry is a dict with:
#:   "latlon"  : [lat_deg, lon_deg]  — position in geographic coordinates
#:   "marker"  : Matplotlib marker string  (e.g. "P", "+", "x", "*", "^")
#:   "color"   : colour string
#:   "ms"      : marker size in points
#:   "name"    : label string (shown in legend); None = no legend entry
#: Any additional Matplotlib plot kwargs (mew, mfc, zorder, …) are accepted.

# Ubinas
MAP_MARKERS = [
    # dict(latlon=[-16.3169, -70.9673], marker="x", color="red", ms=8,
    #      name="test point, 4457m"),
    dict(latlon=[-16.3450, -70.8972], marker="*", color="red", ms=8,
         name="ubinas crater"),
    # dict(latlon=[-16.363436, -70.868025], marker="+", color="blue", ms=8,
    #      name="mesh origin, borehole1"),
    # dict(latlon=[-16.351, -70.9016], marker="^", color="magenta", ms=6,
    #      name="max elev"),
]

## Misti
# MAP_MARKERS = [
#     dict(latlon=[-16.2991, -71.4056], marker="*", color="red", ms=12,
#          name="misti crater"),
#     dict(latlon=[-16.248170, -71.506431], marker="+", color="blue", ms=12,
#          name="mesh origin, borehole1"),
#     # dict(latlon=[-16.351, -70.9016], marker="^", color="magenta", ms=6,
#     #      name="max elev"),
# ]

# ---------------------------------------------------------------------------
# Model-centre marker
# ---------------------------------------------------------------------------
#: Mark the model origin (model-local x=0, y=0) on every "map" panel
#: whenever DISPLAY_COORDS is "utm" or "latlon", so the mesh centre stays
#: identifiable once the axes are relabelled in UTM / lat-lon.  No effect
#: for DISPLAY_COORDS="model" (the origin is trivially the (0, 0) point
#: there already).  Default style: black "+", ms=10.  To override the
#: style, add a dict with "is_model_centre": True to MAP_MARKERS, e.g.
#:   dict(is_model_centre=True, marker="+", color="blue", ms=12)
#: Set False to force the marker off even in UTM/latlon display.
SHOW_MODEL_CENTRE = True

# ---------------------------------------------------------------------------
# Plotting — requires femtic_viz and Matplotlib
# ---------------------------------------------------------------------------

#: Figure DPI for saved file.
PLOT_DPI = 600
#: Matplotlib colormap name.
PLOT_CMAP = "turbo_r"
#: Colour limits [log10(ρ_min), log10(ρ_max)] — None = auto.
PLOT_CLIM = [-0., 3.0]      # log10(Ω·m)
#: Flat colour for ocean / lake cells.  None → use colormap.
PLOT_OCEAN_COLOR = "lightgrey"
#: Flat colour for air polygons on slice panels.  None → blank gaps.
PLOT_AIR_COLOR = "whitesmoke"
#: Axes facecolor for air / background.  None = figure default.
PLOT_AIR_BGCOLOR = None
#: Font size for axis tick labels (main panels and colourbar ticks).
TICK_FONTSIZE = 12
#: Font size for axis labels, panel titles, and colourbar label.
LABEL_FONTSIZE = 14

# ---------------------------------------------------------------------------
# Convergence annotation  (reads femtic.cnv)
# ---------------------------------------------------------------------------
#: Figure title (above all panels).
#:   ""  or False      no title (default).
#:   "auto", None, True the automatic title: the block-file name (batch: see
#:                     BATCH_TITLE).  None keeps its old meaning in this script.
#:   a string          used as given.  Placeholders, filled in per model:
#:                     {label} run directory name, {tag} "iterN", {iter} the
#:                     iteration ("?" if unknown), {nrms} nRMS (3 decimals; a
#:                     number, so {nrms:.2f} works; "n/a" if unknown), {alpha},
#:                     {file} block-file name, {key} = {label}.  Text with
#:                     other braces is used as it stands.
#:   a dict            one entry per model, keyed by its label, e.g.
#:                       {"run1": "Reference", "default": "auto"}
#:                     values as above; "default" (else "auto") covers the rest.
#: Several models (batch): BATCH_TITLE, when set, is used instead.
FIGURE_TITLE = ""
#: Panel titles ("Map  z = 1.0 km", "N-S  easting = 0.0 km", ...): True = as
#: before, False = none.  One panel is titled by adding title="text" to its
#: PLOT_SLICES entry (title="" leaves just that panel untitled).
PANEL_TITLES = True

# ---------------------------------------------------------------------------
# Colour bar
# ---------------------------------------------------------------------------
#: "panel"  one colour bar next to every panel (as before);
#: "right", "bottom", "top", "left"
#:          ONE colour bar for all panels of a figure (they share one colour
#:          scale), at that side of the whole panel grid.
CBAR_LOCATION = "panel"
#: Joint colour bar only: its length as a fraction of the side of the panel
#: grid it runs along (0 < value <= 1), and its thickness in cm (None =
#: 1.5 x LABEL_FONTSIZE points).
CBAR_SHRINK = 0.7
CBAR_THICKNESS = None  # cm
#: Panel index (0-based) on which to place the nRMS annotation.
#: None → first panel.
CNV_PANEL = None

# ---------------------------------------------------------------------------
# Alpha / blanking by second block file  (optional)
# ---------------------------------------------------------------------------
#: Path to a second resistivity_block_iterX.dat with the SAME mesh and
#: region structure as MODEL_FILE.  Its region_rho values are interpreted
#: as log10 weights.  Cells with weight < ALPHA_BLANK_THRESH are suppressed.
#: Set to None to disable.
ALPHA_FILE = None   # e.g. WORK_DIR + "sensitivity_log10.dat"

#: How alpha weights drive visibility:
#:   "fade"  — proportional transparency between ALPHA_BLANK_THRESH (=0) and 0
#:   "blank" — hard cutoff: below ALPHA_BLANK_THRESH the polygon is omitted
ALPHA_MODE = "fade"

#: Log10 threshold at or below which polygons are blanked / fully faded.
ALPHA_BLANK_THRESH = 0.0

# ---------------------------------------------------------------------------
# Slice specification
# ---------------------------------------------------------------------------
#: List of slice dicts — one per panel (left to right).
#:
#: Slices use exact tetrahedron-plane intersection (no selection slab).
#: Each dict must contain:
#:   kind   : "map"     — horizontal slice at z = z0
#:            "ns"      — N-S vertical section at x = x0   (y vs depth)
#:            "ew"      — E-W vertical section at y = y0   (x vs depth)
#:            "plane"   — arbitrary plane by strike / dip / point
#:            "profile" — fence section defined by two endpoints (vertical,
#:                        or dipping with ``dip``: a fault from two surface
#:                        points); strike is computed from the p1->p2 azimuth
#:   z0     : (map     only)  depth in metres
#:   x0     : (ns      only)  easting — plain float = model-local metres;
#:            or (value, "utm") / (value, "latlon") for CRS tagging
#:   y0     : (ew      only)  northing — plain float / CRS tuple
#:   point  : (plane   only)  [x, y, z] any point on the plane
#:            or ([lon, lat, z], "latlon") / ([E, N, z], "utm")
#:   p1, p2 : (profile only)  endpoint position specs; each accepts:
#:              [x, y]                  model-local metres (bare list)
#:              ([x, y], "model")       explicit model-local
#:              ([E, N], "utm")         UTM easting / northing [m]
#:              ([lon, lat], "latlon")  decimal degrees
#:            A 3-element [x, y, z] variant is also accepted (z ignored).
#:   z_top  : (profile only)  shallowest depth of fence panel [m, z-down]
#:            default 0.0; for dip != 90 also the depth at which the plane
#:            contains the trace p1-p2 (-elevation for a trace above sea level)
#:   z_bot  : (profile only)  deepest depth of fence panel [m, z-down]
#:            default 20 000 m
#:   dip    : (profile only)  0 < dip < 180, default 90 (vertical). The plane
#:            dips to the RIGHT of p1 -> p2; dip = 180 - d dips d degrees to the
#:            left (same as swapping p1 and p2)
#:   overlay: (any kind)  True or Line2D keywords (+ label): also draw this
#:            slice as a line on every other panel (fault trace on maps,
#:            dipping line on sections)
#:   panel  : (any kind)  False = not a panel: the entry only supplies its
#:            overlay line (fault trace in all the slices, no fault panel).
#:            Default True. At least one entry must stay a panel.
#:   strike : (plane   only)  azimuth of the strike line, degrees clockwise
#:            from North (every dip)
#:   dip    : (plane   only)  degrees below the horizontal, 0..180; the plane
#:            dips towards strike + 90 (90 = vertical section)
#:   xlim   : [xmin, xmax] : easting axis limit; for a plane: along-strike
#:            range [m] relative to ``point``
#:   ylim   : [ymin, ymax] : northing axis limit; for a plane with dip != 90:
#:            down-dip range [m] relative to ``point``
#:   zlim   : [zmin, zmax] — depth axis limit (ns/ew/profile panels)
#:   invert_x : True → flip the horizontal axis left-to-right after rendering
#:   title  : optional string override
#:
#: Per-panel xlim/ylim/zlim override PLOT_XLIM/PLOT_YLIM/PLOT_ZLIM.
##: MISTI
# PLOT_SLICES = [
#     dict(kind="ns",  x0=(-71.536322, "latlon")),
#     dict(kind="ew",  y0=(-16.196900, "latlon")),
#     dict(kind="map", z0=-4000.0),
#     dict(kind="map", z0= 10000.0),
#     # Two-point vertical profile example (uncomment to use):
#     # dict(kind="profile",
#     #      p1=([-71.536, -16.197], "latlon"),
#     #      p2=([-71.406, -16.299], "latlon"),
#     #      z_top=0.0, z_bot=25000.0,
#     #      title="NW-SE profile"),
# ]

##: UBINAS [-16.3450, -70.8972]
PLOT_SLICES = [
    # dict(kind="ns",  x0=(-70.8972, "latlon")),
    dict(kind="ew",  y0=(-16.3450, "latlon"), title="Ubinas crater"),
    # dict(kind="map", z0=-4000.0),
    # dict(kind="map", z0= 5000.0),
    # Fault from two surface points + dip, drawn as a panel (the model on the
    # plane) and, with overlay, as a line on every other panel (uncomment to
    # use). With panel=False there is no fault panel, only its trace in all
    # the other slices:
    # dict(kind="profile",
    #      p1=([-71.536, -16.197], "latlon"),
    #      p2=([-71.406, -16.299], "latlon"),
    #      dip=70.0, z_top=0.0, z_bot=25000.0,
    #      overlay=dict(color="k", linewidth=1.5), title="fault",
    #      panel=False),
    # Two-point vertical profile example (uncomment to use):
    # dict(kind="profile",
    #      p1=([-71.536, -16.197], "latlon"),
    #      p2=([-71.406, -16.299], "latlon"),
    #      z_top=0.0, z_bot=25000.0,
    #      title="NW-SE profile"),
]


#: Global axis limits in model-local metres.  None → auto.
PLOT_XLIM = [-15000., 15000.]   # [xmin, xmax] metres — easting
PLOT_YLIM = [-15000., 15000.]   # [ymin, ymax] metres — northing
PLOT_ZLIM = [-6000.,  15000.]   # [zmin, zmax] metres — depth (z positive-down)

#: Polylines drawn on the slice panels (optional): faults, profiles, borehole
#: traces, licence boundaries, ... A dict {name: entry}; {} = none. An entry
#: is the point list itself or a dict with the keys
#:   points    [[x, y], ...] or [[x, y, z], ...], at least 2 rows, model-local
#:             metres; z = depth positive down (missing z = 0). CRS-tagged
#:             like the `point` of a plane: (pts, "model" | "utm" | "latlon")
#:             with "utm" = [E, N(, z)] UTM metres, "latlon" = [lon, lat(, z)]
#:             degrees (longitude first); z is metres in every case
#:   style     matplotlib Line2D keywords, e.g. dict(color="k", linestyle="--",
#:             linewidth=1.5); default 1.5 pt solid, colour cycled per entry
#:   label     legend text (lower right of the panel); default = the key,
#:             None = no legend entry
#:   kinds     where to draw it: "map", "ns", "ew", "plane" (also the
#:             "profile" panels), "vertical" (= ns + ew + plane) or a list of
#:             them; default = everywhere
#:   max_dist  metres; vertical sections only: the parts of the polyline
#:             within max_dist of the section plane are projected into it.
#:             Default PROJECTION_DIST (the whole polyline if that is None);
#:             0 = only parts lying in the section plane
#:   z_range   [zmin, zmax] metres; maps only: draw only on maps whose z0 lies
#:             in the range (default: plan view on every map)
#: Maps always show the whole polyline in plan view. The axis limits are
#: never changed by the polylines.
#: Intersection line of two slices (model-local metres; ready-made entries for
#: many pairs: fviz.slice_intersection_polylines):
#:   _p = fviz.slice_intersection_polyline(
#:       dict(kind="map", z0=5000.0), dict(kind="ns", x0=0.0),
#:       xlim=PLOT_XLIM, ylim=PLOT_YLIM, zlim=PLOT_ZLIM)
#:   PLOT_POLYLINES["map 5 km x ns 0"] = dict(
#:       points=_p, max_dist=0.0, z_range=[5000.0, 5000.0])
PLOT_POLYLINES = {}
# PLOT_POLYLINES = {
#     "fault F1": dict(
#         points=[[-4000., -3000., 500.], [0., 0., 2500.], [4000., 3000., 4500.]],
#         style=dict(color="white", linewidth=2.0), max_dist=1500.),
#     "profile": dict(points=([[-70.95, -16.40], [-70.85, -16.30]], "latlon"),
#                     kinds="map", style=dict(color="k", linestyle=":")),
# }

#: Equal aspect ratio for map and curtain panels.
PLOT_EQUAL_ASPECT = True

#: Display depth axis in km instead of metres on curtain / plane panels.
DEPTH_KM = True

#: Display horizontal axes in km when DISPLAY_COORDS is "model".
HORIZ_KM = True

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
#: Number of subplot rows and columns.  None → auto (1 row, len(PLOT_SLICES) cols).
PLOT_NROWS = 1 #None
PLOT_NCOLS = 1 #None

#: Panel height in cm.
PLOT_PANEL_HEIGHT = 16.0   # cm

#: Fixed panel width in cm.  None → auto from aspect ratio.
PLOT_PANEL_WIDTH = None

#: Full figure size [width, height] in cm.  Overrides auto-sizing when set.
PLOT_FIGSIZE = None   # e.g. [40., 25.]

# ---------------------------------------------------------------------------
# Mesh-centre estimation from site.dat UTM coordinates  (optional)
# ---------------------------------------------------------------------------
#: Method used to estimate UTM_ORIGIN_E / UTM_ORIGIN_N from SITE_DAT:
#:   None      — use the hard-coded values above
#:   "box"     — midpoint of the UTM bounding box of all sites in SITE_DAT
#:   "average" — arithmetic mean of all site UTM coordinates in SITE_DAT
ORIGIN_METHOD = "box"   # None | "box" | "average"


# ===========================================================================
# Main
# ===========================================================================

# --- (0) Resolve the model(s) to plot --------------------------------------
MODELS = fem.resolve_model_files(
    MODEL_FILE, MODEL_ITER,
    cnv_name=CNV_FILE if CNV_FILE else fem.CNV_DEFAULT_NAMES,
    out=OUT)
if not MODELS:
    sys.exit(f"No model found for MODEL_FILE = {MODEL_FILE!r} "
             f"(MODEL_ITER = {MODEL_ITER!r}).")
BATCH = len(MODELS) > 1
print(f"{len(MODELS)} model(s) to plot"
      + (": one figure per model (batch)" if BATCH else ""))
print()

# --- (1) Optionally estimate UTM origin from site.dat ----------------------
if ORIGIN_METHOD is not None:
    if SITE_DAT is None or not os.path.isfile(SITE_DAT):
        print(f"  WARNING: ORIGIN_METHOD={ORIGIN_METHOD!r} requested but "
              f"SITE_DAT is not available — using hard-coded origin.")
    else:
        _sdat = fem.read_site_dat(SITE_DAT)
        if not _sdat:
            print(f"  WARNING: SITE_DAT is empty — using hard-coded origin.")
        else:
            _Es = np.array([d["easting"]  for d in _sdat])
            _Ns = np.array([d["northing"] for d in _sdat])
            if ORIGIN_METHOD == "box":
                UTM_ORIGIN_E = 0.5 * (_Es.min() + _Es.max())
                UTM_ORIGIN_N = 0.5 * (_Ns.min() + _Ns.max())
            elif ORIGIN_METHOD == "average":
                UTM_ORIGIN_E = float(_Es.mean())
                UTM_ORIGIN_N = float(_Ns.mean())
            else:
                sys.exit(f"Unknown ORIGIN_METHOD {ORIGIN_METHOD!r}; "
                         f"use None, 'box', or 'average'.")
            _lats = np.array([d["lat"] for d in _sdat])
            _lons = np.array([d["lon"] for d in _sdat])
            UTM_ZONE, UTM_NORTHERN = utl.utm_zone_from_latlon(
                float(_lats.mean()), float(_lons.mean()),
                override=UTM_ZONE_OVERRIDE)
            UTM_ORIGIN_LAT, UTM_ORIGIN_LON = utl.utm_to_latlon_zn(
                UTM_ORIGIN_E, UTM_ORIGIN_N, UTM_ZONE, UTM_NORTHERN)
            if OUT:
                print(f"Origin estimated ({ORIGIN_METHOD}, {len(_sdat)} sites):")
                print(f"  UTM_ORIGIN_E   = {UTM_ORIGIN_E:.1f} m")
                print(f"  UTM_ORIGIN_N   = {UTM_ORIGIN_N:.1f} m")
                print(f"  UTM_ORIGIN_LAT = {UTM_ORIGIN_LAT:.6f}°")
                print(f"  UTM_ORIGIN_LON = {UTM_ORIGIN_LON:.6f}°")
                print(f"  UTM_ZONE       = {UTM_ZONE}{'N' if UTM_NORTHERN else 'S'}")
                print()

# --- (2) Derive UTM zone from finalised mesh-origin coordinates ------------
UTM_ZONE, UTM_NORTHERN = utl.utm_zone_from_latlon(
    UTM_ORIGIN_LAT, UTM_ORIGIN_LON, override=UTM_ZONE_OVERRIDE)
hemi = "N" if UTM_NORTHERN else "S"
print(f"UTM zone: {UTM_ZONE}{hemi}  "
      f"(origin lat={UTM_ORIGIN_LAT:.4f}°, lon={UTM_ORIGIN_LON:.4f}°)")
print()

# --- (3) Resolve slice positions to model-local metres ---------------------
slices_resolved = fem.resolve_slice_positions(
    PLOT_SLICES, UTM_ZONE, UTM_NORTHERN,
    UTM_ORIGIN_E, UTM_ORIGIN_N,
    UTM_ORIGIN_LAT, UTM_ORIGIN_LON,
    verbose=OUT,
)
if OUT:
    print()

# --- (4) Optionally read site position(s) ----------------------------------
site_xys = []
_sites_from_obs = False
_need_sites = PLOT_SITES_MAPS or PLOT_SITES_SLICES
if _need_sites and SITE_DAT is not None:
    print(f"Reading site positions from site.dat: {SITE_DAT}")
    _rows = fem.read_site_dat(SITE_DAT, site_names=SITE_NAMES)
    for row in _rows:
        sx_m, sy_m = fem.utm_to_model(row["easting"], row["northing"],
                                      UTM_ORIGIN_E, UTM_ORIGIN_N)
        site_xys.append((row["name"], sx_m, sy_m, float(row.get("elev", 0.0))))
        print(f"  {row['name']}: model-local x = {sx_m / 1000:.3f} km,"
              f"  y = {sy_m / 1000:.3f} km")
        if DISPLAY_COORDS in ("utm", "latlon"):
            print(f"           UTM         E = {sx_m + UTM_ORIGIN_E:.1f} m,"
                  f"  N = {sy_m + UTM_ORIGIN_N:.1f} m")
    if not site_xys:
        print("  (no matching sites found in site.dat)")
    print()
elif _need_sites and SITE_NUMBER is not None:
    _site_nums = (SITE_NUMBER if isinstance(SITE_NUMBER, (list, tuple))
                  else [SITE_NUMBER])
    print(f"Reading site positions from observe.dat: {OBSERVE_FILE}")
    print(f"  Note: observe.dat provides model-local coordinates only.")
    for _sn in _site_nums:
        sx_m, sy_m = fem.read_site_position(OBSERVE_FILE, _sn)
        site_xys.append((_sn, sx_m, sy_m, 0.0))
        print(f"  site {_sn}: model-local x = {sx_m / 1000:.3f} km,"
              f"  y = {sy_m / 1000:.3f} km")
    _sites_from_obs = True
    print()

# --- (5) Plot 2-D slice panels --------------------------------------------
if fviz is None:
    sys.exit("femtic_viz not available — cannot plot.  Check your installation.")

# Options shared by every figure of this run.
_common = dict(
    mesh_file=MESH_FILE,
    slices=slices_resolved,
    cmap=PLOT_CMAP,
    clim=PLOT_CLIM,
    xlim=PLOT_XLIM,
    ylim=PLOT_YLIM,
    zlim=PLOT_ZLIM,
    ocean_color=PLOT_OCEAN_COLOR,
    ocean_value=OCEAN_RHO,
    air_color=PLOT_AIR_COLOR,
    air_bgcolor=PLOT_AIR_BGCOLOR,
    site_xys=site_xys,
    obs_coords_only=_sites_from_obs,
    projection_dist=PROJECTION_DIST,
    sites_in_maps=PLOT_SITES_MAPS,
    sites_in_slices=PLOT_SITES_SLICES,
    site_marker=SITE_MARKER,
    site_marker_slices=SITE_MARKER_SLICES,
    map_markers=MAP_MARKERS,
    show_model_centre=SHOW_MODEL_CENTRE,
    polylines=PLOT_POLYLINES,
    display_coords=DISPLAY_COORDS,
    utm_origin_e=UTM_ORIGIN_E,
    utm_origin_n=UTM_ORIGIN_N,
    utm_zone=UTM_ZONE,
    utm_northern=UTM_NORTHERN,
    utm_to_latlon_fn=utl.utm_to_latlon_zn,
    latlon_to_model_fn=fem.latlon_to_model,
    dpi=PLOT_DPI,
    equal_aspect=PLOT_EQUAL_ASPECT,
    depth_km=DEPTH_KM,
    horiz_km=HORIZ_KM,
    nrows=PLOT_NROWS,
    ncols=PLOT_NCOLS,
    panel_height=PLOT_PANEL_HEIGHT / 2.54,
    panel_width=PLOT_PANEL_WIDTH / 2.54 if PLOT_PANEL_WIDTH is not None else None,
    figsize=[v / 2.54 for v in PLOT_FIGSIZE] if PLOT_FIGSIZE is not None else None,
    alpha_file=ALPHA_FILE,
    alpha_mode=ALPHA_MODE,
    alpha_blank_thresh=ALPHA_BLANK_THRESH,
    tick_fontsize=TICK_FONTSIZE,
    label_fontsize=LABEL_FONTSIZE,
    panel_titles=bool(PANEL_TITLES),
    cbar_location=CBAR_LOCATION,
    cbar_shrink=CBAR_SHRINK,
    cbar_thickness=CBAR_THICKNESS / 2.54 if CBAR_THICKNESS is not None else None,
    out=OUT,
)


class _Number(float):
    """float whose plain str() / "{}" has 3 decimals ({nrms:.2f} still works)."""
    def __str__(self):
        return f"{float(self):.3f}"


def _figure_title(m, template, nrms=None, alpha=None):
    """Figure title of the model `m` (an entry of MODELS) from `template`
    (BATCH_TITLE or FIGURE_TITLE): see FIGURE_TITLE. Returns the argument
    `figure_title` of plot_model_slices (None = block-file name)."""
    spec = "auto" if template is None else template
    nrms = m.get("nrms") if nrms is None else nrms
    alpha = m.get("alpha") if alpha is None else alpha
    return fviz.resolve_title(
        spec, m["label"], None, name="FIGURE_TITLE",
        label=m["label"], tag=m["tag"],
        iter="?" if m.get("iter") is None else m["iter"],
        nrms="n/a" if nrms is None else _Number(nrms),
        alpha="n/a" if alpha is None else alpha,
        file=os.path.basename(str(m["file"])))

if not BATCH:
    # --- one model: PLOT_FILE; nRMS and alpha from the last row of the
    # convergence file (first of CNV_FILE found in the directory of the model
    # file, else in WORK_DIR); columns are taken by name from its header row.
    _nrms_annotation = None
    _cnv_path = None
    for _cdir in (os.path.dirname(os.path.abspath(str(MODELS[0]["file"]))),
                  WORK_DIR):
        _cnv_path = fem.find_cnv_file(_cdir, CNV_FILE)
        if _cnv_path is not None:
            break
    if _cnv_path is None:
        print(f"  cnv: none of {', '.join(fem.cnv_names(CNV_FILE))} found "
              f"-- nRMS annotation skipped.")
    else:
        _last = fem.read_cnv_last(_cnv_path)
        if _last is None:
            print(f"  Warning: could not read {_cnv_path} "
                  f"-- nRMS annotation skipped.")
        else:
            _nrms_val  = float(_last["RMS"])
            _alpha_val = float(_last["Alpha"])
            _nrms_annotation = dict(nrms=_nrms_val, alpha=_alpha_val,
                                    panel=CNV_PANEL)
            print(f"  cnv: {os.path.basename(str(_cnv_path))}: "
                  f"alpha={_alpha_val}, nRMS={_nrms_val:.4f}")

    print(f"Plotting model: {MODELS[0]['file']}")
    fviz.plot_model_slices_robust(
        PLOT_FILE,
        model_file=MODELS[0]["file"],
        nrms_annotation=_nrms_annotation,
        figure_title=_figure_title(
            MODELS[0], FIGURE_TITLE,
            nrms=_nrms_annotation["nrms"] if _nrms_annotation else None,
            alpha=_nrms_annotation["alpha"] if _nrms_annotation else None),
        **_common,
    )
    print("Slice plot done.")

else:
    # --- several models: one figure per model (+ optional catalog) ---------
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    _fmts = BATCH_FORMATS if BATCH_FORMATS is not None else (
        os.path.splitext(PLOT_FILE)[1].lstrip(".") or "pdf",)
    _fmts = [str(_f).lstrip(".") for _f in
             ((_fmts,) if isinstance(_fmts, str) else _fmts)]

    _cat_path = None
    if BATCH_CATALOG:
        if "pdf" not in _fmts:
            print("  catalog: \"pdf\" is not among the formats - skipped.")
        elif BATCH_CATALOG == "auto":
            _cat_path = os.path.join(
                BATCH_OUT_DIR or os.path.commonpath([_m["dir"] for _m in MODELS]),
                "catalog_slices.pdf")
        else:
            _cat_path = str(BATCH_CATALOG)
    if BATCH_OUT_DIR:
        os.makedirs(BATCH_OUT_DIR, exist_ok=True)
    _pdf = None
    if _cat_path:
        os.makedirs(os.path.dirname(os.path.abspath(_cat_path)), exist_ok=True)
        _pdf = PdfPages(_cat_path)
    _n_pages = 0
    try:
        for _i, _m in enumerate(MODELS, 1):
            _title = _figure_title(_m, BATCH_TITLE if BATCH_TITLE
                                   else FIGURE_TITLE)
            _ann = (dict(nrms=_m["nrms"], alpha=_m["alpha"], panel=CNV_PANEL)
                    if _m["nrms"] is not None else None)
            _stem = os.path.join(BATCH_OUT_DIR or _m["dir"],
                                 f"{_m['label']}_{_m['tag']}_slices")
            print(f"[{_i}/{len(MODELS)}] {_m['label']} / {_m['tag']}")
            for _fmt in _fmts:
                _fig = fviz.plot_model_slices_robust(
                    f"{_stem}.{_fmt}",
                    model_file=_m["file"],
                    nrms_annotation=_ann,
                    figure_title=_title,
                    **_common,
                )
                if _fmt == "pdf" and _pdf is not None:
                    _n_pages += bool(fviz.append_catalog_page(
                        _pdf, _fig, _m["label"]))
                del _fig
            gc.collect()   # Matplotlib figures hold reference cycles
    finally:
        if _pdf is not None:
            _info = _pdf.infodict()
            _info["Title"] = "femtic_mod_plot_slice catalog"
            _info["Author"] = "femtic_mod_plot_slice.py"
            _pdf.close()
    print(f"Slice plots done: {len(MODELS)} model(s)")
    if _cat_path:
        print(f"  catalog: {_n_pages} page(s) -> {_cat_path}")



# ---------------------------------------------------------------------------
# Parameter summary
# ---------------------------------------------------------------------------
utl.write_param_summary(fname, out_dir=WORK_DIR)
