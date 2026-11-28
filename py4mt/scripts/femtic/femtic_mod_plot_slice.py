#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_mod_plot_slice.py -- slice figures of single FEMTIC resistivity models
(mod family): horizontal maps, N-S / E-W curtains, arbitrary planes and fence
sections by exact tetrahedron-plane intersection (fviz.plot_model_slices).

The settings are those of femtic_ens_plot_slices.py (same MOD_* names, same
meaning, lengths in km) without everything that belongs to ensemble
statistics. Several runs can be given, as in femtic_mod_misfit.py: every entry
of RUN_PATHS is a model of its own (no ensemble) and gets its own figure.

Usage
-----
    python femtic_mod_plot_slice.py                       # RUN_PATHS
    python femtic_mod_plot_slice.py kra_rto_0 kra_rto_1   # two run directories
    python femtic_mod_plot_slice.py "kra*/"               # quote globs
    python femtic_mod_plot_slice.py -w /data/ann ann_rto_orig-err_0/resistivity_block_iter10.dat

Runs
----
An entry of RUN_PATHS (or of the command line, which replaces it) is a run
directory (the model of ITERATION), a block file, a results_iterN.h5 file
(= the block file of that iteration; the HDF5 file is not read) or a glob of
these. Relative entries start at WORK_DIR (-w / --work-dir replaces it; None
or "" = current directory). Quote globs so the shell does not expand them.

MOD_MESH, MOD_SITE_DAT, MOD_OBSERVE_FILE, MOD_REF_FILE and MOD_ALPHA_FILE are
looked up per run: a relative name in the run directory first, then in
WORK_DIR; an absolute path is used for every run. Mesh centre, sites and the
automatic region of interest are therefore determined run by run.

An entry without a model is reported and skipped. A model that fails is
reported and skipped too; the others are still plotted. Return code: 0 = all
models found were plotted, 1 = one failed or none found, 2 = WORK_DIR missing.

Output
------
Per model and MOD_PLOT_FORMAT entry:

    <dir>/[<OUT_PREFIX>_]<label>_<tag>_slices.<fmt>

label = run directory name, tag = "iterN"; <dir> = OUT_DIR, else
<run dir>/<OUT_PREFIX> (if OUT_PREFIX is set and OUT_PREFIX_DIR), else the run
directory. MOD_QC_FILE names the files of a single model explicitly;
MOD_CATALOG collects all figures in one multi-page PDF. With
MOD_SINGLE_PANELS = True every slice becomes a figure of its own
(..._slices_01_<kind>.<fmt>, ..._02_<kind>, ...). The parameter summary
femtic_mod_plot_slice_summary.md goes to WORK_DIR.

Slice positions
---------------
All lengths are km, UTM coordinates too.
x0 / y0: model-local km, or (value, "utm") [km] / (value, "latlon") [deg];
z0: km, positive down. "plane" entries: point in km (or tagged), strike, dip.
"profile" entries: p1, p2 (model-local km, UTM km or latlon), z_top, z_bot
in km, as in femtic_ens_plot_slices.py. See the comments at MOD_SLICES.

Needs femtic.py, femtic_viz.py and util.py (PY4MTX_ROOT, PY4MTX_DATA).

Sister scripts: femtic_mod_plot_bh.py (borehole logs), femtic_mod_plot_3d.py
(PyVista), femtic_mod_misfit.py (data misfit); ensemble counterpart:
femtic_ens_plot_slices.py.

AI-generated code (Claude Sonnet 5.5, Anthropic, 2026-11-26) -- review before
production use.

@author vrath
@project py4mt
@created 2026-05-06 (Claude Sonnet 4.6, Anthropic)
@modified 2026-11-26 (Claude Sonnet 5.5, Anthropic)

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

    2026-11-26  Claude Sonnet 5.5 (Anthropic)
                Rewritten with the settings of femtic_ens_plot_slices.py
                (without the ensemble statistics) and several runs as in
                femtic_mod_misfit.py. CHANGED: MODEL_FILE / PLOT_FILE /
                MODEL_ITER / BATCH_* -> RUN_PATHS, ITERATION (default "best"),
                command line and -w / --work-dir; OUT_DIR, OUT_PREFIX,
                OUT_PREFIX_DIR, MOD_QC_FILE, MOD_CATALOG, MOD_PLOT_FORMAT.
                PLOT_* and unprefixed settings are now MOD_*, and lengths are
                km (model-local positions, z0, limits, projection distance,
                polylines, markers). New: MOD_REF_*, MOD_ROI_*, MOD_VERT_EXAG,
                MOD_TICK_DECIMALS, MOD_SHOW_IN_SPYDER, main(). Figure titles
                are on by default (MOD_TITLE = "auto"). Mesh, site list,
                reference and alpha file, mesh centre and sites are per run.
                nRMS box and title use the row of the plotted iteration in the
                run's own convergence file (before: one model = last row, with
                WORK_DIR as fall-back). Fix: the broken WORK_DIR line raised
                NameError. Needs femtic.py and femtic_viz.py of the same date.
                Tested on synthetic files only.
                AI-generated; review before production use.

    2026-11-26b Claude Sonnet 5.5 (Anthropic)
                Optional single-panel output: new MOD_SINGLE_PANELS (default
                False) draws one figure per displayed slice
                (fviz.single_panel_jobs). RUN_PATHS now defaults to the glob
                "*_L2" (all run directories, ITERATION) instead of one block
                file. Needs femtic_viz.py of the same date. Tested on
                synthetic files only. AI-generated; review before production
                use.
    2026-11-26c Claude Sonnet 5.5 (Anthropic)
                Default settings replaced by those of
                femtic_ens_plot_slices.py; the former ones (carried over from
                the earlier script) did not fit the current surveys: WORK_DIR
                (Krafla ensemble), RUN_PATHS ["kra*/"], mesh, site list,
                observe and reference file in templates/, MOD_SLICES (maps at
                0 ... 5 km), MOD_MAP_MARKERS (model centre), UTM display,
                sites on maps, region of interest from the sites
                (MOD_ROI_AUTO, MOD_*LIM None), "jet_r" [0, 4], 3 x 3 layout,
                one colour bar at the bottom, font sizes, MOD_DPI 400. No
                change of the code. Tested on synthetic files only.
                AI-generated; review before production use.

2026-11-26 (km)  Claude Sonnet 5.5 (Anthropic)
              User settings are km throughout, UTM coordinates too (before:
              UTM and some per-entry values in metres). Personal settings in
              metres must be converted; values that look like metres stop
              with an error (fem.KmUnitError). Concerns MOD_UTM_ORIGIN_E /
              _N, (value, 'utm') positions, profile p1 / p2 / z_top / z_bot
              / xlim / ylim, per-panel limits and polylines. Needs femtic.py
              of the same date. Tested on synthetic files only.
              AI-generated; review before production use.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

for _base in [PY4MTX_ROOT + "/py4mt/modules/"]:
    for _p in [Path(_base), *Path(_base).rglob("*")]:
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

import femtic as fem
import femtic_viz as fviz
import util as utl
from version import versionstrg

version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng + "\n\n")

# ===========================================================================
# USER SECTION -- all user-set parameters below are UPPERCASE
# ===========================================================================
# ---------------------------------------------------------------------------
# Runs to plot (as in femtic_mod_misfit.py / femtic_ens_misfit.py)
# ---------------------------------------------------------------------------
#: Base directory: relative entries of RUN_PATHS and of the command line, and
#: relative OUT_DIR, MOD_CATALOG and MOD_QC_FILE start here; it is also the
#: second place where the per-run files (mesh, site list, ...) are looked for.
#: None or "" = current directory. Command line: -w / --work-dir.
WORK_DIR = r"/media/vrath/LargeBack/Ensembles/sabancaya2026/initial_nodist_restart/"
# WORK_DIR = r"/media/vrath/LargeBack/Ensembles/annecy2026/RTO/ann_rto_orig-err/"
# WORK_DIR = r"/media/vrath/LargeBack/Ensembles/krafla2026/krafla_fine/krafla_ens_fine_paired/"

#: Runs to plot: one entry or many, every model gets its own figure(s).
#:   a run directory   "1_L2"          -> the model of ITERATION in it
#:   a block file      "1_L2/resistivity_block_iter21.dat"
#:   a results file    "1_L2/results_iter21.h5" -> block file of iteration 21
#:   a glob of these   "*_L2"
#: Command-line arguments replace this list.
RUN_PATHS = [
    # "kra*/",                                 # every member run in WORK_DIR
    # single runs / block files instead:
    # "krafla_rto_0/",
    # "krafla_rto_0/resistivity_block_iter10.dat",
    r"/media/vrath/LargeBack/Ensembles/sabancaya2026/initial_nodist_restart/ini_*/"
]

#: Iteration taken from a run directory (not used for files):
#:   "best"  lowest nRMS in the convergence file
#:   "last"  last row of the convergence file
#:   N       resistivity_block_iterN.dat (0 = starting model / perturbed prior)
#:   a list  e.g. [0, "best"]: one figure per entry and run
ITERATION = "best"

#: Convergence file of a run: one name or candidates (first that exists; case
#: is ignored). Used for ITERATION and the nRMS annotation / title.
CNV_FILE = ["femtic.cnv", "FEMTIC-DABIC.cnv"]

# ---------------------------------------------------------------------------
# Mesh (required)
# ---------------------------------------------------------------------------
#: Relative names: looked for in each run directory, then in WORK_DIR.
MOD_MESH = "templates/mesh.dat"

#: Swap of the first two node-coordinate columns when reading MOD_MESH.
#: FEMTIC stores mesh.dat nodes as x = northing, y = easting, z down; the
#: plotting code (femtic_viz, sites, slice positions) works in a geographic
#: frame (x = easting, y = northing). True (default) swaps on read; False draws
#: the model transposed (only for meshes already written as x = easting).
#: None = leave femtic's module default (False) untouched.
MOD_MESH_SWAP_XY = True

# --- Ocean / air handling (must match the inversion setup) ----------------
MOD_OCEAN = None        # None = infer from the block file (reference model)
MOD_AIR_RHO = 1.0e9     # Ohm.m (region 0); information only
MOD_OCEAN_RHO = 0.25    # Ohm.m (region 1 when treated as ocean)

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
#: One or more file extensions ("pdf", "svg", "png", "jpg", "tif", ...); a
#: bare string works too. Every figure is saved once per entry.
MOD_PLOT_FORMAT = ["pdf"]

#: One directory for the figures of all runs (relative: to WORK_DIR), used as
#: given. None = per run, see OUT_PREFIX.
OUT_DIR = None
#: Put in front of every file name; with OUT_DIR None it also names a
#: sub-directory of each run directory (OUT_PREFIX_DIR). None = neither.
OUT_PREFIX = None
OUT_PREFIX_DIR = True   # False: with OUT_DIR None write into the run directory

#: Explicit output path WITHOUT extension (relative: to WORK_DIR), used only
#: when exactly one model is plotted. None = the naming scheme above.
MOD_QC_FILE = None      # e.g. "1_L2/UbiSlice_alpha_10"

#: Multi-page PDF with one page per figure: None = none; "auto" =
#: <OUT_DIR or the common parent of the run directories>/catalog_slices.pdf;
#: or a path. Needs "pdf" in MOD_PLOT_FORMAT.
MOD_CATALOG = None

# ---------------------------------------------------------------------------
# Geographic / UTM origin of the mesh centre
# ---------------------------------------------------------------------------
#: None when MOD_ORIGIN_METHOD estimates the origin from MOD_SITE_DAT.
MOD_UTM_ORIGIN_LAT = None   # decimal degrees, positive = North
MOD_UTM_ORIGIN_LON = None   # decimal degrees, positive = East
MOD_UTM_ORIGIN_E = None     # UTM easting  [km]
MOD_UTM_ORIGIN_N = None     # UTM northing [km]
MOD_UTM_ZONE_OVERRIDE = None  # override the auto-derived zone; None = auto

#: "box"     midpoint of the UTM bounding box of all sites in MOD_SITE_DAT
#: "average" arithmetic mean of the site UTM coordinates
#: None      use the literals above
MOD_ORIGIN_METHOD = "box"

# --- Display coordinate system ---------------------------------------------
#: "model" (axis ticks model-local) | "utm" | "latlon"
MOD_DISPLAY_COORDS = "utm"
#: Decimal digits of the axis tick labels; None = plot_model_slices' own.
MOD_TICK_DECIMALS = 0

# --- Site overlay ------------------------------------------------------------
#: Primary source: mt_make_sitelist.py CSV (name,lat,lon,elev,sitenum,E,N).
#: Set to None to fall back to MOD_OBSERVE_FILE / MOD_SITE_NUMBER.
MOD_SITE_DAT = "templates/site.dat"
#: Fallback source (used when there is no site list).
MOD_OBSERVE_FILE = "templates/observe.dat"
MOD_SITE_NAMES = None          # list of names to plot, or None = all sites
MOD_SITE_NUMBER = None    # site number(s) in observe.dat; None = none

MOD_PLOT_SITES_MAPS = True    # show markers on map panels
MOD_PLOT_SITES_SLICES = False  # show markers on curtain / plane panels
#: Max distance [km] from a curtain plane for a site to appear on it.
MOD_PROJECTION_DIST = 1.0      # km; None = all sites on every panel

MOD_SITE_MARKER = dict(marker="v", color="black", ms=8, zorder=10, label=None)
MOD_SITE_MARKER_SLICES = None

#: Point markers (epicentres, wells, ...): a list of dicts. Every marker is
#: drawn on all map panels; on the vertical sections those with a depth, within
#: max_dist of the section plane. Keys:
#:   latlon     [lat, lon], degrees
#:   depth      km, positive down
#:   depth_err  km, +- of a depth bar on the sections (err_color, err_lw,
#:              err_cap)
#:   kinds      "map", "ns", "ew", "plane", "vertical", "all" or a list;
#:              default "all" with a depth, else "map"
#:   max_dist   km (default MOD_PROJECTION_DIST)
#:   name       legend label; None = no label
#:   marker, color, ms, zorder and any matplotlib Line2D keyword
MOD_MAP_MARKERS = [
    # Epagny 1996 earthquake: 45.9383 N, 6.0883 E, depth 2 +- 1 km. Yellow
    # star on every map; on a section through it (or within max_dist) at 2 km
    # depth with a bar from 1 to 3 km. The Vuache fault plane (panel=True
    # below) passes ~0.2 km from it at 2 km depth, so it shows on that panel.
    # dict(latlon=[45.9383, 6.0883], marker="*", color="black",
    #      markerfacecolor="yellow", markeredgecolor="black", ms=30, mew=1.0,
    #      zorder=14, depth=2.0, depth_err=1.0, name="Epagny 1996 (2 +- 1 km)"),
    # # Model centre (model-local x = y = 0): a black cross on the maps and, at
    # depth 0, on the sections that pass within max_dist of it. Without this
    # entry the centre is the default yellow star on the maps only.
    dict(is_model_centre=True, marker="+", color="black", ms=16, mew=2.2,
         kinds="all", depth=0.0, name="model centre"),
]
#: Mark the model origin on every map panel when MOD_DISPLAY_COORDS is
#: "utm" / "latlon"; override the style by a MOD_MAP_MARKERS entry with
#: "is_model_centre": True. False = no centre marker.
MOD_SHOW_MODEL_CENTRE = True

# ---------------------------------------------------------------------------
# Slice specification
# ---------------------------------------------------------------------------
#: One dict per panel (left to right):
#:   kind   "map" | "ns" | "ew" | "plane" | "profile"
#:   z0     (map) depth, km, positive down
#:   x0     (ns)  model-local km, or (value, "utm") [km] / (value, "latlon")
#:   y0     (ew)  as x0
#:   point  (plane) [x, y(, z)] model-local km, or ([x, y, z], "model"),
#:          ([E, N, z], "utm") [all km],
#:          ([lon, lat, z], "latlon") [z km]
#:   strike (plane) azimuth, degrees clockwise from north
#:   dip    (plane) degrees below horizontal towards strike + 90; 90 = vertical
#:          xlim / ylim of a plane: along-strike / down-dip range, km relative
#:          to point
#:   p1, p2 (profile) end points: [x, y] model-local km,
#:          ([E, N], "utm") [km] or ([lon, lat], "latlon");
#:          z_top (default 0), z_bot (default 20) km; dip (default 90;
#:          else the plane dips to the right of p1 -> p2); limit_km,
#:          alpha_outside, fade_km limit a fault to the stretch p1 ... p2.
#:   overlay  (any kind) True or Line2D keywords (+ label): also draw this
#:            slice as a line on every other panel (fault trace)
#:   panel    (any kind) False = no panel, only the overlay line
#:   invert_x True = flip the horizontal axis
#:   title    panel title ("" = none)
#:   xlim, ylim, zlim (any kind) per-panel limits, km
#: All lengths are km (UTM coordinates too), as in femtic_ens_plot_slices.py.
MOD_SLICES = [
    # dict(kind="map", z0=-0.25),  # km
    dict(kind="map", z0=-2.0),  # km
    # dict(kind="map", z0=0.5),  # km
    dict(kind="map", z0=.0),  # km
    dict(kind="map", z0=5.0),  # km
    # dict(kind="map", z0=2.5),  # km
    dict(kind="map", z0=10.0),  # km
    dict(kind="map", z0=20.0),  # km
    dict(kind="map", z0=50.0),  # km
    # dict(kind="plane", point=[0.0, 0.0], strike=45.0, dip=90.0),
    # dict(kind="ns", x0=0.0),  # km
    # dict(kind="ew", y0=0.0),  # km
    #: Vertical section along a strike through ``point`` (model-local km,
    #: [0, 0] = model origin). strike = geographic azimuth, degrees
    #: clockwise from north: 45 = SW (left) -> NE (right), 135 = NW -> SE.
    #: Horizontal axis = along-strike distance from ``point`` in model km;
    #: its range follows MOD_XLIM x MOD_YLIM (projected onto the strike),
    #: its depth range MOD_ZLIM. Optional: xlim=[min, max] (km) to override.
    # dict(kind="plane", point=[0.0, 0.0], strike=45.0, dip=90.0),
    #: Dipping plane (dip < 90): strike = azimuth (deg clockwise from north)
    #: of the strike line, the plane dips towards strike + 90. ``point`` may
    #: be model-local km or CRS-tagged (z = depth in km, positive down):
    #:   point=[x, y, z]                       model-local km
    #:   point=([x, y, z], "model")            model-local km
    #:   point=([E, N, z], "utm")              UTM km, z in km
    #:   point=([lon, lat, z], "latlon")       degrees (longitude first)
    #: Axes = along-strike and down-dip distance from ``point`` (km if
    #: MOD_DEPTH_KM else m, true aspect, see the readme). Default extent = the plane cut
    #: with the MOD_XLIM x MOD_YLIM x MOD_ZLIM box; optional per-entry
    #: xlim=[min, max] (along strike) and ylim=[min, max] (down dip), km
    #: relative to ``point``.
    # dict(kind="plane", point=([6.12, 45.90, 0.0], "latlon"), strike=325.0, dip=70.0),
    # dict(kind="plane", point=([E_utm, N_utm, 2.0], "utm"), strike=325.0, dip=45.0),
    #: Fault from two surface points + dip (kind="profile"): the plane
    #: through p1 -> p2 that dips to the RIGHT of the direction p1 -> p2
    #: (dip=110 dips the same amount to the left; or swap p1 and p2). p1, p2:
    #: ([lon, lat], "latlon"), ([E, N], "utm") [km] or [x, y] model-local km.
    #: z_top = depth of the plane at the trace (default 0; -elevation for a
    #: trace above sea level), z_bot = bottom of the panel, both in km.
    #: overlay=True (or Line2D keywords, optionally label=...) also draws the
    #: fault on every other panel of every figure of this script: trace on the
    #: maps, dipping line on the sections. panel=False: the entry is NOT a
    #: panel, it only supplies that trace (the fault line in all the other
    #: slices, no fault panel); leave it out to get the fault panel (the
    #: model on the fault plane) as well. xlim=[a, b] (km along strike,
    #: relative to the midpoint of p1-p2) limits its length.
    #: Limit to the defining points (femtic_viz.py of 2026-11-15): the part of
    #: the fault between p1 and p2 is drawn as given, the rest -- the
    #: extrapolation along strike -- fades out. limit_km (km, default 0.0 for
    #: a fault, i.e. exactly p1 ... p2): > 0 widens that stretch at both ends,
    #: < 0 narrows it (e.g. 2.0 = 2 km beyond each point), None = no
    #: distinction (the line is drawn alike everywhere, as before).
    #: alpha_outside (0 ... 1, default 0.25): opacity far outside, relative
    #: to the line's own alpha; 0 = fully transparent, 1 = no difference.
    #: fade_km (km, default 2.0): the opacity falls linearly from 1 at the
    #: ends of the stretch to alpha_outside over this width, as
    #: MOD_ALPHA_MODE="fade" does for cells; 0 = abrupt step. All three also
    #: act on the fault panel (panel=True, the part outside is faded towards
    #: white).
    # dict(kind="profile",
    #      p1=([6.030869, 45.962166], "latlon"),
    #      p2=([6.115936, 45.910655], "latlon"),
    #      dip=110.0,  # = 70 deg to the NE (p1 -> p2 runs NW -> SE; dip < 90
    #                  # would dip to its right, i.e. SW). Third point of the
    #                  # same plane: (6.079571, 45.941367) at 2 km depth.
    #      z_top=0.0, z_bot=20.0,
    #      overlay=dict(color="k", linewidth=1.5), title="fault",
    #      limit_km=0.0, alpha_outside=0.25, fade_km=2.0,     # see above
    #      panel=False),
    # dict(kind="ns",  x0=(-71.40723, 'latlon')),    # km
    # dict(kind="ew",  y0=(-16.299593, 'latlon')),    # km
    #: Sections through the MOD_MAP_MARKERS examples (uncomment): the Epagny
    #: epicentre (star, depth bar) and the model centre (black cross).
    # dict(kind="ns", x0=(6.0883, "latlon")),    # N-S through the epicentre
    # dict(kind="ew", y0=(45.9383, "latlon")),   # E-W through the epicentre
    # dict(kind="ns", x0=0.0),                   # N-S through the model centre
    # dict(kind="ew", y0=0.0),                   # E-W through the model centre
    # dict(kind="profile",
    #  p1=([6.030869, 45.962166], "latlon"),
    #  p2=([6.115936, 45.910655], "latlon"),
    #  dip=110.0, z_top=0.0, z_bot=20.0,
    #  overlay=dict(color="magenta", linestyle=":", linewidth=10.0, alpha=0.9),
    #  title="Vuache fault",
    #  limit_km=0.0, alpha_outside=0.1,   # p1 ... p2 as given, then fading
    #  fade_km=2.0,                        # ... to 25 % over 2 km (0 = abrupt)
    #  panel=False)
]

# --- Polylines (optional) ----------------------------------------------------
#: Lines drawn on the slice panels: faults, profiles, boundaries, ... A dict
#: {name: entry}; {} = none. An entry is the point list or a dict:
#:   points    [[x, y(, z)], ...] model-local km (z = depth), or
#:             (pts, "utm") [E, N, z in km] / (pts, "latlon")
#:             [lon, lat, z km]
#:   style     Line2D keywords
#:   label     legend text (default = the key, None = no entry)
#:   kinds     "map", "ns", "ew", "plane", "vertical" or a list
#:   max_dist  km, vertical sections: parts within this distance are projected
#:   z_range   [zmin, zmax] km, maps only
MOD_POLYLINES = {}
# MOD_POLYLINES = {
#     "fault F1": dict(
#         points=[[-4.0, -3.0, 0.5], [0.0, 0.0, 2.5], [4.0, 3.0, 4.5]],
#         style=dict(color="white", linewidth=2.0), max_dist=1.5),
#     "profile": dict(points=([[-70.95, -16.40], [-70.85, -16.30]], "latlon"),
#                     kinds="map", style=dict(color="k", linestyle=":")),
# }

# --- Reference model (optional) ----------------------------------------------
#: Resistivity block file on the SAME mesh, drawn as iso-resistivity lines when
#: MOD_REF_SLICES is True. None = no reference.
MOD_REF_FILE = "templates/referencemodel.dat"     # e.g. "referencemodel.dat"
MOD_REF_SLICES = False
#: Contour levels, log10(Ohm.m); None = the integer decades inside MOD_CLIM.
MOD_REF_LEVELS = None
MOD_REF_STYLE = dict(colors="k", linewidths=0.8)
MOD_REF_LABEL = "reference model"

# --- Limits ------------------------------------------------------------------
MOD_XLIM = None   # [xmin, xmax] model-local km; None = auto
MOD_YLIM = None   # [ymin, ymax] model-local km; None = auto
MOD_ZLIM = None    # [zmin, zmax] km, positive down; None = auto

#: True: x / y limits from the site bounding box of each run + MOD_ROI_PAD_XY,
#: z limits from MOD_ROI_ZLIM (None = keep MOD_ZLIM). Without sites the limits
#: above are used.
MOD_ROI_AUTO = True
MOD_ROI_PAD_XY = 2.0       # km
MOD_ROI_ZLIM = [-1.0, 6.0]        # e.g. [-1.0, 6.0]

# --- Colours ------------------------------------------------------------------
MOD_DPI = 400
MOD_CMAP = "jet_r"
MOD_CLIM = [-1.0, 4.0]      # [log10_min, log10_max] Ohm.m; None = auto
MOD_OCEAN_COLOR = "lightgrey"   # None = colormap
MOD_AIR_COLOR = "whitesmoke"    # None = blank
MOD_AIR_BGCOLOR = None          # axes facecolor for air

# --- Convergence annotation ---------------------------------------------------
#: Box with nRMS and alpha of the plotted iteration (from the run's own
#: convergence file; no box if it has no row for that iteration).
MOD_NRMS_ANNOTATE = True
#: Panel with the box (0-based). None = first panel; with MOD_SINGLE_PANELS
#: None = every figure.
MOD_NRMS_PANEL = None

# --- Blanking / fading by a second block file (optional) ----------------------
#: Block file with the same mesh and region structure whose region values are
#: log10 weights (e.g. a sensitivity); None = off.
MOD_ALPHA_FILE = None      # e.g. "sensitivity_log10.dat"
MOD_ALPHA_MODE = "fade"    # "fade" | "blank"
MOD_ALPHA_BLANK_THRESH = 0.0

# --- Figure layout -------------------------------------------------------------
#: False = one figure per model with a panel per MOD_SLICES entry;
#: True  = one figure per model AND slice (..._slices_01_<kind>.<fmt>).
#: MOD_NROWS / MOD_NCOLS are then not used; MOD_FIGSIZE applies to each figure.
MOD_SINGLE_PANELS = False
MOD_EQUAL_ASPECT = True
#: Vertical exaggeration of the section panels; None = as MOD_EQUAL_ASPECT.
MOD_VERT_EXAG = None
MOD_DEPTH_KM = True
MOD_HORIZ_KM = True
MOD_NROWS = 3           # None = auto (1 row)
MOD_NCOLS = 3           # None = auto (len(MOD_SLICES) columns)
MOD_PANEL_HEIGHT = 20.0    # cm
MOD_PANEL_WIDTH = None     # cm; None = from the aspect ratio
MOD_FIGSIZE = None         # [w, h] cm; overrides auto sizing

MOD_TICK_FONTSIZE = 16
MOD_LABEL_FONTSIZE = 16
MOD_TITLE_FONTSIZE = MOD_LABEL_FONTSIZE + 10  # None = plot_model_slices' default

# --- Titles -------------------------------------------------------------------
#: Figure title:
#:   "auto"            block-file name; run, iteration and nRMS
#:   None, False or "" no title
#:   a string          used as given; placeholders {label} (= {run} = {key} =
#:                     {best}: run directory name), {tag}, {iter}, {nrms}
#:                     ({nrms:.2f} works), {alpha}, {file}, {prefix}
#:                     (OUT_PREFIX); "?" / "n/a" if unknown
#:   a dict            one entry per run label, e.g.
#:                       {"1_L2": "alpha 10", "default": "auto"}
MOD_TITLE = "auto"
#: Panel titles: True / False (one panel: title="..." in its MOD_SLICES entry).
MOD_PANEL_TITLES = True

# --- Colour bar ----------------------------------------------------------------
#: "panel" = one per panel; "right" / "left" / "top" / "bottom" = one joint bar.
MOD_CBAR_LOCATION = "bottom"
MOD_CBAR_SHRINK = 0.7
MOD_CBAR_THICKNESS = None  # cm

#: Show the figures inline when running under Spyder (files are saved anyway).
MOD_SHOW_IN_SPYDER = True

# ---------------------------------------------------------------------------
# Verbose output
# ---------------------------------------------------------------------------
OUT = True


# ===========================================================================
# Helpers
# ===========================================================================

class _Number(float):
    """float whose "{}" shows 3 decimals; "{nrms:.2f}" still works."""

    def __format__(self, spec):
        return float.__format__(self, spec or ".3f")


def _formats():
    """MOD_PLOT_FORMAT as a list of extensions without dot."""
    raw = ([MOD_PLOT_FORMAT] if isinstance(MOD_PLOT_FORMAT, str)
           else list(MOD_PLOT_FORMAT))
    fmts = [str(f).lstrip(".").lower() for f in raw]
    if not fmts:
        raise ValueError("MOD_PLOT_FORMAT is empty.")
    return fmts


def _figure_title(model):
    """Figure title of one model, see MOD_TITLE ("" = none)."""
    nrms, alpha, it = model.get("nrms"), model.get("alpha"), model.get("iter")
    block = os.path.basename(model["file"])
    info = [model["label"]]
    if it is not None:
        info.append(f"iter {it}")
    if nrms is not None:
        info.append(f"nRMS {nrms:.4f}")
    label = model["label"]
    return fviz.resolve_title(
        MOD_TITLE, label, f"{block}\n{'  '.join(info)}", name="MOD_TITLE",
        label=label, run=label, best=label, tag=model["tag"],
        iter="?" if it is None else it,
        nrms="n/a" if nrms is None else _Number(nrms),
        alpha="n/a" if alpha is None else alpha,
        file=block, prefix=OUT_PREFIX or "",
    )


def _crs_tags_used():
    """CRS tags needing the mesh-centre origin: {"utm", "latlon"} or less."""
    values = [spec.get(key) for spec in MOD_SLICES
              for key in ("x0", "y0", "point", "p1", "p2")]
    values += [entry.get("points") if isinstance(entry, dict) else entry
               for entry in (MOD_POLYLINES or {}).values()]
    tags = {
        v[1].lower() for v in values
        if isinstance(v, (list, tuple)) and len(v) == 2
        and isinstance(v[1], str)
    }
    tags.add(str(MOD_DISPLAY_COORDS).lower())
    return tags & {"utm", "latlon"}


def _optional_file(name, setting, run_dir, work_dir):
    """Per-run path of an optional file; a set but missing one is reported."""
    path = fem.find_run_file(setting, run_dir, work_dir)
    if setting and path is None:
        print(f"  {name} = {setting!r} not found in {run_dir} or {work_dir}.")
    return path


def _plot_model(model, work_dir, *, stem=None, catalog=None, show=False):
    """All figures of one model (an entry of fem.resolve_model_files()).

    stem : output path without extension (None = the naming scheme);
    catalog : open PdfPages or None; show : display the first format.
    Returns the number of catalog pages written.
    """
    run_dir = model["dir"]

    # --- per-run files -----------------------------------------------------
    mesh_file = fem.find_run_file(MOD_MESH, run_dir, work_dir)
    if mesh_file is None:
        raise FileNotFoundError(
            f"mesh MOD_MESH = {MOD_MESH!r} not found in {run_dir} or "
            f"{work_dir}.")
    site_dat = _optional_file("MOD_SITE_DAT", MOD_SITE_DAT, run_dir, work_dir)
    observe = fem.find_run_file(MOD_OBSERVE_FILE, run_dir, work_dir)
    alpha_file = None
    if MOD_ALPHA_FILE:
        alpha_file = fem.find_run_file(MOD_ALPHA_FILE, run_dir, work_dir)
        if alpha_file is None:
            raise FileNotFoundError(
                f"MOD_ALPHA_FILE = {MOD_ALPHA_FILE!r} not found in {run_dir} "
                f"or {work_dir}.")

    # --- mesh-centre origin and sites ---------------------------------------
    want_sites = MOD_PLOT_SITES_MAPS or MOD_PLOT_SITES_SLICES or MOD_ROI_AUTO
    if (want_sites and site_dat and MOD_ORIGIN_METHOD is None
            and (MOD_UTM_ORIGIN_E is None or MOD_UTM_ORIGIN_N is None)):
        print("  sites: no UTM origin (MOD_ORIGIN_METHOD = None, no "
              "MOD_UTM_ORIGIN_E / _N) - sites not used.")
        want_sites = False
    origin_e, origin_n = fem.utm_km_to_m(
        MOD_UTM_ORIGIN_E, MOD_UTM_ORIGIN_N, name="MOD_UTM_ORIGIN_E / _N")
    (utm_e, utm_n, utm_lat, utm_lon, utm_zone, utm_north, site_xys,
     obs_coords_only) = fem.resolve_origin_and_sites(
        site_dat=site_dat,
        site_names=MOD_SITE_NAMES,
        site_number=MOD_SITE_NUMBER,
        observe_file=observe,
        origin_e=origin_e,
        origin_n=origin_n,
        origin_lat=MOD_UTM_ORIGIN_LAT,
        origin_lon=MOD_UTM_ORIGIN_LON,
        zone_override=MOD_UTM_ZONE_OVERRIDE,
        origin_method=MOD_ORIGIN_METHOD,
        plot_sites=want_sites,
    )
    have_utm = utm_e is not None and utm_n is not None
    have_geo = have_utm and utm_zone is not None and utm_lat is not None
    tags = _crs_tags_used()
    if (tags and not have_utm) or ("latlon" in tags and not have_geo):
        raise ValueError(
            f"{' / '.join(sorted(tags))} coordinates in MOD_SLICES, "
            "MOD_POLYLINES or MOD_DISPLAY_COORDS need the mesh-centre origin: "
            "give MOD_SITE_DAT (with MOD_ORIGIN_METHOD) or "
            "MOD_UTM_ORIGIN_E / _N / _LAT / _LON.")
    if OUT and have_utm:
        zone_txt = (f", zone {utm_zone}{'N' if utm_north else 'S'}"
                    if have_geo else "")
        print(f"  mesh centre: E {utm_e / 1000.0:.4f} km, "
              f"N {utm_n / 1000.0:.4f} km{zone_txt}")

    # --- region of interest --------------------------------------------------
    xlim, ylim, zlim = MOD_XLIM, MOD_YLIM, MOD_ZLIM
    if MOD_ROI_AUTO and site_xys:
        sx = np.array([s[1] for s in site_xys], dtype=float) / 1000.0
        sy = np.array([s[2] for s in site_xys], dtype=float) / 1000.0
        xlim = [float(sx.min()) - MOD_ROI_PAD_XY,
                float(sx.max()) + MOD_ROI_PAD_XY]
        ylim = [float(sy.min()) - MOD_ROI_PAD_XY,
                float(sy.max()) + MOD_ROI_PAD_XY]
        if MOD_ROI_ZLIM is not None:
            zlim = MOD_ROI_ZLIM
        if OUT:
            print(f"  ROI: x {xlim[0]:.2f} ... {xlim[1]:.2f} km, "
                  f"y {ylim[0]:.2f} ... {ylim[1]:.2f} km")
    elif MOD_ROI_AUTO:
        print("  ROI: no sites available - MOD_XLIM / YLIM / ZLIM used.")

    # --- slice geometry (km -> m, positions -> model-local) ------------------
    # "plane" entries are converted here; all other kinds go through
    # fem.resolve_slice_positions. The order of MOD_SLICES is kept.
    others = [sp for sp in MOD_SLICES if sp.get("kind") != "plane"]
    others_res = iter(
        fem.resolve_slice_positions(
            fem.slices_km_to_m(others), utm_zone, utm_north, utm_e, utm_n,
            utm_lat, utm_lon, verbose=OUT)
        if others else []
    )
    slices = [
        fem.plane_spec_km_to_m(
            sp, xlim, ylim, zone=utm_zone, northern=utm_north,
            origin_E=utm_e, origin_N=utm_n)
        if sp.get("kind") == "plane" else next(others_res)
        for sp in MOD_SLICES
    ]

    # --- reference model, annotation -----------------------------------------
    ref_kw = {}
    if MOD_REF_SLICES and MOD_REF_FILE:
        ref_path = (fem.find_run_file(MOD_REF_FILE, run_dir, work_dir)
                    or fem.anchor_path(MOD_REF_FILE, run_dir))
        ref = fem.ReferenceModel(ref_path, ocean=MOD_OCEAN,
                                 ocean_rho=MOD_OCEAN_RHO)
        ref_kw = ref.slice_kwargs(
            model["file"], levels=MOD_REF_LEVELS, clim=MOD_CLIM,
            style=MOD_REF_STYLE, label=MOD_REF_LABEL)

    annotation = None
    if MOD_NRMS_ANNOTATE:
        if model["nrms"] is None:
            print("  no nRMS of this iteration in the convergence file - "
                  "annotation skipped.")
        else:
            annotation = dict(nrms=model["nrms"], alpha=model["alpha"],
                              panel=MOD_NRMS_PANEL)

    # --- figures ---------------------------------------------------------------
    if stem is None:
        stem = fem.model_output_stem(
            model, "slices", out_dir=OUT_DIR, prefix=OUT_PREFIX,
            prefix_dir=OUT_PREFIX_DIR, base_dir=work_dir)
    os.makedirs(os.path.dirname(os.path.abspath(stem)), exist_ok=True)
    title = _figure_title(model)

    kwargs = dict(
        model_file=model["file"],
        mesh_file=mesh_file,
        slices=slices,
        cmap=MOD_CMAP,
        clim=MOD_CLIM,
        xlim=fem.lim_km_to_m(xlim, name="MOD_XLIM"),
        ylim=fem.lim_km_to_m(ylim, name="MOD_YLIM"),
        zlim=fem.lim_km_to_m(zlim, name="MOD_ZLIM"),
        ocean_color=MOD_OCEAN_COLOR,
        ocean_value=MOD_OCEAN_RHO,
        air_color=MOD_AIR_COLOR,
        air_bgcolor=MOD_AIR_BGCOLOR,
        site_xys=site_xys,
        obs_coords_only=obs_coords_only,
        sites_in_maps=MOD_PLOT_SITES_MAPS,
        sites_in_slices=MOD_PLOT_SITES_SLICES,
        site_marker=MOD_SITE_MARKER,
        site_marker_slices=MOD_SITE_MARKER_SLICES,
        map_markers=fem.markers_km_to_m(MOD_MAP_MARKERS),
        show_model_centre=MOD_SHOW_MODEL_CENTRE,
        polylines=fem.polylines_km_to_m(MOD_POLYLINES),
        projection_dist=fem.km_to_m(
            MOD_PROJECTION_DIST, name="MOD_PROJECTION_DIST"),
        display_coords=MOD_DISPLAY_COORDS,
        # without an origin only model-local coordinates are in use (checked
        # above); lat/lon markers are then skipped by the plot function
        utm_origin_e=utm_e if have_utm else 0.0,
        utm_origin_n=utm_n if have_utm else 0.0,
        utm_zone=utm_zone if have_geo else 1,
        utm_northern=utm_north if have_geo else True,
        utm_to_latlon_fn=utl.utm_to_latlon_zn if have_geo else None,
        latlon_to_model_fn=fem.latlon_to_model if have_geo else None,
        depth_km=MOD_DEPTH_KM,
        horiz_km=MOD_HORIZ_KM,
        equal_aspect=MOD_EQUAL_ASPECT,
        vert_exag=MOD_VERT_EXAG,
        panel_height=MOD_PANEL_HEIGHT / 2.54,
        panel_width=(MOD_PANEL_WIDTH / 2.54
                     if MOD_PANEL_WIDTH is not None else None),
        figsize=([v / 2.54 for v in MOD_FIGSIZE]
                 if MOD_FIGSIZE is not None else None),
        nrows=MOD_NROWS,
        ncols=MOD_NCOLS,
        tick_fontsize=MOD_TICK_FONTSIZE,
        label_fontsize=MOD_LABEL_FONTSIZE,
        figure_title=title,
        figure_title_fontsize=MOD_TITLE_FONTSIZE,
        panel_titles=bool(MOD_PANEL_TITLES),
        cbar_location=MOD_CBAR_LOCATION,
        cbar_shrink=MOD_CBAR_SHRINK,
        cbar_thickness=(MOD_CBAR_THICKNESS / 2.54
                        if MOD_CBAR_THICKNESS is not None else None),
        tick_decimals=MOD_TICK_DECIMALS,
        alpha_file=alpha_file,
        alpha_mode=MOD_ALPHA_MODE,
        alpha_blank_thresh=MOD_ALPHA_BLANK_THRESH,
        nrms_annotation=annotation,
        **ref_kw,
        dpi=MOD_DPI,
    )
    if MOD_SINGLE_PANELS:
        jobs = fviz.single_panel_jobs(
            slices, polylines=kwargs["polylines"], nrms_annotation=annotation)
    else:
        jobs = [dict(suffix="", label="", kwargs={})]
    pages = 0
    for job in jobs:
        for i, fmt in enumerate(_formats()):
            figure = fviz.plot_model_slices_robust(
                f"{stem}{job['suffix']}.{fmt}", out=OUT,
                show=show and i == 0, **{**kwargs, **job["kwargs"]})
            if fmt == "pdf" and catalog is not None:
                name = f"{model['label']} {job['label']}".strip()
                pages += bool(fviz.append_catalog_page(catalog, figure, name))
            del figure
    return pages


def _catalog_path(models, work_dir):
    """File of the multi-page PDF (MOD_CATALOG), None = no catalog."""
    if not MOD_CATALOG:
        return None
    if "pdf" not in _formats():
        print('  catalog: "pdf" is not in MOD_PLOT_FORMAT - skipped.')
        return None
    if str(MOD_CATALOG).strip().lower() != "auto":
        return fem.anchor_path(MOD_CATALOG, work_dir)
    directory = (fem.anchor_path(OUT_DIR, work_dir)
                 or os.path.commonpath(
                     [os.path.abspath(m["dir"]) for m in models]))
    name = "catalog_slices.pdf"
    if OUT_PREFIX:
        name = f"{OUT_PREFIX}_{name}"
    return os.path.join(directory, name)


# ===========================================================================
# Main
# ===========================================================================

def main(argv=None) -> int:
    """Plot all runs; 0 = done, 1 = something failed, 2 = bad WORK_DIR."""
    parser = argparse.ArgumentParser(
        description="Slice figures of single FEMTIC models.")
    parser.add_argument(
        "runs", nargs="*",
        help="run directories, block files, results files or globs "
             "(default: RUN_PATHS)")
    parser.add_argument(
        "-w", "--work-dir", default=None,
        help="base directory of relative paths (default: WORK_DIR)")
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))

    wd_set = args.work_dir if args.work_dir is not None else WORK_DIR
    work_dir = os.path.abspath(os.path.expanduser(str(wd_set or ".")))
    if not os.path.isdir(work_dir):
        print(f"WORK_DIR does not exist: {work_dir} (set WORK_DIR or use -w)")
        return 2
    run_paths = list(args.runs) or RUN_PATHS

    if MOD_MESH_SWAP_XY is not None:
        fem.MESH_SWAP_XY = bool(MOD_MESH_SWAP_XY)

    models = fem.resolve_model_files(
        run_paths, ITERATION, cnv_name=CNV_FILE or None, base_dir=work_dir,
        out=True)  # skipped entries are always reported
    if not models:
        print(f"No model found for {run_paths!r} (ITERATION = {ITERATION!r}, "
              f"WORK_DIR = {work_dir}).")
        return 1
    print(f"\n{len(models)} model(s) to plot\n")

    stem = None
    if MOD_QC_FILE:
        if len(models) == 1:
            stem = fem.anchor_path(MOD_QC_FILE, work_dir)
            base, ext = os.path.splitext(stem)
            if ext.lstrip(".").lower() in _formats():
                stem = base
        else:
            print("MOD_QC_FILE is only used for a single model - ignored.")

    show = bool(MOD_SHOW_IN_SPYDER) and utl.runtime_env() == "spyder"
    catalog_path = _catalog_path(models, work_dir)
    catalog = None
    if catalog_path:
        from matplotlib.backends.backend_pdf import PdfPages

        os.makedirs(os.path.dirname(catalog_path), exist_ok=True)
        catalog = PdfPages(catalog_path)

    done, pages = 0, 0
    try:
        for i, model in enumerate(models, 1):
            print(f"[{i}/{len(models)}] {model['label']} / {model['tag']}: "
                  f"{model['file']}")
            try:
                pages += _plot_model(model, work_dir, stem=stem,
                                     catalog=catalog, show=show)
                done += 1
            except Exception as exc:  # one bad run must not stop the others
                print(f"  FAILED: {type(exc).__name__}: {exc}")
                import matplotlib.pyplot as plt

                plt.close("all")
            gc.collect()  # Matplotlib figures hold reference cycles
    finally:
        if catalog is not None:
            info = catalog.infodict()
            info["Title"] = "femtic_mod_plot_slice catalog"
            info["Author"] = "femtic_mod_plot_slice.py"
            catalog.close()

    print(f"\nSlice plots: {done} of {len(models)} model(s) done")
    if catalog_path:
        print(f"  catalog: {pages} page(s) -> {catalog_path}")

    utl.write_param_summary(
        __file__,
        namespace=dict(globals(), WORK_DIR=work_dir, RUN_PATHS=run_paths),
        out_dir=work_dir)

    # entries of run_paths without any model were reported by the resolver
    return 0 if done == len(models) else 1


if __name__ == "__main__":
    _rc = main(sys.argv[1:])
    if _rc:  # no SystemExit on success (keeps Spyder sessions quiet)
        sys.exit(_rc)
