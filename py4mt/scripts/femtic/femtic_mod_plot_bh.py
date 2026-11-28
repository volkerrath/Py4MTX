#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_mod_plot_bh.py -- 1-D resistivity-depth logs along boreholes for single
FEMTIC resistivity models (mod family).

The settings are those of femtic_ens_plot_bh.py (same MOD_* / MOD_BH_* names,
same MOD_BH entries, lengths in km) without everything that belongs to ensemble
statistics. Several runs can be given, as in femtic_mod_misfit.py: every entry
of RUN_PATHS is a model of its own (no ensemble) and gets its own figure(s)
and profile file. The profiles are sampled by fem.sample_model_boreholes() and
drawn by fviz.draw_borehole_figures(), the routine of the ensemble script. For
the ensemble view of the same boreholes use femtic_ens_plot_bh.py.

Usage
-----
    python femtic_mod_plot_bh.py                          # RUN_PATHS
    python femtic_mod_plot_bh.py kra_rto_0 kra_rto_1      # two run directories
    python femtic_mod_plot_bh.py "kra*/"                  # quote globs
    python femtic_mod_plot_bh.py -w /data/ann ann_rto_orig-err_0/resistivity_block_iter10.dat

Runs
----
An entry of RUN_PATHS (or of the command line, which replaces it) is a run
directory (the model of ITERATION), a block file, a results_iterN.h5 file
(= the block file of that iteration; the HDF5 file is not read) or a glob of
these. Relative entries start at WORK_DIR (-w / --work-dir replaces it; None
or "" = current directory).

MOD_MESH, MOD_SITE_DAT and MOD_REF_FILE are looked up per run: a relative name
in the run directory first, then in WORK_DIR; an absolute path is used for
every run. The mesh centre is therefore determined run by run.

An entry without a model is reported and skipped. A model that fails is
reported and skipped too; the others are still plotted. Return code: 0 = all
models found were plotted, 1 = one failed or none found, 2 = WORK_DIR missing.

Output
------
Per model:

    <dir>/[<OUT_PREFIX>_]<label>_<tag>_boreholes.<fmt>   (and .npz)

label = run directory name, tag = "iterN"; <dir> = OUT_DIR, else
<run dir>/<OUT_PREFIX> (if OUT_PREFIX is set and OUT_PREFIX_DIR), else the run
directory. More than MOD_BH_PER_FIG boreholes give figures _01, _02, ...
With MOD_SINGLE_PANELS = True every borehole becomes a figure of its own
(..._boreholes_01_<name>.<fmt>, ...).
MOD_BH_FILE names the files of a single model explicitly; MOD_CATALOG collects
all figures in one multi-page PDF. The parameter summary
femtic_mod_plot_bh_summary.md goes to WORK_DIR.

NPZ: depth_<name> [m, positive down], model_<name> and ref_<name> [Ohm.m, NaN
= air / outside the mesh], "header" (JSON). Blanks in <name> become "_".

Needs femtic.py, femtic_viz.py and util.py (PY4MTX_ROOT, PY4MTX_DATA).

Sister scripts: femtic_mod_plot_slice.py, femtic_mod_plot_3d.py,
femtic_mod_misfit.py.

AI-generated code (Claude Sonnet 5.5, Anthropic, 2026-11-26) -- review before
production use.

@author vrath
@project py4mt
@created 2026-06-04 (Claude Sonnet 4.6, Anthropic)
@modified 2026-11-26 (Claude Sonnet 5.5, Anthropic)

Provenance
----------
    2026-06-04  vrath / Claude Sonnet 4.6 (Anthropic)
                Split from femtic_mod_plot_slice.py.  Borehole config block
                (BOREHOLE_*, PLOT_BOREHOLE) and execution step moved here.
                fviz (femtic_viz) used directly.
    2026-06-04  vrath / Claude Sonnet 4.6 (Anthropic)
                Added BOREHOLE_MARKERS config block for free annotations
                (arrows + text) and LEGEND_FONTSIZE for the legend / panel
                title font size.  Both forwarded to
                ``fviz.plot_borehole_logs``.

@modified: 2026-06-04  vrath / Claude Sonnet 4.6 (Anthropic)
@modified: 2026-08-13  Claude Sonnet 5 (Anthropic) -- Added
    femtic_mod_plot_bh_summary.md output at end of run: writes user-set
    (UPPERCASE) parameters, script path, and run date/time via
    utl.write_param_summary().
@modified: 2026-10-28  Claude Sonnet 5.5 (Anthropic) -- The parameter
    summary femtic_mod_plot_bh_summary.md is now written to WORK_DIR
    instead of next to the script (utl.write_param_summary(fname,
    out_dir=WORK_DIR)). Needs the util.py of the same date. AI-generated;
    review before production use.
@modified: 2026-11-08  Claude Sonnet 5.5 (Anthropic) -- Several models in one
    run: MODEL_FILE may be a list, a glob or run directories (MODEL_ITER
    selects the iteration), giving one figure per model plus an optional
    catalog PDF (BATCH_* settings; femtic.resolve_model_files,
    fviz.append_catalog_page). One model behaves exactly as before.
    fviz.plot_borehole_logs now returns its figure. AI-generated; review
    before production use.
@modified: 2026-11-10  Claude Sonnet 5.5 (Anthropic) -- New setting
    MESH_SWAP_XY (default True): sets fem.MESH_SWAP_XY before the mesh is
    read. mesh.dat holds x = northing, y = easting, but the borehole
    positions (site.dat, UTM -> model) are in a geographic frame
    (x = easting, y = northing); without the swap (the default of
    femtic.py since 2026-10-11) the mesh is transposed relative to the
    positions and the logs are taken at the wrong places. None leaves
    femtic's default, False forces no swap. AI-generated; review before
    production use.
@modified: 2026-11-11  Claude Sonnet 5.5 (Anthropic) -- Titles in the form of
    all plot scripts. New FIGURE_TITLE ("auto" = the block-file name, as
    before; None / False / "" = no title; text with {placeholders}; dict per
    model) for the single-model figure and, when BATCH_TITLE is unset, for
    the batch figures; BATCH_TITLE now also takes "auto", False and a dict.
    New PANEL_TITLES (False = no panel titles) and FIGURE_TITLE_FONTSIZE.
    Defaults give the same figures as before. Needs the femtic_viz.py of the
    same date. AI-generated; review before production use.
@modified: 2026-11-11b  Claude Sonnet 5.5 (Anthropic) -- New setting CNV_FILE
    (one name or a list of candidate names of the convergence file, default
    ["femtic.cnv", "FEMTIC-DABIC.cnv"]; the first one that exists is used),
    passed to fem.resolve_model_files() as cnv_name; it matters for
    MODEL_ITER = "last" / "best" with run directories. Needs the femtic.py of
    the same date. AI-generated; review before production use.

@modified: 2026-11-26  Claude Sonnet 5.5 (Anthropic) -- Rewritten with the
    settings of femtic_ens_plot_bh.py (without the ensemble statistics) and
    several runs as in femtic_mod_misfit.py. CHANGED: MODEL_FILE /
    BOREHOLE_FILE / MODEL_ITER / BATCH_* -> RUN_PATHS, ITERATION (default
    "best"), command line and -w / --work-dir; OUT_DIR, OUT_PREFIX,
    OUT_PREFIX_DIR, MOD_BH_FILE, MOD_CATALOG, MOD_PLOT_FORMAT. BOREHOLE_SITES
    -> MOD_BH (depths and dz now km, line styles in "style"), BOREHOLE_STYLE ->
    MOD_BH_STYLE, BOREHOLE_XLIM / _SHARED / _MARKERS / _NPZ -> MOD_BH_*
    (marker depths km), other settings MOD_*. New: "sites" templates,
    MOD_REF_*, MOD_DISPLAY_COORDS, main(). Profiles by
    fem.sample_model_boreholes(), figures by fviz.draw_borehole_figures()
    (instead of fviz.plot_borehole_logs()); NPZ arrays are now depth_<name>,
    model_<name>, ref_<name> plus a JSON header. Mesh, site list and reference
    are per run. Needs femtic.py and femtic_viz.py of the same date. Tested on
    synthetic files only. AI-generated; review before production use.
@modified: 2026-11-26b  Claude Sonnet 5.5 (Anthropic) -- Optional
    single-panel output: new MOD_SINGLE_PANELS (default False) draws one
    figure per borehole. Needs femtic_viz.py of the same date. Tested on
    synthetic files only. AI-generated; review before production use.
@modified: 2026-11-26c  Claude Sonnet 5.5 (Anthropic) -- Default settings
    replaced by those of femtic_ens_plot_bh.py: WORK_DIR (Annecy ensemble),
    RUN_PATHS ["ann*/"], mesh, site list and reference file in templates/,
    MOD_BH one panel per site (surface to 4 km), black dashed reference line,
    MOD_CLIM [0, 4], MOD_BH_XLIM None, panel size, font sizes, MOD_DPI 400,
    no annotations. No change of the code. Tested on synthetic files only.
    AI-generated; review before production use.

2026-11-26 (km)  Claude Sonnet 5.5 (Anthropic)
              User settings are km throughout, UTM coordinates too (before:
              UTM and some per-entry values in metres). Personal settings in
              metres must be converted; values that look like metres stop
              with an error (fem.KmUnitError). Concerns MOD_UTM_ORIGIN_E /
              _N and (value, 'utm') borehole positions. Needs femtic.py of
              the same date. Tested on synthetic files only. AI-generated;
              review before production use.
"""

from __future__ import annotations

import argparse
import gc
import json
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
#: relative OUT_DIR, MOD_CATALOG and MOD_BH_FILE start here; it is also the
#: second place where the per-run files (mesh, site list, ...) are looked for.
#: None or "" = current directory. Command line: -w / --work-dir.
WORK_DIR = r"/media/vrath/LargeBack/Ensembles/annecy2026/RTO/ann_rto_orig-err/"

#: Runs to plot (every entry is a model of its own):
#:   a run directory, a block file, a results_iterN.h5 file, or a glob of these.
#: Command-line arguments replace this list.
RUN_PATHS = [
    "ann*/",                                 # every member run in WORK_DIR
    # single runs / block files instead:
    # "ann_rto_orig-err_0/",
]

#: Iteration taken from a run directory (not used for files): "best" (lowest
#: nRMS in the convergence file), "last" (its last row), N
#: (resistivity_block_iterN.dat, 0 = starting model), or a list [0, "best"].
ITERATION = "best"

#: Convergence file of a run: one name or candidates (first that exists).
CNV_FILE = ["femtic.cnv", "FEMTIC-DABIC.cnv"]

# ---------------------------------------------------------------------------
# Mesh (required)
# ---------------------------------------------------------------------------
#: Relative names: looked for in each run directory, then in WORK_DIR.
MOD_MESH = "templates/mesh.dat"

#: Swap of the first two node-coordinate columns when reading MOD_MESH
#: (mesh.dat holds x = northing, y = easting; borehole positions are in a
#: geographic frame x = easting, y = northing). True (default) swaps; False
#: only for meshes already written as x = easting; None = femtic's default.
MOD_MESH_SWAP_XY = True

# --- Ocean / air handling (must match the inversion setup) ----------------
MOD_OCEAN = None        # None = infer from the block file; True / False
MOD_AIR_RHO = 1.0e9     # Ohm.m (region 0); information only
MOD_OCEAN_RHO = 0.25    # Ohm.m (region 1 when treated as ocean)

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
#: One or more file extensions; a bare string works too.
MOD_PLOT_FORMAT = ["pdf"]
#: One directory for the files of all runs (relative: to WORK_DIR), used as
#: given. None = per run, see OUT_PREFIX.
OUT_DIR = None
#: Put in front of every file name; with OUT_DIR None it also names a
#: sub-directory of each run directory (OUT_PREFIX_DIR). None = neither.
OUT_PREFIX = None
OUT_PREFIX_DIR = True   # False: with OUT_DIR None write into the run directory
#: Explicit output path WITHOUT extension (relative: to WORK_DIR), used only
#: when exactly one model is plotted. None = the naming scheme.
MOD_BH_FILE = None
#: Export the sampled profiles to <figure name>.npz. False = skip.
MOD_BH_NPZ = True
#: Multi-page PDF with all figures: None = none; "auto" =
#: <OUT_DIR or the common parent of the run directories>/catalog_boreholes.pdf;
#: or a path.
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
#: "box" | "average" (of the sites in MOD_SITE_DAT) | None (literals above)
MOD_ORIGIN_METHOD = "box"
#: Position text of the panel titles / legend: "model" | "utm" | "latlon"
MOD_DISPLAY_COORDS = "latlon"
#: Site list (name,lat,lon,elev,sitenum,E,N): mesh centre and "sites" entries
#: of MOD_BH. None = off.
MOD_SITE_DAT = "templates/site.dat"

# ---------------------------------------------------------------------------
# Reference model (optional)
# ---------------------------------------------------------------------------
#: Resistivity block file on the SAME mesh, added as a dashed profile.
MOD_REF_FILE = "templates/referencemodel.dat"     # e.g. "referencemodel.dat"
#: Add it to every borehole (an entry can say ref=False, or list "ref" in its
#: "what" to force it on).
MOD_REF_BH = True
MOD_REF_LABEL = "reference model"   # legend entry; None = none

# ---------------------------------------------------------------------------
# Figure parameters
# ---------------------------------------------------------------------------
#: Default bottom of a profile ("z_bot"), [zmin, zmax] km, only zmax is used.
#: None = 20 km.
MOD_ZLIM = None
#: [log10_min, log10_max] Ohm.m: resistivity range if MOD_BH_XLIM is None.
MOD_CLIM = [0.0, 4.0]
MOD_DPI = 400
MOD_DEPTH_KM = True        # depth axis in km (False: m)
#: Points. Panel titles use MOD_TICK_FONTSIZE - 2, the legend - 4.
MOD_TICK_FONTSIZE = 16
MOD_LABEL_FONTSIZE = 16
MOD_TITLE_FONTSIZE = MOD_LABEL_FONTSIZE + 10
#: Figure title:
#:   "auto"            block-file name; run, iteration and nRMS
#:   None, False or "" no title
#:   a string          used as given; placeholders {label} (= {run} = {key} =
#:                     {best}: run directory name), {tag}, {iter}, {nrms}
#:                     ({nrms:.2f} works), {alpha}, {file}, {prefix}
#:                     (OUT_PREFIX); "?" / "n/a" if unknown
#:   a dict            one entry per run label, plus "default"
MOD_TITLE = "auto"
#: Panel titles (borehole name and position): True / False.
MOD_PANEL_TITLES = True

# ---------------------------------------------------------------------------
# Boreholes
# ---------------------------------------------------------------------------
#: One dict per borehole:
#:   "name"        label (panel title / legend; names the NPZ arrays)
#:   "x", "y"      model-local km, or (value, "utm") [km] /
#:                 (value, "latlon") [deg; x = longitude, y = latitude]
#:   "z_top"       km (positive down) or "surface" (default)
#:   "z_bot"       km (default MOD_ZLIM[1], else 20)
#:   "dz"          km (default 0.05)
#:   "xlim"        [rho_min, rho_max] Ohm.m of this panel
#:   "what"        ["model"] (default), ["model", "ref"]; ensemble items of a
#:                 femtic_ens_plot_bh.py entry are ignored
#:   "ref"         False = no reference line in this panel
#:   "style"       overrides of MOD_BH_STYLE for this borehole
#: or a template  dict(sites="all" | [names], n_random=None, seed=None, ...):
#: one borehole per site of MOD_SITE_DAT.
MOD_BH = [
    dict(sites="all", z_top="surface", z_bot=4.0, dz=0.05,   # one panel per site
         xlim=[1., 10000.]),
    # dict(sites=["SITE01", "SITE07"], z_bot=5.0),
    # dict(name="SV-108",
    #      x=(6.053423, "latlon"), y=(45.92574, "latlon"),
    #      z_top="surface", z_bot=2., dz=0.05, xlim=[1., 10000.]),
    # dict(name="SV-109",
    #      x=(6.05340119, "latlon"), y=(45.940467, "latlon"),
    #      z_top="surface", z_bot=2., dz=0.05, xlim=[1., 10000.]),
    # dict(name="model centre", x=0.0, y=0.0,   # km, model-local
    #      z_top="surface", z_bot=4.0, dz=0.05),
    # dict(sites="all", n_random=6, seed=1, z_bot=5.0),   # 6 random sites
]
#: Line styles ("model", "ref"), any Matplotlib Line2D keyword. A "ref" style
#: without "color" takes the colour of its model line.
MOD_BH_STYLE = {
    "model": dict(lw=2.0, ls="-"),
    "ref": dict(color="black", lw=2.2, ls="--"),
}
#: True = all boreholes of a figure on one axes; False = one panel each.
MOD_BH_SHARED = False
#: False = panels in a grid (MOD_BH_NCOLS, MOD_BH_PER_FIG);
#: True  = one figure per borehole (..._boreholes_01_<name>.<fmt>);
#: MOD_BH_NCOLS / MOD_BH_PER_FIG are then not used. MOD_BH_SHARE_DEPTH = True
#: gives all these figures the same depth range. No effect with MOD_BH_SHARED.
MOD_SINGLE_PANELS = False
#: x-axis (resistivity) range in Ohm.m, log scale. None = 10**MOD_CLIM, else
#: auto.
MOD_BH_XLIM = None
#: Panel size [width, height] in cm, panel columns (None = up to 4), maximum
#: panels (boreholes) per figure (None = all in one).
MOD_BH_PANEL_SIZE = [16.0, 20.0]
MOD_BH_NCOLS = 2
MOD_BH_PER_FIG = 10
#: True = one common depth range for all panels of a figure.
MOD_BH_SHARE_DEPTH = True
MOD_BH_LEGEND_LOC = "best"      # None = no legend

#: Annotations (arrows + text) on the depth panels; [] or None = none.
#:   "depth"       arrow tip, km positive down (required)
#:   "rho"         arrow tip [Ohm.m]; default: left edge of the axis
#:   "rho_text"    text position [Ohm.m]; default 3 * rho
#:   "depth_text"  text position [km]; default depth - 0.3
#:   "text"        label
#:   "borehole"    name or list of names; default: every panel
#:   "arrowprops"  passed to Axes.annotate; other keys too (color, fontsize..)
MOD_BH_MARKERS = [
    # dict(depth=1.5, rho=40.0, text="conductor"),
]

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
    """Figure title of one model, see MOD_TITLE (None = no title)."""
    nrms, alpha, it = model.get("nrms"), model.get("alpha"), model.get("iter")
    block = os.path.basename(model["file"])
    info = [model["label"]]
    if it is not None:
        info.append(f"iter {it}")
    if nrms is not None:
        info.append(f"nRMS {nrms:.4f}")
    label = model["label"]
    title = fviz.resolve_title(
        MOD_TITLE, label, f"{block}\n{'  '.join(info)}", name="MOD_TITLE",
        label=label, run=label, best=label, tag=model["tag"],
        iter="?" if it is None else it,
        nrms="n/a" if nrms is None else _Number(nrms),
        alpha="n/a" if alpha is None else alpha,
        file=block, prefix=OUT_PREFIX or "",
    )
    return title if title != "" else None


def _latlon_used(specs):
    """True if a borehole position is given in geographic coordinates."""
    return any(
        isinstance(v, (list, tuple)) and len(v) == 2
        and str(v[1]).lower() == "latlon"
        for sp in specs for v in (sp.get("x"), sp.get("y"))
    )


def _plot_model(model, work_dir, *, meshes, stem=None, catalog=None,
                show=False):
    """Profiles, NPZ and figure(s) of one model.

    model : entry of fem.resolve_model_files(); meshes : cache {path:
    fem.BoreholeMesh}, filled here; stem : output path without extension
    (None = the naming scheme); catalog : open PdfPages or None.
    """
    run_dir = model["dir"]

    # --- per-run files -----------------------------------------------------
    mesh_file = fem.find_run_file(MOD_MESH, run_dir, work_dir)
    if mesh_file is None:
        raise FileNotFoundError(
            f"mesh MOD_MESH = {MOD_MESH!r} not found in {run_dir} or "
            f"{work_dir}.")
    site_dat = fem.find_run_file(MOD_SITE_DAT, run_dir, work_dir)
    if MOD_SITE_DAT and site_dat is None:
        print(f"  MOD_SITE_DAT = {MOD_SITE_DAT!r} not found in {run_dir} or "
              f"{work_dir}.")
    struct = fem._read_resistivity_block_struct(
        model["file"], ocean=MOD_OCEAN, out=False)
    key = os.path.realpath(mesh_file)
    if key not in meshes:
        meshes[key] = fem.BoreholeMesh(mesh_file)
    mesh = meshes[key]

    # --- mesh-centre origin --------------------------------------------------
    origin_e, origin_n = fem.utm_km_to_m(
        MOD_UTM_ORIGIN_E, MOD_UTM_ORIGIN_N, name="MOD_UTM_ORIGIN_E / _N")
    (utm_e, utm_n, _lat, _lon, utm_zone, utm_north, _sites,
     _obs) = fem.resolve_origin_and_sites(
        site_dat=site_dat,
        origin_e=origin_e,
        origin_n=origin_n,
        origin_lat=MOD_UTM_ORIGIN_LAT,
        origin_lon=MOD_UTM_ORIGIN_LON,
        zone_override=MOD_UTM_ZONE_OVERRIDE,
        origin_method=MOD_ORIGIN_METHOD,
        plot_sites=False,
    )

    # --- reference model, borehole list ----------------------------------------
    ref_el, ref_path = None, None
    wanted = MOD_REF_BH or any("ref" in (t.get("what") or ()) for t in MOD_BH)
    if MOD_REF_FILE and wanted:
        ref_path = (fem.find_run_file(MOD_REF_FILE, run_dir, work_dir)
                    or fem.anchor_path(MOD_REF_FILE, run_dir))
        ref_el = fem.ReferenceModel(
            ref_path, ocean=MOD_OCEAN, ocean_rho=MOD_OCEAN_RHO
        ).element_log10(mesh.n_elem)

    specs = fem.expand_site_templates(MOD_BH, site_dat, utm_e, out=OUT)
    if _latlon_used(specs) and (utm_e is None or utm_zone is None):
        raise ValueError(
            "lat/lon borehole positions need the mesh-centre origin: give "
            "MOD_SITE_DAT (with MOD_ORIGIN_METHOD) or MOD_UTM_ORIGIN_E / _N / "
            "_LAT / _LON.")

    # --- profiles --------------------------------------------------------------
    smp = fem.sample_model_boreholes(
        specs,
        mesh=mesh,
        struct=struct,
        utm=(utm_e, utm_n, utm_zone, utm_north),
        ref_el=ref_el,
        ref_default=MOD_REF_BH,
        ocean_rho=MOD_OCEAN_RHO,
        zlim_km=MOD_ZLIM,
        display_coords=MOD_DISPLAY_COORDS,
        out=OUT,
    )
    if not smp["profiles"]:
        raise ValueError("MOD_BH: no usable borehole - nothing plotted.")

    if stem is None:
        stem = fem.model_output_stem(
            model, "boreholes", out_dir=OUT_DIR, prefix=OUT_PREFIX,
            prefix_dir=OUT_PREFIX_DIR, base_dir=work_dir)
    os.makedirs(os.path.dirname(os.path.abspath(stem)), exist_ok=True)

    if MOD_BH_NPZ:
        header = dict(
            model_file=model["file"],
            iteration=model["iter"],
            nrms=model["nrms"],
            alpha=model["alpha"],
            mesh_file=mesh_file,
            reference_file=ref_path if ref_el is not None else None,
            depth_unit="m, z positive-down, model datum",
            rho_unit="Ohm.m (NaN = air / outside mesh)",
            boreholes=smp["hdr"],
        )
        arrays = dict(smp["npz"])
        arrays["header"] = np.array(json.dumps(header, indent=2))
        np.savez_compressed(stem + ".npz", **arrays)
        if OUT:
            print(f"  profiles saved -> {stem}.npz")

    # --- figures ---------------------------------------------------------------
    single = bool(MOD_SINGLE_PANELS) and not MOD_BH_SHARED
    if MOD_SINGLE_PANELS and MOD_BH_SHARED:
        print("  MOD_BH_SHARED: all boreholes on one axes - "
              "MOD_SINGLE_PANELS not used.")
    plan = fviz.plan_borehole_figures(
        smp["profiles"], None, None,
        per_fig=1 if single else MOD_BH_PER_FIG)
    opts = fviz.BoreholeFigureOptions(
        line_style=fem.model_style_keys(MOD_BH_STYLE),
        xlim=fem.borehole_global_xlim(MOD_BH_XLIM, MOD_CLIM),
        ncols=MOD_BH_NCOLS,
        panel_size=MOD_BH_PANEL_SIZE,
        share_depth=MOD_BH_SHARE_DEPTH,
        legend_loc=MOD_BH_LEGEND_LOC,
        panel_titles=bool(MOD_PANEL_TITLES),
        depth_km=MOD_DEPTH_KM,
        tick_fontsize=MOD_TICK_FONTSIZE,
        label_fontsize=MOD_LABEL_FONTSIZE,
        title_fontsize=MOD_TITLE_FONTSIZE,
        dpi=MOD_DPI,
        formats=_formats(),
        show=show,
        ref_label=MOD_REF_LABEL if ref_el is not None else None,
        suptitle=_figure_title(model),
        shared=MOD_BH_SHARED,
        markers=MOD_BH_MARKERS,
        key_labels={"best": "model"},
        ref_follows_color=True,
        catalog=catalog,
        out=OUT,
    )
    fviz.draw_borehole_figures(plan, None, stem, opts, name_suffix=single)


def _catalog_path(models, work_dir):
    """File of the multi-page PDF (MOD_CATALOG), None = no catalog."""
    if not MOD_CATALOG:
        return None
    if str(MOD_CATALOG).strip().lower() != "auto":
        return fem.anchor_path(MOD_CATALOG, work_dir)
    directory = (fem.anchor_path(OUT_DIR, work_dir)
                 or os.path.commonpath(
                     [os.path.abspath(m["dir"]) for m in models]))
    name = "catalog_boreholes.pdf"
    if OUT_PREFIX:
        name = f"{OUT_PREFIX}_{name}"
    return os.path.join(directory, name)


# ===========================================================================
# Main
# ===========================================================================

def main(argv=None) -> int:
    """Plot all runs; 0 = done, 1 = something failed, 2 = bad WORK_DIR."""
    parser = argparse.ArgumentParser(
        description="Borehole resistivity logs of single FEMTIC models.")
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
    if not MOD_BH:
        print("MOD_BH is empty - nothing to plot.")
        return 1

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
    if MOD_BH_FILE:
        if len(models) == 1:
            stem = fem.anchor_path(MOD_BH_FILE, work_dir)
            base, ext = os.path.splitext(stem)
            if ext.lstrip(".").lower() in _formats() + ["npz"]:
                stem = base
        else:
            print("MOD_BH_FILE is only used for a single model - ignored.")

    show = bool(MOD_SHOW_IN_SPYDER) and utl.runtime_env() == "spyder"
    catalog_path = _catalog_path(models, work_dir)
    catalog = None
    if catalog_path:
        from matplotlib.backends.backend_pdf import PdfPages

        os.makedirs(os.path.dirname(catalog_path), exist_ok=True)
        catalog = PdfPages(catalog_path)

    done, meshes = 0, {}
    try:
        for i, model in enumerate(models, 1):
            print(f"[{i}/{len(models)}] {model['label']} / {model['tag']}: "
                  f"{model['file']}")
            try:
                _plot_model(model, work_dir, meshes=meshes, stem=stem,
                            catalog=catalog, show=show)
                done += 1
            except Exception as exc:  # one bad run must not stop the others
                print(f"  FAILED: {type(exc).__name__}: {exc}")
                import matplotlib.pyplot as plt

                plt.close("all")
            gc.collect()  # Matplotlib figures hold reference cycles
    finally:
        pages = 0
        if catalog is not None:
            pages = catalog.get_pagecount()
            info = catalog.infodict()
            info["Title"] = "femtic_mod_plot_bh catalog"
            info["Author"] = "femtic_mod_plot_bh.py"
            catalog.close()

    print(f"\nBorehole plots: {done} of {len(models)} model(s) done")
    if catalog_path:
        print(f"  catalog: {pages} page(s) -> {catalog_path}")

    utl.write_param_summary(
        __file__,
        namespace=dict(globals(), WORK_DIR=work_dir, RUN_PATHS=run_paths),
        out_dir=work_dir)

    return 0 if done == len(models) else 1


if __name__ == "__main__":
    _rc = main(sys.argv[1:])
    if _rc:  # no SystemExit on success (keeps Spyder sessions quiet)
        sys.exit(_rc)
