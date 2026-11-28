#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_mod_plot_3d.py -- PyVista 3-D rendering and VTK/VTU export of single
FEMTIC resistivity models (mod family).

Slice planes, oblique planes and iso-surfaces of the model are rendered by
fviz.plot_model_3d(). The settings are named as in the other plot scripts
(MOD_*, lengths in km), and several runs can be given, as in
femtic_mod_misfit.py: every entry of RUN_PATHS is a model of its own (no
ensemble) and gets its own scene and grid file.

Usage
-----
    python femtic_mod_plot_3d.py                          # RUN_PATHS
    python femtic_mod_plot_3d.py kra_rto_0 kra_rto_1      # two run directories
    python femtic_mod_plot_3d.py "kra*/"                  # quote globs
    python femtic_mod_plot_3d.py -w /data/ann ann_rto_orig-err_0/resistivity_block_iter10.dat

Runs
----
An entry of RUN_PATHS (or of the command line, which replaces it) is a run
directory (the model of ITERATION), a block file, a results_iterN.h5 file
(= the block file of that iteration; the HDF5 file is not read) or a glob of
these. Relative entries start at WORK_DIR (-w / --work-dir replaces it; None
or "" = current directory).

MOD_MESH and MOD_SITE_DAT are looked up per run: a relative name in the run
directory first, then in WORK_DIR; an absolute path is used for every run.
With MOD_ROI_AUTO the clip box follows the sites of MOD_SITE_DAT.

An entry without a model is reported and skipped. A model that fails is
reported and skipped too; the others are still plotted. Return code: 0 = all
models found were done, 1 = one failed, none found, or nothing to do
(PyVista missing, no output selected), 2 = WORK_DIR missing.

Output
------
Per model:

    <dir>/[<OUT_PREFIX>_]<label>_<tag>_3d.<fmt>       (MOD_PLOT_FORMAT)
    <dir>/[<OUT_PREFIX>_]<label>_<tag>_3d.<vtu|vtk>   (MOD_3D_VTU)

label = run directory name, tag = "iterN"; <dir> = OUT_DIR, else
<run dir>/<OUT_PREFIX> (if OUT_PREFIX is set and OUT_PREFIX_DIR), else the run
directory. MOD_3D_FILE names the files of a single model explicitly. The
parameter summary femtic_mod_plot_3d_summary.md goes to WORK_DIR.

Coordinates are model-local: x = easting, y = northing (with
MOD_MESH_SWAP_XY True), z positive down.

Needs femtic.py, femtic_viz.py, util.py (PY4MTX_ROOT, PY4MTX_DATA) and
PyVista (conda install -c conda-forge pyvista).

Sister scripts: femtic_mod_plot_slice.py, femtic_mod_plot_bh.py,
femtic_mod_misfit.py.

AI-generated code (Claude Sonnet 5.5, Anthropic, 2026-11-26) -- review before
production use.

@author vrath
@project py4mt
@created 2026-06-03 (Claude Sonnet 4.6, Anthropic)
@modified 2026-11-26 (Claude Sonnet 5.5, Anthropic)

Provenance
----------
    2026-05-13  vrath / Claude Sonnet 4.6   Created as part of
                femtic_mod_plot.py; PLOT3D config block with axis-aligned
                x/y/z slices, oblique planes, and iso-surfaces via
                fviz.plot_model_3d.  Output: interactive HTML or screenshot.
    2026-05-26  Claude Sonnet 4.6 (Anthropic)
                Added PLOT3D_VTU_FILE config var; changed PLOT3D_FILE default
                from .html to .png.  plot_model_3d moved into femtic_viz.py.
    2026-05-27  vrath / Claude Sonnet 4.6 (Anthropic)
                Passed PLOT_XLIM/YLIM/ZLIM to plot_model_3d for spatial
                clipping of the VTU export and PyVista scene.
    2026-05-31  vrath / Claude Sonnet 4.6 (Anthropic)
                Origin estimation now runs before UTM zone derivation.
                Hard-coded UTM_ORIGIN_* set to None (derived at runtime).
    2026-06-03  Claude Sonnet 4.6 (Anthropic)
                Split from femtic_mod_plot.py into femtic_mod_plot_slice.py
                (2-D slices + boreholes) and femtic_mod_plot_3d.py (this
                script, PyVista 3-D rendering only).
    2026-08-13  Claude Sonnet 5 (Anthropic)
                Added femtic_mod_plot_3d_summary.md output at end of run:
                writes user-set (UPPERCASE) parameters, script path, and
                run date/time via utl.write_param_summary().
    2026-10-28  Claude Sonnet 5.5 (Anthropic)
                The parameter summary femtic_mod_plot_3d_summary.md is now
                written to the working directory set in this script
                (WORK_DIR) instead of next to the script:
                utl.write_param_summary(fname, out_dir=WORK_DIR). Needs
                the util.py of the same date (new out_dir argument).
                AI-generated; review before production use.

    2026-11-26  Claude Sonnet 5.5 (Anthropic)
                Settings named as in the other plot scripts and several runs as
                in femtic_mod_misfit.py. CHANGED: MODEL_FILE -> RUN_PATHS,
                ITERATION (default "best"), command line and -w / --work-dir;
                OUT_DIR, OUT_PREFIX, OUT_PREFIX_DIR; PLOT3D_FILE ->
                MOD_PLOT_FORMAT / MOD_3D_FILE (file names
                <run>_iterN_3d.<ext>); PLOT3D_VTU_FILE -> MOD_3D_VTU; PLOT_*
                and PLOT3D_* -> MOD_* / MOD_3D_*, with slice positions, plane
                origins and limits in km; MOD_CLIM and MOD_3D_ISOVALUES are
                log10(Ohm.m) for both scalars; PLOT_DPI (unused) ->
                MOD_3D_SCREENSHOT_SCALE. Removed: the UTM origin and SITE_DAT
                settings (computed and printed only, not used for the scene;
                the script stopped without them). New: main().
                Fix: MOD_MESH_SWAP_XY (default True) now sets
                fem.MESH_SWAP_XY; without it x / y followed the mesh file
                (x = northing) instead of x = easting as documented.
                Fix: region 1 was always set to MOD_OCEAN_RHO; now only if
                the model has an ocean (MOD_OCEAN).
                Needs femtic.py and femtic_viz.py of the same date. Run with
                a PyVista stand-in only (no rendering tested).
                AI-generated; review before production use.
    2026-11-26  Claude Sonnet 5.5 (Anthropic)
                Default settings taken from the femtic_ens_plot_* scripts:
                WORK_DIR (Krafla ensemble), RUN_PATHS ["kra*/"], MOD_MESH in
                templates/, MOD_CMAP "jet_r", MOD_CLIM [0, 4], no fixed clip
                box (MOD_XLIM / YLIM / ZLIM None). New: MOD_ROI_AUTO,
                MOD_ROI_PAD_XY, MOD_ROI_ZLIM, MOD_SITE_DAT, MOD_UTM_* and
                MOD_ORIGIN_METHOD (clip box from the sites, as in the slice
                script). MOD_3D_SLICE_Z [1, 3] km. Run with a PyVista
                stand-in only. AI-generated; review before production use.

2026-11-26 (km)  Claude Sonnet 5.5 (Anthropic)
              User settings are km throughout, UTM coordinates too (before:
              UTM and some per-entry values in metres). Personal settings in
              metres must be converted; values that look like metres stop
              with an error (fem.KmUnitError). Concerns MOD_UTM_ORIGIN_E /
              _N. Needs femtic.py of the same date. Tested on synthetic
              files only. AI-generated; review before production use.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

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
#: relative OUT_DIR and MOD_3D_FILE start here; it is also the second place
#: where MOD_MESH and MOD_SITE_DAT are looked for. None or "" = current
#: directory.
#: Command line: -w / --work-dir.
WORK_DIR = r"/media/vrath/LargeBack/Ensembles/krafla2026/krafla_fine/krafla_ens_fine/"
# WORK_DIR = r"/media/vrath/LargeBack/Ensembles/annecy2026/RTO/ann_rto_orig-err/"
# WORK_DIR = r"/media/vrath/LargeBack/Ensembles/krafla2026/krafla_fine/krafla_ens_fine_paired/"

#: Runs to plot (every entry is a model of its own):
#:   a run directory, a block file, a results_iterN.h5 file, or a glob of these.
#: Command-line arguments replace this list.
RUN_PATHS = [
    "kra*/",                                 # every member run in WORK_DIR
    # single runs / block files instead:
    # "krafla_rto_0/",
    # "krafla_rto_0/resistivity_block_iter10.dat",
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
#: (mesh.dat holds x = northing, y = easting). True (default) gives
#: x = easting, y = northing; False = as stored; None = femtic's default.
MOD_MESH_SWAP_XY = True

# --- Ocean / air handling (must match the inversion setup) ----------------
MOD_OCEAN = None        # None = infer from the block file; True / False
MOD_AIR_RHO = 1.0e9     # Ohm.m (region 0); information only
MOD_OCEAN_RHO = 0.25    # Ohm.m (region 1 when it is the ocean)

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
#: Rendered view: one or more of "png", "jpg", "tif", "bmp" (screenshot) and
#: "html" (interactive scene, needs trame_vtk; else png); a bare string works
#: too. Each format renders the scene again.
#:   None = open an interactive PyVista window per model (needs a display);
#:   []   = no rendering, only the grid file of MOD_3D_VTU.
MOD_PLOT_FORMAT = ["png"]
#: Grid file for ParaView / data deposit: "vtu" (recommended), "vtk", or None.
MOD_3D_VTU = "vtu"
#: One directory for the files of all runs (relative: to WORK_DIR), used as
#: given. None = per run, see OUT_PREFIX.
OUT_DIR = None
#: Put in front of every file name; with OUT_DIR None it also names a
#: sub-directory of each run directory (OUT_PREFIX_DIR). None = neither.
OUT_PREFIX = None
OUT_PREFIX_DIR = True   # False: with OUT_DIR None write into the run directory
#: Explicit output path WITHOUT extension (relative: to WORK_DIR), used only
#: when exactly one model is plotted. None = the naming scheme.
MOD_3D_FILE = None

# ---------------------------------------------------------------------------
# Spatial clip box (grid file and scene)
# ---------------------------------------------------------------------------
#: Model-local km; None = no clipping on that axis.
MOD_XLIM = None            # [xmin, xmax] easting
MOD_YLIM = None            # [ymin, ymax] northing
MOD_ZLIM = None            # [zmin, zmax] km, positive down

#: True: x / y limits from the site bounding box of each run + MOD_ROI_PAD_XY,
#: z limits from MOD_ROI_ZLIM (None = keep MOD_ZLIM). Without sites the limits
#: above are used.
MOD_ROI_AUTO = True
MOD_ROI_PAD_XY = 2.0       # km
MOD_ROI_ZLIM = [-1.0, 6.0]        # e.g. [-1.0, 6.0]

# ---------------------------------------------------------------------------
# Sites and origin of the mesh centre (used by MOD_ROI_AUTO only)
# ---------------------------------------------------------------------------
#: mt_make_sitelist.py CSV (name,lat,lon,elev,sitenum,E,N); per run: run
#: directory first, then WORK_DIR. None = off.
MOD_SITE_DAT = "templates/site.dat"
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

# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
#: "log10_resistivity" or "resistivity".
MOD_3D_SCALAR = "log10_resistivity"
#: [log10_min, log10_max] Ohm.m (also for scalar "resistivity"); None = auto.
MOD_CLIM = [0.0, 4.0]
MOD_CMAP = "jet_r"

#: Axis-aligned slices, model-local km. [] or None = none along that axis.
MOD_3D_SLICE_X = [0.0]          # YZ planes: N-S sections at these eastings
MOD_3D_SLICE_Y = [0.0]          # XZ planes: E-W sections at these northings
MOD_3D_SLICE_Z = [1.0, 3.0]     # XY planes: maps at these depths

#: Oblique slices: dict(origin=[x, y, z] km, normal=[nx, ny, nz]).
MOD_3D_SLICE_PLANES = [
    # dict(origin=[0.0, 0.0, 8.0], normal=[1.0, 1.0, 0.0]),  # NE vertical
]

#: Iso-surfaces, log10(Ohm.m): 1 = 10, 2 = 100, 3 = 1000 Ohm.m. [] = none.
MOD_3D_ISOVALUES = [1.0, 2.0, 3.0]
MOD_3D_ISO_OPACITY = 0.35       # 0 = transparent, 1 = solid

MOD_3D_WINDOW_SIZE = [1600, 900]    # pixels [width, height]
MOD_3D_SCREENSHOT_SCALE = 2         # screenshot size = window size * scale

# ---------------------------------------------------------------------------
# Verbose output
# ---------------------------------------------------------------------------
OUT = True


# ===========================================================================
# Helpers
# ===========================================================================

_IMAGE_FORMATS = ("png", "jpg", "jpeg", "tif", "tiff", "bmp", "html")
_GRID_FORMATS = ("vtu", "vtk")


def _formats():
    """MOD_PLOT_FORMAT as list of extensions; None = interactive window."""
    if MOD_PLOT_FORMAT is None:
        return None
    raw = ([MOD_PLOT_FORMAT] if isinstance(MOD_PLOT_FORMAT, str)
           else list(MOD_PLOT_FORMAT))
    fmts = [str(f).lstrip(".").lower() for f in raw]
    bad = [f for f in fmts if f not in _IMAGE_FORMATS]
    if bad:
        raise ValueError(
            f"MOD_PLOT_FORMAT: {bad} not possible for a 3-D view; use "
            f"{', '.join(_IMAGE_FORMATS)} (grid files: MOD_3D_VTU).")
    return fmts


def _grid_format():
    """MOD_3D_VTU as extension, None = no grid file."""
    if not MOD_3D_VTU:
        return None
    ext = str(MOD_3D_VTU).lstrip(".").lower()
    if ext not in _GRID_FORMATS:
        raise ValueError(
            f"MOD_3D_VTU must be 'vtu', 'vtk' or None, got {MOD_3D_VTU!r}.")
    return ext


def _slices_m(values, name):
    """Slice positions km -> m (None passes through)."""
    if values is None:
        return None
    return [fem.km_to_m(v, name=name) for v in values]


def _planes_m(planes):
    """Oblique planes with "origin" converted from km to m."""
    out = []
    for spec in planes or []:
        spec = dict(spec)
        if spec.get("origin") is not None:
            spec["origin"] = [
                fem.km_to_m(v, name="MOD_3D_SLICE_PLANES origin")
                for v in spec["origin"]]
        out.append(spec)
    return out


def _scalar_values(values):
    """log10(Ohm.m) settings in the units of MOD_3D_SCALAR."""
    if values is None:
        return None
    if MOD_3D_SCALAR == "resistivity":
        return [10.0 ** float(v) for v in values]
    return [float(v) for v in values]


def _limits(run_dir, work_dir):
    """(xlim, ylim, zlim) in km: MOD_ROI_AUTO from the sites, else MOD_*LIM."""
    xlim, ylim, zlim = MOD_XLIM, MOD_YLIM, MOD_ZLIM
    if not MOD_ROI_AUTO:
        return xlim, ylim, zlim
    site_dat = fem.find_run_file(MOD_SITE_DAT, run_dir, work_dir)
    if MOD_SITE_DAT and site_dat is None:
        print(f"  MOD_SITE_DAT = {MOD_SITE_DAT!r} not found in {run_dir} or "
              f"{work_dir}.")
    have_origin = (MOD_ORIGIN_METHOD is not None
                   or (MOD_UTM_ORIGIN_E is not None
                       and MOD_UTM_ORIGIN_N is not None))
    origin_e, origin_n = fem.utm_km_to_m(
        MOD_UTM_ORIGIN_E, MOD_UTM_ORIGIN_N, name="MOD_UTM_ORIGIN_E / _N")
    sites = []
    if site_dat and have_origin:
        sites = fem.resolve_origin_and_sites(
            site_dat=site_dat,
            origin_e=origin_e,
            origin_n=origin_n,
            origin_lat=MOD_UTM_ORIGIN_LAT,
            origin_lon=MOD_UTM_ORIGIN_LON,
            zone_override=MOD_UTM_ZONE_OVERRIDE,
            origin_method=MOD_ORIGIN_METHOD,
            plot_sites=True,
        )[6]
    if not sites:
        print("  ROI: no sites / origin available - MOD_XLIM / YLIM / ZLIM "
              "used.")
        return xlim, ylim, zlim
    sx = [float(s[1]) / 1000.0 for s in sites]
    sy = [float(s[2]) / 1000.0 for s in sites]
    xlim = [min(sx) - MOD_ROI_PAD_XY, max(sx) + MOD_ROI_PAD_XY]
    ylim = [min(sy) - MOD_ROI_PAD_XY, max(sy) + MOD_ROI_PAD_XY]
    if MOD_ROI_ZLIM is not None:
        zlim = MOD_ROI_ZLIM
    if OUT:
        print(f"  ROI: x {xlim[0]:.2f} ... {xlim[1]:.2f} km, "
              f"y {ylim[0]:.2f} ... {ylim[1]:.2f} km")
    return xlim, ylim, zlim


def _plot_model(model, work_dir, *, fmts, grid_fmt, stem=None):
    """Scene(s) and grid file of one model.

    model : entry of fem.resolve_model_files(); fmts : image formats (None =
    interactive window); grid_fmt : "vtu" / "vtk" / None; stem : output path
    without extension (None = the naming scheme).
    """
    run_dir = model["dir"]
    mesh_file = fem.find_run_file(MOD_MESH, run_dir, work_dir)
    if mesh_file is None:
        raise FileNotFoundError(
            f"mesh MOD_MESH = {MOD_MESH!r} not found in {run_dir} or "
            f"{work_dir}.")
    struct = fem._read_resistivity_block_struct(
        model["file"], ocean=MOD_OCEAN, out=False)
    ocean = bool(struct["meta"]["ocean_present"])
    if OUT:
        print(f"  ocean: {'region 1' if ocean else 'none'}")

    xlim, ylim, zlim = _limits(run_dir, work_dir)

    if stem is None:
        stem = fem.model_output_stem(
            model, "3d", out_dir=OUT_DIR, prefix=OUT_PREFIX,
            prefix_dir=OUT_PREFIX_DIR, base_dir=work_dir)
    grid_file = None if grid_fmt is None else f"{stem}.{grid_fmt}"
    if grid_file or fmts:
        os.makedirs(os.path.dirname(os.path.abspath(stem)), exist_ok=True)

    # (plot_file, vtu_file) of every call; the grid file is written once
    if fmts is None:
        jobs = [(None, grid_file)]
    elif not fmts:
        jobs = [(grid_file, None)]
    else:
        jobs = [(f"{stem}.{f}", grid_file if i == 0 else None)
                for i, f in enumerate(fmts)]

    for plot_file, vtu_file in jobs:
        fviz.plot_model_3d(
            mesh_file=mesh_file,
            block_file=model["file"],
            scalar=MOD_3D_SCALAR,
            clim=_scalar_values(MOD_CLIM),
            cmap=MOD_CMAP,
            slice_x=_slices_m(MOD_3D_SLICE_X, "MOD_3D_SLICE_X"),
            slice_y=_slices_m(MOD_3D_SLICE_Y, "MOD_3D_SLICE_Y"),
            slice_z=_slices_m(MOD_3D_SLICE_Z, "MOD_3D_SLICE_Z"),
            slice_planes=_planes_m(MOD_3D_SLICE_PLANES),
            isovalues=_scalar_values(MOD_3D_ISOVALUES),
            iso_opacity=MOD_3D_ISO_OPACITY,
            iso_cmap=MOD_CMAP,
            ocean_value=MOD_OCEAN_RHO if ocean else None,
            air_region_index=0,
            ocean_region_index=1,
            xlim=fem.lim_km_to_m(xlim, name="MOD_XLIM"),
            ylim=fem.lim_km_to_m(ylim, name="MOD_YLIM"),
            zlim=fem.lim_km_to_m(zlim, name="MOD_ZLIM"),
            window_size=MOD_3D_WINDOW_SIZE,
            screenshot_scale=MOD_3D_SCREENSHOT_SCALE,
            plot_file=plot_file,
            vtu_file=vtu_file,
            out=OUT,
        )


# ===========================================================================
# Main
# ===========================================================================

def main(argv=None) -> int:
    """Plot all runs; 0 = done, 1 = something failed, 2 = bad WORK_DIR."""
    parser = argparse.ArgumentParser(
        description="3-D view / VTK export of single FEMTIC models.")
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

    fmts, grid_fmt = _formats(), _grid_format()
    if fmts == [] and grid_fmt is None:
        print("MOD_PLOT_FORMAT is empty and MOD_3D_VTU is None - nothing to "
              "do.")
        return 1
    if fviz.pv is None:
        print("PyVista is not available (conda install -c conda-forge "
              "pyvista) - nothing done.")
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
    if MOD_3D_FILE:
        if len(models) == 1:
            stem = fem.anchor_path(MOD_3D_FILE, work_dir)
            base, ext = os.path.splitext(stem)
            if ext.lstrip(".").lower() in _IMAGE_FORMATS + _GRID_FORMATS:
                stem = base
        else:
            print("MOD_3D_FILE is only used for a single model - ignored.")

    done = 0
    for i, model in enumerate(models, 1):
        print(f"[{i}/{len(models)}] {model['label']} / {model['tag']}: "
              f"{model['file']}")
        try:
            _plot_model(model, work_dir, fmts=fmts, grid_fmt=grid_fmt,
                        stem=stem)
            done += 1
        except Exception as exc:  # one bad run must not stop the others
            print(f"  FAILED: {type(exc).__name__}: {exc}")
        gc.collect()

    print(f"\n3-D plots: {done} of {len(models)} model(s) done")

    utl.write_param_summary(
        __file__,
        namespace=dict(globals(), WORK_DIR=work_dir, RUN_PATHS=run_paths),
        out_dir=work_dir)

    return 0 if done == len(models) else 1


if __name__ == "__main__":
    _rc = main(sys.argv[1:])
    if _rc:  # no SystemExit on success (keeps Spyder sessions quiet)
        sys.exit(_rc)
