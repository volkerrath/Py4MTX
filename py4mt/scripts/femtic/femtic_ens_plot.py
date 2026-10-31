#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_plot_ensemble.py — Ensemble slice plot for a set of FEMTIC inversion runs.

Scans ENSEMBLE_DIR for converged ensemble members exactly the way
femtic_ens_post.py does (ENSEMBLE_NAME* sub-directories, femtic.cnv,
NRMS_MAX threshold), then produces:

  (1) [default, PER_MEMBER_PLOT=True] Per converged member, two
      fviz.plot_model_slices() figures in their own sub-directory
      ENSEMBLE_DIR/<label>/: the perturbed prior model (iter0.<ext>,
      from resistivity_block_iter0.dat, colormap/scale PLOT_CMAP_ITER0
      / PLOT_CLIM_ITER0) and the best-fit model (best.<ext>, from
      resistivity_block_iter{numit}.dat, numit from femtic.cnv,
      colormap/scale PLOT_CMAP_BEST / PLOT_CLIM_BEST). One file per format in PLOT_FORMAT (e.g.
      ["pdf", "jpg"]). When "pdf" is among the formats,
      PER_MEMBER_PDF_CATALOG_MODE selects which per-member pdf pages
      (none / iter0 / best / both, interlaced per member) are also
      combined into one multi-page catalog PDF
      (PER_MEMBER_CATALOG_FILE).

  (2) Optionally, a borehole resistivity log figure (same as step (6) in
      femtic_mod_plot.py), sampled from the first converged member's
      best-fit model (ENS_MODEL_SOURCE = "perturbed": its perturbed
      reference), and/or (BOREHOLE_HEATMAP) a heat map of the whole
      ensemble along the same boreholes: at every depth the members'
      log10(rho) are histogrammed and shown as a colour image, with the
      median / quantile lines on top.

  (3) [optional, RESULTS_NPZ set] NPZ mode: instead of scanning the member
      directories, work from the results .npz written by femtic_ens_post.py:
      best-fit figures of the stored members (RESULTS_MEMBERS), figures of
      the stored ensemble statistics (RESULTS_STATS) and the borehole figure
      (RESULTS_BOREHOLE_MODEL), all with the same slices, PLOT_* styling,
      vertical exaggeration and reference model. The perturbed-prior (iter0)
      figures need the member directories and are not available there.

Slice positions, UTM/geographic coordinate handling, site overlay, and
all PLOT_* parameters follow the same conventions as femtic_mod_plot.py.
See that script and its README for full documentation.

Provenance
----------
    2026-05-24  vrath / Claude Sonnet 4.6   Created, based on
                femtic_mod_plot.py and snippets.py (Snippet 1).
                ENS_DIRS replaces ENS_FILES: the script loops over
                directories and builds the file list automatically.
    2026-05-31  vrath / Claude Sonnet 4.6   Aligned with femtic_mod_plot.py:
                replaced ESTIMATE_ORIGIN/CALIBRATION_SITES/UPDATE_CONFIG
                with ORIGIN_METHOD (None|"box"|"average"); origin estimation
                now runs before UTM zone derivation.  Removed local
                coordinate helpers (delegated to fem/utl).  site_xys tuples
                now carry elev.  plot_ensemble_slices call extended with
                site_xys, utm_origin_e/n, utm_zone, utm_northern,
                utm_to_latlon_fn, latlon_to_model_fn, display_coords,
                depth_km, horiz_km, equal_aspect, panel_height, nrows,
                ncols kwargs.  Added DEPTH_KM, HORIZ_KM, PLOT_EQUAL_ASPECT,
                PLOT_PANEL_HEIGHT, PLOT_NROWS, PLOT_NCOLS, PLOT_SITES_MAPS,
                PLOT_SITES_SLICES, SITE_MARKER_SLICES, MAP_MARKERS,
                DISPLAY_COORDS config vars.
    2026-08-13  Claude Sonnet 5 (Anthropic)
                Added femtic_ens_plot_summary.md output at end of run:
                writes user-set (UPPERCASE) parameters, script path, and
                run date/time via utl.write_param_summary().
    2026-08-14  Claude Sonnet 5 (Anthropic)
                Rewrote member handling to match femtic_ens_post.py:
                converged members are now discovered by scanning
                ENSEMBLE_DIR for "ENSEMBLE_NAME*" sub-directories and
                reading femtic.cnv (same NRMS_MAX threshold, same
                nRMS-column FEMTIC-version logic), replacing the old
                fixed ENS_DIRS/BLOCK_PATTERN/ENS_ITER file list.
                For each converged member, two single-model figures are
                now produced via fviz.plot_model_slices(): the perturbed
                prior (resistivity_block_iter0.dat) and the best-fit
                model (resistivity_block_iter{numit}.dat, numit from
                femtic.cnv), saved as "<label>_iter0.<ext>" and
                "<label>_best.<ext>" in ENSEMBLE_DIR. Controlled by new
                PER_MEMBER_PLOT flag (default True). The previous joint
                multi-row figure (fviz.plot_ensemble_slices, with
                mean/std/median summary rows) is kept as an optional
                extra, now gated by PLOT_JOINT (default False) and still
                built from the same converged-member file list. Its
                fviz.plot_ensemble_slices() call still has the
                previously flagged keyword mismatch against the
                function's current signature (would raise TypeError if
                PLOT_JOINT=True) — this remains out of scope here and
                is called out in a code comment at the call site.
    2026-08-15  Claude Sonnet 5 (Anthropic)
                Per-member output reworked: (1) PLOT_FORMAT replaces
                PER_MEMBER_FORMAT and now accepts a list, e.g.
                ["pdf", "jpg"] (normalised to _PLOT_FORMATS), mirroring
                MOD_PLOT_FORMAT in femtic_ens_post.py -- one savefig()
                call per format, figure rebuilt per format since
                plot_model_slices doesn't expose a re-save path.
                (2) Per-member files now live in their own sub-directory
                ENSEMBLE_DIR/<label>/ as "iter0.<ext>" / "best.<ext>",
                replacing the flat "<label>_iter0.<ext>" naming.
                (3) New PER_MEMBER_PDF_CATALOG flag (default True):
                when "pdf" is among _PLOT_FORMATS, every per-member pdf
                Figure (iter0 + best, in plot order) is additionally
                collected and written as one multi-page catalog via
                matplotlib.backends.backend_pdf.PdfPages, to
                PER_MEMBER_CATALOG_FILE.
    2026-08-16  Claude Sonnet 5 (Anthropic)
                Reduced peak memory for large ensembles: the pdf catalog
                is now opened once before the per-member loop and each
                pdf figure is written via PdfPages.savefig() and
                released (del + gc.collect()) immediately after it's
                built, instead of accumulating every member's Figure
                object in a Python list and writing them all at the end.
                _plot_member_slice() gained a pdf_catalog= kwarg and no
                longer returns a Figure.
    2026-08-20  Claude Sonnet 5 (Anthropic)
                _plot_member_slice(): the pdf-format savefig call is
                now wrapped in mpl.rc_context({"pdf.compression": 0})
                to disable matplotlib's PDF indexed-colour image path
                (backend_pdf.PdfFile._writeImg / np.searchsorted
                palette mapping, gated on the pdf.compression
                rcParam) at the source, plus a try/except IndexError
                fallback (render PNG, convert to PDF with Pillow) as
                a safety net in case the same class of bug (same code
                path as matplotlib/matplotlib#25806) is hit some other
                way. Data-/render-dependent, not a bug in this script
                or in plot_model_slices -- generated/reviewed by
                Claude, should be checked before relying on it in
                production.
    2026-08-23  Claude Sonnet 5 (Anthropic)
                Fixed the fallout of the 2026-08-20 indexed-colour
                workaround above: pdf.compression=0 was being applied
                pre-emptively to every per-member pdf save, making
                those files several times larger than necessary, and
                -- because the catalog page for the same figure was
                written by a separate PdfPages.savefig() call outside
                that rc_context -- the catalog page kept the default
                (compressed) size while the per-member file did not,
                which is why per-member files were much larger than
                the catalog. pdf.compression now stays at its default
                for both the per-member save and the catalog append;
                the compression-disabled retry (and, if that also
                fails, the PNG->PDF fallback) is only triggered
                reactively on an actual IndexError, for whichever
                specific save hit it. The catalog-append savefig()
                call is now guarded the same way, since it is an
                independent write of the same figure and can in
                principle hit the same bug on its own.
    2026-08-20  Claude Sonnet 5 (Anthropic)
                All user-facing length parameters in the Configuration
                section (PLOT_SLICES z0/x0/y0/point, PLOT_XLIM/YLIM/ZLIM,
                PROJECTION_DIST, MAP_MARKERS positions, BOREHOLE_SITES
                x/y/z_top/z_bot/dz) are now entered in kilometres instead
                of metres. A new "Unit conversion" block, run once right
                after the config section, converts everything to
                model-local metres before fem.resolve_slice_positions()
                or fviz.plot_model_slices()/plot_ensemble_slices() see
                it -- those functions and femtic_ens_post.py's own
                convention are unchanged (still metres internally).
                CRS-tagged specs, e.g. (value, "utm") / (value,
                "latlon"), are absolute coordinates and are left
                unchanged by the conversion; only plain numbers /
                implicit-"model" specs are treated as model-local km.
                Fixes a bug where PLOT_SLICES z0 values and
                PLOT_XLIM/YLIM/ZLIM were written with "# km" comments
                but silently consumed as metres (e.g. z0=25.0 plotted a
                slice at 25 m depth, not 25 km, and PLOT_XLIM=[-25, 25]
                restricted the map panels to a 50 m-wide box instead of
                50 km). PROJECTION_DIST and the BOREHOLE_SITES example
                were also updated from metres to km.
    2026-08-20  Claude Sonnet 5 (Anthropic)
                (1) PER_MEMBER_PDF_CATALOG (bool) replaced by
                PER_MEMBER_PDF_CATALOG_MODE ("none" | "iter0" | "best" |
                "both", default "both"), selecting which per-member pdf
                pages go into the catalog: none, prior-only, best-only,
                or both interlaced per member in plot order (iter0_A,
                best_A, iter0_B, best_B, ...) -- "both" reproduces the
                previous PER_MEMBER_PDF_CATALOG=True behaviour, since
                the per-member loop already visits iter0 then best for
                each member in turn. Validated against
                _VALID_CATALOG_MODES at load time.
                (2) PLOT_CLIM split into PLOT_CLIM_ITER0 and
                PLOT_CLIM_BEST so the perturbed-prior and best-fit
                model plots can use distinct log10(rho) colour scales
                (the prior's resistivity range often differs
                substantially from the inverted result). _plot_member_
                slice() gained a required clim= kwarg -- the two call
                sites in the per-member loop now pass PLOT_CLIM_ITER0 /
                PLOT_CLIM_BEST respectively; the joint ensemble figure
                (PLOT_JOINT), which only plots best-fit models, uses
                PLOT_CLIM_BEST.
                (3) PLOT_CMAP likewise split into PLOT_CMAP_ITER0 and
                PLOT_CMAP_BEST, mirroring (2), so the colormap itself
                -- not just the colour limits -- can differ between
                the prior and best-fit plots. _plot_member_slice()
                gained a required cmap= kwarg alongside clim=; the
                joint ensemble figure uses PLOT_CMAP_BEST.
    2026-08-25  Claude Sonnet 5 (Anthropic)
                Removed PLOT_JOINT and the entire joint multi-row
                ensemble figure it gated (the fviz.plot_ensemble_slices
                call in the former step (6b)), along with its
                now-unused config vars ENS_STAT_ROWS, PLOT_ENS_FILE,
                ENS_PER_MEMBER. This removes the previously flagged
                keyword-mismatch call (site_xys, obs_coords_only,
                utm_origin_e/n, panel_height, nrows, ncols, and others
                not present in plot_ensemble_slices()'s current
                signature) rather than fixing it, since PER_MEMBER_PLOT
                is the only path actually in use. PLOT_CMAP_BEST /
                PLOT_CLIM_BEST doc comments that referenced the joint
                figure updated accordingly; ENS_FILES is kept (still
                used by the borehole step, now (7)). Also added
                PLOT_PANEL_WIDTH (cm; None = auto from aspect ratio),
                mirroring femtic_ens_post.py's MOD_PANEL_WIDTH: when
                set it overrides plot_model_slices()'s per-column
                auto-width with one fixed value, taking precedence over
                PLOT_PANEL_HEIGHT-driven aspect-ratio sizing; when None
                (default, unchanged prior behaviour) PLOT_PANEL_HEIGHT
                alone drives sizing as before. Wired into the single
                shared _kwargs dict in _plot_member_slice(), so both
                the iter0 and best-fit per-member figures pick it up.
    2026-09-06  Claude Sonnet 5 (Anthropic)
                Added SHOW_MODEL_CENTRE (default True): marks the model
                origin on "map" panels whenever DISPLAY_COORDS is
                "utm"/"latlon" via fviz.plot_model_slices'
                show_model_centre parameter; override style with a
                MAP_MARKERS entry carrying "is_model_centre": True.
                Wired into the single shared _kwargs dict, so every
                plot_model_slices() call site here (iter0, best-fit, and
                the pdf-fallback retries) picks it up.
    2026-09-07  Claude Sonnet 5 (Anthropic)
                Removed the FEMTIC config variable and its "4.3"/"5."
                version-string switch on the nRMS column index (6 vs 8):
                that switch only matched one specific column layout, and
                silently misread nRMS (e.g. reading the Distortion or
                Misfit column instead) for any run whose actual column
                count didn't match the assumed FEMTIC-version pairing.
                Now uses fem.read_cnv(), matching femtic_ens_post.py's
                get_nrms()-based fix of 2026-09-02 -- column positions
                are read from femtic.cnv's own header row every time,
                independent of version or Beta/Distortion presence.
    2026-10-08  Claude Sonnet 5.5 (Anthropic)
                Convergence file name is now configurable: new CNV_FILE
                (default "femtic.cnv"; a list of candidate names is
                allowed, first existing wins), resolved per member via
                fem.resolve_cnv_path(). Requires the femtic.py of the same
                date. AI-generated; review before production use.
    2026-10-27  Claude Sonnet 5.5 (Anthropic)
                Publication-quality slices and an optional reference model,
                using the new femtic_viz.py of the same date.
                (1) Equal axes or a defined vertical exaggeration: new
                PLOT_VERT_EXAG (None = decided by PLOT_EQUAL_ASPECT: True ->
                1.0, False -> free aspect). The section aspect is computed
                from the real axis units, so it now also works for
                DISPLAY_COORDS="latlon" (before, PLOT_EQUAL_ASPECT had no
                effect there). Note: with the default PLOT_EQUAL_ASPECT=True
                and "latlon" the sections therefore change from free to true
                scale; set PLOT_EQUAL_ASPECT=False to keep the old look.
                The duplicate PLOT_EQUAL_ASPECT definition was removed.
                (2) New options passed to fviz.plot_model_slices, as in
                femtic_ens_post.py: PLOT_FIGSIZE, PLOT_TICK_FONTSIZE,
                PLOT_LABEL_FONTSIZE, PLOT_TITLE_FONTSIZE, PLOT_TICK_DECIMALS,
                PLOT_AIR_COLOR (defaults = previous output).
                (3) Optional reference model: REF_FILE (block file on the
                MESH_FILE mesh; None = off), REF_SLICES ("slices" | "best" |
                "none"), REF_BOREHOLE, REF_LEVELS, REF_STYLE, REF_LABEL.
                Slices: iso-resistivity lines on every panel; boreholes: a
                dashed reference profile per borehole.
                (4) The local plot_borehole_logs() was replaced by
                fviz.plot_borehole_logs(): resistivity on a log axis, depth
                in km pointing down (the local version inverted the axis
                once per borehole, so even counts pointed up), z_top="surface"
                (minimum mesh-node depth), per-spec line styles, NPZ export
                (BOREHOLE_NPZ).
                _km_to_m_point() accepts [x, y] points; z_top="surface" in
                BOREHOLE_SITES survives the km conversion.
                AI-generated code -- review before production use.
    2026-10-27b Claude Sonnet 5.5 (Anthropic)
                Can work from the results .npz of femtic_ens_post.py (also
                what its plot-only mode reads). New settings RESULTS_NPZ
                (None = scan member directories, as before), RESULTS_PREFIX,
                RESULTS_TEMPLATE, RESULTS_OUT_DIR, RESULTS_MEMBERS ("none" |
                "best" | "all" | indices / labels), RESULTS_STATS,
                RESULTS_STATS_STYLE, RESULTS_BOREHOLE_MODEL. In that mode no
                member directory, cnv or block file is read: each figure is a
                block file written (femtic.insert_model, template = the best
                member's structure) from a stored vector -- member rows of
                the ensemble matrix, avg / med / err / mad / p* / qdiff_* /
                *_boot / sens_* / simrc_* -- and drawn by the same
                _plot_member_slice() (new keywords stats_panel, cbar_label,
                ref_clim), so slices, layout, PLOT_VERT_EXAG, fonts and the
                reference model (REF_*) act as usual; statistic panels get
                the reference at resistivity decades (PLOT_CLIM_BEST), not at
                decades of the plotted statistic. The borehole figure uses
                RESULTS_BOREHOLE_MODEL ("best", a member index, "avg", "med",
                "p<level>"), validated before any figure is made. Not
                available from a results file: the iter0 prior figures
                (skipped, noted at run time); NRMS_MAX is not applied. Air /
                ocean of these figures are marked by exact sentinel values,
                so a free region with a resistivity near OCEAN_RHO is no
                longer drawn as ocean (in the normal mode fviz flags cells
                within 0.05 in log10 of OCEAN_RHO). The normal path is
                unchanged (figures and borehole output identical to the
                2026-10-27 version, tested).
                AI-generated code -- review before production use.
    2026-10-28  Claude Sonnet 5.5 (Anthropic)
                The parameter summary femtic_ens_plot_summary.md is now
                written to the working directory set in this script
                (ENSEMBLE_DIR) instead of next to the script:
                utl.write_param_summary(fname, out_dir=ENSEMBLE_DIR).
                Needs the util.py of the same date (new out_dir argument).
                AI-generated; review before production use.
    2026-10-29  Claude Sonnet 5.5 (Anthropic)
                Borehole heat map. New settings BOREHOLE_HEATMAP (default
                False), BOREHOLE_HEATMAP_FILE, BOREHOLE_HEAT_BINS,
                BOREHOLE_HEAT_NORM ("fraction" | "row_max"),
                BOREHOLE_HEAT_SCALE ("linear" | "sqrt" | "log" colour scale),
                BOREHOLE_HEAT_VMIN, BOREHOLE_HEAT_VMAX (explicit limits show
                as arrows on the colour bar; a log scale leaves empty bins
                blank), BOREHOLE_HEAT_CMAP,
                BOREHOLE_HEAT_OVERLAY ("avg" | "med" | "sigma1" | "sigma2" |
                "ref"), BOREHOLE_HEAT_STYLE, BOREHOLE_HEAT_PANEL_SIZE,
                BOREHOLE_HEAT_NCOLS, BOREHOLE_HEAT_NPZ and BOREHOLE_LINES
                (default True; False skips the fviz log figure). With
                PLOT_BOREHOLE and BOREHOLE_SITES the whole ensemble is sampled
                along each borehole (normal mode: the members' best-fit block
                files; NPZ mode: the stored ensemble matrix on the template's
                region structure) and drawn in this script, not by fviz
                (fviz.plot_borehole_logs shows a single model): exact
                point-in-tetrahedron search once per borehole, one lookup per
                member, per-depth histogram of log10(rho) over the
                BOREHOLE_XLIM range (members outside it are not shown; the
                fractions still refer to all valid members), pcolormesh with a
                shared colour bar. Auto colour maximum (BOREHOLE_HEAT_VMAX =
                None) = 99th percentile of the non-zero fractions, so levels
                with a single value in every member (ocean) do not wash out
                the rest. The
                sampling and the histogram are those of femtic_ens_post.py
                (MOD_BH item "heatmap"). The histograms go to an .npz
                (heat_<name>, heat_edges_<name>, depth_<name>). BOREHOLE_SITES
                entries are not extended (their extra keys would reach fviz as
                line keywords). With the new settings at their defaults the
                run is unchanged. AI-generated code -- review before
                production use.
    2026-10-31  Claude Sonnet 5.5 (Anthropic)
                New ENS_MODEL_SOURCE ("converged" default | "perturbed") and
                ENS_PERTURBED_FILE (default "referencemodel.dat"): the
                borehole figures can use each member's PERTURBED REFERENCE
                (the RTO prior draw femtic_rto_prep.py writes per member)
                instead of its converged model: the heat map
                (BOREHOLE_HEATMAP: all members) and the line log
                (BOREHOLE_LINES: the first member). The member set is
                unchanged (NRMS_MAX etc.); REF_FILE, the base unperturbed
                reference, is drawn as before. A missing file skips the
                member; a symbolic link (shared, unperturbed) is reported.
                The per-member iter0 / best-fit slice figures are not
                affected; a note is printed when resistivity_block_iter0.dat
                is a symbolic link (the "perturbed prior" figure then shows
                the shared template prior, not the member's draw). NPZ mode:
                ENS_MODEL_SOURCE is ignored, the file decides - it stores
                <P>_model_source since femtic_ens_post.py 2026-10-31 (older
                files: converged); titles, the heat-map header
                ("ensemble_source") and the member figure say so. With the
                default the run is unchanged. AI-generated code -- review
                before production use.

@author: vrath
"""

import os
import sys
import math
import gc
import inspect
from pathlib import Path

import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages

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
#: ENSEMBLE_DIR - Directory containing one sub-directory per ensemble member.
#: ENSEMBLE_NAME - Member sub-directories are matched via glob "<ENSEMBLE_NAME>*".
# ---------------------------------------------------------------------------
ENSEMBLE_DIR = r"/media/vrath/LargeBack/Ensembles/annecy2026/ensemble_gst_2/"

ENSEMBLE_NAME = "annecy_rnd_2_"


#: Mesh file — always required for plotting.

MESH_FILE = ENSEMBLE_DIR +"/templates/mesh.dat"

#: observe.dat — used by ESTIMATE_ORIGIN and as fallback for SITE_NUMBER.
OBSERVE_FILE = ENSEMBLE_DIR + "/templates/observe.dat"

#: Site list produced by mt_make_sitelist.py (WHAT_FOR="femtic").
#: Format (comma-separated, no header):
#:   name, lat, lon, elev, sitenum, easting, northing
#: Easting/northing are UTM metres; model-local x/y is derived via
#: fem.utm_to_model using the mesh-centre origin.
#: Set to None to fall back to the observe.dat / SITE_NUMBER path.
SITE_DAT = ENSEMBLE_DIR + "/templates/site.dat"   # set to None to disable

# ---------------------------------------------------------------------------
# Ensemble input — converged-member discovery
# ---------------------------------------------------------------------------
#: Member sub-directories are matched via glob "<ENSEMBLE_NAME>*". nRMS is
#: read from femtic.cnv via fem.read_cnv(), which parses column positions
#: from the file's own header row -- no FEMTIC-version setting needed.


#: Name of the FEMTIC convergence file inside each member directory. A single
#: name or a list of candidate names (first existing one is used).  Keep this
#: equal to CNV_FILE in femtic_ens_post.py.
CNV_FILE = "femtic.cnv"

#: Maximum normalised RMS accepted from the convergence file.  Keep this equal to
#: NRMS_MAX in femtic_ens_post.py so this script plots exactly the
#: members ens_post included in its ensemble statistics.
NRMS_MAX = 1.5

#: Labels for the member plots/filenames — one string per converged
#: member, in the order directories are found.  None → use each member
#: directory's basename.
ENS_LABELS = None

#: Which model of each member the ENSEMBLE-based borehole figures use (the
#: heat map, BOREHOLE_HEATMAP, and the line log of the first member,
#: BOREHOLE_LINES):
#:   "converged" (default) -- the member's best-fit model,
#:       resistivity_block_iter<N>.dat.
#:   "perturbed" -- the member's PERTURBED REFERENCE (the RTO prior draw): the
#:       file ENS_PERTURBED_FILE in the member directory. The members are still
#:       those of the converged inversion (NRMS_MAX etc.), so the prior and the
#:       converged ensemble compare like for like. REF_FILE (the base,
#:       unperturbed reference) is drawn as before, i.e. the perturbed models
#:       scatter around it.
#: The per-member slice figures are not affected (they show iter0 and best fit,
#: as before). In NPZ mode (RESULTS_NPZ) the file decides - make it with
#: femtic_ens_post.py ENS_MODEL_SOURCE = "perturbed"; this setting is ignored.
ENS_MODEL_SOURCE = "converged"      # "converged" | "perturbed"

#: File of the perturbed reference in each member directory. femtic_rto_prep.py
#: writes the RTO draw to "referencemodel.dat" (a real file per member).
#: "resistivity_block_iter0.dat" holds it only if the ensemble was prepared
#: with MOD_START_FROM_PERTURBED = True; otherwise it is a symbolic link to the
#: shared, unperturbed template (reported as a warning).
ENS_PERTURBED_FILE = "referencemodel.dat"

if ENS_MODEL_SOURCE not in ("converged", "perturbed"):
    raise ValueError(
        f"ENS_MODEL_SOURCE={ENS_MODEL_SOURCE!r}: use 'converged' or 'perturbed'.")
if not ENS_PERTURBED_FILE or os.path.basename(ENS_PERTURBED_FILE) != ENS_PERTURBED_FILE:
    raise ValueError(
        f"ENS_PERTURBED_FILE={ENS_PERTURBED_FILE!r}: give a file name inside the "
        f"member directory, without a path.")

# ---------------------------------------------------------------------------
# Work from the results .npz of femtic_ens_post.py  (optional)
# ---------------------------------------------------------------------------
#: None (default): scan ENSEMBLE_DIR for member directories as before.
#: Path of a results file (femtic_ens_post.py writes
#: <RESULTS_DIR>/<ENSEMBLE_PREFIX.upper()>_results.npz): NPZ MODE -- no member
#: directory, cnv file or block file of the members is read. The figures are
#: drawn from the vectors stored in the file, with the same slices, layout,
#: PLOT_* styling, vertical exaggeration and reference model as in the normal
#: mode:
#:   * per-member best-fit models (rows of the stored ensemble matrix, with
#:     their nRMS) -> RESULTS_MEMBERS,
#:   * ensemble statistics (mean, median, std, MAD, percentiles, ... as far
#:     as femtic_ens_post.py stored them) -> RESULTS_STATS,
#:   * the borehole figure (PLOT_BOREHOLE) from the model RESULTS_BOREHOLE_MODEL.
#: What the file does not hold cannot be plotted: the perturbed-prior (iter0)
#: figures need the member directories, so they are skipped in this mode, and
#: PER_MEMBER_PLOT / PER_MEMBER_PDF_CATALOG_MODE "iter0" do not apply
#: (a catalog mode "best" / "both" collects the member figures drawn here).
#: NRMS_MAX is not applied (the file holds the accepted members only).
#: Still needed: MESH_FILE (+ SITE_DAT / OBSERVE_FILE for the origin and the
#: site overlay), and a block file for the region structure (RESULTS_TEMPLATE).
RESULTS_NPZ = None   # e.g. ENSEMBLE_DIR + "results/ANNECY_RTO_results.npz"

#: Key prefix inside the file (femtic_ens_post.py's ENSEMBLE_PREFIX).
#: None = the only ensemble in the file.
RESULTS_PREFIX = None

#: Template block file (header, region structure, flag columns; the panels
#: are drawn from the stored vectors, not from this file's own values).
#: None = the best member's block file recorded in the .npz; set it when
#: that file is gone or the results were moved -- any resistivity block file
#: of this ensemble (same mesh and regions, e.g. the starting model) will do.
RESULTS_TEMPLATE = None

#: Output directory of NPZ mode: <dir>/<label>/best.<ext> per member, the
#: statistic figures <dir>/<prefix>_<key>.<ext>, the block files written from
#: the vectors (<dir>/blocks/) and the pdf catalog. Created if missing.
#: None = "ens_plot/" next to the results file.
RESULTS_OUT_DIR = None

#: Which stored members get a best-fit figure ("best.<ext>" in their own
#: sub-directory, labelled by ENS_LABELS or the member directory's name):
#: "none" | "best" (lowest nRMS, default) | "all" | a list of member indices
#: (0-based, order of the file) and/or labels, e.g. [0, 7, "ann_rto_03"].
RESULTS_MEMBERS = "best"

#: Statistic figures, by key: "avg", "med", "var", "err" (std), "mad",
#: "p<level>" (e.g. "p50", "p2_3"), "qdiff_<lo>_<hi>" (e.g. "qdiff_15_9_84_1"),
#: "var_boot" / "err_boot", "sens_<kind>" (e.g. "sens_mean_an"),
#: "simrc_coef" / "simrc_corr" -- whatever the file holds; a key without data
#: is skipped with a message. [] = none.
RESULTS_STATS = ["avg", "med", "err", "mad"]

#: Per-statistic overrides for the figures above, keyed as in RESULTS_STATS:
#: dict(cmap=..., clim=[vmin, vmax] | None, label="colourbar label"). Value
#: statistics (avg, med, p*) default to PLOT_CMAP_BEST / PLOT_CLIM_BEST /
#: "log10(rho / Ohm*m)"; all others to PLOT_CMAP_BEST, auto colour limits and
#: the key as label. The sensitivity and SimRC statistics are log10 (the
#: cv_* ones linear).
RESULTS_STATS_STYLE = {
    "var": dict(clim=[0.0, 0.3], label="var log10(rho)"),
    "err": dict(clim=[0.0, 0.3], label="std log10(rho)"),
    "mad": dict(clim=[0.0, 0.3], label="MAD log10(rho)"),
}

#: Model drawn in the borehole figure (PLOT_BOREHOLE) in NPZ mode: "best"
#: (lowest nRMS), "avg", "med", "p<level>", or a member index. (In the normal
#: mode it is the first converged member.)
RESULTS_BOREHOLE_MODEL = "best"

# ---------------------------------------------------------------------------
# Per-member plots (default): iter0 (perturbed prior) + best-fit model
# ---------------------------------------------------------------------------
#: If True, produce two fviz.plot_model_slices() figures for every
#: converged member: "iter0.<ext>" (resistivity_block_iter0.dat, the
#: perturbed/prior model) and "best.<ext>" (resistivity_block_iter{numit}.dat,
#: the best-fit model at the iteration femtic.cnv reports).  Each member
#: gets its own sub-directory ENSEMBLE_DIR/<label>/.
PER_MEMBER_PLOT = True

#: Output format(s) for per-member plots (Matplotlib Agg-backend savefig
#: formats).  A list saves one file per format per plot (the figure is
#: still rebuilt per format, since plot_model_slices doesn't expose a way
#: to re-save an already-built figure), e.g. ["pdf", "jpg"] writes both a
#: vector version and a raster preview. A bare string ("pdf") also works
#: and is treated as a single-entry list.
#: Supported values: "pdf", "svg", "eps" (vector); "png", "jpg"/"jpeg",
#: "tif"/"tiff", "webp" (raster, rendered at PLOT_DPI).
PLOT_FORMAT = ["pdf"]

#: Normalised to a list regardless of whether a bare string or a list was
#: set above.
_PLOT_FORMATS = (
    [PLOT_FORMAT] if isinstance(PLOT_FORMAT, str) else list(PLOT_FORMAT)
)

#: Controls which per-member figures (if any) are additionally collected
#: into one multi-page catalog PDF via matplotlib.backends.backend_pdf.
#: PdfPages, saved as PER_MEMBER_CATALOG_FILE. Only takes effect when
#: "pdf" is among _PLOT_FORMATS. Does not affect the individual
#: per-member files, which are still written as usual regardless of
#: this setting. One of:
#:   "none"  — no catalog is built
#:   "iter0" — catalog contains only the perturbed-prior (iter0) pages
#:   "best"  — catalog contains only the best-fit pages
#:   "both"  — catalog contains both, interlaced per member in plot
#:             order (iter0_A, best_A, iter0_B, best_B, ...)
PER_MEMBER_PDF_CATALOG_MODE = "both"   # "none" | "iter0" | "best" | "both"

#: Output path for the multi-page pdf catalog.
PER_MEMBER_CATALOG_FILE = ENSEMBLE_DIR + ENSEMBLE_NAME+"_catalog.pdf"

_VALID_CATALOG_MODES = ("none", "iter0", "best", "both")
if PER_MEMBER_PDF_CATALOG_MODE not in _VALID_CATALOG_MODES:
    raise ValueError(
        f"PER_MEMBER_PDF_CATALOG_MODE={PER_MEMBER_PDF_CATALOG_MODE!r} — "
        f"must be one of {_VALID_CATALOG_MODES}."
    )

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
#: Fallback values used only when ORIGIN_METHOD is None or SITE_DAT is
#: unavailable.  When ORIGIN_METHOD is "box" or "average" these are
#: overwritten at runtime from site.dat and may be left as None.
UTM_ORIGIN_LAT = None      # decimal degrees, positive = North  (None → derived)
UTM_ORIGIN_LON = None      # decimal degrees, positive = East   (None → derived)

UTM_ORIGIN_E   = None      # easting  [m]  (None → derived from site.dat)
UTM_ORIGIN_N   = None      # northing [m]  (None → derived from site.dat)

#: Override the auto-derived UTM zone number.  None = auto from origin lat/lon.
UTM_ZONE_OVERRIDE = None

# ---------------------------------------------------------------------------
# Display coordinate system
# ---------------------------------------------------------------------------
#: "model"  — axis ticks in model-local metres (origin = 0, default)
#: "utm"    — axis ticks in absolute UTM metres
#: "latlon" — axis ticks in decimal degrees (lon for easting, lat for northing)
DISPLAY_COORDS = "latlon"

# ---------------------------------------------------------------------------
# Axis scaling and layout
# ---------------------------------------------------------------------------
#: True → depth axis in km; False → metres.
DEPTH_KM = True

#: True → horizontal axes in km (model/utm modes); False → metres.
HORIZ_KM = True

#: Equal axes.  True = map panels 1:1 and the section panels (ns / ew /
#: vertical plane) at true scale, i.e. vertical exaggeration 1, ALSO when
#: DISPLAY_COORDS is "latlon" (the aspect is computed from the real axis
#: units, so degree tick labels do not matter).  False = free aspect (each
#: panel fills its box) unless PLOT_VERT_EXAG is set.
PLOT_EQUAL_ASPECT = True

#: Vertical exaggeration of the section panels (ns, ew, vertical "plane"):
#: None = decided by PLOT_EQUAL_ASPECT (True -> 1.0, False -> free aspect).
#: A number VE > 0 fixes the section aspect exactly, whatever
#: PLOT_EQUAL_ASPECT says: 1.0 = equal axes, 2.0 = the vertical scale is
#: twice the horizontal one, 0.5 = compressed.  Map panels are not affected
#: (they stay 1:1 while PLOT_EQUAL_ASPECT is True, free otherwise).  The panel
#: width follows from the real extents, so PLOT_XLIM / PLOT_YLIM / PLOT_ZLIM
#: (or the slice's own xlim / zlim) should be set.
PLOT_VERT_EXAG = None

#: Panel height in cm.  Width auto-computed from axis limits when PLOT_EQUAL_ASPECT.
PLOT_PANEL_HEIGHT = 16.0   # cm

#: None = auto per-column width from each panel's own aspect ratio (needs
#: PLOT_EQUAL_ASPECT=True and real xlim/ylim/zlim). When set, overrides
#: that auto-width with one fixed value for every column -- same
#: precedence as femtic_ens_post.py's MOD_PANEL_WIDTH: PLOT_PANEL_WIDTH
#: wins if not None, otherwise PLOT_PANEL_HEIGHT alone drives sizing via
#: the aspect-ratio auto-width.
PLOT_PANEL_WIDTH  = None   # cm; None = auto from aspect ratio

#: [width, height] of the whole figure in cm; None = from the panel sizes
#: above.  When set it overrides PLOT_PANEL_HEIGHT / PLOT_PANEL_WIDTH.
PLOT_FIGSIZE = None   # cm

#: Grid layout.  None → 1 row / len(PLOT_SLICES) columns.
PLOT_NROWS = 4
PLOT_NCOLS = 2

#: Annotation sizes (points), as in femtic_ens_post.py (MOD_TICK_FONTSIZE ...)
#: -- for journal figures set them to the final print size.  The defaults are
#: those of fviz.plot_model_slices, i.e. unchanged output.
PLOT_TICK_FONTSIZE = 7     # axis tick labels, colourbar ticks
PLOT_LABEL_FONTSIZE = 8    # axis labels, panel titles, colourbar label
#: Figure title (above the panels); None = PLOT_LABEL_FONTSIZE + 6.
PLOT_TITLE_FONTSIZE = None

#: Decimal digits of the axis tick labels (depth, easting/northing and
#: lat/lon share it).  None = fviz's own per-axis formatting, unchanged.
PLOT_TICK_DECIMALS = None

#: Colour of air cells (and of the area above the topography) in the slices.
PLOT_AIR_COLOR = "white"

# ---------------------------------------------------------------------------
# Site overlay
# ---------------------------------------------------------------------------
#: Site names to overlay from SITE_DAT.  None = all sites in the file.
SITE_NAMES = None   # e.g. ["MT01", "MT05", "MT12"]  or None = all sites

#: Fallback (when SITE_DAT is None): 1-based site number(s) from observe.dat.
#: Int or list of int.  None = no overlay.
SITE_NUMBER = None

#: Show site markers on map panels.
PLOT_SITES_MAPS   = True
#: Show site markers on curtain (ns/ew) panels.
PLOT_SITES_SLICES = False

#: Maximum distance (km) from slice plane for site projection onto curtains.
PROJECTION_DIST = 5.0

#: Marker style for map panels.
SITE_MARKER = dict(marker="v", color="black", ms=8, zorder=10, label=None)

#: Marker style for curtain panels (None → same as SITE_MARKER).
SITE_MARKER_SLICES = None

#: Additional map markers (e.g. known features).  List of dicts:
#:   dict(pos=(x, y), marker="*", color="red", ms=10, label="label")
#: pos accepts model-local km or (value, "utm"/"latlon") tuples (the
#: latter are absolute coordinates -- UTM metres / decimal degrees --
#: and are not affected by the km convention).
MAP_MARKERS = []

#: Mark the model origin (model-local x=0, y=0) on every "map" panel
#: whenever DISPLAY_COORDS is "utm"/"latlon" (no effect for "model").
#: Override style via a MAP_MARKERS entry with "is_model_centre": True.
SHOW_MODEL_CENTRE = True

# ---------------------------------------------------------------------------
# Verbose output
# ---------------------------------------------------------------------------
OUT = True

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
#: Figure DPI for saved files.
PLOT_DPI = 600

#: Matplotlib colormap name — set separately for the perturbed-prior
#: (iter0) and best-fit models, same reasoning as PLOT_CLIM_ITER0 /
#: PLOT_CLIM_BEST below.
PLOT_CMAP_ITER0 = "turbo_r"
PLOT_CMAP_BEST  = "turbo_r"

#: Colour limits [log10(ρ_min), log10(ρ_max)] — None = auto. Set
#: separately for the perturbed-prior (iter0) and best-fit models, since
#: the prior's resistivity range often differs from the inverted result.
PLOT_CLIM_ITER0 = [0.0, 4.0]      # log10(Ω·m)
PLOT_CLIM_BEST  = [0.0, 4.0]      # log10(Ω·m)

#: Flat colour for ocean / lake cells.  None → use colormap.
PLOT_OCEAN_COLOR = "lightgrey"
MOD_AIR_COLOR = "whitesmoke"
#: Axes facecolor for air / background.  None = figure default.
PLOT_AIR_BGCOLOR = "whitesmoke"

#: Slice specification — same format as femtic_mod_plot.py PLOT_SLICES.
#: All lengths below are given in KILOMETRES; the "Unit conversion" block
#: further down converts them to model-local metres before fem/fviz see
#: them.  CRS-tagged specs -- (value, "utm") in UTM metres, (value,
#: "latlon") in decimal degrees -- are absolute coordinates and are left
#: unchanged by that conversion.
#: Each dict must have 'kind' and the matching position key:
#:   kind="map"   → z0   (depth in model-local km)
#:   kind="ns"    → x0   (easting;  plain float = model-local km,
#:                        or (value, "utm") / (value, "latlon"))
#:   kind="ew"    → y0   (northing; same CRS tagging)
#:   kind="plane" → point ([x, y, z] model-local km, or ([x,y,z], "utm"
#:                  / "latlon") with z still model-local km), strike, dip
#:   invert_x     → True to flip horizontal axis on ns/ew/plane panels
#:                  (for comparison with sections using opposite convention)
# PLOT_SLICES = [
#     dict(kind="map",  z0=5.0),
#     dict(kind="map",  z0=15.0),
#     dict(kind="ns",   x0=(-70.8700, "latlon")),
#     dict(kind="ew",   y0=(-16.3500, "latlon")),
# ]
PLOT_SLICES = [    
    dict(kind="map", z0=0.0),    # km
    dict(kind="map", z0=5.0),    # km
    dict(kind="map", z0=10.0),   # km
    dict(kind="map", z0=15.0),   # km
    dict(kind="map", z0=20.0),   # km
    dict(kind="map", z0=25.0),   # km
    dict(kind="ns",  x0=(-71.40723, 'latlon')),    # deg
    dict(kind="ew",  y0=(-16.299593, 'latlon')),    # deg
    dict(kind="plane", point=[0.0, 0.0], strike=45.0, dip=90.0),]

#: Global axis limits in model-local KILOMETRES.  None → auto.
PLOT_XLIM = [-25., 25.]
PLOT_YLIM = [-25., 25.]
PLOT_ZLIM = [  -6., 30.]

# ---------------------------------------------------------------------------
# Reference model  (optional)
# ---------------------------------------------------------------------------
#: A model to compare every member with: a resistivity block file
#: (resistivity_block_iterX.dat) on the SAME mesh as MESH_FILE -- e.g. a
#: deterministic inversion, the starting model or a published model.  Its
#: region structure may differ from the members'.  None = no reference (the
#: options below are then ignored).  A reference on another mesh stops the
#: run with a message.
REF_FILE = None   # e.g. ENSEMBLE_DIR + "/reference/resistivity_block_iter20.dat"

#: Which per-member figures get the reference:
#:   "slices" - iso-resistivity lines on every slice panel (iter0 and best)
#:   "best"   - the same, on the best-fit figures only
#:   "none"   - not on the slices
REF_SLICES = "slices"   # "slices" | "best" | "none"

#: Borehole figure: add the reference as a dashed line, one per borehole, in the
#: colour of that borehole's trace.
REF_BOREHOLE = True

#: Contour levels, log10(Ohm.m), e.g. [1.0, 2.0, 3.0].  None = the integer
#: decades inside the colour limits of the figure being drawn.
REF_LEVELS = None

#: Keywords of the slice lines (matplotlib LineCollection), e.g.
#: dict(colors="k", linewidths=0.8, linestyles="--").
REF_STYLE = dict(colors="k", linewidths=0.8)

#: Legend entry of the slice lines (lower right of each panel); None = none.
REF_LABEL = "reference model"

# ---------------------------------------------------------------------------
# Borehole resistivity logs  (optional — same as step 6 in femtic_mod_plot.py)
# ---------------------------------------------------------------------------
#: Set True to produce a borehole figure after the ensemble plot.
PLOT_BOREHOLE = False

#: Output file for the borehole figure.  None → interactive show().
BOREHOLE_FILE = ENSEMBLE_DIR + "ensemble_boreholes.pdf"

#: List of borehole spec dicts — same format as femtic_mod_plot.py.
#: Keys: "name", "x", "y", "z_top", "z_bot", "dz" — all lengths in km.
#: x/y accept plain float (model-local km) or (value, "utm"/"latlon")
#: tuples (absolute coordinates, left unchanged by the km conversion).
#: z_top may also be "surface" (the minimum mesh-node depth at that position;
#: with an air layer in the mesh that is the top of the air, so give a number
#: when you want the ground surface).
#: Any Line2D keyword in a spec (color, ls, lw, ...) styles that trace.
#: The figure is drawn by fviz.plot_borehole_logs; with REF_FILE set and
#: REF_BOREHOLE True every borehole gets a dashed reference profile.
BOREHOLE_SITES = [
    # dict(name="BH-01", x=0.0, y=0.0, z_top=0.0, z_bot=20.0, dz=0.2),
]

#: Matplotlib line style for borehole traces.
BOREHOLE_STYLE = dict(lw=1.2, marker="none")

#: x-axis limits [log10 min, log10 max] for borehole panels.  None = auto.
BOREHOLE_XLIM = [0.0, 4.0]

#: True = all boreholes on one axes; False = one panel per borehole.
BOREHOLE_SHARED = True

#: True = also save the sampled profiles (depth_<name>, rho_<name>, and
#: reference_rho_<name> when a reference is drawn) as an .npz next to
#: BOREHOLE_FILE (same name, extension .npz).  Needs BOREHOLE_FILE.
BOREHOLE_NPZ = True

#: True = draw the log figure above (one model, fviz.plot_borehole_logs; file
#: BOREHOLE_FILE).  False = skip it, e.g. to make only the heat map.
BOREHOLE_LINES = True

# --- Heat map of the whole ensemble ------------------------------------------
#: True = additionally draw a HEAT MAP of the ensemble along the same boreholes
#: (BOREHOLE_SITES; needs PLOT_BOREHOLE): at every depth level the log10(rho) of
#: all members is histogrammed and shown as a colour image (resistivity on the
#: log x axis, depth pointing down), one panel per borehole, one shared colour
#: bar.  The members are the converged ensemble (normal mode: their best-fit
#: block files; NPZ mode: the rows of the stored ensemble matrix).  Drawn
#: here, not by fviz; independent of BOREHOLE_LINES.  Air and levels outside
#: the mesh are left blank.  Keys of a BOREHOLE_SITES entry used by the heat
#: map: "name", "x", "y", "z_top", "z_bot", "dz" (defaults: z_top "surface",
#: z_bot 20 km, dz 50 m; z_top="surface" is the top of the mesh column, as
#: for the log figure, i.e. the top of the air if the mesh has an air layer).
BOREHOLE_HEATMAP = False

#: Output file of the heat map (extension = format).  None -> interactive show().
BOREHOLE_HEATMAP_FILE = ENSEMBLE_DIR + "ensemble_boreholes_heatmap.pdf"

#: Number of bins in log10(rho) across BOREHOLE_XLIM (the data range when
#: BOREHOLE_XLIM is None), or a list of log10(rho) bin edges.  Members outside
#: the range are not shown (e.g. ocean cells at 0.25 Ohm.m with BOREHOLE_XLIM
#: [0, 4]); the fractions still refer to all valid members, so such a level
#: sums to less than 1.
BOREHOLE_HEAT_BINS = 60

#: Colour scale:  "fraction" = fraction of the valid members per bin (a depth
#: level sums to 1 if all members are inside the range);  "row_max" = every
#: depth level scaled to its own maximum (1 = most populated bin).
BOREHOLE_HEAT_NORM = "fraction"

#: Shape of the colour scale between BOREHOLE_HEAT_VMIN and BOREHOLE_HEAT_VMAX:
#:   "linear" : colour proportional to the value (default)
#:   "sqrt"   : square-root scale -- lifts the low densities, keeps the peaks
#:              distinguishable
#:   "log"    : logarithmic scale over the decades between VMIN and VMAX
#:              (default VMIN = smallest non-zero value of the figure, i.e. one
#:              member); empty bins (0) are left blank.
BOREHOLE_HEAT_SCALE = "linear"

#: Lower end of the colour scale.  None = 0 ("linear" / "sqrt"); the smallest
#: non-zero value of the figure ("log"; a log scale needs a value > 0).  A
#: number sets it explicitly, e.g. 0.02 to blank out bins holding less than 2 %
#: of the members; values below it get the lowest colour (an arrow on the
#: colour bar), and with "log" they are left out.
BOREHOLE_HEAT_VMIN = None

#: Upper end of the colour scale ("fraction" norm; "row_max" is always 1).
#: None = for "linear" / "sqrt" the 99th percentile of the non-zero fractions
#: of the figure, so that a few levels in which every member has the same value
#: (the ocean, fixed regions: fraction 1) do not wash out the rest (the colour
#: bar then ends in an arrow); for "log" the maximum.  1.0 = the full range.
BOREHOLE_HEAT_VMAX = None

#: Matplotlib colormap of the heat map (light = few members).  A single-hue
#: light-to-dark map ("Greys", "Blues", "Purples", ...) keeps the overlay lines
#: readable.
BOREHOLE_HEAT_CMAP = "Greys"

#: Lines drawn on top of the heat map, in this order: "avg" (mean of log10 rho),
#: "med" (median), "sigma1" (P15.87 / P84.13), "sigma2" (P2.28 / P97.72) and
#: "ref" (the reference model, needs REF_FILE and REF_BOREHOLE).  [] = none.
BOREHOLE_HEAT_OVERLAY = ["med", "sigma1"]

#: Matplotlib Line2D keywords per overlay item.
BOREHOLE_HEAT_STYLE = {
    "avg": dict(color="black", lw=1.6, ls="-"),
    "med": dict(color="crimson", lw=1.8, ls="-"),
    "sigma1": dict(color="navy", lw=1.2, ls="--"),
    "sigma2": dict(color="steelblue", lw=1.0, ls=":"),
    "ref": dict(color="black", lw=1.8, ls="--"),
}

#: Panel size [width, height] in cm and number of panel columns.
BOREHOLE_HEAT_PANEL_SIZE = [8.0, 16.0]
BOREHOLE_HEAT_NCOLS = 3

#: True = also save the histograms (heat_<name>, heat_edges_<name>, depth_<name>,
#: n_members) as an .npz next to BOREHOLE_HEATMAP_FILE (same name, extension
#: .npz).  Needs BOREHOLE_HEATMAP_FILE.
BOREHOLE_HEAT_NPZ = True

# ---------------------------------------------------------------------------
# Mesh-centre estimation from site.dat  (optional)
# ---------------------------------------------------------------------------
#: Method used to estimate UTM_ORIGIN_E / UTM_ORIGIN_N from SITE_DAT:
#:   None      — use the hard-coded UTM_ORIGIN_E / UTM_ORIGIN_N above
#:   "box"     — midpoint of the UTM bounding box of all sites (femticPY-compatible)
#:   "average" — arithmetic mean of all site UTM coordinates
#: Requires SITE_DAT to be set and readable.
ORIGIN_METHOD = "box"   # None | "box" | "average"


# ===========================================================================
# Unit conversion: user parameters above are in km  →  metres below
# ===========================================================================
# fem.resolve_slice_positions() and fviz.plot_model_slices() /
# plot_ensemble_slices() require model-local metres (the convention shared
# with femtic_ens_post.py and femtic.py, left unchanged here). Everything
# the user sets above -- PLOT_SLICES z0/x0/y0/point, PLOT_XLIM/YLIM/ZLIM,
# PROJECTION_DIST, MAP_MARKERS positions, BOREHOLE_SITES x/y/z_top/z_bot/dz
# -- is in km, so it is converted exactly once here, immediately after the
# config section. Nothing past this point needs to change; the rest of the
# script (including femtic.py / femtic_viz.py) still works in metres.
#
# CRS-tagged position specs -- (value, "utm") in absolute UTM metres, or
# (value, "latlon") in decimal degrees -- are absolute coordinates, not
# model-local lengths, so they pass through unchanged. Only plain numbers
# (or explicit ("model") tags) are treated as model-local km.

def _km_to_m(v):
    """Convert a plain scalar length from km to m; None passes through."""
    return None if v is None else float(v) * 1000.0


def _km_to_m_lim(lim):
    """Convert an [lo, hi] limit pair from km to m; None (either) passes."""
    if lim is None:
        return None
    return [None if x is None else float(x) * 1000.0 for x in lim]


def _km_to_m_pos(raw):
    """Convert an x0/y0-style position spec from km to m.

    Plain numbers and explicit ("model") tags are model-local km and are
    converted; ("utm") / ("latlon") tags are absolute coordinates and are
    returned unchanged.
    """
    if isinstance(raw, (int, float)):
        return float(raw) * 1000.0
    if isinstance(raw, (list, tuple)) and len(raw) == 2 and isinstance(raw[1], str):
        val, crs = raw
        return (float(val) * 1000.0, crs) if crs == "model" else raw
    raise ValueError(f"_km_to_m_pos: unexpected position spec {raw!r}")


def _km_to_m_point(raw):
    """Convert a plane-slice 'point' spec ([x, y, z] or [x, y], optionally
    CRS-tagged) from km to m. z (if given) is always model-local and is
    always converted; x/y are converted only under an implicit/explicit
    "model" CRS -- ("utm") / ("latlon") x/y are absolute and left unchanged.
    """
    tagged = (isinstance(raw, (list, tuple)) and len(raw) == 2
              and isinstance(raw[1], str))
    pt, crs = (raw[0], raw[1]) if tagged else (raw, "model")
    pt = list(pt)
    if crs == "model":
        pt[:2] = [v * 1000.0 for v in pt[:2]]
    if len(pt) > 2:
        pt[2] = pt[2] * 1000.0
    return (pt, crs) if tagged else pt


def _km_to_m_slice(spec: dict) -> dict:
    """Return a copy of one PLOT_SLICES entry with lengths converted km → m."""
    s = dict(spec)
    if "z0" in s:
        s["z0"] = _km_to_m(s["z0"])
    if "x0" in s:
        s["x0"] = _km_to_m_pos(s["x0"])
    if "y0" in s:
        s["y0"] = _km_to_m_pos(s["y0"])
    if "point" in s:
        s["point"] = _km_to_m_point(s["point"])
    for _k in ("xlim", "ylim", "zlim"):
        if _k in s:
            s[_k] = _km_to_m_lim(s[_k])
    return s


PLOT_SLICES = [_km_to_m_slice(_s) for _s in PLOT_SLICES]

PLOT_XLIM = _km_to_m_lim(PLOT_XLIM)
PLOT_YLIM = _km_to_m_lim(PLOT_YLIM)
PLOT_ZLIM = _km_to_m_lim(PLOT_ZLIM)

PROJECTION_DIST = _km_to_m(PROJECTION_DIST)

_converted_markers = []
for _mk in MAP_MARKERS:
    _mk = dict(_mk)
    if "pos" in _mk:
        _mk["pos"] = _km_to_m_pos(_mk["pos"])
    _converted_markers.append(_mk)
MAP_MARKERS = _converted_markers

_converted_boreholes = []
for _bh in BOREHOLE_SITES:
    _bh = dict(_bh)
    if "x" in _bh:
        _bh["x"] = _km_to_m_pos(_bh["x"])
    if "y" in _bh:
        _bh["y"] = _km_to_m_pos(_bh["y"])
    for _k in ("z_top", "z_bot", "dz"):
        # z_top may be the string "surface" (minimum mesh-node depth there)
        if _k in _bh and not isinstance(_bh[_k], str):
            _bh[_k] = _km_to_m(_bh[_k])
    _converted_boreholes.append(_bh)
BOREHOLE_SITES = _converted_boreholes

# Reference model: a missing file is reported once and switches it off;
# REF_SLICES is validated here so a typo fails before any figure is made.
if REF_SLICES not in ("slices", "best", "none"):
    raise ValueError(
        f"REF_SLICES={REF_SLICES!r} -- must be 'slices', 'best' or 'none'.")
if REF_FILE is not None and not os.path.isfile(REF_FILE):
    print(f"  WARNING: REF_FILE not found ({REF_FILE}) -- reference off.")
    REF_FILE = None

# Borehole heat map settings: validated here, so a typo fails before any
# figure is made.
if BOREHOLE_HEAT_NORM not in ("fraction", "row_max"):
    raise ValueError(
        f"BOREHOLE_HEAT_NORM={BOREHOLE_HEAT_NORM!r} -- must be 'fraction' "
        f"or 'row_max'.")
if BOREHOLE_HEAT_SCALE not in ("linear", "sqrt", "log"):
    raise ValueError(
        f"BOREHOLE_HEAT_SCALE={BOREHOLE_HEAT_SCALE!r} -- must be 'linear', "
        f"'sqrt' or 'log'.")
if BOREHOLE_HEAT_VMIN is not None and BOREHOLE_HEAT_VMIN < 0:
    raise ValueError("BOREHOLE_HEAT_VMIN must be >= 0 (or None).")
if (BOREHOLE_HEAT_SCALE == "log" and BOREHOLE_HEAT_VMIN is not None
        and BOREHOLE_HEAT_VMIN <= 0):
    raise ValueError("BOREHOLE_HEAT_SCALE='log' needs BOREHOLE_HEAT_VMIN > 0.")
_bad_overlay = [k for k in (BOREHOLE_HEAT_OVERLAY or [])
                if k not in ("avg", "med", "sigma1", "sigma2", "ref")]
if _bad_overlay:
    raise ValueError(
        f"BOREHOLE_HEAT_OVERLAY: unknown item(s) {_bad_overlay} -- use "
        f"'avg', 'med', 'sigma1', 'sigma2', 'ref'.")


# ===========================================================================
# Borehole helper
# ===========================================================================
# The borehole figure itself is drawn by fviz.plot_borehole_logs (log x axis,
# depth in km pointing down, optional reference model, NPZ export); only the
# position resolution stays here, because it needs this script's UTM origin.

def _resolve_borehole_xy(spec: dict) -> tuple[float, float]:
    """Resolve borehole x/y position specs to model-local metres.

    Passed to fviz.plot_borehole_logs as ``resolve_xy_fn``; plain numbers
    are model-local metres (already converted from km above), CRS-tagged
    specs are absolute and resolved against the UTM origin.
    """
    return (
        fem.resolve_pos_x(spec["x"], UTM_ZONE, UTM_NORTHERN,
                          UTM_ORIGIN_E, UTM_ORIGIN_N,
                          UTM_ORIGIN_LAT, UTM_ORIGIN_LON),
        fem.resolve_pos_y(spec["y"], UTM_ZONE, UTM_NORTHERN,
                          UTM_ORIGIN_E, UTM_ORIGIN_N,
                          UTM_ORIGIN_LAT, UTM_ORIGIN_LON),
    )


def _vert_exag():
    """Vertical exaggeration passed to fviz: PLOT_VERT_EXAG, else 1.0 for
    PLOT_EQUAL_ASPECT=True, else None (free aspect)."""
    if PLOT_VERT_EXAG is not None:
        return float(PLOT_VERT_EXAG)
    return 1.0 if PLOT_EQUAL_ASPECT else None


def _ref_levels(clim):
    """Reference contour levels: REF_LEVELS, else the integer decades
    strictly inside ``clim`` (None -> left to fviz: the figure's own range)."""
    if REF_LEVELS is not None:
        return [float(v) for v in np.atleast_1d(REF_LEVELS)]
    if clim is None:
        return None
    lo, hi = float(min(clim)), float(max(clim))
    lev = np.arange(math.floor(lo) + 1, math.ceil(hi), dtype=float)
    return [float(v) for v in lev] if lev.size else [0.5 * (lo + hi)]


# ===========================================================================
# Borehole heat map (BOREHOLE_HEATMAP)
# ===========================================================================
# fviz.plot_borehole_logs draws ONE model. The heat map needs every member
# along the borehole, so it is sampled and drawn here: an exact
# point-in-tetrahedron search once per borehole, then one array lookup per
# member, and a per-depth histogram. Sampling and histogram are the same as
# in femtic_ens_post.py (MOD_BH item "heatmap").
_HM_CACHE = {}   # mesh arrays, built once


def _hm_mesh():
    """Mesh arrays for the borehole search (read once, fem.read_femtic_mesh).

    nodes[:, 0] = easting, nodes[:, 1] = northing, nodes[:, 2] = z-down
    (model-local metres); per-element bounding boxes for the lateral filter.
    """
    if "verts" not in _HM_CACHE:
        nodes, conn = fem.read_femtic_mesh(MESH_FILE)
        verts = nodes[conn]                       # (n_elem, 4, 3)
        _HM_CACHE["n_elem"] = int(conn.shape[0])
        _HM_CACHE["verts"] = verts
        for _i, _ax in enumerate("xyz"):
            _HM_CACHE[_ax + "min"] = verts[:, :, _i].min(axis=1)
            _HM_CACHE[_ax + "max"] = verts[:, :, _i].max(axis=1)
    return _HM_CACHE


def _hm_column(x_m, y_m):
    """Indices of the elements whose (x, y) bounding box holds the point."""
    c = _hm_mesh()
    return np.where((c["xmin"] <= x_m) & (x_m <= c["xmax"])
                    & (c["ymin"] <= y_m) & (y_m <= c["ymax"]))[0]


def _hm_locate(x_m, y_m, depths_m, cand):
    """Tetrahedron index containing (x_m, y_m, z) for each depth, -1 outside.

    Exact barycentric test over the candidate elements ``cand`` of the column.
    """
    c = _hm_mesh()
    depths_m = np.asarray(depths_m, dtype=float)
    elem = np.full(depths_m.size, -1, dtype=np.int64)
    if cand.size == 0:
        return elem
    V = c["verts"][cand]                          # (nc, 4, 3)
    v0 = V[:, 0, :]
    T = np.transpose(V[:, 1:, :] - v0[:, None, :], (0, 2, 1))
    det = np.linalg.det(T)
    ok = np.isfinite(det) & (np.abs(det) > 0.0)
    cand, v0, T = cand[ok], v0[ok], T[ok]
    if cand.size == 0:
        return elem
    Tinv = np.linalg.inv(T)
    zmin, zmax = c["zmin"][cand], c["zmax"][cand]
    tol = 1.0e-10
    for i, z in enumerate(depths_m):
        sel = np.where((zmin <= z) & (z <= zmax))[0]
        if sel.size == 0:
            continue
        lam = np.einsum("nij,nj->ni", Tinv[sel],
                        np.array([x_m, y_m, z]) - v0[sel])
        inside = (lam.min(axis=1) >= -tol) & (lam.sum(axis=1) <= 1.0 + tol)
        hit = np.where(inside)[0]
        if hit.size:
            elem[i] = cand[sel[hit[0]]]
    return elem


def _hm_region_rho(st):
    """Resistivity (Ohm.m) per region of a block-file structure: air (region
    0) = NaN, ocean (region 1, when present) = OCEAN_RHO."""
    rr = np.asarray(st["region_rho"], dtype=float).copy()
    rr[0] = np.nan
    if int(st["nreg"]) > 1 and st["meta"]["ocean_present"]:
        rr[1] = OCEAN_RHO
    return rr


def _hm_levels(elem_region, region_rho, elem):
    """Resistivity at each depth level (NaN where elem < 0 or air)."""
    out = np.full(elem.size, np.nan)
    ok = elem >= 0
    out[ok] = region_rho[elem_region[elem[ok]]]
    return out


def _hm_pos_str(x_m, y_m):
    """Short position text for the panel title, following DISPLAY_COORDS."""
    try:
        if DISPLAY_COORDS == "latlon":
            lat, lon = utl.utm_to_latlon_zn(UTM_ORIGIN_E + x_m,
                                            UTM_ORIGIN_N + y_m,
                                            UTM_ZONE, UTM_NORTHERN)
            return f"{lat:.4f}, {lon:.4f} (lat, lon)"
        if DISPLAY_COORDS == "utm":
            return (f"E {(UTM_ORIGIN_E + x_m) / 1000.0:.2f}, "
                    f"N {(UTM_ORIGIN_N + y_m) / 1000.0:.2f} km")
    except Exception:
        pass
    return f"x {x_m / 1000.0:.2f}, y {y_m / 1000.0:.2f} km"


def _hm_sample(sites):
    """Sample the ensemble (``model_list``) along the BOREHOLE_SITES.

    Returns a list of dicts (name, x_m, y_m, depth in m z-down, ens
    (n_members, n_levels) in Ohm.m with NaN for air / outside the mesh, ref
    (n_levels) or None). Boreholes that are unusable are reported and left out.
    """
    mesh_n = _hm_mesh()["n_elem"]
    profs, used = [], set()
    for i, spec in enumerate(sites):
        name = str(spec.get("name", f"BH{i + 1}"))
        nm0, dup = name, 1
        while name in used:                       # names key the .npz arrays
            dup += 1
            name = f"{nm0}_{dup}"
        try:
            x_m, y_m = _resolve_borehole_xy(spec)
        except (KeyError, ValueError) as e:
            print(f"  heat map '{name}': {e} - skipped.")
            continue
        cand = _hm_column(x_m, y_m)
        zt_raw = spec.get("z_top", "surface")
        if isinstance(zt_raw, str) and zt_raw.strip().lower() == "surface":
            if cand.size == 0:
                print(f"  heat map '{name}': position is outside the mesh "
                      f"- skipped.")
                continue
            zt = float(_hm_mesh()["zmin"][cand].min())
        else:
            zt = float(zt_raw)
        zb = float(spec.get("z_bot", 20000.0))
        dz = float(spec.get("dz", 50.0))
        if zb <= zt or dz <= 0.0:
            print(f"  heat map '{name}': need z_bot > z_top and dz > 0 "
                  f"- skipped.")
            continue
        depth = np.arange(zt, zb + 0.5 * dz, dz)
        elem = _hm_locate(x_m, y_m, depth, cand)
        if not (elem >= 0).any():
            print(f"  heat map '{name}': profile is entirely outside the "
                  f"mesh - skipped.")
            continue
        used.add(name)
        profs.append(dict(name=name, x_m=x_m, y_m=y_m, depth=depth,
                          elem=elem, ref=None,
                          ens=np.full((len(model_list), depth.size), np.nan)))
        print(f"  {name}: x={x_m / 1000.0:.3f} km, y={y_m / 1000.0:.3f} km, "
              f"{depth.size} levels")
    if not profs:
        return profs

    def _check(er, what):
        if er.size != mesh_n:
            sys.exit(f"Heat map: {what} has {er.size} elements but MESH_FILE "
                     f"{mesh_n} - it must live on the same mesh.")

    if RESULTS_NPZ is not None:
        # NPZ mode: members are rows of the stored log10 matrix, the region
        # structure is the template's
        er = np.asarray(_tst["elem_region"], dtype=np.int64)
        _check(er, "the template block file")
        base = _hm_region_rho(_tst)
        free = np.asarray(_tst["free_idx"], dtype=np.int64)
        for k, m in enumerate(model_list):
            rr = base.copy()
            rr[free] = 10.0 ** np.asarray(m["vec"], dtype=float)
            for p in profs:
                p["ens"][k] = _hm_levels(er, rr, p["elem"])
    else:
        for k, m in enumerate(model_list):
            st = fem._read_resistivity_block_struct(
                m["ens_file"], ocean=OCEAN, out=False)
            er = np.asarray(st["elem_region"], dtype=np.int64)
            _check(er, m["ens_file"])
            rr = _hm_region_rho(st)
            for p in profs:
                p["ens"][k] = _hm_levels(er, rr, p["elem"])
            if OUT:
                print(f"    member {k + 1}/{len(model_list)} sampled "
                      f"({m['label']})")

    if "ref" in (BOREHOLE_HEAT_OVERLAY or []) and REF_FILE and REF_BOREHOLE:
        st = fem._read_resistivity_block_struct(REF_FILE, ocean=OCEAN,
                                                out=False)
        er = np.asarray(st["elem_region"], dtype=np.int64)
        if er.size != mesh_n:
            print(f"  heat map: REF_FILE has {er.size} elements but MESH_FILE "
                  f"{mesh_n} - reference line skipped.")
        else:
            rr = _hm_region_rho(st)
            for p in profs:
                p["ref"] = _hm_levels(er, rr, p["elem"])
    return profs


def _hm_edges(ens_rho):
    """log10(rho) bin edges: BOREHOLE_HEAT_BINS as a list of edges, or N equal
    bins across BOREHOLE_XLIM (the range of the finite values if None)."""
    if np.ndim(BOREHOLE_HEAT_BINS) > 0:
        edges = np.asarray(BOREHOLE_HEAT_BINS, dtype=float)
        if edges.size < 2 or not np.all(np.diff(edges) > 0):
            raise ValueError("BOREHOLE_HEAT_BINS must be an int or an "
                             "increasing list of >= 2 log10(rho) edges.")
        return edges
    nb = int(BOREHOLE_HEAT_BINS)
    if nb < 1:
        raise ValueError("BOREHOLE_HEAT_BINS must be >= 1.")
    if BOREHOLE_XLIM is not None:
        lo, hi = float(BOREHOLE_XLIM[0]), float(BOREHOLE_XLIM[1])
    else:
        ok = np.isfinite(ens_rho) & (ens_rho > 0)
        lg = np.log10(ens_rho[ok])
        lo, hi = float(lg.min()), float(lg.max())
        if hi - lo < 1.0e-6:
            lo, hi = lo - 0.5, hi + 0.5
    return np.linspace(lo, hi, nb + 1)


def _hm_hist(ens_rho, edges):
    """(n_levels, n_bins) fraction of the valid members per bin at each depth
    level. Members outside the edges are not binned but still count in the
    denominator (a row then sums to less than 1; the last bin includes its
    upper edge); levels without a valid member are NaN."""
    n_mem, n_lvl = ens_rho.shape
    nb = len(edges) - 1
    with np.errstate(invalid="ignore", divide="ignore"):
        lg = np.log10(ens_rho)
    valid = np.isfinite(lg)
    inside = valid & (lg >= edges[0]) & (lg <= edges[-1])
    idx = np.clip(np.searchsorted(edges, lg, side="right") - 1, 0, nb - 1)
    flat = idx + nb * np.arange(n_lvl)[None, :]
    counts = np.bincount(flat[inside], minlength=n_lvl * nb).reshape(n_lvl, nb)
    n_val = valid.sum(axis=0).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = counts / n_val[:, None]
    frac[n_val == 0, :] = np.nan
    return frac


def _hm_display(frac):
    """Heat map values as drawn (BOREHOLE_HEAT_NORM) from the fractions."""
    if BOREHOLE_HEAT_NORM == "row_max":
        import warnings
        with warnings.catch_warnings(), np.errstate(invalid="ignore",
                                                    divide="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN levels
            return frac / np.nanmax(frac, axis=1, keepdims=True)
    return frac


def _hm_log_ticks(norm):
    """1-2-5 colour-bar ticks for a log scale of less than 3 decades (the
    default shows only the powers of ten), else None."""
    if np.log10(norm.vmax / norm.vmin) >= 3.0:
        return None
    cand = np.array([m * 10.0 ** e for e in range(-8, 2) for m in (1.0, 2.0, 5.0)])
    tk = cand[(cand >= norm.vmin * (1 - 1e-9)) & (cand <= norm.vmax * (1 + 1e-9))]
    return tk if tk.size >= 2 else None


def _hm_scale(heats):
    """Colour scale of the heat map: (norm, colour-bar extend).

    ``heats`` : the displayed arrays (_hm_display) of the panels.  The norm is
    BOREHOLE_HEAT_SCALE between BOREHOLE_HEAT_VMIN and BOREHOLE_HEAT_VMAX (or
    their automatic values); ``extend`` is "neither" / "min" / "max" / "both"
    for data beyond the scale.
    """
    from matplotlib import colors
    log = BOREHOLE_HEAT_SCALE == "log"
    fin = np.concatenate([h[np.isfinite(h)] for h in heats])
    pos = fin[fin > 0]
    d_max = float(pos.max()) if pos.size else 1.0
    d_min = float(pos.min()) if pos.size else 1.0
    if BOREHOLE_HEAT_NORM == "row_max":
        vmax = 1.0
    elif BOREHOLE_HEAT_VMAX is not None:
        vmax = float(BOREHOLE_HEAT_VMAX)
    elif log or not pos.size:
        vmax = d_max
    else:
        vmax = float(np.percentile(pos, 99.0))
    if BOREHOLE_HEAT_VMIN is not None:
        vmin = float(BOREHOLE_HEAT_VMIN)
    else:
        vmin = d_min if log else 0.0
    if not vmax > vmin:      # one value only, or an unlucky explicit pair
        vmax = vmin * 10.0 if log else vmin + 1.0
    # zeros are blank on a log scale, so they do not count there
    lo = d_min if log else (float(fin.min()) if fin.size else 0.0)
    ext_min = lo < vmin * (1.0 - 1.0e-9)
    ext_max = d_max > vmax * (1.0 + 1.0e-9)
    extend = {(0, 0): "neither", (1, 0): "min", (0, 1): "max",
              (1, 1): "both"}[(int(ext_min), int(ext_max))]
    if log:
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    elif BOREHOLE_HEAT_SCALE == "sqrt":
        norm = colors.PowerNorm(0.5, vmin=vmin, vmax=vmax)
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    return norm, extend


def _hm_depth_edges(depth):
    """Cell edges (n + 1) around the depth levels."""
    d = np.asarray(depth, dtype=float)
    if d.size == 1:
        return np.array([d[0] - 0.5, d[0] + 0.5])
    first = d[0] - 0.5 * (d[1] - d[0])
    last = d[-1] + 0.5 * (d[-1] - d[-2])
    return np.concatenate(([first], 0.5 * (d[1:] + d[:-1]), [last]))


def _hm_overlay(ens_rho):
    """Overlay lines in Ohm.m: {"avg": a, "med": a, "sigma1": (lo, hi), ...}."""
    import warnings
    with np.errstate(invalid="ignore", divide="ignore"):
        lg = np.log10(ens_rho)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN levels
        out["avg"] = 10.0 ** np.nanmean(lg, axis=0)
        out["med"] = 10.0 ** np.nanmedian(lg, axis=0)
        out["sigma1"] = tuple(10.0 ** np.nanpercentile(lg, q, axis=0)
                              for q in (15.8655, 84.1345))
        out["sigma2"] = tuple(10.0 ** np.nanpercentile(lg, q, axis=0)
                              for q in (2.2750, 97.7250))
    return out


def _plot_borehole_heatmaps(profs, out_file):
    """One figure, a heat-map panel per borehole, one shared colour bar.

    ``profs`` from _hm_sample(). Writes ``out_file`` (None -> plt.show()) and
    returns the dict saved to the .npz (heat_<name>, heat_edges_<name>,
    depth_<name>).
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n = len(profs)
    ncols = max(1, min(int(BOREHOLE_HEAT_NCOLS), n))
    nrows = int(math.ceil(n / ncols))
    w, h = BOREHOLE_HEAT_PANEL_SIZE
    fig, axs = plt.subplots(nrows, ncols, squeeze=False,
                            figsize=(w * ncols / 2.54, h * nrows / 2.54))
    axs = axs.ravel()
    scale = 1000.0 if DEPTH_KM else 1.0
    tick_fs, label_fs = PLOT_TICK_FONTSIZE, PLOT_LABEL_FONTSIZE
    title_fs = (PLOT_TITLE_FONTSIZE if PLOT_TITLE_FONTSIZE is not None
                else label_fs + 6)

    saved = {}
    for p in profs:
        p["edges"] = _hm_edges(p["ens"])
        p["frac"] = _hm_hist(p["ens"], p["edges"])
        saved[f"depth_{p['name'].replace(' ', '_')}"] = p["depth"]
        saved[f"heat_{p['name'].replace(' ', '_')}"] = p["frac"]
        saved[f"heat_edges_{p['name'].replace(' ', '_')}"] = p["edges"]
    norm, extend = _hm_scale([_hm_display(p["frac"]) for p in profs])

    mesh = None
    for k, (ax, p) in enumerate(zip(axs, profs)):
        depth = p["depth"] / scale
        hval = _hm_display(p["frac"])
        # NaN (air) is always blank; on a log scale so are the empty bins
        hval = np.ma.masked_where(
            ~(hval > 0) if BOREHOLE_HEAT_SCALE == "log" else ~np.isfinite(hval),
            hval)
        mesh = ax.pcolormesh(
            10.0 ** p["edges"], _hm_depth_edges(depth), hval,
            cmap=BOREHOLE_HEAT_CMAP, norm=norm, shading="flat",
            rasterized=True, zorder=0)
        ov = _hm_overlay(p["ens"])
        if p["ref"] is not None:
            ov["ref"] = p["ref"]
        handles = []
        names = {"avg": "mean", "med": "median", "sigma1": "1-sigma quantiles",
                 "sigma2": "2-sigma quantiles", "ref": REF_LABEL or "reference"}
        for key in (BOREHOLE_HEAT_OVERLAY or []):
            if key not in ov:
                continue
            st = dict(BOREHOLE_HEAT_STYLE.get(key, {}))
            lines = ov[key] if isinstance(ov[key], tuple) else (ov[key],)
            for j, ln in enumerate(lines):
                ax.plot(ln, depth, zorder=3,
                        label=names[key] if j == 0 else "_nolegend_", **st)
            handles.append(Line2D([], [], label=names[key], **st))
        ax.set_xscale("log")
        ax.set_xlim(10.0 ** p["edges"][0], 10.0 ** p["edges"][-1])
        ax.set_title(f"{p['name']}\n{_hm_pos_str(p['x_m'], p['y_m'])}",
                     fontsize=tick_fs - 2 if tick_fs > 2 else tick_fs)
        ax.set_xlabel("resistivity (Ohm.m)", fontsize=label_fs)
        if k % ncols == 0:
            ax.set_ylabel("depth (km)" if DEPTH_KM else "depth (m)",
                          fontsize=label_fs)
        ax.tick_params(labelsize=tick_fs)
        ax.grid(True, which="both", axis="x", lw=0.3, alpha=0.5)
        ax.grid(True, which="major", axis="y", lw=0.3, alpha=0.5)
        if handles:
            ax.legend(handles=handles, fontsize=max(tick_fs - 1, 5),
                      loc="lower right", framealpha=0.85)
    for ax in axs[n:]:
        ax.set_visible(False)

    # one common depth range (union), y axis pointing down
    lo = min(float(np.nanmin(p["depth"])) for p in profs) / scale
    hi = max(float(np.nanmax(p["depth"])) for p in profs) / scale
    for ax in axs[:n]:
        ax.set_ylim(hi, lo)

    fig.suptitle(f"Borehole heat maps: {os.path.basename(os.path.normpath(ENSEMBLE_DIR))} "
                 f"({len(model_list)} members"
                 + (", perturbed reference" if _ENS_SRC == "perturbed" else "")
                 + ")", fontsize=title_fs)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    cb = fig.colorbar(mesh, ax=list(axs[:n]), location="right",
                      fraction=0.03, pad=0.02, shrink=0.6,
                      extend=extend)
    cb.set_label("fraction of ensemble members per bin"
                 if BOREHOLE_HEAT_NORM == "fraction"
                 else "fraction of the per-depth maximum", fontsize=label_fs)
    cb.ax.tick_params(labelsize=tick_fs)
    if BOREHOLE_HEAT_SCALE == "log" and _hm_log_ticks(norm) is not None:
        from matplotlib import ticker
        cb.set_ticks(_hm_log_ticks(norm))
        cb.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
        cb.ax.minorticks_off()

    if out_file:
        os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
        fig.savefig(out_file, dpi=PLOT_DPI, bbox_inches="tight")
        print(f"  saved -> {out_file}")
    else:
        plt.show()
    plt.close(fig)
    return saved


# ===========================================================================
# NPZ mode helpers (RESULTS_NPZ)
# ===========================================================================
#: Sentinel resistivities of block files written from a stored vector (as in
#: femtic_ens_post.py): air / ocean are then recognised by an exact value, so
#: no value of a plotted statistic (a std, a sensitivity, ...) can be taken
#: for air (log10 rho > 8) or ocean (rho ~ OCEAN_RHO).
_STATS_AIR_RHO = 1.0e30
_STATS_OCEAN_RHO = 1.0e-30
_STATS_AIR_LOG10_THRESH = 29.0


def _write_vec_block(vec, template, out_file, *, real_model=False):
    """Write a free-region log10(rho) vector into a block file.

    The template supplies header, region structure and flag columns.
    ``real_model=False`` (figures): sentinel air / ocean values, plotted with
    ``_plot_member_slice(stats_panel=True)``. ``real_model=True`` (borehole
    figure): air at AIR_RHO and ocean at OCEAN_RHO, as in a member's own
    block file. Non-finite entries become 0 Ohm.m (not drawn).
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    fem.insert_model(
        template=template,
        model=np.where(np.isfinite(vec), vec, -np.inf),
        model_file=out_file,
        ocean=OCEAN,
        air_rho=AIR_RHO if real_model else _STATS_AIR_RHO,
        ocean_rho=OCEAN_RHO if real_model else _STATS_OCEAN_RHO,
        out=False,
    )


def _load_results_npz(path, prefix=None):
    """Read what NPZ mode needs from a femtic_ens_post.py results file.

    Returns a dict: ``prefix`` (key prefix used), ``ens`` (n_members x n_free
    log10 matrix), ``members`` (list of dicts file / numit / nrms, one per
    row of ``ens``), ``stats`` (key -> (vector, description)) and
    ``value_keys`` (the statistics that are log10 resistivities) and
    ``source`` ("converged" | "perturbed": which model of each member the rows
    of ``ens`` are, as stored by femtic_ens_post.py's ENS_MODEL_SOURCE).
    Statistics missing from the file are skipped; avg / var / err / med / mad
    are recomputed from the members when absent.
    """
    try:
        z = np.load(path, allow_pickle=False)
    except (ValueError, OSError) as e:
        sys.exit(f"NPZ mode: cannot read {path}: {e}")
    keys = set(z.files)
    cand = sorted(
        k[: -len("_ens")] for k in keys
        if k.endswith("_ens") and f"{k[: -len('_ens')]}_avg" in keys
    )
    if prefix is not None:
        if f"{prefix}_ens" not in keys:
            sys.exit(f"NPZ mode: no '{prefix}_ens' in {path}. Ensemble "
                     f"prefixes found: {cand if cand else 'none'}.")
        pk = prefix
    elif len(cand) == 1:
        pk = cand[0]
    else:
        sys.exit(f"NPZ mode: {path} holds {len(cand)} ensembles {cand} -- "
                 f"set RESULTS_PREFIX.")

    ens = np.asarray(z[f"{pk}_ens"], dtype=float)
    if f"{pk}_model_list" not in keys:
        sys.exit(f"NPZ mode: '{pk}_model_list' missing from {path}.")
    # mixed-type lists are saved as a string array: [file, iter, nRMS]
    members = [
        dict(file=str(r[0]), numit=int(float(r[1])), nrms=float(r[2]))
        for r in np.asarray(z[f"{pk}_model_list"]).reshape(-1, 3)
    ]
    if len(members) != ens.shape[0]:
        sys.exit(f"NPZ mode: model_list has {len(members)} entries but the "
                 f"ensemble matrix {ens.shape[0]} rows - inconsistent file.")

    def get(name, recompute):
        k = f"{pk}_{name}"
        if k in keys:
            return np.asarray(z[k], dtype=float)
        print(f"  NPZ mode: '{k}' not in the file - recomputed from members.")
        return recompute()

    var = get("var", lambda: np.var(ens, axis=0))
    med = get("med", lambda: np.median(ens, axis=0))
    stats = {
        "avg": (get("avg", lambda: np.mean(ens, axis=0)), "mean"),
        "var": (var, "variance"),
        "err": (get("err", lambda: np.sqrt(var)), "error (std = sqrt(var))"),
        "med": (med, "median"),
        "mad": (get("mad", lambda: np.median(np.abs(ens - med), axis=0)),
                "MAD"),
    }
    value_keys = {"avg", "med"}
    if f"{pk}_prc" in keys and f"{pk}_prc_levels" in keys:
        prc = np.asarray(z[f"{pk}_prc"], dtype=float)
        for i, lv in enumerate(np.asarray(z[f"{pk}_prc_levels"]).ravel()):
            key = "p" + f"{float(lv):g}".replace(".", "_")
            stats[key] = (prc[i], f"{float(lv):g}th percentile")
            value_keys.add(key)
    for k in sorted(keys):
        if k.startswith(f"{pk}_qdiff_"):
            key = k[len(pk) + 1:]
            stats[key] = (np.asarray(z[k], dtype=float), f"|{key}| spread")
        elif k.startswith(f"{pk}_sens_"):
            key = k[len(pk) + 1:]
            kind = key[len("sens_"):]
            stats[key] = (np.asarray(z[k], dtype=float),
                          f"sensitivity {kind} "
                          f"({'linear' if kind.startswith('cv') else 'log10'})")
    if f"{pk}_var_boot" in keys:
        vb = np.asarray(z[f"{pk}_var_boot"], dtype=float)
        stats["var_boot"] = (vb, "bootstrap variance")
        stats["err_boot"] = (
            np.asarray(z[f"{pk}_err_boot"], dtype=float)
            if f"{pk}_err_boot" in keys else np.sqrt(vb),
            "bootstrap error (std)")
    for key, desc in (("simrc_coef", "log10 SimRC cumulative |regression coeff.|"),
                      ("simrc_corr", "log10 SimRC cumulative |correlation|")):
        if f"{pk}_{key}" in keys:
            stats[key] = (np.asarray(z[f"{pk}_{key}"], dtype=float), desc)
    source = (str(z[f"{pk}_model_source"]) if f"{pk}_model_source" in keys
              else "converged")   # files from before ENS_MODEL_SOURCE existed
    z.close()
    return dict(prefix=pk, ens=ens, members=members, stats=stats,
                value_keys=value_keys, source=source)


def _stat_style(key, value_keys):
    """(cmap, clim, label) of a statistic figure: RESULTS_STATS_STYLE over
    the defaults (value statistics at PLOT_CMAP_BEST / PLOT_CLIM_BEST)."""
    st = (RESULTS_STATS_STYLE or {}).get(key, {})
    if key in value_keys:
        d_clim, d_label = PLOT_CLIM_BEST, "log10(rho / Ohm*m)"
    else:
        d_clim, d_label = None, key
    return (st.get("cmap", PLOT_CMAP_BEST), st.get("clim", d_clim),
            st.get("label", d_label))


def _select_members(spec, labels):
    """Indices selected by RESULTS_MEMBERS: "none" | "best" is handled by
    the caller; "all" or a list of 0-based indices and / or labels."""
    n = len(labels)
    if isinstance(spec, str) and spec == "all":
        return list(range(n))
    out = []
    for item in (spec if isinstance(spec, (list, tuple)) else [spec]):
        if isinstance(item, (int, np.integer)) and 0 <= int(item) < n:
            out.append(int(item))
        elif isinstance(item, str) and item in labels:
            out.append(labels.index(item))
        else:
            print(f"  RESULTS_MEMBERS: {item!r} is not a member index "
                  f"(0..{n - 1}) or label - ignored.")
    return sorted(set(out))


def _plot_member_slice(
    block_file: str,
    out_stem: str,
    *,
    slices_resolved: list,
    site_xys: list,
    obs_coords_only: bool,
    figure_title: str,
    cmap: str,
    clim: list | None,
    nrms_annotation: dict | None = None,
    pdf_catalog: "PdfPages | None" = None,
    with_reference: bool = False,
    stats_panel: bool = False,
    cbar_label: str | None = None,
    ref_clim: list | None = None,
) -> None:
    """Call fviz.plot_model_slices once per _PLOT_FORMATS entry.

    Mirrors femtic_ens_post.py's ``_plot_slice`` helper: same PLOT_*
    options (CRS handling, site overlay, figure layout) used throughout
    this script, applied to one resistivity block file at a time.

    Parameters
    ----------
    out_stem : str
        Output path *without* an extension. ".<fmt>" is appended for
        each entry in PLOT_FORMAT / _PLOT_FORMATS (e.g. base "foo" +
        ["pdf", "jpg"] -> "foo.pdf" and "foo.jpg", same figure, one
        savefig() call per format — plot_model_slices doesn't expose a
        way to re-save an already-built figure under a second
        extension, so it is rebuilt fresh for each format.
    cmap : str
        Matplotlib colormap name for this call — pass PLOT_CMAP_ITER0
        or PLOT_CMAP_BEST from the caller so the perturbed-prior and
        best-fit plots can use distinct colormaps.
    clim : list or None
        Colour limits [log10(ρ_min), log10(ρ_max)] for this call —
        pass PLOT_CLIM_ITER0 or PLOT_CLIM_BEST from the caller so the
        perturbed-prior and best-fit plots can use distinct scales.
    pdf_catalog : PdfPages, optional
        If given and "pdf" is among _PLOT_FORMATS, the pdf-format
        Figure is written to this already-open PdfPages immediately
        after it's built, then the local reference is dropped so it
        can be garbage-collected before the next member is plotted —
        this keeps peak memory to one figure at a time instead of
        holding every member's figure until the whole ensemble is
        done. plot_model_slices already closes the figure
        (plt.close(fig)) before returning it; the object is still
        valid for a PdfPages.savefig() call right after.
    with_reference : bool
        True = draw the reference model (REF_FILE; REF_LEVELS / REF_STYLE /
        REF_LABEL) as iso-resistivity lines on every slice panel. Ignored
        when REF_FILE is None.
    stats_panel : bool
        True for block files written from a stored vector (NPZ mode,
        _write_vec_block): air / ocean carry the sentinel resistivities
        _STATS_AIR_RHO / _STATS_OCEAN_RHO, so a value of the plotted
        quantity can never be mistaken for air or ocean. False (default) =
        a member's own block file, ocean at OCEAN_RHO, as always.
    cbar_label : str or None
        Colourbar label; None = fviz's default.
    ref_clim : list or None
        Colour range the reference iso-line levels are derived from (decades
        inside it); None = ``clim``. Statistic panels pass PLOT_CLIM_BEST
        here: the reference is a resistivity model, its lines stay at
        resistivity decades whatever the panel shows.

    Figure geometry and annotation come from PLOT_EQUAL_ASPECT /
    PLOT_VERT_EXAG (equal axes or a defined vertical exaggeration of the
    section panels), PLOT_FIGSIZE, PLOT_TICK_FONTSIZE, PLOT_LABEL_FONTSIZE,
    PLOT_TITLE_FONTSIZE, PLOT_TICK_DECIMALS and PLOT_AIR_COLOR.
    """
    if fviz is None:
        print("  plot_member_slice: femtic_viz not available — skipping.")
        return

    _ref_kw = {}
    if with_reference and REF_FILE is not None:
        _ref_kw = dict(
            reference_file             = REF_FILE,
            reference_levels           = _ref_levels(
                clim if ref_clim is None else ref_clim),
            reference_style            = dict(REF_STYLE or {}),
            reference_label            = REF_LABEL,
            reference_ocean_value      = OCEAN_RHO,
            reference_air_log10_thresh = 8.0,
        )

    for _fmt in _PLOT_FORMATS:
        _out_file = f"{out_stem}.{_fmt}"
        _kwargs = dict(
            model_file          = block_file,
            mesh_file           = MESH_FILE,
            slices              = slices_resolved,
            cmap                = cmap,
            clim                = clim,
            xlim                = PLOT_XLIM,
            ylim                = PLOT_YLIM,
            zlim                = PLOT_ZLIM,
            ocean_color         = PLOT_OCEAN_COLOR,
            ocean_value         = _STATS_OCEAN_RHO if stats_panel else OCEAN_RHO,
            air_color           = PLOT_AIR_COLOR,
            air_bgcolor         = PLOT_AIR_BGCOLOR,
            site_xys            = site_xys,
            obs_coords_only     = obs_coords_only,
            sites_in_maps       = PLOT_SITES_MAPS,
            sites_in_slices     = PLOT_SITES_SLICES,
            site_marker         = SITE_MARKER,
            site_marker_slices  = SITE_MARKER_SLICES,
            map_markers         = MAP_MARKERS,
            show_model_centre   = SHOW_MODEL_CENTRE,
            projection_dist     = PROJECTION_DIST,
            display_coords      = DISPLAY_COORDS,
            utm_origin_e        = UTM_ORIGIN_E,
            utm_origin_n        = UTM_ORIGIN_N,
            utm_zone            = UTM_ZONE,
            utm_northern        = UTM_NORTHERN,
            utm_to_latlon_fn    = utl.utm_to_latlon_zn,
            latlon_to_model_fn  = fem.latlon_to_model,
            depth_km            = DEPTH_KM,
            horiz_km            = HORIZ_KM,
            equal_aspect        = PLOT_EQUAL_ASPECT,
            vert_exag           = _vert_exag(),
            panel_height        = PLOT_PANEL_HEIGHT / 2.54,
            panel_width         = PLOT_PANEL_WIDTH / 2.54 if PLOT_PANEL_WIDTH is not None else None,
            figsize             = ([v / 2.54 for v in PLOT_FIGSIZE]
                                   if PLOT_FIGSIZE is not None else None),
            nrows               = PLOT_NROWS,
            ncols               = PLOT_NCOLS,
            tick_fontsize       = PLOT_TICK_FONTSIZE,
            label_fontsize      = PLOT_LABEL_FONTSIZE,
            tick_decimals       = PLOT_TICK_DECIMALS,
            nrms_annotation     = nrms_annotation,
            figure_title        = figure_title,
            figure_title_fontsize = PLOT_TITLE_FONTSIZE,
            dpi                 = PLOT_DPI,
            out                 = OUT,
            **_ref_kw,
        )
        if stats_panel:
            _kwargs["air_log10_thresh"] = _STATS_AIR_LOG10_THRESH
        if cbar_label is not None:
            _kwargs["cbar_label"] = cbar_label

        # matplotlib's PDF backend only tries to write images as an
        # "Indexed" PDF colour space (backend_pdf.PdfFile._writeImg,
        # gated on the pdf.compression rcParam) when pdf.compression is
        # truthy. That indexed-colour path is what can trigger a
        # np.searchsorted IndexError (composited raster with an
        # anti-aliased pixel not in the pre-counted palette maps past
        # the end of the palette array — data-/render-dependent, not a
        # bug in plot_model_slices or in this script; same class of
        # issue as matplotlib/matplotlib#25806).
        #
        # 2026-08-23: previously pdf.compression was disabled
        # pre-emptively for every pdf save, which forced the always-
        # safe plain-RGB branch but made every per-member pdf several
        # times larger than it needed to be -- and, since the catalog
        # page for the same figure was written by a second, separate
        # PdfPages.savefig() call *outside* that rc_context, the two
        # copies of the same figure ended up with different
        # compression settings (catalog: default/compressed,
        # per-member file: forced-uncompressed), which is why the
        # per-member files were much larger than the catalog. Fixed:
        # compression now stays at its default (compressed) for both
        # the per-member file and the catalog page, and the
        # uncompressed / composite-disabled workaround is only applied
        # reactively, to whichever specific save actually hits the
        # IndexError -- so the fix's file-size cost is paid only by
        # the (rare) affected figure, not by the whole ensemble.
        try:
            _fig = fviz.plot_model_slices(plot_file=_out_file, **_kwargs)
        except IndexError as _err:
            if _fmt != "pdf":
                raise
            print(f"    ! pdf save hit matplotlib indexed-colour bug "
                  f"({_err}) -- retrying {_out_file} with "
                  f"pdf.compression disabled")
            try:
                with mpl.rc_context({"pdf.compression": 0}):
                    _fig = fviz.plot_model_slices(
                        plot_file=_out_file, **_kwargs)
            except IndexError as _err2:
                # Belt-and-braces fallback: render PNG (unaffected
                # code path) and convert that to PDF with Pillow.
                print(f"    ! retry also failed ({_err2}) -- falling "
                      f"back to PNG->PDF for {_out_file}")
                _png_fallback = f"{out_stem}__pdf_fallback.png"
                _fig = fviz.plot_model_slices(
                    plot_file=_png_fallback, **_kwargs)
                from PIL import Image
                Image.open(_png_fallback).convert("RGB").save(
                    _out_file, "PDF", resolution=PLOT_DPI)
                os.remove(_png_fallback)

        if OUT:
            print(f"    saved -> {_out_file}")

        if _fmt == "pdf" and pdf_catalog is not None:
            # This is a second, independent backend_pdf write of the
            # same Figure and can in principle hit the same
            # IndexError even when the primary save above succeeded
            # (e.g. the primary save succeeded on retry with
            # compression disabled, but this catalog append runs with
            # default settings again). Guard it the same way rather
            # than letting the whole run crash on a catalog-only
            # failure.
            try:
                pdf_catalog.savefig(_fig)
            except IndexError as _err:
                print(f"    ! catalog page for {_out_file} hit "
                      f"matplotlib indexed-colour bug ({_err}) -- "
                      f"retrying with pdf.compression disabled")
                try:
                    with mpl.rc_context({"pdf.compression": 0}):
                        pdf_catalog.savefig(_fig)
                except IndexError as _err2:
                    print(f"    ! catalog page for {_out_file} could "
                          f"not be written even with compression "
                          f"disabled ({_err2}) -- skipping this page "
                          f"in the catalog (per-member file above is "
                          f"unaffected)")

        # Drop the reference immediately so the figure (axes, colorbars,
        # gridded image data) can be garbage-collected before the next
        # format/member is built, instead of accumulating across the
        # whole ensemble. plt.close(fig) inside plot_model_slices already
        # detached it from pyplot's figure registry; this just releases
        # the last strong reference held here.
        del _fig


# ===========================================================================
# Main
# ===========================================================================

# --- (1) Estimate origin from site.dat (before zone derivation) -----------
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
            _zone_boot, _north_boot = utl.utm_zone_from_latlon(
                float(_lats.mean()), float(_lons.mean()), override=UTM_ZONE_OVERRIDE)
            UTM_ORIGIN_LAT, UTM_ORIGIN_LON = utl.utm_to_latlon_zn(
                UTM_ORIGIN_E, UTM_ORIGIN_N, _zone_boot, _north_boot)
            if OUT:
                print(f"Origin estimated ({ORIGIN_METHOD}, {len(_sdat)} sites):")
                print(f"  UTM_ORIGIN_E   = {UTM_ORIGIN_E:.1f} m")
                print(f"  UTM_ORIGIN_N   = {UTM_ORIGIN_N:.1f} m")
                print(f"  UTM_ORIGIN_LAT = {UTM_ORIGIN_LAT:.6f}°")
                print(f"  UTM_ORIGIN_LON = {UTM_ORIGIN_LON:.6f}°")
                print()

# --- (2) Derive UTM zone from finalised origin -----------------------------
UTM_ZONE, UTM_NORTHERN = utl.utm_zone_from_latlon(
    UTM_ORIGIN_LAT, UTM_ORIGIN_LON, override=UTM_ZONE_OVERRIDE)
hemi = "N" if UTM_NORTHERN else "S"
print(f"UTM zone: {UTM_ZONE}{hemi}  "
      f"(origin lat={UTM_ORIGIN_LAT:.4f}°, lon={UTM_ORIGIN_LON:.4f}°)")
print()

# --- (3) Resolve slice positions to model-local metres --------------------
slices_resolved = fem.resolve_slice_positions(
    PLOT_SLICES, UTM_ZONE, UTM_NORTHERN,
    UTM_ORIGIN_E, UTM_ORIGIN_N,
    UTM_ORIGIN_LAT, UTM_ORIGIN_LON,
    verbose=OUT,
)
if OUT:
    print()

# --- (4) Read site positions ----------------------------------------------
site_xys = []
_sites_from_obs = False
if SITE_DAT is not None and os.path.isfile(SITE_DAT):
    print(f"Reading site positions from site.dat: {SITE_DAT}")
    _rows = fem.read_site_dat(SITE_DAT, site_names=SITE_NAMES)
    for row in _rows:
        sx_m, sy_m = fem.utm_to_model(row["easting"], row["northing"],
                                      UTM_ORIGIN_E, UTM_ORIGIN_N)
        site_xys.append((row["name"], sx_m, sy_m, float(row.get("elev", 0.0))))
        if OUT:
            print(f"  {row['name']}: model-local x = {sx_m/1000:.3f} km,"
                  f"  y = {sy_m/1000:.3f} km")
    if not site_xys:
        print("  (no matching sites found in site.dat)")
    print()
elif SITE_NUMBER is not None:
    _site_nums = (SITE_NUMBER if isinstance(SITE_NUMBER, (list, tuple))
                  else [SITE_NUMBER])
    print(f"Reading site positions from observe.dat: {OBSERVE_FILE}")
    for _sn in _site_nums:
        sx_m, sy_m = fem.read_site_position(OBSERVE_FILE, _sn)
        site_xys.append((_sn, sx_m, sy_m, 0.0))
        if OUT:
            print(f"  site {_sn}: model-local x = {sx_m/1000:.3f} km,"
                  f"  y = {sy_m/1000:.3f} km")
    _sites_from_obs = True
    print()

# --- (5) Scan ensemble directories for converged members -------------------
# Mirrors femtic_ens_post.py Step (1) exactly: same NRMS_MAX threshold,
# same fem.read_cnv()-based column lookup, same
# resistivity_block_iter{numit}.dat naming for the best-fit model — so
# this script plots precisely the members ens_post included in its
# ensemble statistics.
#
# NPZ mode (RESULTS_NPZ set): nothing is scanned; model_list is filled from
# the results file right after the loop instead.
dir_list = ([] if RESULTS_NPZ is not None else utl.get_filelist(
    searchstr=[ENSEMBLE_NAME + "*"],
    searchpath=ENSEMBLE_DIR,
    fullpath=True,
))
if RESULTS_NPZ is None:
    print(f"Found {len(dir_list)} sub-directory/ies matching "
          f"'{ENSEMBLE_NAME}'.")

model_list = []   # list of dicts: label, dir, numit, nrms, iter0_file, best_file, ens_file
_pert_links = []  # ENS_MODEL_SOURCE="perturbed": members whose file is a symlink
_iter0_links = []  # members whose resistivity_block_iter0.dat is a symlink
for _d in dir_list:
    if not os.path.isdir(_d):
        print(f"\n  {_d}: not a directory — skipped (not an ensemble run).")
        continue

    print(f"\n  Inversion run: {_d}")
    _cnv_file = str(fem.resolve_cnv_path(_d, CNV_FILE))
    if not os.path.isfile(_cnv_file):
        print(f"    {os.path.basename(_cnv_file)} not found -- skipped.")
        continue

    # Column positions are read from this file's own header row via
    # fem.read_cnv() (case-insensitive substring match, e.g. "rms" ->
    # "RMS"), so this works regardless of FEMTIC version or whether
    # Beta/Distortion columns are present -- previously hardcoded
    # indices selected by a "4.3"/"5." version string silently matched
    # only one specific column layout and misread nRMS for any run whose
    # actual column count didn't match that assumption.
    try:
        _rows = fem.read_cnv(_cnv_file)["rows"]
    except ValueError as _e:
        print(f"    {_e}")
        continue
    if not _rows:
        print(f"    {os.path.basename(_cnv_file)} is empty -- skipped.")
        continue
    _last = _rows[-1]
    _numit = int(round(_last["Iter"]))
    _nrms  = float(_last["RMS"])

    if _nrms > NRMS_MAX:
        print(f"    nRMS={_nrms:.4f} > NRMS_MAX={NRMS_MAX} — skipped.")
        continue

    _best_file  = os.path.join(_d, f"resistivity_block_iter{_numit}.dat")
    _iter0_file = os.path.join(_d, "resistivity_block_iter0.dat")

    if not os.path.isfile(_best_file):
        print(f"    {_best_file} not found — skipped.")
        continue
    if not os.path.isfile(_iter0_file):
        print(f"    {_iter0_file} not found — skipped.")
        continue

    # Model of the ensemble-based borehole figures (ENS_MODEL_SOURCE).
    _ens_file = _best_file
    if ENS_MODEL_SOURCE == "perturbed":
        _ens_file = os.path.join(_d, ENS_PERTURBED_FILE)
        if not os.path.isfile(_ens_file):
            print(f"    {_ens_file} not found — skipped.")
            continue
        if os.path.islink(_ens_file):
            _pert_links.append(_d)
    if os.path.islink(_iter0_file):
        _iter0_links.append(_d)

    _idx   = len(model_list)
    _label = (ENS_LABELS[_idx] if ENS_LABELS and _idx < len(ENS_LABELS)
              else os.path.basename(os.path.normpath(_d)))

    print(f"    iter={_numit}  nRMS={_nrms:.4f}  {_best_file}")
    if ENS_MODEL_SOURCE == "perturbed":
        print(f"    borehole model (perturbed reference): {_ens_file}")
    model_list.append(dict(
        label=_label, dir=_d, numit=_numit, nrms=_nrms,
        iter0_file=_iter0_file, best_file=_best_file, ens_file=_ens_file,
    ))

_res = None   # NPZ mode: arrays read from RESULTS_NPZ
_bi = None    # NPZ mode: index of the lowest-nRMS member
if RESULTS_NPZ is not None:
    if not os.path.isfile(RESULTS_NPZ):
        sys.exit(f"NPZ mode: results file not found: {RESULTS_NPZ}")
    print(f"NPZ mode: reading {RESULTS_NPZ}")
    _res = _load_results_npz(RESULTS_NPZ, RESULTS_PREFIX)
    _used = set()
    for _idx, _mm in enumerate(_res["members"]):
        _d = os.path.dirname(_mm["file"])
        _label = (ENS_LABELS[_idx] if ENS_LABELS and _idx < len(ENS_LABELS)
                  else os.path.basename(os.path.normpath(_d)) or f"member_{_idx:03d}")
        if _label in _used:   # labels name directories / files: keep unique
            _label = f"{_label}_{_idx:03d}"
        _used.add(_label)
        model_list.append(dict(
            label=_label, dir=_d, numit=_mm["numit"], nrms=_mm["nrms"],
            iter0_file=None, best_file=_mm["file"], ens_file=_mm["file"],
            vec=_res["ens"][_idx],
        ))
    _bi = min(range(len(model_list)), key=lambda i: model_list[i]["nrms"])

    # Template block file (structure only): the best member's own file as
    # recorded in the .npz, or RESULTS_TEMPLATE when that is gone / elsewhere.
    _tpl = RESULTS_TEMPLATE if RESULTS_TEMPLATE else model_list[_bi]["best_file"]
    if not os.path.isfile(_tpl):
        sys.exit(
            f"NPZ mode: template block file not found: {_tpl}\n"
            f"  The stored best-member file is {model_list[_bi]['best_file']}.\n"
            f"  Set RESULTS_TEMPLATE to any resistivity block file of this "
            f"ensemble (same mesh and regions) -- the figures are drawn from "
            f"the stored vectors, only its structure is used.")
    _tst = fem._read_resistivity_block_struct(_tpl, ocean=OCEAN, out=False)
    _nfree = int(np.asarray(_tst["free_idx"]).size)
    if _nfree != _res["ens"].shape[1]:
        sys.exit(f"NPZ mode: template {_tpl} has {_nfree} free regions but "
                 f"the ensemble has {_res['ens'].shape[1]} - it is not a "
                 f"block file of this ensemble.")
    NPZ_TEMPLATE = _tpl

    # Model of the borehole figure: chosen (and validated) here, so a typo
    # stops the run before any figure is made.
    NPZ_BH = None
    if PLOT_BOREHOLE and BOREHOLE_SITES and BOREHOLE_LINES:
        _bsel = RESULTS_BOREHOLE_MODEL
        _bvalid = sorted(_res["value_keys"] & set(_res["stats"]))
        if isinstance(_bsel, str) and _bsel == "best":
            NPZ_BH = (model_list[_bi]["vec"],
                      f"best member {model_list[_bi]['label']}")
        elif (isinstance(_bsel, (int, np.integer))
              and 0 <= int(_bsel) < len(model_list)):
            NPZ_BH = (model_list[int(_bsel)]["vec"],
                      f"member {model_list[int(_bsel)]['label']}")
        elif isinstance(_bsel, str) and _bsel in _bvalid:
            NPZ_BH = (_res["stats"][_bsel][0], _res["stats"][_bsel][1])
        else:
            sys.exit(f"RESULTS_BOREHOLE_MODEL={_bsel!r}: use 'best', a member "
                     f"index 0..{len(model_list) - 1} or one of {_bvalid}.")
    NPZ_OUT_DIR = (RESULTS_OUT_DIR if RESULTS_OUT_DIR else os.path.join(
        os.path.dirname(os.path.abspath(RESULTS_NPZ)), "ens_plot"))
    NPZ_BLOCK_DIR = os.path.join(NPZ_OUT_DIR, "blocks")
    print(f"  {len(model_list)} members, {_nfree} free parameters, prefix "
          f"'{_res['prefix']}', best nRMS {model_list[_bi]['nrms']:.4f} "
          f"({model_list[_bi]['label']})")
    print(f"  the file holds the {_res['source']} models of the members")
    print(f"  statistics in the file: {sorted(_res['stats'])}")
    print(f"  output directory: {NPZ_OUT_DIR}")
    if PER_MEMBER_PLOT:
        print("  NPZ mode: the iter0 (prior) figures need the member "
              "directories - not made; RESULTS_MEMBERS / RESULTS_STATS "
              "decide what is drawn.")

n_members = len(model_list)
print(f"\nConverged members: {n_members}")

if n_members == 0:
    sys.exit("No converged members found. Nothing to do.")

ENS_FILES = [m["ens_file"] for m in model_list]   # kept for borehole plotting
#: What the rows / files of the ensemble are, for titles and headers.
_ENS_SRC = (_res["source"] if RESULTS_NPZ is not None else ENS_MODEL_SOURCE)
_ENS_SRC_TXT = " (perturbed reference)" if _ENS_SRC == "perturbed" else ""

if RESULTS_NPZ is None:
    if ENS_MODEL_SOURCE == "perturbed" and _pert_links:
        print(f"WARNING: {ENS_PERTURBED_FILE} is a symbolic link in "
              f"{len(_pert_links)} of {n_members} member(s) (e.g. "
              f"{os.path.basename(os.path.normpath(_pert_links[0]))}) - a link "
              f"points to the shared template, i.e. an UNPERTURBED model. "
              f"femtic_rto_prep.py writes the draw to referencemodel.dat; "
              f"resistivity_block_iter0.dat holds it only with "
              f"MOD_START_FROM_PERTURBED = True.")
    if PER_MEMBER_PLOT and _iter0_links:
        print(f"NOTE: resistivity_block_iter0.dat is a symbolic link in "
              f"{len(_iter0_links)} of {n_members} member(s): the "
              f"\"perturbed prior (iter0)\" figures then show the shared, "
              f"unperturbed template prior. The member's own perturbed model "
              f"is {ENS_PERTURBED_FILE}.")
else:
    if ENS_MODEL_SOURCE == "perturbed" and _ENS_SRC != "perturbed":
        print("NOTE: ENS_MODEL_SOURCE = 'perturbed' is ignored in NPZ mode; the "
              "file holds the converged models. Make the file with "
              "femtic_ens_post.py ENS_MODEL_SOURCE = 'perturbed'.")
ENS_LABELS_resolved = [m["label"] for m in model_list]

if OUT:
    print(f"\nEnsemble: {n_members} converged member(s)")
    for _m in model_list:
        _i0 = (os.path.basename(_m['iter0_file'])
               if _m['iter0_file'] else "n/a (NPZ mode)")
        print(f"  {_m['label']:30s}  iter0={_i0}"
              f"  best=iter{_m['numit']} (nRMS={_m['nrms']:.4f})")
    print()

# --- (6) Per-member plots: iter0 (perturbed prior) + best-fit model -------
# Each member gets its own sub-directory ENSEMBLE_DIR/<label>/ containing
# iter0.<ext> and best.<ext> (one pair of files per entry in _PLOT_FORMATS).
#
# Memory note: the pdf catalog (if enabled) is opened ONCE before this loop
# and every pdf figure is written to it and released immediately after it's
# built (see pdf_catalog= in _plot_member_slice), rather than accumulating
# all figures in a Python list and writing them at the end. For a 50-member
# ensemble that's the difference between ~2 live figures at a time and ~100
# — the earlier approach held every member's Matplotlib Figure (axes,
# colorbars, gridded slice arrays) in memory simultaneously until the very
# last member was plotted.
if PER_MEMBER_PLOT and RESULTS_NPZ is None:
    if fviz is None:
        sys.exit("femtic_viz not available — cannot plot.  Check your installation.")

    print(f"Plotting {n_members} converged member(s), "
          f"2 figures each (iter0 + best), formats={_PLOT_FORMATS} …")

    _use_catalog = PER_MEMBER_PDF_CATALOG_MODE != "none" and "pdf" in _PLOT_FORMATS
    _catalog_iter0 = _use_catalog and PER_MEMBER_PDF_CATALOG_MODE in ("iter0", "both")
    _catalog_best  = _use_catalog and PER_MEMBER_PDF_CATALOG_MODE in ("best", "both")
    _pdf_catalog = PdfPages(PER_MEMBER_CATALOG_FILE) if _use_catalog else None
    _n_catalog_pages = 0

    try:
        for _m in model_list:
            _label      = _m["label"]
            _member_dir = os.path.join(ENSEMBLE_DIR, _label)
            os.makedirs(_member_dir, exist_ok=True)

            print(f"\n  member {_label!r}: perturbed prior (iter0)")
            _plot_member_slice(
                block_file      = _m["iter0_file"],
                out_stem        = os.path.join(_member_dir, "iter0"),
                slices_resolved = slices_resolved,
                site_xys        = site_xys,
                obs_coords_only = _sites_from_obs,
                figure_title    = f"{_label} — perturbed prior (iter0)",
                cmap            = PLOT_CMAP_ITER0,
                clim            = PLOT_CLIM_ITER0,
                pdf_catalog     = _pdf_catalog if _catalog_iter0 else None,
                with_reference  = REF_SLICES == "slices",
            )
            if _catalog_iter0:
                _n_catalog_pages += 1

            print(f"  member {_label!r}: best fit (iter{_m['numit']}, "
                  f"nRMS={_m['nrms']:.4f})")
            _plot_member_slice(
                block_file      = _m["best_file"],
                out_stem        = os.path.join(_member_dir, "best"),
                slices_resolved = slices_resolved,
                site_xys        = site_xys,
                obs_coords_only = _sites_from_obs,
                figure_title    = f"{_label} — best fit (iter{_m['numit']})",
                cmap            = PLOT_CMAP_BEST,
                clim            = PLOT_CLIM_BEST,
                nrms_annotation = dict(nrms=_m["nrms"]),
                pdf_catalog     = _pdf_catalog if _catalog_best else None,
                with_reference  = REF_SLICES in ("slices", "best"),
            )
            if _catalog_best:
                _n_catalog_pages += 1

            # Matplotlib Figure/Axes hold reference cycles, so refcounting
            # alone won't free them right away; force a cyclic-gc pass
            # after each member so peak memory stays bounded by ~one
            # member's figures rather than drifting upward over the run.
            gc.collect()
    finally:
        if _pdf_catalog is not None:
            _pdf_info = _pdf_catalog.infodict()
            _pdf_info["Title"] = "femtic_ens_plot ensemble catalog"
            _pdf_info["Author"] = "femtic_ens_plot.py"
            _pdf_catalog.close()

    print("\nPer-member plots done.")
    if _use_catalog:
        print(f"  pdf catalog ({PER_MEMBER_PDF_CATALOG_MODE}): "
              f"{_n_catalog_pages} page(s) → {PER_MEMBER_CATALOG_FILE}")
    elif PER_MEMBER_PDF_CATALOG_MODE != "none":
        print("  pdf catalog: \"pdf\" not in PLOT_FORMAT — skipped.")

# --- (6n) NPZ mode: member and statistic figures from the stored vectors ---
# Every figure is a block file written from a vector of RESULTS_NPZ (template =
# the best member's structure) and drawn by the same _plot_member_slice() as
# in the normal mode, i.e. with the same slices, layout, PLOT_* styling,
# PLOT_VERT_EXAG and reference model.
if RESULTS_NPZ is not None:
    if fviz is None:
        sys.exit("femtic_viz not available — cannot plot.  Check your installation.")
    os.makedirs(NPZ_OUT_DIR, exist_ok=True)
    _labels = [m["label"] for m in model_list]

    # --- members: best-fit model of the selected rows ----------------------
    if isinstance(RESULTS_MEMBERS, str) and RESULTS_MEMBERS == "none":
        _sel = []
    elif isinstance(RESULTS_MEMBERS, str) and RESULTS_MEMBERS == "best":
        _sel = [_bi]
    else:
        _sel = _select_members(RESULTS_MEMBERS, _labels)
    if _sel:
        print(f"\nNPZ mode: {len(_sel)} member figure(s), "
              f"formats={_PLOT_FORMATS} ...")
    _use_catalog = (bool(_sel)
                    and PER_MEMBER_PDF_CATALOG_MODE in ("best", "both")
                    and "pdf" in _PLOT_FORMATS)
    _catalog_file = os.path.join(
        NPZ_OUT_DIR, os.path.basename(PER_MEMBER_CATALOG_FILE))
    _pdf_catalog = PdfPages(_catalog_file) if _use_catalog else None
    try:
        for _i in _sel:
            _m = model_list[_i]
            _label = _m["label"]
            _blk = os.path.join(NPZ_BLOCK_DIR, f"resistivity_block_{_label}_best.dat")
            _write_vec_block(_m["vec"], NPZ_TEMPLATE, _blk)
            _member_dir = os.path.join(NPZ_OUT_DIR, _label)
            os.makedirs(_member_dir, exist_ok=True)
            print(f"\n  member {_label!r}: "
                  + ("perturbed reference" if _ENS_SRC == "perturbed"
                     else f"best fit (iter{_m['numit']})")
                  + f", nRMS={_m['nrms']:.4f}")
            _plot_member_slice(
                block_file      = _blk,
                out_stem        = os.path.join(_member_dir, "best"),
                slices_resolved = slices_resolved,
                site_xys        = site_xys,
                obs_coords_only = _sites_from_obs,
                figure_title    = (f"{_label} — perturbed reference"
                                   if _ENS_SRC == "perturbed" else
                                   f"{_label} — best fit (iter{_m['numit']})"),
                cmap            = PLOT_CMAP_BEST,
                clim            = PLOT_CLIM_BEST,
                nrms_annotation = dict(nrms=_m["nrms"]),
                pdf_catalog     = _pdf_catalog,
                with_reference  = REF_SLICES in ("slices", "best"),
                stats_panel     = True,
            )
            gc.collect()
    finally:
        if _pdf_catalog is not None:
            _pdf_info = _pdf_catalog.infodict()
            _pdf_info["Title"] = "femtic_ens_plot ensemble catalog"
            _pdf_info["Author"] = "femtic_ens_plot.py"
            _pdf_catalog.close()
    if _use_catalog:
        print(f"  pdf catalog ({PER_MEMBER_PDF_CATALOG_MODE}): "
              f"{len(_sel)} page(s) -> {_catalog_file}")

    # --- statistics --------------------------------------------------------
    for _key in (RESULTS_STATS or []):
        if _key not in _res["stats"]:
            print(f"\n  RESULTS_STATS: '{_key}' is not in the results file "
                  f"- skipped.")
            continue
        _vec, _desc = _res["stats"][_key]
        _cmap, _clim, _cbl = _stat_style(_key, _res["value_keys"])
        _blk = os.path.join(
            NPZ_BLOCK_DIR, f"resistivity_block_{_res['prefix']}_{_key}.dat")
        _write_vec_block(_vec, NPZ_TEMPLATE, _blk)
        print(f"\n  statistic {_key!r} ({_desc}): cmap={_cmap}, clim={_clim}")
        _plot_member_slice(
            block_file      = _blk,
            out_stem        = os.path.join(NPZ_OUT_DIR, f"{_res['prefix']}_{_key}"),
            slices_resolved = slices_resolved,
            site_xys        = site_xys,
            obs_coords_only = _sites_from_obs,
            figure_title    = f"{_res['prefix']}: {_desc}",
            cmap            = _cmap,
            clim            = _clim,
            with_reference  = REF_SLICES == "slices",
            stats_panel     = True,
            cbar_label      = _cbl,
            ref_clim        = PLOT_CLIM_BEST,
        )
        gc.collect()
    print("\nNPZ mode: figures done.")

# --- (7) Borehole resistivity logs ------------------------------------------
if PLOT_BOREHOLE:
    if not BOREHOLE_SITES:
        print("  Borehole plot skipped: BOREHOLE_SITES is empty.")
    else:
        if BOREHOLE_LINES:
            # Normal mode: the first ensemble member is the model of the log.
            _bh_model_file = ENS_FILES[0]
            _bh_desc = f"member 0 ({ENS_LABELS_resolved[0]}){_ENS_SRC_TXT}"
            _bh_label = ENS_LABELS_resolved[0] + _ENS_SRC_TXT
            if RESULTS_NPZ is not None:
                # NPZ mode: RESULTS_BOREHOLE_MODEL (chosen above), written as a
                # block file with the real air / ocean values (a log10 model).
                _bvec, _bh_label = NPZ_BH
                _bh_label = _bh_label + _ENS_SRC_TXT
                _bh_model_file = os.path.join(
                    NPZ_BLOCK_DIR, "resistivity_block_borehole.dat")
                _write_vec_block(_bvec, NPZ_TEMPLATE, _bh_model_file,
                                 real_model=True)
                _bh_desc = _bh_label
            print(f"Sampling {len(BOREHOLE_SITES)} borehole(s) "
                  f"from {_bh_desc} ...")
            if fviz is None:
                sys.exit("femtic_viz not available - cannot plot boreholes.")
            fviz.plot_borehole_logs(
                model_file     = _bh_model_file,
                mesh_file      = MESH_FILE,
                borehole_sites = BOREHOLE_SITES,
                resolve_xy_fn  = _resolve_borehole_xy,
                utm_zone       = UTM_ZONE,
                utm_northern   = UTM_NORTHERN,
                utm_origin_e   = UTM_ORIGIN_E,
                utm_origin_n   = UTM_ORIGIN_N,
                ocean_value    = OCEAN_RHO,
                # BOREHOLE_XLIM is log10(Ohm.m); fviz takes Ohm.m
                clim           = ([10.0 ** float(v) for v in BOREHOLE_XLIM]
                                  if BOREHOLE_XLIM is not None else None),
                borehole_style = BOREHOLE_STYLE,
                shared         = BOREHOLE_SHARED,
                tick_fontsize  = PLOT_TICK_FONTSIZE,
                label_fontsize = PLOT_LABEL_FONTSIZE,
                figure_title   = f"Borehole logs: {_bh_label}",
                reference_file  = REF_FILE if REF_BOREHOLE else None,
                reference_label = REF_LABEL if REF_LABEL else None,
                npz_file       = (os.path.splitext(BOREHOLE_FILE)[0] + ".npz"
                                  if (BOREHOLE_NPZ and BOREHOLE_FILE is not None)
                                  else False),
                plot_file      = BOREHOLE_FILE,
                dpi            = PLOT_DPI,
                out            = OUT,
            )
            print("Borehole plot done.")

        # Heat map of the whole ensemble along the same boreholes (drawn here,
        # not by fviz; independent of BOREHOLE_LINES).
        if BOREHOLE_HEATMAP:
            print(f"Sampling {len(BOREHOLE_SITES)} borehole(s) from "
                  f"{len(model_list)} members for the heat map ...")
            _hm_profs = _hm_sample(BOREHOLE_SITES)
            if not _hm_profs:
                print("  Heat map skipped: no usable borehole.")
            else:
                _hm_saved = _plot_borehole_heatmaps(_hm_profs,
                                                    BOREHOLE_HEATMAP_FILE)
                if BOREHOLE_HEAT_NPZ and BOREHOLE_HEATMAP_FILE is not None:
                    import json
                    _hm_saved["n_members"] = np.array(len(model_list))
                    _hm_saved["header"] = np.array(json.dumps(dict(
                        heat_unit=("heat_<name>: (n_levels, n_bins) fraction "
                                   "of the valid members per bin (rows sum "
                                   "to 1); heat_edges_<name>: bin edges in "
                                   "log10(Ohm.m); depth_<name>: m, z "
                                   "positive-down, model datum"),
                        rows_sum_to=("1 when all members are inside the "
                                     "edges, else less"),
                        source=(RESULTS_NPZ if RESULTS_NPZ is not None
                                else ENSEMBLE_DIR),
                        ensemble_source=_ENS_SRC,
                        members=[m["ens_file"] for m in model_list],
                        boreholes=[dict(name=_p["name"], x_m=float(_p["x_m"]),
                                        y_m=float(_p["y_m"]))
                                   for _p in _hm_profs],
                    ), indent=2))
                    _hm_npz = os.path.splitext(BOREHOLE_HEATMAP_FILE)[0] + ".npz"
                    np.savez_compressed(_hm_npz, **_hm_saved)
                    print(f"  heat-map histograms saved -> {_hm_npz}")
                print("Borehole heat map done.")


# ---------------------------------------------------------------------------
# Parameter summary
# ---------------------------------------------------------------------------
utl.write_param_summary(fname, out_dir=ENSEMBLE_DIR)
