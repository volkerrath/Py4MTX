#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Run the randomize-then-optimize (RTO) algorithm:

    for i = 1 : nsamples do
        Draw perturbed data set: d_pert∼ N (d, Cd)
        Draw prior model: m̃ ∼ N (0, 1/mu (LT L)^−1 )
        Solve determistic problem  to get the model m_i
    end

See:

Bardsley, J. M.; Solonen, A.; Haario, H. & Laine, M.
    Randomize-Then-Optimize: a Method for Sampling from Posterior
    Distributions in Nonlinear Inverse Problems
    SIAM J. Sci. Comp., 2014, 36, A1895-A1910

Blatter, D.; Morzfeld, M.; Key, K. & Constable, S.
    Uncertainty quantification for regularized inversion of electromagnetic
    geophysical data. Part I: Motivation and Theory
    Geophysical Journal International, doi:10.1093/gji/ggac241, 2022

Blatter, D.; Morzfeld, M.; Key, K. & Constable, S.
    Uncertainty quantification for regularized inversion of electromagnetic
    geophysical data – Part II: application in 1-D and 2-D problems
    Geophysical Journal International, , doi:10.1093/gji/ggac242, 2022


Created on Wed Apr 30 16:33:13 2025

@author: vrath

Provenance:
    2025-04-30  vrath   Created.
    2026-03-03  Claude  Renamed user-set parameters to UPPERCASE.
    2026-03-24  Claude  Added visualization config blocks (data + model
                        ensemble plots); helper functions live in
                        femtic_viz.plot_data_ensemble and
                        femtic_viz.plot_model_ensemble.
    2026-03-28  Claude  Moved visualization blocks into their respective
                        perturbation sections; matplotlib imported at top
                        level; VIZ_SAMPLES moved to base setup.
    2026-03-29  Claude  Consolidated all visualization parameters into a
                        single Visualization config section; replaced fixed
                        VIZ_SAMPLES list with VIZ_N_SAMPLES (random draw);
                        added VIZ_N_SITES for random site sub-sampling in
                        plot_data_ensemble.
    2026-03-30  Claude  Fixed rho_a unit mismatch (Z in SI Ohm vs mV/km/nT);
                        changed DAT_SHOW_ERRORS to False (raw template errors
                        are noisy at long periods); switched to per-curve
                        show_errors_orig/show_errors_pert flags so perturbed
                        curves show reset relative-error envelopes only.
    2026-03-31  Claude  Pass R directly (not Q=R^T R); randomized SVD replaces
                eigsh in low-rank branch; new MOD_N_EIG / MOD_N_OVERSAMPLING /
                MOD_N_POWER_ITER / MOD_SIGMA2_RESIDUAL config params; full-rank
                params MOD_LAM / MOD_LAM_MODE / MOD_LAM_ALPHA / MOD_SOLVER /
                MOD_PRECOND exposed; MOD_R sentinel variable removed.
    2026-04-01  Claude  Moved MOD_REF_BASE inside PERTURB_MOD block; added
                MOD_REF = MOD_REF_BASE = None in else-branch to prevent
                NameError when PERTURB_MOD = False.
    2026-04-02  Claude  Added MOD_XLIM / MOD_YLIM / MOD_ZLIM to visualization
                config; wired into plot_model_ensemble call.
    2026-04-03  Claude  Added DAT_ALPHA_ORIG / DAT_ALPHA_PERT to data-plot
                config; added MOD_MESH_LINES / MOD_MESH_LW / MOD_MESH_COLOR
                to model-plot config; wired all into their respective
                plot_data_ensemble / plot_model_ensemble calls.
    2026-04-12  Claude  Added DAT_COMP_MARKERS / DAT_MARKERSIZE / DAT_MARKEVERY
                to data-plot config (different symbols for ii, ij and invariant
                components); added DAT_ERROR_STYLE_ORIG / DAT_ERROR_STYLE_PERT
                for independent control of error rendering (shade / bar / both)
                on original vs. perturbed curves (no shared fallback variable).
                DAT_WHAT changed to a list to support multi-panel layouts
                (e.g. ["rho", "phase", "tipper"]); DAT_COMPS changed to a
                parallel list (one entry per panel; single string still
                accepted and broadcast to all rho/phase panels).
    2026-04-12  Claude  Per-sample plot loop: both PLOT_DATA and PLOT_MODEL
                now iterate over VIZ_SAMPLES, calling plot_data_ensemble /
                plot_model_ensemble with sample_indices=[i] and saving
                rto_data<PLOT_STR>.pdf / rto_model<PLOT_STR>.pdf into each
                member's own subdirectory (not the shared plots/ dir).
                Removed shared PLOT_DIR save paths for these figures.
                Fixed PLOT_DIR path (os.makedirs replacing os.mkdir).
    2026-04-12  Claude  Added DAT_PERLIMS / DAT_RHOLIMS / DAT_PHSLIMS /
                DAT_VTFLIMS / DAT_PTLIMS to data-plot config; wired into
                plot_data_ensemble call (perlims / rholims / phslims /
                vtflims / ptlims kwargs).
    2026-04-12  Claude  Replaced DAT_SHOW_ERRORS with DAT_SHOW_ERRORS_ORIG /
                DAT_SHOW_ERRORS_PERT for independent per-curve error-envelope
                control; removed hardcoded show_errors_orig=True from the
                plot_data_ensemble call; fixed PLOT_DIR path (was erroneously
                prepended with ENSEMBLE_DIR).
    2026-04-27  Claude  Renamed ens.generate_model_ensemble call to
                ens.generate_rto_model_ensemble for consistency with the
                new ens.generate_gst_model_ensemble.
    2026-05-13  Claude  Added ensemble slice plot block (PLOT_SLICES_ENS):
                ENS_SLICES / ENS_CMAP / ENS_CLIM / ENS_STAT_ROWS config;
                calls fviz.plot_ensemble_slices for exact tet-plane
                intersection figure.  Requires PLOT_DATA or PLOT_MODEL True.
    2026-05-27  vrath / Claude Sonnet 4.6 (Anthropic)
                Added QC slice plot step after model ensemble generation:
                PLOT_SLICES_QC / QC_SLICES / QC_CMAP / QC_CLIM / QC_XLIM /
                QC_YLIM / QC_ZLIM / QC_OCEAN_COLOR / QC_DPI config vars;
                calls fviz.plot_model_slices per member, saves rto_qc*.pdf
                in each member's subdirectory.
    2026-05-31  vrath / Claude Sonnet 4.6   MOD_SLICES updated to use
                "kind" key ("map"/"ns"/"ew") instead of "type".
                depth_km=True, horiz_km=True added to plot_model_slices
                and plot_ensemble_slices QC/ENS calls.
    2026-05-28  Claude Sonnet 4.6 (Anthropic)
                Added RELATIVE_LINKS config variable (default True); passed as
                relative_links to ens.generate_directories.  Relative symlinks
                survive tgz/copy to another machine; set False for legacy
                absolute-path behaviour.
    2026-06-07  Claude Sonnet 4.6 (Anthropic)
                Replaced MOD_MODE/MOD_LOG10/MOD_MESH_LINES/MOD_MESH_LW/
                MOD_MESH_COLOR and QC_SLICES/QC_CMAP/QC_CLIM/QC_XLIM/YLIM/
                ZLIM/QC_OCEAN_COLOR/QC_DPI with full femtic_mod_plot_slice
                config block (MOD_UTM_ORIGIN_*, MOD_DISPLAY_COORDS,
                MOD_SITE_DAT, MOD_SITE_NAMES, MOD_SITE_NUMBER,
                MOD_PLOT_SITES_MAPS/SLICES, MOD_PROJECTION_DIST,
                MOD_SITE_MARKER/SLICES, MOD_MAP_MARKERS, MOD_ORIGIN_METHOD,
                MOD_DPI, MOD_OCEAN_COLOR, MOD_AIR_COLOR/BGCOLOR,
                MOD_ALPHA_FILE/MODE/BLANK_THRESH, MOD_EQUAL_ASPECT,
                MOD_DEPTH_KM, MOD_HORIZ_KM, MOD_NROWS/NCOLS,
                MOD_PANEL_HEIGHT/WIDTH/FIGSIZE).
                PLOT_MODEL and PLOT_SLICES_QC execution blocks replaced by a
                shared _plot_member_slices() helper calling
                fviz.plot_model_slices with UTM-origin resolution, CRS-aware
                slice positions (fem.resolve_slice_positions), and site
                overlay — matching the femtic_mod_plot_slice workflow exactly.
                ENS_* variables now default-assigned from MOD_* counterparts.
"""

import os
import sys
from pathlib import Path
import numpy as np
import inspect
import matplotlib.pyplot as plt

"""
Specialized toolboxes settings and imports.
"""
import scipy.sparse as scs

"""
Py4MTX-specific settings and imports.
"""
PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

for _base in [PY4MTX_ROOT + "/py4mt/modules/"]:
    for _p in [Path(_base), *Path(_base).rglob("*")]:
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

# import modules
from version import versionstrg
import util as utl
import femtic as fem
import ensembles as ens
import femtic_viz as fviz

from util import stop

N_THREADS = "32"
os.environ["OMP_NUM_THREADS"] = N_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = N_THREADS
os.environ["MKL_NUM_THREADS"] = N_THREADS

rng = np.random.default_rng()
nan = np.nan  # float("NaN")
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())

titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng + "\n\n")

OUT = True

"""
Base setup.
"""
N_SAMPLES = 32
# ENSEMBLE_DIR = r"/home/vrath/Py4MTX/py4mt/data/rto/ubinas/ensemble/"
# ENSEMBLE_NAME = "ubinas_rto_"

ENSEMBLE_DIR = r"/home/vrath/Py4MTX/py4mt/data/rto/misti/ensemble/"
ENSEMBLE_NAME = "misti_rto_"

# TEMPLATES = ENSEMBLE_DIR + "templates/"
TEMPLATES = ENSEMBLE_DIR + "/templates/"
if not os.path.isdir(TEMPLATES):
     sys.exit(" Directory: %s does not exist, needs to be copied !" % TEMPLATES)
     
COPY_LIST = ["observe.dat",
             "referencemodel.dat",]
LINK_LIST = ["control.dat",
             "mesh.dat",
             "resistivity_block_iter0.dat",
             "distortion_iter0.dat", "site.dat",
             "run_femtic_dias.sh","run_femtic_kraken.sh"]
RELATIVE_LINKS = True   # True: portable relative symlinks (default, survives tgz);
                        # False: absolute symlinks (legacy behaviour)


"""
Control number of ensemble members for increase of sample number or restart
of badly converged samples (see femtic_rto_post.py)
"""
FROM_TO = None


"""
Set up mode of model perturbations.
"""
PERTURB_MOD = True
if PERTURB_MOD:
    MOD_REF = TEMPLATES + "referencemodel.dat"
    # filename only — used inside ensemble dirs
    MOD_REF_BASE = os.path.basename(MOD_REF)
    MOD_METHOD = "add"
    # if ModCov is not None, this needs to be normal
    MOD_MU = 1.5
    MOD_PDF = ["normal", 0., 1.5]
    # ["exp", L], ["gauss", L], ["matern", L, MatPars], ["femtic"], None
    R_FILE = TEMPLATES + r"/R_coo"

    # Sampling algorithm: "low rank" (randomized SVD, recommended) or "full rank" (CG).
    # Low-rank is 10-100x faster for typical FEMTIC mesh sizes.
    MOD_ALGO = "low rank"

    # --- low-rank options (used when MOD_ALGO = "low rank") ---
    # More n_eig -> smoother samples; cost is linear in n_eig.
    # n_power_iter=3-4 sharpens accuracy for slowly decaying roughness spectra.
    # sigma2_residual adds short-wavelength variability beyond the rank-k subspace;
    # set to ~10% of typical log10(rho) variance (e.g. 1e-3 for std ~0.1 in log10).
    MOD_N_EIG = 128 
    MOD_N_OVERSAMPLING = 10     # extra columns in range-finder; 10-15 is fine
    MOD_N_POWER_ITER = 3        # 3-4 recommended for FEMTIC roughness spectra
    MOD_SIGMA2_RESIDUAL = 1e-3  # isotropic residual variance; 0 = disabled

    # --- full-rank options (used when MOD_ALGO = "full rank") ---
    # lam_alpha is the key speed lever: raise to 1e-4 or 1e-3 if CG is slow.
    MOD_LAM = 0.0
    # auto diagonal shift from diag(R^T R)
    MOD_LAM_MODE = "scaled_median_diag"
    MOD_LAM_ALPHA = 1.0e-4                # raise to 1e-3 if CG convergence is slow
    MOD_SOLVER = "cg"                     # CG is optimal for SPD Q = R^T R
    MOD_PRECOND = "ilu"                   # ILU beats jacobi by 3-5x fewer iterations

else:
    R_FILE = None
    MOD_REF = None
    MOD_REF_BASE = None

"""
Set up mode of data perturbations.
"""
PERTURB_DAT = True
if PERTURB_DAT:
    DAT_METHOD = "add"
    DAT_PDF = ["normal", 0., 1.]

RESET_ERRORS = True
if RESET_ERRORS:
    ERRORS = [
        [0.25, .1, .1, 0.25] * 2,         # Impedance
        [0.05, 0.05] * 2,                   # VTF
        [.5, .2, .2, .5],                 # PT
    ]
else:
    ERRORS = []


"""
Visualization config.
---------------------
All diagnostic-plot settings in one place.  Both plot_data_ensemble and
plot_model_ensemble use the same randomly drawn set of ensemble members
(VIZ_N_SAMPLES).  plot_data_ensemble additionally sub-samples a fixed number
of MT sites per row (VIZ_N_SITES); set to None to show all sites.
"""
PLOT_DATA = True #True
PLOT_MODEL = True #True

if PLOT_DATA or PLOT_MODEL:
    PLOT_STR = ""
    PLOT_DIR = ENSEMBLE_DIR + "/plots/"
    if PLOT_DIR is not None:
        print(" Plots written to: %s" % PLOT_DIR)
        if not os.path.isdir(PLOT_DIR):
            print(" Directory: %s does not exist, but will be created" % PLOT_DIR)
            os.makedirs(PLOT_DIR, exist_ok=True)
    
    
    # Number of ensemble members to include in both diagnostic plots.
    # Members are drawn without replacement from 0 … N_SAMPLES-1.
    VIZ_N_SAMPLES = N_SAMPLES
    
    # Number of MT sites to include in each data-plot row.
    # Sites are drawn without replacement from the full site list.
    # Set to None to show all available sites.
    VIZ_N_SITES = 10
    
    # --- data plot ---
    # what:  list of panel types — one subplot column per entry.
    #        Allowed values: 'rho', 'phase', 'tipper', 'pt'.
    # comps: list of component strings, one per entry in what.
    #        Ignored for 'tipper' and 'pt' panels (use None or '' there).
    #        A single string is also accepted and is broadcast to all panels.
    DAT_WHAT = ["rho", "phase"]            # e.g. ["rho", "phase", "tipper"]
    DAT_COMPS = ["xx,xy,yx,yy",            # rho   — all four components
                 "xx,xy,yx,yy"]            # phase — all four components
    DAT_COMPS = ["xy,yx",            # rho   — all four components
                 "xy,yx"]
    
    # Shorter alternative (single string broadcast to every rho/phase column):
    # DAT_COMPS = "xy,yx"
    DAT_SHOW_ERRORS_ORIG = False     # show error envelopes on original curves
    # (raw template errors can be noisy at long
    # periods — set False to hide them)
    DAT_SHOW_ERRORS_PERT = True    # show error envelopes on perturbed curves
    # (reset relative errors are compact; set True
    # to display ±σ bands on perturbed curves)
    # 0.6        # opacity for original curves (1 = fully opaque)
    DAT_ALPHA_ORIG = 0.5
    # 1.        # opacity for perturbed curves (< 1 lets original show through)
    DAT_ALPHA_PERT = 1.
    
    # Marker symbols per component class.
    # 'ii'  → diagonal components (xx, yy)   — circles
    # 'ij'  → off-diagonal components (xy, yx) — squares
    # 'inv' → invariants (det, bahr, …)      — triangles up
    # Set to None to use the DEFAULT_COMP_MARKERS from femtic_viz.
    # Set to {} to disable markers entirely (lines only).
    DAT_COMP_MARKERS = None
    DAT_MARKERSIZE = 3.0        # marker size in points
    DAT_MARKEVERY = 2        # plot marker every N-th period; None = every period
    
    # Error rendering style — applies when show_errors_orig / show_errors_pert is True.
    # 'shade' : semi-transparent fill_between band (default, existing behaviour)
    # 'bar'   : discrete errorbar caps at each period
    # 'both'  : shade AND bar simultaneously
    DAT_ERROR_STYLE_ORIG = "bar"
    DAT_ERROR_STYLE_PERT = "shade"
    
    # Axis limits for data plots.
    # DAT_PERLIMS applies to the period (x) axis of every panel type.
    # The remaining limits apply to the y-axis of the indicated panel type only.
    # Set any to None for Matplotlib auto-scaling.
    DAT_PERLIMS = (1.e-4, 1.e4)     # (T_min, T_max) in seconds; None = auto
    DAT_RHOLIMS = None              # (rho_min, rho_max) Ω·m (or log₁₀); None = auto
    DAT_PHSLIMS = (-180., 180.)     # (phs_min, phs_max) degrees; None = auto
    DAT_VTFLIMS = (-1.,    +1.)     # (vtf_min, vtf_max); None = auto
    DAT_PTLIMS = None               # (pt_min,  pt_max);  None = auto
    
    
    MOD_MESH = TEMPLATES + "mesh.dat"

    # --- Ocean / air handling (must match the inversion setup) ---------------
    #: None → auto-infer; True / False → force ocean-present / ocean-absent.
    MOD_OCEAN     = None
    MOD_AIR_RHO   = 1.0e9   # Ω·m  (region 0)
    MOD_OCEAN_RHO = 0.25    # Ω·m  (region 1 when treated as ocean)

    # --- Geographic / UTM origin of the mesh centre --------------------------
    #: Set to None when ORIGIN_METHOD will estimate the origin from SITE_DAT.
    MOD_UTM_ORIGIN_LAT = None   # decimal degrees, positive = North
    MOD_UTM_ORIGIN_LON = None   # decimal degrees, positive = East
    MOD_UTM_ORIGIN_E   = None   # UTM easting  [m]
    MOD_UTM_ORIGIN_N   = None   # UTM northing [m]
    MOD_UTM_ZONE_OVERRIDE = None  # override auto-derived zone; None = auto

    # --- Display coordinate system -------------------------------------------
    #: "model"  — axis ticks in model-local metres (default)
    #: "utm"    — axis ticks in absolute UTM metres
    #: "latlon" — axis ticks in decimal degrees
    MOD_DISPLAY_COORDS = "model"

    # --- Site overlay ---------------------------------------------------------
    #: Primary source: mt_make_sitelist.py CSV (name,lat,lon,elev,sitenum,E,N).
    #: Set to None to fall back to observe.dat / MOD_SITE_NUMBER.
    MOD_SITE_DAT   = TEMPLATES + "site.dat"   # set to None to disable
    MOD_SITE_NAMES = None   # list of names to plot, or None = all sites
    #: Fallback: site number(s) from observe.dat (int or list of ints).
    MOD_SITE_NUMBER = None
    MOD_PLOT_SITES_MAPS   = True   # show markers on map panels
    MOD_PLOT_SITES_SLICES = True   # show markers on curtain / plane panels
    #: Max distance [m] from a curtain plane for a site to appear on it.
    MOD_PROJECTION_DIST = 1000.   # metres; None = show all sites on every panel
    MOD_SITE_MARKER = dict(marker="v", color="black", ms=4, zorder=10, label=None)
    MOD_SITE_MARKER_SLICES = dict(marker="v", color="black", ms=4, zorder=10, label=None)
    #: Extra point markers on map panels only (each dict: latlon, marker, color, ms, name).
    MOD_MAP_MARKERS = []

    # --- Mesh-centre estimation from site.dat (optional) ---------------------
    #: None → use hard-coded values above.
    #: "box"     → midpoint of UTM bounding box of all sites in MOD_SITE_DAT.
    #: "average" → arithmetic mean of UTM coordinates in MOD_SITE_DAT.
    MOD_ORIGIN_METHOD = "box"   # None | "box" | "average"

    # --- Plotting -----------------------------------------------------------
    MOD_DPI       = 200
    MOD_CMAP      = "turbo_r"
    MOD_CLIM      = [0.0, 3.0]     # [log10_min, log10_max] Ω·m; None = auto
    MOD_OCEAN_COLOR  = "lightgrey" # flat colour for ocean cells; None = colormap
    MOD_AIR_COLOR    = "whitesmoke"
    MOD_AIR_BGCOLOR  = None

    # --- Alpha / blanking by second block file (optional) -------------------
    MOD_ALPHA_FILE        = None   # path to sensitivity block; None = disabled
    MOD_ALPHA_MODE        = "fade" # "fade" | "blank"
    MOD_ALPHA_BLANK_THRESH = 0.0

    # --- Slice specification -------------------------------------------------
    #: Slice positions accept plain floats (model-local m) or CRS-tagged tuples:
    #:   (value, "utm") | (value, "latlon")
    #: Depth z0 is always model-local metres (no CRS tagging).
    MOD_SLICES = [
        dict(kind="map", z0=1000.0),
        dict(kind="map", z0=25000.0),
        # dict(kind="ns",  x0=0.0),
        # dict(kind="ew",  y0=0.0),
        dict(kind="ns",  x0=(-71.536322, "latlon")),
        dict(kind="ew",  y0=(-16.196900, "latlon")),
    ]
    MOD_XLIM = [-25000., 25000.]   # [xmin, xmax] model-local m; None = auto
    MOD_YLIM = [-25000., 25000.]   # [ymin, ymax] model-local m; None = auto
    MOD_ZLIM = [ -10000., 30000.]   # [zmin, zmax] model-local m; None = auto

    # --- Figure layout -------------------------------------------------------
    MOD_EQUAL_ASPECT  = True
    MOD_DEPTH_KM      = True
    MOD_HORIZ_KM      = True
    MOD_NROWS         = None   # None = auto (1 row)
    MOD_NCOLS         = None   # None = auto (len(MOD_SLICES) cols)
    MOD_PANEL_HEIGHT  = 16.0   # cm
    MOD_PANEL_WIDTH   = None   # cm; None = auto from aspect ratio
    MOD_FIGSIZE       = None   # [w, h] cm; overrides auto when set

    # --- Ensemble slice plot (femtic_viz.plot_ensemble_slices) ---------------
    #: Set True to produce a joint member × slice figure after generation.
    PLOT_SLICES_ENS = False
    ENS_SLICES      = MOD_SLICES   # reuse same slice specs; override if needed
    ENS_CMAP        = MOD_CMAP
    ENS_CLIM        = MOD_CLIM
    ENS_XLIM        = MOD_XLIM
    ENS_YLIM        = MOD_YLIM
    ENS_ZLIM        = MOD_ZLIM
    ENS_OCEAN_COLOR = MOD_OCEAN_COLOR
    ENS_STAT_ROWS   = ["mean", "std"]   # subset of "mean", "std", "median"
    ENS_PER_MEMBER  = False
    ENS_PLOT_DPI    = 300
    ENS_PLOT_FILE   = PLOT_DIR + "rto_ensemble_slices" + PLOT_STR + ".pdf"

    # --- QC slice plot of perturbed initial models ---------------------------
    #: Set True to produce one slice figure per selected member.
    PLOT_SLICES_QC = False


"""
Generate ensemble directories and copy template files.
"""
dir_list = ens.generate_directories(alg="rto",
                                    dir_base=ENSEMBLE_DIR + ENSEMBLE_NAME,
                                    templates=TEMPLATES,
                                    copy_list=COPY_LIST,
                                    link_list=LINK_LIST,
                                    n_samples=N_SAMPLES,
                                    fromto=FROM_TO,
                                    relative_links=RELATIVE_LINKS,
                                    out=True)
print("\n")

"""
Draw a random subset of ensemble members for visualization.
Used by both the data plot and the model plot.
"""
_n_viz = min(VIZ_N_SAMPLES, N_SAMPLES)
VIZ_SAMPLES = sorted(rng.choice(
    N_SAMPLES, size=_n_viz, replace=False).tolist())
print(f"Visualization members: {VIZ_SAMPLES}\n")

"""
Draw perturbed data sets: d̃ ∼ N(d, Cd)
"""
data_ensemble = ens.generate_data_ensemble(alg="rto",
                                           dir_base=ENSEMBLE_DIR + ENSEMBLE_NAME,
                                           n_samples=N_SAMPLES,
                                           fromto=FROM_TO,
                                           file_in="observe.dat",
                                           draw_from=DAT_PDF,
                                           method=DAT_METHOD,
                                           errors=ERRORS,
                                           out=True)
print("data ensemble ready!")
print("\n")

"""
Data visualization
------------------
Joint plot of original vs. perturbed observe.dat for the selected samples.
One subplot row per sample; original (solid) and perturbed (dashed) curves
are overlaid on the same axes.  VIZ_N_SITES randomly chosen sites are shown
per row (all sites if VIZ_N_SITES is None).

Helper: femtic_viz.plot_data_ensemble
"""
if PLOT_DATA:
    dat_orig_file = TEMPLATES + "observe.dat"
    dat_ens_files = [
        ENSEMBLE_DIR + ENSEMBLE_NAME + f"{i}/observe.dat"
        for i in range(N_SAMPLES)
    ]

    for i_samp in VIZ_SAMPLES:
        fig_dat, axs_dat = fviz.plot_data_ensemble(
            orig_file=dat_orig_file,
            ens_files=dat_ens_files,
            sample_indices=[i_samp],
            what=DAT_WHAT,
            comps=DAT_COMPS,
            show_errors_orig=DAT_SHOW_ERRORS_ORIG,
            show_errors_pert=DAT_SHOW_ERRORS_PERT,
            error_style_orig=DAT_ERROR_STYLE_ORIG,
            error_style_pert=DAT_ERROR_STYLE_PERT,
            n_sites=VIZ_N_SITES,
            alpha_orig=DAT_ALPHA_ORIG,
            alpha_pert=DAT_ALPHA_PERT,
            comp_markers=DAT_COMP_MARKERS,
            markersize=DAT_MARKERSIZE,
            markevery=DAT_MARKEVERY,
            perlims=DAT_PERLIMS,
            rholims=DAT_RHOLIMS,
            phslims=DAT_PHSLIMS,
            vtflims=DAT_VTFLIMS,
            ptlims=DAT_PTLIMS,
            out=True,
        )
        member_dir = ENSEMBLE_DIR + ENSEMBLE_NAME + f"{i_samp}/"
        plot_path = member_dir + "rto_data" + PLOT_STR + ".pdf"
        fig_dat.savefig(plot_path, bbox_inches="tight")
        plt.close(fig_dat)
        print(f"  data plot saved: {plot_path}")
    print("data ensemble plots saved.")

"""
Draw perturbed model sets: m̃ ∼ N(m, Cm)

R is passed directly — Q = R^T R is formed implicitly inside
generate_rto_model_ensemble, never explicitly materialised.
"""
R = scs.load_npz(R_FILE + ".npz")
print("roughness R loaded with shape:", np.shape(R))

model_ensemble = ens.generate_rto_model_ensemble(
    alg="rto",
    dir_base=ENSEMBLE_DIR + ENSEMBLE_NAME,
    n_samples=N_SAMPLES,
    fromto=FROM_TO,
    refmod=MOD_REF_BASE,
    method=MOD_METHOD,
    algo=MOD_ALGO,
    q=R,                          # pass R directly — Q formed implicitly
    # low-rank options
    n_eig=MOD_N_EIG,
    n_oversampling=MOD_N_OVERSAMPLING,
    n_power_iter=MOD_N_POWER_ITER,
    sigma2_residual=MOD_SIGMA2_RESIDUAL,
    # full-rank options
    lam=MOD_LAM,
    lam_mode=MOD_LAM_MODE,
    lam_alpha=MOD_LAM_ALPHA,
    solver_method=MOD_SOLVER,
    precond=MOD_PRECOND,
    rng=rng,
    out=True,
)
print("\n")
print("model ensemble ready!")

"""
QC and model slice plots of perturbed initial models
----------------------------------------------------
Uses fviz.plot_model_slices (exact tetrahedron-plane intersection) with the
full femtic_mod_plot_slice config: geographic CRS support, site overlay,
display-coordinate system, alpha/blanking.

PLOT_SLICES_QC: one figure per selected member, saved as rto_qc<PLOT_STR>.pdf
                in each member's subdirectory.
PLOT_MODEL:     same call but for the ensemble member's final-iterate model
                (or iter0 before inversion).  Saved as rto_model<PLOT_STR>.pdf.
Both flags are independent; set either or both to True.
"""
if (PLOT_DATA or PLOT_MODEL or PLOT_SLICES_QC) and (PLOT_MODEL or PLOT_SLICES_QC):

    # --- resolve UTM origin --------------------------------------------------
    _mod_utm_origin_lat = MOD_UTM_ORIGIN_LAT
    _mod_utm_origin_lon = MOD_UTM_ORIGIN_LON
    _mod_utm_origin_e   = MOD_UTM_ORIGIN_E
    _mod_utm_origin_n   = MOD_UTM_ORIGIN_N

    if MOD_ORIGIN_METHOD is not None:
        _mod_site_dat_path = MOD_SITE_DAT if MOD_SITE_DAT and os.path.isfile(MOD_SITE_DAT) else None
        if _mod_site_dat_path is None:
            print("  WARNING: MOD_ORIGIN_METHOD set but MOD_SITE_DAT not available "
                  "— using hard-coded origin.")
        else:
            _sdat = fem.read_site_dat(_mod_site_dat_path)
            if _sdat:
                _Es = np.array([d["easting"]  for d in _sdat])
                _Ns = np.array([d["northing"] for d in _sdat])
                if MOD_ORIGIN_METHOD == "box":
                    _mod_utm_origin_e = 0.5 * (_Es.min() + _Es.max())
                    _mod_utm_origin_n = 0.5 * (_Ns.min() + _Ns.max())
                elif MOD_ORIGIN_METHOD == "average":
                    _mod_utm_origin_e = float(_Es.mean())
                    _mod_utm_origin_n = float(_Ns.mean())
                _lats = np.array([d["lat"] for d in _sdat])
                _lons = np.array([d["lon"] for d in _sdat])
                _mod_utm_zone, _mod_utm_northern = utl.utm_zone_from_latlon(
                    float(_lats.mean()), float(_lons.mean()),
                    override=MOD_UTM_ZONE_OVERRIDE)
                _mod_utm_origin_lat, _mod_utm_origin_lon = utl.utm_to_latlon_zn(
                    _mod_utm_origin_e, _mod_utm_origin_n,
                    _mod_utm_zone, _mod_utm_northern)
                print(f"  Model plot origin ({MOD_ORIGIN_METHOD}): "
                      f"E={_mod_utm_origin_e:.1f} m, N={_mod_utm_origin_n:.1f} m, "
                      f"lat={_mod_utm_origin_lat:.5f}°, lon={_mod_utm_origin_lon:.5f}°")

    _mod_utm_zone, _mod_utm_northern = utl.utm_zone_from_latlon(
        _mod_utm_origin_lat, _mod_utm_origin_lon,
        override=MOD_UTM_ZONE_OVERRIDE)

    # --- resolve slice positions ---------------------------------------------
    _mod_slices_resolved = fem.resolve_slice_positions(
        MOD_SLICES,
        _mod_utm_zone, _mod_utm_northern,
        _mod_utm_origin_e, _mod_utm_origin_n,
        _mod_utm_origin_lat, _mod_utm_origin_lon,
        verbose=OUT,
    )

    # --- read site positions for overlay ------------------------------------
    _mod_site_xys = []
    _mod_sites_from_obs = False
    _mod_need_sites = MOD_PLOT_SITES_MAPS or MOD_PLOT_SITES_SLICES
    if _mod_need_sites and MOD_SITE_DAT is not None and os.path.isfile(MOD_SITE_DAT):
        _rows = fem.read_site_dat(MOD_SITE_DAT, site_names=MOD_SITE_NAMES)
        for row in _rows:
            sx_m, sy_m = fem.utm_to_model(row["easting"], row["northing"],
                                          _mod_utm_origin_e, _mod_utm_origin_n)
            _mod_site_xys.append((row["name"], sx_m, sy_m,
                                  float(row.get("elev", 0.0))))
    elif _mod_need_sites and MOD_SITE_NUMBER is not None:
        _obs_file = TEMPLATES + "observe.dat"
        _site_nums = (MOD_SITE_NUMBER if isinstance(MOD_SITE_NUMBER, (list, tuple))
                      else [MOD_SITE_NUMBER])
        for _sn in _site_nums:
            sx_m, sy_m = fem.read_site_position(_obs_file, _sn)
            _mod_site_xys.append((_sn, sx_m, sy_m, 0.0))
        _mod_sites_from_obs = True

    # --- helper: call plot_model_slices for one model file -------------------
    def _plot_member_slices(mod_file, out_pdf):
        fviz.plot_model_slices(
            model_file      = mod_file,
            mesh_file       = MOD_MESH,
            slices          = _mod_slices_resolved,
            cmap            = MOD_CMAP,
            clim            = MOD_CLIM,
            xlim            = MOD_XLIM,
            ylim            = MOD_YLIM,
            zlim            = MOD_ZLIM,
            ocean_color     = MOD_OCEAN_COLOR,
            ocean_value     = MOD_OCEAN_RHO,
            air_color       = MOD_AIR_COLOR,
            air_bgcolor     = MOD_AIR_BGCOLOR,
            site_xys        = _mod_site_xys,
            obs_coords_only = _mod_sites_from_obs,
            projection_dist = MOD_PROJECTION_DIST,
            sites_in_maps   = MOD_PLOT_SITES_MAPS,
            sites_in_slices = MOD_PLOT_SITES_SLICES,
            site_marker     = MOD_SITE_MARKER,
            site_marker_slices = MOD_SITE_MARKER_SLICES,
            map_markers     = MOD_MAP_MARKERS,
            display_coords  = MOD_DISPLAY_COORDS,
            utm_origin_e    = _mod_utm_origin_e,
            utm_origin_n    = _mod_utm_origin_n,
            utm_zone        = _mod_utm_zone,
            utm_northern    = _mod_utm_northern,
            utm_to_latlon_fn   = utl.utm_to_latlon_zn,
            latlon_to_model_fn = fem.latlon_to_model,
            plot_file       = out_pdf,
            dpi             = MOD_DPI,
            equal_aspect    = MOD_EQUAL_ASPECT,
            depth_km        = MOD_DEPTH_KM,
            horiz_km        = MOD_HORIZ_KM,
            nrows           = MOD_NROWS,
            ncols           = MOD_NCOLS,
            panel_height    = MOD_PANEL_HEIGHT / 2.54,
            panel_width     = MOD_PANEL_WIDTH / 2.54 if MOD_PANEL_WIDTH is not None else None,
            figsize         = [v / 2.54 for v in MOD_FIGSIZE] if MOD_FIGSIZE is not None else None,
            alpha_file      = MOD_ALPHA_FILE,
            alpha_mode      = MOD_ALPHA_MODE,
            alpha_blank_thresh = MOD_ALPHA_BLANK_THRESH,
            out             = OUT,
        )

    # --- QC plots of perturbed initial models (iter0) -----------------------
    if PLOT_SLICES_QC:
        _qc_files = [
            ENSEMBLE_DIR + ENSEMBLE_NAME + f"{i}/resistivity_block_iter0.dat"
            for i in range(N_SAMPLES)
        ]
        for i_samp in VIZ_SAMPLES:
            if not os.path.isfile(_qc_files[i_samp]):
                print(f"  QC: {_qc_files[i_samp]} not found — skipped.")
                continue
            _qc_pdf = (ENSEMBLE_DIR + ENSEMBLE_NAME
                       + f"{i_samp}/rto_qc{PLOT_STR}.pdf")
            _plot_member_slices(_qc_files[i_samp], _qc_pdf)
            print(f"  QC slice plot saved: {_qc_pdf}")
        print("QC slice plots done.")

    # --- Per-member model slice plots ----------------------------------------
    if PLOT_MODEL:
        _mod_files = [
            ENSEMBLE_DIR + ENSEMBLE_NAME + f"{i}/{MOD_REF_BASE}"
            for i in range(N_SAMPLES)
        ]
        for i_samp in VIZ_SAMPLES:
            if not os.path.isfile(_mod_files[i_samp]):
                print(f"  Model: {_mod_files[i_samp]} not found — skipped.")
                continue
            _mod_pdf = (ENSEMBLE_DIR + ENSEMBLE_NAME
                        + f"{i_samp}/rto_model{PLOT_STR}.pdf")
            _plot_member_slices(_mod_files[i_samp], _mod_pdf)
            print(f"  model slice plot saved: {_mod_pdf}")
        print("model slice plots done.")

"""
Ensemble slice plot
-------------------
Joint figure of all ensemble members using exact tetrahedron-plane intersection.
One row per member, columns = slices defined by ENS_SLICES.
Optional stat rows (mean, std, median of log10(ρ)) are appended at the bottom.

Helper: femtic_viz.plot_ensemble_slices
"""
if PLOT_DATA or PLOT_MODEL:   # only runs when the viz block was entered
    if PLOT_SLICES_ENS:
        # Build the list of converged model files for all members.
        # Adjust the filename pattern to match the desired iteration.
        _ens_block_files = [
            ENSEMBLE_DIR + ENSEMBLE_NAME + f"{i}/resistivity_block_iter0.dat"
            for i in range(N_SAMPLES)
        ]
        _ens_labels = [f"{ENSEMBLE_NAME}{i}" for i in range(N_SAMPLES)]

        fviz.plot_ensemble_slices(
            member_files    = _ens_block_files,
            mesh_file       = MOD_MESH,
            slices          = ENS_SLICES,
            labels          = _ens_labels,
            stat_rows       = ENS_STAT_ROWS,
            cmap            = ENS_CMAP,
            clim            = ENS_CLIM,
            xlim            = ENS_XLIM,
            ylim            = ENS_YLIM,
            zlim            = ENS_ZLIM,
            ocean_color     = ENS_OCEAN_COLOR,
            ocean_value     = 0.25,
            depth_km        = True,
            horiz_km        = True,
            per_member_file = ENS_PER_MEMBER,
            plot_file       = ENS_PLOT_FILE,
            dpi             = ENS_PLOT_DPI,
            out             = True,
        )
        print("ensemble slice plot saved.")
