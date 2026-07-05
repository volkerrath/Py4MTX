#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Run the Geostatistical (GST) ensemble algorithm for MT inversion:

    for i = 1 : nsamples do
        Draw perturbed data set: d_pert ~ N(d, Cd)
        Draw log10(rho) values at pilot points, either
            ~ Uniform(rho_min, rho_max)                      [MOD_PP_VALUE_MODE = "uniform"]
            or referencemodel(pilot point) +- delta (log10)  [MOD_PP_VALUE_MODE = "reference"]
        Kriging-interpolate pilot-point values to all mesh cells -> initial model m0_i
        Solve deterministic inversion starting from m0_i to get model m_i = m(m0_i)
    end

The model perturbation follows the geostatistical initial-model ensemble approach
described in:

    Suzuki, K.; ...
        Geostatistical initial-model ensemble for magnetotelluric uncertainty
        quantification.
        [full reference to be completed]

The data perturbation follows:

    Bardsley, J. M.; Solonen, A.; Haario, H. & Laine, M.
        Randomize-Then-Optimize: a Method for Sampling from Posterior
        Distributions in Nonlinear Inverse Problems.
        SIAM J. Sci. Comp., 2014, 36, A1895-A1910

Pilot points may be placed on a user-supplied grid, drawn randomly inside the
survey bounding box, or mixed (a fixed set supplemented by random points).
Ordinary Kriging (gstools) is used to interpolate from the pilot-point cloud
to all FEMTIC mesh cell centres.  The variogram model (type, range, sill,
nugget, anisotropy) is fully user-configurable.

The low-level Kriging is encapsulated in:
    ensembles.generate_gst_model_ensemble()

Created on Mon Apr 27 00:00:00 2026

@author: vrath

Provenance:
    2026-04-27  vrath / Claude   Created, modelled on femtic_rto_prep.py.
                                 GST model perturbation replaces roughness-matrix
                                 prior draw with pilot-point Ordinary Kriging
                                 (gstools).  Low-level Kriging lives in
                                 ens.generate_gst_model_ensemble().
                                 Data perturbation block unchanged from RTO.
    2026-05-13  Claude           Added PLOT_SLICES_ENS block: ENS_SLICES /
                                 ENS_CMAP / ENS_CLIM / ENS_STAT_ROWS config;
                                 calls fviz.plot_ensemble_slices for exact
                                 tet-plane intersection ensemble figure.
                                 Member file list uses MOD_RESISTIVITY_FILE.
    2026-05-27  vrath / Claude Sonnet 4.6 (Anthropic)
                                 Added QC slice plot step after model ensemble
                                 generation: PLOT_SLICES_QC / QC_* config;
                                 calls fviz.plot_model_slices per member,
                                 saves gst_qc*.pdf in each member's subdir.
    2026-05-31  vrath / Claude Sonnet 4.6   MOD_SLICES updated to use
                "kind" key ("map"/"ns"/"ew") instead of "type".
                depth_km=True, horiz_km=True added to plot_model_slices
                and plot_ensemble_slices QC/ENS calls.
    2026-05-28  Claude Sonnet 4.6 (Anthropic)
                Added RELATIVE_LINKS config variable (default True); passed as
                relative_links to ens.generate_directories.  Relative symlinks
                survive tgz/copy to another machine; set False for legacy
                absolute-path behaviour.
    2026-06-06  Claude Sonnet 4.6 (Anthropic)
                Added "extrema" pilot-point mode: MOD_PP_ROI, MOD_PP_EXTREMA_K,
                MOD_PP_EXTREMA_WHICH config variables; passed to
                ens.generate_gst_model_ensemble as pp_roi, pp_extrema_k,
                pp_extrema_which.  Pilot points are seeded at local log10(rho)
                minima/maxima of the reference model within the ROI, plus
                MOD_N_PP random fill points.  Requires scipy.spatial (KDTree).
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
    2026-07-05  vrath / Claude Sonnet 5 (Anthropic)
                Added MOD_PP_VALUE_MODE / MOD_PP_VALUE_DELTA config
                variables, passed through to ens.generate_gst_model_ensemble
                as pp_value_mode / pp_value_delta.  "uniform" (default)
                preserves the original Uniform(MOD_LOG_RHO_MIN,
                MOD_LOG_RHO_MAX) pilot-point draw; "reference" instead draws
                pilot-point values as referencemodel(nearest free region)
                +- MOD_PP_VALUE_DELTA (log10 Ohm.m), keeping the ensemble
                anchored to the reference structure.
"""

import os
import sys
from pathlib import Path
import numpy as np
import inspect
import matplotlib.pyplot as plt

"""
Py4MTX-specific settings and imports.
"""
PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

for _base in [PY4MTX_ROOT + "/py4mt/modules/"]:
    for _p in [Path(_base), *Path(_base).rglob("*")]:
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

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
nan = np.nan
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
# ENSEMBLE_NAME = "ubinas_gst_suzuki_"

ENSEMBLE_DIR = r"/home/vrath/Py4MTX/py4mt/data/ensembles/misti/ensemble/"
ENSEMBLE_NAME = "misti_gst_suzuki_"

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
of badly converged samples (see femtic_gst_post.py).
"""
FROM_TO = None


"""
Set up mode of model perturbations.
-----------------------------------------------------------------------
The GST method generates a distinct initial model m0 for each ensemble
member by:
    (1) Assigning log10(rho) values at a set of pilot points covering the
        survey area and depth range of interest, either drawn uniformly
        at random or as a perturbation of the reference model
        (see MOD_PP_VALUE_MODE).
    (2) Kriging-interpolating those values to every mesh cell centre.
    (3) Clamping the result to [MOD_LOG_RHO_MIN, MOD_LOG_RHO_MAX].
    (4) Writing the result as resistivity_block_iter0.dat and/or
        referencemodel.dat for that member.

No roughness matrix is required.
Low-level Kriging: ens.generate_gst_model_ensemble()
-----------------------------------------------------------------------
"""
PERTURB_MOD = True
if PERTURB_MOD:
    MOD_REF = TEMPLATES + "referencemodel.dat"
    MOD_REF_BASE = os.path.basename(MOD_REF)
    MOD_MESH = TEMPLATES + "mesh.dat"
    # ------------------------------------------------------------------
    # Pilot-point configuration
    # ------------------------------------------------------------------
    # MOD_PP_MODE controls how pilot-point locations are determined:
    #   "random"  : MOD_N_PP points drawn uniformly inside MOD_PP_BBOX.
    #   "fixed"   : use the explicit list in MOD_PP_COORDS (Nx3 array,
    #               columns = easting, northing, depth [m]).
    #   "mixed"   : start with MOD_PP_COORDS and add MOD_N_PP random points.
    #   "extrema" : seed pilot points at local log10(rho) minima and/or
    #               maxima of the reference model within MOD_PP_ROI, then
    #               add MOD_N_PP random fill points inside MOD_PP_BBOX.
    #               Structural skeleton is the same every member; values
    #               and fill locations change.  Requires scipy.spatial.
    MOD_PP_MODE = "extrema" # "random"     # "random" | "fixed" | "mixed" | "extrema"

    # Number of randomly drawn pilot points per member.
    # Used when MOD_PP_MODE = "random", "mixed", or "extrema" (fill).
    # Recommended: 50–200 for typical 3-D MT survey volumes.
    MOD_N_PP = 100

    # Bounding box for random pilot-point placement:
    #   [x_min, x_max, y_min, y_max, z_min, z_max]  (metres, z positive-down)
    MOD_PP_BBOX = [-25000., 25000.,   # easting  range (m)
                   -25000., 25000.,   # northing range (m)
                        0., 60000.]   # depth     range (m, positive-down)

    # Explicit pilot-point coordinates used when MOD_PP_MODE = "fixed"
    # or "mixed".  Shape: (N, 3) — columns: [easting, northing, depth].
    MOD_PP_COORDS = None  # e.g. np.array([[x, y, z], ...])

    # ------------------------------------------------------------------
    # "extrema" mode — pilot points seeded at local resistivity extrema
    # ------------------------------------------------------------------
    # MOD_PP_ROI: bounding box [x_min, x_max, y_min, y_max, z_min, z_max]
    #   restricting which free regions are eligible as extremum seeds.
    #   None = full free-region extent (equivalent to MOD_PP_BBOX).
    #   z positive-down (FEMTIC convention).
    #   Tip: tighten to the survey footprint to exclude deep/lateral padding.
    #
    # MOD_PP_EXTREMA_K: neighbourhood size (number of nearest neighbours,
    #   including self) for the local extremum test.  Larger k → smoother
    #   field, fewer extrema.  Recommended: 7–15 for typical FEMTIC meshes.
    #
    # MOD_PP_EXTREMA_WHICH: which extrema to use as seeds.
    #   "both"   — conductive and resistive anomaly cores (recommended).
    #   "minima" — conductive anomalies only (low resistivity).
    #   "maxima" — resistive anomalies only (high resistivity).
    MOD_PP_ROI           = None   # None = full extent; or [xmn,xmx,ymn,ymx,zmn,zmx]
    MOD_PP_EXTREMA_K     = 32   # neighbourhood size for extremum detection
    MOD_PP_EXTREMA_WHICH = "both" # "both" | "minima" | "maxima"

    # ------------------------------------------------------------------
    # Resistivity range for pilot-point values
    # ------------------------------------------------------------------
    # Pilot-point values are drawn Uniform(MOD_LOG_RHO_MIN, MOD_LOG_RHO_MAX)
    # in log10(Ohm.m).  After Kriging the interpolated field is clamped to
    # the same interval.
    MOD_LOG_RHO_MIN = 0.0    # log10(1 Ohm.m)
    MOD_LOG_RHO_MAX = 4.0    # log10(10000 Ohm.m)

    # ------------------------------------------------------------------
    # Pilot-point value mode
    # ------------------------------------------------------------------
    # MOD_PP_VALUE_MODE controls how pilot-point log10(rho) values are drawn:
    #   "uniform"   : Uniform(MOD_LOG_RHO_MIN, MOD_LOG_RHO_MAX) at every
    #                 pilot point, independent of location (default,
    #                 original behaviour).
    #   "reference" : referencemodel(pilot point) +- MOD_PP_VALUE_DELTA
    #                 (log10 Ohm.m).  The reference log10(rho) is looked up
    #                 at the free region nearest each pilot point and
    #                 perturbed by Uniform(-MOD_PP_VALUE_DELTA,
    #                 +MOD_PP_VALUE_DELTA).  Result is still clamped to
    #                 [MOD_LOG_RHO_MIN, MOD_LOG_RHO_MAX].  Use this to keep
    #                 the ensemble anchored to the reference structure
    #                 instead of exploring the full resistivity range at
    #                 each pilot point.
    MOD_PP_VALUE_MODE = "reference"  # "uniform" | "reference"

    # Half-width (log10 Ohm.m) of the symmetric perturbation around the
    # reference value.  Only used when MOD_PP_VALUE_MODE = "reference".
    # Typical: 0.3-1.0 (factor ~2-10 in resistivity).
    MOD_PP_VALUE_DELTA = 0.5

    # ------------------------------------------------------------------
    # Variogram model
    # ------------------------------------------------------------------
    # MOD_VARIO_MODEL: gstools covariance model class name (string).
    #   Common choices: "Spherical", "Gaussian", "Exponential",
    #                   "Matern", "Linear", "PowerLaw".
    #
    # MOD_VARIO_RANGE: correlation length (m).  A 2-tuple
    #   (horizontal_range, vertical_range) sets geometric anisotropy.
    #   A scalar applies isotropically.
    #   Recommended: h_range ≈ half survey aperture; v_range ≈ half target depth.
    #
    # MOD_VARIO_SILL: sill (variance) in (log10 Ohm.m)^2.
    #   Recommended: 0.25–0.5  (≈ ±0.5–0.7 log10 units 1-sigma).
    #
    # MOD_VARIO_NUGGET: nugget in (log10 Ohm.m)^2.
    #   Keep ≤ 10% of sill for spatial coherence.
    #
    # MOD_VARIO_ANGLES: rotation [α, β, γ] in degrees (optional).
    #   None = axis-aligned anisotropy.
    MOD_VARIO_MODEL   = "Spherical"     # gstools covariance model class
    MOD_VARIO_RANGE   = (8000., 4000.) # (horizontal, vertical) ranges in m
    MOD_VARIO_SILL    = 0.5             # sill in (log10 Ohm.m)^2
    MOD_VARIO_NUGGET  = 0.01            # nugget in (log10 Ohm.m)^2
    MOD_VARIO_ANGLES  = None            # [alpha, beta, gamma] deg; None = axis-aligned

    # ------------------------------------------------------------------
    # Output target: which file(s) receive the Kriged initial model?
    # ------------------------------------------------------------------
    # "resistivity_block" — writes resistivity_block_iter0.dat only.
    # "referencemodel"    — writes referencemodel.dat only.
    # "both"              — writes both (recommended).
    MOD_OUTPUT_TARGET    = "both"
    MOD_RESISTIVITY_FILE = "resistivity_block_iter0.dat"
    MOD_REFERENCE_FILE   = "referencemodel.dat"

else:
    MOD_REF      = None
    MOD_REF_BASE = None


"""
Set up mode of data perturbations.
(Identical to the RTO data-perturbation block.)
"""
PERTURB_DAT = False
if PERTURB_DAT:
    DAT_METHOD = "add"
    DAT_PDF = ["normal", 0., 1.]

RESET_ERRORS = True
if RESET_ERRORS:
    ERRORS = [
        [0.25, .1, .1, 0.25] * 2,   # Impedance
        [0.05, 0.05] * 2,            # VTF
        [.5, .2, .2, .5],            # PT
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
PLOT_DATA  = False  # True
PLOT_MODEL = True  # True

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
    # Set to None to show all available sites.
    VIZ_N_SITES = 10

    # --- data plot ---
    DAT_WHAT  = ["rho", "phase"]
    DAT_COMPS = ["xy,yx",
                 "xy,yx"]

    DAT_SHOW_ERRORS_ORIG = False
    DAT_SHOW_ERRORS_PERT = True
    DAT_ALPHA_ORIG = 0.5
    DAT_ALPHA_PERT = 1.

    # 'ii'  → diagonal components (xx, yy)    — circles
    # 'ij'  → off-diagonal components (xy, yx) — squares
    # 'inv' → invariants                       — triangles up
    DAT_COMP_MARKERS = None
    DAT_MARKERSIZE   = 3.0
    DAT_MARKEVERY    = 2

    # 'shade' | 'bar' | 'both'
    DAT_ERROR_STYLE_ORIG = "bar"
    DAT_ERROR_STYLE_PERT = "shade"

    # Axis limits.  None = Matplotlib auto-scaling.
    DAT_PERLIMS = (1.e-4, 1.e4)
    DAT_RHOLIMS = None
    DAT_PHSLIMS = (-180., 180.)
    DAT_VTFLIMS = (-1.,   +1.)
    DAT_PTLIMS  = None

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
    MOD_CLIM      = [0.0, 4.0]     # [log10_min, log10_max] Ω·m; None = auto
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
        dict(kind="map", z0=5000.0),
        dict(kind="map", z0=15000.0),
        dict(kind="ns",  x0=0.0),
        dict(kind="ew",  y0=0.0),
    ]
    MOD_XLIM = [-15000., 15000.]   # [xmin, xmax] model-local m; None = auto
    MOD_YLIM = [-15000., 15000.]   # [ymin, ymax] model-local m; None = auto
    MOD_ZLIM = [    0.,  30000.]   # [zmin, zmax] model-local m; None = auto

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
    ENS_PLOT_FILE   = PLOT_DIR + "gst_ensemble_slices" + PLOT_STR + ".pdf"

    # --- QC slice plot of Kriged initial models ------------------------------
    #: Set True to produce one slice figure per selected member.
    PLOT_SLICES_QC = False


"""
Generate ensemble directories and copy template files.
"""
dir_list = ens.generate_directories(alg="gst",
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
_n_viz = min(VIZ_N_SAMPLES, N_SAMPLES) if (PLOT_DATA or PLOT_MODEL) else 0
if _n_viz > 0:
    VIZ_SAMPLES = sorted(rng.choice(
        N_SAMPLES, size=_n_viz, replace=False).tolist())
    print(f"Visualization members: {VIZ_SAMPLES}\n")
else:
    VIZ_SAMPLES = []


if PERTURB_DAT:
    """
    Draw perturbed data sets: d̃ ∼ N(d, Cd)
    (Same as RTO — the GST method changes only the model perturbation.)
    """
    data_ensemble = ens.generate_data_ensemble(alg="gst",
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
            plot_path = member_dir + "gst_data" + PLOT_STR + ".pdf"
            fig_dat.savefig(plot_path, bbox_inches="tight")
            plt.close(fig_dat)
            print(f"  data plot saved: {plot_path}")
        print("data ensemble plots saved.")
    

"""
Generate geostatistical initial models: m0_i via pilot-point Kriging
--------------------------------------------------------------------
For each ensemble member i:
    (1) Determine pilot-point locations (random / fixed / mixed).
    (2) Draw log10(rho) values at pilot points, per MOD_PP_VALUE_MODE:
        "uniform"   ~ Uniform(MOD_LOG_RHO_MIN, MOD_LOG_RHO_MAX)
        "reference" = referencemodel(nearest free region) +- MOD_PP_VALUE_DELTA
    (3) Ordinary-Krig values to all mesh cell centres (gstools).
    (4) Clamp to [MOD_LOG_RHO_MIN, MOD_LOG_RHO_MAX].
    (5) Write to member directory (controlled by MOD_OUTPUT_TARGET).

Low-level Kriging: ens.generate_gst_model_ensemble()
--------------------------------------------------------------------
"""
if PERTURB_MOD:
    model_ensemble = ens.generate_gst_model_ensemble(
        alg="gst",
        dir_base=ENSEMBLE_DIR + ENSEMBLE_NAME,
        n_samples=N_SAMPLES,
        fromto=FROM_TO,
        ref_mod_file=MOD_REF,
        mesh_file=MOD_MESH,
        pp_mode=MOD_PP_MODE,
        n_pp=MOD_N_PP,
        pp_bbox=MOD_PP_BBOX,
        pp_coords=MOD_PP_COORDS,
        pp_roi=MOD_PP_ROI,
        pp_extrema_k=MOD_PP_EXTREMA_K,
        pp_extrema_which=MOD_PP_EXTREMA_WHICH,
        log_rho_min=MOD_LOG_RHO_MIN,
        log_rho_max=MOD_LOG_RHO_MAX,
        pp_value_mode=MOD_PP_VALUE_MODE,
        pp_value_delta=MOD_PP_VALUE_DELTA,
        vario_model=MOD_VARIO_MODEL,
        vario_range=MOD_VARIO_RANGE,
        vario_sill=MOD_VARIO_SILL,
        vario_nugget=MOD_VARIO_NUGGET,
        vario_angles=MOD_VARIO_ANGLES,
        output_target=MOD_OUTPUT_TARGET,
        resistivity_file=MOD_RESISTIVITY_FILE,
        reference_file=MOD_REFERENCE_FILE,
        rng=rng,
        out=True,
    )
    print("\nmodel ensemble (geostatistical initial models) ready!")
    print("\n")

"""
QC and model slice plots of Kriged initial models
--------------------------------------------------
Uses fviz.plot_model_slices (exact tetrahedron-plane intersection) with the
full femtic_mod_plot_slice config: geographic CRS support, site overlay,
display-coordinate system, alpha/blanking.

PLOT_SLICES_QC: one figure per selected member, saved as gst_qc<PLOT_STR>.pdf
                in each member's subdirectory.
PLOT_MODEL:     same call for the final-iterate (or iter0) model file.
                Saved as gst_model<PLOT_STR>.pdf.
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

    # --- QC plots of Kriged initial models (iter0) --------------------------
    if PLOT_SLICES_QC:
        _qc_files = [
            ENSEMBLE_DIR + ENSEMBLE_NAME + f"{i}/{MOD_RESISTIVITY_FILE}"
            for i in range(N_SAMPLES)
        ]
        for i_samp in VIZ_SAMPLES:
            if not os.path.isfile(_qc_files[i_samp]):
                print(f"  QC: {_qc_files[i_samp]} not found — skipped.")
                continue
            _qc_pdf = (ENSEMBLE_DIR + ENSEMBLE_NAME
                       + f"{i_samp}/gst_qc{PLOT_STR}.pdf")
            _plot_member_slices(_qc_files[i_samp], _qc_pdf)
            print(f"  QC slice plot saved: {_qc_pdf}")
        print("QC slice plots done.")

    # --- Per-member model slice plots ----------------------------------------
    if PLOT_MODEL and PERTURB_MOD:
        _mod_files = [
            ENSEMBLE_DIR + ENSEMBLE_NAME + f"{i}/{MOD_RESISTIVITY_FILE}"
            for i in range(N_SAMPLES)
        ]
        for i_samp in VIZ_SAMPLES:
            if not os.path.isfile(_mod_files[i_samp]):
                print(f"  Model: {_mod_files[i_samp]} not found — skipped.")
                continue
            _mod_pdf = (ENSEMBLE_DIR + ENSEMBLE_NAME
                        + f"{i_samp}/gst_model{PLOT_STR}.pdf")
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
if PLOT_DATA or PLOT_MODEL:
    if PLOT_SLICES_ENS:
        # Build the list of initial-model files for all members.
        # Uses MOD_RESISTIVITY_FILE so the filename matches the one written by
        # generate_gst_model_ensemble.  Change to a converged-iterate filename
        # (e.g. "resistivity_block_iter10.dat") after inversion has run.
        _ens_block_files = [
            ENSEMBLE_DIR + ENSEMBLE_NAME + f"{i}/{MOD_RESISTIVITY_FILE}"
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
