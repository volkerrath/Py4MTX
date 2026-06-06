#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Run the Geostatistical (GST) ensemble algorithm for MT inversion:

    for i = 1 : nsamples do
        Draw perturbed data set: d_pert ~ N(d, Cd)
        Draw random log10(rho) values at pilot points ~ Uniform(rho_min, rho_max)
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

ENSEMBLE_DIR = r"/home/vrath/Py4MTX/py4mt/data/rto/misti/ensemble/"
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
    (1) Assigning random log10(rho) values at a set of pilot points
        covering the survey area and depth range of interest.
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
    MOD_PP_MODE = "random"     # "random" | "fixed" | "mixed" | "extrema"

    # Number of randomly drawn pilot points per member.
    # Used when MOD_PP_MODE = "random", "mixed", or "extrema" (fill).
    # Recommended: 50–200 for typical 3-D MT survey volumes.
    MOD_N_PP = 100

    # Bounding box for random pilot-point placement:
    #   [x_min, x_max, y_min, y_max, z_min, z_max]  (metres, z positive-down)
    MOD_PP_BBOX = [-50000., 50000.,   # easting  range (m)
                   -50000., 50000.,   # northing range (m)
                        0., 80000.]   # depth     range (m, positive-down)

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
    MOD_PP_EXTREMA_K     = 9      # neighbourhood size for extremum detection
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
PLOT_MODEL = False  # True

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

    MOD_MODE   = "tri"
    MOD_LOG10  = True
    MOD_CMAP   = "jet_r"
    MOD_CLIM   = None
    MOD_XLIM   = None
    MOD_YLIM   = None
    MOD_ZLIM   = None
    MOD_MESH_LINES = False
    MOD_MESH_LW    = 0.3
    MOD_MESH_COLOR = "k"
    MOD_SLICES = [
        {"type": "map",     "z0": 5000,  "dz": 50},
        {"type": "map",     "z0": -5000, "dz": 50},
        {"type": "curtain",
         "polyline": np.array([[0., 0.], [10000., 0.]]),
         "width": 500},
    ]

    # --- ensemble slice plot (femtic_viz.plot_ensemble_slices) ---
    # Uses the same exact tet-plane intersection as femtic_mod_plot.
    # Set PLOT_SLICES_ENS = True to produce a joint member × slice figure.
    PLOT_SLICES_ENS = False

    #: Slice specs in model-local metres — same format as femtic_mod_plot PLOT_SLICES.
    #: Supported kinds: "map" (z0), "ns" (x0), "ew" (y0), "plane" (point/strike/dip).
    #: Plain floats only — no CRS tagging here.
    ENS_SLICES = [
        dict(kind="map", z0=5000.0),
        dict(kind="map", z0=15000.0),
        dict(kind="ns",  x0=0.0),
        dict(kind="ew",  y0=0.0),
    ]
    ENS_CMAP         = "turbo_r"
    ENS_CLIM         = [0.0, 4.0]    # log10(Ω·m); None = auto
    ENS_XLIM         = None           # [xmin, xmax] model-local metres; None = auto
    ENS_YLIM         = None
    ENS_ZLIM         = None
    ENS_OCEAN_COLOR  = "lightgrey"
    ENS_STAT_ROWS    = ["mean", "std"]   # any subset of "mean", "std", "median"
    ENS_PER_MEMBER   = False             # also save one figure per member
    ENS_PLOT_DPI     = 300
    ENS_PLOT_FILE    = PLOT_DIR + "gst_ensemble_slices" + PLOT_STR + ".pdf"

    # --- QC slice plot of Kriged initial models ---
    # Uses fviz.plot_model_slices (exact tet-plane intersection, model-local
    # metres only).  One figure per selected member saved into that member's
    # subdirectory.  Set PLOT_SLICES_QC = True to enable.
    PLOT_SLICES_QC   = False

    QC_SLICES = [
        dict(kind="map", z0=5000.0),
        dict(kind="map", z0=15000.0),
        dict(kind="ns",  x0=0.0),
        dict(kind="ew",  y0=0.0),
    ]
    QC_CMAP        = "turbo_r"
    QC_CLIM        = [0.0, 4.0]
    QC_XLIM        = None
    QC_YLIM        = None
    QC_ZLIM        = None
    QC_OCEAN_COLOR = "lightgrey"
    QC_DPI         = 200


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
    (2) Draw log10(rho) values at pilot points ~ Uniform(min, max).
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
QC slice plots of Kriged initial models
----------------------------------------
One figure per selected ensemble member saved as gst_qc<PLOT_STR>.pdf in
that member's subdirectory.  Uses fviz.plot_model_slices (exact
tetrahedron-plane intersection, model-local metres, no geographic conversion).
Controlled by PLOT_SLICES_QC / QC_* config vars in the Visualization block.
"""
if (PLOT_DATA or PLOT_MODEL) and PLOT_SLICES_QC:
    _qc_files = [
        ENSEMBLE_DIR + ENSEMBLE_NAME + f"{i}/{MOD_RESISTIVITY_FILE}"
        for i in range(N_SAMPLES)
    ]
    for i_samp in VIZ_SAMPLES:
        _qc_path = ENSEMBLE_DIR + ENSEMBLE_NAME + f"{i_samp}/"
        _qc_file = _qc_path + "gst_qc" + PLOT_STR + ".pdf"
        if not os.path.isfile(_qc_files[i_samp]):
            print(f"  QC: {_qc_files[i_samp]} not found — skipped.")
            continue
        fviz.plot_model_slices(
            model_file  = _qc_files[i_samp],
            mesh_file   = MOD_MESH,
            slices      = QC_SLICES,
            cmap        = QC_CMAP,
            clim        = QC_CLIM,
            xlim        = QC_XLIM,
            ylim        = QC_YLIM,
            zlim        = QC_ZLIM,
            ocean_color = QC_OCEAN_COLOR,
            ocean_value = 0.25,
            depth_km    = True,
            horiz_km    = True,
            plot_file   = _qc_file,
            dpi         = QC_DPI,
            out         = OUT,
        )
        print(f"  QC slice plot saved: {_qc_file}")
    print("QC slice plots done.")


"""
Model visualization
-------------------
Per-sample plot of reference vs. Kriged initial model across 1-5
user-defined slices (map or curtain).  One figure per selected ensemble
member; each figure is saved into that member's own subdirectory as
``gst_model<PLOT_STR>.pdf``.

Helper: femtic_viz.plot_model_ensemble
"""
if PLOT_MODEL and PERTURB_MOD:
    MOD_ORIG = MOD_REF

    mod_ens_files = [
        ENSEMBLE_DIR + ENSEMBLE_NAME + f"{i}/{MOD_RESISTIVITY_FILE}"
        for i in range(N_SAMPLES)
    ]

    for i_samp in VIZ_SAMPLES:
        fig_mod, axs_mod = fviz.plot_model_ensemble(
            orig_mod_file=MOD_ORIG,
            ens_mod_files=mod_ens_files,
            mesh_file=MOD_MESH,
            sample_indices=[i_samp],
            slices=MOD_SLICES,
            mode=MOD_MODE,
            log10=MOD_LOG10,
            cmap=MOD_CMAP,
            clim=MOD_CLIM,
            xlim=MOD_XLIM,
            ylim=MOD_YLIM,
            zlim=MOD_ZLIM,
            mesh_lines=MOD_MESH_LINES,
            mesh_lw=MOD_MESH_LW,
            mesh_color=MOD_MESH_COLOR,
            out=True,
        )
        member_dir = ENSEMBLE_DIR + ENSEMBLE_NAME + f"{i_samp}/"
        plot_path = member_dir + "gst_model" + PLOT_STR + ".pdf"
        fig_mod.savefig(plot_path, bbox_inches="tight")
        plt.close(fig_mod)
        print(f"  model plot saved: {plot_path}")
    print("model ensemble plots saved.")

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
