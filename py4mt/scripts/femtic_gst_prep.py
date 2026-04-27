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
"""

import os
import sys
import numpy as np
import inspect
import matplotlib.pyplot as plt

"""
Py4MTX-specific settings and imports.
"""
PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [PY4MTX_ROOT + "/py4mt/modules/", PY4MTX_ROOT + "/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)

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

"""
Base setup.
"""
N_SAMPLES = 32
ENSEMBLE_DIR = r"/home/vrath/Py4MTX/py4mt/data/gst/ubinas/ensemble/"

TEMPLATES = ENSEMBLE_DIR + "/templates/"
if not os.path.isdir(TEMPLATES):
    sys.exit(" Directory: %s does not exist, needs to be copied !" % TEMPLATES)

COPY_LIST = ["observe.dat",
             "referencemodel.dat",]
LINK_LIST = ["control.dat",
             "mesh.dat",
             "distortion_iter0.dat",
             "run_femtic_dias.sh", "run_femtic_kraken.sh"]

ENSEMBLE_NAME = "ubinas_gst_"

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

    # ------------------------------------------------------------------
    # Pilot-point configuration
    # ------------------------------------------------------------------
    # MOD_PP_MODE controls how pilot-point locations are determined:
    #   "random"  : MOD_N_PP points drawn uniformly inside MOD_PP_BBOX.
    #   "fixed"   : use the explicit list in MOD_PP_COORDS (Nx3 array,
    #               columns = easting, northing, depth [m]).
    #   "mixed"   : start with MOD_PP_COORDS and add MOD_N_PP random points.
    MOD_PP_MODE = "random"     # "random" | "fixed" | "mixed"

    # Number of randomly drawn pilot points per member.
    # Used when MOD_PP_MODE = "random" or "mixed".
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
    MOD_VARIO_RANGE   = (20000., 5000.) # (horizontal, vertical) ranges in m
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
PERTURB_DAT = True
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
        pp_mode=MOD_PP_MODE,
        n_pp=MOD_N_PP,
        pp_bbox=MOD_PP_BBOX,
        pp_coords=MOD_PP_COORDS,
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
