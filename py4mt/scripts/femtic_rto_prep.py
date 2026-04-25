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
"""

import os
import sys
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

mypath = [PY4MTX_ROOT + "/py4mt/modules/", PY4MTX_ROOT + "/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)

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

"""
Base setup.
"""
N_SAMPLES = 32
ENSEMBLE_DIR = r"/home/vrath/Py4MTX/py4mt/data/rto/ubinas/ensemble/"
# TEMPLATES = ENSEMBLE_DIR + "templates/"
TEMPLATES = r"/home/vrath/Py4MTX/py4mt/data/rto/ubinas/templates/"
COPY_LIST = ["observe.dat",
             "referencemodel.dat",]
LINK_LIST = ["control.dat",
             "mesh.dat",
             "resistivity_block_iter0.dat",
             "distortion_iter0.dat",
             "run_femtic_dias.sh",]
# "run_femtic_kraken.sh"]

ENSEMBLE_NAME = "ubinas_rto_"

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
    MOD_PDF = ["normal", 0., 0.5]
    # ["exp", L], ["gauss", L], ["matern", L, MatPars], ["femtic"], None
    R_FILE = r"/home/vrath/Py4MTX/py4mt/data/rto/ubinas/R_coo"

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
PLOT_DATA = True
PLOT_MODEL = True

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
# MOD_ORIG is derived from MOD_REF below (after the PERTURB_MOD block)

# Define 1-5 slices.  Each dict must have "type": "map" or "curtain"
# plus the keyword arguments forwarded to the femtic_viz slicer.
MOD_MODE = "tri"        # "tri" | "scatter" | "grid"
MOD_LOG10 = True
MOD_CMAP = "jet_r"
# (vmin, vmax) in log10(Ohm.m); None = auto from original
MOD_CLIM = None
# Axis limits for map and curtain slices.
# Map slices:    MOD_XLIM = easting range (m),  MOD_YLIM = northing range (m).
# Curtain slices: MOD_YLIM = along-profile distance range (m),
#                 MOD_ZLIM = depth range (m, negative-down, e.g. (-5000, 0)).
# Set any to None for Matplotlib auto-scaling.
# Individual slice dicts may carry 'xlim'/'ylim'/'zlim' keys to override per-slice.
MOD_XLIM = None         # (xmin, xmax) in metres; None = auto
MOD_YLIM = None         # (ymin, ymax) in metres; None = auto
MOD_ZLIM = None         # (zmin, zmax) in metres; None = auto
MOD_MESH_LINES = False  # overlay triangulation edges on filled patches
MOD_MESH_LW = 0.3       # mesh edge line width (points)
MOD_MESH_COLOR = "k"    # mesh edge colour
MOD_SLICES = [
    {"type": "map", "z0": 5000, "dz": 50},
    {"type": "map", "z0": -5000, "dz": 50},
    {"type": "curtain",
     "polyline": np.array([[0., 0.], [10000., 0.]]),
     "width": 500},
]


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
generate_model_ensemble, never explicitly materialised.
"""
R = scs.load_npz(R_FILE + ".npz")
print("roughness R loaded with shape:", np.shape(R))

model_ensemble = ens.generate_model_ensemble(
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
Model visualization
-------------------
Per-sample plot of original vs. perturbed resistivity models, shown across
1-5 user-defined slices (map or curtain).  One figure per selected ensemble
member; each figure is saved into that member's own subdirectory as
``rto_model<PLOT_STR>.pdf``.

Helper: femtic_viz.plot_model_ensemble
"""
if PLOT_MODEL:
    MOD_ORIG = MOD_REF  # MOD_REF is already the full path

    mod_ens_files = [
        ENSEMBLE_DIR + ENSEMBLE_NAME + f"{i}/{MOD_REF_BASE}"
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
        plot_path = member_dir + "rto_model" + PLOT_STR + ".pdf"
        fig_mod.savefig(plot_path, bbox_inches="tight")
        plt.close(fig_mod)
        print(f"  model plot saved: {plot_path}")
    print("model ensemble plots saved.")
