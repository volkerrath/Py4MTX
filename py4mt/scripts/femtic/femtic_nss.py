#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_nss.py — Nullspace Shuttle for FEMTIC inversion results
==============================================================

Workflow
--------
    (1) Read final model and data from HDF5 file ``Inversion_results.h5``.
        Datasets read: ``model``, ``observed``, ``calculated``, ``errors``,
        ``jacobian``.
    (2) Compute the scaled (data-weighted) Jacobian:
            Js = diag(1/error) @ J
        and the normalised residual:
            rs = (observed - calculated) / error
    (3) Compute the randomised SVD of Js  (Halko et al., 2011).
        Singular values and vectors are used to define the data-space and
        null-space projectors.
    (4) **Model-modification placeholder** — edit the ``_modify_model``
        function to inject prior geological knowledge, perturb the model,
        or construct a starting ensemble before shuttling.
    (5) Nullspace shuttle: project the model perturbation onto the null-space
        of Js so that it cannot change the predicted data, then add it to the
        final model.

Theory (brief)
--------------
The null-space N(Js) is spanned by the right singular vectors of Js whose
singular values are zero (or below the threshold ``NSS_SV_THRESH``).  Given
any model perturbation δm̃, the null-space component is::

    δm_null = (I - Vr @ Vr.T) @ δm̃       (*)

where Vr = Vt[:rank].T contains the top-rank right singular vectors.  Adding
δm_null to the current model produces a new model with identical predicted
data (to within the truncation rank).

Literature
----------

[1] M. Deal, G. Nolet (1996) “Nullspace shuttles", Geophysical Journal International, 124, 372–380,doi:10.1111/j.1365-246X.1996.tb07027.x

[2] G. Muñoz, V. Rath (2006) “Beyond smooth inversion: the use of nullspace projection for the exploration of non-uniqueness in MT", Geophysical Journal International, 164, 301–311, 2006, doi:10.1111/j.1365-246X.2005.02825.x



Provenance
----------
    2026-05-17  vrath / Claude Sonnet 4.6   Created, modelled on
                femtic_mod_edit.py.  Uses ``inverse.rsvd`` for the randomised
                SVD and a local ``_nullspace_shuttle`` helper for step (5).
    2026-06-23  vrath / Claude Sonnet 4.6   Merged GST model-generation from
                femtic_gst_prep.py into step 4.  PERTURB_MODE = "gst" draws
                a geostatistically perturbed model via pilot-point Ordinary
                Kriging (ensembles.generate_gst_model_ensemble), computes the
                delta w.r.t. the reference model, and passes it to the null-
                space shuttle.  PERTURB_MODE = "random" retains the original
                uniform-Gaussian placeholder.  GST config block mirrors
                femtic_gst_prep.py exactly (pp_mode, variogram, etc.).
    2026-07-09  vrath / Claude Sonnet 5 (Anthropic)
                Added the shared MOD_* plotting config block (MOD_MESH,
                MOD_OCEAN/AIR_RHO/OCEAN_RHO, MOD_UTM_ORIGIN_*, MOD_ORIGIN_
                METHOD, MOD_DISPLAY_COORDS, MOD_SITE_*, MOD_SLICES,
                MOD_XLIM/YLIM/ZLIM, MOD_CMAP/CLIM, MOD_OCEAN_COLOR/
                AIR_COLOR/AIR_BGCOLOR, MOD_ALPHA_*, MOD_EQUAL_ASPECT/
                DEPTH_KM/HORIZ_KM/NROWS/NCOLS/PANEL_HEIGHT/PANEL_WIDTH/
                FIGSIZE) plus MOD_QC/MOD_QC_FILE/MOD_DPI, identical in
                name and order to femtic_ens_post.py and femtic_gst_prep.py.
                Added femtic_viz import, _resolve_origin_and_sites() and
                _plot_slice() helpers (mirroring femtic_ens_post.py), and an
                optional QC slice plot of MODEL_OUT at the end of the run.
                A plotting config block can now be copied between all three
                scripts with no renaming. Uses a single MOD_DPI knob (no
                per-plot-type split), matching femtic_gst_prep.py and the
                now-simplified femtic_ens_post.py.

@author: vrath
"""

import os
import sys
import inspect

import time

import numpy as np
import h5py

# ---------------------------------------------------------------------------
# Py4MTX-specific settings and imports
# ---------------------------------------------------------------------------
PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [PY4MTX_ROOT + "/py4mt/modules/", PY4MTX_ROOT + "/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)

from version import versionstrg
import util as utl
import femtic as fem
import inverse as inv
import ensembles as ens

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
WORK_DIR = r"/home/vrath/Py4MTX/work/"

#: HDF5 file produced by the final FEMTIC inversion step.
HDF5_FILE = WORK_DIR + "Inversion_results.h5"

#: Output resistivity block (nullspace-modified model).
MODEL_OUT = WORK_DIR + "resistivity_block_nss.dat"

#: Template resistivity block for header / flag / fixed-region metadata.
#: Only the free-region values are replaced; all other columns are preserved.
MODEL_TEMPLATE = WORK_DIR + "resistivity_block_iter0.dat"

# ---------------------------------------------------------------------------
# Randomised SVD parameters  (step 3)
# ---------------------------------------------------------------------------
#: Target rank for the rSVD decomposition of Js.
#: Should be << min(nd, nm).  Increase until the singular-value spectrum
#: flattens to capture all significant data information content.
RSVD_RANK = 300

#: Oversampling parameter (None → 2 × RSVD_RANK as in Halko et al.).
RSVD_OVERSAMPLES = None

#: Number of subspace (power) iterations.  More iterations → more accurate
#: decomposition at the cost of extra matrix-vector products.
RSVD_SUBSPACE_ITERS = 2

# ---------------------------------------------------------------------------
# Nullspace shuttle parameters  (step 5)
# ---------------------------------------------------------------------------
#: Singular-value threshold below which a right singular vector is treated as
#: belonging to the null space.  Expressed as a fraction of the largest
#: singular value s[0].
NSS_SV_THRESH = 1.0e-3

#: Amplitude scale applied to the null-space perturbation before adding it to
#: the final model.  Start small (e.g. 0.1) and increase to explore the null
#: space more aggressively.
NSS_AMPLITUDE = 1.0

# ---------------------------------------------------------------------------
# Verbose output
# ---------------------------------------------------------------------------
OUT = True

# ---------------------------------------------------------------------------
# Mesh (required for the QC slice plot)
# ---------------------------------------------------------------------------
MOD_MESH = WORK_DIR + "mesh.dat"

# --- Ocean / air handling (must match the inversion setup) ----------------
MOD_OCEAN     = None
MOD_AIR_RHO   = 1.0e9   # Ω·m  (region 0)
MOD_OCEAN_RHO = 0.25    # Ω·m  (region 1 when treated as ocean)

# ---------------------------------------------------------------------------
# QC slice plot — the nullspace-shuttled output model
# ---------------------------------------------------------------------------
#: Set True to plot MODEL_OUT after it is written.
MOD_QC      = False
MOD_QC_FILE = WORK_DIR + "nss_qc.pdf"

# ---------------------------------------------------------------------------
# Shared slice / plot parameters
# (identical config surface to femtic_gst_prep.py / femtic_rto_prep.py /
#  femtic_ens_post.py — used by MOD_QC below)
# ---------------------------------------------------------------------------

# --- Geographic / UTM origin of the mesh centre ----------------------------
#: Set to None when MOD_ORIGIN_METHOD will estimate the origin from MOD_SITE_DAT.
MOD_UTM_ORIGIN_LAT    = None   # decimal degrees, positive = North
MOD_UTM_ORIGIN_LON    = None   # decimal degrees, positive = East
MOD_UTM_ORIGIN_E      = None   # UTM easting  [m]
MOD_UTM_ORIGIN_N      = None   # UTM northing [m]
MOD_UTM_ZONE_OVERRIDE = None   # override auto-derived zone; None = auto

#: "box"     → midpoint of UTM bounding box of all sites in MOD_SITE_DAT.
#: "average" → arithmetic mean of UTM coordinates in MOD_SITE_DAT.
#: None      → use the hard-coded literals above.
MOD_ORIGIN_METHOD = "box"

# --- Display coordinate system ---------------------------------------------
#: "model"  — axis ticks in model-local metres (default)
#: "utm"    — axis ticks in absolute UTM metres
#: "latlon" — axis ticks in decimal degrees
MOD_DISPLAY_COORDS = "model"

# --- Site overlay ------------------------------------------------------------
#: Primary source: mt_make_sitelist.py CSV (name,lat,lon,elev,sitenum,E,N).
#: Set to None to fall back to observe.dat / MOD_SITE_NUMBER.
MOD_SITE_DAT    = WORK_DIR + "site.dat"
MOD_SITE_NAMES  = None   # list of names to plot, or None = all sites
#: Fallback: site number(s) from observe.dat (int or list of ints).
MOD_SITE_NUMBER = None

MOD_PLOT_SITES_MAPS   = True    # show markers on map panels
MOD_PLOT_SITES_SLICES = False   # show markers on curtain / plane panels
#: Max distance [m] from a curtain plane for a site to appear on it.
MOD_PROJECTION_DIST = 5000.0    # metres; None = show all sites on every panel

MOD_SITE_MARKER        = dict(marker="v", color="black", ms=8, zorder=10, label=None)
MOD_SITE_MARKER_SLICES = None
#: Extra point markers on map panels only (each dict: latlon, marker, color, ms, name).
MOD_MAP_MARKERS = []

# --- Slice specification ----------------------------------------------------
#: Slice positions accept plain floats (model-local m) or CRS-tagged tuples:
#:   (value, "utm") | (value, "latlon")
#: Depth z0 is always model-local metres (no CRS tagging).
MOD_SLICES = [
    dict(kind="map", z0=5000.0),
    dict(kind="map", z0=15000.0),
    dict(kind="ns",  x0=0.0),
    dict(kind="ew",  y0=0.0),
]
MOD_XLIM = None    # [xmin, xmax] model-local metres; None = auto
MOD_YLIM = None    # [ymin, ymax] model-local metres; None = auto
MOD_ZLIM = None    # [zmin, zmax] model-local metres; None = auto

MOD_DPI         = 600            # figure DPI
MOD_CMAP        = "turbo_r"
MOD_CLIM        = [0.0, 4.0]     # [log10_min, log10_max] Ω·m; None = auto
MOD_OCEAN_COLOR = "lightgrey"    # flat colour for ocean cells; None = colormap
MOD_AIR_COLOR   = "whitesmoke"
MOD_AIR_BGCOLOR = None           # axes facecolor for air; None = figure default

# --- Alpha / blanking by second block file (optional) -----------------------
MOD_ALPHA_FILE         = None    # path to sensitivity block; None = disabled
MOD_ALPHA_MODE         = "fade"  # "fade" | "blank"
MOD_ALPHA_BLANK_THRESH = 0.0

# --- Figure layout -----------------------------------------------------------
MOD_EQUAL_ASPECT = True
MOD_DEPTH_KM     = True
MOD_HORIZ_KM     = True
MOD_NROWS        = None   # None = auto (1 row)
MOD_NCOLS        = None   # None = auto (len(MOD_SLICES) cols)
MOD_PANEL_HEIGHT = 16.0   # cm
MOD_PANEL_WIDTH  = None   # cm; None = auto from aspect ratio
MOD_FIGSIZE      = None   # [w, h] cm; overrides auto when set

# ===========================================================================
# Step 4 configuration — model perturbation mode
# ===========================================================================

#: ``"random"`` — uniform Gaussian placeholder (original behaviour).
#: ``"gst"``    — geostatistical perturbation via pilot-point Kriging.
PERTURB_MODE = "gst"

# ---------------------------------------------------------------------------
# GST model perturbation config  (used only when PERTURB_MODE == "gst")
# Mirrors the config block in femtic_gst_prep.py exactly.
# ---------------------------------------------------------------------------
if PERTURB_MODE == "gst":
    # Reference model and mesh (same files used for NSS template).
    GST_REF_MOD  = MODEL_TEMPLATE   # free-region structure source
    GST_MESH     = WORK_DIR + "mesh.dat"

    # --- Pilot-point placement -----------------------------------------------
    # MOD_PP_MODE: "random" | "fixed" | "mixed" | "extrema"
    #   "random"  — n_pp points drawn uniformly inside MOD_PP_BBOX each call.
    #   "fixed"   — locations from MOD_PP_COORDS (same every call, values vary).
    #   "mixed"   — MOD_PP_COORDS plus n_pp random fill points.
    #   "extrema" — seed at local log10(ρ) extrema of reference model (within
    #               MOD_PP_ROI), plus n_pp random fill points.
    GST_PP_MODE = "random"      # "random" | "fixed" | "mixed" | "extrema"

    # Number of random pilot points per shuttle realisation.
    # Recommended: 50–200 for typical 3-D MT survey volumes.
    GST_N_PP = 100

    # Bounding box for random pilot-point placement:
    #   [x_min, x_max, y_min, y_max, z_min, z_max]  (metres, z positive-down)
    GST_PP_BBOX = [-50000., 50000.,   # easting  range (m)
                   -50000., 50000.,   # northing range (m)
                        0., 80000.]   # depth     range (m)

    # Explicit pilot-point coordinates for "fixed" or "mixed" mode.
    # Shape: (N, 3) — columns: [easting, northing, depth].
    GST_PP_COORDS = None   # e.g. np.array([[x, y, z], ...])

    # --- "extrema" mode ------------------------------------------------------
    # GST_PP_ROI: sub-volume [x_min,x_max,y_min,y_max,z_min,z_max] restricting
    #   which free regions are eligible as extremum seeds.  None = full extent.
    GST_PP_ROI           = None   # None = full extent
    GST_PP_EXTREMA_K     = 9      # neighbourhood size for extremum detection
    GST_PP_EXTREMA_WHICH = "both" # "both" | "minima" | "maxima"

    # --- Resistivity range ---------------------------------------------------
    # Pilot-point values drawn Uniform(GST_LOG_RHO_MIN, GST_LOG_RHO_MAX) in
    # log10(Ω·m).  Post-Kriging field clamped to the same interval.
    GST_LOG_RHO_MIN = 0.0    # log10(1 Ω·m)
    GST_LOG_RHO_MAX = 4.0    # log10(10 000 Ω·m)

    # --- Variogram model -----------------------------------------------------
    # GST_VARIO_MODEL: gstools covariance model class name (string).
    #   Common choices: "Spherical", "Gaussian", "Exponential", "Matern".
    #
    # GST_VARIO_RANGE: correlation length (m).  A 2-tuple
    #   (horizontal_range, vertical_range) sets geometric anisotropy.
    #   Recommended: h_range ≈ half survey aperture; v_range ≈ half target depth.
    #
    # GST_VARIO_SILL: sill (variance) in (log10 Ω·m)².  Typical 0.25–0.5.
    #
    # GST_VARIO_NUGGET: nugget in (log10 Ω·m)².  Keep ≤ 10 % of sill.
    #
    # GST_VARIO_ANGLES: rotation [α, β, γ] in degrees; None = axis-aligned.
    GST_VARIO_MODEL   = "Spherical"
    GST_VARIO_RANGE   = (8000., 4000.)  # (horizontal, vertical) in m
    GST_VARIO_SILL    = 0.5
    GST_VARIO_NUGGET  = 0.01
    GST_VARIO_ANGLES  = None   # [alpha, beta, gamma] deg; None = axis-aligned


# ===========================================================================
# Plotting helpers
# ===========================================================================

def _resolve_origin_and_sites():
    """Estimate UTM origin from MOD_SITE_DAT; collect site model-local coords.

    Mirrors the origin-resolution block in femtic_gst_prep.py /
    femtic_ens_post.py so all scripts behave identically, including the
    observe.dat / MOD_SITE_NUMBER fallback when MOD_SITE_DAT is absent.
    """
    _e   = MOD_UTM_ORIGIN_E
    _n   = MOD_UTM_ORIGIN_N
    _lat = MOD_UTM_ORIGIN_LAT
    _lon = MOD_UTM_ORIGIN_LON
    _zone, _north = None, None

    if MOD_ORIGIN_METHOD is not None and MOD_SITE_DAT and os.path.isfile(MOD_SITE_DAT):
        _sdat = fem.read_site_dat(MOD_SITE_DAT)
        if _sdat:
            _Es  = np.array([d["easting"]  for d in _sdat])
            _Ns  = np.array([d["northing"] for d in _sdat])
            if MOD_ORIGIN_METHOD == "box":
                _e = 0.5 * (_Es.min() + _Es.max())
                _n = 0.5 * (_Ns.min() + _Ns.max())
            elif MOD_ORIGIN_METHOD == "average":
                _e = float(_Es.mean())
                _n = float(_Ns.mean())
            _lats = np.array([d["lat"] for d in _sdat])
            _lons = np.array([d["lon"] for d in _sdat])
            _zone, _north = utl.utm_zone_from_latlon(
                float(_lats.mean()), float(_lons.mean()),
                override=MOD_UTM_ZONE_OVERRIDE,
            )
            _lat, _lon = utl.utm_to_latlon_zn(_e, _n, _zone, _north)

    if _lat is not None and _lon is not None:
        _zone, _north = utl.utm_zone_from_latlon(
            _lat, _lon, override=MOD_UTM_ZONE_OVERRIDE
        )

    site_xys = []
    obs_coords_only = False
    _need_sites = MOD_PLOT_SITES_MAPS or MOD_PLOT_SITES_SLICES
    if _need_sites and MOD_SITE_DAT and os.path.isfile(MOD_SITE_DAT):
        for row in fem.read_site_dat(MOD_SITE_DAT, site_names=MOD_SITE_NAMES):
            sx, sy = fem.utm_to_model(
                row["easting"], row["northing"], _e, _n
            )
            site_xys.append(
                (row["name"], sx, sy, float(row.get("elev", 0.0)))
            )
    elif _need_sites and MOD_SITE_NUMBER is not None:
        _obs_file = WORK_DIR + "observe.dat"
        _site_nums = (MOD_SITE_NUMBER if isinstance(MOD_SITE_NUMBER, (list, tuple))
                      else [MOD_SITE_NUMBER])
        for _sn in _site_nums:
            sx, sy = fem.read_site_position(_obs_file, _sn)
            site_xys.append((_sn, sx, sy, 0.0))
        obs_coords_only = True

    return _e, _n, _lat, _lon, _zone, _north, site_xys, obs_coords_only


def _plot_slice(block_file: str, pdf_file: str,
                utm_e, utm_n, utm_lat, utm_lon,
                utm_zone, utm_north, site_xys: list,
                obs_coords_only: bool = False) -> None:
    """Call fviz.plot_model_slices with the shared MOD_* config.

    Mirrors the plotting call in femtic_gst_prep.py / femtic_ens_post.py
    exactly, so the NSS QC figure uses the same options (CRS handling,
    site overlay, alpha/blanking, figure layout) as the other scripts.
    """
    if fviz is None:
        print("  plot_slice: femtic_viz not available -- skipping.")
        return

    _slices_resolved = fem.resolve_slice_positions(
        MOD_SLICES, utm_zone, utm_north,
        utm_e, utm_n, utm_lat, utm_lon,
        verbose=OUT,
    )
    fviz.plot_model_slices(
        model_file          = block_file,
        mesh_file           = MOD_MESH,
        slices              = _slices_resolved,
        cmap                = MOD_CMAP,
        clim                = MOD_CLIM,
        xlim                = MOD_XLIM,
        ylim                = MOD_YLIM,
        zlim                = MOD_ZLIM,
        ocean_color         = MOD_OCEAN_COLOR,
        ocean_value         = MOD_OCEAN_RHO,
        air_color           = MOD_AIR_COLOR,
        air_bgcolor         = MOD_AIR_BGCOLOR,
        site_xys            = site_xys,
        obs_coords_only     = obs_coords_only,
        sites_in_maps       = MOD_PLOT_SITES_MAPS,
        sites_in_slices     = MOD_PLOT_SITES_SLICES,
        site_marker         = MOD_SITE_MARKER,
        site_marker_slices  = MOD_SITE_MARKER_SLICES,
        map_markers         = MOD_MAP_MARKERS,
        projection_dist     = MOD_PROJECTION_DIST,
        display_coords      = MOD_DISPLAY_COORDS,
        utm_origin_e        = utm_e,
        utm_origin_n        = utm_n,
        utm_zone            = utm_zone,
        utm_northern        = utm_north,
        utm_to_latlon_fn    = utl.utm_to_latlon_zn,
        latlon_to_model_fn  = fem.latlon_to_model,
        depth_km            = MOD_DEPTH_KM,
        horiz_km            = MOD_HORIZ_KM,
        equal_aspect        = MOD_EQUAL_ASPECT,
        panel_height        = MOD_PANEL_HEIGHT / 2.54,
        panel_width         = MOD_PANEL_WIDTH / 2.54 if MOD_PANEL_WIDTH is not None else None,
        figsize             = [v / 2.54 for v in MOD_FIGSIZE] if MOD_FIGSIZE is not None else None,
        nrows               = MOD_NROWS,
        ncols               = MOD_NCOLS,
        alpha_file          = MOD_ALPHA_FILE,
        alpha_mode          = MOD_ALPHA_MODE,
        alpha_blank_thresh  = MOD_ALPHA_BLANK_THRESH,
        plot_file           = pdf_file,
        dpi                 = MOD_DPI,
        out                 = OUT,
    )
    if OUT:
        print(f"  saved -> {pdf_file}")


# ===========================================================================
# Step 1 — Read HDF5 inversion results
# ===========================================================================

print("=" * 72)
print("Step 1: Reading inversion results from HDF5")
print("=" * 72)

t0_total = time.perf_counter()
t0 = time.perf_counter()

with h5py.File(HDF5_FILE, "r") as hf:
    model      = hf["model"][:]       # shape (nm,)  — log10(ρ) or raw ρ
    observed   = hf["observed"][:]    # shape (nd,)
    calculated = hf["calculated"][:]  # shape (nd,)
    errors     = hf["errors"][:]      # shape (nd,)  — positive data errors
    jacobian   = hf["jacobian"][:]    # shape (nd, nm)

nm = model.shape[0]
nd = observed.shape[0]

if OUT:
    print(f"  model      : {model.shape}")
    print(f"  observed   : {observed.shape}")
    print(f"  calculated : {calculated.shape}")
    print(f"  errors     : {errors.shape}")
    print(f"  jacobian   : {jacobian.shape}")
    print(f"  nd={nd}, nm={nm}")
    print(f"  elapsed    : {time.perf_counter() - t0:.2f} s")


# ===========================================================================
# Step 2 — Scaled Jacobian and normalised residual
# ===========================================================================

print("\n" + "=" * 72)
print("Step 2: Computing scaled Jacobian Js = diag(1/error) @ J")
print("=" * 72)

t0 = time.perf_counter()

inv_err = 1.0 / errors                        # shape (nd,)
Js = inv_err[:, np.newaxis] * jacobian        # shape (nd, nm)  — broadcast
rs = (observed - calculated) * inv_err        # shape (nd,)     — weighted residual

if OUT:
    print(f"  Js shape : {Js.shape}")
    print(f"  ||rs||   : {np.linalg.norm(rs):.4f}")
    print(f"  RMS      : {np.sqrt(np.mean(rs**2)):.4f}")
    print(f"  elapsed  : {time.perf_counter() - t0:.2f} s")


# ===========================================================================
# Step 3 — Randomised SVD of Js
# ===========================================================================

print("\n" + "=" * 72)
print("Step 3: Randomised SVD of Js")
print("=" * 72)

t0 = time.perf_counter()

rank = min(RSVD_RANK, nd, nm)

U, S, Vt = inv.rsvd(
    Js,
    rank=rank,
    n_oversamples=RSVD_OVERSAMPLES,
    n_subspace_iters=RSVD_SUBSPACE_ITERS,
)

# U  : (nd, rank)  — left singular vectors  (data space)
# S  : (rank,)     — singular values
# Vt : (rank, nm)  — right singular vectors transposed (model space)

if OUT:
    print(f"  Decomposition: U {U.shape}, S {S.shape}, Vt {Vt.shape}")
    print(f"  s[0]  = {S[0]:.4e}  (largest)")
    print(f"  s[-1] = {S[-1]:.4e}  (smallest in truncated set)")
    # Determine effective rank at the chosen threshold
    s_thresh = NSS_SV_THRESH * S[0]
    r_eff = int(np.sum(S >= s_thresh))
    print(f"  Effective rank at threshold {NSS_SV_THRESH:.1e}: {r_eff} / {rank}")
    print(f"  elapsed  : {time.perf_counter() - t0:.2f} s")


# ===========================================================================
# Step 4 — Model perturbation
# ===========================================================================
#
# Two modes controlled by PERTURB_MODE (set in the configuration section):
#
#   "random" — uniform Gaussian placeholder (original behaviour).
#              Replace the body of _make_perturbation_random() with any
#              prior-based or exploratory modification in log10(ρ) space.
#              The amplitude is subsequently scaled by NSS_AMPLITUDE in step 5.
#
#   "gst"    — geostatistical perturbation via pilot-point Ordinary Kriging
#              (ens.generate_gst_model_ensemble).  A single Kriged model is
#              generated in a temporary directory; the perturbation is the
#              delta between that model and the reference model in log10(ρ)
#              space.  The null-space shuttle then projects out any data-
#              sensitive component so the predicted data remain unchanged.

def _make_perturbation_random(m: np.ndarray) -> np.ndarray:
    """Return a random Gaussian perturbation in log10(ρ) space (placeholder).

    Parameters
    ----------
    m : numpy.ndarray, shape (nm,)
        Current final model in log10(ρ) (free regions only).  Not used in the
        default implementation but available for amplitude scaling.

    Returns
    -------
    dm : numpy.ndarray, shape (nm,)
        Desired model perturbation (same shape as m).
    """
    # -----------------------------------------------------------------------
    # *** EDIT BELOW THIS LINE to replace the Gaussian placeholder ***
    # -----------------------------------------------------------------------

    rng_local = np.random.default_rng(seed=0)
    dm = rng_local.standard_normal(m.size)

    # -----------------------------------------------------------------------
    # *** EDIT ABOVE THIS LINE ***
    # -----------------------------------------------------------------------
    return dm


def _make_perturbation_gst(m_ref: np.ndarray) -> np.ndarray:
    """Generate a geostatistical model perturbation via pilot-point Kriging.

    Calls ``ens.generate_gst_model_ensemble`` for a single realisation in a
    temporary subdirectory inside WORK_DIR.  The perturbation is the difference
    between the Kriged model and the reference model in log10(ρ) space::

        dm = m_gst - m_ref

    The null-space shuttle in step 5 projects out any data-sensitive component.

    Parameters
    ----------
    m_ref : numpy.ndarray, shape (nm,)
        Reference model in log10(ρ) (free regions only), as read from HDF5.

    Returns
    -------
    dm : numpy.ndarray, shape (nm,)
        Perturbation in log10(ρ) space (same shape as m_ref).
    """
    import tempfile, shutil

    tmp_base = os.path.join(WORK_DIR, "_nss_gst_tmp_")

    try:
        # generate_gst_model_ensemble writes member 0 into ``tmp_base + "0/"``
        ens.generate_gst_model_ensemble(
            alg              = "gst",
            dir_base         = tmp_base,
            n_samples        = 1,
            fromto           = None,
            ref_mod_file     = GST_REF_MOD,
            mesh_file        = GST_MESH,
            pp_mode          = GST_PP_MODE,
            n_pp             = GST_N_PP,
            pp_bbox          = GST_PP_BBOX,
            pp_coords        = GST_PP_COORDS,
            pp_roi           = GST_PP_ROI,
            pp_extrema_k     = GST_PP_EXTREMA_K,
            pp_extrema_which = GST_PP_EXTREMA_WHICH,
            log_rho_min      = GST_LOG_RHO_MIN,
            log_rho_max      = GST_LOG_RHO_MAX,
            vario_model      = GST_VARIO_MODEL,
            vario_range      = GST_VARIO_RANGE,
            vario_sill       = GST_VARIO_SILL,
            vario_nugget     = GST_VARIO_NUGGET,
            vario_angles     = GST_VARIO_ANGLES,
            output_target    = "resistivity_block",
            resistivity_file = "resistivity_block_iter0.dat",
            reference_file   = "referencemodel.dat",
            rng              = np.random.default_rng(),
            out              = OUT,
        )

        # Read back the Kriged initial model from the temporary member directory
        gst_block_file = tmp_base + "0/resistivity_block_iter0.dat"
        gst_block = fem.read_model(gst_block_file)
        # extract_free_values returns the free-region log10(ρ) vector
        m_gst = fem.extract_model(gst_block)

    finally:
        # Always clean up the temporary tree
        if os.path.isdir(tmp_base + "0"):
            shutil.rmtree(tmp_base + "0", ignore_errors=True)

    return m_gst - m_ref


print("\n" + "=" * 72)
print(f"Step 4: Model perturbation  [PERTURB_MODE = '{PERTURB_MODE}']")
print("=" * 72)

t0 = time.perf_counter()

if PERTURB_MODE == "gst":
    dm_raw = _make_perturbation_gst(model)
elif PERTURB_MODE == "random":
    dm_raw = _make_perturbation_random(model)
else:
    raise ValueError(f"Unknown PERTURB_MODE: '{PERTURB_MODE}'. "
                     "Choose 'random' or 'gst'.")

if OUT:
    print(f"  ||dm_raw||  = {np.linalg.norm(dm_raw):.4e}")
    print(f"  dm_raw range: [{dm_raw.min():.3f}, {dm_raw.max():.3f}]")
    print(f"  elapsed     : {time.perf_counter() - t0:.2f} s")


# ===========================================================================
# Step 5 — Nullspace shuttle
# ===========================================================================

def _nullspace_shuttle(
    dm: np.ndarray,
    Vt: np.ndarray,
    S: np.ndarray,
    *,
    sv_thresh: float = 1.0e-3,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Project a model perturbation onto the null space of the scaled Jacobian.

    Given the truncated SVD  Js ≈ U S Vt, the data-space projector onto the
    row space of Js is  Vr @ Vr.T  where Vr = Vt[r_eff].T.  The null-space
    projector is  I - Vr @ Vr.T.

    The shuttle perturbation is::

        δm_null = amplitude * (I - Vr @ Vr.T) @ dm

    Adding δm_null to the current model produces a model with (approximately)
    identical predicted data.

    Parameters
    ----------
    dm : numpy.ndarray, shape (nm,)
        Raw model perturbation from ``_modify_model``.
    Vt : numpy.ndarray, shape (rank, nm)
        Right singular vectors (transposed) from the rSVD of Js.
    S : numpy.ndarray, shape (rank,)
        Singular values from the rSVD.
    sv_thresh : float
        Fraction of s[0] below which a singular vector is treated as null.
    amplitude : float
        Scale factor applied to the null-space perturbation.

    Returns
    -------
    dm_null : numpy.ndarray, shape (nm,)
        Null-space component of dm scaled by amplitude.
    """
    s_thresh = sv_thresh * S[0]
    r_eff = int(np.sum(S >= s_thresh))

    Vr = Vt[:r_eff].T             # shape (nm, r_eff) — row-space basis

    # Null-space projection: remove row-space component
    dm_row  = Vr @ (Vr.T @ dm)    # component in row space  (data-sensitive)
    dm_null = dm - dm_row          # component in null space (data-invisible)

    return amplitude * dm_null, r_eff


print("\n" + "=" * 72)
print("Step 5: Nullspace shuttle")
print("=" * 72)

t0 = time.perf_counter()

dm_null, r_eff = _nullspace_shuttle(
    dm_raw,
    Vt,
    S,
    sv_thresh=NSS_SV_THRESH,
    amplitude=NSS_AMPLITUDE,
)

model_nss = model + dm_null

if OUT:
    print(f"  Effective rank used for projection : {r_eff}")
    print(f"  ||dm_null||  = {np.linalg.norm(dm_null):.4e}")
    print(f"  ||dm_row ||  = {np.linalg.norm(dm_raw - dm_null / NSS_AMPLITUDE):.4e}")

    # Verification: predicted-data change should be negligible
    dy = Js @ dm_null
    print(f"  ||Js @ dm_null|| (should be ~0) = {np.linalg.norm(dy):.4e}")
    print(f"  model_nss range : [{model_nss.min():.3f}, {model_nss.max():.3f}]")
    print(f"  elapsed         : {time.perf_counter() - t0:.2f} s")


# ===========================================================================
# Write modified model
# ===========================================================================

print("\n" + "=" * 72)
print("Writing nullspace-shuttled model")
print("=" * 72)

t0 = time.perf_counter()

# ``fem.insert_model`` merges the free-region vector back into the template
# structure preserving header, bounds, fixed regions, air, and ocean.
model_block = fem.read_model(MODEL_TEMPLATE)
model_block_nss = fem.insert_model(model_block, model_nss)
fem.write_model(MODEL_OUT, model_block_nss)

print(f"  Written : {MODEL_OUT}")
print(f"  elapsed : {time.perf_counter() - t0:.2f} s")

# ===========================================================================
# QC slice plot of the nullspace-shuttled model
# ===========================================================================

if MOD_QC:
    if fviz is None:
        print("\n  MOD_QC: femtic_viz not available -- skipping.")
    else:
        print("\n" + "=" * 72)
        print("QC slice plot of nullspace-shuttled model")
        print("=" * 72)
        (utm_e, utm_n, utm_lat, utm_lon,
         utm_zone, utm_north, site_xys, obs_coords_only) = _resolve_origin_and_sites()
        _plot_slice(
            block_file      = MODEL_OUT,
            pdf_file        = MOD_QC_FILE,
            utm_e           = utm_e,
            utm_n           = utm_n,
            utm_lat         = utm_lat,
            utm_lon         = utm_lon,
            utm_zone        = utm_zone,
            utm_north       = utm_north,
            site_xys        = site_xys,
            obs_coords_only = obs_coords_only,
        )

print(f"\n  Total elapsed : {time.perf_counter() - t0_total:.2f} s")
