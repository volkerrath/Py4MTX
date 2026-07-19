#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
modem_nss.py — Nullspace Shuttle for ModEM inversion results
==============================================================

Generalisation of ``femtic_nss.py`` to ModEM's structured rectilinear grid.
The null-space-shuttle mathematics (steps 2, 3, 5) are identical; what
differs is I/O (ModEM Jacobian archive + ``.rho`` model instead of a FEMTIC
HDF5 archive) and the geostatistical perturbation machine (step 4), which
now runs directly on ModEM grid cell centres via
``ensembles.generate_gst_perturbation_modem`` instead of FEMTIC's
unstructured free-region barycentres.

Workflow
--------
    (1) Read the final model (``modem.read_mod``) and the processed
        Jacobian archive (``<JFile>_jac.npz`` + ``<JFile>_info.npz``, as
        produced by the py4mt ``jac_proc`` pipeline — same files consumed
        by ``modem_jac_svd.py``).
    (2) Form the (data-)scaled Jacobian Js.  See "Jacobian scaling" below —
        by default this script assumes the archived Jacobian is *already*
        error-scaled (matching the usage in ``modem_jac_svd.py``, which
        feeds ``Jac.T`` directly into the randomised SVD with no further
        reweighting).  Set ``JAC_ALREADY_SCALED = False`` and supply
        ``Dat["Scale"]`` if that is not the case for your archive.
    (3) Randomised SVD of Js via ``jac_proc.rsvd`` (Halko et al., 2011) —
        the same routine and calling convention as ``modem_jac_svd.py``.
    (4) Model perturbation.  ``PERTURB_MODE = "gst"`` (default) draws a
        geostatistical perturbation via pilot-point Ordinary Kriging
        directly on the ModEM rectilinear grid
        (``ensembles.generate_gst_perturbation_modem``) — no mesh file, no
        FEMTIC free-region concept, no temporary directory / ensemble I/O
        round-trip.  ``PERTURB_MODE = "random"`` retains a Gaussian
        placeholder.
    (5) Nullspace shuttle: project the perturbation onto the null space of
        Js and add it to the final model.  Identical maths to
        ``femtic_nss.py``.

Theory (brief)
--------------
The null-space N(Js) is spanned by the right singular vectors of Js whose
singular values are zero (or below the threshold ``NSS_SV_THRESH``).  Given
any model perturbation δm̃, the null-space component is::

    δm_null = (I - Vr @ Vr.T) @ δm̃       (*)

where Vr = Vt[:r_eff].T contains the top-r_eff right singular vectors.
Adding δm_null to the current model produces a new model with (to within
the truncation rank) identical predicted data.

Jacobian scaling
----------------
The FEMTIC pipeline (``femtic_nss.py``) reads ``observed``, ``calculated``,
and ``errors`` from an HDF5 archive and explicitly forms
``Js = diag(1/error) @ J`` at run time (step 2).  The ModEM ``jac_proc``
archive used by ``modem_jac_svd.py`` instead stores a single processed
Jacobian (``<JFile>_jac.npz``) alongside per-datum metadata
(``Freq, Comp, Site, DTyp, Data, Scale, Info`` in ``<JFile>_info.npz``);
``modem_jac_svd.py`` uses ``Jac.T`` directly, which suggests the archived
Jacobian is already data-scaled upstream in ``jac_proc``.  This script
follows that convention by default (``JAC_ALREADY_SCALED = True``).
**Verify this against your own ``jac_proc`` version** — if the archived
Jacobian is raw (unscaled), set ``JAC_ALREADY_SCALED = False``; this script
will then form ``Js = Scale[:, None] * Jac`` from ``Dat["Scale"]`` before
the SVD.  Confirm the sign/units convention of ``Scale`` (multiplicative
weight vs. divisive error) against ``jac_proc`` before trusting the
verification norm printed in step 5.

Model / Jacobian column correspondence
---------------------------------------
FEMTIC's Jacobian columns correspond one-to-one with *free regions* (air
and ocean already excluded upstream).  It is not yet established in this
codebase whether the ModEM archived Jacobian's ``nm`` columns correspond
to (a) the full grid ``nx*ny*nz``, or (b) free (non-air) cells only, as
identified by ``aircells``.  This script checks both possibilities against
``Jac.shape[1]`` at runtime (see step 1) and raises a clear error if
neither matches, rather than silently guessing.

Literature
----------
[1] M. Deal, G. Nolet (1996) "Nullspace shuttles", Geophysical Journal
    International, 124, 372-380, doi:10.1111/j.1365-246X.1996.tb07027.x

[2] G. Munoz, V. Rath (2006) "Beyond smooth inversion: the use of
    nullspace projection for the exploration of non-uniqueness in MT",
    Geophysical Journal International, 164, 301-311,
    doi:10.1111/j.1365-246X.2005.02825.x

[3] N. Halko, P.-G. Martinsson, J. A. Tropp (2011) "Finding structure with
    randomness: Probabilistic algorithms for constructing approximate
    matrix decompositions", SIAM Review, 53(2), 217-288,
    doi:10.1137/090771806

Provenance
----------
    2026-07-19  vrath / Claude Sonnet 5 (Anthropic)
                Created, generalising femtic_nss.py to ModEM's structured
                rectilinear grid. Jacobian I/O modelled on
                modem_jac_svd.py (jac_proc.rsvd, scs.load_npz archive
                layout). Step-4 GST perturbation now calls
                ensembles.generate_gst_perturbation_modem, the new
                mesh-agnostic perturbation machine added to ensembles.py
                on the same date, instead of FEMTIC's directory-based
                generate_gst_model_ensemble. Nullspace-shuttle maths
                (steps 2 core / 5) unchanged from femtic_nss.py.
                *** ASSUMPTIONS requiring verification against jac_proc.py
                and modem.py are flagged inline with "ASSUMPTION:" — see
                also modem_nss_readme.md, section "Assumptions to verify". ***

@author: vrath
"""

import os
import sys
import inspect
import time

import numpy as np
import scipy.sparse as scs

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
import modem as mod
import jac_proc as jac
import ensembles as ens

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
WORK_DIR = r"/home/vrath/ModEM_work/Ub25_ZT_600_PT_jac/"

#: Base name for the processed Jacobian archive: expects
#: ``<JFile>_jac.npz`` (sparse Jacobian, scipy.sparse) and
#: ``<JFile>_info.npz`` (per-datum metadata: Freq, Comp, Site, DTyp, Data,
#: Scale, Info) — same layout consumed by modem_jac_svd.py.
JFile = WORK_DIR + "Ub25_ZPT_nerr_sp-6"

#: Final ModEM resistivity model (without extension), read via
#: modem.read_mod(MFile, ".rho", trans="LOGE").  Also used as the
#: reference model for the GST perturbation (PERTURB_MODE = "gst").
MFile = WORK_DIR + "Ub_600ZT4_PT_NLCG_009"

#: Output resistivity model (nullspace-shuttled).
MODEL_OUT = WORK_DIR + "Ub_600ZT4_PT_NLCG_009_nss"

#: RHOAIR sentinel — must match the value used when MFile was written.
RHOAIR = 1.0e17

# ---------------------------------------------------------------------------
# Jacobian scaling  (step 2) — see "Jacobian scaling" in the module
# docstring above.  Verify against your jac_proc version before trusting
# the ||Js @ dm_null|| verification norm printed in step 5.
# ---------------------------------------------------------------------------
JAC_ALREADY_SCALED = True

# ---------------------------------------------------------------------------
# Randomised SVD parameters  (step 3) — same knobs as modem_jac_svd.py.
# ---------------------------------------------------------------------------
#: Target rank for the rSVD of Js.  Should be << min(nd, nm).  Increase
#: until the singular-value spectrum flattens.
RSVD_RANK = 300

#: Oversampling factor, multiplied by RSVD_RANK (matches modem_jac_svd.py's
#: ``n_oversamples = noversmp * nsingval`` convention).
RSVD_OVERSAMPLE_FACTOR = 2

#: Number of subspace (power) iterations.
RSVD_SUBSPACE_ITERS = 2

# ---------------------------------------------------------------------------
# Nullspace shuttle parameters  (step 5) — identical role to femtic_nss.py.
# ---------------------------------------------------------------------------
#: Singular-value threshold (fraction of s[0]) below which a right
#: singular vector is treated as belonging to the null space.
NSS_SV_THRESH = 1.0e-3

#: Amplitude scale applied to the null-space perturbation.  Start small
#: (e.g. 0.1) and increase to explore the null space more aggressively.
NSS_AMPLITUDE = 1.0

# ---------------------------------------------------------------------------
# Verbose output
# ---------------------------------------------------------------------------
OUT = True

# ===========================================================================
# Step 4 configuration — model perturbation mode
# ===========================================================================

#: "random" — uniform Gaussian placeholder.
#: "gst"    — geostatistical perturbation via pilot-point Kriging, applied
#:            directly to the ModEM rectilinear grid
#:            (ensembles.generate_gst_perturbation_modem).
PERTURB_MODE = "gst"

if PERTURB_MODE == "gst":
    # --- Pilot-point placement ---------------------------------------------
    # GST_PP_MODE: "random" | "fixed" | "mixed" | "extrema"
    GST_PP_MODE = "random"

    #: Number of random pilot points per realisation.
    GST_N_PP = 100

    #: Bounding box for random pilot-point placement, in the *model-local*
    #: coordinate system used by modem_gst_cell_centers (x, y from
    #: cumulative dx, dy starting at 0; z = depth, positive-down, from
    #: cumulative dz starting at 0):
    #:   [x_min, x_max, y_min, y_max, z_min, z_max]  (metres)
    #: ASSUMPTION: set this to bracket your model's free-cell extent —
    #: check against dx.sum(), dy.sum(), dz.sum() from the read-in model.
    GST_PP_BBOX = [0., 100000., 0., 100000., 0., 80000.]

    GST_PP_COORDS = None    # required for "fixed" / "mixed"
    GST_PP_ROI = None       # sub-volume for "extrema" mode; None = full extent
    GST_PP_EXTREMA_K = 30
    GST_PP_EXTREMA_WHICH = "both"

    # --- Resistivity range (log10 Ohm.m — see "Unit convention" below) ----
    GST_LOG_RHO_MIN = 0.0
    GST_LOG_RHO_MAX = 4.0

    # --- Pilot-point value mode --------------------------------------------
    GST_PP_VALUE_MODE = "uniform"   # "uniform" | "reference"
    GST_PP_VALUE_DELTA = 0.5

    # --- Variogram -----------------------------------------------------------
    GST_VARIO_MODEL = "Spherical"
    GST_VARIO_RANGE = (8000., 4000.)   # (horizontal, vertical) metres
    GST_VARIO_SILL = 0.5               # (log10 Ohm.m)^2
    GST_VARIO_NUGGET = 0.01
    GST_VARIO_ANGLES = None

# Unit convention:
#   modem.read_mod(..., trans="LOGE") returns the model in natural-log
#   resistivity, whereas the GST defaults above (and the FEMTIC GST
#   tooling this is modelled on) are expressed in log10(Ohm.m). This
#   script Krigs in log10 space and converts the resulting *delta* back to
#   loge before adding it to the model (see step 4 below) — the loge model
#   values themselves are never Kriged directly.
LOG10 = np.log(10.0)


# ===========================================================================
# Step 1 — Read model and Jacobian archive
# ===========================================================================

print("=" * 72)
print("Step 1: Reading model and Jacobian archive")
print("=" * 72)

t0_total = time.perf_counter()
t0 = time.perf_counter()

dx, dy, dz, model_grid, refmod, _ = mod.read_mod(MFile, ".rho", trans="LOGE")
nx, ny, nz = model_grid.shape
aircells = model_grid > np.log(RHOAIR / 10.0)
free_mask = ~aircells
n_free = int(free_mask.sum())

Jac = scs.load_npz(JFile + "_jac.npz")
Dat = np.load(JFile + "_info.npz", allow_pickle=True)

nd, nm_jac = Jac.shape

# ASSUMPTION: Jacobian columns correspond to either the full grid or the
# free (non-air) cells only.  Check both; fail loudly if neither matches
# rather than silently mis-indexing.
if nm_jac == nx * ny * nz:
    jac_index_mode = "full_grid"
elif nm_jac == n_free:
    jac_index_mode = "free_cells"
else:
    raise ValueError(
        f"modem_nss.py: Jacobian has {nm_jac} columns, but the model grid "
        f"has {nx * ny * nz} cells total ({n_free} free / non-air). "
        "Cannot determine the model<->Jacobian column correspondence -- "
        "check jac_proc's column ordering (and whether it excludes air "
        "cells) and adjust this script's Step 1 accordingly."
    )

if OUT:
    print(f"  model grid  : nx={nx}, ny={ny}, nz={nz}  ({nx*ny*nz} cells, "
          f"{n_free} free / {nx*ny*nz - n_free} air)")
    print(f"  Jacobian    : {Jac.shape}  (nd={nd}, nm={nm_jac})")
    print(f"  column mode : '{jac_index_mode}'")
    print(f"  elapsed     : {time.perf_counter() - t0:.2f} s")


# ===========================================================================
# Step 2 — Scaled Jacobian
# ===========================================================================

print("\n" + "=" * 72)
print("Step 2: Forming the (data-)scaled Jacobian Js")
print("=" * 72)

t0 = time.perf_counter()

if JAC_ALREADY_SCALED:
    Js = Jac
    if OUT:
        print("  JAC_ALREADY_SCALED = True -> using the archived Jacobian "
              "as-is (matches modem_jac_svd.py usage).")
else:
    # ASSUMPTION: Dat["Scale"] is a per-datum multiplicative weight
    # (e.g. 1/error) with shape (nd,), applied row-wise to Jac.  Verify the
    # sign/units convention against jac_proc before trusting downstream
    # results.
    Scale = np.asarray(Dat["Scale"]).reshape(-1)
    if Scale.shape[0] != nd:
        raise ValueError(
            f"modem_nss.py: Dat['Scale'] has shape {Scale.shape}, expected "
            f"({nd},) to match Jac's {nd} rows."
        )
    Js = scs.diags(Scale) @ Jac
    if OUT:
        print("  JAC_ALREADY_SCALED = False -> Js = diag(Scale) @ Jac.")

if OUT:
    print(f"  Js shape : {Js.shape}")
    print(f"  elapsed  : {time.perf_counter() - t0:.2f} s")


# ===========================================================================
# Step 3 — Randomised SVD of Js
# ===========================================================================

print("\n" + "=" * 72)
print("Step 3: Randomised SVD of Js")
print("=" * 72)

t0 = time.perf_counter()

rank = min(RSVD_RANK, nd, nm_jac)

# Matches modem_jac_svd.py's calling convention: jac.rsvd(Jac.T, ...).
U, S, Vt = jac.rsvd(
    Js.T,
    rank=rank,
    n_oversamples=RSVD_OVERSAMPLE_FACTOR * rank,
    n_subspace_iters=RSVD_SUBSPACE_ITERS,
)

# U  : (nm, rank)
# S  : (rank,)
# Vt : (rank, nd)
#
# NOTE: because jac.rsvd was called on Js.T (not Js), the roles of U/Vt are
# swapped relative to femtic_nss.py's inv.rsvd(Js, ...) call: here U spans
# model space and Vt spans data space.  The null-space projector for model
# space therefore uses **U**, not Vt -- see step 5.

if OUT:
    print(f"  Decomposition: U {U.shape}, S {S.shape}, Vt {Vt.shape}")
    print(f"  s[0]  = {S[0]:.4e}  (largest)")
    print(f"  s[-1] = {S[-1]:.4e}  (smallest in truncated set)")
    s_thresh = NSS_SV_THRESH * S[0]
    r_eff = int(np.sum(S >= s_thresh))
    print(f"  Effective rank at threshold {NSS_SV_THRESH:.1e}: {r_eff} / {rank}")
    print(f"  elapsed  : {time.perf_counter() - t0:.2f} s")


# ===========================================================================
# Step 4 — Model perturbation
# ===========================================================================

def _make_perturbation_random(m: np.ndarray) -> np.ndarray:
    """Gaussian placeholder perturbation, log-resistivity units of ``m``.

    Parameters
    ----------
    m : numpy.ndarray, shape (n_free,)
        Current free-cell model values (loge, as read).  Not used in the
        default implementation but available for amplitude scaling.

    Returns
    -------
    dm : numpy.ndarray, shape (n_free,)
    """
    # -------------------------------------------------------------------
    # *** EDIT BELOW THIS LINE to replace the Gaussian placeholder ***
    # -------------------------------------------------------------------
    rng_local = np.random.default_rng(seed=0)
    dm = rng_local.standard_normal(m.size)
    # -------------------------------------------------------------------
    # *** EDIT ABOVE THIS LINE ***
    # -------------------------------------------------------------------
    return dm


def _make_perturbation_gst(model_loge: np.ndarray) -> np.ndarray:
    """Geostatistical perturbation via pilot-point Kriging on the ModEM grid.

    Calls ``ensembles.generate_gst_perturbation_modem`` directly on the
    rectilinear grid cell centres derived from ``dx, dy, dz`` — no mesh
    file, no FEMTIC free-region concept, no temporary directory.

    Parameters
    ----------
    model_loge : numpy.ndarray, shape (nx, ny, nz)
        Reference model in natural-log resistivity (as read by
        ``modem.read_mod``).

    Returns
    -------
    dm : numpy.ndarray, shape (nx, ny, nz)
        Perturbation in natural-log resistivity (air cells are exactly
        zero; only free cells carry a nonzero delta).
    """
    ref_log10 = model_loge / LOG10

    field_log10 = ens.generate_gst_perturbation_modem(
        dx, dy, dz, ref_log10,
        aircells=aircells,
        pp_mode=GST_PP_MODE,
        n_pp=GST_N_PP,
        pp_bbox=GST_PP_BBOX,
        pp_coords=GST_PP_COORDS,
        pp_roi=GST_PP_ROI,
        pp_extrema_k=GST_PP_EXTREMA_K,
        pp_extrema_which=GST_PP_EXTREMA_WHICH,
        log_rho_min=GST_LOG_RHO_MIN,
        log_rho_max=GST_LOG_RHO_MAX,
        pp_value_mode=GST_PP_VALUE_MODE,
        pp_value_delta=GST_PP_VALUE_DELTA,
        vario_model=GST_VARIO_MODEL,
        vario_range=GST_VARIO_RANGE,
        vario_sill=GST_VARIO_SILL,
        vario_nugget=GST_VARIO_NUGGET,
        vario_angles=GST_VARIO_ANGLES,
        rng=np.random.default_rng(),
        out=OUT,
    )

    dm_log10 = field_log10 - ref_log10
    dm_loge = dm_log10 * LOG10
    dm_loge[aircells] = 0.0
    return dm_loge


print("\n" + "=" * 72)
print(f"Step 4: Model perturbation  [PERTURB_MODE = '{PERTURB_MODE}']")
print("=" * 72)

t0 = time.perf_counter()

if PERTURB_MODE == "gst":
    dm_grid = _make_perturbation_gst(model_grid)          # shape (nx, ny, nz)
    dm_free = dm_grid[free_mask]                           # shape (n_free,)
elif PERTURB_MODE == "random":
    dm_free = _make_perturbation_random(model_grid[free_mask])
else:
    raise ValueError(f"Unknown PERTURB_MODE: '{PERTURB_MODE}'. "
                      "Choose 'random' or 'gst'.")

# Map the free-cell perturbation onto the Jacobian's column indexing.
if jac_index_mode == "free_cells":
    dm_raw = dm_free
else:  # "full_grid"
    dm_raw = np.zeros(nx * ny * nz)
    dm_raw[free_mask.ravel(order="C")] = dm_free

if OUT:
    print(f"  ||dm_raw||  = {np.linalg.norm(dm_raw):.4e}")
    print(f"  dm_raw range: [{dm_raw.min():.4f}, {dm_raw.max():.4f}]  (loge)")
    print(f"  elapsed     : {time.perf_counter() - t0:.2f} s")


# ===========================================================================
# Step 5 — Nullspace shuttle
# ===========================================================================

def _nullspace_shuttle(dm, Ur, S, *, sv_thresh=1.0e-3, amplitude=1.0):
    """Project a model perturbation onto the null space of Js.

    Parameters
    ----------
    dm : numpy.ndarray, shape (nm,)
        Raw model perturbation.
    Ur : numpy.ndarray, shape (nm, rank)
        Model-space singular vectors from ``jac.rsvd(Js.T, ...)`` (note:
        this is ``U``, not ``Vt`` -- see the note at the end of step 3).
    S : numpy.ndarray, shape (rank,)
        Singular values.
    sv_thresh : float
        Fraction of S[0] below which a singular vector is treated as null.
    amplitude : float
        Scale factor applied to the null-space perturbation.

    Returns
    -------
    dm_null : numpy.ndarray, shape (nm,)
    r_eff : int
    """
    s_thresh = sv_thresh * S[0]
    r_eff = int(np.sum(S >= s_thresh))

    Ur_eff = Ur[:, :r_eff]           # (nm, r_eff) -- row-space basis in model space
    dm_row = Ur_eff @ (Ur_eff.T @ dm)
    dm_null = dm - dm_row

    return amplitude * dm_null, r_eff


print("\n" + "=" * 72)
print("Step 5: Nullspace shuttle")
print("=" * 72)

t0 = time.perf_counter()

dm_null, r_eff = _nullspace_shuttle(
    dm_raw, U, S, sv_thresh=NSS_SV_THRESH, amplitude=NSS_AMPLITUDE,
)

if OUT:
    print(f"  Effective rank used for projection : {r_eff}")
    print(f"  ||dm_null||  = {np.linalg.norm(dm_null):.4e}")
    dy_check = Js @ dm_null
    print(f"  ||Js @ dm_null|| (should be ~0) = {np.linalg.norm(dy_check):.4e}")
    print(f"  elapsed         : {time.perf_counter() - t0:.2f} s")


# ===========================================================================
# Write nullspace-shuttled model
# ===========================================================================

print("\n" + "=" * 72)
print("Writing nullspace-shuttled model")
print("=" * 72)

t0 = time.perf_counter()

model_flat = model_grid.ravel(order="C").copy()
if jac_index_mode == "free_cells":
    idx_free = np.flatnonzero(free_mask.ravel(order="C"))
    model_flat[idx_free] += dm_null
else:
    model_flat += dm_null

model_nss = model_flat.reshape(nx, ny, nz, order="C")
model_nss[aircells] = np.log(RHOAIR)

mod.write_mod(
    MODEL_OUT, modext=".rho", trans="LOGE",
    dx=dx, dy=dy, dz=dz, mval=model_nss,
    reference=refmod, mvalair=1e17, aircells=aircells, header="",
)

if OUT:
    print(f"  model_nss range : [{model_nss[free_mask].min():.4f}, "
          f"{model_nss[free_mask].max():.4f}]  (loge, free cells)")
    print(f"  Written : {MODEL_OUT}.rho")
    print(f"  elapsed : {time.perf_counter() - t0:.2f} s")

print(f"\n  Total elapsed : {time.perf_counter() - t0_total:.2f} s")
