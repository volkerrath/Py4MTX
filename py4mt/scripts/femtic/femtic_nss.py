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

Provenance
----------
    2026-05-17  vrath / Claude Sonnet 4.6   Created, modelled on
                femtic_mod_edit.py.  Uses ``inverse.rsvd`` for the randomised
                SVD and a local ``_nullspace_shuttle`` helper for step (5).

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
# Step 4 — Model modification (placeholder)
# ===========================================================================

def _modify_model(m: np.ndarray) -> np.ndarray:
    """Apply a model perturbation before nullspace projection.

    Replace the body of this function with any prior-based, geological, or
    exploratory modification in log10(ρ) space.  The perturbation returned
    here is projected onto the null space of Js in step 5, so it will not
    alter the predicted data.

    Parameters
    ----------
    m : numpy.ndarray, shape (nm,)
        Current final model in log10(ρ) (free regions only).

    Returns
    -------
    dm : numpy.ndarray, shape (nm,)
        Desired model perturbation (same shape as m).  The null-space
        shuttle will zero out any data-sensitive component before adding
        this to the model.

    Notes
    -----
    Examples of what to put here:

    - Smooth / sharpen a feature::

          dm = np.zeros_like(m)
          dm[region_indices] = 1.0   # push target region towards +1 log unit

    - Apply a large-scale bias::

          dm = np.full_like(m, 0.5)  # shift everything +0.5 log units

    - Draw from a random perturbation::

          rng = np.random.default_rng(seed=42)
          dm = rng.standard_normal(m.size) * 0.2

    The amplitude is subsequently scaled by NSS_AMPLITUDE in step 5.
    """
    # -----------------------------------------------------------------------
    # *** EDIT BELOW THIS LINE ***
    # -----------------------------------------------------------------------

    # Default placeholder: uniform random perturbation (unit Gaussian)
    rng = np.random.default_rng(seed=0)
    dm = rng.standard_normal(m.size)

    # -----------------------------------------------------------------------
    # *** EDIT ABOVE THIS LINE ***
    # -----------------------------------------------------------------------
    return dm


print("\n" + "=" * 72)
print("Step 4: Model modification")
print("=" * 72)

t0 = time.perf_counter()
dm_raw = _modify_model(model)

if OUT:
    print(f"  ||dm_raw||  = {np.linalg.norm(dm_raw):.4e}")
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
print(f"\n  Total elapsed : {time.perf_counter() - t0_total:.2f} s")
