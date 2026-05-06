#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
modem_compress.py

Apply spectral / basis-function compression to a ModEM resistivity model,
reconstruct, optionally write the result, and report accuracy statistics.

One method block is active at a time (set METHOD below).  All other blocks
are present but commented out so they can be switched in without rewriting.

Methods implemented
-------------------
  'dct'        3-D DCT-II, radial (L2-wavenumber) truncation
  'dct_sep'    3-D DCT-II, separable (box) truncation
  'wavelet'    3-D discrete wavelet transform, hard thresholding
  'legdct'     Legendre-z x DCT-xy separable mixed basis
  'bspdct'     B-spline-z x DCT-xy separable mixed basis
  'kl'         Karhunen-Loève / PCA from a model ensemble

@author: vrath
Written: April 2026
"""

import os
import sys
import inspect
import time

import numpy as np

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [PY4MTX_ROOT + "/py4mt/modules/", PY4MTX_ROOT + "/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)

import modem as mod
import util as utl
from version import versionstrg

version, _ = versionstrg()
titstrng = utl.print_title(
    version=version, fname=inspect.getfile(inspect.currentframe()), out=False
)
print(titstrng + "\n\n")

# =============================================================================
#  Global configuration
# =============================================================================

RHOAIR = 1.0e17

MOD_FILE_IN  = PY4MTX_DATA + "/Peru/1_feb_ell/TAC_100"
MOD_FILE_OUT = MOD_FILE_IN + "_compressed"

# Which compression method to run.
# One of: 'dct', 'dct_sep', 'wavelet', 'legdct', 'bspdct', 'kl'
METHOD = "dct"

# Write the reconstructed model to a ModEM .rho file?
WRITE_OUTPUT = True

# Write compressed coefficients + metadata to a .npz archive?
# The archive contains everything needed to reconstruct the model
# without re-running the compression.
WRITE_NPZ = True

# Run the truncation sweep (accuracy vs compression ratio) before compressing?
RUN_ANALYSIS = True

# =============================================================================
#  Method-specific parameters
#  Edit the block that matches METHOD; all others are ignored.
# =============================================================================

# -----------------------------------------------------------------------------
#  DCT — radial (L2-wavenumber) truncation
#  Specify exactly one of FRAC_KEEP, N_KEEP, or KMAX.
# -----------------------------------------------------------------------------
DCT = {
    "frac_keep": 0.05,   # fraction of total coefficients to retain
    "n_keep":    None,   # explicit count (overrides frac_keep if set)
    "kmax":      None,   # maximum L2 wavenumber (alternative to count/frac)
    "n_levels":  20,     # levels for truncation analysis
}

# -----------------------------------------------------------------------------
#  DCT separable — box truncation, independent compression per axis
# -----------------------------------------------------------------------------
DCT_SEP = {
    "frac_x": 0.4,    # fraction of nx DCT coefficients to retain
    "frac_y": 0.4,    # fraction of ny DCT coefficients to retain
    "frac_z": 0.4,    # fraction of nz DCT coefficients to retain
    # alternatively set explicit counts:
    "nx_keep": None,
    "ny_keep": None,
    "nz_keep": None,
}

# -----------------------------------------------------------------------------
#  Wavelet — 3-D DWT with hard thresholding
#  Requires:  pip install PyWavelets
#  Specify exactly one of FRAC_KEEP, N_KEEP, or THRESH.
# -----------------------------------------------------------------------------
WAVELET = {
    "wavelet":   "db4",  # 'db4', 'sym4', 'coif2', …
    "level":     None,   # decomposition depth (None = maximum)
    "frac_keep": 0.05,   # fraction of coefficients to keep
    "n_keep":    None,
    "thresh":    None,   # hard amplitude threshold (alternative)
    "n_levels":  20,
}

# -----------------------------------------------------------------------------
#  Legendre-z x DCT-xy
#  frac_leg controls depth compression; frac_dct controls horizontal.
# -----------------------------------------------------------------------------
LEGDCT = {
    "frac_leg":  0.4,    # fraction of nz cells used as Legendre orders
    "n_leg":     None,   # explicit count (overrides frac_leg if set)
    "frac_dct":  0.4,    # fraction of nx/ny cells used as DCT coefficients
    "nx_dct":    None,
    "ny_dct":    None,
    "n_levels":  20,
}

# -----------------------------------------------------------------------------
#  B-spline-z x DCT-xy
#  knot_style='quantile' places knots at depth quantiles of the cell centres,
#  concentrating basis functions near the surface where cells are finest.
# -----------------------------------------------------------------------------
BSPDCT = {
    "frac_basis":  0.4,        # fraction of nz cells used as B-spline basis fns
    "n_basis":     None,       # explicit count (overrides frac_basis if set)
    "k":           3,          # spline degree (3 = cubic)
    "knot_style":  "quantile", # 'quantile', 'uniform', or 'log'
    "frac_dct":    0.4,
    "nx_dct":      None,
    "ny_dct":      None,
    "n_levels":    20,
}

# -----------------------------------------------------------------------------
#  KL / PCA — ensemble-based Karhunen-Loève
#  Requires a set of existing model files to build the ensemble.
#  svd_method='auto' uses randomized SVD when n_modes << n_models (fast),
#  otherwise falls back to the full economy SVD.
# -----------------------------------------------------------------------------
KL = {
    "ensemble_files": [         # list of .rho files (without extension)
        PY4MTX_DATA + "/Peru/1_feb_ell/TAC_100",
        PY4MTX_DATA + "/Peru/1_feb_ell/TAC_200",
        PY4MTX_DATA + "/Peru/1_feb_ell/TAC_300",
    ],
    "n_modes":       20,        # KL modes to retain (None = all)
    "frac_modes":    None,      # fraction of available modes (alternative)
    "svd_method":    "auto",    # 'auto', 'exact', 'randomized', 'truncated'
    "n_oversamples": 10,        # randomized SVD oversampling
    "n_power_iter":  4,         # randomized SVD power iterations
    "random_state":  None,
    "n_levels":      20,
}

# =============================================================================
#  Read model
# =============================================================================

total = 0.0
start = time.perf_counter()

dx, dy, dz, rho, refmod, _ = mod.read_mod(MOD_FILE_IN, ".rho", trans="LOGE")
aircells = np.where(rho > np.log(RHOAIR / 10.0))

elapsed = time.perf_counter() - start
total += elapsed
print(" Used %7.4f s for reading model from %s"
      % (elapsed, MOD_FILE_IN + ".rho"))

# =============================================================================
#  Compression
# =============================================================================

rho_rec = None   # reconstructed model — filled by whichever block runs

# -----------------------------------------------------------------------------
#  DCT — radial truncation
# -----------------------------------------------------------------------------
if METHOD == "dct":

    if RUN_ANALYSIS:
        print("\n Running DCT truncation analysis ...")
        mod.dct_truncation_analysis(rho, n_levels=DCT["n_levels"])
        elapsed = time.perf_counter() - start
        print(" Used %7.4f s for DCT truncation analysis" % elapsed)

    print("\n Compressing with DCT ...")
    rho_rec, coeff, keep_mask = mod.dct_compress(
        rho,
        n_keep    = DCT["n_keep"],
        frac_keep = DCT["frac_keep"],
        kmax      = DCT["kmax"],
    )
    elapsed = time.perf_counter() - start
    total += elapsed
    print(" Used %7.4f s for DCT compression" % elapsed)

# -----------------------------------------------------------------------------
#  DCT separable — box truncation
# -----------------------------------------------------------------------------
elif METHOD == "dct_sep":

    nx, ny, nz = rho.shape
    nx_keep = DCT_SEP["nx_keep"] or max(1, int(round(DCT_SEP["frac_x"] * nx)))
    ny_keep = DCT_SEP["ny_keep"] or max(1, int(round(DCT_SEP["frac_y"] * ny)))
    nz_keep = DCT_SEP["nz_keep"] or max(1, int(round(DCT_SEP["frac_z"] * nz)))

    print("\n Compressing with separable DCT ...")
    coeff_block, shape_full, shape_keep = mod.model_to_dct_separable(
        rho,
        nx_keep = nx_keep,
        ny_keep = ny_keep,
        nz_keep = nz_keep,
    )
    rho_rec = mod.dct_separable_to_model(coeff_block, shape_full)

    elapsed = time.perf_counter() - start
    total += elapsed
    print(" Used %7.4f s for separable DCT compression" % elapsed)

# -----------------------------------------------------------------------------
#  Wavelet
# -----------------------------------------------------------------------------
elif METHOD == "wavelet":

    if RUN_ANALYSIS:
        print("\n Running wavelet truncation analysis ...")
        mod.wavelet_truncation_analysis(
            rho,
            wavelet  = WAVELET["wavelet"],
            n_levels = WAVELET["n_levels"],
        )
        elapsed = time.perf_counter() - start
        print(" Used %7.4f s for wavelet truncation analysis" % elapsed)

    print("\n Compressing with wavelet (%s) ..." % WAVELET["wavelet"])
    rho_rec, coeffs, n_kept = mod.wavelet_compress(
        rho,
        wavelet   = WAVELET["wavelet"],
        level     = WAVELET["level"],
        n_keep    = WAVELET["n_keep"],
        frac_keep = WAVELET["frac_keep"],
        thresh    = WAVELET["thresh"],
    )
    elapsed = time.perf_counter() - start
    total += elapsed
    print(" Used %7.4f s for wavelet compression" % elapsed)

# -----------------------------------------------------------------------------
#  Legendre-z x DCT-xy
# -----------------------------------------------------------------------------
elif METHOD == "legdct":

    if RUN_ANALYSIS:
        print("\n Running Legendre-z x DCT-xy truncation analysis ...")
        mod.legdct_truncation_analysis(rho, n_levels=LEGDCT["n_levels"])
        elapsed = time.perf_counter() - start
        print(" Used %7.4f s for Legendre-z x DCT-xy truncation analysis"
              % elapsed)

    print("\n Compressing with Legendre-z x DCT-xy ...")
    rho_rec, C, params = mod.legdct_compress(
        rho,
        n_leg   = LEGDCT["n_leg"],
        frac_leg= LEGDCT["frac_leg"],
        nx_dct  = LEGDCT["nx_dct"],
        ny_dct  = LEGDCT["ny_dct"],
        frac_dct= LEGDCT["frac_dct"],
    )
    elapsed = time.perf_counter() - start
    total += elapsed
    print(" Used %7.4f s for Legendre-z x DCT-xy compression" % elapsed)

# -----------------------------------------------------------------------------
#  B-spline-z x DCT-xy
# -----------------------------------------------------------------------------
elif METHOD == "bspdct":

    if RUN_ANALYSIS:
        print("\n Running B-spline-z x DCT-xy truncation analysis ...")
        mod.bspdct_truncation_analysis(
            rho,
            dz         = dz,
            k          = BSPDCT["k"],
            knot_style = BSPDCT["knot_style"],
            n_levels   = BSPDCT["n_levels"],
        )
        elapsed = time.perf_counter() - start
        print(" Used %7.4f s for B-spline-z x DCT-xy truncation analysis"
              % elapsed)

    print("\n Compressing with B-spline-z x DCT-xy ...")
    rho_rec, C, params = mod.bspdct_compress(
        rho,
        dz         = dz,
        n_basis    = BSPDCT["n_basis"],
        frac_basis = BSPDCT["frac_basis"],
        k          = BSPDCT["k"],
        knot_style = BSPDCT["knot_style"],
        nx_dct     = BSPDCT["nx_dct"],
        ny_dct     = BSPDCT["ny_dct"],
        frac_dct   = BSPDCT["frac_dct"],
    )
    elapsed = time.perf_counter() - start
    total += elapsed
    print(" Used %7.4f s for B-spline-z x DCT-xy compression" % elapsed)

# -----------------------------------------------------------------------------
#  KL / PCA — ensemble-based
# -----------------------------------------------------------------------------
elif METHOD == "kl":

    # --- build ensemble from file list ---
    print("\n Reading ensemble models ...")
    ensemble_list = []
    for ef in KL["ensemble_files"]:
        _, _, _, m, _, _ = mod.read_mod(ef, ".rho", trans="LOGE")
        ensemble_list.append(m)
    ensemble = np.stack(ensemble_list)          # (n_models, nx, ny, nz)

    elapsed = time.perf_counter() - start
    print(" Used %7.4f s for reading %d ensemble models"
          % (elapsed, len(ensemble_list)))

    # --- compute KL basis ---
    print("\n Computing KL basis ...")
    modes, sv, mean_model, kl_shape = mod.ensemble_to_kl(
        ensemble,
        n_modes      = KL["n_modes"],
        frac_modes   = KL["frac_modes"],
        svd_method   = KL["svd_method"],
        n_oversamples= KL["n_oversamples"],
        n_power_iter = KL["n_power_iter"],
        random_state = KL["random_state"],
    )

    # variance spectrum
    mod.kl_variance_spectrum(sv)

    elapsed = time.perf_counter() - start
    print(" Used %7.4f s for KL basis computation" % elapsed)

    # --- optional truncation sweep on target model ---
    if RUN_ANALYSIS:
        print("\n Running KL truncation analysis ...")
        mod.kl_truncation_analysis(
            rho, modes, mean_model,
            shape          = rho.shape,
            singular_values= sv,
        )
        elapsed = time.perf_counter() - start
        print(" Used %7.4f s for KL truncation analysis" % elapsed)

    # --- project and reconstruct ---
    print("\n Compressing with KL ...")
    alpha   = mod.model_to_kl(rho, modes, mean_model)
    rho_rec = mod.kl_to_model(alpha, modes, mean_model, shape=rho.shape)

    elapsed = time.perf_counter() - start
    total += elapsed
    print(" Used %7.4f s for KL projection and reconstruction" % elapsed)

else:
    sys.exit(" modem_compress: unknown METHOD '%s'. "
             "Choose one of: dct, dct_sep, wavelet, legdct, bspdct, kl" % METHOD)

# =============================================================================
#  Reconstruction accuracy
# =============================================================================

rms_err = mod.dct_reconstruction_error(rho, rho_rec, norm="rms")
rel_err = mod.dct_reconstruction_error(rho, rho_rec, norm="rel_rms")
max_err = mod.dct_reconstruction_error(rho, rho_rec, norm="max")

print("\n Reconstruction errors (log-resistivity):")
print("   RMS      : %12.6g" % rms_err)
print("   Rel. RMS : %12.6g" % rel_err)
print("   Max      : %12.6g" % max_err)

# =============================================================================
#  Write reconstructed model (.rho)
# =============================================================================

out_file = MOD_FILE_OUT + "_" + METHOD

if WRITE_OUTPUT:
    rho_out = rho_rec.copy()
    rho_out[aircells] = np.log(RHOAIR)

    mod.write_mod(
        out_file, modext=".rho", trans="LOGE",
        dx=dx, dy=dy, dz=dz, mval=rho_out,
        reference=refmod, mvalair=1e17, aircells=aircells, header="",
    )
    elapsed = time.perf_counter() - start
    total += elapsed
    print("\n Used %7.4f s for writing model to %s"
          % (elapsed, out_file + ".rho"))

# =============================================================================
#  Save compressed representation (.npz)
#
#  Each archive contains:
#    mesh         dx, dy, dz, reference
#    errors       rms_err, rel_err, max_err
#    meta         method, trans ('LOGE'), mod_file_in
#    coefficients the compressed data specific to each method (see below)
#
#  DCT           coeff (1-D), keep_mask (bool), shape
#  DCT_SEP       coeff_block (3-D), shape_full, shape_keep
#  WAVELET       one array per subband (keys prefixed 'wb_'), wavelet name
#  LEGDCT        C (3-D), n_leg, nx_dct, ny_dct
#  BSPDCT        C (3-D), n_basis, k, knot_style, nx_dct, ny_dct, B, Bpinv
#  KL            alpha (1-D scores), modes, singular_values, mean_model
# =============================================================================

if WRITE_NPZ:
    npz_file = out_file + ".npz"

    # --- common arrays for all methods ---
    common = dict(
        dx        = dx,
        dy        = dy,
        dz        = dz,
        reference = np.asarray(refmod),
        rms_err   = np.array(rms_err),
        rel_err   = np.array(rel_err),
        max_err   = np.array(max_err),
        method    = np.array(METHOD),
        trans     = np.array("LOGE"),
        mod_file_in = np.array(MOD_FILE_IN),
    )

    # --- method-specific coefficient arrays ---
    if METHOD == "dct":
        extra = dict(
            coeff     = coeff,
            keep_mask = keep_mask,
            shape     = np.array(rho.shape),
        )

    elif METHOD == "dct_sep":
        extra = dict(
            coeff_block = coeff_block,
            shape_full  = np.array(shape_full),
            shape_keep  = np.array(shape_keep),
        )

    elif METHOD == "wavelet":
        # Flatten the PyWavelets coefficient dict: one array per subband,
        # keys stored as a comma-separated string for portability.
        wb_keys   = list(coeffs.keys())
        wb_arrays = {("wb_" + k): coeffs[k] for k in wb_keys}
        extra = dict(
            wavelet  = np.array(WAVELET["wavelet"]),
            wb_keys  = np.array(",".join(wb_keys)),
            **wb_arrays,
        )

    elif METHOD == "legdct":
        extra = dict(
            C      = C,
            n_leg  = np.array(params["n_leg"]),
            nx_dct = np.array(params["nx_dct"]),
            ny_dct = np.array(params["ny_dct"]),
            shape  = np.array(rho.shape),
        )

    elif METHOD == "bspdct":
        extra = dict(
            C          = C,
            n_basis    = np.array(params["n_basis"]),
            k          = np.array(params["k"]),
            knot_style = np.array(params["knot_style"]),
            nx_dct     = np.array(params["nx_dct"]),
            ny_dct     = np.array(params["ny_dct"]),
            B          = params["B"],
            Bpinv      = params["Bpinv"],
            shape      = np.array(rho.shape),
        )

    elif METHOD == "kl":
        extra = dict(
            alpha           = alpha,
            modes           = modes,
            singular_values = sv,
            mean_model      = mean_model,
            shape           = np.array(rho.shape),
        )

    np.savez_compressed(npz_file, **common, **extra)

    elapsed = time.perf_counter() - start
    total += elapsed
    print("\n Used %7.4f s for writing compressed archive to %s"
          % (elapsed, npz_file))

# =============================================================================
#  Summary
# =============================================================================

print("\n Total time used:  %f s" % total)

