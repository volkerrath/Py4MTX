#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ensembles.py

High-level ensemble utilities used with FEMTIC workflows.

This module contains:

- File-system helpers for creating per-ensemble working directories.
- Ensemble perturbation helpers for FEMTIC inputs (observe.dat and
  resistivity_block_iterX.dat).
- Roughness/precision matrix helpers and sampling routines.
- Streaming per-cell mean and variance (Welford update, Chan pairwise
  merge), third and fourth central moments (skewness, kurtosis) and
  per-cell quantiles (fixed-bin histograms) for ensembles too large to hold
  in memory.
- Ensemble analysis tools (EOFs / PCA).
- Weighted KL decomposition and a KL+PCE surrogate builder for log-resistivity
  fields on unstructured FEMTIC meshes.
- GST parameter estimation: four strategies for choosing variogram parameters
  and pilot-point layouts (gst_variogram_from_rto_samples,
  gst_pilot_point_cv, gst_sill_from_jacobian, gst_parameter_diagnostics).

All functions are importable; no code is executed on import.

Author: Volker Rath (DIAS)
Created with the help of ChatGPT (GPT-5 Thinking) on 2026-01-02 (UTC)

Provenance
----------
2026-03-31  Claude (Anthropic)
            removed debug print in sample_rtr_full_rank; removed dead
            commented-out code in generate_rto_model_ensemble; removed
            redundant per-sample print.
2026-03-31  Claude (Anthropic)
            consolidated _diag_rtr into _rtr_diag (single helper);
            removed dead estimate_low_rank_eigpairs; enriched docstrings
            with tuning recommendations.
2026-04-02  Claude (Anthropic)
            fixed FileNotFoundError in generate_rto_model_ensemble —
            template argument to fem.insert_model now uses the full
            per-member path (_orig.dat backup) instead of the bare
            basename.
2026-04-02  Claude (Anthropic)
            fixed generate_rto_model_ensemble write-back loop — now
            reads reference log10-resistivity from the backup template
            and adds the perturbation before calling insert_model
            (method='add'), so perturbed models are reference +
            delta_log10 rather than bare perturbations.
2026-04-11  Claude Sonnet 4.6 (Anthropic)
            moved check_sparse_matrix here from femtic.py (consolidation
            of all matrix/roughness tools into ensembles); femtic.py
            Section 2 now imports these functions from ensembles rather
            than duplicating them.
2026-04-27  Claude Sonnet 4.6 (Anthropic)
            added generate_gst_model_ensemble — geostatistical
            initial-model ensemble via pilot-point Ordinary Kriging
            (gstools).  No roughness matrix required.
2026-04-27  Claude Sonnet 4.6 (Anthropic)
            renamed generate_model_ensemble to
            generate_rto_model_ensemble for consistency with
            generate_gst_model_ensemble.
2026-05-28  Claude Sonnet 4.6 (Anthropic)
            put_files and generate_directories gained a relative_links
            parameter (default True): relative symlinks survive tgz/copy
            to another machine; False restores the previous
            absolute-path behaviour.
2026-05-28  Claude Sonnet 4.6 (Anthropic)
            fixed template-clobbering bug in generate_gst_model_ensemble
            — when output_target='both', the first insert_model call
            (resistivity_block) could overwrite reference_file before
            the second call read it as template; template_path is now
            resolved once before either write.
2026-05-28  Claude Sonnet 4.6 (Anthropic)
            added GST parameter estimation section — four functions for
            choosing variogram parameters before committing to a full
            ensemble run: gst_variogram_from_rto_samples (Strategy 1:
            fit variogram to RTO samples), gst_pilot_point_cv (Strategy
            2: LOO-CV on reference model), gst_sill_from_jacobian
            (Strategy 3: linearised Jacobian sill calibration),
            gst_parameter_diagnostics (Strategy 4: integrating
            diagnostic with optional plot).
2026-06-06  Claude Sonnet 4.6 (Anthropic)
            added "extrema" pilot-point mode to
            generate_gst_model_ensemble.  New helper
            _find_extrema_pilot_points (KDTree-based local extremum
            detection on free-region barycentres, optional ROI mask).
            New parameters pp_roi, pp_extrema_k, pp_extrema_which;
            graceful fallback to "random" if no extrema are found.
            Requires scipy.spatial (already a transitive dependency).
2026-06-10  Claude Sonnet 4.6 (Anthropic)
            added _resolve_fromto helper; all four ensemble functions
            (generate_directories, generate_rto_model_ensemble,
            generate_gst_model_ensemble, generate_data_ensemble) now
            accept an explicit list of member indices in addition to
            None (all).  Range semantics ([start, stop]) removed — a
            list always means explicit indices.  Type hints updated to
            Optional[List[int]]; Union/Tuple removed from fromto
            signatures.  Matching change in femtic_rto_prep.py: FROM_TO
            renamed to ENS_LIST.
2026-07-05  Claude Sonnet 5 (Anthropic)
            added pp_value_mode / pp_value_delta to
            generate_gst_model_ensemble.  pp_value_mode="uniform"
            (default) preserves the original Uniform(log_rho_min,
            log_rho_max) draw; pp_value_mode="reference" instead draws
            pilot-point values as reference_model(nearest free region) ±
            pp_value_delta (log10 Ohm.m), using a scipy.spatial.KDTree
            nearest-neighbour lookup against the free-region
            barycentres.  Reference log10(rho) at free regions is now
            computed unconditionally (previously only inside the
            "extrema" pp_mode branch) so it is available to both
            "extrema" placement and "reference" value mode.
2026-07-05  Claude Sonnet 5 (Anthropic)
            raised the default neighbourhood size for extremum detection
            from k=9 to k=30 in both _find_extrema_pilot_points and
            generate_gst_model_ensemble's pp_extrema_k — the
            strictly-less/greater-than-all-neighbours test was flagging
            too many spurious local minima/maxima at small k on typical
            FEMTIC meshes; recommended range updated from 7-15 to 20-40.
2026-07-17  Claude Sonnet 5 (Anthropic)
            migrated from legacy scipy.sparse matrix classes to the
            array-equivalent API throughout: removed unused isspmatrix
            import; updated scipy.sparse.spmatrix type hints to
            scipy.sparse.sparray (get_roughness, make_prior_cov,
            prune_inplace, prune_rebuild, dense_to_csr, matrix_reduce,
            check_sparse_matrix, build_rtr_operator,
            make_rtr_preconditioner,
            make_sparse_cholesky_precision_solver, _rtr_diag,
            pick_lam_from_rtr_diag); replaced scipy.sparse.identity()
            (returns a legacy matrix) with scipy.sparse.eye_array() in
            make_rtr_preconditioner and
            make_sparse_cholesky_precision_solver. Also fixed a latent
            bug in get_roughness/make_prior_cov where `eye` was imported
            aliased as `eye_array` (silently returning a legacy sparse
            matrix instead of an array); now imports the genuine
            eye_array constructor.
2026-07-19  Claude Sonnet 5 (Anthropic)
            added a mesh-agnostic GST perturbation machine:
            krige_pilot_points_to_targets(), _draw_pilot_points(),
            generate_gst_perturbation() factor the pilot-point placement
            + Ordinary-Kriging core out of generate_gst_model_ensemble()
            so a single realisation can be generated in memory (no
            directory tree, no file I/O) given nothing but a target
            point cloud. Also added the ModEM rectilinear-grid wrappers
            modem_gst_cell_centers() and
            generate_gst_perturbation_modem().
            generate_gst_model_ensemble() itself is unchanged. Written
            to support modem_nss.py (ModEM null-space shuttle, analogous
            to femtic_nss.py).
2026-07-25  Claude Sonnet 5 (Anthropic)
            generate_gst_model_ensemble gained save_pilot_points /
            pilot_points_file / seed: when save_pilot_points=True, every
            member's pilot-point coordinates (pp_x/y/z) and drawn
            log10(rho) values (pp_vals) are accumulated during the
            member loop and written to a single compressed .npz (default
            f"{dir_base}pilot_points.npz") after it completes, alongside
            pp_mode/pp_value_mode/variogram/log_rho_min/log_rho_max/seed
            metadata for a self-describing archive. seed is
            informational only (recorded, not used to seed anything --
            pass a seeded rng for that); reproducible runs are the
            caller's responsibility (femtic_gst_prep.py now does this
            via a RANDOM_SEED config variable). generate_data_ensemble
            gained an rng parameter, forwarded to femtic.modify_data
            (which already accepted one) for each member, so the
            data-perturbation path can share the same seeded generator
            as the model-perturbation path instead of each modify_data
            call silently falling back to its own unseeded generator.
2026-08-25  Claude Sonnet 5 (Anthropic)
            generate_gst_model_ensemble: pp_bbox default changed from a
            fixed generic tuple to None; pp_bbox / pp_roi are now
            resolved against the model's actual free-region extent
            (axis-wise min/max of the free-region barycentres computed
            from mesh_file/ref_mod_file) right after that extent becomes
            available. None resolves to the full extent on every axis; a
            supplied box has each of its six bounds independently
            clamped to that extent, so a bbox/ROI wider than the true
            model on some axis (e.g. reused from a different survey) is
            truncated to the model's actual size on that axis rather
            than placing pilot points outside the free-region domain or
            (for pp_roi) silently accepting an over-wide box. A one-line
            log message is printed (out=True) whenever clamping actually
            changes a supplied box. generate_gst_perturbation /
            _draw_pilot_points (the mesh-agnostic perturbation machine
            used by femtic_nss.py / modem_nss.py) are unchanged by this
            update -- they take a bare target point cloud with no mesh
            to derive an extent from.
2026-09-09  Claude Sonnet 5 (Anthropic)
            added eof_model_from_covariance(),
            sample_new_models_from_ensemble(), and
            sample_new_models_from_covariance() (EOF/PCA section, after
            sample_physical_ensemble()), for femtic_ens_repair.py's new
            MOD_REPAIR_METHOD="eof_sample" option.
            eof_model_from_covariance() builds an EOFModel directly from
            a precomputed covariance (either the "low_rank"
            eigval/eigvec factorisation or a dense "full" cov, as saved
            by femtic_ens_post.py's COMPUTE_COV) instead of fitting one
            from a raw ensemble matrix, so sample_physical_ensemble()
            can be reused unchanged regardless of whether the covariance
            is fit on the fly or loaded from disk. The two
            sample_new_models_from_* wrappers handle the samples-as-rows
            <-> samples-as-columns transpose between femtic_ens_*.py's
            ens_matrix convention (n_members, n_free) and
            fit_eof_model()/sample_physical_ensemble()'s (ncells,
            nsamples) convention, so callers never touch
            EOFModel/fit_eof_model directly. Note:
            eof_generate_ensemble() (defined earlier in this section) is
            dead code -- its body is only a docstring, so it always
            returns None -- pre-existing and unrelated to this change;
            flagged here rather than fixed, pending confirmation of
            whether it should be implemented or removed.
2026-09-09  Claude Sonnet 5 (Anthropic)
            added pp_regen_every to generate_gst_model_ensemble():
            optional int controlling how often the *random* pilot-point
            component is redrawn (every member, as before, when None;
            once per block of pp_regen_every consecutive processed
            members otherwise, via a new position-keyed cache --
            pp_mode="random"'s whole point set, or the random-fill
            portion of "mixed"/"extrema"; pp_coords / the extrema
            skeleton are never affected either way). Pilot-point values
            are still redrawn every member regardless, so blocked
            members still differ from each other; only the locations are
            shared within a block. No effect for pp_mode="fixed"
            (nothing random to regenerate; a warning is printed if set
            anyway). Recorded in the pilot_points.npz archive
            (save_pilot_points=True) as "pp_regen_every" (-1 if not
            set), alongside the existing "seed" metadata.
            femtic_gst_prep.py's new MOD_PP_REGEN_EVERY threads through
            to this parameter -- see its README's matching entry.
2026-09-17  Claude Sonnet 5 (Anthropic)
            generate_data_ensemble gained a derive_pt_from_z parameter
            (default False), forwarded verbatim to femtic.modify_data.
            When True and a member's observe.dat has both an MT and a PT
            block, that member's PT data is overridden, after its normal
            reset/perturb pass, by the phase tensor of that same
            member's own already-perturbed Z (matched by site name and
            frequency), instead of PT being perturbed independently of
            Z; sites/frequencies without an MT match keep their
            independently-perturbed PT value.
2026-10-07  Claude Sonnet 5.5 (Anthropic)
            Reformatted this module docstring: the changelog was a mix of
            flush-left "Updated ..." paragraphs and indented "Modified: ..."
            blocks with inconsistent separators; now one "Provenance" list
            (date, author, indented text) as in the other py4mt modules.
            Wording of all earlier entries is unchanged; no code change.
2026-10-16  Claude Sonnet 5.5 (Anthropic)
            added a streaming mean/variance section (after
            generate_data_ensemble, before the EOF/PCA section):
            welford_update() (one-sample Welford update), batch_moments()
            ((n, mean, M2) of a chunk), chan_merge() (pairwise merge of
            two accumulators, Chan, Golub and LeVeque 1983, for combining
            chunks or MPI ranks) and moments_variance() (M2 / (n - ddof)).
            For ensembles too large to hold in memory, e.g. many
            nullspace-shuttle models.  Vectorised over cells; no change to
            existing functions.
2026-10-16  Claude Sonnet 5.5 (Anthropic)
            added streaming per-cell quantiles to the same section:
            hist_update() (accumulate chunks into per-cell fixed-bin
            histograms; histograms of equal range and bin count merge by
            plain addition, also across MPI ranks) and hist_quantiles()
            (quantiles with linear interpolation inside the bin, e.g. the
            1-sigma and 2-sigma probabilities of a normal distribution).
            Within one bin width of the empirical inverse-CDF quantile.  No
            change to existing functions.
2026-10-17  Claude Sonnet 5.5 (Anthropic)
            added streaming higher moments to the same section:
            batch_moments4() ((n, mean, M2, M3, M4) of a chunk),
            merge_moments4() (pairwise merge of two such accumulators,
            Pebay 2008 / Terriberry 2007 / Pebay et al. 2016; reduces to
            chan_merge() for n, mean, M2), skew_kurtosis() (skewness and
            excess kurtosis, optionally bias-corrected) and
            quantile_skewness() (Bowley coefficient from three quantiles,
            e.g. the 1-sigma or 2-sigma quantiles of hist_quantiles()).
            No change to existing functions.
2026-10-21  Claude Sonnet 5.5 (Anthropic)
            RTO prior perturbation corrected (audit of the femtic_rto
            workflow: ensemble spreads far too small compared with GST).
            (1) NEW sample_rto_prior() (+ prior_energy_ratio()): exact draws
            from N(0, (alpha (R^T R + lam I))^-1) via
            x = alpha^-1/2 Q^-1 (R^T xi1 + sqrt(lam) xi2), sparse LU or
            Jacobi-CG with tight tolerance.  (2) generate_rto_model_ensemble
            now uses it; new REQUIRED argument alpha (regularisation weight
            of the member inversions; the old code sampled with an implicit
            alpha = 1 and MOD_MU/MOD_PDF in femtic_rto_prep.py were never
            passed on); algo is now "auto"/"splu"/"cg" (legacy "low rank" /
            "full rank" are mapped with a DeprecationWarning, the old
            low-rank arguments are ignored with a warning); exactly
            len(members) draws are made; prints RMS std and the energy ratio
            of the draws; optional start_from_perturbed copies the perturbed
            model to resistivity_block_iter0.dat (symlink replaced, template
            untouched).  (3) sample_rtr_low_rank() DEPRECATED: randomized_svd
            returns the LARGEST singular values of R, i.e. the roughest modes
            with the SMALLEST prior variance 1/s^2 (0.1-0.5 % of the prior
            variance for 128 modes in a 3-D test; ensemble spread 0.2-0.3 of
            the correct value in a linear RTO test); it now warns when
            called.  (4) sample_rtr_full_rank() fixed: it drew
            (R^T R + lam I)^-1 R^T xi, whose covariance is
            Q^-1 R^T R Q^-1, not Q^-1 (smoothest modes damped by e/(e+lam),
            null space of R without variance); the missing sqrt(lam) * xi2
            term was added (lam is resolved by the new helper _resolve_lam,
            shared with make_precision_solver).  Also affects
            femtic.modify_model_npz(method="precision_sample").  When the
            solver is built inside the sampler, CG now defaults to
            rtol 1e-8 / atol 0 (was 1e-3 / 1e-4: 0.73 of the exact RMS std).
            (5) make_precision_solver docstring: rtol = 1e-2 was recommended
            for sampling, but a loose CG tolerance leaves the smoothest
            (highest-variance) modes under-resolved (9 % of their variance at
            rtol 1e-2, 72 % at 1e-3 in a Laplacian-type test); CG with
            ILU did not converge there (spilu is not an SPD preconditioner).
            (6) generate_rto_model_ensemble: optional remove_mean (default
            False) subtracts each draw's global mean; the constant mode of
            R^T R has a variance set by the arbitrary shift lam
            (std ~ 1/sqrt(n * alpha * lam) per cell) and dominates the
            printed RMS std although |R x|^2 does not see it; the printout
            now also gives the RMS std without the global mean.
2026-10-23  Claude Sonnet 5.5 (Anthropic)
            generate_gst_model_ensemble: two new optional parameters,
            pp_cache and pp_pos_offset, that let a caller who invokes the
            function repeatedly with n_samples=1 (e.g. femtic_nss.py, one
            Kriging call per null-space shuttle) keep the block structure
            of pp_regen_every across calls.  pp_cache is a caller-owned
            dict that holds the random pilot-point layout of the current
            block; pp_pos_offset is added to the position before the
            block id is computed (block_id = (pos + offset) // K), i.e.
            the global index of the first model of the call.  With both
            left at their defaults the behaviour is unchanged (the cache
            is local to the call).  Blocks older than the current one are
            dropped from the cache when a new block is opened, so the
            cache stays O(1) however many models are drawn.  Passing
            pp_cache without pp_regen_every raises ValueError (it would
            be silently ignored otherwise).

AI-generated; please review before production use.
"""

from __future__ import annotations

import os
import sys
import shutil
import time

import itertools
import math
from dataclasses import dataclass

from typing import (
    Callable,
    Optional,
    Sequence,
    Tuple,
    Literal,
    List,
    Union,
)
from numpy.typing import ArrayLike

import warnings
import numpy as np
from numpy.polynomial.hermite import hermval
from numpy.polynomial.legendre import legval
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

from numpy.random import Generator, default_rng
from scipy.sparse import issparse, eye
from scipy.sparse.linalg import LinearOperator, cg, eigsh
from scipy.sparse.linalg import bicgstab, spilu, splu
from numpy.linalg import eigh

# Optional but kept for compatibility with earlier versions
import joblib  # noqa: F401

try:
    from . import femtic as fem  # type: ignore
except Exception:  # pragma: no cover
    import femtic as fem



def _resolve_fromto(
    fromto: "Optional[List[int]]",
    n_samples: int,
) -> np.ndarray:
    """Resolve the *fromto* argument used by ensemble generation functions.

    Parameters
    ----------
    fromto : None or list of int
        * ``None``           — all members ``0 … n_samples-1``
        * ``[i, j, k, …]``  — explicit list of member indices to process
    n_samples : int
        Total ensemble size; used only when *fromto* is ``None``.

    Returns
    -------
    np.ndarray of int
        Ordered array of member indices to process.
    """
    if fromto is None:
        return np.arange(n_samples)
    return np.asarray(list(fromto), dtype=int)


def generate_directories(
    alg: str = "rto",
    dir_base: str = "./ens_",
    templates: str = "",
    copy_list: Sequence[str] = (
        "observe.dat",
        "resistivity_block_iter0.dat",
        "distortion_iter0.dat",
        "prior.dat",
        "run_dub.sh",
        "run_oar_sh",
    ),
    link_list: Sequence[str] = (
        "control.dat",
        "mesh.dat",
        "resistivity_block_iter0.dat",
        "distortion_iter0.dat",
    ),
    n_samples: int = 1,
    fromto: Optional[List[int]] = None,
    relative_links: bool = True,
    out: bool = True,
) -> list[str]:
    """
    Create ensemble directories and copy template files into each of them.

    Parameters
    ----------
    dir_base : str, optional
        Base name for ensemble directories (e.g. "./ens_").
    templates : str, optional
        Directory prefix from which the template files are copied.
    copy_list : sequence of str
        Names of files to copy into each ensemble directory.
    link_list : sequence of str
        Names of files to link into each ensemble directory.
    n_samples : int
        Number of ensemble members if ``fromto`` is None.
    fromto : None or list of int, optional
        Controls which members are processed:

        * ``None`` (default) — all members ``0 … n_samples-1``.
        * ``[i, j, k, …]``  — explicit list of member indices to process.
    relative_links : bool, optional
        If True (default), symlink targets are relative to each member
        directory, making the ensemble tree portable after tgz/copy to
        another machine.  If False, absolute paths are used instead.
    out : bool, optional
        If True, print status messages.

    Returns
    -------
    dir_list : list of str
        List of created ensemble directory paths.
    """
    from_to = _resolve_fromto(fromto, n_samples)

    dir_list: list[str] = []
    for iens in from_to:
        directory = f"{dir_base}{iens}/"
        os.makedirs(directory, exist_ok=True)
        put_files(copy_list, link_list, directory, templates,
                  relative_links=relative_links)
        dir_list.append(directory)


    if out:
        print("list of directories:")
        print(dir_list)

    return dir_list



def get_roughness(
    filerough: str = "roughening_matrix.out",
    regeps: Optional[float] = None,
    spformat: str = "csc",
    out: bool = True,
) -> scipy.sparse.sparray:
    """
    Read FEMTIC roughening_matrix.dat and build sparse roughness matrix R.

    Parameters
    ----------
    filerough : str
        Path to roughening_matrix.dat file.
    regeps : float or None
        If not None, add regeps * I to R.
    spformat : {"csr", "csc", "coo"}
        Sparse format of the returned matrix.
    out : bool
        If True, print timings and matrix statistics.

    Returns
    -------
    R : sparse matrix
        Roughness matrix in chosen sparse format.

    Notes
    -----
    The function implements the logic implied by FEMTIC's C++ code for
    user-defined roughening matrices. It first counts blocks with zero
    non-zeros, then builds the triplet arrays (row, col, val).
    """
    from scipy.sparse import csr_array, csc_array, coo_array, eye_array

    start = time.perf_counter()
    if out:
        print("get_roughness: Reading from", filerough)
    irow: list[int] = []
    icol: list[int] = []
    vals: list[float] = []

    with open(filerough, "r") as file:
        content = file.readlines()

    num_elem = int(content[0].split()[0])
    if out:
        print("get_roughness: File read:", time.perf_counter() - start, "s")
        print("get_roughness: Number of elements:", num_elem)

    iline = 0
    zeros = 0
    # first pass: count zero blocks
    while iline < len(content) - 1:
        iline += 1
        ele = int(content[iline].split()[0])
        nel = int(content[iline + 1].split()[0])
        if nel == 0:
            iline += 1
            zeros += 1
            if out:
                print("passed", ele, nel, iline)
        else:
            iline += 2
    if out:
        print("Zero elements:", zeros)

    # second pass: collect non-zero triplets
    start = time.perf_counter()
    iline = 0
    while iline < len(content) - 1:
        iline += 1
        ele = int(content[iline].split()[0])
        nel = int(content[iline + 1].split()[0])
        if nel == 0:
            iline += 1
            continue
        else:
            irow += [ele - zeros] * nel
            col = [int(x) - zeros for x in content[iline + 1].split()[1:]]
            icol += col
            val = [float(x) for x in content[iline + 2].split()]
            vals += val
            iline += 2

    irow_arr = np.asarray(irow, dtype=int)
    icol_arr = np.asarray(icol, dtype=int)
    vals_arr = np.asarray(vals, dtype=float)

    R = coo_array((vals_arr, (irow_arr, icol_arr)))
    if out:
        print(R.shape)
        print("get_roughness: R sparse format is", R.format)

    if regeps is not None:
        R = R + regeps * eye_array(R.shape[0], format=R.format)
        if out:
            print(regeps, "added to diag(R)")

    if out:
        print("get_roughness: R generated:", time.perf_counter() - start, "s")
        print("get_roughness: R sparse format is", R.format)
        print(R.shape, R.nnz)

    if "csc" in spformat.lower():
        R = csc_array((vals_arr, (irow_arr, icol_arr)))
    elif "csr" in spformat.lower():
        R = csr_array((vals_arr, (irow_arr, icol_arr)))
    else:
        R = coo_array((vals_arr, (irow_arr, icol_arr)))

    if out:
        print("get_roughness: Output sparse format:", spformat)
        print("get_roughness: R sparse format is", R.format)
        print(R.shape, R.nnz)
        print(R.nnz, "nonzeros, ", 100 * R.nnz / R.shape[0] ** 2, "%")
        print("get_roughness: Done!\n\n")

    return R


def make_prior_cov(
    rough: scipy.sparse.sparray,
    regeps: float = 1.0e-5,
    spformat: str = "csr",
    spthresh: float = 1.0e-4,
    spfill: float = 10.0,
    spsolver: Optional[str] = "ilu",
    spmeth: str = "basic,area",
    outmatrix: str = "invRTR",
    nthreads: int = 16,
    out: bool = True,
) -> scipy.sparse.sparray | np.ndarray:
    """
    Generate a prior covariance proxy M from a roughness matrix R.

    Strategy
    --------
    1. Stabilise by adding a small diagonal (regeps).
    2. Invert R (approximately) using either:
       - sparse LU ("slu"), or
       - incomplete LU ("ilu").
    3. Optionally sparsify the inverse via :func:`matrix_reduce`.
    4. Build

           M = invR @ invR.T        if outmatrix contains "rtr"
           M = invR                 otherwise

    Parameters
    ----------
    rough : sparse matrix
        Roughness matrix R.
    regeps : float
        Small diagonal added to R for stability.
    spformat : {"csr", "csc", "coo"}
        Sparse format used in some steps.
    spthresh : float
        Drop tolerance for ILU and for matrix_reduce.
    spfill : float
        Fill-factor for ILU.
    spsolver : {"slu", "ilu"}
        Choice of sparse direct solve / factorization.
    spmeth : str
        Drop-rule string for ILU; currently not used explicitly.
    outmatrix : {"invR", "invRTR", "invRTR_deco"}
        Output matrix type.
    nthreads : int
        Thread limit for underlying BLAS/LAPACK where supported.
    out : bool
        If True, print info.

    Returns
    -------
    M : sparse or dense array
        Prior covariance proxy.
    """
    from scipy.sparse import (
        csr_array,
        csc_array,
        coo_array,
        eye_array,
        issparse,
    )
    from threadpoolctl import threadpool_limits

    if rough is None:
        sys.exit("make_prior_cov: No roughness matrix given! Exit.")

    if not issparse(rough):
        sys.exit("make_prior_cov: Roughness matrix is not sparse! Exit.")

    start = time.perf_counter()

    if out:
        print("make_prior_cov: Shape of input roughness is", rough.shape)
        print("make_prior_cov: Format of input roughness is", rough.format)

    if regeps is not None:
        rough = rough + regeps * eye_array(rough.shape[0], format=spformat.lower())
        if out:
            print(regeps, "added to diag(R)")

    if spsolver is None:
        sys.exit("make_prior_cov: spsolver must be 'slu' or 'ilu'.")

    if "slu" in spsolver.lower():
        from scipy.sparse.linalg import spsolve

        R = csc_array(rough)
        RHS = eye_array(R.shape[0], format=R.format)
        with threadpool_limits(limits=nthreads):
            invR = spsolve(R, RHS)

    elif "ilu" in spsolver.lower():
        R = csc_array(rough)
        RHS = eye_array(R.shape[0], format=R.format)
        beg = time.perf_counter()
        with threadpool_limits(limits=nthreads):
            iluR = spilu(R, drop_tol=spthresh, fill_factor=spfill)
            if out:
                print("spilu decomposed:", time.perf_counter() - beg, "s")
            beg = time.perf_counter()
            invR = iluR.solve(RHS.toarray())
            if out:
                print("spilu solved:", time.perf_counter() - beg, "s")
    else:
        sys.exit(f"make_prior_cov: solver {spsolver} not available! Exit")

    if out:
        print("make_prior_cov: invR generated:", time.perf_counter() - start, "s")
        print("make_prior_cov: invR type", type(invR))

    if spthresh is not None:
        invR = matrix_reduce(
            M=invR,
            howto="relative",
            spthresh=spthresh,
            spformat=spformat,
            out=out,
        )

    M = invR
    if "rtr" in outmatrix.lower():
        M = invR @ invR.T

    if out:
        print("make_prior_cov: M generated:", time.perf_counter() - start, "s")
        print("make_prior_cov: M is", outmatrix)
        print("make_prior_cov: M type", type(M))
        print("make_prior_cov:  Done!\n\n")
    return M


def prune_inplace(M: scipy.sparse.sparray, threshold: float) -> scipy.sparse.sparray:
    """
    In-place pruning of small entries in a CSR sparse matrix.

    Parameters
    ----------
    M : sparse matrix
        Sparse matrix to prune.
    threshold : float
        Entries with |M_ij| < threshold are set to zero.

    Returns
    -------
    M : sparse matrix
        The pruned matrix (same object, modified in-place).
    """
    from scipy.sparse import csr_array, issparse

    if issparse(M):
        if M.format != "csr":
            M = M.tocsr()
    else:
        M = csr_array(M)

    mask = np.abs(M.data) < threshold
    if mask.any():
        M.data[mask] = 0.0
        M.eliminate_zeros()
    return M


def prune_rebuild(M: scipy.sparse.sparray, threshold: float) -> scipy.sparse.sparray:
    """
    Rebuild a CSR sparse matrix from COO representation after pruning.

    Parameters
    ----------
    M : sparse matrix
        Matrix to prune.
    threshold : float
        Entries with |M_ij| < threshold are dropped.

    Returns
    -------
    M_csr : sparse matrix
        New CSR matrix with pruned entries.
    """
    from scipy.sparse import csr_array

    coo = M.tocoo()
    absdata = np.abs(coo.data)
    keep = absdata >= threshold
    if not keep.all():
        return csr_array(
            (coo.data[keep], (coo.row[keep], coo.col[keep])),
            shape=M.shape,
        )
    else:
        return M.tocsr()


def dense_to_csr(
    M: np.ndarray,
    threshold: float = 0.0,
    chunk_rows: int = 1000,
    dtype: Optional[np.dtype] = None,
) -> scipy.sparse.sparray:
    """
    Convert a dense matrix to CSR, dropping entries below a threshold.

    Parameters
    ----------
    M : ndarray
        Dense matrix.
    threshold : float
        Entries with |M_ij| <= threshold are dropped.
    chunk_rows : int
        Process rows in chunks of this size to reduce memory peaks.
    dtype : numpy dtype, optional
        Output dtype; if None, use M.dtype.

    Returns
    -------
    M_csr : sparse matrix
        CSR matrix.
    """
    from scipy.sparse import csr_array

    rows_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []
    data_list: list[np.ndarray] = []
    nrows = M.shape[0]
    for r0 in range(0, nrows, chunk_rows):
        r1 = min(nrows, r0 + chunk_rows)
        block = M[r0:r1]
        mask = np.abs(block) > threshold
        rr, cc = np.nonzero(mask)
        rows_list.append((rr + r0).astype(np.int64))
        cols_list.append(cc.astype(np.int64))
        data_list.append(
            block[rr, cc].astype(dtype if dtype is not None else M.dtype)
        )

    if not rows_list:
        return csr_array(M.shape, dtype=dtype if dtype is not None else M.dtype)

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    data = np.concatenate(data_list)
    return csr_array((data, (rows, cols)), shape=M.shape)


def save_spilu(filename: str = "ILU.npz", ILU=None) -> None:
    """
    Save a SciPy ILU decomposition object to a single .npz file.

    Parameters
    ----------
    filename : str
        Output npz file.
    ILU : object
        ILU object returned by scipy.sparse.linalg.spilu.

    Notes
    -----
    To restore, use :func:`load_spilu`.
    """
    if ILU is None:
        sys.exit("save_spilu: No ILU object given! Exit.")

    np.savez(
        filename,
        L_data=ILU.L.data,
        L_indices=ILU.L.indices,
        L_indptr=ILU.L.indptr,
        L_shape=ILU.L.shape,
        U_data=ILU.U.data,
        U_indices=ILU.U.indices,
        U_indptr=ILU.U.indptr,
        U_shape=ILU.U.shape,
        perm_r=ILU.perm_r,
        perm_c=ILU.perm_c,
    )


def load_spilu(filename: str = "ILU.npz"):
    """
    Load ILU decomposition components from npz file.

    Parameters
    ----------
    filename : str
        npz file created by :func:`save_spilu`.

    Returns
    -------
    L, U, perm_r, perm_c
        Components of the ILU decomposition.
    """
    from scipy.sparse import csc_array

    data = np.load(filename)
    L = csc_array(
        (data["L_data"], data["L_indices"], data["L_indptr"]),
        shape=tuple(data["L_shape"]),
    )
    U = csc_array(
        (data["U_data"], data["U_indices"], data["U_indptr"]),
        shape=tuple(data["U_shape"]),
    )
    perm_r = data["perm_r"]
    perm_c = data["perm_c"]
    return L, U, perm_r, perm_c


def matrix_reduce(
    M: np.ndarray | scipy.sparse.sparray,
    howto: str = "relative",
    spformat: str = "csr",
    spthresh: float = 1.0e-6,
    prune: str = "rebuild",
    out: bool = True,
) -> scipy.sparse.sparray:
    """
    Reduce a (possibly dense) matrix to sparse form by dropping small entries.

    Parameters
    ----------
    M : array_like or sparse matrix
        Matrix to sparsify.
    howto : {"relative", "absolute"}
        If "relative", use spthresh * max(|M|) as threshold;
        if "absolute", use spthresh directly.
    spformat : {"csr", "csc", "coo"}
        Output sparse format.
    spthresh : float
        Threshold parameter.
    prune : {"inplace", "rebuild"}
        Strategy for pruning; currently "rebuild" is more robust.
    out : bool
        If True, print info.

    Returns
    -------
    M_sp : sparse matrix
        Sparsified matrix.
    """
    from scipy.sparse import csr_array, csc_array, coo_array, issparse

    if M is None:
        sys.exit("matrix_reduce: no matrix given! Exit.")

    if issparse(M):
        M_sp = M.tocsr()
        if out:
            print("matrix_reduce: Matrix is sparse.")
            print("matrix_reduce: Type:", type(M_sp))
            print("matrix_reduce: Format:", M_sp.format)
            print("matrix_reduce: Shape:", M_sp.shape)
    else:
        M_sp = csr_array(M)
        if out:
            print("matrix_reduce: Matrix is dense.")
            print("matrix_reduce: Type:", type(M_sp))
            print("matrix_reduce: Shape:", M_sp.shape)

    n = M_sp.shape[0]
    if out:
        print(
            "matrix_reduce:",
            M_sp.nnz,
            "nonzeros, ",
            100 * M_sp.nnz / n ** 2,
            "%",
        )

    if "abs" in howto.lower():
        threshold = spthresh
    else:
        maxM = np.max(np.abs(M_sp.data))
        threshold = spthresh * maxM

    if issparse(M_sp):
        if "in" in prune:
            M_sp = prune_inplace(M_sp, threshold)
        else:
            M_sp = prune_rebuild(M_sp, threshold)
    else:
        M_sp = dense_to_csr(M_sp, threshold=threshold, chunk_rows=10000)

    if "csr" in spformat.lower():
        M_sp = M_sp.tocsr()
    elif "csc" in spformat.lower():
        M_sp = M_sp.tocsc()
    else:
        M_sp = M_sp.tocoo()

    if out:
        print("matrix_reduce: New Format:", M_sp.format)
        print("matrix_reduce: Shape:", M_sp.shape)
        print(
            "matrix_reduce:",
            M_sp.nnz,
            "nonzeros, ",
            100 * M_sp.nnz / n ** 2,
            "%",
        )
        print("matrix_reduce: Done!\n\n")
    return M_sp


def check_sparse_matrix(M: scipy.sparse.sparray, condition: bool = True) -> None:
    """
    Print diagnostic information about a sparse matrix.

    Parameters
    ----------
    M : sparse matrix
        Matrix to be tested.
    condition : bool
        Placeholder (not used).

    Returns
    -------
    None
    """
    from scipy.sparse import issparse

    if M is None:
        sys.exit("check_sparse_matrix: No matrix given! Exit.")

    if not issparse(M):
        sys.exit("check_sparse_matrix: Matrix is not sparse! Exit.")

    print("check_sparse_matrix: Type:", type(M))
    print("check_sparse_matrix: Format:", M.format)
    print("check_sparse_matrix: Shape:", M.shape)
    print(
        "check_sparse_matrix:",
        M.nnz,
        "nonzeros, ",
        100 * M.nnz / M.shape[0] ** 2,
        "%",
    )

    if M.shape[0] == M.shape[1]:
        print("check_sparse_matrix: Matrix is square!")
        test = M - M.T
        print("   R-R^T max/min:", test.max(), test.min())
        if test.max() + test.min() == 0.0:
            print("check_sparse_matrix: Matrix is symmetric!")
        else:
            print("check_sparse_matrix: Matrix is not symmetric!")

    maxaM = np.amax(np.abs(M))
    minaM = np.amin(np.abs(M))
    print("check_sparse_matrix: M max/min:", M.max(), M.min())
    print("check_sparse_matrix: M abs max/min:", maxaM, minaM)

    if np.any(np.abs(M.diagonal()) == 0):
        print("check_sparse_matrix: M diagonal element is 0!")
        print(np.where(np.abs(M.diagonal()) == 0))

    print("check_sparse_matrix: Done!\n\n")


def put_files(
    copies: Sequence[str],
    links: Sequence[str],
    directory: str,
    templates: str,
    relative_links: bool = True,
) -> None:
    """
    Copy or symlink a list of files from `templates` prefix into `directory`.

    Parameters
    ----------
    copies : sequence of str
        Files to copy (full copy via shutil.copy2).
    links : sequence of str
        Files to symlink (symbolic link via os.symlink).
    directory : str
        Target directory.
    templates : str
        Template directory (prefix) for input files.
    relative_links : bool, optional
        If True (default), symlink targets are stored as paths relative to
        ``directory``, so the ensemble tree remains portable after tgz/copy
        to another machine.  If False, absolute paths are used instead.
    """
    for fname in copies:
        src = templates + fname
        shutil.copy2(src, directory)

    for fname in links:
        src_abs = os.path.abspath(templates + fname)
        dst = os.path.join(directory, fname)
        if relative_links:
            src_lnk = os.path.relpath(src_abs, start=os.path.abspath(directory))
        else:
            src_lnk = src_abs
        if os.path.islink(dst):
            os.remove(dst)
        os.symlink(src_lnk, dst)


def generate_rto_model_ensemble(
    alg: str = 'rto',
    dir_base: str = "./ens_",
    n_samples: int = 1,
    fromto: Optional[List[int]] = None,
    refmod: str = "resistivity_block_iter0.dat",
    q: Optional[scipy.sparse.sparray | np.ndarray] = None,
    algo: str = "auto",
    method: str = "add",
    alpha: Optional[float] = None,
    lam: float = 0.0,
    lam_alpha: float = 1.0e-4,
    solver_rtol: float = 1.0e-8,
    start_from_perturbed: bool = False,
    remove_mean: bool = False,
    iter0_file: str = "resistivity_block_iter0.dat",
    rng: Optional[Generator] = None,
    out: bool = True,
    **legacy,
) -> list[str]:
    """
    Generate a resistivity model ensemble for RTO by perturbing the reference
    model with exact draws from the roughness prior N(0, (alpha (R^T R + lam I))^-1).

    Pass the roughness matrix **R** directly (not Q = R^T R).  Sampling is done
    by :func:`sample_rto_prior`:

        x = alpha^(-1/2) Q^(-1) ( R^T xi1 + sqrt(lam) xi2 ),   Q = R^T R + lam I

    which has covariance exactly (alpha Q)^(-1) (in exact arithmetic).

    The resulting log10-resistivity perturbations are added to the reference
    model of each member (``method="add"``) and inserted into the FEMTIC
    block file ``refmod`` of that member directory.

    Parameters
    ----------
    dir_base : str
        Ensemble base directory (e.g. ``"./ens_"``).
    n_samples : int
        Number of samples if ``fromto`` is None.
    fromto : None or list of int, optional
        Controls which members are written.  See :func:`_resolve_fromto`.
        Exactly ``len(members)`` prior draws are made.
    refmod : str
        Name of the reference block file inside each ensemble directory.
    q : sparse matrix or ndarray
        **Roughness matrix R** (not Q).
    alpha : float
        **Required.**  The regularisation weight with which the member
        inversions are run, in the convention
        ``cost = (d - F(m))^T Cd^-1 (d - F(m)) + alpha |R (m - m_ref)|^2``
        (FEMTIC: check ``ObjFunc = Misfit + Alpha * Roughness`` in
        ``femtic.cnv``).  The prior covariance is (alpha R^T R)^-1, so the
        perturbation std scales as alpha^(-1/2).  Use ONE fixed alpha for the
        draws and for all member inversions.  There is deliberately no
        default: the former implicit alpha = 1 gave wrong ensemble spreads.
    algo : {"auto", "splu", "cg"}
        Solver for Q: sparse LU (factor once) or Jacobi-preconditioned CG.
        "auto" uses LU up to 60000 unknowns, CG above.  The legacy values
        ``"low rank"`` and ``"full rank"`` are accepted with a warning and
        mapped to "auto" and "cg"; the old randomized-SVD sampler kept the
        roughest instead of the smoothest modes (see :func:`sample_rtr_low_rank`).
    method : {"add", "replace"}
        ``"add"`` adds the perturbation to the reference log10 model (correct
        RTO usage); ``"replace"`` uses the sample as absolute log10 values.
    lam : float
        Fixed diagonal shift.  If 0 (default), ``lam = lam_alpha * median(diag(R^T R))``.
        Small and only there because R^T R is singular (constants).
    lam_alpha : float
        Scale for the automatic shift (default 1e-4).
    solver_rtol : float
        CG relative residual tolerance (only ``algo="cg"``).  Keep it tight
        (<= 1e-6): CG converges last on the smoothest modes, which carry the
        largest prior variance.
    start_from_perturbed : bool
        If True, also copy the perturbed model to ``iter0_file`` in each member
        directory (replacing a shared symlink by a real file; the template is
        never written through the link), so every inversion starts at its own
        prior draw as in Blatter et al. (2022).  Default False (all members
        start from the shared ``resistivity_block_iter0.dat``).
    remove_mean : bool
        If True, subtract each draw's mean over the free regions.  R^T R is
        singular (constants), so the draws contain a global offset of the whole
        model with std ~ 1/sqrt(n * alpha * lam) that is set by the arbitrary
        shift ``lam``, not by the data or the regularisation; ``|R x|^2``
        (hence the member inversion, when the objective depends on the
        reference model only through ``R (m - m_ref)``) does not see it.
        It shows up in the RMS std printed with ``out=True`` and makes a
        ``start_from_perturbed`` starting model needlessly far from the data.
        Default False (exact sampler output); the energy ratio is unchanged
        by the removal if R annihilates constants (checked; warning otherwise).
    iter0_file : str
        Name of the starting-model file used with ``start_from_perturbed``.
    rng : numpy.random.Generator, optional
        Shared random generator.
    out : bool
        If True, print status messages and the prior-perturbation diagnostics.
    **legacy
        Removed arguments of the earlier randomized-SVD implementation
        (``n_eig``, ``n_oversampling``, ``n_power_iter``, ``sigma2_residual``,
        ``lam_mode``, ``solver_method``, ``precond``) are accepted and ignored,
        with a warning, so that old scripts do not crash.

    Returns
    -------
    mod_list : list of str
        Paths to the perturbed resistivity block files.

    Notes
    -----
    A printed ``energy ratio`` = mean(alpha |R x|^2) / (rank of R) near 1 shows
    that the perturbations are drawn with the intended strength and in all
    directions.  (The former low-rank sampler gave ~ n_eig / n_free.)
    """
    if q is None:
        raise ValueError("generate_rto_model_ensemble: roughness matrix R (q) must be provided.")
    if alpha is None:
        raise ValueError(
            "generate_rto_model_ensemble: 'alpha' must be given (the regularisation "
            "weight of the member inversions, e.g. the Alpha column of femtic.cnv). "
            "The prior covariance is (alpha R^T R)^-1; the former implicit alpha = 1 "
            "gave wrong ensemble spreads."
        )
    if not (np.isfinite(alpha) and alpha > 0.0):
        raise ValueError(f"generate_rto_model_ensemble: alpha must be positive and finite, got {alpha!r}.")

    if legacy:
        warnings.warn(
            "generate_rto_model_ensemble: ignoring legacy argument(s) "
            f"{sorted(legacy)} of the removed randomized-SVD implementation.",
            DeprecationWarning, stacklevel=2,
        )

    algo_l = str(algo).strip().lower()
    if algo_l.startswith("low"):
        warnings.warn(
            "generate_rto_model_ensemble: algo='low rank' is no longer available "
            "(its randomized SVD kept the roughest modes, ~0.1-0.5 % of the prior "
            "variance); using algo='auto'.", DeprecationWarning, stacklevel=2)
        algo_l = "auto"
    elif algo_l.startswith("full"):
        warnings.warn(
            "generate_rto_model_ensemble: algo='full rank' is now algo='cg' "
            "(or 'splu'/'auto').", DeprecationWarning, stacklevel=2)
        algo_l = "cg"

    rng = default_rng() if rng is None else rng
    fromto_arr = _resolve_fromto(fromto, n_samples)

    if out:
        print(f"RTO prior sampling started (alpha={alpha:g}, solver={algo_l}).")
    samples, info = sample_rto_prior(
        q,
        n_samples=len(fromto_arr),
        alpha=float(alpha),
        lam=(float(lam) if lam and lam > 0.0 else None),
        lam_alpha=float(lam_alpha),
        method=algo_l,
        rtol=float(solver_rtol),
        rng=rng,
        return_info=True,
    )
    if remove_mean:
        Rq = q.tocsr() if issparse(q) else scipy.sparse.csr_array(np.asarray(q, dtype=np.float64))
        ones = np.ones(Rq.shape[1])
        r1 = float(np.linalg.norm(Rq @ ones))
        r1_ref = float(np.linalg.norm(abs(Rq) @ ones))
        if r1 > 1.0e-6 * max(r1_ref, 1.0e-300):
            warnings.warn(
                "generate_rto_model_ensemble: remove_mean=True, but R does not annihilate constants "
                f"(|R 1| / |abs(R) 1| = {r1 / max(r1_ref, 1e-300):.2e}); the mean removal then "
                "changes the roughness energy of the draws.", RuntimeWarning, stacklevel=2)
        samples = samples - samples.mean(axis=1, keepdims=True)

    if out:
        e_ratio = prior_energy_ratio(q, samples, alpha=float(alpha))
        if len(samples) > 1:
            rms_std = float(np.sqrt(np.mean(np.var(samples, axis=0, ddof=1))))
            zm = samples - samples.mean(axis=1, keepdims=True)
            rms_zm = float(np.sqrt(np.mean(np.var(zm, axis=0, ddof=1))))
        else:
            rms_std = rms_zm = float("nan")
        print(f"RTO prior sampling done: n={samples.shape[1]}, lam={info['lam']:.3e}, "
              f"solver={info['method']}, RMS std = {rms_std:.4f} (log10 units; "
              f"{rms_zm:.4f} after removing each draw's global mean), "
              f"energy ratio = {e_ratio:.3f} (expect ~1).")
        if not (0.5 < e_ratio < 2.0):
            warnings.warn(
                f"generate_rto_model_ensemble: prior energy ratio {e_ratio:.3f} is far from 1; "
                "check R, alpha and the solver tolerance.", RuntimeWarning, stacklevel=2)

    mod_list: list[str] = []
    for iens, sample in zip(fromto_arr, samples):
        file = f"{dir_base}{iens}/{refmod}"
        orig_file = file.replace(".dat", "_orig.dat")
        shutil.copy(file, orig_file)

        # Read reference log10-resistivity from the unmodified backup.
        # sample is a zero-mean perturbation in log10 space.
        # method="add": perturbed model = reference + perturbation (correct RTO usage).
        # method="replace": sample IS the full log10 model (absolute values).
        ref_log10 = fem.read_model(orig_file, model_trans="log10", out=False)
        if ref_log10.size != sample.size:
            raise ValueError(
                f"generate_rto_model_ensemble: R has {sample.size} columns but the free "
                f"region vector of {orig_file} has {ref_log10.size} entries; R rows/columns "
                "must correspond to the free regions in region-index order."
            )
        if method.lower() == "add":
            model_log10 = ref_log10 + sample
        else:  # "replace"
            model_log10 = sample

        fem.insert_model(
            template=orig_file,
            model=model_log10,
            model_file=file,
            model_name=f"sample{iens}")

        if start_from_perturbed:
            iter0 = f"{dir_base}{iens}/{iter0_file}"
            if os.path.abspath(iter0) != os.path.abspath(file):
                # Remove first: if iter0 is a symlink to the shared template, copying
                # onto it would write through the link and modify the template.
                if os.path.islink(iter0) or os.path.exists(iter0):
                    os.remove(iter0)
                shutil.copy2(file, iter0)

        mod_list.append(file)

    if out:
        print("\nlist of perturbed model files:")
        print(mod_list)

    return mod_list


def _find_extrema_pilot_points(
    cx: np.ndarray,
    cy: np.ndarray,
    cz: np.ndarray,
    ref_log_rho: np.ndarray,
    k: int = 30,
    which: str = "both",
    roi: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Return pilot-point coordinates at local log₁₀(ρ) extrema of a model.

    For each free region whose barycentre lies inside *roi* (or everywhere
    if *roi* is None), the region is declared a local minimum (maximum) if
    its log₁₀(ρ) value is **strictly less (greater) than all k−1 nearest
    neighbours'** values.  The neighbour search uses all free regions (not
    just those inside the ROI) so that boundary regions are not spuriously
    flagged as extrema due to a truncated neighbourhood.

    Parameters
    ----------
    cx, cy, cz : ndarray, shape (n_cells,)
        Barycentre coordinates of free mesh regions (model-local metres,
        z positive-down, FEMTIC convention).
    ref_log_rho : ndarray, shape (n_cells,)
        log₁₀(ρ) of the reference model at each free region.
    k : int
        Neighbourhood size including self (k−1 actual neighbours compared).
        Larger k produces a smoother field with fewer detected extrema.
        Recommended: 20–40 for typical FEMTIC mesh densities; increase
        further (e.g. 30+) if the test flags an excessive number of
        spurious local minima/maxima on dense or noisy meshes.
    which : {"both", "minima", "maxima"}
        Which extrema to return:
        ``"minima"``  — conductive anomaly cores (low ρ),
        ``"maxima"``  — resistive anomaly cores (high ρ),
        ``"both"``    — both (default).
    roi : sequence of 6 floats, optional
        ``[x_min, x_max, y_min, y_max, z_min, z_max]`` (metres) restricting
        the search to a subset of the model volume.  None = full extent.

    Returns
    -------
    pp_coords : ndarray, shape (M, 3)
        Barycentre coordinates of extremum regions, columns [x, y, z].
        May be empty (shape (0, 3)) if no extrema are found.

    Notes
    -----
    Requires ``scipy.spatial`` (KDTree).  This is already a transitive
    dependency of gstools and numpy, so no additional installation is needed.

    Author: Claude Sonnet 4.6 (Anthropic), 2026-06-06.
    AI-generated — review before use in production.
    """
    from scipy.spatial import KDTree

    coords = np.column_stack([cx, cy, cz])   # (n_cells, 3)

    # --- ROI mask: restricts which regions are *candidates* for extrema ---
    if roi is not None:
        roi_list = list(roi)
        in_roi = (
            (cx >= roi_list[0]) & (cx <= roi_list[1]) &
            (cy >= roi_list[2]) & (cy <= roi_list[3]) &
            (cz >= roi_list[4]) & (cz <= roi_list[5])
        )
    else:
        in_roi = np.ones(len(cx), dtype=bool)

    roi_idx = np.where(in_roi)[0]
    if roi_idx.size == 0:
        return np.empty((0, 3), dtype=float)

    roi_coords  = coords[roi_idx]            # (m, 3)
    roi_log_rho = ref_log_rho[roi_idx]       # (m,)

    # Build KD-tree on ALL free regions so boundary regions compare against
    # neighbours outside the ROI (prevents edge artefacts).
    k_eff = min(k, len(coords))
    tree = KDTree(coords)
    _, nn_idx = tree.query(roi_coords, k=k_eff)  # (m, k_eff)

    # nn_idx[:, 0] is the query point itself; [:,1:] are its neighbours.
    neighbor_vals = ref_log_rho[nn_idx[:, 1:]]   # (m, k_eff-1)

    is_min = roi_log_rho < neighbor_vals.min(axis=1)
    is_max = roi_log_rho > neighbor_vals.max(axis=1)

    if which == "minima":
        extremum_mask = is_min
    elif which == "maxima":
        extremum_mask = is_max
    else:  # "both"
        extremum_mask = is_min | is_max

    return roi_coords[extremum_mask]


def generate_gst_model_ensemble(
    alg: str = "gst",
    dir_base: str = "./ens_",
    n_samples: int = 1,
    fromto: Optional[List[int]] = None,
    # --- mesh ---
    ref_mod_file: str = "",
    mesh_file: str = "",
    # --- pilot points ---
    pp_mode: str = "random",
    n_pp: int = 100,
    pp_bbox: Optional[Sequence[float]] = None,
    pp_coords: Optional[np.ndarray] = None,
    pp_roi: Optional[Sequence[float]] = None,
    pp_extrema_k: int = 30,
    pp_extrema_which: str = "both",
    pp_regen_every: Optional[int] = None,
    pp_cache: Optional[dict] = None,
    pp_pos_offset: int = 0,
    # --- resistivity range ---
    log_rho_min: float = 0.0,
    log_rho_max: float = 4.0,
    # --- pilot-point value mode ---
    pp_value_mode: str = "uniform",
    pp_value_delta: float = 0.5,
    # --- variogram ---
    vario_model: str = "Spherical",
    vario_range: float | Sequence[float] = 20000.,
    vario_sill: float = 0.5,
    vario_nugget: float = 0.01,
    vario_angles: Optional[Sequence[float]] = None,
    # --- output ---
    output_target: str = "both",
    resistivity_file: str = "resistivity_block_iter0.dat",
    reference_file: str = "referencemodel.dat",
    rng: Optional[Generator] = None,
    out: bool = True,
    # --- reproducibility / pilot-point export ---
    save_pilot_points: bool = False,
    pilot_points_file: Optional[str] = None,
    seed: Optional[int] = None,
) -> list[str]:
    """Generate a geostatistical initial-model ensemble via pilot-point Kriging.

    For each ensemble member *i*:

    1. Place pilot points inside the survey volume (random, fixed, or mixed).
    2. Draw log₁₀(ρ) values at the pilot points, using one of two modes
       (see ``pp_value_mode``):

       - ``"uniform"``   — Uniform(log_rho_min, log_rho_max), independent of
         location (original behaviour).
       - ``"reference"`` — reference_model(pilot point) ± ``pp_value_delta``,
         i.e. the log₁₀(ρ) of the reference model at the free region nearest
         each pilot point, perturbed by Uniform(-pp_value_delta,
         +pp_value_delta).  Keeps every member close to the reference
         structure while still exploring the pilot-point/Kriging spatial
         randomness.
    3. Ordinary-Krig the values to all FEMTIC mesh cell centres (gstools).
    4. Clamp the field to [log_rho_min, log_rho_max].
    5. Write the result into the member directory as
       ``resistivity_block_iter0.dat`` and/or ``referencemodel.dat``
       (controlled by ``output_target``).

    No roughness matrix is required.  The ensemble spread comes entirely from
    the spatially random pilot-point values; its spatial character is governed
    by the variogram model.

    Parameters
    ----------
    alg : str
        Algorithm tag (informational only, default ``"gst"``).
    dir_base : str
        Ensemble base directory, e.g. ``"./ubinas_gst_"``.
    n_samples : int
        Number of ensemble members if ``fromto`` is None.
    fromto : None or list of int, optional
        Explicit member indices to process.  If None, use 0 … n_samples-1.
    ref_mod_file : str
        Full path to the template reference model file.  Used to identify
        free regions and their element membership.
    mesh_file : str
        Full path to the FEMTIC mesh file (``mesh.dat``).  Used together with
        ``ref_mod_file`` to compute per-free-region barycentres for Kriging.

    Pilot-point parameters
    ----------------------
    pp_mode : {"random", "fixed", "mixed", "extrema"}
        Pilot-point placement strategy:

        - ``"random"`` — ``n_pp`` points drawn uniformly inside ``pp_bbox``
          (fresh locations **and** fresh values every member).
        - ``"fixed"``  — locations taken from ``pp_coords`` (same geometry
          every member, only values change).
        - ``"mixed"``  — ``pp_coords`` plus ``n_pp`` additional random points.
        - ``"extrema"`` — structural skeleton at local log₁₀(ρ) minima and/or
          maxima of the reference model within ``pp_roi`` (same geometry every
          member), plus ``n_pp`` random fill points inside ``pp_bbox``
          (fresh every member).  Requires ``scipy.spatial``.
    n_pp : int
        Number of randomly drawn pilot points per member.  Used when
        ``pp_mode`` is ``"random"``, ``"mixed"``, or ``"extrema"`` (fill).
        Recommended: 50–200 for typical 3-D MT survey volumes.
    pp_bbox : sequence of 6 floats, optional
        Bounding box ``[x_min, x_max, y_min, y_max, z_min, z_max]`` (metres,
        z positive-down) for random pilot-point placement.
        ``None`` (default) uses the full free-region extent of the model
        (axis-wise min/max of the free-region barycentres actually present
        in ``mesh_file`` / ``ref_mod_file``).  If a bbox is supplied, each
        of the six values is independently clamped, axis-wise, to that same
        full-extent box — a bound wider than the true model size (e.g. a
        generic template value left over from a different survey) is simply
        truncated to the model's actual size on that axis rather than
        placing pilot points outside the free-region domain; a bound
        already inside the model extent is left unchanged.
    pp_coords : ndarray, shape (N, 3), optional
        Explicit pilot-point coordinates (easting, northing, depth).
        Required when ``pp_mode`` is ``"fixed"`` or ``"mixed"``.
    pp_roi : sequence of 6 floats, optional
        ``[x_min, x_max, y_min, y_max, z_min, z_max]`` (metres, z positive-
        down) restricting the extremum search to a sub-volume of the model.
        Only used when ``pp_mode`` is ``"extrema"``.  Resolved with the same
        None-means-full-extent / axis-wise-clamp-if-supplied rule as
        ``pp_bbox`` above.  Tip: set tighter than ``pp_bbox`` to exclude
        padding cells.
    pp_extrema_k : int
        Neighbourhood size (including self) for the local extremum test in
        ``"extrema"`` mode.  Larger k yields fewer, smoother extrema.
        Recommended: 20–40; increase further (e.g. 30+) if too many
        spurious minima/maxima are detected on dense or noisy meshes.
        Default: 30.
    pp_extrema_which : {"both", "minima", "maxima"}
        Which extrema to use as pilot-point seeds in ``"extrema"`` mode.
        ``"both"`` (default) seeds both conductive and resistive anomaly cores.
    pp_regen_every : int, optional
        If ``None`` (default), the per-member *random* pilot-point
        component keeps its existing behaviour: freshly redrawn for
        **every** member when ``pp_mode="random"`` (the whole point set)
        or for the random-fill portion of ``"mixed"``/``"extrema"``
        (skeleton/``pp_coords`` untouched either way, as always). If set
        to an integer ``K >= 1``, that random component is instead drawn
        once per **block of K consecutive processed members** (grouped
        by position within ``fromto``/``range(n_samples)``, not by
        member index — relevant only if ``fromto`` has gaps) and reused,
        unchanged, by the other ``K-1`` members in that block; a new
        draw is taken at the start of the next block. Pilot-point
        **values** (``pp_vals``) are always redrawn every member
        regardless of this setting, so members sharing a block still
        differ from each other — only the pilot-point *locations* are
        held in common within a block. This decouples geometry churn
        from value churn: e.g. with ``K=10`` you get 10 members
        exploring value-randomness alone at one random pilot-point
        layout, then a fresh layout for the next 10 -- useful for
        variance decomposition / GST parameter diagnostics that want to
        separate the two sources of ensemble spread. No effect when
        ``pp_mode="fixed"`` (there is no random component to regenerate
        — a one-line warning is printed if set in that case).
    pp_cache : dict, optional
        Caller-owned dictionary that stores the random pilot-point layout
        of the current ``pp_regen_every`` block.  Needed only when this
        function is called **repeatedly with small batches** (typically
        ``n_samples=1``) and the block structure of ``pp_regen_every`` is
        to be honoured *across* calls: the cache that ``pp_regen_every``
        uses internally is local to one call, so without ``pp_cache`` every
        single-member call would start a fresh block (= a new layout for
        every member).  Pass the **same** dict object, initially empty, to
        every call of the series, together with ``pp_pos_offset``.
        Default: ``None`` (call-local cache, original behaviour).
        Requires ``pp_regen_every`` to be set (``ValueError`` otherwise).
        Models must be requested in non-decreasing order of global index;
        blocks older than the current one are discarded.  The dict holds
        arrays of the layout only, so it is tiny and can be reused for a
        later series after ``pp_cache.clear()``.  If ``rng`` is shared
        between the calls, a series of ``n_samples=1`` calls reproduces
        one ``n_samples=N`` call exactly.
    pp_pos_offset : int
        Added to the position of each member within this call before the
        block id is computed, ``block_id = (pos + pp_pos_offset) //
        pp_regen_every``.  Set it to the **global** index of the first
        model of the call (0, 1, 2, ... for consecutive ``n_samples=1``
        calls).  Default: 0.  Has no effect when ``pp_regen_every`` is
        ``None``.

    Resistivity range
    -----------------
    log_rho_min : float
        Minimum resistivity in log₁₀(Ω·m).  Used as both the lower draw
        bound and a post-Kriging clamp.
    log_rho_max : float
        Maximum resistivity in log₁₀(Ω·m).  Used as both the upper draw
        bound and a post-Kriging clamp.
    pp_value_mode : {"uniform", "reference"}
        How pilot-point log₁₀(ρ) values are drawn:

        - ``"uniform"`` (default) — Uniform(log_rho_min, log_rho_max) at
          every pilot point, independent of location (original behaviour).
        - ``"reference"`` — reference_model(pilot point) ± ``pp_value_delta``.
          The reference log₁₀(ρ) is looked up at the free-region barycentre
          nearest each pilot point (nearest-neighbour, via ``scipy.spatial.
          KDTree``) and perturbed by Uniform(-pp_value_delta,
          +pp_value_delta).  The result is still clamped to
          [log_rho_min, log_rho_max].  Use this mode to keep the ensemble
          anchored to the reference structure rather than exploring the
          full resistivity range at each pilot point.
    pp_value_delta : float
        Half-width, in log₁₀(Ω·m), of the symmetric perturbation applied
        around the reference value when ``pp_value_mode="reference"``.
        Ignored when ``pp_value_mode="uniform"``.  Typical: 0.3–1.0
        (≈ factor 2–10 in resistivity).

    Variogram parameters
    --------------------
    vario_model : str
        gstools covariance model class name.  Common choices:
        ``"Spherical"`` (default), ``"Gaussian"``, ``"Exponential"``,
        ``"Matern"``, ``"Linear"``, ``"PowerLaw"``.
    vario_range : float or (float, float)
        Correlation length in metres.  A scalar applies isotropically.
        A 2-tuple ``(h_range, v_range)`` sets horizontal and vertical ranges
        separately (geometric anisotropy); horizontal usually >> vertical for MT.
        Recommended: h_range ≈ half the survey aperture;
        v_range ≈ half the target depth.
    vario_sill : float
        Sill (variance) in (log₁₀ Ω·m)².  Typical range 0.1–1.0.
        Recommended: 0.25–0.5 (±0.5–0.7 log₁₀ units 1-sigma).
    vario_nugget : float
        Nugget in (log₁₀ Ω·m)².  Keep ≤ 10 % of sill for coherence.
    vario_angles : sequence of float, optional
        Rotation angles ``[α, β, γ]`` in **degrees** orienting the anisotropy
        axes (converted to radians internally).  ``None`` = axis-aligned.

    Output parameters
    -----------------
    output_target : {"resistivity_block", "referencemodel", "both"}
        Which FEMTIC file(s) receive the Kriged model per member.
        ``"both"`` is recommended for a fully geostatistical prior.
    resistivity_file : str
        Filename for the initial model (default ``resistivity_block_iter0.dat``).
    reference_file : str
        Filename for the prior / reference model (default ``referencemodel.dat``).
    rng : numpy.random.Generator, optional
        Shared random generator.  If None, uses ``np.random.default_rng()``.
        Pass a generator seeded by the caller (e.g.
        ``np.random.default_rng(RANDOM_SEED)``) for a fully reproducible
        ensemble — same pilot-point locations and values every run, given
        identical templates/config.
    out : bool
        If True, print progress messages.

    Reproducibility / pilot-point export
    -------------------------------------
    save_pilot_points : bool
        If True, write every member's pilot-point coordinates and drawn
        log₁₀(ρ) values to ``pilot_points_file`` as a single compressed
        ``.npz`` after the member loop completes. Useful for reproducibility
        audits, GST parameter diagnostics, or re-plotting the pilot-point
        cloud without re-running Kriging. Default False (no extra I/O).
    pilot_points_file : str, optional
        Output path for the pilot-point archive. ``None`` (default) writes
        to ``f"{dir_base}pilot_points.npz"`` — alongside the per-member
        directories, at the ensemble root. Ignored when
        ``save_pilot_points=False``.
    seed : int, optional
        Informational only — recorded in the ``pilot_points.npz`` archive
        (key ``"seed"``, ``-1`` if not given) so the archive is
        self-describing. Does **not** seed the RNG itself; pass a seeded
        generator via ``rng`` for that. Typically the same integer used to
        construct ``rng`` in the caller.

    Returns
    -------
    mod_list : list of str
        Paths to all files written (one or two per member, depending on
        ``output_target``).

    Notes
    -----
    Requires ``gstools`` (pip install gstools).

    ``pilot_points.npz`` contents (when ``save_pilot_points=True``)
    -----------------------------------------------------------------
    ``member_ids``   : (n_members,) int — the processed member indices
                        (``fromto_arr``).
    ``pp_x/y/z``     : (n_members, n_pp_total) float — pilot-point
                        coordinates (easting, northing, depth) per member.
                        ``n_pp_total`` is constant across members for a
                        given ``pp_mode`` (fixed skeleton size + ``n_pp``
                        random fill points, where applicable).
    ``pp_vals``      : (n_members, n_pp_total) float — log₁₀(ρ) values
                        drawn at each pilot point, per member.
    ``pp_mode``, ``pp_value_mode``, ``log_rho_min``, ``log_rho_max``,
    ``vario_model``, ``vario_range``, ``vario_sill``, ``vario_nugget``,
    ``seed``, ``pp_regen_every`` (``-1`` if not set) — scalar/array
    metadata mirroring the call's configuration, for self-describing
    archives.

    Author: Volker Rath (DIAS)
    Created with the help of Claude Sonnet 4.6 (Anthropic), 2026-04-27.
    """
    try:
        import gstools as gs
    except ImportError:
        raise ImportError(
            "generate_gst_model_ensemble requires gstools.  "
            "Install with: pip install gstools"
        )

    rng = default_rng() if rng is None else rng

    # ------------------------------------------------------------------
    # Read mesh cell centres once from the reference model + mesh file.
    # Per-region barycentres: mean of element barycentres for each
    # free region, computed from mesh node coordinates and connectivity.
    # ------------------------------------------------------------------
    if not ref_mod_file:
        raise ValueError(
            "generate_gst_model_ensemble: ref_mod_file must be supplied."
        )
    if not mesh_file:
        raise ValueError(
            "generate_gst_model_ensemble: mesh_file must be supplied."
        )
    if out:
        print(f"Reading reference model from: {ref_mod_file}")
        print(f"Reading mesh from:            {mesh_file}")

    # resistivity block: elem→region map, rho, flag
    rblock    = fem.read_resistivity_block(ref_mod_file)
    elem2reg  = rblock["region_of_elem"]   # shape (nelem,), 0-based
    nreg      = int(rblock["nreg"])
    rho_reg   = rblock["region_rho"]       # shape (nreg,)
    flag_reg  = rblock["region_flag"]      # shape (nreg,)

    # mirror read_model fixed-region logic
    fixed_mask = np.zeros(nreg, dtype=bool)
    fixed_mask[0] = True                   # air always fixed
    fixed_mask |= (flag_reg == 1)
    if nreg > 1 and (flag_reg[1] == 1) and (rho_reg[1] <= 1.0):
        fixed_mask[1] = True               # ocean heuristic
    free_idx = np.where(~fixed_mask)[0]    # free region global indices

    # mesh: node coordinates and tetrahedral connectivity
    nodes, conn = fem.read_femtic_mesh(mesh_file)   # (nn,3), (nelem,4)

    # element barycentres: (nelem, 3)
    elem_bary = nodes[conn].mean(axis=1)

    # per-region barycentre = mean over elements belonging to that region
    region_bary = np.zeros((nreg, 3), dtype=float)
    for ireg in free_idx:
        mask_e = (elem2reg == ireg)
        if mask_e.any():
            region_bary[ireg] = elem_bary[mask_e].mean(axis=0)

    cx = region_bary[free_idx, 0]
    cy = region_bary[free_idx, 1]
    cz = region_bary[free_idx, 2]
    n_cells = len(free_idx)

    if out:
        print(f"  {n_cells} free regions.")

    # ------------------------------------------------------------------
    # Resolve pp_bbox / pp_roi against the true model size.
    #
    # full_extent is the axis-wise bounding box of the free-region
    # barycentres actually present in this mesh/reference-model pair —
    # i.e. the real "full model size" for pilot-point placement (excludes
    # fixed air/ocean regions).
    #
    #   pp_bbox / pp_roi = None           -> full_extent (unchanged axis
    #                                         behaviour: an all-None ROI
    #                                         already meant "no mask", which
    #                                         is equivalent to full_extent).
    #   pp_bbox / pp_roi = [xmn,...,zmx]  -> each of the six bounds is
    #                                         clamped, axis-wise, to
    #                                         full_extent; a bound that
    #                                         already lies inside the model
    #                                         is left untouched, one that
    #                                         exceeds the model size on
    #                                         that axis is truncated to it.
    # ------------------------------------------------------------------
    full_extent = [
        float(cx.min()), float(cx.max()),
        float(cy.min()), float(cy.max()),
        float(cz.min()), float(cz.max()),
    ]

    def _resolve_pp_box(box, label):
        if box is None:
            return list(full_extent)
        box = [float(v) for v in box]
        clamped = [
            max(box[0], full_extent[0]), min(box[1], full_extent[1]),
            max(box[2], full_extent[2]), min(box[3], full_extent[3]),
            max(box[4], full_extent[4]), min(box[5], full_extent[5]),
        ]
        if out and clamped != box:
            print(f"  {label} exceeds model extent — clamped axis-wise "
                  f"{box} -> {clamped}")
        return clamped

    pp_bbox = _resolve_pp_box(pp_bbox, "pp_bbox")
    pp_roi  = _resolve_pp_box(pp_roi,  "pp_roi")

    if out:
        print(f"  Model free-region extent (m): {full_extent}")
        print(f"  pp_bbox (resolved):           {pp_bbox}")

    # ------------------------------------------------------------------
    # Build gstools variogram model.
    # ------------------------------------------------------------------
    vario_cls = getattr(gs, vario_model)

    if np.isscalar(vario_range):
        len_scale = float(vario_range)
        anis = [1.0, 1.0]
    else:
        h_range, v_range = vario_range
        len_scale = float(h_range)
        anis = [1.0, float(v_range) / float(h_range)]

    vario_kwargs: dict = dict(
        dim=3,
        var=vario_sill,
        len_scale=len_scale,
        nugget=vario_nugget,
        anis=anis,
    )
    if vario_angles is not None:
        vario_kwargs["angles"] = np.deg2rad(vario_angles)

    variogram = vario_cls(**vario_kwargs)

    if out:
        print(
            f"Variogram: {vario_model}, range={vario_range} m, "
            f"sill={vario_sill}, nugget={vario_nugget}"
        )

    # ------------------------------------------------------------------
    # Reference log10(rho) at free-region barycentres (used by "extrema"
    # pilot-point placement and/or "reference" pilot-point value mode).
    # ------------------------------------------------------------------
    ref_log_rho_free = np.log10(np.maximum(rho_reg[free_idx], 1e-10))

    if pp_value_mode not in ("uniform", "reference"):
        raise ValueError(
            f"pp_value_mode must be 'uniform' or 'reference', got "
            f"'{pp_value_mode}'."
        )

    ref_tree = None
    if pp_value_mode == "reference":
        from scipy.spatial import KDTree
        ref_tree = KDTree(np.column_stack([cx, cy, cz]))
        if out:
            print(
                f"Pilot-point values: reference model ± {pp_value_delta} "
                f"(log10 Ohm.m)"
            )
    elif out:
        print(
            f"Pilot-point values: Uniform({log_rho_min}, {log_rho_max}) "
            f"(log10 Ohm.m)"
        )

    # ------------------------------------------------------------------
    # Validate fixed pilot-point coordinates if needed.
    # ------------------------------------------------------------------
    if pp_mode in ("fixed", "mixed"):
        if pp_coords is None:
            raise ValueError(
                f"pp_coords must be supplied when pp_mode='{pp_mode}'."
            )
        pp_fixed = np.asarray(pp_coords, dtype=float)
        if pp_fixed.ndim != 2 or pp_fixed.shape[1] != 3:
            raise ValueError("pp_coords must have shape (N, 3).")
    else:
        pp_fixed = np.empty((0, 3), dtype=float)

    # pp_bbox already resolved to a plain list of 6 floats above.

    # ------------------------------------------------------------------
    # Pre-compute extrema pilot-point skeleton (same geometry each member).
    # Only used when pp_mode == "extrema".
    # ------------------------------------------------------------------
    if pp_mode == "extrema":
        pp_extrema = _find_extrema_pilot_points(
            cx=cx, cy=cy, cz=cz,
            ref_log_rho=ref_log_rho_free,
            k=pp_extrema_k,
            which=pp_extrema_which,
            roi=pp_roi,
        )
        n_extrema = len(pp_extrema)
        if out:
            print(
                f"  Extrema pilot-point skeleton: {n_extrema} points "
                f"({pp_extrema_which}, k={pp_extrema_k})"
            )
        if n_extrema == 0:
            warnings.warn(
                "generate_gst_model_ensemble: no extrema found in ROI — "
                "falling back to pure random pilot-point placement.",
                RuntimeWarning,
                stacklevel=2,
            )
            pp_mode = "random"
    else:
        pp_extrema = np.empty((0, 3), dtype=float)

    # ------------------------------------------------------------------
    # Validate pp_regen_every and set up the block-caching helper for the
    # random pilot-point component. See the pp_regen_every docstring
    # entry above for the semantics.
    # ------------------------------------------------------------------
    if pp_cache is not None and pp_regen_every is None:
        raise ValueError(
            "pp_cache was given but pp_regen_every is None; the cache would "
            "be ignored. Set pp_regen_every (>= 1) or drop pp_cache."
        )
    if pp_regen_every is not None:
        if int(pp_regen_every) < 1:
            raise ValueError(
                f"pp_regen_every must be >= 1 (or None), got {pp_regen_every!r}."
            )
        pp_regen_every = int(pp_regen_every)
        if pp_mode == "fixed":
            print(f"  WARNING: pp_regen_every={pp_regen_every} has no effect "
                  f"with pp_mode='fixed' — there is no random pilot-point "
                  f"component to regenerate (locations come entirely from "
                  f"pp_coords).")
        elif out:
            print(f"  Pilot-point locations: random component regenerated "
                  f"every {pp_regen_every} member(s) (pp_mode='{pp_mode}'); "
                  f"pilot-point values still redrawn every member.")

    #: (block_id -> (rnd_x, rnd_y, rnd_z)) cache used only when
    #: pp_regen_every is set; block_id = position-in-fromto_arr //
    #: pp_regen_every, so members sharing a block reuse the same random
    #: pilot-point locations while still drawing fresh values each time.
    #: A caller-supplied pp_cache replaces the call-local dict so that the
    #: block structure survives across repeated n_samples=1 calls (see the
    #: pp_cache / pp_pos_offset docstring entries).
    _regen_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = (
        pp_cache if pp_cache is not None else {}
    )
    _pos_offset = int(pp_pos_offset)

    def _draw_or_reuse_random_component(pos: int):
        """(n_pp,) uniform draws in pp_bbox -- fresh every call if
        pp_regen_every is None (original behaviour), otherwise cached
        per block of pp_regen_every consecutive (global) positions."""
        if pp_regen_every is None:
            return (
                rng.uniform(pp_bbox[0], pp_bbox[1], n_pp),
                rng.uniform(pp_bbox[2], pp_bbox[3], n_pp),
                rng.uniform(pp_bbox[4], pp_bbox[5], n_pp),
            )
        block_id = (pos + _pos_offset) // pp_regen_every
        if block_id not in _regen_cache:
            # Models arrive in non-decreasing order, so older blocks are
            # never needed again: drop them to keep the cache O(1).
            for _old in [k for k in _regen_cache if k < block_id]:
                del _regen_cache[_old]
            _regen_cache[block_id] = (
                rng.uniform(pp_bbox[0], pp_bbox[1], n_pp),
                rng.uniform(pp_bbox[2], pp_bbox[3], n_pp),
                rng.uniform(pp_bbox[4], pp_bbox[5], n_pp),
            )
        return _regen_cache[block_id]

    # ------------------------------------------------------------------
    # Member loop.
    # ------------------------------------------------------------------
    fromto_arr = _resolve_fromto(fromto, n_samples)

    if out:
        print(f"\nGenerating {len(fromto_arr)} geostatistical initial models ...")

    mod_list: list[str] = []

    # Accumulators for the optional pilot-point archive (save_pilot_points).
    # Lists of (n_pp_total,) arrays, one entry appended per member; stacked
    # into (n_members, n_pp_total) after the loop. n_pp_total is constant
    # across members for a given pp_mode, so stacking is always safe.
    _pp_x_all: list[np.ndarray] = []
    _pp_y_all: list[np.ndarray] = []
    _pp_z_all: list[np.ndarray] = []
    _pp_vals_all: list[np.ndarray] = []

    for _pos, iens in enumerate(fromto_arr):
        member_dir = f"{dir_base}{iens}/"

        # --- pilot-point locations ---
        if pp_mode == "random":
            pp_x, pp_y, pp_z = _draw_or_reuse_random_component(_pos)
        elif pp_mode == "fixed":
            pp_x = pp_fixed[:, 0]
            pp_y = pp_fixed[:, 1]
            pp_z = pp_fixed[:, 2]
        elif pp_mode == "mixed":
            rnd_x, rnd_y, rnd_z = _draw_or_reuse_random_component(_pos)
            pp_x = np.concatenate([pp_fixed[:, 0], rnd_x])
            pp_y = np.concatenate([pp_fixed[:, 1], rnd_y])
            pp_z = np.concatenate([pp_fixed[:, 2], rnd_z])
        else:  # "extrema"
            # Fixed structural skeleton (same geometry every member) plus
            # n_pp random fill points -- fresh every member by default,
            # or regenerated every pp_regen_every members instead.
            rnd_x, rnd_y, rnd_z = _draw_or_reuse_random_component(_pos)
            pp_x = np.concatenate([pp_extrema[:, 0], rnd_x])
            pp_y = np.concatenate([pp_extrema[:, 1], rnd_y])
            pp_z = np.concatenate([pp_extrema[:, 2], rnd_z])

        # --- log10(rho) values at pilot points ---
        if pp_value_mode == "reference":
            _, nn_idx = ref_tree.query(np.column_stack([pp_x, pp_y, pp_z]))
            pp_ref = ref_log_rho_free[nn_idx]
            pp_vals = pp_ref + rng.uniform(
                -pp_value_delta, pp_value_delta, len(pp_x)
            )
            pp_vals = np.clip(pp_vals, log_rho_min, log_rho_max)
        else:  # "uniform"
            pp_vals = rng.uniform(log_rho_min, log_rho_max, len(pp_x))

        if save_pilot_points:
            _pp_x_all.append(pp_x)
            _pp_y_all.append(pp_y)
            _pp_z_all.append(pp_z)
            _pp_vals_all.append(pp_vals)

        # --- Ordinary Kriging ---
        krig = gs.krige.Ordinary(
            model=variogram,
            cond_pos=(pp_x, pp_y, pp_z),
            cond_val=pp_vals,
        )
        krig_field, _ = krig((cx, cy, cz))

        # --- clamp ---
        krig_field = np.clip(krig_field, log_rho_min, log_rho_max)

        # --- write output file(s) ---
        # Resolve the template path once so that writing resistivity_block
        # first (in the "both" branch) cannot clobber the template before
        # the referencemodel write reads it.
        template_path = member_dir + reference_file

        if output_target in ("resistivity_block", "both"):
            out_path = member_dir + resistivity_file
            fem.insert_model(
                template=template_path,
                model=krig_field,
                model_file=out_path,
                model_name=f"gst_sample{iens}",
            )
            mod_list.append(out_path)

        if output_target in ("referencemodel", "both"):
            out_path = member_dir + reference_file
            fem.insert_model(
                template=template_path,
                model=krig_field,
                model_file=out_path,
                model_name=f"gst_sample{iens}",
            )
            if out_path not in mod_list:
                mod_list.append(out_path)

        if out and ((iens - fromto_arr[0] + 1) % 10 == 0
                    or iens == fromto_arr[-1]):
            print(f"  member {iens + 1:>4d}/{fromto_arr[-1] + 1} done.")

    if out:
        print("\nlist of written model files:")
        print(mod_list)

    if save_pilot_points:
        _pp_file = (pilot_points_file if pilot_points_file is not None
                    else f"{dir_base}pilot_points.npz")
        _pp_x_arr    = np.asarray(_pp_x_all)      # (n_members, n_pp_total)
        _pp_y_arr    = np.asarray(_pp_y_all)
        _pp_z_arr    = np.asarray(_pp_z_all)
        _pp_vals_arr = np.asarray(_pp_vals_all)
        np.savez_compressed(
            _pp_file,
            member_ids=fromto_arr,
            pp_x=_pp_x_arr,
            pp_y=_pp_y_arr,
            pp_z=_pp_z_arr,
            pp_vals=_pp_vals_arr,
            pp_mode=pp_mode,
            pp_value_mode=pp_value_mode,
            log_rho_min=log_rho_min,
            log_rho_max=log_rho_max,
            vario_model=vario_model,
            vario_range=np.asarray(vario_range, dtype=float),
            vario_sill=vario_sill,
            vario_nugget=vario_nugget,
            seed=seed if seed is not None else -1,
            pp_regen_every=pp_regen_every if pp_regen_every is not None else -1,
        )
        if out:
            print(f"\nPilot points written: {_pp_file}  "
                  f"({_pp_x_arr.shape[0]} members x {_pp_x_arr.shape[1]} points)")

    return mod_list


# =============================================================================
# Section: Mesh-agnostic GST perturbation machine
# =============================================================================
#
# generate_gst_model_ensemble() (above) couples three separable concerns:
#
#   (a) pilot-point placement / value-drawing strategy
#       (pp_mode: random | fixed | mixed | extrema;
#        pp_value_mode: uniform | reference)
#   (b) Ordinary Kriging of pilot-point values onto a target point cloud
#   (c) FEMTIC-specific I/O — unstructured tetrahedral mesh, free-region
#       barycentres via fem.read_femtic_mesh / fem.read_resistivity_block,
#       per-member directory tree, resistivity_block file writing.
#
# The functions below factor (a)+(b) out into a mesh-agnostic "perturbation
# machine" that only needs a target point cloud ``target_xyz`` — it has no
# knowledge of FEMTIC's tetrahedral mesh, ModEM's rectilinear grid, or any
# other mesh geometry.  A single realisation is returned directly as an
# array (no directory tree, no file I/O), which is what a null-space-shuttle
# script needs: one candidate perturbation per call.
#
# generate_gst_model_ensemble() itself is left untouched (it is production
# code with its own file-based ensemble workflow); the new functions are
# additive.  A future cleanup could reimplement its per-member loop as a
# thin wrapper around generate_gst_perturbation(), but that is not done here
# to avoid touching working FEMTIC code paths.
#
# Used by:
#   - femtic_nss.py  (_make_perturbation_gst can be pointed at this instead
#     of round-tripping through generate_gst_model_ensemble + a temp dir)
#   - modem_nss.py   (ModEM has no unstructured mesh / free-region concept;
#     it calls this directly with rectilinear grid cell centres — see
#     modem_gst_cell_centers() / generate_gst_perturbation_modem() below)
#
# Author: Volker Rath (DIAS) / Claude Sonnet 5 (Anthropic), 2026-07-19.
# =============================================================================


def krige_pilot_points_to_targets(
    target_xyz: np.ndarray,
    pp_xyz: np.ndarray,
    pp_vals: np.ndarray,
    variogram,
    log_rho_min: float,
    log_rho_max: float,
) -> np.ndarray:
    """Ordinary-Krig pilot-point values onto an arbitrary target point cloud.

    Mesh-agnostic: ``target_xyz`` may be FEMTIC free-region barycentres,
    ModEM rectilinear cell centres, or any other point cloud.

    Parameters
    ----------
    target_xyz : ndarray, shape (n_targets, 3)
        Target coordinates (x, y, z) — metres, z convention consistent with
        ``pp_xyz`` (FEMTIC / ModEM both use z positive-down).
    pp_xyz : ndarray, shape (n_pp, 3)
        Pilot-point coordinates.
    pp_vals : ndarray, shape (n_pp,)
        Pilot-point log₁₀(ρ) values.
    variogram : gstools CovModel instance
        Already-constructed variogram (e.g. ``gstools.Spherical(...)``).
    log_rho_min, log_rho_max : float
        Clamp applied to the Kriged field.

    Returns
    -------
    field : ndarray, shape (n_targets,)
        Kriged and clamped log₁₀(ρ) field at ``target_xyz``.

    Author: Claude Sonnet 5 (Anthropic), 2026-07-19.
    AI-generated — review before use in production.
    """
    try:
        import gstools as gs
    except ImportError:
        raise ImportError(
            "krige_pilot_points_to_targets requires gstools.  "
            "Install with: pip install gstools"
        )

    krig = gs.krige.Ordinary(
        model=variogram,
        cond_pos=(pp_xyz[:, 0], pp_xyz[:, 1], pp_xyz[:, 2]),
        cond_val=pp_vals,
    )
    field, _ = krig((target_xyz[:, 0], target_xyz[:, 1], target_xyz[:, 2]))
    return np.clip(field, log_rho_min, log_rho_max)


def _draw_pilot_points(
    pp_mode: str,
    n_pp: int,
    pp_bbox: Sequence[float],
    pp_coords: Optional[np.ndarray],
    pp_extrema: np.ndarray,
    rng: Generator,
) -> np.ndarray:
    """Draw pilot-point locations for one realisation (shared by all callers).

    Same placement semantics as ``generate_gst_model_ensemble``'s member
    loop: ``"random"`` / ``"fixed"`` / ``"mixed"`` / ``"extrema"``.

    Returns
    -------
    pp_xyz : ndarray, shape (n, 3)
    """
    pp_bbox = list(pp_bbox)

    if pp_mode == "random":
        x = rng.uniform(pp_bbox[0], pp_bbox[1], n_pp)
        y = rng.uniform(pp_bbox[2], pp_bbox[3], n_pp)
        z = rng.uniform(pp_bbox[4], pp_bbox[5], n_pp)
        return np.column_stack([x, y, z])

    if pp_mode in ("fixed", "mixed"):
        if pp_coords is None:
            raise ValueError(f"pp_coords must be supplied when pp_mode='{pp_mode}'.")
        fixed = np.asarray(pp_coords, dtype=float)
        if fixed.ndim != 2 or fixed.shape[1] != 3:
            raise ValueError("pp_coords must have shape (N, 3).")
        if pp_mode == "fixed":
            return fixed
        rnd_x = rng.uniform(pp_bbox[0], pp_bbox[1], n_pp)
        rnd_y = rng.uniform(pp_bbox[2], pp_bbox[3], n_pp)
        rnd_z = rng.uniform(pp_bbox[4], pp_bbox[5], n_pp)
        return np.vstack([fixed, np.column_stack([rnd_x, rnd_y, rnd_z])])

    if pp_mode == "extrema":
        rnd_x = rng.uniform(pp_bbox[0], pp_bbox[1], n_pp)
        rnd_y = rng.uniform(pp_bbox[2], pp_bbox[3], n_pp)
        rnd_z = rng.uniform(pp_bbox[4], pp_bbox[5], n_pp)
        return np.vstack([pp_extrema, np.column_stack([rnd_x, rnd_y, rnd_z])])

    raise ValueError(
        f"pp_mode must be 'random', 'fixed', 'mixed', or 'extrema', got '{pp_mode}'."
    )


def generate_gst_perturbation(
    target_xyz: np.ndarray,
    *,
    pp_mode: str = "random",
    n_pp: int = 100,
    pp_bbox: Sequence[float] = (-50000., 50000., -50000., 50000., 0., 80000.),
    pp_coords: Optional[np.ndarray] = None,
    pp_roi: Optional[Sequence[float]] = None,
    pp_extrema_k: int = 30,
    pp_extrema_which: str = "both",
    log_rho_min: float = 0.0,
    log_rho_max: float = 4.0,
    pp_value_mode: str = "uniform",
    pp_value_delta: float = 0.5,
    ref_log_rho: Optional[np.ndarray] = None,
    vario_model: str = "Spherical",
    vario_range: float | Sequence[float] = 20000.,
    vario_sill: float = 0.5,
    vario_nugget: float = 0.01,
    vario_angles: Optional[Sequence[float]] = None,
    rng: Optional[Generator] = None,
    out: bool = True,
) -> np.ndarray:
    """Generate one geostatistical (pilot-point Kriging) field on a target grid.

    Mesh-agnostic generalisation of the per-member body of
    ``generate_gst_model_ensemble``: given nothing but a target point cloud
    ``target_xyz``, place pilot points, draw log₁₀(ρ) values at them, Krig
    to every target point, and return the resulting field directly (no
    file I/O, no directory tree).  This is the "perturbation machine" used
    by null-space-shuttle scripts (``femtic_nss.py``, ``modem_nss.py``) to
    generate individual perturbation candidates for null-space projection.

    Parameters
    ----------
    target_xyz : ndarray, shape (n_targets, 3)
        Target coordinates (easting, northing, depth), z positive-down.
        For FEMTIC: free-region barycentres.  For ModEM: rectilinear grid
        cell centres (see ``modem_gst_cell_centers``).
    pp_mode, n_pp, pp_bbox, pp_coords, pp_roi, pp_extrema_k, pp_extrema_which :
        Pilot-point placement — identical semantics to
        ``generate_gst_model_ensemble`` (see its docstring).
    log_rho_min, log_rho_max : float
        Draw bounds and post-Kriging clamp, log₁₀(Ω·m).
    pp_value_mode : {"uniform", "reference"}
        ``"uniform"`` draws Uniform(log_rho_min, log_rho_max) at each pilot
        point.  ``"reference"`` draws ``ref_log_rho(nearest target) ±
        pp_value_delta`` — requires ``ref_log_rho``.
    pp_value_delta : float
        Half-width of the perturbation around the reference value (only
        used when ``pp_value_mode="reference"``).
    ref_log_rho : ndarray, shape (n_targets,), optional
        Reference log₁₀(ρ) at each ``target_xyz`` row.  Required when
        ``pp_mode="extrema"`` or ``pp_value_mode="reference"``.
    vario_model, vario_range, vario_sill, vario_nugget, vario_angles :
        Variogram parameters — identical semantics to
        ``generate_gst_model_ensemble``.
    rng : numpy.random.Generator, optional
        Shared random generator.  If None, uses ``np.random.default_rng()``.
    out : bool
        If True, print a short progress summary.

    Returns
    -------
    field : ndarray, shape (n_targets,)
        Kriged, clamped log₁₀(ρ) field at ``target_xyz``.

    Notes
    -----
    Requires ``gstools``.  For a perturbation *delta* to feed into a
    null-space shuttle, subtract a reference model from the returned field,
    e.g. ``dm = generate_gst_perturbation(...) - m_ref``.

    Author: Volker Rath (DIAS) / Claude Sonnet 5 (Anthropic), 2026-07-19.
    AI-generated — review before use in production.
    """
    rng = default_rng() if rng is None else rng
    target_xyz = np.asarray(target_xyz, dtype=float)
    cx, cy, cz = target_xyz[:, 0], target_xyz[:, 1], target_xyz[:, 2]

    if pp_value_mode not in ("uniform", "reference"):
        raise ValueError(
            f"pp_value_mode must be 'uniform' or 'reference', got '{pp_value_mode}'."
        )
    if pp_value_mode == "reference" and ref_log_rho is None:
        raise ValueError("ref_log_rho must be supplied when pp_value_mode='reference'.")
    if pp_mode == "extrema" and ref_log_rho is None:
        raise ValueError("ref_log_rho must be supplied when pp_mode='extrema'.")

    # --- variogram ---
    try:
        import gstools as gs
    except ImportError:
        raise ImportError(
            "generate_gst_perturbation requires gstools.  "
            "Install with: pip install gstools"
        )
    vario_cls = getattr(gs, vario_model)
    if np.isscalar(vario_range):
        len_scale, anis = float(vario_range), [1.0, 1.0]
    else:
        h_range, v_range = vario_range
        len_scale, anis = float(h_range), [1.0, float(v_range) / float(h_range)]
    vario_kwargs = dict(dim=3, var=vario_sill, len_scale=len_scale,
                         nugget=vario_nugget, anis=anis)
    if vario_angles is not None:
        vario_kwargs["angles"] = np.deg2rad(vario_angles)
    variogram = vario_cls(**vario_kwargs)

    # --- pilot-point locations ---
    pp_extrema = np.empty((0, 3), dtype=float)
    if pp_mode == "extrema":
        pp_extrema = _find_extrema_pilot_points(
            cx=cx, cy=cy, cz=cz, ref_log_rho=ref_log_rho,
            k=pp_extrema_k, which=pp_extrema_which, roi=pp_roi,
        )
        if len(pp_extrema) == 0:
            warnings.warn(
                "generate_gst_perturbation: no extrema found in ROI — "
                "falling back to pure random pilot-point placement.",
                RuntimeWarning, stacklevel=2,
            )
            pp_mode = "random"

    pp_xyz = _draw_pilot_points(pp_mode, n_pp, pp_bbox, pp_coords, pp_extrema, rng)

    # --- pilot-point values ---
    if pp_value_mode == "reference":
        from scipy.spatial import KDTree
        tree = KDTree(target_xyz)
        _, nn_idx = tree.query(pp_xyz)
        pp_vals = ref_log_rho[nn_idx] + rng.uniform(-pp_value_delta, pp_value_delta, len(pp_xyz))
        pp_vals = np.clip(pp_vals, log_rho_min, log_rho_max)
    else:
        pp_vals = rng.uniform(log_rho_min, log_rho_max, len(pp_xyz))

    field = krige_pilot_points_to_targets(
        target_xyz, pp_xyz, pp_vals, variogram, log_rho_min, log_rho_max,
    )

    if out:
        print(
            f"generate_gst_perturbation: {len(pp_xyz)} pilot points "
            f"({pp_mode}/{pp_value_mode}) -> {len(target_xyz)} targets, "
            f"field range [{field.min():.3f}, {field.max():.3f}]"
        )

    return field


# -----------------------------------------------------------------------
# Structured-grid (ModEM) helpers
# -----------------------------------------------------------------------

def modem_gst_cell_centers(
    dx: np.ndarray,
    dy: np.ndarray,
    dz: np.ndarray,
    free_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre coordinates for a ModEM rectilinear grid.

    ModEM has no unstructured mesh / free-region concept: cell centres
    follow directly from the cumulative cell widths ``dx, dy, dz`` (as
    read by ``modem.read_mod``), relative to the model's own local origin
    (x=0, y=0 at the northwest corner in the ModEM convention; z=0 at the
    surface, positive down).  This is the ModEM analogue of the
    tetrahedral-mesh barycentre computation used for FEMTIC in
    ``generate_gst_model_ensemble``.

    Parameters
    ----------
    dx, dy, dz : ndarray, shape (nx,), (ny,), (nz,)
        Cell widths in metres, as returned by ``modem.read_mod``.
    free_mask : ndarray of bool, shape (nx, ny, nz), optional
        True for cells to include (typically ``~aircells``).  If None, all
        cells are included.

    Returns
    -------
    xyz : ndarray, shape (n_free, 3)
        Cell-centre coordinates [x, y, z] (z positive-down) of the
        selected (free) cells, in the same flattening order as
        ``mval[free_mask]``.
    flat_idx : ndarray, shape (n_free,)
        Linear (C-order) indices into the flattened (nx*ny*nz,) model
        array, so a field returned by ``generate_gst_perturbation`` can be
        scattered back with
        ``full = np.full(nx*ny*nz, fill); full[flat_idx] = field``.

    Author: Claude Sonnet 5 (Anthropic), 2026-07-19.
    AI-generated — review before use in production, in particular the
    origin/sign convention against your ModEM coordinate setup.
    """
    x_edges = np.concatenate([[0.0], np.cumsum(dx)])
    y_edges = np.concatenate([[0.0], np.cumsum(dy)])
    z_edges = np.concatenate([[0.0], np.cumsum(dz)])
    cx1d = 0.5 * (x_edges[:-1] + x_edges[1:])
    cy1d = 0.5 * (y_edges[:-1] + y_edges[1:])
    cz1d = 0.5 * (z_edges[:-1] + z_edges[1:])

    nx, ny, nz = len(dx), len(dy), len(dz)
    CX, CY, CZ = np.meshgrid(cx1d, cy1d, cz1d, indexing="ij")

    if free_mask is None:
        free_mask = np.ones((nx, ny, nz), dtype=bool)

    flat_idx = np.flatnonzero(free_mask)
    xyz = np.column_stack([CX.ravel()[flat_idx],
                            CY.ravel()[flat_idx],
                            CZ.ravel()[flat_idx]])
    return xyz, flat_idx


def generate_gst_perturbation_modem(
    dx: np.ndarray,
    dy: np.ndarray,
    dz: np.ndarray,
    ref_log_rho: np.ndarray,
    aircells: Optional[np.ndarray] = None,
    **gst_kwargs,
) -> np.ndarray:
    """ModEM structured-grid wrapper around ``generate_gst_perturbation``.

    Computes rectilinear cell centres via ``modem_gst_cell_centers``,
    restricted to non-air cells, then Krigs a single geostatistical
    realisation onto them via the mesh-agnostic perturbation machine.
    Air cells are passed through unchanged from ``ref_log_rho``.

    Parameters
    ----------
    dx, dy, dz : ndarray
        Cell widths (metres), as returned by ``modem.read_mod``.
    ref_log_rho : ndarray, shape (nx, ny, nz)
        Reference model in log₁₀(ρ) (or loge — see Notes), used as the
        pass-through value for air cells and, if
        ``pp_value_mode="reference"`` or ``pp_mode="extrema"``, as the
        anchor field for pilot-point values / extremum detection.
    aircells : ndarray of bool, shape (nx, ny, nz), optional
        True for air (or otherwise fixed) cells to exclude from Kriging.
        If None, all cells are treated as free.
    **gst_kwargs
        Forwarded to ``generate_gst_perturbation`` (pp_mode, n_pp, pp_bbox,
        log_rho_min, log_rho_max, vario_model, vario_range, vario_sill,
        vario_nugget, vario_angles, rng, out, ...).

    Returns
    -------
    field : ndarray, shape (nx, ny, nz)
        Kriged model: free cells replaced by the Kriged values, air cells
        unchanged from ``ref_log_rho``.

    Notes
    -----
    ``modem.read_mod`` stores models in natural-log (``trans="LOGE"``) by
    convention in this codebase, whereas the FEMTIC GST tooling and its
    ``log_rho_min``/``log_rho_max``/``vario_sill`` defaults are expressed
    in log10(Ω·m).  Convert with ``rho_log10 = rho_loge / np.log(10.0)``
    before calling, and convert the returned field back
    (``rho_loge = rho_log10 * np.log(10.0)``) — or pass ``log_rho_min`` /
    ``log_rho_max`` / ``vario_sill`` already rescaled for natural-log units
    if you prefer to work in loge throughout.  This conversion is left to
    the caller (see ``modem_nss.py``) rather than hard-coded here, since it
    depends on which transform the rest of the pipeline uses.

    Author: Volker Rath (DIAS) / Claude Sonnet 5 (Anthropic), 2026-07-19.
    AI-generated — review before use in production.
    """
    nx, ny, nz = ref_log_rho.shape
    free_mask = ~aircells if aircells is not None else np.ones((nx, ny, nz), dtype=bool)

    xyz, flat_idx = modem_gst_cell_centers(dx, dy, dz, free_mask=free_mask)
    ref_flat = ref_log_rho.ravel(order="C")[flat_idx]

    field_free = generate_gst_perturbation(
        xyz, ref_log_rho=ref_flat, **gst_kwargs,
    )

    out_flat = ref_log_rho.ravel(order="C").copy()
    out_flat[flat_idx] = field_free
    return out_flat.reshape(nx, ny, nz, order="C")


# =============================================================================
# Section: GST parameter estimation
# =============================================================================
#
# Four complementary strategies for choosing variogram parameters and
# pilot-point layouts before committing to a full GST ensemble run.
#
# Strategy 1  gst_variogram_from_rto_samples
#   Fit a gstools variogram to the spatial covariance already encoded in
#   existing RTO samples.  Makes GST statistically consistent with RTO.
#
# Strategy 2  gst_pilot_point_cv
#   Leave-one-out cross-validation of Ordinary Kriging at the pilot points
#   using the reference model as the "truth".  Scores variogram parameter
#   sets without running FEMTIC.
#
# Strategy 3  gst_sill_from_jacobian
#   Linearised propagation of model variance to data space via the Jacobian
#   J.  Calibrates the sill so that the ensemble forward-response spread
#   matches the observed data error level.
#
# Strategy 4  gst_parameter_diagnostics
#   Integrating diagnostic: given an already-generated GST sample array,
#   computes ensemble std maps, back-fits an empirical variogram, evaluates
#   CV scores at pilot points, and optionally propagates spread to data
#   space (if J is provided).  Use to score and compare parameter sets.
# =============================================================================


def gst_variogram_from_rto_samples(
    samples: np.ndarray,
    coords: np.ndarray,
    vario_model: str = "Spherical",
    n_bins: int = 30,
    max_lag: Optional[float] = None,
    fit_kwargs: Optional[dict] = None,
    out: bool = True,
) -> tuple:
    """Fit a gstools variogram model to an array of RTO (or GST) samples.

    Strategy 1: use the spatial covariance structure already present in RTO
    samples to derive variogram parameters for a geostatistically consistent
    GST ensemble.

    The empirical variogram is estimated from the sample ensemble using
    ``gstools.vario_estimate``, then fitted via ``model.fit_variogram``.

    Parameters
    ----------
    samples : ndarray, shape (n_samples, n_cells)
        Log10-resistivity perturbations (zero-mean) or absolute values.
        Rows are ensemble members; columns correspond to ``coords``.
    coords : ndarray, shape (n_cells, 3)
        Spatial coordinates (easting, northing, depth) of the free mesh
        cells, in metres.  Must match the column order of ``samples``.
    vario_model : str
        gstools covariance model class name to fit, e.g. ``"Spherical"``,
        ``"Gaussian"``, ``"Exponential"``.  Default ``"Spherical"``.
    n_bins : int
        Number of lag bins for the empirical variogram.  Default 30.
    max_lag : float, optional
        Maximum lag distance in metres.  If None, defaults to half the
        maximum pairwise distance estimated from the coordinate range.
    fit_kwargs : dict, optional
        Extra keyword arguments forwarded to ``model.fit_variogram``
        (e.g. ``{"nugget": False}`` to fix the nugget at zero).
    out : bool
        If True, print fitted model parameters.

    Returns
    -------
    model : gstools covariance model
        Fitted gstools model instance.  Access fitted parameters via
        ``model.var``, ``model.len_scale``, ``model.nugget``,
        ``model.anis``.
    bin_centers : ndarray, shape (n_bins,)
        Lag bin centres used for the empirical estimate (metres).
    gamma : ndarray, shape (n_bins,)
        Empirical semi-variogram values at each bin centre.

    Notes
    -----
    Requires ``gstools`` (``pip install gstools``).

    The empirical variogram is computed from pair-wise differences over all
    ensemble members simultaneously.  For large meshes (> 50 k cells) the
    pairwise distance computation can be expensive; set ``max_lag`` to limit
    the contributing pairs.

    Author: Volker Rath (DIAS)
    Created with the help of Claude Sonnet 4.6 (Anthropic), 2026-05-28.
    """
    try:
        import gstools as gs
    except ImportError:
        raise ImportError(
            "gst_variogram_from_rto_samples requires gstools.  "
            "Install with: pip install gstools"
        )

    samples = np.asarray(samples, dtype=float)
    coords  = np.asarray(coords,  dtype=float)

    if samples.ndim != 2:
        raise ValueError("samples must be 2-D (n_samples, n_cells).")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (n_cells, 3).")
    if samples.shape[1] != coords.shape[0]:
        raise ValueError(
            f"samples.shape[1]={samples.shape[1]} does not match "
            f"coords.shape[0]={coords.shape[0]}."
        )

    cx, cy, cz = coords[:, 0], coords[:, 1], coords[:, 2]

    if max_lag is None:
        span = np.array([cx.ptp(), cy.ptp(), cz.ptp()])
        max_lag = 0.5 * float(np.linalg.norm(span))

    bin_edges = np.linspace(0.0, max_lag, n_bins + 1)

    # Remove ensemble mean per cell before computing variogram so that
    # the result reflects spatial correlation, not a DC offset.
    samples_dm = samples - samples.mean(axis=0, keepdims=True)

    # Concatenate all members as independent spatial realisations.
    pos = (cx, cy, cz)
    bin_centers, gamma = gs.vario_estimate(
        pos,
        samples_dm,          # gstools accepts (n_samples, n_cells) fields
        bin_edges=bin_edges,
    )

    vario_cls = getattr(gs, vario_model)
    model = vario_cls(dim=3)

    fit_kw = {} if fit_kwargs is None else dict(fit_kwargs)
    model.fit_variogram(bin_centers, gamma, **fit_kw)

    if out:
        print(
            f"gst_variogram_from_rto_samples: fitted {vario_model}\n"
            f"  var (sill)   = {model.var:.4f} (log10 Ohm.m)^2\n"
            f"  len_scale    = {model.len_scale:.1f} m\n"
            f"  nugget       = {model.nugget:.4f}\n"
            f"  anis         = {model.anis}"
        )

    return model, bin_centers, gamma


def gst_pilot_point_cv(
    ref_log_rho: np.ndarray,
    coords: np.ndarray,
    pp_coords: np.ndarray,
    vario_model: str = "Spherical",
    vario_range: float | Sequence[float] = 20000.0,
    vario_sill: float = 0.5,
    vario_nugget: float = 0.01,
    vario_angles: Optional[Sequence[float]] = None,
    out: bool = True,
) -> dict:
    """Leave-one-out cross-validation of Ordinary Kriging at pilot points.

    Strategy 2: score a variogram parameter set without running FEMTIC.
    The reference model is treated as one realisation of the spatial field;
    its log10(rho) values at the pilot-point locations are predicted by
    Kriging from all *other* pilot points, and the prediction error is
    measured.

    Parameters
    ----------
    ref_log_rho : ndarray, shape (n_cells,)
        Log10(rho) values of the reference model at the free mesh cell centres.
    coords : ndarray, shape (n_cells, 3)
        Spatial coordinates of the free mesh cells (metres), matching
        ``ref_log_rho``.
    pp_coords : ndarray, shape (n_pp, 3)
        Pilot-point coordinates (metres).  Each will be held out in turn.
        Their reference-model values are obtained by nearest-cell look-up.
    vario_model : str
        gstools covariance model class name.
    vario_range : float or (float, float)
        Horizontal range, or (h_range, v_range) for anisotropy (metres).
    vario_sill : float
        Sill (variance) in (log10 Ohm.m)^2.
    vario_nugget : float
        Nugget in (log10 Ohm.m)^2.
    vario_angles : sequence of float, optional
        Rotation angles [alpha, beta, gamma] in degrees.  None = axis-aligned.
    out : bool
        If True, print a summary of CV scores.

    Returns
    -------
    result : dict with keys
        ``"pp_coords"``   -- pilot-point coordinates, shape (n_pp, 3)
        ``"pp_true"``     -- reference log10(rho) at each pilot point
        ``"pp_pred"``     -- leave-one-out Kriging prediction at each point
        ``"pp_var"``      -- Kriging variance at each point
        ``"residuals"``   -- pp_pred - pp_true
        ``"rmse"``        -- root-mean-square prediction error
        ``"mae"``         -- mean absolute error
        ``"skill"``       -- 1 - RMSE / std(pp_true): fraction of variance explained

    Notes
    -----
    Requires ``gstools`` (``pip install gstools``).

    Author: Volker Rath (DIAS)
    Created with the help of Claude Sonnet 4.6 (Anthropic), 2026-05-28.
    """
    try:
        import gstools as gs
    except ImportError:
        raise ImportError(
            "gst_pilot_point_cv requires gstools.  "
            "Install with: pip install gstools"
        )

    ref_log_rho = np.asarray(ref_log_rho, dtype=float)
    coords      = np.asarray(coords,      dtype=float)
    pp_coords   = np.asarray(pp_coords,   dtype=float)

    if ref_log_rho.ndim != 1 or coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            "ref_log_rho must be 1-D; coords must be (n_cells, 3)."
        )
    if pp_coords.ndim != 2 or pp_coords.shape[1] != 3:
        raise ValueError("pp_coords must have shape (n_pp, 3).")

    # --- build variogram ---
    vario_cls = getattr(gs, vario_model)
    if np.isscalar(vario_range):
        len_scale = float(vario_range)
        anis = [1.0, 1.0]
    else:
        h_range, v_range = vario_range
        len_scale = float(h_range)
        anis = [1.0, float(v_range) / float(h_range)]

    vario_kwargs: dict = dict(
        dim=3, var=vario_sill, len_scale=len_scale,
        nugget=vario_nugget, anis=anis,
    )
    if vario_angles is not None:
        vario_kwargs["angles"] = np.deg2rad(vario_angles)
    variogram = vario_cls(**vario_kwargs)

    # --- nearest-cell look-up to assign reference values to pilot points ---
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    _, nn_idx = tree.query(pp_coords)
    pp_true = ref_log_rho[nn_idx]          # shape (n_pp,)

    n_pp = len(pp_true)
    pp_pred = np.empty(n_pp, dtype=float)
    pp_var  = np.empty(n_pp, dtype=float)

    pp_x = pp_coords[:, 0]
    pp_y = pp_coords[:, 1]
    pp_z = pp_coords[:, 2]

    for i in range(n_pp):
        mask = np.ones(n_pp, dtype=bool)
        mask[i] = False
        krig = gs.krige.Ordinary(
            model=variogram,
            cond_pos=(pp_x[mask], pp_y[mask], pp_z[mask]),
            cond_val=pp_true[mask],
        )
        pred, var = krig(
            (pp_x[[i]], pp_y[[i]], pp_z[[i]]),
            return_var=True,
        )
        pp_pred[i] = float(pred[0])
        pp_var[i]  = float(var[0])

    residuals = pp_pred - pp_true
    rmse  = float(np.sqrt(np.mean(residuals ** 2)))
    mae   = float(np.mean(np.abs(residuals)))
    std_t = float(np.std(pp_true))
    skill = float(1.0 - rmse / std_t) if std_t > 0.0 else np.nan

    if out:
        print(
            f"gst_pilot_point_cv ({vario_model}, range={vario_range}, "
            f"sill={vario_sill}, nugget={vario_nugget})\n"
            f"  n_pp  = {n_pp}\n"
            f"  RMSE  = {rmse:.4f}  log10 Ohm.m\n"
            f"  MAE   = {mae:.4f}  log10 Ohm.m\n"
            f"  skill = {skill:.3f}  (1 - RMSE/std; 1=perfect)"
        )

    return {
        "pp_coords":  pp_coords,
        "pp_true":    pp_true,
        "pp_pred":    pp_pred,
        "pp_var":     pp_var,
        "residuals":  residuals,
        "rmse":       rmse,
        "mae":        mae,
        "skill":      skill,
    }


def gst_sill_from_jacobian(
    J: np.ndarray,
    data_var: np.ndarray,
    coords: np.ndarray,
    vario_model: str = "Spherical",
    vario_range: float | Sequence[float] = 20000.0,
    vario_nugget: float = 0.01,
    vario_angles: Optional[Sequence[float]] = None,
    coverage: float = 0.95,
    out: bool = True,
) -> float:
    """Calibrate the GST variogram sill using a linearised Jacobian.

    Strategy 3: propagate the model covariance (implied by the variogram) to
    data space and match it to the observed data noise level.  This calibrates
    the *amplitude* of the perturbations so that the ensemble forward-response
    spread brackets the data at the requested coverage level.

    The data-space covariance is::

        C_d_ens = J @ C_m @ J.T                (linearised)

    A scalar sill ``s`` scales the unit model covariance C_m_unit
    (computed with sill=1), so::

        diag(C_d_ens) = s * diag(J @ C_m_unit @ J.T)

    The sill is solved such that the median ensemble data spread (in
    standard-deviation units) matches the ``coverage`` quantile of the
    data error distribution.

    Parameters
    ----------
    J : ndarray, shape (n_data, n_cells)
        Jacobian (sensitivity) matrix in the same units as the data.
    data_var : ndarray, shape (n_data,)
        Per-datum noise variance (squared standard deviation).
    coords : ndarray, shape (n_cells, 3)
        Free-cell coordinates (metres), matching columns of J.
    vario_model : str
        gstools covariance model class name.
    vario_range : float or (float, float)
        Horizontal range, or (h_range, v_range) (metres).
    vario_nugget : float
        Nugget in (log10 Ohm.m)^2.
    vario_angles : sequence of float, optional
        Rotation angles [alpha, beta, gamma] in degrees.
    coverage : float
        Target coverage fraction (default 0.95).
    out : bool
        If True, print the calibrated sill.

    Returns
    -------
    sill : float
        Calibrated variogram sill in (log10 Ohm.m)^2.

    Notes
    -----
    Requires ``gstools`` (``pip install gstools``).

    For large problems (n_data x n_cells >> 1e6) the full J @ C_m @ J.T
    product is expensive.  This function assembles C_m explicitly only
    when n_cells <= 5000; for larger problems it falls back to a diagonal
    approximation (C_m ~ sill * I) which gives a lower-bound sill estimate.

    Author: Volker Rath (DIAS)
    Created with the help of Claude Sonnet 4.6 (Anthropic), 2026-05-28.
    """
    try:
        import gstools as gs
    except ImportError:
        raise ImportError(
            "gst_sill_from_jacobian requires gstools.  "
            "Install with: pip install gstools"
        )

    J        = np.asarray(J,        dtype=float)
    data_var = np.asarray(data_var, dtype=float)
    coords   = np.asarray(coords,   dtype=float)

    if J.ndim != 2:
        raise ValueError("J must be 2-D (n_data, n_cells).")
    n_data, n_cells = J.shape
    if data_var.shape != (n_data,):
        raise ValueError("data_var must have shape (n_data,).")
    if coords.shape != (n_cells, 3):
        raise ValueError("coords must have shape (n_cells, 3).")

    # Build unit variogram (sill = 1).
    vario_cls = getattr(gs, vario_model)
    if np.isscalar(vario_range):
        len_scale = float(vario_range)
        anis = [1.0, 1.0]
    else:
        h_range, v_range = vario_range
        len_scale = float(h_range)
        anis = [1.0, float(v_range) / float(h_range)]
    vario_kwargs: dict = dict(
        dim=3, var=1.0, len_scale=len_scale,
        nugget=vario_nugget, anis=anis,
    )
    if vario_angles is not None:
        vario_kwargs["angles"] = np.deg2rad(vario_angles)
    variogram_unit = vario_cls(**vario_kwargs)

    # Model covariance: full or diagonal approximation.
    MAX_EXPLICIT = 5000
    approx = False
    if n_cells <= MAX_EXPLICIT:
        cx, cy, cz = coords[:, 0], coords[:, 1], coords[:, 2]
        dx = cx[:, None] - cx[None, :]
        dy = cy[:, None] - cy[None, :]
        dz = cz[:, None] - cz[None, :]
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        C_m_unit = variogram_unit.covariance(dist)   # (n_cells, n_cells)
        JCm = J @ C_m_unit                           # (n_data, n_cells)
        d_var_unit = np.einsum("ij,ij->i", JCm, J)  # diag(J C_m J^T)
    else:
        # Diagonal approximation: C_m ~ I.
        d_var_unit = np.einsum("ij,ij->i", J, J)    # (n_data,)
        approx = True
        if out:
            print(
                f"gst_sill_from_jacobian: n_cells={n_cells} > {MAX_EXPLICIT}; "
                "using diagonal C_m approximation (lower-bound sill estimate)."
            )

    # Solve for sill.
    # For coverage fraction p, the Normal quantile q satisfies
    # P(|x| < q*sigma) = p, i.e. q = norm.ppf((1+p)/2).
    from scipy.stats import norm as _norm
    q = _norm.ppf(0.5 * (1.0 + coverage))

    valid = d_var_unit > 0.0
    if not valid.any():
        raise ValueError(
            "gst_sill_from_jacobian: all diagonal entries of J C_m J^T are zero."
        )
    sill = float(np.median((q**2 * data_var[valid]) / d_var_unit[valid]))
    sill = max(sill, 0.0)

    if out:
        tag = " (diagonal approx)" if approx else ""
        print(
            f"gst_sill_from_jacobian{tag}: calibrated sill = {sill:.4f} "
            f"(log10 Ohm.m)^2  [coverage={coverage:.0%}]"
        )

    return sill


def gst_parameter_diagnostics(
    samples: np.ndarray,
    coords: np.ndarray,
    variogram,
    pp_coords: Optional[np.ndarray] = None,
    ref_log_rho: Optional[np.ndarray] = None,
    J: Optional[np.ndarray] = None,
    data_var: Optional[np.ndarray] = None,
    n_bins: int = 30,
    max_lag: Optional[float] = None,
    log_rho_min: float = 0.0,
    log_rho_max: float = 4.0,
    out: bool = True,
    plot: bool = False,
    plot_file: Optional[str] = None,
) -> dict:
    """Diagnostic summary for a GST sample array.

    Strategy 4: score an already-generated GST ensemble against the target
    variogram, compute spatial spread statistics, optionally evaluate
    leave-one-out CV at pilot points, and optionally propagate the spread
    to data space via a Jacobian.

    Call this function after running ``generate_gst_model_ensemble`` (or
    loading saved samples) to decide whether the variogram parameters are
    appropriate before committing to a full FEMTIC run.

    Parameters
    ----------
    samples : ndarray, shape (n_samples, n_cells)
        Log10(rho) values for each ensemble member and free mesh cell.
    coords : ndarray, shape (n_cells, 3)
        Spatial coordinates of the free cells (metres).
    variogram : gstools covariance model instance
        The *target* variogram used to generate ``samples``.  Its curve is
        overlaid on the empirical variogram in the diagnostic plot.
    pp_coords : ndarray, shape (n_pp, 3), optional
        Pilot-point coordinates.  If supplied together with ``ref_log_rho``,
        LOO-CV scores are computed via :func:`gst_pilot_point_cv`.
    ref_log_rho : ndarray, shape (n_cells,), optional
        Reference-model log10(rho) at the free cells.  Required for CV
        scoring (``pp_coords`` must also be supplied).
    J : ndarray, shape (n_data, n_cells), optional
        Jacobian matrix.  If supplied together with ``data_var``, the
        linearised data-space spread ratio is computed.
    data_var : ndarray, shape (n_data,), optional
        Per-datum noise variance.  Required when ``J`` is supplied.
    n_bins : int
        Number of lag bins for the empirical variogram.  Default 30.
    max_lag : float, optional
        Maximum lag distance (metres).  None = half the coordinate span.
    log_rho_min : float
        Expected lower bound for ensemble values (used to flag violations).
    log_rho_max : float
        Expected upper bound for ensemble values.
    out : bool
        If True, print a structured summary to stdout.
    plot : bool
        If True, produce a diagnostic figure (requires matplotlib).
    plot_file : str, optional
        If given, save the figure to this path; otherwise show interactively.

    Returns
    -------
    diag : dict with keys
        ``"ensemble_mean"``     -- per-cell mean, shape (n_cells,)
        ``"ensemble_std"``      -- per-cell std,  shape (n_cells,)
        ``"global_std_mean"``   -- mean   of per-cell std (scalar)
        ``"global_std_median"`` -- median of per-cell std (scalar)
        ``"frac_clipped"``      -- fraction of values outside [min, max]
        ``"bin_centers"``       -- variogram lag bins (metres)
        ``"gamma_empirical"``   -- empirical semi-variogram
        ``"gamma_target"``      -- target model evaluated at same lags
        ``"variogram_rmse"``    -- RMSE between empirical and target
        ``"cv"``                -- dict from gst_pilot_point_cv, or None
        ``"data_spread_ratio"`` -- median(ens_std_data / noise_std), or None

    Notes
    -----
    Requires ``gstools`` (``pip install gstools``).

    Author: Volker Rath (DIAS)
    Created with the help of Claude Sonnet 4.6 (Anthropic), 2026-05-28.
    """
    try:
        import gstools as gs
    except ImportError:
        raise ImportError(
            "gst_parameter_diagnostics requires gstools.  "
            "Install with: pip install gstools"
        )

    samples = np.asarray(samples, dtype=float)
    coords  = np.asarray(coords,  dtype=float)

    if samples.ndim != 2 or coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            "samples must be (n_samples, n_cells); coords must be (n_cells, 3)."
        )
    if samples.shape[1] != coords.shape[0]:
        raise ValueError("samples.shape[1] must match coords.shape[0].")

    n_samples, n_cells = samples.shape
    cx, cy, cz = coords[:, 0], coords[:, 1], coords[:, 2]

    # ------------------------------------------------------------------
    # 1. Ensemble spread statistics.
    # ------------------------------------------------------------------
    ens_mean = samples.mean(axis=0)
    ens_std  = samples.std(axis=0, ddof=1)
    global_std_mean   = float(ens_std.mean())
    global_std_median = float(np.median(ens_std))
    frac_clipped = float(
        np.mean((samples < log_rho_min) | (samples > log_rho_max))
    )

    # ------------------------------------------------------------------
    # 2. Empirical variogram vs. target.
    # ------------------------------------------------------------------
    if max_lag is None:
        span = np.array([cx.ptp(), cy.ptp(), cz.ptp()])
        max_lag = 0.5 * float(np.linalg.norm(span))

    bin_edges = np.linspace(0.0, max_lag, n_bins + 1)
    samples_dm = samples - samples.mean(axis=0, keepdims=True)

    bin_centers, gamma_emp = gs.vario_estimate(
        (cx, cy, cz),
        samples_dm,
        bin_edges=bin_edges,
    )
    gamma_target = variogram.variogram(bin_centers)
    valid_bins = np.isfinite(gamma_emp) & np.isfinite(gamma_target)
    vario_rmse = float(
        np.sqrt(np.mean((gamma_emp[valid_bins] - gamma_target[valid_bins]) ** 2))
    )

    # ------------------------------------------------------------------
    # 3. LOO cross-validation (optional).
    # ------------------------------------------------------------------
    cv_result = None
    if pp_coords is not None and ref_log_rho is not None:
        cv_result = gst_pilot_point_cv(
            ref_log_rho=ref_log_rho,
            coords=coords,
            pp_coords=pp_coords,
            vario_model=type(variogram).__name__,
            vario_range=float(variogram.len_scale),
            vario_sill=float(variogram.var),
            vario_nugget=float(variogram.nugget),
            out=False,
        )

    # ------------------------------------------------------------------
    # 4. Data-space spread ratio (optional).
    # ------------------------------------------------------------------
    data_spread_ratio = None
    if J is not None and data_var is not None:
        J_arr    = np.asarray(J,        dtype=float)
        dv_arr   = np.asarray(data_var, dtype=float)
        ens_data = samples_dm @ J_arr.T             # (n_samples, n_data)
        std_data = ens_data.std(axis=0, ddof=1)     # (n_data,)
        noise_std = np.sqrt(np.maximum(dv_arr, 0.0))
        valid_d = noise_std > 0.0
        if valid_d.any():
            data_spread_ratio = float(
                np.median(std_data[valid_d] / noise_std[valid_d])
            )

    # ------------------------------------------------------------------
    # 5. Print summary.
    # ------------------------------------------------------------------
    if out:
        print("=" * 60)
        print("gst_parameter_diagnostics")
        print("=" * 60)
        print(f"  n_samples             = {n_samples}")
        print(f"  n_cells               = {n_cells}")
        print(f"  ensemble std (mean)   = {global_std_mean:.4f}  log10 Ohm.m")
        print(f"  ensemble std (median) = {global_std_median:.4f}  log10 Ohm.m")
        print(f"  fraction clipped      = {100*frac_clipped:.2f} %  "
              f"(outside [{log_rho_min}, {log_rho_max}])")
        print(f"  variogram RMSE        = {vario_rmse:.4f}  "
              "(empirical vs. target)")
        if cv_result is not None:
            print(f"  LOO-CV RMSE           = {cv_result['rmse']:.4f}  log10 Ohm.m")
            print(f"  LOO-CV skill          = {cv_result['skill']:.3f}")
        if data_spread_ratio is not None:
            print(f"  data spread ratio     = {data_spread_ratio:.3f}  "
                  "(median ens_std / noise_std; ~1 = well-calibrated)")
        print("=" * 60)

    # ------------------------------------------------------------------
    # 6. Optional plot.
    # ------------------------------------------------------------------
    if plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib not available; skipping plot.")
        else:
            n_panels = 2 + (cv_result is not None) + (data_spread_ratio is not None)
            fig, axs = plt.subplots(1, n_panels,
                                    figsize=(4.5 * n_panels, 4.0))
            axs = np.atleast_1d(axs)
            panel = 0

            # Panel 0: empirical vs. target variogram.
            ax = axs[panel]; panel += 1
            ax.plot(bin_centers / 1e3, gamma_emp,
                    "o", ms=4, label="empirical")
            ax.plot(bin_centers / 1e3, gamma_target,
                    "-", lw=2, label=f"target ({type(variogram).__name__})")
            ax.set_xlabel("Lag (km)")
            ax.set_ylabel("Semi-variance  (log10 Ohm.m)^2")
            ax.set_title(f"Variogram  RMSE={vario_rmse:.4f}")
            ax.legend(fontsize=8)

            # Panel 1: ensemble std histogram.
            ax = axs[panel]; panel += 1
            ax.hist(ens_std, bins=40, edgecolor="none")
            ax.axvline(global_std_median, color="r", lw=1.5,
                       label=f"median={global_std_median:.3f}")
            ax.set_xlabel("Per-cell std  (log10 Ohm.m)")
            ax.set_ylabel("Count")
            ax.set_title("Ensemble spread")
            ax.legend(fontsize=8)

            # Panel 2 (optional): CV residuals.
            if cv_result is not None:
                ax = axs[panel]; panel += 1
                ax.scatter(cv_result["pp_true"], cv_result["pp_pred"],
                           s=20, alpha=0.7)
                lims = [
                    min(cv_result["pp_true"].min(), cv_result["pp_pred"].min()),
                    max(cv_result["pp_true"].max(), cv_result["pp_pred"].max()),
                ]
                ax.plot(lims, lims, "k--", lw=1)
                ax.set_xlabel("True (ref model)  log10 Ohm.m")
                ax.set_ylabel("LOO-CV prediction")
                ax.set_title(
                    f"Pilot-point CV  skill={cv_result['skill']:.3f}"
                )

            # Panel 3 (optional): data spread ratio histogram.
            if data_spread_ratio is not None:
                J_arr2    = np.asarray(J,        dtype=float)
                dv_arr2   = np.asarray(data_var, dtype=float)
                ens_data2 = samples_dm @ J_arr2.T
                std_data2 = ens_data2.std(axis=0, ddof=1)
                noise_std2 = np.sqrt(np.maximum(dv_arr2, 0.0))
                valid_d2   = noise_std2 > 0.0
                ratio      = std_data2[valid_d2] / noise_std2[valid_d2]
                ax = axs[panel]; panel += 1
                ax.hist(ratio, bins=40, edgecolor="none")
                ax.axvline(1.0, color="k", lw=1, ls="--", label="ratio=1")
                ax.axvline(float(np.median(ratio)), color="r", lw=1.5,
                           label=f"median={np.median(ratio):.3f}")
                ax.set_xlabel("ens_std / noise_std")
                ax.set_ylabel("Count")
                ax.set_title("Data-space spread ratio")
                ax.legend(fontsize=8)

            fig.tight_layout()
            if plot_file:
                fig.savefig(plot_file, dpi=150, bbox_inches="tight")
                if out:
                    print(f"  Diagnostic figure saved to: {plot_file}")
            else:
                plt.show()
            plt.close(fig)

    return {
        "ensemble_mean":     ens_mean,
        "ensemble_std":      ens_std,
        "global_std_mean":   global_std_mean,
        "global_std_median": global_std_median,
        "frac_clipped":      frac_clipped,
        "bin_centers":       bin_centers,
        "gamma_empirical":   gamma_emp,
        "gamma_target":      gamma_target,
        "variogram_rmse":    vario_rmse,
        "cv":                cv_result,
        "data_spread_ratio": data_spread_ratio,
    }


# =============================================================================
# End section: GST parameter estimation
# =============================================================================


def generate_data_ensemble(alg: str = 'rto',
    dir_base: str = "./ens_",
    n_samples: int = 1,
    fromto: Optional[List[int]] = None,
    file_in: str = "observe.dat",
    draw_from: Sequence[float | str] = ("normal", 0.0, 1.0),
    method: str = "add",
    errors: Sequence[Sequence[float]] | Sequence[list] = (),
    rng: Optional[Generator] = None,
    out: bool = True,
    derive_pt_from_z: bool = False,
) -> list[str]:
    """
    Generate an ensemble of perturbed observation files in ensemble directories.

    For each ensemble member i, reads
        dir_base + f"{i}/{file_in}"
    makes a backup <file>_orig.dat,
    and calls :func:`modify_data` to inject Gaussian noise.

    Parameters
    ----------
    dir_base : str
        Base directory for ensemble members (e.g. "./ens_").
    n_samples : int
        Number of ensemble members if ``fromto`` is None.
    fromto : None, (int, int), or list of int, optional
        Controls which members are processed.  See :func:`_resolve_fromto`.
    file_in : str
        Name of the observation file in each ensemble directory.
    draw_from : sequence
        Currently kept for future extension; noise is driven by errors.
    method : str
        Placeholder for different perturbation strategies (currently unused).
    errors : sequence
        Error specifications forwarded to :func:`modify_data`.
    rng : numpy.random.Generator, optional
        Shared random generator, forwarded to :func:`femtic.modify_data` for
        each member.  Pass a generator seeded by the caller for a fully
        reproducible data ensemble.  If None, each call to
        :func:`modify_data` falls back to its own fresh, non-reproducible
        generator — pass a shared, seeded ``rng`` here whenever the model
        ensemble (``generate_gst_model_ensemble`` / ``generate_rto_model_
        ensemble``) is also being seeded, so both perturbation paths are
        reproducible together.
    out : bool
        If True, print status messages.
    derive_pt_from_z : bool
        Forwarded to :func:`femtic.modify_data`. If True and a member's
        ``file_in`` contains both an MT and a PT block, each PT site's
        perturbed data is overridden by the phase tensor of that member's
        own already-perturbed Z (matched by site name and frequency),
        instead of being perturbed independently of Z. PT sites/frequencies
        with no MT match keep their independently-perturbed value. Defaults
        to False (previous behaviour: PT perturbed independently).

    Returns
    -------
    obs_list : list of str
        Paths to the perturbed observation files.
    """
    fromto_arr = _resolve_fromto(fromto, n_samples)

    obs_list: list[str] = []
    for iens in fromto_arr:
        file = f"{dir_base}{iens}/{file_in}"
        shutil.copy(file, file.replace(".dat", "_orig.dat"))
        fem.modify_data(
            template_file=file,
            draw_from=draw_from,
            method=method,
            errors=errors,
            rng=rng,
            out=out,
            derive_pt_from_z=derive_pt_from_z,
        )
        obs_list.append(file)

    if out:
        print("list of perturbed observation files:")
        print(obs_list)

    return obs_list


# -----------------------------------------------------------------------------
# Streaming mean / variance (Welford update, Chan pairwise merge) and
# streaming quantiles (fixed-bin histograms)
# -----------------------------------------------------------------------------
# For ensembles too large to hold in memory (e.g. many nullspace-shuttle
# models), the per-cell mean and variance can be accumulated from three
# quantities only:
#
#     n    : number of samples seen so far (scalar, or array over cells)
#     mean : running mean, one value per cell
#     M2   : running sum of squared deviations from the mean, per cell
#
# with  var = M2 / (n - ddof).  Partial results from different chunks or
# MPI ranks are combined with chan_merge().
#
# References
#   Welford, B. P. (1962), Technometrics 4(3), 419-420,
#       doi:10.1080/00401706.1962.10490022  (DOI supplied by the user; not
#       independently confirmed)
#   Chan, T. F., Golub, G. H., LeVeque, R. J. (1983), The American
#       Statistician 37(3), 242-247, doi:10.1080/00031305.1983.10483115
#
# For log-normally distributed quantities (resistivity) accumulate log10(rho).
# -----------------------------------------------------------------------------

def welford_update(
    n: Union[int, np.ndarray],
    mean: np.ndarray,
    m2: np.ndarray,
    x: ArrayLike,
) -> Tuple[Union[int, np.ndarray], np.ndarray, np.ndarray]:
    """Add one sample to a running (n, mean, M2) accumulator (Welford, 1962).

    Parameters
    ----------
    n : int or ndarray
        Number of samples accumulated so far.
    mean, m2 : ndarray
        Running mean and running sum of squared deviations (same shape).
    x : float or ndarray
        New sample, same shape as ``mean`` (one value per cell).

    Returns
    -------
    n, mean, m2
        Updated accumulator.  The inputs are not modified.
    """
    x = np.asarray(x, dtype=float)
    n_new = n + 1
    delta = x - mean
    mean_new = mean + delta / n_new
    m2_new = m2 + delta * (x - mean_new)
    return n_new, mean_new, m2_new


def batch_moments(
    x: ArrayLike,
    axis: int = 0,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """Return ``(n, mean, M2)`` of a batch of samples along ``axis``.

    Two-pass computation (mean first, then squared deviations), numerically
    stable for a chunk held in memory.  NaNs are not handled; mask or remove
    them beforehand.

    Parameters
    ----------
    x : ndarray
        Batch of samples, e.g. shape ``(n_models, n_cells)`` with ``axis=0``.
    axis : int, optional
        Axis along which the samples are stacked (default 0).

    Returns
    -------
    n : int
        Number of samples in the batch.
    mean, m2 : ndarray
        Mean and sum of squared deviations, with ``axis`` removed.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[axis]
    if n == 0:
        shp = list(x.shape)
        del shp[axis]
        return 0, np.zeros(shp), np.zeros(shp)
    mean = x.mean(axis=axis)
    dev = x - np.expand_dims(mean, axis)
    m2 = np.sum(dev * dev, axis=axis)
    return n, mean, m2


def chan_merge(
    n_a: Union[int, np.ndarray],
    mean_a: ArrayLike,
    m2_a: ArrayLike,
    n_b: Union[int, np.ndarray],
    mean_b: ArrayLike,
    m2_b: ArrayLike,
) -> Tuple[Union[int, np.ndarray], np.ndarray, np.ndarray]:
    """Combine two (n, mean, M2) accumulators (Chan, Golub, LeVeque, 1983).

    The pairwise update is::

        n    = n_a + n_b
        d    = mean_b - mean_a
        mean = mean_a + d * n_b / n
        M2   = M2_a + M2_b + d**2 * n_a * n_b / n

    It is safe if either side is empty (``n = 0``).  Use it to combine
    chunks processed one after another, or the per-rank results of an MPI
    run.  The merge is associative up to rounding, so any reduction order
    gives the same result to working precision.

    Parameters
    ----------
    n_a, mean_a, m2_a : accumulator A
    n_b, mean_b, m2_b : accumulator B
        Counts may be scalars or arrays broadcastable against the means.

    Returns
    -------
    n, mean, m2
        Merged accumulator.  ``n`` is an ``int`` for scalar input counts.
    """
    n_a = np.asarray(n_a, dtype=float)
    n_b = np.asarray(n_b, dtype=float)
    mean_a = np.asarray(mean_a, dtype=float)
    mean_b = np.asarray(mean_b, dtype=float)
    m2_a = np.asarray(m2_a, dtype=float)
    m2_b = np.asarray(m2_b, dtype=float)

    n = n_a + n_b
    safe_n = np.where(n > 0, n, 1.0)
    frac_b = n_b / safe_n
    delta = mean_b - mean_a
    mean = mean_a + delta * frac_b
    m2 = m2_a + m2_b + delta * delta * n_a * frac_b

    if n.ndim == 0:
        n = int(n)
    return n, mean, m2


def moments_variance(
    n: Union[int, np.ndarray],
    m2: ArrayLike,
    ddof: int = 1,
) -> np.ndarray:
    """Variance from an accumulator: ``M2 / (n - ddof)``; NaN where n <= ddof.

    Parameters
    ----------
    n : int or ndarray
        Sample count(s).
    m2 : ndarray
        Sum of squared deviations from the mean.
    ddof : int, optional
        Delta degrees of freedom (default 1, sample variance).
    """
    n = np.asarray(n, dtype=float)
    m2 = np.asarray(m2, dtype=float)
    denom = n - ddof
    out = np.full(np.broadcast(m2, denom).shape, np.nan)
    ok = denom > 0
    np.divide(m2, denom, out=out, where=ok)
    return out


# Streaming quantiles.  Exact quantiles need all samples.  A fixed-bin
# histogram per cell (ncell x nbins counts) is a mergeable summary: chunks,
# and MPI ranks that use the same (lo, hi, nbins), combine by plain addition
# of the count arrays.  The result is within one bin width (hi - lo) / nbins
# of the empirical (inverse-CDF) quantile, i.e. of the order statistic that
# holds the requested probability; definitions that interpolate between
# neighbouring order statistics (numpy's default) can differ by the spacing
# of those, which is large in the tails when the ensemble is small.
# Values outside [lo, hi] are counted in the first / last bin (hist_update()
# returns how many); quantiles that fall in those bins are then only bounds.
# For members of one Markov-chain-like or otherwise correlated set the
# result is the empirical quantile of the visited models, not an error bar
# for independent samples.

def hist_update(
    counts: np.ndarray,
    x: ArrayLike,
    lo: float,
    hi: float,
) -> int:
    """Add a chunk of samples to per-cell fixed-bin histograms (in place).

    Parameters
    ----------
    counts : ndarray, shape (ncell, nbins), C-contiguous, integer
        Histogram counts, modified in place.  Start from
        ``np.zeros((ncell, nbins), dtype=np.uint32)``.
    x : ndarray, shape (n_models, ncell) or (ncell,)
        Samples, one value per cell.  Must be finite.
    lo, hi : float
        Value range covered by the bins, ``hi > lo``; bin width is
        ``(hi - lo) / nbins``.  Use the same values for every chunk and rank.

    Returns
    -------
    n_clipped : int
        Number of values outside ``[lo, hi]`` that were put in the first or
        last bin.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    if counts.ndim != 2 or not counts.flags.c_contiguous:
        raise ValueError("counts must be a C-contiguous 2-D array")
    ncell, nbins = counts.shape
    if x.ndim != 2 or x.shape[1] != ncell:
        raise ValueError(
            f"x has shape {x.shape}, expected (n_models, {ncell})")
    if not hi > lo:
        raise ValueError("hist_update: need hi > lo")
    if not np.all(np.isfinite(x)):
        raise ValueError("hist_update: x contains NaN or Inf")

    n_clipped = int(np.count_nonzero((x < lo) | (x > hi)))
    idx = np.floor((x - lo) * (nbins / (hi - lo))).astype(np.intp)
    np.clip(idx, 0, nbins - 1, out=idx)

    flat = counts.reshape(-1)                    # view; counts is C-contiguous
    offs = np.arange(ncell, dtype=np.intp) * nbins
    for row in idx:                              # one bin per cell and sample
        flat[offs + row] += 1                    # indices are unique within a row
    return n_clipped


def hist_quantiles(
    counts: np.ndarray,
    lo: float,
    hi: float,
    probs: ArrayLike,
    block: int = 20000,
) -> np.ndarray:
    """Per-cell quantiles from histograms built with :func:`hist_update`.

    The cumulative counts are interpolated linearly inside the bin that
    contains the requested probability.  The result is within one bin width
    of the empirical inverse-CDF quantile (``method="inverted_cdf"`` in
    numpy.quantile); numpy's default interpolation between neighbouring
    order statistics can differ by their spacing in sparsely sampled tails.  Work is done in blocks of ``block``
    cells to bound the memory of the cumulative sums.

    Parameters
    ----------
    counts : ndarray, shape (ncell, nbins)
        Histogram counts.
    lo, hi : float
        Range used when the histogram was filled.
    probs : array_like of float, shape (nq,)
        Probabilities in (0, 1), e.g. the normal-distribution probabilities
        of -2, -1, 0, +1, +2 sigma: 0.0228, 0.1587, 0.5, 0.8413, 0.9772.
    block : int, optional
        Number of cells processed at a time.

    Returns
    -------
    q : ndarray, shape (nq, ncell)
        Quantile values; NaN for cells without samples.
    """
    counts = np.asarray(counts)
    probs = np.atleast_1d(np.asarray(probs, dtype=float))
    if np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError("probs must lie in [0, 1]")
    ncell, nbins = counts.shape
    width = (hi - lo) / nbins
    q = np.full((probs.size, ncell), np.nan)

    for s in range(0, ncell, block):
        c = counts[s:s + block].astype(np.int64)
        cum = np.cumsum(c, axis=1)
        tot = cum[:, -1]
        for ip, p in enumerate(probs):
            target = p * tot
            # first bin whose cumulative count reaches the target
            b = np.minimum(np.sum(cum < target[:, np.newaxis], axis=1),
                           nbins - 1)
            b1 = b[:, np.newaxis]
            cnt = np.take_along_axis(c, b1, axis=1)[:, 0]
            prev = np.where(
                b > 0,
                np.take_along_axis(cum, np.maximum(b1 - 1, 0), axis=1)[:, 0],
                0)
            frac = np.zeros(target.shape)
            np.divide(target - prev, cnt, out=frac, where=cnt > 0)
            frac = np.clip(frac, 0.0, 1.0)
            val = lo + (b + frac) * width
            val[tot == 0] = np.nan
            q[ip, s:s + block] = val
    return q


# Streaming third and fourth central moments.  Carry (n, mean, M2, M3, M4)
# per cell with Mk = sum((x - mean)**k); chunks and MPI ranks merge with
# merge_moments4().  Skewness g1 = sqrt(n) * M3 / M2**1.5, excess kurtosis
# g2 = n * M4 / M2**2 - 3.
#
# References
#   Terriberry, T. B. (2007, updated 2008), Computing Higher-Order Moments
#       Online, https://people.xiph.org/~tterribe/notes/homs.html
#   Pebay, P. (2008), Formulas for robust, one-pass parallel computation of
#       covariances and arbitrary-order statistical moments, Sandia report
#       SAND2008-6212, doi:10.2172/1028931
#   Pebay, P., Terriberry, T. B., Kolla, H., Bennett, J. (2016), Computational
#       Statistics 31(4), 1305-1325, doi:10.1007/s00180-015-0637-z

def batch_moments4(
    x: ArrayLike,
    axis: int = 0,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(n, mean, M2, M3, M4)`` of a batch of samples along ``axis``.

    Two-pass computation (mean first, then centred powers), as in
    :func:`batch_moments`.  NaNs are not handled.

    Parameters
    ----------
    x : ndarray
        Batch of samples, e.g. shape ``(n_models, n_cells)`` with ``axis=0``.
    axis : int, optional
        Axis along which the samples are stacked (default 0).

    Returns
    -------
    n : int
        Number of samples in the batch.
    mean, m2, m3, m4 : ndarray
        Mean and sums of the 2nd, 3rd and 4th powers of the deviations from
        the mean, with ``axis`` removed.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[axis]
    if n == 0:
        shp = list(x.shape)
        del shp[axis]
        z = np.zeros(shp)
        return 0, z, z.copy(), z.copy(), z.copy()
    mean = x.mean(axis=axis)
    dev = x - np.expand_dims(mean, axis)
    d2 = dev * dev
    m2 = np.sum(d2, axis=axis)
    m3 = np.sum(d2 * dev, axis=axis)
    m4 = np.sum(d2 * d2, axis=axis)
    return n, mean, m2, m3, m4


def merge_moments4(
    acc_a: Tuple,
    acc_b: Tuple,
) -> Tuple[Union[int, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Combine two ``(n, mean, M2, M3, M4)`` accumulators.

    With ``d = mean_b - mean_a`` and ``n = n_a + n_b``::

        mean = mean_a + d * n_b / n
        M2 = M2a + M2b + d**2 * n_a * n_b / n
        M3 = M3a + M3b + d**3 * n_a * n_b * (n_a - n_b) / n**2
             + 3 * d * (n_a * M2b - n_b * M2a) / n
        M4 = M4a + M4b + d**4 * n_a * n_b * (n_a**2 - n_a * n_b + n_b**2) / n**3
             + 6 * d**2 * (n_a**2 * M2b + n_b**2 * M2a) / n**2
             + 4 * d * (n_a * M3b - n_b * M3a) / n

    ``n``, ``mean`` and ``M2`` are those of :func:`chan_merge`.  Safe if
    either side is empty (``n = 0``); start a run from
    ``(0, zeros, zeros, zeros, zeros)``.

    Parameters
    ----------
    acc_a, acc_b : tuple
        ``(n, mean, m2, m3, m4)``; counts may be scalars or arrays
        broadcastable against the means.

    Returns
    -------
    n, mean, m2, m3, m4
        Merged accumulator (``n`` is an ``int`` for scalar input counts).
    """
    n_a, mean_a, m2_a, m3_a, m4_a = (np.asarray(v, dtype=float) for v in acc_a)
    n_b, mean_b, m2_b, m3_b, m4_b = (np.asarray(v, dtype=float) for v in acc_b)

    n = n_a + n_b
    safe_n = np.where(n > 0, n, 1.0)
    d = mean_b - mean_a
    d2 = d * d
    frac_b = n_b / safe_n

    mean = mean_a + d * frac_b
    m2 = m2_a + m2_b + d2 * n_a * frac_b
    m3 = (m3_a + m3_b
          + d2 * d * n_a * n_b * (n_a - n_b) / (safe_n * safe_n)
          + 3.0 * d * (n_a * m2_b - n_b * m2_a) / safe_n)
    m4 = (m4_a + m4_b
          + d2 * d2 * n_a * n_b * (n_a * n_a - n_a * n_b + n_b * n_b)
          / (safe_n ** 3)
          + 6.0 * d2 * (n_a * n_a * m2_b + n_b * n_b * m2_a) / (safe_n * safe_n)
          + 4.0 * d * (n_a * m3_b - n_b * m3_a) / safe_n)

    if n.ndim == 0:
        n = int(n)
    return n, mean, m2, m3, m4


def skew_kurtosis(
    n: Union[int, np.ndarray],
    m2: ArrayLike,
    m3: ArrayLike,
    m4: ArrayLike,
    bias: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Skewness and excess kurtosis from ``(n, M2, M3, M4)``.

    ``bias=True`` gives the moment coefficients ``g1 = sqrt(n) * M3 / M2**1.5``
    and ``g2 = n * M4 / M2**2 - 3``.  ``bias=False`` applies the usual
    small-sample corrections ``G1 = g1 * sqrt(n * (n - 1)) / (n - 2)`` and
    ``G2 = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 + 6)``.  The result
    is NaN where ``M2 = 0`` or ``n`` is too small (``n < 3`` for the skewness,
    ``n < 4`` for the corrected kurtosis).

    Returns
    -------
    skew, ex_kurt : ndarray
    """
    n = np.asarray(n, dtype=float)
    m2 = np.asarray(m2, dtype=float)
    m3 = np.asarray(m3, dtype=float)
    m4 = np.asarray(m4, dtype=float)
    shape = np.broadcast(n, m2, m3, m4).shape
    ok = (m2 > 0) & (n > 0)
    skew = np.full(shape, np.nan)
    kurt = np.full(shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        g1 = np.sqrt(n) * m3 / m2 ** 1.5
        g2 = n * m4 / (m2 * m2) - 3.0
        if bias:
            skew = np.where(ok, g1, np.nan)
            kurt = np.where(ok, g2, np.nan)
        else:
            skew = np.where(ok & (n > 2),
                            g1 * np.sqrt(n * (n - 1.0)) / (n - 2.0), np.nan)
            kurt = np.where(ok & (n > 3),
                            (n - 1.0) / ((n - 2.0) * (n - 3.0))
                            * ((n + 1.0) * g2 + 6.0), np.nan)
    return skew, kurt


def quantile_skewness(
    q_low: ArrayLike,
    q_mid: ArrayLike,
    q_high: ArrayLike,
) -> np.ndarray:
    """Bowley-type skewness ``(q_high + q_low - 2 q_mid) / (q_high - q_low)``.

    Needs only three quantiles, e.g. the 1-sigma (0.1587, 0.5, 0.8413) or
    2-sigma (0.0228, 0.5, 0.9772) quantiles from :func:`hist_quantiles`.  It
    is 0 for a symmetric distribution, lies in [-1, 1] and is not dominated
    by the tails; NaN where ``q_high == q_low``.
    """
    q_low = np.asarray(q_low, dtype=float)
    q_mid = np.asarray(q_mid, dtype=float)
    q_high = np.asarray(q_high, dtype=float)
    width = q_high - q_low
    out = np.full(np.broadcast(q_low, q_mid, q_high).shape, np.nan)
    num = q_high + q_low - 2.0 * q_mid
    np.divide(num, width, out=out, where=width > 0)
    return out


def compute_eofs(
    E: np.ndarray,
    *,
    k: Optional[int] = None,
    method: Literal["svd", "sample_space"] = "svd",
    demean: bool = True,
    ddof: int = 1,
    eps: float = 1e-15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute EOFs (Empirical Orthogonal Functions) from an ensemble matrix.

    Assumes E has shape (ncells, nsamples), i.e. each column is one sample.
    Returns spatial EOFs (patterns) and PCs (coefficients per sample).

    Parameters
    ----------
    E : ndarray, shape (ncells, nsamples)
        Ensemble matrix. Columns are samples.
    k : int or None, optional
        Number of leading EOFs to return. If None, return all (<= nsamples).
    method : {"svd", "sample_space"}, optional
        - "svd": thin SVD on anomaly matrix (recommended).
        - "sample_space": eigen-decomposition of nsamples×nsamples Gram matrix.
    demean : bool, optional
        If True, remove per-cell mean across samples.
    ddof : int, optional
        Degrees of freedom for covariance scaling. Use 1 for sample covariance
        (divide by nsamples-1), or 0 for population (divide by nsamples).
    eps : float, optional
        Small cutoff to avoid division by ~0 eigenvalues.

    Returns
    -------
    eofs : ndarray, shape (ncells, k)
        Spatial EOF patterns (columns). Orthonormal in Euclidean inner product.
    pcs : ndarray, shape (k, nsamples)
        Principal components (coefficients per sample).
    evals : ndarray, shape (k,)
        Eigenvalues of the (scaled) covariance matrix (variance explained).
    frac : ndarray, shape (k,)
        Fraction of total variance explained.
    mean : ndarray, shape (ncells,)
        Mean removed from each cell (zeros if demean=False).

    Notes
    -----
    - EOFs are defined up to sign; both EOF and corresponding PC may flip sign.
    - With nsamples << ncells, both methods are efficient. "svd" is simplest.
    """
    E = np.asarray(E)
    if E.ndim != 2:
        raise ValueError(f"E must be 2D (ncells, nsamples). Got shape {E.shape}.")

    ncells, nsamples = E.shape
    if nsamples < 1:
        raise ValueError("nsamples must be >= 1.")

    if ddof not in (0, 1):
        raise ValueError("ddof must be 0 or 1.")
    denom_cov = nsamples - ddof
    if denom_cov <= 0:
        raise ValueError(
            f"Invalid ddof={ddof} for nsamples={nsamples} (nsamples - ddof must be > 0)."
        )

    # anomalies
    if demean:
        mean = E.mean(axis=1)
        A = E - mean[:, None]
    else:
        mean = np.zeros(ncells, dtype=E.dtype)
        A = E

    # how many modes
    max_modes = min(ncells, nsamples)
    if k is None:
        k = max_modes
    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1.")
    k = min(k, max_modes)

    method = method.lower()
    if method == "svd":
        # A = U S V^T (thin)
        U, s, Vt = np.linalg.svd(A, full_matrices=False)

        # Cov eigenvalues: s^2 / (nsamples - ddof)
        evals_all = (s * s) / denom_cov
        total = evals_all.sum()
        frac_all = evals_all / total if total > 0 else evals_all

        eofs = U[:, :k]
        pcs = (s[:k, None] * Vt[:k, :])  # (k, nsamples)
        evals = evals_all[:k]
        frac = frac_all[:k]
        return eofs, pcs, evals, frac, mean

    if method == "sample_space":
        # Gram matrix: G = A^T A / (nsamples - ddof)
        G = (A.T @ A) / denom_cov  # (nsamples, nsamples)

        # eigen-decomposition (ascending), sort descending
        w, V = np.linalg.eigh(G)
        idx = np.argsort(w)[::-1]
        w = w[idx]
        V = V[:, idx]

        w_k = w[:k]
        V_k = V[:, :k]

        # EOFs: eofs = A V / sqrt((nsamples - ddof) * w)
        # Derivation: A = U S V^T and w = s^2 / (nsamples - ddof)
        scale = np.sqrt(np.maximum(w_k, 0.0) * denom_cov)
        safe = np.where(scale > eps, scale, np.nan)

        eofs = (A @ V_k) / safe[None, :]  # (ncells, k)
        # PCs consistent with SVD convention: pcs = S V^T = sqrt((nsamples-ddof)*w) * V^T
        pcs = (scale[:, None] * V_k.T)    # (k, nsamples)

        # handle any near-zero eigenvalues: set corresponding EOFs/PCs to 0
        bad = ~np.isfinite(eofs).all(axis=0)
        if np.any(bad):
            eofs[:, bad] = 0.0
            pcs[bad, :] = 0.0
            w_k = w_k.copy()
            w_k[bad] = 0.0

        total = w.sum()
        frac = w_k / total if total > 0 else w_k
        return eofs, pcs, w_k, frac, mean

    raise ValueError(f"Unknown method='{method}'. Use 'svd' or 'sample_space'.")


def eof_sample(
    eofs: ArrayLike,
    pcs: ArrayLike,
    mean: ArrayLike | None = None,
    *,
    k0: int = 0,
    k1: int | None = None,
    groups: list[tuple[int, int]] | None = None,
    rng: np.random.Generator | None = None,
    method: str = "empirical_diag",
    whitened: bool = False,
    ddof: int = 1,
    return_components: bool = False,
) -> np.ndarray:
    """
    Draw one new physical-space sample from a truncated EOF model.

    Parameters
    ----------
    eofs : (ncells, r)
        EOF spatial modes (columns).
    pcs : (r, nsamples)
        Training PCs. Used to estimate per-mode variance unless whitened=True.
    mean : (ncells,), optional
        Mean field to add back.
    k0, k1 : int
        Truncation window [k0, k1) in Python slicing convention.
    groups : list[(k0,k1)] or None
        If provided, sample each window; return sum (default) or components.
    rng : np.random.Generator, optional
        Random generator. If None, uses default_rng().
    method : {"empirical_diag", "empirical_full"}
        - "empirical_diag": assume PCs independent; use per-mode variance.
        - "empirical_full": estimate full covariance across selected modes.
    whitened : bool
        If True, assume PCs already have unit variance; sample N(0, I).
    ddof : int
        ddof for variance/covariance estimation.
    return_components : bool
        If True, return (ngroups, ncells) components; else sum to (ncells,).

    Returns
    -------
    x : ndarray
        Sample in physical space: (ncells,) or (ngroups, ncells).
    """
    U = np.asarray(eofs, dtype=float)
    A = np.asarray(pcs, dtype=float)

    if U.ndim != 2 or A.ndim != 2:
        raise ValueError("eofs and pcs must be 2-D arrays.")
    if U.shape[1] != A.shape[0]:
        raise ValueError(f"Shape mismatch: eofs {U.shape}, pcs {A.shape}.")

    r = U.shape[1]
    if k1 is None:
        k1 = r
    if groups is None:
        groups = [(k0, k1)]

    for a, b in groups:
        if not (0 <= a < b <= r):
            raise ValueError(f"Invalid mode window (k0,k1)=({a},{b}) for r={r}.")

    if rng is None:
        rng = np.random.default_rng()

    comps = []
    for a, b in groups:
        Ui = U[:, a:b]          # (ncells, m)
        Pi = A[a:b, :]          # (m, nsamples)
        m = b - a

        if whitened:
            coeff = rng.standard_normal(size=m)
        else:
            if method == "empirical_diag":
                var = Pi.var(axis=1, ddof=ddof)
                var = np.maximum(var, 0.0)
                coeff = rng.standard_normal(size=m) * np.sqrt(var)
            elif method == "empirical_full":
                # covariance across modes: (m, m)
                Ci = np.cov(Pi, bias=(ddof == 0))  # uses ddof=1 if bias=False
                # numerical safety
                Ci = 0.5 * (Ci + Ci.T)
                # sample N(0, Ci) via eig
                w, V = np.linalg.eigh(Ci)
                w = np.maximum(w, 0.0)
                z = rng.standard_normal(size=m)
                coeff = V @ (np.sqrt(w) * (V.T @ z))
            else:
                raise ValueError("method must be 'empirical_diag' or 'empirical_full'.")

        comps.append(Ui @ coeff)  # (ncells,)

    comps_arr = np.stack(comps, axis=0)  # (ngroups, ncells)

    if mean is not None:
        mu = np.asarray(mean, dtype=float).reshape(-1)
        if mu.shape[0] != U.shape[0]:
            raise ValueError("mean has wrong length.")
        comps_arr = comps_arr + mu[None, :]

    return comps_arr if return_components else comps_arr.sum(axis=0)


def _normalize_groups(
    r: int,
    *,
    nmodes: int | None = None,
    k0: int | None = None,
    k1: int | None = None,
    groups: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """
    Build a list of mode windows [(k0,k1), ...] in Python slice convention.

    Priority:
      1) groups if provided
      2) (k0,k1) if provided
      3) nmodes if provided -> (0, nmodes)
      4) default -> (0, r)
    """
    if groups is not None:
        windows = list(groups)
    elif (k0 is not None) or (k1 is not None):
        a = 0 if k0 is None else int(k0)
        b = r if k1 is None else int(k1)
        windows = [(a, b)]
    elif nmodes is not None:
        windows = [(0, int(nmodes))]
    else:
        windows = [(0, r)]

    norm: list[tuple[int, int]] = []
    for a, b in windows:
        a = int(a)
        b = int(b)
        if a < 0 or b < 0 or a > r or b > r or b <= a:
            raise ValueError(f"Invalid mode window (k0,k1)=({a},{b}) for r={r}.")
        norm.append((a, b))
    return norm


def eof_reconstruct(
    eofs: ArrayLike,
    pcs: ArrayLike,
    mean: ArrayLike | None = None,
    *,
    nmodes: int | None = None,
    k0: int | None = None,
    k1: int | None = None,
    groups: list[tuple[int, int]] | None = None,
    return_components: bool = False,
) -> np.ndarray:
    """
    Reconstruct a physical ensemble from EOFs and PC coefficients with optional grouping.

    Parameters
    ----------
    eofs : array_like, shape (ncells, r)
        EOF spatial modes (columns).
    pcs : array_like, shape (r, nsamples)
        PC coefficients.
    mean : array_like or None, shape (ncells,), optional
        Mean field to add back.
    nmodes : int or None, optional
        Convenience for using modes (0, nmodes).
    k0, k1 : int or None, optional
        Mode window in Python slicing convention: modes k0..k1-1.
    groups : list of (k0,k1) or None, optional
        Multiple mode windows; each window is reconstructed and either summed or returned.
    return_components : bool, optional
        If False (default), sum all group reconstructions into one ensemble.
        If True, return stacked components with shape (ngroups, ncells, nsamples).

    Returns
    -------
    ensemble_rec : ndarray
        If return_components=False: (ncells, nsamples)
        If return_components=True:  (ngroups, ncells, nsamples)
    """
    U = np.asarray(eofs, dtype=float)
    A = np.asarray(pcs, dtype=float)

    if U.ndim != 2 or A.ndim != 2:
        raise ValueError("eofs and pcs must be 2-D arrays.")
    if U.shape[1] != A.shape[0]:
        raise ValueError(f"Shape mismatch: eofs {U.shape}, pcs {A.shape}.")

    r = U.shape[1]
    windows = _normalize_groups(r, nmodes=nmodes, k0=k0, k1=k1, groups=groups)

    comps = []
    for a, b in windows:
        comps.append(U[:, a:b] @ A[a:b, :])
    comps_arr = np.stack(comps, axis=0)  # (ngroups, ncells, nsamples)

    if mean is not None:
        mu = np.asarray(mean, dtype=float).reshape(-1)
        if mu.shape[0] != U.shape[0]:
            raise ValueError("mean has wrong length.")
        comps_arr = comps_arr + mu[None, :, None] if return_components else comps_arr

    if return_components:
        return comps_arr

    Erec = comps_arr.sum(axis=0)
    if mean is not None:
        Erec = Erec + mu[:, None]
    return Erec


def eof_reconstruct_from_svd(
    U: ArrayLike,
    s: ArrayLike,
    Vt: ArrayLike,
    mean: ArrayLike | None = None,
    *,
    nmodes: int | None = None,
    k0: int | None = None,
    k1: int | None = None,
    groups: list[tuple[int, int]] | None = None,
    return_components: bool = False,
) -> np.ndarray:
    """
    Reconstruct a physical ensemble from SVD of anomalies: X = U diag(s) Vt,
    with optional grouping (k0-k1 windows).

    Returns same shapes as eof_reconstruct.
    """
    U = np.asarray(U, dtype=float)
    s = np.asarray(s, dtype=float).reshape(-1)
    Vt = np.asarray(Vt, dtype=float)

    if U.ndim != 2 or Vt.ndim != 2 or s.ndim != 1:
        raise ValueError("U and Vt must be 2-D; s must be 1-D.")

    r = U.shape[1]
    if s.shape[0] != r or Vt.shape[0] != r:
        raise ValueError("Inconsistent SVD shapes.")

    windows = _normalize_groups(r, nmodes=nmodes, k0=k0, k1=k1, groups=groups)

    comps = []
    for a, b in windows:
        # X_ab = U[:,a:b] diag(s[a:b]) Vt[a:b,:]
        comps.append(U[:, a:b] @ (s[a:b, None] * Vt[a:b, :]))
    comps_arr = np.stack(comps, axis=0)  # (ngroups, ncells, nsamples)

    if return_components:
        if mean is not None:
            mu = np.asarray(mean, dtype=float).reshape(-1)
            if mu.shape[0] != U.shape[0]:
                raise ValueError("mean has wrong length.")
            comps_arr = comps_arr + mu[None, :, None]
        return comps_arr

    Xrec = comps_arr.sum(axis=0)
    if mean is not None:
        mu = np.asarray(mean, dtype=float).reshape(-1)
        if mu.shape[0] != Xrec.shape[0]:
            raise ValueError("mean has wrong length.")
        Xrec = Xrec + mu[:, None]
    return Xrec

def eof_check_orthonormal(eofs: ArrayLike, *, tol: float = 1e-6) -> float:
    """
    Return max absolute deviation of (U^T U) from identity.
    """
    U = np.asarray(eofs, dtype=float)
    G = U.T @ U
    I = np.eye(G.shape[0])
    return float(np.max(np.abs(G - I)))



def eof_captured_curve_from_pcs(pcs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Cumulative captured anomaly energy fraction from PCs only.

    Assumes EOFs are orthonormal (usual EOF/SVD case), so captured fraction
    by modes 0..k-1 equals sum_{i<k} ||PC_i||^2 / sum_{all} ||PC_i||^2.

    Parameters
    ----------
    pcs : ndarray, shape (r, nsamples)
        PC coefficients for ensemble anomalies.

    Returns
    -------
    k : ndarray, shape (r,)
        Mode counts 1..r
    frac : ndarray, shape (r,)
        Cumulative captured fraction for first k modes.
    """


    A = np.asarray(pcs, dtype=float)
    if A.ndim != 2:
        raise ValueError("pcs must be 2-D (r, nsamples).")

    # energy per mode = sum over samples of PC^2
    e_mode = np.sum(A * A, axis=1)  # (r,)
    total = float(np.sum(e_mode))
    if total <= 0.0:
        k = np.arange(1, A.shape[0] + 1)
        return k, np.zeros_like(k, dtype=float)

    frac = np.cumsum(e_mode) / total
    k = np.arange(1, A.shape[0] + 1)
    return k, frac


def plot_eof_captured_curve(pcs: np.ndarray, *,
                            title: str = "EOF cumulative captured fraction") -> None:
    """
    Plot cumulative captured fraction vs number of EOF modes.

    """
    import matplotlib.pyplot as plt
    k, frac = eof_captured_curve_from_pcs(pcs)

    plt.figure()
    plt.plot(k, frac)
    plt.xlabel("Number of EOF modes (k)")
    plt.ylabel("Cumulative captured fraction")
    plt.title(title)
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.show()


# --- Example usage ---
# pcs = ...  # your array, shape (r, nsamples)
# plot_eof_captured_curve(pcs)

def eof_captured_fraction_from_pcs(
    pcs: ArrayLike,
    *,
    k0: int = 0,
    k1: int | None = None,
    groups: list[tuple[int, int]] | None = None,
) -> float:
    """
    Fraction of anomaly energy captured by a truncated EOF reconstruction, using PCs only.

    Assumes EOFs are orthonormal so that ||U_S A_S||_F^2 = ||A_S||_F^2.

    Parameters
    ----------
    pcs : (r, nsamples)
        PC coefficient matrix for the ensemble anomalies.
    k0, k1 : int
        Mode window [k0, k1). Ignored if groups is provided.
    groups : list[(k0,k1)] or None
        Multiple windows included together (union).

    Returns
    -------
    frac : float
        Captured anomaly energy fraction in [0, 1].
    """
    A = np.asarray(pcs, dtype=float)
    if A.ndim != 2:
        raise ValueError("pcs must be 2-D (r, nsamples).")

    r = A.shape[0]
    if k1 is None:
        k1 = r
    if groups is None:
        groups = [(k0, k1)]

    mask = np.zeros(r, dtype=bool)
    for a, b in groups:
        a = int(a)
        b = int(b)
        if not (0 <= a < b <= r):
            raise ValueError(f"Invalid mode window (k0,k1)=({a},{b}) for r={r}.")
        mask[a:b] = True

    den = float(np.sum(A * A))
    if den <= 0.0:
        return 0.0

    num = float(np.sum(A[mask, :] * A[mask, :]))
    return num / den


def eof_generate_ensemble(
    eofs: ArrayLike,
    pcs: ArrayLike,
    mean: ArrayLike | None = None,
    *,
    nmodes: int | None = None,
) -> np.ndarray:




    """
    EOF-based ensemble model + random physical ensemble generator.

    This module builds an EOF model from an existing ensemble (ncells × nsamples)
    and can generate new random “physical” realizations by sampling in EOF space
    and mapping back to the original parameter space.

    Conventions
    -----------
    - Input ensemble X has shape (ncells, nsamples).
    - EOF model uses anomalies A = X - mean[:, None].
    - Sampling is Gaussian by default: x = mean + U_k @ (sqrt(lam_k) * z),
      where z ~ N(0, I).

    Prewhitening option
    -------------------
    If prewhiten=True, anomalies are scaled cellwise by inverse standard deviation
    before EOF decomposition:
        Aw = W @ A,   W = diag(1 / (std + eps))
    Then sampling occurs in the whitened space, and realizations are unwhitened:
        A = W^{-1} @ Aw

    This is a pragmatic variance-normalization (not full whitening of covariance).

    Author: Volker Rath (DIAS)
    Created with the help of ChatGPT (GPT-5 Thinking) on 2025-12-22
    """

@dataclass(frozen=True)
class EOFModel:
    """Container for an EOF model suitable for fast ensemble sampling.

    Attributes
    ----------
    mean : (ncells,) ndarray
        Ensemble mean in physical space.
    U : (ncells, k) ndarray
        EOF modes (left singular vectors) in the *whitened* space if prewhiten=True,
        otherwise in physical space.
    s : (k,) ndarray
        Singular values corresponding to the retained modes (from SVD of anomalies).
    n_samples_fit : int
        Number of samples used to fit the model.
    prewhiten : bool
        Whether the model was fit with variance-normalizing prewhitening.
    w : (ncells,) ndarray or None
        Prewhitening weights w = 1/(std+eps). Present only if prewhiten=True.
    eps : float
        Epsilon used for stable prewhitening.
    """

    mean: np.ndarray
    U: np.ndarray
    s: np.ndarray
    n_samples_fit: int
    prewhiten: bool
    w: Optional[np.ndarray]
    eps: float


def fit_eof_model(
    X: np.ndarray,
    *,
    nmodes: Optional[int] = None,
    var_fraction: Optional[float] = None,
    prewhiten: bool = False,
    eps: float = 1.0e-12,
    demean: bool = True,
    dtype: np.dtype = np.float64,
) -> EOFModel:
    """Fit an EOF model to an ensemble matrix X (ncells × nsamples).

    Parameters
    ----------
    X : ndarray, shape (ncells, nsamples)
        Input ensemble (each column is one realization).
    nmodes : int, optional
        Number of EOF modes to retain. If None, determined by var_fraction
        (if provided) or defaults to full rank.
    var_fraction : float, optional
        If provided, retain the smallest k such that cumulative explained
        variance >= var_fraction (0 < var_fraction <= 1). Ignored if nmodes
        is given.
    prewhiten : bool, optional
        If True, scale anomalies cellwise by inverse std before EOF decomposition
        and unscale when generating realizations.
    eps : float, optional
        Stabilizer for std in prewhitening weights: w = 1/(std + eps).
    demean : bool, optional
        If True (default), subtract the ensemble mean before decomposition.
        If False, mean is set to zeros and decomposition uses X as-is.
    dtype : numpy dtype, optional
        Computation dtype (default float64).

    Returns
    -------
    EOFModel
        Model containing mean, retained modes, singular values, and (optionally)
        prewhitening weights.

    Notes
    -----
    Uses economy SVD on the (ncells × nsamples) anomaly matrix (or its whitened
    version). This is typically efficient for ncells up to O(1e5) with nsamples
    O(1e2).
    """
    X = np.asarray(X, dtype=dtype)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (ncells, nsamples); got shape {X.shape}")

    ncells, ns = X.shape
    if ns < 2:
        raise ValueError("Need at least 2 samples to fit an EOF model.")

    if demean:
        mean = np.mean(X, axis=1)
        A = X - mean[:, None]
    else:
        mean = np.zeros(ncells, dtype=dtype)
        A = X

    w = None
    if prewhiten:
        std = np.std(A, axis=1, ddof=1)
        w = 1.0 / (std + eps)
        Aw = w[:, None] * A
    else:
        Aw = A

    # Economy SVD: Aw = U S V^T, with U: (ncells, r), S: (r,), r<=ns
    U, s, _Vt = np.linalg.svd(Aw, full_matrices=False)

    # Decide how many modes to keep
    r = s.size
    if nmodes is not None:
        k = int(nmodes)
        if k < 1 or k > r:
            raise ValueError(f"nmodes must be in [1, {r}], got {nmodes}")
    elif var_fraction is not None:
        vf = float(var_fraction)
        if not (0.0 < vf <= 1.0):
            raise ValueError("var_fraction must be in (0, 1].")
        # Explained variance proportional to s^2
        ev = s**2
        cumev = np.cumsum(ev) / np.sum(ev)
        k = int(np.searchsorted(cumev, vf) + 1)
    else:
        k = r

    return EOFModel(
        mean=mean,
        U=U[:, :k].copy(),
        s=s[:k].copy(),
        n_samples_fit=ns,
        prewhiten=prewhiten,
        w=None if w is None else w.copy(),
        eps=float(eps),
    )


def sample_physical_ensemble(
    model: EOFModel,
    nsamples: int,
    *,
    rng: Optional[np.random.Generator] = None,
    coef: Literal["gaussian", "rademacher"] = "gaussian",
    scale: float = 1.0,
    dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    """Generate a new random physical ensemble from a fitted EOFModel.

    Parameters
    ----------
    model : EOFModel
        A model returned by fit_eof_model().
    nsamples : int
        Number of new realizations to generate.
    rng : numpy.random.Generator, optional
        Random generator. If None, uses np.random.default_rng().
    coef : {"gaussian", "rademacher"}, optional
        Distribution of latent coefficients z:
        - "gaussian": z ~ N(0, 1)
        - "rademacher": z in {-1, +1} with equal probability (sometimes useful
          for stress-testing / non-Gaussian perturbations)
    scale : float, optional
        Overall scale factor applied to sampled anomalies (default 1.0).
        Values > 1 inflate variability; values < 1 deflate it.
    dtype : numpy dtype, optional
        Output dtype. If None, uses model.mean dtype.

    Returns
    -------
    Xnew : ndarray, shape (ncells, nsamples)
        New ensemble in physical space.

    Notes
    -----
    If the model was fit with SVD of anomalies Aw = U S V^T, then
    covariance in that space is:
        Cov(Aw) = U diag(S^2/(n-1)) U^T
    To sample consistent anomalies:
        Aw_sample = U @ (S/sqrt(n-1) * z)
    then optionally unwhiten and add mean.
    """
    if nsamples < 1:
        raise ValueError("nsamples must be >= 1.")

    if rng is None:
        rng = np.random.default_rng()

    out_dtype = model.mean.dtype if dtype is None else np.dtype(dtype)

    U = model.U
    s = model.s
    k = s.size
    nfit = model.n_samples_fit
    if nfit < 2:
        raise ValueError("Model was fit with too few samples (need >=2).")

    if coef == "gaussian":
        Z = rng.standard_normal(size=(k, nsamples))
    elif coef == "rademacher":
        Z = rng.integers(0, 2, size=(k, nsamples)) * 2 - 1
        Z = Z.astype(out_dtype, copy=False)
    else:
        raise ValueError("coef must be 'gaussian' or 'rademacher'.")

    # Map latent coefficients to anomalies in (possibly whitened) space
    # factor = S/sqrt(n-1) gives sqrt(eigenvalues) scaling
    factor = (s / np.sqrt(nfit - 1.0))[:, None]
    Aw_new = (U @ (factor * Z)) * float(scale)

    if model.prewhiten:
        if model.w is None:
            raise ValueError("Model indicates prewhiten=True but has no weights.")
        # Unwhiten: A = W^{-1} Aw = Aw / w
        A_new = Aw_new / model.w[:, None]
    else:
        A_new = Aw_new

    X_new = (model.mean[:, None] + A_new).astype(out_dtype, copy=False)
    return X_new


# --- Minimal example (comment out in your library code) ---
if __name__ == "__main__":
    ncells, ns = 100_000, 100
    rng = np.random.default_rng(0)

    # Toy training ensemble
    X = rng.standard_normal((ncells, ns))

    # Fit EOF model (try prewhiten=True for variance-normalized modes)
    model = fit_eof_model(X, var_fraction=0.95, prewhiten=True)

    # Generate new random physical ensemble
    Xnew = sample_physical_ensemble(model, nsamples=200, rng=rng)

    print(X.shape, "->", Xnew.shape)


# ---------------------------------------------------------------------------
# Bridging a precomputed ensemble covariance (e.g. femtic_ens_post.py's
# COMPUTE_COV output) into the same EOFModel container used by
# fit_eof_model()/sample_physical_ensemble() above, plus two femtic-shaped
# convenience wrappers (samples-as-rows, matching ens_matrix throughout the
# femtic_ens_* scripts, instead of EOFModel's samples-as-columns
# convention) for drawing new candidate models from an existing ensemble --
# used by femtic_ens_repair.py's MOD_REPAIR_METHOD="eof_sample" option.
# ---------------------------------------------------------------------------


def eof_model_from_covariance(
    mean: ArrayLike,
    n_samples_fit: int,
    *,
    eigval: Optional[ArrayLike] = None,
    eigvec: Optional[ArrayLike] = None,
    cov: Optional[ArrayLike] = None,
    nmodes: Optional[int] = None,
    var_fraction: Optional[float] = None,
    dtype: np.dtype = np.float64,
) -> "EOFModel":
    """Build an :class:`EOFModel` from an already-computed covariance,
    instead of fitting one from a raw ensemble matrix (:func:`fit_eof_model`).

    Two input forms are accepted, matching femtic_ens_post.py's two
    ``COV_METHOD`` outputs:

    - ``eigval`` (r,) + ``eigvec`` (ncells, r) -- the low-rank
      factorisation saved as ``f"{P}_cov_eigval"``/``f"{P}_cov_eigvec"``
      when ``COV_METHOD="low_rank"``. These are exact eigenpairs of the
      sample covariance ``C = eigvec @ diag(eigval) @ eigvec.T``, so no
      further decomposition is needed -- just re-derive the singular
      values ``s = sqrt(eigval * (n_samples_fit - 1))`` that
      :class:`EOFModel`/:func:`sample_physical_ensemble` expect.
    - ``cov`` (ncells, ncells) -- the dense covariance saved as
      ``f"{P}_cov"`` when ``COV_METHOD="full"``. Eigendecomposed here
      (``scipy.linalg.eigh``, descending) and truncated the same way
      :func:`fit_eof_model` truncates its SVD.

    Exactly one of ``eigval``/``eigvec`` or ``cov`` must be given.

    Parameters
    ----------
    mean : (ncells,)
        Ensemble mean in physical space (e.g. femtic_ens_post.py's
        ``f"{P}_avg"``, log10(rho)).
    n_samples_fit : int
        Number of ensemble members the covariance was estimated from
        (``n_members`` in femtic_ens_post.py) -- needed to recover
        singular values from eigenvalues (and hence match
        :func:`sample_physical_ensemble`'s ``factor = s/sqrt(nfit-1)``
        convention) and to guard against sampling from a covariance
        fit to too few members.
    eigval, eigvec : array_like, optional
        Low-rank factorisation (see above).
    cov : array_like, optional
        Dense covariance matrix (see above).
    nmodes, var_fraction : optional
        Same truncation options as :func:`fit_eof_model`; only used for
        the ``cov`` input (the ``eigval``/``eigvec`` input is already
        truncated to whatever rank femtic_ens_post.py saved -- pass
        ``nmodes`` <= that rank to truncate further if wanted).
    dtype : numpy dtype, optional
        Computation dtype.

    Returns
    -------
    EOFModel
        ``prewhiten=False``, ``w=None`` in both cases -- neither
        femtic_ens_post.py covariance path applies prewhitening, so
        there is nothing to unwhiten here; this model can be passed
        directly to :func:`sample_physical_ensemble`.

    Raises
    ------
    ValueError
        If neither or both of ``(eigval, eigvec)``/``cov`` are given,
        or ``n_samples_fit < 2``.
    """
    mean = np.asarray(mean, dtype=dtype).reshape(-1)
    ncells = mean.shape[0]

    if n_samples_fit < 2:
        raise ValueError(
            f"n_samples_fit={n_samples_fit} < 2 -- covariance is degenerate "
            f"or undefined below 2 samples."
        )

    have_lowrank = eigval is not None and eigvec is not None
    have_dense = cov is not None
    if have_lowrank == have_dense:
        raise ValueError(
            "eof_model_from_covariance: pass exactly one of "
            "(eigval, eigvec) or cov."
        )

    if have_lowrank:
        eigval = np.asarray(eigval, dtype=dtype).reshape(-1)
        eigvec = np.asarray(eigvec, dtype=dtype)
        if eigvec.shape[0] != ncells or eigvec.shape[1] != eigval.shape[0]:
            raise ValueError(
                f"eigvec shape {eigvec.shape} inconsistent with mean "
                f"({ncells},) / eigval ({eigval.shape[0]},)."
            )
        # Descending order expected downstream (truncation keeps the
        # leading/largest modes); femtic_ens_post.py's np.linalg.svd
        # already returns them descending, but sort defensively in case
        # this is fed a covariance from elsewhere.
        order = np.argsort(eigval)[::-1]
        eigval = eigval[order]
        eigvec = eigvec[:, order]

        r = eigval.size
        if nmodes is not None:
            k = min(int(nmodes), r)
        elif var_fraction is not None:
            vf = float(var_fraction)
            if not (0.0 < vf <= 1.0):
                raise ValueError("var_fraction must be in (0, 1].")
            cumev = np.cumsum(eigval) / np.sum(eigval)
            k = int(np.searchsorted(cumev, vf) + 1)
        else:
            k = r

        eigval_k = np.clip(eigval[:k], 0.0, None)
        U = eigvec[:, :k]
        s = np.sqrt(eigval_k * (n_samples_fit - 1))

    else:
        cov = np.asarray(cov, dtype=dtype)
        if cov.shape != (ncells, ncells):
            raise ValueError(
                f"cov shape {cov.shape} inconsistent with mean ({ncells},)."
            )
        w, V = scipy.linalg.eigh(cov)          # ascending
        order = np.argsort(w)[::-1]
        w = np.clip(w[order], 0.0, None)
        V = V[:, order]

        r = w.size
        if nmodes is not None:
            k = min(int(nmodes), r)
        elif var_fraction is not None:
            vf = float(var_fraction)
            if not (0.0 < vf <= 1.0):
                raise ValueError("var_fraction must be in (0, 1].")
            cumev = np.cumsum(w) / np.sum(w)
            k = int(np.searchsorted(cumev, vf) + 1)
        else:
            k = r

        U = V[:, :k]
        s = np.sqrt(w[:k] * (n_samples_fit - 1))

    return EOFModel(
        mean=mean,
        U=U.copy(),
        s=s.copy(),
        n_samples_fit=int(n_samples_fit),
        prewhiten=False,
        w=None,
        eps=0.0,
    )


def sample_new_models_from_ensemble(
    ens_matrix: ArrayLike,
    n_new: int,
    *,
    nmodes: Optional[int] = None,
    var_fraction: Optional[float] = None,
    prewhiten: bool = False,
    rng: Optional[np.random.Generator] = None,
    scale: float = 1.0,
    coef: Literal["gaussian", "rademacher"] = "gaussian",
) -> np.ndarray:
    """Draw ``n_new`` new candidate models from an ensemble's own empirical
    covariance, fit on the fly from ``ens_matrix``.

    Thin, femtic-shaped convenience wrapper around
    :func:`fit_eof_model` + :func:`sample_physical_ensemble`: those two
    use the ``(ncells, nsamples)`` (samples-as-columns) convention, while
    every ``femtic_ens_*.py`` script (and hence ``ens_matrix`` as passed
    in here) uses ``(n_members, n_free)`` (samples-as-rows) -- this
    wrapper transposes on the way in and out so callers never have to
    think about the mismatch.

    Parameters
    ----------
    ens_matrix : array_like, shape (n_members, n_free)
        Existing ensemble (e.g. the converged members' log10(rho)
        models), same orientation as ``ens_matrix``/``ens_avg`` etc.
        throughout femtic_ens_post.py/femtic_ens_repair.py.
    n_new : int
        Number of new candidate models to draw.
    nmodes, var_fraction, prewhiten : optional
        Passed straight through to :func:`fit_eof_model`.
    rng : numpy.random.Generator, optional
        Random generator; ``None`` uses fresh OS entropy.
    scale : float, optional
        Overall scale factor on the sampled anomalies (see
        :func:`sample_physical_ensemble`); 1.0 = draw from the ensemble's
        own covariance as-is.
    coef : {"gaussian", "rademacher"}, optional
        Latent-coefficient distribution; see :func:`sample_physical_ensemble`.

    Returns
    -------
    new_models : ndarray, shape (n_new, n_free)
        New candidate models, same orientation as ``ens_matrix``.
    """
    X = np.asarray(ens_matrix, dtype=float).T          # (n_free, n_members)
    model = fit_eof_model(
        X, nmodes=nmodes, var_fraction=var_fraction, prewhiten=prewhiten,
    )
    X_new = sample_physical_ensemble(
        model, n_new, rng=rng, coef=coef, scale=scale,
    )                                                    # (n_free, n_new)
    return X_new.T                                       # (n_new, n_free)


def sample_new_models_from_covariance(
    mean: ArrayLike,
    n_samples_fit: int,
    n_new: int,
    *,
    eigval: Optional[ArrayLike] = None,
    eigvec: Optional[ArrayLike] = None,
    cov: Optional[ArrayLike] = None,
    nmodes: Optional[int] = None,
    var_fraction: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    scale: float = 1.0,
    coef: Literal["gaussian", "rademacher"] = "gaussian",
) -> np.ndarray:
    """Draw ``n_new`` new candidate models from a *precomputed* ensemble
    covariance (e.g. loaded from femtic_ens_post.py's ``.npz`` output),
    without needing the raw ensemble matrix at all.

    Femtic-shaped convenience wrapper around
    :func:`eof_model_from_covariance` + :func:`sample_physical_ensemble`
    (samples-as-rows in/out; see :func:`sample_new_models_from_ensemble`
    for why the transpose is needed). See
    :func:`eof_model_from_covariance` for the ``eigval``/``eigvec``
    vs. ``cov`` input forms (exactly one must be given).

    Parameters
    ----------
    mean : (n_free,)
        Ensemble mean, e.g. femtic_ens_post.py's ``f"{P}_avg"``.
    n_samples_fit : int
        Number of members the covariance was estimated from
        (femtic_ens_post.py's ``n_members``).
    n_new : int
        Number of new candidate models to draw.
    eigval, eigvec, cov, nmodes, var_fraction :
        See :func:`eof_model_from_covariance`.
    rng, scale, coef :
        See :func:`sample_physical_ensemble`.

    Returns
    -------
    new_models : ndarray, shape (n_new, n_free)
    """
    model = eof_model_from_covariance(
        mean, n_samples_fit,
        eigval=eigval, eigvec=eigvec, cov=cov,
        nmodes=nmodes, var_fraction=var_fraction,
    )
    X_new = sample_physical_ensemble(
        model, n_new, rng=rng, coef=coef, scale=scale,
    )                                                    # (n_free, n_new)
    return X_new.T                                       # (n_new, n_free)


# ---------------------------------------------------------------------------
# Weighted KL decomposition on FEMTIC mesh
# ---------------------------------------------------------------------------


def compute_weighted_kl(
    rho_samples: np.ndarray,
    volumes: np.ndarray,
    n_modes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a weighted Karhunen-Loève (KL) expansion of log-rho fields on a
    FEMTIC mesh.

    Parameters
    ----------
    rho_samples : ndarray, shape (n_cells, n_samples)
        Log-resistivity samples (e.g. log10(rho)) on FEMTIC cells.
        Each column is one realization.
    volumes : ndarray, shape (n_cells,)
        Positive weights per cell (e.g. cell volumes). These define the
        spatial inner product
            <f, g>_w = sum_i volumes[i] * f[i] * g[i]
        and should be non-negative. Zero-volume cells are allowed but will
        not contribute to the inner product.
    n_modes : int
        Number of KL modes to retain.

    Returns
    -------
    mean_field : ndarray, shape (n_cells,)
        Sample mean of rho_samples over realizations.
    modes : ndarray, shape (n_cells, n_modes)
        Spatial KL modes phi_k on the FEMTIC mesh, orthonormal w.r.t. the
        weighted inner product:
            modes[:, k].T @ (volumes * modes[:, l]) ≈ delta_kl
    eigvals : ndarray, shape (n_modes,)
        Eigenvalues (mode variances) corresponding to each KL mode.
    mode_coeffs : ndarray, shape (n_modes, n_samples)
        KL coefficients a_k^(s) for each mode k and realization s:
            a_k^(s) = <rho_samples[:, s] - mean_field, modes[:, k]>_w

    Notes
    -----
    Implementation uses an SVD-based algorithm on the weighted data matrix:

        X = rho_samples - mean_field[:, None]
        X_w = sqrt(volumes)[:, None] * X
        X_w = U S V^T

    Then eigenvalues are S^2 / (n_samples - 1) and the physical modes are

        phi_k = U_k / sqrt(volumes)

    Finally the mode coefficients are computed explicitly using the weighted
    inner product for robustness:

        a_k^(s) = modes[:, k].T @ (volumes * X[:, s])

    """
    rho_samples = np.asarray(rho_samples, dtype=float)
    volumes = np.asarray(volumes, dtype=float)

    if rho_samples.ndim != 2:
        raise ValueError("rho_samples must be 2D (n_cells, n_samples).")
    if volumes.ndim != 1:
        raise ValueError("volumes must be 1D (n_cells,).")

    n_cells, n_samples = rho_samples.shape
    if volumes.shape[0] != n_cells:
        raise ValueError(
            f"volumes.shape[0] = {volumes.shape[0]} "
            f"does not match rho_samples.shape[0] = {n_cells}."
        )
    if n_modes > min(n_cells, n_samples):
        raise ValueError(
            "n_modes cannot exceed min(n_cells, n_samples); "
            f"got n_modes={n_modes}."
        )

    # Sample mean over realizations (per cell)
    mean_field = rho_samples.mean(axis=1)

    # Centered samples
    X = rho_samples - mean_field[:, None]  # (n_cells, n_samples)

    # Weighted data matrix: multiply each row by sqrt(volume)
    # Guard against zero or negative volumes: treat <=0 as zero.
    vol_clipped = np.clip(volumes, a_min=0.0, a_max=None)
    sqrt_w = np.sqrt(vol_clipped)
    # Avoid division by zero later by replacing 0 with 1 temporarily
    sqrt_w_safe = np.where(sqrt_w > 0.0, sqrt_w, 1.0)

    X_w = sqrt_w_safe[:, None] * X  # (n_cells, n_samples)

    # Economy SVD: X_w = U S Vt
    U, S, Vt = np.linalg.svd(X_w, full_matrices=False)

    # Eigenvalues of covariance operator
    eigvals_full = (S ** 2) / (n_samples - 1)

    # Truncate
    modes_w = U[:, :n_modes]            # weighted modes (orthonormal in Euclidean sense)
    eigvals = eigvals_full[:n_modes]

    # Transform back to physical modes, normalized w.r.t. weighted inner product
    # phi_k = modes_w[:, k] / sqrt_w  (elementwise)
    modes = modes_w / sqrt_w_safe[:, None]

    # Explicit KL coefficients using weighted inner product
    # a_k^(s) = phi_k^T (volumes * X[:, s])
    WX = vol_clipped[:, None] * X           # (n_cells, n_samples)
    mode_coeffs = modes.T @ WX              # (n_modes, n_samples)

    return mean_field, modes, eigvals, mode_coeffs


# ---------------------------------------------------------------------------
# Polynomial Chaos utilities
# ---------------------------------------------------------------------------


def total_degree_multiindex(M: int, p_max: int) -> List[Tuple[int, ...]]:
    """
    Generate all multi-indices alpha in N_0^M with total degree <= p_max.

    Parameters
    ----------
    M : int
        Number of random input variables xi_1, ..., xi_M.
    p_max : int
        Maximum total polynomial degree.

    Returns
    -------
    multiindex : list of tuple[int, ...]
        List of multi-indices alpha = (alpha_1, ..., alpha_M) with
        sum(alpha) <= p_max. The first element is always (0, ..., 0).
    """
    if M <= 0:
        raise ValueError("M must be positive.")
    if p_max < 0:
        raise ValueError("p_max must be non-negative.")

    multiindex: List[Tuple[int, ...]] = []
    for total_deg in range(p_max + 1):
        for alpha in itertools.product(range(total_deg + 1), repeat=M):
            if sum(alpha) == total_deg:
                multiindex.append(alpha)
    return multiindex


# --- 1D orthonormal polynomial evaluation ---------------------------------


def _hermite_orthonormal(x: np.ndarray | float, degree: int) -> np.ndarray | float:
    """
    Orthonormal Hermite polynomial (physicists' Hermite) for N(0, 1) inputs.

    H_n(x) are the physicists' Hermite polynomials (numpy.polynomial.hermite),
    orthogonal w.r.t. exp(-x^2). The orthonormal basis for standard normal
    N(0, 1) is

        psi_n(x) = H_n(x) / sqrt(2^n n! sqrt(pi))

    Parameters
    ----------
    x : float or ndarray
        Evaluation points.
    degree : int
        Polynomial degree n >= 0.

    Returns
    -------
    psi_n : float or ndarray
        Orthonormal Hermite polynomial psi_n(x).
    """
    if degree < 0:
        raise ValueError("degree must be non-negative.")

    # Coefficients for H_n: [0, 0, ..., 1] of length degree+1
    coeffs = np.zeros(degree + 1)
    coeffs[-1] = 1.0
    Hn = hermval(x, coeffs)

    norm_sq = (2.0 ** degree) * math.factorial(degree) * math.sqrt(math.pi)
    norm = math.sqrt(norm_sq)
    return Hn / norm


def _legendre_orthonormal(x: np.ndarray | float, degree: int) -> np.ndarray | float:
    """
    Orthonormal Legendre polynomial for uniform[-1, 1] inputs.

    P_n(x) are the standard Legendre polynomials (numpy.polynomial.legendre),
    orthogonal on [-1, 1] with weight 1. The orthonormal basis is

        psi_n(x) = P_n(x) * sqrt((2n + 1)/2)

    Parameters
    ----------
    x : float or ndarray
        Evaluation points.
    degree : int
        Polynomial degree n >= 0.

    Returns
    -------
    psi_n : float or ndarray
        Orthonormal Legendre polynomial psi_n(x).
    """
    if degree < 0:
        raise ValueError("degree must be non-negative.")

    coeffs = np.zeros(degree + 1)
    coeffs[-1] = 1.0
    Pn = legval(x, coeffs)

    norm = math.sqrt((2 * degree + 1) / 2.0)
    return Pn * norm


def eval_1d_poly(
    x: np.ndarray | float,
    degree: int,
    family: str = "hermite",
) -> np.ndarray | float:
    """
    Evaluate a 1D orthonormal polynomial of given degree and family.

    Parameters
    ----------
    x : float or ndarray
        Evaluation points.
    degree : int
        Polynomial degree >= 0.
    family : {"hermite", "legendre"}
        Polynomial family:
        - "hermite"  : standard normal inputs N(0, 1)
        - "legendre" : uniform inputs on [-1, 1]

    Returns
    -------
    value : float or ndarray
        psi_n(x) for the chosen family and degree.
    """
    if family == "hermite":
        return _hermite_orthonormal(x, degree)
    if family == "legendre":
        return _legendre_orthonormal(x, degree)
    raise ValueError(f"Unknown polynomial family {family!r}.")


def build_design_matrix(
    Xi: np.ndarray,
    multiindex: List[Tuple[int, ...]],
    family: str = "hermite",
) -> np.ndarray:
    """
    Build the PCE design matrix Psi for given input samples and multi-index set.

    Psi[s, k] = Psi_k(Xi[s, :]) where Psi_k is a multivariate orthonormal
    polynomial defined by multiindex[k].

    Parameters
    ----------
    Xi : ndarray, shape (n_samples, M)
        Input random samples. For "hermite", Xi should be i.i.d. N(0, 1).
        For "legendre", Xi should be within [-1, 1]^M.
    multiindex : list of tuple[int, ...]
        Multi-indices alpha defining the multivariate polynomials.
    family : {"hermite", "legendre"}, optional
        Polynomial family (same for all dimensions).

    Returns
    -------
    Psi : ndarray, shape (n_samples, n_basis)
        Design matrix, where n_basis = len(multiindex).

    Notes
    -----
    For clarity this implementation uses nested Python loops. For large
    n_samples and n_basis this can be optimized by caching 1D polynomial
    evaluations per dimension and degree.
    """
    Xi = np.asarray(Xi, dtype=float)
    if Xi.ndim != 2:
        raise ValueError("Xi must be 2D (n_samples, M).")

    n_samples, M = Xi.shape
    n_basis = len(multiindex)
    Psi = np.empty((n_samples, n_basis), dtype=float)

    for s in range(n_samples):
        xi_s = Xi[s, :]
        for k, alpha in enumerate(multiindex):
            # multivariate polynomial = product_m psi_{alpha_m}(xi_m)
            val = 1.0
            for m, deg in enumerate(alpha):
                if deg == 0:
                    # psi_0 = 1 for any family
                    continue
                val *= eval_1d_poly(xi_s[m], deg, family=family)
            Psi[s, k] = val

    return Psi


# ---------------------------------------------------------------------------
# PCE for KL mode coefficients
# ---------------------------------------------------------------------------


def fit_pce_for_kl_modes(
    mode_coeffs: np.ndarray,
    Xi: np.ndarray,
    multiindex: List[Tuple[int, ...]],
    family: str = "hermite",
) -> np.ndarray:
    """
    Fit a scalar PCE for each KL mode coefficient a_k(xi).

    Parameters
    ----------
    mode_coeffs : ndarray, shape (n_modes, n_samples)
        KL coefficients from the weighted KL decomposition.
        mode_coeffs[k, s] = a_k^(s).
    Xi : ndarray, shape (n_samples, M)
        Input random samples Xi[s, :] corresponding to the realizations in
        mode_coeffs.
    multiindex : list of tuple[int, ...]
        Multi-index set defining the multivariate PCE basis.
    family : {"hermite", "legendre"}, optional
        Polynomial family.

    Returns
    -------
    pce_coeffs : ndarray, shape (n_modes, n_basis)
        PCE coefficients for each mode. Row k contains the coefficients
        c_{k, j} for basis function j.

    Notes
    -----
    We solve the least squares problem

        a_k ≈ Psi c_k

    for each mode k, where a_k is shape (n_samples,) and Psi is the
    design matrix of shape (n_samples, n_basis).
    """
    mode_coeffs = np.asarray(mode_coeffs, dtype=float)
    Xi = np.asarray(Xi, dtype=float)

    if mode_coeffs.ndim != 2:
        raise ValueError("mode_coeffs must be 2D (n_modes, n_samples).")
    if Xi.ndim != 2:
        raise ValueError("Xi must be 2D (n_samples, M).")

    n_modes, n_samples = mode_coeffs.shape
    n_samples_Xi, _ = Xi.shape
    if n_samples_Xi != n_samples:
        raise ValueError(
            "Inconsistent number of samples: "
            f"mode_coeffs.shape[1]={n_samples}, Xi.shape[0]={n_samples_Xi}."
        )

    Psi = build_design_matrix(Xi, multiindex, family=family)  # (n_samples, n_basis)
    n_basis = Psi.shape[1]

    pce_coeffs = np.empty((n_modes, n_basis), dtype=float)
    for k in range(n_modes):
        y = mode_coeffs[k, :]
        c_k, *_ = np.linalg.lstsq(Psi, y, rcond=None)
        pce_coeffs[k, :] = c_k

    return pce_coeffs


def evaluate_kl_pce_surrogate(
    mean_field: np.ndarray,
    modes: np.ndarray,
    pce_coeffs: np.ndarray,
    Xi_new: np.ndarray,
    multiindex: List[Tuple[int, ...]],
    family: str = "hermite",
) -> np.ndarray:
    """
    Evaluate the full KL+PCE surrogate at new random inputs Xi_new.

    Parameters
    ----------
    mean_field : ndarray, shape (n_cells,)
        Mean log-rho field on the FEMTIC mesh.
    modes : ndarray, shape (n_cells, n_modes)
        Spatial KL modes phi_k on the FEMTIC mesh.
    pce_coeffs : ndarray, shape (n_modes, n_basis)
        PCE coefficients c_{k, j} for each KL mode a_k(xi).
    Xi_new : ndarray, shape (n_new, M)
        New input random samples where the surrogate should be evaluated.
    multiindex : list of tuple[int, ...]
        The same multi-index set as used in fitting.
    family : {"hermite", "legendre"}, optional
        Polynomial family.

    Returns
    -------
    rho_new : ndarray, shape (n_cells, n_new)
        Approximate log-rho fields for each new input sample.

    Notes
    -----
    For each new sample xi^*, we compute:

        a_k(xi^*) ≈ Σ_j c_{k, j} Psi_j(xi^*)

    and then reconstruct the field:

        rho(x, xi^*) ≈ mean_field(x) + Σ_k phi_k(x) a_k(xi^*).
    """
    mean_field = np.asarray(mean_field, dtype=float)
    modes = np.asarray(modes, dtype=float)
    pce_coeffs = np.asarray(pce_coeffs, dtype=float)
    Xi_new = np.asarray(Xi_new, dtype=float)

    if Xi_new.ndim != 2:
        raise ValueError("Xi_new must be 2D (n_new, M).")

    n_cells, n_modes = modes.shape
    if mean_field.shape[0] != n_cells:
        raise ValueError(
            "mean_field length does not match modes.shape[0]."
        )

    # Evaluate PCE for all modes at once:
    # Psi_new: (n_new, n_basis)
    Psi_new = build_design_matrix(Xi_new, multiindex, family=family)
    # a_new: (n_modes, n_new) = c @ Psi_new^T
    a_new = pce_coeffs @ Psi_new.T

    # Reconstruct fields: rho_new = mean_field[:, None] + modes @ a_new
    rho_new = mean_field[:, None] + modes @ a_new
    return rho_new


# ---------------------------------------------------------------------------
# Dataclass wrapper for convenience
# ---------------------------------------------------------------------------


@dataclass
class KLPCEModel:
    """
    Container for a fitted KL+PCE model on a FEMTIC mesh.

    Attributes
    ----------
    mean_field : ndarray, shape (n_cells,)
        Mean log-rho field on the FEMTIC mesh.
    modes : ndarray, shape (n_cells, n_modes)
        Spatial KL modes, orthonormal with respect to the weighted inner
        product defined by the cell volumes.
    eigvals : ndarray, shape (n_modes,)
        KL eigenvalues (mode variances).
    volumes : ndarray, shape (n_cells,)
        Cell volumes used in the weighted KL.
    multiindex : list of tuple[int, ...]
        Multi-index set defining the multivariate PCE basis.
    pce_coeffs : ndarray, shape (n_modes, n_basis)
        Scalar PCE coefficients for each mode.
    family : {"hermite", "legendre"}
        Polynomial family used in the PCE.
    """

    mean_field: np.ndarray
    modes: np.ndarray
    eigvals: np.ndarray
    volumes: np.ndarray
    multiindex: List[Tuple[int, ...]]
    pce_coeffs: np.ndarray
    family: str = "hermite"

    def evaluate(self, Xi_new: np.ndarray) -> np.ndarray:
        """
        Evaluate the KL+PCE surrogate at new random inputs Xi_new.

        Parameters
        ----------
        Xi_new : ndarray, shape (n_new, M)
            New input random samples.

        Returns
        -------
        rho_new : ndarray, shape (n_cells, n_new)
            Approximate log-rho fields on FEMTIC cells.
        """
        return evaluate_kl_pce_surrogate(
            self.mean_field,
            self.modes,
            self.pce_coeffs,
            Xi_new,
            self.multiindex,
            family=self.family,
        )


def fit_kl_pce_model(
    rho_samples: np.ndarray,
    volumes: np.ndarray,
    Xi: np.ndarray,
    n_modes: int,
    p_max: int,
    family: str = "hermite",
) -> KLPCEModel:
    """
    High-level convenience function: fit a full KL+PCE model in one call.

    Parameters
    ----------
    rho_samples : ndarray, shape (n_cells, n_samples)
        Log-rho fields on FEMTIC cells (each column is one realization).
    volumes : ndarray, shape (n_cells,)
        Cell volumes used as weights in the spatial inner product.
    Xi : ndarray, shape (n_samples, M)
        Input random samples used to generate rho_samples.
    n_modes : int
        Number of KL modes to retain.
    p_max : int
        Maximum total degree for the polynomial chaos basis.
    family : {"hermite", "legendre"}, optional
        Polynomial family for PCE ("hermite" for Gaussian, "legendre" for
        uniform on [-1, 1]).

    Returns
    -------
    model : KLPCEModel
        Fitted model that can be evaluated at new Xi via model.evaluate(Xi_new).

    Example
    -------
    >>> mean_field, modes, eigvals, mode_coeffs = compute_weighted_kl(
    ...     rho_samples, volumes, n_modes=10)
    >>> multiindex = total_degree_multiindex(M=Xi.shape[1], p_max=2)
    >>> pce_coeffs = fit_pce_for_kl_modes(mode_coeffs, Xi, multiindex)
    >>> model = KLPCEModel(mean_field, modes, eigvals, volumes,
    ...                    multiindex, pce_coeffs, family="hermite")
    """
    mean_field, modes, eigvals, mode_coeffs = compute_weighted_kl(
        rho_samples, volumes, n_modes=n_modes
    )
    M = Xi.shape[1]
    multiindex = total_degree_multiindex(M, p_max)
    pce_coeffs = fit_pce_for_kl_modes(mode_coeffs, Xi, multiindex, family=family)

    return KLPCEModel(
        mean_field=mean_field,
        modes=modes,
        eigvals=eigvals,
        volumes=np.asarray(volumes, dtype=float),
        multiindex=multiindex,
        pce_coeffs=pce_coeffs,
        family=family,
    )

# ============================================================================
# ============================================================================
# SECTION 3: Gaussian sampling with Q = R.T @ R (+ λI)
# ============================================================================


def build_rtr_operator(
    R: np.ndarray | scipy.sparse.sparray,
    lam: float = 0.0,
) -> LinearOperator:
    """Create a LinearOperator representing Q = R.T @ R + lam * I.

    This is the **matrix-free** representation used by iterative solvers.

    Parameters
    ----------
    R : array_like or scipy.sparse.sparray, shape (m, n)
        Matrix defining the precision via Q = R.T @ R. In FEMTIC workflows,
        this is typically a roughness / regularisation operator.
    lam : float, optional
        Diagonal Tikhonov shift. If ``lam > 0`` then
        Q = R.T @ R + lam * I is strictly positive definite.

    Returns
    -------
    Q_op : scipy.sparse.linalg.LinearOperator, shape (n, n)
        Linear operator that applies Q to a vector.

    Notes
    -----
    The matvec evaluates::

        y = R @ x
        z = R.T @ y
        z += lam * x

    without explicitly forming Q.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-12-19
    """
    _, n = R.shape

    def matvec(x: np.ndarray) -> np.ndarray:
        """Compute z = (R.T @ R + lam * I) @ x."""
        y = R @ x
        z = R.T @ y
        if lam != 0.0:
            z = z + lam * x
        return np.asarray(z, dtype=np.float64)

    return LinearOperator((n, n), matvec=matvec, rmatvec=matvec, dtype=np.float64)


def make_rtr_preconditioner(
    R: np.ndarray | scipy.sparse.sparray,
    lam: float = 0.0,
    *,
    precond: Optional[str] = None,
    precond_kwargs: Optional[dict] = None,
) -> Optional[LinearOperator]:
    """Build a CG-compatible preconditioner M ≈ Q^{-1} for Q = R.T@R + lam*I.

    Parameters
    ----------
    R : array_like or sparse matrix, shape (m, n)
        Operator defining Q.
    lam : float, optional
        Diagonal shift used in Q.
    precond : {None, 'jacobi', 'ilu', 'amg', 'identity'}, optional
        Preconditioner type:

        - None: no preconditioning (M = None).
        - 'jacobi': diagonal (inverse) preconditioner using diag(R.T R) + lam.
          **Does not form Q**.
        - 'ilu': incomplete LU on sparse Q (forms Q as sparse).
        - 'amg': algebraic multigrid on sparse Q (requires `pyamg`, forms Q).
        - 'identity': explicit identity operator.

    precond_kwargs : dict, optional
        Options for the chosen preconditioner.

        For 'jacobi':
            - min_diagonal (float): floor on the diagonal before inversion
              (default 1e-30).

        For 'ilu' (scipy.sparse.linalg.spilu):
            - drop_tol (float)
            - fill_factor (float)
            - diag_pivot_thresh (float)
            - permc_spec (str)

        For 'amg' (pyamg.smoothed_aggregation_solver):
            - any kwargs accepted by pyamg, plus:
            - tol (float): solve tolerance used per application (default 1e-8)
            - maxiter (int): max iterations per application (default 1)

    Returns
    -------
    M : scipy.sparse.linalg.LinearOperator or None
        Preconditioner suitable for SciPy iterative solvers.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-12-19
    """
    if precond is None:
        return None

    key = str(precond).strip().lower()
    opts = {} if precond_kwargs is None else dict(precond_kwargs)

    if key in {"none"}:
        return None

    if key in {"identity", "i"}:
        n = R.shape[1]
        return LinearOperator((n, n), matvec=lambda x: x, dtype=np.float64)

    if key in {"jacobi", "diag"}:
        d = _rtr_diag(R)
        if lam != 0.0:
            d = d + float(lam)
        floor = float(opts.pop("min_diagonal", 1.0e-30))
        d = np.maximum(d, floor)
        invd = 1.0 / d

        def mvec(x: np.ndarray) -> np.ndarray:
            """Apply Jacobi Mx = diag(Q)^{-1} x."""
            return invd * x

        return LinearOperator((invd.size, invd.size), matvec=mvec, dtype=np.float64)

    # The remaining options require forming sparse Q.
    if not scipy.sparse.issparse(R):
        raise ValueError(
            f"Preconditioner {precond!r} requires sparse R (to form sparse Q)."
        )

    Q = (R.T @ R).tocsc()
    if lam != 0.0:
        Q = Q + float(lam) * scipy.sparse.eye_array(Q.shape[0], format="csc")

    if key in {"ilu"}:
        ilu = scipy.sparse.linalg.spilu(
            Q,
            drop_tol=float(opts.pop("drop_tol", 1.0e-4)),
            fill_factor=float(opts.pop("fill_factor", 10.0)),
            diag_pivot_thresh=float(opts.pop("diag_pivot_thresh", 0.0)),
            permc_spec=str(opts.pop("permc_spec", "COLAMD")),
        )

        def mvec(x: np.ndarray) -> np.ndarray:
            """Apply ILU preconditioner via one triangular solve."""
            return ilu.solve(x)

        return LinearOperator(Q.shape, matvec=mvec, dtype=np.float64)

    if key in {"amg"}:
        try:
            import pyamg  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("Preconditioner 'amg' requires pyamg.") from exc

        tol = float(opts.pop("tol", 1.0e-8))
        maxiter = int(opts.pop("maxiter", 1))
        ml = pyamg.smoothed_aggregation_solver(Q, **opts)

        def mvec(x: np.ndarray) -> np.ndarray:
            """Apply one (or few) AMG V-cycles as approximate inverse."""
            return ml.solve(x, tol=tol, maxiter=maxiter)

        return LinearOperator(Q.shape, matvec=mvec, dtype=np.float64)

    raise ValueError(
        "Unknown preconditioner. Use None|'jacobi'|'ilu'|'amg'|'identity' "
        f"(got {precond!r})."
    )


def make_sparse_cholesky_precision_solver(
    R: np.ndarray | scipy.sparse.sparray,
    lam: float = 0.0,
    *,
    use_cholmod: bool = True,
) -> Callable[[np.ndarray], np.ndarray]:
    """Construct a direct solver for Qx=b using sparse Cholesky (or fallback LU).

    Parameters
    ----------
    R : array_like or sparse matrix, shape (m, n)
        Matrix defining Q = R.T @ R (+ lam * I).
    lam : float, optional
        Diagonal shift added to Q.
    use_cholmod : bool, optional
        If True (default), try SuiteSparse CHOLMOD via ``scikit-sparse``.
        If unavailable, fall back to SciPy sparse LU with a RuntimeWarning.

    Returns
    -------
    solve_Q : callable
        A function solving Qx=b.

    Notes
    -----
    For large 3-D problems, explicit factorisation can be memory intensive
    due to fill-in. It is most useful when Q is moderate in size or when many
    solves with the same Q are needed.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-12-19
    """
    if scipy.sparse.issparse(R):
        Q = (R.T @ R).tocsc()
        if lam != 0.0:
            Q = Q + float(lam) * scipy.sparse.eye_array(Q.shape[0], format="csc")

        if use_cholmod:
            try:
                from sksparse.cholmod import cholesky  # type: ignore
            except Exception:  # pragma: no cover
                warnings.warn(
                    "CHOLMOD (scikit-sparse) not available; falling back to SciPy splu.",
                    RuntimeWarning,
                )
                lu = scipy.sparse.linalg.splu(Q)

                def solve_Q(b: np.ndarray) -> np.ndarray:
                    """Solve using SciPy sparse LU."""
                    return lu.solve(b)

                return solve_Q

            factor = cholesky(Q)

            def solve_Q(b: np.ndarray) -> np.ndarray:
                """Solve using CHOLMOD sparse Cholesky."""
                return factor(b)

            return solve_Q

        lu = scipy.sparse.linalg.splu(Q)

        def solve_Q(b: np.ndarray) -> np.ndarray:
            """Solve using SciPy sparse LU."""
            return lu.solve(b)

        return solve_Q

    # Dense fallback (small problems)
    Rm = np.asarray(R, dtype=np.float64)
    Q = Rm.T @ Rm
    if lam != 0.0:
        Q = Q + float(lam) * np.identity(Q.shape[0], dtype=np.float64)

    c, lower = scipy.linalg.cho_factor(Q, overwrite_a=False, check_finite=True)

    def solve_Q(b: np.ndarray) -> np.ndarray:
        """Solve using dense Cholesky."""
        return scipy.linalg.cho_solve((c, lower), b, check_finite=True)

    return solve_Q


def _rtr_diag(
    R: np.ndarray | scipy.sparse.sparray,
) -> np.ndarray:
    """Compute the diagonal of ``R.T @ R`` without forming the product.

    Consolidated single implementation; the former ``_diag_rtr`` alias has
    been removed.

    Parameters
    ----------
    R
        Dense or sparse matrix of shape (m, n).

    Returns
    -------
    ndarray
        1-D array of length ``n`` with entries ``diag(R.T @ R)``.

    Notes
    -----
    For a matrix ``R``, the diagonal entries of ``R.T @ R`` are the column-wise
    sum of squares::

        diag(R.T @ R)[j] = sum_i R[i, j]^2

    This helper computes that efficiently for both dense and sparse inputs
    without materialising the (n × n) product.
    """
    if issparse(R):
        Rsq = R.multiply(R)
        d = np.asarray(Rsq.sum(axis=0)).ravel()
        return d.astype(np.float64, copy=False)

    A = np.asarray(R, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(f"_rtr_diag: R must be 2-D, got shape {A.shape}.")
    return np.sum(A * A, axis=0, dtype=np.float64)


def pick_lam_from_rtr_diag(
    R: np.ndarray | scipy.sparse.sparray,
    *,
    alpha: float = 1.0e-5,
    statistic: str = "median",
    min_lam: float = 0.0,
) -> float:
    """Pick a diagonal shift ``lam`` from the scale of ``diag(R.T @ R)``.

    Implements the practical rule-of-thumb discussed in the chat:

        ``lam = alpha * median(diag(R.T @ R))``

    with ``alpha`` typically in the range ``1e-6 ... 1e-3``.

    Parameters
    ----------
    R
        Dense or sparse matrix defining the roughness / precision proxy.
    alpha
        Scale factor applied to the chosen statistic of ``diag(R.T @ R)``.
    statistic
        Which statistic to use for the diagonal scale. Supported values:
        ``"median"`` (default) and ``"mean"``.
    min_lam
        Lower bound for the returned lambda.

    Returns
    -------
    float
        Suggested ``lam`` value.

    Notes
    -----
    The diagonal entries of ``R.T @ R`` can contain zeros (e.g., constrained / fixed
    parameters). This helper uses only finite entries, and prefers positive entries
    when computing the statistic.
    """
    a = float(alpha)
    if not np.isfinite(a) or a < 0.0:
        raise ValueError(f"pick_lam_from_rtr_diag: alpha must be finite and >=0, got {alpha!r}.")

    d = _rtr_diag(R)
    finite = np.isfinite(d)
    if not finite.any():
        return float(min_lam)

    dfin = d[finite]
    pos = dfin > 0.0
    if pos.any():
        dsel = dfin[pos]
    else:
        dsel = dfin

    stat = statistic.strip().lower()
    if stat in {"median", "med"}:
        scale = float(np.median(dsel))
    elif stat in {"mean", "avg", "average"}:
        scale = float(np.mean(dsel))
    else:
        raise ValueError(f"pick_lam_from_rtr_diag: unsupported statistic={statistic!r}.")

    if not np.isfinite(scale) or scale <= 0.0:
        return float(min_lam)

    lam = a * scale
    if not np.isfinite(lam):
        lam = float(min_lam)
    return float(max(float(min_lam), float(lam)))


def _resolve_lam(
    R: np.ndarray | scipy.sparse.sparray,
    lam: float = 0.0,
    lam_mode: str = "fixed",
    lam_alpha: Optional[float] = None,
    lam_statistic: str = "median",
    lam_min: float = 0.0,
) -> float:
    """Resolve the effective diagonal shift for Q = R^T R + lam I.

    Single implementation of the rule used by :func:`make_precision_solver`
    and :func:`sample_rtr_full_rank` (the sampler needs the same value for its
    ``sqrt(lam) * xi2`` term).  See ``make_precision_solver`` for the modes.
    """
    mode = str(lam_mode).strip().lower()
    lam_eff = float(lam)
    if mode in {"scaled_median_diag", "median_diag", "scaled_diag", "diag"}:
        alpha = float(lam_alpha) if lam_alpha is not None else float(lam)
        lam_eff = pick_lam_from_rtr_diag(
            R,
            alpha=alpha,
            statistic=lam_statistic,
            min_lam=lam_min,
        )
    elif mode in {"auto", "auto_diag", "auto_median_diag"}:
        if not np.isfinite(lam_eff) or lam_eff <= 0.0:
            alpha = float(lam_alpha) if lam_alpha is not None else 1.0e-5
            lam_eff = pick_lam_from_rtr_diag(
                R,
                alpha=alpha,
                statistic=lam_statistic,
                min_lam=lam_min,
            )
        else:
            lam_eff = float(max(float(lam_min), lam_eff))
    else:
        lam_eff = float(max(float(lam_min), lam_eff))
    return lam_eff


def make_precision_solver(
    R: np.ndarray | scipy.sparse.sparray,
    lam: float = 0.0,
    atol: float = 1.0e-4,
    rtol: float = 1.e-3,
    maxiter: Optional[int] = None,
    M: Optional[LinearOperator] = None,
    msolver: Optional[str] = "cg",
    mprec: Optional[str] = "jacobi",
    *,
    solver_method: Optional[str] = None,
    precond: Optional[str] = None,
    precond_kwargs: Optional[dict] = None,
    use_cholmod: bool = True,
    lam_mode: str = "fixed",
    lam_alpha: Optional[float] = None,
    lam_statistic: str = "median",
    lam_min: float = 0.0,
) -> Callable[[np.ndarray], np.ndarray]:
    """Construct a solver for ``Qx=b`` with ``Q = R.T@R + lam*I``.

    This is a compatibility-preserving upgrade of the earlier FEMTIC helper:

    - **Iterative solvers**: CG (recommended for SPD Q) and BiCGStab.
    - **Direct solver**: sparse Cholesky via CHOLMOD (or SciPy LU fallback).

    New in this cleaned-up version
    ------------------------------
    Optional automatic diagonal shift selection:

        ``lam = alpha * median(diag(R.T @ R))``

    This can be enabled by setting ``lam_mode``.

    Parameters
    ----------
    R
        Operator defining the precision Q.
    lam
        Diagonal shift (used when ``lam_mode='fixed'``).
    rtol, atol
        Iterative solver tolerances (SciPy style).  **For sampling keep rtol
        tight (<= 1e-6)**: CG converges last on the smoothest modes, which
        carry the largest prior variance; in a Laplacian-type test the
        variance of the 12 smoothest modes was 72 % of target at rtol 1e-3
        and 9 % at rtol 1e-2 (2026-10-21 audit).  atol=0 lets rtol alone
        govern convergence.
    maxiter
        Maximum CG/BiCGStab iterations.  **Recommended 500–1000** to cap
        runaway solves on ill-conditioned problems.  With good λ and ILU
        preconditioning, convergence is often fast on FEMTIC-type problems,
        but it stagnated on a Laplacian-type lattice test (2026-10-21);
        check the iteration count and the energy ratio of the samples.
        None (default) is unlimited.
    M
        Explicit preconditioner LinearOperator.  If provided, overrides
        ``mprec/precond``.
    msolver
        Iterative solver choice (legacy argument name).
    mprec
        Preconditioner choice (legacy argument name). See ``precond``.
    solver_method
        New preferred name for the solver. Overrides ``msolver`` if provided.
        ``"cg"`` is optimal for SPD Q = R^T R + λI; use ``"bicgstab"`` only
        for non-symmetric operators (not applicable here).
    precond
        New preferred name for the preconditioner. Overrides ``mprec``.
        ``"ilu"`` often reduces the CG iteration count, but ``spilu`` is not
        a symmetric positive definite preconditioner and CG did not converge
        with it on a Laplacian-type test matrix (2026-10-21 audit); ``"jacobi"``
        (no explicit Q needed) is the safe choice.
        ``"amg"`` (requires pyamg) is the strongest option for very large
        systems.  ``None`` = no preconditioning (slow for ill-conditioned Q).
    precond_kwargs
        Extra options for the chosen preconditioner.
    use_cholmod
        If True (default), try CHOLMOD for the ``"cholesky"`` method.
    lam_mode
        How to interpret / choose ``lam``:

        - ``"fixed"``: use ``lam`` exactly (default).
        - ``"scaled_median_diag"`` (alias: ``"median_diag"``, **recommended**):
          use ``lam = alpha * median(diag(R.T@R))`` where ``alpha`` is
          ``lam_alpha`` if provided, otherwise ``lam``.
        - ``"auto"``: if ``lam > 0`` use it, else fall back to the scaled
          median-diagonal rule.
    lam_alpha
        Scale factor α for the scaled-diagonal rule.  **Recommended 1e-4**
        (range 1e-5 … 1e-3); this is the single biggest speed lever — a
        larger shift improves CG conditioning dramatically.  Raise to 1e-3
        if convergence is still slow.
    lam_statistic
        Statistic used for the diagonal scale (``"median"`` or ``"mean"``).
        ``"median"`` is robust to large outliers in the diagonal.
    lam_min
        Lower bound applied to the chosen ``lam``.

    Returns
    -------
    solve_Q
        Callable that solves ``Qx=b``.

    Author: Volker Rath (DIAS)
    Created with the help of ChatGPT (GPT-5 Thinking) on 2026-01-02 (UTC)
    """
    meth = (solver_method or msolver or "cg").strip().lower()

    # Resolve lam (optionally from diag(R.T R)); shared with sample_rtr_full_rank
    lam_eff = _resolve_lam(
        R, lam=lam, lam_mode=lam_mode, lam_alpha=lam_alpha,
        lam_statistic=lam_statistic, lam_min=lam_min,
    )

    if meth in {"chol", "cholesky", "sparse_cholesky", "direct"}:
        return make_sparse_cholesky_precision_solver(
            R=R,
            lam=lam_eff,
            use_cholmod=use_cholmod,
        )

    # Iterative branch
    Q_op = build_rtr_operator(R, lam=lam_eff)

    if M is None:
        M = make_rtr_preconditioner(
            R=R,
            lam=lam_eff,
            precond=precond if precond is not None else mprec,
            precond_kwargs=precond_kwargs,
        )

    def solve_Q(b: np.ndarray) -> np.ndarray:
        """Solve ``Qx=b`` using the selected iterative method."""
        if meth in {"cg", "pcg"}:
            x, info = cg(Q_op, b, rtol=rtol, atol=atol, maxiter=maxiter, M=M)
        else:
            x, info = bicgstab(Q_op, b, rtol=rtol, atol=atol, maxiter=maxiter, M=M)

        if info > 0:
            raise RuntimeError(
                f"Iterative solver did not converge within {info} iterations."
            )
        if info < 0:
            raise RuntimeError(f"Iterative solver failed (info={info}).")
        return np.asarray(x, dtype=np.float64)

    return solve_Q


def sample_rtr_full_rank(
    R: np.ndarray | scipy.sparse.sparray,
    n_samples: int = 1,
    lam: float = 1.0e-4,
    solver: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    rng: Optional[Generator] = None,
    *,
    solver_method: str = "cg",
    solver_kwargs: Optional[dict] = None,
    lam_mode: str = "fixed",
    lam_alpha: Optional[float] = None,
    lam_statistic: str = "median",
    lam_min: float = 0.0,
) -> np.ndarray:
    """Sample from ``N(0, (R.T@R + lam I)^{-1})`` using full-rank solves.

    Parameters
    ----------
    R
        Dense or sparse matrix of shape (m, n) defining the precision.
    n_samples
        Number of samples.
    lam
        Diagonal shift. Interpreted according to ``lam_mode``.
    solver
        Pre-built solver for ``Qx=b``. If None, it is created from ``solver_method``.
    rng
        Random generator.
    solver_method
        Solver used when ``solver`` is None ("cg", "bicgstab", "cholesky").
    solver_kwargs
        Extra args passed to :func:`make_precision_solver`.  When the solver
        is built here, ``rtol`` defaults to 1e-8 and ``atol`` to 0 (changed
        2026-10-21 from the loose make_precision_solver defaults 1e-3 / 1e-4,
        which under-resolve the smoothest modes); pass ``rtol``/``atol`` to
        override.
    lam_mode, lam_alpha, lam_statistic, lam_min
        Passed through to :func:`make_precision_solver`. In particular,

        - ``lam_mode='scaled_median_diag'`` (recommended) activates
          ``lam = lam_alpha * median(diag(R.T@R))``.
        - ``lam_alpha`` is the key speed lever: **recommended 1e-4**
          (range 1e-5 … 1e-3).  A larger value improves conditioning and
          typically halves iteration counts vs the default 1e-5.

    Returns
    -------
    ndarray
        Array of shape ``(n_samples, n)``.

    Notes
    -----
    Uses the identity::

        x = (R.T R + lam I)^{-1} ( R.T ξ1 + sqrt(lam) ξ2 ),   ξ1 ~ N(0, I_m), ξ2 ~ N(0, I_n)

    whose covariance is exactly (R.T R + lam I)^{-1}.  (Until 2026-10-21 the
    ``sqrt(lam) ξ2`` term was missing, giving Q^{-1} R.T R Q^{-1}: the
    smoothest modes were damped by e/(e+lam) and the null space of R had no
    variance.)  ``lam`` is resolved with the same rule as in
    :func:`make_precision_solver`; if a prebuilt ``solver`` is passed, pass the
    same ``lam``/``lam_mode``/``lam_alpha``/``lam_statistic``/``lam_min`` that
    were used to build it.  For RTO prefer :func:`sample_rto_prior`, which also
    applies the regularisation weight alpha.

    Author: Volker Rath (DIAS)
    Created with the help of ChatGPT (GPT-5 Thinking) on 2026-01-02 (UTC)
    Corrected by Claude Sonnet 5.5 (Anthropic) on 2026-10-21.
    """
    rng = default_rng() if rng is None else rng
    m, n = R.shape

    lam_eff = _resolve_lam(
        R, lam=lam, lam_mode=lam_mode, lam_alpha=lam_alpha,
        lam_statistic=lam_statistic, lam_min=lam_min,
    )
    sqrt_lam = float(np.sqrt(lam_eff)) if lam_eff > 0.0 else 0.0

    if solver is None:
        kw = dict(solver_kwargs or {})
        # Sampling needs a tight tolerance: CG converges last on the smoothest modes,
        # which carry most of the prior variance (rtol 1e-3 gave 0.73 and 1e-2 gave
        # 0.50 of the exact RMS std in the 2026-10-21 lattice test).  Override via
        # solver_kwargs if speed matters more than accuracy.
        kw.setdefault("rtol", 1.0e-8)
        kw.setdefault("atol", 0.0)
        solver = make_precision_solver(
            R=R,
            lam=lam,
            solver_method=solver_method,
            lam_mode=lam_mode,
            lam_alpha=lam_alpha,
            lam_statistic=lam_statistic,
            lam_min=lam_min,
            **kw,
        )

    samples = np.empty((n_samples, n), dtype=np.float64)
    for ix in range(n_samples):
        xi = rng.standard_normal(size=m)
        b = R.T @ xi
        if sqrt_lam > 0.0:
            b = b + sqrt_lam * rng.standard_normal(size=n)
        samples[ix, :] = solver(b)

    return samples


def sample_rtr_low_rank(
    R: np.ndarray | scipy.sparse.sparray,
    n_samples: int = 1,
    n_eig: int = 32,
    sigma2_residual: float = 0.0,
    rng: Optional[Generator] = None,
    *,
    n_oversampling: int = 10,
    n_power_iter: int = 2,
) -> np.ndarray:
    """DEPRECATED - do not use for RTO.  See :func:`sample_rto_prior`.

    **Known error (2026-10-21 audit).**  ``randomized_svd`` returns the
    *largest* singular values s_k of R, i.e. the roughest modes.  For
    N(0, (R^T R)^-1) the variance of mode k is 1/s_k^2, so those are the
    modes with the *smallest* variance; the smooth modes that carry most of
    the prior variance are discarded.  In a 3-D lattice test (n = 2744,
    n_eig = 128) the retained modes held 0.08-0.5 % of the prior variance,
    the roughness-term energy ratio was 0.05 instead of 1 and an RTO
    ensemble built on it had 0.20-0.27 of the correct spread; the loss grows
    as n_eig/n_free shrinks.  More ``n_eig`` or power iterations do not help.
    A correct low-rank sampler needs the *smallest* eigenpairs of R^T R
    (shift-invert / Lanczos) plus a model for the remainder.

    Original description:
    Approximate sampling from N(0, (R^T R)^{-1}) using randomized SVD of R.

    Replaces the former ``eigsh``-based approach (which targeted smallest
    eigenvalues of Q = R^T R and was prohibitively slow for large problems).
    Randomized SVD operates directly on R via O(n_eig) matvecs with R and
    R^T, and converges rapidly regardless of the spectral gap.

    The leading k right singular vectors V_k of R are the leading eigenvectors
    of Q = R^T R, with eigenvalues sigma_k^2.  Sampling uses::

        x = V_k @ (1/sigma_k * z),   z ~ N(0, I_k)

    which draws from N(0, (R^T R)^{-1}) restricted to the rank-k subspace.
    An optional isotropic residual N(0, sigma2_residual * I) can be added to
    represent unresolved directions.

    Parameters
    ----------
    R : array_like or sparse matrix, shape (m, n)
        Roughness matrix defining Q = R^T R.
    n_samples : int
        Number of samples to draw.
    n_eig : int
        Number of singular triplets (rank of the low-rank approximation).
        **Recommended 128–256** for FEMTIC meshes: more modes give smoother,
        more faithful samples; cost scales linearly with n_eig.  Values below
        64 may miss large-scale structure; values above 512 give diminishing
        returns for typical roughness spectra.
    sigma2_residual : float
        Variance of an isotropic residual added to each sample to account for
        directions not captured by the rank-k approximation.
        **Recommended ~1e-3** (~10 % of typical log10(ρ) variance).
        Set to 0 to disable (default); without it, samples live entirely in
        the rank-k subspace and lack short-wavelength variability.
    rng : numpy.random.Generator, optional
        Random generator.  If None, uses ``np.random.default_rng()``.
    n_oversampling : int
        Extra columns drawn in the randomized range-finder (default 10).
        10–15 is sufficient for nearly all cases; increasing beyond 20
        gives negligible accuracy improvement.
    n_power_iter : int
        Number of power-iteration steps to sharpen the range approximation.
        **Recommended 3–4** for FEMTIC roughness matrices, whose spectra
        decay slowly (each iteration roughly halves the range-finder error
        at O(n_eig) matvec cost).  Default 2 is a safe minimum.

    Returns
    -------
    samples : ndarray, shape (n_samples, n)
        Samples from the rank-k approximation of N(0, Q^{-1}).

    Notes
    -----
    Requires ``sklearn.utils.extmath.randomized_svd``.  This is a standard
    dependency of the scientific Python stack and is already present wherever
    ``scikit-learn`` is installed.

    Author: Volker Rath (DIAS)
    Updated by Claude (Anthropic) on 2026-03-31 — replaced eigsh with
    randomized SVD for O(n_eig) matvec cost and reliable convergence.
    """
    from sklearn.utils.extmath import randomized_svd

    warnings.warn(
        "sample_rtr_low_rank keeps the LARGEST singular values of R (the roughest "
        "modes, smallest prior variance) and therefore gives far too small "
        "perturbations; use sample_rto_prior instead.",
        RuntimeWarning, stacklevel=2,
    )
    rng = default_rng() if rng is None else rng
    # sklearn's randomized_svd accepts a random_state int or RandomState;
    # extract a seed from our Generator for compatibility.
    seed = int(rng.integers(0, 2**31))

    _, s, Vt = randomized_svd(
        R,
        n_components=n_eig,
        n_oversamples=n_oversampling,
        n_iter=n_power_iter,
        random_state=seed,
    )

    s = np.asarray(s, dtype=np.float64)
    Vt = np.asarray(Vt, dtype=np.float64)   # (n_eig, n)
    n = Vt.shape[1]

    # Guard: singular values must be positive for inversion
    if np.any(s <= 0.0):
        warnings.warn(
            "sample_rtr_low_rank: some singular values are <= 0; "
            "clamping to machine epsilon.",
            RuntimeWarning,
        )
        s = np.maximum(s, np.finfo(float).eps)

    inv_s = 1.0 / s   # 1/sigma_k  (= 1/sqrt(eigenvalue_k) of Q)

    samples = np.empty((n_samples, n), dtype=np.float64)
    for ix in range(n_samples):
        z = rng.standard_normal(size=n_eig)
        x = Vt.T @ (inv_s * z)          # (n,)
        if sigma2_residual > 0.0:
            x = x + np.sqrt(sigma2_residual) * rng.standard_normal(size=n)
        samples[ix, :] = x

    return samples


def sample_rto_prior(
    R: np.ndarray | scipy.sparse.sparray,
    n_samples: int = 1,
    *,
    alpha: float = 1.0,
    lam: Optional[float] = None,
    lam_alpha: float = 1.0e-4,
    method: str = "auto",
    rtol: float = 1.0e-8,
    maxiter: Optional[int] = None,
    direct_max_n: int = 60000,
    rng: Optional[Generator] = None,
    return_info: bool = False,
):
    """Exact samples from ``N(0, (alpha (R^T R + lam I))^-1)`` for RTO.

    This is the prior perturbation of the reference model in
    Randomize-Then-Optimize: the member inversion minimises

        (d~ - F(m))^T Cd^-1 (d~ - F(m)) + alpha |R (m - m~_ref)|^2,

    with d~ ~ N(d, Cd) and m~_ref = m_ref + x, x ~ N(0, (alpha R^T R)^-1)
    (Bardsley et al. 2014; Blatter et al. 2022, Eq. 5).

    Method (exact in exact arithmetic, for any solver of Q = R^T R + lam I)::

        x = alpha^(-1/2) Q^(-1) ( R^T xi1 + sqrt(lam) xi2 ),
        xi1 ~ N(0, I_m), xi2 ~ N(0, I_n)

        Cov(x) = alpha^(-1) Q^(-1) (R^T R + lam I) Q^(-1) = (alpha Q)^(-1).

    Parameters
    ----------
    R : sparse matrix or ndarray, shape (m, n)
        Roughness matrix (not R^T R).
    n_samples : int
        Number of draws.
    alpha : float
        Regularisation weight of the member inversions (prior precision is
        alpha * R^T R).  Samples scale as alpha^(-1/2).
    lam : float or None
        Diagonal shift.  None or <= 0: ``lam = lam_alpha * median(diag(R^T R))``
        (:func:`pick_lam_from_rtr_diag`).  Needed because R^T R is singular
        (constants); a constant offset of the reference model does not enter
        |R (m - m_ref)|^2, so the shift does not matter for the inversion.
    method : {"auto", "splu", "cg"}
        "splu": sparse LU of Q, factored once, all right-hand sides at once.
        "cg": Jacobi-preconditioned CG with tolerance ``rtol`` (needs a tight
        ``rtol``: CG converges last on the smoothest modes, which carry most of
        the prior variance).  "auto": "splu" if n <= ``direct_max_n``, else "cg".
        (ILU preconditioning is not used: ``spilu`` is not a symmetric positive
        definite preconditioner and did not converge in tests.)
    rtol, maxiter : float, int
        CG controls.  A RuntimeWarning is issued if CG does not converge.
    rng : numpy.random.Generator, optional
    return_info : bool
        If True return ``(samples, info)`` with ``info = {"lam", "method", "n"}``.

    Returns
    -------
    samples : ndarray, shape (n_samples, n)
        Perturbations in the unit R acts on (log10 resistivity in FEMTIC workflows).

    Notes
    -----
    Replaces :func:`sample_rtr_low_rank` (wrong end of the spectrum) in the RTO
    workflow.  Check a result with :func:`prior_energy_ratio` (should be ~ 1).

    Author: Claude Sonnet 5.5 (Anthropic), 2026-10-21.
    AI-generated; please review before production use.
    """
    if not (np.isfinite(alpha) and alpha > 0.0):
        raise ValueError(f"sample_rto_prior: alpha must be positive and finite, got {alpha!r}.")
    rng = default_rng() if rng is None else rng
    if issparse(R):
        Rm = R.tocsr()
    else:
        Rm = scipy.sparse.csr_array(np.asarray(R, dtype=np.float64))
    m, n = Rm.shape

    lam_eff = float(lam) if (lam is not None and lam > 0.0) else pick_lam_from_rtr_diag(
        Rm, alpha=lam_alpha, statistic="median")
    if not (np.isfinite(lam_eff) and lam_eff > 0.0):
        raise ValueError("sample_rto_prior: diagonal shift lam must be > 0 (R^T R is singular for "
                         "roughness operators); got lam=%r." % (lam_eff,))

    Q = (Rm.T @ Rm + lam_eff * scipy.sparse.eye_array(n)).tocsc()

    xi1 = rng.standard_normal(size=(m, n_samples))
    xi2 = rng.standard_normal(size=(n, n_samples))
    B = np.asarray(Rm.T @ xi1, dtype=np.float64) + np.sqrt(lam_eff) * xi2        # (n, n_samples)

    meth = str(method).strip().lower()
    if meth in {"auto"}:
        meth = "splu" if n <= int(direct_max_n) else "cg"
    if meth in {"splu", "lu", "direct"}:
        meth = "splu"
        X = scipy.sparse.linalg.splu(Q).solve(np.ascontiguousarray(B))
    elif meth in {"cg", "pcg"}:
        meth = "cg"
        dq = Q.diagonal()
        M = LinearOperator(Q.shape, matvec=lambda v: v / dq, dtype=np.float64)   # Jacobi (SPD)
        mi = int(maxiter) if maxiter is not None else max(5000, 2 * n)
        X = np.empty_like(B)
        for j in range(n_samples):
            x, info_cg = cg(Q, B[:, j], rtol=float(rtol), atol=0.0, M=M, maxiter=mi)
            if info_cg != 0:
                warnings.warn(
                    f"sample_rto_prior: CG did not reach rtol={rtol:g} for sample {j} "
                    f"(info={info_cg}); the long-wavelength modes are then too small. "
                    "Use method='splu' or raise maxiter.", RuntimeWarning, stacklevel=2)
            X[:, j] = x
    else:
        raise ValueError("sample_rto_prior: method must be 'auto', 'splu' or 'cg'.")

    samples = (X / np.sqrt(alpha)).T
    if return_info:
        return samples, {"lam": lam_eff, "method": meth, "n": n}
    return samples


def prior_energy_ratio(
    R: np.ndarray | scipy.sparse.sparray,
    samples: np.ndarray,
    alpha: float = 1.0,
    rank: Optional[int] = None,
) -> float:
    """Diagnostic for RTO prior perturbations: mean(alpha |R x|^2) / rank.

    For ``x ~ N(0, (alpha (R^T R + lam I))^-1)`` this is ~ 1 (it is the
    chi^2 per degree of freedom of the roughness term).  Values << 1 mean the
    perturbation is too weak or confined to too few directions for the
    objective ``alpha |R (m - m_ref)|^2`` to feel it; the former randomized-SVD
    sampler gave ~ n_eig / n_free.

    Parameters
    ----------
    R : roughness matrix, shape (m, n)
    samples : ndarray, shape (n_samples, n) or (n,)
        Perturbations (e.g. from :func:`sample_rto_prior`, or the difference of
        a member's perturbed and original log10 reference model).
    alpha : float
        Regularisation weight used for the draws.
    rank : int, optional
        Rank of R; default ``min(R.shape) - 1`` (constants in the null space).
    """
    Rm = R.tocsr() if issparse(R) else scipy.sparse.csr_array(np.asarray(R, dtype=np.float64))
    X = np.atleast_2d(np.asarray(samples, dtype=np.float64))
    RX = np.asarray(Rm @ X.T)                                   # (m, n_samples)
    e = float(alpha) * np.sum(RX ** 2, axis=0)
    r = (min(Rm.shape) - 1) if rank is None else int(rank)
    return float(np.mean(e) / r)


def sample_prior_from_roughness(
    R: np.ndarray | scipy.sparse.sparray,
    n_samples: int = 1,
    lam: float = 1.0e-4,
    mode: Literal["full", "low-rank"] = "full",
    n_eig: int = 32,
    sigma2_residual: float = 0.0,
    rng: Optional[Generator] = None,
    msolver: Optional[str] = "cg",
    mprec: Optional[str] = "jacobi",
    rtol: float = 1.0e-6,
    atol: float = 0.0,
    maxiter: Optional[int] = None,
    *,
    solver_method: Optional[str] = None,
    precond: Optional[str] = None,
    precond_kwargs: Optional[dict] = None,
    use_cholmod: bool = True,
) -> np.ndarray:
    """Sample from the Gaussian prior implied by a FEMTIC roughness matrix.

    The implied precision is::

        Q = R.T @ R + lam * I

    Parameters
    ----------
    R : array_like or sparse matrix
        Roughness matrix.
    n_samples : int
        Number of samples.
    lam : float
        Diagonal shift in Q.
    mode : {"full", "low-rank"}
        Full-rank sampling uses iterative/direct solves. Low-rank uses eigenpairs.
    n_eig : int
        Number of eigenpairs in low-rank mode.
    sigma2_residual : float
        Residual isotropic variance in low-rank mode.
    rng : numpy.random.Generator, optional
    msolver, mprec, rtol, atol, maxiter
        Legacy iterative-solver settings (kept for compatibility).
    solver_method, precond, precond_kwargs, use_cholmod
        Preferred new names; see :func:`make_precision_solver`.

    Returns
    -------
    samples : ndarray, shape (n_samples, n)
        Samples from N(0, Q^{-1}).

    Notes
    -----
    ``mode="low-rank"`` uses the deprecated :func:`sample_rtr_low_rank`
    (wrong end of the spectrum).  For RTO use :func:`sample_rto_prior`.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-12-19
    """
    mode_l = str(mode).lower()
    if mode_l.startswith("low"):
        return sample_rtr_low_rank(
            R=R,
            n_samples=n_samples,
            n_eig=n_eig,
            sigma2_residual=sigma2_residual,
            rng=rng,
        )

    solver = make_precision_solver(
        R=R,
        lam=lam,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
        msolver=msolver,
        mprec=mprec,
        solver_method=solver_method,
        precond=precond,
        precond_kwargs=precond_kwargs,
        use_cholmod=use_cholmod,
    )
    return sample_rtr_full_rank(
        R=R,
        n_samples=n_samples,
        lam=lam,
        solver=solver,
        rng=rng,
        solver_method=solver_method or (msolver or "cg"),
        solver_kwargs={},  # solver already built above
    )

def randomized_smooth_modes(R, k=20, oversample=10,
                            n_iter=3, shift=1e-6,
                            random_state=None):
    """
    Approximate the k smoothest eigenmodes of R = L.T @ L.

    Returns
    -------
    eigenvalues : ndarray
        Approximate smallest eigenvalues of R.
    modes : ndarray
        Corresponding orthonormal spatial modes.

    Author: Volker Rath (DIAS)
    Created with the help of ChatGPT (GPT-6) on 2026-10-24
    """
    n = R.shape[0]
    ell = min(n, k + oversample)

    rng = np.random.default_rng(random_state)
    omega = rng.standard_normal((n, ell))

    A = (R + shift * eye(n, format="csc")).tocsc()
    solver = splu(A)

    Q, _ = np.linalg.qr(omega, mode="reduced")

    for _ in range(n_iter):
        Y = solver.solve(Q)
        Q, _ = np.linalg.qr(Y, mode="reduced")

    B = Q.T @ (R @ Q)
    values, vectors = eigh(B)

    idx = np.argsort(values)[:k]

    return values[idx], Q @ vectors[:, idx]
