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
- Ensemble analysis tools (EOFs / PCA).
- Weighted KL decomposition and a KL+PCE surrogate builder for log-resistivity
  fields on unstructured FEMTIC meshes.
- GST parameter estimation: four strategies for choosing variogram parameters
  and pilot-point layouts (gst_variogram_from_rto_samples,
  gst_pilot_point_cv, gst_sill_from_jacobian, gst_parameter_diagnostics).

All functions are importable; no code is executed on import.

Author: Volker Rath (DIAS)
Created with the help of ChatGPT (GPT-5 Thinking) on 2026-01-02 (UTC)
Updated 2026-03-31 by Claude (Anthropic): removed debug print in
sample_rtr_full_rank; removed dead commented-out code in
generate_rto_model_ensemble; removed redundant per-sample print.
Updated 2026-03-31 by Claude (Anthropic): consolidated _diag_rtr
into _rtr_diag (single helper); removed dead estimate_low_rank_eigpairs;
enriched docstrings with tuning recommendations.
Updated 2026-04-02 by Claude (Anthropic): fixed FileNotFoundError in
generate_rto_model_ensemble — template argument to fem.insert_model now uses
the full per-member path (_orig.dat backup) instead of the bare basename.
Updated 2026-04-02 by Claude (Anthropic): fixed generate_rto_model_ensemble
write-back loop — now reads reference log10-resistivity from the backup
template and adds the perturbation before calling insert_model (method='add'),
so perturbed models are reference + delta_log10 rather than bare perturbations.
Updated 2026-04-11 by Claude Sonnet 4.6 (Anthropic): moved check_sparse_matrix
here from femtic.py (consolidation of all matrix/roughness tools into ensembles);
femtic.py Section 2 now imports these functions from ensembles rather than
duplicating them.
Updated 2026-04-27 by Claude Sonnet 4.6 (Anthropic): added
generate_gst_model_ensemble — geostatistical initial-model ensemble via
pilot-point Ordinary Kriging (gstools).  No roughness matrix required.
Updated 2026-04-27 by Claude Sonnet 4.6 (Anthropic): renamed
generate_model_ensemble to generate_rto_model_ensemble for consistency
with generate_gst_model_ensemble.
Updated 2026-05-28 by Claude Sonnet 4.6 (Anthropic): put_files and
generate_directories gained a relative_links parameter (default True):
relative symlinks survive tgz/copy to another machine; False restores the
previous absolute-path behaviour.
Updated 2026-05-28 by Claude Sonnet 4.6 (Anthropic): fixed template-clobbering
bug in generate_gst_model_ensemble — when output_target='both', the first
insert_model call (resistivity_block) could overwrite reference_file before
the second call read it as template; template_path is now resolved once before
either write.
Updated 2026-05-28 by Claude Sonnet 4.6 (Anthropic): added GST parameter
estimation section — four functions for choosing variogram parameters before
committing to a full ensemble run: gst_variogram_from_rto_samples (Strategy 1:
fit variogram to RTO samples), gst_pilot_point_cv (Strategy 2: LOO-CV on
reference model), gst_sill_from_jacobian (Strategy 3: linearised Jacobian
sill calibration), gst_parameter_diagnostics (Strategy 4: integrating
diagnostic with optional plot).
Updated 2026-06-06 by Claude Sonnet 4.6 (Anthropic): added "extrema" pilot-
point mode to generate_gst_model_ensemble.  New helper
_find_extrema_pilot_points (KDTree-based local extremum detection on free-
region barycentres, optional ROI mask).  New parameters pp_roi, pp_extrema_k,
pp_extrema_which; graceful fallback to "random" if no extrema are found.
Requires scipy.spatial (already a transitive dependency).
Updated 2026-06-10 by Claude Sonnet 4.6 (Anthropic): added _resolve_fromto
helper; all four ensemble functions (generate_directories,
generate_rto_model_ensemble, generate_gst_model_ensemble,
generate_data_ensemble) now accept an explicit list of member indices in
addition to None (all).  Range semantics ([start, stop]) removed — a list
always means explicit indices.  Type hints updated to Optional[List[int]];
Union/Tuple removed from fromto signatures.  Matching change in
femtic_rto_prep.py: FROM_TO renamed to ENS_LIST.
Updated 2026-07-05 by Claude Sonnet 5 (Anthropic): added pp_value_mode /
pp_value_delta to generate_gst_model_ensemble.  pp_value_mode="uniform"
(default) preserves the original Uniform(log_rho_min, log_rho_max) draw;
pp_value_mode="reference" instead draws pilot-point values as
reference_model(nearest free region) ± pp_value_delta (log10 Ohm.m),
using a scipy.spatial.KDTree nearest-neighbour lookup against the free-
region barycentres.  Reference log10(rho) at free regions is now computed
unconditionally (previously only inside the "extrema" pp_mode branch) so
it is available to both "extrema" placement and "reference" value mode.
Updated 2026-07-05 by Claude Sonnet 5 (Anthropic): raised the default
neighbourhood size for extremum detection from k=9 to k=30 in both
_find_extrema_pilot_points and generate_gst_model_ensemble's pp_extrema_k
— the strictly-less/greater-than-all-neighbours test was flagging too
many spurious local minima/maxima at small k on typical FEMTIC meshes;
recommended range updated from 7-15 to 20-40.
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
from scipy.sparse import isspmatrix, issparse
from scipy.sparse.linalg import LinearOperator, cg, eigsh, bicgstab, spilu

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
) -> scipy.sparse.spmatrix:
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
    from scipy.sparse import csr_array, csc_array, coo_array, eye as eye_array

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
    rough: scipy.sparse.spmatrix,
    regeps: float = 1.0e-5,
    spformat: str = "csr",
    spthresh: float = 1.0e-4,
    spfill: float = 10.0,
    spsolver: Optional[str] = "ilu",
    spmeth: str = "basic,area",
    outmatrix: str = "invRTR",
    nthreads: int = 16,
    out: bool = True,
) -> scipy.sparse.spmatrix | np.ndarray:
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
        eye as eye_array,
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


def prune_inplace(M: scipy.sparse.spmatrix, threshold: float) -> scipy.sparse.spmatrix:
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


def prune_rebuild(M: scipy.sparse.spmatrix, threshold: float) -> scipy.sparse.spmatrix:
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
) -> scipy.sparse.spmatrix:
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
    M: np.ndarray | scipy.sparse.spmatrix,
    howto: str = "relative",
    spformat: str = "csr",
    spthresh: float = 1.0e-6,
    prune: str = "rebuild",
    out: bool = True,
) -> scipy.sparse.spmatrix:
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


def check_sparse_matrix(M: scipy.sparse.spmatrix, condition: bool = True) -> None:
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
    q: Optional[scipy.sparse.spmatrix | np.ndarray] = None,
    algo: str = "low rank",
    method: str = "add",
    # --- low-rank options (randomized SVD) ---
    n_eig: int = 64,
    n_oversampling: int = 10,
    n_power_iter: int = 2,
    sigma2_residual: float = 0.0,
    # --- full-rank options (iterative solver) ---
    lam: float = 0.0,
    lam_mode: str = "scaled_median_diag",
    lam_alpha: float = 1.0e-5,
    solver_method: str = "cg",
    precond: str = "ilu",
    rng: Optional[Generator] = None,
    out: bool = True,
) -> list[str]:
    """
    Generate a resistivity model ensemble by sampling from N(0, (R^T R)^{-1}).

    Pass the roughness matrix **R** directly (not Q = R^T R); both sampling
    branches work with R and form Q implicitly.

    Two sampling strategies are supported via ``algo``:

    - ``"low rank"`` — randomized SVD of R (fast, recommended).
      Computes the leading k right singular vectors/values of R in
      O(k) matvec passes, then samples in that subspace.  Much faster
      than the former ``eigsh``-based approach.
    - ``"full rank"`` — iterative CG solves for each sample (accurate,
      but slower per sample for very large R).

    The resulting log10-resistivity perturbations are inserted into the
    FEMTIC ``resistivity_block_iterX.dat`` file in each ensemble directory.

    Parameters
    ----------
    dir_base : str
        Ensemble base directory (e.g. ``"./ens_"``).
    n_samples : int
        Number of samples to generate.
    fromto : None, (int, int), or list of int, optional
        Controls which members are written.  See :func:`_resolve_fromto`.
    refmod : str
        Name of the reference block file inside each ensemble directory.
    q : sparse matrix or ndarray
        **Roughness matrix R** (not Q).  Both branches use R directly.
    algo : {"low rank", "full rank"}
        Sampling algorithm.
    method : {"add", "replace"}
        How to combine samples with existing log10 resistivity:
        ``"add"`` adds the perturbation; ``"replace"`` overwrites.

    Low-rank parameters (``algo="low rank"``) — recommended for production
    -----------------------------------------------------------------------
    n_eig : int
        Number of singular triplets retained (rank of approximation).
        **Recommended 128–256** for FEMTIC meshes; more modes give smoother,
        more faithful samples and cost scales linearly with n_eig.
    n_oversampling : int
        Extra columns in the randomized range-finder.  10–15 is sufficient
        for nearly all cases; rarely needs increasing.
    n_power_iter : int
        Power-iteration steps to sharpen the range-finder.
        **Recommended 3–4** for FEMTIC roughness matrices, whose singular-value
        spectra decay slowly; each iteration roughly halves the approximation
        error at O(n_eig) matvec cost.
    sigma2_residual : float
        Isotropic residual variance added to each sample to represent
        directions outside the rank-k subspace.  **Recommended ~1e-3**
        (~10 % of typical log10(ρ) variance); set to 0 to disable.

    Full-rank parameters (``algo="full rank"``) — for accuracy verification
    ------------------------------------------------------------------------
    lam : float
        Diagonal shift seed (interpreted according to ``lam_mode``).
    lam_mode : str
        ``"scaled_median_diag"`` (default, recommended) or ``"fixed"``.
        The auto rule sets λ = lam_alpha × median(diag(R^T R)).
    lam_alpha : float
        Scale factor α for the scaled-diagonal λ rule.
        **Recommended 1e-4** (range 1e-5 … 1e-3); this is the single
        biggest speed lever — a larger shift improves CG conditioning
        dramatically.  Raise to 1e-3 if convergence is still slow.
    solver_method : str
        ``"cg"`` (default, optimal for SPD Q) or ``"bicgstab"``.
        CG is always preferred here; BiCGStab gives no benefit for
        symmetric positive-definite systems.
    precond : str
        Preconditioner: ``"ilu"`` (default, recommended) or ``"jacobi"``.
        ILU typically reduces CG iteration count by 3–5× compared with
        Jacobi, at the cost of forming sparse Q = R^T R once.

    Other
    -----
    rng : numpy.random.Generator, optional
        Shared random generator.
    out : bool
        If True, print status messages.

    Returns
    -------
    mod_list : list of str
        Paths to the perturbed resistivity block files.
    """
    if q is None:
        raise ValueError("generate_rto_model_ensemble: roughness matrix R (q) must be provided.")

    rng = default_rng() if rng is None else rng

    if "low" in algo.lower():
        if out:
            print(f"Low-rank sampling started (n_eig={n_eig}, randomized SVD).")
        samples = sample_rtr_low_rank(
            q,
            n_samples=n_samples,
            n_eig=n_eig,
            sigma2_residual=sigma2_residual,
            rng=rng,
            n_oversampling=n_oversampling,
            n_power_iter=n_power_iter,
        )
        if out:
            print("Low-rank sampling done.")
    else:
        if out:
            print("Full-rank sampling started (CG).")
        samples = sample_rtr_full_rank(
            R=q,
            n_samples=n_samples,
            lam=lam,
            rng=rng,
            solver_method=solver_method,
            solver_kwargs={"precond": precond},
            lam_mode=lam_mode,
            lam_alpha=lam_alpha,
        )
        if out:
            print("Full-rank sampling done.")

    fromto_arr = _resolve_fromto(fromto, n_samples)

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
        if method.lower() == "add":
            model_log10 = ref_log10 + sample
        else:  # "replace"
            model_log10 = sample

        fem.insert_model(
            template=orig_file,
            model=model_log10,
            model_file=file,
            model_name=f"sample{iens}")

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
    pp_bbox: Sequence[float] = (-50000., 50000., -50000., 50000., 0., 80000.),
    pp_coords: Optional[np.ndarray] = None,
    pp_roi: Optional[Sequence[float]] = None,
    pp_extrema_k: int = 30,
    pp_extrema_which: str = "both",
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
    pp_bbox : sequence of 6 floats
        Bounding box ``[x_min, x_max, y_min, y_max, z_min, z_max]`` (metres,
        z positive-down) for random pilot-point placement.
    pp_coords : ndarray, shape (N, 3), optional
        Explicit pilot-point coordinates (easting, northing, depth).
        Required when ``pp_mode`` is ``"fixed"`` or ``"mixed"``.
    pp_roi : sequence of 6 floats, optional
        ``[x_min, x_max, y_min, y_max, z_min, z_max]`` (metres, z positive-
        down) restricting the extremum search to a sub-volume of the model.
        Only used when ``pp_mode`` is ``"extrema"``.  None = full free-region
        extent.  Tip: set tighter than ``pp_bbox`` to exclude padding cells.
    pp_extrema_k : int
        Neighbourhood size (including self) for the local extremum test in
        ``"extrema"`` mode.  Larger k yields fewer, smoother extrema.
        Recommended: 20–40; increase further (e.g. 30+) if too many
        spurious minima/maxima are detected on dense or noisy meshes.
        Default: 30.
    pp_extrema_which : {"both", "minima", "maxima"}
        Which extrema to use as pilot-point seeds in ``"extrema"`` mode.
        ``"both"`` (default) seeds both conductive and resistive anomaly cores.

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
    out : bool
        If True, print progress messages.

    Returns
    -------
    mod_list : list of str
        Paths to all files written (one or two per member, depending on
        ``output_target``).

    Notes
    -----
    Requires ``gstools`` (pip install gstools).

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

    pp_bbox = list(pp_bbox)

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
    # Member loop.
    # ------------------------------------------------------------------
    fromto_arr = _resolve_fromto(fromto, n_samples)

    if out:
        print(f"\nGenerating {len(fromto_arr)} geostatistical initial models ...")

    mod_list: list[str] = []

    for iens in fromto_arr:
        member_dir = f"{dir_base}{iens}/"

        # --- pilot-point locations ---
        if pp_mode == "random":
            pp_x = rng.uniform(pp_bbox[0], pp_bbox[1], n_pp)
            pp_y = rng.uniform(pp_bbox[2], pp_bbox[3], n_pp)
            pp_z = rng.uniform(pp_bbox[4], pp_bbox[5], n_pp)
        elif pp_mode == "fixed":
            pp_x = pp_fixed[:, 0]
            pp_y = pp_fixed[:, 1]
            pp_z = pp_fixed[:, 2]
        elif pp_mode == "mixed":
            rnd_x = rng.uniform(pp_bbox[0], pp_bbox[1], n_pp)
            rnd_y = rng.uniform(pp_bbox[2], pp_bbox[3], n_pp)
            rnd_z = rng.uniform(pp_bbox[4], pp_bbox[5], n_pp)
            pp_x = np.concatenate([pp_fixed[:, 0], rnd_x])
            pp_y = np.concatenate([pp_fixed[:, 1], rnd_y])
            pp_z = np.concatenate([pp_fixed[:, 2], rnd_z])
        else:  # "extrema"
            # Fixed structural skeleton (same geometry every member) plus
            # n_pp random fill points drawn fresh for each member.
            rnd_x = rng.uniform(pp_bbox[0], pp_bbox[1], n_pp)
            rnd_y = rng.uniform(pp_bbox[2], pp_bbox[3], n_pp)
            rnd_z = rng.uniform(pp_bbox[4], pp_bbox[5], n_pp)
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

    return mod_list


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
    out: bool = True,
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
    out : bool
        If True, print status messages.

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
            out=out,
        )
        obs_list.append(file)

    if out:
        print("list of perturbed observation files:")
        print(obs_list)

    return obs_list

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
    R: np.ndarray | scipy.sparse.spmatrix,
    lam: float = 0.0,
) -> LinearOperator:
    """Create a LinearOperator representing Q = R.T @ R + lam * I.

    This is the **matrix-free** representation used by iterative solvers.

    Parameters
    ----------
    R : array_like or scipy.sparse.spmatrix, shape (m, n)
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
    R: np.ndarray | scipy.sparse.spmatrix,
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
        Q = Q + float(lam) * scipy.sparse.identity(Q.shape[0], format="csc")

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
    R: np.ndarray | scipy.sparse.spmatrix,
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
            Q = Q + float(lam) * scipy.sparse.identity(Q.shape[0], format="csc")

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
    R: np.ndarray | scipy.sparse.spmatrix,
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
    R: np.ndarray | scipy.sparse.spmatrix,
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


def make_precision_solver(
    R: np.ndarray | scipy.sparse.spmatrix,
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
        Iterative solver tolerances (SciPy style).  **Recommended rtol=1e-2**
        for Monte Carlo sampling — high accuracy is unnecessary and 1e-2
        typically halves CG iteration count versus the default 1e-3.
        atol=0 (default) lets rtol alone govern convergence.
    maxiter
        Maximum CG/BiCGStab iterations.  **Recommended 500–1000** to cap
        runaway solves on ill-conditioned problems.  With good λ and ILU
        preconditioning, convergence typically occurs in <200 iterations.
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
        **Recommended ``"ilu"``** — typically reduces CG iteration count 3–5×
        vs ``"jacobi"`` (which requires no explicit Q, but is weaker).
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

    # Resolve lam (optionally from diag(R.T R))
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
    R: np.ndarray | scipy.sparse.spmatrix,
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
        Extra args passed to :func:`make_precision_solver`.
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

        x = (R.T R + lam I)^{-1} R.T ξ,   ξ ~ N(0, I)

    Author: Volker Rath (DIAS)
    Created with the help of ChatGPT (GPT-5 Thinking) on 2026-01-02 (UTC)
    """
    rng = default_rng() if rng is None else rng
    m, n = R.shape

    if solver is None:
        kw = dict(solver_kwargs or {})
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
        samples[ix, :] = solver(b)

    return samples


def sample_rtr_low_rank(
    R: np.ndarray | scipy.sparse.spmatrix,
    n_samples: int = 1,
    n_eig: int = 32,
    sigma2_residual: float = 0.0,
    rng: Optional[Generator] = None,
    *,
    n_oversampling: int = 10,
    n_power_iter: int = 2,
) -> np.ndarray:
    """Approximate sampling from N(0, (R^T R)^{-1}) using randomized SVD of R.

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


def sample_prior_from_roughness(
    R: np.ndarray | scipy.sparse.spmatrix,
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
