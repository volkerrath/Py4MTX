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

All functions are importable; no code is executed on import.

Author: Volker Rath (DIAS)
Created with the help of ChatGPT (GPT-5 Thinking) on 2026-01-02 (UTC)
Updated 2026-03-31 by Claude (Anthropic): removed debug print in
sample_rtr_full_rank; removed dead commented-out code in
generate_model_ensemble; removed redundant per-sample print.
Updated 2026-03-31 by Claude (Anthropic): consolidated _diag_rtr
into _rtr_diag (single helper); removed dead estimate_low_rank_eigpairs;
enriched docstrings with tuning recommendations.
Updated 2026-04-02 by Claude (Anthropic): fixed FileNotFoundError in
generate_model_ensemble — template argument to fem.insert_model now uses
the full per-member path (_orig.dat backup) instead of the bare basename.
Updated 2026-04-02 by Claude (Anthropic): fixed generate_model_ensemble
write-back loop — now reads reference log10-resistivity from the backup
template and adds the perturbation before calling insert_model (method='add'),
so perturbed models are reference + delta_log10 rather than bare perturbations.
Updated 2026-04-11 by Claude Sonnet 4.6 (Anthropic): moved check_sparse_matrix
here from femtic.py (consolidation of all matrix/roughness tools into ensembles);
femtic.py Section 2 now imports these functions from ensembles rather than
duplicating them.
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
    fromto: Optional[Tuple[int, int]] = None,
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
    fromto : (int, int), optional
        Explicit start/stop indices (Python-style: start, stop) for ensemble
        member numbering. If None, indices range from 0..n_samples-1.
    out : bool, optional
        If True, print status messages.

    Returns
    -------
    dir_list : list of str
        List of created ensemble directory paths.
    """
    if fromto is None:
        from_to = np.arange(n_samples)
    else:
        from_to = np.arange(fromto[0], fromto[1])

    dir_list: list[str] = []
    for iens in from_to:
        directory = f"{dir_base}{iens}/"
        os.makedirs(directory, exist_ok=True)
        put_files(copy_list, link_list, directory, templates)
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
    """
    for fname in copies:
        src = templates + fname
        shutil.copy2(src, directory)

    for fname in links:
        src = os.path.abspath(templates + fname)
        dst = os.path.join(directory, fname)
        if os.path.islink(dst):
            os.remove(dst)
        os.symlink(src, dst)


def generate_model_ensemble(
    alg: str = 'rto',
    dir_base: str = "./ens_",
    n_samples: int = 1,
    fromto: Optional[Tuple[int, int]] = None,
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
    fromto : (int, int), optional
        Ensemble index range ``[start, stop)``. If None, use ``0..n_samples-1``.
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
        raise ValueError("generate_model_ensemble: roughness matrix R (q) must be provided.")

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

    if fromto is None:
        fromto_arr = np.arange(n_samples)
    else:
        fromto_arr = np.arange(fromto[0], fromto[1])

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

def generate_data_ensemble(alg: str = 'rto',
    dir_base: str = "./ens_",
    n_samples: int = 1,
    fromto: Optional[Tuple[int, int]] = None,
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
    fromto : (int, int), optional
        Explicit start/end indices (Python-style).
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
    if fromto is None:
        fromto_arr = np.arange(n_samples)
    else:
        fromto_arr = np.arange(fromto[0], fromto[1])

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
