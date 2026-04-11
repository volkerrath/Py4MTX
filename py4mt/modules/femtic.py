#!/usr/bin/env python3
"""femtic.py

FEMTIC-specific I/O utilities: reading and writing FEMTIC data files, model
resistivity blocks, mesh files, and conversion to/from NPZ / VTK / NetCDF.

This module covers:

- **Data I/O** — read and modify ``observe.dat`` (MT, VTF, phase tensor),
  including Gaussian perturbation for RTO ensembles (``modify_data``,
  ``modify_data_fcn``).
- **Distortion I/O** — ``read_distortion_file`` decomposes distortion files
  into C and C′ matrices.
- **Resistivity-block model workflow** — 3-step read → NPZ → modify → write
  pipeline (``read_model``, ``insert_model``, ``read_model_to_npz``,
  ``modify_model_npz``, ``write_model_from_npz``).
- **Mesh I/O** — parse FEMTIC ``mesh.dat`` tetrahedral meshes.
- **NPZ ↔ VTK / VTU** — convert NPZ model files for ParaView / PyVista.
- **NPZ ↔ NetCDF / HDF5** — CF-compliant and HDF5 export/import.
- **CLI interface** — subcommand-style command-line usage for batch conversion.

Roughness / prior-covariance / matrix tools (``get_roughness``,
``make_prior_cov``, ``matrix_reduce``, ``check_sparse_matrix``, etc.) are
canonical in **ensembles.py** and re-exported from here for backward
compatibility (Section 2).  Ensemble generation (``generate_directories``,
``generate_model_ensemble``, sampling helpers) lives exclusively in
``ensembles.py``.

Command-line interface
----------------------
    python femtic.py femtic-to-npz \\
        --mesh mesh.dat \\
        --rho-block resistivity_block_iter0.dat \\
        --out-npz femtic_model.npz

    python femtic.py npz-to-vtk \\
        --npz femtic_model.npz \\
        --out-vtu model.vtu \\
        --out-legacy model.vtk

    python femtic.py npz-to-femtic \\
        --npz femtic_model.npz \\
        --mesh-out mesh_reconstructed.dat \\
        --rho-block-out resistivity_block_iter0_reconstructed.dat

All functionality is also available as regular Python functions.

Author: Volker Rath (DIAS)
Created with the help of ChatGPT (GPT-5 Thinking) on 2026-01-02 (UTC)
Modified: 2026-04-11 by Claude Sonnet 4.6 (Anthropic) — Section 2
    (matrix/roughness tools) now imported from ensembles.py rather than
    duplicated; femtic.py restricted to FEMTIC-specific I/O and conversion.
"""
from __future__ import annotations

import os
import sys
import time
import json
import datetime
from pathlib import Path
from typing import (
    Callable,
    Optional,
    Sequence,
    Tuple,
    Dict,
    Literal,
    Any,
    TYPE_CHECKING,
)
from numpy.typing import ArrayLike

import numpy as np
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

from numpy.random import Generator, default_rng
from scipy.sparse import isspmatrix, issparse
from scipy.sparse.linalg import LinearOperator, cg, eigsh, bicgstab, spilu

# Optional but kept for compatibility with earlier versions
import joblib  # noqa: F401

def read_distortion_file(path=None):
    """

    Returns site IDs and 2×2 matrices for each site.

    Parameters
    ----------
    path : string
        FEMTIV distortion file

    Returns
    -------
    nsites : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.

    """
    """
    Return site IDs and 2×2 matrices for each site.
    """
    lines = Path(path).read_text().strip().splitlines()
    # nsites = int(lines[0].strip())
    data = []
    sites = []
    for line in lines[1:]:
        parts = line.split()
        sites.append(int(parts[0]))
        c00, c01, c10, c11 = map(float, parts[1:5])
        M = np.array([[c00, c01], [c10, c11]], dtype=float)
        data.append(M)
    data = np.asarray(data)

    c_dash=[]
    c = []
    for s in np.arange(np.shape(data)[0]):
        c_dash.append(data[s,:])
        c.append(data[s,:] + np.identity(2))

    c = np.asarray(c)
    c_dash = np.asarray(c_dash)
    # print(np.shape(c))
    # print(type(c))
    return c, c_dash

def get_nrms(directory=None):
    '''
    Get best (smallest) nRMS from FEMTIC run.

    Parameters
    ----------
    directory : string, optional
        Directory containing FEMTIC convergence file. The default is None.

    Returns
    -------
    num_best : int
        DESCRIPTION.
    nrm_best : float
        DESCRIPTION.

    '''
    if directory is None:
        sys.exit('get_nrms: No directory given! Exit.')
    convergence = []
    fline = -1
    with open(directory+'/femtic.cnv') as cnv:
        content = cnv.readlines()

        for line in content:

            if '#' in line:
                continue
            fline = fline + 1
            #print (line)
            nline = line.split()
            if len(nline)==0:
                continue
            print(nline)
            itern = int(nline[0])
            retry = int(nline[1])
            alpha = float(nline[2])
            rough = float(nline[5])
            misft = float(nline[7])
            nrmse = float(nline[8])

            convergence.append([itern, retry, alpha, rough, misft, nrmse])

    if len(convergence)==0:
        print (directory, '/femtic.cnv', ' is empty!')
        num_best = -1
        nrm_best = 1e32
    else:
        c = np.array(convergence)
        index_min = np.argmin(c[:,5])
        nrm_best = c[index_min,5]
        num_best = int(round(c[index_min,0]))

    return num_best, nrm_best

def _parse_region_line(line: str) -> tuple[int, float, float, float, float, int]:
    """Parse one region line from a FEMTIC resistivity block.

    Expected format (whitespace-separated columns):

        ireg  rho  rho_lower  rho_upper  n  flag

    Returns
    -------
    (ireg, rho, rho_lower, rho_upper, n, flag)
    """
    parts = line.split()
    if len(parts) < 6:
        raise ValueError(f"Invalid region line (need ≥6 columns): {line!r}")
    ireg = int(parts[0])
    rho = float(parts[1])
    rho_lower = float(parts[2])
    rho_upper = float(parts[3])
    n = float(parts[4])
    flag = int(parts[5])
    return ireg, rho, rho_lower, rho_upper, n, flag


def _format_region_line(
    ireg: int,
    rho: float,
    rho_lower: float,
    rho_upper: float,
    n: float,
    flag: int,
) -> str:
    """Format a region line in FEMTIC resistivity block style."""
    return (
        f" {ireg:9d}        {rho:.6e}   {rho_lower:.6e}   {rho_upper:.6e}   "
        f"{n:.6e} {flag:9d}"
    )


def _infer_ocean_present(region1_line: str) -> bool:
    """Infer whether region 1 is an 'ocean' fixed block.

    Heuristic (conservative):
    - flag == 1 (fixed)
    - rho <= 1 Ωm (very conductive, typical ocean ~0.25 Ωm)

    This can be overridden by passing ``ocean=True`` or ``ocean=False`` to
    :func:`read_model` / :func:`insert_model`.
    """
    _, rho, _, _, _, flag = _parse_region_line(region1_line)
    return (flag == 1) and (rho <= 1.0)


def read_model(
    model_file: str | Path,
    model_trans: str = "log10",
    out: bool = True,
    *,
    ocean: bool | None = None,
    include_fixed: bool = False,
) -> np.ndarray:
    """Read a FEMTIC ``resistivity_block_iterX.dat`` and return a model vector.

    FEMTIC resistivity blocks define ``nreg`` regions. Each region has a resistivity
    value (ρ) and additional metadata columns. Regions may be marked *fixed* via a
    ``flag`` column (typically ``flag == 1``).

    Common conventions are:

    - region 0: **air** (always treated as fixed by this helper)
    - region 1: **ocean** (often fixed, but **may be absent** in some setups)

    In addition, **other fixed regions** may exist (e.g., prescribed background blocks).
    Therefore, this function no longer assumes that only air (and maybe ocean) are fixed.

    Default behaviour (``include_fixed=False``):

    - always excludes region 0 (air),
    - excludes any region with ``flag == 1``,
    - additionally excludes region 1 if it is treated as ocean (auto-inferred unless
      overridden via ``ocean=...``).

    Parameters
    ----------
    model_file
        Path to FEMTIC resistivity block file.
    model_trans
        Either ``"log10"`` (default) to return ``log10(rho)``, or ``"none"`` to
        return ``rho`` in Ωm.
    out
        If True, print a short info line.
    ocean
        If ``None`` (default), attempt to infer whether region 1 is ocean.
        If ``True`` / ``False``, force ocean-present / ocean-absent handling.
        Note that fixed-ness is still governed by the region flag: a fixed region
        (``flag == 1``) stays fixed even if ``ocean=False``.
    include_fixed
        If True, include *all* regions (including air and any fixed regions).

    Returns
    -------
    np.ndarray
        1-D vector of model parameters in region-index order (selected regions only).

    Notes
    -----
    Ocean inference is intentionally conservative (see :func:`_infer_ocean_present`).
    If the heuristic mis-classifies region 1, pass ``ocean=True`` or ``ocean=False``.
    """
    model_path = Path(model_file)

    with model_path.open("r", encoding="utf-8", errors="replace") as f:
        header = f.readline()
        hdr_parts = header.split()
        if len(hdr_parts) < 2:
            raise ValueError(f"Invalid resistivity block header: {hdr_parts!r}")
        nelem = int(hdr_parts[0])
        nreg = int(hdr_parts[1])

        # Skip element->region mapping
        for _ in range(nelem):
            f.readline()

        if nreg <= 0:
            raise ValueError("No regions in resistivity block (nreg<=0).")

        region_lines: list[str] = []
        region_rho = np.zeros(nreg, dtype=float)
        region_flag = np.zeros(nreg, dtype=int)

        for i in range(nreg):
            line = f.readline()
            if not line:
                raise ValueError(
                    f"Unexpected EOF while reading region lines: expected {nreg}, got {i}."
                )
            ireg, rho, _, _, _, flag = _parse_region_line(line)
            if ireg != i:
                raise ValueError(f"Expected region index {i} at line {i}, got {ireg}.")
            region_lines.append(line)
            region_rho[i] = rho
            region_flag[i] = flag

    # Decide whether region 1 is treated as ocean (optional)
    ocean_present = False
    if nreg > 1:
        if ocean is None:
            ocean_present = _infer_ocean_present(region_lines[1])
        else:
            ocean_present = bool(ocean)

    # Determine fixed regions: air + flagged fixed + optional ocean
    fixed_mask = np.zeros(nreg, dtype=bool)
    fixed_mask[0] = True  # air always fixed here
    fixed_mask |= (region_flag == 1)
    if nreg > 1 and ocean_present:
        fixed_mask[1] = True

    if include_fixed:
        sel = np.arange(nreg, dtype=int)
    else:
        sel = np.where(~fixed_mask)[0]

    rho_sel = region_rho[sel].astype(float, copy=False)

    if model_trans.lower() == "log10":
        out_vec = np.log10(rho_sel)
    elif model_trans.lower() in ("none", "rho"):
        out_vec = rho_sel
    else:
        raise ValueError(f"Unknown model_trans={model_trans!r}; use 'log10' or 'none'.")

    if out:
        n_fixed = int(fixed_mask.sum())
        print(
            f"read_model: file={model_path.name}, nelem={nelem}, nreg={nreg}, "
            f"ocean_present={ocean_present}, fixed={n_fixed}, returned={out_vec.size}."
        )

    return out_vec

def insert_model(
    template: str = "resistivity_block_iter0.dat",
    model: np.ndarray | Sequence[float] | None = None,
    model_file: Optional[str] = None,
    model_name: str = "",
    out: bool = True,
    *,
    ocean: bool | None = None,
    air_rho: float = 1.0e9,
    ocean_rho: float = 2.5e-1,
) -> None:
    """Insert a (log10) resistivity vector into a FEMTIC resistivity block file.

    The input ``model`` is interpreted as **log10(ρ)** values for *free* regions.

    Fixed-region handling (updated):

    - region 0 (air) is always treated as fixed by this helper and written as ``air_rho``.
    - any region with ``flag == 1`` in the template is treated as fixed and preserved.
    - region 1 is additionally treated as fixed and written as ``ocean_rho`` if it is
      treated as ocean (auto-inferred unless overridden via ``ocean=...``).

    This update supports **additional fixed blocks** beyond air/ocean.

    The template's metadata columns (lower/upper bounds, ``n``, and flag) are preserved;
    only the resistivity value is replaced for free regions (and optionally air/ocean).

    Parameters
    ----------
    template
        Template block file to read header, mapping and region metadata from.
    model
        1-D array-like of **log10(ρ)** values for *free* regions (i.e., regions that
        are not fixed by the rules above).
    model_file
        Output file name. If None, derived from ``template`` and ``model_name``.
    model_name
        Optional suffix for output file naming.
    out
        If True, print small info.
    ocean
        If ``None`` (default), infer whether region 1 is ocean (see
        :func:`_infer_ocean_present`). If ``True`` / ``False``, force ocean-present /
        ocean-absent handling.
    air_rho
        Resistivity enforced for region 0 (air).
    ocean_rho
        Resistivity enforced for region 1 if treated as ocean.

    Returns
    -------
    None
        Writes the modified resistivity block to ``model_file``.
    """
    if model is None:
        raise ValueError("insert_model: 'model' must be provided (free-region vector).")

    model_arr = np.asarray(model, dtype=float).ravel()

    template_path = Path(template)
    if model_file is None:
        stem = template_path.name
        if model_name:
            model_file = f"{stem}.{model_name}"
        else:
            model_file = f"{stem}.new"
    out_path = Path(model_file)

    with template_path.open("r", encoding="utf-8", errors="replace") as fin, out_path.open(
        "w", encoding="utf-8"
    ) as fout:
        header = fin.readline()
        hdr_parts = header.split()
        if len(hdr_parts) < 2:
            raise ValueError(f"Invalid resistivity block header: {hdr_parts!r}")
        nelem = int(hdr_parts[0])
        nreg = int(hdr_parts[1])

        # Write header and element->region mapping unchanged
        fout.write(header)
        for _ in range(nelem):
            fout.write(fin.readline())

        if nreg <= 0:
            raise ValueError("No regions in resistivity block (nreg<=0).")

        # Read all region lines first (we need flags to identify fixed blocks)
        reg_meta: list[tuple[int, float, float, float, float, int]] = []
        reg_lines: list[str] = []
        for i in range(nreg):
            line = fin.readline()
            if not line:
                raise ValueError(
                    f"Unexpected EOF while reading region lines: expected {nreg}, got {i}."
                )
            ireg, rho, lo, hi, nn, flag = _parse_region_line(line)
            if ireg != i:
                raise ValueError(f"Expected region index {i} at line {i}, got {ireg}.")
            reg_lines.append(line)
            reg_meta.append((ireg, rho, lo, hi, nn, flag))

        # Determine whether region 1 is treated as ocean (optional)
        ocean_present = False
        if nreg > 1:
            if ocean is None:
                ocean_present = _infer_ocean_present(reg_lines[1])
            else:
                ocean_present = bool(ocean)

        region_flag = np.array([m[5] for m in reg_meta], dtype=int)

        fixed_mask = np.zeros(nreg, dtype=bool)
        fixed_mask[0] = True  # air always fixed here
        fixed_mask |= (region_flag == 1)
        if nreg > 1 and ocean_present:
            fixed_mask[1] = True

        free_idx = np.where(~fixed_mask)[0].tolist()
        n_free = len(free_idx)
        n_fixed = int(fixed_mask.sum())

        if model_arr.size != n_free:
            raise ValueError(
                "insert_model: size mismatch for free-region model."
                f"  template: nreg={nreg}, fixed={n_fixed}, free={n_free}"
                f"  provided model size={model_arr.size}"
                "Hint: your template may contain additional fixed regions (flag==1). "
                "Use read_model(...) on the same template to get a consistent free vector."
            )

        # Write regions, preserving metadata; replace rho for free regions (and air/ocean)
        model_ptr = 0
        for ireg, rho0, lo, hi, nn, flag in reg_meta:
            if ireg == 0:
                rho = float(air_rho)
            elif ireg == 1 and ocean_present:
                rho = float(ocean_rho)
            elif fixed_mask[ireg]:
                rho = float(rho0)  # preserve fixed-region rho from template
            else:
                rho = float(10.0 ** model_arr[model_ptr])
                model_ptr += 1

            fout.write(_format_region_line(ireg, rho, lo, hi, nn, flag) + "\n")

    if out:
        print(f"File {out_path} successfully written.")
        print(
            f"insert_model: nreg={nreg}, ocean_present={ocean_present}, "
            f"fixed={n_fixed}, free={n_free}."
        )


# ============================================================================
# SECTION 1B: FEMTIC resistivity-block model workflow (read → NPZ → modify → write)
# ============================================================================


def read_model_to_npz(
    model_file: str | Path,
    npz_file: str | Path,
    *,
    model_trans: str = "log10",
    ocean: bool | None = None,
    air_rho: float = 1.0e9,
    ocean_rho: float = 2.5e-1,
    out: bool = True,
) -> Path:
    """Read a FEMTIC resistivity block file and write a compact NPZ representation.

    This is step (1) of the 3-step model workflow:

        (1) read FEMTIC model → NPZ
        (2) modify model in NPZ (free regions only)
        (3) write FEMTIC model from NPZ (+ updated NPZ)

    The produced NPZ is designed to be **compatible with precision-matrix sampling**
    conventions:

    - It stores the full region table (ρ, bounds, n, flag) and the element→region
      mapping, so the file can be reconstructed exactly.
    - It stores ``fixed_mask`` and ``free_idx`` so downstream modification and
      sampling can operate strictly on the *free* parameters.
    - It stores ``model_free`` in the chosen transform space (default: log10(ρ))
      for direct use as a parameter vector in sampling / inversion.

    Parameters
    ----------
    model_file
        Path to ``resistivity_block_iterX.dat``.
    npz_file
        Output NPZ path (written with ``np.savez_compressed``).
    model_trans
        Parameterisation for the stored free vector:

        - ``"log10"`` (default): store ``log10(ρ)`` for free regions.
        - ``"none"`` / ``"rho"``: store ρ in Ωm for free regions.
    ocean
        If None (default), infer whether region 1 is an ocean block.
        If True / False, force the decision.
    air_rho, ocean_rho
        Conventional resistivities used when enforcing fixed air/ocean on write.
        Stored in metadata so later steps are consistent.
    out
        If True, print a one-line summary.

    Returns
    -------
    Path
        Path to the written NPZ file.

    Author: Volker Rath (DIAS)
    Created with the help of ChatGPT (GPT-5 Thinking) on 2026-01-02 (UTC)
    """
    struct = _read_resistivity_block_struct(
        model_file=model_file,
        model_trans=model_trans,
        ocean=ocean,
        air_rho=air_rho,
        ocean_rho=ocean_rho,
        out=out,
    )
    _save_resistivity_block_npz(struct, npz_file=npz_file, out=out)
    return Path(npz_file)


def modify_model_npz(
    npz_in: str | Path,
    npz_out: str | Path,
    *,
    method: str = "precision_sample",
    # --- general modifiers
    set_free: np.ndarray | Sequence[float] | None = None,
    add_sigma: float | np.ndarray | Sequence[float] | None = None,
    rng: Generator | None = None,
    seed: int | None = None,
    # --- precision sampling options
    roughness: str | Path | None = None,
    R: np.ndarray | scipy.sparse.spmatrix | None = None,
    n_samples: int = 1,
    add_to_current: bool = True,
    lam: float = 1.0e-5,
    lam_mode: str = "fixed",
    lam_alpha: float | None = None,
    lam_statistic: str = "median",
    lam_min: float = 0.0,
    solver_method: str = "cg",
    solver_kwargs: dict | None = None,
    precond: str | None = None,
    precond_kwargs: dict | None = None,
    scale: float = 1.0,
    enforce_air_ocean: bool = True,
    out: bool = True,
) -> Path:
    """Modify a resistivity-block NPZ (free regions only) and write an updated NPZ.

    This is step (2) of the 3-step model workflow.

    The modification always respects fixed regions: **only entries in ``free_idx``**
    may change. Fixed regions (air/ocean and any ``flag==1`` blocks) are preserved.

    Supported methods
    -----------------
    ``method`` (case-insensitive) selects the update rule:

    - ``"set_free"``:
        Replace the free-parameter vector with ``set_free`` (in the NPZ' transform
        space, i.e. log10(ρ) by default).

    - ``"add_noise"``:
        Add i.i.d. Gaussian noise to the free vector:
            ``m_free_new = m_free + N(0, add_sigma)``
        where ``add_sigma`` can be a scalar or length ``n_free``.

    - ``"precision_sample"`` (default):
        Draw a perturbation from the Gaussian prior induced by a roughness matrix:

            δ ~ N(0, (R.T R + λ I)^{-1})

        using the precision-matrix sampler from ``ensembles.py`` (if available).
        The diagonal shift λ is chosen according to ``lam_mode`` and friends
        (notably ``lam_mode='scaled_median_diag'``).

        The update is then either:

        - additive: ``m_free_new = m_free + scale * δ`` (default), or
        - replacement: ``m_free_new = scale * δ`` if ``add_to_current=False``.

    Roughness / precision compatibility
    -----------------------------------
    The roughness matrix can be provided in two ways:

    - ``roughness=...``: path to FEMTIC ``roughening_matrix.out`` (parsed with
      :func:`get_roughness`).
    - ``R=...``: matrix already in memory.

    The sampler expects the number of columns of R to match the number of model
    parameters being sampled. This function supports both common situations:

    - ``R.shape[1] == n_free`` (already restricted): used as-is.
    - ``R.shape[1] == nreg`` (includes fixed): internally sliced to free columns.

    Parameters
    ----------
    npz_in, npz_out
        Input and output NPZ files.
    method
        One of ``set_free``, ``add_noise``, ``precision_sample``.
    set_free
        Free-parameter vector for ``method='set_free'``.
    add_sigma
        Noise level for ``method='add_noise'`` (scalar or length n_free).
    rng
        Random number generator. If None, uses ``default_rng(seed)`` (or an
        unseeded generator when ``seed`` is also None).
    seed
        Convenience integer seed.  Ignored when ``rng`` is provided.  Pass an
        integer for reproducible draws.
    roughness, R
        Roughness matrix source for ``method='precision_sample'``.
    n_samples
        Number of samples drawn (only the first is applied; remaining are ignored).
        This is mainly for convenience / debugging.
    add_to_current
        If True (default), add the draw to the current free model. If False, replace.
    lam, lam_mode, lam_alpha, lam_statistic, lam_min
        Diagonal shift controls passed to the precision sampler.
    solver_method, solver_kwargs
        Precision solver controls passed through to the sampler.
    precond, precond_kwargs
        Optional convenience arguments for iterative preconditioning. If provided,
        they are injected into ``solver_kwargs`` as ``{'precond': precond, 'precond_kwargs': precond_kwargs}``
        unless those keys are already present.
    scale
        Multiplier applied to the draw before applying it to the free model.
    enforce_air_ocean
        If True, enforce ``air_rho`` / ``ocean_rho`` for region 0/1 (if ocean-present),
        regardless of their values in the NPZ.
    out
        If True, print status.

    Returns
    -------
    Path
        Path to the updated NPZ.

    Raises
    ------
    ValueError
        If inputs are inconsistent (missing R for precision sampling, size mismatch, etc).

    Author: Volker Rath (DIAS)
    Created with the help of ChatGPT (GPT-5 Thinking) on 2026-01-02 (UTC)
    """
    rng = default_rng(seed) if rng is None else rng

    struct = _load_resistivity_block_npz(npz_in)
    meta = struct["meta"]
    free_idx = struct["free_idx"]
    fixed_mask = struct["fixed_mask"]
    model_trans = meta.get("model_trans", "log10")

    m_free = struct["model_free"].astype(float, copy=True)

    meth = str(method).strip().lower()
    if meth in {"set", "set_free", "replace"}:
        if set_free is None:
            raise ValueError("modify_model_npz(method='set_free'): 'set_free' must be provided.")
        m_new = np.asarray(set_free, dtype=float).ravel()
        if m_new.size != m_free.size:
            raise ValueError(
                f"modify_model_npz(set_free): size mismatch: got {m_new.size}, expected {m_free.size}."
            )
        m_free_new = m_new

    elif meth in {"add_noise", "noise", "gauss", "gaussian"}:
        if add_sigma is None:
            raise ValueError("modify_model_npz(method='add_noise'): 'add_sigma' must be provided.")
        sig = np.asarray(add_sigma, dtype=float)
        if sig.ndim == 0:
            eps = rng.standard_normal(size=m_free.size) * float(sig)
        else:
            sig = sig.ravel()
            if sig.size != m_free.size:
                raise ValueError(
                    f"modify_model_npz(add_noise): add_sigma must be scalar or length {m_free.size}, got {sig.size}."
                )
            eps = rng.standard_normal(size=m_free.size) * sig
        m_free_new = m_free + eps

    elif meth in {"precision_sample", "precision", "rtr", "prior"}:
        # Acquire roughness matrix
        if R is None:
            if roughness is None:
                raise ValueError(
                    "modify_model_npz(method='precision_sample') requires 'R' or 'roughness=...'."
                )
            R_use = get_roughness(str(roughness), spformat="csc", out=out)
        else:
            R_use = R

        # Restrict to free parameters if required
        nreg = int(struct["region_rho"].size)
        nfree = int(free_idx.size)
        if R_use.shape[1] == nreg:
            R_use = R_use[:, free_idx]
        elif R_use.shape[1] != nfree:
            raise ValueError(
                "modify_model_npz(precision_sample): R has incompatible number of columns. "
                f"Got {R_use.shape[1]}, expected {nfree} (free) or {nreg} (all regions)."
            )

        # Import sampling tools from ensembles.py (preferred implementation)
        sampler = _import_precision_sampler()

        kw = dict(solver_kwargs or {})
        if precond is not None and ("precond" not in kw) and ("mprec" not in kw):
            kw["precond"] = str(precond)
        if precond_kwargs is not None and ("precond_kwargs" not in kw):
            kw["precond_kwargs"] = dict(precond_kwargs)

        draw = sampler(
            R_use,
            n_samples=int(n_samples),
            lam=float(lam),
            solver_method=str(solver_method),
            solver_kwargs=kw,
            lam_mode=str(lam_mode),
            lam_alpha=lam_alpha,
            lam_statistic=str(lam_statistic),
            lam_min=float(lam_min),
            rng=rng,
        )
        delta = np.asarray(draw[0], dtype=float).ravel()
        if delta.size != m_free.size:
            raise RuntimeError(
                f"Precision sampler returned wrong size: got {delta.size}, expected {m_free.size}."
            )

        if add_to_current:
            m_free_new = m_free + float(scale) * delta
        else:
            m_free_new = float(scale) * delta

    else:
        raise ValueError(f"modify_model_npz: unsupported method={method!r}.")

    # Apply update to region resistivities while preserving fixed regions
    region_rho = struct["region_rho"].astype(float, copy=True)

    if model_trans.lower() == "log10":
        rho_free = 10.0 ** m_free_new
    elif model_trans.lower() in {"none", "rho"}:
        rho_free = m_free_new
    else:
        raise ValueError(f"modify_model_npz: unknown model_trans={model_trans!r} in NPZ meta.")

    region_rho[free_idx] = rho_free

    if enforce_air_ocean:
        air_rho = float(meta.get("air_rho", 1.0e9))
        ocean_rho = float(meta.get("ocean_rho", 2.5e-1))
        ocean_present = bool(meta.get("ocean_present", False))
        region_rho[0] = air_rho
        if ocean_present and region_rho.size > 1:
            region_rho[1] = ocean_rho

    # Store back into struct
    struct["region_rho"] = region_rho
    struct["model_free"] = np.asarray(m_free_new, dtype=float)
    struct["fixed_mask"] = np.asarray(fixed_mask, dtype=bool)

    # History / provenance
    hist = list(meta.get("history", []))
    hist.append(
        {
            "timestamp_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "action": "modify_model_npz",
            "method": str(method),
            "lam_mode": str(lam_mode),
            "lam": float(lam),
            "scale": float(scale),
            "add_to_current": bool(add_to_current),
        }
    )
    meta["history"] = hist
    struct["meta"] = meta

    _save_resistivity_block_npz(struct, npz_file=npz_out, out=out)
    return Path(npz_out)


def write_model_from_npz(
    npz_file: str | Path,
    model_file: str | Path,
    *,
    also_write_npz: str | Path | None = None,
    out: bool = True,
) -> Path:
    """Write a FEMTIC resistivity block from an NPZ representation.

    This is step (3) of the 3-step model workflow.

    The output file is a standard ``resistivity_block_iterX.dat``-style block:

    - header line: ``nelem nreg``
    - element→region mapping (nelem lines)
    - region table (nreg lines with ρ and metadata)

    Parameters
    ----------
    npz_file
        NPZ produced by :func:`read_model_to_npz` and optionally updated by
        :func:`modify_model_npz`.
    model_file
        Output FEMTIC resistivity block file.
    also_write_npz
        If provided, write an additional NPZ copy (useful for provenance when
        you want the NPZ to sit next to the written FEMTIC file). If None, no
        extra NPZ is written.
    out
        If True, print status.

    Returns
    -------
    Path
        Path to the written FEMTIC model file.

    Author: Volker Rath (DIAS)
    Created with the help of ChatGPT (GPT-5 Thinking) on 2026-01-02 (UTC)
    """
    struct = _load_resistivity_block_npz(npz_file)
    _write_resistivity_block_struct(struct, model_file=model_file, out=out)

    if also_write_npz is not None:
        _save_resistivity_block_npz(struct, npz_file=also_write_npz, out=out)

    return Path(model_file)


def _read_resistivity_block_struct(
    model_file: str | Path,
    *,
    model_trans: str = "log10",
    ocean: bool | None = None,
    air_rho: float = 1.0e9,
    ocean_rho: float = 2.5e-1,
    out: bool = True,
) -> dict:
    """Low-level reader producing a complete in-memory structure for a resistivity block."""
    model_path = Path(model_file)

    with model_path.open("r", encoding="utf-8", errors="replace") as f:
        header = f.readline().split()
        if len(header) < 2:
            raise ValueError(f"Invalid resistivity block header: {header!r}")
        nelem = int(header[0])
        nreg = int(header[1])

        elem_region = np.empty(nelem, dtype=np.int32)
        for i in range(nelem):
            parts = f.readline().split()
            if len(parts) < 2:
                raise ValueError(f"Invalid element-region line at element {i}: {parts!r}")
            ie = int(parts[0])
            ir = int(parts[1])
            if ie != i:
                # FEMTIC typically uses 0..nelem-1, but don't hard-fail if different.
                # We still store in file order.
                pass
            elem_region[i] = ir

        region_rho = np.empty(nreg, dtype=np.float64)
        region_lo = np.empty(nreg, dtype=np.float64)
        region_hi = np.empty(nreg, dtype=np.float64)
        region_n = np.empty(nreg, dtype=np.float64)
        region_flag = np.empty(nreg, dtype=np.int32)

        region_lines: list[str] = []
        for i in range(nreg):
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading region lines at i={i}.")
            ireg, rho, lo, hi, nn, flag = _parse_region_line(line)
            if ireg != i:
                raise ValueError(f"Expected region index {i} but got {ireg}.")
            region_lines.append(line)
            region_rho[i] = rho
            region_lo[i] = lo
            region_hi[i] = hi
            region_n[i] = nn
            region_flag[i] = flag

    # Ocean inference (optional)
    ocean_present = False
    if nreg > 1:
        if ocean is None:
            ocean_present = _infer_ocean_present(region_lines[1])
        else:
            ocean_present = bool(ocean)

    fixed_mask = np.zeros(nreg, dtype=bool)
    fixed_mask[0] = True
    fixed_mask |= (region_flag == 1)
    if nreg > 1 and ocean_present:
        fixed_mask[1] = True

    free_idx = np.where(~fixed_mask)[0].astype(np.int32)

    if model_trans.lower() == "log10":
        model_free = np.log10(region_rho[free_idx])
    elif model_trans.lower() in {"none", "rho"}:
        model_free = region_rho[free_idx].copy()
    else:
        raise ValueError(f"Unknown model_trans={model_trans!r}; use 'log10' or 'none'.")

    meta = {
        "source_model_file": str(model_path),
        "model_trans": str(model_trans),
        "ocean_present": bool(ocean_present),
        "air_rho": float(air_rho),
        "ocean_rho": float(ocean_rho),
        "created_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "history": [
            {
                "timestamp_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "action": "read_model_to_npz",
                "model_file": str(model_path),
            }
        ],
    }

    if out:
        print(
            f"_read_resistivity_block_struct: file={model_path.name}, nelem={nelem}, nreg={nreg}, "
            f"ocean_present={ocean_present}, fixed={int(fixed_mask.sum())}, free={int(free_idx.size)}."
        )

    return {
        "nelem": int(nelem),
        "nreg": int(nreg),
        "elem_region": elem_region,
        "region_rho": region_rho,
        "region_lo": region_lo,
        "region_hi": region_hi,
        "region_n": region_n,
        "region_flag": region_flag,
        "fixed_mask": fixed_mask,
        "free_idx": free_idx,
        "model_free": np.asarray(model_free, dtype=np.float64),
        "meta": meta,
    }


def _write_resistivity_block_struct(struct: dict, model_file: str | Path, *, out: bool = True) -> None:
    """Low-level writer for the resistivity-block structure."""
    out_path = Path(model_file)

    nelem = int(struct["nelem"])
    nreg = int(struct["nreg"])
    elem_region = np.asarray(struct["elem_region"], dtype=np.int64).ravel()
    region_rho = np.asarray(struct["region_rho"], dtype=np.float64).ravel()
    region_lo = np.asarray(struct["region_lo"], dtype=np.float64).ravel()
    region_hi = np.asarray(struct["region_hi"], dtype=np.float64).ravel()
    region_n = np.asarray(struct["region_n"], dtype=np.float64).ravel()
    region_flag = np.asarray(struct["region_flag"], dtype=np.int64).ravel()

    if elem_region.size != nelem:
        raise ValueError(f"_write_resistivity_block_struct: elem_region size mismatch: {elem_region.size} vs {nelem}.")
    if region_rho.size != nreg:
        raise ValueError(f"_write_resistivity_block_struct: region arrays size mismatch: {region_rho.size} vs {nreg}.")

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"{nelem:10d} {nreg:10d}\n")
        for i in range(nelem):
            f.write(f"{i:10d} {int(elem_region[i]):10d}\n")
        for i in range(nreg):
            f.write(
                _format_region_line(
                    ireg=int(i),
                    rho=float(region_rho[i]),
                    rho_lower=float(region_lo[i]),
                    rho_upper=float(region_hi[i]),
                    n=float(region_n[i]),
                    flag=int(region_flag[i]),
                )
                + "\n"
            )

    if out:
        print(f"_write_resistivity_block_struct: wrote {out_path} (nelem={nelem}, nreg={nreg}).")


def _save_resistivity_block_npz(struct: dict, *, npz_file: str | Path, out: bool = True) -> None:
    """Save resistivity-block structure to NPZ with JSON metadata."""
    npz_path = Path(npz_file)

    meta_json = json.dumps(struct["meta"], sort_keys=True)
    np.savez_compressed(
        npz_path,
        nelem=np.asarray(struct["nelem"], dtype=np.int64),
        nreg=np.asarray(struct["nreg"], dtype=np.int64),
        elem_region=np.asarray(struct["elem_region"], dtype=np.int32),
        region_rho=np.asarray(struct["region_rho"], dtype=np.float64),
        region_lo=np.asarray(struct["region_lo"], dtype=np.float64),
        region_hi=np.asarray(struct["region_hi"], dtype=np.float64),
        region_n=np.asarray(struct["region_n"], dtype=np.float64),
        region_flag=np.asarray(struct["region_flag"], dtype=np.int32),
        fixed_mask=np.asarray(struct["fixed_mask"], dtype=np.bool_),
        free_idx=np.asarray(struct["free_idx"], dtype=np.int32),
        model_free=np.asarray(struct["model_free"], dtype=np.float64),
        meta_json=np.asarray(meta_json, dtype=object),
    )

    if out:
        print(f"_save_resistivity_block_npz: wrote {npz_path}.")


def _load_resistivity_block_npz(npz_file: str | Path) -> dict:
    """Load resistivity-block NPZ created by :func:`read_model_to_npz`."""
    p = Path(npz_file)
    with np.load(p, allow_pickle=True) as z:
        meta_json = z["meta_json"].item()
        meta = json.loads(meta_json) if isinstance(meta_json, str) else json.loads(str(meta_json))
        return {
            "nelem": int(np.asarray(z["nelem"]).ravel()[0]),
            "nreg": int(np.asarray(z["nreg"]).ravel()[0]),
            "elem_region": np.asarray(z["elem_region"], dtype=np.int32),
            "region_rho": np.asarray(z["region_rho"], dtype=np.float64),
            "region_lo": np.asarray(z["region_lo"], dtype=np.float64),
            "region_hi": np.asarray(z["region_hi"], dtype=np.float64),
            "region_n": np.asarray(z["region_n"], dtype=np.float64),
            "region_flag": np.asarray(z["region_flag"], dtype=np.int32),
            "fixed_mask": np.asarray(z["fixed_mask"], dtype=bool),
            "free_idx": np.asarray(z["free_idx"], dtype=np.int32),
            "model_free": np.asarray(z["model_free"], dtype=np.float64),
            "meta": meta,
        }


def _import_precision_sampler() -> Callable[..., np.ndarray]:
    """Import the preferred precision-matrix sampler from ``ensembles.py``.

    Returns a function with signature compatible with::

        sampler(R, n_samples=..., lam=..., solver_method=..., solver_kwargs=...,
                lam_mode=..., lam_alpha=..., lam_statistic=..., lam_min=..., rng=...)

    The current preferred implementation is ``ensembles.sample_rtr_full_rank``.
    """
    try:
        # Local import to avoid hard dependency during non-UQ use.
        import ensembles  # type: ignore

        if hasattr(ensembles, "sample_rtr_full_rank"):
            return ensembles.sample_rtr_full_rank  # type: ignore[attr-defined]
        raise ImportError("ensembles.sample_rtr_full_rank not found.")
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Precision sampler not available. Ensure ensembles.py is importable "
            "and provides sample_rtr_full_rank (cleaned version with lam_mode support)."
        ) from exc


def modify_data_fcn(
    template_file: str = "observe.dat",
    draw_from: Sequence[float | str] = ("normal", 0.0, 1.0),
    scalfac: float = 1.0,
    out: bool = True,
    *,
    rng: Generator | None = None,
    seed: int | None = None,
) -> None:
    """
    Simpler variant of :func:`modify_data` using existing error columns scaled
    by a factor `scalfac`.

    Parameters
    ----------
    template_file : str
        FEMTIC observe.dat path.
    draw_from : sequence
        Noise distribution spec.  The first element names the distribution;
        subsequent elements are parameters.  Supported distributions:

        - ``("normal", loc, scale)`` — Gaussian N(loc, scale²); ``loc`` and
          ``scale`` are *additive* offsets/factors applied on top of the
          per-datum error-scaled value.  In the typical case ``("normal", 0.0,
          1.0)`` the draw is simply ``Normal(val, err * scalfac)``.
        - ``("uniform", a, b)`` — Uniform[a, b] multiplied element-wise with
          the scaled error (centred draw: a=−1, b=1 gives the full ±err band).

        The ``loc`` / ``a`` / ``b`` parameters are provided for future
        extensibility; the usual setting is ``("normal", 0.0, 1.0)``.
    scalfac : float
        Scale factor applied to existing error columns before sampling.
        The effective standard deviation for datum *i* is
        ``err_i * scalfac``.
    out : bool
        If True, print status messages.
    rng : numpy.random.Generator, optional
        Random number generator.  Preferred over ``seed`` when the caller
        manages the generator directly.  If both ``rng`` and ``seed`` are
        None a fresh generator is created (non-reproducible).
    seed : int, optional
        Convenience seed.  Ignored when ``rng`` is provided.  Pass an integer
        for reproducible output.

    Notes
    -----
    This function preserves the existing relative error structure, only scales
    it and draws noise accordingly.  It is a simpler alternative to
    :func:`modify_data` that does not require re-parsing the file into an
    internal representation.
    """
    # ---- RNG setup -------------------------------------------------------
    if rng is None:
        rng = default_rng(seed)

    # ---- parse draw_from -------------------------------------------------
    draw_seq = list(draw_from)
    dist_name = str(draw_seq[0]).lower() if draw_seq else "normal"
    if dist_name not in ("normal", "uniform"):
        raise ValueError(
            f"modify_data_fcn: unsupported draw_from distribution {draw_seq[0]!r}. "
            "Use 'normal' or 'uniform'."
        )

    if template_file is None:
        template_file = "observe.dat"

    with open(template_file, "r") as file:
        content = file.readlines()

    line0 = content[0].split()
    obs_type = line0[0]

    start_lines_datablock: list[int] = []
    for number, line in enumerate(content, 0):
        l = line.split()
        if len(l) == 2:
            start_lines_datablock.append(number)
            if out:
                print(" data block", l[0], "with", l[1], "sites begins at line", number)
        if "END" in l:
            start_lines_datablock.append(number - 1)
            if out:
                print(" no further data block in file")

    def _draw_one(val: float, err_scaled: float) -> float:
        """Draw a single perturbed datum using the configured distribution."""
        if not np.isfinite(err_scaled) or err_scaled <= 0.0:
            return val
        if dist_name == "normal":
            return float(rng.normal(loc=val, scale=err_scaled))
        else:  # uniform
            a = float(draw_seq[1]) if len(draw_seq) > 1 else -1.0
            b = float(draw_seq[2]) if len(draw_seq) > 2 else 1.0
            return float(val + rng.uniform(a, b) * err_scaled)

    num_datablock = len(start_lines_datablock) - 1
    for block in np.arange(num_datablock):
        start_block = start_lines_datablock[block]
        end_block = start_lines_datablock[block + 1]
        data_block = content[start_block:end_block]

        if out:
            print(np.shape(data_block))
        start_lines_site: list[int] = []
        num_freqs: list[int] = []
        for number, line in enumerate(data_block, 0):
            l = line.split()
            if len(l) == 4:
                if out:
                    print(l)
                start_lines_site.append(number)
                num_freqs.append(int(data_block[number + 1].split()[0]))
                if out:
                    print("  site", l[0], "begins at line", number)
            if "END" in l:
                start_lines_datablock.append(number - 1)
                if out:
                    print(" no further site block in file")
        if out:
            print("\n")

        num_sites = len(start_lines_site)
        for site in np.arange(num_sites):
            start_site = start_lines_site[site]
            end_site = start_site + num_freqs[site] + 2
            site_block = data_block[start_site:end_site]

            if "MT" in obs_type:
                dat_length = 8
                num_freq = int(site_block[1].split()[0])
                if out:
                    print("   site ", site, "has", num_freq, "frequencies")

                obs: list[list[float]] = []
                for line in site_block[2:]:
                    tmp = [float(x) for x in line.split()]
                    obs.append(tmp)

                for line in obs:
                    for ii in range(1, dat_length + 1):
                        val = line[ii]
                        err_scaled = line[ii + dat_length] * scalfac
                        line[ii] = _draw_one(val, err_scaled)

                for f in range(num_freq - 1):
                    site_block[f + 2] = "    ".join(
                        f"{x:.8E}" for x in obs[f]
                    ) + "\n"

            elif "VTF" in obs_type:
                dat_length = 4
                num_freq = int(site_block[1].split()[0])
                if out:
                    print("   site ", site, "has", num_freq, "frequencies")

                obs = []
                for line in site_block[2:]:
                    tmp = [float(x) for x in line.split()]
                    obs.append(tmp)

                for line in obs:
                    for ii in range(1, dat_length + 1):
                        val = line[ii]
                        err_scaled = line[ii + dat_length] * scalfac
                        line[ii] = _draw_one(val, err_scaled)

                for f in range(num_freq - 1):
                    site_block[f + 2] = "    ".join(
                        f"{x:.8E}" for x in obs[f]
                    ) + "\n"
            else:
                sys.exit(f"modify_data_fcn: {obs_type} not yet implemented! Exit.")

            data_block[start_site:end_site] = site_block

        content[start_block:end_block] = data_block

    with open(template_file, "w") as f:
        f.writelines(content)

    if out:
        print(f"File {template_file} successfully written.")


def get_femtic_sorted(files: Sequence[str], out: bool = True) -> list[str]:
    """
    Sort FEMTIC sensitivity matrix files of the form 'sensMatFreqXXXX'
    by their integer suffix and return a sorted list.

    Parameters
    ----------
    files : sequence of str
        File names.
    out : bool
        If True, print the sorted list.

    Returns
    -------
    listfiles : list of str
        Sorted file names.
    """
    numbers = [int(f[11:]) for f in files]
    numbers = sorted(numbers)

    listfiles = [f"sensMatFreq{ii}" for ii in numbers]
    if out:
        print(listfiles)

    return listfiles


def get_femtic_sites(
    imp_file: str = "result_MT.txt",
    vtf_file: str = "result_VTF.txt",
    pt_file: str = "results_PT.txt",
) -> None:
    """
    Generate sites_XXX.txt files from FEMTIC results files.

    Parameters
    ----------
    imp_file : str
        MT result file -> sites_MT.txt
    vtf_file : str
        VTF result file -> sites_VTF.txt
    pt_file : str
        PT result file -> sites_PT.txt

    Notes
    -----
    The function mirrors the original femtic.py behaviour but fixes some
    minor robustness issues. It writes files where each line is

        sitename sitename

    for each unique site encountered in the corresponding results file.
    """
    # MT
    if len(imp_file) > 0 and os.path.exists(imp_file):
        with open(imp_file, "r") as filein_imp:
            site = ""
            fileout_imp = open(imp_file.replace("results", "sites"), "w")
            filein_imp.readline()
            for line in filein_imp:
                nextsite = line.strip().split()[0]
                if nextsite != site:
                    fileout_imp.write(nextsite + " " + nextsite + "\n")
                    site = nextsite
            fileout_imp.close()
    else:
        if len(imp_file) > 0:
            print(imp_file, "does not exist!")
        else:
            print("imp_file not defined!")

    # VTF
    if len(vtf_file) > 0 and os.path.exists(vtf_file):
        with open(vtf_file, "r") as filein_vtf:
            site = ""
            fileout_vtf = open(vtf_file.replace("results", "sites"), "w")
            filein_vtf.readline()
            for line in filein_vtf:
                nextsite = line.strip().split()[0]
                if nextsite != site:
                    fileout_vtf.write(nextsite + " " + nextsite + "\n")
                    site = nextsite
            fileout_vtf.close()
    else:
        if len(vtf_file) > 0:
            print(vtf_file, "does not exist!")
        else:
            print("vtf_file not defined!")

    # PT
    if len(pt_file) > 0 and os.path.exists(pt_file):
        with open(pt_file, "r") as filein_pt:
            site = ""
            fileout_pt = open(pt_file.replace("results", "sites"), "w")
            filein_pt.readline()
            for line in filein_pt:
                nextsite = line.strip().split()[0]
                if nextsite != site:
                    fileout_pt.write(nextsite + " " + nextsite + "\n")
                    site = nextsite
            fileout_pt.close()
    else:
        if len(pt_file) > 0:
            print(pt_file, "does not exist!")
        else:
            print("pt_file not defined!")


def get_femtic_data(
    data_file: str,
    site_file: str,
    data_type: str = "rhophas",
    out: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Read FEMTIC results (MT, VTF or PT) and site information.

    Parameters
    ----------
    data_file : str
        results_XXX.txt file from FEMTIC.
    site_file : str
        site metadata CSV-like file with columns:
            name, lat, lon, elev, num
    data_type : {"rhophas", "imp", "vtf", "pt"}
        Type of FEMTIC results file.
    out : bool
        If True, print debugging info.

    Returns
    -------
    data_dict : dict
        Dictionary containing:
            - "sites": site indices (0-based)
            - "frq": frequencies
            - "per": periods
            - "lat", "lon", "elv"
            - "num": site index (0-based)
            - "nam": site name
            - "cal", "obs", "err" (arrays of appropriate shape)
    """
    data: list[list[float]] = []
    with open(data_file, "r") as f:
        iline = -1
        for line in f:
            iline += 1
            if iline == 0:
                continue
            l = [float(x) for x in line.split()]
            l[0] = int(l[0])
            data.append(l)
    data_arr = np.asarray(data, dtype=float)

    info: list[list[float | str]] = []
    with open(site_file, "r") as f:
        for line in f:
            l = line.split(",")
            l[1] = float(l[1])
            l[2] = float(l[2])
            l[3] = float(l[3])
            l[4] = int(l[4])
            info.append(l)
    info_arr = np.asarray(info, dtype=object)

    sites = np.unique(data_arr[:, 0]).astype(int) - 1

    head_dict: Dict[str, np.ndarray] = {
        "sites": sites,
        "frq": data_arr[:, 1],
        "per": 1.0 / data_arr[:, 1],
        "lat": np.asarray(info_arr[:, 1], dtype=float),
        "lon": np.asarray(info_arr[:, 2], dtype=float),
        "elv": np.asarray(info_arr[:, 3], dtype=float),
        "num": data_arr[:, 0].astype(int) - 1,
        "nam": np.asarray(info_arr[:, 0])[sites.astype(int)],
    }

    dtyp = data_type.lower()
    if "rhophas" in dtyp:
        type_dict = {
            "cal": data_arr[:, 2:10],
            "obs": data_arr[:, 10:18],
            "err": data_arr[:, 18:26],
        }
    elif "imp" in dtyp:
        ufact = 1.0e-4 / (4.0 * np.pi)
        type_dict = {
            "cal": ufact * data_arr[:, 2:10],
            "obs": ufact * data_arr[:, 10:18],
            "err": ufact * data_arr[:, 18:26],
        }
    elif "vtf" in dtyp:
        type_dict = {
            "cal": data_arr[:, 2:6],
            "obs": data_arr[:, 6:10],
            "err": data_arr[:, 10:14],
        }
    elif "pt" in dtyp:
        type_dict = {
            "cal": data_arr[:, 2:6],
            "obs": data_arr[:, 6:10],
            "err": data_arr[:, 10:14],
        }
    else:
        sys.exit(f"get_femtic_data: data type {dtyp} not implemented! Exit.")

    data_dict = {**head_dict, **type_dict}
    if out:
        print("get_femtic_data: loaded", data_type, "with shape", data_arr.shape)
    return data_dict


def centroid_tetrahedron(nodes: np.ndarray) -> np.ndarray:
    """
    Compute centroid of a tetrahedron given node coordinates.

    Parameters
    ----------
    nodes : ndarray, shape (4, 3) or (3, 4)
        Node coordinates of the tetrahedron. Both shapes are accepted;
        the function will transpose if needed.

    Returns
    -------
    centre : ndarray, shape (3,)
        Centroid coordinates.
    """
    arr = np.asarray(nodes, dtype=float)
    if arr.shape == (3, 4):
        arr = arr.T
    if arr.shape != (4, 3):
        sys.exit("centroid_tetrahedron: Nodes shape must be (4,3) or (3,4)! Exit.")
    centre = np.mean(arr, axis=0)
    return centre


# ============================================================================
# SECTION 2: roughness / prior covariance / matrix tools
# Canonical implementations live in ensembles.py; imported here for
# backward-compatible access via femtic.<name>().
# ============================================================================

try:
    from ensembles import (  # type: ignore
        get_roughness,
        make_prior_cov,
        prune_inplace,
        prune_rebuild,
        dense_to_csr,
        save_spilu,
        load_spilu,
        matrix_reduce,
        check_sparse_matrix,
    )
except ImportError:  # pragma: no cover
    raise ImportError(
        "femtic.py requires ensembles.py to be importable for matrix/roughness tools. "
        "Ensure ensembles.py is on sys.path."
    )


# ============================================================================
# SECTION 4: mesh.dat + resistivity_block_iterX.dat → NPZ
# ============================================================================


def read_femtic_mesh(mesh_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a FEMTIC TETRA mesh file and return node coordinates and connectivity.

    Parameters
    ----------
    mesh_path : str
        Path to the mesh file (usually "mesh.dat").

    Returns
    -------
    nodes : ndarray, shape (nn, 3)
        Node coordinates [x, y, z].
    conn : ndarray, shape (nelem, 4)
        Tetrahedral connectivity (0-based node indices).
    """
    with open(mesh_path, "r", errors="ignore") as f:
        header = f.readline().strip()
        if header.upper() != "TETRA":
            raise ValueError(f"Unsupported mesh type '{header}', expected 'TETRA'.")

        nn_line = f.readline().split()
        if not nn_line:
            raise ValueError("Missing node count after 'TETRA' header.")
        nn = int(nn_line[0])

        nodes = np.empty((nn, 3), dtype=float)
        for _ in range(nn):
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF while reading node coordinates.")
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Node line has too few columns: {line!r}")
            idx = int(parts[0])
            x, y, z = map(float, parts[1:4])
            if not (0 <= idx < nn):
                raise ValueError(f"Node index {idx} out of range 0..{nn-1}.")
            nodes[idx] = (x, y, z)

        nelem_line = f.readline().split()
        if not nelem_line:
            raise ValueError("Missing element count line after node block.")
        nelem = int(nelem_line[0])

        conn = np.empty((nelem, 4), dtype=int)
        for _ in range(nelem):
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF while reading element block.")
            parts = line.split()
            if len(parts) < 9:
                raise ValueError(f"Element line has too few columns: {line!r}")
            ie = int(parts[0])
            n1, n2, n3, n4 = map(int, parts[-4:])
            if not (0 <= ie < nelem):
                raise ValueError(f"Element index {ie} out of range 0..{nelem-1}.")
            conn[ie] = (n1, n2, n3, n4)

    return nodes, conn


def read_resistivity_block(block_path: str) -> Dict[str, np.ndarray]:
    """
    Read a FEMTIC resistivity_block_iterX.dat and return region-based data.

    Parameters
    ----------
    block_path : str
        Path to resistivity_block_iterX.dat file.

    Returns
    -------
    data : dict
        Keys:
            "nelem", "nreg",
            "region_of_elem",
            "region_rho", "region_rho_lower", "region_rho_upper",
            "region_n", "region_flag"
    """
    with open(block_path, "r", errors="ignore") as f:
        first = f.readline().split()
        if len(first) < 2:
            raise ValueError("First line must contain 'nelem nreg'.")
        nelem = int(first[0])
        nreg = int(first[1])

        region_of_elem = np.empty(nelem, dtype=int)
        for _ in range(nelem):
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF while reading element-region map.")
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Element-region line has too few columns: {line!r}")
            ie = int(parts[0])
            ireg = int(parts[1])
            if not (0 <= ie < nelem):
                raise ValueError(f"Element index {ie} out of range 0..{nelem-1}.")
            region_of_elem[ie] = ireg

        region_rho = np.empty(nreg, dtype=float)
        region_rho_lower = np.empty(nreg, dtype=float)
        region_rho_upper = np.empty(nreg, dtype=float)
        region_n = np.empty(nreg, dtype=float)
        region_flag = np.empty(nreg, dtype=int)

        for _ in range(nreg):
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF while reading region lines.")
            parts = line.split()
            if len(parts) < 6:
                raise ValueError(f"Region line has too few columns: {line!r}")
            ireg = int(parts[0])
            rho = float(parts[1])
            rho_min = float(parts[2])
            rho_max = float(parts[3])
            n = float(parts[4])
            flag = int(parts[5])
            if not (0 <= ireg < nreg):
                raise ValueError(f"Region index {ireg} out of range 0..{nreg-1}.")
            region_rho[ireg] = rho
            region_rho_lower[ireg] = rho_min
            region_rho_upper[ireg] = rho_max
            region_n[ireg] = n
            region_flag[ireg] = flag

    return {
        "nelem": np.array(nelem, dtype=int),
        "nreg": np.array(nreg, dtype=int),
        "region_of_elem": region_of_elem,
        "region_rho": region_rho,
        "region_rho_lower": region_rho_lower,
        "region_rho_upper": region_rho_upper,
        "region_n": region_n,
        "region_flag": region_flag,
    }


def build_element_arrays(
    nodes: np.ndarray,
    conn: np.ndarray,
    region_of_elem: np.ndarray,
    region_rho: np.ndarray,
    region_rho_lower: np.ndarray,
    region_rho_upper: np.ndarray,
    region_n: np.ndarray,
    region_flag: np.ndarray,
    clip_eps: float = 1.0e-30,
) -> Dict[str, np.ndarray]:
    """
    Build per-element arrays (centroids, log10 resistivity, bounds, flags, n).

    Parameters
    ----------
    nodes : ndarray, shape (nn, 3)
    conn : ndarray, shape (nelem, 4)
    region_of_elem : ndarray, shape (nelem,)
    region_rho, region_rho_lower, region_rho_upper : ndarray, shape (nreg,)
    region_n : ndarray, shape (nreg,)
    region_flag : ndarray, shape (nreg,)
    clip_eps : float
        Values smaller than this are clipped before log10.

    Returns
    -------
    arrays : dict
        Element-based arrays, including:
            centroid, region, log10_resistivity,
            rho_lower, rho_upper, flag, n
    """
    nodes = np.asarray(nodes, dtype=float)
    conn = np.asarray(conn, dtype=int)
    region_of_elem = np.asarray(region_of_elem, dtype=int)

    nelem = conn.shape[0]
    if region_of_elem.shape[0] != nelem:
        raise ValueError("region_of_elem length does not match number of elements.")

    coords = nodes[conn]
    centroid = coords.mean(axis=1)

    rho = np.clip(region_rho, clip_eps, np.inf)
    rho_min = np.clip(region_rho_lower, clip_eps, np.inf)
    rho_max = np.clip(region_rho_upper, clip_eps, np.inf)

    rid = region_of_elem
    rho_elem = rho[rid]
    rho_min_elem = rho_min[rid]
    rho_max_elem = rho_max[rid]
    n_elem = region_n[rid]
    flag_elem = region_flag[rid]

    log10_rho = np.log10(rho_elem)
    log10_rho_min = np.log10(rho_min_elem)
    log10_rho_max = np.log10(rho_max_elem)

    return {
        "centroid": centroid,
        "region": rid,
        "log10_resistivity": log10_rho,
        "rho_lower": log10_rho_min,
        "rho_upper": log10_rho_max,
        "flag": flag_elem,
        "n": n_elem,
    }



def _rotation_matrix_zyx(angles_deg: Sequence[float]) -> np.ndarray:
    """Return a 3x3 rotation matrix for intrinsic Z-Y-X (yaw-pitch-roll) rotations.

    Parameters
    ----------
    angles_deg : sequence of float
        Three angles in degrees interpreted as ``(yaw_z, pitch_y, roll_x)``.
        The rotations are applied in the order:

            1) yaw about global z-axis
            2) pitch about global y-axis
            3) roll about global x-axis

        The returned matrix ``R`` maps **local** coordinates to **global** coordinates:

            p_global = R @ p_local

        Therefore, to transform a global point into the local ellipsoid frame, use:

            p_local = R.T @ (p_global - center)

    Returns
    -------
    ndarray, shape (3, 3)
        Rotation matrix.

    Notes
    -----
    This convention is chosen because it is common in geoscience workflows and easy
    to invert (via transpose). If you need a different convention, implement a
    dedicated helper and use it in :func:`ellipsoid_mask`.
    """
    ang = np.asarray(list(angles_deg), dtype=float).ravel()
    if ang.size != 3:
        raise ValueError(f"_rotation_matrix_zyx: expected 3 angles, got {ang.size}.")
    yaw, pitch, roll = np.deg2rad(ang)

    cz, sz = float(np.cos(yaw)), float(np.sin(yaw))
    cy, sy = float(np.cos(pitch)), float(np.sin(pitch))
    cx, sx = float(np.cos(roll)), float(np.sin(roll))

    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=float)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)

    # Intrinsic Z-Y-X
    return Rz @ Ry @ Rx


def _rotation_matrix_sds(angles_deg: Sequence[float]) -> np.ndarray:
    """Return a 3x3 rotation matrix for strike/dip/slant angles.

    Parameters
    ----------
    angles_deg : sequence of float
        Three angles in degrees interpreted as ``(strike, dip, slant)``.

        This uses a common geophysical Euler-angle convention:

            R = Rz(strike) @ Rx(dip) @ Rz(slant)

        where rotations are **right-handed** about the indicated axes, and the
        returned matrix ``R`` maps **local** coordinates to **global** coordinates:

            p_global = R @ p_local

        To transform a global point into the local ellipsoid frame:

            p_local = R.T @ (p_global - center)

    Returns
    -------
    ndarray, shape (3, 3)
        Rotation matrix.

    Notes
    -----
    The meaning of strike depends on your coordinate system. This implementation
    treats ``strike`` as a right-handed rotation about the +z axis in the model
    coordinate system.

    If your strike is defined clockwise from North (common in geology), convert
    to a mathematical CCW-from +x convention before calling, e.g.:

        strike_math = 90.0 - strike_geo
    """
    ang = np.asarray(list(angles_deg), dtype=float).ravel()
    if ang.size != 3:
        raise ValueError(f"_rotation_matrix_sds: expected 3 angles, got {ang.size}.")
    strike, dip, slant = np.deg2rad(ang)

    cz1, sz1 = float(np.cos(strike)), float(np.sin(strike))
    cx, sx = float(np.cos(dip)), float(np.sin(dip))
    cz2, sz2 = float(np.cos(slant)), float(np.sin(slant))

    Rz1 = np.array([[cz1, -sz1, 0.0], [sz1, cz1, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
    Rz2 = np.array([[cz2, -sz2, 0.0], [sz2, cz2, 0.0], [0.0, 0.0, 1.0]], dtype=float)

    return Rz1 @ Rx @ Rz2


def ellipsoid_mask(
    centroids: np.ndarray,
    *,
    center: Sequence[float],
    axes: Sequence[float],
    angles_deg: Sequence[float] = (0.0, 0.0, 0.0),
    convention: Literal["zyx", "sds"] = "zyx",
) -> np.ndarray:
    """Compute a boolean mask for points inside a rotated ellipsoid.

    Parameters
    ----------
    centroids : ndarray, shape (n, 3)
        Point coordinates (typically element centroids).
    center : sequence of float
        Ellipsoid centre ``(cx, cy, cz)`` in the same coordinate system as centroids.
    axes : sequence of float
        Semi-axis lengths ``(a, b, c)``. Must be positive.
    angles_deg : sequence of float, optional
        Rotation angles in degrees. Interpretation depends on ``convention``.
        Default is no rotation.
    convention : {"zyx", "sds"}
        Rotation convention.

        - ``"zyx"``: intrinsic Z-Y-X (yaw, pitch, roll), see :func:`_rotation_matrix_zyx`.
        - ``"sds"``: strike/dip/slant, see :func:`_rotation_matrix_sds`.

    Returns
    -------
    ndarray, shape (n,)
        Boolean mask where True indicates the point lies inside (or on) the ellipsoid:

            (x/a)^2 + (y/b)^2 + (z/c)^2 <= 1

        after translating by ``center`` and rotating into the ellipsoid frame.

    Raises
    ------
    ValueError
        If input shapes or parameters are invalid.
    """
    pts = np.asarray(centroids, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"ellipsoid_mask: centroids must have shape (n,3); got {pts.shape}.")

    c = np.asarray(list(center), dtype=float).ravel()
    if c.size != 3:
        raise ValueError(f"ellipsoid_mask: center must have length 3; got {c.size}.")

    ax = np.asarray(list(axes), dtype=float).ravel()
    if ax.size != 3:
        raise ValueError(f"ellipsoid_mask: axes must have length 3; got {ax.size}.")
    if np.any(ax <= 0.0) or not np.all(np.isfinite(ax)):
        raise ValueError("ellipsoid_mask: axes must be finite and > 0.")

    conv = str(convention).strip().lower()
    if conv == "zyx":
        R = _rotation_matrix_zyx(angles_deg)
    elif conv == "sds":
        R = _rotation_matrix_sds(angles_deg)
    else:
        raise ValueError(f"ellipsoid_mask: unsupported convention={convention!r}.")

    # Transform to local ellipsoid coordinates: p_local = R^T (p - c)
    local = (pts - c[None, :]) @ R  # (p-c) @ R == (R^T (p-c))^T, but we need per-row
    # Above uses row-vectors; equivalent to local_i = (p_i-c) * R
    # Test quadratic form
    q = (local[:, 0] / ax[0]) ** 2 + (local[:, 1] / ax[1]) ** 2 + (local[:, 2] / ax[2]) ** 2
    return q <= 1.0


def ellipsoid_fill_element_npz(
    npz_in: str | Path,
    npz_out: str | Path,
    *,
    center: Sequence[float],
    axes: Sequence[float],
    angles_deg: Sequence[float] = (0.0, 0.0, 0.0),
    angle_convention: Literal["zyx", "sds"] = "zyx",
    fill_value: float,
    fill_space: Literal["log10", "rho"] = "log10",
    fill_flag: int = 0,
    fill_bounds: tuple[float, float] | None = None,
    fill_n: float = 1.0,
    respect_fixed: bool = True,
    out: bool = True,
) -> Path:
    """Fill an ellipsoidal region (by element centroids) in an *element* NPZ model.

    This helper operates on the NPZ format produced by :func:`mesh_and_block_to_npz`,
    i.e. the file that contains:

    - ``nodes``, ``conn`` (mesh)
    - ``region_of_elem`` and per-region tables (``region_rho``, bounds, flags, ...)
    - per-element arrays (``centroid``, ``log10_resistivity``, ...)

    Because FEMTIC resistivity blocks are region-based, a *spatially local* edit is
    implemented by:

    1) creating a **new region** with the requested resistivity (``fill_value``),
    2) re-assigning elements whose centroids fall inside the ellipsoid to that new region,
    3) updating the per-element arrays so they remain consistent.

    Fixed-region compatibility
    --------------------------
    If ``respect_fixed=True`` (default), elements belonging to fixed regions are not
    re-assigned. Fixed regions are defined as:

    - region 0 (air) always treated as fixed for this operation
    - any region with ``region_flag == 1``

    Parameters
    ----------
    npz_in, npz_out
        Input and output NPZ paths.
    center, axes, angles_deg, angle_convention
        Ellipsoid parameters; see :func:`ellipsoid_mask`.
    fill_value
        Fill value for the new region. Interpreted according to ``fill_space``.
    fill_space
        ``"log10"`` (default) means ``fill_value`` is log10(ρ) and is converted to ρ.
        ``"rho"`` means ``fill_value`` is ρ in Ωm.
    fill_flag
        Region flag for the new region (default 0 = free). Use 1 to make the filled
        region fixed in subsequent workflows.
    fill_bounds
        Optional (rho_lower, rho_upper) bounds for the new region (in Ωm). If None,
        bounds are set equal to the fill resistivity.
    fill_n
        Region ``n`` value written for the new region (stored as float).
    respect_fixed
        If True, do not re-assign elements that belong to fixed regions.
    out
        If True, print a one-line summary.

    Returns
    -------
    Path
        Path to the written NPZ.

    Raises
    ------
    KeyError
        If required NPZ fields are missing.
    ValueError
        If shapes/parameters are inconsistent.

    Author: Volker Rath (DIAS)
    Created with the help of ChatGPT (GPT-5 Thinking) on 2026-01-03 (UTC)
    """
    p_in = Path(npz_in)
    with np.load(p_in, allow_pickle=True) as z:
        data = {k: z[k] for k in z.files}

    required = [
        "nodes",
        "conn",
        "region_of_elem",
        "region_rho",
        "region_rho_lower",
        "region_rho_upper",
        "region_n",
        "region_flag",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"ellipsoid_fill_element_npz: NPZ missing required fields: {missing}")

    nodes = np.asarray(data["nodes"], dtype=float)
    conn = np.asarray(data["conn"], dtype=int)
    region_of_elem = np.asarray(data["region_of_elem"], dtype=int).ravel()

    nelem = int(region_of_elem.size)
    if conn.shape[0] != nelem:
        raise ValueError(
            "ellipsoid_fill_element_npz: conn and region_of_elem inconsistent: "
            f"conn has {conn.shape[0]} elements, region_of_elem has {nelem}."
        )

    # Centroids: use stored centroids if present, otherwise compute.
    if "centroid" in data:
        centroid = np.asarray(data["centroid"], dtype=float)
        if centroid.shape != (nelem, 3):
            raise ValueError(
                f"ellipsoid_fill_element_npz: centroid shape mismatch: {centroid.shape} vs ({nelem},3)."
            )
    else:
        centroid = nodes[conn].mean(axis=1)

    # Compute mask for elements inside ellipsoid
    inside = ellipsoid_mask(centroid, center=center, axes=axes, angles_deg=angles_deg, convention=angle_convention)

    region_rho = np.asarray(data["region_rho"], dtype=float).ravel()
    region_rho_lo = np.asarray(data["region_rho_lower"], dtype=float).ravel()
    region_rho_hi = np.asarray(data["region_rho_upper"], dtype=float).ravel()
    region_n = np.asarray(data["region_n"], dtype=float).ravel()
    region_flag = np.asarray(data["region_flag"], dtype=int).ravel()

    nreg_old = int(region_rho.size)
    if any(arr.size != nreg_old for arr in [region_rho_lo, region_rho_hi, region_n, region_flag]):
        raise ValueError("ellipsoid_fill_element_npz: region arrays must all have length nreg.")

    # Determine elements allowed to change
    if respect_fixed:
        fixed_regions = set([0])
        fixed_regions.update(np.where(region_flag == 1)[0].tolist())
        can_change = ~np.isin(region_of_elem, np.asarray(sorted(fixed_regions), dtype=int))
        sel = inside & can_change
    else:
        sel = inside

    # Convert fill value to rho
    fs = str(fill_space).strip().lower()
    if fs == "log10":
        rho_fill = float(10.0 ** float(fill_value))
    elif fs in {"rho", "none"}:
        rho_fill = float(fill_value)
    else:
        raise ValueError(f"ellipsoid_fill_element_npz: fill_space must be 'log10' or 'rho', got {fill_space!r}.")

    if not np.isfinite(rho_fill) or rho_fill <= 0.0:
        raise ValueError("ellipsoid_fill_element_npz: fill resistivity must be finite and > 0.")

    if fill_bounds is None:
        rho_lo_fill = rho_fill
        rho_hi_fill = rho_fill
    else:
        rho_lo_fill = float(fill_bounds[0])
        rho_hi_fill = float(fill_bounds[1])

    # Append new region
    new_reg = nreg_old
    region_rho = np.concatenate([region_rho, np.array([rho_fill], dtype=float)])
    region_rho_lo = np.concatenate([region_rho_lo, np.array([rho_lo_fill], dtype=float)])
    region_rho_hi = np.concatenate([region_rho_hi, np.array([rho_hi_fill], dtype=float)])
    region_n = np.concatenate([region_n, np.array([float(fill_n)], dtype=float)])
    region_flag = np.concatenate([region_flag, np.array([int(fill_flag)], dtype=int)])

    # Reassign selected elements to new region
    region_of_elem_new = region_of_elem.copy()
    region_of_elem_new[sel] = int(new_reg)

    # Update derived per-element arrays if they exist
    rid = region_of_elem_new
    clip_eps = 1.0e-30
    rho_clip = np.clip(region_rho, clip_eps, np.inf)
    rho_lo_clip = np.clip(region_rho_lo, clip_eps, np.inf)
    rho_hi_clip = np.clip(region_rho_hi, clip_eps, np.inf)

    log10_rho = np.log10(rho_clip[rid])
    log10_rho_lo = np.log10(rho_lo_clip[rid])
    log10_rho_hi = np.log10(rho_hi_clip[rid])
    flag_elem = region_flag[rid]
    n_elem = region_n[rid]

    # Write output NPZ preserving all keys, updating the consistent ones
    data["nreg"] = np.array(int(new_reg + 1), dtype=int)
    data["region_rho"] = region_rho
    data["region_rho_lower"] = region_rho_lo
    data["region_rho_upper"] = region_rho_hi
    data["region_n"] = region_n
    data["region_flag"] = region_flag
    data["region_of_elem"] = region_of_elem_new

    if "region" in data:
        data["region"] = region_of_elem_new
    if "log10_resistivity" in data:
        data["log10_resistivity"] = log10_rho
    if "rho_lower" in data:
        data["rho_lower"] = log10_rho_lo
    if "rho_upper" in data:
        data["rho_upper"] = log10_rho_hi
    if "flag" in data:
        data["flag"] = flag_elem
    if "n" in data:
        data["n"] = n_elem
    if "centroid" not in data:
        data["centroid"] = centroid

    p_out = Path(npz_out)
    np.savez_compressed(p_out, **data)

    if out:
        n_inside = int(inside.sum())
        n_sel = int(sel.sum())
        print(
            f"ellipsoid_fill_element_npz: in={p_in.name}, out={p_out.name}, "
            f"inside={n_inside}, modified={n_sel}, new_region={new_reg}."
        )

    return p_out
def save_element_npz_with_mesh_and_regions(
    out_path: str,
    nodes: np.ndarray,
    conn: np.ndarray,
    block: Dict[str, np.ndarray],
    arrays: Dict[str, np.ndarray],
) -> None:
    """
    Save mesh, region data, and element arrays into a compressed NPZ file.

    Parameters
    ----------
    out_path : str
        Output npz path.
    nodes : ndarray, shape (nn, 3)
    conn : ndarray, shape (nelem, 4)
    block : dict
        Region-level data from :func:`read_resistivity_block`.
    arrays : dict
        Element arrays from :func:`build_element_arrays`.
    """
    np.savez_compressed(
        out_path,
        nodes=nodes,
        conn=conn,
        nelem=block["nelem"],
        nreg=block["nreg"],
        region_of_elem=block["region_of_elem"],
        region_rho=block["region_rho"],
        region_rho_lower=block["region_rho_lower"],
        region_rho_upper=block["region_rho_upper"],
        region_n=block["region_n"],
        region_flag=block["region_flag"],
        **arrays,
    )


def mesh_and_block_to_npz(
    mesh_path: str,
    rho_block_path: str,
    out_npz: str,
) -> None:
    """
    High-level helper: mesh.dat + resistivity_block_iterX.dat → element NPZ.

    Parameters
    ----------
    mesh_path : str
        FEMTIC mesh.dat.
    rho_block_path : str
        FEMTIC resistivity_block_iterX.dat.
    out_npz : str
        Output npz file.
    """
    nodes, conn = read_femtic_mesh(mesh_path)
    block = read_resistivity_block(rho_block_path)
    arrays = build_element_arrays(
        nodes=nodes,
        conn=conn,
        region_of_elem=block["region_of_elem"],
        region_rho=block["region_rho"],
        region_rho_lower=block["region_rho_lower"],
        region_rho_upper=block["region_rho_upper"],
        region_n=block["region_n"],
        region_flag=block["region_flag"],
    )
    save_element_npz_with_mesh_and_regions(out_npz, nodes, conn, block, arrays)


# ============================================================================
# SECTION 5: NPZ → VTK/VTU
# ============================================================================


def npz_to_unstructured_grid(npz_path: str):
    """
    Build a PyVista UnstructuredGrid from a FEMTIC NPZ file.

    Parameters
    ----------
    npz_path : str
        NPZ created by :func:`mesh_and_block_to_npz`.

    Returns
    -------
    grid : pyvista.UnstructuredGrid
        Unstructured grid with cell_data fields.
    """
    try:
        import pyvista as pv
    except Exception as exc:  # pragma: no cover
        raise ImportError("pyvista is required for VTK export.") from exc

    data = np.load(npz_path)
    if "nodes" not in data or "conn" not in data:
        raise KeyError("NPZ must at least contain 'nodes' and 'conn'.")

    nodes = np.asarray(data["nodes"], dtype=float)
    conn = np.asarray(data["conn"], dtype=int)

    nelem = conn.shape[0]
    if conn.shape[1] != 4:
        raise ValueError("Connectivity must have 4 nodes per element (tetra).")

    n_per_cell = 4
    cells = np.hstack(
        [np.full((nelem, 1), n_per_cell, dtype=np.int64), conn.astype(np.int64)]
    ).ravel()

    try:
        celltypes = np.full(nelem, pv.CellType.TETRA, dtype=np.uint8)
    except Exception:  # old pyvista
        celltypes = np.full(nelem, 10, dtype=np.uint8)

    grid = pv.UnstructuredGrid(cells, celltypes, nodes)

    preferred_keys = [
        "log10_resistivity",
        "rho_lower",
        "rho_upper",
        "flag",
        "n",
        "region",
        "region_of_elem",
    ]
    for key in preferred_keys:
        if key in data:
            arr = np.asarray(data[key])
            if arr.shape[0] == nelem:
                name = "region" if key == "region_of_elem" else key
                grid.cell_data[name] = arr

    return grid


def save_vtk_from_npz(
    npz_path: str,
    out_vtu: str,
    out_legacy: Optional[str] = None,
    scalar_name: str = "log10_resistivity",
) -> None:
    """
    Export a FEMTIC NPZ model to VTK/VTU unstructured grid file(s).

    Parameters
    ----------
    npz_path : str
    out_vtu : str
        VTU output file.
    out_legacy : str, optional
        Optional legacy .vtk file.
    scalar_name : str
        Name of scalar to use for plotting.
    """
    grid = npz_to_unstructured_grid(npz_path)
    grid.save(out_vtu)
    print("Wrote VTU:", out_vtu)

    if out_legacy is not None:
        grid.save(out_legacy)
        print("Wrote legacy VTK:", out_legacy)

    if scalar_name in grid.cell_data:
        print(f"Scalar '{scalar_name}' available for plotting.")
    else:
        print(f"Scalar '{scalar_name}' not present in grid.cell_data.")


# ============================================================================
# ============================================================================
# SECTION 6: NPZ ↔ HDF5 / NetCDF
# ============================================================================


def npz_to_hdf5(
    npz_path: str,
    hdf5_path: str,
    group: str = "femtic_model",
    compression: str = "gzip",
    compression_opts: int = 4,
) -> None:
    """
    Write all arrays from a FEMTIC NPZ file to an HDF5 file.

    Parameters
    ----------
    npz_path : str
        Input NPZ file created by :func:`mesh_and_block_to_npz`.
    hdf5_path : str
        Output HDF5 file.
    group : str, optional
        Name of the top-level group into which all datasets are written.
    compression : str, optional
        HDF5 compression method for datasets (default "gzip").
    compression_opts : int, optional
        Compression level (only used for some compressors such as "gzip").

    Notes
    -----
    - Each key in the NPZ is written as one dataset under the given group.
    - Scalar values (0-D arrays) are written as scalar datasets.
    - Basic attributes "source_npz" and "created_by" are attached to the group.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-12-12
    """
    try:
        import h5py
    except Exception as exc:  # pragma: no cover
        raise ImportError("h5py is required for npz_to_hdf5().") from exc

    data = np.load(npz_path)
    with h5py.File(hdf5_path, "w") as h5:
        g = h5.create_group(group)
        g.attrs["source_npz"] = str(npz_path)
        g.attrs["created_by"] = "femtic.npz_to_hdf5"

        for key in data.files:
            arr = data[key]
            if arr.shape == ():
                dset = g.create_dataset(key, data=arr[()])
            else:
                dset = g.create_dataset(
                    key,
                    data=arr,
                    compression=compression,
                    compression_opts=compression_opts,
                )
            dset.attrs["dtype"] = str(arr.dtype)


def hdf5_to_npz(
    hdf5_path: str,
    npz_path: str,
    group: str = "femtic_model",
) -> None:
    """
    Convert a FEMTIC-style HDF5 file back to NPZ.

    Parameters
    ----------
    hdf5_path : str
        Input HDF5 file previously written by :func:`npz_to_hdf5`.
    npz_path : str
        Output NPZ file path.
    group : str, optional
        Name of the group that contains the FEMTIC datasets. If the group
        is not found, the HDF5 root is used instead.

    Notes
    -----
    The function simply restores one NPZ field per dataset contained in
    the group.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-12-12
    """
    try:
        import h5py
    except Exception as exc:  # pragma: no cover
        raise ImportError("hdf5_to_npz requires the 'h5py' package.") from exc

    import numpy as _np

    with h5py.File(hdf5_path, "r") as h5:
        if group in h5:
            g = h5[group]
        else:
            g = h5
        arrays: dict[str, _np.ndarray] = {}
        for key, item in g.items():
            if isinstance(item, h5py.Dataset):
                arrays[key] = _np.array(item[()])

    _np.savez(npz_path, **arrays)


def npz_to_netcdf(
    npz_path: str,
    netcdf_path: str,
) -> None:
    """
    Convert a FEMTIC NPZ file to a NetCDF file with simple, named dimensions.

    Parameters
    ----------
    npz_path : str
        Input NPZ file created by :func:`mesh_and_block_to_npz`.
    netcdf_path : str
        Output NetCDF file (NetCDF4 format).

    Notes
    -----
    The function tries to create meaningful dimensions when the standard FEMTIC
    NPZ layout is present:

    - "node", "xyz" for ``nodes`` (nn, 3)
    - "cell", "nne" for ``conn`` (nelem, 4)
    - "cell" for cell-based 1-D arrays (e.g. ``log10_resistivity``)
    - "region" for region-based arrays (e.g. ``region_rho``)

    Any remaining arrays are stored using automatically generated dimensions.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-12-12
    """
    try:
        from netCDF4 import Dataset
    except Exception as exc:  # pragma: no cover
        raise ImportError("netCDF4 is required for npz_to_netcdf().") from exc

    import numpy as _np

    data = np.load(npz_path)

    ds = Dataset(netcdf_path, "w", format="NETCDF4")
    ds.setncattr("source_npz", str(npz_path))
    ds.setncattr("created_by", "femtic.npz_to_netcdf")

    def ensure_dim(name: str, size: int) -> None:
        """Create a dimension if it does not yet exist."""
        if name not in ds.dimensions:
            ds.createDimension(name, size)

    # Nodes: (nn, 3) → dims ("node", "xyz")
    if "nodes" in data:
        nodes = data["nodes"]
        if nodes.ndim == 2:
            nn, nxyz = nodes.shape
            ensure_dim("node", nn)
            ensure_dim("xyz", nxyz)
            var = ds.createVariable("nodes", "f8", ("node", "xyz"))
            var[:, :] = nodes

    # Connectivity: (nelem, 4) → dims ("cell", "nne")
    if "conn" in data:
        conn = data["conn"]
        if conn.ndim == 2:
            nelem, nne = conn.shape
            ensure_dim("cell", nelem)
            ensure_dim("nne", nne)
            var = ds.createVariable("conn", "i4", ("cell", "nne"))
            var[:, :] = conn

    # Element centroids: (nelem, 3) → dims ("cell", "xyz")
    if "centroid" in data and "cell" in ds.dimensions and "xyz" in ds.dimensions:
        cent = data["centroid"]
        if cent.shape == (len(ds.dimensions["cell"]), len(ds.dimensions["xyz"])):
            var = ds.createVariable("centroid", "f8", ("cell", "xyz"))
            var[:, :] = cent

    # Scalars nelem / nreg
    if "nelem" in data:
        var = ds.createVariable("nelem", "i4")
        var[...] = int(data["nelem"])
    if "nreg" in data:
        var = ds.createVariable("nreg", "i4")
        var[...] = int(data["nreg"])

    # Region-based arrays
    if "nreg" in data:
        nreg = int(data["nreg"])
        ensure_dim("region", nreg)
        region_keys = [
            "region_rho",
            "region_rho_lower",
            "region_rho_upper",
            "region_n",
            "region_flag",
        ]
        for key in region_keys:
            if key in data:
                arr = data[key]
                if arr.ndim == 1 and arr.shape[0] == nreg:
                    dtype = "f8" if arr.dtype.kind in "fc" else "i4"
                    var = ds.createVariable(key, dtype, ("region",))
                    var[:] = arr

    # Cell-based arrays
    if "cell" in ds.dimensions:
        nelem = len(ds.dimensions["cell"])
        cell_keys = [
            "region_of_elem",
            "log10_resistivity",
            "rho_lower",
            "rho_upper",
            "flag",
            "n",
        ]
        for key in cell_keys:
            if key in data:
                arr = data[key]
                if arr.ndim == 1 and arr.shape[0] == nelem:
                    dtype = "f8" if arr.dtype.kind in "fc" else "i4"
                    var = ds.createVariable(key, dtype, ("cell",))
                    var[:] = arr

    # Any remaining arrays that have not yet been written
    written = set(ds.variables.keys())
    for key in data.files:
        if key in written:
            continue

        arr = data[key]
        if arr.shape == ():
            var = ds.createVariable(key, "f8")
            var[...] = arr[()]
            continue

        dims = []
        for iax, size in enumerate(arr.shape):
            dname = f"{key}_dim{iax}"
            ensure_dim(dname, size)
            dims.append(dname)
        dtype = "f8" if arr.dtype.kind in "fc" else "i4"
        var = ds.createVariable(key, dtype, tuple(dims))
        var[...] = arr

    ds.close()


def netcdf_to_npz(
    netcdf_path: str,
    npz_path: str,
) -> None:
    """
    Convert a NetCDF file written by :func:`npz_to_netcdf` back to NPZ.

    Parameters
    ----------
    netcdf_path : str
        Input NetCDF file (NetCDF4 format).
    npz_path : str
        Output NPZ file path.

    Notes
    -----
    This helper is deliberately simple: it converts each NetCDF variable to
    one NPZ field with the same name. Masked values (if any) are filled with
    ``NaN`` before being written.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-12-12
    """
    try:
        from netCDF4 import Dataset
    except Exception as exc:  # pragma: no cover
        raise ImportError("netcdf_to_npz requires the 'netCDF4' package.") from exc

    import numpy as _np
    import numpy.ma as _ma

    ds = Dataset(netcdf_path, "r")
    arrays: dict[str, _np.ndarray] = {}
    try:
        for name, var in ds.variables.items():
            data = var[...]
            if isinstance(data, _ma.MaskedArray):
                data = data.filled(_np.nan)
            arrays[name] = _np.array(data)
    finally:
        ds.close()

    _np.savez(npz_path, **arrays)


def write_femtic_mesh(
    out_path: str,
    nodes: np.ndarray,
    conn: np.ndarray,
) -> None:
    """
    Write a FEMTIC-style mesh.dat from nodes and connectivity.

    Parameters
    ----------
    out_path : str
    nodes : ndarray, shape (nn, 3)
    conn : ndarray, shape (nelem, 4)
    """
    nodes = np.asarray(nodes, dtype=float)
    conn = np.asarray(conn, dtype=int)
    nn = nodes.shape[0]
    nelem = conn.shape[0]

    with open(out_path, "w") as f:
        f.write("TETRA\n")
        f.write(f"{nn:d}\n")
        for i in range(nn):
            x, y, z = nodes[i]
            f.write(f"{i:d} {x:.8e} {y:.8e} {z:.8e}\n")

        f.write(f"{nelem:d}\n")
        for ie in range(nelem):
            n1, n2, n3, n4 = conn[ie]
            f.write(f"{ie:d} -1 -1 -1 -1 {n1:d} {n2:d} {n3:d} {n4:d}\n")


def write_resistivity_block(
    out_path: str,
    nelem: int,
    nreg: int,
    region_of_elem: np.ndarray,
    region_rho: np.ndarray,
    region_rho_lower: np.ndarray,
    region_rho_upper: np.ndarray,
    region_n: np.ndarray,
    region_flag: np.ndarray,
    fmt: str = "{:.6g}",
) -> None:
    """
    Write a FEMTIC resistivity_block_iterX.dat from region arrays.

    Parameters
    ----------
    out_path : str
    nelem : int
    nreg : int
    region_of_elem : ndarray, shape (nelem,)
    region_rho, region_rho_lower, region_rho_upper : ndarray, shape (nreg,)
    region_n : ndarray, shape (nreg,)
    region_flag : ndarray, shape (nreg,)
    fmt : str
        Float format.
    """
    region_of_elem = np.asarray(region_of_elem, dtype=int)
    region_rho = np.asarray(region_rho, dtype=float)
    region_rho_lower = np.asarray(region_rho_lower, dtype=float)
    region_rho_upper = np.asarray(region_rho_upper, dtype=float)
    region_n = np.asarray(region_n, dtype=float)
    region_flag = np.asarray(region_flag, dtype=int)

    if region_of_elem.shape[0] != nelem:
        raise ValueError("region_of_elem length does not match nelem.")
    if any(
        arr.shape[0] != nreg
        for arr in [region_rho, region_rho_lower, region_rho_upper, region_n, region_flag]
    ):
        raise ValueError("Region arrays must all have length nreg.")

    with open(out_path, "w") as f:
        f.write(f"{nelem:d} {nreg:d}\n")
        for ie in range(nelem):
            f.write(f"{ie:d} {int(region_of_elem[ie]):d}\n")

        for ireg in range(nreg):
            rho = fmt.format(region_rho[ireg])
            rmin = fmt.format(region_rho_lower[ireg])
            rmax = fmt.format(region_rho_upper[ireg])
            nval = int(region_n[ireg])
            flag = int(region_flag[ireg])
            f.write(f"{ireg:d} {rho} {rmin} {rmax} {nval:d} {flag:d}\n")


def npz_to_femtic(
    npz_path: str,
    mesh_out: str,
    rho_block_out: str,
    fmt: str = "{:.6g}",
) -> None:
    """
    Reconstruct FEMTIC mesh.dat and resistivity_block_iterX.dat from NPZ.

    Parameters
    ----------
    npz_path : str
    mesh_out : str
    rho_block_out : str
    fmt : str
        Float format for resistivities and bounds.
    """
    data = np.load(npz_path)

    required = [
        "nodes",
        "conn",
        "nelem",
        "nreg",
        "region_of_elem",
        "region_rho",
        "region_rho_lower",
        "region_rho_upper",
        "region_n",
        "region_flag",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"NPZ is missing required arrays: {missing}")

    nodes = data["nodes"]
    conn = data["conn"]
    nelem = int(data["nelem"])
    nreg = int(data["nreg"])
    region_of_elem = data["region_of_elem"]
    region_rho = data["region_rho"]
    region_rho_lower = data["region_rho_lower"]
    region_rho_upper = data["region_rho_upper"]
    region_n = data["region_n"]
    region_flag = data["region_flag"]

    write_femtic_mesh(mesh_out, nodes, conn)
    write_resistivity_block(
        rho_block_out,
        nelem,
        nreg,
        region_of_elem,
        region_rho,
        region_rho_lower,
        region_rho_upper,
        region_n,
        region_flag,
        fmt=fmt,
    )


"""
read_data.py

Read, perturb, and rewrite FEMTIC-style observation files (typically ``observe.dat``).

The FEMTIC observation file format used here consists of one or more *blocks*.
Each block starts with a header line

    <OBS_TYPE>  <N_SITES>

where ``OBS_TYPE`` is one of ``MT``, ``VTF``, or ``PT``. A block contains one or
more *site* sections, each of the form:

- a site header line with **exactly 4 tokens** (kept verbatim)
- a line containing the number of frequencies (integer)
- ``N_FREQ`` data rows

Each data row is assumed to contain:

- frequency (first column)
- ``dat_length`` data values
- ``dat_length`` error values (standard deviations, σ)

so the minimum number of numeric columns is ``1 + 2 * dat_length``.

For ``MT`` blocks (``dat_length=8``), the data columns are assumed to be ordered as
real/imag pairs:

    Zxx_re, Zxx_im, Zxy_re, Zxy_im, Zyx_re, Zyx_im, Zyy_re, Zyy_im

and similarly for errors.

Derived MT quantities (optional)
--------------------------------
For MT sites the parser can compute and attach derived quantities to each site dict:

1) Phase tensor Φ and bootstrap standard deviations per element
2) Determinant impedance (Berdichevsky invariant):
       Zdet = sqrt(det(Z)) = sqrt(Zxx*Zyy - Zxy*Zyx)
   and bootstrap standard deviations of Re/Im
3) SSQ impedance (Rung-Arunwan / Szarka & Menvielle invariant):
       Zssq = sqrt( (Zxx^2 + Zxy^2 + Zyx^2 + Zyy^2) / 2 )
   and bootstrap standard deviations of Re/Im

Bootstrap here means Monte-Carlo resampling consistent with the per-component
standard deviations provided in the file: for each draw we perturb the real/imag
parts independently by ``Normal(0, σ)`` and recompute the derived quantity.

This module provides:

- :func:`read_observe_dat`   -> parse file into a nested dict structure
- :func:`sites_as_dict_list` -> flatten sites into a list of dicts (EDI-like convenience)
- :func:`modify_data`        -> overwrite error columns from relative errors (optional),
                               draw Gaussian perturbations, and rewrite the file

Author: Volker Rath (DIAS)
Created with the help of ChatGPT (GPT-5 Thinking) on 2025-12-31
"""

_OBS_DATALEN: dict[str, int] = {
    "MT": 8,
    "VTF": 4,
    "PT": 4,
}


_COMPONENT_LABELS: dict[str, list[str]] = {
    # Conventional MT complex impedance ordering (real/imag pairs).
    "MT": ["Zxx_re", "Zxx_im", "Zxy_re", "Zxy_im", "Zyx_re", "Zyx_im", "Zyy_re", "Zyy_im"],
    # Labels for VTF/PT can vary between workflows; keep generic but stable.
    "VTF": ["c0", "c1", "c2", "c3"],
    "PT": ["c0", "c1", "c2", "c3"],
}


def _is_block_header(tokens: list[str]) -> bool:
    """Return True if a tokenized line is a FEMTIC data-block header.

    Parameters
    ----------
    tokens : list of str
        Tokens obtained by splitting a line.

    Returns
    -------
    bool
        True if the token sequence matches the expected block-header pattern.

    Notes
    -----
    A block-header line is recognized as:

    - exactly two tokens
    - first token in {'MT', 'VTF', 'PT'}
    - second token parseable as int (number of sites)

    This heuristic deliberately avoids mis-detecting other lines.
    """
    if len(tokens) != 2:
        return False
    if tokens[0] not in _OBS_DATALEN:
        return False
    try:
        int(tokens[1])
    except Exception:
        return False
    return True


def _rel_err_array(
    err_spec: Sequence[float] | Sequence[int] | np.ndarray,
    dat_length: int,
) -> np.ndarray | None:
    """Normalize a relative-error specification to a length-``dat_length`` array.

    Parameters
    ----------
    err_spec : sequence or ndarray
        Relative error specification. Supported forms are:

        - empty (length 0): return None (do not overwrite existing error columns)
        - scalar (length 1): broadcast to length ``dat_length``
        - vector (length ``dat_length``): used component-wise
    dat_length : int
        Number of data components per frequency for the current observation type.

    Returns
    -------
    ndarray or None
        Relative error array of shape ``(dat_length,)`` or None if no overwrite.

    Raises
    ------
    ValueError
        If a non-empty vector is provided but does not have length 1 or ``dat_length``.
    """
    a = np.asarray(err_spec, dtype=float).ravel()
    if a.size == 0:
        return None
    if a.size == 1:
        return np.repeat(float(a.item()), dat_length)
    if a.size != dat_length:
        raise ValueError(f"Relative error length must be 1 or {dat_length}, got {a.size}")
    return a


def _find_end_index(lines: list[str]) -> int:
    """Return the index of the first line containing token ``END`` (exclusive boundary).

    Parameters
    ----------
    lines : list of str
        File lines including newlines.

    Returns
    -------
    int
        Index of the first ``END`` line. If no such line exists, returns ``len(lines)``.

    Notes
    -----
    The returned index is intended to be used as an exclusive upper bound when slicing.
    """
    for i, line in enumerate(lines):
        if "END" in line.split():
            return i
    return len(lines)


def _format_float(v: float) -> str:
    """Format a float in a stable scientific notation used for rewritten data rows.

    Parameters
    ----------
    v : float
        Value to format.

    Returns
    -------
    str
        Formatted float string.
    """
    return f"{float(v):.8E}"


def _format_row(values: Sequence[float]) -> str:
    """Format a numeric row as a whitespace-separated FEMTIC data line.

    Parameters
    ----------
    values : sequence of float
        Numeric values, expected to begin with frequency, followed by data and errors.

    Returns
    -------
    str
        A single formatted line **including** the trailing newline character.
    """
    return "    ".join(_format_float(v) for v in values) + "\n"


def _mt_data_to_tensor(data_row: np.ndarray) -> np.ndarray:
    """Convert one MT data row (length 8: re/im pairs) to a 2x2 complex tensor.

    Parameters
    ----------
    data_row : ndarray
        One row of MT data values with length 8 and ordering:
        ``Zxx_re, Zxx_im, Zxy_re, Zxy_im, Zyx_re, Zyx_im, Zyy_re, Zyy_im``.

    Returns
    -------
    ndarray
        Complex tensor ``Z`` of shape (2, 2) with entries:
        ``[[Zxx, Zxy], [Zyx, Zyy]]``.
    """
    zxx = complex(float(data_row[0]), float(data_row[1]))
    zxy = complex(float(data_row[2]), float(data_row[3]))
    zyx = complex(float(data_row[4]), float(data_row[5]))
    zyy = complex(float(data_row[6]), float(data_row[7]))
    return np.array([[zxx, zxy], [zyx, zyy]], dtype=np.complex128)


def _phase_tensor_from_Z(Z: np.ndarray) -> np.ndarray:
    """Compute the phase tensor Φ from an MT impedance tensor Z.

    Parameters
    ----------
    Z : ndarray
        Complex impedance tensor of shape (2, 2).

    Returns
    -------
    ndarray
        Phase tensor Φ of shape (2, 2), real-valued.

    Notes
    -----
    Using the standard definition:

        Z = X + iY,   Φ = X^{-1} Y

    where X=Re(Z) and Y=Im(Z).
    """
    X = np.real(Z).astype(float)
    Y = np.imag(Z).astype(float)
    try:
        phi = np.linalg.solve(X, Y)
    except np.linalg.LinAlgError:
        phi = np.full((2, 2), np.nan, dtype=float)
    return phi


def _zdet_from_Z(Z: np.ndarray) -> complex:
    """Compute determinant impedance invariant Zdet = sqrt(det(Z))."""
    detZ = Z[0, 0] * Z[1, 1] - Z[0, 1] * Z[1, 0]
    return complex(np.sqrt(detZ))


def _zssq_from_Z(Z: np.ndarray) -> complex:
    """Compute SSQ impedance invariant Zssq = sqrt((sum(Zij^2))/2)."""
    ssq = (Z[0, 0] ** 2 + Z[0, 1] ** 2 + Z[1, 0] ** 2 + Z[1, 1] ** 2) / 2.0
    return complex(np.sqrt(ssq))


def _bootstrap_mt_derived(
    data: np.ndarray,
    error: np.ndarray,
    *,
    n_boot: int,
    rng: Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap MT-derived quantities using per-component standard deviations.

    Parameters
    ----------
    data : ndarray
        MT data array of shape (nfreq, 8) in re/im pairs order.
    error : ndarray
        MT error array of shape (nfreq, 8), interpreted as std deviations per numeric component.
    n_boot : int
        Number of bootstrap draws.
    rng : numpy.random.Generator
        RNG used for resampling.

    Returns
    -------
    phi_std : ndarray
        Standard deviation of phase tensor elements, shape (nfreq, 2, 2).
    zdet_std_re : ndarray
        Standard deviation of Re(Zdet), shape (nfreq,).
    zdet_std_im : ndarray
        Standard deviation of Im(Zdet), shape (nfreq,).
    zssq_std_reim : ndarray
        Standard deviations of Re/Im(Zssq), shape (nfreq, 2).

    Notes
    -----
    The implementation perturbs each real/imag component independently by Normal(0, σ).
    """
    nfreq = int(data.shape[0])
    phi_boot = np.full((n_boot, nfreq, 2, 2), np.nan, dtype=float)
    zdet_boot = np.full((n_boot, nfreq), np.nan, dtype=np.complex128)
    zssq_boot = np.full((n_boot, nfreq), np.nan, dtype=np.complex128)

    sigma = np.asarray(error, dtype=float)
    base = np.asarray(data, dtype=float)

    for b in range(n_boot):
        pert = base + rng.normal(loc=0.0, scale=sigma, size=base.shape)
        for i in range(nfreq):
            Z = _mt_data_to_tensor(pert[i, :])
            phi_boot[b, i, :, :] = _phase_tensor_from_Z(Z)
            zdet_boot[b, i] = _zdet_from_Z(Z)
            zssq_boot[b, i] = _zssq_from_Z(Z)

    phi_std = np.nanstd(phi_boot, axis=0, ddof=1)

    zdet_std_re = np.nanstd(np.real(zdet_boot), axis=0, ddof=1)
    zdet_std_im = np.nanstd(np.imag(zdet_boot), axis=0, ddof=1)

    zssq_std_re = np.nanstd(np.real(zssq_boot), axis=0, ddof=1)
    zssq_std_im = np.nanstd(np.imag(zssq_boot), axis=0, ddof=1)
    zssq_std_reim = np.stack([zssq_std_re, zssq_std_im], axis=1)

    return phi_std, zdet_std_re, zdet_std_im, zssq_std_reim


def _augment_mt_site(
    site: dict[str, Any],
    *,
    n_boot: int,
    rng: Generator,
) -> None:
    """Compute MT derived quantities and attach them to the site dict in-place.

    Parameters
    ----------
    site : dict
        Site dictionary with keys ``data`` and ``error`` for MT (shapes (nfreq,8)).
    n_boot : int
        Number of bootstrap draws for error estimates. If <= 0, errors are not computed
        (only the point estimates are attached).
    rng : numpy.random.Generator
        RNG used for bootstrap resampling.
    """
    data = np.asarray(site["data"], dtype=float)
    err = np.asarray(site["error"], dtype=float)
    nfreq = int(site["nfreq"])

    phi = np.full((nfreq, 2, 2), np.nan, dtype=float)
    zdet = np.full(nfreq, np.nan + 1j * np.nan, dtype=np.complex128)
    zssq = np.full(nfreq, np.nan + 1j * np.nan, dtype=np.complex128)

    for i in range(nfreq):
        Z = _mt_data_to_tensor(data[i, :])
        phi[i, :, :] = _phase_tensor_from_Z(Z)
        zdet[i] = _zdet_from_Z(Z)
        zssq[i] = _zssq_from_Z(Z)

    site["phase_tensor"] = phi
    site["Zdet"] = zdet
    site["Zssq"] = zssq

    if n_boot and n_boot > 0:
        phi_std, zdet_std_re, zdet_std_im, zssq_std_reim = _bootstrap_mt_derived(
            data,
            err,
            n_boot=int(n_boot),
            rng=rng,
        )
        site["phase_tensor_err"] = phi_std
        site["Zdet_err"] = zdet_std_re.astype(float) + 1j * zdet_std_im.astype(float)
        site["Zssq_err"] = zssq_std_reim[:, 0].astype(float) + 1j * zssq_std_reim[:, 1].astype(float)


def read_observe_dat(
    path: str | Path,
    *,
    compute_mt_derived: bool = True,
    bootstrap_n: int = 200,
    bootstrap_rng: Generator | None = None,
    bootstrap_seed: int | None = None,
) -> dict[str, Any]:
    """Parse a FEMTIC-style ``observe.dat`` into a structured nested dictionary.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the observation file.
    compute_mt_derived : bool
        If True, compute and attach MT derived quantities (phase tensor, Zdet, Zssq)
        to each MT site dictionary.
    bootstrap_n : int
        Number of bootstrap draws for MT derived-error estimates. If <= 0, errors are
        not computed (point estimates only).
    bootstrap_rng : numpy.random.Generator, optional
        RNG used for bootstrap resampling.  Preferred over ``bootstrap_seed`` when the
        caller manages the generator.  If both are None, ``default_rng(0)`` is used for
        reproducible error estimates.
    bootstrap_seed : int, optional
        Convenience integer seed for the bootstrap RNG.  Ignored when ``bootstrap_rng``
        is provided.  Pass an integer to override the default seed of 0.

    Returns
    -------
    dict
        Parsed structure with keys:

        - ``preamble_lines`` : list[str]
            Any lines before the first detected block header (kept verbatim).
        - ``blocks`` : list[dict]
            Each block dict contains:

            - ``obs_type`` : str
            - ``n_sites_header`` : int
            - ``header_line`` : str (verbatim)
            - ``sites`` : list[dict]

        - ``end_line`` : str or None
            The ``END`` line (verbatim) if present.
        - ``tail_lines`` : list[str]
            Lines after ``END`` (kept verbatim).

        Each site dict contains:

        - ``site_header_tokens`` : list[str]
        - ``site_header_line`` : str (verbatim)
        - ``nfreq`` : int
        - ``freq`` : ndarray, shape (nfreq,)
        - ``data`` : ndarray, shape (nfreq, dat_length)
        - ``error`` : ndarray, shape (nfreq, dat_length)
        - ``extras`` : list[list[float]]
            Any numeric columns after the expected ``1 + 2*dat_length`` values per row.
        - ``component_labels`` : list[str]

        If ``obs_type`` is MT and ``compute_mt_derived`` is True, additional keys are added:

        - ``phase_tensor`` : ndarray, shape (nfreq, 2, 2)
        - ``phase_tensor_err`` : ndarray, shape (nfreq, 2, 2)  (bootstrap std), if bootstrap_n > 0
        - ``Zdet`` : complex ndarray, shape (nfreq,)
        - ``Zdet_err`` : complex ndarray, shape (nfreq,)   (std(Re) + i*std(Im)), if bootstrap_n > 0
        - ``Zssq`` : complex ndarray, shape (nfreq,)
        - ``Zssq_err`` : complex ndarray, shape (nfreq,)   (std(Re) + i*std(Im)), if bootstrap_n > 0

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file cannot be parsed according to the expected layout.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    if bootstrap_rng is None:
        boot_rng = default_rng(0 if bootstrap_seed is None else int(bootstrap_seed))
    else:
        boot_rng = bootstrap_rng

    lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
    if not lines:
        raise ValueError(f"Empty file: {p}")

    end_idx = _find_end_index(lines)
    end_line = lines[end_idx] if end_idx < len(lines) else None
    tail_lines = lines[end_idx + 1 :] if end_idx + 1 < len(lines) else []

    # Detect block headers up to END.
    block_starts: list[int] = []
    for i in range(0, end_idx):
        toks = lines[i].split()
        if _is_block_header(toks):
            block_starts.append(i)

    if not block_starts:
        raise ValueError(f"No data blocks detected in {p}.")

    preamble_lines = lines[: block_starts[0]]

    # Append the END boundary as a sentinel for slicing.
    block_starts.append(end_idx)

    blocks: list[dict[str, Any]] = []
    for bi in range(len(block_starts) - 1):
        start = block_starts[bi]
        stop = block_starts[bi + 1]
        block_lines = lines[start:stop]

        hdr_toks = block_lines[0].split()
        if not _is_block_header(hdr_toks):
            raise ValueError(f"Expected block header at line {start + 1} in {p}")

        obs_type = hdr_toks[0]
        n_sites_header = int(hdr_toks[1])
        dat_length = _OBS_DATALEN[obs_type]
        component_labels = _COMPONENT_LABELS.get(obs_type, [f"c{i}" for i in range(dat_length)])

        sites: list[dict[str, Any]] = []
        li = 1  # cursor within block_lines (after header)
        while li < len(block_lines):
            toks = block_lines[li].split()

            # Skip blank lines inside block.
            if len(toks) == 0:
                li += 1
                continue

            # Site header must be exactly 4 tokens by convention here.
            if len(toks) != 4:
                raise ValueError(
                    f"Cannot parse site header at line {start + li + 1} in {p}: "
                    f"expected 4 tokens, got {len(toks)}."
                )

            site_header_line = block_lines[li]
            site_header_tokens = toks[:]
            if li + 1 >= len(block_lines):
                raise ValueError(f"Unexpected end of block after site header near line {start + li + 1} in {p}")

            try:
                nfreq = int(block_lines[li + 1].split()[0])
            except Exception as e:
                raise ValueError(
                    f"Cannot read number of frequencies after site header near line {start + li + 1} in {p}: {e}"
                ) from e

            data_start = li + 2
            data_stop = data_start + nfreq
            if data_stop > len(block_lines):
                raise ValueError(
                    f"Site near line {start + li + 1} in {p} claims {nfreq} frequencies, "
                    f"but block ends early."
                )

            freq = np.empty(nfreq, dtype=float)
            data = np.empty((nfreq, dat_length), dtype=float)
            error = np.empty((nfreq, dat_length), dtype=float)
            extras: list[list[float]] = []

            for ri, row_line in enumerate(block_lines[data_start:data_stop]):
                row_toks = row_line.split()
                try:
                    row = [float(x) for x in row_toks]
                except Exception as e:
                    raise ValueError(f"Non-numeric data row at line {start + data_start + ri + 1} in {p}: {e}") from e

                expected = 1 + 2 * dat_length
                if len(row) < expected:
                    raise ValueError(
                        f"Data row at line {start + data_start + ri + 1} has {len(row)} columns "
                        f"but expected at least {expected} (freq + {dat_length} data + {dat_length} error)."
                    )

                freq[ri] = row[0]
                data[ri, :] = row[1 : 1 + dat_length]
                error[ri, :] = row[1 + dat_length : 1 + 2 * dat_length]
                extras.append(row[1 + 2 * dat_length :])

            site_dict: dict[str, Any] = {
                "obs_type": obs_type,
                "site_header_tokens": site_header_tokens,
                "site_header_line": site_header_line,
                "nfreq": nfreq,
                "freq": freq,
                "data": data,
                "error": error,
                "extras": extras,
                "component_labels": component_labels,
                "nfreq_line": block_lines[li + 1],
            }

            if compute_mt_derived and obs_type == "MT":
                _augment_mt_site(site_dict, n_boot=int(bootstrap_n), rng=boot_rng)

            sites.append(site_dict)
            li = data_stop  # move to next site

        blocks.append(
            {
                "obs_type": obs_type,
                "n_sites_header": n_sites_header,
                "header_line": block_lines[0],
                "sites": sites,
            }
        )

    return {
        "path": str(p),
        "preamble_lines": preamble_lines,
        "blocks": blocks,
        "end_line": end_line,
        "tail_lines": tail_lines,
    }


def sites_as_dict_list(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a parsed observe.dat structure into a list of per-site dictionaries.

    Parameters
    ----------
    parsed : dict
        Output from :func:`read_observe_dat`.

    Returns
    -------
    list of dict
        A list where each element corresponds to one site section. Each dict contains
        (at least) keys: ``obs_type``, ``site_header_tokens``, ``nfreq``, ``freq``,
        ``data``, ``error``, and ``component_labels``.
    """
    sites: list[dict[str, Any]] = []
    for block in parsed.get("blocks", []):
        for site in block.get("sites", []):
            sites.append(site)
    return sites


def _site_header_to_meta(site_header_tokens: list[str]) -> dict[str, Any]:
    """
    Parse FEMTIC site header tokens (length 4) into a small metadata dict.

    Parameters
    ----------
    site_header_tokens : list of str
        Exactly 4 tokens as read from observe.dat.

    Returns
    -------
    meta : dict
        Contains:
        - name : str
        - xyz  : ndarray shape (3,) if tokens[1:4] parse as float, else None
        - raw_tokens : list[str]
    """
    name = str(site_header_tokens[0])
    xyz = None
    try:
        xyz = np.asarray([float(site_header_tokens[1]), float(site_header_tokens[2]), float(site_header_tokens[3])], dtype=float)
    except Exception:
        xyz = None
    return {"name": name, "xyz": xyz, "raw_tokens": site_header_tokens[:]}


def _mt_arrays_to_complex_tensors(
    data: np.ndarray,
    err: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert MT real/imag pair arrays to complex 2x2 tensors per frequency.

    Parameters
    ----------
    data : ndarray, shape (nfreq, 8)
        Columns: Zxx_re, Zxx_im, Zxy_re, Zxy_im, Zyx_re, Zyx_im, Zyy_re, Zyy_im
    err : ndarray, shape (nfreq, 8)
        Same ordering as data, standard deviations for each real/imag component.

    Returns
    -------
    Z : ndarray, shape (nfreq, 2, 2), complex128
        Complex impedance tensor per frequency.
    Zerr : ndarray, shape (nfreq, 2, 2), complex128
        Complex “error tensor” with std(Re) + i*std(Im) for each component.
    """
    d = np.asarray(data, dtype=float)
    e = np.asarray(err, dtype=float)
    if d.shape[1] != 8 or e.shape[1] != 8:
        raise ValueError("_mt_arrays_to_complex_tensors: expected (nfreq, 8) for MT data/error.")

    Zxx = d[:, 0] + 1j * d[:, 1]
    Zxy = d[:, 2] + 1j * d[:, 3]
    Zyx = d[:, 4] + 1j * d[:, 5]
    Zyy = d[:, 6] + 1j * d[:, 7]

    Z = np.empty((d.shape[0], 2, 2), dtype=np.complex128)
    Z[:, 0, 0] = Zxx
    Z[:, 0, 1] = Zxy
    Z[:, 1, 0] = Zyx
    Z[:, 1, 1] = Zyy

    Zxxe = e[:, 0] + 1j * e[:, 1]
    Zxye = e[:, 2] + 1j * e[:, 3]
    Zyxe = e[:, 4] + 1j * e[:, 5]
    Zyye = e[:, 6] + 1j * e[:, 7]

    Zerr = np.empty((e.shape[0], 2, 2), dtype=np.complex128)
    Zerr[:, 0, 0] = Zxxe
    Zerr[:, 0, 1] = Zxye
    Zerr[:, 1, 0] = Zyxe
    Zerr[:, 1, 1] = Zyye

    return Z, Zerr


def _mt_rhoa_phase_from_Z(
    freq: np.ndarray,
    Z: np.ndarray,
    Zerr: np.ndarray | None = None,
    *,
    n_mc: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Compute apparent resistivity and phase from complex impedance tensors.

    Uses:
        rhoa = |Z|^2 / (mu0 * omega),  omega = 2*pi*f
        phase = atan2(Im(Z), Re(Z)) in degrees

    If Zerr is provided, uncertainty is estimated by Monte Carlo assuming
    independent Gaussian errors in Re and Im with std given by Zerr.

    Parameters
    ----------
    freq : ndarray, shape (nfreq,)
        Frequencies in Hz.
    Z : ndarray, shape (nfreq, 2, 2)
        Complex impedance tensor.
    Zerr : ndarray, shape (nfreq, 2, 2), optional
        Complex error tensor std(Re) + i*std(Im) per component.
    n_mc : int
        Number of Monte Carlo draws (per frequency) for derived error estimates.
    seed : int
        RNG seed.

    Returns
    -------
    out : dict
        Keys:
        - rhoa : ndarray (nfreq, 2, 2)
        - phase_deg : ndarray (nfreq, 2, 2)
        - rhoa_err : ndarray (nfreq, 2, 2) or None
        - phase_err_deg : ndarray (nfreq, 2, 2) or None
    """
    f = np.asarray(freq, dtype=float).ravel()
    if f.ndim != 1:
        raise ValueError("_mt_rhoa_phase_from_Z: freq must be 1D.")
    if Z.shape[0] != f.size:
        raise ValueError("_mt_rhoa_phase_from_Z: Z and freq size mismatch.")

    mu0 = 4.0e-7 * np.pi
    omega = 2.0 * np.pi * f
    omega = omega[:, None, None]

    absZ2 = np.abs(Z) ** 2
    rhoa = absZ2 / (mu0 * omega)
    phase_deg = np.degrees(np.arctan2(Z.imag, Z.real))

    rhoa_err = None
    phase_err_deg = None

    if Zerr is not None and n_mc > 0:
        rng = default_rng(int(seed))
        sig_re = np.asarray(Zerr.real, dtype=float)
        sig_im = np.asarray(Zerr.imag, dtype=float)

        rhoa_samp = np.empty((n_mc, f.size, 2, 2), dtype=float)
        ph_samp = np.empty((n_mc, f.size, 2, 2), dtype=float)

        # vectorized MC: draw Re/Im perturbations
        for j in range(n_mc):
            d_re = rng.normal(loc=0.0, scale=sig_re)
            d_im = rng.normal(loc=0.0, scale=sig_im)
            Zj = (Z.real + d_re) + 1j * (Z.imag + d_im)

            rhoa_samp[j, ...] = (np.abs(Zj) ** 2) / (mu0 * omega)
            ph_samp[j, ...] = np.degrees(np.arctan2(Zj.imag, Zj.real))

        rhoa_err = rhoa_samp.std(axis=0, ddof=1)
        phase_err_deg = ph_samp.std(axis=0, ddof=1)

    return {
        "rhoa": rhoa,
        "phase_deg": phase_deg,
        "rhoa_err": rhoa_err,
        "phase_err_deg": phase_err_deg,
    }


def observe_to_site_viz_list(
    observe_path: str | Path,
    *,
    obs_type: Literal["MT", "VTF", "PT"] = "MT",
    compute_mt_derived: bool = True,
    bootstrap_n: int = 200,
    bootstrap_seed: int = 0,
    add_rhoa_phase: bool = True,
    mc_n: int = 200,
    mc_seed: int = 0,
) -> list[dict[str, Any]]:
    """Read observe.dat and return a per-site list of plot-ready dictionaries.

    This is the most convenient entry point for visualisation routines that loop
    over sites and plot curves vs period / frequency.

    Parameters
    ----------
    observe_path : str or pathlib.Path
        Path to ``observe.dat``.
    obs_type : {"MT", "VTF", "PT"}
        Which block type to return.  Sites belonging to other block types in the
        same file are silently skipped.
    compute_mt_derived : bool
        If True, the parser computes and attaches MT-derived fields (phase
        tensor, Zdet, Zssq) — including bootstrap error estimates when
        ``bootstrap_n > 0``.
    bootstrap_n : int
        Number of bootstrap draws for MT-derived error estimates (passed to
        :func:`read_observe_dat`).  Set to 0 to skip error computation.
    bootstrap_seed : int
        Integer seed for the parser's bootstrap RNG.
    add_rhoa_phase : bool
        If True **and** ``obs_type == "MT"``, compute apparent resistivity
        (``rhoa``) and impedance phase (``phase_deg``) for each site via
        :func:`_mt_rhoa_phase_from_Z`, including Monte Carlo uncertainty
        estimates when ``mc_n > 0``.
    mc_n : int
        Number of Monte Carlo draws for rhoa / phase error propagation.
        Set to 0 to skip error computation.
    mc_seed : int
        Integer seed for the Monte Carlo RNG used in rhoa / phase error
        propagation.

    Returns
    -------
    sites_viz : list of dict
        One dict per site of type ``obs_type``.  Every dict contains:

        Common fields (all obs types)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        - ``obs_type`` : str — ``"MT"``, ``"VTF"``, or ``"PT"``
        - ``name`` : str — site name from the header
        - ``xyz``  : ndarray shape (3,) or None — site coordinates
        - ``nfreq`` : int — number of frequencies
        - ``freq``  : ndarray (nfreq,) — frequencies in Hz
        - ``per``   : ndarray (nfreq,) — periods in s (= 1 / freq)
        - ``data``  : ndarray (nfreq, dat_length) — raw observed values
        - ``error`` : ndarray (nfreq, dat_length) — raw per-component std devs
        - ``component_labels`` : list[str]

        MT-specific fields (when ``obs_type == "MT"``)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        - ``Z``    : ndarray (nfreq, 2, 2), complex — impedance tensors
        - ``Zerr`` : ndarray (nfreq, 2, 2), complex — std(Re)+i*std(Im) per component

        MT fields added when ``add_rhoa_phase=True``
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        - ``rhoa``          : ndarray (nfreq, 2, 2) — apparent resistivity (Ω m)
        - ``phase_deg``     : ndarray (nfreq, 2, 2) — impedance phase (°)
        - ``rhoa_err``      : ndarray (nfreq, 2, 2) or None — MC std of rhoa
        - ``phase_err_deg`` : ndarray (nfreq, 2, 2) or None — MC std of phase

        MT-derived fields added by the parser when ``compute_mt_derived=True``
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        - ``phase_tensor``     : ndarray (nfreq, 2, 2)
        - ``phase_tensor_err`` : ndarray (nfreq, 2, 2) — bootstrap std (if bootstrap_n > 0)
        - ``Zdet``             : complex ndarray (nfreq,)
        - ``Zdet_err``         : complex ndarray (nfreq,) — std(Re)+i*std(Im) (if bootstrap_n > 0)
        - ``Zssq``             : complex ndarray (nfreq,)
        - ``Zssq_err``         : complex ndarray (nfreq,) — std(Re)+i*std(Im) (if bootstrap_n > 0)

    Raises
    ------
    FileNotFoundError
        If ``observe_path`` does not exist.
    ValueError
        If the file cannot be parsed or contains no block of the requested type.
    """
    obs_type = str(obs_type).upper()
    if obs_type not in _OBS_DATALEN:
        raise ValueError(
            f"observe_to_site_viz_list: obs_type={obs_type!r} not supported. "
            f"Choose from {list(_OBS_DATALEN)}."
        )

    parsed = read_observe_dat(
        observe_path,
        compute_mt_derived=bool(compute_mt_derived),
        bootstrap_n=int(bootstrap_n),
        bootstrap_seed=int(bootstrap_seed),
    )

    sites_viz: list[dict[str, Any]] = []

    for site in sites_as_dict_list(parsed):
        # Filter to the requested block type.
        if str(site.get("obs_type", "")).upper() != obs_type:
            continue

        # ---- meta --------------------------------------------------------
        meta = _site_header_to_meta(site["site_header_tokens"])

        freq = np.asarray(site["freq"], dtype=float)
        # Guard against zero or negative frequencies (should not occur in
        # well-formed files, but avoids division-by-zero in period).
        with np.errstate(divide="ignore", invalid="ignore"):
            per = np.where(freq > 0.0, 1.0 / freq, np.nan)

        entry: dict[str, Any] = {
            "obs_type": obs_type,
            "name": meta["name"],
            "xyz": meta["xyz"],
            "nfreq": int(site["nfreq"]),
            "freq": freq,
            "per": per,
            "data": np.asarray(site["data"], dtype=float),
            "error": np.asarray(site["error"], dtype=float),
            "component_labels": list(site.get("component_labels", [])),
        }

        # ---- MT-specific processing --------------------------------------
        if obs_type == "MT":
            data = np.asarray(site["data"], dtype=float)
            err = np.asarray(site["error"], dtype=float)

            Z, Zerr = _mt_arrays_to_complex_tensors(data, err)
            entry["Z"] = Z
            entry["Zerr"] = Zerr

            # Propagate parser-attached MT-derived fields verbatim.
            for _key in (
                "phase_tensor", "phase_tensor_err",
                "Zdet", "Zdet_err",
                "Zssq", "Zssq_err",
            ):
                if _key in site:
                    entry[_key] = site[_key]

            # rhoa / phase (+ MC errors)
            if add_rhoa_phase:
                _rp = _mt_rhoa_phase_from_Z(
                    freq,
                    Z,
                    Zerr if mc_n > 0 else None,
                    n_mc=int(mc_n),
                    seed=int(mc_seed),
                )
                entry["rhoa"] = _rp["rhoa"]
                entry["phase_deg"] = _rp["phase_deg"]
                entry["rhoa_err"] = _rp["rhoa_err"]          # None if mc_n == 0
                entry["phase_err_deg"] = _rp["phase_err_deg"]  # None if mc_n == 0

        sites_viz.append(entry)

    return sites_viz


def write_observe_dat(parsed: dict[str, Any], path: str | Path) -> None:
    """Write a parsed (and possibly modified) observe.dat structure to disk.

    Parameters
    ----------
    parsed : dict
        Parsed structure produced by :func:`read_observe_dat` (potentially modified).
    path : str or pathlib.Path
        Output path to write.

    Notes
    -----
    - Preamble lines, header lines, site header lines, the END line, and tail lines are
      written back verbatim where available.
    - Data rows are rewritten in a clear numeric layout: frequency first, then values,
      then errors, and finally any extra numeric columns if present.
    - Floats are formatted in scientific notation with 8 digits after the decimal.
    """
    out = Path(path)

    out_lines: list[str] = []
    out_lines.extend(parsed.get("preamble_lines", []))

    blocks = parsed.get("blocks", [])
    for block in blocks:
        obs_type = str(block["obs_type"])
        dat_length = _OBS_DATALEN.get(obs_type)
        if dat_length is None:
            raise NotImplementedError(f"write_observe_dat: obs_type={obs_type!r} not supported")

        # Keep original header line if present; otherwise create a minimal one.
        header_line = block.get("header_line")
        if header_line is None:
            n_sites = len(block.get("sites", []))
            header_line = f"{obs_type}    {n_sites}\n"
        if not header_line.endswith("\n"):
            header_line = header_line + "\n"
        out_lines.append(header_line)

        for site in block.get("sites", []):
            shl = site.get("site_header_line", "")
            if not shl.endswith("\n"):
                shl = shl + "\n"
            out_lines.append(shl)

            nfl = site.get("nfreq_line")
            if nfl is None:
                nfl = f"{int(site['nfreq'])}\n"
            if not nfl.endswith("\n"):
                nfl = nfl + "\n"
            out_lines.append(nfl)

            freq = np.asarray(site["freq"], dtype=float).ravel()
            data = np.asarray(site["data"], dtype=float)
            err = np.asarray(site["error"], dtype=float)
            extras = site.get("extras", [])

            nfreq = int(site["nfreq"])
            if freq.size != nfreq or data.shape[0] != nfreq or err.shape[0] != nfreq:
                raise ValueError("write_observe_dat: inconsistent nfreq vs array shapes in site data")

            for i in range(nfreq):
                row_vals: list[float] = [float(freq[i])]
                row_vals.extend([float(x) for x in data[i, :].tolist()])
                row_vals.extend([float(x) for x in err[i, :].tolist()])

                extra_row = extras[i] if i < len(extras) else []
                row_vals.extend([float(x) for x in extra_row])

                out_lines.append(_format_row(row_vals))

    end_line = parsed.get("end_line")
    if end_line is not None:
        if not end_line.endswith("\n"):
            end_line = end_line + "\n"
        out_lines.append(end_line)

    out_lines.extend(parsed.get("tail_lines", []))

    out.write_text("".join(out_lines), encoding="utf-8")


def modify_data(
    template_file: str | Path = "observe.dat",
    *,
    errors: Sequence[Sequence[float] | np.ndarray] = ([], [], []),
    draw_from: Sequence[float | str] = ("normal", 0.0, 1.0),
    scalfac: float = 1.0,
    method: str | None = None,
    rng: Generator | None = None,
    seed: int | None = None,
    out: bool = True,
    return_sites: bool = False,
    compute_mt_derived: bool = True,
    bootstrap_n: int = 200,
    bootstrap_rng: Generator | None = None,
    bootstrap_seed: int | None = None,
) -> list[dict[str, Any]] | None:
    """Perturb a FEMTIC-style observation file (``observe.dat``) and rewrite it in-place.

    For each datum ``d`` with associated standard deviation ``σ`` the perturbation is:

        ``d_new = Normal(d, σ)``

    If *relative* errors are provided via ``errors`` for the current observation type,
    the error columns are overwritten before drawing perturbations:

        ``σ := abs(d) * rel_error``

    If ``scalfac`` is provided (> 1.0), the effective error is scaled **before** drawing:

        ``σ_eff := σ * scalfac``

    This mirrors the behaviour of :func:`modify_data_fcn` and is useful for inflating
    uncertainties prior to ensemble generation.

    Parameters
    ----------
    template_file : str or pathlib.Path
        Path to the observation file. The file is rewritten **in-place**.
    errors : sequence of length 3
        ``[errors_MT, errors_VTF, errors_PT]``. Each element can be:

        - [] (empty): keep existing per-datum error columns
        - [scalar]: broadcast to all components
        - [vector]: component-wise relative errors (length MT=8, VTF/PT=4)
    draw_from : sequence
        Noise distribution spec.  The first element names the distribution;
        subsequent elements are parameters.  Supported distributions:

        - ``("normal", loc, scale)`` — Gaussian; ``loc`` is an additive offset
          applied to the datum value (normally 0.0) and ``scale`` multiplies the
          per-datum sigma (normally 1.0).  The effective draw is:
          ``Normal(val + loc, σ_eff * scale)``.
        - ``("uniform", a, b)`` — Uniform perturbation scaled by ``σ_eff``:
          ``val + Uniform(a, b) * σ_eff``.

        The default ``("normal", 0.0, 1.0)`` reproduces the original behaviour.
    scalfac : float
        Multiplicative scale factor applied to each per-datum sigma **before**
        drawing.  The effective standard deviation is ``σ_eff = σ * scalfac``.
        Defaults to 1.0 (no scaling).
    method : str, optional
        Deprecated.  Accepted for backward compatibility with older callers (e.g.
        ``ensembles.generate_data_ensemble``) but has no effect.  Use ``draw_from``
        to select the perturbation distribution.
    rng : numpy.random.Generator, optional
        Random number generator used for perturbations.  Preferred over
        ``seed`` when the caller manages the generator.  If both are None a
        fresh (non-reproducible) generator is created.
    seed : int, optional
        Convenience integer seed.  Ignored when ``rng`` is provided.
    out : bool
        If True, print basic status messages about detected blocks and sites.
    return_sites : bool
        If True, return a flattened list of per-site dictionaries (see
        :func:`sites_as_dict_list`). If False, return None.
    compute_mt_derived : bool
        If True, compute and attach MT derived quantities (phase tensor, Zdet, Zssq)
        to each MT site dictionary after perturbation.
    bootstrap_n : int
        Number of bootstrap draws for MT derived-error estimates. If <= 0, errors are
        not computed (point estimates only).
    bootstrap_rng : numpy.random.Generator, optional
        RNG used for bootstrap resampling.  If None, ``default_rng(bootstrap_seed)``
        is used (or a seeded-0 generator when ``bootstrap_seed`` is also None).
    bootstrap_seed : int, optional
        Convenience seed for the bootstrap RNG.  Ignored when ``bootstrap_rng``
        is provided.

    Returns
    -------
    list of dict or None
        If ``return_sites`` is True, returns a flattened list of per-site dictionaries
        after perturbation. Otherwise returns None.
    """
    # ---- RNG setup -------------------------------------------------------
    rng = default_rng(seed) if rng is None else rng
    if bootstrap_rng is None:
        boot_rng = default_rng(0 if bootstrap_seed is None else int(bootstrap_seed))
    else:
        boot_rng = bootstrap_rng

    # ---- backward-compat: 'method' param (accepted but unused) -----------
    if method is not None:
        import warnings
        warnings.warn(
            "modify_data: the 'method' parameter is deprecated and has no effect. "
            "Use 'draw_from' to select the perturbation distribution.",
            DeprecationWarning,
            stacklevel=2,
        )

    # ---- parse draw_from -------------------------------------------------
    draw_seq = list(draw_from)
    dist_name = str(draw_seq[0]).lower() if draw_seq else "normal"
    if dist_name not in ("normal", "uniform"):
        raise ValueError(
            f"modify_data: unsupported draw_from distribution {draw_seq[0]!r}. "
            "Use 'normal' or 'uniform'."
        )
    # For normal: optional (loc_offset, scale_factor) after distribution name.
    _loc_offset = float(draw_seq[1]) if len(draw_seq) > 1 else 0.0
    _scale_factor = float(draw_seq[2]) if len(draw_seq) > 2 else 1.0
    # For uniform: (a, b) multipliers of σ_eff.
    _unif_a = float(draw_seq[1]) if len(draw_seq) > 1 else -1.0
    _unif_b = float(draw_seq[2]) if len(draw_seq) > 2 else 1.0

    parsed = read_observe_dat(
        template_file,
        compute_mt_derived=False,  # recompute after perturbation
        bootstrap_n=0,
        bootstrap_rng=boot_rng,
    )
    blocks = parsed["blocks"]

    if out:
        print(f"Detected {len(blocks)} data block(s) in {parsed['path']}.")

    for bi, block in enumerate(blocks):
        obs_type = str(block["obs_type"])
        dat_length = _OBS_DATALEN.get(obs_type)
        if dat_length is None:
            raise NotImplementedError(f"modify_data: obs_type={obs_type!r} not supported")

        # Select relative errors for this block type (if given).
        rel: np.ndarray | None
        if obs_type == "MT":
            rel = _rel_err_array(errors[0], dat_length)
        elif obs_type == "VTF":
            rel = _rel_err_array(errors[1], dat_length)
        elif obs_type == "PT":
            rel = _rel_err_array(errors[2], dat_length)
        else:
            rel = None

        if out:
            print(f"Block {bi}: {obs_type} (sites: {len(block.get('sites', []))})")

        for si, site in enumerate(block.get("sites", [])):
            data = np.asarray(site["data"], dtype=float)
            err = np.asarray(site["error"], dtype=float)

            if data.shape != err.shape or data.shape[1] != dat_length:
                raise ValueError("modify_data: inconsistent data/error shapes in site")

            nfreq = int(site["nfreq"])
            if out:
                msg = f"  Site {si}: nfreq={nfreq}"
                if rel is not None:
                    msg += " (overwriting errors from relative errors)"
                print(msg)

            # Overwrite errors (optional) and draw perturbations component-wise.
            for i in range(nfreq):
                for k in range(dat_length):
                    val = float(data[i, k])

                    if rel is not None:
                        sigma = float(abs(val) * float(rel[k]))
                        err[i, k] = sigma
                    else:
                        sigma = float(err[i, k])

                    sigma_eff = sigma * float(scalfac)

                    if not np.isfinite(sigma_eff) or sigma_eff <= 0.0:
                        continue

                    if dist_name == "normal":
                        data[i, k] = float(
                            rng.normal(loc=val + _loc_offset, scale=sigma_eff * _scale_factor)
                        )
                    else:  # uniform
                        data[i, k] = float(val + rng.uniform(_unif_a, _unif_b) * sigma_eff)

            site["data"] = data
            site["error"] = err

            if compute_mt_derived and obs_type == "MT":
                _augment_mt_site(site, n_boot=int(bootstrap_n), rng=boot_rng)

    # Rewrite in-place (frequency first, values then errors).
    write_observe_dat(parsed, template_file)

    if out:
        print(f"File {template_file} successfully written.")

    if return_sites:
        return sites_as_dict_list(parsed)
    return None

def convert_observe_dat(
    in_path: str | Path,
    out_path: str | Path,
    *,
    compute_mt_derived: bool = True,
    bootstrap_n: int = 200,
    bootstrap_seed: int = 0,
    sites_out: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Convert an ``observe.dat`` file to a rewritten output file and return site dicts.

    Parameters
    ----------
    in_path : str or pathlib.Path
        Input observe.dat path.
    out_path : str or pathlib.Path
        Output path to write. The file is written in the same clear numeric layout used
        by :func:`write_observe_dat` (frequency first, then values, then errors).
    compute_mt_derived : bool
        If True, compute and attach MT derived quantities (phase tensor, Zdet, Zssq)
        to each MT site dictionary.
    bootstrap_n : int
        Number of bootstrap draws for MT derived-error estimates. If <= 0, errors are
        not computed (point estimates only).
    bootstrap_seed : int
        Seed for the bootstrap RNG (for reproducibility).
    sites_out : str or pathlib.Path, optional
        If given, save the flattened list of per-site dictionaries. Supported formats:

        - ``.npz`` : saved as a compressed NumPy object array (key: ``sites``)
        - ``.pkl`` : saved via pickle

    Returns
    -------
    list of dict
        Flattened list of per-site dictionaries (EDI-like container).
    """
    boot_rng = default_rng(int(bootstrap_seed))
    parsed = read_observe_dat(
        in_path,
        compute_mt_derived=bool(compute_mt_derived),
        bootstrap_n=int(bootstrap_n),
        bootstrap_rng=boot_rng,
    )
    write_observe_dat(parsed, out_path)
    sites = sites_as_dict_list(parsed)

    if sites_out is not None:
        _save_sites(sites, sites_out)

    return sites


def _save_sites(sites: list[dict[str, Any]], sites_out: str | Path) -> None:
    """Save a site list to disk in either NPZ (object array) or pickle format.

    Parameters
    ----------
    sites : list of dict
        Flattened per-site dictionaries (possibly containing numpy arrays and complex).
    sites_out : str or pathlib.Path
        Output filename. If suffix is ``.npz``, saves an object array under key ``sites``.
        If suffix is ``.pkl``, saves via pickle.

    Raises
    ------
    ValueError
        If the file suffix is not supported.
    """
    out = Path(sites_out)
    suf = out.suffix.lower()
    if suf == ".npz":
        np.savez_compressed(out, sites=np.asarray(sites, dtype=object))
        return
    if suf == ".pkl":
        import pickle

        with out.open("wb") as f:
            pickle.dump(sites, f, protocol=pickle.HIGHEST_PROTOCOL)
        return
    raise ValueError(f"Unsupported sites_out format {out.suffix!r}; use .npz or .pkl")

# ============================================================================
# SECTION 7: CLI wrapper for unified module
# ============================================================================


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    CLI entry point for femtic.py with subcommands:

    - femtic-to-npz
    - npz-to-vtk
    - npz-to-femtic
    """
    import argparse

    ap = argparse.ArgumentParser(
        description="Unified FEMTIC utilities: ensembles, mesh↔NPZ↔VTK/FEMTIC."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # femtic-to-npz
    p_m2n = sub.add_parser(
        "femtic-to-npz",
        help="Build FEMTIC mesh + region + element NPZ from mesh.dat and block.",
    )
    p_m2n.add_argument("--mesh", required=True, help="Path to FEMTIC mesh.dat.")
    p_m2n.add_argument(
        "--rho-block",
        required=True,
        help="Path to resistivity_block_iterX.dat.",
    )
    p_m2n.add_argument("--out-npz", required=True, help="Output NPZ file.")

    # npz-to-vtk
    p_n2v = sub.add_parser(
        "npz-to-vtk",
        help="Export FEMTIC NPZ model to VTK/VTU unstructured grid.",
    )
    p_n2v.add_argument("--npz", required=True, help="Input NPZ file.")
    p_n2v.add_argument("--out-vtu", required=True, help="Output VTU file.")
    p_n2v.add_argument("--out-legacy", default=None, help="Optional legacy .vtk file.")
    p_n2v.add_argument(
        "--scalar",
        default="log10_resistivity",
        help="Cell scalar name for plotting (default 'log10_resistivity').",
    )

    # hdf5-to-npz
    p_h2n = sub.add_parser(
        "hdf5-to-npz",
        help="Convert FEMTIC-style HDF5 (written by npz-to-hdf5) back to NPZ.",
    )
    p_h2n.add_argument("--hdf5", required=True, help="Input HDF5 file.")
    p_h2n.add_argument("--out-npz", required=True, help="Output NPZ file.")
    p_h2n.add_argument(
        "--group",
        default="femtic_model",
        help="HDF5 group containing the FEMTIC datasets (default 'femtic_model').",
    )

    # netcdf-to-npz
    p_nc2n = sub.add_parser(
        "netcdf-to-npz",
        help="Convert FEMTIC-style NetCDF (written by npz-to-netcdf) back to NPZ.",
    )
    p_nc2n.add_argument("--netcdf", required=True, help="Input NetCDF file.")
    p_nc2n.add_argument("--out-npz", required=True, help="Output NPZ file.")

    # npz-to-femtic
    p_n2f = sub.add_parser(
        "npz-to-femtic",
        help="Recreate FEMTIC mesh.dat and resistivity_block from NPZ.",
    )
    p_n2f.add_argument("--npz", required=True, help="Input NPZ file.")
    p_n2f.add_argument("--mesh-out", required=True, help="Output mesh.dat.")
    p_n2f.add_argument(
        "--rho-block-out",
        required=True,
        help="Output resistivity_block_iterX.dat.",
    )
    p_n2f.add_argument(
        "--format",
        dest="fmt",
        default="{:.6g}",
        help='Float format for resistivity and bounds (default "{:.6g}").',
    )

    # npz-ellipsoid-fill
    p_ell = sub.add_parser(
        "npz-ellipsoid-fill",
        help="Create a new region and fill an ellipsoid (by element centroids) in an element NPZ model.",
    )
    p_ell.add_argument("npz_in", help="Input element NPZ (from mesh-and-block-to-npz).")
    p_ell.add_argument("npz_out", help="Output element NPZ.")
    p_ell.add_argument(
        "--center",
        nargs=3,
        type=float,
        required=True,
        metavar=("CX", "CY", "CZ"),
        help="Ellipsoid center (cx cy cz) in model coordinates.",
    )
    p_ell.add_argument(
        "--axes",
        nargs=3,
        type=float,
        required=True,
        metavar=("A", "B", "C"),
        help="Ellipsoid semi-axes (a b c), must be > 0.",
    )
    p_ell.add_argument(
        "--angles",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("A1", "A2", "A3"),
        help="Rotation angles in degrees. Interpretation depends on --angle-convention.",
    )

    p_ell.add_argument(
        "--angle-convention",
        choices=["zyx", "sds"],
        default="zyx",
        help=(
            "Angle convention for --angles: "
            "'zyx' = intrinsic Z-Y-X (yaw, pitch, roll); "
            "'sds' = strike/dip/slant (Rz(strike)@Rx(dip)@Rz(slant))."
        ),
    )
    p_ell.add_argument(
        "--fill",
        type=float,
        required=True,
        help="Fill value (interpretation depends on --fill-space).",
    )
    p_ell.add_argument(
        "--fill-space",
        choices=["log10", "rho"],
        default="log10",
        help="Interpretation of --fill: 'log10' (default) or 'rho' (Ohm m).",
    )
    p_ell.add_argument(
        "--fill-flag",
        type=int,
        default=0,
        choices=[0, 1],
        help="Flag for the new region: 0=free (default), 1=fixed.",
    )
    p_ell.add_argument(
        "--fill-bounds",
        nargs=2,
        type=float,
        default=None,
        metavar=("RHO_LO", "RHO_HI"),
        help="Optional bounds for the new region (rho_lower rho_upper) in Ohm m.",
    )
    p_ell.add_argument(
        "--fill-n",
        type=float,
        default=1.0,
        help="Optional 'n' value for the new region (default 1.0).",
    )
    p_ell.add_argument(
        "--no-respect-fixed",
        action="store_true",
        help="If set, also modify elements in fixed regions (NOT recommended).",
    )

    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.cmd == "femtic-to-npz":
        mesh_and_block_to_npz(args.mesh, args.rho_block, args.out_npz)
        print("Saved mesh + region + element NPZ:", args.out_npz)
        return 0

    if args.cmd == "npz-to-vtk":
        save_vtk_from_npz(
            npz_path=args.npz,
            out_vtu=args.out_vtu,
            out_legacy=args.out_legacy,
            scalar_name=args.scalar,
        )
        return 0

    if args.cmd == "hdf5-to-npz":
        hdf5_to_npz(
            hdf5_path=args.hdf5,
            npz_path=args.out_npz,
            group=args.group,
        )
        print("Wrote NPZ:", args.out_npz)
        return 0

    if args.cmd == "netcdf-to-npz":
        netcdf_to_npz(
            netcdf_path=args.netcdf,
            npz_path=args.out_npz,
        )
        print("Wrote NPZ:", args.out_npz)
        return 0

    if args.cmd == "npz-to-femtic":
        npz_to_femtic(
            npz_path=args.npz,
            mesh_out=args.mesh_out,
            rho_block_out=args.rho_block_out,
            fmt=args.fmt,
        )
        print("Wrote mesh to:", args.mesh_out)
        print("Wrote resistivity block to:", args.rho_block_out)
        return 0

    if args.cmd == "npz-ellipsoid-fill":
        ellipsoid_fill_element_npz(
            args.npz_in,
            args.npz_out,
            center=args.center,
            axes=args.axes,
            angles_deg=args.angles,
            angle_convention=args.angle_convention,
            fill_value=args.fill,
            fill_space=args.fill_space,
            fill_flag=args.fill_flag,
            fill_bounds=tuple(args.fill_bounds) if args.fill_bounds is not None else None,
            fill_n=args.fill_n,
            respect_fixed=not args.no_respect_fixed,
            out=True,
        )
        print("Wrote NPZ:", args.npz_out)
        return 0

    ap.error(f"Unknown command {args.cmd!r}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
