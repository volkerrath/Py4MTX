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
- **results_iterX.h5 readers** — read_results_hdf5() and its sub-readers for
  FEMTIC's combined model+data+distortion HDF5 output (Section 6a).
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

    python femtic.py edi-to-observe \\
        SITE01.edi SITE02.edi \\
        --xy-csv positions.csv \\
        --out observe.dat

All functionality is also available as regular Python functions.

Author: Volker Rath (DIAS)
Created with the help of ChatGPT (GPT-5 Thinking) on 2026-01-02 (UTC)
Modified: 2026-04-11 by Claude Sonnet 4.6 (Anthropic) — Section 2
    (matrix/roughness tools) now imported from ensembles.py rather than
    duplicated; femtic.py restricted to FEMTIC-specific I/O and conversion.
Modified: 2026-04-13 by Claude Sonnet 4.6 (Anthropic) — z-convention
    documentation and consistency fixes across observe.dat interfaces;
    added edi_list_to_observe_dat() and edi-to-observe CLI subcommand.
Modified: 2026-05-23 by Claude Sonnet 4.6 (Anthropic) — added model-local
    coordinate helpers: utm_to_model(), latlon_to_model(), parse_pos_crs(),
    resolve_pos_x(), resolve_pos_y(), resolve_pos_point(),
    resolve_slice_positions(); delegates geographic conversions to util.py.
Modified: 2026-05-24 by Claude Sonnet 4.6 (Anthropic) — added Section 6b:
    read_site_position(), read_site_dat(), estimate_utm_origin(),
    _point_in_tet(), extract_borehole_log(); moved from femtic_mod_plot.py.
Modified: 2026-05-28 by Claude Sonnet 4.6 (Anthropic) — fixed insert_model:
    template is now read fully into memory before the output file is opened,
    so in-place rewrites (template == model_file, e.g. referencemodel.dat)
    no longer truncate the template before it can be parsed.
Modified: 2026-06-08 by Claude Sonnet 4.6 (Anthropic) — promoted three
    geometry helpers from femtic_mod_edit.py into the mesh/geometry section:
    tet_volumes() (vectorised tetrahedral volume via scalar triple product),
    build_region_geometry() (volume-weighted centroids + volumes per free
    region), and brick_mask() (rotated rectangular prism containment test,
    parallel to ellipsoid_mask() with the same convention/keyword-only
    signature).
Modified: 2026-06-19 by Claude Sonnet 4.6 (Anthropic) — added
    resolve_pos_two_point_profile() helper and kind="profile" branch in
    resolve_slice_positions(): a vertical fence section is defined by two
    geographic/UTM/model-local endpoint positions; strike is derived
    automatically and the spec is converted to kind="plane" with dip=90.
Modified: 2026-06-10 by Claude Sonnet 4.6 (Anthropic) — added two summary
    helpers: summarise_model_file() counts air / ocean / other-fixed /
    parameter elements in a resistivity_block_iterXX.dat, reusing the
    existing _parse_region_line / _infer_ocean_present infrastructure;
    _print_model_summary() formats the table.  summarise_observe_dat()
    reports sites / frequencies / data-vector size per obs-type block from
    an observe.dat (accepts path or pre-parsed dict); _print_observe_summary()
    formats the table.  Both functions are also exposed as standalone CLI
    scripts: femtic_summarize_model_cells.py and
    femtic_summarize_observe_dat.py.
Modified: 2026-07-17 by Claude Sonnet 5 (Anthropic) — migrated from legacy
    scipy.sparse matrix classes to the array-equivalent API: removed unused
    isspmatrix import (issparse covers both matrix and array sparse types),
    and updated the scipy.sparse.spmatrix type hint to scipy.sparse.sparray
    in sample_precision_gaussian_gmrf(). No functional change; module
    already used csr_array/csc_array/coo_array internally.
Modified: 2026-09-08 by Claude Sonnet 5 (Anthropic) — added two functions
    supporting ensemble-level sensitivity diagnostics (mitigating the
    "low ensemble spread means well-resolved OR null-space" ambiguity in
    RTO/GST ensembles): read_h5_sensitivity() reads or derives a
    per-free-parameter cumulative sensitivity vector from a (schema-
    provisional) model_iterX.h5 file, either from a precomputed dataset
    or by summing |Jacobian| columns (optionally error-weighted), cf.
    Christiansen & Auken (2012), Geophysics 77, WB171,
    doi:10.1190/geo2011-0393.1; collect_forward_response() flattens one
    ensemble member's calculated ("cal") forward response across MT/VTF/
    PT result_XXX.txt files into a single vector, for building a
    per-member forward-response ensemble used by the "SimRC" ensemble-
    native sensitivity measure of Bobe, Keller & Van De Vijver (2021),
    Geophysical Prospecting, doi:10.1111/1365-2478.13068. Both consumed
    by femtic_ens_post.py's new COMPUTE_SENS/COMPUTE_SIMRC pipeline; see
    femtic_ens_post_readme.md.
Modified: 2026-09-11 by Claude Sonnet 5 (Anthropic) — added
    convert_resistivity_block_to_anisotropic(), a helper that converts a
    v4 (isotropic-only) resistivity_block_iterX.dat into v5 (anisotropic)
    format for use with anisotropy-capable FEMTIC builds. By default every
    region is re-written as v5 aniso_type 0 (isotropic) with the same
    resistivity, so the file is a pure format upgrade; optionally, selected
    region indices can be marked transverse-isotropic (type 1) or generally
    anisotropic (type 2) with caller-supplied rhoYY/rhoZZ ratios, rotation
    angles, and fix flags. Reuses the existing _parse_region_line_v4 /
    _format_region_line_v5 / _ANISO_* infrastructure (see the 2026-06-22
    changelog entry). Also exposed as CLI subcommand
    "iso-to-aniso-block". This code is AI-generated (Claude, Anthropic) and
    should be reviewed before production use.

    IMPORTANT CAVEAT: the anisotropic region-line layout below was
    reverse-engineered from the two example files and the manual slides
    shipped in FemticAniso.zip (Manual_Of_FEMTIC_Anisotropic.pdf), not from
    FEMTIC's own source code. Please double-check the written file against
    a fresh FEMTIC-anisotropic run (or against the Anisotropic/ example)
    before trusting it in production.
Modified: 2026-09-13 by Claude Sonnet 5 (Anthropic) — added Section 6a:
    read_results_model(), read_results_data(), read_results_distortion(),
    and the umbrella read_results_hdf5(), reading FEMTIC's new
    results_iterX.h5 (the C++ HDF5 output extension now merges the former
    model_iterX.h5 + data_iterX.h5 pair into one file, as sibling /model,
    /data, /distortion groups; /data rows also gained cal_re/cal_im
    calculated-response fields alongside the existing observed re_val/
    im_val/re_err/im_err). All four readers transparently fall back to the
    older, standalone model_iterX.h5/data_iterX.h5 layout (root-level
    content, no /distortion) when pointed at a pre-2026-09-13 file, EXCEPT
    that read_results_data(decode_labels=True) (the default) raises
    KeyError on such a file, since cal_re/cal_im cannot be reconstructed
    after the fact -- pass decode_labels=False for observed-data-only
    access to old files. This closes the "HDF5 archive schema
    compatibility with FEMTIC v5" open item read_h5_sensitivity()'s
    2026-09-08 entry pointed at: that function's schema was provisional
    because no writer existed yet; it now defaults to the settled
    /model/sensitivity/raw (or .../volume_normalised) location in
    results_iterX.h5 for the precomputed path, tried automatically even
    with group=None, and to jacobian.h5's real dataset name /data_errors
    (previously the placeholder "data_error", singular, which never
    matched any real file) for the derived-from-Jacobian fallback path.
    Tested against a minimal in-memory h5py-API mock (h5py itself was not
    installable in the environment this was authored in), covering: the
    new combined-file schema end-to-end; both legacy fallback shapes,
    including the case that actually needed the isinstance(..., h5py.Group)
    guard in _h5_find_group -- a legacy data_iterX.h5 has a top-level
    DATASET literally named "data" (not a group), which would otherwise be
    silently mis-detected as the new schema and fail confusingly several
    lines later; the decode_labels=True KeyError on old files; and
    read_h5_sensitivity()'s new automatic model/sensitivity fallback plus
    the jacobian.h5 path with error weighting. Not yet re-verified against
    a real FEMTIC-written file on disk -- run
    inspect_file()-style spot checks (or just read_results_hdf5() itself,
    since it prints a one-line summary of everything it found) against an
    actual run's output as a final check.
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

import math
import numpy as np
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

from numpy.random import Generator, default_rng
from scipy.sparse import issparse
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

#: Canonical femtic.cnv column names, matched against each header token
#: via a case-insensitive **substring** test rather than an exact string
#: match. FEMTIC's own header spelling/casing is not guaranteed to stay
#: fixed across versions (e.g. "Roughness" could just as well appear as
#: "roughness", "Rough", or similar in some future release) -- matching
#: on a short, lower-case substring of the canonical name is robust to
#: that, in the same way the original name/position-independent design
#: of read_cnv() was robust to the Beta/Distortion columns coming and
#: going. Order matters only in that no pattern below may be a substring
#: of a different pattern's own canonical token; first match wins.
_CNV_CANONICAL_PATTERNS = [
    ("iter",        "Iter"),
    ("retrial",     "Retrial"),
    ("alpha",       "Alpha"),
    ("beta",        "Beta"),
    ("damp",        "Damp"),
    ("rough",       "Roughness"),
    ("distort",     "Distortion"),
    ("misfit",      "Misfit"),
    ("rms",         "RMS"),
    ("obj",         "ObjFunc"),
    ("lmdcg",       "LmdCG"),
    ("crossgra",    "CrossGra"),
    ("mupdatemean", "MupdateMean"),
    ("abic",        "ABIC"),
]


def _cnv_canonical_name(token: str) -> str:
    """Map one femtic.cnv header token to its canonical column name via a
    case-insensitive substring match against ``_CNV_CANONICAL_PATTERNS``
    (e.g. a token containing "rough", any case, -> "Roughness"). Tokens
    that don't match any known pattern are returned verbatim (trailing
    ``#`` stripped) so unrecognised/future columns are still captured
    rather than silently dropped.
    """
    raw = token.rstrip("#")
    low = raw.lower()
    for pattern, canonical in _CNV_CANONICAL_PATTERNS:
        if pattern in low:
            return canonical
    return raw


def read_cnv(source, *, columnar: bool = False) -> dict:
    """Read a FEMTIC convergence file (``femtic.cnv``) using its own header
    row to determine column positions.

    FEMTIC's ``.cnv`` column layout is not fixed: the ``Beta``/``Distortion``
    (and, in ABIC/cross-gradient runs, ``LmdCG``/``CrossGra``/
    ``MupdateMean``/``ABIC``) columns only appear depending on run options,
    shifting every column after them -- independent of the FEMTIC version
    number. Parsing by name off the file's own header row (e.g.
    ``Iter#  Retrial#  Alpha  Beta  Damp  Roughness  Distortion  Misfit  RMS
    ObjFunc``) is robust to that, and to any future column additions,
    without needing a version/feature switch at all.

    Header tokens are matched to canonical column names (``"Iter"``,
    ``"Roughness"``, ``"RMS"``, ...) case-insensitively via a short
    substring of each canonical name (see ``_CNV_CANONICAL_PATTERNS``,
    e.g. ``"rough"`` for ``"Roughness"``, ``"rms"`` for ``"RMS"``) rather
    than an exact, case-sensitive match against one hardcoded spelling --
    so a header written with different capitalisation, or a column named
    e.g. "Rough" instead of "Roughness", still resolves to the same
    canonical key every caller relies on (``row["RMS"]``,
    ``row["Roughness"]``, ...). Header-row *detection* itself uses the
    same substring test against "iter" and "rms" (both required, either
    case), rather than requiring the literal token ``"RMS"``.

    Parameters
    ----------
    source : str | Path
        Path to a ``femtic.cnv`` file, or to a directory containing one
        (``femtic.cnv`` is looked up inside it).
    columnar : bool, optional
        If True, also build a ``"data"`` entry: a dict mapping each header
        column name to a 1-D ``np.ndarray`` of that column's values across
        the whole convergence history (row order preserved), e.g.
        ``cnv["data"]["RMS"]`` for the full nRMS trace. Default False --
        the per-row ``"rows"`` list already covers most uses, so this is
        opt-in extra work/memory for convergence-curve plotting and
        similar whole-column access.

    Returns
    -------
    dict with keys:

        ``"columns"`` : dict[str, int]
            Canonical column name -> 0-based index (see
            ``_CNV_CANONICAL_PATTERNS``; an unrecognised header token is
            kept verbatim, trailing ``#`` stripped, e.g. a genuinely novel
            future column). Only columns present in *this* file's header
            appear here -- e.g. no ``"Beta"``/``"Distortion"`` keys for a
            without-distortion run.
        ``"rows"`` : list[dict[str, float]]
            One dict per data row, keyed by the same names as
            ``"columns"``. ``"Iter"``/``"Retrial"`` are integer-valued
            floats (round cleanly with ``int(...)``).
        ``"data"`` : dict[str, np.ndarray], only present if ``columnar=True``
            Column name -> array of that column's values over all rows,
            same names as ``"columns"``.

    Raises
    ------
    ValueError
        If no header row (a line naming both an "Iter"-like and an
        "RMS"-like column, matched case-insensitively) is found before
        the first data row.

    Author: Claude Sonnet 5 (Anthropic), 2026-09-02.
    Updated: Claude Sonnet 5 (Anthropic), 2026-09-07 -- case-insensitive,
    substring-based header/column matching (see docstring above).
    """
    path = Path(source)
    if path.is_dir():
        path = path / "femtic.cnv"

    columns: Optional[dict] = None
    rows: list = []

    with open(path, "r", errors="ignore") as f:
        for line in f:
            tokens = line.split()
            if not tokens:
                continue
            if columns is None:
                _low = [tok.lower() for tok in tokens]
                _has_iter = any("iter" in tok for tok in _low)
                _has_rms  = any("rms" in tok for tok in _low)
                if _has_iter and _has_rms:
                    columns = {_cnv_canonical_name(tok): i
                               for i, tok in enumerate(tokens)}
                continue  # any line before the header is found is a
                          # comment/blank row, header-detection or not
            try:
                row = {name: float(tokens[idx]) for name, idx in columns.items()}
            except (ValueError, IndexError):
                continue  # trailing comment / malformed row
            rows.append(row)

    if columns is None:
        raise ValueError(
            f"read_cnv: no header row (naming both an 'Iter'-like and an "
            f"'RMS'-like column, case-insensitive) found in {path}."
        )

    result = {"columns": columns, "rows": rows}
    if columnar:
        result["data"] = {
            name: np.array([row[name] for row in rows], dtype=float)
            for name in columns
        }
    return result


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
        Iteration number with the smallest nRMS.
    nrm_best : float
        The corresponding (smallest) nRMS value.

    Notes
    -----
    Column positions are read from femtic.cnv's own header row via
    :func:`read_cnv`, so this works regardless of FEMTIC version or
    whether distortion parameters (the ``Beta``/``Distortion`` columns)
    are present -- previously hardcoded indices (``misft=nline[7]``,
    ``nrmse=nline[8]``) silently matched only the with-distortion column
    layout. Fixed 2026-09-02.
    '''
    if directory is None:
        sys.exit('get_nrms: No directory given! Exit.')

    try:
        cnv = read_cnv(os.path.join(directory, 'femtic.cnv'))
        rows = cnv["rows"]
    except ValueError:
        rows = []

    if len(rows) == 0:
        print (directory, '/femtic.cnv', ' is empty!')
        num_best = -1
        nrm_best = 1e32
    else:
        nrmse = np.array([r["RMS"] for r in rows])
        index_min = int(np.argmin(nrmse))
        nrm_best = float(nrmse[index_min])
        num_best = int(round(rows[index_min]["Iter"]))

    return num_best, nrm_best

def _detect_block_format(region_line: str) -> str:
    """Detect whether a resistivity-block region line is v4 or v5 format.

    v4 format (isotropic only)::

        ireg  rho  rho_lo  rho_hi  n  flag          (6 columns, col[1] is float rho)

    v5 format (iso / transverse-iso / general anisotropy)::

        ireg  aniso_type  ...                        (col[1] is int 0/1/2)

    Detection heuristic: if the second token is an integer 0, 1, or 2 **and** the
    third token looks like a float, the line is v5; otherwise v4.

    Returns ``"v4"`` or ``"v5"``.
    """
    parts = region_line.split()
    if len(parts) < 3:
        return "v4"
    try:
        second = int(parts[1])
    except ValueError:
        return "v4"
    if second not in (0, 1, 2):
        return "v4"
    # Second column is 0/1/2 — could still be v4 if resistivity happens to be 0,1,2
    # (extremely unlikely in practice, but disambiguate by checking col[1] for v4's
    # weighting constant n which is typically 1.0, vs aniso_type which is a small int).
    # The safest check: in v5 col[2] is a float resistivity (>0), in v4 col[1] IS rho.
    try:
        float(parts[2])
        return "v5"
    except ValueError:
        return "v4"


# ---------------------------------------------------------------------------
# Anisotropy-type constants (mirrors ResistivityBlockAnisotropic enums)
# ---------------------------------------------------------------------------
_ANISO_ISO = 0      # isotropic
_ANISO_TI  = 1      # transverse isotropy (rhoXX, rhoYY, strike, dip)
_ANISO_GA  = 2      # general anisotropy  (rhoXX, rhoYY, rhoZZ, strike, dip, slant)


def _parse_region_line_v4(line: str) -> tuple[int, float, float, float, float, int]:
    """Parse one v4 region line.

    Format::

        ireg  rho  rho_lower  rho_upper  n  flag

    Returns
    -------
    (ireg, rho, rho_lower, rho_upper, n, flag)
    """
    parts = line.split()
    if len(parts) < 6:
        raise ValueError(f"Invalid v4 region line (need ≥6 columns): {line!r}")
    ireg      = int(parts[0])
    rho       = float(parts[1])
    rho_lower = float(parts[2])
    rho_upper = float(parts[3])
    n         = float(parts[4])
    flag      = int(parts[5])
    return ireg, rho, rho_lower, rho_upper, n, flag


def _parse_region_line_v5(line: str) -> dict:
    """Parse one v5 region line.

    Returns a dict with keys:

    ``ireg``, ``aniso_type``, ``rho_lo``, ``rho_hi``, ``flag``,
    ``rhoXX``, ``rhoYY``, ``rhoZZ`` (Ω·m),
    ``strike``, ``dip``, ``slant`` (degrees),
    ``fix_rhoXX``, ``fix_rhoYY``, ``fix_rhoZZ``,
    ``fix_strike``, ``fix_dip``, ``fix_slant`` (int 0/1).

    For isotropic cells (``aniso_type == 0``): rhoYY = rhoZZ = rhoXX,
    strike = dip = slant = 0.0, all per-parameter fix flags = 0.

    For TI cells (``aniso_type == 1``): rhoZZ = rhoXX, slant = 0.0,
    fix_rhoZZ = 0, fix_slant = 0.
    """
    parts = line.split()
    if len(parts) < 2:
        raise ValueError(f"Invalid v5 region line (too short): {line!r}")
    ireg       = int(parts[0])
    aniso_type = int(parts[1])

    result = dict(
        ireg=ireg, aniso_type=aniso_type,
        rhoXX=0.0, rhoYY=0.0, rhoZZ=0.0,
        strike=0.0, dip=0.0, slant=0.0,
        rho_lo=0.0, rho_hi=0.0, flag=0,
        fix_rhoXX=0, fix_rhoYY=0, fix_rhoZZ=0,
        fix_strike=0, fix_dip=0, fix_slant=0,
    )

    if aniso_type == _ANISO_ISO:
        # ireg  0  rho  rho_lo  rho_hi  flag
        if len(parts) < 6:
            raise ValueError(f"v5 ISO region line needs ≥6 columns: {line!r}")
        rho            = float(parts[2])
        result["rhoXX"]  = rho
        result["rhoYY"]  = rho
        result["rhoZZ"]  = rho
        result["rho_lo"] = float(parts[3])
        result["rho_hi"] = float(parts[4])
        result["flag"]   = int(parts[5])

    elif aniso_type == _ANISO_TI:
        # ireg  1  rhoXX  rhoYY  strike  dip  rho_lo  rho_hi
        #        fix_rhoXX  fix_rhoYY  fix_strike  fix_dip
        if len(parts) < 12:
            raise ValueError(f"v5 TI region line needs ≥12 columns: {line!r}")
        result["rhoXX"]     = float(parts[2])
        result["rhoYY"]     = float(parts[3])
        result["rhoZZ"]     = float(parts[2])   # TI: rhoZZ = rhoXX
        result["strike"]    = float(parts[4])
        result["dip"]       = float(parts[5])
        result["slant"]     = 0.0
        result["rho_lo"]    = float(parts[6])
        result["rho_hi"]    = float(parts[7])
        result["fix_rhoXX"] = int(parts[8])
        result["fix_rhoYY"] = int(parts[9])
        result["fix_strike"]= int(parts[10])
        result["fix_dip"]   = int(parts[11])

    elif aniso_type == _ANISO_GA:
        # ireg  2  rhoXX  rhoYY  rhoZZ  strike  dip  slant  rho_lo  rho_hi
        #        fix_rhoXX  fix_rhoYY  fix_rhoZZ  fix_strike  fix_dip  fix_slant
        if len(parts) < 16:
            raise ValueError(f"v5 GA region line needs ≥16 columns: {line!r}")
        result["rhoXX"]      = float(parts[2])
        result["rhoYY"]      = float(parts[3])
        result["rhoZZ"]      = float(parts[4])
        result["strike"]     = float(parts[5])
        result["dip"]        = float(parts[6])
        result["slant"]      = float(parts[7])
        result["rho_lo"]     = float(parts[8])
        result["rho_hi"]     = float(parts[9])
        result["fix_rhoXX"]  = int(parts[10])
        result["fix_rhoYY"]  = int(parts[11])
        result["fix_rhoZZ"]  = int(parts[12])
        result["fix_strike"] = int(parts[13])
        result["fix_dip"]    = int(parts[14])
        result["fix_slant"]  = int(parts[15])
    else:
        raise ValueError(f"Unknown aniso_type {aniso_type} in v5 region line: {line!r}")

    return result


def _parse_region_line(line: str, fmt: str = "v4") -> tuple[int, float, float, float, float, int]:
    """Parse one region line, returning the v4-compatible 6-tuple.

    For v5 lines, ``rho`` in the returned tuple is ``rhoXX`` (the primary
    resistivity used for inversion bookkeeping); the full anisotropic
    parameters are accessible via :func:`_parse_region_line_v5`.

    Parameters
    ----------
    line
        Raw text line from the resistivity block file.
    fmt
        ``"v4"`` (default) or ``"v5"``.

    Returns
    -------
    (ireg, rho, rho_lower, rho_upper, n, flag)
        ``n`` is 1.0 for v5 lines (field not present in v5 format).
    """
    if fmt == "v5":
        d = _parse_region_line_v5(line)
        return d["ireg"], d["rhoXX"], d["rho_lo"], d["rho_hi"], 1.0, d["flag"]
    return _parse_region_line_v4(line)


def _format_region_line_v5(d: dict) -> str:
    """Format a v5 region line from the dict produced by :func:`_parse_region_line_v5`."""
    aniso_type = int(d["aniso_type"])
    ireg       = int(d["ireg"])
    if aniso_type == _ANISO_ISO:
        return (
            f" {ireg:9d}         {aniso_type:1d}"
            f"   {d['rhoXX']:.6e}   {d['rho_lo']:.6e}   {d['rho_hi']:.6e}"
            f"         {int(d['flag'])}"
        )
    elif aniso_type == _ANISO_TI:
        return (
            f" {ireg:9d}         {aniso_type:1d}"
            f"        {d['rhoXX']:.6e}   {d['rhoYY']:.6e}"
            f"   {d['strike']:.6e}   {d['dip']:.6e}"
            f"   {d['rho_lo']:.6e}   {d['rho_hi']:.6e}"
            f"         {int(d['fix_rhoXX'])}"
            f"         {int(d['fix_rhoYY'])}"
            f"         {int(d['fix_strike'])}"
            f"         {int(d['fix_dip'])}"
        )
    elif aniso_type == _ANISO_GA:
        return (
            f" {ireg:9d}         {aniso_type:1d}"
            f"        {d['rhoXX']:.6e}   {d['rhoYY']:.6e}   {d['rhoZZ']:.6e}"
            f"   {d['strike']:.6e}   {d['dip']:.6e}   {d['slant']:.6e}"
            f"   {d['rho_lo']:.6e}   {d['rho_hi']:.6e}"
            f"         {int(d['fix_rhoXX'])}"
            f"         {int(d['fix_rhoYY'])}"
            f"         {int(d['fix_rhoZZ'])}"
            f"         {int(d['fix_strike'])}"
            f"         {int(d['fix_dip'])}"
            f"         {int(d['fix_slant'])}"
        )
    else:
        raise ValueError(f"Unknown aniso_type {aniso_type}")


def _format_region_line(
    ireg: int,
    rho: float,
    rho_lower: float,
    rho_upper: float,
    n: float,
    flag: int,
) -> str:
    """Format a v4 region line in FEMTIC resistivity block style."""
    return (
        f" {ireg:9d}        {rho:.6e}   {rho_lower:.6e}   {rho_upper:.6e}   "
        f"{n:.6e} {flag:9d}"
    )


def _infer_ocean_present(region1_line: str, fmt: str = "v4") -> bool:
    """Infer whether region 1 is an 'ocean' fixed block.

    Heuristic (conservative):
    - flag == 1 (fixed)
    - rho <= 1 Ωm (very conductive, typical ocean ~0.25 Ωm)

    This can be overridden by passing ``ocean=True`` or ``ocean=False`` to
    :func:`read_model` / :func:`insert_model`.
    """
    _, rho, _, _, _, flag = _parse_region_line(region1_line, fmt=fmt)
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

        # Peek at the first region line to detect v4 / v5 format
        first_region_line = f.readline()
        if not first_region_line:
            raise ValueError("Unexpected EOF before first region line.")
        fmt = _detect_block_format(first_region_line)

        region_lines: list[str] = []
        region_rho = np.zeros(nreg, dtype=float)
        region_flag = np.zeros(nreg, dtype=int)

        # Re-process the already-read first line, then continue
        all_region_lines = [first_region_line] + [f.readline() for _ in range(nreg - 1)]
        for i, line in enumerate(all_region_lines):
            if not line:
                raise ValueError(
                    f"Unexpected EOF while reading region lines: expected {nreg}, got {i}."
                )
            ireg, rho, _, _, _, flag = _parse_region_line(line, fmt=fmt)
            if ireg != i:
                raise ValueError(f"Expected region index {i} at line {i}, got {ireg}.")
            region_lines.append(line)
            region_rho[i] = rho
            region_flag[i] = flag

    # Decide whether region 1 is treated as ocean (optional)
    ocean_present = False
    if nreg > 1:
        if ocean is None:
            ocean_present = _infer_ocean_present(region_lines[1], fmt=fmt)
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
            f"read_model: file={model_path.name}, fmt={fmt}, nelem={nelem}, nreg={nreg}, "
            f"ocean_present={ocean_present}, fixed={n_fixed}, returned={out_vec.size}."
        )

    return out_vec


def summarise_model_file(
    path: str | Path,
    *,
    ocean: bool | None = None,
    out: bool = True,
) -> dict:
    """Summarise the cell counts in a FEMTIC ``resistivity_block_iterXX.dat``.

    Counts how many mesh elements fall into each category:

    - **air**         — region 0 (always fixed)
    - **ocean**       — region 1 when inferred / forced as ocean
    - **other fixed** — any other region with ``flag == 1``
    - **parameters**  — free (unfixed) regions used in the inversion

    Parameters
    ----------
    path : str or Path
        Path to a ``resistivity_block_iterXX.dat`` file.
    ocean : bool | None
        If None (default), auto-infer whether region 1 is ocean via
        :func:`_infer_ocean_present`.  Pass ``True`` / ``False`` to override.
    out : bool
        If True, print a formatted summary table to stdout.

    Returns
    -------
    dict with keys:

    ``file``, ``nelem``, ``nreg``,
    ``n_air``, ``n_ocean``, ``n_other_fixed``, ``n_params``,
    ``ocean_present``, ``ocean_rho``,
    ``region_rho`` (list[float]), ``region_flag`` (list[int])
    """
    import collections

    model_path = Path(path)
    with model_path.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    idx = 0
    hdr = lines[idx].split(); idx += 1
    if len(hdr) < 2:
        raise ValueError(f"Invalid resistivity block header: {hdr!r}")
    nelem = int(hdr[0])
    nreg  = int(hdr[1])

    elem_region: list[int] = []
    for _ in range(nelem):
        parts = lines[idx].split(); idx += 1
        elem_region.append(int(parts[1]))

    region_rho:   list[float] = []
    region_flag:  list[int]   = []
    region_lines: list[str]   = []
    # Detect format from first region line
    fmt = _detect_block_format(lines[idx]) if nreg > 0 else "v4"
    for i in range(nreg):
        line = lines[idx]; idx += 1
        ireg, rho, _, _, _, flag = _parse_region_line(line, fmt=fmt)
        region_rho.append(rho)
        region_flag.append(flag)
        region_lines.append(line)

    ocean_present = False
    ocean_rho_val = None
    if nreg > 1:
        ocean_present = _infer_ocean_present(region_lines[1], fmt=fmt) if ocean is None else bool(ocean)
        if ocean_present:
            ocean_rho_val = region_rho[1]

    fixed = [False] * nreg
    fixed[0] = True
    for i in range(nreg):
        if region_flag[i] == 1:
            fixed[i] = True
    if ocean_present and nreg > 1:
        fixed[1] = True

    reg_counts: collections.Counter = collections.Counter(elem_region)
    n_air         = reg_counts.get(0, 0)
    n_ocean       = reg_counts.get(1, 0) if ocean_present else 0
    n_other_fixed = sum(
        reg_counts.get(i, 0)
        for i in range(nreg)
        if fixed[i] and i != 0 and not (i == 1 and ocean_present)
    )
    n_params = sum(
        reg_counts.get(i, 0) for i in range(nreg) if not fixed[i]
    )

    summary = {
        "file":          model_path.name,
        "nelem":         nelem,
        "nreg":          nreg,
        "n_air":         n_air,
        "n_ocean":       n_ocean,
        "n_other_fixed": n_other_fixed,
        "n_params":      n_params,
        "ocean_present": ocean_present,
        "ocean_rho":     ocean_rho_val,
        "region_rho":    region_rho,
        "region_flag":   region_flag,
    }

    if out:
        _print_model_summary(summary)

    return summary


def _print_model_summary(s: dict) -> None:
    """Print a formatted model-cell count table (used by summarise_model_file)."""
    n_check = s["n_air"] + s["n_ocean"] + s["n_other_fixed"] + s["n_params"]
    ocean_str = (
        f"yes (rho={s['ocean_rho']:.4g} Ω·m)" if s["ocean_present"] else "no"
    )
    print(f"\n{'─' * 60}")
    print(f"  File          : {s['file']}")
    print(f"  Total cells   : {s['nelem']:>10,d}   ({s['nreg']} regions)")
    print(f"  Ocean inferred: {ocean_str}")
    print(f"  ┌─────────────────────────────────────────┐")
    print(f"  │  Air cells        : {s['n_air']:>10,d}            │")
    print(f"  │  Ocean cells      : {s['n_ocean']:>10,d}            │")
    print(f"  │  Other fixed      : {s['n_other_fixed']:>10,d}            │")
    print(f"  │  Parameters (free): {s['n_params']:>10,d}            │")
    print(f"  │─────────────────────────────────────────│")
    print(f"  │  Check sum        : {n_check:>10,d}            │")
    print(f"  └─────────────────────────────────────────┘")
    if n_check != s["nelem"]:
        print(f"  *** WARNING: check sum {n_check} ≠ nelem {s['nelem']} ***")


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

    # Read the entire template into memory first so that opening the output
    # file for writing (which truncates it) cannot corrupt the template even
    # when template == model_file (in-place rewrite of referencemodel.dat).
    with template_path.open("r", encoding="utf-8", errors="replace") as fin:
        template_lines = fin.readlines()

    if not template_lines:
        raise ValueError(f"insert_model: template file is empty: {template_path}")

    header = template_lines[0]
    hdr_parts = header.split()
    if len(hdr_parts) < 2:
        raise ValueError(f"Invalid resistivity block header: {hdr_parts!r}")
    nelem = int(hdr_parts[0])
    nreg = int(hdr_parts[1])

    if nreg <= 0:
        raise ValueError("No regions in resistivity block (nreg<=0).")

    # Parse element->region mapping and region lines from the in-memory list.
    elem_lines = template_lines[1 : 1 + nelem]
    region_start = 1 + nelem

    # Detect v4 / v5 format from the first region line
    fmt = _detect_block_format(template_lines[region_start]) if nreg > 0 else "v4"

    # reg_meta holds either:
    #   v4: (ireg, rho, lo, hi, n, flag)   tuples
    #   v5: the full dict from _parse_region_line_v5 (preserves all aniso fields)
    reg_meta_v4: list[tuple[int, float, float, float, float, int]] = []
    reg_meta_v5: list[dict] = []
    reg_lines: list[str] = []
    region_flag_list: list[int] = []

    for i in range(nreg):
        idx = region_start + i
        if idx >= len(template_lines):
            raise ValueError(
                f"Unexpected EOF while reading region lines: expected {nreg}, got {i}."
            )
        line = template_lines[idx]
        ireg, rho, lo, hi, nn, flag = _parse_region_line(line, fmt=fmt)
        if ireg != i:
            raise ValueError(f"Expected region index {i} at line {i}, got {ireg}.")
        reg_lines.append(line)
        region_flag_list.append(flag)
        if fmt == "v5":
            reg_meta_v5.append(_parse_region_line_v5(line))
        else:
            reg_meta_v4.append((ireg, rho, lo, hi, nn, flag))

    with out_path.open("w", encoding="utf-8") as fout:
        # Write header and element->region mapping unchanged
        fout.write(header)
        for line in elem_lines:
            fout.write(line)

        # Determine whether region 1 is treated as ocean (optional)
        ocean_present = False
        if nreg > 1:
            if ocean is None:
                ocean_present = _infer_ocean_present(reg_lines[1], fmt=fmt)
            else:
                ocean_present = bool(ocean)

        region_flag = np.array(region_flag_list, dtype=int)

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
                f"  template: nreg={nreg}, fmt={fmt}, fixed={n_fixed}, free={n_free}  "
                f"provided model size={model_arr.size}  "
                "Hint: your template may contain additional fixed regions (flag==1). "
                "Use read_model(...) on the same template to get a consistent free vector."
            )

        # Write regions, preserving metadata; replace rho for free regions (and air/ocean)
        model_ptr = 0
        for i in range(nreg):
            ireg = i
            is_fixed = bool(fixed_mask[ireg])
            is_air   = (ireg == 0)
            is_ocean = (ireg == 1 and ocean_present)

            if fmt == "v5":
                d = dict(reg_meta_v5[i])  # copy so we can mutate
                if is_air:
                    d["rhoXX"] = d["rhoYY"] = d["rhoZZ"] = float(air_rho)
                elif is_ocean:
                    d["rhoXX"] = d["rhoYY"] = d["rhoZZ"] = float(ocean_rho)
                elif not is_fixed:
                    new_rho = float(10.0 ** model_arr[model_ptr])
                    model_ptr += 1
                    # For TI/GA cells scale rhoYY and rhoZZ by the same log10 ratio
                    # as rhoXX (preserve relative anisotropy), unless the cell is
                    # actually isotropic in the template.
                    if d["aniso_type"] == _ANISO_ISO:
                        d["rhoXX"] = d["rhoYY"] = d["rhoZZ"] = new_rho
                    else:
                        ratio = new_rho / d["rhoXX"] if d["rhoXX"] > 0 else 1.0
                        d["rhoYY"] = d["rhoYY"] * ratio
                        d["rhoZZ"] = d["rhoZZ"] * ratio
                        d["rhoXX"] = new_rho
                fout.write(_format_region_line_v5(d) + "\n")
            else:
                ireg_, rho0, lo, hi, nn, flag = reg_meta_v4[i]
                if is_air:
                    rho = float(air_rho)
                elif is_ocean:
                    rho = float(ocean_rho)
                elif is_fixed:
                    rho = float(rho0)
                else:
                    rho = float(10.0 ** model_arr[model_ptr])
                    model_ptr += 1
                fout.write(_format_region_line(ireg, rho, lo, hi, nn, flag) + "\n")

    if out:
        print(f"File {out_path} successfully written.")
        print(
            f"insert_model: fmt={fmt}, nreg={nreg}, ocean_present={ocean_present}, "
            f"fixed={n_fixed}, free={n_free}."
        )


def convert_resistivity_block_to_anisotropic(
    model_file_in: str | Path,
    model_file_out: str | Path,
    *,
    aniso_regions: Sequence[int] | None = None,
    aniso_type: int = _ANISO_TI,
    rhoYY_ratio: float = 1.0,
    rhoZZ_ratio: float = 1.0,
    strike: float = 0.0,
    dip: float = 0.0,
    slant: float = 0.0,
    fix_strike: bool = True,
    fix_dip: bool = True,
    fix_slant: bool = True,
    fix_rho_from_flag: bool = True,
    out: bool = True,
) -> Path:
    """Convert a v4 (isotropic-only) resistivity block to v5 (anisotropic) format.

    Author: Claude (Anthropic). Generated 2026-09-11 for Volker Rath (DIAS).
    AI-generated code — please review before production use (see the
    IMPORTANT CAVEAT below and the matching module-docstring changelog entry).

    FEMTIC's anisotropic build reads ``resistivity_block_iterX.dat`` in the
    "v5" per-region layout (``ireg  aniso_type  ...``, see
    :func:`_parse_region_line_v5` / :func:`_format_region_line_v5`) instead
    of the plain isotropic "v4" layout (``ireg  rho  rho_lo  rho_hi  n  flag``)
    used by the standard (non-anisotropic) build. This helper rewrites a
    v4 file into v5 so it can be fed to an anisotropy-capable FEMTIC build.

    By default (``aniso_regions=None``) this is a **pure format conversion**:
    every region is written with ``aniso_type=0`` (v5-isotropic) and the same
    resistivity, bounds, and fix flag as in the input file — the model is
    physically unchanged, only the file layout changes. This matches how the
    FEMTIC-anisotropic example files were derived from their isotropic
    originals (mesh.dat/observe.dat/element→region mapping identical;
    only the resistivity-parameter section reformatted).

    Passing ``aniso_regions`` additionally marks the given region indices as
    transverse-isotropic (``aniso_type=1``, the default) or generally
    anisotropic (``aniso_type=2``), using the same ``rhoXX`` as the original
    isotropic value, ``rhoYY = rhoXX * rhoYY_ratio`` (and, for general
    anisotropy, ``rhoZZ = rhoXX * rhoZZ_ratio``), and the given rotation
    angles. With the defaults (``rhoYY_ratio=rhoZZ_ratio=1.0``,
    ``strike=dip=slant=0.0``) the marked regions remain numerically
    isotropic but are flagged as anisotropic cell types — exactly how the
    shipped ``Anisotropic/resistivity_block_iter0.dat`` example marks its
    boundary regions, presumably so that they can later be perturbed away
    from isotropy (e.g. by hand or in a subsequent inversion step) without a
    further file-format change.

    Parameters
    ----------
    model_file_in
        Path to a v4 (isotropic) ``resistivity_block_iterX.dat``.
    model_file_out
        Path to write the v5 (anisotropic-format) file to.
    aniso_regions
        Region indices (0-based, matching the file's own numbering) to mark
        as anisotropic. ``None`` (default) converts the format only, leaving
        every region at ``aniso_type=0``.
    aniso_type
        ``_ANISO_TI`` (1, transverse isotropy) or ``_ANISO_GA`` (2, general
        anisotropy) — applied to every region listed in ``aniso_regions``.
    rhoYY_ratio, rhoZZ_ratio
        Multiplicative ratios applied to the original (isotropic) resistivity
        to obtain ``rhoYY`` and ``rhoZZ`` for marked regions. ``rhoZZ_ratio``
        is only used when ``aniso_type == _ANISO_GA``.
    strike, dip, slant
        Rotation angles in degrees applied to marked regions. ``slant`` is
        only used when ``aniso_type == _ANISO_GA``.
    fix_strike, fix_dip, fix_slant
        Whether the corresponding rotation angle is held fixed (not
        inverted) for marked regions. ``fix_slant`` is only used for
        ``_ANISO_GA``.
    fix_rho_from_flag
        If True (default), ``fix_rhoXX``/``fix_rhoYY``(``/fix_rhoZZ``) for
        marked regions are copied from the original v4 ``flag`` column
        (0 = free, 1 = fixed), matching the input file's own fixed/free
        bookkeeping. If False, the principal resistivities of marked
        regions are always left free (0) regardless of the input flag.
    out
        If True, print a short summary line.

    Returns
    -------
    Path
        ``Path(model_file_out)``.

    Raises
    ------
    ValueError
        If ``model_file_in`` is already in v5 format, if ``aniso_type`` is
        not ``_ANISO_TI``/``_ANISO_GA``, or if ``aniso_regions`` contains an
        index outside ``[0, nreg)``.
    """
    if aniso_type not in (_ANISO_TI, _ANISO_GA):
        raise ValueError(
            f"convert_resistivity_block_to_anisotropic: aniso_type must be "
            f"_ANISO_TI ({_ANISO_TI}) or _ANISO_GA ({_ANISO_GA}), got {aniso_type!r}."
        )

    in_path = Path(model_file_in)
    out_path = Path(model_file_out)

    # Read the entire input file into memory first (mirrors insert_model's
    # approach) so that in-place conversion (model_file_out == model_file_in)
    # cannot truncate the source before it has been fully parsed.
    with in_path.open("r", encoding="utf-8", errors="replace") as fin:
        lines = fin.readlines()

    if not lines:
        raise ValueError(f"convert_resistivity_block_to_anisotropic: empty file: {in_path}")

    header = lines[0]
    hdr_parts = header.split()
    if len(hdr_parts) < 2:
        raise ValueError(f"Invalid resistivity block header: {hdr_parts!r}")
    nelem = int(hdr_parts[0])
    nreg = int(hdr_parts[1])

    if nreg <= 0:
        raise ValueError("No regions in resistivity block (nreg<=0).")

    elem_lines = lines[1 : 1 + nelem]
    region_start = 1 + nelem
    region_lines = lines[region_start : region_start + nreg]
    if len(region_lines) != nreg:
        raise ValueError(
            f"Unexpected EOF while reading region lines: expected {nreg}, got {len(region_lines)}."
        )

    fmt = _detect_block_format(region_lines[0])
    if fmt == "v5":
        raise ValueError(
            f"convert_resistivity_block_to_anisotropic: {in_path} is already "
            "in v5 (anisotropic) format — nothing to convert."
        )

    aniso_set: set[int] = set()
    if aniso_regions is not None:
        aniso_set = {int(r) for r in aniso_regions}
        bad = sorted(r for r in aniso_set if not (0 <= r < nreg))
        if bad:
            raise ValueError(
                f"convert_resistivity_block_to_anisotropic: aniso_regions "
                f"contains out-of-range indices for nreg={nreg}: {bad}."
            )

    with out_path.open("w", encoding="utf-8") as fout:
        fout.write(header)
        for line in elem_lines:
            fout.write(line)

        for i in range(nreg):
            ireg, rho, lo, hi, _n, flag = _parse_region_line_v4(region_lines[i])
            if ireg != i:
                raise ValueError(f"Expected region index {i} at line {i}, got {ireg}.")

            if i not in aniso_set:
                d = dict(
                    ireg=ireg, aniso_type=_ANISO_ISO,
                    rhoXX=rho, rhoYY=rho, rhoZZ=rho,
                    strike=0.0, dip=0.0, slant=0.0,
                    rho_lo=lo, rho_hi=hi, flag=flag,
                    fix_rhoXX=0, fix_rhoYY=0, fix_rhoZZ=0,
                    fix_strike=0, fix_dip=0, fix_slant=0,
                )
            else:
                fix_rho = int(flag) if fix_rho_from_flag else 0
                d = dict(
                    ireg=ireg, aniso_type=int(aniso_type),
                    rhoXX=rho, rhoYY=rho * rhoYY_ratio, rhoZZ=rho * rhoZZ_ratio,
                    strike=float(strike), dip=float(dip), slant=float(slant),
                    rho_lo=lo, rho_hi=hi, flag=flag,
                    fix_rhoXX=fix_rho, fix_rhoYY=fix_rho, fix_rhoZZ=fix_rho,
                    fix_strike=int(fix_strike), fix_dip=int(fix_dip), fix_slant=int(fix_slant),
                )

            fout.write(_format_region_line_v5(d) + "\n")

    if out:
        print(
            f"convert_resistivity_block_to_anisotropic: wrote {out_path} "
            f"(nelem={nelem}, nreg={nreg}, aniso_type={aniso_type}, "
            f"marked_anisotropic={len(aniso_set)})."
        )

    return out_path


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
    R: np.ndarray | scipy.sparse.sparray | None = None,
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

        # Peek at first region line to detect format
        first_region_line = f.readline()
        if not first_region_line:
            raise ValueError("Unexpected EOF before first region line (_read_resistivity_block_struct).")
        fmt = _detect_block_format(first_region_line)

        region_lines: list[str] = []
        all_rlines = [first_region_line] + [f.readline() for _ in range(nreg - 1)]
        for i, line in enumerate(all_rlines):
            if not line:
                raise ValueError(f"Unexpected EOF while reading region lines at i={i}.")
            ireg, rho, lo, hi, nn, flag = _parse_region_line(line, fmt=fmt)
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
            ocean_present = _infer_ocean_present(region_lines[1], fmt=fmt)
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
            f"_read_resistivity_block_struct: file={model_path.name}, fmt={fmt}, nelem={nelem}, nreg={nreg}, "
            f"ocean_present={ocean_present}, fixed={int(fixed_mask.sum())}, free={int(free_idx.size)}."
        )

    return {
        "nelem": int(nelem),
        "nreg": int(nreg),
        "fmt": fmt,                          # "v4" or "v5"
        "region_lines_raw": region_lines,    # original text lines (needed for v5 round-trip)
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

    fmt = struct.get("fmt", "v4")
    region_lines_raw: list[str] | None = struct.get("region_lines_raw", None)

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"{nelem:10d} {nreg:10d}\n")
        for i in range(nelem):
            f.write(f"{i:10d} {int(elem_region[i]):10d}\n")
        for i in range(nreg):
            if fmt == "v5" and region_lines_raw is not None:
                # Parse the stored raw line, update rhoXX (and scale YY/ZZ), re-format
                d = _parse_region_line_v5(region_lines_raw[i])
                new_rhoXX = float(region_rho[i])
                if d["aniso_type"] == _ANISO_ISO:
                    d["rhoXX"] = d["rhoYY"] = d["rhoZZ"] = new_rhoXX
                else:
                    ratio = new_rhoXX / d["rhoXX"] if d["rhoXX"] > 0 else 1.0
                    d["rhoYY"] = d["rhoYY"] * ratio
                    d["rhoZZ"] = d["rhoZZ"] * ratio
                    d["rhoXX"] = new_rhoXX
                f.write(_format_region_line_v5(d) + "\n")
            else:
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
        print(f"_write_resistivity_block_struct: wrote {out_path} (fmt={fmt}, nelem={nelem}, nreg={nreg}).")


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


def collect_forward_response(
    result_files: Dict[str, str],
    site_file: str,
    data_kind: str = "rhophas",
    out: bool = True,
) -> Tuple[np.ndarray, list]:
    """
    Flatten one FEMTIC member's *calculated* forward response across one
    or more observation types into a single 1-D vector.

    Intended for building an ensemble data matrix (n_members, n_data) for
    ensemble-native sensitivity measures (e.g. SimRC / correlation
    sensitivity, Bobe, Keller & Van De Vijver, 2021,
    Geophysical Prospecting, doi:10.1111/1365-2478.13068) directly from
    per-member ``result_XXX.txt`` files, with no Jacobian required.

    Parameters
    ----------
    result_files : dict
        Mapping ``{data_type: path}`` for the ``result_XXX.txt`` files
        present in one ensemble member's run directory, e.g.
        ``{"rhophas": "run_003/result_MT.txt", "vtf": "run_003/result_VTF.txt"}``.
        Only entries whose path exists are used; a member missing every
        requested type raises ``FileNotFoundError``.
    site_file : str
        Site metadata file shared across the ensemble (see
        :func:`get_femtic_data`) -- normally identical for every member,
        since only the model (and hence the calculated response) differs
        across an RTO/GST ensemble, not the site geometry.
    data_kind : str, optional
        Passed to :func:`get_femtic_data` as ``data_type`` for every
        entry in ``result_files`` whose key does not itself look like a
        recognised type (``"imp"``/``"vtf"``/``"pt"``/``"rhophas"``); use
        per-type dict keys (e.g. ``"vtf"``) to mix types in one call.
    out : bool, optional
        Print a one-line summary per data type read.

    Returns
    -------
    d_cal : ndarray, shape (n_data,)
        Concatenated, flattened "cal" (calculated/forward) values across
        all data types found, in the order ``result_files`` is iterated
        (dict insertion order) -- this order must be identical for every
        ensemble member for the resulting per-member vectors to line up
        column-for-column.
    manifest : list of (str, int)
        ``(data_type, n_values)`` pairs recording how ``d_cal`` was
        assembled, for sanity-checking that every member produced the
        same composition/length before stacking into an ensemble matrix.

    Raises
    ------
    FileNotFoundError
        If none of the paths in ``result_files`` exist.

    Author: Claude Sonnet 5 (Anthropic), 2026-09-08
    """
    chunks: list[np.ndarray] = []
    manifest: list[Tuple[str, int]] = []

    for dtyp, path in result_files.items():
        if not os.path.isfile(path):
            continue
        kind = dtyp if dtyp.lower() in ("imp", "vtf", "pt", "rhophas") else data_kind
        dd = get_femtic_data(path, site_file, data_type=kind, out=out)
        vals = np.asarray(dd["cal"], dtype=float).ravel()
        chunks.append(vals)
        manifest.append((dtyp, vals.size))

    if not chunks:
        raise FileNotFoundError(
            f"collect_forward_response: none of {list(result_files.values())} "
            f"exist."
        )

    d_cal = np.concatenate(chunks)
    if out:
        print(f"collect_forward_response: assembled {d_cal.size} values from "
              f"{[m[0] for m in manifest]}")

    return d_cal, manifest


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
        # v5 per-block anisotropy arrays (filled for all blocks; zero for ISO)
        region_aniso_type = np.zeros(nreg, dtype=int)
        region_rhoYY = np.empty(nreg, dtype=float)
        region_rhoZZ = np.empty(nreg, dtype=float)
        region_strike = np.zeros(nreg, dtype=float)
        region_dip    = np.zeros(nreg, dtype=float)
        region_slant  = np.zeros(nreg, dtype=float)

        # Detect format from first region line
        first_region_line = f.readline()
        if not first_region_line:
            raise ValueError("Unexpected EOF before first region line (read_resistivity_block).")
        fmt = _detect_block_format(first_region_line)
        all_rlines = [first_region_line] + [f.readline() for _ in range(nreg - 1)]

        for line in all_rlines:
            if not line:
                raise ValueError("Unexpected EOF while reading region lines.")
            ireg, rho, rho_min, rho_max, n, flag = _parse_region_line(line, fmt=fmt)
            if not (0 <= ireg < nreg):
                raise ValueError(f"Region index {ireg} out of range 0..{nreg-1}.")
            region_rho[ireg] = rho
            region_rho_lower[ireg] = rho_min
            region_rho_upper[ireg] = rho_max
            region_n[ireg] = n
            region_flag[ireg] = flag
            if fmt == "v5":
                d = _parse_region_line_v5(line)
                region_aniso_type[ireg] = d["aniso_type"]
                region_rhoYY[ireg]      = d["rhoYY"]
                region_rhoZZ[ireg]      = d["rhoZZ"]
                region_strike[ireg]     = d["strike"]
                region_dip[ireg]        = d["dip"]
                region_slant[ireg]      = d["slant"]
            else:
                region_rhoYY[ireg] = rho
                region_rhoZZ[ireg] = rho

    result = {
        "nelem": np.array(nelem, dtype=int),
        "nreg": np.array(nreg, dtype=int),
        "fmt": fmt,
        "region_of_elem": region_of_elem,
        "region_rho": region_rho,
        "region_rho_lower": region_rho_lower,
        "region_rho_upper": region_rho_upper,
        "region_n": region_n,
        "region_flag": region_flag,
    }
    if fmt == "v5":
        result.update({
            "region_aniso_type": region_aniso_type,
            "region_rhoYY":      region_rhoYY,
            "region_rhoZZ":      region_rhoZZ,
            "region_strike":     region_strike,
            "region_dip":        region_dip,
            "region_slant":      region_slant,
        })
    return result


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


def tet_volumes(nodes: np.ndarray, conn: np.ndarray) -> np.ndarray:
    """Compute the volume of each tetrahedron (vectorised).

    Uses the scalar triple product: V = |det([b-a, c-a, d-a])| / 6.

    Parameters
    ----------
    nodes : ndarray, shape (nn, 3)
        Node coordinates.
    conn : ndarray, shape (nelem, 4)
        Tetrahedral connectivity (0-based node indices).

    Returns
    -------
    vol : ndarray, shape (nelem,)
        Absolute volumes in the same units as the node coordinates cubed.
    """
    v = nodes[conn]                        # (nelem, 4, 3)
    a, b, c, d = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
    cross = np.cross(b - a, c - a)         # (nelem, 3)
    det   = (cross * (d - a)).sum(axis=1)  # (nelem,)
    return np.abs(det) / 6.0


def build_region_geometry(
    nodes: np.ndarray,
    conn: np.ndarray,
    elem_region: np.ndarray,
    free_idx: np.ndarray,
) -> tuple:
    """Compute volume-weighted centroid and total volume for each free region.

    Parameters
    ----------
    nodes : ndarray, shape (nn, 3)
        Node coordinates.
    conn : ndarray, shape (nelem, 4)
        Tetrahedral connectivity (0-based node indices).
    elem_region : ndarray, shape (nelem,)
        Region index assigned to each element.
    free_idx : ndarray, shape (n_free,)
        Ordered sequence of region indices that are free (inversion parameters).

    Returns
    -------
    region_ctr : ndarray, shape (n_free, 3)
        Volume-weighted centroid ``[x, y, z]`` for each free region.
    region_vol : ndarray, shape (n_free,)
        Total volume (m³) for each free region.  Regions with no assigned
        elements receive centroid ``[0, 0, 0]`` and volume ``0``.
    """
    elem_ctr = nodes[conn].mean(axis=1)        # (nelem, 3)
    elem_vol = tet_volumes(nodes, conn)         # (nelem,)

    n_free = len(free_idx)
    region_ctr = np.zeros((n_free, 3), dtype=float)
    region_vol = np.zeros(n_free,      dtype=float)

    for k, ireg in enumerate(free_idx):
        sel = elem_region == ireg
        if np.any(sel):
            vols = elem_vol[sel]
            region_vol[k] = vols.sum()
            region_ctr[k] = (vols[:, np.newaxis] * elem_ctr[sel]).sum(axis=0) / region_vol[k]

    return region_ctr, region_vol


def brick_mask(
    centroids: np.ndarray,
    *,
    center: Sequence[float],
    axes: Sequence[float],
    angles_deg: Sequence[float] = (0.0, 0.0, 0.0),
    convention: Literal["zyx", "sds"] = "zyx",
) -> np.ndarray:
    """Compute a boolean mask for points inside a rotated rectangular prism (brick).

    The brick is defined by its centre, half-extents along each local axis, and
    a rotation that maps local axes to global coordinates.  The containment test
    in the local frame is:

        |x'| ≤ a  AND  |y'| ≤ b  AND  |z'| ≤ c

    Parameters
    ----------
    centroids : ndarray, shape (n, 3)
        Point coordinates (typically element or region centroids).
    center : sequence of float
        Brick centre ``(cx, cy, cz)`` in the same coordinate system as centroids.
    axes : sequence of float
        Half-extents ``(a, b, c)`` along the local x, y, z axes.  Must be positive.
    angles_deg : sequence of float, optional
        Rotation angles in degrees.  Interpretation depends on ``convention``.
        Default is no rotation.
    convention : {"zyx", "sds"}
        Rotation convention — same options as :func:`ellipsoid_mask`.

        - ``"zyx"``: intrinsic Z-Y-X (yaw, pitch, roll).
        - ``"sds"``: strike/dip/slant.

    Returns
    -------
    ndarray, shape (n,)
        Boolean mask where True indicates the point lies inside (or on) the brick.

    Raises
    ------
    ValueError
        If input shapes or parameters are invalid.
    """
    pts = np.asarray(centroids, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"brick_mask: centroids must have shape (n,3); got {pts.shape}.")

    c = np.asarray(list(center), dtype=float).ravel()
    if c.size != 3:
        raise ValueError(f"brick_mask: center must have length 3; got {c.size}.")

    ax = np.asarray(list(axes), dtype=float).ravel()
    if ax.size != 3:
        raise ValueError(f"brick_mask: axes must have length 3; got {ax.size}.")
    if np.any(ax <= 0.0) or not np.all(np.isfinite(ax)):
        raise ValueError("brick_mask: axes must be finite and > 0.")

    conv = str(convention).strip().lower()
    if conv == "zyx":
        R = _rotation_matrix_zyx(angles_deg)
    elif conv == "sds":
        R = _rotation_matrix_sds(angles_deg)
    else:
        raise ValueError(f"brick_mask: unsupported convention={convention!r}.")

    local = (pts - c[None, :]) @ R   # (n, 3) in rotated frame
    return (
        (np.abs(local[:, 0]) <= ax[0]) &
        (np.abs(local[:, 1]) <= ax[1]) &
        (np.abs(local[:, 2]) <= ax[2])
    )


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


def read_h5_sensitivity(
    h5_path: str,
    group: str | None = None,
    cumsens_key: str = "raw",
    jacobian_key: str = "jacobian",
    error_key: str | None = "data_errors",
    normalize: bool = False,
    out: bool = True,
) -> np.ndarray:
    """
    Read (or derive) a per-free-parameter cumulative sensitivity vector,
    either from a FEMTIC ``results_iterX.h5`` file's precomputed
    ``/model/sensitivity/*`` datasets, or derived from a ``jacobian.h5``
    file's raw Jacobian.

    Schema
    ------
    As of 2026-09-13 this schema is settled (see
    :mod:`Section 6a <femtic>`/``OutputHDF5.h`` in the FEMTIC C++ source
    tree) -- this function was originally written against a provisional
    schema before that pipeline existed; the defaults below now match the
    real files, and the ``*_key``/``group`` arguments remain configurable
    for anyone reading an older or customised export. Two alternative
    contents are supported, tried in this order:

    1. A pre-computed 1-D dataset ``cumsens_key`` (default ``"raw"``),
       shape ``(n_free,)`` -- used as-is if present. FEMTIC itself writes
       exactly this at ``results_iterX.h5``'s ``/model/sensitivity/raw``
       (and ``/model/sensitivity/volume_normalised`` as an alternative --
       pass ``cumsens_key="volume_normalised"`` for the volume-normalised
       version), whenever sensitivity was computed for that iteration; see
       ``getSensitivityScalarValuesReduced()``/``outputModelToHDF5()`` in
       the FEMTIC C++ source. `group` defaults to `None`, which this
       function's lookup treats as "try the file root, then
       ``model/sensitivity`` automatically" -- so
       ``read_h5_sensitivity("results_iter7.h5")`` with no other arguments
       works out of the box against FEMTIC's own output.
    2. A 2-D Jacobian/sensitivity-matrix dataset ``jacobian_key`` (default
       ``"jacobian"``), shape ``(n_data, n_free)``, with entries
       G_ij = d(log d_i) / d(log m_j) -- the standard FEMTIC sensitivity
       convention (cf. Christiansen & Auken, 2012, Geophysics 77, WB171).
       This matches ``jacobian.h5``'s ``/jacobian`` dataset (root-level, no
       group needed -- see ``outputJacobianToHDF5()``). The cumulative
       sensitivity is then computed here as

           S_j = sum_i |G_ij|                      (error_key is None)
           S_j = sum_i |G_ij| / sigma_i             (error_key given)

       i.e. optionally weighted by the inverse of a 1-D per-datum error
       vector ``error_key`` (default ``"data_errors"``, shape
       ``(n_data,)``, matching ``jacobian.h5``'s ``/data_errors``),
       matching the error-normalised sensitivity used by Christiansen &
       Auken (2012) and Oldenburg & Li (1999). If ``error_key`` is set but
       the dataset is absent, an unweighted sum is used with a printed
       warning.

    Note that the two paths above normally come from two DIFFERENT files
    (``results_iterX.h5`` for the precomputed path, ``jacobian.h5`` for the
    raw-Jacobian path) -- point `h5_path` at whichever file has the
    information you want; this function does not merge the two.

    Parameters
    ----------
    h5_path : str
        Path to a ``results_iterX.h5`` (precomputed path) or ``jacobian.h5``
        (raw-Jacobian path) file.
    group : str or None, optional
        HDF5 group containing the sensitivity dataset(s). ``None``
        (default) tries the file root first, then ``"model/sensitivity"``
        (the precomputed dataset's real location in ``results_iterX.h5``).
        Pass an explicit group to override either lookup.
    cumsens_key : str, optional
        Dataset name for a pre-computed cumulative sensitivity vector
        (default ``"raw"``; pass ``"volume_normalised"`` for the
        volume-normalised alternative FEMTIC also writes).
    jacobian_key : str, optional
        Dataset name for the raw Jacobian/sensitivity matrix, used only
        if ``cumsens_key`` is not found.
    error_key : str or None, optional
        Dataset name for a 1-D per-datum error vector used to weight the
        Jacobian sum (``None`` disables weighting).
    normalize : bool, optional
        If True, divide the returned vector by its own maximum (so it
        runs from 0 to 1). Off by default so that per-member magnitudes
        remain comparable before ensemble aggregation -- normalise after
        aggregating across members instead, if at all (see
        ``femtic_ens_post.py``'s SENS_NORMALIZE).
    out : bool, optional
        Print a one-line summary.

    Returns
    -------
    cum_sens : ndarray, shape (n_free,)
        Cumulative sensitivity per free parameter, in the same
        cell/region ordering as the corresponding
        ``resistivity_block_iterX.dat`` / :func:`read_model` output.

    Raises
    ------
    FileNotFoundError
        If ``h5_path`` does not exist.
    KeyError
        If neither ``cumsens_key`` nor ``jacobian_key`` is found (in
        ``group``, at the root, or -- for ``cumsens_key`` -- under
        ``model/sensitivity``).

    See Also
    --------
    read_results_model : reads the whole ``/model`` group at once (including
        both sensitivity datasets, when present), if you need more than
        just the cumulative sensitivity vector.

    Author: Claude Sonnet 5 (Anthropic), 2026-09-08.
    Updated 2026-09-13 (schema settled; defaults now match FEMTIC's real
    ``results_iterX.h5``/``jacobian.h5`` output; automatic
    ``model/sensitivity`` fallback added) by Claude Sonnet 5 (Anthropic).
    """
    try:
        import h5py
    except Exception as exc:  # pragma: no cover
        raise ImportError("read_h5_sensitivity requires the 'h5py' package.") from exc

    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"read_h5_sensitivity: {h5_path} not found.")

    def _lookup(h5, key):
        """
        Try, in order: the explicit `group` (if given), the file root, and
        -- only for the precomputed cumsens_key path, since that's the one
        real dataset name that moved when model_iterX.h5 was merged into
        results_iterX.h5 -- "model/sensitivity" (skipped for jacobian_key/
        error_key, which always live at jacobian.h5's root).
        """
        candidate_groups = [group] if group is not None else []
        candidate_groups.append(None)  # root
        if key == cumsens_key:
            candidate_groups.append("model/sensitivity")
        for grp in candidate_groups:
            if grp is None:
                if key in h5:
                    return np.asarray(h5[key][()])
            elif grp in h5 and key in h5[grp]:
                return np.asarray(h5[grp][key][()])
        return None


    with h5py.File(h5_path, "r") as h5:
        cum_sens = _lookup(h5, cumsens_key)
        if cum_sens is not None:
            if out:
                print(f"read_h5_sensitivity: using precomputed '{cumsens_key}' "
                      f"from {h5_path}  shape={cum_sens.shape}")
        else:
            jac = _lookup(h5, jacobian_key)
            if jac is None:
                raise KeyError(
                    f"read_h5_sensitivity: neither '{cumsens_key}' nor "
                    f"'{jacobian_key}' found in {h5_path} "
                    f"(group={group!r})."
                )
            jac = np.abs(jac)
            if error_key is not None:
                err = _lookup(h5, error_key)
                if err is None:
                    print(f"  WARNING: '{error_key}' not found in {h5_path} -- "
                          f"using unweighted |Jacobian| sum.")
                else:
                    err = np.asarray(err, dtype=float)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        jac = jac / err[:, np.newaxis]
                    jac[~np.isfinite(jac)] = 0.0
            cum_sens = np.sum(jac, axis=0)
            if out:
                print(f"read_h5_sensitivity: derived cumulative sensitivity from "
                      f"'{jacobian_key}' {jac.shape} in {h5_path}  "
                      f"-> shape={cum_sens.shape}")

    cum_sens = np.asarray(cum_sens, dtype=float).ravel()
    if normalize:
        _mx = np.nanmax(cum_sens)
        if _mx > 0:
            cum_sens = cum_sens / _mx

    if out:
        print(f"  cumulative sensitivity range: "
              f"[{np.nanmin(cum_sens):.4e}, {np.nanmax(cum_sens):.4e}]")

    return cum_sens


# ============================================================================
# SECTION 6a: results_iterX.h5 readers (model + data + distortion)
# ============================================================================
#
# As of 2026-09-13, FEMTIC's C++ HDF5 output (``_HDF5_OUT``, all three active
# source trees: femtic_v4_src, femtic_v5_src, femtic_dabic_v2.7_src) writes
# ONE file per iteration, ``results_iterX.h5``, with three sibling top-level
# groups -- replacing the older, separate ``model_iterX.h5`` +
# ``data_iterX.h5`` pair (``distortion_iterX.dat`` is still ALSO written
# separately, unchanged, alongside the new HDF5 embedding). This is the
# schema that :func:`read_h5_sensitivity` above called "provisional" -- it
# is now settled; see ``OutputHDF5.h`` (``outputResultsToHDF5``) in the
# FEMTIC C++ source tree for the authoritative, currently-maintained schema
# comment, which the readers below mirror field-for-field.
#
#   /model/metadata             -- attrs: iterNum, nElem, nNodes, nBlocks,
#                                   anisotropic (v5 only; absent on v4/dabic)
#   /model/element_block_map    -- int[nElem]
#   /model/blocks               -- compound[nBlocks]: blockID, resistivity,
#                                   rho_min, rho_max, weight, type
#                                   (+ rho_xx, rho_yy, rho_zz, strike, dip,
#                                   slant on v5 only)
#   /model/mesh/node_coords     -- double[nNodes, 3]
#   /model/mesh/elem_nodes      -- int[nElem, nNodesPerElem]
#   /model/sensitivity/raw               -- double[nBlocks] (only present
#   /model/sensitivity/volume_normalised -- double[nBlocks]  when sensitivity
#                                            was computed this iteration)
#
#   /data/metadata   -- attrs: nRows, iterNum, nMissingCalc
#   /data/data       -- compound[nRows]: freq, datatype, site_id, site_x,
#                        site_y, site_z, re_val, im_val, re_err, im_err,
#                        cal_re, cal_im, component (dataset itself absent
#                        when nRows == 0)
#
#   /distortion/metadata -- attrs: iterNum, nRows, type
#   /distortion/params   -- compound[nRows]: site_id, param1..param4,
#                            isFixed (group entirely absent when distortion
#                            estimation was disabled for the run)
#
# Older ``model_iterX.h5``/``data_iterX.h5`` files (written before
# 2026-09-13) are also readable here: they have the same dataset content but
# at the file root instead of under ``/model``/``/data``, and never have a
# ``/distortion`` group -- the readers below fall back to the root
# transparently, so old and new files both work with the same call.

RESULTS_H5_DATATYPE_NAMES: dict[int, str] = {
    0: "MT",
    1: "APP_RES_AND_PHS",
    2: "HTF",
    3: "VTF",
    4: "PT",
    5: "NMT",
    6: "NMT2",
    7: "NMT2_APP_RES_AND_PHS",
}

RESULTS_H5_COMPONENT_NAMES: dict[str, list[str]] = {
    "MT":                   ["Zxx", "Zxy", "Zyx", "Zyy"],
    "NMT2":                 ["Zxx", "Zxy", "Zyx", "Zyy"],
    "APP_RES_AND_PHS":      ["rhoXX", "rhoXY", "rhoYX", "rhoYY",
                              "phsXX", "phsXY", "phsYX", "phsYY"],
    "NMT2_APP_RES_AND_PHS": ["rhoXX", "rhoXY", "rhoYX", "rhoYY",
                              "phsXX", "phsXY", "phsYX", "phsYY"],
    "HTF":                  ["Txx", "Txy", "Tyx", "Tyy"],
    "VTF":                  ["Tzx", "Tzy"],
    "PT":                   ["PTxx", "PTxy", "PTyx", "PTyy"],
    "NMT":                  ["Yx", "Yy"],
}

# AnalysisControl::TypeOfDistortion, as written to /distortion/metadata "type"
RESULTS_H5_DISTORTION_TYPE_NAMES: dict[int, str] = {
    0: "NO_DISTORTION",
    1: "ESTIMATE_DISTORTION_MATRIX_DIFFERENCE",
    2: "ESTIMATE_GAINS_AND_ROTATIONS",
    3: "ESTIMATE_GAINS_ONLY",
}


def _h5_open_source(source, h5py_module):
    """
    Return ``(h5_obj, should_close)`` for `source`, which may be a path
    (``str``/``Path``) or an already-open ``h5py.File``/``h5py.Group``
    (the latter lets :func:`read_results_hdf5` open the file once and hand
    the same handle to each sub-reader).
    """
    if isinstance(source, (str, Path)):
        return h5py_module.File(source, "r"), True
    return source, False


def _h5_find_group(h5, h5py_module, new_name: str, root_markers: Sequence[str]):
    """
    Return the group to read from: ``h5[new_name]`` if present AND it is
    actually an ``h5py.Group`` (current schema, e.g.
    ``"model"``/``"data"``/``"distortion"``); otherwise `h5` itself if it
    already looks like that group (pre-2026-09-13 files, whose content sits
    at the root -- detected via `root_markers`, e.g. ``("metadata",
    "blocks")`` for the model group); otherwise ``None``.

    The explicit ``isinstance(..., Group)`` check matters for one specific
    legacy case: the old, standalone ``data_iterX.h5`` wrote its rows to a
    top-level DATASET literally named ``/data`` (not a group) -- so
    ``"data" in h5`` is true for such a file, but ``h5["data"]`` is a
    Dataset, not a Group, and callers here always expect a Group (they index
    into it with ``g["metadata"]`` etc). Without this check that legacy file
    would be silently mis-detected as the new schema and then fail with a
    confusing TypeError several lines later instead of falling through to
    the root-marker fallback below, where it belongs.
    """
    if new_name in h5 and isinstance(h5[new_name], h5py_module.Group):
        return h5[new_name]
    if all(marker in h5 for marker in root_markers):
        return h5
    return None


def read_results_model(source, *, out: bool = True) -> dict:
    """
    Read the ``/model`` group of a ``results_iterX.h5`` file (or an older,
    standalone ``model_iterX.h5``).

    Parameters
    ----------
    source : str, Path, h5py.File, or h5py.Group
        Path to a ``results_iterX.h5``/``model_iterX.h5`` file, or an
        already-open h5py file/group (as passed internally by
        :func:`read_results_hdf5`).
    out : bool, optional
        Print a one-line summary.

    Returns
    -------
    dict with keys:
        iterNum, nElem, nNodes, nBlocks : int
        anisotropic : bool or None
            None if the "anisotropic" attribute is absent (v4/dabic v2.7
            runs, which have no anisotropic resistivity blocks).
        element_block_map : ndarray, shape (nElem,)
        blocks : structured ndarray, shape (nBlocks,)
            Fields: blockID, resistivity, rho_min, rho_max, weight, type,
            and (v5 only) rho_xx, rho_yy, rho_zz, strike, dip, slant.
        node_coords : ndarray, shape (nNodes, 3)
        elem_nodes : ndarray, shape (nElem, nNodesPerElem)
        sensitivity_raw, sensitivity_volume_normalised : ndarray or None
            Shape (nBlocks,) each; None if sensitivity was not computed
            this iteration (see FEMTIC's ``doesCalculateSensitivity()``).

    Raises
    ------
    KeyError
        If no ``/model`` group (or root-level fallback content) is found.

    Author: Claude Sonnet 5 (Anthropic), 2026-09-13
    """
    try:
        import h5py
    except Exception as exc:  # pragma: no cover
        raise ImportError("read_results_model requires the 'h5py' package.") from exc

    h5, should_close = _h5_open_source(source, h5py)
    try:
        g = _h5_find_group(h5, h5py, "model", ("metadata", "blocks"))
        if g is None:
            raise KeyError(
                f"read_results_model: no '/model' group (or root-level "
                f"model content) found in {getattr(h5, 'filename', source)!r}."
            )

        meta = dict(g["metadata"].attrs)
        result = {
            "iterNum": int(meta.get("iterNum", -1)),
            "nElem": int(meta.get("nElem", 0)),
            "nNodes": int(meta.get("nNodes", 0)),
            "nBlocks": int(meta.get("nBlocks", 0)),
            "anisotropic": bool(meta["anisotropic"]) if "anisotropic" in meta else None,
            "element_block_map": np.asarray(g["element_block_map"][()]),
            "blocks": np.asarray(g["blocks"][()]),
            "node_coords": np.asarray(g["mesh/node_coords"][()]),
            "elem_nodes": np.asarray(g["mesh/elem_nodes"][()]),
            "sensitivity_raw": (
                np.asarray(g["sensitivity/raw"][()]) if "sensitivity" in g else None
            ),
            "sensitivity_volume_normalised": (
                np.asarray(g["sensitivity/volume_normalised"][()])
                if "sensitivity" in g else None
            ),
        }
    finally:
        if should_close:
            h5.close()

    if out:
        sens_note = ", sensitivity present" if result["sensitivity_raw"] is not None else ""
        aniso_note = (
            f", anisotropic={result['anisotropic']}" if result["anisotropic"] is not None else ""
        )
        print(f"read_results_model: iter {result['iterNum']}: "
              f"{result['nElem']} elements, {result['nNodes']} nodes, "
              f"{result['nBlocks']} blocks{aniso_note}{sens_note}")

    return result


def read_results_data(source, *, out: bool = True, decode_labels: bool = True) -> dict:
    """
    Read the ``/data`` group of a ``results_iterX.h5`` file (or an older,
    standalone ``data_iterX.h5``).

    Parameters
    ----------
    source : str, Path, h5py.File, or h5py.Group
        Path to a ``results_iterX.h5``/``data_iterX.h5`` file, or an
        already-open h5py file/group.
    out : bool, optional
        Print a one-line summary (including a warning if `nMissingCalc` > 0).
    decode_labels : bool, optional
        If True (default), also return human-readable ``datatype_name`` and
        ``component_name`` string arrays alongside the raw integer codes
        (see :data:`RESULTS_H5_DATATYPE_NAMES` / :data:`RESULTS_H5_COMPONENT_NAMES`).

    Returns
    -------
    dict with keys:
        iterNum, nRows, nMissingCalc : int
            `nMissingCalc` counts rows where no PE reported a calculated
            value for that datum (should be 0 in a normal run -- see the
            WARNING logged by FEMTIC itself when this is nonzero).
        rows : structured ndarray, shape (nRows,)
            Fields: freq, datatype, site_id, site_x, site_y, site_z,
            re_val, im_val, re_err, im_err, cal_re, cal_im, component.
            `re_val`/`im_val` and `re_err`/`im_err` are the OBSERVED value
            and its standard error; `cal_re`/`cal_im` are the CALCULATED
            (forward-modelled) response for the same datum -- added
            2026-09-13, absent from pre-2026-09-13 ``data_iterX.h5`` files
            (in which case this function raises, since `cal_re`/`cal_im`
            cannot be reconstructed after the fact).
        datatype_name, component_name : ndarray of str, shape (nRows,)
            Only present if `decode_labels` is True and `nRows` > 0.

    Notes
    -----
    `cal_im` is 0 (not NaN) for real-valued datatypes (APP_RES_AND_PHS, PT).
    NaN in `cal_re`/`cal_im` indicates a genuinely missing calculated value
    for that row (see `nMissingCalc` above), not a real-valued datatype.

    Author: Claude Sonnet 5 (Anthropic), 2026-09-13
    """
    try:
        import h5py
    except Exception as exc:  # pragma: no cover
        raise ImportError("read_results_data requires the 'h5py' package.") from exc

    h5, should_close = _h5_open_source(source, h5py)
    try:
        g = _h5_find_group(h5, h5py, "data", ("metadata",))
        if g is None:
            # Oldest fallback: pre-2026-09-13 data_iterX.h5 kept nRows/iterNum
            # as attributes directly on the file root, with no /metadata
            # subgroup at all.
            if "data" in h5 or ("nRows" in h5.attrs and "iterNum" in h5.attrs):
                g = h5
                meta = dict(h5.attrs)
            else:
                raise KeyError(
                    f"read_results_data: no '/data' group (or root-level "
                    f"data content) found in {getattr(h5, 'filename', source)!r}."
                )
        else:
            meta = dict(g["metadata"].attrs) if "metadata" in g else dict(g.attrs)

        nRows = int(meta.get("nRows", 0))
        nMissingCalc = int(meta.get("nMissingCalc", 0))
        rows = np.asarray(g["data"][()]) if "data" in g and nRows > 0 else np.array([])

        result = {
            "iterNum": int(meta.get("iterNum", -1)),
            "nRows": nRows,
            "nMissingCalc": nMissingCalc,
            "rows": rows,
        }

        if decode_labels and nRows > 0:
            if "cal_re" not in rows.dtype.names:
                raise KeyError(
                    "read_results_data: this file predates the 2026-09-13 "
                    "schema change and has no cal_re/cal_im fields -- "
                    "calculated response values cannot be reconstructed "
                    "after the fact. Re-run FEMTIC to get a results_iterX.h5 "
                    "with calculated values, or pass decode_labels=False and "
                    "ignore cal_re/cal_im if you only need the observed data."
                )
            dtype_name = np.array(
                [RESULTS_H5_DATATYPE_NAMES.get(int(d), f"UNKNOWN({d})") for d in rows["datatype"]]
            )
            component_name = np.array([
                (RESULTS_H5_COMPONENT_NAMES[dt][c]
                 if dt in RESULTS_H5_COMPONENT_NAMES and c < len(RESULTS_H5_COMPONENT_NAMES[dt])
                 else f"c{c}")
                for dt, c in zip(dtype_name, rows["component"])
            ])
            result["datatype_name"] = dtype_name
            result["component_name"] = component_name
    finally:
        if should_close:
            h5.close()

    if out:
        warn = f"  WARNING: {nMissingCalc} row(s) missing a calculated value!" if result["nMissingCalc"] else ""
        print(f"read_results_data: iter {result['iterNum']}: {result['nRows']} rows{warn}")

    return result


def read_results_distortion(source, *, out: bool = True) -> dict | None:
    """
    Read the ``/distortion`` group of a ``results_iterX.h5`` file.

    There is no pre-2026-09-13 equivalent (distortion parameters were only
    ever written to the text file ``distortion_iterX.dat``, not embedded in
    HDF5) -- this returns ``None`` whenever the group is absent, which
    covers both older files and runs with distortion estimation disabled.

    Parameters
    ----------
    source : str, Path, h5py.File, or h5py.Group
        Path to a ``results_iterX.h5`` file, or an already-open h5py
        file/group.
    out : bool, optional
        Print a one-line summary.

    Returns
    -------
    dict or None
        None if the run had no distortion estimation (or predates this
        group). Otherwise, a dict with keys:
            iterNum, nRows : int
            type : int
                Raw ``AnalysisControl::TypeOfDistortion`` code.
            type_name : str
                Human-readable name, see
                :data:`RESULTS_H5_DISTORTION_TYPE_NAMES`.
            rows : structured ndarray, shape (nRows,)
                Fields: site_id, param1, param2, param3, param4, isFixed.
                Meaning of param1..4 depends on `type_name` -- mirrors
                ``ObservedData::outputDistortionParams()``
                (``distortion_iterX.dat``) field-for-field:
                    ESTIMATE_DISTORTION_MATRIX_DIFFERENCE: Cxx, Cxy, Cyx, Cyy
                    ESTIMATE_GAINS_AND_ROTATIONS: ExGain, EyGain,
                        ExRotation(deg), EyRotation(deg)
                    ESTIMATE_GAINS_ONLY: ExGain, EyGain; param3/4 unused (0.0)

    Author: Claude Sonnet 5 (Anthropic), 2026-09-13
    """
    try:
        import h5py
    except Exception as exc:  # pragma: no cover
        raise ImportError("read_results_distortion requires the 'h5py' package.") from exc

    h5, should_close = _h5_open_source(source, h5py)
    try:
        g = _h5_find_group(h5, h5py, "distortion", ("metadata", "params"))
        if g is None:
            result = None
        else:
            meta = dict(g["metadata"].attrs)
            nRows = int(meta.get("nRows", 0))
            typeCode = int(meta.get("type", -1))
            rows = np.asarray(g["params"][()]) if "params" in g and nRows > 0 else np.array([])
            result = {
                "iterNum": int(meta.get("iterNum", -1)),
                "nRows": nRows,
                "type": typeCode,
                "type_name": RESULTS_H5_DISTORTION_TYPE_NAMES.get(typeCode, f"UNKNOWN({typeCode})"),
                "rows": rows,
            }
    finally:
        if should_close:
            h5.close()

    if out:
        if result is None:
            print("read_results_distortion: no /distortion group in this file "
                  "(distortion estimation disabled, or a pre-2026-09-13 file).")
        else:
            print(f"read_results_distortion: iter {result['iterNum']}: "
                  f"{result['nRows']} rows, type={result['type_name']}")

    return result


def read_results_hdf5(h5_path: str | Path, *, out: bool = True) -> dict:
    """
    Read a full ``results_iterX.h5`` file: model, data, and (if present)
    distortion, in one call, opening the file only once.

    Also accepts older, standalone ``model_iterX.h5``/``data_iterX.h5``
    files for backward compatibility (see :func:`read_results_model` /
    :func:`read_results_data`); in that case whichever group is absent from
    the given file comes back as None.

    Parameters
    ----------
    h5_path : str or Path
        Path to the ``results_iterX.h5`` (or legacy ``model_iterX.h5`` /
        ``data_iterX.h5``) file.
    out : bool, optional
        Print a one-line summary (and forward `out` to each sub-reader).

    Returns
    -------
    dict with keys "model", "data", "distortion"
        Each is either the corresponding sub-reader's return value, or
        None if that group is not present in this file.

    Raises
    ------
    FileNotFoundError
        If `h5_path` does not exist.

    Author: Claude Sonnet 5 (Anthropic), 2026-09-13
    """
    try:
        import h5py
    except Exception as exc:  # pragma: no cover
        raise ImportError("read_results_hdf5 requires the 'h5py' package.") from exc

    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"read_results_hdf5: {h5_path} not found.")

    with h5py.File(h5_path, "r") as h5:
        model = read_results_model(h5, out=out) if _h5_find_group(h5, h5py, "model", ("metadata", "blocks")) is not None else None
        data = read_results_data(h5, out=out) if _h5_find_group(h5, h5py, "data", ("metadata",)) is not None else None
        distortion = read_results_distortion(h5, out=out)

    if out:
        print(f"read_results_hdf5: {h5_path} -> "
              f"model={'yes' if model is not None else 'no'}, "
              f"data={'yes' if data is not None else 'no'}, "
              f"distortion={'yes' if distortion is not None else 'no'}")

    return {"model": model, "data": data, "distortion": distortion}


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
    float_fmt: str = "{:.6g}",
    *,
    block_fmt: str = "v4",
    region_rhoYY: np.ndarray | None = None,
    region_rhoZZ: np.ndarray | None = None,
    region_aniso_type: np.ndarray | None = None,
    region_strike: np.ndarray | None = None,
    region_dip: np.ndarray | None = None,
    region_slant: np.ndarray | None = None,
    region_fix_flags: np.ndarray | None = None,
) -> None:
    """Write a FEMTIC resistivity_block_iterX.dat from region arrays.

    Supports both **v4** (isotropic-only) and **v5** (anisotropic) formats.

    Parameters
    ----------
    out_path : str
    nelem, nreg : int
    region_of_elem : ndarray, shape (nelem,)
    region_rho, region_rho_lower, region_rho_upper : ndarray, shape (nreg,)
        For v5 these correspond to rhoXX, lo, hi.
    region_n : ndarray, shape (nreg,)
        v4 only (weighting constant n). Ignored for v5.
    region_flag : ndarray, shape (nreg,)
        Fixed-block flag (0 = free, 1 = fixed).
    float_fmt : str
        Python format string for floats (v4 only).
    block_fmt : ``"v4"`` or ``"v5"``
        Output format.  When ``"v5"``, the keyword-only arrays below are used.
    region_rhoYY, region_rhoZZ : ndarray, shape (nreg,), optional
        v5 principal resistivities. Defaults to rho (isotropic) if not supplied.
    region_aniso_type : ndarray of int, shape (nreg,), optional
        0 = ISO, 1 = TI, 2 = GA. Defaults to 0 (all isotropic).
    region_strike, region_dip, region_slant : ndarray, shape (nreg,), optional
        Euler angles in degrees. Defaults to 0.0.
    region_fix_flags : ndarray of int, shape (nreg, 6), optional
        Per-parameter fix flags for v5:
        columns → [fix_rhoXX, fix_rhoYY, fix_rhoZZ, fix_strike, fix_dip, fix_slant].
        Defaults to all 0 (unfixed).
    """
    region_of_elem   = np.asarray(region_of_elem, dtype=int)
    region_rho       = np.asarray(region_rho, dtype=float)
    region_rho_lower = np.asarray(region_rho_lower, dtype=float)
    region_rho_upper = np.asarray(region_rho_upper, dtype=float)
    region_n         = np.asarray(region_n, dtype=float)
    region_flag      = np.asarray(region_flag, dtype=int)

    if region_of_elem.shape[0] != nelem:
        raise ValueError("region_of_elem length does not match nelem.")
    for arr in [region_rho, region_rho_lower, region_rho_upper, region_n, region_flag]:
        if arr.shape[0] != nreg:
            raise ValueError("Region arrays must all have length nreg.")

    with open(out_path, "w") as f:
        f.write(f"{nelem:d} {nreg:d}\n")
        for ie in range(nelem):
            f.write(f"{ie:d} {int(region_of_elem[ie]):d}\n")

        if block_fmt == "v5":
            _rhoYY  = np.asarray(region_rhoYY,  dtype=float) if region_rhoYY  is not None else region_rho.copy()
            _rhoZZ  = np.asarray(region_rhoZZ,  dtype=float) if region_rhoZZ  is not None else region_rho.copy()
            _atype  = np.asarray(region_aniso_type, dtype=int) if region_aniso_type is not None else np.zeros(nreg, dtype=int)
            _strike = np.asarray(region_strike, dtype=float) if region_strike is not None else np.zeros(nreg)
            _dip    = np.asarray(region_dip,    dtype=float) if region_dip    is not None else np.zeros(nreg)
            _slant  = np.asarray(region_slant,  dtype=float) if region_slant  is not None else np.zeros(nreg)
            _fix    = np.asarray(region_fix_flags, dtype=int) if region_fix_flags is not None else np.zeros((nreg, 6), dtype=int)
            if _fix.ndim == 1:
                _fix = _fix.reshape(nreg, 6)
            for ireg in range(nreg):
                d = dict(
                    ireg=ireg,
                    aniso_type=int(_atype[ireg]),
                    rhoXX=float(region_rho[ireg]),
                    rhoYY=float(_rhoYY[ireg]),
                    rhoZZ=float(_rhoZZ[ireg]),
                    strike=float(_strike[ireg]),
                    dip=float(_dip[ireg]),
                    slant=float(_slant[ireg]),
                    rho_lo=float(region_rho_lower[ireg]),
                    rho_hi=float(region_rho_upper[ireg]),
                    flag=int(region_flag[ireg]),
                    fix_rhoXX=int(_fix[ireg, 0]),
                    fix_rhoYY=int(_fix[ireg, 1]),
                    fix_rhoZZ=int(_fix[ireg, 2]),
                    fix_strike=int(_fix[ireg, 3]),
                    fix_dip=int(_fix[ireg, 4]),
                    fix_slant=int(_fix[ireg, 5]),
                )
                f.write(_format_region_line_v5(d) + "\n")
        else:
            for ireg in range(nreg):
                rho  = float_fmt.format(region_rho[ireg])
                rmin = float_fmt.format(region_rho_lower[ireg])
                rmax = float_fmt.format(region_rho_upper[ireg])
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
        Exactly 4 tokens as read from observe.dat: ``[name, x, y, z]``.

    Returns
    -------
    meta : dict
        Contains:

        - ``name``       : str — station name
        - ``xyz``        : ndarray shape (3,) or None — Cartesian coordinates
          in the **FEMTIC model frame** (metres).  The z-axis points
          **downward**, so ``xyz[2]`` is positive for points below the datum
          and **negative** for surface/near-surface stations.
          Relation to geodetic elevation: ``xyz[2] == -elev_m``.
        - ``elev_m``     : float or None — geodetic elevation in metres above
          datum (z-up convention), derived as ``-xyz[2]``.  Provided as a
          convenience so callers need not remember the sign flip.
        - ``raw_tokens`` : list[str] — verbatim token list

    Notes
    -----
    FEMTIC uses a right-handed coordinate system with **z positive downward**
    (depth), consistent with the FEM mesh stored in ``mesh.dat``.  EDI files
    store elevation with the opposite sign (positive upward).  When building
    an ``observe.dat`` site header from EDI metadata the correct mapping is::

        z_femtic = -elev_m          # e.g. elev_m=500 m → z=-500 m

    Any code that writes the site header must apply this negation; omitting it
    silently places the station at an incorrect (sub-surface) position.
    """
    name = str(site_header_tokens[0])
    xyz = None
    try:
        xyz = np.asarray(
            [float(site_header_tokens[1]),
             float(site_header_tokens[2]),
             float(site_header_tokens[3])],
            dtype=float,
        )
    except Exception:
        xyz = None
    elev_m = float(-xyz[2]) if xyz is not None else None
    return {"name": name, "xyz": xyz, "elev_m": elev_m, "raw_tokens": site_header_tokens[:]}


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
        - ``name``   : str — site name from the header
        - ``xyz``    : ndarray shape (3,) or None — site Cartesian coordinates
          in the **FEMTIC model frame** (metres).  ``xyz[2]`` follows the
          **z-positive-downward** convention used internally by FEMTIC:
          surface stations have ``xyz[2] < 0``.  Relation to geodetic
          elevation: ``xyz[2] == -elev_m``.
        - ``elev_m`` : float or None — geodetic elevation above datum in
          metres (z-up), i.e. ``-xyz[2]``.  Provided as a convenience alias
          so callers working with EDI-style metadata do not need to negate
          ``xyz[2]`` manually.
        - ``nfreq``  : int — number of frequencies
        - ``freq``   : ndarray (nfreq,) — frequencies in Hz
        - ``per``    : ndarray (nfreq,) — periods in s (= 1 / freq)
        - ``data``   : ndarray (nfreq, dat_length) — raw observed values
        - ``error``  : ndarray (nfreq, dat_length) — raw per-component std devs
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
            # elev_m: geodetic elevation (z-up, metres).  FEMTIC stores z with
            # the opposite sign (z positive downward), so elev_m == -xyz[2].
            # Provided as a convenience alias for EDI-style workflows.
            "elev_m": meta["elev_m"],
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


def summarise_observe_dat(
    path_or_parsed: "str | Path | dict[str, Any]",
    *,
    out: bool = True,
) -> dict:
    """Summarise the data content of a FEMTIC ``observe.dat``.

    Reports the number of sites, frequencies, and data values per observation
    type (MT, VTF, PT), plus global totals.

    Parameters
    ----------
    path_or_parsed : str, Path, or already-parsed dict
        Either a filesystem path to an ``observe.dat`` (will be parsed with
        ``compute_mt_derived=False`` for speed) or a dict previously returned
        by :func:`read_observe_dat`.
    out : bool
        If True, print a formatted summary table to stdout.

    Returns
    -------
    dict with keys:

    ``path`` (str),
    ``blocks`` (list[dict]) — one entry per obs-type block, each with:

        ``obs_type`` (str),
        ``n_sites`` (int),
        ``dat_length`` (int)  — number of data values per frequency per site,
        ``n_freq_per_site`` (list[int])  — nfreq for each site,
        ``n_freq_total`` (int)  — sum of nfreq across all sites,
        ``n_data_total`` (int)  — n_freq_total × dat_length

    ``n_sites_total`` (int),
    ``n_freq_total`` (int),
    ``n_data_total`` (int)   — total length of the data vector
    """
    if isinstance(path_or_parsed, dict):
        parsed = path_or_parsed
        src_path = parsed.get("path", "<in-memory>")
    else:
        parsed = read_observe_dat(
            path_or_parsed,
            compute_mt_derived=False,
            bootstrap_n=0,
        )
        src_path = str(path_or_parsed)

    block_summaries: list[dict] = []
    for blk in parsed["blocks"]:
        obs_type  = blk["obs_type"]
        sites     = blk["sites"]
        dat_len   = _OBS_DATALEN.get(obs_type, 0)
        nfreqs    = [s["nfreq"] for s in sites]
        n_freq_total = sum(nfreqs)
        block_summaries.append({
            "obs_type":       obs_type,
            "n_sites":        len(sites),
            "dat_length":     dat_len,
            "n_freq_per_site": nfreqs,
            "n_freq_total":   n_freq_total,
            "n_data_total":   n_freq_total * dat_len,
        })

    n_sites_total = sum(b["n_sites"]      for b in block_summaries)
    n_freq_total  = sum(b["n_freq_total"] for b in block_summaries)
    n_data_total  = sum(b["n_data_total"] for b in block_summaries)

    summary = {
        "path":          src_path,
        "blocks":        block_summaries,
        "n_sites_total": n_sites_total,
        "n_freq_total":  n_freq_total,
        "n_data_total":  n_data_total,
    }

    if out:
        _print_observe_summary(summary)

    return summary


def _print_observe_summary(s: dict) -> None:
    """Print a formatted observe.dat statistics table."""
    fname = Path(s["path"]).name
    print(f"\n{'─' * 60}")
    print(f"  File          : {fname}")
    print(f"  ┌──────────┬────────┬────────┬──────────┬──────────┐")
    print(f"  │ Obs type │ Sites  │ d/freq │ Freq tot │ Data tot │")
    print(f"  ├──────────┼────────┼────────┼──────────┼──────────┤")
    for b in s["blocks"]:
        nfreqs = b["n_freq_per_site"]
        freq_range = (
            f"{min(nfreqs)}–{max(nfreqs)}"
            if len(set(nfreqs)) > 1
            else str(nfreqs[0]) if nfreqs else "—"
        )
        print(
            f"  │ {b['obs_type']:<8s} │ {b['n_sites']:>6d} │ "
            f"{b['dat_length']:>6d} │ {b['n_freq_total']:>8d} │ "
            f"{b['n_data_total']:>8d} │"
        )
        # Per-site freq range on a sub-line if sites differ
        if len(set(nfreqs)) > 1:
            print(f"  │          │        │  nfreq range: {freq_range:<28s}│")
    print(f"  ├──────────┴────────┴────────┼──────────┼──────────┤")
    print(f"  │ Total                      │ {s['n_freq_total']:>8d} │ {s['n_data_total']:>8d} │")
    print(f"  └────────────────────────────┴──────────┴──────────┘")
    print(f"  Sites total   : {s['n_sites_total']:>8d}")


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

    **z-convention (z positive downward)**
        FEMTIC uses a coordinate system in which the z-axis points
        **downward**.  Surface and near-surface stations therefore have a
        *negative* z coordinate in the site header.  Geodetic elevation
        (z-up, as stored in EDI ``ELEV=`` fields) must be **negated** before
        writing::

            z_femtic = -elev_m

        This function does not rewrite site header lines (they are preserved
        verbatim), so it cannot correct a wrong sign introduced upstream.  A
        warning is issued if any site header is detected to have a positive z
        value, which is almost certainly a sign error for MT survey stations.
    """
    import warnings

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
            # Defensive check: warn if z coordinate appears to be positive
            # (z-up / geodetic sign), which indicates a likely negation error.
            _toks = shl.split()
            if len(_toks) >= 4:
                try:
                    _z_val = float(_toks[3])
                    if _z_val > 0.0:
                        import warnings
                        warnings.warn(
                            f"write_observe_dat: site '{_toks[0]}' has z={_z_val:g} m "
                            f"(positive z in FEMTIC = below datum).  If this value "
                            f"came from geodetic elevation, it must be negated: "
                            f"z_femtic = -elev_m.",
                            stacklevel=2,
                        )
                except (ValueError, IndexError):
                    pass
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


# ---------------------------------------------------------------------------
# EDI → observe.dat conversion
# ---------------------------------------------------------------------------

#: Conversion factor: mV km⁻¹ nT⁻¹  →  SI Ω  (multiply EDI Z by this)
#: FEMTIC observe.dat stores Z in SI units (Ω); EDI files use MT field units.
#: Z_SI = Z_MT * mu0 * 1e3
_Z_MT_TO_SI: float = 4.0e-7 * np.pi * 1.0e3   # ≈ 1.2566e-3


def _edi_Z_to_observe_row(
    Z: np.ndarray,
    Z_err: np.ndarray | None,
    err_kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert one frequency's complex 2×2 Z tensor to an 8-element flat row.

    Parameters
    ----------
    Z : ndarray, shape (2, 2), complex
        Impedance tensor in **SI Ω** (after unit conversion from EDI field units).
    Z_err : ndarray, shape (2, 2), complex or real, or None
        Error tensor.  If complex, real/imag parts are std(Re)/std(Im).
        If real-valued, treated as the same std for both real and imaginary.
        If ``None``, a zeros row is returned.
    err_kind : {"var", "std"}
        How errors are stored in the EDI dict.  ``"var"`` → take sqrt first;
        ``"std"`` → use directly.

    Returns
    -------
    data_row : ndarray, shape (8,)
        [Zxx_re, Zxx_im, Zxy_re, Zxy_im, Zyx_re, Zyx_im, Zyy_re, Zyy_im]
    err_row : ndarray, shape (8,)
        Same layout, standard deviations.
    """
    order = [(0, 0), (0, 1), (1, 0), (1, 1)]
    data_row = np.empty(8, dtype=float)
    err_row = np.zeros(8, dtype=float)

    for k, (i, j) in enumerate(order):
        data_row[2 * k]     = Z[i, j].real
        data_row[2 * k + 1] = Z[i, j].imag

    if Z_err is not None:
        Ze = np.asarray(Z_err, dtype=complex if np.iscomplexobj(Z_err) else float)
        for k, (i, j) in enumerate(order):
            if Ze.dtype == complex or np.iscomplexobj(Ze):
                re_std = Ze[i, j].real
                im_std = Ze[i, j].imag
            else:
                re_std = Ze[i, j].real
                im_std = re_std
            if err_kind == "var":
                re_std = float(np.sqrt(np.maximum(re_std, 0.0)))
                im_std = float(np.sqrt(np.maximum(im_std, 0.0)))
            err_row[2 * k]     = float(re_std)
            err_row[2 * k + 1] = float(im_std)

    return data_row, err_row


def edi_list_to_observe_dat(
    edi_list: list[dict[str, Any]],
    path: str | Path,
    *,
    x_key: str = "x_m",
    y_key: str = "y_m",
    fallback_x: float = 0.0,
    fallback_y: float = 0.0,
    obs_type: str = "MT",
    preamble: str | list[str] | None = None,
) -> None:
    """Write a list of EDI site dicts to a FEMTIC-style ``observe.dat`` file.

    Each EDI dict is the structure returned by :func:`data_proc.load_edi`.
    Only MT impedance data (``Z`` / ``Z_err``) is currently supported;
    tipper and phase-tensor blocks are silently ignored.

    Parameters
    ----------
    edi_list : list of dict
        One dict per site, each as returned by ``data_proc.load_edi``.
        Required keys per dict:

        - ``freq``    : ndarray (nfreq,) — frequencies in Hz
        - ``Z``       : ndarray (nfreq, 2, 2), complex — impedance in
          **MT field units** (mV km⁻¹ nT⁻¹) as stored by ``load_edi``.
        - ``Z_err``   : ndarray (nfreq, 2, 2) or None — errors (variance or
          std depending on ``err_kind``).
        - ``err_kind``: ``"var"`` (default from ``load_edi``) or ``"std"``.
        - ``station`` : str — site name written to the header token.
        - ``elev_m``  : float or None — geodetic elevation (m above datum,
          **z-up**).  Negated automatically: ``z_femtic = -elev_m``.

        Optional position keys (see ``x_key`` / ``y_key``):

        - Model-frame x and y coordinates in metres.  If absent, ``fallback_x``
          / ``fallback_y`` are used and a warning is issued.

    path : str or pathlib.Path
        Output file path.  Overwrites any existing file.
    x_key : str
        Key in each EDI dict holding the model-frame x coordinate (metres).
        Default is ``"x_m"``.  Common alternatives: ``"x"``, ``"east_m"``.
    y_key : str
        Key in each EDI dict holding the model-frame y coordinate (metres).
        Default is ``"y_m"``.  Common alternatives: ``"y"``, ``"north_m"``.
    fallback_x, fallback_y : float
        Values used when ``x_key`` / ``y_key`` are absent from an EDI dict.
        A :class:`UserWarning` is issued per affected site.
    obs_type : str
        Block type to write.  Currently only ``"MT"`` is supported.
    preamble : str or list of str or None
        Optional text lines written verbatim before the data block(s).
        Useful for adding comments or FEMTIC control-file references.
        If a single string, it is split on ``\\n``.  Each line will have a
        trailing newline appended if absent.

    Returns
    -------
    None
        The file is written to ``path``; nothing is returned.

    Raises
    ------
    ValueError
        If ``obs_type`` is not ``"MT"`` or if any site has no valid
        frequencies after dropping NaN/sentinel values.
    NotImplementedError
        If ``obs_type`` other than ``"MT"`` is requested (VTF/PT not yet
        implemented).

    Notes
    -----
    **Unit conversion**
        ``load_edi`` returns ``Z`` in MT field units (mV km⁻¹ nT⁻¹).
        FEMTIC ``observe.dat`` stores Z in SI units (Ω).  The conversion
        applied here is::

            Z_SI = Z_MT × μ₀ × 10³       (μ₀ = 4π × 10⁻⁷ H m⁻¹)

        i.e. ``Z_SI ≈ Z_MT × 1.2566 × 10⁻³``.

    **z-convention**
        EDI elevation ``elev_m`` is geodetic (positive upward).  FEMTIC uses
        z positive downward, so the site header z is written as ``-elev_m``.
        Sites with ``elev_m = None`` are placed at ``z = 0.0`` with a warning.

    **Frequency ordering**
        Frequencies are written in the order they appear in ``edi["freq"]``.
        Pass ``freq_order="inc"`` to :func:`data_proc.load_edi` for ascending
        frequency (FEMTIC convention).

    **FT convention**
        The function assumes all EDI dicts are already in the standard
        e⁻ⁱωᵗ convention (as guaranteed by ``load_edi`` after Phoenix
        correction).  No additional sign correction is applied.

    Examples
    --------
    >>> import data_proc as dp
    >>> import femtic as fem
    >>>
    >>> edi_files = ["SITE01.edi", "SITE02.edi"]
    >>> edis = [dp.load_edi(f, manufacturer="metronix") for f in edi_files]
    >>>
    >>> # Attach model-frame XY (e.g. from a projection step)
    >>> for edi, (x, y) in zip(edis, [(1000.0, 2000.0), (3000.0, 4000.0)]):
    ...     edi["x_m"] = x
    ...     edi["y_m"] = y
    >>>
    >>> fem.edi_list_to_observe_dat(edis, "observe.dat")
    """
    import warnings

    obs_type = str(obs_type).upper()
    if obs_type != "MT":
        raise NotImplementedError(
            f"edi_list_to_observe_dat: obs_type={obs_type!r} is not yet supported. "
            f"Only 'MT' is currently implemented."
        )

    out = Path(path)
    lines: list[str] = []

    # ---- preamble -----------------------------------------------------------
    if preamble is not None:
        if isinstance(preamble, str):
            preamble_lines: list[str] = preamble.split("\n")
        else:
            preamble_lines = list(preamble)
        for pl in preamble_lines:
            lines.append(pl if pl.endswith("\n") else pl + "\n")

    # ---- block header -------------------------------------------------------
    n_sites = len(edi_list)
    lines.append(f"{obs_type}    {n_sites}\n")

    # ---- sites --------------------------------------------------------------
    for edi in edi_list:
        # -- station name
        name = str(edi.get("station") or "UNKNOWN")

        # -- model-frame XY
        if x_key in edi:
            x_m = float(edi[x_key])
        else:
            warnings.warn(
                f"edi_list_to_observe_dat: site '{name}' has no key '{x_key}'; "
                f"using fallback_x={fallback_x}.",
                UserWarning,
                stacklevel=2,
            )
            x_m = float(fallback_x)

        if y_key in edi:
            y_m = float(edi[y_key])
        else:
            warnings.warn(
                f"edi_list_to_observe_dat: site '{name}' has no key '{y_key}'; "
                f"using fallback_y={fallback_y}.",
                UserWarning,
                stacklevel=2,
            )
            y_m = float(fallback_y)

        # -- elevation → FEMTIC z (z positive downward → negate)
        # Use explicit None checks: elev_m=0.0 (sea-level station) is valid
        # and must not be treated as absent by a falsy "or" chain.
        elev_m = edi.get("elev_m")
        if elev_m is None:
            elev_m = edi.get("elev")
        if elev_m is None:
            warnings.warn(
                f"edi_list_to_observe_dat: site '{name}' has no elevation; "
                f"using z=0.0 (surface at datum).",
                UserWarning,
                stacklevel=2,
            )
            z_femtic = 0.0
        else:
            z_femtic = -float(elev_m)   # EDI z-up → FEMTIC z-down

        # -- frequencies and impedance
        freq = np.asarray(edi["freq"], dtype=float).ravel()
        Z_mt = np.asarray(edi["Z"], dtype=np.complex128)   # (nfreq, 2, 2), mV/km/nT
        Z_err_raw = edi.get("Z_err")
        err_kind = str(edi.get("err_kind", "var"))

        # Convert Z from MT field units to SI Ω
        Z_si = Z_mt * _Z_MT_TO_SI
        Z_err_si = (np.asarray(Z_err_raw, dtype=complex) * _Z_MT_TO_SI
                    if Z_err_raw is not None else None)

        # Guard: drop any frequency rows where Z is non-finite
        finite_mask = np.all(
            np.isfinite(Z_si.real) & np.isfinite(Z_si.imag),
            axis=(1, 2),
        )
        if not np.any(finite_mask):
            warnings.warn(
                f"edi_list_to_observe_dat: site '{name}' has no finite Z values; "
                f"skipping.",
                UserWarning,
                stacklevel=2,
            )
            continue
        freq = freq[finite_mask]
        Z_si = Z_si[finite_mask]
        if Z_err_si is not None:
            Z_err_si = Z_err_si[finite_mask]

        nfreq = len(freq)

        # -- site header line: name  x  y  z
        lines.append(f"{name}    {x_m:.4f}    {y_m:.4f}    {z_femtic:.4f}\n")
        lines.append(f"{nfreq}\n")

        # -- data rows
        for fi in range(nfreq):
            Z_err_f = Z_err_si[fi] if Z_err_si is not None else None
            data_row, err_row = _edi_Z_to_observe_row(Z_si[fi], Z_err_f, err_kind)
            row_vals: list[float] = [float(freq[fi])] + data_row.tolist() + err_row.tolist()
            lines.append(_format_row(row_vals))

    # ---- END ----------------------------------------------------------------
    lines.append("END\n")

    out.write_text("".join(lines), encoding="utf-8")


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
# SECTION: Model-local coordinate helpers
# ============================================================================
# These functions bridge the pure geographic conversions in util.py (which
# know nothing about FEMTIC) and the script-level configuration in
# femtic_mod_plot.py (which holds the mesh-origin UTM coordinates).
#
# Callers pass origin_E / origin_N explicitly so the functions remain
# stateless and testable independent of any script globals.
# ============================================================================

def _utl():
    """Lazy import of util to avoid circular dependencies."""
    import util as _u
    return _u


def utm_to_model(E_m: float, N_m: float,
                 origin_E: float, origin_N: float) -> tuple[float, float]:
    """Translate UTM metres to model-local metres.

    The FEMTIC mesh origin is assumed to coincide with (origin_E, origin_N)
    in the UTM frame of the survey zone.  Model-local axes are aligned with
    UTM east/north.

    Parameters
    ----------
    E_m, N_m         : UTM easting and northing [m]
    origin_E, origin_N : UTM coordinates of the mesh centre [m]

    Returns
    -------
    x_m, y_m : model-local metres

    VR 2026-05-23, Claude Sonnet 4.6 (Anthropic)
    """
    return float(E_m) - float(origin_E), float(N_m) - float(origin_N)


def latlon_to_model(lat: float, lon: float,
                    zone: int, northern: bool,
                    origin_E: float, origin_N: float) -> tuple[float, float]:
    """Convert WGS-84 geographic coordinates to model-local metres.

    Chains ``util.latlon_to_utm_zn`` → ``fem.utm_to_model``.

    Parameters
    ----------
    lat, lon           : decimal degrees (positive = N / E)
    zone, northern     : UTM zone and hemisphere (from ``util.utm_zone_from_latlon``)
    origin_E, origin_N : UTM coordinates of the mesh centre [m]

    Returns
    -------
    x_m, y_m : model-local metres

    VR 2026-05-23, Claude Sonnet 4.6 (Anthropic)
    """
    E_m, N_m = _utl().latlon_to_utm_zn(lat, lon, zone, northern)
    return utm_to_model(E_m, N_m, origin_E, origin_N)


def parse_pos_crs(raw) -> tuple:
    """Parse a position spec into ``(value, crs_str)``.

    Accepts:
      scalar (int / float)   → ``(float, "model")``
      ``(value, "crs")``     → ``(value, crs)``

    Used by ``resolve_pos_x``, ``resolve_pos_y``, ``resolve_pos_point``.

    VR 2026-05-23, Claude Sonnet 4.6 (Anthropic)
    """
    if isinstance(raw, (int, float)):
        return float(raw), "model"
    try:
        val, crs = raw
        return val, str(crs)
    except (TypeError, ValueError):
        raise ValueError(
            f"parse_pos_crs: {raw!r} must be a scalar or (value, 'crs') tuple."
        )


def resolve_pos_x(raw,
                  zone: int, northern: bool,
                  origin_E: float, origin_N: float,
                  origin_lat: float, origin_lon: float) -> float:
    """Resolve an x0 position spec (easting / NS curtain) to model-local metres.

    Parameters
    ----------
    raw        : scalar (model-local m) or ``(value, "crs")`` tuple where
                 ``crs`` is ``"model"``, ``"utm"``, or ``"latlon"``.
                 For ``"latlon"`` *value* is a **longitude** in decimal degrees;
                 ``origin_lat`` is used for the northing component.
    zone, northern     : UTM zone and hemisphere
    origin_E, origin_N : UTM coordinates of the mesh centre [m]
    origin_lat, origin_lon : geographic coordinates of the mesh centre [°]

    Returns
    -------
    x_m : model-local easting [m]

    VR 2026-05-23, Claude Sonnet 4.6 (Anthropic)
    """
    val, crs = parse_pos_crs(raw)
    if crs == "model":
        return float(val)
    if crs == "utm":
        x_m, _ = utm_to_model(float(val), origin_N, origin_E, origin_N)
        return x_m
    if crs == "latlon":
        # val = longitude; hold northing at origin latitude
        x_m, _ = latlon_to_model(origin_lat, float(val),
                                  zone, northern, origin_E, origin_N)
        return x_m
    raise ValueError(
        f"resolve_pos_x: unknown crs={crs!r}. Choose 'model', 'utm', or 'latlon'."
    )


def resolve_pos_y(raw,
                  zone: int, northern: bool,
                  origin_E: float, origin_N: float,
                  origin_lat: float, origin_lon: float) -> float:
    """Resolve a y0 position spec (northing / EW curtain) to model-local metres.

    Parameters
    ----------
    raw        : scalar (model-local m) or ``(value, "crs")`` tuple where
                 ``crs`` is ``"model"``, ``"utm"``, or ``"latlon"``.
                 For ``"latlon"`` *value* is a **latitude** in decimal degrees;
                 ``origin_lon`` is used for the easting component.
    zone, northern     : UTM zone and hemisphere
    origin_E, origin_N : UTM coordinates of the mesh centre [m]
    origin_lat, origin_lon : geographic coordinates of the mesh centre [°]

    Returns
    -------
    y_m : model-local northing [m]

    VR 2026-05-23, Claude Sonnet 4.6 (Anthropic)
    """
    val, crs = parse_pos_crs(raw)
    if crs == "model":
        return float(val)
    if crs == "utm":
        _, y_m = utm_to_model(origin_E, float(val), origin_E, origin_N)
        return y_m
    if crs == "latlon":
        # val = latitude; hold easting at origin longitude
        _, y_m = latlon_to_model(float(val), origin_lon,
                                  zone, northern, origin_E, origin_N)
        return y_m
    raise ValueError(
        f"resolve_pos_y: unknown crs={crs!r}. Choose 'model', 'utm', or 'latlon'."
    )


def resolve_pos_point(raw,
                      zone: int, northern: bool,
                      origin_E: float, origin_N: float) -> list[float]:
    """Resolve a plane-slice point [x, y, z] to model-local metres.

    Accepts:
      ``[x, y, z]``               plain list → model-local metres
      ``([x, y, z], "model")``    explicit model-local
      ``([E, N, z], "utm")``      UTM metres for E/N; z model-local
      ``([lon, lat, z], "latlon")``  decimal degrees for lon/lat; z model-local

    Parameters
    ----------
    raw                : point spec (see above)
    zone, northern     : UTM zone and hemisphere
    origin_E, origin_N : UTM coordinates of the mesh centre [m]

    Returns
    -------
    [x_m, y_m, z_m] : model-local metres (z positive-down, unchanged)

    VR 2026-05-23, Claude Sonnet 4.6 (Anthropic)
    """
    crs = "model"
    pt  = raw
    if (isinstance(raw, (list, tuple)) and len(raw) == 2
            and isinstance(raw[-1], str)):
        pt, crs = raw[0], raw[1]
    pt = list(pt)
    if len(pt) != 3:
        raise ValueError(
            f"resolve_pos_point: 'point' must have 3 elements, got {len(pt)}."
        )
    z_m = float(pt[2])
    if crs == "model":
        return [float(pt[0]), float(pt[1]), z_m]
    if crs == "utm":
        x_m, y_m = utm_to_model(float(pt[0]), float(pt[1]), origin_E, origin_N)
        return [x_m, y_m, z_m]
    if crs == "latlon":
        # pt = [lon_deg, lat_deg, z_m]
        x_m, y_m = latlon_to_model(float(pt[1]), float(pt[0]),
                                    zone, northern, origin_E, origin_N)
        return [x_m, y_m, z_m]
    raise ValueError(
        f"resolve_pos_point: unknown crs={crs!r}. Choose 'model', 'utm', or 'latlon'."
    )


def resolve_pos_two_point_profile(p1_raw, p2_raw,
                                   zone: int, northern: bool,
                                   origin_E: float, origin_N: float,
                                   z_top: float = 0.0,
                                   z_bot: float = 20000.0) -> dict:
    """Derive a vertical ``plane`` slice spec from two endpoint positions.

    The two endpoints define a vertical fence (curtain) section.  The
    horizontal midpoint of the segment becomes the plane anchor ``point``;
    the azimuth from *p1* to *p2* (clockwise from North) is the ``strike``;
    ``dip`` is fixed at 90° (strictly vertical plane).

    Each endpoint is a CRS-tagged position spec accepted by
    ``resolve_pos_point``.  Accepted forms:

    ``[x, y]``                    plain list → model-local metres
    ``([x, y], "model")``         explicit model-local
    ``([E, N], "utm")``           UTM metres
    ``([lon, lat], "latlon")``    decimal degrees (lon first, then lat)

    The *z* component of the anchor is set to ``z_top`` (the midpoint depth),
    and the panel vertical extent is controlled via ``zlim`` in the returned
    dict (``[z_top, z_bot]``).

    Parameters
    ----------
    p1_raw, p2_raw : position specs (see above)
    zone, northern : UTM zone and hemisphere
    origin_E, origin_N : UTM coordinates of the mesh centre [m]
    z_top : float
        Shallowest depth for the slice panel [m, z-positive-down].
        Default 0 (surface).
    z_bot : float
        Deepest depth for the slice panel [m, z-positive-down].
        Default 20 000 m.

    Returns
    -------
    dict
        A ``kind="plane"`` slice spec (resolved to model-local metres) with
        keys ``kind``, ``point``, ``strike``, ``dip``, ``zlim``,
        ``p1_model``, ``p2_model`` (the two resolved endpoint model-local
        [x, y] pairs, for reference / axis-limit computation).

    Raises
    ------
    ValueError
        If the two endpoints are coincident (zero-length segment).

    Notes
    -----
    The horizontal axis of the resulting panel runs along-strike from *p1*
    to *p2*.  Use ``invert_x=True`` in the slice spec to reverse the sense.

    VR 2026-06-19, Claude Sonnet 4.6 (Anthropic)
    """
    # ------------------------------------------------------------------
    # Resolve each endpoint to model-local [x, y] (ignore / discard z)
    # ------------------------------------------------------------------
    def _resolve_xy(raw) -> tuple[float, float]:
        """Accept 2-element (no z) or 3-element (with z) specs."""
        # normalise: allow ([x, y], "crs") as well as ([x, y, z], "crs")
        crs = "model"
        pt  = raw
        if (isinstance(raw, (list, tuple)) and len(raw) == 2
                and isinstance(raw[-1], str)):
            pt, crs = raw[0], raw[1]
        pt = list(pt)
        if len(pt) == 2:
            # inject a dummy z so resolve_pos_point can be reused
            pt_with_z = pt + [0.0]
        elif len(pt) == 3:
            pt_with_z = pt
        else:
            raise ValueError(
                f"resolve_pos_two_point_profile: endpoint must have 2 or 3 "
                f"elements, got {len(pt)}."
            )
        x_m, y_m, _ = resolve_pos_point(
            (pt_with_z, crs) if crs != "model" else pt_with_z,
            zone, northern, origin_E, origin_N,
        )
        return x_m, y_m

    x1, y1 = _resolve_xy(p1_raw)
    x2, y2 = _resolve_xy(p2_raw)

    dx = x2 - x1
    dy = y2 - y1
    seg_len = math.hypot(dx, dy)
    if seg_len < 1e-6:
        raise ValueError(
            "resolve_pos_two_point_profile: the two endpoints are coincident "
            f"(distance = {seg_len:.3g} m)."
        )

    # ------------------------------------------------------------------
    # Strike: azimuth of the segment, clockwise from North [deg]
    # FEMTIC / geology convention: strike = atan2(dx_east, dy_north)
    # model frame: x = East, y = North  →  dx = Δeasting, dy = Δnorthing
    # ------------------------------------------------------------------
    strike_deg = math.degrees(math.atan2(dx, dy)) % 360.0

    # Midpoint as plane anchor
    x_mid = 0.5 * (x1 + x2)
    y_mid = 0.5 * (y1 + y2)

    return {
        "kind":     "plane",
        "point":    [x_mid, y_mid, float(z_top)],
        "strike":   strike_deg,
        "dip":      90.0,
        "zlim":     [float(z_top), float(z_bot)],
        # stash resolved endpoints for downstream use (e.g. xlim computation)
        "p1_model": [x1, y1],
        "p2_model": [x2, y2],
    }


def resolve_slice_positions(slices: list,
                            zone: int, northern: bool,
                            origin_E: float, origin_N: float,
                            origin_lat: float, origin_lon: float,
                            *,
                            verbose: bool = False) -> list:
    """Return a new list of slice specs with all positions in model-local metres.

    Only horizontal position keys (``x0``, ``y0``, ``point``) are converted;
    ``z0`` and all other keys (``xlim``, ``ylim``, ``zlim``, ``title``,
    ``strike``, ``dip``, ``kind``) pass through unchanged.
    The original *slices* list is not mutated.

    For ``kind="profile"`` specs (two-point vertical fence definition), the
    spec is converted into a resolved ``kind="plane"`` dict with ``dip=90``
    before being appended.  See :func:`resolve_pos_two_point_profile` for
    the full endpoint CRS syntax.

    Parameters
    ----------
    slices                 : list of slice-spec dicts (from ``PLOT_SLICES``)
    zone, northern         : UTM zone and hemisphere
    origin_E, origin_N     : UTM coordinates of the mesh centre [m]
    origin_lat, origin_lon : geographic coordinates of the mesh centre [°]
    verbose                : print conversion details when True

    Returns
    -------
    list of slice-spec dicts with model-local float values for x0, y0, point

    VR 2026-05-23, Claude Sonnet 4.6 (Anthropic)
    Modified 2026-06-19, Claude Sonnet 4.6 (Anthropic) — added kind="profile"
    """
    resolved = []
    for i, spec in enumerate(slices):
        s    = dict(spec)
        kind = s.get("kind", "map")

        # Normalise reversed-direction aliases: "sn" → "ns", "we" → "ew".
        # Both set invert_x=True (unless the user already set it explicitly).
        if kind in ("sn", "we"):
            kind = "ns" if kind == "sn" else "ew"
            s["kind"]     = kind
            s["invert_x"] = not s.get("invert_x", False)

        # ----------------------------------------------------------------
        # kind = "profile" — two-point vertical fence  →  plane, dip=90
        # ----------------------------------------------------------------
        if kind == "profile":
            if "p1" not in s or "p2" not in s:
                raise ValueError(
                    f"slice[{i}] kind='profile' requires 'p1' and 'p2' keys."
                )
            z_top = float(s.pop("z_top", 0.0))
            z_bot = float(s.pop("z_bot", 20000.0))
            plane_spec = resolve_pos_two_point_profile(
                s.pop("p1"), s.pop("p2"),
                zone, northern, origin_E, origin_N,
                z_top=z_top, z_bot=z_bot,
            )
            # Merge any remaining per-panel overrides (xlim, invert_x, title …)
            # but do NOT let them clobber the geometry keys just derived.
            _geometry_keys = {"kind", "point", "strike", "dip", "zlim",
                               "p1_model", "p2_model"}
            for k, v in s.items():
                if k not in _geometry_keys and k not in ("z_top", "z_bot", "p1", "p2"):
                    plane_spec[k] = v
            s = plane_spec

            if verbose:
                p1m = s["p1_model"]
                p2m = s["p2_model"]
                print(
                    f"  slice[{i}] profile → plane  "
                    f"p1=({p1m[0]:.0f}, {p1m[1]:.0f}) m  "
                    f"p2=({p2m[0]:.0f}, {p2m[1]:.0f}) m  "
                    f"strike={s['strike']:.1f} deg  dip=90  "
                    f"zlim={s['zlim']} m (model-local)"
                )

            resolved.append(s)
            continue

        # ----------------------------------------------------------------
        # Existing kinds: ns / ew / map / plane
        # ----------------------------------------------------------------
        if kind == "ns" and "x0" in s:
            raw     = s["x0"]
            s["x0"] = resolve_pos_x(raw, zone, northern,
                                     origin_E, origin_N, origin_lat, origin_lon)
            _, crs  = parse_pos_crs(raw)
            if verbose and crs != "model":
                print(f"  slice[{i}] ns  x0: {raw!r}  →  {s['x0']:.1f} m (model-local)")

        if kind == "ew" and "y0" in s:
            raw     = s["y0"]
            s["y0"] = resolve_pos_y(raw, zone, northern,
                                     origin_E, origin_N, origin_lat, origin_lon)
            _, crs  = parse_pos_crs(raw)
            if verbose and crs != "model":
                print(f"  slice[{i}] ew  y0: {raw!r}  →  {s['y0']:.1f} m (model-local)")

        if kind == "plane" and "point" in s:
            raw        = s["point"]
            s["point"] = resolve_pos_point(raw, zone, northern, origin_E, origin_N)
            crs = (raw[1] if (isinstance(raw, (list, tuple))
                               and len(raw) == 2
                               and isinstance(raw[-1], str))
                   else "model")
            if verbose and crs != "model":
                print(f"  slice[{i}] plane point: {raw!r}  →  {s['point']} (model-local)")

        resolved.append(s)
    return resolved



# ============================================================================
# SECTION 6b: Site-list, observe.dat, and mesh-centre helpers
# ============================================================================


def read_site_position(observe_file: str, site_number: int) -> tuple[float, float]:
    """Return (x_m, y_m) model-local position for *site_number* from observe.dat.

    The file format alternates between site-header lines::

        <n>  <n>  <x_km>  <y_km>

    (where the first two tokens are the site number repeated) and data blocks.
    This function scans the file linearly, identifies site-header lines by the
    pattern ``int int float float``, and returns the (x, y) pair that matches
    *site_number*.

    Parameters
    ----------
    observe_file : str
        Path to observe.dat.
    site_number : int
        1-based site index to find.

    Returns
    -------
    x_m, y_m : float
        Model-local coordinates in metres (converted from km).

    Raises
    ------
    FileNotFoundError
        If *observe_file* does not exist.
    ValueError
        If *site_number* is not found in the file.
    """
    if not os.path.isfile(observe_file):
        raise FileNotFoundError(f"observe.dat not found: {observe_file}")

    with open(observe_file) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                n1   = int(parts[0])
                int(parts[1])
                x_km = float(parts[2])
                y_km = float(parts[3])
            except ValueError:
                continue
            if n1 == site_number:
                return x_km * 1000.0, y_km * 1000.0

    raise ValueError(
        f"Site {site_number} not found in {observe_file}.  Check SITE_NUMBER."
    )


def read_site_dat(path: str,
                  site_names=None) -> list:
    """Read site positions from a FEMTIC sitelist CSV (mt_make_sitelist.py).

    Format (comma-separated, no header; lines starting with ``#`` ignored)::

        name, lat, lon, elev, sitenum, easting, northing

    where *easting* / *northing* are UTM metres and *lat* / *lon* are
    decimal degrees.

    Parameters
    ----------
    path : str
        Path to the sitelist CSV file.
    site_names : str, list of str, or None
        If given, only rows whose ``name`` field matches are returned.
        None = all rows.

    Returns
    -------
    list of dict
        Each dict has keys ``"name"`` (str), ``"lat"`` (float),
        ``"lon"`` (float), ``"elev"`` (float), ``"sitenum"`` (int),
        ``"easting"`` (float), ``"northing"`` (float).

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If any data line cannot be parsed.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"site.dat not found: {path}")

    if site_names is not None:
        if isinstance(site_names, str):
            site_names = [site_names]
        site_names = set(site_names)

    sites = []
    with open(path) as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.split("#")[0].strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                raise ValueError(
                    f"{path}:{lineno}: expected 7 columns "
                    f"(name,lat,lon,elev,sitenum,easting,northing), "
                    f"got {len(parts)}: {raw.rstrip()}"
                )
            try:
                name     = parts[0]
                lat      = float(parts[1])
                lon      = float(parts[2])
                elev     = float(parts[3])
                sitenum  = int(parts[4])
                easting  = float(parts[5])
                northing = float(parts[6])
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{lineno}: cannot parse values: {exc}"
                ) from exc
            if site_names is not None and name not in site_names:
                continue
            sites.append(dict(
                name=name, lat=lat, lon=lon, elev=elev,
                sitenum=sitenum, easting=easting, northing=northing,
            ))
    return sites


def estimate_utm_origin(calibration_sites: list,
                        observe_file: str,
                        zone: int,
                        northern: bool,
                        *,
                        site_dat: str | None = None,
                        out: bool = True) -> tuple[float, float]:
    """Estimate UTM coordinates of the FEMTIC mesh centre.

    Two methods are available, selected automatically:

    **Bounding-box centre** (femticPY-compatible, default when
    *calibration_sites* is empty):
        Reads all site UTM coordinates from *site_dat* (the sitelist
        produced by ``mt_make_sitelist.py``) and sets the origin to the
        midpoint of the bounding box::

            origin_E = (E_min + E_max) / 2
            origin_N = (N_min + N_max) / 2

        No model-local positions or ``observe.dat`` are needed.

    **Calibration-site pairs** (classic, used when *calibration_sites* is
    non-empty):
        Each entry provides a site whose model-local position (from
        *observe_file* or inline ``x_km``/``y_km``) and geographic position
        are both known.  The mesh centre satisfies::

            E_site = origin_E + x_m_site
            N_site = origin_N + y_m_site

        With N ≥ 1 sites the least-squares solution is the mean of the N
        implied origins; residuals expose gross coordinate errors.

    Parameters
    ----------
    calibration_sites : list of dict
        Each dict has keys ``"site"`` (int), ``"crs"`` (``"latlon"`` or
        ``"utm"``), ``"coords"`` ([lon, lat] or [E_m, N_m]).  May be empty
        — triggers bounding-box fallback.
    observe_file : str
        Path to observe.dat (used for calibration-site method).
    zone : int
        UTM zone number (used for lat/lon → UTM conversion).
    northern : bool
        Hemisphere flag.
    site_dat : str or None
        Path to FEMTIC sitelist CSV (mt_make_sitelist.py format; used for
        bounding-box method).  If None and *calibration_sites* is empty,
        raises ValueError.
    out : bool
        Print per-site details and result when True.

    Returns
    -------
    origin_E, origin_N : float
        Estimated UTM coordinates of the mesh centre [m].

    Raises
    ------
    ValueError
        If both *calibration_sites* is empty and *site_dat* is None or
        the file contains no sites.
    """
    utl = _utl()

    # ------------------------------------------------------------------
    # Bounding-box centre (femticPY-compatible default)
    # ------------------------------------------------------------------
    if not calibration_sites:
        if site_dat is None:
            raise ValueError(
                "estimate_utm_origin: CALIBRATION_SITES is empty and "
                "site_dat is None — cannot estimate origin."
            )
        rows = read_site_dat(site_dat)
        if not rows:
            raise ValueError(
                f"estimate_utm_origin: site_dat {site_dat!r} is empty."
            )
        eastings  = np.array([r["easting"]  for r in rows])
        northings = np.array([r["northing"] for r in rows])
        origin_E  = float((eastings.min()  + eastings.max())  / 2.0)
        origin_N  = float((northings.min() + northings.max()) / 2.0)
        if out:
            center_lat, center_lon = utl.utm_to_latlon_zn(
                origin_E, origin_N, zone, northern
            )
            print("Estimating mesh-centre UTM origin from sitelist bounding box:")
            print(f"  sites          : {len(rows)}")
            print(f"  E range        : {eastings.min():.1f} \u2013 {eastings.max():.1f} m")
            print(f"  N range        : {northings.min():.1f} \u2013 {northings.max():.1f} m")
            print(f"  UTM_ORIGIN_E   = {origin_E:.1f}")
            print(f"  UTM_ORIGIN_N   = {origin_N:.1f}")
            print(f"  UTM_ORIGIN_LAT = {center_lat:.6f}")
            print(f"  UTM_ORIGIN_LON = {center_lon:.6f}")
            print(f"  Copy these values into the Configuration block.")
            print()
        return origin_E, origin_N

    # ------------------------------------------------------------------
    # Calibration-site pairs
    # ------------------------------------------------------------------
    offsets_E = []
    offsets_N = []

    if out:
        print("Estimating mesh-centre UTM origin from calibration sites:")
        print(f"  {'site':>5}  {'x_model':>10}  {'y_model':>10}  "
              f"{'E_utm':>12}  {'N_utm':>14}  {'dE':>8}  {'dN':>8}")
        print("  " + "-" * 77)

    for entry in calibration_sites:
        site_num = int(entry["site"])
        crs      = str(entry["crs"])
        coords   = list(entry["coords"])

        if "x_km" in entry and "y_km" in entry:
            x_m = float(entry["x_km"]) * 1e3
            y_m = float(entry["y_km"]) * 1e3
        else:
            x_m, y_m = read_site_position(observe_file, site_num)

        if crs == "latlon":
            lon_deg, lat_deg = coords
            E_site, N_site = utl.latlon_to_utm_zn(lat_deg, lon_deg, zone, northern)
        elif crs == "utm":
            E_site, N_site = float(coords[0]), float(coords[1])
        else:
            raise ValueError(
                f"Calibration site {site_num}: unknown crs={crs!r}. "
                f"Use 'latlon' or 'utm'."
            )

        oE = E_site - x_m
        oN = N_site - y_m
        offsets_E.append(oE)
        offsets_N.append(oN)

        if out:
            print(f"  {site_num:>5}  {x_m/1000:>10.3f}  {y_m/1000:>10.3f}  "
                  f"{E_site:>12.1f}  {N_site:>14.1f}  "
                  f"{oE:>8.1f}  {oN:>8.1f}")

    origin_E = float(np.mean(offsets_E))
    origin_N = float(np.mean(offsets_N))

    if out and len(calibration_sites) > 1:
        print()
        print(f"  {'site':>5}  {'res_E (m)':>10}  {'res_N (m)':>10}")
        print("  " + "-" * 30)
        for entry, oE, oN in zip(calibration_sites, offsets_E, offsets_N):
            print(f"  {int(entry['site']):>5}  "
                  f"{oE - origin_E:>10.2f}  {oN - origin_N:>10.2f}")
        rms_E = float(np.sqrt(np.mean((np.array(offsets_E) - origin_E) ** 2)))
        rms_N = float(np.sqrt(np.mean((np.array(offsets_N) - origin_N) ** 2)))
        print(f"  {'RMS':>5}  {rms_E:>10.2f}  {rms_N:>10.2f}")

    print()
    print(f"  Estimated mesh-centre UTM origin:")
    print(f"    UTM_ORIGIN_E = {origin_E:.1f}")
    print(f"    UTM_ORIGIN_N = {origin_N:.1f}")
    print(f"  Copy these values into the Configuration block.")
    print()

    return origin_E, origin_N


def _point_in_tet(p: np.ndarray, verts: np.ndarray) -> bool:
    """Test whether point *p* lies inside the tetrahedron *verts* (4×3).

    Uses barycentric coordinates: p = v0 + T·λ where T = [v1-v0, v2-v0, v3-v0].
    Point is inside when λ₁ ≥ 0, λ₂ ≥ 0, λ₃ ≥ 0, and λ₁+λ₂+λ₃ ≤ 1.
    A small tolerance (1 × 10⁻¹⁰) is used on each test to handle points
    exactly on faces or edges.

    Parameters
    ----------
    p : np.ndarray, shape (3,)
        Query point in model-local metres.
    verts : np.ndarray, shape (4, 3)
        Node coordinates of the tetrahedron.

    Returns
    -------
    bool
    """
    tol = 1e-10
    v0 = verts[0]
    T  = (verts[1:] - v0).T
    try:
        lam = np.linalg.solve(T, p - v0)
    except np.linalg.LinAlgError:
        return False
    return bool(
        lam[0] >= -tol and lam[1] >= -tol and lam[2] >= -tol
        and lam[0] + lam[1] + lam[2] <= 1.0 + tol
    )


def extract_borehole_log(
    nodes: np.ndarray,
    conn: np.ndarray,
    rho_elem: np.ndarray,
    x_m: float,
    y_m: float,
    z_top: float,
    z_bot: float,
    dz: float,
    *,
    out: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample resistivity along a vertical borehole by point-in-element search.

    For each depth level the function locates the tetrahedron that contains
    the query point ``(x_m, y_m, z)`` and returns the element resistivity.
    Levels that fall outside every tetrahedron (air, below the mesh, etc.)
    are set to ``NaN``.

    The search is accelerated by a bounding-box pre-filter: for each query
    point only the tetrahedra whose axis-aligned bounding box contains
    ``(x_m, y_m)`` *and* whose z-range brackets ``z`` are tested exactly.

    Parameters
    ----------
    nodes : np.ndarray, shape (nn, 3)
        Node coordinate array.
    conn : np.ndarray, shape (nelem, 4)
        Connectivity array (0-based node indices).
    rho_elem : np.ndarray, shape (nelem,)
        Per-element resistivity [Ω·m].
    x_m : float
        Borehole easting [model-local m].
    y_m : float
        Borehole northing [model-local m].
    z_top : float
        Start depth [model-local m, z positive-down].
    z_bot : float
        End depth [model-local m, z positive-down].
    dz : float
        Depth sampling interval [m] (positive).
    out : bool
        Print progress / hit statistics when True.

    Returns
    -------
    depths : np.ndarray, shape (nz,)
        Depth levels [m, z positive-down].
    rho : np.ndarray, shape (nz,)
        Resistivity [Ω·m]; NaN where no element found.
    """
    depths = np.arange(z_top, z_bot + 0.5 * dz, dz)
    rho    = np.full(len(depths), np.nan)

    verts_all = nodes[conn]
    xmin_e = verts_all[:, :, 0].min(axis=1)
    xmax_e = verts_all[:, :, 0].max(axis=1)
    ymin_e = verts_all[:, :, 1].min(axis=1)
    ymax_e = verts_all[:, :, 1].max(axis=1)
    zmin_e = verts_all[:, :, 2].min(axis=1)
    zmax_e = verts_all[:, :, 2].max(axis=1)

    lat_mask = (
        (xmin_e <= x_m) & (x_m <= xmax_e) &
        (ymin_e <= y_m) & (y_m <= ymax_e)
    )
    lat_idx = np.where(lat_mask)[0]

    n_hit = 0
    for i, z in enumerate(depths):
        p = np.array([x_m, y_m, z])
        z_ok  = (zmin_e[lat_idx] <= z) & (z <= zmax_e[lat_idx])
        cands = lat_idx[z_ok]
        for k in cands:
            if _point_in_tet(p, verts_all[k]):
                rho[i] = rho_elem[k]
                n_hit += 1
                break

    if out:
        n_nan = int(np.sum(~np.isfinite(rho)))
        print(f"    borehole ({x_m:.0f}, {y_m:.0f}): "
              f"{len(depths)} levels, {n_hit} hit, {n_nan} outside mesh")
    return depths, rho


# ============================================================================
# SECTION 7: CLI wrapper for unified module
# ============================================================================


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    CLI entry point for femtic.py with subcommands:

    - femtic-to-npz
    - npz-to-vtk
    - npz-to-femtic
    - iso-to-aniso-block
    - edi-to-observe
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

    # iso-to-aniso-block
    p_i2a = sub.add_parser(
        "iso-to-aniso-block",
        help="Convert a v4 (isotropic) resistivity_block_iterX.dat to v5 (anisotropic) format.",
    )
    p_i2a.add_argument("--in", dest="block_in", required=True, help="Input v4 resistivity block.")
    p_i2a.add_argument("--out", dest="block_out", required=True, help="Output v5 resistivity block.")
    p_i2a.add_argument(
        "--aniso-regions",
        nargs="+",
        type=int,
        default=None,
        metavar="IREG",
        help="Region indices to mark anisotropic (default: none — format conversion only).",
    )
    p_i2a.add_argument(
        "--aniso-type",
        type=int,
        default=1,
        choices=[1, 2],
        help="1=transverse isotropy (default), 2=general anisotropy.",
    )
    p_i2a.add_argument("--rhoyy-ratio", type=float, default=1.0, help="rhoYY = rhoXX * ratio (default 1.0).")
    p_i2a.add_argument("--rhozz-ratio", type=float, default=1.0, help="rhoZZ = rhoXX * ratio (general anisotropy only).")
    p_i2a.add_argument("--strike", type=float, default=0.0, help="Strike angle in degrees (default 0.0).")
    p_i2a.add_argument("--dip", type=float, default=0.0, help="Dip angle in degrees (default 0.0).")
    p_i2a.add_argument("--slant", type=float, default=0.0, help="Slant angle in degrees (general anisotropy only).")
    p_i2a.add_argument("--free-strike", action="store_true", help="Leave strike free (default: fixed).")
    p_i2a.add_argument("--free-dip", action="store_true", help="Leave dip free (default: fixed).")
    p_i2a.add_argument("--free-slant", action="store_true", help="Leave slant free (default: fixed).")

    # edi-to-observe
    p_e2o = sub.add_parser(
        "edi-to-observe",
        help="Convert a set of EDI files to a FEMTIC observe.dat (MT only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Read one or more EDI files and write a FEMTIC-style observe.dat.\n\n"
            "Site positions must be supplied as model-frame (x, y) via the\n"
            "--xy-csv option (a CSV file with columns: station,x_m,y_m) or by\n"
            "embedding x_m / y_m keys in the EDI HEADER (not standard).\n\n"
            "Elevation is taken from the EDI ELEV field and negated:\n"
            "  z_femtic = -elev_m   (FEMTIC z positive downward)\n\n"
            "Z is converted from MT field units (mV/km/nT) to SI Ohm:\n"
            "  Z_SI = Z_MT * mu0 * 1e3"
        ),
    )
    p_e2o.add_argument(
        "edi_files",
        nargs="+",
        metavar="EDI",
        help="One or more EDI files to convert (glob patterns must be pre-expanded).",
    )
    p_e2o.add_argument(
        "--out",
        required=True,
        metavar="OBSERVE_DAT",
        help="Output observe.dat path.",
    )
    p_e2o.add_argument(
        "--xy-csv",
        default=None,
        metavar="CSV",
        help=(
            "CSV file with columns 'station,x_m,y_m' (no header assumed unless "
            "first row is non-numeric).  Matched against the EDI station name."
        ),
    )
    p_e2o.add_argument(
        "--x-key",
        default="x_m",
        metavar="KEY",
        help="Dict key for model-frame x (default: 'x_m').",
    )
    p_e2o.add_argument(
        "--y-key",
        default="y_m",
        metavar="KEY",
        help="Dict key for model-frame y (default: 'y_m').",
    )
    p_e2o.add_argument(
        "--manufacturer",
        default="metronix",
        choices=["metronix", "phoenix", "delta"],
        help="EDI FT convention (default: metronix).",
    )
    p_e2o.add_argument(
        "--err-kind",
        default="var",
        choices=["var", "std"],
        help="Error interpretation in EDI files (default: var).",
    )
    p_e2o.add_argument(
        "--freq-order",
        default="inc",
        choices=["inc", "dec", "keep"],
        help="Frequency ordering when loading EDI (default: inc).",
    )
    p_e2o.add_argument(
        "--preamble",
        default=None,
        metavar="TEXT",
        help="Optional comment text prepended to observe.dat (use \\n for newlines).",
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

    if args.cmd == "iso-to-aniso-block":
        convert_resistivity_block_to_anisotropic(
            args.block_in,
            args.block_out,
            aniso_regions=args.aniso_regions,
            aniso_type=args.aniso_type,
            rhoYY_ratio=args.rhoyy_ratio,
            rhoZZ_ratio=args.rhozz_ratio,
            strike=args.strike,
            dip=args.dip,
            slant=args.slant,
            fix_strike=not args.free_strike,
            fix_dip=not args.free_dip,
            fix_slant=not args.free_slant,
            out=True,
        )
        return 0

    if args.cmd == "edi-to-observe":
        import importlib
        try:
            dp = importlib.import_module("data_proc")
        except ImportError as exc:
            print(f"ERROR: could not import data_proc: {exc}", file=sys.stderr)
            return 1

        # Load XY table if provided
        xy_table: dict[str, tuple[float, float]] = {}
        if args.xy_csv is not None:
            import csv
            with open(args.xy_csv, newline="") as fh:
                reader = csv.reader(fh)
                for row in reader:
                    if len(row) < 3:
                        continue
                    try:
                        xy_table[row[0].strip()] = (float(row[1]), float(row[2]))
                    except ValueError:
                        pass  # skip header or malformed rows

        edi_list: list[dict[str, Any]] = []
        for edi_path in args.edi_files:
            try:
                edi = dp.load_edi(
                    edi_path,
                    manufacturer=args.manufacturer,
                    err_kind=args.err_kind,
                    freq_order=args.freq_order,
                )
            except Exception as exc:
                print(f"WARNING: could not load {edi_path}: {exc}", file=sys.stderr)
                continue

            name = str(edi.get("station") or Path(edi_path).stem)
            edi["station"] = name

            if name in xy_table:
                edi[args.x_key] = xy_table[name][0]
                edi[args.y_key] = xy_table[name][1]

            edi_list.append(edi)

        if not edi_list:
            print("ERROR: no EDI files loaded successfully.", file=sys.stderr)
            return 1

        preamble = args.preamble.replace("\\n", "\n") if args.preamble else None

        edi_list_to_observe_dat(
            edi_list,
            args.out,
            x_key=args.x_key,
            y_key=args.y_key,
            preamble=preamble,
        )
        print(f"Wrote {len(edi_list)} site(s) to: {args.out}")
        return 0

    ap.error(f"Unknown command {args.cmd!r}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
