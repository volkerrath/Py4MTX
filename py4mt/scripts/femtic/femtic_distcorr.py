#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_distcorr.py
==================

Apply the galvanic-distortion result of a FEMTIC inversion to the
observations: read observe.dat and distortion_iter<N>.dat (N = the best
iteration of the run, taken from femtic.cnv) and save a new observations
file in which the impedances of every MT site are corrected for the
estimated distortion matrix of that site.

@author vrath
@project py4mt
@created 2026-11-13
@modified 2026-11-13

AI-GENERATED CODE NOTICE
------------------------
Author    : Claude Sonnet 5.5 (Anthropic)
Generated : 2026-11-13
This module is AI-generated. It was tested only with synthetic observe.dat /
distortion files (see femtic_distcorr_readme.md), not with a real FEMTIC run.
Review before production or publication use.

What the script does
--------------------
FEMTIC models the observed impedance of a site as the distorted response of
the model,

    Z_obs = C Z_model ,         C = I + C'

(E_obs = C E, the magnetic field is not distorted). FEMTIC estimates the
difference matrix C' (type ESTIMATE_DISTORTION_MATRIX_DIFFERENCE) and writes
it, one line per site, to distortion_iter<N>.dat:

    nsites
    siteID  C'xx  C'xy  C'yx  C'yy

C = C' + I is formed by fem.read_distortion_file(). Per site, matched by the
site ID (first column of the file / first token of the MT station line):

    MODE = "remove" :  Z_new = C^-1 Z_obs     (distortion correction)
    MODE = "apply"  :  Z_new = C Z_obs        (distort the data further)

Errors (ERROR_MODE = "propagate"), first order, the errors of the four
impedance elements taken as independent and C taken as exact:

    err(Re Z_new_ij) = sqrt( sum_k M_ik^2 err(Re Z_kj)^2 ) ,   M = C^-1 or C

and the same for Im. ERROR_MODE = "keep" leaves the error columns as they are.

Only the MT blocks are changed. VTF blocks (magnetic quantities) are not
affected by galvanic distortion. PT blocks are left as they are as well: the
phase tensor is unchanged by a real distortion matrix (Caldwell et al., 2004).
Blocks that femtic.read_observe_dat does not read (APP_RES_AND_PHS, HTF) make
it stop with its own error message.

Station lines, the preamble and the END line are written back as they were
(fem.write_observe_dat). All data rows of the new file are rewritten in the
format %.8E; this changes only the number format of the VTF / PT rows, not
their values (relative difference below 1e-8).

Sites of the MT block without a row in the distortion file, and sites whose
C is singular (condition number above MAX_COND), are kept unchanged and are
named in the report. The original observe.dat is never overwritten.

Plots (PLOT = True)
-------------------
The original and the new file are plotted with
femtic_viz.plot_data_ensemble (apparent resistivity and phase, PLOT_WHAT /
PLOT_COMPS; original solid, corrected dashed, error envelopes if PLOT_ERRORS):

    <PLOT_FILE>_overview.<fmt>   all MT sites in one figure (PLOT_OVERVIEW)
    <PLOT_FILE>_site<ID>.<fmt>   one figure per site (PLOT_SITES: the N most
                                 distorted sites, "all", or a list of IDs /
                                 names); title: site, cond(C)

Both files are read back from disk, so the figures show what was written. The
plot step is the last one; a failure there is reported but does not undo the
output file.

Limits
------
- Only distortion files of type ESTIMATE_DISTORTION_MATRIX_DIFFERENCE are
  understood (the type for which fem.read_distortion_file gives C = C' + I).
  The distortion files of the gains-only and the gains-and-rotations types hold
  other parameters; they are NOT implemented. If results_iter<N>.h5 is next to
  the files (and h5py is available) its /distortion type is checked and the run
  stops for any other type.
- Z_obs = C Z_model is the standard galvanic-distortion model; that FEMTIC uses
  this very form was not checked against the FEMTIC source here. Check the
  direction with one site (MODE = "apply" should give back approximately the
  data of the original observe.dat when run on the corrected file).

Usage
-----
Edit the configuration block below and run

    python femtic_distcorr.py

Requires
--------
numpy; femtic.py, util.py, version.py (py4mt, found via PY4MTX_ROOT);
femtic_viz.py, data_viz.py and matplotlib for the plots (PLOT = True);
h5py only for the optional type check of results_iter<N>.h5.

References
----------
- Caldwell, T. G., Bibby, H. M., Brown, C. (2004). The magnetotelluric phase
  tensor. Geophysical Journal International, 158(2), 457-469.
  doi:10.1111/j.1365-246X.2004.02281.x
- Groom, R. W., Bailey, R. C. (1989). Decomposition of magnetotelluric
  impedance tensors in the presence of local three-dimensional galvanic
  distortion. Journal of Geophysical Research, 94(B2), 1913-1925.
  doi:10.1029/JB094iB02p01913
- Lilley, F. E. M. (2016). The distortion tensor of magnetotellurics: a
  tutorial on some properties. Exploration Geophysics, 47(2), 85-99.
  doi:10.1071/EG14093

Provenance
----------
    2026-11-13  Claude Sonnet 5.5 (Anthropic)
                Created. Reads observe.dat and distortion_iter<N>.dat (N from
                femtic.cnv: "best", "last" or a number), corrects (or applies)
                the distortion matrices of the MT sites with first-order error
                propagation, writes a new observe file and a per-site report,
                and re-reads the written file as a check. AI-generated; review
                before production use.
    2026-11-13b Claude Sonnet 5.5 (Anthropic)
                Plots of the original and the corrected data with
                femtic_viz.plot_data_ensemble: overview of all MT sites and one
                figure per site for the most distorted sites (new settings
                PLOT, PLOT_OVERVIEW, PLOT_SITES, PLOT_WHAT, PLOT_COMPS,
                PLOT_ERRORS, PLOT_PERIOD_LIMITS, PLOT_FILE, PLOT_FORMATS,
                PLOT_DPI, PLOT_TITLE, SHOW). Needs the femtic_viz.py /
                data_viz.py of the project. AI-generated; review before use.
"""

from __future__ import annotations

import os
import re
import sys
import copy
import inspect
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

for _base in [PY4MTX_ROOT + "/py4mt/modules/"]:
    for _p in [Path(_base), *Path(_base).rglob("*")]:
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

import femtic as fem
import femtic_viz as fviz
import util as utl
from version import versionstrg

rng = np.random.default_rng()
nan = np.nan
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())
titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng + "\n\n")

# =============================================================================
#  Configuration
# =============================================================================

#: Run directory of the FEMTIC inversion (observe.dat, femtic.cnv,
#: distortion_iter<N>.dat). The output files are written here as well.
WORK_DIR = r"/home/vrath/work/femtic_run/"

#: Observation file to be corrected (name inside WORK_DIR, or a full path).
OBSERVE_FILE = "observe.dat"

#: Convergence file: one name or a list of candidate names (the first one that
#: exists is used). Same setting in all femtic_* scripts. Used for
#: ITERATION = "best" and "last".
CNV_FILE = ["femtic.cnv", "FEMTIC-DABIC.cnv"]

#: Iteration whose distortion result is used:
#:   "best" - iteration with the smallest nRMS in the convergence file
#:            (as fem.get_nrms; iteration 0 is included)
#:   "last" - iteration of the last row of the convergence file
#:   N      - an integer iteration number
ITERATION = "best"

#: File names; {iter} is replaced by the iteration number. Names inside
#: WORK_DIR, or full paths.
DISTORTION_FILE = "distortion_iter{iter}.dat"
OUTPUT_FILE = "observe_distcorr_iter{iter}.dat"
#: ASCII table of the sites (None = no report file; it is printed anyway).
REPORT_FILE = "observe_distcorr_iter{iter}_report.txt"
#: results_iter<N>.h5 of the same iteration, used only to check the type of the
#: distortion parameters (None = no check; skipped without h5py or file).
RESULTS_H5_FILE = "results_iter{iter}.h5"

#: "remove": Z_new = C^-1 Z_obs (distortion correction);
#: "apply":  Z_new = C Z_obs.
MODE = "remove"

#: "propagate": errors of the new impedances from the errors of the old ones
#: (see above); "keep": the error columns are not changed.
ERROR_MODE = "propagate"

#: How MT sites and rows of the distortion file are paired:
#:   "id"    - site ID (first token of the station line = first column of the
#:             distortion file)
#:   "order" - i-th MT site of observe.dat = i-th row of the distortion file
#:             (both must have the same number of sites)
#:   "auto"  - "id" if all MT station lines have the FEMTIC layout
#:             (ID H-ID ... X Y), "order" for the legacy "name x y z" layout
MATCH_BY = "auto"

#: An MT site without a row in the distortion file: "keep" (leave it
#: unchanged, warn) or "error" (stop).
MISSING_SITES = "keep"

#: Sites whose distortion matrix has a larger condition number (or is not
#: invertible) are kept unchanged; the correction would amplify the noise.
MAX_COND = 1.0e6

#: Re-read the written file and compare it with the expected result.
VERIFY_OUTPUT = True

# --- plots of the original and the corrected data (femtic_viz) ---------------
#: False = no figures.
PLOT = True
#: One figure with all MT sites (solid: original, dashed: new file).
PLOT_OVERVIEW = True
#: Draw only this many randomly chosen sites in the overview (None = all).
PLOT_OVERVIEW_NSITES = None
#: One figure per site:
#:   None, 0 or []  - none
#:   N (integer)    - the N corrected sites with the largest ||C - I||
#:   "all"          - every corrected site
#:   list           - site IDs and/or names, e.g. [3, "A01"]
PLOT_SITES = 6
#: Panels (columns): "rho", "phase", "tipper", "pt" (tipper and PT are not
#: changed by the correction; they are of use only to compare with the file).
PLOT_WHAT = ["rho", "phase"]
#: Components of the "rho" / "phase" panels, comma separated subset of
#: "xx,xy,yx,yy".
PLOT_COMPS = "xy,yx"
#: Error envelopes of both data sets.
PLOT_ERRORS = True
#: Period limits (T_min, T_max) [s], or None.
PLOT_PERIOD_LIMITS = None
#: File name stem of the figures (inside WORK_DIR, or a full path); {iter} as
#: above. "_overview" / "_site<ID>" and the format are appended.
PLOT_FILE = "observe_distcorr_iter{iter}_plot"
PLOT_FORMATS = ["png"]
PLOT_DPI = 200
#: Figure title: "auto"; None, False or "" = none; text (placeholders {site},
#: {id}, {cond}, {iter}, {new}); or a dict {"overview": spec, "site": spec,
#: "default": spec}.
PLOT_TITLE = "auto"
#: Show the figures on screen as well (needs a display).
SHOW = False

# =============================================================================
#  Functions (explicit parameters; the configuration is supplied in main())
# =============================================================================


def _full_path(work_dir: str, name: str) -> Path:
    """Return ``name`` as is if it is an absolute path, else inside ``work_dir``."""
    p = Path(name)
    return p if p.is_absolute() else Path(work_dir) / p


def select_iteration(work_dir: str, iteration, cnv_file) -> Tuple[int, Optional[float]]:
    """Resolve ``iteration`` ("best", "last" or an integer) to a number.

    Parameters
    ----------
    work_dir : str
        Run directory (holds the convergence file).
    iteration : str or int
        ``"best"``: smallest nRMS (``fem.get_nrms``); ``"last"``: last row of
        the convergence file; an integer is used as given.
    cnv_file : str or list of str
        Name(s) of the convergence file.

    Returns
    -------
    (int, float or None)
        Iteration number and its nRMS (None if the iteration was given as a
        number).
    """
    if isinstance(iteration, (int, np.integer)) and not isinstance(iteration, bool):
        return int(iteration), None
    key = str(iteration).strip().lower()
    if key == "best":
        num, nrms = fem.get_nrms(work_dir, cnv_file)
        if num < 0:
            raise RuntimeError(f"no usable rows in the convergence file of {work_dir}")
        return int(num), float(nrms)
    if key == "last":
        row = fem.read_cnv_last(work_dir, cnv_file)
        if row is None:
            raise RuntimeError(f"no usable convergence file / rows in {work_dir}")
        return int(round(row["Iter"])), float(row["RMS"])
    try:
        return int(key), None
    except ValueError as exc:
        raise ValueError(
            f"ITERATION must be 'best', 'last' or an integer, got {iteration!r}") from exc


def _read_distortion_ids(path) -> List[int]:
    """Site-ID column of a FEMTIC distortion file, in file order.

    ``fem.read_distortion_file`` parses the column but does not return it.
    """
    lines = Path(path).read_text().strip().splitlines()
    ids = []
    for line in lines[1:]:
        parts = line.split()
        if parts:
            ids.append(int(parts[0]))
    return ids


def load_distortion(path) -> Tuple[List[int], np.ndarray]:
    """Read a distortion file: site IDs and the matrices C = C' + I.

    Returns
    -------
    ids : list of int
        Site IDs in file order.
    C : ndarray, shape (nsites, 2, 2)
        Distortion matrices C = I + C' (``fem.read_distortion_file``).
    """
    c, _c_dash = fem.read_distortion_file(str(path))
    ids = _read_distortion_ids(path)
    if len(ids) != len(c):
        raise RuntimeError(
            f"{path}: {len(ids)} site IDs but {len(c)} matrices - check the file format.")
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        warnings.warn(f"{path}: duplicate site IDs {dup}; the last row of each is used.")
    return ids, np.asarray(c, dtype=float)


def check_distortion_type(h5_path) -> Optional[str]:
    """Stop if ``results_iter<N>.h5`` says the distortion is not a matrix difference.

    Returns a short note (str) if the check could not be made, else None.
    Raises RuntimeError if the /distortion group has another type than
    ESTIMATE_DISTORTION_MATRIX_DIFFERENCE (1).
    """
    p = Path(h5_path)
    if not p.is_file():
        return f"{p.name} not found - type of the distortion parameters not checked"
    try:
        res = fem.read_results_distortion(str(p), out=False)
    except ImportError:
        return "h5py not available - type of the distortion parameters not checked"
    except Exception as exc:  # unreadable file: not a reason to stop
        return f"{p.name} not readable ({exc}) - type of the distortion parameters not checked"
    if res is None:
        return f"{p.name} has no /distortion group - type not checked"
    if int(res["type"]) != 1:
        raise RuntimeError(
            f"{p.name}: distortion type is {res['type_name']}; only "
            f"ESTIMATE_DISTORTION_MATRIX_DIFFERENCE is implemented here "
            f"(its distortion file holds C' = C - I).")
    return None


def apply_matrix_to_mt(data: np.ndarray, error: np.ndarray, M: np.ndarray,
                       error_mode: str = "propagate") -> Tuple[np.ndarray, np.ndarray]:
    """Multiply the impedance tensors of one MT site by the real 2x2 matrix ``M``.

    Parameters
    ----------
    data, error : ndarray, shape (nfreq, 8)
        Zxx_re, Zxx_im, Zxy_re, Zxy_im, Zyx_re, Zyx_im, Zyy_re, Zyy_im, and the
        errors in the same order (FEMTIC observe.dat MT layout).
    M : ndarray, shape (2, 2)
        Real matrix: C^-1 for the correction, C for "apply".
    error_mode : {"propagate", "keep"}
        See module docstring.

    Returns
    -------
    new_data, new_error : ndarray, shape (nfreq, 8)
        ``Z_new = M Z`` per frequency, and the errors.
    """
    data = np.asarray(data, dtype=float)
    error = np.asarray(error, dtype=float)
    nf = data.shape[0]
    M = np.asarray(M, dtype=float)

    z = (data[:, 0::2] + 1j * data[:, 1::2]).reshape(nf, 2, 2)   # [[xx, xy], [yx, yy]]
    zn = np.einsum("ik,fkj->fij", M, z).reshape(nf, 4)
    new_data = np.empty_like(data)
    new_data[:, 0::2] = zn.real
    new_data[:, 1::2] = zn.imag

    if error_mode == "keep":
        return new_data, error.copy()
    if error_mode != "propagate":
        raise ValueError(f"ERROR_MODE must be 'propagate' or 'keep', got {error_mode!r}")

    m2 = M ** 2
    new_error = np.empty_like(error)
    for part in (0, 1):                      # 0: real parts, 1: imaginary parts
        e2 = (error[:, part::2] ** 2).reshape(nf, 2, 2)
        new_error[:, part::2] = np.sqrt(np.einsum("ik,fkj->fij", m2, e2)).reshape(nf, 4)
    return new_data, new_error


def _site_id(site: dict) -> Optional[int]:
    """Integer ID of a station with the FEMTIC layout, else None."""
    hdr = site.get("header", {})
    if hdr.get("layout") == "femtic":
        try:
            return int(hdr["name"])
        except (KeyError, ValueError):
            return None
    return None


def distcorr(parsed: dict, ids: List[int], C: np.ndarray, *, mode: str = "remove",
             error_mode: str = "propagate", match_by: str = "auto",
             missing: str = "keep", max_cond: float = 1.0e6) -> List[dict]:
    """Correct (or distort) the MT sites of a parsed observe.dat in place.

    Parameters
    ----------
    parsed : dict
        Result of ``fem.read_observe_dat``; the MT blocks are modified.
    ids, C : list of int, ndarray (nsites, 2, 2)
        From :func:`load_distortion`.
    mode, error_mode, match_by, missing, max_cond
        See the configuration block (MODE, ERROR_MODE, MATCH_BY, MISSING_SITES,
        MAX_COND).

    Returns
    -------
    list of dict
        One report row per MT site: ``id, name, nfreq, cond, det, C, status``.
        ``status`` is one of ``corrected``, ``identity``, ``no distortion row``,
        ``singular``, ``no data``.
    """
    if mode not in ("remove", "apply"):
        raise ValueError(f"MODE must be 'remove' or 'apply', got {mode!r}")
    if match_by not in ("id", "order", "auto"):
        raise ValueError(f"MATCH_BY must be 'id', 'order' or 'auto', got {match_by!r}")
    if missing not in ("keep", "error"):
        raise ValueError(f"MISSING_SITES must be 'keep' or 'error', got {missing!r}")

    mt_sites = [s for b in parsed["blocks"] if b["obs_type"] == "MT" for s in b["sites"]]
    if not mt_sites:
        raise RuntimeError("no MT block in the observation file - nothing to correct.")

    if match_by == "auto":
        match_by = "id" if all(_site_id(s) is not None for s in mt_sites) else "order"

    by_id: Dict[int, np.ndarray] = {}
    for i, sid in enumerate(ids):
        by_id[sid] = C[i]

    if match_by == "order" and len(mt_sites) != len(ids):
        raise RuntimeError(
            f"MATCH_BY = 'order' needs equal numbers of sites, but observe.dat has "
            f"{len(mt_sites)} MT sites and the distortion file {len(ids)} rows.")
    if match_by == "id":
        n_hit = sum(1 for s in mt_sites if _site_id(s) in by_id)
        if n_hit == 0:
            raise RuntimeError(
                "none of the MT site IDs of observe.dat is in the distortion file "
                f"(IDs in observe.dat: {[_site_id(s) for s in mt_sites[:5]]}..., "
                f"in the distortion file: {ids[:5]}...). Check that both files belong "
                "to the same run, or try MATCH_BY = 'order'.")

    rows: List[dict] = []
    used = set()
    eye = np.eye(2)
    for k, site in enumerate(mt_sites):
        if match_by == "id":
            sid = _site_id(site)
            Cs = by_id.get(sid) if sid is not None else None
        else:
            sid = ids[k]
            Cs = C[k]
        name = str(site.get("header", {}).get("name", sid))
        row = dict(id=sid, name=name, nfreq=int(site["nfreq"]), cond=nan, det=nan,
                   C=None, status="")

        if Cs is None:
            row["status"] = "no distortion row"
            rows.append(row)
            continue
        used.add(sid)
        row["C"] = Cs
        row["det"] = float(np.linalg.det(Cs))
        cond = float(np.linalg.cond(Cs))
        row["cond"] = cond

        if not np.isfinite(cond) or cond > max_cond:
            row["status"] = "singular"
            rows.append(row)
            continue
        if int(site["nfreq"]) == 0:
            row["status"] = "no data"
            rows.append(row)
            continue
        if np.allclose(Cs, eye, rtol=0.0, atol=1.0e-12):
            row["status"] = "identity"
            rows.append(row)
            continue

        M = np.linalg.inv(Cs) if mode == "remove" else Cs
        site["data"], site["error"] = apply_matrix_to_mt(
            site["data"], site["error"], M, error_mode)
        row["status"] = "corrected"
        rows.append(row)

    unused = [i for i in ids if i not in used]
    if match_by == "id" and unused:
        warnings.warn(
            f"{len(unused)} site(s) of the distortion file are not MT sites of "
            f"observe.dat: {unused[:20]}{' ...' if len(unused) > 20 else ''}")
    absent = [r for r in rows if r["status"] == "no distortion row"]
    if absent:
        msg = (f"{len(absent)} MT site(s) without a row in the distortion file "
               f"(kept unchanged): {[r['name'] for r in absent][:20]}")
        if missing == "error":
            raise RuntimeError(msg)
        warnings.warn(msg)
    return rows


def format_report(rows: List[dict], *, iteration: int, nrms: Optional[float],
                  mode: str, error_mode: str, observe: str, distortion: str,
                  output: str) -> str:
    """ASCII table with one line per MT site, and a header with the settings."""
    n = {}
    for r in rows:
        n[r["status"]] = n.get(r["status"], 0) + 1
    nrms_txt = "n/a" if nrms is None else f"{nrms:.4g}"
    out = []
    out.append("=" * 100)
    out.append("femtic_distcorr: distortion applied to the observations")
    out.append("=" * 100)
    out.append(f"observe file    : {observe}")
    out.append(f"distortion file : {distortion}")
    out.append(f"output file     : {output}")
    out.append(f"iteration       : {iteration}  (nRMS {nrms_txt})")
    out.append(f"mode            : {mode}   (Z_new = "
               f"{'C^-1' if mode == 'remove' else 'C'} Z_obs),  errors: {error_mode}")
    out.append("-" * 100)
    out.append(f"{'ID':>6s}  {'name':<12s} {'nfreq':>5s} {'cond(C)':>9s} {'det(C)':>9s}"
               f" {'Cxx':>9s} {'Cxy':>9s} {'Cyx':>9s} {'Cyy':>9s}  status")
    out.append("-" * 100)
    for r in rows:
        sid = "-" if r["id"] is None else str(r["id"])
        if r["C"] is None:
            cs = " " * 9 + " " + " " * 9 + " " + " " * 9 + " " + " " * 9
            cond = det = " " * 9
        else:
            c = r["C"]
            cs = " ".join(f"{v:9.4f}" for v in (c[0, 0], c[0, 1], c[1, 0], c[1, 1]))
            cond = f"{r['cond']:9.3g}"
            det = f"{r['det']:9.4f}"
        flag = ""
        if r["status"] == "corrected" and r["det"] < 0.0:
            flag = " (det < 0)"
        out.append(f"{sid:>6s}  {r['name'][:12]:<12s} {r['nfreq']:5d} {cond} {det} "
                   f"{cs}  {r['status']}{flag}")
    out.append("-" * 100)
    out.append(f"{len(rows)} MT site(s): "
               + ", ".join(f"{v} {k}" for k, v in sorted(n.items())))
    n_neg = sum(1 for r in rows if r["status"] == "corrected" and r["det"] < 0.0)
    if n_neg:
        out.append(f"note: {n_neg} corrected site(s) have det(C) < 0 (a distortion "
                   "matrix that reverses orientation; check these sites).")
    out.append("=" * 100)
    return "\n".join(out) + "\n"


def verify_output(path_in: str, path_out: str, ids: List[int], C: np.ndarray,
                  rows: List[dict], *, mode: str, error_mode: str) -> List[str]:
    """Re-read the written file and compare it with the expected result.

    Checks: same blocks, sites and frequencies; the MT data of every corrected
    site equal ``M Z`` of the input; all other data (unchanged MT sites, VTF,
    PT) equal the input to the precision of the %.8E format.

    Returns a list of problem descriptions (empty if everything is as expected).
    """
    a = fem.read_observe_dat(path_in, compute_mt_derived=False, bootstrap_n=0)
    b = fem.read_observe_dat(path_out, compute_mt_derived=False, bootstrap_n=0)
    problems: List[str] = []
    if len(a["blocks"]) != len(b["blocks"]):
        return [f"number of blocks differs: {len(a['blocks'])} vs {len(b['blocks'])}"]

    corrected = {r["name"]: r["C"] for r in rows if r["status"] == "corrected"}
    for ba, bb in zip(a["blocks"], b["blocks"]):
        if ba["obs_type"] != bb["obs_type"] or len(ba["sites"]) != len(bb["sites"]):
            problems.append(f"block {ba['obs_type']}: structure differs")
            continue
        for sa, sb in zip(ba["sites"], bb["sites"]):
            nm = str(sa["header"].get("name"))
            if sa["site_header_line"] != sb["site_header_line"] or sa["nfreq"] != sb["nfreq"]:
                problems.append(f"{ba['obs_type']} site {nm}: station line / nfreq changed")
                continue
            if sa["nfreq"] == 0:
                continue
            if not np.allclose(sa["freq"], sb["freq"], rtol=1e-8, atol=0.0):
                problems.append(f"{ba['obs_type']} site {nm}: frequencies changed")
            exp_d, exp_e = sa["data"], sa["error"]
            if ba["obs_type"] == "MT" and nm in corrected:
                Cs = corrected[nm]
                M = np.linalg.inv(Cs) if mode == "remove" else Cs
                exp_d, exp_e = apply_matrix_to_mt(sa["data"], sa["error"], M, error_mode)
            for what, e, g in (("data", exp_d, sb["data"]), ("error", exp_e, sb["error"])):
                scale = max(float(np.max(np.abs(e))), 1e-300)
                if not np.allclose(g, e, rtol=1e-7, atol=1e-8 * scale):
                    problems.append(
                        f"{ba['obs_type']} site {nm}: {what} differs from the expected "
                        f"values (max abs diff {np.max(np.abs(g - e)):.3g})")
    return problems


_PANEL_TITLES = {"rho": "Apparent resistivity", "phase": "Phase",
                 "tipper": "Tipper", "pt": "Phase tensor"}


def select_plot_sites(rows: List[dict], spec) -> List[int]:
    """Indices into ``rows`` (= MT sites in file order) for the per-site figures.

    ``spec``: None / 0 / empty list -> none; ``"all"`` -> all corrected sites;
    an integer N -> the N corrected sites with the largest ``||C - I||``
    (returned in file order); a list (or one value) of site IDs and / or names.
    """
    if spec is None or spec is False:
        return []
    if isinstance(spec, str) and spec.strip().lower() == "all":
        return [i for i, r in enumerate(rows) if r["status"] == "corrected"]
    if isinstance(spec, (int, np.integer)) and not isinstance(spec, (bool, np.bool_)):
        n = int(spec)
        if n <= 0:
            return []
        cand = [i for i, r in enumerate(rows) if r["status"] == "corrected"]
        dist = {i: float(np.linalg.norm(rows[i]["C"] - np.eye(2))) for i in cand}
        return sorted(sorted(cand, key=lambda i: -dist[i])[:n])
    items = [spec] if isinstance(spec, str) else list(spec)
    out: List[int] = []
    for it in items:
        hit = [i for i, r in enumerate(rows)
               if str(r["name"]) == str(it) or (r["id"] is not None and str(r["id"]) == str(it))]
        if not hit:
            warnings.warn(f"PLOT_SITES: site {it!r} is not among the MT sites.")
        out.extend(i for i in hit if i not in out)
    return out


def _single_site_parsed(parsed: dict, k: int) -> dict:
    """Minimal parsed observe.dat holding only the k-th MT site of ``parsed``."""
    mt_sites = [s for b in parsed["blocks"] if b["obs_type"] == "MT" for s in b["sites"]]
    return {"preamble_lines": [],
            "blocks": [{"obs_type": "MT", "n_sites_header": 1,
                        "header_line": "MT    1\n", "sites": [copy.deepcopy(mt_sites[k])]}],
            "end_line": "END\n", "tail_lines": []}


def _color_by_component(ax, ncomp: int, fontsize: int = 8) -> None:
    """Give the curves of one "rho" / "phase" panel one colour per component.

    ``fviz.plot_data_ensemble`` continues the colour cycle over all curves it
    draws, so the dashed (new) curve of a component gets another colour than
    its solid (original) curve, and the legend names only the first. The
    plotters draw per site, per component, so curve ``i`` is component
    ``i % ncomp``. Colours only; nothing else of the panel is changed. The
    legend is made again from the recoloured curves.
    """
    import matplotlib.pyplot as plt
    lines = list(ax.lines)
    if ncomp < 1 or not lines or len(lines) % ncomp != 0:
        return
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
    for i, ln in enumerate(lines):
        ln.set_color(cycle[(i % ncomp) % len(cycle)])
    colls = list(ax.collections)
    if len(colls) == len(lines):                       # one error envelope per curve
        for i, c in enumerate(colls):
            c.set_color(cycle[(i % ncomp) % len(cycle)])
    handles, labels = ax.get_legend_handles_labels()
    seen: Dict[str, object] = {}
    for h, lab in zip(handles, labels):
        seen.setdefault(lab, h)
    if seen:
        ax.legend(list(seen.values()), list(seen.keys()), fontsize=fontsize)


def _save_figure(fig, stem: Path, formats: Sequence[str], dpi: int) -> List[str]:
    saved = []
    for fmt in formats:
        p = Path(f"{stem}.{fmt}")
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        saved.append(str(p))
    return saved


def plot_orig_vs_corrected(path_orig: str, path_new: str, rows: List[dict], *,
                           stem: Path, iteration: int, mode: str = "remove",
                           what: Union[str, Sequence[str]] = ("rho", "phase"),
                           comps: str = "xy,yx", show_errors: bool = True,
                           overview: bool = True, overview_nsites: Optional[int] = None,
                           sites: Union[None, int, str, Sequence] = 6,
                           period_limits: Optional[Tuple[float, float]] = None,
                           formats: Sequence[str] = ("png",), dpi: int = 200,
                           title="auto", show: bool = False) -> List[str]:
    """Plot the original and the new observe file with ``fviz.plot_data_ensemble``.

    Original data are drawn solid, the new data dashed. Both files are read
    from disk. One overview figure with all MT sites, and one figure per site
    in ``sites`` (see :func:`select_plot_sites`); a single-site figure is made
    from small temporary single-site files.

    Returns the list of saved files.
    """
    import matplotlib.pyplot as plt
    if not show:
        plt.switch_backend("Agg")

    what_l = [what] if isinstance(what, str) else list(what)
    new_word = "corrected" if mode == "remove" else "distorted"
    saved: List[str] = []

    ncomp = len([c for c in str(comps).split(",") if c.strip()])

    def finish(fig, axs, auto_title, key, fields, out_stem):
        for j, w in enumerate(what_l):
            axs[0, j].set_title(_PANEL_TITLES.get(str(w).lower(), str(w)), fontsize=9)
            if str(w).lower() in ("rho", "phase"):
                _color_by_component(axs[0, j], ncomp)
        txt = utl.resolve_title(title, key=key, auto=auto_title, name="PLOT_TITLE", **fields)
        if txt:
            fig.suptitle(txt, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.95 if txt else 1))
        saved.extend(_save_figure(fig, out_stem, formats, dpi))
        if show:
            plt.show()
        plt.close(fig)

    common = dict(what=what_l, comps=comps, show_errors=show_errors,
                  perlims=period_limits, out=False)
    figsize = (5.0 * len(what_l), 3.8)

    if overview:
        fig, axs = fviz.plot_data_ensemble(
            path_orig, [path_new], [0], n_sites=overview_nsites, figsize=figsize, **common)
        auto = f"all MT sites: original (solid), {new_word} (dashed)"
        finish(fig, axs, auto, "overview", dict(site="all", id="", cond="", iter=iteration,
                                                 new=new_word),
               Path(f"{stem}_overview"))

    idx = select_plot_sites(rows, sites)
    if idx:
        po = fem.read_observe_dat(path_orig, compute_mt_derived=False, bootstrap_n=0)
        pn = fem.read_observe_dat(path_new, compute_mt_derived=False, bootstrap_n=0)
        with tempfile.TemporaryDirectory() as tmp:
            for k in idx:
                r = rows[k]
                fo, fn = Path(tmp) / "orig.dat", Path(tmp) / "new.dat"
                fem.write_observe_dat(_single_site_parsed(po, k), str(fo))
                fem.write_observe_dat(_single_site_parsed(pn, k), str(fn))
                fig, axs = fviz.plot_data_ensemble(
                    str(fo), [str(fn)], [0], n_sites=None, figsize=figsize, **common)
                tag = re.sub(r"[^A-Za-z0-9_.-]", "_", str(r["id"] if r["id"] is not None
                                                           else r["name"]))
                cond = r["cond"]
                auto = (f"site {r['name']}: original (solid), {new_word} (dashed); "
                        f"cond(C) = {cond:.3g}")
                finish(fig, axs, auto, "site",
                       dict(site=r["name"], id=r["id"], cond=f"{cond:.3g}", iter=iteration,
                            new=new_word),
                       Path(f"{stem}_site{tag}"))
    return saved


# =============================================================================
#  Run
# =============================================================================


def main() -> int:
    work_dir = WORK_DIR
    iteration, nrms = select_iteration(work_dir, ITERATION, CNV_FILE)
    print(f"[femtic_distcorr] iteration {iteration}"
          + ("" if nrms is None else f" (nRMS {nrms:.4g})"))

    def name(tpl):
        return None if tpl is None else tpl.format(iter=iteration)

    f_obs = _full_path(work_dir, OBSERVE_FILE)
    f_dis = _full_path(work_dir, name(DISTORTION_FILE))
    f_out = _full_path(work_dir, name(OUTPUT_FILE))
    f_rep = None if REPORT_FILE is None else _full_path(work_dir, name(REPORT_FILE))
    f_h5 = None if RESULTS_H5_FILE is None else _full_path(work_dir, name(RESULTS_H5_FILE))

    for p, what in ((f_obs, "observation"), (f_dis, "distortion")):
        if not p.is_file():
            raise FileNotFoundError(f"{what} file not found: {p}")
    if f_out.resolve() == f_obs.resolve():
        raise ValueError("OUTPUT_FILE is the input file; the original is never overwritten.")

    if f_h5 is not None:
        note = check_distortion_type(f_h5)
        if note:
            print(f"[femtic_distcorr] note: {note}")

    ids, C = load_distortion(f_dis)
    print(f"[femtic_distcorr] {f_dis.name}: {len(ids)} site(s)")

    parsed = fem.read_observe_dat(str(f_obs), compute_mt_derived=False, bootstrap_n=0)
    rows = distcorr(parsed, ids, C, mode=MODE, error_mode=ERROR_MODE,
                    match_by=MATCH_BY, missing=MISSING_SITES, max_cond=MAX_COND)

    fem.write_observe_dat(parsed, str(f_out))
    print(f"[femtic_distcorr] written: {f_out}")

    report = format_report(rows, iteration=iteration, nrms=nrms, mode=MODE,
                           error_mode=ERROR_MODE, observe=str(f_obs),
                           distortion=str(f_dis), output=str(f_out))
    print(report)
    if f_rep is not None:
        f_rep.write_text(report, encoding="ascii", errors="replace")
        print(f"[femtic_distcorr] report: {f_rep}")

    if VERIFY_OUTPUT:
        problems = verify_output(str(f_obs), str(f_out), ids, C, rows,
                                 mode=MODE, error_mode=ERROR_MODE)
        if problems:
            print("[femtic_distcorr] VERIFY FAILED:")
            for p in problems:
                print("   " + p)
            return 1
        print("[femtic_distcorr] verified: the written file matches the expected result.")

    if PLOT:
        # last step: a failure here does not affect the output file
        try:
            saved = plot_orig_vs_corrected(
                str(f_obs), str(f_out), rows, stem=_full_path(work_dir, name(PLOT_FILE)),
                iteration=iteration, mode=MODE, what=PLOT_WHAT, comps=PLOT_COMPS,
                show_errors=PLOT_ERRORS, overview=PLOT_OVERVIEW,
                overview_nsites=PLOT_OVERVIEW_NSITES, sites=PLOT_SITES,
                period_limits=PLOT_PERIOD_LIMITS, formats=PLOT_FORMATS, dpi=PLOT_DPI,
                title=PLOT_TITLE, show=SHOW)
            print(f"[femtic_distcorr] {len(saved)} figure file(s):")
            for s in saved:
                print("   " + s)
        except Exception as exc:  # noqa: BLE001
            print(f"[femtic_distcorr] plots failed ({type(exc).__name__}: {exc}); "
                  "the observation file is written.")

    utl.write_param_summary(fname, out_dir=str(work_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
