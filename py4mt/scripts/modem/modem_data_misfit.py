#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
modem_data_misfit.py

Data-misfit diagnostics for ModEM observed/calculated data file pairs:

  1. "crossplot"  - scatter plots of observed vs. calculated data, one panel
                    per data type and component, log10 frequency as colour.
  2. "histogram"  - normalized-residual histograms in frequency (or period)
                    bands, following Baba (2023): residuals normalized by the
                    observed error only (conventional, RMS1) and by the
                    observed and the synthesized error (proposed, RMS2).
  3. "qq"         - normal Q-Q plots of the same normalized residuals in the
                    same frequency (or period) bands, with the ideal 1:1
                    line, a quartile line (slope = robust scale, intercept
                    = bias), and a pointwise confidence envelope.

All three work on single runs (one observed/calculated pair, optionally the
best/last/given iteration among several calculated candidates) or on an
ensemble of runs (chosen calculated response of each member).

@author   vrath
@project  py4mt
@created  2026-09-26
@modified 2026-09-26

Provenance
----------
Author      : Claude Sonnet 5 (Anthropic)
Generated   : 2026-09-26
Notice      : This code is AI-generated. Review and test it before any
              production use.

Reference
---------
Baba, K. (2023): A simple method to evaluate the uncertainty of
magnetotelluric forward modeling for practical three-dimensional
conductivity structure models. Earth, Planets and Space 75, 67.
https://doi.org/10.1186/s40623-023-01832-5

Based on / template
--------------------
Directly modelled on femtic_data_misfit.py (Claude Sonnet 5, 2026-09-18..21d):
the run-resolution, dataset-building, panel, plotting, fit-measure and
table-formatting machinery below is the same design, adapted for ModEM's
file layout. See "Differences from femtic_data_misfit.py" below.

Why ModEM needs an extra step that FEMTIC does not
---------------------------------------------------
FEMTIC's results_iterN.h5 already bundles the OBSERVED value/error and the
CALCULATED (forward) response for every datum in one file per iteration.
ModEM instead writes them to two separate, identically-formatted ASCII data
files: the observed data file (fixed for a given inversion) and a
calculated/response data file (one per iteration, or one for a single
forward run). modules/modem.py gained three routines for this (2026-09-26):

    modem.read_data_rows(file)              -- parse one ModEM data file
    modem.read_data_calc(file)               -- same, named for a calc file
    modem.read_data_misfit(obs, calc)        -- match an obs/calc pair into
                                                 one per-datum table, the
                                                 ModEM analogue of
                                                 femtic.read_results_data()

Differences from femtic_data_misfit.py
---------------------------------------
- A "run" is an (observed file, calculated file) PAIR, not a single results
  file. RUN_PATHS is therefore a list of (obs, calc) tuples, not bare paths.
- `calc` may be an exact file, a directory (searched with CALC_PATTERN,
  default "*_NLCG_*.dat", ModEM's usual NLCG-iteration naming), or a glob
  pattern; ITERATION ("best"/"last"/int) then picks one candidate exactly
  as in femtic_data_misfit.py (by overall error-normalized RMS, or by the
  iteration number parsed from the file name).
- If `obs` itself is a glob (e.g. an RTO ensemble's per-member observed
  files), `calc` must also be a glob matching the same count; the two sorted
  lists are zipped pairwise into that many (obs, calc) runs -- see
  `expand_run_pairs()`.
- Real-valued vs. complex data types are told apart by column count (10 vs.
  11) while parsing, not guessed from the component-code text; see
  `modem.read_data_rows()`. `modem.modem_is_real_only_datatype()` replaces
  femtic's hardcoded REAL_ONLY_TYPES tuple membership test.
- ModEM gives one error per datum, applied to both the real and imaginary
  part (`modem.read_data_misfit()`); FEMTIC's separate re_err/im_err become
  identical here.
- KEY_ROUND controls the period-rounding tolerance used to match observed
  and calculated rows (see `modem.match_data_rows()`); not needed by FEMTIC,
  which already has matched rows in one file.

Everything else -- dataset building (ensemble alignment/averaging), panel
construction, fit measures, the three plot methods, and the text-table
output -- is unchanged from femtic_data_misfit.py.

Fit measures (from inverse.py): for the error-normalized residuals r,
    nrms  = sqrt(mean(r^2))            expected 1    for r ~ N(0,1)
    r_mae = sqrt(pi/2) * mean(|r|)     expected 1
    r_med = median(|r|) / 0.6745       expected 1
    q95   = 95th percentile of |r|     expected 1.96
plus the mean of r (bias). They are given for both normalizations (err =
conventional, err+syn = Baba's RMS2; the latter only if M > 1) in the overall
table (<prefix>_fit_measures.txt) and in the crossplot and histogram tables.
Robust measures (r_mae, r_med, q95) that stay near their expected values
while nrms is large point to a few outliers rather than a general misfit.

Output location: OUT_DIR = None (default) writes into the run directory
(single: the directory of the chosen calculated file; ensemble: the common
parent directory of the member calculated files); a string forces one
directory.

Usage
-----
    python modem_data_misfit.py                                  # config below
    python modem_data_misfit.py single   obs1.dat calc1.dat obs2.dat calc2.dat
    python modem_data_misfit.py ensemble obs1.dat calc1.dat obs2.dat calc2.dat

Verification
------------
- ast.parse() passes; no non-ASCII characters in the source.
- netCDF4/numba/h5py were not installable in the authoring environment;
  modules/modem.py's new routines (read_data_rows, read_data_calc,
  match_data_rows, read_data_misfit, modem_is_real_only_datatype) were
  smoke-tested directly (netCDF4/numba stubbed, since neither is used by
  these routines) against synthetic two-datatype (Full_Impedance +
  Phase_Tensor), two-site, three-period observed/calculated file pairs:
  column-count-based real/complex detection, block-header DataType
  tracking, row matching (including an unmatched extra calculated row and
  a malformed line), and the shared-error convention were all checked.
- This script's own dataset/panel/plotting machinery is carried over
  unchanged from femtic_data_misfit.py, whose synthetic-data verification
  is documented in femtic_data_misfit_readme.md; it was not re-run here.
- **Not yet run against real ModEM output.** In particular, the assumed
  canonical DataType spelling ("Full_Impedance", "Phase_Tensor", ...) and
  the default CALC_PATTERN ("*_NLCG_*.dat") should be checked against your
  actual files -- both are flagged as unverified assumptions in
  modem_data_split_readme.md for the same reasons.

Changelog
---------
2026-09-26  Claude Sonnet 5 (Anthropic): initial version, using
            femtic_data_misfit.py (2026-09-21d) as the template; obs/calc
            file-pair run resolution and dataset building added on top of
            the new modules/modem.py routines (read_data_rows,
            read_data_calc, match_data_rows, read_data_misfit,
            modem_is_real_only_datatype); crossplot/histogram/Q-Q plotting,
            fit measures and table formatting carried over unchanged.
"""
from __future__ import annotations

import glob
import math
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# py4mt environment: make the modules importable (modem.py, inverse.py)
# ---------------------------------------------------------------------------
PY4MTX_ROOT = os.environ.get("PY4MTX_ROOT", "")
if PY4MTX_ROOT:
    for _pth in (os.path.join(PY4MTX_ROOT, "py4mt", "modules"),
                 os.path.join(PY4MTX_ROOT, "py4mt", "scripts")):
        if os.path.isdir(_pth) and _pth not in sys.path:
            sys.path.insert(0, _pth)

import modem as mdm  # noqa: E402  (read_data_rows, read_data_calc, read_data_misfit, ...)
import inverse as inv  # noqa: E402  (nrms, r_mae, r_med, q95)

# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------
# --- what to run ---

# "single":   every (obs, calc) entry of RUN_PATHS is processed on its own
#             (a list of singles; they do NOT form an ensemble)
# "ensemble": all entries together form one ensemble
RUN_MODE: str = "single"               # "single" | "ensemble"
RUN_PATHS: List[Tuple[str, str]] = [
    ("/media/vrath/LargeBack/ModEM/Annecy/annecy_obs.dat",
     "/media/vrath/LargeBack/ModEM/Annecy/annecy_run/"),
    ]        # (observed file, calculated file/dir/glob) pairs (overridden by argv)
# RUN_MODE: str = "ensemble"
# RUN_PATHS: List[Tuple[str, str]] = [
#     ("/media/vrath/LargeBack/ModEM/Annecy/rto/annecy_rto_*_obs.dat",
#      "/media/vrath/LargeBack/ModEM/Annecy/rto/annecy_rto_*_NLCG_050.dat"),
#     ]        # matching obs/calc globs of equal length -> one pair per member

ITERATION: Union[str, int] = "best"          # "best" | "last" | integer
CALC_PATTERN: str = "*_NLCG_*.dat"           # searched if `calc` is a directory
KEY_ROUND: int = 8                           # period-rounding decimals for
                                              # obs/calc row matching (see
                                              # modem.match_data_rows)
METHODS: Tuple[str, ...] = ("crossplot", "histogram", "qq")

# --- ensemble handling ---
ENSEMBLE_OBS: str = "mean"     # observed values/errors: "mean" over members
                               # (RTO-perturbed data average back to the
                               # original) or "first" member
REF_MEMBER: Optional[int] = None   # RMS1 reference: None = ensemble mean;
                                   # integer = calculated response of that
                                   # member (as in Baba's "6th model")

# --- output ---
OUT_DIR: Optional[str] = None      # None -> into the run directory (single:
                                   # each run's own; ensemble: common parent
                                   # of the member directories); or a path
OUT_PREFIX: Optional[str] = "Misfits"  # None -> derived from run / ensemble;
                                   # if set with several singles it is
                                   # prepended: <OUT_PREFIX>_<run>_iter<N>
PLOT_FORMATS: Tuple[str, ...] = (".pdf",)   # a tuple: note the comma!
PLOT_DPI: int = 300
WRITE_STATS: bool = True
FIT_MEASURES: bool = True   # overall table of fit measures (per data type
                            # and pooled) -> <prefix>_fit_measures.txt
SHOW: bool = False

# --- selection (None = everything) ---
DATATYPES: Optional[Sequence[str]] = None     # e.g. ["Full_Impedance"]
COMPONENTS: Optional[Sequence[str]] = None    # e.g. ["ZXY", "ZYX"]

# --- crossplot ---
NCOLS: int = 4
PANEL_SIZE: float = 3.3
EQUAL_ASPECT: bool = True
CMAP: str = "turbo"
COLOR_RANGE: str = "global"   # "global" | "panel" (range of log10 f)
MARKER_SIZE: float = 14.0
ALPHA: float = 0.85
LOG_RHO: bool = True          # log-log axes for rho* components
SHOW_ERRORBARS: bool = False

# --- residual histograms ---
HIST_BAND_AXIS: str = "frequency"   # "frequency" | "period"
HIST_BANDS_PER_DECADE: int = 2
HIST_BY_COMPONENT: bool = False     # False: one figure per data type
                                    # (components pooled, as in Baba 2023)
HIST_LIMIT: float = 10.0            # histogram range +/- (values outside
                                    # are counted, not drawn)
HIST_BIN_WIDTH: float = 0.5
HIST_MIN_N: int = 5                 # skip bands with fewer values
HIST_NCOLS: int = 4
HIST_PANEL_SIZE: float = 3.0

# --- Q-Q plots (same normalized residuals as the histograms) ---
QQ_BAND_AXIS: str = "frequency"     # "frequency" | "period"
QQ_BANDS_PER_DECADE: int = 2
QQ_BY_COMPONENT: bool = False       # False: per data type, components pooled
QQ_LIMIT: float = 10.0              # max |sample quantile| shown
QQ_EQUAL_AXES: bool = False         # True: same range on both axes, square
                                    # panels (1:1 line at 45 degrees)
QQ_CONFIDENCE: Optional[float] = 0.95   # pointwise envelope; None = off
QQ_MIN_N: int = 10                  # skip bands with fewer values
QQ_NCOLS: int = 4
QQ_PANEL_SIZE: float = 3.0


# ---------------------------------------------------------------------------
# Run / iteration selection
# ---------------------------------------------------------------------------
def _iter_from_name(path: str) -> int:
    """Iteration number from a calculated-data file name: ModEM's usual
    '..._NLCG_037.dat' naming first, else the last run of digits in the
    file's basename, else -1 (unresolved)."""
    base = os.path.basename(path)
    m = re.search(r"NLCG[_-]?(\d+)", base, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)(?!.*\d)", base)
    return int(m.group(1)) if m else -1


def data_rms(d: dict) -> float:
    """
    Overall error-normalized RMS of a matched dataset's /rows.

    Real and imaginary parts count separately; the imaginary part is left
    out for real-valued data types (`modem.modem_is_real_only_datatype`).
    Non-finite values and non-positive errors are skipped. Returns NaN if
    nothing is usable.
    """
    rows = d["rows"]
    if rows.size == 0:
        return float("nan")
    dn = rows["datatype_name"]
    uniq = np.unique(dn)
    real_only = np.array([mdm.modem_is_real_only_datatype(u) for u in uniq])
    real_only_set = set(uniq[real_only].tolist())
    cpx = ~np.isin(dn, list(real_only_set)) if real_only_set else np.ones(dn.size, bool)
    r_all = []
    for obs, cal, err, m in (
        (rows["re_val"], rows["cal_re"], rows["re_err"], np.ones(rows.size, bool)),
        (rows["im_val"], rows["cal_im"], rows["im_err"], cpx),
    ):
        obs = np.asarray(obs, float)
        cal = np.asarray(cal, float)
        err = np.asarray(err, float)
        ok = m & np.isfinite(obs) & np.isfinite(cal) & np.isfinite(err) & (err > 0)
        r_all.append((obs[ok] - cal[ok]) / err[ok])
    r = np.concatenate(r_all)
    return float(np.sqrt(np.mean(r**2))) if r.size else float("nan")


def resolve_run(obs_path: str, calc_spec: str, *, iteration: Union[str, int],
                calc_pattern: str, key_round: int) -> dict:
    """
    Match a fixed observed file against one calculated-data file, chosen
    from `calc_spec`.

    Parameters
    ----------
    obs_path : str
        Observed ModEM data file (fixed for this run).
    calc_spec : str
        A calculated-data file, a directory (searched with `calc_pattern`),
        or a glob pattern.
    iteration : "best", "last" or int
        Ignored if `calc_spec` resolves to exactly one file. "best": lowest
        overall error-normalized RMS against `obs_path`; "last": highest
        iteration number parsed from the file name; int: exact match.

    Returns
    -------
    dict with keys obs, run, path, iter, rms, data, n_candidates.
    """
    if os.path.isfile(calc_spec):
        cands = [calc_spec]
    elif os.path.isdir(calc_spec):
        cands = sorted(glob.glob(os.path.join(calc_spec, calc_pattern)))
    elif any(ch in calc_spec for ch in "*?["):
        cands = sorted(glob.glob(calc_spec))
    else:
        raise FileNotFoundError(f"resolve_run: {calc_spec} not found.")
    if not cands:
        raise FileNotFoundError(
            f"resolve_run: no candidate calculated file for {calc_spec!r} "
            f"(pattern {calc_pattern!r}).")

    single_file = len(cands) == 1 and os.path.isfile(cands[0]) and os.path.isfile(calc_spec)

    best = None
    best_score = float("inf")
    n_ok = 0
    for c in cands:
        try:
            d = mdm.read_data_misfit(obs_path, c, key_round=key_round, out=False)
        except (FileNotFoundError, OSError, ValueError) as exc:
            print(f"modem_data_misfit: skipping {c}: {exc}")
            continue
        if d["nRows"] == 0:
            continue
        n_ok += 1
        it = _iter_from_name(c)
        rms = data_rms(d)
        if single_file:
            score = 0.0
        elif iteration == "best":
            score = rms if np.isfinite(rms) else float("inf")
        elif iteration == "last":
            score = -float(it)
        else:
            score = 0.0 if it == int(iteration) else float("inf")
        if best is None or score < best_score:
            best, best_score = dict(obs=obs_path, run=calc_spec, path=c, iter=it,
                                    rms=rms, data=d), score
    if best is None or (best_score == float("inf") and n_ok > 0
                        and iteration not in ("best", "last")):
        raise ValueError(
            f"resolve_run: no usable calculated file (iteration={iteration!r}) "
            f"for {calc_spec!r} against {obs_path!r}.")
    best["n_candidates"] = n_ok
    return best


def expand_run_pairs(run_paths: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Expand each (obs_spec, calc_spec) entry into concrete (obs_file,
    calc_spec) pairs.

    If `obs_spec` is a glob pattern it is expanded; if `calc_spec` is also
    a glob it must expand to exactly as many entries (sorted, zipped
    pairwise) -- e.g. an RTO ensemble's per-member observed/calculated file
    pairs. Otherwise `calc_spec` is passed through unresolved (a single
    fixed calculated file, directory, or glob handled by `resolve_run`).
    Non-glob entries are kept as-is.
    """
    out: List[Tuple[str, str]] = []
    for obs_spec, calc_spec in run_paths:
        if any(ch in obs_spec for ch in "*?["):
            obs_list = sorted(glob.glob(obs_spec))
            if not obs_list:
                raise FileNotFoundError(
                    f"expand_run_pairs: no match for observed glob {obs_spec!r}.")
            if any(ch in calc_spec for ch in "*?["):
                calc_list = sorted(glob.glob(calc_spec))
                if len(calc_list) != len(obs_list):
                    raise ValueError(
                        f"expand_run_pairs: observed glob {obs_spec!r} matched "
                        f"{len(obs_list)} file(s) but calculated glob "
                        f"{calc_spec!r} matched {len(calc_list)} -- cannot "
                        f"zip pairwise.")
                out.extend(zip(obs_list, calc_list))
            else:
                out.extend((o, calc_spec) for o in obs_list)
        else:
            out.append((obs_spec, calc_spec))
    return out


# ---------------------------------------------------------------------------
# Building the (ensemble) dataset
# ---------------------------------------------------------------------------
def _row_keys(rows: np.ndarray) -> Tuple[np.ndarray, ...]:
    f = np.asarray(rows["freq"], float)
    logf = np.round(np.log10(np.where(f > 0, f, np.nan)), 9)
    return (np.asarray(rows["datatype_name"]), np.asarray(rows["site"]),
            np.asarray(rows["component_name"]), np.nan_to_num(logf, nan=-999.0))


def align_members(datas: Sequence[dict]) -> List[np.ndarray]:
    """
    Row indices into each member's rows such that row k is the same datum
    (datatype, site, component, frequency) in every member.
    """
    keys = [_row_keys(d["rows"]) for d in datas]
    k0 = keys[0]
    same = all(len(k[0]) == len(k0[0]) and all(np.array_equal(a, b) for a, b in zip(k0, k))
               for k in keys[1:])
    if same:
        return [np.arange(len(k0[0])) for _ in datas]
    lists = [list(zip(*[a.tolist() for a in k])) for k in keys]
    maps = [{key: i for i, key in enumerate(lst)} for lst in lists]
    common = [key for key in lists[0] if all(key in m for m in maps[1:])]
    print(f"modem_data_misfit: members differ in row layout; "
          f"{len(common)} of {len(lists[0])} rows are common to all.")
    return [np.array([m[key] for key in common], dtype=np.int64) for m in maps]


def build_dataset(datas: Sequence[dict], *, obs_mode: str = "mean",
                  ref_member: Optional[int] = None) -> dict:
    """
    Combine M >= 1 matched datasets (`modem.read_data_misfit()` output)
    into one set of aligned arrays.

    Returns a dict with M and per-row arrays: freq, site_id, component,
    datatype_name, component_name, obs_re/im, err_re/im, cal_re/im (ensemble
    mean), ref_re/im (reference response for RMS1), eps_syn (Baba Eq. 13,
    zero for M = 1), plus n_dropped_nan.
    """
    M = len(datas)
    idx = align_members(datas)
    rows = [d["rows"][ix] for d, ix in zip(datas, idx)]

    def stack(name: str) -> np.ndarray:
        return np.vstack([np.asarray(r[name], float) for r in rows])

    cal_re, cal_im = stack("cal_re"), stack("cal_im")
    ok = np.all(np.isfinite(cal_re) & np.isfinite(cal_im), axis=0)
    n_drop = int(np.count_nonzero(~ok))

    def obs_like(name: str) -> np.ndarray:
        s = stack(name)
        return s.mean(axis=0) if obs_mode == "mean" else s[0]

    mu_re, mu_im = cal_re.mean(axis=0), cal_im.mean(axis=0)
    if M > 1:
        var = (((cal_re - mu_re) ** 2 + (cal_im - mu_im) ** 2).sum(axis=0)) / (M - 1)
        eps = np.sqrt(var / M)
    else:
        eps = np.zeros(mu_re.shape)
    if ref_member is None:
        ref_re, ref_im = mu_re, mu_im
    else:
        ref_re, ref_im = cal_re[ref_member], cal_im[ref_member]

    r0 = rows[0]
    ds = dict(
        M=M,
        freq=np.asarray(r0["freq"], float),
        site_id=np.asarray(r0["site"]),
        component=np.asarray(r0["component_name"]),
        datatype_name=np.asarray(datas[0]["datatype_name"])[idx[0]],
        component_name=np.asarray(datas[0]["component_name"])[idx[0]],
        obs_re=obs_like("re_val"), obs_im=obs_like("im_val"),
        err_re=obs_like("re_err"), err_im=obs_like("im_err"),
        cal_re=mu_re, cal_im=mu_im, ref_re=ref_re, ref_im=ref_im,
        eps_syn=eps,
    )
    for k in list(ds):
        if isinstance(ds[k], np.ndarray) and ds[k].shape[:1] == ok.shape:
            ds[k] = ds[k][ok]
    ds["n_dropped_nan"] = n_drop
    return ds


# ---------------------------------------------------------------------------
# Panels (one per datatype / component / part)
# ---------------------------------------------------------------------------
def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def build_panels(ds: dict, *, datatypes: Optional[Sequence[str]] = None,
                 components: Optional[Sequence[str]] = None) -> Dict[str, List[dict]]:
    """Split a dataset into panels: {datatype_name: [panel, ...]}."""
    dt_all = ds["datatype_name"]
    if dt_all.size == 0:
        raise ValueError("build_panels: dataset is empty.")
    _, first = np.unique(dt_all, return_index=True)
    dtype_order = [dt_all[i] for i in np.sort(first)]

    panels: Dict[str, List[dict]] = {}
    for dname in dtype_order:
        if datatypes is not None and dname not in datatypes:
            continue
        m_dt = dt_all == dname
        for code in np.unique(ds["component"][m_dt]):
            m = m_dt & (ds["component"] == code)
            cname = str(ds["component_name"][m][0])
            if components is not None and cname not in components:
                continue
            has_im = (not mdm.modem_is_real_only_datatype(dname)) and (
                np.any(np.nan_to_num(ds["obs_im"][m]) != 0.0)
                or np.any(np.nan_to_num(ds["cal_im"][m]) != 0.0))
            parts = [("re", "obs_re", "cal_re", "ref_re", "err_re")]
            if has_im:
                parts.append(("im", "obs_im", "cal_im", "ref_im", "err_im"))
            freq = ds["freq"][m]
            for part, ko, kc, kr, ke in parts:
                obs, cal, ref = ds[ko][m], ds[kc][m], ds[kr][m]
                err, eps = ds[ke][m], ds["eps_syn"][m]
                ok = (np.isfinite(obs) & np.isfinite(cal) & np.isfinite(ref)
                      & np.isfinite(freq) & (freq > 0.0))
                if has_im:
                    label = f"{'Re' if part == 're' else 'Im'} {cname}"
                    tag = part
                else:
                    label, tag = cname, "real"
                panels.setdefault(dname, []).append(dict(
                    datatype=dname, component=cname, part=tag, label=label,
                    freq=freq[ok], obs=obs[ok], cal=cal[ok], ref=ref[ok],
                    err=err[ok], eps=eps[ok], n_dropped=int(np.count_nonzero(~ok)),
                ))
    return panels


def fit_measures(r: np.ndarray) -> dict:
    """
    Fit measures of error-normalized residuals (non-finite values ignored):
    n, nrms, r_mae, r_med, q95 (all from inverse.py) and the mean (bias).
    Expected for N(0,1) residuals: nrms = r_mae = r_med = 1, q95 = 1.96.
    """
    r = np.asarray(r, dtype=float).ravel()
    r = r[np.isfinite(r)]
    return dict(
        n=int(r.size),
        nrms=float(inv.nrms(r)),
        r_mae=float(inv.r_mae(r)),
        r_med=float(inv.r_med(r)),
        q95=float(inv.q95(r)),
        mean=float(np.mean(r)) if r.size else float("nan"),
    )


def panel_stats(p: dict) -> dict:
    """
    n, plain rms, and the fit measures of the panel's normalized residuals:
    m1 (conventional, obs error only) and m2 (obs + synthesized error);
    nrms1 / nrms2 are shortcuts for m1["nrms"] / m2["nrms"].
    """
    res = p["obs"] - p["cal"]
    n = res.size
    rms = float(np.sqrt(np.mean(res**2))) if n else float("nan")
    _, r1, r2 = normalized_residuals(p)
    m1, m2 = fit_measures(r1), fit_measures(r2)
    return dict(n=n, n_dropped=p["n_dropped"], rms=rms, m1=m1, m2=m2,
                nrms1=m1["nrms"], nrms2=m2["nrms"])


def normalized_residuals(p: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (freq, r1, r2) for the rows with finite, positive error:
      r1 = (obs - ref) / err                       (conventional)
      r2 = (obs - mean) / sqrt(err^2 + eps_syn^2)  (Baba 2023, RMS2)
    """
    ok = np.isfinite(p["err"]) & (p["err"] > 0.0)
    err, eps = p["err"][ok], p["eps"][ok]
    r1 = (p["obs"][ok] - p["ref"][ok]) / err
    r2 = (p["obs"][ok] - p["cal"][ok]) / np.sqrt(err**2 + eps**2)
    return p["freq"][ok], r1, r2


def global_logf_range(panels: Dict[str, List[dict]]) -> Optional[Tuple[float, float]]:
    fs = [p["freq"] for lst in panels.values() for p in lst if p["freq"].size]
    if not fs:
        return None
    f = np.concatenate(fs)
    return float(np.log10(f.min())), float(np.log10(f.max()))


# ---------------------------------------------------------------------------
# Method 1: crossplots
# ---------------------------------------------------------------------------
def _axis_limits(x: np.ndarray, y: np.ndarray, *, log: bool) -> Tuple[float, float]:
    v = np.concatenate([x, y])
    lo, hi = float(np.min(v)), float(np.max(v))
    if log:
        if lo == hi:
            return lo / 2.0, hi * 2.0
        f = (hi / lo) ** 0.05
        return lo / f, hi * f
    if lo == hi:
        d = abs(lo) * 0.1 if lo != 0.0 else 1.0
        return lo - d, hi + d
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def plot_crossplot_datatype(
    dname: str, panels: Sequence[dict], *, title_extra: str, has_syn: bool,
    ncols: int, panel_size: float, cmap: str,
    color_range: Optional[Tuple[float, float]], marker_size: float,
    alpha: float, log_rho: bool, show_errorbars: bool, equal_aspect: bool,
):
    """One figure with all crossplot panels of a data type."""
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    n = len(panels)
    nc = max(1, min(ncols, n))
    nr = int(math.ceil(n / nc))
    fig, axes = plt.subplots(
        nr, nc, figsize=(panel_size * nc + 1.2, panel_size * nr + 0.6),
        squeeze=False, constrained_layout=True)
    axf = axes.ravel()

    sm_norm = None
    for k, p in enumerate(panels):
        ax = axf[k]
        x, y, err = p["obs"], p["cal"], p["err"]
        if x.size == 0:
            ax.text(0.5, 0.5, "no valid data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(p["label"], fontsize=10)
            continue
        logf = np.log10(p["freq"])
        if color_range is not None:
            norm = Normalize(vmin=color_range[0], vmax=color_range[1])
        else:
            norm = Normalize(vmin=float(logf.min()), vmax=float(logf.max()))
        if sm_norm is None:
            sm_norm = norm
        use_log = (log_rho and p["component"].lower().startswith("rho")
                   and bool(np.all(x > 0.0)) and bool(np.all(y > 0.0)))
        if show_errorbars:
            xerr = np.where(np.isfinite(err) & (err > 0.0), err, 0.0)
            ax.errorbar(x, y, xerr=xerr, fmt="none", ecolor="0.75",
                        elinewidth=0.5, zorder=1)
        ax.scatter(x, y, c=logf, cmap=cmap, norm=norm, s=marker_size,
                   alpha=alpha, edgecolors="none", zorder=2)
        if use_log:
            ax.set_xscale("log")
            ax.set_yscale("log")
        lo, hi = _axis_limits(x, y, log=use_log)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.plot([lo, hi], [lo, hi], color="k", lw=0.8, zorder=3)
        if equal_aspect:
            ax.set_aspect("equal", adjustable="box")
        st = panel_stats(p)
        txt = (f"n={st['n']}\nnRMS1={st['nrms1']:.2f}\nnRMS2={st['nrms2']:.2f}"
               if has_syn else f"n={st['n']}\nnRMS={st['nrms1']:.2f}")
        ax.text(0.04, 0.96, txt, transform=ax.transAxes, ha="left", va="top",
                fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.8))
        ax.set_title(p["label"], fontsize=10)
        ax.set_xlabel("observed", fontsize=9)
        ax.set_ylabel("calculated (ensemble mean)" if has_syn else "calculated", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, which="major", lw=0.3, alpha=0.5)
    for ax in axf[n:]:
        ax.set_visible(False)
    if sm_norm is not None:
        sm = cm.ScalarMappable(norm=sm_norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=list(axf[:n]), shrink=0.85, pad=0.02)
        cb.set_label("log10 frequency (Hz)")
    fig.suptitle(f"{dname}: observed vs. calculated, {title_extra}", fontsize=12)
    return fig


# ---------------------------------------------------------------------------
# Method 2: normalized-residual histograms in frequency / period bands
# ---------------------------------------------------------------------------
def band_index(freq: np.ndarray, *, per_decade: int, axis: str) -> np.ndarray:
    """Integer band number; bands are anchored at whole decades."""
    x = np.log10(freq) if axis == "frequency" else -np.log10(freq)
    return np.floor(x * per_decade + 1e-9).astype(int)


def band_label(k: int, *, per_decade: int, axis: str) -> str:
    lo, hi = 10.0 ** (k / per_decade), 10.0 ** ((k + 1) / per_decade)
    unit = "Hz" if axis == "frequency" else "s"
    return f"{lo:.3g} - {hi:.3g} {unit}"


def select_bands(freq: np.ndarray, *, per_decade: int, axis: str,
                 min_n: int) -> List[Tuple[str, np.ndarray]]:
    """[("all bands", mask), (band label, mask), ...]; sparse bands dropped."""
    bidx = band_index(freq, per_decade=per_decade, axis=axis)
    sel = [("all bands", np.ones(freq.size, bool))]
    for k in np.unique(bidx):
        m = bidx == k
        if np.count_nonzero(m) >= min_n:
            sel.append((band_label(int(k), per_decade=per_decade, axis=axis), m))
    return sel


def _rms(a: np.ndarray) -> float:
    return float(inv.nrms(a))


def plot_residual_histograms(
    title: str, freq: np.ndarray, r1: np.ndarray, r2: np.ndarray, *,
    has_syn: bool, per_decade: int, axis: str, ncols: int, panel_size: float,
    limit: float, bin_width: float, min_n: int, run_label: str = "",
):
    """
    Histograms of normalized residuals: first panel pools all bands, then
    one panel per frequency (period) band. Bars: residuals normalized by
    sqrt(err^2 + eps_syn^2) (RMS2) if `has_syn`, else by err (RMS1). With
    `has_syn` the RMS1 histogram is added as a step outline. Red: N(0, RMS^2)
    of the bars; blue dashed: +/- RMS of the bars (Baba 2023, Fig. 10a).

    Returns (figure, stats) where stats is a list of dicts with keys
    band, n, m1, m2 (fit_measures of the RMS1- / RMS2-normalized residuals),
    n_out (values of the plotted variant outside +/- limit).
    """
    import matplotlib.pyplot as plt

    sel = select_bands(freq, per_decade=per_decade, axis=axis, min_n=min_n)

    n = len(sel)
    nc = max(1, min(ncols, n))
    nr = int(math.ceil(n / nc))
    fig, axes = plt.subplots(
        nr, nc, figsize=(panel_size * nc + 0.4, panel_size * nr * 0.85 + 0.6),
        squeeze=False, sharex=True, constrained_layout=True)
    axf = axes.ravel()

    edges = np.arange(-limit, limit + 0.5 * bin_width, bin_width)
    centres = 0.5 * (edges[:-1] + edges[1:])
    xs = np.linspace(-limit, limit, 400)
    stats: List[dict] = []

    for k, (label, m) in enumerate(sel):
        ax = axf[k]
        a1, a2 = r1[m], r2[m]
        main = a2 if has_syn else a1
        nn = main.size
        rms1, rms2 = _rms(a1), _rms(a2)
        rms_main = rms2 if has_syn else rms1
        n_out = int(np.count_nonzero(np.abs(main) > limit))

        counts, _ = np.histogram(main, bins=edges)
        ax.bar(centres, 100.0 * counts / nn, width=bin_width, color="0.75",
               edgecolor="0.3", linewidth=0.4,
               label="RMS2-normalized" if has_syn else "normalized")
        if has_syn:
            c1, _ = np.histogram(a1, bins=edges)
            ax.step(centres, 100.0 * c1 / nn, where="mid", color="tab:orange",
                    linewidth=1.0, label="RMS1-normalized")
        if np.isfinite(rms_main) and rms_main > 0.0:
            g = 100.0 * bin_width * np.exp(-xs**2 / (2.0 * rms_main**2)) \
                / (rms_main * math.sqrt(2.0 * math.pi))
            ax.plot(xs, g, color="red", linewidth=1.0, label="N(0, RMS^2)")
            for s in (-rms_main, rms_main):
                ax.axvline(s, color="tab:blue", linestyle="--", linewidth=0.8)
        ax.set_xlim(-limit, limit)
        txt = (f"n={nn}\nRMS1={rms1:.2f}\nRMS2={rms2:.2f}" if has_syn
               else f"n={nn}\nRMS={rms1:.2f}")
        if n_out:
            txt += f"\nout: {n_out}"
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, ha="left", va="top",
                fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.8))
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("normalized residual", fontsize=9)
        ax.set_ylabel("frequency [%]", fontsize=9)
        ax.tick_params(labelsize=8)
        if k == 0:
            ax.legend(fontsize=7, loc="upper right")
        stats.append(dict(band=label, n=nn, m1=fit_measures(a1),
                          m2=fit_measures(a2), n_out=n_out))
    for ax in axf[n:]:
        ax.set_visible(False)
    fig.suptitle(f"{title}: normalized residuals, {axis} bands "
                 f"({per_decade} per decade)" + (f"\n{run_label}" if run_label else ""),
                 fontsize=12)
    return fig, stats


def qq_points(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normal Q-Q coordinates: theoretical N(0,1) quantiles at plotting
    positions p_i = (i - 0.5) / n, and the sorted sample.
    """
    from scipy.stats import norm
    a = np.sort(a[np.isfinite(a)])
    p = (np.arange(1, a.size + 1) - 0.5) / a.size
    return norm.ppf(p), a


def qq_quartile_line(sample_sorted: np.ndarray) -> Tuple[float, float]:
    """(slope, intercept) of the line through the sample's and the standard
    normal's first and third quartiles (robust scale and location)."""
    from scipy.stats import norm
    q1, q3 = np.quantile(sample_sorted, [0.25, 0.75])
    z1, z3 = norm.ppf([0.25, 0.75])
    slope = float((q3 - q1) / (z3 - z1))
    return slope, float(q1 - slope * z1)


def plot_qq_bands(
    title: str, freq: np.ndarray, r1: np.ndarray, r2: np.ndarray, *,
    has_syn: bool, per_decade: int, axis: str, ncols: int, panel_size: float,
    limit: float, confidence: Optional[float], min_n: int,
    equal_axes: bool = False, run_label: str = "",
):
    """
    Normal Q-Q plots of the normalized residuals: first panel pools all
    bands, then one panel per frequency (period) band.

    Dots: residuals normalized by sqrt(err^2 + eps_syn^2) (RMS2) if
    `has_syn`, else by err (RMS1); with `has_syn` the RMS1 residuals are
    added as orange dots. Black line: ideal N(0,1) (calibrated errors).
    Red dashed: quartile line of the plotted dots (slope = robust scale,
    intercept = bias). Gray band: pointwise confidence envelope around the
    quartile line, from the asymptotic standard error of order statistics,
    se_i = sqrt(p_i (1 - p_i) / n) / phi(z_i), so it judges the Gaussian
    shape irrespective of scale and bias. With `equal_axes` both axes get
    the same range (at most +/- `limit`) and the panels are square.

    Returns (figure, stats); stats rows have keys band, n, slope1, slope2,
    icpt, out_pct (percentage of the dots outside the envelope).
    """
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    sel = select_bands(freq, per_decade=per_decade, axis=axis, min_n=min_n)
    n = len(sel)
    nc = max(1, min(ncols, n))
    nr = int(math.ceil(n / nc))
    fig, axes = plt.subplots(
        nr, nc, figsize=(panel_size * nc + 0.4, panel_size * nr + 0.6),
        squeeze=False, sharex=True, sharey=True, constrained_layout=True)
    axf = axes.ravel()

    main_all = r2 if has_syn else r1
    ymax = min(limit, 1.05 * float(np.max(np.abs(main_all)))) if main_all.size else limit
    zc = norm.ppf(0.5 + 0.5 * confidence) if confidence else None
    stats: List[dict] = []

    for k, (label, m) in enumerate(sel):
        ax = axf[k]
        a1, a2 = r1[m], r2[m]
        main = a2 if has_syn else a1
        z, q = qq_points(main)
        nn = q.size
        slope2, icpt = qq_quartile_line(q)
        slope1 = qq_quartile_line(np.sort(a1))[0] if has_syn else slope2

        if has_syn:
            z1, q1 = qq_points(a1)
            ax.scatter(z1, q1, s=6, color="tab:orange", alpha=0.7,
                       edgecolors="none", label="RMS1-normalized", zorder=2)
        ax.scatter(z, q, s=6, color="0.25", edgecolors="none", zorder=3,
                   label="RMS2-normalized" if has_syn else "normalized")
        zl = np.array([z.min(), z.max()])
        ax.plot(zl, zl, color="k", linewidth=0.8, label="N(0,1)", zorder=4)
        fit = icpt + slope2 * z
        ax.plot(z, fit, color="red", linestyle="--", linewidth=0.9,
                label="quartile line", zorder=4)
        out_pct = float("nan")
        if zc is not None:
            p = (np.arange(1, nn + 1) - 0.5) / nn
            se = np.sqrt(p * (1.0 - p) / nn) / norm.pdf(z)
            lo, hi = fit - zc * slope2 * se, fit + zc * slope2 * se
            ax.fill_between(z, lo, hi, color="0.5", alpha=0.25, linewidth=0, zorder=1)
            out_pct = 100.0 * float(np.count_nonzero((q < lo) | (q > hi))) / nn
        txt = (f"n={nn}\nslope1={slope1:.2f}\nslope2={slope2:.2f}" if has_syn
               else f"n={nn}\nslope={slope2:.2f}")
        txt += f"\nicpt={icpt:.2f}"
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, ha="left", va="top",
                fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.8))
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("theoretical N(0,1) quantile", fontsize=9)
        ax.set_ylabel("sample quantile", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, linewidth=0.3, alpha=0.5)
        if k == 0:
            ax.legend(fontsize=7, loc="lower right")
        stats.append(dict(band=label, n=nn, slope1=slope1, slope2=slope2,
                          icpt=icpt, out_pct=out_pct))
    if equal_axes:
        zmax = float(np.max(np.abs(qq_points(main_all)[0]))) if main_all.size else 1.0
        lim = min(limit, max(ymax, 1.05 * zmax))
        axf[0].set_xlim(-lim, lim)
        axf[0].set_ylim(-lim, lim)
        for ax in axf[:n]:
            ax.set_aspect("equal", adjustable="box")
    else:
        axf[0].set_ylim(-ymax, ymax)
    for ax in axf[n:]:
        ax.set_visible(False)
    env = f", {100 * confidence:.0f}% envelope" if confidence else ""
    fig.suptitle(f"{title}: normal Q-Q, {axis} bands ({per_decade} per decade){env}"
                 + (f"\n{run_label}" if run_label else ""), fontsize=12)
    return fig, stats


def group_panels(panels: Dict[str, List[dict]], *, by_component: bool) -> Dict[str, List[dict]]:
    """Group panels for the histograms: per data type, or per data type + component."""
    groups: Dict[str, List[dict]] = {}
    for dname, lst in panels.items():
        if by_component:
            for p in lst:
                groups.setdefault(f"{dname}_{p['component']}", []).append(p)
        else:
            groups[dname] = list(lst)
    return groups


# ---------------------------------------------------------------------------
# Text tables (plain ASCII)
# ---------------------------------------------------------------------------
FIT_MEASURE_NOTE = (
    "# r = error-normalized residuals; norm 'err' = conventional (Baba RMS1),\n"
    "# 'err+syn' = obs + synthesized error (Baba RMS2, ensemble only).\n"
    "# nrms = sqrt(mean(r^2)); r_mae = sqrt(pi/2)*mean(|r|); r_med = median(|r|)/0.6745;\n"
    "# q95 = 95th percentile of |r|; mean = bias.\n"
    "# Expected for N(0,1) residuals: nrms = r_mae = r_med = 1, q95 = 1.96, mean = 0."
)
_MK = ("nrms", "r_mae", "r_med", "q95")
_MN = {"nrms": "nrms", "r_mae": "rmae", "r_med": "rmed", "q95": "q95"}


def format_crossplot_stats(panels: Dict[str, List[dict]], *, has_syn: bool) -> str:
    cols = [f"{_MN[k]}{sfx}" for k in _MK for sfx in (("1", "2") if has_syn else ("",))]
    hdr = (f"{'datatype':<26} {'component':<8} {'part':<5} {'n':>7} "
           f"{'dropped':>8} {'rms':>12}" + "".join(f" {c:>8}" for c in cols))
    lines = ["=" * len(hdr), hdr, "-" * len(hdr)]
    for dname, lst in panels.items():
        for p in lst:
            s = panel_stats(p)
            vals = [s[m][k] for k in _MK for m in (("m1", "m2") if has_syn else ("m1",))]
            lines.append(f"{dname:<26} {p['component']:<8} {p['part']:<5} {s['n']:>7d} "
                         f"{s['n_dropped']:>8d} {s['rms']:>12.4e}"
                         + "".join(f" {v:>8.3f}" for v in vals))
    lines.append("=" * len(hdr))
    return "\n".join(lines)


def _measure_rows(label_a: str, label_b: str, n: int, m1: dict, m2: dict, *,
                  has_syn: bool, extra: Optional[Tuple[str, str]] = None) -> List[str]:
    """Table rows (norm 'err' and, if has_syn, 'err+syn') for one residual set."""
    out = []
    variants = [("err", m1)] + ([("err+syn", m2)] if has_syn else [])
    for k, (nm, m) in enumerate(variants):
        ex = "" if extra is None else f" {extra[k] if k < len(extra) else '':>6}"
        out.append(f"{label_a:<26} " + (f"{label_b:<24} " if label_b else "")
                   + f"{nm:<8} {m['n']:>7d} "
                   f"{m['nrms']:>8.3f} {m['r_mae']:>8.3f} {m['r_med']:>8.3f} "
                   f"{m['q95']:>8.3f} {m['mean']:>8.3f}{ex}")
    return out


def _measure_header(first: str, second: str, *, extra: str = "") -> str:
    return (f"{first:<26} " + (f"{second:<24} " if second else "")
            + f"{'norm':<8} {'n':>7} {'nrms':>8} {'r_mae':>8} "
            f"{'r_med':>8} {'q95':>8} {'mean':>8}" + (f" {extra:>6}" if extra else ""))


def format_hist_stats(all_stats: Dict[str, List[dict]], *, has_syn: bool) -> str:
    hdr = _measure_header("group", "band", extra="n_out")
    lines = ["=" * len(hdr), hdr, "-" * len(hdr)]
    for g, lst in all_stats.items():
        for s in lst:
            # n_out refers to the plotted variant (err+syn if has_syn, else err)
            extra = ("", str(s["n_out"])) if has_syn else (str(s["n_out"]),)
            lines.extend(_measure_rows(g, s["band"], s["n"], s["m1"], s["m2"],
                                       has_syn=has_syn, extra=extra))
    lines.append("=" * len(hdr))
    return "\n".join(lines)


def format_fit_measures(panels: Dict[str, List[dict]], *, has_syn: bool) -> str:
    """Overall fit measures: pooled over everything ('ALL') and per data type."""
    scopes: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for dname, lst in panels.items():
        parts = [normalized_residuals(p) for p in lst]
        scopes[dname] = (np.concatenate([q[1] for q in parts]),
                         np.concatenate([q[2] for q in parts]))
    scopes = {"ALL": (np.concatenate([v[0] for v in scopes.values()]),
                      np.concatenate([v[1] for v in scopes.values()])), **scopes}
    hdr = _measure_header("scope", "")
    lines = ["=" * len(hdr), hdr, "-" * len(hdr)]
    for name, (a1, a2) in scopes.items():
        lines.extend(_measure_rows(name, "", a1.size, fit_measures(a1),
                                   fit_measures(a2), has_syn=has_syn))
    lines.append("=" * len(hdr))
    return "\n".join(lines)


def format_qq_stats(all_stats: Dict[str, List[dict]], *, has_syn: bool) -> str:
    hdr = (f"{'group':<26} {'band':<24} {'n':>7} {'slope1':>8}"
           + (f" {'slope2':>8}" if has_syn else "") + f" {'icpt':>8} {'out%':>6}")
    lines = ["=" * len(hdr), hdr, "-" * len(hdr)]
    for g, lst in all_stats.items():
        for s in lst:
            lines.append(f"{g:<26} {s['band']:<24} {s['n']:>7d} {s['slope1']:>8.3f}"
                         + (f" {s['slope2']:>8.3f}" if has_syn else "")
                         + f" {s['icpt']:>8.3f} {s['out_pct']:>6.1f}")
    lines.append("=" * len(hdr))
    return "\n".join(lines)


def format_runs(runs: Sequence[dict]) -> str:
    hdr = f"{'#':>4} {'iter':>5} {'cands':>6} {'rms':>9}  calc path (obs)"
    lines = ["=" * 100, hdr, "-" * 100]
    for i, r in enumerate(runs):
        lines.append(f"{i:>4d} {r['iter']:>5d} {r['n_candidates']:>6d} {r['rms']:>9.3f}  "
                     f"{r['path']}  (obs: {r['obs']})")
    lines.append("=" * 100)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Script driver
# ---------------------------------------------------------------------------
def _save(fig, stem: str, formats: Union[str, Sequence[str]], dpi: int) -> None:
    """
    Save `fig` as stem + ext for every extension in `formats`. A plain string
    (e.g. ".pdf") counts as one format; extensions may be given with or
    without the leading dot; the file format is passed explicitly, so an
    unsupported extension raises instead of silently producing a PNG.
    """
    if isinstance(formats, str):
        formats = (formats,)
    for ext in formats:
        ext = "." + str(ext).lstrip(".")
        out = stem + ext
        fig.savefig(out, dpi=dpi, format=ext[1:])
        print(f"modem_data_misfit: wrote {out}")


def _write(path: str, header: str, body: str) -> None:
    with open(path, "w", encoding="ascii") as fh:
        fh.write(header + "\n" + body + "\n")
    print(f"modem_data_misfit: wrote {path}")


def default_prefix(runs: Sequence[dict], mode: str) -> str:
    """File-name prefix: <calcfile>_iter<N> (single), ensemble_M<M> (ensemble)."""
    if mode == "single":
        base = os.path.splitext(os.path.basename(runs[0]["path"]))[0]
        it = runs[0]["iter"]
        return base if it < 0 else f"{base}_iter{it}"
    return f"ensemble_M{len(runs)}"


def default_out_dir(runs: Sequence[dict]) -> str:
    """
    Directory of the chosen calculated file(s): the run directory for one
    run; for an ensemble their common parent directory.
    """
    dirs = [os.path.dirname(os.path.abspath(r["path"])) for r in runs]
    if len(set(dirs)) == 1:
        return dirs[0]
    try:
        return os.path.commonpath(dirs)
    except ValueError:
        return os.getcwd()


def process_runs(runs: Sequence[dict], *, mode: str, out_dir: str, prefix: str) -> int:
    """
    Run the selected METHODS on one dataset built from `runs` (one run for
    "single", all members for "ensemble"). Returns 0 on success.
    """
    import matplotlib.pyplot as plt

    ds = build_dataset([r["data"] for r in runs], obs_mode=ENSEMBLE_OBS,
                       ref_member=REF_MEMBER)
    M = ds["M"]
    has_syn = M > 1
    print(f"modem_data_misfit: mode={mode}, members M={M}, rows={ds['freq'].size}, "
          f"dropped (non-finite calc)={ds['n_dropped_nan']}")
    if mode == "ensemble" and M == 1:
        print("modem_data_misfit: WARNING: ensemble with one member; RMS2 = RMS1.")
    runs_txt = format_runs(runs)
    if M <= 30:
        print(runs_txt)
    else:
        rr = np.array([r["rms"] for r in runs], float)
        print(f"modem_data_misfit: member RMS min/median/max = "
              f"{np.nanmin(rr):.3f} / {np.nanmedian(rr):.3f} / {np.nanmax(rr):.3f}")

    os.makedirs(out_dir, exist_ok=True)
    print(f"modem_data_misfit: output directory {out_dir}")
    if M > 1:
        title_extra = f"ensemble of {M} runs"
    else:
        run_name = os.path.basename(os.path.dirname(os.path.abspath(runs[0]["path"])))
        it = runs[0]["iter"]
        title_extra = f"{run_name}" + (f", iteration {it}" if it >= 0 else "")
    if WRITE_STATS:
        _write(os.path.join(out_dir, f"{prefix}_runs.txt"), f"# mode={mode}", runs_txt)

    panels = build_panels(ds, datatypes=DATATYPES, components=COMPONENTS)
    if not panels:
        print("modem_data_misfit: nothing left after DATATYPES/COMPONENTS selection.")
        return 1

    if FIT_MEASURES:
        table = format_fit_measures(panels, has_syn=has_syn)
        print(table)
        if WRITE_STATS:
            _write(os.path.join(out_dir, f"{prefix}_fit_measures.txt"),
                   f"# {title_extra}\n{FIT_MEASURE_NOTE}", table)

    if "crossplot" in METHODS:
        crange = global_logf_range(panels) if COLOR_RANGE == "global" else None
        for dname, lst in panels.items():
            fig = plot_crossplot_datatype(
                dname, lst, title_extra=title_extra, has_syn=has_syn, ncols=NCOLS,
                panel_size=PANEL_SIZE, cmap=CMAP, color_range=crange,
                marker_size=MARKER_SIZE, alpha=ALPHA, log_rho=LOG_RHO,
                show_errorbars=SHOW_ERRORBARS, equal_aspect=EQUAL_ASPECT)
            _save(fig, os.path.join(out_dir, f"{prefix}_crossplot_{_safe_name(dname)}"),
                  PLOT_FORMATS, PLOT_DPI)
            if SHOW:
                plt.show()
            plt.close(fig)
        table = format_crossplot_stats(panels, has_syn=has_syn)
        print(table)
        if WRITE_STATS:
            _write(os.path.join(out_dir, f"{prefix}_crossplot_stats.txt"),
                   f"# {title_extra}\n{FIT_MEASURE_NOTE}\n"
                   "# columns with suffix 1 / 2: conventional / obs+syn normalization "
                   "(ensemble only); rmae = r_mae, rmed = r_med", table)

    if "histogram" in METHODS:
        if not has_syn:
            print("modem_data_misfit: single run: no synthesized error available, "
                  "histograms show the conventional (RMS1) normalization only.")
        all_stats: Dict[str, List[dict]] = {}
        for gname, lst in group_panels(panels, by_component=HIST_BY_COMPONENT).items():
            parts = [normalized_residuals(p) for p in lst]
            freq = np.concatenate([q[0] for q in parts])
            r1 = np.concatenate([q[1] for q in parts])
            r2 = np.concatenate([q[2] for q in parts])
            if r1.size == 0:
                print(f"modem_data_misfit: {gname}: no rows with positive error, skipped.")
                continue
            fig, stats = plot_residual_histograms(
                gname, freq, r1, r2, has_syn=has_syn,
                per_decade=HIST_BANDS_PER_DECADE, axis=HIST_BAND_AXIS,
                ncols=HIST_NCOLS, panel_size=HIST_PANEL_SIZE, limit=HIST_LIMIT,
                bin_width=HIST_BIN_WIDTH, min_n=HIST_MIN_N,
                run_label=title_extra)
            all_stats[gname] = stats
            _save(fig, os.path.join(out_dir, f"{prefix}_hist_{_safe_name(gname)}"),
                  PLOT_FORMATS, PLOT_DPI)
            if SHOW:
                plt.show()
            plt.close(fig)
        table = format_hist_stats(all_stats, has_syn=has_syn)
        print(table)
        if WRITE_STATS:
            _write(os.path.join(out_dir, f"{prefix}_hist_stats.txt"),
                   f"# {title_extra}; bands: {HIST_BANDS_PER_DECADE} per decade "
                   f"({HIST_BAND_AXIS})\n{FIT_MEASURE_NOTE}\n"
                   "# n_out: values of the plotted variant outside +/- HIST_LIMIT", table)

    if "qq" in METHODS:
        qq_stats: Dict[str, List[dict]] = {}
        for gname, lst in group_panels(panels, by_component=QQ_BY_COMPONENT).items():
            parts = [normalized_residuals(p) for p in lst]
            freq = np.concatenate([q[0] for q in parts])
            r1 = np.concatenate([q[1] for q in parts])
            r2 = np.concatenate([q[2] for q in parts])
            if r1.size < 4:
                print(f"modem_data_misfit: {gname}: too few rows for Q-Q plots, skipped.")
                continue
            fig, stats = plot_qq_bands(
                gname, freq, r1, r2, has_syn=has_syn,
                per_decade=QQ_BANDS_PER_DECADE, axis=QQ_BAND_AXIS,
                ncols=QQ_NCOLS, panel_size=QQ_PANEL_SIZE, limit=QQ_LIMIT,
                confidence=QQ_CONFIDENCE, min_n=QQ_MIN_N,
                equal_axes=QQ_EQUAL_AXES, run_label=title_extra)
            qq_stats[gname] = stats
            _save(fig, os.path.join(out_dir, f"{prefix}_qq_{_safe_name(gname)}"),
                  PLOT_FORMATS, PLOT_DPI)
            if SHOW:
                plt.show()
            plt.close(fig)
        table = format_qq_stats(qq_stats, has_syn=has_syn)
        print(table)
        if WRITE_STATS:
            _write(os.path.join(out_dir, f"{prefix}_qq_stats.txt"),
                   f"# {title_extra}; bands: {QQ_BANDS_PER_DECADE} per decade "
                   f"({QQ_BAND_AXIS})", table)
    return 0


def _pairs_from_argv(argv: Sequence[str]) -> List[Tuple[str, str]]:
    argv = list(argv)
    if len(argv) % 2 != 0:
        raise ValueError(
            "modem_data_misfit: command-line run paths must come in "
            "(obs, calc) pairs -- got an odd number of arguments.")
    return [(argv[i], argv[i + 1]) for i in range(0, len(argv), 2)]


def main(argv: Sequence[str]) -> int:
    import matplotlib
    if not SHOW:
        matplotlib.use("Agg")

    argv = list(argv)
    mode = RUN_MODE
    if argv and argv[0] in ("single", "ensemble"):
        mode = argv.pop(0)
    try:
        raw_pairs = _pairs_from_argv(argv) if argv else list(RUN_PATHS)
        pairs = expand_run_pairs(raw_pairs)
    except (FileNotFoundError, ValueError) as exc:
        print(f"modem_data_misfit: {exc}")
        return 1
    if not pairs:
        print("modem_data_misfit: no run paths given.")
        return 1

    if mode == "ensemble":
        try:
            runs = [resolve_run(o, c, iteration=ITERATION, calc_pattern=CALC_PATTERN,
                                key_round=KEY_ROUND) for o, c in pairs]
        except (FileNotFoundError, ValueError) as exc:
            print(f"modem_data_misfit: {exc}")
            return 1
        prefix = OUT_PREFIX or default_prefix(runs, mode)
        out_dir = OUT_DIR or default_out_dir(runs)
        return process_runs(runs, mode=mode, out_dir=out_dir, prefix=prefix)

    # single: every entry on its own (not an ensemble)
    rc = 0
    for k, (o, c) in enumerate(pairs):
        print(f"modem_data_misfit: ===== run {k + 1} of {len(pairs)}: obs={o} calc={c}")
        try:
            run = resolve_run(o, c, iteration=ITERATION, calc_pattern=CALC_PATTERN,
                              key_round=KEY_ROUND)
        except (FileNotFoundError, ValueError) as exc:
            print(f"modem_data_misfit: skipping obs={o} calc={c}: {exc}")
            rc = 1
            continue
        derived = default_prefix([run], "single")
        if not OUT_PREFIX:
            prefix = derived
        elif len(pairs) == 1:
            prefix = OUT_PREFIX
        else:
            prefix = f"{OUT_PREFIX}_{derived}"
        out_dir = OUT_DIR or default_out_dir([run])
        try:
            rc = max(rc, process_runs([run], mode="single", out_dir=out_dir, prefix=prefix))
        except (ValueError, KeyError) as exc:
            print(f"modem_data_misfit: obs={o} calc={c} failed: {exc}")
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
