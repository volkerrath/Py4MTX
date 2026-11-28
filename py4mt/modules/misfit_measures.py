#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
misfit_measures.py

Measures of data fit of error-normalized residuals, and the choice of the
measures that the misfit figures show. Used by femtic_misfit.py (the engine
of femtic_mod_misfit.py / femtic_ens_misfit.py) and by
scripts/modem/modem_data_misfit.py. All functions are stateless: everything
is passed as an argument.

Measures (r = error-normalized residuals; expected value for r ~ N(0,1))
-------------------------------------------------------------------------
    nrms   sqrt(mean(r^2))                          1
    r_mae  sqrt(pi/2) * mean(|r|)                   1
    r_med  median(|r|) / 0.67449                    1
    q95    95th percentile of |r|                   1.96
    mean   mean(r) (bias)                           0
    std    sqrt(mean((r - mean)^2))                 1
    skew   m3 / m2^1.5                              0
    kurt   m4 / m2^2 - 3 (excess kurtosis)          0
nrms, r_mae, r_med and q95 are computed by inverse.py.

Contents
--------
    sample_moments, fit_measures      the measures of one sample
    qq_quartile_line                  slope / intercept of the Q-Q quartile line
    site_measures                     parse the setting SITE_MEASURES
    box_items, box_text               the setting PANEL_MEASURES and the text
                                      of a panel box
    hist_scale                        parse the setting HIST_SCALE_MEASURE
    site_value                        one value of a per-site table
    MEASURE_LABELS, FIT_MEASURE_NAMES, SIGNED_MEASURES, MEASURE_ALIASES,
    BOX_KINDS, BOX_ITEMS, BOX_DEFAULT, HIST_SCALES

@author   vrath
@project  py4mt
@created  2026-11-27
@modified 2026-11-27

Provenance
----------
Author      : Claude (Anthropic)
Generated   : 2026-11-27 (sample_moments, fit_measures and qq_quartile_line
              moved here from femtic_misfit.py, where they dated from
              2026-11-09 and later)
Notice      : This code is AI-generated. Review and test it before any
              production use.

Changelog
---------
2026-11-27  Claude Sonnet 5.5 (Anthropic), AI-generated, review before use:
            new module. sample_moments(), fit_measures() and
            qq_quartile_line() moved here from femtic_misfit.py without a
            change of their results; new: the selection of the plotted
            measures (site_measures, box_items, box_text, hist_scale,
            site_value and the constants). Tested on synthetic data only.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import inverse as inv  # nrms, r_mae, r_med, q95


# ---------------------------------------------------------------------------
# The measures of one sample
# ---------------------------------------------------------------------------
def sample_moments(r: np.ndarray) -> dict:
    """
    Central moments of a sample (non-finite values ignored): std, skew, kurt.

    std   sqrt(m2), m_k = mean((r - mean(r))**k)         (spread about the mean)
    skew  m3 / m2**1.5                                   (Fisher-Pearson g1)
    kurt  m4 / m2**2 - 3                                 (EXCESS kurtosis g2)

    These are the plain (biased, 1/n) moment estimators, the defaults of
    scipy.stats.skew / scipy.stats.kurtosis, centred on the sample mean, so a
    bias does not enter. Expected for N(0, s^2) samples: skew = kurt = 0; the
    standard errors are about sqrt(6/n) and sqrt(24/n) for independent values.
    Positive kurt = heavy tails (outliers), negative = light tails; skew > 0
    = long tail towards positive residuals (observed > calculated). NaN where
    undefined: skew needs n >= 3, kurt n >= 4, and a non-constant sample.
    """
    nan = float("nan")
    r = np.asarray(r, dtype=float).ravel()
    r = r[np.isfinite(r)]
    n = r.size
    out = dict(std=nan, skew=nan, kurt=nan)
    if n == 0:
        return out
    d = r - r.mean()
    m2 = float(np.mean(d ** 2))
    out["std"] = math.sqrt(m2)
    if n < 3 or out["std"] <= 1.0e-12 * float(np.max(np.abs(r))):
        return out                                # constant (or all-zero) sample
    out["skew"] = float(np.mean(d ** 3)) / m2 ** 1.5
    if n >= 4:
        out["kurt"] = float(np.mean(d ** 4)) / m2 ** 2 - 3.0
    return out


def fit_measures(r: np.ndarray) -> dict:
    """
    Fit measures of error-normalized residuals (non-finite values ignored):
    n, nrms, r_mae, r_med, q95 (all from inverse.py), the mean (bias) and the
    higher moments std, skew, kurt (excess) of sample_moments().
    Expected for N(0,1) residuals: nrms = r_mae = r_med = 1, q95 = 1.96,
    mean = skew = kurt = 0, std = 1.
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
        **sample_moments(r),
    )


# ---------------------------------------------------------------------------
# Selection of the plotted measures
# ---------------------------------------------------------------------------
#: selectable fit measures of error-normalized residuals -> label in figures
MEASURE_LABELS: Dict[str, str] = {
    "nrms": "nRMS", "r_mae": "rMAE", "r_med": "rMED", "q95": "q95",
    "mean": "mean", "std": "std", "skew": "skew", "kurt": "kurt"}
FIT_MEASURE_NAMES: Tuple[str, ...] = tuple(MEASURE_LABELS)
#: measures that can be negative (bias and shape): worst / best by |value|,
#: diverging colour scale about 0, no log10
SIGNED_MEASURES: Tuple[str, ...] = ("mean", "skew", "kurt")
MEASURE_ALIASES: Dict[str, str] = {
    "rms": "nrms", "rmae": "r_mae", "rmed": "r_med", "bias": "mean",
    "intercept": "icpt"}

#: figures with a text box per panel, the entries of the box and the defaults
BOX_KINDS: Tuple[str, ...] = ("crossplot", "histogram", "qq")
BOX_ITEMS: Tuple[str, ...] = (("n",) + FIT_MEASURE_NAMES
                              + ("slope", "icpt", "n_out", "out_pct"))
BOX_DEFAULT: Dict[str, Tuple[str, ...]] = {
    "crossplot": ("n", "nrms"),
    "histogram": ("n", "nrms", "n_out"),
    "qq": ("n", "slope", "icpt")}
_BOX_ONLY: Dict[str, Tuple[str, ...]] = {          # entries of some figures only
    "n_out": ("histogram", "qq"), "out_pct": ("qq",)}

#: HIST_SCALE_MEASURE: name -> (divisor that turns it into a Gaussian sigma,
#: legend label of the curve)
HIST_SCALES: Dict[str, Tuple[float, str]] = {
    "nrms": (1.0, "N(0, RMS^2)"),
    "r_mae": (1.0, "N(0, rMAE^2)"),
    "r_med": (1.0, "N(0, rMED^2)"),
    "q95": (1.959963984540054, "N(0, (q95/1.96)^2)")}


def _split_names(spec, setting: str) -> List[str]:
    """A name, "a, b" or a sequence of names -> list of canonical names."""
    if isinstance(spec, str):
        spec = [t for t in re.split(r"[,;\s]+", spec.strip()) if t]
    try:
        names = [str(t).strip().lower() for t in spec]
    except TypeError:
        raise ValueError(f"{setting}: expected a name or a list of names, got {spec!r}.")
    out: List[str] = []
    for t in names:
        t = MEASURE_ALIASES.get(t, t)
        if t not in out:
            out.append(t)
    return out


def site_measures(spec, *, setting: str = "SITE_MEASURES") -> Tuple[str, ...]:
    """
    SITE_MEASURES as a tuple of names of FIT_MEASURE_NAMES (at least one, in
    the order given, without repeats). None = ("nrms",). ValueError for an
    unknown name or an empty list.
    """
    if spec is None:
        return ("nrms",)
    names = [] if isinstance(spec, bool) else _split_names(spec, setting)
    bad = [t for t in names if t not in FIT_MEASURE_NAMES]
    if bad or not names:
        raise ValueError(f"{setting} must name one or more of {FIT_MEASURE_NAMES}; "
                         f"got {spec!r}.")
    return tuple(names)


def box_items(spec, kind: str, *, setting: str = "PANEL_MEASURES") -> Tuple[str, ...]:
    """
    The entries of the panel text box of one kind of figure ("crossplot",
    "histogram", "qq") from the setting PANEL_MEASURES.

    spec  None -> BOX_DEFAULT[kind]; False, "" or () -> no entry; a name,
          "a, b" or a list -> these entries in this order (entries that do not
          apply to `kind`, e.g. "out_pct" of the Q-Q plots, are left out); a
          dict {kind: list, "default": list} (a kind that is not in the dict:
          "default", else BOX_DEFAULT). An entry named in the list of its own
          kind that does not apply there is an error.

    ValueError for an unknown entry, kind or dict key.
    """
    if kind not in BOX_KINDS:
        raise ValueError(f"box_items: kind {kind!r} not in {BOX_KINDS}.")
    own = False                                  # list given for this kind only
    if isinstance(spec, dict):
        bad = [k for k in spec if k not in BOX_KINDS + ("default",)]
        if bad:
            raise ValueError(f"{setting}: unknown key(s) {bad}; use "
                             f"{BOX_KINDS + ('default',)}.")
        own = kind in spec
        spec = spec[kind] if own else spec.get("default")
    if spec is None or spec is True:
        return BOX_DEFAULT[kind]
    if spec is False:
        return ()
    names = _split_names(spec, setting)
    bad = [t for t in names if t not in BOX_ITEMS]
    if bad:
        raise ValueError(f"{setting}: unknown entry(ies) {bad}; use {BOX_ITEMS}.")
    wrong = [t for t in names if kind not in _BOX_ONLY.get(t, BOX_KINDS)]
    if wrong and own:
        raise ValueError(f"{setting}[{kind!r}]: {wrong} not available for this figure.")
    return tuple(t for t in names if t not in wrong)


def hist_scale(spec, *, setting: str = "HIST_SCALE_MEASURE") -> Optional[str]:
    """HIST_SCALE_MEASURE as a key of HIST_SCALES, or None (nothing drawn)."""
    if spec is None or spec is False or str(spec).strip().lower() in ("", "none"):
        return None
    name = str(spec).strip().lower()
    name = MEASURE_ALIASES.get(name, name)
    if name not in HIST_SCALES:
        raise ValueError(f"{setting} must be None or one of {tuple(HIST_SCALES)}; "
                         f"got {spec!r}.")
    return name


def box_text(items: Sequence[str], kind: str, *, n: int, r1: np.ndarray,
             r2: np.ndarray, has_syn: bool, limit: Optional[float] = None,
             out_pct: float = float("nan")) -> str:
    """
    Text of a panel box: one line per value of the entries `items` (see
    box_items), in their order. n: the number written for "n"; r1 / r2: the
    RMS1- / RMS2-normalized residuals of the panel. With `has_syn` the fit
    measures and "slope" are written for both (suffix 1 / 2); "icpt" and
    "n_out" (values with |r| > `limit`; no line if 0) refer to the plotted
    variant (r2 if has_syn, else r1). "out_pct" is written if finite.
    """
    r1 = np.asarray(r1, float).ravel()
    r2 = np.asarray(r2, float).ravel()
    r1, r2 = r1[np.isfinite(r1)], r2[np.isfinite(r2)]
    variants = (("1", r1), ("2", r2)) if has_syn else (("", r1),)
    main = r2 if has_syn else r1
    meas: Optional[List[dict]] = None
    lines: List[str] = []

    def line(a: np.ndarray) -> Tuple[float, float]:
        return (qq_quartile_line(np.sort(a)) if a.size
                else (float("nan"), float("nan")))

    for item in items:
        if item == "n":
            lines.append(f"n={n}")
        elif item in MEASURE_LABELS:
            if meas is None:
                meas = [fit_measures(a) for _, a in variants]
            label = MEASURE_LABELS[item]
            if item == "nrms" and kind == "histogram":
                label = "RMS"
            for (sfx, _), m in zip(variants, meas):
                if sfx and label[-1].isdigit():
                    sfx = f"({sfx})"
                lines.append(f"{label}{sfx}={m[item]:.2f}")
        elif item == "slope":
            lines.extend(f"slope{sfx}={line(a)[0]:.2f}" for sfx, a in variants)
        elif item == "icpt":
            lines.append(f"icpt={line(main)[1]:.2f}")
        elif item == "n_out":
            k = 0 if limit is None else int(np.count_nonzero(np.abs(main) > limit))
            if k:
                lines.append(f"out: {k}")
        elif item == "out_pct" and np.isfinite(out_pct):
            lines.append(f"outside env.: {out_pct:.1f}%")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Further quantities of the figures and tables
# ---------------------------------------------------------------------------
def qq_quartile_line(sample_sorted: np.ndarray) -> Tuple[float, float]:
    """(slope, intercept) of the line through the sample's and the standard
    normal's first and third quartiles (robust scale and location)."""
    from scipy.stats import norm
    q1, q3 = np.quantile(sample_sorted, [0.25, 0.75])
    z1, z3 = norm.ppf([0.25, 0.75])
    slope = float((q3 - q1) / (z3 - z1))
    return slope, float(q1 - slope * z1)


def site_value(d: dict, measure: str = "nrms") -> float:
    """The per-site value `measure` of one entry of the site table (NaN if absent)."""
    return float(d.get(measure, float("nan")))
