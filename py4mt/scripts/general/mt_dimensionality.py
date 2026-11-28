#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mt_dimensionality.py

Dimensionality of the impedance tensors of a survey, read from EDI files or
from py4mt .npz collections, with the module dimens:

    WAL class       nine classes of WALDIM 1.2 (rotational invariants), with
                    strike, twist and shear, per frequency and band
    PT class        1D / 2D / 3D from skew angle beta and ellipticity of the
                    phase tensor, with its parameters
    skews           Swift skew and Bahr phase-sensitive skew
    reliability     reason of "undetermined", agreement of the WAL class over
                    a range of thresholds, optional bootstrap

The configuration block below sets input, output and parameters.

Output (OUT_DIR, default WORK_DIR/mt_dimensionality/):
    <SUMMARY_NAME>.txt   settings, class counts per band and per site, one
                         line per site and band
    <SUMMARY_NAME>.csv   one row per site and band, all summary values
    <SUMMARY_NAME>.npz   the same as arrays (sites x bands), see
                         dimens.survey_summary
    sites/<site>_dim.npz every parameter of one site, per frequency and band,
                         see dimens.site_parameters
    <SUMMARY_NAME>_DIM_<th>.dat, _BANDCLASS_<th>.dat
                         WALDIM-style tables for mt_kmz_waldim.py
Existing files are overwritten; files of other runs are not removed.

Input .npz files hold pickled objects: read only files that you trust.

@author vrath
@project py4mt
@created 2026-11-28
@modified 2026-11-28

Provenance
----------
Author: Claude (Anthropic), model Claude Sonnet 5.5.
Date generated: 2026-11-28.
This code is AI-generated and should be reviewed before production use.

Changelog
---------
2026-11-28  Claude Sonnet 5.5 (Anthropic): initial version. Replaces the
            driver mt_dimens.py: EDI files and .npz collections, phase tensor
            classes, skews, survey summary in .txt, .csv and .npz.
"""

from __future__ import annotations

import glob
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

for _base in [PY4MTX_ROOT + "/py4mt/modules/"]:
    for _p in [Path(_base), *Path(_base).rglob("*")]:
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

import util as utl
from version import versionstrg
import dimens

# =============================================================================
#  Configuration
# =============================================================================
WORK_DIR = PY4MTX_DATA + "/dimensionality/"

# Input: a directory with EDI files, an EDI file, a .npz collection (from
# mt_data_processor) or station file, a pattern with * or ?, or a list of
# these. Station names must be unique over all inputs.
INPUT = WORK_DIR + "edi/"
# INPUT = WORK_DIR + "proc/MySurvey_proc_collection.npz"
EDI_OPTIONS = {"manufacturer": "metronix"}   # further load_edi keywords
USE_FILENAME_AS_STATION = True       # EDI files; False: DATAID of the header

# --- output ------------------------------------------------------------------
OUT_DIR = WORK_DIR + "mt_dimensionality/"
SUMMARY_NAME = "mt_dimensionality"   # <OUT_DIR>/<SUMMARY_NAME>.txt/.csv/.npz
SITE_NPZ_DIR = OUT_DIR + "sites/"    # one .npz per site; None: no such files
SITE_NPZ_SUFFIX = "_dim"             # file name: <site><suffix>.npz
WALDIM_TABLES = True                 # tables for mt_kmz_waldim.py

# --- errors ------------------------------------------------------------------
ERROR_PERCENT = None                 # number: use instead of the data errors
MISSING_ERROR_PERCENT = 5.0          # for sites without errors; None: stop

# --- WAL classification ------------------------------------------------------
THRESHOLD = 0.15                     # invariants I3 to I7
Q_THRESHOLD = 0.10                   # invariant Q
N_REALIZATIONS = 100                 # noise realizations for the errors
SEED = 0
WALDIM_COMPAT = False                # True: rules of WALDIM 1.2

# --- phase tensor classification ---------------------------------------------
# Time dependence of the data: "auto" takes it from the sign of the phase
# tensor trace per site; "e+iwt" (phase of Zxy about +45 deg) or "e-iwt"
# (about -45 deg) force it. The WAL classes do not depend on it.
FT_CONVENTION = "auto"
PT_BETA_THRESHOLD = 3.0              # deg, |beta| above it: 3D
PT_ELLIPTICITY_THRESHOLD = 0.1       # ellipticity above it: 2D, else 1D

# --- reliability of the WAL class --------------------------------------------
STABILITY_THRESHOLDS = (0.10, 0.12, 0.18, 0.20)   # None: no threshold sweep
N_BOOTSTRAP = 0                      # resampled data sets, e.g. 200; 0: none

# --- bands -------------------------------------------------------------------
BANDS = True                         # False: one summary over all data
BANDS_PER_DECADE = 1.0
PERIOD_RANGE = None                  # (Tmin, Tmax) in s; None: all data


# =============================================================================
#  Functions
# =============================================================================
def expand_input(item: Any) -> Any:
    """Replace patterns with * or ? by the matching files."""
    if isinstance(item, (list, tuple)):
        out: List[Any] = []
        for part in item:
            part = expand_input(part)
            out += part if isinstance(part, list) else [part]
        return out
    text = str(item)
    if ("*" in text or "?" in text) and not os.path.exists(text):
        files = sorted(glob.glob(text))
        if not files:
            sys.exit(f"no file matches {text}")
        return files
    return item


def band_edges() -> Optional[np.ndarray]:
    """Band edges for PERIOD_RANGE, or None."""
    if not BANDS or PERIOD_RANGE is None:
        return None
    tmin, tmax = PERIOD_RANGE
    # as WALDIM: bands start at Tmin; an incomplete last band is left out
    count = int(np.log10(tmax / tmin) * BANDS_PER_DECADE + 1.0e-9)
    if count < 1:
        sys.exit("PERIOD_RANGE is narrower than one band")
    return tmin * 10.0 ** (np.arange(count + 1) / BANDS_PER_DECADE)


def settings_lines(inputs: Any) -> List[str]:
    """The settings, for the head of the summary file."""
    items = inputs if isinstance(inputs, list) else [inputs]
    lines = ["input: " + str(item) for item in items]
    errors = (f"{ERROR_PERCENT} percent of |Z| for all sites"
              if ERROR_PERCENT is not None else
              "from the data; sites without errors: "
              + ("stop" if MISSING_ERROR_PERCENT is None else
                 f"{MISSING_ERROR_PERCENT} percent of |Z|"))
    lines += [
        "errors: " + errors,
        f"WAL thresholds: {THRESHOLD} (I3-I7), {Q_THRESHOLD} (Q); "
        f"{N_REALIZATIONS} realizations, seed {SEED}, "
        f"waldim_compat {WALDIM_COMPAT}",
        f"phase tensor thresholds: |beta| {PT_BETA_THRESHOLD} deg, "
        f"ellipticity {PT_ELLIPTICITY_THRESHOLD}; time dependence "
        f"{FT_CONVENTION}",
        f"threshold sweep: {STABILITY_THRESHOLDS or 'none'}; "
        f"bootstrap: {N_BOOTSTRAP or 'none'}",
    ]
    if BANDS:
        limits = "all data" if PERIOD_RANGE is None else \
            f"{PERIOD_RANGE[0]} - {PERIOD_RANGE[1]} s"
        lines.append(f"bands: {BANDS_PER_DECADE} per decade, {limits}")
    else:
        lines.append("bands: none")
    return lines


def count_text(counts: Dict[str, int]) -> str:
    """The classes that occur, with their numbers."""
    return ", ".join(f"{k} {v}" for k, v in counts.items() if v) or "-"


def median_text(values: np.ndarray, digits: int = 2) -> str:
    """Median of the defined values."""
    values = values[np.isfinite(values)]
    return f"{np.median(values):.{digits}f}" if values.size else "-"


def print_site(name: str, site: Dict[str, Any]) -> None:
    """Screen report of one site."""
    print(f"\n {name}: {site['freq'].size} frequencies "
          f"({site['n_dropped']} removed, errors: {site['error_source']})")
    if not site["freq"].size:
        return
    print("   WAL classes : " + count_text(site["counts"]))
    reasons = {k: v for k, v in site["reason_counts"].items()
               if k != "classified"}
    if any(reasons.values()):
        print("   undetermined: " + count_text(reasons))
    print("   PT classes  : " + count_text(site["pt_counts"])
          + f"  (data read as {site['pt_ft_convention']}, "
          f"{100.0 * site['pt_ft_agreement']:.0f} % of frequencies agree)")
    line = (f"   medians     : |beta| "
            f"{median_text(np.abs(site['pt_beta_deg']), 1)} deg, ellipticity "
            f"{median_text(site['pt_ellipticity'])}, Swift "
            f"{median_text(site['swift_skew'])}, Bahr "
            f"{median_text(site['bahr_skew'])}")
    print(line)
    for key, label in (("threshold_agreement", "threshold agreement"),
                       ("bootstrap_agreement", "bootstrap agreement")):
        if key in site:
            print(f"   mean {label}: {np.mean(site[key]):.2f}")
    for band in site.get("bands", []):
        if band["n_periods"] == 0:
            continue
        pt = dimens.PT_CLASS_NAMES[band["pt_class"]]
        line = (f"   {band['period_min']:9.3g} - "
                f"{band['period_max']:9.3g} s  {band['class_name']:<12s} "
                f"PT {pt:<9s}")
        if np.isfinite(band["strike_deg"]):
            line += (f" strike {band['strike_deg']:5.1f} "
                     f"({band['strike_err_deg']:.1f})")
        if np.isfinite(band["twist_deg"]):
            line += (f" twist {band['twist_deg']:5.1f} "
                     f"({band['twist_err_deg']:.1f})"
                     f" shear {band['shear_deg']:5.1f} "
                     f"({band['shear_err_deg']:.1f})")
        print(line)


def main() -> int:
    """Read, analyse, write."""
    version, _ = versionstrg()
    titstrng = utl.print_title(
        version=version, fname=inspect.getfile(inspect.currentframe()),
        out=False)
    print(titstrng + "\n\n")

    inputs = expand_input(INPUT)
    results = dimens.classify_dimensionality(
        inputs,
        threshold=THRESHOLD,
        q_threshold=Q_THRESHOLD,
        error_percent=ERROR_PERCENT,
        missing_error_percent=MISSING_ERROR_PERCENT,
        bands=BANDS,
        bands_per_decade=BANDS_PER_DECADE,
        band_edges=band_edges(),
        n_realizations=N_REALIZATIONS,
        seed=SEED,
        waldim_compat=WALDIM_COMPAT,
        stability_thresholds=STABILITY_THRESHOLDS,
        n_bootstrap=N_BOOTSTRAP,
        pt_beta_threshold=PT_BETA_THRESHOLD,
        pt_ellipticity_threshold=PT_ELLIPTICITY_THRESHOLD,
        ft_convention=FT_CONVENTION,
        use_filename_as_station=USE_FILENAME_AS_STATION,
        edi_options=EDI_OPTIONS,
    )
    if not results:
        sys.exit("no sites in the input")

    for name, site in results.items():
        print_site(name, site)

    # the site files first: they check that the names are unique, so that
    # nothing is written otherwise
    out_dir = Path(OUT_DIR)
    sites: List[Path] = []
    if SITE_NPZ_DIR is not None:
        sites = dimens.save_site_npz(
            results, SITE_NPZ_DIR, suffix=SITE_NPZ_SUFFIX)
    written = dimens.write_summary(
        results, out_dir, name=SUMMARY_NAME, header=settings_lines(inputs))
    if WALDIM_TABLES:
        written += dimens.write_results(results, out_dir / SUMMARY_NAME)

    print(f"\n {len(results)} sites")
    for path in written:
        print(" written: " + str(path))
    if sites:
        print(f" written: {len(sites)} site files in {SITE_NPZ_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
