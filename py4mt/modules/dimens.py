"""
dimens.py

Dimensionality analysis of magnetotelluric (MT) impedance tensors: the WAL
rotational invariants (Python port of the Fortran code WALDIM, version 1.2,
May 2015, by A. Marti, P. Queralt and J. Ledo: WALDIM1.2.f, external1.2.f,
inoutdata1.2.f), the phase tensor and the Swift and Bahr skews. EDI files are
read with data_proc.load_edi of py4mt.

Main interface
--------------
classify_dimensionality(sites, ...)
    Returns, for every site, per frequency: the WAL dimensionality class with
    its reason code, apparent resistivity, phase, strike, twist and shear, the
    phase tensor with its parameters and class, the skews, errors, the
    stability of the class under other thresholds and (optional) under noise;
    and, optionally, the same summarised in period bands.
summarise_band(result, members)
    The summary of one site over any set of its frequencies.
write_results(results, root)
    Writes the WALDIM tables root_DIM_<th>.dat and root_BANDCLASS_<th>.dat.
site_parameters(result)
    Collects the parameters of one site in a flat dict.
save_site_npz(results, directory)
    Saves that dict to one compressed .npz file per site.
survey_summary(results)
    Arrays (site, band) of the band summaries of all sites.
write_summary(results, directory)
    Saves the survey summary as .txt, .csv and .npz.

The script mt_dimensionality.py runs all of this on EDI files or collections;
femtic_plot_sitemap.py maps the results.

Classes (codes as in WALDIM; 0 means that no class could be assigned)
---------------------------------------------------------------------
0  undetermined
1  1D
2  2D
3  3D/2Dtw      3D/2D, twist only
4  3D/2D        3D/2D, general galvanic distortion
5  3D
6  3D/2Ddiag    diagonally distorted case
7  3D/1D2D      galvanic distortion, strike not recoverable
8  ANIS-HINT1   anisotropy hint (WALDIM: "over homogeneous medium")
9  ANIS-HINT2   anisotropy hint (WALDIM: "over 2D medium")

Phase tensor classes: 0 undefined, 1 1D, 2 2D, 3 3D.  3D: |beta| above
pt_beta_threshold (3 deg); else 2D: ellipticity above
pt_ellipticity_threshold (0.1); else 1D. The threshold applies to |beta|, not
to 2 |beta|.

Reason codes of the WAL class: 0 classified; 1 bias test (true and Monte
Carlo invariants differ); 2 error bars (an invariant cannot be decided within
its error); 3 no rule (the combination of invariants has no class).

Definitions
-----------
Phase tensor (Caldwell et al., 2004): Phi = X^-1 Y with Z = X + iY;
beta = atan2(P12 - P21, P11 + P22) / 2, alpha = atan2(P12 + P21, P11 - P22)
/ 2, azimuth of the major axis = alpha - beta, ellipticity = (max - min) /
(max + min).
Skews, with S1 = Zxx + Zyy, S2 = Zxy + Zyx, D1 = Zxx - Zyy, D2 = Zxy - Zyx
and [a, b] = Re a Im b - Im a Re b:
Swift = |S1| / |D2|; Bahr = sqrt(|[D1, S2] - [S1, D2]|) / |D2|.

Units and conventions
---------------------
Z in mV/km/nT (py4mt), rho_a = |Z|^2 * mu0 * 1e6 / omega = 0.2 |Z|^2 / f.
Angles in degrees. Strikes and phase tensor angles include the data rotation
angle "rot"; strikes and the azimuth carry the usual 90 degree ambiguity; the
sign of the shear angle belongs to the strike value that is reported with it.
All errors are 1-sigma standard errors.

Time convention: the WAL classes, strikes, twist, shear, phases and skews do
not depend on it. The phase tensor needs impedance phases of Zxy in the first
quadrant, exp(+iwt); data stored as exp(-iwt) (the standard of py4mt) are
conjugated for it. With ft_convention="auto" the convention of each site is
taken from the sign of the trace of the phase tensor at most frequencies
(results "pt_ft_convention", "pt_ft_agreement").

Differences from WALDIM 1.2
---------------------------
By default (waldim_compat=False) strikes are treated modulo 90 degrees,
percentage errors use max(|Re|, |Im|), strike, twist and shear belong to the
same strike branch, and the error of a band average is the larger of the
propagated data error and the standard error from the scatter in the band.
waldim_compat=True restores the original rules.
Corrected in both modes: the leftover index in the apparent resistivity error
(WALDIM1.2.f line 751); the Monte Carlo twist and shear (rotation angle
converted with 180/pi, line 858; missing factors of 2 and reversed signs,
lines 859-878); the fixed site index in the band strike average (line 1400);
the analytic error of the third strike (incomplete derivative, lines
768-785); the unset strike error of class 9 in the per-frequency table.
Additions: phase tensor, skews, reason codes, threshold stability, bootstrap.

Not ported
----------
The Fortran EDI reader (replaced by load_edi), the param.cfg file (replaced
by keyword arguments and the script mt_dimensionality.py), the output tables
other than _DIM_ and _BANDCLASS_, and the comparison tables with Swift and
Bahr parameters (the skews themselves are computed).

References
----------
Marti, Queralt, Ledo (2009), Computers & Geosciences 35(12), 2295-2303,
    doi:10.1016/j.cageo.2009.03.004
Marti, Queralt, Ledo, Farquharson (2010), Phys. Earth Planet. Inter.
    182(3-4), 139-151, doi:10.1016/j.pepi.2010.07.007
Marti, Queralt, Jones, Ledo (2005), Geophys. J. Int. 163(1), 38-41,
    doi:10.1111/j.1365-246X.2005.02748.x
Weaver, Agarwal, Lilley (2000), Geophys. J. Int. 141(2), 321-336
    (DOI not confirmed)
Caldwell, Bibby, Brown (2004), Geophys. J. Int. 158(2), 457-469,
    doi:10.1111/j.1365-246X.2004.02281.x
Booker (2014), Surveys in Geophysics 35(1), 7-40,
    doi:10.1007/s10712-013-9234-2
Bahr (1991), Phys. Earth Planet. Inter. 66(1-2), 24-38,
    doi:10.1016/0031-9201(91)90101-M
Bahr (1988), J. Geophys. 62, 119-127 (no DOI confirmed)
Swift (1967), PhD thesis, Massachusetts Institute of Technology (no DOI
    confirmed)

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
2026-11-28  Claude Sonnet 5.5 (Anthropic): initial Python port of the
            WALDIM 1.2 classification, with optional period/frequency bands.
2026-11-28  Claude Sonnet 5.5 (Anthropic): (b) EDI reading through py4mt
            data_proc.load_edi with explicit options (edi_options,
            missing_error_percent, use_filename_as_station, default True);
            corrected rules are now the default (waldim_compat=False); added
            apparent resistivity, phase, strike, twist and shear with errors,
            their band averages, and write_results(); corrected the source
            errors listed above; standard-deviation errors holding the EDI
            empty value are now recognised as missing.
2026-11-28  Claude Sonnet 5.5 (Anthropic): (c) added site_parameters() and
            save_site_npz(); results record waldim_compat.
2026-11-28  Claude Sonnet 5.5 (Anthropic): (d) module renamed from waldim.py
            to dimens.py; added the phase tensor, Swift and Bahr skews, phase
            tensor classes (ft_convention), reason codes and flags, classes at
            other thresholds (stability_thresholds), an optional bootstrap
            (n_bootstrap), band errors that include the scatter, the input of
            .npz collections and station files, summarise_band(),
            survey_summary() and write_summary(); save_site_npz() compares
            file names without case.
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
CLASS_NAMES: Dict[int, str] = {
    0: "undetermined",
    1: "1D",
    2: "2D",
    3: "3D/2Dtw",
    4: "3D/2D",
    5: "3D",
    6: "3D/2Ddiag",
    7: "3D/1D2D",
    8: "ANIS-HINT1",
    9: "ANIS-HINT2",
}
NO_DATA = -1
NO_DATA_NAME = "no data"
INVARIANT_NAMES: Tuple[str, ...] = (
    "I1", "I2", "I3", "I4", "I5", "I6", "I7", "Q")
REASON_NAMES: Dict[int, str] = {   # why a frequency is undetermined
    0: "classified",
    1: "bias test",     # true and Monte Carlo invariants differ
    2: "error bars",    # an invariant cannot be decided within its error
    3: "no rule",       # the combination of flags has no class
}
STABILITY_THRESHOLDS: Tuple[float, ...] = (0.10, 0.12, 0.18, 0.20)
PT_CLASS_NAMES: Dict[int, str] = {  # dimensionality from the phase tensor
    0: "undefined",
    1: "1D",
    2: "2D",
    3: "3D",
}
PT_BETA_THRESHOLD = 3.0           # deg, |beta| above it: 3D
PT_ELLIPTICITY_THRESHOLD = 0.1    # ellipticity above it (and low |beta|): 2D
FT_CONVENTIONS: Tuple[str, ...] = ("auto", "e+iwt", "e-iwt")

_NULL_LIMIT = 1.0e31      # EDI empty values (1.0e32) lie above this
_ZERO_INVARIANT = 1.0e-5  # replaces invariants that are exactly zero
_ROUND_TOL = 1.0e-10      # round-off allowance in the bias test
_EDGE_TOL = 1.0e-9        # relative tolerance on band edges
_STRIKE_TOL = 10.0        # degrees, strike agreement
_OPEN = -1                # flag: not classifiable (Fortran 34000)
_NOQ = 1000               # flag: I7 undefined or not usable (Fortran 1000)
_PAIR_STRIKE = (2, 8)     # classes with strike from the Re and Im parts
_MAIN_STRIKE = (3, 4, 9)  # classes with strike from the third estimate
_DISTORTION = (3, 4)      # classes with twist and shear
_SUMMARY_VALUES = (       # band values in the survey summary
    "strike_deg", "strike_err_deg", "twist_deg", "twist_err_deg",
    "shear_deg", "shear_err_deg", "pt_azimuth_deg", "pt_azimuth_err_deg",
    "pt_abs_beta_deg", "pt_ellipticity_median", "swift_skew", "bahr_skew")
_SHORT_NAMES = {          # column titles of the text summary
    "undetermined": "undet", "3D/2Dtw": "3/2tw", "3D/2D": "3/2D",
    "3D/2Ddiag": "3/2dg", "3D/1D2D": "3/12D", "ANIS-HINT1": "ANIS1",
    "ANIS-HINT2": "ANIS2", "undefined": "undef"}
_Z_UNITS = "mV/km/nT"
_COMPONENTS = ("Zxx", "Zxy", "Zyx", "Zyy")
_PT_NAMES = (             # phase tensor parameters and their errors
    ("pt_phimax_deg", "pt_phimax_err_deg"),
    ("pt_phimin_deg", "pt_phimin_err_deg"),
    ("pt_alpha_deg", "pt_alpha_err_deg"),
    ("pt_beta_deg", "pt_beta_err_deg"),
    ("pt_azimuth_deg", "pt_azimuth_err_deg"),
    ("pt_ellipticity", "pt_ellipticity_err"))
_PT_PERIODIC = ("pt_alpha_deg", "pt_beta_deg", "pt_azimuth_deg")  # 180 deg
_SKEW_NAMES = (           # skews and their errors
    ("swift_skew", "swift_skew_err"),
    ("bahr_skew", "bahr_skew_err"))
_BAND_MEDIANS = (         # band medians: name and per-frequency quantity
    ("pt_abs_beta_deg", "pt_beta_deg"),
    ("pt_ellipticity_median", "pt_ellipticity"),
    ("swift_skew", "swift_skew"),
    ("bahr_skew", "bahr_skew"))
_PT_BAND = (              # band averages: name and period of the quantity
    ("pt_phimax_deg", None), ("pt_phimin_deg", None),
    ("pt_beta_deg", 180.0), ("pt_azimuth_deg", 180.0),
    ("pt_ellipticity", None))
_SITE_ARRAYS = (          # per-frequency entries of site_parameters()
    "freq", "period", "index", "class_id", "class_name", "invariants",
    "flags", "reason_id", "reason_name",
    "rho_a", "rho_a_err", "phase_deg", "phase_err_deg",
    "strike_deg", "strike_err_deg", "twist_deg", "twist_err_deg",
    "shear_deg", "shear_err_deg",
    "pt", "pt_err", "pt_class", "pt_class_name") + tuple(
    n for pair in _PT_NAMES + _SKEW_NAMES for n in pair)


def _error_names(name: str) -> Tuple[str, str, str]:
    """Names of the total, propagated and scatter error of a band average."""
    stem, unit = (name[:-4], "_deg") if name.endswith("_deg") else (name, "")
    return tuple(  # type: ignore[return-value]
        f"{stem}_err{kind}{unit}" for kind in ("", "_prop", "_scatter"))


_BAND_VALUES = ("strike_deg", "twist_deg", "shear_deg") + tuple(
    name for name, _ in _PT_BAND)
_BAND_ARRAYS = (          # per-band entries and their types
    ("period_min", float), ("period_max", float),
    ("data_period_min", float), ("data_period_max", float),
    ("n_periods", int), ("n_classified", int), ("n_averaged", int),
    ("class_id", int), ("class_name", "U12"), ("pt_class", int)) + tuple(
    (key, float) for name in _BAND_VALUES
    for key in (name,) + _error_names(name)) + tuple(
    (name, float) for name, _ in _BAND_MEDIANS)


# ---------------------------------------------------------------------------
# public interface
# ---------------------------------------------------------------------------
def classify_dimensionality(
    sites: Any,
    *,
    threshold: float = 0.15,
    q_threshold: float = 0.10,
    error_percent: Optional[float] = None,
    missing_error_percent: Optional[float] = None,
    bands: bool = False,
    bands_per_decade: float = 1.0,
    band_edges: Optional[Sequence[float]] = None,
    band_domain: str = "period",
    n_realizations: int = 100,
    seed: Optional[int] = 0,
    waldim_compat: bool = False,
    stability_thresholds: Optional[Sequence[float]] = STABILITY_THRESHOLDS,
    n_bootstrap: int = 0,
    pt_beta_threshold: float = PT_BETA_THRESHOLD,
    pt_ellipticity_threshold: float = PT_ELLIPTICITY_THRESHOLD,
    ft_convention: str = "auto",
    use_filename_as_station: bool = True,
    edi_options: Optional[Mapping[str, Any]] = None,
    loader: Optional[Callable[..., Mapping[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Analyse the dimensionality of MT data, site by site.

    All parameters but the first are keywords.

    Parameters
    ----------
    sites : dict, sequence, str or Path
        One of: a py4mt data dict; a sequence of data dicts and/or EDI paths
        (for example the list returned by data_proc.make_collection); a
        mapping {site name: data dict or EDI path}; the path of one EDI file;
        the path of a directory holding *.edi files; the path of a .npz
        collection (data_proc.save_list_of_dicts_npz, key "records") or of a
        station file (data_proc.save_npz). Paths may stand in a sequence or
        mapping. A mapping with a key "Z" is one data dict. .npz input holds
        pickled objects: read only files that you trust.
        A data dict needs "freq" (n,) in Hz and complex "Z" (n, 2, 2) in
        mV/km/nT. It may hold "Z_err" (n, 2, 2), "err_kind" ("var", default,
        or "std"), "rot" (n,) or scalar in degrees, "station", "lat_deg",
        "lon_deg". As in WALDIM, the error of an element applies to its real
        and to its imaginary part. The classes do not depend on the overall
        unit of Z; the apparent resistivity does.
    threshold : float, optional
        Threshold for the invariants I3 to I7, in [0, 1].
    q_threshold : float, optional
        Threshold for the invariant Q, in [0, 1].
    error_percent : float, optional
        If given, "Z_err" is ignored for all sites and the standard error of
        each tensor element is this percentage of the element (WALDIM "error
        percentage", whose default is 5).
    missing_error_percent : float, optional
        The same, but only for sites without errors (for example Phoenix
        spectra EDI files). Without it such a site raises an exception.
    bands : bool, optional
        Also summarise the results in period bands.
    bands_per_decade : float, optional
        Number of bands per decade (default 1). Edges lie at 10**(k / n) and
        are common to all sites.
    band_edges : sequence of float, optional
        Explicit band edges (turns banding on and overrides
        bands_per_decade). Periods in s or frequencies in Hz, see
        band_domain.
    band_domain : {"period", "frequency"}, optional
        Meaning of band_edges. Results are always listed by ascending period,
        and each band is (period_min, period_max].
    n_realizations : int, optional
        Monte Carlo realizations for the bias test and the errors (WALDIM:
        100).
    seed : int or None, optional
        Random seed. Each site uses its own generator started from this seed.
    waldim_compat : bool, optional
        False (default) uses corrected rules: strikes modulo 90 degrees,
        percentage errors from max(|Re|, |Im|), shear sign matched to the
        reported strike, band errors that include the scatter. True
        reproduces the WALDIM 1.2 rules. Monte Carlo and analytic errors are
        computed correctly in both cases.
    stability_thresholds : sequence of float or None, optional
        The classes are also determined with these values of threshold
        (default 0.10, 0.12, 0.18, 0.20; threshold itself is added;
        q_threshold stays). None or empty: not done.
    n_bootstrap : int, optional
        Number of repetitions of the classification with noise of the size of
        the errors added to Z (default 0: none; the cost grows in
        proportion). It does not change any other result. The fraction of
        repetitions with the same class is a flag of the reliability of a
        class, not a probability, and the most frequent class of the
        repetitions is biased towards complex classes: use class_id.
    pt_beta_threshold : float, optional
        Phase tensor class: 3D if |beta| exceeds this [deg].
    pt_ellipticity_threshold : float, optional
        Phase tensor class: otherwise 2D if the ellipticity exceeds this,
        else 1D.
    ft_convention : {"auto", "e+iwt", "e-iwt"}, optional
        Time convention of Z, used for the phase tensor (see the module
        docstring). "auto": from the data of each site; a site with a low
        "pt_ft_agreement" deserves a look.
    use_filename_as_station : bool, optional
        For EDI files: take the site name from the file name (default, as in
        data_proc.make_collection) instead of the EDI header. Data dicts, also
        the records of .npz files, are named after their "station" entry (a
        record without one: the file name, numbered if there are several).
    edi_options : mapping, optional
        Further keyword arguments for the EDI reader, for example
        {"manufacturer": "phoenix"} or {"prefer_spectra": False}. With the
        default reader, err_kind="var", drop_invalid_periods=False and
        freq_order="inc" are set unless given here.
    loader : callable, optional
        Function (path, **edi_options) -> data dict for EDI paths. Default is
        data_proc.load_edi, which must then be importable.

    Returns
    -------
    dict
        {site name: result}, each result holding

        station, lat_deg, lon_deg, threshold, q_threshold, waldim_compat
        freq, period : (m,) valid data, sorted by ascending period
        index : (m,) position of each row in the input arrays
        reason_id, reason_name : (m,) why a frequency is undetermined
            (REASON_NAMES; 0 classified); reason_counts {name: number}
        flags : (m, 5) I3 ... I7 below (0) or above (1) the threshold, -1
            undecided (I7: not defined or Q too small)
        class_id : (m,) int, class codes 0 to 9
        class_name : (m,) str
        counts : {class name: number of frequencies}, all ten entries
        invariants : (m, 8) |I1| ... |I7|, Q (see INVARIANT_NAMES)
        rho_a, rho_a_err : (m,) ohm m, from I1 and I2
        phase_deg, phase_err_deg : (m,) from I1 and I2, in [0, 90]
        strikes_deg, strikes_err_deg : (m, 3) strikes from Re, from Im, and
            from the distortion decomposition
        strike_deg, strike_err_deg : (m,) strike of the class: from Re and Im
            (classes 2, 8), third estimate (3, 4, 9), else NaN
        twist_deg, twist_err_deg, shear_deg, shear_err_deg : (m,) classes 3
            and 4, else NaN
        distortion : dict of (m,) arrays for every frequency, whatever its
            class: angle_sum_deg (twist + shear), angle_diff_deg (twist -
            shear), twist_deg, shear_deg, and the same names with _err
        analytic_errors : dict, first-order errors of rho_a (m,), phase_deg
            (m,) and strikes_deg (m, 3); the entries without this prefix are
            Monte Carlo standard deviations, as reported by WALDIM
        pt, pt_err : (m, 2, 2) phase tensor in the coordinates of the data
        pt_phimax_deg, pt_phimin_deg, pt_alpha_deg, pt_beta_deg,
            pt_azimuth_deg, pt_ellipticity : (m,), errors with _err (for
            example pt_beta_err_deg, pt_ellipticity_err); alpha and azimuth
            include rot, in [-90, 90) and [0, 180)
        swift_skew, bahr_skew and swift_skew_err, bahr_skew_err : (m,)
        pt_class, pt_class_name : (m,); pt_counts {name: number}
        pt_beta_threshold_deg, pt_ellipticity_threshold
        pt_ft_convention : "e+iwt" or "e-iwt", as used; pt_ft_agreement :
            share of frequencies that agree with it
        threshold_values (k,), class_by_threshold (m, k),
            threshold_agreement (m,) : with stability_thresholds; agreement =
            share of the thresholds that give the class
        n_bootstrap, bootstrap_class_id (m, n_bootstrap),
            bootstrap_class_prob (m, 10), bootstrap_agreement (m,) : with
            n_bootstrap > 0
        n_dropped : number of null or invalid input frequencies
        error_source : "Z_err" or "percent"
        bands : list of dicts (only with banding), keys period_min,
            period_max, freq_min, freq_max and those of summarise_band().
            class_id is the WALDIM band class (3 is merged into 4), 0 if
            nothing was classified and -1 if the band holds no data. Strike,
            twist and shear are averaged over the n_averaged frequencies of
            the band class and are NaN where the class has no such parameter;
            the phase tensor values use all frequencies of the band.

    Raises
    ------
    ValueError
        On invalid parameters, inconsistent array shapes, missing errors or
        duplicate site names.

    Warns
    -----
    UserWarning
        For a site without any valid frequency, a tensor component that
        load_edi returned as zero, missing rotation angles, or unexpected
        impedance units.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    if not 0.0 <= q_threshold <= 1.0:
        raise ValueError("q_threshold must lie in [0, 1]")
    for label, value in (("error_percent", error_percent),
                         ("missing_error_percent", missing_error_percent)):
        if value is not None and not value >= 0.0:
            raise ValueError(f"{label} must be >= 0")
    if n_realizations < 2:
        raise ValueError("n_realizations must be at least 2")
    if not bands_per_decade > 0.0:
        raise ValueError("bands_per_decade must be > 0")
    if band_domain not in ("period", "frequency"):
        raise ValueError("band_domain must be 'period' or 'frequency'")
    if n_bootstrap < 0 or int(n_bootstrap) != n_bootstrap:
        raise ValueError("n_bootstrap must be an integer >= 0")
    for label, value in (
            ("pt_beta_threshold", pt_beta_threshold),
            ("pt_ellipticity_threshold", pt_ellipticity_threshold)):
        if not value >= 0.0:
            raise ValueError(f"{label} must be >= 0")
    if ft_convention not in FT_CONVENTIONS:
        raise ValueError(f"ft_convention must be one of {FT_CONVENTIONS}")
    sweep: Optional[np.ndarray] = None
    if stability_thresholds is not None and len(stability_thresholds):
        sweep = np.asarray(stability_thresholds, dtype=float).ravel()
        if not np.all((sweep >= 0.0) & (sweep <= 1.0)):
            raise ValueError("stability_thresholds must lie in [0, 1]")
        sweep = np.unique(np.append(sweep, threshold))

    results: Dict[str, Dict[str, Any]] = {}
    gathered = _gather_sites(
        sites, loader, edi_options, use_filename_as_station)
    for name, data in gathered:
        if name in results:
            raise ValueError(f"duplicate site name: {name!r}")
        results[name] = _analyse_site(
            name, data,
            threshold=threshold,
            q_threshold=q_threshold,
            error_percent=error_percent,
            missing_error_percent=missing_error_percent,
            n_realizations=n_realizations,
            seed=seed,
            waldim_compat=waldim_compat,
            sweep=sweep,
            n_bootstrap=int(n_bootstrap),
            pt_beta_threshold=float(pt_beta_threshold),
            pt_ellipticity_threshold=float(pt_ellipticity_threshold),
            ft_convention=ft_convention,
        )

    if bands or band_edges is not None:
        if results:
            periods = np.concatenate([r["period"] for r in results.values()])
        else:
            periods = np.empty(0)
        edges = _band_edges(periods, bands_per_decade, band_edges, band_domain)
        for res in results.values():
            res["bands"] = _band_results(
                res, edges, waldim_compat=waldim_compat)

    return results


def write_results(
    results: Mapping[str, Mapping[str, Any]],
    root: Any,
) -> List[Path]:
    """
    Write WALDIM-style tables from the output of classify_dimensionality.

    Parameters
    ----------
    results : mapping
        Output of classify_dimensionality.
    root : str or Path
        Output file root, including the directory, which must exist.

    Returns
    -------
    list of Path
        root_DIM_<th>.dat (one line per site and frequency) and, if the
        results hold bands, root_BANDCLASS_<th>.dat (one line per site and
        band with data). <th> is the threshold with two decimals. Existing
        files are overwritten.

    Notes
    -----
    The columns follow WALDIM 1.2 and are read by mt_kmz_waldim.py. Blanks and
    commas in site names become "_". Undefined values are written as "nan"
    (WALDIM writes 0 for strike, twist and shear of bands without them). The
    phase tensor, the skews and the other additions are not in these tables.
    """
    if not results:
        raise ValueError("no results to write")
    thresholds = {float(r["threshold"]) for r in results.values()}
    if len(thresholds) != 1:
        raise ValueError("results were computed with different thresholds")
    tag = f"{thresholds.pop():4.2f}"

    names = {key: _table_name(key) for key in results}
    if len(set(names.values())) != len(names):
        raise ValueError("site names are not unique after removing blanks")
    width = max(12, max(len(v) for v in names.values()))

    head = (f"{'Site':<{width}s} {'Longitude':>12s} {'Latitude':>12s} "
            f"{'F(Hz)':>13s} {'Per(s)':>13s} {'DIM':>3s}")
    for number in range(3, 8):
        head += f" {'st' + str(number):>10s} {'errst' + str(number):>10s}"
    lines = [head]
    for key, res in results.items():
        dist = res["distortion"]
        values = np.column_stack([
            res["strikes_deg"], dist["angle_sum_deg"],
            dist["angle_diff_deg"]])
        errors = np.column_stack([
            res["strikes_err_deg"], dist["angle_sum_err_deg"],
            dist["angle_diff_err_deg"]])
        for k in range(res["freq"].size):
            row = (f"{names[key]:<{width}s} "
                   f"{_table_coord(res['lon_deg'])} "
                   f"{_table_coord(res['lat_deg'])} "
                   f"{res['freq'][k]:13.6e} {res['period'][k]:13.6e} "
                   f"{int(res['class_id'][k]):3d}")
            for value, error in zip(values[k], errors[k]):
                row += f" {value:10.4f} {error:10.4f}"
            lines.append(row)
    paths = [Path(f"{root}_DIM_{tag}.dat")]
    paths[0].write_text("\n".join(lines) + "\n", encoding="utf-8")

    if not any("bands" in r for r in results.values()):
        return paths

    head = (f"{'Site':<{width}s} {'Longitude':>12s} {'Latitude':>12s} "
            f"{'BAND':>4s} {'Tmin':>13s} {'Tmax':>13s} {'nper':>4s} "
            f"{'DIM':>3s}")
    for label in ("strike", "errstr", "twi", "errtwi", "shear", "errsh"):
        head += f" {label:>10s}"
    head += f" {'cont':>4s} {'scale':>10s} {'strikecomp':>10s}"
    lines = [head]
    for key, res in results.items():
        for number, band in enumerate(res.get("bands", []), start=1):
            if band["n_periods"] == 0:
                continue
            code = band["class_id"]
            error = band["strike_err_deg"]
            flag = int(2 <= code <= 4 or code >= 8)
            if np.isnan(error):
                scale = np.nan
            else:
                scale = 5.0 if error <= 0.0 else min(1.0 + 1.0 / error, 5.0)
            row = (f"{names[key]:<{width}s} "
                   f"{_table_coord(res['lon_deg'])} "
                   f"{_table_coord(res['lat_deg'])} "
                   f"{number:4d} {band['data_period_min']:13.6e} "
                   f"{band['data_period_max']:13.6e} "
                   f"{band['n_classified']:4d} {code:3d}")
            for item in ("strike", "strike_err", "twist", "twist_err",
                         "shear", "shear_err"):
                row += f" {band[item + '_deg']:10.4f}"
            row += (f" {flag:4d} {scale:10.4f} "
                    f"{-band['strike_deg']:10.4f}")
            lines.append(row)
    paths.append(Path(f"{root}_BANDCLASS_{tag}.dat"))
    paths[1].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths


def site_parameters(result: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Collect the parameters of one site in a flat dict.

    The entries of the optional parts (bands, thresholds, bootstrap) exist
    only if these were computed.

    Parameters
    ----------
    result : mapping
        One value of the dict returned by classify_dimensionality.

    Returns
    -------
    dict
        Every site gets the same keys (band keys only if banding was used).
        A strike, twist or shear angle that does not apply to the class of a
        frequency or band is NaN, as is its error.

        Scalars:
            station, lat_deg, lon_deg (NaN if unknown), threshold,
            q_threshold, waldim_compat, error_source, n_dropped,
            pt_ft_convention, pt_ft_agreement, pt_beta_threshold_deg,
            pt_ellipticity_threshold, n_bootstrap (with the bootstrap)
        Per frequency, (m,), by ascending period:
            freq, period, index, class_id, class_name
            invariants (m, 8), named in invariant_names
            rho_a, rho_a_err, phase_deg, phase_err_deg : all classes
            strike_deg, strike_err_deg : classes 2, 3, 4, 8, 9
            twist_deg, twist_err_deg, shear_deg, shear_err_deg : classes 3, 4
            reason_id, reason_name, flags (m, 5)
            pt, pt_err (m, 2, 2), the phase tensor parameters and skews with
            their errors, pt_class, pt_class_name
            class_by_threshold (m, k), threshold_agreement, threshold_values
            bootstrap_class_prob (m, 10), bootstrap_agreement
        Class statistics:
            class_codes, class_names, class_counts (10,)
            reason_codes, reason_names, reason_counts (4,)
            pt_class_codes, pt_class_names, pt_class_counts (4,)
        Per band, (nb,), all bands including empty ones (class_id -1):
            band_period_min, band_period_max, band_data_period_min,
            band_data_period_max, band_n_periods, band_n_classified,
            band_n_averaged, band_class_id, band_class_name
            band_strike_deg, band_strike_err_deg : band classes 2, 4, 8, 9
            band_twist_deg, band_twist_err_deg, band_shear_deg,
            band_shear_err_deg : band class 4 (includes class 3)
            band_class_counts (nb, 10)
            band_pt_class, band_pt_class_counts (nb, 4)
            band_pt_..._deg, band_pt_ellipticity : means; every band average
            also with ..._err_prop and ..._err_scatter
            band_pt_abs_beta_deg, band_pt_ellipticity_median, band_swift_skew,
            band_bahr_skew : medians
            band_reason_counts (nb, 4)
            band_class_by_threshold (nb, k), band_threshold_agreement
            band_bootstrap_class_prob (nb, 10), band_bootstrap_agreement
    """
    names = list(CLASS_NAMES.values())
    out: Dict[str, Any] = {
        "station": str(result["station"]),
        "lat_deg": _coordinate(result["lat_deg"]),
        "lon_deg": _coordinate(result["lon_deg"]),
        "threshold": float(result["threshold"]),
        "q_threshold": float(result["q_threshold"]),
        "waldim_compat": bool(result["waldim_compat"]),
        "error_source": str(result["error_source"]),
        "n_dropped": int(result["n_dropped"]),
    }
    for key in _SITE_ARRAYS:
        out[key] = np.array(result[key])
    out["invariant_names"] = np.array(INVARIANT_NAMES)
    out["class_codes"] = np.array(list(CLASS_NAMES), dtype=int)
    out["class_names"] = np.array(names)
    out["class_counts"] = np.array(
        [result["counts"][n] for n in names], dtype=int)
    reasons = list(REASON_NAMES.values())
    out["reason_codes"] = np.array(list(REASON_NAMES), dtype=int)
    out["reason_names"] = np.array(reasons)
    out["reason_counts"] = np.array(
        [result["reason_counts"][n] for n in reasons], dtype=int)
    pt_names = list(PT_CLASS_NAMES.values())
    out["pt_ft_convention"] = str(result["pt_ft_convention"])
    out["pt_ft_agreement"] = float(result["pt_ft_agreement"])
    out["pt_beta_threshold_deg"] = float(result["pt_beta_threshold_deg"])
    out["pt_ellipticity_threshold"] = float(
        result["pt_ellipticity_threshold"])
    out["pt_class_codes"] = np.array(list(PT_CLASS_NAMES), dtype=int)
    out["pt_class_names"] = np.array(pt_names)
    out["pt_class_counts"] = np.array(
        [result["pt_counts"][n] for n in pt_names], dtype=int)
    if "class_by_threshold" in result:
        for key in ("threshold_values", "class_by_threshold",
                    "threshold_agreement"):
            out[key] = np.array(result[key])
    if "bootstrap_class_prob" in result:
        out["n_bootstrap"] = int(result["n_bootstrap"])
        for key in ("bootstrap_class_prob", "bootstrap_agreement"):
            out[key] = np.array(result[key])

    if "bands" in result:
        bands = result["bands"]
        for key, kind in _BAND_ARRAYS:
            out["band_" + key] = np.array(
                [b[key] for b in bands], dtype=kind)
        out["band_class_counts"] = np.array(
            [[b["counts"][n] for n in names] for b in bands],
            dtype=int).reshape(len(bands), len(names))
        out["band_pt_class_counts"] = np.array(
            [[b["pt_counts"][n] for n in pt_names] for b in bands],
            dtype=int).reshape(len(bands), len(pt_names))
        out["band_reason_counts"] = np.array(
            [[b["reason_counts"][n] for n in reasons] for b in bands],
            dtype=int).reshape(len(bands), len(reasons))
        if "class_by_threshold" in result:
            out["band_class_by_threshold"] = np.array(
                [b["class_by_threshold"] for b in bands],
                dtype=int).reshape(len(bands), out["threshold_values"].size)
            out["band_threshold_agreement"] = np.array(
                [b["threshold_agreement"] for b in bands], dtype=float)
        if "bootstrap_class_prob" in result:
            out["band_bootstrap_class_prob"] = np.array(
                [b["bootstrap_class_prob"] for b in bands],
                dtype=float).reshape(len(bands), len(names))
            out["band_bootstrap_agreement"] = np.array(
                [b["bootstrap_agreement"] for b in bands], dtype=float)
    return out


def save_site_npz(
    results: Mapping[str, Mapping[str, Any]],
    directory: Any,
    *,
    suffix: str = "_dim",
) -> List[Path]:
    """
    Save the parameters of every site to its own compressed .npz file.

    Parameters
    ----------
    results : mapping
        Output of classify_dimensionality.
    directory : str or Path
        Output directory; created if it does not exist.
    suffix : str, optional
        The files are <directory>/<site><suffix>.npz.

    Returns
    -------
    list of Path
        The files written, in the order of the sites. Existing files are
        overwritten.

    Raises
    ------
    ValueError
        If two sites would get the same file name.

    Notes
    -----
    Each file holds the entries of site_parameters(). No entry needs pickle,
    so numpy.load works with its defaults; data_proc.load_npz of py4mt reads
    the files too and returns the scalars as Python values. In site names,
    blanks, commas and path separators become "_".
    """
    names = {
        key: _table_name(key).replace("/", "_").replace("\\", "_")
        for key in results}
    seen: Dict[str, str] = {}
    for key, value in names.items():
        other = seen.setdefault(value.casefold(), key)
        if other != key:
            raise ValueError(
                "site names do not give unique file names (case is "
                f"ignored): {other!r} and {key!r}")
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for key, result in results.items():
        path = folder / f"{names[key]}{suffix}.npz"
        np.savez_compressed(path, **site_parameters(result))
        paths.append(path)
    return paths


def survey_summary(
    results: Mapping[str, Mapping[str, Any]],
) -> Dict[str, np.ndarray]:
    """
    Collect the band summaries of all sites in arrays.

    Parameters
    ----------
    results : mapping
        Output of classify_dimensionality, all sites with the same bands (a
        band may carry a "label"). Without bands, every site is summarised
        over all its frequencies.

    Returns
    -------
    dict
        station, lat_deg, lon_deg, n_freq, n_dropped, error_source,
            pt_ft_convention, pt_ft_agreement : (ns,)
        site_class_counts (ns, 10), site_pt_class_counts (ns, 4) : number of
            frequencies of each site by class
        band_label, band_period_min, band_period_max : (nb,)
        (ns, nb): n_periods, n_classified, class_id (-1: no data), pt_class,
            strike_deg, strike_err_deg, twist_deg, twist_err_deg,
            shear_deg, shear_err_deg, pt_azimuth_deg, pt_azimuth_err_deg,
            pt_abs_beta_deg, pt_ellipticity_median, swift_skew, bahr_skew,
            and threshold_agreement, bootstrap_agreement if computed
        class_codes, class_names, pt_class_codes, pt_class_names
        class_site_counts (nb, 10), pt_class_site_counts (nb, 4) : number of
            sites with data in each band, by class

    Raises
    ------
    ValueError
        Without results or if the bands of the sites differ.
    """
    if not results:
        raise ValueError("no results to summarise")
    sites = list(results.values())
    if all("bands" in r for r in sites):
        tables = [list(r["bands"]) for r in sites]
    elif not any("bands" in r for r in sites):
        periods = np.concatenate([r["period"] for r in sites])
        limits = (float(periods.min()), float(periods.max())) \
            if periods.size else (float("nan"), float("nan"))
        tables = [[dict(
            summarise_band(r, np.arange(r["period"].size)),
            period_min=limits[0], period_max=limits[1], label="all data")]
            for r in sites]
    else:
        raise ValueError("some sites have bands and others not")

    def edges(table: Sequence[Mapping[str, Any]]) -> np.ndarray:
        return np.array(
            [(b["period_min"], b["period_max"]) for b in table],
            dtype=float).reshape(len(table), 2)

    first = edges(tables[0])
    for table in tables[1:]:
        if not np.array_equal(edges(table), first, equal_nan=True):
            raise ValueError("the bands of the sites differ")
    shape = (len(sites), len(tables[0]))
    out: Dict[str, np.ndarray] = {
        "station": np.array([str(r["station"]) for r in sites]),
        "lat_deg": np.array([_coordinate(r["lat_deg"]) for r in sites]),
        "lon_deg": np.array([_coordinate(r["lon_deg"]) for r in sites]),
        "n_freq": np.array([r["period"].size for r in sites], dtype=int),
        "n_dropped": np.array([r["n_dropped"] for r in sites], dtype=int),
        "error_source": np.array([str(r["error_source"]) for r in sites]),
        "pt_ft_convention": np.array(
            [str(r["pt_ft_convention"]) for r in sites]),
        "pt_ft_agreement": np.array(
            [r["pt_ft_agreement"] for r in sites], dtype=float),
        "site_class_counts": np.array(
            [[r["counts"][n] for n in CLASS_NAMES.values()]
             for r in sites], dtype=int),
        "site_pt_class_counts": np.array(
            [[r["pt_counts"][n] for n in PT_CLASS_NAMES.values()]
             for r in sites], dtype=int),
        "band_label": np.array([
            str(b.get("label") or
                f"{b['period_min']:.3g} - {b['period_max']:.3g} s")
            for b in tables[0]], dtype=str),
        "band_period_min": first[:, 0],
        "band_period_max": first[:, 1],
    }
    keys = [(name, int) for name in (
        "n_periods", "n_classified", "class_id", "pt_class")]
    keys += [(name, float) for name in _SUMMARY_VALUES]
    for name in ("threshold_agreement", "bootstrap_agreement"):
        if all(name in b for table in tables for b in table):
            keys.append((name, float))
    for name, kind in keys:
        out[name] = np.array(
            [[b[name] for b in table] for table in tables],
            dtype=kind).reshape(shape)
    for prefix, names, key in (
            ("class", CLASS_NAMES, "class_id"),
            ("pt_class", PT_CLASS_NAMES, "pt_class")):
        out[prefix + "_codes"] = np.array(list(names), dtype=int)
        out[prefix + "_names"] = np.array(list(names.values()))
        out[prefix + "_site_counts"] = np.stack(
            [np.sum(out[key] == code, axis=0) for code in names], axis=1)
    return out


def write_summary(
    results: Mapping[str, Mapping[str, Any]],
    directory: Any,
    *,
    name: str = "mt_dimensionality",
    header: Sequence[str] = (),
) -> List[Path]:
    """
    Write the survey summary as text, CSV table and .npz file.

    Parameters
    ----------
    results : mapping
        As for survey_summary.
    directory : str or Path
        Output directory; created if it does not exist.
    name : str, optional
        The files are <directory>/<name>.txt, .csv and .npz.
    header : sequence of str, optional
        Extra lines at the top of the text file.

    Returns
    -------
    list of Path
        The three files. Existing files are overwritten.

    Notes
    -----
    The .txt file lists the number of sites per class and band, the number
    of frequencies per class and site (n: used, out: removed) and one line
    per site and band with data; the .csv file holds the same lines with all
    columns of survey_summary (empty field: undefined); the .npz file holds
    survey_summary and loads without pickle.
    """
    table = survey_summary(results)
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    paths = [folder / f"{name}{suffix}" for suffix in (".txt", ".csv", ".npz")]
    paths[0].write_text(
        _summary_text(table, name, header), encoding="utf-8")

    per_band = ["n_periods", "n_classified", "class_id", "pt_class"]
    per_band += [k for k in _SUMMARY_VALUES + (
        "threshold_agreement", "bootstrap_agreement") if k in table]
    with paths[1].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["station", "lat_deg", "lon_deg", "band", "period_min",
             "period_max", "class_name", "pt_class_name"] + per_band)
        for i, station in enumerate(table["station"]):
            for k, label in enumerate(table["band_label"]):
                if not table["n_periods"][i, k]:
                    continue
                row = [station]
                row += [_csv_number(table[key][i])
                        for key in ("lat_deg", "lon_deg")]
                row += [label]
                row += [_csv_number(table[key][k])
                        for key in ("band_period_min", "band_period_max")]
                row += [CLASS_NAMES[int(table["class_id"][i, k])],
                        PT_CLASS_NAMES[int(table["pt_class"][i, k])]]
                row += [_csv_number(table[key][i, k]) for key in per_band]
                writer.writerow(row)
    np.savez_compressed(paths[2], **table)
    return paths


def _csv_number(value: Any) -> str:
    """A number for the CSV table; "" if undefined."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{value:.6g}" if np.isfinite(value) else ""


def _summary_text(
    table: Mapping[str, np.ndarray], name: str, header: Sequence[str],
) -> str:
    """The text of the summary file."""
    labels = [str(v) for v in table["band_label"]]
    stations = [str(v) for v in table["station"]]
    wide = max([len(v) for v in labels] + [4])
    long = max([len(v) for v in stations] + [7])
    def count(number: int, word: str) -> str:
        return f"{number} {word}{'' if number == 1 else 's'}"

    lines = [f"{name}: dimensionality summary of "
             f"{count(len(stations), 'site')}, {count(len(labels), 'band')}"]
    lines += [str(v) for v in header]

    def short_names(prefix: str) -> List[str]:
        return [_SHORT_NAMES.get(str(v), str(v))
                for v in table[prefix + "_names"]]

    titles = (("WAL (WALDIM) classes", "class"),
              ("phase tensor classes", "pt_class"))
    for title, prefix in titles:
        short = short_names(prefix)
        lines += ["", f"Number of sites per band: {title}", "-" * 78]
        lines.append(f"{'band':<{wide}s}" + "".join(f" {v:>6s}" for v in short))
        for k, label in enumerate(labels):
            lines.append(f"{label:<{wide}s}" + "".join(
                f" {n:6d}" for n in table[prefix + "_site_counts"][k]))

    for title, prefix in titles:
        short = short_names(prefix)
        phase = prefix == "pt_class"
        lines += ["", f"Number of frequencies per site: {title}", "-" * 78]
        if phase:
            lines.append("conv: time dependence assumed for the phase tensor, "
                         "agree: share of frequencies")
        head = f"{'station':<{long}s} {'n':>4s} {'out':>4s}" + "".join(
            f" {v:>6s}" for v in short)
        lines.append(head + (f" {'conv':>6s} {'agree':>5s}" if phase else ""))
        for i, station in enumerate(stations):
            line = (
                f"{station:<{long}s} {table['n_freq'][i]:4d} "
                f"{table['n_dropped'][i]:4d}" + "".join(
                    f" {n:6d}"
                    for n in table[f"site_{prefix}_counts"][i]))
            if phase:
                share = table["pt_ft_agreement"][i]
                text = f"{share:.2f}" if np.isfinite(share) else "-"
                line += (f" {str(table['pt_ft_convention'][i]):>6s} "
                         f"{text:>5s}")
            lines.append(line)

    columns = (
        ("strike_deg", "strike", 7, 1), ("strike_err_deg", "err", 5, 1),
        ("twist_deg", "twist", 6, 1), ("shear_deg", "shear", 6, 1),
        ("pt_azimuth_deg", "PTazim", 6, 1),
        ("pt_abs_beta_deg", "|beta|", 6, 1),
        ("pt_ellipticity_median", "ellip", 5, 2),
        ("swift_skew", "swift", 5, 2), ("bahr_skew", "bahr", 5, 2),
        ("threshold_agreement", "thr", 4, 1),
        ("bootstrap_agreement", "boot", 4, 2))
    columns = tuple(c for c in columns if c[0] in table)
    head = (f"{'station':<{long}s} {'band':<{wide}s} {'n':>3s} "
            f"{'class':<12s} {'PT':<5s}")
    head += "".join(f" {title:>{width}s}" for _, title, width, _ in columns)
    lines += ["", "Sites and bands (angles in degrees; - : undefined)",
              "-" * len(head), head]
    for i, station in enumerate(stations):
        for k, label in enumerate(labels):
            if not table["n_periods"][i, k]:
                continue
            pt = PT_CLASS_NAMES[int(table["pt_class"][i, k])]
            line = (f"{station:<{long}s} {label:<{wide}s} "
                    f"{table['n_periods'][i, k]:3d} "
                    f"{CLASS_NAMES[int(table['class_id'][i, k])]:<12s} "
                    f"{_SHORT_NAMES.get(pt, pt):<5s}")
            for key, _, width, digits in columns:
                value = table[key][i, k]
                text = f"{value:.{digits}f}" if np.isfinite(value) else "-"
                if len(text) > width:   # e.g. skews of diagonal tensors
                    text = f"{value:.0e}".replace("e+0", "e")
                    text = text.replace("e+", "e")
                line += f" {text:>{width}s}"
            lines.append(line)
    return "\n".join(lines) + "\n"


def _table_name(name: str) -> str:
    """Site name usable as a blank-separated table field."""
    return "_".join(str(name).replace(",", " ").split()) or "site"


def _coordinate(value: Any) -> float:
    """Longitude or latitude as a float; NaN if unknown."""
    return float("nan") if value is None else float(value)


def _table_coord(value: Any) -> str:
    """Longitude or latitude field; "nan" if unknown."""
    return f"{_coordinate(value):12.6f}"


# ---------------------------------------------------------------------------
# input handling
# ---------------------------------------------------------------------------
def _default_loader() -> Callable[..., Mapping[str, Any]]:
    """Return data_proc.load_edi, imported only when a path has to be read."""
    try:
        from data_proc import load_edi
    except ImportError as exc:
        raise ImportError(
            "reading EDI paths needs py4mt data_proc.load_edi on sys.path "
            f"(import failed: {exc}); alternatively pass loader=... or data "
            "dicts") from exc
    return load_edi


def _read_edi(
    path: str,
    loader: Optional[Callable[..., Mapping[str, Any]]],
    edi_options: Optional[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """
    Read one EDI file.

    The default reader is told to keep invalid rows and to return variances:
    load_edi would otherwise also drop rows with an invalid tipper, which is
    not used here, and set components that are invalid everywhere to zero.
    Invalid impedance rows are removed by _prepare_site.
    """
    options: Dict[str, Any] = {}
    if loader is None:
        loader = _default_loader()
        options = {"err_kind": "var", "drop_invalid_periods": False,
                   "freq_order": "inc"}
    options.update(edi_options or {})
    return loader(path, **options)


def _station_name(data: Mapping[str, Any], default: str) -> str:
    """Station name from a data dict, with fallback."""
    value = data.get("station")
    if value is None:
        return default
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    return str(value).strip() or default


def _edi_files(path: Path) -> List[Path]:
    """The EDI files of a directory, sorted."""
    files = sorted(
        p for p in path.iterdir()
        if p.suffix.lower() == ".edi" and not p.name.startswith("."))
    if not files:
        raise FileNotFoundError(f"no EDI files in {path}")
    return files


def _read_npz(path: Path) -> List[Mapping[str, Any]]:
    """
    The data dicts of a py4mt collection (data_proc.save_list_of_dicts_npz)
    or of one station file (data_proc.save_npz). Such files hold pickled
    objects: read only files that you trust.
    """
    with np.load(path, allow_pickle=True) as stored:
        if "records" in stored.files:
            return list(stored["records"].tolist())
        if "freq" in stored.files and "Z" in stored.files:
            record = {}
            for key in stored.files:
                value = stored[key]
                record[key] = value.item() if value.ndim == 0 else value
            return [record]
    raise ValueError(
        f"{path}: neither a collection (key 'records') nor a station file "
        "(keys 'freq' and 'Z')")


def _gather_sites(
    sites: Any,
    loader: Optional[Callable[..., Mapping[str, Any]]],
    edi_options: Optional[Mapping[str, Any]],
    use_filename_as_station: bool,
) -> List[Tuple[str, Mapping[str, Any]]]:
    """Turn any accepted form of `sites` into a list of (name, data dict)."""
    if isinstance(sites, (str, Path)):
        items = [(None, sites)]
    elif isinstance(sites, Mapping):
        items = [(None, sites)] if "Z" in sites else list(sites.items())
    else:
        items = [(None, s) for s in sites]

    # (given name, data dict or EDI path, default name)
    entries: List[Tuple[Any, Any, Optional[str]]] = []
    for key, item in items:
        if not isinstance(item, (str, Path)):
            entries.append((key, item, None))
            continue
        path = Path(item)
        if path.is_dir():
            if key is not None:
                raise ValueError(f"{key}: {path} is a directory, not a site")
            entries += [(None, p, p.stem) for p in _edi_files(path)]
        elif path.suffix.lower() == ".npz":
            records = _read_npz(path)
            if key is not None and len(records) != 1:
                raise ValueError(
                    f"{key}: {path} holds {len(records)} stations")
            for number, record in enumerate(records, start=1):
                default = path.stem if len(records) == 1 \
                    else f"{path.stem}_{number:03d}"
                entries.append((key, record, default))
        else:
            entries.append((key, item, path.stem))

    out: List[Tuple[str, Mapping[str, Any]]] = []
    for number, (key, item, default) in enumerate(entries, start=1):
        from_file = isinstance(item, (str, Path))
        data = _read_edi(str(item), loader, edi_options) if from_file else item
        if default is None:
            default = f"site{number:03d}"
        if not isinstance(data, Mapping):
            raise TypeError(f"site {number}: expected a data dict")
        if key is not None:
            name = key
        elif from_file and use_filename_as_station:
            name = default
        else:
            name = _station_name(data, default)
        out.append((str(name), data))
    return out


def _prepare_site(
    name: str,
    data: Mapping[str, Any],
    rng: np.random.Generator,
    *,
    error_percent: Optional[float],
    waldim_compat: bool,
    missing_error_percent: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Check one data dict, apply the WALDIM pre-processing, remove null
    frequencies and sort by ascending period.
    """
    freq = np.asarray(data["freq"], dtype=float).ravel()
    n = freq.size
    z = np.asarray(data["Z"], dtype=complex)
    if z.shape != (n, 2, 2):
        raise ValueError(f"{name}: Z must have shape ({n}, 2, 2)")

    units = data.get("Z_units")
    if units is not None and str(units) != _Z_UNITS:
        warnings.warn(
            f"{name}: Z_units is {units!s}, expected {_Z_UNITS}; the apparent "
            "resistivity will be wrong", UserWarning, stacklevel=2)

    rot_in = data.get("rot")
    if rot_in is None:
        rot = np.zeros(n)
    else:
        rot = np.array(rot_in, dtype=float).ravel()
        if rot.size == 1:
            rot = np.full(n, rot[0])
        if rot.size != n:
            raise ValueError(
                f"{name}: rot must be a scalar or have {n} values")
        if n and not np.any(np.isfinite(rot) & (np.abs(rot) < _NULL_LIMIT)):
            warnings.warn(
                f"{name}: no valid rotation angle, 0 is assumed; strikes "
                "refer to the coordinate frame of the data",
                UserWarning, stacklevel=2)
            rot = np.zeros(n)

    # load_edi returns a component that is missing or invalid at all
    # frequencies as zero
    if n and "source_kind" in data:
        empty = [label for label, column in zip(
            _COMPONENTS, z.reshape(n, 4).T) if np.all(column == 0.0)]
        if empty:
            warnings.warn(
                f"{name}: {', '.join(empty)} is zero at all frequencies; if "
                "the component is missing in the EDI file, the results of "
                "this site are meaningless", UserWarning, stacklevel=2)

    zr = np.array(z.real, dtype=float)
    zi = np.array(z.imag, dtype=float)

    # exact zeros are replaced by tiny random numbers
    for part, scale in ((zr, 1.1e-10), (zi, 1.0e-10)):
        zero = part == 0.0
        part[zero] = scale * rng.uniform(-1.0, 1.0, int(zero.sum()))

    percent = error_percent
    err = None
    if percent is None:
        err_in = data.get("Z_err")
        if err_in is not None:
            err = np.real(np.asarray(err_in)).astype(float)
            if err.shape != (n, 2, 2):
                raise ValueError(f"{name}: Z_err must have shape ({n}, 2, 2)")
        # no errors at all, or none for at least one component
        absent = err is None or (n > 0 and not np.all(np.any(
            np.isfinite(err) & (np.abs(err) < _NULL_LIMIT), axis=0)))
        if absent and missing_error_percent is not None:
            percent = missing_error_percent
        elif err is None:
            raise ValueError(
                f"{name}: no Z_err; set error_percent or "
                "missing_error_percent")

    if percent is not None:
        if waldim_compat:
            ref = np.maximum(zr, zi)
        else:
            ref = np.maximum(np.abs(zr), np.abs(zi))
        var = (0.01 * percent * ref) ** 2
        bad_err = ~np.isfinite(var) | (var >= _NULL_LIMIT)
    else:
        kind = str(data.get("err_kind", "var")).strip().lower()
        bad_err = ~np.isfinite(err) | (np.abs(err) >= _NULL_LIMIT)
        if kind.startswith("std"):
            with np.errstate(over="ignore"):
                var = np.where(bad_err, 0.0, err) ** 2
        elif kind.startswith("var"):
            bad_err |= err < 0.0
            var = err
        else:
            raise ValueError(f"{name}: unknown err_kind {kind!r}")
        bad_err |= var >= _NULL_LIMIT

    # avoid Zxy == -Zyx exactly (division by zero in the strike angles)
    for part in (zr, zi):
        anti = part[:, 0, 1] == -part[:, 1, 0]
        part[anti, 1, 0] *= 1.000001

    valid = (
        np.isfinite(freq) & (freq > 0.0) & (freq < _NULL_LIMIT)
        & np.isfinite(rot) & (np.abs(rot) < _NULL_LIMIT)
        & np.all(np.isfinite(zr) & (np.abs(zr) < _NULL_LIMIT), axis=(1, 2))
        & np.all(np.isfinite(zi) & (np.abs(zi) < _NULL_LIMIT), axis=(1, 2))
        & ~np.any(bad_err, axis=(1, 2))
    )
    index = np.flatnonzero(valid)
    index = index[np.argsort(-freq[index], kind="stable")]
    if index.size == 0:
        warnings.warn(
            f"{name}: no valid frequency (check impedances and errors)",
            UserWarning, stacklevel=2)

    return {
        "index": index,
        "freq": freq[index],
        "zr": zr[index],
        "zi": zi[index],
        "var": var[index],
        "rot": rot[index],
        "n_dropped": int(n - index.size),
        "error_source": "Z_err" if percent is None else "percent",
    }


# ---------------------------------------------------------------------------
# invariants and errors
# ---------------------------------------------------------------------------
def _decompose(part: np.ndarray) -> np.ndarray:
    """(n, 2, 2) real tensor -> (n, 4) linear combinations xi1 ... xi4."""
    xx, xy = part[:, 0, 0], part[:, 0, 1]
    yx, yy = part[:, 1, 0], part[:, 1, 1]
    return 0.5 * np.stack([xx + yy, xy + yx, xx - yy, xy - yx], axis=-1)


def _component_errors(var: np.ndarray) -> np.ndarray:
    """(n, 2, 2) variances -> (n, 4) standard errors of xi1 ... xi4."""
    diag = 0.5 * np.sqrt(var[:, 0, 0] + var[:, 1, 1])
    off = 0.5 * np.sqrt(var[:, 0, 1] + var[:, 1, 0])
    return np.stack([diag, off, diag, off], axis=-1)


def _invariants(
    xi: np.ndarray, eta: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    WAL invariants.

    Returns the signed invariants (n, 8) in the order I1 ... I7, Q and the
    auxiliary d_ij terms.
    """
    x1, x2, x3, x4 = xi.T
    h1, h2, h3, h4 = eta.T

    i1 = np.sqrt(x4 ** 2 + x1 ** 2)
    i2 = np.sqrt(h4 ** 2 + h1 ** 2)
    norm = i1 * i2
    d = {
        "12": (x1 * h2 - h1 * x2) / norm,
        "13": (x1 * h3 - h1 * x3) / norm,
        "23": (x2 * h3 - h2 * x3) / norm,
        "24": (x2 * h4 - h2 * x4) / norm,
        "34": (x3 * h4 - h3 * x4) / norm,
        "41": (x4 * h1 - h4 * x1) / norm,
    }
    first = [
        i1,
        i2,
        np.sqrt(x2 ** 2 + x3 ** 2) / i1,
        np.sqrt(h2 ** 2 + h3 ** 2) / i2,
        (x4 * h1 + x1 * h4) / norm,
        d["41"],
    ]
    q = np.sqrt((d["12"] - d["34"]) ** 2 + (d["13"] + d["24"]) ** 2)

    def nonzero(a: np.ndarray) -> np.ndarray:
        return np.where(a == 0.0, _ZERO_INVARIANT, a)

    first = [nonzero(a) for a in first]
    q = nonzero(q)
    i7 = (d["41"] - d["23"]) / q
    return np.stack(first + [i7, q], axis=-1), d


def _invariant_errors(
    xi: np.ndarray,
    eta: np.ndarray,
    inv: np.ndarray,
    d41: np.ndarray,
    efn: np.ndarray,
) -> np.ndarray:
    """Classical error propagation for I3 ... I6, returned as (n, 4)."""
    x1, x2, x3, x4 = xi.T
    h1, h2, h3, h4 = eta.T
    e1, e2, e3, e4 = efn.T
    i1, i2, i3, i4, i5 = (inv[:, k] for k in range(5))

    d3 = np.sqrt(
        (x2 ** 2 * e2 ** 2 + x3 ** 2 * e3 ** 2) / i3 ** 2
        + i3 ** 2 * (x1 ** 2 * e1 ** 2 + x4 ** 2 * e4 ** 2)) / i1 ** 2
    d4 = np.sqrt(
        (h2 ** 2 * e2 ** 2 + h3 ** 2 * e3 ** 2) / i4 ** 2
        + i4 ** 2 * (h1 ** 2 * e1 ** 2 + h4 ** 2 * e4 ** 2)) / i2 ** 2
    d5 = np.sqrt(
        (h4 - i5 * i2 * x1 / i1) ** 2 * e1 ** 2
        + (h1 - i5 * i2 * x4 / i1) ** 2 * e4 ** 2
        + (x4 - i5 * i1 * h1 / i2) ** 2 * e1 ** 2
        + (x1 - i5 * i1 * h4 / i2) ** 2 * e4 ** 2) / (i1 * i2)

    # derivatives of d41 with respect to xi1, xi4, eta1, eta4
    df1 = -h4 / (i1 * i2) - d41 * x1 / i1 ** 2
    df4 = h1 / (i1 * i2) - d41 * x4 / i1 ** 2
    dm1 = x4 / (i1 * i2) - d41 * h1 / i2 ** 2
    dm4 = -x1 / (i1 * i2) - d41 * h4 / i2 ** 2
    d6 = np.sqrt(
        (df1 ** 2 + dm1 ** 2) * e1 ** 2 + (df4 ** 2 + dm4 ** 2) * e4 ** 2)

    return np.stack([d3, d4, d5, d6], axis=-1)


def _monte_carlo(
    xi: np.ndarray, eta: np.ndarray, efn: np.ndarray, noise: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mean and standard deviation of I3 ... I6 over noisy realizations.

    noise holds standard normal deviates of shape (n_realizations, n, 8):
    four for the real parts, then four for the imaginary parts.
    """
    x = xi[None, :, :] + noise[:, :, :4] * efn[None, :, :]
    h = eta[None, :, :] + noise[:, :, 4:] * efn[None, :, :]
    x1, x2, x3, x4 = (x[:, :, k] for k in range(4))
    h1, h2, h3, h4 = (h[:, :, k] for k in range(4))

    i1 = np.sqrt(x4 ** 2 + x1 ** 2)
    i2 = np.sqrt(h4 ** 2 + h1 ** 2)
    samples = np.stack([
        np.sqrt(x2 ** 2 + x3 ** 2) / i1,
        np.sqrt(h2 ** 2 + h3 ** 2) / i2,
        (x4 * h1 + x1 * h4) / (i1 * i2),
        (x4 * h1 - x1 * h4) / (i1 * i2),
    ], axis=-1)
    return samples.mean(axis=0), samples.std(axis=0, ddof=1)


# ---------------------------------------------------------------------------
# strikes
# ---------------------------------------------------------------------------
def _first_quadrant(angle: np.ndarray) -> np.ndarray:
    """WALDIM function FQ (degrees): a single pass, not a true modulo."""
    a = np.array(angle, dtype=float)
    a = np.where((a < 0.0) & (a >= -1.0), -a, a)
    a = np.where((a <= 91.0) & (a > 90.0), 180.0 - a, a)
    a = np.where(a > 91.0, a - 90.0, a)
    a = np.where(a < -1.0, a + 90.0, a)
    return a


def _strikes(
    xi: np.ndarray,
    eta: np.ndarray,
    d: Mapping[str, np.ndarray],
    rot: np.ndarray,
    *,
    waldim_compat: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Strike angles (n, 3) in degrees: from Re, from Im, from the d_ij.

    Returns the angles folded into the first quadrant and the unfolded ones.
    """
    x2, x3 = xi[:, 1], xi[:, 2]
    h2, h3 = eta[:, 1], eta[:, 2]
    rt = np.deg2rad(rot)

    s3 = 0.5 * np.arctan(-x3 / x2) + rt
    s4 = np.where(
        h2 != 0.0,
        0.5 * np.arctan(-h3 / h2),
        0.5 * np.arcsin(np.copysign(1.0, h3))) + rt
    den = d["13"] + d["24"]
    s5 = np.where(
        den != 0.0,
        0.5 * np.arctan((d["12"] - d["34"]) / den),
        0.0) + rt

    st = np.stack([s3, s4, s5], axis=-1)
    st = np.rad2deg(np.where(st == 0.0, 1.0e-6, st))
    if waldim_compat:
        return _first_quadrant(st), st
    return np.mod(st, 90.0), st


def _distance_mod90(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest angular distance of two strikes with 90 degree periodicity."""
    return np.abs(np.mod(a - b + 45.0, 90.0) - 45.0)


def _strike_tests(
    strikes: np.ndarray, *, waldim_compat: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (aligned, consistent): do the Re and Im strikes agree, and does the
    third strike agree with their mean?
    """
    s3, s4, s5 = strikes.T
    if waldim_compat:
        diff = np.abs(s3 - s4)
        aligned = (diff < _STRIKE_TOL) | (diff > 90.0 - _STRIKE_TOL)
        consistent = np.abs(s5 - (s3 + s4) / 2.0) < _STRIKE_TOL
    else:
        aligned = _distance_mod90(s3, s4) < _STRIKE_TOL
        phasor = (np.exp(4j * np.deg2rad(s3)) + np.exp(4j * np.deg2rad(s4)))
        mean = np.rad2deg(np.angle(phasor)) / 4.0
        consistent = _distance_mod90(s5, mean) < _STRIKE_TOL
    return aligned, consistent


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------
def _flag_invariants(
    value: np.ndarray, error: np.ndarray, threshold: float,
) -> np.ndarray:
    """
    Flag I3 ... I6 as vanishing (0), non-vanishing (1) or open (-1), taking
    the error bars into account.
    """
    hi = value + error
    lo = value - error
    one = (hi > threshold) & (lo > threshold)
    zero = (hi <= threshold) & (np.abs(lo) <= threshold)
    rest = ~(one | zero) & (hi > threshold) & (np.abs(lo) <= threshold)
    one |= rest & (value > threshold)
    one |= rest & (value <= threshold) & (hi <= 1.0)

    flags = np.full(value.shape, _OPEN, dtype=int)
    flags[zero] = 0
    flags[one] = 1
    return flags


def _bias_ok(
    value: np.ndarray, mean: np.ndarray, std: np.ndarray, threshold: float,
) -> np.ndarray:
    """
    WALDIM bias test on I3 ... I6: reject a frequency if the true and the
    Monte Carlo values differ by more than one standard deviation, unless
    both (and their difference) are below the threshold.
    """
    bias = np.abs(value - mean)
    harmless = (value < threshold) & (mean < threshold) & (bias < threshold)
    bad = (bias > std + _ROUND_TOL) & ~harmless
    return ~np.any(bad, axis=1)


def _classify(
    zr: np.ndarray,
    zi: np.ndarray,
    var: np.ndarray,
    rot: np.ndarray,
    noise: np.ndarray,
    *,
    threshold: float,
    q_threshold: float,
    waldim_compat: bool,
    sweep: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Classify all frequencies of one pre-processed site. With sweep, the
    classes at these thresholds are returned too ("class_by_threshold").
    """
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        xi = _decompose(zr)
        eta = _decompose(zi)
        efn = _component_errors(var)
        signed, d = _invariants(xi, eta)
        err = _invariant_errors(xi, eta, signed, d["41"], efn)
        mc_mean, mc_std = _monte_carlo(xi, eta, efn, noise)
        strikes, unfolded = _strikes(
            xi, eta, d, rot, waldim_compat=waldim_compat)
        inv = np.abs(signed)
        aligned, consistent = _strike_tests(
            strikes, waldim_compat=waldim_compat)

    def decide(th: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _decide(
            inv, err, np.abs(mc_mean), mc_std, xi, eta, aligned, consistent,
            threshold=th, q_threshold=q_threshold)

    cls, reason, flags = decide(threshold)
    out = {
        "class_id": cls,
        "reason_id": reason,
        "flags": flags,
        "invariants": inv,
        "errors": err,
        "mc_mean": np.abs(mc_mean),
        "mc_std": mc_std,
        "strikes_deg": strikes,
        "strikes_unfolded_deg": unfolded,
    }
    if sweep is not None:
        out["class_by_threshold"] = np.stack(
            [decide(float(th))[0] for th in sweep], axis=1)
    return out


def _decide(
    inv: np.ndarray,
    err: np.ndarray,
    mc_mean: np.ndarray,
    mc_std: np.ndarray,
    xi: np.ndarray,
    eta: np.ndarray,
    aligned: np.ndarray,
    consistent: np.ndarray,
    *,
    threshold: float,
    q_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The WALDIM decision rules for one threshold. Returns the class, the
    reason code (REASON_NAMES) and the flags (m, 5) of I3 ... I7: 0 below
    and 1 above the threshold, -1 undecided (I7: undefined).
    """
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        i1, i2, i7, q = inv[:, 0], inv[:, 1], inv[:, 6], inv[:, 7]
        gate = _bias_ok(inv[:, 2:6], mc_mean, mc_std, threshold)
        flags = _flag_invariants(inv[:, 2:6], err, threshold)

        f3, f4, f5, f6 = flags.T
        f7 = np.where(
            (q <= q_threshold) | (i7 > 1.0),
            _NOQ,
            np.where(i7 > threshold, 1, 0))

        decided = np.all(flags != _OPEN, axis=1)
        usable = gate & decided
        two_d = (f3 == 1) | (f4 == 1)
        no_dist = (f5 == 0) & (f6 == 0)
        low7 = (f7 == 0) | (f7 == _NOQ)
        twist = usable & two_d & (f5 == 1) & (f6 == 0)
        family = usable & two_d & no_dist & low7
        diagonal = ((np.abs(xi[:, 3]) < threshold * i1)
                    & (np.abs(eta[:, 3]) < threshold * i2))
        regular = family & ~diagonal

    # the order of these assignments follows the Fortran code
    cls = np.zeros(q.size, dtype=int)
    cls[usable & (f3 == 0) & (f4 == 0) & no_dist & low7] = 1
    cls[family & diagonal] = 6
    cls[regular & aligned & consistent] = 2
    cls[regular & aligned & ~consistent] = 8
    cls[regular & ~aligned] = 9
    cls[twist & (f7 == 0)] = 3
    cls[usable & (f6 == 1) & (f7 == 0)] = 4
    cls[usable & (f7 == 1)] = 5
    cls[twist & (f7 == _NOQ)] = 7

    reason = np.select([cls != 0, ~gate, ~decided], [0, 1, 2], 3)
    all_flags = np.column_stack(
        [flags, np.where(f7 == _NOQ, _OPEN, f7)]).astype(int)
    return cls, reason, all_flags


# ---------------------------------------------------------------------------
# apparent resistivity, phase, strike errors, twist and shear
# ---------------------------------------------------------------------------
def _wrap(angle: np.ndarray, period: float) -> np.ndarray:
    """Angle difference mapped into [-period/2, period/2)."""
    return np.mod(angle + 0.5 * period, period) - 0.5 * period


def _rho_phase(
    freq: np.ndarray, i1: np.ndarray, i2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Invariant apparent resistivity (ohm m) and phase (deg)."""
    return (0.2 / freq) * (i1 ** 2 + i2 ** 2), np.rad2deg(np.arctan2(i2, i1))


def _distortion_angles(
    xi: np.ndarray, strike_deg: np.ndarray, rot_deg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Angles atan(A22 / A12) = twist + shear and atan(-A11 / A21) = twist -
    shear (deg), where A is the real part rotated into the strike frame.
    xi is (..., 4); strike and rotation must broadcast against xi[..., 0].
    """
    x1, x2, x3, x4 = (xi[..., k] for k in range(4))
    xx, yy = x1 + x3, x1 - x3
    xy, yx = x2 + x4, x2 - x4
    angle = np.deg2rad(strike_deg - rot_deg)
    s, c = np.sin(angle), np.cos(angle)
    a11 = (xy + yx) * s * c + xx * c ** 2 + yy * s ** 2
    a12 = (yy - xx) * s * c + xy * c ** 2 - yx * s ** 2
    a21 = (yy - xx) * s * c - xy * s ** 2 + yx * c ** 2
    a22 = -(xy + yx) * s * c + xx * s ** 2 + yy * c ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        total = np.rad2deg(np.arctan(a22 / a12))
        diff = np.rad2deg(np.arctan(-a11 / a21))
    return total, diff


def _analytic_errors(
    freq: np.ndarray,
    xi: np.ndarray,
    eta: np.ndarray,
    efn: np.ndarray,
    i1: np.ndarray,
    i2: np.ndarray,
) -> Dict[str, np.ndarray]:
    """First-order errors of apparent resistivity, phase and the strikes."""
    x1, x2, x3, x4 = xi.T
    h1, h2, h3, h4 = eta.T
    e1, e2, e3, e4 = efn.T

    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = np.sqrt(x1 ** 2 * e1 ** 2 + x4 ** 2 * e4 ** 2) / i1
        d2 = np.sqrt(h1 ** 2 * e1 ** 2 + h4 ** 2 * e4 ** 2) / i2
        rho = (0.4 / freq) * np.sqrt(i1 ** 2 * d1 ** 2 + i2 ** 2 * d2 ** 2)
        phase = np.sqrt(
            (i2 * d1) ** 2 + (i1 * d2) ** 2) / (i1 ** 2 + i2 ** 2)

        s3 = 0.5 * np.sqrt(
            x3 ** 2 * e2 ** 2 + x2 ** 2 * e3 ** 2) / (x2 ** 2 + x3 ** 2)
        s4 = 0.5 * np.sqrt(
            h3 ** 2 * e2 ** 2 + h2 ** 2 * e3 ** 2) / (h2 ** 2 + h3 ** 2)

        # strike 3 is atan(num / den) / 2; the normalisation by I1 I2 cancels
        num = (x1 * h2 - h1 * x2) - (x3 * h4 - h3 * x4)
        den = (x1 * h3 - h1 * x3) + (x2 * h4 - h2 * x4)
        num_x = np.stack([h2, -h1, -h4, h3], axis=-1)
        den_x = np.stack([h3, h4, -h1, -h2], axis=-1)
        num_h = np.stack([-x2, x1, x4, -x3], axis=-1)
        den_h = np.stack([-x3, -x4, x1, x2], axis=-1)
        grad_x = num_x * den[:, None] - num[:, None] * den_x
        grad_h = num_h * den[:, None] - num[:, None] * den_h
        s5 = 0.5 * np.sqrt(np.sum(
            (grad_x ** 2 + grad_h ** 2) * efn ** 2, axis=1)
        ) / (num ** 2 + den ** 2)

    return {
        "rho_a": rho,
        "phase_deg": np.rad2deg(phase),
        "strikes_deg": np.rad2deg(np.stack([s3, s4, s5], axis=-1)),
    }


def _parameters(
    freq: np.ndarray,
    xi: np.ndarray,
    eta: np.ndarray,
    efn: np.ndarray,
    rot: np.ndarray,
    invariants: np.ndarray,
    strikes: np.ndarray,
    unfolded: np.ndarray,
    noise: np.ndarray,
    *,
    waldim_compat: bool,
) -> Dict[str, Any]:
    """
    Apparent resistivity, phase, distortion angles and the errors of these and
    of the strikes. Errors are standard deviations over the realizations;
    angle differences are taken modulo their period.
    """
    i1, i2 = invariants[:, 0], invariants[:, 1]
    rho, phase = _rho_phase(freq, i1, i2)

    # strike used to rotate the real part; WALDIM takes it unfolded
    base = unfolded[:, 2] if waldim_compat else strikes[:, 2]
    total, diff = _distortion_angles(xi, base, rot)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        x = xi[None, :, :] + noise[:, :, :4] * efn[None, :, :]
        h = eta[None, :, :] + noise[:, :, 4:] * efn[None, :, :]
        x1, x2, x3, x4 = (x[:, :, k] for k in range(4))
        h1, h2, h3, h4 = (h[:, :, k] for k in range(4))

        rho_r, phase_r = _rho_phase(
            freq, np.hypot(x1, x4), np.hypot(h1, h4))
        s3 = 0.5 * np.arctan(-x3 / x2)
        s4 = 0.5 * np.arctan(-h3 / h2)
        num = (x1 * h2 - h1 * x2) - (x3 * h4 - h3 * x4)
        den = (x1 * h3 - h1 * x3) + (x2 * h4 - h2 * x4)
        s5 = 0.5 * np.arctan(num / den)
        samples = np.rad2deg(np.stack([s3, s4, s5], axis=-1)) + rot[:, None]
        shift = _wrap(samples - unfolded[None, :, :], 90.0)

        # each realization is rotated on the branch of the reference strike
        total_r, diff_r = _distortion_angles(x, base + shift[:, :, 2], rot)
        d_total = _wrap(total_r - total, 180.0)
        d_diff = _wrap(diff_r - diff, 180.0)

    def spread(a: np.ndarray) -> np.ndarray:
        return a.std(axis=0, ddof=1)

    distortion = {
        "angle_sum_deg": total,
        "angle_diff_deg": diff,
        "twist_deg": 0.5 * (total + diff),
        "shear_deg": 0.5 * (total - diff),
        "angle_sum_err_deg": spread(d_total),
        "angle_diff_err_deg": spread(d_diff),
        "twist_err_deg": spread(0.5 * (d_total + d_diff)),
        "shear_err_deg": spread(0.5 * (d_total - d_diff)),
    }
    return {
        "rho_a": rho,
        "rho_a_err": spread(rho_r),
        "phase_deg": phase,
        "phase_err_deg": spread(phase_r),
        "strikes_err_deg": spread(shift),
        "distortion": distortion,
        "analytic_errors": _analytic_errors(freq, xi, eta, efn, i1, i2),
    }


def _reported(
    cls: np.ndarray,
    strikes: np.ndarray,
    strikes_err: np.ndarray,
    distortion: Mapping[str, np.ndarray],
    *,
    waldim_compat: bool,
) -> Dict[str, np.ndarray]:
    """Strike, twist and shear belonging to the class of each frequency."""
    s3, s4, s5 = strikes.T
    if waldim_compat:
        pair = 0.5 * (s3 + s4)
    else:
        pair = np.mod(s3 + 0.5 * _wrap(s4 - s3, 90.0), 90.0)
    pair_err = 0.5 * np.hypot(strikes_err[:, 0], strikes_err[:, 1])

    from_pair = np.isin(cls, _PAIR_STRIKE)
    from_main = np.isin(cls, _MAIN_STRIKE)
    distorted = np.isin(cls, _DISTORTION)
    out = {
        "strike_deg": np.select(
            [from_pair, from_main], [pair, s5], np.nan),
        "strike_err_deg": np.select(
            [from_pair, from_main], [pair_err, strikes_err[:, 2]], np.nan),
    }
    for key in ("twist_deg", "twist_err_deg", "shear_deg", "shear_err_deg"):
        out[key] = np.where(distorted, distortion[key], np.nan)
    return out


# ---------------------------------------------------------------------------
# phase tensor
# ---------------------------------------------------------------------------
def _phase_tensor(zr: np.ndarray, zi: np.ndarray) -> np.ndarray:
    """Phase tensor X^-1 Y for real and imaginary parts (..., 2, 2)."""
    a, b = zr[..., 0, 0], zr[..., 0, 1]
    c, d = zr[..., 1, 0], zr[..., 1, 1]
    e, f = zi[..., 0, 0], zi[..., 0, 1]
    g, h = zi[..., 1, 0], zi[..., 1, 1]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        det = a * d - b * c
        rows = [
            np.stack([d * e - b * g, d * f - b * h], axis=-1),
            np.stack([a * g - c * e, a * h - c * f], axis=-1)]
        return np.stack(rows, axis=-2) / det[..., None, None]


def _pt_invariants(pt: np.ndarray, rot_deg: np.ndarray) -> Dict[str, Any]:
    """
    Parameters of phase tensors (..., 2, 2) in the convention of Caldwell et
    al. (2004). rot_deg must broadcast against pt[..., 0, 0].
    """
    p11, p12 = pt[..., 0, 0], pt[..., 0, 1]
    p21, p22 = pt[..., 1, 0], pt[..., 1, 1]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        beta = 0.5 * np.rad2deg(np.arctan2(p12 - p21, p11 + p22))
        alpha = 0.5 * np.rad2deg(np.arctan2(p12 + p21, p11 - p22))
        half_diff = 0.5 * np.hypot(p11 - p22, p12 + p21)
        half_sum = 0.5 * np.hypot(p11 + p22, p12 - p21)
        return {
            "pt_phimax_deg": np.rad2deg(np.arctan(half_sum + half_diff)),
            "pt_phimin_deg": np.rad2deg(np.arctan(half_sum - half_diff)),
            "pt_alpha_deg": _wrap(alpha + rot_deg, 180.0),
            "pt_beta_deg": beta,
            "pt_azimuth_deg": np.mod(alpha - beta + rot_deg, 180.0),
            "pt_ellipticity": half_diff / half_sum,
        }


def _ft_sign(
    zr: np.ndarray, zi: np.ndarray, ft_convention: str,
) -> Tuple[float, float]:
    """
    Sign to apply to the imaginary parts so that the impedance follows
    exp(+iwt), and the share of frequencies whose phase tensor trace agrees
    with it (NaN without a finite trace). With "auto" the sign is that of
    the majority of the traces: positive for exp(+iwt).
    """
    pt = _phase_tensor(zr, zi)
    with np.errstate(invalid="ignore", over="ignore"):
        trace = pt[:, 0, 0] + pt[:, 1, 1]
    trace = trace[np.isfinite(trace)]
    positive, negative = int(np.sum(trace > 0)), int(np.sum(trace < 0))
    if ft_convention == "auto":
        sign = -1.0 if negative > positive else 1.0
    else:
        sign = 1.0 if ft_convention == "e+iwt" else -1.0
    if not trace.size:
        return sign, float("nan")
    return sign, (positive if sign > 0 else negative) / trace.size


def _phase_tensor_parameters(
    zr: np.ndarray,
    zi: np.ndarray,
    var: np.ndarray,
    rot: np.ndarray,
    noise: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Phase tensor, its parameters and their standard deviations over noisy
    realizations. noise holds standard normal deviates (r, m, 2, 2, 2):
    every element, real and imaginary part.
    """
    pt = _phase_tensor(zr, zi)
    sd = np.sqrt(var)
    with np.errstate(invalid="ignore", over="ignore"):
        samples = _phase_tensor(
            zr[None] + noise[..., 0] * sd[None],
            zi[None] + noise[..., 1] * sd[None])
    value = _pt_invariants(pt, rot)
    spread = _pt_invariants(samples, rot[None, :])
    out: Dict[str, np.ndarray] = {"pt": pt}
    with np.errstate(invalid="ignore", over="ignore"):
        out["pt_err"] = samples.std(axis=0, ddof=1)
        for name, error in _PT_NAMES:
            diff = spread[name] - value[name]
            if name in _PT_PERIODIC:
                diff = _wrap(diff, 180.0)
            out[name] = value[name]
            out[error] = diff.std(axis=0, ddof=1)
    return out


def _skews(zr: np.ndarray, zi: np.ndarray) -> Dict[str, np.ndarray]:
    """Swift skew and Bahr phase-sensitive skew of tensors (..., 2, 2)."""
    z = zr + 1j * zi
    s1 = z[..., 0, 0] + z[..., 1, 1]
    s2 = z[..., 0, 1] + z[..., 1, 0]
    d1 = z[..., 0, 0] - z[..., 1, 1]
    d2 = z[..., 0, 1] - z[..., 1, 0]

    def commutator(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a.real * b.imag - a.imag * b.real

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return {
            "swift_skew": np.abs(s1) / np.abs(d2),
            "bahr_skew": np.sqrt(np.abs(
                commutator(d1, s2) - commutator(s1, d2))) / np.abs(d2),
        }


def _skew_parameters(
    zr: np.ndarray,
    zi: np.ndarray,
    var: np.ndarray,
    noise: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Skews and their standard deviations over noisy realizations."""
    sd = np.sqrt(var)
    value = _skews(zr, zi)
    with np.errstate(invalid="ignore", over="ignore"):
        spread = _skews(
            zr[None] + noise[..., 0] * sd[None],
            zi[None] + noise[..., 1] * sd[None])
        out: Dict[str, np.ndarray] = {}
        for name, error in _SKEW_NAMES:
            out[name] = value[name]
            out[error] = spread[name].std(axis=0, ddof=1)
    return out


def _pt_class(
    beta: np.ndarray,
    ellipticity: np.ndarray,
    beta_threshold: float,
    ellipticity_threshold: float,
) -> np.ndarray:
    """Phase tensor dimensionality (PT_CLASS_NAMES) of each frequency."""
    code = np.where(
        np.abs(beta) > beta_threshold, 3,
        np.where(ellipticity > ellipticity_threshold, 2, 1))
    known = np.isfinite(beta) & np.isfinite(ellipticity)
    return np.where(known, code, 0).astype(int)


def _count_pt_classes(code: np.ndarray) -> Dict[str, int]:
    """Number of cases of every phase tensor class."""
    return {name: int(np.sum(code == number))
            for number, name in PT_CLASS_NAMES.items()}


def _count_reasons(reason: np.ndarray) -> Dict[str, int]:
    """Number of cases of every reason code."""
    return {name: int(np.sum(reason == code))
            for code, name in REASON_NAMES.items()}


def _count_classes(cls: np.ndarray) -> Dict[str, int]:
    """Number of cases of every class."""
    return {name: int(np.sum(cls == code))
            for code, name in CLASS_NAMES.items()}


def _analyse_site(
    name: str,
    data: Mapping[str, Any],
    *,
    threshold: float,
    q_threshold: float,
    error_percent: Optional[float],
    missing_error_percent: Optional[float],
    n_realizations: int,
    seed: Optional[int],
    waldim_compat: bool,
    sweep: Optional[np.ndarray] = None,
    n_bootstrap: int = 0,
    pt_beta_threshold: float = PT_BETA_THRESHOLD,
    pt_ellipticity_threshold: float = PT_ELLIPTICITY_THRESHOLD,
    ft_convention: str = "auto",
) -> Dict[str, Any]:
    """Pre-process and analyse one site."""
    rng = np.random.default_rng(seed)
    site = _prepare_site(
        name, data, rng,
        error_percent=error_percent, waldim_compat=waldim_compat,
        missing_error_percent=missing_error_percent)
    noise = rng.standard_normal((n_realizations, site["freq"].size, 8))
    core = _classify(
        site["zr"], site["zi"], site["var"], site["rot"], noise,
        threshold=threshold, q_threshold=q_threshold,
        waldim_compat=waldim_compat, sweep=sweep)
    cls = core["class_id"]
    par = _parameters(
        site["freq"], _decompose(site["zr"]), _decompose(site["zi"]),
        _component_errors(site["var"]), site["rot"], core["invariants"],
        core["strikes_deg"], core["strikes_unfolded_deg"], noise,
        waldim_compat=waldim_compat)
    reported = _reported(
        cls, core["strikes_deg"], par["strikes_err_deg"], par["distortion"],
        waldim_compat=waldim_compat)

    result = {
        "station": name,
        "lat_deg": data.get("lat_deg", data.get("lat")),
        "lon_deg": data.get("lon_deg", data.get("lon")),
        "threshold": threshold,
        "q_threshold": q_threshold,
        "waldim_compat": waldim_compat,
        "freq": site["freq"],
        "period": 1.0 / site["freq"],
        "index": site["index"],
        "class_id": cls,
        "class_name": np.array(
            [CLASS_NAMES[int(c)] for c in cls], dtype="U12"),
        "counts": _count_classes(cls),
        "invariants": core["invariants"],
        "strikes_deg": core["strikes_deg"],
        "n_dropped": site["n_dropped"],
        "error_source": site["error_source"],
    }
    result.update(par)
    result.update(reported)

    reason = core["reason_id"]
    result["flags"] = core["flags"]
    result["reason_id"] = reason
    result["reason_name"] = np.array(
        [REASON_NAMES[int(c)] for c in reason], dtype="U12")
    result["reason_counts"] = _count_reasons(reason)

    pt_noise = rng.standard_normal(
        (n_realizations, site["freq"].size, 2, 2, 2))
    # the phase tensor is defined for impedance phases in the first quadrant
    # (exp(+iwt)); data with the opposite convention are conjugated here
    sign, share = _ft_sign(site["zr"], site["zi"], ft_convention)
    result["pt_ft_convention"] = "e+iwt" if sign > 0 else "e-iwt"
    result["pt_ft_agreement"] = share
    result.update(_phase_tensor_parameters(
        site["zr"], sign * site["zi"], site["var"], site["rot"], pt_noise))
    result.update(_skew_parameters(
        site["zr"], sign * site["zi"], site["var"], pt_noise))
    pt_class = _pt_class(
        result["pt_beta_deg"], result["pt_ellipticity"],
        pt_beta_threshold, pt_ellipticity_threshold)
    result["pt_beta_threshold_deg"] = pt_beta_threshold
    result["pt_ellipticity_threshold"] = pt_ellipticity_threshold
    result["pt_class"] = pt_class
    result["pt_class_name"] = np.array(
        [PT_CLASS_NAMES[int(c)] for c in pt_class], dtype="U12")
    result["pt_counts"] = _count_pt_classes(pt_class)

    if sweep is not None:
        table = core["class_by_threshold"]
        result["threshold_values"] = np.array(sweep)
        result["class_by_threshold"] = table
        result["threshold_agreement"] = np.mean(table == cls[:, None], axis=1)

    if n_bootstrap:
        # a separate generator: switching the bootstrap on changes nothing else
        boot_rng = np.random.default_rng(
            None if seed is None else [seed, 1])
        sd = np.sqrt(site["var"])
        draws = np.empty((cls.size, n_bootstrap), dtype=int)
        for b in range(n_bootstrap):
            draws[:, b] = _classify(
                site["zr"] + boot_rng.standard_normal(sd.shape) * sd,
                site["zi"] + boot_rng.standard_normal(sd.shape) * sd,
                site["var"], site["rot"], noise,
                threshold=threshold, q_threshold=q_threshold,
                waldim_compat=waldim_compat)["class_id"]
        prob = np.stack(
            [np.mean(draws == code, axis=1) for code in CLASS_NAMES], axis=1)
        result["n_bootstrap"] = n_bootstrap
        result["bootstrap_class_id"] = draws
        result["bootstrap_class_prob"] = prob
        result["bootstrap_agreement"] = prob[np.arange(cls.size), cls]
    return result


# ---------------------------------------------------------------------------
# period / frequency bands
# ---------------------------------------------------------------------------
def _band_edges(
    periods: np.ndarray,
    bands_per_decade: float,
    band_edges: Optional[Sequence[float]],
    band_domain: str,
) -> np.ndarray:
    """Ascending band edges in period (s)."""
    if band_edges is not None:
        edges = np.asarray(band_edges, dtype=float).ravel()
        if not np.all(np.isfinite(edges) & (edges > 0.0)):
            raise ValueError("band_edges must be positive numbers")
        if band_domain == "frequency":
            edges = 1.0 / edges
        edges = np.unique(edges)
        if edges.size < 2:
            raise ValueError("band_edges needs at least two distinct values")
        return edges
    if periods.size == 0:
        return np.empty(0)
    lo = np.log10(periods.min()) * bands_per_decade
    hi = np.log10(periods.max()) * bands_per_decade
    first = int(np.ceil(lo - _EDGE_TOL)) - 1
    last = int(np.ceil(hi - _EDGE_TOL))
    return 10.0 ** (np.arange(first, last + 1) / bands_per_decade)


def _band_class(cls: np.ndarray) -> int:
    """
    Representative class of one band, following WALDIM: class 3 counts as 4;
    the most frequent class wins and ties go to the lowest code; if that is
    3D, any other class holding at least half of the classified frequencies
    replaces it. Undetermined frequencies are ignored.
    """
    if cls.size == 0:
        return NO_DATA
    merged = np.where(cls == 3, 4, cls)
    merged = merged[merged != 0]
    if merged.size == 0:
        return 0
    tally = np.bincount(merged, minlength=len(CLASS_NAMES))
    best = int(np.argmax(tally))
    if best == 5:
        for code in (1, 2, 4, 6, 7, 8, 9):
            if tally[code] > 0 and tally[code] >= merged.size / 2.0:
                return code
    return best


def _angle_mean(angles: np.ndarray, period: float = 90.0) -> float:
    """Mean of angles with the given periodicity (deg), in [0, period)."""
    harmonic = 360.0 / period
    phasor = np.sum(np.exp(1j * harmonic * np.deg2rad(angles)))
    if not np.abs(phasor) > 1.0e-9 * angles.size:
        return float("nan")
    centre = np.rad2deg(np.angle(phasor)) / harmonic
    return float(np.mod(
        centre + np.mean(_wrap(angles - centre, period)), period))


def _scatter_error(values: np.ndarray, period: Optional[float]) -> float:
    """
    Standard error of a mean estimated from the scatter of the values; NaN
    for fewer than two values. period: periodicity of angles, else None.
    """
    n = values.size
    if n < 2:
        return float("nan")
    with np.errstate(invalid="ignore", over="ignore"):
        if period is None:
            dev = values - np.mean(values)
        else:
            dev = _wrap(values - _angle_mean(values, period), period)
        return float(np.sqrt(np.sum(dev ** 2) / (n * (n - 1))))


def _set_band_error(
    out: Dict[str, Any], name: str, prop: float, scatter: float,
    *, use_scatter: bool = True,
) -> None:
    """Store propagated, scatter and total error of a band average."""
    total, from_prop, from_scatter = _error_names(name)
    out[from_prop] = float(prop)
    out[from_scatter] = float(scatter)
    out[total] = float(np.fmax(prop, scatter) if use_scatter else prop)


def _band_pt(members: np.ndarray, result: Mapping[str, Any]) -> Dict[str, Any]:
    """Averages of the phase tensor parameters over all rows of one band."""
    out: Dict[str, Any] = {}
    for name, period in _PT_BAND:
        values = np.asarray(result[name])[members]
        errors = np.asarray(result[_error_names(name)[0]])[members]
        out[name] = prop = scatter = float("nan")
        if values.size:
            with np.errstate(invalid="ignore", over="ignore"):
                if period is None:
                    out[name] = float(np.mean(values))
                else:
                    out[name] = _angle_mean(values, period)
                prop = float(np.sqrt(np.sum(errors ** 2)) / values.size)
            if name == "pt_beta_deg":
                out[name] = float(_wrap(out[name], 180.0))
            scatter = _scatter_error(values, period)
        _set_band_error(out, name, prop, scatter)
    return out


def _band_parameters(
    code: int,
    members: np.ndarray,
    result: Mapping[str, Any],
    *,
    waldim_compat: bool,
) -> Dict[str, Any]:
    """
    Average strike, twist and shear over the frequencies of one band that have
    the band class. members holds the (consecutive) row numbers of the band.
    """
    out: Dict[str, Any] = {"n_averaged": 0}
    for name in ("strike_deg", "twist_deg", "shear_deg"):
        for key in (name,) + _error_names(name):
            out[key] = float("nan")
    if code <= 0:
        return out
    cls = result["class_id"]
    merged = np.where(cls == 3, 4, cls)
    rows = members[merged[members] == code]
    count = rows.size
    out["n_averaged"] = int(count)
    if code not in (2, 4, 8, 9):
        return out

    strikes = result["strikes_deg"]
    errors = result["strikes_err_deg"]
    if code in _PAIR_STRIKE:
        s3, s4 = strikes[rows, 0], strikes[rows, 1]
        if waldim_compat:
            # the source mirrors (90 - x) one of two strikes that lie at
            # opposite ends of the quadrant
            far = np.abs(s3 - s4) > 90.0 - _STRIKE_TOL
            s4_new = np.where(far & (s4 < s3), 90.0 - s4, s4)
            s3 = np.where(far & ~(s4 < s3), 90.0 - s3, s3)
            out["strike_deg"] = float(np.mean(0.5 * (s3 + s4_new)))
        else:
            pair = np.mod(s3 + 0.5 * _wrap(s4 - s3, 90.0), 90.0)
            out["strike_deg"] = _angle_mean(pair)
        total = np.sum(errors[rows, 0] ** 2 + errors[rows, 1] ** 2) / 4.0
        s3, s4 = strikes[rows, 0], strikes[rows, 1]
        pair = np.mod(s3 + 0.5 * _wrap(s4 - s3, 90.0), 90.0)
        _set_band_error(
            out, "strike_deg", np.sqrt(total) / count,
            _scatter_error(pair, 90.0), use_scatter=not waldim_compat)
        return out

    s5 = strikes[:, 2]
    if waldim_compat:
        # the source mirrors a strike if its predecessor in the band lies at
        # the other end of the quadrant
        work = np.array(s5)
        for k in rows:
            if k != members[0] and (
                    (work[k - 1] >= 80.0 and work[k] <= 10.0)
                    or (work[k - 1] <= 10.0 and work[k] >= 80.0)):
                work[k] = 90.0 - work[k]
        mean = float(np.mean(work[rows]))
    else:
        mean = _angle_mean(s5[rows])
    out["strike_deg"] = mean
    _set_band_error(
        out, "strike_deg", np.sqrt(np.sum(errors[rows, 2] ** 2)) / count,
        _scatter_error(s5[rows], 90.0), use_scatter=not waldim_compat)

    if code == 4:
        dist = result["distortion"]
        shear = dist["shear_deg"][rows]
        if not waldim_compat:
            # the shear angle changes sign with a 90 degree strike shift
            moved = mean + _wrap(s5[rows] - mean, 90.0) - s5[rows]
            shear = np.where(np.abs(moved) > 45.0, -shear, shear)
            if np.isnan(mean):
                shear = np.full(count, np.nan)
        for name, values in (("twist", dist["twist_deg"][rows]),
                             ("shear", shear)):
            out[name + "_deg"] = float(np.mean(values))
            _set_band_error(
                out, name + "_deg",
                np.sqrt(np.sum(dist[name + "_err_deg"][rows] ** 2)) / count,
                _scatter_error(values, None), use_scatter=not waldim_compat)
    return out


def summarise_band(
    result: Mapping[str, Any],
    members: Any,
    *,
    waldim_compat: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Summarise one site over a set of its frequencies, for example a band.

    Parameters
    ----------
    result : mapping
        One value of the dict returned by classify_dimensionality.
    members : sequence of int
        Row numbers (positions in result["period"]) of the band; may be
        empty. With waldim_compat they should be consecutive, as the rows of
        a period band are.
    waldim_compat : bool, optional
        Rules of the strike averages; default: as used for the site.

    Returns
    -------
    dict
        n_periods, n_classified, class_id (WALDIM band class, -1 without
        data), class_name, counts, data_period_min, data_period_max;
        n_averaged and the averages strike_deg, twist_deg, shear_deg over
        the rows of the band class; pt_phimax_deg, pt_phimin_deg,
        pt_beta_deg, pt_azimuth_deg, pt_ellipticity: means over all rows.
        Every average has _err (the larger of the next two), _err_prop
        (propagated from the data errors) and _err_scatter (standard error
        from the scatter in the band, NaN for fewer than two rows); with
        waldim_compat the strike, twist and shear errors are the propagated
        ones. Medians over all rows, without errors: pt_abs_beta_deg
        (|beta|), pt_ellipticity_median, swift_skew, bahr_skew. pt_class (the
        most frequent phase tensor class, the higher one in a tie, -1
        without data), pt_counts, reason_counts and, where the site has them,
        class_by_threshold, threshold_agreement, bootstrap_class_prob,
        bootstrap_agreement.
    """
    if waldim_compat is None:
        waldim_compat = bool(result["waldim_compat"])
    members = np.sort(np.asarray(members, dtype=int).ravel())
    period, cls = result["period"], result["class_id"]
    sel = cls[members]
    code = _band_class(sel)
    band = {
        "n_periods": int(sel.size),
        "n_classified": int(np.sum(sel != 0)),
        "class_id": code,
        "class_name": CLASS_NAMES.get(code, NO_DATA_NAME),
        "counts": _count_classes(sel),
        "data_period_min": float(period[members].min())
        if members.size else float("nan"),
        "data_period_max": float(period[members].max())
        if members.size else float("nan"),
    }
    band.update(_band_parameters(
        code, members, result, waldim_compat=waldim_compat))
    band.update(_band_pt(members, result))
    for name, source in _BAND_MEDIANS:
        values = np.asarray(result[source])[members]
        if name == "pt_abs_beta_deg":
            values = np.abs(values)
        values = values[np.isfinite(values)]
        band[name] = float(np.median(values)) if values.size else float("nan")
    pt_class = np.asarray(result["pt_class"])[members]
    number = np.bincount(pt_class, minlength=len(PT_CLASS_NAMES))
    if not members.size:
        band["pt_class"] = NO_DATA
    elif not number[1:].any():
        band["pt_class"] = 0
    else:   # the last of the largest counts: the higher class in a tie
        band["pt_class"] = int(
            number.size - 1 - np.argmax(number[:0:-1]))
    band["pt_counts"] = _count_pt_classes(pt_class)
    band["reason_counts"] = _count_reasons(result["reason_id"][members])
    if "class_by_threshold" in result:
        table = result["class_by_threshold"][members]
        by_th = np.array(
            [_band_class(column) for column in table.T], dtype=int)
        band["class_by_threshold"] = by_th
        band["threshold_agreement"] = float(
            np.mean(by_th == code)) if members.size else float("nan")
    if "bootstrap_class_id" in result:
        draws = result["bootstrap_class_id"][members]
        prob = np.zeros(len(CLASS_NAMES))
        if members.size:
            codes = [_band_class(column) for column in draws.T]
            prob = np.bincount(codes, minlength=prob.size) / len(codes)
        band["bootstrap_class_prob"] = prob
        band["bootstrap_agreement"] = float(
            prob[code]) if members.size else float("nan")
    return band


def _band_results(
    result: Mapping[str, Any],
    edges: np.ndarray,
    *,
    waldim_compat: bool,
) -> List[Dict[str, Any]]:
    """Summary of one site in the bands (edge[i], edge[i+1]]."""
    if edges.size < 2:
        return []
    which = np.searchsorted(
        edges * (1.0 + _EDGE_TOL), result["period"], side="left") - 1
    out: List[Dict[str, Any]] = []
    for number, (tmin, tmax) in enumerate(zip(edges[:-1], edges[1:])):
        band = {
            "period_min": float(tmin),
            "period_max": float(tmax),
            "freq_min": float(1.0 / tmax),
            "freq_max": float(1.0 / tmin),
        }
        band.update(summarise_band(
            result, np.flatnonzero(which == number),
            waldim_compat=waldim_compat))
        out.append(band)
    return out
