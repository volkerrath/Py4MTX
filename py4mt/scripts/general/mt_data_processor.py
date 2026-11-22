#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-process MT station EDI files.

Reads EDI files, computes derived quantities, optionally interpolates /
sets errors / rotates, then exports in multiple formats (EDI, NPZ, HDF, MAT,
and, optionally, one FEMTIC observe.dat for all stations).
A collection NPZ file with all stations is saved at the end.

@author:    Volker Rath (DIAS)
@project:   py4mt — Python for Magnetotellurics
@created:   2026-02-13 with the help of ChatGPT (GPT-5 Thinking)
@modified:  2026-03-16 — freq_order, D+/rho+ test (DPLUS), add_rhoplus plot; Claude Sonnet 4.6 (Anthropic)
@modified:  2026-03-18 — add_noise option in SET_ERRORS (mode='fix' only); Claude Sonnet 4.6 (Anthropic)
@modified:  2026-03-25 — FT_CORRECTION block (manufacturer, from/to convention); Claude Sonnet 4.6 (Anthropic)
@modified:  2026-03-30 — adapt plot block to updated data_viz: per-call ylim overrides (no PLTARGS mutation), DPLUS subplot row, remove bogus required-key guard from add_tipper/add_pt; Claude Sonnet 4.6 (Anthropic)
@modified:  2026-06-12 — correct_ft_convention imported from data_proc (absorbed from mt_ft_convention); Claude Sonnet 4.6 (Anthropic)
@modified:  2026-06-13 — compute_zavg added to INVARS block (Zavg, Zavg_err); Claude Sonnet 4.6 (Anthropic)
@modified:  2026-11-11 -- FIGURE_TITLE: title of the station figures ("auto" = station | name as before; None / "" = none; text with {station}, {name_str}; dict per station); Claude Sonnet 5.5 (Anthropic), AI-generated, review before use
@modified:  2026-11-20 -- (1) INTERPOLATE: one frequency grid for all stations (freq_per_dec points per decade on absolute decades, band = common band of all stations or an explicit range); the interpolation now runs first, so PT, invariants, errors and rho/phi live on the new grid (they were left on the old one); the DPLUS block reads Z/Z_err from data_dict. (2) MANUFACTURER = "auto": FT convention (exp(-iwt) / exp(+iwt)) detected per station from the data (data_proc.detect_ft_convention) with a summary table; the notes on FT_CORRECTION corrected (load_edi has already converted Phoenix data). (3) OUT_FILES token "observe": one FEMTIC observe.dat for all stations, optionally with the dummy VTF stations (femtic.edi_list_to_observe_dat). (4) PLOT_DIR is only created when PLOT is set; EDI list sorted. Claude Sonnet 5.5 (Anthropic), AI-generated, review before use
@modified:  2026-11-20 (b) -- INTERPOLATE: the grid is preset for the whole survey (freq_range "survey" = band of all stations, default; "common" = intersection; or (fmin, fmax)) and every station is written on the part of it that its own band covers; extrap_decades (1/x decade) allows extrapolation just outside the band (flag data_dict["extrapolated"]); stations with no grid frequency are skipped; the old "identical frequencies" check is replaced by a coverage report and a <COLL_NAME>_freq_coverage.csv; allow_extrapolation removed. Claude Sonnet 5.5 (Anthropic), AI-generated, review before use
@modified:  2026-11-20 (c) -- INTERPOLATE: new interp_pars key n_freq = exactly N frequencies, log-spaced between the limits of freq_range ('survey', 'common' or (fmin, fmax)) with both limits as grid points (data_proc.make_freq_grid_n); freq_per_dec and n_freq are alternatives (exactly one must be set); n_freq is popped before interpolate_data. Claude Sonnet 5.5 (Anthropic), AI-generated, review before use
@modified:  2026-11-20 (c) -- observe.dat: OBS_INVALID = "drop" | "mark" (and OBS_INVALID_SE) chooses how invalid data are written: the frequency is left out ("drop", as before) or only the bad component is written with a negative error, which FEMTIC does not use ("mark"); the per-station counts are printed. Claude Sonnet 5.5 (Anthropic), AI-generated, review before use
@modified:  2026-11-20 (d) -- phase panels: where a phase curve leaves the +-180 deg window the solid line is interrupted and the jump is drawn as a dotted connection; PHASE_WRAP_MODE ('direct', 'short', 'off') and PHASE_WRAP_STYLE (dict of Line2D keywords, False = no connection) in the plot configuration, passed to data_viz.add_phase; Claude Sonnet 5.5 (Anthropic); AI-generated, review before use
"""

import os
import sys
from pathlib import Path
import inspect

import numpy as np
import matplotlib.pyplot as plt

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

for _base in [PY4MTX_ROOT + "/py4mt/modules/"]:
    for _p in [Path(_base), *Path(_base).rglob("*")]:
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

import util as utl
from data_viz import add_phase, add_rho, add_rhoplus, add_tipper, add_pt
from data_proc import (
    get_edi_list,
    load_edi, save_edi, save_ncd, save_hdf, save_npz, save_mat,
    save_list_of_dicts_npz, dataframe_from_edi,
    interpolate_data, common_freq_grid, set_errors, estimate_errors, rotate_data,
    compute_pt, compute_zdet, compute_zssq, compute_zavg, compute_rhophas, compute_rhoplus,
    correct_ft_convention,
)

from version import versionstrg

rng = np.random.default_rng()
nan = np.nan
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())
titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng + "\n\n")

# =============================================================================
#  Configuration
# WORK_DIR = PY4MTX_ROOT + "py4mt/data/edi/"
# WORK_DIR = "/home/vrath/Py4MTX/py4mt/data/rto/ubinas/edi/"
# WORK_DIR = "/home/vrath/MT_Data/waldim/edi_synth_iso/"
###WORK_DIR = "/home/vrath/MT_Data/Peru/Tacna/NEW/edis_26b/"
# WORK_DIR = "/home/vrath/MT_Data/TEST/edis_exp_interp/"
# WORK_DIR = "/home/vrath/ACTIVE/mt_processor/2022_25/"
WORK_DIR = "/home/vrath/ACTIVE/mt_processor/edi_files/"
if not os.path.isdir(WORK_DIR):
   sys.exit(" File: %s does not exist! EXit." % WORK_DIR)


DATA_DIR_IN = WORK_DIR
DATA_DIR_OUT = WORK_DIR + "/proc/"

NAME_STR = ""
COLL_NAME = "Test"

# Tokens (comma separated): edi, npz, hdf, mat, ncd, observe
#   "observe" = one FEMTIC observe.dat for all stations (see the OBSERVE block)
OUT_FILES = "edi, npz, observe"
STAT_FILE = True
STAT_UPPER = True


DROP_INVALID = True
FREQ_ORDER = "dec"   # "inc", "dec", or "keep"

MANUFACTURER = "auto"   # "metronix", "phoenix", "delta", or "auto"
                             # Controls FT sign-convention correction at load time.
                             # Phoenix (e+iwt) -> Im(Z) and Im(T) are negated so all
                             # data in the loop are in the standard e-iwt convention.
                             # "auto": the convention is detected for every station
                             # from the data itself (sign of the phase quadrant of
                             # Zdet, Zxy, Zyx; an explicit exp(+-iwt) tag in the EDI
                             # is used if the data are inconclusive) - for EDIs
                             # whose writer is unknown. A summary table is printed.
if MANUFACTURER.lower() == "auto":
    FT_AUTO_FALLBACK = "metronix"   # assumed if the detection is inconclusive
                                    # ("metronix"/"delta" = e-iwt, "phoenix" = e+iwt)
    FT_DETECT_PARS = {
        "margin_deg": 5.0,    # ignore phases this close to a quadrant boundary
        "min_votes":  6,      # minimum number of votes
        "decisive":   0.75,   # share of the votes the winner needs
    }

FT_CORRECTION = True
if FT_CORRECTION:
    # NOTE: load_edi has already converted Phoenix data (and "auto" results)
    # to e-iwt, so the stored data are e-iwt after loading.  Keep
    # from_convention = "e-iwt" unless you loaded with manufacturer "metronix" /
    # "delta" AND know that the files are e+iwt; with MANUFACTURER = "phoenix"
    # (or "auto") and from_convention = "e+iwt" the data would be flipped twice.
    # from_convention: FT convention of the data in the loop after loading.
    #   "e-iwt" — standard / Metronix / DELTA (no correction applied)
    #   "e+iwt" - Phoenix MTU convention, data NOT yet converted
    # to_convention: target convention stored in data_dict and written to output EDI.
    #   Almost always "e-iwt" (standard).  Set "e+iwt" only if the downstream
    #   inversion code expects Phoenix convention.
    ftc_pars = {
        "from_convention": "e-iwt",
        "to_convention":   "e-iwt",
    }


PHAS_TENS = True
RHOPHAS = True
INVARS = True
DPLUS = False   # D+/rho+ test (Parker 1980; also known as dplus in Cordell mtcode)

ROTATE = False
if ROTATE:
    ANGLE = 0.0

DEC_DEG = True

SET_ERRORS = True
if SET_ERRORS:
    # Passed as **err_pars to set_errors(data_dict, mode=..., **err_pars)
    # Z_rel    : [xx, xy, yx, yy] relative levels
    # Z_rel_mode: "ij"  → σ_ij = Z_rel_ij * |Z_ij(ω)|
    #             "ij*ii" → σ_ij = Z_rel_ij * sqrt(|Z_ii|*|Z_ij|) off-diag
    # T_abs    : [Tx, Ty] absolute (constant) tipper error
    # PT_abs   : [xx, xy, yx, yy] absolute (constant) phase-tensor error
    # add_noise: if True, perturb data by N(0, σ) — only valid with mode='fix'
    ADD_NOISE = False
    err_pars = {
        "Z_rel":      [0.1, 0.1, 0.1, 0.1],
        "Z_rel_mode": "ij",          # "ij" or "ij*ii"
        "T_abs":      [0.03, 0.03],
        "PT_abs":     [0.1, 0.1, 0.1, 0.1],
        "mode":       "fix",         # "floor"
        "add_noise":  ADD_NOISE,
        "random_state": rng,         # module-level Generator; set seed for reproducibility
    }


INTERPOLATE = True
if INTERPOLATE:
    # ONE preset frequency grid for the whole survey (inversions are parallelised
    # per frequency).  Two ways to define it - set ONE of them, the other None:
    #   freq_per_dec = x   x points per decade on absolute decades,
    #                      f = anchor * 10**(k / x), k integer; the number of
    #                      frequencies depends on the band (data_proc.make_freq_grid)
    #   n_freq = N         exactly N frequencies, equally spaced in log f, with
    #                      BOTH limits of the band as grid points (anchor is not
    #                      used; data_proc.make_freq_grid_n).  The band is the one
    #                      given by freq_range below, e.g. (0.01, 100.) with
    #                      n_freq = 25 gives 25 frequencies from 0.01 to 100 Hz.
    # EVERY station is written on the part of this grid that its own band covers
    # (no frequency is made up outside the band, except within extrap_decades);
    # stations with the same band get the same frequencies, and every frequency
    # of every station is a member of the grid.
    #   freq_range = "survey"        band of the whole survey (lowest minimum to
    #                                highest maximum): no station limits the grid
    #              = "common"        only the band that ALL stations cover
    #              = (fmin, fmax)    explicit band [Hz]
    # With n_freq the limits are the first and last frequency, so with "survey"
    # the grid starts at the lowest and ends at the highest frequency of any
    # station (not rounded to nice values); give (fmin, fmax) for round limits.
    # Alternatively give "newfreqs" (explicit array [Hz]); it is used as it is.
    # The order of the grid follows FREQ_ORDER ("dec" -> descending).
    #   extrap_decades: grid frequencies at most this far OUTSIDE a station's band
    #       are extrapolated (straight line in log f from the value and slope at
    #       the band edge; errors keep their edge value; flagged in
    #       data_dict["extrapolated"]).  Give 1/x for "less than 1/x of a decade",
    #       e.g. 1.0/6.0 = one grid step at 6 per decade; 0.0 = no extrapolation.
    interp_pars = {
        "freq_per_dec":   4,           # points per decade (None if n_freq is used)
        "n_freq":         None,   #20,        # exactly this many frequencies (None if freq_per_dec is used)
        "freq_range":     "survey",
        "anchor":         1.0,         # Hz; a grid point (..., 0.1, 1, 10, ...); not used with n_freq
        "interp_method":  "gcvspline",
        "extrap_decades": 0.1,
    }

ESTIMATE_ERRORS = False
if ESTIMATE_ERRORS:
    sys.exit("Work in progress! Exit")
    SPREAD = 2.0  # *std-dev
    ERR_METHOD = ["gcvspline", SPREAD]

# --- FEMTIC observe.dat (token "observe" in OUT_FILES) -----------------------
OBSERVE = "observe" in OUT_FILES.lower()
if OBSERVE:
    OBSERVE_FILE = DATA_DIR_OUT + "observe.dat"
    # Positions: site.dat (CSV of mt_make_sitelist.py: name,lat,lon,elev,sitenum,
    # easting,northing). Names are matched with the EDI file names (case-insensitive).
    SITE_DAT = DATA_DIR_IN + "site.dat"
    # Model origin (UTM): "box" = middle of the bounding box of the sites in
    # site.dat (the canonical choice), "average" = mean, None = OBS_ORIGIN_E/N.
    # It MUST be the origin of the mesh the observe.dat will be used with.
    OBS_ORIGIN_METHOD = "box"
    OBS_ORIGIN_E = None            # UTM easting of the model origin [m]  (if method None)
    OBS_ORIGIN_N = None            # UTM northing of the model origin [m] (if method None)
    OBS_UTM_ZONE_OVERRIDE = None
    # Dummy VTF stations (FEMTIC manual slides 55-58; control.dat: OWNER_ELEMENT -1):
    # every MT station gets H-ID = ID + offset, and a dummy VTF station (0 frequencies)
    # at the same place carries the owner element of the magnetic field.
    OBS_DUMMY_VTF = True
    OBS_SITE_TYPE = "land"         # "land" or "seafloor" (decides the owner types)
    OBS_DUMMY_ID_OFFSET = 1000     # raised to the next power of ten if < number of sites
    # Invalid data (a non-finite value or error in a component of the impedance).
    # FEMTIC has no marker value: a datum is used only if its error is > 0
    # (manual: "If SE is negative, the corresponding component is not used").
    #   "drop" : a frequency with a non-finite value/error is left out of the file
    #            (errors that are exactly 0 are written as 0 and are not used either)
    #   "mark" : the bad component stays in the file with the error OBS_INVALID_SE
    #            (value 0.0 if it was not finite), so that the valid components of
    #            that frequency are kept; components with an error <= 0 are marked too
    #   NOTE: invalid values only reach this stage with DROP_INVALID = False (otherwise the
    #   periods are removed when the EDI is read); components that were invalid at all
    #   frequencies (zeroed on reading, error 0) are handled in both cases.
    OBS_INVALID = "drop"
    OBS_INVALID_SE = -1.0          # must be negative

PLOT = True
if PLOT:
    PLOT_DIR = WORK_DIR +"/plots/"
    # Figure title: "auto" = "<station><NAME_STR with | for _>" (as before);
    # None, False or "" = no title; a string with the placeholders {station},
    # {name_str} (other braces are used as they stand); or a dict
    # {"<station>": spec, "default": spec}
    FIGURE_TITLE = "auto"
    PLTARGS = {
        "show_errors": True,
        "xlim": None,       # period limits (s), e.g. (1e-3, 1e3); None = auto
        "ylim": None,       # y-axis limits; None = auto
    }
    # Phase panels: where a phase curve leaves the +-180 deg window and
    # re-enters on the other side, the solid line is interrupted and the jump
    # is drawn in PHASE_WRAP_STYLE instead (only for these two; they are
    # passed to add_phase alone, not through PLTARGS, which also goes to add_rho).
    #   PHASE_WRAP_MODE  "direct" one line from point to point
    #                    "short"  two short stubs through the window edge
    #                    "off"    solid line through the jump (old behaviour)
    PHASE_WRAP_MODE = "direct"
    #   PHASE_WRAP_STYLE dict of matplotlib Line2D keywords, on top of the
    #                    defaults (dotted, colour/width of the curve), e.g.
    #                    {"linestyle": (0, (1, 3)), "alpha": 0.5, "color": "0.4"};
    #                    False = no connection at all
    PHASE_WRAP_STYLE = {"linestyle": ":"}
    PLOT_FORMAT = [".pdf"]

    if not os.path.isdir(PLOT_DIR):
        print(" File: %s does not exist, but will be created" % PLOT_DIR)
        os.mkdir(PLOT_DIR)
if not os.path.isdir(DATA_DIR_OUT):
    print(" File: %s does not exist, but will be created" % DATA_DIR_OUT)
    os.mkdir(DATA_DIR_OUT)


edi_files = get_edi_list(DATA_DIR_IN, sort=True, fullpath=True)
ns = np.size(edi_files)

# =============================================================================
#  Stage 1: load all stations (FT convention is settled here)
# =============================================================================
loaded = []
for edi in edi_files:
    print("\n\nFound edi file: ", edi)

    _load_kw = {}
    if MANUFACTURER.lower() == "auto":
        _load_kw = dict(auto_fallback=FT_AUTO_FALLBACK, ft_detect_pars=FT_DETECT_PARS)

    data_dict = load_edi(
        edi,
        drop_invalid_periods=DROP_INVALID,
        freq_order=FREQ_ORDER,
        manufacturer=MANUFACTURER,
        **_load_kw,
    )

    if FT_CORRECTION:
        correct_ft_convention(data_dict, **ftc_pars)

    loaded.append((edi, data_dict))

if MANUFACTURER.lower() == "auto":
    print("\n" + "=" * 78)
    print(" FT convention detected per station (data are converted to e-iwt)")
    print("-" * 78)
    print(" %-20s %-10s %9s %7s  %-18s" % ("station", "detected", "fraction", "votes", "source"))
    _cnt = {}
    for _edi, _d in loaded:
        _det = _d.get("ft_detected", "?")
        _cnt[_det] = _cnt.get(_det, 0) + 1
        print(" %-20s %-10s %9.2f %7d  %-18s" % (
            Path(_edi).stem[:20], _det, _d.get("ft_detect_fraction", np.nan),
            _d.get("ft_detect_nvotes", 0), _d.get("ft_detect_source", "")))
    print("-" * 78)
    print(" summary: " + ", ".join("%s: %d" % (k, v) for k, v in sorted(_cnt.items())))
    if len(_cnt) > 1:
        print(" NOTE: the stations differ in convention (mixed processing?), "
              "each one was converted on its own.")
    print("=" * 78 + "\n")

# =============================================================================
#  Stage 2: one frequency grid for all stations
# =============================================================================
interp_site = None
grid = None
if INTERPOLATE and not DROP_INVALID:
    print("\n WARNING: INTERPOLATE with DROP_INVALID = False: invalid values (1e32 markers) "
          "would go into the spline and spread to their neighbours. Use DROP_INVALID = True.")
if INTERPOLATE:
    interp_site = dict(interp_pars)
    _fpd = interp_site.pop("freq_per_dec", None)
    _nfreq = interp_site.pop("n_freq", None)
    _range = interp_site.pop("freq_range", "survey")
    _anchor = interp_site.pop("anchor", 1.0)
    _extrap = float(interp_site.get("extrap_decades", 0.0) or 0.0)
    if interp_site.get("newfreqs") is None:
        interp_site.pop("newfreqs", None)
        if _fpd is None and _nfreq is None:
            sys.exit(" INTERPOLATE needs freq_per_dec, n_freq or newfreqs in interp_pars! Exit.")
        if _fpd is not None and _nfreq is not None:
            sys.exit(" INTERPOLATE: give freq_per_dec OR n_freq in interp_pars, not both "
                     "(set the other to None)! Exit.")
        print("\n One preset frequency grid for all %d stations:" % len(loaded))
        interp_site["newfreqs"] = common_freq_grid(
            [d for _, d in loaded], _fpd, n_freq=_nfreq, freq_range=_range, anchor=_anchor,
            order=FREQ_ORDER if FREQ_ORDER in ("inc", "dec") else "inc",
            extrap_decades=_extrap,
            names=[Path(e).stem for e, _ in loaded], verbose=True)
    grid = np.asarray(interp_site["newfreqs"], dtype=float)

# =============================================================================
#  Stage 3: processing loop
# =============================================================================
all_data = []
all_stems = []
skipped = []
for edi, data_dict in loaded:
    print("\n\nProcessing edi file: ", edi)

    # Interpolate first: everything derived below (PT, invariants, errors,
    # rho/phi) is then computed on the common grid.
    if INTERPOLATE:
        try:
            data_dict = interpolate_data(data_dict, **interp_site)
        except ValueError as _err:
            print(" WARNING: station skipped, no grid frequency in its band: %s" % _err)
            skipped.append(Path(edi).stem)
            continue
        _nx = int(np.sum(data_dict["extrapolated"]))
        print(" %d of %d grid frequencies in the band of this station%s"
              % (data_dict["freq"].size, grid.size,
                 " (%d extrapolated)" % _nx if _nx else ""))

    station = data_dict["station"]
    Z = data_dict["Z"]
    Zerr = data_dict["Z_err"]
    T = data_dict["T"]
    Terr = data_dict["T_err"]

    # --- Task block ---

    if PHAS_TENS:
        P, Perr = compute_pt(Z, Zerr)
        data_dict["P"] = P
        data_dict["P_err"] = Perr

    if INVARS:
        Zdet, Zdeterr = compute_zdet(Z, Zerr)
        data_dict["Zdet"] = Zdet
        data_dict["Zdet_err"] = Zdeterr
        Zssq, Zssqerr = compute_zssq(Z, Zerr)
        data_dict["Zssq"] = Zssq
        data_dict["Zssq_err"] = Zssqerr
        Zavg, Zavgerr = compute_zavg(Z, Zerr)
        data_dict["Zavg"] = Zavg
        data_dict["Zavg_err"] = Zavgerr


    if ESTIMATE_ERRORS:
        data_dict = estimate_errors(data_dict=data_dict, method=ERR_METHOD)

    if SET_ERRORS:
        data_dict = set_errors(data_dict, **err_pars)

    if ROTATE:
        data_dict = rotate_data(data_dict, angle=ANGLE)

    # --- Refresh apparent resistivity/phase after any Z modification ---
    if RHOPHAS:
        if data_dict.get("freq") is not None and data_dict.get("Z") is not None:
            _ek = str(data_dict.get("err_kind", "var")).strip().lower()
            _ek = "std" if _ek.startswith("std") else "var"
            rho, phi, rho_err, phi_err = compute_rhophas(
                freq=np.asarray(data_dict["freq"]),
                Z=np.asarray(data_dict["Z"]),
                Z_err=(
                    np.asarray(data_dict["Z_err"])
                    if data_dict.get("Z_err") is not None
                    else None
                ),
                err_kind=_ek,
                err_method="analytic",
            )
            data_dict["rho"] = rho
            data_dict["phi"] = phi
            if rho_err is not None:
                data_dict["rho_err"] = rho_err
            if phi_err is not None:
                data_dict["phi_err"] = phi_err

    if DPLUS:
        # D+/rho+ test on Zxy, Zyx, and Zdet (if available); uses the current
        # arrays of data_dict (after interpolation / set_errors / rotation)
        _freq = np.asarray(data_dict["freq"])
        _Z = np.asarray(data_dict["Z"])
        _Zerr = data_dict.get("Z_err")
        _dplus_results = {}
        for _comp, _zi, _zj in [("xy", 0, 1), ("yx", 1, 0)]:
            _ze = _Zerr[:, _zi, _zj] if _Zerr is not None else None
            _rho_plus, _rho_a, _ok = compute_rhoplus(_freq, _Z[:, _zi, _zj], _ze)
            _dplus_results[_comp] = {
                "rho_plus": _rho_plus, "rho_a": _rho_a, "pass": _ok
            }
            _nviol = int((~_ok).sum())
            print(f"  D+ {_comp.upper()}: {_nviol}/{len(_ok)} violations")
        if data_dict.get("Zdet") is not None:
            _ze = data_dict.get("Zdet_err")
            _rho_plus, _rho_a, _ok = compute_rhoplus(_freq, data_dict["Zdet"], _ze)
            _dplus_results["det"] = {
                "rho_plus": _rho_plus, "rho_a": _rho_a, "pass": _ok
            }
            print(f"  D+ DET: {int((~_ok).sum())}/{len(_ok)} violations")
        data_dict["dplus"] = _dplus_results


    all_data.append(data_dict)
    all_stems.append(Path(edi).stem)

    statname = station
    if STAT_FILE:
        nam, ext = os.path.splitext(os.path.basename(edi))
        statname = nam

    if STAT_UPPER:
        statname =  statname.upper()
        data_dict["station"] = statname

    if "edi" in OUT_FILES.lower():
        save_edi(
            **data_dict,
            path=DATA_DIR_OUT + statname + NAME_STR + ".edi",
        )
        print("Wrote file: ", DATA_DIR_OUT + statname + NAME_STR + ".edi")

    if "ncd" in OUT_FILES.lower():
        save_ncd(
            **data_dict,
            path=DATA_DIR_OUT + statname + NAME_STR + ".ncd",
        )
        print("Wrote file: ", DATA_DIR_OUT + statname + NAME_STR + ".ncd")

    if "hdf" in OUT_FILES.lower():
        save_hdf(
            **data_dict,
            path=DATA_DIR_OUT + statname + NAME_STR + ".hdf",
        )
        print("Wrote file: ", DATA_DIR_OUT + statname + NAME_STR + ".hdf")

    if "mat" in OUT_FILES.lower():
        save_mat(
            **data_dict,
            path=DATA_DIR_OUT + statname + NAME_STR + ".mat",
        )
        print("Wrote file: ", DATA_DIR_OUT + statname + NAME_STR + ".mat")

    if "npz" in OUT_FILES.lower():
        save_npz(
            **data_dict,
            path=DATA_DIR_OUT + statname + NAME_STR + ".npz",
        )
        print("Wrote file: ", DATA_DIR_OUT + statname + NAME_STR + ".npz")

    if PLOT:
        nrows = 3 + (1 if DPLUS else 0)
        fig, axs = plt.subplots(nrows, 2, figsize=(
            8, 14 + 4 * (nrows - 3)), sharex=True)

        _sup = utl.resolve_title(
            FIGURE_TITLE, statname, statname + NAME_STR.replace("_", " | "),
            name="FIGURE_TITLE", station=statname, name_str=NAME_STR)
        if _sup:
            fig.suptitle(_sup)

        df_rp = dataframe_from_edi(
            data_dict, include_tipper=False, include_pt=False
        )

        # Use per-call ylim overrides rather than mutating the shared PLTARGS dict.
        add_rho(df_rp,   comps="xy,yx", ax=axs[0, 0], **{**PLTARGS, "xlim": (1.e-4, 1.e+3), "ylim": (1.e-4, 1.e+4)})
        add_phase(df_rp, comps="xy,yx", ax=axs[0, 1], **{**PLTARGS, "xlim": (1.e-4, 1.e+3), "ylim": (-180., +180.)},
                  wrap_mode=PHASE_WRAP_MODE, wrap_style=PHASE_WRAP_STYLE)
        add_rho(df_rp,   comps="xx,yy", ax=axs[1, 0], **{**PLTARGS, "xlim": (1.e-4, 1.e+3), "ylim": (1.e-4, 1.e+4)})
        add_phase(df_rp, comps="xx,yy", ax=axs[1, 1], **{**PLTARGS, "xlim": (1.e-4, 1.e+3), "ylim": (-180., +180.)},
                  wrap_mode=PHASE_WRAP_MODE, wrap_style=PHASE_WRAP_STYLE)

        if DPLUS and data_dict.get("dplus"):
            add_rhoplus(data_dict, comps="xy,yx", ax=axs[2, 0],
                        **{**PLTARGS, "xlim": (1.e-4, 1.e+3), "ylim": (1.e-4, 1.e+4)})
            axs[2, 1].set_visible(False)   # no second D+ panel
            _tipper_row = 3
        else:
            _tipper_row = 2

        # add_tipper and add_pt accept the raw data_dict (auto-converted internally).
        add_tipper(data_dict, ax=axs[_tipper_row, 0],
                   **{**PLTARGS, "xlim": (1.e-4, 1.e+3), "ylim": (-1., +1.)})
        add_pt(data_dict,     ax=axs[_tipper_row, 1],
               **{**PLTARGS, "xlim": (1.e-4, 1.e+3), "ylim": (-1., +1.)})

        for ax in axs.flat:
            if not ax.get_visible():
                continue
            if not ax.lines and not ax.images and not ax.collections:
                fig.delaxes(ax)

        fig.tight_layout(rect=[0, 0, 1, 0.97])

        for f in PLOT_FORMAT:
            plt.savefig(PLOT_DIR + statname + NAME_STR + f, dpi=600)

        plt.show()

save_list_of_dicts_npz(
    records=all_data,
    path=DATA_DIR_OUT + COLL_NAME + NAME_STR + "_collection.npz",
)

# --- which station sits on which frequency of the grid? ----------------------
if INTERPOLATE and all_data:
    _g = np.asarray(grid, dtype=float)
    _cov = np.zeros((len(all_data), _g.size), dtype=int)     # 0 absent, 1 data, 2 extrapolated
    for _i, _d in enumerate(all_data):
        _idx = np.array([int(np.argmin(np.abs(np.log10(_g) - np.log10(f))))
                         for f in np.asarray(_d["freq"], dtype=float)], dtype=int)
        if not np.allclose(_g[_idx], np.asarray(_d["freq"], dtype=float), rtol=1.0e-9, atol=0.0):
            print(" WARNING: %s has frequencies that are not on the grid!" % all_stems[_i])
        _cov[_i, _idx] = np.where(np.asarray(_d["extrapolated"], dtype=bool), 2, 1)
    _nsite = (_cov > 0).sum(axis=0)
    _nfull = int(np.sum((_cov > 0).all(axis=1)))
    print("\n Grid: %d frequencies (%.4g - %.4g Hz); %d of %d stations are on all of them."
          % (_g.size, _g.min(), _g.max(), _nfull, len(all_data)))
    if _nsite.min() < len(all_data):
        print(" Stations per frequency: %d (at %.4g Hz) to %d.  Frequencies with fewer than "
              "all stations:" % (_nsite.min(), _g[int(np.argmin(_nsite))], _nsite.max()))
        _short = _nsite < len(all_data)
        _k = 0
        while _k < _g.size:
            if _short[_k]:
                _m = _k
                while _m + 1 < _g.size and _short[_m + 1]:
                    _m += 1
                print("   %.4g - %.4g Hz (%d points): %d to %d stations"
                      % (min(_g[_k], _g[_m]), max(_g[_k], _g[_m]), _m - _k + 1,
                         _nsite[_k:_m + 1].min(), _nsite[_k:_m + 1].max()))
                _k = _m + 1
            else:
                _k += 1
    _nxt = int((_cov == 2).sum())
    if _nxt:
        print(" %d extrapolated values (data_dict['extrapolated']); their errors are the edge "
              "values, inflate them if needed." % _nxt)
    if skipped:
        print(" WARNING: %d station(s) have no frequency of the grid and were skipped: %s"
              % (len(skipped), ", ".join(skipped[:10])))
    _covfile = DATA_DIR_OUT + COLL_NAME + NAME_STR + "_freq_coverage.csv"
    with open(_covfile, "w") as _fh:
        _fh.write("# 0 = no data, 1 = data, 2 = extrapolated; columns: grid frequencies [Hz]\n")
        _fh.write("station," + ",".join("%.6g" % v for v in _g) + "\n")
        for _i, _st in enumerate(all_stems):
            _fh.write(_st + "," + ",".join(str(int(v)) for v in _cov[_i]) + "\n")
    print(" Wrote coverage table: %s" % _covfile)

# =============================================================================
#  FEMTIC observe.dat (one file, all stations; optional dummy VTF stations)
# =============================================================================
if OBSERVE:
    import femtic as fem

    if not os.path.isfile(SITE_DAT):
        sys.exit(" observe.dat: site list %s not found (mt_make_sitelist.py)! Exit." % SITE_DAT)
    if OBS_ORIGIN_METHOD is None and (OBS_ORIGIN_E is None or OBS_ORIGIN_N is None):
        sys.exit(" observe.dat: give OBS_ORIGIN_E/N or OBS_ORIGIN_METHOD! Exit.")

    # model-local positions [m] of all sites in site.dat: x = easting, y = northing
    # relative to the origin (fem.utm_to_model, as in the other femtic_* scripts)
    (_e0, _n0, _lat0, _lon0, _zone, _north,
     _site_xys, _) = fem.resolve_origin_and_sites(
        site_dat=SITE_DAT, origin_method=OBS_ORIGIN_METHOD,
        origin_e=OBS_ORIGIN_E, origin_n=OBS_ORIGIN_N,
        zone_override=OBS_UTM_ZONE_OVERRIDE, plot_sites=True)
    _xy = {str(n).lower(): (x, y) for n, x, y, _z in _site_xys}

    def _ascending(d):
        """Copy of a station dict with freq, Z, Z_err in ascending frequency."""
        o = np.argsort(np.asarray(d["freq"], dtype=float))
        out = dict(d)
        for k in ("freq", "Z", "Z_err"):
            if out.get(k) is not None:
                out[k] = np.asarray(out[k])[o]
        return out

    obs_list, _missing = [], []
    for d, stem in zip(all_data, all_stems):
        xy = _xy.get(stem.lower()) or _xy.get(str(d.get("station", "")).lower())
        if xy is None:
            _missing.append(stem)
            continue
        dd = _ascending(d)
        dd["station"] = stem.upper() if STAT_UPPER else stem    # same names as the output EDIs
        dd["x_m"], dd["y_m"] = float(xy[0]), float(xy[1])
        obs_list.append(dd)
    obs_list.sort(key=lambda d: str(d["station"]))

    if _missing:
        print("\n WARNING: no position in %s for %d station(s), not written to observe.dat: %s"
              % (SITE_DAT, len(_missing), ", ".join(_missing[:10]) + (" ..." if len(_missing) > 10 else "")))
    if not obs_list:
        sys.exit(" observe.dat: no station with a position! Exit.")

    # checks on what goes into the file
    if str(OBS_INVALID).lower() not in ("drop", "mark"):
        sys.exit(" OBS_INVALID must be 'drop' or 'mark'! Exit.")
    for d in obs_list:
        if str(d.get("ft_convention", "e-iwt")).lower() == "e+iwt":
            print(" WARNING: %s is stored as e+iwt; the observe.dat writer assumes e-iwt." % d["station"])
        Ze = d.get("Z_err")
        if Ze is None:
            print(" WARNING: %s has no Z errors; observe.dat gets zero errors (not used by FEMTIC)." % d["station"])
        elif str(OBS_INVALID).lower() == "drop":
            _nz = int(np.sum(np.isfinite(Ze) & (np.asarray(Ze) <= 0.0)))
            if _nz:
                print(" NOTE: %s: %d Z error value(s) are 0 (e.g. a component that is invalid at all "
                      "frequencies); written as 0, FEMTIC does not use these data." % (d["station"], _nz))

    _nxo = sum(int(np.sum(d.get("extrapolated", 0))) for d in obs_list)
    if _nxo:
        print(" NOTE: %d extrapolated value(s) are written to observe.dat like measured data."
              % _nxo)

    _off = int(OBS_DUMMY_ID_OFFSET)
    if OBS_DUMMY_VTF and _off < len(obs_list):
        _off = int(10 ** np.ceil(np.log10(len(obs_list) + 1)))
        print(" dummy ID offset raised to %d (more sites than OBS_DUMMY_ID_OFFSET)" % _off)

    _written = fem.edi_list_to_observe_dat(
        obs_list, OBSERVE_FILE,
        dummy_vtf=OBS_DUMMY_VTF, site_type=OBS_SITE_TYPE, dummy_id_offset=_off,
        id_map_file=OBSERVE_FILE.replace(".dat", "_ids.csv"),
        invalid=OBS_INVALID, invalid_se=OBS_INVALID_SE)

    _bad = [r for r in _written if r["n_dropped"] or r["n_marked"]]
    if _bad:
        print("\n Invalid data (OBS_INVALID = '%s'):" % str(OBS_INVALID).lower())
        for r in _bad:
            print("   %-20s %d frequencies left out, %d values written with error %g"
                  % (r["name"], r["n_dropped"], r["n_marked"], OBS_INVALID_SE))
    _skipped_obs = len(obs_list) - len(_written)
    if _skipped_obs:
        print(" WARNING: %d station(s) without any usable frequency are not in observe.dat."
              % _skipped_obs)

    print("\n Wrote FEMTIC observation file: %s" % OBSERVE_FILE)
    print("   %d MT stations%s; id <-> name table: %s"
          % (len(_written), " + %d dummy VTF stations" % len(_written) if OBS_DUMMY_VTF else "",
             OBSERVE_FILE.replace(".dat", "_ids.csv")))
    print("   model origin (UTM zone %s%s): E = %.1f m, N = %.1f m"
          % (_zone, "N" if _north else "S", _e0, _n0))
    if OBS_DUMMY_VTF:
        print("   set  OWNER_ELEMENT  -1  in control.dat (site type: %s)" % OBS_SITE_TYPE)
