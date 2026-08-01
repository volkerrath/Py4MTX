#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split ModEM data files into period-band subsets.

Reads ModEM data files and writes separate files for each specified
period interval, suitable for band-by-band inversion or analysis.

@author: vrath (Feb 2021 / May 2024)
Cleanup: 4 Mar 2026 by Claude (Anthropic)
NRMS option added: 30 Jul 2026 by Claude (Anthropic)
"""

import os
import sys
from pathlib import Path
import inspect

import numpy as np

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

for _base in [PY4MTX_ROOT + "/py4mt/modules/"]:
    for _p in [Path(_base), *Path(_base).rglob("*")]:
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

import util as utl
import inverse as inv
from version import versionstrg

version, _ = versionstrg()
titstrng = utl.print_title(
    version=version, fname=inspect.getfile(inspect.currentframe()), out=False
)
print(titstrng + "\n\n")

# =============================================================================
#  Configuration
# =============================================================================
DAT_DIR_IN = "/home/vrath/Py4MTX/work/rms/" # PY4MTX_DATA + "/Fogo/"
DAT_DIR_OUT = DAT_DIR_IN

if not os.path.isdir(DAT_DIR_OUT):
    print("Directory: %s does not exist, but will be created" % DAT_DIR_OUT)
    os.mkdir(DAT_DIR_OUT)

DAT_FILES_IN = ["TAC26b_ZT_5_10a.dat"]

# Mapping from each observed data file to its corresponding calculated
# (predicted / forward-response) data file.
# ASSUMPTION (flagged, not verified against actual ModEM output naming):
# response files are assumed to sit in DAT_DIR_IN and are named by
# replacing "_in.dat" with "_calc.dat". Edit this mapping to match
# whatever ModEM run actually produced, e.g. "*_NLCG_050.dat".
CALC_FILES_IN = {
    datfile: datfile.replace(".dat", "_calc.dat") for datfile in DAT_FILES_IN
}


PER_INTERVALS = [
    [0.0001, 0.001],
    [0.001, 0.01],
    [0.01, 0.1],
    [0.1, 1.0],
    [1.0, 10.0],
    [10.0, 100.0],
    [100.0, 1000.0],
    [1000.0, 10000.0],
    [10000.0, 1000000.0],
]

#PER_NUM_MIN = 0
NUM_BANDS = len(PER_INTERVALS)

# -----------------------------------------------------------------------
#  NRMS configuration
# -----------------------------------------------------------------------
# Set True to compute total and subgroup NRMS/SRMS via inverse.calc_rms,
# comparing each observed file against a matching calculated (forward
# response) file.
COMPUTE_NRMS = True


# %%
# CALC_FILES_IN = {"datafile": "TACG26b_100ZT_Alpha05_NLCG_035.dat"}

# Subgroups to report NRMS/SRMS for, in addition to the total over all
# matched data. Choose any combination of: "datatype", "component",
# "site", "band" (period band, using PER_INTERVALS above).
NRMS_GROUP_BY = ["datatype", "component", "band", "site"]

# Optional explicit subsets. If a list is non-empty, only matching
# records are included in the NRMS/SRMS calculation (and hence in any
# subgroup breakdown above); leave a list empty ([]) to include
# everything for that attribute.
NRMS_SITE_LIST = []       # e.g. ["101", "102", "115"]  (site codes, exact match)
NRMS_COMP_LIST = []       # e.g. ["ZXY", "ZYX"]          (component codes, exact match)
NRMS_FREQ_LIST = []       # e.g. [0.1, 1.0, 10.0]         (frequencies in Hz)
NRMS_FREQ_RTOL = 1e-4     # relative tolerance for matching NRMS_FREQ_LIST to periods

# SMAPE (the second value returned by inverse.calc_rms) blows up whenever
# |dobs|+|dcalc| is small relative to the noise floor - this happens
# routinely for Tipper components, which cross zero, but rarely for
# impedances. A datum is included in SMAPE only if its combined
# obs+calc amplitude exceeds SMAPE_MIN_SNR * error; NRMS always uses
# every matched datum regardless of this threshold.
SMAPE_MIN_SNR = 1.0

NRMS_OUT_FILE = DAT_DIR_OUT + "modem_nrms_summary.txt"

# =============================================================================
#  Split data by period band
# =============================================================================
for datfile in DAT_FILES_IN:

    for ibnd in np.arange(NUM_BANDS):

        lowstr = str(1.0 / PER_INTERVALS[ibnd][0]) + "Hz"
        uppstr = str(1.0 / PER_INTERVALS[ibnd][1]) + "Hz"

        with open(DAT_DIR_IN + datfile) as fd:
            head = []
            data = []
            site = []
            perd = []
            for line in fd:
                if line.startswith("#") or line.startswith(">"):
                    head.append(line)
                    continue

                per = float(line.split()[0])
                sit = line.split()[1]
                if per >= PER_INTERVALS[ibnd][0] and per < PER_INTERVALS[ibnd][1]:
                    data.append(line)
                    site.append(sit)
                    perd.append(per)

        nper = len(np.unique(perd))
        nsit = len(np.unique(site))
        print(nper, "periods from", nsit, "sites")

        if nper > 0 and nsit > 0:
            phead = head.copy()
            phead = [lins.replace("per", str(nper)) for lins in phead]
            phead = [lins.replace("sit", str(nsit)) for lins in phead]

            base, ext = os.path.splitext(datfile)
            outfile = DAT_DIR_IN + base + "_perband" + str(ibnd) + ext

            # Safety check: never allow the split output to overwrite the
            # input file. Previously, outfile was built by replacing the
            # substring "_in.dat" in the input filename - if the filename
            # didn't contain that exact substring (e.g. "TAC26b_ZT_5_10a.dat"),
            # the replace was a no-op and outfile silently equalled the
            # input path, so writing the split output truncated and
            # overwrote the original observed-data file.
            if os.path.abspath(outfile) == os.path.abspath(DAT_DIR_IN + datfile):
                raise RuntimeError(
                    "Refusing to write split output to the same path as the "
                    "input file: %s" % outfile
                )
            print("output to", outfile)

            with open(outfile, "w") as fo:
                for ilin in np.arange(len(phead)):
                    fo.write(phead[ilin])
                for ilin in np.arange(len(data)):
                    fo.write(data[ilin])


# =============================================================================
#  NRMS / SRMS (total + subgroups), via inverse.calc_rms
# =============================================================================
def parse_modem_dat(path):
    """
    Parse a ModEM-format data file into a list of records.

    ASSUMPTION: standard ModEM column layout
        period  code  lat  lon  x  y  z  component  real  imag  error
    (whitespace-separated columns 0-10). Header/comment lines starting
    with '#' or '>' are skipped. Not verified against every ModEM data
    type (e.g. some Tipper/PT files may use fewer columns).
    """
    records = []
    with open(path) as fd:
        for line in fd:
            if line.startswith("#") or line.startswith(">") or not line.strip():
                continue
            cols = line.split()
            if len(cols) < 11:
                continue
            records.append(
                {
                    "period": float(cols[0]),
                    "site": cols[1],
                    "comp": cols[7],
                    "real": float(cols[8]),
                    "imag": float(cols[9]),
                    "error": float(cols[10]),
                }
            )
    return records


def band_index(period, intervals):
    for i, (lo, hi) in enumerate(intervals):
        if lo <= period < hi:
            return i
    return -1


def freq_in_list(period, freq_list, rtol):
    """True if 1/period matches any entry in freq_list within rtol."""
    if not freq_list:
        return True
    freq = 1.0 / period
    return any(abs(freq - f) <= rtol * f for f in freq_list)


def comp_datatype(comp):
    """
    Map a ModEM component code to a data-type label, so that files
    holding more than one data type (e.g. combined "ZT" impedance +
    tipper files) are split into separate NRMS groups per record.

    ASSUMPTION: component codes starting with "PT" are phase tensor
    ("P"), codes starting with "Z" are impedance ("Z"), codes starting
    with "T" are tipper ("T"). Falls back to the component's first
    character for anything else - adjust here if your component
    codes use a different convention.
    """
    c = comp.upper()
    if c.startswith("PT"):
        return "P"
    if c.startswith("Z"):
        return "Z"
    if c.startswith("T"):
        return "T"
    return c[0] if c else "?"


if COMPUTE_NRMS:

    all_dobs = []
    all_dcalc = []
    all_werr = []
    all_tags = []  # (datatype, component, site, band) per residual entry
    all_periods = []  # one entry per matched record (not doubled for re/im)

    for datfile in DAT_FILES_IN:

        if datfile not in CALC_FILES_IN:
            print("No calculated-data file configured for", datfile,
                  "- skipping NRMS")
            continue

        obs_path = DAT_DIR_IN + datfile
        calc_path = DAT_DIR_IN + CALC_FILES_IN[datfile]

        if not os.path.isfile(calc_path):
            print("Calculated-data file not found:", calc_path,
                  "- skipping NRMS for", datfile)
            continue

        obs_records = parse_modem_dat(obs_path)
        calc_records = parse_modem_dat(calc_path)

        calc_lookup = {
            (round(r["period"], 8), r["site"], r["comp"]): r
            for r in calc_records
        }

        n_matched = 0
        for r in obs_records:
            if NRMS_SITE_LIST and r["site"] not in NRMS_SITE_LIST:
                continue
            if NRMS_COMP_LIST and r["comp"] not in NRMS_COMP_LIST:
                continue
            if not freq_in_list(r["period"], NRMS_FREQ_LIST, NRMS_FREQ_RTOL):
                continue

            key = (round(r["period"], 8), r["site"], r["comp"])
            c = calc_lookup.get(key)
            if c is None:
                continue
            n_matched += 1
            bnd = band_index(r["period"], PER_INTERVALS)
            dtype = comp_datatype(r["comp"])

            # Real and imaginary parts are treated as independent
            # real-valued data points, both normalised by the same
            # error estimate (standard ModEM convention).
            all_dobs.extend([r["real"], r["imag"]])
            all_dcalc.extend([c["real"], c["imag"]])
            all_werr.extend([1.0 / r["error"], 1.0 / r["error"]])
            all_tags.extend([(dtype, r["comp"], r["site"], bnd)] * 2)
            all_periods.append(r["period"])

        print(datfile, ": matched", n_matched, "of", len(obs_records),
              "observations with", calc_path)

    if len(all_dobs) == 0:
        print("\nNo matched observed/calculated data found - NRMS not computed")
    else:
        dobs = np.array(all_dobs)
        dcalc = np.array(all_dcalc)
        werr = np.array(all_werr)
        tags = all_tags

        summary_lines = []

        def compute_metrics(mask):
            """
            NRMS uses every matched datum in `mask`. SMAPE additionally
            excludes data whose combined obs+calc amplitude does not
            exceed SMAPE_MIN_SNR * error, since SMAPE is unstable for
            near-zero-crossing data (e.g. Tipper). Returns
            (nrms, smape_or_None, n_total, n_excluded_from_smape).
            """
            n_full = int(mask.sum())
            if n_full == 0:
                return None, None, 0, 0

            sub_dcalc = dcalc[mask]
            sub_dobs = dobs[mask]
            sub_werr = werr[mask]

            nrms_val, _ = inv.calc_rms(dcalc=sub_dcalc, dobs=sub_dobs, Wd=sub_werr)

            sub_err = 1.0 / sub_werr
            denom = np.abs(sub_dobs) + np.abs(sub_dcalc)
            valid = denom > SMAPE_MIN_SNR * sub_err
            n_smape = int(valid.sum())
            if n_smape >= 1:
                _, smape_val = inv.calc_rms(
                    dcalc=sub_dcalc[valid], dobs=sub_dobs[valid], Wd=sub_werr[valid]
                )
            else:
                smape_val = None

            return nrms_val, smape_val, n_full, n_full - n_smape

        mask_all = np.ones(len(dobs), dtype=bool)
        nrms_tot, smape_tot, n_tot, n_excl_tot = compute_metrics(mask_all)
        smape_str = "n/a" if smape_tot is None else "%.2f%%" % smape_tot
        excl_note = "" if n_excl_tot == 0 else " (excl %d from SMAPE)" % n_excl_tot
        summary_lines.append(
            "TOTAL: N=%d  NRMS=%.4f  SMAPE=%s%s"
            % (n_tot, nrms_tot, smape_str, excl_note)
        )

        # Diagnostic: how much of the matched period range is actually
        # covered by more than one of the configured PER_INTERVALS bands.
        # If the range spans multiple bands but the grouped-by-band
        # breakdown below still shows only one populated band, that is a
        # sign something is off (e.g. wrong column parsed as "period",
        # or PER_INTERVALS boundaries don't match this survey's units).
        pmin, pmax = min(all_periods), max(all_periods)
        n_bands_spanned = sum(
            1 for (lo, hi) in PER_INTERVALS if lo < pmax and hi > pmin
        )
        summary_lines.append(
            "Matched period range: %.6g - %.6g s "
            "(overlaps %d of %d configured PER_INTERVALS bands)"
            % (pmin, pmax, n_bands_spanned, NUM_BANDS)
        )

        def group_key(tag, kind):
            datatype, comp, site, bnd = tag
            if kind == "datatype":
                return datatype
            if kind == "component":
                return comp
            if kind == "site":
                return site
            if kind == "band":
                return bnd
            return None

        for kind in NRMS_GROUP_BY:
            summary_lines.append("\n--- grouped by %s ---" % kind)
            keys = sorted(set(group_key(t, kind) for t in tags), key=str)
            for k in keys:
                mask = np.array([group_key(t, kind) == k for t in tags])
                nrms_g, smape_g, n_g, n_excl_g = compute_metrics(mask)
                if n_g < 1:
                    continue
                label = k
                if kind == "band" and k >= 0:
                    lo, hi = PER_INTERVALS[k]
                    label = "band%d [%.4g-%.4g s]" % (k, lo, hi)
                smape_str = "n/a" if smape_g is None else "%.2f%%" % smape_g
                excl_note = "" if n_excl_g == 0 else " (excl %d)" % n_excl_g
                summary_lines.append(
                    "  %-25s N=%-6d NRMS=%.4f  SMAPE=%s%s"
                    % (str(label), n_g, nrms_g, smape_str, excl_note)
                )

        print("\n".join(summary_lines))

        with open(NRMS_OUT_FILE, "w") as fo:
            fo.write("\n".join(summary_lines) + "\n")
        print("\nNRMS summary written to", NRMS_OUT_FILE)
