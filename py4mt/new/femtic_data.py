#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_data.py — FEMTIC result merger + MT sounding-curve plotter

Merges per-PE FEMTIC result CSV files, computes per-site and total RMS,
writes merged result tables (MT, VTF, HTF, PT, NMT, NMT2,
APP_RES_AND_PHS, NMT2_APP_RES_AND_PHS), and optionally plots MT apparent
resistivity / phase sounding curves for all stations using PyGMT.

Usage
-----
    python femtic_data.py <iter> <numPE> [options]

Options
-------
    -name   <file>  Read site-ID → site-name mapping from <file>
    -err    <file>  Read true-error file from <file>
    -csv            Write output as CSV (default: fixed-width text)
    -undist         Apply galvanic-distortion correction (reads control.dat
                    and distortion_iter<iter>.dat)
    -appphs         Convert impedance tensors to apparent resistivity / phase

    -plot           Plot MT sounding curves after writing result files
    -sites  <file>  Two-column site list for MT plot order/labels:
                        <site_name_or_id>  <display_label>
                    If omitted, all MT sites are plotted in data order.
    -out    <pfx>   Output PDF prefix for MT plots  (default: imp_all_curv)
    -perpage <N>    Stations per PDF page            (default: 12)
    -cols   <N>     Columns per page                 (default: 6)

    -plotvtf            Plot VTF (tipper) sounding curves
    -sitesvtf  <file>   Two-column site list for VTF plot order/labels
    -outvtf    <pfx>    Output PDF prefix for VTF plots (default: vtf_all_curv)
    -perpageVTF <N>     Stations per PDF page for VTF  (default: 12)
    -colsVTF    <N>     Columns per page for VTF        (default: 6)

Provenance
----------
    Original C++ : Yoshiya Usui (MIT License, 2021)
    Python port  : Claude (Anthropic), 2026-04-03
    Plotting      : Claude (Anthropic), 2026-04-06
"""

import sys
import math
import cmath
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Plotting imports (optional -- only required when -plot is used)
try:
    import numpy as np
    import pandas as pd
    import pygmt
    _PLOT_AVAILABLE = True
except ImportError:
    _PLOT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
MU0     = 4.0 * math.pi * 1.0e-7
RAD2DEG = 180.0 / math.pi
DEG2RAD = math.pi / 180.0
FACTOR  = 1.0

# ---------------------------------------------------------------------------
# Index labels
# ---------------------------------------------------------------------------
COMP_INDEX    = ["xx", "xy", "yx", "yy", "zx", "zy"]
COMP_INDEX_PT = ["11", "12", "21", "22"]
COMP_INDEX_XY = ["x", "y"]

# Tensor component indices
XX, XY, YX, YY = 0, 1, 2, 3

# ---------------------------------------------------------------------------
# Station types
# ---------------------------------------------------------------------------
MT                   = 0
VTF                  = 1
PT                   = 2
HTF                  = 3
NMT                  = 4
NMT2                 = 5
APP_RES_AND_PHS      = 6
NMT2_APP_RES_AND_PHS = 7
ADDITINAL_OUTPUT_POINT = 8

# Distortion types
NO_DISTORTION                       = 0
ESTIMATE_DISTORTION_MATRIX_DIFFERENCE = 1
ESTIMATE_GAINS_AND_ROTATIONS         = 2
ESTIMATE_GAINS_ONLY                  = 3

# ---------------------------------------------------------------------------
# Data structures (plain dataclasses replacing C++ structs)
# ---------------------------------------------------------------------------

@dataclass
class ImpedanceTensor:
    Z: List[complex] = field(default_factory=lambda: [0+0j]*4)

@dataclass
class VerticalMagneticTransferFunction:
    TZ: List[complex] = field(default_factory=lambda: [0+0j]*2)

@dataclass
class HorizontalMagneticTransferFunction:
    T: List[complex] = field(default_factory=lambda: [0+0j]*4)

@dataclass
class PhaseTensor:
    Phi: List[float] = field(default_factory=lambda: [0.0]*4)

@dataclass
class NMTResponseFunction:
    Y: List[complex] = field(default_factory=lambda: [0+0j]*2)

@dataclass
class ApparentResistivityAndPhase:
    apparentResistivity: List[float] = field(default_factory=lambda: [0.0]*4)
    phase:               List[float] = field(default_factory=lambda: [0.0]*4)

@dataclass
class DistortionMatrix:
    C: List[float] = field(default_factory=lambda: [0.0]*4)

@dataclass
class MTData:
    freq: float = 0.0
    Cal:  ImpedanceTensor = field(default_factory=ImpedanceTensor)
    Res:  ImpedanceTensor = field(default_factory=ImpedanceTensor)
    Obs:  ImpedanceTensor = field(default_factory=ImpedanceTensor)
    Err:  ImpedanceTensor = field(default_factory=ImpedanceTensor)

@dataclass
class VTFData:
    freq: float = 0.0
    Cal:  VerticalMagneticTransferFunction = field(default_factory=VerticalMagneticTransferFunction)
    Res:  VerticalMagneticTransferFunction = field(default_factory=VerticalMagneticTransferFunction)
    Obs:  VerticalMagneticTransferFunction = field(default_factory=VerticalMagneticTransferFunction)
    Err:  VerticalMagneticTransferFunction = field(default_factory=VerticalMagneticTransferFunction)

@dataclass
class HTFData:
    freq: float = 0.0
    Cal:  HorizontalMagneticTransferFunction = field(default_factory=HorizontalMagneticTransferFunction)
    Res:  HorizontalMagneticTransferFunction = field(default_factory=HorizontalMagneticTransferFunction)
    Obs:  HorizontalMagneticTransferFunction = field(default_factory=HorizontalMagneticTransferFunction)
    Err:  HorizontalMagneticTransferFunction = field(default_factory=HorizontalMagneticTransferFunction)

@dataclass
class PTData:
    freq: float = 0.0
    Cal:  PhaseTensor = field(default_factory=PhaseTensor)
    Res:  PhaseTensor = field(default_factory=PhaseTensor)
    Obs:  PhaseTensor = field(default_factory=PhaseTensor)
    Err:  PhaseTensor = field(default_factory=PhaseTensor)

@dataclass
class NMTData:
    freq: float = 0.0
    Cal:  NMTResponseFunction = field(default_factory=NMTResponseFunction)
    Res:  NMTResponseFunction = field(default_factory=NMTResponseFunction)
    Obs:  NMTResponseFunction = field(default_factory=NMTResponseFunction)
    Err:  NMTResponseFunction = field(default_factory=NMTResponseFunction)

@dataclass
class ApparentResistivityAndPhaseData:
    freq: float = 0.0
    Cal:  ApparentResistivityAndPhase = field(default_factory=ApparentResistivityAndPhase)
    Res:  ApparentResistivityAndPhase = field(default_factory=ApparentResistivityAndPhase)
    Obs:  ApparentResistivityAndPhase = field(default_factory=ApparentResistivityAndPhase)
    Err:  ApparentResistivityAndPhase = field(default_factory=ApparentResistivityAndPhase)

# True-error entries: list of (real_err, imag_err) pairs per component
@dataclass
class MTTrueError:
    freq:  float = 0.0
    error: List[Tuple[float,float]] = field(default_factory=lambda: [(0.,0.)]*4)

@dataclass
class VTFTrueError:
    freq:  float = 0.0
    error: List[Tuple[float,float]] = field(default_factory=lambda: [(0.,0.)]*2)

@dataclass
class HTFTrueError:
    freq:  float = 0.0
    error: List[Tuple[float,float]] = field(default_factory=lambda: [(0.,0.)]*4)

@dataclass
class PTTrueError:
    freq:  float = 0.0
    error: List[float] = field(default_factory=lambda: [0.]*4)

@dataclass
class NMTTrueError:
    freq:  float = 0.0
    error: List[Tuple[float,float]] = field(default_factory=lambda: [(0.,0.)]*2)

@dataclass
class NumDataAndSumResidual:
    numData:     int   = 0
    sumResidual: float = 0.0

# ---------------------------------------------------------------------------
# Global state (mirrors the C++ globals)
# ---------------------------------------------------------------------------
station_type_cur: int = -1
read_true_error_file: bool = False
true_error_file_name: str = ""
output_csv: bool = False
impedance_converted_to_app_res: bool = False
type_of_distortion: int = NO_DISTORTION

mt_data_list:              List[Tuple[int, MTData]]                           = []
vtf_data_list:             List[Tuple[int, VTFData]]                          = []
htf_data_list:             List[Tuple[int, HTFData]]                          = []
pt_data_list:              List[Tuple[int, PTData]]                           = []
nmt_data_list:             List[Tuple[int, NMTData]]                          = []
nmt2_data_list:            List[Tuple[int, MTData]]                           = []
app_res_phs_data_list:     List[Tuple[int, ApparentResistivityAndPhaseData]]  = []
nmt2_app_res_phs_data_list:List[Tuple[int, ApparentResistivityAndPhaseData]]  = []

mt_true_error_list:              List[Tuple[int, MTTrueError]]  = []
vtf_true_error_list:             List[Tuple[int, VTFTrueError]] = []
htf_true_error_list:             List[Tuple[int, HTFTrueError]] = []
pt_true_error_list:              List[Tuple[int, PTTrueError]]  = []
nmt_true_error_list:             List[Tuple[int, NMTTrueError]] = []
nmt2_true_error_list:            List[Tuple[int, MTTrueError]]  = []
app_res_phs_true_error_list:     List[Tuple[int, MTTrueError]]  = []
nmt2_app_res_phs_true_error_list:List[Tuple[int, MTTrueError]]  = []

site_id_to_type: Dict[int, int] = {}
distortion_matrix_list: Dict[int, DistortionMatrix] = {}
site_id_to_name: Dict[int, str] = {}


# ===========================================================================
# I/O helpers
# ===========================================================================

def write_field(f, width: int, value, is_csv: bool) -> None:
    """Write a single field in either CSV or fixed-width format."""
    if is_csv:
        f.write(f"{value},")
    else:
        f.write(f"{str(value):>{width}}")


def fmt_sci(v: float) -> str:
    """Format a float as scientific notation with 6 decimal places."""
    return f"{v:.6e}"


def write_field_sci(f, width: int, value: float, is_csv: bool) -> None:
    s = fmt_sci(value)
    if is_csv:
        f.write(s + ",")
    else:
        f.write(f"{s:>{width}}")


# ===========================================================================
# Lookup helpers (mirrors getIteratorTo*TrueError)
# ===========================================================================

def _find_true_error(error_list, site_id: int, freq: float):
    for sid, te in error_list:
        if sid == site_id and abs(freq - te.freq) < 1e-6:
            return te
    print(f"Site ID={site_id} and frequency={freq} is not found !!", file=sys.stderr)
    sys.exit(1)

def get_mt_true_error(site_id, freq):
    return _find_true_error(mt_true_error_list, site_id, freq)

def get_vtf_true_error(site_id, freq):
    return _find_true_error(vtf_true_error_list, site_id, freq)

def get_htf_true_error(site_id, freq):
    return _find_true_error(htf_true_error_list, site_id, freq)

def get_pt_true_error(site_id, freq):
    return _find_true_error(pt_true_error_list, site_id, freq)

def get_nmt_true_error(site_id, freq):
    return _find_true_error(nmt_true_error_list, site_id, freq)

def get_nmt2_true_error(site_id, freq):
    return _find_true_error(nmt2_true_error_list, site_id, freq)

def get_app_res_phs_true_error(site_id, freq):
    return _find_true_error(app_res_phs_true_error_list, site_id, freq)

def get_nmt2_app_res_phs_true_error(site_id, freq):
    return _find_true_error(nmt2_app_res_phs_true_error_list, site_id, freq)


def get_site_type(site_id: int) -> int:
    if site_id not in site_id_to_type:
        print(f"Site ID ({site_id}) is not found !!", file=sys.stderr)
        return -1
    return site_id_to_type[site_id]


def convert_site_id_to_name(site_id: int) -> str:
    return site_id_to_name.get(site_id, str(site_id))


def is_same_error(site_id, freq, index, err_real, err_imag):
    eps = 1e-10
    if abs(err_real - err_imag) > eps:
        print(f"Errors of real and imaginary part are different. "
              f"Station ID: {site_id}, Frequency [Hz]: {freq}, "
              f"Component index: {index}", file=sys.stderr)
        print(f"Error (real part): {err_real}", file=sys.stderr)
        print(f"Error (imaginary part): {err_imag}", file=sys.stderr)


def add_num_data_and_sum_residual(
    site_id: int,
    accum: Dict[int, NumDataAndSumResidual],
    residual: float,
) -> None:
    if site_id in accum:
        accum[site_id].numData     += 1
        accum[site_id].sumResidual += residual * residual
    else:
        accum[site_id] = NumDataAndSumResidual(numData=1, sumResidual=residual * residual)


# ===========================================================================
# Read functions
# ===========================================================================

def read_result(iteration_number: int, num_pe: int) -> None:
    """Read all per-PE result CSV files."""
    global station_type_cur

    for pe in range(num_pe):
        fname = f"result_{pe}_iter{iteration_number}.csv"
        try:
            f = open(fname)
        except OSError:
            print(f"File open error : {fname} !!", file=sys.stderr)
            sys.exit(1)
        print(f"Read result from {fname}")

        for line in f:
            line = line.rstrip("\n")
            parts = line.split(",")
            if not parts:
                continue
            first = parts[0].strip()

            # Section headers
            type_map = {
                "MT": MT, "VTF": VTF, "PT": PT, "HTF": HTF,
                "NMT": NMT, "NMT2": NMT2,
                "APP_RES_AND_PHS": APP_RES_AND_PHS,
                "NMT2_APP_RES_AND_PHS": NMT2_APP_RES_AND_PHS,
                "ADDITINAL_OUTPUT_POINT": ADDITINAL_OUTPUT_POINT,
            }
            if first in type_map:
                station_type_cur = type_map[first]
                continue
            if "StaID" in first:
                continue

            # Parse station ID
            try:
                stat_id = int(first)
            except ValueError:
                continue

            p = iter(parts[1:])   # remaining tokens

            def next_float():
                return float(next(p).strip())

            def next_complex():
                r = float(next(p).strip())
                i = float(next(p).strip())
                return complex(r, i)

            if station_type_cur == MT:
                data = MTData(freq=next_float())
                for arr in (data.Cal.Z, data.Res.Z, data.Obs.Z, data.Err.Z):
                    for k in range(4):
                        arr[k] = next_complex()
                mt_data_list.append((stat_id, data))
                site_id_to_type[stat_id] = MT

            elif station_type_cur == VTF:
                data = VTFData(freq=next_float())
                for arr in (data.Cal.TZ, data.Res.TZ, data.Obs.TZ, data.Err.TZ):
                    for k in range(2):
                        arr[k] = next_complex()
                vtf_data_list.append((stat_id, data))
                site_id_to_type[stat_id] = VTF

            elif station_type_cur == HTF:
                data = HTFData(freq=next_float())
                for arr in (data.Cal.T, data.Res.T, data.Obs.T, data.Err.T):
                    for k in range(4):
                        arr[k] = next_complex()
                htf_data_list.append((stat_id, data))
                site_id_to_type[stat_id] = HTF

            elif station_type_cur == PT:
                data = PTData(freq=next_float())
                for arr in (data.Cal.Phi, data.Res.Phi, data.Obs.Phi, data.Err.Phi):
                    for k in range(4):
                        arr[k] = next_float()
                pt_data_list.append((stat_id, data))
                site_id_to_type[stat_id] = PT

            elif station_type_cur == NMT:
                data = NMTData(freq=next_float())
                for arr in (data.Cal.Y, data.Res.Y, data.Obs.Y, data.Err.Y):
                    for k in range(2):
                        arr[k] = next_complex()
                nmt_data_list.append((stat_id, data))
                site_id_to_type[stat_id] = NMT

            elif station_type_cur == NMT2:
                data = MTData(freq=next_float())
                for arr in (data.Cal.Z, data.Res.Z, data.Obs.Z, data.Err.Z):
                    for k in range(4):
                        arr[k] = next_complex()
                nmt2_data_list.append((stat_id, data))
                site_id_to_type[stat_id] = NMT2

            elif station_type_cur == APP_RES_AND_PHS:
                data = ApparentResistivityAndPhaseData(freq=next_float())
                for obj in (data.Cal, data.Res, data.Obs, data.Err):
                    for k in range(4):
                        obj.apparentResistivity[k] = next_float()
                        obj.phase[k]               = next_float()
                app_res_phs_data_list.append((stat_id, data))
                site_id_to_type[stat_id] = APP_RES_AND_PHS

            elif station_type_cur == NMT2_APP_RES_AND_PHS:
                data = ApparentResistivityAndPhaseData(freq=next_float())
                for obj in (data.Cal, data.Res, data.Obs, data.Err):
                    for k in range(4):
                        obj.apparentResistivity[k] = next_float()
                        obj.phase[k]               = next_float()
                nmt2_app_res_phs_data_list.append((stat_id, data))
                site_id_to_type[stat_id] = NMT2_APP_RES_AND_PHS

            elif station_type_cur == ADDITINAL_OUTPUT_POINT:
                break
            else:
                print(f"Unsupported station type : {station_type_cur}", file=sys.stderr)
                break

        f.close()


def read_control_data(file_name: str) -> None:
    global type_of_distortion
    try:
        f = open(file_name)
    except OSError:
        print(f"File open error : {file_name} !!", file=sys.stderr)
        sys.exit(1)

    found = False
    for line in f:
        token = line.strip().split()
        if not token:
            continue
        keyword = token[0]
        if keyword.startswith("DISTORTION"):
            ibuf = int(token[1]) if len(token) > 1 else int(next(iter(f)).strip())
            valid = {NO_DISTORTION, ESTIMATE_DISTORTION_MATRIX_DIFFERENCE,
                     ESTIMATE_GAINS_AND_ROTATIONS, ESTIMATE_GAINS_ONLY}
            if ibuf not in valid:
                print(f"Error : Wrong type ID below DISTORTION : {ibuf}", file=sys.stderr)
                sys.exit(1)
            type_of_distortion = ibuf
            found = True
        elif keyword.startswith("END"):
            break
    f.close()

    if found:
        print(f"Type of distortion : {type_of_distortion}")
    else:
        print(f"DISTORTION is not found in {file_name} !!", file=sys.stderr)


def read_true_error(file_name: str) -> None:
    try:
        f = open(file_name)
    except OSError:
        print(f"File open error : {file_name} !!", file=sys.stderr)
        sys.exit(1)

    for line in f:
        parts = line.split()
        if not parts:
            continue
        stat_id = int(parts[0])
        if stat_id < 0:
            break
        stype = get_site_type(stat_id)
        rest = iter(parts[1:])

        def nf():
            return float(next(rest))

        if stype == MT:
            te = MTTrueError(freq=nf(), error=[(nf(), nf()) for _ in range(4)])
            mt_true_error_list.append((stat_id, te))
        elif stype == VTF:
            te = VTFTrueError(freq=nf(), error=[(nf(), nf()) for _ in range(2)])
            vtf_true_error_list.append((stat_id, te))
        elif stype == HTF:
            te = HTFTrueError(freq=nf(), error=[(nf(), nf()) for _ in range(4)])
            htf_true_error_list.append((stat_id, te))
        elif stype == PT:
            te = PTTrueError(freq=nf(), error=[nf() for _ in range(4)])
            pt_true_error_list.append((stat_id, te))
        elif stype == NMT:
            te = NMTTrueError(freq=nf(), error=[(nf(), nf()) for _ in range(2)])
            nmt_true_error_list.append((stat_id, te))
        elif stype == NMT2:
            te = MTTrueError(freq=nf(), error=[(nf(), nf()) for _ in range(4)])
            nmt2_true_error_list.append((stat_id, te))
        elif stype == APP_RES_AND_PHS:
            te = MTTrueError(freq=nf(), error=[(nf(), nf()) for _ in range(4)])
            app_res_phs_true_error_list.append((stat_id, te))
        elif stype == NMT2_APP_RES_AND_PHS:
            te = MTTrueError(freq=nf(), error=[(nf(), nf()) for _ in range(4)])
            nmt2_app_res_phs_true_error_list.append((stat_id, te))
        else:
            print(f"Unsupported station type : {stype}", file=sys.stderr)
            break
    f.close()


def read_distortion_matrix(iteration_number: int) -> None:
    if type_of_distortion == NO_DISTORTION:
        return
    fname = f"distortion_iter{iteration_number}.dat"
    try:
        f = open(fname)
    except OSError:
        print(f"{fname} is not found.", file=sys.stderr)
        sys.exit(1)
    print(f"Read distortion matrix from {fname}")

    tokens = f.read().split()
    f.close()
    it = iter(tokens)

    def nf(): return float(next(it))
    def ni(): return int(next(it))

    num_site = ni()
    for _ in range(num_site):
        site_id = ni()
        dm = DistortionMatrix()
        if type_of_distortion == ESTIMATE_DISTORTION_MATRIX_DIFFERENCE:
            for k in range(4):
                dm.C[k] = nf()
            dm.C[XX] += 1.0
            dm.C[YY] += 1.0
        elif type_of_distortion == ESTIMATE_GAINS_AND_ROTATIONS:
            dbuf = [nf() for _ in range(4)]
            gX = 10.0 ** dbuf[0]
            gY = 10.0 ** dbuf[1]
            betaX = dbuf[2] * DEG2RAD
            betaY = dbuf[3] * DEG2RAD
            dm.C[XX] =  gX * math.cos(betaX)
            dm.C[XY] = -gY * math.sin(betaY)
            dm.C[YX] =  gX * math.sin(betaX)
            dm.C[YY] =  gY * math.cos(betaY)
        elif type_of_distortion == ESTIMATE_GAINS_ONLY:
            dbuf = [nf() for _ in range(2)]
            dm.C[XX] = 10.0 ** dbuf[0]
            dm.C[XY] = 0.0
            dm.C[YX] = 0.0
            dm.C[YY] = 10.0 ** dbuf[1]
        else:
            print(f"Unsupported type of distortion : {type_of_distortion}", file=sys.stderr)
        distortion_matrix_list[site_id] = dm
        ni()   # trailing integer per entry


def read_relation_site_id_to_name(relation_file: str) -> None:
    try:
        f = open(relation_file)
    except OSError:
        print(f"File open error : {relation_file} !!", file=sys.stderr)
        sys.exit(1)
    print(f"Read relation between site ID and site name from {relation_file}")
    for line in f:
        parts = line.split()
        if len(parts) >= 2:
            site_id_to_name[int(parts[0])] = parts[1]
    f.close()


# ===========================================================================
# Distortion correction
# ===========================================================================

def calc_undistorted_impedance_tensor(
    site_id: int, Z: ImpedanceTensor
) -> ImpedanceTensor:
    out = ImpedanceTensor()
    dm = distortion_matrix_list.get(site_id)
    if dm is None:
        out.Z = list(Z.Z)
    else:
        det = dm.C[XX] * dm.C[YY] - dm.C[XY] * dm.C[YX]
        CInvXX =  dm.C[YY] / det
        CInvXY = -dm.C[XY] / det
        CInvYX = -dm.C[YX] / det
        CInvYY =  dm.C[XX] / det
        out.Z[XX] = CInvXX * Z.Z[XX] + CInvXY * Z.Z[YX]
        out.Z[XY] = CInvXX * Z.Z[XY] + CInvXY * Z.Z[YY]
        out.Z[YX] = CInvYX * Z.Z[XX] + CInvYY * Z.Z[YX]
        out.Z[YY] = CInvYX * Z.Z[XY] + CInvYY * Z.Z[YY]
    return out


def calc_undistorted_app_res_and_phase(
    site_id: int, freq: float, app: ApparentResistivityAndPhase
) -> ApparentResistivityAndPhase:
    out = ApparentResistivityAndPhase()
    dm = distortion_matrix_list.get(site_id)
    if dm is None:
        out.apparentResistivity = list(app.apparentResistivity)
        out.phase = list(app.phase)
    else:
        omega = 2.0 * math.pi * freq
        dist = ImpedanceTensor()
        for i in range(4):
            absZ   = math.sqrt(app.apparentResistivity[i] * MU0 * omega)
            phsRad = app.phase[i] * DEG2RAD
            dist.Z[i] = complex(absZ * math.cos(phsRad), absZ * math.sin(phsRad))
        undist = calc_undistorted_impedance_tensor(site_id, dist)
        for i in range(4):
            out.apparentResistivity[i] = (undist.Z[i].real**2 + undist.Z[i].imag**2) / (omega * MU0)
            out.phase[i] = math.atan2(undist.Z[i].imag, undist.Z[i].real) * RAD2DEG
    return out


# ===========================================================================
# RMS calculation
# ===========================================================================

def calc_true_rms(does_read_true_error: bool, true_error_fname: str) -> None:
    if does_read_true_error:
        read_true_error(true_error_fname)

    accum: Dict[int, NumDataAndSumResidual] = {}

    # MT
    for site_id, d in mt_data_list:
        freq = d.freq
        if does_read_true_error:
            te = get_mt_true_error(site_id, freq)
            for i in range(4):
                if te.error[i][0] > 0.0:
                    res = (d.Cal.Z[i].real - d.Obs.Z[i].real) / te.error[i][0]
                    add_num_data_and_sum_residual(site_id, accum, res)
                if te.error[i][1] > 0.0:
                    res = (d.Cal.Z[i].imag - d.Obs.Z[i].imag) / te.error[i][1]
                    add_num_data_and_sum_residual(site_id, accum, res)
        else:
            for i in range(4):
                if d.Err.Z[i].real > 0.0:
                    add_num_data_and_sum_residual(site_id, accum, d.Res.Z[i].real)
                if d.Err.Z[i].imag > 0.0:
                    add_num_data_and_sum_residual(site_id, accum, d.Res.Z[i].imag)

    # VTF
    for site_id, d in vtf_data_list:
        freq = d.freq
        if does_read_true_error:
            te = get_vtf_true_error(site_id, freq)
            for i in range(2):
                if te.error[i][0] > 0.0:
                    res = (d.Cal.TZ[i].real - d.Obs.TZ[i].real) / te.error[i][0]
                    add_num_data_and_sum_residual(site_id, accum, res)
                if te.error[i][1] > 0.0:
                    res = (d.Cal.TZ[i].imag - d.Obs.TZ[i].imag) / te.error[i][1]
                    add_num_data_and_sum_residual(site_id, accum, res)
        else:
            for i in range(2):
                if d.Err.TZ[i].real > 0.0:
                    add_num_data_and_sum_residual(site_id, accum, d.Res.TZ[i].real)
                if d.Err.TZ[i].imag > 0.0:
                    add_num_data_and_sum_residual(site_id, accum, d.Res.TZ[i].imag)

    # HTF
    for site_id, d in htf_data_list:
        freq = d.freq
        if does_read_true_error:
            te = get_htf_true_error(site_id, freq)
            for i in range(4):
                if te.error[i][0] > 0.0:
                    res = (d.Cal.T[i].real - d.Obs.T[i].real) / te.error[i][0]
                    add_num_data_and_sum_residual(site_id, accum, res)
                if te.error[i][1] > 0.0:
                    res = (d.Cal.T[i].imag - d.Obs.T[i].imag) / te.error[i][1]
                    add_num_data_and_sum_residual(site_id, accum, res)
        else:
            for i in range(4):
                if d.Err.T[i].real > 0.0:
                    add_num_data_and_sum_residual(site_id, accum, d.Res.T[i].real)
                if d.Err.T[i].imag > 0.0:
                    add_num_data_and_sum_residual(site_id, accum, d.Res.T[i].imag)

    # PT
    for site_id, d in pt_data_list:
        freq = d.freq
        if does_read_true_error:
            te = get_pt_true_error(site_id, freq)
            for i in range(4):
                if te.error[i] > 0.0:
                    res = (d.Cal.Phi[i] - d.Obs.Phi[i]) / te.error[i]
                    add_num_data_and_sum_residual(site_id, accum, res)
        else:
            for i in range(4):
                if d.Err.Phi[i] > 0.0:
                    add_num_data_and_sum_residual(site_id, accum, d.Res.Phi[i])

    # NMT
    for site_id, d in nmt_data_list:
        freq = d.freq
        if does_read_true_error:
            te = get_nmt_true_error(site_id, freq)
            for i in range(2):
                if te.error[i][0] > 0.0:
                    res = (d.Cal.Y[i].real - d.Obs.Y[i].real) / te.error[i][0]
                    add_num_data_and_sum_residual(site_id, accum, res)
                if te.error[i][1] > 0.0:
                    res = (d.Cal.Y[i].imag - d.Obs.Y[i].imag) / te.error[i][1]
                    add_num_data_and_sum_residual(site_id, accum, res)
        else:
            for i in range(2):
                if d.Err.Y[i].real > 0.0:
                    add_num_data_and_sum_residual(site_id, accum, d.Res.Y[i].real)
                if d.Err.Y[i].imag > 0.0:
                    add_num_data_and_sum_residual(site_id, accum, d.Res.Y[i].imag)

    # NMT2
    for site_id, d in nmt2_data_list:
        freq = d.freq
        if does_read_true_error:
            te = get_nmt2_true_error(site_id, freq)
            for i in range(4):
                if te.error[i][0] > 0.0:
                    res = (d.Cal.Z[i].real - d.Obs.Z[i].real) / te.error[i][0]
                    add_num_data_and_sum_residual(site_id, accum, res)
                if te.error[i][1] > 0.0:
                    res = (d.Cal.Z[i].imag - d.Obs.Z[i].imag) / te.error[i][1]
                    add_num_data_and_sum_residual(site_id, accum, res)
        else:
            for i in range(4):
                if d.Err.Z[i].real > 0.0:
                    add_num_data_and_sum_residual(site_id, accum, d.Res.Z[i].real)
                if d.Err.Z[i].imag > 0.0:
                    add_num_data_and_sum_residual(site_id, accum, d.Res.Z[i].imag)

    # APP_RES_AND_PHS
    for site_id, d in app_res_phs_data_list:
        freq = d.freq
        if does_read_true_error:
            te = get_app_res_phs_true_error(site_id, freq)
            for i in range(4):
                if te.error[i][0] > 0.0:
                    error = te.error[i][0] / d.Obs.apparentResistivity[i] / math.log(10)
                    res = math.log10(d.Cal.apparentResistivity[i] / d.Obs.apparentResistivity[i]) / error
                    add_num_data_and_sum_residual(site_id, accum, res)
                if te.error[i][1] > 0.0:
                    res = (d.Cal.phase[i] - d.Obs.phase[i]) / te.error[i][1]
                    add_num_data_and_sum_residual(site_id, accum, res)
        else:
            for i in range(4):
                if d.Err.apparentResistivity[i] > 0.0:
                    add_num_data_and_sum_residual(site_id, accum, d.Res.apparentResistivity[i])
                if d.Err.phase[i] > 0.0:
                    add_num_data_and_sum_residual(site_id, accum, d.Res.phase[i])

    # NMT2_APP_RES_AND_PHS
    for site_id, d in nmt2_app_res_phs_data_list:
        freq = d.freq
        if does_read_true_error:
            te = get_nmt2_app_res_phs_true_error(site_id, freq)
            for i in range(4):
                if te.error[i][0] > 0.0:
                    res = math.log10(d.Cal.apparentResistivity[i] / d.Obs.apparentResistivity[i]) / \
                          math.log10(1.0 + te.error[i][0] / d.Obs.apparentResistivity[i])
                    add_num_data_and_sum_residual(site_id, accum, res)
                if te.error[i][1] > 0.0:
                    res = (d.Cal.phase[i] - d.Obs.phase[i]) / te.error[i][1]
                    add_num_data_and_sum_residual(site_id, accum, res)
        else:
            for i in range(4):
                if d.Err.apparentResistivity[i] > 0.0:
                    add_num_data_and_sum_residual(site_id, accum, d.Res.apparentResistivity[i])
                if d.Err.phase[i] > 0.0:
                    add_num_data_and_sum_residual(site_id, accum, d.Res.phase[i])

    # Write RMS.out
    try:
        ofile = open("RMS.out", "w")
    except OSError:
        print("File open error : RMS.out !!", file=sys.stderr)
        sys.exit(1)

    ofile.write(f"{'Site':>10}{'#Data':>10}{'RMS':>15}\n")
    sum_residual_all = 0.0
    num_data_all = 0
    for sid in sorted(accum):
        nd  = accum[sid].numData
        sr  = accum[sid].sumResidual
        rms = math.sqrt(sr / nd)
        ofile.write(f"{sid:>10}{nd:>10}{rms:>15.6e}\n")
        sum_residual_all += sr
        num_data_all     += nd
    if num_data_all > 0:
        rms_total = math.sqrt(sum_residual_all / num_data_all)
    else:
        rms_total = 0.0
    ofile.write(f"{'Total':>10}{num_data_all:>10}{rms_total:>15.6e}\n")
    ofile.close()


# ===========================================================================
# Write functions
# ===========================================================================

def _open_output(stem: str) -> object:
    fname = f"{stem}.csv" if output_csv else f"{stem}.txt"
    try:
        return open(fname, "w")
    except OSError:
        print(f"File open error : {fname} !!", file=sys.stderr)
        sys.exit(1)


def _wf(f, width, value, sci=False):
    """Shorthand: write field, optionally scientific."""
    if sci and isinstance(value, float):
        write_field_sci(f, width, value, output_csv)
    else:
        write_field(f, width, value, output_csv)


def write_result_mt() -> None:
    if not mt_data_list:
        return

    mt_data_list.sort(key=lambda x: (x[0], x[1].freq))

    ofile = _open_output("result_MT")
    _wf(ofile, 10, "Site");  _wf(ofile, 15, "Frequency")

    if impedance_converted_to_app_res:
        for i in range(4):
            _wf(ofile, 15, f"AppR{COMP_INDEX[i]}Cal")
            _wf(ofile, 15, f"Phs{COMP_INDEX[i]}Cal")
        if type_of_distortion != NO_DISTORTION:
            for i in range(4):
                _wf(ofile, 15, f"AppR{COMP_INDEX[i]}Undist")
                _wf(ofile, 15, f"Phs{COMP_INDEX[i]}Undist")
        for i in range(4):
            _wf(ofile, 15, f"AppR{COMP_INDEX[i]}Obs")
            _wf(ofile, 15, f"Phs{COMP_INDEX[i]}Obs")
        for i in range(4):
            _wf(ofile, 15, f"AppR{COMP_INDEX[i]}Err")
            _wf(ofile, 15, f"Phs{COMP_INDEX[i]}Err")
    else:
        for i in range(4):
            _wf(ofile, 15, f"ReZ{COMP_INDEX[i]}Cal")
            _wf(ofile, 15, f"ImZ{COMP_INDEX[i]}Cal")
        if type_of_distortion != NO_DISTORTION:
            for i in range(4):
                _wf(ofile, 15, f"ReZ{COMP_INDEX[i]}Undist")
                _wf(ofile, 15, f"ImZ{COMP_INDEX[i]}Undist")
        for i in range(4):
            _wf(ofile, 15, f"ReZ{COMP_INDEX[i]}Obs")
            _wf(ofile, 15, f"ImZ{COMP_INDEX[i]}Obs")
        for i in range(4):
            _wf(ofile, 15, f"ReZ{COMP_INDEX[i]}Err")
            _wf(ofile, 15, f"ImZ{COMP_INDEX[i]}Err")
    ofile.write("\n")

    for site_id, d in mt_data_list:
        freq  = d.freq
        omega = 2.0 * math.pi * freq
        Z_undist = calc_undistorted_impedance_tensor(site_id, d.Cal)

        app_cal      = [0.0]*4; phs_cal      = [0.0]*4
        app_obs      = [0.0]*4; phs_obs      = [0.0]*4
        app_err      = [0.0]*4; phs_err      = [0.0]*4
        app_cal_und  = [0.0]*4; phs_cal_und  = [0.0]*4

        for i in range(4):
            if read_true_error_file:
                te = get_mt_true_error(site_id, freq)
                is_same_error(site_id, freq, i, te.error[i][0], te.error[i][1])
                if te.error[i][0] > 0.0 and te.error[i][1] > 0.0:
                    error = max(te.error[i][0], te.error[i][1])
                else:
                    error = 1.0e10
            else:
                is_same_error(site_id, freq, i, d.Err.Z[i].real, d.Err.Z[i].imag)
                error = max(d.Err.Z[i].real, d.Err.Z[i].imag)
            error *= FACTOR

            absZ_cal = abs(d.Cal.Z[i]); absZ_obs = abs(d.Obs.Z[i])
            app_cal[i]     = absZ_cal**2 / (omega * MU0)
            app_obs[i]     = absZ_obs**2 / (omega * MU0)
            app_err[i]     = 2.0 * absZ_obs * error / (omega * MU0)
            phs_cal[i]     = RAD2DEG * math.atan2(d.Cal.Z[i].imag, d.Cal.Z[i].real)
            phs_obs[i]     = RAD2DEG * math.atan2(d.Obs.Z[i].imag, d.Obs.Z[i].real)
            tmp = error / absZ_obs if absZ_obs > 0 else 1e10
            phs_err[i]     = RAD2DEG * math.asin(tmp) if tmp <= 1.0 else 180.0
            app_cal_und[i] = abs(Z_undist.Z[i])**2 / (omega * MU0)
            phs_cal_und[i] = RAD2DEG * math.atan2(Z_undist.Z[i].imag, Z_undist.Z[i].real)

        _wf(ofile, 10, convert_site_id_to_name(site_id))
        _wf(ofile, 15, freq, sci=True)
        if impedance_converted_to_app_res:
            for i in range(4): _wf(ofile,15,app_cal[i],sci=True); _wf(ofile,15,phs_cal[i],sci=True)
            if type_of_distortion != NO_DISTORTION:
                for i in range(4): _wf(ofile,15,app_cal_und[i],sci=True); _wf(ofile,15,phs_cal_und[i],sci=True)
            for i in range(4): _wf(ofile,15,app_obs[i],sci=True); _wf(ofile,15,phs_obs[i],sci=True)
            for i in range(4): _wf(ofile,15,app_err[i],sci=True); _wf(ofile,15,phs_err[i],sci=True)
        else:
            for i in range(4): _wf(ofile,15,d.Cal.Z[i].real,sci=True); _wf(ofile,15,d.Cal.Z[i].imag,sci=True)
            if type_of_distortion != NO_DISTORTION:
                for i in range(4): _wf(ofile,15,Z_undist.Z[i].real,sci=True); _wf(ofile,15,Z_undist.Z[i].imag,sci=True)
            for i in range(4): _wf(ofile,15,d.Obs.Z[i].real,sci=True); _wf(ofile,15,d.Obs.Z[i].imag,sci=True)
            for i in range(4): _wf(ofile,15,d.Err.Z[i].real,sci=True); _wf(ofile,15,d.Err.Z[i].imag,sci=True)
        ofile.write("\n")
    ofile.close()


def write_result_vtf() -> None:
    if not vtf_data_list:
        return
    vtf_data_list.sort(key=lambda x: (x[0], x[1].freq))
    ofile = _open_output("result_VTF")
    _wf(ofile, 10, "Site"); _wf(ofile, 15, "Frequency")
    for i in range(2):
        _wf(ofile, 15, f"ReT{COMP_INDEX[i+4]}Cal"); _wf(ofile, 15, f"ImT{COMP_INDEX[i+4]}Cal")
    for i in range(2):
        _wf(ofile, 15, f"ReT{COMP_INDEX[i+4]}Obs"); _wf(ofile, 15, f"ImT{COMP_INDEX[i+4]}Obs")
    for i in range(2):
        _wf(ofile, 15, f"ReT{COMP_INDEX[i+4]}Err"); _wf(ofile, 15, f"ImT{COMP_INDEX[i+4]}Err")
    ofile.write("\n")

    for site_id, d in vtf_data_list:
        freq = d.freq
        re_err = [0.0]*2; im_err = [0.0]*2
        for i in range(2):
            if read_true_error_file:
                te = get_vtf_true_error(site_id, freq)
                is_same_error(site_id, freq, i, te.error[i][0], te.error[i][1])
                error = max(te.error[i][0], te.error[i][1]) if (te.error[i][0]>0 and te.error[i][1]>0) else 1e10
            else:
                is_same_error(site_id, freq, i, d.Err.TZ[i].real, d.Err.TZ[i].imag)
                error = max(d.Err.TZ[i].real, d.Err.TZ[i].imag)
            error *= FACTOR
            re_err[i] = im_err[i] = error
        _wf(ofile, 10, convert_site_id_to_name(site_id)); _wf(ofile, 15, freq, sci=True)
        for i in range(2): _wf(ofile,15,d.Cal.TZ[i].real,sci=True); _wf(ofile,15,d.Cal.TZ[i].imag,sci=True)
        for i in range(2): _wf(ofile,15,d.Obs.TZ[i].real,sci=True); _wf(ofile,15,d.Obs.TZ[i].imag,sci=True)
        for i in range(2): _wf(ofile,15,re_err[i],sci=True); _wf(ofile,15,im_err[i],sci=True)
        ofile.write("\n")
    ofile.close()


def write_result_htf() -> None:
    if not htf_data_list:
        return
    htf_data_list.sort(key=lambda x: (x[0], x[1].freq))
    ofile = _open_output("result_HTF")
    _wf(ofile, 10, "Site"); _wf(ofile, 15, "Frequency")
    for i in range(4):
        _wf(ofile, 15, f"ReT{COMP_INDEX[i]}Cal"); _wf(ofile, 15, f"ImT{COMP_INDEX[i]}Cal")
    for i in range(4):
        _wf(ofile, 15, f"ReT{COMP_INDEX[i]}Obs"); _wf(ofile, 15, f"ImT{COMP_INDEX[i]}Obs")
    for i in range(4):
        _wf(ofile, 15, f"ReT{COMP_INDEX[i]}Err"); _wf(ofile, 15, f"ImT{COMP_INDEX[i]}Err")
    ofile.write("\n")

    for site_id, d in htf_data_list:
        freq = d.freq
        re_err = [0.0]*4; im_err = [0.0]*4
        for i in range(4):
            if read_true_error_file:
                te = get_htf_true_error(site_id, freq)
                is_same_error(site_id, freq, i, te.error[i][0], te.error[i][1])
                error = max(te.error[i][0], te.error[i][1]) if (te.error[i][0]>0 and te.error[i][1]>0) else 1e10
            else:
                is_same_error(site_id, freq, i, d.Err.T[i].real, d.Err.T[i].imag)
                error = max(d.Err.T[i].real, d.Err.T[i].imag)
            error *= FACTOR
            re_err[i] = im_err[i] = error
        _wf(ofile, 10, convert_site_id_to_name(site_id)); _wf(ofile, 15, freq, sci=True)
        for i in range(4): _wf(ofile,15,d.Cal.T[i].real,sci=True); _wf(ofile,15,d.Cal.T[i].imag,sci=True)
        for i in range(4): _wf(ofile,15,d.Obs.T[i].real,sci=True); _wf(ofile,15,d.Obs.T[i].imag,sci=True)
        for i in range(4): _wf(ofile,15,re_err[i],sci=True); _wf(ofile,15,im_err[i],sci=True)
        ofile.write("\n")
    ofile.close()


def write_result_pt() -> None:
    if not pt_data_list:
        return
    pt_data_list.sort(key=lambda x: (x[0], x[1].freq))
    ofile = _open_output("result_PT")
    _wf(ofile, 10, "Site"); _wf(ofile, 15, "Frequency")
    for i in range(4): _wf(ofile, 15, f"Phi{COMP_INDEX_PT[i]}Cal")
    for i in range(4): _wf(ofile, 15, f"Phi{COMP_INDEX_PT[i]}Obs")
    for i in range(4): _wf(ofile, 15, f"Phi{COMP_INDEX_PT[i]}Err")
    ofile.write("\n")

    for site_id, d in pt_data_list:
        freq = d.freq
        err = [0.0]*4
        for i in range(4):
            if read_true_error_file:
                te = get_pt_true_error(site_id, freq)
                error = te.error[i] if te.error[i] else 1e10
            else:
                error = d.Err.Phi[i]
            err[i] = error * FACTOR
        _wf(ofile, 10, convert_site_id_to_name(site_id)); _wf(ofile, 15, freq, sci=True)
        for i in range(4): _wf(ofile, 15, d.Cal.Phi[i], sci=True)
        for i in range(4): _wf(ofile, 15, d.Obs.Phi[i], sci=True)
        for i in range(4): _wf(ofile, 15, err[i], sci=True)
        ofile.write("\n")
    ofile.close()


def write_result_nmt() -> None:
    if not nmt_data_list:
        return
    nmt_data_list.sort(key=lambda x: (x[0], x[1].freq))
    ofile = _open_output("result_NMT")
    _wf(ofile, 10, "Site"); _wf(ofile, 15, "Frequency")
    for i in range(2):
        _wf(ofile, 15, f"ReY{COMP_INDEX_XY[i]}Cal"); _wf(ofile, 15, f"ImY{COMP_INDEX_XY[i]}Cal")
    for i in range(2):
        _wf(ofile, 15, f"ReY{COMP_INDEX_XY[i]}Obs"); _wf(ofile, 15, f"ImY{COMP_INDEX_XY[i]}Obs")
    for i in range(2):
        _wf(ofile, 15, f"ReY{COMP_INDEX_XY[i]}Err"); _wf(ofile, 15, f"ImY{COMP_INDEX_XY[i]}Err")
    ofile.write("\n")

    for site_id, d in nmt_data_list:
        freq = d.freq
        re_err = [0.0]*2; im_err = [0.0]*2
        for i in range(2):
            if read_true_error_file:
                te = get_nmt_true_error(site_id, freq)
                is_same_error(site_id, freq, i, te.error[i][0], te.error[i][1])
                error = max(te.error[i][0], te.error[i][1]) if (te.error[i][0]>0 and te.error[i][1]>0) else 1e10
            else:
                is_same_error(site_id, freq, i, d.Err.Y[i].real, d.Err.Y[i].imag)
                error = max(d.Err.Y[i].real, d.Err.Y[i].imag)
            error *= FACTOR
            re_err[i] = im_err[i] = error
        _wf(ofile, 10, convert_site_id_to_name(site_id)); _wf(ofile, 15, freq, sci=True)
        for i in range(2): _wf(ofile,15,d.Cal.Y[i].real,sci=True); _wf(ofile,15,d.Cal.Y[i].imag,sci=True)
        for i in range(2): _wf(ofile,15,d.Obs.Y[i].real,sci=True); _wf(ofile,15,d.Obs.Y[i].imag,sci=True)
        for i in range(2): _wf(ofile,15,re_err[i],sci=True); _wf(ofile,15,im_err[i],sci=True)
        ofile.write("\n")
    ofile.close()


def write_result_nmt2() -> None:
    if not nmt2_data_list:
        return
    nmt2_data_list.sort(key=lambda x: (x[0], x[1].freq))
    ofile = _open_output("result_NMT2")
    _wf(ofile, 10, "Site"); _wf(ofile, 15, "Frequency")
    if impedance_converted_to_app_res:
        for i in range(4): _wf(ofile,15,f"AppR{COMP_INDEX[i]}Cal"); _wf(ofile,15,f"Phs{COMP_INDEX[i]}Cal")
        for i in range(4): _wf(ofile,15,f"AppR{COMP_INDEX[i]}Obs"); _wf(ofile,15,f"Phs{COMP_INDEX[i]}Obs")
        for i in range(4): _wf(ofile,15,f"AppR{COMP_INDEX[i]}Err"); _wf(ofile,15,f"Phs{COMP_INDEX[i]}Err")
    else:
        for i in range(4): _wf(ofile,15,f"ReZ{COMP_INDEX[i]}Cal"); _wf(ofile,15,f"ImZ{COMP_INDEX[i]}Cal")
        for i in range(4): _wf(ofile,15,f"ReZ{COMP_INDEX[i]}Obs"); _wf(ofile,15,f"ImZ{COMP_INDEX[i]}Obs")
        for i in range(4): _wf(ofile,15,f"ReZ{COMP_INDEX[i]}Err"); _wf(ofile,15,f"ImZ{COMP_INDEX[i]}Err")
    ofile.write("\n")

    for site_id, d in nmt2_data_list:
        freq = d.freq
        omega = 2.0 * math.pi * freq
        app_cal=[0.]*4; phs_cal=[0.]*4; app_obs=[0.]*4; phs_obs=[0.]*4; app_err=[0.]*4; phs_err=[0.]*4
        for i in range(4):
            if read_true_error_file:
                te = get_nmt2_true_error(site_id, freq)
                is_same_error(site_id, freq, i, te.error[i][0], te.error[i][1])
                error = max(te.error[i][0], te.error[i][1]) if (te.error[i][0]>0 and te.error[i][1]>0) else 1e10
            else:
                is_same_error(site_id, freq, i, d.Err.Z[i].real, d.Err.Z[i].imag)
                error = max(d.Err.Z[i].real, d.Err.Z[i].imag)
            error *= FACTOR
            absZ_cal = abs(d.Cal.Z[i]); absZ_obs = abs(d.Obs.Z[i])
            app_cal[i] = absZ_cal**2 / (omega * MU0)
            app_obs[i] = absZ_obs**2 / (omega * MU0)
            app_err[i] = 2.0 * absZ_obs * error / (omega * MU0)
            phs_cal[i] = RAD2DEG * math.atan2(d.Cal.Z[i].imag, d.Cal.Z[i].real)
            phs_obs[i] = RAD2DEG * math.atan2(d.Obs.Z[i].imag, d.Obs.Z[i].real)
            tmp = error / absZ_obs if absZ_obs > 0 else 1e10
            phs_err[i] = RAD2DEG * math.asin(tmp) if tmp <= 1.0 else 180.0
        _wf(ofile, 10, convert_site_id_to_name(site_id)); _wf(ofile, 15, freq, sci=True)
        if impedance_converted_to_app_res:
            for i in range(4): _wf(ofile,15,app_cal[i],sci=True); _wf(ofile,15,phs_cal[i],sci=True)
            for i in range(4): _wf(ofile,15,app_obs[i],sci=True); _wf(ofile,15,phs_obs[i],sci=True)
            for i in range(4): _wf(ofile,15,app_err[i],sci=True); _wf(ofile,15,phs_err[i],sci=True)
        else:
            for i in range(4): _wf(ofile,15,d.Cal.Z[i].real,sci=True); _wf(ofile,15,d.Cal.Z[i].imag,sci=True)
            for i in range(4): _wf(ofile,15,d.Obs.Z[i].real,sci=True); _wf(ofile,15,d.Obs.Z[i].imag,sci=True)
            for i in range(4): _wf(ofile,15,d.Err.Z[i].real,sci=True); _wf(ofile,15,d.Err.Z[i].imag,sci=True)
        ofile.write("\n")
    ofile.close()


def write_result_app_res_and_phs() -> None:
    if not app_res_phs_data_list:
        return
    app_res_phs_data_list.sort(key=lambda x: (x[0], x[1].freq))
    ofile = _open_output("result_APP_RES_AND_PHS")
    _wf(ofile, 10, "Site"); _wf(ofile, 15, "Frequency")
    for i in range(4): _wf(ofile,15,f"AppR{COMP_INDEX[i]}Cal"); _wf(ofile,15,f"Phs{COMP_INDEX[i]}Cal")
    if type_of_distortion != NO_DISTORTION:
        for i in range(4): _wf(ofile,15,f"AppR{COMP_INDEX[i]}Undist"); _wf(ofile,15,f"Phs{COMP_INDEX[i]}Undist")
    for i in range(4): _wf(ofile,15,f"AppR{COMP_INDEX[i]}Obs"); _wf(ofile,15,f"Phs{COMP_INDEX[i]}Obs")
    for i in range(4): _wf(ofile,15,f"AppR{COMP_INDEX[i]}Err"); _wf(ofile,15,f"Phs{COMP_INDEX[i]}Err")
    ofile.write("\n")

    for site_id, d in app_res_phs_data_list:
        freq = d.freq
        undist = calc_undistorted_app_res_and_phase(site_id, freq, d.Cal)
        app_cal=[0.]*4; phs_cal=[0.]*4; app_und=[0.]*4; phs_und=[0.]*4
        app_obs=[0.]*4; phs_obs=[0.]*4; app_err=[0.]*4; phs_err=[0.]*4
        for i in range(4):
            if read_true_error_file:
                te = get_app_res_phs_true_error(site_id, freq)
                err_app = te.error[i][0] if te.error[i][0] > 0 else 1e10
                err_phs = te.error[i][1] if te.error[i][1] > 0 else 1e10
            else:
                err_app = d.Err.apparentResistivity[i]
                err_phs = d.Err.phase[i]
            app_cal[i] = d.Cal.apparentResistivity[i]
            app_obs[i] = d.Obs.apparentResistivity[i]
            app_err[i] = err_app * FACTOR
            phs_cal[i] = d.Cal.phase[i]
            phs_obs[i] = d.Obs.phase[i]
            phs_err[i] = min(err_phs * FACTOR, 180.0)
            app_und[i] = undist.apparentResistivity[i]
            phs_und[i] = undist.phase[i]
        _wf(ofile, 10, convert_site_id_to_name(site_id)); _wf(ofile, 15, freq, sci=True)
        for i in range(4): _wf(ofile,15,app_cal[i],sci=True); _wf(ofile,15,phs_cal[i],sci=True)
        if type_of_distortion != NO_DISTORTION:
            for i in range(4): _wf(ofile,15,app_und[i],sci=True); _wf(ofile,15,phs_und[i],sci=True)
        for i in range(4): _wf(ofile,15,app_obs[i],sci=True); _wf(ofile,15,phs_obs[i],sci=True)
        for i in range(4): _wf(ofile,15,app_err[i],sci=True); _wf(ofile,15,phs_err[i],sci=True)
        ofile.write("\n")
    ofile.close()


def write_result_nmt2_app_res_and_phs() -> None:
    if not nmt2_app_res_phs_data_list:
        return
    nmt2_app_res_phs_data_list.sort(key=lambda x: (x[0], x[1].freq))
    ofile = _open_output("result_NMT2_APP_RES_AND_PHS")
    _wf(ofile, 10, "Site"); _wf(ofile, 15, "Frequency")
    for i in range(4): _wf(ofile,15,f"AppR{COMP_INDEX[i]}Cal"); _wf(ofile,15,f"Phs{COMP_INDEX[i]}Cal")
    for i in range(4): _wf(ofile,15,f"AppR{COMP_INDEX[i]}Obs"); _wf(ofile,15,f"Phs{COMP_INDEX[i]}Obs")
    for i in range(4): _wf(ofile,15,f"AppR{COMP_INDEX[i]}Err"); _wf(ofile,15,f"Phs{COMP_INDEX[i]}Err")
    ofile.write("\n")

    for site_id, d in nmt2_app_res_phs_data_list:
        freq = d.freq
        app_cal=[0.]*4; phs_cal=[0.]*4; app_obs=[0.]*4; phs_obs=[0.]*4; app_err=[0.]*4; phs_err=[0.]*4
        for i in range(4):
            if read_true_error_file:
                te = get_nmt2_app_res_phs_true_error(site_id, freq)
                err_app = te.error[i][0] if te.error[i][0] > 0 else 1e10
                err_phs = te.error[i][1] if te.error[i][1] > 0 else 1e10
            else:
                err_app = d.Err.apparentResistivity[i]
                err_phs = d.Err.phase[i]
            app_cal[i] = d.Cal.apparentResistivity[i]
            app_obs[i] = d.Obs.apparentResistivity[i]
            app_err[i] = err_app * FACTOR
            phs_cal[i] = d.Cal.phase[i]
            phs_obs[i] = d.Obs.phase[i]
            phs_err[i] = min(err_phs * FACTOR, 180.0)
        _wf(ofile, 10, convert_site_id_to_name(site_id)); _wf(ofile, 15, freq, sci=True)
        for i in range(4): _wf(ofile,15,app_cal[i],sci=True); _wf(ofile,15,phs_cal[i],sci=True)
        for i in range(4): _wf(ofile,15,app_obs[i],sci=True); _wf(ofile,15,phs_obs[i],sci=True)
        for i in range(4): _wf(ofile,15,app_err[i],sci=True); _wf(ofile,15,phs_err[i],sci=True)
        ofile.write("\n")
    ofile.close()


def write_result() -> None:
    write_result_mt()
    write_result_vtf()
    write_result_htf()
    write_result_pt()
    write_result_nmt()
    write_result_nmt2()
    write_result_app_res_and_phs()
    write_result_nmt2_app_res_and_phs()




# ===========================================================================
# MT Sounding-curve plotting  (requires numpy, pandas, pygmt)
# ===========================================================================
#
# Design: works entirely from the in-memory data structures populated by
# read_result() -- no intermediate text file is read.
#
# Data sources used:
#   mt_data_list          : List[(site_id, MTData)]
#   app_res_phs_data_list : List[(site_id, ApparentResistivityAndPhaseData)]
#   site_id_to_name       : Dict[int, str]
#   impedance_converted_to_app_res : bool  (global flag)
#
# When the data are raw impedance (impedance_converted_to_app_res == False)
# the same AppRes / Phase arithmetic as write_result_mt() is applied here
# so that the plots are always in AppRes / Phase space.

# --- Plot style constants (matching the original shell script) ---
_PLT_COMPS = ["xx", "xy", "yx", "yy"]
_PLT_COLOR = {
    "xx": "255/128/0",   # orange
    "xy": "255/0/0",     # red
    "yx": "0/0/255",     # blue
    "yy": "0/255/0",     # green
}
_RHO_REGION = [1e-4, 1e4, 1e-1, 1e4]   # [T_min, T_max, rho_min, rho_max]
_PHS_REGION = [1e-4, 1e4, -180, 180]
_RHO_W      = 3.8    # cm – panel width
_RHO_H      = 3.0    # cm – apparent resistivity panel height
_PHS_H      = 3.0    # cm – phase panel height
_LINE_PEN   = "0.5p"
_SYM_SIZE   = "0.15c"
_ERR_CAP    = "0.15c"
_RHO_TO_PHS = _RHO_H + 0.5          # vertical shift: rho top → phase top
_PANEL_DX   = _RHO_W + 0.5          # horizontal column pitch
_PANEL_DY   = _RHO_TO_PHS + _PHS_H + 0.7   # full row height


def _plt_rho_frame(left: bool) -> list:
    w = "W" if left else "w"
    return [f"{w}sne",
            "xa1f3+lLog(Period,s)",
            "ya1f3g1+lLog(App. Resistivity,@~W@~m)"]


def _plt_phs_frame(left: bool, bottom: bool) -> list:
    w = "W" if left   else "w"
    s = "S" if bottom else "s"
    return [f"{w}{s}ne",
            "xa1f3+lLog(Period,s)",
            "ya45f10g45+lPhase,deg."]


def _mt_to_appphs_arrays(site_id: int, records: list) -> dict:
    """
    Convert a list of MTData records for one site into numpy arrays of
    apparent resistivity, phase, and their errors -- using the same
    arithmetic as write_result_mt() / write_result_nmt2().

    Returns a dict with keys:
        period,
        AppR{xx,xy,yx,yy}Cal, Phs{xx,xy,yx,yy}Cal,
        AppR{xx,xy,yx,yy}Obs, Phs{xx,xy,yx,yy}Obs,
        AppR{xx,xy,yx,yy}Err, Phs{xx,xy,yx,yy}Err
    """
    records = sorted(records, key=lambda d: d.freq)
    n = len(records)

    freq   = np.array([d.freq for d in records])
    omega  = 2.0 * math.pi * freq
    period = 1.0 / freq

    out = {"period": period}

    comp_map = {"xx": XX, "xy": XY, "yx": YX, "yy": YY}
    for c, k in comp_map.items():
        rho_cal = np.empty(n); phs_cal = np.empty(n)
        rho_obs = np.empty(n); phs_obs = np.empty(n)
        rho_err = np.empty(n); phs_err = np.empty(n)

        for j, d in enumerate(records):
            om = omega[j]

            absC = abs(d.Cal.Z[k])
            absO = abs(d.Obs.Z[k])

            # error: max(re_err, im_err) -- same convention as femtic_data.py
            err = max(d.Err.Z[k].real, d.Err.Z[k].imag)
            err *= FACTOR

            rho_cal[j] = absC**2 / (om * MU0)
            phs_cal[j] = RAD2DEG * math.atan2(d.Cal.Z[k].imag, d.Cal.Z[k].real)
            rho_obs[j] = absO**2 / (om * MU0)
            phs_obs[j] = RAD2DEG * math.atan2(d.Obs.Z[k].imag, d.Obs.Z[k].real)
            rho_err[j] = 2.0 * absO * err / (om * MU0)
            ratio = err / absO if absO > 0 else 10.0
            phs_err[j] = RAD2DEG * math.asin(ratio) if ratio <= 1.0 else 180.0

        out[f"AppR{c}Cal"] = rho_cal; out[f"Phs{c}Cal"] = phs_cal
        out[f"AppR{c}Obs"] = rho_obs; out[f"Phs{c}Obs"] = phs_obs
        out[f"AppR{c}Err"] = rho_err; out[f"Phs{c}Err"] = phs_err

    return out


def _appphs_to_arrays(site_id: int, records: list) -> dict:
    """
    Convert a list of ApparentResistivityAndPhaseData records for one site
    into the same array dict format as _mt_to_appphs_arrays().
    """
    records = sorted(records, key=lambda d: d.freq)
    n = len(records)

    period = np.array([1.0 / d.freq for d in records])
    out = {"period": period}

    comp_map = {"xx": XX, "xy": XY, "yx": YX, "yy": YY}
    for c, k in comp_map.items():
        out[f"AppR{c}Cal"] = np.array([d.Cal.apparentResistivity[k] for d in records])
        out[f"Phs{c}Cal"]  = np.array([d.Cal.phase[k]               for d in records])
        out[f"AppR{c}Obs"] = np.array([d.Obs.apparentResistivity[k] for d in records])
        out[f"Phs{c}Obs"]  = np.array([d.Obs.phase[k]               for d in records])

        err_r = np.array([d.Err.apparentResistivity[k] * FACTOR for d in records])
        err_p = np.array([min(d.Err.phase[k] * FACTOR, 180.0)   for d in records])
        out[f"AppR{c}Err"] = err_r
        out[f"Phs{c}Err"]  = err_p

    return out


def _plot_station(fig, arrays: dict, label: str,
                  show_left: bool, show_bottom: bool) -> None:
    """
    Draw the apparent-resistivity panel (at current origin) and the phase
    panel (shifted down by _RHO_TO_PHS cm).  Origin is restored to the
    rho-panel top before returning.
    """
    period   = arrays["period"]
    rho_proj = f"X{_RHO_W}l/{_RHO_H}l"
    phs_proj = f"X{_RHO_W}l/{_PHS_H}"

    # ------------------------------------------------------------------ rho --
    frame_todo = _plt_rho_frame(show_left)

    for c in _PLT_COMPS:
        col = _PLT_COLOR[c]

        rho_mod = arrays[f"AppR{c}Cal"]
        mask = rho_mod > 0
        if mask.any():
            fig.plot(x=period[mask], y=rho_mod[mask],
                     region=_RHO_REGION, projection=rho_proj,
                     frame=frame_todo, pen=f"{_LINE_PEN},{col}")
            frame_todo = None

        rho_obs = arrays[f"AppR{c}Obs"]
        err_rho = arrays[f"AppR{c}Err"]
        valid = rho_obs > 0
        if valid.any():
            fig.plot(x=period[valid], y=rho_obs[valid],
                     region=_RHO_REGION, projection=rho_proj,
                     style=f"c{_SYM_SIZE}", pen=f"{_LINE_PEN},{col}",
                     no_clip=True)
            fig.plot(
                data=pd.DataFrame({"x": period[valid],
                                   "y": rho_obs[valid],
                                   "ey": err_rho[valid]}),
                region=_RHO_REGION, projection=rho_proj,
                style=f"ey{_ERR_CAP}/0.3p", pen=f"{_LINE_PEN},{col}",
                no_clip=True)

    if frame_todo is not None:
        fig.basemap(region=_RHO_REGION, projection=rho_proj,
                    frame=frame_todo)

    # site label (top-left inside rho panel)
    fig.text(x=period.min(), y=_RHO_REGION[3] * 0.6,
             text=label,
             region=_RHO_REGION, projection=rho_proj,
             justify="TL", font="8p,Helvetica,black",
             no_clip=True, offset="0.08c/-0.05c")

    # --------------------------------------------------------------- phase --
    fig.shift_origin(yshift=f"-{_RHO_TO_PHS}c")
    frame_todo = _plt_phs_frame(show_left, show_bottom)

    for c in _PLT_COMPS:
        col = _PLT_COLOR[c]

        # sign-flipped phase (same convention as original shell script)
        phs_mod = -arrays[f"Phs{c}Cal"]
        fig.plot(x=period, y=phs_mod,
                 region=_PHS_REGION, projection=phs_proj,
                 frame=frame_todo, pen=f"{_LINE_PEN},{col}")
        frame_todo = None

        phs_obs = -arrays[f"Phs{c}Obs"]
        err_phs = arrays[f"Phs{c}Err"]
        fig.plot(x=period, y=phs_obs,
                 region=_PHS_REGION, projection=phs_proj,
                 style=f"c{_SYM_SIZE}", pen=f"{_LINE_PEN},{col}",
                 no_clip=True)
        fig.plot(
            data=pd.DataFrame({"x": period,
                               "y": phs_obs,
                               "ey": err_phs}),
            region=_PHS_REGION, projection=phs_proj,
            style=f"ey{_ERR_CAP}/0.3p", pen=f"{_LINE_PEN},{col}",
            no_clip=True)

    # restore origin to rho-panel top
    fig.shift_origin(yshift=f"{_RHO_TO_PHS}c")


def _load_sites_file(path: str) -> list:
    """Read optional two-column site-list file → [(site_name, label), ...]."""
    pairs = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            name  = parts[0]
            label = parts[1] if len(parts) >= 2 else name
            pairs.append((name, label))
    return pairs


def plot_mt_sounding_curves(
    sites_file: "Optional[str]" = None,
    out_prefix: str = "imp_all_curv",
    per_page: int = 12,
    n_cols: int = 6,
) -> None:
    """
    Plot apparent resistivity and phase sounding curves for all MT (and
    APP_RES_AND_PHS) stations.

    Data are read directly from the module-level lists populated by
    read_result().  No intermediate text files are touched.

    Parameters
    ----------
    sites_file : optional path to a two-column whitespace-separated file
                 mapping site names/IDs to display labels (plotting order
                 follows the file order).  If None, all MT sites are
                 plotted in the order they appear in mt_data_list /
                 app_res_phs_data_list.
    out_prefix : prefix for output PDF filenames  (e.g. "imp_all_curv"
                 → "imp_all_curv_1.pdf", "imp_all_curv_2.pdf", …)
    per_page   : number of stations per output PDF
    n_cols     : number of panel columns per page
    """
    # ----------------------------------------------------------------
    # 1. Build per-site arrays from in-memory data
    # ----------------------------------------------------------------
    # Collect MT impedance records grouped by site_id
    mt_by_site: Dict[int, list] = {}
    for sid, d in mt_data_list:
        mt_by_site.setdefault(sid, []).append(d)

    # Collect APP_RES_AND_PHS records grouped by site_id
    arp_by_site: Dict[int, list] = {}
    for sid, d in app_res_phs_data_list:
        arp_by_site.setdefault(sid, []).append(d)

    # Build the master dict: site_id → array-dict
    arrays_by_id: Dict[int, dict] = {}
    for sid, records in mt_by_site.items():
        if impedance_converted_to_app_res:
            # The write path would have computed AppRes; replicate it here.
            # Re-use the same helper used for the raw-impedance case because
            # the data in mt_data_list are always the raw complex Z values
            # regardless of the -appphs flag.
            pass
        arrays_by_id[sid] = _mt_to_appphs_arrays(sid, records)

    for sid, records in arp_by_site.items():
        if sid not in arrays_by_id:   # prefer MT if site appears in both
            arrays_by_id[sid] = _appphs_to_arrays(sid, records)

    if not arrays_by_id:
        print("plot_mt_sounding_curves: no MT or APP_RES_AND_PHS data to plot.",
              file=sys.stderr)
        return

    # ----------------------------------------------------------------
    # 2. Determine plotting order and labels
    # ----------------------------------------------------------------
    if sites_file:
        raw_pairs = _load_sites_file(sites_file)
        # site_name in the file may be either the numeric ID string or the
        # mapped name string; resolve to integer site_id
        name_to_id = {v: k for k, v in site_id_to_name.items()}
        site_pairs = []   # [(site_id, label), ...]
        for name, label in raw_pairs:
            # try direct int first, then look up via name mapping
            try:
                sid = int(name)
            except ValueError:
                sid = name_to_id.get(name)
            if sid is None or sid not in arrays_by_id:
                print(f"  WARNING: site '{name}' not found in data, skipping.",
                      file=sys.stderr)
                continue
            site_pairs.append((sid, label))
    else:
        # Use the order sites first appear in mt_data_list, then arp list
        seen = {}
        for sid, _ in mt_data_list:
            if sid not in seen:
                seen[sid] = convert_site_id_to_name(sid)
        for sid, _ in app_res_phs_data_list:
            if sid not in seen:
                seen[sid] = convert_site_id_to_name(sid)
        site_pairs = [(sid, lbl) for sid, lbl in seen.items()
                      if sid in arrays_by_id]

    n_stations = len(site_pairs)
    if n_stations == 0:
        print("plot_mt_sounding_curves: no plottable stations.", file=sys.stderr)
        return

    n_rows = math.ceil(per_page / n_cols)

    # ----------------------------------------------------------------
    # 3. GMT global settings
    # ----------------------------------------------------------------
    pygmt.config(FONT_TITLE="14p", FONT_LABEL="12p", MAP_TICK_LENGTH="0.1c")

    fig          = None
    current_page = -1

    for j, (site_id, label) in enumerate(site_pairs):
        page_idx    = j // per_page
        idx_on_page = j % per_page
        col_idx     = idx_on_page % n_cols
        row_idx     = idx_on_page // n_cols

        site_name = convert_site_id_to_name(site_id)
        print(f"  Plotting station {j+1}/{n_stations}: "
              f"id={site_id} name={site_name!r} label={label!r} "
              f"page={page_idx+1} row={row_idx} col={col_idx}")

        # ---- new page ----
        if page_idx != current_page:
            if fig is not None:
                out_path = f"{out_prefix}_{current_page + 1}.pdf"
                fig.savefig(out_path, dpi=300)
                print(f"  Saved {out_path}")
            fig = pygmt.Figure()
            current_page = page_idx
            fig.shift_origin(xshift="3c",
                             yshift=f"{3.0 + n_rows * _PANEL_DY}c")

        # ---- shift to this panel's top-left corner ----
        if idx_on_page == 0:
            pass                          # already positioned by page init
        elif col_idx == 0:
            fig.shift_origin(
                xshift=f"-{(n_cols - 1) * _PANEL_DX}c",
                yshift=f"-{_PANEL_DY}c",
            )
        else:
            fig.shift_origin(xshift=f"{_PANEL_DX}c")

        arrays = arrays_by_id[site_id]
        _plot_station(fig, arrays, label,
                      show_left=(col_idx == 0),
                      show_bottom=(row_idx == n_rows - 1))

    # ---- save last page ----
    if fig is not None:
        out_path = f"{out_prefix}_{current_page + 1}.pdf"
        fig.savefig(out_path, dpi=300)
        print(f"  Saved {out_path}")

    print("Plotting done.")



# ===========================================================================
# VTF (tipper) Sounding-curve plotting  (requires numpy, pandas, pygmt)
# ===========================================================================
#
# Design: works entirely from vtf_data_list populated by read_result().
#
# result_VTF.txt column layout (written by write_result_vtf):
#   Site  Frequency
#   ReTzxCal ImTzxCal  ReTzyCal ImTzyCal     ← calibrated (model)
#   ReTzxObs ImTzxObs  ReTzyObs ImTzyObs     ← observed
#   ReTzxErr ImTzxErr  ReTzyErr ImTzyErr      ← errors (re==im per component)
#
# Each station panel layout (matches shell script exactly):
#   TOP    : Tzx panel  — Re(Tzx)=blue, -Im(Tzx)=cyan ("water")
#   BOTTOM : Tzy panel  — Re(Tzy)=red,  -Im(Tzy)=pink
#   y-axis : linear [-0.5, 0.5]   x-axis: log period [1e-4, 1e4] s
#   Legend : drawn at the bottom-left of the last station on each page.

_VTF_REGION  = [1e-4, 1e4, -0.5, 0.5]   # [T_min, T_max, T_min_y, T_max_y]
_VTF_W       = 3.8    # cm – panel width
_VTF_H       = 3.0    # cm – panel height (both Tzx and Tzy)
_VTF_GAP     = 0.5    # cm – gap between the two sub-panels
_VTF_TO_TZY  = _VTF_H + _VTF_GAP        # shift from Tzx top to Tzy top
_VTF_PANEL_DX = _VTF_W + 0.5            # horizontal column pitch
_VTF_PANEL_DY = _VTF_TO_TZY + _VTF_H + 0.7   # full row height

# colours
_VTF_BLUE    = "0/0/255"      # Re(Tzx)
_VTF_CYAN    = "0/255/255"    # -Im(Tzx)  ("water" in original)
_VTF_RED     = "255/0/0"      # Re(Tzy)
_VTF_PINK    = "255/0/255"    # -Im(Tzy)

_VTF_LINE_PEN = "0.5p"
_VTF_SYM_SIZE = "0.15c"
_VTF_ERR_CAP  = "0.15c"


def _vtf_tzx_frame(left: bool) -> list:
    w = "W" if left else "w"
    return [f"{w}sne",
            "xa1f3",
            "ya0.5f0.5g0.5+lTzx"]


def _vtf_tzy_frame(left: bool, bottom: bool) -> list:
    w = "W" if left   else "w"
    s = "S" if bottom else "s"
    return [f"{w}{s}ne",
            "xa1f3+lLog(Period,s)",
            "ya0.5f0.5g0.5+lTzy"]


def _vtf_to_arrays(records: list) -> dict:
    """
    Convert a list of VTFData records for one site into numpy arrays.

    VTFData.Cal/Obs/Err.TZ[0] = Tzx,  TZ[1] = Tzy   (index mapping from
    COMP_INDEX[4]="zx", COMP_INDEX[5]="zy" in femtic_data.py).

    Error scalar = max(re_err, im_err) per component, same convention as
    write_result_vtf() which asserts re_err == im_err but takes the max.

    Returns dict with keys:
        period,
        ReTzxCal, ImTzxCal, ReTzyCal, ImTzyCal,
        ReTzxObs, ImTzxObs, ReTzyObs, ImTzyObs,
        ErrTzx,   ErrTzy
    """
    records = sorted(records, key=lambda d: d.freq)
    period = np.array([1.0 / d.freq for d in records])

    out = {"period": period}
    for label, k in [("Tzx", 0), ("Tzy", 1)]:
        out[f"Re{label}Cal"] = np.array([d.Cal.TZ[k].real for d in records])
        out[f"Im{label}Cal"] = np.array([d.Cal.TZ[k].imag for d in records])
        out[f"Re{label}Obs"] = np.array([d.Obs.TZ[k].real for d in records])
        out[f"Im{label}Obs"] = np.array([d.Obs.TZ[k].imag for d in records])
        # err = max(re_err, im_err) — same convention as write_result_vtf()
        out[f"Err{label}"] = np.array(
            [max(d.Err.TZ[k].real, d.Err.TZ[k].imag) * FACTOR for d in records]
        )
    return out


def _plot_vtf_legend(fig, proj: str) -> None:
    """
    Draw the four-entry colour legend that the shell script adds at the
    bottom-left of the last station on each page.
    """
    legend_region = [0, 40, 0, 2]
    entries = [
        (0,  "Re(Tzx)", _VTF_BLUE),
        (10, "Im(Tzx)", _VTF_CYAN),
        (20, "Re(Tzy)", _VTF_RED),
        (30, "Im(Tzy)", _VTF_PINK),
    ]
    for x, txt, col in entries:
        fig.text(x=x, y=0,
                 text=txt,
                 region=legend_region,
                 projection=proj,
                 justify="BL",
                 font=f"9p,Helvetica,{col}",
                 no_clip=True)


def _plot_vtf_station(fig, arrays: dict, label: str,
                      show_left: bool, show_bottom: bool,
                      draw_legend: bool) -> None:
    """
    Draw Tzx panel (top) and Tzy panel (bottom) for one VTF station.
    Origin is restored to the Tzx-panel top before returning.

    Parameters
    ----------
    fig         : active PyGMT figure
    arrays      : dict returned by _vtf_to_arrays()
    label       : station display label
    show_left   : True for leftmost column (draw W-side annotations)
    show_bottom : True for bottom row (draw S-side tick labels)
    draw_legend : True for the last station on a page
    """
    period   = arrays["period"]
    proj     = f"X{_VTF_W}l/{_VTF_H}"

    # ----------------------------------------------------------- Tzx panel --
    frame_todo = _vtf_tzx_frame(show_left)

    def _draw_component(ydata, col, is_first_on_panel):
        nonlocal frame_todo
        fig.plot(x=period, y=ydata,
                 region=_VTF_REGION, projection=proj,
                 frame=frame_todo if is_first_on_panel else None,
                 pen=f"{_VTF_LINE_PEN},{col}")
        if is_first_on_panel:
            frame_todo = None

    def _draw_obs_and_err(ydata, err, col):
        fig.plot(x=period, y=ydata,
                 region=_VTF_REGION, projection=proj,
                 style=f"c{_VTF_SYM_SIZE}", pen=f"{_VTF_LINE_PEN},{col}",
                 no_clip=True)
        fig.plot(data=pd.DataFrame({"x": period, "y": ydata, "ey": err}),
                 region=_VTF_REGION, projection=proj,
                 style=f"ey{_VTF_ERR_CAP}/0.3p", pen=f"{_VTF_LINE_PEN},{col}",
                 no_clip=True)

    # Re(Tzx) model + obs
    _draw_component(arrays["ReTzxCal"], _VTF_BLUE, is_first_on_panel=True)
    _draw_obs_and_err(arrays["ReTzxObs"], arrays["ErrTzx"], _VTF_BLUE)

    # -Im(Tzx) model + obs  (sign flip, same as shell: `awk { print $1, -$3 }`)
    _draw_component(-arrays["ImTzxCal"], _VTF_CYAN, is_first_on_panel=False)
    _draw_obs_and_err(-arrays["ImTzxObs"], arrays["ErrTzx"], _VTF_CYAN)

    if frame_todo is not None:   # nothing was plotted
        fig.basemap(region=_VTF_REGION, projection=proj, frame=frame_todo)

    # site label (top-left inside Tzx panel)
    fig.text(x=period.min(), y=_VTF_REGION[3] * 0.85,
             text=label,
             region=_VTF_REGION, projection=proj,
             justify="TL", font="8p,Helvetica,black",
             no_clip=True, offset="0.08c/-0.05c")

    # ----------------------------------------------------------- Tzy panel --
    fig.shift_origin(yshift=f"-{_VTF_TO_TZY}c")
    frame_todo = _vtf_tzy_frame(show_left, show_bottom)

    _draw_component(arrays["ReTzyCal"], _VTF_RED, is_first_on_panel=True)
    _draw_obs_and_err(arrays["ReTzyObs"], arrays["ErrTzy"], _VTF_RED)

    _draw_component(-arrays["ImTzyCal"], _VTF_PINK, is_first_on_panel=False)
    _draw_obs_and_err(-arrays["ImTzyObs"], arrays["ErrTzy"], _VTF_PINK)

    if frame_todo is not None:
        fig.basemap(region=_VTF_REGION, projection=proj, frame=frame_todo)

    # ---------------------------------------------------------- legend (last station on page) --
    if draw_legend:
        legend_proj = f"X{_VTF_W * 4}l/{_VTF_H}"   # wide enough for 4 entries
        fig.shift_origin(yshift=f"-{_VTF_H + 0.3}c")
        _plot_vtf_legend(fig, legend_proj)
        fig.shift_origin(yshift=f"{_VTF_H + 0.3}c")

    # restore origin to Tzx-panel top
    fig.shift_origin(yshift=f"{_VTF_TO_TZY}c")


def plot_vtf_sounding_curves(
    sites_file: "Optional[str]" = None,
    out_prefix: str = "vtf_all_curv",
    per_page: int = 12,
    n_cols: int = 6,
) -> None:
    """
    Plot VTF (tipper) sounding curves for all VTF stations.

    Data are read directly from vtf_data_list populated by read_result().
    No intermediate text files are touched.

    Parameters
    ----------
    sites_file : optional two-column whitespace-separated file
                 (site_name_or_id  display_label) controlling plot order.
                 If None, all VTF sites are plotted in data order.
    out_prefix : prefix for output PDF filenames
    per_page   : stations per output PDF
    n_cols     : panel columns per page
    """
    # ----------------------------------------------------------------
    # 1. Build per-site arrays from in-memory vtf_data_list
    # ----------------------------------------------------------------
    vtf_by_site: Dict[int, list] = {}
    for sid, d in vtf_data_list:
        vtf_by_site.setdefault(sid, []).append(d)

    if not vtf_by_site:
        print("plot_vtf_sounding_curves: no VTF data to plot.", file=sys.stderr)
        return

    arrays_by_id: Dict[int, dict] = {
        sid: _vtf_to_arrays(recs) for sid, recs in vtf_by_site.items()
    }

    # ----------------------------------------------------------------
    # 2. Determine plotting order and labels
    # ----------------------------------------------------------------
    if sites_file:
        raw_pairs = _load_sites_file(sites_file)
        name_to_id = {v: k for k, v in site_id_to_name.items()}
        site_pairs = []
        for name, label in raw_pairs:
            try:
                sid = int(name)
            except ValueError:
                sid = name_to_id.get(name)
            if sid is None or sid not in arrays_by_id:
                print(f"  WARNING: VTF site '{name}' not found, skipping.",
                      file=sys.stderr)
                continue
            site_pairs.append((sid, label))
    else:
        seen = {}
        for sid, _ in vtf_data_list:
            if sid not in seen:
                seen[sid] = convert_site_id_to_name(sid)
        site_pairs = [(sid, lbl) for sid, lbl in seen.items()
                      if sid in arrays_by_id]

    n_stations = len(site_pairs)
    if n_stations == 0:
        print("plot_vtf_sounding_curves: no plottable VTF stations.", file=sys.stderr)
        return

    n_rows = math.ceil(per_page / n_cols)

    # ----------------------------------------------------------------
    # 3. Plot
    # ----------------------------------------------------------------
    pygmt.config(FONT_TITLE="14p", FONT_LABEL="12p", MAP_TICK_LENGTH="0.1c")

    fig          = None
    current_page = -1

    for j, (site_id, label) in enumerate(site_pairs):
        page_idx    = j // per_page
        idx_on_page = j % per_page
        col_idx     = idx_on_page % n_cols
        row_idx     = idx_on_page // n_cols

        # Is this the last station on the current page?
        next_page = (j + 1) // per_page
        is_last_on_page = (next_page != page_idx) or (j == n_stations - 1)

        site_name = convert_site_id_to_name(site_id)
        print(f"  Plotting VTF station {j+1}/{n_stations}: "
              f"id={site_id} name={site_name!r} label={label!r} "
              f"page={page_idx+1} row={row_idx} col={col_idx}")

        # ---- new page ----
        if page_idx != current_page:
            if fig is not None:
                out_path = f"{out_prefix}_{current_page + 1}.pdf"
                fig.savefig(out_path, dpi=300)
                print(f"  Saved {out_path}")
            fig = pygmt.Figure()
            current_page = page_idx
            fig.shift_origin(xshift="3c",
                             yshift=f"{3.0 + n_rows * _VTF_PANEL_DY}c")

        # ---- shift to this panel's top-left corner ----
        if idx_on_page == 0:
            pass
        elif col_idx == 0:
            fig.shift_origin(
                xshift=f"-{(n_cols - 1) * _VTF_PANEL_DX}c",
                yshift=f"-{_VTF_PANEL_DY}c",
            )
        else:
            fig.shift_origin(xshift=f"{_VTF_PANEL_DX}c")

        arrays = arrays_by_id[site_id]
        _plot_vtf_station(fig, arrays, label,
                          show_left=(col_idx == 0),
                          show_bottom=(row_idx == n_rows - 1),
                          draw_legend=is_last_on_page)

    # ---- save last page ----
    if fig is not None:
        out_path = f"{out_prefix}_{current_page + 1}.pdf"
        fig.savefig(out_path, dpi=300)
        print(f"  Saved {out_path}")

    print("VTF plotting done.")

# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    global read_true_error_file, true_error_file_name
    global output_csv, impedance_converted_to_app_res

    if len(sys.argv) < 3:
        print("You must specify iteration number and process number !!", file=sys.stderr)
        sys.exit(1)

    iteration_number = int(sys.argv[1])
    num_pe           = int(sys.argv[2])

    read_result(iteration_number, num_pe)

    # MT plot options
    do_plot      = False
    plot_sites   = None
    plot_out     = "imp_all_curv"
    plot_perpage = 12
    plot_cols    = 6
    # VTF plot options
    do_plot_vtf      = False
    plot_vtf_sites   = None
    plot_vtf_out     = "vtf_all_curv"
    plot_vtf_perpage = 12
    plot_vtf_cols    = 6

    i = 3
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "-name":
            read_relation_site_id_to_name(sys.argv[i + 1])
            i += 2
        elif arg == "-err":
            true_error_file_name = sys.argv[i + 1]
            read_true_error_file = True
            i += 2
        elif arg == "-csv":
            output_csv = True
            i += 1
        elif arg == "-undist":
            read_control_data("control.dat")
            read_distortion_matrix(iteration_number)
            i += 1
        elif arg == "-appphs":
            print("Impedance tensors are converted to apparent resistivity and phase.")
            impedance_converted_to_app_res = True
            i += 1
        elif arg == "-plot":
            do_plot = True
            i += 1
        elif arg == "-sites":
            plot_sites = sys.argv[i + 1]
            i += 2
        elif arg == "-out":
            plot_out = sys.argv[i + 1]
            i += 2
        elif arg == "-perpage":
            plot_perpage = int(sys.argv[i + 1])
            i += 2
        elif arg == "-cols":
            plot_cols = int(sys.argv[i + 1])
            i += 2
        elif arg == "-plotvtf":
            do_plot_vtf = True
            i += 1
        elif arg == "-sitesvtf":
            plot_vtf_sites = sys.argv[i + 1]
            i += 2
        elif arg == "-outvtf":
            plot_vtf_out = sys.argv[i + 1]
            i += 2
        elif arg == "-perpageVTF":
            plot_vtf_perpage = int(sys.argv[i + 1])
            i += 2
        elif arg == "-colsVTF":
            plot_vtf_cols = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1

    calc_true_rms(read_true_error_file, true_error_file_name)
    write_result()

    if do_plot:
        if not _PLOT_AVAILABLE:
            print("ERROR: -plot requires numpy, pandas and pygmt. "
                  "Install them and retry.", file=sys.stderr)
            sys.exit(1)
        plot_mt_sounding_curves(
            sites_file=plot_sites,
            out_prefix=plot_out,
            per_page=plot_perpage,
            n_cols=plot_cols,
        )

    if do_plot_vtf:
        if not _PLOT_AVAILABLE:
            print("ERROR: -plotvtf requires numpy, pandas and pygmt. "
                  "Install them and retry.", file=sys.stderr)
            sys.exit(1)
        plot_vtf_sounding_curves(
            sites_file=plot_vtf_sites,
            out_prefix=plot_vtf_out,
            per_page=plot_vtf_perpage,
            n_cols=plot_vtf_cols,
        )


if __name__ == "__main__":
    main()
