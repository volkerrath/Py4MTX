#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_proc.py
=================
Magnetotelluric (MT) data processing and I/O utilities.

This module provides a light-weight, self-contained interface to read and
write standard MT transfer functions in EDI format:

- Impedance tensor Z (2×2 complex)
- Tipper T (1×2 complex), if present
- Phase tensor Φ (2×2 real), computed from Z

Two EDI "flavours" are supported:

1) Phoenix / SPECTRA-based EDIs
-------------------------------
Phoenix EDIs may contain ``>SPECTRA`` blocks with a 7×7 real-valued matrix
encoding auto- and cross-spectra plus metadata, for example::

    >SPECTRA  FREQ=1.040E+04 ROTSPEC=0 BW=2.6000E+03 AVGT=5.1245E+05 // 49

If any ``>SPECTRA`` blocks are present and ``prefer_spectra=True`` (default),
the module will:

- parse the SPECTRA blocks (:func:`parse_spectra_blocks`)
- reconstruct the complex Hermitian spectra matrix (:func:`reconstruct_S_phoenix`)
- recover Z and T from S (:func:`ZT_from_S`)

No Z/T error estimates are derived for SPECTRA EDIs in this implementation.

2) Classical table-based EDIs
-----------------------------
Classical EDIs provide tabulated values using blocks such as::

    >FREQ
    >ZXXR  >ZXXI  >ZXYR  >ZXYI  >ZYXR  >ZYXI  >ZYYR  >ZYYI
    >ZXX.VAR  >ZXY.VAR  >ZYX.VAR  >ZYY.VAR
    >TXR.EXP  >TXI.EXP  >TYR.EXP  >TYI.EXP
    >TXVAR.EXP  >TYVAR.EXP
    >ZROT

For table-style EDIs, the module:

- parses component blocks (:func:`_extract_block_values`)
- assembles Z and T (:func:`_build_impedance`, :func:`_build_tipper`)
- reads optional Z rotation angles (``>ZROT``) and stores them in ``edi["rot"]``

High-level entry points
-----------------------

- :func:`load_edi`:
  Read an EDI file (Phoenix or classical) and return an in-memory dictionary.
- :func:`compute_pt`:
  Compute phase tensor and (optionally) propagate impedance errors via
  Monte-Carlo simulation.
- :func:`save_edi`:
  Write an EDI dictionary back to a classical table-style EDI file.

Dictionary layout
-----------------

A typical dictionary returned by :func:`load_edi` looks like::

    edi = {
        "freq": (n,),                      # Hz
        "Z": (n, 2, 2) complex,            # impedance
        "T": (n, 1, 2) complex or None,    # tipper
        "Z_err": (n, 2, 2) float or None,  # variance or std (see "err_kind")
        "T_err": (n, 1, 2) float or None,
        "P": (n, 2, 2) float or None,      # phase tensor (filled by compute_pt)
        "P_err": (n, 2, 2) float or None,
        "rot": (n,) float or None,         # degrees; from ROTSPEC or ZROT
        "err_kind": "var" or "std",
        "source_kind": "spectra" or "tables",
        "station": str or None,
        "lat_deg": float or None,
        "lon_deg": float or None,
        "elev_m": float or None,
        # convenience aliases used in some legacy code:
        "lat": float or None,
        "lon": float or None,
        "elev": float or None,
    }

Notes
-----

- Frequencies are returned in **ascending** order by default (``freq_order="inc"``).
  Pass ``freq_order="dec"`` for descending order or ``freq_order="keep"`` to
  preserve the original EDI file order.
- For table-style EDIs, variance blocks such as ``ZXX.VAR`` and ``TXVAR.EXP``
  are read. By default these are interpreted as **variances** (``err_kind="var"``).

Author: Volker Rath (DIAS)
Created with the help of ChatGPT (GPT-5 Thinking) on 2026-02-13 (UTC)
"""

from __future__ import annotations

import re
import sys
import os
import json
import inspect

from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Mapping

import numpy as np
import pandas as pd
from scipy.interpolate import make_smoothing_spline

_MU0: float = 4.0 * np.pi * 1.0e-7

# ---------------------------------------------------------------------------
# Basic text helpers
# ---------------------------------------------------------------------------

def get_data_list(dirname=None, ext= '.edi', sort=False, fullpath=True):



    """
    List files with given extension in a directory.

    Parameters
    ----------
    datarname : str
        Directory containing data files.
    ext : str

    sort : bool
        If True, return files in sorted order.
    fullpath : bool
        If True, return full paths; otherwise return file names only.

    Returns
    -------
    data_files : list[str]
        List of data file paths or names.

    Notes
    -----
    Hidden files (starting with '.') are ignored. Raises SystemExit if no data files
    are found.
    """
    data_files = []
    files = os.listdir(dirname)
    for entry in files:
        # print(entry)
        if entry.endswith(ext) and not entry.startswith('.'):
            if fullpath:
                data_files.append(dirname+entry)
            else:
                data_files.append(entry)

    ns = np.size(data_files)
    if ns ==0:
        sys.exit('No data files ('+ext +') found in '+dirname+'! Exit.')

    if sort:
        data_files = sorted(data_files)

    return data_files

# Alias
get_edi_list = get_data_list

def read_edi_text(path: str | Path, encoding: str = "latin-1") -> str:
    """Read an EDI file as raw text.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the EDI file.
    encoding : str, optional
        Text encoding used to decode the file. Phoenix/Metronix outputs are
        commonly encoded in ISO-8859-1 (``"latin-1"``). Default is
        ``"latin-1"``.

    Returns
    -------
    str
        Complete file content as a single string.
    """
    p = Path(path)
    return p.read_text(encoding=encoding, errors="ignore")


def _has_spectra(text: str) -> bool:
    """Return True if Phoenix ``>SPECTRA`` blocks occur in the file text."""
    return ">SPECTRA" in text.upper()


def _split_lines(text: str) -> List[str]:
    """Split text into lines (no further processing)."""
    return text.splitlines()


def _extract_header_lines(lines: List[str]) -> List[str]:
    """Extract header lines up to the first data or spectra block.

    Everything from the start of the file up to the first line starting with
    ``">FREQ"``, ``">ZXXR"`` or ``">SPECTRA"`` is considered header.
    """
    header: List[str] = []
    for ln in lines:
        u = ln.strip().upper()
        if u.startswith(">FREQ") or u.startswith(">ZXXR") or u.startswith(">SPECTRA"):
            break
        header.append(ln)
    return header


def _parse_simple_meta(header_lines: List[str]) -> Dict[str, Any]:
    """Parse basic metadata fields from header lines.

    The parser is intentionally conservative and only tries to guess a few
    common keys found in many EDIs.

    Parameters
    ----------
    header_lines : list of str
        Header lines from an EDI file.

    Returns
    -------
    dict
        Contains (possibly None) entries for:

        - ``station``: station / site name
        - ``lat_deg``: latitude in decimal degrees
        - ``lon_deg``: longitude in decimal degrees
        - ``elev_m``: elevation in meters
    """
    station = None
    lat_deg = None
    lon_deg = None
    elev_m = None

    for ln in header_lines:
        u = ln.upper()

        if "DATAID" in u and station is None:
            try:
                rhs = ln.split("=", 1)[1].strip()
                if rhs.startswith('"') and '"' in rhs[1:]:
                    station = rhs.split('"', 2)[1]
                else:
                    station = rhs.split()[0]
            except Exception:
                pass

        if "LAT" in u and "REFLAT" not in u and lat_deg is None:
            try:
                rhs = ln.split("=", 1)[1].strip().strip('"')
                if ":" in rhs:
                    deg, minute, sec = (float(p) for p in rhs.split(":")[:3])
                    if deg >= 0.:
                        lat_deg = deg + minute / 60.0 + sec / 3600.0
                    else:
                        lat_deg = deg - minute / 60.0 - sec / 3600.0
                else:
                    lat_deg = float(rhs.split()[0])
            except Exception:
                pass

        if ("LON" in u or "LONG" in u) and "REFLON" not in u and lon_deg is None:
            try:
                rhs = ln.split("=", 1)[1].strip().strip('"')
                if ":" in rhs:
                    #print(rhs)
                    deg, minute, sec = (float(p) for p in rhs.split(":")[:3])
                    if deg >= 0.:
                        lon_deg = deg + minute / 60.0 + sec / 3600.0
                    else:
                        lon_deg = deg - minute / 60.0 - sec / 3600.0
                    #print(lon_deg)
                else:
                    lon_deg = float(rhs.split()[0])
            except Exception:
                pass

        if "ELEV" in u and "REFELEV" not in u and elev_m is None:
            try:
                rhs = ln.split("=", 1)[1].strip().strip('"')
                elev_m = float(rhs.split()[0])
            except Exception:
                pass

    return {
        "station": station,
        "lat_deg": lat_deg,
        "lon_deg": lon_deg,
        "elev_m": elev_m,
    }


def _extract_block_values(lines: List[str], keyword: str) -> Optional[np.ndarray]:
    """Extract numeric values from a classical EDI block.

    Parameters
    ----------
    lines : list of str
        EDI file lines.
    keyword : str
        Block keyword without leading ``">"`` (e.g. ``"FREQ"``, ``"ZXXR"``,
        ``"ZXX.VAR"``, ``"TXR.EXP"``). Matching is case-insensitive.

    Returns
    -------
    numpy.ndarray or None
        1D numeric array if the block exists, otherwise ``None``.

    Notes
    -----
    - The scan starts at the block header line and continues until the next
      line beginning with ``">"`` (or EOF).
    - Numbers in Fortran ``D`` exponent format are supported.
    - Comments starting with ``"//"`` are ignored.
    """
    up_kw = ">" + keyword.upper()
    n = len(lines)
    start_idx = None

    for i, ln in enumerate(lines):
        if ln.strip().upper().startswith(up_kw):
            start_idx = i + 1
            break

    if start_idx is None:
        return None

    vals: List[float] = []
    idx = start_idx
    num_pattern = r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[EeDd][+\-]?\d+)?"

    while idx < n:
        ln = lines[idx].strip()
        if ln.startswith(">"):
            break
        if ln.startswith("//") or ln == "":
            idx += 1
            continue
        for s in re.findall(num_pattern, ln):
            try:
                vals.append(float(s.replace("D", "E")))
            except ValueError:
                pass
        idx += 1

    if not vals:
        return None
    return np.asarray(vals, dtype=float)


# ---------------------------------------------------------------------------
# Classical table-based impedance, tipper, rotation
# ---------------------------------------------------------------------------


def _build_impedance(
    freq: np.ndarray,
    z_real_blocks: Dict[str, np.ndarray],
    z_imag_blocks: Dict[str, np.ndarray],
    z_var_blocks: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Assemble the complex impedance tensor Z and (optional) variance array."""
    n = freq.size
    Z = np.zeros((n, 2, 2), dtype=np.complex128)
    Z_var = np.full((n, 2, 2), np.nan, dtype=float)

    comp_map = {"ZXX": (0, 0), "ZXY": (0, 1), "ZYX": (1, 0), "ZYY": (1, 1)}

    for base, (i, j) in comp_map.items():
        r = z_real_blocks.get(base + "R")
        im = z_imag_blocks.get(base + "I")
        if r is None or im is None:
            continue
        if r.size < n or im.size < n:
            raise ValueError(f"Block {base}R/I has fewer than {n} entries.")
        Z[:, i, j] = r[:n] + 1j * im[:n]

        var_arr = z_var_blocks.get(base + ".VAR")
        if var_arr is not None:
            if var_arr.size < n:
                raise ValueError(f"Block {base}.VAR has fewer than {n} entries.")
            Z_var[:, i, j] = var_arr[:n]

    if np.all(np.isnan(Z_var)):
        return Z, None
    return Z, Z_var


def _build_tipper(
    freq: np.ndarray,
    t_blocks_re: Dict[str, np.ndarray],
    t_blocks_im: Dict[str, np.ndarray],
    t_var_blocks: Dict[str, np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Assemble the complex tipper T and (optional) variance array."""
    n = freq.size

    txr = t_blocks_re.get("TXR.EXP")
    txi = t_blocks_im.get("TXI.EXP")
    tyr = t_blocks_re.get("TYR.EXP")
    tyi = t_blocks_im.get("TYI.EXP")

    if txr is None or txi is None or tyr is None or tyi is None:
        return None, None

    for arr_name, arr in [
        ("TXR.EXP", txr),
        ("TXI.EXP", txi),
        ("TYR.EXP", tyr),
        ("TYI.EXP", tyi),
    ]:
        if arr.size < n:
            raise ValueError(f"Block {arr_name} has fewer than {n} entries.")

    Tx = txr[:n] + 1j * txi[:n]
    Ty = tyr[:n] + 1j * tyi[:n]

    T = np.zeros((n, 1, 2), dtype=np.complex128)
    T[:, 0, 0] = Tx
    T[:, 0, 1] = Ty

    txvar = t_var_blocks.get("TXVAR.EXP")
    tyvar = t_var_blocks.get("TYVAR.EXP")
    if txvar is None or tyvar is None:
        return T, None

    if txvar.size < n or tyvar.size < n:
        raise ValueError("Tipper variance blocks have insufficient entries.")

    T_var = np.full((n, 1, 2), np.nan, dtype=float)
    T_var[:, 0, 0] = txvar[:n]
    T_var[:, 0, 1] = tyvar[:n]

    return T, T_var


# ---------------------------------------------------------------------------
# Phoenix SPECTRA handling
# ---------------------------------------------------------------------------


def parse_spectra_blocks(edi_text: str) -> List[Tuple[float, float, float, np.ndarray]]:
    """Parse Phoenix ``>SPECTRA`` blocks.

    Returns a list of tuples ``(freq_hz, avgt_s, rot_deg, mat7)`` where ``mat7``
    is the 7×7 Phoenix real-valued encoding.
    """
    lines = edi_text.splitlines()
    n_lines = len(lines)
    i = 0
    num_pattern = r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[EeDd][+\-]?\d+)?"

    blocks: List[Tuple[float, float, float, np.ndarray]] = []

    while i < n_lines:
        line = lines[i]
        if not line.lstrip().upper().startswith(">SPECTRA"):
            i += 1
            continue

        block_lines = [line]
        i += 1
        while i < n_lines and not lines[i].lstrip().startswith(">"):
            block_lines.append(lines[i])
            i += 1

        block_text = "\n".join(block_lines) + "\n"

        fm = re.search(
            r"FREQ\s*=?\s*([0-9.]+[ED][+\-]?\d+|[0-9.]+)",
            block_text,
            flags=re.IGNORECASE,
        )
        if not fm:
            continue
        freq_hz = float(fm.group(1).replace("D", "E"))

        am = re.search(
            r"AVGT\s*=?\s*([0-9.]+[ED][+\-]?\d+|[0-9.]+)",
            block_text,
            flags=re.IGNORECASE,
        )
        avgt_s = float(am.group(1).replace("D", "E")) if am else float("nan")

        rm = re.search(
            r"ROTSPEC\s*=?\s*([0-9.\-+]+[ED][+\-]?\d+|[0-9.\-+]+)",
            block_text,
            flags=re.IGNORECASE,
        )
        rot_deg = float(rm.group(1).replace("D", "E")) if rm else float("nan")

        num_strings: List[str] = []
        for ln in block_lines:
            if not ln.strip():
                continue
            if ln.lstrip().startswith(">"):
                continue
            num_strings.extend(re.findall(num_pattern, ln))

        if len(num_strings) < 49:
            continue

        vals = [float(s.replace("D", "E")) for s in num_strings[:49]]
        mat7 = np.array(vals, dtype=float).reshape(7, 7)
        blocks.append((freq_hz, avgt_s, rot_deg, mat7))

    return blocks


def reconstruct_S_phoenix(mat7: np.ndarray) -> np.ndarray:
    """Reconstruct complex 7×7 spectra matrix from Phoenix real-valued encoding."""
    S = np.zeros((7, 7), dtype=np.complex128)
    for i in range(7):
        S[i, i] = mat7[i, i]
        for j in range(i + 1, 7):
            re_ij = mat7[j, i]
            im_ij = mat7[i, j]
            S[i, j] = re_ij + 1j * im_ij
            S[j, i] = re_ij - 1j * im_ij
    return S


def ZT_from_S(S: np.ndarray, ref: str = "RH") -> Tuple[np.ndarray, np.ndarray]:
    """Recover impedance Z and tipper T from a Phoenix spectra matrix S."""
    h1, h2 = 0, 1
    hz = 2
    ex, ey = 3, 4

    SHH = np.array([[S[h1, h1], S[h1, h2]], [S[h2, h1], S[h2, h2]]], dtype=np.complex128)
    SEH = np.array([[S[ex, h1], S[ex, h2]], [S[ey, h1], S[ey, h2]]], dtype=np.complex128)
    SBH = np.array([[S[hz, h1], S[hz, h2]]], dtype=np.complex128)

    try:
        SHH_inv = np.linalg.inv(SHH)
    except np.linalg.LinAlgError:
        SHH_inv = np.linalg.pinv(SHH)

    Z = SEH @ SHH_inv
    T = SBH @ SHH_inv

    # Phoenix scaling (µV/m per nT -> V/m per T)
    Z /= 1.0e3

    if ref.upper() == "LH":
        Z = -Z
        T = -T

    return Z, T


def _load_from_spectra(text: str, *, ref: str = "RH") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build ``freq``, ``Z``, ``T`` and ``rot`` arrays from Phoenix SPECTRA blocks."""
    blocks = parse_spectra_blocks(text)
    if not blocks:
        raise RuntimeError("No usable >SPECTRA blocks found in EDI text.")

    n = len(blocks)
    freq = np.empty(n, dtype=float)
    Z = np.empty((n, 2, 2), dtype=np.complex128)
    T = np.empty((n, 1, 2), dtype=np.complex128)
    rot = np.empty(n, dtype=float)

    for k, (f, _avgt, r, mat7) in enumerate(blocks):
        freq[k] = f
        rot[k] = r
        S = reconstruct_S_phoenix(mat7)
        Zk, Tk = ZT_from_S(S, ref=ref)
        Z[k] = Zk
        T[k] = Tk

    return freq, Z, T, rot


def load_edi(
    path: str | Path,
    *,
    prefer_spectra: bool = True,
    ref: str = "RH",
    err_kind: str = "var",
    drop_invalid_periods: bool = True,
    invalid_sentinel: float = 1.0e30,
    freq_order: str = "inc",
) -> Dict[str, Any]:
    """Load an EDI file into a dictionary.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the EDI file.
    prefer_spectra : bool, optional
        If ``True`` (default) and Phoenix ``>SPECTRA`` blocks are present,
        reconstruct Z/T from the spectra.  Set to ``False`` to force
        classical table parsing.
    ref : str, optional
        Reference channel used when recovering Z/T from Phoenix spectra.
        Default is ``"RH"``.
    err_kind : {"var", "std"}, optional
        Interpretation of variance blocks — ``"var"`` (default) keeps them as
        variances; ``"std"`` converts to standard deviations.
    drop_invalid_periods : bool, optional
        If ``True`` (default), rows whose Z or T values exceed
        ``invalid_sentinel`` or are non-finite are removed.
    invalid_sentinel : float, optional
        Absolute threshold above which a value is considered invalid.
        Default is ``1e30``.
    freq_order : {"inc", "dec", "keep"}, optional
        Controls the order of frequencies in the returned dictionary:

        - ``"inc"``  — ascending frequency (default, equivalent to the
          previous hard-wired behaviour).
        - ``"dec"``  — descending frequency (i.e. ascending period).
        - ``"keep"`` — preserve the order as found in the EDI file.

    Returns
    -------
    dict
        Site dictionary; see the module docstring for the full layout.
    """
    if err_kind not in {"var", "std"}:
        raise ValueError(f"Unknown err_kind {err_kind!r}; expected 'var' or 'std'.")
    freq_order = freq_order.lower()
    if freq_order not in {"inc", "dec", "keep"}:
        raise ValueError(f"Unknown freq_order {freq_order!r}; expected 'inc', 'dec', or 'keep'.")

    text = read_edi_text(path)
    lines = _split_lines(text)
    header_lines = _extract_header_lines(lines)
    meta = _parse_simple_meta(header_lines)

    use_spectra = _has_spectra(text) and prefer_spectra
    rot: Optional[np.ndarray] = None

    if use_spectra:
        freq, Z, T, rot = _load_from_spectra(text, ref=ref)
        Z_var = None
        T_var = None
        source_kind = "spectra"
    else:
        freq_vals = _extract_block_values(lines, "FREQ")
        if freq_vals is None:
            raise RuntimeError("Could not find >FREQ block in EDI file.")
        freq = freq_vals.copy()

        z_real_blocks: Dict[str, np.ndarray] = {}
        z_imag_blocks: Dict[str, np.ndarray] = {}
        z_var_blocks: Dict[str, np.ndarray] = {}

        for key in ["ZXXR", "ZXYR", "ZYXR", "ZYYR"]:
            arr = _extract_block_values(lines, key)
            if arr is not None:
                z_real_blocks[key] = arr
        for key in ["ZXXI", "ZXYI", "ZYXI", "ZYYI"]:
            arr = _extract_block_values(lines, key)
            if arr is not None:
                z_imag_blocks[key] = arr
        for key in ["ZXX.VAR", "ZXY.VAR", "ZYX.VAR", "ZYY.VAR"]:
            arr = _extract_block_values(lines, key)
            if arr is not None:
                z_var_blocks[key] = arr

        Z, Z_var = _build_impedance(freq, z_real_blocks, z_imag_blocks, z_var_blocks)

        t_blocks_re: Dict[str, np.ndarray] = {}
        t_blocks_im: Dict[str, np.ndarray] = {}
        t_var_blocks: Dict[str, np.ndarray] = {}

        for key in ["TXR.EXP", "TYR.EXP"]:
            arr = _extract_block_values(lines, key)
            if arr is not None:
                t_blocks_re[key] = arr
        for key in ["TXI.EXP", "TYI.EXP"]:
            arr = _extract_block_values(lines, key)
            if arr is not None:
                t_blocks_im[key] = arr
        for key in ["TXVAR.EXP", "TYVAR.EXP"]:
            arr = _extract_block_values(lines, key)
            if arr is not None:
                t_var_blocks[key] = arr

        T, T_var = _build_tipper(freq, t_blocks_re, t_blocks_im, t_var_blocks)

        zrot_vals = _extract_block_values(lines, "ZROT")
        if zrot_vals is not None:
            if zrot_vals.size < freq.size:
                raise ValueError("ZROT block has fewer entries than frequencies.")
            rot = zrot_vals[: freq.size].copy()

        source_kind = "tables"

    # Sort / reorder frequencies according to freq_order
    if freq_order == "inc":
        order = np.argsort(freq)
    elif freq_order == "dec":
        order = np.argsort(freq)[::-1]
    else:  # "keep" — preserve EDI file order
        order = np.arange(freq.size, dtype=int)
    freq = freq[order]
    Z = Z[order]
    if T is not None:
        T = T[order]
    if Z_var is not None:
        Z_var = Z_var[order]
    if T_var is not None:
        T_var = T_var[order]
    if rot is not None:
        rot = np.asarray(rot, dtype=float)[order]

    # Drop invalid rows (if requested)
    mask_valid = np.ones(freq.shape, dtype=bool)
    if drop_invalid_periods:

        def _collapse_any(bad: np.ndarray) -> np.ndarray:
            if bad.ndim == 1:
                return bad
            axes = tuple(range(1, bad.ndim))
            return np.any(bad, axis=axes)

        bad_Z = ~np.isfinite(Z.real) | ~np.isfinite(Z.imag)
        bad_Z |= (np.abs(Z.real) > invalid_sentinel) | (np.abs(Z.imag) > invalid_sentinel)
        mask_valid &= ~_collapse_any(bad_Z)


        if T is not None:
            bad_T = ~np.isfinite(T.real) | ~np.isfinite(T.imag)
            bad_T |= (np.abs(T.real) > invalid_sentinel) | (np.abs(T.imag) > invalid_sentinel)
            t_valid = ~_collapse_any(bad_T)
            if np.any(t_valid):
                # Only constrain by tipper if it contains at least some valid rows.
                mask_valid &= t_valid
            else:
                # Common case: EDI contains no tipper, but an EMPTY sentinel was parsed into T.
                # Treat tipper as absent so we do not drop otherwise valid impedance periods.
                T = None
                T_var = None

        if Z_var is not None:
            bad_Zvar = ~np.isfinite(Z_var) | (np.abs(Z_var) > invalid_sentinel)
            mask_valid &= ~_collapse_any(bad_Zvar)

        if T_var is not None:
            bad_Tvar = ~np.isfinite(T_var) | (np.abs(T_var) > invalid_sentinel)
            mask_valid &= ~_collapse_any(bad_Tvar)

    freq = freq[mask_valid]
    Z = Z[mask_valid]
    if Z_var is not None:
        Z_var = Z_var[mask_valid]
    if T is not None:
        T = T[mask_valid]
    if T_var is not None:
        T_var = T_var[mask_valid]
    if rot is not None:
        rot = rot[mask_valid]

    # Map var->err according to err_kind
    Z_err = None if Z_var is None else (Z_var if err_kind == "var" else np.sqrt(Z_var))
    T_err = None if T_var is None else (T_var if err_kind == "var" else np.sqrt(T_var))

    edi: Dict[str, Any] = {
        "freq": freq,
        "Z": Z,
        "T": T,
        "Z_err": Z_err,
        "T_err": T_err,
        "P": None,
        "P_err": None,
        "rot": rot,
        "err_kind": err_kind,
        "freq_order": freq_order,
        "header_raw": header_lines,
        "source_kind": source_kind,
        # metadata
        "station": meta.get("station"),
        "lat_deg": meta.get("lat_deg"),
        "lon_deg": meta.get("lon_deg"),
        "elev_m": meta.get("elev_m"),
        # convenience aliases
        "lat": meta.get("lat_deg"),
        "lon": meta.get("lon_deg"),
        "elev": meta.get("elev_m"),
    }

    # Keep full meta dict too (for forward compatibility)
    edi.update(meta)

    return edi


def estimate_errors(edi_dict: Dict[str, Any], method: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate *new* error levels from the spread of a resampled dataset.

    This is a pragmatic helper to replace error estimates by comparing an
    interpolated representation against the original data. It is intended for
    exploratory work and should be used with caution.
    """
    edi_new = dict(edi_dict)  # shallow copy

    if "freq" not in edi_new:
        raise KeyError("edi_dict must contain 'freq'.")

    old_logf = np.log10(np.asarray(edi_new["freq"], dtype=float).ravel())
    new_logf = np.asarray(method.get("newfreqs"), dtype=float).ravel()
    if new_logf.size == 0:
        raise ValueError("method['newfreqs'] must be a non-empty array.")
    nf = new_logf.size

    edi_new["freq"] = np.power(10.0, new_logf)

    def _interp_complex(arr: np.ndarray) -> np.ndarray:
        out = np.zeros((nf,) + arr.shape[1:], dtype=arr.dtype)
        it = np.ndindex(arr.shape[1:])
        for idx in it:
            y = arr[(slice(None),) + idx]
            if np.iscomplexobj(y):
                sr = make_spline(old_logf, y.real, lam=None)
                si = make_spline(old_logf, y.imag, lam=None)
                out[(slice(None),) + idx] = sr(new_logf) + 1j * si(new_logf)
            else:
                s = make_spline(old_logf, y, lam=None)
                out[(slice(None),) + idx] = s(new_logf)
        return out

    # Estimate errors as std of interpolation mismatch (very rough)
    if edi_new.get("Z") is not None:
        Z_old = np.asarray(edi_new["Z"])
        Z_new = _interp_complex(Z_old)
        edi_new["Z"] = Z_new
        edi_new["Z_err"] = np.std(Z_new, axis=0, ddof=1)

    if edi_new.get("T") is not None:
        T_old = np.asarray(edi_new["T"])
        T_new = _interp_complex(T_old)
        edi_new["T"] = T_new
        edi_new["T_err"] = np.std(T_new, axis=0, ddof=1)

    if edi_new.get("P") is not None:
        P_old = np.asarray(edi_new["P"])
        P_new = _interp_complex(P_old.astype(float)).astype(float)
        edi_new["P"] = P_new
        edi_new["P_err"] = np.std(P_new, axis=0, ddof=1)

    # mark errors as std
    edi_new["err_kind"] = "std"
    return edi_new


def interpolate_data(edi_dict: Dict[str, Any], method: Dict[str, Any]) -> Dict[str, Any]:
    """Interpolate MT transfer functions to a new frequency grid."""
    edi_new = dict(edi_dict)

    old_logf = np.log10(np.asarray(edi_new["freq"], dtype=float).ravel())
    new_logf = np.asarray(method.get("newfreqs"), dtype=float).ravel()
    if new_logf.size == 0:
        raise ValueError("method['newfreqs'] must be a non-empty array.")
    nf = new_logf.size

    edi_new["freq"] = np.power(10.0, new_logf)

    def _interp_any(arr: np.ndarray) -> np.ndarray:
        out = np.zeros((nf,) + arr.shape[1:], dtype=arr.dtype)
        it = np.ndindex(arr.shape[1:])
        for idx in it:
            y = arr[(slice(None),) + idx]
            if np.iscomplexobj(y):
                sr = make_spline(old_logf, y.real, lam=None)
                si = make_spline(old_logf, y.imag, lam=None)
                out[(slice(None),) + idx] = sr(new_logf) + 1j * si(new_logf)
            else:
                s = make_spline(old_logf, y, lam=None)
                out[(slice(None),) + idx] = s(new_logf)
        return out

    for key in ["Z", "T", "P", "Z_err", "T_err", "P_err", "rot"]:
        if key in edi_new and edi_new[key] is not None:
            edi_new[key] = _interp_any(np.asarray(edi_new[key]))

    return edi_new


def set_errors(edi_dict: Dict[str, Any], errors: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Replace error arrays using provided relative errors (placeholder)."""
    edi_new = dict(edi_dict)
    # TODO: implement a clear contract for the `errors` dict and update Z_err/T_err/P_err.
    return edi_new


def rotate_data(edi_dict: Dict[str, Any], angle: float = 0.0, degrees: bool = True) -> Dict[str, Any]:
    """Rotate transfer functions by a fixed horizontal rotation angle."""
    edi_new = dict(edi_dict)

    ang = np.radians(angle) if degrees else float(angle)
    c = np.cos(ang)
    s = np.sin(ang)
    R = np.array([[c, s], [-s, c]], dtype=float)

    if edi_new.get("Z") is not None:
        Z = np.asarray(edi_new["Z"], dtype=np.complex128)
        Zr = np.empty_like(Z)
        for k in range(Z.shape[0]):
            Zr[k] = R @ Z[k] @ R.T
        edi_new["Z"] = Zr

    for key in ["Z_err", "P", "P_err"]:
        if edi_new.get(key) is not None:
            A = np.asarray(edi_new[key], dtype=float)
            Ar = np.empty_like(A)
            for k in range(A.shape[0]):
                Ar[k] = R @ A[k] @ R.T
            edi_new[key] = Ar

    if edi_new.get("T") is not None:
        T = np.asarray(edi_new["T"], dtype=np.complex128)
        Tr = np.empty_like(T)
        for k in range(T.shape[0]):
            Tr[k] = T[k] @ R.T
        edi_new["T"] = Tr

    if edi_new.get("T_err") is not None:
        Te = np.asarray(edi_new["T_err"], dtype=float)
        Ter = np.empty_like(Te)
        for k in range(Te.shape[0]):
            Ter[k] = Te[k] @ R.T
        edi_new["T_err"] = Ter

    if edi_new.get("rot") is not None:
        rot = np.asarray(edi_new["rot"], dtype=float)
        edi_new["rot"] = rot + (angle if degrees else np.degrees(angle)) * np.ones_like(rot)

    return edi_new


# ---------------------------------------------------------------------------
# Error propagation helpers (analytic delta method and parametric bootstrap)
# ---------------------------------------------------------------------------

def _sigma_from_err(
    err: np.ndarray,
    *,
    err_kind: str,
) -> np.ndarray:
    """Convert an error/uncertainty array to 1-sigma standard deviations.

    Parameters
    ----------
    err : numpy.ndarray
        Error array (typically ``Z_err``) of shape ``(n, 2, 2)``.
    err_kind : {"var", "std"}
        Interpretation of ``err``. If ``"var"``, entries are variances and
        are square-rooted. If ``"std"``, entries are already 1-sigma standard
        deviations.

    Returns
    -------
    numpy.ndarray
        1-sigma standard deviations with the same shape as ``err``.

    Notes
    -----
    This helper treats the provided errors as referring to the *complex*
    impedance entry. When generating perturbations, real and imaginary parts
    are assumed independent and receive equal variance, i.e.
    ``Var(Re Z) = Var(Im Z) = sigma_complex**2 / 2``.
    """
    err = np.asarray(err, dtype=float)
    if err_kind == "var":
        return np.sqrt(err)
    if err_kind == "std":
        return err
    raise ValueError("err_kind must be 'var' or 'std'.")


def _z_to_x8(Zk: np.ndarray) -> np.ndarray:
    """Pack a complex 2×2 impedance into an 8-vector [Re/Im per entry].

    Parameters
    ----------
    Zk : numpy.ndarray
        Complex impedance matrix of shape ``(2, 2)``.

    Returns
    -------
    numpy.ndarray
        1-D float array of length 8 ordered as::

            [Zxx_re, Zxx_im, Zxy_re, Zxy_im, Zyx_re, Zyx_im, Zyy_re, Zyy_im]
    """
    Zk = np.asarray(Zk, dtype=np.complex128)
    return np.array(
        [
            Zk[0, 0].real, Zk[0, 0].imag,
            Zk[0, 1].real, Zk[0, 1].imag,
            Zk[1, 0].real, Zk[1, 0].imag,
            Zk[1, 1].real, Zk[1, 1].imag,
        ],
        dtype=float,
    )


def _x8_to_z(x: np.ndarray) -> np.ndarray:
    """Unpack an 8-vector [Re/Im per entry] into a complex 2×2 impedance.

    Parameters
    ----------
    x : numpy.ndarray
        1-D float array of length 8 as produced by :func:`_z_to_x8`.

    Returns
    -------
    numpy.ndarray
        Complex array of shape ``(2, 2)``.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size != 8:
        raise ValueError("x must have length 8.")
    return np.array(
        [
            [x[0] + 1j * x[1], x[2] + 1j * x[3]],
            [x[4] + 1j * x[5], x[6] + 1j * x[7]],
        ],
        dtype=np.complex128,
    )


def _x8_part_variances(sigma_complex_2x2: np.ndarray) -> np.ndarray:
    """Build per-component variances for the 8-vector representation.

    Parameters
    ----------
    sigma_complex_2x2 : numpy.ndarray
        1-sigma standard deviations of complex impedance entries, shape
        ``(2, 2)``.

    Returns
    -------
    numpy.ndarray
        Variances for the packed 8-vector (length 8) assuming independent
        real/imag parts with equal split, i.e. ``Var(Re)=Var(Im)=sigma^2/2``.
    """
    s = np.asarray(sigma_complex_2x2, dtype=float)
    if s.shape != (2, 2):
        raise ValueError("sigma_complex_2x2 must have shape (2, 2).")
    v00 = (s[0, 0] ** 2) / 2.0
    v01 = (s[0, 1] ** 2) / 2.0
    v10 = (s[1, 0] ** 2) / 2.0
    v11 = (s[1, 1] ** 2) / 2.0
    return np.array([v00, v00, v01, v01, v10, v10, v11, v11], dtype=float)


def _finite_diff_jacobian(
    fun,
    x0: np.ndarray,
    *,
    eps: float = 1.0e-6,
) -> np.ndarray:
    """Central finite-difference Jacobian of a vector-valued function.

    Parameters
    ----------
    fun : callable
        Function mapping ``x -> y`` where ``x`` is shape ``(p,)`` and the
        returned ``y`` is 1-D (shape ``(m,)``).
    x0 : numpy.ndarray
        Expansion point (shape ``(p,)``).
    eps : float, optional
        Relative step factor. The step for component i is
        ``h_i = eps * (abs(x0_i) + 1)``.

    Returns
    -------
    numpy.ndarray
        Jacobian matrix with shape ``(m, p)``.
    """
    x0 = np.asarray(x0, dtype=float).ravel()
    y0 = np.asarray(fun(x0), dtype=float).ravel()
    m = y0.size
    p = x0.size
    J = np.zeros((m, p), dtype=float)

    for i in range(p):
        h = eps * (abs(x0[i]) + 1.0)
        xp = x0.copy()
        xm = x0.copy()
        xp[i] += h
        xm[i] -= h
        yp = np.asarray(fun(xp), dtype=float).ravel()
        ym = np.asarray(fun(xm), dtype=float).ravel()
        if yp.size != m or ym.size != m:
            raise ValueError("fun must return a consistent 1-D shape.")
        J[:, i] = (yp - ym) / (2.0 * h)

    return J


def _analytic_var_from_Z(
    Z: np.ndarray,
    sigma: np.ndarray,
    fun_real,
    *,
    fd_eps: float = 1.0e-6,
) -> np.ndarray:
    """Delta-method variances for real-valued outputs derived from Z.

    Parameters
    ----------
    Z : numpy.ndarray
        Complex impedance tensor of shape ``(n, 2, 2)``.
    sigma : numpy.ndarray
        1-sigma standard deviations of the complex impedance entries with the
        same shape as ``Z``.
    fun_real : callable
        Function accepting a complex ``(2, 2)`` impedance matrix and returning
        a real-valued array (any shape). The function is applied per frequency.
    fd_eps : float, optional
        Relative step factor for finite differences.

    Returns
    -------
    numpy.ndarray
        Variances of the flattened output for each frequency, shape
        ``(n, m)`` where ``m`` is the number of output entries. The caller
        can reshape the last dimension to match ``fun_real`` output.
    """
    Z = np.asarray(Z, dtype=np.complex128)
    sigma = np.asarray(sigma, dtype=float)
    if Z.shape != sigma.shape or Z.shape[-2:] != (2, 2):
        raise ValueError("Z and sigma must have the same shape (n, 2, 2).")

    n = Z.shape[0]
    # Determine output size
    y0 = np.asarray(fun_real(Z[0]), dtype=float).ravel()
    m = y0.size
    var_out = np.full((n, m), np.nan, dtype=float)

    for k in range(n):
        Zk = Z[k]
        sigk = sigma[k]
        x0 = _z_to_x8(Zk)
        var_x = _x8_part_variances(sigk)

        def wrap(x):
            return np.asarray(fun_real(_x8_to_z(x)), dtype=float).ravel()

        J = _finite_diff_jacobian(wrap, x0, eps=fd_eps)
        var_out[k] = np.sum((J ** 2) * var_x[None, :], axis=1)

    return var_out


def _analytic_var_complex_scalar_from_Z(
    Z: np.ndarray,
    sigma: np.ndarray,
    fun_complex,
    *,
    fd_eps: float = 1.0e-6,
) -> np.ndarray:
    """Delta-method variance for a complex scalar derived from Z.

    Parameters
    ----------
    Z : numpy.ndarray
        Complex impedance tensor of shape ``(n, 2, 2)``.
    sigma : numpy.ndarray
        1-sigma standard deviations of the complex impedance entries with the
        same shape as ``Z``.
    fun_complex : callable
        Function accepting a complex ``(2, 2)`` impedance matrix and returning
        a complex scalar (Python complex or 0-D numpy complex).
    fd_eps : float, optional
        Relative step factor for finite differences.

    Returns
    -------
    numpy.ndarray
        Variance proxy for the complex output, shape ``(n,)``. We compute
        variances of real and imaginary parts and return their sum.
    """
    def fun_realimag(Zk):
        v = complex(fun_complex(Zk))
        return np.array([v.real, v.imag], dtype=float)

    var_ri = _analytic_var_from_Z(Z, sigma, fun_realimag, fd_eps=fd_eps)
    return var_ri[:, 0] + var_ri[:, 1]


def compute_pt(
    Z: np.ndarray,
    Z_err: Optional[np.ndarray] = None,
    *,
    err_kind: str = "var",
    err_method: str = "bootstrap",
    nsim: int = 200,
    fd_eps: float = 1.0e-6,
    random_state: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Compute phase tensor and optionally propagate impedance errors.

    Parameters
    ----------
    Z : numpy.ndarray
        Complex impedance tensor of shape ``(n, 2, 2)``.
    Z_err : numpy.ndarray or None, optional
        Impedance uncertainties with the same shape as ``Z``. Interpretation
        is set by ``err_kind``.
    err_kind : {"var", "std"}, optional
        Specifies whether ``Z_err`` contains variances (``"var"``) or
        1-sigma standard deviations (``"std"``) of the *complex* impedance
        entries. Default is ``"var"``.
    err_method : {"none", "analytic", "bootstrap", "both"}, optional
        Error propagation method:

        - ``"none"``: do not compute errors (return ``PT_err=None``).
        - ``"analytic"``: delta-method (finite-difference Jacobian).
        - ``"bootstrap"``: parametric bootstrap (Monte-Carlo) using ``nsim``.
        - ``"both"``: return a dict with both error arrays.

        Default is ``"bootstrap"`` to keep backward behaviour.
    nsim : int, optional
        Number of bootstrap realisations for ``err_method="bootstrap"``.
        Default is 200.
    fd_eps : float, optional
        Relative finite-difference step for ``err_method="analytic"``.
        Default is 1e-6.
    random_state : numpy.random.Generator, optional
        Random generator to use for bootstrap. If None, a fresh generator is
        created.

    Returns
    -------
    PT : numpy.ndarray
        Phase tensor array of shape ``(n, 2, 2)`` (real-valued).
    PT_err : numpy.ndarray or None
        Error estimate with shape ``(n, 2, 2)`` in the same convention as
        ``err_kind`` (variance if ``"var"``, standard deviation if ``"std"``),
        or a dict ``{"analytic": ..., "bootstrap": ...}`` if
        ``err_method="both"``.
    """
    Z = np.asarray(Z, dtype=np.complex128)
    if Z.shape[-2:] != (2, 2):
        raise ValueError("Z must have shape (n, 2, 2).")

    n = Z.shape[0]
    PT = np.full((n, 2, 2), np.nan, dtype=float)

    # Deterministic phase tensor
    for k in range(n):
        X = Z[k].real
        Y = Z[k].imag
        try:
            X_inv = np.linalg.inv(X)
        except np.linalg.LinAlgError:
            continue
        PT[k] = X_inv @ Y

    if Z_err is None or err_method.lower() in ("none", "off", "false", "0"):
        return PT, None

    Z_err = np.asarray(Z_err, dtype=float)
    if Z_err.shape != Z.shape:
        raise ValueError("Z_err must have the same shape as Z.")

    method = err_method.lower()
    if method not in ("analytic", "bootstrap", "both"):
        raise ValueError("err_method must be one of: 'none', 'analytic', 'bootstrap', 'both'.")

    sigma = _sigma_from_err(Z_err, err_kind=err_kind)

    # ---------------------- analytic (delta method)
    PT_err_analytic = None
    if method in ("analytic", "both"):

        def fun_real(Zk):
            X = Zk.real
            Y = Zk.imag
            try:
                X_inv = np.linalg.inv(X)
            except np.linalg.LinAlgError:
                return np.full((2, 2), np.nan, dtype=float)
            return (X_inv @ Y)

        var_flat = _analytic_var_from_Z(Z, sigma, fun_real, fd_eps=fd_eps)
        var_PT = var_flat.reshape(n, 2, 2)
        PT_err_analytic = var_PT if err_kind == "var" else np.sqrt(var_PT)

    # ---------------------- bootstrap (parametric)
    PT_err_boot = None
    if method in ("bootstrap", "both"):
        rng = np.random.default_rng() if random_state is None else random_state
        P_sims = np.full((nsim, n, 2, 2), np.nan, dtype=float)

        for sidx in range(nsim):
            d_re = rng.standard_normal(Z.shape) * sigma / np.sqrt(2.0)
            d_im = rng.standard_normal(Z.shape) * sigma / np.sqrt(2.0)
            Zs = (Z.real + d_re) + 1j * (Z.imag + d_im)

            for k in range(n):
                X = Zs[k].real
                Y = Zs[k].imag
                try:
                    X_inv = np.linalg.inv(X)
                except np.linalg.LinAlgError:
                    continue
                P_sims[sidx, k] = X_inv @ Y

        with np.errstate(invalid="ignore"):
            var_PT = np.nanvar(P_sims, axis=0)
        PT_err_boot = var_PT if err_kind == "var" else np.sqrt(var_PT)

    if method == "analytic":
        return PT, PT_err_analytic
    if method == "bootstrap":
        return PT, PT_err_boot
    return PT, {"analytic": PT_err_analytic, "bootstrap": PT_err_boot}


def compute_zdet(
    Z: np.ndarray,
    Z_err: Optional[np.ndarray] = None,
    *,
    err_kind: str = "var",
    err_method: str = "bootstrap",
    nsim: int = 200,
    fd_eps: float = 1.0e-6,
    random_state: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Compute Berdichevsky's determinant rotational invariant of impedance.

    The determinant impedance is:

        ``Z_det = sqrt(det(Z)) = sqrt(Z_xx Z_yy - Z_xy Z_yx)``

    Parameters
    ----------
    Z : numpy.ndarray
        Complex impedance tensor of shape ``(n, 2, 2)``.
    Z_err : numpy.ndarray or None, optional
        Impedance uncertainties with the same shape as ``Z``. Interpretation
        is controlled by ``err_kind``.
    err_kind : {"var", "std"}, optional
        Interpretation of ``Z_err`` (variance or standard deviation of the
        complex impedance entries). Default is ``"var"``.
    err_method : {"none", "analytic", "bootstrap", "both"}, optional
        Error propagation method. See :func:`compute_pt` for details.
        Default is ``"bootstrap"``.
    nsim : int, optional
        Number of bootstrap realisations (if requested). Default is 200.
    fd_eps : float, optional
        Relative step for the analytic delta method. Default is 1e-6.
    random_state : numpy.random.Generator, optional
        Random generator for bootstrap.

    Returns
    -------
    zdet : numpy.ndarray
        Complex determinant impedance, shape ``(n,)``.
    zdet_err : numpy.ndarray or None
        Error estimate of shape ``(n,)`` (variance or standard deviation,
        depending on ``err_kind``), or a dict if ``err_method="both"``.
    """
    Z = np.asarray(Z, dtype=np.complex128)
    if Z.shape[-2:] != (2, 2):
        raise ValueError("Z must have shape (n, 2, 2).")

    detZ = Z[:, 0, 0] * Z[:, 1, 1] - Z[:, 0, 1] * Z[:, 1, 0]
    zdet = np.sqrt(detZ)

    if Z_err is None or err_method.lower() in ("none", "off", "false", "0"):
        return zdet, None

    Z_err = np.asarray(Z_err, dtype=float)
    if Z_err.shape != Z.shape:
        raise ValueError("Z_err must have the same shape as Z.")

    method = err_method.lower()
    if method not in ("analytic", "bootstrap", "both"):
        raise ValueError("err_method must be one of: 'none', 'analytic', 'bootstrap', 'both'.")

    sigma = _sigma_from_err(Z_err, err_kind=err_kind)

    # analytic
    zdet_err_analytic = None
    if method in ("analytic", "both"):

        def fun_complex(Zk):
            detk = Zk[0, 0] * Zk[1, 1] - Zk[0, 1] * Zk[1, 0]
            return np.sqrt(detk)

        var_c = _analytic_var_complex_scalar_from_Z(Z, sigma, fun_complex, fd_eps=fd_eps)
        zdet_err_analytic = var_c if err_kind == "var" else np.sqrt(var_c)

    # bootstrap
    zdet_err_boot = None
    if method in ("bootstrap", "both"):
        rng = np.random.default_rng() if random_state is None else random_state
        sims = np.full((nsim, zdet.size), np.nan, dtype=np.complex128)

        for sidx in range(nsim):
            d_re = rng.standard_normal(Z.shape) * sigma / np.sqrt(2.0)
            d_im = rng.standard_normal(Z.shape) * sigma / np.sqrt(2.0)
            Zs = (Z.real + d_re) + 1j * (Z.imag + d_im)
            dets = Zs[:, 0, 0] * Zs[:, 1, 1] - Zs[:, 0, 1] * Zs[:, 1, 0]
            sims[sidx] = np.sqrt(dets)

        with np.errstate(invalid="ignore"):
            var_re = np.nanvar(sims.real, axis=0)
            var_im = np.nanvar(sims.imag, axis=0)
        var_c = var_re + var_im
        zdet_err_boot = var_c if err_kind == "var" else np.sqrt(var_c)

    if method == "analytic":
        return zdet, zdet_err_analytic
    if method == "bootstrap":
        return zdet, zdet_err_boot
    return zdet, {"analytic": zdet_err_analytic, "bootstrap": zdet_err_boot}


def compute_zssq(
    Z: np.ndarray,
    Z_err: Optional[np.ndarray] = None,
    *,
    err_kind: str = "var",
    err_method: str = "bootstrap",
    nsim: int = 200,
    fd_eps: float = 1.0e-6,
    random_state: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Compute the ssq rotational invariant (Rung–Arunwan / Szarka–Menvielle).

    Definition (one common convention):

        ``Z_ssq = sqrt( (Z_xx^2 + Z_xy^2 + Z_yx^2 + Z_yy^2) / 2 )``

    Parameters
    ----------
    Z : numpy.ndarray
        Complex impedance tensor of shape ``(n, 2, 2)``.
    Z_err : numpy.ndarray or None, optional
        Impedance uncertainties with the same shape as ``Z``. Interpretation
        is controlled by ``err_kind``.
    err_kind : {"var", "std"}, optional
        Interpretation of ``Z_err`` (variance or standard deviation of the
        complex impedance entries). Default is ``"var"``.
    err_method : {"none", "analytic", "bootstrap", "both"}, optional
        Error propagation method. See :func:`compute_pt` for details.
        Default is ``"bootstrap"``.
    nsim : int, optional
        Number of bootstrap realisations (if requested). Default is 200.
    fd_eps : float, optional
        Relative step for the analytic delta method. Default is 1e-6.
    random_state : numpy.random.Generator, optional
        Random generator for bootstrap.

    Returns
    -------
    zssq : numpy.ndarray
        Complex ssq impedance, shape ``(n,)``.
    zssq_err : numpy.ndarray or None
        Error estimate of shape ``(n,)`` (variance or standard deviation,
        depending on ``err_kind``), or a dict if ``err_method="both"``.
    """
    Z = np.asarray(Z, dtype=np.complex128)
    if Z.shape[-2:] != (2, 2):
        raise ValueError("Z must have shape (n, 2, 2).")

    ssqZ = (Z[:, 0, 0] ** 2 + Z[:, 0, 1] ** 2 + Z[:, 1, 0] ** 2 + Z[:, 1, 1] ** 2) / 2.0
    zssq = np.sqrt(ssqZ)

    if Z_err is None or err_method.lower() in ("none", "off", "false", "0"):
        return zssq, None

    Z_err = np.asarray(Z_err, dtype=float)
    if Z_err.shape != Z.shape:
        raise ValueError("Z_err must have the same shape as Z.")

    method = err_method.lower()
    if method not in ("analytic", "bootstrap", "both"):
        raise ValueError("err_method must be one of: 'none', 'analytic', 'bootstrap', 'both'.")

    sigma = _sigma_from_err(Z_err, err_kind=err_kind)

    # analytic
    zssq_err_analytic = None
    if method in ("analytic", "both"):

        def fun_complex(Zk):
            ssqk = (Zk[0, 0] ** 2 + Zk[0, 1] ** 2 + Zk[1, 0] ** 2 + Zk[1, 1] ** 2) / 2.0
            return np.sqrt(ssqk)

        var_c = _analytic_var_complex_scalar_from_Z(Z, sigma, fun_complex, fd_eps=fd_eps)
        zssq_err_analytic = var_c if err_kind == "var" else np.sqrt(var_c)

    # bootstrap
    zssq_err_boot = None
    if method in ("bootstrap", "both"):
        rng = np.random.default_rng() if random_state is None else random_state
        sims = np.full((nsim, zssq.size), np.nan, dtype=np.complex128)

        for sidx in range(nsim):
            d_re = rng.standard_normal(Z.shape) * sigma / np.sqrt(2.0)
            d_im = rng.standard_normal(Z.shape) * sigma / np.sqrt(2.0)
            Zs = (Z.real + d_re) + 1j * (Z.imag + d_im)
            ssq = (Zs[:, 0, 0] ** 2 + Zs[:, 0, 1] ** 2 + Zs[:, 1, 0] ** 2 + Zs[:, 1, 1] ** 2) / 2.0
            sims[sidx] = np.sqrt(ssq)

        with np.errstate(invalid="ignore"):
            var_re = np.nanvar(sims.real, axis=0)
            var_im = np.nanvar(sims.imag, axis=0)
        var_c = var_re + var_im
        zssq_err_boot = var_c if err_kind == "var" else np.sqrt(var_c)

    if method == "analytic":
        return zssq, zssq_err_analytic
    if method == "bootstrap":
        return zssq, zssq_err_boot
    return zssq, {"analytic": zssq_err_analytic, "bootstrap": zssq_err_boot}


def compute_rhoplus(
    freq: np.ndarray,
    Z_scalar: np.ndarray,
    Z_scalar_err: Optional[np.ndarray] = None,
    *,
    n_lambda_per_freq: int = 4,
    mu0: float = _MU0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the D⁺ (rho-plus) upper bound on apparent resistivity.

    The D⁺ test (Parker 1980; Parker & Whaler 1981) checks whether a scalar
    impedance dataset is consistent with *any* 1-D conductivity profile.  For
    a valid 1-D response the measured apparent resistivity ρ_a must satisfy

        ρ_a(ω) ≤ ρ⁺(ω)   for all ω,

    where ρ⁺ is the apparent resistivity of the optimum D⁺ model — the
    physically admissible response that best fits the observed data.

    The D⁺ model has the form

        c(ω) = Σ_k  Δa_k / (λ_k + iω),   Δa_k ≥ 0, λ_k > 0,

    where c = Z / (iωμ₀) is the Schmucker c-response.  Given a fixed grid of
    λ values, the coefficients Δa_k are found by non-negative least squares
    (NNLS).  The resulting ρ⁺(ω) = μ₀ω |c⁺(ω)|² is then an upper bound: any
    single ρ_a(ω_j) > ρ⁺(ω_j) signals an inconsistency with a 1-D model.

    Also known as **dplus** in Cordell's mtcode (Cordell et al., 2022).

    Parameters
    ----------
    freq : array_like, shape (n,)
        Frequencies in Hz.  Must be positive and in any order; they are
        sorted internally.
    Z_scalar : array_like, shape (n,)
        Complex scalar impedance at each frequency [V/m per A/m = Ω].
        Typically one of: Zxy, Zyx, Z_det, or Z_ssq.
    Z_scalar_err : array_like, shape (n,) or None, optional
        Standard deviation (or square-root of variance) of each complex
        impedance value.  When provided, each row of the NNLS system is
        divided by the corresponding σ so that noisier data contributes less.
        Pass ``None`` (default) for unweighted fitting.
    n_lambda_per_freq : int, optional
        Number of λ grid points per data frequency.  The λ grid spans
        [ω_min / 100, ω_max × 100] on a log scale with
        ``n_lambda_per_freq * n`` points total.  Default is 4.
    mu0 : float, optional
        Magnetic permeability.  Default is ``_MU0`` (vacuum).

    Returns
    -------
    rho_plus : numpy.ndarray, shape (n,)
        D⁺ upper bound on apparent resistivity [Ω·m] at each frequency.
    rho_a : numpy.ndarray, shape (n,)
        Observed apparent resistivity ρ_a = |Z|² / (μ₀ω) [Ω·m].
    pass_test : numpy.ndarray of bool, shape (n,)
        ``True`` where ρ_a ≤ ρ⁺ (consistent with 1-D); ``False`` where
        ρ_a > ρ⁺ (violation).

    Notes
    -----
    The λ grid covers many decades beyond the data bandwidth.  The NNLS
    solve is typically very fast (O(n²) for n ≲ 100 frequencies).

    The test is a *necessary* condition: a pass does not prove 1-D
    consistency; a failure proves that no 1-D model can explain the data.

    For tensor data, apply this function to each desired component
    independently, e.g. Zxy, Zyx, Z_det, or Z_ssq.

    References
    ----------
    Parker, R. L. (1980). The inverse problem of electromagnetic induction:
    existence and construction of solutions based on incomplete data.
    *J. Geophys. Res.*, 85, 4421–4428.

    Parker, R. L., & Whaler, K. A. (1981). Numerical methods for establishing
    solutions to the inverse problem of electromagnetic induction.
    *J. Geophys. Res.*, 86, 9574–9584.

    Parker, R. L., & Booker, J. R. (1996). Optimal one-dimensional inversion
    and bounding of magnetotelluric apparent resistivity and phase
    measurements. *Phys. Earth Planet. Inter.*, 98, 269–282.

    Cordell, D., Lee, B., & Unsworth, M. J. (2022). mtcode: A repository of
    MATLAB scripts for magnetotelluric data analysis, data editing, model
    building, and model viewing. doi:10.5281/zenodo.6784201
    https://github.com/darcycordell/mtcode

    Examples
    --------
    >>> rho_plus, rho_a, ok = compute_rhoplus(site['freq'], site['Z'][:, 0, 1])
    >>> print('D+ violations:', (~ok).sum(), 'of', len(ok))
    """
    from scipy.optimize import nnls as _nnls

    freq = np.asarray(freq, dtype=float).ravel()
    Z_scalar = np.asarray(Z_scalar, dtype=np.complex128).ravel()
    if Z_scalar.shape != freq.shape:
        raise ValueError("freq and Z_scalar must have the same length.")

    # Sort by frequency
    order = np.argsort(freq)
    freq = freq[order]
    Z_scalar = Z_scalar[order]

    omega = 2.0 * np.pi * freq
    n = freq.size

    # c-response: c = Z / (i * omega * mu0)
    c = Z_scalar / (1j * omega * mu0)

    # Lambda grid
    n_lam = n_lambda_per_freq * n
    lam = np.logspace(
        np.log10(omega.min() / 100.0),
        np.log10(omega.max() * 100.0),
        n_lam,
    )

    # Build real-valued system matrix A (2n x n_lam)
    #   Re[1/(lam+iw)] =  lam / (lam^2 + w^2)
    #   Im[1/(lam+iw)] = -w   / (lam^2 + w^2)
    lam2 = lam[np.newaxis, :] ** 2          # (1, n_lam)
    w2 = (omega ** 2)[:, np.newaxis]        # (n, 1)
    denom = lam2 + w2                       # (n, n_lam)

    A_re = lam[np.newaxis, :] / denom       # (n, n_lam)
    A_im = -omega[:, np.newaxis] / denom    # (n, n_lam)
    A = np.vstack([A_re, A_im])             # (2n, n_lam)

    b = np.concatenate([c.real, c.imag])    # (2n,)

    # Optional error weighting
    if Z_scalar_err is not None:
        Z_scalar_err = np.asarray(Z_scalar_err, dtype=float).ravel()[order]
        # sigma of c = sigma_Z / (omega * mu0)
        sigma_c = Z_scalar_err / (omega * mu0)
        sigma_c = np.where(sigma_c > 0, sigma_c, sigma_c[sigma_c > 0].min())
        weights = np.concatenate([1.0 / sigma_c, 1.0 / sigma_c])
        A = A * weights[:, np.newaxis]
        b = b * weights

    # Non-negative least squares: da_k >= 0
    da, _ = _nnls(A, b)

    # Reconstruct c+ from the D+ model (using the *unweighted* A columns)
    if Z_scalar_err is not None:
        # Rebuild unweighted matrix for reconstruction
        A_re_uw = lam[np.newaxis, :] / (lam2 + w2)
        A_im_uw = -omega[:, np.newaxis] / (lam2 + w2)
        c_plus = A_re_uw @ da + 1j * (A_im_uw @ da)
    else:
        c_plus = A_re @ da + 1j * (A_im @ da)

    rho_plus = mu0 * omega * np.abs(c_plus) ** 2
    rho_a = np.abs(Z_scalar) ** 2 / (mu0 * omega)
    pass_test = rho_a <= rho_plus

    return rho_plus, rho_a, pass_test


def _format_block(values: np.ndarray, n_per_line: int = 6) -> str:
    """Format a 1D array into an EDI numeric block string."""
    vals = np.asarray(values, dtype=float).ravel()
    chunks: List[str] = []
    for i in range(0, vals.size, n_per_line):
        line_vals = vals[i : i + n_per_line]
        line = " ".join(f"{v: .6E}" for v in line_vals)
        chunks.append(" " + line)
    return "\n".join(chunks)


def save_edi(
    edi: Optional[Dict[str, Any]] = None,
    *,
    path: str | Path,
    numbers_per_line: int = 6,
    add_pt_blocks: bool = True,
    pt_err_kind: Optional[str] = None,
    lon_keyword: str = "LON",
    **kwargs: Any,
) -> None:
    """Write an EDI dictionary to a classical table-style EDI file.

    Can be called as either::

        save_edi(data_dict, path="out.edi")
        save_edi(**data_dict, path="out.edi")
    """
    if edi is None:
        edi = kwargs
    p = Path(path)

    freq = np.asarray(edi["freq"], dtype=float).ravel()
    nfreq = freq.size

    Z = np.asarray(edi["Z"], dtype=np.complex128)
    if Z.shape != (freq.size, 2, 2):
        raise ValueError("edi['Z'] must have shape (n, 2, 2) matching freq.")

    Z_err = edi.get("Z_err")
    T = edi.get("T")
    T_err = edi.get("T_err")
    P = edi.get("P")
    P_err = edi.get("P_err")
    rot = edi.get("rot")

    station = edi.get("station", "UNKNOWN")
    lat_deg = edi.get("lat_deg", edi.get("lat"))
    lon_deg = edi.get("lon_deg", edi.get("lon"))
    elev_m = edi.get("elev_m", edi.get("elev"))
    err_kind = edi.get("err_kind", "var")

    # Determine frequency order for the >FREQ header tag.
    # Use stored freq_order if available; otherwise infer from the data.
    freq_order = edi.get("freq_order")
    if freq_order is None:
        if nfreq < 2:
            freq_order = "inc"
        elif freq[-1] > freq[0]:
            freq_order = "inc"
        elif freq[-1] < freq[0]:
            freq_order = "dec"
        else:
            freq_order = "keep"
    _order_tag = str(freq_order).upper()

    lines: List[str] = []

    # HEAD
    lines.append(">HEAD")
    lines.append(f'  DATAID="{station}"')
    if lat_deg is not None:
        lines.append(f"  LAT={float(lat_deg): .6f}")
    if lon_deg is not None:
        lines.append(f"  {lon_keyword}={float(lon_deg): .6f}")
    if elev_m is not None:
        lines.append(f"  ELEV={float(elev_m): .6f}")
    lines.append('  STDVERS="SEG 1.0"')
    lines.append('  PROGVERS="data_proc.py"')
    # lines.append('  PROGDATE="2025-12-21"')
    lines.append(f'  PROGDATE="{date.today().isoformat()}"')
    lines.append("  EMPTY=1.0E32")
    lines.append(f"  NFREQ={int(nfreq)}")
    lines.append("")

    # INFO
    if edi.get("info"):
        lines.append(">INFO")
        lines.append(edi.get("info"))

    # FREQ
    lines.append(f">FREQ NFREQ={int(nfreq)} ORDER={_order_tag} //{int(nfreq)}")
    lines.append(_format_block(freq, n_per_line=numbers_per_line))
    lines.append("")

    # Optional rotation block (ZROT)
    if rot is not None:
        rot_arr = np.asarray(rot, dtype=float).ravel()
        if rot_arr.size != freq.size:
            raise ValueError("rot must have same length as freq.")
        lines.append(">ZROT")
        lines.append(_format_block(rot_arr, n_per_line=numbers_per_line))
        lines.append("")

    # Impedance blocks
    comp_map = {"ZXX": (0, 0), "ZXY": (0, 1), "ZYX": (1, 0), "ZYY": (1, 1)}

    z_real_blocks: Dict[str, np.ndarray] = {}
    z_imag_blocks: Dict[str, np.ndarray] = {}
    z_var_blocks: Dict[str, np.ndarray] = {}

    for base, (i, j) in comp_map.items():
        Zij = Z[:, i, j]
        z_real_blocks[base + "R"] = Zij.real
        z_imag_blocks[base + "I"] = Zij.imag

        if Z_err is not None:
            Z_err_arr = np.asarray(Z_err, dtype=float)
            if Z_err_arr.shape != Z.shape:
                raise ValueError("edi['Z_err'] must have the same shape as edi['Z'].")
            var = Z_err_arr[:, i, j] ** 2 if err_kind == "std" else Z_err_arr[:, i, j]
            z_var_blocks[base + ".VAR"] = var

    for key, arr in z_real_blocks.items():
        lines.append(">" + key)
        lines.append(_format_block(arr, n_per_line=numbers_per_line))
        lines.append("")
    for key, arr in z_imag_blocks.items():
        lines.append(">" + key)
        lines.append(_format_block(arr, n_per_line=numbers_per_line))
        lines.append("")
    for key, arr in z_var_blocks.items():
        lines.append(">" + key)
        lines.append(_format_block(arr, n_per_line=numbers_per_line))
        lines.append("")

    # Tipper blocks
    n = freq.size
    if T is not None:
        T = np.asarray(T, dtype=np.complex128)
        if T.shape != (n, 1, 2):
            raise ValueError("edi['T'] must have shape (n, 1, 2).")

        Tx = T[:, 0, 0]
        Ty = T[:, 0, 1]

        lines.append(">TXR.EXP")
        lines.append(_format_block(Tx.real, n_per_line=numbers_per_line))
        lines.append("")
        lines.append(">TXI.EXP")
        lines.append(_format_block(Tx.imag, n_per_line=numbers_per_line))
        lines.append("")
        lines.append(">TYR.EXP")
        lines.append(_format_block(Ty.real, n_per_line=numbers_per_line))
        lines.append("")
        lines.append(">TYI.EXP")
        lines.append(_format_block(Ty.imag, n_per_line=numbers_per_line))
        lines.append("")

        if T_err is not None:
            T_err_arr = np.asarray(T_err, dtype=float)
            if T_err_arr.shape != T.shape:
                raise ValueError("edi['T_err'] must have the same shape as edi['T'].")
            txvar = T_err_arr[:, 0, 0] ** 2 if err_kind == "std" else T_err_arr[:, 0, 0]
            tyvar = T_err_arr[:, 0, 1] ** 2 if err_kind == "std" else T_err_arr[:, 0, 1]

            lines.append(">TXVAR.EXP")
            lines.append(_format_block(txvar, n_per_line=numbers_per_line))
            lines.append("")
            lines.append(">TYVAR.EXP")
            lines.append(_format_block(tyvar, n_per_line=numbers_per_line))
            lines.append("")

    # Phase tensor blocks
    if add_pt_blocks and P is not None:
        P = np.asarray(P, dtype=float)
        if P.shape != (n, 2, 2):
            raise ValueError("edi['P'] must have shape (n, 2, 2).")

        lines.append(">PXX")
        lines.append(_format_block(P[:, 0, 0], n_per_line=numbers_per_line))
        lines.append("")
        lines.append(">PXY")
        lines.append(_format_block(P[:, 0, 1], n_per_line=numbers_per_line))
        lines.append("")
        lines.append(">PYX")
        lines.append(_format_block(P[:, 1, 0], n_per_line=numbers_per_line))
        lines.append("")
        lines.append(">PYY")
        lines.append(_format_block(P[:, 1, 1], n_per_line=numbers_per_line))
        lines.append("")

        if P_err is not None:
            P_err_arr = np.asarray(P_err, dtype=float)
            if P_err_arr.shape != P.shape:
                raise ValueError("edi['P_err'] must have the same shape as edi['P'].")

            if pt_err_kind is None:
                pt_err_kind = err_kind

            if pt_err_kind == "std":
                var_xx = P_err_arr[:, 0, 0] ** 2
                var_xy = P_err_arr[:, 0, 1] ** 2
                var_yx = P_err_arr[:, 1, 0] ** 2
                var_yy = P_err_arr[:, 1, 1] ** 2
            elif pt_err_kind == "var":
                var_xx = P_err_arr[:, 0, 0]
                var_xy = P_err_arr[:, 0, 1]
                var_yx = P_err_arr[:, 1, 0]
                var_yy = P_err_arr[:, 1, 1]
            else:
                raise ValueError("pt_err_kind must be 'std', 'var', or None.")

            lines.append(">PXX.VAR")
            lines.append(_format_block(var_xx, n_per_line=numbers_per_line))
            lines.append("")
            lines.append(">PXY.VAR")
            lines.append(_format_block(var_xy, n_per_line=numbers_per_line))
            lines.append("")
            lines.append(">PYX.VAR")
            lines.append(_format_block(var_yx, n_per_line=numbers_per_line))
            lines.append("")
            lines.append(">PYY.VAR")
            lines.append(_format_block(var_yy, n_per_line=numbers_per_line))
            lines.append("")

    p.write_text("\n".join(lines) + "\n", encoding="latin-1")


def dataframe_from_edi(
    edi: Dict[str, Any],
    *,
    include_rho_phi: bool = True,
    include_tipper: bool = True,
    include_pt: bool = True,
    add_period: bool = True,
    err_kind: Optional[str] = None,
    mu0: float = _MU0,
) -> pd.DataFrame:
    """Convert an EDI dictionary to a tidy pandas DataFrame."""
    if "freq" not in edi or "Z" not in edi:
        raise KeyError("edi must contain at least 'freq' and 'Z' keys.")

    freq = np.asarray(edi["freq"], dtype=float).ravel()
    Z = np.asarray(edi["Z"], dtype=np.complex128)
    if Z.shape != (freq.size, 2, 2):
        raise ValueError("edi['Z'] must have shape (n, 2, 2) matching freq.")

    n = freq.size
    df = pd.DataFrame({"freq": freq})

    if add_period:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["period"] = np.where(freq > 0.0, 1.0 / freq, np.nan)

    rot = edi.get("rot")
    if rot is not None:
        rot_arr = np.asarray(rot, dtype=float).ravel()
        if rot_arr.size == n:
            df["rot"] = rot_arr

    if err_kind is None:
        err_kind = edi.get("err_kind", "var")
    if err_kind not in ("var", "std"):
        raise ValueError("err_kind must be 'var' or 'std'.")

    if include_rho_phi:
        omega = 2.0 * np.pi * freq
        comp_map = {"xx": (0, 0), "xy": (0, 1), "yx": (1, 0), "yy": (1, 1)}

        # Prefer precomputed apparent resistivity / phase if present in the EDI dict.
        # Supported conventions:
        #   - tensor arrays:  rho, phi   with shape (n,2,2) or (n,4)
        #   - per-component:  rho_xx, rho_xy, ... and phi_xx, phi_xy, ...
        used_precomputed = False

        rho_arr = None
        phi_arr = None

        for k in ("rho", "rhoa", "rho_a"):
            if k in edi:
                rho_arr = edi[k]
                break
        for k in ("phi", "pha", "phase"):
            if k in edi:
                phi_arr = edi[k]
                break

        if rho_arr is not None and phi_arr is not None:
            rho_arr = np.asarray(rho_arr, dtype=float)
            phi_arr = np.asarray(phi_arr, dtype=float)

            if rho_arr.ndim == 2 and rho_arr.shape == (n, 4):
                rho_arr = rho_arr.reshape(n, 2, 2)
            if phi_arr.ndim == 2 and phi_arr.shape == (n, 4):
                phi_arr = phi_arr.reshape(n, 2, 2)

            if rho_arr.shape != (n, 2, 2) or phi_arr.shape != (n, 2, 2):
                raise ValueError(
                    "Precomputed rho/phi must have shape (n,2,2) or (n,4) matching freq."
                )

            for name, (i, j) in comp_map.items():
                df[f"rho_{name}"] = rho_arr[:, i, j]
                df[f"phi_{name}"] = phi_arr[:, i, j]

            # Optional precomputed errors (variance or std, controlled by err_kind).
            rho_err = None
            phi_err = None
            for k in ("rho_err", "rhoa_err", "rho_a_err"):
                if k in edi:
                    rho_err = edi[k]
                    break
            for k in ("phi_err", "pha_err", "phase_err"):
                if k in edi:
                    phi_err = edi[k]
                    break

            if rho_err is not None:
                rho_err = np.asarray(rho_err, dtype=float)
                if rho_err.ndim == 2 and rho_err.shape == (n, 4):
                    rho_err = rho_err.reshape(n, 2, 2)
                if rho_err.shape != (n, 2, 2):
                    raise ValueError("Precomputed rho_err must have shape (n,2,2) or (n,4).")
                sigma_rho = np.sqrt(rho_err) if err_kind == "var" else rho_err
                for name, (i, j) in comp_map.items():
                    df[f"rho_{name}_err"] = sigma_rho[:, i, j]

            if phi_err is not None:
                phi_err = np.asarray(phi_err, dtype=float)
                if phi_err.ndim == 2 and phi_err.shape == (n, 4):
                    phi_err = phi_err.reshape(n, 2, 2)
                if phi_err.shape != (n, 2, 2):
                    raise ValueError("Precomputed phi_err must have shape (n,2,2) or (n,4).")
                sigma_phi = np.sqrt(phi_err) if err_kind == "var" else phi_err
                for name, (i, j) in comp_map.items():
                    df[f"phi_{name}_err"] = sigma_phi[:, i, j]

            used_precomputed = True

        if not used_precomputed:
            have_rho_cols = all(f"rho_{name}" in edi for name in comp_map)
            have_phi_cols = all(f"phi_{name}" in edi for name in comp_map)
            if have_rho_cols and have_phi_cols:
                for name in comp_map:
                    r = np.asarray(edi[f"rho_{name}"], dtype=float).ravel()
                    p = np.asarray(edi[f"phi_{name}"], dtype=float).ravel()
                    if r.size != n or p.size != n:
                        raise ValueError("Precomputed rho_*/phi_* must have length n.")
                    df[f"rho_{name}"] = r
                    df[f"phi_{name}"] = p

                # Optional per-component errors (variance or std)
                have_rho_err_cols = all(f"rho_{name}_err" in edi for name in comp_map)
                if have_rho_err_cols:
                    for name in comp_map:
                        e = np.asarray(edi[f"rho_{name}_err"], dtype=float).ravel()
                        if e.size != n:
                            raise ValueError("Precomputed rho_*_err must have length n.")
                        df[f"rho_{name}_err"] = np.sqrt(e) if err_kind == "var" else e

                have_phi_err_cols = all(f"phi_{name}_err" in edi for name in comp_map)
                if have_phi_err_cols:
                    for name in comp_map:
                        e = np.asarray(edi[f"phi_{name}_err"], dtype=float).ravel()
                        if e.size != n:
                            raise ValueError("Precomputed phi_*_err must have length n.")
                        df[f"phi_{name}_err"] = np.sqrt(e) if err_kind == "var" else e

                used_precomputed = True

        if not used_precomputed:
            for name, (i, j) in comp_map.items():
                Zij = Z[:, i, j]
                mag = np.abs(Zij)
                df[f"rho_{name}"] = (mag**2) / (mu0 * omega)
                with np.errstate(divide="ignore", invalid="ignore"):
                    df[f"phi_{name}"] = np.degrees(np.arctan2(Zij.imag, Zij.real))

        # If rho/phi errors were not provided, optionally propagate from Z_err
        Z_err = edi.get("Z_err")
        if Z_err is not None:
            Z_err = np.asarray(Z_err, dtype=float)
            if Z_err.shape != Z.shape:
                raise ValueError("edi['Z_err'] must have the same shape as 'Z'.")

            sigma_Z = np.sqrt(Z_err) if err_kind == "var" else Z_err

            for name, (i, j) in comp_map.items():
                mag = np.abs(Z[:, i, j])
                denom = mu0 * omega
                std_rho = 2.0 * sigma_Z[:, i, j] * mag / np.where(denom > 0.0, denom, np.nan)
                if f"rho_{name}_err" not in df.columns:
                    df[f"rho_{name}_err"] = std_rho

                with np.errstate(divide="ignore", invalid="ignore"):
                    std_phi_rad = np.where(mag > 0.0, sigma_Z[:, i, j] / mag, np.nan)
                if f"phi_{name}_err" not in df.columns:
                    df[f"phi_{name}_err"] = std_phi_rad * (180.0 / np.pi)

    if include_tipper and edi.get("T") is not None:
        T = np.asarray(edi["T"], dtype=np.complex128)
        if T.shape != (n, 1, 2):
            raise ValueError("edi['T'] must have shape (n, 1, 2).")
        tx = T[:, 0, 0]
        ty = T[:, 0, 1]
        df["Tx_re"] = tx.real
        df["Tx_im"] = tx.imag
        df["Ty_re"] = ty.real
        df["Ty_im"] = ty.imag

        T_err = edi.get("T_err")
        if T_err is not None:
            T_err = np.asarray(T_err, dtype=float)
            if T_err.shape != T.shape:
                raise ValueError("edi['T_err'] must have the same shape as 'T'.")
            sigma_T = np.sqrt(T_err) if err_kind == "var" else T_err
            df["Tx_re_err"] = sigma_T[:, 0, 0]
            df["Tx_im_err"] = sigma_T[:, 0, 0]
            df["Ty_re_err"] = sigma_T[:, 0, 1]
            df["Ty_im_err"] = sigma_T[:, 0, 1]

    if include_pt and edi.get("P") is not None:
        P = np.asarray(edi["P"], dtype=float)
        if P.shape != (n, 2, 2):
            raise ValueError("edi['P'] must have shape (n, 2, 2).")
        df["ptxx"] = P[:, 0, 0]
        df["ptxy"] = P[:, 0, 1]
        df["ptyx"] = P[:, 1, 0]
        df["ptyy"] = P[:, 1, 1]

        P_err = edi.get("P_err")
        if P_err is not None:
            P_err = np.asarray(P_err, dtype=float)
            if P_err.shape != P.shape:
                raise ValueError("edi['P_err'] must have the same shape as 'P'.")
            sigma_P = np.sqrt(P_err) if err_kind == "var" else P_err
            df["ptxx_err"] = sigma_P[:, 0, 0]
            df["ptxy_err"] = sigma_P[:, 0, 1]
            df["ptyx_err"] = sigma_P[:, 1, 0]
            df["ptyy_err"] = sigma_P[:, 1, 1]

    for key in ("station", "lat_deg", "lon_deg", "elev_m", "err_kind"):
        if key in edi:
            df.attrs[key] = edi[key]

    return df


def make_spline(x: np.ndarray, y: np.ndarray, lam: float | None = None):
    """Fit a smoothing spline (SciPy) and return the spline object."""
    sort_idx = np.argsort(x)
    x_sorted, y_sorted = x[sort_idx], y[sort_idx]
    return make_smoothing_spline(x_sorted, y_sorted, lam=lam)


def estimate_variance(y_true: np.ndarray, y_fit: np.ndarray) -> float:
    """Estimate residual variance (unbiased, ddof=1)."""
    residuals = y_true - y_fit
    return float(np.var(residuals, ddof=1))


def bootstrap_confidence_band(
    x: np.ndarray,
    y: np.ndarray,
    lam: float | None = None,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap confidence band for smoothing-spline predictions."""
    sort_idx = np.argsort(x)
    x_eval, y_sorted = x[sort_idx], y[sort_idx]
    n = len(x_eval)
    preds = np.full((n_bootstrap, n), np.nan, dtype=float)

    rng = np.random.default_rng()

    for i in range(n_bootstrap):
        idx = rng.choice(len(x), len(x), replace=True)
        x_res, y_res = x[idx], y[idx]

        order = np.argsort(x_res)
        x_rs, y_rs = x_res[order], y_res[order]

        unique_x, unique_idx = np.unique(x_rs, return_index=True)
        y_unique = y_rs[unique_idx]
        if unique_x.size < 4:
            continue

        spline = make_smoothing_spline(unique_x, y_unique, lam=lam)
        preds[i, :] = spline(x_eval)

    alpha = 1.0 - ci
    lower = np.nanpercentile(preds, 100.0 * alpha / 2.0, axis=0)
    upper = np.nanpercentile(preds, 100.0 * (1.0 - alpha / 2.0), axis=0)
    return x_eval, lower, upper


def choose_lambda_gcv(x: np.ndarray, y: np.ndarray, lam_grid: Optional[np.ndarray] = None):
    """Choose a smoothing parameter by a simple GCV-like score."""
    if lam_grid is None:
        lam_grid = np.logspace(-3, 3, 50)

    best_score = np.inf
    best_lam = None
    best_spline = None

    for lam in lam_grid:
        spline = make_smoothing_spline(x, y, lam=float(lam))
        residuals = y - spline(x)

        dof = getattr(spline, "c", np.asarray([0])).size
        denom = max(1e-12, 1.0 - (dof / max(1, len(x))))
        score = float(np.mean(residuals**2) / (denom**2))

        if score < best_score:
            best_score, best_lam, best_spline = score, float(lam), spline

    return best_spline, best_lam


def save_npz(
    data_dict: Any = None,
    *,
    path: str | Path,
    **kwargs: Any,
) -> None:
    """
    Save a site dictionary to a compressed ``.npz`` archive.

    Each dict entry is stored as a separate array so that individual
    fields can be accessed directly::

        np.load("SITE.npz", allow_pickle=True)["Z"]   # → (n,2,2) complex

    Can be called as either::

        save_npz(data_dict, path="out.npz")
        save_npz(**data_dict, path="out.npz")

    Non-array values (strings, None, scalars) are stored as 0-d object
    arrays and require ``allow_pickle=True`` on load.
    """
    if data_dict is None:
        data_dict = kwargs
    path = Path(path)

    # Convert every value to something np.savez can handle
    save_dict = {}
    for k, v in data_dict.items():
        if v is None:
            save_dict[k] = np.array(None, dtype=object)
        elif isinstance(v, np.ndarray):
            save_dict[k] = v
        elif isinstance(v, (str, bytes)):
            save_dict[k] = np.array(v, dtype=object)
        elif isinstance(v, (int, float, complex, bool, np.number)):
            save_dict[k] = np.array(v)
        else:
            # lists, nested structures → object array (pickle)
            save_dict[k] = np.array(v, dtype=object)

    np.savez_compressed(path.as_posix(), **save_dict)


def load_npz(
    path: str | Path,
) -> dict:
    """
    Load a site dictionary from a ``.npz`` archive written by :func:`save_npz`.

    Returns a plain dict whose values are numpy arrays (or scalars/strings
    extracted from 0-d object arrays).

    Example::

        site = load_npz("SITE.npz")
        Z = site["Z"]          # (n, 2, 2) complex
        station = site["station"]  # str
    """
    path = Path(path)
    out = {}
    with np.load(path.as_posix(), allow_pickle=True) as z:
        for k in z.files:
            arr = z[k]
            # Unwrap 0-d object arrays back to Python scalars/strings
            if arr.ndim == 0:
                val = arr.item()
                out[k] = val
            else:
                out[k] = arr
    return out


def save_list_of_dicts_npz(
    records: list[dict[str, Any]],
    path: str | Path,
    *,
    key: str = "records",
) -> None:
    """
    Save a list of dictionaries to a compressed ``.npz`` file (Option A).

    Stores the list as a NumPy object array (pickle-based).

    Notes
    -----
    Loading requires ``np.load(..., allow_pickle=True)``.
    """
    path = Path(path)
    arr = np.array(records, dtype=object)
    np.savez_compressed(path.as_posix(), **{key: arr})


def load_list_of_dicts_npz(
    path: str | Path,
    *,
    key: str = "records",
) -> list[dict[str, Any]]:
    """
    Load a list of dictionaries written by :func:`save_list_of_dicts_npz`.
    """
    path = Path(path)
    with np.load(path.as_posix(), allow_pickle=True) as z:
        arr = z[key]
    return list(arr.tolist())
def _is_scalar(x: Any) -> bool:
    return isinstance(x, (str, bytes, int, float, bool, np.number))


def _as_1d_array(x: Any) -> Optional[np.ndarray]:
    """Try to coerce x to a 1D numpy array (or return None)."""
    if isinstance(x, np.ndarray):
        if x.ndim == 1:
            return x
        # accept column/row vectors (N,1) or (1,N)
        if x.ndim == 2 and 1 in x.shape:
            return x.ravel()
        return None
    if isinstance(x, (list, tuple)):
        arr = np.asarray(x)
        if arr.ndim == 1:
            return arr
        if arr.ndim == 2 and 1 in arr.shape:
            return arr.ravel()
        return None
    return None


def _unwrap_edidict(data_dict: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    Unwrap a site-style container to a flat EDI-style data dictionary.

    In parts of Py4MTX, "sites" are represented as wrapper dictionaries
    (e.g. ``{"data_dict": <edidict>, ...}``). I/O helpers such as
    :func:`_edidict_to_dataframe` expect a *flat* dictionary where keys like
    ``"freq"`` or ``"period"`` live at the top level. This helper makes the
    conversion robust by returning the inner mapping when it looks like an
    EDI-style data dictionary.

    Parameters
    ----------
    data_dict : mapping
        Either a flat EDI-style dictionary or a wrapper mapping with a
        ``"data_dict"`` entry.

    Returns
    -------
    mapping
        The unwrapped EDI-style dictionary if present; otherwise the input.
    """
    try:
        inner = data_dict.get("data_dict")  # type: ignore[call-arg]
    except Exception:
        inner = None

    if isinstance(inner, Mapping):
        # unwrap if the inner dict looks like an edidict
        if ("freq" in inner) or ("period" in inner) or ("Z" in inner) or ("Z_err" in inner):
            return inner

    return data_dict



def _flatten_meta_for_attrs(meta: Mapping[str, Any], *, sep: str = ".") -> dict[str, Any]:
    """
    Flatten nested dict-ish metadata to simple key/value attrs.

    NetCDF/HDF attrs work best with scalars and short strings.
    For non-simple types, we store a string repr.
    """
    out: dict[str, Any] = {}

    def rec(prefix: str, obj: Any) -> None:
        if isinstance(obj, Mapping):
            for k, v in obj.items():
                rec(f"{prefix}{sep}{k}" if prefix else str(k), v)
            return
        if _is_scalar(obj):
            out[prefix] = obj
            return
        # numpy scalar
        if isinstance(obj, np.generic):
            out[prefix] = obj.item()
            return
        # small 1D arrays -> store as repr (attrs can't safely hold big arrays)
        if isinstance(obj, np.ndarray):
            if obj.ndim == 0:
                out[prefix] = obj.item()
            else:
                out[prefix] = repr(obj)
            return
        out[prefix] = repr(obj)

    rec("", meta)
    return out


def _sanitize_meta_for_hdf(meta: Mapping[str, Any]) -> dict[str, Any]:
    """
    Sanitize metadata for storage in an HDF5 table without PyTables pickling.

    Pandas/PyTables will pickle columns of dtype ``object`` that contain
    non-scalar Python objects (e.g. lists, dicts). This is slow and can be
    fragile across versions. This helper converts non-scalar values to strings.

    Rules
    -----
    - Scalars (str/bytes/int/float/bool/numpy scalar) are kept.
    - ``None`` is kept as ``None`` (later castable to pandas StringDtype if desired).
    - lists/tuples/dicts are JSON-serialized (fallback: ``repr``).
    - numpy arrays and other objects become ``repr`` strings.

    Parameters
    ----------
    meta : mapping
        Metadata dict.

    Returns
    -------
    dict
        Sanitized metadata dict safe to write as a single-row HDF table.
    """

    out: dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            out[k] = None
            continue

        if _is_scalar(v):
            out[k] = v
            continue

        if isinstance(v, np.generic):
            out[k] = v.item()
            continue

        if isinstance(v, (list, tuple, dict)):
            try:
                s = json.dumps(v, ensure_ascii=False, default=str)
                if len(s) > 200000:
                    s = s[:200000] + " ... (truncated)"
                out[k] = s
            except Exception:
                out[k] = repr(v)
            continue

        if isinstance(v, np.ndarray):
            # arrays in attrs are usually not intended; store a compact repr
            out[k] = repr(v)
            continue

        out[k] = repr(v)

    return out



def _sanitize_meta_for_mat(meta: Mapping[str, Any]) -> dict[str, Any]:
    """
    Sanitize metadata for storage in a MATLAB .mat file.

    SciPy's ``savemat`` is fairly permissive, but some Python objects (e.g. dicts,
    sets, custom classes, or ``None``) can lead to confusing MATLAB objects.
    This helper aims to keep MATLAB consumption straightforward:

    - Scalars are kept.
    - ``None`` becomes an empty array ``[]``.
    - numpy arrays are kept (including complex arrays).
    - lists/tuples become numpy arrays where possible; mixed lists become object arrays.
    - dicts become JSON strings (fallback: ``repr``).
    - everything else becomes ``repr``.
    """

    out: dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            out[k] = np.asarray([])
            continue

        if _is_scalar(v):
            out[k] = v
            continue

        if isinstance(v, np.generic):
            out[k] = v.item()
            continue

        if isinstance(v, np.ndarray):
            out[k] = v
            continue

        if isinstance(v, (list, tuple)):
            try:
                arr = np.asarray(v)
                if arr.dtype == object:
                    out[k] = np.asarray(list(v), dtype=object)
                else:
                    out[k] = arr
            except Exception:
                out[k] = np.asarray([repr(v)], dtype=object)
            continue

        if isinstance(v, dict):
            try:
                out[k] = json.dumps(v, ensure_ascii=False, default=str)
            except Exception:
                out[k] = repr(v)
            continue

        out[k] = repr(v)

    return out



def _sanitize_mapping_for_mat_struct(d: Mapping[str, Any]) -> dict[str, Any]:
    """
    Sanitize an arbitrary mapping for MATLAB struct storage.

    Similar to :func:`_sanitize_meta_for_mat`, but intended for storing a full
    EDI-style data dict (including arrays) into a MATLAB struct. Keys are kept
    as-is; values are coerced to MATLAB-friendly representations.

    - ndarrays (including complex) are kept
    - scalars/strings are kept
    - None -> empty array
    - lists/tuples -> arrays
    - dicts -> JSON strings (fallback: repr)
    - everything else -> repr
    """

    out: dict[str, Any] = {}
    for k, v in d.items():
        if v is None:
            out[str(k)] = np.asarray([])
            continue

        if isinstance(v, str):
            out[str(k)] = v
            continue

        if _is_scalar(v):
            out[str(k)] = v
            continue

        if isinstance(v, np.generic):
            out[str(k)] = v.item()
            continue

        if isinstance(v, np.ndarray):
            out[str(k)] = v
            continue

        if isinstance(v, (list, tuple)):
            try:
                arr = np.asarray(v)
                if arr.dtype == object:
                    out[str(k)] = np.asarray(list(v), dtype=object)
                else:
                    out[str(k)] = arr
            except Exception:
                out[str(k)] = np.asarray([repr(v)], dtype=object)
            continue

        if isinstance(v, dict):
            try:
                out[str(k)] = json.dumps(v, ensure_ascii=False, default=str)
            except Exception:
                out[str(k)] = repr(v)
            continue

        out[str(k)] = repr(v)

    return out


def _edidict_to_dataframe(
    data_dict: Mapping[str, Any],
    *,
    dim_preference: Tuple[str, ...] = ("period", "freq"),
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    """
    Convert an data_dict to a DataFrame containing only tabular vars, plus meta.

    Rules:
    - Choose a main dimension (period or freq) if present as 1D array.
    - Include 1D arrays of length N as columns.
    - Expand arrays with shape (N, ... ) into multiple columns.
    - Complex arrays become *_re / *_im columns.
    - Everything else goes into meta.
    """
    data_dict = _unwrap_edidict(data_dict)
    # 1) choose dimension / coordinate
    dim_name = None
    coord = None
    for d in dim_preference:
        arr = _as_1d_array(data_dict.get(d))
        if arr is not None and arr.size > 0:
            dim_name = d
            coord = np.asarray(arr)
            break
    if dim_name is None:
        raise ValueError(f"data_dict must contain a 1D '{dim_preference[0]}' or '{dim_preference[1]}' array.")

    n = int(coord.size)

    cols: dict[str, np.ndarray] = {dim_name: coord}
    meta: dict[str, Any] = {}

    def add_column(name: str, arr: np.ndarray) -> None:
        cols[name] = np.asarray(arr)

    def expand(name: str, arr: np.ndarray) -> None:
        # arr has shape (n, ...)
        if np.iscomplexobj(arr):
            expand(name + "_re", np.asarray(arr.real))
            expand(name + "_im", np.asarray(arr.imag))
            return

        if arr.ndim == 1:
            add_column(name, arr)
            return

        # flatten remaining dimensions into separate columns
        tail = int(np.prod(arr.shape[1:]))
        flat = arr.reshape(n, tail)
        for j in range(tail):
            add_column(f"{name}_{j}", flat[:, j])

    # 2) populate table/meta
    for k, v in data_dict.items():
        if k == dim_name:
            continue

        if _is_scalar(v) or v is None:
            meta[k] = v
            continue

        if isinstance(v, np.ndarray):
            arr = v
        elif isinstance(v, (list, tuple)):
            arr = np.asarray(v)
        else:
            # nested dicts or custom objects -> meta
            meta[k] = v
            continue

        if arr.ndim == 1 and arr.size == n:
            expand(k, arr)
        elif arr.ndim >= 2 and arr.shape[0] == n:
            expand(k, arr)
        else:
            meta[k] = v

    df = pd.DataFrame(cols)
    df.attrs.update(meta)
    return df, meta, dim_name


def save_hdf(
    data_dict: Optional[Mapping[str, Any] | pd.DataFrame] = None,
    *,
    path: str | Path,
    key: str = "mt",
    mode: str = "w",
    complevel: int = 4,
    **kwargs: Any,
) -> None:
    """
    Save a site dictionary to HDF5 with each entry as a dataset.

    Can be called as either::

        save_hdf(data_dict, path="out.hdf")
        save_hdf(**data_dict, path="out.hdf")

    Arrays (including complex) are stored as HDF5 datasets::

        h5py.File("SITE.hdf")["Z"]   # → (n,2,2) complex128

    Scalars and strings are stored as HDF5 attributes on the group.

    If *data_dict* is a ``pandas.DataFrame``, it is converted to a dict
    of arrays internally (one dataset per column).

    Requires the ``h5py`` package.
    """
    if data_dict is None:
        data_dict = kwargs
    path = Path(path)

    try:
        import h5py
    except ImportError as exc:
        raise ImportError("save_hdf requires the 'h5py' package.") from exc

    # Convert DataFrame to dict
    if isinstance(data_dict, pd.DataFrame):
        dd: dict[str, Any] = {}
        for c in data_dict.columns:
            dd[c] = data_dict[c].to_numpy()
        dd.update(data_dict.attrs)
        data_dict = dd

    with h5py.File(path.as_posix(), mode) as f:
        grp = f.require_group(key)

        for k, v in data_dict.items():
            if v is None:
                grp.attrs[k] = "__NONE__"
            elif isinstance(v, np.ndarray):
                if v.dtype.kind == "U":
                    # HDF5 can't store numpy unicode directly; convert
                    grp.attrs[k] = str(v.item()) if v.ndim == 0 else str(v)
                else:
                    opts = {"compression": "gzip", "compression_opts": complevel} if v.size > 1 else {}
                    if k in grp:
                        del grp[k]
                    grp.create_dataset(k, data=v, **opts)
            elif isinstance(v, (str, bytes)):
                grp.attrs[k] = v
            elif isinstance(v, (int, float, complex, bool, np.number)):
                grp.attrs[k] = v
            elif isinstance(v, (list, tuple)):
                arr = np.asarray(v)
                if arr.dtype.kind in ("U", "O"):
                    grp.attrs[k] = str(v)
                else:
                    if k in grp:
                        del grp[k]
                    grp.create_dataset(k, data=arr)
            else:
                grp.attrs[k] = str(v)


def save_ncd(
    data_dict: Optional[Mapping[str, Any] | pd.DataFrame] = None,
    *,
    path: str | Path,
    engine: Optional[str] = None,
    dataset_name: str = "mt",
    **kwargs: Any,
) -> None:
    """
    Save a site dictionary to NetCDF with each entry as a variable.

    Can be called as either::

        save_ncd(data_dict, path="out.ncd")
        save_ncd(**data_dict, path="out.ncd")

    Arrays are stored as NetCDF variables with auto-generated dimension
    names.  Complex arrays are split into ``<key>_re`` and ``<key>_im``.
    Scalars and strings are stored as global attributes.

    Example::

        import xarray as xr
        ds = xr.open_dataset("SITE.ncd")
        Z_re = ds["Z_re"].values   # (n, 2, 2)
        Z_im = ds["Z_im"].values   # (n, 2, 2)
        freq = ds["freq"].values   # (n,)

    Requires the ``xarray`` package.
    """
    if data_dict is None:
        data_dict = kwargs
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError("save_ncd requires the 'xarray' package.") from exc

    path = Path(path)

    # Convert DataFrame to dict
    if isinstance(data_dict, pd.DataFrame):
        dd: dict[str, Any] = {}
        for c in data_dict.columns:
            dd[c] = data_dict[c].to_numpy()
        dd.update(data_dict.attrs)
        data_dict = dd

    data_vars: dict[str, Any] = {}
    attrs: dict[str, Any] = {"dataset_name": dataset_name}

    for k, v in data_dict.items():
        if v is None:
            attrs[k] = "__NONE__"
            continue

        if isinstance(v, (str, bytes)):
            attrs[k] = v if isinstance(v, str) else v.decode()
            continue

        if isinstance(v, (int, float, bool, np.number)):
            # store real scalars as attrs; complex scalar as 0-d var
            if isinstance(v, (complex, np.complexfloating)):
                data_vars[f"{k}_re"] = ((), np.real(v))
                data_vars[f"{k}_im"] = ((), np.imag(v))
            else:
                attrs[k] = float(v) if isinstance(v, (float, np.floating)) else v
            continue

        arr = np.asarray(v)
        if arr.dtype.kind == "U" or arr.dtype.kind == "O":
            attrs[k] = str(v)
            continue

        # Auto-generate dimension names based on shape
        dim_names = tuple(f"{k}_d{i}" for i in range(arr.ndim))

        # Use "freq" or "period" as a shared first dimension when possible
        if arr.ndim >= 1:
            for shared_dim in ("freq", "period"):
                if shared_dim in data_dict:
                    ref = np.asarray(data_dict[shared_dim])
                    if ref.ndim == 1 and arr.shape[0] == ref.size:
                        dim_names = (shared_dim,) + dim_names[1:]
                        break

        if np.iscomplexobj(arr):
            data_vars[f"{k}_re"] = (dim_names, arr.real)
            data_vars[f"{k}_im"] = (dim_names, arr.imag)
        else:
            data_vars[k] = (dim_names, arr)

    ds = xr.Dataset(data_vars=data_vars, attrs=attrs)
    ds.to_netcdf(path.as_posix(), engine=engine)


def save_mat(
    data_dict: Optional[Mapping[str, Any] | pd.DataFrame] = None,
    *,
    path: str | Path,
    do_compression: bool = True,
    **kwargs: Any,
) -> None:
    """
    Save a site dictionary to a MATLAB ``.mat`` file with each entry as a variable.

    Can be called as either::

        save_mat(data_dict, path="out.mat")
        save_mat(**data_dict, path="out.mat")

    Arrays (including complex) are stored as MATLAB variables::

        S = load('SITE.mat');
        Z    = S.Z;        % (n,2,2) complex
        freq = S.freq;     % (n,1) double

    Scalars and strings become scalar MATLAB variables.
    Keys are sanitized for MATLAB compatibility (no dots, leading digits, etc.).

    Requires ``scipy.io.savemat``.
    """
    if data_dict is None:
        data_dict = kwargs
    from scipy.io import savemat

    path = Path(path)

    # Convert DataFrame to dict
    if isinstance(data_dict, pd.DataFrame):
        dd: dict[str, Any] = {}
        for c in data_dict.columns:
            dd[c] = data_dict[c].to_numpy()
        dd.update(data_dict.attrs)
        data_dict = dd

    out: dict[str, Any] = {}
    for k, v in data_dict.items():
        # Sanitize key for MATLAB variable names
        k_safe = re.sub(r"\W+", "_", str(k).strip())
        if not k_safe:
            k_safe = "x_unnamed"
        if k_safe[0].isdigit():
            k_safe = "x" + k_safe

        if v is None:
            out[k_safe] = np.array([])  # empty → MATLAB []
        elif isinstance(v, np.ndarray):
            out[k_safe] = v
        elif isinstance(v, (str, bytes)):
            out[k_safe] = v
        elif isinstance(v, (int, float, complex, bool, np.number)):
            out[k_safe] = v
        elif isinstance(v, (list, tuple)):
            arr = np.asarray(v)
            if arr.dtype.kind in ("U", "O"):
                out[k_safe] = np.asarray(v, dtype=object)
            else:
                out[k_safe] = arr
        else:
            out[k_safe] = str(v)

    savemat(path.as_posix(), out, do_compression=bool(do_compression))


def read_emtf_xml(path: str | Path) -> Dict[str, Any]:
    """Read MT data from a (simplified) EMTF-XML file (experimental)."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(str(path))
    root = tree.getroot()

    ns_uri = ""
    if root.tag.startswith("{") and "}" in root.tag:
        ns_uri = root.tag.split("}", 1)[0].strip("{")
    ns = {"emtf": ns_uri} if ns_uri else {}

    station = root.find("emtf:Station", ns) if ns else root.find("Station")
    metadata = {}
    if station is not None:
        metadata = {
            "id": station.get("id"),
            "name": station.findtext("emtf:Name" if ns else "Name", default="", namespaces=ns),
            "latitude": float(station.findtext("emtf:Latitude" if ns else "Latitude", default="nan", namespaces=ns)),
            "longitude": float(station.findtext("emtf:Longitude" if ns else "Longitude", default="nan", namespaces=ns)),
        }

    tf_elements = root.findall("emtf:TransferFunction/emtf:Component", ns) if ns else root.findall("TransferFunction/Component")
    transfer_functions: Dict[str, np.ndarray] = {}
    for comp in tf_elements:
        cname = comp.get("name")
        values = [float(v.text) for v in comp.findall("emtf:Value" if ns else "Value", ns)]
        if cname:
            transfer_functions[cname] = np.asarray(values, dtype=float)

    return {"metadata": metadata, "transfer_functions": transfer_functions}


def write_emtf_xml(data: Dict[str, Any], path: str | Path) -> None:
    """Write MT data to a (simplified) EMTF-XML file (experimental)."""
    import xml.etree.ElementTree as ET

    ns = "http://emtf.org/schema"
    ET.register_namespace("emtf", ns)

    root = ET.Element(f"{{{ns}}}EMTF")

    metadata = data.get("metadata", {})
    tf_data = data.get("transfer_functions", {})

    station = ET.SubElement(root, f"{{{ns}}}Station", id=str(metadata.get("id", "")))
    ET.SubElement(station, f"{{{ns}}}Name").text = str(metadata.get("name", ""))
    ET.SubElement(station, f"{{{ns}}}Latitude").text = str(metadata.get("latitude", ""))
    ET.SubElement(station, f"{{{ns}}}Longitude").text = str(metadata.get("longitude", ""))

    tf = ET.SubElement(root, f"{{{ns}}}TransferFunction")
    for cname, values in tf_data.items():
        comp = ET.SubElement(tf, f"{{{ns}}}Component", name=str(cname))
        for v in np.asarray(values).ravel():
            ET.SubElement(comp, f"{{{ns}}}Value").text = str(float(v))

    tree = ET.ElementTree(root)
    tree.write(str(path), encoding="utf-8", xml_declaration=True)


def edi_to_emtf(edi_path: str | Path, emtf_path: str | Path) -> str:
    """Convert an EDI file to the simplified EMTF-XML representation."""
    edi = load_edi(edi_path)
    meta = {
        "id": edi.get("station", ""),
        "name": edi.get("station", ""),
        "latitude": edi.get("lat_deg", edi.get("lat")),
        "longitude": edi.get("lon_deg", edi.get("lon")),
    }
    tf = {
        "freq": np.asarray(edi["freq"], dtype=float),
        "Zxx_re": np.asarray(edi["Z"][:, 0, 0].real),
        "Zxx_im": np.asarray(edi["Z"][:, 0, 0].imag),
        "Zxy_re": np.asarray(edi["Z"][:, 0, 1].real),
        "Zxy_im": np.asarray(edi["Z"][:, 0, 1].imag),
        "Zyx_re": np.asarray(edi["Z"][:, 1, 0].real),
        "Zyx_im": np.asarray(edi["Z"][:, 1, 0].imag),
        "Zyy_re": np.asarray(edi["Z"][:, 1, 1].real),
        "Zyy_im": np.asarray(edi["Z"][:, 1, 1].imag),
    }
    if edi.get("T") is not None:
        T = np.asarray(edi["T"])
        tf["Tx_re"] = T[:, 0, 0].real
        tf["Tx_im"] = T[:, 0, 0].imag
        tf["Ty_re"] = T[:, 0, 1].real
        tf["Ty_im"] = T[:, 0, 1].imag

    write_emtf_xml({"metadata": meta, "transfer_functions": tf}, emtf_path)
    return str(emtf_path)


def emtf_to_edi(emtf_path: str | Path, edi_path: str | Path) -> str:
    """Convert simplified EMTF-XML back to a classical EDI file."""
    data = read_emtf_xml(emtf_path)
    tf = data.get("transfer_functions", {})
    required = ["freq", "Zxx_re", "Zxx_im", "Zxy_re", "Zxy_im", "Zyx_re", "Zyx_im", "Zyy_re", "Zyy_im"]
    if not all(k in tf for k in required):
        raise NotImplementedError(
            "emtf_to_edi expects the simplified component names created by edi_to_emtf()."
        )

    freq = np.asarray(tf["freq"], dtype=float)
    n = freq.size
    Z = np.zeros((n, 2, 2), dtype=np.complex128)
    Z[:, 0, 0] = np.asarray(tf["Zxx_re"]) + 1j * np.asarray(tf["Zxx_im"])
    Z[:, 0, 1] = np.asarray(tf["Zxy_re"]) + 1j * np.asarray(tf["Zxy_im"])
    Z[:, 1, 0] = np.asarray(tf["Zyx_re"]) + 1j * np.asarray(tf["Zyx_im"])
    Z[:, 1, 1] = np.asarray(tf["Zyy_re"]) + 1j * np.asarray(tf["Zyy_im"])

    edi = {
        "freq": freq,
        "Z": Z,
        "T": None,
        "Z_err": None,
        "T_err": None,
        "P": None,
        "P_err": None,
        "rot": None,
        "err_kind": "var",
        "station": data.get("metadata", {}).get("name", None),
        "lat_deg": data.get("metadata", {}).get("latitude", None),
        "lon_deg": data.get("metadata", {}).get("longitude", None),
        "elev_m": None,
    }

    if "Tx_re" in tf and "Tx_im" in tf and "Ty_re" in tf and "Ty_im" in tf:
        T = np.zeros((n, 1, 2), dtype=np.complex128)
        T[:, 0, 0] = np.asarray(tf["Tx_re"]) + 1j * np.asarray(tf["Tx_im"])
        T[:, 0, 1] = np.asarray(tf["Ty_re"]) + 1j * np.asarray(tf["Ty_im"])
        edi["T"] = T

    save_edi(edi_path, edi, add_pt_blocks=False)
    return str(edi_path)


def compute_rhophas(
    freq: np.ndarray,
    Z: np.ndarray,
    Z_err: Optional[np.ndarray] = None,
    *,
    err_kind: str = "var",
    err_method: str = "bootstrap",
    nsim: int = 200,
    mu0: float = _MU0,
    random_state: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute apparent resistivity and phase from impedance, optionally with errors.

    Parameters
    ----------
    freq : numpy.ndarray
        Frequencies in Hz, shape ``(n,)``.
    Z : numpy.ndarray
        Complex impedance tensor of shape ``(n, 2, 2)``.
    Z_err : numpy.ndarray or None, optional
        Impedance uncertainties with the same shape as ``Z`` (variance or
        standard deviation, see ``err_kind``).
    err_kind : {"var", "std"}, optional
        Interpretation of ``Z_err`` (variance or standard deviation of the
        complex impedance entries). Default is ``"var"``.
    err_method : {"none", "analytic", "bootstrap", "both"}, optional
        Error propagation method:

        - ``"none"``: return ``rho_err=None`` and ``phi_err=None``.
        - ``"analytic"``: first-order propagation (delta method) assuming
          independent real/imag parts with equal variance split.
        - ``"bootstrap"``: parametric bootstrap (Monte-Carlo) using ``nsim``.
        - ``"both"``: return dicts for both error estimates.

        Default is ``"analytic"`` because it is fast and often sufficient.
    nsim : int, optional
        Number of bootstrap realisations for ``err_method="bootstrap"``.
        Default is 200.
    mu0 : float, optional
        Magnetic permeability used in the resistivity conversion. Default is
        vacuum permeability ``_MU0``.
    random_state : numpy.random.Generator, optional
        Random generator to use for bootstrap.

    Returns
    -------
    rho : numpy.ndarray
        Apparent resistivity ``rho_a`` in Ω·m, shape ``(n, 2, 2)``.
    phi : numpy.ndarray
        Phase in degrees, shape ``(n, 2, 2)``.
    rho_err : numpy.ndarray or None
        Error estimate for ``rho`` with the same shape, returned as variance
        if ``err_kind="var"`` or as standard deviation if ``err_kind="std"``.
        If ``err_method="both"``, a dict is returned with keys
        ``"analytic"`` and ``"bootstrap"``.
    phi_err : numpy.ndarray or None
        Error estimate for ``phi`` with the same shape (variance/std as above),
        or dict if ``err_method="both"``.

    Notes
    -----
    This function uses the standard MT definition:

        ``rho_a = |Z|^2 / (mu0 * omega)``, with ``omega = 2π f``.

    Analytic error propagation uses commonly applied approximations:

    - ``std(rho_a) ≈ 2 |Z| std(Z) / (mu0 * omega)``
    - ``std(phi_rad) ≈ std(Z) / |Z|``, converted to degrees.

    Bootstrap treats ``Z_err`` as defining independent Gaussian perturbations
    on the complex entries.
    """
    freq = np.asarray(freq, dtype=float).ravel()
    Z = np.asarray(Z, dtype=np.complex128)
    if Z.shape != (freq.size, 2, 2):
        Z.reshape((freq.size,2, 2))
        # raise ValueError("Z must have shape (n, 2, 2) matching freq.")
    omega = 2.0 * np.pi * freq
    denom = mu0 * omega[:, None, None]

    rho = (np.abs(Z) ** 2) / denom
    phi = np.degrees(np.angle(Z))

    method = err_method.lower()
    if Z_err is None or method in ("none", "off", "false", "0"):
        return rho, phi, None, None

    Z_err = np.asarray(Z_err, dtype=float)
    if Z_err.shape != Z.shape:
        raise ValueError("Z_err must have the same shape as Z.")
    if method not in ("analytic", "bootstrap", "both"):
        raise ValueError("err_method must be one of: 'none', 'analytic', 'bootstrap', 'both'.")

    sigma = _sigma_from_err(Z_err, err_kind=err_kind)
    absZ = np.abs(Z)

    # ------------- analytic approximations
    rho_err_analytic = None
    phi_err_analytic = None
    if method in ("analytic", "both"):
        with np.errstate(divide="ignore", invalid="ignore"):
            std_rho = 2.0 * sigma * absZ / denom
            std_phi_deg = (sigma / absZ) * (180.0 / np.pi)
        if err_kind == "var":
            rho_err_analytic = std_rho ** 2
            phi_err_analytic = std_phi_deg ** 2
        else:
            rho_err_analytic = std_rho
            phi_err_analytic = std_phi_deg

    # ------------- bootstrap
    rho_err_boot = None
    phi_err_boot = None
    if method in ("bootstrap", "both"):
        rng = np.random.default_rng() if random_state is None else random_state
        rho_sims = np.full((nsim,) + rho.shape, np.nan, dtype=float)
        phi_sims = np.full((nsim,) + phi.shape, np.nan, dtype=float)

        for sidx in range(nsim):
            d_re = rng.standard_normal(Z.shape) * sigma / np.sqrt(2.0)
            d_im = rng.standard_normal(Z.shape) * sigma / np.sqrt(2.0)
            Zs = (Z.real + d_re) + 1j * (Z.imag + d_im)
            rho_sims[sidx] = (np.abs(Zs) ** 2) / denom
            phi_sims[sidx] = np.degrees(np.angle(Zs))

        with np.errstate(invalid="ignore"):
            var_rho = np.nanvar(rho_sims, axis=0)
            var_phi = np.nanvar(phi_sims, axis=0)
        if err_kind == "var":
            rho_err_boot = var_rho
            phi_err_boot = var_phi
        else:
            rho_err_boot = np.sqrt(var_rho)
            phi_err_boot = np.sqrt(var_phi)

    if method == "analytic":
        return rho, phi, rho_err_analytic, phi_err_analytic
    if method == "bootstrap":
        return rho, phi, rho_err_boot, phi_err_boot
    return rho, phi, {"analytic": rho_err_analytic, "bootstrap": rho_err_boot}, {"analytic": phi_err_analytic, "bootstrap": phi_err_boot}

def calc_rhoa_phas(freq: np.ndarray, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute apparent resistivity and phase from impedance (no errors).

    This is a backward-compatible wrapper around :func:`compute_rhophas`.
    """
    rho, phi, _, _ = compute_rhophas(freq, Z, Z_err=None, err_method="none")
    return rho, phi


def mt1dfwd(
    freq: np.ndarray,
    sig: np.ndarray,
    d: np.ndarray,
    inmod: str = "r",
    out: str = "imp",
    magfield: str = "b",
):
    """Compute 1D MT forward response for a layered Earth."""
    mu0 = _MU0
    sig = np.array(sig, dtype=float)
    freq = np.array(freq, dtype=float)
    d = np.array(d, dtype=float)

    if inmod.lower().startswith("r"):
        sig = 1.0 / sig

    if sig.ndim > 1:
        raise ValueError("sig must be 1D.")

    nlay = sig.size
    Z = np.zeros_like(freq, dtype=complex)
    w = 2.0 * np.pi * freq

    for ifr, omega in enumerate(w):
        imp = np.empty(nlay, dtype=complex)
        imp[-1] = np.sqrt(1j * omega * mu0 / sig[-1])

        for layer in range(nlay - 2, -1, -1):
            sl = sig[layer]
            dl = d[layer]
            dj = np.sqrt(1j * omega * mu0 * sl)
            wj = dj / sl
            ej = np.exp(-2.0 * dl * dj)
            impb = imp[layer + 1]
            rj = (wj - impb) / (wj + impb)
            reff = rj * ej
            imp[layer] = wj * ((1.0 - reff) / (1.0 + reff))

        Z[ifr] = imp[0]

    if out.lower() == "imp":
        return Z / mu0 if magfield.lower() == "b" else Z

    absZ = np.abs(Z)
    rhoa = (absZ**2) / (mu0 * w)
    phase = np.rad2deg(np.angle(Z))

    if out.lower() == "rho":
        return rhoa, phase
    return Z, rhoa, phase


def wait1d(periods: np.ndarray, thick: np.ndarray, res: np.ndarray):
    """Alternative 1D MT forward modelling implementation (legacy)."""
    mu = _MU0
    omega = 2.0 * np.pi / periods

    cond = 1.0 / np.asarray(res, dtype=float)

    spn = np.size(periods)
    Z = np.zeros(spn, dtype=complex)

    for idx, w in enumerate(omega):
        prop_const = np.sqrt(1j * mu * cond[-1] * w)
        C = np.zeros(spn, dtype=complex)
        C[-1] = 1.0 / prop_const
        if len(thick) > 0:
            for k in reversed(range(len(res) - 1)):
                prop_layer = np.sqrt(1j * w * mu * cond[k])
                k1 = (C[k + 1] * prop_layer + np.tanh(prop_layer * thick[k]))
                k2 = ((C[k + 1] * prop_layer * np.tanh(prop_layer * thick[k])) + 1.0)
                C[k] = (1.0 / prop_layer) * (k1 / k2)
        Z[idx] = 1j * w * mu * C[0]

    rhoa = (np.abs(Z) ** 2) / omega
    phi = np.angle(Z, deg=True)
    return rhoa, phi, np.real(Z), np.imag(Z)
