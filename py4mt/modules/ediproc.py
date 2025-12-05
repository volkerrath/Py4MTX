#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ediproc.py
=================
Compact I/O utilities for magnetotelluric EDI files.

This module provides a light-weight, self-contained interface to read and
write standard MT transfer functions (impedance, tipper, phase tensor) in
EDI format. It supports two main data sources:

1. Phoenix / SPECTRA-based EDIs
   -----------------------------
   These contain ``>SPECTRA`` blocks with a 7×7 real-valued matrix encoding
   the auto- and cross-spectra of E and H channels plus metadata such as::

       >SPECTRA  FREQ=1.040E+04 ROTSPEC=0 BW=2.6000E+03 AVGT=5.1245E+05 // 49

   The workflow is:

   - :func:`parse_spectra_blocks`
       Identify ``>SPECTRA`` blocks and extract frequency ``FREQ``,
       averaging time ``AVGT``, rotation angle ``ROTSPEC`` (if present) and
       the Phoenix 7×7 matrix.
   - :func:`reconstruct_S_phoenix`
       Convert the real-valued Phoenix 7×7 encoding into a complex Hermitian
       spectra matrix ``S``.
   - :func:`ZT_from_S`
       Recover impedance ``Z`` and tipper ``T`` from ``S`` with a constant
       scaling factor that converts Phoenix units (roughly µV/m per nT) to
       SI Ohm (V/m per T).

   When ``>SPECTRA`` blocks are present and ``prefer_spectra=True`` (default),
   the Phoenix path is used and classical table blocks (if any) are ignored.

2. Classical table-based EDIs
   ---------------------------
   These provide tabulated Z/T values via blocks such as::

       >FREQ
       >ZXXR  >ZXXI  >ZXYR  >ZXYI  >ZYXR  >ZYXI  >ZYYR  >ZYYI
       >ZXX.VAR  >ZXY.VAR  >ZYX.VAR  >ZYY.VAR
       >TXR.EXP  >TXI.EXP  >TYR.EXP  >TYI.EXP
       >TXVAR.EXP  >TYVAR.EXP
       >ZROT      // impedance rotation angles in degrees

   The workflow is:

   - :func:`_extract_block_values`
       Parse numeric values for individual blocks (``FREQ``, ``ZXXR``, …).
   - :func:`_build_impedance`
       Assemble the complex 2×2 impedance tensor ``Z`` (and optional variance
       ``Z_err``) from the component blocks.
   - :func:`_build_tipper`
       Assemble the complex tipper ``T`` (and optional variance ``T_err``).
   - Rotation:
       If a ``>ZROT`` block is present, it is parsed into a per-frequency
       rotation angle array in degrees and stored as ``edi["rot"]``.

High-level functions
--------------------

- :func:`load_edi`
    Read an EDI file (Phoenix or classical) and return a dictionary.
- :func:`compute_pt`
    Compute the phase tensor from ``Z`` and optionally propagate impedance
    errors via Monte-Carlo.
- :func:`save_edi`
    Write an EDI dictionary back to a classical table-style EDI file,
    optionally including P blocks, variance blocks and a ``>ZROT`` block.

Returned dictionary layout
--------------------------

Typical output from :func:`load_edi`::

    edi = {
        "freq": (n,),
        "Z": (n, 2, 2) complex,
        "T": (n, 1, 2) complex or None,
        "Z_err": (n, 2, 2) float or None,   # from .VAR blocks (tables only)
        "T_err": (n, 1, 2) float or None,
        "P": (n, 2, 2) float or None,      # filled by compute_pt
        "P_err": (n, 2, 2) float or None,
        "rot": (n,) float or None,      # from ROTSPEC or ZROT if present
        "err_kind": "var" or "std",         # meaning of *_err arrays
        "station": str or None,
        "lat_deg": float or None,
        "lon_deg": float or None,
        "elev_m": float or None,
        "header_raw": list[str],            # raw header lines
        "source_kind": "spectra" or "tables",
    }

Notes
-----

- Frequencies are returned in **ascending** order (lowest to highest), even if
  the file lists them in descending order.
- For Phoenix spectra, no Z/T error estimates are derived at present, so
  ``Z_err`` and ``T_err`` are set to ``None``.
- For table-style EDIs, variance blocks such as ``ZXX.VAR`` and ``TXVAR.EXP``
  are read into ``Z_err`` / ``T_err``. By default these are interpreted as
  **variances** (``err_kind="var"``).
- This module is intentionally minimal and transparent; it is not meant to
  replace full-featured libraries (e.g. mtpy), but provides a stable backbone
  for the EDI project and plotting via :mod:`ediviz` and :mod:`edidat`.

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-11-20
"""

from __future__ import annotations

import sys
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import make_smoothing_spline

_MU0: float = 4.0 * np.pi * 1.0e-7

# ---------------------------------------------------------------------------
# Basic text helpers
# ---------------------------------------------------------------------------


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
        The complete file content as a single string.
    """
    p = Path(path)
    return p.read_text(encoding=encoding, errors="ignore")


def _has_spectra(text: str) -> bool:
    """Check whether the EDI contains Phoenix ``>SPECTRA`` blocks."""
    return ">SPECTRA" in text.upper()


def _split_lines(text: str) -> List[str]:
    """Split EDI text into individual lines."""
    return text.splitlines()


def _extract_header_lines(lines: List[str]) -> List[str]:
    """Extract header lines up to the first data or spectra block.

    Everything from the top of the file up to the first line starting with
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
    """Parse a few simple metadata fields from header lines.

    Parameters
    ----------
    header_lines : list of str
        Header lines from the EDI file.

    Returns
    -------
    dict
        Dictionary with keys ``"station"``, ``"lat_deg"``, ``"lon_deg"``,
        ``"elev_m"`` if they can be guessed, otherwise values are ``None``.
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
                rhs = ln.split("=", 1)[1].strip()
                if ":" in rhs:
                    parts = rhs.strip('"').split(":")
                    deg, minute, sec = (float(p) for p in parts[:3])
                    lat_deg = deg + minute / 60.0 + sec / 3600.0
                else:
                    lat_deg = float(rhs.split()[0])
            except Exception:
                pass

        if ("LON" in u or "LONG" in u) and "REFLON" not in u and lon_deg is None:
            try:
                rhs = ln.split("=", 1)[1].strip()
                if ":" in rhs:
                    parts = rhs.strip('"').split(":")
                    deg, minute, sec = (float(p) for p in parts[:3])
                    lon_deg = deg + minute / 60.0 + sec / 3600.0
                else:
                    lon_deg = float(rhs.split()[0])
            except Exception:
                pass

        if "ELEV" in u and "REFELEV" not in u and elev_m is None:
            try:
                rhs = ln.split("=", 1)[1].strip()
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
    """Extract numerical values for a given EDI data block.

    Parameters
    ----------
    lines : list of str
        All EDI lines.
    keyword : str
        Block keyword without leading ``">"`` (for example ``"FREQ"``,
        ``"ZXXR"``, ``"ZXX.VAR"``, ``"TXR.EXP"``). Matching is
        case-insensitive.

    Returns
    -------
    numpy.ndarray or None
        1-D array of values if the block exists, otherwise ``None``.

    Notes
    -----
    - All lines from the block header (line starting with ``">keyword"``)
      up to (but not including) the next line starting with ``">"`` or the
      end of file are scanned for floating point numbers.
    - Any comments starting with ``"//"`` are ignored.
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
    """Construct impedance tensor and optional variance from blocks."""
    n = freq.size
    Z = np.zeros((n, 2, 2), dtype=np.complex128)
    Z_var = np.zeros((n, 2, 2), dtype=float) * np.nan

    comp_map = {
        "ZXX": (0, 0),
        "ZXY": (0, 1),
        "ZYX": (1, 0),
        "ZYY": (1, 1),
    }

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
    """Construct tipper and optional variance from blocks."""
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

    T_var = np.zeros((n, 1, 2), dtype=float) * np.nan
    T_var[:, 0, 0] = txvar[:n]
    T_var[:, 0, 1] = tyvar[:n]

    return T, T_var


# ---------------------------------------------------------------------------
# Phoenix SPECTRA handling (incl. ROTSPEC)
# ---------------------------------------------------------------------------


def parse_spectra_blocks(
    edi_text: str,
) -> List[Tuple[float, float, float, np.ndarray]]:
    """Parse Phoenix ``>SPECTRA`` blocks.

    Parameters
    ----------
    edi_text : str
        Entire EDI file content.

    Returns
    -------
    list of tuples
        Each tuple is ``(f, avgt, rot, mat7)`` where

        * ``f`` is the frequency in Hz (float),
        * ``avgt`` is the averaging time in seconds (float, or NaN if not
          present),
        * ``rot`` is the rotation angle from ``ROTSPEC`` in degrees
          (float, or NaN if not present),
        * ``mat7`` is a real-valued (7, 7) array in Phoenix encoding:

          - diagonal: autospectra,
          - lower triangle: real parts,
          - upper triangle: imaginary parts.
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
        f = float(fm.group(1).replace("D", "E"))

        am = re.search(
            r"AVGT\s*=?\s*([0-9.]+[ED][+\-]?\d+|[0-9.]+)",
            block_text,
            flags=re.IGNORECASE,
        )
        avgt = float(am.group(1).replace("D", "E")) if am else float("nan")

        rm = re.search(
            r"ROTSPEC\s*=?\s*([0-9.\-+]+[ED][+\-]?\d+|[0-9.\-+]+)",
            block_text,
            flags=re.IGNORECASE,
        )
        rot = float(rm.group(1).replace("D", "E")) if rm else float("nan")

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
        blocks.append((f, avgt, rot, mat7))

    return blocks


def reconstruct_S_phoenix(mat7: np.ndarray) -> np.ndarray:
    """Reconstruct complex 7×7 spectra matrix from Phoenix real-valued encoding.

    Parameters
    ----------
    mat7 : numpy.ndarray
        Real-valued (7, 7) array with autos on the diagonal, lower triangle
        = Re(S_ij), upper triangle = Im(S_ij).

    Returns
    -------
    numpy.ndarray
        Complex Hermitian matrix S of shape (7, 7).
    """
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
    """Derive impedance tensor Z and tipper T from a Phoenix spectra matrix S.

    Parameters
    ----------
    S : numpy.ndarray
        Complex 7×7 spectra matrix as returned by :func:`reconstruct_S_phoenix`.
    ref : {"RH", "LH"}, optional
        Reference convention for impedance handedness. Default is ``"RH"``.

    Returns
    -------
    Z : numpy.ndarray
        Complex impedance tensor of shape (2, 2).
    T : numpy.ndarray
        Complex tipper of shape (1, 2).

    Notes
    -----
    Channel indices follow the standard Phoenix ordering (Hx, Hy, Hz, Ex, Ey,
    ...). The recovered impedance comes out in units of µV/m per nT and is
    divided by 1e3 to yield SI Ohm (V/m per T).
    """
    h1, h2 = 0, 1  # Hx, Hy
    hz = 2         # Hz
    ex, ey = 3, 4  # Ex, Ey

    SHH = np.array(
        [[S[h1, h1], S[h1, h2]],
         [S[h2, h1], S[h2, h2]]],
        dtype=np.complex128,
    )
    SEH = np.array(
        [[S[ex, h1], S[ex, h2]],
         [S[ey, h1], S[ey, h2]]],
        dtype=np.complex128,
    )
    SBH = np.array([[S[hz, h1], S[hz, h2]]], dtype=np.complex128)

    try:
        SHH_inv = np.linalg.inv(SHH)
    except np.linalg.LinAlgError:
        SHH_inv = np.linalg.pinv(SHH)

    Z = SEH @ SHH_inv
    T = SBH @ SHH_inv

    # Phoenix spectra typically give Z in µV/m per nT.
    # That is 1e3 times larger than SI impedance (V/m)/T; correct by 1e3.
    Z /= 1.0e3

    if ref.upper() == "LH":
        Z = -Z
        T = -T

    return Z, T


def _load_from_spectra(
    text: str,
    *,
    ref: str = "RH",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """High-level helper: build freq, Z, T, rot from Phoenix ``>SPECTRA``."""
    blocks = parse_spectra_blocks(text)
    if not blocks:
        raise RuntimeError("No usable >SPECTRA blocks found in EDI text.")

    n = len(blocks)
    freq = np.empty(n, dtype=float)
    Z = np.empty((n, 2, 2), dtype=np.complex128)
    T = np.empty((n, 1, 2), dtype=np.complex128)
    rot = np.empty(n, dtype=float)

    for k, (f, avgt, rot, mat7) in enumerate(blocks):
        freq[k] = f
        rot[k] = rot
        S = reconstruct_S_phoenix(mat7)
        Zk, Tk = ZT_from_S(S, ref=ref)
        Z[k] = Zk
        T[k] = Tk

    return freq, Z, T, rot


# ---------------------------------------------------------------------------
# High-level loader
# ---------------------------------------------------------------------------


def load_edi(
    path: str | Path,
    *,
    prefer_spectra: bool = True,
    ref: str = "RH",
    err_kind: str = "var",
    drop_invalid_periods: bool = True,
    invalid_sentinel: float = 1.0e30,
) -> Dict[str, Any]:
    """Load an EDI file into a dictionary.

    The decision between Phoenix SPECTRA and classical Z/T tables is made
    purely based on the presence of Phoenix ``>SPECTRA`` blocks:

    - If any ``>SPECTRA`` blocks are found and ``prefer_spectra=True`` (default),
      the SPECTRA path is used to reconstruct Z and T.
    - Otherwise, the classical Z/T table path is used.

    Parameters
    ----------
    path : str or pathlib.Path
        Input EDI file path.
    prefer_spectra : bool, optional
        If True (default) and Phoenix ``>SPECTRA`` blocks are present, use
        the spectra path. If False, ignore spectra and use classical blocks
        only.
    ref : {"RH", "LH"}, optional
        Handedness convention for the SPECTRA path. Default is ``"RH"``.
    err_kind : {"var", "std"}, optional
        Interpretation of returned error arrays:

        - ``"var"`` (default): errors such as ``Z_err`` / ``T_err`` are
          variances (e.g. from ``Z??.VAR``).
        - ``"std"``: errors are one-sigma standard deviations.

        For Phoenix spectra, no errors are derived at present, so ``Z_err``
        and ``T_err`` are ``None`` regardless of this flag.
        The value is stored under ``"err_kind"`` in the output dictionary.
    drop_invalid_periods : bool, optional
        If True (default), rows that contain missing values (NaN or absolute
        values exceeding ``invalid_sentinel``) in any impedance or tipper
        component *or their errors* are dropped before returning the dictionary.
    invalid_sentinel : float, optional
        Values with absolute magnitude larger than this threshold are treated
        as invalid. Default is ``1.0e30``.

    Returns
    -------
    dict
        EDI dictionary as described in the module-level docstring.

    Raises
    ------
    RuntimeError
        If neither SPECTRA nor classical Z/T blocks can be interpreted.
    ValueError
        If an unknown ``err_kind`` is given.
    """
    if err_kind not in {"var", "std"}:
        raise ValueError(f"Unknown err_kind {err_kind!r}; expected 'var' or 'std'.")

    text = read_edi_text(path)
    lines = _split_lines(text)
    header_lines = _extract_header_lines(lines)
    meta = _parse_simple_meta(header_lines)

    # Extract basic site metadata (if available) for explicit inclusion
    lat = meta.get("lat", None)
    # normalise lon: prefer 'lon', fall back to 'long'
    lon = meta.get("lon", meta.get("long", None))
    elev = meta.get("elev", None)

    # Decide which path to use
    use_spectra = _has_spectra(text) and prefer_spectra

    rot: Optional[np.ndarray]

    if use_spectra:
        freq, Z, T, rot = _load_from_spectra(text, ref=ref)
        Z_var = None
        T_var = None
        source_kind = "spectra"
    else:
        # Classical table-style path
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

        # Rotation block (ZROT) if present
        zrot_vals = _extract_block_values(lines, "ZROT")
        if zrot_vals is not None:
            if zrot_vals.size < freq.size:
                raise ValueError(
                    "ZROT block has fewer entries than frequencies."
                )
            rot = zrot_vals[: freq.size].copy()
        else:
            rot = None

        source_kind = "tables"

    # Sort frequencies ascending
    order = np.argsort(freq)
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

    # Optionally drop invalid rows (Z, T, and their errors)
    mask_valid = np.ones(freq.shape, dtype=bool)
    if drop_invalid_periods:
        def _collapse_any(bad: np.ndarray) -> np.ndarray:
            if bad.ndim == 1:
                return bad
            axes = tuple(range(1, bad.ndim))
            return np.any(bad, axis=axes)

        # Z itself
        bad_Z = ~np.isfinite(Z.real) | ~np.isfinite(Z.imag)
        bad_Z |= (np.abs(Z.real) > invalid_sentinel) | (
            np.abs(Z.imag) > invalid_sentinel
        )
        mask_valid &= ~_collapse_any(bad_Z)

        # T itself (if present)
        if T is not None:
            bad_T = ~np.isfinite(T.real) | ~np.isfinite(T.imag)
            bad_T |= (np.abs(T.real) > invalid_sentinel) | (
                np.abs(T.imag) > invalid_sentinel
            )
            mask_valid &= ~_collapse_any(bad_T)

        # Z variances/errors (if present)
        if Z_var is not None:
            bad_Zvar = ~np.isfinite(Z_var) | (np.abs(Z_var) > invalid_sentinel)
            mask_valid &= ~_collapse_any(bad_Zvar)

        # T variances/errors (if present)
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

    # Map internal variance arrays -> returned error arrays depending on err_kind
    if Z_var is not None:
        Z_err = Z_var if err_kind == "var" else np.sqrt(Z_var)
    else:
        Z_err = None

    if T_var is not None:
        T_err = T_var if err_kind == "var" else np.sqrt(T_var)
    else:
        T_err = None

    edi: Dict[str, Any] = {
        "freq": freq,
        "Z": Z,
        "T": T,
        "Z_err": Z_err,
        "T_err": T_err,
        "P": None,
        "P_err": None,
        "rot": rot if rot is not None else None,
        "err_kind": err_kind,
        "header_raw": header_lines,
        "source_kind": source_kind,
        # explicit site metadata (normalised)
        "lat": lat,
        "lon": lon,
        "elev": elev,
    }

    # Add everything else from meta (site name, survey info, possibly 'long', etc.)
    edi.update(meta)

    return edi


# if EstimateErrors:
#     edi_dict = estimate_errors(edi_dict=edi_dict, method=ErrMethod)

def estimate_errors(edi_dict: Dict[str, Any],
               method: List)-> Dict[str, Any]:
    '''
    Estimate new errors using real spread of data.

    Parameters
    ----------
    edi_dict : Dict[str, Any]
        data dictionary
    errors : Dict[str,np.ndarray]
        error dictionary which will be uded to replace the errors.
        relative errors.

    Returns
    -------
    new_edi_dict
        with errors replaced.

    '''
    sys.exit('estimate_errors: not yet implementd! Exit.')
    edi_dict_new = edi_dict.copy


    return edi_dict_new




# if Interpolate:
#     edi_dict = interpolate_data(edi_dict=edi_dict, method=Method)
def interpolate_data(edi_dict: Dict[str, Any],
               method: List,
               )-> Dict[str, Any]:
    '''
    Interpolate data to k points per decade for all data


    Parameters
    ----------
    edi_dict : Dict[str, Any]
        impuut data dict
    method : Dict[str, Any]
        DESCRIPTION.
     : TYPE
        DESCRIPTION.

    Returns
    -------
    Dict[str, Any]
        DESCRIPTION.

    '''
    # sys.exit('interpolate_data: not yet implementd! Exit.')
    edi_dict_new = edi_dict.copy

    freqs = edi_dict_new['freq']
    spl_meth = method['meth']
    new_freqs = method['logfreq']
    nf = len(new_freqs)

    if 'Z' in edi_dict_new:
        tmp  = edi_dict_new['Z']
        tmp_new = np.zeros((nf,2,2), dtype=complex)
        spline = make_spline(freqs, tmp[:, 0, 0], lam=None)
        tmp_new[:, 0, 0] = spline(new_freqs)
        spline = make_spline(freqs, tmp[:, 1, 0], lam=None)
        tmp_new[:, 1, 0] = spline(new_freqs)
        spline = make_spline(freqs, tmp[:, 0, 1], lam=None)
        tmp_new[:, 0, 1] = spline(new_freqs)
        spline = make_spline(freqs, tmp[:, 1, 1], lam=None)
        tmp_new[:, 1, 1] = spline(new_freqs)
        edi_dict_new['Z'] = tmp_new

        tmp  = edi_dict_new['Zerr']
        tmp_new = np.zeros((nf,2,2), dtype=complex)
        spline = make_spline(freqs, tmp[:, 0, 0], lam=None)
        tmp_new[:, 0, 0] = spline(new_freqs)
        spline = make_spline(freqs, tmp[:, 1, 0], lam=None)
        tmp_new[:, 1, 0] = spline(new_freqs)
        spline = make_spline(freqs, tmp[:, 0, 1], lam=None)
        tmp_new[:, 0, 1] = spline(new_freqs)
        spline = make_spline(freqs, tmp[:, 1, 1], lam=None)
        tmp_new[:, 1, 1] = spline(new_freqs)
        edi_dict_new['Zerr'] = tmp_new

    if 'T' in edi_dict_new:
        tmp  = edi_dict_new['T']
        tmp_new = np.zeros((nf,2,1), dtype=complex)
        spline = make_spline(freqs, tmp[:, 0, 0], lam=None)
        tmp_new[:, 0, 0 ] = spline(new_freqs)
        spline = make_spline(freqs, tmp[:, 1, 0], lam=None)
        tmp_new[:, 1, 0] = spline(new_freqs)
        edi_dict_new['T'] = tmp_new

        tmp  = edi_dict_new['Terr']
        tmp_new = np.tmperos((nf,2,1), dtype=complex)
        spline = make_spline(freqs, tmp[:, 0, 0], lam=None)
        tmp_new[:, 0, 0] = spline(new_freqs)
        spline = make_spline(freqs, tmp[:, 1, 0], lam=None)
        tmp_new[:, 1, 0] = spline(new_freqs)
        edi_dict_new['Terr'] = tmp_new

    if 'P' in edi_dict_new:
         tmp  = edi_dict_new['P']
         tmp_new = np.tmperos((nf,2,2), dtype=float)
         spline = make_spline(freqs, tmp[:, 0, 0], lam=None)
         tmp_new[:, 0, 0 ] = spline(new_freqs)
         spline = make_spline(freqs, tmp[:, 1, 0], lam=None)
         tmp_new[:, 1, 0] = spline(new_freqs)
         spline = make_spline(freqs, tmp[:, 0, 1], lam=None)
         tmp_new[:, 0, 1 ] = spline(new_freqs)
         spline = make_spline(freqs, tmp[:, 1, 1], lam=None)
         tmp_new[:, 1, 1] = spline(new_freqs)
         edi_dict_new['P'] = tmp_new

         tmp  = edi_dict_new['Perr']
         tmp_new = np.tmperos((nf,2,2), dtype=float)
         spline = make_spline(freqs, tmp[:, 0, 0], lam=None)
         tmp_new[:, 0, 0 ] = spline(new_freqs)
         spline = make_spline(freqs, tmp[:, 1, 0], lam=None)
         tmp_new[:, 1, 0] = spline(new_freqs)
         spline = make_spline(freqs, tmp[:, 0, 1], lam=None)
         tmp_new[:, 0, 1 ] = spline(new_freqs)
         spline = make_spline(freqs, tmp[:, 1, 1], lam=None)
         tmp_new[:, 1, 1] = spline(new_freqs)
         edi_dict_new['Perr'] = tmp_new

    return edi_dict_new



# if SetErrors:
#     edi_dict = set_errors(edi_dict=edi_dict, errors=Errors)
def set_errors(edi_dict: Dict[str, Any],
               errors: Dict[str,np.ndarray])-> Dict[str, Any]:
    '''
    Replace errors with given relative errors.

    Parameters
    ----------
    edi_dict : Dict[str, Any]
        data dictionary
    errors : Dict[str,np.ndarray]
        error dictionary which will be uded to replace the errors.
        relative errors.

    Returns
    -------
    new_edi_dict
        with errors replaced.

    '''
    sys.exit('set_errors: not yet implementd! Exit.')
    edi_dict_new = edi_dict.copy


    return edi_dict_new

# if Rotation:
#     edi_dict = rotate_data(edi_dict=edi_dict, angle=Angle)
def rotate_data(edi_dict: Dict[str, Any],
               angle: float = 0.,
               degrees=True )-> Dict[str, Any]:
    '''
    Rotate data (e.g for magnetic to geographic system)

    Parameters
    ----------
    edi_dict : Dict[str, Any]
        DESCRIPTION.
    angle : float
        Rotation angle.
    degrees : boolean, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    Dict[str, Any]
        edi_dict with rotated data

    '''
    edi_dict_new = edi_dict.copy

    if degrees:
        ang = np.radians(angle)
    else:
        ang = angle

    c = np.cos(ang)
    s = np.sin(ang)
    R = np.array([[c,  s], [-s,  c]])


    freq = edi_dict_new['freq']
    rot = edi_dict_new['rot']
    Z = edi_dict_new['Z']
    Zerr = edi_dict_new['Zerr']
    T = edi_dict_new['T']
    Terr = edi_dict_new['Terr']
    P = edi_dict_new['P']
    Perr = edi_dict_new['Perr']

    for f in np.arange(len(freq)):
        Z = R @ Z[f,:,:] @ R.T
        Zerr = R @ Zerr[f,:,:] @ R.T
        T = R @ T[f,:,:]
        Terr = R @ Terr[f,:,:]
        P = R @ P[f,:,:] @ R.T
        Perr = R @ Perr[f,:,:] @ R.T

    edi_dict_new ['rot'] = rot + angle*np.ones_like(rot)
    edi_dict_new ['Z'] = Z
    edi_dict_new ['Zerr'] = Zerr
    edi_dict_new ['T'] = T
    edi_dict_new ['Terr'] = Terr
    edi_dict_new ['P'] = P
    edi_dict_new ['Perr'] = Perr

    return edi_dict_new

# ---------------------------------------------------------------------------
# Phase tensor + Monte-Carlo error propagation
# ---------------------------------------------------------------------------


def compute_pt(
    Z: np.ndarray,
    Z_err: Optional[np.ndarray] = None,
    *,
    err_kind: str = "var",
    nsim: int = 200,
    random_state: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Compute the phase tensor from impedance and optionally propagate errors.

    The phase tensor Φ is defined as

    .. math::

        \\Phi = X^{-1} Y,

    where :math:`Z = X + i Y` and X, Y are real 2×2 matrices. This function
    computes Φ for each frequency and, if ``Z_err`` is provided, performs a
    simple Monte-Carlo error propagation to estimate the variance (or
    standard deviation) of Φ entries.

    Parameters
    ----------
    Z : numpy.ndarray
        Complex impedance array with shape ``(n, 2, 2)``.
    Z_err : numpy.ndarray or None, optional
        Impedance uncertainty array with the same shape as ``Z``. The
        interpretation is controlled by ``err_kind``:

        - ``"var"`` (default): values are treated as **variances** of the
          complex impedance entries (for example values read from
          ``Z??.VAR`` blocks).
        - ``"std"``: values are treated as 1-sigma **standard deviations``.

        If None, only the phase tensor is returned.
    err_kind : {"var", "std"}, optional
        Interpretation of ``Z_err``. See above. Default is ``"var"``.
    nsim : int, optional
        Number of Monte-Carlo realisations used for error propagation.
        Default is 200.
    random_state : numpy.random.Generator, optional
        Optional random number generator to use. If None (default), a fresh
        ``Generator`` instance is created.

    Returns
    -------
    P : numpy.ndarray
        Phase tensor array with shape ``(n, 2, 2)``.
    P_err : numpy.ndarray or None
        Estimated variance or standard deviation of Φ entries with the same
        shape as ``P``. If ``Z_err`` is None, ``P_err`` is None.

    Notes
    -----
    - The Monte-Carlo approach assumes independent Gaussian errors for each
      complex impedance entry.
    - ``P_err`` contains **variances** if ``err_kind="var"`` and standard
      deviations if ``err_kind="std"``.
    """
    Z = np.asarray(Z, dtype=np.complex128)
    if Z.shape[-2:] != (2, 2):
        raise ValueError("Z must have shape (n, 2, 2).")

    n = Z.shape[0]
    P = np.zeros((n, 2, 2), dtype=float)

    # Deterministic phase tensor
    for k in range(n):
        X = Z[k].real
        Y = Z[k].imag
        try:
            X_inv = np.linalg.inv(X)
        except np.linalg.LinAlgError:
            P[k] = np.nan
            continue
        P[k] = X_inv @ Y

    if Z_err is None:
        return P, None

    Z_err = np.asarray(Z_err, dtype=float)
    if Z_err.shape != Z.shape:
        raise ValueError("Z_err must have the same shape as Z.")

    if random_state is None:
        rng = np.random.default_rng()
    else:
        rng = random_state

    if err_kind == "var":
        sigma = np.sqrt(Z_err)
    elif err_kind == "std":
        sigma = Z_err
    else:
        raise ValueError("err_kind must be 'var' or 'std'.")

    P_sims = np.zeros((nsim, n, 2, 2), dtype=float)
    for s in range(nsim):
        d_re = rng.standard_normal(Z.shape) * sigma / np.sqrt(2.0)
        d_im = rng.standard_normal(Z.shape) * sigma / np.sqrt(2.0)
        Z_samp = (Z.real + d_re) + 1j * (Z.imag + d_im)
        for k in range(n):
            X = Z_samp[k].real
            Y = Z_samp[k].imag
            try:
                X_inv = np.linalg.inv(X)
            except np.linalg.LinAlgError:
                P_sims[s, k] = np.nan
                continue
            P_sims[s, k] = X_inv @ Y

    with np.errstate(invalid="ignore"):
        var_P = np.nanvar(P_sims, axis=0)

    if err_kind == "var":
        P_err = var_P
    else:
        P_err = np.sqrt(var_P)

    return P, P_err


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def _format_block(values: np.ndarray, n_per_line: int = 6) -> str:
    """Format 1-D numeric data into multi-line EDI block text."""
    vals = np.asarray(values, dtype=float).ravel()
    chunks: List[str] = []
    for i in range(0, vals.size, n_per_line):
        line_vals = vals[i : i + n_per_line]
        line = " ".join(f"{v: .6E}" for v in line_vals)
        chunks.append(" " + line)
    return "\n".join(chunks)


def save_edi(
    path: str | Path,
    edi: Dict[str, Any],
    *,
    numbers_per_line: int = 6,
    add_pt_blocks: bool = True,
    pt_err_kind: Optional[str] = None,
    lon_keyword: str = "LON",
) -> None:
    """Write an EDI dictionary to a classical table-style EDI file.

    Parameters
    ----------
    path : str or pathlib.Path
        Output EDI file path.
    edi : dict
        EDI dictionary with keys such as ``"freq"``, ``"Z"``, ``"T"``,
        ``"Z_err"``, ``"T_err"``, ``"P"``, ``"P_err"``, and metadata like
        ``"station"``, ``"lat_deg"``, ``"lon_deg"``, ``"elev_m"``.
    numbers_per_line : int, optional
        Number of numeric values printed per data line. Default is 6.
    add_pt_blocks : bool, optional
        If True (default) and ``"P"`` is present, write phase tensor blocks
        ``PXX``, ``PXY``, ``PYX``, ``PYY`` and, if available, corresponding
        variance blocks ``PXX.VAR``, etc.
    pt_err_kind : {"var", "std"} or None, optional
        How to interpret ``edi["P_err"]``. If ``"var"``, values are written
        as variances. If ``"std"``, values are squared before writing so that
        the EDI file stores variances. If None (default), the function tries
        to infer a sensible value from ``edi.get("err_kind")``.
    lon_keyword : {"LON", "LONG"}, optional
        Longitude keyword to use in the header. Some programs expect
        ``"LONG"`` instead of ``"LON"``. Default is ``"LON"``.
    """
    p = Path(path)

    freq = np.asarray(edi["freq"], dtype=float).ravel()
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
    lat_deg = edi.get("lat_deg")
    lon_deg = edi.get("lon_deg")
    elev_m = edi.get("elev_m")
    err_kind = edi.get("err_kind", "var")

    lines: List[str] = []

    # HEAD
    lines.append(">HEAD")
    lines.append(f'  DATAID="{station}"')
    if lat_deg is not None:
        lines.append(f"  LAT={lat_deg: .6f}")
    if lon_deg is not None:
        lines.append(f"  {lon_keyword}={lon_deg: .6f}")
    if elev_m is not None:
        lines.append(f"  ELEV={elev_m: .6f}")
    lines.append('  STDVERS="SEG 1.0"')
    lines.append('  PROGVERS="ediproc.py"')
    lines.append('  PROGDATE="2025-11-20"')
    lines.append("  EMPTY=1.0E32")
    lines.append("")

    # FREQ
    lines.append(">FREQ")
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
    n = freq.size
    comp_map = {
        "ZXX": (0, 0),
        "ZXY": (0, 1),
        "ZYX": (1, 0),
        "ZYY": (1, 1),
    }

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
            if err_kind == "std":
                var = Z_err_arr[:, i, j] ** 2
            else:
                var = Z_err_arr[:, i, j]
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
            if err_kind == "std":
                txvar = T_err_arr[:, 0, 0] ** 2
                tyvar = T_err_arr[:, 0, 1] ** 2
            else:
                txvar = T_err_arr[:, 0, 0]
                tyvar = T_err_arr[:, 0, 1]

            lines.append(">TXVAR.EXP")
            lines.append(_format_block(txvar, n_per_line=numbers_per_line))
            lines.append("")
            lines.append(">TYVAR.EXP")
            lines.append(_format_block(tyvar, n_per_line=numbers_per_line))
            lines.append("")

    # Phase tensor blocks (optional)
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
    """Build a tidy :class:`pandas.DataFrame` from an EDI dictionary.

    Parameters
    ----------
    edi : dict
        EDI dictionary, typically returned by :func:`ediproc.load_edi`. It
        must contain at least:

        ``"freq"`` : 1-D array of frequencies [Hz].
        ``"Z"``    : complex impedance, shape ``(n, 2, 2)``.

        Additional optional keys that are recognised:

        ``"T"`` : complex tipper, shape ``(n, 1, 2)``.
        ``"P"`` : phase tensor, shape ``(n, 2, 2)`` (real).
        ``"Z_err"`` : impedance uncertainties, same shape as ``"Z"``.
        ``"T_err"`` : tipper uncertainties, same shape as ``"T"``.
        ``"P_err"`` : phase tensor uncertainties, same shape as ``"P"``.
        ``"rot"`` : rotation angles in degrees, shape ``(n,)``.
        ``"station"``, ``"lat_deg"``, ``"lon_deg"``, ``"elev_m"``,
        ``"err_kind"`` : metadata copied into ``df.attrs``.

    include_rho_phi : bool, optional
        If True (default), compute apparent resistivity and phase from Z
        and add corresponding columns.
    include_tipper : bool, optional
        If True (default) and a tipper array is present, include tipper
        components and their uncertainties.
    include_pt : bool, optional
        If True (default) and a phase tensor array is present, include P
        components and their uncertainties.
    add_period : bool, optional
        If True (default), add a ``"period"`` column (1/f).
    err_kind : {"var", "std"} or None, optional
        Interpretation of ``Z_err``, ``T_err`` and ``P_err``. If None
        (default), the function tries to infer from ``edi.get("err_kind")``
        and falls back to ``"var"``.
    mu0 : float, optional
        Magnetic permeability [H/m] for converting impedance magnitude to
        apparent resistivity. Default is the vacuum value.

    Returns
    -------
    pandas.DataFrame
        Dataframe with columns and attributes as described above.

    Notes
    -----
    - Error propagation is approximate; it assumes independent Gaussian
      uncertainties for each impedance/tipper component.
    - For impedance, ``Z_err`` is interpreted as variance (or std) of the
      complex entry; real and imaginary parts are treated symmetrically.
    """
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
            period = np.where(freq > 0.0, 1.0 / freq, np.nan)
        df["period"] = period

    # Optional rotation column
    rot = edi.get("rot")
    if rot is not None:
        rot_arr = np.asarray(rot, dtype=float).ravel()
        if rot_arr.size == n:
            df["rot"] = rot_arr

    # err_kind handling
    if err_kind is None:
        err_kind = edi.get("err_kind", "var")
    if err_kind not in ("var", "std"):
        raise ValueError("err_kind must be 'var' or 'std'.")

    # ------------------------------------------------------------------ ρ and φ
    if include_rho_phi:
        omega = 2.0 * np.pi * freq  # angular frequency
        comp_map = {"xx": (0, 0), "xy": (0, 1), "yx": (1, 0), "yy": (1, 1)}

        for name, (i, j) in comp_map.items():
            Zij = Z[:, i, j]
            mag = np.abs(Zij)
            rho = (mag**2) / (mu0 * omega)
            with np.errstate(divide="ignore", invalid="ignore"):
                phi = np.degrees(np.arctan2(Zij.imag, Zij.real))
            df[f"rho_{name}"] = rho
            df[f"phi_{name}"] = phi

        Z_err = edi.get("Z_err")
        if Z_err is not None:
            Z_err = np.asarray(Z_err, dtype=float)
            if Z_err.shape != Z.shape:
                raise ValueError("edi['Z_err'] must have the same shape as 'Z'.")

            if err_kind == "var":
                sigma_Z = np.sqrt(Z_err)
            else:
                sigma_Z = Z_err

            for name, (i, j) in comp_map.items():
                Zij = Z[:, i, j]
                mag = np.abs(Zij)
                rho = df[f"rho_{name}"].to_numpy()
                denom = mu0 * omega

                # Approximate std of ρ using error propagation
                std_rho = 2.0 * sigma_Z[:, i, j] * mag / np.where(
                    denom > 0.0, denom, np.nan
                )
                df[f"rho_{name}_err"] = std_rho

                # Phase error (radians) ~ σ / |Z|
                with np.errstate(divide="ignore", invalid="ignore"):
                    std_phi_rad = np.where(mag > 0.0, sigma_Z[:, i, j] / mag, np.nan)
                std_phi_deg = std_phi_rad * (180.0 / np.pi)
                df[f"phi_{name}_err"] = std_phi_deg

    # ------------------------------------------------------------------ Tipper
    if include_tipper and "T" in edi and edi["T"] is not None:
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

            if err_kind == "var":
                sigma_T = np.sqrt(T_err)
            else:
                sigma_T = T_err

            # Same sigma for real and imaginary parts of a given component
            df["Tx_re_err"] = sigma_T[:, 0, 0]
            df["Tx_im_err"] = sigma_T[:, 0, 0]
            df["Ty_re_err"] = sigma_T[:, 0, 1]
            df["Ty_im_err"] = sigma_T[:, 0, 1]

    # ------------------------------------------------------------------ Phase tensor
    if include_pt and "P" in edi and edi["P"] is not None:
        P = np.asarray(edi["P"], dtype=float)
        if P.shape != (n, 2, 2):
            raise ValueError("edi['P'] must have shape (n, 2, 2).")

        df["ptxx_re"] = P[:, 0, 0]
        df["ptxy_re"] = P[:, 0, 1]
        df["ptyx_re"] = P[:, 1, 0]
        df["ptyy_re"] = P[:, 1, 1]

        P_err = edi.get("P_err")
        if P_err is not None:
            P_err = np.asarray(P_err, dtype=float)
            if P_err.shape != P.shape:
                raise ValueError("edi['P_err'] must have the same shape as 'P'.")

            if err_kind == "var":
                sigma_P = np.sqrt(P_err)
            else:
                sigma_P = P_err

            df["ptxx_re_err"] = sigma_P[:, 0, 0]
            df["ptxy_re_err"] = sigma_P[:, 0, 1]
            df["ptyx_re_err"] = sigma_P[:, 1, 0]
            df["ptyy_re_err"] = sigma_P[:, 1, 1]

    # ------------------------------------------------------------------ Metadata in attrs
    meta_keys = ("station", "lat_deg", "lon_deg", "elev_m", "err_kind", "rot")
    for key in meta_keys:
        if key in edi:
            if key == "err_kind":
                df.attrs[key] = err_kind
            else:
                df.attrs[key] = edi[key]

    return df


def make_spline(x: np.ndarray, y: np.ndarray, lam: float | None = None):
    """
    Fit a smoothing spline using SciPy's make_smoothing_spline.
    Ensures x is sorted ascending.

    Parameters
    ----------
    x : np.ndarray
        1D array of x-values.
    y : np.ndarray
        1D array of y-values.
    lam : float, optional
        Smoothing parameter.

    Returns
    -------
    spline : PPoly
        Fitted spline object.

    Usage:

    # Choose new x-values for interpolation
    x_new = np.linspace(0, 10, 200)

    # Evaluate spline at new points
    y_new = spline_obj(x_new)

    Remarks
    -------
    Author: Volker Rath (DIAS), Date: 2025-11-21
    Generated by Copilot v1.0
    """
    sort_idx = np.argsort(x)
    x_sorted, y_sorted = x[sort_idx], y[sort_idx]
    spline_obj = make_smoothing_spline(x_sorted, y_sorted, lam=lam)

    if lam is None:
        print("$\lambda$ chosen via GCV (not exposed by BSpline)")
    else:
        print("$\lambsa$ is", lam)
    return spline_obj

def estimate_variance(y_true: np.ndarray, y_fit: np.ndarray) -> float:
    """
    Estimate residual variance.

    Parameters
    ----------
    y_true : np.ndarray
        Observed data.
    y_fit : np.ndarray
        Fitted values from spline.

    Returns
    -------
    var : float
        Residual variance estimate.

    Remarks
    -------
    Author: Volker Rath (DIAS), Date: 2025-11-21
    Generated by Copilot v1.0
    """
    residuals = y_true - y_fit
    return np.var(residuals, ddof=1)


def bootstrap_confidence_band(
    x: np.ndarray,
    y: np.ndarray,
    lam: float | None = None,
    n_bootstrap: int = 1000,
    ci: float = 0.95
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute bootstrap confidence intervals for spline predictions.
    Ensures each resample is strictly ascending before fitting.

    Parameters
    ----------
    x : np.ndarray
        Input x-values for evaluation.
    y : np.ndarray
        Observed y-values.
    lam : float, optional
        Smoothing parameter for spline.
    n_bootstrap : int
        Number of bootstrap resamples.
    ci : float
        Confidence level (e.g., 0.95 for 95%).

    Returns
    -------
    x_eval : np.ndarray
        Sorted x-values used for evaluation.
    lower : np.ndarray
        Lower confidence band.
    upper : np.ndarray
        Upper confidence band.

    Remarks
    -------
    Author: Volker Rath (DIAS), Date: 2025-11-21
    Generated by Copilot v1.0
    """
    # define evaluation grid (sorted original x)
    sort_idx = np.argsort(x)
    x_eval, y_sorted = x[sort_idx], y[sort_idx]
    n = len(x_eval)
    preds = np.zeros((n_bootstrap, n))

    for i in range(n_bootstrap):
        idx = np.random.choice(len(x), len(x), replace=True)
        x_res, y_res = x[idx], y[idx]

        # sort resampled data
        sort_idx_res = np.argsort(x_res)
        x_res_sorted, y_res_sorted = x_res[sort_idx_res], y_res[sort_idx_res]

        # enforce strictly ascending x by removing duplicates
        unique_x, unique_idx = np.unique(x_res_sorted, return_index=True)
        y_res_unique = y_res_sorted[unique_idx]

        if len(unique_x) < 4:
            # not enough unique points to fit spline, skip this bootstrap
            continue

        spline_res = make_smoothing_spline(unique_x, y_res_unique, lam=lam)
        preds[i, :] = spline_res(x_eval)

    alpha = 1 - ci
    lower = np.percentile(preds, 100 * alpha / 2, axis=0)
    upper = np.percentile(preds, 100 * (1 - alpha / 2), axis=0)
    return x_eval, lower, upper


def choose_lambda_gcv(x, y, lam_grid=None):
    if lam_grid is None:
        lam_grid = np.logspace(-3, 3, 50)
    best_score, best_lam, best_spline = np.inf, None, None
    for lam in lam_grid:
        spline = make_smoothing_spline(x, y, lam=lam)
        residuals = y - spline(x)
        score = np.mean((residuals**2)) / (1 - (len(spline.c) / len(x)))**2
        if score < best_score:
            best_score, best_lam, best_spline = score, lam, spline
    return best_spline, best_lam



def save_hdf(
    df: pd.DataFrame,
    path: str | Path,
    *,
    key: str = "mt",
    mode: str = "w",
    complevel: int = 4,
    complib: str = "zlib",
    **kwargs: Any,
) -> None:
    """Save a dataframe to an HDF5 file via :mod:`pandas`."""
    path = Path(path)
    try:
        df.to_hdf(
            path.as_posix(),
            key=key,
            mode=mode,
            complevel=complevel,
            complib=complib,
            **kwargs,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "pandas HDF5 support requires the 'tables' package."
        ) from exc


def save_ncd(
    df: pd.DataFrame,
    path: str | Path,
    *,
    engine: Optional[str] = None,
    dim: str = "period",
    dataset_name: str = "mt",
) -> None:
    """Save a dataframe to a NetCDF file using :mod:`xarray`."""
    try:
        import xarray as xr  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("save_netcdf requires the 'xarray' package.") from exc

    path = Path(path)

    if dim in df.columns:
        coord = df[dim].to_numpy()
        dim_name = dim
    elif "freq" in df.columns:
        coord = df["freq"].to_numpy()
        dim_name = "freq"
    else:
        raise ValueError(
            "DataFrame must contain either the dimension column given by 'dim' "
            "or a 'freq' column."
        )

    data_vars = {}
    for col in df.columns:
        if col == dim_name:
            continue
        data = df[col].to_numpy()
        data_vars[col] = (dim_name, data)

    coords = {dim_name: coord}
    ds = xr.Dataset(data_vars=data_vars, coords=coords)

    # Propagate attributes
    for k, v in df.attrs.items():
        ds.attrs[k] = v
    ds.attrs["dataset_name"] = dataset_name

    ds.to_netcdf(path.as_posix(), engine=engine)

# ----------------------------------------------------------------------
# EMTF-XML I/O (new)
# ----------------------------------------------------------------------

def read_emtf_xml(path):
    """
    Read MT data from an EMTF-XML file.

    Parameters
    ----------
    path : str
        Path to EMTF-XML file.

    Returns
    -------
    dict
        Dictionary with metadata and transfer functions.
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(path)
    root = tree.getroot()

    ns = {"emtf": "http://emtf.org/schema"}  # adjust namespace if needed

    # Example: parse station metadata
    station = root.find("emtf:Station", ns)
    metadata = {
        "id": station.get("id"),
        "name": station.findtext("emtf:Name", default="", namespaces=ns),
        "latitude": float(station.findtext("emtf:Latitude", default="nan", namespaces=ns)),
        "longitude": float(station.findtext("emtf:Longitude", default="nan", namespaces=ns)),
    }

    # Example: parse transfer function values
    tf_elements = root.findall("emtf:TransferFunction/emtf:Component", ns)
    transfer_functions = {}
    for comp in tf_elements:
        cname = comp.get("name")
        values = [float(v.text) for v in comp.findall("emtf:Value", ns)]
        transfer_functions[cname] = np.array(values)

    return {"metadata": metadata, "transfer_functions": transfer_functions}


def write_emtf_xml(data, path):
    """
    Write MT data to an EMTF-XML file.

    Parameters
    ----------
    data : dict
        Dictionary with metadata and transfer functions.
    path : str
        Output file path.
    """
    import xml.etree.ElementTree as ET

    ns = "http://emtf.org/schema"
    ET.register_namespace("emtf", ns)

    root = ET.Element("{%s}EMTF" % ns)

    # Station metadata
    station = ET.SubElement(root, "{%s}Station" % ns, id=data["metadata"].get("id", ""))
    ET.SubElement(station, "{%s}Name" % ns).text = data["metadata"].get("name", "")
    ET.SubElement(station, "{%s}Latitude" % ns).text = str(data["metadata"].get("latitude", ""))
    ET.SubElement(station, "{%s}Longitude" % ns).text = str(data["metadata"].get("longitude", ""))

    # Transfer functions
    tf = ET.SubElement(root, "{%s}TransferFunction" % ns)
    for cname, values in data["transfer_functions"].items():
        comp = ET.SubElement(tf, "{%s}Component" % ns, name=cname)
        for v in values:
            ET.SubElement(comp, "{%s}Value" % ns).text = str(v)

    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)

 # ----------------------------------------------------------------------
# Conversion helpers: EDI <-> EMTF-XML
# ----------------------------------------------------------------------

def edi_to_emtf(edi_path, emtf_path):
    """
    Convert an EDI file into EMTF-XML.

    Parameters
    ----------
    edi_path : str
        Path to input EDI file.
    emtf_path : str
        Path to output EMTF-XML file.
    """
    data = load_edi(edi_path)          # parse EDI into dict
    write_emtf_xml(data, emtf_path)    # write dict into EMTF-XML
    return emtf_path


def emtf_to_edi(emtf_path, edi_path):
    """
    Convert an EMTF-XML file into EDI.

    Parameters
    ----------
    emtf_path : str
        Path to input EMTF-XML file.
    edi_path : str
        Path to output EDI file.
    """
    data = read_emtf_xml(emtf_path)    # parse EMTF-XML into dict
    save_edi(data, edi_path)          # write dict into EDI
    return edi_path
