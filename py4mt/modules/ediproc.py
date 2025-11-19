#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ediproc.py
=================
Consolidated I/O utilities for Magnetotelluric EDI files used in the EDI
project.

This module provides a small, focused API for reading, processing, and writing
MT transfer functions from EDI files. It supports two main data sources:

1. Phoenix / SPECTRA-based EDIs
   -----------------------------
   These contain ">SPECTRA" blocks with a 7x7 real-valued matrix encoding the
   auto- and cross-spectra of E and H channels. The workflow is:

   - parse_spectra_blocks(...)
       Identify ">SPECTRA" blocks, extract frequency FREQ, averaging time AVGT,
       and the 7x7 matrix in the Phoenix encoding.
   - reconstruct_S_phoenix(...)
       Convert the real-valued 7x7 Phoenix matrix into a complex Hermitian
       spectra matrix S.
   - ZT_from_S(...)
       Recover impedance Z and tipper T from the spectra matrix S, applying
       a constant scaling factor that converts the raw Phoenix units to SI.

2. Classical table-based EDIs (for example ModEM / GFZ)
   ----------------------------------------------------
   These provide tabulated Z/T values via ">FREQ" and ">Zxxr", ">Zxxi", etc.,
   possibly including ".EXP" suffixes. The workflow is:

   - parse_block_values(...)
       Parse frequencies and Z/T from classical value blocks, supporting both:
         * "FREQ =" per-line style, and
         * ">FREQ NFREQ=..." block style.

High-level functions
--------------------

- read_edi_text(path)
- parse_spectra_blocks(edi_text)
- reconstruct_S_phoenix(mat7)
- ZT_from_S(S, ref='RH')
- parse_block_values(edi_text)
- load_edi(path, *, prefer_spectra=True, ref='RH',
           rotate_deg=0.0, fill_invalid=None,
           drop_invalid_periods=False, invalid_sentinel=1.0e30)
- dataframe_from_arrays(freq, Z, T=None, PT=None, station=None)
- write_edi(path, *, station, freq, Z, T=None,
            lat_deg=None, lon_deg=None, elev_m=None,
            header_meta=None, numbers_per_line=6)
- save_edi(edi_path, out_path=None, *,
           prefer_spectra=True, ref='RH', rotate_deg=0.0,
           fill_invalid=None, drop_invalid_periods=False,
           invalid_sentinel=1.0e30, numbers_per_line=6,
           lat_deg=None, lon_deg=None, elev_m=None,
           header_meta=None)
- write_edi_from_npz(npz_path, out_path=None, *,
                     numbers_per_line=6,
                     lat_deg=None, lon_deg=None, elev_m=None)

Notes
-----
- The Phoenix SPECTRA path is only used when `prefer_spectra=True` and
  ">SPECTRA" blocks are actually present. For files like PONT.edi, which are
  pure table-style EDIs, only the table path is used.
- Invalid data handling is controlled by `drop_invalid_periods` and
  `invalid_sentinel`. Rows with NaN/inf or very large sentinel values (for
  example 1.0e32) in Z/T can be removed before further processing.

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-11-18
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator, Iterable, Optional, Tuple, Dict, Any

import io
import re

import numpy as np
import pandas as pd

MU0: float = 4e-7 * np.pi

__all__ = [
    "read_edi_text",
    "parse_spectra_blocks",
    "reconstruct_S_phoenix",
    "ZT_from_S",
    "parse_block_values",
    "load_edi",
    "compute_phase_tensor",
    "dataframe_from_arrays",
    "write_edi",
    "save_edi",
    "write_edi_from_npz",
]


# ----------------------------------------------------------------------
# Small linear-algebra helpers
# ----------------------------------------------------------------------


def _rot2(theta_rad: float) -> np.ndarray:
    """Return the 2x2 rotation matrix for a counter-clockwise angle.

    Parameters
    ----------
    theta_rad : float
        Rotation angle in radians (counter-clockwise).

    Returns
    -------
    numpy.ndarray
        2x2 array representing the rotation matrix.
    """
    c = float(np.cos(theta_rad))
    s = float(np.sin(theta_rad))
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _apply_rot2(Z: np.ndarray, theta_rad: float) -> np.ndarray:
    """Rotate a 2x2 tensor (or stack of them) by angle ``theta_rad``.

    Parameters
    ----------
    Z : numpy.ndarray
        Either shape (2, 2) or (n, 2, 2). The last two dimensions are treated
        as the tensor to be rotated.
    theta_rad : float
        Rotation angle in radians (counter-clockwise).

    Returns
    -------
    numpy.ndarray
        Array of the same shape as ``Z``, rotated by ``theta_rad``.
    """
    R = _rot2(theta_rad)
    if Z.ndim == 2:
        return R @ Z @ R.T
    return np.einsum("ab,...bc,cd->...ad", R, Z, R.T)


def _rotate_ZT(
    Z: np.ndarray,
    T: Optional[np.ndarray],
    rotate_deg: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Rotate Z and T by a given angle in degrees.

    Parameters
    ----------
    Z : numpy.ndarray
        Impedance tensor of shape (n, 2, 2).
    T : numpy.ndarray or None
        Tipper array of shape (n, 1, 2), or None if no tipper is available.
    rotate_deg : float
        Rotation angle in degrees (counter-clockwise).

    Returns
    -------
    Z_rot : numpy.ndarray
        Rotated impedance tensor with the same shape as Z.
    T_rot : numpy.ndarray or None
        Rotated tipper with the same shape as T, or None if T was None.
    """
    if rotate_deg == 0.0:
        return Z, T

    theta = np.deg2rad(rotate_deg)
    Z_rot = _apply_rot2(Z, theta)

    if T is None:
        return Z_rot, None

    R = _rot2(theta)
    t_vec = T.reshape(-1, 2).T  # (2, n)
    t_rot = (R @ t_vec).T.reshape(T.shape)
    return Z_rot, t_rot


def _fill_invalid_inplace(arr: Optional[np.ndarray], fill_value: float) -> None:
    """Replace non-finite entries in an array in-place with a constant.

    Parameters
    ----------
    arr : numpy.ndarray or None
        Array whose non-finite values should be replaced. If None, nothing is
        done.
    fill_value : float
        Value used to replace NaN/inf entries.
    """
    if arr is None:
        return
    mask = ~np.isfinite(arr)
    if not np.any(mask):
        return
    if np.iscomplexobj(arr):
        arr[mask] = fill_value + 0j
    else:
        arr[mask] = fill_value


# ----------------------------------------------------------------------
# Core text / parsing helpers
# ----------------------------------------------------------------------


def read_edi_text(path: Path | str) -> str:
    """Read an EDI file as text using latin-1 with 'ignore' error handling.

    Parameters
    ----------
    path : str or pathlib.Path
        Filesystem path to the EDI file.

    Returns
    -------
    str
        Full file content as a single string.
    """
    return Path(path).read_text(encoding="latin-1", errors="ignore")


def _iter_blocks(edi_text: str) -> Generator[Tuple[str, str], None, None]:
    """Yield (tag, block_text) pairs for lines starting with '>'.

    Parameters
    ----------
    edi_text : str
        Full EDI content.

    Yields
    ------
    tuple
        (tag, block_text) where tag is the block identifier (for example
        'FREQ', 'ZXXR', 'SPECTRA') and block_text is all lines up to (but not
        including) the next '>' or end of file.
    """
    lines = edi_text.splitlines()
    current_tag: Optional[str] = None
    buf: list[str] = []
    for line in lines:
        if line.startswith(">"):
            if current_tag is not None:
                yield current_tag, "\n".join(buf) + "\n"
            parts = line[1:].split()
            current_tag = parts[0] if parts else ""
            buf = []
        else:
            if current_tag is not None:
                buf.append(line)
    if current_tag is not None:
        yield current_tag, "\n".join(buf) + "\n"


# ----------------------------------------------------------------------
# Phoenix SPECTRA handling
# ----------------------------------------------------------------------


def parse_spectra_blocks(
    edi_text: str,
) -> Generator[Tuple[float, float, np.ndarray], None, None]:
    """Yield Phoenix >SPECTRA blocks as (freq_Hz, avgt, mat7x7_real).

    This parser follows the conventions of Phoenix MT SPECTRA sections, but is
    written to be tolerant with respect to formatting. In particular:

    - It looks for FREQ and AVGT metadata anywhere inside each SPECTRA block.
    - It ignores any lines starting with '>' when parsing the 7x7 numeric
      matrix.
    - It accepts both E and H channels in their instrument units. Time series
      units are handled later in :func:`ZT_from_S`.

    Parameters
    ----------
    edi_text : str
        Entire EDI file content.

    Yields
    ------
    tuple
        (f, avgt, mat7) where

        f : float
            Frequency in Hz.
        avgt : float
            Averaging time in seconds if present (NaN if not present).
        mat7 : numpy.ndarray
            Real-valued (7, 7) array with autos on the diagonal, lower triangle
            containing real parts, and upper triangle containing imaginary
            parts in the Phoenix encoding.
    """
    lines = edi_text.splitlines()
    n_lines = len(lines)
    i = 0
    num_pattern = r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[EeDd][+\-]?\d+)?"

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

        num_strings: list[str] = []
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
        yield f, avgt, mat7


def reconstruct_S_phoenix(mat7: np.ndarray) -> np.ndarray:
    """Reconstruct complex 7x7 spectra matrix from Phoenix real-valued encoding.

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
        Complex 7x7 spectra matrix as returned by :func:`reconstruct_S_phoenix`.
    ref : {'RH', 'LH'}, optional
        Reference convention for impedance tensor handedness. Default is 'RH'.

    Returns
    -------
    Z : numpy.ndarray
        Complex impedance tensor of shape (2, 2).
    T : numpy.ndarray
        Complex tipper of shape (1, 2).
    """
    # Channel indices for the standard Phoenix ordering:
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

    # Phoenix spectra typically give Z in (microvolt/m)/nT.
    # That is 1e3 times larger than the SI impedance (V/m)/T,
    # so correct by dividing by 1e3.
    Z /= 1.0e3

    if ref.upper() == "LH":
        Z = -Z
        T = -T

    return Z, T


# ----------------------------------------------------------------------
# Classical Z/T table parsing
# ----------------------------------------------------------------------


def parse_block_values(
    edi_text: str,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Parse standard EDI value blocks into (freqs, Z, T).

    This parser supports two common EDI variants:

    1. Classical style with one line per frequency::

           FREQ = 1.234E+0
           FREQ = 5.678E-1
           ...

       In this case, the frequency list is built from all "FREQ =" matches.

    2. ModEM / GFZ style with a dedicated ">FREQ" block and "NFREQ=" metadata,
       for example::

           >FREQ NFREQ=28 ORDER=DEC //28
           9.42507e+2    2.28571e+2   ...

       Here, the frequency list is taken from the numeric values inside the
       ">FREQ" block.

    Impedance and tipper blocks are searched using flexible tags to handle
    slightly different conventions, such as::

        >ZXXR ROT=0.0 // 28
        >ZXXI ROT=0.0 // 28
        >TXR.EXP ROT=0.0 // 28
        >TXI.EXP ROT=0.0 // 28

    i.e. rotation info or ".EXP" suffixes are allowed after the tag.

    Parameters
    ----------
    edi_text : str
        Full EDI content as a single string.

    Returns
    -------
    tuple or None
        (freqs, Z, T) or None if no usable frequency / impedance information
        is found.
    """
    f_matches = re.findall(
        r"\bFREQ\s*=\s*([0-9.]+[ED][+\-]?\d+|[0-9.]+)",
        edi_text,
        flags=re.IGNORECASE,
    )
    if f_matches:
        freqs = np.array(
            [float(s.replace("D", "E")) for s in f_matches],
            dtype=float,
        )
    else:
        m_freq = re.search(
            r">FREQ[^\n]*\n"
            r"((?:[^\n]*\n)+?)(?=>[A-Z>]|>END|$)",
            edi_text,
            flags=re.IGNORECASE,
        )
        if not m_freq:
            return None
        num_strings = re.findall(
            r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[EeDd][+\-]?\d+)?",
            m_freq.group(1),
        )
        if not num_strings:
            return None
        freqs = np.array(
            [float(s.replace("D", "E")) for s in num_strings],
            dtype=float,
        )

    n = freqs.size

    def _get_arr_exact(tag: str) -> Optional[np.ndarray]:
        pat = (
            r">" + re.escape(tag) + r"[^\n]*\n"
            r"((?:[^\n]*\n)+?)(?=>[A-Z>]|>END|$)"
        )
        m = re.search(pat, edi_text, flags=re.IGNORECASE)
        if not m:
            return None
        nums = re.findall(
            r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[EeDd][+\-]?\d+)?",
            m.group(1),
        )
        if not nums:
            return None
        arr = np.array(
            [float(v.replace("D", "E")) for v in nums],
            dtype=float,
        )
        return arr[:n] if arr.size >= n else None

    def _get_arr_multi(*tags: str) -> Optional[np.ndarray]:
        for tg in tags:
            arr = _get_arr_exact(tg)
            if arr is not None:
                return arr
        return None

    Z = np.zeros((n, 2, 2), dtype=np.complex128)
    ok = False
    for comp, (i, j) in {
        "ZXX": (0, 0),
        "ZXY": (0, 1),
        "ZYX": (1, 0),
        "ZYY": (1, 1),
    }.items():
        re_arr = _get_arr_multi(
            comp + "R",
            comp + "R.EXP",
            comp + ".RE",
            comp + "_RE",
        )
        im_arr = _get_arr_multi(
            comp + "I",
            comp + "I.EXP",
            comp + ".IM",
            comp + "_IM",
        )
        if re_arr is not None and im_arr is not None:
            ok = True
            Z[:, i, j] = re_arr + 1j * im_arr

    if not ok:
        return None

    T = np.zeros((n, 1, 2), dtype=np.complex128)

    txr = _get_arr_multi("TXR", "TXR.EXP", "TX.RE", "TX_RE")
    txi = _get_arr_multi("TXI", "TXI.EXP", "TX.IM", "TX_IM")
    tyr = _get_arr_multi("TYR", "TYR.EXP", "TY.RE", "TY_RE")
    tyi = _get_arr_multi("TYI", "TYI.EXP", "TY.IM", "TY_IM")

    if txr is not None and txi is not None:
        T[:, 0, 0] = txr + 1j * txi
    if tyr is not None and tyi is not None:
        T[:, 0, 1] = tyr + 1j * tyi

    return freqs, Z, T


# ----------------------------------------------------------------------
# High-level loader
# ----------------------------------------------------------------------

def load_edi(
    path: Path | str,
    *,
    ref: str = "RH",
    rotate_deg: float = 0.0,
    fill_invalid: Optional[float] = None,
    drop_invalid_periods: bool = True,
    invalid_sentinel: float = 1.0e30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    """Load an EDI file, automatically choosing SPECTRA or table path.

    The decision is made purely based on the presence of Phoenix >SPECTRA
    blocks:

    - If any >SPECTRA blocks are found, the SPECTRA path is used to reconstruct
      Z and T (regardless of the value of ``prefer_spectra``).
    - If no >SPECTRA blocks are found, the classical Z/T table path is used.

    This reflects the practical convention that real-world EDI files contain
    *either* Phoenix SPECTRA *or* Z/T tables, but not both.

    Parameters
    ----------
    path : str or pathlib.Path
        EDI file path.
    prefer_spectra : bool, optional
        Kept for backward compatibility but effectively ignored: if SPECTRA
        blocks are present they are always used; otherwise tables are used.
    ref : str, optional
        Magnetic reference for SPECTRA path ("H" or "RH"). Default "RH".
    rotate_deg : float, optional
        Rotate Z/T by this angle in degrees (counter-clockwise). Default 0.0.
    fill_invalid : float or None, optional
        Replace non-finite values in Z/T with this constant. If None, leave as
        is. Default is None.
    drop_invalid_periods : bool, optional
        If True, remove any periods (rows) where Z or T contain invalid values
        before rotation / fill. Invalid values include NaN, inf and "sentinel"
        values with absolute magnitude greater than or equal to
        ``invalid_sentinel``. Default is False.
    invalid_sentinel : float, optional
        Threshold for detecting sentinel values (for example 1.0e30 or 1.0e32)
        used to mark invalid data. Default is 1.0e30.

    Returns
    -------
    freq : numpy.ndarray
        Frequencies (Hz), sorted in descending order.
    Z : numpy.ndarray
        Impedance tensor of shape (n, 2, 2).
    T : numpy.ndarray
        Tipper of shape (n, 1, 2).
    station : str
        Station name extracted from DATAID (or "UNKNOWN").
    source_kind : str
        Either "spectra" if Phoenix >SPECTRA blocks were used, or "tables"
        if classical Z/T tables were used.
    """
    text = read_edi_text(path)

    # Station name
    m = re.search(
        r'DATAID\s*=\s*"?([A-Za-z0-9_\-\.]+)"?',
        text,
        flags=re.IGNORECASE,
    )
    station = m.group(1) if m else "UNKNOWN"

    # Decide path purely based on presence of >SPECTRA blocks
    spectra_blocks = list(parse_spectra_blocks(text))
    if spectra_blocks:
        # --- Phoenix SPECTRA path -----------------------------------------
        spectra_blocks.sort(key=lambda x: x[0], reverse=True)
        n = len(spectra_blocks)
        freq = np.array([b[0] for b in spectra_blocks], dtype=float)
        Z = np.zeros((n, 2, 2), dtype=np.complex128)
        T = np.zeros((n, 1, 2), dtype=np.complex128)
        for k, (f, avgt, mat7) in enumerate(spectra_blocks):
            S = reconstruct_S_phoenix(mat7)
            Zk, Tk = ZT_from_S(S, ref=ref)
            Z[k] = Zk
            T[k] = Tk
        source_kind = "spectra"
    else:
        # --- Classical Z/T table path -------------------------------------
        parsed = parse_block_values(text)
        if parsed is None:
            # Helpful debug: show which block tags we actually found
            tags = re.findall(
                r"^>([A-Za-z0-9_.:-]+)",
                text,
                flags=re.MULTILINE,
            )
            tag_list = sorted(set(t.upper() for t in tags))
            raise RuntimeError(
                "Could not find SPECTRA blocks or standard Z/T tables in EDI. "
                f"Found block tags: {tag_list}"
            )
        freq, Z, T = parsed
        source_kind = "tables"

    # Optionally drop periods with invalid Z/T entries
    if drop_invalid_periods:
        invalid_Z = ~np.isfinite(Z) | (np.abs(Z) >= invalid_sentinel)
        invalid_T = ~np.isfinite(T) | (np.abs(T) >= invalid_sentinel)
        mask = ~(invalid_Z.any(axis=(1, 2)) | invalid_T.any(axis=(1, 2)))

        freq = freq[mask]
        Z = Z[mask, ...]
        T = T[mask, ...]

        if freq.size == 0:
            raise RuntimeError(
                "All periods were removed by drop_invalid_periods; "
                "no finite Z/T values remain."
            )

    # Rotation and invalid fill
    Z, T = _rotate_ZT(Z, T, rotate_deg)
    if fill_invalid is not None:
        _fill_invalid_inplace(Z, fill_invalid)
        _fill_invalid_inplace(T, fill_invalid)

    # Sort by descending frequency
    order = np.argsort(freq)[::-1]
    freq = freq[order]
    Z = Z[order]
    T = T[order]

    return freq, Z, T, station, source_kind


def compute_phase_tensor(Z: np.ndarray) -> np.ndarray:
    """Compute the 2x2 real phase tensor from a complex impedance tensor.

    The phase tensor Φ is defined as

        Φ = (Im Z) (Re Z)^{-1}

    where Z is the 2x2 impedance tensor for each period (Caldwell et al., 2004).

    Parameters
    ----------
    Z : numpy.ndarray
        Complex impedance tensor of shape (n, 2, 2).

    Returns
    -------
    numpy.ndarray
        Real-valued phase tensor array of shape (n, 2, 2).

    Notes
    -----
    A Moore–Penrose pseudo-inverse is used for Re(Z) at each period to avoid
    numerical problems for nearly singular 2x2 matrices.
    """
    Z = np.asarray(Z, dtype=np.complex128)
    if Z.ndim != 3 or Z.shape[1:] != (2, 2):
        raise ValueError("Z must have shape (n, 2, 2).")

    n = Z.shape[0]
    PT = np.zeros((n, 2, 2), dtype=float)
    for k in range(n):
        Zr = np.real(Z[k])
        Zi = np.imag(Z[k])
        # Robust inverse of Re(Z)
        Zr_inv = np.linalg.pinv(Zr)
        PT[k] = Zi @ Zr_inv

    return PT

# ----------------------------------------------------------------------
# DataFrame helper
# ----------------------------------------------------------------------


def dataframe_from_arrays(
    freq: np.ndarray,
    Z: np.ndarray,
    T: Optional[np.ndarray] = None,
    PT: Optional[np.ndarray] = None,
    station: Optional[str] = None,
) -> pd.DataFrame:
    """Build a pandas DataFrame with derived MT quantities from arrays.

    Parameters
    ----------
    freq : numpy.ndarray
        One-dimensional array of frequencies in Hz, shape (n,).
    Z : numpy.ndarray
        Impedance tensor of shape (n, 2, 2).
    T : numpy.ndarray, optional
        Tipper array of shape (n, 1, 2). If None, tipper columns are omitted.
    PT : numpy.ndarray, optional
        Phase tensor array of shape (n, 2, 2). If None, PT columns are omitted.
    station : str, optional
        Station name. If given, a 'station' column with this constant value is
        added to the DataFrame.

    Returns
    -------
    pandas.DataFrame
        Data frame with columns for frequency, period, Z components, apparent
        resistivities, phases, optional tipper, optional phase tensor, and an
        optional station column.
    """
    freq = np.asarray(freq, dtype=float).ravel()
    Z = np.asarray(Z, dtype=np.complex128)
    if Z.shape[-2:] != (2, 2):
        raise ValueError("Z must have shape (n, 2, 2).")

    n = freq.size
    if Z.shape[0] != n:
        raise ValueError("freq and Z must agree in their first dimension.")

    period = 1.0 / freq
    data: Dict[str, Any] = {
        "freq": freq,
        "period": period,
    }

    comps = {
        "xx": (0, 0),
        "xy": (0, 1),
        "yx": (1, 0),
        "yy": (1, 1),
    }

    omega = 2.0 * np.pi * freq
    for name, (i, j) in comps.items():
        Zij = Z[:, i, j]
        data[f"Z{name}_re"] = np.real(Zij)
        data[f"Z{name}_im"] = np.imag(Zij)
        absZ2 = np.abs(Zij) ** 2
        rho = absZ2 / (MU0 * omega)
        phi = np.rad2deg(np.arctan2(np.imag(Zij), np.real(Zij)))
        data[f"rho_{name}"] = rho
        data[f"phi_{name}"] = phi

    if T is not None:
        T = np.asarray(T, dtype=np.complex128)
        if T.shape != (n, 1, 2):
            raise ValueError("T must have shape (n, 1, 2).")
        data["Tx_re"] = np.real(T[:, 0, 0])
        data["Tx_im"] = np.imag(T[:, 0, 0])
        data["Ty_re"] = np.real(T[:, 0, 1])
        data["Ty_im"] = np.imag(T[:, 0, 1])

    if PT is not None:
        PT = np.asarray(PT, dtype=float)
        if PT.shape != (n, 2, 2):
            raise ValueError("PT must have shape (n, 2, 2).")
        data["ptxx_re"] = PT[:, 0, 0]
        data["ptxy_re"] = PT[:, 0, 1]
        data["ptyx_re"] = PT[:, 1, 0]
        data["ptyy_re"] = PT[:, 1, 1]

    if station is not None:
        data["station"] = np.full(n, station, dtype=object)

    df = pd.DataFrame(data)
    return df


# ----------------------------------------------------------------------
# EDI writer
# ----------------------------------------------------------------------


def _format_values(values: np.ndarray, numbers_per_line: int = 6) -> str:
    """Format a 1-D array of floats into wrapped lines.

    Parameters
    ----------
    values : numpy.ndarray
        One-dimensional array of floats.
    numbers_per_line : int, optional
        Maximum number of values per output line. Default is 6.

    Returns
    -------
    str
        Multi-line string with formatted numbers.
    """
    buf = io.StringIO()
    n = values.size
    for i in range(0, n, numbers_per_line):
        chunk = values[i:i + numbers_per_line]
        buf.write(" ".join(f"{v: .6e}" for v in chunk))
        buf.write("\n")
    return buf.getvalue()

def write_edi(
    path: Path | str,
    *,
    station: str,
    freq: np.ndarray,
    Z: np.ndarray,
    T: Optional[np.ndarray] = None,
    PT: Optional[np.ndarray] = None,
    lat_deg: Optional[float] = None,
    lon_deg: Optional[float] = None,
    elev_m: Optional[float] = None,
    header_meta: Optional[Dict[str, str]] = None,
    numbers_per_line: int = 6,
) -> str:
    """Write a simple EDI file from arrays, including optional PT blocks."""
    path = Path(path)
    freq = np.asarray(freq, dtype=float).ravel()
    n = freq.size

    Z = np.asarray(Z, dtype=np.complex128)
    if Z.shape != (n, 2, 2):
        raise ValueError("Z must have shape (n, 2, 2).")

    meta: Dict[str, str] = dict(header_meta) if header_meta is not None else {}
    meta.setdefault("DATAID", station)
    meta.setdefault("ACQBY", "UNKNOWN")
    meta.setdefault("FILEBY", "ediproc.py")

    with path.open("w", encoding="latin-1") as f:
        f.write(">HEAD\n")
        f.write("DATAID={0}\n".format(station))
        if lat_deg is not None and lon_deg is not None:
            f.write("LAT={0: .6f}\n".format(lat_deg))
            f.write("LON={0: .6f}\n".format(lon_deg))
        if elev_m is not None:
            f.write("ELEV={0: .2f}\n".format(elev_m))
        for k, v in meta.items():
            if k.upper() in {"DATAID", "LAT", "LON", "ELEV"}:
                continue
            f.write("{0}={1}\n".format(k, v))
        f.write(">END\n\n")

        f.write(">FREQ NFREQ={0:d}\n".format(n))
        f.write(_format_values(freq, numbers_per_line))
        f.write(">END\n\n")

        comps = {
            "ZXX": (0, 0),
            "ZXY": (0, 1),
            "ZYX": (1, 0),
            "ZYY": (1, 1),
        }
        for name, (i, j) in comps.items():
            re_vals = np.real(Z[:, i, j])
            im_vals = np.imag(Z[:, i, j])

            f.write(">{0}R\n".format(name))
            f.write(_format_values(re_vals, numbers_per_line))
            f.write(">END\n\n")

            f.write(">{0}I\n".format(name))
            f.write(_format_values(im_vals, numbers_per_line))
            f.write(">END\n\n")

        if T is not None:
            T = np.asarray(T, dtype=np.complex128)
            if T.shape != (n, 1, 2):
                raise ValueError("T must have shape (n, 1, 2).")
            tx_re = np.real(T[:, 0, 0])
            tx_im = np.imag(T[:, 0, 0])
            ty_re = np.real(T[:, 0, 1])
            ty_im = np.imag(T[:, 0, 1])

            f.write(">TXR\n")
            f.write(_format_values(tx_re, numbers_per_line))
            f.write(">END\n\n")

            f.write(">TXI\n")
            f.write(_format_values(tx_im, numbers_per_line))
            f.write(">END\n\n")

            f.write(">TYR\n")
            f.write(_format_values(ty_re, numbers_per_line))
            f.write(">END\n\n")

            f.write(">TYI\n")
            f.write(_format_values(ty_im, numbers_per_line))
            f.write(">END\n\n")

        if PT is not None:
            PT = np.asarray(PT, dtype=float)
            if PT.shape != (n, 2, 2):
                raise ValueError("PT must have shape (n, 2, 2).")

            pt_comps = {
                "PTXX": (0, 0),
                "PTXY": (0, 1),
                "PTYX": (1, 0),
                "PTYY": (1, 1),
            }
            for name, (i, j) in pt_comps.items():
                vals = PT[:, i, j]
                f.write(">{0}\n".format(name))
                f.write(_format_values(vals, numbers_per_line))
                f.write(">END\n\n")

    return str(path)

# ----------------------------------------------------------------------
# High-level "load, clean, save" helper
# ----------------------------------------------------------------------


def save_edi(
    edi_path: Path | str,
    out_path: Optional[Path | str] = None,
    *,
    prefer_spectra: bool = True,
    ref: str = "RH",
    rotate_deg: float = 0.0,
    fill_invalid: Optional[float] = None,
    drop_invalid_periods: bool = False,
    invalid_sentinel: float = 1.0e30,
    numbers_per_line: int = 6,
    lat_deg: Optional[float] = None,
    lon_deg: Optional[float] = None,
    elev_m: Optional[float] = None,
    header_meta: Optional[Dict[str, str]] = None,
    add_pt_blocks: bool = False,          # <- NEW
) -> str:

    """Load an EDI file, optionally clean/rotate it, and write a new EDI file.

    This is a high-level convenience wrapper around :func:`load_edi` and
    :func:`write_edi`. It is intended for workflows where an existing EDI is
    read, optionally filtered for invalid periods, rotated, and then written
    back to disk as a new EDI file.

    Parameters
    ----------
    edi_path : str or pathlib.Path
        Path of the input EDI file to read.
    out_path : str or pathlib.Path, optional
        Output EDI file path. If None, a new file with suffix ".proc.edi" is
        created next to ``edi_path``.
    prefer_spectra : bool, optional
        Forwarded to :func:`load_edi`. If True (default), Phoenix SPECTRA
        blocks are preferred when present.
    ref : {'RH', 'LH'}, optional
        Handedness convention for impedance / tipper; forwarded to
        :func:`load_edi`.
    rotate_deg : float, optional
        Rotation angle in degrees; forwarded to :func:`load_edi`.
    fill_invalid : float or None, optional
        Replacement value for non-finite entries in Z/T; forwarded to
        :func:`load_edi`.
    drop_invalid_periods : bool, optional
        If True, rows (periods) for which Z or T contain invalid entries are
        removed before writing; forwarded to :func:`load_edi`.
    invalid_sentinel : float, optional
        Sentinel threshold for marking invalid values; forwarded to
        :func:`load_edi`.
    numbers_per_line : int, optional
        Number of values per line in the output EDI value blocks.
    lat_deg, lon_deg, elev_m : float, optional
        Optional location metadata written into the HEAD block.
    header_meta : dict, optional
        Additional key-value pairs written into the HEAD block. These are
        merged with automatically generated provenance keys; user-supplied
        keys take precedence.

    Returns
    -------
    str
        Path to the written EDI file.
    """
    edi_path = Path(edi_path)

    freq, Z, T, station, source_kind = load_edi(
        edi_path,
        ref=ref,
        rotate_deg=rotate_deg,
        fill_invalid=fill_invalid,
        drop_invalid_periods=drop_invalid_periods,
        invalid_sentinel=invalid_sentinel,
    )
    # Optionally compute phase tensor for writing PT blocks.
    PT = compute_phase_tensor(Z) if add_pt_blocks else None

    if out_path is None:
        out_path = edi_path.with_suffix(".proc.edi")
    out_path = Path(out_path)

    meta: Dict[str, str] = dict(header_meta) if header_meta is not None else {}
    auto_meta = {
        "SOURCE_KIND": str(source_kind),
        "REF": str(ref),
        "ROTATE_DEG": f"{rotate_deg:.3f}",
        "PREFER_SPECTRA": str(prefer_spectra),
        "FILL_INVALID": "None" if fill_invalid is None else str(fill_invalid),
        "DROP_INVALID_PERIODS": str(drop_invalid_periods),
        "INVALID_SENTINEL": f"{invalid_sentinel:.3e}",
        "ORIGIN_EDI": edi_path.name,
    }
    for key, value in auto_meta.items():
        meta.setdefault(key, value)


    return write_edi(
           out_path,
           station=station,
           freq=freq,
           Z=Z,
           T=T,
           PT=PT,
           lat_deg=lat_deg,
           lon_deg=lon_deg,
           elev_m=elev_m,
           header_meta=meta,
           numbers_per_line=numbers_per_line,
       )


# ----------------------------------------------------------------------
# NPZ-based writer
# ----------------------------------------------------------------------


def write_edi_from_npz(
    npz_path: Path | str,
    out_path: Optional[Path | str] = None,
    *,
    numbers_per_line: int = 6,
    lat_deg: Optional[float] = None,
    lon_deg: Optional[float] = None,
    elev_m: Optional[float] = None,
) -> str:
    """Write an EDI file from an NPZ bundle.

    The NPZ bundle is expected to contain at least:

    - "freq": (n,) float array of frequencies in Hz.
    - "Z": (n, 2, 2) complex array of impedance values.

    Optionally it may also contain:

    - "T": (n, 1, 2) complex array of tipper values.
    - "station": station name (string or array containing a single string).
    - "lat_deg", "lon_deg", "elev_m": location metadata.

    Parameters
    ----------
    npz_path : str or pathlib.Path
        Path to the input NPZ file.
    out_path : str or pathlib.Path, optional
        Output EDI file path. If None, the suffix is changed to ".edi".
    numbers_per_line : int, optional
        Number of values per line in the numeric blocks.
    lat_deg, lon_deg, elev_m : float, optional
        Optional location metadata. If not given, and the NPZ contains these
        keys, the NPZ values are used.

    Returns
    -------
    str
        Path to the written EDI file.
    """
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=True) as data:
        freq = data["freq"]
        Z = data["Z"]
        T = data["T"] if "T" in data else None
        PT = data["PT"] if "PT" in data else None   # <- NEW
        if "station" in data:
            station_val = data["station"]
            if np.ndim(station_val) == 0:
                station = str(station_val)
            else:
                station = str(station_val[0])
        else:
            station = npz_path.stem

        if lat_deg is None and "lat_deg" in data:
            lat_deg = float(data["lat_deg"])
        if lon_deg is None and "lon_deg" in data:
            lon_deg = float(data["lon_deg"])
        if elev_m is None and "elev_m" in data:
            elev_m = float(data["elev_m"])

    if out_path is None:
        out_path = npz_path.with_suffix(".edi")

    return write_edi(
        out_path,
        station=station,
        freq=freq,
        Z=Z,
        T=T,
        PT=PT,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        elev_m=elev_m,
        header_meta=None,
        numbers_per_line=numbers_per_line,
    )
