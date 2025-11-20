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
- load_edi(path, *, ref='RH', rotate_deg=0.0,
           fill_invalid=None, drop_invalid_periods=True,
           invalid_sentinel=1.0e30)
- dataframe_from_arrays(freq, Z, T=None, PT=None, station=None)
- write_edi(path, *, station, freq, Z, T=None, PT=None,
            lat_deg=None, lon_deg=None, elev_m=None,
            header_meta=None, numbers_per_line=6,
            lon_keyword='LON')
- save_edi(out_path, freq, Z, T, station, source_kind='',
           PT=None, numbers_per_line=6,
           lat_deg=None, lon_deg=None, elev_m=None,
           header_meta=None, add_pt_blocks=False,
           lon_keyword='LON')

Notes
-----
- The Phoenix SPECTRA path is used whenever ">SPECTRA" blocks are present.
  For files like PONT.edi, which are pure table-style EDIs, only the table
  path is used.
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
    "write_edi",
    "save_edi",
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
# ----------------------------------------------------------------------
# Location parsing helpers
# ----------------------------------------------------------------------
def _parse_angle(value: str) -> Optional[float]:
    """Parse a latitude/longitude string into decimal degrees (DD).

    The parser accepts:
    - Plain decimal degrees: ``"47.1234"``
    - DMD/DMS style: ``"47:30.0"``, ``"47:30:15"``, or space-separated
      equivalents.
    - Optional N/S/E/W suffixes and inline comments after ``//``, ``!``,
      ``#``, or ``;``.

    Parameters
    ----------
    value : str
        Raw header value after the ``=``.

    Returns
    -------
    float or None
        Angle in decimal degrees (DD), or None if parsing fails.
    """
    if value is None:
        return None

    txt = str(value).strip()
    if not txt:
        return None

    # Strip inline comments
    for sep in ("//", "!", "#", ";"):
        if sep in txt:
            txt = txt.split(sep, 1)[0].strip()

    if not txt:
        return None

    # Determine sign from N/S/E/W if present
    sign = 1.0
    upper = txt.upper()
    if "S" in upper or "W" in upper:
        sign = -1.0

    # Extract numeric fields (deg, minutes, seconds)
    nums = re.findall(r"[-+]?\d+(?:\.\d*)?", txt)
    if not nums:
        return None

    if len(nums) == 1:
        # Already decimal degrees
        val = float(nums[0])
        if sign < 0 and val > 0:
            val = -val
        return val

    # DMD / DMS: deg, minutes, optional seconds
    deg = float(nums[0])
    minutes = float(nums[1])
    seconds = float(nums[2]) if len(nums) >= 3 else 0.0

    abs_deg = abs(deg) + minutes / 60.0 + seconds / 3600.0

    # If degrees already negative, that dominates
    if deg < 0:
        sign = -1.0

    return sign * abs_deg


def _parse_location_meta(
    edi_text: str,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Extract LAT, LON/LONG and ELEV from an EDI header as floats.

    Latitude and longitude are converted to decimal degrees (DD), even if
    the original header uses DMD/DMS formats such as ``47:30.25`` or
    ``47:30:15``.

    Parameters
    ----------
    edi_text : str
        Full EDI file content.

    Returns
    -------
    lat_deg : float or None
        Latitude in decimal degrees, or None if not present.
    lon_deg : float or None
        Longitude in decimal degrees, or None if not present. Values from
        both ``LON`` and ``LONG`` are recognised.
    elev_m : float or None
        Elevation in metres, or None if not present.
    """
    lat_match = re.search(
        r"^LAT\s*=\s*([^\n\r]+)",
        edi_text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    lon_match = re.search(
        r"^(?:LON|LONG)\s*=\s*([^\n\r]+)",
        edi_text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    elev_match = re.search(
        r"^ELEV\s*=\s*([^\n\r]+)",
        edi_text,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    lat_deg = _parse_angle(lat_match.group(1)) if lat_match else None
    lon_deg = _parse_angle(lon_match.group(1)) if lon_match else None

    if elev_match:
        txt = elev_match.group(1)
        for sep in ("//", "!", "#", ";"):
            if sep in txt:
                txt = txt.split(sep, 1)[0].strip()
        mnum = re.search(
            r"[-+]?\d+(?:\.\d*)?(?:[Ee][+\-]?\d+)?",
            txt,
        )
        elev_m = float(mnum.group(0)) if mnum else None
    else:
        elev_m = None

    return lat_deg, lon_deg, elev_m


def load_edi(
    path: Path | str,
    *,
    ref: str = "RH",
    rotate_deg: float = 0.0,
    fill_invalid: Optional[float] = None,
    drop_invalid_periods: bool = True,
    invalid_sentinel: float = 1.0e30,
) -> Dict[str, Any]:
    """
    Load an EDI file and return a dictionary of arrays and metadata.

    The decision between Phoenix SPECTRA and classical Z/T tables is made
    purely based on the presence of Phoenix >SPECTRA blocks:

    - If any >SPECTRA blocks are found, the SPECTRA path is used to reconstruct
      Z and T.
    - If no >SPECTRA blocks are found, the classical Z/T table path is used.

    This reflects the practical convention that real-world EDI files contain
    *either* Phoenix SPECTRA *or* Z/T tables, but not both.

    Parameters
    ----------
    path : str or pathlib.Path
        EDI file path.
    ref : {'RH', 'LH'}, optional
        Magnetic reference for the SPECTRA path. Default "RH".
    rotate_deg : float, optional
        Rotate Z/T by this angle in degrees (counter-clockwise). Default 0.0.
    fill_invalid : float or None, optional
        Replace non-finite values in Z/T with this constant. If None, leave as
        is. Default is None.
    drop_invalid_periods : bool, optional
        If True, remove any periods (rows) where Z or T contain invalid values
        before rotation / fill. Invalid values include NaN, inf and "sentinel"
        values with absolute magnitude greater than or equal to
        ``invalid_sentinel``. Default is True.
    invalid_sentinel : float, optional
        Threshold for detecting sentinel values (for example 1.0e30 or 1.0e32)
        used to mark invalid data. Default is 1.0e30.

    Returns
    -------
    dict
        Dictionary with at least the following keys:

        - ``'freq'`` : (n,) float array of frequencies (Hz), descending.
        - ``'Z'`` : (n, 2, 2) complex impedance tensor.
        - ``'T'`` : (n, 1, 2) complex tipper.
        - ``'station'`` : station name (string).
        - ``'source_kind'`` : "spectra" or "tables".
        - ``'lat_deg'`` : latitude in decimal degrees, or None.
        - ``'lon_deg'`` : longitude in decimal degrees, or None.
        - ``'elev_m'`` : elevation in metres, or None.
        - ``'Z_err'`` : (n, 2, 2) float array of impedance errors/variances,
          or None if no error blocks were parsed.
        - ``'T_err'`` : (n, 1, 2) float array of tipper errors/variances,
          or None if no error blocks were parsed.
    """
    path = Path(path)
    text = read_edi_text(path)

    # Station name -----------------------------------------------------------
    m = re.search(
        r'^DATAID\s*=\s*"?([^\r\n"]+)"?',
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    station = m.group(1).strip() if m else path.stem

    # Location metadata (LAT, LON/LONG, ELEV) --------------------------------
    lat_deg, lon_deg, elev_m = _parse_location_meta(text)

    # Decide path purely based on presence of >SPECTRA blocks -----------------
    spectra_blocks = list(parse_spectra_blocks(text))
    if spectra_blocks:
        # --- Phoenix SPECTRA path -------------------------------------------
        spectra_blocks.sort(key=lambda x: x[0], reverse=True)
        n = len(spectra_blocks)
        freq = np.empty(n, dtype=float)
        Z = np.empty((n, 2, 2), dtype=np.complex128)
        T = np.empty((n, 1, 2), dtype=np.complex128)

        for k, (f, avgt, mat7) in enumerate(spectra_blocks):
            freq[k] = f
            S = reconstruct_S_phoenix(mat7)
            Zk, Tk = ZT_from_S(S, ref=ref)
            Z[k] = Zk
            T[k] = Tk

        source_kind = "spectra"
        Z_err = None
        T_err = None
    else:
        # --- Classical Z/T table path ---------------------------------------
        parsed = parse_block_values(text)
        if parsed is None:
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

        # Backwards-compatible unpacking if parse_block_values is older
        if len(parsed) == 3:
            freq, Z, T = parsed  # type: ignore[misc]
            Z_err = None
            T_err = None
        else:
            freq, Z, T, Z_err, T_err = parsed  # type: ignore[misc]

        source_kind = "tables"

    # Optionally drop periods with invalid Z/T entries ------------------------
    if drop_invalid_periods:
        invalid_Z = ~np.isfinite(Z) | (np.abs(Z) >= invalid_sentinel)
        invalid_T = ~np.isfinite(T) | (np.abs(T) >= invalid_sentinel)

        mask = ~(invalid_Z.any(axis=(1, 2)) | invalid_T.any(axis=(1, 2)))

        freq = freq[mask]
        Z = Z[mask, ...]
        T = T[mask, ...]

        if Z_err is not None:
            Z_err = Z_err[mask, ...]
        if T_err is not None:
            T_err = T_err[mask, ...]

        if freq.size == 0:
            raise RuntimeError(
                "All periods were removed by drop_invalid_periods; "
                "no finite Z/T values remain."
            )

    # Rotation and invalid fill ----------------------------------------------
    Z, T = _rotate_ZT(Z, T, rotate_deg)
    if fill_invalid is not None:
        _fill_invalid_inplace(Z, fill_invalid)
        _fill_invalid_inplace(T, fill_invalid)

    # Sort by descending frequency -------------------------------------------
    order = np.argsort(freq)[::-1]
    freq = freq[order]
    Z = Z[order]
    T = T[order]
    if Z_err is not None:
        Z_err = Z_err[order]
    if T_err is not None:
        T_err = T_err[order]

    result: Dict[str, Any] = {
        "freq": freq,
        "Z": Z,
        "T": T,
        "station": station,
        "source_kind": source_kind,
        "lat_deg": lat_deg,
        "lon_deg": lon_deg,
        "elev_m": elev_m,
        "Z_err": Z_err,
        "T_err": T_err,
    }
    return result


def compute_pt(
    Z: np.ndarray,
    Z_err: Optional[np.ndarray] = None,
    *,
    err_kind: str = "std",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Compute phase tensor P and optionally propagate impedance errors.

    Parameters
    ----------
    Z : array_like, shape (n, 2, 2) or (2, 2)
        Complex impedance tensor Z = X + i Y for one or several periods.
    Z_err : array_like, shape (n, 2, 2) or (2, 2), optional
        Uncertainties of the complex impedance components. Depending on
        ``err_kind`` they are interpreted as standard deviations or variances
        of the complex impedance entries. If provided, uncertainties for the
        phase tensor are returned.
    err_kind : {'std', 'var'}, optional
        Interpretation of the values in ``Z_err``:
        - 'std': Z_err contains 1-sigma standard deviations.
        - 'var': Z_err contains variances.
        Default is 'std'.

    Returns
    -------
    PT : ndarray, shape (n, 2, 2) or (2, 2)
        Real phase tensor for each period, defined as P = X^{-1} Y with
        X = Re(Z), Y = Im(Z).
    PT_err : ndarray or None
        If ``Z_err`` is given, array of the same shape as PT containing
        propagated uncertainties for the phase tensor. If ``err_kind`` is
        'std', these are 1-sigma standard deviations; if 'var', they are
        variances. If ``Z_err`` is None, PT_err is None.

    Notes
    -----
    Phase tensor is computed as in Caldwell et al. (2004):

        Z = X + i Y,    P = X^{-1} Y

    Error propagation uses the linear approximation

        δP = -X^{-1} (δX) X^{-1} Y + X^{-1} (δY),

    assuming that:
    - different impedance components are uncorrelated, and
    - real and imaginary parts of a given component share the same variance.

    Under these assumptions the variance of P_ij is

        Var(P_ij) = sum_ab [
            (∂P_ij/∂X_ab)^2 Var(X_ab) + (∂P_ij/∂Y_ab)^2 Var(Y_ab)
        ],

    where ∂P/∂X_ab and ∂P/∂Y_ab are evaluated from the expression above.

    In practice we take Var(X_ab) = Var(Y_ab) equal to the variance implied
    by ``Z_err[a,b]`` for that component.
    """
    Z = np.asarray(Z, dtype=np.complex128)
    single = False
    if Z.shape == (2, 2):
        Z = Z[None, ...]
        single = True
    if Z.ndim != 3 or Z.shape[1:] != (2, 2):
        raise ValueError("Z must have shape (n, 2, 2) or (2, 2).")

    n = Z.shape[0]
    PT = np.empty((n, 2, 2), dtype=float)
    PT_err: Optional[np.ndarray] = None

    if Z_err is not None:
        Z_err = np.asarray(Z_err, dtype=float)
        if Z_err.shape == (2, 2):
            Z_err = Z_err[None, ...]
        if Z_err.shape != (n, 2, 2):
            raise ValueError("Z_err must have shape (n, 2, 2) or (2, 2) matching Z.")
        PT_err = np.full((n, 2, 2), np.nan, dtype=float)
        if err_kind not in {"std", "var"}:
            raise ValueError("err_kind must be 'std' or 'var'.")
        as_var = err_kind == "var"
    else:
        as_var = False  # unused, just to keep the name defined

    for k in range(n):
        Zk = Z[k]
        X = Zk.real
        Y = Zk.imag

        # Invert X for the phase tensor
        try:
            X_inv = np.linalg.inv(X)
        except np.linalg.LinAlgError:
            PT[k, :, :] = np.nan
            if PT_err is not None:
                PT_err[k, :, :] = np.nan
            continue

        Pk = X_inv @ Y
        PT[k, :, :] = Pk

        if PT_err is not None:
            var_P = np.zeros((2, 2), dtype=float)

            for a in range(2):
                for b in range(2):
                    val = Z_err[k, a, b]
                    if not np.isfinite(val):
                        continue

                    # Interpret error value as std or variance
                    var_val = val if as_var else val ** 2

                    # Unit matrix for the (a, b) perturbation
                    E = np.zeros((2, 2), dtype=float)
                    E[a, b] = 1.0

                    # Derivatives w.r.t. X_ab and Y_ab
                    dP_dX = -X_inv @ E @ X_inv @ Y
                    dP_dY = X_inv @ E

                    dP_dX = dP_dX.real
                    dP_dY = dP_dY.real

                    # Assume same variance for X_ab and Y_ab, add contributions
                    var_P += (dP_dX ** 2 + dP_dY ** 2) * var_val

            PT_err[k, :, :] = np.sqrt(var_P) if not as_var else var_P

    if single:
        PT = PT[0]
        if PT_err is not None:
            PT_err = PT_err[0]

    return PT, PT_err


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
    Z_err: Optional[np.ndarray] = None,
    T_err: Optional[np.ndarray] = None,
    PT_err: Optional[np.ndarray] = None,
    lat_deg: Optional[float] = None,
    lon_deg: Optional[float] = None,
    elev_m: Optional[float] = None,
    header_meta: Optional[Dict[str, str]] = None,
    numbers_per_line: int = 6,
    lon_keyword: str = "LON",
) -> str:
    """Low-level EDI writer.

    Parameters are as in :func:`save_edi`, but here PT_err and Z_err/T_err
    all directly control optional *.VAR blocks for Z, T and PT.
    """
    path = Path(path)
    freq = np.asarray(freq, dtype=float).ravel()
    Z = np.asarray(Z, dtype=np.complex128)
    n = freq.size

    if Z.shape != (n, 2, 2):
        raise ValueError("Z must have shape (n, 2, 2) and match freq.size.")

    if T is not None:
        T = np.asarray(T, dtype=np.complex128)
        if T.shape != (n, 1, 2):
            raise ValueError("T must have shape (n, 1, 2) and match freq.size.")

    if PT is not None:
        PT = np.asarray(PT, dtype=float)
        if PT.shape != (n, 2, 2):
            raise ValueError("PT must have shape (n, 2, 2) and match freq.size.")

    if Z_err is not None:
        Z_err = np.asarray(Z_err, dtype=float)
        if Z_err.shape != (n, 2, 2):
            raise ValueError("Z_err must have shape (n, 2, 2) and match freq.size.")

    if T_err is not None:
        T_err = np.asarray(T_err, dtype=float)
        if T_err.shape != (n, 1, 2):
            raise ValueError("T_err must have shape (n, 1, 2) and match freq.size.")

    if PT_err is not None:
        PT_err = np.asarray(PT_err, dtype=float)
        if PT_err.shape != (n, 2, 2):
            raise ValueError("PT_err must have shape (n, 2, 2) and match freq.size.")

    lines: list[str] = []

    # ------------------------------------------------------------------ HEAD
    # (use your existing header construction; this is schematic)
    lines.append(">HEAD")
    lines.append(f"   DATAID= \"{station}\"")
    if lat_deg is not None:
        lines.append(f"   LAT={lat_deg:12.6f}")
    if lon_deg is not None:
        lines.append(f"   {lon_keyword}={lon_deg:12.6f}")
    if elev_m is not None:
        lines.append(f"   ELEV={elev_m:12.2f}")
    if header_meta:
        for k, v in header_meta.items():
            lines.append(f"   {k}={v}")
    lines.append(">END")

    # ------------------------------------------------------------------ FREQ
    lines.append(">FREQ")
    lines.extend(_format_values(freq, numbers_per_line))
    lines.append(">END")

    # ------------------------------------------------------------------ Z blocks (R/I)
    comp_map = {
        "ZXX": (0, 0),
        "ZXY": (0, 1),
        "ZYX": (1, 0),
        "ZYY": (1, 1),
    }
    for name, (i, j) in comp_map.items():
        zr = Z[:, i, j].real
        zi = Z[:, i, j].imag

        lines.append(f">{name}R")
        lines.extend(_format_values(zr, numbers_per_line))
        lines.append(">END")

        lines.append(f">{name}I")
        lines.extend(_format_values(zi, numbers_per_line))
        lines.append(">END")

    # ------------------------------------------------------------------ Z error blocks
    if Z_err is not None:
        for name, (i, j) in comp_map.items():
            err = Z_err[:, i, j]
            if not np.isfinite(err).any():
                continue
            lines.append(f">{name}.VAR")
            lines.extend(_format_values(err, numbers_per_line))
            lines.append(">END")

    # ------------------------------------------------------------------ T blocks (R/I)
    if T is not None:
        txr = T[:, 0, 0].real
        txi = T[:, 0, 0].imag
        tyr = T[:, 0, 1].real
        tyi = T[:, 0, 1].imag

        lines.append(">TXR")
        lines.extend(_format_values(txr, numbers_per_line))
        lines.append(">END")

        lines.append(">TXI")
        lines.extend(_format_values(txi, numbers_per_line))
        lines.append(">END")

        lines.append(">TYR")
        lines.extend(_format_values(tyr, numbers_per_line))
        lines.append(">END")

        lines.append(">TYI")
        lines.extend(_format_values(tyi, numbers_per_line))
        lines.append(">END")

    # ------------------------------------------------------------------ T error blocks
    if T_err is not None:
        txe = T_err[:, 0, 0]
        tye = T_err[:, 0, 1]

        if np.isfinite(txe).any():
            lines.append(">TX.VAR")
            lines.extend(_format_values(txe, numbers_per_line))
            lines.append(">END")

        if np.isfinite(tye).any():
            lines.append(">TY.VAR")
            lines.extend(_format_values(tye, numbers_per_line))
            lines.append(">END")

    # ------------------------------------------------------------------ PT blocks
    if PT is not None:
        pt_map = {
            "PTXX": (0, 0),
            "PTXY": (0, 1),
            "PTYX": (1, 0),
            "PTYY": (1, 1),
        }
        for name, (i, j) in pt_map.items():
            vals = PT[:, i, j]
            lines.append(f">{name}")
            lines.extend(_format_values(vals, numbers_per_line))
            lines.append(">END")

        # -------------------------------------------------------------- PT error blocks
        if PT_err is not None:
            for name, (i, j) in pt_map.items():
                err = PT_err[:, i, j]
                if not np.isfinite(err).any():
                    continue
                lines.append(f">{name}.VAR")
                lines.extend(_format_values(err, numbers_per_line))
                lines.append(">END")

    # ------------------------------------------------------------------ final write
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="latin-1")
    return str(path)


def save_edi(
    out_path: Path | str,
    edi: Dict[str, Any],
    *,
    PT: Optional[np.ndarray] = None,
    PT_err: Optional[np.ndarray] = None,
    numbers_per_line: int = 6,
    header_meta: Optional[Dict[str, str]] = None,
    add_pt_blocks: bool = False,
    lon_keyword: str = "LON",
    pt_err_kind: str = "std",
) -> str:
    """Write a processed EDI file from a dictionary (e.g. from :func:`load_edi`).

    The dictionary is expected to contain at least:

    - ``'freq'`` : (n,) float array of frequencies in Hz.
    - ``'Z'`` : (n, 2, 2) complex impedance tensor.
    - ``'T'`` : (n, 1, 2) complex tipper (optional but recommended).
    - ``'station'`` : station name (string).
    - ``'source_kind'`` : "spectra" or "tables" (optional).
    - ``'lat_deg'``, ``'lon_deg'``, ``'elev_m'`` : location metadata (optional).

    Phase tensor information can be supplied or computed:

    - ``PT`` argument and ``PT_err`` argument (preferred),
    - or ``edi['PT']`` / ``edi['PT_err']`` entries,
    - or, if ``add_pt_blocks`` is True, PT (and PT_err if Z_err is present)
      are computed from Z (and Z_err) via :func:`compute_pt`.

    Error arrays (``'Z_err'`` and ``'T_err'``) in the dict are passed on to
    :func:`write_edi`, which writes standard ``*.VAR`` blocks for Z and T.
    Phase tensor errors are written as PT??.VAR blocks if available.

    Parameters
    ----------
    out_path : str or pathlib.Path
        Output EDI file path.
    edi : dict
        Dictionary with EDI data and metadata, typically returned by
        :func:`load_edi`.
    PT : numpy.ndarray, optional
        Phase tensor of shape (n, 2, 2). If provided, PT blocks (PTXX, PTXY,
        PTYX, PTYY) are written to the EDI.
    PT_err : numpy.ndarray, optional
        Uncertainties (std or var, see pt_err_kind) for the phase tensor,
        shape (n, 2, 2). If present, dedicated PTXX.VAR, PTXY.VAR, PTYX.VAR
        and PTYY.VAR blocks are written.
    numbers_per_line : int, optional
        Maximum number of values per numeric line in the output blocks.
    header_meta : dict, optional
        Additional key-value pairs for the HEAD block. These are merged with
        any "header_meta" entry present in ``edi`` (if any), with this
        parameter taking precedence on conflicts.
    add_pt_blocks : bool, optional
        If True and PT is None and no PT is stored in ``edi``, compute the
        phase tensor (and optionally its errors) from Z using
        :func:`compute_pt`.
    lon_keyword : {"LON", "LONG"}, optional
        Header keyword used for longitude in the output EDI. Default "LON".
    pt_err_kind : {"std", "var"}, optional
        Interpretation of the numbers in PT_err (and PT_err computed from
        Z_err via :func:`compute_pt`):
        - "std" : PT_err is 1-sigma standard deviation.
        - "var" : PT_err is variance.
        This is passed through unchanged; it is only informational here.

    Returns
    -------
    str
        Path to the written EDI file.
    """
    out_path = Path(out_path)

    # Extract core fields from dictionary ------------------------------------
    freq = np.asarray(edi["freq"], dtype=float).ravel()
    Z = np.asarray(edi["Z"], dtype=np.complex128)
    T = edi.get("T")

    station = str(edi.get("station", "UNKNOWN"))
    source_kind = str(edi.get("source_kind", ""))

    lat_deg = edi.get("lat_deg")
    lon_deg = edi.get("lon_deg")
    elev_m = edi.get("elev_m")

    Z_err = edi.get("Z_err")
    T_err = edi.get("T_err")

    n = freq.size
    if Z.shape != (n, 2, 2):
        raise ValueError("edi['Z'] must have shape (n, 2, 2) and match freq.size.")

    if T is not None:
        T = np.asarray(T, dtype=np.complex128)
        if T.shape != (n, 1, 2):
            raise ValueError("edi['T'] must have shape (n, 1, 2) and match freq.size.")

    # Phase tensor handling --------------------------------------------------
    # Priority order:
    #   1. PT argument / PT_err argument
    #   2. edi["PT"] / edi["PT_err"]
    #   3. compute_pt(Z, Z_err) if add_pt_blocks is True
    if PT is not None:
        PT = np.asarray(PT, dtype=float)
        if PT.shape != (n, 2, 2):
            raise ValueError("PT must have shape (n, 2, 2) and match freq.size.")

        if PT_err is None:
            if "PT_err" in edi:
                PT_err = np.asarray(edi["PT_err"], dtype=float)
        elif PT_err is not None:
            PT_err = np.asarray(PT_err, dtype=float)
            if PT_err.shape != (n, 2, 2):
                raise ValueError("PT_err must have shape (n, 2, 2) and match freq.size.")

    else:
        # no PT argument: try dict
        if "PT" in edi:
            PT = np.asarray(edi["PT"], dtype=float)
            if PT.shape != (n, 2, 2):
                raise ValueError("edi['PT'] must have shape (n, 2, 2) and match freq.size.")
            if "PT_err" in edi and PT_err is None:
                PT_err = np.asarray(edi["PT_err"], dtype=float)
                if PT_err.shape != (n, 2, 2):
                    raise ValueError("edi['PT_err'] must have shape (n, 2, 2) and match freq.size.")
        elif add_pt_blocks:
            # compute from Z (and Z_err if available)
            Z_err_arr = None
            if Z_err is not None:
                Z_err_arr = np.asarray(Z_err, dtype=float)
                if Z_err_arr.shape != (n, 2, 2):
                    raise ValueError("edi['Z_err'] must have shape (n, 2, 2) and match freq.size.")

            from .ediprocx import compute_pt  # adjust import if needed

            PT_auto, PT_err_auto = compute_pt(Z, Z_err_arr, err_kind=pt_err_kind)
            PT = np.asarray(PT_auto, dtype=float)
            if PT.shape != (n, 2, 2):
                raise RuntimeError("compute_pt returned PT with unexpected shape.")
            if PT_err is None and PT_err_auto is not None:
                PT_err = np.asarray(PT_err_auto, dtype=float)
                if PT_err.shape != (n, 2, 2):
                    raise RuntimeError("compute_pt returned PT_err with unexpected shape.")

    # Header meta: merge edi['header_meta'] (if any) with header_meta argument
    meta: Dict[str, str] = {}
    if "header_meta" in edi and edi["header_meta"] is not None:
        meta.update(dict(edi["header_meta"]))
    if header_meta is not None:
        meta.update(dict(header_meta))

    if source_kind:
        existing_keys = {k.upper() for k in meta.keys()}
        if "SOURCE_KIND" not in existing_keys:
            meta["SOURCE_KIND"] = str(source_kind)

    return write_edi(
        out_path,
        station=station,
        freq=freq,
        Z=Z,
        T=T,
        PT=PT,
        Z_err=Z_err,
        T_err=T_err,
        PT_err=PT_err,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        elev_m=elev_m,
        header_meta=meta,
        numbers_per_line=numbers_per_line,
        lon_keyword=lon_keyword,
    )
