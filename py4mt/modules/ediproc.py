#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ediproc.py
=================
Consolidated I/O utilities for Magnetotelluric EDI files used in the EDI project.

This module extracts and refactors the input/output functions scattered across
`edi_processor.py` and `edi_writer.py` into a clean, reusable API. It supports:
- Reading/parsing EDI text for standard Z/T tables and Phoenix SPECTRA blocks.
- Reconstructing complex spectral matrices and deriving Z/T from 
  Phoenix 7×7 spectra.
- Loading EDI from disk with a single convenience function that returns 
  (freq, Z, T, station, source_kind).
- Writing EDI files from arrays or from NPZ bundles written by the processor.
- Optional support for variance blocks (.VAR) for Z, T, and Phase Tensor (PT).

Public API
----------
- read_edi_text(path)
- parse_spectra_blocks(edi_text)
- reconstruct_S_phoenix(mat7)
- ZT_from_S(S, ref='RH')
- parse_block_values(edi_text)
- load_edi(path, prefer_spectra=True, ref='RH', rotate_deg=0.0, fill_invalid=None)
- write_edi(path, *, station, freq, Z, T=None, lat_deg=None, lon_deg=None, 
            elev_m=None,
            header_meta=None, numbers_per_line=6, Z_var_re=None, Z_var_im=None,
            T_var_re=None, T_var_im=None, PT=None, PT_var=None)
- write_edi_from_npz(npz_path, out_path=None, *, numbers_per_line=6, 
                     lat_deg=None, lon_deg=None, elev_m=None)

Notes
-----
- Rotation and fill-invalid logic are kept for convenience when loading EDIs
- All functions have low-level docstrings as requested for this project.

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-11-12 18:11:44 UTC
"""

from __future__ import annotations

from typing import Generator, Iterable, Optional, Tuple, Dict
from pathlib import Path
import re
import io
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
    "write_edi",
    "write_edi_from_npz",
]


# -----------------------------
# Small linear-algebra helpers
# -----------------------------

def _rot2(theta_rad: float) -> np.ndarray:
    """Return the 2×2 rotation matrix for a counter-clockwise angle.

    Parameters
    ----------
    theta_rad : float
        Rotation angle in radians (counter-clockwise).

    Returns
    -------
    numpy.ndarray
        2×2 array representing the rotation matrix.
    """
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s], [s,  c]], dtype=np.float64)


def _rotate_ZT(Z: Optional[np.ndarray], T: Optional[np.ndarray], theta_deg: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Rotate impedance and tipper tensors by a given angle in degrees (CCW).

    Parameters
    ----------
    Z : numpy.ndarray or None
        Complex impedance tensor of shape (2, 2) or (n, 2, 2). If None, no rotation is applied.
    T : numpy.ndarray or None
        Complex tipper tensor of shape (1, 2) or (n, 1, 2). If None, no rotation is applied.
    theta_deg : float
        Rotation angle in degrees (counter-clockwise) in the E/H frame.

    Returns
    -------
    (Z_rot, T_rot) : tuple
        Rotated arrays matching the input shapes (or None if respective input is None).
    """
    if (Z is None) and (T is None):
        return None, None
    th = np.deg2rad(theta_deg)
    R = _rot2(th)
    Rm = _rot2(-th)

    def _rotZ(z):
        if z is None:
            return None
        if z.ndim == 2:
            return (R @ z @ Rm)
        # vectorized over first axis
        return np.einsum('ij,njk,kl->nil', R, z, Rm)

    def _rotT(t):
        if t is None:
            return None
        if t.ndim == 2:
            return (t @ Rm)
        return np.einsum('nij,jk->nik', t, Rm)

    return _rotZ(Z), _rotT(T)


def _fill_invalid_inplace(a: Optional[np.ndarray], fill_value: Optional[float]) -> None:
    """Replace non-finite entries in-place for real/imag parts of an array.

    Parameters
    ----------
    a : numpy.ndarray or None
        Array to sanitize. If complex, both real and imaginary parts are processed.
    fill_value : float or None
        Replacement constant for non-finite values. If None, this routine is a no-op.

    Returns
    -------
    None
        Operates in-place; does not return a result.
    """
    if a is None or fill_value is None:
        return
    if np.iscomplexobj(a):
        re = a.real.copy()
        im = a.imag.copy()
        re[~np.isfinite(re)] = fill_value
        im[~np.isfinite(im)] = fill_value
        a[...] = re + 1j * im
    else:
        a[~np.isfinite(a)] = fill_value


# -----------------------------
# Reading / parsing (from edi_processor.py)
# -----------------------------

COMP_LIST = ['hx', 'hy', 'hz', 'ex', 'ey', 'rhx', 'rhy']
IDX = {c: i for i, c in enumerate(COMP_LIST)}


def dataframe_from_arrays(freq: np.ndarray, Z: np.ndarray, T: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Build a pandas DataFrame with columns matching `edi_processor` CSV schema.

    Parameters
    ----------
    freq : ndarray, shape (n,)
        Frequencies in Hz.
    Z : ndarray, shape (n,2,2), complex128
        Impedance tensor per frequency.
    T : ndarray or None, shape (n,1,2), complex128, optional
        Tipper per frequency. If None, zeroed columns are used.

    Returns
    -------
    pandas.DataFrame
        Columns: freq, Zxx_re/_im, Zxy_re/_im, Zyx_re/_im, Zyy_re/_im,
                 Tx_re/_im, Ty_re/_im, plus derived rho/phi and PT entries.
    """
    n = freq.shape[0]
    df = pd.DataFrame({
        'freq': freq,
        'Zxx_re': Z[:,0,0].real, 'Zxx_im': Z[:,0,0].imag,
        'Zxy_re': Z[:,0,1].real, 'Zxy_im': Z[:,0,1].imag,
        'Zyx_re': Z[:,1,0].real, 'Zyx_im': Z[:,1,0].imag,
        'Zyy_re': Z[:,1,1].real, 'Zyy_im': Z[:,1,1].imag,
    })
    if T is None:
        df['Tx_re'] = 0.0; df['Tx_im'] = 0.0
        df['Ty_re'] = 0.0; df['Ty_im'] = 0.0
    else:
        df['Tx_re'] = T[:,0,0].real; df['Tx_im'] = T[:,0,0].imag
        df['Ty_re'] = T[:,0,1].real; df['Ty_im'] = T[:,0,1].imag

    # Derived quantities
    period = 1.0 / np.asarray(freq, dtype=float)
    omega_mu = 2*np.pi*freq*MU0

    for lab, zr, zi in (
        ('xx', df['Zxx_re'].to_numpy(), df['Zxx_im'].to_numpy()),
        ('xy', df['Zxy_re'].to_numpy(), df['Zxy_im'].to_numpy()),
        ('yx', df['Zyx_re'].to_numpy(), df['Zyx_im'].to_numpy()),
        ('yy', df['Zyy_re'].to_numpy(), df['Zyy_im'].to_numpy()),
    ):
        Zc = zr + 1j*zi
        rho = np.abs(Zc)**2 / omega_mu
        phi = np.degrees(np.arctan2(zi, zr))
        df[f'rho_{{lab}}'] = rho
        df[f'phi_{{lab}}'] = phi

    # Phase Tensor
    PT = np.zeros((n,2,2), dtype=float)
    for i in range(n):
        X = np.array([[df['Zxx_re'].iat[i], df['Zxy_re'].iat[i]],
                      [df['Zyx_re'].iat[i], df['Zyy_re'].iat[i]]], dtype=float)
        Y = np.array([[df['Zxx_im'].iat[i], df['Zxy_im'].iat[i]],
                      [df['Zyx_im'].iat[i], df['Zyy_im'].iat[i]]], dtype=float)
        try:
            Xinv = np.linalg.inv(X)
        except np.linalg.LinAlgError:
            Xinv = np.linalg.pinv(X)
        PT[i] = Y @ Xinv
    df['ptxx_re'] = PT[:,0,0]; df['ptxy_re'] = PT[:,0,1]
    df['ptyx_re'] = PT[:,1,0]; df['ptyy_re'] = PT[:,1,1]

    return df



def read_edi_text(path: Path | str) -> str:
    """Read EDI file as text using latin-1 with 'ignore' errors.

    Parameters
    ----------
    path : str or pathlib.Path
        Filesystem path to the EDI file.

    Returns
    -------
    str
        Full file content as a single string.
    """
    return Path(path).read_text(encoding='latin-1', errors='ignore')


def parse_spectra_blocks(edi_text: str):
    """Yield Phoenix >SPECTRA blocks as (freq_Hz, avgt, mat7x7_real).

    This follows the implementation used in `edi_processor.py` (see project source).

    Parameters
    ----------
    edi_text : str
        Entire EDI file content.

    Yields
    ------
    tuple
        (f, avgt, mat7) where:
          - f : float          -> frequency in Hz
          - avgt : float       -> averaging time if present (nan if not present)
          - mat7 : (7,7) float -> real-valued matrix with diag autos, lower=Re, upper=Im.
    """
    for m in re.finditer(r'>SPECTRA[^\n]*\n((?:[^\n]*\n)+?)(?=>SPECTRA|>END|$)', edi_text):
        header = m.group(0).splitlines()[0]
        body = m.group(1)
        fm = re.search(
            r'FREQ\s*=\s*([0-9.]+[ED][+\-]?\d+|[0-9.]+)', header, flags=re.IGNORECASE)
        if not fm:
            continue
        f = float(fm.group(1).replace('D', 'E'))
        am = re.search(
            r'AVGT\s*=\s*([0-9.]+[ED][+\-]?\d+|[0-9.]+)', header, flags=re.IGNORECASE)
        avgt = float(am.group(1).replace('D', 'E')) if am else np.nan
        rows = [ln for ln in body.splitlines() if ln.strip()]
        arr = np.array([
            list(map(float, re.findall(
                r'[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[EeDd][+\-]?\d+)?', ln)))
            for ln in rows
        ])
        if arr.shape == (7, 7):
            yield f, avgt, arr


def reconstruct_S_phoenix(mat7: np.ndarray) -> np.ndarray:
    """Reconstruct complex Hermitian spectral matrix S (7×7) from Phoenix 7×7 real layout.

    Parameters
    ----------
    mat7 : numpy.ndarray
        Real-valued (7,7) array with autos on diagonal, lower triangle = Re, upper = Im.

    Returns
    -------
    numpy.ndarray
        Complex Hermitian matrix S of shape (7,7).
    """
    S = np.zeros((7, 7), dtype=np.complex128)
    for i in range(7):
        S[i, i] = mat7[i, i]
        for j in range(i + 1, 7):
            Re = mat7[j, i]
            Im = mat7[i, j]
            S[i, j] = Re + 1j * Im
            S[j, i] = Re - 1j * Im
    return S


def ZT_from_S(S: np.ndarray, ref: str = 'RH'):
    """Compute impedance (Z) and tipper (T) from spectral matrix using chosen H reference.

    Parameters
    ----------
    S : numpy.ndarray
        Complex spectral matrix of shape (7,7).
    ref : str, optional
        Either 'H' for (Hx, Hy) or 'RH' for (RHx, RHy) as magnetic reference.

    Returns
    -------
    (Z, T) : tuple of numpy.ndarray
        Z : (2,2) complex impedance tensor
        T : (1,2) complex tipper
    """
    if ref.upper() == 'H':
        h1, h2 = IDX['hx'], IDX['hy']
    else:
        h1, h2 = IDX['rhx'], IDX['rhy']
    ex, ey, hz = IDX['ex'], IDX['ey'], IDX['hz']
    SHH = np.array([[S[h1, h1], S[h1, h2]], [
                   S[h2, h1], S[h2, h2]]], dtype=np.complex128)
    SEH = np.array([[S[ex, h1], S[ex, h2]], [
                   S[ey, h1], S[ey, h2]]], dtype=np.complex128)
    SBH = np.array([[S[hz, h1], S[hz, h2]]], dtype=np.complex128)
    try:
        SHH_inv = np.linalg.inv(SHH)
    except np.linalg.LinAlgError:
        SHH_inv = np.linalg.pinv(SHH)
    Z = SEH @ SHH_inv
    T = SBH @ SHH_inv
    return Z, T


def parse_block_values(edi_text: str):
    """Parse standard EDI value blocks into (freqs, Z, T).

    Parameters
    ----------
    edi_text : str
        Full EDI content.

    Returns
    -------
    tuple or None
        (freqs, Z, T) where:
          - freqs : (n,) float array [Hz]
          - Z     : (n,2,2) complex impedance
          - T     : (n,1,2) complex tipper (may be zeros if missing)
        Returns None if no frequencies/Z could be parsed.

    Notes
    -----
    Mirrors the behavior used in `edi_processor.py`. Reference implementation:
    see project source (function names and regex layout preserved).
    """
    f_matches = re.findall(
        r'FREQ\s*=\s*([0-9.]+[ED][+\-]?\d+|[0-9.]+)', edi_text, flags=re.IGNORECASE)
    if not f_matches:
        return None
    freqs = np.array([float(s.replace('D', 'E'))
                     for s in f_matches], dtype=float)
    n = freqs.size

    def get_arr(tag: str):
        pat = rf'>{{tag}}[^\n]*\n((?:[^\n]*\n)+?)(?=>[A-Z]|>END|$)'
        m = re.search(pat, edi_text, flags=re.IGNORECASE)
        if not m:
            return None
        nums = re.findall(
            r'[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[EeDd][+\-]?\d+)?', m.group(1))
        arr = np.array([float(v.replace('D', 'E')) for v in nums], dtype=float)
        return arr[:n] if arr.size >= n else None

    Z = np.zeros((n, 2, 2), dtype=np.complex128)
    ok = False
    for c, (i, j) in {'ZXX': (0, 0), 'ZXY': (0, 1), 'ZYX': (1, 0), 'ZYY': (1, 1)}.items():
        re_arr = get_arr(c + 'R') or get_arr(c + '.RE') or get_arr(c + '_RE')
        im_arr = get_arr(c + 'I') or get_arr(c + '.IM') or get_arr(c + '_IM')
        if re_arr is not None and im_arr is not None:
            ok = True
            Z[:, i, j] = re_arr + 1j * im_arr
    if not ok:
        return None

    T = np.zeros((n, 1, 2), dtype=np.complex128)
    txr = get_arr('TXR') or get_arr('TX.RE') or get_arr('TX_RE')
    txi = get_arr('TXI') or get_arr('TX.IM') or get_arr('TX_IM')
    tyr = get_arr('TYR') or get_arr('TY.RE') or get_arr('TY_RE')
    tyi = get_arr('TYI') or get_arr('TY.IM') or get_arr('TY_IM')
    if txr is not None and txi is not None:
        T[:, 0, 0] = txr + 1j * txi
    if tyr is not None and tyi is not None:
        T[:, 0, 1] = tyr + 1j * tyi

    return freqs, Z, T


def load_edi(path: Path | str, *, prefer_spectra: bool = True, ref: str = 'RH',
             rotate_deg: float = 0.0, fill_invalid: Optional[float] = None):
    """Load an EDI file, optionally preferring Phoenix SPECTRA reconstruction.

    Parameters
    ----------
    path : str or pathlib.Path
        EDI file path.
    prefer_spectra : bool, optional
        If True and >SPECTRA blocks are present, reconstruct Z/T from spectra; otherwise
        use the standard tabulated Z/T blocks. Default is True.
    ref : str, optional
        Magnetic reference for SPECTRA path ('H' or 'RH'). Default 'RH'.
    rotate_deg : float, optional
        Rotate Z/T by this angle in degrees (CCW). Default 0.0.
    fill_invalid : float or None, optional
        Replace non-finite values in Z/T with this constant. If None, leave as-is.

    Returns
    -------
    (freq, Z, T, station, source_kind) : tuple
        freq : (n,) float array in Hz
        Z    : (n,2,2) complex impedance
        T    : (n,1,2) complex tipper (zeros if not present in the table path)
        station : str extracted from DATAID (or 'UNKNOWN')
        source_kind : str either 'spectra' or 'tables'

    Notes
    -----
    Replicates the logic of `edi_processor.run` for data extraction, but without
    any CSV/HDF5 writing or plotting. This function is intended purely for I/O.
    """
    text = read_edi_text(path)
    m = re.search(
        r'DATAID\s*=\s*"?([A-Za-z0-9_\-\.]+)"?', text, flags=re.IGNORECASE)
    station = m.group(1) if m else 'UNKNOWN'

    spectra_blocks = list(parse_spectra_blocks(text))
    if prefer_spectra and spectra_blocks:
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
        source_kind = 'spectra'
    else:
        parsed = parse_block_values(text)
        if parsed is None:
            raise RuntimeError(
                "Could not find SPECTRA blocks or standard Z/T tables in EDI.")
        freq, Z, T = parsed
        source_kind = 'tables'

    # Optional rotation and sanitization
    if rotate_deg:
        Z, T = _rotate_ZT(Z, T, rotate_deg)
    if fill_invalid is not None:
        _fill_invalid_inplace(Z, fill_invalid)
        _fill_invalid_inplace(T, fill_invalid)

    # Sort by descending frequency to match typical project expectations
    order = np.argsort(freq)[::-1]
    freq = freq[order]
    Z = Z[order]
    T = T[order]
    return freq, Z, T, station, source_kind


# -----------------------------
# Writing (from edi_writer.py) with fixed, explicit parameters
# -----------------------------

def _fmt_block(values: np.ndarray, per_line: int = 6, fmt: str = "{: .8E}") -> str:
    """Format a 1-D array into fixed-width lines for EDI value blocks.

    Parameters
    ----------
    values : array-like
        Sequence of floats to be written.
    per_line : int, optional
        Number of values per line (default 6).
    fmt : str, optional
        Format string applied to each float (default "{: .8E}").

    Returns
    -------
    str
        Multi-line string containing values formatted per EDI conventions.
    """
    v = np.asarray(values).ravel()
    out_lines = []
    for i in range(0, v.size, per_line):
        chunk = v[i:i+per_line]
        out_lines.append(" ".join(fmt.format(float(x)) for x in chunk))
    return "\n".join(out_lines)


def _ensure_1d(x: np.ndarray, name: str) -> np.ndarray:
    """Ensure that an input array is 1-D, raising ValueError otherwise.

    Parameters
    ----------
    x : numpy.ndarray
        Input array to validate/reshape.
    name : str
        Descriptive name of the parameter for error messages.

    Returns
    -------
    numpy.ndarray
        Flattened 1-D array.

    Raises
    ------
    ValueError
        If the input array is not 1-D.
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1-D (got shape {x.shape})")
    return x


def write_edi(path: Path | str, *, station: str, freq: np.ndarray, Z: np.ndarray,
              T: Optional[np.ndarray] = None, lat_deg: Optional[float] = None,
              lon_deg: Optional[float] = None, elev_m: Optional[float] = None,
              header_meta: Optional[Dict[str, str]] = None, numbers_per_line: int = 6,
              Z_var_re: Optional[np.ndarray] = None, Z_var_im: Optional[np.ndarray] = None,
              T_var_re: Optional[np.ndarray] = None, T_var_im: Optional[np.ndarray] = None,
              PT: Optional[np.ndarray] = None, PT_var: Optional[np.ndarray] = None) -> str:
    """Write an ASCII EDI file with standard Z/T (+ optional .VAR) and PT blocks.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination file path.
    station : str
        Station identifier written to >HEAD/DATAID.
    freq : numpy.ndarray
        1-D float array of frequencies [Hz] with shape (n,).
    Z : numpy.ndarray
        Complex impedance array with shape (n, 2, 2).
    T : numpy.ndarray, optional
        Complex tipper array with shape (n, 1, 2). If None, tipper blocks are omitted.
    lat_deg, lon_deg, elev_m : float, optional
        Optional coordinates and elevation metadata for >HEAD.
    header_meta : dict, optional
        Additional key-value pairs written to >HEAD as "KEY= VALUE",
        intended for provenance information.
    numbers_per_line : int, optional
        Count of numbers per line for EDI value blocks.
    Z_var_re, Z_var_im : numpy.ndarray, optional
        Real-valued variances for the real/imag parts of Z with shape (n,2,2).
    T_var_re, T_var_im : numpy.ndarray, optional
        Real-valued variances for T real/imag with shape (n,1,2).
    PT : numpy.ndarray, optional
        Real-valued Phase Tensor entries with shape (n,2,2). If None, computed from Z.
    PT_var : numpy.ndarray, optional
        Real-valued variances for PT with shape (n,2,2).

    Returns
    -------
    str
        The string path of the written file.

    Raises
    ------
    ValueError
        If array shapes are inconsistent with (n,2,2) or (n,1,2).
    """
    path = Path(path)
    freq = _ensure_1d(np.asarray(freq, dtype=float), "freq")
    Z = np.asarray(Z)
    if Z.shape != (freq.size, 2, 2):
        raise ValueError(
            f"Z must have shape (n,2,2) with n={freq.size}, got {Z.shape}")

    if T is not None:
        T = np.asarray(T)
        if T.shape != (freq.size, 1, 2):
            raise ValueError(
                f"T must have shape (n,1,2) with n={freq.size}, got {T.shape}")

    def _check_opt(name: str, arr: Optional[np.ndarray], shape: Tuple[int, ...]) -> Optional[np.ndarray]:
        if arr is None:
            return None
        a = np.asarray(arr)
        if a.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {a.shape}")
        return a

    Z_var_re = _check_opt("Z_var_re", Z_var_re, (freq.size, 2, 2))
    Z_var_im = _check_opt("Z_var_im", Z_var_im, (freq.size, 2, 2))
    T_var_re = _check_opt("T_var_re", T_var_re,
                          (freq.size, 1, 2)) if T is not None else None
    T_var_im = _check_opt("T_var_im", T_var_im,
                          (freq.size, 1, 2)) if T is not None else None

    # Phase Tensor: compute if not provided
    if PT is None:
        PT = np.zeros((freq.size, 2, 2), dtype=float)
        for i in range(freq.size):
            X = Z[i].real
            Y = Z[i].imag
            try:
                Xinv = np.linalg.inv(X)
            except np.linalg.LinAlgError:
                Xinv = np.linalg.pinv(X)
            PT[i] = (Y @ Xinv)
    else:
        PT = _check_opt("PT", PT, (freq.size, 2, 2))

    PT_var = _check_opt("PT_var", PT_var, (freq.size, 2, 2)
                        ) if PT_var is not None else None

    Zxx = Z[:, 0, 0]
    Zxy = Z[:, 0, 1]
    Zyx = Z[:, 1, 0]
    Zyy = Z[:, 1, 1]
    blocks: Dict[str, np.ndarray] = {
        "ZXXR": Zxx.real, "ZXXI": Zxx.imag,
        "ZXYR": Zxy.real, "ZXYI": Zxy.imag,
        "ZYXR": Zyx.real, "ZYXI": Zyx.imag,
        "ZYYR": Zyy.real, "ZYYI": Zyy.imag,
    }

    if Z_var_re is not None:
        blocks.update({
            "ZXXR.VAR": Z_var_re[:, 0, 0], "ZXYR.VAR": Z_var_re[:, 0, 1],
            "ZYXR.VAR": Z_var_re[:, 1, 0], "ZYYR.VAR": Z_var_re[:, 1, 1],
        })
    if Z_var_im is not None:
        blocks.update({
            "ZXXI.VAR": Z_var_im[:, 0, 0], "ZXYI.VAR": Z_var_im[:, 0, 1],
            "ZYXI.VAR": Z_var_im[:, 1, 0], "ZYYI.VAR": Z_var_im[:, 1, 1],
        })

    if T is not None:
        Tx, Ty = T[:, 0, 0], T[:, 0, 1]
        blocks.update({"TXR": Tx.real, "TXI": Tx.imag,
                      "TYR": Ty.real, "TYI": Ty.imag})
        if T_var_re is not None:
            blocks.update(
                {"TXR.VAR": T_var_re[:, 0, 0], "TYR.VAR": T_var_re[:, 0, 1]})
        if T_var_im is not None:
            blocks.update(
                {"TXI.VAR": T_var_im[:, 0, 0], "TYI.VAR": T_var_im[:, 0, 1]})

    # Phase Tensor real components (+ optional variances)
    blocks.update({"PTXX": PT[:, 0, 0], "PTXY": PT[:, 0, 1],
                  "PTYX": PT[:, 1, 0], "PTYY": PT[:, 1, 1]})
    if PT_var is not None:
        blocks.update({"PTXX.VAR": PT_var[:, 0, 0], "PTXY.VAR": PT_var[:, 0, 1],
                       "PTYX.VAR": PT_var[:, 1, 0], "PTYY.VAR": PT_var[:, 1, 1]})

    buf = io.StringIO()
    buf.write(">HEAD\n")
    buf.write(f"    DATAID= {station}\n")
    if lat_deg is not None and lon_deg is not None:
        buf.write(f"    LAT= {lat_deg:.6f}\n")
        buf.write(f"    LON= {lon_deg:.6f}\n")
    if elev_m is not None:
        buf.write(f"    ELEV= {elev_m:.3f}\n")
    buf.write(f"    CREATED= {np.datetime64('now')}\n")
    buf.write(f"    CREATED_BY= ChatGPT (GPT-5 Thinking)\n")
    if header_meta:
        for k, v in header_meta.items():
            buf.write(f"    {k}= {v}\n")

    buf.write(">FREQ\n")
    buf.write(_fmt_block(freq, per_line=numbers_per_line))
    buf.write("\n")

    preferred_order = [
        "ZXXR", "ZXXI", "ZXYR", "ZXYI", "ZYXR", "ZYXI", "ZYYR", "ZYYI",
        "ZXXR.VAR", "ZXXI.VAR", "ZXYR.VAR", "ZXYI.VAR", "ZYXR.VAR", "ZYXI.VAR", "ZYYR.VAR", "ZYYI.VAR",
        "TXR", "TXI", "TYR", "TYI", "TXR.VAR", "TXI.VAR", "TYR.VAR", "TYI.VAR",
        "PTXX", "PTXY", "PTYX", "PTYY", "PTXX.VAR", "PTXY.VAR", "PTYX.VAR", "PTYY.VAR",
    ]
    for tag in preferred_order:
        if tag not in blocks:
            continue
        buf.write(f">{tag}\n")
        buf.write(_fmt_block(blocks[tag], per_line=numbers_per_line))
        buf.write("\n")

    buf.write(">END\n")
    path.write_text(buf.getvalue(), encoding="utf-8")
    return str(path)


def write_edi_from_npz(npz_path: Path | str, out_path: Optional[Path | str] = None, *,
                       numbers_per_line: int = 6, lat_deg: Optional[float] = None,
                       lon_deg: Optional[float] = None, elev_m: Optional[float] = None) -> str:
    """Write an EDI file directly from an NPZ bundle created by the processor.

    Parameters
    ----------
    npz_path : str or pathlib.Path
        Path to NPZ file containing keys: 'station', 'freq', 'Z', optionally 'T',
        and optionally metadata like 'source_kind', 'ref', 'rotate_deg', 'prefer_spectra'.
    out_path : str or pathlib.Path, optional
        Target EDI file path. If None, use "<station>.edi" next to the NPZ.
    numbers_per_line : int, optional
        Numbers per line for value blocks (default 6).
    lat_deg, lon_deg, elev_m : float, optional
        Optional location metadata.

    Returns
    -------
    str
        Path of the written EDI file.
    """
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    station = str(data["station"]) if "station" in data.files else "UNKNOWN"
    freq = data["freq"]
    Z = data["Z"]
    T = data["T"] if "T" in data.files else None
    if out_path is None:
        out_path = npz_path.with_suffix("").with_name(f"{station}.edi")
    header_meta = {
        "SOURCE_FILE": str(npz_path.name),
    }
    for k in ("source_kind", "ref", "rotate_deg", "prefer_spectra"):
        if k in data.files:
            header_meta[k.upper()] = str(data[k])
    return write_edi(
        out_path,
        station=station,
        freq=freq,
        Z=Z,
        T=T,
        lat_deg=lat_deg, lon_deg=lon_deg, elev_m=elev_m,
        header_meta=header_meta,
        numbers_per_line=numbers_per_line,
    )
