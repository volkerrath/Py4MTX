#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
edi_processor.py
================

Processing CLI for Magnetotelluric EDI files. Delegates I/O to :mod:`edi_funcs`,
and (optionally) plotting to :mod:`edi_viz`.

Updates in this version
-----------------------
- **EDI path is now a keyword argument**: `--edi /path/to/file.edi` (no positional).
- **Plotting options** (saved as PNGs):
  - `--plot-dir PATH`  (directory for figures; created if needed)
  - `--plot-rho [xx,xy,yx,yy]`  (comma-separated components; defaults xy,yx)
  - `--plot-phase [xx,xy,yx,yy]`
  - `--plot-tipper`
  - `--plot-pt`

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-11-12 18:23:33 UTC
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import argparse
import numpy as np

# Local consolidated I/O helpers
from edi_funcs import load_edi

def _as_rec(freq: np.ndarray, Z: np.ndarray, T: Optional[np.ndarray]) -> np.ndarray:
    """Pack frequency, impedance, and tipper arrays into a 1-D structured record array.

    Parameters
    ----------
    freq : ndarray, shape (n,)
        Frequencies in Hz.
    Z : ndarray, shape (n, 2, 2)
        Complex impedance tensor per frequency.
    T : ndarray or None, shape (n, 1, 2)
        Complex tipper per frequency; may be None.

    Returns
    -------
    numpy.ndarray
        Structured array with fields:
        - 'freq' (float64)
        - 'Zxx','Zxy','Zyx','Zyy' (complex128)
        - 'Tx','Ty' (complex128, zero if T is None)
    """
    n = freq.shape[0]
    out = np.zeros(n, dtype=[
        ('freq','<f8'),
        ('Zxx','<c16'),('Zxy','<c16'),('Zyx','<c16'),('Zyy','<c16'),
        ('Tx','<c16'),('Ty','<c16'),
    ])
    out['freq'] = freq
    out['Zxx'] = Z[:,0,0]
    out['Zxy'] = Z[:,0,1]
    out['Zyx'] = Z[:,1,0]
    out['Zyy'] = Z[:,1,1]
    if T is None:
        out['Tx'] = 0.0+0.0j
        out['Ty'] = 0.0+0.0j
    else:
        out['Tx'] = T[:,0,0]
        out['Ty'] = T[:,0,1]
    return out


def _phase_tensor(Z: np.ndarray) -> np.ndarray:
    """Compute the real-valued Phase Tensor Φ = Y @ X^{-1} per frequency.

    Parameters
    ----------
    Z : ndarray, shape (n, 2, 2)
        Complex impedance per frequency.

    Returns
    -------
    ndarray, shape (n, 2, 2)
        Real-valued phase tensor per frequency (computed from Re/Im parts).
    """
    n = Z.shape[0]
    PT = np.zeros((n,2,2), dtype=float)
    for i in range(n):
        X = Z[i].real
        Y = Z[i].imag
        try:
            Xinv = np.linalg.inv(X)
        except np.linalg.LinAlgError:
            Xinv = np.linalg.pinv(X)
        PT[i] = Y @ Xinv
    return PT


def export_csv(path: Path | str, rec: np.ndarray, *, include_phase_tensor: bool = True) -> str:
    """Write CSV with freq, complex Z/T (split Re/Im), and optional PT.

    Parameters
    ----------
    path : str or Path
        CSV output path.
    rec : ndarray
        Structured array produced by :func:`_as_rec`.
    include_phase_tensor : bool, optional
        If True, append PT columns (PTxx, PTxy, PTyx, PTyy).

    Returns
    -------
    str
        String path of the written CSV.
    """
    import csv
    path = Path(path)
    Z = np.stack([rec['Zxx'],rec['Zxy'],rec['Zyx'],rec['Zyy']], axis=1)
    PT = _phase_tensor(Z.reshape(-1,2,2))

    cols = ['freq',
            'Zxx_re','Zxx_im','Zxy_re','Zxy_im','Zyx_re','Zyx_im','Zyy_re','Zyy_im',
            'Tx_re','Tx_im','Ty_re','Ty_im']
    if include_phase_tensor:
        cols += ['PTxx','PTxy','PTyx','PTyy']

    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(rec.shape[0]):
            row = [
                float(rec['freq'][i]),
                rec['Zxx'][i].real, rec['Zxx'][i].imag,
                rec['Zxy'][i].real, rec['Zxy'][i].imag,
                rec['Zyx'][i].real, rec['Zyx'][i].imag,
                rec['Zyy'][i].real, rec['Zyy'][i].imag,
                rec['Tx'][i].real,  rec['Tx'][i].imag,
                rec['Ty'][i].real,  rec['Ty'][i].imag,
            ]
            if include_phase_tensor:
                row += list(PT[i].ravel())
            w.writerow(row)
    return str(path)


def export_hdf5(path: Path | str, *, freq: np.ndarray, Z: np.ndarray, T: Optional[np.ndarray]) -> str:
    """Write HDF5 file (if pandas/h5py available) containing freq, Z, T.

    Parameters
    ----------
    path : str or Path
        HDF5 output path.
    freq : ndarray
        Frequencies (n,).
    Z : ndarray
        Complex impedance (n,2,2).
    T : ndarray or None
        Complex tipper (n,1,2) or None.

    Returns
    -------
    str
        String path of the written HDF5.

    Notes
    -----
    Uses pandas.HDFStore if available; otherwise raises ImportError.
    """
    import pandas as pd
    path = Path(path)
    df = pd.DataFrame({
        'freq': freq,
        'Zxx_re': Z[:,0,0].real, 'Zxx_im': Z[:,0,0].imag,
        'Zxy_re': Z[:,0,1].real, 'Zxy_im': Z[:,0,1].imag,
        'Zyx_re': Z[:,1,0].real, 'Zyx_im': Z[:,1,0].imag,
        'Zyy_re': Z[:,1,1].real, 'Zyy_im': Z[:,1,1].imag,
    })
    if T is not None:
        df['Tx_re'] = T[:,0,0].real
        df['Tx_im'] = T[:,0,0].imag
        df['Ty_re'] = T[:,0,1].real
        df['Ty_im'] = T[:,0,1].imag
    with pd.HDFStore(str(path), mode='w') as store:
        store['Z_T'] = df
    return str(path)


def export_npz(path: Path | str, *, station: str, source_kind: str, freq: np.ndarray, Z: np.ndarray, T: Optional[np.ndarray]) -> str:
    """Write an NPZ bundle with station metadata and arrays for freq, Z, and T.

    Parameters
    ----------
    path : str or Path
        NPZ output path.
    station : str
        Station identifier.
    source_kind : str
        'spectra' or 'tables' per :func:`edi_funcs.load_edi`.
    freq : ndarray
        (n,) float array of frequencies [Hz].
    Z : ndarray
        (n,2,2) complex impedance.
    T : ndarray or None
        (n,1,2) complex tipper or None.

    Returns
    -------
    str
        Path of the written NPZ file.
    """
    path = Path(path)
    np.savez(path,
             station=station,
             source_kind=source_kind,
             freq=freq, Z=Z, T=T if T is not None else np.array([]),
             prefer_spectra=True)
    return str(path)


def build_argparser() -> argparse.ArgumentParser:
    """Create an argument parser for the EDI processor CLI.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser instance with all CLI options defined.
    """
    p = argparse.ArgumentParser(description="Process Magnetotelluric EDI files.")
    p.add_argument('--edi', type=Path, required=True, help='Path to input .edi file')
    p.add_argument('--prefer-spectra', action='store_true', default=False, help='Prefer Phoenix >SPECTRA when present')
    p.add_argument('--ref', choices=['H','RH'], default='RH', help='Magnetic reference for SPECTRA path')
    p.add_argument('--rotate', type=float, default=0.0, help='Rotate Z/T by degrees CCW')
    p.add_argument('--fill-invalid', type=float, default=None, help='Replace non-finite entries with this value')
    p.add_argument('--csv', type=Path, default=None, help='Write CSV to this path')
    p.add_argument('--hdf5', type=Path, default=None, help='Write HDF5 to this path')
    p.add_argument('--npz', type=Path, default=None, help='Write NPZ bundle to this path')
    p.add_argument('--no-pt', action='store_true', help='Do not append Phase Tensor to CSV')

    # Plotting
    p.add_argument('--plot-dir', type=Path, default=None, help='Directory to save plots (PNG). If unset, no plots are generated.')
    p.add_argument('--plot-rho', type=str, default=None, help='Comma-separated Z components to plot for apparent resistivity (e.g. "xy,yx").')
    p.add_argument('--plot-phase', type=str, default=None, help='Comma-separated Z components to plot for phase (e.g. "xy,yx").')
    p.add_argument('--plot-tipper', action='store_true', help='Plot tipper (Tx, Ty).')
    p.add_argument('--plot-pt', action='store_true', help='Plot Phase Tensor components.')
    return p


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for the EDI processor command-line tool.

    Parameters
    ----------
    argv : list of str, optional
        Argument vector (for testing). If None, use sys.argv[1:].

    Returns
    -------
    int
        Process exit code (0 on success, non-zero on failure).
    """
    import sys
    args = build_argparser().parse_args(argv)

    freq, Z, T, station, source_kind = load_edi(
        args.edi,
        prefer_spectra=args.prefer_spectra,
        ref=args.ref,
        rotate_deg=args.rotate,
        fill_invalid=args.fill_invalid,
    )

    rec = _as_rec(freq, Z, T)

    if args.csv is not None:
        export_csv(args.csv, rec, include_phase_tensor=not args.no_pt)

    if args.hdf5 is not None:
        export_hdf5(args.hdf5, freq=freq, Z=Z, T=T)

    if args.npz is not None:
        export_npz(args.npz, station=station, source_kind=source_kind, freq=freq, Z=Z, T=T)

    # Plotting (if requested)
    if args.plot_dir is not None and any([args.plot_rho, args.plot_phase, args.plot_tipper, args.plot_pt]):
        from edi_viz import dataframe_from_arrays, plot_rho, plot_phase, plot_tipper, plot_pt
        df = dataframe_from_arrays(freq, Z, T)
        if args.plot_rho is not None:
            plot_rho(df, station, args.plot_dir, comps=args.plot_rho)
        if args.plot_phase is not None:
            plot_phase(df, station, args.plot_dir, comps=args.plot_phase)
        if args.plot_tipper:
            plot_tipper(df, station, args.plot_dir)
        if args.plot_pt:
            plot_pt(df, station, args.plot_dir)

    # Minimal console summary
    print(f"Station: {{station}} | Source: {{source_kind}} | Nfreq={{freq.size}} | CSV={{args.csv is not None}} HDF5={{args.hdf5 is not None}} NPZ={{args.npz is not None}}" )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
