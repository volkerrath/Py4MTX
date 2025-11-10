#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_io.py
============
I/O helpers for EDI transfer-function artifacts written by `edi_processor.py`.

Contents
--------
- SiteTF dataclass (container for arrays + meta)
- load_site_npz(path) → SiteTF
- save_site_npz(site, path, *, rotate_deg, prefer_spectra, ref, source_file, source_kind) → str
- load_meta_json(path) → dict

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-11-10 20:33:04 UTC
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Union
import json
import numpy as np


@dataclass
class SiteTF:
    """Container for a site's transfer-function arrays and metadata.

    Parameters
    ----------
    station : str
        Station identifier.
    freq : numpy.ndarray
        Frequencies [Hz], shape (n,).
    Z : numpy.ndarray
        Impedance tensor, complex128, shape (n, 2, 2).
    T : numpy.ndarray
        Tipper, complex128, shape (n, 1, 2).
    rho : numpy.ndarray
        Apparent resistivities, float64, shape (n, 4) for (xx, xy, yx, yy).
    phase_deg : numpy.ndarray
        Phases in degrees, float64, shape (n, 4) for (xx, xy, yx, yy).
    PT : numpy.ndarray
        Phase Tensor (real), float64, shape (n, 2, 2).
    meta : dict
        Additional metadata (rotate_deg, prefer_spectra, ref, source_file, source_kind).
    """
    station: str
    freq: np.ndarray
    Z: np.ndarray
    T: np.ndarray
    rho: np.ndarray
    phase_deg: np.ndarray
    PT: np.ndarray
    meta: Dict[str, Any]


def load_site_npz(path: Union[str, bytes]) -> SiteTF:
    """Load a site NPZ archive produced by `edi_processor.py --npz`.

    Parameters
    ----------
    path : str or bytes
        Path to the .npz archive.

    Returns
    -------
    SiteTF
        Dataclass with arrays and the meta dict (basic fields parsed to native types).
    """
    data = np.load(path, allow_pickle=True)
    meta = {
        "station": str(data["station"]),
        "source_file": str(data["source_file"]),
        "source_kind": str(data["source_kind"]),
        "rotate_deg": float(data["rotate_deg"]),
        "prefer_spectra": bool(data["prefer_spectra"]),
        "ref": str(data["ref"]),
    }
    return SiteTF(
        station=meta["station"],
        freq=data["freq"],
        Z=data["Z"],
        T=data["T"],
        rho=data["rho"],
        phase_deg=data["phase_deg"],
        PT=data["PT"],
        meta=meta,
    )


def save_site_npz(site: SiteTF, path: Union[str, bytes], *, rotate_deg: float = 0.0,
                  prefer_spectra: bool = False, ref: str = "RH",
                  source_file: Optional[str] = None, source_kind: str = "unknown") -> str:
    """Save a SiteTF container to a .npz compatible with `edi_processor.py`.

    Parameters
    ----------
    site : SiteTF
        Dataclass instance (station, freq, Z, T, rho, phase_deg, PT).
    path : str or bytes
        Output file path ending with .npz.
    rotate_deg : float, optional
        Rotation applied to E/H frame when the data were computed (metadata only).
    prefer_spectra : bool, optional
        Whether Phoenix SPECTRA path was preferred (metadata only).
    ref : {"H","RH"}, optional
        Reference used in spectra path (metadata only).
    source_file : str, optional
        Original source file, if known.
    source_kind : str, optional
        "spectra" or "tables" or "unknown".

    Returns
    -------
    str
        The path written.
    """
    np.savez(
        path,
        station=site.station,
        source_file=source_file or "",
        source_kind=source_kind,
        freq=site.freq,
        Z=site.Z,
        T=site.T,
        rho=site.rho,
        phase_deg=site.phase_deg,
        PT=site.PT,
        rotate_deg=float(rotate_deg),
        prefer_spectra=bool(prefer_spectra),
        ref=str(ref),
    )
    return str(path)


def load_meta_json(path: Union[str, bytes]) -> Dict[str, Any]:
    """Load the JSON metadata written by `edi_processor.py --json`.

    Parameters
    ----------
    path : str or bytes
        Path to the JSON metadata file (e.g., '<out>_TF_meta.json').

    Returns
    -------
    dict
        Parsed metadata dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse
    import pandas as pd
    from pathlib import Path

    ap = argparse.ArgumentParser(description="CLI helpers for NPZ/JSON artifacts.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_show = sub.add_parser("show", help="Print a concise summary of an NPZ site archive")
    p_show.add_argument("npz", type=Path)

    p_dump = sub.add_parser("dump-meta", help="Pretty-print a JSON metadata file")
    p_dump.add_argument("json", type=Path)

    p_csv = sub.add_parser("to-csv", help="Write a wide CSV from an NPZ site archive")
    p_csv.add_argument("npz", type=Path)
    p_csv.add_argument("--out", type=Path, default=None, help="Output CSV path (default: <npz>_wide.csv)")

    args = ap.parse_args()

    if args.cmd == "show":
        site = load_site_npz(str(args.npz))
        n = int(site.freq.shape[0])
        fmin = float(site.freq.min()) if n else float("nan")
        fmax = float(site.freq.max()) if n else float("nan")
        print(f"Station: {site.station}")
        print(f"Frequencies: n={n}, fmin={fmin:.6g} Hz, fmax={fmax:.6g} Hz")
        print(f"Z shape: {site.Z.shape}, T shape: {site.T.shape}, PT shape: {site.PT.shape}")
        print(f"Source: kind={site.meta.get('source_kind')}, file={site.meta.get('source_file')}")
        print(f"Meta: rotate_deg={site.meta.get('rotate_deg')}, ref={site.meta.get('ref')}, prefer_spectra={site.meta.get('prefer_spectra')}")

    elif args.cmd == "dump-meta":
        meta = load_meta_json(str(args.json))
        import pprint
        pprint.pprint(meta)

    elif args.cmd == "to-csv":
        site = load_site_npz(str(args.npz))
        rows = []
        for i, f in enumerate(site.freq):
            Z = site.Z[i]
            import numpy as _np
            T = site.T[i] if site.T is not None and len(site.T.shape) == 3 else _np.array([[_np.nan, _np.nan]], dtype=_np.complex128)
            PT = site.PT[i]
            rho = site.rho[i] if site.rho is not None else _np.full(4, _np.nan)
            phi = site.phase_deg[i] if site.phase_deg is not None else _np.full(4, _np.nan)
            rows.append({
                "freq_Hz": float(f),
                "zxx_re": float(Z[0,0].real), "zxx_im": float(Z[0,0].imag),
                "zxy_re": float(Z[0,1].real), "zxy_im": float(Z[0,1].imag),
                "zyx_re": float(Z[1,0].real), "zyx_im": float(Z[1,0].imag),
                "zyy_re": float(Z[1,1].real), "zyy_im": float(Z[1,1].imag),
                "rho_xx": float(rho[0]), "phi_xx_deg": float(phi[0]),
                "rho_xy": float(rho[1]), "phi_xy_deg": float(phi[1]),
                "rho_yx": float(rho[2]), "phi_yx_deg": float(phi[2]),
                "rho_yy": float(rho[3]), "phi_yy_deg": float(phi[3]),
                "tx_re": float(T[0,0].real), "tx_im": float(T[0,0].imag),
                "ty_re": float(T[0,1].real), "ty_im": float(T[0,1].imag),
                "ptxx_re": float(PT[0,0]), "ptxx_im": 0.0,
                "ptxy_re": float(PT[0,1]), "ptxy_im": 0.0,
                "ptyx_re": float(PT[1,0]), "ptyx_im": 0.0,
                "ptyy_re": float(PT[1,1]), "ptyy_im": 0.0,
            })
        import pandas as pd
        df = pd.DataFrame(rows).sort_values("freq_Hz", ascending=False).reset_index(drop=True)
        out = args.out if args.out is not None else args.npz.with_suffix("").with_name(args.npz.stem + "_wide.csv")
        df.to_csv(out, index=False)
        print(f"Wrote {out}")
