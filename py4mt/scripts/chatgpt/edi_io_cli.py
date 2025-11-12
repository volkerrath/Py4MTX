#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
edi_io_cli.py
================
Standalone CLI wrapper around `edi_io.py` helpers.

Subcommands
-----------
- show <file.npz>
- dump-meta <file.json>
- to-csv <file.npz> [--out <file.csv>]

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-11-10 20:33:04 UTC
"""
from __future__ import annotations

import argparse
from pathlib import Path
from edi_io import load_site_npz, load_meta_json

def main(argv=None) -> None:
    """Entry point for the CLI wrapper.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; defaults to sys.argv[1:].
    """
    ap = argparse.ArgumentParser(description="CLI for loading/inspecting NPZ/JSON artifacts.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_show = sub.add_parser("show", help="Print a concise summary of an NPZ site archive")
    p_show.add_argument("npz", type=Path)

    p_dump = sub.add_parser("dump-meta", help="Pretty-print a JSON metadata file")
    p_dump.add_argument("json", type=Path)

    p_csv = sub.add_parser("to-csv", help="Write a wide CSV from an NPZ site archive")
    p_csv.add_argument("npz", type=Path)
    p_csv.add_argument("--out", type=Path, default=None, help="Output CSV path (default: <npz>_wide.csv)")

    args = ap.parse_args(argv)

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
        import pprint
        meta = load_meta_json(str(args.json))
        pprint.pprint(meta)

    elif args.cmd == "to-csv":
        from edi_io import load_site_npz as _load, SiteTF
        import pandas as pd
        import numpy as np
        site = _load(str(args.npz))
        rows = []
        for i, f in enumerate(site.freq):
            Z = site.Z[i]
            T = site.T[i] if site.T is not None and len(site.T.shape) == 3 else np.array([[np.nan, np.nan]], dtype=np.complex128)
            PT = site.PT[i]
            rho = site.rho[i] if site.rho is not None else np.full(4, np.nan)
            phi = site.phase_deg[i] if site.phase_deg is not None else np.full(4, np.nan)
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
        df = pd.DataFrame(rows).sort_values("freq_Hz", ascending=False).reset_index(drop=True)
        out = args.out if args.out is not None else args.npz.with_suffix("").with_name(args.npz.stem + "_wide.csv")
        df.to_csv(out, index=False)
        print(f"Wrote {out}")

if __name__ == "__main__":
    main()
