#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
edi_writer.py
=============
Minimal EDI writer for MT transfer functions.

Writes ASCII EDI with blocks: >FREQ, >Z??R/I (+ optional .VAR), optional >TX*/>TY* (+ optional .VAR), and >PT?? (+ optional .VAR).
Compatible with the reader implemented in `edi_processor.py`.

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-11-11 17:53:13 UTC
"""
from __future__ import annotations

import io
from typing import Optional, Dict
from pathlib import Path
import numpy as np



def phase_tensor(Z: np.ndarray) -> np.ndarray:
    \"\"\"Compute Phase Tensor Φ = Im(Z) @ inv(Re(Z)); uses pinv if singular.\"\"\"
    X = Z.real
    Y = Z.imag
    try:
        Xinv = np.linalg.inv(X)
    except np.linalg.LinAlgError:
        Xinv = np.linalg.pinv(X)
    return Y @ Xinv  # real 2x2


def _fmt_block(values: np.ndarray, per_line: int = 6, fmt: str = "{: .8E}") -> str:
    v = np.asarray(values).ravel()
    out_lines = []
    for i in range(0, v.size, per_line):
        chunk = v[i:i+per_line]
        out_lines.append(" ".join(fmt.format(float(x)) for x in chunk))
    return "\n".join(out_lines)


def _ensure_1d(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1-D (got shape {x.shape})")
    return x


def write_edi(path: Path | str, *, station: str, freq: np.ndarray, Z: np.ndarray,
              T: Optional[np.ndarray] = None, lat_deg: Optional[float] = None,
              lon_deg: Optional[float] = None, elev_m: Optional[float] = None,
              header_meta: Optional[Dict[str, str]] = None, numbers_per_line: int = 6) -> str:
    """Write an EDI file with Z/T blocks."""
    path = Path(path)
    freq = _ensure_1d(np.asarray(freq, dtype=float), "freq")
    Z = np.asarray(Z)
    if Z.shape != (freq.size, 2, 2):
        raise ValueError(f"Z must have shape (n,2,2) with n={freq.size}, got {Z.shape}")
    if T is not None:
        T = np.asarray(T)
        if T.shape != (freq.size, 1, 2):
            raise ValueError(f"T must have shape (n,1,2) with n={freq.size}, got {T.shape}")
    # Optional variances validation: expect same shapes as components
    if Z_var_re is not None:
        Z_var_re = np.asarray(Z_var_re)
        if Z_var_re.shape != (freq.size, 2, 2):
            raise ValueError(f"Z_var_re must have shape (n,2,2), got {Z_var_re.shape}")
    if Z_var_im is not None:
        Z_var_im = np.asarray(Z_var_im)
        if Z_var_im.shape != (freq.size, 2, 2):
            raise ValueError(f"Z_var_im must have shape (n,2,2), got {Z_var_im.shape}")
    if T is not None:
        if T_var_re is not None:
            T_var_re = np.asarray(T_var_re)
            if T_var_re.shape != (freq.size, 1, 2):
                raise ValueError(f"T_var_re must have shape (n,1,2), got {T_var_re.shape}")
        if T_var_im is not None:
            T_var_im = np.asarray(T_var_im)
            if T_var_im.shape != (freq.size, 1, 2):
                raise ValueError(f"T_var_im must have shape (n,1,2), got {T_var_im.shape}")
    # Phase Tensor: compute if not provided
    if PT is None:
        PT = np.zeros((freq.size, 2, 2), dtype=float)
        for i in range(freq.size):
            PT[i] = phase_tensor(Z[i])
    else:
        PT = np.asarray(PT)
        if PT.shape != (freq.size, 2, 2):
            raise ValueError(f"PT must have shape (n,2,2), got {PT.shape}")
    if PT_var is not None:
        PT_var = np.asarray(PT_var)
        if PT_var.shape != (freq.size, 2, 2):
            raise ValueError(f"PT_var must have shape (n,2,2), got {PT_var.shape}")


    Zxx = Z[:,0,0]; Zxy = Z[:,0,1]; Zyx = Z[:,1,0]; Zyy = Z[:,1,1]
    blocks = {
        "ZXXR": Zxx.real, "ZXXI": Zxx.imag,
        "ZXYR": Zxy.real, "ZXYI": Zxy.imag,
        "ZYXR": Zyx.real, "ZYXI": Zyx.imag,
        "ZYYR": Zyy.real, "ZYYI": Zyy.imag,
    }
    # Optional variance blocks for Z real/imag
    if Z_var_re is not None:
        blocks.update({"ZXXR.VAR": Z_var_re[:,0,0], "ZXYR.VAR": Z_var_re[:,0,1], "ZYXR.VAR": Z_var_re[:,1,0], "ZYYR.VAR": Z_var_re[:,1,1]})
    if Z_var_im is not None:
        blocks.update({"ZXXI.VAR": Z_var_im[:,0,0], "ZXYI.VAR": Z_var_im[:,0,1], "ZYXI.VAR": Z_var_im[:,1,0], "ZYYI.VAR": Z_var_im[:,1,1]})
    if T is not None:
        Tx, Ty = T[:,0,0], T[:,0,1]
        blocks.update({"TXR": Tx.real, "TXI": Tx.imag, "TYR": Ty.real, "TYI": Ty.imag})
        if T_var_re is not None:
            blocks.update({"TXR.VAR": T_var_re[:,0,0], "TYR.VAR": T_var_re[:,0,1]})
        if T_var_im is not None:
            blocks.update({"TXI.VAR": T_var_im[:,0,0], "TYI.VAR": T_var_im[:,0,1]})
    # Phase Tensor real components
    blocks.update({"PTXX": PT[:,0,0], "PTXY": PT[:,0,1], "PTYX": PT[:,1,0], "PTYY": PT[:,1,1]})
    if PT_var is not None:
        blocks.update({"PTXX.VAR": PT_var[:,0,0], "PTXY.VAR": PT_var[:,0,1], "PTYX.VAR": PT_var[:,1,0], "PTYY.VAR": PT_var[:,1,1]})

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
        for k,v in header_meta.items():
            buf.write(f"    {k}= {v}\n")

    buf.write(">FREQ\n")
    buf.write(_fmt_block(freq, per_line=numbers_per_line)); buf.write("\n")

    for tag in ["FREQ","ZXXR","ZXXI","ZXYR","ZXYI","ZYXR","ZYXI","ZYYR","ZYYI","ZXXR.VAR","ZXXI.VAR","ZXYR.VAR","ZXYI.VAR","ZYXR.VAR","ZYXI.VAR","ZYYR.VAR","ZYYI.VAR","TXR","TXI","TYR","TYI","TXR.VAR","TXI.VAR","TYR.VAR","TYI.VAR","PTXX","PTXY","PTYX","PTYY","PTXX.VAR","PTXY.VAR","PTYX.VAR","PTYY.VAR"]:
        if tag not in blocks:
            continue
        buf.write(f">{tag}\n")
        buf.write(_fmt_block(blocks[tag], per_line=numbers_per_line)); buf.write("\n")

    buf.write(">END\n")
    path.write_text(buf.getvalue(), encoding="utf-8")
    return str(path)


def write_edi_from_npz(npz_path: Path | str, out_path: Optional[Path | str] = None, *,
                       numbers_per_line: int = 6, lat_deg: Optional[float] = None,
                       lon_deg: Optional[float] = None, elev_m: Optional[float] = None) -> str:
    """Write an EDI file directly from an NPZ bundle created by `edi_processor.py --npz`."""
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    station = str(data["station"])
    freq = data["freq"]
    Z = data["Z"]
    T = data["T"] if "T" in data.files else None
    if out_path is None:
        out_path = npz_path.with_suffix("").with_name(f"{station}.edi")
    return write_edi(
        out_path,
        station=station,
        freq=freq,
        Z=Z,
        T=T,
        lat_deg=lat_deg, lon_deg=lon_deg, elev_m=elev_m,
        header_meta={
            "SOURCE_FILE": str(npz_path.name),
            "SOURCE_KIND": str(data["source_kind"]) if "source_kind" in data.files else "unknown",
            "REF": str(data["ref"]) if "ref" in data.files else "",
            "ROTATE_DEG": f"{float(data['rotate_deg'])}" if "rotate_deg" in data.files else "",
            "PREFER_SPECTRA": f"{bool(data['prefer_spectra'])}" if "prefer_spectra" in data.files else "",
        },
        numbers_per_line=numbers_per_line,
    )


def _build_cli():
    import argparse
    ap = argparse.ArgumentParser(description="Write an MT EDI file from arrays or NPZ bundles.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_npz = sub.add_parser("from-npz", help="Convert <station>_TF.npz to <station>.edi")
    p_npz.add_argument("npz", type=Path, help="Path to NPZ bundle (from edi_processor.py)")
    p_npz.add_argument("--out", type=Path, default=None, help="Output EDI path")
    p_npz.add_argument("--per-line", type=int, default=6, help="Numbers per line in value blocks")
    p_npz.add_argument("--lat", type=float, default=None, help="Station latitude (deg)")
    p_npz.add_argument("--lon", type=float, default=None, help="Station longitude (deg)")
    p_npz.add_argument("--elev", type=float, default=None, help="Station elevation (m)")

    p_dir = sub.add_parser("direct", help="Write EDI from provided *.npy paths (freq,Z, optional T)")
    p_dir.add_argument("--out", type=Path, required=True, help="Output EDI path")
    p_dir.add_argument("--station", type=str, required=True)
    p_dir.add_argument("--freq", type=Path, required=True, help="Path to 1-D .npy file with freq [Hz]")
    p_dir.add_argument("--Z", type=Path, required=True, help="Path to (n,2,2) complex .npy file")
    p_dir.add_argument("--T", type=Path, default=None, help="Optional path to (n,1,2) complex .npy file")
    p_dir.add_argument("--per-line", type=int, default=6, help="Numbers per line in value blocks")
    p_dir.add_argument("--lat", type=float, default=None, help="Station latitude (deg)")
    p_dir.add_argument("--lon", type=float, default=None, help="Station longitude (deg)")
    p_dir.add_argument("--elev", type=float, default=None, help="Station elevation (m)")

    return ap

def main(argv=None):
    ap = _build_cli()
    args = ap.parse_args(argv)
    if args.cmd == "from-npz":
        out = write_edi_from_npz(
            args.npz, args.out,
            numbers_per_line=args.per_line,
            lat_deg=args.lat, lon_deg=args.lon, elev_m=args.elev
        )
        print(f"Wrote {out}")
    elif args.cmd == "direct":
        freq = np.load(args.freq)
        Z = np.load(args.Z)
        T = np.load(args.T) if args.T is not None else None
        out = write_edi(
            args.out,
            station=args.station,
            freq=freq,
            Z=Z,
            T=T,
            lat_deg=args.lat, lon_deg=args.lon, elev_m=args.elev,
            numbers_per_line=args.per_line,
        )
        print(f"Wrote {out}")

if __name__ == "__main__":
    main()
