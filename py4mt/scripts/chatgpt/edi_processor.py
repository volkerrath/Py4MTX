#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
edi_processor.py — Unified EDI processor (standard Z/T + Phoenix SPECTRA) with optional plotting.

This tool parses an EDI file and outputs CSV/HDF5 with Z, T, apparent resistivity,
phase, and the phase tensor. It supports Phoenix SPECTRA reconstruction as well as
tabulated Z/T blocks, plus optional PNG plots (ρ, φ, tipper).

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-11-10 17:15:08 UTC
"""

import re
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MU0 = 4e-7 * np.pi

def rot2(theta_rad: float) -> np.ndarray:
    """Return 2×2 rotation matrix for angle θ (radians, CCW)."""
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s],[s, c]], dtype=np.float64)

def rotate_ZT(Z: np.ndarray, T: np.ndarray, theta_deg: float):
    """Rotate impedance and tipper by θ degrees CCW (E/H frame)."""
    if Z is None and T is None: return None, None
    th = np.deg2rad(theta_deg)
    R = rot2(th); Rm = rot2(-th)
    Zp = (R @ Z @ Rm) if Z is not None else None
    Tp = (T @ Rm) if T is not None else None
    return Zp, Tp

def rho_phase_from_Z(Z: np.ndarray, f: float):
    """Return (rho[4], phase_deg[4]) for [xx, xy, yx, yy] at frequency f."""
    comps = [Z[0,0], Z[0,1], Z[1,0], Z[1,1]]
    rho = np.array([np.abs(z)**2/(MU0*2*np.pi*f) for z in comps], dtype=float)
    phi = np.degrees(np.array([np.arctan2(z.imag, z.real) for z in comps], dtype=float))
    return rho, phi

def phase_tensor(Z: np.ndarray):
    """Compute Phase Tensor Φ = Im(Z) @ inv(Re(Z)); use pinv if necessary."""
    X = Z.real
    Y = Z.imag
    try:
        Xinv = np.linalg.inv(X)
    except np.linalg.LinAlgError:
        Xinv = np.linalg.pinv(X)
    return Y @ Xinv  # real 2×2

def fill_invalid_inplace(a: np.ndarray, fill_value: float):
    """Replace non-finite values in real/imag parts of complex array with fill_value."""
    if a is None: return
    if np.iscomplexobj(a):
        re = a.real.copy(); im = a.imag.copy()
        re[~np.isfinite(re)] = fill_value
        im[~np.isfinite(im)] = fill_value
        a[:] = re + 1j*im
    else:
        a[~np.isfinite(a)] = fill_value

# ---------- SPECTRA path (Phoenix 7×7) ----------
COMP_LIST = ['hx','hy','hz','ex','ey','rhx','rhy']
IDX = {c:i for i,c in enumerate(COMP_LIST)}

def parse_spectra_blocks(edi_text: str):
    """Yield (freq_Hz, avgt, mat7x7_real) for each >SPECTRA block."""
    for m in re.finditer(r'>SPECTRA[^\n]*\n((?:[^\n]*\n)+?)(?=>SPECTRA|>END|$)', edi_text):
        header = m.group(0).splitlines()[0]
        body = m.group(1)
        fm = re.search(r'FREQ\s*=\s*([0-9.]+[ED][+\-]?\d+|[0-9.]+)', header, flags=re.IGNORECASE)
        if not fm: continue
        f = float(fm.group(1).replace('D','E'))
        am = re.search(r'AVGT\s*=\s*([0-9.]+[ED][+\-]?\d+|[0-9.]+)', header, flags=re.IGNORECASE)
        avgt = float(am.group(1).replace('D','E')) if am else np.nan
        rows = [ln for ln in body.splitlines() if ln.strip()]
        arr = np.array([
            list(map(float, re.findall(r'[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[EeDd][+\-]?\d+)?', ln)))
            for ln in rows
        ])
        if arr.shape == (7,7):
            yield f, avgt, arr

def reconstruct_S_phoenix(mat7: np.ndarray) -> np.ndarray:
    """diag autos; lower=Re; upper=Im; return complex Hermitian S (7×7)."""
    S = np.zeros((7,7), dtype=np.complex128)
    for i in range(7):
        S[i,i] = mat7[i,i]
        for j in range(i+1,7):
            Re = mat7[j,i]; Im = mat7[i,j]
            S[i,j] = Re + 1j*Im
            S[j,i] = Re - 1j*Im
    return S

def ZT_from_S(S: np.ndarray, ref: str='RH'):
    """Compute Z (2×2) and T (1×2) from spectral S using chosen H reference."""
    if ref.upper() == 'H':
        h1, h2 = IDX['hx'], IDX['hy']
    else:
        h1, h2 = IDX['rhx'], IDX['rhy']
    ex, ey, hz = IDX['ex'], IDX['ey'], IDX['hz']
    SHH = np.array([[S[h1,h1], S[h1,h2]],[S[h2,h1], S[h2,h2]]], dtype=np.complex128)
    SEH = np.array([[S[ex,h1], S[ex,h2]],[S[ey,h1], S[ey,h2]]], dtype=np.complex128)
    SBH = np.array([[S[hz,h1], S[hz,h2]]], dtype=np.complex128)
    try:
        SHH_inv = np.linalg.inv(SHH)
    except np.linalg.LinAlgError:
        SHH_inv = np.linalg.pinv(SHH)
    Z = SEH @ SHH_inv
    T = SBH @ SHH_inv
    return Z, T

# ---------- Standard EDI Z/T path ----------
def parse_block_values(edi_text: str):
    """Parse arrays like FREQ, ZXXR/ZXXI, ..., TX,TY if present."""
    f_matches = re.findall(r'FREQ\s*=\s*([0-9.]+[ED][+\-]?\d+|[0-9.]+)', edi_text, flags=re.IGNORECASE)
    if not f_matches: return None
    freqs = np.array([float(s.replace('D','E')) for s in f_matches], dtype=float)
    n = freqs.size
    def get_arr(tag):
        pat = rf'>{tag}[^\n]*\n((?:[^\n]*\n)+?)(?=>[A-Z]|>END|$)'
        m = re.search(pat, edi_text, flags=re.IGNORECASE)
        if not m: return None
        nums = re.findall(r'[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[EeDd][+\-]?\d+)?', m.group(1))
        arr = np.array([float(v.replace('D','E')) for v in nums], dtype=float)
        return arr[:n] if arr.size>=n else None
    Z = np.zeros((n,2,2), dtype=np.complex128)
    ok = False
    for c,(i,j) in {'ZXX':(0,0),'ZXY':(0,1),'ZYX':(1,0),'ZYY':(1,1)}.items():
        re_arr = get_arr(c+'R') or get_arr(c+'.RE') or get_arr(c+'_RE')
        im_arr = get_arr(c+'I') or get_arr(c+'.IM') or get_arr(c+'_IM')
        if re_arr is not None and im_arr is not None:
            ok = True
            Z[:,i,j] = re_arr + 1j*im_arr
    if not ok: return None
    T = np.zeros((n,1,2), dtype=np.complex128)
    txr = get_arr('TXR') or get_arr('TX.RE') or get_arr('TX_RE')
    txi = get_arr('TXI') or get_arr('TX.IM') or get_arr('TX_IM')
    tyr = get_arr('TYR') or get_arr('TY.RE') or get_arr('TY_RE')
    tyi = get_arr('TYI') or get_arr('TY.IM') or get_arr('TY_IM')
    if txr is not None and txi is not None: T[:,0,0] = txr + 1j*txi
    if tyr is not None and tyi is not None: T[:,0,1] = tyr + 1j*tyi
    return freqs, Z, T



def _parse_comps(comp_str: str):
    """Parse comma-separated comps into a validated list among ['xx','xy','yx','yy']."""
    allowed = ["xx","xy","yx","yy"]
    comps = [c.strip().lower() for c in (comp_str or "").split(",") if c.strip()]
    comps = [c for c in comps if c in allowed]
    return comps if comps else ["xy","yx"]

def plot_rho(df, station: str, outdir: Path, comps=None):
    """Plot apparent resistivity vs period for selected components; one chart per component."""
    comps = _parse_comps(comps)
    period = 1.0 / df["freq_Hz"].to_numpy()
    for c in comps:
        key = f"rho_{c}"
        if key not in df.columns:
            continue
        fig = plt.figure()
        plt.loglog(period, df[key].to_numpy(), marker="o", linestyle="-")
        plt.gca().invert_xaxis()
        plt.xlabel("Period (s)")
        plt.ylabel(f"Apparent resistivity ρ_{c} (Ω·m)")
        plt.title(f"{station} — ρ_{c}")
        fig.savefig(outdir / f"{station}_rho_{c}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

def plot_phase(df, station: str, outdir: Path, comps=None):
    """Plot phase vs period for selected components; one chart per component."""
    comps = _parse_comps(comps)
    period = 1.0 / df["freq_Hz"].to_numpy()
    for c in comps:
        key = f"phi_{c}_deg"
        if key not in df.columns:
            continue
        fig = plt.figure()
        plt.semilogx(period, df[key].to_numpy(), marker="o", linestyle="-")
        plt.gca().invert_xaxis()
        plt.xlabel("Period (s)")
        plt.ylabel(f"Phase φ_{c} (deg)")
        plt.title(f"{station} — φ_{c}")
        fig.savefig(outdir / f"{station}_phi_{c}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

def plot_pt(df, station: str, outdir: Path):
    """Plot Phase Tensor components vs period as a single line plot (PTxx, PTxy, PTyx, PTyy)."""
    import numpy as np
    period = 1.0 / df["freq_Hz"].to_numpy()
    if "ptxx_re" not in df.columns:
        return
    ptxx = df["ptxx_re"].to_numpy()
    ptxy = df["ptxy_re"].to_numpy()
    ptyx = df["ptyx_re"].to_numpy()
    ptyy = df["ptyy_re"].to_numpy()

    fig = plt.figure()
    plt.semilogx(period, ptxx, marker="o", linestyle="-", label="PTxx")
    plt.semilogx(period, ptxy, marker="^", linestyle="-", label="PTxy")
    plt.semilogx(period, ptyx, marker="s", linestyle="-", label="PTyx")
    plt.semilogx(period, ptyy, marker="d", linestyle="-", label="PTyy")
    plt.gca().invert_xaxis()
    plt.xlabel("Period (s)")
    plt.ylabel("Phase Tensor entries (dimensionless)")
    plt.title(f"{station} — Phase Tensor Φ components")
    plt.legend()
    fig.savefig(outdir / f"{station}_PT_components.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
# ---------- Plotting helpers ----------
def _ensure_plot_dir(plot_dir: Path) -> Path:
    """Ensure plot directory exists; return the absolute path."""
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir.resolve()

def plot_rho(df, station: str, outdir: Path):
    """Plot apparent resistivity vs period for XY and YX components (separate charts)."""
    period = 1.0 / df["freq_Hz"].to_numpy()
    fig = plt.figure()
    plt.loglog(period, df["rho_xy"].to_numpy(), marker="o", linestyle="-", label="rho_xy")
    plt.gca().invert_xaxis()
    plt.xlabel("Period (s)")
    plt.ylabel("Apparent resistivity (Ω·m)")
    plt.title(f"{station} — ρ_xy")
    plt.legend()
    fig.savefig(outdir / f"{station}_rho_xy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure()
    plt.loglog(period, df["rho_yx"].to_numpy(), marker="o", linestyle="-", label="rho_yx")
    plt.gca().invert_xaxis()
    plt.xlabel("Period (s)")
    plt.ylabel("Apparent resistivity (Ω·m)")
    plt.title(f"{station} — ρ_yx")
    plt.legend()
    fig.savefig(outdir / f"{station}_rho_yx.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_phase(df, station: str, outdir: Path):
    """Plot phase vs period for XY and YX (separate charts)."""
    period = 1.0 / df["freq_Hz"].to_numpy()
    fig = plt.figure()
    plt.semilogx(period, df["phi_xy_deg"].to_numpy(), marker="o", linestyle="-", label="phi_xy")
    plt.gca().invert_xaxis()
    plt.xlabel("Period (s)")
    plt.ylabel("Phase (deg)")
    plt.title(f"{station} — φ_xy")
    plt.legend()
    fig.savefig(outdir / f"{station}_phi_xy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure()
    plt.semilogx(period, df["phi_yx_deg"].to_numpy(), marker="o", linestyle="-", label="phi_yx")
    plt.gca().invert_xaxis()
    plt.xlabel("Period (s)")
    plt.ylabel("Phase (deg)")
    plt.title(f"{station} — φ_yx")
    plt.legend()
    fig.savefig(outdir / f"{station}_phi_yx.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_tipper(df, station: str, outdir: Path):
    """Plot tipper magnitude and argument for Tx and Ty (two charts)."""
    import numpy as np
    period = 1.0 / df["freq_Hz"].to_numpy()
    if "tx_re" in df and "ty_re" in df:
        tx = (df["tx_re"].to_numpy() + 1j*df["tx_im"].to_numpy())
        ty = (df["ty_re"].to_numpy() + 1j*df["ty_im"].to_numpy())
        fig = plt.figure()
        plt.semilogx(period, np.abs(tx), marker="o", linestyle="-", label="|Tx|")
        plt.semilogx(period, np.abs(ty), marker="^", linestyle="-", label="|Ty|")
        plt.gca().invert_xaxis()
        plt.xlabel("Period (s)")
        plt.ylabel("Tipper magnitude")
        plt.title(f"{station} — |T|")
        plt.legend()
        fig.savefig(outdir / f"{station}_tipper_amp.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure()
        plt.semilogx(period, np.degrees(np.angle(tx)), marker="o", linestyle="-", label="arg(Tx)")
        plt.semilogx(period, np.degrees(np.angle(ty)), marker="^", linestyle="-", label="arg(Ty)")
        plt.gca().invert_xaxis()
        plt.xlabel("Period (s)")
        plt.ylabel("Tipper arg (deg)")
        plt.title(f"{station} — arg(T)")
        plt.legend()
        fig.savefig(outdir / f"{station}_tipper_arg.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

def run(in_edi: Path, out_base: Path, write_csv: bool, write_hdf5: bool,
        rotate_deg: float, fill_invalid: float, prefer_spectra: bool, ref: str, long: bool,
        do_plot: bool=False, plot_dir: Path=None, plot_kind: str="all", plot_comps: str="xy,yx"):
    text = in_edi.read_text(encoding='latin-1', errors='ignore')
    m = re.search(r'DATAID\s*=\s*"?([A-Za-z0-9_\-\.]+)"?', text, flags=re.IGNORECASE)
    station = m.group(1) if m else 'UNKNOWN'
    rows = []
    h5_payload = None

    spectra_blocks = list(parse_spectra_blocks(text))
    if prefer_spectra and spectra_blocks:
        spectra_blocks.sort(key=lambda x: x[0], reverse=True)
        n = len(spectra_blocks)
        freqs = np.array([b[0] for b in spectra_blocks], dtype=float)
        Z_all = np.zeros((n,2,2), dtype=np.complex128)
        T_all = np.zeros((n,1,2), dtype=np.complex128)
        rho_all = np.zeros((n,4), dtype=float)
        phi_all = np.zeros((n,4), dtype=float)
        PT_all = np.zeros((n,2,2), dtype=float)
        for k,(f, avgt, mat7) in enumerate(spectra_blocks):
            S = reconstruct_S_phoenix(mat7)
            Z, T = ZT_from_S(S, ref=ref)
            if rotate_deg: Z, T = rotate_ZT(Z, T, rotate_deg)
            if fill_invalid is not None:
                fill_invalid_inplace(Z, fill_invalid)
                fill_invalid_inplace(T, fill_invalid)
            rho, phi = rho_phase_from_Z(Z, f)
            PT = phase_tensor(Z)
            Z_all[k] = Z; T_all[k] = T; rho_all[k] = rho; phi_all[k] = phi; PT_all[k] = PT
        for i,f in enumerate(freqs):
            Z=Z_all[i]; T=T_all[i]; rho=rho_all[i]; phi=phi_all[i]; PT=PT_all[i]
            rows.append({
                'freq_Hz': f,
                'zxx_re': Z[0,0].real, 'zxx_im': Z[0,0].imag,
                'zxy_re': Z[0,1].real, 'zxy_im': Z[0,1].imag,
                'zyx_re': Z[1,0].real, 'zyx_im': Z[1,0].imag,
                'zyy_re': Z[1,1].real, 'zyy_im': Z[1,1].imag,
                'rho_xx': rho[0], 'phi_xx_deg': phi[0],
                'rho_xy': rho[1], 'phi_xy_deg': phi[1],
                'rho_yx': rho[2], 'phi_yx_deg': phi[2],
                'rho_yy': rho[3], 'phi_yy_deg': phi[3],
                'tx_re': T[0,0].real, 'tx_im': T[0,0].imag,
                'ty_re': T[0,1].real, 'ty_im': T[0,1].imag,
                'ptxx_re': PT[0,0], 'ptxx_im': 0.0,
                'ptxy_re': PT[0,1], 'ptxy_im': 0.0,
                'ptyx_re': PT[1,0], 'ptyx_im': 0.0,
                'ptyy_re': PT[1,1], 'ptyy_im': 0.0,
            })
        h5_payload = ('spectra', station, freqs, Z_all, T_all, rho_all, phi_all, PT_all)
    else:
        parsed = parse_block_values(text)
        if parsed is None:
            raise RuntimeError("Could not find SPECTRA blocks or standard Z/T tables in EDI.")
        freqs, Z_all, T_all = parsed
        n = freqs.size
        rho_all = np.zeros((n,4), dtype=float)
        phi_all = np.zeros((n,4), dtype=float)
        PT_all = np.zeros((n,2,2), dtype=float)
        for i in range(n):
            Z = Z_all[i]; T = T_all[i]
            if rotate_deg: Z, T = rotate_ZT(Z, T, rotate_deg)
            if fill_invalid is not None:
                fill_invalid_inplace(Z, fill_invalid)
                fill_invalid_inplace(T, fill_invalid)
            rho, phi = rho_phase_from_Z(Z, freqs[i])
            PT = phase_tensor(Z)
            rho_all[i]=rho; phi_all[i]=phi; PT_all[i]=PT
        for i,f in enumerate(freqs):
            Z=Z_all[i]; T=T_all[i]; rho=rho_all[i]; phi=phi_all[i]; PT=PT_all[i]
            rows.append({
                'freq_Hz': f,
                'zxx_re': Z[0,0].real, 'zxx_im': Z[0,0].imag,
                'zxy_re': Z[0,1].real, 'zxy_im': Z[0,1].imag,
                'zyx_re': Z[1,0].real, 'zyx_im': Z[1,0].imag,
                'zyy_re': Z[1,1].real, 'zyy_im': Z[1,1].imag,
                'rho_xx': rho[0], 'phi_xx_deg': phi[0],
                'rho_xy': rho[1], 'phi_xy_deg': phi[1],
                'rho_yx': rho[2], 'phi_yx_deg': phi[2],
                'rho_yy': rho[3], 'phi_yy_deg': phi[3],
                'tx_re': T[0,0].real if T is not None else np.nan,
                'tx_im': T[0,0].imag if T is not None else np.nan,
                'ty_re': T[0,1].real if T is not None else np.nan,
                'ty_im': T[0,1].imag if T is not None else np.nan,
                'ptxx_re': PT[0,0], 'ptxx_im': 0.0,
                'ptxy_re': PT[0,1], 'ptxy_im': 0.0,
                'ptyx_re': PT[1,0], 'ptyx_im': 0.0,
                'ptyy_re': PT[1,1], 'ptyy_im': 0.0,
            })
        h5_payload = ('tables', station, freqs, Z_all, T_all, rho_all, phi_all, PT_all)

    df = pd.DataFrame(rows).sort_values('freq_Hz', ascending=False).reset_index(drop=True)

    # Optional plotting
    if do_plot:
        outdir = _ensure_plot_dir(plot_dir if plot_dir is not None else out_base.parent)
        kind = (plot_kind or "all").lower()
        if kind in ("all","rho"):
            plot_rho(df, station, outdir)
        if kind in ("all","phase"):
            plot_phase(df, station, outdir)
        if kind in ("all","tipper"):
            plot_tipper(df, station, outdir)

    csv_path = out_base.with_name(out_base.name + "_TF.csv")
    h5_path = out_base.with_name(out_base.name + "_TF.h5")

    if write_csv:
        header = [
            f"# EDI processor | Source: {in_edi.name}",
            f"# Station: {station} | Rotate: {rotate_deg:.3f} deg | Prefer SPECTRA: {prefer_spectra} | Ref: {ref}",
            "# Columns include Z (re/im), rho/phi, T (re/im), and Phase Tensor (PTxx..).",
            f"# Created by ChatGPT (GPT-5 Thinking) on 2025-11-10 17:15:08 UTC",
        ]
        with open(csv_path, 'w') as f:
            f.write('\\n'.join(header)+'\\n')
        df.to_csv(csv_path, mode='a', index=False, float_format='%.8e')

    if write_hdf5:
        import h5py
        kind, station, freqs, Z_all, T_all, rho_all, phi_all, PT_all = h5_payload
        with h5py.File(h5_path, 'w') as h5:
            g = h5.create_group('data')
            g.create_dataset('freq', data=freqs)
            g.create_dataset('Z', data=Z_all)
            g.create_dataset('T', data=T_all)
            g.create_dataset('rho', data=rho_all)
            g.create_dataset('phase_deg', data=phi_all)
            g.create_dataset('PT', data=PT_all)
            g.attrs['source_kind'] = kind
            h5.attrs['station'] = station
            h5.attrs['source_file'] = str(in_edi)
            h5.attrs['created_by'] = "ChatGPT (GPT-5 Thinking)"
            h5.attrs['created_utc'] = "2025-11-10 17:15:08 UTC"
            h5.attrs['note'] = "Z/T and Phase Tensor exported from EDI."

    if long:
        recs = []
        for _,r in df.iterrows():
            f = r['freq_Hz']
            for comp in ['zxx','zxy','zyx','zyy','tx','ty','ptxx','ptxy','ptyx','ptyy']:
                recs.append({'freq_Hz': f, 'component': comp+'_re', 'value': r[comp+'_re']})
                recs.append({'freq_Hz': f, 'component': comp+'_im', 'value': r[comp+'_im']})
            for comp in ['rho_xx','rho_xy','rho_yx','rho_yy','phi_xx_deg','phi_xy_deg','phi_yx_deg','phi_yy_deg']:
                recs.append({'freq_Hz': f, 'component': comp, 'value': r[comp]})
        dfl = pd.DataFrame(recs).sort_values(['freq_Hz','component'], ascending=[False, True]).reset_index(drop=True)
        long_path = out_base.with_name(out_base.name + "_TF_long.csv")
        dfl.to_csv(long_path, index=False, float_format='%.8e')

    return csv_path if write_csv else None, h5_path if write_hdf5 else None

def main(argv=None):
    ap = argparse.ArgumentParser(description="Process EDI (tables or Phoenix spectra) to CSV/HDF5 with Z, T, ρ/φ, Phase Tensor, and optional plots.")
    ap.add_argument("edi", type=Path, help="Path to input EDI file")
    ap.add_argument("--out-base", type=Path, default=None, help="Output base path (no suffix). Defaults to <station>")
    ap.add_argument("--csv", action="store_true", help="Write CSV")
    ap.add_argument("--hdf5", action="store_true", help="Write HDF5")
    ap.add_argument("--rotate-deg", type=float, default=0.0, help="Rotate E/H frame by θ degrees (CCW)")
    ap.add_argument("--fill-invalid", type=float, default=None, help="Fill non-finite entries with this constant (e.g., 1e-30)")
    ap.add_argument("--prefer-spectra", action="store_true", help="If SPECTRA present, prefer spectra-reconstructed Z/T")
    ap.add_argument("--ref", choices=["H","RH"], default="RH", help="H reference for SPECTRA path (default RH)")
    ap.add_argument("--long", action="store_true", help="Also write a tidy long-format CSV with a 'component' column")
    ap.add_argument("--plot", action="store_true", help="Generate PNG plots (rho, phase, tipper)")
    ap.add_argument("--plot-dir", type=Path, default=None, help="Directory to save plots (default: output base directory)")
    ap.add_argument("--plot-kind", choices=["all","rho","phase","tipper"], default="all", help="Which plots to generate")
    ap.add_argument("--plot-comps", type=str, default="xy,yx", help="Impedance components to plot for ρ/φ (comma-separated subset of xx,xy,yx,yy)")

    edi_text = args.edi.read_text(encoding='latin-1', errors='ignore')
    m = re.search(r'DATAID\s*=\s*"?([A-Za-z0-9_\-\.]+)"?', edi_text, flags=re.IGNORECASE)
    station = m.group(1) if m else 'UNKNOWN'
    out_base = args.out_base if args.out_base is not None else args.edi.with_name(station)

    csv_path, h5_path = run(
        in_edi=args.edi,
        out_base=out_base,
        write_csv=args.csv or (not args.hdf5),
        write_hdf5=args.hdf5 or (not args.csv),
        rotate_deg=args.rotate_deg,
        fill_invalid=args.fill_invalid,
        prefer_spectra=args.prefer_spectra,
        ref=args.ref,
        long=args.long,
        do_plot=args.plot,
        plot_dir=args.plot_dir,
        plot_kind=args.plot_kind,
        plot_comps=args.plot_comps,
    )
    if csv_path: print(f"CSV written: {csv_path}")
    if h5_path: print(f"HDF5 written: {h5_path}")

if __name__ == "__main__":
    main()
