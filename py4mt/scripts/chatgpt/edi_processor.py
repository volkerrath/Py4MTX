#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
edi_to_csv.py — Unify EDI→CSV/HDF conversion (standard Z/T blocks + Phoenix SPECTRA).

This command-line utility reads an EDI file and produces CSV and/or HDF5 outputs
with impedance tensor **Z**, tipper **T**, derived apparent resistivity/phase,
and the **Phase Tensor** Φ = Im(Z) @ inv(Re(Z)). It supports two data sources:

1) Phoenix **SPECTRA** blocks:
   - Reconstruct complex spectral matrix S (7×7) per user convention:
       • diagonal = autospectra (real)
       • lower triangle = Re{cross-spectra}
       • upper triangle = Im{cross-spectra}
   - Compute Z = S_EH @ inv(S_HH), T = S_BH @ inv(S_HH).

   - Potentially important: conjugation -see m-file below:

        function[output_spec] = spec2spec(input_spec)
                % Spectra data comes into the script as [nch x nch] matrix
                fspec = input_spec;
                nch = size(fspec,1);

                % Rearrange spectra for mtpy methods
                mspec = nan(size(fspec));
                for i = 1:nch
                    for j = i:nch
                        if i==j
                            mspec(i,j) = fspec(i,j);
                        else
                            % complex conjugation of the original entries
                            mspec(i,j) = fspec(j,i)-1i*fspec(i,j);
                            % keep complex conjugated entries in the lower
                            % triangular matrix:
                            mspec(j,i) = fspec(j,i)+1i*fspec(i,j);
                        end
                    end
                end
                output_spec = mspec;
            end




2) Standard EDI **impedance/tipper tables** (ZXX,ZXY,ZYX,ZYY per frequency, optional TX,TY):
   - Parse values directly from the EDI.
   - Compute Phase Tensor and derived quantities.

Outputs include wide CSV columns and optionally a tidy "long" format with a
`component` column (Zxx_re, Zxx_im, etc., including PTxx..).

Usage
-----
$ python edi_to_csv.py INPUT.edi \
    --out-base OUTPUT_PREFIX \
    --csv --hdf5 \
    --ref RH \
    --fill-invalid 1e-30 \
    --rotate-deg 0 \
    --long

Notes
-----
- Apparent resistivity: ρ = |Z|² / (μ0·2πf); phase φ = atan2(Im Z, Re Z) in degrees.
- Phase Tensor Φ = Im(Z) @ inv(Re(Z)) is **real 2×2**; emitted as re/im pairs (im≈0).
- If S_HH is singular (spectra path) or Re(Z) is singular (phase tensor path), a
  pseudo-inverse is used.
- Rotation applies as: Z' = R(θ)·Z·R(-θ), T' = T·R(-θ).

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-11-09 11:36:19 UTC
"""

import re
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

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

def run(in_edi: Path, out_base: Path, write_csv: bool, write_hdf5: bool,
        rotate_deg: float, fill_invalid: float, prefer_spectra: bool, ref: str, long: bool):
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

    csv_path = out_base.with_name(out_base.name + "_TF.csv")
    h5_path = out_base.with_name(out_base.name + "_TF.h5")

    if write_csv:
        header = [
            f"# EDI to CSV by edi_to_csv.py | Source: {in_edi.name}",
            f"# Station: {station} | Rotate: {rotate_deg:.3f} deg | Prefer SPECTRA: {prefer_spectra} | Ref: {ref}",
            "# Columns include Z (re/im), rho/phi, T (re/im), and Phase Tensor (PTxx..).",
            f"# Created by ChatGPT (GPT-5 Thinking) on 2025-11-09 11:36:19 UTC",
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
            h5.attrs['created_utc'] = "2025-11-09 11:36:19 UTC"
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
    ap = argparse.ArgumentParser(description="Convert EDI (standard or Phoenix spectra) to CSV/HDF5 with Z, T, ρ/φ, Phase Tensor.")
    ap.add_argument("edi", type=Path, help="Path to input EDI file")
    ap.add_argument("--out-base", type=Path, default=None, help="Output base path (no suffix). Defaults to <station>")
    ap.add_argument("--csv", action="store_true", help="Write CSV")
    ap.add_argument("--hdf5", action="store_true", help="Write HDF5")
    ap.add_argument("--rotate-deg", type=float, default=0.0, help="Rotate E/H frame by θ degrees (CCW)")
    ap.add_argument("--fill-invalid", type=float, default=None, help="Fill non-finite entries with this constant (e.g., 1e-30)")
    ap.add_argument("--prefer-spectra", action="store_true", help="If SPECTRA present, prefer spectra-reconstructed Z/T")
    ap.add_argument("--ref", choices=["H","RH"], default="RH", help="H reference for SPECTRA path (default RH)")
    ap.add_argument("--long", action="store_true", help="Also write a tidy long-format CSV with a 'component' column")
    args = ap.parse_args(argv)

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
    )
    if csv_path: print(f"CSV written: {csv_path}")
    if h5_path: print(f"HDF5 written: {h5_path}")

if __name__ == "__main__":
    main()
