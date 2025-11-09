#!/usr/bin/env python3
"""
edi_to_csv.py — Batch convert MT EDI files to a single CSV or HDF5 (no MTpy required)

Created: 2025-11-09
Created by: ChatGPT (GPT-5)

Features
--------
- Parses impedance (Zxx, Zxy, Zyx, Zyy)
- Parses tipper (Tx, Ty)
- Parses phase tensor (PTXX, PTXY, PTYX, PTYY) if present
- Optionally derives Phase Tensor from Z: Φ = (Re Z)^{-1} · Im Z
- Computes apparent resistivity and phase for Z (optional)
- Adds a per-row processing timestamp (UTC): processed_utc
- Invalid numeric replacement within valid frequencies: --fill-invalid VALUE
- NEW: HDF5 output via --out-h5 (Behavior B: write ONLY HDF5 if provided; otherwise CSV)

Usage
-----
    # CSV
    python edi_to_csv.py /path/to/edis --out mt_data.csv --rho-phase --pt-from-z --fill-invalid 1e-30

    # HDF5 only
    python edi_to_csv.py /path/to/edis --out-h5 mt_data.h5 --rho-phase --pt-from-z --fill-invalid 1e-30

HDF5 Layout
-----------
/data_table             : tabular dataset with the same columns as CSV
attrs:
  tool                  : "edi_to_csv.py"
  created_by            : "ChatGPT (GPT-5)"
  created_utc           : ISO-8601 timestamp
  notes                 : brief description

Output table columns
--------------------
processed_utc, file, station, latitude, longitude, elevation_m,
frequency_hz, period_s, component, re, im, abs, phase_deg, rho_ohm_m, group
"""

import argparse, csv, math, os, re, datetime
from typing import Dict, List, Optional, Tuple

MU0 = 4e-7 * math.pi
FLOAT_RE = r'[+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?'


# ---------- Helper utilities ----------
def _is_continuation(line: str) -> bool:
    s = line.strip()
    if not s or '=' in s or s.startswith('>'):
        return False
    return re.search(FLOAT_RE, s) is not None

def _collect_numbers_from_line(line: str) -> List[float]:
    return [float(x) for x in re.findall(FLOAT_RE, line)]

def _parse_scalar(pattern: str, text: str) -> Optional[str]:
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1).strip() if m else None

def _parse_vector(name: str, text_lines: List[str], expected_len: Optional[int]) -> Optional[List[float]]:
    import re as _re
    regex = _re.compile(rf'\\b{name}\\s*=\\s*(.*)', _re.IGNORECASE)
    for i, line in enumerate(text_lines):
        m = regex.search(line)
        if m:
            vals = _collect_numbers_from_line(m.group(1))
            j = i + 1
            while j < len(text_lines) and _is_continuation(text_lines[j]):
                vals.extend(_collect_numbers_from_line(text_lines[j]))
                j += 1
            if expected_len is not None:
                vals = vals[:expected_len]
            return vals
    return None


# ---------- EDI parsing ----------
def parse_edi(path: str) -> Dict:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    lines = text.splitlines()

    nf_str = _parse_scalar(r'\\bNFREQ\\s*=\\s*(\\d+)', text)
    nf = int(nf_str) if nf_str else None

    freq = _parse_vector('FREQ', lines, nf)
    if not freq:
        per = _parse_vector('PERIOD', lines, nf)
        if per:
            freq = [1.0 / p if p != 0 else float('nan') for p in per]
            nf = len(freq)

    station = _parse_scalar(r'\\b(?:STATION|SITE)\\s*=\\s*([^\\r\\n]+)', text)
    lat = _parse_scalar(r'\\bLAT(?:ITUDE)?\\s*=\\s*(' + FLOAT_RE + r')', text)
    lon = _parse_scalar(r'\\bLON(?:G|GITUDE)?\\s*=\\s*(' + FLOAT_RE + r')', text)
    elev = _parse_scalar(r'\\bELEV(?:ATION)?\\s*=\\s*(' + FLOAT_RE + r')', text)

    comps = {}
    if nf:
        # Impedance Z
        for comp in ["ZXX", "ZXY", "ZYX", "ZYY"]:
            r, i = _parse_vector(comp + "R", lines, nf), _parse_vector(comp + "I", lines, nf)
            if r and i: comps[comp] = (r, i)
        # Tipper T
        for comp in ["TX", "TY"]:
            r, i = _parse_vector(comp + "R", lines, nf), _parse_vector(comp + "I", lines, nf)
            if r and i: comps[comp] = (r, i)
        # Phase Tensor PT
        for comp in ["PTXX", "PTXY", "PTYX", "PTYY"]:
            r, i = _parse_vector(comp + "R", lines, nf), _parse_vector(comp + "I", lines, nf)
            if r and i: comps[comp] = (r, i)

    return {
        "nfreq": nf,
        "freq": freq,
        "station": station.strip() if station else "",
        "lat": float(lat) if lat else None,
        "lon": float(lon) if lon else None,
        "elev": float(elev) if elev else None,
        "components": comps,
    }


# ---------- Derived quantities ----------
def rho_phase_from_Z(re_v: float, im_v: float, freq_hz: float) -> Tuple[float, float]:
    mag2 = re_v * re_v + im_v * im_v
    if not (freq_hz and freq_hz > 0):
        rho = float("nan")
    else:
        rho = mag2 / (MU0 * 2.0 * math.pi * freq_hz)
    phase = math.degrees(math.atan2(im_v, re_v))
    return rho, phase

def derive_phase_tensor_from_Z(components: Dict[str, Tuple[List[float], List[float]]]) -> Optional[Dict[str, Tuple[List[float], List[float]]]]:
    """Return PTXX..PTYY derived from Z components as Φ = (Re Z)^(-1) · Im Z.
    If Z is missing or singular at a frequency, fill with NaN.
    Φ is real, so imaginary parts are zero arrays.
    """
    needed = ["ZXX","ZXY","ZYX","ZYY"]
    if not all(k in components for k in needed):
        return None
    ZXXr, ZXXi = components["ZXX"]
    ZXYr, ZXYi = components["ZXY"]
    ZYXr, ZYXi = components["ZYX"]
    ZYYr, ZYYi = components["ZYY"]
    n = min(len(ZXXr), len(ZXYr), len(ZYXr), len(ZYYr))
    PT = {k: ([], []) for k in ["PTXX","PTXY","PTYX","PTYY"]}
    for i in range(n):
        X00, X01 = ZXXr[i], ZXYr[i]
        X10, X11 = ZYXr[i], ZYYr[i]
        Y00, Y01 = ZXXi[i], ZXYi[i]
        Y10, Y11 = ZYXi[i], ZYYi[i]
        det = X00*X11 - X01*X10
        if det == 0 or not math.isfinite(det):
            pxx = pxy = pyx = pyy = float("nan")
        else:
            invX00 =  X11/det
            invX01 = -X01/det
            invX10 = -X10/det
            invX11 =  X00/det
            # Φ = X^{-1} · Y
            pxx = invX00*Y00 + invX01*Y10
            pxy = invX00*Y01 + invX01*Y11
            pyx = invX10*Y00 + invX11*Y10
            pyy = invX10*Y01 + invX11*Y11
        # PT is real; set imaginary parts to zero
        PT["PTXX"][0].append(pxx); PT["PTXX"][1].append(0.0)
        PT["PTXY"][0].append(pxy); PT["PTXY"][1].append(0.0)
        PT["PTYX"][0].append(pyx); PT["PTYX"][1].append(0.0)
        PT["PTYY"][0].append(pyy); PT["PTYY"][1].append(0.0)
    return PT


# ---------- CSV/HDF5 output ----------
def write_csv(rows: List[Dict], out_path: str) -> None:
    fields = [
        "processed_utc",
        "file", "station", "latitude", "longitude", "elevation_m",
        "frequency_hz", "period_s", "component",
        "re", "im", "abs", "phase_deg", "rho_ohm_m", "group"
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def write_h5(rows: List[Dict], out_path: str) -> None:
    import h5py, numpy as np
    # Define dtypes (variable-length strings for text columns)
    vlen_str = h5py.special_dtype(vlen=str)
    # Build structured array
    def to_tuple(r):
        return (
            r["processed_utc"],
            r["file"],
            r["station"],
            r["latitude"] if r["latitude"] is not None else np.nan,
            r["longitude"] if r["longitude"] is not None else np.nan,
            r["elevation_m"] if r["elevation_m"] is not None else np.nan,
            r["frequency_hz"],
            r["period_s"],
            r["component"],
            r["re"], r["im"], r["abs"], r["phase_deg"], r["rho_ohm_m"],
            r["group"],
        )
    data_tuples = [to_tuple(r) for r in rows]
    dtype = np.dtype([
        ("processed_utc", vlen_str),
        ("file",         vlen_str),
        ("station",      vlen_str),
        ("latitude",     "f8"),
        ("longitude",    "f8"),
        ("elevation_m",  "f8"),
        ("frequency_hz", "f8"),
        ("period_s",     "f8"),
        ("component",    vlen_str),
        ("re",           "f8"),
        ("im",           "f8"),
        ("abs",          "f8"),
        ("phase_deg",    "f8"),
        ("rho_ohm_m",    "f8"),
        ("group",        vlen_str),
    ])
    arr = np.array(data_tuples, dtype=dtype)

    with h5py.File(out_path, "w") as h5:
        ds = h5.create_dataset("/data_table", data=arr, compression="gzip", compression_opts=4, shuffle=True)
        # File-level attributes
        h5.attrs["tool"] = "edi_to_csv.py"
        h5.attrs["created_by"] = "ChatGPT (GPT-5)"
        h5.attrs["created_utc"] = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
        h5.attrs["notes"] = "Magnetotelluric transfer-function rows; same columns as CSV"

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Batch convert MT EDI files to CSV or HDF5 (with PT, no MTpy).")
    ap.add_argument("input", help="EDI file or directory")
    ap.add_argument("--out", default="edi_data.csv", help="Output CSV filename")
    ap.add_argument("--out-h5", default=None, help="Output HDF5 filename; if set, CSV is skipped")
    ap.add_argument("--rho-phase", action="store_true", help="Compute rho_a and phase for impedance components")
    ap.add_argument("--pt-from-z", action="store_true", help="Derive Phase Tensor from Z as Φ=(Re Z)^(-1)·Im Z (fills PT if missing).")
    ap.add_argument("--fill-invalid", type=float, default=None,
                    help="Replace NaN/±Inf numeric values within valid frequencies with this constant (e.g., 1e-30).")
    args = ap.parse_args()

    # Collect EDI files
    edi_files = []
    if os.path.isdir(args.input):
        for r, _, fs in os.walk(args.input):
            for fn in fs:
                if fn.lower().endswith(".edi"):
                    edi_files.append(os.path.join(r, fn))
    else:
        edi_files.append(args.input)
    if not edi_files:
        raise SystemExit("No EDI files found.")

    timestamp = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    def _fill(x: float) -> float:
        if args.fill_invalid is None:
            return x
        return x if (x is not None and math.isfinite(x)) else float(args.fill_invalid)

    all_rows = []
    for fp in sorted(edi_files):
        try:
            ed = parse_edi(fp)
        except Exception as e:
            print(f"[WARN] Failed to parse {fp}: {e}")
            continue

        # Derive PT if requested or missing
        have_pt = all(k in ed["components"] for k in ["PTXX","PTXY","PTYX","PTYY"])
        if args.pt_from_z or not have_pt:
            derived = derive_phase_tensor_from_Z(ed["components"])
            if derived:
                if args.pt_from_z or not have_pt:
                    for k, v in derived.items():
                        ed["components"][k] = v

        freq = ed["freq"] or []
        nf = len(freq)
        for comp, (rvec, ivec) in ed["components"].items():
            group = ("tipper" if comp in ["TX", "TY"]
                     else "phasetensor" if comp.startswith("PT")
                     else "impedance")
            for i in range(min(nf, len(rvec), len(ivec))):
                fr = freq[i]
                per = 1.0 / fr if fr and fr > 0 else float("nan")
                re_v, im_v = rvec[i], ivec[i]
                mag = math.sqrt(re_v ** 2 + im_v ** 2)
                phase = math.degrees(math.atan2(im_v, re_v))
                rho = float("nan")

                if args.rho_phase and group == "impedance":
                    rho, phase = rho_phase_from_Z(re_v, im_v, fr)

                # Apply invalid fill ONLY within valid indices
                fr_o = _fill(fr)
                per_o = _fill(per)
                re_o = _fill(re_v)
                im_o = _fill(im_v)
                abs_o = _fill(mag)
                ph_o = _fill(phase)
                rho_o = _fill(rho)

                all_rows.append({
                    "processed_utc": timestamp,
                    "file": os.path.basename(fp),
                    "station": ed["station"],
                    "latitude": ed["lat"],
                    "longitude": ed["lon"],
                    "elevation_m": ed["elev"],
                    "frequency_hz": fr_o,
                    "period_s": per_o,
                    "component": comp,
                    "re": re_o,
                    "im": im_o,
                    "abs": abs_o,
                    "phase_deg": ph_o,
                    "rho_ohm_m": rho_o,
                    "group": group,
                })

    # Output: Behavior B
    if args.out_h5:
        write_h5(all_rows, args.out_h5)
        print(f"Wrote HDF5 to {args.out_h5} with {len(all_rows)} rows (/data_table)")
    else:
        write_csv(all_rows, args.out)
        print(f"Wrote CSV to {args.out} with {len(all_rows)} rows")


if __name__ == "__main__":
    main()
