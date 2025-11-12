#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MT time-series I/O (Metronix ADU and Phoenix MTU) with instrument TF handling.

This module provides:
- Robust readers for Metronix ADU-08/07 ATS files (with paired XML)
- Reader for Phoenix MTU raw int32 channels (with paired XML/.log metadata)
- Calibration/transfer-function loaders for common Metronix and Phoenix formats
- Frequency-domain deconvolution to physical units (nT for H, mV/km for E)
- Export to NPZ/HDF5/CSV with compact metadata sidecar (JSON)
- Simple CLI to batch process a directory of channels

Dependencies
------------
numpy, scipy (signal, interpolate), pandas (optional), h5py (optional)
Optional: obspy (I/O conveniences), matplotlib (quick plots), pyyaml (config)

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-11-11
"""
from __future__ import annotations

import os
import re
import json
import warnings
import datetime as _dt
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, Iterable

import numpy as np

# Optional scientific stack
try:
    from scipy import signal, interpolate  # type: ignore
except Exception:  # pragma: no cover
    signal = None
    interpolate = None

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None

# Optionals for plotting / configs (best-effort)
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


@dataclass
class ChannelMeta:
    """
    Metadata for a single MT channel.

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    orientation : str
        Channel code, e.g., 'Ex', 'Ey', 'Hx', 'Hy', 'Hz'.
    start_time : Optional[str]
        ISO8601 start time if known.
    location : Optional[Tuple[float, float, float]]
        (latitude, longitude, elevation_m) if available.
    adc_bits : Optional[int]
        ADC nominal bit depth (e.g., 24).
    adc_range_vpp : Optional[float]
        ADC input full-scale, Volts peak-to-peak.
    preamp_gain : float
        Linear gain of front-end prior to digitization (dimensionless).
    sensor_sensitivity : Optional[float]
        For coils: V/(nT * Hz) or V/nT@1Hz conventions vary; see sensor_tf.
    sensor_model : Optional[str]
        Free text (e.g., 'MFS-06e', 'MTC-50').
    notes : dict
        Extra fields as parsed from vendor XML/logs.
    """
    fs: float
    orientation: str
    start_time: Optional[str] = None
    location: Optional[Tuple[float, float, float]] = None
    adc_bits: Optional[int] = None
    adc_range_vpp: Optional[float] = None
    preamp_gain: float = 1.0
    sensor_sensitivity: Optional[float] = None
    sensor_model: Optional[str] = None
    notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TimeSeries:
    """
    In-memory time series with metadata.

    Parameters
    ----------
    data : np.ndarray
        1-D array of samples (float64). May initially be counts before deconvolution.
    meta : ChannelMeta
        Channel metadata.
    units : str
        Units string for `data`. Common: 'counts', 'V', 'nT', 'mV/km'.
    """
    data: np.ndarray
    meta: ChannelMeta
    units: str = "counts"


@dataclass
class TransferFunction:
    """
    Instrument transfer function (sensor + analog chain if embedded).

    Parameters
    ----------
    f : np.ndarray
        Frequency vector (Hz), strictly positive, sorted ascending.
    H : np.ndarray
        Complex response evaluated at `f`, with convention:
        output_voltage(f) = H(f) * physical_quantity(f)
        Example: For coils, H has units V/nT (or V/(nT*Hz) depending on spec).
    meta : dict
        Free-form details (sensor model, serial, notes).
    """
    f: np.ndarray
    H: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)


def _read_text(path: str) -> str:
    """
    Read a text file with universal newlines.

    Parameters
    ----------
    path : str
        Path to text file.

    Returns
    -------
    str
        File content.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _find_neighbor(path: str, exts: Iterable[str]) -> Optional[str]:
    """
    Find a sibling file next to `path` with any extension in `exts`.

    Parameters
    ----------
    path : str
        Reference file path.
    exts : Iterable[str]
        Candidate extensions (e.g., [".xml", ".log"]).

    Returns
    -------
    Optional[str]
        First match if any; else None.
    """
    base, _ = os.path.splitext(path)
    for e in exts:
        cand = base + e
        if os.path.isfile(cand):
            return cand
    return None


def _interp_complex(f_src: np.ndarray, H_src: np.ndarray, f_tgt: np.ndarray) -> np.ndarray:
    """
    Interpolate a complex spectrum magnitude/phase robustly.

    Parameters
    ----------
    f_src : np.ndarray
        Source frequencies (Hz), >0 ascending.
    H_src : np.ndarray
        Complex response at f_src.
    f_tgt : np.ndarray
        Target frequencies (Hz).

    Returns
    -------
    np.ndarray
        Complex response evaluated at f_tgt.

    Notes
    -----
    Interpolates log|H| and unwrap(phase) over log f for smoothness.
    """
    if interpolate is None:
        raise RuntimeError("scipy.interpolate not available")

    f_src = np.asarray(f_src, float)
    H_src = np.asarray(H_src, complex)
    f_tgt = np.asarray(f_tgt, float)

    if np.any(f_src <= 0) or np.any(f_tgt <= 0):
        raise ValueError("Frequencies must be > 0")

    mag = np.log(np.abs(H_src) + 1e-300)
    pha = np.unwrap(np.angle(H_src))
    xsrc = np.log(f_src)

    f_mag = interpolate.interp1d(xsrc, mag, kind="linear", fill_value="extrapolate")
    f_pha = interpolate.interp1d(xsrc, pha, kind="linear", fill_value="extrapolate")
    xtgt = np.log(f_tgt)

    mag_t = np.exp(f_mag(xtgt))
    pha_t = f_pha(xtgt)
    return mag_t * np.exp(1j * pha_t)


def read_metronix_ats(ats_path: str, xml_path: Optional[str] = None) -> TimeSeries:
    """
    Read a Metronix ADU-08/07 ATS channel with paired XML header.

    Parameters
    ----------
    ats_path : str
        Path to `.ats` binary file (int32 samples).
    xml_path : Optional[str]
        Path to channel `.xml` (same basename). If None, autodetect.

    Returns
    -------
    TimeSeries
        Time series (float64) in raw counts, with metadata.

    Notes
    -----
    Scaling to Volts needs adc_bits and adc_range_vpp from XML.
    """
    if not os.path.isfile(ats_path):
        raise FileNotFoundError(ats_path)

    if xml_path is None:
        xml_path = _find_neighbor(ats_path, [".xml", "_ch.xml", "_xml"])

    counts = np.fromfile(ats_path, dtype="<i4")

    fs = None
    adc_bits = None
    adc_range_vpp = None
    preamp_gain = 1.0
    start_time = None
    orientation = "UNKNOWN"
    sensor_model = None
    loc = None
    notes: Dict[str, Any] = {}

    if xml_path and os.path.isfile(xml_path):
        xml = _read_text(xml_path)

        def _tag(pattern: str, cast=None, default=None):
            m = re.search(pattern, xml, flags=re.IGNORECASE)
            if not m:
                return default
            val = m.group(1).strip()
            return cast(val) if cast else val

        fs = _tag(r"<samplingrate[^>]*>([\d\.Ee\+\-]+)</samplingrate>", float, None) or \
             _tag(r"<sampleRate[^>]*>([\d\.Ee\+\-]+)</sampleRate>", float, None)

        adc_bits = _tag(r"<adcBits[^>]*>(\d+)</adcBits>", int, None)
        adc_range_vpp = _tag(r"<adcRangeP2P[^>]*>([\d\.Ee\+\-]+)</adcRangeP2P>", float, None)
        preamp_gain = _tag(r"<preampGain[^>]*>([\d\.Ee\+\-]+)</preampGain>", float, 1.0)

        start_time = _tag(r"<startTime[^>]*>([^<]+)</startTime>", str, None)

        orientation = _tag(r"<orientation[^>]*>([^<]+)</orientation>", str, orientation)
        orientation = _tag(r"<sensorType[^>]*>([^<]+)</sensorType>", str, orientation)

        sensor_model = _tag(r"<sensorModel[^>]*>([^<]+)</sensorModel>", str, None)

        lat = _tag(r"<latitude[^>]*>([\d\.\+\-Ee]+)</latitude>", float, None)
        lon = _tag(r"<longitude[^>]*>([\d\.\+\-Ee]+)</longitude>", float, None)
        elev = _tag(r"<elevation[^>]*>([\d\.\+\-Ee]+)</elevation>", float, None)
        if lat is not None and lon is not None:
            loc = (lat, lon, elev if elev is not None else 0.0)

        notes["xml_path"] = xml_path
    else:
        warnings.warn("Metronix XML not found; metadata will be minimal.")

    if fs is None:
        raise ValueError("Sampling rate not found in Metronix XML.")

    meta = ChannelMeta(
        fs=float(fs),
        orientation=orientation,
        start_time=start_time,
        location=loc,
        adc_bits=adc_bits,
        adc_range_vpp=adc_range_vpp,
        preamp_gain=float(preamp_gain),
        sensor_sensitivity=None,
        sensor_model=sensor_model,
        notes=notes,
    )
    return TimeSeries(data=counts.astype(np.float64), meta=meta, units="counts")


def read_phoenix_mtu(raw_path: str, xml_or_log_path: Optional[str] = None) -> TimeSeries:
    """
    Read Phoenix MTU raw int32 channel with metadata from .xml or .log.

    Parameters
    ----------
    raw_path : str
        Path to raw int32 file (typical MTU export per channel).
    xml_or_log_path : Optional[str]
        Companion .xml or .log file. If None, autodetect.

    Returns
    -------
    TimeSeries
        Time series (float64) in counts with parsed metadata where possible.

    Notes
    -----
    If sampling rate is missing, raises an error.
    """
    if not os.path.isfile(raw_path):
        raise FileNotFoundError(raw_path)

    counts = np.fromfile(raw_path, dtype="<i4")

    if xml_or_log_path is None:
        xml_or_log_path = _find_neighbor(raw_path, [".xml", ".XML", ".log", ".LOG"])

    fs = None
    adc_bits = None
    adc_range_vpp = None
    preamp_gain = 1.0
    start_time = None
    orientation = "UNKNOWN"
    sensor_model = None
    loc = None
    notes: Dict[str, Any] = {}

    if xml_or_log_path and os.path.isfile(xml_or_log_path):
        txt = _read_text(xml_or_log_path)

        def _grab(regex: str, cast=None, default=None):
            m = re.search(regex, txt, flags=re.IGNORECASE)
            if not m:
                return default
            val = m.group(1).strip()
            return cast(val) if cast else val

        fs = _grab(r"<samplingRate[^>]*>([\d\.Ee\+\-]+)</samplingRate>", float, fs) or \
             _grab(r"Sampling\s*Rate\s*[:=]\s*([\d\.Ee\+\-]+)", float, None)

        adc_bits = _grab(r"(?:ADC\s*Bits|adcBits)\s*[:=]\s*(\d+)", int, adc_bits)
        adc_range_vpp = _grab(r"(?:ADC\s*Range|adcRange)\s*(?:P2P)?\s*[:=]\s*([\d\.Ee\+\-]+)", float, adc_range_vpp)
        preamp_gain = _grab(r"(?:Preamp\s*Gain|preampGain)\s*[:=]\s*([\d\.Ee\+\-]+)", float, preamp_gain)

        start_time = _grab(r"(?:Start\s*Time|startTime)\s*[:=]\s*([^\r\n<]+)", str, start_time)

        orientation = _grab(r"(?:Channel|orientation|sensorType)\s*[:=]\s*([A-Za-z]{2})", str, orientation)
        sensor_model = _grab(r"(?:Sensor\s*Model|sensorModel)\s*[:=]\s*([^\r\n<]+)", str, sensor_model)

        lat = _grab(r"(?:Latitude|latitude)\s*[:=]\s*([\d\.\+\-Ee]+)", float, None)
        lon = _grab(r"(?:Longitude|longitude)\s*[:=]\s*([\d\.\+\-Ee]+)", float, None)
        elev = _grab(r"(?:Elevation|elevation)\s*[:=]\s*([\d\.\+\-Ee]+)", float, None)
        if lat is not None and lon is not None:
            loc = (lat, lon, elev if elev is not None else 0.0)

        notes["info_path"] = xml_or_log_path
    else:
        warnings.warn("Phoenix XML/LOG not found; metadata will be minimal.")

    if fs is None:
        raise ValueError("Sampling rate not found for Phoenix channel.")

    meta = ChannelMeta(
        fs=float(fs),
        orientation=orientation,
        start_time=start_time,
        location=loc,
        adc_bits=adc_bits,
        adc_range_vpp=adc_range_vpp,
        preamp_gain=float(preamp_gain),
        sensor_sensitivity=None,
        sensor_model=sensor_model,
        notes=notes,
    )
    return TimeSeries(data=counts.astype(np.float64), meta=meta, units="counts")


def load_tf_metronix_sensor(sensor_file: str) -> TransferFunction:
    """
    Load Metronix sensor TF from XML or CSV (freq, amp, phase).

    Parameters
    ----------
    sensor_file : str
        Path to sensor calibration file (e.g., MFS-06e XML).

    Returns
    -------
    TransferFunction
        Complex response H(f) on tabulated frequencies.

    Notes
    -----
    Amplitude often given in V/(nT*Hz); phase in degrees.
    """
    text = _read_text(sensor_file)
    model = None

    if sensor_file.lower().endswith(".xml"):
        f_vals = re.findall(r"<Frequency[^>]*>([\d\.\+\-Ee]+)</Frequency>", text, flags=re.I)
        a_vals = re.findall(r"<Amplitude[^>]*>([\d\.\+\-Ee]+)</Amplitude>", text, flags=re.I)
        p_vals = re.findall(r"<Phase[^>]*>([\d\.\+\-Ee]+)</Phase>", text, flags=re.I)
        model_m = re.search(r"<Model[^>]*>([^<]+)</Model>", text, flags=re.I)
        if model_m:
            model = model_m.group(1).strip()

        if not (f_vals and a_vals and p_vals) or not (len(f_vals) == len(a_vals) == len(p_vals)):
            raise ValueError("Could not parse freq/amp/phase arrays from Metronix XML.")

        f = np.array([float(x) for x in f_vals], float)
        amp = np.array([float(x) for x in a_vals], float)
        pha_deg = np.array([float(x) for x in p_vals], float)
    else:
        if pd is None:
            raise RuntimeError("pandas required to read CSV calibration.")
        df = pd.read_csv(sensor_file)
        def _pick(*names):
            for n in names:
                if n in df.columns:
                    return df[n].to_numpy()
            raise KeyError("Missing expected columns among {}".format(names))
        f = _pick("f_Hz", "Frequency_Hz", "freq_Hz", "Frequency")
        amp = _pick("amp", "Amplitude", "A")
        pha_deg = _pick("phase_deg", "Phase_deg", "Phase")

    H = amp * np.exp(1j * np.deg2rad(pha_deg))
    return TransferFunction(f=f, H=H, meta={"vendor": "Metronix", "model": model or "unknown"})


def load_tf_phoenix_sensor(cal_file: str) -> TransferFunction:
    """
    Load Phoenix sensor TF from CSV-like calibration (.cal/.coi).

    Parameters
    ----------
    cal_file : str
        Path to calibration file containing frequency, amplitude, phase.

    Returns
    -------
    TransferFunction
        Complex response H(f) on tabulated frequencies.
    """
    text = _read_text(cal_file)
    rows = [r.strip() for r in text.splitlines() if r.strip() and not r.strip().startswith("#")]
    vals = []
    for r in rows:
        parts = re.split(r"[,\s;]+", r)
        if len(parts) < 3:
            continue
        try:
            f, a, p = float(parts[0]), float(parts[1]), float(parts[2])
            vals.append((f, a, p))
        except Exception:
            continue
    if not vals:
        raise ValueError("Could not parse Phoenix calibration file (need f, amp, phase per line).")

    arr = np.array(vals, float)
    f, amp, pha_deg = arr[:, 0], arr[:, 1], arr[:, 2]
    H = amp * np.exp(1j * np.deg2rad(pha_deg))
    return TransferFunction(f=f, H=H, meta={"vendor": "Phoenix"})


def counts_to_volts(ts: TimeSeries) -> TimeSeries:
    """
    Convert ADC counts to Volts at ADC input.

    Parameters
    ----------
    ts : TimeSeries
        Time series in 'counts' with meta.adc_bits and meta.adc_range_vpp.

    Returns
    -------
    TimeSeries
        New timeseries in Volts.
    """
    bits = ts.meta.adc_bits
    vpp = ts.meta.adc_range_vpp
    if bits is None or vpp is None:
        raise ValueError("adc_bits and adc_range_vpp are required for counts->Volts.")

    lsb = vpp / (2 ** bits)
    v = ts.data * lsb
    return TimeSeries(data=v, meta=ts.meta, units="V")


def deconvolve_sensor(
    ts: TimeSeries,
    tf: TransferFunction,
    out_units: str = "auto",
    reg_eps: float = 1e-3,
    taper: Optional[str] = "hann",
) -> TimeSeries:
    """
    Deconvolve instrument TF in frequency domain.

    Parameters
    ----------
    ts : TimeSeries
        Time series in Volts at sensor output (after preamp) OR at ADC input.
    tf : TransferFunction
        Instrument response H(f) such that V(f) = H(f) * X(f), we want X.
    out_units : str
        Output units. 'auto' -> 'nT' for H*, 'mV/km' for E*.
    reg_eps : float
        Tikhonov epsilon relative to |H| max to stabilize division.
    taper : Optional[str]
        Window name (e.g., 'hann') or None.

    Returns
    -------
    TimeSeries
        Deconvolved time series X(t).
    """
    if signal is None or interpolate is None:
        raise RuntimeError("scipy.signal/interpolate required for deconvolution.")

    x = np.asarray(ts.data, float)
    n = x.size
    fs = ts.meta.fs
    if fs is None:
        raise ValueError("Sampling rate missing in metadata.")

    if taper:
        win = signal.get_window(taper, n, fftbins=True)
        xw = x * win
    else:
        xw = x

    X = np.fft.rfft(xw)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    if f.size > 1:
        f[0] = f[1]

    H_f = _interp_complex(np.asarray(tf.f, float), np.asarray(tf.H, complex), f)

    units = tf.meta.get("units", "").lower()
    if "v/(nt*hz" in units.replace(" ", "").replace("·", ""):
        H_eff = H_f * (2.0 * np.pi * f)
    else:
        H_eff = H_f

    pre = ts.meta.preamp_gain if ts.meta.preamp_gain else 1.0
    if not tf.meta.get("includes_preamp", False):
        H_eff = H_eff * pre

    eps = reg_eps * np.nanmax(np.abs(H_eff))
    denom = H_eff.conj() * H_eff + eps**2
    H_inv = H_eff.conj() / denom

    X_phys = X * H_inv
    x_phys = np.fft.irfft(X_phys, n=n)

    orientation = (ts.meta.orientation or "").lower()
    if out_units == "auto":
        if orientation.startswith("h"):
            out_units = "nT"
        elif orientation.startswith("e"):
            out_units = "mV/km"
        else:
            out_units = "physical"

    if out_units.lower() in ("mv/km", "mv_per_km"):
        L_km = ts.meta.notes.get("dipole_km", None)
        if L_km is not None and L_km > 0:
            x_phys = (x_phys / L_km) * 1e3

    return TimeSeries(data=x_phys.astype(np.float64), meta=ts.meta, units=out_units)


def quick_plot(ts: TimeSeries, title: Optional[str] = None) -> None:
    """
    Quick-look plot of a time series using matplotlib, if available.

    Parameters
    ----------
    ts : TimeSeries
        Time series to plot.
    title : Optional[str]
        Title override.
    """
    if plt is None:
        warnings.warn("matplotlib not available for plotting.")
        return
    t = np.arange(ts.data.size) / float(ts.meta.fs or 1.0)
    import matplotlib.pyplot as _plt
    _plt.figure()
    _plt.plot(t, ts.data)
    _plt.xlabel("Time [s]")
    _plt.ylabel(ts.units)
    _plt.title(title or f"Channel {ts.meta.orientation} ({ts.units})")
    _plt.tight_layout()
    _plt.show()


def export_npz(path: str, ts: TimeSeries, extra_meta: Optional[dict] = None) -> None:
    """
    Save time series to NPZ with JSON sidecar.

    Parameters
    ----------
    path : str
        Output path (without extension or ending in .npz).
    ts : TimeSeries
        Time series to save.
    extra_meta : Optional[dict]
        Additional metadata to include.
    """
    base, ext = os.path.splitext(path)
    if ext.lower() != ".npz":
        path = base + ".npz"

    np.savez_compressed(path, data=ts.data, fs=ts.meta.fs, units=ts.units)
    side = {
        "orientation": ts.meta.orientation,
        "start_time": ts.meta.start_time,
        "location": ts.meta.location,
        "adc_bits": ts.meta.adc_bits,
        "adc_range_vpp": ts.meta.adc_range_vpp,
        "preamp_gain": ts.meta.preamp_gain,
        "sensor_sensitivity": ts.meta.sensor_sensitivity,
        "sensor_model": ts.meta.sensor_model,
        "notes": ts.meta.notes,
        "export_time": _dt.datetime.utcnow().isoformat() + "Z",
    }
    if extra_meta:
        side.update(extra_meta)

    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(side, f, indent=2)


def export_hdf5(path: str, ts: TimeSeries, group: str = "/", attrs: Optional[dict] = None) -> None:
    """
    Save time series to HDF5.

    Parameters
    ----------
    path : str
        Output .h5 path.
    ts : TimeSeries
        Time series to save.
    group : str
        Group name inside HDF5 file.
    attrs : Optional[dict]
        Attributes to attach to dataset.
    """
    if h5py is None:
        raise RuntimeError("h5py not available.")

    import h5py as _h5

    with _h5.File(path, "a") as f:
        g = f.require_group(group)
        dset_name = "data"
        if dset_name in g:
            del g[dset_name]
        d = g.create_dataset(dset_name, data=ts.data, compression="gzip")
        g.attrs["fs"] = ts.meta.fs
        g.attrs["units"] = ts.units
        g.attrs["orientation"] = ts.meta.orientation
        g.attrs["start_time"] = ts.meta.start_time or ""
        if attrs:
            for k, v in attrs.items():
                g.attrs[k] = v


def export_csv(path: str, ts: TimeSeries) -> None:
    """
    Save time series to CSV with a simple header.

    Parameters
    ----------
    path : str
        Output .csv path.
    ts : TimeSeries
        Time series to save.
    """
    if pd is None:
        raise RuntimeError("pandas not available.")

    base, ext = os.path.splitext(path)
    if ext.lower() != ".csv":
        path = base + ".csv"

    df = pd.DataFrame({"data": ts.data})
    header = [
        f"# fs={ts.meta.fs}",
        f"# units={ts.units}",
        f"# orientation={ts.meta.orientation}",
        f"# start_time={ts.meta.start_time}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        df.to_csv(f, index=False)


def _cli():
    """
    CLI to read Metronix/Phoenix channels, (optionally) deconvolve, and export.

    Examples
    --------
    python mt_timeseries_io.py --vendor metronix --in Hx.ats --xml Hx.xml \\
        --tf MFS-06e.xml --out out/Hx --export npz h5 csv --deconv --dipole-km 0.1
    """
    import argparse

    p = argparse.ArgumentParser(description="MT time-series reader & deconvolver")
    p.add_argument("--vendor", required=True, choices=["metronix", "phoenix"])
    p.add_argument("--in", dest="inp", required=True, help="Input channel file")
    p.add_argument("--xml", dest="xml", default=None, help="Paired XML/LOG")
    p.add_argument("--tf", dest="tf", default=None, help="Sensor TF file")
    p.add_argument("--out", dest="out", required=True, help="Output base path")
    p.add_argument("--export", nargs="+", default=["npz"], help="npz h5 csv")
    p.add_argument("--deconv", action="store_true", help="Apply deconvolution")
    p.add_argument("--dipole-km", type=float, default=None, help="E-field dipole length (km)")
    p.add_argument("--reg-eps", type=float, default=1e-3, help="Tikhonov epsilon")
    args = p.parse_args()

    if args.vendor == "metronix":
        ts = read_metronix_ats(args.inp, args.xml)
        tf = load_tf_metronix_sensor(args.tf) if args.tf else None
    else:
        ts = read_phoenix_mtu(args.inp, args.xml)
        tf = load_tf_phoenix_sensor(args.tf) if args.tf else None

    try:
        ts_v = counts_to_volts(ts)
    except Exception as e:
        warnings.warn(f"counts->Volts skipped: {e}")
        ts_v = ts

    if args.dipole_km is not None:
        ts_v.meta.notes["dipole_km"] = float(args.dipole_km)

    out_ts = ts_v

    if args.deconv:
        if tf is None:
            raise SystemExit("Deconvolution requested but --tf is missing.")
        out_ts = deconvolve_sensor(ts_v, tf, reg_eps=args.reg_eps)

    extra = {"source_file": args.inp, "xml": args.xml, "tf": args.tf}

    if "npz" in args.export:
        export_npz(args.out, out_ts, extra_meta=extra)
    if "h5" in args.export:
        export_hdf5(args.out if args.out.endswith(".h5") else args.out + ".h5", out_ts, group="/", attrs=extra)
    if "csv" in args.export:
        export_csv(args.out, out_ts)

    print(f"Saved: {args.out} ({', '.join(args.export)})")


if __name__ == "__main__":
    _cli()
