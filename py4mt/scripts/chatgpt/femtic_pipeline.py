
"""
femtic_pipeline.py

End-to-end pipeline for FEMTIC MT processing:
  1) Load mesh + resistivity.
  2) Build fast, vectorized element arrays (pure NumPy).
  3) Apply optional edits to log10 resistivity and limits.
  4) Aggregate per region (median/mean in log10 space).
  5) Inject values (and optional limits) into a template ro-file, save new file.

Intermediates can be saved to NPZ/HDF5/CSV for audit & reuse.

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-11-09
"""

import numpy as np
from typing import Optional, Tuple
from femtic_mesh_reader import load_femtic_mesh
from femtic_element_array import build_element_arrays, save_npz, save_hdf5, save_csv_compact
from femtic_ro_inject import write_ro_from_vector_with_flags

def apply_log10_edits(rho_log: np.ndarray,
                      shift: float = 0.0,
                      scale: float = 1.0,
                      clip_low: Optional[float] = None,
                      clip_high: Optional[float] = None) -> np.ndarray:
    res = scale * rho_log + shift
    if clip_low is not None or clip_high is not None:
        lo = -np.inf if clip_low is None else clip_low
        hi = np.inf if clip_high is None else clip_high
        res = np.clip(res, lo, hi)
    return res

def rebuild_limits_from_margin(rho_log: np.ndarray, margin: float) -> Tuple[np.ndarray, np.ndarray]:
    return rho_log - margin, rho_log + margin

def aggregate_per_region(region: np.ndarray,
                         rho_log: np.ndarray,
                         lo_log: Optional[np.ndarray],
                         hi_log: Optional[np.ndarray],
                         method: str = "median_log10"):
    reg_ids = np.unique(region.astype(int))
    reg_ids.sort()
    K = reg_ids.size
    v_log = np.empty(K, dtype=float)
    lo_agg = np.empty(K, dtype=float)
    hi_agg = np.empty(K, dtype=float)
    for i, rid in enumerate(reg_ids):
        m = (region == rid)
        vals = rho_log[m]
        if vals.size == 0:
            v_log[i] = np.nan
        else:
            v_log[i] = float(np.mean(vals)) if method == "mean_log10" else float(np.median(vals))
        if lo_log is not None and hi_log is not None and m.any():
            lo_agg[i] = float(np.median(lo_log[m]))
            hi_agg[i] = float(np.median(hi_log[m]))
        else:
            lo_agg[i] = np.nan
            hi_agg[i] = np.nan
    return reg_ids, v_log, lo_agg, hi_agg

def run_pipeline(mesh_path: str,
                 rho_path: Optional[str],
                 template_ro: str,
                 out_ro: str,
                 margin: float = 0.5,
                 shift: float = 0.0,
                 scale: float = 1.0,
                 clip_low: Optional[float] = None,
                 clip_high: Optional[float] = None,
                 agg: str = "median_log10",
                 write_space: str = "linear",
                 append_limits: bool = True,
                 save_npz_path: Optional[str] = None,
                 save_hdf5_path: Optional[str] = None,
                 save_csv_path: Optional[str] = None,
                 fmt: str = "{:.6g}",
                 value_col: Optional[int] = None,
                 lower_col: int = 2,
                 upper_col: int = 3,
                 n_col: int = 4,
                 override_n: Optional[int] = None,
                 flag_col: Optional[int] = None,
                 set_flag: Optional[int] = None) -> None:
    mesh = load_femtic_mesh(mesh_path, rho_path)
    arr = build_element_arrays(mesh, margin=margin)

    rho_log = arr["log10_resistivity"].copy()
    rho_log = apply_log10_edits(rho_log, shift=shift, scale=scale, clip_low=clip_low, clip_high=clip_high)

    lo_log, hi_log = rebuild_limits_from_margin(rho_log, margin)
    arr["log10_resistivity"] = rho_log
    arr["rho_lower"] = lo_log
    arr["rho_upper"] = hi_log

    if save_npz_path:
        save_npz(arr, save_npz_path, compressed=True)
    if save_hdf5_path:
        save_hdf5(arr, save_hdf5_path)
    if save_csv_path:
        save_csv_compact(arr, save_csv_path)

    reg_ids, v_log, lo_reg, hi_reg = aggregate_per_region(arr["region"], rho_log, lo_log, hi_log, method=agg)

    if write_space == "linear":
        values = np.power(10.0, v_log)
        lower = np.power(10.0, lo_reg) if append_limits else None
        upper = np.power(10.0, hi_reg) if append_limits else None
    else:
        values = v_log
        lower = lo_reg if append_limits else None
        upper = hi_reg if append_limits else None

    write_ro_from_vector_with_flags(
        template_ro, values, out_ro,
        lower=lower, upper=upper,
        override_n=override_n,
        value_col=value_col,
        lower_col=lower_col, upper_col=upper_col, n_col=n_col,
        flag_col=flag_col, set_flag=set_flag,
        fmt=fmt
    )

def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="FEMTIC end-to-end pipeline: arrays -> edits -> aggregate -> inject ro (fixed cols, flags, n).")
    ap.add_argument("--mesh", required=True, help="Path to mesh.dat")
    ap.add_argument("--rho", default=None, help="Path to resistivity_block_iterX.dat")
    ap.add_argument("--template-ro", required=True, help="Template ro-file to inject into")
    ap.add_argument("--out-ro", required=True, help="Output ro-file path (new name)")
    ap.add_argument("--margin", type=float, default=0.5, help="Log10 margin for lower/upper limits around edited values")
    ap.add_argument("--shift", type=float, default=0.0, help="Additive shift in log10 space (value += shift)")
    ap.add_argument("--scale", type=float, default=1.0, help="Multiplicative scale in log10 space (value *= scale)")
    ap.add_argument("--clip-low", type=float, default=None, help="Minimum allowed log10 value after edits")
    ap.add_argument("--clip-high", type=float, default=None, help="Maximum allowed log10 value after edits")
    ap.add_argument("--agg", choices=["median_log10","mean_log10"], default="median_log10", help="Aggregation method per region")
    ap.add_argument("--write-space", choices=["linear","log10"], default="linear", help="Write ro-file values in this space")
    ap.add_argument("--no-append-limits", action="store_true", help="Do not inject lower/upper bounds")
    ap.add_argument("--save-npz", default=None, help="Optional: write intermediate arrays to NPZ")
    ap.add_argument("--save-hdf5", default=None, help="Optional: write intermediate arrays to HDF5")
    ap.add_argument("--save-csv", default=None, help="Optional: write intermediate arrays to compact CSV")
    ap.add_argument("--format", dest="fmt", default="{:.6g}", help="Number format for ro-file values, e.g. '{:.6g}' or '{:.6f}'")
    ap.add_argument("--value-col", type=int, default=None, help="1-based column for main value (default: last numeric in line)")
    ap.add_argument("--lower-col", type=int, default=2, help="1-based lower column (default: 2)")
    ap.add_argument("--upper-col", type=int, default=3, help="1-based upper column (default: 3)")
    ap.add_argument("--n-col", type=int, default=4, help="1-based sharpness column (default: 4)")
    ap.add_argument("--n", dest="override_n", type=int, choices=[1,2,3], default=None, help="Override sharpness n (otherwise keep file)")
    ap.add_argument("--flag-col", type=int, default=None, help="1-based flag column (default: preserve; -1 for last numeric)")
    ap.add_argument("--set-flag", type=int, choices=[0,1], default=None, help="Override all flags (0/1); default keep file values")

    args = ap.parse_args()

    run_pipeline(
        mesh_path=args.mesh,
        rho_path=args.rho,
        template_ro=args.template_ro,
        out_ro=args.out_ro,
        margin=args.margin,
        shift=args.shift,
        scale=args.scale,
        clip_low=args.clip_low,
        clip_high=args.clip_high,
        agg=args.agg,
        write_space=args.write_space,
        append_limits=(not args.no_append_limits),
        save_npz_path=args.save_npz,
        save_hdf5_path=args.save_hdf5,
        save_csv_path=args.save_csv,
        fmt=args.fmt,
        value_col=args.value_col,
        lower_col=args.lower_col,
        upper_col=args.upper_col,
        n_col=args.n_col,
        override_n=args.override_n,
        flag_col=args.flag_col,
        set_flag=args.set_flag
    )
    print(f"Wrote ro-file: {args.out_ro}")

if __name__ == "__main__":
    _cli()
