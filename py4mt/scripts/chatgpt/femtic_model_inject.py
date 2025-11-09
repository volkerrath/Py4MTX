
"""
femtic_ro_inject.py

Inject updated resistivity values (and limits) into a FEMTIC ro-file.

New behavior (2025-11-09):
- Columns **2, 3, 4** (1-based) are assumed to already contain:
    2: lower bound
    3: upper bound
    4: sharpness parameter n
  These should be **overwritten in place** (no appending).
- The main resistivity value is, by default, the **last numeric** on each data line (as before).
- 'n' is **kept unchanged** unless an optional override (1/2/3) is provided.

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-11-09
"""

from typing import List, Tuple, Optional
import numpy as np

def parse_ro_file_lines(path: str) -> List[str]:
    with open(path, "r") as f:
        return f.readlines()

def _is_number(tok: str) -> bool:
    try:
        float(tok)
        return True
    except Exception:
        return False

def _find_last_numeric_index(parts: List[str]) -> Optional[int]:
    for j in range(len(parts) - 1, -1, -1):
        if _is_number(parts[j]):
            return j
    return None

def inject_values_into_columns(lines: List[str],
                               values: np.ndarray,
                               lower: Optional[np.ndarray] = None,
                               upper: Optional[np.ndarray] = None,
                               override_n: Optional[int] = None,
                               value_col: Optional[int] = None,
                               lower_col: int = 2,
                               upper_col: int = 3,
                               n_col: int = 4,
                               fmt: str = "{:.6g}") -> List[str]:
    """Inject series into specified columns of the ro-file (1-based columns)."""
    def to0(i: Optional[int]) -> Optional[int]:
        return None if i is None else max(0, i - 1)

    vcol0 = to0(value_col)
    lcol0 = to0(lower_col)
    ucol0 = to0(upper_col)
    ncol0 = to0(n_col)

    out: List[str] = []
    vi = 0
    for ln in lines:
        s = ln.strip()
        if s == "" or s.startswith("#") or s.startswith("!"):
            out.append(ln)
            continue

        parts = ln.rstrip("\n").split()

        tgt_v = vcol0 if vcol0 is not None else _find_last_numeric_index(parts)
        if tgt_v is None or vi >= len(values):
            out.append(ln); continue

        parts[tgt_v] = fmt.format(float(values[vi]))

        if lower is not None and lcol0 is not None and lcol0 < len(parts) and _is_number(parts[lcol0]):
            parts[lcol0] = fmt.format(float(lower[vi]))
        if upper is not None and ucol0 is not None and ucol0 < len(parts) and _is_number(parts[ucol0]):
            parts[ucol0] = fmt.format(float(upper[vi]))

        if override_n is not None:
            if override_n not in (1, 2, 3):
                raise ValueError("override_n must be one of {1,2,3}")
            if ncol0 is not None and ncol0 < len(parts) and _is_number(parts[ncol0]):
                parts[ncol0] = str(int(override_n))

        out.append(" ".join(parts) + "\n")
        vi += 1

    if vi < len(values):
        print(f"[WARN] Only injected {vi} of {len(values)} values; extra values ignored.")
    return out

def write_ro_from_npz(template_ro: str,
                      npz_path: str,
                      out_ro: str,
                      method: str = "median_log10",
                      to_space: str = "linear",
                      value_col: Optional[int] = None,
                      lower_col: int = 2,
                      upper_col: int = 3,
                      n_col: int = 4,
                      override_n: Optional[int] = None,
                      fmt: str = "{:.6g}") -> None:
    """Aggregate per-region from NPZ, then inject into specified columns."""
    data = np.load(npz_path)
    region = data["region"]
    v_log = data["log10_resistivity"]
    lo_log = data.get("rho_lower", None)
    hi_log = data.get("rho_upper", None)

    reg_ids = np.unique(region.astype(int)); reg_ids.sort()
    v_agg = np.empty_like(reg_ids, dtype=float)
    lo_agg = np.empty_like(reg_ids, dtype=float) if lo_log is not None else None
    hi_agg = np.empty_like(reg_ids, dtype=float) if hi_log is not None else None
    for i, rid in enumerate(reg_ids):
        m = (region == rid)
        vals = v_log[m]
        v_agg[i] = float(np.mean(vals)) if method == "mean_log10" else float(np.median(vals))
        if lo_agg is not None:
            lo_agg[i] = float(np.median(lo_log[m]))
        if hi_agg is not None:
            hi_agg[i] = float(np.median(hi_log[m]))

    if to_space == "linear":
        vals = np.power(10.0, v_agg)
        lo = np.power(10.0, lo_agg) if lo_agg is not None else None
        hi = np.power(10.0, hi_agg) if hi_agg is not None else None
    else:
        vals, lo, hi = v_agg, lo_agg, hi_agg

    lines = parse_ro_file_lines(template_ro)
    new_lines = inject_values_into_columns(lines, vals, lo, hi,
                                           override_n=override_n,
                                           value_col=value_col,
                                           lower_col=lower_col,
                                           upper_col=upper_col,
                                           n_col=n_col,
                                           fmt=fmt)
    with open(out_ro, "w") as f:
        f.writelines(new_lines)

def write_ro_from_vector(template_ro: str,
                         values: np.ndarray,
                         out_ro: str,
                         lower: Optional[np.ndarray] = None,
                         upper: Optional[np.ndarray] = None,
                         override_n: Optional[int] = None,
                         value_col: Optional[int] = None,
                         lower_col: int = 2,
                         upper_col: int = 3,
                         n_col: int = 4,
                         fmt: str = "{:.6g}") -> None:
    lines = parse_ro_file_lines(template_ro)
    new_lines = inject_values_into_columns(
        lines, values, lower, upper,
        override_n=override_n,
        value_col=value_col,
        lower_col=lower_col, upper_col=upper_col, n_col=n_col,
        fmt=fmt
    )
    with open(out_ro, "w") as f:
        f.writelines(new_lines)

def _cli():
    import argparse, os
    ap = argparse.ArgumentParser(description="Inject updated values into a FEMTIC ro-file (fixed columns for lower/upper/n).")
    ap.add_argument("--template", required=True, help="Template ro-file to read and modify.")
    ap.add_argument("--out", required=True, help="Output ro-file path.")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--from-npz", dest="from_npz", help="Path to NPZ (from femtic_element_array). Aggregates per region.")
    mode.add_argument("--from-vector", dest="from_vector", help="Path to .npy or .npz containing 'values' (and optional 'lower','upper').")
    ap.add_argument("--method", choices=["median_log10","mean_log10"], default="median_log10", help="Aggregation for --from-npz.")
    ap.add_argument("--space", choices=["linear","log10"], default="linear", help="Write values in this numeric space.")
    ap.add_argument("--value-col", type=int, default=None, help="1-based numeric column of the main value (default: last numeric in line)")
    ap.add_argument("--lower-col", type=int, default=2, help="1-based column for lower bound (default: 2)")
    ap.add_argument("--upper-col", type=int, default=3, help="1-based column for upper bound (default: 3)")
    ap.add_argument("--n-col", type=int, default=4, help="1-based column for sharpness n (default: 4)")
    ap.add_argument("--n", dest="override_n", type=int, choices=[1,2,3], default=None, help="Override sharpness n (otherwise keep original)")
    ap.add_argument("--format", dest="fmt", default="{:.6g}", help="Number format for floats, e.g. '{:.6f}'")
    args = ap.parse_args()

    if args.from_npz:
        write_ro_from_npz(args.template, args.from_npz, args.out,
                          method=args.method, to_space=args.space,
                          value_col=args.value_col,
                          lower_col=args.lower_col, upper_col=args.upper_col, n_col=args.n_col,
                          override_n=args.override_n, fmt=args.fmt)
        print(f"Wrote ro-file: {args.out}")
    else:
        vals = None; lo = None; hi = None
        if args.from_vector.endswith(".npz"):
            d = np.load(args.from_vector)
            if "values" in d:
                vals = d["values"]
            else:
                raise ValueError("NPZ must contain a 'values' array for --from-vector.")
            lo = d.get("lower", None); hi = d.get("upper", None)
        elif args.from_vector.endswith(".npy"):
            vals = np.load(args.from_vector)
        else:
            raise ValueError("--from-vector must be .npy or .npz with 'values'.")
        write_ro_from_vector(args.template, vals, args.out,
                             lower=lo, upper=hi,
                             override_n=args.override_n,
                             value_col=args.value_col,
                             lower_col=args.lower_col, upper_col=args.upper_col, n_col=args.n_col,
                             fmt=args.fmt)
        print(f"Wrote ro-file: {args.out}")

if __name__ == "__main__":
    _cli()
