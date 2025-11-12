"""
femtic_rho_inject.py

Inject values (and optional lower/upper/n and flags) into a FEMTIC rho-file.
Functions use *rho* throughout; no legacy *ro* names remain.

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking)
"""
from typing import List, Optional
import numpy as np


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def _find_last_numeric_index(parts: List[str]) -> Optional[int]:
    idx = None
    for i, t in enumerate(parts):
        if _is_number(t):
            idx = i
    return idx


def parse_rho_file_lines(path: str) -> List[str]:
    """Read a rho-file as raw lines (keeps comments and spacing)."""
    with open(path, "r") as f:
        return f.readlines()


def _resolve_col_index(parts: List[str], col1_based: int) -> int:
    """Resolve a 1-based column to 0-based; -1 means 'last numeric token'."""
    if col1_based == -1:
        idx = _find_last_numeric_index(parts)
        return -1 if idx is None else idx
    return max(0, col1_based - 1)


def inject_values_with_flags(lines: List[str],
                             values: np.ndarray,
                             lower: Optional[np.ndarray] = None,
                             upper: Optional[np.ndarray] = None,
                             override_n: Optional[int] = None,
                             value_col_1b: Optional[int] = None,
                             lower_col_1b: int = 2,
                             upper_col_1b: int = 3,
                             n_col_1b: int = 4,
                             flag_col_1b: Optional[int] = None,
                             set_flag: Optional[int] = None,
                             flag_vec: Optional[np.ndarray] = None,
                             fmt: str = "{:.6g}") -> List[str]:
    """Core injector: write values (+limits/n/flags) into text lines of a rho-file."""
    out: List[str] = []
    vi = 0
    for ln in lines:
        s = ln.strip()
        if s == "" or s.startswith("#") or s.startswith("!"):
            out.append(ln)
            continue
        parts = ln.rstrip("\n").split()

        if value_col_1b is None:
            tgt_v = _find_last_numeric_index(parts)
        else:
            tgt_v = _resolve_col_index(parts, value_col_1b)
            if tgt_v == -1:
                tgt_v = _find_last_numeric_index(parts)

        if tgt_v is None or vi >= len(values):
            out.append(ln)
            continue

        parts[tgt_v] = fmt.format(float(values[vi]))

        if lower is not None:
            lc = _resolve_col_index(parts, lower_col_1b)
            if lc != -1 and lc < len(parts) and _is_number(parts[lc]):
                parts[lc] = fmt.format(float(lower[vi]))
        if upper is not None:
            uc = _resolve_col_index(parts, upper_col_1b)
            if uc != -1 and uc < len(parts) and _is_number(parts[uc]):
                parts[uc] = fmt.format(float(upper[vi]))

        if override_n is not None:
            if override_n not in (1, 2, 3):
                raise ValueError("override_n must be 1,2, or 3")
            nc = _resolve_col_index(parts, n_col_1b)
            if nc != -1 and nc < len(parts) and _is_number(parts[nc]):
                parts[nc] = str(int(override_n))

        if set_flag is not None or flag_vec is not None:
            if flag_col_1b is not None:
                fc = _resolve_col_index(parts, flag_col_1b)
                if fc != -1 and fc < len(parts) and _is_number(parts[fc]):
                    if set_flag is not None:
                        parts[fc] = str(int(set_flag))
                    elif flag_vec is not None and vi < len(flag_vec):
                        parts[fc] = str(int(flag_vec[vi]))

        out.append(" ".join(parts) + "\n")
        vi += 1
    if vi < len(values):
        print(f"[WARN] Only injected {vi} of {
              len(values)} rows; extra values ignored.")
    return out


def write_rho_from_vector_with_flags(template_rho: str,
                                     values: np.ndarray,
                                     out_rho: str,
                                     lower: Optional[np.ndarray] = None,
                                     upper: Optional[np.ndarray] = None,
                                     override_n: Optional[int] = None,
                                     value_col: Optional[int] = None,
                                     lower_col: int = 2,
                                     upper_col: int = 3,
                                     n_col: int = 4,
                                     flag_col: Optional[int] = None,
                                     set_flag: Optional[int] = None,
                                     flag_vec: Optional[np.ndarray] = None,
                                     fmt: str = "{:.6g}") -> None:
    """Direct vector injection (no aggregation)."""
    lines = parse_rho_file_lines(template_rho)
    new_lines = inject_values_with_flags(lines, values, lower, upper,
                                         override_n=override_n,
                                         value_col_1b=value_col,
                                         lower_col_1b=lower_col,
                                         upper_col_1b=upper_col,
                                         n_col_1b=n_col,
                                         flag_col_1b=flag_col,
                                         set_flag=set_flag,
                                         flag_vec=flag_vec,
                                         fmt=fmt)
    with open(out_rho, "w") as f:
        f.writelines(new_lines)


def write_rho_from_npz_with_flags(template_rho: str,
                                  npz_path: str,
                                  out_rho: str,
                                  method: str = "median_log10",
                                  to_space: str = "linear",
                                  value_col: Optional[int] = None,
                                  lower_col: int = 2,
                                  upper_col: int = 3,
                                  n_col: int = 4,
                                  override_n: Optional[int] = None,
                                  flag_col: Optional[int] = None,
                                  set_flag: Optional[int] = None,
                                  fmt: str = "{:.6g}") -> None:
    """Aggregate per-region from NPZ (region, log10_resistivity, optional rho_lower/upper, flag) and inject."""
    data = np.load(npz_path)
    region = data["region"]
    v_log = data["log10_resistivity"]
    lo_log = data.get("rho_lower", None)
    hi_log = data.get("rho_upper", None)
    flag_vec = data.get("flag", None)

    reg_ids = np.unique(region.astype(int))
    reg_ids.sort()
    v_agg = np.empty_like(reg_ids, dtype=float)
    lo_agg = np.empty_like(
        reg_ids, dtype=float) if lo_log is not None else None
    hi_agg = np.empty_like(
        reg_ids, dtype=float) if hi_log is not None else None
    f_agg = np.empty_like(reg_ids, dtype=int) if flag_vec is not None else None

    for i, rid in enumerate(reg_ids):
        m = (region == rid)
        vals = v_log[m]
        v_agg[i] = float(np.median(vals)) if method == "median_log10" else float(
            np.mean(vals))
        if lo_agg is not None:
            lo_agg[i] = float(np.median(lo_log[m]))
        if hi_agg is not None:
            hi_agg[i] = float(np.median(hi_log[m]))
        if f_agg is not None:
            uniq, counts = np.unique(
                flag_vec[m].astype(int), return_counts=True)
            f_agg[i] = int(uniq[np.argmax(counts)]) if uniq.size else 0

    if to_space == "linear":
        vals = np.power(10.0, v_agg)
        lo = np.power(10.0, lo_agg) if lo_agg is not None else None
        hi = np.power(10.0, hi_agg) if hi_agg is not None else None
    else:
        vals, lo, hi = v_agg, lo_agg, hi_agg

    lines = parse_rho_file_lines(template_rho)
    new_lines = inject_values_with_flags(lines, vals, lo, hi,
                                         override_n=override_n,
                                         value_col_1b=value_col,
                                         lower_col_1b=lower_col,
                                         upper_col_1b=upper_col,
                                         n_col_1b=n_col,
                                         flag_col_1b=flag_col,
                                         set_flag=set_flag,
                                         flag_vec=f_agg,
                                         fmt=fmt)
    with open(out_rho, "w") as f:
        f.writelines(new_lines)


if __name__ == "__main__":
    import argparse
    import numpy as _np
    ap = argparse.ArgumentParser(
        description="Inject values into a FEMTIC rho-file (with lower/upper/n and flags).")
    ap.add_argument("--template", required=True,
                    help="Template rho-file to read and modify.")
    ap.add_argument("--out", required=True, help="Output rho-file path.")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--from-npz", dest="from_npz",
                      help="Path to NPZ (aggregates per region).")
    mode.add_argument("--from-vector", dest="from_vector",
                      help="Path to .npy or .npz with 'values' (+ optional 'lower','upper','flag').")
    ap.add_argument(
        "--method", choices=["median_log10", "mean_log10"], default="median_log10")
    ap.add_argument("--space", choices=["linear", "log10"], default="linear")
    ap.add_argument("--value-col", type=int, default=None,
                    help="1-based value column (default: last numeric on line)")
    ap.add_argument("--lower-col", type=int, default=2,
                    help="1-based lower column (default: 2)")
    ap.add_argument("--upper-col", type=int, default=3,
                    help="1-based upper column (default: 3)")
    ap.add_argument("--n-col", type=int, default=4,
                    help="1-based sharpness column (default: 4)")
    ap.add_argument("--n", dest="override_n", type=int, choices=[
                    1, 2, 3], default=None, help="Override sharpness n (otherwise keep file)")
    ap.add_argument("--flag-col", type=int, default=None,
                    help="1-based flag column (default: preserve; -1 for last numeric)")
    ap.add_argument("--set-flag", type=int, choices=[
                    0, 1], default=None, help="Override all flags (0/1); default keep file values")
    ap.add_argument("--format", dest="fmt",
                    default="{:.6g}", help="Number format for floats, e.g. '{:.6f}'")
    args = ap.parse_args()

    if args.from_npz:
        write_rho_from_npz_with_flags(args.template, args.from_npz, args.out,
                                      method=args.method, to_space=args.space,
                                      value_col=args.value_col,
                                      lower_col=args.lower_col, upper_col=args.upper_col, n_col=args.n_col,
                                      override_n=args.override_n,
                                      flag_col=args.flag_col, set_flag=args.set_flag, fmt=args.fmt)
        print("Wrote rho-file:", args.out)
    else:
        vals = None
        lo = None
        hi = None
        flag_vec = None
        if args.from_vector.endswith(".npz"):
            d = _np.load(args.from_vector)
            if "values" in d:
                vals = d["values"]
            else:
                raise ValueError(
                    "NPZ must contain a 'values' array for --from-vector.")
            lo = d.get("lower", None)
            hi = d.get("upper", None)
            flag_vec = d.get("flag", None)
        elif args.from_vector.endswith(".npy"):
            vals = _np.load(args.from_vector)
        else:
            raise ValueError(
                "--from-vector must be .npy or .npz with 'values'.")
        write_rho_from_vector_with_flags(args.template, vals, args.out,
                                         lower=lo, upper=hi,
                                         override_n=args.override_n,
                                         value_col=args.value_col,
                                         lower_col=args.lower_col, upper_col=args.upper_col, n_col=args.n_col,
                                         flag_col=args.flag_col, set_flag=args.set_flag, flag_vec=flag_vec,
                                         fmt=args.fmt)
        print("Wrote rho-file:", args.out)
