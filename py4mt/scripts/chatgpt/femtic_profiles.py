"""
femtic_profiles.py

Vertical resistivity profiles from element centroids via interpolation.
Supports local K-NN RBF and IDW fallback. CLI can output PNG/CSV/NPZ/HDF5 and plot multiple boreholes.

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking)
"""
from typing import Literal, Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from femtic_borehole_viz import plot_vertical_profile, plot_vertical_profiles


def _to_linear(values: np.ndarray, space: Literal["linear", "log10"]) -> np.ndarray:
    return values if space == "linear" else np.power(10.0, values)


def _from_linear(values: np.ndarray, space: Literal["linear", "log10"]) -> np.ndarray:
    return values if space == "linear" else np.log10(np.clip(values, 1e-300, None))


def _kernel(r: np.ndarray, kernel: str, epsilon: float) -> np.ndarray:
    if kernel == "gaussian":
        return np.exp(-(epsilon * r) ** 2)
    if kernel == "multiquadric":
        return np.sqrt(1.0 + (epsilon * r) ** 2)
    if kernel == "inverse_multiquadric":
        return 1.0 / np.sqrt(1.0 + (epsilon * r) ** 2)
    if kernel == "linear":
        return r
    if kernel == "thin_plate":
        out = np.zeros_like(r)
        m = r > 0
        rm = r[m]
        out[m] = (rm*rm) * np.log(rm)
        return out
    raise ValueError(f"Unknown kernel: {kernel}")


def _pairwise_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.sum(a*a, axis=1)[:, None]
    bb = np.sum(b*b, axis=1)[None, :]
    return np.sqrt(np.maximum(aa + bb - 2.0 * a @ b.T, 0.0))


def vertical_profile_rbf_local(x0: float, y0: float, z_samples: np.ndarray,
                               centroids: np.ndarray, values: np.ndarray,
                               k: int = 50, kernel: str = "gaussian",
                               epsilon: Optional[float] = None, smooth: float = 1e-6,
                               in_space: Literal["linear", "log10"] = "log10",
                               out_space: Literal["linear", "log10"] = "linear") -> np.ndarray:
    """Local K-NN RBF profile at (x0,y0). Ignores fixed elements if flag present (separate 'flag' or centroid col 5)."""
    # Mask fixed via centroid col-5 if present
    if centroids.shape[1] >= 6:
        m = (centroids[:, 5] != 1)
        centroids = centroids[m]
        values = values[m]
    v_log = _from_linear(values, space=in_space)
    M = centroids.shape[0]
    P = z_samples.shape[0]
    prof_log = np.empty(P, dtype=float)
    cx, cy, cz = centroids[:, 0], centroids[:, 1], centroids[:, 2]
    for i, z in enumerate(z_samples):
        d = np.sqrt((cx - x0)**2 + (cy - y0)**2 + (cz - z)**2)
        if k >= M:
            nn = np.arange(M)
        else:
            nn = np.argpartition(d, k-1)[:k]
        pts = centroids[nn]
        vals = v_log[nn]
        D = _pairwise_dists(pts, pts)
        dq = np.linalg.norm(pts - np.array([x0, y0, z]), axis=1)
        if epsilon is None:
            md = np.median(D[D > 0]) if np.any(D > 0) else np.median(d[nn])
            eps = 1.0 / max(md, 1e-9)
        else:
            eps = float(epsilon)
        A = _kernel(D, kernel=kernel, epsilon=eps)
        A.flat[::A.shape[0]+1] += smooth
        b = vals
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, b, rcond=None)[0]
        phi_q = _kernel(dq, kernel=kernel, epsilon=eps)
        prof_log[i] = float(phi_q @ w)
    return _to_linear(prof_log, space=out_space)


def vertical_profile_idw(x0: float, y0: float, z_samples: np.ndarray,
                         centroids: np.ndarray, values: np.ndarray,
                         power: float = 2.0, eps: float = 1e-6,
                         in_space: Literal["linear", "log10"] = "log10",
                         out_space: Literal["linear", "log10"] = "linear") -> np.ndarray:
    """IDW fallback profile. Ignores fixed via centroid col-5 if present."""
    if centroids.shape[1] >= 6:
        m = (centroids[:, 5] != 1)
        centroids = centroids[m]
        values = values[m]
    v_log = _from_linear(values, space=in_space)
    cx, cy, cz = centroids[:, 0], centroids[:, 1], centroids[:, 2]
    out_log = np.empty_like(z_samples, dtype=float)
    for i, z in enumerate(z_samples):
        d2 = (cx - x0)**2 + (cy - y0)**2 + (cz - z)**2 + eps**2
        w = 1.0 / (d2 ** (power/2.0))
        out_log[i] = np.sum(w * v_log) / np.sum(w)
    return _to_linear(out_log, space=out_space)

# ---- CLI ----


def _cli():
    import argparse
    import numpy as _np
    import os
    ap = argparse.ArgumentParser(
        description="Vertical resistivity profile(s) from centroids (RBF or IDW).")
    ap.add_argument("--npz", required=True,
                    help="Path to NPZ with 'centroid' and values.")
    ap.add_argument("--values-key", default="log10_resistivity",
                    help="Key for values in NPZ (default: 'log10_resistivity').")
    ap.add_argument("--xy", action="append", nargs=2, type=float, metavar=("X", "Y"),
                    help="Add a profile at (X,Y). Can be repeated. If omitted, requires --x and --y.")
    ap.add_argument("--x", type=float,
                    help="Single profile X if --xy not given")
    ap.add_argument("--y", type=float,
                    help="Single profile Y if --xy not given")
    ap.add_argument("--zmin", type=float, required=True)
    ap.add_argument("--zmax", type=float, required=True)
    ap.add_argument("--n", dest="nsamples", type=int, default=201,
                    help="Number of Z samples (default: 201)")
    ap.add_argument("--method", choices=["rbf", "idw"], default="rbf")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--kernel", choices=["gaussian", "multiquadric",
                    "inverse_multiquadric", "linear", "thin_plate"], default="gaussian")
    ap.add_argument("--epsilon", type=float, default=None)
    ap.add_argument("--smooth", type=float, default=1e-6)
    ap.add_argument("--power", type=float, default=2.0)
    ap.add_argument("--in-space", choices=["linear", "log10"], default="log10")
    ap.add_argument(
        "--out-space", choices=["linear", "log10"], default="linear")
    ap.add_argument("--logx", action="store_true",
                    help="Use logarithmic x-scale for resistivity")
    ap.add_argument("--z-positive-down", action="store_true",
                    help="Make Z positive downward")
    ap.add_argument("--out-png", default=None, help="Save plot PNG (optional)")
    ap.add_argument("--out-csv", default=None,
                    help="Save profile CSV for single profile")
    ap.add_argument("--out-npz", default=None,
                    help="Save NPZ with profile arrays (single profile)")
    ap.add_argument("--out-hdf5", default=None,
                    help="Save HDF5 with profile datasets (single profile)")
    args = ap.parse_args()

    data = _np.load(args.npz)
    if "centroid" not in data:
        raise KeyError("NPZ must contain 'centroid' (M,3) array.")
    centroids = data["centroid"]
    if "flag" in data:
        m = (data["flag"] != 1)
        centroids = centroids[m]
    if args.values_key in data:
        values = data[args.values_key]
        in_space = args.in_space
    else:
        if "log10_resistivity" in data:
            values = data["log10_resistivity"]
            in_space = "log10"
        elif "resistivity" in data:
            values = data["resistivity"]
            in_space = "linear"
        else:
            raise KeyError("Could not find values in NPZ.")
    if "flag" in data:
        values = values[m]

    z_samples = _np.linspace(args.zmin, args.zmax, args.nsamples)
    coords = [(args.x, args.y)] if not args.xy else [
        (float(x), float(y)) for x, y in args.xy]
    if not coords or coords[0][0] is None or coords[0][1] is None:
        raise ValueError(
            "Provide either repeated --xy X Y or a single --x and --y.")

    profs_for_plot = []
    labels = []
    for (x, y) in coords:
        if args.method == "rbf":
            prof = vertical_profile_rbf_local(x, y, z_samples, centroids, values,
                                              k=args.k, kernel=args.kernel, epsilon=args.epsilon, smooth=args.smooth,
                                              in_space=in_space, out_space=args.out_space)
            label = f"({x:g},{y:g}) RBF"
        else:
            prof = vertical_profile_idw(x, y, z_samples, centroids, values,
                                        power=args.power, in_space=in_space, out_space=args.out_space)
            label = f"({x:g},{y:g}) IDW"
        # prepare for plotting if logx
        if args.logx and args.out_space == "log10":
            prof_plot = 10.0 ** prof
        else:
            prof_plot = prof
        profs_for_plot.append(prof_plot)
        labels.append(label)

    fig, ax = plot_vertical_profiles(z_samples, profs_for_plot, labels=labels,
                                     logx=args.logx, z_positive_down=args.z_positive_down)

    if args.out_png:
        fig.savefig(args.out_png, dpi=200, bbox_inches="tight")
        print("Saved PNG:", args.out_png)

    if len(coords) == 1:
        (x, y) = coords[0]
        if args.method == "rbf":
            prof_exact = vertical_profile_rbf_local(x, y, z_samples, centroids, values,
                                                    k=args.k, kernel=args.kernel, epsilon=args.epsilon, smooth=args.smooth,
                                                    in_space=in_space, out_space=args.out_space)
        else:
            prof_exact = vertical_profile_idw(x, y, z_samples, centroids, values,
                                              power=args.power, in_space=in_space, out_space=args.out_space)
        if args.out_csv:
            import csv
            with open(args.out_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["z", "value_" + args.out_space])
                for z, v in zip(z_samples, prof_exact):
                    w.writerow([z, v])
            print("Saved CSV:", args.out_csv)
        if args.out_npz:
            _np.savez_compressed(args.out_npz, z=z_samples, value=prof_exact, x=x, y=y,
                                 method=args.method, kernel=(
                                     args.kernel if args.method == "rbf" else ""),
                                 k=(args.k if args.method == "rbf" else -1),
                                 epsilon=(args.epsilon if (
                                     args.method == "rbf" and args.epsilon is not None) else _np.nan),
                                 smooth=(args.smooth if args.method ==
                                         "rbf" else _np.nan),
                                 power=(args.power if args.method ==
                                        "idw" else _np.nan),
                                 in_space=in_space, out_space=args.out_space)
            print("Saved NPZ:", args.out_npz)
        if args.out_hdf5:
            try:
                import h5py
                with h5py.File(args.out_hdf5, "w") as h5:
                    h5.create_dataset("z", data=z_samples)
                    h5.create_dataset("value", data=prof_exact)
                    h5.attrs["x"] = x
                    h5.attrs["y"] = y
                    h5.attrs["method"] = args.method
                    h5.attrs["kernel"] = args.kernel if args.method == "rbf" else ""
                    h5.attrs["k"] = args.k if args.method == "rbf" else -1
                    h5.attrs["epsilon"] = args.epsilon if (
                        args.method == "rbf" and args.epsilon is not None) else _np.nan
                    h5.attrs["smooth"] = args.smooth if args.method == "rbf" else _np.nan
                    h5.attrs["power"] = args.power if args.method == "idw" else _np.nan
                    h5.attrs["in_space"] = in_space
                    h5.attrs["out_space"] = args.out_space
                print("Saved HDF5:", args.out_hdf5)
            except Exception as e:
                print("Failed to save HDF5:", e)

    if not (args.out_png or args.out_csv or args.out_npz or args.out_hdf5):
        try:
            import matplotlib.pyplot as _plt
            _plt.show()
        except Exception:
            pass


if __name__ == "__main__":
    _cli()
