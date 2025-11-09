
"""
femtic_profiles.py

Vertical resistivity profiles from element centroids via interpolation.
Supports a fast **local RBF** (per-sample K-neighbour system) and a simple **IDW** fallback.

Typical usage
-------------
>>> import numpy as np
>>> from femtic_profiles import vertical_profile_rbf_local, plot_vertical_profile
>>> data = np.load("elements_arrays.npz")  # created by femtic_element_array.save_npz()
>>> xyz = data["centroid"]                 # (M,3)
>>> vlog = data["log10_resistivity"]       # (M,)
>>> x0, y0 = 1000.0, 500.0
>>> z_samples = np.linspace(-2000, 0, 101) # depths or elevations (same units as xyz[:,2])
>>> prof = vertical_profile_rbf_local(x0, y0, z_samples, xyz, vlog,
...                                   k=50, kernel="gaussian", epsilon=None,
...                                   smooth=1e-6, in_space="log10", out_space="linear")
>>> fig, ax = plot_vertical_profile(z_samples, prof, label="RBF(local)")

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-11-09
"""

from typing import Literal, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from femtic_borehole_viz import plot_vertical_profile, plot_vertical_profiles


# ------------------------
# Utilities
# ------------------------

def _to_linear(values: np.ndarray, space: Literal["linear","log10"]) -> np.ndarray:
    if space == "linear":
        return values
    return np.power(10.0, values)


def _from_linear(values: np.ndarray, space: Literal["linear","log10"]) -> np.ndarray:
    if space == "linear":
        return values
    # clip to positive before log10
    return np.log10(np.clip(values, 1e-300, None))


def _kernel(r: np.ndarray,
            kernel: Literal["gaussian","multiquadric","inverse_multiquadric","linear","thin_plate"],
            epsilon: float) -> np.ndarray:
    # r >= 0
    if kernel == "gaussian":
        return np.exp(-(epsilon * r) ** 2)
    if kernel == "multiquadric":
        return np.sqrt(1.0 + (epsilon * r) ** 2)
    if kernel == "inverse_multiquadric":
        return 1.0 / np.sqrt(1.0 + (epsilon * r) ** 2)
    if kernel == "linear":
        return r
    if kernel == "thin_plate":
        # phi(r) = r^2 log(r), define phi(0)=0
        out = np.zeros_like(r)
        mask = r > 0
        rm = r[mask]
        out[mask] = (rm * rm) * np.log(rm)
        return out
    raise ValueError(f"Unknown kernel: {kernel}")


def _pairwise_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: (Na,3), b: (Nb,3) -> (Na,Nb) Euclidean
    # Efficient via (x^2 + y^2 + z^2 - 2 x.x)
    aa = np.sum(a * a, axis=1)[:, None]
    bb = np.sum(b * b, axis=1)[None, :]
    return np.sqrt(np.maximum(aa + bb - 2.0 * a @ b.T, 0.0))


# ---------------------------------
# Local RBF vertical profile (fast)
# ---------------------------------

def vertical_profile_rbf_local(x0: float,
                               y0: float,
                               z_samples: np.ndarray,
                               centroids: np.ndarray,   # (M,3)
                               values: np.ndarray,      # (M,)
                               k: int = 50,
                               kernel: Literal["gaussian","multiquadric","inverse_multiquadric","linear","thin_plate"] = "gaussian",
                               epsilon: Optional[float] = None,
                               smooth: float = 1e-6,
                               in_space: Literal["linear","log10"] = "log10",
                               out_space: Literal["linear","log10"] = "linear") -> np.ndarray:
    """Compute a vertical profile at (x0,y0) by **local** RBF interpolation from centroids.

    Strategy:
      For each depth z_s, select the K nearest centroids to the query point (x0,y0,z_s),
      solve a KxK RBF system with Tikhonov smoothing (lambda = smooth), then evaluate
      at the query location. This avoids the O(M^3) full-system cost.

    Parameters
    ----------
    x0, y0 : float
        Horizontal location for the profile.
    z_samples : (P,) array
        Vertical coordinates (depth or elevation) in same units as centroids[:,2].
    centroids : (M,3) float
        Element centroids.
    values : (M,) float
        Resistivity values at centroids, in the space given by `in_space`.
    k : int, default 50
        Number of neighbors per sample. Choose 20..200 depending on sampling density.
    kernel : str, default 'gaussian'
        RBF kernel. Options: 'gaussian', 'multiquadric', 'inverse_multiquadric', 'linear', 'thin_plate'.
    epsilon : float, optional
        RBF shape parameter. If None, use 1 / median(neighbor distance) per sample (robust heuristic).
    smooth : float, default 1e-6
        Tikhonov regularization factor added on the diagonal for stability.
    in_space : 'linear' or 'log10', default 'log10'
        Numeric space of input `values`.
    out_space : 'linear' or 'log10', default 'linear'
        Desired numeric space of output profile.

    Returns
    -------
    prof : (P,) array
        Interpolated profile in `out_space`.
    """
    assert centroids.shape[1] == 3, "centroids must be (M,3)"
    assert values.shape[0] == centroids.shape[0], "values must match centroids"

    # Work in log10 space internally for stability if inputs are linear
    v_log = _from_linear(values, space=in_space)

    M = centroids.shape[0]
    P = z_samples.shape[0]
    prof_log = np.empty(P, dtype=float)

    # Pre-split x,y,z
    cx, cy, cz = centroids[:,0], centroids[:,1], centroids[:,2]

    for i, z in enumerate(z_samples):
    # Exclude fixed elements (flag==1) if provided as separate array or in centroids column 5
    try:
        _flag = None
        # If caller passed a dict-like 'data' or we are in CLI, this block is skipped; here we only use args
    except Exception:
        pass
    # If a separate 'flag' variable is available in outer scope, functions won't see it; so rely on inputs
    if centroids.shape[0] == values.shape[0]:
        # Try to detect a flag column in centroids (6th column, index 5)
        if centroids.shape[1] >= 6:
            _mask = centroids[:, 5] != 1
            centroids = centroids[_mask]
            values = values[_mask]
    # Note: the CLI also applies masking using a separate 'flag' vector if present in the NPZ
        # Query point
        q = np.array([x0, y0, z])

        # Find K nearest centroids in 3D (brute force distance, fine for P*logM scale)
        d = np.sqrt((cx - x0)**2 + (cy - y0)**2 + (cz - z)**2)
        if k >= M:
            nn_idx = np.arange(M)
        else:
            nn_idx = np.argpartition(d, kth=k-1)[:k]

        pts = centroids[nn_idx]      # (K,3)
        vals = v_log[nn_idx]         # (K,)

        # distances among neighbors and to query
        D = _pairwise_dists(pts, pts)   # (K,K)
        dq = np.linalg.norm(pts - q, axis=1)  # (K,)

        # pick epsilon
        if epsilon is None:
            # robust scale: 1 / median of pairwise neighbor distances (non-zero)
            md = np.median(D[D > 0]) if np.any(D > 0) else np.median(d[nn_idx])
            eps = 1.0 / max(md, 1e-9)
        else:
            eps = float(epsilon)

        # Build RBF matrix and right-hand side
        A = _kernel(D, kernel=kernel, epsilon=eps)
        # Add Tikhonov regularization
        A.flat[::A.shape[0]+1] += smooth
        b = vals

        # Solve A w = b
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # fallback to least squares
            w = np.linalg.lstsq(A, b, rcond=None)[0]

        # Evaluate at query
        phi_q = _kernel(dq, kernel=kernel, epsilon=eps)  # (K,)
        prof_log[i] = float(phi_q @ w)

    return _to_linear(prof_log, space=out_space)


# ------------------------
# IDW fallback
# ------------------------

def vertical_profile_idw(x0: float,
                         y0: float,
                         z_samples: np.ndarray,
                         centroids: np.ndarray,
                         values: np.ndarray,
                         power: float = 2.0,
                         eps: float = 1e-6,
                         in_space: Literal["linear","log10"] = "log10",
                         out_space: Literal["linear","log10"] = "linear") -> np.ndarray:
    """Inverse Distance Weighting (IDW) profile as a simple fallback."""
    v_log = _from_linear(values, space=in_space)
    cx, cy, cz = centroids[:,0], centroids[:,1], centroids[:,2]
    out_log = np.empty_like(z_samples, dtype=float)
    for i, z in enumerate(z_samples):
    # Exclude fixed elements (flag==1) if provided as separate array or in centroids column 5
    try:
        _flag = None
        # If caller passed a dict-like 'data' or we are in CLI, this block is skipped; here we only use args
    except Exception:
        pass
    # If a separate 'flag' variable is available in outer scope, functions won't see it; so rely on inputs
    if centroids.shape[0] == values.shape[0]:
        # Try to detect a flag column in centroids (6th column, index 5)
        if centroids.shape[1] >= 6:
            _mask = centroids[:, 5] != 1
            centroids = centroids[_mask]
            values = values[_mask]
    # Note: the CLI also applies masking using a separate 'flag' vector if present in the NPZ
        d2 = (cx - x0)**2 + (cy - y0)**2 + (cz - z)**2 + eps**2
        w = 1.0 / (d2 ** (power/2.0))
        out_log[i] = np.sum(w * v_log) / np.sum(w)
    return _to_linear(out_log, space=out_space)


# ------------------------
# Plotting helper
# ------------------------

def plot_vertical_profile(z: np.ndarray,
                          v: np.ndarray,
                          label: Optional[str] = None,
                          ax: Optional[plt.Axes] = None,
                          invert_z: bool = True) -> Tuple[plt.Figure, plt.Axes]:
    """Quick plot of a vertical profile v(z)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,6))
    else:
        fig = ax.figure
    ax.plot(v, z, label=label)
    ax.set_xlabel("Resistivity")
    ax.set_ylabel("Z")
    if invert_z:
        ax.invert_yaxis()
    if label:
        ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


# ------------------------
# CLI
# ------------------------



def _cli():
    import argparse, numpy as _np, os

    ap = argparse.ArgumentParser(description="Vertical resistivity profile(s) from centroids (RBF or IDW).")
    ap.add_argument("--npz", required=True, help="Path to NPZ with arrays (expects 'centroid' and a value array).")
    ap.add_argument("--values-key", default="log10_resistivity",
                    help="Key for values in NPZ (default: 'log10_resistivity'). Use 'log10_resistivity' or 'resistivity'.")

    # Accept multiple coordinates
    ap.add_argument("--xy", action="append", nargs=2, type=float, metavar=("X","Y"),
                    help="Add a profile at (X,Y). Can be repeated. If omitted, requires --x and --y.")
    ap.add_argument("--x", type=float, help="Single profile X location (used if --xy is not given)")
    ap.add_argument("--y", type=float, help="Single profile Y location (used if --xy is not given)")

    ap.add_argument("--zmin", type=float, required=True, help="Minimum Z for profile samples")
    ap.add_argument("--zmax", type=float, required=True, help="Maximum Z for profile samples")
    ap.add_argument("--n", dest="nsamples", type=int, default=201, help="Number of Z samples (default: 201)")

    # Method controls
    ap.add_argument("--method", choices=["rbf","idw"], default="rbf", help="Interpolation method (default: rbf)")

    # RBF params
    ap.add_argument("--k", type=int, default=50, help="RBF: number of neighbors (default: 50)")
    ap.add_argument("--kernel", choices=["gaussian","multiquadric","inverse_multiquadric","linear","thin_plate"],
                    default="gaussian", help="RBF kernel (default: gaussian)")
    ap.add_argument("--epsilon", type=float, default=None, help="RBF shape parameter; default auto per depth")
    ap.add_argument("--smooth", type=float, default=1e-6, help="RBF Tikhonov smoothing (default: 1e-6)")

    # IDW params
    ap.add_argument("--power", type=float, default=2.0, help="IDW power (default: 2.0)")

    # Spaces
    ap.add_argument("--in-space", choices=["linear","log10"], default="log10",
                    help="Numeric space of the input values (default: log10)")
    ap.add_argument("--out-space", choices=["linear","log10"], default="linear",
                    help="Numeric space for output profile (default: linear)")

    # Plot options
    ap.add_argument("--logx", action="store_true", help="Use logarithmic x-scale for resistivity")
    ap.add_argument("--z-positive-down", action="store_true", help="Make Z positive downward")

    # Outputs
    ap.add_argument("--out-png", default=None, help="Save plot PNG (optional)")
    ap.add_argument("--out-csv", default=None, help="Save profile CSV for single profile")
    ap.add_argument("--out-npz", default=None, help="Save NPZ with profile arrays (single profile)")
    ap.add_argument("--out-hdf5", default=None, help="Save HDF5 with profile datasets (single profile)")

    args = ap.parse_args()

    data = _np.load(args.npz)
    if "centroid" not in data:
        raise KeyError("NPZ must contain 'centroid' (M,3) array.")
    centroids = data["centroid"]
    # Value selection with graceful fallback
    if args.values_key in data:
        values = data[args.values_key]
        in_space = args.in_space
    else:
        if "log10_resistivity" in data:
            values = data["log10_resistivity"]; in_space = "log10"
        elif "resistivity" in data:
            values = data["resistivity"]; in_space = "linear"
        else:
            raise KeyError(f"Could not find values under '{args.values_key}', 'log10_resistivity', or 'resistivity'.")

    # Optional masking: if 'flag' present, drop rows with flag==1 (fixed)
    if 'flag' in data:
        _m = (data['flag'] != 1)
        centroids = centroids[_m]
        values = values[_m]

    z_samples = _np.linspace(args.zmin, args.zmax, args.nsamples)

    coords = []
    if args.xy:
        coords = [(float(x), float(y)) for x, y in args.xy]
    else:
        if args.x is None or args.y is None:
            raise ValueError("Provide either repeated --xy X Y or a single --x and --y.")
        coords = [(args.x, args.y)]

    # Compute profiles
    profs = []
    labels = []
    for (x, y) in coords:
        if args.method == "rbf":
            prof = vertical_profile_rbf_local(
                x, y, z_samples, centroids, values,
                k=args.k, kernel=args.kernel, epsilon=args.epsilon, smooth=args.smooth,
                in_space=in_space, out_space=args.out_space
            )
            label = f"({x:g},{y:g}) RBF"
        else:
            prof = vertical_profile_idw(
                x, y, z_samples, centroids, values,
                power=args.power, in_space=in_space, out_space=args.out_space
            )
            label = f"({x:g},{y:g}) IDW"

        # Prepare for plotting: ensure positive linear values if log-x requested
        if args.logx:
            prof_plot = 10.0 ** prof if args.out_space == "log10" else prof
        else:
            prof_plot = prof

        profs.append(prof_plot)
        labels.append(label)

    # Plot multi
    fig, ax = plot_vertical_profiles(
        z_samples, profs, labels=labels,
        logx=args.logx, z_positive_down=args.z_positive_down
    )

    # Save plot if requested
    if args.out_png:
        fig.savefig(args.out_png, dpi=200, bbox_inches="tight")
        print("Saved PNG:", args.out_png)

    # If single profile AND outputs requested, save data exactly in requested out-space
    if len(profs) == 1:
        # Recompute in requested out-space (no plotting transforms)
        (x, y) = coords[0]
        if args.method == "rbf":
            prof_exact = vertical_profile_rbf_local(
                x, y, z_samples, centroids, values,
                k=args.k, kernel=args.kernel, epsilon=args.epsilon, smooth=args.smooth,
                in_space=in_space, out_space=args.out_space
            )
        else:
            prof_exact = vertical_profile_idw(
                x, y, z_samples, centroids, values,
                power=args.power, in_space=in_space, out_space=args.out_space
            )

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
                                 method=args.method, kernel=(args.kernel if args.method=="rbf" else ""),
                                 k=(args.k if args.method=="rbf" else -1),
                                 epsilon=(args.epsilon if (args.method=="rbf" and args.epsilon is not None) else _np.nan),
                                 smooth=(args.smooth if args.method=="rbf" else _np.nan),
                                 power=(args.power if args.method=="idw" else _np.nan),
                                 in_space=in_space, out_space=args.out_space)
            print("Saved NPZ:", args.out_npz)

        if args.out_hdf5:
            try:
                import h5py
                with h5py.File(args.out_hdf5, "w") as h5:
                    h5.create_dataset("z", data=z_samples)
                    h5.create_dataset("value", data=prof_exact)
                    h5.attrs["x"] = x; h5.attrs["y"] = y
                    h5.attrs["method"] = args.method
                    h5.attrs["kernel"] = args.kernel if args.method == "rbf" else ""
                    h5.attrs["k"] = args.k if args.method == "rbf" else -1
                    h5.attrs["epsilon"] = args.epsilon if (args.method=="rbf" and args.epsilon is not None) else _np.nan
                    h5.attrs["smooth"] = args.smooth if args.method == "rbf" else _np.nan
                    h5.attrs["power"] = args.power if args.method == "idw" else _np.nan
                    h5.attrs["in_space"] = in_space; h5.attrs["out_space"] = args.out_space
                print("Saved HDF5:", args.out_hdf5)
            except Exception as e:
                print("Failed to save HDF5:", e)

    # If nothing was saved, try showing the plot
    if not args.out_png and not args.out_csv and not args.out_npz and not args.out_hdf5:
        try:
            import matplotlib.pyplot as _plt
            _plt.show()
        except Exception:
            pass

if __name__ == "__main__":
    _cli()

# ------------------------
# Multi-profile plotting
# ------------------------

from typing import Optional, List, Tuple

def plot_vertical_profiles(
    z: np.ndarray,
    profiles: List[np.ndarray],
    labels: Optional[List[str]] = None,
    logx: bool = True,
    z_positive_down: bool = True,
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot multiple vertical profiles on a single Matplotlib axis.

    Parameters
    ----------
    z : (P,) array
        Vertical coordinates (depth or elevation).
    profiles : list of (P,) arrays
        Each entry is a profile sampled at the same z.
    labels : list[str] or None
        Legend labels for each profile.
    logx : bool, default True
        If True, use a logarithmic x-scale for resistivity.
    z_positive_down : bool, default True
        If True, configure axis so that z increases downward.
    ax : matplotlib.axes.Axes or None
        Optional axis to draw on.

    Returns
    -------
    (fig, ax)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 6.5))
    else:
        fig = ax.figure

    # Plot each profile
    for i, prof in enumerate(profiles):
        lab = labels[i] if (labels and i < len(labels)) else None
        ax.plot(prof, z, label=lab)

    # Axis formatting
    ax.set_xlabel("Resistivity")
    ax.set_ylabel("Z")
    if logx:
        ax.set_xscale("log")

    # Configure vertical direction
    if z_positive_down:
        # If z is descending (e.g., 0, -10, -20), invert to make positive downward display
        if z.size >= 2 and z[1] < z[0]:
            ax.invert_yaxis()
    else:
        # If z is ascending and user wants positive upward, invert
        if z.size >= 2 and z[1] > z[0]:
            ax.invert_yaxis()

    if labels:
        ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig, ax
