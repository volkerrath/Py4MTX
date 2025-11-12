"""
femtic_polyline_slice.py

Generate a vertical curtain slice along an XY polyline using RBF/IDW from centroids.
PyVista visualization (StructuredGrid) and export to VTK/PNG/CSV/NPZ.

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking)
"""
from typing import Optional, Literal, Tuple, List
import numpy as np


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


def _rbf_local(q: np.ndarray, pts: np.ndarray, vals: np.ndarray,
               k: int, kernel: str, epsilon: Optional[float], smooth: float) -> float:
    d = np.linalg.norm(pts - q, axis=1)
    nn = np.arange(
        pts.shape[0]) if k >= pts.shape[0] else np.argpartition(d, k-1)[:k]
    P = pts[nn]
    v = vals[nn]
    D = _pairwise_dists(P, P)
    dq = np.linalg.norm(P - q, axis=1)
    if epsilon is None:
        md = np.median(D[D > 0]) if np.any(D > 0) else np.median(d[nn])
        eps = 1.0 / max(md, 1e-9)
    else:
        eps = float(epsilon)
    A = _kernel(D, kernel=kernel, epsilon=eps)
    A.flat[::A.shape[0]+1] += smooth
    b = v
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, b, rcond=None)[0]
    phi_q = _kernel(dq, kernel=kernel, epsilon=eps)
    return float(phi_q @ w)


def _idw(q: np.ndarray, pts: np.ndarray, vals: np.ndarray, power: float, eps: float = 1e-6) -> float:
    d2 = np.sum((pts - q)**2, axis=1) + eps**2
    w = 1.0 / (d2 ** (power/2.0))
    return float(np.sum(w * vals) / np.sum(w))


def build_polyline(points_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    diffs = np.diff(points_xy, axis=0)
    seglen = np.sqrt(np.sum(diffs**2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    dirs = (diffs / np.maximum(seglen[:, None], 1e-12))
    return s, dirs


def sample_polyline(points_xy: np.ndarray, ns: int) -> Tuple[np.ndarray, np.ndarray]:
    s, _ = build_polyline(points_xy)
    total = s[-1]
    S = np.linspace(0.0, total, ns)
    XY = np.empty((ns, 2), dtype=float)
    seg_s = s
    for i, si in enumerate(S):
        j = np.searchsorted(seg_s, si, side="right") - 1
        j = max(0, min(j, len(seg_s)-2))
        t = (si - seg_s[j]) / max(seg_s[j+1]-seg_s[j], 1e-12)
        XY[i] = (1-t)*points_xy[j] + t*points_xy[j+1]
    return S, XY


def curtain_slice(points_xy: np.ndarray, z_samples: np.ndarray,
                  centroids: np.ndarray, values: np.ndarray,
                  method: Literal["rbf", "idw"] = "rbf", k: int = 50, kernel: str = "gaussian",
                  epsilon: Optional[float] = None, smooth: float = 1e-6, power: float = 2.0,
                  in_space: Literal["linear", "log10"] = "log10",
                  out_space: Literal["linear", "log10"] = "linear",
                  ns: int = 301):
    """Interpolate onto a (s,z) grid along the polyline. Returns (S,Z,V,XYZ)."""
    if centroids.shape[1] >= 6:
        m = (centroids[:, 5] != 1)
        centroids = centroids[m]
        values = values[m]
    v_log = _from_linear(values, space=in_space)
    nz = z_samples.size
    S, XY = sample_polyline(points_xy, ns)
    XYZ = np.column_stack([XY, np.zeros(ns)])
    V_log = np.empty((nz, ns), dtype=float)
    for j in range(ns):
        x, y = XY[j]
        for i, z in enumerate(z_samples):
            q = np.array([x, y, z])
            if method == "rbf":
                V_log[i, j] = _rbf_local(
                    q, centroids, v_log, k, kernel, epsilon, smooth)
            else:
                V_log[i, j] = _idw(q, centroids, v_log, power)
    V = _to_linear(V_log, space=out_space)
    return S, z_samples.copy(), V, XYZ


def plot_curtain_pyvista(S: np.ndarray, Z: np.ndarray, V: np.ndarray, XYZ_curve: np.ndarray,
                         title: str = "Resistivity curtain", logscale: bool = True,
                         clim: Optional[Tuple[float, float]] = None, cmap: str = "viridis"):
    try:
        import pyvista as pv
    except Exception as e:
        raise ImportError(
            "PyVista is required for plot_curtain_pyvista(). Install pyvista.") from e
    ns = S.size
    nz = Z.size
    X = np.repeat(XYZ_curve[:, 0][None, :], nz, axis=0)
    Y = np.repeat(XYZ_curve[:, 1][None, :], nz, axis=0)
    ZZ = np.repeat(Z[:, None], ns, axis=1)
    grid = pv.StructuredGrid(
        X.astype(float), Y.astype(float), ZZ.astype(float))
    grid["resistivity"] = V.ravel(order="C")
    p = pv.Plotter()
    if logscale:
        scalars = np.log10(np.clip(V, 1e-12, None)).ravel(order="C")
        grid["log10_resistivity"] = scalars
        scal_name = "log10_resistivity"
    else:
        scal_name = "resistivity"
    sargs = dict(title="log10(ρ)" if logscale else "ρ (Ω·m)")
    p.add_mesh(grid, scalars=scal_name, cmap=cmap,
               clim=clim, scalar_bar_args=sargs)
    p.add_axes()
    p.add_text(title, font_size=12)
    p.show_grid()
    return p, grid


def _cli():
    import argparse
    import numpy as _np
    ap = argparse.ArgumentParser(
        description="Curtain slice along XY polyline (RBF/IDW) with PyVista plotting.")
    ap.add_argument("--npz", required=True,
                    help="NPZ with 'centroid' and values.")
    ap.add_argument("--values-key", default="log10_resistivity",
                    help="Key for values in NPZ (default: log10_resistivity)")
    ap.add_argument(
        "--polyline-csv", help="CSV with columns x,y listing polyline vertices (in order).")
    ap.add_argument("--xy", action="append", nargs=2, type=float, metavar=("X", "Y"),
                    help="Add a polyline vertex (can repeat). Ignored if --polyline-csv is provided.")
    ap.add_argument("--zmin", type=float, required=True)
    ap.add_argument("--zmax", type=float, required=True)
    ap.add_argument("--nz",  type=int,   default=201)
    ap.add_argument("--ns",  type=int,   default=301)
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
    ap.add_argument("--vtk", default=None, help="Write VTK file (.vts)")
    ap.add_argument("--npz-out", default=None, help="Write NPZ with S, Z, V")
    ap.add_argument("--csv-out", default=None, help="Write CSV (s,z,v) rows")
    ap.add_argument("--screenshot", default=None,
                    help="Save PyVista screenshot PNG")
    ap.add_argument("--no-show", action="store_true",
                    help="Do not show interactive window")
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

    if args.polyline_csv:
        import csv
        pts = []
        with open(args.polyline_csv, "r") as f:
            r = csv.reader(f)
            for row in r:
                if not row:
                    continue
                try:
                    x = float(row[0])
                    y = float(row[1])
                    pts.append([x, y])
                except Exception:
                    continue
        points_xy = _np.asarray(pts, dtype=float)
    else:
        if not args.xy or len(args.xy) < 2:
            raise ValueError(
                "Provide at least two vertices via --xy or use --polyline-csv.")
        points_xy = _np.asarray([[float(x), float(y)]
                                for x, y in args.xy], dtype=float)

    z_samples = _np.linspace(args.zmin, args.zmax, args.nz)
    S, Z, V, XYZ = curtain_slice(points_xy, z_samples, centroids, values,
                                 method=args.method, k=args.k, kernel=args.kernel, epsilon=args.epsilon, smooth=args.smooth,
                                 power=args.power, in_space=in_space, out_space=args.out_space, ns=args.ns)
    try:
        p, grid = plot_curtain_pyvista(S, Z, V, XYZ, title=f"Curtain ({args.method})",
                                       logscale=(args.out_space == 'linear'), cmap="viridis")
        if args.vtk:
            try:
                grid.save(args.vtk)
                print("Saved VTK:", args.vtk)
            except Exception as e:
                print("Failed to save VTK:", e)
        if args.screenshot:
            try:
                p.screenshot(args.screenshot)
                print("Saved screenshot:", args.screenshot)
            except Exception as e:
                print("Failed to save screenshot:", e)
        if not args.no_show:
            p.show()
    except ImportError as e:
        print("PyVista not available; skipping interactive plot.", e)

    if args.npz_out:
        _np.savez_compressed(args.npz_out, S=S, Z=Z, V=V, polyline=points_xy)
        print("Saved NPZ:", args.npz_out)
    if args.csv_out:
        import csv
        with open(args.csv_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["s", "z", "value_"+args.out_space])
            for j, s in enumerate(S):
                for i, z in enumerate(Z):
                    w.writerow([s, z, V[i, j]])
        print("Saved CSV:", args.csv_out)


if __name__ == "__main__":
    _cli()
