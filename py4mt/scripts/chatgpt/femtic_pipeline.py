"""
femtic_pipeline.py

End-to-end pipeline:
  1) Load mesh + resistivity (external readers).
  2) Build element arrays (external).
  3) Apply optional edits to log10 resistivity and limits.
  4) Aggregate per region (median/mean in log10 space).
  5) Inject values (and optional limits) into a template rho-file, save new file.
  6) Optional: vertical curtain slice along polyline with PyVista preview.

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking)
"""
import numpy as np
from typing import Optional, Tuple
# External deps assumed available in your project:
# from femtic_mesh_reader import load_femtic_mesh
# from femtic_element_array import build_element_arrays, save_npz, save_hdf5, save_csv_compact
from femtic_rho_inject import write_rho_from_vector_with_flags
from femtic_polyline_slice import curtain_slice, plot_curtain_pyvista
import matplotlib.pyplot as plt

def _plot_plan_view(points_xy, centroids_xy, out_png=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    if centroids_xy is not None and np.size(centroids_xy) > 0:
        ax.scatter(centroids_xy[:,0], centroids_xy[:,1], s=2, alpha=0.2)
    ax.plot(points_xy[:,0], points_xy[:,1], '-o', ms=4, lw=2)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    ax.set_title('Polyline plan view')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if out_png: fig.savefig(out_png, dpi=200, bbox_inches='tight'); print('Saved plan-view PNG:', out_png)
    return fig, ax

def apply_log10_edits(rho_log: np.ndarray, shift: float = 0.0, scale: float = 1.0,
                      clip_low: Optional[float] = None, clip_high: Optional[float] = None) -> np.ndarray:
    res = scale * rho_log + shift
    if clip_low is not None or clip_high is not None:
        lo = -np.inf if clip_low is None else clip_low; hi = np.inf if clip_high is None else clip_high
        res = np.clip(res, lo, hi)
    return res

def rebuild_limits_from_margin(rho_log: np.ndarray, margin: float) -> Tuple[np.ndarray, np.ndarray]:
    return rho_log - margin, rho_log + margin

def aggregate_per_region(region: np.ndarray, rho_log: np.ndarray,
                         lo_log: Optional[np.ndarray], hi_log: Optional[np.ndarray],
                         method: str = "median_log10"):
    reg_ids = np.unique(region.astype(int)); reg_ids.sort(); K = reg_ids.size
    v_log = np.empty(K, dtype=float); lo_agg = np.empty(K, dtype=float); hi_agg = np.empty(K, dtype=float)
    for i, rid in enumerate(reg_ids):
        m = (region == rid); vals = rho_log[m]
        v_log[i] = float(np.mean(vals)) if method == "mean_log10" else float(np.median(vals))
        lo_agg[i] = float(np.median(lo_log[m])) if (lo_log is not None and m.any()) else np.nan
        hi_agg[i] = float(np.median(hi_log[m])) if (hi_log is not None and m.any()) else np.nan
    return reg_ids, v_log, lo_agg, hi_agg

def run_pipeline(mesh_path: str, rho_path: Optional[str],
                 template_rho: str, out_rho: str,
                 margin: float = 0.5, shift: float = 0.0, scale: float = 1.0,
                 clip_low: Optional[float] = None, clip_high: Optional[float] = None,
                 agg: str = "median_log10", write_space: str = "linear",
                 append_limits: bool = True, save_npz_path: Optional[str] = None,
                 save_hdf5_path: Optional[str] = None, save_csv_path: Optional[str] = None,
                 fmt: str = "{:.6g}", value_col: Optional[int] = None,
                 lower_col: int = 2, upper_col: int = 3, n_col: int = 4,
                 override_n: Optional[int] = None, flag_col: Optional[int] = None,
                 set_flag: Optional[int] = None,
                 # slice options
                 slice_enabled: bool = False, slice_polyline_csv: Optional[str] = None, slice_xy: Optional[list] = None,
                 slice_zmin: Optional[float] = None, slice_zmax: Optional[float] = None,
                 slice_nz: int = 201, slice_ns: int = 301, slice_method: str = 'rbf',
                 slice_k: int = 50, slice_kernel: str = 'gaussian', slice_epsilon: Optional[float] = None,
                 slice_smooth: float = 1e-6, slice_power: float = 2.0, slice_in_space: str = 'log10',
                 slice_out_space: str = 'linear', slice_vtk: Optional[str] = None, slice_npz_out: Optional[str] = None,
                 slice_csv_out: Optional[str] = None, slice_screenshot: Optional[str] = None,
                 slice_no_show: bool = True, slice_plan_png: Optional[str] = None) -> None:
    # Placeholder: load data using external modules if available (mesh_reader/element_array)
    # Here we assume `arr` already exists externally; otherwise this function should be extended.
    raise NotImplementedError("run_pipeline depends on project-specific mesh/array readers not included here.")

def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="(Skeleton) FEMTIC pipeline: arrays -> edits -> aggregate -> inject rho (+ optional slice).")
    ap.add_argument("--mesh", required=False, help="Path to mesh.dat (reader not included here)")
    ap.add_argument("--rho", default=None, help="Path to resistivity_block_iterX.dat")
    ap.add_argument("--template-rho", required=True, help="Template rho-file to inject into")
    ap.add_argument("--out-rho", required=True, help="Output rho-file path (new name)")
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--shift", type=float, default=0.0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--clip-low", type=float, default=None)
    ap.add_argument("--clip-high", type=float, default=None)
    ap.add_argument("--agg", choices=["median_log10","mean_log10"], default="median_log10")
    ap.add_argument("--write-space", choices=["linear","log10"], default="linear")
    ap.add_argument("--no-append-limits", action="store_true")
    ap.add_argument("--save-npz", default=None)
    ap.add_argument("--save-hdf5", default=None)
    ap.add_argument("--save-csv", default=None)
    ap.add_argument("--format", dest="fmt", default="{:.6g}")
    ap.add_argument("--value-col", type=int, default=None)
    ap.add_argument("--lower-col", type=int, default=2)
    ap.add_argument("--upper-col", type=int, default=3)
    ap.add_argument("--n-col", type=int, default=4)
    ap.add_argument("--n", dest="override_n", type=int, choices=[1,2,3], default=None)
    ap.add_argument("--flag-col", type=int, default=None)
    ap.add_argument("--set-flag", type=int, choices=[0,1], default=None)
    args = ap.parse_args()
    print("This pipeline skeleton requires your project-specific mesh/array readers. Use femtic_profiles.py or femtic_polyline_slice.py directly for now.")

if __name__ == "__main__":
    _cli()
