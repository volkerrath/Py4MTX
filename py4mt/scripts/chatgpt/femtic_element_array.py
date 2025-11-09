
"""
femtic_element_array.py

Pure-NumPy fast builders & writers for FEMTIC element-wise data.

Arrays produced (all vectorized, no per-element Python loops):
  id                  (M,)   int64
  nodes               (M,4)  int64
  coords              (M,4,3) float64
  centroid            (M,3)  float64
  region              (M,)   int64
  log10_resistivity   (M,)   float64
  rho_lower           (M,)   float64
  rho_upper           (M,)   float64

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-11-09
"""

from typing import Dict, Any, Optional
import numpy as np
from femtic_mesh_reader import load_femtic_mesh, map_regions_to_resistivity

ArrayDict = Dict[str, np.ndarray]


def build_element_arrays(mesh: Dict[str, Any],
                         margin: float = 0.5,
                         clip_nan: bool = True) -> ArrayDict:
    """
    Vectorized build of per-element arrays.

    Parameters
    ----------
    mesh : dict
        Must contain: 'nodes' (N,3), 'elements' (M,4), 'regions' (M,)
        and optionally 'resistivity' (R,)
    margin : float, default 0.5
        Log10 margin for lower/upper bounds.
    clip_nan : bool, default True
        Replace NaNs in log10 resistivity with 0.

    Returns
    -------
    arrays : dict[str, np.ndarray]
        Keys: id, nodes, coords, centroid, region, log10_resistivity, rho_lower, rho_upper
    """
    nodes = mesh["nodes"]
    elements = mesh["elements"]
    regions = mesh["regions"]

    coords = nodes[elements]
    centroid = coords.mean(axis=1)

    rho = map_regions_to_resistivity(regions, mesh.get("resistivity"))
    rho_log = np.where(rho > 0.0, np.log10(rho), np.nan)
    if clip_nan:
        rho_log = np.nan_to_num(rho_log, nan=0.0)

    rho_lower = rho_log - margin
    rho_upper = rho_log + margin

    M = elements.shape[0]
    arrs: ArrayDict = {
        "id": np.arange(M, dtype=np.int64),
        "nodes": elements.astype(np.int64, copy=False),
        "coords": coords.astype(np.float64, copy=False),
        "centroid": centroid.astype(np.float64, copy=False),
        "region": regions.astype(np.int64, copy=False),
        "log10_resistivity": rho_log.astype(np.float64, copy=False),
        "rho_lower": rho_lower.astype(np.float64, copy=False),
        "rho_upper": rho_upper.astype(np.float64, copy=False),
    }
    return arrs


def save_npz(arrays: ArrayDict, path: str, compressed: bool = True) -> None:
    """Save arrays to a (compressed) NPZ."""
    if compressed:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def save_hdf5(arrays: ArrayDict, path: str,
              compression: Optional[str] = "gzip",
              compression_opts: int = 4) -> None:
    """Save arrays to HDF5 datasets (requires h5py)."""
    import h5py
    with h5py.File(path, "w") as h5:
        for k, v in arrays.items():
            h5.create_dataset(
                k, data=v,
                compression=compression,
                compression_opts=(compression_opts if compression else None)
            )


def save_csv_compact(arrays: ArrayDict, path: str) -> None:
    """
    Save a compact CSV (one row per element) with key fields only:
    id,region,node0..node3,cx,cy,cz,log10_resistivity,rho_lower,rho_upper
    """
    idv = arrays["id"]
    region = arrays["region"]
    nodes = arrays["nodes"]
    ctr = arrays["centroid"]
    rho_log = arrays["log10_resistivity"]
    lo = arrays["rho_lower"]
    hi = arrays["rho_upper"]

    data = np.column_stack([
        idv, region,
        nodes[:, 0], nodes[:, 1], nodes[:, 2], nodes[:, 3],
        ctr[:, 0], ctr[:, 1], ctr[:, 2],
        rho_log, lo, hi,
    ])
    header = "id,region,node0,node1,node2,node3,cx,cy,cz,log10_resistivity,rho_lower,rho_upper"
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def _cli():
    """Command-line interface for fast FEMTIC element array building and export."""
    import argparse

    ap = argparse.ArgumentParser(description="FEMTIC element arrays (pure NumPy, fast).")
    ap.add_argument("--mesh", required=True, help="mesh.dat")
    ap.add_argument("--rho", default=None, help="resistivity_block_iterX.dat")
    ap.add_argument("--margin", type=float, default=0.5, help="log10 margin for bounds")
    ap.add_argument("--out-npz", default=None, help="write NPZ file")
    ap.add_argument("--out-hdf5", default=None, help="write HDF5 file")
    ap.add_argument("--out-csv", default=None, help="write compact CSV")
    args = ap.parse_args()

    mesh = load_femtic_mesh(args.mesh, args.rho)
    arrays = build_element_arrays(mesh, margin=args.margin)

    if args.out_npz:
        save_npz(arrays, args.out_npz)
        print(f"Wrote NPZ: {args.out_npz}")
    if args.out_hdf5:
        save_hdf5(arrays, args.out_hdf5)
        print(f"Wrote HDF5: {args.out_hdf5}")
    if args.out_csv:
        save_csv_compact(arrays, args.out_csv)
        print(f"Wrote CSV: {args.out_csv}")


if __name__ == "__main__":
    _cli()
