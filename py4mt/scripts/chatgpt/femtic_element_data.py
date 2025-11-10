
"""
femtic_element_data.py

Builds a per-element data structure for FEMTIC meshes and writes it to ASCII (CSV/JSON),
HDF5, and NPZ formats.

Per-element fields
------------------
- id : int
- nodes : list[int]           (length 4, 0-based)
- coords : list[list[float]]  (4 x 3; node coordinates of the element)
- centroid : list[float]      (3, mean of node coordinates)
- region : int
- log10_resistivity : float
- rho_lower : float           (log10 bound lower)
- rho_upper : float           (log10 bound upper)

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-11-09
"""

from typing import Dict, Any, List, Optional
import numpy as np

from femtic_mesh_reader import (
    load_femtic_mesh,
    element_centroids,
    map_regions_to_resistivity,
)


def build_element_data_structure(
    mesh: Dict[str, Any],
    margin: float = 0.5,
    clip_nan: bool = True,
) -> List[Dict[str, Any]]:
    """
    Build a detailed per-element data structure from a FEMTIC mesh.

    Parameters
    ----------
    mesh : dict
        FEMTIC mesh dict with keys 'nodes', 'elements', 'regions', optional 'resistivity'.
    margin : float, default 0.5
        Margin in log10-space applied to bounds:
        rho_lower = log10_resistivity - margin
        rho_upper = log10_resistivity + margin
    clip_nan : bool, default True
        Replace NaNs in log10_resistivity (e.g., from non-positive rho) by 0.0.

    Returns
    -------
    elements : list of dict
        One dict per element with the fields documented in the module docstring.
    """
    nodes = mesh["nodes"]
    elements = mesh["elements"]
    regions = mesh["regions"]
    resistivity = mesh.get("resistivity", None)

    rho = map_regions_to_resistivity(regions, resistivity)
    rho_log = np.where(rho > 0.0, np.log10(rho), np.nan)
    if clip_nan:
        rho_log = np.nan_to_num(rho_log, nan=0.0)

    ctr = element_centroids(nodes, elements)

    out: List[Dict[str, Any]] = []
    for i, conn in enumerate(elements):
        coords = nodes[conn]  # (4,3)
        rlog = float(rho_log[i])
        out.append(
            {
                "id": int(i),
                "nodes": [int(v) for v in conn.tolist()],
                "coords": coords.tolist(),
                "centroid": ctr[i].tolist(),
                "region": int(regions[i]),
                "log10_resistivity": rlog,
                "rho_lower": rlog - margin,
                "rho_upper": rlog + margin,
            }
        )
    return out


def _structured_arrays_from_elements(elements: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """
    Convert list of element dicts to numpy arrays suitable for CSV/HDF5/NPZ writing.

    Parameters
    ----------
    elements : list of dict
        Output of build_element_data_structure().

    Returns
    -------
    arrays : dict[str, np.ndarray]
        Keys:
          - id : (M,) int64
          - nodes : (M,4) int64
          - coords : (M,4,3) float64
          - centroid : (M,3) float64
          - region : (M,) int64
          - log10_resistivity : (M,) float64
          - rho_lower : (M,) float64
          - rho_upper : (M,) float64
    """
    M = len(elements)
    ids = np.empty(M, dtype=np.int64)
    nodes = np.empty((M, 4), dtype=np.int64)
    coords = np.empty((M, 4, 3), dtype=np.float64)
    centroid = np.empty((M, 3), dtype=np.float64)
    region = np.empty(M, dtype=np.int64)
    rho_log = np.empty(M, dtype=np.float64)
    rho_lo = np.empty(M, dtype=np.float64)
    rho_hi = np.empty(M, dtype=np.float64)

    for i, el in enumerate(elements):
        ids[i] = el["id"]
        nodes[i] = np.asarray(el["nodes"], dtype=np.int64)
        coords[i] = np.asarray(el["coords"], dtype=np.float64)
        centroid[i] = np.asarray(el["centroid"], dtype=np.float64)
        region[i] = el["region"]
        rho_log[i] = el["log10_resistivity"]
        rho_lo[i] = el["rho_lower"]
        rho_hi[i] = el["rho_upper"]

    return {
        "id": ids,
        "nodes": nodes,
        "coords": coords,
        "centroid": centroid,
        "region": region,
        "log10_resistivity": rho_log,
        "rho_lower": rho_lo,
        "rho_upper": rho_hi,
    }


def write_elements_ascii(
    elements: List[Dict[str, Any]],
    path: str,
    fmt: str = "csv",
) -> None:
    """
    Write element data to an ASCII file.

    Parameters
    ----------
    elements : list of dict
        Output of build_element_data_structure().
    path : str
        Output path. If fmt='csv' the file will be CSV; if 'json' it will be JSON.
    fmt : {'csv','json'}, default 'csv'
        Output format.

    Returns
    -------
    None
    """
    import json as _json
    arrays = _structured_arrays_from_elements(elements)

    if fmt.lower() == "json" or path.lower().endswith(".json"):  # JSON full dict list
        with open(path, "w") as f:
            _json.dump(elements, f, indent=2)
        return

    # CSV flat: id, region, nodes[4], centroid[3], log10, lower, upper
    header = "id,region,node0,node1,node2,node3,cx,cy,cz,log10_resistivity,rho_lower,rho_upper"
    data = np.column_stack([
        arrays["id"],
        arrays["region"],
        arrays["nodes"][:, 0],
        arrays["nodes"][:, 1],
        arrays["nodes"][:, 2],
        arrays["nodes"][:, 3],
        arrays["centroid"][:, 0],
        arrays["centroid"][:, 1],
        arrays["centroid"][:, 2],
        arrays["log10_resistivity"],
        arrays["rho_lower"],
        arrays["rho_upper"],
    ])
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def write_elements_hdf5(
    elements: List[Dict[str, Any]],
    path: str,
    compression: Optional[str] = "gzip",
    compression_opts: int = 4,
) -> None:
    """
    Write element data to an HDF5 file with fixed-shape datasets.

    Datasets created:
      /id                  (M,)
      /nodes               (M,4)
      /coords              (M,4,3)
      /centroid            (M,3)
      /region              (M,)
      /log10_resistivity   (M,)
      /rho_lower           (M,)
      /rho_upper           (M,)

    Parameters
    ----------
    elements : list of dict
        Output of build_element_data_structure().
    path : str
        Output HDF5 file path.
    compression : str or None, default 'gzip'
        Compression filter for datasets (None to disable).
    compression_opts : int, default 4
        Compression level/opts (if applicable).

    Returns
    -------
    None
    """
    import h5py  # type: ignore

    arr = _structured_arrays_from_elements(elements)
    with h5py.File(path, "w") as h5:
        for key, val in arr.items():
            h5.create_dataset(
                key,
                data=val,
                compression=compression,
                compression_opts=compression_opts if compression else None,
            )


def write_elements_npz(
    elements: List[Dict[str, Any]],
    path: str,
    compressed: bool = True,
) -> None:
    """
    Write element data to a NumPy NPZ archive.

    Parameters
    ----------
    elements : list of dict
        Output of build_element_data_structure().
    path : str
        Output file path (e.g., 'elements.npz').
    compressed : bool, default True
        If True, use numpy.savez_compressed(); else use numpy.savez().

    Returns
    -------
    None
    """
    arrays = _structured_arrays_from_elements(elements)
    if compressed:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


# -------------------------
# CLI
# -------------------------

def _cli():
    """
    Command-line tool to build element data and write ASCII/HDF5/NPZ outputs.

    Usage examples
    --------------
    python -m femtic_element_data --mesh mesh.dat --rho resistivity_block_iter0.dat \\
        --margin 0.3 --out-csv elements.csv --out-hdf5 elements.h5 --out-npz elements.npz

    python -m femtic_element_data --mesh mesh.dat --rho resistivity_block_iter0.dat \\
        --json elements.json
    """
    import argparse

    ap = argparse.ArgumentParser(
        description="Build per-element data and export to CSV/JSON/HDF5/NPZ")
    ap.add_argument("--mesh", required=True, help="Path to mesh.dat")
    ap.add_argument("--rho", default=None,
                    help="Path to resistivity_block_iterX.dat")
    ap.add_argument("--margin", type=float, default=0.5,
                    help="Log10 margin for lower/upper limits")
    ap.add_argument("--out-csv", default=None,
                    help="Write compact CSV (id, region, nodes, centroid, log10, lower, upper)")
    ap.add_argument("--json", dest="out_json", default=None,
                    help="Write full JSON with coords included")
    ap.add_argument("--out-hdf5", default=None,
                    help="Write HDF5 with full arrays including coords")
    ap.add_argument("--out-npz", default=None,
                    help="Write compressed NumPy NPZ archive with all arrays")
    args = ap.parse_args()

    mesh = load_femtic_mesh(args.mesh, args.rho)
    elements = build_element_data_structure(mesh, margin=args.margin)

    if args.out_csv:
        write_elements_ascii(elements, args.out_csv, fmt="csv")
        print(f"Wrote CSV: {args.out_csv}")  # noqa: T201
    if args.out_json:
        write_elements_ascii(elements, args.out_json, fmt="json")
        print(f"Wrote JSON: {args.out_json}")  # noqa: T201
    if args.out_hdf5:
        write_elements_hdf5(elements, args.out_hdf5)
        print(f"Wrote HDF5: {args.out_hdf5}")  # noqa: T201
    if args.out_npz:
        write_elements_npz(elements, args.out_npz)
        print(f"Wrote NPZ: {args.out_npz}")  # noqa: T201


if __name__ == "__main__":
    _cli()
