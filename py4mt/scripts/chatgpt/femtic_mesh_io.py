
"""
femtic_mesh_io.py
==================
Lightweight parser for FEMTIC unstructured TETRA meshes to extract **cell centroids**.

Format (inferred from sample):
- Line 1: literal 'TETRA'
- Line 2: N_nodes (integer)
- Next N_nodes lines: <idx> <x> <y> <z>
- Next line: N_tetra (integer)
- Next N_tetra lines: <idx> n1 n2 n3 n4 [ ... extras ... ]

We compute tetra centroids as the mean of the four node coordinates.

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-11-10
"""

from __future__ import annotations
from typing import Tuple
import numpy as np

def read_femtic_tetra_centroids(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a FEMTIC TETRA mesh and return (centroids, tet_ids).

    Parameters
    ----------
    path : str
        Path to the mesh file.

    Returns
    -------
    centroids : (N_tet, 3) float64 ndarray
        Tetrahedron centroids.
    tet_ids : (N_tet,) int64 ndarray
        Tetra indices as listed in the file (0-based as seen in sample).

    Notes
    -----
    - We ignore any extra integers in the element lines beyond the four node IDs.
    - Assumes node indexing in the file matches the given IDs (no +1 shift).
    """
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        kind = f.readline().strip()
        if kind.upper() != 'TETRA':
            raise ValueError(f"Unsupported mesh kind '{kind}' (expected 'TETRA').")
        n_nodes = int(f.readline().strip())

        # Read node block
        node_idx = np.empty(n_nodes, dtype=np.int64)
        nodes = np.empty((n_nodes, 3), dtype=np.float64)
        for i in range(n_nodes):
            parts = f.readline().split()
            if len(parts) < 4:
                raise ValueError("Malformed node line: expected <id> x y z")
            node_idx[i] = int(parts[0])
            nodes[i, 0] = float(parts[1]); nodes[i, 1] = float(parts[2]); nodes[i, 2] = float(parts[3])

        # Potential reindexing map if ids are not [0..n_nodes-1]
        if not np.array_equal(node_idx, np.arange(n_nodes, dtype=np.int64)):
            # Build mapping from file id -> 0..n_nodes-1
            inv = -np.ones(node_idx.max()+1, dtype=np.int64)
            inv[node_idx] = np.arange(n_nodes, dtype=np.int64)
        else:
            inv = None

        n_tet = int(f.readline().strip())
        tet_ids = np.empty(n_tet, dtype=np.int64)
        centroids = np.empty((n_tet, 3), dtype=np.float64)

        # Parse elements
        for i in range(n_tet):
            parts = f.readline().split()
            if len(parts) < 5:
                raise ValueError("Malformed tet line: expected <id> n1 n2 n3 n4 ...")
            tid = int(parts[0])
            n1, n2, n3, n4 = (int(parts[-4]), int(parts[-3]), int(parts[-2]), int(parts[-1]))
            if inv is not None:
                n1 = int(inv[n1]); n2 = int(inv[n2]); n3 = int(inv[n3]); n4 = int(inv[n4])
            c = (nodes[n1] + nodes[n2] + nodes[n3] + nodes[n4]) / 4.0
            tet_ids[i] = tid
            centroids[i, :] = c
    return centroids, tet_ids
