'''
femtic_mesh_io.py
=================

Lightweight parser for FEMTIC unstructured TETRA meshes to extract
cell centroids.

Format (as observed)
--------------------
- Line 1: literal "TETRA"
- Line 2: N_nodes (int)
- Next N_nodes lines: <node_id> <x> <y> <z>
- Next line: N_tets (int)
- Next N_tets lines: <tet_id> <... neighbors ...> <n1> <n2> <n3> <n4> [extras...]

We compute tetra centroids as the mean of the four node coordinates, using
the last four integers on each tet line as node IDs.

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-11-11
'''
from __future__ import annotations

from typing import Tuple
import numpy as np


def read_femtic_tetra_centroids(path: str) -> Tuple[np.ndarray, np.ndarray]:
    '''Read a FEMTIC TETRA mesh and return centroids and tetra IDs.'''
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        kind = f.readline().strip()
        if kind.upper() != "TETRA":
            raise ValueError("Unsupported mesh kind (expected 'TETRA').")
        n_nodes = int(f.readline().strip())

        node_idx = np.empty(n_nodes, dtype=np.int64)
        nodes = np.empty((n_nodes, 3), dtype=np.float64)

        for i in range(n_nodes):
            parts = f.readline().split()
            if len(parts) < 4:
                raise ValueError("Malformed node line: expected <id> x y z.")
            node_idx[i] = int(parts[0])
            nodes[i, 0] = float(parts[1])
            nodes[i, 1] = float(parts[2])
            nodes[i, 2] = float(parts[3])

        inv = None
        if not np.array_equal(node_idx, np.arange(n_nodes, dtype=np.int64)):
            inv = -np.ones(int(node_idx.max()) + 1, dtype=np.int64)
            inv[node_idx] = np.arange(n_nodes, dtype=np.int64)

        n_tet = int(f.readline().strip())
        tet_ids = np.empty(n_tet, dtype=np.int64)
        centroids = np.empty((n_tet, 3), dtype=np.float64)

        for i in range(n_tet):
            parts = f.readline().split()
            if len(parts) < 5:
                raise ValueError("Malformed tet line: expected <id> ... <n1> <n2> <n3> <n4>.")
            tid = int(parts[0])
            n1, n2, n3, n4 = (int(parts[-4]), int(parts[-3]), int(parts[-2]), int(parts[-1]))
            if inv is not None:
                n1, n2, n3, n4 = int(inv[n1]), int(inv[n2]), int(inv[n3]), int(inv[n4])
            c = (nodes[n1] + nodes[n2] + nodes[n3] + nodes[n4]) / 4.0
            tet_ids[i] = tid
            centroids[i, :] = c

    return centroids, tet_ids
