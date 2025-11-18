"""
femtic_mesh_io.py
=================

Parser utilities for FEMTIC unstructured TETRA meshes.

The parser extracts tetrahedron centroids from a mesh file in the
following format (as observed in this project):

    - Line 1: literal "TETRA"
    - Line 2: N_nodes (integer)
    - Next N_nodes lines:
        <node_id> <x> <y> <z>
    - Next line: N_tets (integer)
    - Next N_tets lines:
        <tet_id> <... neighbors ...> <n1> <n2> <n3> <n4> [optional extras ...]

The last four integers on each tetra line are interpreted as the
node indices defining the tetrahedron. Centroids are computed as
the average of the four node coordinates.

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-11-18
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def read_femtic_tetra_centroids(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a FEMTIC TETRA mesh and return centroids and tetra IDs.

    Parameters
    ----------
    path
        Path to the mesh file (e.g. ``mesh.dat``).

    Returns
    -------
    centroids
        Array of shape ``(N_tet, 3)`` with tetrahedron centroids.
    tet_ids
        Array of shape ``(N_tet,)`` with tetra indices as given in the
        file.

    Raises
    ------
    ValueError
        If the first line is not ``"TETRA"`` or if a node/element line
        does not match the expected structure.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        kind = f.readline().strip()
        if kind.upper() != "TETRA":
            raise ValueError("Unsupported mesh kind (expected 'TETRA').")

        n_nodes = int(f.readline().strip())

        node_idx = np.empty(n_nodes, dtype=np.int64)
        nodes = np.empty((n_nodes, 3), dtype=np.float64)

        # Read node coordinates
        for i in range(n_nodes):
            parts = f.readline().split()
            if len(parts) < 4:
                raise ValueError("Malformed node line: expected <id> x y z.")
            node_idx[i] = int(parts[0])
            nodes[i, 0] = float(parts[1])
            nodes[i, 1] = float(parts[2])
            nodes[i, 2] = float(parts[3])

        # Build optional ID -> index mapping if node indices are not 0..N-1
        inv = None
        if not np.array_equal(node_idx, np.arange(n_nodes, dtype=np.int64)):
            inv = -np.ones(int(node_idx.max()) + 1, dtype=np.int64)
            inv[node_idx] = np.arange(n_nodes, dtype=np.int64)

        n_tet = int(f.readline().strip())
        tet_ids = np.empty(n_tet, dtype=np.int64)
        centroids = np.empty((n_tet, 3), dtype=np.float64)

        # Read tetrahedra
        for i in range(n_tet):
            parts = f.readline().split()
            if len(parts) < 5:
                raise ValueError(
                    "Malformed tet line: expected <id> ... <n1> <n2> <n3> <n4>.",
                )

            tid = int(parts[0])
            n1, n2, n3, n4 = (
                int(parts[-4]),
                int(parts[-3]),
                int(parts[-2]),
                int(parts[-1]),
            )

            if inv is not None:
                n1, n2, n3, n4 = (
                    int(inv[n1]),
                    int(inv[n2]),
                    int(inv[n3]),
                    int(inv[n4]),
                )

            c = (nodes[n1] + nodes[n2] + nodes[n3] + nodes[n4]) / 4.0

            tet_ids[i] = tid
            centroids[i, :] = c

    return centroids, tet_ids
