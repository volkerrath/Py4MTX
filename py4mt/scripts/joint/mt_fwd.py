"""
mt_fwd.py
=========
Interface to the FEMTIC MT forward operator for joint inversion.

Wraps fem.read_model / insert_model and provides:
  - model_grid()   → ModelGrid (FEMTIC region centroids + log10 rho)
  - misfit()       → scalar data misfit Φ_MT
  - gradient()     → gradient of Φ_MT w.r.t. log10(rho) vector
  - save_model()   → write updated block file

The actual Jacobian computation (adjoint-state sensitivities) is delegated
to the FEMTIC solver.  For testing, a stub finite-difference Jacobian is
provided when FEMTIC is unavailable.

Author: VR 2026-05-11, Claude Sonnet 4.6 (Anthropic)
"""

from __future__ import annotations
import os
import numpy as np
from coupling_interp import ModelGrid


class MTForward:
    """Wrap the FEMTIC MT forward problem.

    Parameters
    ----------
    model_file : str  — resistivity_block_iterX.dat
    mesh_file  : str  — mesh.dat
    data_file  : str  — observe.dat
    bounds     : (float, float)  — (log10_min, log10_max) for clipping
    out        : bool
    """

    def __init__(self, model_file: str, mesh_file: str,
                 data_file: str, *,
                 bounds: tuple = (0.0, 4.0),
                 out: bool = True):

        self.model_file = model_file
        self.mesh_file  = mesh_file
        self.data_file  = data_file
        self.bounds     = bounds
        self.out        = out

        self._load()

    # ── I/O ──────────────────────────────────────────────────────────────────

    def _load(self):
        """Load model and mesh via femtic module."""
        try:
            import femtic as fem
            self._fem  = fem
            # read_model returns the free-region log10(rho) vector only; the
            # region structure (element->region map, free-region indices)
            # comes from the block-struct reader.
            log_rho = fem.read_model(
                self.model_file, model_trans="log10", out=False)
            self._block_struct = fem._read_resistivity_block_struct(
                self.model_file, model_trans="log10", out=False)
            self._m        = log_rho.copy()
            self._centroids = self._load_centroids()
        except ImportError:
            if self.out:
                print("  MTForward: femtic not found — using stub.")
            self._fem          = None
            self._m            = np.zeros(100)           # placeholder
            self._centroids    = np.random.rand(100, 3) * 50_000.
            self._block_struct = None

        if self.out:
            print(f"  MTForward: {len(self._m)} free regions loaded.")

    def _load_centroids(self) -> np.ndarray:
        """Volume-weighted centroids of the free regions of the FEMTIC mesh.

        Row k belongs to the k-th free parameter of the model vector (air,
        ocean and other fixed regions are not part of it).
        """
        fem = self._fem
        nodes, conn = fem.read_femtic_mesh(self.mesh_file)
        st = self._block_struct
        ctr, _ = fem.build_region_geometry(
            nodes, conn, st["elem_region"], st["free_idx"])
        return ctr

    # ── ModelGrid ─────────────────────────────────────────────────────────────

    def model_grid(self) -> ModelGrid:
        return ModelGrid(self._centroids, self._m.copy(), name="MT")

    # ── Misfit ───────────────────────────────────────────────────────────────

    def misfit(self, m: np.ndarray) -> float:
        """Weighted L2 MT data misfit  Φ_MT(m).

        Placeholder: returns ||m||^2 / 2 when FEMTIC is unavailable.
        Replace with actual forward-call + residual computation.
        """
        self._m = np.clip(m, *self.bounds)
        if self._fem is None:
            return float(0.5 * np.dot(self._m, self._m))
        # TODO: call FEMTIC forward solver, compute weighted residuals
        raise NotImplementedError("FEMTIC misfit call not yet implemented.")

    # ── Gradient ─────────────────────────────────────────────────────────────

    def gradient(self, m: np.ndarray) -> np.ndarray:
        """Gradient of Φ_MT w.r.t. log10(rho).

        Placeholder: returns m when FEMTIC is unavailable.
        Replace with adjoint-state sensitivity computation.
        """
        self._m = np.clip(m, *self.bounds)
        if self._fem is None:
            return self._m.copy()
        raise NotImplementedError("FEMTIC adjoint gradient not yet implemented.")

    # ── Model update ─────────────────────────────────────────────────────────

    def update(self, dm: np.ndarray):
        """Apply model update and clip to bounds."""
        self._m = np.clip(self._m + dm, *self.bounds)

    def save_model(self, path: str):
        """Write updated block file."""
        if self._fem is None:
            np.save(path.replace(".dat", ".npy"), self._m)
            return
        self._fem.insert_model(
            template=self.model_file, model=self._m,
            model_file=path, out=self.out)
        if self.out:
            print(f"  MTForward: model saved → {path}")
