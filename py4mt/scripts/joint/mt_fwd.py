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
from model_interp import ModelGrid


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
            log_rho, self._block_struct = fem.read_model(
                self.model_file, model_trans="log10")
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
        """Compute region centroids from the FEMTIC mesh."""
        fem = self._fem
        nodes, conn, region_id = fem.read_femtic_mesh(self.mesh_file)
        # Average node coordinates per region
        n_reg = self._m.shape[0]
        ctr   = np.zeros((n_reg, 3))
        cnt   = np.zeros(n_reg, dtype=int)
        for ie, rid in enumerate(region_id):
            if 0 <= rid < n_reg:
                ctr[rid] += nodes[conn[ie]].mean(axis=0)
                cnt[rid] += 1
        cnt = np.maximum(cnt, 1)
        return ctr / cnt[:, np.newaxis]

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
            self._m, template=self.model_file,
            model_trans="log10", out_file=path)
        if self.out:
            print(f"  MTForward: model saved → {path}")
