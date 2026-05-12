"""
seistomo_fwd.py
===============
Interface to the seismic traveltime tomography forward operator.

Provides the same API as MTForward so that JointInversionState can treat
both methods symmetrically:
  - model_grid()   → ModelGrid (voxel centroids + Vp [km/s])
  - misfit()       → scalar traveltime misfit Φ_seis
  - gradient()     → gradient of Φ_seis w.r.t. Vp vector
  - save_model()   → write updated velocity file

The seismic forward problem is modelled here as straight-ray traveltime
tomography.  The sensitivity matrix (ray-path length in each voxel) is
computed analytically.  Replace with a finite-difference eikonal solver
(e.g. FMTOMO, Podvin-Lecomte) for production use.

Data file format (ASCII, one ray per line):
    xs ys zs  xr yr zr  t_obs  sigma_t
where xs,ys,zs = source; xr,yr,zr = receiver; t_obs = observed traveltime [s];
sigma_t = standard deviation [s].

Author: VR 2026-05-11, Claude Sonnet 4.6 (Anthropic)
"""

from __future__ import annotations
import numpy as np
from model_interp import ModelGrid


class SeisTomoForward:
    """Straight-ray traveltime tomography forward operator.

    Parameters
    ----------
    vel_file  : str   — NPY file, shape (N_seis, 4): [x, y, z, Vp]
    data_file : str   — ASCII traveltime data (see format above)
    bounds    : (float, float)  — Vp [km/s] bounds
    n_seg     : int   — ray segments per ray for sensitivity matrix
    out       : bool
    """

    def __init__(self, vel_file: str, data_file: str, *,
                 bounds: tuple = (1.5, 8.5),
                 n_seg: int = 50,
                 out: bool = True):

        self.vel_file  = vel_file
        self.data_file = data_file
        self.bounds    = bounds
        self.n_seg     = n_seg
        self.out       = out

        self._load()

    # ── I/O ──────────────────────────────────────────────────────────────────

    def _load(self):
        """Load velocity model and ray data."""
        try:
            vel_data      = np.load(self.vel_file)   # (N, 4)
            self._coords  = vel_data[:, :3]
            self._m       = vel_data[:, 3].copy()    # Vp [km/s]
        except FileNotFoundError:
            if self.out:
                print("  SeisTomoForward: vel file not found — using stub.")
            n = 200
            self._coords = np.random.rand(n, 3) * 50_000.
            self._m      = np.full(n, 6.0)

        try:
            raw = np.loadtxt(self.data_file)
            self._src    = raw[:, 0:3]    # source xyz [m]
            self._rec    = raw[:, 3:6]    # receiver xyz [m]
            self._t_obs  = raw[:, 6]      # observed traveltimes [s]
            self._sigma  = raw[:, 7]      # standard deviations [s]
        except (FileNotFoundError, IndexError):
            if self.out:
                print("  SeisTomoForward: data file not found — using stub.")
            n_ray = 50
            self._src   = np.random.rand(n_ray, 3) * 50_000.
            self._rec   = np.random.rand(n_ray, 3) * 50_000.
            self._t_obs = np.ones(n_ray) * 10.0
            self._sigma = np.ones(n_ray) * 0.05

        # Pre-compute sensitivity matrix (ray-path lengths per voxel)
        self._A = self._build_sensitivity()

        if self.out:
            print(f"  SeisTomoForward: {len(self._m)} voxels, "
                  f"{len(self._t_obs)} rays loaded.")

    def _build_sensitivity(self) -> np.ndarray:
        """Build straight-ray sensitivity matrix A, shape (N_ray, N_vox).

        A[i, j] = path length of ray i through voxel j [m].
        Voxel assignment by nearest-neighbour to the ray sampling points.
        """
        from scipy.spatial import cKDTree
        tree    = cKDTree(self._coords)
        n_ray   = len(self._t_obs)
        n_vox   = len(self._m)
        A       = np.zeros((n_ray, n_vox), dtype=float)
        t_pts   = np.linspace(0., 1., self._n_seg + 1)

        for i in range(n_ray):
            pts  = (self._src[i] + t_pts[:, np.newaxis] *
                    (self._rec[i] - self._src[i]))         # (n_seg+1, 3)
            segs = pts[1:] - pts[:-1]                      # (n_seg, 3)
            dl   = np.linalg.norm(segs, axis=1)            # segment lengths
            mids = 0.5 * (pts[1:] + pts[:-1])              # midpoints
            _, nn = tree.query(mids, k=1)
            np.add.at(A[i], nn, dl)

        return A   # slowness Jacobian: t = A @ (1/Vp)

    # ── ModelGrid ─────────────────────────────────────────────────────────────

    def model_grid(self) -> ModelGrid:
        return ModelGrid(self._coords, self._m.copy(), name="seismic")

    # ── Misfit ───────────────────────────────────────────────────────────────

    def _t_pred(self, m: np.ndarray) -> np.ndarray:
        """Predicted traveltimes [s] for velocity model m [km/s]."""
        slowness = 1.0 / (m * 1e3)   # s/m
        return self._A @ slowness

    def misfit(self, m: np.ndarray) -> float:
        """Weighted L2 traveltime misfit  Φ_seis(m)."""
        self._m = np.clip(m, *self.bounds)
        res     = (self._t_pred(self._m) - self._t_obs) / self._sigma
        return float(0.5 * np.dot(res, res))

    # ── Gradient ─────────────────────────────────────────────────────────────

    def gradient(self, m: np.ndarray) -> np.ndarray:
        """Gradient of Φ_seis w.r.t. Vp [km/s]."""
        self._m  = np.clip(m, *self.bounds)
        res      = (self._t_pred(self._m) - self._t_obs) / self._sigma ** 2
        # ∂t/∂Vp = -A / (Vp^2 * 1e3)
        dslow_dv = -1.0 / (self._m ** 2 * 1e3)   # (N_vox,)
        # Chain: ∂Φ/∂Vp_j = Σ_i res_i * A_ij * dslow_dv_j
        return (self._A.T @ res) * dslow_dv

    # ── Model update ─────────────────────────────────────────────────────────

    def update(self, dm: np.ndarray):
        self._m = np.clip(self._m + dm, *self.bounds)

    def save_model(self, path: str):
        out = np.column_stack([self._coords, self._m])
        np.save(path, out)
        if self.out:
            print(f"  SeisTomoForward: model saved → {path}")
