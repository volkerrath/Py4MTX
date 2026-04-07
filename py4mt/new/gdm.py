#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gdm.py — Gradual Deformation Method (GDM) for mesh-based geophysical models.

The GDM generates a continuous, norm-preserving path through model space by
rotating between independent prior realisations:

    m(t) = m1 * cos(t) + m2 * sin(t),   t ∈ [0, 2π)

If m1 and m2 are drawn from N(0, C) then m(t) ~ N(0, C) for all t, so the
prior statistics are preserved exactly along the entire path.

In the context of a FEMTIC-style workflow the "prior" is N(0, Q⁻¹) where
Q = R^T R + λI, and realisations are produced by the same low-rank or
full-rank samplers already in ensembles.py.  GDM then provides:

  1. A continuous parameterisation for gradient-based optimisation in
     perturbation space (history-matching style).
  2. A structured way to interpolate between ensemble members while
     staying on the prior manifold.
  3. A path sampler: sweep t and evaluate forward model along the path.

References
----------
Roggero & Hu (SPE 49004, 1998)
Hu, Blanc & Noetinger (Math. Geology, 2001)
Le Ravalec-Dupin & Noetinger (Math. Geology, 2002)
Hu & Le Ravalec-Dupin (Math. Geology, 2004)
Caers (Math. Geosciences, 2007)

Provenance
----------
2026-04-03  Claude  Created.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Array = np.ndarray
Sampler = Callable[[int], Array]   # sampler(n_samples) -> (n_free, n_samples)


# ===========================================================================
# Core GDM object
# ===========================================================================

class GDM:
    """Gradual Deformation Method path between two prior realisations.

    Parameters
    ----------
    m1, m2 : array_like, shape (n,)
        Two **independent** realisations drawn from the same prior N(0, C).
        They span the 2-D rotation plane; all points on the path have the
        same prior statistics.
    ref : array_like, shape (n,) or None
        Reference / background model added to the perturbation.
        If None, the path is a pure perturbation (zero-mean).

    Examples
    --------
    >>> path = GDM(m1, m2, ref=log10_rho_ref)
    >>> m_half = path(np.pi / 4)          # single t value
    >>> ms = path(np.linspace(0, np.pi, 9))  # sweep
    """

    def __init__(
        self,
        m1: Array,
        m2: Array,
        ref: Optional[Array] = None,
    ) -> None:
        m1 = np.asarray(m1, dtype=float)
        m2 = np.asarray(m2, dtype=float)
        if m1.shape != m2.shape:
            raise ValueError(
                f"m1 and m2 must have the same shape, "
                f"got {m1.shape} and {m2.shape}."
            )
        # Orthonormalize via Gram-Schmidt so that ‖m(t)‖ is constant.
        # ‖m(t)‖² = ‖m1‖²cos²t + ‖m2‖²sin²t + 2(m1·m2)costsint
        # is constant iff m1 ⊥ m2 AND ‖m1‖ = ‖m2‖.
        # We preserve the norm of m1 and project m2 onto the orthogonal
        # complement, then rescale to the same norm.
        r = float(np.linalg.norm(m1))
        if r == 0.0:
            raise ValueError("m1 has zero norm; cannot define a rotation plane.")
        e1 = m1 / r
        m2_orth = m2 - np.dot(m2, e1) * e1
        norm_orth = float(np.linalg.norm(m2_orth))
        if norm_orth < 1e-14 * np.linalg.norm(m2):
            raise ValueError(
                "m1 and m2 are (nearly) collinear; "
                "cannot define a 2-D rotation plane."
            )
        self.m1 = e1 * r            # same as original m1
        self.m2 = m2_orth / norm_orth * r   # orthogonal to m1, same norm
        self.ref = np.asarray(ref, dtype=float) if ref is not None else None

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def __call__(self, t: float | Array) -> Array:
        """Evaluate the GDM path at parameter value(s) t (radians).

        Parameters
        ----------
        t : float or array_like
            Rotation angle(s) in radians.  A scalar returns shape (n,);
            an array of shape (k,) returns shape (n, k).

        Returns
        -------
        m : ndarray
            Model perturbation(s).  If ``ref`` was supplied, the reference
            model is added so the result is an absolute model.
        """
        t = np.asarray(t, dtype=float)
        scalar = t.ndim == 0
        t = np.atleast_1d(t)

        # m(t) = m1 cos(t) + m2 sin(t)  — shape (n, k)
        pert = self.m1[:, None] * np.cos(t) + self.m2[:, None] * np.sin(t)

        if self.ref is not None:
            pert = pert + self.ref[:, None]

        return pert[:, 0] if scalar else pert

    def gradient(self, t: float | Array) -> Array:
        """Derivative dm/dt at t (useful for gradient-based optimisation).

        dm/dt = -m1 sin(t) + m2 cos(t)
        """
        t = np.asarray(t, dtype=float)
        scalar = t.ndim == 0
        t = np.atleast_1d(t)
        grad = -self.m1[:, None] * np.sin(t) + self.m2[:, None] * np.cos(t)
        return grad[:, 0] if scalar else grad

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def norm(self, t: float | Array) -> float | Array:
        """L2 norm of the perturbation at t (should be constant along path)."""
        t = np.asarray(t, dtype=float)
        scalar = t.ndim == 0
        t = np.atleast_1d(t)
        pert = self.m1[:, None] * np.cos(t) + self.m2[:, None] * np.sin(t)
        n = np.linalg.norm(pert, axis=0)
        return float(n[0]) if scalar else n

    def sweep(
        self,
        n_steps: int = 37,
        t_range: Tuple[float, float] = (0.0, 2 * np.pi),
    ) -> Tuple[Array, Array]:
        """Evaluate the path at ``n_steps`` equally spaced t values.

        Parameters
        ----------
        n_steps : int
            Number of sample points (default 37, i.e. every 10°).
        t_range : (float, float)
            Start and end of t sweep in radians.

        Returns
        -------
        t_vals : ndarray, shape (n_steps,)
        models : ndarray, shape (n_model_params, n_steps)
        """
        t_vals = np.linspace(t_range[0], t_range[1], n_steps)
        return t_vals, self(t_vals)

    def __repr__(self) -> str:
        has_ref = self.ref is not None
        return (
            f"GDM(n={self.m1.size}, "
            f"‖m1‖={np.linalg.norm(self.m1):.3g}, "
            f"‖m2‖={np.linalg.norm(self.m2):.3g}, "
            f"ref={'yes' if has_ref else 'no'})"
        )


# ===========================================================================
# Multi-seed GDM path (chain of rotations)
# ===========================================================================

class GDMChain:
    """Chain of GDM segments for multi-step exploration.

    Each segment rotates between successive pairs of realisations:

        segment k: m(t) = m_k * cos(t) + m_{k+1} * sin(t)

    This allows the optimisation to escape local minima by refreshing one
    end of the rotation plane at each step (Hu & Le Ravalec-Dupin, 2004).

    Parameters
    ----------
    realisations : sequence of arrays, length ≥ 2
        Independent prior draws.  The chain has ``len(realisations) - 1``
        segments.
    ref : array_like or None
        Reference model added to every perturbation.
    """

    def __init__(
        self,
        realisations: Sequence[Array],
        ref: Optional[Array] = None,
    ) -> None:
        if len(realisations) < 2:
            raise ValueError("Need at least 2 realisations for a GDMChain.")
        self.realisations = [np.asarray(r, dtype=float) for r in realisations]
        self.ref = np.asarray(ref, dtype=float) if ref is not None else None
        self._segments = [
            GDM(self.realisations[i], self.realisations[i + 1], ref=ref)
            for i in range(len(self.realisations) - 1)
        ]

    def segment(self, k: int) -> GDM:
        """Return the k-th GDM segment (0-indexed)."""
        return self._segments[k]

    def __len__(self) -> int:
        return len(self._segments)

    def __repr__(self) -> str:
        return (
            f"GDMChain(n_segments={len(self)}, "
            f"n={self.realisations[0].size})"
        )


# ===========================================================================
# Factory: build a GDM from a prior sampler
# ===========================================================================

def gdm_from_sampler(
    sampler: Sampler,
    ref: Optional[Array] = None,
    rng: Optional[np.random.Generator] = None,
) -> GDM:
    """Draw two independent realisations and return a GDM path.

    Parameters
    ----------
    sampler : callable
        A function ``sampler(n) -> ndarray of shape (n_free, n)`` that draws
        ``n`` independent samples from the prior.  Compatible with the
        ``sample_rtr_low_rank`` / ``sample_rtr_full_rank`` API in
        ``ensembles.py``.
    ref : array_like or None
        Reference model.
    rng : numpy Generator or None
        Passed through to ``sampler`` if it accepts it; otherwise ignored.

    Returns
    -------
    GDM
    """
    draws = sampler(2)          # shape (n_free, 2)
    return GDM(draws[:, 0], draws[:, 1], ref=ref)


def gdm_chain_from_sampler(
    sampler: Sampler,
    n_segments: int = 3,
    ref: Optional[Array] = None,
) -> GDMChain:
    """Draw ``n_segments + 1`` realisations and return a GDMChain.

    Parameters
    ----------
    sampler : callable
        Same contract as in :func:`gdm_from_sampler`.
    n_segments : int
        Number of rotation planes in the chain.
    ref : array_like or None
        Reference model.

    Returns
    -------
    GDMChain
    """
    draws = sampler(n_segments + 1)   # shape (n_free, n_segments+1)
    reals = [draws[:, k] for k in range(n_segments + 1)]
    return GDMChain(reals, ref=ref)


# ===========================================================================
# Optimisation helper: 1-D line search over t
# ===========================================================================

def gdm_line_search(
    path: GDM,
    objective: Callable[[Array], float],
    t_init: float = 0.0,
    n_bracket: int = 12,
    refine: bool = True,
) -> Tuple[float, float, Array]:
    """Find the t ∈ [0, 2π) that minimises ``objective(m(t))``.

    Uses a coarse bracket scan followed by an optional golden-section
    refinement within the best interval.

    Parameters
    ----------
    path : GDM
        The rotation path to search along.
    objective : callable
        ``objective(m) -> float``.  Lower is better (data misfit + regulariser).
    t_init : float
        Starting angle in radians (default 0, i.e. start = m1).
    n_bracket : int
        Number of equally spaced candidate t values for the coarse scan.
    refine : bool
        If True, refine with golden-section search in the best bracket.

    Returns
    -------
    t_opt : float
        Optimal rotation angle.
    obj_opt : float
        Objective value at t_opt.
    m_opt : ndarray
        Model at t_opt.
    """
    t_cands = np.linspace(0.0, 2 * np.pi, n_bracket, endpoint=False)
    # Shift so that t_init is first
    t_cands = (t_cands + t_init) % (2 * np.pi)

    obj_vals = np.array([objective(path(t)) for t in t_cands])
    best_idx = int(np.argmin(obj_vals))
    t_opt = t_cands[best_idx]
    obj_opt = obj_vals[best_idx]

    if refine:
        # Golden-section within [t_prev, t_next] bracket
        t_lo = t_cands[(best_idx - 1) % n_bracket]
        t_hi = t_cands[(best_idx + 1) % n_bracket]
        # Unwrap if bracket crosses 2π
        if t_lo > t_hi:
            t_hi += 2 * np.pi
        t_opt, obj_opt = _golden_section(objective, path, t_lo, t_hi)
        t_opt = t_opt % (2 * np.pi)

    return t_opt, obj_opt, path(t_opt)


def _golden_section(
    objective: Callable[[Array], float],
    path: GDM,
    a: float,
    b: float,
    tol: float = 1e-4,
    max_iter: int = 60,
) -> Tuple[float, float]:
    """Golden-section minimisation of objective(path(t)) on [a, b]."""
    gr = (np.sqrt(5) - 1) / 2       # golden ratio conjugate ≈ 0.618
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = objective(path(c))
    fd = objective(path(d))
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = objective(path(c))
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = objective(path(d))
    t_opt = (a + b) / 2
    return t_opt, objective(path(t_opt))


# ===========================================================================
# Iterative GDM calibration (Algorithm 1 of Hu & Le Ravalec-Dupin 2004)
# ===========================================================================

def gdm_iterate(
    sampler: Sampler,
    objective: Callable[[Array], float],
    ref: Array,
    n_iter: int = 10,
    n_bracket: int = 18,
    tol: float = 1e-3,
    verbose: bool = True,
) -> Tuple[Array, list[float]]:
    """Iterative GDM calibration: minimise ``objective`` over prior draws.

    At each iteration a new prior draw m2 is generated.  The rotation
    between the current best model (reprojected onto a unit sphere as m1)
    and m2 is searched for the minimum of ``objective``.  The minimiser
    becomes m1 for the next iteration (Hu & Le Ravalec-Dupin, 2004).

    Parameters
    ----------
    sampler : callable
        ``sampler(1) -> ndarray of shape (n_free, 1)``
    objective : callable
        ``objective(m) -> float``.
    ref : array_like
        Reference / background model added to every perturbation.
    n_iter : int
        Maximum number of GDM iterations.
    n_bracket : int
        Coarse scan resolution per iteration.
    tol : float
        Stop when the relative improvement in objective falls below this.
    verbose : bool
        Print iteration log.

    Returns
    -------
    m_best : ndarray
        Best absolute model found.
    history : list of float
        Objective value after each iteration.
    """
    ref = np.asarray(ref, dtype=float)

    # Initialise: m1 = zero perturbation (i.e. start at reference model)
    n = ref.size
    m1 = np.zeros(n)
    obj_best = objective(ref)
    history = [obj_best]

    if verbose:
        print(f"GDM iter  0: obj = {obj_best:.6g}  (reference model)")

    for k in range(1, n_iter + 1):
        m2 = sampler(1)[:, 0]
        # On the first iteration m1 may be zero (starting from the reference).
        # In that case bootstrap with a second fresh draw as m1.
        if np.linalg.norm(m1) == 0.0:
            m1 = sampler(1)[:, 0]
        path = GDM(m1, m2, ref=ref)
        t_opt, obj_new, m_opt = gdm_line_search(
            path, objective, n_bracket=n_bracket
        )

        improvement = (obj_best - obj_new) / (abs(obj_best) + 1e-30)
        history.append(obj_new)

        if verbose:
            print(
                f"GDM iter {k:2d}: obj = {obj_new:.6g}  "
                f"Δ = {improvement:+.3g}  t* = {np.degrees(t_opt):.1f}°"
            )

        if obj_new < obj_best:
            obj_best = obj_new
            # New m1 = perturbation component of m_opt (ref already inside path)
            m1 = m_opt - ref

        if improvement < tol and k > 1:
            if verbose:
                print(f"  → converged (improvement < {tol})")
            break

    m_best = ref + m1
    return m_best, history


# ===========================================================================
# Integration shim for ensembles.py samplers
# ===========================================================================

def make_sampler_from_rsvd(
    R,
    n_eig: int = 128,
    n_oversampling: int = 10,
    n_power_iter: int = 3,
    sigma2_residual: float = 1e-3,
    rng: Optional[np.random.Generator] = None,
) -> Sampler:
    """Return a sampler compatible with ``gdm_from_sampler`` using the
    randomized-SVD low-rank branch from ``ensembles.sample_rtr_low_rank``.

    Parameters
    ----------
    R : scipy sparse matrix
        The FEMTIC roughness matrix (passed directly, Q not formed).
    n_eig, n_oversampling, n_power_iter, sigma2_residual
        Same as in ``ensembles.generate_model_ensemble``.
    rng : numpy Generator or None

    Returns
    -------
    sampler : callable
        ``sampler(n) -> ndarray, shape (n_free, n)``
    """
    try:
        from ensembles import sample_rtr_low_rank  # type: ignore
    except ImportError as e:
        raise ImportError(
            "ensembles.py must be on sys.path to use make_sampler_from_rsvd."
        ) from e

    if rng is None:
        rng = np.random.default_rng()

    def _sampler(n: int) -> Array:
        return sample_rtr_low_rank(
            R,
            n_samples=n,
            n_eig=n_eig,
            n_oversampling=n_oversampling,
            n_power_iter=n_power_iter,
            sigma2_residual=sigma2_residual,
            rng=rng,
        )

    return _sampler


def make_sampler_from_rng_gaussian(
    cov_sqrt: Array,
    rng: Optional[np.random.Generator] = None,
) -> Sampler:
    """Toy sampler using a pre-computed square root of the prior covariance.

    m = cov_sqrt @ z,  z ~ N(0, I)

    Useful for unit tests and synthetic examples without a full FEMTIC setup.

    Parameters
    ----------
    cov_sqrt : ndarray, shape (n, k)
        Factor such that ``cov_sqrt @ cov_sqrt.T ≈ C``.
    rng : numpy Generator or None

    Returns
    -------
    sampler : callable
    """
    if rng is None:
        rng = np.random.default_rng()
    cov_sqrt = np.asarray(cov_sqrt, dtype=float)

    def _sampler(n: int) -> Array:
        z = rng.standard_normal((cov_sqrt.shape[1], n))
        return cov_sqrt @ z   # shape (n_free, n)

    return _sampler


# ===========================================================================
# Quick self-test
# ===========================================================================

def _self_test() -> None:
    """Verify norm-preservation and basic API."""
    rng = np.random.default_rng(42)
    n = 200

    # Toy covariance: exponential decay
    x = np.linspace(0, 1, n)
    C = np.exp(-np.abs(x[:, None] - x[None, :]) / 0.1)
    L = np.linalg.cholesky(C + 1e-8 * np.eye(n))
    sampler = make_sampler_from_rng_gaussian(L, rng=rng)

    m1, m2 = sampler(2).T
    ref = rng.standard_normal(n) * 0.5

    path = GDM(m1, m2, ref=ref)
    print(path)

    # Norm should be constant along path
    t_vals = np.linspace(0, 2 * np.pi, 361)
    norms = path.norm(t_vals)          # perturbation norms
    print(f"Norm variation: min={norms.min():.6f}  max={norms.max():.6f}  "
          f"(should be constant ≈ {norms.mean():.4f})")
    assert np.allclose(norms, norms[0], rtol=1e-10), "Norm not constant!"

    # Gradient check
    dt = 1e-5
    t0 = 1.2
    grad_fd = (path(t0 + dt) - path(t0 - dt)) / (2 * dt)
    grad_an = path.gradient(t0)
    err = np.linalg.norm(grad_fd - grad_an) / np.linalg.norm(grad_an)
    print(f"Gradient relative error: {err:.2e}  (should be ~1e-10)")
    assert err < 1e-8, "Gradient check failed!"

    # Factory helpers
    path2 = gdm_from_sampler(sampler, ref=ref)
    chain = gdm_chain_from_sampler(sampler, n_segments=3, ref=ref)
    print(chain)

    # Line search with a toy quadratic objective
    t_true = 1.1
    m_target = path(t_true)

    def obj(m: Array) -> float:
        return float(np.sum((m - m_target) ** 2))

    t_opt, obj_opt, m_opt = gdm_line_search(path, obj, n_bracket=36)
    print(f"Line search: t_opt={np.degrees(t_opt):.2f}°  "
          f"(true={np.degrees(t_true):.2f}°)  obj={obj_opt:.2e}")
    assert obj_opt < 1e-6, "Line search did not find minimum!"

    # Iterative calibration (toy: match m_target)
    m_best, history = gdm_iterate(
        sampler, obj, ref=ref, n_iter=8, verbose=True
    )
    print(f"Final residual: {history[-1]:.3e}")

    print("\nAll tests passed.")


if __name__ == "__main__":
    _self_test()
