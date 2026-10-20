"""
poisson_seismic.py

Poisson's ratio from seismic velocities (isotropic, elastic medium).

Provenance
----------
Author         : Claude (Anthropic)
Date generated : 2026-10-17
Note           : This code is AI-generated. Review and test it before
                 any production use.

Relations
---------
    sigma = (Vp^2 - 2 Vs^2) / (2 (Vp^2 - Vs^2))
    sigma = (k^2 - 2) / (2 (k^2 - 1)),   k = Vp/Vs
    k     = sqrt((1 - sigma) / (0.5 - sigma))

Valid range: -1 < sigma <= 0.5 (positive bulk and shear moduli),
i.e. Vp/Vs > sqrt(4/3) ~ 1.155. sigma = 0.5 corresponds to Vs = 0 (fluid).
"""

import numpy as np

__all__ = ["poisson_from_vp_vs", "poisson_from_vpvs", "vpvs_from_poisson"]

_K_MIN = np.sqrt(4.0 / 3.0)  # Vp/Vs lower bound (K > 0)


def poisson_from_vp_vs(vp, vs, strict=True):
    """
    Poisson's ratio from P- and S-wave velocities.

    Parameters
    ----------
    vp, vs : float or array_like
        P and S velocities (same units, any). Broadcastable.
    strict : bool
        If True, raise ValueError for non-physical input
        (vp <= 0, vs < 0, or vp/vs <= sqrt(4/3)).
        If False, return NaN for those entries.

    Returns
    -------
    sigma : float or ndarray
    """
    vp = np.asarray(vp, dtype=float)
    vs = np.asarray(vs, dtype=float)
    bad = ~((vp > 0) & (vs >= 0) & (vp * vp > (4.0 / 3.0) * vs * vs))
    if strict and np.any(bad):
        raise ValueError("Non-physical Vp/Vs: need vp > 0, vs >= 0 and "
                         "vp/vs > sqrt(4/3).")
    vp2 = np.where(bad, np.nan, vp * vp)
    vs2 = np.where(bad, np.nan, vs * vs)
    sigma = (vp2 - 2.0 * vs2) / (2.0 * (vp2 - vs2))
    return float(sigma) if sigma.ndim == 0 else sigma


def poisson_from_vpvs(vpvs, strict=True):
    """
    Poisson's ratio from the velocity ratio k = Vp/Vs.

    Parameters
    ----------
    vpvs : float or array_like
        Vp/Vs ratio (np.inf allowed, gives 0.5).
    strict : bool
        Raise ValueError if vpvs <= sqrt(4/3) (else NaN).

    Returns
    -------
    sigma : float or ndarray
    """
    k = np.asarray(vpvs, dtype=float)
    bad = ~(k > _K_MIN)
    if strict and np.any(bad):
        raise ValueError("Non-physical Vp/Vs: need vp/vs > sqrt(4/3) ~ 1.155.")
    k2 = np.where(bad, np.nan, k * k)
    with np.errstate(divide="ignore", invalid="ignore"):
        # 0.5 - 1/(2(k^2-1)) is stable and gives 0.5 for k = inf
        sigma = 0.5 - 1.0 / (2.0 * (k2 - 1.0))
    return float(sigma) if sigma.ndim == 0 else sigma


def vpvs_from_poisson(sigma, strict=True):
    """
    Vp/Vs ratio from Poisson's ratio.

    Parameters
    ----------
    sigma : float or array_like
        Poisson's ratio, -1 < sigma <= 0.5 (0.5 gives np.inf).
    strict : bool
        Raise ValueError outside that range (else NaN).

    Returns
    -------
    vpvs : float or ndarray
    """
    s = np.asarray(sigma, dtype=float)
    bad = ~((s > -1.0) & (s <= 0.5))
    if strict and np.any(bad):
        raise ValueError("Poisson's ratio must satisfy -1 < sigma <= 0.5.")
    s = np.where(bad, np.nan, s)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.sqrt((1.0 - s) / (0.5 - s))
    return float(k) if k.ndim == 0 else k


if __name__ == "__main__":
    # Self-test
    assert abs(poisson_from_vpvs(np.sqrt(3.0)) - 0.25) < 1e-12
    assert abs(poisson_from_vp_vs(6.0, 6.0 / np.sqrt(3.0)) - 0.25) < 1e-12
    assert poisson_from_vp_vs(1.5, 0.0) == 0.5
    assert poisson_from_vpvs(np.inf) == 0.5
    k = np.array([1.6, 1.73, 1.9, 2.5])
    assert np.allclose(vpvs_from_poisson(poisson_from_vpvs(k)), k)
    assert np.isnan(poisson_from_vp_vs(1.0, 1.0, strict=False))
    print("%-8s %-8s" % ("Vp/Vs", "sigma"))
    print("-" * 17)
    for kk in k:
        print("%-8.3f %-8.4f" % (kk, poisson_from_vpvs(kk)))
    print("All self-tests passed.")
