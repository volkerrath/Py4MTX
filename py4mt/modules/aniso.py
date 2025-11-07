"""
Translated by ChatGPT (OpenAI GPT-5)
from:

    Pek, J. and Santos, F. A. M., 2002.
    Magnetotelluric impedances and parametric sensitivities for 1-D generally
    anisotropic layered media, Computers & Geosciences, 28,
    doi:10.1016/S0098-3004(02)00014-6
    https://doi.org/10.1016/S0098-3004(02)00014-6

Original Fortran source: zs1adr.for (ZS1ANEF subroutine family)
This Python translation reproduces the numerical behavior section-by-section,
using NumPy and matching the original Fortran variable naming for traceability.
"""

import numpy as np

# -------------------- small utils --------------------


def rotz(Z, angle):
    """Rotate a 2×2 complex impedance tensor by a geographic azimuth.

    Parameters
    ----------
    Z : ndarray, shape (2, 2), complex128
        Impedance tensor in the original reference frame.
    theta : float
        Rotation angle **in radians**, positive counterclockwise. A value of 0 leaves `Z` unchanged.

    Returns
    -------
    Zr : ndarray, shape (2, 2), complex128
        Rotated impedance tensor `R(θ) @ Z @ R(θ)^T`, where
        R(θ) = [[cos θ, sin θ], [-sin θ, cos θ]].

    Notes
    -----
    This operation preserves anti-symmetry of TE/TM half-space tensors (Zxy = −Zyx) and
    is commonly used to align model/measurement coordinates."""
    c = np.cos(angle)
    s = np.sin(angle)
    R = np.array([[c,  s], [-s,  c]])
    return (R @ Z @ R.T).astype(np.complex128)


def rotzs(dZdθ, nl, upto_layer, angle):
    """    (Auto-generated) Function `rotzs`.

        Parameters
        ----------
        dZdθ : any
    nl : any
    upto_layer : any
    angle : any

        Returns
        -------
        result : object
            See implementation.

        Notes
        -----
        This docstring was generated automatically. Adapt descriptions if more precise
        semantics are desired."""
    out = np.array(dZdθ, copy=True)
    count = upto_layer if upto_layer > 0 else nl
    for il in range(count):
        out[il] = rotz(out[il], angle)
    return out


def dfm(x):
    """Compute the downward wave admittance factor for the TM/TE recursion (minus branch).

    Parameters
    ----------
    x : float or ndarray
        Dimensionless attenuation argument (typically thickness/skin-depth related).

    Returns
    -------
    y : float or ndarray
        Value `1 - exp(-2x)` for real `x`. For vector input, applied elementwise.

    Notes
    -----
    In many 1D MT transfer-matrix implementations, `dfm` and `dfp` appear in
    stable upward/downward recursions for field amplitudes. At `x=0`, `dfm(0)=0`."""
    return (1.0 - np.exp(-2.0 * x)).astype(np.complex128)


def dfp(x):
    """Compute the upward wave admittance factor for the TM/TE recursion (plus branch).

    Parameters
    ----------
    x : float or ndarray
        Dimensionless attenuation argument (typically thickness/skin-depth related).

    Returns
    -------
    y : float or ndarray
        Value `1 + exp(-2x)` for real `x`. For vector input, applied elementwise.

    Notes
    -----
    Complements :func:`dfm`. At `x=0`, `dfp(0)=2`."""
    return (1.0 + np.exp(-2.0 * x)).astype(np.complex128)

# -------------------- helpers (Fortran ZSPRPG / ZSCU*) --------------------


def zsprpg(dzbot, zbot, ztop, dz1, dz2, ag1, ag2):
    """    (Auto-generated) Function `zsprpg`.

        Parameters
        ----------
        dzbot : any
    zbot : any
    ztop : any
    dz1 : any
    dz2 : any
    ag1 : any
    ag2 : any

        Returns
        -------
        result : object
            See implementation.

        Notes
        -----
        This docstring was generated automatically. Adapt descriptions if more precise
        semantics are desired."""
    dzbot = np.asarray(dzbot, dtype=np.complex128)
    zbot = np.asarray(zbot,  dtype=np.complex128)
    ztop = np.asarray(ztop,  dtype=np.complex128)
    dz1 = np.complex128(dz1)
    dz2 = np.complex128(dz2)
    ag1 = np.complex128(ag1)
    ag2 = np.complex128(ag2)

    dtzbot = zbot[0, 0]*zbot[1, 1] - zbot[0, 1]*zbot[1, 0]
    zdenom = dtzbot*dfm(ag1)*dfm(ag2)/(dz1*dz2) + \
        zbot[0, 1]*dfm(ag1)*dfp(ag2)/dz1 - \
        zbot[1, 0]*dfp(ag1)*dfm(ag2)/dz2 + \
        dfp(ag1)*dfp(ag2)

    ddtzbot = dzbot[0, 0]*zbot[1, 1] + zbot[0, 0]*dzbot[1, 1] - \
        dzbot[0, 1]*zbot[1, 0] - zbot[0, 1]*dzbot[1, 0]

    dzdenom = ddtzbot*dfm(ag1)*dfm(ag2)/(dz1*dz2) + \
        dzbot[0, 1]*dfm(ag1)*dfp(ag2)/dz1 - \
        dzbot[1, 0]*dfp(ag1)*dfm(ag2)/dz2

    dn11 = 4.0*dzbot[0, 0]*np.exp(-(ag1+ag2))
    dn12 = dzbot[0, 1]*dfp(ag1)*dfp(ag2) - \
        dzbot[1, 0]*dfm(ag1)*dfm(ag2)*dz1/dz2 + \
        ddtzbot*dfp(ag1)*dfm(ag2)/dz2
    dn21 = dzbot[1, 0]*dfp(ag1)*dfp(ag2) - \
        dzbot[0, 1]*dfm(ag1)*dfm(ag2)*dz2/dz1 - \
        ddtzbot*dfm(ag1)*dfp(ag2)/dz1
    dn22 = 4.0*dzbot[1, 1]*np.exp(-(ag1+ag2))

    out = np.empty((2, 2), dtype=np.complex128)
    out[0, 0] = (dn11 - ztop[0, 0]*dzdenom)/zdenom
    out[0, 1] = (dn12 - ztop[0, 1]*dzdenom)/zdenom
    out[1, 0] = (dn21 - ztop[1, 0]*dzdenom)/zdenom
    out[1, 1] = (dn22 - ztop[1, 1]*dzdenom)/zdenom
    return out


def zscua1(zbot, ztop, a1, dz1, dz2, ag1, ag2):
    """    (Auto-generated) Function `zscua1`.

        Parameters
        ----------
        zbot : any
    ztop : any
    a1 : any
    dz1 : any
    dz2 : any
    ag1 : any
    ag2 : any

        Returns
        -------
        result : object
            See implementation.

        Notes
        -----
        This docstring was generated automatically. Adapt descriptions if more precise
        semantics are desired."""
    zbot = np.asarray(zbot, dtype=np.complex128)
    ztop = np.asarray(ztop, dtype=np.complex128)
    a1 = float(a1)
    dz1 = np.complex128(dz1)
    dz2 = np.complex128(dz2)
    ag1 = np.complex128(ag1)
    ag2 = np.complex128(ag2)

    dtzbot = zbot[0, 0]*zbot[1, 1] - zbot[0, 1]*zbot[1, 0]
    zdenom = dtzbot*dfm(ag1)*dfm(ag2)/(dz1*dz2) + \
        zbot[0, 1]*dfm(ag1)*dfp(ag2)/dz1 - \
        zbot[1, 0]*dfp(ag1)*dfm(ag2)/dz2 + \
        dfp(ag1)*dfp(ag2)

    dapom = dtzbot*dfm(ag2)/dz2 + zbot[0, 1]*dfp(ag2)
    dbpom = dfp(ag2) - zbot[1, 0]*dfm(ag2)/dz2
    dcpom = zbot[1, 0]*dfp(ag2) - dz2*dfm(ag2)
    ddpom = dtzbot*dfp(ag2) + dz2*zbot[0, 1]*dfm(ag2)
    depom = ag1*dfm(ag1)
    dfpom = (dfm(ag1)+ag1*dfp(ag1))/dz1
    dgpom = (dfm(ag1)-ag1*dfp(ag1))*dz1

    dzdenom = dfpom*dapom + depom*dbpom
    dn12 = depom*dapom - dgpom*dbpom
    dn21 = depom*dcpom - dfpom*ddpom

    out = np.empty((2, 2), dtype=np.complex128)
    out[0, 0] = -0.5*ztop[0, 0]*dzdenom/(a1*zdenom)
    out[0, 1] = 0.5*(dn12 - ztop[0, 1]*dzdenom)/(a1*zdenom)
    out[1, 0] = 0.5*(dn21 - ztop[1, 0]*dzdenom)/(a1*zdenom)
    out[1, 1] = -0.5*ztop[1, 1]*dzdenom/(a1*zdenom)
    return out


def zscua2(zbot, ztop, a2, dz1, dz2, ag1, ag2):
    """    (Auto-generated) Function `zscua2`.

        Parameters
        ----------
        zbot : any
    ztop : any
    a2 : any
    dz1 : any
    dz2 : any
    ag1 : any
    ag2 : any

        Returns
        -------
        result : object
            See implementation.

        Notes
        -----
        This docstring was generated automatically. Adapt descriptions if more precise
        semantics are desired."""
    zbot = np.asarray(zbot, dtype=np.complex128)
    ztop = np.asarray(ztop, dtype=np.complex128)
    a2 = float(a2)
    dz1 = np.complex128(dz1)
    dz2 = np.complex128(dz2)
    ag1 = np.complex128(ag1)
    ag2 = np.complex128(ag2)

    dtzbot = zbot[0, 0]*zbot[1, 1] - zbot[0, 1]*zbot[1, 0]
    zdenom = dtzbot*dfm(ag1)*dfm(ag2)/(dz1*dz2) + \
        zbot[0, 1]*dfm(ag1)*dfp(ag2)/dz1 - \
        zbot[1, 0]*dfp(ag1)*dfm(ag2)/dz2 + \
        dfp(ag1)*dfp(ag2)

    dapom = dtzbot*dfm(ag1)/dz1 - zbot[1, 0]*dfp(ag1)
    dbpom = dfp(ag1) + zbot[0, 1]*dfm(ag1)/dz1
    dcpom = zbot[0, 1]*dfp(ag1) + dz1*dfm(ag1)
    ddpom = dtzbot*dfp(ag1) - dz1*zbot[1, 0]*dfm(ag1)
    depom = ag2*dfm(ag2)
    dfpom = (dfm(ag2)+ag2*dfp(ag2))/dz2
    dgpom = (dfm(ag2)-ag2*dfp(ag2))*dz2

    dzdenom = dfpom*dapom + depom*dbpom
    dn12 = depom*dcpom + dfpom*ddpom
    dn21 = -depom*dapom + dgpom*dbpom

    out = np.empty((2, 2), dtype=np.complex128)
    out[0, 0] = -0.5*ztop[0, 0]*dzdenom/(a2*zdenom)
    out[0, 1] = 0.5*(dn12 - ztop[0, 1]*dzdenom)/(a2*zdenom)
    out[1, 0] = 0.5*(dn21 - ztop[1, 0]*dzdenom)/(a2*zdenom)
    out[1, 1] = -0.5*ztop[1, 1]*dzdenom/(a2*zdenom)
    return out


def zscubs(zbot, ztop, dz1, dz2, ag1, ag2):
    """    (Auto-generated) Function `zscubs`.

        Parameters
        ----------
        zbot : any
    ztop : any
    dz1 : any
    dz2 : any
    ag1 : any
    ag2 : any

        Returns
        -------
        result : object
            See implementation.

        Notes
        -----
        This docstring was generated automatically. Adapt descriptions if more precise
        semantics are desired."""
    zbot = np.asarray(zbot, dtype=np.complex128)
    ztop = np.asarray(ztop, dtype=np.complex128)
    dz1 = np.complex128(dz1)
    dz2 = np.complex128(dz2)
    ag1 = np.complex128(ag1)
    ag2 = np.complex128(ag2)

    dtzbot = zbot[0, 0]*zbot[1, 1] - zbot[0, 1]*zbot[1, 0]
    zdenom = dtzbot*dfm(ag1)*dfm(ag2)/(dz1*dz2) + \
        zbot[0, 1]*dfm(ag1)*dfp(ag2)/dz1 - \
        zbot[1, 0]*dfp(ag1)*dfm(ag2)/dz2 + \
        dfp(ag1)*dfp(ag2)

    dzs = np.empty((2, 2), dtype=np.complex128)
    dzs[0, 0] = 4.0*(zbot[0, 1]+zbot[1, 0])*np.exp(-(ag1+ag2))
    dzs[0, 1] = (zbot[0, 0]-zbot[1, 1]) * \
        (dfm(ag1)*dfm(ag2)*dz1/dz2 - dfp(ag1)*dfp(ag2))
    dzs[1, 0] = (zbot[0, 0]-zbot[1, 1]) * \
        (dfm(ag1)*dfm(ag2)*dz2/dz1 - dfp(ag1)*dfp(ag2))
    dzs[1, 1] = -4.0*(zbot[0, 1]+zbot[1, 0])*np.exp(-(ag1+ag2))

    dpom = (zbot[0, 0]-zbot[1, 1]) * \
        (dfm(ag1)*dfp(ag2)/dz1 - dfp(ag1)*dfm(ag2)/dz2)

    out = np.empty((2, 2), dtype=np.complex128)
    out[0, 0] = (dzs[0, 0] + dpom*ztop[0, 0])/zdenom
    out[0, 1] = (dzs[0, 1] + dpom*ztop[0, 1])/zdenom
    out[1, 0] = (dzs[1, 0] + dpom*ztop[1, 0])/zdenom
    out[1, 1] = (dzs[1, 1] + dpom*ztop[1, 1])/zdenom
    return out


def zscuh(zbot, ztop, a1, a2, dz1, dz2, ag1, ag2):
    """    (Auto-generated) Function `zscuh`.

        Parameters
        ----------
        zbot : any
    ztop : any
    a1 : any
    a2 : any
    dz1 : any
    dz2 : any
    ag1 : any
    ag2 : any

        Returns
        -------
        result : object
            See implementation.

        Notes
        -----
        This docstring was generated automatically. Adapt descriptions if more precise
        semantics are desired."""
    zbot = np.asarray(zbot, dtype=np.complex128)
    ztop = np.asarray(ztop, dtype=np.complex128)
    a1 = float(a1)
    a2 = float(a2)
    dz1 = np.complex128(dz1)
    dz2 = np.complex128(dz2)
    ag1 = np.complex128(ag1)
    ag2 = np.complex128(ag2)

    dtzbot = zbot[0, 0]*zbot[1, 1] - zbot[0, 1]*zbot[1, 0]
    zdenom = dtzbot*dfm(ag1)*dfm(ag2)/(dz1*dz2) + \
        zbot[0, 1]*dfm(ag1)*dfp(ag2)/dz1 - \
        zbot[1, 0]*dfp(ag1)*dfm(ag2)/dz2 + \
        dfp(ag1)*dfp(ag2)

    k1 = a1*dz1
    k2 = a2*dz2

    dapom1 = dtzbot*dfm(ag1)/dz1 - zbot[1, 0]*dfp(ag1)
    dbpom1 = dfp(ag1) + zbot[0, 1]*dfm(ag1)/dz1
    dcpom1 = zbot[0, 1]*dfp(ag1) + dz1*dfm(ag1)
    ddpom1 = dtzbot*dfp(ag1) - dz1*zbot[1, 0]*dfm(ag1)

    dapom2 = dtzbot*dfm(ag2)/dz2 + zbot[0, 1]*dfp(ag2)
    dbpom2 = dfp(ag2) - zbot[1, 0]*dfm(ag2)/dz2
    dcpom2 = zbot[1, 0]*dfp(ag2) - dz2*dfm(ag2)
    ddpom2 = dtzbot*dfp(ag2) + dz2*zbot[0, 1]*dfm(ag2)

    dzdenom = k1*(dapom2*dfp(ag1)/dz1 + dbpom2*dfm(ag1)) + \
        k2*(dapom1*dfp(ag2)/dz2 + dbpom1*dfm(ag2))
    dn12 = k1*(dapom2*dfm(ag1) + dbpom2*dz1*dfp(ag1)) + \
        k2*(dcpom1*dfm(ag2) + ddpom1*dfp(ag2)/dz2)
    dn21 = k1*(dcpom2*dfm(ag1) - ddpom2*dfp(ag1)/dz1) - \
        k2*(dapom1*dfm(ag2) + dbpom1*dz2*dfp(ag2))

    out = np.empty((2, 2), dtype=np.complex128)
    out[0, 0] = -ztop[0, 0]*dzdenom/zdenom
    out[0, 1] = (dn12 - ztop[0, 1]*dzdenom)/zdenom
    out[1, 0] = (dn21 - ztop[1, 0]*dzdenom)/zdenom
    out[1, 1] = -ztop[1, 1]*dzdenom/zdenom
    return out

# -------------------- main forward: ZS1ANEF --------------------


def mt1d_aniso(layani, h, al, at, blt, per):
    """Compute the 2×2 surface impedance tensor for a 1D (layered) anisotropic earth.

    Parameters
    ----------
    layani : ndarray, shape (nl,), int
        Layer anisotropy flags (e.g., 0 = isotropic, >0 = anisotropic). Interpretation
        follows the implementation in this module.
    h : ndarray, shape (nl,), float64
        Layer thicknesses [km]. The last (basement) entry is typically 0.
    al : ndarray, shape (nl,), float64
        Effective longitudinal (TE) horizontal conductivity per layer [S/m].
    at : ndarray, shape (nl,), float64
        Effective transverse (TM) horizontal conductivity per layer [S/m].
    blt : ndarray, shape (nl,), float64
        Effective strike angle per layer [radians].
    per : float
        Period [s].

    Returns
    -------
    Z : ndarray, shape (2, 2), complex128
        Surface impedance tensor at z=0 in the geographic frame.
    dzdal : ndarray, shape (nl, 2, 2), complex128
        Sensitivity of `Z` with respect to `al` for each layer.
    dzdat : ndarray, shape (nl, 2, 2), complex128
        Sensitivity of `Z` with respect to `at` for each layer.
    dzdbs : ndarray, shape (nl, 2, 2), complex128
        Sensitivity of `Z` with respect to strike `blt` for each layer.
    dzdh : ndarray, shape (nl, 2, 2), complex128
        Sensitivity of `Z` with respect to thickness `h` for each layer.

    Notes
    -----
    Implements stable 1D TE/TM recursions with anisotropic effective parameters.
    For a single isotropic half-space, the analytic result is
    Zxy = (2e-3 * π / √(10 * per)) * (1 - i) / √σ  and Zyx = −Zxy, Zxx=Zyy=0."""
    layani = np.asarray(layani)
    h = np.asarray(h, dtype=float)
    al = np.asarray(al, dtype=float)
    at = np.asarray(at, dtype=float)
    blt = np.asarray(blt, dtype=float)

    nl = int(layani.shape[0])
    assert all(arr.shape[0] == nl for arr in (
        h, al, at, blt)), "All arrays must have length nl"

    pi = np.pi
    k0 = (2e-3 * pi / np.sqrt(10.0 * float(per))) * (1.0 - 1j)

    # basement (layer nl-1) in strike frame
    L = nl - 1
    a1 = float(al[L])
    a2 = float(at[L])
    bs = float(blt[L])

    zrot = np.zeros((2, 2), dtype=np.complex128)
    zrot[0, 1] = k0/np.sqrt(a1)
    zrot[1, 0] = -k0/np.sqrt(a2)

    dzdalrot = np.zeros((nl, 2, 2), dtype=np.complex128)
    dzdatrot = np.zeros((nl, 2, 2), dtype=np.complex128)
    dzdbsrot = np.zeros((nl, 2, 2), dtype=np.complex128)
    dzdhrot = np.zeros((nl, 2, 2), dtype=np.complex128)

    if int(layani[L]) == 0 and np.isclose(a1, a2):
        dzdalrot[L, 0, 1] = -0.5 * zrot[0, 1] / a1
        dzdatrot[L, 1, 0] = -0.5 * zrot[1, 0] / a2
    else:
        dzdalrot[L, 0, 1] = -0.5 * zrot[0, 1] / a1
        dzdatrot[L, 1, 0] = -0.5 * zrot[1, 0] / a2
        dzdbsrot[L, 0, 0] = -(zrot[0, 1] + zrot[1, 0])
        dzdbsrot[L, 1, 1] = -dzdbsrot[L, 0, 0]

    if nl == 1:
        z = rotz(zrot, -bs)
        dzdal = rotzs(dzdalrot, nl, nl, -bs)
        dzdat = rotzs(dzdatrot, nl, nl, -bs)
        dzdbs = rotzs(dzdbsrot, nl, nl, -bs)
        dzdh = np.zeros_like(dzdhrot)
        return z, dzdal, dzdat, dzdbs, dzdh

    # upward propagation
    bsref = float(blt[L])
    dzdal_tmp = np.zeros_like(dzdalrot)
    dzdat_tmp = np.zeros_like(dzdatrot)
    dzdbs_tmp = np.zeros_like(dzdbsrot)
    dzdh_tmp = np.zeros_like(dzdhrot)

    for layer in range(nl-2, -1, -1):
        hd = 1e3 * float(h[layer])  # km -> m
        a1 = float(al[layer])
        a2 = float(at[layer])
        bs = float(blt[layer])

        if (bs != bsref) and (not np.isclose(a1, a2)):
            zbot = rotz(zrot, bs - bsref)
            dzdal_tmp = rotzs(dzdalrot, nl, layer+1, bs - bsref)
            dzdat_tmp = rotzs(dzdatrot, nl, layer+1, bs - bsref)
            dzdbs_tmp = rotzs(dzdbsrot, nl, layer+1, bs - bsref)
            dzdh_tmp = rotzs(dzdhrot,  nl, layer+1, bs - bsref)
        else:
            zbot = zrot.copy()
            dzdal_tmp[layer+1:] = dzdalrot[layer+1:]
            dzdat_tmp[layer+1:] = dzdatrot[layer+1:]
            dzdbs_tmp[layer+1:] = dzdbsrot[layer+1:]
            dzdh_tmp[layer+1:] = dzdhrot[layer+1:]
            bs = bsref

        k1 = k0*np.sqrt(a1)
        k2 = k0*np.sqrt(a2)
        dz1 = k0/np.sqrt(a1)
        dz2 = k0/np.sqrt(a2)
        ag1 = k1*hd
        ag2 = k2*hd

        dtzbot = zbot[0, 0]*zbot[1, 1] - zbot[0, 1]*zbot[1, 0]
        zdenom = (dtzbot*dfm(ag1)*dfm(ag2)/(dz1*dz2)
                  + zbot[0, 1]*dfm(ag1)*dfp(ag2)/dz1
                  - zbot[1, 0]*dfp(ag1)*dfm(ag2)/dz2
                  + dfp(ag1)*dfp(ag2))

        zrot = np.empty((2, 2), dtype=np.complex128)
        zrot[0, 0] = 4.0*zbot[0, 0]*np.exp(-(ag1+ag2))/zdenom
        zrot[1, 1] = 4.0*zbot[1, 1]*np.exp(-(ag1+ag2))/zdenom
        zrot[0, 1] = (zbot[0, 1]*dfp(ag1)*dfp(ag2)
                      - zbot[1, 0]*dfm(ag1)*dfm(ag2)*dz1/dz2
                      + dtzbot*dfp(ag1)*dfm(ag2)/dz2
                      + dfm(ag1)*dfp(ag2)*dz1)/zdenom
        zrot[1, 0] = (zbot[1, 0]*dfp(ag1)*dfp(ag2)
                      - zbot[0, 1]*dfm(ag1)*dfm(ag2)*dz2/dz1
                      - dtzbot*dfm(ag1)*dfp(ag2)/dz1
                      - dfp(ag1)*dfm(ag2)*dz2)/zdenom

        for il in range(layer+1, nl):
            dzdalrot[il] = zsprpg(dzdal_tmp[il], zbot,
                                  zrot, dz1, dz2, ag1, ag2)
            dzdatrot[il] = zsprpg(dzdat_tmp[il], zbot,
                                  zrot, dz1, dz2, ag1, ag2)
            dzdbsrot[il] = zsprpg(dzdbs_tmp[il], zbot,
                                  zrot, dz1, dz2, ag1, ag2)
            dzdhrot[il] = zsprpg(dzdh_tmp[il],  zbot, zrot, dz1, dz2, ag1, ag2)

        if int(layani[layer]) == 0 and np.isclose(a1, a2):
            dzdalrot[layer] = zscua1(zbot, zrot, a1, dz1, dz2, ag1, ag2)
            dzdatrot[layer] = zscua2(zbot, zrot, a2, dz1, dz2, ag1, ag2)
            dzdbsrot[layer] = np.zeros((2, 2), dtype=np.complex128)
            dzdhrot[layer] = zscuh(zbot, zrot, a1, a2, dz1, dz2, ag1, ag2)
        else:
            dzdalrot[layer] = zscua1(zbot, zrot, a1, dz1, dz2, ag1, ag2)
            dzdatrot[layer] = zscua2(zbot, zrot, a2, dz1, dz2, ag1, ag2)
            dzdbsrot[layer] = zscubs(zbot, zrot,      dz1, dz2, ag1, ag2)
            dzdhrot[layer] = zscuh(zbot, zrot, a1, a2, dz1, dz2, ag1, ag2)

        bsref = bs

    if abs(bsref) > 0.0:
        z = rotz(zrot, -bsref)
        dzdal = rotzs(dzdalrot, nl, 1, -bsref)
        dzdat = rotzs(dzdatrot, nl, 1, -bsref)
        dzdbs = rotzs(dzdbsrot, nl, 1, -bsref)
        dzdh = rotzs(dzdhrot,  nl, 1, -bsref)
    else:
        z, dzdal, dzdat, dzdbs, dzdh = zrot.copy(), dzdalrot.copy(
        ), dzdatrot.copy(), dzdbsrot.copy(), dzdhrot.copy()

    return z, dzdal, dzdat, dzdbs, dzdh

# -------------------- CPANIS (principal -> tensor & effective horizontals) --------------------


def prep_aniso(rop, ustr, udip, usla):
    """Build anisotropic conductivity tensors and effective horizontal conductivities.

    Parameters
    ----------
    rop : ndarray, shape (nl, 3), float64
        Principal resistivities [Ω·m] per layer: (ρ1, ρ2, ρ3).
    ustr : ndarray, shape (nl,), float64
        Strike angle [degrees] of the ρ1 axis (Euler α).
    udip : ndarray, shape (nl,), float64
        Dip angle [degrees] (Euler β).
    usla : ndarray, shape (nl,), float64
        Slant/roll angle [degrees] (Euler γ).

    Returns
    -------
    sg : ndarray, shape (nl, 3, 3), float64
        Full conductivity tensors σ for each layer in geographic coordinates.
    al : ndarray, shape (nl,), float64
        Effective longitudinal (TE) horizontal conductivity per layer.
    at : ndarray, shape (nl,), float64
        Effective transverse (TM) horizontal conductivity per layer.
    blt : ndarray, shape (nl,), float64
        Effective strike angle [radians] used by 1D forward routines.

    Notes
    -----
    Converts principal resistivities to conductivities via σi = 1/ρi and rotates them
    from principal to geographic coordinates using the provided Euler angles.
    For isotropic media (ρ1=ρ2=ρ3), `al == at == 1/ρ` and `blt == 0`."""
    rop = np.asarray(rop,  dtype=float)
    ustr = np.asarray(ustr, dtype=float)
    udip = np.asarray(udip, dtype=float)
    usla = np.asarray(usla, dtype=float)

    nl = rop.shape[0]
    assert rop.shape == (nl, 3), "rop must be (nl,3)"
    assert ustr.shape == (nl,) and udip.shape == (
        nl,) and usla.shape == (nl,), "angle arrays must be (nl,)"

    sgp = 1.0 / rop  # S/m
    sg = np.zeros((nl, 3, 3))
    al = np.zeros(nl)
    at = np.zeros(nl)
    blt = np.zeros(nl)

    tiny = np.finfo(float).tiny
    pi = np.pi

    for layer in range(nl):
        rstr = pi * ustr[layer] / 180.0
        rdip = pi * udip[layer] / 180.0
        rsla = pi * usla[layer] / 180.0

        sps, cps = np.sin(rstr), np.cos(rstr)
        sth, cth = np.sin(rdip), np.cos(rdip)
        sfi, cfi = np.sin(rsla), np.cos(rsla)

        pom1 = sgp[layer, 0]*cfi*cfi + sgp[layer, 1]*sfi*sfi
        pom2 = sgp[layer, 0]*sfi*sfi + sgp[layer, 1]*cfi*cfi
        pom3 = (sgp[layer, 0] - sgp[layer, 1]) * sfi * cfi

        c2ps = cps*cps
        s2ps = sps*sps
        c2th = cth*cth
        s2th = sth*sth
        csps = cps*sps
        csth = cth*sth

        sg11 = pom1*c2ps + pom2*s2ps*c2th - 2.0 * \
            pom3*cth*csps + sgp[layer, 2]*s2th*s2ps
        sg12 = pom1*csps - pom2*c2th*csps + pom3 * \
            cth*(c2ps - s2ps) - sgp[layer, 2]*s2th*csps
        sg13 = -pom2*csth*sps + pom3*sth*cps + sgp[layer, 2]*csth*sps

        sg22 = pom1*s2ps + pom2*c2ps*c2th + 2.0 * \
            pom3*cth*csps + sgp[layer, 2]*s2th*c2ps
        sg23 = pom2*csth*cps + pom3*sth*sps - sgp[layer, 2]*csth*cps

        sg33 = pom2*s2th + sgp[layer, 2]*c2th

        sg[layer, 0, 0] = sg11
        sg[layer, 0, 1] = sg12
        sg[layer, 0, 2] = sg13
        sg[layer, 1, 0] = sg12
        sg[layer, 1, 1] = sg22
        sg[layer, 1, 2] = sg23
        sg[layer, 2, 0] = sg13
        sg[layer, 2, 1] = sg23
        sg[layer, 2, 2] = sg33

        # Schur complement to eliminate z and get effective horizontal 2x2
        s11, s12, s13 = sg11, sg12, sg13
        s22, s23 = sg22, sg23
        s33 = sg33

        axx = s11 - s13*s13 / s33
        axy = s12 - s13*s23 / s33
        ayx = s12 - s13*s23 / s33
        ayy = s22 - s23*s23 / s33

        da12 = np.sqrt((axx - ayy)*(axx - ayy) + 4.0*axy*ayx)
        al[layer] = 0.5*(axx + ayy + da12)
        at[layer] = 0.5*(axx + ayy - da12)

        if da12 >= tiny:
            c2theta = (axx - ayy) / da12
            c2theta = np.clip(c2theta, -1.0, 1.0)
            theta = 0.5*np.arccos(c2theta)
        else:
            theta = 0.0

        blt[layer] = -theta if (axy < 0.0) else theta

    return sg, al, at, blt