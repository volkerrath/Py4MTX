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
using NumPy (float64 / complex128 precision) and matching the original Fortran
variable naming for traceability.
"""

from __future__ import annotations
import numpy as np

# ---------- Types & small utilities ----------
Float = np.float64
Cplx = np.complex128
pi = Float(np.pi)


def rotz(Z: np.ndarray, angle: Float) -> np.ndarray:
    """Rotate 2x2 impedance tensor by 'angle' (radians) in the xy-plane."""
    c = Float(np.cos(angle))
    s = Float(np.sin(angle))
    R = np.array([[c,  s], [-s,  c]], dtype=Float)
    return (R @ Z @ R.T).astype(Cplx)


def rotzs(dZdθ: np.ndarray, nl: int, upto_layer: int, angle: Float) -> np.ndarray:
    """Rotate sensitivities for layers 0..upto_layer-1 by 'angle'."""
    out = np.array(dZdθ, copy=True)
    count = upto_layer if upto_layer > 0 else nl
    for il in range(count):
        out[il] = rotz(out[il], angle)
    return out


def dfm(x):
    """1 - exp(-2x), complex-safe."""
    return (1.0 - np.exp(-2.0 * x)).astype(Cplx)


def dfp(x):
    """1 + exp(-2x), complex-safe."""
    return (1.0 + np.exp(-2.0 * x)).astype(Cplx)

# ---------- Helper routines (from ZSPRPG / ZSCU*) ----------


def zsprpg(dzbot, zbot, ztop, dz1, dz2, ag1, ag2):
    dzbot = np.asarray(dzbot, dtype=Cplx)
    zbot = np.asarray(zbot,  dtype=Cplx)
    ztop = np.asarray(ztop,  dtype=Cplx)
    dz1 = Cplx(dz1)
    dz2 = Cplx(dz2)
    ag1 = Cplx(ag1)
    ag2 = Cplx(ag2)

    dtzbot = zbot[0, 0]*zbot[1, 1] - zbot[0, 1]*zbot[1, 0]
    zdenom = dtzbot*dfm(ag1)*dfm(ag2)/(dz1*dz2) + zbot[0, 1]*dfm(ag1)*dfp(
        ag2)/dz1 - zbot[1, 0]*dfp(ag1)*dfm(ag2)/dz2 + dfp(ag1)*dfp(ag2)

    ddtzbot = dzbot[0, 0]*zbot[1, 1] + zbot[0, 0]*dzbot[1, 1] - \
        dzbot[0, 1]*zbot[1, 0] - zbot[0, 1]*dzbot[1, 0]

    dzdenom = ddtzbot*dfm(ag1)*dfm(ag2)/(dz1*dz2) + \
        dzbot[0, 1]*dfm(ag1)*dfp(ag2)/dz1 - dzbot[1, 0]*dfp(ag1)*dfm(ag2)/dz2

    dn11 = 4.0*dzbot[0, 0]*np.exp(-(ag1+ag2))
    dn12 = dzbot[0, 1]*dfp(ag1)*dfp(ag2) - dzbot[1, 0] * \
        dfm(ag1)*dfm(ag2)*dz1/dz2 + ddtzbot*dfp(ag1)*dfm(ag2)/dz2
    dn21 = dzbot[1, 0]*dfp(ag1)*dfp(ag2) - dzbot[0, 1] * \
        dfm(ag1)*dfm(ag2)*dz2/dz1 - ddtzbot*dfm(ag1)*dfp(ag2)/dz1
    dn22 = 4.0*dzbot[1, 1]*np.exp(-(ag1+ag2))

    out = np.empty((2, 2), dtype=Cplx)
    out[0, 0] = (dn11 - ztop[0, 0]*dzdenom)/zdenom
    out[0, 1] = (dn12 - ztop[0, 1]*dzdenom)/zdenom
    out[1, 0] = (dn21 - ztop[1, 0]*dzdenom)/zdenom
    out[1, 1] = (dn22 - ztop[1, 1]*dzdenom)/zdenom
    return out


def zscua1(zbot, ztop, a1, dz1, dz2, ag1, ag2):
    zbot = np.asarray(zbot, dtype=Cplx)
    ztop = np.asarray(ztop, dtype=Cplx)
    a1 = Float(a1)
    dz1 = Cplx(dz1)
    dz2 = Cplx(dz2)
    ag1 = Cplx(ag1)
    ag2 = Cplx(ag2)

    dtzbot = zbot[0, 0]*zbot[1, 1] - zbot[0, 1]*zbot[1, 0]
    zdenom = dtzbot*dfm(ag1)*dfm(ag2)/(dz1*dz2) + zbot[0, 1]*dfm(ag1)*dfp(
        ag2)/dz1 - zbot[1, 0]*dfp(ag1)*dfm(ag2)/dz2 + dfp(ag1)*dfp(ag2)

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

    out = np.empty((2, 2), dtype=Cplx)
    out[0, 0] = -0.5*ztop[0, 0]*dzdenom/(a1*zdenom)
    out[0, 1] = 0.5*(dn12 - ztop[0, 1]*dzdenom)/(a1*zdenom)
    out[1, 0] = 0.5*(dn21 - ztop[1, 0]*dzdenom)/(a1*zdenom)
    out[1, 1] = -0.5*ztop[1, 1]*dzdenom/(a1*zdenom)
    return out


def zscua2(zbot, ztop, a2, dz1, dz2, ag1, ag2):
    zbot = np.asarray(zbot, dtype=Cplx)
    ztop = np.asarray(ztop, dtype=Cplx)
    a2 = Float(a2)
    dz1 = Cplx(dz1)
    dz2 = Cplx(dz2)
    ag1 = Cplx(ag1)
    ag2 = Cplx(ag2)

    dtzbot = zbot[0, 0]*zbot[1, 1] - zbot[0, 1]*zbot[1, 0]
    zdenom = dtzbot*dfm(ag1)*dfm(ag2)/(dz1*dz2) + zbot[0, 1]*dfm(ag1)*dfp(
        ag2)/dz1 - zbot[1, 0]*dfp(ag1)*dfm(ag2)/dz2 + dfp(ag1)*dfp(ag2)

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

    out = np.empty((2, 2), dtype=Cplx)
    out[0, 0] = -0.5*ztop[0, 0]*dzdenom/(a2*zdenom)
    out[0, 1] = 0.5*(dn12 - ztop[0, 1]*dzdenom)/(a2*zdenom)
    out[1, 0] = 0.5*(dn21 - ztop[1, 0]*dzdenom)/(a2*zdenom)
    out[1, 1] = -0.5*ztop[1, 1]*dzdenom/(a2*zdenom)
    return out


def zscubs(zbot, ztop, dz1, dz2, ag1, ag2):
    zbot = np.asarray(zbot, dtype=Cplx)
    ztop = np.asarray(ztop, dtype=Cplx)
    dz1 = Cplx(dz1)
    dz2 = Cplx(dz2)
    ag1 = Cplx(ag1)
    ag2 = Cplx(ag2)

    dtzbot = zbot[0, 0]*zbot[1, 1] - zbot[0, 1]*zbot[1, 0]
    zdenom = dtzbot*dfm(ag1)*dfm(ag2)/(dz1*dz2) + zbot[0, 1]*dfm(ag1)*dfp(
        ag2)/dz1 - zbot[1, 0]*dfp(ag1)*dfm(ag2)/dz2 + dfp(ag1)*dfp(ag2)

    dzbot_s = np.empty((2, 2), dtype=Cplx)
    dzbot_s[0, 0] = 4.0*(zbot[0, 1]+zbot[1, 0])*np.exp(-(ag1+ag2))
    dzbot_s[0, 1] = (zbot[0, 0]-zbot[1, 1]) * \
        (dfm(ag1)*dfm(ag2)*dz1/dz2 - dfp(ag1)*dfp(ag2))
    dzbot_s[1, 0] = (zbot[0, 0]-zbot[1, 1]) * \
        (dfm(ag1)*dfm(ag2)*dz2/dz1 - dfp(ag1)*dfp(ag2))
    dzbot_s[1, 1] = -4.0*(zbot[0, 1]+zbot[1, 0])*np.exp(-(ag1+ag2))

    dpom = (zbot[0, 0]-zbot[1, 1]) * \
        (dfm(ag1)*dfp(ag2)/dz1 - dfp(ag1)*dfm(ag2)/dz2)

    out = np.empty((2, 2), dtype=Cplx)
    out[0, 0] = (dzbot_s[0, 0] + dpom*ztop[0, 0])/zdenom
    out[0, 1] = (dzbot_s[0, 1] + dpom*ztop[0, 1])/zdenom
    out[1, 0] = (dzbot_s[1, 0] + dpom*ztop[1, 0])/zdenom
    out[1, 1] = (dzbot_s[1, 1] + dpom*ztop[1, 1])/zdenom
    return out


def zscuh(zbot, ztop, a1, a2, dz1, dz2, ag1, ag2):
    zbot = np.asarray(zbot, dtype=Cplx)
    ztop = np.asarray(ztop, dtype=Cplx)
    a1 = Float(a1)
    a2 = Float(a2)
    dz1 = Cplx(dz1)
    dz2 = Cplx(dz2)
    ag1 = Cplx(ag1)
    ag2 = Cplx(ag2)

    dtzbot = zbot[0, 0]*zbot[1, 1] - zbot[0, 1]*zbot[1, 0]
    zdenom = dtzbot*dfm(ag1)*dfm(ag2)/(dz1*dz2) + zbot[0, 1]*dfm(ag1)*dfp(
        ag2)/dz1 - zbot[1, 0]*dfp(ag1)*dfm(ag2)/dz2 + dfp(ag1)*dfp(ag2)

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

    out = np.empty((2, 2), dtype=Cplx)
    out[0, 0] = -ztop[0, 0]*dzdenom/zdenom
    out[0, 1] = (dn12 - ztop[0, 1]*dzdenom)/zdenom
    out[1, 0] = (dn21 - ztop[1, 0]*dzdenom)/zdenom
    out[1, 1] = -ztop[1, 1]*dzdenom/zdenom
    return out

# ---------- Main routine: ZS1ANEF ----------


def zs1anef(layani: np.ndarray,
            h: np.ndarray,
            al: np.ndarray,
            at: np.ndarray,
            blt: np.ndarray,
            per: Float):
    """Compute Z and sensitivities at the surface for a 1-D generally anisotropic stack."""
    layani = np.asarray(layani)
    h = np.asarray(h, dtype=Float)
    al = np.asarray(al, dtype=Float)
    at = np.asarray(at, dtype=Float)
    blt = np.asarray(blt, dtype=Float)

    nl = int(layani.shape[0])
    assert all(arr.shape[0] == nl for arr in (
        h, al, at, blt)), "All arrays must have length nl"

    k0 = (Float(2e-3) * pi / np.sqrt(Float(10.0) * Float(per))) * (Float(1.0) - 1j)

    # ----- Section B: basement -----
    L = nl - 1
    a1 = Float(al[L])
    a2 = Float(at[L])
    bs = Float(blt[L])

    a1is = Float(1.0)/np.sqrt(a1)
    a2is = Float(1.0)/np.sqrt(a2)

    zrot = np.zeros((2, 2), dtype=Cplx)
    zrot[0, 1] = k0*a1is
    zrot[1, 0] = -k0*a2is

    dzdalrot = np.zeros((nl, 2, 2), dtype=Cplx)
    dzdatrot = np.zeros((nl, 2, 2), dtype=Cplx)
    dzdbsrot = np.zeros((nl, 2, 2), dtype=Cplx)
    dzdhrot = np.zeros((nl, 2, 2), dtype=Cplx)

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

    # ----- Section C: upward propagation -----
    bsref = Float(blt[L])
    dzdal_tmp = np.zeros_like(dzdalrot)
    dzdat_tmp = np.zeros_like(dzdatrot)
    dzdbs_tmp = np.zeros_like(dzdbsrot)
    dzdh_tmp = np.zeros_like(dzdhrot)

    for layer in range(nl-2, -1, -1):
        hd = Float(1e3) * Float(h[layer])  # km -> m
        a1 = Float(al[layer])
        a2 = Float(at[layer])
        bs = Float(blt[layer])

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
        a1is = Float(1.0)/np.sqrt(a1)
        a2is = Float(1.0)/np.sqrt(a2)
        dz1 = k0*a1is
        dz2 = k0*a2is
        ag1 = k1*hd
        ag2 = k2*hd

        dtzbot = zbot[0, 0]*zbot[1, 1] - zbot[0, 1]*zbot[1, 0]
        zdenom = (dtzbot*dfm(ag1)*dfm(ag2)/(dz1*dz2)
                  + zbot[0, 1]*dfm(ag1)*dfp(ag2)/dz1
                  - zbot[1, 0]*dfp(ag1)*dfm(ag2)/dz2
                  + dfp(ag1)*dfp(ag2))

        zrot = np.empty((2, 2), dtype=Cplx)
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
            dzdbsrot[layer] = np.zeros((2, 2), dtype=Cplx)
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

# ---------- CPANIS: build conductivity tensors & effective horizontal parameters ----------


def cpanis(rop: np.ndarray, ustr: np.ndarray, udip: np.ndarray, usla: np.ndarray):
    """Compute per-layer conductivity tensors and effective horizontal (AL, AT, BLT).
    Inputs angles are in degrees; BLT output is radians.
    rop are principal resistivities (Ohm·m); outputs use S/m.
    """
    rop = np.asarray(rop,  dtype=Float)
    ustr = np.asarray(ustr, dtype=Float)
    udip = np.asarray(udip, dtype=Float)
    usla = np.asarray(usla, dtype=Float)

    nl = rop.shape[0]
    assert rop.shape == (nl, 3), "rop must be (nl,3)"
    assert ustr.shape == (nl,) and udip.shape == (
        nl,) and usla.shape == (nl,), "angle arrays must be (nl,)"

    tiny = np.finfo(Float).tiny

    # Principal conductivities (S/m)
    sgp = 1.0 / rop  # (nl,3)

    sg = np.zeros((nl, 3, 3), dtype=Float)
    al = np.zeros(nl, dtype=Float)
    at = np.zeros(nl, dtype=Float)
    blt = np.zeros(nl, dtype=Float)

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

        # Fill symmetric tensor
        sg[layer, 0, 0] = sg11
        sg[layer, 0, 1] = sg12
        sg[layer, 0, 2] = sg13
        sg[layer, 1, 0] = sg12
        sg[layer, 1, 1] = sg22
        sg[layer, 1, 2] = sg23
        sg[layer, 2, 0] = sg13
        sg[layer, 2, 1] = sg23
        sg[layer, 2, 2] = sg33

        # Effective horizontal via Schur complement
        s11, s12, s13 = sg11, sg12, sg13
        s21, s22, s23 = sg12, sg22, sg23
        s33 = sg33

        axx = s11 - s13*s13 / s33
        axy = s12 - s13*s23 / s33
        ayx = s21 - s13*s23 / s33
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
