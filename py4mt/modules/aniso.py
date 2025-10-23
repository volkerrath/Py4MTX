# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 15:08:41 2025

@author: vrath
"""
import numpy as np
import math


def dphase(Z):
    """
    Computes the phase (angle) of a complex number z16 in radians,
    using atan2 for robust quadrant handling.
    """
    return math.atan2(Z.imag, Z.real)


def cpanis(rop, ustr, udip, usla):
    """
    Computes effective azimuthal anisotropy parameters from principal resistivities
    and Euler angles for a stack of anisotropic layers.

    Parameters:
    ----------
    rop : ndarray of shape (nl, 3)
        Principal resistivities [Ohm·m]
    ustr, udip, usla : ndarray of shape (nl,)
        Euler angles [degrees]: strike, dip, slant

    Returns:
    -------
    sg : ndarray of shape (nl, 3, 3)
        Conductivity tensors [S/m]
    al : ndarray of shape (nl,)
        Maximum effective horizontal conductivities [S/m]
    at : ndarray of shape (nl,)
        Minimum effective horizontal conductivities [S/m]
    blt : ndarray of shape (nl,)
        Effective horizontal anisotropy strike [radians]
    """
    pi = np.pi
    nl = rop.shape[0]
    sg = np.zeros((nl, 3, 3), dtype=np.float64)
    al = np.zeros(nl, dtype=np.float64)
    at = np.zeros(nl, dtype=np.float64)
    blt = np.zeros(nl, dtype=np.float64)

    sgp = 1.0 / rop.astype(np.float64)

    for layer in range(nl):
        rstr = pi * ustr[layer] / 180.0
        rdip = pi * udip[layer] / 180.0
        rsla = pi * usla[layer] / 180.0

        sps, cps = np.sin(rstr), np.cos(rstr)
        sth, cth = np.sin(rdip), np.cos(rdip)
        sfi, cfi = np.sin(rsla), np.cos(rsla)

        pom1 = sgp[layer, 0] * cfi**2 + sgp[layer, 1] * sfi**2
        pom2 = sgp[layer, 0] * sfi**2 + sgp[layer, 1] * cfi**2
        pom3 = (sgp[layer, 0] - sgp[layer, 1]) * sfi * cfi

        c2ps, s2ps = cps**2, sps**2
        c2th, s2th = cth**2, sth**2
        csps, csth = cps * sps, cth * sth

        # Conductivity tensor
        sg[layer, 0, 0] = pom1 * c2ps + pom2 * s2ps * c2th - \
            2.0 * pom3 * cth * csps + sgp[layer, 2] * s2th * s2ps
        sg[layer, 0, 1] = pom1 * csps - pom2 * c2th * csps + \
            pom3 * cth * (c2ps - s2ps) - sgp[layer, 2] * s2th * csps
        sg[layer, 0, 2] = -pom2 * csth * sps + pom3 * \
            sth * cps + sgp[layer, 2] * csth * sps
        sg[layer, 1, 0] = sg[layer, 0, 1]
        sg[layer, 1, 1] = pom1 * s2ps + pom2 * c2ps * c2th + \
            2.0 * pom3 * cth * csps + sgp[layer, 2] * s2th * c2ps
        sg[layer, 1, 2] = pom2 * csth * cps + pom3 * \
            sth * sps - sgp[layer, 2] * csth * cps
        sg[layer, 2, 0] = sg[layer, 0, 2]
        sg[layer, 2, 1] = sg[layer, 1, 2]
        sg[layer, 2, 2] = pom2 * s2th + sgp[layer, 2] * c2th

        # Effective horizontal conductivity tensor
        axx = sg[layer, 0, 0] - sg[layer, 0, 2] * \
            sg[layer, 2, 0] / sg[layer, 2, 2]
        axy = sg[layer, 0, 1] - sg[layer, 0, 2] * \
            sg[layer, 2, 1] / sg[layer, 2, 2]
        ayx = sg[layer, 1, 0] - sg[layer, 2, 0] * \
            sg[layer, 1, 2] / sg[layer, 2, 2]
        ayy = sg[layer, 1, 1] - sg[layer, 1, 2] * \
            sg[layer, 2, 1] / sg[layer, 2, 2]

        da12 = np.sqrt((axx - ayy)**2 + 4.0 * axy * ayx)
        al[layer] = 0.5 * (axx + ayy + da12)
        at[layer] = 0.5 * (axx + ayy - da12)

        # Anisotropy strike
        if da12 >= np.finfo(float).tiny:
            blt[layer] = 0.5 * np.arccos((axx - ayy) / da12)
        else:
            blt[layer] = 0.0
        if axy < 0.0:
            blt[layer] = -blt[layer]

    return sg, al, at, blt


def z1anis(nl, h, al, at, blt, per):
    """
    Computes surface impedance tensor for 1D layered anisotropic media.

    Parameters:
    ----------
    nl : int
        Number of layers including basement
    h : ndarray of shape (nl,)
        Layer thicknesses in km
    al, at : ndarray of shape (nl,)
        Maximum and minimum horizontal conductivities [S/m]
    blt : ndarray of shape (nl,)
        Anisotropy strike angles [radians]
    per : ndarray of shape (np,)
        Periods [s]

    Returns:
    -------
    z : ndarray of shape (2, 2, np)
        Surface impedance tensors [Ohm]
    """
    pi = np.pi
    ic = 1j
    mu0 = 4e-7 * pi
    np_ = len(per)
    z = np.zeros((2, 2, np_), dtype=complex)

    def dfp(x): return 1.0 + np.exp(-2.0 * x)
    def dfm(x): return 1.0 - np.exp(-2.0 * x)

    for kk in range(np_):
        period = per[kk]
        omega = 2.0 * pi / period
        k0 = (1.0 - ic) * 2e-3 * pi / np.sqrt(10.0 * period)

        layer = nl - 1
        a1 = al[layer]
        a2 = at[layer]
        bs = blt[layer]
        a1is = 1.0 / np.sqrt(a1)
        a2is = 1.0 / np.sqrt(a2)

        zrot = np.zeros((2, 2), dtype=complex)
        zrot[0, 1] = k0 * a1is
        zrot[1, 0] = -k0 * a2is

        if nl == 1:
            zp = rotz(zrot, -bs)
            z[:, :, kk] = zp
            continue

        bsref = bs

        for layer in range(nl - 2, -1, -1):
            hd = 1e3 * h[layer]
            a1 = al[layer]
            a2 = at[layer]
            bs = blt[layer]

            dtzbot = zrot[0, 0] * zrot[1, 1] - zrot[0, 1] * zrot[1, 0]
            if bs != bsref and a1 != a2:
                zbot = rotz(zrot, bs - bsref)
            else:
                zbot = zrot.copy()
                bs = bsref

            k1 = k0 * np.sqrt(a1)
            k2 = k0 * np.sqrt(a2)
            a1is = 1.0 / np.sqrt(a1)
            a2is = 1.0 / np.sqrt(a2)
            dz1 = k0 * a1is
            dz2 = k0 * a2is
            ag1 = k1 * hd
            ag2 = k2 * hd

            zdenom = (dtzbot * dfm(ag1) * dfm(ag2) / (dz1 * dz2) +
                      zbot[0, 1] * dfm(ag1) * dfp(ag2) / dz1 -
                      zbot[1, 0] * dfp(ag1) * dfm(ag2) / dz2 +
                      dfp(ag1) * dfp(ag2))

            zrot[0, 0] = 4.0 * zbot[0, 0] * np.exp(-ag1 - ag2) / zdenom
            zrot[0, 1] = (zbot[0, 1] * dfp(ag1) * dfp(ag2) -
                         zbot[1, 0] * dfm(ag1) * dfm(ag2) * dz1 / dz2 +
                         dtzbot * dfp(ag1) * dfm(ag2) / dz2 +
                         dfm(ag1) * dfp(ag2) * dz1) / zdenom
            zrot[1, 0] = (zbot[1, 0] * dfp(ag1) * dfp(ag2) -
                         zbot[0, 1] * dfm(ag1) * dfm(ag2) * dz2 / dz1 -
                         dtzbot * dfm(ag1) * dfp(ag2) / dz1 -
                         dfp(ag1) * dfm(ag2) * dz2) / zdenom
            zrot[1, 1] = 4.0 * zbot[1, 1] * np.exp(-ag1 - ag2) / zdenom

            bsref = bs

        if bsref != 0.0:
            zp = rotz(zrot, -bsref)
        else:
            zp = zrot.copy()

        z[:, :, kk] = zp

    return z


def rotz(za, betrad):
    """
    Rotates a 2×2 complex impedance tensor by a given angle in radians.

    This function applies a coordinate rotation to the impedance tensor `za`
    using the angle `betrad`, which represents the rotation from the original
    coordinate system to the new one. The rotation is performed using a
    double-angle transformation (2 × betrad), consistent with MT tensor rotation
    theory.

    Parameters:
    ----------
    za : ndarray of shape (2, 2), dtype=complex
        Original impedance tensor in the unrotated coordinate system.
    betrad : float
        Rotation angle in radians (positive = counterclockwise).

    Returns:
    -------
    zb : ndarray of shape (2, 2), dtype=complex
        Rotated impedance tensor in the new coordinate system.
    """

    co2 = np.cos(2.0 * betrad)
    si2 = np.sin(2.0 * betrad)
    sum1 = za[0, 0] + za[1, 1]
    sum2 = za[0, 1] + za[1, 0]
    dif1 = za[0, 0] - za[1, 1]
    dif2 = za[0, 1] - za[1, 0]
    zb = np.empty((2, 2), dtype=complex)
    zb[0, 0] = 0.5 * (sum1 + dif1 * co2 + sum2 * si2)
    zb[0, 1] = 0.5 * (dif2 + sum2 * co2 - dif1 * si2)
    zb[1, 0] = 0.5 * (-dif2 + sum2 * co2 - dif1 * si2)
    zb[1, 1] = 0.5 * (sum1 - dif1 * co2 - sum2 * si2)
    return zb

def rotzs(dzrot, nl, layer, betrad):
    """
    Rotates a stack of 2×2 complex sensitivity tensors by a given angle in radians.

    This function applies a coordinate rotation to the sensitivity tensors
    `dzrot[layer:nl,:,:]` using the angle `betrad`, consistent with the rotation
    applied to the impedance tensor. The rotation uses the same double-angle
    transformation (2 × betrad) and is applied individually to each layer's
    sensitivity tensor.

    Parameters:
    ----------
    dzrot : ndarray of shape (nl, 2, 2), dtype=complex
        Sensitivity tensors in the unrotated coordinate system.
    nl : int
        Total number of layers in the model.
    layer : int
        Starting layer index (1-based in Fortran, 0-based in Python).
    betrad : float
        Rotation angle in radians (positive = counterclockwise).

    Returns:
    -------
    dzout : ndarray of shape (nl, 2, 2), dtype=complex
        Rotated sensitivity tensors in the new coordinate system.
    """

    co2 = np.cos(2.0 * betrad)
    si2 = np.sin(2.0 * betrad)
    dzout = np.zeros_like(dzrot)
    for il in range(layer - 1, nl):
        dz = dzrot[il]
        sum1 = dz[0, 0] + dz[1, 1]
        sum2 = dz[0, 1] + dz[1, 0]
        dif1 = dz[0, 0] - dz[1, 1]
        dif2 = dz[0, 1] - dz[1, 0]
        dzout[il, 0, 0] = 0.5 * (sum1 + dif1 * co2 + sum2 * si2)
        dzout[il, 0, 1] = 0.5 * (dif2 + sum2 * co2 - dif1 * si2)
        dzout[il, 1, 0] = 0.5 * (-dif2 + sum2 * co2 - dif1 * si2)
        dzout[il, 1, 1] = 0.5 * (sum1 - dif1 * co2 - sum2 * si2)
    return dzout

def dfm(x):
    """
    Computes the regularized hyperbolic sine function used in MT impedance propagation.

    Parameters:
    ----------
    x : complex
        Attenuation factor (e.g., k * h)

    Returns:
    -------
    complex
        dfm(x) = 1 - exp(-2x)
    """
    return 1.0 - np.exp(-2.0 * x)

def dfp(x):
    """
    Computes the regularized hyperbolic cosine function used in MT impedance propagation.

    Parameters:
    ----------
    x : complex
        Attenuation factor (e.g., k * h)

    Returns:
    -------
    complex
        dfp(x) = 1 + exp(-2x)
    """
    return 1.0 + np.exp(-2.0 * x)


def zsprpg(dzbot, zbot, zrot, dz1, dz2, ag1, ag2):
    """
    Propagates the sensitivity of the impedance tensor through a single anisotropic layer.

    This function updates the sensitivity tensor at the top of the layer (`dztop`) by propagating
    the sensitivity from the bottom (`dzbot`) using the impedance tensors and attenuation factors.

    Parameters:
    ----------
    dzbot : ndarray (2,2)
        Sensitivity tensor at the bottom of the layer (e.g., ∂Z/∂parameter)
    zbot : ndarray (2,2)
        Impedance tensor at the bottom of the layer
    zrot : ndarray (2,2)
        Impedance tensor at the top of the layer (after propagation)
    dz1, dz2 : complex
        Propagation constants for the principal conductivities a1 and a2
    ag1, ag2 : complex
        Attenuation factors for a1 and a2 (typically k1*h and k2*h)

    Returns:
    -------
    dztop : ndarray (2,2)
        Sensitivity tensor at the top of the layer after propagation
    """
    ...

    dfp = lambda x: 1.0 + np.exp(-2.0 * x)
    dfm = lambda x: 1.0 - np.exp(-2.0 * x)

    dtzbot = zbot[0, 0] * zbot[1, 1] - zbot[0, 1] * zbot[1, 0]
    zdenom = (dtzbot * dfm(ag1) * dfm(ag2) / (dz1 * dz2) +
              zbot[0, 1] * dfm(ag1) * dfp(ag2) / dz1 -
              zbot[1, 0] * dfp(ag1) * dfm(ag2) / dz2 +
              dfp(ag1) * dfp(ag2))

    dztop = np.zeros((2, 2), dtype=complex)
    for i in range(2):
        for j in range(2):
            dztop[i, j] = dzbot[i, j] / zdenom  # Placeholder logic

    return dztop


def zs1anef(layani, h, al, at, blt, nl, per):
    pi = np.pi
    ic = 1j
    mu0 = 4e-7 * pi
    omega = 2.0 * pi / per
    k0 = (1.0 - ic) * 2e-3 * pi / np.sqrt(10.0 * per)

    z = np.zeros((2, 2), dtype=complex)
    dzdal = np.zeros((nl, 2, 2), dtype=complex)
    dzdat = np.zeros((nl, 2, 2), dtype=complex)
    dzdbs = np.zeros((nl, 2, 2), dtype=complex)
    dzdh = np.zeros((nl, 2, 2), dtype=complex)

    dzdalrot = np.zeros_like(dzdal)
    dzdatrot = np.zeros_like(dzdat)
    dzdbsrot = np.zeros_like(dzdbs)
    dzdhrot = np.zeros_like(dzdh)

    layer = nl - 1
    a1, a2, bs = al[layer], at[layer], blt[layer]
    a1is, a2is = 1.0 / np.sqrt(a1), 1.0 / np.sqrt(a2)
    zrot = np.zeros((2, 2), dtype=complex)
    zrot[0, 1] = k0 * a1is
    zrot[1, 0] = -k0 * a2is
    zprd = rotz(zrot, -bs)

    if layani[layer] == 0 and a1 == a2:
        dzdalrot[layer, 0, 1] = -0.5 * zrot[0, 1] / a1
        dzdatrot[layer, 1, 0] = -0.5 * zrot[1, 0] / a2
    else:
        dzdalrot[layer, 0, 1] = -0.5 * zrot[0, 1] / a1
        dzdatrot[layer, 1, 0] = -0.5 * zrot[1, 0] / a2
        dzdbsrot[layer, 0, 0] = -zrot[0, 1] - zrot[1, 0]
        dzdbsrot[layer, 1, 1] = -dzdbsrot[layer, 0, 0]
        layani[layer] = 1

    if nl == 1:
        z = rotz(zrot, -bs)
        dzdal = rotzs(dzdalrot, nl, 1, -bs)
        dzdat = rotzs(dzdatrot, nl, 1, -bs)
        dzdbs = rotzs(dzdbsrot, nl, 1, -bs)
        dzdh[layer] = np.zeros((2, 2), dtype=complex)
        return z, dzdal, dzdat, dzdbs, dzdh

    bsref = bs
    for layer in range(nl - 2, -1, -1):
        layer1 = layer + 1
        hd = 1e3 * h[layer]
        a1, a2, bs = al[layer], at[layer], blt[layer]
        dtzbot = zrot[0, 0] * zrot[1, 1] - zrot[0, 1] * zrot[1, 0]

        if bs != bsref and a1 != a2:
            zbot = rotz(zrot, bs - bsref)
            dzdal = rotzs(dzdalrot, nl, layer1, bs - bsref)
            dzdat = rotzs(dzdatrot, nl, layer1, bs - bsref)
            dzdbs = rotzs(dzdbsrot, nl, layer1, bs - bsref)
            dzdh = rotzs(dzdhrot, nl, layer1, bs - bsref)
        else:
            zbot = zrot.copy()
            bs = bsref
            dzdal[layer1:] = dzdalrot[layer1:]
            dzdat[layer1:] = dzdatrot[layer1:]
            dzdbs[layer1:] = dzdbsrot[layer1:]
            dzdh[layer1:] = dzdhrot[layer1:]

        k1, k2 = k0 * np.sqrt(a1), k0 * np.sqrt(a2)
        a1is, a2is = 1.0 / np.sqrt(a1), 1.0 / np.sqrt(a2)
        dz1, dz2 = k0 * a1is, k0 * a2is
        ag1, ag2 = k1 * hd, k2 * hd

        zdenom = (dtzbot * dfm(ag1) * dfm(ag2) / (dz1 * dz2) +
                  zbot[0, 1] * dfm(ag1) * dfp(ag2) / dz1 -
                  zbot[1, 0] * dfp(ag1) * dfm(ag2) / dz2 +
                  dfp(ag1) * dfp(ag2))

        zrot[0, 0] = 4.0 * zbot[0, 0] * np.exp(-ag1 - ag2) / zdenom
        zrot[0, 1] = (zbot[0, 1] * dfp(ag1) * dfp(ag2) -
                     zbot[1, 0] * dfm(ag1) * dfm(ag2) * dz1 / dz2 +
                     dtzbot * dfp(ag1) * dfm(ag2) / dz2 +
                     dfm(ag1) * dfp(ag2) * dz1) / zdenom
        zrot[1, 0] = (zbot[1, 0] * dfp(ag1) * dfp(ag2) -
                     zbot[0, 1] * dfm(ag1) * dfm(ag2) * dz2 / dz1 -
                     dtzbot * dfm(ag1) * df
                         # Final rotation to original coordinate system
    if bsref != 0.0:
        z = rotz(zrot, -bsref)
        dzdal = rotzs(dzdalrot, nl, 1, -bsref)
        dzdat = rotzs(dzdatrot, nl, 1, -bsref)
        dzdbs = rotzs(dzdbsrot, nl, 1, -bsref)
        dzdh = rotzs(dzdhrot, nl, 1, -bsref)
    else:
        z = zrot.copy()
        dzdal = dzdalrot.copy()
        dzdat = dzdatrot.copy()
        dzdbs = dzdbsrot.copy()
        dzdh = dzdhrot.copy()

    return z, dzdal, dzdat, dzdbs, dzdh

def zscua1(zbot, ztop, a1, dz1, dz2, ag1, ag2):
    """
    Computes the derivative of the impedance tensor with respect to maximum horizontal conductivity (AL).

    Parameters:
    ----------
    zbot : ndarray (2,2)
        Impedance tensor at bottom of layer
    ztop : ndarray (2,2)
        Impedance tensor at top of layer
    a1 : float
        Maximum horizontal conductivity (S/m)
    dz1, dz2 : complex
        Propagation constants for a1 and a2
    ag1, ag2 : complex
        Attenuation factors for a1 and a2

    Returns:
    -------
    dztop : ndarray (2,2)
        Sensitivity of impedance tensor w.r.t. AL
    """
    dtzbot = zbot[0,0]*zbot[1,1] - zbot[0,1]*zbot[1,0]
    zdenom = (dtzbot * dfm(ag1) * dfm(ag2) / (dz1 * dz2) +
              zbot[0,1] * dfm(ag1) * dfp(ag2) / dz1 -
              zbot[1,0] * dfp(ag1) * dfm(ag2) / dz2 +
              dfp(ag1) * dfp(ag2))

    dapom = dtzbot * dfm(ag2) / dz2 + zbot[0,1] * dfp(ag2)
    dbpom = dfp(ag2) - zbot[1,0] * dfm(ag2) / dz2
    dcpom = zbot[1,0] * dfp(ag2) - dz2 * dfm(ag2)
    ddpom = dtzbot * dfp(ag2) + dz2 * zbot[0,1] * dfm(ag2)
    depom = ag1 * dfm(ag1)
    dfpom = (dfm(ag1) + ag1 * dfp(ag1)) / dz1
    dgpom = (dfm(ag1) - ag1 * dfp(ag1)) * dz1

    dzdenom = dfpom * dapom + depom * dbpom
    dn12 = depom * dapom - dgpom * dbpom
    dn21 = depom * dcpom - dfpom * ddpom

    dztop = np.zeros((2,2), dtype=complex)
    dztop[0,0] = -0.5 * ztop[0,0] * dzdenom / (a1 * zdenom)
    dztop[0,1] =  0.5 * (dn12 - ztop[0,1] * dzdenom) / (a1 * zdenom)
    dztop[1,0] =  0.5 * (dn21 - ztop[1,0] * dzdenom) / (a1 * zdenom)
    dztop[1,1] = -0.5 * ztop[1,1] * dzdenom / (a1 * zdenom)
    return dztop

def zscua2(zbot, ztop, a2, dz1, dz2, ag1, ag2):
    """
    Computes the derivative of the impedance tensor with respect to minimum horizontal conductivity (AT).

    Parameters:
    ----------
    zbot : ndarray (2,2)
        Impedance tensor at bottom of layer
    ztop : ndarray (2,2)
        Impedance tensor at top of layer
    a2 : float
        Minimum horizontal conductivity (S/m)
    dz1, dz2 : complex
        Propagation constants for a1 and a2
    ag1, ag2 : complex
        Attenuation factors for a1 and a2

    Returns:
    -------
    dztop : ndarray (2,2)
        Sensitivity of impedance tensor w.r.t. AT
    """
    ...

    dtzbot = zbot[0,0]*zbot[1,1] - zbot[0,1]*zbot[1,0]
    zdenom = (dtzbot * dfm(ag1) * dfm(ag2) / (dz1 * dz2) +
              zbot[0,1] * dfm(ag1) * dfp(ag2) / dz1 -
              zbot[1,0] * dfp(ag1) * dfm(ag2) / dz2 +
              dfp(ag1) * dfp(ag2))

    dapom = dtzbot * dfm(ag1) / dz1 - zbot[1,0] * dfp(ag1)
    dbpom = dfp(ag1) + zbot[0,1] * dfm(ag1) / dz1
    dcpom = zbot[0,1] * dfp(ag1) + dz1 * dfm(ag1)
    ddpom = dtzbot * dfp(ag1) - dz1 * zbot[1,0] * dfm(ag1)
    depom = ag2 * dfm(ag2)
    dfpom = (dfm(ag2) + ag2 * dfp(ag2)) / dz2
    dgpom = (dfm(ag2) - ag2 * dfp(ag2)) * dz2

    dzdenom = dfpom * dapom + depom * dbpom
    dn12 = depom * dcpom + dfpom * ddpom
    dn21 = -depom * dapom + dgpom * dbpom

    dztop = np.zeros((2,2), dtype=complex)
    dztop[0,0] = -0.5 * ztop[0,0] * dzdenom / (a2 * zdenom)
    dztop[0,1] =  0.5 * (dn12 - ztop[0,1] * dzdenom) / (a2 * zdenom)
    dztop[1,0] =  0.5 * (dn21 - ztop[1,0] * dzdenom) / (a2 * zdenom)
    dztop[1,1] = -0.5 * ztop[1,1] * dzdenom / (a2 * zdenom)
    return dztop


def zscubs(zbot, ztop, dz1, dz2, ag1, ag2):
    """
    Computes the derivative of the impedance tensor with respect to anisotropy strike (BLT).

    Parameters:
    ----------
    zbot : ndarray (2,2)
        Impedance tensor at bottom of layer
    ztop : ndarray (2,2)
        Impedance tensor at top of layer
    dz1, dz2 : complex
        Propagation constants for a1 and a2
    ag1, ag2 : complex
        Attenuation factors for a1 and a2

    Returns:
    -------
    dztop : ndarray (2,2)
        Sensitivity of impedance tensor w.r.t. BLT
    """
    ...

    dtzbot = zbot[0,0]*zbot[1,1] - zbot[0,1]*zbot[1,0]
    zdenom = (dtzbot * dfm(ag1) * dfm(ag2) / (dz1 * dz2) +
              zbot[0,1] * dfm(ag1) * dfp(ag2) / dz1 -
              zbot[1,0] * dfp(ag1) * dfm(ag2) / dz2 +
              dfp(ag1) * dfp(ag2))

    dzbot = np.zeros((2,2), dtype=complex)
    dzbot[0,0] = 4.0 * (zbot[0,1] + zbot[1,0]) * np.exp(-ag1 - ag2)
    dzbot[0,1] = (zbot[0,0] - zbot[1,1]) * (dfm(ag1)*dfm(ag2)*dz1/dz2 - dfp(ag1)*dfp(ag2))
    dzbot[1,0] = (zbot[0,0] - zbot[1,1]) * (dfm(ag1)*dfm(ag2)*dz2/dz1 - dfp(ag1)*dfp(ag2))
    dzbot[1,1] = -dzbot[0,0]

    dpom = (zbot[0,0] - zbot[1,1]) * (dfm(ag1)*dfp(ag2)/dz1 - dfp(ag1)*dfm(ag2)/dz2)

    dztop = np.zeros((2,2), dtype=complex)
    dztop[0,0] = -ztop[0,1] - ztop[1,0] + (dzbot[0,0] + dpom * ztop[0,0]) / zdenom
    dztop[0,1] =  ztop[0,0] - ztop[1,1] + (dzbot[0,1] + dpom * ztop[0,1]) / zdenom
    dztop[1,0] =  dztop[0,1]
    dztop[1,1] = -dztop[0,0]
    return dztop

def zscuh(zbot, ztop, a1, a2, dz1, dz2, ag1, ag2):
    """
    Computes the derivative of the impedance tensor with respect to layer thickness h.

    Parameters:
    ----------
    zbot : ndarray (2,2)
        Impedance tensor at bottom of layer
    ztop : ndarray (2,2)
        Impedance tensor at top of layer
    a1, a2 : float
        Maximum and minimum horizontal conductivities
    dz1, dz2 : complex
        Propagation constants for a1 and a2
    ag1, ag2 : complex
        Attenuation factors for a1 and a2

    Returns:
    -------
    dztop : ndarray (2,2)
        Sensitivity of impedance tensor w.r.t. layer thickness
    """
    dtzbot = zbot[0,0]*zbot[1,1] - zbot[0,1]*zbot[1,0]
    zdenom = (dtzbot * dfm(ag1) * dfm(ag2) / (dz1 * dz2) +
              zbot[0,1] * dfm(ag1) * dfp(ag2) / dz1 -
              zbot[1,0] * dfp(ag1) * dfm(ag2) / dz2 +
              dfp(ag1) * dfp(ag2))

    k1 = a1 * dz1
    k2 = a2 * dz2

    # Partial derivatives w.r.t. ag1 and ag2
    dapom1 = dtzbot * dfm(ag1) / dz1 - zbot[1,0] * dfp(ag1)
    dbpom1 = dfp(ag1) + zbot[0,1] * dfm(ag1) / dz1
    dcpom1 = zbot[0,1] * dfp(ag1) + dz1 * dfm(ag1)
    ddpom1 = dtzbot * dfp(ag1) - dz1 * zbot[1,0] * dfm(ag1)

    dapom2 = dtzbot * dfm(ag2) / dz2 + zbot[0,1] * dfp(ag2)
    dbpom2 = dfp(ag2) - zbot[1,0] * dfm(ag2) / dz2
    dcpom2 = zbot[1,0] * dfp(ag2) - dz2 * dfm(ag2)
    ddpom2 = dtzbot * dfp(ag2) + dz2 * zbot[0,1] * dfm(ag2)

    dzdenom = (k1 * (dapom2 * dfp(ag1) / dz1 + dbpom2 * dfm(ag1)) +
               k2 * (dapom1 * dfp(ag2) / dz2 + dbpom1 * dfm(ag2)))

    dn12 = (k1 * (dapom2 * dfm(ag1) + dbpom2 * dz1 * dfp(ag1)) +
            k2 * (dcpom1 * dfm(ag2) + ddpom1 * dfp(ag2) / dz2))

    dn21 = (k1 * (dcpom2 * dfm(ag1) - ddpom2 * dfp(ag1) / dz1) -
            k2 * (dapom1 * dfm(ag2) + dbpom1 * dz2 * dfp(ag2)))

    dztop = np.zeros((2,2), dtype=complex)
    dztop[0,0] = -ztop[0,0] * dzdenom / zdenom
    dztop[0,1] = (dn12 - ztop[0,1] * dzdenom) / zdenom
    dztop[1,0] = (dn21 - ztop[1,0] * dzdenom) / zdenom
    dztop[1,1] = -ztop[1,1] * dzdenom / zdenom

    return dztop
