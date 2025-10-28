#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 15:08:41 2025

@author: vrath
"""
import numpy as np

def aniso1d_fwd(model=None, per=None, out=False):
    '''
    Calculates data vector from parameter vector

    Parameters
    ----------
    model : np.array of shape (nl, 7)
        Model definition. The default is None.
        Flattened concatenation of:
            h : ndarray of shape (nl,)
            rop : ndarray of shape (nl, 3)
                Principal resistivities [Ohm·m]
            ustr, udip, usla : ndarray of shape (nl,)
                Euler angles [degrees]: strike, dip, slant

    per : np.array
        Periods. The default is None.
    out : logical
        Output control. The default is None.

    Returns
    -------
    data : np. array
        DESCRIPTION.

    '''
    
    # unpack model
    h   = model[:,0]
    rop = model[:,1:4]
    ustr = model[:,4]
    udip =  model[:,5]
    usla =  model[:,6]
    
    

    data = []
    
    return data


def pack_model(h, rop, ustr, udip, usla):
    # model = np.zeros((h.shape[0],7))
    # model[:,0] = h
    # model[:,1:4] = rop
    # model[:,4] = ustr
    # model[:,5] = udip
    # model[:,6] = usla
    model = np.hstack((h, rop, ustr, udip, usla))
    return model

def unpack_model(model):
    
    h   = model[:,0]
    rop = model[:,1:4]
    ustr = model[:,4]
    udip =  model[:,5]
    usla =  model[:,6]
    
    return h, rop, ustr, udip, usla




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


def rotz(za, betrad):
    """
    Rotates the 2x2 complex impedance tensor `za` by angle `betrad` (in radians).

    Parameters:
    ----------
    za : ndarray of shape (2, 2), dtype=complex
        Original impedance tensor
    betrad : float
        Rotation angle in radians

    Returns:
    -------
    zb : ndarray of shape (2, 2), dtype=complex
        Rotated impedance tensor
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

