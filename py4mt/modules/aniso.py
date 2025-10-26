import numpy as np
"""
module for 1-D anisotropic mt calculations
based on:

     Pek, J. and Santos, F. A. M., 2002. Magnetotelluric impedances and
         parametric sensitivities for 1-D generally anisotropic layered 
         media, Computers & Geosciences, 28
         doi:10.1016/S0098-3004(02)00014-6
         https://doi.org/10.1016/S0098-3004(02)00014-6}
    
    python routines translated with the help of CoPilot
    
    vr, Oct 26, 2025
"""

def zs1anis(
    layani, h, rop, ustr, udip, usla, al, at, blt, nl, per,
    z, dzdal, dzdat, dzdbs, dzdh,
    dzdsgpx, dzdsgpy, dzdsgpz, dzdstr, dzddip, dzdsla
):
    """
    Compute impedance tensor and sensitivities including geographic and orientation parameters.

    Parameters:
        layani : (nl,) int array
        h      : (nl,) float array - layer thicknesses
        rop    : (nl, 3) float array - survey positions (x, y, z)
        ustr   : (nl,) float array - strike angles (radians)
        udip   : (nl,) float array - dip angles (radians)
        usla   : (nl,) float array - slant angles (radians)
        al, at : (nl,) float arrays - principal conductivities
        blt    : (nl,) float array - anisotropy strike
        per    : float - period
        z      : (2, 2) complex array - output impedance tensor
        dzdal, dzdat, dzdbs, dzdh : (nl, 2, 2) complex arrays - parametric sensitivities
        dzdsgpx, dzdsgpy, dzdsgpz, dzdstr, dzddip, dzdsla : (nl, 2, 2) complex arrays - geographic sensitivities
    """
    for i in range(nl):
        # Extract orientation and location
        strike = ustr[i]
        dip = udip[i]
        slant = usla[i]
        sgpx, sgpy, sgpz = rop[i]

        # Rotate conductivity tensor and compute derivatives
        sigma, dsigma_dstrike, dsigma_ddip, dsigma_dslant = rotate_conductivity_tensor_with_derivatives(
            al[i], at[i], strike, dip, slant
        )

        # Extract 2D impedance-plane components
        axx, axy, ayy = sigma[0, 0], sigma[0, 1], sigma[1, 1]
        daxx_dstr, daxy_dstr, dayy_dstr = dsigma_dstrike[0, 0], dsigma_dstrike[0, 1], dsigma_dstrike[1, 1]
        daxx_ddip, daxy_ddip, dayy_ddip = dsigma_ddip[0, 0], dsigma_ddip[0, 1], dsigma_ddip[1, 1]
        daxx_dsla, daxy_dsla, dayy_dsla = dsigma_dslant[0, 0], dsigma_dslant[0, 1], dsigma_dslant[1, 1]

        # Store geographic sensitivities
        dzdstr[i]  = np.array([[daxx_dstr, daxy_dstr], [daxy_dstr, dayy_dstr]], dtype=np.complex128)
        dzddip[i]  = np.array([[daxx_ddip, daxy_ddip], [daxy_ddip, dayy_ddip]], dtype=np.complex128)
        dzdsla[i]  = np.array([[daxx_dsla, daxy_dsla], [daxy_dsla, dayy_dsla]], dtype=np.complex128)

        # Placeholder: sensitivities w.r.t. survey position (assumed zero unless spatial variation is modeled)
        dzdsgpx[i] = np.zeros((2, 2), dtype=np.complex128)
        dzdsgpy[i] = np.zeros((2, 2), dtype=np.complex128)
        dzdsgpz[i] = np.zeros((2, 2), dtype=np.complex128)

        # Update al, at, blt with rotated tensor components if needed
        al[i] = axx
        at[i] = ayy
        blt[i] = 0.5 * np.arctan2(2 * axy, axx - ayy)  # Effective anisotropy strike

    # Call zs1anef to compute impedance and parametric sensitivities
    z, dzdal, dzdat, dzdbs, dzdh = zs1anef(layani, h, al, at, blt, nl, per)

    return z, dzdal, dzdat, dzdbs, dzdh, dzdsgpx, dzdsgpy, dzdsgpz, dzdstr, dzddip, dzdsla

def rotate_conductivity_tensor_with_derivatives(al, at, strike, dip, slant):
    """
    Rotate a diagonal conductivity tensor into global coordinates and compute its
    sensitivities with respect to strike, dip, and slant angles.

    Parameters:
        al     : float - conductivity along strike
        at     : float - conductivity across strike and vertical
        strike : float - strike angle (radians)
        dip    : float - dip angle (radians)
        slant  : float - slant angle (radians)

    Returns:
        sigma_global     : (3, 3) ndarray - rotated conductivity tensor
        dsigma_dstrike   : (3, 3) ndarray - derivative w.r.t. strike
        dsigma_ddip      : (3, 3) ndarray - derivative w.r.t. dip
        dsigma_dslant    : (3, 3) ndarray - derivative w.r.t. slant
    """
    # Local conductivity tensor (diagonal)
    sigma_local = np.diag([al, at, at])

    # Rotation matrices
    Rstrike = np.array([
        [np.cos(strike), -np.sin(strike), 0],
        [np.sin(strike),  np.cos(strike), 0],
        [0,               0,              1]
    ])

    Rdip = np.array([
        [1, 0,              0],
        [0, np.cos(dip), -np.sin(dip)],
        [0, np.sin(dip),  np.cos(dip)]
    ])

    Rslant = np.array([
        [ np.cos(slant), 0, np.sin(slant)],
        [0,              1, 0],
        [-np.sin(slant), 0, np.cos(slant)]
    ])

    # Full rotation matrix
    R = Rslant @ Rdip @ Rstrike

    # Rotated conductivity tensor
    sigma_global = R @ sigma_local @ R.T

    # Derivatives of rotation matrices
    dRstrike_dstrike = np.array([
        [-np.sin(strike), -np.cos(strike), 0],
        [ np.cos(strike), -np.sin(strike), 0],
        [0,                0,              0]
    ])

    dRdip_ddip = np.array([
        [0, 0, 0],
        [0, -np.sin(dip), -np.cos(dip)],
        [0,  np.cos(dip), -np.sin(dip)]
    ])

    dRslant_dslant = np.array([
        [-np.sin(slant), 0,  np.cos(slant)],
        [0,              0,  0],
        [-np.cos(slant), 0, -np.sin(slant)]
    ])

    # Chain rule for full rotation derivatives
    dR_dstrike = Rslant @ Rdip @ dRstrike_dstrike
    dR_ddip    = Rslant @ dRdip_ddip @ Rstrike
    dR_dslant  = dRslant_dslant @ Rdip @ Rstrike

    # Tensor derivatives
    dsigma_dstrike = dR_dstrike @ sigma_local @ R.T + R @ sigma_local @ dR_dstrike.T
    dsigma_ddip    = dR_ddip    @ sigma_local @ R.T + R @ sigma_local @ dR_ddip.T
    dsigma_dslant  = dR_dslant  @ sigma_local @ R.T + R @ sigma_local @ dR_dslant.T

    return sigma_global, dsigma_dstrike, dsigma_ddip, dsigma_dslant


def propagate_impedance(zbot, dz1, dz2, ag1, ag2):
    dtzbot = zbot[0, 0] * zbot[1, 1] - zbot[0, 1] * zbot[1, 0]
    denom = (
        dtzbot * dfm(ag1) * dfm(ag2) / (dz1 * dz2)
        + zbot[0, 1] * dfm(ag1) * dfp(ag2) / dz1
        - zbot[1, 0] * dfp(ag1) * dfm(ag2) / dz2
        + dfp(ag1) * dfp(ag2)
    )
    ztop = np.zeros((2, 2), dtype=np.complex128)
    ztop[0, 0] = 4 * zbot[0, 0] * np.exp(-ag1 - ag2) / denom
    ztop[1, 1] = 4 * zbot[1, 1] * np.exp(-ag1 - ag2) / denom
    ztop[0, 1] = (
        zbot[0, 1] * dfp(ag1) * dfp(ag2)
        - zbot[1, 0] * dfm(ag1) * dfm(ag2) * dz1 / dz2
        + dtzbot * dfp(ag1) * dfm(ag2) / dz2
        + dfm(ag1) * dfp(ag2) * dz1
    ) / denom
    ztop[1, 0] = (
        zbot[1, 0] * dfp(ag1) * dfp(ag2)
        - zbot[0, 1] * dfm(ag1) * dfm(ag2) * dz2 / dz1
        - dtzbot * dfm(ag1) * dfp(ag2) / dz1
        - dfp(ag1) * dfm(ag2) * dz2
    ) / denom
    return ztop

def propagate_sensitivity(dzbot, zbot, ztop, dz1, dz2, ag1, ag2):
    dfm1, dfm2 = dfm(ag1), dfm(ag2)
    dfp1, dfp2 = dfp(ag1), dfp(ag2)
    dtzbot = zbot[0, 0] * zbot[1, 1] - zbot[0, 1] * zbot[1, 0]
    denom = (
        dtzbot * dfm1 * dfm2 / (dz1 * dz2)
        + zbot[0, 1] * dfm1 * dfp2 / dz1
        - zbot[1, 0] * dfp1 * dfm2 / dz2
        + dfp1 * dfp2
    )
    dztop = np.zeros((2, 2), dtype=np.complex128)
    for i in range(2):
        for j in range(2):
            term1 = 4 * dzbot[i, j] * np.exp(-ag1 - ag2) / denom
            # Simplified: omit ∂denom/∂zbot terms for now
            dztop[i, j] = term1
    return dztop

def zs1anef(layani, h, al, at, blt, nl, per):

    pi = np.pi
    ic = 1j
    mu0 = 4e-7 * pi

    omega = 2 * pi / per
    k0 = (1 - ic) * 2e-3 * pi / np.sqrt(10 * per)

    z = np.zeros((2, 2), dtype=np.complex128)
    dzdal = np.zeros((nl, 2, 2), dtype=np.complex128)
    dzdat = np.zeros((nl, 2, 2), dtype=np.complex128)
    dzdbs = np.zeros((nl, 2, 2), dtype=np.complex128)
    dzdh  = np.zeros((nl, 2, 2), dtype=np.complex128)

    # Bottom layer setup
    layer = nl - 1
    a1, a2, bs = al[layer], at[layer], blt[layer]
    a1is, a2is = 1 / np.sqrt(a1), 1 / np.sqrt(a2)
    zrot = np.array([[0, k0 * a1is], [-k0 * a2is, 0]], dtype=np.complex128)

    if layani[layer] == 0 and a1 == a2:
        dzdal[layer, 1, 0] = -0.5 * zrot[1, 0] / a1
        dzdat[layer, 0, 1] = -0.5 * zrot[0, 1] / a2
    else:
        dzdal[layer, 1, 0] = -0.5 * zrot[1, 0] / a1
        dzdat[layer, 0, 1] = -0.5 * zrot[0, 1] / a2
        dzdbs[layer, 0, 0] = -zrot[0, 1] - zrot[1, 0]
        dzdbs[layer, 1, 1] = -dzdbs[layer, 0, 0]
        layani[layer] = 1

    bsref = bs

    # Upward propagation
    for layer in reversed(range(nl - 1)):
        hd = 1e3 * h[layer]
        a1, a2, bs = al[layer], at[layer], blt[layer]
        a1is, a2is = 1 / np.sqrt(a1), 1 / np.sqrt(a2)
        dz1, dz2 = k0 * a1is, k0 * a2is
        ag1, ag2 = k0 * np.sqrt(a1) * hd, k0 * np.sqrt(a2) * hd

        if bs != bsref and a1 != a2:
            zbot = rotz(zrot, bs - bsref)
            dzdal[layer+1:] = rotzs(dzdal[layer+1:], layer+1, bs - bsref)
            dzdat[layer+1:] = rotzs(dzdat[layer+1:], layer+1, bs - bsref)
            dzdbs[layer+1:] = rotzs(dzdbs[layer+1:], layer+1, bs - bsref)
            dzdh[layer+1:]  = rotzs(dzdh[layer+1:],  layer+1, bs - bsref)
        else:
            zbot = zrot.copy()

        zrot = propagate_impedance(zbot, dz1, dz2, ag1, ag2)

        for il in range(layer + 1, nl):
            dzdal[il] = propagate_sensitivity(dzdal[il], zbot, zrot, dz1, dz2, ag1, ag2)
            dzdat[il] = propagate_sensitivity(dzdat[il], zbot, zrot, dz1, dz2, ag1, ag2)
            dzdbs[il] = propagate_sensitivity(dzdbs[il], zbot, zrot, dz1, dz2, ag1, ag2)
            dzdh[il]  = propagate_sensitivity(dzdh[il],  zbot, zrot, dz1, dz2, ag1, ag2)

        # Layer sensitivities (stubbed)
        dzdal[layer] = np.zeros((2, 2), dtype=np.complex128)
        dzdat[layer] = np.zeros((2, 2), dtype=np.complex128)
        dzdbs[layer] = np.zeros((2, 2), dtype=np.complex128)
        dzdh[layer]  = np.zeros((2, 2), dtype=np.complex128)

        bsref = bs

     # Final rotation to surface
    if bsref != 0:
        z = rotz(zrot, -bsref)
        dzdal = rotzs(dzdal, 0, -bsref)
        dzdat = rotzs(dzdat, 0, -bsref)
        dzdbs = rotzs(dzdbs, 0, -bsref)
        dzdh  = rotzs(dzdh,  0, -bsref)
    else:
        z = zrot.copy()

    return z, dzdal, dzdat, dzdbs, dzdh

def z1anis(nl, h, al, at, blt, per):
    """
    Stable impedance propagation for 1-D layered anisotropic media.

    Args:
        nl: int, number of layers including basement.
        h: array_like, shape (nl,), layer thicknesses in km (float32/float64).
        al: array_like, shape (nl,), max horizontal conductivities (S/m).
        at: array_like, shape (nl,), min horizontal conductivities (S/m).
        blt: array_like, shape (nl,), anisotropy strike angles in radians.
        per: array_like, shape (np,), periods in seconds.

    Returns:
        z: ndarray complex shape (2,2,len(per)), impedance tensor on surface for each period.
    """
    pi = np.pi
    ic = 1j

    h = np.asarray(h, dtype=float)
    al = np.asarray(al, dtype=float)
    at = np.asarray(at, dtype=float)
    blt = np.asarray(blt, dtype=float)
    per = np.asarray(per, dtype=float)

    if h.shape[0] < nl or al.shape[0] < nl or at.shape[0] < nl or blt.shape[0] < nl:
        raise ValueError("Input arrays h, al, at, blt must have length >= nl")

    np_per = per.shape[0]
    z = np.empty((2, 2, np_per), dtype=np.complex128)

    # constants
    mu0 = 4.0e-7 * pi
    # k0 as in original: (1 - i)*2/3*pi/sqrt(10*period)  -> factor simplified using period inside loop
    # keep structure: k0 = (1 - i) * 2.d-3 * pi / dsqrt(10.d0*period)  so
    # 2.d-3 = 0.002
    k0_prefactor = (1.0 - 1.0j) * 0.002 * pi

    for kk in range(np_per):
        period = per[kk]
        omega = 2.0 * pi / period
        iom = -ic * omega * mu0
        k0 = k0_prefactor / np.sqrt(10.0 * period)

        # basement (last layer)
        layer = nl - 1
        a1 = al[layer]
        a2 = at[layer]
        bs = blt[layer]

        k1 = k0 * np.sqrt(a1)
        k2 = k0 * np.sqrt(a2)
        a1is = 1.0 / np.sqrt(a1)
        a2is = 1.0 / np.sqrt(a2)

        zrot = np.zeros((2, 2), dtype=np.complex128)
        zrot[0, 0] = 0.0 + 0.0j
        zrot[0, 1] = k0 * a1is
        zrot[1, 0] = -k0 * a2is
        zrot[1, 1] = 0.0 + 0.0j

        # if only basement present, rotate back and store result
        if nl == 1:
            zp = rotz(zrot, -bs)
            z[:, :, kk] = zp
            continue

        bsref = bs
        # process layers above basement from layer nl-2 down to 0
        for layer in range(nl - 2, -1, -1):
            hd = 1.0 + 3.0 * float(h[layer])  # hd as in original: 1.d+3*dble(h)
            a1 = al[layer]
            a2 = at[layer]
            bs = blt[layer]

            # rotate zrot into current layer coordinates if needed
            dtzbot = zrot[0, 0] * zrot[1, 1] - zrot[0, 1] * zrot[1, 0]
            if (bs != bsref) and (a1 != a2):
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

            # denominator and rotated impedance zrot at top of this layer
            zdenom = (
                dtzbot * dfm(ag1) * dfm(ag2) / (dz1 * dz2)
                + zbot[0, 1] * dfm(ag1) * dfp(ag2) / dz1
                - zbot[1, 0] * dfp(ag1) * dfm(ag2) / dz2
                + dfp(ag1) * dfp(ag2)
            )

            # build new zrot (top of current layer, in strike-aligned coords)
            zrot_new = np.empty((2, 2), dtype=np.complex128)
            zrot_new[0, 0] = 4.0 * zbot[0, 0] * np.exp(-ag1 - ag2) / zdenom
            zrot_new[0, 1] = (
                zbot[0, 1] * dfp(ag1) * dfp(ag2)
                - zbot[1, 0] * dfm(ag1) * dfm(ag2) * dz1 / dz2
                + dtzbot * dfp(ag1) * dfm(ag2) / dz2
                + dfm(ag1) * dfp(ag2) * dz1
            ) / zdenom
            zrot_new[1, 0] = (
                zbot[1, 0] * dfp(ag1) * dfp(ag2)
                - zbot[0, 1] * dfm(ag1) * dfm(ag2) * dz2 / dz1
                - dtzbot * dfm(ag1) * dfp(ag2) / dz1
                - dfp(ag1) * dfm(ag2) * dz2
            ) / zdenom
            zrot_new[1, 1] = 4.0 * zbot[1, 1] * np.exp(-ag1 - ag2) / zdenom

            zrot = zrot_new
            bsref = bs

        # rotate final zrot into original coordinate system (if needed) and store
        if bsref != 0.0:
            zp = rotz(zrot, -bsref)
        else:
            zp = zrot.copy()

        z[0, 0, kk] = zp[0, 0]
        z[0, 1, kk] = zp[0, 1]
        z[1, 0, kk] = zp[1, 0]
        z[1, 1, kk] = zp[1, 1]

    return z

def zsprpg(dzbot, zbot, ztop, dz1, dz2, ag1, ag2):
    """
    Propagate parametric sensitivity from bottom (dzbot) to top (dztop)
    of an anisotropic layer.

    Args:
        dzbot: array-like complex shape (2,2) sensitivity at bottom.
        zbot: array-like complex shape (2,2) impedance at bottom.
        ztop: array-like complex shape (2,2) impedance at top.
        dz1, dz2, ag1, ag2: complex scalars.

    Returns:
        dztop: ndarray complex shape (2,2) sensitivity at top.
    """
    dzbot = np.asarray(dzbot, dtype=np.complex128)
    zbot = np.asarray(zbot, dtype=np.complex128)
    ztop = np.asarray(ztop, dtype=np.complex128)

    # dtzbot = det(zbot)
    dtzbot = zbot[0, 0] * zbot[1, 1] - zbot[0, 1] * zbot[1, 0]

    # denominator zdenom (same structure as in propagation routine)
    zdenom = (
        dtzbot * dfm(ag1) * dfm(ag2) / (dz1 * dz2)
        + zbot[0, 1] * dfm(ag1) * dfp(ag2) / dz1
        - zbot[1, 0] * dfp(ag1) * dfm(ag2) / dz2
        + dfp(ag1) * dfp(ag2)
    )

    # ddtzbot = derivative of det(zbot) w.r.t. parameters (given dzbot)
    ddtzbot = (
        dzbot[0, 0] * zbot[1, 1]
        + zbot[0, 0] * dzbot[1, 1]
        - dzbot[0, 1] * zbot[1, 0]
        - zbot[0, 1] * dzbot[1, 0]
    )

    # dzdenom: derivative of zdenom contributed by dzbot entries
    dzdenom = (
        ddtzbot * dfm(ag1) * dfm(ag2) / (dz1 * dz2)
        + dzbot[0, 1] * dfm(ag1) * dfp(ag2) / dz1
        - dzbot[1, 0] * dfp(ag1) * dfm(ag2) / dz2
    )

    # numerators dn11, dn12, dn21, dn22
    exp_term = np.exp(-ag1 - ag2)
    dn11 = 4.0 * dzbot[0, 0] * exp_term
    dn12 = (
        dzbot[0, 1] * dfp(ag1) * dfp(ag2)
        - dzbot[1, 0] * dfm(ag1) * dfm(ag2) * dz1 / dz2
        + ddtzbot * dfp(ag1) * dfm(ag2) / dz2
    )
    dn21 = (
        dzbot[1, 0] * dfp(ag1) * dfp(ag2)
        - dzbot[0, 1] * dfm(ag1) * dfm(ag2) * dz2 / dz1
        - ddtzbot * dfm(ag1) * dfp(ag2) / dz1
    )
    dn22 = 4.0 * dzbot[1, 1] * exp_term

    # assemble dztop
    dztop = np.empty((2, 2), dtype=np.complex128)
    dztop[0, 0] = (dn11 - ztop[0, 0] * dzdenom) / zdenom
    dztop[0, 1] = (dn12 - ztop[0, 1] * dzdenom) / zdenom
    dztop[1, 0] = (dn21 - ztop[1, 0] * dzdenom) / zdenom
    dztop[1, 1] = (dn22 - ztop[1, 1] * dzdenom) / zdenom

    return dztop

def cpanis(rop: np.ndarray, ustr: np.ndarray, udip: np.ndarray, usla: np.ndarray):
    """
    Compute conductivity tensors and effective horizontal anisotropy parameters.

    Args:
        rop: ndarray, shape (nl, 3) principal resistivities (Ohm·m).
        ustr: ndarray, shape (nl,) strike angles in degrees.
        udip: ndarray, shape (nl,) dip angles in degrees.
        usla: ndarray, shape (nl,) slant angles in degrees.

    Returns:
        sg: ndarray, shape (nl, 3, 3) conductivity tensors (S/m).
        al: ndarray, shape (nl,) maximum effective horizontal conductivity (S/m).
        at: ndarray, shape (nl,) minimum effective horizontal conductivity (S/m).
        blt: ndarray, shape (nl,) anisotropy strike in radians.
    """
    rop = np.asarray(rop, dtype=float)
    ustr = np.asarray(ustr, dtype=float)
    udip = np.asarray(udip, dtype=float)
    usla = np.asarray(usla, dtype=float)

    if rop.ndim != 2 or rop.shape[1] != 3:
        raise ValueError("rop must have shape (nl, 3)")
    nl = rop.shape[0]
    if not (ustr.shape[0] == udip.shape[0] == usla.shape[0] == nl):
        raise ValueError("ustr, udip, usla must have length nl")

    sg = np.zeros((nl, 3, 3), dtype=float)
    al = np.zeros(nl, dtype=float)
    at = np.zeros(nl, dtype=float)
    blt = np.zeros(nl, dtype=float)

    tiny_eps = np.finfo(float).tiny

    for layer in range(nl):
        sgp1 = 1.0 / float(rop[layer, 0])
        sgp2 = 1.0 / float(rop[layer, 1])
        sgp3 = 1.0 / float(rop[layer, 2])

        rstr = np.pi * float(ustr[layer]) / 180.0
        rdip = np.pi * float(udip[layer]) / 180.0
        rsla = np.pi * float(usla[layer]) / 180.0

        sps = np.sin(rstr)
        cps = np.cos(rstr)
        sth = np.sin(rdip)
        cth = np.cos(rdip)
        sfi = np.sin(rsla)
        cfi = np.cos(rsla)

        pom1 = sgp1 * cfi * cfi + sgp2 * sfi * sfi
        pom2 = sgp1 * sfi * sfi + sgp2 * cfi * cfi
        pom3 = (sgp1 - sgp2) * sfi * cfi

        c2ps = cps * cps
        s2ps = sps * sps
        c2th = cth * cth
        s2th = sth * sth
        csps = cps * sps
        csth = cth * sth

        sg[layer, 0, 0] = (
            pom1 * c2ps
            + pom2 * s2ps * c2th
            - 2.0 * pom3 * cth * csps
            + sgp3 * s2th * s2ps
        )
        sg[layer, 0, 1] = (
            pom1 * csps
            - pom2 * c2th * csps
            + pom3 * cth * (c2ps - s2ps)
            - sgp3 * s2th * csps
        )
        sg[layer, 0, 2] = -pom2 * csth * sps + pom3 * sth * cps + sgp3 * csth * sps
        sg[layer, 1, 0] = sg[layer, 0, 1]
        sg[layer, 1, 1] = (
            pom1 * s2ps
            + pom2 * c2ps * c2th
            + 2.0 * pom3 * cth * csps
            + sgp3 * s2th * c2ps
        )
        sg[layer, 1, 2] = pom2 * csth * cps + pom3 * sth * sps - sgp3 * csth * cps
        sg[layer, 2, 0] = sg[layer, 0, 2]
        sg[layer, 2, 1] = sg[layer, 1, 2]
        sg[layer, 2, 2] = pom2 * s2th + sgp3 * c2th

        denom = sg[layer, 2, 2]
        if np.abs(denom) < tiny_eps:
            denom = tiny_eps
        axx = sg[layer, 0, 0] - sg[layer, 0, 2] * sg[layer, 2, 0] / denom
        axy = sg[layer, 0, 1] - sg[layer, 0, 2] * sg[layer, 2, 1] / denom
        ayx = sg[layer, 1, 0] - sg[layer, 2, 0] * sg[layer, 1, 2] / denom
        ayy = sg[layer, 1, 1] - sg[layer, 1, 2] * sg[layer, 2, 1] / denom

        da12 = np.sqrt((axx - ayy) * (axx - ayy) + 4.0 * axy * ayx)
        al[layer] = 0.5 * (axx + ayy + da12)
        at[layer] = 0.5 * (axx + ayy - da12)

        if da12 >= tiny_eps:
            cos2 = (axx - ayy) / da12
            cos2 = np.clip(cos2, -1.0, 1.0)
            blt_val = 0.5 * np.arccos(cos2)
        else:
            blt_val = 0.0

        if axy < 0.0:
            blt_val = -blt_val
        blt[layer] = blt_val

    return sg, al, at, blt



def rotz(za, betrad: float):
    """
    Rotate a 2x2 impedance matrix za by angle betrad (radians) and return zb.

    Args:
        za: array-like shape (2,2) complex.
        betrad: float rotation angle in radians.

    Returns:
        zb: ndarray shape (2,2) complex.
    """
    za = np.asarray(za, dtype=np.complex128)
    if za.shape != (2, 2):
        raise ValueError("za must have shape (2,2)")

    co2 = np.cos(2.0 * betrad)
    si2 = np.sin(2.0 * betrad)

    sum1 = za[0, 0] + za[1, 1]
    sum2 = za[0, 1] + za[1, 0]
    dif1 = za[0, 0] - za[1, 1]
    dif2 = za[0, 1] - za[1, 0]

    zb = np.empty((2, 2), dtype=np.complex128)
    zb[0, 0] = 0.5 * (sum1 + dif1 * co2 + sum2 * si2)
    zb[0, 1] = 0.5 * (dif2 + sum2 * co2 - dif1 * si2)
    zb[1, 0] = 0.5 * (-dif2 + sum2 * co2 - dif1 * si2)
    zb[1, 1] = 0.5 * (sum1 - dif1 * co2 - sum2 * si2)

    return zb


def dfp(x):
    """Regularized hyperbolic cosinus:  dfp(x) = 1 + exp(-2*x)."""
    return 1.0 + np.exp(-2.0 * x)

def dfm(x):
    """Regularized hyperbolic sinus-like: dfm(x) = 1 - exp(-2*x)."""
    return 1.0 - np.exp(-2.0 * x)

def zscua1(zbot, ztop, a1, dz1, dz2, ag1, ag2):
    """
    Compute derivative of the impedance tensor with respect to maximum
    horizontal conductivity a1 within an anisotropic layer.

    Args:
        zbot: array-like complex shape (2,2) bottom impedance matrix.
        ztop: array-like complex shape (2,2) top impedance matrix.
        a1: float scalar (real).
        dz1: complex scalar.
        dz2: complex scalar.
        ag1: complex scalar.
        ag2: complex scalar.

    Returns:
        dztop: ndarray complex shape (2,2) derivative of ztop w.r.t. a1.
    """
    zbot = np.asarray(zbot, dtype=np.complex128)
    ztop = np.asarray(ztop, dtype=np.complex128)

    # intermediate products and combinations
    dtzbot = zbot[0, 0] * zbot[1, 1] - zbot[0, 1] * zbot[1, 0]

    zdenom = (
        dtzbot * dfm(ag1) * dfm(ag2) / (dz1 * dz2)
        + zbot[0, 1] * dfm(ag1) * dfp(ag2) / dz1
        - zbot[1, 0] * dfp(ag1) * dfm(ag2) / dz2
        + dfp(ag1) * dfp(ag2)
    )

    dapom = dtzbot * dfm(ag2) / dz2 + zbot[0, 1] * dfp(ag2)
    dbpom = dfp(ag2) - zbot[1, 0] * dfm(ag2) / dz2
    dcpom = zbot[1, 0] * dfp(ag2) - dz2 * dfm(ag2)
    ddpom = dtzbot * dfp(ag2) + dz2 * zbot[0, 1] * dfm(ag2)

    depom = ag1 * dfm(ag1)
    dfpom = (dfm(ag1) + ag1 * dfp(ag1)) / dz1
    dgpom = (dfm(ag1) - ag1 * dfp(ag1)) * dz1

    dzdenom = dfpom * dapom + depom * dbpom
    dn12 = depom * dapom - dgpom * dbpom
    dn21 = depom * dcpom - dfpom * ddpom

    # assemble derivative top impedance
    # guard against division by zero of zdenom and a1
    zdenom_safe = zdenom if zdenom != 0 else 1e-300 + 0j
    a1_safe = a1 if a1 != 0 else 1e-300

    dztop = np.empty((2, 2), dtype=np.complex128)
    dztop[0, 0] = -0.5 * ztop[0, 0] * dzdenom / (a1_safe * zdenom_safe)
    dztop[0, 1] = 0.5 * (dn12 - ztop[0, 1] * dzdenom) / (a1_safe * zdenom_safe)
    dztop[1, 0] = 0.5 * (dn21 - ztop[1, 0] * dzdenom) / (a1_safe * zdenom_safe)
    dztop[1, 1] = -0.5 * ztop[1, 1] * dzdenom / (a1_safe * zdenom_safe)

    return dztop

def zscua2(zbot, ztop, a2, dz1, dz2, ag1, ag2):
    """
    Compute derivative of the impedance tensor with respect to minimum
    horizontal conductivity a2 (AT) within an anisotropic layer.

    Args:
        zbot: array-like complex shape (2,2) bottom impedance matrix.
        ztop: array-like complex shape (2,2) top impedance matrix.
        a2: float scalar (real) minimum horizontal conductivity.
        dz1: complex scalar.
        dz2: complex scalar.
        ag1: complex scalar.
        ag2: complex scalar.

    Returns:
        dztop: ndarray complex shape (2,2) derivative of ztop w.r.t. a2.
    """
    zbot = np.asarray(zbot, dtype=np.complex128)
    ztop = np.asarray(ztop, dtype=np.complex128)

    dtzbot = zbot[0, 0] * zbot[1, 1] - zbot[0, 1] * zbot[1, 0]

    zdenom = (
        dtzbot * dfm(ag1) * dfm(ag2) / (dz1 * dz2)
        + zbot[0, 1] * dfm(ag1) * dfp(ag2) / dz1
        - zbot[1, 0] * dfp(ag1) * dfm(ag2) / dz2
        + dfp(ag1) * dfp(ag2)
    )

    dapom = dtzbot * dfm(ag1) / dz1 - zbot[1, 0] * dfp(ag1)
    dbpom = dfp(ag1) + zbot[0, 1] * dfm(ag1) / dz1
    dcpom = zbot[0, 1] * dfp(ag1) + dz1 * dfm(ag1)
    ddpom = dtzbot * dfp(ag1) - dz1 * zbot[1, 0] * dfm(ag1)

    depom = ag2 * dfm(ag2)
    dfpom = (dfm(ag2) + ag2 * dfp(ag2)) / dz2
    dgpom = (dfm(ag2) - ag2 * dfp(ag2)) * dz2

    dzdenom = dfpom * dapom + depom * dbpom
    dn12 = depom * dcpom + dfpom * ddpom
    dn21 = -depom * dapom + dgpom * dbpom

    # guard against division by zero
    zdenom_safe = zdenom if zdenom != 0 else 1e-300 + 0j
    a2_safe = a2 if a2 != 0 else 1e-300

    dztop = np.empty((2, 2), dtype=np.complex128)
    dztop[0, 0] = -0.5 * ztop[0, 0] * dzdenom / (a2_safe * zdenom_safe)
    dztop[0, 1] = 0.5 * (dn12 - ztop[0, 1] * dzdenom) / (a2_safe * zdenom_safe)
    dztop[1, 0] = 0.5 * (dn21 - ztop[1, 0] * dzdenom) / (a2_safe * zdenom_safe)
    dztop[1, 1] = -0.5 * ztop[1, 1] * dzdenom / (a2_safe * zdenom_safe)

    return dztop

def zscubs(zbot, ztop, dz1, dz2, ag1, ag2):
    """
    Compute derivative of the impedance tensor with respect to the
    effective horizontal anisotropy strike (in the strike-aligned frame).

    Args:
        zbot: array-like complex shape (2,2) bottom impedance matrix.
        ztop: array-like complex shape (2,2) top impedance matrix.
        dz1: complex scalar.
        dz2: complex scalar.
        ag1: complex scalar.
        ag2: complex scalar.

    Returns:
        dztop: ndarray complex shape (2,2) derivative of ztop w.r.t. strike.
    """
    zbot = np.asarray(zbot, dtype=np.complex128)
    ztop = np.asarray(ztop, dtype=np.complex128)

    # initial part: derivative components from ztop
    dztop = np.empty((2, 2), dtype=np.complex128)
    dztop[0, 0] = -ztop[0, 1] - ztop[1, 0]
    dztop[0, 1] = ztop[0, 0] - ztop[1, 1]
    dztop[1, 0] = dztop[0, 1]
    dztop[1, 1] = -dztop[0, 0]

    # determinant-like combination of zbot
    dtzbot = zbot[0, 0] * zbot[1, 1] - zbot[0, 1] * zbot[1, 0]

    # denominator used in correction terms
    zdenom = (
        dtzbot * dfm(ag1) * dfm(ag2) / (dz1 * dz2)
        + zbot[0, 1] * dfm(ag1) * dfp(ag2) / dz1
        - zbot[1, 0] * dfp(ag1) * dfm(ag2) / dz2
        + dfp(ag1) * dfp(ag2)
    )

    # dzbot components
    exp_term = np.exp(- (ag1 + ag2))
    dzbot = np.empty((2, 2), dtype=np.complex128)
    dzbot[0, 0] = 4.0 * (zbot[0, 1] + zbot[1, 0]) * exp_term
    dzbot[0, 1] = (zbot[0, 0] - zbot[1, 1]) * (
        dfm(ag1) * dfm(ag2) * dz1 / dz2 - dfp(ag1) * dfp(ag2)
    )
    dzbot[1, 0] = (zbot[0, 0] - zbot[1, 1]) * (
        dfm(ag1) * dfm(ag2) * dz2 / dz1 - dfp(ag1) * dfp(ag2)
    )
    dzbot[1, 1] = -4.0 * (zbot[0, 1] + zbot[1, 0]) * exp_term

    # dpom
    dpom = (zbot[0, 0] - zbot[1, 1]) * (
        dfm(ag1) * dfp(ag2) / dz1 - dfp(ag1) * dfm(ag2) / dz2
    )

    # add correction terms to dztop
    # guard against zero denominator by relying on numpy complex behavior
    dztop[0, 0] = dztop[0, 0] + (dzbot[0, 0] + dpom * ztop[0, 0]) / zdenom
    dztop[0, 1] = dztop[0, 1] + (dzbot[0, 1] + dpom * ztop[0, 1]) / zdenom
    dztop[1, 0] = dztop[1, 0] + (dzbot[1, 0] + dpom * ztop[1, 0]) / zdenom
    dztop[1, 1] = dztop[1, 1] + (dzbot[1, 1] + dpom * ztop[1, 1]) / zdenom

    return dztop

def dphase(z16):
    """
    Compute the real phase (float) of a complex value z16, following the
    original Fortran logic and returning an angle in radians in the range
    (-pi, pi].

    Args:
        z16: complex or numpy scalar of complex dtype

    Returns:
        float: phase of z16 in radians
    """
    pi = np.pi
    tiny = np.finfo(float).tiny

    # ensure a Python complex for scalar operations
    z = complex(z16)
    re = z.real
    im = z.imag

    if abs(re) >= tiny:
        pom = np.arctan(im / re)
        if re < 0.0:
            if im >= 0.0:
                return float(pom + pi)
            else:
                return float(pom - pi)
        else:
            return float(pom)
    else:
        if im < tiny:
            return 0.0
        else:
            if im > 0.0:
                return 0.5 * pi
            else:
                return -0.5 * pi


def rotzs(dza, la: int, nla: int, betrad: float):
    """
    Rotate sensitivity matrices dza for layers la..nla (inclusive) by angle betrad (radians).
    Args:
        dza: array-like, shape (nl, 2, 2), complex. Sensitivities for all layers.
        la: int, start layer index (Python 0-based).
        nla: int, end layer index (Python 0-based, inclusive).
        betrad: float, rotation angle in radians.
    Returns:
        dzb: ndarray, shape (nl, 2, 2), complex. Rotated sensitivities (same shape as dza).
    """
    dza = np.asarray(dza, dtype=np.complex128)
    if dza.ndim != 3 or dza.shape[1:] != (2, 2):
        raise ValueError("dza must have shape (nl, 2, 2)")
    nl = dza.shape[0]
    if not (0 <= la <= nla < nl):
        raise ValueError("la and nla must be 0 <= la <= nla < nl (Python 0-based indices)")

    co2 = np.cos(2.0 * betrad)
    si2 = np.sin(2.0 * betrad)

    dzb = np.array(dza, copy=True)  # preserve shape and dtype

    for l in range(la, nla + 1):
        sum1 = dza[l, 0, 0] + dza[l, 1, 1]
        sum2 = dza[l, 0, 1] + dza[l, 1, 0]
        dif1 = dza[l, 0, 0] - dza[l, 1, 1]
        dif2 = dza[l, 0, 1] - dza[l, 1, 0]

        dzb[l, 0, 0] = 0.5 * (sum1 + dif1 * co2 + sum2 * si2)
        dzb[l, 0, 1] = 0.5 * (dif2 + sum2 * co2 - dif1 * si2)
        dzb[l, 1, 0] = 0.5 * (-dif2 + sum2 * co2 - dif1 * si2)
        dzb[l, 1, 1] = 0.5 * (sum1 - dif1 * co2 - sum2 * si2)

    return

#if __name__ == "__main__":
    ## small self-test example
    #rop = np.array([[100.0, 200.0, 300.0]])
    #ustr = np.array([0.0])
    #udip = np.array([0.0])
    #usla = np.array([0.0])

    #sg, al, at, blt = cpanis(rop, ustr, udip, usla)
    #print("sg[0]:\n", sg[0])
    #print("al, at, blt (rad):", al[0], at[0], blt[0])

    ## dfp examples
    #print("dfp(0.5) =", dfp(0.5))
    #print("dfp(1+0.2j) =", dfp(1 + 0.2j))

    ## rotz example
    #za = np.array([[1 + 2j, 0.3 - 0.1j], [-0.1 + 0.2j, 0.5 + 0.6j]])
    #zb = rotz(za, np.pi / 6)
    #print("zb:\n", zb)
