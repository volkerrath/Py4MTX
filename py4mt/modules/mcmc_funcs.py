#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 19:58:45 2025

@author: vrath
"""
import sys
import numpy as np
#import multiprocessing as mp
import pymc as pm
from aesara.compile.ops import as_op
import aesara.tensor as at
from aniso import prep_aniso, mt1d_aniso


def rhophi_all(Z, per):
    """Return all 4 apparent resistivities and phases for 2x2 impedance tensor Z."""
    mu0 = 4e-7 * np.pi
    w = 2.0 * np.pi / per
    out = np.zeros(8)
    n = 0
    for i in range(2):
        for j in range(2):
            Zij = Z[i, j]
            rho = (abs(Zij)**2) / (mu0 * w)
            phi = np.degrees(np.arctan2(np.imag(Zij), np.real(Zij)))
            out[n:n+2] = [rho.real, phi]
            n += 2
    # order: xx, xy, yx, yy
    return out


def aniso1d_fwd(model=None, per=None, dataout='rhophi'):
    """
    1-D generally anisotropic MT forward, returns 
    full 2x2 impedance and all rho/phi pairs.

    Parameters
    ----------
    model : ndarray (nl,7)
        Layer parameters [h_km, rop1, rop2, rop3, ustr_deg, udip_deg, usla_deg]
    per : ndarray (m,)
        Periods [s]
    outputs :  str
        Any of "Z" or "rhophi"

    Returns
    -------
      Imped  : (m,4) complex arrays, surface impedance per period
      rhophi : (m,8) float array, per period:
                  [ρxx, φxx, ρxy, φxy, ρyx, φyx, ρyy, φyy]
    """
    if model is None or per is None:
        raise ValueError("Provide model (nl,7) and per (m,)")

    model = np.asarray(model, dtype=float)
    per = np.asarray(per, dtype=float)

    if model.ndim != 2 or model.shape[1] != 7:
        raise ValueError(
            "model must be (nl,7): [h, rop1, rop2, rop3, ustr, udip, usla]")

     
    h, rop, ustr, udip, usla = unpack_model(model)

    sg, al, at, blt = prep_aniso(rop, ustr, udip, usla)
    layani = np.where(np.isclose(al, at), 0, 1).astype(int)


    imped = np.zeros((per.size, 8), dtype=float)
    if 'rho' in dataout.lower():
       rhoph = np.zeros((per.size, 8), dtype=complex)

    for iper, psec in enumerate(per):
        Z, _, _, _, _ = mt1d_aniso(layani, h, al, at, blt, psec)
        imped[iper,:] = Z
        if 'rho' in dataout.lower():
            rhoph[iper, :] = rhophi_all(Z, psec)
        else:
            imped[iper,:] = Z
            

        if 'rho' in dataout.lower():
            data = rhoph
        else:
            data = imped

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

    h = model[:, 0]
    rop = model[:, 1:4]
    ustr = model[:, 4]
    udip = model[:, 5]
    usla = model[:, 6]

    return h, rop, ustr, udip, usla
 
# # -----------------------------
# # 1) Observations (FULL tensor)
# # -----------------------------
# # Periods [s], shape (nper,)
# per = np.asarray(PERIODS, float)

# # Choose ONE observation style:
# # (A) Complex impedances for all 4 components, each shape (nper,)
# obs = {
#     "type": "Z",  # "Z" or "rhophi"
#     # Zxx
#     "Re_Zxx": np.asarray(RE_ZXX), "Im_Zxx": np.asarray(IM_ZXX),
#     "sig_Re_Zxx": np.asarray(SIG_RE_ZXX), "sig_Im_Zxx": np.asarray(SIG_IM_ZXX),
#     # Zxy
#     "Re_Zxy": np.asarray(RE_ZXY), "Im_Zxy": np.asarray(IM_ZXY),
#     "sig_Re_Zxy": np.asarray(SIG_RE_ZXY), "sig_Im_Zxy": np.asarray(SIG_IM_ZXY),
#     # Zyx
#     "Re_Zyx": np.asarray(RE_ZYX), "Im_Zyx": np.asarray(IM_ZYX),
#     "sig_Re_Zyx": np.asarray(SIG_RE_ZYX), "sig_Im_Zyx": np.asarray(SIG_IM_ZYX),
#     # Zyy
#     "Re_Zyy": np.asarray(RE_ZYY), "Im_Zyy": np.asarray(IM_ZYY),
#     "sig_Re_Zyy": np.asarray(SIG_RE_ZYY), "sig_Im_Zyy": np.asarray(SIG_IM_ZYY),
# }

# # (B) Apparent resistivity & phase (deg) for all 4 components, each shape (nper,)
# # obs = {
# #     "type": "rhophi",
# #     # xx
# #     "rho_xx": np.asarray(RHO_XX), "phi_xx": np.asarray(PHI_XX_DEG),
# #     "sig_rho_xx": np.asarray(SIG_RHO_XX), "sig_phi_xx": np.asarray(SIG_PHI_XX_DEG),
# #     # xy
# #     "rho_xy": np.asarray(RHO_XY), "phi_xy": np.asarray(PHI_XY_DEG),
# #     "sig_rho_xy": np.asarray(SIG_RHO_XY), "sig_phi_xy": np.asarray(SIG_PHI_XY_DEG),
# #     # yx
# #     "rho_yx": np.asarray(RHO_YX), "phi_yx": np.asarray(PHI_YX_DEG),
# #     "sig_rho_yx": np.asarray(SIG_RHO_YX), "sig_phi_yx": np.asarray(SIG_PHI_YX_DEG),
# #     # yy
# #     "rho_yy": np.asarray(RHO_YY), "phi_yy": np.asarray(PHI_YY_DEG),
# #     "sig_rho_yy": np.asarray(SIG_RHO_YY), "sig_phi_yy": np.asarray(SIG_PHI_YY_DEG),
# # }

# # Optional: a mask (True=use, False=ignore) per component if you want to exclude some (e.g., diagonals)
# use = dict(Zxx=True, Zxy=True, Zyx=True, Zyy=True)  # flip any to False to drop from likelihood

# # -----------------------------
# # 2) Fixed model structure
# # -----------------------------
# nl = NL
# layani = np.ones(nl, dtype=int)

# mu_h_km, sigma_h_km = 0.5, 0.5
# mu_log10_r, sigma_log10_r = 1.5, 1.0
# angle_low, angle_high = -90.0, 90.0

# -----------------------------
# 3) Forward helpers
# -----------------------------
def _forward_stack(h_km, rop, ustr, udip, usla):
    """
    Returns dict of complex arrays (nper,) for all four Z components: Zxx, Zxy, Zyx, Zyy
    """
    _sg, al, at, blt = prep_aniso(rop, ustr, udip, usla)
    Zxx = np.empty(per.size, complex)
    Zxy = np.empty(per.size, complex)
    Zyx = np.empty(per.size, complex)
    Zyy = np.empty(per.size, complex)
    for i, T in enumerate(per):
        Z, *_ = mt1d_aniso(layani, h_km, al, at, blt, T)  # Z is 2x2 at surface
        Zxx[i], Zxy[i], Zyx[i], Zyy[i] = Z[0,0], Z[0,1], Z[1,0], Z[1,1]
    return {"Zxx": Zxx, "Zxy": Zxy, "Zyx": Zyx, "Zyy": Zyy}

def _to_rho_phi(Z):
    mu0 = 4e-7*np.pi
    omega = 2*np.pi/per
    rho = np.abs(Z)**2 / (mu0*omega)
    phi = np.degrees(np.arctan2(np.imag(Z), np.real(Z)))
    return rho, phi

# Helper to optionally drop components from the stacked vector
def _pack_components(vecs, names_order, use_flags):
    picked = [v for n, v in zip(names_order, vecs) if use_flags[n]]
    return np.concatenate(picked, axis=0) if picked else np.array([], float)

# ----------------------------------------
# 4) Aesara-wrapped forward (full tensor)
# ----------------------------------------
names = ["Zxx","Zxy","Zyx","Zyy"]  # fixed order for packing

if obs["type"] == "Z":
    @as_op(itypes=[at.dvector, at.dmatrix, at.dvector, at.dvector, at.dvector],
           otypes=[at.dvector])
    def fwd_op(h_km, rop, ustr, udip, usla):
        out = _forward_stack(h_km, rop, ustr, udip, usla)
        # real/imag stacks for each component
        comps_re = [np.real(out[n]) for n in names]
        comps_im = [np.imag(out[n]) for n in names]
        vec_re = _pack_components(comps_re, names, use)
        vec_im = _pack_components(comps_im, names, use)
        return np.concatenate([vec_re, vec_im]).astype(float)

    data_re = _pack_components(
        [np.asarray(obs["Re_"+n.upper()]) for n in names], [n.upper() for n in names], {k.upper():v for k,v in use.items()}
    )
    data_im = _pack_components(
        [np.asarray(obs["Im_"+n.upper()]) for n in names], [n.upper() for n in names], {k.upper():v for k,v in use.items()}
    )
    sig_re = _pack_components(
        [np.asarray(obs["sig_Re_"+n.upper()]) for n in names], [n.upper() for n in names], {k.upper():v for k,v in use.items()}
    )
    sig_im = _pack_components(
        [np.asarray(obs["sig_Im_"+n.upper()]) for n in names], [n.upper() for n in names], {k.upper():v for k,v in use.items()}
    )

    data_vec = np.concatenate([data_re, data_im]).astype(float)
    sigma_vec = np.concatenate([sig_re, sig_im]).astype(float)

else:  # "rhophi"
    @as_op(itypes=[at.dvector, at.dmatrix, at.dvector, at.dvector, at.dvector],
           otypes=[at.dvector])
    def fwd_op(h_km, rop, ustr, udip, usla):
        out = _forward_stack(h_km, rop, ustr, udip, usla)
        rhos, phis = [], []
        for n in names:
            rho, phi = _to_rho_phi(out[n])
            rhos.append(rho); phis.append(phi)
        vec_rho = _pack_components(rhos, names, use)
        vec_phi = _pack_components(phis, names, use)
        return np.concatenate([vec_rho, vec_phi]).astype(float)

    data_rho = _pack_components(
        [np.asarray(obs["rho_"+n[-2:]]) for n in names], [n[-2:] for n in names], {k[-2:]:v for k,v in use.items()}
    )
    data_phi = _pack_components(
        [np.asarray(obs["phi_"+n[-2:]]) for n in names], [n[-2:] for n in names], {k[-2:]:v for k,v in use.items()}
    )
    sig_rho = _pack_components(
        [np.asarray(obs["sig_rho_"+n[-2:]]) for n in names], [n[-2:] for n in names], {k[-2:]:v for k,v in use.items()}
    )
    sig_phi = _pack_components(
        [np.asarray(obs["sig_phi_"+n[-2:]]) for n in names], [n[-2:] for n in names], {k[-2:]:v for k,v in use.items()}
    )

    data_vec = np.concatenate([data_rho, data_phi]).astype(float)
    sigma_vec = np.concatenate([sig_rho, sig_phi]).astype(float)

# ----------------------------------------
# 5) PyMC model (unchanged, but now “full”)
# ----------------------------------------
with pm.Model() as model:
    # thicknesses (km) — last is basement
    h_free = pm.HalfNormal("h_free", sigma=sigma_h_km, shape=nl-1)
    h = at.concatenate([h_free, at.as_tensor_variable([0.0])])

    # principal resistivities (Ohm·m)
    log10_r = pm.Normal("log10_r", mu=mu_log10_r, sigma=sigma_log10_r, shape=(nl,3))
    rop = pm.Deterministic("rop", 10**log10_r)

    # Euler angles (deg)
    ustr = pm.Uniform("ustr", lower=angle_low, upper=angle_high, shape=nl)
    udip = pm.Uniform("udip", lower=angle_low, upper=angle_high, shape=nl)
    usla = pm.Uniform("usla", lower=angle_low, upper=angle_high, shape=nl)

    # forward + likelihood
    pred = fwd_op(h, rop, ustr, udip, usla)

    tau = pm.HalfNormal("tau", sigma=1.0)  # inflate/deflate reported sigmas
    like_sigma = pm.Deterministic("like_sigma", at.clip(tau * at.as_tensor_variable(sigma_vec), 1e-12, 1e9))

    pm.Normal("obs", mu=pred, sigma=like_sigma, observed=data_vec)

    trace = pm.sample(draws=2000, tune=2000, chains=4, cores=4,
                      step=pm.DEMetropolisZ(), target_accept=0.9, random_seed=42)

# quick summary
import arviz as az
az.summary(trace, var_names=["h_free","tau","log10_r","ustr","udip","usla"])
