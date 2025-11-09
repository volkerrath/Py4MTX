#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Thu Oct 23 15:19:01 2025

@author: vrath
'''



import os
import sys
import time
from datetime import datetime
import warnings
import csv
import inspect

# Import numerical or other specialised modules
import numpy as np
# from mtpy.core.mt import MT
# import mtpy.core.mt as mt

import pymc as pm
import pytensor.tensor as pt
from pytensor.compile.ops import as_op


PY4MTX_DATA = os.environ['PY4MTX_DATA']
PY4MTX_ROOT = os.environ['PY4MTX_ROOT']

mypath = [PY4MTX_ROOT+'/py4mt/modules/', PY4MTX_ROOT+'/py4mt/scripts/']
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)

from version import versionstrg

import util as utl
import viz
import modem 
from aniso import prep_aniso, mt1d_aniso
from mtproc import calc_rhoa_phas
# from mcmc_funcs import pack_model, unpack_model


rhoair = 1.e17
rng = np.random.default_rng()
nan = np.nan  # float('NaN')
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())

titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng+'\n\n')

WorkDir = PY4MTX_ROOT+'/aniso/'
if not os.path.isdir(WorkDir):
    print(' File: %s does not exist, but will be created' % WorkDir)
    os.mkdir(WorkDir)
EdiDir = WorkDir+'edi'
ResDir = WorkDir+'results'


Periods = np.logspace(-4., 4., 41)

pi = np.pi
mu0 = 4e-7 * pi


SearchStrng = '*.edi'
edi_files = utl.get_filelist(searchstr=[SearchStrng], searchpath=EdiDir, 
                            sortedlist =True, fullpath=True)

dimlist = []
sit = 0
for filename in edi_files:
    sit = sit + 1
    print('reading data from: ' + filename)
    name, ext = os.path.splitext(filename)
    file_i = filename

# # Create MT object
#     mt_obj = MT()
#     mt_obj.read(file_i)

#     site = mt_obj.station_metadata.id
#     lat = mt_obj.station_metadata.location.latitude
#     lon = mt_obj.station_metadata.location.longitude
#     elev = mt_obj.station_metadata.location.elevation

#     Z_obj = mt_obj.Z
#     per = Z_obj.period

#     print(' site %s at :  % 10.6f % 10.6f % 8.1f' % (name, lat, lon, elev ))
    
#     print(type(per))
# --- 0) Imports & wiring (PyMC5 + PyTensor) ---
# --- 1) Observations (choose Z OR rhophi) ---
# per = np.asarray(PERIODS, float)  # (nper,) in seconds

# # (A) COMPLEX IMPEDANCES for all four tensor components
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

# # (B) ρa/φ (deg) for all four tensor components
# # obs = {
# #   "type": "rhophi",
# #   "rho_xx": np.asarray(RHO_XX), "phi_xx": np.asarray(PHI_XX_DEG),
# #   "sig_rho_xx": np.asarray(SIG_RHO_XX), "sig_phi_xx": np.asarray(SIG_PHI_XX_DEG),
# #   "rho_xy": np.asarray(RHO_XY), "phi_xy": np.asarray(PHI_XY_DEG),
# #   "sig_rho_xy": np.asarray(SIG_RHO_XY), "sig_phi_xy": np.asarray(SIG_PHI_XY_DEG),
# #   "rho_yx": np.asarray(RHO_YX), "phi_yx": np.asarray(PHI_YX_DEG),
# #   "sig_rho_yx": np.asarray(SIG_RHO_YX), "sig_phi_yx": np.asarray(SIG_PHI_YX_DEG),
# #   "rho_yy": np.asarray(RHO_YY), "phi_yy": np.asarray(PHI_YY_DEG),
# #   "sig_rho_yy": np.asarray(SIG_RHO_YY), "sig_phi_yy": np.asarray(SIG_PHI_YY_DEG),
# # }

# # Optionally drop some components from the likelihood (e.g., diagonals)
# use = dict(Zxx=True, Zxy=True, Zyx=True, Zyy=True)

# # --- 2) Model structure & priors ---
# nl = NL                            # number of layers
# layani = np.ones(nl, dtype=int)    # per-layer flags (edit if needed)

# mu_h_km, sigma_h_km = 0.5, 0.5     # thickness prior (km)
# mu_log10_r, sigma_log10_r = 1.5, 1.0
# angle_low, angle_high = -90.0, 90.0

# # --- 3) Forward helpers (NumPy) ---
# def _forward_stack(h_km, rop, ustr, udip, usla):
#     """Return dict of complex arrays (nper,) for Zxx, Zxy, Zyx, Zyy."""
#     _sg, al, at, blt = prep_aniso(rop, ustr, udip, usla)
#     Zxx = np.empty(per.size, complex)
#     Zxy = np.empty(per.size, complex)
#     Zyx = np.empty(per.size, complex)
#     Zyy = np.empty(per.size, complex)
#     for i, T in enumerate(per):
#         Z, *_ = mt1d_aniso(layani, h_km, al, at, blt, T)  # 2×2 top impedance
#         Zxx[i], Zxy[i], Zyx[i], Zyy[i] = Z[0,0], Z[0,1], Z[1,0], Z[1,1]
#     return {"Zxx": Zxx, "Zxy": Zxy, "Zyx": Zyx, "Zyy": Zyy}

# def _to_rho_phi(Z):
#     mu0 = 4e-7*np.pi
#     omega = 2*np.pi/per
#     rho = np.abs(Z)**2 / (mu0*omega)                     # Ω·m
#     phi = np.degrees(np.arctan2(np.imag(Z), np.real(Z))) # deg
#     return rho, phi

# def _pack(vecs, names_order, use_flags):
#     picked = [v for n, v in zip(names_order, vecs) if use_flags[n]]
#     return np.concatenate(picked, axis=0) if picked else np.array([], float)

# names = ["Zxx","Zxy","Zyx","Zyy"]

# # --- 4) PyTensor-wrapped black-box forward ---
# if obs["type"] == "Z":
#     @as_op(itypes=[pt.dvector, pt.dmatrix, pt.dvector, pt.dvector, pt.dvector],
#            otypes=[pt.dvector])
#     def fwd_op(h_km, rop, ustr, udip, usla):
#         out = _forward_stack(h_km, rop, ustr, udip, usla)
#         re_parts = [np.real(out[n]) for n in names]
#         im_parts = [np.imag(out[n]) for n in names]
#         vec_re = _pack(re_parts, names, use)
#         vec_im = _pack(im_parts, names, use)
#         return np.concatenate([vec_re, vec_im]).astype(float)

#     data_re = _pack([np.asarray(obs["Re_"+k]) for k in names], names, use)
#     data_im = _pack([np.asarray(obs["Im_"+k]) for k in names], names, use)
#     sig_re  = _pack([np.asarray(obs["sig_Re_"+k]) for k in names], names, use)
#     sig_im  = _pack([np.asarray(obs["sig_Im_"+k]) for k in names], names, use)
#     data_vec  = np.concatenate([data_re, data_im]).astype(float)
#     sigma_vec = np.concatenate([sig_re,  sig_im ]).astype(float)

# else:  # "rhophi"
#     @as_op(itypes=[pt.dvector, pt.dmatrix, pt.dvector, pt.dvector, pt.dvector],
#            otypes=[pt.dvector])
#     def fwd_op(h_km, rop, ustr, udip, usla):
#         out = _forward_stack(h_km, rop, ustr, udip, usla)
#         rhos, phis = [], []
#         for n in names:
#             r, p = _to_rho_phi(out[n])
#             rhos.append(r); phis.append(p)
#         vec_rho = _pack(rhos, names, use)
#         vec_phi = _pack(phis, names, use)
#         return np.concatenate([vec_rho, vec_phi]).astype(float)

#     data_rho = _pack([np.asarray(obs["rho_"+n[-2:]]) for n in names], [n[-2:] for n in names],
#                      {k: use[k] for k in names})
#     data_phi = _pack([np.asarray(obs["phi_"+n[-2:]]) for n in names], [n[-2:] for n in names],
#                      {k: use[k] for k in names})
#     sig_rho  = _pack([np.asarray(obs["sig_rho_"+n[-2:]]) for n in names], [n[-2:] for n in names],
#                      {k: use[k] for k in names})
#     sig_phi  = _pack([np.asarray(obs["sig_phi_"+n[-2:]]) for n in names], [n[-2:] for n in names],
#                      {k: use[k] for k in names})
#     data_vec  = np.concatenate([data_rho, data_phi]).astype(float)
#     sigma_vec = np.concatenate([sig_rho,  sig_phi ]).astype(float)

# # --- 5) PyMC model using PyTensor tensors ---
# with pm.Model() as model:
#     # Thicknesses: last is basement (0 km)
#     h_free = pm.HalfNormal("h_free", sigma=sigma_h_km, shape=nl-1)
#     h = pt.concatenate([h_free, pt.as_tensor_variable([0.0])])

#     # Principal resistivities (Ohm·m), log10-normal
#     log10_r = pm.Normal("log10_r", mu=mu_log10_r, sigma=sigma_log10_r, shape=(nl,3))
#     rop = pm.Deterministic("rop", 10**log10_r)

#     # Euler angles (deg)
#     ustr = pm.Uniform("ustr", lower=angle_low, upper=angle_high, shape=nl)
#     udip = pm.Uniform("udip", lower=angle_low, upper=angle_high, shape=nl)
#     usla = pm.Uniform("usla", lower=angle_low, upper=angle_high, shape=nl)

#     # Black-box forward and likelihood
#     pred = fwd_op(h, rop, ustr, udip, usla)

#     tau = pm.HalfNormal("tau", sigma=1.0)
#     sigma_pt = pt.clip(tau * pt.as_tensor_variable(sigma_vec), 1e-12, 1e9)
#     pm.Normal("obs", mu=pred, sigma=sigma_pt, observed=data_vec)

#     trace = pm.sample(
#         draws=2000, tune=2000, chains=4, cores=4,
#         step=pm.DEMetropolisZ(), target_accept=0.9, random_seed=42
#     )

# # --- 6) Summaries ---
# import arviz as az
# print(az.summary(trace, var_names=["h_free","tau","log10_r","ustr","udip","usla"]))
