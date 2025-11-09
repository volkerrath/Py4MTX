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
import pytensor.tensor as pt
# from pytensor.compile.ops import as_op

from aniso import prep_aniso, mt1d_aniso


# def rhophi_all(Z, per):
#     """Return all 4 apparent resistivities and phases for 2x2 impedance tensor Z."""
#     mu0 = 4e-7 * np.pi
#     w = 2.0 * np.pi / per
#     out = np.zeros(8)
#     n = 0
#     for i in range(2):
#         for j in range(2):
#             Zij = Z[i, j]
#             rho = (abs(Zij)**2) / (mu0 * w)
#             phi = np.degrees(np.arctan2(np.imag(Zij), np.real(Zij)))
#             out[n:n+2] = [rho.real, phi]
#             n += 2
#     # order: xx, xy, yx, yy
#     return out


# def aniso1d_fwd(model=None, per=None, dataout='rhophi'):
#     """
#     1-D generally anisotropic MT forward, returns 
#     full 2x2 impedance and all rho/phi pairs.

#     Parameters
#     ----------
#     model : ndarray (nl,7)
#         Layer parameters [h_km, rop1, rop2, rop3, ustr_deg, udip_deg, usla_deg]
#     per : ndarray (m,)
#         Periods [s]
#     outputs :  str
#         Any of "Z" or "rhophi"

#     Returns
#     -------
#       Imped  : (m,4) complex arrays, surface impedance per period
#       rhophi : (m,8) float array, per period:
#                   [ρxx, φxx, ρxy, φxy, ρyx, φyx, ρyy, φyy]
#     """
#     if model is None or per is None:
#         raise ValueError("Provide model (nl,7) and per (m,)")

#     model = np.asarray(model, dtype=float)
#     per = np.asarray(per, dtype=float)

#     if model.ndim != 2 or model.shape[1] != 7:
#         raise ValueError(
#             "model must be (nl,7): [h, rop1, rop2, rop3, ustr, udip, usla]")

     
#     h, rop, ustr, udip, usla = unpack_model(model)

#     sg, al, at, blt = prep_aniso(rop, ustr, udip, usla)
#     layani = np.where(np.isclose(al, at), 0, 1).astype(int)


#     imped = np.zeros((per.size, 8), dtype=float)
#     if 'rho' in dataout.lower():
#        rhoph = np.zeros((per.size, 8), dtype=complex)

#     for iper, psec in enumerate(per):
#         Z, _, _, _, _ = mt1d_aniso(layani, h, al, at, blt, psec)
#         imped[iper,:] = Z
#         if 'rho' in dataout.lower():
#             rhoph[iper, :] = rhophi_all(Z, psec)
#         else:
#             imped[iper,:] = Z
            

#         if 'rho' in dataout.lower():
#             data = rhoph
#         else:
#             data = imped

#     return data


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

