# -*- coding: utf-8 -*-
"""
Created on Sun Nov  30, 2024

@author: vrath
"""

import os
import sys
import ast
import numpy as np
from scipy.ndimage import gaussian_filter, laplace, convolve, gaussian_gradient_magnitude
from scipy.linalg import norm
from sys import exit as error
import fnmatch
# from numba import jit
# from shapely.geometry import Point
# from shapely.geometry.polygon import Polygon
import pyproj
from pyproj import CRS, Transformer


from sys import exit as error

from mtpy import MT , MTData, MTCollection



def get_edi_list(edirname=None, sort=False):



    edi_files = []
    files = os.listdir(edirname)
    for entry in files:
        # print(entry)
        if entry.endswith(".edi") and not entry.startswith("."):
            edi_files.append(edirname+entry)
    ns = np.size(edi_files)
    if ns ==0:
        error("No edi files found in "+edirname+"! Exit.")

    if sort:
        edi_files

    return edi_files

def make_data(edirname=None,
                    collection="./My_Collection.h5",
                    metaid="my_collection",
                    survey="my_survey",
                    savedata=True,
                    utm_epsg=32629
                    ):
    """
    Make MTData object from edi-files, optionally save to MTCollection

    Parameters
    ----------
    edirname : TYPE, optional
        DESCRIPTION. The default is None.
    collection : TYPE, optional
        DESCRIPTION. The default is "./My_Collection.h5".
    metaid : TYPE, optional
        DESCRIPTION. The default is "my_collection".
    survey : TYPE, optional
        DESCRIPTION. The default is "my_survey".
    savedata : TYPE, optional
        DESCRIPTION. The default is True.
    utm_epsg : TYPE, optional
        DESCRIPTION. The default is 32629.

    Returns
    -------

    mtd: MtData object

    """


    edi_files = []
    files = os.listdir(edirname)
    for entry in files:
        # print(entry)
        if entry.endswith(".edi") and not entry.startswith("."):
            edi_files.append(edirname+entry)
    ns = np.size(edi_files)
    if ns ==0:
        error("No edi files found in "+edirname+"! Exit.")

    mtd = MTData()
    sit = 0
    for fil in edi_files:
        sit = sit + 1
        print("reading data from: " + fil)
        file_i = fil

    # Create MT object

        mt_obj = MT()
        mt_obj.read(file_i)
        mt_obj.survey_metadata.id = survey
        mtd.add_station(mt_obj)

    mtd.utm_crs = utm_epsg


    if savedata:
        with MTCollection() as mtc:
            mtc.open_collection(collection)
            mtc.from_mt_data(mtd)
            # mtc.working_dataframe = mtc.master_dataframe.loc[mtc.master_dataframe.survey == survey]
            mtc.close_collection()


    return mtd

def make_collection(edirname=None,
                    collection="./My_Collection.h5",
                    metaid="my_collection",
                    survey="my_survey",
                    returndata=True,
                    utm_epsg=32629
                    ):
    """
    Make MTCollectionlection from edi-files, optionally returns MTData object

    Parameters
    ----------
    edirname : TYPE, optional
        DESCRIPTION. The default is None.
    collection : TYPE, optional
        DESCRIPTION. The default is "./My_Collection.h5".
    metaid : TYPE, optional
        DESCRIPTION. The default is "my_collection".
    survey : TYPE, optional
        DESCRIPTION. The default is "my_survey".
    returndata : TYPE, optional
        DESCRIPTION. The default is True.
    utm_epsg : TYPE, optional
        DESCRIPTION. The default is 32629.

    Returns
    -------
    mtd : TYPE
        DESCRIPTION.

    """



    edi_files = []
    files = os.listdir(edirname)
    for entry in files:
        # print(entry)
        if entry.endswith(".edi") and not entry.startswith("."):
            edi_files.append(edirname+entry)
    ns = np.size(edi_files)
    if ns ==0:
        error("No edi files found in "+edirname+"! Exit.")

    mtc = MTCollection()
    mtc.open_collection(collection)
    sit = 0
    for fil in edi_files:
        sit = sit + 1
        print("reading data from: " + fil)
        file_i = fil

    # Create MT object

        mt_obj = MT()
        mt_obj.read(file_i)
        mt_obj.survey_metadata.id = metaid
        mtc.add_tf(mt_obj)

    mtc.working_dataframe = mtc.master_dataframe.loc[mtc.master_dataframe.survey == survey]
    mtc.utm_crs = utm_epsg
    mtd = mtc.to_mt_data()
    mtc.close_collection()

    if returndata:
        return mtd



def calc_rhoa_phas(freq=None, Z=None):

    mu0 = 4.0e-7 * np.pi  # Magnetic Permeability (H/m)
    omega = 2.*np*freq

    rhoa = np.power(np.abs(Z), 2) / (mu0 * omega)
    # phi = np.rad2deg(np.arctan(Z.imag / Z.real))
    phi = np.angle(Z, deg=True)

    return rhoa, phi

def mt1dfwd(freq, sig, d, inmod="r", out="imp", magfield="b"):
    """
    Calulate 1D magnetotelluric forward response.

    based on A. Pethik's script at www.digitalearthlab.com
    Last change vr Nov 20, 2020
    """
    mu0 = 4.0e-7 * np.pi  # Magnetic Permeability (H/m)

    sig = np.array(sig)
    freq = np.array(freq)
    d = np.array(d)

    if inmod[0] == "c":
        sig = np.array(sig)
    elif inmod[0] == "r":
        sig = 1.0 / np.array(sig)

    if sig.ndim > 1:
        error("IP not yet implemented")

    n = np.size(sig)

    Z = np.zeros_like(freq) + 1j * np.zeros_like(freq)
    w = np.zeros_like(freq)

    ifr = -1
    for f in freq:
        ifr = ifr + 1
        w[ifr] = 2.0 * np.pi * f
        imp = np.array(range(n)) + np.array(range(n)) * 1j

        # compute basement impedance
        imp[n - 1] = np.sqrt(1j * w[ifr] * mu0 / sig[n - 1])

        for layer in range(n - 2, -1, -1):
            sl = sig[layer]
            dl = d[layer]
            # 3. Compute apparent rho from top layer impedance
            # Step 2. Iterate from bottom layer to top(not the basement)
            #   Step 2.1 Calculate the intrinsic impedance of current layer
            dj = np.sqrt(1j * w[ifr] * mu0 * sl)
            wj = dj / sl
            #   Step 2.2 Calculate Exponential factor from intrinsic impedance
            ej = np.exp(-2 * dl * dj)

            #   Step 2.3 Calculate reflection coeficient using current layer
            #          intrinsic impedance and the below layer impedance
            impb = imp[layer + 1]
            rj = (wj - impb) / (wj + impb)
            re = rj * ej
            Zj = wj * ((1 - re) / (1 + re))
            imp[layer] = Zj

        Z[ifr] = imp[0]
        # print(Z[ifr])

    if out.lower() == "imp":

        if magfield.lower() =="b":
            return Z/mu0
        else:
            return Z

    elif out.lower() == "rho":
        absZ = np.abs(Z)
        rhoa = (absZ * absZ) / (mu0 * w)
        phase = np.rad2deg(np.arctan(Z.imag / Z.real))

        return rhoa, phase
    else:
        absZ = np.abs(Z)
        rhoa = (absZ * absZ) / (mu0 * w)
        phase = np.rad2deg(np.arctan(Z.imag / Z.real))
        return Z, rhoa, phase

def wait1d(periods=None, thick=None, res=None):

    scale = 1 / (4 * np.pi / 10000000)
    mu = 4 * np.pi * 1e-7 * scale
    omega = 2 * np.pi / periods

    cond = 1 / np.array(res)

    sp = np.size(periods)
    Z = np.zeros(sp, dtype=complex)
    rhoa = np.zeros(sp)
    phi = np.zeros(sp)

    for freq, w in enumerate(omega):
        prop_const = np.sqrt(1j*mu*cond[-1] * w)
        C = np.zeros(sp, dtype=complex)
        C[-1] = 1 / prop_const
        if len(thick) > 1:
            for k in reversed(range(len(res) - 1)):
                prop_layer = np.sqrt(1j*w*mu*cond[k])
                k1 = (C[k+1] * prop_layer + np.tanh(prop_layer * thick[k]))
                k2 = ((C[k+1] * prop_layer * np.tanh(prop_layer * thick[k])) + 1)
                C[k] = (1 / prop_layer) * (k1 / k2)
        Z[freq] = 1j * w * mu * C[0]

    rhoa = 1/omega*np.abs(Z)**2
    phi = np.angle(Z, deg=True)
    return rhoa, phi, np.real(Z), np.imag(Z)
