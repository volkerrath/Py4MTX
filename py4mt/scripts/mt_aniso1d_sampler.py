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
import emcee


PY4MTX_DATA = os.environ['PY4MTX_DATA']
PY4MTX_ROOT = os.environ['PY4MTX_ROOT']

mypath = [PY4MTX_ROOT+'/py4mt/modules/', PY4MTX_ROOT+'/py4mt/scripts/']
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)

from aniso import aniso1d_impedance_sens

from data_viz import add_phase, add_rho, add_tipper, add_pt
from data_proc import load_edi, save_edi, save_ncd, save_hdf, save_npz
from data_proc import compute_pt, interpolate_data
from data_proc import set_errors, estimate_errors, rotate_data

import viz
import util as utl
from util import dict_to_namespace
from version import versionstrg

# from mcmc_funcs import pack_model, unpack_model


rhoair = 1.e17
rng = np.random.default_rng()
nan = np.nan  # float('NaN')
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())

titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng+'\n\n')

PY4MTX_DATA = os.environ['PY4MTX_DATA']
PY4MTX_ROOT = os.environ['PY4MTX_ROOT']

WorkDir = '/home/vrath/MT_Data/waldim/edi_jc/'
# WorkDir = PY4MTX_ROOT+'/aniso/'
if not os.path.isdir(WorkDir):
    print(' File: %s does not exist, but will be created' % WorkDir)
    os.mkdir(WorkDir)


EdiDir = WorkDir
ResDir = EdiDir

pi = np.pi
mu0 = 4e-7 * pi


SearchStrng = '*.edi'
dat_files = utl.get_filelist(searchstr=[SearchStrng], searchpath=EdiDir,
                             sortedlist=True, fullpath=True)

dimlist = []
sit = 0
for filename in dat_files:
    sit = sit + 1
    print('reading data from: ' + filename)
    name, ext = os.path.splitext(filename)
    if 'npz' in ext.lower():
        data_dict = np.load(filename)
    elif 'edi' in ext.lower():
        data_dict = load_edi(filename, prefer_spectra=True, err_kind="var")
        P, P_err = compute_pt(data_dict["Z"], data_dict.get("Z_err"), err_kind=data_dict.get("err_kind", "var"))
        data_dict["P"] = P
        data_dict["P_err"] = P_err
        print(np.shape(data_dict))
        save_npz(path=filename.replace('.edi', '.npz'), data_dict=data_dict)

    # utl.stop()
    # edi: Dict[str, Any] = {
    #     "freq": freq,
    #     "Z": Z,
    #     "T": T,
    #     "Z_err": Z_err,
    #     "T_err": T_err,
    #     "P": None,
    #     "P_err": None,
    #     "rot": rot,
    #     "err_kind": err_kind,
    #     "header_raw": header_lines,
    #     "source_kind": source_kind,
    #     # metadata
    #     "station": meta.get("station"),
    #     "lat_deg": meta.get("lat_deg"),
    #     "lon_deg": meta.get("lon_deg"),
    #     "elev_m": meta.get("elev_m"),
    #     # convenience aliases
    #     "lat": meta.get("lat_deg"),
    #     "lon": meta.get("lon_deg"),
    #     "elev": meta.get("elev_m"),
    # }

    site = data_dict['station']
    lat = data_dict['lat']
    lon = data_dict['lon']
    elev = data_dict['elev']

    freq =  data_dict['freq']
    imp = data_dict['Z']
    imp_err =  data_dict['Z_err']
    pht =  data_dict['P']
    pht_err = data_dict['P_err']

    print(' site %s at :  % 10.6f % 10.6f % 8.1f' % (site, lat, lon, elev ))
    print(np.shape(imp))
    #
    imp = imp.ravel()
    imp_err = imp_err.ravel()

    pht = pht.ravel()
    pht_err = pht_err.ravel()
