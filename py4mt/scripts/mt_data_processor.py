#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Thu May 29 10:09:45 2025

@author: vrath

Author: Volker Rath (DIAS)
Created with the help of ChatGPT (GPT-5 Thinking) on 2026-02-13 (UTC)
'''

# Import python modules
# edit according to your needs
import os
import sys
import inspect

# Import numerical or other specialised modules
import numpy as np
import matplotlib.pyplot as plt

PY4MTX_DATA = os.environ['PY4MTX_DATA']
PY4MTX_ROOT = os.environ['PY4MTX_ROOT']

# add py4mt modules to pythonpath
mypath = [PY4MTX_ROOT + '/py4mt/modules/',
          PY4MTX_ROOT + '/py4mt/scripts/']
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)

# Import required py4mt modules for your script
import util as utl
# import jacproc as jac
# import mt_proc as mtp
# import plotrjmcmc as plmc
# import viz
# import inverse as inv
from data_viz import add_phase, add_rho, add_tipper, add_pt
from data_proc import (
    get_edi_list,
    load_edi, save_edi, save_ncd, save_hdf, save_npz,
    save_list_of_dicts_npz, dataframe_from_edi,
    interpolate_data, set_errors, estimate_errors, rotate_data,
    compute_pt, compute_zdet, compute_zssq, compute_rhophas)

from version import versionstrg


rng = np.random.default_rng()
nan = np.nan  # float('NaN')
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())
titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng + '\n\n')

# WorkDir = '/home/vrath/ChatGPT_tests/'
# WorkDir = '/home/vrath/Py4MTX/work/edis_2025/'
WorkDir =  "/home/vrath/Py4MTX/py4mt/data/edi/"

if not os.path.isdir(WorkDir):
    print(' File: %s does not exist, but will be created' % WorkDir)
    os.mkdir(WorkDir)

DataDir = WorkDir
EdiDir = WorkDir # +'/orig/'
edi_files = get_edi_list(EdiDir, fullpath=True)
ns = np.size(edi_files)


OutFiles = 'edi, npz'

Plot = False
if Plot:
    pltargs = {'show_errors': True}
    PlotFormat = ['.png', '.pdf']
# %%
NameStr = '' #'_dd'
CollName = 'ANN3_aniso'

SetErrors = False
Errors = {'Zerr': [0.1, 0.1, 0.1, 0.1],
          'Terr': [0.03, 0.03, 0.03, 0.03],
          'PTerr': [0.1, 0.1, 0.1, 0.1]
          }

PhasTens = True
Invars = True

Interpolate = False
if Interpolate:
    FreqPerDec = 6
    IntMethod = [None, FreqPerDec]

EstimateErrors = False
if EstimateErrors:
    sys.exit('Work in progress! Exit')
    spread = 2.  # *std-dev
    ErrMethod = ['gcvspline', spread]


Rotate = False
if Rotate:
    Angle = 0.  # 2.68   #E
    DecDeg = True

all_data = []
for edi in edi_files:

    edi_dict = load_edi(edi, drop_invalid_periods=True)

    station = edi_dict['station']
    Z = edi_dict['Z']
    Zerr = edi_dict['Z_err']
    T = edi_dict['T']
    Terr = edi_dict['T_err']

    '''
    Task block
    '''

    if PhasTens:
        P, Perr = compute_pt(Z, Zerr)
        edi_dict['P'] = P

        edi_dict['P_err'] = Perr

    if Invars:

        Zdet, Zdeterr = compute_zdet(Z, Zerr)
        edi_dict['Zdet'] = Zdet
        edi_dict['Zdet_err'] = Zdeterr
        Zssq, Zssqerr = compute_zssq(Z, Zerr)
        edi_dict['Zssq'] = Zssq
        edi_dict['Zssq_err'] = Zssqerr

    if EstimateErrors:
        edi_dict = estimate_errors(edi_dict=edi_dict, method=ErrMethod)

    if SetErrors:
        edi_dict = set_errors(edi_dict=edi_dict, errors=Errors)

    if Interpolate:
        edi_dict = interpolate_data(edi_dict=edi_dict, method=IntMethod)

    if Rotate:
        edi_dict = rotate_data(edi_dict=edi_dict, angle=Angle)

    # --- refresh apparent resistivity/phase after any Z modification (new rhophas treatment)
    if edi_dict.get('freq') is not None and edi_dict.get('Z') is not None:
        _ek = str(edi_dict.get('err_kind', 'var')).strip().lower()
        _ek = 'std' if _ek.startswith('std') else 'var'
        rho, phi, rho_err, phi_err = compute_rhophas(
            freq=np.asarray(edi_dict['freq']),
            Z=np.asarray(edi_dict['Z']),
            Z_err=np.asarray(edi_dict['Z_err']) if edi_dict.get('Z_err') is not None else None,
            err_kind=_ek,
            err_method='analytic',
        )
        edi_dict['rho'] = rho
        edi_dict['phi'] = phi
        if rho_err is not None:
            edi_dict['rho_err'] = rho_err
        if phi_err is not None:
            edi_dict['phi_err'] = phi_err

    all_data.append(edi_dict)

    # print(np.shape(Z), np.shape(Zerr),
    #       np.shape(T), np.shape(Terr),
    #       np.shape(P), np.shape(Perr))
    # print(list(edi_dict.keys()))


    if 'edi' in OutFiles.lower():
        _ = save_edi(
            path=DataDir + station + NameStr + '.edi',
            edi=edi_dict
        )

    if 'ncd' in OutFiles.lower():
        _ = save_ncd(
            path=DataDir + station + NameStr + '.ncd',
            data_dict=edi_dict)

    if 'hdf' in OutFiles.lower():
        _ = save_hdf(
            path=DataDir + station + NameStr + '.hdf',
            data_dict=edi_dict)

    if 'npz' in OutFiles.lower():
        _ = save_npz(
            path=DataDir + station + NameStr + '.npz',
            data_dict=edi_dict)

    if Plot:
        fig, axs = plt.subplots(3, 2, figsize=(8, 14), sharex=True)
        # Use dataframe_from_edi so precomputed rho/phi are honoured (new rhophas treatment)
        df_rp = dataframe_from_edi(edi_dict, include_tipper=False, include_pt=False)
        add_rho(df_rp, comps="xy,yx", ax=axs[0, 0], **pltargs)
        add_phase(df_rp, comps="xy,yx", ax=axs[0, 1], **pltargs)
        add_rho(df_rp, comps="xx,yy", ax=axs[1, 0], **pltargs)
        add_phase(df_rp, comps="xx,yy", ax=axs[1, 1], **pltargs)
        add_tipper(edi_dict, ax=axs[2, 0], **pltargs)
        add_pt(edi_dict, ax=axs[2, 1], **pltargs)
        fig.suptitle(station + NameStr.replace('_', ' | '))

        # for ax in np.atleast_1d(axs).ravel():
        # Remove empty axes
        for ax in axs.flat:
            if not ax.lines and not ax.images and not ax.collections:
                fig.delaxes(ax)

        fig.tight_layout(rect=[0, 0, 1, 0.97])

        for f in PlotFormat:
            plt.savefig(WorkDir + station + NameStr + f, dpi=600)

        plt.show()


save_list_of_dicts_npz(
    records=all_data,
    path=DataDir + CollName + NameStr + '_collection.npz')
