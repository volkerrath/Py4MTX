#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Thu May 29 10:09:45 2025

@author: vrath
'''

# Import python modules
# edit according to your needs
import os
import sys

import time
from datetime import datetime
import warnings
import csv
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
from femtic import read_distortion_file


from data_viz import add_phase, add_rho, add_tipper, add_pt
from data_proc import (
    get_edi_list,
    load_edi, save_edi, save_ncd, save_hdf, save_npz,
    save_list_of_dicts_npz, dataframe_from_edi,
    interpolate_data, set_errors, estimate_errors, rotate_data,
    compute_pt, compute_zdet, compute_zssq)

from version import versionstrg


rng = np.random.default_rng()
nan = np.nan  # float('NaN')
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())
titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng + '\n\n')

# WorkDir = '/home/vrath/ChatGPT_tests/'
# WorkDir = '/home/vrath/Py4MTX/work/edis_2025/'
WorkDir = '/home/vrath/Current/Annecy/'


EdiDir = WorkDir +'/edi_files/'
edi_files = get_edi_list(EdiDir, fullpath=True)
ns = np.size(edi_files)

OutDir = WorkDir +'/corrected/'
if not os.path.isdir(OutDir):
    print(' File: %s does not exist, but will be created' % OutDir)
    os.mkdir(OutDir)
OutFiles = 'edi, npz'

Plot = True
if Plot:
    PlotDir = WorkDir +'/plots/'
    if not os.path.isdir(PlotDir):
        print(' File: %s does not exist, but will be created' % PlotDir)
        os.mkdir(PlotDir)
    pltargs = {'show_errors': True}
    PlotFormat = ['.pdf']
# %%

NameStr = '_distcorr' #'_dd'
CollName = 'AnnAll_distcorr'

PhasTens = True
Invars = True

FEMDist_file = 'distortion_iter13.dat'
NamesNumbers_file = 'names_numbers_femtic.csv'

# get translation table

mapping = {}
with open(WorkDir+NamesNumbers_file, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    for num, name in reader:
        mapping[name] = int(num)

print(mapping)

# get distortion matrix (c, not c')
distortion, _ = read_distortion_file(WorkDir+FEMDist_file)


all_data = []
for edi in edi_files:

    edi_dict = load_edi(edi, drop_invalid_periods=True)

    station = edi_dict['station']
    Z = edi_dict['Z']
    edi_dict['Z_orig'] = Z.copy()
    Zerr = edi_dict['Z_err']
    T = edi_dict['T']
    Terr = edi_dict['T_err']

    '''
    Task block
    '''
    sitenum = mapping.get(station)
    C = distortion[sitenum,:,:]
    for f in np.arange(np.shape(Z)[0]):
        Z[f,:,:] = C@Z[f,:,:]
    edi_dict['Z'] = Z

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




    edi_dict['distortion'] = C


    all_data.append(edi_dict)

    # print(np.shape(Z), np.shape(Zerr),
    #       np.shape(T), np.shape(Terr),
    #       np.shape(P), np.shape(Perr))
    # print(list(edi_dict.keys()))


    if 'edi' in OutFiles.lower():
        _ = save_edi(
            path=OutDir + station + NameStr + '.edi',
            edi=edi_dict
        )

    if 'ncd' in OutFiles.lower():
        _ = save_ncd(
            path=OutDir + station + NameStr + '.ncd',
            data_dict=edi_dict)

    if 'hdf' in OutFiles.lower():
        _ = save_hdf(
            path=OutDir + station + NameStr + '.hdf',
            data_dict=edi_dict)

    if 'npz' in OutFiles.lower():
        _ = save_npz(
            path=OutDir + station + NameStr + '.npz',
            data_dict=edi_dict)

    if Plot:
        fig, axs = plt.subplots(3, 2, figsize=(14, 14), sharex=True)
        add_rho(edi_dict, comps="xy,yx", ax=axs[0, 0], **pltargs)
        add_phase(edi_dict, comps="xy,yx", ax=axs[0, 1], **pltargs)
        add_rho(edi_dict, comps="xx,yy", ax=axs[1, 0], **pltargs)
        add_phase(edi_dict, comps="xx,yy", ax=axs[1, 1], **pltargs)
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
            plt.savefig(PlotDir + station + NameStr + f, dpi=600)

        plt.show()


save_list_of_dicts_npz(
    records=all_data,
    path=OutDir + CollName + NameStr + '_collection.npz')
