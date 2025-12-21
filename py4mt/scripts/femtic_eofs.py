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
import argparse

# Import numerical or other specialised modules
import numpy as np
import scipy as sc

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
import modem as mod
import jacproc as jp
import mtproc as mp
import dataproc as dp
import plotrjmcmc as plmc
import viz
import inverse as inv
import femtic as fem
import ensembles as ens
import femtic_viz as femviz
import cluster as fcm
from version import versionstrg


rng = np.random.default_rng()
nan = np.nan  # float('NaN')
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())
titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng + '\n\n')


EnsembleDir = r'/home/vrath/FEMTIC_work/ens_annecy/'
EnsembleName = 'ann_'
EnsembleFile = 'AnnecyEOF.npz'

SearchStrng = EnsembleName + '*'
dir_list = utl.get_filelist(searchstr=[SearchStrng], searchpath=EnsembleDir,
                            sortedlist=True, fullpath=True)

ens_num = -1
for directory in dir_list:
    ens_num = ens_num + 1
    num_best, nrm_best = fem.get_nrms(directory)

    print('\n', directory)
    print('min(nrmse) = ', nrm_best, 'at iteration', num_best)
    mesh_file = directory + '/mesh.dat'
    modl_file = directory + '/resistivity_block_iter' + str(num_best) + '.dat'
    print(modl_file)

    modl = fem.read_model(
        model_file=modl_file,
        model_trans="log10")

    modl = modl[:, None]
    if ens_num == 0:
        ensemble = modl
    else:
        ensemble = np.concatenate((ensemble, modl), axis=1)
    print(np.shape(ensemble))

storedict = {'ensemble': ensemble}
np.savez_compressed(EnsembleDir + EnsembleFile, **storedict)
