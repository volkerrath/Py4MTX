#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Run the a jackknife-inspired uncertainty algorithm


Created on Wed Apr 30 16:33:13 2025

@author: vrath
"""
import os
import sys
import shutil
import numpy as np
import functools
import inspect


PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [PY4MTX_ROOT+"/py4mt/modules/", PY4MTX_ROOT+"/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)

#import modules
import femtic as fem
import util as utl
from version import versionstrg

import sklearn as skl
from sklearn.covariance import empirical_covariance


rng = np.random.default_rng()
nan = np.nan  # float("NaN")
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())

titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng+"\n\n")


N_samples = 32
EnsembleDir = r'/home/vrath/work/Ensemble/Ubinas_ens/'
Templates = EnsembleDir+'templates/'
Files = ['control.dat',
         'observe.dat',
         'mesh.dat',
         'resistivity_block_iter0.dat',
         'distortion_iter0.dat',
         'run_femtic_dub.sh',
         'run_femtic_oar.sh']

ChoiceMode = ['site', ]
# ChoiceMode = ['subset', 5]


os.chdir(EnsembleDir)

dir_list = fem.generate_directories(
    dir_base=EnsembleDir+'jcn_',
    templates=Templates,
    file_list=Files,
    N_samples=N_samples,
    out=True)


"""
Draw reduced  data sets based on sites
"""

data_ensemble = fem.generate_data_fcn(dir_base=EnsembleDir+'ens_',
                                          N_samples=N_samples,
                                          file_in='observe.dat',
                                          choice_mode=ChoiceMode,
                                          out=True)
