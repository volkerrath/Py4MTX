#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare directories and data subsets for a jackknife-inspired uncertainty analysis.

Creates sub-directories with template files and generates reduced data sets
(e.g., leave-one-site-out) for FEMTIC inversion runs.

@author: vrath
"""

import os
import sys
import inspect

import numpy as np

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [PY4MTX_ROOT + "/py4mt/modules/", PY4MTX_ROOT + "/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)

import femtic as fem
import util as utl
from version import versionstrg

rng = np.random.default_rng()
nan = np.nan
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())
titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng + "\n\n")

# =============================================================================
#  Configuration
# =============================================================================
EnsembleDir = r"/home/vrath/work/Ensemble/Ubinas_ens/"
Templates = EnsembleDir + "templates/"
Files = [
    "control.dat",
    "observe.dat",
    "mesh.dat",
    "resistivity_block_iter0.dat",
    "distortion_iter0.dat",
    "run_femtic_dub.sh",
    "run_femtic_oar.sh",
]

ChoiceMode = ["site"]

# Read site count from control.dat when using site-based jackknife
if "site" in ChoiceMode:
    with open(Templates + "control.dat", "r") as file:
        content = file.readlines()
    tmp = content[0].split()
    N_samples = int(tmp[0])  # number of sites determines jackknife sample count

# Alternative: random subset mode
# N_samples = 32
# ChoiceMode = ['subset', N_samples]

# =============================================================================
#  Generate directories
# =============================================================================
os.chdir(EnsembleDir)

dir_list = fem.generate_directories(
    dir_base=EnsembleDir + "jcn_",
    templates=Templates,
    file_list=Files,
    N_samples=N_samples,
    out=True,
)

# =============================================================================
#  Draw reduced data sets based on sites
# =============================================================================
data_ensemble = fem.generate_data_fcn(
    dir_base=EnsembleDir + "ens_",
    N_samples=N_samples,
    file_in="observe.dat",
    choice_mode=ChoiceMode,
    out=True,
)
