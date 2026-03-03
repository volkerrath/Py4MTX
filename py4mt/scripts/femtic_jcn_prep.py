#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare directories and data subsets for a jackknife-inspired uncertainty analysis.

Creates sub-directories with template files and generates reduced data sets
(e.g., leave-one-site-out) for FEMTIC inversion runs.

@author:   vrath
@project:  py4mt — Python for Magnetotellurics
@inversion: FEMTIC
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
ENSEMBLE_DIR = r"/home/vrath/work/Ensemble/Ubinas_ens/"
TEMPLATES = ENSEMBLE_DIR + "templates/"
FILES = [
    "control.dat",
    "observe.dat",
    "mesh.dat",
    "resistivity_block_iter0.dat",
    "distortion_iter0.dat",
    "run_femtic_dub.sh",
    "run_femtic_oar.sh",
]

CHOICE_MODE = ["site"]

# Read site count from control.dat when using site-based jackknife
if "site" in CHOICE_MODE:
    with open(TEMPLATES + "control.dat", "r") as file:
        content = file.readlines()
    tmp = content[0].split()
    N_SAMPLES = int(tmp[0])  # number of sites determines jackknife sample count

# Alternative: random subset mode
# N_SAMPLES = 32
# CHOICE_MODE = ["subset", N_SAMPLES]

# =============================================================================
#  Generate directories
# =============================================================================
os.chdir(ENSEMBLE_DIR)

dir_list = fem.generate_directories(
    dir_base=ENSEMBLE_DIR + "jcn_",
    templates=TEMPLATES,
    file_list=FILES,
    N_samples=N_SAMPLES,
    out=True,
)

# =============================================================================
#  Draw reduced data sets based on sites
# =============================================================================
data_ensemble = fem.generate_data_fcn(
    dir_base=ENSEMBLE_DIR + "ens_",
    N_samples=N_SAMPLES,
    file_in="observe.dat",
    choice_mode=CHOICE_MODE,
    out=True,
)
