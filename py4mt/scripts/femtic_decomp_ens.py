#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate PCA/ICA decomposition for a FEMTIC model ensemble.

Reads resistivity models from an ensemble of FEMTIC inversion runs,
filters by convergence (nRMS threshold), and performs dimensionality
reduction using PCA or ICA (via scikit-learn).

@author: vrath
"""

import os
import sys
import inspect

import numpy as np
import sklearn.decomposition

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
EnsembleDir = r"/home/vrath/work/Ensembles/RTO/"
EnsembleName = "rto_*"
NRMSmax = 1.4

Proc = "pca"  # Options: 'pca', 'increment', 'ica'
Percentiles = [2.3, 15.9, 50.0, 84.1, 97.7]  # 2-sigma / 1-sigma bounds
EnsembleResults = EnsembleDir + "PCA.npz"

# =============================================================================
#  Collect ensemble members
# =============================================================================
dir_list = utl.get_filelist(
    searchstr=[EnsembleName], searchpath=EnsembleDir, fullpath=True
)

model_list = []
model_count = -1

for directory in dir_list:
    print("\nInversion run", directory)
    cnv_file = directory + "/femtic.cnv"
    if not os.path.isfile(cnv_file):
        print(cnv_file, "not found, run skipped.")
        continue

    with open(cnv_file) as file:
        cnv = file.readlines()

    info = cnv[-1].split()
    numit = int(info[0])
    nrms = float(info[8])

    if nrms > NRMSmax:
        print(directory, "nRMS =", nrms, "- not converged, run skipped.")
        continue

    model_count += 1
    mod_file = directory + "/resistivity_block_iter" + str(numit) + ".dat"
    print(mod_file, ":")
    print(numit, nrms)
    model_list.append([mod_file, numit, nrms])

    model = fem.read_model(model_file=mod_file, model_trans="log10", out=True)

    if model_count == 0:
        ensemble = model
    else:
        ensemble = np.vstack((ensemble, model))

results_dict = {"model_list": model_list, "ensemble": ensemble}

# =============================================================================
#  Decomposition
# =============================================================================
proc_lower = Proc.lower()

if "pca" in proc_lower or "increment" in proc_lower:
    for ipca in np.arange(1, model_count):
        pca = sklearn.decomposition.PCA(n_components=ipca)
        pca.fit(ensemble)
        print(f"\n{ipca} explained variance:")
        print(pca.explained_variance_ratio_)
        print(f"{ipca} cumulative explained variance:")
        print(np.cumsum(pca.explained_variance_ratio_))
        print(f"{ipca} singular values:")
        print(pca.singular_values_)

elif "ica" in proc_lower:
    # BUG FIX: original used undefined `ipca` variable
    for n_comp in np.arange(1, model_count):
        ica = sklearn.decomposition.FastICA(n_components=n_comp)
        ica.fit(ensemble)

np.savez_compressed(EnsembleResults, **results_dict)
print(f"\nResults saved to {EnsembleResults}")
