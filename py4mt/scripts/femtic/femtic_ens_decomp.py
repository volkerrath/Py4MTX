#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate PCA/ICA decomposition for a FEMTIC model ensemble.

Reads resistivity models from an ensemble of FEMTIC inversion runs,
filters by convergence (nRMS threshold), and performs dimensionality
reduction using PCA or ICA (via scikit-learn).

@author: vrath

Provenance:
    2025       vrath   Created (as femtic_decomp_ens.py).
    2026-03-03 Claude  Renamed file to femtic_ens_decomp.py;
                       renamed user-set parameters to UPPERCASE.
"""

import os
import sys
from pathlib import Path
import inspect

import numpy as np
import sklearn.decomposition

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

for _base in [PY4MTX_ROOT + "/py4mt/modules/"]:
    for _p in [Path(_base), *Path(_base).rglob("*")]:
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

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
ENSEMBLE_DIR = r"/home/vrath/work/Ensembles/RTO/"
ENSEMBLE_NAME = "rto_*"
NRMS_MAX = 1.4

PROC = "pca"  # Options: 'pca', 'increment', 'ica'
PERCENTILES = [2.3, 15.9, 50.0, 84.1, 97.7]  # 2-sigma / 1-sigma bounds
ENSEMBLE_RESULTS = ENSEMBLE_DIR + "PCA.npz"

# =============================================================================
#  Collect ensemble members
# =============================================================================
dir_list = utl.get_filelist(
    searchstr=[ENSEMBLE_NAME], searchpath=ENSEMBLE_DIR, fullpath=True
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

    if nrms > NRMS_MAX:
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
proc_lower = PROC.lower()

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

np.savez_compressed(ENSEMBLE_RESULTS, **results_dict)
print(f"\nResults saved to {ENSEMBLE_RESULTS}")
