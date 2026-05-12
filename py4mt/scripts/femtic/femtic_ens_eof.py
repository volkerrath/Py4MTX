#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute Empirical Orthogonal Functions (EOFs) from a FEMTIC model ensemble.

Reads an ensemble of converged FEMTIC inversion results, computes EOF
decomposition via SVD, and generates truncated/component reconstructions.

@author: vrath

Provenance:
    2025       vrath   Created (as femtic_ensemble_eof.py).
    2026-03-03 Claude  Renamed file to femtic_ens_eof.py;
                       renamed user-set parameters to UPPERCASE.
"""

import os
import sys
from pathlib import Path
import inspect

import numpy as np

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

for _base in [PY4MTX_ROOT + "/py4mt/modules/"]:
    for _p in [Path(_base), *Path(_base).rglob("*")]:
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

import util as utl
import femtic as fem
import ensembles as ens

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
ENSEMBLE_DIR = r"/home/vrath/FEMTIC_work/ens_annecy/"
ENSEMBLE_NAME = "ann_"
ENSEMBLE_FILE = "AnnecyENS"
ENSEMBLE_EOF = "AnnecyEOF"
MESH_FILE = "/mesh.dat"

MIN_RMS = 1.5

GET_COMPONENTS = False
OUT_STRNG = "_comp" if GET_COMPONENTS else "_trunc"

# =============================================================================
#  Read ensemble members
# =============================================================================
SEARCH_STRNG = ENSEMBLE_NAME + "*"
dir_list = utl.get_filelist(
    searchstr=[SEARCH_STRNG], searchpath=ENSEMBLE_DIR,
    sortedlist=True, fullpath=True,
)

ens_num = -1
for directory in dir_list:
    ens_num += 1
    print("\n", directory)
    cnv = directory + "/femtic.cnv"
    if not os.path.isfile(cnv):
        print("file femtic.cnv does not exist!")
        continue

    num_best, nrm_best = fem.get_nrms(directory)
    if nrm_best > MIN_RMS:
        print(nrm_best, "is larger than required minimum:", MIN_RMS)
        continue
    print("min(nrmse) =", nrm_best, "at iteration", num_best)

    mesh_file = directory + MESH_FILE
    if ens_num == 0:
        nodes, conns = fem.read_femtic_mesh(mesh_file)

    modl_file = directory + "/resistivity_block_iter" + str(num_best) + ".dat"
    modl = fem.read_model(model_file=modl_file, model_trans="log10")[2:,]
    modl = modl[:, None]

    if ens_num == 0:
        ensemble = modl
    else:
        ensemble = np.concatenate((ensemble, modl), axis=1)

nens = np.shape(ensemble)[1]

storedict = {"ensemble": ensemble, "nodes": nodes, "conns": conns}
np.savez_compressed(ENSEMBLE_DIR + ENSEMBLE_FILE + ".npz", **storedict)

# =============================================================================
#  Compute EOFs
# =============================================================================
eofs, pcs, w_k, frac, mean = ens.compute_eofs(
    E=ensemble, method="svd", demean=True,
)
print("eofs:", np.shape(eofs))

storedict = {
    "eofs": eofs, "pcs": pcs, "frac": frac, "w_k": w_k, "mean": mean,
    "ensemble": ensemble, "nodes": nodes, "conns": conns,
}
np.savez_compressed(ENSEMBLE_DIR + ENSEMBLE_EOF + ".npz", **storedict)

# =============================================================================
#  Compute samples from truncated EOFs
# =============================================================================
eof_results = []
eof_norms = []

for ie in np.arange(nens):
    if GET_COMPONENTS:
        k0, k1 = ie, ie + 1
    else:
        k0, k1 = 0, ie + 1

    print(k0, k1, nens)
    trunc = ens.eof_sample(
        eofs, pcs, mean, k0=k0, k1=k1,
        method="empirical_diag", return_components=False,
    )
    eof_results.append(trunc)

    frac = ens.eof_captured_fraction_from_pcs(pcs, k0=k0, k1=k1)
    eof_norms.append(frac)
    print(np.shape(eof_results))

enorms = np.array(eof_norms)
print("\nfractions:", enorms)

# =============================================================================
#  Store results into model files
# =============================================================================
tens = len(eof_results)
for ie in np.arange(tens):
    outfile = ENSEMBLE_DIR + ENSEMBLE_EOF + OUT_STRNG + str(ie) + ".npz"
    fem.insert_model(
        template=ENSEMBLE_DIR + "resistivity_block_iter.dat",
        model=eof_results[ie],
        model_file=outfile,
    )
