#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate an equivalent model ensemble from a posterior covariance matrix.

Uses Cholesky decomposition of the covariance to sample new models
following the approach of Osypov et al. (2013).

References:
    Osypov, K. et al. (2013): Model-uncertainty quantification in seismic
    tomography: method and applications. Geophysical Prospecting, 61,
    1114-1134, doi: 10.1111/1365-2478.12058

@author: vrath
"""

import os
import sys

import numpy as np
from sksparse.cholmod import cholesky

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
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng + "\n\n")

# =============================================================================
#  Configuration
# =============================================================================
CovarDir = r"/home/vrath/work/Ensembles/RTO/"
CovarResults = CovarDir + "RTO_results.npz"

NewEnsembleSize = 100
NewEnsembleDir = r"/home/vrath/work/Ensembles/OSY/"
NewEnsembleFile = NewEnsembleDir + "OSY_ensemble.npz"

# =============================================================================
#  Load covariance and compute Cholesky factor
# =============================================================================
tmp = np.load(CovarResults)
cov = tmp["rto_cov"]
ref = tmp["rto_avg"]

sqrtcov = cholesky(cov)

# =============================================================================
#  Generate new ensemble after Osypov (2013)
# =============================================================================
model_size = np.shape(ref)[0]
for imod in np.arange(NewEnsembleSize):
    sample = ref + sqrtcov * rng.normal(loc=0.0, scale=1.0, size=model_size)
    if imod == 0:
        new_ens = sample
    else:
        new_ens = np.vstack((new_ens, sample))

ensemble_dict = {"new_ens": new_ens, "sqrtcov": sqrtcov, "ref": ref}
np.savez_compressed(NewEnsembleFile, **ensemble_dict)
print(f"Ensemble ({NewEnsembleSize} members) saved to {NewEnsembleFile}")
