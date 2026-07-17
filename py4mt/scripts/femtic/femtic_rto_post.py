#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Wed Apr 30 16:33:13 2025

Randomize-then-Optimize approach: postprocessing

References:

    Bardsley, J. M.; Solonen, A.; Haario, H. & Laine, M.
        Randomize-Then-Optimize: a Method for Sampling from Posterior
        Distributions in Nonlinear Inverse Problems
        SIAM J. Sci. Comp., 2014, 36, A1895-A1910

    Blatter, D.; Morzfeld, M.; Key, K. & Constable, S.
        Uncertainty quantification for regularized inversion of electromagnetic
        geophysical data. Part I: Motivation and Theory
        Geophysical Journal International, doi:10.1093/gji/ggac241, 2022

    Blatter, D.; Morzfeld, M.; Key, K. & Constable, S.
        Uncertainty quantification for regularized inversion of electromagnetic
        geophysical data – Part II: application in 1-D and 2-D problems
        Geophysical Journal International, , doi:10.1093/gji/ggac242, 2022


@author: vrath

Provenance:
    2025-04-30  vrath   Created.
    2026-03-03  Claude  Renamed user-set parameters to UPPERCASE;
                        generated README.
    2026-05-27  vrath / Claude Sonnet 4.6 (Anthropic)
                        Added femtic_viz import; MESH_FILE / PLOT_QC /
                        PLOT_QC_FILE / PLOT_QC_SLICES / PLOT_QC_* config
                        vars; QC slice plot of best-nRMS member at end of
                        main block (calls fviz.plot_model_slices).
    2026-07-17  Claude Sonnet 5 (Anthropic)
                        scipy.sparse: migrated from legacy matrix to
                        array-equivalent API — scs.csr_matrix(tmp) →
                        scs.csr_array(tmp) when building the sparsified
                        empirical covariance (rto_covs). No functional
                        change.
'''
import os
import sys
from pathlib import Path
import shutil
import numpy as np


import sklearn as skl
import sklearn.covariance
import sklearn.decomposition

import scipy as sc
import scipy.linalg as scl
import scipy.ndimage as sci
import scipy.sparse as scs

PY4MTX_DATA = os.environ['PY4MTX_DATA']
PY4MTX_ROOT = os.environ['PY4MTX_ROOT']

for _base in [PY4MTX_ROOT + "/py4mt/modules/"]:
    for _p in [Path(_base), *Path(_base).rglob("*")]:
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

import femtic as fem
import ensembles as ens
import util as utl
from version import versionstrg

try:
    import femtic_viz as fviz
except ImportError:
    fviz = None



rng = np.random.default_rng()
nan = np.nan  # float('NaN')
version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng+'\n\n')

ENSEMBLE_DIR = r'/home/vrath/work/Ensembles/RTO/'
ENSEMBLE_NAME = 'rto_*'

NRMS_MAX = 1.4
# PERCENTILES = numpy.array([10., 20., 30., 40., 50., 60., 70., 80., 90.]) # linear
PERCENTILES = [2.3, 15.9, 50., 84.1,97.7]                   # 95/68
ENSEMBLE_RESULTS = ENSEMBLE_DIR+'RTO_results.npz'

SPARSIFY = True
SPARSE_THRESH = 1.e-8

# ---------------------------------------------------------------------------
# QC slice plot — best-nRMS member (optional)
# ---------------------------------------------------------------------------
#: Mesh file required for slicing.
MESH_FILE = ENSEMBLE_DIR + "templates/mesh.dat"

#: Set True to produce a QC slice figure of the best-converged member.
PLOT_QC = False

#: Output path for the QC figure.  None → interactive show().
PLOT_QC_FILE = ENSEMBLE_DIR + "rto_qc.pdf"

#: Figure DPI.
PLOT_QC_DPI = 200

PLOT_QC_SLICES = [
    dict(kind="map", z0=5000.0),
    dict(kind="map", z0=15000.0),
    dict(kind="ns",  x0=0.0),
    dict(kind="ew",  y0=0.0),
]
PLOT_QC_CMAP        = "turbo_r"
PLOT_QC_CLIM        = [0.0, 4.0]   # log10(Ω·m); None = auto
PLOT_QC_XLIM        = None          # [xmin, xmax] model-local metres; None = auto
PLOT_QC_YLIM        = None
PLOT_QC_ZLIM        = None
PLOT_QC_OCEAN_COLOR = "lightgrey"
PLOT_QC_OCEAN_RHO   = 0.25

dir_list = utl.get_filelist(
    searchstr=[ENSEMBLE_NAME],
    searchpath=ENSEMBLE_DIR,
    fullpath=True)


model_list = []
model_count = -1
for dir in dir_list:
    print('\nInversion run',dir)
    cnv_file = dir+'/femtic.cnv'
    if not os.path.isfile(cnv_file):
        print(cnv_file, 'not found, run skipped.')
        continue

    with open(cnv_file) as file:
        cnv = file.readlines()
    info = cnv[-1].split()
    numit = int(info[0])
    nrms = float(info[8])


    if nrms > NRMS_MAX:
        print(dir,'nRMS =',nrms)
        print(dir,'not converged, run skipped.')
        continue
    model_count = model_count+1
    mod_file = dir+'/resistivity_block_iter'+str(numit)+'.dat'
    print( mod_file, ':')
    print(numit, nrms)
    model_list.append([mod_file,numit, nrms])

    model = fem.read_model(model_file=mod_file, model_trans='log10', out=True)

    if model_count==0:
        rto_ens = model
    else:
        rto_ens = np.vstack((rto_ens, model))

rto_cov = sklearn.covariance.empirical_covariance(rto_ens)

if SPARSIFY:
    tmp = rto_cov.copy()
    maxval = np.amax(tmp)
    tmp[np.abs(tmp)/np.amax(tmp) <= SPARSE_THRESH] = 0.
    rto_covs = scs.csr_array(tmp)



ne = np.shape(rto_ens)
rto_avg = np.mean(rto_ens, axis=1)
# rto_std = np.std(rto_ens, axis=1)
rto_var = np.var(rto_ens, axis=1)
rto_med = np.median(rto_ens, axis=1)
# print(np.shape(rto_ens), np.shape(rto_med))
# print(ne)
rto_mad = np.median(
    np.abs(rto_ens.T - np.tile(rto_med, (ne[1], 1))))
rto_prc = np.percentile(rto_ens, PERCENTILES)

rto_dict ={'model_list' : model_list,
    'rto_ens' : rto_ens,
    'rto_cov' : rto_cov,
    'rto_avg' : rto_avg,
    'rto_var' : rto_var,
    'rto_med' : rto_med,
    'rto_mad' : rto_mad,
    'rto_prc' : rto_prc}

np.savez_compressed(ENSEMBLE_RESULTS, **rto_dict)

# ---------------------------------------------------------------------------
# QC slice plot — best-nRMS converged member
# ---------------------------------------------------------------------------
if PLOT_QC:
    if fviz is None:
        print("  PLOT_QC: femtic_viz not available — skipping.")
    elif not model_list:
        print("  PLOT_QC: no converged members — skipping.")
    else:
        # Pick member with lowest nRMS
        _best = min(model_list, key=lambda x: x[2])
        _best_file, _best_iter, _best_nrms = _best
        print(f"  QC: plotting best member  nRMS={_best_nrms:.3f}  "
              f"iter={_best_iter}  {_best_file}")
        fviz.plot_model_slices(
            model_file  = _best_file,
            mesh_file   = MESH_FILE,
            slices      = PLOT_QC_SLICES,
            cmap        = PLOT_QC_CMAP,
            clim        = PLOT_QC_CLIM,
            xlim        = PLOT_QC_XLIM,
            ylim        = PLOT_QC_YLIM,
            zlim        = PLOT_QC_ZLIM,
            ocean_color = PLOT_QC_OCEAN_COLOR,
            ocean_value = PLOT_QC_OCEAN_RHO,
            plot_file   = PLOT_QC_FILE,
            dpi         = PLOT_QC_DPI,
            out         = True,
        )
        print(f"  QC: saved → {PLOT_QC_FILE}")
