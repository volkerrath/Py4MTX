#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Thu Oct 23 15:19:01 2025

@author: vrath
'''



import os
import sys
import time
from datetime import datetime
import warnings
import csv
import inspect

from pathlib import Path

# Import numerical or other specialised modules
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.cm as cm
import mpl_toolkits.axes_grid1
import matplotlib.ticker


PY4MTX_DATA = os.environ['PY4MTX_DATA']
PY4MTX_ROOT = os.environ['PY4MTX_ROOT']

mypath = [PY4MTX_ROOT+'/py4mt/modules/', PY4MTX_ROOT+'/py4mt/scripts/']
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)

from aniso import aniso1d_impedance_sens
from data_viz import add_phase, add_rho, add_tipper, add_pt
from data_proc import load_edi, save_edi, save_ncd, save_hdf, save_npz
from data_proc import compute_pt, interpolate_data
from data_proc import set_errors, estimate_errors, rotate_data
from data_proc import calc_rhoa_phas


import mcmc
import mcmc_viz as mv
import util as utl
from version import versionstrg

pi = np.pi
mu0 = 4e-7 * pi
rng = np.random.default_rng()
nan = np.nan  # float('NaN')
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())

titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng+'\n\n')


DataDir = "/home/vrath/Py4MTX/py4mt/data/edi/"
SummDir = DataDir + '/pmc_demetropolis_hfix/'
PlotDir = DataDir +'/plots/'
if not os.path.isdir(PlotDir):
    print(' File: %s does not exist, but will be created' % PlotDir)
    os.mkdir(PlotDir)

PlotFormat = ['.pdf']
NameStrng ="_demetropolis_hfix"



fig, axs = plt.subplots(3, 1, figsize=(6, 10))

SearchStrng = DataDir + "Ann*.npz"   # or "*.npz"
file_list = mcmc.glob_inputs(SearchStrng)
if not file_list:
    sys.exit(f"No matching file found: {SearchStrng}! Exit.")


for f in file_list:
    site = mcmc.load_site(f)
    station = site["station"]
    print(f"--- {station} ---")

    '''
      idata = mcmc.sample_pymc(
          pm_model,
          draws=int(DRAWS),
          tune=int(TUNE),
          chains=int(CHAINS),
          cores=int(CORES),
          step_method=str(STEP_METHOD),
          target_accept=float(TARGET_ACCEPT),
          random_seed=int(RANDOM_SEED) if RANDOM_SEED is not None else None,
          progressbar=bool(PROGRESSBAR),
      )

      nc_path = Path(outdir) / f"{station}_pmc.nc"
      sum_path = Path(outdir) / f"{station}_pmc_summary.npz"
      mcmc.save_idata(idata, nc_path)

      summary = mcmc.build_summary_npz(
          station=station,
          site=site,
          idata=idata,
          spec=spec,
          model0=model0,
          info=info,
          qpairs=QPAIRS,
      )
      mcmc.save_summary_npz(summary, sum_path)

      print(f"Wrote: {nc_path}")
      print(f"Wrote: {sum_path}")

    '''

    s = mv.load_summary_npz(SummDir+station+"_pmc_summary.npz")
    idata = mv.open_idata(SummDir+station+"_pmc.nc")
    mv.plot_theta_trace(axs[0], idata, idx=0, name="theta[0]")
    mv.plot_theta_density(axs[1], idata, idx=0, qpairs=s.get("theta_qpairs"))
    mv.plot_vertical_resistivity(axs[2], s, comp=0, use_log10=True)
    fig.tight_layout()
    for fmt in PlotFormat:
        plt.savefig(PlotDir + station + NameStrng + fmt, dpi=600)

    #fig.show()
# MODEL_NPZ = DATA_DIR + "model0.npz"
