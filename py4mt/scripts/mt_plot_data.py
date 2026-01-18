#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''

Run the randomize-then-optimize (RTO) algorithm:

    for i = 1 : nsamples do
        Draw perturbed data set: d_pert∼ N (d, Cd)
        Draw prior model: m̃ ∼ N (0, 1/mu (LT L)^−1 )
        Solve determistic problem  to get the model m_i
    end

See:

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


Created on Wed Apr 30 16:33:13 2025

@author: vrath
'''

import os
import sys
import shutil

import functools
import inspect
import time
from datetime import datetime
import warnings
import csv
import getpass

'''
specialized toolboxes settings and imports.
'''
# import sklearn as skl
# from sklearn.covariance import empirical_covariance
import scipy.sparse as scs
from scipy.interpolate import make_smoothing_spline

# Import numerical or other specialised modules
import numpy as np
import scipy as sci
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

'''
Py4MTX-specific settings and imports.
'''
PY4MTX_DATA = os.environ['PY4MTX_DATA']
PY4MTX_ROOT = os.environ['PY4MTX_ROOT']

mypath = [PY4MTX_ROOT + '/py4mt/modules/', PY4MTX_ROOT + '/py4mt/scripts/']
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)

# import modules
from version import versionstrg
import util as utl
import femtic as fem

from data_viz import add_phase, add_rho, add_tipper, add_pt
from data_proc import load_edi, save_edi, save_ncd, save_hdf
from data_proc import compute_pt, dataframe_from_arrays, interpolate_data
from data_proc import set_errors, estimate_errors, rotate_data


from util import stop

N_THREADS = '10'
os.environ['OMP_NUM_THREADS'] = N_THREADS
os.environ['OPENBLAS_NUM_THREADS'] = N_THREADS
os.environ['MKL_NUM_THREADS'] = N_THREADS

rng = np.random.default_rng()
nan = np.nan  # float('NaN')
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())

titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng + '\n\n')

'''
Base setup.

'''
DatDir = r'/home/vrath/FEMTIC_work/ens_misti/misti_rto_01/'
PltDir = DatDir + '/plots'

UseEDI = False
UseNPZ = True
UseDAT = False


if UseDAT:
    DatList= [DatDir+'/observation.dat']

elif UseEDI:
    edi_files = utl.get_filelist(searchstr=['.edi'], searchpath=DatDir, sortedlist =True, fullpath=True)
    ns = np.size(edi_files)

    DatList =  ['misti_rto_01/observation.npz']

elif UseNPZ:
    DatList =  ['misti_rto_01/observation.npz']


StrngOut = ''

# The next block determines the graphical output. if _PDFCatalog_ is set, a catalogue
# including all generated figures, named _PDFCatName_ is generated. This option is only
# available if '.pdf' is included in the output file format list (_PlotFmt_).

FilesOnly = False    # for headless plotting.
PltFmt = ['.png', '.pdf']

CatName = PltDir+'Annecy_processed.pdf'
Catalog = True
if '.pdf' in PltFmt:
    pass
else:
    print(' No pdf files generated. No catalog possible!')
    Catalog = False

# This block sets graphical parameters related to the \textit{matplotlib}.
# package. A list of available plotting styles can be found on matplotlib's
# website at https://matplotlib.org/stable/users/explain/customizing.htm, or
# entering the python command
# _print(matplotlib.pyplot.style.available)} in an appropriate_
# window.
#

plt.style.use('seaborn-v0_8-paper')


# For just plotting to files ('headless plotting'), choose the
# cairo backend (eps, pdf, png, jpg...).

if FilesOnly:
    mpl.use('cairo')

if Catalog:
    pdf_list = []
    catalog =mpl.backends.backend_pdf.PdfPages(CatName)

pltargs = {
    'figure.dpi' :  400,
    'axes.linewidth' :  0.5,
    'savefig.facecolor' : 'none',
    'savefig.transparent' : True,
    'savefig.bbox' : 'tight'}

Fontsize = 8
Labelsize = Fontsize
Titlesize = 8
Fontsizes = [Fontsize, Labelsize, Titlesize]

Linewidths= [0.6]
Markersize = 4

ncols = 11
Colors = plt.cm.jet(np.linspace(0,1,ncols))
Grey = 0.7




'''
plot data sets
'''


for site in DatList:
    data = np.load(site)
    station = data['station']

    df = pd.DataFrame(data)


    fig, axs = plt.subplots(3, 2, figsize=(8, 14), sharex=True)

    add_rho(df, comps="xy,yx", ax=axs[0, 0])
    add_phase(df, comps="xy,yx", ax=axs[0, 1])
    add_rho(df, comps="xx,yy", ax=axs[1, 0])
    add_phase(df, comps="xx,yy", ax=axs[1, 1])
    add_tipper(df, ax=axs[2, 0])
    add_pt(df, ax=axs[2, 1])
    fig.suptitle(station)
    # Remove empty axes
    for ax in axs:
        if not ax.lines and not ax.images and not ax.collections:
            fig.delaxes(ax)

    for f in PltFmt:
        plt.savefig(PltDir + station + StrngOut + f, dpi=600)


    if Catalog:
       catalog.savefig(fig)



if Catalog:
    print(pdf_list)
    # viz.make_pdf_catalog(PDFList=pdf_list, FileName=PDFCatName)
    d = catalog.infodict()
    d['Title'] =  CatName
    d['Author'] = getpass.getuser()
    d['CreationDate'] = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    catalog.close()


plt.show()


print('data plots ready!')
