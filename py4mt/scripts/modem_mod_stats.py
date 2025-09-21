#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Reads ModEM'smodel files, does cellwise stats on them.

    Percentiles represent the area under the normal curve, increasing from left to right. Each
    standard deviation represents a fixed percentile. Thus, rounding to two decimal places,
    −3σ is the 0.13th percentile,
    −2σ the 2.28th percentile,
    −1σ the 15.87th percentile,
    0σ  the 50th percentile (median of the distribution),
    +1σ the 84.13th percentile,
    +2σ the 97.72nd percentile,
    +3σ the 99.87th percentile.

@author: vr +sb Nov 2024

'''
from version import versionstrg
import util as utl
import modem as mod
import os
import sys

import time
from datetime import datetime
import warnings
import inspect


import numpy as np
import netCDF4 as nc
import scipy.ndimage as spn
import scipy.linalg as spl

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages



PY4MTX_ROOT = os.environ['PY4MTX_ROOT']
mypath = [PY4MTX_ROOT+'/py4mt/modules/', PY4MTX_ROOT+'/py4mt/scripts/']
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)


rng = np.random.default_rng()
blank = np.nan
rhoair = 17.

version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=inspect.getfile(
    inspect.currentframe()), out=False)
print(titstrng+'\n\n')

# PY4MTX_DATA =os.environ['PY4MTX_DATA']

# PY4MTX_DATA = fullpath you like..


hpads = 10
vpads = 50
OutFormat = 'modem'
ModHist = True
if ModHist:
    PlotFormat = ['png', '.pdf']


# WorkDir = PY4MTX_DATA+'/Annecy/Jacobians/'
# WorkDir = PY4MTX_DATA+'/Peru/Sabancaya//SABA8_Jac/'
WorkDir = '/home/vrath/work/MT_Data/Sabancaya/Saba_best/SABA13a/'

if not WorkDir.endswith('/'):
    WorkDir = WorkDir+'/'

# MFile = WorkDir + 'SABA8_best.rho'
MFile = WorkDir + 'SABA13a'

Models = [MFile]


PDFCatalog= False
if len(Models)>1:
    ModFileEns = 'ModEns'
    InfoFileEns = 'ModEns'
    PlotFmt = ['.png']
    DPI = 600

    if '.pdf' in PlotFmt:   
        PDFCatalog = True
        PDFCatalogName  = 'ModEnsCatalog.pdf'
        pdf_list = []   
        catalog = PdfPages(PDFCatalogName)
    else:
        PDFCatalog= False
        print('No PDF catalog because no pdf output!')     
        
# total = 0
# start = time.time()

imod = -1
for f in Models:
    imod += 1
    dx, dy, dz, rho, ref, trans = mod.read_mod(
        file=f,
        modext='.rho',
        trans='LOG10',
        blank=1.e-30,
        out=True)
    dims = np.shape(rho)
    aircells = np.where(rho > rhoair)

    print(f + '.rho'+', shape is', dims)
    rtmp = rho.ravel()
    print(np.shape(rtmp))
    if imod == 0:

        ModEns = rtmp
    else:
        ModEns = np.vstack((ModEns, rtmp))

    dims = np.shape(rho)
    sdims = np.size(rho)

    aircells = np.where(rho > rhoair-1.)
    rho[aircells] = blank

    rho[:hpads, :, :] = blank
    rho[dims[0]-hpads:dims[0], :, :] = blank

    rho[:, :hpads, :] = blank
    rho[:, dims[1]-hpads:dims[1], :,] = blank

    rho[:, :, dims[2]-vpads:dims[2]] = blank

    print('\n\n')
    rhoavg = np.nanmean(rho)
    print('Mean resistivity is', np.power(10., rhoavg))
    print('Mean log resistivity is', rhoavg)
    rhostd = np.sqrt(np.nanvar(rho))
    print('Std log resistivity is', rhostd)

    print('\n\n')
    rhomed = np.nanmedian(rho)
    print('Median resistivity is', np.power(10., rhomed))
    print('Median log resistivity is', rhomed)
    rhoquant = np.nanquantile(rho, [0.16, 0.84])
    print('1-sigma quantiles:', rhoquant)
    rhoquant = np.nanquantile(rho, [0.023, 0.977])
    print('2-sigma quantiles:', rhoquant)
    print('\n\n')
    print()

    if ModHist:
        
        rtmp = rho.ravel()
        rtmp = rtmp[np.isfinite(rtmp)]
        #rtmp = np.power(10.,rtmp)
        fig, ax = plt.subplots()
        counts, bins = np.histogram(rtmp, bins=51,range=(-1,4))
        #plt.bar(counts, bins)
        plt.stairs(counts, bins, fill=True)
        plt.xlabel(r'log resistivity $\Omega m$')
        plt.ylabel(r'counts')
        plt.grid('on')
        plt.title(os.path.splitext(os.path.basename(f))[0])
        plt.tight_layout()

        pname = os.path.splitext(f)[0]
        for fmt in PlotFormat:
            plt.savefig(pname+fmt)


        if PDFCatalog:
            pdf_list.append(pname+'.pdf')
            catalog.savefig(pname+'.pdf')
    
if len(Models) > 1:
    
    ModAvg = np.mean(ModEns, axis=1).reshape(dims)
    ModVar = np.var(ModEns, axis=1).reshape(dims)
    ModMed = np.median(ModEns, axis=1).reshape(dims)

    ModQnt = [np.percentile(ModEns, 15.9, axis=1).reshape(dims),
              np.percentile(ModEns, 50., axis=1).reshape(dims),
              np.percentile(ModEns, 84.1, axis=1).reshape(dims)]

    if 'mod' in OutFormat.lower():
        # for modem_readable files

        mod.write_mod(ModFileEns, modext='_avg.rho',
                      dx=dx, dy=dy, dz=dz, mval=ModAvg,
                      reference=ref,
                      mvalair=blank,
                      aircells=aircells,
                      header='Model log-average')
        print('Averages (ModEM format) written to '+ModFileEns+'_avg.rho')
        mod.write_mod(ModFileEns, modext='_var.rho',
                      dx=dx, dy=dy, dz=dz, mval=np.sqrt(ModVar),
                      reference=ref, mvalair=blank, aircells=aircells, header='Model log-std')
        print('Variances (ModEM format) written to '+ModFileEns+'_var.rho')

    # if 'ubc' in OutFormat.lower():
        # elev = -ref[2]
        # refubc =  [MOrig[0], MOrig[1], elev]
        # mod.write_ubc(ModFileAvg, modext='_ubc.avg', mshext='_ubc.msh',
        # dx=dx, dy=dy, dz=dz, mval=vol, reference=refubc, mvalair=Blank, aircells=aircells, header='Model log-average')
        # print(' Cell volumes (UBC format) written to '+VolFile)

    if 'rlm' in OutFormat.lower():
        mod.write_rlm(ModFileEns, modext='_avg.rlm',
                      dx=dx, dy=dy, dz=dz, mval=ModAvg, reference=ref, mvalair=blank, aircells=aircells, comment='Model log-average')
        print(' Averages (CGG format) written to '+ModFileEns+'_avg.rlm')
        mod.write_rlm(ModFileEns, modext='_avg.rlm',
                      dx=dx, dy=dy, dz=dz, mval=np.sqrt(ModVar), reference=ref, mvalair=blank, aircells=aircells, comment='Model log-std')
        print(' Variances (CGG format) written to '+ModFileEns+'_var.rlm')


# ModVar = np.zeros()
# write_model(ModFile_out+'.rho', dx, dy, dz, rho,reference,out = True)
if PDFCatalog:
        catalog.close()