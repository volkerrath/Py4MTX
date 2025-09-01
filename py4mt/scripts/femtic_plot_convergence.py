#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''

Lcurve 

vr July 2025


Created on Wed Apr 30 16:33:13 2025

@author: vrath
'''
import os
import sys
import shutil
import numpy as np
import functools
import inspect

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PY4MTX_DATA = os.environ['PY4MTX_DATA']
PY4MTX_ROOT = os.environ['PY4MTX_ROOT']

mypath = [PY4MTX_ROOT+'/py4mt/modules/', PY4MTX_ROOT+'/py4mt/scripts/']
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)

#import modules
import femtic as fem
import util as utl
from version import versionstrg



rng = np.random.default_rng()
nan = np.nan  # float('NaN')
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())

titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng+'\n\n')

WorkDir = r'/home/vrath/FEMTIC_work/Misti_lcurve/'

WhatPlot = 'nrms'
PlotName  = WorkDir+'Misti'+'_nRMS'



# os.chdir(EnsembleDir)
SearchStrng = 'lc_'
dir_list = utl.get_filelist(searchstr=[SearchStrng], searchpath=WorkDir, 
                            sortedlist =True, fullpath=True)

# dir_list = []

for directory in dir_list:
    convergence = []
    with open(directory+'femtic.cnv') as cnv:
        content = cnv.readlines()
        for line in content:
            nline = line.split()
            itern = int(line(0))
            alpha = float(line[2])
            rough = float(line[5])
            misft = float(line[7])
            nrmse = float(line[8])
    
            convergence.append = [itern, alpha, rough, misft, nrmse]
    
    c = np.array(convergence).reshape((5,-1))
    itern = c[0,:]
    alpha = c[1,:]
    rough = c[2,:]
    misft = c[3,:]
    nrmse = c[4,:]
    
    
    if 'nrms' in WhatPlot.lower():
        fig, ax = plt.subplots()
        
        plt.loglog(itern, nrmse, 
                 color='green', 
                 marker='o', 
                 linestyle='dashed',
                 linewidth=1, 
                 markersize=7,
                 markeredgecolor='red',
                 markerfacecolor='white'
                 )
        
        # for k in np.arange(len(i)):
        #     alph = round(a[k], -int(np.floor(np.log10(abs(a[k])))))
        #     plt.annotate(str(alph),[m[k],r[k]])
        
        # misftformula = '$\parallel\mathbf{C}_d^{-1/2} (\mathbf{d}_{obs}-\mathbf{d}_{calc})\parallel_2$'
        # roughformula = '$\parallel\mathbf{C}_m^{-1/2} \mathbf{m}\parallel_2$'
        nrmsformula = '$\sqrt{1/N \mathbf{C}_d^{-1/2} (\mathbf{d}_{obs}-\mathbf{d}_{calc})_2}$'
        
        
        plt.xlabel(r'iteration',fontsize=18)  
        plt.ylabel(r'nRMS'+nrmsformula,fontsize=18)
        
        # plt.tick_params(labelsize='x-large')
        plt.grid('on')
        plt.tight_layout()
        
        plt.savefig(PlotName+'.pdf')
        plt.savefig(PlotName+'.png')
