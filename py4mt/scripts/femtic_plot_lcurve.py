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
PlotName  = WorkDir+'Misti'+'_L-Curve'

# os.chdir(EnsembleDir)
SearchStrng = 'lc_*'
dir_list = utl.get_filelist(searchstr=[SearchStrng], searchpath=WorkDir, 
                            sortedlist =True, fullpath=True)

PlotWhat = 'nrms'
FontLabel = 16


l_curve=[]
for directory in dir_list:
    with open(directory+'/'+'femtic.cnv') as cnv:
        content=cnv.readlines()

    line = content[-1].split()
    print(line)
    alpha = float(line[2])
    rough = float(line[5])
    misft = float(line[7])
    nrmse = float(line[8])
    
    l_curve.append([alpha, rough, misft, nrmse ])
  
 
lc = np.array(l_curve).reshape((-1,4))
ind = np.argsort( lc[:,0] ); 
lc_sorted  = lc[ind]
lc_sorted = np.delete(lc_sorted, 0, 0)
#lc_sorted = np.sort(lc, axis=1)

a = lc_sorted[:,0]
r = lc_sorted[:,1]
m = lc_sorted[:,2]
n = lc_sorted[:,3]

fig, ax = plt.subplots()

if 'nrms' in PlotWhat.lower():

    plt.plot(n, r, 
             color='green', 
             marker='o', 
             linestyle='dashed',
             linewidth=1, 
             markersize=7,
             markeredgecolor='red',
             markerfacecolor='white'
             )
    
    for k in np.arange(len(lc_sorted)):
        alph = round(a[k], -int(np.floor(np.log10(abs(a[k])))))
        plt.annotate(str(alph),[n[k],r[k]])
        
    
    
    xformula = '$nRMS$'
    plt.xlabel(r'misfit '+xformula,fontsize=FontLabel)
    
    yformula = '$\parallel\mathbf{C}_m^{-1/2} \mathbf{m}\parallel_2$'
    plt.ylabel(r'roughness '+yformula,fontsize=FontLabel)

# plt.tick_params(labelsize='x-large')
    plt.grid('on')
    plt.tight_layout()
    
    plt.savefig(PlotName+'.pdf')
    plt.savefig(PlotName+'.png')

else:
    
    plt.plot(m, r, 
             color='green', 
             marker='o', 
             linestyle='dashed',
             linewidth=1, 
             markersize=7,
             markeredgecolor='red',
             markerfacecolor='white'
             )
    
    for k in np.arange(len(lc_sorted)):
        alph = round(a[k], -int(np.floor(np.log10(abs(a[k])))))
        plt.annotate(str(alph),[n[k],r[k]])
        
    
    
    xformula = '$\parallel\mathbf{C}_d^{-1/2} (\mathbf{d}_{obs}-\mathbf{d}_{calc})\parallel_2$'
    plt.xlabel(r'misfit '+xformula,fontsize=FontLabel)
    
    yformula = '$\parallel\mathbf{C}_m^{-1/2} \mathbf{m}\parallel_2$'
    plt.ylabel(r'roughness '+yformula,fontsize=FontLabel)

# plt.tick_params(labelsize='x-large')
    plt.grid('on')
    plt.tight_layout()
    
    plt.savefig(PlotName+'.pdf')
    plt.savefig(PlotName+'.png')