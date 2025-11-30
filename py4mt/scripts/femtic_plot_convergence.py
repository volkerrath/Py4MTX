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

WorkDir = r'/home/vrath/FEMTIC_work/krafla6big_L2_L_curve/'
PlotName  = r'Krafla_L2_Convergence'


# os.chdir(EnsembleDir)
SearchStrng = 'kra*'
dir_list = utl.get_filelist(searchstr=[SearchStrng], searchpath=WorkDir,
                            sortedlist =True, fullpath=True)


for directory in dir_list:
    convergence = []
    with open(directory+'/femtic.cnv') as cnv:
        content = cnv.readlines()
        for line in content:

            if '#' in line:
                continue
            #print (line)
            nline = line.split()
            #print(nline)
            itern = int(nline[0])
            retry = int(nline[1])
            if retry>0:
                itern = itern+retry
            print(itern)
            alpha = float(nline[2])
            rough = float(nline[5])
            misft = float(nline[7])
            nrmse = float(nline[8])
    
            convergence.append([itern, alpha, rough, misft, nrmse])
    
    c = np.array(convergence)
    #print(np.shape(c))
    ind = np.argsort( c[:,0] );
    c_sorted  = c[ind]

    #print(c)
    itern = c[:,0]
    alpha = c[:,1]
    rough = c[:,2]
    misft = c[:,3]
    nrmse = c[:,4]
    
    print('#iter', itern)
    print('#misfit', misft)


    fig, ax = plt.subplots()

    plt.semilogy(itern, misft,
            color='green',
            marker='o',
            linestyle='dashed',
            linewidth=1,
            markersize=7,
            markeredgecolor='red',
            markerfacecolor='white'
            )

    #nrmsformula = r'$\sqrt{N^{-1} \mathbf{C}_d^{-1/2} (\mathbf{d}_{obs}-\mathbf{d}_{calc})_2}$'
    formula = r'$\Vert\mathbf{C}_d^{-1/2} (\mathbf{d}_{obs}-\mathbf{d}_{calc})\Vert_2$'

    plt.title(PlotName+r'   $\alpha$ = '+str(round(alpha[0],2)))
    plt.xlabel(r'iteration',fontsize=18)
    plt.ylabel(r'misfit '+formula,fontsize=18)

    # plt.tick_params(labelsize='x-large')
    plt.grid('on')
    plt.tight_layout()
        
    plt.savefig(WorkDir+PlotName+'_alpha'+str(round(alpha[0],2))+'.pdf')
