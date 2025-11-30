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
PlotWhat = 'rough'

# os.chdir(EnsembleDir)
SearchStrng = 'kra*'
dir_list = utl.get_filelist(searchstr=[SearchStrng], searchpath=WorkDir,
                            sortedlist =True, fullpath=True)


for directory in dir_list:
    convergence = []
    iteration = -1
    with open(directory+'/femtic.cnv') as cnv:
        content = cnv.readlines()
        for line in content:

            if '#' in line:
                continue
            iteration = iteration +1
            #print (line)
            nline = line.split()
            #print(nline)
            itern = int(nline[0])
            retry = int(nline[1])
            if retry>0:
                itern = itern+retry
            # print(itern)
            alpha = float(nline[2])
            rough = float(nline[5])
            misft = float(nline[7])
            nrmse = float(nline[8])
    
            convergence.append([iteration, alpha, rough, misft, nrmse])

    if len(convergence)==0:
        print (directory, '/femtic.cnv', ' is empty!')
        continue

    c = np.array(convergence)
    print(np.shape(c))
    #ind = np.argsort( c[:,0] );
    #c_sorted  = c  #[ind]

    #print(c)
    itern = c[:,0]
    alpha = c[:,1]
    rough = c[:,2]
    misft = c[:,3]
    nrmse = c[:,4]
    
    print('#iter', itern)
    print('#misfit', misft)
    print('#nrmse', nrmse)
    print('#rough', rough)




    fig, ax = plt.subplots()

    if 'mis' in PlotWhat.lower():
        print('plotting misfit')
        conv = misft
        formula = r'$\Vert\mathbf{C}_d^{-1/2} (\mathbf{d}_{obs}-\mathbf{d}_{calc})\Vert_2$'
        plt.semilogy(itern, conv,
            color='green',
            marker='o',
            linestyle='dashed',
            linewidth=1,
            markersize=7,
            markeredgecolor='red',
            markerfacecolor='white'
            )
        plt.title(PlotName+r'   $\alpha$ = '+str(round(alpha[0],2)))
        plt.xlabel(r'iteration',fontsize=14)
        plt.ylabel(r'misfit '+formula,fontsize=14)
        # plt.tick_params(labelsize='x-large')
        plt.grid('on')
        plt.tight_layout()

    elif 'rms' in PlotWhat.lower():
        print('plotting nrmse')
        conv = nrmse
        formula = r'$\sqrt{N^{-1} \mathbf{C}_d^{-1/2} (\mathbf{d}_{obs}-\mathbf{d}_{calc})_2}$'
        plt.plot(itern, conv,
                color='green',
                marker='o',
                linestyle='dashed',
                linewidth=1,
                markersize=7,
                markeredgecolor='red',
                markerfacecolor='white'
                )
        plt.title(PlotName.replace('_',' ')+r' |   $\alpha$ = '+str(round(alpha[0],2)))
        plt.xlabel(r'iteration',fontsize=14)
        plt.ylabel(r'nRMS '+formula,fontsize=14)
        # plt.tick_params(labelsize='x-large')
        plt.grid('on')
        plt.tight_layout()

    elif 'rough' in PlotWhat.lower():
        print('plotting roughness')
        conv = rough
        formula = r'$\Vert\mathbf{C}_m^{-1/2} \mathbf{m}\Vert_2$'
        plt.semilogy(itern, conv,
                color='green',
                marker='o',
                linestyle='dashed',
                linewidth=1,
                markersize=7,
                markeredgecolor='red',
                markerfacecolor='white'
                )
        plt.title(PlotName.replace('_',' ')+r' |   $\alpha$ = '+str(round(alpha[0],2)))
        plt.xlabel(r'iteration',fontsize=14)
        plt.ylabel(r'roughness '+formula,fontsize=14)
        # plt.tick_params(labelsize='x-large')
        plt.grid('on')
        plt.tight_layout()


    else:
        sys.exit('plot_convergence: plotting parameter',PlotWhat.lower(),'not implemented! Exit.')


    plt.savefig(WorkDir+PlotName+'_'+PlotWhat.lower()+'_alpha'+str(round(alpha[0],2))+'.pdf')
