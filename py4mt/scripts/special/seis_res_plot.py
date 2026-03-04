#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Sat Jan 11 13:48:20 2025

@author: sbyrd
'''

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


'''

Visualizes  statistics for Seismicity-Resistivity correlation


@author: Svetlana Byrdina & Volker Rath,  Jan 2025


'''

import os
import sys

import time
from datetime import datetime
import csv
import inspect

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


JACOPYAN_DATA = os.environ['JACOPYAN_DATA']
JACOPYAN_ROOT = os.environ['JACOPYAN_ROOT']

mypath = [JACOPYAN_ROOT+'/modules/', JACOPYAN_ROOT+'/scripts/']
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)

#import modules
import util as utl
from version import versionstrg


rng = np.random.default_rng()
nan = np.nan  # float('NaN')

version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=inspect.getfile(inspect.currentframe()), out=False)
print(titstrng+'\n\n')


SeisFile = r'/home/sbyrd/Desktop/PEROU/MT_Profil/STAT/Catalog_new_shallow.csv'
ModFile = r'/home/sbyrd/Desktop/PEROU/MT_Profil/STAT/TACG_Z2_Alpha02_NLCG_016'


temp = np.load(SeisFile.replace('.csv','.npz'))
num_out = temp['num_out']
rho_out = temp['rho_out']

index = np.where(np.isfinite(rho_out))
res0 = rho_out[index]
num_seis0 = num_out[index]

indv = np.argsort(res0)

res1 = res0[indv]
num_seis1 = num_seis0[indv]

res_int = np.linspace(-1.,4.,42)

width = 0.1 #0.8*5./27.


num_bin = np.zeros(len(res_int)-1,dtype=int)
res_bin = np.zeros(len(res_int)-1)
for ibin in np.arange(len(res_int)-2):
    ind1 = np.where( (res1 > res_int[ibin]) & (res1 <= res_int[ibin+1]))
    num_bin[ibin] = np.sum(num_seis1[ind1])
    res_bin[ibin] = res_int[ibin] #np.mean(res1[ind1])



#creating a dictionary
font = {'size': 16}
# using rc function
plt.rc('font', **font)




fig, ax = plt.subplots()
plt.bar(res_bin, num_bin, width=width,  edgecolor='white', linewidth=0.7)
#plt.stairs(num_bin, res_int, linewidth=2)


plt.ylabel('Number of events')
plt.xlabel('Log10 resistivity')

plt.grid('on')
plt.savefig(SeisFile.replace('.csv','.png'), dpi=600, bbox_inches='tight')
