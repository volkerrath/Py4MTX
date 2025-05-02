#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 16:59:10 2025

@author: vrath
"""

import numpy as np
import matplotlib.pyplot as plt


rho = [1.e-4, 1.e-3, 1.e-2, 1.e-1, 1.]
per = [1.e-3, 1.e-2, 1.e-1, 1., 10, 100., 1000] # period in s

count = 0
for r in rho:
    tmp = np.array([500*np.sqrt(r*T) for T in per]) #skin depth
    
    if count==0:
        d = tmp
    else:
        d = np.vstack((d, tmp))
    count = count+1


plt.loglog(per,d.T)
plt.grid ()
rlabel = [str(np.log10(r)) for r in rho]
plt.legend(rlabel, title='log10(rho) in $\Omega$m' )
plt.xlabel('period in sec')
plt.ylabel('induction scale/skin depth in m')
plt.savefig('skindepth_logrho.png')