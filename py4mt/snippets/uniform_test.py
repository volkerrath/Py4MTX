#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 09:53:34 2025

@author: vrath
"""
import numpy as np


ndraws = 10
low=0. 
high=4. 
nsamp = 30
trueavg = np.mean([low, high])
print(trueavg, '\n\n')

for n in np.arange(ndraws):
    samples = np.random.uniform(low=low, high=high, size=nsamp)
    avg = np.mean(samples)
    print(avg)


for n in np.arange(nsamp):
    for m in np.arange(ndraws):
        samp = np.random.uniform(low=low, high=high, size=n)
        avg = np.mean(samp)
        print(m, n, avg)
    
