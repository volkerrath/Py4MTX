#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
# ---

"""
Created on Wed Nov  4 15:18:34 2020

@author: vrath
"""
import sys
from time import process_time 
from sys import exit as error

import numpy as np

from modem import mt1dfwd


frequency   = [0.0001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,10000]
resistivity = [300, 300, 300, 300, 300]
conductivity = list(1./r for r in resistivity)
print(np.shape(resistivity),np.shape(conductivity))
thickness   = np.array([200, 400, 40, 500])

# resistivity= [300, 2500, 0.8, 3000, 2500];
# conductivity = list(1./r for r in resistivity)
# thickness = [200, 400, 40, 500];

# print(conductivity)

start = process_time()
Z        = mt1dfwd(frequency, conductivity, thickness,inmod="c",out = "imp")
print('Impedance:')
print(Z)   
print('time taken = ', process_time() - start, 's \n')

start = process_time()
Z        = mt1dfwd(frequency, resistivity, thickness,inmod="r",out = "imp")
print('Impedance:')
print(Z)   
print('time taken = ', process_time() - start, 's \n')

start = process_time()    
rho, phas = mt1dfwd(frequency, conductivity, thickness,inmod="c",out = "rho")
print('Rho/Phas:')
print(rho)
print(phas)
print('time taken = ', process_time() - start, 's \n')
