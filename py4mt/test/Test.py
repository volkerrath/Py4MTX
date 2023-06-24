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
#       jupytext_version: 1.8.0
# ---

"""
Created on Wed Jul 29 17:24:03 2020

@author: vrath
"""
import time
import numpy as np
from modules.ModEM import read_model, write_model


ModFile_inp = r"./work/AnnPriorZTP.rho"


start = time.time()
dx, dy,dz, rho1, center = read_model(ModFile_inp, trans="LINEAR",out = True)
print (' min (rho) = %7.4g , max(rho) %7.4g ' % (np.min(rho1),np.max(rho1)))
# dx, dy,dz, rho2, center = read_model(ModFile_inp, trans="LOGE",out = True)
# print (' min (rho) = %7.4g , max(rho) %7.4g ' % (np.min(rho2),np.max(rho2)))
# dx, dy,dz, rho3, center = read_model(ModFile_inp, trans="LOG10",out = True)
# print (' min (rho) = %7.4g , max(rho) %7.4g ' % (np.min(rho3),np.max(rho3)))
elapsed  = time.time()
print (' Used %7.4f s for reading model from %s ' % (elapsed-start,ModFile_inp))

ModFile_out = r"./work/AnnPriorZTP_log10.rho"
write_model(ModFile_out, dx, dy, dz, rho1, center, trans="LOG10",out = True)
ModFile_out = r"./work/AnnPriorZTP_loge.rho"
write_model(ModFile_out, dx, dy, dz, rho1, center, trans="LOGE",out = True)
ModFile_out = r"./work/AnnPriorZTP_linear.rho"
write_model(ModFile_out, dx, dy, dz, rho1, center, trans="LINEAR",out = True)




# print(np.shape(rho))
# d[d<0.8] = 0.0
# scipy.sparse.csr_matrix(d)
