#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 17:01:14 2024

@author: vrath
"""

#!/usr/bin/env python3

"""
Reads ModEM's Jacobian, does fancy things.

@author: vrath   Feb 2021

"""

# Import required modules

import os
# from https://stackoverflow.com/questions/73391779/setting-number-of-threads-in-python
# nthreads = 8  # tinney  62
# os.environ["OMP_NUM_THREADS"] = str(nthreads)
# os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
# os.environ["MKL_NUM_THREADS"] = str(nthreads)

import sys

# import struct
import time
from datetime import datetime
import warnings


import jax.numpy as nj
import jax.scipy as sj

import numpy as np

# from sklearn.utils.extmath import randomized_svd
# from numba import njit

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [PY4MTX_ROOT+"/py4mt/modules/", PY4MTX_ROOT+"/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)


import util as utl
from version import versionstrg


# RunParallel = False
# if RunParallel:
#     pth = PY4MTX_ROOT+"/external/PyParSVD/pyparsvd/"
#     if pth not in sys.path:
#         sys.path.insert(0,pth)




version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")

rng = np.random.default_rng()
nan = np.nan

DatDir_in = PY4MTX_DATA + "/Fogo/"
DatDir_out = DatDir_in 

if not os.path.isdir(DatDir_out):
    print("File: %s does not exist, but will be created" % DatDir_out)
    os.mkdir(DatDir_out)
    
    
DatFiles_in = ["FOG_Z_in.dat", "FOG_P_in.dat", "FOG_T_in.dat"] 


PerIntervals = [ [0.0001, 0.001], 
              [0.001, 0.01], 
              [0.01, 0.1], 
              [0.1, 1.], 
              [1., 10.], 
              [10., 100.], 
              [100., 1000.], 
              [1000., 10000.],
              10000., 1000000.,]

PerNumMin = 1

NumBands = len(PerIntervals)

    
for datfile in DatFiles_in:
    
    for ibnd in np.arange(NumBands):
        
        lowstr=str(1./PerIntervals[ibnd][0])+"Hz"            
        uppstr=str(1./PerIntervals[ibnd][1])+"Hz"                   
        
        
        with open(DatDir_in+datfile) as fd:
            head = []
            data = []
            site = []
            perd = []
            for line in fd:
                if line.startswith("#") or line.startswith(">"):
                    head.append(line)
                    continue
                

                per = float(line.split()[0])
                sit = line.split()[1]
                if (per>=PerIntervals[ibnd][0]) & (per<PerIntervals[ibnd][1]):
                   data.append(line)
                   site.append(sit) 
                   perd.append(per)
        
        nper = len(np.unique(perd))
        nsit = len(np.unique(site))
        print(nper, "periods from",nsit,"sites")
        if nper>0 and nsit>0:
            phead = head.copy()
            phead = [lins.replace("per", str(nper)) for lins in phead]
            phead = [lins.replace("sit", str(nsit)) for lins in phead]
            outfile = DatDir_in+datfile
            outfile = outfile.replace("_in.dat", "_perband"+str(ibnd)+".dat")
            print("ouput to", outfile)
            fo = open(outfile,"w")
            for ilin in np.arange(len(phead)):
                fo.write(phead[ilin])
            for ilin in np.arange(len(data)):
                fo.write(data[ilin])
            fo.close()
        
                   
                
