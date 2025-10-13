#!/usr/bin/env python3

'''
Get static shofts from FEMTIC - aply to ModEM

@author: vrath   Oct 2025

'''

# Import required modules

import os
# from https://stackoverflow.com/questions/73391779/setting-number-of-threads-in-python
# nthreads = 8  # tinney  62
# os.environ['OMP_NUM_THREADS'] = str(nthreads)
# os.environ['OPENBLAS_NUM_THREADS'] = str(nthreads)
# os.environ['MKL_NUM_THREADS'] = str(nthreads)

import sys
import inspect

# import struct
import time
from datetime import datetime
import warnings


import jax.numpy as nj
import jax.scipy as sj

import numpy as np

# from sklearn.utils.extmath import randomized_svd
# from numba import njit

PY4MTX_DATA = os.environ['PY4MTX_DATA']
PY4MTX_ROOT = os.environ['PY4MTX_ROOT']

mypath = [PY4MTX_ROOT+'/py4mt/modules/', PY4MTX_ROOT+'/py4mt/scripts/']
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)


import util as utl
from version import versionstrg


version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=inspect.getfile(inspect.currentframe()), out=False)
print(titstrng+'\n\n')

rng = np.random.default_rng()
nan = np.nan

DataDir_in = PY4MTX_DATA + '/Fogo/'
DataDir_out = DataDir_in 

if not os.path.isdir(DataDir_out):
    print('File: %s does not exist, but will be created' % DataDir_out)
    os.mkdir(DataDir_out)
    
    
DataFile = 'FOG_Z_in.dat' 
SiteFile = ''
DistFile = ''
    
   
with open(DataDir_in+DataFile) as fd:
      head = []
      data = []
      site = []
      perd = []
      for line in fd:
          if line.startswith('#') or line.startswith('>'):
              head.append(line)
              continue
          
          l = line.split()
          per = float(l[0])
          sit = l[1]

          data.append(line)
          site.append(sit) 
          perd.append(per)
  
      nper = len(np.unique(perd))
      nsit = len(np.unique(site))
      print(nper, 'periods from',nsit,'sites')
      if nper>0 and nsit>0:
          phead = head.copy()
          phead = [lins.replace('per', str(nper)) for lins in phead]
          phead = [lins.replace('sit', str(nsit)) for lins in phead]
          
          
outfile = DataDir_in+DataFile
outfile = outfile.replace('_in.dat', '_distcor.dat')
print('ouput to', outfile)
with open(outfile,'w') as fo:
    for ilin in np.arange(len(phead)):
        fo.write(phead[ilin])
    for ilin in np.arange(len(data)):
        fo.write(data[ilin])

            
                   
                
