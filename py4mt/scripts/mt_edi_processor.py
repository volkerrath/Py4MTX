#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Thu May 29 10:09:45 2025

@author: vrath
'''

# Import python modules
# edit according to your needs
import os
import sys

import time
from datetime import datetime
import warnings
import csv
import inspect

# Import numerical or other specialised modules
import numpy as np
import scipy as sci

PY4MTX_DATA = os.environ['PY4MTX_DATA']
PY4MTX_ROOT = os.environ['PY4MTX_ROOT']

# add py4mt modules to pythonpath
mypath = [PY4MTX_ROOT+'/py4mt/modules/',
          PY4MTX_ROOT+'/py4mt/scripts/']
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)
        
# Import required py4mt modules for your script
import util as utl
import modem as mod
#import jacproc as jac
import mtproc as mtp
#import plotrjmcmc as plmc
import viz
import inverse as inv
import femtic as fem

import mtviz_funcs as mtv
import mtio_funcs as mtio
#import cluster as fcm
from version import versionstrg


rng = np.random.default_rng()
nan = np.nan  # float('NaN')
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())
titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng+'\n\n')

WorkDir = '/home/vrath/ChatGPT_tests/'
if not os.path.isdir(WorkDir):
    print(' File: %s does not exist, but will be created' % WorkDir)
    os.mkdir(WorkDir)
    
EdiDir =  WorkDir # +'/edi/'
edi_files = mtp.get_edi_list(EdiDir, fullpath=False)
ns = np.size(edi_files)

String_out = ''
Declination = 0. # 2.68   #E
DecDeg = True