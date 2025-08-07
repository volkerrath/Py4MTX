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

# Import numoerical or other specialsed modules
import numpy as np
from mtpy.core.mt import MT

# addpy4mt modules to pythonpath
mypath = ['/home/vrath/Py4MT/py4mt/modules/',
          '/home/vrath/Py4MT/py4mt/scripts/']
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)
        
# Import required py4mt modules for your script

import util as utl
import modem as mod
import jacproc as jac
import mtproc as proc
import plotrjmcmc as plmc
import viz
import inverse as inv
import femtic as fem
import cluster as fcm
from version import versionstrg


rng = np.random.default_rng()
nan = np.nan  # float('NaN')
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())

titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng+'\n\n')
