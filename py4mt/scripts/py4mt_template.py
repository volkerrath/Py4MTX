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
from pathlib import Path

import time
from datetime import datetime
import warnings
import csv
import inspect
import argparse

# Import numerical or other specialized modules
import numpy as np
import pandas as pd


PY4MTX_DATA = os.environ['PY4MTX_DATA']
PY4MTX_ROOT = os.environ['PY4MTX_ROOT']

# add py4mt modules to pythonpath
for _base in [PY4MTX_ROOT + "/py4mt/modules/"]:
    for _p in [Path(_base), *Path(_base).rglob("*")]:
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

# Import required py4mt modules for your script
import util as utl
from version import versionstrg

import data_proc as mp
import plotrjmcmc as plmc
import viz
import inverse as inv
import femtic as fem
import femtic_viz as femviz


rng = np.random.default_rng()
nan = np.nan  # float('NaN')
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())
titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng+'\n\n')
