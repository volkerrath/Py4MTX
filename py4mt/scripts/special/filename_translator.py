#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 16:07:10 2026

@author: vrath
"""
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
import matplotlib.pyplot as plt

PY4MTX_DATA = os.environ['PY4MTX_DATA']
PY4MTX_ROOT = os.environ['PY4MTX_ROOT']

# add py4mt modules to pythonpath
mypath = [PY4MTX_ROOT + '/py4mt/modules/',
          PY4MTX_ROOT + '/py4mt/scripts/']
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)

# Import required py4mt modules for your script
import util as utl



WorkDir = '/home/vrath/Current/Annecy/'

NamesNumbers_file = 'names_numbers_femtic.csv'
NamesNumbers_corr = 'names_numbers_corr.csv'

# get translation table

rows = []
with open(WorkDir+NamesNumbers_file, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    for num, name in reader:
        name = name.lower()
        rows.append([name, int(num)+1])

with open(WorkDir+NamesNumbers_corr, 'w', newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(rows)
