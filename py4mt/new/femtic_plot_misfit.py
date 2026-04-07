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
import argparse

# Import numerical or other specialized modules
import numpy as np
import pandas as pd


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
from version import versionstrg

import femtic_data as fd


# import data_proc as mp
# import plotrjmcmc as plmc
# import viz
# import inverse as inv
# import femtic as fem
# import femtic_viz as femviz


rng = np.random.default_rng()
nan = np.nan  # float('NaN')
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())
titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng+'\n\n')


# =============================================================================
#  Configuration
INPUT_DIR = PY4MTX_ROOT + "py4mt/data/edi/"
# WORK_DIR = "/home/vrath/MT_Data/waldim/edi_synth_iso/"
if not os.path.isdir(INPUT_DIR):
   sys.exit(" File: %s does not exist! Exit." % INPUT_DIR)

TRUEERROR = "eee"
NAME = "vvv"
CSV = False
UNDIST = False 


global read_true_error_file, true_error_file_name
global output_csv, impedance_converted_to_app_res

iteration_number = int(sys.argv[1])
num_pe           = int(sys.argv[2])

read_result(iteration_number, num_pe)

i = 3
while i < len(sys.argv):
    arg = sys.argv[i]
    if arg == "-name":
        read_relation_site_id_to_name(sys.argv[i + 1])
        i += 2
    elif arg == "-err":
        true_error_file_name = sys.argv[i + 1]
        read_true_error_file = True
        i += 2
    elif arg == "-csv":
        output_csv = True
        i += 1
    elif arg == "-undist":
        read_control_data("control.dat")
        read_distortion_matrix(iteration_number)
        i += 1
    elif arg == "-appphs":
        print("Impedance tensors are converted to apparent resistivity and phase.")
        impedance_converted_to_app_res = True
        i += 1
    else:
        i += 1

calc_true_rms(read_true_error_file, true_error_file_name)
write_result()


if __name__ == "__main__":
    main()
