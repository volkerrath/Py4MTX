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
#       jupytext_version: 1.11.3
# ---

"""
Created on Thu Jul 15 10:59:54 2021

@author: vrath
"""


import os
import sys
import warnings
import time

from sys import exit as error
from datetime import datetime

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

PY4MT_ROOT = os.environ["PY4MT_ROOT"]
mypath = [PY4MT_ROOT+"/py4mt/modules/", PY4MT_ROOT+"/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)


import modem as mod
import util as utl
from version import versionstrg

Strng, _ = versionstrg()
now = datetime.now()
print("\n\n"+Strng)
print("Plot Catalog MT data "+"\n"+"".join("Date " + now.strftime("%m/%d/%Y, %H:%M:%S")))
print("\n\n")

warnings.simplefilter(action="ignore", category=FutureWarning)

WorkDir =  r"/home/vrath/work/MT/Annecy/ANN26/"
PredFile = r"/home/vrath/work/MT/Annecy/ANN26/Ann26_ZoPT_200_Alpha04_NLCG_017"
ObsvFile = r"/home/vrath/work/MT/Annecy/ANN26/Ann26_ZoPT"
PlotDir = WorkDir + 'Plots/'


PlotPred = True
if PredFile == "":
    PlotPred = False
PlotObsv = True
if ObsvFile == "":
    PlotObsv = False
