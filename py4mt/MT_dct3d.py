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
#       jupytext_version: 1.10.3
# ---

"""
Read ModEM model, ModEM's Jacobian.

Model order reduction by discreste (co)sinus transform.

@author: vrath Feb 2021

"""
import os
import sys
from sys import exit as error
import time
import warnings

import numpy as np
import math as ma
import netCDF4 as nc
import scipy.ndimage as spn
import scipy.linalg as spl

import vtk
import pyvista as pv
import PVGeo as pvg


mypath = ["/home/vrath/Py4MT/py4mt/modules/", "/home/vrath/Py4MT/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.append(pth)

import modules
import modem as mod
import util as utl


warnings.simplefilter(action="ignore", category=FutureWarning)

