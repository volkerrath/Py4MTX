# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Preparation of the bathymetry, topography, coast_line files needed to create the FEMTIC mesh
#
# Example with the misti South Australia Dataset

import sys
sys.path.append('../../src')
sys.path.append("/home/sbyrd/python_scripts/femticPy/src")
import femticPy

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# ## 1. Set up DataGen object
#  
#  First a femticpy.DataGen object has to be created to load the MT data and coordinates

# the file will be written in the input_data directory
misti_inv = femticPy.DataGen(survey = 'ubinas', outdir = './input_data')

# Loading the data and the data coordinates
misti_inv.read_MTdata('../input_data/Edi_Ubi_edit_interp/')
misti_inv.read_MTdata_coordinates('../input_data', 'Sitelist_utm.csv')   #utm elevation negatives in meters

# we center the data to a anchor point (center of the data set) which will also be the center of the future mesh
misti_inv.center_data()
misti_inv.anchor


# -


