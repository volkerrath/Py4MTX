# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
# ---

import os
import sys
import warnings
import time

from sys import exit as error
from datetime import datetime

import numpy as np
import scipy as sc
import scipy.ndimage as sci
import scipy.linalg as scl
import fnmatch

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
import pyproj as proj

mypath = ["/home/vrath/Py4MT/py4mt/modules/",
          "/home/vrath/Py4MT/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)

import modem as mod
import util as utl
import mimdas as mim
from version import versionstrg

warnings.simplefilter(action="ignore", category=FutureWarning)

Strng, _ = versionstrg()
now = datetime.now()
print("\n"+Strng)
print("Read and transform MIMDAS CSEM data"+"\n"+"".join("Date " + now.strftime("%m/%d/%Y, %H:%M:%S")))
print("\n")



# Graphical paramter. Determine the plot formats produced,
# and the required resolution:
DDir =   r"/home/vrath/Py4MT/py4mt/MIMDAS/"
DFile = DDir+r"/Block1.dat"
MDir = DDir
MFile = MDir+r"/Block_ModEM.dat"


Freq =np.array(range(1, 104, 2))*25./256
Freq_dec2 = Freq[range(0, 52, 2)]
print(np.shape(Freq_dec2))
Freq_dec3 = Freq[range(0, 52, 3)]
print(np.shape(Freq_dec3))
Freq_dec4 = Freq[range(0, 52, 4)]
print(np.shape(Freq_dec4))

"""
Get reference point UTM coordinates
"""
Lat = -34.173965
Lon = 148.737549
EPSG,_ = utl.get_utm_list(Lat, Lon)
UTMx, UTMy = utl.proj_latlon_to_utm(Lat, Lon, utm_zone=EPSG)
print ("Reference Point (WGS84):  "+str(Lat)+"   "+str(Lon))
print ("Reference Point (UTM)  :  "+str(np.around(UTMx,1))+"   "+str(np.around(UTMy,1)))
print ("\n")

"""
Read data
"""
D, H = mim.read_csem_data(DFile)
nD = np.shape(D)

D[:,0:4]=D[:,0:4]-UTMx
D[:,4:8]=D[:,4:8]-UTMy

RXx = 0.5*(D[:,2]+D[:,3])
RXy = 0.5*(D[:,6]+D[:,7])
RXz = 5.

TXx = D[:,0]
TXy = D[:,4]
TXz = 5.










nRx = nD[0]
# for irx in np.arange(nRx):

#     # print(irx)
