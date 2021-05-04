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


DataDir =   r"/home/vrath/Py4MT/py4mt/MIMDAS/"
DataFile = DataDir+r"Block1.dat"
ModemDir = DataDir
ModemFile = ModemDir+r"Block_ModEM.dat"
PlotDir = DataDir+"Plots/"
PlotFile = "Block1"
PlotFormat = [".pdf", ".png", ".svg"]

print(' Plots written to: %s' % PlotDir)
if not os.path.isdir(PlotDir):
    print(' File: %s does not exist, but will be created' % PlotDir)
    os.mkdir(PlotDir)


NSample =9
SeedSample= None #110652
MinDist =(300., 100.)
d_margin = 0.01
RanMeth = "con"

Freq =np.array([range(1, 104, 2)])*25./256
Freq_single = [Freq[0,0]]
# Freq_dec2 = Freq[range(0, 52, 2)]
# print(np.shape(Freq_dec2))
# Freq_dec3 = Freq[range(0, 52, 3)]
# print(np.shape(Freq_dec3))
# Freq_dec4 = Freq[range(0, 52, 4)]
# print(np.shape(Freq_dec4))
# Freq_dec5 = Freq[range(0, 52, 5)]
# print(np.shape(Freq_dec5))

# FR = np.array(Freq_single)
FR = Freq_single
print("Frequencies (Hz):")
print(FR)
# print("Periods (s):")
# print(1./FR)
# nFR=np.shape(Freq_single)
# print(nFR)

"""
Set graphics parameter
"""
# print(plt.style.available)
plt.style.use("seaborn-paper")
mpl.rcParams["figure.dpi"] = 400
mpl.rcParams["axes.linewidth"] = 0.5
Fontsize = 10
Labelsize = Fontsize
Linewidth= 2
Markersize = 4
Grey = 0.7
Lcycle  = (cycler("linestyle", ["-", "--", ":", "-."])
          * cycler("color", ["r", "g", "b", "y"]))
Ccycle =  cycler("color", ["r", "g", "b", "y"])
cm = 1/2.54  # centimeters in inches

"""
Get reference point UTM coordinates
"""
Lat = -34.173965
Lon = 148.737549
EPSG,_ = utl.get_utm_list(Lat, Lon)
UTMx, UTMy = utl.proj_latlon_to_utm(Lat, Lon, utm_zone=EPSG)
print ("\n")
print ("Reference Point (WGS84):  "+str(Lat)+"   "+str(Lon))
print ("Reference Point (UTM)  :  "+str(np.around(UTMx,1))+"   "+str(np.around(UTMy,1)))
print ("\n")

"""
Read data
"""
D, H = mim.read_csem_data(DataFile)
nD = np.shape(D)

D[:,0:4]=D[:,0:4]-UTMx
D[:,4:8]=D[:,4:8]-UTMy

RXx = 0.5*(D[:,2]+D[:,3])
RXy = 0.5*(D[:,6]+D[:,7])
RXz = 5.
maxRXx = np.max(RXx)
minRXx = np.min(RXx)
maxRXy = np.max(RXy)
minRXy = np.min(RXy)
print("Rx area:   "+str(np.around(minRXx,1))+" - "+str(np.around(maxRXx,1))
      +" / "
      +str(np.around(minRXy,1))+" - "+str(np.around(maxRXy,1)))



TXx = D[:,0]
TXy = D[:,4]
TXz = 5.
maxTXx = np.max(TXx)
minTXx = np.min(TXx)
maxTXy = np.max(TXy)
minTXy = np.min(TXy)

print("Tx area:   "+str(np.around(minTXx,1))+" - "+str(np.around(maxTXx,1))
      +" / "
      +str(np.around(minTXy,1))+" - "+str(np.around(maxTXy,1)))

Re =D[0,15::2]
Im =D[0,16::2]

"""
generate transmitter subset

"""
if RanMeth[0].lower() == "c":
    Ind_s, TXx_s, TXy_s = mim.get_randomTX_constr(TXx,TXy,
                            Nsamples=NSample,
                            Seedsamples=SeedSample,
                            Mindist=MinDist)
else:
    Ind_s, TXx_s, TXy_s = mim.get_randomTX_simple(TXx, TXy,
                            Nsamples=NSample,
                            Seedsamples=SeedSample)


fig, ax = plt.subplots() #figsize = (16*cm, 16*cm))
ax.scatter(RXx,RXy, s=(Markersize+2)**2, c ="k")
ax.scatter(TXx,TXy, s=Markersize**2, c ="b")
# ax.scatter(xtest, ytest, s=Markersize**2, c="g", marker="+")
ax.scatter(TXx_s,TXy_s,s=(Markersize+4)**2, c="r", marker="x")
ax.legend(["RX", "TX","TxR"])
ax.tick_params(labelsize=Labelsize-1)
ax.set_ylabel("UTM$_y$", fontsize=Fontsize)
ax.set_xlabel("UTM$_x$", fontsize=Fontsize)
ax.grid("major", "both", linestyle="-", linewidth=0.5)
ax.axis('equal')

for F in PlotFormat:
        plt.savefig(PlotDir+PlotFile+F, dpi=400)

plt.show()
plt.close(fig)


nFR = np.shape(FR)[0]
nRX = np.shape(RXx)[0]
nTX = np.shape(TXx_s)[0]

for ifreq in np.arange(nFR):

    for irx in np.arange(nRX):

        for itx in np.arange(nTX):
            continue

