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

import os
import sys
import warnings

from sys import exit as error
from datetime import datetime

import numpy as np


import matplotlib as mpl
import matplotlib.pyplot as plt
# from cycler import cycler



mypath = ["/home/vrath/Py4MT/py4mt/modules/",
          "/home/vrath/Py4MT/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)

import util as utl
import mimdas as mim
from version import versionstrg

warnings.simplefilter(action="ignore", category=FutureWarning)

# rans = np.random.default_rng()
RanState= None #110652
RanGen =np.random.PCG64()
rans = np.random.default_rng(RanGen)

Strng, _ = versionstrg()
now = datetime.now()
print("\n"+Strng)
print("Read and transform MIMDAS CSEM data"+"\n"
      +"".join("Date " + now.strftime("%m/%d/%Y, %H:%M:%S")))
print("\n")

ident = str(rans.integers(1,10000)).zfill(5)
DataDir =   r"/home/vrath/Py4MT/py4mt/MIMDAS/"
DataFile = DataDir+r"Block1.dat"
ModemDir = DataDir
ModemFile = ModemDir+r"Block1_ModEM_"+ident+".dat"
ModemName = "B01"
ModemHead = ("# MIMDAS data for Block1"+"\n"
             +"".join("# Date " + now.strftime("%m/%d/%Y, %H:%M:%S"))+"\n"
             +"Identifier: "+ident+"\n")
PlotDir = DataDir+"Plots/"
PlotFile = r"Block1_ModEM_"+ident
PlotFormat = [".pdf", ".png", ".svg"]


print(' Plots written to: %s' % PlotDir)
if not os.path.isdir(PlotDir):
    print(' File: %s does not exist, but will be created' % PlotDir)
    os.mkdir(PlotDir)

"""
Parameters for generating random set of transmitters.
"""
NSample =9
MinDist =(400., 100.)
d_margin = 0.01
RanMeth = "con"

RanStatFile = DataDir+"RanStat"+ident+".npz"
np.savez(RanStatFile, RanState=rans)


"""
Coefficents for error function. Error model including
multiplicative and additive noise, following Brodie (2015).
"""
Err_mul, Err_add = 0.05, 0.

FreqBase =np.array([range(1, 104, 2)])*25./256
Freq_single = [FreqBase[0,0]]
# Freq_dec2 = Freq[range(0, 52, 2)]
# print(np.shape(Freq_dec2))
# Freq_dec3 = Freq[range(0, 52, 3)]
# print(np.shape(Freq_dec3))
# Freq_dec4 = Freq[range(0, 52, 4)]
# print(np.shape(Freq_dec4))
# Freq_dec5 = Freq[range(0, 52, 5)]
# print(np.shape(Freq_dec5))

Freq = Freq_single
print("Frequencies (Hz):")
print(Freq)


"""
Set graphics parameter
"""
# print(plt.style.available)
plt.style.use("seaborn-paper")
mpl.rcParams["figure.dpi"] = 600
mpl.rcParams["axes.linewidth"] = 0.5
Fontsize = 10
Labelsize = Fontsize
Linewidth= 2
Markersize = 4
# Grey = 0.7
# Lcycle  = (cycler("linestyle", ["-", "--", ":", "-."])
#           * cycler("color", ["r", "g", "b", "y"]))
# Ccycle =  cycler("color", ["r", "g", "b", "y"])
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

ReDat =D[:,15::2]
ImDat =D[:,16::2]

"""
generate transmitter subset

"""
if RanMeth[0].lower() == "c":
    Ind_s, TXx_s, TXy_s = mim.get_randomTX_constr(TXx,TXy,
                            Nsamples=NSample,
                            Ranstate=rans,
                            Mindist=MinDist)
else:
    Ind_s, TXx_s, TXy_s = mim.get_randomTX_simple(TXx, TXy,
                            Nsamples=NSample,
                            Ranstate=rans)


fig, ax = plt.subplots() #figsize = (16*cm, 16*cm))
ax.scatter(RXx,RXy, s=(Markersize+2)**2, c ="r")
ax.scatter(TXx,TXy, s=Markersize**2, c ="b")
# ax.scatter(xtest, ytest, s=Markersize**2, c="g", marker="+")
ax.scatter(TXx_s,TXy_s,s=(Markersize+4)**2, c="g", marker="x")
ax.legend(["RX", "TX","TxR"])
ax.set_title("MIMDAS set "+ident)
ax.tick_params(labelsize=Labelsize-1)
ax.set_ylabel("UTM$_y$", fontsize=Fontsize)
ax.set_xlabel("UTM$_x$", fontsize=Fontsize)
ax.grid("major", "both", linestyle="-", linewidth=0.5)
ax.axis('equal')

for F in PlotFormat:
        plt.savefig(PlotDir+PlotFile+F, dpi=400)

plt.show()
plt.close(fig)


nFreq = np.shape(Freq)[0]
nRX = np.shape(RXx)[0]
nTX = np.shape(TXx_s)[0]


DataBlock = np.zeros((1,11))
for ifr in np.arange(nFreq):
    Per= 1./Freq[ifr]
    print("Write  Freq = "+str(Freq[ifr])+"    Per = "+str(Per))
    for itx in np.arange(nTX):
        indx = np.where((TXx==TXx_s[itx]) & (TXy==TXy_s[itx]))
        nind = np.size(indx)
        if nind>0:
            Rxi = RXx[indx].reshape(nind,1)
            Ryi = RXy[indx].reshape(nind,1)
            Rzi = 5.*np.ones_like(Rxi)

            Txi = TXx_s[itx]*np.ones_like(Rxi)
            Tyi = TXy_s[itx]*np.ones_like(Rxi)
            Tzi = 5.*np.ones_like(Rxi)

            P = Per*np.ones_like(Rxi)
            ReD = ReDat[indx, ifr].reshape(nind,1)
            ImD = ImDat[indx, ifr].reshape(nind,1)
            Amp = np.sqrt(ReD**2+ImD**2)
            Err = mim.error_model(Amp, Err_mul, Err_add)
            Nam0 = (ModemName
            +"F"+str(ifr).zfill(2)
            +"T"+str(itx).zfill(2))
            Nam = []
            for irx in np.arange(np.size(Rxi)):
                Nam.append(Nam0+"R"+str(irx).zfill(2))
            Nam = np.asarray(Nam,dtype=object).reshape(nind,1)

            Datai=np.concatenate((P,Txi,Tyi,Tzi,Nam,Rxi,Ryi,Rzi,ReD,ImD,Err),
                                 axis=1)

            DataBlock = np.append(DataBlock,Datai,axis=0)


DataBlock = np.delete(DataBlock,0,axis = 0)
mim.write_csem_data(DatFile=ModemFile,
                    Dat=DataBlock,
                    Head=ModemHead)
