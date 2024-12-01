#!/usr/bin/env python3

"""
Reads ModEM's Jacobian, does fancy things.

@author: vrath   Feb 2021

"""

# Import required modules

import os
import sys
from sys import exit as error
# import struct
import time
from datetime import datetime
import warnings
import gc

import numpy as np
import numpy.linalg as npl
import scipy.linalg as scl
import scipy.sparse as scs



JACOPYAN_DATA = os.environ["JACOPYAN_DATA"]
JACOPYAN_ROOT = os.environ["JACOPYAN_ROOT"]

mypath = [JACOPYAN_ROOT+"/modules/", JACOPYAN_ROOT+"/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)


import jacproc as jac
import modem as mod
from version import versionstrg
import util as utl

version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")


gc.enable()

rng = np.random.default_rng()
blank = 1.e-30 # np.nan
rhoair = 1.e17

InpFormat = "sparse"


# WorkDir = JACOPYAN_DATA+"/Annecy/Jacobians/"
# WorkDir = JACOPYAN_DATA+"/Peru/Sabancaya//SABA8_Jac/"
WorkDir = JACOPYAN_DATA+"/Peru/Ubinas/"

if not WorkDir.endswith("/"):
    WorkDir = WorkDir+"/"
    
# MFile = WorkDir + "SABA8_best.rho"
MFile = WorkDir + "UBI9_best"

# necessary for ubc, but not relevant  for synthetic model
#MOrig = [-15.767401, -71.854095] # ANN
#MOrig = [45.941551, 6.079800] # SABA
MOrig = [45.941551, 6.079800] # ANN

JacName = "UBI9_ZPTss_nerr_sp-8"
# JacName = "ANN_Z_nerr_sp-8"
JFile = WorkDir + JacName

OutFile = JFile+"_stats.dat"
ofile=open(OutFile, "w")
 



Splits = ["comp", "site", "freq"]
PerIntervals = [ 
                [0.0001, 0.001], 
                [0.001, 0.01], 
                [0.01, 0.1], 
                [0.1, 1.], 
                [1., 10.], 
                [10., 100.], 
                [100., 1000.], 
                [1000., 10000.]
                ]


total = 0.0

start = time.perf_counter()
dx, dy, dz, rho, refmod, _ = mod.read_mod(MFile, trans="linear", volumes=True)
elapsed = time.perf_counter() - start
total = total + elapsed
print(" Used %7.4f s for reading model from %s " % (elapsed, MFile))

dims = np.shape(rho)
sdims = np.size(rho)

aircells = np.where(rho>rhoair/10)
jacmask = jac.set_airmask(rho=rho, aircells=aircells, blank= blank, flat = False, out=True)
jdims= np.shape(jacmask)
j0 = jacmask.reshape(dims)
j0[aircells] = blank
jacmask = j0.reshape(jdims)
jacflat = jacmask.flatten(order="F")

#rhotest = jacmask.reshape(dims)*rho
#TSTFile = WorkDir+JacName+"1_MaskTest.rho"
#mod.write_mod(TSTFile, dx, dy, dz, rhotest, refmod, trans="LOGE", mvalair=blank, aircells=aircells)


name, ext = os.path.splitext(JFile)

start = time.perf_counter()
print("Reading Jacobian from "+JFile)

if "spa" in InpFormat:
    Jac = scs.load_npz(JFile +"_jac.npz")
    normalized = True
    
    tmp = np.load( JFile +"_info.npz", allow_pickle=True)
    Freqs = tmp["Freq"]
    Comps = tmp["Comp"]
    Sites = tmp["Site"]
    Dtype = tmp["DTyp"]
    print(np.unique(Dtype))

else:  
    
    Jac, tmp = mod.read_jac(JFile + ".jac")    
    normalized = False
    Freqs = tmp[:,0]
    Comps = tmp[:,1]
    Sites = tmp[:,2]
    Data, Site, Freq, Comp, Dtype, Head = mod.read_data_jac(JFile + "_jac.dat")
    dsh = np.shape(Data)
    err = np.reshape(Data[:, 5], (dsh[0], 1))
    Jac = jac.normalize_jac(Jac, err)
    
print("Full Jacobian")
jac.print_stats(jac=Jac, jacmask=jacflat, outfile=ofile)
print("\n")               
print("\n")
ofile.write("Full Jacobian")

start = time.perf_counter()


       
for Split in Splits:
        
    if "comp" in Split.lower():
        
        start = time.perf_counter()
    
        """
        Full_Impedance              = 1
        Off_Diagonal_Impedance      = 2
        Full_Vertical_Components    = 3
        Full_Interstation_TF        = 4
        Off_Diagonal_Rho_Phase      = 5
        Phase_Tensor                = 6
        """
        compstr = ["zfull", "zoff", "tp", "mf", "rpoff", "pt"]
    
        ExistType = np.unique(Dtype)
        
        for icmp in ExistType:
            indices = np.where(Dtype==icmp)
            JacTmp = Jac[indices]
            print("Component: ",icmp)
            ofile.write("\n Component: "+str(icmp))
            jac.print_stats(jac=JacTmp, jacmask=jacflat, outfile=ofile)
            print("\n")
            
        print("\n")
    
    if "site" in Split.lower():
        start = time.perf_counter()
        

        SiteNames = Sites[np.sort(np.unique(Sites, return_index=True)[1])] 
    
        
        for sit in SiteNames:       
            indices = np.where(sit==Sites)
            JacTmp = Jac[indices]
            print("Site: ",sit)
            ofile.write("\n Site: "+sit)
            jac.print_stats(jac=JacTmp, jacmask=jacflat, outfile=ofile)
            print("\n")
        
        print("\n")

    if "freq" in Split.lower():
        start = time.perf_counter()
        
        nF = len(PerIntervals)
        
        for ibnd in np.arange(nF):  
            lowstr=str(1./PerIntervals[ibnd][0])+"Hz"            
            uppstr=str(1./PerIntervals[ibnd][1])+"Hz"                   
             
 
            indices = np.where((Freqs>=PerIntervals[ibnd][0]) & (Freqs<PerIntervals[ibnd][1]))       
            JacTmp = Jac[indices]
            
            if np.shape(JacTmp)[0] > 0:  
                print("Freqband: ", lowstr, "to", uppstr)
                ofile.write("\n Freqband: "+lowstr+"to"+uppstr)
                jac.print_stats(jac=JacTmp, jacmask=jacflat, outfile=ofile)
                print("\n")
            else: 
                 print("Frequency band is empty! Continue.")
        
        print("\n")
        print("Done!")

ofile.close()
