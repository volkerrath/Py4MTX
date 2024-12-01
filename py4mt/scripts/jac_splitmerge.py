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
Reads ModEM's Jacobian, does fancy things.

@author: vrath   Feb 2021

"""

# Import required modules

import os
import sys

# import struct
import time
from datetime import datetime
import warnings
from sys import exit as error


import numpy as np
import numpy.linalg as npl
import scipy.linalg as scl
import scipy.sparse as scs




PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [PY4MTX_ROOT+"/modules/", PY4MTX_ROOT+"/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)


import jacproc as jac
import modem as mod
import util as utl
from version import versionstrg

version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")


#WorkDir = PY4MTX_ROOT+"/work/"
WorkDir = PY4MTX_DATA+"/Peru/Sababcaya/SABA8_Jac/"
if not WorkDir.endswith("/"): WorkDir=WorkDir+"/"

Task = "merge"
MergedFile = "SABA8_ZPT__nerr_sp-8_merged"
MFiles = [WorkDir+"SABA8_Z_nerr_sp-8", WorkDir+"SABA8_P_nerr_sp-8",WorkDir+"SABA8_T_nerr_sp-8",]
nF = np.size(MFiles)
print(" The following files will be merged:")
print(MFiles)

# Task = "split"
SFile = WorkDir+"merged/UBI_ZPT_sp-8_merged"



Split = "dtype  site  freq  comp"
print(SFile)
print(" The file will be split into components:")
print(Split)
PerIntervals = [ [0.0001, 0.001], 
              [0.001, 0.01], 
              [0.01, 0.1], 
              [0.1, 1.], 
              [1., 10.], 
              [10., 100.], 
              [100., 1000.], 
              [1000., 10000.]]

if "mer" in Task.lower():
    
    
    for ifile in np.arange(nF):     
       
        start =time.perf_counter()
        print("\nReading Data from "+MFiles[ifile])
        Jac = scs.load_npz(MFiles[ifile] +"_jac.npz")
        
        
        tmp = np.load( MFiles[ifile] +"_info.npz", allow_pickle=True)
        Freq = tmp["Freq"]
        Comp = tmp["Comp"]
        Site = tmp["Site"]
        DTyp = tmp["DTyp"]
        Data = tmp["Data"]
        Scale = tmp["Scale"]
        Info = tmp["Info"]
        
        elapsed = time.perf_counter() - start
        print(" Used %7.4f s for reading Jacobian from %s " % (elapsed, MFiles[ifile]))
        
        
       
        if ifile==0:
            Jac_merged = Jac 
            print(ifile, type(Jac_merged), type(Jac), np.shape(Jac_merged))
            Data_merged = Data 
            Site_merged = Site
            Freq_merged = Freq
            Comp_merged = Comp
            DTyp_merged = DTyp 
            Infblk = Info
            Scales = Scale
        else:
            Jac_merged = scs.vstack((Jac_merged, Jac))            
            print(ifile, type(Jac_merged), np.shape(Jac_merged))
            Data_merged = np.vstack((Data_merged, Data)) 
            Site_merged = np.hstack((Site_merged, Site))
            Freq_merged = np.hstack((Freq_merged, Freq))
            Comp_merged = np.hstack((Comp_merged, Comp))
            DTyp_merged = np.hstack((DTyp_merged, DTyp))
            Infblk = np.vstack((Infblk, Info))
            Scales = np.hstack((Scales, Scale))
          
    # Scale = np.amax(Scales)
            

    start = time.perf_counter()
    np.savez_compressed(WorkDir+MergedFile+"_info.npz",
                        Freq=Freq_merged, Data=Data_merged, Site=Site_merged, 
                        Comp=Comp_merged, Info=Infblk, DTyp=DTyp_merged, 
                        Scale=Scales, allow_pickle=True)
    scs.save_npz(WorkDir+MergedFile+"_jac.npz", matrix=Jac_merged, compressed=True)

    elapsed = time.perf_counter() - start
    print(" Used %7.4f s for storing Jacobian to %s " % (elapsed, WorkDir+MergedFile))
    
    
    

if "spl" in Task.lower():

    start =time.perf_counter()
    print("\nReading Data from "+SFile)
    Jac = scs.load_npz(SFile+"_jac.npz")
    if scs.issparse(Jac): sparse= True
    tmp = np.load(SFile+"_info.npz", allow_pickle=True)
    Freq = tmp["Freq"]
    Comp = tmp["Comp"]
    Site = tmp["Site"]
    DTyp = tmp["DTyp"]
    Data = tmp["Data"]
    Scal = tmp["Scale"]
    Info = tmp["Info"]
    
    tmpinfo =  np.reshape(Info,(len(Info),3))
    freq =  tmpinfo[:,0]
    jcmp =  tmpinfo[:,1]
    nsit =  tmpinfo[:,1]
    
    if "fre" in Split.lower():

        start = time.perf_counter()
        
        nF = len(PerIntervals)
    
    
        for ibnd in np.arange(nF):  
            lowstr=str(1./PerIntervals[ibnd][0])+"Hz"
            uppstr=str(1./PerIntervals[ibnd][1])+"Hz"
              
     
            indices = np.where((Freq>=PerIntervals[ibnd][0]) & (Freq<PerIntervals[ibnd][1]))
            JacTmp = Jac[indices]
            
            FreqTmp=Freq[indices]
            DataTmp=Data[indices,:]
            SiteTmp=Site[indices]                              
            CompTmp=Comp[indices]
            InfoTmp=Info[indices]
            DTypTmp=DTyp[indices]
            ScalTmp=Scal

           
           
            Name = SFile+"_freqband"+lowstr+"_to_"+uppstr
            Head =os.path.basename(Name).replace("_", " | ")              
           
            np.savez_compressed(Name +"_info.npz", 
                                Freq=FreqTmp, Data=DataTmp, Site=SiteTmp, 
                                Comp=CompTmp, Info=InfoTmp, DTyp=DTypTmp, 
                                Scale=ScalTmp, allow_pickle=True)
            if scs.issparse(JacTmp):
                scs.save_npz( Name +"_jac.npz", matrix=JacTmp) #, compressed=True)
            else:
                np.savez_compressed(Name +"_jac.npz", JacTmp)
            
       
               
        elapsed = time.perf_counter() - start
        print(" Used %7.4f s for splitting into frequency bands " % (elapsed))        
        print("\n")
        
        
    if "comp" in Split.lower():
               
            start = time.perf_counter()
        
            """
            Full_Impedance              =  ZXX, ZYY, ZYX, ZXY
            Off_Diagonal_Impedance      =  ZYX, ZXY
            Full_Vertical_Components    = TXR, TYR
            Full_Interstation_TF        = ?
            Off_Diagonal_Rho_Phase      = ?
            Phase_Tensor                =  PTXX, PTYY, PTXY, PTYX
            """
            compstr = [
                "zxy", "zyx", "zxx", "zyy",
                "txr",  "tyr", "txi", "txr",
                "ptxy", "ptyx", "ptxx", "ptyy"]


            ExistComp = np.unique(Comp)
                    
            cnum= -1
            for icmp in ExistComp:
                cnum =cnum +1
                
                
                indices = np.where(jcmp = icmp)
                JacTmp = Jac[indices]
                
                FreqTmp=Freq[indices]
                DataTmp=Data[indices,:]
                SiteTmp=Site[indices]                              
                CompTmp=Comp[indices]
                InfoTmp=Info[indices]
                DTypTmp=DTyp[indices]
              
                
                
                Name = SFile+"_Comp"+compstr[icmp-1]
                Head =os.path.basename(Name).replace("_", " | ")
        
                np.savez_compressed(Name +"_info.npz", 
                                    Freq=FreqTmp, Data=DataTmp, Site=SiteTmp, 
                                    Comp=CompTmp, Info=InfoTmp, DTyp=DTypTmp, 
                                    Scale=ScalTmp, allow_pickle=True)
                if scs.issparse(JacTmp):
                    scs.save_npz( Name +"_jac.npz", matrix=JacTmp) #, compressed=True)
                else:
                    np.savez_compressed(Name +"_jac.npz", JacTmp)
      
                    
            elapsed = time.perf_counter() - start
            print(" Used %7.4f s for splitting into components " % (elapsed))        
            print("\n")
            

    if "dtyp" in Split.lower():
           
        start = time.perf_counter()
    
        """
        Full_Impedance              = 1
        Off_Diagonal_Impedance      = 2
        Full_Vertical_Components    = 3
        Full_Interstation_TF        = 4
        Off_Diagonal_Rho_Phase      = 5
        Phase_Tensor                = 6
        """
        typestr = ["zfull", "zoff", "tp", "mf", "rpoff", "pt"]
    
        ExistType = np.unique(DTyp)
                
        tnum= -1
        for ityp in ExistType:
            tnum =tnum +1
            
            
            indices = np.where(jcmp = ityp)
            JacTmp = Jac[indices]
            
            FreqTmp=Freq[indices]
            DataTmp=Data[indices,:]
            SiteTmp=Site[indices]                              
            CompTmp=Comp[indices]
            InfoTmp=Info[indices]
            DTypTmp=DTyp[indices]
            ScalTmp=Scal[tnum]
            
            
            
            Name = SFile+"_DType"+typestr[ityp-1]
            Head =os.path.basename(Name).replace("_", " | ")
    
            np.savez_compressed(Name +"_info.npz", 
                                Freq=FreqTmp, Data=DataTmp, Site=SiteTmp, 
                                Comp=CompTmp, Info=InfoTmp, DTyp=DTypTmp, 
                                Scale=ScalTmp, allow_pickle=True)
            if scs.issparse(JacTmp):
                scs.save_npz( Name +"_jac.npz", matrix=JacTmp) #, compressed=True)
            else:
                np.savez_compressed(Name +"_jac.npz", JacTmp)
  
                
        elapsed = time.perf_counter() - start
        print(" Used %7.4f s for splitting into components " % (elapsed))        
        print("\n")
        
    if "sit" in Split.lower():          
        
        start = time.perf_counter()
        
        SiteNames = Site[np.sort(np.unique(Site, return_index=True)[1])] 
        

        for sit in SiteNames:  
            indices = np.where(sit==Site)
            JacTmp = Jac[indices]  
            
            FreqTmp=Freq[indices]
            DataTmp=Data[indices,:]
            SiteTmp=Site[indices]                              
            CompTmp=Comp[indices]
            InfoTmp=Info[indices]
            DTypTmp=DTyp[indices] 
            ScalTmp=Scal
            
           
            Name = SFile+"_"+sit.lower()
            Head =os.path.basename(Name).replace("_", " | ")              
           
            np.savez_compressed(Name +"_info.npz", 
                                Freq=FreqTmp, Data=DataTmp, Site=SiteTmp, 
                                Comp=CompTmp, Info=InfoTmp, DTyp=DTypTmp, 
                                Scale=ScalTmp, allow_pickle=True)
            if scs.issparse(JacTmp):
                scs.save_npz( Name +"_jac.npz", matrix=JacTmp) #, compressed=True)
            else:
                np.savez_compressed(Name +"_jac.npz", JacTmp)
            
                
        elapsed = time.perf_counter() - start
        print(" Used %7.4f s for splitting into sites " % (elapsed))        
        print("\n")
        
