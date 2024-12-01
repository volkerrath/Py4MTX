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
import netCDF4 as nc

from numba import njit

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
Blank = 1.e-30 # np.nan
Rhoair = 1.e17

InpFormat = "sparse"
OutFormat = "mod rlm" # "ubc"
# for 3d-Grid:
ModExt = "_sns.rho"

# Annecy Annecy Annecy Annecy Annecy Annecy Annecy Annecy Annecy Annecy
# WorkDir = JACOPYAN_DATA+"/Annecy/Jacobians/"
# if not WorkDir.endswith("/"):
#     WorkDir = WorkDir+"/"
# MFile = WorkDir + "ANN_best"
# MOrig = [45.941551, 6.079800] # ANN
# JacName = "ANN_ZPT_nerr_sp-8

# Sabancaya Sabancaya Sabancata Sabancaya Sabancaya Sabancaya Sabancaya
# WorkDir = JACOPYAN_DATA+"/Peru/Sabancaya/"
# if not WorkDir.endswith("/"):
#     WorkDir = WorkDir+"/"
# MFile = WorkDir + "SABA8_best"
# MOrig = [15.767401, -71.854095]
# JacName = ""

# Ubinas Ubinas Ubinas Ubinas Ubinas Ubinas Ubinas Ubinas Ubinas Ubinas
# WorkDir = JACOPYAN_DATA+"/Peru/Ubinas/"
#WorkDir = "/home/vrath/UBI38_JAC/"
#MOrig = [-16.345800 -70.908249] # UBI
#JacName = "Ubi38_ZPT_nerr_sp-8"
#MFile = WorkDir + "Ubi38_ZssPT_Alpha02_NLCG_023"

# Misti Misti Misti Misti Misti Misti Misti Misti Misti Misti Misti Misti
#WorkDir = JACOPYAN_DATA+"/Peru/Misti/"
#if not WorkDir.endswith("/"):
    #WorkDir = WorkDir+"/"
#MFile = WorkDir + "Misti10_best"
#MOrig = [-16.277300, -71.444397]# Misti
## JacName = "Misti_best_Z5_nerr_sp-8"
#JacName = "Misti_best_ZT_extended_nerr_sp-8"

WorkDir = "/home/vrath/Ubaye/"

#JacName = "Ubaye26_P_nerr_sp-8"
#JFile = WorkDir + JacName
JacName = "Ubaye26_T_nerr_sp-8"
JFile = WorkDir + JacName
#JacName = "Ubaye26_Z_nerr_sp-8"
#JFile = WorkDir + JacName


MFile = WorkDir + "Ub26_ZPT_T200_NLCG_014"
MOrig = [0., 0.]


VolExtract = True
if VolExtract:
    VolFile = MFile
    VolFmt = ""

TopoExtract = True
if TopoExtract:
    TopoFile = WorkDir + "Ubaye_Topo.dat"
    TopoFmt = ""


Splits = "total dtyp site freq comp"
NoReIm = True

NormLocal = True
if (not NormLocal) and ("tot" not in  Splits.lower()):
    Splits = "total"+Splits

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


Type = "euc"
# Type = "euc"
"""
Calculate sensitivities.
Expects that Jacobian is already error-scaled, i.e Jac = C^(-1/2)*J.
Options:
    Type = "raw"     sensitivities summed along the data axis
    Type = "abs"     absolute sensitivities summed along the data axis
                     (often called coverage)
    Type = "euc"     squared sensitivities summed along the data axis.

Usesigma:
    if true, sensitivities with respect to sigma  are calculated.
"""

#Transform = [ "sqr"]
#Transform = [ "sqr", "max"]
Transform = [ "sqr","vol","max"]

"""
Transform sensitivities.
Options:
    Transform = "vol", "siz"    Normalize by the values optional array vol ("volume"),
                                i.e in our case layer thickness. This should always
                                be the first value in Transform list.
    Transform = "max"           Normalize by maximum (absolute) value.
    Transform = "sur"           Normalize by surface value.
    Transform = "sqr"           Take the square root. Only usefull for euc sensitivities.
    Transform = "log"           Take the logaritm. This should always be the
                                last value in Transform list
"""
snsstring = Type.lower()+"_"+"_".join(Transform)

SensDir = WorkDir+"/sens_"+snsstring+"/"
if not SensDir.endswith("/"):
    SensDir = SensDir+"/"
if not os.path.isdir(SensDir):
    print("File: %s does not exist, but will be created" % SensDir)
    os.mkdir(SensDir)


total = 0.0

start = time.perf_counter()
dx, dy, dz, rho, refmod, _ = mod.read_mod(MFile, trans="linear")
elapsed = time.perf_counter() - start
total = total + elapsed
print(" Used %7.4f s for reading model from %s " % (elapsed, MFile))

aircells = np.where(rho>Rhoair/10)



if TopoExtract:

    xcnt, ycnt, topo = mod.get_topo(dx=dx, dy=dy, dz=dz, mval=rho, ref= refmod,
             mvalair = 1.e17, out=True)

    if os.path.isfile(TopoFile):
        os.remove(TopoFile)

    with open(TopoFile, "a") as f:

        for ii in np.arange(len(dx)):
            for jj in np.arange(len(dy)):
                line = str(xcnt[ii])+", "+str(ycnt[jj])+", "+str(topo[ii,jj])+"\n"
                f.write(line)



if VolExtract or ("size" in Transform):
    vol = mod.get_volumes(dx=dx, dy=dy, dz=dz, mval=refmod, out=True)
    print(np.shape(vol), np.shape(rho))
    Header = "# "+MFile

    if "mod" in OutFormat.lower():
        # for modem_readable files

        mod.write_mod(VolFile, modext="_vol.rho",
                      dx=dx, dy=dy, dz=dz, mval=vol,
                      reference=refmod, mvalair=Blank, aircells=aircells, header=Header)
        print(" Cell volumes (ModEM format) written to "+VolFile)

    if "ubc" in OutFormat.lower():
        elev = -refmod[2]
        refubc =  [MOrig[0], MOrig[1], elev]
        mod.write_ubc(VolFile, modext="_ubc.vol", mshext="_ubc.msh",
                      dx=dx, dy=dy, dz=dz, mval=vol, reference=refubc, mvalair=Blank, aircells=aircells, header=Header)
        print(" Cell volumes (UBC format) written to "+VolFile)

    if "rlm" in OutFormat.lower():
        mod.write_rlm(VolFile, modext="_vol.rlm",
                      dx=dx, dy=dy, dz=dz, mval=vol, reference=refmod, mvalair=Blank, aircells=aircells, comment=Header)
        print(" Cell volumes (CGG format) written to "+VolFile)

else:
    vol = np.array([])


# sys.exit()
mdims = np.shape(rho)

aircells = np.where(rho>Rhoair/10)
jacmask = jac.set_airmask(rho=rho, aircells=aircells, blank=Blank, flat = False, out=True)
jacflat = jacmask.flatten(order="F")
name, ext = os.path.splitext(MFile)

# ofile = name
# Header = JacName
# trans = "LINEAR"

# mod.write_mod(ofile, modext="_mod.rho", trans = trans,
#                   dx=dx, dy=dy, dz=dz, mval=rho,
#                   reference=refmod, mvalair=Blank, aircells=aircells, header=Header)
# print(" Model (ModEM format) written to "+ofile)

# # elev = -refmod[2]
# # refubc =  [MOrig[0], MOrig[1], elev]
# # mod.write_ubc(OFile, modext="_rho_ubc.mod", mshext="_rho_ubc.msh",
# #                   dx=dx, dy=dy, dz=dz, mval=rho, reference=refubc, mvalair=Blank, aircells=aircells, header=Header)
# # print(" Model (UBC format) written to "+ofile)

# TSTFile = WorkDir+JacName+"0_MaskTest"
# mod.write_mod(TSTFile, modext="_mod.rho", trans = trans,
#             dx=dx, dy=dy, dz=dz, mval=rho, reference=refmod, mvalair=Blank, aircells=aircells, header=Header)
# rhotest = jacmask.reshape(dims)*rho
# TSTFile = WorkDir+JacName+"1_MaskTest"
# mod.write_mod(TSTFile, modext="_mod.rho", trans = trans,
#             dx=dx, dy=dy, dz=dz, mval=rhotest, reference=refmod, mvalair=Blank, aircells=aircells, header=Header)


# name, ext = os.path.splitext(JFile)

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

    Data, Sites, Freqs, Comps, Dtype, Head = mod.read_data_jac(JFile + "_jac.dat")
    dsh = np.shape(Data)
    err = np.reshape(Data[:, 5], (dsh[0], 1))
    Jac = jac.normalize_jac(Jac, err)



elapsed = time.perf_counter() - start
print(" Used %7.4f s for reading Jacobian/data from %s" % (elapsed, JFile))
total = total + elapsed

print("Full Jacobian")
jac.print_stats(jac=Jac, jacmask=jacflat)
print("\n")
print("\n")


start = time.perf_counter()

if "tot"in Splits.lower():

    SensTmp = jac.calc_sensitivity(Jac,
                        Type = Type, OutInfo=False)
    SensTot, MaxTotal = jac.transform_sensitivity(S=SensTmp, Vol=vol,
                            Transform=Transform, OutInfo=False)

    SensFile = SensDir+JacName+"_total_"+snsstring
    Header = "# "+SensFile.replace("_", " | ")

    S = SensTot.reshape(mdims, order="F")

    if "mod" in OutFormat.lower():
        mod.write_mod(SensFile, modext=ModExt,
                    dx=dx, dy=dy, dz=dz, mval=S,
                    reference=refmod, mvalair=Blank, aircells=aircells, header=Header)
        print(" Sensitivities (ModEM format) written to "+SensFile)

    if "ubc" in OutFormat.lower():
        elev = -refmod[2]
        refubc =  [MOrig[0], MOrig[1], elev]
        mod.write_ubc(SensFile, modext="_ubc.sns", mshext="_ubc.msh",
                    dx=dx, dy=dy, dz=dz, mval=S, reference=refubc, mvalair=Blank, aircells=aircells, header=Header)
        print(" Sensitivities (UBC format) written to "+SensFile)

    if "rlm" in OutFormat.lower():
        mod.write_rlm(SensFile, modext="_sns.rlm",
                    dx=dx, dy=dy, dz=dz, mval=S, reference=refmod, mvalair=Blank, aircells=aircells, comment=Header)
        print(" Sensitivities (CGG format) written to "+SensFile)


    elapsed = time.perf_counter() - start
    print(" Used %7.4f s for total sensitivities " % (elapsed))



if "dtyp" in Splits.lower():

    start = time.perf_counter()

    """
    Full_Impedance              = 1
    Off_Diagonal_Impedance      = 2
    Full_vertical_Components    = 3
    Full_Interstation_TF        = 4
    Off_Diagonal_Rho_Phase      = 5
    Phase_Tensor                = 6
    """
    typestr = ["zfull", "zoff", "tp", "mf", "rpoff", "pt"]

    ExistType = np.unique(Dtype)
    print(ExistType)

    for ityp in ExistType:
        indices = np.where(Dtype == ityp)
        JacTmp = Jac[indices]
        print("Data type: ",ityp)
        jac.print_stats(jac=JacTmp, jacmask=jacflat)
        print("\n")

        if NormLocal:
            maxval = None
        else:
            maxval = MaxTotal

        SensTmp = jac.calc_sensitivity(JacTmp,
                     Type = Type, OutInfo=False)
        SensTmp, _ = jac.transform_sensitivity(S=SensTmp, Vol=vol,
                          Transform=Transform, Maxval=maxval, OutInfo=False)
        S = np.reshape(SensTmp, mdims, order="F")


        SensFile = SensDir+JacName+"_Dtype_"+typestr[ityp-1]+"_"+snsstring
        Header = "# "+SensFile.replace("_", " | ")


        if "mod" in OutFormat.lower():
            mod.write_mod(SensFile, ModExt,
                          dx=dx, dy=dy, dz=dz, mval=S,
                          reference=refmod, mvalair=Blank, aircells=aircells, header=Header)
            print(" Data type sensitivities (ModEM format) written to "+SensFile)

        if "ubc" in OutFormat.lower():
            elev = -refmod[2]
            refubc =  [MOrig[0], MOrig[1], elev]
            mod.write_ubc(SensFile, modext="_ubc.sns" ,mshext="_ubc.msh",
                          dx=dx, dy=dy, dz=dz, mval=S,
                          reference=refubc, mvalair=Blank, aircells=aircells, header=Header)
            print(" Data type sensitivities (UBC format) written to "+SensFile)

        if "rlm" in OutFormat.lower():
            mod.write_rlm(SensFile, modext="_sns.rlm",
                          dx=dx, dy=dy, dz=dz, mval=S, reference=refmod, mvalair=Blank, aircells=aircells, comment=Header)
            print(" Data type sensitivities (CGG format) written to "+SensFile)


    elapsed = time.perf_counter() - start
    print(" Used %7.4f s for data type sensitivities " % (elapsed))
    print("\n")

if "comp" in Splits.lower():

    start = time.perf_counter()
    """
    Full_Impedance              =  ZXX, ZYY, ZYX, ZXY
    Off_Diagonal_Impedance      =  ZYX, ZXY
    Full_Vertical_Components    = TXR, TYR
    Full_Interstation_TF        = ?
    Off_Diagonal_Rho_Phase      = ?
    Phase_Tensor                =  PTXX, PTYY, PTXY, PTYX
    """

    ExistComp = np.unique(Comps)

    if NoReIm:
        ExistComp= np.unique([cmp.replace("R","").replace("I","") for cmp in ExistComp])
        Comps = [cmp.replace("R","").replace("I","") for cmp in Comps]

    print(ExistComp)

    for icmp in ExistComp:

        indices = np.where(icmp in Comps)
        JacTmp = Jac[indices]
        print("Component: ",icmp)
        jac.print_stats(jac=JacTmp, jacmask=jacflat)
        print("\n")

        if NormLocal:
            maxval = None
        else:
            maxval = MaxTotal

        SensTmp = jac.calc_sensitivity(JacTmp,
                     Type = Type, OutInfo=False)
        SensTmp, _ = jac.transform_sensitivity(S=SensTmp, Vol=vol,
                          Transform=Transform, Maxval=maxval, OutInfo=False)
        S = np.reshape(SensTmp, mdims, order="F")

        SensFile = SensDir+JacName+"_"+icmp+"_"+snsstring
        Header = "# "+SensFile.replace("_", " | ")

        if "mod" in OutFormat.lower():
            mod.write_mod(SensFile, ModExt,
                          dx=dx, dy=dy, dz=dz, mval=S,
                          reference=refmod, mvalair=Blank, aircells=aircells, header=Header)
            print(" Component sensitivities (ModEM format) written to "+SensFile)

        if "ubc" in OutFormat.lower():
            elev = -refmod[2]
            refubc =  [MOrig[0], MOrig[1], elev]
            mod.write_ubc(SensFile, modext="_ubc.sns" ,mshext="_ubc.msh",
                          dx=dx, dy=dy, dz=dz, mval=S,
                          reference=refubc, mvalair=Blank, aircells=aircells, header=Header)
            print(" Component sensitivities (UBC format) written to "+SensFile)

        if "rlm" in OutFormat.lower():
            mod.write_rlm(SensFile, modext="_sns.rlm",
                          dx=dx, dy=dy, dz=dz, mval=S, reference=refmod, mvalair=Blank, aircells=aircells, comment=Header)
            print(" Component sensitivities (CGG format) written to "+SensFile)


    elapsed = time.perf_counter() - start
    print(" Used %7.4f s for comp sensitivities " % (elapsed))
    print("\n")


if "site" in Splits.lower():
    start = time.perf_counter()


    SiteNames = Sites[np.sort(np.unique(Sites, return_index=True)[1])]
    print(SiteNames)

    for sit in SiteNames:
        indices = np.where(sit==Sites)
        JacTmp = Jac[indices]
        print("Site: ",sit)
        jac.print_stats(jac=JacTmp, jacmask=jacflat)
        print("\n")

        if NormLocal:
             maxval = None
        else:
             maxval = MaxTotal

        SensTmp = jac.calc_sensitivity(JacTmp,
                     Type = Type, OutInfo=False)
        SensTmp, _  = jac.transform_sensitivity(S=SensTmp, Vol=vol,
                          Transform=Transform, Maxval=maxval, OutInfo=False)
        S = np.reshape(SensTmp, mdims, order="F")

        SensFile = SensDir+JacName+"_"+sit.lower()+"_"+snsstring
        Header = "# "+SensFile.replace("_", " | ")


        if "mod" in OutFormat.lower():
             mod.write_mod(SensFile, modext=ModExt,
                           dx=dx, dy=dy, dz=dz, mval=S,
                           reference=refmod, mvalair=Blank, aircells=aircells, header=Header)
             print(" Site sensitivities (ModEM format) written to "+SensFile)

        if "ubc" in OutFormat.lower():
             elev = -refmod[2]
             refubc =  [MOrig[0], MOrig[1], elev]
             mod.write_ubc(SensFile, modext="_ubc.sns", mshext="_ubc.msh",
                           dx=dx, dy=dy, dz=dz, mval=S,
                           reference=refubc, mvalair=Blank, aircells=aircells, header=Header)
             print(" Site sensitivities (UBC format) written to "+SensFile)

        if "rlm" in OutFormat.lower():
             mod.write_rlm(SensFile, modext="_sns.rlm",
                           dx=dx, dy=dy, dz=dz, mval=S, reference=refmod, mvalair=Blank, aircells=aircells, comment=Header)
             print(" Site sensitivities (CGG format) written to "+SensFile)


    elapsed = time.perf_counter() - start
    print(" Used %7.4f s for site sensitivities " % (elapsed))
    print("\n")

if "freq" in Splits.lower():
    start = time.perf_counter()

    nF = len(PerIntervals)

    for ibnd in np.arange(nF):

       lowstr=str(1./PerIntervals[ibnd][0])+"Hz"
       uppstr=str(1./PerIntervals[ibnd][1])+"Hz"


       indices = np.where((Freqs>=PerIntervals[ibnd][0]) & (Freqs<PerIntervals[ibnd][1]))


       JacTmp = Jac[indices]
       if np.shape(JacTmp)[0] > 0:
           print("Freqband: ", lowstr, "to", uppstr)
           jac.print_stats(jac=JacTmp, jacmask=jacflat)
           print("\n")

           if NormLocal:
               maxval = None
           else:
               maxval = MaxTotal

           SensTmp = jac.calc_sensitivity(JacTmp,
                        Type = Type, OutInfo=False)
           SensTmp, _  = jac.transform_sensitivity(S=SensTmp, Vol=vol,
                             Transform=Transform, Maxval=maxval, OutInfo=False)
           S = np.reshape(SensTmp, mdims, order="F")

           SensFile = SensDir+JacName+"_freqband"+lowstr+"_to_"+uppstr+"_"+snsstring
           Header = "# "+SensFile.replace("_", " | ")

           if "mod" in OutFormat.lower():
               mod.write_mod(SensFile, modext=ModExt,
                             dx=dx, dy=dy, dz=dz, mval=S,
                             reference=refmod, mvalair=Blank, aircells=aircells, header=Header)
               print(" Frequency band sensitivities (ModEM format) written to "+SensFile)

           if "ubc" in OutFormat.lower():
               elev = -refmod[2]
               refubc =  [MOrig[0], MOrig[1], elev]
               mod.write_ubc(SensFile, modext="_ubc.sns", mshext="_ubc.msh",
                             dx=dx, dy=dy, dz=dz, mval=S,
                             reference=refubc, mvalair=Blank, aircells=aircells, header=Header)
               print(" Frequency band sensitivities (UBC format) written to "+SensFile)


           if "rlm" in OutFormat.lower():
                mod.write_rlm(SensFile, modext="_sns.rlm",
                              dx=dx, dy=dy, dz=dz, mval=S, reference=refmod, mvalair=Blank, aircells=aircells, comment=Header)
                print(" Cell volumes (CGG format) written to "+SensFile)

       else:
            print("Frequency band is empty! Continue.")


    elapsed = time.perf_counter() - start
    print(" Used %7.4f s for Freq sensitivities " % (elapsed))
    print("\n")
