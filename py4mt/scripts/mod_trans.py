#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 17:17:16 2023

@author: vrath
"""
import os
import sys
import numpy as np

from sys import exit as error
# import struct
import time
from datetime import datetime
import warnings


PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]
PY4MTX_DATA = os.environ["PY4MTX_DATA"]

mypath = [PY4MTX_ROOT+"/modules/", PY4MTX_ROOT+"/scripts/"]
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

debug = False

rhoair = 1.e17
blank = rhoair

InFmt = "mod"
OutFmt =  ["rlm", "ubc"]

InMod =  "/home/vrath/JacoPyAn/work/UBC_format_example/UBI8_Z_Alpha02_NLCG_014"

if "mod" in InFmt.lower():
    print("\n\nTransforming ModEM model file:" )
    print(InMod)

    OutMod = InMod
    lat, lon =  -16.346,  -70.908

    start = time.perf_counter()
    dx, dy, dz, rho, refmod, _ = mod.read_mod(InMod, ".rho",trans="linear", volumes=True)
    dims = np.shape(rho)
    elapsed = time.perf_counter() - start
    print(" Used %7.4f s for reading MOD model from %s " % (elapsed, InMod))
   
    aircells = np.where(rho>rhoair/10)

    if debug: 
        ModExt = ".rho_debug"
        mod.write_model_mod(InMod, ModExt,
                            dx, dy, dz, rho, 
                            reference=refmod, trans="LINEAR", mvalair=blank, aircells=aircells)
    else:
        error(InFmt+" input format not yet implementd! Exit.")



    for fmt in OutFmt:
        if "ubc" in fmt.lower() :
            
            print("\n\nTransforming model file to UBC model & mesh format" )
            start = time.perf_counter()
            elev = -refmod[2]
            refcenter =  [lat, lon, elev]
            MshExt = ".mesh"
            ModExt = ".mod"
            mod.write_model_ubc(OutMod, MshExt, ModExt, 
                                dx, dy, dz, rho, refcenter, mvalair=blank, aircells=aircells)
            elapsed = time.perf_counter() - start
            print(" Used %7.4f s for Writing UBC model to %s " % (elapsed, OutMod))
    
        if "rlm" in fmt.lower() or "cgg" in fmt.lower():
            print("\n\nTransforming model file to RLM/CGG format" )
            start = time.perf_counter()
            ModExt = ".rlm"
            comment = "RLM format mdel"                   
            mod.write_rlm(OutMod, modext=ModExt, 
                          dx=dx, dy=dy, dz=dz, mval=rho, reference=refmod, mvalair=blank, aircells=aircells, comment=comment)
            print(" Cell volumes (CGG format) written to "+OutMod)
         
            elapsed = time.perf_counter() - start
            print(" Used %7.4f s for Writing RML/CGG model to %s " % (elapsed, OutMod))   




