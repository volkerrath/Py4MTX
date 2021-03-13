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
Created on Fri Sep 11 15:41:25 2020

@author: vrath
"""
# Import required modules

import os
import sys
from sys import exit as error
import time
from datetime import datetime
import warnings

import numpy as np

mypath = ["/home/vrath/Py4MT/py4mt/modules/",
          "/home/vrath/Py4MT/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)
# Import required modules

import util as utl
import modem as mod
from version import versionstrg

Strng, _ = versionstrg()
now = datetime.now()
print("\n\n"+Strng)
print("Generate prior test job"+"\n"+"".join("Date " + now.strftime("%m/%d/%Y, %H:%M:%S")))
print("\n\n")



Data_tmp = '.TMP'

Name = 'Maur_PT_R500'
Exe_in = '/home/vrath/work/MaurienneJCN/RunGnuFrog.sh'
Exe_out = '/home/vrath/work/MaurienneJCN/RunGnuFrog_' + Name + '.sh'
utl.strreplace('RUNNAME', Name, Exe_in, Exe_out)

Prior_in = '/home/vrath/work/MaurienneJCN/MaurPrior.rho'
Prior_out = '/home/vrath/work/MaurienneJCN/MaurPrior_R500.rho'
prior_val = 5.00000E+2
utl.strreplace('PRIOR-VAL', str(prior_val), Prior_in, Prior_out)

Covar_in = '/home/vrath/work/MaurienneJCN/Maur.cov'
Covar_out = '/home/vrath/work/MaurienneJCN/Maur_02.cov'
a_x = 0.2
a_y = 0.2
a_z = 0.2
utl.strreplace('COVX', str(a_x), Covar_in, 'tmp1')
utl.strreplace('COVY', str(a_y), 'tmp1', 'tmp2')
utl.strreplace('COVZ', str(a_z), 'tmp2', Covar_out)

DataFiles = ['/home/vrath/work/MaurienneJCN/Maur_PT.dat']
# '/home/vrath/work/MaurienneJCN/Maur_T.dat',
# '/home/vrath/work/MaurienneJCN/Maur_Z.dat']

FwdFile = '/home/vrath/work/MaurienneJCN/Maur.fwd'

InvFilebase = '/home/vrath/work/MaurienneJCN/Maur.inv'
RunNamebase = Name


for f in DataFiles:
    Site, Comp, Data, Head = mod.read_data(f)
    Freqs = utl.unique(Data[:, 0])
    NFreq = np.size(Freqs)
    Sites = utl.unique(Site)
    NSits = np.size(Sites)
    sData = np.shape(Data)
    NdTot = sData[0]
    Nsite = np.size(Sites)
    OldHead = Head[7]

    for place in Sites:
        NdSite = utl.strcount(place, f)
        NdJCN = NdTot - NdSite
        print('Number of data from site ' + place + ' is ' + str(NdSite))
        print('Number of JCN sample is ' + str(NdJCN))

        name, ext = os.path.splitext(f)
        Data_out = name + '_No' + place + ext
        print('New datafile: ' + Data_out)
        utl.strdelete(place, f, Data_tmp, out=True)

        _, _, D, _ = mod.read_data(Data_tmp)
        Freqs_out = utl.unique(D[:, 0])
        NFreq_out = np.size(Freqs_out)
        Headstr = '> ' + str(NFreq_out) + '   ' + str(NSits - 1) + '  \n'
        utl.strreplace(OldHead, Headstr, Data_tmp, Data_out)

        name, ext = os.path.splitext(InvFilebase)
        InvFile = name + '_No' + place + ext
        RunName = RunNamebase + '_No' + place
        utl.strreplace('RUNNAME', RunName, InvFilebase, InvFile)

        filep = Prior_out.split(os.sep)[-1]
        filed = Data_out.split(os.sep)[-1]
        filei = InvFile.split(os.sep)[-1]
        filef = FwdFile.split(os.sep)[-1]
        filec = Covar_out.split(os.sep)[-1]

        exestr1 = 'mpirun --hostfile ${OAR_NODE_FILE} -n 80 -npernode 16 /home/superlana/bin/gMod3DMT7.x -I NLCG '
        exestr2 = filep + ' ' + filed + ' ' + filei + ' ' + \
            filef + ' ' + filec + ' >' + RunName + '.out \n'
        with open(Exe_out, 'a') as exe:
            exe.write(exestr1 + exestr2)
