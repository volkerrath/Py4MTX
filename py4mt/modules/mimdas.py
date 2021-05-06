#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 13:40:12 2021

@author: vrath
"""
import os
import sys
import warnings
import time

from sys import exit as error
from datetime import datetime

import numpy as np

def read_csem_data(DatFile=None, out=True):
    """
    Read CSEM MIMDAS input data.

    Volker Rath
    last changed: Apr 28, 2021

    """
    Data = []
    Head = []

    with open(DatFile) as fd:
        ll = 0
        for line in fd:
            ll=ll+1
            if ll <= 3:
                Head.append(line)
                continue

            tmp = line.split()
            tmp =["1.e32" if x=="*" else x for x in tmp]
            Data.append(tmp)


    Data = np.asarray(Data,dtype=float)


    nD = np.shape(Data)
    if out:
        print("readDat: %i data read from %s" % (nD[0], DatFile))

    return Data, Head


def write_csem_data(DatFile=None, Dat=None, Head = None,
               out=True):
    """
    Write ModEM input data file.

    Volker Rath
    last changed: Apr 30, 2021
    """
    fmt = "%14e"+" %12.1f "*3+" %20s"+" %12.1f "*3+" %14e"*3


    with open(DatFile,"w") as fd:

         fd.write(Head)
         np.savetxt(fd,Dat, fmt = fmt)

def get_randomTX_simple(TXx=None,TXy=None,
                        Nsamples=None,
                        Ranstate=None,
                        d_margin=0.01):
    """
    Generate  uniform  distribution
    of sampling points.

    Returns:
    Indices amd coordinates for corresoonding points.

    Volker Rath, May 2021

    """
    if TXx==[] or TXy==[]:
        error("no data given! Exit.")

    if Ranstate==None:
        rng = np.random.default_rng()
    else:
        rng = Ranstate

    maxTXx = np.max(TXx)
    minTXx = np.min(TXx)
    maxTXy = np.max(TXy)
    minTXy = np.min(TXy)

    dxx = np.abs(maxTXx-minTXx)*d_margin
    dyy = np.abs(maxTXy-minTXy)*d_margin
    xtest = rng.uniform(minTXx-dxx, maxTXx+dxx, Nsamples)
    ytest = rng.uniform(minTXy-dyy, maxTXy+dyy, Nsamples)

    TXx_s = []
    TXy_s = []
    Ind_s = []
    for ii in np.arange(Nsamples):
        d = ((TXx-xtest[ii])**2+(TXy-ytest[ii])**2)**0.5
        indx = np.argwhere(d == np.min(d))
        TXx_s.append(TXx[indx[0]])
        TXy_s.append(TXy[indx[0]])
        Ind_s.append(indx[0])

    TXx_s = np.asarray(TXx_s, dtype=object)
    TXy_s = np.asarray(TXy_s, dtype=object)
    Ind_s = np.asarray(Ind_s, dtype=object)

    print(np.shape(Ind_s))

    return Ind_s, TXx_s, TXy_s

def get_randomTX_constr(TXx=None,TXy=None,
                        Nsamples=None,
                        Ranstate=None,
                        Mindist=250., d_margin=0.01):
    """
    Generate  uniform  distribution
    of sampling points.

    Returns:
    Indices amd coordinates for corresoonding points.

    Volker Rath, May 2021

    """

    if TXx==[] or TXy==[]:
        error("no data given! Exit.")
    else:
        TX = np.unique(TXx)
        TY = np.unique(TXy)
        # NTX = np.shape(TX)
        # NTY = np.shape(TY)
        # print(NTX, NTY)

    if Ranstate==None:
        rng = np.random.default_rng()
    else:
        rng = Ranstate

    TXx_s = []
    TXy_s = []
    Ind_s = []

    testx = rng.choice(TX,size=1)
    testy = rng.choice(TY,size=1)
    TXx_s.append(testx)
    TXy_s.append(testy)
    Ind_s.append(0)

    N = 1
    while N < Nsamples:
        keep = True
        xtest = rng.choice(TX, size=1)
        ytest = rng.choice(TY, size=1)
        for n in np.arange(N):
            dtestx = np.abs(TXx_s[n]-xtest)
            dtesty = np.abs(TXy_s[n]-ytest)
            if dtestx < Mindist[0] and dtesty < Mindist[1]:
                keep = False
        if keep:
            TXx_s.append(xtest)
            TXy_s.append(ytest)
            Ind_s.append(N)
            N = N+1


    print(N,Nsamples)
    print(np.arange(N))



    # TXx_s = np.asarray(TXx_s, dtype=object)
    # TXy_s = np.asarray(TXy_s, dtype=object)
    # Ind_s = np.asarray(Ind_s, dtype=object)
    # print(np.shape(Ind_s))


    return Ind_s, TXx_s, TXy_s


def error_model(data_obs, daterr_mul=0., daterr_add=0.):
    """
    Generate errors.

    Error model including multiplicative and additive noise

    VR Apr 2021

    """

    daterr_a = daterr_add * np.ones_like(data_obs)
    daterr_m = daterr_mul * np.ones_like(data_obs)

    data_err = \
        daterr_m * np.abs(data_obs)+ daterr_a

    # data_err = \
    #     np.sqrt(np.power(daterr_m * data_obs, 2) + np.power(daterr_a, 2))

    return data_err
