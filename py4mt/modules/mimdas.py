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


def write_csem_data(DatFile=None, Dat=None, Site=None, Comp=None, Head = None,
               out=True):
    """
    Write ModEM input data file.

    Volker Rath
    last changed: Apr 30, 2021
    """
    datablock =np.column_stack((Dat[:,0], Site[:], Dat[:,1:6], Comp[:], Dat[:,6:10]))
    nD, _ = np.shape(datablock)

    hlin = 0
    nhead = len(Head)
    nblck = int(nhead/8)
    print(str(nblck)+" blocks will be written.")

    with open(DatFile,"w") as fd:

        for ib in np.arange(nblck):
            blockheader = Head[hlin:hlin+8]
            hlin = hlin + 8
            for ii in np.arange(8):
                fd.write(blockheader[ii])

            if "Impedance" in blockheader[2]:

                fmt = "%14e %14s"+"%15.6f"*2+" %15.1f"*3+" %14s"+" %14e"*3

                indices = []
                block = []
                for ii in np.arange(len(Comp)):
                    if ("ZX" in Comp[ii]) or ("ZY" in Comp[ii]):
                        indices.append(ii)
                        block.append(datablock[ii,:])

                if out:
                    print('Impedances')
                    print(np.shape(block))

            elif "Vertical" in blockheader[2]:

                fmt = "%14e %14s"+"%15.6f"*2+" %15.1f"*3+" %14s"+" %14e"*3

                indices = []
                block = []
                for ii in np.arange(len(Comp)):
                    if ("TX" == Comp[ii]) or ("TY" == Comp[ii]):
                        indices.append(ii)
                        block.append(datablock[ii,:])

                if out:
                    print('Tipper')
                    print(np.shape(block))

            elif "Tensor" in blockheader[2]:

                fmt = "%14e %14s"+"%15.6f"*2+" %15.1f"*3+" %14s"+" %14e"*3

                indices = []
                block = []
                for ii in np.arange(len(Comp)):
                    if ("PT" in Comp[ii]):
                        indices.append(ii)
                        block.append(datablock[ii,:])

                if out:
                    print('Phase Tensor')
                    print(np.shape(block))

            else:
                error("Data type "+blockheader[3]+'not implemented! Exit.')

            np.savetxt(fd,block, fmt = fmt)

def get_randomTX_simple(TXx=None,TXy=None,
                        Nsamples=None,
                        Seedsamples=None,
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

    if Seedsamples==None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(Seedsamples)

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
                        Seedsamples=None,
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

    if Seedsamples==None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(Seedsamples)

    TXx_s = []
    TXy_s = []
    Ind_s = []

    maxTXx = np.max(TXx)
    minTXx = np.min(TXx)
    maxTXy = np.max(TXy)
    minTXy = np.min(TXy)


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
