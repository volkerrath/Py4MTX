#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 12:34:21 2025

@author: vrath
"""

import os
import sys
import binascii
import struct
import numpy as np
import fnmatch

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [PY4MTX_ROOT+"/py4mt/modules/", PY4MTX_ROOT+"/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)

#import modules
import util as utl
from version import versionstrg

def get_femtic_sorted(files=[], out=True):
    numbers = []
    for file in files:
        numbers.append(int(file[11:]))
    numbers = sorted(numbers)

    listfiles = []
    for ii in numbers:
        fil = 'sensMatFreq'+str(ii)
        listfiles.append(fil)

    if out:
        print(listfiles)
    return listfiles

datadir = '/home/vrath/work/tmp/inversion_1/'
sensread=True



if sensread:
    filelist = utl.get_filelist(searchstr=['sensMatFreq*.bin'], searchpath=datadir)
    filelist = get_femtic_sorted(filelist)
    
    fcount = -1
    for file in filelist:
        fcount = fcount+1
        with open(datadir+file,'rb') as readfile:
            tmp = readfile.read(4)
            numdata=int(struct.unpack('i', tmp)[0])
            tmp = readfile.read(4)
            numpara=int(struct.unpack('i', tmp)[0])     
            jacsiz = numdata*numpara
            tmpjac = np.zeros((jacsiz,1))
            print('jacobian is', numdata, 'times', numpara,':',jacsiz)
            for jj in np.arange(jacsiz):
                tmp = readfile.read(8)
                tmpjac[jj,0] = float(struct.unpack('d', tmp)[0])
                
        tmpjac = np.reshape(tmpjac, (numpara, numdata))
        
    
        if fcount == 0:
            jac = tmpjac
        else:
            jac = np.hstack((jac, tmpjac))
              
    print(np.shape(jac))
    
    J = jac.T
    np.savez_compressed(datadir+'JacobianFull', J = J)
else:
    J = np.load(datadir+'JacobianFull.npz')['J']

JTJ = J.T@J
JJT = J@J.T


np.savez_compressed(datadir+'JacobianProducts', JTJ = JTJ, JJT = JJT)


# with open(datadir+'roughening_matrix.out', 'r') as readfile:
#        tmprough = readfile.readlines()
#        #.append((tmprough,line), axis=1)

# print(np.shape(tmprough))

# if sensread:
#     filelist = utl.get_filelist(searchstr=['sensMatFreq*Mod.bin'], searchpath=datadir)
#     filelist = get_femtic_sorted(filelist)
    
#     fcount = -1
#     for file in filelist:
#         fcount = fcount+1
#         with open(datadir+file,'rb') as readfile:
#             tmp = readfile.read(4)
#             numdata=int(struct.unpack('i', tmp)[0])
#             tmp = readfile.read(4)
#             numpara=int(struct.unpack('i', tmp)[0])     
#             jacsiz = numdata*numpara
#             tmpjac = np.zeros((jacsiz,1))
#             print('jacobian is', numdata, 'times', numpara,':',jacsiz)
#             for jj in np.arange(jacsiz):
#                 tmp = readfile.read(8)
#                 tmpjac[jj,0] = float(struct.unpack('d', tmp)[0])
                
#         tmpjac = np.reshape(tmpjac, (numpara, numdata))
        
    
#         if fcount == 0:
#             jac = tmpjac
#         else:
#             jac = np.hstack((jac, tmpjac))
              
#     print(np.shape(jac))
    
#     J = jac.T
#     np.savez_compressed(datadir+'JacobianFull', J = J)
# else:
#     J = np.load(datadir+'JacobianFull.npz')['J']
