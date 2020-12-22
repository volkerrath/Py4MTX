#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 11:33:17 2020

@author: vrath
"""
# import os
import sys
from sys import exit as error
import numpy as np
from scipy.io import FortranFile 
# import scipy.sparse as scp

import netCDF4 as nc
# import h5netcdf as hc


def readJac(JacFile=None, out=False):
    """
    Reads Jacobian from ModEM output
    author: vrath
    last changed: July 10, 2020
    """
    if out:
        print('Opening and reading '+JacFile)
        
    fjac = FortranFile(JacFile, 'r')
    Temp= []
    # print(np.shape(Jac))
    # header1 = 
    _ = fjac.read_record(np.byte)
    # h1 = ''.join([chr(item) for item in header1])
    # print(h1)
    # nAll = 
    _ = fjac.read_ints(np.int32)
    nTx  = fjac.read_ints(np.int32)
    # print(nTx)
    for i1 in range(nTx[0]):
        nDt = fjac.read_ints(np.int32)
        # print(nDt)
        for i2 in range(nDt[0]):
            nSite = fjac.read_ints(np.int32)
            # print(nSite)
            for i3 in range(nSite[0]):
                # header2 
                _ = fjac.read_record(np.byte)
                # h2 = ''.join([chr(item) for item in header2])
                # print(h2)
                nSigma = fjac.read_ints(np.int32)
                # print(nSigma)
                for i4 in range(nSigma[0]):
                    # paramType 
                    _ = fjac.read_ints(np.byte)
                    # p = ''.join([chr(item) for item in paramType])
                    # print(p)
                    # dims 
                    _ = fjac.read_ints(np.int32) 
                    # print(dims)
                    # dx 
                    _ = fjac.read_reals(np.float64)
                    # dy 
                    _ = fjac.read_reals(np.float64)
                    # dz 
                    _ = fjac.read_reals(np.float64)
                    # AirCond  
                    _ = fjac.read_reals(np.float64)
                    ColJac = fjac.read_reals(np.float64).flatten(order='F')
                    # print(np.shape(CellSens))
                    # ColJac =  CellSens.flatten(order='F')  
                    Temp.append(ColJac)
                    # print(np.shape(Temp))
                    
    Jac = np.asarray(Temp)
   
    fjac.close()
    
    if out:
        print('...done reading '+JacFile)
        
    return Jac

  
def writeJacNC(NCFile=None, Jac=None, Dat= None, 
            Site= None, Comp=None, zlib_in=True, shuffle_in=True, out = True):
    """
    Writes Jacobian from ModEM output
    to NETCDF/HDF5 file
    author: vrath
    last changed: July 25, 2020
    """
  

    JacDim = np.shape(Jac)
    DatDim = np.shape(Dat)

    if JacDim[0] != DatDim[0]:
        print ('Error:  Jac dim='+str(JacDim[0])+' does not match Dat dim='+str(DatDim[0]))
        sys.exit(1)
        
   
        
    ncout = nc.Dataset(NCFile,'w', format='NETCDF4');
    ncout.createDimension('data',JacDim[0]);
    ncout.createDimension('param',JacDim[1]);
    
        


    S = ncout.createVariable('site',str,('data'), zlib=zlib_in,shuffle=shuffle_in)
    C = ncout.createVariable('comp',str,('data'), zlib=zlib_in,shuffle=shuffle_in)

    Per = ncout.createVariable('Per','float64',('data'), zlib=zlib_in,shuffle=shuffle_in)
    Lat = ncout.createVariable('Lat','float64',('data'), zlib=zlib_in,shuffle=shuffle_in)
    Lon = ncout.createVariable('Lon','float64',('data'), zlib=zlib_in,shuffle=shuffle_in)
    X = ncout.createVariable('X','float64',('data'), zlib=zlib_in,shuffle=shuffle_in)
    Y = ncout.createVariable('Y','float64',('data'), zlib=zlib_in,shuffle=shuffle_in)
    Z = ncout.createVariable('Z','float64',('data'), zlib=zlib_in,shuffle=shuffle_in)
    Val = ncout.createVariable('Val','float64',('data'), zlib=zlib_in,shuffle=shuffle_in)
    Err = ncout.createVariable('Err','float64',('data'), zlib=zlib_in,shuffle=shuffle_in)
    
    J = ncout.createVariable('Jac','float64',('data','param'), zlib=zlib_in,shuffle=shuffle_in)
    
    S[:] = Site[:,]
    C[:] = Comp[:,]
    Per[:] = Dat[:,0]
    Lat[:] = Dat[:,1] 
    Lon[:] = Dat[:,2]
    X[:] = Dat[:,3]
    Y[:] = Dat[:,4]
    Z[:] = Dat[:,5]
    Val[:] = Dat[:,6]
    Err[:] = Dat[:,7]
    J[:] = Jac
 
    ncout.close()
      
    if out:
        print('writeJacHD: data written to %s in %s format'%(NCFile,ncout.data_model))
  

def readDat(DatFile=None, out = True):
    Data = []
    Site = []
    Comp = []
    Head = []
        
    with open(DatFile) as fd:
        for line in fd:
            if line.startswith('#') or line.startswith('>'):
                Head.append(line)
                continue

            
            t= line.split()
            
                       
            if 'PT' in t[7] or 'RH' in t[7] or 'PH' in t[7] :
                 tmp1 = [float(t[0]),float(t[2]),float(t[3]),float(t[4]),
                   float(t[5]),float(t[6]),float(t[8]),float(t[9])] 
                 Data.append(tmp1)
                 Site.append([t[1]])
                 Comp.append([t[7]])
            else:
                 tmp1 = [float(t[0]),float(t[2]),float(t[3]),float(t[4]),
                       float(t[5]),float(t[6]),float(t[8]),float(t[10])]
                 Data.append(tmp1)
                 tmp2 = [float(t[0]),float(t[2]),float(t[3]),float(t[4]),
                       float(t[5]),float(t[6]),float(t[9]),float(t[10])]
                 Data.append(tmp2)
                 Comp.append([t[7]+'R',t[7]+'I'])
                 Site.append([t[1],t[1]])
   

    Site = [item for sublist in Site for item in sublist]
    Site = np.asarray(Site,dtype=object)
    Comp = [item for sublist in Comp for item in sublist]
    Comp = np.asarray(Comp,dtype=object)
    
    Data = np.asarray(Data)
    
    nD   = np.shape(Data) 
    if out:
        print('readDat: %i data read from %s'%(nD[0], DatFile))
    
    return Site, Comp, Data, Head

    
def writeDatNC(NCFile=None, Dat= None, 
            Site= None, Comp=None, zlib_in=True, shuffle_in=True, out=True):
    """
    Writes Jacobian from ModEM output
    to NETCDF file
    author: vrath
    last changed: July 24, 2020
    """
    try: NCFile.close
    except: pass

    DatDim = np.shape(Dat)
    
       
    ncout = nc.Dataset(NCFile,'w', format='NETCDF4');
    ncout.createDimension('data',DatDim[0]);
    

    S = ncout.createVariable('site',str,('data',), zlib=zlib_in,shuffle=shuffle_in)
    C = ncout.createVariable('comp',str,('data',), zlib=zlib_in,shuffle=shuffle_in)

    Per = ncout.createVariable('Per','float64',('data',), zlib=zlib_in,shuffle=shuffle_in)
    Lat = ncout.createVariable('Lat','float64',('data',), zlib=zlib_in,shuffle=shuffle_in)
    Lon = ncout.createVariable('Lon','float64',('data',), zlib=zlib_in,shuffle=shuffle_in)
    X = ncout.createVariable('X','float64',('data',), zlib=zlib_in,shuffle=shuffle_in)
    Y = ncout.createVariable('Y','float64',('data',), zlib=zlib_in,shuffle=shuffle_in)
    Z = ncout.createVariable('Z','float64',('data',), zlib=zlib_in,shuffle=shuffle_in)
    Val = ncout.createVariable('Val','float64',('data',), zlib=zlib_in,shuffle=shuffle_in)
    Err = ncout.createVariable('Err','float64',('data',), zlib=zlib_in,shuffle=shuffle_in)
     
    
  
    S[:] = Site[:,]
    C[:] = Comp[:,]
    Per[:] = Dat[:,0]
    Lat[:] = Dat[:,1] 
    Lon[:] = Dat[:,2]
    X[:] = Dat[:,3]
    Y[:] = Dat[:,4]
    Z[:] = Dat[:,5]
    Val[:] = Dat[:,6]
    Err[:] = Dat[:,7]
 
    ncout.close()
     
    if out:
        print('writeDatNC: data written to %s in %s format'%(NCFile,ncout.data_model))
 

def writeModNC(NCFile=None, x=None, y=None, z=None, Mod=None, Sens= None, Ref=None, trans="LINEAR",
             zlib_in=True, shuffle_in=True, out = True):
    """
    Writes Model from ModEM output
    to NETCDF/HDF5 file
    author: vrath
    last changed: Dec 20, 2020
    """
  

    ModDim = np.shape(Mod)
   
        
    ncout = nc.Dataset(NCFile,'w', format='NETCDF4');
    
    
    ncout.createDimension('msiz',ModDim);
    ncout.createDimension('nx',ModDim[0]);
    ncout.createDimension('ny',ModDim[1]);
    ncout.createDimension('nz',ModDim[2]);
    
    ncout.createDimension('ref',(3));
    
    X = ncout.createVariable('x','float64',('nx'), zlib=zlib_in,shuffle=shuffle_in)
    Y = ncout.createVariable('y','float64',('ny'), zlib=zlib_in,shuffle=shuffle_in)
    Z = ncout.createVariable('z','float64',('nz'), zlib=zlib_in,shuffle=shuffle_in)
    X[:] = x[:]
    Y[:] = y[:]
    Z[:] = z[:]
    
    trans = trans.upper()    
    
    if   trans == 'LOGE':
        Mod = np.log(Mod)
        if out: print('resistivities to '+NCFile+' transformed to: '+trans)  
    elif trans == 'LOG10':
        Mod = np.log10(Mod)
        if out: print('resistivities to '+NCFile+' transformed to: '+trans)  
    elif trans == 'LINEAR' :
        pass
    else:
        print('Transformation: '+trans+' not defined!')   
        sys.exit(1)
        
   
    M = ncout.createVariable('model',str,('msiz'), zlib=zlib_in,shuffle=shuffle_in)    
    M = Mod

    if Sens != None:
        S = ncout.createVariable('sens',str,('msiz'), zlib=zlib_in,shuffle=shuffle_in)
        S = Sens
    
    if Ref != None:
        R = ncout.createVariable('ref','float64',('ref'), zlib=zlib_in,shuffle=shuffle_in)
        R = Ref
    
    ncout.close()
      
    if out:
        print('writeModNC: data written to %s in %s format'%(NCFile,ncout.data_model))
  

def writeMod(ModFile=None, 
             dx=None, dy=None, dz=None, rho=None,reference=None, trans='LINEAR',
             out = True):
    """
    Reads ModEM model input
    Expects rho in physical units
    
    author: vrath
    last changed: Aug 18, 2020
    """  
    
    dims = np.shape(rho)
    nx = dims[0]
    ny = dims[1] 
    nz = dims[2]
    
    trans = trans.upper()    
    dummy = 0
    
    if   trans == 'LOGE':
        rho = np.log(rho)
        if out: print('resistivities to '+ModFile+' transformed to: '+trans)  
    elif trans == 'LOG10':
        rho = np.log10(rho)
        if out: print('resistivities to '+ModFile+' transformed to: '+trans)  
    elif trans == 'LINEAR' :
        pass
    else:
        print('Transformation: '+trans+' not defined!')   
        sys.exit(1)
    trans=np.array(trans)
    with open(ModFile,'w') as f:
        np.savetxt(f,[' # 3D MT model written by ModEM in WS format'],fmt='%s')
        # line = np.array([nx, ny,nz, dummy, trans],dtype=('i8,i8,i8,i8,U10'))
        line = np.array([nx,ny,nz,dummy,trans])
        np.savetxt(f, line.reshape(1,5), fmt = '%s %s %s %s %s')
        np.savetxt(f,dx.reshape(1,dx.shape[0]), fmt='%12.3f')
        np.savetxt(f,dy.reshape(1,dy.shape[0]), fmt='%12.3f')
        np.savetxt(f,dz.reshape(1,dz.shape[0]), fmt='%12.3f')
        # write out the layers from resmodel
        for zi in range(dz.size):
            f.write('\n')
            for yi in range(dy.size):
                line = rho[:, yi, zi]
                np.savetxt(f, line.reshape(1,nx),fmt='%12.5e')

        f.write('\n')
        
        cnt=np.asarray(reference)
        np.savetxt(f,cnt.reshape(1,cnt.shape[0]), fmt='%10.1f')   
        f.write('%10.2f  \n' % (0.))
    
    
def readMod(ModFile=None, trans="LINEAR",out = True):
    """
    Reads ModEM model input
    returns rho in physical units
    
    author: vrath
    last changed: Aug 18, 2020
    
    """  
         
        
    with open(ModFile,'r') as f:
        lines = f.readlines()

    lines = [l.split() for l in lines]
    dims = [int(sub) for sub in lines[1][0:3]]
    nx, ny,nz  = dims
    trns =  lines[1][4]
    dx   = np.array([float(sub) for sub in lines[2]])
    dy   = np.array([float(sub) for sub in lines[3]])
    dz   = np.array([float(sub) for sub in lines[4]])
    
    rho  = np.array([])
    for line in lines[5:-2]:
        rho = np.append(rho,np.array([float(sub) for sub in line]))
        
        
    if out: print('resistivities in '+ModFile+' are: '+trns)   
    if  trns == 'LOGE':
        rho = np.exp(rho)
    elif trns == 'LOG10' :
        rho = np.power(10.,rho)
    elif trns == 'LINEAR' :
        pass
    else:
        print('Transformation: '+trns+' not defined!')   
        sys.exit(1)
    
    # here rho should be in physical units, not log...
        
    if  trans == 'LOGE':
        rho = np.log(rho)
        if out: print('resistivities transformed to: '+trans)
    elif trans == 'LOG10' :
        rho = np.log10(rho)
        if out: print('resistivities transformed to: '+trans)
    else:
        if out: print('resistivities transformed to: '+trans)
        pass

    
    rho = rho.reshape(dims,order='F')
    

    reference =  [float(sub) for sub in lines[-2][0:3]]

  
    
    if out: print('readMod: %i x %i x %i model read from %s'%(nx,ny,nz,ModFile))
    
    return dx, dy,dz, rho,reference


def mt1dfwd(freq, sig, d, inmod='r',out = "imp"):
    
    """
    1D magnetotelluric forward modelling
    based on A. Pethik's script at www.digitalearthlab.com
    Last change vr Nov 20, 2020
    """
        
    mu0 = 4.E-7*np.pi   		# Magnetic Permeability (H/m)
    
    sig    = np.array(sig)
    freq   = np.array(freq)
    d      = np.array(d)
    
    if inmod[0] == "c":
      sig    = np.array(sig)
    elif inmod[0]=='r':
      sig    = 1./np.array(sig)
      
      
    if sig.ndim >1:
        error('IP not yet implemented')
        
    n = np.size(sig)
    
    
    Z = np.zeros_like(freq)+1j*np.zeros_like(freq)
    w = np.zeros_like(freq)
    
    ifr=-1
    for f in freq:   
        ifr = ifr+1
        w[ifr] =  2.*np.pi*f       
        imp = np.array(range(n))+np.array(range(n))*1j
        
        #compute basement impedance
        imp[n-1] = np.sqrt(1j*w[ifr]*mu0/sig[n-1])
        
        for layer in range(n-2,-1,-1):
            sl   = sig[layer]
            dl   = d[layer]
            # 3. Compute apparent rho from top layer impedance
            #Step 2. Iterate from bottom layer to top(not the basement) 
            #   Step 2.1 Calculate the intrinsic impedance of current layer
            dj = np.sqrt(1j * w[ifr] * mu0 * sl)
            wj = dj/sl
            #   Step 2.2 Calculate Exponential factor from intrinsic impedance
            ej = np.exp(-2*dl*dj)                     
        
            #   Step 2.3 Calculate reflection coeficient using current layer
            #          intrinsic impedance and the below layer impedance 
            impb = imp[layer + 1]
            rj = (wj - impb)/(wj + impb)
            re = rj*ej 
            Zj = wj * ((1 - re)/(1 + re))
            imp[layer] = Zj    
    
        Z[ifr] = imp[0]
        # print(Z[ifr])

    if 	 out.lower() == "imp":        
        
        return Z
    
    elif out.lower() == "rho":
        absZ = np.abs(Z)
        rhoa 	= (absZ*absZ) / (mu0*w)
        phase   = np.rad2deg(np.arctan(Z.imag / Z.real))
        
        return rhoa, phase
    else:
        absZ = np.abs(Z)
        rhoa 	= (absZ*absZ) / (mu0*w)
        phase   = np.rad2deg(np.arctan(Z.imag / Z.real))
        return Z, rhoa, phase
        
