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
from numpy.linalg import norm
from scipy.io import FortranFile
from scipy.ndimage import laplace, convolve, gaussian_gradient_magnitude
from scipy.ndimage import uniform_filter, gaussian_filter



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



    ncout = nc.Dataset(NCFile,'w', format='NETCDF4')
    ncout.createDimension('data',JacDim[0])
    ncout.createDimension('param',JacDim[1])




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


    ncout = nc.Dataset(NCFile,'w', format='NETCDF4')
    ncout.createDimension('data',DatDim[0])


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


    ncout = nc.Dataset(NCFile,'w', format='NETCDF4')


    ncout.createDimension('msiz',ModDim)
    ncout.createDimension('nx',ModDim[0])
    ncout.createDimension('ny',ModDim[1])
    ncout.createDimension('nz',ModDim[2])

    ncout.createDimension('ref',(3))

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


    In Fortran:

    DO iz = 1,Nz
        DO iy = 1,Ny
            DO ix = Nx,1,-1
                READ(10,*) rho(ix,iy,iz)
            ENDDO
        ENDDO
    ENDDO

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
                # line = rho[::-1, yi, zi]
                # line = np.flipud(rho[:, yi, zi])
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

    In Fortran:

    DO iz = 1,Nz
        DO iy = 1,Ny
            DO ix = Nx,1,-1
                READ(10,*) rho(ix,iy,iz)
            ENDDO
        ENDDO
    ENDDO

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



def insert_body(dx=None,dy=None,dz=None,rho_in=None,body=None,
                pad=[-1,-1,-1], smooth=['gaussian',1.],scale=1.,Out=True):
    """
    Inserts 3d ellipsoid or box into given model

    Created on Sun Jan 3 10:35:28 2021
    @author: vrath
    """
    xpad = pad[0]
    ypad = pad[1]
    zpad = pad[2]

    xc, yc, zc =  centers3d(dx,dy,dz)

    nx = np.shape(xc)[0]
    ny = np.shape(yc)[0]
    nz = np.shape(zc)[0]

    rho_out = rho_in

    action  = body[0]
    rhoval  = body[1]
    bcent   = body[2:5]
    baxes   = body[5:8]
    bangl   = body[8:11]

    if Out:
        print('Body type   : '+action+' with rho =',str(rhoval)+' Ohm.m')
        print('Body center : '+str(bcent))
        print('Body axes   : '+str(baxes))
        print('Body angles : '+str(bangl))
        print('Smoothed with '+smooth[0]+' filter, gstd = '+str(smooth[1]))

    if action[0:3] == 'ell':

        for kk in np.arange(0,nz-zpad-1):
            zpoint=zc[kk]
            for jj in np.arange(ypad+1,ny-ypad-1):
                ypoint=yc[jj]
                for ii  in np.arange(xpad+1,nx-xpad-1):
                    xpoint=xc[ii]
                    position = [xpoint,ypoint,zpoint]
                    if in_ellipsoid(position, bcent, baxes, bangl):
                        rho_out[ii,jj,kk] = rhoval


    if action[0:3] == 'box':

        for kk in np.arange(0,nz-zpad-1):
            zpoint=zc[kk]
            for jj in np.arange(ypad+1,ny-ypad-1):
                ypoint=yc[jj]
                for ii  in np.arange(xpad+1,nx-xpad-1):
                    xpoint=xc[ii]
                    position = [xpoint,ypoint,zpoint]
                    if in_box(position, bcent, baxes, bangl):
                        rho_out[ii,jj,kk] = rhoval

    if smooth != None:
        tmp = np.log10(rho_out)
        if  smooth[0][0:3]=='box':
            fsize = smooth[1]
            tmp = uniform_filter(tmp,fsize)
            rho_out = np.power(10.,tmp)
        elif smooth[0][0:3]=='gau':
            gstd = smooth[1]
            tmp = gaussian_filter(tmp,gstd)
            rho_out = np.power(10.,tmp)
        else:
            error('Smoothing filter  '+smooth[0]+' not implemented! Exit.')

    return rho_out

def centers3d(dx,dy,dz):
    '''
    defines cell centers
    Created on Sat Jan 2 10:35:28 2021
@author: vrath

    '''
    x = np.append( 0., np.cumsum(dx))
    xc = 0.5*(x[:-1]+x[1:])
    y = np.append( 0., np.cumsum(dy))
    yc = 0.5*(y[:-1]+y[1:])
    z = np.append( 0., np.cumsum(dz))
    zc = 0.5*(z[:-1]+z[1:])
    return xc, yc,zc

def in_ellipsoid(point=None, cent=[0.,0.,0.], axs=[1.,1.,1.], ang=[0.,0.,0.], find_inside=True):
    '''
    Finds points inside arbitrary ellipsoid, defined by the 3-vectors
    ellcent, ellaxes, elldir.
    vr dec 2020
    '''


    # subtract center
    p = np.array(point)-np.array(cent)
    # rotation matrices
    rz = rotz(ang[2])
    p = np.dot(rz,p)
    ry = roty(ang[1])
    p = np.dot(ry,p)
    rx = rotx(ang[0])
    p = np.dot(rx,p)
    # R = rz*ry*rx
    # p = R*p

    # position in ellipsoid coordinates

    p = p/axs

    t = p[0]*p[0] + p[1]*p[1] + p[2]*p[2] < 1.

    if not find_inside:
        t = not t

    return t


def in_box(point=None, cent=[0.,0.,0.], axs=[1.,1.,1.], ang=[0.,0.,0.], find_inside=True):
    '''
    Finds points inside arbitrary ellipsoid, defined by the 3-vectors
    ellcent, ellaxes, elldir.
    vr dec 2020
    '''
    # subtract center
    p = np.array(point)-np.array(cent)
    # rotation matrices
    rz = rotz(ang[2])
    p = np.dot(rz,p)
    ry = roty(ang[1])
    p = np.dot(ry,p)
    rx = rotx(ang[0])
    p = np.dot(rx,p)
    # R = rz*ry*rx
    # p = R*p

    # position in ellipsoid coordinates

    p = p/axs

    t = (p[0] <= 1. and  p[0] >= -1. and
         p[1] <= 1. and  p[1] >= -1. and
         p[2] <= 1. and  p[2] >= -1.)

    if not find_inside:
        t = not t


def rotz(theta):
    '''
    calculates 3x3 rotation matriz for rotation around z axis
    vr dec 2020
    '''
    t = np.radians(theta)
    s = np.sin(t)
    c = np.cos(t)

    M = np.array([c, -s,  0.,   s, c, 0.,   0., 0., 1.]).reshape(3,3)

    return M

def roty(theta):
    '''
    calculates 3x3 rotation matrix for rotationa around y axis
    vr dec 2020
    '''
    t = np.radians(theta)
    s = np.sin(t)
    c = np.cos(t)

    M = np.array([c, 0., s,   0., 1., 0.,   -s, 0., c]).reshape(3,3)

    return M

def rotx(theta):
    '''
    calculates 3x3 rotation matriz for rotation around x axis
    vr dec 2020
    '''
    t = np.radians(theta)
    s = np.sin(t)
    c = np.cos(t)

    M = np.array([1., 0., 0.,   0., c, -s,   0., s, c]).reshape(3,3)

    return M


def shock3d(M,dt=0.25,maxit=30,filt=[3,3,3,1.],signfunc=None):
    '''
    Simple Shock Filter in nD

    vr  Jan 2021
    '''
    if   signfunc== None or  signfunc== 'sign':
        signcall = 'S = -np.sign(L)'

    elif signfunc[0] == 'sigmoid':

        signcall = '-1./(1. + np.exp(-scale *L))'

    else:
        error('sign func '+signfunc+' not defined! Exit.')

    kersiz = (filt[0],filt[1],filt[2])
    kerstd =   filt[3]
    K = gauss3D(kersiz,kerstd)

    G = M

    for it in range(maxit):

        G = convolve(G,K)
        g = gaussian_gradient_magnitude(G)
        normxyz=norm(g)
        L = laplace(G)

        S = eval(signcall)

        G=G+dt*normxyz*S


    return G


def gauss3D(Kshape=(3,3,3),Ksigma=1.):
    '''
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussiam',[shape],[sigma])

    vr  Jan 2021
    '''
    k,m,n = [(ss-1.)/2. for ss in Kshape]
    z,y,x = np.ogrid[-k:k+1,-m:m+1,-k:k+1]
    h = np.exp( -(x*x + y*y +z*z) / (2.*Ksigma*Ksigma) )
    h[h < np.finfo(h.dtype).eps*h.max() ] = 0
    s = h.sum()
    if s!= 0:
        h /= s

    K = h

    return K
