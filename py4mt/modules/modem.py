# -*- coding: utf-8 -*-
import os
import sys
from sys import exit as error
import numpy as np
from numpy.linalg import norm
from scipy.io import FortranFile
from scipy.ndimage import laplace, convolve
from scipy.ndimage import uniform_filter, gaussian_filter, median_filter


# import scipy.sparse as scp

import netCDF4 as nc

# import h5netcdf as hc

def decode_h2(strng):
    """
    Decode header2 string from ModEM Jacobian (old style).

    ----------
    strng : string
       header string

    Returns
    -------
    i1, i2, i3 : integer
        frequency, dattype, site numbers

    """

    s = strng.replace(";","").split()
    i1 = int(s[3])
    i2 = int(s[5])
    i3 = int(s[7])

    ivals = [i1, i2, i3]
    return ivals

def read_jac(JacFile=None, out=False):
    """
    Read Jacobian from ModEM output.

    author: vrath
    last changed: Feb 10, 2021
    """
    if out:
        print("Opening and reading " + JacFile)

    fjac = FortranFile(JacFile, "r")
    tmp1 = []
    tmp2 = []

    _ = fjac.read_record(np.byte)
    # h1 = ''.join([chr(item) for item in header1])
    # print(h1)
    _ = fjac.read_ints(np.int32)
    # nAll = fjac.read_ints(np.int32)
    # print("nAll"+str(nAll))
    nTx = fjac.read_ints(np.int32)
    # print("ntx"+str(nTx))
    for i1 in range(nTx[0]):
        nDt = fjac.read_ints(np.int32)
        # print("nDt"+str(nDt))
        for i2 in range(nDt[0]):
            nSite = fjac.read_ints(np.int32)
            # print("nSite"+str(nSite))
            for i3 in range(nSite[0]):
                # header2
                header2 = fjac.read_record(np.byte)
                h2 = ''.join([chr(item) for item in header2])
                tmp2.append(decode_h2(h2))
                # print(h2)
                # print(i1,i2,i3)
                nSigma = fjac.read_ints(np.int32)
                # print("nSigma"+str(nSigma))
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
                    ColJac = fjac.read_reals(np.float64)
                    #ColJac = fjac.read_reals(np.float64).flatten()
                    # print(np.shape(CellSens))
                    # ColJac =  CellSens.flatten(order='F')
                    # Coljac = np.fromfile(file, dtype=np.float6)
                    tmp1.append(ColJac)
                    # print(np.shape(tmp1))
                    # tmp2.append()
    Jac = np.asarray(tmp1)
    Inf = np.asarray(tmp2)
#    Inf = np.asarray(tmp2,dtype=object)

    fjac.close()

    if out:
        print("...done reading " + JacFile)

    return Jac, Inf  #, Site, Freq, Comp


def read_data_jac(DatFile=None, out=True):
    """
    Read ModEM input data.

    author: vrath
    last changed: Feb 10, 2021
    """
    Data = []
    Site = []
    Comp = []
    Head = []

    with open(DatFile) as fd:
        for line in fd:
            if line.startswith("#") or line.startswith(">"):
                Head.append(line)
                continue

            t = line.split()
            # print(t)
            if t:
                if "PT" in t[5] or "RH" in t[5] or "PH" in t[5]:
                    tmp1 = [
                        float(t[0]),
                        float(t[2]),
                        float(t[3]),
                        float(t[4]),
                        float(t[6]),
                        float(t[7]),
                    ]
                    Data.append(tmp1)
                    Site.append([t[1]])
                    Comp.append([t[5]])
                else:
                    tmp1 = [
                        float(t[0]),
                        float(t[2]),
                        float(t[3]),
                        float(t[4]),
                        float(t[6]),
                        float(t[8]),
                    ]
                    Data.append(tmp1)
                    tmp2 = [
                        float(t[0]),
                        float(t[2]),
                        float(t[3]),
                        float(t[4]),
                        float(t[7]),
                        float(t[8]),
                    ]
                    Data.append(tmp2)
                    Comp.append([t[5] + "R", t[5] + "I"])
                    Site.append([t[1], t[1]])
            else:
                break

    Site = [item for sublist in Site for item in sublist]
    Site = np.asarray(Site, dtype=object)
    Comp = [item for sublist in Comp for item in sublist]
    Comp = np.asarray(Comp, dtype=object)

    Data = np.asarray(Data)
    Freq = Data[:,0]

    nD = np.shape(Data)
    if out:
        print("readDat: %i data read from %s" % (nD[0], DatFile))

    return Data, Site, Freq, Comp, Head


def write_jac_ncd(NCFile=None, Jac=None, Dat=None, Site=None, Comp=None,
               zlib_in=True, shuffle_in=True, out=True):
    """
    Write Jacobian from ModEM output to NETCDF/HDF5 file.

    author: vrath
    last changed: July 25, 2020
    """
    JacDim = np.shape(Jac)
    DatDim = np.shape(Dat)

    if JacDim[0] != DatDim[0]:
        print(
            "Error:  Jac dim="
            + str(JacDim[0])
            + " does not match Dat dim="
            + str(DatDim[0])
        )
        sys.exit(1)

    ncout = nc.Dataset(NCFile, "w", format="NETCDF4")
    ncout.createDimension("data", JacDim[0])
    ncout.createDimension("param", JacDim[1])

    S = ncout.createVariable(
        "site", str, ("data"), zlib=zlib_in, shuffle=shuffle_in)
    C = ncout.createVariable(
        "comp", str, ("data"), zlib=zlib_in, shuffle=shuffle_in)

    Per = ncout.createVariable(
        "Per", "float64", ("data"), zlib=zlib_in, shuffle=shuffle_in)
    Lat = ncout.createVariable(
        "Lat", "float64", ("data"), zlib=zlib_in, shuffle=shuffle_in)
    Lon = ncout.createVariable(
        "Lon", "float64", ("data"), zlib=zlib_in, shuffle=shuffle_in)
    X = ncout.createVariable(
        "X", "float64", ("data"), zlib=zlib_in, shuffle=shuffle_in)
    Y = ncout.createVariable(
        "Y", "float64", ("data"), zlib=zlib_in, shuffle=shuffle_in)
    Z = ncout.createVariable(
        "Z", "float64", ("data"), zlib=zlib_in, shuffle=shuffle_in)
    Val = ncout.createVariable(
        "Val", "float64", ("data"), zlib=zlib_in, shuffle=shuffle_in)
    Err = ncout.createVariable(
        "Err", "float64", ("data"), zlib=zlib_in, shuffle=shuffle_in)

    J = ncout.createVariable(
        "Jac", "float64", ("data", "param"), zlib=zlib_in, shuffle=shuffle_in)

    S[:] = Site[
        :,
    ]
    C[:] = Comp[
        :,
    ]
    Per[:] = Dat[:, 0]
    Lat[:] = Dat[:, 1]
    Lon[:] = Dat[:, 2]
    X[:] = Dat[:, 3]
    Y[:] = Dat[:, 4]
    Z[:] = Dat[:, 5]
    Val[:] = Dat[:, 6]
    Err[:] = Dat[:, 7]
    J[:] = Jac

    ncout.close()

    if out:
        print(
            "writeJacNC: data written to %s in %s format" %
            (NCFile, ncout.data_model)
        )



def read_data(DatFile=None, out=True):
    """
    Read ModEM input data.

    author: vrath
    last changed: Feb 10, 2021
    """
    Data = []
    Site = []
    Comp = []
    Head = []

    with open(DatFile) as fd:
        for line in fd:
            if line.startswith("#") or line.startswith(">"):
                Head.append(line)

                continue

            t = line.split()

            if "PT" in t[7] or "RH" in t[7] or "PH" in t[7]:
                tmp = [
                    float(t[0]),
                    float(t[2]),
                    float(t[3]),
                    float(t[4]),
                    float(t[5]),
                    float(t[6]),
                    float(t[8]),
                    float(t[9]),
                    0.,
                ]
                Data.append(tmp)
                Site.append([t[1]])
                Comp.append([t[7]])
            else:
                tmp = [
                    float(t[0]),
                    float(t[2]),
                    float(t[3]),
                    float(t[4]),
                    float(t[5]),
                    float(t[6]),
                    float(t[8]),
                    float(t[9]),
                    float(t[10]),
                ]
                Data.append(tmp)
                Comp.append([t[7]])
                Site.append([t[1]])


    Site = [item for sublist in Site for item in sublist]
    Site = np.asarray(Site, dtype=object)
    Comp = [item for sublist in Comp for item in sublist]
    Comp = np.asarray(Comp, dtype=object)
    Data = np.asarray(Data)


    nD = np.shape(Data)
    if out:
        print("readDat: %i data read from %s" % (nD[0], DatFile))

    return Site, Comp, Data, Head


def write_data(DatFile=None, Dat=None, Site=None, Comp=None, Head = None,
               out=True):
    """
    Write ModEM input data file.

    author: vrath
    last changed: Feb 10, 2021
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


def write_data_ncd(
        NCFile=None, Dat=None, Site=None, Comp=None,
        zlib_in=True, shuffle_in=True, out=True
        ):
    """
    Write Jacobian from ModEM output to NETCDF file.

    author: vrath
    last changed: July 24, 2020
    """
    try:
        NCFile.close
    except BaseException:
        pass

    DatDim = np.shape(Dat)

    ncout = nc.Dataset(NCFile, "w", format="NETCDF4")
    ncout.createDimension("data", DatDim[0])

    S = ncout.createVariable(
        "site", str, ("data",), zlib=zlib_in, shuffle=shuffle_in)
    C = ncout.createVariable(
        "comp", str, ("data",), zlib=zlib_in, shuffle=shuffle_in)

    Per = ncout.createVariable(
        "Per", "float64", ("data",), zlib=zlib_in, shuffle=shuffle_in
    )
    Lat = ncout.createVariable(
        "Lat", "float64", ("data",), zlib=zlib_in, shuffle=shuffle_in
    )
    Lon = ncout.createVariable(
        "Lon", "float64", ("data",), zlib=zlib_in, shuffle=shuffle_in
    )
    X = ncout.createVariable(
        "X", "float64", ("data",), zlib=zlib_in, shuffle=shuffle_in
    )
    Y = ncout.createVariable(
        "Y", "float64", ("data",), zlib=zlib_in, shuffle=shuffle_in
    )
    Z = ncout.createVariable(
        "Z", "float64", ("data",), zlib=zlib_in, shuffle=shuffle_in
    )
    Val = ncout.createVariable(
        "Val", "float64", ("data",), zlib=zlib_in, shuffle=shuffle_in
    )
    Err = ncout.createVariable(
        "Err", "float64", ("data",), zlib=zlib_in, shuffle=shuffle_in
    )

    S[:] = Site[
        :,
    ]
    C[:] = Comp[
        :,
    ]
    Per[:] = Dat[:, 0]
    Lat[:] = Dat[:, 1]
    Lon[:] = Dat[:, 2]
    X[:] = Dat[:, 3]
    Y[:] = Dat[:, 4]
    Z[:] = Dat[:, 5]
    Val[:] = Dat[:, 6]
    Err[:] = Dat[:, 7]

    ncout.close()

    if out:
        print(
            "writeDatNC: data written to %s in %s format"
            % (NCFile, ncout.data_model)
        )


def write_model_ncd(
    NCFile=None,
    x=None,
    y=None,
    z=None,
    Mod=None,
    Sens=None,
    Ref=None,
    trans="LINEAR",
    zlib_in=True,
    shuffle_in=True,
    out=True,
):
    """
    Write Model from ModEM output to NETCDF/HDF5 file.

    author: vrath
    last changed: Jan 21, 2021
    """
    ModDim = np.shape(Mod)

    ncout = nc.Dataset(NCFile, "w", format="NETCDF4")

    ncout.createDimension("msiz", ModDim)
    ncout.createDimension("nx", ModDim[0])
    ncout.createDimension("ny", ModDim[1])
    ncout.createDimension("nz", ModDim[2])

    ncout.createDimension("ref", (3))

    X = ncout.createVariable(
        "x", "float64", ("nx"), zlib=zlib_in, shuffle=shuffle_in)
    Y = ncout.createVariable(
        "y", "float64", ("ny"), zlib=zlib_in, shuffle=shuffle_in)
    Z = ncout.createVariable(
        "z", "float64", ("nz"), zlib=zlib_in, shuffle=shuffle_in)
    X[:] = x[:]
    Y[:] = y[:]
    Z[:] = z[:]

    trans = trans.upper()

    if trans == "LOGE":
        Mod = np.log(Mod)
        if out:
            print("resistivities to " + NCFile + " transformed to: " + trans)
    elif trans == "LOG10":
        Mod = np.log10(Mod)
        if out:
            print("resistivities to " + NCFile + " transformed to: " + trans)
    elif trans == "LINEAR":
        pass
    else:
        print("Transformation: " + trans + " not defined!")
        sys.exit(1)

    M = ncout.createVariable(
        "model", "float64", ("msiz"), zlib=zlib_in, shuffle=shuffle_in
    )
    M[:, :, :] = Mod[:, :, :]

    if Sens is not None:
        S = ncout.createVariable(
            "sens", "float64", ("msiz"), zlib=zlib_in, shuffle=shuffle_in
        )
        S[:, :, :] = Sens[:, :, :]

    if Ref is not None:
        R = ncout.createVariable(
            "ref", "float64", ("ref"), zlib=zlib_in, shuffle=shuffle_in
        )
        R[:, :, :] = Ref[:, :, :]

    ncout.close()

    if out:
        print(
            "write_modelNC: data written to %s in %s format"
            % (NCFile, ncout.data_model)
        )


# def write_model_vtk(ModFile=None, dx=None, dy=None, dz=None, rho=None, reference=None,
#                  out=True):
#     """
#     write ModEM model input in .

#     Expects rho in physical units

#     author: vrath
#     last changed: Mar 13, 2021

#     """
#     dims = np.shape(rho)
#     nx = dims[0]
#     ny = dims[1]
#     nz = dims[2]

#     with open(ModFile, "w") as f:
#         np.savetxt(
#             f, [" # 3D MT model written by ModEM in WS format"], fmt="%s")
#         # line = np.array([nx, ny,nz, dummy, trans],dtype=('i8,i8,i8,i8,U10'))
#         np.savetxt(f, dx.reshape(1, dx.shape[0]), fmt="%12.3f")
#         np.savetxt(f, dy.reshape(1, dy.shape[0]), fmt="%12.3f")
#         np.savetxt(f, dz.reshape(1, dz.shape[0]), fmt="%12.3f")
#         # write out the layers from resmodel
#         for zi in range(dz.size):
#             f.write("\n")
#             for yi in range(dy.size):
#                 # line = rho[::-1, yi, zi]
#                 # line = np.flipud(rho[:, yi, zi])
#                 line = rho[:, yi, zi]
#                 np.savetxt(f, line.reshape(1, nx), fmt="%12.5e")

#         f.write("\n")

#         cnt = np.asarray(reference)
#         np.savetxt(f, cnt.reshape(1, cnt.shape[0]), fmt="%10.1f")
#         f.write("%10.2f  \n" % (0.0))


def write_model(ModFile=None, dx=None, dy=None, dz=None, mval=None, reference=None,
                trans=None, aircells = None, mvalair = 1.e17, blank = 1.e17, out=True):
    """
    Write ModEM model input.

    Expects mval in physical units (linear).

    author: vrath
    last changed: Aug 18, 2020


    In Fortran:

    DO iz = 1,Nz
        DO iy = 1,Ny
            DO ix = Nx,1,-1
                READ(10,*) mval(ix,iy,iz)
            ENDDO
        ENDDO
    ENDDO

    """
    dims = np.shape(mval)

    nx = dims[0]
    ny = dims[1]
    nz = dims[2]
    dummy = 0

    if trans is not None:

        trans = trans.upper()

        if trans == "LOGE":
            mval = np.log(mval)
            mvalair = np.log(mvalair)
            if out:
                print("values to " + ModFile + " transformed to: " + trans)
        elif trans == "LOG10":
            mval = np.log10(mval)
            mvalair = np.log10(mvalair)
            if out:
                print("values to " + ModFile + " transformed to: " + trans)
        elif trans == "LINEAR":
            pass

        else:
            print("Transformation: " + trans + " not defined!")
            sys.exit(1)


    else:
        trans = "LINEAR"

    if not aircells == None:
        mval.reshape(dims)[aircells] = mvalair

    if not blank == None:
        blanks = np.where(~np.isfinite(mval))
        mval.reshape(dims)[blanks] = mvalair

    trans = np.array(trans)
    with open(ModFile, "w") as f:
        np.savetxt(
            f, ["# 3D MT model written by ModEM in WS format"], fmt="%s")
        line = np.array([nx, ny,nz, dummy, trans],dtype="object")
        # line = np.array([nx, ny, nz, dummy, trans])
        # np.savetxt(f, line.reshape(1, 5), fmt="   %s"*5)
        np.savetxt(f, line.reshape(1, 5), fmt =["  %i","  %i","  %i","  %i", "  %s"])

        np.savetxt(f, dx.reshape(1, dx.shape[0]), fmt="%12.3f")
        np.savetxt(f, dy.reshape(1, dy.shape[0]), fmt="%12.3f")
        np.savetxt(f, dz.reshape(1, dz.shape[0]), fmt="%12.3f")
        # write out the layers from resmodel
        for zi in range(dz.size):
            f.write("\n")
            for yi in range(dy.size):
                line = mval[::-1, yi, zi]
                # line = np.flipud(mval[:, yi, zi])
                # line = mval[:, yi, zi]
                np.savetxt(f, line.reshape(1, nx), fmt="%12.5e")

        f.write("\n")

        cnt = np.asarray(reference)
        np.savetxt(f, cnt.reshape(1, cnt.shape[0]), fmt="%10.1f")
        f.write("%10.2f  \n" % (0.0))


def read_model(ModFile=None, trans="LINEAR", volumes=False, out=True):
    """
    Read ModEM model input.

    Returns mval in physical units

    author: vrath
    last changed: Aug 18, 2020

    In Fortran:

    DO iz = 1,Nz
        DO iy = 1,Ny
            DO ix = Nx,1,-1
                READ(10,*) mval(ix,iy,iz)
            ENDDO
        ENDDO
    ENDDO

    """
    with open(ModFile, "r") as f:
        lines = f.readlines()

    lines = [line.split() for line in lines]
    dims = [int(sub) for sub in lines[1][0:3]]
    nx, ny, nz = dims
    trns = lines[1][4]
    dx = np.array([float(sub) for sub in lines[2]])
    dy = np.array([float(sub) for sub in lines[3]])
    dz = np.array([float(sub) for sub in lines[4]])

    mval = np.array([])
    for line in lines[5:-2]:
        line = np.flipud(line) #  np.fliplr(line)
        mval = np.append(mval, np.array([float(sub) for sub in line]))

    if out:
        print("values in " + ModFile + " are: " + trns)
    if trns == "LOGE":
        mval = np.exp(mval)
    elif trns == "LOG10":
        mval = np.power(10.0, mval)
    elif trns == "LINEAR":
        pass
    else:
        print("Transformation: " + trns + " not defined!")
        sys.exit(1)

    # here mval should be in physical units, not log...
    if "loge" in trans.lower() or "ln" in trans.lower():
        mval = np.log(mval)
        if out:
            print("values transformed to: " + trans)
    elif "log10" in trans.lower():
        mval = np.log10(mval)
        if out:
            print("values transformed to: " + trans)
    else:
        if out:
            print("values transformed to: " + trans)
        pass

    mval = mval.reshape(dims, order="F")

    reference = [float(sub) for sub in lines[-2][0:3]]

    if out:
        print(
            "read_model: %i x %i x %i model read from %s" % (nx, ny, nz, ModFile))

    if volumes:
        vcell = np.zeros_like(mval)
        for ii in np.arange(len(dx)):
            for jj in np.arange(len(dy)):
                for kk in np.arange(len(dz)):
                    vcell[ii,jj,kk] = dx[ii]*dy[jj]*dz[kk]

        if out:
            print(
                "read_model: %i x %i x %i cell volumes calculated" % (nx, ny, nz))

        return dx, dy, dz, mval, reference, trans, vcell



    else:
        return dx, dy, dz, mval, reference, trans


def proc_covar(covfile_i=None, 
               covfile_o=None,
               modsize=[], 
               fixed="2", 
               method = "border",
               border=5, 
               fixdist= 30000.,
               cellcent = [ np.array([]), np.array([])],
               sitepos = [ np.array([]), np.array([])],  
               unit = "km",
               out=True):
    """
    Read and process ModEM covar input.

    author: vrath
    last changed: June, 2023

    """
    air = "0"
    ocean = "9"
    comments = ["#", "|",">", "+","/"]

    with open(covfile_i, "r") as f_i: 
       l_i = f_i.readlines()
       
       
       
    l_o = l_i.copy()
 
    
    done = False
    for line in l_i:
        if len(line.split())==3:
                [block_len, line_len, num_lay] =[int(t) for t in line.split()]
                print(block_len, line_len, num_lay)
                done = True
        if done: 
            break
        
    if "bord" in method.lower():
        rows = list(range(0, block_len))
        index_row1 = [index for index in rows if rows[index] < border]
        index_row2 = [index for index in rows if rows[index] > block_len-border-1]
        cols = list(range(0, line_len))
        index_col1 = [index for index in cols if cols[index] < border]        
        index_col2 = [index for index in cols if cols[index] > line_len-border-1]      
    if "dist" in method.lower():
         sits = list(range(0, np.shape(sitepos)[0]))
         rows = list(range(0, block_len)) 
         cols = list(range(0, line_len))
         xs = sitepos[:][0]
         ys = sitepos[:][1]
         
         
    blocks = [ii for ii in range(len(l_i)) if len(l_i[ii].split()) == 2]
    if len(blocks) != num_lay:
        error("read_covar: Number of blocks wrong! Exit.")
        
    for ib in blocks:
        new_block = []
        block = l_i[ib+1:ib+block_len+1] 
        tmp =[line.split() for line in block]
        
        if "bord" in method.lower():

            for ii in rows:
                if (ii in index_row1) or (ii in index_row2):
                      tmp[ii] = [tmp[ii][cell].replace("1", fixed) for cell in cols]
                      
                # print(ii,tmp[ii] ,"\n")
                
                for jj in cols:
                    if (jj in index_col1) or (jj in index_col2):
                        tmp[ii][jj] = tmp[ii][jj].replace("1", fixed)

                tmp[ii].append("\n")    
                new_block.append(" ".join(tmp[ii]))
                

        if "dist" in method.lower():
                
            for ii in rows:
                # print(ii)
                xc = cellcent[0][ii]
                for jj in cols:
                    yc = cellcent[1][jj]
                    dist = []
                    for kk in sits:
                        dist.append(np.sqrt((xc-xs)**2 + (yc-ys)**2))
                    
                    dmin = np.amin(dist)
                    if dmin > fixdist:
                        tmp[ii][jj] = tmp[ii][jj].replace("1", fixed)
                        
                tmp[ii].append("\n")    
                new_block.append(" ".join(tmp[ii]))
                             
            
        l_o[ib+1:ib+block_len+1] = new_block
            
    with open(covfile_o, "w") as f_o: 
            f_o.writelines(l_o)



    if out:
        print("read_covar: covariance matrix read from %s" % (covfile_i))
        print("read_covar: covariance matrix written to %s" % (covfile_o))
        if "bord" in method.lower():
            print(str(border)+" border  cells fixed (zone "+str(fixed)+")")
        else:
            if unit=="km":
                print("cells with min distance to site > "
                      +str(fixdist/1000)+"km fixed (zone "+str(fixed)+")")
            else:
                print("cells with min distance to site > "
                     +str(fixdist)+"m fixed (zone "+str(fixed)+")")
       
    return l_o

def linear_interpolation(p1, p2, x0):
    """
    Function that receives as arguments the coordinates of two points (x,y)
    and returns the linear interpolation of a y0 in a given x0 position. This is the
    equivalent to obtaining y0 = y1 + (y2 - y1)*((x0-x1)/(x2-x1)).
    Look into https://en.wikipedia.org/wiki/Linear_interpolation for more
    information.

    Parameters
    ----------
    p1     : tuple (floats)
        Tuple (x,y) of a first point in a line.
    p2     : tuple (floats)
        Tuple (x,y) of a second point in a line.
    x0     : float
        X coordinate on which you want to interpolate a y0.

    Return float (interpolated y0 value)
    """
    y0 = p1[1] + (p2[1] - p1[1]) * ((x0 - p1[0]) / (p2[0] - p1[0]))

    return y0


def clip_model(x, y, z, rho,
               pad=[0, 0, 0], centers=False, scale=[1., 1., 1.]):
    """
    Clip model to ROI.

    Parameters
    ----------
    x, y, z : float
        Node coordinates
    rho : float
        resistivity/sensitivity/diff values.
    pad : integer, optional
        padding in x/y/z. The default is [0, 0, 0].
    centers: bool, optional
        nodes or centers. The default is False (i.e. nodes).
    scale: float
        scling, e.g. to km (1E-3). The default is [1., 1.,1.].

    Returns
    -------
    xn, yn, zn, rhon

    """
    if np.size(scale) == 1:
        scale = [scale, scale, scale]

    p_x, p_y, p_z = pad
    s_x, s_y, s_z = scale

    xn = s_x * x[p_x:-p_x]
    yn = s_y * y[p_y:-p_y]
    zn = s_z * z[0:-p_z]
    rhon = rho[p_x:-p_x, p_y:-p_y, 0:-p_z]

    if centers:
        print("cells3d returning cell center coordinates.")
        xn = 0.5 * (xn[:-1] + xn[1:])
        yn = 0.5 * (yn[:-1] + yn[1:])
        zn = 0.5 * (zn[:-1] + zn[1:])

    return xn, yn, zn, rhon


def mt1dfwd(freq, sig, d, inmod="r", out="imp", magfield="b"):
    """
    Calulate 1D magnetotelluric forward response.

    based on A. Pethik's script at www.digitalearthlab.com
    Last change vr Nov 20, 2020
    """
    mu0 = 4.0e-7 * np.pi  # Magnetic Permeability (H/m)

    sig = np.array(sig)
    freq = np.array(freq)
    d = np.array(d)

    if inmod[0] == "c":
        sig = np.array(sig)
    elif inmod[0] == "r":
        sig = 1.0 / np.array(sig)

    if sig.ndim > 1:
        error("IP not yet implemented")

    n = np.size(sig)

    Z = np.zeros_like(freq) + 1j * np.zeros_like(freq)
    w = np.zeros_like(freq)

    ifr = -1
    for f in freq:
        ifr = ifr + 1
        w[ifr] = 2.0 * np.pi * f
        imp = np.array(range(n)) + np.array(range(n)) * 1j

        # compute basement impedance
        imp[n - 1] = np.sqrt(1j * w[ifr] * mu0 / sig[n - 1])

        for layer in range(n - 2, -1, -1):
            sl = sig[layer]
            dl = d[layer]
            # 3. Compute apparent rho from top layer impedance
            # Step 2. Iterate from bottom layer to top(not the basement)
            #   Step 2.1 Calculate the intrinsic impedance of current layer
            dj = np.sqrt(1j * w[ifr] * mu0 * sl)
            wj = dj / sl
            #   Step 2.2 Calculate Exponential factor from intrinsic impedance
            ej = np.exp(-2 * dl * dj)

            #   Step 2.3 Calculate reflection coeficient using current layer
            #          intrinsic impedance and the below layer impedance
            impb = imp[layer + 1]
            rj = (wj - impb) / (wj + impb)
            re = rj * ej
            Zj = wj * ((1 - re) / (1 + re))
            imp[layer] = Zj

        Z[ifr] = imp[0]
        # print(Z[ifr])

    if out.lower() == "imp":

        if magfield.lower() =="b":
            return Z/mu0
        else:
            return Z

    elif out.lower() == "rho":
        absZ = np.abs(Z)
        rhoa = (absZ * absZ) / (mu0 * w)
        phase = np.rad2deg(np.arctan(Z.imag / Z.real))

        return rhoa, phase
    else:
        absZ = np.abs(Z)
        rhoa = (absZ * absZ) / (mu0 * w)
        phase = np.rad2deg(np.arctan(Z.imag / Z.real))
        return Z, rhoa, phase


def insert_body(
    dx=None,
    dy=None,
    dz=None,
    rho_in=None,
    body=None,
    pad=[0, 0, 0],
    smooth=None,
    scale=1.0,
    Out=True,
):
    """
    Insert 3d ellipsoid or box into given model.

    Created on Sun Jan 3 10:35:28 2021
    @author: vrath
    """
    xpad = pad[0]
    ypad = pad[1]
    zpad = pad[2]

    xc, yc, zc = cells3d(dx, dy, dz, otype='c')

    modcenter = [0.5 * np.sum(dx), 0.5 * np.sum(dy), 0.0]

    xc = xc - modcenter[0]
    yc = yc - modcenter[1]
    zc = zc - modcenter[2]

    nx = np.shape(xc)[0]
    ny = np.shape(yc)[0]
    nz = np.shape(zc)[0]

    rho_out = np.log10(rho_in)

    geom = body[0]
    action = body[1]
    rhoval = body[2]
    bcent = body[3:6]
    baxes = body[6:9]
    bangl = body[9:12]

    if action[0:3] == "rep":
        actstring = "rhoval"
    elif action[0:3] == "add":
        actstring = "rho_out[ii,jj,kk] + rhoval"
    else:
        error("Action" + action + " not implemented! Exit.")

    if Out:
        print(
            "Body type   : " + geom + ", " + action + " rho =",
            str(np.power(10.0, rhoval)) + " Ohm.m",
        )
        print("Body center : " + str(bcent))
        print("Body axes   : " + str(baxes))
        print("Body angles : " + str(bangl))
        print("Smoothed with " + smooth[0] + " filter")

    if geom[0:3] == "ell":

        for kk in np.arange(0, nz - zpad - 1):
            zpoint = zc[kk]
            for jj in np.arange(ypad + 1, ny - ypad - 1):
                ypoint = yc[jj]
                for ii in np.arange(xpad + 1, nx - xpad - 1):
                    xpoint = xc[ii]
                    position = [xpoint, ypoint, zpoint]
                    # if Out:
                    # print('position')
                    # print(position)
                    # print( bcent)
                    if in_ellipsoid(position, bcent, baxes, bangl):
                        rho_out[ii, jj, kk] = eval(actstring)
                        # if Out:
                        #     print("cell %i %i %i" % (ii, jj, kk))

    if geom[0:3] == "box":

        for kk in np.arange(0, nz - zpad - 1):
            zpoint = zc[kk]
            for jj in np.arange(ypad + 1, ny - ypad - 1):
                ypoint = yc[jj]
                for ii in np.arange(xpad + 1, nx - xpad - 1):
                    xpoint = xc[ii]
                    position = [xpoint, ypoint, zpoint]
                    # if Out:
                    # print('position')
                    # print(position)
                    # print( bcent)

                    if in_box(position, bcent, baxes, bangl):
                        rho_out[ii, jj, kk] = eval(actstring)
                        # if Out:
                        #     print("cell %i %i %i" % (ii, jj, kk))

    if smooth is not None:
        if smooth[0][0:3] == "uni":
            fsize = smooth[1]
            rho_out = uniform_filter(rho_out, fsize)

        elif smooth[0][0:3] == "gau":
            gstd = smooth[1]
            rho_out = gaussian_filter(rho_out, gstd)

        else:
            error("Smoothing filter  " + smooth[0] + " not implemented! Exit.")

    rho_out = np.power(10.0, rho_out)

    return rho_out


def cells3d(dx, dy, dz, center=False, reference=[0., 0., 0.]):
    """
    Define cell coordinates.

    dx, dy, dz in m,
    Created on Sat Jan 2 10:35:28 2021

    @author: vrath

    """
    x = np.append(0.0, np.cumsum(dx))
    y = np.append(0.0, np.cumsum(dy))
    z = np.append(0.0, np.cumsum(dz))

    x = x + reference[0]
    y = y + reference[1]
    z = z + reference[2]

    if center:
        print("cells3d returning cell center coordinates.")
        xc = 0.5 * (x[:-1] + x[1:])
        yc = 0.5 * (y[:-1] + y[1:])
        zc = 0.5 * (z[:-1] + z[1:])
        return xc, yc, zc

    else:
        print("cells3d returning node coordinates.")
        return x, y, z


def in_ellipsoid(
    point=None,
    cent=[0.0, 0.0, 0.0],
    axs=[1.0, 1.0, 1.0],
    ang=[0.0, 0.0, 0.0],
    find_inside=True,
):
    """
    Find points inside arbitrary box.

    Defined by the 3-vectors cent, axs, ang
    vr dec 2020

    """
    # subtract center
    p = np.array(point) - np.array(cent)
    # rotation matrices
    rz = rotz(ang[2])
    p = np.dot(rz, p)
    ry = roty(ang[1])
    p = np.dot(ry, p)
    rx = rotx(ang[0])
    p = np.dot(rx, p)
    # R = rz*ry*rx
    # p = R*p

    # position in ellipsoid coordinates

    p = p / axs

    t = p[0] * p[0] + p[1] * p[1] + p[2] * p[2] < 1.0
    # print(p,t)
    if not find_inside:
        t = not t

    return t


def in_box(
    point=None,
    cent=[0.0, 0.0, 0.0],
    axs=[1.0, 1.0, 1.0],
    ang=[0.0, 0.0, 0.0],
    find_inside=True,
):
    """
    Find points inside arbitrary ellipsoid.

    Defined by the 3-vectors cent, axs, ang
    vr dec 2020

    """
    # subtract center
    p = np.array(point) - np.array(cent)
    # rotation matrices
    rz = rotz(ang[2])
    p = np.dot(rz, p)
    ry = roty(ang[1])
    p = np.dot(ry, p)
    rx = rotx(ang[0])
    p = np.dot(rx, p)
    # R = rz*ry*rx
    # p = R*p

    # position in ellipsoid coordinates

    p = p / axs

    t = (
        p[0] <= 1.0
        and p[0] >= -1.0
        and p[1] <= 1.0
        and p[1] >= -1.0
        and p[2] <= 1.0
        and p[2] >= -1.0
    )
    # print(p,t)

    if not find_inside:
        t = not t

    return t


def rotz(theta):
    """
    Calculate 3x3 rotation matriz for rotation around z axis.

    vr dec 2020
    """
    t = np.radians(theta)
    s = np.sin(t)
    c = np.cos(t)

    M = np.array([c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)

    return M


def roty(theta):
    """
    Calculate 3x3 rotation matrix for rotationa around y axis.

    vr dec 2020
    """
    t = np.radians(theta)
    s = np.sin(t)
    c = np.cos(t)

    M = np.array([c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c]).reshape(3, 3)

    return M


def rotx(theta):
    """
    Calculate 3x3 rotation matriz for rotation around x axis.

    vr dec 2020
    """
    t = np.radians(theta)
    s = np.sin(t)
    c = np.cos(t)

    M = np.array([1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c]).reshape(3, 3)

    return M

def crossgrad(m1=np.array([]), 
                m2=np.array([]),  
                mesh= [np.array([]), np.array([]), np.array([])],
                Out=True):
    """
    
    Crossgrad function
    
    
    See:
    Rosenkjaer GK, Gasperikova E, Newman, GA, Arnason K, Lindsey NJ (2015) 
        Comparison of 3D MT inversions for geothermal exploration: Case studies 
        for Krafla and Hengill geothermal systems in Iceland 
        Geothermics , Vol. 57, 258-274
        
    Schnaidt, S. (2015) 
        Improving Uncertainty Estimation in Geophysical Inversion Modelling 
        PhD thesis, University of Adelaide, AU

    vr  July 2023
    """
    sm = np.shape(m1)
    dm = m1.dim
    if dm==1:
        error("crossgrad: For dim="+str(dm)+" no crossgrad! Exit.")
    elif dm==2:
        cgdim = 1
    else:
        cgdim = 3
        
    gm1 = np.gradient(m1)
    gm2 = np.gradient(m2)
    sgm = np.shape(gm1)
    
    g1 = np.ravel(gm1)
    g2 = np.ravel(gm2)
    
    cgm = np.zeros_like(g1,cgdim)
    for k in np.arange(np.size(g1)):
        cgm[k,:] = np.cross (g1[k], g2[k])

    cgm =np.reshape(cgm,(sm+cgdim))
    
    cgnm = np.abs(cgm)/(np.abs(gm1)*np.abs(gm2))

    return cgm, cgmn

def medfilt3D(
        M,
        kernel_size=[3, 3, 3], boundary_mode="nearest", maxiter=1, Out=True):
    """
    Run iterated median filter in nD.

    vr  Jan 2021
    """
    tmp = M.copy()
    for it in range(maxiter):
        if Out:
            print("iteration: " + str(it))
        tmp = median_filter(tmp, size=kernel_size, mode=boundary_mode)

    G = tmp.copy()

    return G


def anidiff3D(
        M,
        ckappa=50, dgamma=0.1, foption=1, maxiter=30, Out=True):
    """
    Apply anisotropic nonlinear diffusion in nD.

    vr  Jan 2021
    """
    tmp = M.copy()

    tmp = anisodiff3D(
        tmp,
        niter=maxiter,
        kappa=ckappa,
        gamma=dgamma,
        step=(1.0, 1.0, 1.0),
        option=foption)

    G = tmp.copy()

    return G


def anisodiff3D(
        stack,
        niter=1, kappa=50, gamma=0.1, step=(1.0, 1.0, 1.0), option=1,
        ploton=False):
    """
    Apply 3D Anisotropic diffusion.

    Usage:
    stackout = anisodiff(stack, niter, kappa, gamma, option)

    Arguments:
            stack  - input stack
            niter  - number of iterations
            kappa  - conduction coefficient 20-100 ?
            gamma  - max value of .25 for stability
            step   - tuple, the distance between adjacent pixels in (z,y,x)
            option - 1 Perona Malik diffusion equation No 1
                     2 Perona Malik diffusion equation No 2
            ploton - if True, the middle z-plane will be plotted on every
                     iteration

    Returns:
            stackout   - diffused stack.

    kappa controls conduction as a function of gradient.  If kappa is low
    small intensity gradients are able to block conduction and hence diffusion
    across step edges.  A large value reduces the influence of intensity
    gradients on conduction.

    gamma controls speed of diffusion (you usually want it at a maximum of
    0.25)

    step is used to scale the gradients in case the spacing between adjacent
    pixels differs in the x,y and/or z axes

    Diffusion equation 1 favours high contrast edges over low contrast ones.
    Diffusion equation 2 favours wide regions over smaller ones.

    Reference:
    P. Perona and J. Malik.
    Scale-space and edge detection using ansotropic diffusion.
    IEEE Transactions on Pattern Analysis and Machine Intelligence,
    12(7):629-639, July 1990.

    Original MATLAB code by Peter Kovesi
    School of Computer Science & Software Engineering
    The University of Western Australia
    pk @ csse uwa edu au
    <http://www.csse.uwa.edu.au>

    Translated to Python and optimised by Alistair Muldal
    Department of Pharmacology
    University of Oxford
    <alistair.muldal@pharm.ox.ac.uk>

    June 2000  original version.
    March 2002 corrected diffusion eqn No 2.
    July 2012 translated to Python
    Jan 2021 slightly adapted python3 VR
    """
    # initialize output array
    if ploton:
        import pylab as pl
        from time import sleep

    stackout = stack.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(stackout)
    deltaE = deltaS.copy()
    deltaD = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    UD = deltaS.copy()
    gS = np.ones_like(stackout)
    gE = gS.copy()
    gD = gS.copy()

    # create the plot figure, if requested
    if ploton:

        showplane = stack.shape[0] // 2

        fig = pl.figure(figsize=(20, 5.5), num="Anisotropic diffusion")
        ax1, ax2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)

        ax1.imshow(
            stack[showplane, ...].squeeze(),
            interpolation="nearest")
        ih = ax2.imshow(
            stackout[showplane, ...].squeeze(),
            interpolation="nearest", animated=True
        )
        ax1.set_title("Original stack (Z = %i)" % showplane)
        ax2.set_title("Iteration 0")

        fig.canvas.draw()

    for ii in range(niter):

        # calculate the diffs
        deltaD[:-1, :, :] = np.diff(stackout, axis=0)
        deltaS[:, :-1, :] = np.diff(stackout, axis=1)
        deltaE[:, :, :-1] = np.diff(stackout, axis=2)

        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gD = np.exp(-((deltaD / kappa) ** 2.0)) / step[0]
            gS = np.exp(-((deltaS / kappa) ** 2.0)) / step[1]
            gE = np.exp(-((deltaE / kappa) ** 2.0)) / step[2]
        elif option == 2:
            gD = 1.0 / (1.0 + (deltaD / kappa) ** 2.0) / step[0]
            gS = 1.0 / (1.0 + (deltaS / kappa) ** 2.0) / step[1]
            gE = 1.0 / (1.0 + (deltaE / kappa) ** 2.0) / step[2]

        # update matrices
        D = gD * deltaD
        E = gE * deltaE
        S = gS * deltaS

        # subtract a copy that has been shifted 'Up/North/West' by one
        # pixel. don't as questions. just do it. trust me.
        UD[:] = D
        NS[:] = S
        EW[:] = E
        UD[1:, :, :] -= D[:-1, :, :]
        NS[:, 1:, :] -= S[:, :-1, :]
        EW[:, :, 1:] -= E[:, :, :-1]

        # update the image
        stackout += gamma * (UD + NS + EW)

        if ploton:
            iterstring = "Iteration %i" % (ii + 1)
            ih.set_data(stackout[showplane, ...].squeeze())
            ax2.set_title(iterstring)
            fig.canvas.draw()
            # sleep(0.01)

    return stackout


def shock3d(
        M,
        dt=0.2, maxiter=30, filt=[3, 3, 3, 0.5],
        boundary_mode="nearest", signfunc=None):
    """
    Apply shock filter in nD.

    vr  Jan 2021
    """
    if signfunc is None or signfunc == "sign":
        signcall = "-np.sign(L)"

    elif signfunc[0] == "sigmoid":
        scale = 1.0
        signcall = "-1./(1. + np.exp(-scale *L))"

    else:
        error("sign func " + signfunc + " not defined! Exit.")

    kersiz = (filt[0], filt[1], filt[2])
    kerstd = filt[3]
    K = gauss3D(kersiz, kerstd)
    # print(np.sum(K.flat))
    G = M

    for it in range(maxiter):

        G = convolve(G, K, mode=boundary_mode)

        g = np.gradient(G)
    #         print(np.shape(g))
    #         normg=norm(g)
    #         normg=np.sqrt(g[0])
    #         print(np.shape(normg))
    #         L = laplace(G)

    #         S = eval(signcall)

    #         G=G+dt*normg*S

    return G


def gauss3D(Kshape=(3, 3, 3), Ksigma=0.5):
    """
    Define 2D gaussian mask.

    Should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])

    vr  Jan 2021
    """
    k, m, n = [(ss - 1) / 2 for ss in Kshape]
    x, y, z = np.ogrid[-n:n+1, -m:m+1, -k:k+1]
    h = np.exp(-(x * x + y * y + z * z) / (2.0 * Ksigma * Ksigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    s = h.sum()
    if s != 0:
        h /= s

    K = h

    return K


def prepare_model(rho, rhoair=1.0e17):
    """
    Prepare model for filtering etc.

    Mainly redefining the boundaries (in the case of topograpy)
    Air domain is filed with vertical surface value
    Created on Tue Jan  5 11:59:42 2021

    @author: vrath
    """
    nn = np.shape(rho)

    rho_new = rho

    for ii in range(nn[0]):
        for jj in range(nn[1]):
            tmp = rho[ii, jj, :]
            na = np.argwhere(tmp < rhoair / 100.0)[0]
            # print(' orig')
            # print(tmp)
            tmp[: na[0]] = tmp[na[0]]
            # print(' prep')
            # print(tmp)
            rho_new[ii, jj, :] = tmp

    return rho_new
