import os
import sys
from sys import exit as error
import string
import time

import numpy as np
from numpy.linalg import norm
from scipy.io import FortranFile
from scipy.ndimage import laplace, convolve
from scipy.ndimage import uniform_filter, gaussian_filter, median_filter
from numba import jit

import util as utl

# import scipy.sparse as scs

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
    # old format
    # s = strng.replace(";","").split()
    # i1 = int(s[3])
    # i2 = int(s[5])
    # i3 = int(s[7])

    s = strng.split()

    # print(" in s[0]:  ", s[0] )

    i1 = int(s[0])
    i2 = int(s[1])
    i3 = int(s[2])

    ivals = [i1, i2, i3]
    return ivals

def read_jac(Jacfile=None, out=False):
    """
    Read Jacobian from ModEM output.

    author: vrath
    last changed: Dec 17, 2023
    """
    if out:
        print("Opening and reading " + Jacfile)

    eof = False
    fjac = FortranFile(Jacfile, "r")
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
                # if int(header2[0])==1 or int(header2[0])==0:
                #     eof = True
                #     break

                h2 = ''.join([chr(item) for item in header2])

                # print("\n\n\n",type(header2))
                # print(header2[0])
                # print(isinstance(header2[0], int))
                # print(isinstance(header2[0], str))
                # print(int(header2[0]))
                # # print("this is header2 ",header2)
                # # print("this is H2 ",h2)
                # print(decode_h2(h2))
                tmp2.append(decode_h2(h2))

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
        #     if eof: break
        # if eof: break


    Jac = np.asarray(tmp1)
    Inf = np.asarray(tmp2)
#    Inf = np.asarray(tmp2,dtype=object)

    fjac.close()

    if out:
        print("...done reading " + Jacfile)

    return Jac, Inf  #, Site, Freq, Comp


def read_data_jac(Datfile=None, out=True):
    """
    Read ModEM input data.

    author: vrath
    last changed: Dec 17, 2023
    """
    Data = []
    Site = []
    Comp = []
    Head = []
    Dtyp = []
    """
    !    Full_Impedance              = 1
    !    Off_Diagonal_Impedance      = 2
    !    Full_Vertical_Components    = 3
    !    Full_Interstation_TF        = 4
    !    Off_Diagonal_Rho_Phase      = 5
    !    Phase_Tensor                = 6
    """

    with open(Datfile) as fd:
        for line in fd:
            if line.startswith("#") or line.startswith(">"):
                Head.append(line)
                continue

            t = line.split()

            if t:
                if int(t[8]) in [1,2,3,6,5]:

                    #print(" 1: ", t[5], t[6], len(t))
                    tmp= [
                        float(t[0]), float(t[2]), float(t[3]), float(t[4]),
                        float(t[5]), float(t[6]), float(t[9]), float(t[10]),
                        ]
                    Data.append(tmp)
                    Site.append([t[1]])
                    Comp.append([t[7]])
                    Dtyp.append([int(t[8])])


    Site = [item for sublist in Site for item in sublist]
    Site = np.asarray(Site, dtype=object)
    Comp = [item for sublist in Comp for item in sublist]
    Comp = np.asarray(Comp, dtype=object)

    Dtyp =  [item for sublist in Dtyp for item in sublist]
    Dtyp =  np.asarray(Dtyp, dtype=object)

    Data = np.asarray(Data)

    if np.shape(Data)[0]==0:
        error("read_data_jac: No data read! Exit.")

    Freq = Data[:,0]

    nD = np.shape(Data)
    if out:
        print("readDat: %i data read from %s" % (nD[0], Datfile))

    return Data, Site, Freq, Comp, Dtyp, Head


def write_jac_ncd(NCfile=None, Jac=None, Dat=None, Site=None, Comp=None,
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

    ncout = nc.Dataset(NCfile, "w", format="NETCDF4")
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
            (NCfile, ncout.data_model)
        )



def read_data(Datfile=None,  modext=".dat", out=True):
    """
    Read ModEM input data.

    author: vrath
    last changed: Feb 10, 2024


    """

    file = Datfile+modext

    Data = []
    Site = []
    Comp = []
    Head = []

    with open(file) as fd:
        for line in fd:
            if line.startswith("#") or line.startswith(">"):
                Head.append(line)

                continue

            t = line.split()

            if "PT" in t[7] or "RH" in t[7] or "PH" in t[7]:
                tmp = [
                    float(t[0]), float(t[2]), float(t[3]), float(t[4]),
                    float(t[5]), float(t[6]), float(t[8]),
                    float(t[9]),  0.,
                ]
                Data.append(tmp)
                Site.append([t[1]])
                Comp.append([t[7]])
            else:
                tmp = [
                    float(t[0]), float(t[2]), float(t[3]), float(t[4]),
                    float(t[5]), float(t[6]), float(t[8]),
                    float(t[9]), float(t[10]),
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
        print("readDat: %i data read from %s" % (nD[0], file))

    return Site, Comp, Data, Head


def write_data(Datfile=None, Dat=None, Site=None, Comp=None, Head = None,
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

    with open(Datfile,"w") as fd:

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
        NCfile=None, Dat=None, Site=None, Comp=None,
        zlib_in=True, shuffle_in=True, out=True
        ):
    """
    Write Jacobian from ModEM output to NETCDF file.

    author: vrath
    last changed: July 24, 2020
    """
    try:
        NCfile.close
    except BaseException:
        pass

    DatDim = np.shape(Dat)

    ncout = nc.Dataset(NCfile, "w", format="NETCDF4")
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
            % (NCfile, ncout.data_model)
        )


def write_mod_ncd(
    NCfile=None,
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

    ncout = nc.Dataset(NCfile, "w", format="NETCDF4")

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
            print("resistivities to " + NCfile + " transformed to: " + trans)
    elif trans == "LOG10":
        Mod = np.log10(Mod)
        if out:
            print("resistivities to " + NCfile + " transformed to: " + trans)
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
        R[:] = Ref[:]

    ncout.close()

    if out:
        print(
            "write_modelNC: data written to %s in %s format"
            % (NCfile, ncout.data_model)
        )


def write_mod_npz(file=None,
                    dx=None, dy=None, dz=None, mval=None, reference=None,
                    compressed=True, trans="LINEAR",
                    aircells=None, mvalair=1.e17, blank=1.e-30, header="",
                    out=True):
    """
    Write ModEM model input.

    Expects mval in physical units (linear).

    author: vrath
    last changed: Feb 26, 2024

    """




    dims = np.shape(mval)
    if mval.dim==3:
        nx, ny, nz = dims
    else:
        nx ,ny ,nz, nset = dims


    if not aircells  is None:
        mval[aircells] = mvalair

    if not blank  is None:
        blanks = np.where(~np.isfinite(mval))
        mval[blanks] = blank

    if len(header)==0:
        header ="# 3D MT model written by ModEM in WS format"

    if header[0] != "#":
        header = "#"+header

    if trans is not None:
        trans = trans.upper()

        if trans == "LOGE":
            mval = np.log(mval)
            mvalair = np.log(mvalair)
            if out:
                print("values to " + file + " transformed to: " + trans)
        elif trans == "LOG10":
            mval = np.log10(mval)
            mvalair = np.log10(mvalair)
            if out:
                print("values to " + file + " transformed to: " + trans)
        elif trans == "LINEAR":
            pass

        else:
            print("Transformation: " + trans + " not defined!")
            sys.exit(1)

    else:
        trans == "LINEAR"

    trns = np.array(trans)

    if reference is None:
        ncorner = -0.5*np.sum(dx)
        ecorner = -0.5*np.sum(dy)
        elev = 0.
        cnt = np.array([ncorner, ecorner, elev])
    else:
        cnt = np.asarray(reference)

    info = np.array([trns], dtype="object")

    if compressed:
        modext=".npz"
        modf = file+modext

        np.savez_compressed(modf, header=header, info=info,
                            dx=dx, dy=dy, dz=dz, mval=mval, reference=cnt)
        print("model written to "+modf)
    else:
        modext=".npy"
        modf = file+modext
        np.savez(modf, header=header, info=info,
                            dx=dx, dy=dy, dz=dz, mval=mval, reference=cnt)
        print("model written to "+modf)

def write_mod(file=None, modext=".rho",
                    dx=None, dy=None, dz=None, mval=None, reference=None,
                    trans="LINEAR", aircells = None, mvalair = 1.e17, blank = 1.e-30, header="", out=True):
    """
    Write ModEM model input.

    Expects mval in physical units (linear).

    author: vrath
    last changed: Aug 28, 2023


    Modem model format in Fortran:

    DO iz = 1,Nz
        DO iy = 1,Ny
            DO ix = Nx,1,-1
                READ(10,*) mval(ix,iy,iz)
            ENDDO
        ENDDO
    ENDDO

    """


    modf = file+modext

    dims = np.shape(mval)

    nx = dims[0]
    ny = dims[1]
    nz = dims[2]
    dummy = 0


    if not aircells  is None:
        mval[aircells] = mvalair

    if not blank  is None:
        blanks = np.where(~np.isfinite(mval))
        mval[blanks] = blank

    if len(header)==0:
        header ="# 3D MT model written by ModEM in WS format"

    if header[0] != "#":
        header = "#"+header

    if trans is not None:
        trans = trans.upper()

        if trans == "LOGE":
            mval = np.log(mval)
            mvalair = np.log(mvalair)
            if out:
                print("values to " + file + " transformed to: " + trans)
        elif trans == "LOG10":
            mval = np.log10(mval)
            mvalair = np.log10(mvalair)
            if out:
                print("values to " + file + " transformed to: " + trans)
        elif trans == "LINEAR":
            pass

        else:
            print("Transformation: " + trans + " not defined!")
            sys.exit(1)

    else:
        trans == "LINEAR"

    trns = np.array(trans)

    if reference is None:
        ncorner = -0.5*np.sum(dx)
        ecorner = -0.5*np.sum(dy)
        elev = 0.
        cnt = np.array([ncorner, ecorner, elev])
    else:
        cnt = np.asarray(reference)


    with open(modf, "w") as f:
        np.savetxt(
            f, [header], fmt="%s")
        line = np.array([nx, ny,nz, dummy, trns],dtype="object")
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


        np.savetxt(f, cnt.reshape(1, cnt.shape[0]), fmt="%10.1f")
        f.write("%10.2f  \n" % (0.0))

def write_rlm(file=None, modext=".rlm",
                    dx=None, dy=None, dz=None, mval=None, reference=None,
                    aircells = None, mvalair = 1.e17, blank = 1.e-30,
                    comment="", name="", out=True):
    """
    Write GGG model input.

    conventions:
        x = east, y = south, z = down
        expects mval in physical units (?).


    author: vrath
    last changed: jan 18, 2024


    """


    modf = file+modext

    nx, ny, nz  = np.shape(mval)

    if not aircells  is None:
        mval[aircells] = mvalair

    if not blank  is None:
        blanks = np.where(~np.isfinite(mval))
        mval[blanks] = blank

    if len(comment)==0:
        comment ="# 3D MT model in RLM format"

    comment = comment.strip()
    if comment[0] != "#":
        comment = "#"+comment

    if len(name)==0:
        name=file
    if reference is None:
        ncorner = -0.5*np.sum(dx)
        ecorner = -0.5*np.sum(dy)
        elev = 0.
        cnt = np.array([ncorner, ecorner, elev])
    else:
        cnt = np.asarray(reference)

    with open(modf, "w") as f:

        line = np.array([nx, ny,nz],dtype="object")
        np.savetxt(f, line.reshape(1, 3), fmt ="  %i")
        np.savetxt(f, dx.reshape(1, dx.shape[0]), fmt="%12.3f")
        np.savetxt(f, dy.reshape(1, dy.shape[0]), fmt="%12.3f")
        np.savetxt(f, dz.reshape(1, dz.shape[0]), fmt="%12.3f")

        # write out the layers from resmodel
        for zi in range(dz.size):
            f.write(str(zi+1))
            for yi in range(dy.size):
                line = mval[:, yi, zi]
                np.savetxt(f, line.reshape(1, nx), fmt="%12.5e")


        np.savetxt(
            f, [comment], fmt="%s")
        np.savetxt(
            f, [name], fmt="%s")

        np.savetxt(
            f, [1, 1], fmt="%i")

        np.savetxt(f, [cnt[0], cnt[1]], fmt="%16.6g")
        f.write("%10.2f  \n" % (0.0))
        np.savetxt(f, [cnt[2]], fmt="%16.6g")



def write_ubc(file=None,  mshext=".mesh", modext=".ubc",
                    dx=None, dy=None, dz=None, mval=None, reference=None,
                    aircells = None, mvalair = 1.e17, blank = 1.e17, header="", out=True):
    """
    Write UBC model input.

    Expects mval in physical units (linear).

    author: vrath
    last changed: Aug 28, 2023

    """

    modf = file+modext
    mesh = file+mshext


    dims = np.shape(mval)


    if not aircells  is None:
        mval.reshape(dims)[aircells] = mvalair

    if not blank  is None:
        blanks = np.where(~np.isfinite(mval))
        mval.reshape(dims)[blanks] = mvalair



    dyu = np.flipud(dx.reshape(1, dx.shape[0]))
    dxu = dy.reshape(1, dy.shape[0])
    dzu = dz.reshape(1, dz.shape[0])

    lat = reference[0]
    lon = reference[1]
    utm_zone = utl.get_utm_zone(lat,lon)
    utme, utmn = utl.proj_latlon_to_utm(lat, lon, utm_zone=utm_zone[0])
    ubce = utme - 0.5*np.sum(dxu)
    ubcn = utmn - 0.5*np.sum(dyu)
    refu = np.array([ubce, ubcn, reference[2], utm_zone[0]]).reshape(1,4)
    # print(refu)


    val = np.transpose(mval, (1,0,2))

    dimu = np.shape(val)
    dimu = np.asarray(dimu)
    dimu = dimu.reshape(1, dimu.shape[0])
    val = val.flatten(order="C")


    with open(mesh , "w") as f:
        np.savetxt(f, dimu, fmt="%i")
        np.savetxt(f, refu, fmt="%14.3f %14.3f %14.3f %10i")

        np.savetxt(f, dxu, fmt="%12.3f")
        np.savetxt(f, dyu, fmt="%12.3f")
        np.savetxt(f, dzu, fmt="%12.3f")



    with open(modf , "w") as f:
        np.savetxt(f, val, fmt="%14.5g")




def read_ubc(file=None, modext=".mod", mshext=".msh",
                   trans="LINEAR", volumes=False, out=True):
    """
    Read UBC model input.

    author: vrath
    last changed: Aug 30, 2023

    """

    modf = file+modext
    mesh = file+mshext


    with open(mesh, "r") as f:
        lines = f.readlines()

    lines = [line.split() for line in lines]

    dims = [int(sub) for sub in lines[0][:2]]
    refs = [float(sub) for sub in lines[1][:4]]

    dxu = np.array([float(sub) for sub in lines[2]])
    dyu = np.array([float(sub) for sub in lines[3]])
    dzu = np.array([float(sub) for sub in lines[4]])


    dx = np.flipud(dyu.reshape(1, dyu.shape[0]))
    nx = dx.size
    dy = dxu.reshape(1, dxu.shape[0])
    ny = dy.size
    dz = dzu.reshape(1, dzu.shape[0])
    nz = dz.size

    ubce, ubcn, elev, utmz = refs
    mode = ubce + 0.5*np.sum(dxu)
    modn = ubcn + 0.5*np.sum(dyu)
    lat, lon = utl.proj_utm_to_latlon(mode, modn, utm_zone=utmz)
    # print(lat, lon)

    refx = -0.5*np.sum(dx)
    refy = -0.5*np.sum(dy)
    refz = -refs[2]
    utmz = refs[3]
    refubc = np.array([refx, refy, refz, utmz])


    with open(modf, "r") as f:
        lines = f.readlines()

    val = np.array([])
    for line in lines:
        val = np.append(val, float(line))
    val = np.reshape(val, (ny, nx, nz))
    val = np.transpose(val, (1,0,2))


    # here mval should be in physical units, not log...
    if "loge" in trans.lower() or "ln" in trans.lower():
        val = np.log(val)
        if out:
            print("values transformed to: " + trans)
    elif "log10" in trans.lower():
        val = np.log10(val)
        if out:
            print("values transformed to: " + trans)
    else:
        if out:
            print("values transformed to: " + trans)
        pass

    if out:
        print(
            "read_model: %i x %i x %i model-like read from %s" % (nx, ny, nz, file))

    return dx, dy, dz, val, refubc, trans

def get_volumes(dx=None, dy=None, dz=None, mval=None, out=True):
    nx, ny,nz = np.shape(mval)
    vcell = np.zeros_like(mval)
    for ii in np.arange(nx):
        for jj in np.arange(ny):
            for kk in np.arange(nz):
                vcell[ii,jj,kk] = dx[ii]*dy[jj]*dz[kk]

    if out:
        print(
            "ger_volumes: %i x %i x %i cell volumes calculated" %
            (nx, ny, nz))

    return vcell


def get_topo(dx=None, dy=None, dz=None, mval=None, ref= [0., 0., 0.],
             mvalair = 1.e17, out=True):
        nx, ny,nz = np.shape(mval)


        x = np.append(0.0, np.cumsum(dx))
        xcnt = 0.5 * (x[0:nx] + x[1:nx+1]) + ref[0]

        y = np.append(0.0, np.cumsum(dy))
        ycnt = 0.5 * (y[0:ny] + y[1:ny+1]) + ref[1]

        ztop = np.append(0.0, np.cumsum(dz)) + ref[2]

        topo = np.zeros((nx, ny))
        for ii in np.arange(nx):
            for jj in np.arange(ny):
                col = mval[ii,jj,:]
                nsurf = np.argmax(col<mvalair)
                topo[ii, jj]= ztop[nsurf]

        if out:
            print(
                "get topo: %i x %i x %i ccell surfaces marked" % (nx, ny, nz))

        return xcnt, ycnt, topo

def read_mod(file=None, modext=".rho", trans="LINEAR", blank=1.e-30, out=True):
    """
    Read ModEM model input.

    Returns mval in physical units

    author: vrath
    last changed: Aug 31, 2023

    """


    modf = file+modext

    with open(modf, "r") as f:
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
        print("values in " + file + " are: " + trns)

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
    mval[np.where(np.abs(mval)<blank)]=blank


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
            "read_model: %i x %i x %i model read from %s" % (nx, ny, nz, file))



    return dx, dy, dz, mval, reference, trans

        # if trim:
        #     for ix in range(trim[0]):
        #         model.dx_delete(0)
        #         model.dx_delete(model.nx)
        #     for ix in range(trim[1]):
        #         model.dy_delete(0)
        #         model.dy_delete(model.ny)
        #     for ix in range(trim[2]):
        #         model.dz_delete(model.nz)

def write_mod_vtk(file=None, dx=None, dy=None, dz=None, rho=None,
                  trim=[10, 10, 30], reference=None, scale = [1., 1., -1.],
                  trans="LINEAR", out=True):
    """
    write ModEM model input in


    author: vrath
    last changed: Mar 13, 2024

    """
    from evtk.hl import gridToVTK

    if trim is not None:
        print("model trimmed"
              +", x="+str(trim[0])
              +", y="+str(trim[1])
              +", z="+str(trim[2])
              )

        for ix in range(trim[0]):
            dx  = np.delete(dx, (0,-1))
        for ix in range(trim[1]):
            dy  = np.delete(dy, (0,-1))
        for ix in range(trim[2]):
            dz  = np.delete(dz, (-1))


    X =  np.append(0.0, np.cumsum(dy))*scale[1]
    Y =  np.append(0.0, np.cumsum(dx))*scale[1]
    Z =  np.append(0.0, np.cumsum(dz))*scale[2]



    gridToVTK(file, X, Y, -Z , cellData = {'resistivity (in Ohm)' : rho})
    print("model-like parameter written to %s" % (file))


def write_dat_vtk(Sitfile=None, sx=None, sy=None, sz=None, sname=None,
                   reference=None, scale = [1., 1., -1.], out=True):
    """
    Convert ModEM data file to VTK station set (unstructured grid)


    """
    from evtk.hl import pointsToVTK


    N = sx*scale[0]
    E = sy*scale[1]
    D = sz*scale[2]


    #dummy scalar values
    dummy = np.ones((len(N)))

    pointsToVTK(Sitfile, N, E, D, data = {"value" : dummy})

    print("site positions written to %s" % (Sitfile))


def fix_cells( covfile_i=None,
               covfile_o=None,
               modfile_i=None,
               modfile_o=None,
               datfile_i=None,
               fixed="2",
               method = ["border", 3],
               fixmod = ["prior"],
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


    dx, dy, dz, rho, reference, _ = read_mod(modfile_i, out=True)
    modsize = np.shape(rho)

    if "dist" in method[0].lower():
        fixdist = method[1]
        x = np.append(0., np.cumsum(dx)) + reference[0]
        xc =0.5*(x[0:len(x)-1]+x[1:len(x)])
        y = np.append(0., np.cumsum(dy)) + reference[1]
        yc =0.5*(y[0:len(y)-1]+y[1:len(y)])
        cellcent = [xc, yc]

        # print(len(xc),len(yc))
        Site , _, Data, _ = read_data(datfile_i, out=True)

        xs = []
        ys = []
        for idt in range(0, np.size(Site)):
            ss = Site[idt]
            if idt == 0:
                site = Site[idt]
                xs.append(Data[idt,3])
                ys.append(Data[idt,4])
            elif ss != site:
                site = Site[idt]
                xs.append(Data[idt,3])
                ys.append(Data[idt,4])

        sitepos = [xs, ys]


    if "bord" in method[0].lower():
        border = method[1]


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

    if "bord" in method[0].lower():
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
        error("fix_cells: Number of blocks wrong! Exit.")

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
                    print(dmin)
                    if dmin > fixdist:
                        tmp[ii][jj] = tmp[ii][jj].replace("1", fixed)

                tmp[ii].append("\n")
                new_block.append(" ".join(tmp[ii]))


        l_o[ib+1:ib+block_len+1] = new_block

    with open(covfile_o, "w") as f_o:
            f_o.writelines(l_o)
    if out:
        print("fix_cells: covariance control read from %s" % (covfile_i))
        print("fix_cells: covariance control written to %s" % (covfile_o))
        if "bord" in method.lower():
            print(str(border)+" border  cells fixed (zone "+str(fixed)+")")
        else:
            if unit=="km":
                print("cells with min distance to site > "
                      +str(fixdist/1000)+"km fixed (zone "+str(fixed)+")")
            else:
                print("cells with min distance to site > "
                     +str(fixdist)+"m fixed (zone "+str(fixed)+")")

    if "val" in fixmod[0].lower():
        write_mod(modfile_o, dx, dy, dz, rho,reference,out = True)
        if out:
            print("fix_cells: model written to %s" % (covfile_o))
            print("fix_cells: model in %s fixed to %g Ohm.m" % (covfile_i))
    else:
        if out:
            print("fix_cells: model in %s fixed to prior" % (modfile_i))



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




def data_to_pv(data=None, site=None, reference=None, scale=1.):

    x =  data[:, 3]
    y =  data[:, 4]
    z =  data[:, 5]

    if reference is not None:
        x =  x + reference[0]
        y =  y + reference[1]
        z =  z + reference[2]

    y, x, z  = scale*x, scale*y, -scale*z



    sites, siteindex = np.unique(site, return_index=True)
    x = x[siteindex]
    y = y[siteindex]
    z = z[siteindex]

    sites = sites.astype('<U4')
    siten= np.array([ii for ii in np.arange(len(z))])

    # z = -z

    return x, y, z, sites, siten


def model_to_pv(dx=None, dy=None,dz=None, rho=None, reference=None,
                scale=1., pad = [12, 12, 30.]):


    x, y, z = cells3d(dx, dy, dz)


    x =  x + reference[0]
    y =  y + reference[1]
    z =  z + reference[2]

    x, y, z  = scale*x, scale*y, scale*z

    x, y, z, rho = clip_model(x, y, z, rho, pad = pad)

    # vals = np.swapaxes(np.flip(rho, 2), 0, 1).flatten(order="F")
    vals = rho.copy()
    vals = np.swapaxes(vals, 0, 1)
    # vals = np.flip(rho.copy(), 2)
    vals = vals.flatten(order="F")

    z = -z
    return x, y, z, vals


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


def insert_body(dx=None, dy=None, dz=None,
    rho_in=None, body=None,
    pad=[0, 0, 0],
    smooth=None, scale=1.0, reference = None,
    out=True):
    """
    Insert 3d body (ellipsoid or box) into given model.

    Created on Sun Jan 3 10:35:28 2021

    @author: vrath
    """
    xpad = pad[0]
    ypad = pad[1]
    zpad = pad[2]

    xc, yc, zc = cells3d(dx, dy, dz)

    if reference is None:
        modcenter = [0.5 * np.sum(dx), 0.5 * np.sum(dy), 0.0]
    else:
        modcenter = reference

    xc = xc - modcenter[0]
    yc = yc - modcenter[1]
    zc = zc - modcenter[2]

    nx = np.shape(xc)[0]
    ny = np.shape(yc)[0]
    nz = np.shape(zc)[0]

    rho_out = np.log(rho_in.copy())

    geom= body[0]
    action = body[1]
    rhoval = body[2]
    bcent = body[3:6]
    baxes = body[6:9]
    bangl = body[9:12]


    rhoval = np.log(rhoval)

    if action[0:3] == "rep":
        actstring = "rhoval"
    elif action[0:3] == "add":
        actstring = "rho_out[ii,jj,kk] + rhoval"
    else:
        error("Action" + action + " not implemented! Exit.")

    if out:
        print(
            "Body type   : " + geom + ", " + action + " rho =",
            str(np.exp(rhoval)) + " Ohm.m",
        )
        print("Body center : " + str(bcent))
        print("Body axes   : " + str(baxes))
        print("Body angles : " + str(bangl))
        print("Smoothed with " + smooth[0] + " filter")

    if "ell" in geom.lower():

        for kk in np.arange(0, nz - zpad - 1):
            zpoint = zc[kk]
            for jj in np.arange(ypad + 1, ny - ypad - 1):
                ypoint = yc[jj]
                for ii in np.arange(xpad + 1, nx - xpad - 1):
                    xpoint = xc[ii]

                    position = [xpoint, ypoint, zpoint]
                    # if Out:
                    print('position', position)
                    # print(position)
                    # print( bcent)
                    if in_ellipsoid(position, bcent, baxes, bangl):
                        rho_out[ii, jj, kk] = eval(actstring)
                        # if Out:
                        #     print("cell %i %i %i" % (ii, jj, kk))

    if "box" in geom.lower():
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

    rho_out = np.exp(rho_out)

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
        print("cells3d returning raw node coordinates.")
        return x, y, z


def in_ellipsoid(
    point=None,
    cent=[0.0, 0.0, 0.0],
    axs=[1.0, 1.0, 1.0],
    ang=[0.0, 0.0, 0.0],
    find_inside=True):
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
    find_inside=True,):
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

    return cgm, cgnm

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

    rho_new = rho.copy()

    for ii in range(nn[0]):
        for jj in range(nn[1]):
            tmp = rho[ii, jj, :]
            na = np.argwhere(tmp < rhoair / 10.0)[0]
            # print(' orig')
            # print(tmp)
            tmp[: na[0]] = tmp[na[0]]
            # print(' prep')
            # print(tmp)
            rho_new[ii, jj, :] = tmp

    return rho_new


def insert_body_ijk(template = None, rho_in=None,
                    perturb=None, bodymask=None, out=True):
    """
    Insert 3d box into given model.

    @author: vrath
    """
    if template is None:
        error("insert_body_ijk: no template! Exit.")

    if rho_in is None:
        error("insert_body_ijk: no base model! Exit.")

    if perturb is None:
        error("insert_body_ijk: no perturbation! Exit.")

    if bodymask is None:
        error("insert_body_ijk: no body! Exit.")

    rho_out = np.log(rho_in.copy())


    if out:
        print("Perturbation amplitude: "+str(perturb) + " log10")

    centers = np.where(template != 0.)
    _, nbody = np.shape(centers)

    bw = bodymask

    for ibody in np.arange(nbody):

        bc = centers[:,ibody]

        ib = np.arange(bc[0]-bw[0],bc[0]+bw[0]+1)
        jb = np.arange(bc[1]-bw[1],bc[1]+bw[1]+1)
        kb = np.arange(bc[1]-bw[1],bc[1]+bw[1]+1)
        rho_out[ib, jb, kb] = template[bc]*perturb+rho_in[ib, jb, kb]


        if out:
            print("Body center at: " + str(bc))


    return rho_out

def distribute_bodies_ijk(model=None,
                      method=["random", 25, "uniform", [1, 1,   1, 1,   1, 1]],
                      valmark=1, flip="alternate", scale="ijk"):
    """
    construct templates for  distributing test boduies  within model.

    Parameters
    ----------
    model : np.array, float
        model setup in ModEM format. The default is None.
    method : list of objects, optional

        Contains parameters  for  generating centers.

        method[0] = "regular":
            ['regular', bounding box (cell indices), bodymask, step]
            Example: method = ["regular", [1, 1,   1, 1,   1, 1], [4, 4, 6],]

        method[0] = "ramdom":
            ['random', number of bodies, bounding box (cell indices),
             distribution (currently only uniform), minimum distance]
            Example: method = ["random", 25, [1, 1,   1, 1,   1, 1], "uniform", ].

    valmark : float
            Marker value (e.g 1.),

    flip : string or None
         flip = "alt"          sign change modulo 2
         flip = "ran"          random sign change


    Returns
    -------
    template : np.array, float
        zeros, like input model, with marker values at body centers git push


    @author: vrath, Feb 2024

    """
    if "ijk" not in scale:
       error("distribute_bodies: currently only index sales possible! Exit.")


    if model is None:
        error("distribute_bodies: no model given! Exit.")

    rng = np.random.default_rng()
    template = np.zeros_like(model)

    if   "reg" in method[0].lower():

        bbox = method[1]
        step = method[2]

        ci = np.arange(bbox[0],bbox[1],step[0])
        cj = np.arange(bbox[2],bbox[3],step[1])
        ck = np.arange(bbox[4],bbox[5],step[2])
        centi, centj, centk = np.meshgrid(ci, cj, ck, indexing='ij')

        bnum = np.shape(centi)
        print(bnum)

        for ibody in np.arange(bnum):
            val = valmark
            if "alt" in flip:
                if  np.mod(ibody,2)==0:
                    val = -valmark
            if "ran" in flip:
                if rng.random()>0.5:
                    val = -valmark


            template[centi[ibody], centj[ibody], centk[ibody]] = val

    elif "ran" in method[0].lower():
        error("distribute_bodies: method"+method.lower()+"not implemented! Exit.")

        bnum = method[1]
        bbox = method[2]
        bpdf = method[3]
        print("distribute_bodies: currently only uniform diributions"\
              +" implemented! input pdf ignored!")
        mdist = method[4]


        ci = np.arange(bbox[0],bbox[1],1)
        cj = np.arange(bbox[2],bbox[3],1)
        ck = np.arange(bbox[4],bbox[5],1)
        centers = []

        for ibody in np.arange(bnum):
            centi = rng.choice(ci)
            centj = rng.choice(cj)
            centk = rng.choice(ck)
            ctest = np.array([centi, centj,centk])
            if ibody==0:
                centers = ctest
            else:
                print(np.shape(centers))
                for itest in np.arange(np.shape(centers)[1]):
                    test = norm(ctest-centers[itest-1])
                    if test>=mdist:
                        template[centi, centj, centk] = val
                    else:
                        print("distribute_bodies: too near!")
    else:
        error("distribute_bodies: method"+method.lower()+"not implemented! Exit.")


    return template


def set_mesh(d=None, center=False):
    """
    Define cell geometry.

    VR Jan 2024

    """
    ncell = np.shape(d)[0]
    xn = np.append(0.0, np.cumsum(d))
    xc = 0.5 * (xn[0:ncell] + xn[1:ncell+1])

    if center:
        c = 0.5*(xn[ncell] - xn[0])
        xn = xn -c

    return xn, xc

def mask_mesh(x=None, y=None, z=None, mod=None,
              mask=None,
              ref = [0., 0., 0.],
              method="index"):
    """
    mask model-like parameters and mesh

    VR Jan 2024
    """
    msh = np.shape(mod)
    mod_out = mod.copy()
    x_out = x.copy()
    y_out = y.copy()
    z_out = z.copy()


    if ("ind" in method.lower()) or ("ijk" in method.lower()):

        ijk = mask

        x_out = x[ijk[0]:msh[0]-ijk[1]]
        y_out = y[ijk[2]:msh[1]-ijk[3]]
        z_out = z[ijk[4]:msh[2]-ijk[5]]

        mod_out = mod_out[
            ijk[0]:msh[0]-ijk[1],
            ijk[2]:msh[1]-ijk[3],
            ijk[4]:msh[2]-ijk[5]]

    elif "dis" in method.lower():

        x =  x + ref[0]
        y =  y + ref[1]
        z =  z + ref[2]

        xc = 0.5 * (x[0:msh[0]] + x[1:msh[0]+1])
        yc = 0.5 * (y[0:msh[1]] + y[1:msh[1]+1])
        zc = 0.5 * (z[0:msh[2]] + z[1:msh[2]+1])

        ix = []
        for ii in np.arange(len(xc)):
            if np.logical_and(xc[ii]>=mask[0], xc[ii]<=mask[1]):
                 ix.append(ii)
        aixt = tuple(np.array(ix).T)

        iy = []
        for ii in np.arange(len(yc)):
            if np.logical_and(yc[ii]>=mask[2], yc[ii]<=mask[3]):
                 iy.append(ii)
        aiyt = tuple(np.array(iy).T)

        iz = []
        for ii in np.arange(len(zc)):
            if np.logical_and(zc[ii]>=mask[2], zc[ii]<=mask[3]):
                 iz.append(ii)
        aizt = tuple(np.array(iz).T)

        x_out = x_out[ix.append(ix[-1]+1)]
        y_out = y_out[iy.append(iy[-1]+1)]
        z_out = z_out[iz.append(iz[-1]+1)]
        # np.append(ix,ix[-1]+1)
        print("x ",x_out)
        print("y ",y_out)
        print("x ",z_out)

        mod_out =mod_out[aixt,:,:]
        mod_out =mod_out[:,aiyt,:]
        mod_out =mod_out[:,:,aizt]
        print(np.shape(mod_out))
        print(np.shape(ix),np.shape(iy),np.shape(iz))

        return x_out, y_out, z_out, mod_out
