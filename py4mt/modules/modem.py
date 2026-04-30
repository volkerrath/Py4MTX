"""
modem.py

File I/O and utility routines for the ModEM 3-D MT inversion package.

Includes Jacobian/data/model readers and writers, NetCDF export, format conversions (UBC/RLM), and a collection of numerical utilities used in model preparation.

Dependencies
------------
- numpy
- scipy
- netCDF4
- numba (optional)
- pyproj (optional; only needed if util.py is absent)

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-12-21
"""

import os
import sys

import string
import time
import inspect

import numpy as np
from numpy.linalg import norm
from scipy.io import FortranFile
from scipy.ndimage import laplace, convolve
from scipy.ndimage import uniform_filter, gaussian_filter, median_filter
from scipy.fft import dctn, idctn
from scipy.special import eval_legendre
from scipy.interpolate import BSpline
from numba import jit

try:
    import pywt
except ImportError:
    pywt = None

try:
    import util as utl  # Project-local helpers (coordinate conversion, etc.).
except Exception:  # pragma: no cover
    utl = None

    def _get_utm_zone(lat: float, lon: float):
        """Return (zone_number, hemisphere) for WGS84 UTM."""
        zone = int((lon + 180.0) // 6.0) + 1
        hemi = "N" if lat >= 0.0 else "S"
        return zone, hemi

    def _proj_latlon_to_utm(lat: float, lon: float, utm_zone: int, hemisphere: str = "N"):
        """Project WGS84 lat/lon to UTM easting/northing using pyproj if available."""
        try:
            import pyproj
        except Exception as exc:
            raise RuntimeError("pyproj is required for UTM projections when util is unavailable") from exc
        epsg = (32600 + int(utm_zone)) if hemisphere.upper().startswith("N") else (32700 + int(utm_zone))
        crs_utm = pyproj.CRS.from_epsg(epsg)
        crs_ll = pyproj.CRS.from_epsg(4326)
        transformer = pyproj.Transformer.from_crs(crs_ll, crs_utm, always_xy=True)
        e, n = transformer.transform(lon, lat)
        return e, n

    def _proj_utm_to_latlon(e: float, n: float, utm_zone: int, hemisphere: str = "N"):
        """Project UTM easting/northing to WGS84 lat/lon using pyproj if available."""
        try:
            import pyproj
        except Exception as exc:
            raise RuntimeError("pyproj is required for UTM projections when util is unavailable") from exc
        epsg = (32600 + int(utm_zone)) if hemisphere.upper().startswith("N") else (32700 + int(utm_zone))
        crs_utm = pyproj.CRS.from_epsg(epsg)
        crs_ll = pyproj.CRS.from_epsg(4326)
        transformer = pyproj.Transformer.from_crs(crs_utm, crs_ll, always_xy=True)
        lon, lat = transformer.transform(e, n)
        return lat, lon

    class _UtlFallback:
        """Minimal subset of the original `util` API used by this module."""

        @staticmethod
        def get_utm_zone(lat: float, lon: float):
            return _get_utm_zone(lat, lon)

        @staticmethod
        def proj_latlon_to_utm(lat: float, lon: float, utm_zone: int):
            _, hemi = _get_utm_zone(lat, lon)
            return _proj_latlon_to_utm(lat, lon, utm_zone, hemisphere=hemi)

        @staticmethod
        def proj_utm_to_latlon(e: float, n: float, utm_zone: int):
            hemi = os.environ.get("MODEM_UTM_HEMISPHERE", os.environ.get("UTM_HEMISPHERE", "N"))
            return _proj_utm_to_latlon(e, n, utm_zone, hemisphere=hemi)

    utl = _UtlFallback()
# import scipy.sparse as scs

import netCDF4 as nc

# import h5netcdf as hc


def decode_h2(strng):
    """
    decode_h2.
    
    Parameters
    ----------
    strng : object
        Parameter strng.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    Decode header2 string from ModEM Jacobian (old style).
    
    ---------- strng : string    header string
    
    Returns ------- i1, i2, i3 : integer     frequency, dattype, site numbers
    """
    # old format
    # s = strng.replace(';','').split()
    # i1 = int(s[3])
    # i2 = int(s[5])
    # i3 = int(s[7])

    s = strng.split()

    # print(' in s[0]:  ', s[0] )

    i1 = int(s[0])
    i2 = int(s[1])
    i3 = int(s[2])

    ivals = [i1, i2, i3]
    return ivals


def read_jac(Jacfile=None, out=False):
    """
    read_jac.
    
    Parameters
    ----------
    Jacfile : object
        Parameter Jacfile.
    out : object
        Parameter out.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    Read Jacobian from ModEM output.
    
    author: vrath last changed: Dec 17, 2023
    """
    if out:
        print('Opening and reading ' + Jacfile)

    eof = False
    fjac = FortranFile(Jacfile, 'r')
    tmp1 = []
    tmp2 = []

    _ = fjac.read_record(np.byte)
    # h1 = ''.join([chr(item) for item in header1])
    # print(h1)
    _ = fjac.read_ints(np.int32)
    # nAll = fjac.read_ints(np.int32)
    # print('nAll'+str(nAll))
    nTx = fjac.read_ints(np.int32)
    # print('ntx'+str(nTx))
    for i1 in range(nTx[0]):
        nDt = fjac.read_ints(np.int32)
        # print('nDt'+str(nDt))
        for i2 in range(nDt[0]):
            nSite = fjac.read_ints(np.int32)
            # print('nSite'+str(nSite))
            for i3 in range(nSite[0]):
                # header2
                header2 = fjac.read_record(np.byte)
                # if int(header2[0])==1 or int(header2[0])==0:
                #     eof = True
                #     break

                h2 = ''.join([chr(item) for item in header2])

                # print('\n\n\n',type(header2))
                # print(header2[0])
                # print(isinstance(header2[0], int))
                # print(isinstance(header2[0], str))
                # print(int(header2[0]))
                # # print('this is header2 ',header2)
                # # print('this is H2 ',h2)
                # print(decode_h2(h2))
                tmp2.append(decode_h2(h2))

                nSigma = fjac.read_ints(np.int32)
                # print('nSigma'+str(nSigma))
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
                    # ColJac = fjac.read_reals(np.float64).flatten()
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
        print('...done reading ' + Jacfile)

    return Jac, Inf  # , Site, Freq, Comp


def read_data_jac(Datfile=None, out=True):
    """
    read_data_jac.
    
    Parameters
    ----------
    Datfile : object
        Parameter Datfile.
    out : object
        Parameter out.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    Read ModEM input data.
    
    author: vrath last changed: Dec 17, 2023
    """
    Data = []
    Site = []
    Comp = []
    Head = []
    Dtyp = []
    '''
    !    Full_Impedance              = 1
    !    Off_Diagonal_Impedance      = 2
    !    Full_Vertical_Components    = 3
    !    Full_Interstation_TF        = 4
    !    Off_Diagonal_Rho_Phase      = 5
    !    Phase_Tensor                = 6
    '''

    with open(Datfile) as fd:
        for line in fd:
            if line.startswith('#') or line.startswith('>'):
                Head.append(line)
                continue

            t = line.split()

            if t:
                if int(t[8]) in [1, 2, 3, 6, 5]:

                    # print(' 1: ', t[5], t[6], len(t))
                    tmp = [
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

    Dtyp = [item for sublist in Dtyp for item in sublist]
    Dtyp = np.asarray(Dtyp, dtype=object)

    Data = np.asarray(Data)

    if np.shape(Data)[0] == 0:
        sys.exit('read_data_jac: No data read! Exit.')

    Freq = Data[:, 0]

    nD = np.shape(Data)
    if out:
        print('readDat: %i data read from %s' % (nD[0], Datfile))

    return Data, Site, Freq, Comp, Dtyp, Head


def write_jac_ncd(NCfile=None, Jac=None, Dat=None, Site=None, Comp=None,
                  zlib_in=True, shuffle_in=True, out=True):
    """
    write_jac_ncd.
    
    Parameters
    ----------
    NCfile : object
        Parameter NCfile.
    Jac : object
        Parameter Jac.
    Dat : object
        Parameter Dat.
    Site : object
        Parameter Site.
    Comp : object
        Parameter Comp.
    zlib_in : object
        Parameter zlib_in.
    shuffle_in : object
        Parameter shuffle_in.
    out : object
        Parameter out.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    Write Jacobian from ModEM output to NETCDF/HDF5 file.
    
    author: vrath last changed: July 25, 2020
    """
    JacDim = np.shape(Jac)
    DatDim = np.shape(Dat)

    if JacDim[0] != DatDim[0]:
        print(
            'Error:  Jac dim='
            + str(JacDim[0])
            + ' does not match Dat dim='
            + str(DatDim[0])
        )
        sys.exit(1)

    ncout = nc.Dataset(NCfile, 'w', format='NETCDF4')
    ncout.createDimension('data', JacDim[0])
    ncout.createDimension('param', JacDim[1])

    S = ncout.createVariable(
        'site', str, ('data'), zlib=zlib_in, shuffle=shuffle_in)
    C = ncout.createVariable(
        'comp', str, ('data'), zlib=zlib_in, shuffle=shuffle_in)

    Per = ncout.createVariable(
        'Per', 'float64', ('data'), zlib=zlib_in, shuffle=shuffle_in)
    Lat = ncout.createVariable(
        'Lat', 'float64', ('data'), zlib=zlib_in, shuffle=shuffle_in)
    Lon = ncout.createVariable(
        'Lon', 'float64', ('data'), zlib=zlib_in, shuffle=shuffle_in)
    X = ncout.createVariable(
        'X', 'float64', ('data'), zlib=zlib_in, shuffle=shuffle_in)
    Y = ncout.createVariable(
        'Y', 'float64', ('data'), zlib=zlib_in, shuffle=shuffle_in)
    Z = ncout.createVariable(
        'Z', 'float64', ('data'), zlib=zlib_in, shuffle=shuffle_in)
    Val = ncout.createVariable(
        'Val', 'float64', ('data'), zlib=zlib_in, shuffle=shuffle_in)
    Err = ncout.createVariable(
        'Err', 'float64', ('data'), zlib=zlib_in, shuffle=shuffle_in)

    J = ncout.createVariable(
        'Jac', 'float64', ('data', 'param'), zlib=zlib_in, shuffle=shuffle_in)

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
            'writeJacNC: data written to %s in %s format' %
            (NCfile, ncout.data_model)
        )


def read_data(Datfile=None,  modext='.dat', out=True):
    """
    read_data.
    
    Parameters
    ----------
    Datfile : object
        Parameter Datfile.
    modext : object
        Parameter modext.
    out : object
        Parameter out.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    Read ModEM input data.
    
    author: vrath last changed: Feb 10, 2024
    """

    file = Datfile+modext

    Data = []
    Site = []
    Comp = []
    Head = []

    with open(file) as fd:
        for line in fd:
            if line.startswith('#') or line.startswith('>'):
                Head.append(line)

                continue

            t = line.split()

            if 'PT' in t[7] or 'RH' in t[7] or 'PH' in t[7]:
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
        print('readDat: %i data read from %s' % (nD[0], file))

    return Site, Comp, Data, Head


def write_data(Datfile=None, Dat=None, Site=None, Comp=None, Head=None,
               out=True):
    """
    write_data.
    
    Parameters
    ----------
    Datfile : object
        Parameter Datfile.
    Dat : object
        Parameter Dat.
    Site : object
        Parameter Site.
    Comp : object
        Parameter Comp.
    Head : object
        Parameter Head.
    out : object
        Parameter out.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    Write ModEM input data file.
    
    author: vrath last changed: Feb 10, 2021
    """
    datablock = np.column_stack(
        (Dat[:, 0], Site[:], Dat[:, 1:6], Comp[:], Dat[:, 6:10]))
    nD, _ = np.shape(datablock)

    hlin = 0
    nhead = len(Head)
    nblck = int(nhead/8)
    print(str(nblck)+' blocks will be written.')

    with open(Datfile, 'w') as fd:

        for ib in np.arange(nblck):
            blockheader = Head[hlin:hlin+8]
            hlin = hlin + 8
            for ii in np.arange(8):
                fd.write(blockheader[ii])

            if 'Impedance' in blockheader[2]:

                fmt = '%14e %14s'+'%15.6f'*2+' %15.1f'*3+' %14s'+' %14e'*3

                indices = []
                block = []
                for ii in np.arange(len(Comp)):
                    if ('ZX' in Comp[ii]) or ('ZY' in Comp[ii]):
                        indices.append(ii)
                        block.append(datablock[ii, :])

                if out:
                    print('Impedances')
                    print(np.shape(block))

            elif 'Vertical' in blockheader[2]:

                fmt = '%14e %14s'+'%15.6f'*2+' %15.1f'*3+' %14s'+' %14e'*3

                indices = []
                block = []
                for ii in np.arange(len(Comp)):
                    if ('TX' == Comp[ii]) or ('TY' == Comp[ii]):
                        indices.append(ii)
                        block.append(datablock[ii, :])

                if out:
                    print('Tipper')
                    print(np.shape(block))

            elif 'Tensor' in blockheader[2]:

                fmt = '%14e %14s'+'%15.6f'*2+' %15.1f'*3+' %14s'+' %14e'*3

                indices = []
                block = []
                for ii in np.arange(len(Comp)):
                    if ('PT' in Comp[ii]):
                        indices.append(ii)
                        block.append(datablock[ii, :])

                if out:
                    print('Phase Tensor')
                    print(np.shape(block))

            else:

                print('Data type '+blockheader[2]+'not implemented! Exit.')

            np.savetxt(fd, block, fmt=fmt)


def write_data_ncd(
        NCfile=None, Dat=None, Site=None, Comp=None,
        zlib_in=True, shuffle_in=True, out=True
):
    """
    write_data_ncd.
    
    Parameters
    ----------
    NCfile : object
        Parameter NCfile.
    Dat : object
        Parameter Dat.
    Site : object
        Parameter Site.
    Comp : object
        Parameter Comp.
    zlib_in : object
        Parameter zlib_in.
    shuffle_in : object
        Parameter shuffle_in.
    out : object
        Parameter out.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    Write Jacobian from ModEM output to NETCDF file.
    
    author: vrath last changed: July 24, 2020
    """
    try:
        NCfile.close
    except BaseException:
        pass

    DatDim = np.shape(Dat)

    ncout = nc.Dataset(NCfile, 'w', format='NETCDF4')
    ncout.createDimension('data', DatDim[0])

    S = ncout.createVariable(
        'site', str, ('data',), zlib=zlib_in, shuffle=shuffle_in)
    C = ncout.createVariable(
        'comp', str, ('data',), zlib=zlib_in, shuffle=shuffle_in)

    Per = ncout.createVariable(
        'Per', 'float64', ('data',), zlib=zlib_in, shuffle=shuffle_in
    )
    Lat = ncout.createVariable(
        'Lat', 'float64', ('data',), zlib=zlib_in, shuffle=shuffle_in
    )
    Lon = ncout.createVariable(
        'Lon', 'float64', ('data',), zlib=zlib_in, shuffle=shuffle_in
    )
    X = ncout.createVariable(
        'X', 'float64', ('data',), zlib=zlib_in, shuffle=shuffle_in
    )
    Y = ncout.createVariable(
        'Y', 'float64', ('data',), zlib=zlib_in, shuffle=shuffle_in
    )
    Z = ncout.createVariable(
        'Z', 'float64', ('data',), zlib=zlib_in, shuffle=shuffle_in
    )
    Val = ncout.createVariable(
        'Val', 'float64', ('data',), zlib=zlib_in, shuffle=shuffle_in
    )
    Err = ncout.createVariable(
        'Err', 'float64', ('data',), zlib=zlib_in, shuffle=shuffle_in
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
            'writeDatNC: data written to %s in %s format'
            % (NCfile, ncout.data_model)
        )

def write_pars(
    parfile=None,
    outformat='mod',
    dx=None,
    dy=None,
    dz=None,
    val=None,
    reference=None,
    mvalair=None,
    aircells=None,
    header=''
               ):

        # for modem-readable files
        """
        write_pars.
        
        Parameters
        ----------
        parfile : object
            Parameter parfile.
        outformat : object
            Parameter outformat.
        dx : object
            Parameter dx.
        dy : object
            Parameter dy.
        dz : object
            Parameter dz.
        val : object
            Parameter val.
        reference : object
            Parameter reference.
        mvalair : object
            Parameter mvalair.
        aircells : object
            Parameter aircells.
        header : object
            Parameter header.
        
        Returns
        -------
        out : object
            Function return value.
        
        Notes
        -----
        Auto-generated docstring for write_pars.
        """
        if 'mod' in outfmt.lower():
            mod.write_mod(parfile+'_mod', modext='.rho',
                        dx=dx, dy=dy, dz=dz, mval=val,
                        reference=reference, mvalair=Blank, aircells=aircells, header=header)
            print(' Cell volumes (ModEM format) written to '+parfile)
        elif 'ubc' in outfmt.lower():
            elev = -reference[2]
            refubc =  [reference[0], reference[1], elev]
            mod.write_ubc(parfile+'_ubc', modext='.mod', mshext='.msh',
                        dx=dx, dy=dy, dz=dz, mval=val, reference=refubc, mvalair=mvalair, aircells=aircells, header=header)
            print(' Cell volumes (UBC format) written to '+parfile)

        elif 'rlm' in outfmt.lower():
            mod.write_rlm(parfile+'_rlm', modext='_siz.rlm',
                        dx=dx, dy=dy, dz=dz, mval=val, reference=reference, mvalair=Blank, aircells=aircells, comment=header)
            print(' Cell volumes (CGG format) written to '+parfile)


def write_mod_ncd(
    NCfile=None,
    x=None,
    y=None,
    z=None,
    Mod=None,
    Sens=None,
    Ref=None,
    trans='LINEAR',
    zlib_in=True,
    shuffle_in=True,
    out=True,
):
    """
    write_mod_ncd.
    
    Parameters
    ----------
    NCfile : object
        Parameter NCfile.
    x : object
        Parameter x.
    y : object
        Parameter y.
    z : object
        Parameter z.
    Mod : object
        Parameter Mod.
    Sens : object
        Parameter Sens.
    Ref : object
        Parameter Ref.
    trans : object
        Parameter trans.
    zlib_in : object
        Parameter zlib_in.
    shuffle_in : object
        Parameter shuffle_in.
    out : object
        Parameter out.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    Write Model from ModEM output to NETCDF/HDF5 file.
    
    author: vrath last changed: Jan 21, 2021
    """
    ModDim = np.shape(Mod)

    ncout = nc.Dataset(NCfile, 'w', format='NETCDF4')

    ncout.createDimension('msiz', ModDim)
    ncout.createDimension('nx', ModDim[0])
    ncout.createDimension('ny', ModDim[1])
    ncout.createDimension('nz', ModDim[2])

    ncout.createDimension('ref', (3))

    X = ncout.createVariable(
        'x', 'float64', ('nx'), zlib=zlib_in, shuffle=shuffle_in)
    Y = ncout.createVariable(
        'y', 'float64', ('ny'), zlib=zlib_in, shuffle=shuffle_in)
    Z = ncout.createVariable(
        'z', 'float64', ('nz'), zlib=zlib_in, shuffle=shuffle_in)
    X[:] = x[:]
    Y[:] = y[:]
    Z[:] = z[:]

    trans = trans.upper()

    if trans == 'LOGE':
        Mod = np.log(Mod)
        if out:
            print('resistivities to ' + NCfile + ' transformed to: ' + trans)
    elif trans == 'LOG10':
        Mod = np.log10(Mod)
        if out:
            print('resistivities to ' + NCfile + ' transformed to: ' + trans)
    elif trans == 'LINEAR':
        pass
    else:
        print('Transformation: ' + trans + ' not defined!')
        sys.exit(1)

    M = ncout.createVariable(
        'model', 'float64', ('msiz'), zlib=zlib_in, shuffle=shuffle_in
    )
    M[:, :, :] = Mod[:, :, :]

    if Sens is not None:
        S = ncout.createVariable(
            'sens', 'float64', ('msiz'), zlib=zlib_in, shuffle=shuffle_in
        )
        S[:, :, :] = Sens[:, :, :]

    if Ref is not None:
        R = ncout.createVariable(
            'ref', 'float64', ('ref'), zlib=zlib_in, shuffle=shuffle_in
        )
        R[:] = Ref[:]

    ncout.close()

    if out:
        print(
            'write_modelNC: data written to %s in %s format'
            % (NCfile, ncout.data_model)
        )


def write_mod_npz(file=None,
                  dx=None, dy=None, dz=None, mval=None, reference=None,
                  compressed=True, trans='LINEAR',
                  aircells=None, mvalair=1.e17, blank=1.e-30, header='',
                  out=True):
    """
    write_mod_npz.
    
    Parameters
    ----------
    file : object
        Parameter file.
    dx : object
        Parameter dx.
    dy : object
        Parameter dy.
    dz : object
        Parameter dz.
    mval : object
        Parameter mval.
    reference : object
        Parameter reference.
    compressed : object
        Parameter compressed.
    trans : object
        Parameter trans.
    aircells : object
        Parameter aircells.
    mvalair : object
        Parameter mvalair.
    blank : object
        Parameter blank.
    header : object
        Parameter header.
    out : object
        Parameter out.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    Write ModEM model input.
    
    Expects mval in physical units (linear).
    
    author: vrath last changed: Feb 26, 2024
    """

    dims = np.shape(mval)
    if mval.dim == 3:
        nx, ny, nz = dims
    else:
        nx, ny, nz, nset = dims

    if not aircells is None:
        mval[aircells] = mvalair

    if not blank is None:
        blanks = np.where(~np.isfinite(mval))
        mval[blanks] = blank

    if len(header) == 0:
        header = '# 3D MT model written by ModEM in WS format'

    if header[0] != '#':
        header = '#'+header

    if trans is not None:
        trans = trans.upper()

        if trans == 'LOGE':
            mval = np.log(mval)
            mvalair = np.log(mvalair)
            if out:
                print('values to ' + file + ' transformed to: ' + trans)
        elif trans == 'LOG10':
            mval = np.log10(mval)
            mvalair = np.log10(mvalair)
            if out:
                print('values to ' + file + ' transformed to: ' + trans)
        elif trans == 'LINEAR':
            pass

        else:
            print('Transformation: ' + trans + ' not defined!')
            sys.exit(1)

    else:
        trans == 'LINEAR'

    trns = np.array(trans)

    if reference is None:
        ncorner = -0.5*np.sum(dx)
        ecorner = -0.5*np.sum(dy)
        elev = 0.
        cnt = np.array([ncorner, ecorner, elev])
    else:
        cnt = np.asarray(reference)

    info = np.array([trns], dtype='object')

    if compressed:
        modext = '.npz'
        modf = file+modext

        np.savez_compressed(modf, header=header, info=info,
                            dx=dx, dy=dy, dz=dz, mval=mval, reference=cnt)
        print('model written to '+modf)
    else:
        modext = '.npy'
        modf = file+modext
        np.savez(modf, header=header, info=info,
                 dx=dx, dy=dy, dz=dz, mval=mval, reference=cnt)
        print('model written to '+modf)


def write_mod(file=None, modext='.rho',
              dx=None, dy=None, dz=None, mval=None, reference=None,
              trans='LINEAR', aircells=None, mvalair=1.e17, blank=1.e-30, header='', out=True):
    """
    write_mod.
    
    Parameters
    ----------
    file : object
        Parameter file.
    modext : object
        Parameter modext.
    dx : object
        Parameter dx.
    dy : object
        Parameter dy.
    dz : object
        Parameter dz.
    mval : object
        Parameter mval.
    reference : object
        Parameter reference.
    trans : object
        Parameter trans.
    aircells : object
        Parameter aircells.
    mvalair : object
        Parameter mvalair.
    blank : object
        Parameter blank.
    header : object
        Parameter header.
    out : object
        Parameter out.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    Write ModEM model input.
    
    Expects mval in physical units (linear).
    
    author: vrath last changed: Aug 28, 2023
    
    Modem model format in Fortran:
    
    DO iz = 1,Nz     DO iy = 1,Ny         DO ix = Nx,1,-1             READ(10,*)
    mval(ix,iy,iz)         ENDDO     ENDDO ENDDO
    """

    modf = file+modext

    dims = np.shape(mval)

    nx = dims[0]
    ny = dims[1]
    nz = dims[2]
    dummy = 0

    if not aircells is None:
        mval[aircells] = mvalair

    if not blank is None:
        blanks = np.where(~np.isfinite(mval))
        mval[blanks] = blank

    if len(header) == 0:
        header = '# 3D MT model written by ModEM in WS format'

    if header[0] != '#':
        header = '#'+header

    if trans is not None:
        trans = trans.upper()

        if trans == 'LOGE':
            mval = np.log(mval)
            mvalair = np.log(mvalair)
            if out:
                print('values to ' + file + ' transformed to: ' + trans)
        elif trans == 'LOG10':
            mval = np.log10(mval)
            mvalair = np.log10(mvalair)
            if out:
                print('values to ' + file + ' transformed to: ' + trans)
        elif trans == 'LINEAR':
            pass

        else:
            print('Transformation: ' + trans + ' not defined!')
            sys.exit(1)

    else:
        trans == 'LINEAR'

    trns = np.array(trans)

    if reference is None:
        ncorner = -0.5*np.sum(dx)
        ecorner = -0.5*np.sum(dy)
        elev = 0.
        cnt = np.array([ncorner, ecorner, elev])
    else:
        cnt = np.asarray(reference)

    with open(modf, 'w') as f:
        np.savetxt(
            f, [header], fmt='%s')
        line = np.array([nx, ny, nz, dummy, trns], dtype='object')
        # line = np.array([nx, ny, nz, dummy, trans])
        # np.savetxt(f, line.reshape(1, 5), fmt='   %s'*5)
        np.savetxt(f, line.reshape(1, 5), fmt=[
                   '  %i', '  %i', '  %i', '  %i', '  %s'])

        np.savetxt(f, dx.reshape(1, dx.shape[0]), fmt='%12.3f')
        np.savetxt(f, dy.reshape(1, dy.shape[0]), fmt='%12.3f')
        np.savetxt(f, dz.reshape(1, dz.shape[0]), fmt='%12.3f')
        # write out the layers from resmodel
        for zi in range(dz.size):
            f.write('\n')
            for yi in range(dy.size):
                line = mval[::-1, yi, zi]
                # line = np.flipud(mval[:, yi, zi])
                # line = mval[:, yi, zi]
                np.savetxt(f, line.reshape(1, nx), fmt='%12.5e')

        f.write('\n')

        np.savetxt(f, cnt.reshape(1, cnt.shape[0]), fmt='%10.1f')
        f.write('%10.2f  \n' % (0.0))


def write_rlm(file=None, modext='.rlm',
              dx=None, dy=None, dz=None, mval=None, reference=None,
              aircells=None, mvalair=1.e17, blank=1.e-30,
              comment='', name='', out=True):
    """
    write_rlm.
    
    Parameters
    ----------
    file : object
        Parameter file.
    modext : object
        Parameter modext.
    dx : object
        Parameter dx.
    dy : object
        Parameter dy.
    dz : object
        Parameter dz.
    mval : object
        Parameter mval.
    reference : object
        Parameter reference.
    aircells : object
        Parameter aircells.
    mvalair : object
        Parameter mvalair.
    blank : object
        Parameter blank.
    comment : object
        Parameter comment.
    name : object
        Parameter name.
    out : object
        Parameter out.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    Write GGG model input.
    
    conventions:     x = east, y = south, z = down     expects mval in physical
    units (?).
    
    author: vrath last changed: jan 18, 2024
    """

    modf = file+modext

    nx, ny, nz = np.shape(mval)

    if not aircells is None:
        mval[aircells] = mvalair

    if not blank is None:
        blanks = np.where(~np.isfinite(mval))
        mval[blanks] = blank

    if len(comment) == 0:
        comment = '# 3D MT model in RLM format'

    comment = comment.strip()
    if comment[0] != '#':
        comment = '#'+comment

    if len(name) == 0:
        name = file
    if reference is None:
        ncorner = -0.5*np.sum(dx)
        ecorner = -0.5*np.sum(dy)
        elev = 0.
        cnt = np.array([ncorner, ecorner, elev])
    else:
        cnt = np.asarray(reference)

    with open(modf, 'w') as f:

        line = np.array([nx, ny, nz], dtype='object')
        np.savetxt(f, line.reshape(1, 3), fmt='  %i')
        np.savetxt(f, dx.reshape(1, dx.shape[0]), fmt='%12.3f')
        np.savetxt(f, dy.reshape(1, dy.shape[0]), fmt='%12.3f')
        np.savetxt(f, dz.reshape(1, dz.shape[0]), fmt='%12.3f')

        # write out the layers from resmodel
        for zi in range(dz.size):
            f.write(str(zi+1))
            for yi in range(dy.size):
                line = mval[:, yi, zi]
                np.savetxt(f, line.reshape(1, nx), fmt='%12.5e')

        np.savetxt(
            f, [comment], fmt='%s')
        np.savetxt(
            f, [name], fmt='%s')

        np.savetxt(
            f, [1, 1], fmt='%i')

        np.savetxt(f, [cnt[0], cnt[1]], fmt='%16.6g')
        f.write('%10.2f  \n' % (0.0))
        np.savetxt(f, [cnt[2]], fmt='%16.6g')


def write_ubc(file=None,  mshext='.mesh', modext='.ubc',
              dx=None, dy=None, dz=None, mval=None, reference=None,
              aircells=None, mvalair=1.e17, blank=1.e17, header='', out=True):
    """
    write_ubc.
    
    Parameters
    ----------
    file : object
        Parameter file.
    mshext : object
        Parameter mshext.
    modext : object
        Parameter modext.
    dx : object
        Parameter dx.
    dy : object
        Parameter dy.
    dz : object
        Parameter dz.
    mval : object
        Parameter mval.
    reference : object
        Parameter reference.
    aircells : object
        Parameter aircells.
    mvalair : object
        Parameter mvalair.
    blank : object
        Parameter blank.
    header : object
        Parameter header.
    out : object
        Parameter out.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    Write UBC model input.
    
    Expects mval in physical units (linear).
    
    author: vrath last changed: Aug 28, 2023
    """

    modf = file+modext
    mesh = file+mshext

    dims = np.shape(mval)

    if not aircells is None:
        mval.reshape(dims)[aircells] = mvalair

    if not blank is None:
        blanks = np.where(~np.isfinite(mval))
        mval.reshape(dims)[blanks] = mvalair

    dyu = np.flipud(dx.reshape(1, dx.shape[0]))
    dxu = dy.reshape(1, dy.shape[0])
    dzu = dz.reshape(1, dz.shape[0])

    lat = reference[0]
    lon = reference[1]
    utm_zone = utl.get_utm_zone(lat, lon)
    utme, utmn = utl.proj_latlon_to_utm(lat, lon, utm_zone=utm_zone[0])
    ubce = utme - 0.5*np.sum(dxu)
    ubcn = utmn - 0.5*np.sum(dyu)
    refu = np.array([ubce, ubcn, reference[2], utm_zone[0]]).reshape(1, 4)
    # print(refu)

    val = np.transpose(mval, (1, 0, 2))

    dimu = np.shape(val)
    dimu = np.asarray(dimu)
    dimu = dimu.reshape(1, dimu.shape[0])
    val = val.flatten(order='C')

    with open(mesh, 'w') as f:
        np.savetxt(f, dimu, fmt='%i')
        np.savetxt(f, refu, fmt='%14.3f %14.3f %14.3f %10i')

        np.savetxt(f, dxu, fmt='%12.3f')
        np.savetxt(f, dyu, fmt='%12.3f')
        np.savetxt(f, dzu, fmt='%12.3f')

    with open(modf, 'w') as f:
        np.savetxt(f, val, fmt='%14.5g')


def read_ubc(file=None, modext='.mod', mshext='.msh',
             trans='LINEAR', volumes=False, out=True):
    """
    read_ubc.
    
    Parameters
    ----------
    file : object
        Parameter file.
    modext : object
        Parameter modext.
    mshext : object
        Parameter mshext.
    trans : object
        Parameter trans.
    volumes : object
        Parameter volumes.
    out : object
        Parameter out.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    Read UBC model input.
    
    author: vrath last changed: Aug 30, 2023
    """

    modf = file+modext
    mesh = file+mshext

    with open(mesh, 'r') as f:
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

    with open(modf, 'r') as f:
        lines = f.readlines()

    val = np.array([])
    for line in lines:
        val = np.append(val, float(line))
    val = np.reshape(val, (ny, nx, nz))
    val = np.transpose(val, (1, 0, 2))

    # here mval should be in physical units, not log...
    if 'loge' in trans.lower() or 'ln' in trans.lower():
        val = np.log(val)
        if out:
            print('values transformed to: ' + trans)
    elif 'log10' in trans.lower():
        val = np.log10(val)
        if out:
            print('values transformed to: ' + trans)
    else:
        if out:
            print('values transformed to: ' + trans)
        pass

    if out:
        print(
            'read_model: %i x %i x %i model-like read from %s' % (nx, ny, nz, file))

    return dx, dy, dz, val, refubc, trans


# def get_volumes(dx=None, dy=None, dz=None, mval=None, out=True):
#     '''

#     Extract volumes from model.

#     Parameters
#     ----------
#     dx, dy, dz : float arrays
#         Mesh cell sizes.
#     mval : float array
#         Resistivity of cells.
#     out : logical, optional
#         Controls ouput. The default is True.

#     Returns
#     -------
#     vcell :  float array
#         Cell volumes in model mesh.

#     '''
#     nx, ny, nz = np.shape(mval)
#     vcell = np.zeros_like(mval)
#     for ii in np.arange(nx):
#         for jj in np.arange(ny):
#             for kk in np.arange(nz):
#                 vcell[ii, jj, kk] = dx[ii]*dy[jj]*dz[kk]

#     if out:
#         print(
#             'get_volumes: %i x %i x %i cell volumes calculated' %
#             (nx, ny, nz))

#     return vcell


def get_size(dx=None, dy=None, dz=None, mval=None, how='vol', out=True):
    """
    get_size.
    
    Parameters
    ----------
    dx : object
        Parameter dx.
    dy : object
        Parameter dy.
    dz : object
        Parameter dz.
    mval : object
        Parameter mval.
    how : object
        Parameter how.
    out : object
        Parameter out.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    Extract volumes or oother measures of size from model.
    
    Parameters ---------- dx, dy, dz : float arrays     Mesh cell sizes. mval :
    float array     Resistivity of cells. out : logical, optional     Controls
    ouput. The default is True.
    
    Returns ------- cell_size :  float array     Cell volumes in model mesh.
    """
    nx, ny, nz = np.shape(mval)
    cell_size = np.zeros_like(mval)

    if 'vol' in how.lower():
        # for ii in np.arange(nx)[::-1]:
        for ii in np.arange(nx):
            for jj in np.arange(ny):
                for kk in np.arange(nz):
                    cell_size[ii, jj, kk] = dx[ii]*dy[jj]*dz[kk]

        if out:
            print(
                'get_size: %i x %i x %i cell volumes calculated' %
                (nx, ny, nz))

    elif 'hsiz' in how.lower():
        #for ii in np.arange(nx)[::-1]:
        for ii in np.arange(nx):
            for jj in np.arange(ny):
                for kk in np.arange(nz):
                    cell_size[ii, jj, kk] = np.min([dx[ii], dy[jj]])

    elif 'vsiz' in how.lower():
        #for ii in np.arange(nx)[::-1]:
        for ii in np.arange(nx):
            for jj in np.arange(ny):
                for kk in np.arange(nz):
                    cell_size[ii, jj, kk] = dz[kk]

    elif 'area' in how.lower():
        #for ii in np.arange(nx)[::-1]:
        for ii in np.arange(nx):
            for jj in np.arange(ny):
                for kk in np.arange(nz):
                    cell_size[ii, jj, kk] = dx[ii]*dy[jj]
    else:
        sys.exit('get size: method '+how.lower(), 'not implemented! Exit.')

    return cell_size


def get_topo(dx=None, dy=None, dz=None, mval=None,
             ref=[0., 0., 0.],
             mvalair=1.e17,
             out=True):
    """
    get_topo.
    
    Parameters
    ----------
    dx : object
        Parameter dx.
    dy : object
        Parameter dy.
    dz : object
        Parameter dz.
    mval : object
        Parameter mval.
    ref : object
        Parameter ref.
    mvalair : object
        Parameter mvalair.
    out : object
        Parameter out.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    Extract topography from model.
    
    Parameters ---------- dx, dy, dz : float arrays     Mesh cell sizes. mval :
    float array     Resistivity of cells. ref : TYPE, optional     reference
    coordinates. The default is [0., 0., 0.]. mvalair : float, optional     Value
    for air resistivity. needs to be in the     units of input model. The default
    is 1.e17 (physical values). out : logical, optional     Controls ouput. The
    default is True.
    
    Returns ------- xcnt, ycnt: float arrays     Coordinates of cell centers in x-y
    plane. topo: float array nx x ny     Elevation values
    """
    nx, ny, nz = np.shape(mval)

    x = np.append(0.0, np.cumsum(dx))
    xcnt = 0.5 * (x[0:nx] + x[1:nx+1]) + ref[0]

    y = np.append(0.0, np.cumsum(dy))
    ycnt = 0.5 * (y[0:ny] + y[1:ny+1]) + ref[1]

    ztop = np.append(0.0, np.cumsum(dz)) + ref[2]

    topo = np.zeros((nx, ny))
    for ii in np.arange(nx):
        for jj in np.arange(ny):
            col = mval[ii, jj, :]
            nsurf = np.argmax(col < mvalair)
            topo[ii, jj] = ztop[nsurf]

    if out:
        print(
            'get topo: %i x %i cell surfaces marked' % (nx, ny))

    return xcnt, ycnt, topo


def read_mod(file=None,
             modext='.rho',
             trans='LINEAR',
             blank=1.e-30,
             out=True):
    """
    read_mod.
    
    Parameters
    ----------
    file : object
        Parameter file.
    modext : object
        Parameter modext.
    trans : object
        Parameter trans.
    blank : object
        Parameter blank.
    out : object
        Parameter out.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    Read ModEM model input.
    
    Returns mval in physical units
    
    author: vrath last changed: Aug 31, 2023
    """

    modf = file+modext

    with open(modf, 'r') as f:
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
        line = np.flipud(line)  # np.fliplr(line)
        mval = np.append(mval, np.array([float(sub) for sub in line]))

    if out:
        print('values in ' + file + ' are: ' + trns)

    if trns == 'LOGE':
        mval = np.exp(mval)
    elif trns == 'LOG10':
        mval = np.power(10.0, mval)
    elif trns == 'LINEAR':
        pass
    else:
        print('Transformation: ' + trns + ' not defined!')
        sys.exit(1)

    # here mval should be in physical units, not log...
    mval[np.where(np.abs(mval) < blank)] = blank

    if 'loge' in trans.lower() or 'ln' in trans.lower():
        mval = np.log(mval)
        if out:
            print('values transformed to: ' + trans)
    elif 'log10' in trans.lower():
        mval = np.log10(mval)
        if out:
            print('values transformed to: ' + trans)
    else:
        if out:
            print('values transformed to: ' + trans)
        pass

    mval = mval.reshape(dims, order='F')

    reference = [float(sub) for sub in lines[-2][0:3]]

    if out:
        print(
            'read_model: %i x %i x %i model read from %s' % (nx, ny, nz, file))

    return dx, dy, dz, mval, reference, trans


def read_mod_aniso(file=None,
                   components=3,
                   modext='.rho',
                   trans='LINEAR',
                   blank=1.e-30,
                   out=True):
    """
    read_mod_aniso.
    
    Parameters
    ----------
    file : object
        Parameter file.
    components : object
        Parameter components.
    modext : object
        Parameter modext.
    trans : object
        Parameter trans.
    blank : object
        Parameter blank.
    out : object
        Parameter out.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    Read ModEM model input.
    
    author: vrath last changed: Oct 22, 2025
    """

    modf = file+modext

    with open(modf, 'r') as f:
        lines = f.readlines()

    lines = [line.split() for line in lines]
    dims = [int(sub) for sub in lines[1][0:3]]
    nx, ny, nz = dims
    trns = lines[1][4]
    dx = np.array([float(sub) for sub in lines[2]])
    dy = np.array([float(sub) for sub in lines[3]])
    dz = np.array([float(sub) for sub in lines[4]])
    if out:
        print('values in ' + file + ' are: ' + trns)

    mcomps = []
    for icmp in np.arange(components):
        mval = np.array([])
        for line in lines[5:-2]:
            line = np.flipud(line)  # np.fliplr(line)
            mval = np.append(mval, np.array([float(sub) for sub in line]))

        if out:
            print(
                'read_model %i : %i x %i x %i model read from %s' % (icmp, nx, ny, nz, file))

        if trns == 'LOGE':
            mval = np.exp(mval)
        elif trns == 'LOG10':
            mval = np.power(10.0, mval)
        elif trns == 'LINEAR':
            pass
        else:
            print('Transformation: ' + trns + ' not defined!')
            sys.exit(1)

        # here mval should be in physical units, not log...
        mval[np.where(np.abs(mval) < blank)] = blank

        if 'loge' in trans.lower() or 'ln' in trans.lower():
            mval = np.log(mval)
            if out:
                print('values transformed to: ' + trans)
        elif 'log10' in trans.lower():
            mval = np.log10(mval)
            if out:
                print('values transformed to: ' + trans)
        else:
            if out:
                print('values transformed to: ' + trans)
            pass

        mval = mval.reshape(dims, order='F')

        if components > 1:
            mcomps.append(mval)
        else:
            mcomps = mval

    reference = [float(sub) for sub in lines[-2][0:3]]

    return dx, dy, dz, mcomps, reference, trans

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
                  trim=[10, 10, 30], reference=None, scale=[1., 1., -1.],
                  trans='LINEAR', out=True):
    """
    write_mod_vtk.
    
    Parameters
    ----------
    file : object
        Parameter file.
    dx : object
        Parameter dx.
    dy : object
        Parameter dy.
    dz : object
        Parameter dz.
    rho : object
        Parameter rho.
    trim : object
        Parameter trim.
    reference : object
        Parameter reference.
    scale : object
        Parameter scale.
    trans : object
        Parameter trans.
    out : object
        Parameter out.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    write ModEM model input in
    
    author: vrath last changed: Mar 13, 2024
    """
    from evtk.hl import gridToVTK

    if trim is not None:
        print('model trimmed'
              + ', x='+str(trim[0])
              + ', y='+str(trim[1])
              + ', z='+str(trim[2])
              )

        for ix in range(trim[0]):
            dx = np.delete(dx, (0, -1))
        for ix in range(trim[1]):
            dy = np.delete(dy, (0, -1))
        for ix in range(trim[2]):
            dz = np.delete(dz, (-1))

    X = np.append(0.0, np.cumsum(dy))*scale[1]
    Y = np.append(0.0, np.cumsum(dx))*scale[1]
    Z = np.append(0.0, np.cumsum(dz))*scale[2]

    gridToVTK(file, X, Y, -Z, cellData={'resistivity (in Ohm)': rho})
    print('model-like parameter written to %s' % (file))


def write_dat_vtk(Sitfile=None, sx=None, sy=None, sz=None, sname=None,
                  reference=None, scale=[1., 1., -1.], out=True):
    """
    write_dat_vtk.
    
    Parameters
    ----------
    Sitfile : object
        Parameter Sitfile.
    sx : object
        Parameter sx.
    sy : object
        Parameter sy.
    sz : object
        Parameter sz.
    sname : object
        Parameter sname.
    reference : object
        Parameter reference.
    scale : object
        Parameter scale.
    out : object
        Parameter out.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    Convert ModEM data file to VTK station set (unstructured grid)
    """
    from evtk.hl import pointsToVTK

    N = sx*scale[0]
    E = sy*scale[1]
    D = sz*scale[2]

    # dummy scalar values
    dummy = np.ones((len(N)))

    pointsToVTK(Sitfile, N, E, D, data={'value': dummy})

    print('site positions written to %s' % (Sitfile))


def fix_cells(covfile_i=None,
              covfile_o=None,
              modfile_i=None,
              modfile_o=None,
              datfile_i=None,
              fixed='2',
              method=['border', 3],
              fixmod=['prior'],
              unit='km',
              out=True):
    """
    fix_cells.
    
    Parameters
    ----------
    covfile_i : object
        Parameter covfile_i.
    covfile_o : object
        Parameter covfile_o.
    modfile_i : object
        Parameter modfile_i.
    modfile_o : object
        Parameter modfile_o.
    datfile_i : object
        Parameter datfile_i.
    fixed : object
        Parameter fixed.
    method : object
        Parameter method.
    fixmod : object
        Parameter fixmod.
    unit : object
        Parameter unit.
    out : object
        Parameter out.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    Read and process ModEM covar input.
    
    author: vrath last changed: June, 2023
    """
    air = '0'
    ocean = '9'
    comments = ['#', '|', '>', '+', '/']

    dx, dy, dz, rho, reference, _ = read_mod(modfile_i, out=True)
    modsize = np.shape(rho)

    if 'dist' in method[0].lower():
        fixdist = method[1]
        x = np.append(0., np.cumsum(dx)) + reference[0]
        xc = 0.5*(x[0:len(x)-1]+x[1:len(x)])
        y = np.append(0., np.cumsum(dy)) + reference[1]
        yc = 0.5*(y[0:len(y)-1]+y[1:len(y)])
        cellcent = [xc, yc]

        # print(len(xc),len(yc))
        Site, _, Data, _ = read_data(datfile_i, out=True)

        xs = []
        ys = []
        for idt in range(0, np.size(Site)):
            ss = Site[idt]
            if idt == 0:
                site = Site[idt]
                xs.append(Data[idt, 3])
                ys.append(Data[idt, 4])
            elif ss != site:
                site = Site[idt]
                xs.append(Data[idt, 3])
                ys.append(Data[idt, 4])

        sitepos = [xs, ys]

    if 'bord' in method[0].lower():
        border = method[1]

    with open(covfile_i, 'r') as f_i:
        l_i = f_i.readlines()

    l_o = l_i.copy()

    done = False
    for line in l_i:
        if len(line.split()) == 3:
            [block_len, line_len, num_lay] = [int(t) for t in line.split()]
            print(block_len, line_len, num_lay)
            done = True
        if done:
            break

    if 'bord' in method[0].lower():
        rows = list(range(0, block_len))
        index_row1 = [index for index in rows if rows[index] < border]
        index_row2 = [index for index in rows if rows[index]
                      > block_len-border-1]
        cols = list(range(0, line_len))
        index_col1 = [index for index in cols if cols[index] < border]
        index_col2 = [index for index in cols if cols[index]
                      > line_len-border-1]
    if 'dist' in method.lower():
        sits = list(range(0, np.shape(sitepos)[0]))
        rows = list(range(0, block_len))
        cols = list(range(0, line_len))
        xs = sitepos[:][0]
        ys = sitepos[:][1]

    blocks = [ii for ii in range(len(l_i)) if len(l_i[ii].split()) == 2]
    if len(blocks) != num_lay:
        print('fix_cells: Number of blocks wrong! Exit.')

    for ib in blocks:
        new_block = []
        block = l_i[ib+1:ib+block_len+1]
        tmp = [line.split() for line in block]

        if 'bord' in method.lower():

            for ii in rows:
                if (ii in index_row1) or (ii in index_row2):
                    tmp[ii] = [tmp[ii][cell].replace(
                        '1', fixed) for cell in cols]

                # print(ii,tmp[ii] ,'\n')

                for jj in cols:
                    if (jj in index_col1) or (jj in index_col2):
                        tmp[ii][jj] = tmp[ii][jj].replace('1', fixed)

                tmp[ii].append('\n')
                new_block.append(' '.join(tmp[ii]))

        if 'dist' in method.lower():

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
                        tmp[ii][jj] = tmp[ii][jj].replace('1', fixed)

                tmp[ii].append('\n')
                new_block.append(' '.join(tmp[ii]))

        l_o[ib+1:ib+block_len+1] = new_block

    with open(covfile_o, 'w') as f_o:
        f_o.writelines(l_o)
    if out:
        print('fix_cells: covariance control read from %s' % (covfile_i))
        print('fix_cells: covariance control written to %s' % (covfile_o))
        if 'bord' in method.lower():
            print(str(border)+' border  cells fixed (zone '+str(fixed)+')')
        else:
            if unit == 'km':
                print('cells with min distance to site > '
                      + str(fixdist/1000)+'km fixed (zone '+str(fixed)+')')
            else:
                print('cells with min distance to site > '
                      + str(fixdist)+'m fixed (zone '+str(fixed)+')')

    if 'val' in fixmod[0].lower():
        write_mod(modfile_o, dx, dy, dz, rho, reference, out=True)
        if out:
            print('fix_cells: model written to %s' % (covfile_o))
            print('fix_cells: model in %s fixed to %g Ohm.m' % (covfile_i))
    else:
        if out:
            print('fix_cells: model in %s fixed to prior' % (modfile_i))


def linear_interpolation(p1, p2, x0):
    """
    linear_interpolation.
    
    Parameters
    ----------
    p1 : object
        Parameter p1.
    p2 : object
        Parameter p2.
    x0 : object
        Parameter x0.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    Function that receives as arguments the coordinates of two points (x,y)
    
    and returns the linear interpolation of a y0 in a given x0 position. This is
    the equivalent to obtaining y0 = y1 + (y2 - y1)*((x0-x1)/(x2-x1)). Look into
    https://en.wikipedia.org/wiki/Linear_interpolation for more information.
    
    Parameters ---------- p1     : tuple (floats)     Tuple (x,y) of a first point
    in a line. p2     : tuple (floats)     Tuple (x,y) of a second point in a line.
    x0     : float     X coordinate on which you want to interpolate a y0.
    
    Return float (interpolated y0 value)
    """
    y0 = p1[1] + (p2[1] - p1[1]) * ((x0 - p1[0]) / (p2[0] - p1[0]))

    return y0


def data_to_pv(data=None, site=None, reference=None, scale=1.):

    """
    data_to_pv.
    
    Parameters
    ----------
    data : object
        Parameter data.
    site : object
        Parameter site.
    reference : object
        Parameter reference.
    scale : object
        Parameter scale.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    Auto-generated docstring for data_to_pv.
    """
    x = data[:, 3]
    y = data[:, 4]
    z = data[:, 5]

    if reference is not None:
        x = x + reference[0]
        y = y + reference[1]
        z = z + reference[2]

    y, x, z = scale*x, scale*y, -scale*z

    sites, siteindex = np.unique(site, return_index=True)
    x = x[siteindex]
    y = y[siteindex]
    z = z[siteindex]

    sites = sites.astype('<U4')
    siten = np.array([ii for ii in np.arange(len(z))])

    # z = -z

    return x, y, z, sites, siten


def model_to_pv(dx=None, dy=None, dz=None, rho=None, reference=None,
                scale=1., pad=[12, 12, 30.]):

    """
    model_to_pv.
    
    Parameters
    ----------
    dx : object
        Parameter dx.
    dy : object
        Parameter dy.
    dz : object
        Parameter dz.
    rho : object
        Parameter rho.
    reference : object
        Parameter reference.
    scale : object
        Parameter scale.
    pad : object
        Parameter pad.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    Auto-generated docstring for model_to_pv.
    """
    x, y, z = cells3d(dx, dy, dz)

    x = x + reference[0]
    y = y + reference[1]
    z = z + reference[2]

    x, y, z = scale*x, scale*y, scale*z

    x, y, z, rho = clip_model(x, y, z, rho, pad=pad)

    # vals = np.swapaxes(np.flip(rho, 2), 0, 1).flatten(order='F')
    vals = rho.copy()
    vals = np.swapaxes(vals, 0, 1)
    # vals = np.flip(rho.copy(), 2)
    vals = vals.flatten(order='F')

    z = -z
    return x, y, z, vals


def clip_model(x, y, z, rho,
               pad=[0, 0, 0], centers=False, scale=[1., 1., 1.]):
    """
    clip_model.
    
    Parameters
    ----------
    x : object
        Parameter x.
    y : object
        Parameter y.
    z : object
        Parameter z.
    rho : object
        Parameter rho.
    pad : object
        Parameter pad.
    centers : object
        Parameter centers.
    scale : object
        Parameter scale.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    Clip model to ROI.
    
    Parameters ---------- x, y, z : float     Node coordinates rho : float
    resistivity/sensitivity/diff values. pad : integer, optional     padding in
    x/y/z. The default is [0, 0, 0]. centers: bool, optional     nodes or centers.
    The default is False (i.e. nodes). scale: float     scling, e.g. to km (1E-3).
    The default is [1., 1.,1.].
    
    Returns ------- xn, yn, zn, rhon
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
        print('cells3d returning cell center coordinates.')
        xn = 0.5 * (xn[:-1] + xn[1:])
        yn = 0.5 * (yn[:-1] + yn[1:])
        zn = 0.5 * (zn[:-1] + zn[1:])

    return xn, yn, zn, rhon


def insert_body_condition(dx=None, dy=None, dz=None,
                          rho_in=None, body=None,
                          smooth=None, scale=1.0,
                          reference=None, pad=[0, 0, 0],
                          out=True):
    """
    insert_body_condition.
    
    Parameters
    ----------
    dx : object
        Parameter dx.
    dy : object
        Parameter dy.
    dz : object
        Parameter dz.
    rho_in : object
        Parameter rho_in.
    body : object
        Parameter body.
    smooth : object
        Parameter smooth.
    scale : object
        Parameter scale.
    reference : object
        Parameter reference.
    pad : object
        Parameter pad.
    out : object
        Parameter out.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
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

    xc = xc + modcenter[0]
    yc = yc + modcenter[1]
    zc = zc + modcenter[2]

    print(' center is', modcenter)

    nx = np.shape(xc)[0]
    ny = np.shape(yc)[0]
    nz = np.shape(zc)[0]

    rho_out = np.log(rho_in.copy())

    # geom= body[0]
    # action = body[1]
    # rhoval = body[2]
    # bcent = body[3:6]
    # baxes = body[6:9]
    # bangl = body[9:12]
    # ell = ['ell', action, 10000.,    0., 0., 10000.,    30000., 30000., 5000.,     0., 0.,0.]
    geom = body[0]
    action = body[1]
    condit = body[2]
    bcent = body[3:6]
    baxes = body[6:9]
    bangl = body[9:12]
    # action = ['rep', 30.]
    # condition = 'val <= np.log10(30.)'
    # ell = ['ell', action, condition,    0., 0., 10000.,    30000., 30000., 5000.,     0., 0.,0.]
    rhoval = action[1]
    rhoval = np.log(rhoval)

    if 'rep' in action[0]:
        actstring = 'rhoval'

    elif 'add' in action[0]:
        if 'avg' in action[0]:
            if condit is None:
                actstring = 'rho_avg + rhoval'
            else:
                print('Average add option not consistent with contition! Exit.')
        else:
            actstring = 'rho_out[point] + rhoval'

    else:
        print('Action' + action + ' not implemented! Exit.')

    if out:
        print(
            'Body type   : ' + geom + ', ' + action[0] + ' rho =',
            str(np.exp(rhoval)) + ' Ohm.m',
        )
        print('Body center : ' + str(bcent))
        print('Body axes   : ' + str(baxes))
        print('Body angles : ' + str(bangl))
        print('Action is '+action[0])
        print('Smoothed with ' + smooth[0] + ' filter')

    if 'ell' in geom.lower():
        if 'avg' in actstring:
            rho_avg = 0.
            n_inside = 0
            for kk in np.arange(0, nz - zpad - 1):
                zpoint = zc[kk]
                for jj in np.arange(ypad + 1, ny - ypad - 1):
                    ypoint = yc[jj]
                    for ii in np.arange(xpad + 1, nx - xpad - 1):
                        xpoint = xc[ii]
                        point = [ii, jj, kk]
                        position = [xpoint, ypoint, zpoint]
                        if in_ellipsoid(position, bcent, baxes, bangl):
                            n_inside = n_inside + 1
                            rho_avg = rho_avg + rho_out[ii, jj, kk]
            if n_inside > 0:
                rho_avg = rho_avg/n_inside
            else:
                print('insert_body: no cell centers inside ellipsoid! Exit.')

        n_inside = 0
        n_changed = 0
        for kk in np.arange(0, nz - zpad - 1):
            zpoint = zc[kk]
            for jj in np.arange(ypad + 1, ny - ypad - 1):
                ypoint = yc[jj]
                for ii in np.arange(xpad + 1, nx - xpad - 1):
                    xpoint = xc[ii]
                    point = [ii, jj, kk]

                    position = [xpoint, ypoint, zpoint]
                    if in_ellipsoid(position, bcent, baxes, bangl):
                        n_inside = n_inside + 1
                        if condit is None:
                            rho_out[ii, jj, kk] = eval(actstring)
                        else:
                            val = rho_out[ii, jj, kk]
                            if eval(condit):
                                n_changed = n_changed + 1
                                rho_out[ii, jj, kk] = eval(actstring)

        if n_inside > 0:
            print(n_inside, ' cell centers in ellipsoid found.')
            print(n_changed, ' cells changed.')
        else:
            print('insert_body: no cell centers inside ellipsoid! Exit.')

    if 'box' in geom.lower():
        if 'avg' in actstring:
            rho_avg = 0.
            n_inside = 0
            for kk in np.arange(0, nz - zpad - 1):
                zpoint = zc[kk]
                for jj in np.arange(ypad + 1, ny - ypad - 1):
                    ypoint = yc[jj]
                    for ii in np.arange(xpad + 1, nx - xpad - 1):
                        xpoint = xc[ii]
                        point = [ii, jj, kk]
                        position = [xpoint, ypoint, zpoint]
                        if in_box(position, bcent, baxes, bangl):
                            n_inside = n_inside + 1
                            rho_avg = rho_avg + rho_out[ii, jj, kk]
            if n_inside > 0:
                rho_avg = rho_avg/n_inside
            else:
                print('insert_body: no cell centers inside box! Exit.')

        n_inside = 0
        n_changed = 0
        for kk in np.arange(0, nz - zpad - 1):
            zpoint = zc[kk]
            for jj in np.arange(ypad + 1, ny - ypad - 1):
                ypoint = yc[jj]
                for ii in np.arange(xpad + 1, nx - xpad - 1):
                    xpoint = xc[ii]
                    point = [ii, jj, kk]
                    position = [xpoint, ypoint, zpoint]
                    if in_box(position, bcent, baxes, bangl):
                        n_inside = n_inside + 1
                        if condit is None:
                            rho_out[ii, jj, kk] = eval(actstring)
                        else:
                            val = rho_out[ii, jj, kk]
                            if eval(condit):
                                n_changed = n_changed + 1
                                rho_out[ii, jj, kk] = eval(actstring)
        if n_inside > 0:
            print(n_inside, ' cell centers in box found.')
            print(n_changed, ' cells changed.')
        else:
            print('insert_body: no cell centers inside box! Exit.')

    if smooth is not None:
        if 'uni' in smooth[0].lower():
            fsize = smooth[1]
            rho_out = uniform_filter(rho_out, fsize)

        elif 'gau' in smooth[0].lower():
            gstd = smooth[1]
            rho_out = gaussian_filter(rho_out, gstd)

        else:
            print('Smoothing filter  ' + smooth[0] + ' not implemented! Exit.')

    rho_out = np.exp(rho_out)

    return rho_out


def insert_body(dx=None, dy=None, dz=None,
                rho_in=None, body=None,
                pad=[0, 0, 0],
                smooth=None, scale=1.0, reference=None,
                out=True):
    """
    insert_body.
    
    Parameters
    ----------
    dx : object
        Parameter dx.
    dy : object
        Parameter dy.
    dz : object
        Parameter dz.
    rho_in : object
        Parameter rho_in.
    body : object
        Parameter body.
    pad : object
        Parameter pad.
    smooth : object
        Parameter smooth.
    scale : object
        Parameter scale.
    reference : object
        Parameter reference.
    out : object
        Parameter out.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
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

    xc = xc + modcenter[0]
    yc = yc + modcenter[1]
    zc = zc + modcenter[2]

    print(' center is', modcenter)

    nx = np.shape(xc)[0]
    ny = np.shape(yc)[0]
    nz = np.shape(zc)[0]

    rho_out = np.log(rho_in.copy())

    geom = body[0]
    action = body[1]
    rhoval = body[2]
    bcent = body[3:6]
    baxes = body[6:9]
    bangl = body[9:12]

    rhoval = np.log(rhoval)

    if 'rep' in action:
        actstring = 'rhoval'

    elif 'add' in action:

        if 'avg' in action:
            actstring = 'rho_avg + rhoval'
        else:
            actstring = 'rho_out[point] + rhoval'

    else:
        print('Action' + action + ' not implemented! Exit.')

    if out:
        print(
            'Body type   : ' + geom + ', ' + action + ' rho =',
            str(np.exp(rhoval)) + ' Ohm.m',
        )
        print('Body center : ' + str(bcent))
        print('Body axes   : ' + str(baxes))
        print('Body angles : ' + str(bangl))
        print('Action is '+action)
        print('Smoothed with ' + smooth[0] + ' filter')

    if 'ell' in geom.lower():
        if 'avg' in actstring:
            rho_avg = 0.
            n_inside = 0
            for kk in np.arange(0, nz - zpad - 1):
                zpoint = zc[kk]
                for jj in np.arange(ypad + 1, ny - ypad - 1):
                    ypoint = yc[jj]
                    for ii in np.arange(xpad + 1, nx - xpad - 1):
                        xpoint = xc[ii]
                        position = [xpoint, ypoint, zpoint]
                        if in_ellipsoid(position, bcent, baxes, bangl):
                            n_inside = n_inside + 1
                            rho_avg = rho_avg + rho_out[ii, jj, kk]
            if n_inside > 0:
                rho_avg = rho_avg/n_inside
            else:
                print('insert_body: no points inside ellipsoid! Exit.')

        n_inside = 0
        for kk in np.arange(0, nz - zpad - 1):
            zpoint = zc[kk]
            for jj in np.arange(ypad + 1, ny - ypad - 1):
                ypoint = yc[jj]
                for ii in np.arange(xpad + 1, nx - xpad - 1):
                    xpoint = xc[ii]
                    position = [xpoint, ypoint, zpoint]
                    if in_ellipsoid(position, bcent, baxes, bangl):
                        n_inside = n_inside + 1
                        rho_out[ii, jj, kk] = eval(actstring)

        print(n_inside, ' points in ellipsoid found.')

    if 'box' in geom.lower():
        if 'avg' in actstring:
            rho_avg = 0.

            n_inside = 0
            for kk in np.arange(0, nz - zpad - 1):
                zpoint = zc[kk]
                for jj in np.arange(ypad + 1, ny - ypad - 1):
                    ypoint = yc[jj]
                    for ii in np.arange(xpad + 1, nx - xpad - 1):
                        xpoint = xc[ii]
                        position = [xpoint, ypoint, zpoint]
                        if in_box(position, bcent, baxes, bangl):
                            n_inside = n_inside + 1
                            rho_avg = rho_avg + rho_out[ii, jj, kk]
            if n_inside > 0:
                rho_avg = rho_avg/n_inside
            else:
                print('insert_body: no points inside box! Exit.')

        n_inside = 0
        for kk in np.arange(0, nz - zpad - 1):
            zpoint = zc[kk]
            for jj in np.arange(ypad + 1, ny - ypad - 1):
                ypoint = yc[jj]
                for ii in np.arange(xpad + 1, nx - xpad - 1):
                    xpoint = xc[ii]
                    position = [xpoint, ypoint, zpoint]
                    if in_box(position, bcent, baxes, bangl):
                        n_inside = n_inside + 1
                        rho_out[ii, jj, kk] = eval(actstring)

        print(n_inside, ' points in box found.')

    if smooth is not None:
        if 'uni' in smooth[0].lower():
            fsize = smooth[1]
            rho_out = uniform_filter(rho_out, fsize)

        elif 'gau' in smooth[0].lower():
            gstd = smooth[1]
            rho_out = gaussian_filter(rho_out, gstd)

        else:
            print('Smoothing filter  ' + smooth[0] + ' not implemented! Exit.')

    rho_out = np.exp(rho_out)

    return rho_out


def cells3d(dx, dy, dz, center=False, reference=[0., 0., 0.]):
    """
    cells3d.
    
    Parameters
    ----------
    dx : object
        Parameter dx.
    dy : object
        Parameter dy.
    dz : object
        Parameter dz.
    center : object
        Parameter center.
    reference : object
        Parameter reference.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    Define cell coordinates.
    
    dx, dy, dz in m, Created on Sat Jan 2 10:35:28 2021
    
    @author: vrath
    """
    x = np.append(0.0, np.cumsum(dx))
    y = np.append(0.0, np.cumsum(dy))
    z = np.append(0.0, np.cumsum(dz))

    x = x + reference[0]
    y = y + reference[1]
    z = z + reference[2]

    if center:
        print('cells3d returning cell center coordinates.')
        xc = 0.5 * (x[:-1] + x[1:])
        yc = 0.5 * (y[:-1] + y[1:])
        zc = 0.5 * (z[:-1] + z[1:])
        return xc, yc, zc

    else:
        print('cells3d returning raw node coordinates.')
        return x, y, z


def in_ellipsoid(
        point=None,
        cent=[0.0, 0.0, 0.0],
        axs=[1.0, 1.0, 1.0],
        ang=[0.0, 0.0, 0.0],
        find_inside=True):
    """
    in_ellipsoid.
    
    Parameters
    ----------
    point : object
        Parameter point.
    cent : object
        Parameter cent.
    axs : object
        Parameter axs.
    ang : object
        Parameter ang.
    find_inside : object
        Parameter find_inside.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    Find points inside arbitrary box.
    
    Defined by the 3-vectors cent, axs, ang vr dec 2020
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
    in_box.
    
    Parameters
    ----------
    point : object
        Parameter point.
    cent : object
        Parameter cent.
    axs : object
        Parameter axs.
    ang : object
        Parameter ang.
    find_inside : object
        Parameter find_inside.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    Find points inside arbitrary ellipsoid.
    
    Defined by the 3-vectors cent, axs, ang vr dec 2020
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
    rotz.
    
    Parameters
    ----------
    theta : object
        Parameter theta.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
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
    roty.
    
    Parameters
    ----------
    theta : object
        Parameter theta.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
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
    rotx.
    
    Parameters
    ----------
    theta : object
        Parameter theta.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
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
              mesh=[np.array([]), np.array([]), np.array([])],
              Out=True):
    """
    crossgrad.
    
    Parameters
    ----------
    m1 : object
        Parameter m1.
    m2 : object
        Parameter m2.
    mesh : object
        Parameter mesh.
    Out : object
        Parameter Out.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    Crossgrad function
    
    See: Rosenkjaer GK, Gasperikova E, Newman, GA, Arnason K, Lindsey NJ (2015)
    Comparison of 3D MT inversions for geothermal exploration: Case studies     for
    Krafla and Hengill geothermal systems in Iceland     Geothermics , Vol. 57,
    258-274
    
    Schnaidt, S. (2015)     Improving Uncertainty Estimation in Geophysical
    Inversion Modelling     PhD thesis, University of Adelaide, AU
    
    vr  July 2023
    """
    sm = np.shape(m1)
    dm = m1.dim
    if dm == 1:
        print('crossgrad: For dim='+str(dm)+' no crossgrad! Exit.')
    elif dm == 2:
        cgdim = 1
    else:
        cgdim = 3

    gm1 = np.gradient(m1)
    gm2 = np.gradient(m2)
    sgm = np.shape(gm1)

    g1 = np.ravel(gm1)
    g2 = np.ravel(gm2)

    cgm = np.zeros_like(g1, cgdim)
    for k in np.arange(np.size(g1)):
        cgm[k, :] = np.cross(g1[k], g2[k])

    cgm = np.reshape(cgm, (sm+cgdim))

    cgnm = np.abs(cgm)/(np.abs(gm1)*np.abs(gm2))

    return cgm, cgnm


def medfilt3D(
        M,
        kernel_size=[3, 3, 3], boundary_mode='nearest', maxiter=1, Out=True):
    """
    medfilt3D.
    
    Parameters
    ----------
    M : object
        Parameter M.
    kernel_size : object
        Parameter kernel_size.
    boundary_mode : object
        Parameter boundary_mode.
    maxiter : object
        Parameter maxiter.
    Out : object
        Parameter Out.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    Run iterated median filter in nD.
    
    vr  Jan 2021
    """
    tmp = M.copy()
    for it in range(maxiter):
        if Out:
            print('iteration: ' + str(it))
        tmp = median_filter(tmp, size=kernel_size, mode=boundary_mode)

    G = tmp.copy()

    return G


def anidiff3D(
        M,
        ckappa=50, dgamma=0.1, foption=1, maxiter=30, Out=True):
    """
    anidiff3D.
    
    Parameters
    ----------
    M : object
        Parameter M.
    ckappa : object
        Parameter ckappa.
    dgamma : object
        Parameter dgamma.
    foption : object
        Parameter foption.
    maxiter : object
        Parameter maxiter.
    Out : object
        Parameter Out.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
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
    anisodiff3D.
    
    Parameters
    ----------
    stack : object
        Parameter stack.
    niter : object
        Parameter niter.
    kappa : object
        Parameter kappa.
    gamma : object
        Parameter gamma.
    step : object
        Parameter step.
    option : object
        Parameter option.
    ploton : object
        Parameter ploton.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    Apply 3D Anisotropic diffusion.
    
    Usage: stackout = anisodiff(stack, niter, kappa, gamma, option)
    
    Arguments:         stack  - input stack         niter  - number of iterations
    kappa  - conduction coefficient 20-100 ?         gamma  - max value of .25 for
    stability         step   - tuple, the distance between adjacent pixels in
    (z,y,x)         option - 1 Perona Malik diffusion equation No 1
    2 Perona Malik diffusion equation No 2         ploton - if True, the middle
    z-plane will be plotted on every                  iteration
    
    Returns:         stackout   - diffused stack.
    
    kappa controls conduction as a function of gradient.  If kappa is low small
    intensity gradients are able to block conduction and hence diffusion across
    step edges.  A large value reduces the influence of intensity gradients on
    conduction.
    
    gamma controls speed of diffusion (you usually want it at a maximum of 0.25)
    
    step is used to scale the gradients in case the spacing between adjacent pixels
    differs in the x,y and/or z axes
    
    Diffusion equation 1 favours high contrast edges over low contrast ones.
    Diffusion equation 2 favours wide regions over smaller ones.
    
    Reference: P. Perona and J. Malik. Scale-space and edge detection using
    ansotropic diffusion. IEEE Transactions on Pattern Analysis and Machine
    Intelligence, 12(7):629-639, July 1990.
    
    Original MATLAB code by Peter Kovesi School of Computer Science & Software
    Engineering The University of Western Australia pk @ csse uwa edu au
    <http://www.csse.uwa.edu.au>
    
    Translated to Python and optimised by Alistair Muldal Department of
    Pharmacology University of Oxford <alistair.muldal@pharm.ox.ac.uk>
    
    June 2000  original version. March 2002 corrected diffusion eqn No 2. July 2012
    translated to Python Jan 2021 slightly adapted python3 VR
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

        fig = pl.figure(figsize=(20, 5.5), num='Anisotropic diffusion')
        ax1, ax2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)

        ax1.imshow(
            stack[showplane, ...].squeeze(),
            interpolation='nearest')
        ih = ax2.imshow(
            stackout[showplane, ...].squeeze(),
            interpolation='nearest', animated=True
        )
        ax1.set_title('Original stack (Z = %i)' % showplane)
        ax2.set_title('Iteration 0')

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
            iterstring = 'Iteration %i' % (ii + 1)
            ih.set_data(stackout[showplane, ...].squeeze())
            ax2.set_title(iterstring)
            fig.canvas.draw()
            # sleep(0.01)

    return stackout


def shock3d(
        M,
        dt=0.2, maxiter=30, filt=[3, 3, 3, 0.5],
        boundary_mode='nearest', signfunc=None):
    """
    shock3d.
    
    Parameters
    ----------
    M : object
        Parameter M.
    dt : object
        Parameter dt.
    maxiter : object
        Parameter maxiter.
    filt : object
        Parameter filt.
    boundary_mode : object
        Parameter boundary_mode.
    signfunc : object
        Parameter signfunc.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    Apply shock filter in nD.
    
    vr  Jan 2021
    """
    if signfunc is None or signfunc == 'sign':
        signcall = '-np.sign(L)'

    elif signfunc[0] == 'sigmoid':
        scale = 1.0
        signcall = '-1./(1. + np.exp(-scale *L))'

    else:
        print('sign func ' + signfunc + ' not defined! Exit.')

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
    gauss3D.
    
    Parameters
    ----------
    Kshape : object
        Parameter Kshape.
    Ksigma : object
        Parameter Ksigma.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    Define 2D gaussian mask.
    
    Should give the same result as MATLAB's fspecial('gaussian',[shape],[sigma])
    
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
    prepare_model.
    
    Parameters
    ----------
    rho : object
        Parameter rho.
    rhoair : object
        Parameter rhoair.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    Prepare model for filtering etc.
    
    Mainly redefining the boundaries (in the case of topograpy) Air domain is filed
    with vertical surface value Created on Tue Jan  5 11:59:42 2021
    
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


def insert_body_ijk(template=None, rho_in=None,
                    perturb=None, bodymask=None, out=True):
    """
    insert_body_ijk.
    
    Parameters
    ----------
    template : object
        Parameter template.
    rho_in : object
        Parameter rho_in.
    perturb : object
        Parameter perturb.
    bodymask : object
        Parameter bodymask.
    out : object
        Parameter out.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    Insert 3d box into given model.
    
    @author: vrath
    """
    if template is None:
        print('insert_body_ijk: no template! Exit.')

    if rho_in is None:
        print('insert_body_ijk: no base model! Exit.')

    if perturb is None:
        print('insert_body_ijk: no perturbation! Exit.')

    if bodymask is None:
        print('insert_body_ijk: no body! Exit.')

    rho_out = np.log(rho_in.copy())

    if out:
        print('Perturbation amplitude: '+str(perturb) + ' log10')

    centers = np.where(template != 0.)
    _, nbody = np.shape(centers)

    bw = bodymask

    for ibody in np.arange(nbody):

        bc = centers[:, ibody]

        ib = np.arange(bc[0]-bw[0], bc[0]+bw[0]+1)
        jb = np.arange(bc[1]-bw[1], bc[1]+bw[1]+1)
        kb = np.arange(bc[1]-bw[1], bc[1]+bw[1]+1)
        rho_out[ib, jb, kb] = template[bc]*perturb+rho_in[ib, jb, kb]

        if out:
            print('Body center at: ' + str(bc))

    return rho_out


def distribute_bodies_ijk(model=None,
                          method=['random', 25, 'uniform',
                                  [1, 1,   1, 1,   1, 1]],
                          valmark=1, flip='alternate', scale='ijk'):
    """
    distribute_bodies_ijk.
    
    Parameters
    ----------
    model : object
        Parameter model.
    method : object
        Parameter method.
    valmark : object
        Parameter valmark.
    flip : object
        Parameter flip.
    scale : object
        Parameter scale.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    construct templates for  distributing test boduies  within model.
    
    Parameters ---------- model : np.array, float     model setup in ModEM format.
    The default is None. method : list of objects, optional
    
    Contains parameters  for  generating centers.
    
    method[0] = 'regular':         ['regular', bounding box (cell indices),
    bodymask, step]         Example: method = ['regular', [1, 1,   1, 1,   1, 1],
    [4, 4, 6],]
    
    method[0] = 'ramdom':         ['random', number of bodies, bounding box (cell
    indices),          distribution (currently only uniform), minimum distance]
    Example: method = ['random', 25, [1, 1,   1, 1,   1, 1], 'uniform', ].
    
    valmark : float         Marker value (e.g 1.),
    
    flip : string or None      flip = 'alt'          sign change modulo 2      flip
    = 'ran'          random sign change
    
    Returns ------- template : np.array, float     zeros, like input model, with
    marker values at body centers git push
    
    @author: vrath, Feb 2024
    """
    if 'ijk' not in scale:
        print('distribute_bodies: currently only index sales possible! Exit.')

    if model is None:
        print('distribute_bodies: no model given! Exit.')

    rng = np.random.default_rng()
    template = np.zeros_like(model)

    if 'reg' in method[0].lower():

        bbox = method[1]
        step = method[2]

        ci = np.arange(bbox[0], bbox[1], step[0])
        cj = np.arange(bbox[2], bbox[3], step[1])
        ck = np.arange(bbox[4], bbox[5], step[2])
        centi, centj, centk = np.meshgrid(ci, cj, ck, indexing='ij')

        bnum = np.shape(centi)
        print(bnum)

        for ibody in np.arange(bnum):
            val = valmark
            if 'alt' in flip:
                if np.mod(ibody, 2) == 0:
                    val = -valmark
            if 'ran' in flip:
                if rng.random() > 0.5:
                    val = -valmark

            template[centi[ibody], centj[ibody], centk[ibody]] = val

    elif 'ran' in method[0].lower():
        print('distribute_bodies: method' +
              method.lower()+'not implemented! Exit.')

        bnum = method[1]
        bbox = method[2]
        bpdf = method[3]
        print('distribute_bodies: currently only uniform diributions'
              + ' implemented! input pdf ignored!')
        mdist = method[4]

        ci = np.arange(bbox[0], bbox[1], 1)
        cj = np.arange(bbox[2], bbox[3], 1)
        ck = np.arange(bbox[4], bbox[5], 1)
        centers = []

        for ibody in np.arange(bnum):
            centi = rng.choice(ci)
            centj = rng.choice(cj)
            centk = rng.choice(ck)
            ctest = np.array([centi, centj, centk])
            if ibody == 0:
                centers = ctest
            else:
                print(np.shape(centers))
                for itest in np.arange(np.shape(centers)[1]):
                    test = norm(ctest-centers[itest-1])
                    if test >= mdist:
                        template[centi, centj, centk] = val
                    else:
                        print('distribute_bodies: too near!')
    else:
        print('distribute_bodies: method' +
              method.lower()+'not implemented! Exit.')

    return template


def set_mesh(d=None, center=False):
    """
    set_mesh.
    
    Parameters
    ----------
    d : object
        Parameter d.
    center : object
        Parameter center.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    Define cell geometry.
    
    VR Jan 2024
    """
    ncell = np.shape(d)[0]
    xn = np.append(0.0, np.cumsum(d))
    xc = 0.5 * (xn[0:ncell] + xn[1:ncell+1])

    if center:
        c = 0.5*(xn[ncell] - xn[0])
        xn = xn - c

    return xn, xc


def mask_mesh(x=None, y=None, z=None, mod=None,
              mask=None,
              ref=[0., 0., 0.],
              method='index'):
    """
    mask_mesh.
    
    Parameters
    ----------
    x : object
        Parameter x.
    y : object
        Parameter y.
    z : object
        Parameter z.
    mod : object
        Parameter mod.
    mask : object
        Parameter mask.
    ref : object
        Parameter ref.
    method : object
        Parameter method.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    mask model-like parameters and mesh
    
    VR Jan 2024
    """
    msh = np.shape(mod)
    mod_out = mod.copy()
    x_out = x.copy()
    y_out = y.copy()
    z_out = z.copy()

    if ('ind' in method.lower()) or ('ijk' in method.lower()):

        ijk = mask

        x_out = x[ijk[0]:msh[0]-ijk[1]]
        y_out = y[ijk[2]:msh[1]-ijk[3]]
        z_out = z[ijk[4]:msh[2]-ijk[5]]

        mod_out = mod_out[
            ijk[0]:msh[0]-ijk[1],
            ijk[2]:msh[1]-ijk[3],
            ijk[4]:msh[2]-ijk[5]]

    elif 'dis' in method.lower():

        x = x + ref[0]
        y = y + ref[1]
        z = z + ref[2]

        xc = 0.5 * (x[0:msh[0]] + x[1:msh[0]+1])
        yc = 0.5 * (y[0:msh[1]] + y[1:msh[1]+1])
        zc = 0.5 * (z[0:msh[2]] + z[1:msh[2]+1])

        ix = []
        for ii in np.arange(len(xc)):
            if np.logical_and(xc[ii] >= mask[0], xc[ii] <= mask[1]):
                ix.append(ii)
        aixt = tuple(np.array(ix).T)

        iy = []
        for ii in np.arange(len(yc)):
            if np.logical_and(yc[ii] >= mask[2], yc[ii] <= mask[3]):
                iy.append(ii)
        aiyt = tuple(np.array(iy).T)

        iz = []
        for ii in np.arange(len(zc)):
            if np.logical_and(zc[ii] >= mask[2], zc[ii] <= mask[3]):
                iz.append(ii)
        aizt = tuple(np.array(iz).T)

        x_out = x_out[ix.append(ix[-1]+1)]
        y_out = y_out[iy.append(iy[-1]+1)]
        z_out = z_out[iz.append(iz[-1]+1)]
        # np.append(ix,ix[-1]+1)
        print('x ', x_out)
        print('y ', y_out)
        print('x ', z_out)

        mod_out = mod_out[aixt, :, :]
        mod_out = mod_out[:, aiyt, :]
        mod_out = mod_out[:, :, aizt]
        print(np.shape(mod_out))
        print(np.shape(ix), np.shape(iy), np.shape(iz))

        return x_out, y_out, z_out, mod_out


def generate_alphas(dz, beg_lin=[0., 0.1, 0.1], end_lin=[999., 0.9, 0.9]):
    """
    generate_alphas.
    
    Parameters
    ----------
    dz : object
        Parameter dz.
    beg_lin : object
        Parameter beg_lin.
    end_lin : object
        Parameter end_lin.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    Generates linspace depth-dependent horizontal alphas
    
    Parameters ---------- dz: array     verical cell sizes. beg_lin, end_lin: tuple
    depth, val_x, val_y
    
    Returns ------- a_x, a_y : np.arrays     Linspace depth-dependent horizontal
    alphas
    
    
    
    VR Aug 2024
    """

    a_x = np.nan * np.ones_like(dz)
    a_y = np.nan * np.ones_like(dz)

    _, depth = set_mesh(d=dz)

    start_z = beg_lin[0]
    end_z = end_lin[0]

    i0 = (depth >= start_z).argmax()
    i1 = (depth >= end_z).argmax()
    idif = np.abs(i1-i0)

    a_x[:i0] = beg_lin[1]
    a_x[i0:i1] = np.linspace(beg_lin[1], end_lin[1], idif)
    a_x[i1:] = end_lin[1]

    a_y[:i0] = beg_lin[2]
    a_y[i0:i1] = np.linspace(beg_lin[2], end_lin[2], idif)
    a_y[i1:] = end_lin[2]

    return a_x, a_y


# =============================================================================
#  DCT model compression
# =============================================================================

def _dct_sort_by_wavenumber(shape):
    """
    _dct_sort_by_wavenumber.

    Parameters
    ----------
    shape : tuple of int
        (nx, ny, nz) shape of the DCT coefficient array.

    Returns
    -------
    sorted_flat : ndarray, int
        Flat (ravelled) indices sorted from lowest to highest L2 wavenumber
        k = sqrt(kx^2 + ky^2 + kz^2).
    wavenumbers : ndarray, float
        Corresponding wavenumber values (same order as sorted_flat).

    Notes
    -----
    Internal helper used by model_to_dct and dct_spectrum.
    Builds a meshgrid of DCT mode indices and sorts by the radial wavenumber.
    The DC component (index [0,0,0]) is always first.

    VR Apr 2026
    """
    nx, ny, nz = shape
    kx = np.arange(nx, dtype=float)
    ky = np.arange(ny, dtype=float)
    kz = np.arange(nz, dtype=float)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K = np.sqrt(KX**2 + KY**2 + KZ**2)
    sorted_flat = np.argsort(K.ravel(), kind='stable')
    wavenumbers = K.ravel()[sorted_flat]
    return sorted_flat, wavenumbers


def model_to_dct(mval=None, n_keep=None, frac_keep=None, kmax=None, out=True):
    """
    model_to_dct.

    Parameters
    ----------
    mval : ndarray, float, shape (nx, ny, nz)
        Log-resistivity model (or any real-valued 3-D field on the ModEM mesh).
    n_keep : int, optional
        Number of lowest-wavenumber DCT coefficients to retain.
    frac_keep : float, optional
        Fraction of total coefficients to retain (0 < frac_keep <= 1).
        Converted to n_keep = round(frac_keep * nx*ny*nz).
    kmax : float, optional
        Maximum L2 wavenumber to retain.  All coefficients with
        sqrt(kx^2 + ky^2 + kz^2) <= kmax are kept.
    out : bool, optional
        Print compression summary. Default True.

    Returns
    -------
    coeff : ndarray, float
        Retained DCT coefficients, 1-D, length n_keep (or nx*ny*nz if
        no truncation is requested).
    shape : tuple of int
        (nx, ny, nz) — required for reconstruction.
    keep_mask : ndarray, bool, shape (nx, ny, nz), or None
        True where a coefficient is retained.  None when no truncation is
        applied (all coefficients kept).

    Notes
    -----
    Uses the 3-D DCT-II with orthonormal normalisation (norm='ortho'),
    which is unitary and assumes zero-flux boundary conditions at cell edges.
    This matches the natural behaviour of ModEM models, which taper to a
    background resistivity at the mesh boundary.

    Exactly one of n_keep, frac_keep, or kmax should be supplied to activate
    truncation.  Omitting all three returns the full untruncated coefficient
    vector.

    The DC coefficient (index [0,0,0]) is always retained regardless of kmax.

    VR Apr 2026
    """
    mval = np.asarray(mval, dtype=float)
    shape = mval.shape
    n_total = int(np.prod(shape))

    C = dctn(mval, norm='ortho')

    if frac_keep is not None and n_keep is None and kmax is None:
        n_keep = max(1, int(round(frac_keep * n_total)))

    if n_keep is not None:
        n_keep = min(n_keep, n_total)
        sorted_flat, _ = _dct_sort_by_wavenumber(shape)
        keep_flat = sorted_flat[:n_keep]
        keep_mask = np.zeros(n_total, dtype=bool)
        keep_mask[keep_flat] = True
        keep_mask = keep_mask.reshape(shape)

    elif kmax is not None:
        sorted_flat, wn_sorted = _dct_sort_by_wavenumber(shape)
        keep_flat = sorted_flat[wn_sorted <= kmax]
        keep_flat = np.union1d(keep_flat, [0])   # always keep DC
        keep_mask = np.zeros(n_total, dtype=bool)
        keep_mask[keep_flat] = True
        keep_mask = keep_mask.reshape(shape)

    else:
        keep_mask = None

    coeff = C[keep_mask] if keep_mask is not None else C.ravel()

    if out:
        n_kept = coeff.size
        ratio = n_total / max(n_kept, 1)
        print(
            'model_to_dct: %d / %d coefficients retained  (ratio %.1f:1)'
            % (n_kept, n_total, ratio)
        )

    return coeff, shape, keep_mask


def dct_to_model(coeff=None, shape=None, keep_mask=None, out=True):
    """
    dct_to_model.

    Parameters
    ----------
    coeff : ndarray, float
        1-D array of DCT coefficients as returned by model_to_dct.
    shape : tuple of int
        (nx, ny, nz) of the original model.
    keep_mask : ndarray, bool, shape (nx, ny, nz), or None
        Boolean mask identifying retained coefficients.  Pass None when
        coeff spans all cells (no truncation).
    out : bool, optional
        Print reconstruction summary. Default True.

    Returns
    -------
    mval_rec : ndarray, float, shape (nx, ny, nz)
        Reconstructed model in the same units as the input to model_to_dct.

    Notes
    -----
    Inverse of model_to_dct.  Places coeff back into the full coefficient
    array (zeroing missing coefficients) and applies the 3-D inverse DCT-II.

    VR Apr 2026
    """
    C_full = np.zeros(shape, dtype=float)
    if keep_mask is not None:
        C_full[keep_mask] = coeff
    else:
        C_full = coeff.reshape(shape)

    mval_rec = idctn(C_full, norm='ortho')

    if out:
        print('dct_to_model: model reconstructed, shape %s' % str(shape))

    return mval_rec


def dct_reconstruction_error(mval=None, mval_rec=None, norm='rms'):
    """
    dct_reconstruction_error.

    Parameters
    ----------
    mval : ndarray, float
        Original model, shape (nx, ny, nz).
    mval_rec : ndarray, float
        Reconstructed model, same shape.
    norm : str, optional
        Error metric.
        'rms'     — root-mean-square of the difference (default).
        'max'     — L-inf (maximum absolute difference).
        'rel_rms' — RMS normalised by the RMS of the original.

    Returns
    -------
    error : float
        Scalar error measure.

    Notes
    -----
    Convenience wrapper around standard error norms.  Typically called after
    dct_to_model to assess how much information was lost by truncation.

    VR Apr 2026
    """
    mval = np.asarray(mval, dtype=float)
    mval_rec = np.asarray(mval_rec, dtype=float)
    diff = mval_rec - mval

    if norm == 'rms':
        return float(np.sqrt(np.mean(diff**2)))
    elif norm == 'max':
        return float(np.max(np.abs(diff)))
    elif norm == 'rel_rms':
        rms_orig = np.sqrt(np.mean(mval**2))
        if rms_orig == 0.0:
            return 0.0
        return float(np.sqrt(np.mean(diff**2)) / rms_orig)
    else:
        raise ValueError(
            "dct_reconstruction_error: norm must be 'rms', 'max', or 'rel_rms', "
            "got: " + str(norm)
        )


def dct_compress(mval=None, n_keep=None, frac_keep=None, kmax=None, out=True):
    """
    dct_compress.

    Parameters
    ----------
    mval : ndarray, float, shape (nx, ny, nz)
        Log-resistivity model.
    n_keep : int, optional
        Number of coefficients to keep (see model_to_dct).
    frac_keep : float, optional
        Fraction of coefficients to keep (see model_to_dct).
    kmax : float, optional
        Maximum wavenumber to keep (see model_to_dct).
    out : bool, optional
        Print compression statistics. Default True.

    Returns
    -------
    mval_rec : ndarray, float, shape (nx, ny, nz)
        Reconstructed model.
    coeff : ndarray, float
        Retained DCT coefficients.
    keep_mask : ndarray, bool or None
        Mask of retained coefficients.

    Notes
    -----
    Convenience wrapper: forward DCT + truncation + inverse DCT in one call.
    Useful for quick quality checks at a given compression ratio before
    committing to a parameterisation.

    VR Apr 2026
    """
    coeff, shape, keep_mask = model_to_dct(
        mval, n_keep=n_keep, frac_keep=frac_keep, kmax=kmax, out=False
    )
    mval_rec = dct_to_model(coeff, shape, keep_mask, out=False)

    if out:
        n_total = int(np.prod(shape))
        n_kept = coeff.size
        rms_err = dct_reconstruction_error(mval, mval_rec, norm='rms')
        rel_err = dct_reconstruction_error(mval, mval_rec, norm='rel_rms')
        ratio = n_total / max(n_kept, 1)
        print(
            'dct_compress: %d / %d coefficients  (ratio %.1f:1)'
            '  RMS err = %.4g,  rel. RMS = %.4g'
            % (n_kept, n_total, ratio, rms_err, rel_err)
        )

    return mval_rec, coeff, keep_mask


def dct_spectrum(mval=None, n_bins=50, out=True):
    """
    dct_spectrum.

    Parameters
    ----------
    mval : ndarray, float, shape (nx, ny, nz)
        Log-resistivity model.
    n_bins : int, optional
        Number of wavenumber bins. Default 50.
    out : bool, optional
        Print the spectrum table. Default True.

    Returns
    -------
    bin_centers : ndarray, float, shape (n_bins,)
        Centre wavenumber of each bin.
    power : ndarray, float, shape (n_bins,)
        Mean squared DCT coefficient within each bin.
    cum_power_frac : ndarray, float, shape (n_bins,)
        Cumulative fraction of total power up to and including each bin.

    Notes
    -----
    Bins the squared DCT-II coefficients by L2 wavenumber
    k = sqrt(kx^2 + ky^2 + kz^2) into n_bins equally spaced annuli.
    Useful for diagnosing spatial frequency content and for choosing a
    sensible truncation wavenumber kmax for model_to_dct.

    VR Apr 2026
    """
    mval = np.asarray(mval, dtype=float)
    shape = mval.shape
    C = dctn(mval, norm='ortho')

    nx, ny, nz = shape
    kx = np.arange(nx, dtype=float)
    ky = np.arange(ny, dtype=float)
    kz = np.arange(nz, dtype=float)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K = np.sqrt(KX**2 + KY**2 + KZ**2).ravel()
    power_flat = C.ravel()**2

    k_max_all = K.max()
    edges = np.linspace(0.0, k_max_all, n_bins + 1)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    power = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (K >= edges[i]) & (K < edges[i + 1])
        if mask.any():
            power[i] = np.mean(power_flat[mask])

    total = power.sum()
    cum_power_frac = np.cumsum(power) / (total if total > 0.0 else 1.0)

    if out:
        print('\n  DCT power spectrum  (n_bins=%d)' % n_bins)
        print('  %10s  %14s  %16s' % ('k_center', 'mean_power', 'cum_power_frac'))
        for i in range(n_bins):
            print('  %10.2f  %14.6g  %16.4f'
                  % (bin_centers[i], power[i], cum_power_frac[i]))

    return bin_centers, power, cum_power_frac


def dct_truncation_analysis(mval=None, n_levels=20, out=True):
    """
    dct_truncation_analysis.

    Parameters
    ----------
    mval : ndarray, float, shape (nx, ny, nz)
        Log-resistivity model.
    n_levels : int, optional
        Number of compression levels to test. Default 20.
    out : bool, optional
        Print summary table. Default True.

    Returns
    -------
    fracs : ndarray, float, shape (n_levels,)
        Fraction of coefficients retained at each level.
    rms_errors : ndarray, float, shape (n_levels,)
        RMS reconstruction error at each level.
    rel_errors : ndarray, float, shape (n_levels,)
        Relative RMS reconstruction error at each level.

    Notes
    -----
    Sweeps over n_levels values of frac_keep (logarithmically spaced
    from 0.001 to 1.0) and reports reconstruction accuracy at each level.
    Use this to select a truncation that gives acceptable accuracy at the
    desired compression ratio before passing the parameters to dct_compress
    or model_to_dct.

    VR Apr 2026
    """
    fracs = np.clip(np.logspace(-3, 0, n_levels), 1e-6, 1.0)
    rms_errors = np.zeros(n_levels)
    rel_errors = np.zeros(n_levels)
    n_total = int(np.prod(mval.shape))

    if out:
        print('\n  DCT truncation analysis')
        print('  %12s  %10s  %14s  %14s'
              % ('frac_keep', 'n_kept', 'rms_error', 'rel_rms_error'))

    for i, frac in enumerate(fracs):
        coeff, shape, keep_mask = model_to_dct(mval, frac_keep=float(frac), out=False)
        mval_rec = dct_to_model(coeff, shape, keep_mask, out=False)
        rms_errors[i] = dct_reconstruction_error(mval, mval_rec, norm='rms')
        rel_errors[i] = dct_reconstruction_error(mval, mval_rec, norm='rel_rms')
        if out:
            print('  %12.4g  %10d  %14.6g  %14.6g'
                  % (frac, coeff.size, rms_errors[i], rel_errors[i]))

    return fracs, rms_errors, rel_errors


def model_to_dct_separable(mval=None,
                           nx_keep=None, ny_keep=None, nz_keep=None,
                           out=True):
    """
    model_to_dct_separable.

    Parameters
    ----------
    mval : ndarray, float, shape (nx, ny, nz)
        Log-resistivity model.
    nx_keep : int, optional
        Number of coefficients to retain along x. Default: all (nx).
    ny_keep : int, optional
        Number of coefficients to retain along y. Default: all (ny).
    nz_keep : int, optional
        Number of coefficients to retain along z. Default: all (nz).
    out : bool, optional
        Print compression summary. Default True.

    Returns
    -------
    coeff_block : ndarray, float, shape (nx_keep, ny_keep, nz_keep)
        Truncated coefficient sub-volume (low-wavenumber corner).
    shape_full : tuple of int
        Full model shape (nx, ny, nz).
    shape_keep : tuple of int
        (nx_keep, ny_keep, nz_keep).

    Notes
    -----
    Separable (per-axis) alternative to model_to_dct.  Retains a rectangular
    box in wavenumber space rather than a radial sphere, which allows
    anisotropic truncation — e.g. keeping more horizontal resolution than
    vertical in a layered model.  Use dct_separable_to_model to reconstruct.

    VR Apr 2026
    """
    mval = np.asarray(mval, dtype=float)
    nx, ny, nz = mval.shape
    nx_keep = nx if nx_keep is None else min(nx_keep, nx)
    ny_keep = ny if ny_keep is None else min(ny_keep, ny)
    nz_keep = nz if nz_keep is None else min(nz_keep, nz)

    C = dctn(mval, norm='ortho')
    coeff_block = C[:nx_keep, :ny_keep, :nz_keep].copy()

    if out:
        n_total = nx * ny * nz
        n_kept = nx_keep * ny_keep * nz_keep
        print(
            'model_to_dct_separable: %d / %d coefficients retained  '
            '(ratio %.1f:1)  keep shape %s'
            % (n_kept, n_total, n_total / max(n_kept, 1),
               str((nx_keep, ny_keep, nz_keep)))
        )

    return coeff_block, (nx, ny, nz), (nx_keep, ny_keep, nz_keep)


def dct_separable_to_model(coeff_block=None, shape_full=None,
                           shape_keep=None, out=True):
    """
    dct_separable_to_model.

    Parameters
    ----------
    coeff_block : ndarray, float
        Truncated DCT coefficients, shape (nx_keep, ny_keep, nz_keep),
        as returned by model_to_dct_separable.
    shape_full : tuple of int
        Target shape (nx, ny, nz) of the reconstructed model.
    shape_keep : tuple of int, optional
        Shape of coeff_block.  Inferred from coeff_block.shape if None.
    out : bool, optional
        Print reconstruction summary. Default True.

    Returns
    -------
    mval_rec : ndarray, float, shape (nx, ny, nz)
        Reconstructed model.

    Notes
    -----
    Inverse of model_to_dct_separable.  Pads the coefficient block with
    zeros to shape_full and applies the 3-D inverse DCT-II.

    VR Apr 2026
    """
    if shape_keep is None:
        shape_keep = coeff_block.shape
    nx_k, ny_k, nz_k = shape_keep

    C_full = np.zeros(shape_full, dtype=float)
    C_full[:nx_k, :ny_k, :nz_k] = coeff_block

    mval_rec = idctn(C_full, norm='ortho')

    if out:
        print('dct_separable_to_model: model reconstructed, shape %s'
              % str(shape_full))

    return mval_rec


# =============================================================================
#  Wavelet model compression
# =============================================================================

def _check_pywt():
    """Raise ImportError with an install hint if pywt is not available."""
    if pywt is None:
        raise ImportError(
            'PyWavelets is required for wavelet compression. '
            'Install with:  pip install PyWavelets'
        )


def model_to_wavelet(mval=None, wavelet='db4', level=None,
                     n_keep=None, frac_keep=None, thresh=None,
                     out=True):
    """
    model_to_wavelet.

    Parameters
    ----------
    mval : ndarray, float, shape (nx, ny, nz)
        Log-resistivity model (or any real-valued 3-D field on the ModEM mesh).
    wavelet : str, optional
        PyWavelets wavelet name.  Good defaults for smooth geophysical models:
        'db4' (Daubechies-4), 'sym4' (Symlet-4), 'coif2' (Coiflet-2).
        Default 'db4'.
    level : int or None, optional
        Decomposition depth.  None lets PyWavelets choose the maximum level
        for the array size and wavelet filter length.
    n_keep : int, optional
        Retain the n_keep largest-magnitude wavelet coefficients (hard
        threshold by count).
    frac_keep : float, optional
        Fraction of total coefficients to retain (0 < frac_keep <= 1).
        Converted to n_keep = round(frac_keep * n_total).
    thresh : float, optional
        Hard amplitude threshold: zero all coefficients whose absolute value
        is below thresh.
    out : bool, optional
        Print compression summary. Default True.

    Returns
    -------
    coeffs : list
        PyWavelets coefficient list as returned by pywt.dwtn (one dict per
        level plus the approximation array).  Thresholded in-place when
        truncation is requested.
    shapes : list of dict
        Original subband shapes before thresholding, needed for reconstruction.
    n_kept : int
        Number of nonzero coefficients after thresholding.
    n_total : int
        Total number of coefficients (= nx * ny * nz).

    Notes
    -----
    The 3-D stationary discrete wavelet transform (DWT) via pywt.dwtn /
    pywt.idwtn is used.  Boundary extension mode is 'periodization' (fewest
    extra coefficients) — change mode= if edge artefacts are visible.

    Spatial localisation is the key advantage over DCT: a wavelet coefficient
    encodes a frequency *at a specific location*, so conductive anomalies in
    one corner of the model are represented efficiently regardless of what
    the rest of the model looks like.

    Requires PyWavelets (pip install PyWavelets).

    VR Apr 2026
    """
    _check_pywt()
    mval = np.asarray(mval, dtype=float)
    n_total = mval.size

    coeffs = pywt.dwtn(mval, wavelet=wavelet, mode='periodization')

    # Flatten all subbands into a single vector for thresholding
    keys = list(coeffs.keys())
    shapes = {k: coeffs[k].shape for k in keys}
    flat = np.concatenate([coeffs[k].ravel() for k in keys])

    if frac_keep is not None and n_keep is None and thresh is None:
        n_keep = max(1, int(round(frac_keep * flat.size)))

    if n_keep is not None:
        n_keep = min(n_keep, flat.size)
        cutoff = np.sort(np.abs(flat))[-n_keep]
        flat[np.abs(flat) < cutoff] = 0.0
    elif thresh is not None:
        flat[np.abs(flat) < thresh] = 0.0

    # Write thresholded values back into coefficient dict
    pos = 0
    for k in keys:
        size = coeffs[k].size
        coeffs[k] = flat[pos:pos + size].reshape(shapes[k])
        pos += size

    n_kept = int(np.count_nonzero(flat))

    if out:
        ratio = flat.size / max(n_kept, 1)
        print(
            'model_to_wavelet: wavelet=%s  level=%s  '
            '%d / %d coefficients nonzero  (ratio %.1f:1)'
            % (wavelet, str(level), n_kept, flat.size, ratio)
        )

    return coeffs, shapes, n_kept, n_total


def wavelet_to_model(coeffs=None, wavelet='db4', shape=None, out=True):
    """
    wavelet_to_model.

    Parameters
    ----------
    coeffs : list or dict
        PyWavelets coefficient structure as returned by model_to_wavelet.
    wavelet : str, optional
        Must match the wavelet used in model_to_wavelet. Default 'db4'.
    shape : tuple of int, optional
        Expected output shape (nx, ny, nz).  Used only for a consistency
        check; the shape is determined by the coefficient arrays.
    out : bool, optional
        Print reconstruction summary. Default True.

    Returns
    -------
    mval_rec : ndarray, float, shape (nx, ny, nz)
        Reconstructed model, cropped to match the original shape if the
        wavelet boundary extension added padding.

    Notes
    -----
    Inverse of model_to_wavelet.  Applies pywt.idwtn and trims any
    boundary-extension padding introduced during the forward transform.

    VR Apr 2026
    """
    _check_pywt()
    mval_rec = pywt.idwtn(coeffs, wavelet=wavelet, mode='periodization')

    if shape is not None:
        mval_rec = mval_rec[tuple(slice(0, s) for s in shape)]

    if out:
        print('wavelet_to_model: model reconstructed, shape %s'
              % str(mval_rec.shape))

    return mval_rec


def wavelet_compress(mval=None, wavelet='db4', level=None,
                     n_keep=None, frac_keep=None, thresh=None, out=True):
    """
    wavelet_compress.

    Parameters
    ----------
    mval : ndarray, float, shape (nx, ny, nz)
        Log-resistivity model.
    wavelet : str, optional
        Wavelet name. Default 'db4'.
    level : int or None, optional
        Decomposition depth. Default None (maximum).
    n_keep : int, optional
        Number of largest-magnitude coefficients to keep.
    frac_keep : float, optional
        Fraction of coefficients to keep.
    thresh : float, optional
        Hard amplitude threshold.
    out : bool, optional
        Print compression statistics. Default True.

    Returns
    -------
    mval_rec : ndarray, float, shape (nx, ny, nz)
        Reconstructed model.
    coeffs : dict
        Thresholded wavelet coefficient dict.
    n_kept : int
        Number of nonzero coefficients retained.

    Notes
    -----
    Convenience one-call wrapper: forward wavelet transform + thresholding +
    inverse transform + printed statistics.

    VR Apr 2026
    """
    coeffs, shapes, n_kept, n_total = model_to_wavelet(
        mval, wavelet=wavelet, level=level,
        n_keep=n_keep, frac_keep=frac_keep, thresh=thresh, out=False
    )
    mval_rec = wavelet_to_model(coeffs, wavelet=wavelet,
                                shape=mval.shape, out=False)

    if out:
        rms_err = dct_reconstruction_error(mval, mval_rec, norm='rms')
        rel_err = dct_reconstruction_error(mval, mval_rec, norm='rel_rms')
        ratio = n_total / max(n_kept, 1)
        print(
            'wavelet_compress: wavelet=%s  %d / %d coefficients  '
            '(ratio %.1f:1)  RMS err = %.4g  rel. RMS = %.4g'
            % (wavelet, n_kept, n_total, ratio, rms_err, rel_err)
        )

    return mval_rec, coeffs, n_kept


def wavelet_truncation_analysis(mval=None, wavelet='db4', n_levels=20,
                                out=True):
    """
    wavelet_truncation_analysis.

    Parameters
    ----------
    mval : ndarray, float, shape (nx, ny, nz)
        Log-resistivity model.
    wavelet : str, optional
        Wavelet name. Default 'db4'.
    n_levels : int, optional
        Number of compression levels to test. Default 20.
    out : bool, optional
        Print summary table. Default True.

    Returns
    -------
    fracs : ndarray, float, shape (n_levels,)
        Fraction of coefficients retained at each level.
    rms_errors : ndarray, float, shape (n_levels,)
        RMS reconstruction error.
    rel_errors : ndarray, float, shape (n_levels,)
        Relative RMS reconstruction error.

    Notes
    -----
    Mirrors dct_truncation_analysis for the wavelet basis.
    Sweeps frac_keep logarithmically from 0.001 to 1.0.

    VR Apr 2026
    """
    _check_pywt()
    fracs = np.clip(np.logspace(-3, 0, n_levels), 1e-6, 1.0)
    rms_errors = np.zeros(n_levels)
    rel_errors = np.zeros(n_levels)

    if out:
        print('\n  Wavelet truncation analysis  (wavelet=%s)' % wavelet)
        print('  %12s  %10s  %14s  %14s'
              % ('frac_keep', 'n_kept', 'rms_error', 'rel_rms_error'))

    for i, frac in enumerate(fracs):
        coeffs, _, n_kept, _ = model_to_wavelet(
            mval, wavelet=wavelet, frac_keep=float(frac), out=False
        )
        mval_rec = wavelet_to_model(coeffs, wavelet=wavelet,
                                    shape=mval.shape, out=False)
        rms_errors[i] = dct_reconstruction_error(mval, mval_rec, norm='rms')
        rel_errors[i] = dct_reconstruction_error(mval, mval_rec, norm='rel_rms')
        if out:
            print('  %12.4g  %10d  %14.6g  %14.6g'
                  % (frac, n_kept, rms_errors[i], rel_errors[i]))

    return fracs, rms_errors, rel_errors


# =============================================================================
#  Legendre-z x DCT-xy separable compression
# =============================================================================

def _legendre_basis_1d(nz, n_leg):
    """
    _legendre_basis_1d.

    Parameters
    ----------
    nz : int
        Number of depth cells.
    n_leg : int
        Number of Legendre basis functions to evaluate (orders 0 .. n_leg-1).

    Returns
    -------
    P : ndarray, float, shape (nz, n_leg)
        Orthonormal Legendre basis matrix evaluated at the nz cell-centre
        positions mapped to [-1, 1].  Columns are L2-normalised.

    Notes
    -----
    The nz cell centres are mapped linearly to the interval [-1, 1].
    Each column P[:, l] contains the l-th Legendre polynomial P_l(z)
    evaluated at those points and divided by its L2 norm, giving an
    orthonormal column basis.

    VR Apr 2026
    """
    z = np.linspace(-1.0, 1.0, nz)
    P = np.zeros((nz, n_leg))
    for l in range(n_leg):
        col = eval_legendre(l, z)
        col_norm = np.sqrt(np.sum(col**2))
        P[:, l] = col / (col_norm if col_norm > 0.0 else 1.0)
    return P


def model_to_legdct(mval=None, n_leg=None, frac_leg=0.5,
                    nx_dct=None, ny_dct=None, frac_dct=0.5,
                    out=True):
    """
    model_to_legdct.

    Parameters
    ----------
    mval : ndarray, float, shape (nx, ny, nz)
        Log-resistivity model.
    n_leg : int, optional
        Number of Legendre basis functions along z (depth).
        Defaults to round(frac_leg * nz).
    frac_leg : float, optional
        Fraction of depth cells to use as Legendre orders when n_leg is None.
        Default 0.5.
    nx_dct : int, optional
        Number of DCT-II coefficients to retain along x.
        Defaults to round(frac_dct * nx).
    ny_dct : int, optional
        Number of DCT-II coefficients to retain along y.
        Defaults to round(frac_dct * ny).
    frac_dct : float, optional
        Fraction of horizontal cells to use as DCT coefficients when nx_dct
        or ny_dct are None. Default 0.5.
    out : bool, optional
        Print compression summary. Default True.

    Returns
    -------
    C : ndarray, float, shape (nx_dct, ny_dct, n_leg)
        Compressed coefficient array: DCT along x and y, Legendre along z.
    shape_full : tuple of int
        Original model shape (nx, ny, nz).
    params : dict
        Dictionary containing n_leg, nx_dct, ny_dct (needed for reconstruction).

    Notes
    -----
    Applies a separable mixed basis:
        - 1-D Legendre polynomial expansion along z (depth axis).
        - 2-D DCT-II along x and y (horizontal axes).

    This respects the physical anisotropy of MT models: the depth dimension
    has strong gradients (resistivity varies by orders of magnitude with
    depth) that Legendre polynomials represent compactly, while horizontal
    structure is quasi-periodic and well suited to the DCT.

    The transform is:
        1. Project mval onto the Legendre basis along z:
               A[ix, iy, l] = sum_k mval[ix, iy, k] * P[k, l]
        2. Apply 2-D DCT-II to A along axes 0 and 1:
               C = dctn(A, axes=(0, 1), norm='ortho')[:nx_dct, :ny_dct, :]

    VR Apr 2026
    """
    mval = np.asarray(mval, dtype=float)
    nx, ny, nz = mval.shape

    if n_leg is None:
        n_leg = max(1, int(round(frac_leg * nz)))
    n_leg = min(n_leg, nz)

    if nx_dct is None:
        nx_dct = max(1, int(round(frac_dct * nx)))
    if ny_dct is None:
        ny_dct = max(1, int(round(frac_dct * ny)))
    nx_dct = min(nx_dct, nx)
    ny_dct = min(ny_dct, ny)

    # Step 1: project onto Legendre basis along z
    P = _legendre_basis_1d(nz, n_leg)           # (nz, n_leg)
    A = np.tensordot(mval, P, axes=([2], [0]))  # (nx, ny, n_leg)

    # Step 2: 2-D DCT along horizontal axes, then truncate
    C_full = dctn(A, axes=(0, 1), norm='ortho')
    C = C_full[:nx_dct, :ny_dct, :].copy()

    params = dict(n_leg=n_leg, nx_dct=nx_dct, ny_dct=ny_dct)

    if out:
        n_total = nx * ny * nz
        n_kept = nx_dct * ny_dct * n_leg
        ratio = n_total / max(n_kept, 1)
        print(
            'model_to_legdct: Legendre orders=%d  DCT x=%d y=%d  '
            '%d / %d coefficients  (ratio %.1f:1)'
            % (n_leg, nx_dct, ny_dct, n_kept, n_total, ratio)
        )

    return C, (nx, ny, nz), params


def legdct_to_model(C=None, shape_full=None, params=None, out=True):
    """
    legdct_to_model.

    Parameters
    ----------
    C : ndarray, float, shape (nx_dct, ny_dct, n_leg)
        Compressed coefficient array as returned by model_to_legdct.
    shape_full : tuple of int
        Original model shape (nx, ny, nz).
    params : dict
        Must contain keys 'n_leg', 'nx_dct', 'ny_dct' (as returned by
        model_to_legdct).
    out : bool, optional
        Print reconstruction summary. Default True.

    Returns
    -------
    mval_rec : ndarray, float, shape (nx, ny, nz)
        Reconstructed model.

    Notes
    -----
    Inverse of model_to_legdct:
        1. Pad C with zeros to (nx, ny, n_leg) and apply inverse 2-D DCT
           along axes 0 and 1 to recover A[ix, iy, l].
        2. Reconstruct depth profiles:
               mval_rec[ix, iy, k] = sum_l A[ix, iy, l] * P[k, l]

    VR Apr 2026
    """
    nx, ny, nz = shape_full
    n_leg = params['n_leg']
    nx_dct = params['nx_dct']
    ny_dct = params['ny_dct']

    # Step 1: inverse 2-D DCT with zero-padding
    C_pad = np.zeros((nx, ny, n_leg), dtype=float)
    C_pad[:nx_dct, :ny_dct, :] = C
    A = idctn(C_pad, axes=(0, 1), norm='ortho')  # (nx, ny, n_leg)

    # Step 2: inverse Legendre projection along z
    P = _legendre_basis_1d(nz, n_leg)                  # (nz, n_leg)
    mval_rec = np.tensordot(A, P.T, axes=([2], [0]))   # (nx, ny, nz)

    if out:
        print('legdct_to_model: model reconstructed, shape %s'
              % str(shape_full))

    return mval_rec


def legdct_compress(mval=None, n_leg=None, frac_leg=0.5,
                    nx_dct=None, ny_dct=None, frac_dct=0.5, out=True):
    """
    legdct_compress.

    Parameters
    ----------
    mval : ndarray, float, shape (nx, ny, nz)
        Log-resistivity model.
    n_leg : int, optional
        Legendre orders along z. Default round(frac_leg * nz).
    frac_leg : float, optional
        Fraction of depth cells as Legendre orders. Default 0.5.
    nx_dct : int, optional
        DCT coefficients along x. Default round(frac_dct * nx).
    ny_dct : int, optional
        DCT coefficients along y. Default round(frac_dct * ny).
    frac_dct : float, optional
        Fraction of horizontal cells as DCT coefficients. Default 0.5.
    out : bool, optional
        Print statistics. Default True.

    Returns
    -------
    mval_rec : ndarray, float, shape (nx, ny, nz)
        Reconstructed model.
    C : ndarray, float
        Compressed coefficient array.
    params : dict
        Compression parameters (n_leg, nx_dct, ny_dct).

    Notes
    -----
    One-call wrapper: forward Legendre-z × DCT-xy + inverse + statistics.

    VR Apr 2026
    """
    C, shape_full, params = model_to_legdct(
        mval, n_leg=n_leg, frac_leg=frac_leg,
        nx_dct=nx_dct, ny_dct=ny_dct, frac_dct=frac_dct, out=False
    )
    mval_rec = legdct_to_model(C, shape_full, params, out=False)

    if out:
        nx, ny, nz = shape_full
        n_total = nx * ny * nz
        n_kept = params['nx_dct'] * params['ny_dct'] * params['n_leg']
        rms_err = dct_reconstruction_error(mval, mval_rec, norm='rms')
        rel_err = dct_reconstruction_error(mval, mval_rec, norm='rel_rms')
        ratio = n_total / max(n_kept, 1)
        print(
            'legdct_compress: Legendre orders=%d  DCT x=%d y=%d  '
            '%d / %d coefficients  (ratio %.1f:1)  '
            'RMS err = %.4g  rel. RMS = %.4g'
            % (params['n_leg'], params['nx_dct'], params['ny_dct'],
               n_kept, n_total, ratio, rms_err, rel_err)
        )

    return mval_rec, C, params


def legdct_truncation_analysis(mval=None, n_levels=20, out=True):
    """
    legdct_truncation_analysis.

    Parameters
    ----------
    mval : ndarray, float, shape (nx, ny, nz)
        Log-resistivity model.
    n_levels : int, optional
        Number of compression levels to test. Default 20.
    out : bool, optional
        Print summary table. Default True.

    Returns
    -------
    fracs : ndarray, float, shape (n_levels,)
        Fraction of coefficients retained at each level.
        Both frac_leg and frac_dct are set to sqrt(frac) so that the
        overall compression ratio scales uniformly.
    rms_errors : ndarray, float, shape (n_levels,)
        RMS reconstruction error.
    rel_errors : ndarray, float, shape (n_levels,)
        Relative RMS reconstruction error.

    Notes
    -----
    Mirrors dct_truncation_analysis for the Legendre-z x DCT-xy basis.
    The two fractions (frac_leg and frac_dct) are coupled via
    frac_leg = frac_dct = sqrt(frac_keep) so that varying a single
    parameter sweeps the overall compression ratio.

    VR Apr 2026
    """
    fracs = np.clip(np.logspace(-3, 0, n_levels), 1e-6, 1.0)
    rms_errors = np.zeros(n_levels)
    rel_errors = np.zeros(n_levels)

    if out:
        print('\n  Legendre-z x DCT-xy truncation analysis')
        print('  %12s  %10s  %14s  %14s'
              % ('frac_keep', 'n_kept', 'rms_error', 'rel_rms_error'))

    nx, ny, nz = mval.shape
    for i, frac in enumerate(fracs):
        f = float(np.sqrt(frac))
        C, shape_full, params = model_to_legdct(
            mval, frac_leg=f, frac_dct=f, out=False
        )
        mval_rec = legdct_to_model(C, shape_full, params, out=False)
        rms_errors[i] = dct_reconstruction_error(mval, mval_rec, norm='rms')
        rel_errors[i] = dct_reconstruction_error(mval, mval_rec, norm='rel_rms')
        n_kept = params['nx_dct'] * params['ny_dct'] * params['n_leg']
        if out:
            print('  %12.4g  %10d  %14.6g  %14.6g'
                  % (frac, n_kept, rms_errors[i], rel_errors[i]))

    return fracs, rms_errors, rel_errors


# =============================================================================
#  B-spline-z x DCT-xy separable compression
# =============================================================================

def _bspline_basis_1d(dz, n_basis, k=3, knot_style='quantile'):
    """
    _bspline_basis_1d.

    Parameters
    ----------
    dz : ndarray, float, shape (nz,)
        Cell thicknesses along the depth axis (metres).  Used to compute
        cell-centre positions and, for knot_style='quantile', to place knots
        at depth quantiles of the actual cell centres.
    n_basis : int
        Number of B-spline basis functions.  Must satisfy n_basis >= k + 1.
        The number of free interior knots is n_basis - k - 1; setting
        n_basis = k + 1 gives a single polynomial spanning the whole depth
        range (no interior knots).
    k : int, optional
        Spline degree.  Default 3 (cubic).
    knot_style : str, optional
        How to place interior knots:

        'uniform'   Equally spaced in normalised depth [0, 1].  Best for
                    models with uniform cell sizes.
        'quantile'  Knots at depth quantiles of the cell-centre positions.
                    Concentrates knots where the grid is fine (near surface),
                    matching ModEM's typical cell-size progression.  Recommended
                    default for real models.
        'log'       Logarithmically spaced in normalised depth.  Alternative
                    to 'quantile' when the depth axis spans many decades.

    Returns
    -------
    B : ndarray, float, shape (nz, n_basis)
        B-spline collocation (design) matrix.  Column l contains the l-th
        basis function evaluated at each cell-centre.  Not orthonormal;
        the pseudo-inverse is used for the inverse transform.
    Bpinv : ndarray, float, shape (n_basis, nz)
        Moore-Penrose pseudo-inverse of B.  Pre-computed once and stored in
        the params dict so it is not recomputed on every reconstruction call.
    t : ndarray, float
        Full knot vector (clamped, length n_basis + k + 1).
    z_norm : ndarray, float, shape (nz,)
        Normalised cell-centre depths in [0, 1].

    Notes
    -----
    Cell centres are computed from dz as zc = cumsum([0, dz]) midpoints,
    then mapped linearly to [0, 1].  The clamped knot vector has k+1
    coincident knots at 0 and 1 (zero-slope boundary conditions at top and
    bottom of the model), which is the standard choice for geophysical depth
    profiles.

    The B-spline collocation matrix is obtained via
    scipy.interpolate.BSpline.design_matrix, available in scipy >= 1.8.

    VR Apr 2026
    """
    dz = np.asarray(dz, dtype=float)
    nz = len(dz)

    if n_basis < k + 1:
        raise ValueError(
            '_bspline_basis_1d: n_basis=%d must be >= k+1=%d'
            % (n_basis, k + 1)
        )

    # Cell-centre depths, normalised to [0, 1]
    zn = np.r_[0.0, np.cumsum(dz)]
    zc = 0.5 * (zn[:-1] + zn[1:])
    z_norm = zc / zc[-1]

    n_interior = n_basis - k - 1

    if n_interior == 0:
        interior = np.array([])
    elif knot_style == 'uniform':
        interior = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
    elif knot_style == 'quantile':
        interior = np.quantile(z_norm, np.linspace(0.0, 1.0, n_interior + 2)[1:-1])
    elif knot_style == 'log':
        eps = z_norm[z_norm > 0.0].min() * 0.5
        interior = np.exp(
            np.linspace(np.log(eps), 0.0, n_interior + 2)[1:-1]
        )
        interior = np.clip(interior, 0.0, 1.0)
    else:
        raise ValueError(
            "_bspline_basis_1d: knot_style must be 'uniform', 'quantile', "
            "or 'log', got: " + str(knot_style)
        )

    # Clamped knot vector
    t = np.r_[np.repeat(0.0, k + 1), interior, np.repeat(1.0, k + 1)]

    B = BSpline.design_matrix(z_norm, t, k=k).toarray()  # (nz, n_basis)
    Bpinv = np.linalg.pinv(B)                            # (n_basis, nz)

    return B, Bpinv, t, z_norm


def model_to_bspdct(mval=None, dz=None, n_basis=None, frac_basis=0.5,
                    k=3, knot_style='quantile',
                    nx_dct=None, ny_dct=None, frac_dct=0.5,
                    out=True):
    """
    model_to_bspdct.

    Parameters
    ----------
    mval : ndarray, float, shape (nx, ny, nz)
        Log-resistivity model.
    dz : ndarray, float, shape (nz,)
        Cell thicknesses along depth (metres).  Required for knot placement.
    n_basis : int, optional
        Number of B-spline basis functions along z.
        Defaults to round(frac_basis * nz).
    frac_basis : float, optional
        Fraction of depth cells to use as B-spline basis functions when
        n_basis is None.  Default 0.5.
    k : int, optional
        Spline degree.  Default 3 (cubic).
    knot_style : str, optional
        Knot placement strategy: 'uniform', 'quantile' (default), or 'log'.
        See _bspline_basis_1d for details.
    nx_dct : int, optional
        Number of DCT-II coefficients to retain along x.
        Defaults to round(frac_dct * nx).
    ny_dct : int, optional
        Number of DCT-II coefficients to retain along y.
        Defaults to round(frac_dct * ny).
    frac_dct : float, optional
        Fraction of horizontal cells to use as DCT coefficients when
        nx_dct or ny_dct are None.  Default 0.5.
    out : bool, optional
        Print compression summary.  Default True.

    Returns
    -------
    C : ndarray, float, shape (nx_dct, ny_dct, n_basis)
        Compressed coefficient array: DCT along x and y, B-spline along z.
    shape_full : tuple of int
        Original model shape (nx, ny, nz).
    params : dict
        All parameters needed for reconstruction:
        n_basis, k, nx_dct, ny_dct, Bpinv, B, knot_style.

    Notes
    -----
    Separable mixed-basis transform:

        1. Project mval onto B-spline basis along z (depth axis):
               A[ix, iy, l] = sum_k mval[ix, iy, k] * B[k, l]
           where B is the (nz, n_basis) collocation matrix.

        2. Apply 2-D DCT-II to A along axes 0 and 1, then truncate:
               C = dctn(A, axes=(0,1), norm='ortho')[:nx_dct, :ny_dct, :]

    Advantages over Legendre-z x DCT-xy:
    - B-splines are locally supported: a control point at 10 km depth
      has zero influence below ~30 km (depending on knot spacing).
      Legendre polynomials are global.
    - The knot vector can follow the actual cell-size distribution via
      knot_style='quantile', placing more basis functions near the surface
      where resolution is highest and fewer at depth where cells are coarse.
    - Zero-slope (clamped) boundary conditions at top and bottom are
      physically reasonable for the depth axis.

    VR Apr 2026
    """
    mval = np.asarray(mval, dtype=float)
    nx, ny, nz = mval.shape

    if dz is None:
        raise ValueError('model_to_bspdct: dz (cell thicknesses) is required.')
    dz = np.asarray(dz, dtype=float)

    if n_basis is None:
        n_basis = max(k + 1, int(round(frac_basis * nz)))
    n_basis = min(n_basis, nz)

    if nx_dct is None:
        nx_dct = max(1, int(round(frac_dct * nx)))
    if ny_dct is None:
        ny_dct = max(1, int(round(frac_dct * ny)))
    nx_dct = min(nx_dct, nx)
    ny_dct = min(ny_dct, ny)

    B, Bpinv, t, z_norm = _bspline_basis_1d(dz, n_basis, k=k,
                                              knot_style=knot_style)

    # Step 1: project onto B-spline basis along z
    A = np.tensordot(mval, B, axes=([2], [0]))          # (nx, ny, n_basis)

    # Step 2: 2-D DCT along horizontal axes, then truncate
    C_full = dctn(A, axes=(0, 1), norm='ortho')
    C = C_full[:nx_dct, :ny_dct, :].copy()

    params = dict(n_basis=n_basis, k=k, nx_dct=nx_dct, ny_dct=ny_dct,
                  Bpinv=Bpinv, B=B, knot_style=knot_style)

    if out:
        n_total = nx * ny * nz
        n_kept = nx_dct * ny_dct * n_basis
        ratio = n_total / max(n_kept, 1)
        print(
            'model_to_bspdct: B-spline degree=%d  knots=%s  n_basis=%d  '
            'DCT x=%d y=%d  %d / %d coefficients  (ratio %.1f:1)'
            % (k, knot_style, n_basis, nx_dct, ny_dct, n_kept, n_total, ratio)
        )

    return C, (nx, ny, nz), params


def bspdct_to_model(C=None, shape_full=None, params=None, out=True):
    """
    bspdct_to_model.

    Parameters
    ----------
    C : ndarray, float, shape (nx_dct, ny_dct, n_basis)
        Compressed coefficient array as returned by model_to_bspdct.
    shape_full : tuple of int
        Original model shape (nx, ny, nz).
    params : dict
        Must contain keys 'n_basis', 'nx_dct', 'ny_dct', 'Bpinv' (as
        returned by model_to_bspdct).
    out : bool, optional
        Print reconstruction summary.  Default True.

    Returns
    -------
    mval_rec : ndarray, float, shape (nx, ny, nz)
        Reconstructed model.

    Notes
    -----
    Inverse of model_to_bspdct:

        1. Pad C with zeros to (nx, ny, n_basis) and apply inverse 2-D DCT
           along axes 0 and 1 to recover A[ix, iy, l].
        2. Reconstruct depth profiles using the stored pseudo-inverse:
               mval_rec[ix, iy, k] = sum_l A[ix, iy, l] * Bpinv[l, k]

    The pseudo-inverse Bpinv is pre-computed in model_to_bspdct and stored
    in params to avoid recomputing it on every reconstruction call.

    VR Apr 2026
    """
    nx, ny, nz = shape_full
    n_basis = params['n_basis']
    nx_dct = params['nx_dct']
    ny_dct = params['ny_dct']
    Bpinv = params['Bpinv']            # (n_basis, nz)

    # Step 1: inverse 2-D DCT with zero-padding
    C_pad = np.zeros((nx, ny, n_basis), dtype=float)
    C_pad[:nx_dct, :ny_dct, :] = C
    A = idctn(C_pad, axes=(0, 1), norm='ortho')    # (nx, ny, n_basis)

    # Step 2: inverse B-spline projection
    mval_rec = np.tensordot(A, Bpinv, axes=([2], [0]))  # (nx, ny, nz)

    if out:
        print('bspdct_to_model: model reconstructed, shape %s'
              % str(shape_full))

    return mval_rec


def bspdct_compress(mval=None, dz=None, n_basis=None, frac_basis=0.5,
                    k=3, knot_style='quantile',
                    nx_dct=None, ny_dct=None, frac_dct=0.5, out=True):
    """
    bspdct_compress.

    Parameters
    ----------
    mval : ndarray, float, shape (nx, ny, nz)
        Log-resistivity model.
    dz : ndarray, float, shape (nz,)
        Cell thicknesses along depth (metres).
    n_basis : int, optional
        B-spline basis functions along z.  Default round(frac_basis * nz).
    frac_basis : float, optional
        Fraction of depth cells as B-spline basis functions.  Default 0.5.
    k : int, optional
        Spline degree.  Default 3 (cubic).
    knot_style : str, optional
        Knot placement: 'uniform', 'quantile' (default), or 'log'.
    nx_dct : int, optional
        DCT coefficients along x.  Default round(frac_dct * nx).
    ny_dct : int, optional
        DCT coefficients along y.  Default round(frac_dct * ny).
    frac_dct : float, optional
        Fraction of horizontal cells as DCT coefficients.  Default 0.5.
    out : bool, optional
        Print statistics.  Default True.

    Returns
    -------
    mval_rec : ndarray, float, shape (nx, ny, nz)
        Reconstructed model.
    C : ndarray, float
        Compressed coefficient array.
    params : dict
        Compression parameters.

    Notes
    -----
    One-call wrapper: forward B-spline-z × DCT-xy + inverse + statistics.

    VR Apr 2026
    """
    C, shape_full, params = model_to_bspdct(
        mval, dz=dz, n_basis=n_basis, frac_basis=frac_basis,
        k=k, knot_style=knot_style,
        nx_dct=nx_dct, ny_dct=ny_dct, frac_dct=frac_dct, out=False
    )
    mval_rec = bspdct_to_model(C, shape_full, params, out=False)

    if out:
        nx, ny, nz = shape_full
        n_total = nx * ny * nz
        n_kept = params['nx_dct'] * params['ny_dct'] * params['n_basis']
        rms_err = dct_reconstruction_error(mval, mval_rec, norm='rms')
        rel_err = dct_reconstruction_error(mval, mval_rec, norm='rel_rms')
        ratio = n_total / max(n_kept, 1)
        print(
            'bspdct_compress: degree=%d  knots=%s  n_basis=%d  DCT x=%d y=%d  '
            '%d / %d coefficients  (ratio %.1f:1)  '
            'RMS err = %.4g  rel. RMS = %.4g'
            % (params['k'], params['knot_style'], params['n_basis'],
               params['nx_dct'], params['ny_dct'],
               n_kept, n_total, ratio, rms_err, rel_err)
        )

    return mval_rec, C, params


def bspdct_truncation_analysis(mval=None, dz=None, k=3, knot_style='quantile',
                               n_levels=20, out=True):
    """
    bspdct_truncation_analysis.

    Parameters
    ----------
    mval : ndarray, float, shape (nx, ny, nz)
        Log-resistivity model.
    dz : ndarray, float, shape (nz,)
        Cell thicknesses along depth (metres).
    k : int, optional
        Spline degree.  Default 3.
    knot_style : str, optional
        Knot placement strategy.  Default 'quantile'.
    n_levels : int, optional
        Number of compression levels to test.  Default 20.
    out : bool, optional
        Print summary table.  Default True.

    Returns
    -------
    fracs : ndarray, float, shape (n_levels,)
        Overall fraction of coefficients retained at each level.
        frac_basis and frac_dct are both set to sqrt(frac) so that the
        compression ratio scales uniformly with a single parameter.
    rms_errors : ndarray, float, shape (n_levels,)
        RMS reconstruction error.
    rel_errors : ndarray, float, shape (n_levels,)
        Relative RMS reconstruction error.

    Notes
    -----
    Mirrors legdct_truncation_analysis for the B-spline-z × DCT-xy basis.
    The minimum n_basis is clamped to k+1 so the basis is always valid.

    VR Apr 2026
    """
    fracs = np.clip(np.logspace(-3, 0, n_levels), 1e-6, 1.0)
    rms_errors = np.zeros(n_levels)
    rel_errors = np.zeros(n_levels)

    if out:
        print('\n  B-spline-z x DCT-xy truncation analysis'
              '  (degree=%d  knots=%s)' % (k, knot_style))
        print('  %12s  %10s  %14s  %14s'
              % ('frac_keep', 'n_kept', 'rms_error', 'rel_rms_error'))

    nx, ny, nz = mval.shape
    for i, frac in enumerate(fracs):
        f = float(np.sqrt(frac))
        C, shape_full, params = model_to_bspdct(
            mval, dz=dz, frac_basis=f, k=k, knot_style=knot_style,
            frac_dct=f, out=False
        )
        mval_rec = bspdct_to_model(C, shape_full, params, out=False)
        rms_errors[i] = dct_reconstruction_error(mval, mval_rec, norm='rms')
        rel_errors[i] = dct_reconstruction_error(mval, mval_rec, norm='rel_rms')
        n_kept = params['nx_dct'] * params['ny_dct'] * params['n_basis']
        if out:
            print('  %12.4g  %10d  %14.6g  %14.6g'
                  % (frac, n_kept, rms_errors[i], rel_errors[i]))

    return fracs, rms_errors, rel_errors




def ensemble_to_kl(ensemble=None, n_modes=None, frac_modes=None,
                   centre=True, svd_method='auto',
                   n_oversamples=10, n_power_iter=4, random_state=None,
                   out=True):
    """
    ensemble_to_kl.

    Parameters
    ----------
    ensemble : ndarray, float, shape (n_models, nx, ny, nz) or (n_models, n_cells)
        Collection of log-resistivity models forming the prior or posterior
        ensemble.  Each row is one model (flattened or 3-D).
    n_modes : int, optional
        Number of KL eigenmodes to retain.  Defaults to round(frac_modes *
        min(n_models, n_cells)).
    frac_modes : float, optional
        Fraction of available modes to retain when n_modes is None.
        Default 1.0 (all modes).
    centre : bool, optional
        Subtract the ensemble mean before computing the SVD (standard PCA).
        Set False to work with un-centred covariance. Default True.
    svd_method : str, optional
        Which SVD algorithm to use.  Choices:

        'auto'       Select automatically: 'randomized' when n_modes is set
                     and n_modes < 0.5 * min(n_models, n_cells), otherwise
                     'exact'. (default)
        'exact'      Full economy SVD via numpy.linalg.svd.  Exact singular
                     values and vectors.  Cost O(n_models^2 * n_cells).
                     Required when all modes are needed or n_modes is close
                     to min(n_models, n_cells).
        'randomized' Randomized SVD (Halko et al. 2011).  Only computes the
                     leading n_modes modes.  Cost O(n_models * n_cells *
                     n_modes).  Strongly preferred when n_modes << n_models.
                     Uses sklearn if available; falls back to inverse.rsvd
                     (py4mt Algorithms 4.1 + 4.4) when sklearn is absent.
        'truncated'  Truncated SVD via scipy.sparse.linalg.svds (ARPACK).
                     Deterministic iterations; useful for sparse matrices.
                     Cost similar to 'randomized'.

    n_oversamples : int, optional
        Extra random projections used by 'randomized' SVD to improve
        accuracy.  Larger values give more accurate singular vectors at
        slightly higher cost.  Ignored for 'exact' and 'truncated'.
        Default 10.
    n_power_iter : int, optional
        Number of power iterations used by 'randomized' SVD to improve
        accuracy on matrices with slowly decaying singular values.  Set to
        0 for speed, 4–7 for accuracy.  Ignored for 'exact' and 'truncated'.
        Default 4.
    random_state : int or None, optional
        Random seed for reproducibility of 'randomized' SVD.  Default None.
    out : bool, optional
        Print summary including SVD method and variance explained. Default True.

    Returns
    -------
    modes : ndarray, float, shape (n_modes, n_cells)
        KL eigenmodes (right singular vectors of the centred ensemble matrix),
        each of unit L2 norm, sorted by descending explained variance.
    singular_values : ndarray, float, shape (n_modes,)
        Singular values (square root of variance explained by each mode).
        For 'randomized' and 'truncated' the total variance estimate is
        approximate; see Notes.
    mean_model : ndarray, float, shape (n_cells,)
        Ensemble mean (zeros if centre=False).
    shape : tuple of int
        Original spatial shape (nx, ny, nz) or (n_cells,) if input was 2-D.

    Notes
    -----
    The KL (Karhunen-Loève) basis is the theoretically optimal linear basis
    for a given ensemble: it minimises the mean-square reconstruction error
    for any fixed number of retained modes.

    SVD method comparison
    ~~~~~~~~~~~~~~~~~~~~~
    Method          Cost                    Exact?  When to use
    'exact'         O(n_m^2 * n_c)          Yes     n_modes ~ n_models, or
                                                    full variance spectrum needed
    'randomized'    O(n_m * n_c * n_modes)  ~Yes*   n_modes << n_models (default
                    + power iterations              for large ensembles).
                                                    sklearn preferred; falls back
                                                    to inverse.rsvd (py4mt).
    'truncated'     O(n_m * n_c * n_modes)  ~Yes*   Sparse matrices; deterministic
                                                    ARPACK iterations.
    n_m = n_models, n_c = n_cells.
    *Accuracy improves with n_oversamples and n_power_iter.

    For 'randomized' and 'truncated', only the retained singular values are
    computed, so the variance-explained fraction printed by out=True uses
    the Frobenius norm of the full (centred) ensemble matrix as the total
    variance denominator, which is exact regardless of SVD method.

    References
    ----------
    Halko N, Martinsson PG, Tropp JA (2011) Finding structure with randomness:
    Probabilistic algorithms for constructing approximate matrix decompositions.
    SIAM Review 53(2):217–288.  https://doi.org/10.1137/090771806

    VR Apr 2026
    """
    ensemble = np.asarray(ensemble, dtype=float)

    if ensemble.ndim == 4:
        shape = ensemble.shape[1:]
        n_models = ensemble.shape[0]
        E = ensemble.reshape(n_models, -1)
    elif ensemble.ndim == 2:
        shape = (ensemble.shape[1],)
        n_models = ensemble.shape[0]
        E = ensemble.copy()
    else:
        raise ValueError(
            'ensemble_to_kl: ensemble must be 2-D (n_models, n_cells) or '
            '4-D (n_models, nx, ny, nz), got shape %s' % str(ensemble.shape)
        )

    n_cells = E.shape[1]

    if centre:
        mean_model = E.mean(axis=0)
        E = E - mean_model
    else:
        mean_model = np.zeros(n_cells)

    max_modes = min(n_models, n_cells)
    if frac_modes is not None and n_modes is None:
        n_modes = max(1, int(round(frac_modes * max_modes)))
    if n_modes is None:
        n_modes = max_modes
    n_modes = min(n_modes, max_modes)

    # --- resolve 'auto' ---
    if svd_method == 'auto':
        if n_modes < 0.5 * max_modes:
            svd_method = 'randomized'
        else:
            svd_method = 'exact'

    # --- compute SVD ---
    if svd_method == 'exact':
        U, S_all, Vt = np.linalg.svd(E, full_matrices=False)
        modes = Vt[:n_modes, :]
        singular_values = S_all[:n_modes]
        var_total = np.sum(S_all**2)

    elif svd_method == 'randomized':
        try:
            from sklearn.utils.extmath import randomized_svd as _rsvd
            U, singular_values, Vt = _rsvd(
                E, n_components=n_modes,
                n_oversamples=n_oversamples,
                n_iter=n_power_iter,
                random_state=random_state,
            )
            modes = Vt
        except ImportError:
            # sklearn absent: fall back to inverse.rsvd (Halko et al. 2011,
            # Algorithms 4.1 + 4.4), which is already part of py4mt.
            # find_range uses np.random.default_rng() internally, so we seed
            # the global RNG for reproducibility when random_state is set.
            if random_state is not None:
                np.random.seed(int(random_state))
            try:
                import inverse as inv
                U, singular_values, Vt = inv.rsvd(
                    E,
                    rank=n_modes,
                    n_oversamples=n_oversamples,
                    n_subspace_iters=n_power_iter,
                )
                modes = Vt
            except ImportError:
                raise ImportError(
                    'ensemble_to_kl: randomized SVD requires either sklearn '
                    '(pip install scikit-learn) or the py4mt inverse module '
                    'on sys.path.'
                )
            if out:
                print('ensemble_to_kl: sklearn not found, '
                      'using inverse.rsvd (py4mt)')

        var_total = np.linalg.norm(E, 'fro')**2

    if svd_method == 'truncated':
        from scipy.sparse.linalg import svds as _svds
        # svds returns singular values in ascending order
        U, singular_values, Vt = _svds(E, k=n_modes)
        idx = np.argsort(singular_values)[::-1]
        singular_values = singular_values[idx]
        modes = Vt[idx, :]
        var_total = np.linalg.norm(E, 'fro')**2

    var_kept = np.sum(singular_values**2)
    frac_var = var_kept / var_total if var_total > 0.0 else 1.0

    if out:
        exact_flag = '' if svd_method == 'exact' else ' (approx)'
        print(
            'ensemble_to_kl: %d models  method=%s  '
            '%d modes retained / %d available  '
            'variance explained = %.2f%%%s'
            % (n_models, svd_method, n_modes, max_modes,
               100.0 * frac_var, exact_flag)
        )

    return modes, singular_values, mean_model, shape


def model_to_kl(mval=None, modes=None, mean_model=None, out=True):
    """
    model_to_kl.

    Parameters
    ----------
    mval : ndarray, float, shape (nx, ny, nz) or (n_cells,)
        Log-resistivity model to project onto the KL basis.
    modes : ndarray, float, shape (n_modes, n_cells)
        KL eigenmodes as returned by ensemble_to_kl.
    mean_model : ndarray, float, shape (n_cells,), optional
        Ensemble mean.  Subtracted before projection.  Pass None or zeros
        if ensemble_to_kl was called with centre=False.
    out : bool, optional
        Print summary. Default True.

    Returns
    -------
    alpha : ndarray, float, shape (n_modes,)
        KL expansion coefficients (scores): the model projected onto each
        eigenmode.

    Notes
    -----
    Projects the centred model onto each eigenmode by dot product:
        alpha[l] = dot(mval_flat - mean, modes[l])

    Because the modes are orthonormal this is equivalent to a change of
    basis.  The coefficients alpha are the coordinates of the model in
    KL space and are the quantities to be optimised in a KL-parameterised
    inversion.

    VR Apr 2026
    """
    mval = np.asarray(mval, dtype=float).ravel()
    n_cells = mval.size

    if mean_model is None:
        mean_model = np.zeros(n_cells)
    mean_model = np.asarray(mean_model, dtype=float).ravel()

    centred = mval - mean_model
    alpha = modes @ centred      # (n_modes,)

    if out:
        print('model_to_kl: projected onto %d KL modes' % modes.shape[0])

    return alpha


def kl_to_model(alpha=None, modes=None, mean_model=None,
                shape=None, out=True):
    """
    kl_to_model.

    Parameters
    ----------
    alpha : ndarray, float, shape (n_modes,) or (n_modes_used,)
        KL expansion coefficients.
    modes : ndarray, float, shape (n_modes, n_cells)
        KL eigenmodes as returned by ensemble_to_kl.
    mean_model : ndarray, float, shape (n_cells,), optional
        Ensemble mean.  Added back after reconstruction.
    shape : tuple of int, optional
        Spatial shape (nx, ny, nz) for reshaping output.  If None the output
        is returned as a 1-D array of length n_cells.
    out : bool, optional
        Print summary. Default True.

    Returns
    -------
    mval_rec : ndarray, float, shape (nx, ny, nz) or (n_cells,)
        Reconstructed log-resistivity model.

    Notes
    -----
    Inverse of model_to_kl:
        mval_rec = mean + sum_l alpha[l] * modes[l]

    Only the first len(alpha) modes are used, allowing reconstruction with
    a truncated coefficient vector.

    VR Apr 2026
    """
    alpha = np.asarray(alpha, dtype=float)
    n_used = alpha.size
    modes_used = modes[:n_used, :]          # allow truncated alpha

    n_cells = modes.shape[1]
    if mean_model is None:
        mean_model = np.zeros(n_cells)
    mean_model = np.asarray(mean_model, dtype=float).ravel()

    mval_flat = mean_model + alpha @ modes_used   # (n_cells,)

    if shape is not None:
        mval_rec = mval_flat.reshape(shape)
    else:
        mval_rec = mval_flat

    if out:
        print('kl_to_model: reconstructed from %d KL modes, shape %s'
              % (n_used, str(mval_rec.shape)))

    return mval_rec


def kl_variance_spectrum(singular_values=None, out=True):
    """
    kl_variance_spectrum.

    Parameters
    ----------
    singular_values : ndarray, float, shape (n_modes,)
        Singular values as returned by ensemble_to_kl.
    out : bool, optional
        Print the variance table. Default True.

    Returns
    -------
    variance : ndarray, float, shape (n_modes,)
        Variance explained by each mode (singular_value^2 / total).
    cum_variance : ndarray, float, shape (n_modes,)
        Cumulative variance explained.

    Notes
    -----
    Analogue of dct_spectrum for the KL basis.  Use to decide how many
    modes to retain: typically the first few modes capture > 90% of the
    ensemble variance.

    VR Apr 2026
    """
    S2 = singular_values**2
    total = S2.sum()
    variance = S2 / (total if total > 0.0 else 1.0)
    cum_variance = np.cumsum(variance)

    if out:
        print('\n  KL variance spectrum')
        print('  %6s  %14s  %16s' % ('mode', 'variance_frac', 'cum_variance'))
        for i, (v, cv) in enumerate(zip(variance, cum_variance)):
            print('  %6d  %14.6g  %16.4f' % (i, v, cv))

    return variance, cum_variance


def kl_truncation_analysis(mval=None, modes=None, mean_model=None,
                           shape=None, singular_values=None, out=True):
    """
    kl_truncation_analysis.

    Parameters
    ----------
    mval : ndarray, float, shape (nx, ny, nz)
        Log-resistivity model.
    modes : ndarray, float, shape (n_modes, n_cells)
        KL eigenmodes from ensemble_to_kl.
    mean_model : ndarray, float, shape (n_cells,)
        Ensemble mean from ensemble_to_kl.
    shape : tuple of int
        Spatial shape (nx, ny, nz).
    singular_values : ndarray, float, shape (n_modes,), optional
        Used only for printing cumulative variance alongside error.
    out : bool, optional
        Print summary table. Default True.

    Returns
    -------
    n_modes_list : ndarray, int
        Number of modes used at each test level.
    rms_errors : ndarray, float
        RMS reconstruction error.
    rel_errors : ndarray, float
        Relative RMS reconstruction error.

    Notes
    -----
    Projects mval onto the KL basis and reconstructs using an increasing
    number of modes (1 to n_modes_available), reporting reconstruction
    accuracy at each level.

    VR Apr 2026
    """
    alpha_full = model_to_kl(mval, modes, mean_model, out=False)
    n_max = alpha_full.size
    n_modes_list = np.unique(
        np.round(np.logspace(0, np.log10(n_max), 20)).astype(int)
    )
    n_modes_list = np.clip(n_modes_list, 1, n_max)
    rms_errors = np.zeros(len(n_modes_list))
    rel_errors = np.zeros(len(n_modes_list))

    if singular_values is not None:
        S2 = singular_values**2
        total_var = S2.sum()
        cum_var = np.cumsum(S2) / (total_var if total_var > 0.0 else 1.0)
    else:
        cum_var = None

    if out:
        hdr = ('  %8s  %14s  %14s' % ('n_modes', 'rms_error', 'rel_rms'))
        if cum_var is not None:
            hdr += '  %14s' % 'cum_var_frac'
        print('\n  KL truncation analysis')
        print(hdr)

    for i, nm in enumerate(n_modes_list):
        mval_rec = kl_to_model(alpha_full[:nm], modes, mean_model,
                               shape=shape, out=False)
        rms_errors[i] = dct_reconstruction_error(mval, mval_rec, norm='rms')
        rel_errors[i] = dct_reconstruction_error(mval, mval_rec, norm='rel_rms')
        if out:
            row = '  %8d  %14.6g  %14.6g' % (nm, rms_errors[i], rel_errors[i])
            if cum_var is not None:
                row += '  %14.4f' % cum_var[nm - 1]
            print(row)

    return n_modes_list, rms_errors, rel_errors
