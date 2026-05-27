#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 15:12:01 2026

@author: vrath
"""


import os
import sys
from pathlib import Path
import inspect
import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Py4MTX-specific settings and imports
# ---------------------------------------------------------------------------
PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

for _base in [PY4MTX_ROOT + "/py4mt/modules/"]:
    for _p in [Path(_base), *Path(_base).rglob("*")]:
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

from version import versionstrg
import util as utl

version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())
titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng + "\n\n")

WORK_DIR = "/home/vrath/work/MT_Data/Peru/SeisTomoPeru/"
DATA_DIR = WORK_DIR + "/kan2025/"

# Load file
ncfile = DATA_DIR + "FD_vp_model.nc"
print("\nRead File", ncfile )
ds_vp = xr.open_dataset(ncfile)
print(ds_vp)
print(ds_vp.data_vars)

ncfile = DATA_DIR + "FD_vs_model.nc"
print("\nRead File", ncfile )
ds_vs = xr.open_dataset(ncfile)
print(ds_vs)
print(ds_vs.data_vars)

ncfile = DATA_DIR + "FD_rho_model.nc"
print("\nRead File", ncfile )
ds_rho = xr.open_dataset(ncfile)
print(ds_rho)
print(ds_rho.data_vars)

lat = ds_vp["lat"]
lon = ds_vp["lon"]
depth = ["depth"]

v_p =  ds_vp["data"]
np.savez_compressed("v_p.npz", 
                   lat=lat, lon=lon, depth=depth,
                   data = v_p,
                   allow_pickle=True)
v_s =  ds_vs["data"]
np.savez_compressed("v_s.npz", 
                   lat=lat, lon=lon, depth=depth,
                   data = v_s,
                   allow_pickle=True)
v_r = v_s/v_p
np.savez_compressed("v_r.npz", 
                   lat=lat, lon=lon, depth=depth,
                   data = v_r,
                   allow_pickle=True)

rho = ds_rho["data"]

np.savez_compressed("rho.npz", 
                   lat=lat, lon=lon, depth=depth,
                   data = rho,
                   allow_pickle=True)

# ds.to_netcdf("model.nc")

print("Done!")


def tomo_save(file="data.h5",
           dataset=None,
           compression="gzip",
           complevel=4):
 """
 Save tomography dataset to HDF5 or NetCDF.

 Parameters
 ----------
 file : str
     Output filename (.h5 or .nc).

 dataset : xarray.Dataset
     Dataset containing:
         - coordinates: lat, lon, depth
         - variable: data

 compression : str
     HDF5 compression method.

 complevel : int
     Compression level.
 """

 import sys
 import h5py

 if dataset is None:
     raise ValueError("dataset must not be None")

 fl = file.lower()

 # ---------------------------------------------------------
 # HDF5
 # ---------------------------------------------------------
 if fl.endswith(".h5") or fl.endswith(".hdf5"):

     with h5py.File(file, "w") as f:

         # -------------------------------------------------
         # coordinates
         # -------------------------------------------------
         for coord in ["lat", "lon", "depth"]:

             if coord in dataset.coords:
                 dset = f.create_dataset(
                     coord,
                     data=dataset[coord].values
                 )

                 # coordinate attributes
                 for k, v in dataset[coord].attrs.items():
                     dset.attrs[k] = v

         # -------------------------------------------------
         # main data
         # -------------------------------------------------
         h5data = f.create_dataset(
             "data",
             data=dataset["data"].values,
             compression=compression,
             compression_opts=complevel
         )

         # variable attributes
         for k, v in dataset["data"].attrs.items():
             h5data.attrs[k] = v

         # global attributes
         for k, v in dataset.attrs.items():
             f.attrs[k] = v

 # ---------------------------------------------------------
 # NetCDF
 # ---------------------------------------------------------
 elif fl.endswith(".nc"):

     encoding = {
         "data": {
             "zlib": True,
             "complevel": complevel
         }
     }

     dataset.to_netcdf(
         file,
         format="NETCDF4",
         encoding=encoding
     )

 # ---------------------------------------------------------
 # unknown
 # ---------------------------------------------------------
 else:

     raise ValueError(
         f"tomo_save: unknown file type: {file}"
     )

         
tomo_save("vs.h5", ds_vs)
tomo_save("vp.h5", ds_vp)

tomo_save("rho.h5", ds_rho)
