# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
# ---

import os
import sys
import warnings
import time

from sys import exit as error
from datetime import datetime

import numpy as np
import gdal
import scipy as sc
import vtk
import pyvista as pv
import pyvistaqt as pvqt
import discretize
import tarfile
import pylab as pl
from time import sleep

mypath = ["/home/vrath/Py4MT/py4mt/modules/",
          "/home/vrath/Py4MT/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)


import modem as mod
import util as utl
from version import versionstrg

Strng, _ = versionstrg()
now = datetime.now()
print("\n\n"+Strng)
print("Image processing on model"+"\n"+"".join("Date " + now.strftime("%m/%d/%Y, %H:%M:%S")))
print("\n\n")


warnings.simplefilter(action="ignore", category=FutureWarning)

rhoair = 1.e17

total = 0

PFile = r"/home/vrath/Py4MT/py4mt/data/ANN21_Jacobian/Ann21_T"
DFile = r"/home/vrath/Py4MT/py4mt/data/ANN21_Jacobian/Ann21_T3.dat"
MFile = r"/home/vrath/Py4MT/py4mt/data/ANN21_Jacobian/Ann21_Prior100_T_NLCG_033.rho"
SFile = r"/home/vrath/Py4MT/py4mt/data/ANN21_Jacobian/Ann21_Prior100_T-Z3.sns"


start = time.time()
dx, dy, dz, rho, reference = mod.read_model(MFile, trans="LOG10")
elapsed = time.time() - start
total = total + elapsed
print("Used %7.4f s for reading model from %s " % (elapsed, MFile))
print("ModEM reference is "+str(reference))
print("Min/max rho = "+str(np.min(rho))+"/"+str(np.max(rho)))

start = time.time()
dx, dy, dz, sns, reference = mod.read_model(SFile, trans="LOG10")
elapsed = time.time() - start
total = total + elapsed
print("Used %7.4f s for reading model from %s " % (elapsed, SFile))

start = time.time()
Site, Comp, Data, Head = mod.read_data(DFile)
elapsed = time.time() - start
total = total + elapsed
print("Used %7.4f s for reading data from %s " % (elapsed, DFile))


Bounds = [-5.,5., -5.,5., -1. ,3.]
Scale =1.e-3

x, y, z = mod.cells3d(dx, dy, dz, reference=reference)

x, y, z, rho = mod.clip_model(x, y, z, rho, pad = [12, 12, 10])
x, y, z  = Scale*x, Scale*y, -Scale*z
xm, ym, zm = np.meshgrid(x, y, z)

rho[rho>15.] = np.nan

pv.set_plot_theme("document")

cmap = pl.cm.get_cmap("viridis",128)
dargs = dict(cmap=cmap, clim=[1.4, 2.6])


mod = pv.RectilinearGrid(xm, ym, zm)
mod.cell_arrays["resistivity"] = rho.flatten('F')

# contours = mod.contour(np.linspace(1.5, 2.5, 6))

p = pv.Plotter(window_size=[1024, 768])
_ = p.add_mesh(mod.outline(), color="k")
# p.add_mesh(contours, opacity=0.25, clim=[1.4, 2.6])
p.show_grid()
slices = mod.slice_orthogonal(x=0, y=0, z=-1.)
# _ = p.add_mesh(mod, scalars="resistivity")
_ = p.add_mesh(slices, scalars="resistivity", **dargs)
p.add_title("Annecy")
p.add_axes()
p.show(screenshot='my_image.png',auto_close=True)
p.close()

# slices = mod.slice_orthogonal()

# slices.plot(cmap=cmap)
