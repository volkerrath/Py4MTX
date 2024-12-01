#!/usr/bin/env python3

import os
import sys
import warnings
import time

from sys import exit as error
from datetime import datetime

import numpy as np
from osgeo import gdal
import scipy as sc

import vtk
import pyvista as pv
from pyvista import themes

import PVGeo as pg

import discretize
import tarfile
import pylab as pl


import matplotlib as mpl
import matplotlib.pyplot as plt


JACOPYAN_DATA = os.environ["JACOPYAN_DATA"]
JACOPYAN_ROOT = os.environ["JACOPYAN_ROOT"]

mypath = [JACOPYAN_ROOT+"/modules/", JACOPYAN_ROOT+"/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)


import modem as mod
import util as utl
from version import versionstrg

version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")


warnings.simplefilter(action="ignore", category=FutureWarning)



Bounds = [-5.,5., -5.,5., -1. ,3.]
Pads = [14, 14, 42]

Scale = 1.e-3
LimlogSns = [-5., 0.]
LimLogRes = [-1, 4.]
StepContrs = 0.5


PlotModl = True
PlotSens = False
PlotUnct = False
PlotData = False

StepContrs=0.5


Cmap = "jet"

position = (0., 0., -1.)
normal = (1., 1., 0.)

# Ubinas Ubinas Ubinas Ubinas Ubinas Ubinas Ubinas Ubinas Ubinas Ubinas
Title = "Ubinas Volcano, Peru"
WorkDir = JACOPYAN_DATA+"/Peru/Ubinas/"
if not WorkDir.endswith("/"): WorkDir = WorkDir+"/"

Datfile = WorkDir + "U38_ZPTss1"
ModFile = WorkDir + "Ubi38_ZssPT_Alpha02_NLCG_023"
SnsFile = WorkDir+"/sens_euc/Ubi38_ZPT_nerr_sp-8_total_euc_sqr_max_sns"

Plotfile = WorkDir + "Ubi38_ZssPT_Alpha02_NLCG_023"



if PlotModl:
    dx, dy, dz, rho, reference, _ = mod.read_mod(ModFile, ".rho",trans="log10")
    aircells = np.where(rho>15.)
    rho[aircells]=np.NaN
    print("Reading model from %s " % (ModFile))
    print("ModEM reference is "+str(reference))
    print("Min/max rho = "+str(np.nanmin(rho))+"/"+str(np.nanmax(rho)))
    x, y, z, vals = mod.model_to_pv(dx=dx, dy=dy, dz=dz, rho=rho, 
                                    reference=reference, scale=Scale, pad=Pads)
    model = pv.RectilinearGrid(y, x, z)
    model.cell_data["resistivity"] = vals
    
if PlotSens:
    dx, dy, dz, sns, reference, _ = mod.read_mod(SnsFile,".rho",trans="log10")
    print("Reading sensitivity from %s " % (SnsFile))
    _, _, _, sens = mod.model_to_pv(dx=dx, dy=dy, dz=dz, rho=rho, 
                                    reference=reference, scale=Scale, pad=Pads)
    sensi = pv.RectilinearGrid(y, x, z)
    sensi.cell_data["sensitivity"] = sens
    
if PlotUnct:
    dx, dy, dz, rho, reference, _ = mod.read_mod(ModFile, ".rho",trans="log10")
    aircells = np.where(rho>15.)
    rho[aircells]=np.NaN
    print("Reading model from %s " % (ModFile))
    print("ModEM reference is "+str(reference))
    print("Min/max rho = "+str(np.nanmin(rho))+"/"+str(np.nanmax(rho)))
    x, y, z, vals = mod.model_to_pv(dx=dx, dy=dy, dz=dz, rho=rho, 
                                    reference=reference, scale=Scale, pad=Pads)
    model = pv.RectilinearGrid(y, x, z)
    model.cell_data["resistivity"] = vals
    
    dx, dy, dz, sns, reference, _ = mod.read_mod(SnsFile,".rho",trans="log10")
    print("Reading sensitivity from %s " % (SnsFile))
    _, _, _, sens = mod.model_to_pv(dx=dx, dy=dy, dz=dz, rho=rho, 

                              reference=reference, scale=Scale, pad=Pads)
    sensi = pv.RectilinearGrid(y, x, z)
    sensi.cell_data["sensitivity"] = sens   

if PlotData:
    if (not PlotSens) and (not PlotModl) and (not PlotUnct):
        error("No mesh given, data cannot be plotted! Exit.")
    site, _, data, _ = mod.read_data(Datfile=Datfile)
    xdat, ydat, zdat, sitenam, sitenum = mod.data_to_pv(data=data, site=site, 
                                                scale=Scale)
    sdata = np.column_stack(( xdat, ydat, zdat))
    sitep = pv.PolyData(sdata)
    sitel = sitenam

cmap = mpl.colormaps[Cmap] 
#pv.set_jupyter_backend("trame")
mtheme = pv.themes.DocumentTheme()
mtheme.nan_color = pv.Color("darkgray", opacity=0)
mtheme.above_range_color = pv.Color("darkgrey", opacity=0)
# mtheme.lighting = False
# mtheme.show_edges = True
# mtheme.axes.box = True

# mtheme.colorbar_orientation = "vertical"
# mtheme.colorbar_vertical.height=0.6
# mtheme.colorbar_vertical.position_y=0.2
# mtheme.colorbar_vertical.title="log resistivity"
mtheme.font.size=14
mtheme.font.title_size=16
mtheme.font.label_size=16


pv.global_theme.load_theme(mtheme)


lut = pv.LookupTable()
lut.apply_cmap(cmap, n_values=128, flip=True)

lut.scalar_range = LimLogRes
lut.above_range_color = None
lut.nan_color = None




p = pv.Plotter(window_size=[1600, 1300], theme=mtheme, notebook=False, off_screen=True)

p.add_title(Title)
_ = p.add_mesh(model.outline(), color="k")
grid_labels = dict(ztitle="elev (km)", xtitle="E-W (km)", ytitle="S-N (km)")
p.show_grid(**grid_labels)
#p.view_xy()

slicepars = dict(clim=LimLogRes, 
             cmap=lut,
             above_color="white", 
             nan_color="white",
             nan_opacity=0.8,
             opacity=0.,
             use_transparency=True,
             interpolate_before_map=True,
             show_scalar_bar=False,
             log_scale=False)


if PlotModl:
    slices = model.slice(normal=normal, origin=position)
    _ = p.add_mesh(slices, scalars="resistivity", **slicepars)
    
if PlotSens:
    slices = sensi.slice(normal=normal, origin=position)
    _ = p.add_mesh(slices, scalars="resistivity", **slicepars)
    
    
    
_ = p.add_scalar_bar(title="log res", 
                         vertical=True, 
                         position_y=0.2,
                         position_x=0.9,
                         height=0.6,
                         title_font_size=26, 
                         label_font_size=16,
                         bold=False,
                         n_labels = 6,
                         fmt="%3.1f")


p.add_axes()
p.save_graphic("test_save.pdf")
p.save_graphic("test_save.svg")
p.show()



p.close()