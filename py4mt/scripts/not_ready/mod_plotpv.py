#!/usr/bin/env python3
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: "1.5"
#       jupytext_version: 1.11.3
# ---

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


PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [PY4MTX_ROOT+"/modules/", PY4MTX_ROOT+"/scripts/"]
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


PlotType = "ortho"
PlotType = "slice"
# PlotType = "lines"
# PlotType = "iso"

Cmap = "jet"

if "ortho" in PlotType.lower():
    position = (0., -3., -8.)
    
if "slice" in PlotType.lower():
    position = (0., 0., -1.)
    normal = (1., 1., 0.)

# Ubinas Ubinas Ubinas Ubinas Ubinas Ubinas Ubinas Ubinas Ubinas Ubinas
Title = "Ubinas Volcano, Peru"
WorkDir = PY4MTX_DATA+"/Peru/Ubinas/"
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
# my_theme.lighting = False
# my_theme.show_edges = True
# my_theme.axes.box = True

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
if "ortho" in PlotType.lower():
    slicepars = dict(clim=LimLogRes, 
                     cmap=lut,
                     above_color=pv.Color("darkgrey", opacity=0), 
                     nan_color="white",
                     nan_opacity=0.,
                     show_scalar_bar=False,
                     use_transparency=True,
                     lighting="none",
                     interpolate_before_map=True,
                     log_scale=False)
    if PlotModl:
        slices = model.slice_orthogonal(position[0], position[1], position[2])
        _ = p.add_mesh(slices, scalars="resistivity", **slicepars)
    if PlotSens:
        slices = sensi.slice_orthogonal(position[0], position[1], position[2])
        _ = p.add_mesh(slices, scalars="resistivity", **slicepars)
    
    if PlotData:
        points = p.add_points(sitep, point_size=10, color="red")
        labels = p.add_point_labels(sitep, sitel, render=True,
                         point_size=100, show_points=True,always_visible=True,
                         render_points_as_spheres=True,shape_opacity=0.0,
                         point_color="red")        

    if PlotUnct:
        slices = model.slice_orthogonal(position[0], position[1], position[2])
        _ = p.add_mesh(slices, scalars="resistivity",  **slicepars)
        points = p.add_points(sitep, point_size=10, color="red")
        labels = p.add_point_labels(sitep, sitel, render=True,
                         point_size=100, show_points=True,always_visible=True,
                         render_points_as_spheres=True,shape_opacity=0.0,
                         point_color="red")        

    
elif"slice" in PlotType.lower():
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
    
    # if PlotData:
    #     _ = p.add_points(sitep, 
    #                      point_size=100.0, 
    #                      render_points_as_spheres=True,
    #                      color="red")
    #     _ = p.add_point_labels(sitep, sitel)
    
# ###############################################################################
# # Opacity by Array
# # ++++++++++++++++
# #
# # You can also use a scalar array associated with the mesh to give each cell
# # its own opacity/transparency value derived from a scalar field. For example,
# # an uncertainty array from a modelling result could be used to hide regions of
# # a mesh that are uncertain and highlight regions that are well resolved.
# #
# # The following is a demonstration of plotting a mesh with colored values and
# # using a second array to control the transparency of the mesh

# model = examples.download_model_with_variance()
# contours = model.contour(10, scalars="Temperature")
# contours.array_names

# ###############################################################################
# # Make sure to flag ``use_transparency=True`` since we want areas of high
# # variance to have high transparency.
# #
# # Also, since the opacity array must be between 0 and 1, we normalize
# # the temperature variance array by the maximum value.  That way high
# # variance will be completely transparent.

# contours["Temperature_var"] /= contours["Temperature_var"].max()

# p = pv.Plotter(shape=(1, 2))

# p.subplot(0, 0)
# p.add_text("Opacity by Array")
# p.add_mesh(
#     contours.copy(),
#     scalars="Temperature",
#     opacity="Temperature_var",
#     use_transparency=True,
#     cmap="bwr",
# )

# p.subplot(0, 1)
# p.add_text("No Opacity")
# p.add_mesh(contours, scalars="Temperature", cmap="bwr")
# p.show()

# elif"spread" in PlotType.lower():
#     slices = model.slice_orthogonal(x=0., y=0., z=-5.)
#     _ = p.add_mesh(slices, scalars="resistivity", **slicepars)
#     slicepars = dict(clim=LimLogRes, 
#                  cmap=lut,
#                  above_color=None, 
#                  nan_color="white",
#                  nan_opacity=1.,
#                  opacity=1.,
#                  #use_transparency=True,
#                  interpolate_before_map=True,
#                  show_scalar_bar=False,
#                  log_scale=False)
      
# elif "cont" in PlotType.lower():
    
#     cntrs = np.arange(LimLogRes[0], LimLogRes[1],0.5)
#     modiso = model.contour(cntrs, scalars="resistivity")
#     _ = p.add_mesh(modiso, scalars="resistivity", **slicepars)

# elif "line" in PlotType.lower():
#     slicepars = dict(clim=LimLogRes, 
#                  cmap=lut,
#                  above_color=None, 
#                  nan_color="white",
#                  nan_opacity=1.,
#                  opacity=1.,
#                  #use_transparency=True,
#                  interpolate_before_map=True,
#                  show_scalar_bar=False,
#                  log_scale=False)
#     slices = model.slice_orthogonal(x=0., y=0., z=-5.)
    
    
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
p.show()

 
# plt.box(False)
# fig = plt.imshow(p.image)
# fig.frameon=False
# fig.axes.get_xaxis().set_visible(False)
# fig.axes.get_yaxis().set_visible(False)
# plt.savefig("test.pdf", dpi=600, edgecolor=None)
# plt.savefig("test.png", dpi=600, transparent=True, edgecolor=None)
# plt.savefig("test.svg", dpi=600, transparent=True, edgecolor=None)

p.close()
