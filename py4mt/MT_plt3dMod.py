# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.2
# ---

import os
import sys
from sys import exit as error
import numpy as np
import gdal
import scipy as sc
import vtk
import pyvista as pv
import pylab as pl
from time import sleep

import modem as mod
import util as utl


def plotsliceMod(dx=None ,dy=None ,dz=None ,rho=None, position=None):
    """
    Plot slices through

    Parameters
    ----------
    dx : float
        Cell delta x in m. The default is None.
    dy : float
        Cell delta z in m. The default is None.
    dz : float
        Cell delta y in m. The default is None.
    rho : float
        Resitivity of subsurface in Ohm.m, no log transform.
        The default is None.
    plane : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    fig = pl.figure(figsize=(20, 5.5))

    fig.canvas.draw()


def plot3dMod(dx=None, dy=None, dz=None, rho=None, TopoFile=None,
              View=None, PlotFile=None, PlotFmt='png'):
    """
    Plot 3d view of ModEM model.

    Parameters
    ----------
    dx : float
        Cell delta x in m. The default is None.
    dy : float
        Cell delta z in m. The default is None.
    dz : float
        Cell delta y in m. The default is None.
    rho : float
        Resitivity of subsurface in Ohm.m, no log transform.
        The default is None.
    TopoFile : TYPE, optional
        DESCRIPTION. The default is None.
    View : TYPE, optional
        DESCRIPTION. The default is None.
    PlotFile : TYPE, optional
        DESCRIPTION. The default is None.
    PlotFmt : TYPE, optional
        DESCRIPTION. The default is 'png'.

    Returns
    -------
    None.

    """
    if dx is None or dy is None or dz is None or rho is None:
        error('Mesh erroneous! Exit.')

    x, y, z = mod.cells3d(dx, dy, dz, otype='n')

    xm, ym, zm = np.meshgrid(x, y, z)
    grid = pv.StructuredGrid(xm, ym, -zm)

    Tile = '/home/vrath/work/MT/N45E006.hgt'
    srtm1_0 = gdal.Open(Tile)
    myarray = np.array(Tile.GetRasterBand(1).ReadAsArray())


