#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
calculates  statistics for Seismicity-Resistivity correlation


@author: Svetlana Byrdina & Volker Rath,  Jan 2025


"""

import os
import sys
from sys import exit as error
import time
from datetime import datetime
import csv
import numpy as np

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [PY4MTX_ROOT+"/py4mt/modules/", PY4MTX_ROOT+"/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)

#import modules
import modem as mod
import util as utl
import jacproc as jac
from version import versionstrg
import matplotlib.pyplot as plt

rng = np.random.default_rng()
nan = np.nan  # float("NaN")

version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")

rhoair = 1.e17



SeisFile = r"/home/vrath/work/MT_Data/SeisRes/data/Catalog_new_shallow.csv"
ModFile = r"/home/vrath/work/MT_Data/SeisRes/data/TACG_Z2_Alpha02_NLCG_016"

geocenter = [-17.489151, -70.031403] #Tac

utm_E, utm_N = utl.proj_latlon_to_utm(geocenter[0], geocenter[1], utm_zone=32719)
utmcenter = [utm_E, utm_N, 0.0]


#profile = [339263., 429899., 7984401., 8113508.]# x == E
profile = [-60000.,  40000., -72000., 50000.] #x0, x1, y0, y1 model koordinates

profile_center = [profile[0] + 0.5*(profile[1] - profile[0]),
                  profile[2] + 0.5*(profile[3] - profile[2])]


boundary = ["box", profile_center[1], profile_center[0], 15000., 100000., 60000., 15000., 0., 0., -30.] #north is y!

n_ml = 6
ml_bins = np.linspace(0., 5., n_ml)


total = 0
start = time.perf_counter()

# read seismology file
seis = np.loadtxt(SeisFile, delimiter=',', skiprows=1)
#[sLat, sLon, sDep, ML]
E_sutm, N_sutm = utl.proj_latlon_to_utm(seis[:,0], seis[:,1], utm_zone=32719)
Z_s = seis[:,2]*1000.
M_l = seis[:,3]

#seismic events in model coordinates (utmcenter is really a model center while modcenter is bottom-left point)
Es = E_sutm - utmcenter[0]
Ns = N_sutm - utmcenter[1]
Zs = Z_s - utmcenter[2]


dx, dy, dz, rho, reference,_= mod.read_mod(ModFile , out=True, trans="LOG10")
# write_model_mod(ModFile_out+'.rho', dx, dy, dz, rho,reference,out = True)

aircells = np.where(rho>rhoair/10)

elapsed = time.perf_counter() - start
total = total + elapsed
print(" Used %7.4f s for reading model from %s "
      % (elapsed, ModFile + ".rho"))

xc, yc, zc = mod.cells3d(dx, dy, dz)

if reference is None:
    modcenter = [0.5 * np.sum(dx), 0.5 * np.sum(dy), 0.0]
else:
    modcenter = reference

xc = xc + modcenter[0]
yc = yc + modcenter[1]
zc = zc + modcenter[2]

print(" center is", modcenter)

nx = np.shape(xc)[0]
ny = np.shape(yc)[0]
nz = np.shape(zc)[0]


geom = boundary[0]
bcent = boundary[1:4]
baxes = boundary[4:7]
bangl = boundary[7:10]

md = np.shape(rho)

rho_out = rho.copy()
num_out = np.zeros(md,dtype=int)
num_out  = np.zeros((md[0],md[1],md[2],n_ml))


for kk in np.arange(nz-1):
    zpoint = zc[kk]
    for jj in np.arange(ny-1):
        ypoint = yc[jj]
        for ii in np.arange(nx-1):
            xpoint = xc[ii]
            position = [xpoint, ypoint, zpoint]
            if not mod.in_box(position, bcent, baxes, bangl):
                rho_out[ii, jj, kk] = np.nan


x, y, z = mod.cells3d(dx, dy, dz, center=False)

x = x +  modcenter[0]
y = y +  modcenter[1]
z = z +  modcenter[2]


for event in np.arange(np.shape(seis)[0]):
#for event in np.arange(50):
    position = [Es[event], Ns[event], Zs[event]]
    magnitude = M_l[event]
    print("event number", event)
    for kk in np.arange(nz-1):
        for jj in np.arange(ny-1):
            for ii in np.arange(nx-1):
                if (
                    (position[0] > x[ii]) & (position[0] < x[ii+1]) &
                    (position[1] > y[jj]) & (position[1] < y[jj+1]) &
                    (position[2] > z[kk]) & (position[2] < z[kk+1])
                    ):
                    num_out[ii,jj,kk, 0] = num_out[ii,jj,kk,0] + 1
                    for nb in np.arange(n_ml-1):
                        if (magnitude>ml_bins[nb]) & (magnitude<ml_bins[nb]):
                            num_out[ii,jj,kk,nb+1]=num_out[ii,jj,kk,nb+1]+1



index = np.where(np.isfinite(rho_out))
v1 = rho_out[index]
v2 = num_out[index]

np.savez_compressed(SeisFile.replace(".csv",".npz"),
                    index= index,
                    num_out=num_out,
                    rho_out=rho_out,
                    seis = seis,
                    boundary=boundary
                    )

elapsed = time.perf_counter() - start

print(" Used %7.4f s for processing/writing model "
     % (elapsed))
print("\n")

total = total + elapsed
print(" Total time used:  %f s " % (total))
