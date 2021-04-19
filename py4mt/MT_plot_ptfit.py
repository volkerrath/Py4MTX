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
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
import pyproj as proj

mypath = ["/home/vrath/Py4MT/py4mt/modules/",
          "/home/vrath/Py4MT/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)

import mtplots as mtp

# import gdal
# import scipy as sc
# import vtk
# import pyvista as pv
# import pyvistaqt as pvqt
# import discretize
# import tarfile
# import pylab as pl
# from time import sleep

import mtplots as mtp
import modem as mod
import util as utl
from version import versionstrg

Strng, _ = versionstrg()
now = datetime.now()
print("\n\n"+Strng)
print("Plot Phase Tensor fit"+"\n"+"".join("Date " + now.strftime("%m/%d/%Y, %H:%M:%S")))
print("\n\n")


warnings.simplefilter(action="ignore", category=FutureWarning)


# Graphical paramter. Determine the plot formats produced,
# and the required resolution:

_, Fontsize, Labelsize, Linewidth, Lcycle, Grey = mtp.get_plot_params()

PlotFormat = [".pdf", ".png"]

plt.style.use('seaborn-white')
# PredFile = r"/home/vrath/Py4MT/py4mt/tmp/FBB902A_calculated"
# ObsvFile = r"/home/vrath/Py4MT/py4mt/tmp/FBB902A_observed"
PredFile = r"/home/vrath/work/MT/Fogo/final_inversions/PTT_100s/run3_NLCG_039_Refsite_FOG933A"
ObsvFile = r"/home/vrath/work/MT/Fogo/final_inversions/PTT_100s/fogo_modem_phaset_tip_100s_data_Refsite_FOG933A"

PlotFile = PredFile+"_PT_"

"""

"""
EPSG = 5015

start = time.time()

FF = ObsvFile
Siteo, Comp, Data, Head = mod.read_data(FF+".dat")
obs_dat = Data[:, 6]
obs_err = Data[:, 7]
obs_per = Data[:, 0]
obs_cmp = Comp
lat = Data[:,1]
lon = Data[:,2]
x = Data[:,3]
y = Data[:,4]
z = Data[:,5]

FF = PredFile
Sitec, Comp, Data, Head = mod.read_data(FF+".dat")
cal_dat= Data[:, 6]
cal_per= Data[:, 0]
cal_cmp= Comp



Sites = np.unique(Siteo)

for s in Sites[0:3]:

    site = (Siteo==s)
    print(s, site)
    site_dobs = obs_dat[site].copy()
    site_errs = obs_err[site].copy()
    site_ocmp = obs_cmp[site].copy()
    site_oper = obs_per[site].copy()
    site_lat = lat[site][0].copy()
    site_lon = lon[site][0].copy()
    site_utmx, site_utmy = utl.proj_latlon_to_utm(site_lat, site_lon, utm_zone=EPSG)
    site_utmx = int(np.round(site_utmx))
    site_utmy = int(np.round(site_utmy))
    site_elev = z[site][0]


    site = (Sitec==s)
    site_dcal = cal_dat[site].copy()
    site_ccmp = cal_cmp[site].copy()
    site_cper = cal_per[site].copy()

    cmp ="PTXX"
    cmpo = (site_ocmp==cmp)
    PTxxo = site_dobs[cmpo]
    PTxxe = site_errs[cmpo]
    Perxxo = site_oper[cmpo]
    cmpc = (site_ccmp==cmp)
    PTxxc = site_dcal[cmpc]
    Perxxc = site_cper[cmpc]
    print(Perxxc)
    print(PTxxc)

    cmp ="PTXY"
    cmpo = (site_ocmp==cmp)
    PTxyo = site_dobs[cmpo]
    PTxye = site_errs[cmpo]
    Perxyo = site_oper[cmpo]
    cmpc = (site_ccmp==cmp)
    PTxyc = site_dcal[cmpc]
    Perxyc = site_cper[cmpc]

    cmp ="PTYX"
    cmpo = (site_ocmp==cmp)
    PTyxo = site_dobs[cmpo]
    PTyxe = site_errs[cmpo]
    Peryxo = site_oper[cmpo]
    cmpc = (site_ccmp==cmp)
    PTyxc = site_dcal[cmpc]
    Peryxc = site_cper[cmpc]

    cmp ="PTYY"
    cmpo = (site_ocmp==cmp)
    PTyyo = site_dobs[cmpo]
    PTyye = site_errs[cmpo]
    Peryyo = site_oper[cmpo]
    cmpc = (site_ccmp==cmp)
    PTyyc = site_dcal[cmpc]
    Peryyc = site_cper[cmpc]



    # mpl.rcParams["figure.dpi"] = 300
    fig, axes = plt.subplots(2, 2)
    fig.suptitle(r"Site: "+s
                 +"  /  Lat: "+str(site_lat)+"   Lon: "+str(site_lon)
                 +"  /  UTMX: "+str(site_utmx)+"   UTMY: "+str(site_utmy)
                 +" (EPSG="+str(EPSG)+")  /  Elev: "+ str(site_elev)+" m",
                 ha="left", x=0.,fontsize=Fontsize)


    axes[0,0].plot(Perxxc, PTxxc, "-r")
    axes[0,0].errorbar(Perxxo,PTxxo, yerr=PTxxe,
                            linestyle="",
                            marker="o",
                            color="b",
                            lw=0.99,
                            markersize=5)
    axes[0,0].set_ylabel("PTXX", fontsize=Fontsize)
    axes[0,0].xaxis.set_ticklabels([])
    axes[0,0].grid("major", "both", linestyle=":", lw=0.8)
    axes[0,0].set_xscale("log")


    axes[0,1].plot(Perxyc, PTxyc, "-r")
    axes[0,1].errorbar(Perxyo,PTxyo, yerr=PTxye,
                            linestyle="",
                            marker="o",
                            color="b",
                            lw=0.99,
                            markersize=5)
    axes[0,1].set_ylabel("PTXY", fontsize=Fontsize)
    axes[0,1].xaxis.set_ticklabels([])
    axes[0,1].tick_params(bottom='off', labelbottom='off')

    axes[0,1].grid("major", "both", linestyle=":", lw=0.8)
    axes[0,1].set_xscale("log")


    axes[1,0].plot(Peryxc, PTyxc, "-r")
    axes[1,0].errorbar(Peryxo,PTyxo, yerr=PTyxe,
                            linestyle="",
                            marker="o",
                            color="b",
                            lw=0.99,
                            markersize=5)
    axes[1,0].set_xlabel("Period (s)", fontsize=Fontsize)
    axes[1,0].set_ylabel("PTYX", fontsize=Fontsize)
    axes[1,0].grid("major", "both", linestyle=":", lw=0.8)
    axes[1,0].set_xscale("log")

    axes[1,1].plot(Peryyc, PTyyc, "-r")
    axes[1,1].errorbar(Peryyo,PTyyo, yerr=PTyye,
                            linestyle="",
                            marker="o",
                            color="b",
                            lw=0.99,
                            markersize=5)
    axes[1,1].set_xlabel("Period (s)", fontsize=Fontsize)
    axes[1,1].set_ylabel("PTYY", fontsize=Fontsize)
    axes[1,1].grid("major", "both", linestyle=":", lw=0.8)
    axes[1,1].set_xscale("log")

    fig.tight_layout()
    plt.show()

    for F in PlotFormat:
        fig = plt.savefig(PlotFile+s+F, dpi=400)

    #plt.close(fig)

# elapsed = time.time() - start
# print("Used %7.4f s for processing data files" % (elapsed))
