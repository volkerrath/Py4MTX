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
print("Plot Magnetic tranfer function fit"+"\n"+"".join("Date " + now.strftime("%m/%d/%Y, %H:%M:%S")))
print("\n\n")


warnings.simplefilter(action="ignore", category=FutureWarning)



PredFile = r"/home/vrath/work/MT/Fogo/final_inversions/PTT_100s/run3_NLCG_039_Refsite_FOG933A"
ObsvFile = r"/home/vrath/work/MT/Fogo/final_inversions/PTT_100s/fogo_modem_phaset_tip_100s_data_Refsite_FOG933A"

PerLimits = (0.001, 100.)
TpLimits = (-1., 1.)

PlotFormat = [".pdf", ".png"]
PlotFile = PredFile+"_T_"

"""

required virtual size
"""
EPSG = 5015

start = time.time()

FF = ObsvFile
SiteObs, CompObs, DataObs, HeadObs = mod.read_data(FF+".dat")
obs_rdat = DataObs[:, 6]
obs_idat = DataObs[:, 7]
obs_err = DataObs[:, 8]
obs_per = DataObs[:, 0]
obs_cmp = CompObs
obs_sit = SiteObs
lat = DataObs[:,1]
lon = DataObs[:,2]
x = DataObs[:,3]
y = DataObs[:,4]
z = DataObs[:,5]

FF = PredFile
SiteCal, CompCal, DataCal, HeadCal = mod.read_data(FF+".dat")
cal_rdat = DataCal[:, 6]
cal_idat = DataCal[:, 7]
cal_per = DataCal[:, 0]
cal_cmp = CompCal
cal_sit = SiteCal

np.shape(cal_per)


# Determine graphical parameter.
# print(plt.style.available)
plt.style.use('seaborn-paper')
mpl.rcParams["figure.dpi"] = 400
mpl.rcParams['axes.linewidth'] = 0.5
Fontsize = 10
Labelsize = Fontsize
Linewidth= 1
Grey = 0.7
Lcycle =Lcycle = (cycler('linestyle', ['-', '--', ':', '-.'])
          * cycler('color', ['r', 'g', 'b', 'y']))
mpl.rcParams["figure.dpi"] = 400
mpl.rcParams['axes.linewidth'] = 0.5


Sites = np.unique(SiteObs)

for s in Sites:
    print("Plotting site: "+s)
    site = (obs_sit==s)
    test = ((obs_cmp=="TX") | (obs_cmp=="TY")) & (obs_sit==s)

    if np.any(test) == True:
        site_lon = lon[site][0]
        site_lat = lat[site][0]
        site_utmx, site_utmy = utl.proj_latlon_to_utm(site_lat, site_lon, utm_zone=EPSG)
        site_utmx = int(np.round(site_utmx))
        site_utmy = int(np.round(site_utmy))
        site_elev = z[site][0]

        cmp ="TX"
        cmpo = np.where((obs_cmp==cmp) & (obs_sit==s))
        Tpxro = obs_rdat[cmpo]
        Tpxio = obs_idat[cmpo]
        Tpxe = obs_err[cmpo]
        Perxo = obs_per[cmpo]
        cmpc = np.where((cal_cmp==cmp) & (cal_sit==s))
        Tpxrc = cal_rdat[cmpc]
        Tpxic = cal_idat[cmpc]
        Perxc  = cal_per[cmpc]

        cmp ="TY"
        cmpo = np.where((obs_cmp==cmp) & (obs_sit==s))
        Tpyro = obs_rdat[cmpo]
        Tpyio = obs_idat[cmpo]
        Tpye = obs_err[cmpo]
        Peryo = obs_per[cmpo]
        cmpc = np.where((cal_cmp==cmp) & (cal_sit==s))
        Tpyrc = cal_rdat[cmpc]
        Tpyic = cal_idat[cmpc]
        Peryc = cal_per[cmpc]


        cm = 1/2.54  # centimeters in inches
        fig, axes = plt.subplots(2,1)   #, figsize = (12*cm, 12*cm))
        fig.suptitle(r"Site: "+s
                     +"\nLat: "+str(site_lat)+"   Lon: "+str(site_lon)
                     +"\nUTMX: "+str(site_utmx)+"   UTMY: "+str(site_utmy)
                     +" (EPSG="+str(EPSG)+")  \nElev: "+ str(abs(site_elev))+" m",
                     ha="left", x=0.1,fontsize=Fontsize-1)

        axes[0,0].plot(Perxc.flat, Tpxrc.flat, color="r",linestyle=":")
        axes[0,0].errorbar(Perxo,Tpxro, yerr=Tpxe,
                                linestyle="",
                                marker="o",
                                color="r",
                                lw=0.99,
                                markersize=3)
        axes[0,0].plot(Perxc, Tpxic, ":b")
        axes[0,0].errorbar(Perxo,Tpxio, yerr=Tpxe,
                                linestyle="",
                                marker="o",
                                color="b",
                                lw=0.99,
                                markersize=3)
        axes[0,0].set_xscale("log")
        axes[0,0].set_xlim(PerLimits)
        axes[0,0].legend(["predicted", "observed"])
        axes[0,0].xaxis.set_ticklabels([])
        axes[0,0].tick_params(labelsize=Labelsize-1)
        axes[0,0].set_ylabel("Tpy", fontsize=Fontsize)
        axes[0,0].grid("major", "both", linestyle=":", lw=0.5)


        axes[0,1].plot(Peryc, Tpyrc, ":r")
        axes[0,1].errorbar(Peryo,Tpyro, yerr=Tpxe,
                                linestyle="",
                                marker="o",
                                color="r",
                                lw=0.99,
                                markersize=3)
        axes[0,1].plot(Peryc, Tpyic, ":b")
        axes[0,1].errorbar(Peryc,Tpxio, yerr=Tpxe,
                                linestyle="",
                                marker="o",
                                color="b",
                                lw=0.99,
                                markersize=3)

        axes[0,1].set_xscale("log")
        axes[0,1].set_xlim(PerLimits)
        axes[0,1].legend(["predicted", "observed"])
        axes[0,1].xaxis.set_ticklabels([])
        axes[0,1].tick_params(labelsize=Labelsize-1)
        axes[0,1].set_ylabel("Tpx", fontsize=Fontsize)
        axes[0,1].grid("major", "both", linestyle=":", lw=0.5)



        fig.tight_layout()

        for F in PlotFormat:
            plt.savefig(PlotFile+s+F, dpi=400)


        plt.show()
        plt.close(fig)
    else:
        print("No Tipper for site "+s+"!")
