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
print("Plot Impedance fit"+"\n"+"".join("Date " + now.strftime("%m/%d/%Y, %H:%M:%S")))
print("\n\n")

warnings.simplefilter(action="ignore", category=FutureWarning)


# Graphical paramter. Determine the plot formats produced,
# and the required resolution:
WorkDir =  r"/home/vrath/work/MT/Fogo/final_inversions/ZZT_100s/"
PredFile = r"/home/vrath/work/MT/Fogo/final_inversions/ZZT_100s/run7_NLCG_035_Refsite_FOG933A"
ObsvFile = r"/home/vrath/work/MT/Fogo/final_inversions/ZZT_100s/fogo_modem_data_zzt_3pc_003_100s_edited_Refsite_FOG933A"

PerLimits = (0.001, 200.)
ZLimitsXX = ()
ZLimitsXY = ()
PlotFile = "Fogo_ZZ_final"



PlotFormat = [".pdf", ".png", ".svg"]
PdfCatalog = True
if not ".pdf" in PlotFormat:
    error(" No pdfs generated. No catalog possible!")
    PdfCatalog = False
PdfCName = PlotFile


"""

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


# Determine graphical parameter.
# print(plt.style.available)
plt.style.use("seaborn-paper")
mpl.rcParams["figure.dpi"] = 400
mpl.rcParams["axes.linewidth"] = 0.5
Fontsize = 10
Labelsize = Fontsize
Linewidth= 1
Grey = 0.7
Lcycle =Lcycle = (cycler("linestyle", ["-", "--", ":", "-."])
          * cycler("color", ["r", "g", "b", "y"]))
mpl.rcParams["figure.dpi"] = 400
mpl.rcParams["axes.linewidth"] = 0.5


plt.style.use("seaborn-paper")

Sites = np.unique(SiteObs)

for s in Sites:
    print("Plotting site: "+s)
    site = (obs_sit==s)
    site_lon = lon[site][0]
    site_lat = lat[site][0]
    site_utmx, site_utmy = utl.proj_latlon_to_utm(site_lat, site_lon, utm_zone=EPSG)
    site_utmx = int(np.round(site_utmx))
    site_utmy = int(np.round(site_utmy))
    site_elev = z[site][0]

    cmp ="ZXX"
    cmpo = np.where((obs_cmp==cmp) & (obs_sit==s))
    Zxxro = np.abs(obs_rdat[cmpo])
    Zxxio = np.abs(obs_idat[cmpo])
    Zxxe = obs_err[cmpo]
    Perxxo = obs_per[cmpo]
    cmpc = np.where((cal_cmp==cmp) & (cal_sit==s))
    Zxxrc = np.abs(cal_rdat[cmpc])
    Zxxic = np.abs(cal_idat[cmpc])
    Perxxc = cal_per[cmpc]

    cmp ="ZXY"
    cmpo = np.where((obs_cmp==cmp) & (obs_sit==s))
    Zxyro = np.abs(obs_rdat[cmpo])
    Zxyio = np.abs(obs_idat[cmpo])
    Zxye = obs_err[cmpo]
    Perxyo = obs_per[cmpo]
    cmpc = np.where((cal_cmp==cmp) & (cal_sit==s))
    Zxyrc = np.abs(cal_rdat[cmpc])
    Zxyic = np.abs(cal_idat[cmpc])
    Perxyc = cal_per[cmpc]

    cmp ="ZYX"
    cmpo = np.where((obs_cmp==cmp) & (obs_sit==s))
    Zyxro = np.abs(obs_rdat[cmpo])
    Zyxio = np.abs(obs_idat[cmpo])
    Zyxe = obs_err[cmpo]
    Peryxo = obs_per[cmpo]
    cmpc = np.where((cal_cmp==cmp) & (cal_sit==s))
    Zyxrc = np.abs(cal_rdat[cmpc])
    Zyxic = np.abs(cal_idat[cmpc])
    Peryxc = cal_per[cmpc]

    cmp ="ZYY"
    cmpo = np.where((obs_cmp==cmp) & (obs_sit==s))
    Zyyro = np.abs(obs_rdat[cmpo])
    Zyyio = np.abs(obs_idat[cmpo])
    Zyye = obs_err[cmpo]
    Peryyo = obs_per[cmpo]
    cmpc = np.where((cal_cmp==cmp) & (cal_sit==s))
    Zyyrc = np.abs(cal_rdat[cmpc])
    Zyyic = np.abs(cal_idat[cmpc])
    Peryyc = cal_per[cmpc]


    cm = 1/2.54  # centimeters in inches
    fig, axes = plt.subplots(2,2, figsize = (16*cm, 12*cm))
    fig.suptitle(r"Site: "+s
                 +"\nLat: "+str(site_lat)+"   Lon: "+str(site_lon)
                 +"\nUTMX: "+str(site_utmx)+"   UTMY: "+str(site_utmy)
                 +" (EPSG="+str(EPSG)+")  \nElev: "+ str(abs(site_elev))+" m",
                 ha="left", x=0.1,fontsize=Fontsize-1)

    axes[0,0].plot(Perxxc, Zxxrc, color="r",linestyle=":")
    axes[0,0].errorbar(Perxxo,Zxxro, yerr=Zxxe,
                            linestyle="",
                            marker="o",
                            color="r",
                            lw=0.99,
                            markersize=3)
    axes[0,0].plot(Perxxc, Zxxic, color="b",linestyle=":")
    axes[0,0].errorbar(Perxxo,Zxxio, yerr=Zxxe,
                            linestyle="",
                            marker="o",
                            color="b",
                            lw=0.99,
                            markersize=3)
    axes[0,0].set_xscale("log")
    axes[0,0].set_yscale("log")
    axes[0,0].set_xlim(PerLimits)
    if ZLimitsXX != ():
        axes[0,0].set_ylim(ZLimitsXX)
    axes[0,0].legend(["real", "imag"])
    axes[0,0].xaxis.set_ticklabels([])
    axes[0,0].tick_params(labelsize=Labelsize-1)
    axes[0,0].set_ylabel("|ZXX|", fontsize=Fontsize)
    axes[0,0].grid("major", "both", linestyle=":", lw=0.5)


    axes[0,1].plot(Perxyc, Zxyrc, color="r",linestyle=":")
    axes[0,1].errorbar(Perxyo,Zxyro, yerr=Zxye,
                            linestyle="",
                            marker="o",
                            color="r",
                            lw=0.99,
                            markersize=3)
    axes[0,1].plot(Perxyc, Zxyic, color="b",linestyle=":")
    axes[0,1].errorbar(Perxyo,Zxyio, yerr=Zxye,
                            linestyle="",
                            marker="o",
                            color="b",
                            lw=0.99,
                            markersize=3)
    axes[0,1].set_xscale("log")
    axes[0,1].set_yscale("log")
    axes[0,1].set_xlim(PerLimits)
    if ZLimitsXY != ():
        axes[0,1].set_ylim(ZLimitsXY)
    axes[0,1].legend(["real", "imag"])
    axes[0,1].xaxis.set_ticklabels([])
    axes[0,1].tick_params(labelsize=Labelsize-1)
    axes[0,1].set_ylabel("|ZXY|", fontsize=Fontsize)
    axes[0,1].grid("major", "both", linestyle=":", lw=0.5)


    axes[1,0].plot(Peryxc, Zyxrc, color="r",linestyle=":")
    axes[1,0].errorbar(Peryxo,Zyxro, yerr=Zyxe,
                            linestyle="",
                            marker="o",
                            color="r",
                            lw=0.99,
                            markersize=3)
    axes[1,0].plot(Peryxc, Zyxic, color="b",linestyle=":")
    axes[1,0].errorbar(Peryxo,Zyxio, yerr=Zyxe,
                            linestyle="",
                            marker="o",
                            color="b",
                            lw=0.99,
                            markersize=3)
    axes[1,0].set_xscale("log")
    axes[1,0].set_yscale("log")
    axes[1,0].set_xlim(PerLimits)
    if ZLimitsXY != ():
        axes[1,0].set_ylim(ZLimitsXY)
    axes[1,0].legend(["real", "imag"])
    axes[1,0].tick_params(labelsize=Labelsize-1)
    axes[1,0].set_ylabel("|ZYX|", fontsize=Fontsize)
    axes[1,0].grid("major", "both", linestyle=":", lw=0.5)

    axes[1,1].plot(Peryyc, Zyyrc, color="r",linestyle=":")
    axes[1,1].errorbar(Peryyo,Zyyro, yerr=Zyye,
                            linestyle="",
                            marker="o",
                            color="r",
                            lw=0.99,
                            markersize=3)
    axes[1,1].plot(Peryyc, Zyyic, color="b",linestyle=":")
    axes[1,1].errorbar(Peryyo,Zyyio, yerr=Zyye,
                            linestyle="",
                            marker="o",
                            color="b",
                            lw=0.99,
                            markersize=3)
    axes[1,1].set_xscale("log")
    axes[1,1].set_yscale("log")
    axes[1,1].set_xlim(PerLimits)
    if ZLimitsXX != ():
        axes[1,1].set_ylim(ZLimitsXX)
    axes[1,1].legend(["real", "imag"])
    axes[1,1].tick_params(labelsize=Labelsize-1)
    axes[1,1].set_ylabel("|ZYY|", fontsize=Fontsize)
    axes[1,1].grid("major", "both", linestyle=":", lw=0.5)



    fig.tight_layout()

    for F in PlotFormat:
        plt.savefig(WorkDir+PlotFile+"_"+s+F, dpi=400)


    plt.show()
    plt.close(fig)

if PdfCatalog:
    utl.make_pdf_catalog(WorkDir, PdfCName)

