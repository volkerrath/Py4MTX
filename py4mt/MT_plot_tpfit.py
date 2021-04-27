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
print("Plot Magnetic transfer function (tipper) fit"+"\n"+"".join("Date " + now.strftime("%m/%d/%Y, %H:%M:%S")))
print("\n\n")


warnings.simplefilter(action="ignore", category=FutureWarning)

WorkDir =  r"/home/vrath/work/MT/Annecy/ANN25a_best/"
PredFile = r"/home/vrath/work/MT/Annecy/ANN25a_best/Ann25c_ZPT_200_Alpha01_NLCG_017"
ObsvFile = r"/home/vrath/work/MT/Annecy/ANN25a_best/Ann25_ZPTb"
PlotDir = WorkDir + 'Plots/'
print(' Plots written to: %s' % PlotDir)
if not os.path.isdir(PlotDir):
    print(' File: %s does not exist, but will be created' % PlotDir)
    os.mkdir(PlotDir)


PerLimits = (0.0001, 3.)
TpLimits = (-.5, 0.5)
ShowErrors = True
ShowRMS = True

PlotFormat = [".pdf", ".png", ".svg"]
PlotFile = "Annecy_Tp_final"
PdfCatalog = True
if not ".pdf" in PlotFormat:
    error(" No pdfs generated. No catalog possible!")
    PdfCatalog = False
PdfCName = PlotFile


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


# Determine graphical parameter.
# print(plt.style.available)
plt.style.use("seaborn-paper")
mpl.rcParams["figure.dpi"] = 400
mpl.rcParams["axes.linewidth"] = 0.5
Fontsize = 10
Labelsize = Fontsize
Linewidth= 2
Markersize = 4
Grey = 0.7
Lcycle =Lcycle = (cycler("linestyle", ["-", "--", ":", "-."])
          * cycler("color", ["r", "g", "b", "y"]))
mpl.rcParams["figure.dpi"] = 400
mpl.rcParams["axes.linewidth"] = 0.5

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
        indx =np.argsort(Perxo)
        Tpxro = Tpxro[indx]
        Tpxio = Tpxio[indx]
        Perxo=Perxo[indx]
        cmpc = np.where((cal_cmp==cmp) & (cal_sit==s))
        Tpxrc = cal_rdat[cmpc]
        Tpxic = cal_idat[cmpc]
        Perxc  = cal_per[cmpc]
        indx =np.argsort(Perxc)
        Tpxrc = Tpxrc[indx]
        Tpxic = Tpxic[indx]
        Perxc=Perxc[indx]

        if ShowRMS:
            RnormTpxr, ResTpxr = utl.calc_resnorm(Tpxro, Tpxrc, Tpxe)
            nRMSTpxr, _ = utl.calc_rms(Tpxro, Tpxrc, 1.0/Tpxe)
            RnormTpxi, ResTpxi = utl.calc_resnorm(Tpxio, Tpxic, Tpxe)
            nRMSTpxi, _ = utl.calc_rms(Tpxio, Tpxic, 1.0/Tpxe)

        cmp ="TY"
        cmpo = np.where((obs_cmp==cmp) & (obs_sit==s))
        Tpyro = obs_rdat[cmpo]
        Tpyio = obs_idat[cmpo]
        Tpye = obs_err[cmpo]
        Peryo = obs_per[cmpo]
        indx =np.argsort(Peryo)
        Tpyro = Tpyro[indx]
        Tpyio = Tpyio[indx]
        Peryo=Peryo[indx]
        cmpc = np.where((cal_cmp==cmp) & (cal_sit==s))
        Tpyrc = cal_rdat[cmpc]
        Tpyic = cal_idat[cmpc]
        Peryc = cal_per[cmpc]
        indx =np.argsort(Peryc)
        Tpyrc = Tpyrc[indx]
        Tpyic = Tpyic[indx]
        Peryc=Peryc[indx]

        if ShowRMS:
            RnormTpyr, ResTpyr = utl.calc_resnorm(Tpyro, Tpyrc, Tpye)
            nRMSTpyr, _ = utl.calc_rms(Tpyro, Tpyrc, 1.0/Tpye)
            RnormTpyi, ResTpyi = utl.calc_resnorm(Tpyio, Tpyic, Tpye)
            nRMSTpyi, _ = utl.calc_rms(Tpyio, Tpyic, 1.0/Tpye)



        cm = 1/2.54  # centimeters in inches
        fig, axes = plt.subplots(1, 2, figsize = (16*cm, 7*cm), squeeze=False)

        fig.suptitle(r"Site: "+s
                     +"\nLat: "+str(site_lat)+"   Lon: "+str(site_lon)
                     +"\nUTMX: "+str(site_utmx)+"   UTMY: "+str(site_utmy)
                     +" (EPSG="+str(EPSG)+")  \nElev: "+ str(abs(site_elev))+" m",
                     ha="left", x=0.1,fontsize=Fontsize-1)

        axes[0,0].plot(Perxc, Tpxrc, color="r",linestyle="-", linewidth=Linewidth)
        axes[0,0].errorbar(Perxo,Tpxro, yerr=Tpxe,
                                linestyle="",
                                marker="o",
                                color="r",
                                linewidth=Linewidth,
                                markersize=Markersize)
        axes[0,0].plot(Perxc, Tpxic, color="b",linestyle="-", linewidth=Linewidth)
        axes[0,0].errorbar(Perxo,Tpxio, yerr=Tpxe,
                                linestyle="",
                                marker="o",
                                color="b",
                                linewidth=Linewidth,
                                markersize=Markersize)
        axes[0,0].set_xscale("log")
        axes[0,0].set_xlim(PerLimits)
        if TpLimits != ():
            axes[0,0].set_ylim(TpLimits)
        axes[0,0].legend(["real", "imag"])
        # axes[0,0].xaxis.set_ticklabels([])
        axes[0,0].tick_params(labelsize=Labelsize-1)
        axes[0,0].set_ylabel("Tpy", fontsize=Fontsize)
        axes[0,0].grid("major", "both", linestyle=":", linewidth=0.5)
        if ShowRMS:
            nRMSr = np.around(nRMSTpxr,1)
            nRMSi = np.around(nRMSTpxi,1)
            StrRMS = "nRMS = "+str(nRMSr)+" | "+str(nRMSi)
            axes[0,0].text(0.05, 0.05,StrRMS,
                               transform=axes[0,0].transAxes,
                               fontsize = Fontsize-2,
                               ha="left", va="bottom",
                               bbox={"pad": 2, "facecolor": "white", "edgecolor": "white" ,"alpha": 0.8} )



        axes[0,1].plot(Peryc, Tpyrc, color="r",linestyle="-", linewidth=Linewidth)
        axes[0,1].errorbar(Peryo,Tpyro, yerr=Tpye,
                                linestyle="",
                                marker="o",
                                color="r",
                                linewidth=Linewidth,
                                markersize=Markersize)
        axes[0,1].plot(Peryc, Tpyic, color="b",linestyle="-", linewidth=Linewidth)
        axes[0,1].errorbar(Peryc,Tpyio, yerr=Tpye,
                                linestyle="",
                                marker="o",
                                color="b",
                                linewidth=Linewidth,
                                markersize=Markersize)

        axes[0,1].set_xscale("log")
        axes[0,1].set_xlim(PerLimits)
        if TpLimits != ():
            axes[0,1].set_ylim(TpLimits)
        axes[0,1].legend(["real", "imag"])
        # axes[0,1].xaxis.set_ticklabels([])
        axes[0,1].tick_params(labelsize=Labelsize-1)
        axes[0,1].set_ylabel("Tpx", fontsize=Fontsize)
        axes[0,1].grid("major", "both", linestyle=":", linewidth=0.5)
        if ShowRMS:
            nRMSr = np.around(nRMSTpyr,1)
            nRMSi = np.around(nRMSTpyi,1)
            StrRMS = "nRMS = "+str(nRMSr)+" | "+str(nRMSi)
            axes[0,1].text(0.05, 0.05,StrRMS,
                               transform=axes[0,1].transAxes,
                               fontsize = Fontsize-2,
                               ha="left", va="bottom",
                               bbox={"pad": 2, "facecolor": "white", "edgecolor": "white" ,"alpha": 0.8} )



        fig.tight_layout()

        for F in PlotFormat:
            plt.savefig(PlotDir+PlotFile+"_"+s+F, dpi=400)


        plt.show()
        plt.close(fig)
    else:
        print("No Tipper for site "+s+"!")


if PdfCatalog:
    utl.make_pdf_catalog(PlotDir, PdfCName)

