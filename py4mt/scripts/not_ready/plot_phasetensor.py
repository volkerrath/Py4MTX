#!/usr/bin/env python3
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
# ---

import os
import sys
import warnings
import time

from sys import exit as error
from datetime import datetime
import inspect

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]
mypath = [PY4MTX_ROOT+"/py4mt/modules/", PY4MTX_ROOT+"/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)


import modem as mod
import util as utl
from version import versionstrg

PY4MTX_DATA = os.environ["PY4MTX_DATA"]

version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=inspect.getfile(inspect.currentframe()), out=False)
print(titstrng+"\n\n")

cm = 1./2.54  # centimeters to inches

# WorkDir =  r"/home/vrath/work/MT_Data/Reunion/LydiaModel/"
# PredFile = r"/home/vrath/work/MT_Data/Reunion/LydiaModel/reuf3_NLCG_020"
# ObsvFile = r"/home/vrath/work/MT_Data/Reunion/LydiaModel/reuf2dat-net.dat"


# WorkDir =  r"/home/vrath/work/MT_Data/Ubaye/UB19VR/"
# PredFile = r"/home/vrath/work/MT_Data/Ubaye/UB19VR/Ub19c_ZPT_02_NLCG_010"
# ObsvFile = r"/home/vrath/work/MT_Data/Ubaye/UB19VR/Ub19c_ZPT"

WorkDir = PY4MTX_ROOT+"/work/modem/"
PredFile = WorkDir+"Ann26_ZoPT_200_Alpha02_NLCG_013"
ObsvFile = WorkDir+"Ann26_ZoPT"
PlotFile = "ANN26_PhaseTensor"


PlotDir = WorkDir+"/plots/"

print(' Plots written to: %s' % PlotDir)
if not os.path.isdir(PlotDir):
    print(' File: %s does not exist, but will be created' % PlotDir)
    os.mkdir(PlotDir)

FilesOnly = False
PlotPred = True
if PredFile == "":
    PlotPred = False
PlotObsv = True
if ObsvFile == "":
    PlotObsv = False

PerLimits = (0.0001,100.)
PhTLimitsXX = (-5., 5.)
PhTLimitsXY = (-5., 5.)
ShowErrors = True
ShowRMS = True
EPSG = 0 #5015

# if PlotFull:
FigSize = (16*cm, 12*cm) #
# else:
#     FigSize = (16*cm, 10*cm) #  NoDiag


PlotFormat = [".pdf", ".png",]


PdfCatalog = True
PdfCName = PlotFile+".pdf"
if not ".pdf" in PlotFormat:
    error(" No pdfs generated. No catalog possible!")
    PdfCatalog = False

"""
Determine graphical parameter.
print(plt.style.available)
"""
plt.style.use("seaborn-v0_8-paper")
mpl.rcParams["figure.dpi"] = 600
mpl.rcParams["axes.linewidth"] = 0.5
mpl.rcParams["savefig.facecolor"] = "none"
Fontsize = 10
Labelsize = Fontsize
Titlesize = Fontsize-1
Linewidth= 1
Markersize = 4
Grey = 0.7
Lcycle =Lcycle = (cycler("linestyle", ["-", "--", ":", "-."])
          * cycler("color", ["r", "g", "b", "y"]))

"""
For just plotting to files, choose the cairo backend (eps, pdf, ,png, jpg...).
If you need to see the plots directly (plots window, or jupyter), simply
comment out the following line. In this case matplotlib may run into
memory problems ager a few hundreds of high-resolution plots.
Find other backends by entering %matplotlib -l
"""
if FilesOnly:
    mpl.use("cairo")


start = time.time()

FF = ObsvFile
SiteObs, CompObs, DataObs, HeadObs = mod.read_data(FF)
obs_dat = DataObs[:, 6]
obs_err = DataObs[:, 7]
obs_per = DataObs[:, 0]
obs_cmp = CompObs
obs_sit = SiteObs
lat = DataObs[:,1]
lon = DataObs[:,2]
x = DataObs[:,3]
y = DataObs[:,4]
z = DataObs[:,5]

FF = PredFile
SiteCal, CompCal, DataCal, HeadCal = mod.read_data(FF)
cal_dat= DataCal[:, 6]
cal_per= DataCal[:, 0]
cal_cmp= CompCal
cal_sit = SiteCal


Sites = np.unique(SiteObs)

pdf_list = []
for s in Sites:
    print("Plotting site: "+s)
    site = (obs_sit==s)
    site_lon = lon[site][0]
    site_lat = lat[site][0]

    site_lon = lon[site][0]
    site_lat = lat[site][0]

    if EPSG==0:
        site_utmx = x[site][0]
        site_utmy = y[site][0]
    else:
        site_utmx, site_utmy = utl.project_latlon_to_utm(site_lat, site_lon,
                                                      utm_zone=EPSG)

    site_utmx = int(np.round(site_utmx))
    site_utmy = int(np.round(site_utmy))

    site_elev = z[site][0]

    siteRes = np.empty([0,0])

    cmp ="PTXX"
    cmpo = np.where((obs_cmp==cmp) & (obs_sit==s))
    PhTxxo = obs_dat[cmpo]
    PhTxxe = obs_err[cmpo]
    Perxxo = obs_per[cmpo]
    indx =np.argsort(Perxxo)
    PhTxxo = PhTxxo[indx]
    PhTxxe = PhTxxe[indx]
    Perxxo = Perxxo[indx]
    cmpc = np.where((cal_cmp==cmp) & (cal_sit==s))
    PhTxxc = cal_dat[cmpc]
    Perxxc = cal_per[cmpc]
    indx =np.argsort(Perxxc)
    PhTxxc = PhTxxc[indx]
    Perxxc = Perxxc[indx]

    if np.size(cmpo) > 0:
        siteRes = np.append(siteRes, (PhTxxo-PhTxxc)/PhTxxe)
        if ShowRMS:
            RnormPhTxx, ResPhTxx = utl.calc_resnorm(PhTxxo, PhTxxc, PhTxxe)
            nRMSPhTxx, _ = utl.calc_rms(PhTxxo, PhTxxc, 1.0/PhTxxe)


    cmp ="PTXY"
    cmpo = np.where((obs_cmp==cmp) & (obs_sit==s))
    PhTxyo = obs_dat[cmpo]
    PhTxye = obs_err[cmpo]
    Perxyo = obs_per[cmpo]
    indx =np.argsort(Perxyo)
    PhTxyo = PhTxyo[indx]
    PhTxye = PhTxye[indx]
    Perxyo = Perxyo[indx]
    cmpc = np.where((cal_cmp==cmp) & (cal_sit==s))
    PhTxyc = cal_dat[cmpc]
    Perxyc = cal_per[cmpc]
    indx =np.argsort(Perxyc)
    PhTxyc = PhTxyc[indx]
    Perxyc = Perxyc[indx]
    if np.size(cmpo) > 0:
        siteRes = np.append(siteRes, (PhTxyo-PhTxyc)/PhTxye)
        if ShowRMS:
            RnormPhTxy, ResPhTxy = utl.calc_resnorm(PhTxyo, PhTxyc, PhTxye)
            nRMSPhTxy, _ = utl.calc_rms(PhTxyo, PhTxyc, 1.0/PhTxye)

    cmp ="PTYX"
    cmpo = np.where((obs_cmp==cmp) & (obs_sit==s))
    PhTyxo = obs_dat[cmpo]
    PhTyxe = obs_err[cmpo]
    Peryxo = obs_per[cmpo]
    indx =np.argsort(Peryxo)
    PhTyxo = PhTyxo[indx]
    PhTyxe = PhTyxe[indx]
    Peryxo = Peryxo[indx]
    cmpc = np.where((cal_cmp==cmp) & (cal_sit==s))
    PhTyxc = cal_dat[cmpc]
    Peryxc = cal_per[cmpc]
    indx =np.argsort(Peryxc)
    PhTyxc = PhTyxc[indx]
    Peryxc = Peryxc[indx]
    if np.size(cmpo) > 0:
        siteRes = np.append(siteRes, (PhTyxo-PhTyxc)/PhTyxe)
        if ShowRMS:
            RnormPhTyx, ResPhTyx = utl.calc_resnorm(PhTyxo, PhTxyc, PhTyxe)
            nRMSPhTyx, _ = utl.calc_rms(PhTyxo, PhTyxc, 1.0/PhTyxe)

    cmp ="PTYY"
    cmpo = np.where((obs_cmp==cmp) & (obs_sit==s))
    PhTyyo = obs_dat[cmpo]
    PhTyye = obs_err[cmpo]
    Peryyo = obs_per[cmpo]
    indx =np.argsort(Peryyo)
    PhTyyo = PhTyyo[indx]
    PhTyye = PhTyye[indx]
    Peryyo = Peryyo[indx]
    cmpc = np.where((cal_cmp==cmp) & (cal_sit==s))
    PhTyyc = cal_dat[cmpc]
    Peryyc = cal_per[cmpc]
    indx =np.argsort(Peryyc)
    PhTyyc = PhTyyc[indx]
    Peryyc = Peryyc[indx]
    if np.size(cmpo) > 0:
        siteRes = np.append(siteRes, (PhTyyo-PhTyyc)/PhTyye)
        if ShowRMS:
            RnormPhTyy, ResPhTyy = utl.calc_resnorm(PhTyyo, PhTyyc, PhTyye)
            nRMSPhTyy, _ = utl.calc_rms(PhTyyo, PhTyyc, 1.0/PhTyye)




    sRes = np.asarray(siteRes)
    nD = np.size(sRes)
    siteRMS = np.sqrt(np.sum(np.power(sRes,2.))/(float(nD)-1))
    print("Site nRMS: "+str(siteRMS))
    # Ccprint(sRes)




    fig, axes = plt.subplots(2,2, figsize = FigSize, subplot_kw=dict(box_aspect=1.),
                     sharex=False, sharey=False, constrained_layout=True)

    fig.suptitle(r"Site: "+s+"   nRMS: "+str(np.around(siteRMS,1))
                     +"\nLat: "+str(site_lat)+"   Lon: "+str(site_lon)
                     +"\nX: "+str(site_utmx)+"   Y: "+str(site_utmy)
                     +" (EPSG="+str(EPSG)+")  \nElev: "+ str(abs(site_elev))+" m\n",
                     ha="left", x=0.1,fontsize=Titlesize)

    if PlotPred:
        axes[0,0].plot(Perxxc, PhTxxc, "-r", linewidth =Linewidth)

    if PlotObsv:
        if ShowErrors:
            axes[0,0].errorbar(Perxxo,PhTxxo, yerr=PhTxxe,
                                    linestyle="",
                                    marker="o",
                                    color="b",
                                    linewidth=Linewidth,
                                    markersize=Markersize)
        else:
            axes[0,0].plot(Perxxo, PhTxxo,
                           color="b",
                           marker="o",
                           linestyle="",
                           linewidth =Linewidth,
                           markersize=Markersize)


    axes[0,0].set_xscale("log")
    axes[0,0].set_xlim(PerLimits)
    if PhTLimitsXX != ():
        axes[0,0].set_ylim(PhTLimitsXX)
    axes[0,0].legend(["predicted", "observed"])

    axes[0,0].tick_params(labelsize=Labelsize-1)
    axes[0,0].set_ylabel("PhTXX", fontsize=Fontsize)
    axes[0,0].grid("both", "both", linestyle=":", linewidth=0.5)
    if ShowRMS:
        nRMSr = np.around(nRMSPhTxx,1)
        StrRMS = "nRMS = "+str(nRMSr)
        axes[0,0].text(0.05, 0.05,StrRMS,
                           transform=axes[0,0].transAxes,
                           fontsize = Fontsize-2,
                           ha="left", va="bottom",
                           bbox={"pad": 2, "facecolor": "white", "edgecolor": "white" ,"alpha": 0.8} )

    if PlotPred:
        axes[0,1].plot(Perxyc, PhTxyc, "-r", linewidth =Linewidth)

    if PlotObsv:
        if ShowErrors:
            axes[0,1].errorbar(Perxyo,PhTxyo, yerr=PhTxye,
                        linestyle="",
                        marker="o",
                        color="b",
                        linewidth=Linewidth,
                        markersize=Markersize)

            axes[0,1].errorbar(Perxyo,PhTxyo, yerr=PhTxye,
                            linestyle="",
                            marker="o",
                            color="b",
                            linewidth=Linewidth,
                            markersize=Markersize)
        else:
            axes[0,1].plot(Perxyo, PhTxyo,
                           color="b",
                           marker="o",
                           linestyle="",
                           linewidth =Linewidth,
                           markersize=Markersize)
    axes[0,1].set_xscale("log")
    axes[0,1].set_xlim(PerLimits)
    if PhTLimitsXY != ():
        axes[0,1].set_ylim(PhTLimitsXY)
    axes[0,1].legend(["predicted", "observed"])
    axes[0,1].tick_params(labelsize=Labelsize-1)
    axes[0,1].set_ylabel("PhTXY", fontsize=Fontsize)
    # axes[0,1].xaxis.set_ticklabels([])
    axes[0,1].tick_params(bottom="off", labelbottom="off")
    axes[0,1].grid("both", "both", linestyle=":", linewidth=0.5)
    if ShowRMS:
        nRMSr = np.around(nRMSPhTxy,1)
        StrRMS = "nRMS = "+str(nRMSr)
        axes[0,1].text(0.05, 0.05,StrRMS,
                           transform=axes[0,1].transAxes,
                           fontsize = Fontsize-2,
                           ha="left", va="bottom",
                           bbox={"pad": 2, "facecolor": "white", "edgecolor": "white" ,"alpha": 0.8} )


    if PlotPred:
        axes[1,0].plot(Peryxc, PhTyxc, "-r", linewidth =Linewidth)

    if PlotObsv:
        if ShowErrors:
            axes[1,0].errorbar(Peryxo,PhTyxo, yerr=PhTyxe,
                                    linestyle="",
                                    marker="o",
                                    color="b",
                                    linewidth=Linewidth,
                                    markersize=Markersize)
        else:
            axes[1,0].plot(Peryxo, PhTyxo,
                           color="b",
                           marker="o",
                           linestyle="",
                           linewidth =Linewidth,
                           markersize=Markersize)

    axes[1,0].set_xscale("log")
    axes[1,0].set_xlim(PerLimits)
    if PhTLimitsXY != ():
        axes[1,0].set_ylim(PhTLimitsXY)
    axes[1,0].legend(["predicted", "observed"])
    axes[1,0].tick_params(labelsize=Labelsize-1)
    axes[1,0].set_xlabel("Period (s)", fontsize=Fontsize)
    axes[1,0].set_ylabel("PhTYX", fontsize=Fontsize)
    axes[1,0].grid("both", "both", linestyle=":", linewidth=0.5)
    if ShowRMS:
        nRMSr = np.around(nRMSPhTyx,1)
        StrRMS = "nRMS = "+str(nRMSr)
        axes[1,0].text(0.05, 0.05,StrRMS,
                           transform=axes[1,0].transAxes,
                           fontsize = Fontsize-2,
                           ha="left", va="bottom",
                           bbox={"pad": 2, "facecolor": "white", "edgecolor": "white" ,"alpha": 0.8} )

    if PlotPred:
        axes[1,1].plot(Peryyc, PhTyyc, "-r", linewidth =Linewidth)

    if PlotObsv:
        if ShowErrors:
           axes[1,1].errorbar(Peryyo,PhTyyo, yerr=PhTyye,
                                    linestyle="",
                                    marker="o",
                                    color="b",
                                    linewidth=Linewidth,
                                    markersize=Markersize)
        else:
            axes[1,1].plot(Peryyo, PhTyyo,
                           color="b",
                           marker="o",
                           linestyle="",
                           linewidth =Linewidth,
                           markersize=Markersize)

    axes[1,1].set_xscale("log")
    axes[1,1].set_xlim(PerLimits)
    if PhTLimitsXX != ():
        axes[1,1].set_ylim(PhTLimitsXX)
    axes[1,1].legend(["predicted", "observed"])
    axes[1,1].tick_params(labelsize=Labelsize-1)
    axes[1,1].set_xlabel("Period (s)", fontsize=Fontsize)
    axes[1,1].set_ylabel("PhTYY", fontsize=Fontsize)
    axes[1,1].grid("both", "both", linestyle=":", linewidth=0.5)
    if ShowRMS:
        nRMSr = np.around(nRMSPhTyy,1)
        StrRMS = "nRMS = "+str(nRMSr)
        axes[1,1].text(0.05, 0.05,StrRMS,
                           transform=axes[1,1].transAxes,
                           fontsize = Fontsize-2,
                           ha="left", va="bottom",
                           bbox={"pad": 2, "facecolor": "white", "edgecolor": "white" ,"alpha": 0.8} )

    fig.subplots_adjust(wspace = 0.1,top = 0.8 )

    for F in PlotFormat:
        plt.savefig(PlotDir+PlotFile+"_"+s+F, dpi=400)

    if PdfCatalog:
        pdf_list.append(PlotDir+PlotFile+"_"+s+".pdf")


    plt.show()
    plt.close(fig)


if PdfCatalog:
    utl.make_pdf_catalog(PlotDir, PdfList=pdf_list, FileName=PlotDir+PdfCName)
