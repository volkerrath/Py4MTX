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

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

PY4MT_ROOT = os.environ["PY4MT_ROOT"]
mypath = [PY4MT_ROOT+"/py4mt/modules/", PY4MT_ROOT+"/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)


import modem as mod
import util as utl
from version import versionstrg

Strng, _ = versionstrg()
now = datetime.now()
print("\n\n"+Strng)
print("Plot Phase Tensor comparison"+"\n"+"".join("Date " + now.strftime("%m/%d/%Y, %H:%M:%S")))
print("\n\n")

warnings.simplefilter(action="ignore", category=FutureWarning)

cm = 1/2.54  # centimeters in inches


WorkDir =  r"/home/vrath/work/MT_Data/Ubaye/Volker_rms_off/"
DataFiles = [r"Ub22_ZoffPT", r"Ub22_ZoffPT_02_NLCG_014"]
DataPlot  = [("r", "o"), ("b","-")]
DataRef = 0
PlotFile = "Ubaye22_ZoffPTPhTensor_1"
PlotDir = WorkDir + 'Plots/'

print(' Plots written to: %s' % PlotDir)
if not os.path.isdir(PlotDir):
    print(' File: %s does not exist, but will be created' % PlotDir)
    os.mkdir(PlotDir)
FilesOnly = False

PerLimits = (0.0001,10.)
PhTLimitsXX = (-5., 5.)
PhTLimitsXY = (-5., 5.)
ShowErrors = True
ShowRMS = True
EPSG = 0 #5015

# if PlotFull:
FigSize = (16*cm, 16*cm) #
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
plt.style.use("seaborn-paper")
mpl.rcParams["figure.dpi"] = 400
mpl.rcParams["axes.linewidth"] = 0.5
mpl.rcParams["savefig.facecolor"] = "none"
Fontsize = 10
Labelsize = Fontsize
Titlesize = Fontsize-1
Linewidth= 1
Markersize = 4
Grey = 0.7
Lcycle = (cycler("linestyle", ["-", "--", ":", "-."])
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


Site_list = []
for ff in np.arange(len(DataFiles)):
    FF = WorkDir+DataFiles[ff]
    Site, Comp, Data, Head = mod.read_data(FF+".dat")
    dat = Data[:, 6]
    err = Data[:, 7]
    per = Data[:, 0]
    cmp = Comp
    sit = Site
    lat = Data[:,1]
    lon = Data[:,2]
    x = Data[:,3]
    y = Data[:,4]
    z = Data[:,5]


    Sites = np.unique(Site)
    ss = -1
    for s in Sites:
        ss=ss+1
        print("Plotting site: "+s)
        site = (sit==s)
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

        siteVal = np.empty([0,0])
        siteErr = np.empty([0,0])
        sitePer = np.empty([0,0])
        siteCmp = np.empty([0,0])

        for comp in ["PTXX", "PTXY", "PTYX", "PTYY"]:
            cmpi = np.where((cmp==comp) & (sit==s))
            Val = dat[cmpi]
            Err = err[cmpi]
            Per = per[cmpi]
            Cmp = cmp[cmpi]
            indx =np.argsort(Per)
            Val = Val[indx]
            Err = Err[indx]
            Per = Per[indx]
            Cmp = Cmp[indx]
            siteVal = np.append(siteVal, Val)
            siteErr = np.append(siteErr, Err)
            sitePer = np.append(sitePer, Per)
            siteCmp = np.append(siteCmp, Cmp)


        if ss ==0:
            sVal = np.asarray(siteVal)
            sErr = np.asarray(siteErr)
            sPer = np.asarray(sitePer)
            sCmp = np.asarray(siteCmp)

        else:
            sVal = np.hstack((sVal, np.asarray(siteVal)))
            sErr = np.hstack((sErr, np.asarray(siteErr)))
            sPer = np.hstack((sPer, np.asarray(sitePer)))
            sCmp = np.hstack((sCmp, np.asarray(siteCmp)))

        print(len(sVal))

    Site_list.append([sVal, sErr, sPer, sCmp])

sValRef, sErrRef, sPerRef, sCmpRef = Site_list[DataRef]

Data_List = Site_list.copy()
[Data_List].append(0.*sValRef)

for ff in np.arange(len(DataFiles)):
    sVal, sErr, sPer, sCmp = Site_list[ff]
    sRes = (sVal - sValRef)/sErrRef
    Data_List.append([sVal, sErr, sPer, sCmp, sRes])




    # nD = np.size(sVal)
    # print(nD)


pdf_list = []

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

    fig.subplots_adjust(wspace = 0.4,top = 0.8 )

    for F in PlotFormat:
        plt.savefig(PlotDir+PlotFile+"_"+s+F, dpi=400)

    if PdfCatalog:
        pdf_list.append(PlotDir+PlotFile+"_"+s+".pdf")


    plt.show()
    plt.close(fig)


if PdfCatalog:
    utl.make_pdf_catalog(PlotDir, PdfList=pdf_list, FileName=PlotDir+PdfCName)
