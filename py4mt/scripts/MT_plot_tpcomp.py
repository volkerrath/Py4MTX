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
print("Plot Magnetic transfer function (tipper) fit"+"\n"+"".join("Date " + now.strftime("%m/%d/%Y, %H:%M:%S")))
print("\n\n")


warnings.simplefilter(action="ignore", category=FutureWarning)

cm = 1/2.54  # centimeters in inches
# WorkDir =  r"/home/vrath/work/MT_Data/Reunion/LydiaModel/"
# PredFile = r"/home/vrath/work/MT_Data/Reunion/LydiaModel/reuf3_NLCG_020"
# ObsvFile = r"/home/vrath/work/MT_Data/Reunion/LydiaModel/reuf2dat-net"

# WorkDir =  r"/home/vrath/work/MT_Data/Ubaye/UB19VR/"
# PredFile = r"/home/vrath/work/MT_Data/Ubaye/UB19VR/Ub19c_ZPT_02_NLCG_010"
# ObsvFile = r"/home/vrath/work/MT_Data/Ubaye/UB19VR/Ub19c_ZPT"

# WorkDir =  r"/home/vrath/work/MT_Data/Annecy/ANN26/"
# PredFile = r"/home/vrath/work/MT_Data/Annecy/ANN26/Ann26_ZoPT_200_Alpha02_NLCG_013"
# ObsvFile = r"/home/vrath/work/MT_Data/Annecy/ANN26/Ann26_ZoPT"

# WorkDir =  r"/home/vrath/work/MT_Data/Annecy/ANN26/"
# PredFile = r"/home/vrath/work/MT_Data/Annecy/ANN26/Ann26_ZoPT_200_Alpha04_NLCG_017"
# ObsvFile = r"/home/vrath/work/MT_Data/Annecy/ANN26/Ann26_ZoPT"

# WorkDir =  r"/home/vrath/Blake2016"
# PredFile = r""
# ObsvFile = r"/home/vrath/Blake2016/kil_edited"


WorkDir =  r"/home/vrath/work/MT_Data/Ubaye/Volker_RMS/"
# PredFile = r"/home/vrath/work/MT_Data/Ubaye/Volker_RMS/from_Ub22_ZPT_100"
PredFile = r"/home/vrath/work/MT_Data/Ubaye/Volker_RMS/Ub22_ZofPT_02_NLCG_014"
ObsvFile = r"/home/vrath/work/MT_Data/Ubaye/Volker_RMS/Ub22_ZPT"
PlotFile = "Ubaye_Tipper"


PlotDir = WorkDir + 'Plots/'

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

PerLimits = (0.0001,10.)
TpLimits = (-1, 1)
ShowErrors = True
ShowRMS = True
if not PlotPred:
    ShowRMS = False
EPSG = 0 # 5015

FigSize = (16*cm, 8*cm)
PlotFormat = [".pdf", ".png",]
PdfCatalog = True
PdfCName = PlotFile+".pdf"
if not ".pdf" in PlotFormat:
    error(" No pdfs generated. No catalog possible!")
    PdfCatalog = False

"""
# Determine graphical parameter.
# print(plt.style.available)
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

if PlotPred:
    FF = PredFile
    SiteCal, CompCal, DataCal, HeadCal = mod.read_data(FF+".dat")
    cal_rdat = DataCal[:, 6]
    cal_idat = DataCal[:, 7]
    cal_per = DataCal[:, 0]
    cal_cmp = CompCal
    cal_sit = SiteCal



Sites = np.unique(SiteObs)

pdf_list = []
for s in Sites:
    print("Plotting site: "+s)
    site = (obs_sit==s)
    test = ((obs_cmp=="TX") | (obs_cmp=="TY")) & (obs_sit==s)

    if np.any(test) == True:
        site_lon = lon[site][0]
        site_lat = lat[site][0]

        if EPSG==0:
            site_utmx = x[site][0]
            site_utmy = y[site][0]
        else:
            site_utmx, site_utmy = utl.proj_latlon_to_utm(site_lat, site_lon,
                                                          utm_zone=EPSG)

        site_utmx = int(np.round(site_utmx))
        site_utmy = int(np.round(site_utmy))


        site_elev = z[site][0]

        siteRes = np.empty([0,0])

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

        if PlotPred:
            cmpc = np.where((cal_cmp==cmp) & (cal_sit==s))
            Tpxrc = cal_rdat[cmpc]
            Tpxic = cal_idat[cmpc]
            Perxc  = cal_per[cmpc]
            indx =np.argsort(Perxc)
            Tpxrc = Tpxrc[indx]
            Tpxic = Tpxic[indx]
            Perxc=Perxc[indx]
            if np.size(cmpo) > 0:
                siteRes = np.append(siteRes, (Tpxro-Tpxrc)/Tpxe)
                siteRes = np.append(siteRes, (Tpxio-Tpxic)/Tpxe)

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
        if PlotPred:
            cmpc = np.where((cal_cmp==cmp) & (cal_sit==s))
            Tpyrc = cal_rdat[cmpc]
            Tpyic = cal_idat[cmpc]
            Peryc = cal_per[cmpc]
            indx =np.argsort(Peryc)
            Tpyrc = Tpyrc[indx]
            Tpyic = Tpyic[indx]
            Peryc=Peryc[indx]

            if np.size(cmpo) > 0:
                siteRes = np.append(siteRes, (Tpyro-Tpyrc)/Tpye)
                siteRes = np.append(siteRes, (Tpyio-Tpyic)/Tpye)

                if ShowRMS:
                    RnormTpyr, ResTpyr = utl.calc_resnorm(Tpyro, Tpyrc, Tpye)
                    nRMSTpyr, _ = utl.calc_rms(Tpyro, Tpyrc, 1.0/Tpye)
                    RnormTpyi, ResTpyi = utl.calc_resnorm(Tpyio, Tpyic, Tpye)
                    nRMSTpyi, _ = utl.calc_rms(Tpyio, Tpyic, 1.0/Tpye)


        sRes = np.asarray(siteRes)
        nD = np.size(sRes)

        if PlotPred:
            siteRMS = np.sqrt(np.sum(np.power(sRes,2.))/float(nD))
            print("Site nRMS: "+str(siteRMS))



        fig, axes = plt.subplots(1,2, figsize = FigSize, subplot_kw=dict(box_aspect=1.),
                         sharex=False, sharey=False, constrained_layout=True)

        if PlotPred:
            rmsstrng ="   nRMS: "+str(np.around(siteRMS,1))
        else:
            rmsstrng = ""

        fig.suptitle(r"Site: "+s+rmsstrng
                     +"\nLat: "+str(site_lat)+"   Lon: "+str(site_lon)
                     +"\nX: "+str(site_utmx)+"   Y: "+str(site_utmy)
                     +" (EPSG="+str(EPSG)+")  \nElev: "+ str(abs(site_elev))+" m\n",
                     ha="left", x=0.1,fontsize=Titlesize)


        if PlotPred:
            axes[0,].plot(Perxc, Tpxrc, color="r",linestyle="-", linewidth=Linewidth)

        if PlotObsv:
            axes[0,].errorbar(Perxo,Tpxro, yerr=Tpxe,
                            linestyle="",
                            marker="o",
                            color="r",
                            linewidth=Linewidth,
                            markersize=Markersize)
        if PlotPred:
            axes[0,].plot(Perxc, Tpxic, color="b",linestyle="-", linewidth=Linewidth)

        if PlotObsv:
            axes[0,].errorbar(Perxo,Tpxio, yerr=Tpxe,
                            linestyle="",
                            marker="o",
                            color="b",
                            linewidth=Linewidth,
                            markersize=Markersize)
        axes[0,].set_xscale("log")
        axes[0,].set_xlim(PerLimits)
        if TpLimits != ():
            axes[0,].set_ylim(TpLimits)
        axes[0,].legend(["real", "imag"])
        # axes[0,].xaxis.set_ticklabels([])
        axes[0,].tick_params(labelsize=Labelsize-1)
        axes[0,].set_ylabel("Tpy", fontsize=Fontsize)
        axes[0,].grid("both", "both", linestyle=":", linewidth=0.5)
        if ShowRMS:
            nRMSr = np.around(nRMSTpxr,1)
            nRMSi = np.around(nRMSTpxi,1)
            StrRMS = "nRMS = "+str(nRMSr)+" | "+str(nRMSi)
            axes[0,].text(0.05, 0.05,StrRMS,
                            transform=axes[0,].transAxes,
                            fontsize = Fontsize-2,
                            ha="left", va="bottom",
                            bbox={"pad": 2, "facecolor": "white", "edgecolor": "white" ,"alpha": 0.8} )


        if PlotPred:
            axes[1,].plot(Peryc, Tpyrc, color="r",linestyle="-", linewidth=Linewidth)

        if PlotObsv:
            axes[1,].errorbar(Peryo,Tpyro, yerr=Tpye,
                            linestyle="",
                            marker="o",
                            color="r",
                            linewidth=Linewidth,
                            markersize=Markersize)
        if PlotPred:
            axes[1,].plot(Peryc, Tpyic, color="b",linestyle="-", linewidth=Linewidth)

        if PlotObsv:
            axes[1,].errorbar(Peryo,Tpyio, yerr=Tpye,
                            linestyle="",
                            marker="o",
                            color="b",
                            linewidth=Linewidth,
                            markersize=Markersize)

        axes[1,].set_xscale("log")
        axes[1,].set_xlim(PerLimits)
        if TpLimits != ():
            axes[1,].set_ylim(TpLimits)
        axes[1,].legend(["real", "imag"])
        # axes[1,].xaxis.set_ticklabels([])
        axes[1,].tick_params(labelsize=Labelsize-1)
        axes[1,].set_ylabel("Tpx", fontsize=Fontsize)
        axes[1,].grid("both", "both", linestyle=":", linewidth=0.5)
        if ShowRMS:
            nRMSr = np.around(nRMSTpyr,1)
            nRMSi = np.around(nRMSTpyi,1)
            StrRMS = "nRMS = "+str(nRMSr)+" | "+str(nRMSi)
            axes[1,].text(0.05, 0.05,StrRMS,
                               transform=axes[1,].transAxes,
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
    else:
        print("No Tipper for site "+s+"!")

if PdfCatalog:
    utl.make_pdf_catalog(PlotDir, PdfList=pdf_list, FileName=PlotDir+PdfCName)
