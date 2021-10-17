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
print("Plot Impedance fit"+"\n"+"".join("Date " + now.strftime("%m/%d/%Y, %H:%M:%S")))
print("\n\n")

warnings.simplefilter(action="ignore", category=FutureWarning)

cm = 1/2.54  # centimeters to inches
WorkDir =  r"/home/vrath/work/MT_Data/Reunion/LydiaModel/"
PredFile = r"/home/vrath/work/MT_Data/Reunion/LydiaModel/reuf3_NLCG_020"
ObsvFile = r"/home/vrath/work/MT_Data/Reunion/LydiaModel/reuf2dat-net"


# WorkDir =  r"/home/vrath/work/MT/Annecy/ANN26/"
# PredFile = r"/home/vrath/work/MT/Annecy/ANN26/Ann26_ZoPT_200_Alpha02_NLCG_013"
# ObsvFile = r"/home/vrath/work/MT/Annecy/ANN26/Ann26_ZoPT"

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

PerLimits = (0.0003, 3000.)
ZLimitsXX = ()
ZLimitsXY = ()

ShowErrors = True
ShowRMS = True
PlotFull = False

EPSG = 0  #5644

if PlotFull:
    FigSize = (18*cm, 16*cm) #
else:
    FigSize = (16*cm, 10*cm)

PlotFormat = [".pdf", ".png",]
PlotFile = "Reunion_LydiaModel_Zoffd"

PdfCatalog = True
PdfCName = PlotFile+".pdf"
if not ".pdf" in PlotFormat:
    error(" No pdfs generated. No catalog possible!")
    PdfCatalog = False


"""
Determine graphical parameter.
+> print(plt.style.available)
"""

plt.style.use("seaborn-paper")
mpl.rcParams["figure.dpi"] = 400
mpl.rcParams["axes.linewidth"] = 0.5
mpl.rcParams["savefig.facecolor"] = "none"
Fontsize = 10
Labelsize = Fontsize
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
if FilesOnly==True:
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

    if PlotFull:
        cmp ="ZXX"
        cmpo = np.where((obs_cmp==cmp) & (obs_sit==s))
        Zxxro = np.abs(obs_rdat[cmpo])
        Zxxio = np.abs(obs_idat[cmpo])
        Zxxe = obs_err[cmpo]
        Perxxo = obs_per[cmpo]
        indx =np.argsort(Perxxo)
        Zxxro = Zxxro[indx]
        Zxxio = Zxxio[indx]
        Zxxe = Zxxe[indx]
        Perxxo = Perxxo[indx]
        cmpc = np.where((cal_cmp==cmp) & (cal_sit==s))
        Zxxrc = np.abs(cal_rdat[cmpc])
        Zxxic = np.abs(cal_idat[cmpc])
        Perxxc = cal_per[cmpc]
        indx =np.argsort(Perxxc)
        Zxxrc = Zxxrc[indx]
        Zxxic = Zxxic[indx]
        Perxxc = Perxxc[indx]

        if np.size(cmpo) > 0:
            siteRes = np.append(siteRes, (Zxxro-Zxxrc)/Zxxe)
            siteRes = np.append(siteRes, (Zxxio-Zxxic)/Zxxe)

            if ShowRMS:
                RnormZxxr, ResZxxr = utl.calc_resnorm(Zxxro, Zxxrc, Zxxe)
                nRMSZxxr, _ = utl.calc_rms(Zxxro, Zxxrc, 1.0/Zxxe)
                RnormZxxi, ResZxxi = utl.calc_resnorm(Zxxio, Zxxic, Zxxe)
                nRMSZxxi, _ = utl.calc_rms(Zxxio, Zxxic, 1.0/Zxxe)



    cmp ="ZXY"
    cmpo = np.where((obs_cmp==cmp) & (obs_sit==s))
    Zxyro = np.abs(obs_rdat[cmpo])
    Zxyio = np.abs(obs_idat[cmpo])
    Zxye = obs_err[cmpo]
    Perxyo = obs_per[cmpo]
    indx =np.argsort(Perxyo)
    Zxyro = Zxyro[indx]
    Zxyio = Zxyio[indx]
    Zxye = Zxye[indx]
    Perxyo = Perxyo[indx]
    cmpc = np.where((cal_cmp==cmp) & (cal_sit==s))
    Zxyrc = np.abs(cal_rdat[cmpc])
    Zxyic = np.abs(cal_idat[cmpc])
    Perxyc = cal_per[cmpc]
    indx =np.argsort(Perxyc)
    Zxyrc = Zxyrc[indx]
    Zxyic = Zxyic[indx]
    Perxyc = Perxyc[indx]

    if np.size(cmpo) > 0:
        siteRes = np.append(siteRes, (Zxyro-Zxyrc)/Zxye)
        siteRes = np.append(siteRes, (Zxyio-Zxyic)/Zxye)

        if ShowRMS:
            RnormZxyr, ResZxyr = utl.calc_resnorm(Zxyro, Zxyrc, Zxye)
            nRMSZxyr, _ = utl.calc_rms(Zxyro, Zxyrc, 1.0/Zxye)
            RnormZxyi, ResZxyi = utl.calc_resnorm(Zxyio, Zxyic, Zxye)
            nRMSZxyi, _ = utl.calc_rms(Zxyio, Zxyic, 1.0/Zxye)

    cmp ="ZYX"
    cmpo = np.where((obs_cmp==cmp) & (obs_sit==s))
    Zyxro = np.abs(obs_rdat[cmpo])
    Zyxio = np.abs(obs_idat[cmpo])
    Zyxe = obs_err[cmpo]
    Peryxo = obs_per[cmpo]
    indx =np.argsort(Peryxo)
    Zyxro = Zyxro[indx]
    Zyxio = Zyxio[indx]
    Zyxe = Zyxe[indx]
    Peryxo = Peryxo[indx]
    cmpc = np.where((cal_cmp==cmp) & (cal_sit==s))
    Zyxrc = np.abs(cal_rdat[cmpc])
    Zyxic = np.abs(cal_idat[cmpc])
    Peryxc = cal_per[cmpc]
    indx =np.argsort(Peryxc)
    Zyxrc = Zyxrc[indx]
    Zyxic = Zyxic[indx]
    Peryxc = Peryxc[indx]

    if np.size(cmpo) > 0:
        siteRes = np.append(siteRes, (Zyxro-Zyxrc)/Zyxe)
        siteRes = np.append(siteRes, (Zyxio-Zyxic)/Zyxe)

        if ShowRMS:
            RnormZyxr, ResZyxr = utl.calc_resnorm(Zyxro, Zyxrc, Zyxe)
            nRMSZyxr, _ = utl.calc_rms(Zyxro, Zyxrc, 1.0/Zyxe)
            RnormZyxi, ResZyxi = utl.calc_resnorm(Zyxio, Zyxic, Zyxe)
            nRMSZyxi, _ = utl.calc_rms(Zyxio, Zyxic, 1.0/Zyxe)



    if PlotFull:
        cmp ="ZYY"
        cmpo = np.where((obs_cmp==cmp) & (obs_sit==s))
        Zyyro = np.abs(obs_rdat[cmpo])
        Zyyio = np.abs(obs_idat[cmpo])
        Zyye = obs_err[cmpo]
        Peryyo = obs_per[cmpo]
        indx =np.argsort(Peryyo)
        Zyyro = Zyyro[indx]
        Zyyio = Zyyio[indx]
        Peryyo = Peryyo[indx]
        cmpc = np.where((cal_cmp==cmp) & (cal_sit==s))
        Zyyrc = np.abs(cal_rdat[cmpc])
        Zyyic = np.abs(cal_idat[cmpc])
        Peryyc = cal_per[cmpc]
        indx =np.argsort(Peryyc)
        Zyyrc = Zyyrc[indx]
        Zyyic = Zyyic[indx]
        Peryyc = Peryyc[indx]

        if np.size(cmpo) > 0:
            siteRes = np.append(siteRes, (Zyyro-Zyyrc)/Zyye)
            siteRes = np.append(siteRes, (Zyyio-Zyyic)/Zyye)

            if ShowRMS:
                RnormZyyr, ResZyyr = utl.calc_resnorm(Zyyro, Zyyrc, Zyye)
                nRMSZyyr, _ = utl.calc_rms(Zyyro, Zyyrc, 1.0/Zyye)
                RnormZyyi, ResZyyi = utl.calc_resnorm(Zyyio, Zyyic, Zyye)
                nRMSZyyi, _ = utl.calc_rms(Zyyio, Zyyic, 1.0/Zyye)


    sRes = np.asarray(siteRes)
    nD = np.size(sRes)
    siteRMS = np.sqrt(np.sum(np.power(sRes,2.))/float(nD))
    print("Site nRMS: "+str(siteRMS))
    # Ccprint(sRes)


    if PlotFull:

        fig, axes = plt.subplots(2,2, figsize = FigSize, subplot_kw=dict(box_aspect=1.),
                         sharex=False, sharey=False, constrained_layout=True)

        fig.suptitle(r"Site: "+s+"   nRMS: "+str(np.around(siteRMS,1))
                     +"\nLat: "+str(site_lat)+"   Lon: "+str(site_lon)
                     +"\nUTMX: "+str(site_utmx)+"   UTMY: "+str(site_utmy)
                     +" (EPSG="+str(EPSG)+")  \nElev: "+ str(abs(site_elev))+" m\n",
                     ha="left", x=0.1,fontsize=Fontsize-1)

#  ZXX

        if PlotPred:
            axes[0,0].plot(Perxxc, Zxxrc, color="r",linestyle="-", linewidth=Linewidth)

        if PlotObsv:
            if ShowErrors:
                axes[0,0].errorbar(Perxxo,Zxxro, yerr=Zxxe,
                                linestyle="",
                                marker="o",
                                color="r",
                                linewidth=Linewidth,
                                markersize=Markersize)
            else:
                axes[0,0].plot(Perxxo, Zxxro,
                               color="r",
                               linestyle="",
                               marker="o",
                               markersize=Markersize)

        if PlotPred:
            axes[0,0].plot(Perxxc, Zxxic, color="b",linestyle="-", linewidth=Linewidth)

        if PlotObsv:
            if ShowErrors:
                axes[0,0].errorbar(Perxxo,Zxxio, yerr=Zxxe,
                                linestyle="",
                                marker="o",
                                color="b",
                                linewidth=Linewidth,
                                markersize=Markersize)
            else:
                axes[0,0].plot(Perxxo, Zxxio,
                               color="b",
                               linestyle="",
                               marker="o",
                               markersize=Markersize)

        axes[0,0].set_xscale("log")
        axes[0,0].set_yscale("log")
        axes[0,0].set_xlim(PerLimits)
        if ZLimitsXX != ():
            axes[0,0].set_ylim(ZLimitsXX)
        axes[0,0].legend(["real", "imag"])

        axes[0,0].tick_params(labelsize=Labelsize-1)
        axes[0,0].set_ylabel("|ZXX|", fontsize=Fontsize)
        axes[0,0].grid("both", "both", linestyle="-", linewidth=0.5)
        if ShowRMS:
            nRMSr = np.around(nRMSZxxr,1)
            nRMSi = np.around(nRMSZxxi,1)
            StrRMS = "nRMS = "+str(nRMSr)+" | "+str(nRMSi)
            axes[0,0].text(0.05, 0.05,StrRMS,
                               transform=axes[0,1].transAxes,
                               fontsize = Fontsize-2,
                               ha="left", va="bottom",
                               bbox={"pad": 2, "facecolor": "white", "edgecolor": "white" ,"alpha": 0.8} )


#  ZXY
        if PlotPred:
           axes[0,1].plot(Perxyc, Zxyrc, color="r",linestyle="-", linewidth=Linewidth)

        if PlotObsv:
            if ShowErrors:
                axes[0,1].errorbar(Perxyo,Zxyro, yerr=Zxye,
                                linestyle="",
                                marker="o",
                                color="r",
                                linewidth=Linewidth,
                                markersize=Markersize)
            else:
                axes[0,1].plot(Perxyo,
                               Zxyro, color="r",
                               linestyle="",
                               marker="o",
                               markersize=Markersize)

        if PlotPred:
            axes[0,1].plot(Perxyc, Zxyic, color="b",linestyle="-", linewidth=Linewidth)
        if PlotObsv:
            if ShowErrors:
                axes[0,1].errorbar(Perxyo,Zxyio, yerr=Zxye,
                                linestyle="",
                                marker="o",
                                color="b",
                                linewidth=Linewidth,
                                markersize=Markersize)
            else:
                axes[0,1].plot(Perxyo, Zxyio,
                               color="b",
                               linestyle="",
                               marker="o",
                               markersize=Markersize)

        axes[0,1].set_xscale("log")
        axes[0,1].set_yscale("log")
        axes[0,1].set_xlim(PerLimits)
        if ZLimitsXY != ():
            axes[0,1].set_ylim(ZLimitsXY)
        axes[0,1].legend(["real", "imag"])
        # axes[0,1].xaxis.set_ticklabels([])
        axes[0,1].tick_params(labelsize=Labelsize-1)
        axes[0,1].set_ylabel("|ZXY|", fontsize=Fontsize)
        axes[0,1].grid("both", "both", linestyle="-", linewidth=0.5)
        if ShowRMS:
            nRMSr = np.around(nRMSZxyr,1)
            nRMSi = np.around(nRMSZxyi,1)
            StrRMS = "nRMS = "+str(nRMSr)+" | "+str(nRMSi)
            axes[0,1].text(0.05, 0.05,StrRMS,
                               transform=axes[0,1].transAxes,
                               fontsize = Fontsize-2,
                               ha="left", va="bottom",
                               bbox={"pad": 2, "facecolor": "white", "edgecolor": "white" ,"alpha": 0.8} )


#  ZYX
        if PlotPred:
           axes[1,0].plot(Peryxc, Zyxrc, color="r",linestyle="-", linewidth=Linewidth)

        if PlotObsv:
            if ShowErrors:
                axes[1,0].errorbar(Peryxo,Zyxro, yerr=Zyxe,
                                linestyle="",
                                marker="o",
                                color="r",
                                linewidth=Linewidth,
                                markersize=Markersize)
            else:
                axes[1,0].plot(Peryxo,
                               Zyxro, color="r",
                               linestyle="",
                               marker="o",
                               markersize=Markersize)

        if PlotPred:
            axes[1,0].plot(Peryxc, Zyxic, color="b",linestyle="-", linewidth=Linewidth)
        if PlotObsv:
            if ShowErrors:
                axes[1,0].errorbar(Peryxo,Zyxio, yerr=Zyxe,
                                linestyle="",
                                marker="o",
                                color="b",
                                linewidth=Linewidth,
                                markersize=Markersize)
            else:
                axes[1,0].plot(Peryxo, Zyxio,
                               color="b",
                               linestyle="",
                               marker="o",
                               markersize=Markersize)

        axes[1,0].set_xscale("log")
        axes[1,0].set_yscale("log")
        axes[1,0].set_xlim(PerLimits)
        if ZLimitsXY != ():
            axes[1,0].set_ylim(ZLimitsXY)
        axes[1,0].legend(["real", "imag"])
        # axes[1,0].xaxis.set_ticklabels([])
        axes[1,0].tick_params(labelsize=Labelsize-1)
        axes[1,0].set_ylabel("|ZYX|", fontsize=Fontsize)
        axes[1,0].grid("both", "both", linestyle="-", linewidth=0.5)
        if ShowRMS:
            nRMSr = np.around(nRMSZyxr,1)
            nRMSi = np.around(nRMSZyxi,1)
            StrRMS = "nRMS = "+str(nRMSr)+" | "+str(nRMSi)
            axes[1,0].text(0.05, 0.05,StrRMS,
                               transform=axes[1,0].transAxes,
                               fontsize = Fontsize-2,
                               ha="left", va="bottom",
                               bbox={"pad": 2, "facecolor": "white", "edgecolor": "white" ,"alpha": 0.8} )

#  ZYY

        if PlotPred:
            axes[1,1].plot(Peryyc, Zyyrc, color="r",linestyle="-", linewidth=Linewidth)

        if PlotObsv:
            if ShowErrors:
                axes[1,1].errorbar(Peryyo,Zyyro, yerr=Zyye,
                                linestyle="",
                                marker="o",
                                color="r",
                                linewidth=Linewidth,
                                markersize=Markersize)
            else:
                axes[1,1].plot(Peryyo, Zyyro,
                               color="r",
                               linestyle="",
                               marker="o",
                               markersize=Markersize)

        if PlotPred:
            axes[1,1].plot(Peryyc, Zyyic, color="b",linestyle="-", linewidth=Linewidth)

        if PlotObsv:
            if ShowErrors:
                axes[1,1].errorbar(Peryyo,Zyyio, yerr=Zyye,
                                linestyle="",
                                marker="o",
                                color="b",
                                linewidth=Linewidth,
                                markersize=Markersize)
            else:
                axes[1,1].plot(Peryyo, Zyyio,
                               color="b",
                               linestyle="",
                               marker="o",
                               markersize=Markersize)

        axes[1,1].set_xscale("log")
        axes[1,1].set_yscale("log")
        axes[1,1].set_xlim(PerLimits)
        if ZLimitsXX != ():
            axes[1,1].set_ylim(ZLimitsXX)
        axes[1,1].legend(["real", "imag"])
        # axes[1,1].xaxis.set_ticklabels([])
        axes[1,1].tick_params(labelsize=Labelsize-1)
        axes[1,1].set_ylabel("|ZYY|", fontsize=Fontsize)
        axes[1,1].grid("both", "both", linestyle="-", linewidth=0.5)
        if ShowRMS:
            nRMSr = np.around(nRMSZyyr,1)
            nRMSi = np.around(nRMSZyyi,1)
            StrRMS = "nRMS = "+str(nRMSr)+" | "+str(nRMSi)
            axes[1,1].text(0.05, 0.05,StrRMS,
                               transform=axes[0,1].transAxes,
                               fontsize = Fontsize-2,
                               ha="left", va="bottom",
                               bbox={"pad": 2, "facecolor": "white", "edgecolor": "white" ,"alpha": 0.8} )

    else:

        fig, axes = plt.subplots(1,2, figsize = FigSize, subplot_kw=dict(box_aspect=1.),
                         sharex=False, sharey=False, constrained_layout=True)


        fig.suptitle(r"Site: "+s+"   nRMS: "+str(np.around(siteRMS,1))
                     +"\nLat: "+str(site_lat)+"   Lon: "+str(site_lon)
                     +"\nUTMX: "+str(site_utmx)+"   UTMY: "+str(site_utmy)
                     +" (EPSG="+str(EPSG)+")  \nElev: "+ str(abs(site_elev))+" m\n",
                     ha="left", x=0.1,fontsize=Fontsize-1)



        if PlotPred:
            axes[0,].plot(Perxyc, Zxyrc, color="r",linestyle="-", linewidth=Linewidth)

        if PlotObsv:
            if ShowErrors:
                axes[0,].errorbar(Perxyo,Zxyro, yerr=Zxye,
                                linestyle="",
                                marker="o",
                                color="r",
                                linewidth=Linewidth,
                                markersize=Markersize)
            else:
                axes[0,].plot(Perxyo,
                               Zxyro, color="r",
                               linestyle="",
                               marker="o",
                               markersize=Markersize)

        if PlotPred:
            axes[0,].plot(Perxyc, Zxyic, color="b",linestyle="-", linewidth=Linewidth)

        if PlotObsv:
            if ShowErrors:
                axes[0,].errorbar(Perxyo,Zxyio, yerr=Zxye,
                                linestyle="",
                                marker="o",
                                color="b",
                                linewidth=Linewidth,
                                markersize=Markersize)
            else:
                axes[0,].plot(Perxyo, Zxyio,
                               color="b",
                               linestyle="",
                               marker="o",
                               markersize=Markersize)

        axes[0,].set_xscale("log")
        axes[0,].set_yscale("log")
        axes[0,].set_xlim(PerLimits)
        if ZLimitsXY != ():
            axes[0,].set_ylim(ZLimitsXY)
        axes[0,].legend(["real", "imag"])
        axes[0,].tick_params(labelsize=Labelsize-1)
        axes[0,].set_ylabel("|ZXY|", fontsize=Fontsize)
        axes[0,].grid("both", "both", linestyle="-", linewidth=0.5)
        if ShowRMS:
            nRMSr = np.around(nRMSZxyr,1)
            nRMSi = np.around(nRMSZxyi,1)
            StrRMS = "nRMS = "+str(nRMSr)+" | "+str(nRMSi)
            axes[0,].text(0.05, 0.05,StrRMS,
                               transform=axes[0,].transAxes,
                               fontsize = Fontsize-2,
                               ha="left", va="bottom",
                               bbox={"pad": 2, "facecolor": "white", "edgecolor": "white" ,"alpha": 0.8} )

        if PlotPred:
            axes[1,].plot(Peryxc, Zyxrc, color="r",linestyle="-", linewidth=Linewidth)

        if PlotObsv:
            if ShowErrors:
                axes[1,].errorbar(Peryxo,Zyxro, yerr=Zyxe,
                                linestyle="",
                                marker="o",
                                color="r",
                                linewidth=Linewidth,
                                markersize=Markersize)
            else:
                axes[1,].plot(Peryxo, Zyxro,
                               color="r",
                               linestyle="",
                               marker="o",
                               markersize=Markersize)

        if PlotPred:
            axes[1,].plot(Peryxc, Zyxic, color="b",linestyle="-", linewidth=Linewidth)

        if PlotObsv:
            if ShowErrors:
                axes[1,].errorbar(Peryxo,Zyxio, yerr=Zyxe,
                                linestyle="",
                                marker="o",
                                color="b",
                                linewidth=Linewidth,
                                markersize=Markersize)
            else:
                axes[1,].plot(Peryxo, Zyxio,
                               color="b",
                               linestyle="",
                               marker="o",
                               markersize=Markersize)

        axes[1,].set_xscale("log")
        axes[1,].set_yscale("log")
        axes[1,].set_xlim(PerLimits)
        if ZLimitsXY != ():
            axes[1,].set_ylim(ZLimitsXY)
        axes[1,].legend(["real", "imag"])
        axes[1,].tick_params(labelsize=Labelsize-1)
        axes[1,].set_ylabel("|ZYX|", fontsize=Fontsize)
        axes[1,].grid("both", "both", linestyle="-", linewidth=0.5)
        if ShowRMS:
            nRMSr = np.around(nRMSZyxr,1)
            nRMSi = np.around(nRMSZyxi,1)
            StrRMS = "nRMS = "+str(nRMSr)+" | "+str(nRMSi)
            axes[1,].text(0.05, 0.05,StrRMS,
                               transform=axes[1,].transAxes,
                               fontsize = Fontsize-2,
                               ha="left", va="bottom",
                               bbox={"pad": 2, "facecolor": "white", "edgecolor": "white" ,"alpha": 0.8} )


    # fig.tight_layout()

    for F in PlotFormat:
        plt.savefig(PlotDir+PlotFile+"_"+s+F, dpi=400)

    if PdfCatalog:
        pdf_list.append(PlotDir+PlotFile+"_"+s+".pdf")


    plt.show()
    plt.close(fig)


if PdfCatalog:
   utl.make_pdf_catalog(PlotDir, PdfList=pdf_list, FileName=PdfCName)
