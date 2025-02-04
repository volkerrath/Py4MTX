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
from matplotlib.backends.backend_pdf import PdfPages


from cycler import cycler

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [PY4MTX_ROOT+"/py4mt/modules/", PY4MTX_ROOT+"/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)

import util as utl
from version import versionstrg
import modem as mod
import femtic as fem

version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng+"\n\n")

WorkDir = "/home/vrath/Py4MTX/work/Misti_results/"
DatFile = WorkDir+"MISTI_Results_rhophas.txt"
SitFile = WorkDir+"Sitelist_femtic.txt"
PlotDir = WorkDir+"/plots/"
PlotFile = "Mist01"



print(' Plots written to: %s' % PlotDir)
if not os.path.isdir(PlotDir):
    print(' File: %s does not exist, but will be created' % PlotDir)
    os.mkdir(PlotDir)



PlotPred = True
PlotObsv = True
PlotFull = True


PerLimits = [0.0001,1000.]
RhoLimits = []
PhsLimitsXY = []


ShowErrors = True
ShowRMS = True
if not PlotPred:
    ShowRMS = False

EPSG = 32719

PlotFormat = [".pdf", ".png", ]
PDFCatalog = True
PDFCatalogName = PlotFile+".pdf"
if not ".pdf" in PlotFormat:
    error(" No pdfs generated. No catalog possible!")
    PDFCatalog = False
else:
    pdf_list = []
    catalog = PdfPages(PDFCatalogName)
FilesOnly = False

"""
Determine graphical parameter.
+> print(plt.style.available)
"""

plt.style.use("seaborn-v0_8-paper")
mpl.rcParams["figure.dpi"] = 600
mpl.rcParams["axes.linewidth"] = 0.5
mpl.rcParams["savefig.facecolor"] = "none"

cm = 1./2.54  # centimeters to inches
FigSize = (16*cm, 18*cm)
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
if FilesOnly==True:
    mpl.use("cairo")



data_dict = fem.get_femtic_data(DatFile,SitFile, data_type="rhophas")
sites = data_dict["sites"]
lat = np.float64(data_dict["lat"])  #.reshape(-1,1)
lon = np.float64(data_dict["lon"])  #.reshape(-1,1)
elv = np.float64(data_dict["elv"])  #.reshape(-1,1)
nam = data_dict["nam"]# .reshape(-1,1)



for s in sites:
    print("Plotting site: "+str(s))
    print("Site name is ", nam[s][0])
    site_lon = lon[s] #[site_index]
    site_lat = lat[s] #a = [site_index]
    site_elev = elv[s]
    site_utmx, site_utmy = utl.proj_latlon_to_utm(site_lat, site_lon, utm_zone=EPSG)
    site_utmx = int(np.round(site_utmx))
    site_utmy = int(np.round(site_utmy))
    print(s, "at", site_lat[0], site_lon[0], site_elev[0])
    site_nam = nam[s][0]
    site_index = np.where(data_dict["num"] ==s)
    print(site_index)

    site_obs = data_dict["obs"][site_index]
    site_err = data_dict["err"][site_index]

    if PlotPred:
        site_cal = data_dict["cal"][site_index]
        site_res = np.ones_like(site_cal)

    site_per =  1./data_dict["frq"][site_index]
    nsite = np.shape(site_per)[0]


    nn = 8
    site_res = np.empty((1,nn))
    rhos = [ix for ix in range(nn) if ix % 2 == 0]
    phass = [ix for ix in range(nn) if ix % 2 == 1]
    obs_rho = site_obs[:,rhos]
    obs_phas = site_obs[:,phass]
    err_rho = site_err[:,rhos]
    err_phas = site_err[:,phass]
    if PlotPred:
        cal_rho = site_cal[:,rhos]
        cal_phas = site_cal[:,phass]

    if PlotFull:
        cmp ="ZXX"        
        zxx_obs_rho = np.abs(obs_rho[:, 0])
        zxx_obs_phas = np.abs(obs_phas[:, 0])
        zxx_err_rho = err_rho[:,0]
        zxx_err_phas = err_phas[:,0]
        if PlotPred:
            zxx_cal_rho = np.abs(obs_rho[:, 0])
            zxx_cal_phas = np.abs(obs_phas[:, 0])
            site_res = np.append(site_res, (zxx_obs_rho-zxx_cal_rho)/zxx_err_rho)
            site_res = np.append(site_res, (zxx_obs_phas-zxx_cal_phas)/zxx_err_phas)  
            if ShowRMS:
                nrmszxx_rho, _ = utl.calc_rms(zxx_obs_rho, zxx_cal_rho, 1.0/zxx_err_rho)
                nrmszxx_phas, _ = utl.calc_rms(zxx_obs_phas, zxx_cal_phas, 1.0/zxx_err_phas)


    cmp ="ZXY"
    zxy_obs_rho = np.abs(obs_rho[:, 1])
    zxy_obs_phas = np.abs(obs_phas[:, 1])
    zxy_err_rho = err_rho[:,1]
    zxy_err_phas = err_phas[:,1]
    if PlotPred:
        zxy_cal_rho = np.abs(obs_rho[:, 1])
        zxy_cal_phas = np.abs(obs_phas[:, 1])
        site_res = np.append(site_res, (zxy_obs_rho-zxy_cal_rho)/zxy_err_rho)
        site_res = np.append(site_res, (zxy_obs_phas-zxy_cal_phas)/zxy_err_phas)  
        if ShowRMS:
            nrmszxy_rho, _ = utl.calc_rms(zxy_obs_rho, zxy_cal_rho, 1.0/zxy_err_rho)
            nrmszxy_phas, _ = utl.calc_rms(zxy_obs_phas, zxy_cal_phas, 1.0/zxy_err_phas)


    cmp ="ZYX"
    zyx_obs_rho = np.abs(obs_rho[:, 2])
    zyx_obs_phas = np.abs(obs_phas[:, 2])
    zyx_err_rho = err_rho[:,2]
    zyx_err_phas = err_phas[:,2]
    if PlotPred:
        zyx_cal_rho = np.abs(obs_rho[:, 2])
        zyx_cal_phas = np.abs(obs_phas[:, 2])
        site_res = np.append(site_res, (zyx_obs_rho-zyx_cal_rho)/zyx_err_rho)
        site_res = np.append(site_res, (zyx_obs_phas-zyx_cal_phas)/zyx_err_phas)  
        if ShowRMS:
            nrmszyx_rho, _ = utl.calc_rms(zyx_obs_rho, zyx_cal_rho, 1.0/zyx_err_rho)
            nrmszyx_phas, _ = utl.calc_rms(zyx_obs_phas, zyx_cal_phas, 1.0/zyx_err_phas)




    if PlotFull:
        cmp ="ZYY"
        zyy_obs_rho = np.abs(obs_rho[:, 3])
        zyy_obs_phas = np.abs(obs_phas[:, 3])
        zyy_err_rho = err_rho[:,3]
        zyy_err_phas = err_phas[:,3]
        if PlotPred:
            zyy_cal_rho = np.abs(obs_rho[:, 3])
            zyy_cal_phas = np.abs(obs_phas[:, 3])
            site_res = np.append(site_res, (zyy_obs_rho-zyy_cal_rho)/zyy_err_rho)
            site_res = np.append(site_res, (zyy_obs_phas-zyy_cal_phas)/zyy_err_phas)  
            if ShowRMS:
                nrmszyy_rho, _ = utl.calc_rms(zyy_obs_rho, zyy_cal_rho, 1.0/zyy_err_rho)
                nrmszyy_phas, _ = utl.calc_rms(zyy_obs_phas, zyy_cal_phas, 1.0/zyy_err_phas)


        sRes = np.asarray(site_res)
        nD = np.size(sRes)
        if PlotPred:
            siteRMS = np.sqrt(np.sum(np.power(sRes,2.))/(float(nD)-1))
            rmsstrng ="   nRMS: "+str(np.around(siteRMS,1))
            print("Site nRMS: "+str(siteRMS))
        else:
            rmsstrng = "" 
        
        
        fig, axes = plt.subplots(2,2, figsize = FigSize, subplot_kw=dict(box_aspect=1.),
                 sharex=True, sharey=False, constrained_layout=True)
        
        fig.suptitle(r"Site: "+site_nam+rmsstrng
                 +"\nLat: "+str(site_lat)+"   Lon: "+str(site_lon)
                 +"\nEasting: "+str(site_utmx)+"   Northing: "+str(site_utmy)
                 +" (EPSG="+str(EPSG)+")  \nElev: "+ str(site_elev)+" m\n",
                 ha="left", x=0.1,fontsize=Titlesize)
        
        
        #  ZXX
        
        if PlotFull:
            if PlotPred:
                axes[0,0].plot(site_per, Zxxrc, color="r",linestyle="-", linewidth=Linewidth)
            
            if PlotObsv:
                if ShowErrors:
                    axes[0,0].errorbar(site_per,zxx_obs_rho, yerr=Zxxe,
                                    linestyle="",
                                    marker="o",
                                    color="r",
                                    linewidth=Linewidth,
                                    markersize=Markersize)
                else:
                    axes[0,0].plot(site_per, zxx_obs_rho,
                                   color="r",
                                   linestyle="",
                                   marker="o",
                                   markersize=Markersize)
            
            if PlotPred:
                axes[0,0].plot(site_per, Zxxic, color="b",linestyle="-", linewidth=Linewidth)
            
            if PlotObsv:
                if ShowErrors:
                    axes[0,0].errorbar(site_per,Zxxio, yerr=Zxxe,
                                    linestyle="",
                                    marker="o",
                                    color="b",
                                    linewidth=Linewidth,
                                    markersize=Markersize)
                else:
                    axes[0,0].plot(site_per, Zxxio,
                                   color="b",
                                   linestyle="",
                                   marker="o",
                                   markersize=Markersize)
            
            axes[0,0].set_xscale("log")
            axes[0,0].set_yscale("log")
            axes[0,0].set_xlim(PerLimits)
            if ZLimitsXX != ():
                axes[0,0].set_ylim(ZLimitsXX)
            axes[0,0].legend(["rho", "phas"])
            
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
           axes[0,1].plot(site_per, Zxyrc, color="r",linestyle="-", linewidth=Linewidth)
        
        if PlotObsv:
            if ShowErrors:
                axes[0,1].errorbar(site_per,Zxyro, yerr=Zxye,
                                linestyle="",
                                marker="o",
                                color="r",
                                linewidth=Linewidth,
                                markersize=Markersize)
            else:
                axes[0,1].plot(site_per,
                               Zxyro, color="r",
                               linestyle="",
                               marker="o",
                               markersize=Markersize)
        
        if PlotPred:
            axes[0,1].plot(site_per, Zxyic, color="b",linestyle="-", linewidth=Linewidth)
        if PlotObsv:
            if ShowErrors:
                    axes[0,1].errorbar(site_per,Zxyio, yerr=Zxye,
                                    linestyle="",
                                    marker="o",
                                    color="b",
                                    linewidth=Linewidth,
                                    markersize=Markersize)
            else:
                    axes[0,1].plot(site_per, Zxyio,
                                   color="b",
                                   linestyle="",
                                   marker="o",
                                   markersize=Markersize)
        
        axes[0,1].set_xscale("log")
        axes[0,1].set_yscale("log")
        axes[0,1].set_xlim(PerLimits)
        if ZLimitsXY != ():
            axes[0,1].set_ylim(ZLimitsXY)
        axes[0,1].legend(["rho", "phas"])
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
           axes[1,0].plot(site_per, Zyxrc, color="r",linestyle="-", linewidth=Linewidth)
        
        if PlotObsv:
            if ShowErrors:
                axes[1,0].errorbar(site_per,Zyxro, yerr=Zyxe,
                                linestyle="",
                                marker="o",
                                color="r",
                                linewidth=Linewidth,
                                markersize=Markersize)
            else:
                axes[1,0].plot(site_per,
                               Zyxro, color="r",
                               linestyle="",
                               marker="o",
                               markersize=Markersize)
        
        if PlotPred:
            axes[1,0].plot(site_per, Zyxic, color="b",linestyle="-", linewidth=Linewidth)
            
        if PlotObsv:
            if ShowErrors:
                axes[1,0].errorbar(site_per,Zyxio, yerr=Zyxe,
                                linestyle="",
                                marker="o",
                                color="b",
                                linewidth=Linewidth,
                                markersize=Markersize)
            else:
                axes[1,0].plot(site_per, Zyxio,
                               color="b",
                               linestyle="",
                               marker="o",
                               markersize=Markersize)
        
        axes[1,0].set_xscale("log")
        axes[1,0].set_yscale("log")
        axes[1,0].set_xlim(PerLimits)
        if ZLimitsXY != ():
            axes[1,0].set_ylim(ZLimitsXY)
        axes[1,0].legend(["rho", "phas"])
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
        if PlotFull:
            if PlotPred:
                axes[1,1].plot(site_per, Zyyrc, color="r",linestyle="-", linewidth=Linewidth)
        
            if PlotObsv:
                if ShowErrors:
                    axes[1,1].errorbar(site_per,Zyyro, yerr=Zyye,
                                    linestyle="",
                                    marker="o",
                                    color="r",
                                    linewidth=Linewidth,
                                    markersize=Markersize)
                else:
                    axes[1,1].plot(site_per, Zyyro,
                                   color="r",
                                   linestyle="",
                                   marker="o",
                                   markersize=Markersize)
        
            if PlotPred:
                axes[1,1].plot(site_per, Zyyic, color="b",linestyle="-", linewidth=Linewidth)
        
            if PlotObsv:
                if ShowErrors:
                    axes[1,1].errorbar(site_per,Zyyio, yerr=Zyye,
                                    linestyle="",
                                    marker="o",
                                    color="b",
                                    linewidth=Linewidth,
                                    markersize=Markersize)
                else:
                    axes[1,1].plot(site_per, Zyyio,
                                   color="b",
                                   linestyle="",
                                   marker="o",
                                   markersize=Markersize)
        
            axes[1,1].set_xscale("log")
            axes[1,1].set_yscale("log")
            axes[1,1].set_xlim(PerLimits)
            if ZLimitsXX != ():
                axes[1,1].set_ylim(ZLimitsXX)
            axes[1,1].legend(["rho", "phas"])
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
    
        
        fig.subplots_adjust(wspace = 0.4,top = 0.8 )
        # fig.subplots_adjust(wspace = 0.25)
        # fig.tight_layout()
        
        for F in PlotFormat:
            plt.savefig(PlotDir+PlotFile+"_"+s+F, dpi=400)
        
        if PDFCatalog:
            pdf_list.append(PlotDir+PlotFile+"_"+s+".pdf")
        
        
        plt.show()
        plt.close(fig)
        


if PDFCatalog:
    utl.make_pdf_catalog(PlotDir, pdflist=pdf_list, filename=PlotDir+PDFCatalogName)

