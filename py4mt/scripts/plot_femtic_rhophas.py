#!/usr/bin/env python3

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


PerLimits = [0.001,10000.]
ZLimits = []


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
FigSize = (16*cm, 16*cm)
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
lat = data_dict["lat"]
lon = data_dict["lon"]
elv = data_dict["elv"]
nam = data_dict["nam"]



for s in sites:
    # if s>0:
    #     break
    print("Plotting site: "+str(s))
    print("Site name is ", nam[s][0])
    site_lon = lon[s] #[site_index]
    site_lat = lat[s] #a = [site_index]
    site_elev = elv[s]
    site_utmx, site_utmy = utl.proj_latlon_to_utm(site_lat, site_lon, utm_zone=EPSG)
    site_utmx = int(np.round(site_utmx))
    site_utmy = int(np.round(site_utmy))
    print(s, "at", site_lat, site_lon, site_elev)
    site_nam = nam[s]
    site_index = np.where(data_dict["num"] ==s)
    print(site_index)

    site_obs = data_dict["obs"][site_index]
    site_err = data_dict["err"][site_index]

    if PlotPred:
        site_cal = data_dict["cal"][site_index]
        site_res = np.ones_like(site_cal)

    site_per =  1./data_dict["frq"][site_index]
    # nsite = np.shape(site_per)[0]


    nn = 8
    site_res = np.empty((1,nn%2))
    rhoas = [ix for ix in range(nn) if ix % 2 == 0]
    phass = [ix for ix in range(nn) if ix % 2 == 1]
    obs_rhoa = site_obs[:,rhoas]
    obs_phas = site_obs[:,phass]
    err_rhoa = site_err[:,rhoas]
    err_phas = site_err[:,phass]
    if PlotPred:
        cal_rhoa = site_cal[:,rhoas]
        cal_phas = site_cal[:,phass]

    if PlotFull:
        cmp ="ZXX"        
        zxx_obs_rhoa = np.abs(obs_rhoa[:, 0])
        zxx_obs_phas = np.abs(obs_phas[:, 0])
        zxx_err_rhoa = err_rhoa[:,0]
        zxx_err_phas = err_phas[:,0]
        if PlotPred:
            zxx_cal_rhoa = np.abs(cal_rhoa[:, 0])
            zxx_cal_phas = np.abs(cal_phas[:, 0])
            site_res = np.append(site_res, (zxx_obs_rhoa-zxx_cal_rhoa)/zxx_err_rhoa)
            site_res = np.append(site_res, (zxx_obs_phas-zxx_cal_phas)/zxx_err_phas)  
            if ShowRMS:
                nrmszxx_rhoa, _ = utl.calc_rms(zxx_obs_rhoa, zxx_cal_rhoa, 1.0/zxx_err_rhoa)
                nrmszxx_phas, _ = utl.calc_rms(zxx_obs_phas, zxx_cal_phas, 1.0/zxx_err_phas)


    cmp ="ZXY"
    zxy_obs_rhoa = np.abs(obs_rhoa[:, 1])
    zxy_obs_phas = np.abs(obs_phas[:, 1])
    zxy_err_rhoa = err_rhoa[:,1]
    zxy_err_phas = err_phas[:,1]
    if PlotPred:
        zxy_cal_rhoa = np.abs(cal_rhoa[:, 1])
        zxy_cal_phas = np.abs(cal_phas[:, 1])
        site_res = np.append(site_res, (zxy_obs_rhoa-zxy_cal_rhoa)/zxy_err_rhoa)
        site_res = np.append(site_res, (zxy_obs_phas-zxy_cal_phas)/zxy_err_phas)  
        if ShowRMS:
            nrmszxy_rhoa, _ = utl.calc_rms(zxy_obs_rhoa, zxy_cal_rhoa, 1.0/zxy_err_rhoa)
            nrmszxy_phas, _ = utl.calc_rms(zxy_obs_phas, zxy_cal_phas, 1.0/zxy_err_phas)


    cmp ="ZYX"
    zyx_obs_rhoa = np.abs(obs_rhoa[:, 2])
    zyx_obs_phas = np.abs(obs_phas[:, 2])
    zyx_err_rhoa = err_rhoa[:,2]
    zyx_err_phas = err_phas[:,2]
    if PlotPred:
        zyx_cal_rhoa = np.abs(cal_rhoa[:, 2])
        zyx_cal_phas = np.abs(cal_phas[:, 2])
        site_res = np.append(site_res, (zyx_obs_rhoa-zyx_cal_rhoa)/zyx_err_rhoa)
        site_res = np.append(site_res, (zyx_obs_phas-zyx_cal_phas)/zyx_err_phas)  
        if ShowRMS:
            nrmszyx_rhoa, _ = utl.calc_rms(zyx_obs_rhoa, zyx_cal_rhoa, 1.0/zyx_err_rhoa)
            nrmszyx_phas, _ = utl.calc_rms(zyx_obs_phas, zyx_cal_phas, 1.0/zyx_err_phas)




    if PlotFull:
        cmp ="ZYY"
        zyy_obs_rhoa = np.abs(obs_rhoa[:, 3])
        zyy_obs_phas = np.abs(obs_phas[:, 3])
        zyy_err_rhoa = err_rhoa[:,3]
        zyy_err_phas = err_phas[:,3]
        if PlotPred:
            zyy_cal_rhoa = np.abs(cal_rhoa[:, 3])
            zyy_cal_phas = np.abs(cal_phas[:, 3])
            site_res = np.append(site_res, (zyy_obs_rhoa-zyy_cal_rhoa)/zyy_err_rhoa)
            site_res = np.append(site_res, (zyy_obs_phas-zyy_cal_phas)/zyy_err_phas)  
            if ShowRMS:
                nrmszyy_rhoa, _ = utl.calc_rms(zyy_obs_rhoa, zyy_cal_rhoa, 1.0/zyy_err_rhoa)
                nrmszyy_phas, _ = utl.calc_rms(zyy_obs_phas, zyy_cal_phas, 1.0/zyy_err_phas)


    sRes = np.asarray(site_res)
    nD = np.size(sRes)
    if PlotPred:
        siteRMS = np.sqrt(np.sum(np.power(sRes,2.))/(float(nD)-1))
        rmsstrng ="   nRMS: "+str(np.around(siteRMS,1))
        print("Site nRMS: "+str(siteRMS))
    else:
        rmsstrng = "" 
        
        
    # fig, axes = plt.subplots(2,2, figsize = FigSize, # subplot_kw=dict(box_aspect=1.),
    #          sharex=True, sharey=False) #, constrained_layout=True)

    fig, axes = plt.subplots(2,2, figsize = FigSize, 
                             subplot_kw=dict(box_aspect=1.),
                             sharex=True, 
                             layout='constrained'
                             )

    fig.suptitle(r"Site: "+site_nam+rmsstrng
             +"\nLat: "+str(np.around(site_lat, 6))
             +"   Lon: "+str(np.around(site_lon, 6))
             +"\nEasting: "+str(site_utmx)+"   Northing: "+str(site_utmy)
             +" (EPSG="+str(EPSG)+")  \nElev: "+ str(site_elev)+" m\n",
             ha="left", x=0.1, fontsize=Titlesize)
    
    
    #  ZXX
    
    if PlotFull:
        if PlotPred:
            axes[0,0].plot(site_per, zxx_cal_rhoa, color="r",linestyle="-", linewidth=Linewidth)
        
        if PlotObsv:
            if ShowErrors:
                axes[0,0].errorbar(site_per,zxx_obs_rhoa, yerr=zxx_err_rhoa,
                                linestyle="",
                                marker="o",
                                color="r",
                                linewidth=Linewidth,
                                markersize=Markersize,
                                capsize =2, capthick=0.5)
            else:
                axes[0,0].plot(site_per, zxx_obs_rhoa,
                               color="r",
                               linestyle="",
                               marker="o",
                               markersize=Markersize)
        
        if PlotPred:
            axes[0,0].plot(site_per, zxx_cal_phas, color="b",linestyle="-", linewidth=Linewidth)
        
        if PlotObsv:
            if ShowErrors:
                axes[0,0].errorbar(site_per,zxx_obs_phas, yerr=zxx_err_phas,
                                linestyle="",
                                marker="o",
                                color="b",
                                linewidth=Linewidth,
                                markersize=Markersize,
                                capsize =2, capthick=0.5)
            else:
                axes[0,0].plot(site_per, zxx_obs_phas,
                               color="b",
                               linestyle="",
                               marker="o",
                               markersize=Markersize)
        
        axes[0,0].set_xscale("log")
        axes[0,0].set_yscale("log")
        axes[0,0].set_ylabel("impedance (mV km$^{-1}$ n$^{-1}$)" )
        axes[0,0].set_xlim(PerLimits)
        if len(ZLimits) != 0:
            axes[0,0].set_ylim(ZLimits)
        axes[0,0].legend(["rhoa", "phas"])
        
        axes[0,0].tick_params(labelsize=Labelsize-1)
        axes[0,0].set_title("|ZXX|", fontsize=Fontsize)
        axes[0,0].grid("both", "both", linestyle="-", linewidth=0.5)
        if ShowRMS:
            nRMSr = np.around(nrmszxx_rhoa,1)
            nRMSi = np.around(nrmszxx_phas,1)
            StrRMS = "nRMS = "+str(nRMSr)+" | "+str(nRMSi)
            axes[0,0].text(0.05, 0.05,StrRMS,
                               transform=axes[0,0].transAxes,
                               fontsize = Fontsize-2,
                               ha="left", va="bottom",
                               bbox={"pad": 2, "facecolor": "white", "edgecolor": "white" ,"alpha": 0.8} )
    
    
    #  ZXY
    if PlotPred:
       axes[0,1].plot(site_per, zxy_cal_rhoa, color="r",linestyle="-", linewidth=Linewidth)
    
    if PlotObsv:
        if ShowErrors:
            axes[0,1].errorbar(site_per,zxy_obs_rhoa, yerr=zxy_err_rhoa,
                            linestyle="",
                            marker="o",
                            color="r",
                            linewidth=Linewidth,
                            markersize=Markersize,
                            capsize =2, capthick=0.5)
        else:
            axes[0,1].plot(site_per,
                           zxy_obs_rhoa, color="r",
                           linestyle="",
                           marker="o",
                           markersize=Markersize)
    
    if PlotPred:
        axes[0,1].plot(site_per, zxy_cal_phas, color="b",linestyle="-", linewidth=Linewidth)
    if PlotObsv:
        if ShowErrors:
                axes[0,1].errorbar(site_per,zxy_obs_phas, yerr=zxy_err_phas,
                                linestyle="",
                                marker="o",
                                color="b",
                                linewidth=Linewidth,
                                markersize=Markersize,
                                capsize =2, capthick=0.5)
        else:
                axes[0,1].plot(site_per, zxy_obs_phas,
                               color="b",
                               linestyle="",
                               marker="o",
                               markersize=Markersize)
    
    axes[0,1].set_xscale("log")
    axes[0,1].set_yscale("log")
    axes[0,1].set_ylabel("impedance (mV km$^{-1}$ n$^{-1}$)" )
    axes[0,1].set_xlim(PerLimits)
    if len(ZLimits) != 0:
        axes[0,1].set_ylim(ZLimits)
    axes[0,1].legend(["rhoa", "phas"])
    # axes[0,1].xaxis.set_ticklabels([])
    axes[0,1].tick_params(labelsize=Labelsize-1)
    axes[0,1].set_title("|ZXY|", fontsize=Fontsize)
    axes[0,1].grid("both", "both", linestyle="-", linewidth=0.5)
    if ShowRMS:
        nRMSr = np.around(nrmszxy_rhoa,1)
        nRMSi = np.around(nrmszxy_phas,1)
        StrRMS = "nRMS = "+str(nRMSr)+" | "+str(nRMSi)
        axes[0,1].text(0.05, 0.05,StrRMS,
                           transform=axes[0,1].transAxes,
                           fontsize = Fontsize-2,
                           ha="left", va="bottom",
                           bbox={"pad": 2, "facecolor": "white", "edgecolor": "white" ,"alpha": 0.8} )
    
    
    #  ZYX
    if PlotPred:
       axes[1,0].plot(site_per, zyx_cal_rhoa, color="r",linestyle="-", linewidth=Linewidth)
    
    if PlotObsv:
        if ShowErrors:
            axes[1,0].errorbar(site_per,zyx_obs_rhoa, yerr=zyx_err_rhoa,
                            linestyle="",
                            marker="o",
                            color="r",
                            linewidth=Linewidth,
                            markersize=Markersize,
                            capsize =2, capthick=0.5)
        else:
            axes[1,0].plot(site_per,
                           zyx_obs_rhoa, color="r",
                           linestyle="",
                           marker="o",
                           markersize=Markersize)
    
    if PlotPred:
        axes[1,0].plot(site_per, zyx_cal_phas, color="b",linestyle="-", linewidth=Linewidth)
        
    if PlotObsv:
        if ShowErrors:
            axes[1,0].errorbar(site_per,zyx_obs_phas, yerr=zyx_err_phas,
                            linestyle="",
                            marker="o",
                            color="b",
                            linewidth=Linewidth,
                            markersize=Markersize,
                            capsize =2, capthick=0.5)
        else:
            axes[1,0].plot(site_per, zyx_obs_phas,
                           color="b",
                           linestyle="",
                           marker="o",
                           markersize=Markersize)
    
    axes[1,0].set_xscale("log")
    axes[1,0].set_yscale("log")
    axes[1,0].set_xlabel("period (s)")
    axes[1,0].set_ylabel("impedance (mV km$^{-1}$ n$^{-1}$)" )
    axes[1,0].set_xlim(PerLimits)
    if len(ZLimits) != 0:
        axes[1,0].set_ylim(ZLimits)
    axes[1,0].legend(["rhoa", "phas"])
    # axes[1,0].xaxis.set_ticklabels([])
    axes[1,0].tick_params(labelsize=Labelsize-1)
    axes[1,0].set_title("|ZYX|", fontsize=Fontsize)
    axes[1,0].grid("both", "both", linestyle="-", linewidth=0.5)
    if ShowRMS:
        nRMSr = np.around(nrmszyx_rhoa,1)
        nRMSi = np.around(nrmszyx_phas,1)
        StrRMS = "nRMS = "+str(nRMSr)+" | "+str(nRMSi)
        axes[1,0].text(0.05, 0.05,StrRMS,
                           transform=axes[1,0].transAxes,
                           fontsize = Fontsize-2,
                           ha="left", va="bottom",
                       bbox={"pad": 2, "facecolor": "white", "edgecolor": "white" ,"alpha": 0.8} )
    
    
    #  ZYY
    if PlotFull:
        if PlotPred:
            axes[1,1].plot(site_per, zyy_cal_rhoa, color="r",linestyle="-", linewidth=Linewidth)
    
        if PlotObsv:
            if ShowErrors:
                axes[1,1].errorbar(site_per,zyy_obs_rhoa, yerr=zyy_err_rhoa,
                                linestyle="",
                                marker="o",
                                color="r",
                                linewidth=Linewidth,
                                markersize=Markersize,
                                capsize =2, capthick=0.5)
            else:
                axes[1,1].plot(site_per, zyy_obs_rhoa,
                               color="r",
                               linestyle="",
                               marker="o",
                               markersize=Markersize)
    
        if PlotPred:
            axes[1,1].plot(site_per, zyy_cal_phas, color="b",linestyle="-", linewidth=Linewidth)
    
        if PlotObsv:
            if ShowErrors:
                axes[1,1].errorbar(site_per,zyy_obs_phas, yerr=zyy_err_phas,
                                linestyle="",
                                marker="o",
                                color="b",
                                linewidth=Linewidth,
                                markersize=Markersize,
                                capsize =2, capthick=0.5)
            else:
                axes[1,1].plot(site_per, zyy_obs_phas,
                               color="b",
                               linestyle="",
                               marker="o",
                               markersize=Markersize)
    
        axes[1,1].set_xscale("log")
        axes[1,1].set_yscale("log")
        axes[1,1].set_xlabel("period (s)")
        axes[1,1].set_ylabel("impedance (mV km$^{-1}$ n$^{-1}$)" )
        axes[1,1].set_xlim(PerLimits)
        if len(ZLimits) != 0:
            axes[1,1].set_ylim(ZLimits)
        axes[1,1].legend(["rhoa", "phas"])
        # axes[1,1].xaxis.set_ticklabels([])
        axes[1,1].tick_params(labelsize=Labelsize-1)
        axes[1,1].set_title("|ZYY|", fontsize=Fontsize)
        axes[1,1].grid("both", "both", linestyle="-", linewidth=0.5)
        if ShowRMS:
            nRMSr = np.around(nrmszyy_rhoa,1)
            nRMSi = np.around(nrmszyy_phas,1)
            StrRMS = "nRMS = "+str(nRMSr)+" | "+str(nRMSi)
            axes[1,1].text(0.05, 0.05,StrRMS,
                               transform=axes[1,1].transAxes,
                               fontsize = Fontsize-2,
                               ha="left", va="bottom",
                               bbox={"pad": 2, "facecolor": "white", "edgecolor": "white" ,"alpha": 0.8} )

        

    # fig.subplots_adjust(top=0.8, bottom=0.1, hspace=-0.1, wspace=-0.1)

    fig.tight_layout()
    
    for F in PlotFormat:
        plt.savefig(PlotDir+PlotFile+"_"+nam[s]+F, dpi=400)
    
    if PDFCatalog:
        pdf_list.append(PlotDir+PlotFile+"_"+nam[s]+".pdf")
    
    
    plt.show()
    plt.close(fig)
        


if PDFCatalog:
    utl.make_pdf_catalog(PlotDir, pdflist=pdf_list, filename=PlotDir+PDFCatalogName)
