# # -*- coding: utf-8 -*-
# """
# Created on Sun Dec 27 17:23:34 2020

# @author: vrath
# """

import os
import sys
from sys import exit as error
from time import process_time
from datetime import datetime
import warnings

import math

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.cm as cm
import mpl_toolkits.axes_grid1
import matplotlib.ticker
import cycler

import util as utl

def plot_impedance(thisaxis=None, data=None, **pltargs):
    """
    Plot impedances
    

    Parameters
    ----------
    thisaxis : axis object, optional
        Determines where to plot, if None, new fig/ax pair is generated and plotted
        The default is None.
    data : np.array
        Data to plot. The default is None.
    **pltargs : dict
        kwargs for plotting.

    Returns
    -------
    ax : TYPE
        DESCRIPTION.

    """

    if thisaxis is None:
        fig, ax =  plt.subplots(1, figsize=pltargs["pltsize"])
        # fig.suptitle(pltargs["title"], fontsize=pltargs["fontsizes"][2])
    else:
        ax = thisaxis
    
    
    
    [points, dats] = np.shape(data)     
        
    
    per = data[:,0]
    obs_real = data[:,1]
    obs_imag = data[:,2]
    
    plot_err = False
    if dats > 3: 
        err_real = data[:,3]
        err_imag = data[:,4]  
        plot_err = True
        
    plot_cal = False
    if dats > 6: 
        cal_real = data[:,5]
        cal_imag = data[:,6] 
        plot_cal = True
            

    if plot_cal:
       ax.plot(per, cal_real, 
               color=pltargs["c_cal"][0], linestyle="-", linewidth=pltargs["l_cal"])
    
  
    if plot_err:
        ax.errorbar(per,
                    obs_real,
                    yerr=err_real,
                    linestyle="",
                    marker=pltargs["m_obs"][0],
                    markersize=pltargs["m_size"],
                    color=pltargs["c_obs"][0],
                    linewidth=pltargs["l_cal"],
                    capsize=2, capthick=0.5)
    else:
        ax.plot(per,
                obs_real,
                linestyle="",
                marker=pltargs["m_obs"][0],
                markersize=pltargs["m_size"],
                linewidth=pltargs["l_cal"],
                color=pltargs["c_obs"][0])
        
    if plot_cal:
       ax.plot(per, cal_imag, 
               color=pltargs["c_cal"][1], linestyle="-", linewidth=pltargs["l_cal"])
    
  
    if plot_err:
        ax.errorbar(per,
                    obs_imag,
                    yerr=err_imag,
                    linestyle="",
                    marker=pltargs["m_obs"][1],
                    markersize=pltargs["m_size"],
                    color=pltargs["c_obs"][1],
                    linewidth=pltargs["l_cal"],
                    capsize=2, capthick=0.5)
    else:
        ax.plot(per,
                obs_imag,
                linestyle="",
                marker=pltargs["m_obs"][1],
                markersize=pltargs["m_size"],
                linewidth=pltargs["l_cal"],
                color=pltargs["c_obs"][1])
    
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("impedance [mv km$^{-1}$ nT$^{-1}$]")
    ax.set_xlim(pltargs["perlimits"])
    if len(pltargs["zlimits"]) != 0:
        ax.set_ylim(pltargs["zlimits"])
    ax.legend(["real", "imag"])
    # ax.xaxis.set_ticklabels([])
    ax.tick_params(labelsize=pltargs["labelsize"]-1)
    ax.set_title(pltargs["title"], fontsize=pltargs["fontsize"])
    ax.grid("both", "both", linestyle="-", linewidth=0.5)
    if len(pltargs["nrms"])==2:
        nrmsr = np.around(pltargs["nrms"][0],1)
        nrmsi = np.around(pltargs["nrms"][1],1)
        strrms = "nrms = "+str(nrmsr)+" | "+str(nrmsi)
        ax.text(0.05, 0.05,strrms,
                           transform=ax.transaxes,
                           fontsize = pltargs["fontsize"]-2,
                           ha="left", va="bottom",
                           bbox={"pad": 2, "facecolor": "white", "edgecolor": "white" ,"alpha": 0.8} )       
        
        
        
    if thisaxis is None:
        for f in pltargs["pltformat"]:
            matplotlib.pyplot.savefig(pltargs["pltfile"]+f)
        # matplotlib.pyplot.show()
        # matplotlib.pyplot.clf()
        
    return ax

def plot_rhophas(thisaxis=None, data=None, **pltargs):
    """
    Plot impedances
    

    Parameters
    ----------
    thisaxis : axis object, optional
        Determines where to plot, if None, new fig/ax pair is generated and plotted
        The default is None.
    data : np.array
        Data to plot. The default is None.
    **pltargs : dict
        kwargs for plotting.

    Returns
    -------
    ax : TYPE
        DESCRIPTION.

    """

    if thisaxis is None:
        fig, ax =  plt.subplots(1, figsize=pltargs["pltsize"])
        # fig.suptitle(pltargs["title"], fontsize=pltargs["fontsizes"][2])
    else:
        ax = thisaxis
    
    
    
    [points, dats] = np.shape(data)     
        
    
    per = data[:,0]
    obs_rhoa = data[:,1]
    obs_phas = data[:,2]
    
    plot_err = False
    if dats > 3: 
        err_rhoa = data[:,3]
        err_phas = data[:,4]  
        plot_err = True
        
    plot_cal = False
    if dats > 6: 
        cal_rhoa = data[:,5]
        cal_phas = data[:,6] 
        plot_cal = True
            

    if plot_cal:
       ax.plot(per, cal_rhoa, 
               color=pltargs["c_cal"][0], linestyle="-", linewidth=pltargs["l_cal"])
    
  
    if plot_err:
        ax.errorbar(per,
                    obs_rhoa,
                    yerr=err_rhoa,
                    linestyle="",
                    marker=pltargs["m_obs"][0],
                    markersize=pltargs["m_size"],
                    color=pltargs["c_obs"][0],
                    linewidth=pltargs["l_cal"],
                    capsize=2, capthick=0.5)
    else:
        ax.plot(per,
                obs_rhoa,
                linestyle="",
                marker=pltargs["m_obs"][0],
                markersize=pltargs["m_size"],
                linewidth=pltargs["l_cal"],
                color=pltargs["c_obs"][0])
        
    if plot_cal:
       ax.plot(per, cal_phas, 
               color=pltargs["c_cal"][1], linestyle="-", linewidth=pltargs["l_cal"])
    
  
    if plot_err:
        ax.errorbar(per,
                    obs_phas,
                    yerr=err_phas,
                    linestyle="",
                    marker=pltargs["m_obs"][1],
                    markersize=pltargs["m_size"],
                    color=pltargs["c_obs"][1],
                    linewidth=pltargs["l_cal"],
                    capsize=2, capthick=0.5)
    else:
        ax.plot(per,
                obs_phas,
                linestyle="",
                marker=pltargs["m_obs"][1],
                markersize=pltargs["m_size"],
                linewidth=pltargs["l_cal"],
                color=pltargs["c_obs"][1])
    
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("impedance [mv km$^{-1}$ nT$^{-1}$]")
    ax.set_xlim(pltargs["perlimits"])
    if len(pltargs["zlimits"]) != 0:
        ax.set_ylim(pltargs["zlimits"])
    ax.legend(["rhoa", "phas"])
    # ax.xaxis.set_ticklabels([])
    ax.tick_params(labelsize=pltargs["labelsize"]-1)
    ax.set_title(pltargs["title"], fontsize=pltargs["fontsize"])
    ax.grid("both", "both", linestyle="-", linewidth=0.5)
    if len(pltargs["nrms"])==2:
        nrmsr = np.around(pltargs["nrms"][0],1)
        nrmsi = np.around(pltargs["nrms"][1],1)
        strrms = "nrms = "+str(nrmsr)+" | "+str(nrmsi)
        ax.text(0.05, 0.05,strrms,
                           transform=ax.transaxes,
                           fontsize = pltargs["fontsize"]-2,
                           ha="left", va="bottom",
                           bbox={"pad": 2, "facecolor": "white", "edgecolor": "white" ,"alpha": 0.8} )       
        
        
        
    if thisaxis is None:
        for f in pltargs["pltformat"]:
            matplotlib.pyplot.savefig(pltargs["pltfile"]+f)
        # matplotlib.pyplot.show()
        # matplotlib.pyplot.clf()
        
    return ax

def plot_phastens(thisaxis=None, data=None, **pltargs):
    """
    Plot phase tensor
    

    Parameters
    ----------
    thisaxis : axis object, optional
        Determines where to plot, if None, new fig/ax pair is generated and plotted
        The default is None.

    data : np.array
        Data to plot. The default is None.
    **pltargs : dict
        kwargs for plotting.

    Returns
    -------
    ax : TYPE
        DESCRIPTION.

    """

    if thisaxis is None:
        fig, ax =  plt.subplots(1, 1, figsize=pltargs["pltsize"])
        # fig.suptitle(pltargs["title"], fontsize=pltargs["fontsizes"][2])
    else:
        ax = thisaxis
    
       
    [points, dats] = np.shape(data)     
        
    
    per = data[:,0]
    obs_phast = data[:,1]
    
    plot_err = False
    if dats > 2: 
        err_phast = data[:,2]
        plot_err = True
        
    plot_cal = False
    if dats > 3: 
        cal_phast = data[:,3]
        plot_cal = True
            
    
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("phase tensor [-]")
    ax.set_xlim(pltargs["perlimits"])
    if len(pltargs["plimits"]) != 0:
        ax.set_ylim(pltargs["plimits"])
    # ax.legend(["phast", "imag"])
    # ax.xaxis.set_ticklabels([])
    ax.tick_params(labelsize=pltargs["labelsize"]-1)
    ax.set_title(pltargs["title"], fontsize=pltargs["fontsize"])
    ax.grid("both", "both", linestyle="-", linewidth=0.5)
    if len(pltargs["nrms"])==2:
        nrmsr = np.around(pltargs["nrms"][0],1)
        nrmsi = np.around(pltargs["nrms"][1],1)
        strrms = "nrms = "+str(nrmsr)+" | "+str(nrmsi)
        ax.text(0.05, 0.05,strrms,
                           transform=ax.transaxes,
                           fontsize = pltargs["fontsize"]-2,
                           ha="left", va="bottom",
                           bbox={"pad": 2, "facecolor": "white", "edgecolor": "white" ,"alpha": 0.8} )       
        
        
    if thisaxis is None:
        for f in pltargs["pltformat"]:
            matplotlib.pyplot.savefig(pltargs["pltfile"]+f)
        # matplotlib.pyplot.show()
        # matplotlib.pyplot.clf()
   
    return ax


def plot_vtf(thisaxis=None, data=None, **pltargs):
    """
    

    Parameters
    ----------
    thisax : axis object
        DESCRIPTION.
    period : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.

    Returns
    -------
    thisaxis : axis object
        DESCRIPTION.
        DESCRIPTION.

    """

    if thisaxis is None:
        fig, ax =  plt.subplots(1, figsize=pltargs["figsize"])
        # fig.suptitle(pltargs["title"], fontsize=pltargs["fontsizes"][2])
    else:
        ax = thisaxis
        
        
    if thisaxis is None:
        for f in pltargs["plotformat"]:
            matplotlib.pyplot.savefig(pltargs["plotfile"]+f)
        # matplotlib.pyplot.show()
        # matplotlib.pyplot.clf()
        
    return ax


def plot_depth_prof(
        ThisAxis = None,
        PlotFile = "",
        PlotFormat = ["png",],
        PlotTitle = "",
        FigSize = [8.5*0.3937, 8.5*0.3937],
        Depth = [],
        DLimits = [],
        DLabel = " Depth (m)",
        Params = [],
        Partyp = "",
        PLabel = "",
        PLimits = [],
        Shade = [ 0.25 ],
        XScale = "log",
        PlotType = "steps",
        Legend = [],
        Linecolor = ["r", "g", "b", "m", "y"],
        Linetypes =  ["-","-","-","-","-"],
        Linewidth =  [1, 1, 1, 1,1,],
        Marker = ["v"],
        Markersize =[4],
        Fillcolor = [[0.7, 0.7, 0.7]],
        Logplot = True,
        Fontsizes =[10, 10, 12],
        PlotStrng="",
        StrngPos=[0.05,0.05],
        Save = True,
        Invalid=1.e30):
    """
    General plot of (multiple) depth profiiles

    Parameters
    ----------

    Depth :  np.array
        DESCRIPTION. The default is [].
    Params : np.array
        DESCRIPTION. The default is [].
    PLabels : list of strings, optional
        DESCRIPTION. The default is [].

    XScale: string, optional
        "linear", "log", "symlog", "asinh"
        Last two need further parameters, e.g
        ax.set_yscale("asinh", linear_width=a0)
        x.set_yscale("symlog", linthresh=2,)
    Ptype : string, optional
        Proy type. The default is "steps".

    ALabels: string, optional
        Axis Labels for Params and Depth. The default is "", and "(m}.
    PLimits,  DLimits : lists, optional
        Limits for Params and Depth The default is [].
    Errors : TYPE, optional
        DESCRIPTION. The default is [].
    PlotFile : string, optional
        Plot file name without extension
        The default is None.
    PlotTitle : string, optional
        Plot title
        he default is None.
    PlotFormat : string, optional
        List of output formats. The default is ["png",].
     Linecolor : TYPE, optional
        DESCRIPTION. The default is ["y", "r", "g", "b", "m"].
    Linetypes : TYPE, optional
        DESCRIPTION. The default is "".
    Fontsizes : TYPE, optional
        DESCRIPTION. The default is [10, 10, 12].
    PlotStrng : string, optional
        Annotation. The default is "".
    StrngPos : TYPE, optional
        Annotation proition. The default is [0.05,0.05].

    Returns
    -------
    ax

    Created May 1, 2023
    @author: vrath

    """

    cm = 1/2.54  # centimeters in inches

    if ThisAxis is None:
        fig, ax =  plt.subplots(1, 1, figsize=(FigSize))
        fig.suptitle(PlotTitle, fontsize=Fontsizes[2])
    else:
        ax = ThisAxis


    for iparset in range(len(Params)):

        P = Params[iparset]
        D = Depth[iparset]
        npar = np.shape(P)[0]
        ndat = np.shape(D)[0]

        df = D[-1] + 3*np.abs(D[-1]-D[-2])


        if Partyp=="":

            if "steps" in PlotType.lower():
                d = D
                for pp in np.arange(npar):
                    print("PPPP ",P[pp])
                    p = np.append(P[pp],P[pp][-1])
                    ax.step(p , d ,
                         where='pre',
                         c=Linecolor[pp],
                         ls=Linetypes[pp], lw=Linewidth[pp])
            else:
                d = D
                for pp in np.arange(npar):
                    p = P[pp]
                    p = np.append(P[pp],P[pp][-1])
                    ax.plot(p , d,
                            c=Linecolor[pp],
                            ls=Linetypes[pp], lw=Linewidth[pp])


        if "sens" in Partyp.lower():
            if "steps" in PlotType.lower():
                d = D
                for pp in np.arange(npar):
                    p = P[pp]
                    print(np.shape(d),np.shape(p))
                    ax.step(p , d ,
                         where='pre',
                         c=Linecolor[pp],
                         ls=Linetypes[pp], lw=Linewidth[pp])
            else:
                d = D[:-1]
                for pp in np.arange(npar):
                    p = P[pp]
                    ax.plot(p , d,
                            c=Linecolor[pp],
                            ls=Linetypes[pp], lw=Linewidth[pp])


        if "model" in Partyp.lower():

            d = np.append(D, df)
            # d = D

            if npar==3:


                p = np.append(P[0],P[0][-1])
                ep = np.append(P[1],P[1][-1])
                em = np.append(P[2],P[2][-1])
                # p = P[0]
                # ep = P[1]
                # em = P[2]
                print(np.shape(d),np.shape(em),np.shape(ep))

                if "fill" in PlotType.lower():
                    ax.fill_betweenx(d, em, ep,
                                step='post',
                                color=Fillcolor[0],
                                ls=Linetypes[0], lw=Linewidth[0],
                                alpha=Shade)

                ax.step(p , d ,
                         where='pre',
                         c=Linecolor[0],
                         ls=Linetypes[0], lw=Linewidth[0])
                ax.plot(p[-1] , d[-1],
                        c=Linecolor[0],  ls=Linetypes[0], lw=0,
                        marker=Marker[0], markersize=Markersize[0])
                ax.step(em , d ,
                         where='pre',
                         c=Linecolor[1],
                         ls=Linetypes[1], lw=Linewidth[1])
                ax.step(ep , d ,
                         where='pre',
                         c=Linecolor[1],
                         ls=Linetypes[1], lw=Linewidth[1])

            else:

                if "step" in PlotType.lower():
                    for pp in np.arange(npar):
                        p = P[pp]
                        ax.step(p , d,
                                where='pre',
                                color=Linecolor[pp],
                                    ls=Linetypes[pp],lw=Linewidth[0])
                else:
                    for pp in np.arange(npar):
                        p = P[pp]
                        ax.plot(p , d,
                                c=Linecolor[pp],
                                ls=Linetypes[pp], lw=Linewidth[0])

        ax.set_xlabel(PLabel, fontsize=Fontsizes[1])
        ax.set_ylabel(DLabel, fontsize=Fontsizes[1])
        ax.xaxis.set_label_position("top")
        ax.xaxis.set_ticks_position("both")
        ax.tick_params(labelsize=Fontsizes[0])

        if PLimits != []:
            ax.set_xlim(PLimits)
        if DLimits != []:
            ax.set_ylim(DLimits)

        if "lin" not in XScale:
            ax.set_xscale(XScale)



        ax.legend(Legend, fontsize=Fontsizes[1]-2, loc="best", ncol=1)

        if PLimits != []:
            ax.set_xlim(PLimits)
        if DLimits != []:
            ax.set_ylim(DLimits)

        ax.invert_yaxis()

        ax.grid("major", "both", linestyle=":", lw=0.3)
        ax.text(StrngPos[0], StrngPos[1],
                 PlotStrng, fontsize=Fontsizes[1]-1,transform=ax.transAxes,
                 bbox=dict(facecolor="white", alpha=0.5) )

        if ThisAxis is None:
            for F in PlotFormat:
                 matplotlib.pyplot.savefig(PlotFile+F)

            matplotlib.pyplot.show()
            matplotlib.pyplot.clf()

        return ax

def plot_matrix(
        ThisAxis = None,
        PlotFile = "",
        PlotTitle = "",
        PlotFormat = ["png",],
        FigSize = [8.5*0.3937, 8.5*0.3937],
        Matrix = [],
        TickStr="",
        AxLabels = ["layer #", "layer #"],
        AxTicks = [[], []],
        AxTickLabels = [[], []],
        ColorMap="viridis",
        Fontsizes=[10,10,12],
        Unit = "",
        PlotStrng="",
        StrngPos=[0.05,0.05],
        Aspect = "auto",
        Invalid=1.e30,
        Transpose=False):
    """
    Plots jacobians, covariance and resolution matrices.


    Parameters
    ----------
    PlotFile : TYPE, optional
        DESCRIPTION. The default is None.
    PlotTitle : TYPE, optional
        DESCRIPTION. The default is None.
    PlotFormat : TYPE, optional
        DESCRIPTION. The default is ["png",].


    Returns
    -------
    ax


    Created April 30, 2023
    @author: vrath

    """
    nn = np.shape(Matrix)
    if Transpose:
        Matrix = Matrix.T

    npar = nn[0]
    if Matrix.ndim==1:
        npar =math.isqrt(nn[0])
        Matrix = Matrix.reshape((npar,npar))

    if ThisAxis is None:
        fig, ax =  matplotlib.pyplot.subplots(1, 1, figsize=(FigSize))
        fig.suptitle(PlotTitle, fontsize=Fontsizes[2])
    else:
        ax = ThisAxis

    im = ax.imshow(Matrix, cmap=ColorMap, origin="upper")

    xticks = AxTicks[0]
    xlabels = AxTickLabels[0]
    # print(xticks)
    # print(xlabels)
    ax.set_xticks(xticks, xlabels) #, minor=False)
    ax.set_xlabel(AxLabels[0], fontsize=Fontsizes[1])
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    yticks = AxTicks[1]
    ylabels = AxTickLabels[1]
    # print(yticks)
    # print(ylabels)
    ax.set_yticks(yticks, ylabels) #, minor=False)
    ax.set_ylabel(AxLabels[1], fontsize=Fontsizes[1])

    if Aspect == "equal":
        ax.set_aspect("equal","box")
    else:
        ax.set_aspect(Aspect)

    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = matplotlib.pyplot.colorbar(im, cax=cax)
    cb.ax.set_title(Unit)

    if PlotStrng != "":

        props = dict(facecolor="white", alpha=0.9) # boxstyle="round"
        ax.text(StrngPos[0], StrngPos[1], PlotStrng,
                transform=ax.transAxes,
                fontsize=Fontsizes[1],
                verticalalignment="center", bbox=props)

    if ThisAxis is None:
        for F in PlotFormat:
             matplotlib.pyplot.savefig(PlotFile+F)

        # matplotlib.pyplot.show()
        # matplotlib.pyplot.clf()

    return ax


def make_pdf_catalog(workdir="./", pdflist= None, filename=None):
    """
    Make pdf catalog from site-plot(

    Parameters
    ----------
    Workdir : string
        Working directory.
    Filename : string
        Filename. Files to be appended must begin with this string.

    Returns
    -------
    None.

    """
    # error("not in 3.9! Exit")

    import fitz

    catalog = fitz.open()

    for pdf in pdflist:
        with fitz.open(pdf) as mfile:
            catalog.insert_pdf(mfile)

    catalog.save(filename, garbage=4, clean = True, deflate=True)
    catalog.close()

    print("\n"+str(np.size(pdflist))+" files collected to "+filename)
