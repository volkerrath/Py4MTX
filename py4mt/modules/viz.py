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
