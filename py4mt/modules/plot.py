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

import numpy
import matplotlib as mpl
import matplotlib.pyplot as plt
import cycler



# def get_plot_params(pltfmt="pdf", fsiz=12, lsiz=10, lwid=1, grey = 0.7):
#     """
#     Set plot parameters.


#     Returns
#     -------
#     None.

#     """
#     PltFmt = pltfmt
#     Fsize = fsiz
#     Lsize = lsiz
#     Lwidth = lwid
#     Greyval = grey

#     Lcycle = (cycler('linestyle', ['-', '--', ':', '-.'])
#               * cycler('color', ['r', 'g', 'b', 'y']))


#     return PltFmt, Fsize, Lsize, Lwidth, Lcycle, Greyval

def plot_aem05(
        PlotName = None,
        PlotDir="./",
        PlotFmt = ["png"],
        DPI = 400,
        DataObs=None,
        DataCal=None,
        XLimits = [],
        YLimits = [],
        HLimits = [],
        ProfUnit  = ["(m)"],
        Colors=None,
        Linetypes=["-", ":",],
        Linewidths=[1,2,],
        Fontsizes=[12,10,16],
        Grey=[0.7],
        Logparams=[False,False,10.],
        PlotStrng="",
        PlotAltDiff=False):

    cm = 1./2.54  # centimeters to inches


    Fsize = Fontsizes[0]
    Lsize = Fontsizes[1]
    Tsize = Fontsizes[2]
    Lwidth = Linewidths[0]
    Ltype  = Linetypes[0]
    Greyval = Grey[0]

    LogPlot= Logparams[0]
    LogSym = Logparams[1]
    LinThresh = Logparams[2]


    fline = DataObs[:, 0]
    site_x = DataObs[:, 1]
    site_y = DataObs[:, 2]
    site_gps = DataObs[:, 3]
    site_alt = DataObs[:, 4]
    site_dem = DataObs[:, 5]
    data_obs = DataObs[:, 6:14]

    # ####### Plot data along flight line #########
    # Name = str(fline[0])
    sx = site_x-site_x[0]
    sy = site_y-site_y[0]
    prof_dist = numpy.sqrt(sx * sx + sy * sy)
    prof_min, prof_max = numpy.min(prof_dist), numpy.max(prof_dist)
    prof_samp = numpy.arange(numpy.size(prof_dist))

    IData = data_obs[:, 0:4]
    QData = data_obs[:, 4:8]
    IData_min, IData_max = numpy.min(IData), numpy.max(IData)

    QData_min, QData_max = numpy.min(QData), numpy.max(QData)


    fig, ax = plt.subplots(num=1, clear=True,
        nrows=3, ncols=1, sharex=True, sharey=False,
        figsize=(24.*cm, 16.5*cm), constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=3*cm)


    if PlotAltDiff:
        ax[0].plot(
            prof_dist[:],
            site_alt[:]-(site_gps[:] - site_dem[:]),
            "r-", linewidth=Linewidths[0])
        ax[0].set_title("Altitude Differences", fontsize=Fsize, y=1.0, pad=-(Fsize+2))

    else:
        ax[0].plot(
            prof_dist[:],
            site_alt[:],
            "r",
            prof_dist[:],
            site_gps[:] - site_dem[:],
            "g:", linewidth=Linewidths[0])
        ax[0].set_title("Altitudes", fontsize=Fsize, y=1.0, pad=-(Fsize+2))
        ax[0].legend([" Radar alt", "GPS alt"], fontsize=Lsize, loc="best")
        if HLimits:
            ax[0].set_ylim(HLimits)

    ax[0].set_ylabel(ProfUnit, fontsize=Fsize)
    ax[0].grid(True)
    # color="k", alpha=0.5, linestyle="dotted", linewidth=1.5)
    ax[0].tick_params(labelsize=Lsize)
    ax[0].legend(
        [" Radar alt", "GPS alt"],
        fontsize=Lsize, loc="best")


    ax[1].plot(
        prof_dist[:],
        QData[:, 0],
        "r",
        prof_dist[:],
        QData[:, 1],
        "g",
        prof_dist[:],
        QData[:, 2],
        "b",
        prof_dist[:],
        QData[:, 3],
        "y",
        linewidth=Linewidths[0]
    )
    ax[1].set_title("Quadrature", fontsize=Fsize, y=1.0, pad=-(Fsize+2))
    ax[1].set_ylim([QData_min, QData_max])
    ax[1].set_ylabel("(ppm)", fontsize=Lsize)
    ax[1].grid(True)
    # color="k", alpha=0.5, linestyle="dotted", linewidth=1.5)
    ax[1].tick_params(labelsize=Lsize)
    leg1 =ax[1].legend(
        [" 0.9 kHz", "3 kHz", "12 kHz", "24.5 kHz"],
        fontsize=Lsize-2, loc="best", ncol=2)
    leg1.set_title("Frequency", prop={'size':Lsize})

    if LogPlot:
        if LogSym:
            ax[1].set_yscale("symlog", linthresh=LinThresh)
        else:
            ax[1].set_yscale("log")
    else:
        ax[1].set_yscale("linear")


    ax[2].plot(
        prof_dist[:],#    _, Fsize, Lsize, Lwidth, Lcycle, Greyval=get_plot_params()
        IData[:, 0],
        "r",
        prof_dist[:],
        IData[:, 1],
        "g",
        prof_dist[:],
        IData[:, 2],
        "b",
        prof_dist[:],
        IData[:, 3],
        "y",
        linewidth=Linewidths[0]
    )

    ax[2].set_title("In-Phase", fontsize=Fsize, y=1.0, pad=-(Fsize+2))
    ax[2].set_ylim([IData_min, IData_max])
    ax[2].set_ylabel("(ppm)", fontsize=Fsize)
    ax[2].set_xlabel("Profile distance "+ProfUnit, fontsize=Lsize)
    # secax = ax[2].secondary_xaxis("top", functions =(dist2sample,sample2dist))
    # secax.set_xlabel("sample [-]")

    ax[2].grid(True)
    # color="k", alpha=0.5, linestyle="dotted", linewidth=1.5)
    ax[2].tick_params(labelsize=Lsize)
    leg2 = ax[2].legend([" 0.9 kHz", "3 kHz", "12 kHz", "24.5 kHz"],
                       fontsize=Lsize-2, loc="best", ncol=2)
    leg2.set_title("Frequency", prop={'size':Lsize})

    if LogPlot:
        if LogSym:
            ax[2].set_yscale("symlog", linthresh=LinThresh)
        else:
            ax[2].set_yscale("log")
    else:
        ax[2].set_yscale("linear")



    fig.suptitle("Flightline: "+PlotName+PlotStrng, x=0.5, y=0.94, fontsize=Tsize, ha='center')



    for F in PlotFmt:
        plt.savefig(PlotDir+PlotName+F, dpi=DPI)

    plt.show()
    plt.clf()



def plot_genesis(
        PlotName = None,
        PlotDir="./",
        PlotFmt = ["png"],
        DPI = 400,
        DataObs=None,
        DataCal=None,
        XLimits =[],
        YLimits =[],
        HLimits = [],

        ProfUnit  = "(m)",
        Colors=None,
        Linetypes=["-", ":",],
        Linewidths=[1,2],
        Fontsizes=[12,10,12],
        Grey=[0.7],
        Logparams=[True,True,0,1],
        PlotStrng="",
        PlotAltDiff=False):

    cm = 1./2.54  # centimeters to inches
    # _, Fsize, Lsize, Lwidth, Lcycle, Greyval=get_plot_params()

    Fsize = Fontsizes[0]
    Lsize = Fontsizes[1]
    Tsize = Fontsizes[2]
    Lwidth = Linewidths[0]
    Ltype  = Linetypes[0]
    Greyval = Grey[0]

    LogPlot= Logparams[0]
    LogSym = Logparams[1]
    LinThresh = Logparams[2]


    fline = DataObs[:, 0]

    site_x = DataObs[:, 1]
    site_y = DataObs[:, 2]

    site_gps = DataObs[:, 3]
    site_alt = DataObs[:, 4]
    site_dem = DataObs[:, 5]
    XData = DataObs[:, 6:17]
    ZData = DataObs[:, 17:28]

    # Name = str(fline[0])

    sx = site_x-site_x[0]
    sy = site_y-site_y[0]
    prof_dist = numpy.sqrt(sx * sx + sy * sy)
    prof_min, prof_max = numpy.min(prof_dist), numpy.max(prof_dist)

    prof_samp = numpy.arange(numpy.size(prof_dist))

    X_min, X_max = numpy.min(XData), numpy.max(XData)
    Z_min, Z_max = numpy.min(ZData), numpy.max(ZData)
    X_minpos = numpy.min(XData[XData>0.])
    Z_minpos = numpy.min(ZData[ZData>0.])

    if LinThresh == []:
        LinThresh = numpy.min([Z_minpos])

    print("Xmin,max = "+str(X_min)+",  "+str(X_max))
    print("Zmin,max = "+str(Z_min)+",  "+str(Z_max))
    print("Xminpos,Zminpos  = "+str(X_minpos)+",  "+str(Z_minpos))
    print("LinThresh = "+str( LinThresh))



    fig, ax = plt.subplots(num=1, clear=True,
        nrows=3, ncols=1, sharex=True, sharey=False,
        figsize=(24.*cm, 16.5*cm), constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=3.*cm, w_pad=3.*cm)

    if PlotAltDiff:
        ax[0].plot(
            prof_dist[:],
            site_alt[:]-(site_gps[:] - site_dem[:]),"r-")
        ax[0].set_title("Altitude Differences", fontsize=Fsize, y=1.0, pad=-(Fsize+2))

    else:
        ax[0].plot(
            prof_dist[:], site_alt[:], "r",
            prof_dist[:], site_gps[:] - site_dem[:], "g:")
        ax[0].set_title("Altitudes", fontsize=Fsize, y=1.0, pad=-(Fsize+2))
        ax[0].legend([" Radar alt", "GPS alt"], fontsize=Lsize, loc="best")

        if HLimits:
            ax[0].set_ylim(HLimits)

    ax[0].set_title("Altitude", fontsize=Fsize, y=1.0, pad=-(Fsize+2))
    ax[0].set_ylabel("(m)", fontsize=Fsize)
    ax[0].grid(True)
    ax[0].tick_params(labelsize=Lsize)


    if LogPlot:
        if not LogSym:
            XData[XData<=numpy.min([X_minpos, Z_minpos])] = numpy.nan
            ZData[ZData<=numpy.min([X_minpos, Z_minpos])] = numpy.nan

    ax[1].plot(prof_dist[:], XData[:,:], linewidth=Linewidths[0])
    ax[1].set_title("In-Line", fontsize=Fsize, y=1.0, pad=-(Fsize+2))
    # ax[1].set_ylim([X_minpos, X_max])
    ax[1].set_ylabel("(ppm)", fontsize=Fsize)
    ax[1].grid(True)

    ax[1].tick_params(labelsize=Lsize)
    leg1 = ax[1].legend(
        [r"0.009 ms", r"0.026 ms", r"0.052 ms", r"0.095 ms",
          r"0.156 ms", r"0.243 ms", r"0.365 ms", r"0.547 ms",
          r"0.833 ms", r"1.259 ms", r"1.858 ms"],
        fontsize=Lsize-3, loc="best",ncol=3)
    leg1.set_title("Window (center)", prop={'size':Lsize})

    if LogPlot:
        if LogSym:
            ax[1].set_yscale("symlog", linthresh=LinThresh)
        else:
            ax[1].set_yscale("log")
    else:
        ax[1].set_yscale("linear")


    ax[2].plot(prof_dist[:], ZData[:,:], linewidth=Linewidths[0])
    ax[2].set_title("Vertical", fontsize=Fsize, y=1.0, pad=-(Fsize+2))
    # ax[2].set_ylim([Z_minpos, Z_max])
    ax[2].set_ylabel("(ppm)", fontsize=Fsize)
    ax[2].set_xlabel("Profile distance "+ProfUnit, fontsize=Lsize)
    ax[2].grid(True)
    ax[2].tick_params(labelsize=Lsize)
    leg2 = ax[2].legend(
        [r"0.009 ms", r"0.026 ms", r"0.052 ms", r"0.095 ms",
          r"0.156 ms", r"0.243 ms", r"0.365 ms", r"0.547 ms",
          r"0.833 ms", r"1.259 ms", r"1.858 ms"],
        fontsize=Lsize-3, loc="best",ncol=3,title="Window (center)")
    leg2.set_title("Window (center)", prop={'size':Lsize-2})
    if LogPlot:
        if LogSym:
            ax[2].set_yscale("symlog", linthresh=LinThresh)
        else:
            ax[2].set_yscale("log")
    else:
        ax[2].set_yscale("linear")

    fig.suptitle("Flightline: "+PlotName+PlotStrng, x=0.5, y=0.94, fontsize=Tsize, ha='center')



    for F in PlotFmt:
        plt.savefig(PlotDir+PlotName+F, dpi=400)

    plt.show()
    plt.clf()


# def plot_model(
#         PlotName = None,
#         PlotDir="./",
#         PlotFmt = ["png"],
#         Backend = None,
#         Model=None,
#         Sens = None,
#         positions=None,
#         XLimits =[],
#         ZLimits =[],
#         ProfUnit  = "(m)",
#         Colors=None,
#         Linetypes=["-", ":",],
#         Linewidths=[1,2],
#         Fontsizes=[12,10,12],
#         Grey=[0.7],
#         Logparams=[True,True,0,1],
#         PlotStrng="",
#         PlotAltDiff=False):

# #    _, Fsize, Lsize, Lwidth, Lcycle, Greyval=get_plot_params()


#     Fsize = Fontsizes[0]
#     Lsize = Fontsizes[1]
#     Tsize = Fontsizes[2]
#     Lwidth = Linewidths[0]
#     Ltype  = Linetypes[0]
#     Greyval = Grey[0]

#     LogPlot= Logparams[0]
#     LogSym = Logparams[1]
#     LinThresh = Logparams[2]


#     fline = DataObs[:, 0]

#     site_x = DataObs[:, 1]
#     site_y = DataObs[:, 2]

#     site_gps = DataObs[:, 3]
#     site_alt = DataObs[:, 4]
#     site_dem = DataObs[:, 5]
#     XData = DataObs[:, 6:17]
#     ZData = DataObs[:, 17:28]

#     # Name = str(fline[0])

#     sx = site_x-site_x[0]
#     sy = site_y-site_y[0]
#     prof_dist = numpy.sqrt(sx * sx + sy * sy)
#     prof_min, prof_max = numpy.min(prof_dist), numpy.max(prof_dist)

#     X_min, X_max = numpy.min(XData), numpy.max(XData)
#     Z_min, Z_max = numpy.min(ZData), numpy.max(ZData)
#     X_minpos = numpy.min(XData[XData>0.])
#     Z_minpos = numpy.min(ZData[ZData>0.])

#     if LinThresh == []:
#         LinThresh = numpy.min([Z_minpos])

#     print("Xmin,max = "+str(X_min)+",  "+str(X_max))
#     print("Zmin,max = "+str(Z_min)+",  "+str(Z_max))
#     print("Xminpos,Zminpos  = "+str(X_minpos)+",  "+str(Z_minpos))
#     print("LinThresh = "+str( LinThresh))

#     tmp          = numpy.load(filename)
#     site_model   = tmp['site_model']
#     site_error   = tmp['site_error']
#     site_sens    = tmp['site_sens']
#     site_num     = tmp['site_num']
#     site_num     = abs(site_num)
#     site_data    = tmp['site_data']
#     site_rms     = site_data[:,0]
#     site_conv    = site_num/numpy.abs(site_num)
#     site_conv[0] = 1.

#     m_active     = tmp['m_active']
#     nlyr         = tmp['nlyr']

#     site_x       = tmp['site_x']
#     site_y       = tmp['site_y']
#     site_alt     = tmp['site_alt']
#     site_gps     = tmp['site_gps']


#     site_ref     = site_alt + 57.0            # Geoid Height is approximately 57.0 for Ireland
#     site_topo    = site_gps - site_ref        # Assuming that Digital Terrain Elevation is not distributed by GSI


#     models=numpy.shape(site_model)
#     sites=models[0]
#     param=models[1]
#     site_val=site_model[:,m_active==1]
#     site_thk=site_model[:,6*nlyr:7*nlyr-1]
#     site_r=numpy.sqrt(numpy.power(site_x,2.) +numpy.power(site_y,2.))
#     site_r=site_r-site_r[0]
#     site_sens=site_sens[:,m_active==1]
#     scale=numpy.max(numpy.abs(site_sens.flat))
#     site_sens=site_sens/scale
#     nlay = nlyr
#     dxmed2 = numpy.median(numpy.diff(site_r)) / 2.




#     fig, ax = plt.subplots(figsize=(15, 8))

#     max_topo = max(site_topo)

#     if low_sens:
#     ##   site_val[numpy.abs(site_sens) < lowsens]=numpy.nan
#        nmod=0
#        plotmin = max_topo
#        while nmod < sites:
#            invalid=[]
#            thk = site_thk[nmod,:]
#            thk_halfspace =  thk[-1]
#            thk = numpy.hstack((thk.T, thk_halfspace))
#            z0 = numpy.hstack((0., numpy.cumsum(thk)))

#            zm= 0.5*(z0[0:nlay]+z0[1:nlay+1])
#            invalid = numpy.abs(site_sens[nmod,:])<lowsens
#            site_val[nmod,invalid]=numpy.nan
#            zm = site_topo[nmod] -zm
#            if any(invalid) :
#               plotmin=numpy.min([plotmin,numpy.min(zm[invalid])])
#            nmod=nmod+1
#        plotmin1 = plotmin
#     ## print(plotmin1)

#     if max_depth:
#        nmod=0
#        plotmin =max_topo
#        while nmod < sites:
#            invalid=[]
#            thk = site_thk[nmod,:]
#            thk_halfspace =  thk[-1]
#            thk = numpy.hstack((thk.T, thk_halfspace))
#            z0 = numpy.hstack((0., numpy.cumsum(thk)))
#            zm = 0.5*(z0[0:nlay]+z0[1:nlay+1])
#            zm = numpy.hstack((zm,zm[nlay-1]))
#            invalid = zm>maxdepth
#            site_val[nmod,invalid[0:nlay]]=numpy.nan
#            zm = site_topo[nmod] -zm
#            if any(invalid) :
#               plotmin=numpy.min([plotmin,numpy.min(zm[invalid])])
#            nmod=nmod+1
#        plotmin2 = plotmin


#     site_val = numpy.ma.masked_invalid(site_val)
#     val=numpy.zeros(numpy.shape(site_val))

#     patches = []
#     nmod = 0
#     while nmod < sites:
#         if high_rms and site_rms[nmod]<highrms:
#            val[nmod, :] = site_val[nmod,:]
#            thk = site_thk[nmod,:]
#            thk_halfspace =  thk[-1]
#            thk = numpy.hstack((thk.T, thk_halfspace))
#            z0 = numpy.hstack((0., numpy.cumsum(thk)))
#            z = site_topo[nmod]-z0
#            for j in range(nlay):
#                rect = Rectangle((site_r[nmod] - dxmed2, z[j]), dxmed2 * 2, thk[j])
#                patches.append(rect)
#         nmod=nmod+1

#     p = PatchCollection(patches, cmap=plt.get_cmap(mycmap) , linewidths=0,edgecolor=None)
#     p.set_clim(sl)
#     p.set_array(site_val.ravel())
#     ax.add_collection(p)

#     if plot_depth_adapt:
#        plotmin = numpy.max([plotmin1,plotmin2])
#     else:
#        plotmin = plotdepth

#     if plot_height_adapt:
#        plotmax = max_topo
#     else:
#        plotmax = plotheight


#     ax.set_ylim((plotmax+blank,plotmin-blank))
#     ax.set_xlim((min(site_r) - dxmed2, max(site_r) + dxmed2))
#     ax.invert_yaxis()
#     #ax.axis('equal')
#     #    if title is not None:
#     ax.set_title(title, fontsize=FSize)
#     ax.set_ylabel('elevation (m)', fontsize=FSize)
#     ax.yaxis.set_label_position("right")
#     ax.set_xlabel(' profile distance (m)', fontsize=FSize)
#     ax.tick_params(labelsize=FSize)
#     #ax.grid(b='on')
#     #
#     #    pg.mplviewer.createColorbar(p, cMin=cmin, cMax=cmax, nLevs=5)
#     #
#     cb = plt.colorbar(p, orientation='horizontal',aspect=35,pad=0.15)
#     xt = [-0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3.0, 3.5]
#     cb.set_ticks( xt, [str(xti) for xti in xt] )
#     cb.set_label('$\log_{10}$ ($\Omega$ m)', size=FSize)
#     cb.ax.tick_params(labelsize=FSize)

#     #    plt.draw()
#     #    return fig, ax
#     print (' PLot to '+FileName+plotformat)
#     fig = plt.savefig(FileName+plotformat,dpi=300)
#         for F in PlotFmt:
#         plt.savefig(PlotDir+PlotName+F, dpi=400)

#     plt.show()
#     plt.clf()

#     # fig, ax = plt.subplots(num=1, clear=True,
#     #     nrows=3, ncols=1, sharex=True, sharey=False, figsize=(24., 16.5)
#     # )
#     # if PlotAltDiff:
#     #     ax[0].plot(
#     #         prof_dist[:],
#     #         site_alt[:]-(site_gps[:] - site_dem[:]),"r-")
#     #     ax[0].set_title("Altitude Differences", fontsize=Fsize)

#     # else:
#     #     ax[0].plot(
#     #         prof_dist[:], site_alt[:], "r",
#     #         prof_dist[:], site_gps[:] - site_dem[:], "g:")
#     #     ax[0].set_title("Altitudes", fontsize=Fsize)
#     #     ax[0].legend([" Radar alt", "GPS alt"], fontsize=Lsize, loc="best")

#     # ax[0].set_title("Altitude", fontsize=Fsize)
#     # ax[0].set_ylabel("(m)", fontsize=Fsize)
#     # ax[0].grid(True)
#     # ax[0].tick_params(labelsize=Lsize)


#     # if LogPlot:
#     #     if not LogSym:
#     #         XData[XData<=numpy.min([X_minpos, Z_minpos])] = numpy.nan
#     #         ZData[ZData<=numpy.min([X_minpos, Z_minpos])] = numpy.nan

#     # ax[1].plot(prof_dist[:], XData[:,:])
#     # ax[1].set_title("In-Line", fontsize=Fsize)
#     # # ax[1].set_ylim([X_minpos, X_max])
#     # ax[1].set_ylabel("(ppm)", fontsize=Fsize)
#     # ax[1].grid(True)

#     # ax[1].tick_params(labelsize=Lsize)
#     # ax[1].legend(
#     #     [r"0.009 ms", r"0.026 ms", r"0.052 ms", r"0.095 ms",
#     #       r"0.156 ms", r"0.243 ms", r"0.365 ms", r"0.547 ms",
#     #       r"0.833 ms", r"1.259 ms", r"1.858 ms"],
#     #     fontsize=Lsize-4, loc="best",ncol=3,title="Window (center)")

#     # if LogPlot:
#     #     if LogSym:
#     #         ax[1].set_yscale("symlog", linthresh=LinThresh)
#     #     else:
#     #         ax[1].set_yscale("log")
#     # else:
#     #     ax[1].set_yscale("linear")


#     # ax[2].plot(prof_dist[:], ZData[:,:])
#     # ax[2].set_title("Vertical", fontsize=Fsize)
#     # # ax[2].set_ylim([Z_minpos, Z_max])
#     # ax[2].set_ylabel("(ppm)", fontsize=Fsize)
#     # ax[2].set_xlabel("Profile distance "+ProfUnit, fontsize=Fsize)
#     # ax[2].grid(True)
#     # ax[2].tick_params(labelsize=Lsize)
#     # ax[2].legend(
#     #     [r"0.009 ms", r"0.026 ms", r"0.052 ms", r"0.095 ms",
#     #       r"0.156 ms", r"0.243 ms", r"0.365 ms", r"0.547 ms",
#     #       r"0.833 ms", r"1.259 ms", r"1.858 ms"],
#     #     fontsize=Lsize-4, loc="best",ncol=3,title="Window (center)")

#     # if LogPlot:
#     #     if LogSym:
#     #         ax[2].set_yscale("symlog", linthresh=LinThresh)
#     #     else:
#     #         ax[2].set_yscale("log")
#     # else:
#     #     ax[2].set_yscale("linear")

#     # fig.suptitle("Flightline: "+PlotName+PlotStrng, x=0.5, y=0.94, fontsize=Tsize, ha='center')
#     # plt.tight_layout()


#     for F in PlotFmt:
#         plt.savefig(PlotDir+PlotName+F, dpi=400)

#     plt.show()
#     plt.clf()


def make_pdf_catalog(WorkDir="./", PdfList= None, FileName=None):
    """
    Make pdf catalog from site-plots

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

    error("not in 3.9! Exit")

    import fitz

    catalog = fitz.open()

    for pdf in PdfList:
        with fitz.open(pdf) as mfile:
            catalog.insert_pdf(mfile)

    catalog.save(FileName, garbage=4, clean = True, deflate=True)
    catalog.close()

    print("\n"+str(numpy.size(PdfList))+" files collected to "+FileName)
