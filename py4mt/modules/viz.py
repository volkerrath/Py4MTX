# -*- coding: utf-8 -*-
"""
viz.py
======
Matplotlib visualization functions for MT (magnetotelluric) data and models.

Provides plotters for:
- Impedance tensor components (``plot_impedance``)
- Apparent resistivity and phase (``plot_rhophas``)
- Phase tensor elements (``plot_phastens``)
- Vertical transfer functions / tipper (``plot_vtf``)
- Sparse matrix sparsity patterns (``plot_sparsity_pattern``)
- Vertical cross-section planes (``plot_plane_cross``)
- PDF catalog generation (``make_pdf_catalog``)

All plotters accept ``**pltargs`` keyword arguments for flexible
style customization (colors, markers, font sizes, axis limits, etc.).

Author: Volker Rath (DIAS)
Created: 2020-12-27
"""

import os
import sys

from time import process_time
from datetime import datetime
import warnings
import inspect

import math

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.cm as cm
import mpl_toolkits.axes_grid1
import matplotlib.ticker
import cycler

import scipy.sparse as scs


import util as utl


def plot_impedance(thisaxis=None, data=None, **pltargs):
    '''
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

    '''

    if thisaxis is None:
        fig, ax = plt.subplots(1, figsize=pltargs['pltsize'])
        # fig.suptitle(pltargs['title'], fontsize=pltargs['fontsizes'][2])
    else:
        ax = thisaxis

    [points, dats] = np.shape(data)

    per = data[:, 0]
    obs_real = data[:, 1]
    obs_imag = data[:, 2]

    plot_err = False
    if dats > 3:
        err_real = data[:, 3]
        err_imag = data[:, 4]
        plot_err = True

    plot_cal = False
    if dats > 6:
        cal_real = data[:, 5]
        cal_imag = data[:, 6]
        plot_cal = True

    if plot_cal:
        ax.plot(per, cal_real,
                color=pltargs['c_cal'][0],
                linestyle=pltargs['l_cal'][0],
                inewidth=pltargs['l_cal'][1])

    if plot_err:
        ax.errorbar(per,
                    obs_real,
                    yerr=err_real,
                    linestyle=pltargs['l_obs'][0],
                    marker=pltargs['m_obs'][0],
                    markersize=pltargs['m_size'],
                    color=pltargs['c_obs'][0],
                    linewidth=pltargs['l_obs'][1],
                    capsize=2, capthick=0.5)
    else:
        ax.plot(per,
                obs_real,
                linestyle=pltargs['l_cal'][0],
                marker=pltargs['m_obs'][0],
                markersize=pltargs['m_size'],
                linewidth=pltargs['l_obs'][1],
                color=pltargs['c_obs'][0])

    if plot_cal:
        ax.plot(per, cal_imag,
                color=pltargs['c_cal'][0],
                linestyle=pltargs['l_cal'][0],
                linewidth=pltargs['l_cal'][1])

    if plot_err:
        ax.errorbar(per,
                    obs_imag,
                    yerr=err_imag,
                    linestyle=pltargs['l_obs'][0],
                    marker=pltargs['m_obs'][1],
                    markersize=pltargs['m_size'],
                    color=pltargs['c_obs'][1],
                    linewidth=pltargs['l_obs'][1],
                    capsize=2, capthick=0.5)
    else:
        ax.plot(per,
                obs_imag,
                linestyle=pltargs['l_obs'][0],
                marker=pltargs['m_obs'][1],
                markersize=pltargs['m_size'],
                linewidth=pltargs['l_obs'][1],
                color=pltargs['c_obs'][1])

    ax.set_xscale('log')
    ax.set_xlabel('period [s]', fontsize=pltargs['fontsizes'][1])
    ax.set_yscale(pltargs['yscale'])
    ax.set_ylabel(pltargs['ylabel'], fontsize=pltargs['fontsizes'][1])
    # ax.set_ylabel(r'impedance [$\Omega$]',fontsize=pltargs['fontsizes'][1])
    if len(pltargs['xlimits']) != 0:
        ax.set_xlim(pltargs['xlimits'])
    if len(pltargs['ylimits']) != 0:
        ax.set_ylim(pltargs['ylimits'])
    ax.legend(pltargs['legend'], fontsize=pltargs['fontsizes'][2])
    # ax.xaxis.set_ticklabels([])
    ax.tick_params(labelsize=pltargs['fontsizes'][1])
    ax.set_title(pltargs['title'], fontsize=pltargs['fontsizes'][2])
    ax.grid('both', 'both', linestyle='-', linewidth=0.5)
    if len(pltargs['nrms']) == 2:
        nrmsr = np.around(pltargs['nrms'][0], 1)
        nrmsi = np.around(pltargs['nrms'][1], 1)
        strrms = 'nrms = ' + str(nrmsr) + ' | ' + str(nrmsi)
        ax.text(0.05, 0.05, strrms,
                transform=ax.transaxes,
                fontsize=pltargs['fontsizes'][1],
                ha='left', va='bottom',
                bbox={'pad': 2, 'facecolor': 'white', 'edgecolor': 'white', 'alpha': 0.8})

    if thisaxis is None:
        for f in pltargs['pltformat']:
            plt.savefig(pltargs['pltfile'] + f)
        # plt.show()
        # plt.clf()

    return ax


def plot_rhophas(thisaxis=None, data=None, **pltargs):
    '''
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

    '''

    if thisaxis is None:
        fig, ax = plt.subplots(1, figsize=pltargs['pltsize'])
        # fig.suptitle(pltargs['title'], fontsize=pltargs['fontsizes'][2])
    else:
        ax = thisaxis

    [points, dats] = np.shape(data)

    per = data[:, 0]
    obs_rhoa = data[:, 1]
    obs_phas = data[:, 2]

    plot_err = False
    if dats > 3:
        err_rhoa = data[:, 3]
        err_phas = data[:, 4]
        plot_err = True

    plot_cal = False
    if dats > 6:
        cal_rhoa = data[:, 5]
        cal_phas = data[:, 6]
        plot_cal = True

    if plot_cal:
        ax.plot(per, cal_rhoa,
                color=pltargs['c_cal'][0], linestyle=pltargs['l_cal'][0], linewidth=pltargs['l_cal'][1])

    if plot_err:
        ax.errorbar(per,
                    obs_rhoa,
                    yerr=err_rhoa,
                    linestyle=pltargs['l_obs'][0],
                    marker=pltargs['m_obs'][0],
                    markersize=pltargs['m_size'],
                    color=pltargs['c_obs'][0],
                    linewidth=pltargs['l_obs'][1],
                    capsize=2, capthick=0.5)
    else:
        ax.plot(per,
                obs_rhoa,
                linestyle=pltargs['l_obs'][0],
                marker=pltargs['m_obs'][0],
                markersize=pltargs['m_size'],
                linewidth=pltargs['l_obs'][1],
                color=pltargs['c_obs'][0])

    if plot_cal:
        ax.plot(per, cal_phas,
                color=pltargs['c_cal'][1], linestyle=pltargs['l_cal'][0], linewidth=pltargs['l_cal'][1])

    if plot_err:
        ax.errorbar(per,
                    obs_phas,
                    yerr=err_phas,
                    linestyle=pltargs['l_obs'][0],
                    marker=pltargs['m_obs'][1],
                    markersize=pltargs['m_size'],
                    color=pltargs['c_obs'][1],
                    linewidth=pltargs['l_obs'][1],
                    capsize=2, capthick=0.5)
    else:
        ax.plot(per,
                obs_phas,
                linestyle=pltargs['l_obs'][0],
                marker=pltargs['m_obs'][1],
                markersize=pltargs['m_size'],
                linewidth=pltargs['l_obs'][1],
                color=pltargs['c_obs'][1])

    ax.set_xscale('log')
    ax.set_xlabel('period [s]', fontsize=pltargs['fontsizes'][1])
    ax.set_yscale(pltargs['yscale'])
    ax.set_ylabel(pltargs['ylabel'], fontsize=pltargs['fontsizes'][1])
    if len(pltargs['xlimits']) != 0:
        ax.set_xlim(pltargs['xlimits'])
    if len(pltargs['ylimits']) != 0:
        ax.set_ylim(pltargs['ylimits'])
    ax.legend(pltargs['legend'], fontsize=pltargs['fontsizes'][3])

    # ax.xaxis.set_ticklabels([])
    ax.tick_params(labelsize=pltargs['fontsizes'][0])
    ax.set_title(pltargs['title'], fontsize=pltargs['fontsizes'][2])
    ax.grid('both', 'both', linestyle='-', linewidth=0.5)
    if len(pltargs['nrms']) == 2:
        nrmsr = np.around(pltargs['nrms'][0], 1)
        nrmsi = np.around(pltargs['nrms'][1], 1)
        strrms = 'nrms = ' + str(nrmsr) + ' | ' + str(nrmsi)
        ax.text(0.05, 0.05, strrms,
                transform=ax.transaxes,
                fontsize=pltargs['fontsizes'][1],
                ha='left', va='bottom',
                bbox={'pad': 2, 'facecolor': 'white', 'edgecolor': 'white', 'alpha': 0.8})

    if thisaxis is None:
        for f in pltargs['pltformat']:
            plt.savefig(pltargs['pltfile'] + f)
        # plt.show()
        # plt.clf()

    return ax


def plot_phastens(thisaxis=None, data=None, **pltargs):
    '''
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

    '''

    if thisaxis is None:
        fig, ax = plt.subplots(1, 1, figsize=pltargs['pltsize'])
        # fig.suptitle(pltargs['title'], fontsize=pltargs['fontsizes'][2])
    else:
        ax = thisaxis

    [points, dats] = np.shape(data)

    per = data[:, 0]

    ax.semilogx(per, data[:, 1],
                color=pltargs['c_obs'][0],
                linestyle=pltargs['l_obs'][0],
                linewidth=pltargs['l_obs'][1],
                marker=pltargs['m_obs'][0],
                markersize=pltargs['m_size'])

    ax.semilogx(per, data[:, 2],
                color=pltargs['c_obs'][1],
                linestyle=pltargs['l_obs'][0],
                linewidth=pltargs['l_obs'][1],
                marker=pltargs['m_obs'][1],
                markersize=pltargs['m_size'])

    ax.set_ylabel('phase tensor [-]')
    ax.set_yscale('linear')
    ax.set_xscale('log')

    if len(pltargs['xlimits']) != 0:
        ax.set_xlim(pltargs['xlimits'])
    if len(pltargs['ylimits']) != 0:
        ax.set_ylim(pltargs['ylimits'])

    ax.legend(pltargs['legend'], fontsize=pltargs['fontsizes'][3])
    # ax.xaxis.set_ticklabels([])
    ax.tick_params(labelsize=pltargs['fontsizes'][1])
    ax.set_title(pltargs['title'], fontsize=pltargs['fontsizes'][2])
    ax.grid('both', 'both', linestyle='-', linewidth=0.5)
    if len(pltargs['nrms']) == 2:
        nrmsr = np.around(pltargs['nrms'][0], 1)
        nrmsi = np.around(pltargs['nrms'][1], 1)
        strrms = 'nrms = ' + str(nrmsr) + ' | ' + str(nrmsi)
        ax.text(0.05, 0.05, strrms,
                transform=ax.transaxes,
                fontsize=pltargs['fontsizes'][0],
                ha='left', va='bottom',
                bbox={'pad': 2, 'facecolor': 'white', 'edgecolor': 'white', 'alpha': 0.8})

    if thisaxis is None:
        for f in pltargs['pltformat']:
            plt.savefig(pltargs['pltfile'] + f)
        # plt.show()
        # plt.clf()

    return ax


def plot_vtf(thisaxis=None, data=None, **pltargs):
    '''


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

    '''

    if thisaxis is None:
        fig, ax = plt.subplots(1, figsize=pltargs['figsize'])
        fig.suptitle(pltargs['suptitle'], fontsize=pltargs['fontsizes'][2])
    else:
        ax = thisaxis

    if thisaxis is None:
        for f in pltargs['plotformat']:
            plt.savefig(pltargs['plotfile'] + f)

    return ax

def plot_sparsity_pattern(
        plotfile='',
        plottitle=r'$\mathbf{m}$, sparsity pattern',
        plotformat=['png', '.pdf'],
        figsize=[8.5 * 0.3937, 8.5 * 0.3937],
        matrix=[],
        plotstrng='',
        plotstrngpos=[0.05, 0.05],
        aspect='auto'):

    from scipy.sparse import csr_array, csc_array, coo_array, eye_array, issparse
    from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
    from matspy import spy_to_mpl, spy

    import femtic as fem

    m = coo_matrix(matrix)
    fem.check_sparse_matrix(m)

    # plotting
    options = {'title': plottitle,
               'figsize': 8.,  # inches
               'dpi': 600,
               'shading': 'binary',  # 'absolute' 'relative' 'binary'
               'spy_aa_tweaks_enabled': True,
               'color_full': 'black'}

    fig, ax = spy_to_mpl(m, **options)
    fig.text(plotstrngpos[0], plotstrngpos[1], plotstrng)
    fig.show()

    for fmt in plotformat:
        fig.savefig(plotfile + plotstrng + fmt, bbox_inches='tight')

    plt.close()


def plot_plane_cross(ax, position,
                     tensor,
                     plane,
                     colors=('red', 'blue', 'green'),
                     scale=1.0):

    idx = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}[plane]
    i, j = idx
    block = tensor[[i, j], :][:, [i, j]]
    eigvals, eigvecs = np.linalg.eigh(block)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    # max_ev = np.max(np.abs(eigvals))
    for k in range(2):
        v = eigvecs[:, k]
        mag = eigvals[k] * scale
        x_coords = np.array([-v[0] * mag, v[0] * mag])
        y_coords = np.array([-v[1] * mag, v[1] * mag])
        ax.plot(x_coords, y_coords,
                color=colors[k],
                linewidth=3,
                solid_capstyle='butt')


#
# def plot_model_ensemble(

    #     ThisAxis = None,
    #     PlotFile = '',
    #     PlotFormat = ['png',],
    #     PlotTitle = '',
    #     PlotType = ['lines'], # lines, Percentiles. iso
    #     PlotSize = [8.],
    #     System  = 'aem05',
    #     ModEns = [],
    #     Depth = [],
    #     Percentiles=[2.5, 16.],
    #     Quantiles = [.025, .16],
    #     Fillcolor=['0.8', '0.4'],
    #     Alphas = [0.3 , 0.6],
    #     Labels=[],
    #     Linecolor=['k', 'r', 'g', 'b', 'y', 'm'],
    #     Linetype=['-', ':', ';'],
    #     Linewidth=[1., 1.5, 2.],
    #     Markers = ['v'],
    #     Markersize =[4],
    #     Fontsizes=[10,10,12],
    #     XLimits= [],
    #     YLimits=[],
    #     PlotStrng='',
    #     Invalid=1.e30,
    #     Maxlines=50,
    #     Median=True,
    #     Legend=True):

    # '''

    # '''
    # cm = 1/2.54  # centimeters to inches

    # ax = ThisAxis

    # if np.size(ModEns)==0:
    #    sys.exit('No parameter ensemble given!! Exit.')
    #    # nopar=True

    # if ThisAxis is None:
    #     nplots = 1
    #     fig, ax = plt.subplots(1,
    #                                       figsize=(PlotSize[0]*cm, nplots*PlotSize[0]*cm),
    #                                       gridspec_kw={'height_ratios': [1]})
    #     fig.suptitle(PlotTitle, fontsize=Fontsizes[2])
    #     Legend = True

    # if 'per' in PlotType.lower():
    #     nperc = np.size(Percentiles)
    #     medval=np.percentile(ModEns, 50., axis=0)

    #     for p in np.arange(nperc):
    #             mlow = np.percentile(ModEns,      Percentiles[p], axis=0)
    #             mupp = np.percentile(ModEns, 100.-Percentiles[p], axis=0)
    #             plabel = None
    #             ax.fill_betweenx(Depth, mlow, mupp, step='post',
    #                                linewidth=Linewidth[0],
    #                                color= Fillcolor[p],
    #                                alpha= Alphas[p],
    #                                label=plabel)

    #     ax.step(medval, Depth,
    #         linewidth=Linewidth[0], color= Linecolor[1],
    #         label='medval')

    #     for p in np.arange(nperc):
    #         plabel = 'p='\
    #             +str(100.-Percentiles[p]-Percentiles[p])+' %'
    #         mlow = np.percentile(ModEns,      Percentiles[p], axis=0)
    #         mupp = np.percentile(ModEns, 100.-Percentiles[p], axis=0)
    #         ax.step(mlow, Depth, where='pre',
    #                 linewidth=Linewidth[0]/2, color= Linecolor[p+2])
    #         ax.step(mupp, Depth, where='pre',
    #                 linewidth=Linewidth[0]/2, color= Linecolor[p+2],
    #                 label=plabel)

    # if 'qua' in PlotType.lower():
    #     nperc = np.size(Quantiles)
    #     medval=np.quantile(ModEns, 0.5, axis=0)

    #     for p in np.arange(nperc):
    #             mlow = np.quantile(ModEns,      Quantiles[p], axis=0)
    #             mupp = np.quantile(ModEns, 1.-Quantiles[p], axis=0)
    #             plabel = None
    #             ax.fill_betweenx(Depth, mlow, mupp, step='post',
    #                                linewidth=Linewidth[0],
    #                                color= Fillcolor[p],
    #                                alpha= Alphas[p],
    #                                label=plabel)

    #     ax.step(medval, Depth,
    #         linewidth=Linewidth[0], color= Linecolor[1],
    #         label='medval')

    #     for p in np.arange(nperc):
    #         plabel = 'q='\
    #             +str(1.-Quantiles[p]-Quantiles[p])
    #         mlow = np.quantile(ModEns,      Quantiles[p], axis=0)
    #         mupp = np.quantile(ModEns, 1.-Quantiles[p], axis=0)
    #         ax.step(mlow, Depth, where='pre',
    #                 linewidth=Linewidth[0]/2, color= Linecolor[p+2])
    #         ax.step(mupp, Depth, where='pre',
    #                 linewidth=Linewidth[0]/2, color= Linecolor[p+2],
    #                 label=plabel)

    # elif 'lin' in PlotType.lower():

    #     nens = np.shape(ModEns)
    #     if nens[0] > Maxlines:
    #         sys.exit('plot_model_ensemble: too many lines! Exit.')

    #     medval=np.percentile(ModEns, 50., axis=0)

    #     for ne in np.arange(nens):
    #         plabel = None

    #         ax.step(ModEns[1], Depth,
    #                       where='pre',
    #                       linewidth=Linewidth[0]/2,
    #                       color= Fillcolor[ne],
    #                       alpha= Alphas[ne],
    #                       label=plabel)

    #     # ax.step(medval, Depth,step='pre',
    #     #     linewidth=Linewidth[0]+2, color= Linecolor[1],
    #     #     label='medval')

    # ax.set_xscale('log')
    # ax.set_xlim(XLimits)
    # ax.set_xlabel('resistivity ($\Omega$m)',fontsize=Fontsizes[0])
    # ax.set_ylim(YLimits)
    # ax.set_ylabel('depth (m)',fontsize=Fontsizes[0])
    # ax.xaxis.set_label_position('top')
    # ax.xaxis.set_ticks_position('both')
    # ax.tick_params(labelsize=Fontsizes[1])

    # ax.grid(True)
    # ax.invert_yaxis()
    # ax.grid('major', 'both', linestyle=':', lw=0.3)

    # if Legend:
    #     ax.legend(fontsize=Fontsizes[0], loc='best')

    # if ThisAxis is None:
    #     for F in PlotFormat:
    #         plt.savefig(PlotFile+F)
    #     # plt.show()
    #     # plt.clf()

    # return ax


#     if 'aem05' in System.lower():
#         XLabel = 'frequency (kHz)'
#         YLabel = 'Q/I (ppm)'
#         XAxis = Pars[0]/1000.


#         nsmp,ndat = np.shape(DatEns)

#         Qens =   DatEns[:,0:4]
#         Iens =   DatEns[:,4:8]

#         print(np.shape(XAxis), np.shape(Qens))

#         if ('per' in PlotType.lower()) or ('qua' in PlotType.lower()):


#             for p in np.arange(np):


#                 if 'per' in PlotType.lower():
#                     medQens = np.percentile(Qens, 50., axis=0)
#                     plabel = 'p='\
#                         +str(100.-Percentiles[p]-Percentiles[p])+' %'

#                     dlow = np.percentile(Qens,      Percentiles[p], axis=0)
#                     dupp = np.percentile(Qens, 100.-Percentiles[p], axis=0)

#                 else:
#                     medQens = np.percentile(Qens, 0.5, axis=0)
#                     plabel = 'p='\
#                         +str(1.-Quantiles[p]-Quantiles[p])+' %'

#                     dlow = np.quantile(Qens,    Quantiles[p], axis=0)
#                     dupp = np.quantile(Qens, 1.-Quantiles[p], axis=0)


#                 ax.fill_between(XAxis, dlow, dupp,
#                                         linewidth=Linewidth[0]/2,
#                                         color= Fillcolor[p],
#                                         alpha= Alphas[p],
#                                         label=None)

#                 ax.plot(XAxis, dlow,
#                         linewidth=Linewidth[0]/2, color= Linecolor[p+2])
#                 ax.plot(XAxis, dupp,
#                         linewidth=Linewidth[0]/2, color= Linecolor[p+2])


#             if Median:
#                 ax.plot(XAxis, medQens,
#                         linewidth=Linewidth[0], color= Linecolor[0], linestyle=Linetype[2],
#                         label='medval')


#             for p in np.arange(np):

#                 if 'per' in PlotType.lower():
#                     medIens = np.percentile(Iens, 50., axis=0)
#                     plabel = 'p='\
#                         +str(100.-Percentiles[p]-Percentiles[p])+' %'

#                     dlow = np.percentile(Iens,      Percentiles[p], axis=0)
#                     dupp = np.percentile(Iens, 100.-Percentiles[p], axis=0)

#                 else:
#                     medIens = np.percentile(Iens, 0.5, axis=0)
#                     plabel = 'p='\
#                         +str(1.-Quantiles[p]-Quantiles[p])+' %'

#                     dlow = np.quantile(Iens,    Quantiles[p], axis=0)
#                     dupp = np.quantile(Iens, 1.-Quantiles[p], axis=0)

#                 ax.fill_between(XAxis, dlow, dupp,
#                                         linewidth=Linewidth[0]/2,
#                                         color= Fillcolor[p],
#                                         alpha= Alphas[p],
#                                         label=None)


#                 ax.plot(XAxis, dlow,
#                         linewidth=Linewidth[0]/2, color= Linecolor[p+2])
#                 ax.plot(XAxis, dupp,
#                         linewidth=Linewidth[0]/2, color= Linecolor[p+2],
#                         label=plabel)
#             if Median:
#                 ax.plot(XAxis, medIens,
#                         linewidth=Linewidth[0], color= Linecolor[0], linestyle=Linetype[2],
#                         label=None)


#         elif 'lin' in PlotType.lower():

#             nens = np.shape(Qens)
#             if nens[0] > Maxlines:
#                 sys.exit('plot_data_ensemble: too many lines! Exit.')

#             plabel = None
#             medQens = np.percentile(Qens, 50., axis=0)
#             ax.plot(XAxis, Qens,
#                         linewidth=Linewidth[0]/2,
#                         color= Linecolor[0], alpha=0.5,
#                         label=plabel)

#             if Median:
#                 ax.plot(XAxis, medQens,
#                     linewidth=Linewidth[0], color= Linecolor[2], linestyle=Linetype[2],
#                     label='Q, median')

#             medIens = np.percentile(Iens, 50., axis=0)
#             ax.plot(XAxis, Iens,
#                         linewidth=Linewidth[0]/2,
#                         color= Linecolor[0], alpha=0.5,
#                         label=plabel)
#             if Median:
#                 ax.plot(XAxis, medIens,
#                         linewidth=Linewidth[0], color= Linecolor[2], linestyle=Linetype[2],
#                         label='I, median')

#         ax.set_xscale('log')
#         if len(XLimits) !=0:
#             ax.set_xlim(XLimits)
#         ax.set_xlabel(XLabel,fontsize=Fontsizes[0])

#         if len(YLimits) !=0:
#             ax.set_ylim(YLimits)
#         ax.set_ylabel(YLabel,fontsize=Fontsizes[0])

#         ax.xaxis.set_label_position('top')
#         ax.xaxis.set_ticks_position('both')
#         ax.tick_params(labelsize=Fontsizes[1])

#         ax.grid(True)
#         # ax.invert_yaxis()
#         ax.grid('major', 'both', linestyle=':', lw=0.3)

#         if Legend:
#             ax.legend(fontsize=Fontsizes[0], loc='best')


#     elif 'gen' in System.lower():
#         XLabel = 'time (ms)'
#         YLabel = 'ppm (-)'
#         #XAxis = Pars[0]

#         nsmp,ndat = np.shape(DatEns)

#         Hens =   DatEns[:,0:11]
#         Zens =   DatEns[:,11:22]

#         if DataTrans==2:
#            S = np.amin(DatEns)
# #

#            YLabel = 'atanh H/Z (-)'
#            ax.set_yscale('linear')

#         elif DataTrans==1:
#            YLabel = 'H/Z (ppm)'
#            ax.set_yscale('symlog')
#         else:
#            YLabel = 'H/Z (ppm)'
#            ax.set_yscale('log', nonpositive='clip')


#         if ('per' in PlotType.lower()) or ('qua' in PlotType.lower()):


#                 for p in np.arange(np):

#                     if 'per' in PlotType.lower():
#                         medHens = np.percentile(Hens, 50., axis=0)
#                         plabel = 'p='\
#                             +str(100.-Percentiles[p]-Percentiles[p])+' %'

#                         dlow = np.percentile(Hens,      Percentiles[p], axis=0)
#                         dupp = np.percentile(Hens, 100.-Percentiles[p], axis=0)

#                     else:
#                         medHens = np.percentile(Hens, 0.5, axis=0)
#                         plabel = 'p='\
#                             +str(1.-Quantiles[p]-Quantiles[p])

#                         dlow = np.quantile(Hens,    Quantiles[p], axis=0)
#                         dupp = np.quantile(Hens, 1.-Quantiles[p], axis=0)


#                     ax.fill_between(XAxis, dlow, dupp,
#                                             linewidth=Linewidth[0],
#                                             color= Fillcolor[p],
#                                             alpha= Alphas[p],
#                                             label=None)


#                     ax.plot(XAxis, dlow,
#                             linewidth=Linewidth[0]/2, color= Linecolor[p+2])
#                     ax.plot(XAxis, dupp,
#                             linewidth=Linewidth[0]/2, color= Linecolor[p+2])

#                 if Median:
#                     ax.plot(XAxis, medHens,
#                             linewidth=Linewidth[0], color= Linecolor[2], linestyle=Linetype[2],
#                             label='medval')


#                 for p in np.arange(np):

#                     if 'per' in PlotType.lower():
#                         medZens = np.percentile(Zens, 50., axis=0)
#                         plabel = 'p='\
#                             +str(100.-Percentiles[p]-Percentiles[p])+' %'

#                         dlow = np.percentile(Zens,      Percentiles[p], axis=0)
#                         dupp = np.percentile(Zens, 100.-Percentiles[p], axis=0)

#                     else:
#                         medZens = np.percentile(Zens, 0.5, axis=0)
#                         plabel = 'p='\
#                             +str(1.-Quantiles[p]-Quantiles[p])

#                         dlow = np.quantile(Zens,    Quantiles[p], axis=0)
#                         dupp = np.quantile(Zens, 1.-Quantiles[p], axis=0)

#                     plabel = None
#                     ax.fill_between(XAxis, dlow, dupp,
#                                             linewidth=Linewidth[0]/2,
#                                             color= Fillcolor[p],
#                                             alpha= Alphas[p],
#                                             label=None)


#                     ax.plot(XAxis, dlow,
#                             linewidth=Linewidth[0]/2, color= Linecolor[p+2])
#                     ax.plot(XAxis, dupp,
#                             linewidth=Linewidth[0]/2, color= Linecolor[p+2],
#                             label=plabel)

#                 if Median:
#                     ax.plot(XAxis, medZens,
#                             linewidth=Linewidth[0]/2, color= Linecolor[2], linestyle=Linetype[2],
#                             label=None)


#         elif 'lin' in PlotType.lower():

#             nens = np.shape(Hens)
#             if nens[0] > Maxlines:
#                 sys.exit('plot_data_ensemble: too many lines! Exit.')

#             plabel = None
#             medHens = np.percentile(Hens, 50., axis=0)
#             ax.plot(XAxis, Hens,
#                         linewidth=Linewidth[0]/2,
#                         color= Linecolor[0], alpha=0.5,
#                         label=plabel)
#             if Median:
#                 ax.plot(XAxis, medHens,
#                         linewidth=Linewidth[0], color= Linecolor[2], linestyle=Linetype[2],
#                         label='H, median')

#             medZens = np.percentile(Zens, 50., axis=0)
#             ax.plot(XAxis, Zens,
#                         linewidth=Linewidth[0]/2,
#                         color= Linecolor[0], alpha=0.5,
#                         label=plabel)
#             if Median:
#                 ax.plot(XAxis, medZens,
#                         linewidth=Linewidth[0], color= Linecolor[2], linestyle=Linetype[2],
#                         )


#         ax.set_xscale('log')
#         if len(XLimits) !=0:
#             ax.set_xlim(XLimits)
#         ax.set_xlabel(XLabel,fontsize=Fontsizes[0])

#         if len(YLimits) !=0:
#             ax.set_ylim(YLimits)
#         ax.set_ylabel(YLabel,fontsize=Fontsizes[0])


#         ax.xaxis.set_label_position('top')
#         ax.xaxis.set_ticks_position('both')
#         ax.tick_params(labelsize=Fontsizes[1])

#         ax.grid(True)
#         ax.grid('major', 'both', linestyle=':', lw=0.3)

#         if Legend:
#             ax.legend(fontsize=Fontsizes[0], loc='best')

#     if ThisAxis is None:
#         for F in PlotFormat:
#             plt.savefig(PlotFile+F)
#     # plt.show()
#     # plt.clf()

#     return ax
def make_pdf_catalog(workdir='./', pdflist=None, filename=None):
    '''
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

    '''
    # sys.exit('not in 3.9! Exit')

    import fitz

    catalog = fitz.open()

    for pdf in pdflist:
        with fitz.open(pdf) as mfile:
            catalog.insert_pdf(mfile)

    catalog.save(filename, garbage=4, clean=True, deflate=True)
    catalog.close()

    print('\n' + str(np.size(pdflist)) + ' files collected to ' + filename)
