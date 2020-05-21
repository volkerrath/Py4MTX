#!/usr/bin/env python
"""
Description:
    This script is a translation of Ross Brodie's original matlab 
    plotting routines for rjmcmc inversion results.

References:
 
CreationDate:   2017/10/17
Developer:      rakib.hassan@ga.gov.au
 
Revision History:

03/18   RH
    * initial checkin

03/18   RH
    * updating plotting routines to reflect changes to output data
    
10/19   VR (Volker Rath)
    * adapted colorbar and sizes
"""

import os
import gc
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker

#from matplotlib.ticker import MultipleLocator
# import click
import matplotlib.style as mplstyle
mplstyle.use('fast')
#plt.rc('text', usetex=True)

class Results():
    def __init__(self, path, outputFileName, plotSizeInches='11x8', maxDepth=2000,
                 resLims=[1.,100000.], zLog=False,
                 colormap='gray_r'):
        self._stationDir = path
        self._outputFileName = outputFileName

        self._plotLowestMisfitModels = False
        self._plotMisfits = True
        self._plotSynthetic = False
        self._depthPlotLim = [0, maxDepth]
        self._resPlotLim = resLims
        self._plotWidth = float(plotSizeInches.split('x')[0])
        self._plotHeight = float(plotSizeInches.split('x')[1])
        self._colorMap = colormap
        self._logZ = zLog

        # Extract station information from station_info file
        self._stationInfoFile = os.path.join(self._stationDir, 'station_info.txt')
        f = open(self._stationInfoFile)
        lines = f.readlines()
        self._station = lines[0].strip()
        self._dataType = lines[1].strip()
        self._nchains = int(lines[2])
        self._burninSampls = int(lines[3])
        self._totalSamples = int(lines[4])
        f.close()

        # Check if we inverted Impeadance or AppRes and AppPhase
        self._dtImpedance = 0
        self._dtAppres = 1
        if (self._dataType == 'Impedance'):
            self._dtImpedance = 1
            self._dtAppres = 0
        # end

        # Plot file names
        self._jpgFile = os.path.join(self._stationDir, '%s_lineardepth.jpg' % (self._station))
        self._epsFile = os.path.join(self._stationDir, '%s_lineardepth.eps' % (self._station))
        self._pdfFile = os.path.join(self._stationDir, '%s_lineardepth.pdf' % (self._station))

        # Title
        self._titleString = ('MT Station %s - 1D rj-McMC Inversion' % (self._station))

        # Load data
        d = np.loadtxt(os.path.join(self._stationDir, 'data.txt'))
        n = np.loadtxt(os.path.join(self._stationDir, 'noise.txt'))
      
        valid_test = np.append(d,n,axis=1)
        not_valid  = np.where(valid_test >1.e30)
        valid = np.delete( valid_test, not_valid[0],axis=0)


        self._D = defaultdict(list)
        self._D['f']        = valid[:, 0]
        self._D['data1']    = valid[:, 1]
        self._D['data2']    = valid[:, 2]
        self._D['noise1']   = valid[:, 4]
        self._D['noise2']   = valid[:, 5]  
        self._D['period'] = 1. / self._D['f']



        # Set limits
        self._flim = [np.min(self._D['f']) * 0.5, np.max(self._D['f']) * 2.0]
        self._plim = [np.min(self._D['period']) * 0.5, np.max(self._D['period']) * 2.0]
        self._ndata = 2 * len(self._D['f'])

        if (self._plotSynthetic == True):
            pass
        # end if

        # Read best fit forward models for each chain / process
        self._BFFM = defaultdict(list)
        for chain in range(self._nchains):
            fn = os.path.join(self._stationDir, 'best_model_forward.%03d.txt' % (chain))
            d = np.loadtxt(fn)

            if (chain == 0):
                self._BFFM['f'] = np.zeros((self._nchains, d.shape[0]))
                self._BFFM['data1'] = np.zeros((self._nchains, d.shape[0]))
                self._BFFM['data2'] = np.zeros((self._nchains, d.shape[0]))
                self._BFFM['period'] = np.zeros((self._nchains, d.shape[0]))
            # end if

            self._BFFM['f'][chain, :] = d[:, 0]
            self._BFFM['data1'][chain, :] = d[:, 1]
            self._BFFM['data2'][chain, :] = d[:, 2]
            self._BFFM['period'][chain, :] = 1. / self._BFFM['f'][chain, :]

            fn = os.path.join(self._stationDir, 'best_model.%03d.txt' % (chain))
            d = np.loadtxt(fn, skiprows=1)
            self._BFFM['model'].append(d)
        # end for

        # Load bins
        self._depthBins = np.loadtxt(os.path.join(self._stationDir, 'depth_bins.txt'))
        self._resBins = np.loadtxt(os.path.join(self._stationDir, 'prop_1_bins.txt'))
        self._resLim = np.array([[self._resBins[0], self._resBins[-1]]])

        # Load partition depths
        d = np.loadtxt(os.path.join(self._stationDir, 'interface_depth_hist.txt'))
        self._partitionDepth = d[:, 0]
        self._partitionDepthHist = d[:, 1]
        d = np.loadtxt(os.path.join(self._stationDir, 'npartitions_hist.txt'))
        self._layerNumHist = d
        self._layerNum = np.arange(d.shape[0])

        # Load props
        self._nprops = 1
        self._props = []  # list of defaultdicts
        for i in np.arange(self._nprops):
            dd = defaultdict(list)
            dd['name'] = 'prop_%d' % (i + 1)
            dd['limits'] = self._resLim[i, :]

            fn = os.path.join(self._stationDir, '%s_mean.txt' % (dd['name']))
            d = np.loadtxt(fn)
            dd['depth'] = d[:, 0]
            dd['mean'] = d[:, 1]

            fn = os.path.join(self._stationDir, '%s_median.txt' % (dd['name']))
            d = np.loadtxt(fn)
            dd['median'] = d[:, 1]

            fn = os.path.join(self._stationDir, '%s_mode.txt' % (dd['name']))
            d = np.loadtxt(fn)
            dd['mode'] = d[:, 1]

            fn = os.path.join(self._stationDir, '%s_credmin.txt' % (dd['name']))
            d = np.loadtxt(fn)
            dd['credmin'] = d[:, 1]

            fn = os.path.join(self._stationDir, '%s_credmax.txt' % (dd['name']))
            d = np.loadtxt(fn)
            dd['credmax'] = d[:, 1]

            fn = os.path.join(self._stationDir, '%s_hist.txt' % (dd['name']))
            d = np.loadtxt(fn)
            dd['hist'] = d

            self._props.append(dd)
        # end for
        self._depth = np.power(10., self._props[0]['depth'])

        if (self._plotMisfits):
            self._MF = defaultdict(list)
            # Load Misfits
            for chain in np.arange(self._nchains):
                fn = os.path.join(self._stationDir, 'misfit.%03d.txt' % (chain))
                d = np.loadtxt(fn)

                if (self._MF['sample'] == []):
                    self._MF['sample'] = np.zeros((d.shape[0], self._nchains))
                    self._MF['misfit'] = np.zeros((d.shape[0], self._nchains))
                # end if

                self._MF['sample'][:, chain] = np.arange(d.shape[0])
                self._MF['misfit'][:, chain] = d
    # end func

    def plot(self):

        def ShrinkAxes(ax, wfactor=1, hfactor=1):
            box = ax.get_position()
            shiftx = box.height * (1. - hfactor) * 0.5
            shifty = box.width * (1. - wfactor) * 0.5
            ax.set_position([box.x0 + shiftx, box.y0 + shifty, box.width * wfactor, box.height * hfactor])
        # end func

        plt.rc('font', family='sans-serif')
        plt.rc('font', serif='Helvetica Neue')
        plt.rc('xtick', labelsize='x-small')
        plt.rc('ytick', labelsize='x-small')
        plt.rc('axes', labelsize=8)
        plt.rc('axes', titlesize=12)
        plt.rc('legend', fontsize=8)

        # Set up fig and axes
        self._fig = plt.figure(figsize=(self._plotWidth, self._plotHeight))
        self._hpad = 0.025
        self._vpad = 0.025

        self._ax1 = self._fig.add_axes([0 + self._hpad, 0.6666 + self._vpad,
                                        0.5 - self._hpad, 0.3333 - self._vpad])  # top left
        self._ax2 = self._fig.add_axes([0 + self._hpad, 0.3333 + self._vpad,
                                        0.5 - self._hpad, 0.3333 - self._vpad])  # middle left
        self._ax3 = self._fig.add_axes([0 + self._hpad, 0 + self._vpad,
                                        0.25 - self._hpad, 0.3333 - self._vpad])  # lower-left first
        self._ax4 = self._fig.add_axes([0.25 + self._hpad, 0 + self._vpad,
                                        0.25 - self._hpad, 0.3333 - self._vpad])  # lower-left second
        self._ax5 = self._fig.add_axes([0.5 + self._hpad, 0.05 + self._vpad,
                                        0.3 - self._hpad, 0.95 - self._vpad])  # middle
        self._ax5cb = self._fig.add_axes([0.5 + self._hpad, 0 + self._vpad,
                                        0.3 - self._hpad, 0.05 - self._vpad])  # middle cb
        self._ax6 = self._fig.add_axes([0.8 + self._hpad*0.5, 0.05 + self._vpad,
                                        0.2 - self._hpad, 0.95 - self._vpad])  # right

        if (self._plotMisfits):
            xl = [1, 100 + np.max(self._MF['sample'][:, 0])]
            yl = [np.min(self._MF['misfit'])/10., np.max(self._MF['misfit'])]

            self._ax3.loglog(self._MF['sample'], self._MF['misfit'], 'b', lw=0.2)
            self._ax3.set_xlim(xl)
            self._ax3.set_ylim(yl)
            self._ax3.plot([self._burninSampls, self._burninSampls], yl, '--', dashes=(5, 10), color='k', lw=0.2)
            self._ax3.plot(xl, [self._ndata, self._ndata], '--', dashes=(5, 10), color='k', lw=0.2)
            self._ax3.set_xlabel('Number of Sampling Steps')
            self._ax3.set_ylabel('Data Misfit')

        # Layers histogram
        self._ax4.bar(self._layerNum, self._layerNumHist, ec='w')
        self._ax4.ticklabel_format(style='sci',axis='y') 
        self._ax4.yaxis.set_tick_params(labelrotation=90.)
        ml = ticker.MultipleLocator(1)
        self._ax4.xaxis.set_minor_locator(ml)
       
        nLayer= np.max(self._layerNum)
        if np.mod(nLayer,2) !=0:
            nLayer= nLayer+1
            
        self._ax4.set_xlim([0, nLayer])
        self._ax4.set_xlabel('Number of Layers')
        

        # Resistivity Depth
        im = self._props[0]['hist']
#        a = np.sort(im)

        for ri in np.arange(im.shape[0]):
            #im[ri, :] = im[ri, :] / np.max(im[ri, :])
            pass

        im = np.ma.masked_array(im, mask=im<=0)
        cbinfo = self._ax5.pcolormesh(np.power(10., self._resBins),
                             np.power(10., self._depthBins), im,
                             cmap=plt.get_cmap(self._colorMap, 256),
                             norm=colors.LogNorm(vmin=im.min(), vmax=im.max()))
        self._ax5.set_xlim(self._resPlotLim)
        self._ax5.set_ylim(self._depthPlotLim)
        self._ax5.set_xscale('log')
        self._ax5.set_yscale('linear')
        if self._logZ ==True: 
            self._ax5.set_yscale('log')
            
        self._ax5.invert_yaxis()
        

        self._ax5.plot(np.power(10., self._props[0]['median'], ), self._depth, '-k', lw=1, label='Median')
        self._ax5.plot(np.power(10., self._props[0]['credmin'], ), self._depth, '--k', dashes=(5, 5), lw=1,
                       label='10th & 90th\n percentile')
        self._ax5.plot(np.power(10., self._props[0]['credmax'], ), self._depth, '--k', dashes=(5, 5), lw=1)
        self._ax5.plot(np.power(10., self._props[0]['mean'], ), self._depth, '-b', lw=1, label='Mean')
        self._ax5.plot(np.power(10., self._props[0]['mode'], ), self._depth, '-g', lw=1, label='Mode')

        self._ax5.set_xlabel('Resistivity [$\Omega . m$]')

        self._ax5.legend(loc=3)
        self._ax5.grid(linestyle=':')
        self._ax5.xaxis.set_label_position('top')
        self._ax5.xaxis.set_tick_params(labeltop=True)
        self._ax5.xaxis.set_ticks_position('top')
        
        self._ax5.set_ylabel('Depth [m]',rotation='vertical',labelpad =-25)
        self._ax5.yaxis.set_tick_params(labelrotation=90.)
        
        cbar = self._fig.colorbar(cbinfo, cax=self._ax5cb, orientation='horizontal')
        # cbar.set_label('Log frequency', labelpad=15)        
        cbar.set_label('Log frequency')
#        cbar.set_label('Log conditional probability', labelpad=15)
#        cbar.ax.set_xticklabels([])
        cbar.ax.tick_params(axis=u'both', which=u'both', length=0)

        # Plot partition depths
        xlim = [-10, np.max(self._partitionDepthHist)]
        ylim = self._depthPlotLim

        self._ax6.plot(self._partitionDepthHist, np.power(10., self._partitionDepth), 'b', lw=0.5)
        self._ax6.set_xlim(xlim)
        self._ax6.set_ylim(ylim)
        self._ax6.invert_yaxis()   
        if self._logZ ==True: 
            self._ax6.set_yscale('log')
        self._ax6.grid(linestyle=':')
        self._ax6.set_xlabel('Change point frequency')
        self._ax6.xaxis.set_label_position('top')
        self._ax6.set_yticklabels([])
        self._ax6.set_xticklabels([])

        # Data 1 (Real impedance or App Res data)
        self._ax1.errorbar(self._D['period'], self._D['data1'], self._D['noise1'], color='red',
                           zorder=1, label='Data')

        xlim = self._plim
        for i in np.arange(self._nchains):
            if (i == 0):
                self._ax1.loglog(self._BFFM['period'][i, :], self._BFFM['data1'][i, :], 'b-', lw=1,
                                 zorder=2, label='Best fit for each chain')
            else:
                self._ax1.loglog(self._BFFM['period'][i, :], self._BFFM['data1'][i, :], 'b-', lw=1, zorder=2)
 
        self._ax1.set_xscale('log')
        self._ax1.set_yscale('log')
        self._ax1.set_xlim(xlim)
        self._ax1.legend(loc='best')
        self._ax1.xaxis.set_label_position('top')
        self._ax1.set_xlabel('Period [s]')
        self._ax1.grid(linestyle=':',which='major',axis='both')
        if (self._dtImpedance):
            self._ax1.set_ylabel('Real Impedance \n[mv/km/nT]')
        else:
            self._ax1.set_ylabel('Apparent Resistivity \n[$\Omega . m$]')

        # Data 2 (Imaginary impedance or App phase data)
        self._ax2.errorbar(self._D['period'], self._D['data2'], self._D['noise2'], color='red',
                           zorder=1, label='Data')

        xlim = self._plim
        for i in np.arange(self._nchains):
            if (i == 0):
                self._ax2.loglog(self._BFFM['period'][i, :], self._BFFM['data2'][i, :], 'b-', lw=1,
                                 zorder=2, label='Best fit for each chain')
            else:
                self._ax2.loglog(self._BFFM['period'][i, :], self._BFFM['data2'][i, :], 'b-', lw=1, zorder=2)

        if (self._dtImpedance == True):
            self._ax2.set_xscale('log')
            self._ax2.set_yscale('log')
        else:
            self._ax2.set_xscale('log')
            self._ax2.set_yscale('linear')

        self._ax2.set_xlim(xlim)
        self._ax2.legend(loc='best')
        # self._ax2.set_xlabel([])
        self._ax2.xaxis.set_label_position('top') 
        self._ax2.set_xticklabels([])
        self._ax2.grid(linestyle=':',which='major',axis='both')
        if (self._dtImpedance):
            self._ax2.set_ylabel('Imag Impedance \n[mv/km/nT]')
        else:
            self._ax2.set_ylabel('Apparent Phase [Deg]')

        plt.suptitle(self._titleString, x=0.3, y=1.05)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0.05)
        print(self._outputFileName)
        plt.savefig(self._outputFileName, bbox_inches='tight', dpi=400)
        plt.show()
        gc.collect()
        plt.clf()
        # plt.close('all')


