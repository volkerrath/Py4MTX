# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
# ---

"""
script to visualise rjmcmcmt based on Ross Brodie's original matlab 
plotting routines for rjmcmc inversion results.
 
CreationDate:   2017/10/17  -  Developer:      rakib.hassan@ga.gov.au
 
Revision History:
    
10/19  VR (Volker Rath)
    * adapted colorbar, sizes, and more 
03/20  VR (Volker Rath)
    * minor improvements, less memory 

"""
import os
import modules.plotrjmcmc as  pmc

plot_format ='pdf'

RhoPlotLim = [0.1, 100000]


results_in_dir = r'/media/vrath/MT/RainyRiverTransect/1-D_Results/RRV_results/'
result_files=[]
files= os.listdir(results_in_dir) 
for entry in files:
    if  not entry.startswith('.'):
            result_files.append(entry)

result_files = sorted(result_files)
nfiles = len(result_files)




plots_dir=r'/media/vrath/MT/RainyRiverTransect/1-D_Results/RRV_plots/'
if not os.path.isdir(plots_dir):
    print(' File: %s does not exist, but will be created' % plots_dir)
    os.mkdir(plots_dir)

outstrng='_rjmcmc'



count=0
for path in result_files:
    count=count+1 
    print('\n') 
    print(str(count)+' of '+str(nfiles))
    basefile = os.path.basename(path)
    print(basefile)
    outfile = os.path.join(plots_dir,basefile+outstrng+'.'+plot_format)
    print(outfile)
    r = pmc.Results(os.path.join(results_in_dir, path), 
                         outfile, 
                         plotSizeInches='11x8', 
                         maxDepth=32000,
                         zLog =True,
                         colormap='rainbow')
    
    r.plot()
# for other available colormaps see: https://matplotlib.org/tutorials/colors/colormaps.html
