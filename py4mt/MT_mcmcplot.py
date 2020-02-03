# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.1
# ---

"""
script to visualise rjmcmcmt based on Ross Brodie's original matlab 
plotting routines for rjmcmc inversion results.
 
CreationDate:   2017/10/17  -  Developer:      rakib.hassan@ga.gov.au
 
Revision History:
    
10/19   VR (Volker Rath)
    * adapted colorbar, sizes, and more 

"""
import os
import modules.plotrjmcmc as  pmc




results_dir = 'NEW_results_bbmt_roi_edit/'
plots_dir='NEW_plots_bbmt_roi_edit/'
if not os.path.isdir(plots_dir):
    print(' File: %s does not exist, but will be created' % plots_dir)
    os.mkdir(plots_dir)
    
outstrng='TIM'


result_files = os.listdir(results_dir)
result_files = sorted(result_files)
nfiles = len(result_files)

count=0
for path in result_files:
    count=count+1 
    print('\n') 
    print(str(count)+' of '+str(nfiles))
    print(os.path.basename(path).replace('MT0',outstrng))

    r = pmc.Results(os.path.join(results_dir, path), 
                         os.path.join(plots_dir,'%s.png'%(os.path.basename(path).replace('MT0',outstrng))), 
                         plotSizeInches='11x8', 
                         maxDepth=10000,
                         colormap='gray_r')
#                          colormap='OrRd')
#                         colormap='gray_r'
#                         colormap='rainbow')
#                         
                 
    r.plot()
#    r.close()
