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


# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 17:54:41 2018

@author: sb & vr oct 2019
'''

"""
from mtpy.core.mt import MT
from mtpy.utils.calculator import get_period_list
import os
import numpy as np

interp_type='slinear'
#interp_types = ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']
interp_pdec=5

# Define the path to your edi files
edi_dir = './edifiles_bbmt_roi/'
print(' Edifiles read from: %s' % edi_dir)

edi_files=[]
files= os.listdir(edi_dir) 
for entry in files:
   # print(entry)
   if entry.endswith('.edi') and not entry.endswith('.'):
            edi_files.append(entry)

# Define the path for saving  plots
newedi_dir = './edifiles_bbmt_resampled/' #edi_dir #'./plots_synth/'


for filename in edi_files :
    print('  \n reading data from '+filename)
    name, ext = os.path.splitext(filename)
    # Create an MT object 
    file_i = edi_dir+filename
    mt_obj = MT(file_i)  
    print('Size of Z list :',np.shape(mt_obj.Z.z))
    freq = mt_obj.Z.freq
        
    
    
    maxfreq = np.max(freq)
    minfreq = np.min(freq)
    print('MinFreq: '+str(minfreq)+'   MaxFreq: '+str(maxfreq))
    test_freq_list = 1./get_period_list(1e-3,1e3,interp_pdec) # pdec periods per decade from 0.0001 to 100000 s
    new_freq_list  = np.select(test_freq_list<=maxfreq, test_freq_list>=minfreq,test_freq_list)
       
    print('Size of old Freq list :',np.size(test_freq_list)) 
    print('Size of new Freq list :',np.size(new_freq_list)) 
    # create new Z and Tipper objects containing interpolated data
    new_Z_obj, new_Tipper_obj = mt_obj.interpolate(new_freq_list,interp_type=interp_type)

#    pt_obj = mt_obj.plot_mt_response(plot_num=1, # 1 = yx and xy; 2 = all 4 components
#    # 3 = off diagonal + determinant
#    plot_tipper = 'yri',
#    plot_pt = 'y' # plot phase tensor 'y' or 'n'f
#    )
#    pt_obj.save_plot(os.path.join(save_path,name+".pdf"), fig_dpi=400)    
    mt_obj.write_mt_file(save_dir=newedi_dir, 
                    fn_basename= name, 
                    file_type='edi', # edi or xml format
                    new_Z_obj=new_Z_obj, # provide a z object to update the data
                    new_Tipper_obj=new_Tipper_obj, # provide a tipper object to update the data
                    longitude_format='LONG', # write longitudes as 'LON' or 'LONG'
                    latlon_format='dd' # write as decimal degrees (any other input
                                       # will write as degrees minutes seconds
                    )         

