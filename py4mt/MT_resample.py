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
Resample/interpolate data (Z+T) from a prescribed set of EDI-files.

@author: sb & vr dec 2019

'''

"""

# Import required modules

from mtpy.core.mt import MT
from mtpy.utils.calculator import get_period_list
import os
import numpy as np

# Parameters for interpolation and subsequent resampling. 
# mtpy uses interp1 frm the scypy module, so different methods can be chosen:
# Posibble options are 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'

interp_type='slinear'

# This is the resampling rate, give as points per decade. TYpical values ae between 4 and 8.

interp_pdec=5

# Define the path to your EDI-files:

edi_dir = './edifiles_test/'
print(' Edifiles read from: %s' % edi_dir)

# Define the path and append string for resampled data:

newedi_dir =  edi_dir
print(' Edifiles written to: %s' % newedi_dir)
out_string = '_rot0'

# No changes required after this line!

# Construct list of EDI-files:

edi_files=[]
files= os.listdir(edi_dir) 
for entry in files:

   if entry.endswith('.edi') and not entry.endswith('.'):
            edi_files.append(entry)

for filename in edi_files :
    print('  \n reading data from '+filename)
    name, ext = os.path.splitext(filename)
    
# Create an MT object 
    
    file_i = edi_dir+filename
    mt_obj = MT(file_i)  
    
    freq    = mt_obj.Z.freq
    Z       = mt_obj.Z.z
    s       = np.shape(mt_obj.Z.z)
    print('Size of Z list :',np.shape(Z))
    Z_tmp   = np.reshape(Z,(ss[0],4))
    print('Size of ZZ list :',np.shape(ZZ))
    while any
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
   
# Write a new edi file:
    
    
    file_out=filename+out_string+ext
    mt_obj.write_mt_file(save_dir=newedi_dir, 
                    fn_basename= file_out, 
                    file_type='edi', # edi or xml format
                    new_Z_obj=new_Z_obj, # provide a z object to update the data
                    new_Tipper_obj=new_Tipper_obj, # provide a tipper object to update the data
                    longitude_format='LONG', # write longitudes as 'LON' or 'LONG'
                    latlon_format='dd' # write as decimal degrees (any other input
                                       # will write as degrees minutes seconds
                    )         

