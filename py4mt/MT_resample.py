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

interp_type     = 'slinear'

# This is the resampling rate, give as points per decade. TYpical values ae between 4 and 8.

interp_pdec     = 5

# period buffer:

pbuff           = 2

# Define the path to your EDI-files:

edi_dir         = './edifiles_test/'
print(' Edifiles read from: %s' % edi_dir)

# Define the path and append string for resampled data:

newedi_dir      =  edi_dir
print(' Edifiles written to: %s' % newedi_dir)
out_string      = '_interp'

# Setup frequency lists     

min_per_test    =  1e-3
max_per_test    =  1e4
test_freq_list = 1./get_period_list(min_per_test,max_per_test,interp_pdec) 
 
maxfreq = np.max(test_freq_list)
minfreq = np.min(test_freq_list)
print( 'MinFreqTest: '+str(minfreq)+'   MaxFreqTest: '+str(maxfreq))
    

# No changes required after this line!

# Construct list of EDI-files:

edi_files=[]
files= os.listdir(edi_dir) 
for entry in files:
   if entry.endswith('.edi') and not entry.startswith('.'):
            edi_files.append(entry)

for filename in edi_files :
    print('  \n reading data from '+filename)
    name, ext = os.path.splitext(filename)
    
# Create an MT object 
    
    file_i = edi_dir+filename
    mt_obj = MT(file_i)  
    
 
    freq    = mt_obj.Z.freq
   
# Get Impedance data: 
    
    Z       = mt_obj.Z.z
    sZ      = np.shape(Z)
    print(' Size of Z list :',sZ)
    tmp     = np.reshape(Z,(sZ[0],4))
    
# Find indices of valid impedance data, i. e., there absolute value is 
# zero. This corresponds to the EMPTY key in EDI.

    idx     = np.where([all(np.abs(row))>0. for row in tmp[:,1:]])
    tmpfreq=freq[idx]
    
    maxfreq = tmpfreq[0]
    minfreq = tmpfreq[-1]
    print(' Z: MinFreq: '+str(minfreq)+'   MaxFreq: '+str(maxfreq))
    
    new_freq_list=tmpfreq
    new_Z_obj, _ = mt_obj.interpolate(
        new_freq_list,
        interp_type=interp_type, 
        period_buffer = pbuff)

# Get Tipper data:
    
    T       = mt_obj.Tipper.tipper
    sT      = np.shape(T)
    print(' Size of T list :',sT)
    tmp     = np.reshape(T,(sT[0],2)) 

    
# Find indices of valid tipper data, i. e., there absolute value is 
# zero. This corresponds to the EMPTY key in EDI.

    idx     = np.where([all(np.abs(row))>0. for row in tmp[:,1:]])
    tmpfreq=freq[idx]
    maxfreq = tmpfreq[0]
    minfreq = tmpfreq[-1]
    print(' T: MinFreq: '+str(minfreq)+'   MaxFreq: '+str(maxfreq))
    
    new_freq_list=tmpfreq
    _ , new_Tipper_obj= mt_obj.interpolate(
        new_freq_list,
        interp_type=interp_type, 
        period_buffer = pbuff)
    #    pt_obj = mt_obj.plot_mt_response(plot_num=1, # 1 = yx and xy; 2 = all 4 components
    #    # 3 = off diagonal + determinant
    #    plot_tipper = 'yri',
    #    plot_pt = 'y' # plot phase tensor 'y' or 'n'f
    #    )
    #    pt_obj.save_plot(os.path.join(save_path,name+".pdf"), fig_dpi=400)    
   
# Write a new edi file:
    
    file_out=name+out_string+ext
    
    mt_obj.write_mt_file(save_dir=newedi_dir, 
                    fn_basename= file_out, 
                    file_type='edi', # edi or xml format
                    new_Z_obj=new_Z_obj, # provide a z object to update the data
                    new_Tipper_obj=new_Tipper_obj, # provide a tipper object to update the data
                    longitude_format='LONG', # write longitudes as 'LON' or 'LONG'
                    latlon_format='dd' # write as decimal degrees (any other input
                                        # will write as degrees minutes seconds
                    )         

