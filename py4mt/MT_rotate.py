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

@author: sb & vr oct 2019

"""

# import required modules
import os
from mtpy.core.mt import MT
import numpy as np


# Define the path to your edi files
# edi_in_dir = 'edifiles_bbmt/'
edi_in_dir = 'edifiles_test/'
print(' Edifiles read from: %s' % edi_in_dir)
in_string = 'MT'
# Define the path for saving  edifiles
# edi_out_dir= 'edifiles_bbmt_rot0/'
edi_out_dir= 'edifiles_test/'
print(' Edifiles written from: %s' % edi_out_dir)
out_string = 'MT0'

edi_files=[]
files= os.listdir(edi_in_dir) 
for entry in files:
   # print(entry)
   if entry.endswith('.edi') and not entry.endswith('.'):
            edi_files.append(entry)
ns =  np.size(edi_files)

## loop
for filename in edi_files :
    print('\n Reading data from '+edi_in_dir+filename)
    name, ext = os.path.splitext(filename)
    # Create an MT object 
    file_in = edi_in_dir+filename
    mt_obj = MT(file_in)
    print(' site %s at :  % 10.6f % 10.6f' % (name, mt_obj.lat, mt_obj.lon))
    rot_angle=-1.*mt_obj.Z.rotation_angle
    print(mt_obj.Tipper.rotation_angle)
    mt_obj.Z.rotate(rot_angle)
    mt_obj.Tipper.rotate(-1.*rot_angle)
    print(rot_angle)
# Write a new edi file 
    file_out=filename.replace(in_string,out_string)
    print('Writing data to '+edi_out_dir+file_out)
    mt_obj.write_mt_file(
            save_dir=edi_out_dir,
            fn_basename=file_out,
            file_type='edi',
            new_Z_obj=mt_obj.Z, # provide a z object to update the data
            longitude_format='LONG', # write longitudes as 'LONG' not ‘LON’
            latlon_format='dd'# write as decimal degrees (any other input
            # will write as degrees:minutes:seconds
            )

  
