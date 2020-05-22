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
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

"""
Rotate imedance tensor (Z) and tipper (T)  
by potentially different angles into NS/EW coordinate system

@author: sb & vr Jan 2020

"""

# Import required modules

import os
from mtpy.core.mt import MT
import numpy as np


# Define the path to your EDI-files:

# edi_in_dir = 'edifiles_bbmt_roi_corrected/'
edi_in_dir = 'test/'
print(' Edifiles read from: %s' % edi_in_dir)

# Define the path anf appended string for saved EDI-files:

# edi_out_dir= 'edifiles_bbmt_roigeo0/'
edi_out_dir= edi_in_dir
print(' Edifiles written to: %s' % edi_out_dir)
if not os.path.isdir(edi_out_dir):
    print(' File: %s does not exist, but will be created' % edi_out_dir)
    os.mkdir(edi_out_dir)

out_string = '_geo0'

# This example brings both angles to the normal (measurement) coordinate 
# system, i. e., afterwards the ZROT and TROT fields in tehe EDI-files are 
# both zero. Here Z_angle and T_angle are the rotation angles from the 
# input EDI-file. _rot=('-1.*Z_angle') and T_rot=('-1.*T_angle'), where 
# Z_angle has been obtained from the corresponding mt_obj as 
# Z_angle=mt_obj.Z.rotation_angle.

# Z_rot=('-1.*Z_angle')
# T_rot=('-1.*T_angle')


# Rotate to geographic N: 

Z_rot=('-10.2*np.ones(np.shape(Z_angle))')
T_rot=('-10.2*np.ones(np.shape(T_angle))')

# Both: 

# Z_rot=('-1.*Z_angle-10.2*np.ones(np.shape(Z_angle))')
# T_rot=('-1.*T_angle-10.2*np.ones(np.shape(T_angle))')


# As the actual call is done using the 'eval' function, this script can 
# also be used to do rotations of different kind, e.g. when using a rotated 
# coordinate system for modelling. In this case to rotate from the zero system 
# to a sytem rotated 40 degrees clockwise, the string would look like 
# '40.*np.ones(shape(Z_angle))'  or simpler '40.' for Z and T. When correcting 
# for declination, i.e., from geomagnetic to geographic north, it should be 
# the negative value of the # declination obtained for the time of acquisition 
# and geographic position at
# https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml).




# No changes required after this line!

# Construct list of EDI-files:


edi_files=[]
files= os.listdir(edi_in_dir) # input EDI-file. As the actual call is dine using the 'eval' function, 
for entry in files:
   # print(entry)
   if entry.endswith('.edi') and not entry.startswith('.'):
            edi_files.append(entry)
ns =  np.size(edi_files)

# Enter loop:

for filename in edi_files :
    print('\n Reading data from '+edi_in_dir+filename)
    name, ext = os.path.splitext(filename)

# Create an MT object 

    file_in = edi_in_dir+filename
    mt_obj = MT(file_in)
    print(' site %s at :  % 10.6f % 10.6f' % (name, mt_obj.lat, mt_obj.lon))
    
    Z_angle=mt_obj.Z.rotation_angle
    T_angle=mt_obj.Tipper.rotation_angle
    
    Z_rot_angle = eval(Z_rot)
    T_rot_angle=  eval(T_rot)


    mt_obj.Z.rotate(Z_rot_angle)
    mt_obj.Tipper.rotate(T_rot_angle)


# Write a new edi file:

    file_out=name+out_string+ext
    print('Writing data to '+edi_out_dir+file_out)
    mt_obj.write_mt_file(
            save_dir=edi_out_dir,
            fn_basename=file_out,
            file_type='edi',
            new_Z_obj=mt_obj.Z, 
            new_Tipper_obj=mt_obj.Tipper,
            longitude_format='LONG',
            latlon_format='dd'
            )

  
