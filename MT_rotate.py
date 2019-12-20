# -*- coding: utf-8 -*-
"""

@author: sb & vr oct 2019

"""

# import required modules
import os
from mtpy.core.mt import MT

# Define the path to your edi files
edi_in_dir = 'edifiles_bbmt/'
print(' Edifiles read from: %s' % edi_in_dir)
edi_files = os.listdir(edi_in_dir)
in_string = 'MT'

# Define the path for saving  edifiles
edi_out_dir= 'edifiles_bbmt_rot0/'
out_string = 'MT0'

## loop
for filename in edi_files :
    print('\n Reading data from '+edi_in_dir+filename)
    name, ext = os.path.splitext(filename)
    # Create an MT object 
    file_in = edi_in_dir+filename
    mt_obj = MT(file_in)
    print(' site %s at :  % 10.6f % 10.6f' % (name, mt_obj.lat, mt_obj.lon))
    rot_angle=-1.*mt_obj.Z.rotation_angle
#    print(mt_obj.Z.rotation_angle)
    mt_obj.Z.rotate(rot_angle)
#    print(mt_obj.Z.rotation_angle)
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

  
