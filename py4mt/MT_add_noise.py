# -*- coding: utf-8 -*-
"""

@author: sb & vr oct 2019

"""

# import required modules
import os
from mtpy.core.mt import MT
import numpy as np


ErrPercent = 10.
# Define the path to your edi files
edi_in_dir = './edifiles_synth/'
print(' Edifiles read from: %s' % edi_in_dir)
in_string = '.edi'
# Define the path for saving  edifiles
edi_out_dir= edi_in_dir
out_string = '_Err'+str(ErrPercent)+'Percent.edi'




edi_files=[]
files= os.listdir(edi_in_dir) 
for entry in files:
   # print(entry)
   if entry.endswith('.edi') and not entry.endswith('.'):
            edi_files.append(entry)
ns =  np.size(edi_files)


Z_err_rel=ErrPercent/100. # 
T_err_rel=ErrPercent/100. # 


## loop
for filename in edi_files :
    print('\n Reading data from '+edi_in_dir+filename)
    name, ext = os.path.splitext(filename)
    # Create an MT object 
    file_in = edi_in_dir+filename
    mt_obj = MT(file_in)
    print(' site %s at :  % 10.6f % 10.6f' % (name, mt_obj.lat, mt_obj.lon))

    Z           = mt_obj.Z.z[:]   
#    print('orig') 
#    print(Z)  
    Z_err       = np.abs(Z*Z_err_rel) 
    rZ          = np.real(Z)
    iZ          = np.imag(Z)   
    rZ_err      = np.real(Z_err)
    iZ_err      = np.imag(Z_err)

    rZ_perturb  = np.random.normal(rZ,rZ_err)
    iZ_perturb  = np.random.normal(iZ,iZ_err)
    
    Z_perturb   = rZ_perturb+iZ_perturb*1j
    newZ           = Z_perturb
#    print('\n\n pert') 
#    print(Z)   
    
    T  = mt_obj.Tipper.tipper[:]
#    print('orig') 
#    print(Tipper)  
    
    T_err       = np.abs(T*T_err_rel) 
    rT          = np.real(T)
    iT          = np.imag(T)   
    rT_err      = np.real(T_err)
    iT_err      = np.imag(T_err)

    rT_perturb  = np.random.normal(rT,rT_err)
    iT_perturb  = np.random.normal(iT,iT_err)
    
    T_perturb   = rT_perturb+iT_perturb*1j
    newT        = T_perturb
#    print('\n\n pert') 
#    print(T)   

    mt_obj.Tipper.tipper    = newT
    mt_obj.Z.z              = newZ
##    print(mt_obj.Z.rotation_angle)
# Write a new edi file 
    file_out=filename.replace(in_string,out_string)
    print('Writing data to '+edi_out_dir+file_out)
    mt_obj.write_mt_file(
            save_dir=edi_out_dir,
            fn_basename=file_out,
            file_type='edi',
            new_Z_obj=mt_obj.Z, # provide a z object to update the data
            new_Tipper_obj=mt_obj.Tipper, # provide a z object to update the data
            longitude_format='LONG', # write longitudes as 'LONG' not ‘LON’
            latlon_format='dd'# write as decimal degrees (any other input
            # will write as degrees:minutes:seconds
            )

  
