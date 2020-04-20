#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 15:20:03 2020

@author: sb & vr
(based on mtpy)

"""
# ==============================================================================
import os
import numpy as np
import mtpy.core.mt as mt
import modules.staticshift as ss

ss_radius   = 4000. 
ss_numf     = 40
ss_freqint  =


edi_in_dir = r'/home/geothest/Desktop/WEST_TIMOR/NEW_edifiles_bbmt_roi/out_dist/'
print(' Edifiles reading from: %s' % edi_in_dir)

edi_files=[]
files= os.listdir(edi_in_dir) 
for entry in files:
   if entry.endswith('.edi') and not entry.startswith('.'):
            edi_files.append(entry)
ns =  np.size(edi_files)

edi_out_dir = r'/home/geothest/Desktop/WEST_TIMOR/NEW_edifiles_bbmt_roi/out_ss/'
if not os.path.isdir(edi_out_dir):
    print(' File: %s does not exist, but will be created' % edi_out_dir)
    os.mkdir(edi_out_dir)

ss_out_file = edi_out_dir+'out_ss'
with open(ss_out_file, 'w') as f:
    sitelist = csv.writer(f, delimiter=' ')
    sitelist.writerow(['Sitename', 'Lat', 'Lon', 'SS_x', 'SS_y'])
    sitelist.writerow([ns, ' ', ' '])

for filename in edi_files :
    print('reading data from '+filename)
    name, ext = os.path.splitext(filename)
    # Create an MT object 
    file_i = edi_in_dir+filename
    mt_obj = mt.MT(file_i)
    
    
    sitename = mt_obj.station
    lon = mt_obj.lon
    lat = mt_obj.lat
    # elev = mt_obj.elev
    # east = mt_obj.east
    # north = mt_obj.north
    

    ss_x, ss_y = ss.estimate_static_spatial_median(file_i,
                                                radius=ss_radius,
                                                num_freq=ss_numf,
                                                freq_skip=9,
                                                shift_tol=.05)
    
    # write resuklts to list 
    sitelist.writerow([sitename, lat, lon, ss_x, ss_y])
    
    # remove static shift
    new_z = mt_obj.remove_static_shift(ss_x=ss_x, ss_y=ss_y)
    
    # write to new edi file
    mt_obj.write_mt_file(save_dir=edi_out_dir, 
                    fn_basename= name, 
                    file_type='edi', # edi or xml format
                    new_Z_obj=new_z, # provide a z object to update the data
                    longitude_format='LONG', # write longitudes as 'LON' or 'LONG'
                    latlon_format='dd' # write as decimal degrees (any other input
                                       # will write as degrees minutes seconds
                    )      
    
