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
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

"""
generate pseudo dat for forward modelling studies

@author: sb & vr July 2020

"""

# Import required modules

import os
import sys
import csv
import modules.util as utl
import modules.modem as mod
from mtpy.core.mt import MT
import numpy as np


# edi_gen = 'grid'
# # generate site list
# LonLimits = ( 6.275, 6.39)
# nLon = 31
# LatLimits = (45.37,45.46)
# nLat = 31 
LonLimits = (0,0)
nLon = 0
LatLimits = (0,0)
nLat = 0 

edi_gen = 'readcsv'
# read site list
edi_file = r'/home/vrath/AEM_Limerick/Limerick_pilot.csv'

edi_gen = 'readmod'
# read site list
mod_file = r'/home/vrath/AEM_Limerick/Limerick_pilot_etopo.rho'
nx_bnd = 14
ny_bnd = 14
centerlatlon = ()
centermod    = ()




# Define the path to your EDI-template:

edi_template = r'/home/vrath/AEM_Limerick/EDI_AMT_template.edi'
print(' Edifile template read from: %s' % edi_template)




# Define the path and appended string for saved EDI-files:

edi_out_dir=r'/home/vrath/AEM_Limerick/edi/'
print(' Edifiles written to: %s' % edi_out_dir)
if not os.path.isdir(edi_out_dir):
    print(' File: %s does not exist, but will be created' % edi_out_dir)
    os.mkdir(edi_out_dir)

OutName = 'LimMod'


# No changes required after this line!


# Construct list of EDI-files:

if edi_gen == 'grid': 
    # generate site list
    Lat, Lon = utl.gen_grid(LatLimits, nLat, LonLimits, nLon)
    nn = -1
    for latval in Lat:
        nn=nn+1
        nnstr = str(nn)
        mm = -1
        print(nnstr)
        for lonval in Lon:
            mm=mm+1
            mmstr = str(mm)
            print(mmstr)
    
    
    # # Create an MT object 
    
            file_in = edi_template
            mt_tmp = MT(file_in)
        
            mt_tmp.lat = Lat[nn]
            mt_tmp.lon = Lon[mm]
            mt_tmp.station = OutName+nnstr+'_'+mmstr
        
            file_out = OutName+nnstr+'_'+mmstr+'.edi'
        
            print('\n Generating '+edi_out_dir+file_out)
            print(' site %s at :  % 10.6f % 10.6f' % (mt_tmp.station, mt_tmp.lat, mt_tmp.lon))
    
    #  Write a new edi file:
    
            print('Writing data to '+edi_out_dir+file_out)
            mt_tmp.write_mt_file(
                save_dir=edi_out_dir,
                fn_basename=file_out,
                file_type='edi',
                longitude_format='LONG',
                latlon_format='dd'
                )

elif edi_gen == 'readcsv':
    # read site list
    Site = []
    Data = []
    with open(edi_file) as ef:
        for line in ef:
            print(line)
            d = line.split(',')
            Site.append([d[0]])               
            Data.append([float(d[1]),float(d[2]),float(d[3])])


    Site = [item for sublist in Site for item in sublist]
    Site = np.asarray(Site,dtype=object)
    Data = np.asarray(Data)
     
    Lon   = Data[:,0]
    Lat   = Data[:,1]
    Elev  = Data[:,2]



    # Enter loop:
    nn=-1
    for place in Site:
    # Create an MT object 
        nn=nn+1
        file_in = edi_template
        mt_tmp = MT(file_in)
    
        mt_tmp.lat = Lat[nn]
        mt_tmp.lon = Lon[nn]
        mt_tmp.station = place
    
        file_out = OutName+'_'+place+'.edi'
    
        print('\n Generating '+edi_out_dir+file_out)
        print(' site %s at :  % 10.6f % 10.6f' % (mt_tmp.station, mt_tmp.lat, mt_tmp.lon))
    
    # # Write a new edi file:
    
        print('Writing data to '+edi_out_dir+file_out)
        mt_tmp.write_mt_file(
            save_dir=edi_out_dir,
            fn_basename=file_out,
            file_type='edi',
            longitude_format='LONG',
            latlon_format='dd'
            )



elif edi_gen == 'readmod':
    # read site list
    
    dx, dy, dz, rho, reference = mod.readMod(mod_file)
    
    nx = np.shape(dx)[0]
    x  = np.hstack((0, np.cumsum(dx)))
    xreference = 0.5*(x[0]+x[nx]);
    x=x-xreference
    xc = 0.5*(x[1:nx+1]+x[0:nx])
    
    ny = np.shape(dy)[0]
    y = np.hstack((0, np.cumsum(dy)))    
    yreference = 0.5*(y[0]+y[ny]);
    y=y-yreference;
    yc = 0.5*(y[1:ny+1]+y[0:ny])
    
    
    for ii in np.arange(nx_bnd+1,nx-nx_bnd+1):
                
        for jj in np.arange(ny_bnd+1,ny-ny_bnd+1):
            
            Site = '_'+str(ii)+'_'+str(jj)
            
            
            
    for place in Site:
    # Create an MT object 
        nn=nn+1
        file_in = edi_template
        mt_tmp = MT(file_in)
    
        mt_tmp.lat = Lat[nn]
        mt_tmp.lon = Lon[nn]
        mt_tmp.station = place
    
        file_out = OutName+'_'+place+'.edi'
    
        print('\n Generating '+edi_out_dir+file_out)
        print(' site %s at :  % 10.6f % 10.6f' % (mt_tmp.station, mt_tmp.lat, mt_tmp.lon))
    
    # # Write a new edi file:
    
        print('Writing data to '+edi_out_dir+file_out)
        mt_tmp.write_mt_file(
            save_dir=edi_out_dir,
            fn_basename=file_out,
            file_type='edi',
            longitude_format='LONG',
            latlon_format='dd'
            )


else:
    print ('Error: option '+edi_gen+' not implemented. Exit.\n' )
    sys.exit(1)

    



  
