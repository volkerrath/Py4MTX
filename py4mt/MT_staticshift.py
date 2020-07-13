#!/usr/bin/env python3
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
#       jupytext_version: 1.5.1
# ---

"""
Created on Mon Apr 20 15:20:03 2020

@author: sb & vr
(based on mtpy)

"""
# ==============================================================================
import os
import csv
import numpy as np
import mtpy.core.mt as mt
import modules.staticshift as ss
#import mtpy.imaging.mtplot as mtplot
from mtpy.imaging.plot_mt_response import PlotMTResponse


ss_radius       = 1500. 
freq_interval   =[1.e-3,1.e2]
prefix_remove   = 'A'
append_out      = '_r1500'  

plot_it = True
plot_pdf=True
plot_png=True
plot_eps=False
dpi = 600


edi_in_dir =  r'/home/vrath/ProfileR/BBMT_edited/'
print(' Edifiles read from: %s' % edi_in_dir)

edi_files=[]
files= os.listdir(edi_in_dir) 
for entry in files:
   if entry.endswith('.edi') and not entry.startswith('.'):
        full = os.path.join(edi_in_dir,entry)
        edi_files.append(full)


ns =  np.size(edi_files)


edi_out_dir =  r'/home/vrath/ProfileR/BBMT_r1500m/'
print(' Corrected Edifiles written to: %s' % edi_out_dir)
if not os.path.isdir(edi_out_dir):
    print(' File: %s does not exist, but will be created' % edi_out_dir)
    os.mkdir(edi_out_dir)

if plot_it:
    plots_dir =  edi_out_dir+'sscor_plots/'
    print(' Plots written to: %s' % plots_dir)
    if not os.path.isdir(plots_dir):
        print(' File: %s does not exist, but will be created' % plots_dir)
        os.mkdir(plots_dir)

    

ss_out_file = edi_out_dir+'/ss_list.csv'
with open(ss_out_file, 'w') as f:
    sitelist = csv.writer(f, delimiter=',')
    sitelist.writerow(['Sitename', 'Lat', 'Lon', 'SS_x', 'SS_y'])
    # sitelist.writerow([ns, ' ', ' '])

    for filename in edi_files :
        print(' Reading data from '+filename)
        tmp = os.path.split(filename)
        name, ext = os.path.splitext(os.path.split(filename)[1])
        # Create an MT object 
        file_i = filename
        print(filename)
        print(name)
        mt_obj = mt.MT(file_i)
        
        
        sitename = mt_obj.station
        lon = mt_obj.lon
        lat = mt_obj.lat
        # elev = mt_obj.elev
        # east = mt_obj.east
        # north = mt_obj.north
        
    
        ss_x, ss_y = ss.estimate_static_spatial_median(file_i,
                                                    radius=ss_radius,
                                                    prefix_remove = prefix_remove,
                                                    freq_interval=freq_interval,
                                                    shift_tol=.05)
        
        # write resuklts to list 
        sitelist.writerow([sitename, lat, lon, ss_x, ss_y])
        
        # remove static shift
        new_z = mt_obj.remove_static_shift(ss_x=ss_x, ss_y=ss_y)
        
        # write to new edi file
        print (name+append_out+'.edi')
        mt_obj.write_mt_file(save_dir=edi_out_dir, 
                        fn_basename= name+append_out, 
                        file_type='edi', # edi or xml format
                        new_Z_obj=new_z, # provide a z object to update the data
                        longitude_format='LONG', # write longitudes as 'LON' or 'LONG'
                        latlon_format='dd' # write as decimal degrees (any other input
                                            # will write as degrees minutes seconds
                        )      
        
        if plot_it == True:
            print(name)
            obj0 = mt.MT(file_i)
            obj1 = mt.MT(edi_out_dir+name+append_out+'.edi')
            # plot_num =1 xy + yx; =2 all 4 components; =3 xy yx det
             
            plot_obj = PlotMTResponse(z_object=obj0.Z,  # this is mandatory
                              # t_object=mt_obj.Tipper,
                              # pt_obj=mt_obj.pt,
                              station=obj0.station,
                              #plot_tipper='yr',  # plots the real part of the tipper
                              plot_num=1)
            
            plot_obj.station = obj0.station + " and " + obj1.station+'_no-ss'
            plot_obj.plot(overlay_mt_obj=obj1)
            
            # Finally save figure
    
            if plot_png:
                plot_obj.save_plot(os.path.join(plots_dir,name+".png"),file_format='png',fig_dpi=dpi)
            if plot_pdf:
                plot_obj.save_plot(os.path.join(plots_dir,name+".pdf"),file_format='pdf',fig_dpi=dpi)
            if plot_eps:
                plot_obj.save_plot(os.path.join(plots_dir,name+".eps"),file_format='eps',fig_dpi=dpi)

 
