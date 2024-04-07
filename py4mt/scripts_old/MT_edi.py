#!/usr/bin/env python3

"""
@author: sb & vr oct 2019
"""

# Import required modules

import os
import sys
import csv

mypath = ["/home/vrath/Py4MT/py4mt/modules/", "/home/vrath/Py4MT/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)


import util
import numpy as np

# from mtpy.core import mt, transfer_function
import mtpy.core
from mtpy.core.z import Z, Tipper
from mtpy.core.mt import MT
import matplotlib.pyplot as plt

## Define the path to your EDI-files

site_dir = "/home/vrath/MT_Data/Opf/2023/orig/"
edis_dir = "/home/vrath/MT_Data/Opf/2023/edi_plus/"
print(" Edifiles read from: %s" % edis_dir)


plots_dir =   r"/home/vrath/MT_Data/Opf/2023/plots/"    
print(" Plots read from: %s" % plots_dir)


# open file and read the content in a list
places_file = "/home/vrath/MT_Data/Opf/2023/Sitelist.csv"
# r"/home/vrath/Py4MT/py4mt/M/FWD/Sitelist.csv"
# r"/home/vrath/WestTimor/places.csv"
places_list = []
with open(places_file, "r") as f:
    placelist = csv.reader(f, delimiter=",")
    for row in placelist:
        row[0]=float(row[0])*1000.
        row[1]=float(row[1])*1000.
        # row[2]=0.
        # row[3]= row[3]
        places_list.append(row)
        
# print(places_list)

coil_direction = 1.

# Construct list of EDI-files:


ofiles = []
files = os.listdir(site_dir)
for entry in files:
    ofiles.append(entry)
    
if not os.path.isdir(edis_dir):
    print("File: %s does not exist, but will be created" % edis_dir)
    os.mkdir(edis_dir)
if not os.path.isdir(plots_dir):
    print("File: %s does not exist, but will be created" % plots_dir)
    os.mkdir(plots_dir)    

# Loop over sites
    #print("reading data from "+filename)
for entry in ofiles:
    if not os.path.isfile(entry):  
        pass
           
    for line in places_list:

        sname = line[2]

        if sname.lower()+"." in entry.lower():
            print(sname, entry)
            sfile = entry
            gk_e = line[0]
            gk_n = line[1]
            elev = 0. 
            lat, lon = util.project_gk_to_latlon(gk_e, gk_n) 
            found = True
            break
        else:
            found = False
    #print("reading data from "+filename)
    # name, ext = os.path.splitext(filename)
    # file_i = edi_dir + filename

# Create MT object

    if not found:
        continue
    
    
    print("reading data from "+sfile)
    with open(site_dir+sfile, "r") as file:
        lines = file.readlines()
        oname = lines[0]
        # print(sname, oname)
        lines = lines[1:]
        a = []
        b = []
        for ilin in np.arange(len(lines)):
            # print( lines[ilin])
            tmp = lines[ilin].split()
            tmp = [float(t) for t in tmp]
            # print( tmp)
            if ilin%2 ==0:
                a.append(tmp)
                # print(lines[ilin].split())
            else:
                b.append(tmp)
    
    
    

    # a = [float(t) for t in a]
    a = np.asarray(a)
    # b = [float(t) for t in b]
    b = np.asarray(b)
    data = np.hstack((a,b))
    
    
    freq = 1./data[:,0]    
    print(lat, lon)
    Zxx = coil_direction*np.array([complex(data[n,1] + data[n,3] *1j) for n in np.arange(len(freq))])
    Zxxe = np.array([data[n,9] for n in np.arange(len(freq))])
    Zxy = coil_direction*np.array([complex(data[n,5] + data[n,7] *1j) for n in np.arange(len(freq))])
    Zxye = np.array([data[n,10] for n in np.arange(len(freq))])
    Zyx = coil_direction*np.array([complex(data[n,2] + data[n,4] *1j) for n in np.arange(len(freq))])
    Zyxe = np.array([data[n,11] for n in np.arange(len(freq))])
    Zyy = coil_direction*np.array([complex(data[n,6] + data[n,8] *1j) for n in np.arange(len(freq))])
    Zyye = np.array([data[n,12] for n in np.arange(len(freq))])
    
    z = np.array([Zxx, Zxy, Zyx, Zyy]).T.reshape(-1,2,2)
    z_error =  np.array([Zxxe, Zxye, Zyxe, Zyye]).T.reshape(-1,2,2)
    # rel_err = 0.05
    # z_error = rel_err*np.sqrt(np.abs(Zxy)*np.abs(Zyx))
    # print(np.shape) 
    
    Z_obj = Z(z_array=z, freq=freq)
    Z_obj.z_err = z_error
    # Z_obj = Z(z_array=z, z_err=z_error, freq=freq)
   
    mt_obj = MT(station=sname,
                lat=lat, lon=lon, elev=elev,
                Z=Z_obj)
     
    name, ext = os.path.splitext(os.path.basename(sfile))
    edi_name = name.lower()+ext.lower().replace(".","-")
    mt_obj.write_mt_file(new_Z_obj=Z_obj,
                         save_dir=edis_dir, fn_basename=edi_name+".edi", file_type='edi',
                         longitude_format='LONG',latlon_format='dd')


    # To plot the edi file we read in in Part 1 & save to f
    # edi_obj = mtpy.core.mt.MT(edi_name)
    # plot_obj = edi_obj.plot_mt_response(plot_num=1,
    #                                     plot_tipper="n",
    #                                     plot_pt="y",
    #                                     fig_dpi=400,
    #                                     xy_ls="",yx_ls="", det_ls="",
    #                                     ellipse_colorby="skew",
    #                                     ellipse_range = [-10.,10.,2.],
    #                                     res_limits = [1., 10000.]  ,                                      
    #                                     phase_limits = [-10., 100.]
    #                                     )
    # plot_name = name.lower()+ext.lower().replace(".","-")
    # plot_obj.save_plot(plots_dir+name+strng+".png", DPI=400)
    
