# -*- coding: utf-8 -*-
"""
Created on Fri May  9 13:50:17 2025
This calculates the projection of stations from a loc.txt file with
x y z name
for all stations in model coordinates (in km)
the goal is to have for each profile in a profile_nodes.csv file with 
name,distance,angle,x,y in utm coordinates, the list of stations within a certain 
projection distance, their distance along profile and z for plotting in GMT
@author: charroyj
@ Svet: loc.txt:   0	CARO	0.7426373459190359	-1.46121270493677	3.722
two files are created per profile with sites names and numbers

"""

import pandas as pd
import csv
import numpy as np

#inversion_anchor = [8192.24939354886, 296.4410415307236] # NS EW ubinas 2023
inversion_anchor =  [8189.946089438076, 300.4724201237234] # NS EW ubinas 2025 
#inversion_anchor =  [8202.339954253664, 232.28761872067145] # NS EW misti
#inversion_anchor = [0, 0]

file_loc="./loc.txt" #coord_projected_numbers delim space
file_nodes="./Points_nodes.csv"
proj_distance = 3   #in km
nbprofiles = 6    #number of profiles

df_loc = pd.read_csv(file_loc, sep=r"\s+", names=['station_number', 'station_name', 'ea', 'no', 'z']) #all in model space, in km, ie x=northing in general, no, east!

def distance_to_profile(ea, no, x1, y1, x2, y2):
    numerator = abs((x2 - x1)*(y1 - no) - (x1 - ea)*(y2 - y1))
    denominator = np.hypot(x2 - x1, y2 - y1)
    return numerator / denominator

def distance_along_profile(ea, no, x1, y1, x2, y2):
    # Profile direction vector
    dx = x2 - x1
    dy = y2 - y1
    profile_length = np.hypot(dx, dy)
    # Unit vector in profile direction
    ux, uy = dx / profile_length, dy / profile_length
    # Midpoint of profile
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    # Vector from midpoint to point
    vx, vy = ea - mx, no - my
    # Project onto profile unit vector to get signed distance
    distance = vx * ux + vy * uy
    distance = - distance
    return distance

with open(file_nodes, "r") as infile_nodes:
    reader = csv.reader(infile_nodes, delimiter=',')
    #next(reader)
    for i in range(nbprofiles):
        row1=next(reader)
        row2=next(reader)
        p_name = row1[0]
        if (round(float(row1[3])) < 1000.): 
            k_m = 1. 
        else:
            k_m = 1000.
        
        
        p1_east = round(float(row1[3]) / k_m,3)
        p1_north = round(float(row1[4]) / k_m,3)
        p2_east = round(float(row2[3]) / k_m,3)
        p2_north = round(float(row2[4]) / k_m,3)
        
            
        
        p1_n = p1_north  - inversion_anchor[0]
        p1_e = p1_east - inversion_anchor[1]
        p2_n = p2_north- inversion_anchor[0]
        p2_e = p2_east - inversion_anchor[1]
        
        profile_stations=[]
        profile_stations_names=[]
        filename_profile_stations = f"P{p_name}_stations.csv"
        filename_profile_stations_names = f"P{p_name}_station_names.csv"
        id=-1
        for row in df_loc.itertuples(index=False):
            id=id+1
            dist = distance_to_profile(row.ea,row.no,p1_e,p1_n,p2_e,p2_n)
            print(dist)
            if dist < proj_distance:
                x_profile = distance_along_profile(row.ea,row.no,p1_e,p1_n,p2_e,p2_n)
                #profile_stations.append([row.station_name,x_profile,row.z])
                profile_stations_names.append([row.station_name,x_profile,-row.z])
                profile_stations.append([id,x_profile,-row.z])
                
        
        with open(filename_profile_stations, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            # write header
            # writer.writerow(['station_name', 'distance_along_profile', 'negative_z'])       
            # Write each station's data
            for station in profile_stations:
                writer.writerow(station)
                
        with open(filename_profile_stations_names, 'w', newline='') as csvfile:
             writer = csv.writer(csvfile, delimiter=',')
             #write header
             writer.writerow(['station_name', 'distance_along_profile', 'negative_z'])       
             # Write each station's data
             for station in profile_stations_names:
                 writer.writerow(station)