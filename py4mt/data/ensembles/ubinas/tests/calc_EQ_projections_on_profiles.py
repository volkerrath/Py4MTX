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
"""

import pandas as pd
import csv
import numpy as np

inversion_anchor =  [8189.946089438076, 300.4724201237234] # NS EW ubinas 2025
workdir = "/home/sbyrd/ubinas_HF/sens_075_10_L2_best2/"
points_in = "Points_in.csv"

file_loc="./loc.txt"
file_nodes="./Points_nodes.csv"
file_eq = "/home/sbyrd/python_scripts/femticPy/projects/ubinas_25_HF/cutaway/CATALOGS_seismic/sismos_filter_UBINAS_1998_2009_2013-2020_UTM.csv"
proj_distance=3.   #in km
nbprofiles=6    #number of profiles

df_eq = pd.read_csv(file_eq)  
#df_eq['Mag'] = df_eq['Mag'].fillna(df_eq['Mag'].min())

def distance_to_profile(x, y, x1, y1, x2, y2):
    numerator = abs((x2 - x1)*(y1 - y) - (x1 - x)*(y2 - y1))
    denominator = np.hypot(x2 - x1, y2 - y1)
    return numerator / denominator

def distance_along_profile(x, y, x1, y1, x2, y2):
    # Profile direction vector
    dx = x2 - x1
    dy = y2 - y1
    profile_length = np.hypot(dx, dy)
    # Unit vector in profile direction
    ux, uy = dx / profile_length, dy / profile_length
    # Midpoint of profile
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    # Vector from midpoint to point
    vx, vy = x - mx, y - my
    # Project onto profile unit vector to get signed distance
    distance = vx * ux + vy * uy
    return distance

with open(file_nodes, "r") as infile_nodes:
    reader = csv.reader(infile_nodes, delimiter=',')
    #next(reader)
    for i in range(nbprofiles):
        row1=next(reader)
        row2=next(reader)
        p_name = row1[0]
        p1_east = round(float(row1[3]),3)
        p1_north = round(float(row1[4]),3)
        p2_east = round(float(row2[3]) ,3)
        p2_north = round(float(row2[4]),3)
        
        p1_x = p1_north - inversion_anchor[0]
        p1_y = p1_east - inversion_anchor[1]
        p2_x = p2_north - inversion_anchor[0]
        p2_y = p2_east - inversion_anchor[1]
        
        profile_EQ=[]
        filename_profile_EQ = f"{p_name}_EQ.csv"
        
        for row in df_eq.itertuples(index=False):
            #calculate model space coordinates of EQ
            eq_x_ms=row.UTMn/1000 - inversion_anchor[0]
            eq_y_ms = row.UTMe/1000 - inversion_anchor[1]
            z = row.Depth/1000
            dist = distance_to_profile(eq_x_ms,eq_y_ms,p1_x,p1_y,p2_x,p2_y)
            if dist < proj_distance:
                x_profile = distance_along_profile(eq_x_ms,eq_y_ms,p1_x,p1_y,p2_x,p2_y)
                profile_EQ.append([x_profile,z,row.Mag,row.rms])
        
        with open(filename_profile_EQ, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # write header
            writer.writerow(['distance_along_profile', 'Depth','Mag'])       
            # Write each station's data
            for EQ in profile_EQ:
                writer.writerow(EQ)