# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 14:09:18 2025

#make profile param.txt for cutaway from two profile points
"""
import csv, os
import math

#center
#inversion_anchor =  [8202.339954253664, 232.28761872067145] # NS EW misti
#inversion_anchor = [8192.24939354886, 296.4410415307236] # NS EW ubinas old
inversion_anchor =  [8189.946089438076, 300.4724201237234] # NS EW ubinas 2025
workdir = "/home/sbyrd/ubinas_HF/sens_075_10_L2_best2/"
points_in = "Points_in.csv"
profiles_in = workdir+points_in

nbprofiles=6    #number of profiles

#sites_impf = workdir + "sites_imp"  #two col sta name and number
#with open(sites_impf, "r") as infile:
#    reader = csv.reader(infile, delimiter=',')







with open(profiles_in, "r") as infile:
    reader = csv.reader(infile, delimiter=',')


    filename_out2 = workdir + points_in[0:-7] + '_nodes.csv'
    filename_out3 = workdir + "Points_in_km.csv"

    try:
        os.remove(filename_out2) # delete old output file if it exists 
        os.remove(filename_out3) # delete old output file if it exists 
    except OSError:
        pass


    for i in range(nbprofiles):
        
        #next(reader)        #skip header
        row1=next(reader)
        row2=next(reader)
        p_name = row1[0]
        p1e = round(float(row1[2]) / 1000,3)
        p1n = round(float(row1[1]) / 1000,3)
        p2e = round(float(row2[2]) / 1000,3)
        p2n = round(float(row2[1]) / 1000,3)
        dist = round(math.sqrt((p2e-p1e)**2. + (p2n-p1n)**2.),3)
        p_angle = round(180/3.14*math.atan2((p2e-p1e),(p2n-p1n)),1)
            
        param_angle = -p_angle
        #param_angle = p_angle
            
        pivot_e = (p1e+p2e)/2 - round(inversion_anchor[1],3)
        pivot_n = (p1n+p2n)/2 - round(inversion_anchor[0],3)
            
        filename_out1 = workdir + 'param_V_' + p_name + '.txt'
        #filename_out2 = workdir + points_in[0:-4] + '_nodes.csv'
            

            
        lines_nodes = f"{p_name},0,{p_angle},{p1e},{p1n}\n{p_name},{dist},{param_angle},{p2e},{p2n}\n"
        points_km=f"{p_name},{p1n},{p1e}\n{p_name},{p2n},{p2e}\n"
        lines_param = f"0\niter\n0\n{pivot_n} {pivot_e} 0\n{param_angle}\n1\n0"
        with open(filename_out1,"w") as file_out1:
                file_out1.write(lines_param)
                
        with open(filename_out2,"a") as file_out2:
            file_out2.write(lines_nodes)

        with open(filename_out3,"a") as file_out3:
            file_out3.write(points_km)
     
