#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:13:18 2024

@author: sbyrd
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

directory = "/home/sbyrd/Desktop/PEROU/DAHU/SABA200_LF/"

filename_in = [directory +"SABA200_50_Alpha03_ZT_NLCG.log",
               directory +"SABA200_20_Alpha03_ZT_NLCG.log",
               directory +"SABA200_10_Alpha03_ZT_NLCG.log"]
               
for filename in filename_in:              


    filename_out = filename.replace(".log", ".csv")
    rms= [] 
    
    with open(filename, "r") as file:
        for line in file:
            if (("START:" in line) or ("STARTLS:" in line) ) and ("rms=" in line) :
                        l = line.split()
                        print(l)
                        rms.append(float(l[6]))
    with open(filename_out, 'w') as file:
        for line in np.arange(len(rms)):
            
            file.write(str(line) +"," + str(rms[line]) + "\n")
                        
        print(rms)
        it=np.arange(len(rms))
        plt.plot(rms)
        plt.legend(["prior = 50 Ohmm", "prior = 20 Ohmm", "prior = 10 Ohmm"])
        plt.xlabel("iteration")
        plt.ylabel("nRMS")
        plt.grid("on")
        plt.savefig(directory + "rms.pdf")
