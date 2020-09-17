#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 15:41:25 2020

@author: vrath
"""
# Import required modules

import os
import csv
import simplekml 
from mtpy.core.mt import MT
import numpy as np
from modules.util import keycount, unique
from modules.modem import readDat


ExecFile    = '/home/vrath/work/MaurienneJCN/RunGnuFrog_PT_A.sh'
PriorFile    = '/home/vrath/work/MaurienneJCN/MaurPrior_1000.rho' 
CovarFile    = '/home/vrath/work/MaurienneJCN/Maur_02.rho' 

DataFiles    = ['/home/vrath/work/MaurienneJCN/Maur_T.dat',
                '/home/vrath/work/MaurienneJCN/Maur_P.dat',
                '/home/vrath/work/MaurienneJCN/Maur_Z.dat']

ForwdFile    = '/home/vrath/work/MaurienneJCN/Maur.fwd'
SiteFil     = '/home/vrath/work/MaurienneJCN/Sitelist.csv' 
# open file and read the content in a list
# SiteFil     = '/home/vrath/work/MaurienneJCN/Sitelist.csv'
# data        = np.loadtxt(open(SiteFil, "r"), delimiter=" ",dtype=object)  


for f in DataFiles:
    Site, Comp, Data, Head = readDat(f)
    Freqs  = unique(Data[:,0])
    NFreq  = np.size(Freqs)
    Sites  = unique(Site)          
    sData  = np.shape(Data)
    NdTot   = sData[0]
    
    for place in Sites:
        NdSite = keycount(place, f, what = 'pos')
        NdJCN  = NdTot-NdSite
        print('Number of data from site '+place+' is '+str(NdSite))
        print('Number of JCN sample is '+str(NdJCN))

    # print(Sites)
  
# bad_words = ['bad', 'naughty']

# with open('oldfile.txt') as oldfile, open('newfile.txt', 'w') as newfile:
#     for line in oldfile:
#         if not any(bad_word in line for bad_word in bad_words):
#             newfile.write(line)
# # np.shape(placelist)
# #input file
# fin = open("data.txt", "rt")
# #output
#     file to write the result to
# fout = open("out.txt", "wt")
# #for each line in the input file
# for line in fin:
# 	#read replace the string and write to output file
# 	fout.write(line.replace('pyton', 'python'))
# #close input and output files
# fin.close()
# fout.close()