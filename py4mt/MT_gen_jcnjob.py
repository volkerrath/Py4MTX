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
#       jupytext_version: 1.6.0
# ---

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

DataFiles    = ['/home/vrath/work/MaurienneJCN/Maur_P.dat',
               '/home/vrath/work/MaurienneJCN/Maur_T.dat',
               '/home/vrath/work/MaurienneJCN/Maur_Z.dat']

ForwdFile    = '/home/vrath/work/MaurienneJCN/Maur.fwd'
SiteFil     = '/home/vrath/work/MaurienneJCN/Sitelist.csv' 
# open file and read the content in a list
# SiteFil     = '/home/vrath/work/MaurienneJCN/Sitelist.csv'
# data        = np.loadtxt(open(SiteFil, "r"), delimiter=" ",dtype=object)  


for f in DataFiles:
    Site, Comp, Data = readDat(f)
    Freqs  = unique(Data[:,0])
    NFreq  = np.size(Freqs)
    Sites  = unique(Site)
      
    for place in Sites:
        print(place)

# # for place in placelist:
# with open(DataFile, 'r') as f:

# #         print(place)


# for place in placelist:

#     Ndata = keycount(place, DataFile, what = 'pos')
#     print('Number of data from site '+place+' is '+str(Ndata))


# bad_words = ['bad', 'naughty']

# with open('oldfile.txt') as oldfile, open('newfile.txt', 'w') as newfile:
#     for line in oldfile:
#         if not any(bad_word in line for bad_word in bad_words):
#             newfile.write(line)
# # np.shape(placelist)
# #input file
# fin = open("data.txt", "rt")
# #output file to write the result to
# fout = open("out.txt", "wt")
# #for each line in the input file
# for line in fin:
# 	#read replace the string and write to output file
# 	fout.write(line.replace('pyton', 'python'))
# #close input and output files
# fin.close()
# fout.close()
