# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 10:23:16 2025
This creates the files called sites_vtf.txt and sites_imp.txt based on 
files result_VTF.txt and result_MT.txt as output from applying mergeResultOfFEMTIC 
to femtic inversion results
@authors: charroyj + vrath
"""
import os
import sys

#neither inputs nor outputs should normally need to be changed.
#inputs
imp_file = 'result_MT.txt'
vtf_file = 'result_VTF.txt'
pt_file = ''


if len(imp_file)>0 and os.path.exists(imp_file): 
    with open(imp_file, 'r') as filein_imp:
        site=''
        fileout_imp = open(imp_file.replace('results', 'sites'), 'w')
        filein_imp.readline()
        for line in filein_imp:
            nextsite = line.strip().split()[0]
            if nextsite!=site:
                fileout_imp.write(nextsite+' '+nextsite+'\n')
                site=nextsite
        fileout_imp.close()
else: 
    if len(imp_file)>0:  
        print(imp_file,'does not exist!')    
    else:
        print('pt_file not defined!')
        
if len(vtf_file)>0 and os.path.exists(vtf_file): 
    with open(vtf_file, 'r') as filein_vtf:
        site=''
        fileout_vtf = open(vtf_file.replace('results', 'sites'), 'w')
        filein_imp.readline()
        filein_vtf.readline()
        for line in filein_vtf:
            nextsite = line.strip().split()[0]
            if nextsite!=site:
                fileout_vtf.write(nextsite+' '+nextsite+'\n')
                site=nextsite
        fileout_vtf.close()
else: 
    if len(vtf_file)>0:
        print(vtf_file,'does not exist!')
    else:
        print('vtf_file does not exist!')
    
if len(pt_file)>0 and os.path.exists(pt_file): 
    with open(pt_file, 'r') as filein_pt:
        site=''
        fileout_pt = open(vtf_file.replace('results', 'sites'), 'w')
        filein_pt.readline()
        for line in filein_pt:
            nextsite = line.strip().split()[0]
            if nextsite!=site:
                fileout_pt.write(nextsite+' '+nextsite+'\n')
                site=nextsite
        fileout_pt.close()
else: 
    if len(vtf_file)>0:  
        print(pt_file,'does not exist!')    
    else:
        print('pt_file not defined!')