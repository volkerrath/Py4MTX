import numpy as np

import os
import sys
from sys import exit as error
import shutil

import numpy as np


def generate_directories(dir_base='./run_', N_samples=1, out = True):
    file_list = ['control.dat', 
                 'observe.dat', 
                 'mesh.dat',     
                 'resistivity_block_iter0.dat',
                 'distortion_iter0.dat'] 
    dir_list = []
    for iens in np.arange(N_samples):
        directory=dir_base+str(iens)+'/'
        os.makedirs(os.path.dirname(directory), exist_ok=True)
        shutil.copy(file_list, directory)
        dir_list.append(directory)
        
    if out:
        print('list of directories:')
        print(dir_list)
        
        
    return dir_list
        
        
        

def generate_data_ensemble(dir_base ='./run_',
                           N_samples=1,
                           fil_in='observe.dat', 
                           fac = 1.,
                           out=True):
    '''
    for i = 1 : nsamples do
        Draw perturbed data set: d_pert ∼ N(d, Cd)
        
    '''
    
    obs_list = []
    for iens in np.arange(N_samples):
        file_in = dir_base+str(iens)+'/'+fil_in
        shutil.copy(file_in, file_in.replace('.dat', '_orig.dat'))
        """
        Generate perturbed observe.dat
        """

        file_out = file_in
        obs_list.append(file_out)

    if out:
        print('list of perturbed observation files:')
        print(obs_list)
        
        
    return obs_list
        

def generate_model_ensemble(dir_base ='./run_',
                           N_samples=1,
                           file_in='resistivity_block_iter0.dat', 

                           Cm = None,
                           out=True):
    '''
    for i = 1 : nsamples do
        Draw model: m_pert ∼ N (m, Cm)

    '''

    mod_list = []
    for iens in np.arange(N_samples):
            file_in = dir_base+str(iens)+'/'+file_in
            shutil.copy(file_in, file_in.replace('.dat', '_orig.dat'))
            '''
            generate perturbed model
            '''


    if out:
        print('list of perturbed model files:')
        print(mod_list)
        
    return mod_list



def get_femtic_sorted(files=[], out=True):
    numbers = []
    for file in files:
        numbers.append(int(file[11:]))
    numbers = sorted(numbers)

    listfiles = []
    for ii in numbers:
        fil = 'sensMatFreq'+str(ii)
        listfiles.append(fil)

    if out:
        print(listfiles)
    return listfiles


def get_femtic_sites(imp_file='result_MT.txt',
                     vtf_file = 'result_VTF.txt',
                     pt_file = 'results_PT.txt'):
    """
    Created on Thu Feb 27 10:23:16 2025
    This creates the files called sites_vtf.txt and sites_imp.txt based on 
    files result_VTF.txt and result_MT.txt as output from applying mergeResultOfFEMTIC 
    to femtic inversion results
    @authors: charroyj + vrath
    """
    
    
    #neither inputs nor outputs should normally need to be changed.
    
    
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




def get_femtic_data(data_file=None, site_file=None, data_type="rhophas", out=True):
    """
    

    Parameters
    ----------
    data_file : TYPE, optional
        DESCRIPTION. The default is None.
    site_file : TYPE, optional
        DESCRIPTION. The default is None.
    data_type : TYPE, optional
        DESCRIPTION. The default is "rhophas".
    out : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    data_dict : TYPE
        DESCRIPTION.

   Note: Conversion to appropriate units: FEMTIC uses ohms
         1 ohm = 10000(4*pi) [mV/km/nT]
 
    """


    data = []
    with open(data_file, "r") as f:
        iline = -1
        for line in f:
            iline = iline+1
            if iline > 0:
                l = line.split()
                l = [float(x) for x in l]
                # print(l)
                l[0] = int(l[0])
                data.append(l)
    data = np.array(data)

    info = []
    with open(site_file, "r") as f:
        for line in f:
            l = line.split(',')
            l[1] = float(l[1])
            l[2] = float(l[2])
            l[3] = float(l[3])
            l[4] = int(l[4])
            # print(l)
            info.append(l)
    info = np.array(info)

    sites = np.unique(data[:, 0]).astype("int")-1

    head_dict = dict([
        ("sites", sites),
        ("frq", data[:, 1]),
        ("per", 1./data[:1]),
        ("lat", np.float64(info[:, 1])),
        ("lon", np.float64(info[:, 2])),
        ("elv", np.float64(info[:, 3])),
        ("num", data[:, 0].astype("int")-1),
        ("nam", info[:, 0][sites.astype("int")-1])
        ])

    if "rhophas" in data_type.lower():

        """
         Site      Frequency
         AppRxxCal   PhsxxCal   AppRxyCal   PhsxyCal   AppRyxCal  PhsyxCal  AppRyyCal   PhsyyCal
         AppRxxObs   PhsxxObs   AppRxyObs   PhsxyObs   AppRyxObs  PhsyxObs  AppRyyObs   PhsyyObs
         AppRxxErr   PhsxxErr   AppRxyErr   PhsxyErr   AppRyxErr  PhsyxErr  AppRyyErr   PhsyyErr


        """
        # print(np.shape(data))
        # print(np.shape(data[:, 2:10 ]))
        # print(np.shape(data[:, 10:18 ]))
        # print(np.shape(data[:, 18:26 ]))
        type_dict = dict([
            ("cal", data[:, 2:10]),
            ("obs", data[:, 10:18 ]),
            ("err", data[:, 18:26]),
        ])

    elif "imp" in data_type.lower():

        """
         Site      Frequency
         ReZxxCal   ImZxxCal   ReZxyCal   ImZxyCal   ReZyxCal  ImZyxCal  ReZyyCal   ImZyyCal
         ReZxxObs   ImZxxObs   ReZxyObs   ImZxyObs   ReZyxObs  ImZyxObs  ReZyyObs   ImZyyObs
         ReZxxErr   ImZxxErr   ReZxyErr   ImZxyErr   ReZyxErr  ImZyxErr  ReZyyErr   ImZyyErr
         
         Z_femtic in Ohm: 1 Ohm = 1e4*(4*pi) [mV/km/nT] => Z =  1.e-4/(4*np.pi)*Z_femtic
         
        """
        ufact = 1.e-4/(4*np.pi)
        type_dict = dict([
            ("cal", ufact*data[:, 2:10 ]),
            ("obs", ufact*data[:, 10:18 ]),
            ("err", ufact*data[:, 18:26]),
        ])

    elif "vtf" in data_type.lower():

        """
        Site    Frequency
        ReTzxCal   ImTzxCal   ReTzyCal   ImTzyCal
        ReTzxOb    ImTzxObs   ReTzyObs   ImTzyObs
        ReTzxErr   ImTzxErr   ReTzyErr   ImTzyErr
        """
        type_dict = dict([
            ("cal", data[:, 2:6 ]),
            ("obs", data[:, 6:10 ]),
            ("err", data[:, 10:15]),

        ])

    elif "pt" in data_type.lower():
        """
        Site    Frequency
        ReTzxCal   ImTzxCal   ReTzyCal   ImTzyCal
        ReTzxOb    ImTzxObs   ReTzyObs   ImTzyObs
        ReTzxErr   ImTzxErr   ReTzyErr   ImTzyErr
        """
        type_dict = dict([
            ("cal", data[:, 2:6 ]),
            ("obs", data[:, 6:18 ]),
            ("err", data[:, 10:14]),

        ])

    else:
        error("get_femtic_data: data type "+data_type.lower()+" not implemented! Exit.")

    data_dict = {**head_dict, **type_dict}

    return data_dict
