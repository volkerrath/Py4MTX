
import os
import sys
from sys import exit as error
import shutil
import copy
import inspect

import numpy as np
import scipy
import scipy.linalg
import scipy.sparse.linalg
import scipy.special
import scipy.fftpack
import scipy.sparse

import joblib


def generate_directories(
        dir_base='./ens_',
        templates='',
        file_list=['control.dat',
                   'observe.dat',
                   'mesh.dat',
                   'resistivity_block_iter0.dat',
                   'distortion_iter0.dat',
                   'run_dub.sh',
                   'run_oar_sh'],
        N_samples=1,
        out=True):

    dir_list = []
    for iens in np.arange(N_samples):
        directory = dir_base+str(iens)+'/'
        os.makedirs(directory, exist_ok=True)
        copy_files(file_list, directory, templates)
        dir_list.append(directory)

    if out:
        print('list of directories:')
        print(dir_list)

    return dir_list


def copy_files(filelist, directory, templates):
    for f in np.arange(len(filelist)):
        ff = templates+filelist[f]
        # shutil.copy(ff, directory)
        shutil.copy2(ff, directory)


def generate_data_ensemble(dir_base='./ens_',
                           N_samples=1,
                           file_in='observe.dat',
                           draw_from=['normal', 0., 1.],
                           method='add',
                           errors=[],
                           out=True):
    '''
    for i = 1 : nsamples do
        Draw perturbed data set: d_pert ∼ N(d, Cd)
        
    '''

    obs_list = []
    for iens in np.arange(N_samples):
        file = dir_base+str(iens)+'/'+file_in
        shutil.copy(file, file.replace('.dat', '_orig.dat'))
        '''
        Generate perturbed observe.dat
        '''
        modify_data(template_file=file,
                     draw_from=draw_from,
                     method=method,
                     errors=errors,
                     out=out)
        obs_list.append(file)

    if out:
        print('list of perturbed observation files:')
        print(obs_list)

    return obs_list


def modify_data(template_file='observe.dat',
                 draw_from=['normal', 0., 1.],
                 method='add',
                 errors=[ [], [], [] ],
                 out=True):
    '''
    Created on Thu Apr 17 17:13:38 2025
    
    @author:   vrath   
    '''
#    import numpy as np

#


        
        
    if template_file is None:
        template_file = 'observe.dat'

    print('\n', template_file, ':')

    with open(template_file, 'r') as file:
        content = file.readlines()

    line = content[0].split()
    obs_type = line[0]
    # num_site = int(line[1])
    print(len(content))

    '''  
    find data blocks
    
    '''
    start_lines_datablock = []
    for number, line in enumerate(content, 0):
        l = line.split()
        if len(l) == 2:
          start_lines_datablock.append(number)
          print(' data block', l[0], 'with',
                l[1], 'sites begins at line', number)
        if "END" in l:
            start_lines_datablock.append(number-1)
            print(" no further data block in file")
    '''  
     loop over  data blocks
     
    '''
    num_datablock = len(start_lines_datablock)-1
    for block in np.arange(num_datablock):
        start_block = start_lines_datablock[block]
        end_block = start_lines_datablock[block+1]
        # print(start_block, end_block)
        # print(type(start_block), type(end_block))
        data_block = content[start_block:end_block]
        '''  
        find sites
        '''
        print(np.shape(data_block))
        start_lines_site = []
        num_freqs = []
        for number, line in enumerate(data_block, 0):
            l = line.split()
            if len(l) == 4:
              print(l)
              start_lines_site.append(number)
              num_freqs.append(int(data_block[number+1].split()[0]))          
              print('  site', l[0],'begins at line', number)
            if "END" in l:
                 start_lines_datablock.append(number-1)
                 print(" no further site block in file")
        print('\n')   
        # print(start_lines_site)
        # print(num_freqs)
              
        num_sites = len(start_lines_site)
        for site in np.arange(num_sites):
            start_site = start_lines_site[site]
            end_site = start_site+num_freqs[site]+2
            site_block = data_block[start_site:end_site]
            # print('site',site+1) 
            # print(np.shape(site_block))
            # print(site_block)
                        
            if 'MT' in obs_type:
                
                if len(errors[0])!=0:
                    set_errors_mt = True

                dat_length = 8
                
                num_freq = int(site_block[1].split()[0])
                print('   site ',site,'has',num_freq,'frequencies' )
                obs  = []
                for line in site_block[2:]:
                    # print(line)
                    tmp = [float(x) for x in line.split()]
                    obs.append(tmp)
                    
                if set_errors_mt:
                    new_errors = errors[0]
                    print('MT errors will be replaced with relative errors:')
                    print(new_errors)
                    for line in obs:
                        # print(np.arange(1,dat_length+1))
                        # print(freq)
                        for ii in np.arange(1, dat_length+1):
                            print(site, '   ', ii, ii+dat_length)
                            val = line[ii]
                            err = val*new_errors
                            line[ii+dat_length] = err
                
                for line in obs:

                     for ii in np.arange(1,dat_length+1):
                         print(site, '   ',ii, ii+dat_length)
                         val = line[ii]
                         err = line[ii+dat_length]
                         line[ii] = np.random.normal(loc=val, scale=err)
                         
                '''
                now write new values
                
                '''
                print('obs',np.shape(obs), np.shape(site_block))  
                print(np.arange(num_freq))
                for f in  np.arange(num_freq-1):
                    print(f)
                    print( site_block[f+2])
                    print( obs[f])
                    site_block[f+2] = "    ".join([f"{x:.8E}" for x in obs[f]])+'\n'
                    print( site_block[f+2])     

            elif 'VTF' in obs_type:
                
                if len(errors[1])!=0:
                    set_errors_vtf = True

                dat_length = 4

                num_freq = int(site_block[1].split()[0])
                print('   site ',site,'has',num_freq,'frequencies' )
                obs  = []
                for line in site_block[2:]:
                    # print(line)
                    tmp = [float(x) for x in line.split()]
                    obs.append(tmp)
                    
                    if set_errors_vtf:
                        new_errors = errors[1]
                        print('VTF errors will be replaced with relative errors:')
                        print(new_errors)
                        for line in obs:
                            # print(np.arange(1,dat_length+1))
                            # print(freq)
                            for ii in np.arange(1, dat_length+1):
                                print(site, '   ', ii, ii+dat_length)
                                val = line[ii]
                                err = new_errors
                                line[ii+dat_length] = err
                    
                    for line in obs:
                        # print(np.arange(1,dat_length+1))
                        # print(freq)
                        for ii in np.arange(1, dat_length+1):
                            print(site, '   ', ii, ii+dat_length)
                            val = line[ii]
                            err = line[ii+dat_length]
                            line[ii] = np.random.normal(loc=val, scale=err)

                '''
                now write new values

                '''
                print('obs',np.shape(obs), np.shape(site_block))
                print(np.arange(num_freq))
                for f in  np.arange(num_freq-1):
                    print(f)
                    print( site_block[f+2])
                    print( obs[f])
                    site_block[f+2] = "    ".join([f"{x:.8E}" for x in obs[f]])+'\n'
                    print( site_block[f+2])
                    
            elif 'PT' in obs_type:
                
                if len(errors[2])!=0:
                    set_errors_pt = True

                dat_length = 4

                num_freq = int(site_block[1].split()[0])
                print('   site ',site,'has',num_freq,'frequencies' )
                obs  = []
                for line in site_block[2:]:
                    # print(line)
                    tmp = [float(x) for x in line.split()]
                    obs.append(tmp)
                    
                    if set_errors_pt:
                        new_errors = errors[1]
                        print('VTF errors will be replaced with relative errors:')
                        print(new_errors)
                        for line in obs:
                            # print(np.arange(1,dat_length+1))
                            # print(freq)
                            for ii in np.arange(1, dat_length+1):
                                print(site, '   ', ii, ii+dat_length)
                                val = line[ii]
                                err = new_errors
                                line[ii+dat_length] = err
                    
                    for line in obs:
                        # print(np.arange(1,dat_length+1))
                        # print(freq)
                        for ii in np.arange(1, dat_length+1):
                            print(site, '   ', ii, ii+dat_length)
                            val = line[ii]
                            err = line[ii+dat_length]
                            line[ii] = np.random.normal(loc=val, scale=err)

                '''
                now write new values

                '''
                print('obs',np.shape(obs), np.shape(site_block))
                print(np.arange(num_freq))
                for f in  np.arange(num_freq-1):
                    print(f)
                    print( site_block[f+2])
                    print( obs[f])
                    site_block[f+2] = "    ".join([f"{x:.8E}" for x in obs[f]])+'\n'
                    print( site_block[f+2])
            else:
                
                error(obs_type+' not yet implemented! Exit.')

            data_block[start_site:end_site] = site_block
            
        
        content[start_block:end_block] = data_block            
        

   
    print (np.shape(content))
    with open(template_file, 'w') as f:
        f.writelines(content)


    if out:
        print('File '+template_file+' successfully written.')




def generate_model_ensemble(dir_base='./ens_',
                            N_samples=1,
                            file_in='resistivity_block_iter0.dat',
                            draw_from=['normal', 0., 1.],
                            method='add',
                            out=True):
    '''
    for i = 1 : nsamples do
        Draw model: m_pert ∼ N (m, Cm)

    '''

    mod_list = []
    for iens in np.arange(N_samples):
        file = dir_base+str(iens)+'/'+file_in
        shutil.copy(file, file.replace('.dat', '_orig.dat'))
        '''
        generate perturbed model
        '''
        modify_model(template_file=file,
                      draw_from=draw_from,
                      method=method,
                      out=out)
        mod_list.append(file)

    if out:
        print('\n')
        print('list of perturbed model files:')
        print(mod_list)

    return mod_list


def modify_model(template_file='resistivity_block_iter0.dat',
                  draw_from=['normal', 0., 1.],
                  method='add',
                  out=True):
    '''
    Created on Thu Apr 17 17:13:38 2025
    
    @author:       vrath   
    '''
#    import numpy as np

    # rng = np.random.default_rng()
    if template_file is None:
        template_file = 'resistivity_block_iter0.dat'

    with open(template_file, 'r') as file:
        content = file.readlines()

    nn = content[0].split()
    nn = [int(tmp) for tmp in nn]
    n_cells = nn[1]

    if 'normal' in draw_from[0]:
        samples = np.random.normal(
            loc=draw_from[1], scale=draw_from[2], size=n_cells-2)
    else:
        samples = np.random.uniform(
            low=draw_from[1], high=draw_from[2], size=n_cells-2)
    
    # element groups: air and seawater fixed
    new_lines = [
        '         0        1.000000e+09   1.000000e-20   1.000000e+20   1.000000e+00         1',
        '         1        2.500000e-01   1.000000e-20   1.000000e+20   1.000000e+00         1'
    ]

    print(nn[0], nn[0]+nn[1]-1, nn[1]-1, np.shape(samples))
    
    e_num = 1
    for elem in range(nn[0]+3, nn[0]+nn[1]+1):
        e_num = e_num + 1
        line = content[elem].split()
        x = float(line[1])
      
        if 'add' in method:
            x_log = np.log10(x) + samples[e_num-2]
        else:
            x_log = samples[e_num-2]  
            
        x = 10.**(x_log)

        line = f' {e_num:9d}        {x:.6e}   1.000000e-20   1.000000e+20   1.000000e+00         0'
        new_lines.append(line)
    
  
    new_lines = '\n'.join(new_lines)

    with open(template_file, 'w') as f:
        f.writelines(content[0:nn[0]+1])
        f.writelines(new_lines)

    if out:
        print('File '+template_file+' successfully written.')
        print('Number of perturbations', len(samples))
        print('Average perturbation', np.mean(samples))
        print('StdDev perturbation', np.std(samples))

    # return samples


def read_model(model_file=None,  model_trans='log10',  out=True):
    '''
    
    vrath   Sat Jun  7 06:03:58 PM CEST 2025

    '''
#    import numpy as np

    # rng = np.random.default_rng()
    if model_file is None:
        exit('No model file given! Exit.')

    with open(model_file, 'r') as file:
        content = file.readlines()

    nn = content[0].split()
    nn = [int(tmp) for tmp in nn]

    s_num =0
    for elem in range(nn[0]+1, nn[0]+nn[1]+1): 
        s_num = s_num + 1
        x = float(content[elem].split()[1])
        if s_num==1:
            model = [x]
        else:
            model.append(x)
 
    model = np.array(model)    
    # print(model[0], model[nn[1]-1])
    if 'log10' in model_trans:
       print('model is log10 resistivity!')
       model = np.log10(model)
 
    return model



def insert_model(template_file='resistivity_block_iter0.dat',
                  data=None,
                  data_file=None,
                  data_name= "",
                  out=True):
    '''
    Created on Thu Apr 17 17:13:38 2025
    
    @author:     vrath   
    '''
    # import numpy as np
    # rng = np.random.default_rng()
    
    if data is None:
        error("No data given! Exit.")
        
    if data_file is None:
        error("No data file given! Exit.")
        
    if template_file is None:
        template_file = 'resistivity_block_iter0.dat'
        
    data_file = template_file.replace(".dat", data_name+".dat")

    with open(template_file, 'r') as file:
        content = file.readlines()

    nn = content[0].split()
    nn = [int(tmp) for tmp in nn]
    n_cells = nn[1]

    size=n_cells-2
    
    # element groups: air and seawater fixed
    new_lines = [
        '         0        1.000000e+09   1.000000e-20   1.000000e+20   1.000000e+00         1',
        '         1        2.500000e-01   1.000000e-20   1.000000e+20   1.000000e+00         1'
    ]

    print(nn[0], nn[0]+nn[1]-1, nn[1]-1, np.shape(data))
    
    nn = content[0].split()
    nn = [int(tmp) for tmp in nn]

    data = np.power(10.,data)
 
    e_num = 1
    s_num = -1
    for elem in range(nn[0]+3, nn[0]+nn[1]+1):
        e_num = e_num + 1
        s_num = s_num + 1
        x = data[s_num]

        line = f' {e_num:9d}        {x:.6e}   1.000000e-20   1.000000e+20   1.000000e+00         0'
        new_lines.append(line)
    
    new_lines = '\n'.join(new_lines)
    

    with open(data_file, 'w') as f:
        f.writelines(content[0:nn[0]+1])
        f.writelines(new_lines)

    if out:
        print('File '+data_file+' successfully written.')
        print('Number of data replaced', len(data))

    # return samples



def modify_data_fcn(template_file='observe.dat',
                 draw_from=['normal', 0., 1.],
                 scalfac=1.,
                 out=True):
    '''
    Created on Thu Apr 17 17:13:38 2025
    
    @author:   vrath   
    '''
#    import numpy as np

#
    if template_file is None:
        template_file = 'observe.dat'

    print('\n', template_file, ':')

    with open(template_file, 'r') as file:
        content = file.readlines()

    line = content[0].split()
    obs_type = line[0]
    num_site = int(line[1])
    print(len(content))

    '''  
    find data blocks
    
    '''
    start_lines_datablock = []
    for number, line in enumerate(content, 0):
        l = line.split()
        if len(l) == 2:
          start_lines_datablock.append(number)
          print(' data block', l[0], 'with',
                l[1], 'sites begins at line', number)
        if "END" in l:
            start_lines_datablock.append(number-1)
            print(" no further data block in file")
    '''  
     loop over  data blocks
     
    '''
    num_datablock = len(start_lines_datablock)-1
    for block in np.arange(num_datablock):
        start_block = start_lines_datablock[block]
        end_block = start_lines_datablock[block+1]
        # print(start_block, end_block)
        # print(type(start_block), type(end_block))
        data_block = content[start_block:end_block]
        '''  
        find sites
        '''
        print(np.shape(data_block))
        start_lines_site = []
        num_freqs = []
        for number, line in enumerate(data_block, 0):
            l = line.split()
            if len(l) == 4:
              print(l)
              start_lines_site.append(number)
              num_freqs.append(int(data_block[number+1].split()[0]))          
              print('  site', l[0],'begins at line', number)
            if "END" in l:
                 start_lines_datablock.append(number-1)
                 print(" no further site block in file")
        print('\n')   
        # print(start_lines_site)
        # print(num_freqs)
              
        num_sites = len(start_lines_site)
        for site in np.arange(num_sites):
            start_site = start_lines_site[site]
            end_site = start_site+num_freqs[site]+2
            site_block = data_block[start_site:end_site]
            # print('site',site+1) 
            # print(np.shape(site_block))
            # print(site_block)
                        
            if 'MT' in obs_type:

                dat_length = 8
                
                num_freq = int(site_block[1].split()[0])
                print('   site ',site,'has',num_freq,'frequencies' )
                obs  = []
                for line in site_block[2:]:
                    # print(line)
                    tmp = [float(x) for x in line.split()]
                    obs.append(tmp)
                    
                # print('obs',np.shape(obs), np.shape(site_block))   
                # print(obs)
                # print(np.arange(num_freq))
                
                for line in obs:
                     # print(np.arange(1,dat_length+1))
                     # print(freq)
                     for ii in np.arange(1,dat_length+1):
                         print(site, '   ',ii, ii+dat_length)
                         val = line[ii]
                         err = line[ii+dat_length]*scalfac
                         line[ii] = np.random.normal(loc=val, scale=err)
                         
                '''
                now write new values
                
                '''
                print('obs',np.shape(obs), np.shape(site_block))  
                print(np.arange(num_freq))
                for f in  np.arange(num_freq-1):
                    print(f)
                    print( site_block[f+2])
                    print( obs[f])
                    site_block[f+2] = "    ".join([f"{x:.8E}" for x in obs[f]])
                    print( site_block[f+2])     

            elif 'VTF' in obs_type:

                dat_length = 4

                num_freq = int(site_block[1].split()[0])
                print('   site ',site,'has',num_freq,'frequencies' )
                obs  = []
                for line in site_block[2:]:
                    # print(line)
                    tmp = [float(x) for x in line.split()]
                    obs.append(tmp)

                # print('obs',np.shape(obs), np.shape(site_block))
                # print(obs)
                # print(np.arange(num_freq))

                for line in obs:
                     # print(np.arange(1,dat_length+1))
                     # print(freq)
                     for ii in np.arange(1,dat_length+1):
                         print(site, '   ',ii, ii+dat_length)
                         val = line[ii]
                         err = line[ii+dat_length]*scalfac
                         line[ii] = np.random.normal(loc=val, scale=err)

                '''
                now write new values

                '''
                print('obs',np.shape(obs), np.shape(site_block))
                print(np.arange(num_freq))
                for f in  np.arange(num_freq-1):
                    print(f)
                    print( site_block[f+2])
                    print( obs[f])
                    site_block[f+2] = "    ".join([f"{x:.8E}" for x in obs[f]])
                    print( site_block[f+2])
            else:
                
                error(obs_type+' not yet implemented! Exit.')

            data_block[start_site:end_site] = site_block
            
        
        content[start_block:end_block] = data_block            
        

   
    print (np.shape(content))
    with open(template_file, 'w') as f:
        f.writelines(content)


    if out:
        print('File '+template_file+' successfully written.')




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
    '''
    Created on Thu Feb 27 10:23:16 2025
    This creates the files called sites_vtf.txt and sites_imp.txt based on 
    files result_VTF.txt and result_MT.txt as output from applying mergeResultOfFEMTIC 
    to femtic inversion results
    @authors: charroyj + vrath
    '''
    
    
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




def get_femtic_data(data_file=None, site_file=None, data_type='rhophas', out=True):
    '''
    

    Parameters
    ----------
    data_file : TYPE, optional
        DESCRIPTION. The default is None.
    site_file : TYPE, optional
        DESCRIPTION. The default is None.
    data_type : TYPE, optional
        DESCRIPTION. The default is 'rhophas'.
    out : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    data_dict : TYPE
        DESCRIPTION.

   Note: Conversion to appropriate units: FEMTIC uses ohms
         1 ohm = 10000(4*pi) [mV/km/nT]
 
    '''


    data = []
    with open(data_file, 'r') as f:
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
    with open(site_file, 'r') as f:
        for line in f:
            l = line.split(',')
            l[1] = float(l[1])
            l[2] = float(l[2])
            l[3] = float(l[3])
            l[4] = int(l[4])
            # print(l)
            info.append(l)
    info = np.array(info)

    sites = np.unique(data[:, 0]).astype('int')-1

    head_dict = dict([
        ('sites', sites),
        ('frq', data[:, 1]),
        ('per', 1./data[:1]),
        ('lat', np.float64(info[:, 1])),
        ('lon', np.float64(info[:, 2])),
        ('elv', np.float64(info[:, 3])),
        ('num', data[:, 0].astype('int')-1),
        ('nam', info[:, 0][sites.astype('int')-1])
        ])

    if 'rhophas' in data_type.lower():

        '''
         Site      Frequency
         AppRxxCal   PhsxxCal   AppRxyCal   PhsxyCal   AppRyxCal  PhsyxCal  AppRyyCal   PhsyyCal
         AppRxxObs   PhsxxObs   AppRxyObs   PhsxyObs   AppRyxObs  PhsyxObs  AppRyyObs   PhsyyObs
         AppRxxErr   PhsxxErr   AppRxyErr   PhsxyErr   AppRyxErr  PhsyxErr  AppRyyErr   PhsyyErr


        '''
        # print(np.shape(data))
        # print(np.shape(data[:, 2:10 ]))
        # print(np.shape(data[:, 10:18 ]))
        # print(np.shape(data[:, 18:26 ]))
        type_dict = dict([
            ('cal', data[:, 2:10]),
            ('obs', data[:, 10:18 ]),
            ('err', data[:, 18:26]),
        ])

    elif 'imp' in data_type.lower():

        '''
         Site      Frequency
         ReZxxCal   ImZxxCal   ReZxyCal   ImZxyCal   ReZyxCal  ImZyxCal  ReZyyCal   ImZyyCal
         ReZxxObs   ImZxxObs   ReZxyObs   ImZxyObs   ReZyxObs  ImZyxObs  ReZyyObs   ImZyyObs
         ReZxxErr   ImZxxErr   ReZxyErr   ImZxyErr   ReZyxErr  ImZyxErr  ReZyyErr   ImZyyErr
         
         Z_femtic in Ohm: 1 Ohm = 1e4*(4*pi) [mV/km/nT] => Z =  1.e-4/(4*np.pi)*Z_femtic
         
        '''
        ufact = 1.e-4/(4*np.pi)
        type_dict = dict([
            ('cal', ufact*data[:, 2:10 ]),
            ('obs', ufact*data[:, 10:18 ]),
            ('err', ufact*data[:, 18:26]),
        ])

    elif 'vtf' in data_type.lower():

        '''
        Site    Frequency
        ReTzxCal   ImTzxCal   ReTzyCal   ImTzyCal
        ReTzxOb    ImTzxObs   ReTzyObs   ImTzyObs
        ReTzxErr   ImTzxErr   ReTzyErr   ImTzyErr
        '''
        type_dict = dict([
            ('cal', data[:, 2:6 ]),
            ('obs', data[:, 6:10 ]),
            ('err', data[:, 10:15]),

        ])

    elif 'pt' in data_type.lower():
        '''
        Site    Frequency
        ReTzxCal   ImTzxCal   ReTzyCal   ImTzyCal
        ReTzxOb    ImTzxObs   ReTzyObs   ImTzyObs
        ReTzxErr   ImTzxErr   ReTzyErr   ImTzyErr
        '''
        type_dict = dict([
            ('cal', data[:, 2:6 ]),
            ('obs', data[:, 6:18 ]),
            ('err', data[:, 10:14]),

        ])

    else:
        error('get_femtic_data: data type '+data_type.lower()+' not implemented! Exit.')

    data_dict = {**head_dict, **type_dict}

    return data_dict

def get_work_model(directory=None, file=None, out=True):
    
    work_model = []
    return work_model
