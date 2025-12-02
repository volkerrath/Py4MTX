
from __future__ import annotations


import os
import sys
import shutil
import copy
import inspect
import time
from datetime import datetime



from typing import Callable, Optional, Sequence, Tuple, Dict, Literal
from numpy.random import Generator, default_rng

import numpy as np
import scipy
import scipy.linalg
import scipy.sparse.linalg
import scipy.special
import scipy.fftpack
import scipy.sparse

import joblib

from scipy.sparse.linalg import LinearOperator, cg, eigsh



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
        n_samples=1,
        fromto = None,
        out=True):


    if fromto is None:
        from_to = np.arange(n_samples)
    else:
        from_to = np.arange(fromto[0],fromto[1])


    dir_list = []
    for iens in from_to:
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
                           n_samples=1,
                           fromto = None,
                           file_in='observe.dat',
                           draw_from=['normal', 0., 1.],
                           method='add',
                           errors=[],
                           out=True):
    '''
    for i = 1 : nsamples do
        Draw perturbed data set: d_pert ∼ N(d, Cd)

    '''
    if fromto is None:
        fromto = np.arange(n_samples)

    obs_list = []
    for iens in fromto:
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
                errors=[[], [], []],
                out=True):
    '''
    Created on Thu Mpr 17 17:13:38 2025

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
        if 'END' in l:
            start_lines_datablock.append(number-1)
            print(' no further data block in file')
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
                print('  site', l[0], 'begins at line', number)
            if 'END' in l:
                start_lines_datablock.append(number-1)
                print(' no further site block in file')
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

                if len(errors[0]) != 0:
                    set_errors_mt = True

                dat_length = 8

                num_freq = int(site_block[1].split()[0])
                print('   site ', site, 'has', num_freq, 'frequencies')
                obs = []
                for line in site_block[2:]:
                    # print(line)
                    tmp = [float(x) for x in line.split()]
                    obs.append(tmp)

                if set_errors_mt:
                    new_errors = errors[0]
                    print('MT errors will be replaced with relative errors:')
                    print(new_errors)
                    for comp in obs:
                        # print(np.arange(1,dat_length+1))
                        # print(freq)
                        for ii in np.arange(1, dat_length+1):
                            print(site, '   ', ii, ii+dat_length)
                            val = comp[ii]
                            err = val*new_errors
                            comp[ii+dat_length] = err

                for comp in obs:

                    for ii in np.arange(1, dat_length+1):
                        print(site, '   ', ii, ii+dat_length)
                        val = comp[ii]
                        err = comp[ii+dat_length]
                        comp[ii] = np.random.normal(loc=val, scale=err)

                '''
                now write new values

                '''
                print('obs', np.shape(obs), np.shape(site_block))
                print(np.arange(num_freq))
                for f in np.arange(num_freq-1):
                    print(f)
                    print(site_block[f+2])
                    print(obs[f])
                    site_block[f +
                               2] = '    '.join([f'{x:.8E}' for x in obs[f]])+'\n'
                    print(site_block[f+2])

            elif 'VTF' in obs_type:

                if len(errors[1]) != 0:
                    set_errors_vtf = True

                dat_length = 4

                num_freq = int(site_block[1].split()[0])
                print('   site ', site, 'has', num_freq, 'frequencies')
                obs = []
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

                    for comp in obs:
                        # print(np.arange(1,dat_length+1))
                        # print(freq)
                        for ii in np.arange(1, dat_length+1):
                            print(site, '   ', ii, ii+dat_length)
                            val = comp[ii]
                            err = comp[ii+dat_length]
                            comp[ii] = np.random.normal(loc=val, scale=err)

                '''
                now write new values

                '''
                print('obs', np.shape(obs), np.shape(site_block))
                print(np.arange(num_freq))
                for f in np.arange(num_freq-1):
                    print(f)
                    print(site_block[f+2])
                    print(obs[f])
                    site_block[f +
                               2] = '    '.join([f'{x:.8E}' for x in obs[f]])+'\n'
                    print(site_block[f+2])

            elif 'PT' in obs_type:

                if len(errors[2]) != 0:
                    set_errors_pt = True

                dat_length = 4

                num_freq = int(site_block[1].split()[0])
                print('   site ', site, 'has', num_freq, 'frequencies')
                obs = []
                for line in site_block[2:]:
                    # print(line)
                    tmp = [float(x) for x in line.split()]
                    obs.append(tmp)

                    if set_errors_pt:
                        new_errors = errors[1]
                        print('VTF errors will be replaced with relative errors:')
                        print(new_errors)
                        for comp in obs:
                            # print(np.arange(1,dat_length+1))
                            # print(freq)
                            for ii in np.arange(1, dat_length+1):
                                print(site, '   ', ii, ii+dat_length)
                                val = comp[ii]
                                err = new_errors
                                comp[ii+dat_length] = err

                    for comp in obs:
                        # print(np.arange(1,dat_length+1))
                        # print(freq)
                        for ii in np.arange(1, dat_length+1):
                            print(site, '   ', ii, ii+dat_length)
                            val = comp[ii]
                            err = comp[ii+dat_length]
                            comp[ii] = np.random.normal(loc=val, scale=err)

                '''
                now write new values

                '''
                print('obs', np.shape(obs), np.shape(site_block))
                print(np.arange(num_freq))
                for f in np.arange(num_freq-1):
                    print(f)
                    print(site_block[f+2])
                    print(obs[f])
                    site_block[f +
                               2] = '    '.join([f'{x:.8E}' for x in obs[f]])+'\n'
                    print(site_block[f+2])
            else:

                sys.exit('modify_data:'+obs_type+' not yet implemented! Exit.')

            data_block[start_site:end_site] = site_block

        content[start_block:end_block] = data_block

    print(np.shape(content))
    with open(template_file, 'w') as f:
        f.writelines(content)

    if out:
        print('File '+template_file+' successfully written.')


def generate_model_ensemble(dir_base='./ens_',
                            n_samples=1,
                            fromto = None,
                            refmod='resistivity_block_iter0.dat',
                            q=None,
                            method='add',
                            out=True):
    '''

    Generate perturbed model based on precision matrix.

    See:

    Rue, H. & Held, L., 2005.
        Gaussian Markov Random Fields: Theory and Applications.
        Monographs on Statistics and Applied Probability, Vol. 104,
        Chapman and Hall/CRC. doi:10.1201/9780203492024



    Parameters
    ----------
    dir_base : TYPE, optional
        DESCRIPTION. The default is './ens_'.
    n_samples : TYPE, optional
        DESCRIPTION. The default is 1.
    fromto : TYPE, optional
        DESCRIPTION. The default is None.
    refmod : TYPE, optional
        DESCRIPTION. The default is 'resistivity_block_iter0.dat'.
    q : TYPE, optional
        DESCRIPTION. The default is None.
    method : TYPE, optional
        DESCRIPTION. The default is 'add'.
    out : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    mod_list : TYPE
        DESCRIPTION.

    '''

    samples = sample_gaussian_precision_rtr(
        R=q,
        n_samples=n_samples,
        lam = 0.0,)

    mod_list = []
    for iens in np.arange(n_samples):
        file = dir_base+str(iens)+'/'+refmod
        shutil.copy(file, file.replace('.dat', '_orig.dat'))
        '''
        generate perturbed model
        '''
        insert_model(refmod =refmod,
                         data=samples[iens,:],
                         data_file=None,
                         data_name='sample'+str(iens))
        mod_list.append(file)

    if out:
        print('\n')
        print('list of perturbed model files:')
        print(mod_list)

    return mod_list


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

    s_num = 0
    for elem in range(nn[0]+1, nn[0]+nn[1]+1):
        s_num = s_num + 1
        x = float(content[elem].split()[1])
        if s_num == 1:
            model = [x]
        else:
            model.append(x)

    model = np.array(model)
    # print(model[0], model[nn[1]-1])
    if 'log10' in model_trans:
        print('model is log10 resistivity!')
        model = np.log10(model)

    return model



def modify_model(template_file='resistivity_block_iter0.dat',
                 draw_from=['normal', 0., 1.],
                 method='add',
                 q=None,
                 decomposed=False,
                 regeps=1.e-8,
                 out=True):
    '''
    Created on Thu Mpr 17 17:13:38 2025

    @author:       vrath
    '''

    # def sample_gaussian_precision_rtr(
    #     R: np.ndarray | "scipy.sparse.spmatrix",
    #     n_samples: int = 1,
    #     lam: float = 0.0,
    #     solver: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    #     rng: Optional[Generator] = None,
    # ) -> np.ndarray:
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
        print('Mverage perturbation', np.mean(samples))
        print('StdDev perturbation', np.std(samples))

    # return samples


def insert_model(template='resistivity_block_iter0.dat',
                 data=None,
                 data_file=None,
                 data_name='',
                 out=True):
    '''
    Created on Thu Mpr 17 17:13:38 2025

    @author:     vrath
    '''
    # import numpy as np
    # rng = np.random.default_rng()

    if data is None:
        sys.exit('insert_model: No data given! Exit.')


    if template is None:
        template = 'resistivity_block_iter0.dat'


    if data_file is None:
        data_file = template

    with open(template, 'r') as file:
        content = file.readlines()

    nn = content[0].split()
    nn = [int(tmp) for tmp in nn]
    n_cells = nn[1]

    size = n_cells-2

    # element groups: air and seawater fixed
    new_lines = [
        '         0        1.000000e+09   1.000000e-20   1.000000e+20   1.000000e+00         1',
        '         1        2.500000e-01   1.000000e-20   1.000000e+20   1.000000e+00         1'
    ]

    print(nn[0], nn[0]+nn[1]-1, nn[1]-1, np.shape(data))

    nn = content[0].split()
    nn = [int(tmp) for tmp in nn]

    data = np.power(10., data)

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
    Created on Thu Mpr 17 17:13:38 2025

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
        if 'END' in l:
            start_lines_datablock.append(number-1)
            print(' no further data block in file')
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
                print('  site', l[0], 'begins at line', number)
            if 'END' in l:
                start_lines_datablock.append(number-1)
                print(' no further site block in file')
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
                print('   site ', site, 'has', num_freq, 'frequencies')
                obs = []
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
                    for ii in np.arange(1, dat_length+1):
                        print(site, '   ', ii, ii+dat_length)
                        val = line[ii]
                        err = line[ii+dat_length]*scalfac
                        line[ii] = np.random.normal(loc=val, scale=err)

                '''
                now write new values

                '''
                print('obs', np.shape(obs), np.shape(site_block))
                print(np.arange(num_freq))
                for f in np.arange(num_freq-1):
                    print(f)
                    print(site_block[f+2])
                    print(obs[f])
                    site_block[f+2] = '    '.join([f'{x:.8E}' for x in obs[f]])
                    print(site_block[f+2])

            elif 'VTF' in obs_type:

                dat_length = 4

                num_freq = int(site_block[1].split()[0])
                print('   site ', site, 'has', num_freq, 'frequencies')
                obs = []
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
                    for ii in np.arange(1, dat_length+1):
                        print(site, '   ', ii, ii+dat_length)
                        val = line[ii]
                        err = line[ii+dat_length]*scalfac
                        line[ii] = np.random.normal(loc=val, scale=err)

                '''
                now write new values

                '''
                print('obs', np.shape(obs), np.shape(site_block))
                print(np.arange(num_freq))
                for f in np.arange(num_freq-1):
                    print(f)
                    print(site_block[f+2])
                    print(obs[f])
                    site_block[f+2] = '    '.join([f'{x:.8E}' for x in obs[f]])
                    print(site_block[f+2])
            else:

                sys.exit('modify_data: '+obs_type +
                         ' not yet implemented! Exit.')

            data_block[start_site:end_site] = site_block

        content[start_block:end_block] = data_block

    print(np.shape(content))
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
                     vtf_file='result_VTF.txt',
                     pt_file='results_PT.txt'):
    '''
    Created on Thu Feb 27 10:23:16 2025
    This creates the files called sites_vtf.txt and sites_imp.txt based on
    files result_VTF.txt and result_MT.txt as output from applying mergeResultOfFEMTIC
    to femtic inversion results
    @authors: charroyj + vrath
    '''

    # neither inputs nor outputs should normally need to be changed.

    if len(imp_file) > 0 and os.path.exists(imp_file):
        with open(imp_file, 'r') as filein_imp:
            site = ''
            fileout_imp = open(imp_file.replace('results', 'sites'), 'w')
            filein_imp.readline()
            for line in filein_imp:
                nextsite = line.strip().split()[0]
                if nextsite != site:
                    fileout_imp.write(nextsite+' '+nextsite+'\n')
                    site = nextsite
            fileout_imp.close()
    else:
        if len(imp_file) > 0:
            print(imp_file, 'does not exist!')
        else:
            print('pt_file not defined!')

    if len(vtf_file) > 0 and os.path.exists(vtf_file):
        with open(vtf_file, 'r') as filein_vtf:
            site = ''
            fileout_vtf = open(vtf_file.replace('results', 'sites'), 'w')
            filein_imp.readline()
            filein_vtf.readline()
            for line in filein_vtf:
                nextsite = line.strip().split()[0]
                if nextsite != site:
                    fileout_vtf.write(nextsite+' '+nextsite+'\n')
                    site = nextsite
            fileout_vtf.close()
    else:
        if len(vtf_file) > 0:
            print(vtf_file, 'does not exist!')
        else:
            print('vtf_file does not exist!')

    if len(pt_file) > 0 and os.path.exists(pt_file):
        with open(pt_file, 'r') as filein_pt:
            site = ''
            fileout_pt = open(vtf_file.replace('results', 'sites'), 'w')
            filein_pt.readline()
            for line in filein_pt:
                nextsite = line.strip().split()[0]
                if nextsite != site:
                    fileout_pt.write(nextsite+' '+nextsite+'\n')
                    site = nextsite
            fileout_pt.close()
    else:
        if len(vtf_file) > 0:
            print(pt_file, 'does not exist!')
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
         MppRxxCal   PhsxxCal   MppRxyCal   PhsxyCal   MppRyxCal  PhsyxCal  MppRyyCal   PhsyyCal
         MppRxxObs   PhsxxObs   MppRxyObs   PhsxyObs   MppRyxObs  PhsyxObs  MppRyyObs   PhsyyObs
         MppRxxErr   PhsxxErr   MppRxyErr   PhsxyErr   MppRyxErr  PhsyxErr  MppRyyErr   PhsyyErr


        '''
        # print(np.shape(data))
        # print(np.shape(data[:, 2:10 ]))
        # print(np.shape(data[:, 10:18 ]))
        # print(np.shape(data[:, 18:26 ]))
        type_dict = dict([
            ('cal', data[:, 2:10]),
            ('obs', data[:, 10:18]),
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
            ('cal', ufact*data[:, 2:10]),
            ('obs', ufact*data[:, 10:18]),
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
            ('cal', data[:, 2:6]),
            ('obs', data[:, 6:10]),
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
            ('cal', data[:, 2:6]),
            ('obs', data[:, 6:18]),
            ('err', data[:, 10:14]),

        ])

    else:
        sys.exit('get_femtic_data: data type ' +
                 data_type.lower()+' not implemented! Exit.')

    data_dict = {**head_dict, **type_dict}

    return data_dict

# def get_work_model(directory=None, file=None, out=True):

#     work_model = []
#     return work_model


def centroid_tetrahedron(nodes=None):
    '''
    Created on Thu Jul 17 08:36:04 2025

    @author: vrath
    '''

    if nodes is None:
        sys.exit('centroid: Nodes not set! Exit.')

    if np.shape(nodes) != [3, 4]:
        sys.exit('centroid: Nodes shape is not (3,4)! Exit.')

    # nodes = np.nan*np.zeros((3,4))

    centre = np.mean(nodes, axis=0)

    return centre


def get_roughness(filerough='roughening_matrix.out',
                  regeps=None,
                  spformat='csc',
                  out=True):
    '''
    generate prior covariance for
    ensemble perturbations

    Note: does not include air/sea/distortion parameters!

    Parameters
    ----------
    filerough : string
        name of femtic roughness file . The default is None.

    Returns
    -------
    r or rtr : np.array

    author: vrath,  created on Thu Jul 21, 2025

    ResistivityBlock.cpp. l1778ff

    RougheningMatrix.cpp/RougheningMatrix.cpp: 4
     57: 24: void RougheningMatrix::setStructureMndMddValueByTripletFormat( const int row, const int col, const double val ){
     58: 22: 	DoubleSparseMatrix::setStructureMndMddValueByTripletFormat( row, col, val );
     80: 15: 				RTRMatrix.setStructureMndMddValueByTripletFormat(row, col, value);
     86: 13: 		RTRMatrix.setStructureMndMddValueByTripletFormat(iCol, iCol, smallValueOnDiagonals);
    RougheningMatrix.h/RougheningMatrix.h: 1
     47: 15: 	virtual void setStructureMndMddValueByTripletFormat( const int row, const int col, const double val );
    RougheningSquareMatrix.cpp/RougheningSquareMatrix.cpp: 4
     58: 30: void RougheningSquareMatrix::setStructureMndMddValueByTripletFormat( const int row, const int col, const double val ){
     59: 22: 	DoubleSparseMatrix::setStructureMndMddValueByTripletFormat( row, col, val );
     81: 15: 				RTRMatrix.setStructureMndMddValueByTripletFormat(row, col, value);
     87: 13: 		RTRMatrix.setStructureMndMddValueByTripletFormat(iRow, iRow, smallValueOnDiagonals);
    RougheningSquareMatrix.h/RougheningSquareMatrix.h: 1
     47: 15: 	virtual void setStructureMndMddValueByTripletFormat( const int row, const int col, const double val );


    // *******************************************************************************************************
    // Calculate roughning matrix from user-defined roughning factor
    void ResistivityBlock::calcRougheningMatrixUserDefined( const double factor ){

        // Read user-defined roughening matrix
        const std::string fileName = 'roughening_matrix.dat';
        std::ifstream ifs( fileName.c_str(), std::ios::in );

        if( ifs.fail() ){
                OutputFiles::m_logFile << 'File open error : ' << fileName.c_str() << ' !!' << std::endl;
                exit(1);
        }

        OutputFiles::m_logFile << '# Read user-defined roughening matrix from ' << fileName.c_str() << '.' << std::endl;

        int ibuf(0);RoughType
        ifs >> ibuf;
        const int numBlock(ibuf);
        if( numBlock <= 0 ){
                OutputFiles::m_logFile << 'Error : Total number of resistivity blocks must be positive !! : ' << numBlock << std::endl;
                exit(1);
        }

        for( int iBlock = 0 ; iBlock < numBlock; ++iBlock ){
                ifs >> ibuf;
                if( iBlock != ibuf ){
                        OutputFiles::m_logFile << 'Error : Resistivity block numbers must be numbered consecutively from zero !!' << std::endl;
                        exit(1);
                }

                ifs >> ibuf;
                const int numNonzeros(ibuf);
                std::vector< std::pair<int, double> > blockIDMndFactor;
                blockIDMndFactor.resize(numNonzeros);
                for( int innz = 0 ; innz < numNonzeros; ++innz ){
                        ifs >> ibuf;
                        blockIDMndFactor[innz].first = ibuf;
                }
                for( int innz = 0 ; innz < numNonzeros; ++innz ){
                        double dbuf(0.0);
                        ifs >> dbuf;
                        blockIDMndFactor[innz].second = dbuf;
                }
                for( int innz = 0 ; innz < numNonzeros; ++innz ){
                        m_rougheningMatrix.setStructureMndMddValueByTripletFormat( iBlock, blockIDMndFactor[innz].first, blockIDMndFactor[innz].second );
                }
        }


        ifs.close();

    }

    '''
    from scipy.sparse import csr_array, csc_array, coo_array, eye_array

    start = time.perf_counter()
    print('get_roughness: Reading from', filerough)
    irow = []
    icol = []
    vals = []
    with open(filerough, 'r') as file:
        content = file.readlines()

    num_elem = int(content[0].split()[0])
    print('get_roughness: File read:', time.perf_counter() - start, 's')
    print('get_roughness: Number of elements:', num_elem)

    iline = 0
    zeros = 0
    while iline < len(content)-1:  # -2
        iline = iline + 1
        # print(content[iline])
        ele = int(content[iline].split()[0])
        nel = int(content[iline+1].split()[0])
        if nel == 0:
            iline = iline + 1
            zeros = zeros + 1
            print('passed', ele, nel, iline)
        else:
            iline = iline + 2

    print('Zero elements:', zeros)

    start = time.perf_counter()
    iline = 0
    while iline < len(content)-1:  # -2
        iline = iline + 1
        # print(content[iline])
        ele = int(content[iline].split()[0])
        nel = int(content[iline+1].split()[0])
        if nel == 0:
            iline = iline + 1
            # print('passed', ele, nel, iline)
            # pass
            continue
        else:
            irow += [ele-zeros]*nel
            col = [int(x)-zeros for x in content[iline+1].split()[1:]]
            icol += col
            val = [float(x) for x in content[iline+2].split()]
            vals += val
            iline = iline + 2
            # print('used', ele, nel, iline, val)

    # print(irow[0],icol[0])
    irow = np.asarray(irow)
    icol = np.asarray(icol)
    vals = np.asarray(vals)

    R = coo_array((vals, (irow, icol)))
    print(R.shape)

    print('get_roughness: R sparse format is', R.format)

    if regeps is not None:
        R = R + regeps*eye_array(R.shape[0], format=R.format)
        if out:
            print(regeps, 'added to diag(R)')

    print('get_roughness: R generated:', time.perf_counter() - start, 's')
    if out:
        print('get_roughness: R sparse format is', R.format)
        print(R.shape, R.nnz)

    if 'csc' in spformat.lower():
        R = csc_array((vals, (irow, icol)))
    elif 'csr' in spformat.lower():
        R = csr_array((vals, (irow, icol)))
    else:
        R = coo_array((vals, (irow, icol)))

    if out:
        print('get_roughness: Output sparse format:', spformat)
        print('get_roughness: R sparse format is', R.format)
        print(R.shape, R.nnz)
        print(R.nnz, 'nonzeros, ', 100*R.nnz/R.shape[0]**2, '%')

        print('get_roughness: Done!\n\n')

    return R


def make_prior_cov(rough=None,
                   regeps=1.e-5,
                   spformat='csr',
                   spthresh=1.e-4,
                   spfill=10.,
                   spsolver=None,
                   spmeth='basic,area',
                   outmatrix='invRTR',
                   nthreads=16,
                   out=True):
    '''
    Generate prior covariance for ensemble perturbations

    Note: does not include air/sea/distortion parameters!

    Parameters
    ----------
    rough : sparse array
        Name of femtic roughness. (Default: None).
    regeps : float
        Small value to stabilize. (Default: 1.e-5).
    spsolver: str
        Available:
            'slu'/sparse LU
            'ilu'/sparse incomplete LU
    spformat : str
        Output sparse format from list ['csr','coo', 'csc'].
        (Default: 'csr')
    spthresh : float
        Threshold for drops in ILU decomposition. (Default: 1.e-4).
    spfill : float
        Max Fill factor. (Default: 10.).
    spmeth: list of str
        Comma-separated string of drop rules to use.
        Available rules: basic, prows, column, area, secondary, dynamic, interp.
        (Default: 'basic,area').
   outmatrix: str
        Available: 'invR', 'invRTR'

    Returns
    -------
    M: sparse array
        femtic equivalent of covariance C or inverse of invR


    author: vrath,  created on Thu Jul 26, 2025


    '''

    from scipy.sparse import csr_array, csc_array, coo_array, eye_array, diags_array, issparse
    from threadpoolctl import threadpool_limits

    if rough is None:
        sys.exit('make_prior_cov: No roughness matrix given! Exit.')

    if not issparse(rough):
        exit('make_prior_cov: Roughness matrix is not sparse! Exit.')

    start = time.perf_counter()

    if out:
        print('make_prior_cov: Shape of input roughness is', rough.shape)
        print('make_prior_cov: Format of input roughness is', rough.format)

    if regeps is not None:
        rough = rough + regeps * \
            eye_array(rough.shape[0], format=spformat.lower())
        if out:
            print(regeps, 'added to diag(R)')

    if 'slu' in spsolver.lower():
        from scipy.sparse.linalg import spsolve

        R = csc_array(rough)
        RHS = eye_array(R.shape[0], format=R.format)
        with threadpool_limits(limits=nthreads):
            invR = spsolve(R, RHS)

    elif 'ilu' in spsolver.lower():
        from scipy.sparse.linalg import spilu

        R = csc_array(rough)
        RHS = eye_array(R.shape[0], format=R.format)
        # RHS = np.eye(R.shape[0])
        beg = time.perf_counter()

        with threadpool_limits(limits=nthreads):
            iluR = spilu(R, drop_tol=spthresh, fill_factor=spfill)
            print('spilu decomposed:', time.perf_counter() - beg, 's')

            beg = time.perf_counter()
            invR = iluR.solve(RHS.toarray())
            print('spilu solved:', time.perf_counter() - beg, 's')

    else:
        sys.exit('make_prior_cov: solver' +
                 spsolver.lower()+'not available! Exit')

    if out:
        print('make_prior_cov: invR generated:',
              time.perf_counter() - start, 's')
        print('make_prior_cov: invR type', type(invR))
        # print('invR format', invR.format)

    if spthresh is not None:
        invR = matrix_reduce(M=invR,
                             spthresh=spthresh,
                             spformat=spformat)

    M = invR
    if 'rtr' in outmatrix.lower():
        M = invR@invR.T
        if 'deco' in outmatrix.lower():
            from inverse import msqrt_sparse
            # calculate cholesky factor of M
            M = msqrt_sparse(M)

    if out:

        print('make_prior_cov: M generated:', time.perf_counter() - start, 's')
        print('make_prior_cov: M is', outmatrix)
        print('make_prior_cov: M', type(M))
        print('M', M.format)

    print('make_prior_cov:  Done!\n\n')

    return M


def prune_inplace(M, threshold):
    from scipy.sparse import csr_array, issparse
    # ensure CSR for data/indices/indptr access
    if issparse(M):
        if not M.format == 'csr':  # isspmatrix_csr(M):
            M = M.tocsr()
    else:
        M = csr_array(M)

    # mark tiny entries as explicit zeros in data array
    mask = np.abs(M.data) < threshold
    if mask.any():
        M.data[mask] = 0
        M.eliminate_zeros()
    return M


def prune_rebuild(M, threshold):
    from scipy.sparse import csr_array, issparse
    # convert to COO format (triplet)
    coo = M.tocoo()
    absdata = np.abs(coo.data)
    keep = absdata >= threshold
    if not keep.all():
        # build new CSR from filtered coordinates
        return csr_array((coo.data[keep], (coo.row[keep], coo.col[keep])),
                      shape=M.shape)
    else:
        return M.tocsr()

def dense_to_csr(M, threshold=0.0, chunk_rows=1000, dtype=None):
    from scipy.sparse import csr_array
    # from collections import deque
    rows_list = []
    cols_list = []
    data_list = []
    nrows = M.shape[0]
    for r0 in range(0, nrows, chunk_rows):
        r1 = min(nrows, r0 + chunk_rows)
        block = M[r0:r1]
        mask = np.abs(block) > threshold
        rr, cc = np.nonzero(mask)
        rows_list.append((rr + r0).astype(np.int64))
        cols_list.append(cc.astype(np.int64))
        data_list.append(block[rr, cc].astype(dtype if dtype is not None else M.dtype))

    if not rows_list:
        return csr_array(M.shape, dtype=dtype if dtype is not None else M.dtype)

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    data = np.concatenate(data_list)
    return csr_array((data, (rows, cols)), shape=M.shape)


def save_spilu(filename='ILU.npz', ILU=None):
    '''
    Save spilu decomposition (ILU object) to a single .npz file."""

    Parameters
    ----------
    filename : str, optional
        The default is 'ILU.npz'.
    ILU : ILU object
        The default is None.

    Returns
    -------
    None.

    Load with:
        load_spilu(ILU=ILU)


    vrath + copilot  Oct 22, 2025

    '''
    if ILU is None:
        sys.exit('No ILU object given! Exit.')

    np.savez(filename,
             L_data=ILU.L.data, L_indices=ILU.L.indices,
             L_indptr=ILU.L.indptr, L_shape=ILU.L.shape,
             U_data=ILU.U.data, U_indices=ILU.U.indices,
             U_indptr=ILU.U.indptr, U_shape=ILU.U.shape,
             perm_r=ILU.perm_r, perm_c=ILU.perm_c)


def load_spilu(filename='ILU.npz'):
    '''
    Load spilu decomposition components from a .npz file.

    Parameters
    ----------
    filename : str
        npz file (default is 'ILU.npz')

    Returns
    -------
    L, U, perm_r, perm_c :
        Data from ILU decomposition object

    vrath + copilot  Oct 22, 2025
    '''
    from scipy.sparse import csc_array

    data = np.load(filename)
    L = csc_array((data["L_data"], data["L_indices"], data["L_indptr"]),
                  shape=tuple(data["L_shape"]))
    U = csc_array((data["U_data"], data["U_indices"], data["U_indptr"]),
                  shape=tuple(data["U_shape"]))
    perm_r = data["perm_r"]
    perm_c = data["perm_c"]

    return L, U, perm_r, perm_c


def matrix_reduce(M=None,
                  howto='relative',
                  spformat='csr',
                  spthresh=1.e-6,
                  prune='rebuild',
                  out=True):

    from scipy.sparse import csr_array, csc_array, coo_array, issparse

    if M is None:
        sys.exit('matrix_reduce: no matrix given! Exit.')

    n, _ = np.shape(M)

    if issparse(M):
        M = M.tocsr()  # coo_array(M)
        if out:
            print('matrix_reduce: Matrix is sparse.')
            print('matrix_reduce: Type:', type(M))
            print('matrix_reduce: Format:', M.format)
            print('matrix_reduce: Shape:', M.shape)

    else:
        M = csr_array(M)  # coo_array(M)
        if out:
            print('matrix_reduce: Matrix is dense.')
            print('matrix_reduce: Type:', type(M))
            print('matrix_reduce: Shape:', np.shape(M))

    if out:
        print('matrix_reduce:', M.nnz, 'nonzeros, ', 100*M.nnz/n**2, '%')

    # test = M - M.T
    # if test.max()+test.min()==0.:
    #     if out: print('matrix_reduce: Matrix is symmetric!')
    # else:
    #     if out: print('matrix_reduce: Matrix is not symmetric!')

    if 'abs' in howto.lower():
        # Define absolute threshold
        threshold = spthresh
    else:
        # Define relative threshold (e.g., 1% of max value)
        maxM = np.max(np.abs(M.data))
        threshold = spthresh * maxM

    if issparse(M):
        # Zero out elements below threshold
        if 'in' in prune:
            M = prune_inplace(M, threshold)
        else:
            M = prune_rebuild(M, threshold)
    else:
        M = dense_to_csr(M, threshold=threshold, chunk_rows=10000)

    if 'csr' in spformat.lower():
        M = M.tocsr()  # csr_array(M)
    if 'csc' in spformat.lower():
        M = M.tocsc()  # csc_array(M)
    if 'coo' in spformat.lower():
        M = M.tocoo()  # coo_array(M)

    if out:

        print('matrix_reduce: New Format:', M.format)
        print('matrix_reduce: Shape:', M.shape)
        print('matrix_reduce:', M.nnz, 'nonzeros, ', 100*M.nnz/n**2, '%')

    check_sparse_matrix(M)

    print('matrix_reduce: Done!\n\n')
    return M


def check_sparse_matrix(M, condition=True):
    '''
    Check sparse matrix

    Parameters
    ----------
    M : sparse array
        Matrix to be tested
    Returns
    -------
    None.

    '''

    from scipy.sparse import csr_array, csc_array, coo_array, issparse
    from scipy.sparse import diags_array

    if M is None:
        sys.exit('check_sparse_matrix: No roughness matrix given! Exit.')

    if not issparse(M):
        sys.exit('check_sparse_matrix: Roughness matrix is not sparse! Exit.')

    print('check_sparse_matrix: Type:', type(M))
    print('check_sparse_matrix: Format:', M.format)
    print('check_sparse_matrix: Shape:', M.shape)
    print('check_sparse_matrix:', M.nnz, 'nonzeros, ',
          100*M.nnz/M.shape[0]**2, '%')

    if M.shape[0] == M.shape[1]:
        print('check_sparse_matrix: Matrix is square!')
        test = M - M.T
        print('   R-R^T max/min:', test.max(), test.min())
        if test.max()+test.min() == 0.:
            print('check_sparse_matrix: Matrix is symmetric!')
        else:
            print('check_sparse_matrix: Matrix is not symmetric!')

    maxaM = np.amax(np.abs(M))
    minaM = np.amin(np.abs(M))
    print('check_sparse_matrix: M max/min:', M.max(), M.min())
    print('check_sparse_matrix: M abs max/min:', maxaM, minaM)

    if np.any(np.abs(M.diagonal(0)) == 0):
        print('check_sparse_matrix: M diagonal element is 0!')
        print(np.abs(M.diagonal(0) == 0).nonzero())
        print(np.abs(M.diagonal(0) == 0))

    print('check_sparse_matrix: Done!\n\n')
    # condition = ???

def sampler1(args):

    '''
        import argparse
        import numpy as np




        ap = argparse.ArgumentParser(description="Sample a covariance-driven resistivity field on a FEMTIC TETRA mesh.")
        ap.add_argument("--mesh", required=True, help="Path to FEMTIC mesh.dat (TETRA format).")
        ap.add_argument("--kernel", default="matern", choices=["matern", "exponential", "gaussian"])
        ap.add_argument("--ell", type=float, default=500.0, help="Length-scale (units of mesh coordinates).")
        ap.add_argument("--sigma2", type=float, default=0.5, help="Log-space marginal variance.")
        ap.add_argument("--nu", type=float, default=1.5, help="Matern smoothness (if used).")
        ap.add_argument("--nugget", type=float, default=1e-6, help="Diagonal nugget (log-space).")
        ap.add_argument("--mean", type=float, default=float(np.log(100.0)), help="Mean of log-resistivity.")
        ap.add_argument("--strategy", default="sparse", choices=["dense", "sparse"], help="Dense (Cholesky) or sparse (trunc-eig).")
        ap.add_argument("--radius", type=float, default=None, help="Neighborhood radius for sparse K (default ~ 2.5*ell).")
        ap.add_argument("--trunc_k", type=int, default=1024, help="Rank for truncated-eig sampling when sparse.")
        ap.add_argument("--seed", type=int, default=None, help="Random seed.")
        ap.add_argument("--out", default="rho_sample_on_mesh.npz", help="Output NPZ path.")
        args = ap.parse_args()
    '''
    from femtic_mesh_io import read_femtic_tetra_centroids
    from femtic_sample_resistivity import draw_logrho_field

    centroids, tet_ids = read_femtic_tetra_centroids(args.mesh)

    if args.strategy == "dense":
        rho, logrho = draw_logrho_field(
            centroids,
            kernel=args.kernel,
            sigma2=args.sigma2,
            ell=args.ell,
            nu=args.nu,
            nugget=args.nugget,
            mean_log_rho=args.mean,
            strategy="dense",
            random_state=args.seed,
        )
    else:
        radius = args.radius if args.radius is not None else 2.5 * args.ell
        rho, logrho = draw_logrho_field(
            centroids,
            kernel=args.kernel,
            sigma2=args.sigma2,
            ell=args.ell,
            nu=args.nu,
            nugget=args.nugget,
            mean_log_rho=args.mean,
            strategy="sparse",
            radius=radius,
            trunc_k=args.trunc_k,
            random_state=args.seed,
        )

    np.savez(
        args.out,
        rho=rho,
        logrho=logrho,
        centroids=centroids,
        tet_ids=tet_ids,
        meta=dict(vars(args)),
    )

"""Gaussian sampling with precision matrix Q = R.T @ R.

This module provides tools to sample from multivariate Gaussian distributions
with zero mean and covariance C = (R.T @ R + lambda * I)^{-1}, where R is a
large, typically sparse matrix. The focus is on matrix-free methods that avoid
forming the covariance explicitly and that can exploit sparse linear algebra.

Main ideas
----------
1. Treat Q = R.T @ R (+ lambda * I) as a precision matrix and work with Q
   via matrix-vector products rather than forming C explicitly.
2. Use conjugate gradients (CG) on Q to solve linear systems Q x = b, which
   is the core operation in many simulation algorithms.
3. Optionally use a low-rank approximation based on eigenpairs of Q for
   reduced-rank or smoothed sampling.

The functions in this module are designed to be reasonably general while
remaining explicit about the underlying numerical linear algebra.

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-11-16
"""

def build_rtr_operator(
    R: np.ndarray | "scipy.sparse.spmatrix",
    lam: float = 0.0,
) -> LinearOperator:
    """Create a LinearOperator representing Q = R.T @ R + lam * I.

    Parameters
    ----------
    R : array_like or sparse matrix, shape (m, n)
        Matrix defining the precision Q = R.T @ R. R is typically sparse.
    lam : float, optional
        Diagonal Tikhonov regularisation parameter. If non-zero, the operator
        represents Q = R.T @ R + lam * I, corresponding to covariance
        C = (R.T @ R + lam * I)^{-1}. The default is 0.0.

    Returns
    -------
    Q_op : scipy.sparse.linalg.LinearOperator, shape (n, n)
        Linear operator that applies Q to a vector via matrix-vector products.

    Notes
    -----
    This function avoids forming Q explicitly. The matvec uses two sparse
    matrix-vector products: y = R @ x and z = R.T @ y, plus a possible
    diagonal shift lam * x.

    This is suitable for use with iterative solvers such as conjugate
    gradients (CG).

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-11-16
    """
    R = R  # no copy; caller controls storage
    m, n = R.shape

    def matvec(x: np.ndarray) -> np.ndarray:
        """Matrix-vector product z = Q @ x."""
        y = R @ x
        z = R.T @ y
        if lam != 0.0:
            z = z + lam * x
        return z

    return LinearOperator((n, n), matvec=matvec, rmatvec=matvec, dtype=np.float64)


def make_cg_precision_solver(
    R: np.ndarray | "scipy.sparse.spmatrix",
    lam: float = 0.0,
    rtol: float = 1e-6,
    atol: float =0.,
    maxiter: Optional[int] = None,
    M: Optional[LinearOperator] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Construct a solver for Q x = b with Q = R.T @ R + lam * I using CG.

    Parameters
    ----------
    R : array_like or sparse matrix, shape (m, n)
        Matrix defining the precision Q = R.T @ R (+ lam * I). R is typically
        sparse and should be such that Q is symmetric positive definite.
    lam : float, optional
        Diagonal Tikhonov regularisation parameter. The default is 0.0.
    tol : float, optional
        Relative tolerance for the conjugate-gradient solver. The default is
        1e-8.
    maxiter : int, optional
        Maximum number of CG iterations. If None, SciPy chooses a default.
        The default is None.
    M : scipy.sparse.linalg.LinearOperator, optional
        Preconditioner for Q. If provided, M should approximate Q^{-1} in
        some sense and be inexpensive to apply. The default is None.

    Returns
    -------
    solve_Q : callable
        Function ``solve_Q(b: np.ndarray) -> np.ndarray`` that returns the
        CG solution x of Q x = b.

    Notes
    -----
    This wrapper hides the SciPy interface and provides a convenient closure
    that can be used inside sampling routines. For badly conditioned systems
    it is recommended to supply an appropriate preconditioner M to improve
    convergence.


    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-11-16
    """
    Q_op = build_rtr_operator(R, lam=lam)

    def solve_Q(b: np.ndarray) -> np.ndarray:
        """Solve Q x = b with conjugate gradients."""
        x, info = cg(Q_op, b, rtol=rtol, maxiter=maxiter, M=M)
        if info != 0:
            raise RuntimeError(f"CG did not converge, info={info}")
        return x

    return solve_Q


def sample_gaussian_precision_rtr(
    R: np.ndarray | "scipy.sparse.spmatrix",
    n_samples: int = 1,
    lam: float = 0.0,
    solver: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    rng: Optional[Generator] = None,
) -> np.ndarray:
    """Sample from N(0, C) with C = (R.T @ R + lam * I)^{-1}.

    Parameters
    ----------
    R : array_like or sparse matrix, shape (m, n)
        Matrix defining the precision Q = R.T @ R (+ lam * I). R should have
        full column rank (or be regularised via lam > 0) so that Q is
        positive definite.
    n_samples : int, optional
        Number of independent samples to generate. The default is 1.
    lam : float, optional
        Diagonal Tikhonov regularisation parameter. If non-zero, the precision
        is Q = R.T @ R + lam * I and the covariance is
        C = (R.T @ R + lam * I)^{-1}. The default is 0.0.
    solver : callable, optional
        Existing solver for Q x = b. If None, a CG-based solver is created
        via :func:`make_cg_precision_solver`. The default is None.
    rng : numpy.random.Generator, optional
        Random number generator to use. If None, ``default_rng()`` is used.

    Returns
    -------
    samples : ndarray, shape (n_samples, n)
        Array of Gaussian samples. Each row is one draw x ~ N(0, C).

    Notes
    -----
    The sampling algorithm uses the identity

        x = argmin_y ||R y - xi||_2^2

    with xi ~ N(0, I_m), which yields

        x = (R.T @ R + lam * I)^{-1} R.T @ xi,

    and hence Cov(x) = (R.T @ R + lam * I)^{-1}. Each sample requires one
    multiplication by R and R.T and one solve with Q.

    This scheme is closely related to simulation of Gaussian Markov random
    fields using sparse precision matrices.


    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-11-16
    """
    rng = default_rng() if rng is None else rng
    m, n = R.shape

    if solver is None:
        solver = make_cg_precision_solver(R, lam=lam)

    samples = np.empty((n_samples, n), dtype=np.float64)

    for ix in range(n_samples):
        print('Sample:', str(ix), 'of', str(n_samples))
        xi = rng.standard_normal(size=m)
        b = R.T @ xi
        samples[ix, :] = solver(b)

    return samples


def sample_low_rank_from_precision_eigpairs(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    n_samples: int = 1,
    sigma2_residual: float = 0.0,
    rng: Optional[Generator] = None,
) -> np.ndarray:
    """Sample approximately from N(0, Q^{-1}) using low-rank eigendecomposition.

    Parameters
    ----------
    eigvals : ndarray, shape (k,)
        Eigenvalues of the precision matrix Q corresponding to the columns
        of ``eigvecs``. Typically these are the smallest eigenvalues,
        representing the directions of largest variance.
    eigvecs : ndarray, shape (n, k)
        Matrix of eigenvectors of Q. Columns are assumed orthonormal.
    n_samples : int, optional
        Number of independent samples to generate. The default is 1.
    sigma2_residual : float, optional
        Isotropic residual variance added in directions orthogonal to the
        span of ``eigvecs``. If positive, the effective covariance is
        C ≈ V Λ^{-1} V.T + sigma2_residual * I, where V and Λ are given by
        eigvecs and eigvals. The default is 0.0 (pure low-rank covariance).
    rng : numpy.random.Generator, optional
        Random number generator to use. If None, ``default_rng()`` is used.

    Returns
    -------
    samples : ndarray, shape (n_samples, n)
        Array of approximate Gaussian samples. Each row is one draw.

    Notes
    -----
    This function implements

        C_k = V Λ^{-1} V^T

    with eigenpairs (λ_i, v_i). A sample from N(0, C_k) is given by

        x = V Λ^{-1/2} z,  z ~ N(0, I_k).

    If ``sigma2_residual > 0``, an additional isotropic component is added,

        x = V Λ^{-1/2} z_k + sqrt(sigma2_residual) * z_perp,

    where z_perp ~ N(0, I_n). In high dimensions the second term can be
    expensive; often sigma2_residual is kept at zero or used only when n is
    moderate.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-11-16
    """
    rng = default_rng() if rng is None else rng

    eigvals = np.asarray(eigvals, dtype=np.float64)
    eigvecs = np.asarray(eigvecs, dtype=np.float64)

    if eigvals.ndim != 1:
        raise ValueError("eigvals must be a 1D array of eigenvalues.")
    if eigvecs.ndim != 2:
        raise ValueError("eigvecs must be a 2D array of eigenvectors.")
    if eigvecs.shape[1] != eigvals.shape[0]:
        raise ValueError(
            "eigvecs.shape[1] must equal eigvals.shape[0] (one eigenvalue per vector)."
        )

    n, k = eigvecs.shape
    samples = np.empty((n_samples, n), dtype=np.float64)

    inv_sqrt = 1.0 / np.sqrt(eigvals)

    for ix in range(n_samples):
        print('Sample:', str(ix), 'of', str(n_samples))
        z = rng.standard_normal(size=k)
        scaled = inv_sqrt * z
        x = eigvecs @ scaled

        if sigma2_residual > 0.0:
            z_perp = rng.standard_normal(size=n)
            x = x + np.sqrt(sigma2_residual) * z_perp

        samples[ix, :] = x

    return samples


def estimate_low_rank_eigpairs_from_precision(
    Q: "scipy.sparse.spmatrix | LinearOperator",
    k: int,
    which: str = "SM",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute k extremal eigenpairs of a symmetric precision matrix Q.

    Parameters
    ----------
    Q : sparse matrix or LinearOperator, shape (n, n)
        Symmetric positive definite precision matrix. Q can be provided as
        a SciPy sparse matrix or as a LinearOperator with an appropriate
        matvec implementation.
    k : int
        Number of eigenpairs to compute.
    which : {{'LM', 'SM'}}, optional
        Which eigenvalues to compute. 'LM' requests the largest magnitude
        eigenvalues, 'SM' the smallest magnitude ones. For low-rank covariance
        approximations of Q^{-1}, 'SM' is typically appropriate because the
        smallest eigenvalues correspond to the largest variances in the
        covariance. The default is 'SM'.

    Returns
    -------
    eigvals : ndarray, shape (k,)
        Eigenvalues of Q.
    eigvecs : ndarray, shape (n, k)
        Corresponding eigenvectors of Q (columns).

    Notes
    -----
    This is a thin wrapper around :func:`scipy.sparse.linalg.eigsh`. For very
    large problems, more specialised eigen-solvers or problem-specific
    techniques may be required. The resulting eigenpairs can be passed to
    :func:`sample_low_rank_from_precision_eigpairs` to construct a reduced-rank
    Gaussian sampler.

    Author: Volker Rath (DIAS)
    Created by ChatGPT (GPT-5 Thinking) on 2025-11-16
    """
    # SciPy's eigsh expects a matrix or LinearOperator; we simply forward it.
    eigvals, eigvecs = eigsh(Q, k=k, which=which)
    return eigvals, eigvecs
