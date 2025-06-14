
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
                 errors=[],
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
    
    @author:     charroyj
                 vrath   
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




def calc_covar_simple(x=np.array([]),
                y=np.array([]),
                covscovale=np.array([]),
                method=0, out=True):
    '''
    covalcovulate empiricoval covovariancove for Kalman gain

    covreated on Jul 6, 2022

    @author: vrath


    '''

    if (x.size == 0) and (y.size == 0):
        error('covalcov_encovovar: No data given! Exit.')

    X = x - np.mean(x, axis=0)
    if (y.size == 0):
        Y = X
    else:
        Y = y - np.mean(y, axis=0)

    [N_e, N_x] = np.shape(X)
    [N_e, N_y] = np.shape(Y)

    if method == 0:
        # print(N_e, N_x, N_y)
        # naive version, library versions probably faster)
        cov = np.zeros((N_x, N_y))
        for n in np.arange(N_e):
            # print('XT  ',X.T)
            # print('Y   ',Y)
            covn = X.T@Y
            # print(covn)
            cov = cov + covn

        cov = cov/(N_e-1)

    else:
        # numpy version
        for n in np.arange(N_e):
            X = np.stacovk((X, Y), axis=0)
            # cov = np.covov((X,Y))
            cov = np.covov((X))

    if out:
        print('Ensemble covovariancove is '+str(np.shape(cov)))

    return cov


def calc_covar_nice(x=np.array([]),
                y=np.array([]),
                fac=np.array([]),
                 out=True):
    '''
    Calculate empirical covariance for Kalman gain
    
    
    Method described in:
        
    Vishny, D., Morzfeld M., Gwirtz K., Bach, E., Dunbar, O.R.A. & Hodyss, D.
    High dimensional covariance estimation from a small number of samples
    Journal of Advances in Modeling Earth Systems, 16, 2024,
    doi:10.1029/2024MS004417
 

    Created on Jul 6, 2022

    @author: vrath
    
    Matlab version: 
    function [Cov_NICE,Corr_NICE,L_NICE] = NICE(X,Y,fac)
    Ne = size(X,2) ; 
    [CorrXY,~] = corr(X',Y');
    std_rho = (1-CorrXY.^2)/sqrt(Ne);
    sig_rho = sqrt(sum(sum(std_rho.^2)));
    
    expo2 = 2:2:8;
    for kk = 1:length(expo2)
        L = abs(CorrXY).^expo2(kk);
        Corr_NICE = L.*CorrXY;
        if norm(Corr_NICE - CorrXY,'fro') > fac*sig_rho
            expo2 = expo2(kk);
            break
        end
    end
    expo1 = expo2-2;
    rho_exp1 = CorrXY.^expo1;
    rho_exp2 = CorrXY.^expo2;
    
    al = 0.1:.1:1;
    for kk=1:length(al)
        L = (1-al(kk))*rho_exp1+al(kk)*rho_exp2;
        Corr_NICE = L.*CorrXY;
        if kk>1 && norm(Corr_NICE - CorrXY,'fro') > fac*sig_rho
            Corr_NICE = PrevCorr;
            break
        elseif norm(Corr_NICE - CorrXY,'fro') > fac*sig_rho
            break
        end
        PrevCorr = Corr_NICE;
        L_NICE = L;
    end
    Vy = diag(std(Y,0,2));
    Vx = diag(std(X,0,2));
    Cov_NICE = Vx*Corr_NICE*Vy;
    end
    '''
    
    nc = np.shape(x)[1]
    x = (x - np.mean(x, axis=0))/np.std(x,axis=0)
    y = (y - np.mean(y, axis=0))/np.std(y,axis=0)
    corr = (np.dot(y.T, x)/y.shape[0])[0]
    
    std_rho = (1.-np.power(corr,2))/np.sqrt(nc)
    sig_rho = np.sqrt(np.sum(np.sum(np.power(std_rho, 2))))
    
    
    expo2 = np.arange(2, 8, 2)
    for k in np.arange(len(expo2)):
        t = np.power(np.abs(corr),expo2(k))
        corr_nice = t*corr
        if np.norm(corr_nice - corr,'fro') > fac*sig_rho:
            expo2 = expo2(k)
            break
     
    expo1 = expo2-2
    rho_exp1 = np.power(corr, expo1)
    rho_exp2 = np.power(corr, expo2)


    a = np.arange(0.1, 1., 0.1)
    prevcorr=np.nan_like(corr)
    for k in np.arange(len(a)):
        t = (1.-a(k))*rho_exp1+a(k)*rho_exp2
        corr_nice = t*corr

        
        if k>0 and np.norm(corr_nice - corr,'fro') > fac*sig_rho:
            corr_nice = prevcorr
            break
        elif np.norm(corr_nice - corr,'fro') > fac*sig_rho:
            break
        
        prevcorr = corr_nice
        l_nice = t
        
        vy = np.diag(np.std(y,axis=1))
        vx = np.diag(np.std(x,axis=1));
        cov_nice = vx@corr_nice@vy;
        
    return cov_nice,corr_nice,l_nice



def msqrt_sparse(M=np.array([]), smallval=1.e-12):
    '''
    Calculate sparse Cholesky.

    Missing in scipy.

    Parameters
    ----------
    A : double
        Positive definite sparse matrix.

    Returns
    -------
    CholA: double
        Cholesky factor of A.

    VR Feb 2021

    '''
    n =M.shape[0]
    MM = M.copy() + np.identity(n)*smallval

    LU = scipy.sparse.linalg.splu(
        MM, diag_pivot_thresh=0)  # sparse LU decomposition

    # check the matrix A is positive definite.
    if (LU.perm_r == np.arange(n)).all() and (LU.U.diagonal() > 0).all():
        SqrtM = LU.L.dot(scipy.sparse.diags(LU.U.diagonal() ** 0.5))

    else:
        error('The matrix is not positive definite')

    return SqrtM


def msqrt(M=np.array([]), method='cho', smallval=1.e-12):
    '''
    Computes a matrix square-root (Choleky, or eig).

    Parameter:
    M: M is a positive Hermitian (or positive definite) matrix.

    Return:
    SqrtM, Mevals, Mevecs:
    Here, SqrtM is a matrix such that SqrtM * SqrtM.T = M.
    The vector Mevals contains the eigenvectors of M,
    and the matrix Mevecs the corresponding eigenvectors.

    Also Calculate sparse Cholesky, missing in scipy.

    Parameters
    ----------
    A : double
        Positive definite sparse matrix.
    smallval: double
        small value to guarantee positive definiteness in
        the case of numueerical noise.
    method: str
        eigenvalue or cholesky in case of dense input matrices

    Returns
    -------
    CholM: double
        Cholesky factor of A.

    Last change: VR Mar 2024


    '''
    n = np.shape(M)[0]
    MM = M.copy() + np.identity(n)*smallval

    if 'eig' in method.lower():
        # compute eigenvalues and eigenvectors
        Mevals, Mevecs = scipy.linalg.eigh(MM)
        Mevals = Mevals.clip(min=0.0)
        SqrtM = Mevecs * np.sqrt(Mevals)
        return SqrtM, Mevals, Mevecs

    if 'cho' in method.lower():
        SqrtM = scipy.linalg.cholesky(MM)


    return SqrtM

def isspd(A):

    n = A.shape[0]

    AAT = A@A.T
    if np.allclose(AAT, np.identity(n), rtol = 1.e-8, atol=1.e-8):
        print('A is symmetric.')
    else:
        print('A is NOT symmetric.')

    spd = np.all(np.linalg.eigvals(A) > 1.e-12)


    return spd

def rsvd(A, rank=300,
         n_oversamples=300,
         n_subspace_iters=None,
         return_range=False):
    '''
    =============================================================================
    Randomized SVD. See Halko, Martinsson, Tropp's 2011 SIAM paper:

    'Finding structure with randomness: Probabilistic algorithms for constructing
    approximate matrix decompositions'
    Author: Gregory Gundersen, Princeton, Jan 2019
    =============================================================================
    Randomized SVD (p. 227 of Halko et al).

    :param A:                (m x n) matrix.
    :param rank:             Desired rank approximation.
    :param n_oversamples:    Oversampling parameter for Gaussian random samples.
    :param n_subspace_iters: Number of power iterations.
    :param return_range:     If `True`, return basis for approximate range of A.
    :return:                 U, S, and Vt as in truncated SVD.
    '''
    if n_oversamples is None:
        # This is the default used in the paper.
        n_samples = 2 * rank
    else:
        n_samples = rank + n_oversamples

    # Stage A.
    # print(' stage A')
    Q = find_range(A, n_samples, n_subspace_iters)

    # Stage B.
    # print(' stage B')
    B = Q.T @ A
    # print(np.shape(B))
    # print(' stage B before linalg')
    U_tilde, S, Vt = np.linalg.svd(B)
    # print(' stage B after linalg')
    U = Q @ U_tilde

    # Truncate.
    U, S, Vt = U[:, :rank], S[:rank], Vt[:rank, :]

    # This is useful for computing the actual error of our approximation.
    if return_range:
        return U, S, Vt, Q
    return U, S, Vt


# ------------------------------------------------------------------------------


def find_range(A, n_samples, n_subspace_iters=None):
    '''
    
    Algorithm 4.1: Randomized range finder (p. 240 of Halko et al).

    Given a matrix A and a number of samples, computes an orthonormal matrix
    that approximates the range of A.

    :param A:                (m x n) matrix.
    :param n_samples:        Number of Gaussian random samples.
    :param n_subspace_iters: Number of subspace iterations.
    :return:                 Orthonormal basis for approximate range of A.
    '''
    # print('here we are in range-finder')
    rng = np.random.default_rng()

    m, n = A.shape
    # print(A.shape)
    O = rng.normal(size=(n, n_samples))
    # print(O.shape)
    Y = A @ O

    if n_subspace_iters:
        return subspace_iter(A, Y, n_subspace_iters)
    else:
        return ortho_basis(Y)


# ------------------------------------------------------------------------------


def subspace_iter(A, Y0, n_iters):
    '''
    Algorithm 4.4: Randomized subspace iteration (p. 244 of Halko et al).

    Uses a numerically stable subspace iteration algorithm to down-weight
    smaller singular values.

    :param A:       (m x n) matrix.
    :param Y0:      Initial approximate range of A.
    :param n_iters: Number of subspace iterations.
    :return:        Orthonormalized approximate range of A after power
                    iterations.
    '''
    # print('herere we are in subspace-iter')
    Q = ortho_basis(Y0)
    for _ in range(n_iters):
        Z = ortho_basis(A.T @ Q)
        Q = ortho_basis(A @ Z)
    return Q


# ------------------------------------------------------------------------------


def ortho_basis(M):
    '''
    Computes an orthonormal basis for a matrix.

    :param M: (m x n) matrix.
    :return:  An orthonormal basis for M.
    '''
    # print('herere we are in ortho')
    Q, _ = np.linalg.qr(M)
    return Q







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
