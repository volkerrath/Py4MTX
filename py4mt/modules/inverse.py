import os
import sys
import inspect

import numpy as np
import scipy as scp
import scipy.linalg as scl
import scipy.sparse as scs
import scipy.ndimage as sci

import sklearn.cluster as scl


def soft_thresh(x, lam):
    '''
    function S_lambda = soft_thresh(x,lambda)
        S_lambda = sign(x) .*  max(abs(x) - lambda,0);
    end
    '''
    s_lam = np.sign(x) *  np.amax(np.abs(x) - lam, 0.)

    return s_lam

def splitbreg(J, y, lam, D, c=0., tol=1.e-5, maxiter=10):
    '''
    Solves a constrained optimization problem using
    the Split Bregman method with Total Variation (TV) regularization.

    Inputs:
        J    - Forward operator (Jacobian)
        y    - Observed data vector
        lam  - Regularization parameter
        D    - Difference operator
        c    - Set to zero when using for generate blocky models

    Output:
        x    - Recovered solution

    '''

    (nd,nm) = np.shape(J)

    mu = 2.*lam

    b = np.zeros((nd, 1))
    d = np.zeros((nd, 1))

    A = np.array([J], [np.sqrt(mu)*D])
    xold = np.nan*np.ones((nm,1))
    for kk in np.arange(maxiter):
        r = np.array[[y],[np.sqrt(mu)*(d-b)]]
        x = scl.solve(A,r, assume_a='general')
        s = soft_thresh(c+D@x+d, lam/mu)
        d = s - c
        b = b + D@x -s
        
        if ((kk>0) and (np.norm(xold-x)/np.norm(x)<tol)):
            break
        
        xold = x

    return x


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
        sys.exit('covalcov_encovovar: No data given! Exit.')

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



def msqrt_sparse(M=None, method='chol', smallval=None, nthreads = 16):
    '''
    Computes a matrix square-root (cholesky, lu, or eig).

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
        the case of numerical noise.
    method: str
        eigenvalue, splu or cholesky in case of dense input matrices

    Returns
    -------
    sqrtM: double
        Cholesky factor of A.

    Last change: VR Mar 2024


    '''
    from scipy.sparse import csr_array, csc_array, coo_array, eye_array, diags_array, issparse
    from threadpoolctl import threadpool_limits

    
    n, _ =M.shape


    if smallval is not None:
        M = M + np.identity(n)*smallval


    if 'eigs' in method.lower():
        from scipy.linalg import eigsh
        # compute eigenvalues and eigenvectors
        with threadpool_limits(limits=nthreads):
            mevals, mevecs = eigsh(M)
        mevals = mevals.clip(min=0.0)
        sqrtM = mevecs * np.sqrt(mevals)
        return sqrtM

    elif 'chol' in method.lower():
        from scipy.linalg import cholesky
        with threadpool_limits(limits=nthreads):
            sqrtM = cholesky(M.toarray())
        return sqrtM
        
    elif 'splu' in method.lower():
        from scipy.sparse.linalg import splu, diags
        with threadpool_limits(limits=16):
            LU = scs.linalg.splu(M, diag_pivot_thresh=0)  # sparse LU decomposition
    
        # check the matrix A is positive definite.
        if (LU.perm_r == np.arange(n)).all() and (LU.U.diagonal() > 0).all():
            sqrtM = LU.L.dot(diags(LU.U.diagonal() ** 0.5))
    
        else:
            sys.exit('The matrix is not positive definite')
            
        
        return sqrtM

    else:
         sys.exit()
    
def isspd(A):
    from scipy.sparse import csr_array, csc_array, coo_array, eye_array, diags_array, issparse

    n = A.shape[0]

    if issparse(A):
        from scipy.sparse.linalg import splu
        try:
            # Convert to Compressed Sparse Row format
            spm = csr_array(A)
            # Attempt LU decomposition (Cholesky not directly available for sparse)
            splu(spm)
            return True
        except RuntimeError:
            return False
    else:
        from scipy.linalg import cholesky
        try:
            cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False



    #AAT = A@A.T
    #if np.allclose(AAT, np.identity(n), rtol = 1.e-8, atol=1.e-8):
        #print('A is symmetric.')
    #else:
        #print('A is NOT symmetric.')

    #spd = np.all(np.linalg.eigvals(A) > 1.e-12)


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


