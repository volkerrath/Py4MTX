#!/usr/bin/env python3
"""
jacproc.py

Utilities for Jacobian-based post-processing: sensitivities, sparsification, streaming statistics, and low-rank approximations.

Most workflows assume the Jacobian is already error-scaled, e.g. J_scaled = C^{-1/2} J.

Dependencies
------------
- numpy
- scipy (sparse)
- numba (optional)

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-12-21
"""
import sys
import os
import numpy as np
import scipy.sparse as scs
import numpy.linalg as npl

from numba import jit

def calc_gradient(Jac=None, resid=None, small = 1.e-30, out = False):
    """
    Compute a gradient vector from a scaled Jacobian and residuals.
    
    Parameters
    ----------
    Jac : array_like
        Scaled Jacobian matrix (e.g. C^{-1/2} J).
    resid : array_like | None
        Residual vector; if None, gradient is computed as zeros (legacy behaviour).
    small : float
        Lower bound applied to near-zero entries.
    out : bool
        If True, print diagnostic information.
    
    Returns
    -------
    G : ndarray
        Gradient vector of shape (npar,).
    
    Notes
    -----
    The historical implementation in this project assumes the Jacobian is already
    error-scaled.
    """

    if Jac is None:
        sys.exit('calc_sensitivity: Jacobian size is 0! Exit.')

    [ndat, npar] = np.shape(Jac)

    G = np.zeros((1, npar))
    if out:

        maxval = np.amax(G)
        minval = np.amin(G)
        print('calc_sensitivity:',minval, maxval)

    # print('calc: ', np.any(S==0))
    G[np.where(np.abs(G)<small)]=small
    # print('calc: ', np.any(S==0))
    # S=S.A1
    G = np.asarray(G).ravel()


    return G

def calc_sensitivity(Jac=np.array([]),
                     Type = 'euclidean', UseSigma = False, Small = 1.e-30, OutInfo = False):
    """
    Compute parameter sensitivities from an (optionally sparse) Jacobian.
    
    Parameters
    ----------
    Jac : array_like
        Scaled Jacobian (dense array or scipy sparse).
    Type : str
        Sensitivity measure: 'raw', 'cov' (coverage), 'euc', or 'cum'.
    UseSigma : bool
        If True, compute sensitivities w.r.t. sigma parameters (sign-flip).
    Small : float
        Lower bound applied to sensitivities.
    OutInfo : bool
        If True, print diagnostics.
    
    Returns
    -------
    S : ndarray
        Sensitivity vector of shape (npar,).
    
    Notes
    -----
    Implements several sensitivity definitions commonly used in EM/DC literature.
    """

    if np.size(Jac)==0:
        sys.exit('calc_sensitivity: Jacobian size is 0! Exit.')

    if UseSigma:
        Jac = -Jac


    if 'raw' in  Type.lower():
        S = Jac.sum(axis=0)
        if OutInfo:
            print('raw:', S)
        # else:
        #     print('raw sensitivities')
        # smax = Jac.max(axis = 0)
        # smin = Jac.max(axis = 0)

    elif 'cov' in Type.lower():
        S = np.abs(Jac)
        S = np.sum(S, axis=0)
        if OutInfo:
            print('cov:', S)
        # else:
        #     print('coverage')

    elif 'euc' in Type.lower():
        S = Jac.power(2).sum(axis=0)
        #S = S.sum(axis=0)
        S = np.sqrt(S)
        if OutInfo:
            print('euc:', S)
        # else:
        #     print('euclidean (default)')

    elif 'cum' in Type.lower():
        S = np.abs(Jac)
        S = S.sum(axis=0)
        # print(np.shape(S))
        # S = np.sum(Jac,axis=0)

        S = np.append(0.+1.e-10, np.cumsum(S[-1:0:-1]))
        S = np.flipud(S)
        if OutInfo:
           print('cumulative:', S)
        # else:
        #    print('cumulative sensitivity')

    else:
        print('calc_sensitivity: Type '
              +Type.lower()+' not implemented! Default assumed.')
        S = Jac.power(2).sum(axis=0)
        S = S.sum(axis=0)
        if OutInfo:
            print('euc (default):', S)
        # else:
        #     print('euclidean (default)')

        # S = S.reshape[-1,1]


    if OutInfo:
        maxval = np.amax(S)
        minval = np.amin(S)
        print('calc_sensitivity:',minval, maxval)

    # print('calc: ', np.any(S==0))
    S[np.where(np.abs(S)<Small)]=Small
    # print('calc: ', np.any(S==0))
    # S=S.A1
    S = np.asarray(S).ravel()


    return S


def transform_sensitivity(S=np.array([]), Siz=np.array([]),
                          Transform='max',
                          asinhpar=[0.], Maxval=None, Small= 1.e-30, OutInfo=False):
    """
    Transform/normalise a sensitivity vector.
    
    Parameters
    ----------
    S : array_like
        Sensitivity values.
    Siz : array_like
        Optional size/volume weights used for 'siz' transform.
    Transform : str
        Space-separated list of transforms: 'max', 'siz', 'log', 'asinh', etc.
    asinhpar : list[float]
        Parameters for asinh transform; either [scale] or [method].
    Maxval : float | None
        Fixed maximum used for 'max' scaling.
    Small : float
        Lower bound applied after transforming.
    OutInfo : bool
        If True, print diagnostics.
    
    Returns
    -------
    S_out : ndarray
        Transformed sensitivities.
    scaleval : float
        Scale value used by the 'max' transform (or 1.0).
    
    Notes
    -----
    Transforms are applied in sequence. Use with care when negatives are present.
    """

    if np.size(S)==0:
        sys.exit('transform_sensitivity: Sensitivity size is 0! Exit.')

    ns = np.shape(S)
    print('transform_sensitivity: Shape = ', ns)
    print('trans_sensitivity:',np.amin(S), np.amax(S))


    scaleval = 1.

    transform = Transform.split(' ')

    for item in transform:


        # if 'sqr' in item.lower():
        #     S = np.sqrt(S)
        #     # print('S0s', np.shape(S))
        #     print('trans_sensitivity sqr:',np.amin(S), np.amax(S))

        if 'log' in item.lower():
            S = np.log10(S)
            print('trans_sensitivity log:',np.amin(S), np.amax(S))

        if 'asinh' in item.lower():
            maxval = np.amax(S)
            minval = np.amin(S)
            if maxval>0 and minval>0:
                print('transform_sensitivity: No negatives, switched to log transform!')
                S = np.log10(S)
            else:
                if len(asinhpar)==1:
                    scale = asinhpar[0]
                else:
                    scale = get_scale(S, method=asinhpar[0])

                    S = np.arcsinh(S/scale)
                    
        #if 'vol' in item.lower():
             #print('transformed_sensitivity: Transformed by volumes/layer thickness.')
             #if np.size(Siz)==0:
                 #sys.exit('transform_sensitivity: no volumes given! Exit.')

             #else:
                 #maxval = np.amax(S)
                 #minval = np.amin(S)
                 #print('before volume normalization:',minval, maxval)
                 #print('volume:', np.amax(Siz),np.amax(Siz) )
                 #if 'sqr'  in item.lower():
                     #Siz = np.cbrt(Siz)
                 #S = S/Siz.ravel()
                 #maxval = np.amax(S)
                 #minval = np.amin(S)
                 #print('after size normalization:',minval, maxval)

        if 'siz' in item.lower():
             print(np.shape(S), np.shape(Siz))
             print('transformed_sensitivity: Transformed by other size measure.')
             if np.size(Siz)==0:
                 sys.exit('transform_sensitivity: no volumes given! Exit.')

             else:
                 maxval = np.amax(S)
                 minval = np.amin(S)
                 print('before size normalization:',minval, maxval)
                 print('size:', np.amax(Siz),np.amax(Siz) )
                 S = S/Siz.ravel('F')
                 maxval = np.amax(S)
                 minval = np.amin(S)
                 print('after size normalization:',minval, maxval)

        if 'max' in item.lower():
             print('trans_sensitivity: Transformed by maximum value.')
             if Maxval is None:
                 maxval = np.amax(np.abs(S))
             else:
                 maxval = Maxval
             print('maximum value: ', maxval)
             S = S/maxval
             print('trans_sensitivity max:',np.amin(S), np.amax(S))
             # print('S0m', np.shape(S))
             scaleval = maxval


    if OutInfo:
        print('trans_sensitivity:',np.amin(S), np.amax(S))

    S[np.where(np.abs(S)<Small)]=Small


    return S, scaleval

def get_scale(d=np.array([]), f=0.1, method = 'other', OutInfo = False):
    """
    Compute a scale value for the arcsinh transform.
    
    Parameters
    ----------
    d : array_like
        Input data values.
    f : float
        Weight factor controlling the scale.
    method : str
        Scale selection method, e.g. 's2007' for Scholl (2007) style.
    OutInfo : bool
        If True, print diagnostics.
    
    Returns
    -------
    scale : float
        Scale value.
    
    Notes
    -----
    Used by transform_sensitivity when 'asinh' is selected.
    """

    if np.size(d)==0:
        sys.exit('get_S: No data given! Exit.')

    if 's2007' in method.lower():
        scale = f * np.nanmax(np.abs(d))

    else:
        dmax = np.nanmax(np.abs(d))
        dmin = np.nanmin(np.abs(d))
        denom =f *(np.log(dmax)-np.log(dmin))
        scale = np.abs(dmax/denom)

    if OutInfo:
        print('Scale value S is '+str(scale)+', method '+method)

    return scale

def sparsmat_to_array(mat=None):
    """
    Convert a (possibly sparse) matrix to a 1-D NumPy array.
    
    Parameters
    ----------
    mat : array_like
        Matrix-like input (dense or sparse).
    
    Returns
    -------
    arr : ndarray
        Flattened array view/copy.
    
    Notes
    -----
    This is a thin helper used in legacy code paths.
    """
    arr = np.array([])

    # data = mat.A1
    arr= np.asarray(mat).ravel()

    return arr


def update_avg(k = None, m_k=None, m_a=None, m_v=None):
    """
    Online update of mean and (unnormalised) variance for a data stream.
    
    Parameters
    ----------
    k : int
        Sample index (1-based). If k < 0, returns a normalised variance estimate.
    m_k : array_like
        Current sample.
    m_a : array_like
        Previous running mean.
    m_v : array_like
        Previous running M2 accumulator.
    
    Returns
    -------
    m_avg : ndarray
        Updated running mean.
    m_var : ndarray
        Updated running M2 accumulator (or variance if k < 0).
    
    Notes
    -----
    Implements a stable one-pass update (Knuth / Welford style).
    """
    if k == 1:
        m_avg = m_k
        m_var = np.zeros_like(m_avg)

    md = m_k - m_a
    m_avg = m_a + md/np.abs(k)
    m_var = m_v + md*(m_k - m_avg)

    if k < 0:
        m_var = m_var/(np.abs(k-1))

    return m_avg, m_var

# def update_med(k = None, model_n=None, model_a=None, model_v=None):
#     '''
#     Estimate the quantiles from data stream.

#     T-digest

#     VR  Mar , 2021
#     '''

#     return m_med, m_q1, m_q2

def rsvd(A, rank=300, n_oversamples=None, n_subspace_iters=None, return_range=False):
    """
    Randomized SVD (truncated) for large matrices.
    
    Parameters
    ----------
    A : array_like
        Input matrix (m x n).
    rank : int
        Target rank for the approximation.
    n_oversamples : int | None
        Oversampling parameter; defaults to 2*rank if None.
    n_subspace_iters : int | None
        Number of power iterations (subspace iterations).
    return_range : bool
        If True, also return the approximate range basis Q.
    
    Returns
    -------
    U : ndarray
        Left singular vectors (m x rank).
    S : ndarray
        Singular values (rank,).
    Vt : ndarray
        Right singular vectors transposed (rank x n).
    
    Notes
    -----
    Implements Halko, Martinsson & Tropp (2011) algorithms. Useful for low-rank
    Jacobian approximations.
    """
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
    """
    Compute an approximate orthonormal basis for the range of A.
    
    Parameters
    ----------
    A : array_like
        Input matrix.
    n_samples : int
        Number of random samples.
    n_subspace_iters : int | None
        Optional number of subspace iterations.
    
    Returns
    -------
    Q : ndarray
        Orthonormal basis for the approximate range.
    
    Notes
    -----
    Algorithm 4.1 from Halko et al. (2011).
    """
    # print('here we are in range-finder')
    m, n = A.shape
    O = np.random.default_rng().normal(0., 1., (n, n_samples))
    Y = A @ O

    if n_subspace_iters:
        return subspace_iter(A, Y, n_subspace_iters)
    else:
        return ortho_basis(Y)


# ------------------------------------------------------------------------------


def subspace_iter(A, Y0, n_iters):
    """
    Perform randomized subspace iteration to improve range approximation.
    
    Parameters
    ----------
    A : array_like
        Input matrix.
    Y0 : array_like
        Initial range sample matrix.
    n_iters : int
        Number of power iterations.
    
    Returns
    -------
    Q : ndarray
        Improved orthonormal basis.
    
    Notes
    -----
    Algorithm 4.4 from Halko et al. (2011).
    """
    # print('herere we are in subspace-iter')
    Q = ortho_basis(Y0)
    for _ in range(n_iters):
        Z = ortho_basis(A.T @ Q)
        Q = ortho_basis(A @ Z)
    return Q


# ------------------------------------------------------------------------------


def ortho_basis(M):
    """
    Compute an orthonormal basis for the columns of a matrix via QR.
    
    Parameters
    ----------
    M : array_like
        Input matrix.
    
    Returns
    -------
    Q : ndarray
        Orthonormal basis matrix.
    
    Notes
    -----
    Thin wrapper around numpy.linalg.qr.
    """
    # print('herere we are in ortho')
    Q, _ = np.linalg.qr(M)
    return Q


def sparsify_jac(Jac=None,
                 sparse_thresh=1.0e-6, normalized=False, scalval = 1.,
                 method=None, out=True):
    """
    Threshold sparsification of a dense Jacobian matrix.
    
    Parameters
    ----------
    Jac : ndarray
        Dense Jacobian.
    sparse_thresh : float
        Relative threshold for zeroing entries.
    normalized : bool
        If True, scale Jacobian by the chosen scale before thresholding.
    scalval : float
        Scale value; if negative, uses max(|J|).
    method : str | None
        Reserved for alternative sparsification strategies.
    out : bool
        If True, print diagnostics.
    
    Returns
    -------
    Js : scipy.sparse.csr_matrix
        Sparsified Jacobian in CSR format.
    Scaleval : float
        Scale value used for normalisation.
    
    Notes
    -----
    This is typically used to reduce memory for large Jacobians prior to iterative
    solvers.
    """
    shj = np.shape(Jac)
    if out:
        nel = shj[0] * shj[1]
        print(
            'sparsify_jac: dimension of original J is %i x %i = %i elements'
            % (shj[0], shj[1], nel)
        )


    Jf = Jac.copy()
    # print(np.shape(Jf))

    if scalval <0.:
        Scaleval = np.amax(np.abs(Jf))
        print('sparsify_jac: scaleval is %g (max Jacobian)' % (Scaleval))
    else:
        Scaleval = abs(scalval)
        print('sparsify_jac: scaleval is  %g' % (Scaleval))

    if normalized:
        print('sparsify_jac: output J is scaled by %g' % (Scaleval))
        f = 1.0 / Scaleval
        Jf = normalize_jac(Jac=Jf, fn=f)

    Jf[np.abs(Jf)/Scaleval < sparse_thresh] = 0.0

    # print(np.shape(Jf))

    Js = scs.csr_matrix(Jf)


    if out:
        ns = Js.count_nonzero()
        print('sparsify_jac:'
                +' output J is sparse, and has %i nonzeros, %f percent'
                % (ns, 100.0 * ns / nel))
        test = np.random.default_rng().normal(size=np.shape(Jac)[1])
        normx = npl.norm(Jf@test)
        normo = npl.norm(Jf@test-Js@test)


        normd = npl.norm((Jac-Jf), ord='fro')
        normf = npl.norm(Jac, ord='fro')
        # print(norma)
        # print(normf)
        print(' Sparsified J explains '
              +str(round(100.-100.*normo/normx,2))+'% of full J (Spectral norm)')
        print(' Sparsified J explains '
              +str(round(100.-100.*normd/normf,2))+'% of full J (Frobenius norm)')
        # print('****', nel, ns, 100.0 * ns / nel, round(100.-100.*normd/normf,3) )



    return Js, Scaleval


def normalize_jac(Jac=None, fn=None, out=True):
    """
    Apply error scaling/normalisation to a Jacobian matrix.
    
    Parameters
    ----------
    Jac : array_like
        Jacobian matrix.
    fn : array_like
        Scaling factors. If scalar-like, divides whole Jacobian; if vector, left-
        multiplies by diag(1/fn).
    out : bool
        If True, print diagnostics.
    
    Returns
    -------
    Jac_out : array_like
        Scaled Jacobian.
    
    Notes
    -----
    Used as a building block for sparsify_jac and downstream sensitivity measures.
    """
    shj = np.shape(Jac)
    shf = np.shape(fn)
    # print('fn = ')
    # print(fn)
    if shf[0] == 1:
        f = 1.0 / fn[0]
        Jac = f * Jac
    else:
        fd = 1./fn[:]
        fd = fd.flatten()
        print(fd.shape)
        erri = scs.diags([fd], [0], format='csr')
        Jac = erri @ Jac
        #erri = np.reshape(1.0 / fn, (shj[0], 1))
        #Jac = erri[:] * Jac

    return Jac

def set_padmask(rho=None, pad=[0, 0 , 0, 0, 0, 0], blank= np.nan, flat=True, out=True):
    """
    Create a boolean/float mask for padded model regions.
    
    Parameters
    ----------
    rho : ndarray
        3-D model array.
    pad : list[int]
        Padding widths [x0,x1,y0,y1,z0,z1].
    blank : float
        Fill value outside the active region.
    flat : bool
        If True, return a flattened mask in Fortran order.
    out : bool
        If True, print diagnostics.
    
    Returns
    -------
    mask : ndarray
        Mask array (same shape as rho, or flattened).
    
    Notes
    -----
    Commonly used to exclude padding cells in Jacobian/sensitivity operations.
    """
    shr = np.shape(rho)
    # jm = np.full(shr, np.nan)
    jm = np.full(shr, blank)
    print(np.shape(jm))

    jm[pad[0]:-pad[1], pad[2]:-pad[3], pad[4]:-pad[5]] = 1.
    # print(pad[0], -1-pad[1])
    # jt =jm[0+pa-1-pad[1]-1-pad[1]d[0]:-1-pad[1], 0+pad[2]:-1-pad[3], 0+pad[4]:-1-pad[5]]
    # print(np.shape(jt))
    mask = jm
    if flat:
        # mask = jm.flatten()
        mask = jm.flatten(order='F')

    return mask


def set_airmask(rho=None, aircells=np.array([]), blank= 1.e-30, flat=False, out=True):
    """
    Create a mask that marks air cells in a model.
    
    Parameters
    ----------
    rho : ndarray
        3-D model array.
    aircells : ndarray
        Indices of air cells (format depends on upstream workflow).
    blank : float
        Fill value for air cells.
    flat : bool
        If True, return flattened mask.
    out : bool
        If True, print diagnostics.
    
    Returns
    -------
    mask : ndarray
        Mask array.
    
    Notes
    -----
    Air-cell masking is required when using log-resistivity models with very high
    air resistivity.
    """
    shr = np.shape(rho)
    # jm = np.full(shr, np.nan)
    jm = np.full(shr, 1.)
    print(np.shape(jm), shr)

    jm[aircells] = blank
    mask = jm
    if flat:
        # mask = jm.flatten()
        mask = jm.flatten(order='F')

    return mask


def project_nullspace(U=np.array([]), m_test=np.array([])):
    """
    project_nullspace.
    
    Parameters
    ----------
    U : object
        Parameter U.
    m_test : object
        Parameter m_test.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    Calculates nullspace projection of a vector
    
    Parameters ---------- U : numpy array, float      npar*npar matrix from SAVD oj
    Jacobian. m_test : numpy array, float      npar*vector to be projected.
    
    Returns ------- m: numpy array, float     projected model
    """
    if np.size(U) == 0:
        sys.exit('project_nullspace: V not defined! Exit.')

    m_proj = m_test - U@(U.T@m_test)

    return m_proj

def project_models(m=None, U=None, tst_sample= None, nsamp=1, small=1.0e-14, out=True):
    """
    project_models.
    
    Parameters
    ----------
    m : object
        Parameter m.
    U : object
        Parameter U.
    tst_sample : object
        Parameter tst_sample.
    nsamp : object
        Parameter nsamp.
    small : object
        Parameter small.
    out : object
        Parameter out.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    Project to Nullspace.
    
    (see Munoz & Rath, 2006) author: vrath last changed: Feb 29, 2024
    """
    if m.ndim(m)>1:
        m = m.flatten(order='F')

    if tst_sample  is None:
        print('project_model: '+str(nsamp)+' sample models will be generated!')
        if nsamp==0:
           sys.exit('project_model: No number of samples given! Exit.')
        tst_sample = m + np.random.default_rng().normal(0., 1., (nsamp, len(m)))

    else:
        nsamp = np.shape(tst_sample)[0]

    nss_sample = np.zeros(nsamp, len(m))

    for isamp in np.arange(nsamp):
        b = U.T@tst_sample[isamp,:]
        nss_sample[isamp, :] = m - U@b

    return nss_sample

def sample_pcovar(cpsqrti=None, m=None, tst_sample = None,
                  nsamp = 1, small=1.0e-14, out=True):
    """
    sample_pcovar.
    
    Parameters
    ----------
    cpsqrti : object
        Parameter cpsqrti.
    m : object
        Parameter m.
    tst_sample : object
        Parameter tst_sample.
    nsamp : object
        Parameter nsamp.
    small : object
        Parameter small.
    out : object
        Parameter out.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    Sample Posterior Covariance.
    
    Algorithm given by  Osypov (2013)
    
    Parameters ----------
    
    Returns ------- spc_sanple
    
    References:
    
    Osypov K, Yang Y, Fournier A, Ivanova N, Bachrach R,     Can EY, You Y, Nichols
    D, Woodward M (2013)     Model-uncertainty quantification in seismic
    tomography: method and applications     Geophysical Prospecting, 61, pp.
    1114–1134, 2013, doi: 10.1111/1365-2478.12058.
    """
    sys.exit('sample_pcovar: Not yet fully implemented! Exit.')

    if (cpsqrti is None) or  (m is None):
        sys.exit('sample_pcovar: No covarince or ref model given! Exit.')



    if tst_sample is None:
        print('sample_pcovar: '+str(nsamp)+' sample models will be generated!')
        if nsamp==0:
           sys.exit('sample_pcovar: No number of samples given! Exit.')
        tst_sample = np.random.default_rng().normal(0., 1., (nsamp, len(m)))

    else:
        nsamp = np.shape(tst_sample)[0]


    spc_sample = np.zeros(nsamp, len(m))

    for isamp in np.arange(nsamp):
        spc_sample[isamp,:] = m + cpsqrti@tst_sample[isamp,:]

    return spc_sample


def mult_by_cmsqr(m_like_in=None, smooth=[None, None, None], small=1.0e-14, out=True):
    """
    mult_by_cmsqr.
    
    Parameters
    ----------
    m_like_in : object
        Parameter m_like_in.
    smooth : object
        Parameter smooth.
    small : object
        Parameter small.
    out : object
        Parameter out.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    Multyiply by sqrt of paramter prior covariance (aka 'smoothing')
    
    baed on the ModEM fortran code
    
    Parameters ---------- m_like_in : TYPE, optional     DESCRIPTION. The default
    is None. smooth : TYPE, optional     DESCRIPTION. The default is None. out :
    TYPE, optional     DESCRIPTION. The default is True.
    
    Returns ------- None.
    
    =============================================================================
    
    subroutine RecursiveAR(w,v,n)
    
    ! Implements the recursive autoregression algorithm for a 3D real array.     !
    In our case, the assumed-shape array would be e.g. conductivity     ! in each
    cell of the Nx x Ny x NzEarth grid.
    
    real (kind=prec), intent(in)     :: w(:,:,:)     real (kind=prec), intent(out)
    :: v(:,:,:)     integer, intent(in)                      :: n     integer
    :: Nx, Ny, NzEarth, i, j, k, iSmooth
    
    Nx      = size(w,1)     Ny      = size(w,2)     NzEarth = size(w,3)
    
    if (maxval(abs(shape(w) - shape(v)))>0) then             call errStop('The
    input arrays should be of the same shapes in RecursiveAR')     end if
    
    v = w
    
    do iSmooth = 1,n
    
    ! smooth in the X-direction (Sx)         do k = 1,NzEarth             do j =
    1,Ny                     !v(1,j,k) = v(1,j,k)                     do i = 2,Nx
    v(i,j,k) = SmoothX(i-1,j,k) * v(i-1,j,k) + v(i,j,k)                     end do
    end do         end do
    
    ! smooth in the Y-direction (Sy)         do k = 1,NzEarth             do i =
    1,Nx                     ! v(i,1,k) = v(i,1,k)                     do j = 2,Ny
    v(i,j,k) = SmoothY(i,j-1,k) * v(i,j-1,k) + v(i,j,k)                     end do
    end do         end do
    
    ! smooth in the Z-direction (Sz)         do j = 1,Ny             do i = 1,Nx
    ! v(i,j,1) = v(i,j,1)                     do k = 2,NzEarth
    v(i,j,k) = SmoothZ(i,j,k-1) * v(i,j,k-1) + v(i,j,k)                     end do
    end do         end do !             ! smooth in the Z-direction (Sz^T)
    do j = Ny,1,-1             do i = Nx,1,-1                     ! v(i,j,NzEarth)
    = v(i,j,NzEarth)                     do k = NzEarth,2,-1
    v(i,j,k-1) = v(i,j,k-1) + SmoothZ(i,j,k-1) * v(i,j,k)                     end
    do             end do         end do
    
    ! smooth in the Y-direction (Sy^T)         do k = NzEarth,1,-1             do i
    = Nx,1,-1                     ! v(i,Ny,k) = v(i,Ny,k)                     do j
    = Ny,2,-1                                     v(i,j-1,k) = v(i,j-1,k) +
    SmoothY(i,j-1,k) * v(i,j,k)                     end do             end do
    end do
    
    ! smooth in the X-direction (Sx^T)         do k = NzEarth,1,-1             do j
    = Ny,1,-1                     ! v(Nx,j,k) = v(Nx,j,k)                     do i
    = Nx,2,-1                                     v(i-1,j,k) = v(i-1,j,k) +
    SmoothX(i-1,j,k) * v(i,j,k)                     end do             end do
    end do
    
    end do
    
    ! apply the scaling operator C     do k = 1,NzEarth             do j = 1,Ny
    do i = 1,Nx                             v(i,j,k) = (Scaling(i,j,k)**n) *
    v(i,j,k)                     end do             end do     end do
    
    end subroutine RecursiveAR
    =============================================================================
    """
    # nsmooth = 1


    sys.exit('mult_by_cmsq: Not yet implemented. Exit')

    tmp = m_like_in.copy()

    nx, ny,nz = np.shape(m_like_in)

    sm_x, sm_y, sm_z = smooth

    '''
    		! smooth in the Z-direction (Sz)
     	    for jj in  np.arange(0,ny)
    	    	do i = 1,Nx
     	    		! v(i,j,1) = v(i,j,1)
     	    		do k = 2,NzEarth
     					v(i,j,k) = SmoothZ(i,j,k-1) * v(i,j,k-1) + v(i,j,k)
     	    		end do
    	    	end do
     	    end do
    !
    		! smooth in the Z-direction (Sz^T)
     	    do j = Ny,1,-1
    	    	do i = Nx,1,-1
     	    		! v(i,j,NzEarth) = v(i,j,NzEarth)
     	    		do k = NzEarth,2,-1
     					v(i,j,k-1) = v(i,j,k-1) + SmoothZ(i,j,k-1) * v(i,j,k)
     	    		end do
    	    	end do
     	    end dom_like_in=None, smooth=[None, None, None]
    '''
    for ii in  np.arange(0,nx):
       i = ii
       for jj in np.arange(0,ny):
          j =jj
          for kk in np.arange(2, nz):
              k = kk
              tmp[i,j,k] = sm_z[i,j,k-1] * tmp[i,j,k-1] + tmp[i,j,k]

    for ii in  np.arange(nx,0,-1):
       i = ii-1
       for jj in np.arange(ny, 0, -1):
          j =jj-1
          for kk in np.arange(2, nz):
              k = kk-1
              tmp[i,j,k-1] = tmp[i,j,k-1] * sm_z[i,j,k-1] + tmp[i,j,k]

    m_like_out =  m_like_in
    return m_like_out

def print_stats(jac=np.array([]), jacmask=np.array([]), outfile=None):
    """
    print_stats.
    
    Parameters
    ----------
    jac : object
        Parameter jac.
    jacmask : object
        Parameter jacmask.
    outfile : object
        Parameter outfile.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    Prints dome info on jacobian
    """

    jdims = np.shape(jac)
    print('stats: Jacobian dimensions are:', jdims)
    if outfile is not None:
        outfile.write('Jacobian dimensions are:'+str(jdims))

    if jdims[0]==0:
        return

    mx = np.amax(jac)
    mn = np.amin(jac)
    print('stats: minimum/maximum Jacobian value is '+str(mn)+'/'+str(mx))
    if outfile is not None:
        outfile.write('Mminimum/maximum Jacobian value is '+str(mn)+'/'+str(mx))
    mn = np.amin(np.abs(jac))
    mx = np.amax(np.abs(jac))
    print('stats: minimum/maximum abs Jacobian value is '+str(mn)+'/'+str(mx))
    if outfile is not None:
        outfile.write('Minimum/maximum abs Jacobian value is '+str(mn)+'/'+str(mx))

    mjac = jac*scs.diags(jacmask,0)
    mx = np.amax(mjac)
    mn = np.amin(mjac)
    print('stats: minimum/maximum masked Jacobian value is '+str(mn)+'/'+str(mx))
    if outfile is not None:
        outfile.write('Minimum/maximum masked Jacobian value is '+str(mn)+'/'+str(mx))
    mx = np.amax(np.abs(mjac))
    mn = np.amin(np.abs(mjac))
    print('stats: minimum/maximum masked abs Jacobian value is '+str(mn)+'/'+str(mx))
    if outfile is not None: outfile.write('Minimum/maximum masked abs Jacobian value is '+str(mn)+'/'+str(mx)+'\n')
    print('\n')


def sminmax(S=None, aircells=None, seacells=None, out=True):
    """
    sminmax.
    
    Parameters
    ----------
    S : object
        Parameter S.
    aircells : object
        Parameter aircells.
    seacells : object
        Parameter seacells.
    out : object
        Parameter out.
    
    Returns
    -------
    out : object
        Function return value.
    
    Notes
    -----
    Calculates min/max for regular subsurface cells
    """

    tmp = S.copy()
    if aircells is not None:
        tmp[aircells] = np.nan
    if seacells is not None:
        tmp[seacells] = np.nan

    s_min = np.nanmin(tmp)
    s_max = np.nanmax(tmp)

    if out:
        print('S min =', s_min,' S max =', s_max)

    return s_min, s_max
