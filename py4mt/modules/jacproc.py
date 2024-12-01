#!/usr/bin/env python3
"""
Created on Sun Sep 27 17:36:08 2020

"""
from sys import exit as error
import numpy as np
import scipy.sparse as scs
import numpy.linalg as npl

from numba import jit


def calc_sensitivity(Jac=np.array([]),
                     Type = "euclidean", UseSigma = False, Small = 1.e-30, OutInfo = False):
    """
    Calculate sensitivities.
    Expects that Jacobian is already scaled, i.e Jac = C^(-1/2)*J.

    Several options exist for calculating sensiotivities, all of them
    used in the literature.
    Type:
        "raw"     sensitivities summed along the data axis
        "abs"     absolute sensitivities summed along the data axis
                    (often called coverage)
        "euc"     squared sensitivities summed along the data axis.
        "cum"     cummulated sensitivities as proposed by
                  Christiansen & Auken, 2012. Not usable for negative data.

    Usesigma:
        if true, sensitivities with respect to sigma  are calculated.

    Christiansen, A. V. & Auken, E.
    A global measure for depth of investigation
    Geophysics, 2012, 77, WB171-WB177

    from UBC:
    def depth_of_investigation_christiansen_2012(self, std, thres_hold=0.8):
        pred = self.survey._pred.copy()
        delta_d = std * np.log(abs(self.survey.dobs))
        J = self.getJ(self.model)
        J_sum = abs(Utils.sdiag(1/delta_d/pred) * J).sum(axis=0)
        S = np.cumsum(J_sum[::-1])[::-1]
        active = S-thres_hold > 0.
        doi = abs(self.survey.depth[active]).max()
        return doi, active

    T. Guenther
        Inversion Methods and Resolution Analysis for the 2D/3D Reconstruction
        of Resistivity Structures from DC Measurements
        Fakultaet für Geowissenschaften, Geotechnik und Bergbau,
        Technische Universitaet Bergakademie Freiberg, 2004.

    author:VR 9/23

    """

    if np.size(Jac)==0:
        error("calc_sensitivity: Jacobian size is 0! Exit.")

    if UseSigma:
        Jac = -Jac


    if "raw" in  Type.lower():
        S = Jac.sum(axis=0)
        if OutInfo:
            print("raw:", S)
        # else:
        #     print("raw sensitivities")
        # smax = Jac.max(axis = 0)
        # smin = Jac.max(axis = 0)
        
    elif "cov" in Type.lower():
        S = Jac.abs().sum(axis=0)
        if OutInfo:
            print("cov:", S)
        # else:
        #     print("coverage")

    elif "euc" in Type.lower():
        S = Jac.power(2).sum(axis=0)
        if OutInfo:
            print("euc:", S)
        # else:
        #     print("euclidean (default)")

    elif "cum" in Type.lower():
        S = Jac.abs().sum(axis=0)
        # print(np.shape(S))
        # S = np.sum(Jac,axis=0)

        S = np.append(0.+1.e-10, np.cumsum(S[-1:0:-1]))
        S = np.flipud(S)
        if OutInfo:
           print("cumulative:", S)
        # else:
        #    print("cumulative sensitivity")

    else:
        print("calc_sensitivity: Type "
              +Type.lower()+" not implemented! Default assumed.")
        S = Jac.power(2).sum(axis=0)

        if OutInfo:
            print("euc (default):", S)
        # else:
        #     print("euclidean (default)")

        # S = S.reshape[-1,1]
 
    S[np.where(np.abs(S)<Small)]=Small
    print("calc: ", np.any(S==0))
    # S=S.A1    
    S = np.asarray(S).ravel()
    return S


def transform_sensitivity(S=np.array([]), Vol=np.array([]),
                          Transform=["sqrt", "size","max", ],
                          asinhpar=[0.], Maxval=None, Small= 1.e-30, OutInfo=False):
    """
    Transform sensitivities.

    Several options exist for transforming sensitivities, all of them
    used in the literature.

    Normalize options:
        "siz"       Normalize by the values optional array V ("volume"), 
                    i.e in our case layer thickness. 
        "max"       Normalize by maximum value.
        "sur"       Normalize by surface value.
        "sqr"       Take the square root. Only usefull for euc sensitivities. 
        "log"       Take the logaritm. This should always be the 
                    last value in Transform list
                    
        "asinh"     asinh transform. WARNING: excludes log option, 
                    and should be used only for raw sensitivities
                    (C. Scholl, Die Periodizitaet von Sendesignalen 
                    bei Long-Offset Transient Electromagnetics, 
                    Diploma Thesis, Universität zu Koeln, 2001).

    author:VR 4/23

    """

    if np.size(S)==0:
        error("transform_sensitivity: Sensitivity size is 0! Exit.")
    
    ns = np.shape(S)
    print("transform_sensitivity: Shape = ", ns)
    

    for item in Transform:       
        
            
        if "sqr" in item.lower():
            S = np.sqrt(S)
            # print("S0s", np.shape(S))
            
        if "log" in item.lower():    
            S = np.log10(S)

        if "asinh" in item.lower():
            maxval = np.amax(S)
            minval = np.amin(S)
            if maxval>0 and minval>0:
                print("transform_sensitivity: No negatives, switched to log transform!")
                S = np.log10(S)
            else:
                if len(asinhpar)==1:
                    scale = asinhpar[0]
                else:
                    scale = get_scale(S, method=asinhpar[0])

                    S = np.arcsinh(S/scale)
        
        if ("siz" in item.lower()) or ("vol" in item.lower()):
             print("transformed_sensitivity: Transformed by volumes/layer thickness.")
             if np.size(Vol)==0:
                 error("Transform_sensitivity: no volumes given! Exit.")

             else:
                 maxval = np.amax(S)
                 minval = np.amin(S)
                 print("before volume:",minval, maxval)
                 print("volume:", np.amax(Vol),np.amax(Vol) )
                 S = S/Vol.ravel()
                 maxval = np.amax(S)
                 minval = np.amin(S)
                 print("after volume:",minval, maxval)

        if "max" in item.lower():
             print("trans_sensitivity: Transformed by maximum value.")
             if Maxval is None:
                 _, maxval = sminmax(np.abs(S))
             else:
                 maxval = Maxval
             print("maximum value: ", maxval)
             S = S/maxval
             # print("S0m", np.shape(S))
             
             
        S[np.where(np.abs(S)<Small)]=Small

        
    return S, maxval

def get_scale(d=np.array([]), f=0.1, method = "other", OutInfo = False):
    """
    Get optimal Scale for arcsin transformation.

    Parameters
    ----------
    d : float, required.
        Data vector.
    F : float, optional
        Weight for arcsinh transformation, default from Scholl & Edwards (2007)

    Returns
    -------
    S : float
        Scale value for arcsinh

    C. Scholl
        Die Periodizitaet von Sendesignalen bei Long-Offset Transient Electromagnetics
        Diploma Thesis, Institut für Geophysik und Meteorologie der Universität zu Koeln, 2001.


    """

    if np.size(d)==0:
        error("get_S: No data given! Exit.")

    if "s2007" in method.lower():
        scale = f * np.nanmax(np.abs(d))

    else:
        dmax = np.nanmax(np.abs(d))
        dmin = np.nanmin(np.abs(d))
        denom =f *(np.log(dmax)-np.log(dmin))
        scale = np.abs(dmax/denom)

    if OutInfo:
        print("Scale value S is "+str(scale)+", method "+method)

    return scale

def sparsmat_to_array(mat=None):
    """
    

    Parameters
    ----------
    mat : sparse scipy matrix
        sparse scipy matrix. The default is None.
    arr : np.array
         The default is np.array([]).

    Returns
    -------
    arr : Tnp.array
         The default is np.array([]).YPE


    """
    arr = np.array([])
    
    # data = mat.A1
    arr= np.asarray(mat).ravel()
    
    return arr


def update_avg(k = None, m_k=None, m_a=None, m_v=None):
    """
    Update the mean and variance from data stream.

    Note: final variance needs to be divided by k-1.

    Based on the formulae from
    Knuth, Art of Computer Programming, Vol 2, page 232, 1997.

    VR  Mar 7, 2021

    Note: generalization possible for skewness (M3) and kurtosis (M4)

    delta = x - M1;
    delta_n = delta / n;
    delta_n2 = delta_n * delta_n;
    term1 = delta * delta_n * n1;
    M1 += delta_n;
    M4 += term1 * delta_n2 * (n*n - 3*n + 3) + 6 * delta_n2 * M2 - 4 * delta_n * M3;
    M3 += term1 * delta_n * (n - 2) - 3 * delta_n * M2;
    M2 += term1;

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
#     """
#     Estimate the quantiles from data stream.

#     T-digest

#     VR  Mar , 2021
#     """

#     return m_med, m_q1, m_q2

def rsvd(A, rank=300, n_oversamples=None, n_subspace_iters=None, return_range=False):
    """
    =============================================================================
    Randomized SVD. See Halko, Martinsson, Tropp's 2011 SIAM paper:

    "Finding structure with randomness: Probabilistic algorithms for constructing
    approximate matrix decompositions"
    Author: Gregory Gundersen, Princeton, Jan 2019
    =============================================================================
    Randomized SVD (p. 227 of Halko et al).

    :param A:                (m x n) matrix.
    :param rank:             Desired rank approximation.
    :param n_oversamples:    Oversampling parameter for Gaussian random samples.
    :param n_subspace_iters: Number of power iterations.
    :param return_range:     If `True`, return basis for approximate range of A.
    :return:                 U, S, and Vt as in truncated SVD.
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
    """Algorithm 4.1: Randomized range finder (p. 240 of Halko et al).

    Given a matrix A and a number of samples, computes an orthonormal matrix
    that approximates the range of A.

    :param A:                (m x n) matrix.
    :param n_samples:        Number of Gaussian random samples.
    :param n_subspace_iters: Number of subspace iterations.
    :return:                 Orthonormal basis for approximate range of A.
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
    """Algorithm 4.4: Randomized subspace iteration (p. 244 of Halko et al).

    Uses a numerically stable subspace iteration algorithm to down-weight
    smaller singular values.

    :param A:       (m x n) matrix.
    :param Y0:      Initial approximate range of A.
    :param n_iters: Number of subspace iterations.
    :return:        Orthonormalized approximate range of A after power
                    iterations.
    """
    # print('herere we are in subspace-iter')
    Q = ortho_basis(Y0)
    for _ in range(n_iters):
        Z = ortho_basis(A.T @ Q)
        Q = ortho_basis(A @ Z)
    return Q


# ------------------------------------------------------------------------------


def ortho_basis(M):
    """Computes an orthonormal basis for a matrix.

    :param M: (m x n) matrix.
    :return:  An orthonormal basis for M.
    """
    # print('herere we are in ortho')
    Q, _ = np.linalg.qr(M)
    return Q


def sparsify_jac(Jac=None, 
                 sparse_thresh=1.0e-6, normalized=False, scalval = 1., 
                 method=None, out=True):
    """
    Sparsifies error_scaled Jacobian from ModEM output

    author: vrath
    last changed: Sep 10, 2023
    """
    shj = np.shape(Jac)
    if out:
        nel = shj[0] * shj[1]
        print(
            "sparsify_jac: dimension of original J is %i x %i = %i elements"
            % (shj[0], shj[1], nel)
        )
        
    
    Jf = Jac.copy()
    # print(np.shape(Jf))
    
    if scalval <0.:
        Scaleval = np.amax(np.abs(Jf))
        print("sparsify_jac: scaleval is %g (max Jacobian)" % (Scaleval))  
    else: 
        Scaleval = abs(scalval)
        print("sparsify_jac: scaleval is  %g" % (Scaleval))  
        
    if normalized:        
        print("sparsify_jac: output J is scaled by %g" % (Scaleval)) 
        f = 1.0 / Scaleval
        Jf = normalize_jac(Jac=Jf, fn=f)
    
    Jf[np.abs(Jf)/Scaleval < sparse_thresh] = 0.0
    
    # print(np.shape(Jf))

    Js = scs.csr_matrix(Jf)


    if out:
        ns = Js.count_nonzero()
        print("sparsify_jac:"
                +" output J is sparse, and has %i nonzeros, %f percent"
                % (ns, 100.0 * ns / nel))
        test = np.random.default_rng().normal(size=np.shape(Jac)[1])
        normx = npl.norm(Jf@test)
        normo = npl.norm(Jf@test-Js@test)
        

        normd = npl.norm((Jac-Jf), ord="fro")
        normf = npl.norm(Jac, ord="fro")
        # print(norma)
        # print(normf)        
        print(" Sparsified J explains "
              +str(round(100.-100.*normo/normx,2))+"% of full J (Spectral norm)")
        print(" Sparsified J explains "
              +str(round(100.-100.*normd/normf,2))+"% of full J (Frobenius norm)")
        # print("****", nel, ns, 100.0 * ns / nel, round(100.-100.*normd/normf,3) )

       

    return Js, Scaleval


def normalize_jac(Jac=None, fn=None, out=True):
    """
    normalize Jacobian from ModEM data err.

    author: vrath
    last changed: Sep30, 2023
    """
    shj = np.shape(Jac)
    shf = np.shape(fn)
    # print("fn = ")
    # print(fn)
    if shf[0] == 1:
        f = 1.0 / fn[0]
        Jac = f * Jac
    else:
        fd = 1./fn[:]
        fd = fd.flatten()
        print(fd.shape)
        erri = scs.diags([fd], [0], format="csr")
        Jac = erri @ Jac
        #erri = np.reshape(1.0 / fn, (shj[0], 1))
        #Jac = erri[:] * Jac

    return Jac

def set_padmask(rho=None, pad=[0, 0 , 0, 0, 0, 0], blank= np.nan, flat=True, out=True):
    """
    Set model masc for Jacobian calculations.

    author: vrath
    last changed: Dec 29, 2021

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
        mask = jm.flatten(order="F")

    return mask


def set_airmask(rho=None, aircells=np.array([]), blank= 1.e-30, flat=False, out=True):
    """
    Set aircell masc for Jacobian calculations.

    author: vrath
    last changed: Dec 29, 2021

    """
    shr = np.shape(rho)
    # jm = np.full(shr, np.nan)
    jm = np.full(shr, 1.)
    print(np.shape(jm), shr)
    
    jm[aircells] = blank
    mask = jm
    if flat:
        # mask = jm.flatten()
        mask = jm.flatten(order="F")

    return mask


def project_nullspace(U=np.array([]), m_test=np.array([])):
    """
    Calculates nullspace projection of a vector

    Parameters
    ----------
    U : numpy array, float
         npar*npar matrix from SAVD oj Jacobian.
    m_test : numpy array, float
         npar*vector to be projected.

    Returns
    -------
    m: numpy array, float
        projected model

    """
    if np.size(U) == 0:
        error("project_nullspace: V not defined! Exit.")

    m_proj = m_test - U@(U.T@m_test)

    return m_proj

def project_models(m=None, U=None, tst_sample= None, nsamp=1, small=1.0e-14, out=True):
    """
    Project to Nullspace.

    (see Munoz & Rath, 2006)
    author: vrath
    last changed: Feb 29, 2024
    """
    if m.ndim(m)>1:
        m = m.flatten(order="F")
    
    if tst_sample  is None:
        print("project_model: "+str(nsamp)+" sample models will be generated!")
        if nsamp==0:
           error("project_model: No number of samples given! Exit.") 
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
    Sample Posterior Covariance.
    Algorithm given by  Osypov (2013)

    Parameters
    ----------

    Returns
    -------
    spc_sanple

    References:

    Osypov K, Yang Y, Fournier A, Ivanova N, Bachrach R, 
        Can EY, You Y, Nichols D, Woodward M (2013)
        Model-uncertainty quantification in seismic tomography: method and applications 
        Geophysical Prospecting, 61, pp. 1114–1134, 2013, doi: 10.1111/1365-2478.12058.
  

    """
    error("sample_pcovar: Not yet fully implemented! Exit.")
    
    if (cpsqrti is None) or  (m is None):
        error("sample_pcovar: No covarince or ref model given! Exit.")
    

    
    if tst_sample is None:
        print("sample_pcovar: "+str(nsamp)+" sample models will be generated!")
        if nsamp==0:
           error("sample_pcovar: No number of samples given! Exit.") 
        tst_sample = np.random.default_rng().normal(0., 1., (nsamp, len(m)))
        
    else:
        nsamp = np.shape(tst_sample)[0]
        
        
    spc_sample = np.zeros(nsamp, len(m))
        
    for isamp in np.arange(nsamp):
        spc_sample[isamp,:] = m + cpsqrti@tst_sample[isamp,:]
    
    return spc_sample


def mult_by_cmsqr(m_like_in=None, smooth=[None, None, None], small=1.0e-14, out=True):
    """
    Multyiply by sqrt of paramter prior covariance (aka "smoothing")
    baed on the ModEM fortran code 
    
    Parameters
    ----------
    m_like_in : TYPE, optional
        DESCRIPTION. The default is None.
    smooth : TYPE, optional
        DESCRIPTION. The default is None.
    out : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    =============================================================================
    
       subroutine RecursiveAR(w,v,n)
    
        ! Implements the recursive autoregression algorithm for a 3D real array.
        ! In our case, the assumed-shape array would be e.g. conductivity
        ! in each cell of the Nx x Ny x NzEarth grid.
    
        real (kind=prec), intent(in)     :: w(:,:,:)
        real (kind=prec), intent(out)    :: v(:,:,:)
        integer, intent(in)                      :: n
        integer                                  :: Nx, Ny, NzEarth, i, j, k, iSmooth
    
        Nx      = size(w,1)
     	Ny      = size(w,2)
     	NzEarth = size(w,3)
    
     	if (maxval(abs(shape(w) - shape(v)))>0) then
    		call errStop('The input arrays should be of the same shapes in RecursiveAR')
     	end if
    
     	v = w
    
     	do iSmooth = 1,n
    
    		! smooth in the X-direction (Sx)
     	    do k = 1,NzEarth
    	    	do j = 1,Ny
     	    		!v(1,j,k) = v(1,j,k)
     	    		do i = 2,Nx
     					v(i,j,k) = SmoothX(i-1,j,k) * v(i-1,j,k) + v(i,j,k)
     	    		end do
    	    	end do
     	    end do
    
    		! smooth in the Y-direction (Sy)
     	    do k = 1,NzEarth
    	    	do i = 1,Nx
     	    		! v(i,1,k) = v(i,1,k)
     	    		do j = 2,Ny
     					v(i,j,k) = SmoothY(i,j-1,k) * v(i,j-1,k) + v(i,j,k)
     	    		end do
    	    	end do
     	    end do
    
    		! smooth in the Z-direction (Sz)
     	    do j = 1,Ny
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
     	    end do
    
    		! smooth in the Y-direction (Sy^T)
     	    do k = NzEarth,1,-1
    	    	do i = Nx,1,-1
     	    		! v(i,Ny,k) = v(i,Ny,k)
     	    		do j = Ny,2,-1
     					v(i,j-1,k) = v(i,j-1,k) + SmoothY(i,j-1,k) * v(i,j,k)
     	    		end do
    	    	end do
     	    end do
    
     	    ! smooth in the X-direction (Sx^T)
     	    do k = NzEarth,1,-1
    	    	do j = Ny,1,-1
     	    		! v(Nx,j,k) = v(Nx,j,k)
     	    		do i = Nx,2,-1
     					v(i-1,j,k) = v(i-1,j,k) + SmoothX(i-1,j,k) * v(i,j,k)
     	    		end do
    	    	end do
     	    end do
    
        end do
    
     	! apply the scaling operator C
        do k = 1,NzEarth
         	do j = 1,Ny
        		do i = 1,Nx
    				v(i,j,k) = (Scaling(i,j,k)**n) * v(i,j,k)
        		end do
         	end do
        end do
    
      end subroutine RecursiveAR
    =============================================================================

    """
    # nsmooth = 1
    
    
    error("mult_by_cmsq: Not yet implemented. Exit")
    
    tmp = m_like_in.copy()
    
    nx, ny,nz = np.shape(m_like_in)
    
    sm_x, sm_y, sm_z = smooth   
    
    """
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
    """
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
    Prints dome info on jacobian
    """
    
    jdims = np.shape(jac)
    print("stats: Jacobian dimensions are:", jdims)
    if outfile is not None:
        outfile.write("Jacobian dimensions are:"+str(jdims))
    
    if jdims[0]==0:
        return
        
    mx = np.amax(jac)
    mn = np.amin(jac)
    print("stats: minimum/maximum Jacobian value is "+str(mn)+"/"+str(mx))  
    if outfile is not None:
        outfile.write("Mminimum/maximum Jacobian value is "+str(mn)+"/"+str(mx))
    mn = np.amin(np.abs(jac))
    mx = np.amax(np.abs(jac))
    print("stats: minimum/maximum abs Jacobian value is "+str(mn)+"/"+str(mx))
    if outfile is not None:
        outfile.write("Minimum/maximum abs Jacobian value is "+str(mn)+"/"+str(mx))
  
    mjac = jac*scs.diags(jacmask,0)
    mx = np.amax(mjac)
    mn = np.amin(mjac)
    print("stats: minimum/maximum masked Jacobian value is "+str(mn)+"/"+str(mx))
    if outfile is not None:
        outfile.write("Minimum/maximum masked Jacobian value is "+str(mn)+"/"+str(mx))
    mx = np.amax(np.abs(mjac))
    mn = np.amin(np.abs(mjac))
    print("stats: minimum/maximum masked abs Jacobian value is "+str(mn)+"/"+str(mx))
    if outfile is not None: outfile.write("Minimum/maximum masked abs Jacobian value is "+str(mn)+"/"+str(mx)+"\n")
    print("\n")


def sminmax(S=None, aircells=None, seacells=None, out=True):
    """
    Calculates min/max for regular subsurface cells
    """

    tmp = S.copy()
    tmp[aircells] = np.nan
    tmp[seacells] = np.nan

    s_min = np.nanmin(tmp)
    s_max = np.nanmax(tmp)

    if out:
        print("S min =", s_min," S max =", s_max)

    return s_min, s_max


