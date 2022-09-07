#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 17:36:08 2020

"""
import numpy as np
import scipy.sparse as scp
import numpy.linalg as npl

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
    O = np.random.randn(n, n_samples)
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


def sparsify_jac(Jac=None, sparse_thresh=1.0e-6, normalized=False, method=None, out=True):
    """
    Sparsifies error_scaled Jacobian from ModEM output

    author: vrath
    last changed: Sep 25, 2020
    """
    shj = np.shape(Jac)
    if out:
        nel = shj[0] * shj[1]
        print(
            "sparsifyJac: dimension of original J is %i x %i = %i elements"
            % (shj[0], shj[1], nel)
        )
    Jf = np.abs(Jac.copy())
    Jmax = np.amax(Jf)
    thresh = sparse_thresh
    Jf[Jf < thresh] = 0.0
    Js = scp.csr_matrix(Jf)

    if out:
        if scp.issparse(Js):
            ns = scp.csr_matrix.count_nonzero(Js)
            print(
                "sparsifyJac:"
                +" output J is sparse, and has %i nonzeros, %f percent"
                % (ns, 100.0 * ns / nel)
            )
        x = np.random.normal(size=np.shape(Jac)[1])
        # print(np.shape(x))
        # print(np.shape(Js))
        normx = npl.norm(x)
        norma = npl.norm(Js@x)/normx
        normf = npl.norm(Jf@x)/normx
        # print(norma)
        # print(normf)
        print(" Sparsified J explains "
              +str(round(norma*100./normf))+"% of full J.")

    if normalized:
        f = 1.0 / Jmax
        Js = f * Js

    return Js, Jmax


def normalize_jac(Jac=None, fn=None, out=True):
    """
    normalize Jacobian from ModEM data err.

    author: vrath
    last changed: Mar 8, 2021
    """
    shj = np.shape(Jac)
    shf = np.shape(fn)
    if shf[0] == 1:
        f = 1.0 / fn[0]
        Jac = f * Jac
    else:
        erri = np.reshape(1.0 / fn, (shj[0], 1))
        Jac = erri[:] * Jac

    return Jac

def set_mask(rho=None, pad=[10, 10 , 10, 10, 0, 10], blank= np.nan, flat=True, out=True):
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



def calculate_sens(Jac=None, normalize=False, small=1.0e-15, blank=np.nan, log=False, out=True):
    """
    Calculate sensitivity from ModEM Jacobian.
    Optionally blank elements smaller than theshold, take logarithm, normalize.

    author: vrath
    last changed: Sep 25, 2020
    """
    if scp.issparse(Jac):
        J = Jac.todense()
    else:
        J = Jac

    S = np.sum(np.power(J, 2), axis=0)
    S = np.sqrt(S)

    Smax = np.nanmax(S)
    Smin = np.nanmin(S)

    if out:
        print("Range of S is "+str(Smin)+" to "+str(Smax))

    if normalize:
        S = S / Smax
        if out:
            print("Normalizing with"+str(Smax))

    if log:
        S = np.log10(S)

    S[S < small] = blank

    return S, Smax


def project_model(m=None, U=None, small=1.0e-14, out=True):
    """
    Project to Nullspace.

    (see Munoz & Rath, 2006)
    author: vrath
    last changed: Sep 25, 2020
    """
    b = np.dot(U.T, m)
    # print(m.shape)
    # print(b.shape)
    # print(U.shape)

    mp = m - np.dot(U, b)

    return mp


def transfrom_model(m=None, M=None, small=1.0e-14, out=True):
    """
    Transform Model.

    M should be something like C_m^-1/2
    ( see egg Kelbert 2012, Egbert & kelbert 2014)
    author: vrath
    last changed:  Oct 12, 2020
    """
    transm = np.dot(M, m)

    return transm
