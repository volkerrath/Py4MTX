import os
import sys
import inspect

import numpy as np
import scipy as scp
import scipy.linalg as scl
import scipy.ndimage as sci

import sklearn.cluster as scl


def soft_thresh(x, lam):
    """
    function S_lambda = soft_thresh(x,lambda)
        S_lambda = sign(x) .*  max(abs(x) - lambda,0);
    end
    """
    s_lam = np.sign(x) *  np.amax(np.abs(x) - lam, 0.)

    return s_lam

def splitbreg(J, y, lam, D, c=0., tol=1.e-5, maxiter=10):
    """
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

    """

    (nd,nm) = np.shape(J)

    mu = 2.*lam

    b = np.zeros((nd, 1))
    d = np.zeros((nd, 1))

    A = np.array([J], [np.sqrt(mu)*D])
    xold = np.nan*np.ones((nm,1))
    for kk in np.arange(maxiter):
        r = np.array[[y],[np.sqrt(mu)*(d-b)]]
        x = scl.solve(A,r, assume_a="general")
        s = soft_thresh(c+D@x+d, lam/mu)
        d = s - c
        b = b + D@x -s
        
        if ((kk>0) and (np.norm(xold-x)/np.norm(x)<tol)):
            break
        
        xold = x

    return x
