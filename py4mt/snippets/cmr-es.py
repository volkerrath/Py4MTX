# CMR-ES proposed on ESANN 2022
# Paper "A Fast and Simple Evolution Strategy with Covariance Matrix Estimation"
# by Oliver Kramer, Computational Intelligence Lab, University of Oldenburg

import numpy as np
import math

N = 5
kappa = 200
eta = 20

archive = []

def rosenbrock(x):
	return np.sum([100*(x[i+1]-x[i]**2)**2 + (1-x[i])**2 for i in range(N-1)]) 


def fitfunction(x):
	return rosenbrock(x)


def termination(t,fit):
	return True if (t>3000 or fit < 10e-500) else False


def estimate_C(x_, sigma):
	global archive
	X_ = np.array(archive)
	C = np.cov(X_.T)
	return C


def CMRES():

	global archive
	fitlog = []
	t=1
	happy = False
	sigma = 0.5	

	x = np.zeros(N)
	fit = fitfunction(x)
	C = np.identity(N)
	d = math.sqrt(N)

	while not happy:

		happy = termination(t,fit)
		x_ = x + sigma * np.random.multivariate_normal(np.zeros(N),C)
		fit_ = fitfunction(x_)

		if fit_ <= fit:
			x = x_
			fit = fit_
			sigma*=np.exp(4/5/d)
			archive.insert(0,x)
			archive=archive[:eta]

		else: 
			sigma*=np.exp(-1/5/d)

		fitlog.append(fit)

		if t%kappa==0:
			C = estimate_C(x_=x,sigma=sigma)

		t+=1

		print("Fitness in generation",t,"is",fit)

	return fitlog


runs = [CMRES() for i in range(10)]

print("25 CMR-ES runs on Rosenbrock, N=",N)

print("mean",np.mean([runs_[-1] for runs_ in runs]))
print("std",np.std([runs_[-1] for runs_ in runs]))
