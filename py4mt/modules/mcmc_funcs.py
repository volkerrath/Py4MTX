#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 19:58:45 2025

@author: vrath
"""

import numpy as np

# def run_mcmc_emcee(model, data, **kwargs):
#     results = {} 
#     return results  
    

def run_emcee(nwalkers, ndim, log_posterior, **args):
    import multiprocessing as mp
    import emcee 

    
    with mp.Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(spacings, y_obs, sigma_obs), pool=pool)
        # burn-in
        pos, prob, state = sampler.run_mcmc(p0, 1000, progress=True)
        sampler.reset()
        # production
        sampler.run_mcmc(pos, 30000, progress=True)
        return sampler

    sampler = run_emcee()
    
    
    
def run_pmc():
    
    import pymc as pm
    import matplotlib.pyplot as plt
    import arviz as az
    
    sampler = pm.NUTS()
    
    
    
    
    with pm.Model() as model:
        
        # Log-scale parameters
        log_p1 = pm.Normal("log_p1", mu=0, sigma=1)
        log_p2 = pm.Normal("log_p2", mu=0, sigma=1)
        log_p3 = pm.Normal("log_p3", mu=0, sigma=1)
    
        p1 = pm.Deterministic("p1", pm.math.exp(log_p1))
        p2 = pm.Deterministic("p2", pm.math.exp(log_p2))
        p3 = pm.Deterministic("p3", pm.math.exp(log_p3))
    
        # Linear parameters
        p4 = pm.Normal("p4", mu=0, sigma=1)
        p5 = pm.Normal("p5", mu=0, sigma=1)
        p6 = pm.Normal("p6", mu=0, sigma=1)
    
        # Nonlinear model
        y_est = p1 * pm.math.exp(-p2 * x_data) + p3 + p4 * x_data + p5 * x_data**2 + p6
    
        # Likelihood
        y_obs = pm.Normal("y_obs", mu=y_est, sigma=0.1, observed=y_data)
    
        # Sample using NUTS
        trace = pm.sample(draws=2000, tune=1000, target_accept=0.95, return_inferencedata=True)

        # Sample with NUTS (tune and target_accept tuned for stability)
        trace = pm.sample(step = samplwe, 
                          draws=2000,
                          tune=2000, 
                          chains=4, 
                          cores=4, 
                          target_accept=0.9, 
                          return_inferencedata=True)
# # -------------------------
#     # Diagnostics and results
#     # -------------------------
#     # Summary
#     print(az.summary(trace, var_names=["rho1", "rho2", "h1", "sigma"], round_to=2))
    
#     # Trace and pair plots
#     az.plot_trace(trace, var_names=["rho1", "rho2", "h1"]);
#     az.plot_pair(trace, var_names=["rho1", "rho2", "h1"], kind="kde", marginals=True);
    
#     # Posterior predictive
#     with model:
#         ppc = pm.sample_posterior_predictive(trace, var_names=["obs"], random_seed=2)
#     ppc_y = ppc["obs"]
    
#     # Plot fit
#     plt.figure(figsize=(7,4))
#     plt.errorbar(spacings, y_obs, yerr=sigma_obs, fmt="k.", label="observed")
#     # plot posterior predictive percentiles
#     ppc_q = np.percentile(ppc_y, [5,50,95], axis=0)
#     plt.semilogx(spacings, ppc_q[1], color="C1", label="median posterior pred")
#     plt.fill_between(spacings, ppc_q[0], ppc_q[2], color="C1", alpha=0.3, label="90% CI")
#     plt.xlabel("Spacing (a.u.)")
#     plt.ylabel("Apparent resistivity (Ohm·m)")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# https://copilot.microsoft.com/shares/c9Gb1SVGqT11eAmR942fw