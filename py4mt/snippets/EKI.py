#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 18:57:24 2025

@author: vrath
"""

import numpy as np

def forward_model(u):
    return u**2 + 1  # Example nonlinear model

# Observations
y_obs = np.array([5.0])
noise_std = 0.1

# Initialize ensemble
ensemble_size = 20
u_ensemble = np.random.normal(2.0, 0.5, ensemble_size)

# EKI iterations
for k in range(30):
    y_ensemble = forward_model(u_ensemble)
    u_mean = np.mean(u_ensemble)
    y_mean = np.mean(y_ensemble)

    C_uy = np.mean((u_ensemble - u_mean) * (y_ensemble - y_mean))
    C_yy = np.var(y_ensemble) + noise_std**2
    K = C_uy / C_yy

    for i in range(ensemble_size):
        u_ensemble[i] += -K * (y_ensemble[i] - y_obs)

    print(f"Iter {k+1}: mean u = {np.mean(u_ensemble):.3f}")
