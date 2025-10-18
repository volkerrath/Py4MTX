#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 18:54:57 2025

@author: vrath
"""

import numpy as np
from fenics import *

# Setup mesh and function space
mesh = UnitSquareMesh(32, 32)
V = FunctionSpace(mesh, 'P', 1)
obs_points = [Point(0.25, 0.25), Point(0.75, 0.75)]

# Forward model
def solve_forward(a_vec):
    a_func = Function(V)
    a_func.vector()[:] = a_vec
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(1.0)
    a_expr = interpolate(a_func, V)
    a_form = inner(a_expr * grad(u), grad(v)) * dx
    L = f * v * dx
    u_sol = Function(V)
    solve(a_form == L, u_sol, DirichletBC(V, Constant(0.0), "on_boundary"))
    return np.array([u_sol(p) for p in obs_points])

# Synthetic observations
true_a = interpolate(Expression("1 + 0.5*sin(2*pi*x[0])*cos(2*pi*x[1])", degree=2), V)
y_obs = solve_forward(true_a.vector()) + np.random.normal(0, 0.01, len(obs_points))

# Ensemble initialization
ensemble_size = 50
param_dim = V.dim()
a_ensemble = [np.random.normal(1.0, 0.2, param_dim) for _ in range(ensemble_size)]

# EKS parameters
dt = 0.1
iterations = 30

for k in range(iterations):
    u_ensemble = np.array([solve_forward(a) for a in a_ensemble])
    a_array = np.array(a_ensemble)

    a_mean = np.mean(a_array, axis=0)
    u_mean = np.mean(u_ensemble, axis=0)

    C_au = (a_array - a_mean).T @ (u_ensemble - u_mean) / ensemble_size
    C_uu = (u_ensemble - u_mean).T @ (u_ensemble - u_mean) / ensemble_size + 0.01**2 * np.eye(len(obs_points))
    K = C_au @ np.linalg.inv(C_uu)

    # Langevin update with noise
    for i in range(ensemble_size):
        noise = np.random.normal(0, 1, param_dim)
        a_ensemble[i] -= dt * K @ (u_ensemble[i] - y_obs)
        a_ensemble[i] += np.sqrt(2 * dt) * noise @ np.cov(a_array.T)

    print(f"Iter {k+1}: mean a at center = {np.mean([a[V.dofmap().dofs()[V.tabulate_dof_coordinates().tolist().index([0.5, 0.5])] ] for a in a_ensemble]):.3f}")

# Final posterior samples
posterior_samples = np.array(a_ensemble)
