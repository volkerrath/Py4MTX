
"""
femtic_mesh_sampler.py
======================
Sample a lognormal resistivity field on a FEMTIC TETRA mesh using spatial covariance.

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-11-10
"""

from __future__ import annotations
import argparse, numpy as np, os
from femtic_mesh_io import read_femtic_tetra_centroids
from femtic_sample_resistivity import draw_logrho_field, save_npz

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mesh', required=True, help='Path to FEMTIC mesh.dat (TETRA format)')
    ap.add_argument('--kernel', default='matern', choices=['matern','exponential','gaussian'])
    ap.add_argument('--ell', type=float, default=500.0)
    ap.add_argument('--sigma2', type=float, default=0.5)
    ap.add_argument('--nu', type=float, default=1.5)
    ap.add_argument('--nugget', type=float, default=1e-6)
    ap.add_argument('--mean', type=float, default=np.log(100.0), help='Mean of log-resistivity')
    ap.add_argument('--strategy', default='sparse', choices=['dense','sparse'])
    ap.add_argument('--radius', type=float, default=None)
    ap.add_argument('--trunc_k', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--out', default='rho_sample_on_mesh.npz')
    args = ap.parse_args()

    C, tet_ids = read_femtic_tetra_centroids(args.mesh)

    if args.strategy == 'dense':
        rho, logrho = draw_logrho_field(
            C, kernel=args.kernel, sigma2=args.sigma2, ell=args.ell, nu=args.nu,
            nugget=args.nugget, mean_log_rho=args.mean, strategy='dense', random_state=args.seed
        )
    else:
        if args.radius is None:
            # heuristic: 2.5 * ell cutoff
            args.radius = 2.5 * args.ell
        rho, logrho = draw_logrho_field(
            C, kernel=args.kernel, sigma2=args.sigma2, ell=args.ell, nu=args.nu,
            nugget=args.nugget, mean_log_rho=args.mean, strategy='sparse',
            radius=args.radius, trunc_k=args.trunc_k, random_state=args.seed
        )

    # Save NPZ with mapping to tet ids
    save_npz(args.out, rho=rho, logrho=logrho, centroids=C, tet_ids=tet_ids, meta=dict(vars(args)))

if __name__ == '__main__':
    main()
