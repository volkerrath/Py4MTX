'''
femtic_mesh_sampler.py
======================

CLI to sample lognormal resistivity fields on a FEMTIC TETRA mesh using
spatial covariance (Matern/Exponential/Gaussian).

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-11-11
'''
from __future__ import annotations

import argparse
import numpy as np

from femtic_mesh_io import read_femtic_tetra_centroids
from femtic_sample_resistivity import draw_logrho_field


def main() -> None:
    '''Parse CLI arguments, sample a resistivity field, and save NPZ.'''
    ap = argparse.ArgumentParser(description="Sample a covariance-driven resistivity field on a FEMTIC TETRA mesh.")
    ap.add_argument("--mesh", required=True, help="Path to FEMTIC mesh.dat (TETRA format).")
    ap.add_argument("--kernel", default="matern", choices=["matern", "exponential", "gaussian"])
    ap.add_argument("--ell", type=float, default=500.0, help="Length-scale (units of mesh coordinates).")
    ap.add_argument("--sigma2", type=float, default=0.5, help="Log-space marginal variance.")
    ap.add_argument("--nu", type=float, default=1.5, help="Matern smoothness (if used).")
    ap.add_argument("--nugget", type=float, default=1e-6, help="Diagonal nugget (log-space).")
    ap.add_argument("--mean", type=float, default=float(np.log(100.0)), help="Mean of log-resistivity.")
    ap.add_argument("--strategy", default="sparse", choices=["dense", "sparse"], help="Dense (Cholesky) or sparse (trunc-eig).")
    ap.add_argument("--radius", type=float, default=None, help="Neighborhood radius for sparse K (default ~ 2.5*ell).")
    ap.add_argument("--trunc_k", type=int, default=1024, help="Rank for truncated-eig sampling when sparse.")
    ap.add_argument("--seed", type=int, default=None, help="Random seed.")
    ap.add_argument("--out", default="rho_sample_on_mesh.npz", help="Output NPZ path.")
    args = ap.parse_args()

    centroids, tet_ids = read_femtic_tetra_centroids(args.mesh)

    if args.strategy == "dense":
        rho, logrho = draw_logrho_field(
            centroids,
            kernel=args.kernel,
            sigma2=args.sigma2,
            ell=args.ell,
            nu=args.nu,
            nugget=args.nugget,
            mean_log_rho=args.mean,
            strategy="dense",
            random_state=args.seed,
        )
    else:
        radius = args.radius if args.radius is not None else 2.5 * args.ell
        rho, logrho = draw_logrho_field(
            centroids,
            kernel=args.kernel,
            sigma2=args.sigma2,
            ell=args.ell,
            nu=args.nu,
            nugget=args.nugget,
            mean_log_rho=args.mean,
            strategy="sparse",
            radius=radius,
            trunc_k=args.trunc_k,
            random_state=args.seed,
        )

    np.savez(
        args.out,
        rho=rho,
        logrho=logrho,
        centroids=centroids,
        tet_ids=tet_ids,
        meta=dict(vars(args)),
    )


if __name__ == "__main__":
    main()
