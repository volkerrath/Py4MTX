"""
femtic_mesh_sampler.py
======================

Command-line interface to sample lognormal resistivity fields on a
FEMTIC TETRA mesh using spatial covariance (Matérn, Exponential, or
Gaussian kernels).

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking) on 2025-11-18
"""

from __future__ import annotations

import argparse

import numpy as np

from femtic_mesh_io import read_femtic_tetra_centroids
from femtic_sample_resistivity import draw_logrho_field


def main() -> None:
    """
    Parse CLI arguments, sample a resistivity field, and save the result.

    The script reads a FEMTIC TETRA mesh, builds a covariance matrix on
    tetra centroids, draws a Gaussian field in log-resistivity space,
    exponentiates to obtain resistivity, and writes everything to an NPZ
    file.

    The NPZ file contains:

        - ``rho``: resistivity samples (Ohm·m),
        - ``logrho``: log-resistivity Gaussian field,
        - ``centroids``: tetra centroids,
        - ``tet_ids``: tetra indices from the mesh file,
        - ``meta``: a dictionary of the command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Sample a covariance-driven lognormal resistivity field "
            "on a FEMTIC TETRA mesh."
        ),
    )
    parser.add_argument(
        "--mesh",
        required=True,
        help="Path to FEMTIC mesh.dat (TETRA format).",
    )
    parser.add_argument(
        "--kernel",
        default="matern",
        choices=["matern", "exponential", "gaussian"],
        help="Covariance kernel family.",
    )
    parser.add_argument(
        "--ell",
        type=float,
        default=500.0,
        help="Correlation length scale (same units as mesh coordinates).",
    )
    parser.add_argument(
        "--sigma2",
        type=float,
        default=0.5,
        help="Log-space marginal variance.",
    )
    parser.add_argument(
        "--nu",
        type=float,
        default=1.5,
        help="Matérn smoothness parameter (if Matérn kernel is used).",
    )
    parser.add_argument(
        "--nugget",
        type=float,
        default=1e-6,
        help="Diagonal nugget term in log-space.",
    )
    parser.add_argument(
        "--mean",
        type=float,
        default=float(np.log(100.0)),
        help="Mean of log-resistivity.",
    )
    parser.add_argument(
        "--strategy",
        default="sparse",
        choices=["dense", "sparse"],
        help="Sampling strategy: dense (Cholesky) or sparse (trunc-eig).",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help="Neighborhood radius for sparse covariance (default ~ 2.5*ell).",
    )
    parser.add_argument(
        "--trunc_k",
        type=int,
        default=1024,
        help="Rank for truncated-eigen sampling when using sparse covariance.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--out",
        default="rho_sample_on_mesh.npz",
        help="Output NPZ file path.",
    )

    args = parser.parse_args()

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
