#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Thu May 29 10:09:45 2025

@author: vrath
'''

# Import python modules
# edit according to your needs

import os
import sys

import time
from datetime import datetime
import warnings
import csv
import inspect
import argparse

# Import numerical or other specialized modules
import numpy as np
import pandas as pd


PY4MTX_DATA = os.environ['PY4MTX_DATA']
PY4MTX_ROOT = os.environ['PY4MTX_ROOT']

# add py4mt modules to pythonpath
mypath = [PY4MTX_ROOT+'/py4mt/modules/',
          PY4MTX_ROOT+'/py4mt/scripts/']
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0,pth)

# Import required py4mt modules for your script
import util as utl
from version import versionstrg

import femtic_slice as fs

# import data_proc as mp
# import plotrjmcmc as plmc
# import viz
# import inverse as inv
# import femtic as fem
# import femtic_viz as femviz


rng = np.random.default_rng()
nan = np.nan  # float('NaN')
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())
titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng+'\n\n')

# =============================================================================
#  Configuration
INPUT_DIR = PY4MTX_ROOT + "py4mt/data/edi/"
# WORK_DIR = "/home/vrath/MT_Data/waldim/edi_synth_iso/"
if not os.path.isdir(INPUT_DIR):
   sys.exit(" File: %s does not exist! Exit." % INPUT_DIR)
PLOT_DIR = INPUT_DIR 
# ---------------------------------------------------------------------------
# Entry point  (improvements #6, #7: mesh_path and resdir wired through)
# ---------------------------------------------------------------------------

def run(
    param_file: str,
    mesh_path: str = "mesh.dat",
    resdir: str = ".",
    old_format: bool = False,
) -> None:
    """
    Run the full cutaway pipeline.

    Parameters
    ----------
    param_file : str
        Path to the FEMTIC parameter file.
    mesh_path : str
        Path to the mesh file (default: ``"mesh.dat"`` in the current
        working directory).  (improvement #6)
    resdir : str
        Directory containing ``resistivity_block_iter<N>.dat`` files.
        (improvements #6, #7)
    old_format : bool
        Pass True to use the ``#ifdef _OLD`` resistivity-block format.
        (improvement #1)
    """
    params = read_param_file(param_file)

    center_m     = params["center_km"] * 1000.0
    rotation_rad = params["rotation_deg"] * DEG2RAD

    print(f"Reading {mesh_path} …")
    mesh = read_mesh(mesh_path)
    print(f"  mesh type: {mesh.mesh_type}, "
          f"nodes: {len(mesh.node_xyz)}, elements: {len(mesh.elem_nodes)}")

    rblk_name = f"resistivity_block_iter{params['iter_num']}.dat"
    print(f"Reading {resdir}/{rblk_name} …")
    res_block = read_resistivity_block(
        params["iter_num"], resdir=resdir, old_format=old_format)
    print(f"  {len(res_block.rho)} blocks, "
          f"{int(res_block.fixed.sum())} fixed, "
          f"{int(res_block.is_negative.sum())} negative-sentinel")

    print("Computing cutaway …")
    polygons, log10_rho_list, is_negative_list = make_cutaway(
        mesh, res_block,
        params["plane_type"], center_m, rotation_rad,
        params["excluded_blocks"],
    )
    print(f"  {len(polygons)} polygons found")

    stem = f"resistivity_GMT_iter{params['iter_num']}"
    write_gmt(stem + ".dat", polygons, log10_rho_list)
    write_npz(stem, polygons, log10_rho_list, is_negative_list, params=params)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FEMTIC resistivity cutaway tool (Python port)")
    parser.add_argument("param_file",
                        help="Parameter file (same format as C++ version)")
    parser.add_argument("--mesh", default="mesh.dat", metavar="PATH",
                        help="Path to mesh.dat (default: mesh.dat)")
    parser.add_argument("--resdir", default=".", metavar="DIR",
                        help="Directory containing resistivity_block_iter*.dat "
                             "(default: current directory)")
    parser.add_argument("--old-format", action="store_true",
                        help="Use the legacy (#ifdef _OLD) resistivity-block format")
    args = parser.parse_args()
    run(args.param_file,
        mesh_path  = args.mesh,
        resdir     = args.resdir,
        old_format = args.old_format)


if __name__ == "__main__":
    main()