#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image-processing filters for ModEM resistivity models.

Reads a ModEM .rho model and applies one of three spatial filters
(Gaussian smoothing, median filter, anisotropic diffusion), then
writes the filtered model to a new file.

Authors: vr (Jul 2020), vrath (Feb 2021)
Provenance: cleaned/debugged with Claude (Anthropic), Mar 2026
"""

import os
import sys
import time
import inspect
from datetime import datetime
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

# ---------------------------------------------------------------------------
# Path configuration – adjust to your environment
# ---------------------------------------------------------------------------
PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [PY4MTX_ROOT + "/py4mt/modules/", PY4MTX_ROOT + "/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)

import modem as mod
import util as utl
from version import versionstrg

version, _ = versionstrg()
title_str = utl.print_title(
    version=version,
    fname=inspect.getfile(inspect.currentframe()),
    out=False,
)
print(title_str + "\n\n")

rng = np.random.default_rng()
nan = np.nan

warnings.simplefilter(action="ignore", category=FutureWarning)

# ===========================================================================
# User parameters
# ===========================================================================
RHOAIR = 1.0e17

# File stems (without .rho extension – read_mod appends it automatically)
MODFILE_IN  = r"/home/vrath/work/MT/Annecy/ImageProc/In/ANN20_02_PT_NLCG_016"
MODFILE_OUT = r"/home/vrath/work/MT/Annecy/ImageProc/Out/ANN20_02_PT_NLCG_016_ImProc"

ACTION = "smooth"

# --- Gaussian / uniform smoothing parameters ---
if "smooth" in ACTION.lower():
    FTYPE = "gaussian"
    SIGMA = 2
    ORDER = 0
    BMODE = "nearest"
    MAXIT = 3

# --- Median filter parameters ---
if "med" in ACTION.lower():
    KSIZE = 3
    BMODE = "nearest"
    MAXIT = 3

# --- Anisotropic diffusion parameters ---
if "anidiff" in ACTION.lower():
    MAXIT = 50
    FOPT = 1

# ---------------------------------------------------------------------------
# Read model
# ---------------------------------------------------------------------------
total = 0.0

start = time.time()
dx, dy, dz, rho, reference, trans = mod.read_mod(
    MODFILE_IN, trans="LINEAR", out=True,
)
elapsed = time.time() - start
total += elapsed
print(f" Used {elapsed:7.4f} s for reading model from {MODFILE_IN}.rho")

# Write a copy of the original model
mod.write_mod(
    MODFILE_OUT, dx=dx, dy=dy, dz=dz, mval=rho,
    reference=reference, trans=trans, out=True,
)

# Identify air cells and prepare model boundaries
air = rho > RHOAIR / 10.0
rho = mod.prepare_model(rho, rhoair=RHOAIR)

rho_log = np.log(rho.copy())
rhoair_log = np.log(RHOAIR)

# ---------------------------------------------------------------------------
# Apply selected filter
# ---------------------------------------------------------------------------
start = time.time()

if "smooth" in ACTION.lower():
    rho_tmp = rho_log.copy()
    for ii in range(MAXIT):
        print(f"Smoothing iteration: {ii}")
        if "gaussian" in FTYPE.lower():
            rho_tmp = gaussian_filter(
                rho_tmp, sigma=SIGMA, order=ORDER, mode=BMODE,
            )
        else:
            rho_tmp = uniform_filter(rho_tmp, size=SIGMA, mode=BMODE)

    rho_tmp[air] = rhoair_log
    modout = f"{MODFILE_OUT}_{FTYPE}_sigma{SIGMA}_iter{MAXIT}.rho"
    mod.write_mod(
        modout, modext="",
        dx=dx, dy=dy, dz=dz, mval=rho_tmp,
        reference=reference, out=True,
    )
    elapsed = time.time() - start
    print(f" Used {elapsed:7.4f} s for processing/writing model to {modout}")

if "med" in ACTION.lower():
    rhonew = mod.medfilt3D(
        rho_log,
        kernel_size=KSIZE,
        boundary_mode=BMODE,
        maxiter=MAXIT,
    )
    rhonew[air] = rhoair_log
    modout = f"{MODFILE_OUT}_median_k{KSIZE}_iter{MAXIT}.rho"
    mod.write_mod(
        modout, modext="",
        dx=dx, dy=dy, dz=dz, mval=rhonew,
        reference=reference, out=True,
    )
    elapsed = time.time() - start
    print(f" Used {elapsed:7.4f} s for processing/writing model to {modout}")

if "anidiff" in ACTION.lower():
    rhonew = mod.anidiff3D(
        rho_log,
        ckappa=20,
        dgamma=0.24,
        foption=FOPT,
        maxiter=MAXIT,
        Out=True,
    )
    rhonew[air] = rhoair_log
    modout = f"{MODFILE_OUT}_anisodiff{FOPT}_iter{MAXIT}.rho"
    mod.write_mod(
        modout, modext="",
        dx=dx, dy=dy, dz=dz, mval=rhonew,
        reference=reference, out=True,
    )
    elapsed = time.time() - start
    print(f" Used {elapsed:7.4f} s for processing/writing model to {modout}")

total += elapsed
print(f" Total time used: {total:.4f} s")
