#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare directories and data subsets for a jackknife-inspired uncertainty analysis.

Creates sub-directories with template files and generates reduced data sets
(e.g., leave-one-site-out) for FEMTIC inversion runs.

@author:   vrath
@project:  py4mt — Python for Magnetotellurics
@inversion: FEMTIC

Provenance:
    2026-03-03  Claude          Renamed user-set parameters to UPPERCASE;
                                 generated README.
    2026-07-25  Claude Sonnet 5 (Anthropic)
                Added RANDOM_SEED (default None) for optional reproducible
                runs, matching the pattern used in femtic_rto_prep.py /
                femtic_gst_prep.py / femtic_nss.py: rng =
                np.random.default_rng(RANDOM_SEED), resolved seed echoed
                to the console. rng is threaded into the generate_data_fcn
                call for when random "subset" mode is used.
                Flagged (did not fix) a pre-existing, unrelated issue:
                fem.generate_directories() and fem.generate_data_fcn() do
                not exist in the current femtic.py / ensembles.py — this
                script predates the consolidation of directory/data-
                ensemble generation into ensembles.py and will raise
                AttributeError as written. See the KNOWN ISSUE comment
                block near RANDOM_SEED.
    2026-08-13  Claude Sonnet 5 (Anthropic)
                Added femtic_jcn_prep_summary.md output at end of run:
                writes user-set (UPPERCASE) parameters, script path, and
                run date/time via utl.write_param_summary().
"""

import os
import sys
from pathlib import Path
import inspect

import numpy as np

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

for _base in [PY4MTX_ROOT + "/py4mt/modules/"]:
    for _p in [Path(_base), *Path(_base).rglob("*")]:
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

import femtic as fem
import util as utl
from version import versionstrg

nan = np.nan
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())
titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng + "\n\n")

# -----------------------------------------------------------------------
# Reproducibility (optional)
# -----------------------------------------------------------------------
# Set RANDOM_SEED to an integer for a reproducible run (relevant only if/
# when CHOICE_MODE uses random subset selection, e.g. ["subset", N] --
# leave-one-site-out ("site" mode) is deterministic and doesn't consume
# rng draws at all). None (default) uses fresh OS entropy.
RANDOM_SEED = None   # e.g. 20260725 for a reproducible run; None = fresh entropy

rng = np.random.default_rng(RANDOM_SEED)
print(f"RNG seed: {RANDOM_SEED if RANDOM_SEED is not None else '(fresh entropy — not reproducible)'}\n")

# -----------------------------------------------------------------------
# KNOWN ISSUE (unrelated to the reproducibility change above):
# fem.generate_directories() and fem.generate_data_fcn() below do not
# exist in the current femtic.py / ensembles.py. This script predates the
# consolidation of directory/data-ensemble generation into ensembles.py
# (compare femtic_rto_prep.py / femtic_gst_prep.py, which call
# ens.generate_directories() / ens.generate_data_ensemble()). As written,
# both calls below will raise AttributeError before the RNG is ever used.
# RANDOM_SEED / rng are wired through anyway so this script is consistent
# with the rest of the project once the calls are migrated or a
# jackknife-specific generator is added to ensembles.py -- ask if you'd
# like that migration done as a follow-up.
# -----------------------------------------------------------------------

# =============================================================================
#  Configuration
# =============================================================================
ENSEMBLE_DIR = r"/home/vrath/work/Ensemble/Ubinas_ens/"
TEMPLATES = ENSEMBLE_DIR + "templates/"
FILES = [
    "control.dat",
    "observe.dat",
    "mesh.dat",
    "resistivity_block_iter0.dat",
    "distortion_iter0.dat",
    "run_femtic_dub.sh",
    "run_femtic_oar.sh",
]

CHOICE_MODE = ["site"]

# Read site count from control.dat when using site-based jackknife
if "site" in CHOICE_MODE:
    with open(TEMPLATES + "control.dat", "r") as file:
        content = file.readlines()
    tmp = content[0].split()
    N_SAMPLES = int(tmp[0])  # number of sites determines jackknife sample count

# Alternative: random subset mode
# N_SAMPLES = 32
# CHOICE_MODE = ["subset", N_SAMPLES]

# =============================================================================
#  Generate directories
# =============================================================================
os.chdir(ENSEMBLE_DIR)

dir_list = fem.generate_directories(
    dir_base=ENSEMBLE_DIR + "jcn_",
    templates=TEMPLATES,
    file_list=FILES,
    N_samples=N_SAMPLES,
    out=True,
)

# =============================================================================
#  Draw reduced data sets based on sites
# =============================================================================
data_ensemble = fem.generate_data_fcn(
    dir_base=ENSEMBLE_DIR + "ens_",
    N_samples=N_SAMPLES,
    file_in="observe.dat",
    choice_mode=CHOICE_MODE,
    rng=rng,   # only exercised by random "subset" mode; verify this kwarg
               # exists once generate_data_fcn is restored/migrated
    out=True,
)


# ---------------------------------------------------------------------------
# Parameter summary
# ---------------------------------------------------------------------------
utl.write_param_summary(fname)
