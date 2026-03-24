#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate XYZ clusters and store in model format

Reads a ModEM model file, adds perturbation bodies with optional
smoothing, and writes the modified model.

@author: vrath
Cleanup: 4 Mar 2026 by Claude (Anthropic)
"""

import os
import sys
import inspect
import time

import numpy as np
import cluster

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
titstrng = utl.print_title(
    version=version, fname=inspect.getfile(inspect.currentframe()), out=False
)
print(titstrng + "\n\n")

# =============================================================================
#  Configuration
# =============================================================================
LOGRHOAIR = np.log10(1.0e17)
LOGRHOSEA = np.log10(3.0e-1)

# MOD_FILE_IN = PY4MTX_DATA + "/Peru/1_feb_ell/TAC_100"
MOD_DIR_IN = PY4MTX_ROOT + "/py4mt/data/modem_kong/XYZ/"
MOD_BASE = MOD_DIR_IN + "rot_40_50_jack_starting_best_iso_NLCG_027"
MOD_FILES_IN = [MOD_BASE + "_X", MOD_BASE + "_X", MOD_BASE + "_X"]


# =============================================================================
#  Read model
# =============================================================================
total = 0.0
start = time.perf_counter()


elapsed = time.perf_counter() - start
total += elapsed
# print(" Used %7.4f s for reading model from %s " % (elapsed, MOD_FILE_IN + ".rho"))

data = []
for file in MOD_FILES_IN:
    dx, dy, dz, rho, refmod, _ = mod.read_mod(file, ".rho", trans="log10")
    aircells = np.where(rho > LOGRHOAIR - 1.)
    seacells = np.where(rho == LOGRHOSEA)

    r = rho.copy()

    r[rho > LOGRHOAIR - 1.] = np.nan
    r[rho == LOGRHOSEA] = np.nan

    row= r.ravel()
    data.append(row)






#     mod.write_mod(
#         Modout, modext="_new.rho", trans="LOGE",
#         dx=dx, dy=dy, dz=dz, mval=rho_out,
#         reference=refmod, mvalair=1e17, aircells=aircells, header="",
#     )


#     elapsed = time.perf_counter() - start
#     print(" Used %7.4f s for processing/writing model to %s" % (elapsed, Modout))

# if "add" in ACTION:
#     Modout = MOD_FILE_OUT + "_final"
#     mod.write_mod(
#         Modout, modext="_new.rho", trans="LOGE",
#         dx=dx, dy=dy, dz=dz, mval=rho_out,
#         reference=refmod, mvalair=1e17, aircells=aircells, header="",
#     )

total += elapsed
print(" Total time used:  %f s " % total)
