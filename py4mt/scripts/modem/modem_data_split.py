#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split ModEM data files into period-band subsets.

Reads ModEM data files and writes separate files for each specified
period interval, suitable for band-by-band inversion or analysis.

@author: vrath (Feb 2021 / May 2024)
Cleanup: 4 Mar 2026 by Claude (Anthropic)
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
DAT_DIR_IN = PY4MTX_DATA + "/Fogo/"
DAT_DIR_OUT = DAT_DIR_IN

if not os.path.isdir(DAT_DIR_OUT):
    print("Directory: %s does not exist, but will be created" % DAT_DIR_OUT)
    os.mkdir(DAT_DIR_OUT)

DAT_FILES_IN = ["FOG_Z_in.dat", "FOG_P_in.dat", "FOG_T_in.dat"]

PER_INTERVALS = [
    [0.0001, 0.001],
    [0.001, 0.01],
    [0.01, 0.1],
    [0.1, 1.0],
    [1.0, 10.0],
    [10.0, 100.0],
    [100.0, 1000.0],
    [1000.0, 10000.0],
    [10000.0, 1000000.0],
]

PER_NUM_MIN = 1
NUM_BANDS = len(PER_INTERVALS)

# =============================================================================
#  Split data by period band
# =============================================================================
for datfile in DAT_FILES_IN:

    for ibnd in np.arange(NUM_BANDS):

        lowstr = str(1.0 / PER_INTERVALS[ibnd][0]) + "Hz"
        uppstr = str(1.0 / PER_INTERVALS[ibnd][1]) + "Hz"

        with open(DAT_DIR_IN + datfile) as fd:
            head = []
            data = []
            site = []
            perd = []
            for line in fd:
                if line.startswith("#") or line.startswith(">"):
                    head.append(line)
                    continue

                per = float(line.split()[0])
                sit = line.split()[1]
                if per >= PER_INTERVALS[ibnd][0] and per < PER_INTERVALS[ibnd][1]:
                    data.append(line)
                    site.append(sit)
                    perd.append(per)

        nper = len(np.unique(perd))
        nsit = len(np.unique(site))
        print(nper, "periods from", nsit, "sites")

        if nper > 0 and nsit > 0:
            phead = head.copy()
            phead = [lins.replace("per", str(nper)) for lins in phead]
            phead = [lins.replace("sit", str(nsit)) for lins in phead]

            outfile = DAT_DIR_IN + datfile
            outfile = outfile.replace("_in.dat", "_perband" + str(ibnd) + ".dat")
            print("output to", outfile)

            with open(outfile, "w") as fo:
                for ilin in np.arange(len(phead)):
                    fo.write(phead[ilin])
                for ilin in np.arange(len(data)):
                    fo.write(data[ilin])
