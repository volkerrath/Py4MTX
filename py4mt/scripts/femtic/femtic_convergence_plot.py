#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot convergence curves (misfit, nRMS, or roughness vs. iteration)
from FEMTIC inversion runs.

Reads FEMTIC convergence files (default femtic.cnv, see CNV_FILE) from a
set of inversion directories and generates convergence plots as PDF.

@author: vrath

Provenance:
    2025       vrath   Created.
    2026-03-03 Claude  Renamed user-set parameters to UPPERCASE.
    2026-08-13 Claude Sonnet 5 (Anthropic)
                       Added femtic_convergence_plot_summary.md output at
                       end of run: writes user-set (UPPERCASE) parameters,
                       script path, and run date/time via
                       utl.write_param_summary().
    2026-10-08 Claude Sonnet 5.5 (Anthropic)
                Convergence file name is now configurable (CNV_FILE,
                default "femtic.cnv"; a list of candidates is allowed,
                first existing wins) via fem.resolve_cnv_path(), which
                requires the femtic.py of the same date. Also replaced the
                hard-coded column indices (nline[5], [7], [8]) by
                fem.read_cnv(), which reads column positions from each
                file's header row, as in the other femtic_* scripts.
                AI-generated; review before production use.
"""

import os
import sys
from pathlib import Path
import inspect

import numpy as np
import matplotlib.pyplot as plt

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

for _base in [PY4MTX_ROOT + "/py4mt/modules/"]:
    for _p in [Path(_base), *Path(_base).rglob("*")]:
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

import femtic as fem
import util as utl
from version import versionstrg

rng = np.random.default_rng()
nan = np.nan
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())
titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng + "\n\n")

# =============================================================================
#  Configuration
# =============================================================================
WORK_DIR = r"/home/vrath/FEMTIC_work/krafla6big_L2_L_curve/"
PLOT_NAME = r"Krafla_L2_Convergence"
PLOT_WHAT = "rough"  # Options: 'misfit', 'rms', 'rough'

#: Name of the FEMTIC convergence file inside each run directory. A single
#: name or a list of candidate names (first existing one is used).
CNV_FILE = "femtic.cnv"

SEARCH_STRNG = "kra*"
dir_list = utl.get_filelist(
    searchstr=[SEARCH_STRNG], searchpath=WORK_DIR,
    sortedlist=True, fullpath=True,
)

# =============================================================================
#  Read convergence data and plot
# =============================================================================
for directory in dir_list:
    cnv_path = fem.resolve_cnv_path(directory, CNV_FILE)
    # Column positions come from the file's own header row via
    # fem.read_cnv(), robust to FEMTIC version and to the presence of
    # Beta/Distortion columns.
    try:
        rows = fem.read_cnv(directory, filename=CNV_FILE)["rows"]
    except ValueError as err:
        print(cnv_path, "does not contain a valid .cnv file:", err)
        continue

    if len(rows) == 0:
        print(cnv_path, "is empty!")
        continue

    # x-axis is the running row index (retrials count as separate points).
    c = np.array([[k, r["Alpha"], r["Roughness"], r["Misfit"], r["RMS"]]
                  for k, r in enumerate(rows)])
    itern = c[:, 0]
    alpha = c[:, 1]
    rough = c[:, 2]
    misft = c[:, 3]
    nrmse = c[:, 4]

    print("#iter", itern)
    print("#misfit", misft)
    print("#nrmse", nrmse)
    print("#rough", rough)

    # Plotting parameters
    plot_kwargs = dict(
        color="green", marker="o", linestyle="dashed",
        linewidth=1, markersize=7,
        markeredgecolor="red", markerfacecolor="white",
    )

    fig, ax = plt.subplots()
    plot_what = PLOT_WHAT.lower()

    if "mis" in plot_what:
        print("plotting misfit")
        formula = r"$\Vert\mathbf{C}_d^{-1/2} (\mathbf{d}_{obs}-\mathbf{d}_{calc})\Vert_2$"
        plt.semilogy(itern, misft, **plot_kwargs)
        plt.ylabel(r"misfit " + formula, fontsize=14)

    elif "rms" in plot_what:
        print("plotting nrmse")
        formula = r"$\sqrt{N^{-1} \mathbf{C}_d^{-1/2} (\mathbf{d}_{obs}-\mathbf{d}_{calc})_2}$"
        plt.plot(itern, nrmse, **plot_kwargs)
        plt.ylabel(r"nRMS " + formula, fontsize=14)

    elif "rough" in plot_what:
        print("plotting roughness")
        formula = r"$\Vert\mathbf{C}_m^{-1/2} \mathbf{m}\Vert_2$"
        plt.semilogy(itern, rough, **plot_kwargs)
        plt.ylabel(r"roughness " + formula, fontsize=14)

    else:
        sys.exit(
            f"plot_convergence: plotting parameter '{plot_what}' not implemented! Exit."
        )

    plt.title(
        PLOT_NAME.replace("_", " ") + r" |   $\alpha$ = "
        + str(round(alpha[0], 2))
    )
    plt.xlabel(r"iteration", fontsize=14)
    plt.grid("on")
    plt.tight_layout()

    outfile = (
        WORK_DIR + PLOT_NAME + "_" + plot_what
        + "_alpha" + str(round(alpha[0], 2)) + ".pdf"
    )
    plt.savefig(outfile)
    print(f"Saved {outfile}")


# ---------------------------------------------------------------------------
# Parameter summary
# ---------------------------------------------------------------------------
utl.write_param_summary(fname)
