#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mt_aniso1d_plot.py
=====================

Plotting driver for the simplified anisotropic 1-D MT inversion results.

This script is a *calling script* in the style used throughout your Py4MTX
workflow:

- it keeps the PY4MTX environment variable conventions
- it sets up the module search path
- it controls figure layout and file saving

The actual plotting logic lives in :mod:`mcmc_viz`.

Main output
-----------

For each station, a **three-panel** figure is produced:

``fig, axs = plt.subplots(3, 1, figsize=(8, 8))``

Each panel shows a vertical step profile of one of the 3-parameter sets:

- Resistivity domain (``PARAM_DOMAIN='rho'``)
    - ``PARAM_SET='minmax'``: (rho_min, rho_max, strike)
    - ``PARAM_SET='max_anifac'``: (rho_max, rho_anifac, strike)

- Conductivity domain (``PARAM_DOMAIN='sigma'``)
    - ``PARAM_SET='minmax'``: (sigma_min, sigma_max, strike)
    - ``PARAM_SET='max_anifac'``: (sigma_max, sigma_anifac, strike)

If the netCDF InferenceData exists, posterior samples are used and one or more
quantile bands can be shaded (``QPAIRS``). If not, the script falls back to the
median/qlo/qhi arrays stored in the summary NPZ.

Author: Volker Rath (DIAS)
Created with the help of ChatGPT (GPT-5 Thinking) on 2026-02-08 (UTC)
"""

from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt


# --- Py4MTX environment -------------------------------------------------------

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [f"{PY4MTX_ROOT}/py4mt/modules/", f"{PY4MTX_ROOT}/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)


# Local imports (Py4MTX)
import mcmc
import mcmc_viz as mv
import util as utl
from version import versionstrg


# --- User settings ------------------------------------------------------------

# Domain and parameter set to plot
PARAM_DOMAIN = "rho"  # "rho" or "sigma"
PARAM_SET = "minmax"  # "minmax" or "max_anifac"

# Quantile bands (only used when idata exists)
QPAIRS = ((0.1, 0.9),)  # e.g. ((0.1, 0.9), (0.2, 0.8))

# Optional: show dashed band edges in addition to the shaded bands
SHOW_BAND_EDGES = False

# Files / folders
DATA_DIR = Path(PY4MTX_DATA) / "edi"
SUMM_DIR = DATA_DIR / "pmc_demetropolis_hfix"  # adjust to your sampler output folder
PLOT_DIR = DATA_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

PLOT_FORMATS = (".pdf",)  # (".pdf", ".png")
NAME_SUFFIX = "_pmc_threepanel"

# Station file selector (site NPZ files used only to enumerate stations)
SEARCH_GLOB = str(DATA_DIR / "Ann*.npz")


def main() -> int:
    """Run the plot driver."""

    version, _ = versionstrg()
    fname = inspect.getfile(inspect.currentframe())
    titstrng = utl.print_title(version=version, fname=fname, out=False)
    print(titstrng + "\n\n")

    file_list = mcmc.glob_inputs(SEARCH_GLOB)
    if not file_list:
        raise SystemExit(f"No matching file found: {SEARCH_GLOB} (exit).")

    for f in file_list:
        site = mcmc.load_site(f)
        station = site["station"]
        print(f"--- {station} ---")

        sum_path = SUMM_DIR / f"{station}_pmc_summary.npz"
        nc_path = SUMM_DIR / f"{station}_pmc.nc"

        if not sum_path.is_file():
            print(f"  Missing summary: {sum_path} (skip)")
            continue

        summary = mv.load_summary_npz(sum_path)

        idata = None
        if nc_path.is_file():
            try:
                idata = mv.open_idata(nc_path)
            except Exception as e:
                print(f"  Could not open idata: {nc_path} ({e}). Using summary only.")

        fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharey=True)

        mv.plot_paramset_threepanel(
            axs,
            summary=summary,
            idata=idata,
            param_domain=PARAM_DOMAIN,
            param_set=PARAM_SET,
            qpairs=QPAIRS,
            show_quantile_lines=SHOW_BAND_EDGES,
            prefer_idata=True,
            overlay_single=summary.get("model0") if isinstance(summary.get("model0"), dict) else None,
        )

        fig.suptitle(f"{station} | {PARAM_DOMAIN}:{PARAM_SET}")
        fig.tight_layout()

        for fmt in PLOT_FORMATS:
            out = PLOT_DIR / f"{station}{NAME_SUFFIX}{fmt}"
            fig.savefig(out.as_posix(), dpi=600)

        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
