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

Uncertainty bands
-----------------

Set ``BANDS`` to either **quantile** pairs (0..1) or **percentile** pairs (0..100).

Examples::

    # quantiles
    BANDS = ((0.10, 0.90), (0.25, 0.75))

    # percentiles
    BANDS = ((10.0, 90.0), (25.0, 75.0))

Internally, percentiles are converted to quantiles before being passed into
:mod:`mcmc_viz` (which expects quantiles).

ArviZ compatibility
-------------------

This driver assumes **ArviZ >= 0.23.4** and loads the netCDF via
``arviz.from_netcdf``.

Author: Volker Rath (DIAS)
Created with the help of ChatGPT (GPT-5 Thinking) on 2026-02-10 (UTC)
"""

from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path
from typing import Iterable, Tuple

import arviz as az
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

# Uncertainty bands (quantile pairs 0..1 or percentile pairs 0..100)
BANDS = ((10.0, 90.0),)  # e.g. ((10, 90), (25, 75)) or ((0.1, 0.9),)

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


def _normalize_bands(
    pairs: Iterable[Tuple[float, float]] | None,
) -> Tuple[Tuple[Tuple[float, float], ...] | None, str]:
    """Normalize bands to *quantiles*.

    Parameters
    ----------
    pairs
        Sequence of (lo, hi) pairs. If any value is > 1, pairs are interpreted
        as percentiles (0..100). Otherwise as quantiles (0..1).

    Returns
    -------
    qpairs
        Pairs converted to quantiles (0..1), or None.
    kind
        Either "percentile" or "quantile" (or "none").

    Notes
    -----
    This driver normalizes to quantiles because the plotting backend
    (:func:`mcmc_viz.plot_paramset_threepanel`) expects quantiles.
    """
    if pairs is None:
        return None, "none"

    pairs = tuple((float(a), float(b)) for (a, b) in pairs)
    if not pairs:
        return None, "none"

    maxv = max(max(a, b) for (a, b) in pairs)
    if maxv > 1.0 + 1.0e-12:
        qpairs = tuple((a / 100.0, b / 100.0) for (a, b) in pairs)
        return qpairs, "percentile"
    return pairs, "quantile"


def _open_idata(nc_path: Path):
    """Open an ArviZ InferenceData (netCDF).

    Uses :func:`arviz.from_netcdf` (ArviZ >= 0.23.x) and returns None on failure.
    """
    try:
        return az.from_netcdf(nc_path.as_posix())
    except Exception:
        return None


def main() -> int:
    """Run the plot driver.

    Returns
    -------
    int
        Exit code (0 on success).
    """
    version, _ = versionstrg()
    fname = inspect.getfile(inspect.currentframe())
    titstrng = utl.print_title(version=version, fname=fname, out=False)
    print(titstrng + "\n\n")

    file_list = mcmc.glob_inputs(SEARCH_GLOB)
    if not file_list:
        raise SystemExit(f"No matching file found: {SEARCH_GLOB} (exit).")

    qpairs, band_kind = _normalize_bands(BANDS)

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

        idata = _open_idata(nc_path) if nc_path.is_file() else None
        if idata is None and nc_path.is_file():
            print(f"  Could not open idata: {nc_path}. Using summary only.")

        fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharey=True)

        mv.plot_paramset_threepanel(
            axs,
            summary=summary,
            idata=idata,
            param_domain=PARAM_DOMAIN,
            param_set=PARAM_SET,
            qpairs=qpairs,
            show_quantile_lines=SHOW_BAND_EDGES,
            prefer_idata=True,
            overlay_single=summary.get("model0") if isinstance(summary.get("model0"), dict) else None,
        )

        fig.suptitle(f"{station} | {PARAM_DOMAIN}:{PARAM_SET} | {band_kind}")
        fig.tight_layout()

        for fmt in PLOT_FORMATS:
            out = PLOT_DIR / f"{station}{NAME_SUFFIX}{fmt}"
            fig.savefig(out.as_posix(), dpi=600)

        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
