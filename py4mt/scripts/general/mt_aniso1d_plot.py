#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting driver for anisotropic 1-D MT inversion results.

Produces a three-panel figure per station showing vertical step profiles
of the sampled parameter sets. Supports posterior uncertainty bands from
ArviZ InferenceData (NetCDF) or falls back to summary NPZ arrays.

Panel contents depend on the parameterisation used by the sampler:

- Resistivity domain (PARAM_DOMAIN="rho")
    - PARAM_SET="minmax":     (rho_min, rho_max, strike)
    - PARAM_SET="max_anifac": (rho_max, rho_anifac, strike)

- Conductivity domain (PARAM_DOMAIN="sigma")
    - PARAM_SET="minmax":     (sigma_min, sigma_max, strike)
    - PARAM_SET="max_anifac": (sigma_max, sigma_anifac, strike)

@author:    Volker Rath (DIAS)
@project:   py4mt — Python for Magnetotellurics
@created:   2026-02-13 with the help of ChatGPT (GPT-5 Thinking)
"""

from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import arviz as az

# --- Py4MTX environment -------------------------------------------------------

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

for _base in [PY4MTX_ROOT + "/py4mt/modules/"]:
    for _p in [Path(_base), *Path(_base).rglob("*")]:
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

import mcmc
import mcmc_viz as mv
import util as utl
from version import versionstrg

# =============================================================================
#  Configuration
# =============================================================================

# Domain and parameter set to plot.
# Use "auto" to pick up the parametrisation actually used by the sampler.
PARAM_DOMAIN = "auto"  # "auto", "rho" or "sigma"
PARAM_SET = "auto"     # "auto", "minmax" or "max_anifac"

# Uncertainty bands (used when idata exists). Can be given either as
# quantiles (0..1) or percentiles (0..100).
BANDS = ((10.0, 90.0), (25.0, 75.0))

# Optional: show dashed band edges in addition to the shaded bands
SHOW_BAND_EDGES = False

# Files / folders
DATA_DIR = Path(PY4MTX_DATA) / "edi"
SUMM_DIR = DATA_DIR / "pmc_demetropolis_hfix"
PLOT_DIR = DATA_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

PLOT_FORMATS = (".pdf",)  # (".pdf", ".png")
NAME_SUFFIX = "_pmc_threepanel"

# Station selector — enumerate from sampler summary files
SEARCH_GLOB = str(SUMM_DIR / "*_pmc_summary.npz")


# =============================================================================
#  Helper functions
# =============================================================================
def _bands_to_qpairs(bands):
    """Convert bands (quantiles or percentiles) to quantile pairs."""
    qpairs = []
    if bands is None:
        return tuple(qpairs)

    for lo, hi in bands:
        lo = float(lo)
        hi = float(hi)
        if (lo > 1.0) or (hi > 1.0):
            lo /= 100.0
            hi /= 100.0
        qpairs.append((lo, hi))
    return tuple(qpairs)


def _infer_param_domain(summary, idata):
    """Infer 'rho'/'sigma' from sampler metadata or variable presence."""
    # 1) explicit metadata
    info = summary.get("info") if isinstance(summary, dict) else None
    if isinstance(info, dict):
        v = info.get("param_domain")
        if isinstance(v, str) and v.strip().lower() in ("rho", "sigma"):
            return v.strip().lower()

    v = summary.get("param_domain") if isinstance(summary, dict) else None
    if isinstance(v, str) and v.strip().lower() in ("rho", "sigma"):
        return v.strip().lower()

    # 2) summary arrays
    if isinstance(summary, dict):
        keys = {str(k) for k in summary.keys()}
        if any("sigma_" in k for k in keys):
            return "sigma"
        if any("rho_" in k for k in keys):
            return "rho"

    # 3) idata posterior variable names
    try:
        if idata is not None and hasattr(idata, "posterior"):
            vnames = set(getattr(idata.posterior, "data_vars", {}).keys())
            if any("sigma" in n for n in vnames):
                return "sigma"
            if any("rho" in n for n in vnames):
                return "rho"
    except Exception:
        pass

    return "rho"  # conservative default


def _infer_param_set(summary):
    """Infer parameter set name from sampler metadata; default to 'minmax'."""
    info = summary.get("info") if isinstance(summary, dict) else None
    if isinstance(info, dict):
        v = info.get("param_set")
        if isinstance(v, str) and v.strip().lower() in ("minmax", "max_anifac"):
            return v.strip().lower()

    v = summary.get("param_set") if isinstance(summary, dict) else None
    if isinstance(v, str) and v.strip().lower() in ("minmax", "max_anifac"):
        return v.strip().lower()

    return "minmax"


# =============================================================================
#  Main
# =============================================================================
def main() -> int:
    """Run the plot driver."""

    version, _ = versionstrg()
    fname = inspect.getfile(inspect.currentframe())
    titstrng = utl.print_title(version=version, fname=fname, out=False)
    print(titstrng + "\n\n")

    file_list = mcmc.glob_inputs(SEARCH_GLOB)
    if not file_list:
        raise SystemExit(f"No matching file found: {SEARCH_GLOB} (exit).")

    qpairs = _bands_to_qpairs(BANDS)

    for f in file_list:
        sum_path = Path(f)
        stem = sum_path.stem
        station = stem[:-12] if stem.endswith("_pmc_summary") else stem
        print(f"--- {station} ---")

        nc_path = SUMM_DIR / f"{station}_pmc.nc"

        summary = mv.load_summary_npz(sum_path)

        idata = None
        if nc_path.is_file():
            try:
                idata = az.from_netcdf(nc_path.as_posix())
            except Exception as e:
                print(f"  Could not open idata: {nc_path} ({e}). Using summary only.")

        fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharey=True)

        param_domain = (
            _infer_param_domain(summary, idata)
            if str(PARAM_DOMAIN).strip().lower() == "auto"
            else str(PARAM_DOMAIN).strip().lower()
        )
        param_set = (
            _infer_param_set(summary)
            if str(PARAM_SET).strip().lower() == "auto"
            else str(PARAM_SET).strip().lower()
        )

        mv.plot_paramset_threepanel(
            axs,
            summary=summary,
            idata=idata,
            param_domain=param_domain,
            param_set=param_set,
            qpairs=qpairs,
            show_quantile_lines=SHOW_BAND_EDGES,
            prefer_idata=True,
            overlay_single=(
                summary.get("model0")
                if isinstance(summary.get("model0"), dict)
                else None
            ),
        )

        fig.suptitle(f"{station} | {param_domain}:{param_set}")
        fig.tight_layout()

        for fmt in PLOT_FORMATS:
            out = PLOT_DIR / f"{station}{NAME_SUFFIX}{fmt}"
            fig.savefig(out.as_posix(), dpi=600)

        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
