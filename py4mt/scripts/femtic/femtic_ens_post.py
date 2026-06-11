#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_ens_post.py — Ensemble postprocessing for FEMTIC

Collects all converged members of a FEMTIC ensemble (RTO, GST, or any
directory-based ensemble), computes summary statistics, assembles the
empirical covariance, and saves everything to a compressed ``.npz`` file.

Optionally produces slice figures for the **best-nRMS member** (QC) and
for the **ensemble statistics** (mean, variance, median, MAD).

References
----------
Bardsley, J. M.; Solonen, A.; Haario, H. & Laine, M.
    Randomize-Then-Optimize: a Method for Sampling from Posterior
    Distributions in Nonlinear Inverse Problems.
    SIAM J. Sci. Comp., 2014, 36, A1895-A1910.

Blatter, D.; Morzfeld, M.; Key, K. & Constable, S.
    Uncertainty quantification for regularized inversion of electromagnetic
    geophysical data. Part I: Motivation and Theory.
    Geophysical Journal International, doi:10.1093/gji/ggac241, 2022.

Blatter, D.; Morzfeld, M.; Key, K. & Constable, S.
    Uncertainty quantification for regularized inversion of electromagnetic
    geophysical data – Part II: application in 1-D and 2-D problems.
    Geophysical Journal International, doi:10.1093/gji/ggac242, 2022.

@author: vrath

Provenance
----------
2025-04-30  vrath
            Created as femtic_rto_post.py.
2026-03-03  Claude (Anthropic)
            Renamed user-set parameters to UPPERCASE; generated README.
2026-05-27  vrath / Claude Sonnet 4.6 (Anthropic)
            Added femtic_viz import; MESH_FILE / PLOT_QC / PLOT_QC_FILE /
            PLOT_QC_SLICES / PLOT_QC_* config vars; QC slice plot of
            best-nRMS member at end of main block (calls
            fviz.plot_model_slices).
2026-06-11  vrath / Claude Sonnet 4.6 (Anthropic)
            Renamed femtic_rto_post.py → femtic_ens_post.py for
            algorithm-agnostic use.  Fixed axis bug: mean/var/median/MAD
            were computed over axis=1 (free parameters) instead of axis=0
            (members) — statistics are now correct.  PLOT_QC block
            replaced by full PLOT_SLICES framework matching
            femtic_mod_edit.py / femtic_mod_math.py: UTM origin
            resolution, CRS-aware fem.resolve_slice_positions, site
            overlay via _resolve_origin_and_sites(); new PLOT_STATS block
            plots mean/variance/median/MAD as individual slice figures.
            ENSEMBLE_PREFIX added for generic naming of output keys.
"""
from __future__ import annotations

import os
import sys
import inspect
from pathlib import Path

import numpy as np

import sklearn.covariance
import scipy.sparse as scs

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

for _base in [PY4MTX_ROOT + "/py4mt/modules/"]:
    for _p in [Path(_base), *Path(_base).rglob("*")]:
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

import femtic as fem
import ensembles as ens
import util as utl
from version import versionstrg

try:
    import femtic_viz as fviz
except ImportError:
    fviz = None

rng = np.random.default_rng()
nan = np.nan

version, _ = versionstrg()
titstrng = utl.print_title(version=version, fname=__file__, out=False)
print(titstrng + "\n\n")

# ===========================================================================
# Configuration
# ===========================================================================

# ---------------------------------------------------------------------------
# Ensemble input
# ---------------------------------------------------------------------------
ENSEMBLE_DIR  = r"/home/vrath/work/Ensembles/RTO/"
ENSEMBLE_NAME = "rto_*"

#: Prefix used for .npz output keys and default file/figure names.
#: e.g. "rto" → keys rto_ens, rto_avg, …  and file RTO_results.npz.
ENSEMBLE_PREFIX = "rto"

#: Maximum normalised RMS accepted from femtic.cnv.
NRMS_MAX = 1.4

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
#: Percentile levels. Default: 2-σ / 1-σ normal-equivalent.
PERCENTILES = [2.3, 15.9, 50.0, 84.1, 97.7]

# ---------------------------------------------------------------------------
# Covariance
# ---------------------------------------------------------------------------
SPARSIFY     = True
SPARSE_THRESH = 1.0e-8   # relative threshold for zeroing small entries

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
ENSEMBLE_RESULTS = ENSEMBLE_DIR + ENSEMBLE_PREFIX.upper() + "_results.npz"

# ---------------------------------------------------------------------------
# Mesh (required for any slice plot)
# ---------------------------------------------------------------------------
MESH_FILE = ENSEMBLE_DIR + "templates/mesh.dat"

# ---------------------------------------------------------------------------
# QC slice plot — best-nRMS converged member
# ---------------------------------------------------------------------------
#: Set True to plot the best-nRMS member.
PLOT_QC      = False
PLOT_QC_FILE = ENSEMBLE_DIR + ENSEMBLE_PREFIX + "_qc.pdf"
PLOT_QC_DPI  = 200

# ---------------------------------------------------------------------------
# Statistics slice plots — mean / variance / median / MAD
# ---------------------------------------------------------------------------
#: Set True to write derived stat members as block files and plot them.
#: Requires MESH_FILE and a valid TEMPLATE_FILE (taken from best member).
PLOT_STATS      = False
#: Which statistics to plot.  Subset of: "avg", "var", "med", "mad".
PLOT_STATS_WHAT = ["avg", "var", "med", "mad"]
#: Output directory for stat block files and figures.
PLOT_STATS_DIR  = ENSEMBLE_DIR + "stats_plots/"
PLOT_STATS_DPI  = 200

# ---------------------------------------------------------------------------
# Shared slice / plot parameters (used by both PLOT_QC and PLOT_STATS)
# ---------------------------------------------------------------------------
PLOT_SLICES = [
    dict(kind="map", z0=5000.0),
    dict(kind="map", z0=15000.0),
    dict(kind="ns",  x0=0.0),
    dict(kind="ew",  y0=0.0),
]

PLOT_CMAP        = "turbo_r"
PLOT_CLIM        = [0.0, 4.0]    # log10(Ω·m); None = auto
PLOT_XLIM        = None           # [xmin, xmax] model-local metres; None = auto
PLOT_YLIM        = None
PLOT_ZLIM        = None
PLOT_OCEAN_COLOR = "lightgrey"
PLOT_OCEAN_RHO   = 0.25           # Ω·m sentinel for ocean cells
PLOT_AIR_BGCOLOR = None           # axes facecolor for air; None = figure default

DEPTH_KM          = True
HORIZ_KM          = True
PLOT_EQUAL_ASPECT = True
PLOT_PANEL_HEIGHT = 16.0    # cm
PLOT_NROWS        = None
PLOT_NCOLS        = None

# ---------------------------------------------------------------------------
# Geographic / UTM origin
# ---------------------------------------------------------------------------
UTM_ORIGIN_LAT    = None
UTM_ORIGIN_LON    = None
UTM_ORIGIN_E      = None
UTM_ORIGIN_N      = None
UTM_ZONE_OVERRIDE = None
#: "box" = bounding-box midpoint; "average" = centroid; None = use above literals.
ORIGIN_METHOD     = "box"

DISPLAY_COORDS = "model"    # "model" | "utm" | "latlon"

# ---------------------------------------------------------------------------
# Site overlay
# ---------------------------------------------------------------------------
SITE_DAT    = ENSEMBLE_DIR + "templates/site.dat"
SITE_NAMES  = None

PLOT_SITES_MAPS   = True
PLOT_SITES_SLICES = False
PROJECTION_DIST   = 5000.0  # m

SITE_MARKER        = dict(marker="v", color="black", ms=8, zorder=10, label=None)
SITE_MARKER_SLICES = None
MAP_MARKERS        = []

# ---------------------------------------------------------------------------
# Verbose output
# ---------------------------------------------------------------------------
OUT = True

# ===========================================================================
# Helpers
# ===========================================================================

def _resolve_origin_and_sites():
    """Estimate UTM origin from SITE_DAT; collect site model-local coords.

    Returns
    -------
    utm_e, utm_n, utm_lat, utm_lon : float | None
    utm_zone, utm_northern : str | None, bool | None
    site_xys : list of (name, x_m, y_m, elev)
    """
    _e   = UTM_ORIGIN_E
    _n   = UTM_ORIGIN_N
    _lat = UTM_ORIGIN_LAT
    _lon = UTM_ORIGIN_LON
    _zone, _north = None, None

    if ORIGIN_METHOD is not None and SITE_DAT and os.path.isfile(SITE_DAT):
        _sdat = fem.read_site_dat(SITE_DAT)
        if _sdat:
            _Es  = np.array([d["easting"]  for d in _sdat])
            _Ns  = np.array([d["northing"] for d in _sdat])
            if ORIGIN_METHOD == "box":
                _e = 0.5 * (_Es.min() + _Es.max())
                _n = 0.5 * (_Ns.min() + _Ns.max())
            else:
                _e = float(_Es.mean())
                _n = float(_Ns.mean())
            _lats = np.array([d["lat"] for d in _sdat])
            _lons = np.array([d["lon"] for d in _sdat])
            _zone, _north = utl.utm_zone_from_latlon(
                float(_lats.mean()), float(_lons.mean()),
                override=UTM_ZONE_OVERRIDE,
            )
            _lat, _lon = utl.utm_to_latlon_zn(_e, _n, _zone, _north)

    if _lat is not None and _lon is not None:
        _zone, _north = utl.utm_zone_from_latlon(
            _lat, _lon, override=UTM_ZONE_OVERRIDE
        )

    site_xys = []
    if SITE_DAT and os.path.isfile(SITE_DAT):
        for row in fem.read_site_dat(SITE_DAT, site_names=SITE_NAMES):
            sx, sy = fem.utm_to_model(
                row["easting"], row["northing"], _e, _n
            )
            site_xys.append(
                (row["name"], sx, sy, float(row.get("elev", 0.0)))
            )

    return _e, _n, _lat, _lon, _zone, _north, site_xys


def _plot_slice(block_file: str, pdf_file: str,
                utm_e, utm_n, utm_lat, utm_lon,
                utm_zone, utm_north, site_xys: list,
                dpi: int = 200) -> None:
    """Call fviz.plot_model_slices with full CRS-aware config."""
    if fviz is None:
        print("  plot_slice: femtic_viz not available — skipping.")
        return

    _slices_resolved = fem.resolve_slice_positions(
        PLOT_SLICES, utm_zone, utm_north,
        utm_e, utm_n, utm_lat, utm_lon,
        verbose=OUT,
    )
    fviz.plot_model_slices(
        model_file         = block_file,
        mesh_file          = MESH_FILE,
        slices             = _slices_resolved,
        cmap               = PLOT_CMAP,
        clim               = PLOT_CLIM,
        xlim               = PLOT_XLIM,
        ylim               = PLOT_YLIM,
        zlim               = PLOT_ZLIM,
        ocean_color        = PLOT_OCEAN_COLOR,
        ocean_value        = PLOT_OCEAN_RHO,
        air_bgcolor        = PLOT_AIR_BGCOLOR,
        site_xys           = site_xys,
        obs_coords_only    = False,
        sites_in_maps      = PLOT_SITES_MAPS,
        sites_in_slices    = PLOT_SITES_SLICES,
        site_marker        = SITE_MARKER,
        site_marker_slices = SITE_MARKER_SLICES,
        map_markers        = MAP_MARKERS,
        projection_dist    = PROJECTION_DIST,
        display_coords     = DISPLAY_COORDS,
        utm_origin_e       = utm_e,
        utm_origin_n       = utm_n,
        utm_zone           = utm_zone,
        utm_northern       = utm_north,
        utm_to_latlon_fn   = utl.utm_to_latlon_zn,
        latlon_to_model_fn = fem.latlon_to_model,
        depth_km           = DEPTH_KM,
        horiz_km           = HORIZ_KM,
        equal_aspect       = PLOT_EQUAL_ASPECT,
        panel_height       = PLOT_PANEL_HEIGHT / 2.54,
        nrows              = PLOT_NROWS,
        ncols              = PLOT_NCOLS,
        plot_file          = pdf_file,
        dpi                = dpi,
        out                = OUT,
    )
    if OUT:
        print(f"  saved → {pdf_file}")


# ===========================================================================
# Main
# ===========================================================================

# --- (1) Scan ensemble directories ----------------------------------------
dir_list = utl.get_filelist(
    searchstr=[ENSEMBLE_NAME],
    searchpath=ENSEMBLE_DIR,
    fullpath=True,
)
print(f"Found {len(dir_list)} sub-directory/ies matching '{ENSEMBLE_NAME}'.")

model_list  = []          # list of [block_file, n_iter, nRMS]
model_count = 0
ens_matrix  = None        # will become (n_members, n_free) float64

for d in dir_list:
    print(f"\n  Inversion run: {d}")
    cnv_file = os.path.join(d, "femtic.cnv")
    if not os.path.isfile(cnv_file):
        print(f"    femtic.cnv not found — skipped.")
        continue

    with open(cnv_file) as _fh:
        cnv = _fh.readlines()
    info  = cnv[-1].split()
    numit = int(info[0])
    nrms  = float(info[8])

    if nrms > NRMS_MAX:
        print(f"    nRMS={nrms:.4f} > NRMS_MAX={NRMS_MAX} — skipped.")
        continue

    mod_file = os.path.join(d, f"resistivity_block_iter{numit}.dat")
    if not os.path.isfile(mod_file):
        print(f"    {mod_file} not found — skipped.")
        continue

    print(f"    iter={numit}  nRMS={nrms:.4f}  {mod_file}")
    model_list.append([mod_file, numit, nrms])

    log_m = fem.read_model(model_file=mod_file, model_trans="log10", out=OUT)

    if ens_matrix is None:
        ens_matrix = log_m[np.newaxis, :]         # (1, n_free)
    else:
        ens_matrix = np.vstack((ens_matrix, log_m))   # (k, n_free)

    model_count += 1

n_members = model_count
print(f"\nConverged members: {n_members}")

if n_members == 0:
    sys.exit("No converged members found. Nothing to do.")

# ens_matrix shape: (n_members, n_free)
# axis=0 → reduce over members  (correct for all aggregate statistics)
# axis=1 → reduce over free parameters (was the bug in the original script)

# --- (2) Summary statistics -----------------------------------------------
P        = ENSEMBLE_PREFIX
ne       = ens_matrix.shape

ens_avg  = np.mean  (ens_matrix, axis=0)                           # (n_free,)
ens_var  = np.var   (ens_matrix, axis=0)                           # (n_free,)
ens_med  = np.median(ens_matrix, axis=0)                           # (n_free,)
ens_mad  = np.median(np.abs(ens_matrix - ens_med[np.newaxis, :]),
                     axis=0)                                        # (n_free,)
ens_prc  = np.percentile(ens_matrix, PERCENTILES, axis=0)          # (n_prc, n_free)

print(f"\nStatistics (over {n_members} members, {ne[1]} free parameters):")
print(f"  mean   log10(ρ): [{ens_avg.min():.3f}, {ens_avg.max():.3f}]")
print(f"  var    log10(ρ): [{ens_var.min():.4f}, {ens_var.max():.4f}]")
print(f"  median log10(ρ): [{ens_med.min():.3f}, {ens_med.max():.3f}]")
print(f"  MAD    log10(ρ): [{ens_mad.min():.4f}, {ens_mad.max():.4f}]")

# --- (3) Empirical covariance ---------------------------------------------
print("\nComputing empirical covariance …")
ens_cov = sklearn.covariance.empirical_covariance(ens_matrix)

ens_covs = None
if SPARSIFY:
    tmp    = ens_cov.copy()
    tmp[np.abs(tmp) / np.amax(np.abs(tmp)) <= SPARSE_THRESH] = 0.0
    ens_covs = scs.csr_matrix(tmp)
    nnz      = ens_covs.nnz
    total    = ens_cov.size
    print(f"  Sparse covariance: {nnz}/{total} non-zeros "
          f"({100.0*nnz/total:.2f}%), threshold={SPARSE_THRESH:.1e}")

# --- (4) Save .npz --------------------------------------------------------
ens_dict = {
    f"{P}_model_list": model_list,
    f"{P}_ens":        ens_matrix,
    f"{P}_cov":        ens_cov,
    f"{P}_avg":        ens_avg,
    f"{P}_var":        ens_var,
    f"{P}_med":        ens_med,
    f"{P}_mad":        ens_mad,
    f"{P}_prc":        ens_prc,
}

np.savez_compressed(ENSEMBLE_RESULTS, **ens_dict)
print(f"\nResults saved → {ENSEMBLE_RESULTS}")

# --- (5) Resolve UTM origin and sites (needed for any plot) ---------------
_need_plot = PLOT_QC or PLOT_STATS
if _need_plot:
    (utm_e, utm_n, utm_lat, utm_lon,
     utm_zone, utm_north, site_xys) = _resolve_origin_and_sites()

# --- (6) QC slice plot — best-nRMS member ---------------------------------
if PLOT_QC:
    if fviz is None:
        print("\n  PLOT_QC: femtic_viz not available — skipping.")
    elif not model_list:
        print("\n  PLOT_QC: no converged members — skipping.")
    else:
        _best      = min(model_list, key=lambda x: x[2])
        _best_file, _best_iter, _best_nrms = _best
        print(f"\nQC: best member  nRMS={_best_nrms:.4f}  "
              f"iter={_best_iter}")
        _plot_slice(
            block_file = _best_file,
            pdf_file   = PLOT_QC_FILE,
            utm_e      = utm_e,
            utm_n      = utm_n,
            utm_lat    = utm_lat,
            utm_lon    = utm_lon,
            utm_zone   = utm_zone,
            utm_north  = utm_north,
            site_xys   = site_xys,
            dpi        = PLOT_QC_DPI,
        )

# --- (7) Statistics slice plots -------------------------------------------
if PLOT_STATS:
    if fviz is None:
        print("\n  PLOT_STATS: femtic_viz not available — skipping.")
    elif not model_list:
        print("\n  PLOT_STATS: no converged members — skipping.")
    else:
        os.makedirs(PLOT_STATS_DIR, exist_ok=True)

        # Template = lowest-nRMS member (preserves header / flag columns)
        _best_file = min(model_list, key=lambda x: x[2])[0]

        _stat_map = {
            "avg": (ens_avg, "mean"),
            "var": (ens_var, "variance"),
            "med": (ens_med, "median"),
            "mad": (ens_mad, "MAD"),
        }

        for _key in PLOT_STATS_WHAT:
            if _key not in _stat_map:
                print(f"  PLOT_STATS: unknown stat '{_key}' — skipped.")
                continue
            _vec, _label = _stat_map[_key]
            _block_out = os.path.join(
                PLOT_STATS_DIR,
                f"resistivity_block_{P}_{_key}.dat",
            )
            _pdf_out = os.path.join(
                PLOT_STATS_DIR,
                f"{P}_{_key}.pdf",
            )
            print(f"\nSTATS: writing {_label} → {_block_out}")
            fem.insert_model(
                template   = _best_file,
                model      = _vec,
                model_file = _block_out,
                ocean      = None,
                air_rho    = 1.0e9,
                ocean_rho  = PLOT_OCEAN_RHO,
                out        = OUT,
            )
            print(f"STATS: plotting {_label} → {_pdf_out}")
            _plot_slice(
                block_file = _block_out,
                pdf_file   = _pdf_out,
                utm_e      = utm_e,
                utm_n      = utm_n,
                utm_lat    = utm_lat,
                utm_lon    = utm_lon,
                utm_zone   = utm_zone,
                utm_north  = utm_north,
                site_xys   = site_xys,
                dpi        = PLOT_STATS_DPI,
            )

print("\nfemtic_ens_post.py complete.")
