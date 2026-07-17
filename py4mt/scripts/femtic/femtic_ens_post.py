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
            Added femtic_viz import; MESH_FILE / MOD_QC / MOD_QC_FILE /
            MOD_QC_SLICES / MOD_QC_* config vars; QC slice plot of
            best-nRMS member at end of main block (calls
            fviz.plot_model_slices).
2026-06-11  vrath / Claude Sonnet 4.6 (Anthropic)
            Renamed femtic_rto_post.py → femtic_ens_post.py for
            algorithm-agnostic use.  Fixed axis bug: mean/var/median/MAD
            were computed over axis=1 (free parameters) instead of axis=0
            (members) — statistics are now correct.  MOD_QC block
            replaced by full MOD_SLICES framework matching
            femtic_mod_edit.py / femtic_mod_math.py: UTM origin
            resolution, CRS-aware fem.resolve_slice_positions, site
            overlay via _resolve_origin_and_sites(); new MOD_STATS block
            plots mean/variance/median/MAD as individual slice figures.
            ENSEMBLE_PREFIX added for generic naming of output keys.
2026-07-07  vrath / Claude Sonnet 5 (Anthropic)
            Aligned the entire plotting config surface with
            femtic_gst_prep.py / femtic_rto_prep.py: MESH_FILE →
            MOD_MESH; UTM_ORIGIN_* → MOD_UTM_ORIGIN_*; UTM_ZONE_OVERRIDE →
            MOD_UTM_ZONE_OVERRIDE; ORIGIN_METHOD → MOD_ORIGIN_METHOD;
            DISPLAY_COORDS → MOD_DISPLAY_COORDS; SITE_DAT/SITE_NAMES →
            MOD_SITE_DAT/MOD_SITE_NAMES; MOD_SITES_MAPS/SLICES →
            MOD_PLOT_SITES_MAPS/SLICES; PROJECTION_DIST →
            MOD_PROJECTION_DIST; SITE_MARKER(_SLICES) →
            MOD_SITE_MARKER(_SLICES); MAP_MARKERS → MOD_MAP_MARKERS;
            DEPTH_KM/HORIZ_KM → MOD_DEPTH_KM/MOD_HORIZ_KM.  Added
            MOD_OCEAN/MOD_AIR_RHO, MOD_SITE_NUMBER (observe.dat fallback,
            same as femtic_gst_prep.py), MOD_AIR_COLOR, MOD_ALPHA_FILE/
            MODE/BLANK_THRESH, MOD_PANEL_WIDTH, MOD_FIGSIZE.  Removed a
            latent duplicate MOD_XLIM/YLIM/ZLIM assignment that silently
            discarded the first (non-None) values.  _resolve_origin_and_
            sites() and _plot_slice() now match femtic_gst_prep.py's
            origin-resolution and plot_model_slices() call byte-for-byte
            in option coverage, so QC and statistics figures render
            identically to the ensemble-generation scripts given the
            same MOD_* settings.
2026-07-09  vrath / Claude Sonnet 5 (Anthropic)
            Merged MOD_QC_DPI / MOD_STATS_DPI into a single MOD_DPI knob
            (matching femtic_gst_prep.py / femtic_nss.py — one figure-DPI
            setting per script, not one per plot type).  _plot_slice() no
            longer takes a dpi argument; it reads MOD_DPI directly.
2026-07-17  Claude Sonnet 5 (Anthropic)
            scipy.sparse: migrated from legacy matrix to array-equivalent
            API — scs.csr_matrix(tmp) → scs.csr_array(tmp) when building
            the sparsified empirical covariance (ens_covs). No functional
            change; ens_covs is only used for its .nnz count.
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
FEMTIC="4.3"
# ---------------------------------------------------------------------------
# Ensemble input
# ---------------------------------------------------------------------------

# ENSEMBLE_DIR = r"/home/vrath/Py4MTX/py4mt/data/ensembles/misti/ensemble/"
ENSEMBLE_DIR = r"/media/vrath/LargeBack/misti/ensemble/"
ENSEMBLE_NAME = "misti_gst_suzuki_"
#: Prefix used for .npz output keys and default file/figure names.
#: e.g. "rto" → keys rto_ens, rto_avg, …  and file RTO_results.npz.
ENSEMBLE_PREFIX = "Misti_gstat"

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
MOD_MESH = ENSEMBLE_DIR + "templates/mesh.dat"

# --- Ocean / air handling (must match the inversion setup) ----------------
MOD_OCEAN     = None
MOD_AIR_RHO   = 1.0e9   # Ω·m  (region 0)
MOD_OCEAN_RHO = 0.25    # Ω·m  (region 1 when treated as ocean)

# ---------------------------------------------------------------------------
# QC slice plot — best-nRMS converged member
# ---------------------------------------------------------------------------
#: Set True to plot the best-nRMS member.
MOD_QC      = False
MOD_QC_FILE = ENSEMBLE_DIR + ENSEMBLE_PREFIX + "_qc.pdf"

# ---------------------------------------------------------------------------
# Statistics slice plots — mean / variance / median / MAD
# ---------------------------------------------------------------------------
#: Set True to write derived stat members as block files and plot them.
#: Requires MOD_MESH and a valid template file (taken from best member).
MOD_STATS      = False
#: Which statistics to plot.  Subset of: "avg", "var", "med", "mad".
MOD_STATS_WHAT = ["avg", "var", "med", "mad"]
#: Output directory for stat block files and figures.
MOD_STATS_DIR  = ENSEMBLE_DIR + "/stats_plots/"

# ---------------------------------------------------------------------------
# Shared slice / plot parameters
# (identical config surface to femtic_gst_prep.py / femtic_rto_prep.py /
#  femtic_mod_plot_slice.py — used by both MOD_QC and MOD_STATS below)
# ---------------------------------------------------------------------------

# --- Geographic / UTM origin of the mesh centre ----------------------------
#: Set to None when MOD_ORIGIN_METHOD will estimate the origin from MOD_SITE_DAT.
MOD_UTM_ORIGIN_LAT    = None   # decimal degrees, positive = North
MOD_UTM_ORIGIN_LON    = None   # decimal degrees, positive = East
MOD_UTM_ORIGIN_E      = None   # UTM easting  [m]
MOD_UTM_ORIGIN_N      = None   # UTM northing [m]
MOD_UTM_ZONE_OVERRIDE = None   # override auto-derived zone; None = auto

#: "box"     → midpoint of UTM bounding box of all sites in MOD_SITE_DAT.
#: "average" → arithmetic mean of UTM coordinates in MOD_SITE_DAT.
#: None      → use the hard-coded literals above.
MOD_ORIGIN_METHOD = "box"

# --- Display coordinate system ---------------------------------------------
#: "model"  — axis ticks in model-local metres (default)
#: "utm"    — axis ticks in absolute UTM metres
#: "latlon" — axis ticks in decimal degrees
MOD_DISPLAY_COORDS = "model"

# --- Site overlay ------------------------------------------------------------
#: Primary source: mt_make_sitelist.py CSV (name,lat,lon,elev,sitenum,E,N).
#: Set to None to fall back to observe.dat / MOD_SITE_NUMBER.
MOD_SITE_DAT    = ENSEMBLE_DIR + "templates/site.dat"
MOD_SITE_NAMES  = None   # list of names to plot, or None = all sites
#: Fallback: site number(s) from observe.dat (int or list of ints).
MOD_SITE_NUMBER = None

MOD_PLOT_SITES_MAPS   = True    # show markers on map panels
MOD_PLOT_SITES_SLICES = False   # show markers on curtain / plane panels
#: Max distance [m] from a curtain plane for a site to appear on it.
MOD_PROJECTION_DIST = 5000.0    # metres; None = show all sites on every panel

MOD_SITE_MARKER        = dict(marker="v", color="black", ms=8, zorder=10, label=None)
MOD_SITE_MARKER_SLICES = None
#: Extra point markers on map panels only (each dict: latlon, marker, color, ms, name).
MOD_MAP_MARKERS = []

# --- Slice specification ----------------------------------------------------
#: Slice positions accept plain floats (model-local m) or CRS-tagged tuples:
#:   (value, "utm") | (value, "latlon")
#: Depth z0 is always model-local metres (no CRS tagging).
MOD_SLICES = [
    dict(kind="map", z0=5000.0),
    dict(kind="map", z0=15000.0),
    dict(kind="ns",  x0=0.0),
    dict(kind="ew",  y0=0.0),
]
MOD_XLIM = None    # [xmin, xmax] model-local metres; None = auto
MOD_YLIM = None    # [ymin, ymax] model-local metres; None = auto
MOD_ZLIM = None    # [zmin, zmax] model-local metres; None = auto

MOD_DPI         = 600            # figure DPI, used by both MOD_QC and MOD_STATS
MOD_CMAP        = "turbo_r"
MOD_CLIM        = [0.0, 4.0]     # [log10_min, log10_max] Ω·m; None = auto
MOD_OCEAN_COLOR = "lightgrey"    # flat colour for ocean cells; None = colormap
MOD_AIR_COLOR   = "whitesmoke"
MOD_AIR_BGCOLOR = None           # axes facecolor for air; None = figure default

# --- Alpha / blanking by second block file (optional) -----------------------
MOD_ALPHA_FILE         = None    # path to sensitivity block; None = disabled
MOD_ALPHA_MODE         = "fade"  # "fade" | "blank"
MOD_ALPHA_BLANK_THRESH = 0.0

# --- Figure layout -----------------------------------------------------------
MOD_EQUAL_ASPECT = True
MOD_DEPTH_KM     = True
MOD_HORIZ_KM     = True
MOD_NROWS        = None   # None = auto (1 row)
MOD_NCOLS        = None   # None = auto (len(MOD_SLICES) cols)
MOD_PANEL_HEIGHT = 16.0   # cm
MOD_PANEL_WIDTH  = None   # cm; None = auto from aspect ratio
MOD_FIGSIZE      = None   # [w, h] cm; overrides auto when set

# ---------------------------------------------------------------------------
# Verbose output
# ---------------------------------------------------------------------------
OUT = True

# ===========================================================================
# Helpers
# ===========================================================================

def _resolve_origin_and_sites():
    """Estimate UTM origin from MOD_SITE_DAT; collect site model-local coords.

    Mirrors the origin-resolution block in femtic_gst_prep.py /
    femtic_rto_prep.py so all three scripts behave identically, including
    the observe.dat / MOD_SITE_NUMBER fallback when MOD_SITE_DAT is absent.

    Returns
    -------
    utm_e, utm_n, utm_lat, utm_lon : float | None
    utm_zone, utm_northern : str | None, bool | None
    site_xys : list of (name, x_m, y_m, elev)
    obs_coords_only : bool
        True if site_xys was populated from observe.dat / MOD_SITE_NUMBER
        rather than MOD_SITE_DAT.
    """
    _e   = MOD_UTM_ORIGIN_E
    _n   = MOD_UTM_ORIGIN_N
    _lat = MOD_UTM_ORIGIN_LAT
    _lon = MOD_UTM_ORIGIN_LON
    _zone, _north = None, None

    if MOD_ORIGIN_METHOD is not None and MOD_SITE_DAT and os.path.isfile(MOD_SITE_DAT):
        _sdat = fem.read_site_dat(MOD_SITE_DAT)
        if _sdat:
            _Es  = np.array([d["easting"]  for d in _sdat])
            _Ns  = np.array([d["northing"] for d in _sdat])
            if MOD_ORIGIN_METHOD == "box":
                _e = 0.5 * (_Es.min() + _Es.max())
                _n = 0.5 * (_Ns.min() + _Ns.max())
            elif MOD_ORIGIN_METHOD == "average":
                _e = float(_Es.mean())
                _n = float(_Ns.mean())
            _lats = np.array([d["lat"] for d in _sdat])
            _lons = np.array([d["lon"] for d in _sdat])
            _zone, _north = utl.utm_zone_from_latlon(
                float(_lats.mean()), float(_lons.mean()),
                override=MOD_UTM_ZONE_OVERRIDE,
            )
            _lat, _lon = utl.utm_to_latlon_zn(_e, _n, _zone, _north)

    if _lat is not None and _lon is not None:
        _zone, _north = utl.utm_zone_from_latlon(
            _lat, _lon, override=MOD_UTM_ZONE_OVERRIDE
        )

    site_xys = []
    obs_coords_only = False
    _need_sites = MOD_PLOT_SITES_MAPS or MOD_PLOT_SITES_SLICES
    if _need_sites and MOD_SITE_DAT and os.path.isfile(MOD_SITE_DAT):
        for row in fem.read_site_dat(MOD_SITE_DAT, site_names=MOD_SITE_NAMES):
            sx, sy = fem.utm_to_model(
                row["easting"], row["northing"], _e, _n
            )
            site_xys.append(
                (row["name"], sx, sy, float(row.get("elev", 0.0)))
            )
    elif _need_sites and MOD_SITE_NUMBER is not None:
        _obs_file = ENSEMBLE_DIR + "templates/observe.dat"
        _site_nums = (MOD_SITE_NUMBER if isinstance(MOD_SITE_NUMBER, (list, tuple))
                      else [MOD_SITE_NUMBER])
        for _sn in _site_nums:
            sx, sy = fem.read_site_position(_obs_file, _sn)
            site_xys.append((_sn, sx, sy, 0.0))
        obs_coords_only = True

    return _e, _n, _lat, _lon, _zone, _north, site_xys, obs_coords_only


def _plot_slice(block_file: str, pdf_file: str,
                utm_e, utm_n, utm_lat, utm_lon,
                utm_zone, utm_north, site_xys: list,
                obs_coords_only: bool = False) -> None:
    """Call fviz.plot_model_slices with the shared MOD_* config.

    Mirrors the plotting call in femtic_gst_prep.py / femtic_rto_prep.py
    exactly, so QC and statistics figures use the same options (CRS
    handling, site overlay, alpha/blanking, figure layout) as the
    ensemble-generation scripts.
    """
    if fviz is None:
        print("  plot_slice: femtic_viz not available — skipping.")
        return

    _slices_resolved = fem.resolve_slice_positions(
        MOD_SLICES, utm_zone, utm_north,
        utm_e, utm_n, utm_lat, utm_lon,
        verbose=OUT,
    )
    fviz.plot_model_slices(
        model_file          = block_file,
        mesh_file           = MOD_MESH,
        slices              = _slices_resolved,
        cmap                = MOD_CMAP,
        clim                = MOD_CLIM,
        xlim                = MOD_XLIM,
        ylim                = MOD_YLIM,
        zlim                = MOD_ZLIM,
        ocean_color         = MOD_OCEAN_COLOR,
        ocean_value         = MOD_OCEAN_RHO,
        air_color           = MOD_AIR_COLOR,
        air_bgcolor         = MOD_AIR_BGCOLOR,
        site_xys            = site_xys,
        obs_coords_only     = obs_coords_only,
        sites_in_maps       = MOD_PLOT_SITES_MAPS,
        sites_in_slices     = MOD_PLOT_SITES_SLICES,
        site_marker         = MOD_SITE_MARKER,
        site_marker_slices  = MOD_SITE_MARKER_SLICES,
        map_markers         = MOD_MAP_MARKERS,
        projection_dist     = MOD_PROJECTION_DIST,
        display_coords      = MOD_DISPLAY_COORDS,
        utm_origin_e        = utm_e,
        utm_origin_n        = utm_n,
        utm_zone            = utm_zone,
        utm_northern        = utm_north,
        utm_to_latlon_fn    = utl.utm_to_latlon_zn,
        latlon_to_model_fn  = fem.latlon_to_model,
        depth_km            = MOD_DEPTH_KM,
        horiz_km            = MOD_HORIZ_KM,
        equal_aspect        = MOD_EQUAL_ASPECT,
        panel_height        = MOD_PANEL_HEIGHT / 2.54,
        panel_width         = MOD_PANEL_WIDTH / 2.54 if MOD_PANEL_WIDTH is not None else None,
        figsize             = [v / 2.54 for v in MOD_FIGSIZE] if MOD_FIGSIZE is not None else None,
        nrows               = MOD_NROWS,
        ncols               = MOD_NCOLS,
        alpha_file          = MOD_ALPHA_FILE,
        alpha_mode          = MOD_ALPHA_MODE,
        alpha_blank_thresh  = MOD_ALPHA_BLANK_THRESH,
        plot_file           = pdf_file,
        dpi                 = MOD_DPI,
        out                 = OUT,
    )
    if OUT:
        print(f"  saved → {pdf_file}")


# ===========================================================================
# Main
# ===========================================================================

# --- (1) Scan ensemble directories ----------------------------------------
dir_list = utl.get_filelist(
    searchstr=[ENSEMBLE_NAME+"*"],
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
    if "4.3" in FEMTIC:
        numit = int(info[0])
        nrms  = float(info[6])
    elif "5." in FEMTIC:
        numit = int(info[0])
        nrms  = float(info[8])
    else:
        sys.exit("FEMTIC version"+__file__+": does not exist! Exit.")

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
    ens_covs = scs.csr_array(tmp)
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
_need_plot = MOD_QC or MOD_STATS
if _need_plot:
    (utm_e, utm_n, utm_lat, utm_lon,
     utm_zone, utm_north, site_xys, obs_coords_only) = _resolve_origin_and_sites()

# --- (6) QC slice plot — best-nRMS member ---------------------------------
if MOD_QC:
    if fviz is None:
        print("\n  MOD_QC: femtic_viz not available — skipping.")
    elif not model_list:
        print("\n  MOD_QC: no converged members — skipping.")
    else:
        _best      = min(model_list, key=lambda x: x[2])
        _best_file, _best_iter, _best_nrms = _best
        print(f"\nQC: best member  nRMS={_best_nrms:.4f}  "
              f"iter={_best_iter}")
        _plot_slice(
            block_file      = _best_file,
            pdf_file        = MOD_QC_FILE,
            utm_e           = utm_e,
            utm_n           = utm_n,
            utm_lat         = utm_lat,
            utm_lon         = utm_lon,
            utm_zone        = utm_zone,
            utm_north       = utm_north,
            site_xys        = site_xys,
            obs_coords_only = obs_coords_only,
        )

# --- (7) Statistics slice plots -------------------------------------------
if MOD_STATS:
    if fviz is None:
        print("\n  MOD_STATS: femtic_viz not available — skipping.")
    elif not model_list:
        print("\n  MOD_STATS: no converged members — skipping.")
    else:
        os.makedirs(MOD_STATS_DIR, exist_ok=True)

        # Template = lowest-nRMS member (preserves header / flag columns)
        _best_file = min(model_list, key=lambda x: x[2])[0]

        _stat_map = {
            "avg": (ens_avg, "mean"),
            "var": (ens_var, "variance"),
            "med": (ens_med, "median"),
            "mad": (ens_mad, "MAD"),
        }

        for _key in MOD_STATS_WHAT:
            if _key not in _stat_map:
                print(f"  MOD_STATS: unknown stat '{_key}' — skipped.")
                continue
            _vec, _label = _stat_map[_key]
            _block_out = os.path.join(
                MOD_STATS_DIR,
                f"resistivity_block_{P}_{_key}.dat",
            )
            _pdf_out = os.path.join(
                MOD_STATS_DIR,
                f"{P}_{_key}.pdf",
            )
            print(f"\nSTATS: writing {_label} → {_block_out}")
            fem.insert_model(
                template   = _best_file,
                model      = _vec,
                model_file = _block_out,
                ocean      = MOD_OCEAN,
                air_rho    = MOD_AIR_RHO,
                ocean_rho  = MOD_OCEAN_RHO,
                out        = OUT,
            )
            print(f"STATS: plotting {_label} → {_pdf_out}")
            _plot_slice(
                block_file      = _block_out,
                pdf_file        = _pdf_out,
                utm_e           = utm_e,
                utm_n           = utm_n,
                utm_lat         = utm_lat,
                utm_lon         = utm_lon,
                utm_zone        = utm_zone,
                utm_north       = utm_north,
                site_xys        = site_xys,
                obs_coords_only = obs_coords_only,
            )

print("\nfemtic_ens_post.py complete.")
