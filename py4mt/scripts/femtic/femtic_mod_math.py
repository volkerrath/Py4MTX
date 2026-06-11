#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_mod_math.py — Generate synthetic ensemble members from an N-subset

For a specified subset of FEMTIC ensemble members (directories or explicit
block files), this script loads every converged member in log10(ρ) space,
assembles the N×M matrix, and writes two derived members:

    ``average``         element-wise arithmetic mean in log10 space
    ``smooth_median``   element-wise median, followed by mesh-adaptive
                        spatial smoothing (same kernels as femtic_mod_edit.py)

Both outputs are written as resistivity block files using the template of the
lowest-nRMS member, and optionally plotted as axis-parallel slice figures via
``femtic_viz.plot_model_slices``.

Subset selection (SUBSET_LIST / NRMS_MAX)
-----------------------------------------
Two independent filters are applied in sequence:

1.  **NRMS_MAX** — skip any directory whose ``femtic.cnv`` records a final
    nRMS above this threshold.  Set to ``np.inf`` to disable.
2.  **SUBSET_LIST** — after the nRMS filter, keep only the member indices
    listed here (0-based position in the sorted file list).  ``None`` = all
    converged members.  Explicit lists are the same format used by
    ``ENS_LIST`` in ``femtic_rto_prep.py``.

Input modes (ENSEMBLE_DIR + ENSEMBLE_NAME  vs.  BLOCK_FILES)
-------------------------------------------------------------
*   **Directory scan** (default): ``ENSEMBLE_DIR`` is searched recursively
    for sub-directories matching ``ENSEMBLE_NAME`` glob.  Each sub-directory
    must contain ``femtic.cnv`` and ``resistivity_block_iter<N>.dat``.
*   **Explicit file list** (``BLOCK_FILES`` is not ``None``): skip the
    directory scan entirely; use the supplied list of block-file paths.
    ``NRMS_MAX`` and ``SUBSET_LIST`` still apply (nRMS check is skipped for
    files without a sibling ``femtic.cnv``).

Smoothing (smooth_median)
--------------------------
The median model is smoothed in one pass using the same three kernel modes
exposed in ``femtic_mod_edit.py``:

    ``"physical"``      Global Gaussian, σ = SMOOTH_SIGMA metres.
    ``"knn_uniform"``   Flat average over K nearest neighbours.
    ``"knn_gauss"``     Per-region Gaussian, σ_i = SMOOTH_KNN_SIGMA_FRAC ×
                        distance to K-th neighbour.

All modes require MESH_FILE (tetrahedral mesh for centroid computation).

Outputs
-------
    <OUT_DIR>/resistivity_block_avg.dat
    <OUT_DIR>/resistivity_block_smooth_median.dat
    <OUT_DIR>/math_avg.pdf            (when PLOT = True)
    <OUT_DIR>/math_smooth_median.pdf  (when PLOT = True)

Provenance
----------
    2026-06-11  vrath / Claude Sonnet 4.6 (Anthropic)
                Created, modelled on femtic_rto_post.py (ensemble loading /
                nRMS filter) and femtic_mod_edit.py (smooth operation, plotting
                config, insert_model I/O).  New SUBSET_LIST / BLOCK_FILES
                input modes.

@author: vrath
"""
from __future__ import annotations

import os
import sys
import inspect
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Py4MTX path bootstrap
# ---------------------------------------------------------------------------
PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [PY4MTX_ROOT + "/py4mt/modules/", PY4MTX_ROOT + "/py4mt/scripts/"]
for _pth in mypath:
    if _pth not in sys.path:
        sys.path.insert(0, _pth)

from version import versionstrg
import util as utl
import femtic as fem
import ensembles as ens

try:
    import femtic_viz as fviz
except ImportError:
    fviz = None

version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())
titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng + "\n\n")

# ===========================================================================
# Configuration
# ===========================================================================

# ---------------------------------------------------------------------------
# Input — directory scan
# ---------------------------------------------------------------------------
#: Root directory that contains the ensemble sub-directories.
ENSEMBLE_DIR = r"/home/vrath/work/Ensembles/RTO/"

#: Glob pattern matched against sub-directory names inside ENSEMBLE_DIR.
#: Typical values: ``"rto_*"``, ``"gst_*"``, ``"member_*"``.
ENSEMBLE_NAME = "rto_*"

# ---------------------------------------------------------------------------
# Input — explicit file list (alternative to directory scan)
# ---------------------------------------------------------------------------
#: When not None, skip the directory scan and use these block files directly.
#: Provide absolute paths or paths relative to the working directory.
#: Example:
#:     BLOCK_FILES = [
#:         "/path/to/rto_001/resistivity_block_iter12.dat",
#:         "/path/to/rto_007/resistivity_block_iter15.dat",
#:     ]
BLOCK_FILES = None

# ---------------------------------------------------------------------------
# Mesh file (required for smooth_median)
# ---------------------------------------------------------------------------
MESH_FILE = ENSEMBLE_DIR + "templates/mesh.dat"

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
#: Directory for the two output block files and optional plots.
OUT_DIR = ENSEMBLE_DIR + "math_members/"

# ---------------------------------------------------------------------------
# Subset selection
# ---------------------------------------------------------------------------
#: Maximum nRMS accepted from femtic.cnv.  Members above this value are
#: skipped.  Set to np.inf to accept all converged runs.
NRMS_MAX = 1.4

#: Subset of member indices (0-based position in the sorted, nRMS-filtered
#: list) to include.  None = use all converged members.
#: Example:  SUBSET_LIST = [0, 2, 5, 7]
SUBSET_LIST = None

# ---------------------------------------------------------------------------
# Ocean / air handling (forwarded to fem.insert_model)
# ---------------------------------------------------------------------------
#: None → auto-infer from region-1 heuristic.  True / False → force.
OCEAN = None
AIR_RHO   = 1.0e9   # Ω·m written for region 0 (air)
OCEAN_RHO = 0.25    # Ω·m written for region 1 when ocean is active

# ---------------------------------------------------------------------------
# Smoothing kernel (applied to the median model)
# ---------------------------------------------------------------------------
#: Kernel mode — one of ``"physical"``, ``"knn_uniform"``, ``"knn_gauss"``.
SMOOTH_MODE = "physical"

#: Gaussian σ in metres — used by ``"physical"`` mode only.
SMOOTH_SIGMA = 3000.0

#: Number of nearest neighbours — used by all three modes.
SMOOTH_K = 100

#: Per-region σ fraction — used by ``"knn_gauss"`` mode only.
#: σ_i = SMOOTH_KNN_SIGMA_FRAC × d_{i,K}
SMOOTH_KNN_SIGMA_FRAC = 0.5

#: Maximum RAM (GiB) for the chunked dense fallback (physical mode, no SciPy).
SMOOTH_MAX_GB = 4.0

# ---------------------------------------------------------------------------
# Verbose output
# ---------------------------------------------------------------------------
OUT = True

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
#: Set True to produce slice figures of both output members.
PLOT = True

#: Figure DPI.
PLOT_DPI = 300

#: Matplotlib colormap.
PLOT_CMAP = "turbo_r"

#: Colour limits [log10(ρ_min), log10(ρ_max)] — None = auto.
PLOT_CLIM = [0.0, 4.0]

#: Flat colour for ocean cells.
PLOT_OCEAN_COLOR = "lightgrey"

#: Axes facecolor for air / background.  None = figure default.
PLOT_AIR_BGCOLOR = None

#: Slice list — same format as femtic_mod_edit.py / femtic_mod_plot_slice.py.
PLOT_SLICES = [
    dict(kind="map", z0=5000.0),
    dict(kind="map", z0=15000.0),
    dict(kind="ns",  x0=0.0),
    dict(kind="ew",  y0=0.0),
]

PLOT_XLIM = [-20000., 20000.]
PLOT_YLIM = [-20000., 20000.]
PLOT_ZLIM = [  -6000., 15000.]

DEPTH_KM         = True
HORIZ_KM         = True
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
ORIGIN_METHOD     = "box"   # None | "box" | "average"

DISPLAY_COORDS = "model"    # "model" | "utm" | "latlon"

# ---------------------------------------------------------------------------
# Site overlay
# ---------------------------------------------------------------------------
SITE_DAT    = ENSEMBLE_DIR + "templates/site.dat"
SITE_NAMES  = None

PLOT_SITES_MAPS   = True
PLOT_SITES_SLICES = False
PROJECTION_DIST   = 5000.   # m

SITE_MARKER        = dict(marker="v", color="black", ms=8, zorder=10, label=None)
SITE_MARKER_SLICES = None
MAP_MARKERS        = []


# ===========================================================================
# Helpers
# ===========================================================================

def _smooth(m: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Smooth the log10(ρ) vector *m* using the configured kernel.

    Implements the same three modes as the ``"smooth"`` operation in
    ``femtic_mod_edit.py``; see that script's docstring for full details.

    Parameters
    ----------
    m         : (n_free,) float  log10(ρ) free vector
    centroids : (n_free, 3) float  region centroids in model-local metres

    Returns
    -------
    (n_free,) float  smoothed log10(ρ)
    """
    K    = min(int(SMOOTH_K), len(m))
    mode = str(SMOOTH_MODE).strip().lower()
    n    = len(m)

    _valid = {"physical", "knn_uniform", "knn_gauss"}
    if mode not in _valid:
        raise ValueError(
            f"smooth: unknown SMOOTH_MODE={mode!r}. Choose one of: {sorted(_valid)}."
        )

    try:
        from scipy.spatial import cKDTree as _cKDTree
        tree = _cKDTree(centroids)
        dist, idx = tree.query(centroids, k=K, workers=-1)
        m_nbr = m[idx]                                  # (n, K)

        if mode == "physical":
            two_s2 = 2.0 * float(SMOOTH_SIGMA) ** 2
            W = np.exp(-dist ** 2 / two_s2)

        elif mode == "knn_uniform":
            W = np.ones((n, K), dtype=float)

        else:  # knn_gauss
            frac   = float(SMOOTH_KNN_SIGMA_FRAC)
            d_kmax = dist[:, -1]                        # (n,)  d to K-th nbr
            sigma_i = frac * d_kmax
            degenerate = sigma_i == 0.0
            n_degen = int(degenerate.sum())
            two_s2_i = 2.0 * np.where(degenerate, 1.0, sigma_i) ** 2
            W = np.exp(-dist ** 2 / two_s2_i[:, np.newaxis])
            if n_degen:
                W[degenerate, :] = 1.0
                print(f"  smooth (knn_gauss): {n_degen} degenerate region(s) "
                      f"fell back to uniform weights.")

        return np.einsum("ij,ij->i", W, m_nbr) / W.sum(axis=1)

    except ImportError:
        if mode != "physical":
            raise RuntimeError(
                f"smooth mode '{mode}' requires SciPy. "
                "Install SciPy or set SMOOTH_MODE='physical'."
            )

    # -- Chunked dense fallback (physical, no SciPy) --
    sigma   = float(SMOOTH_SIGMA)
    two_s2  = 2.0 * sigma * sigma
    max_bytes  = int(SMOOTH_MAX_GB * 1024 ** 3)
    chunk_size = max(1, max_bytes // (n * 8))
    chunk_size = min(chunk_size, n)

    ctr = centroids
    w_sum  = np.zeros(n, dtype=float)
    wm_sum = np.zeros(n, dtype=float)
    for start in range(0, n, chunk_size):
        end  = min(start + chunk_size, n)
        blk  = ctr[start:end]
        d2   = (np.sum(blk**2, axis=1, keepdims=True)
                + np.sum(ctr**2, axis=1)[np.newaxis, :]
                - 2.0 * (blk @ ctr.T))
        d2   = np.maximum(d2, 0.0)
        W    = np.exp(-d2 / two_s2)
        wm_sum[start:end] = W @ m
        w_sum [start:end] = W.sum(axis=1)
    return wm_sum / w_sum


def _resolve_origin_and_sites():
    """Estimate UTM origin from SITE_DAT; collect site model-local coordinates.

    Returns
    -------
    utm_e, utm_n, utm_lat, utm_lon : float | None
        UTM origin.
    utm_zone, utm_northern : str | None, bool | None
        UTM zone string and hemisphere flag.
    site_xys : list of (name, x_m, y_m, elev) tuples
    """
    _e, _n, _lat, _lon = UTM_ORIGIN_E, UTM_ORIGIN_N, UTM_ORIGIN_LAT, UTM_ORIGIN_LON
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
                float(_lats.mean()), float(_lons.mean()), override=UTM_ZONE_OVERRIDE)
            _lat, _lon = utl.utm_to_latlon_zn(_e, _n, _zone, _north)

    if _lat is not None and _lon is not None:
        _zone, _north = utl.utm_zone_from_latlon(_lat, _lon, override=UTM_ZONE_OVERRIDE)

    site_xys = []
    if SITE_DAT and os.path.isfile(SITE_DAT):
        for row in fem.read_site_dat(SITE_DAT, site_names=SITE_NAMES):
            sx, sy = fem.utm_to_model(row["easting"], row["northing"], _e, _n)
            site_xys.append((row["name"], sx, sy, float(row.get("elev", 0.0))))

    return _e, _n, _lat, _lon, _zone, _north, site_xys


def _plot_member(block_file: str, pdf_file: str,
                 utm_e, utm_n, utm_lat, utm_lon, utm_zone, utm_north,
                 site_xys: list) -> None:
    """Plot model slices for one block file using fviz.plot_model_slices."""
    if fviz is None:
        print("  PLOT: femtic_viz not available — skipping.")
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
        ocean_value        = OCEAN_RHO,
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
        dpi                = PLOT_DPI,
        out                = OUT,
    )
    if OUT:
        print(f"  saved → {pdf_file}")


# ===========================================================================
# Main
# ===========================================================================

os.makedirs(OUT_DIR, exist_ok=True)

# --- (1) Build candidate file list ----------------------------------------
if BLOCK_FILES is not None:
    # --- explicit file list mode ---
    candidate_files = [str(p) for p in BLOCK_FILES]
    print(f"Explicit BLOCK_FILES: {len(candidate_files)} file(s) provided.")
    model_list = []
    for bf in candidate_files:
        if not os.path.isfile(bf):
            print(f"  WARNING: {bf} not found — skipped.")
            continue
        cnv_file = str(Path(bf).parent / "femtic.cnv")
        if os.path.isfile(cnv_file):
            with open(cnv_file) as _fh:
                _lines = _fh.readlines()
            _info = _lines[-1].split()
            _nrms = float(_info[8])
            _nit  = int(_info[0])
            if _nrms > NRMS_MAX:
                print(f"  {bf}: nRMS={_nrms:.4f} > NRMS_MAX={NRMS_MAX} — skipped.")
                continue
        else:
            _nrms = np.nan
            _nit  = -1
            print(f"  {bf}: no femtic.cnv found — nRMS check skipped.")
        model_list.append(dict(file=bf, nit=_nit, nrms=_nrms))
else:
    # --- directory scan mode ---
    dir_list = utl.get_filelist(
        searchstr=[ENSEMBLE_NAME],
        searchpath=ENSEMBLE_DIR,
        fullpath=True,
    )
    print(f"Directory scan: {len(dir_list)} sub-dir(s) found matching '{ENSEMBLE_NAME}'.")

    model_list = []
    for d in dir_list:
        cnv_file = os.path.join(d, "femtic.cnv")
        if not os.path.isfile(cnv_file):
            print(f"  {d}: femtic.cnv not found — skipped.")
            continue
        with open(cnv_file) as _fh:
            _lines = _fh.readlines()
        _info  = _lines[-1].split()
        _nit   = int(_info[0])
        _nrms  = float(_info[8])
        if _nrms > NRMS_MAX:
            print(f"  {d}: nRMS={_nrms:.4f} > NRMS_MAX={NRMS_MAX} — skipped.")
            continue
        bf = os.path.join(d, f"resistivity_block_iter{_nit}.dat")
        if not os.path.isfile(bf):
            print(f"  {d}: {bf} not found — skipped.")
            continue
        model_list.append(dict(file=bf, nit=_nit, nrms=_nrms))
    print(f"Converged members passing nRMS filter: {len(model_list)}.")

# --- (2) Apply SUBSET_LIST filter -----------------------------------------
subset_idx = ens._resolve_fromto(SUBSET_LIST, len(model_list))
model_list = [model_list[i] for i in subset_idx if i < len(model_list)]
n_members  = len(model_list)

if n_members == 0:
    sys.exit("No members remain after SUBSET_LIST filter. Nothing to do.")

print(f"\nSubset: {n_members} member(s) selected.")
for k, m in enumerate(model_list):
    _nrms_str = f"{m['nrms']:.4f}" if not np.isnan(m['nrms']) else "n/a"
    print(f"  [{k:3d}] nRMS={_nrms_str}  iter={m['nit']}  {m['file']}")

# --- (3) Identify template (lowest-nRMS member) ---------------------------
finite_nrms = [(i, m["nrms"]) for i, m in enumerate(model_list) if not np.isnan(m["nrms"])]
if finite_nrms:
    best_idx = min(finite_nrms, key=lambda x: x[1])[0]
else:
    best_idx = 0   # fall back to first if none have a cnv
TEMPLATE_FILE = model_list[best_idx]["file"]
print(f"\nTemplate (lowest nRMS): {TEMPLATE_FILE}")

# --- (4) Load ensemble into matrix ----------------------------------------
print("\nLoading ensemble …")
ens_matrix = None
for k, m in enumerate(model_list):
    log_m = fem.read_model(
        model_file=m["file"],
        model_trans="log10",
        ocean=OCEAN,
        out=OUT,
    )
    if ens_matrix is None:
        ens_matrix = np.empty((n_members, log_m.size), dtype=float)
    ens_matrix[k] = log_m

n_free = ens_matrix.shape[1]
print(f"Ensemble matrix: {n_members} members × {n_free} free parameters.")

# --- (5) Compute average --------------------------------------------------
log_avg = np.mean(ens_matrix, axis=0)      # (n_free,)
print(f"\nAverage: log10(ρ) range [{log_avg.min():.3f}, {log_avg.max():.3f}]")

# --- (6) Compute median ---------------------------------------------------
log_med = np.median(ens_matrix, axis=0)    # (n_free,)
print(f"Median:  log10(ρ) range [{log_med.min():.3f}, {log_med.max():.3f}]")

# --- (7) Smooth the median ------------------------------------------------
print(f"\nBuilding mesh geometry for smooth_median (mode='{SMOOTH_MODE}') …")
if not os.path.isfile(MESH_FILE):
    sys.exit(f"MESH_FILE not found: {MESH_FILE}")

nodes, conn = fem.read_femtic_mesh(MESH_FILE)
print(f"  mesh: {nodes.shape[0]} nodes, {conn.shape[0]} elements")

_struct      = fem._read_resistivity_block_struct(
    TEMPLATE_FILE, model_trans="log10", ocean=OCEAN, out=False
)
elem_region  = _struct["elem_region"]
free_idx     = _struct["free_idx"]

region_ctr, region_vol = fem.build_region_geometry(nodes, conn, elem_region, free_idx)
print(f"  {len(free_idx)} free regions, total volume={region_vol.sum():.3e} m³")

log_smooth_med = _smooth(log_med, region_ctr)
print(f"Smooth median: log10(ρ) range [{log_smooth_med.min():.3f}, {log_smooth_med.max():.3f}]")

# --- (8) Write output block files -----------------------------------------
avg_file    = os.path.join(OUT_DIR, "resistivity_block_avg.dat")
smed_file   = os.path.join(OUT_DIR, "resistivity_block_smooth_median.dat")

print(f"\nWriting average        → {avg_file}")
fem.insert_model(
    template   = TEMPLATE_FILE,
    model      = log_avg,
    model_file = avg_file,
    ocean      = OCEAN,
    air_rho    = AIR_RHO,
    ocean_rho  = OCEAN_RHO,
    out        = OUT,
)

print(f"Writing smooth_median  → {smed_file}")
fem.insert_model(
    template   = TEMPLATE_FILE,
    model      = log_smooth_med,
    model_file = smed_file,
    ocean      = OCEAN,
    air_rho    = AIR_RHO,
    ocean_rho  = OCEAN_RHO,
    out        = OUT,
)

print("Done writing.")

# --- (9) Resolve UTM origin and site positions for plotting ---------------
if PLOT:
    (utm_e, utm_n, utm_lat, utm_lon,
     utm_zone, utm_north, site_xys) = _resolve_origin_and_sites()

# --- (10) Plot ------------------------------------------------------------
if PLOT:
    print("\nPlotting …")
    _plot_member(
        block_file  = avg_file,
        pdf_file    = os.path.join(OUT_DIR, "math_avg.pdf"),
        utm_e       = utm_e,
        utm_n       = utm_n,
        utm_lat     = utm_lat,
        utm_lon     = utm_lon,
        utm_zone    = utm_zone,
        utm_north   = utm_north,
        site_xys    = site_xys,
    )
    _plot_member(
        block_file  = smed_file,
        pdf_file    = os.path.join(OUT_DIR, "math_smooth_median.pdf"),
        utm_e       = utm_e,
        utm_n       = utm_n,
        utm_lat     = utm_lat,
        utm_lon     = utm_lon,
        utm_zone    = utm_zone,
        utm_north   = utm_north,
        site_xys    = site_xys,
    )

print("\nfemtic_mod_math.py complete.")
