#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cluster.py
=================
Clusters MT resistivity/conductivity + seismic tomography properties
(Vp, Vs, Vp/Vs, density) that have already been interpolated onto a
common grid by interpolate.py, via fuzzy c-means or a
self-organizing map (SOM), and plots depth-slice maps.

This is the clustering-only half of what used to be
cluster_rbf.py / cluster_kriging.py / cluster_som.py:
those three scripts each duplicated grid-building + interpolation +
clustering + plotting, differing only in interpolation method (RBF vs.
kriging) or clustering method (fuzzy c-means vs. SOM). This script does
NOT build any grid or interpolate anything — it just reads whichever
grid interpolate.py already produced (INTERP_FILE), and is
therefore completely agnostic to how that grid was built (a fresh joint
UTM grid or a reused seismic-tomography native grid) or which
interpolation method produced it (RBF/kriging/IDW). CLUSTERING_METHOD
picks fuzzy c-means or SOM independently of any of that.

Pipeline
--------
precompute.py → interpolate.py → {SITE_PREFIX}_interp_<method>.nc
                                                     ↓
                                    cluster.py → {SITE_PREFIX}_clusters.nc + figures

What this script does
----------------------
1. Reads INTERP_FILE (interpolate.py's output) — grid coords/dims,
   per-variable interpolated fields and their units, and enough
   attributes (target_grid_mode, interp_vars) to know how to plot it.
2. Picks CLUSTER_VARS, a subset of whatever variables are actually in
   INTERP_FILE (defaults to all of them) — so you can interpolate a
   superset once and cluster on different subsets without re-running
   interpolate.py.
3. Builds one feature table, drops any grid cell with a NaN in a
   selected variable, optionally standardizes (z-score) each feature,
   then weights (CLUSTER_WEIGHTS).
4. Clusters via CLUSTERING_METHOD:
   - "fcm" — a self-contained (NumPy-only) fuzzy c-means (Bezdek, 1981),
     N_CLUSTERS discrete classes, reporting the fuzzy partition
     coefficient (FPC) as a quick quality check.
   - "som" — a self-contained (NumPy-only) Kohonen self-organizing map,
     SOM_ROWS x SOM_COLS neurons, every point labelled with its
     best-matching unit (BMU) over the FULL neuron grid (not collapsed
     to N clusters), colored with a topological colormap so visually
     similar map colors reflect genuinely similar feature-space
     neighbors.
5. Reconstructs the hard label + membership/quantization-error back onto
   the grid and saves {SITE_PREFIX}_clusters.nc / {SITE_PREFIX}_cluster_centers.csv.
6. Plots horizontal cluster maps at PLOT_DEPTHS_KM, on the same
   topography/bathymetry basemap as plot_seis.py. If INTERP_FILE
   was built on a regular UTM-km ("joint") grid this uses a plain
   imshow(extent=...); if it was built on a reused seismic-tomography
   ("seismic") grid — not regular in UTM space — this uses
   pcolormesh(shading="nearest") against that grid's own 2-D
   utm_easting_km/utm_northing_km coordinates instead.

Dependencies
------------
    numpy, xarray, pandas, matplotlib
plus the local `plotpy.py` helper module (also used by the plot
scripts). The fuzzy c-means / SOM implementations are self-contained
(NumPy only, no scikit-fuzzy/MiniSom).

Authors: Svetlana Byrdina (SMB) & Volker Rath (DIAS)
AI-assisted development: Claude (Anthropic). Split out of
cluster_rbf.py / cluster_kriging.py / cluster_som.py
into a single, interpolation-method-agnostic clustering script that
reads interpolate.py's output instead of building any grid or
interpolating itself: Claude (Anthropic), 2026-08-02.
License: GNU General Public License v3 (GPL-3.0-or-later).
AI-generated code — review before use in production.
"""

import csv
import os
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import plotpy

to_utm_km = plotpy.to_utm_km
compute_hillshade = plotpy.compute_hillshade
clipped_markers = plotpy.clipped_markers
clipped_labels = plotpy.clipped_labels
draw_north_arrow = plotpy.draw_north_arrow

# =====================================================================
# USER SETTINGS
# =====================================================================

# --- Input/output directories ---
# --- Site selector ---
# Must match SITE_PREFIX in precompute.py / interpolate.py — used below
# to find their outputs and to name this script's own outputs.
SITE_PREFIX = "saba"
# SITE_PREFIX = "tacna"  # TACNA

# --- Input/output directories ---
NC_DIR = "../precompute/"   # must match interpolate.py's NC_DIR;
                             # {SITE_PREFIX}_clusters.nc / {SITE_PREFIX}_cluster_centers.csv
                             # are also written here.
PLOT_DIR = "../plots_cluster/"
PLOT_FORMATS = [".pdf", ".jpg"]
PLOT_DPI = 600

# --- Which interpolate.py output to read ---
# Bare filename, looked up under NC_DIR — e.g. "{SITE_PREFIX}_interp_rbf.nc",
# "{SITE_PREFIX}_interp_kriging.nc", "{SITE_PREFIX}_interp_idw.nc", or
# whatever interpolate.py's OUTPUT_FILE was set to.
INTERP_FILE = f"{SITE_PREFIX}_interp_rbf.nc"

# --- Which variables to actually cluster on ---
# None = use every variable in INTERP_FILE (its own interp_vars
# attribute). Set an explicit subset to cluster on fewer than what was
# interpolated — must all be present in INTERP_FILE, or this raises
# telling you to add them to interpolate.py's INTERP_VARS and
# re-run it first.
CLUSTER_VARS = None

# --- Per-variable clustering weights ---
# Multiplies each (already standardized, if STANDARDIZE) feature by
# sqrt(weight) before clustering, so a larger weight makes that variable
# count more toward cluster assignment — equivalent to a weighted
# Euclidean distance d^2 = sum_j weight_j * (x_j - c_j)^2. Any
# CLUSTER_VARS entry not listed here falls back to 1.0 (no effect).
CLUSTER_WEIGHTS = {}

# --- Standardize (z-score) each variable before clustering ---
# Strongly recommended: rho/cond/vps/dens have very different numeric
# ranges and units, and Euclidean distance would otherwise be dominated
# by whichever variable happens to have the largest raw range.
STANDARDIZE = True

# --- Clustering method ---
# "fcm" = fuzzy c-means (Bezdek, 1981), N_CLUSTERS discrete classes.
# "som" = self-organizing map (Kohonen, 1982): trains a SOM_ROWS x
#   SOM_COLS grid of neurons and labels every point with its
#   best-matching unit (BMU) -- i.e. the FULL neuron grid is kept as the
#   output classification (SOM_ROWS*SOM_COLS classes), not collapsed
#   down to N_CLUSTERS. N_CLUSTERS/FUZZINESS/MAX_ITER/TOL below are only
#   used when CLUSTERING_METHOD == "fcm"; SOM_* settings are only used
#   when CLUSTERING_METHOD == "som".
CLUSTERING_METHOD = "som"

# --- Fuzzy c-means settings (CLUSTERING_METHOD == "fcm") ---
N_CLUSTERS = 3
FUZZINESS = 2.0      # "m" in the FCM literature; > 1, 2.0 is conventional
MAX_ITER = 300
TOL = 1e-5            # stop once max membership change between iterations < TOL
RANDOM_SEED = 42

# --- Self-organizing map settings (CLUSTERING_METHOD == "som") ---
SOM_ROWS = 5
SOM_COLS = 5
SOM_N_ITER = 12000     # online (one random sample per iteration) updates,
                       # NOT epochs over the full dataset
SOM_LR0 = 0.5          # initial learning rate; decays exponentially to
                       # 1% of this by the final iteration
SOM_SIGMA0 = None     # initial neighborhood radius, grid units; None =
                       # max(SOM_ROWS, SOM_COLS) / 2. Decays exponentially
                       # to a floor of 0.5.
SOM_SEED = 42

# --- Plotting ---
PLOT_DEPTHS_KM = [1.0, 5.0, 9.0]   # nearest available depth level is used for each
CLUSTER_CMAP = "tab10"             # qualitative colormap, used only for
                                    # CLUSTERING_METHOD == "fcm" (SOM uses
                                    # a topological colormap instead, see
                                    # som_grid_colormap())
CLUSTER_ALPHA = 0.80

SHOW_TOPO_BASEMAP = True
HS_AZIMUTH, HS_ALTITUDE, HS_SIGMA = 315, 45, 1.0
TOPO_VMIN, TOPO_VMAX = 1000, 6000
OCEAN_COLOR = "#6baed6"

MAP_XLIM = None    # e.g. [310.0, 455.]  (easting,  km); None = auto from the grid
MAP_YLIM = None    # e.g. [7971.6, 8125] (northing, km); None = auto from the grid
REGION_MARGIN_KM = 0.0
FIG_WIDTH = 10.0    # cm — map panel width; height is derived, equal-scale by construction

AXES_UNITS = "km"          # "km" | "latlon"
AXES_KM_COMMA = True
LATLON_NTICKS = 5
LATLON_DECIMALS = 2

SHOW_COLORBAR = True
COLORBAR_POSITION = "right"
COLORBAR_SIZE = 0.85
COLORBAR_ASPECT = 20
COLORBAR_PAD = 0.10
COLORBAR_LABEL_SIZE = 10
COLORBAR_TICK_SIZE = 9

SHOW_NORTH_ARROW = True
ARROW_LON, ARROW_LAT, ARROW_LEN_KM = -73.6, -18.1, 4.0
ARROW_STYLE = dict(color="dimgray", lw=2, mutation_scale=14)
ARROW_LABEL_STYLE = dict(fontsize=9, fontweight="bold", color="dimgray")

AXIS_LABEL_SIZE = 12
AXIS_TICK_SIZE = 12
AXIS_TITLE_SIZE = 12

ANNOTATION_TEXT = None
ANNOTATION_POS = (0.01, 0.99)
ANNOTATION_STYLE = dict(fontsize=7, color="gray", ha="left", va="top")

# =====================================================================
# SPECIFIC PLOT SETTINGS — borrowed from plot_seis.py
# =====================================================================
SHOW_SPECIFIC_PLOT = True

CSV_VOLCANES      = "../features/volcanes.csv"
CSV_SEISMCAT      = "../features/catalog_welllocated_15_simple5.csv"
CSV_MT_SITES      = "../features/done/MTsaba_Sitelist.csv"
# CSV_MT_SITES      = "../features/done/MTTacna_Sitelist.csv"  # TACNA
CSV_CITIES        = "../features/cities.csv"
CSV_SEISMIC_SITES = "../features/seismic_sites.csv"  # no header row; columns
                                                       # are network, station,
                                                       # lat, lon, elev_m

SHOW_SEISMICITY       = True
SHOW_MT_SITES         = True
SHOW_SEISMIC_SITES    = True
SHOW_VOLCANOES        = True   # inactive volcano markers + labels
SHOW_VOLCANOES_ACTIVE = True   # active volcano markers
SHOW_CITIES           = True

ZMIN_SEISM = [-7, 1, 9]
ZMAX_SEISM = [1, 9, 30]

VOLC_LABEL_IDX = [5, 12, 13]
# Volcano name column (volcanes.csv). Labels are truncated to their
# first VOLC_LABEL_CHARS characters via VOLC_LABEL_STYLE's mode="firstN"
# (see plotpy.apply_label_mode) rather than reading a separate
# already-abbreviated column (e.g. "VOLCAN2") — one source of truth for
# the name, with truncation as a display-only concern.
VOLC_NAME_COL = "NAME"

EQ_MARKER_STYLE = dict(
    marker="o", s=4.5, facecolors="white", edgecolors="black",
    linewidths=0.2, zorder=11,
)
MT_MARKER_STYLE = dict(
    marker="v", s=10, facecolors="yellow", edgecolors="black",
    linewidths=0.7, zorder=12,
)
MT_LABEL_STYLE = dict(
    fontsize=5, color="black", zorder=14, rotation=90,
    offset_x=0.3, offset_y=0.3, mode="none",
)
SEISMIC_SITES_MARKER_STYLE = dict(
    marker="v", s=10, facecolors="green", edgecolors="black",
    linewidths=0.7, zorder=12,
)
VOLC_INACT_MARKER_STYLE = dict(
    marker="^", s=10, facecolors="blue", edgecolors="black",
    linewidths=0.7, zorder=13,
)
VOLC_LABEL_CHARS = 4   # number of leading characters shown (mode="firstN"
                        # below) — was previously a separate "VOLCAN2"
                        # abbreviated-code column; adjust to taste.
VOLC_LABEL_STYLE = dict(
    fontsize=6, fontweight="bold", color="black", zorder=14,
    offset_x=0.3, offset_y=0.3, mode=f"first{VOLC_LABEL_CHARS}",
)
VOLC_ACT_MARKER_STYLE = dict(
    marker="^", s=10, facecolors="red", edgecolors="black",
    linewidths=0.7, zorder=13,
)
CITY_MARKER_STYLE = dict(
    marker="s", s=6, facecolors="black", edgecolors="black",
    linewidths=0.2, zorder=13,
)
CITY_LABEL_STYLE = dict(
    fontsize=6, color="white", zorder=14,
    offset_x=0.3, offset_y=-0.3, mode="full",
)

if SHOW_SPECIFIC_PLOT and SHOW_SEISMICITY:
    if not (len(ZMIN_SEISM) == len(ZMAX_SEISM) == len(PLOT_DEPTHS_KM)):
        raise SystemExit(
            f"ZMIN_SEISM ({len(ZMIN_SEISM)}), ZMAX_SEISM ({len(ZMAX_SEISM)}), "
            f"and PLOT_DEPTHS_KM ({len(PLOT_DEPTHS_KM)}) must all be the same "
            f"length — one seismicity depth-window pair per depth slice. Pad "
            f"the shorter list(s) with None (= show all seismicity) for any "
            f"slice that doesn't need a filter."
        )

# =====================================================================
# END USER SETTINGS
# =====================================================================

Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)


def ncpath(name):
    """Join a bare NetCDF filename onto NC_DIR."""
    return os.path.join(NC_DIR, name)


def safe_to_netcdf(obj, path):
    """
    Write a Dataset/DataArray to NetCDF, overwriting any existing file at
    `path` even if it's read-only (see interpolate.py's own copy of
    this helper for the full rationale).
    """
    p = Path(path)
    if p.exists():
        try:
            p.unlink()
        except PermissionError:
            os.chmod(p, 0o644)
            p.unlink()
    obj.to_netcdf(path)


def safe_open_w(path, **kwargs):
    """Like open(path, 'w', ...), but first clears a read-only leftover
    file at `path` (same PermissionError issue as safe_to_netcdf)."""
    p = Path(path)
    if p.exists():
        try:
            p.unlink()
        except PermissionError:
            os.chmod(p, 0o644)
            p.unlink()
    return open(path, "w", **kwargs)


# ------------------------------------------------------------------
# Fuzzy c-means (self-contained, NumPy only)
# ------------------------------------------------------------------
def fuzzy_cmeans(X, n_clusters, m=2.0, max_iter=300, tol=1e-5, seed=42):
    """
    Standard (Bezdek, 1981) fuzzy c-means clustering.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Already NaN-free.
    n_clusters : int
    m : float
        Fuzziness exponent, > 1. 2.0 is the conventional default.
    max_iter, tol : stopping criteria.
    seed : random seed for the initial membership matrix.

    Returns
    -------
    centers, U, fpc, n_iter — see body for details.
    """
    rng = np.random.default_rng(seed)
    n_samples, n_features = X.shape
    U = rng.random((n_samples, n_clusters))
    U /= U.sum(axis=1, keepdims=True)

    centers = None
    for it in range(max_iter):
        Um = U ** m
        centers = (Um.T @ X) / Um.sum(axis=0)[:, None]

        dist = np.empty((n_samples, n_clusters))
        for j in range(n_clusters):
            dist[:, j] = np.linalg.norm(X - centers[j], axis=1)
        dist = np.fmax(dist, 1e-12)

        inv = dist ** (-2.0 / (m - 1.0))
        U_new = inv / inv.sum(axis=1, keepdims=True)

        diff = float(np.max(np.abs(U_new - U)))
        U = U_new
        if diff < tol:
            break

    fpc = float(np.sum(U ** 2) / n_samples)
    return centers, U, fpc, it + 1


# ------------------------------------------------------------------
# Self-organizing map (self-contained, NumPy only)
# ------------------------------------------------------------------
def train_som(X, rows, cols, n_iter=8000, lr0=0.5, sigma0=None, seed=42):
    """
    Online (sequential) Kohonen self-organizing map. See
    cluster_som.py's original docstring for the full parameter
    rationale; unchanged here.
    """
    rng = np.random.default_rng(seed)
    n_samples, n_features = X.shape
    n_neurons = rows * cols
    if sigma0 is None:
        sigma0 = max(rows, cols) / 2.0

    data_min, data_max = X.min(axis=0), X.max(axis=0)
    weights = rng.uniform(data_min, data_max, size=(n_neurons, n_features))

    grid_r, grid_c = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    grid_coords = np.column_stack([grid_r.ravel(), grid_c.ravel()]).astype(np.float64)

    sample_idx = rng.integers(0, n_samples, size=n_iter)

    for it in range(n_iter):
        frac = it / max(n_iter - 1, 1)
        lr = lr0 * (0.01 ** frac)
        sigma = max(sigma0 * (0.05 ** frac), 0.5)

        x = X[sample_idx[it]]
        dists = np.linalg.norm(weights - x, axis=1)
        bmu = int(np.argmin(dists))

        grid_dist_sq = np.sum((grid_coords - grid_coords[bmu]) ** 2, axis=1)
        neighborhood = np.exp(-grid_dist_sq / (2.0 * sigma ** 2))

        weights += lr * neighborhood[:, None] * (x - weights)

    return weights, grid_coords


def som_bmu_assign(X, weights):
    """Assign every row of X to its best-matching unit (BMU)."""
    dists = np.linalg.norm(X[:, None, :] - weights[None, :, :], axis=2)
    bmu_idx = np.argmin(dists, axis=1)
    qe = dists[np.arange(len(X)), bmu_idx]
    return bmu_idx, qe


def som_topographic_error(X, weights, grid_coords):
    """Fraction of points whose best- and second-best-matching units are
    NOT adjacent on the SOM grid (4-neighborhood, Manhattan distance 1)."""
    dists = np.linalg.norm(X[:, None, :] - weights[None, :, :], axis=2)
    order = np.argsort(dists, axis=1)
    c1 = grid_coords[order[:, 0]]
    c2 = grid_coords[order[:, 1]]
    grid_dist = np.abs(c1 - c2).sum(axis=1)
    return float(np.mean(grid_dist > 1))


def som_grid_colormap(rows, cols):
    """Topological ListedColormap: nearby SOM neurons get visually
    similar colors — see cluster_som.py's original docstring."""
    colors = np.zeros((rows * cols, 3))
    for r in range(rows):
        for c in range(cols):
            hue = c / max(cols - 1, 1)
            val = 0.55 + 0.45 * (r / max(rows - 1, 1))
            colors[r * cols + c] = mcolors.hsv_to_rgb([hue, 0.75, val])
    return mcolors.ListedColormap(colors)


# ==================================================================
# Read interpolate.py's output
# ==================================================================
interp_path = ncpath(INTERP_FILE)
print(f"Reading interpolated grid: {interp_path} …")
interp_ds = xr.open_dataset(interp_path)

grid_mode = interp_ds.attrs["target_grid_mode"]        # "joint" | "seismic"
dim_names = tuple(interp_ds.attrs["dim_names"].split(", "))
dim_depth, dim_row, dim_col = dim_names
interp_vars_available = interp_ds.attrs["interp_vars"].split(", ")

if CLUSTER_VARS is None:
    active_cluster_vars = list(interp_vars_available)
else:
    active_cluster_vars = list(CLUSTER_VARS)
    missing = [v for v in active_cluster_vars if v not in interp_ds.data_vars]
    if missing:
        raise KeyError(
            f"CLUSTER_VARS {missing} not present in {INTERP_FILE!r} (it has "
            f"{list(interp_ds.data_vars)}) — add them to interpolate.py's "
            f"INTERP_VARS and re-run it first."
        )

print(f"Clustering on: {active_cluster_vars}  "
      f"(grid_mode={grid_mode!r}, from {INTERP_FILE!r})")

d_axis = interp_ds[dim_depth].values.astype(np.float64)
row_coord = interp_ds[dim_row].values
col_coord = interp_ds[dim_col].values

_loaded = {}
resolved_units = {}
for key in active_cluster_vars:
    da = interp_ds[key]
    _loaded[key] = da.values.astype(np.float32)
    resolved_units[key] = da.attrs.get("units", "")

if grid_mode == "joint":
    e_axis = interp_ds[dim_col].values.astype(np.float64)     # "easting"
    n_axis = interp_ds[dim_row].values.astype(np.float64)     # "northing"
    utm_easting_2d = None
    utm_northing_2d = None
else:  # "seismic"
    e_axis = None
    n_axis = None
    utm_easting_2d = interp_ds["utm_easting_km"].values
    utm_northing_2d = interp_ds["utm_northing_km"].values

grid_shape = _loaded[active_cluster_vars[0]].shape  # (depth, row, col)
n_total = int(np.prod(grid_shape))
interp_ds.close()
print(f"Grid shape {dim_names}: {grid_shape} ({n_total} cells total)")

# ==================================================================
# Build feature table
# ==================================================================
feature_stack = np.stack(
    [_loaded[k].ravel() for k in active_cluster_vars], axis=1
)  # (n_total, n_features)
valid_mask = np.all(np.isfinite(feature_stack), axis=1)
n_valid = int(valid_mask.sum())
print(
    f"Valid (finite in every selected variable): {n_valid} / {n_total} "
    f"({100.0 * n_valid / n_total:.1f}%)"
)
if CLUSTERING_METHOD not in ("fcm", "som"):
    raise ValueError(f"CLUSTERING_METHOD must be 'fcm' or 'som', got {CLUSTERING_METHOD!r}.")
n_labels = N_CLUSTERS if CLUSTERING_METHOD == "fcm" else SOM_ROWS * SOM_COLS

if n_valid < n_labels:
    raise RuntimeError(
        f"Only {n_valid} valid points — fewer than the number of classes "
        f"({n_labels}). Check CLUSTER_VARS or interpolate.py's masking settings."
    )

X_raw = feature_stack[valid_mask]  # (n_valid, n_features)

if STANDARDIZE:
    feat_mean = X_raw.mean(axis=0)
    feat_std = X_raw.std(axis=0)
    feat_std[feat_std == 0] = 1.0
    X = (X_raw - feat_mean) / feat_std
else:
    feat_mean = np.zeros(X_raw.shape[1])
    feat_std = np.ones(X_raw.shape[1])
    X = X_raw

weight_vector = np.array(
    [CLUSTER_WEIGHTS.get(k, 1.0) for k in active_cluster_vars], dtype=float
)
if np.any(weight_vector <= 0):
    raise ValueError(
        f"CLUSTER_WEIGHTS must be positive; got "
        f"{dict(zip(active_cluster_vars, weight_vector))}."
    )
print(
    "Cluster weights: " +
    ", ".join(f"{k}={w:g}" for k, w in zip(active_cluster_vars, weight_vector))
)
sqrt_weight = np.sqrt(weight_vector)
X_weighted = X * sqrt_weight[None, :]

# ==================================================================
# Cluster: fuzzy c-means or self-organizing map
# ==================================================================
diagnostics = {}

if CLUSTERING_METHOD == "fcm":
    print(f"\nRunning fuzzy c-means: n_clusters={N_CLUSTERS}, m={FUZZINESS} …")
    centers_weighted, U, fpc, n_iter = fuzzy_cmeans(
        X_weighted, N_CLUSTERS, m=FUZZINESS, max_iter=MAX_ITER, tol=TOL, seed=RANDOM_SEED
    )
    hard_label = np.argmax(U, axis=1)
    membership_max = np.max(U, axis=1)
    centers_std = centers_weighted / sqrt_weight[None, :]
    centers_raw = centers_std * feat_std + feat_mean

    print(
        f"Converged after {n_iter} iterations. "
        f"Fuzzy partition coefficient (FPC): {fpc:.3f} "
        f"(1/{N_CLUSTERS} = {1.0 / N_CLUSTERS:.3f} = fuzziest, 1 = crisp)"
    )
    diagnostics["fuzziness_m"] = FUZZINESS
    diagnostics["fpc"] = fpc
    diagnostics["n_iter"] = n_iter
    diagnostics["random_seed"] = RANDOM_SEED

else:  # CLUSTERING_METHOD == "som"
    print(
        f"\nTraining SOM: grid={SOM_ROWS}x{SOM_COLS} ({n_labels} neurons), "
        f"n_iter={SOM_N_ITER}, lr0={SOM_LR0} …"
    )
    weights_weighted, grid_coords = train_som(
        X_weighted, SOM_ROWS, SOM_COLS,
        n_iter=SOM_N_ITER, lr0=SOM_LR0, sigma0=SOM_SIGMA0, seed=SOM_SEED,
    )
    hard_label, qe = som_bmu_assign(X_weighted, weights_weighted)
    membership_max = qe  # SOM's quantization error, NOT a [0, 1] membership value
    topo_err = som_topographic_error(X_weighted, weights_weighted, grid_coords)
    centers_std = weights_weighted / sqrt_weight[None, :]
    centers_raw = centers_std * feat_std + feat_mean

    print(
        f"Done. Mean quantization error: {qe.mean():.3f} (standardized units); "
        f"topographic error: {topo_err:.3f} "
        f"(0 = perfectly neighborhood-preserving, higher = worse)"
    )
    diagnostics["som_grid"] = f"{SOM_ROWS}x{SOM_COLS}"
    diagnostics["som_n_iter"] = SOM_N_ITER
    diagnostics["som_lr0"] = SOM_LR0
    diagnostics["som_sigma0"] = SOM_SIGMA0 if SOM_SIGMA0 is not None else max(SOM_ROWS, SOM_COLS) / 2.0
    diagnostics["som_seed"] = SOM_SEED
    diagnostics["mean_quantization_error"] = float(qe.mean())
    diagnostics["topographic_error"] = topo_err

print("\nCluster centers (raw units) and sizes:")
header = "  cluster |    n    |  frac  | " + " | ".join(
    f"{k} ({resolved_units[k] or '-'})" for k in active_cluster_vars
)
print(header)
for c in range(n_labels):
    n_c = int(np.sum(hard_label == c))
    frac = n_c / n_valid
    vals = "  ".join(f"{centers_raw[c, j]:9.3f}" for j in range(len(active_cluster_vars)))
    print(f"  {c:7d} | {n_c:7d} | {frac:5.1%} | {vals}")

# ==================================================================
# Reconstruct grid + save
# ==================================================================
label_flat = np.full(n_total, -1, dtype=np.int16)
label_flat[valid_mask] = hard_label.astype(np.int16)
label_grid = label_flat.reshape(grid_shape)

membership_flat = np.full(n_total, np.nan, dtype=np.float32)
membership_flat[valid_mask] = membership_max.astype(np.float32)
membership_grid = membership_flat.reshape(grid_shape)

_method_label = "Fuzzy c-means" if CLUSTERING_METHOD == "fcm" else "SOM (self-organizing map)"
_label_long_name = (
    "Fuzzy c-means hard cluster label (argmax membership)" if CLUSTERING_METHOD == "fcm"
    else "SOM best-matching-unit (BMU) index, flat row*cols+col over the SOM_ROWS x SOM_COLS grid"
)
_membership_long_name = (
    "Membership of the assigned (hard-label) cluster" if CLUSTERING_METHOD == "fcm"
    else "Quantization error: distance (standardized/weighted feature space) from the point to "
         "its BMU weight vector -- smaller is a better fit, NOT a [0, 1] membership value"
)

out_coords = {
    dim_depth: (dim_depth, d_axis, {"units": "km", "positive": "down"}),
    dim_row: (dim_row, row_coord),
    dim_col: (dim_col, col_coord),
}
if grid_mode == "seismic":
    out_coords["utm_easting_km"] = ((dim_row, dim_col), utm_easting_2d)
    out_coords["utm_northing_km"] = ((dim_row, dim_col), utm_northing_2d)

out_ds = xr.Dataset(
    {
        "cluster_label": (
            dim_names, label_grid,
            {
                "long_name": _label_long_name,
                "flag_value_missing": -1,
                "cluster_vars": ", ".join(active_cluster_vars),
                "n_clusters": n_labels,
                "clustering_method": CLUSTERING_METHOD,
            },
        ),
        "membership": (
            dim_names, membership_grid,
            {
                "long_name": _membership_long_name,
                "units": "1" if CLUSTERING_METHOD == "fcm" else "(standardized/weighted feature units)",
            },
        ),
    },
    coords=out_coords,
    attrs={
        "description": (
            f"{_method_label} clustering of " + ", ".join(active_cluster_vars) +
            f" on the grid read from {INTERP_FILE} (see interpolate.py)."
        ),
        "source_interp_file": INTERP_FILE,
        "target_grid_mode": grid_mode,
        "clustering_method": CLUSTERING_METHOD,
        "cluster_weights": ", ".join(
            f"{k}={w:g}" for k, w in zip(active_cluster_vars, weight_vector)
        ),
        "standardized": str(STANDARDIZE),
        **{k: ("" if v is None else v) for k, v in diagnostics.items()},
    },
)
out_nc = ncpath(f"{SITE_PREFIX}_clusters.nc")
safe_to_netcdf(out_ds, out_nc)
print(f"\nSaved: {out_nc}")

centers_csv = ncpath(f"{SITE_PREFIX}_cluster_centers.csv")
with safe_open_w(centers_csv, newline="") as f:
    w = csv.writer(f)
    w.writerow(["cluster", "n_points", "fraction"] + active_cluster_vars)
    w.writerow(["weight", "", ""] + [f"{wt:g}" for wt in weight_vector])
    for c in range(n_labels):
        n_c = int(np.sum(hard_label == c))
        w.writerow(
            [c, n_c, f"{n_c / n_valid:.6f}"]
            + [f"{centers_raw[c, j]:.6g}" for j in range(len(active_cluster_vars))]
        )
print(f"Saved: {centers_csv}")

# ==================================================================
# Load topo/bath basemap
# ==================================================================
print("\nLoading topo/bath grids …")
_topo_da = xr.open_dataarray(ncpath(f"{SITE_PREFIX}_topo_utm.nc"))
topo_x = _topo_da["x"].values
topo_y = _topo_da["y"].values
topo_z = _topo_da.values
_topo_da.close()
dx_km = float(np.median(np.diff(topo_x)))
dy_km = float(np.median(np.diff(topo_y)))

if SHOW_TOPO_BASEMAP:
    topo_hs = compute_hillshade(topo_z, dx_km, dy_km, HS_AZIMUTH, HS_ALTITUDE, HS_SIGMA)
else:
    topo_hs = None

_bath_da = xr.open_dataarray(ncpath(f"{SITE_PREFIX}_bath_utm.nc"))
bath_x = _bath_da["x"].values
bath_y = _bath_da["y"].values
bath_z = _bath_da.values
_bath_da.close()

topo_extent = [topo_x.min(), topo_x.max(), topo_y.min(), topo_y.max()]
bath_extent = [bath_x.min(), bath_x.max(), bath_y.min(), bath_y.max()]
topo_norm = mcolors.Normalize(vmin=TOPO_VMIN, vmax=TOPO_VMAX) if SHOW_TOPO_BASEMAP else None
CMAP_TOPO = plt.get_cmap("gray")

# ==================================================================
# Map region
# ==================================================================
if grid_mode == "joint":
    _e_min, _e_max = float(e_axis.min()), float(e_axis.max())
    _n_min, _n_max = float(n_axis.min()), float(n_axis.max())
else:
    _e_min, _e_max = float(np.nanmin(utm_easting_2d)), float(np.nanmax(utm_easting_2d))
    _n_min, _n_max = float(np.nanmin(utm_northing_2d)), float(np.nanmax(utm_northing_2d))

if MAP_XLIM is not None:
    xmin, xmax = MAP_XLIM
else:
    xmin = _e_min - REGION_MARGIN_KM
    xmax = _e_max + REGION_MARGIN_KM
if MAP_YLIM is not None:
    ymin, ymax = MAP_YLIM
else:
    ymin = _n_min - REGION_MARGIN_KM
    ymax = _n_max + REGION_MARGIN_KM
print(f"Map region (km): [{xmin}, {xmax}, {ymin}, {ymax}]")


def _region():
    return (xmin, xmax, ymin, ymax)


def _colorbar_settings():
    return dict(
        show=SHOW_COLORBAR, position=COLORBAR_POSITION,
        size=COLORBAR_SIZE, pad=COLORBAR_PAD, aspect=COLORBAR_ASPECT,
        label_size=COLORBAR_LABEL_SIZE, tick_size=COLORBAR_TICK_SIZE,
        nticks=n_labels, title_size=AXIS_TITLE_SIZE,
    )


def create_map_figure():
    map_w_in = FIG_WIDTH / 2.54
    map_h_in = map_w_in * (ymax - ymin) / (xmax - xmin)
    return plotpy.build_panel_figure(
        map_w_in, map_h_in, _colorbar_settings(), size_label="map"
    )


def draw_basemap(ax):
    """Topo greyscale + hillshade + ocean fill; enforce map limits."""
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")

    if SHOW_TOPO_BASEMAP:
        ax.imshow(
            CMAP_TOPO(topo_norm(topo_z)), origin="lower", extent=topo_extent,
            aspect="auto", interpolation="bilinear", zorder=1,
        )
        ax.imshow(
            topo_hs, cmap="gray", origin="lower", extent=topo_extent,
            alpha=0.45, aspect="auto", interpolation="bilinear", zorder=2,
        )
    bath_mask = np.where(bath_z <= 0, 1.0, np.nan)
    ax.imshow(
        bath_mask, origin="lower", extent=bath_extent,
        cmap=mcolors.ListedColormap([OCEAN_COLOR]), vmin=0, vmax=1,
        alpha=0.85, aspect="auto", interpolation="none", zorder=3,
    )
    ax.set_xlabel("Easting (km)", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Northing (km)", fontsize=AXIS_LABEL_SIZE)
    if AXES_UNITS == "km" and AXES_KM_COMMA:
        _comma_fmt = mpl.ticker.StrMethodFormatter("{x:,.0f}")
        ax.xaxis.set_major_formatter(_comma_fmt)
        ax.yaxis.set_major_formatter(_comma_fmt)
    ax.tick_params(labelsize=AXIS_TICK_SIZE)
    if SHOW_NORTH_ARROW:
        arr_e, arr_n = to_utm_km([ARROW_LON], [ARROW_LAT])
        plotpy.draw_north_arrow(
            ax, arr_e[0], arr_n[0], _region(),
            ARROW_STYLE, ARROW_LABEL_STYLE, ARROW_LEN_KM,
        )


def save_fig(fig, stem):
    for fmt in PLOT_FORMATS:
        out = os.path.join(PLOT_DIR, stem + fmt)
        fig.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
        print(f"  Saved: {out}")


# ==================================================================
# Feature layers for the specific (annotated) plot — same CSVs/loading
# as plot_seis.py
# ==================================================================
if SHOW_SPECIFIC_PLOT:
    print("\nLoading feature layers for the specific (annotated) plot …")

    volcanes = pd.read_csv(CSV_VOLCANES)
    utmv_e, utmv_n = to_utm_km(
        volcanes["LONG"][VOLC_LABEL_IDX].values,
        volcanes["LAT"][VOLC_LABEL_IDX].values,
    )
    if VOLC_NAME_COL not in volcanes.columns:
        print(f"  WARNING: volcano name column {VOLC_NAME_COL!r} not found in "
              f"{CSV_VOLCANES} — labels will be blank.")
        namev = [""] * len(VOLC_LABEL_IDX)
    else:
        namev = volcanes[VOLC_NAME_COL][VOLC_LABEL_IDX].values

    volc_act_e, volc_act_n = [], []
    for _i in range(len(volcanes)):
        if "ACT" in str(volcanes["ESTADO"][_i]):
            _ae, _an = to_utm_km([volcanes["LONG"][_i]], [volcanes["LAT"][_i]])
            volc_act_e.append(_ae[0])
            volc_act_n.append(_an[0])

    eqs = pd.read_csv(CSV_SEISMCAT, delimiter=" ")
    eq_e0, eq_n0 = to_utm_km(eqs["x"].values, eqs["y"].values)
    zeqs = eqs["z"].values

    _mt = pd.read_csv(CSV_MT_SITES, delimiter=" ")
    mt_e, mt_n = to_utm_km(_mt["x"].values, _mt["y"].values)
    for _name_col in ("Site", "site", "name", "Name", "station", "Station"):
        if _name_col in _mt.columns:
            mt_names = _mt[_name_col].astype(str).tolist()
            break
    else:
        mt_names = [""] * len(_mt)

    seis_sites = pd.read_csv(CSV_SEISMIC_SITES, header=None,
                              names=["network", "station", "lat", "lon", "elev_m"])
    seis_site_e, seis_site_n = to_utm_km(seis_sites["lon"].values, seis_sites["lat"].values)
    seis_site_names = seis_sites["station"].values

    cities = pd.read_csv(CSV_CITIES)
    cit_e, cit_n = to_utm_km(cities["x"].values, cities["y"].values)
    name_cit = cities["Name"].values


def draw_specific_features(ax, eq_e, eq_n):
    """Overlay seismicity/MT-site/seismic-site/volcano/city feature layers
    borrowed from plot_seis.py's draw_features()."""
    if SHOW_SEISMICITY:
        clipped_markers(ax, eq_e, eq_n, _region(), label="Seismicity", **EQ_MARKER_STYLE)

    if SHOW_MT_SITES:
        clipped_markers(ax, mt_e, mt_n, _region(), label="MT site", **MT_MARKER_STYLE)
        clipped_labels(ax, mt_e, mt_n, mt_names, MT_LABEL_STYLE, _region())

    if SHOW_SEISMIC_SITES:
        clipped_markers(ax, seis_site_e, seis_site_n, _region(), label="Seismic site",
                         **SEISMIC_SITES_MARKER_STYLE)

    if SHOW_VOLCANOES:
        clipped_markers(ax, utmv_e, utmv_n, _region(), **VOLC_INACT_MARKER_STYLE)
        clipped_labels(ax, utmv_e, utmv_n, namev, VOLC_LABEL_STYLE, _region())

    if SHOW_VOLCANOES_ACTIVE and volc_act_e:
        clipped_markers(ax, volc_act_e, volc_act_n, _region(),
                         label="Active volcano", **VOLC_ACT_MARKER_STYLE)

    if SHOW_CITIES:
        clipped_markers(ax, cit_e, cit_n, _region(), label="City", **CITY_MARKER_STYLE)
        clipped_labels(ax, cit_e, cit_n, name_cit, CITY_LABEL_STYLE, _region())


# ==================================================================
# Plot cluster maps
# ==================================================================
if CLUSTERING_METHOD == "som":
    cluster_cmap = som_grid_colormap(SOM_ROWS, SOM_COLS)
else:
    cluster_cmap = plt.get_cmap(CLUSTER_CMAP, N_CLUSTERS)
bounds = np.arange(-0.5, n_labels + 0.5, 1.0)
cluster_norm = mcolors.BoundaryNorm(bounds, cluster_cmap.N)

_tick_stride = max(1, -(-n_labels // 12))  # ceil(n_labels / 12)
_cbar_ticks = list(range(0, n_labels, _tick_stride))


def _draw_cluster_overlay(ax, cax, label_slice, actual_depth):
    """Cluster map + title + colorbar + lon/lat ticks + free-text
    annotation — the part shared by both the plain and the specific
    (annotated) cluster maps. Uses imshow(extent=...) for a regular
    "joint" grid, pcolormesh(shading="nearest") against the grid's own
    2-D UTM coords for a reused "seismic" grid (not regular in UTM
    space)."""
    if grid_mode == "joint":
        im = ax.imshow(
            label_slice, cmap=cluster_cmap, norm=cluster_norm, origin="lower",
            extent=[e_axis.min(), e_axis.max(), n_axis.min(), n_axis.max()],
            alpha=CLUSTER_ALPHA, aspect="equal", interpolation="nearest", zorder=5,
        )
    else:
        im = ax.pcolormesh(
            utm_easting_2d, utm_northing_2d, label_slice,
            cmap=cluster_cmap, norm=cluster_norm, shading="nearest",
            alpha=CLUSTER_ALPHA, zorder=5,
        )
    ax.set_title(
        f"{_method_label} classes ({', '.join(active_cluster_vars)}) at {actual_depth:.1f} km",
        fontsize=AXIS_TITLE_SIZE,
    )
    if SHOW_COLORBAR:
        cbar = plotpy.finish_panel_colorbar(cax, im, "Class", _colorbar_settings())
        cbar.set_ticks(_cbar_ticks)
    if AXES_UNITS == "latlon":
        plotpy.add_latlon_ticks(
            ax, _region(), LATLON_NTICKS, LATLON_DECIMALS,
            AXIS_LABEL_SIZE, AXIS_TICK_SIZE,
        )
    plotpy.draw_annotation(ax, ANNOTATION_TEXT, ANNOTATION_POS, ANNOTATION_STYLE)


for i_depth, target_depth in enumerate(PLOT_DEPTHS_KM):
    iz = int(np.argmin(np.abs(d_axis - target_depth)))
    actual_depth = float(d_axis[iz])
    print(
        f"\nPlotting clusters at {target_depth} km "
        f"(nearest available: {actual_depth:.2f} km) …"
    )

    label_slice = label_grid[iz].astype(float)  # (row, col)
    label_slice[label_slice < 0] = np.nan  # -1 (missing) -> NaN, transparent

    tag = f"{actual_depth:.0f}km" if actual_depth == int(actual_depth) else f"{actual_depth:.1f}km"

    # --- Plain cluster map ---
    fig, ax, cax = create_map_figure()
    draw_basemap(ax)
    _draw_cluster_overlay(ax, cax, label_slice, actual_depth)
    save_fig(fig, f"clusters_{tag}_{SITE_PREFIX}")
    plt.show()
    plt.close(fig)

    # --- Specific (annotated) cluster map ---
    if SHOW_SPECIFIC_PLOT:
        if SHOW_SEISMICITY:
            zmin, zmax = ZMIN_SEISM[i_depth], ZMAX_SEISM[i_depth]
            if zmin is None and zmax is None:
                eq_e, eq_n = eq_e0, eq_n0
            else:
                _zmask = np.ones(len(zeqs), dtype=bool)
                if zmin is not None:
                    _zmask &= zeqs >= zmin
                if zmax is not None:
                    _zmask &= zeqs <= zmax
                eq_e, eq_n = eq_e0[_zmask], eq_n0[_zmask]
        else:
            eq_e, eq_n = np.array([]), np.array([])

        fig, ax, cax = create_map_figure()
        draw_basemap(ax)
        _draw_cluster_overlay(ax, cax, label_slice, actual_depth)
        draw_specific_features(ax, eq_e, eq_n)
        save_fig(fig, f"clusters_{tag}_{SITE_PREFIX}_annotated")
        plt.show()
        plt.close(fig)

print("\nDone.")
