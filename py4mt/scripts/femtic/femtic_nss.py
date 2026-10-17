#!  /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
femtic_nss.py — Nullspace Shuttle for FEMTIC inversion results
==============================================================

Workflow
--------
    (1) Read the final model (and an approximate misfit) from the FEMTIC
        results file ``results_iterX.h5`` (``/model/blocks``, ``/data/data``)
        and the Jacobian from the FEMTIC exchange file ``exchange.h5``
        (``/jacobian/values``, ``/jacobian/data_errors``; a legacy
        ``jacobian.h5`` is also accepted).
    (2) Form the data-weighted Jacobian Js.  The Jacobian written by FEMTIC
        is already weighted, J = Cd^(-1/2) dF/dm, so Js = J; it is divided by
        the data errors only if the file says it is not weighted
        (``JAC_WEIGHTED``).
    (3) Compute the randomised SVD of Js with N singular values / vectors
        (``RSVD_RANK``; Halko et al., 2011).  The singular vectors define the
        data-space and null-space projectors.
    (4) Model perturbation: ``PERTURB_MODE`` selects a geostatistical
        ("gst", pilot-point Kriging) or a Gaussian ("random") perturbation;
        edit ``_make_perturbation_random`` to inject prior knowledge.
    (5) Nullspace shuttle, chunk by chunk: ``NSS_N_MODELS`` perturbations are
        drawn ``NSS_CHUNK_SIZE`` at a time, projected onto the null-space of
        Js so that they cannot change the predicted data, and added to the
        final model.  Each chunk is discarded after it has been merged into
        streaming statistics, so the ensemble never has to fit in memory:
        per-block mean and variance (Welford / Chan, pairwise merge of the
        chunk moments) and per-block quantiles at the 1-sigma and 2-sigma
        probabilities (fixed-bin histograms).  The first shuttled model is
        written to ``MODEL_OUT``; the mean and the quantile models are
        written next to it (``NSS_WRITE_STAT_MODELS``) and all statistics
        go to ``NSS_STATS_FILE``.

Theory (brief)
--------------
The null-space N(Js) is spanned by the right singular vectors of Js whose
singular values are zero (or below the threshold ``NSS_SV_THRESH``).  Given
any model perturbation δm̃, the null-space component is::

    δm_null = (I - Vr @ Vr.T) @ δm̃       (*)

where Vr = Vt[:rank].T contains the top-rank right singular vectors.  Adding
δm_null to the current model produces a new model with identical predicted
data (to within the truncation rank).

Literature
----------

[1] M. Deal, G. Nolet (1996) “Nullspace shuttles", Geophysical Journal International, 124, 372–380,doi:10.1111/j.1365-246X.1996.tb07027.x

[2] G. Muñoz, V. Rath (2006) “Beyond smooth inversion: the use of nullspace projection for the exploration of non-uniqueness in MT", Geophysical Journal International, 164, 301–311, 2006, doi:10.1111/j.1365-246X.2005.02825.x



Provenance
----------
    2026-05-17  vrath / Claude Sonnet 4.6   Created, modelled on
                femtic_mod_edit.py.  Uses ``inverse.rsvd`` for the randomised
                SVD and a local ``_nullspace_shuttle`` helper for step (5).
    2026-06-23  vrath / Claude Sonnet 4.6   Merged GST model-generation from
                femtic_gst_prep.py into step 4.  PERTURB_MODE = "gst" draws
                a geostatistically perturbed model via pilot-point Ordinary
                Kriging (ensembles.generate_gst_model_ensemble), computes the
                delta w.r.t. the reference model, and passes it to the null-
                space shuttle.  PERTURB_MODE = "random" retains the original
                uniform-Gaussian placeholder.  GST config block mirrors
                femtic_gst_prep.py exactly (pp_mode, variogram, etc.).
    2026-07-09  vrath / Claude Sonnet 5 (Anthropic)
                Added the shared MOD_* plotting config block (MOD_MESH,
                MOD_OCEAN/AIR_RHO/OCEAN_RHO, MOD_UTM_ORIGIN_*, MOD_ORIGIN_
                METHOD, MOD_DISPLAY_COORDS, MOD_SITE_*, MOD_SLICES,
                MOD_XLIM/YLIM/ZLIM, MOD_CMAP/CLIM, MOD_OCEAN_COLOR/
                AIR_COLOR/AIR_BGCOLOR, MOD_ALPHA_*, MOD_EQUAL_ASPECT/
                DEPTH_KM/HORIZ_KM/NROWS/NCOLS/PANEL_HEIGHT/PANEL_WIDTH/
                FIGSIZE) plus MOD_QC/MOD_QC_FILE/MOD_DPI, identical in
                name and order to femtic_ens_post.py and femtic_gst_prep.py.
                Added femtic_viz import, _resolve_origin_and_sites() and
                _plot_slice() helpers (mirroring femtic_ens_post.py), and an
                optional QC slice plot of MODEL_OUT at the end of the run.
                A plotting config block can now be copied between all three
                scripts with no renaming. Uses a single MOD_DPI knob (no
                per-plot-type split), matching femtic_gst_prep.py and the
                now-simplified femtic_ens_post.py.
    2026-07-25  Claude Sonnet 5 (Anthropic)
                Added RANDOM_SEED (default None): when set to an integer,
                a shared rng = np.random.default_rng(RANDOM_SEED) drives
                Step 4's model perturbation instead of unseeded/hardcoded
                randomness. Two bugs fixed en route: (1) _make_perturbation_
                gst (PERTURB_MODE="gst") called
                ens.generate_gst_model_ensemble(rng=np.random.default_rng())
                -- a fresh, unseeded generator constructed inline every
                call -- now passes the shared rng instead. (2)
                _make_perturbation_random (PERTURB_MODE="random") used
                np.random.default_rng(seed=0) -- hardcoded, always the same
                draw regardless of any seed the user might set -- now uses
                the shared rng too, so it follows RANDOM_SEED like every
                other reproducibility-aware script in this project instead
                of being silently pinned to seed 0. NOTE: Step 3's
                randomised SVD (inv.rsvd) is NOT covered by RANDOM_SEED --
                inverse.py is a separate module and its internal
                range-finder randomness was not verified/wired through;
                see the RANDOM_SEED config comment for details.
    2026-07-25  Claude Sonnet 5 (Anthropic)
                Added MOD_TICK_FONTSIZE / MOD_LABEL_FONTSIZE (defaults 7/8,
                matching fviz.plot_model_slices' own defaults), passed
                through to the MOD_QC call. Axis tick labels, axis labels,
                panel title, and colourbar text were previously fixed at
                plot_model_slices' internal defaults with no way to
                override them from this script.
    2026-07-25  Claude Sonnet 5 (Anthropic)
                Added MOD_SHOW_IN_SPYDER (default True): when this script
                is running inside Spyder (detected once via
                utl.runtime_env() == "spyder"), the MOD_QC figure is also
                displayed inline via plt.show() (fviz.plot_model_slices'
                new show= parameter), without changing what gets saved to
                disk. No effect outside Spyder. Set MOD_SHOW_IN_SPYDER=
                False to disable even under Spyder.
    2026-07-27  Claude Sonnet 5 (Anthropic)
                Homogenized spatial config parameters to km: MOD_XLIM /
                MOD_YLIM / MOD_ZLIM and MOD_PROJECTION_DIST are now
                specified in km instead of metres (values and comments
                updated accordingly). femtic_viz.py itself is unchanged
                and still expects metres (matching mesh.dat); new
                module-level helpers _km_to_m() / _lim_km_to_m()
                convert these config values to metres at the
                fviz.plot_model_slices call site. MOD_UTM_ORIGIN_E/N
                left in metres (absolute UTM geodetic convention, not
                a model-local span).
    2026-07-27  Claude Sonnet 5 (Anthropic)
                MOD_PLOT_SITES_SLICES: default changed False -> True,
                so site markers (projected onto the plane, within
                MOD_PROJECTION_DIST) now also appear on the ns/ew/
                plane curtain panels by default, matching
                femtic_rto_prep.py / femtic_gst_prep.py. No
                femtic_viz.py change needed -- the sites_in_slices /
                site_marker_slices plumbing already existed; only
                this script's default was off.
    2026-07-27  Claude Sonnet 5 (Anthropic)
                Extended the km homogenization to the geostatistical
                (pilot-point/variogram) parameters, which were missed
                in the earlier pass: GST_PP_BBOX, GST_PP_ROI, and
                GST_PP_COORDS are now specified in km (model-local,
                same convention as MOD_XLIM/YLIM/ZLIM); GST_VARIO_RANGE
                (horizontal, vertical correlation lengths) likewise.
                New helpers _range_km_to_m() (scalar-or-tuple) and
                _coords_km_to_m() ((N,3) array) convert these to the
                metres expected by ensembles.py at the
                ens.generate_gst_model_ensemble call site, alongside
                the existing _lim_km_to_m() (reused for the bbox/ROI
                6-element lists). GST_PP_EXTREMA_K (a neighbour count)
                and GST_VARIO_SILL/NUGGET (variance units) are
                unaffected.
    2026-07-27  Claude Sonnet 5 (Anthropic)
                Extended the km homogenization to the MOD_SLICES
                slice-spec dicts, which were missed in the earlier
                passes: numeric x0/y0/z0 values (e.g. dict(kind="map",
                z0=5.0)) are now km instead of metres. New helper
                _slices_km_to_m() walks the list of dicts, scaling any
                plain-numeric x0/y0/z0 by 1000 and leaving (value,
                "latlon") tuples / other keys untouched; applied right
                before fem.resolve_slice_positions(MOD_SLICES, ...).
    2026-08-13  Claude Sonnet 5 (Anthropic)
                Added femtic_nss_summary.md output at end of run: writes
                user-set (UPPERCASE) parameters, script path, and run
                date/time via utl.write_param_summary().
    2026-10-16  Claude Sonnet 5.5 (Anthropic)
                Steps 1-2 now read FEMTIC's own HDF5 output instead of the
                ad-hoc Inversion_results.h5: the final model (log10 rho of
                the free blocks, ascending blockID order) comes from
                results_iterX.h5 (RESULTS_FILE) and the Jacobian from
                exchange.h5 (JACOBIAN_FILE; a legacy jacobian.h5 also
                works).  The exchange.h5 Jacobian is already SD-weighted,
                J = Cd^(-1/2) dF/dm, so it is used as Js directly instead of
                being divided by the errors a second time (JAC_WEIGHTED
                overrides the file attribute).  The Jacobian rows follow the
                inversion's data ordering, not the row order of
                /data/data, so the misfit printed in step 2 is computed
                from the results file alone.  The number of Jacobian columns
                is checked against the number of free blocks and against
                MODEL_TEMPLATE; a differing iteration number in the two files
                is reported.  Step 3 (randomised SVD with N = RSVD_RANK
                vectors) and the shuttle itself are unchanged.  Anisotropic
                (v5) results files are rejected (the model vector is not one
                value per free block there).  HDF5_FILE removed.
                AI-generated code -- review before production use.
    2026-10-16  Claude Sonnet 5.5 (Anthropic)
                Steps 4-5 generate NSS_N_MODELS shuttled models in chunks of
                NSS_CHUNK_SIZE and keep only streaming statistics of the
                log10(rho) of the free blocks: mean and variance (per-chunk
                moments combined with ensembles.chan_merge, Chan, Golub and
                LeVeque 1983) and quantiles at the normal-distribution
                probabilities of -2, -1, 0, +1, +2 sigma (per-block
                fixed-bin histograms, ensembles.hist_update /
                hist_quantiles; error at most one bin width).  Also
                written: Gaussian bands mean +/- k*std for comparison.
                _nullspace_shuttle() now accepts a (n_models, nm) array as
                well as one vector (same projection, one matrix product per
                chunk).  The first model is still written to MODEL_OUT and
                plotted by MOD_QC; the mean and quantile models go to
                MODEL_OUT with suffixes _mean, _q0228, _q1587, _q5000,
                _q8413, _q9772, and everything to NSS_STATS_FILE (.npz).
                New config: NSS_N_MODELS, NSS_CHUNK_SIZE, NSS_SIGMAS,
                NSS_HIST_NBINS, NSS_HIST_RANGE, NSS_WRITE_STAT_MODELS,
                NSS_STATS_FILE.  Needs the 2026-10-16 ensembles.py.
                AI-generated code -- review before production use.

@author: vrath
"""

import os
import sys
import math
import inspect

import time

import numpy as np
import h5py

# ---------------------------------------------------------------------------
# Py4MTX-specific settings and imports
# ---------------------------------------------------------------------------
PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [PY4MTX_ROOT + "/py4mt/modules/", PY4MTX_ROOT + "/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)

from version import versionstrg
import util as utl
import femtic as fem
import inverse as inv
import ensembles as ens

try:
    import femtic_viz as fviz
except ImportError:
    fviz = None


def _km_to_m(val):
    """Convert a scalar distance in km to metres; None passes through.

    Used to convert MOD_PROJECTION_DIST (and similar scalar spatial
    config parameters, now specified in km) to the metres expected by
    femtic_viz.py, which works in model-local/UTM metres throughout
    (matching mesh.dat).
    """
    return None if val is None else float(val) * 1000.0


def _lim_km_to_m(lim):
    """Convert a [min, max] km limit pair to metres; None passes through.

    Used to convert MOD_XLIM/MOD_YLIM/MOD_ZLIM (now specified in km) to
    the metres expected by femtic_viz.py. Also reused for GST_PP_BBOX /
    GST_PP_ROI, which are longer [xmin,xmax,ymin,ymax,zmin,zmax] lists
    in the same model-local km convention.
    """
    return None if lim is None else [float(v) * 1000.0 for v in lim]


def _slices_km_to_m(slices):
    """Convert model-local km slice coordinates (x0/y0/z0) to metres.

    ``slices`` is the MOD_SLICES list of dicts, e.g.
    ``dict(kind="map", z0=5.0)`` or ``dict(kind="ns", x0=0.0)``. Plain
    numeric x0/y0/z0 values (model-local km) are converted to metres;
    (value, "latlon") tuples are left untouched since those are
    degrees, resolved separately by fem.resolve_slice_positions().
    None passes through.
    """
    if slices is None:
        return None
    out = []
    for spec in slices:
        spec = dict(spec)
        for key in ("x0", "y0", "z0"):
            if key in spec and isinstance(spec[key], (int, float)):
                spec[key] = spec[key] * 1000.0
        out.append(spec)
    return out


def _range_km_to_m(rng):
    """Convert a scalar or (horizontal, vertical) km range to metres.

    Used for GST_VARIO_RANGE (variogram correlation length), which may
    be a single scalar (isotropic) or a 2-tuple (horizontal, vertical).
    None passes through.
    """
    if rng is None:
        return None
    if isinstance(rng, (int, float)):
        return float(rng) * 1000.0
    return tuple(float(v) * 1000.0 for v in rng)


def _coords_km_to_m(coords):
    """Convert an (N, 3) array of km pilot-point coordinates to metres.

    Used for GST_PP_COORDS (explicit pilot-point easting/northing/depth,
    same model-local km convention as GST_PP_BBOX). None passes through.
    """
    return None if coords is None else np.asarray(coords, dtype=float) * 1000.0


version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())
titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng + "\n\n")

# ---------------------------------------------------------------------------
# Reproducibility (optional)
# ---------------------------------------------------------------------------
#: Set RANDOM_SEED to an integer for a reproducible run: same GST
#: perturbation (Step 4, PERTURB_MODE = "gst") and same random-Gaussian
#: perturbation (Step 4, PERTURB_MODE = "random"), given identical
#: HDF5 input / templates / config. None (default) uses fresh OS entropy.
#:
#: NOTE: the randomised SVD in Step 3 (inv.rsvd) uses its own internal
#: randomness for the range-finder projection and is NOT currently driven
#: by RANDOM_SEED -- inverse.py is a separate module and its rsvd() may or
#: may not expose an rng/seed parameter. If exact bit-for-bit
#: reproducibility of the SVD itself matters (not just the perturbation
#: step), check inv.rsvd's signature and wire RANDOM_SEED through there
#: too before relying on this flag for full end-to-end determinism.
RANDOM_SEED = None   # e.g. 20260725 for a reproducible run; None = fresh entropy

rng = np.random.default_rng(RANDOM_SEED)
print(f"RNG seed: {RANDOM_SEED if RANDOM_SEED is not None else '(fresh entropy — not reproducible)'}\n")


# ===========================================================================
# Configuration
# ===========================================================================

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORK_DIR = r"/home/vrath/Py4MTX/work/"

#: FEMTIC results file (results_iterX.h5) of the iteration the Jacobian
#: belongs to: /model/blocks (final model) and /data/data (observed and
#: calculated data).  Replace "10" by the iteration number of your run.
RESULTS_FILE = WORK_DIR + "results_iter10.h5"

#: FEMTIC exchange file with the Jacobian (/jacobian/values and
#: /jacobian/data_errors, last accepted iteration).  A legacy jacobian.h5
#: (root-level dataset /jacobian) is also accepted.
JACOBIAN_FILE = WORK_DIR + "exchange.h5"

#: Is the Jacobian in JACOBIAN_FILE already divided by the data errors?
#:   None  -> read the file's "weighted" attribute (exchange.h5 is written
#:            SD-weighted, J = Cd^(-1/2) dF/dm, and defaults to True; a legacy
#:            jacobian.h5 without that attribute raises an error)
#:   True  -> use J as is (Js = J)
#:   False -> divide each row by data_errors (Js = diag(1/error) @ J)
JAC_WEIGHTED = None

#: Output resistivity block (nullspace-modified model).
MODEL_OUT = WORK_DIR + "resistivity_block_nss.dat"

#: Template resistivity block for header / flag / fixed-region metadata.
#: Only the free-region values are replaced; all other columns are preserved.
MODEL_TEMPLATE = WORK_DIR + "resistivity_block_iter0.dat"

# ---------------------------------------------------------------------------
# Randomised SVD parameters  (step 3)
# ---------------------------------------------------------------------------
#: N: number of singular values / vectors computed by the randomised SVD of
#: Js (target rank).  Should be << min(nd, nm).  Increase until the
#: singular-value spectrum flattens to capture all significant data
#: information content.
RSVD_RANK = 300

#: Oversampling parameter (None → 2 × RSVD_RANK as in Halko et al.).
RSVD_OVERSAMPLES = None

#: Number of subspace (power) iterations.  More iterations → more accurate
#: decomposition at the cost of extra matrix-vector products.
RSVD_SUBSPACE_ITERS = 2

# ---------------------------------------------------------------------------
# Nullspace shuttle parameters  (step 5)
# ---------------------------------------------------------------------------
#: Singular-value threshold below which a right singular vector is treated as
#: belonging to the null space.  Expressed as a fraction of the largest
#: singular value s[0].
NSS_SV_THRESH = 1.0e-3

#: Amplitude scale applied to the null-space perturbation before adding it to
#: the final model.  Start small (e.g. 0.1) and increase to explore the null
#: space more aggressively.
NSS_AMPLITUDE = 1.0

#: Number of shuttled models to generate.  The models are never all held in
#: memory; see NSS_CHUNK_SIZE.  With "gst" every model costs one Kriging run,
#: so this is usually the limiting cost.
NSS_N_MODELS = 100

#: Models generated, shuttled and merged into the statistics at a time.
#: Memory is about NSS_CHUNK_SIZE * nm * 8 bytes for the chunk plus a few
#: temporary arrays of the same size.
NSS_CHUNK_SIZE = 25

#: Sigma levels of the quantile models.  Quantiles are computed at the
#: probabilities Phi(-k), 0.5, Phi(+k) of the standard normal distribution
#: for each k, e.g. (1.0, 2.0) -> 0.0228, 0.1587, 0.5, 0.8413, 0.9772.
#: (2.5/97.5 percent would be 1.96 sigma.)
NSS_SIGMAS = (1.0, 2.0)

#: Histogram bins per block for the streaming quantiles.  The quantile error
#: is at most one bin width, (hi - lo) / NSS_HIST_NBINS in log10(rho).  Memory
#: is nm * NSS_HIST_NBINS * 4 bytes.
NSS_HIST_NBINS = 256

#: (lo, hi) of the histograms in log10(rho), common to all blocks.  None =
#: derived from the first chunk (its min and max, padded by half its span,
#: at least 0.5 decades each side).  Values outside the range fall into the
#: first / last bin; the fraction is reported, and quantiles in those bins
#: are only bounds.
NSS_HIST_RANGE = None

#: Also write the mean model and the quantile models as resistivity-block
#: files next to MODEL_OUT (suffixes _mean, _q0228, ...).
NSS_WRITE_STAT_MODELS = True

#: All statistics (n, mean, std, quantiles, Gaussian bands, histogram range)
#: as a NumPy .npz archive.
NSS_STATS_FILE = WORK_DIR + "nss_stats.npz"

# ---------------------------------------------------------------------------
# Verbose output
# ---------------------------------------------------------------------------
OUT = True

#: When True (default) and this script is running inside Spyder, the
#: MOD_QC figure is also displayed inline (Spyder's Plots pane) via
#: plt.show(), in addition to being written to disk -- no change to what
#: gets saved. Has no effect outside Spyder; set False to disable even
#: under Spyder.
MOD_SHOW_IN_SPYDER = True

#: Detected once at import time; utl.runtime_env() returns 'spyder' when
#: running inside Spyder's IPython console (SPYDER_KERNEL env var /
#: spyder_kernels module), 'jupyter'/'ipython-*'/'python' otherwise.
_IN_SPYDER = (utl.runtime_env() == "spyder")
_SHOW_PLOTS = MOD_SHOW_IN_SPYDER and _IN_SPYDER
if _IN_SPYDER:
    print(f"Detected Spyder — inline figure display {'enabled' if _SHOW_PLOTS else 'disabled (MOD_SHOW_IN_SPYDER=False)'}.\n")

# ---------------------------------------------------------------------------
# Mesh (required for the QC slice plot)
# ---------------------------------------------------------------------------
MOD_MESH = WORK_DIR + "mesh.dat"

# --- Ocean / air handling (must match the inversion setup) ----------------
MOD_OCEAN     = None
MOD_AIR_RHO   = 1.0e9   # Ω·m  (region 0)
MOD_OCEAN_RHO = 0.25    # Ω·m  (region 1 when treated as ocean)

# ---------------------------------------------------------------------------
# QC slice plot — the nullspace-shuttled output model
# ---------------------------------------------------------------------------
#: Set True to plot MODEL_OUT after it is written.
MOD_QC      = False
MOD_QC_FILE = WORK_DIR + "nss_qc.pdf"

# ---------------------------------------------------------------------------
# Shared slice / plot parameters
# (identical config surface to femtic_gst_prep.py / femtic_rto_prep.py /
#  femtic_ens_post.py — used by MOD_QC below)
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
MOD_SITE_DAT    = WORK_DIR + "site.dat"
MOD_SITE_NAMES  = None   # list of names to plot, or None = all sites
#: Fallback: site number(s) from observe.dat (int or list of ints).
MOD_SITE_NUMBER = None

MOD_PLOT_SITES_MAPS   = True    # show markers on map panels
MOD_PLOT_SITES_SLICES = True    # show markers on curtain / plane panels
#: Max distance [m] from a curtain plane for a site to appear on it.
MOD_PROJECTION_DIST = 5.0    # km; None = show all sites on every panel

MOD_SITE_MARKER        = dict(marker="v", color="black", ms=8, zorder=10, label=None)
MOD_SITE_MARKER_SLICES = None
#: Extra point markers on map panels only (each dict: latlon, marker, color, ms, name).
MOD_MAP_MARKERS = []

# --- Slice specification ----------------------------------------------------
#: Slice positions accept plain floats (model-local m) or CRS-tagged tuples:
#:   (value, "utm") | (value, "latlon")
#: Depth z0 is always model-local metres (no CRS tagging).
MOD_SLICES = [
    dict(kind="map", z0=5.0),    # km
    dict(kind="map", z0=15.0),   # km
    dict(kind="ns",  x0=0.0),    # km
    dict(kind="ew",  y0=0.0),    # km
]
MOD_XLIM = None    # [xmin, xmax] model-local km; None = auto
MOD_YLIM = None    # [ymin, ymax] model-local km; None = auto
MOD_ZLIM = None    # [zmin, zmax] model-local km; None = auto

MOD_DPI         = 600            # figure DPI
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

#: Axis annotation font sizes, passed through to fviz.plot_model_slices
#: (used by the MOD_QC plot below). Defaults match plot_model_slices'
#: own defaults.
MOD_TICK_FONTSIZE  = 7    # axis tick labels, colourbar ticks
MOD_LABEL_FONTSIZE = 8    # axis labels, panel titles, colourbar label

# ===========================================================================
# Step 4 configuration — model perturbation mode
# ===========================================================================

#: ``"random"`` — uniform Gaussian placeholder (original behaviour).
#: ``"gst"``    — geostatistical perturbation via pilot-point Kriging.
PERTURB_MODE = "gst"

# ---------------------------------------------------------------------------
# GST model perturbation config  (used only when PERTURB_MODE == "gst")
# Mirrors the config block in femtic_gst_prep.py exactly.
# ---------------------------------------------------------------------------
if PERTURB_MODE == "gst":
    # Reference model and mesh (same files used for NSS template).
    GST_REF_MOD  = MODEL_TEMPLATE   # free-region structure source
    GST_MESH     = WORK_DIR + "mesh.dat"

    # --- Pilot-point placement -----------------------------------------------
    # MOD_PP_MODE: "random" | "fixed" | "mixed" | "extrema"
    #   "random"  — n_pp points drawn uniformly inside MOD_PP_BBOX each call.
    #   "fixed"   — locations from MOD_PP_COORDS (same every call, values vary).
    #   "mixed"   — MOD_PP_COORDS plus n_pp random fill points.
    #   "extrema" — seed at local log10(ρ) extrema of reference model (within
    #               MOD_PP_ROI), plus n_pp random fill points.
    GST_PP_MODE = "random"      # "random" | "fixed" | "mixed" | "extrema"

    # Number of random pilot points per shuttle realisation.
    # Recommended: 50–200 for typical 3-D MT survey volumes.
    GST_N_PP = 100

    # Bounding box for random pilot-point placement:
    #   [x_min, x_max, y_min, y_max, z_min, z_max]  (km, model-local, z positive-down)
    GST_PP_BBOX = [-50., 50.,   # easting  range (km)
                   -50., 50.,   # northing range (km)
                     0., 80.]   # depth     range (km)

    # Explicit pilot-point coordinates for "fixed" or "mixed" mode.
    # Shape: (N, 3) — columns: [easting, northing, depth], km, same
    # model-local convention as GST_PP_BBOX.
    GST_PP_COORDS = None   # e.g. np.array([[x, y, z], ...])  (km)

    # --- "extrema" mode ------------------------------------------------------
    # GST_PP_ROI: sub-volume [x_min,x_max,y_min,y_max,z_min,z_max] (km)
    #   restricting which free regions are eligible as extremum seeds.
    #   None = full extent.
    GST_PP_ROI           = None   # None = full extent; km
    GST_PP_EXTREMA_K     = 9      # neighbourhood size for extremum detection
    GST_PP_EXTREMA_WHICH = "both" # "both" | "minima" | "maxima"

    # --- Resistivity range ---------------------------------------------------
    # Pilot-point values drawn Uniform(GST_LOG_RHO_MIN, GST_LOG_RHO_MAX) in
    # log10(Ω·m).  Post-Kriging field clamped to the same interval.
    GST_LOG_RHO_MIN = 0.0    # log10(1 Ω·m)
    GST_LOG_RHO_MAX = 4.0    # log10(10 000 Ω·m)

    # --- Variogram model -----------------------------------------------------
    # GST_VARIO_MODEL: gstools covariance model class name (string).
    #   Common choices: "Spherical", "Gaussian", "Exponential", "Matern".
    #
    # GST_VARIO_RANGE: correlation length (km).  A 2-tuple
    #   (horizontal_range, vertical_range) sets geometric anisotropy.
    #   Recommended: h_range ≈ half survey aperture; v_range ≈ half target depth.
    #
    # GST_VARIO_SILL: sill (variance) in (log10 Ω·m)².  Typical 0.25–0.5.
    #
    # GST_VARIO_NUGGET: nugget in (log10 Ω·m)².  Keep ≤ 10 % of sill.
    #
    # GST_VARIO_ANGLES: rotation [α, β, γ] in degrees; None = axis-aligned.
    GST_VARIO_MODEL   = "Spherical"
    GST_VARIO_RANGE   = (8., 4.)         # (horizontal, vertical) in km
    GST_VARIO_SILL    = 0.5
    GST_VARIO_NUGGET  = 0.01
    GST_VARIO_ANGLES  = None   # [alpha, beta, gamma] deg; None = axis-aligned


# ===========================================================================
# Plotting helpers
# ===========================================================================

def _resolve_origin_and_sites():
    """Estimate UTM origin from MOD_SITE_DAT; collect site model-local coords.

    Mirrors the origin-resolution block in femtic_gst_prep.py /
    femtic_ens_post.py so all scripts behave identically, including the
    observe.dat / MOD_SITE_NUMBER fallback when MOD_SITE_DAT is absent.
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
        _obs_file = WORK_DIR + "observe.dat"
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

    Mirrors the plotting call in femtic_gst_prep.py / femtic_ens_post.py
    exactly, so the NSS QC figure uses the same options (CRS handling,
    site overlay, alpha/blanking, figure layout) as the other scripts.
    """
    if fviz is None:
        print("  plot_slice: femtic_viz not available -- skipping.")
        return

    _slices_resolved = fem.resolve_slice_positions(
        _slices_km_to_m(MOD_SLICES), utm_zone, utm_north,
        utm_e, utm_n, utm_lat, utm_lon,
        verbose=OUT,
    )
    fviz.plot_model_slices(
        model_file          = block_file,
        mesh_file           = MOD_MESH,
        slices              = _slices_resolved,
        cmap                = MOD_CMAP,
        clim                = MOD_CLIM,
        xlim                = _lim_km_to_m(MOD_XLIM),
        ylim                = _lim_km_to_m(MOD_YLIM),
        zlim                = _lim_km_to_m(MOD_ZLIM),
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
        projection_dist     = _km_to_m(MOD_PROJECTION_DIST),
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
        tick_fontsize       = MOD_TICK_FONTSIZE,
        label_fontsize      = MOD_LABEL_FONTSIZE,
        alpha_file          = MOD_ALPHA_FILE,
        alpha_mode          = MOD_ALPHA_MODE,
        alpha_blank_thresh  = MOD_ALPHA_BLANK_THRESH,
        plot_file           = pdf_file,
        dpi                 = MOD_DPI,
        show                = _SHOW_PLOTS,
        out                 = OUT,
    )
    if OUT:
        print(f"  saved -> {pdf_file}")


# ===========================================================================
# Input helpers (steps 1-2)
# ===========================================================================

#: ResistivityBlock::TypeOfResistivityBlock codes, as stored in
#: /model/blocks["type"] of results_iterX.h5, that are NOT inverted for:
#: 1 = FIXED_AND_ISOLATED, 2 = FIXED_AND_CONSTRAINED.  The columns of the
#: Jacobian are the remaining (free) blocks in ascending blockID order.
_FIXED_BLOCK_TYPES = (1, 2)


def _read_results(results_file: str):
    """Read the final model and an approximate misfit from results_iterX.h5.

    Uses ``fem.read_results_hdf5``.  The model vector is log10(rho) of the
    free blocks in ascending blockID order, which is the order of the
    Jacobian columns (modelID order).

    Parameters
    ----------
    results_file : str
        Path to ``results_iterX.h5`` (needs ``/model``; ``/data`` is used for
        the misfit if present).

    Returns
    -------
    model : numpy.ndarray, shape (nm,)
        log10(rho) of the free blocks.
    iter_num : int
        Iteration number stored in the file (-1 if absent).
    rms : tuple or None
        ``(rms, n_used, n_total)``: error-weighted RMS of (observed -
        calculated) over all finite real/imaginary entries with a positive
        error.  Approximate -- it is not FEMTIC's own nRMS definition.
        None if the file has no usable data rows.
    """
    res = fem.read_results_hdf5(results_file, out=OUT)
    mod = res["model"]
    if mod is None:
        raise KeyError(f"{results_file}: no /model group found.")
    if mod["anisotropic"]:
        raise NotImplementedError(
            f"{results_file}: anisotropic (v5) model -- the model vector is "
            "not one value per free block, not supported here."
        )

    blocks = mod["blocks"]
    free = ~np.isin(blocks["type"], _FIXED_BLOCK_TYPES)
    model = np.log10(np.asarray(blocks["resistivity"], dtype=float)[free])

    rms = None
    dat = res["data"]
    if dat is not None and dat["nRows"] > 0:
        rows = dat["rows"]
        obs = np.concatenate([rows["re_val"], rows["im_val"]]).astype(float)
        cal = np.concatenate([rows["cal_re"], rows["cal_im"]]).astype(float)
        err = np.concatenate([rows["re_err"], rows["im_err"]]).astype(float)
        ok = np.isfinite(obs) & np.isfinite(cal) & np.isfinite(err) & (err > 0.0)
        if ok.any():
            r = (obs[ok] - cal[ok]) / err[ok]
            rms = (float(np.sqrt(np.mean(r * r))), int(ok.sum()), int(ok.size))

    return model, int(mod["iterNum"]), rms


def _read_jacobian(jac_file: str, weighted=None):
    """Read the data-weighted Jacobian Js from exchange.h5 or a legacy jacobian.h5.

    ``exchange.h5``: ``/jacobian/values`` (nData, nModel) holds
    J = Cd^(-1/2) dF/dm, already divided by the data errors (attribute
    ``weighted`` = 1), with the per-row errors in ``/jacobian/data_errors``.
    Rows follow the inversion's global data ordering (not the row order of
    ``/data/data`` in results_iterX.h5); columns are modelIDs.

    Legacy ``jacobian.h5``: root-level dataset ``/jacobian`` (and optionally
    ``/data_errors``).

    Parameters
    ----------
    jac_file : str
        Path to the Jacobian file.
    weighted : bool or None
        True: the file's Jacobian is already error-weighted, Js = J.
        False: Js = diag(1/data_errors) @ J.
        None: use the file's ``weighted`` attribute; exchange.h5 defaults to
        True, a legacy file without the attribute raises ValueError.

    Returns
    -------
    Js : numpy.ndarray, shape (nd, nm), float64
    iter_num : int or None
        Iteration number stored in the file, if any.
    info : str
        One-line description of what was read and how it was treated.
    """
    with h5py.File(jac_file, "r") as hf:
        if "jacobian" not in hf:
            raise KeyError(f"{jac_file}: no 'jacobian' group or dataset found.")
        if isinstance(hf["jacobian"], h5py.Group):
            g = hf["jacobian"]
            if "values" not in g:
                raise KeyError(f"{jac_file}: /jacobian has no 'values' dataset.")
            jac = np.asarray(g["values"][()], dtype=float)
            err = (np.asarray(g["data_errors"][()], dtype=float)
                   if "data_errors" in g else None)
            meta = dict(g["metadata"].attrs) if "metadata" in g else {}
            layout, default_weighted = "exchange.h5", True
        else:
            ds = hf["jacobian"]
            jac = np.asarray(ds[()], dtype=float)
            err = (np.asarray(hf["data_errors"][()], dtype=float)
                   if "data_errors" in hf else None)
            meta = dict(hf.attrs)
            meta.update(dict(ds.attrs))
            if "metadata" in hf:
                meta.update(dict(hf["metadata"].attrs))
            layout, default_weighted = "legacy jacobian.h5", None

    if jac.ndim != 2:
        raise ValueError(f"{jac_file}: Jacobian is {jac.ndim}-D, expected 2-D.")
    if "nData" in meta and "nModel" in meta:
        if jac.shape != (int(meta["nData"]), int(meta["nModel"])):
            raise ValueError(
                f"{jac_file}: Jacobian {jac.shape} inconsistent with metadata "
                f"nData={meta['nData']}, nModel={meta['nModel']}."
            )
    if err is not None and err.shape != (jac.shape[0],):
        raise ValueError(
            f"{jac_file}: data_errors {err.shape} does not match "
            f"{jac.shape[0]} Jacobian rows."
        )

    if weighted is None:
        flag = meta.get("weighted", default_weighted)
        if flag is None:
            raise ValueError(
                f"{jac_file}: cannot tell whether the Jacobian is divided by "
                "the data errors (no 'weighted' attribute); set JAC_WEIGHTED "
                "to True or False."
            )
        weighted = bool(flag)

    if weighted:
        how = "already error-weighted, used as is"
    else:
        if err is None or not np.all(err > 0.0):
            raise ValueError(
                f"{jac_file}: JAC_WEIGHTED is False but positive data_errors "
                "are not available to weight the Jacobian."
            )
        jac /= err[:, np.newaxis]
        how = "divided by the data errors"

    iter_num = int(meta["iterNum"]) if "iterNum" in meta else None
    return jac, iter_num, f"{layout}, {how}"


# ===========================================================================
# Step 1 - Read FEMTIC results and Jacobian
# ===========================================================================

print("=" * 72)
print("Step 1: Reading FEMTIC results (model, data) and Jacobian")
print("=" * 72)

t0_total = time.perf_counter()
t0 = time.perf_counter()

model, iter_res, rms_res = _read_results(RESULTS_FILE)
Js, iter_jac, jac_info = _read_jacobian(JACOBIAN_FILE, weighted=JAC_WEIGHTED)

nd, nm = Js.shape

if nm != model.size:
    raise ValueError(
        f"Jacobian has {nm} columns but the results file has {model.size} "
        "free blocks -- the two files do not belong to the same run/iteration."
    )

# The shuttled model is merged back into MODEL_TEMPLATE, so its free-block
# count must match as well.
_m_tmpl = fem.read_model(
    model_file=MODEL_TEMPLATE, model_trans="log10", ocean=MOD_OCEAN, out=False)
if _m_tmpl.size != nm:
    raise ValueError(
        f"MODEL_TEMPLATE has {_m_tmpl.size} free regions but the results "
        f"file / Jacobian have {nm}."
    )
del _m_tmpl

if iter_jac is not None and iter_res >= 0 and iter_jac != iter_res:
    print(f"  WARNING: results file is iteration {iter_res} but the Jacobian "
          f"is from iteration {iter_jac} -- the null space refers to the "
          "Jacobian's model.")

if OUT:
    print(f"  model (free blocks) : {model.shape}  (log10 rho)")
    print(f"  Jacobian            : {Js.shape}  [{jac_info}]")
    print(f"  iteration           : results {iter_res}, Jacobian "
          f"{iter_jac if iter_jac is not None else 'n/a'}")
    print(f"  nd={nd}, nm={nm}")
    print(f"  elapsed             : {time.perf_counter() - t0:.2f} s")


# ===========================================================================
# Step 2 - Data-weighted Jacobian and misfit summary
# ===========================================================================

print("\n" + "=" * 72)
print("Step 2: Data-weighted Jacobian Js and misfit summary")
print("=" * 72)

t0 = time.perf_counter()

# Js was formed in step 1 (FEMTIC writes J = Cd^(-1/2) dF/dm already weighted).
if OUT:
    print(f"  Js shape : {Js.shape}")
    print(f"  ||Js||_F : {np.linalg.norm(Js):.4e}")
    if rms_res is not None:
        print(f"  RMS (approx., from results file) : {rms_res[0]:.4f}  "
              f"({rms_res[1]} of {rms_res[2]} real/imag entries used)")
    print(f"  elapsed  : {time.perf_counter() - t0:.2f} s")


# ===========================================================================
# Step 3 — Randomised SVD of Js
# ===========================================================================

print("\n" + "=" * 72)
print("Step 3: Randomised SVD of Js")
print("=" * 72)

t0 = time.perf_counter()

rank = min(RSVD_RANK, nd, nm)

U, S, Vt = inv.rsvd(
    Js,
    rank=rank,
    n_oversamples=RSVD_OVERSAMPLES,
    n_subspace_iters=RSVD_SUBSPACE_ITERS,
)

# U  : (nd, rank)  — left singular vectors  (data space)
# S  : (rank,)     — singular values
# Vt : (rank, nm)  — right singular vectors transposed (model space)

if OUT:
    print(f"  Decomposition: U {U.shape}, S {S.shape}, Vt {Vt.shape}")
    print(f"  s[0]  = {S[0]:.4e}  (largest)")
    print(f"  s[-1] = {S[-1]:.4e}  (smallest in truncated set)")
    # Determine effective rank at the chosen threshold
    s_thresh = NSS_SV_THRESH * S[0]
    r_eff = int(np.sum(S >= s_thresh))
    print(f"  Effective rank at threshold {NSS_SV_THRESH:.1e}: {r_eff} / {rank}")
    print(f"  elapsed  : {time.perf_counter() - t0:.2f} s")


# ===========================================================================
# Step 4 — Model perturbation
# ===========================================================================
#
# Two modes controlled by PERTURB_MODE (set in the configuration section):
#
#   "random" — uniform Gaussian placeholder (original behaviour).
#              Replace the body of _make_perturbation_random() with any
#              prior-based or exploratory modification in log10(ρ) space.
#              The amplitude is subsequently scaled by NSS_AMPLITUDE in step 5.
#
#   "gst"    — geostatistical perturbation via pilot-point Ordinary Kriging
#              (ens.generate_gst_model_ensemble).  A single Kriged model is
#              generated in a temporary directory; the perturbation is the
#              delta between that model and the reference model in log10(ρ)
#              space.  The null-space shuttle then projects out any data-
#              sensitive component so the predicted data remain unchanged.

def _make_perturbation_random(m: np.ndarray) -> np.ndarray:
    """Return a random Gaussian perturbation in log10(ρ) space (placeholder).

    Parameters
    ----------
    m : numpy.ndarray, shape (nm,)
        Current final model in log10(ρ) (free regions only).  Not used in the
        default implementation but available for amplitude scaling.

    Returns
    -------
    dm : numpy.ndarray, shape (nm,)
        Desired model perturbation (same shape as m).
    """
    # -----------------------------------------------------------------------
    # *** EDIT BELOW THIS LINE to replace the Gaussian placeholder ***
    # -----------------------------------------------------------------------

    # Uses the shared module-level `rng` (see RANDOM_SEED in the
    # configuration section) so this placeholder respects the same
    # reproducibility switch as the "gst" perturbation mode: fresh OS
    # entropy by default, or a fixed draw when RANDOM_SEED is set.
    # (Previously hardcoded to np.random.default_rng(seed=0) -- always the
    # same draw regardless of RANDOM_SEED, and not reproducible-by-choice.)
    dm = rng.standard_normal(m.size)

    # -----------------------------------------------------------------------
    # *** EDIT ABOVE THIS LINE ***
    # -----------------------------------------------------------------------
    return dm


def _make_perturbation_gst(m_ref: np.ndarray) -> np.ndarray:
    """Generate a geostatistical model perturbation via pilot-point Kriging.

    Calls ``ens.generate_gst_model_ensemble`` for a single realisation in a
    temporary subdirectory inside WORK_DIR.  The perturbation is the difference
    between the Kriged model and the reference model in log10(ρ) space::

        dm = m_gst - m_ref

    The null-space shuttle in step 5 projects out any data-sensitive component.

    Parameters
    ----------
    m_ref : numpy.ndarray, shape (nm,)
        Reference model in log10(ρ) (free regions only), as read from HDF5.

    Returns
    -------
    dm : numpy.ndarray, shape (nm,)
        Perturbation in log10(ρ) space (same shape as m_ref).
    """
    import tempfile, shutil

    tmp_base = os.path.join(WORK_DIR, "_nss_gst_tmp_")

    try:
        # generate_gst_model_ensemble writes member 0 into ``tmp_base + "0/"``
        ens.generate_gst_model_ensemble(
            alg              = "gst",
            dir_base         = tmp_base,
            n_samples        = 1,
            fromto           = None,
            ref_mod_file     = GST_REF_MOD,
            mesh_file        = GST_MESH,
            pp_mode          = GST_PP_MODE,
            n_pp             = GST_N_PP,
            pp_bbox          = _lim_km_to_m(GST_PP_BBOX),
            pp_coords        = _coords_km_to_m(GST_PP_COORDS),
            pp_roi           = _lim_km_to_m(GST_PP_ROI),
            pp_extrema_k     = GST_PP_EXTREMA_K,
            pp_extrema_which = GST_PP_EXTREMA_WHICH,
            log_rho_min      = GST_LOG_RHO_MIN,
            log_rho_max      = GST_LOG_RHO_MAX,
            vario_model      = GST_VARIO_MODEL,
            vario_range      = _range_km_to_m(GST_VARIO_RANGE),
            vario_sill       = GST_VARIO_SILL,
            vario_nugget     = GST_VARIO_NUGGET,
            vario_angles     = GST_VARIO_ANGLES,
            output_target    = "resistivity_block",
            resistivity_file = "resistivity_block_iter0.dat",
            reference_file   = "referencemodel.dat",
            rng              = rng,
            out              = OUT,
        )

        # Read back the Kriged initial model from the temporary member directory
        gst_block_file = tmp_base + "0/resistivity_block_iter0.dat"
        # read_model returns the free-region log10(rho) vector directly
        m_gst = fem.read_model(
            model_file=gst_block_file, model_trans="log10",
            ocean=MOD_OCEAN, out=False)

    finally:
        # Always clean up the temporary tree
        if os.path.isdir(tmp_base + "0"):
            shutil.rmtree(tmp_base + "0", ignore_errors=True)

    return m_gst - m_ref


def _draw_perturbations(k: int) -> np.ndarray:
    """Draw ``k`` raw perturbations with the mode set in PERTURB_MODE.

    Returns an array of shape (k, nm); rows are independent draws.
    """
    if PERTURB_MODE == "gst":
        draw = _make_perturbation_gst
    elif PERTURB_MODE == "random":
        draw = _make_perturbation_random
    else:
        raise ValueError(f"Unknown PERTURB_MODE: '{PERTURB_MODE}'. "
                         "Choose 'random' or 'gst'.")
    return np.stack([draw(model) for _ in range(k)])


# ===========================================================================
# Step 5 — Nullspace shuttle
# ===========================================================================

def _nullspace_shuttle(
    dm: np.ndarray,
    Vt: np.ndarray,
    S: np.ndarray,
    *,
    sv_thresh: float = 1.0e-3,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Project a model perturbation onto the null space of the scaled Jacobian.

    Given the truncated SVD  Js ≈ U S Vt, the data-space projector onto the
    row space of Js is  Vr @ Vr.T  where Vr = Vt[r_eff].T.  The null-space
    projector is  I - Vr @ Vr.T.

    The shuttle perturbation is::

        δm_null = amplitude * (I - Vr @ Vr.T) @ dm

    Adding δm_null to the current model produces a model with (approximately)
    identical predicted data.

    Parameters
    ----------
    dm : numpy.ndarray, shape (nm,) or (n_models, nm)
        Raw model perturbation(s); a 2-D array is projected row by row with
        one matrix product (used for chunks).
    Vt : numpy.ndarray, shape (rank, nm)
        Right singular vectors (transposed) from the rSVD of Js.
    S : numpy.ndarray, shape (rank,)
        Singular values from the rSVD.
    sv_thresh : float
        Fraction of s[0] below which a singular vector is treated as null.
    amplitude : float
        Scale factor applied to the null-space perturbation.

    Returns
    -------
    dm_null : numpy.ndarray, same shape as dm
        Null-space component of dm scaled by amplitude.
    r_eff : int
        Number of singular vectors treated as row space.
    """
    s_thresh = sv_thresh * S[0]
    r_eff = int(np.sum(S >= s_thresh))

    Vr = Vt[:r_eff].T             # shape (nm, r_eff) — row-space basis

    # Null-space projection: remove row-space component
    dm_row  = (dm @ Vr) @ Vr.T    # component in row space  (data-sensitive);
                                  # works for dm of shape (nm,) and (k, nm)
    dm_null = dm - dm_row          # component in null space (data-invisible)

    return amplitude * dm_null, r_eff


print("\n" + "=" * 72)
print(f"Step 5: Nullspace shuttle ensemble, {NSS_N_MODELS} models in chunks of "
      f"{NSS_CHUNK_SIZE}  [PERTURB_MODE = '{PERTURB_MODE}']")
print("=" * 72)

t0 = time.perf_counter()

if NSS_N_MODELS < 1 or NSS_CHUNK_SIZE < 1:
    raise ValueError("NSS_N_MODELS and NSS_CHUNK_SIZE must be >= 1")


def _std_normal_cdf(k: float) -> float:
    return 0.5 * (1.0 + math.erf(k / math.sqrt(2.0)))


# Sigma levels -k..-1, 0, +1..+k and the corresponding probabilities.
q_sigma = np.array(sorted({0.0} | {-abs(k) for k in NSS_SIGMAS}
                          | {abs(k) for k in NSS_SIGMAS}))
q_probs = np.array([_std_normal_cdf(k) for k in q_sigma])
q_tags = [f"q{int(round(p * 1.0e4)):04d}" for p in q_probs]   # q0228, ..., q9772

n_acc = 0                                 # streaming (n, mean, M2) per block
mean_acc = np.zeros(nm)
m2_acc = np.zeros(nm)
counts = None                             # (nm, NSS_HIST_NBINS) histograms
h_lo = h_hi = None
n_clipped = 0
model_first = None                        # first shuttled model -> MODEL_OUT
check_null = None                         # max ||Js dm_null|| of first chunk
check_row = None

for i0 in range(0, NSS_N_MODELS, NSS_CHUNK_SIZE):
    k = min(NSS_CHUNK_SIZE, NSS_N_MODELS - i0)
    t_chunk = time.perf_counter()

    dm_raw = _draw_perturbations(k)                       # (k, nm)
    dm_null, r_eff = _nullspace_shuttle(
        dm_raw, Vt, S, sv_thresh=NSS_SV_THRESH, amplitude=NSS_AMPLITUDE)
    X = model[np.newaxis, :] + dm_null                    # (k, nm), log10(rho)

    if i0 == 0:
        model_first = X[0].copy()
        # Verification on the first chunk: predicted-data change ~ 0.
        check_null = float(np.max(np.linalg.norm(Js @ dm_null.T, axis=0)))
        check_row = float(np.mean(np.linalg.norm(
            dm_raw - dm_null / NSS_AMPLITUDE, axis=1)))
        if NSS_HIST_RANGE is None:
            span = max(float(X.max() - X.min()), 1.0)
            pad = max(0.5 * span, 0.5)
            h_lo, h_hi = float(X.min() - pad), float(X.max() + pad)
        else:
            h_lo, h_hi = (float(v) for v in NSS_HIST_RANGE)
        counts = np.zeros((nm, NSS_HIST_NBINS), dtype=np.uint32)
        if OUT:
            print(f"  Effective rank used for projection : {r_eff}")
            print(f"  mean ||dm_row|| (first chunk)       : {check_row:.4e}")
            print(f"  max ||Js @ dm_null|| (should be ~0) : {check_null:.4e}")
            print(f"  histogram range [{h_lo:.3f}, {h_hi:.3f}] log10(rho), "
                  f"{NSS_HIST_NBINS} bins, bin width "
                  f"{(h_hi - h_lo) / NSS_HIST_NBINS:.4f}, "
                  f"{counts.nbytes / 1.0e6:.0f} MB")

    # Chunk moments, merged into the running accumulator (Chan et al. 1983).
    n_c, mean_c, m2_c = ens.batch_moments(X, axis=0)
    n_acc, mean_acc, m2_acc = ens.chan_merge(
        n_acc, mean_acc, m2_acc, n_c, mean_c, m2_c)
    # Chunk histograms (counts simply add).
    n_clipped += ens.hist_update(counts, X, h_lo, h_hi)

    if OUT:
        print(f"  models {i0 + 1:>6d}-{i0 + k:<6d} of {NSS_N_MODELS}: "
              f"{time.perf_counter() - t_chunk:.2f} s")

model_nss = model_first                   # written to MODEL_OUT (and MOD_QC)

# Statistics from the accumulators.
if n_acc > 1:
    std_acc = np.sqrt(ens.moments_variance(n_acc, m2_acc, ddof=1))
    quant = ens.hist_quantiles(counts, h_lo, h_hi, q_probs)       # (nq, nm)
    gauss = mean_acc[np.newaxis, :] + q_sigma[:, np.newaxis] * std_acc[np.newaxis, :]
    clip_frac = n_clipped / float(n_acc * nm)
else:
    std_acc = quant = gauss = None
    clip_frac = 0.0

if OUT:
    print(f"  Models accumulated : {n_acc}")
    print(f"  model_nss (first) range : [{model_nss.min():.3f}, {model_nss.max():.3f}]")
    if std_acc is not None:
        print(f"  std (log10 rho) over blocks : median {np.median(std_acc):.4f}, "
              f"max {std_acc.max():.4f}")
        for j, (ks, pj) in enumerate(zip(q_sigma, q_probs)):
            dq = np.abs(quant[j] - gauss[j])
            print(f"  {ks:+.1f} sigma  p = {pj:.4f}  quantile vs Gaussian "
                  f"mean +/- k*std: median |diff| {np.nanmedian(dq):.4f}")
        print(f"  values outside histogram range : {100.0 * clip_frac:.4f} %")
        if clip_frac > 1.0e-3:
            print("  WARNING: more than 0.1 % of the values were clipped to the "
                  "end bins; set NSS_HIST_RANGE wider.")
    else:
        print("  Only one model: no variance or quantiles.")
    print(f"  elapsed         : {time.perf_counter() - t0:.2f} s")


# ===========================================================================
# Write modified model
# ===========================================================================

print("\n" + "=" * 72)
print("Writing nullspace-shuttled model")
print("=" * 72)

t0 = time.perf_counter()

# ``fem.insert_model`` merges the free-region vector back into the template
# structure preserving header, bounds, fixed regions, air, and ocean.
fem.insert_model(
    template=MODEL_TEMPLATE,
    model=model_nss,
    model_file=MODEL_OUT,
    ocean=MOD_OCEAN,
    air_rho=MOD_AIR_RHO,
    ocean_rho=MOD_OCEAN_RHO,
    out=OUT,
)

print(f"  Written : {MODEL_OUT}  (first of {n_acc} shuttled models)")

if std_acc is not None:
    stem, ext = os.path.splitext(MODEL_OUT)
    if NSS_WRITE_STAT_MODELS:
        stat_models = [("mean", mean_acc)] + list(zip(q_tags, quant))
        for tag, vec in stat_models:
            f_stat = f"{stem}_{tag}{ext}"
            fem.insert_model(
                template=MODEL_TEMPLATE,
                model=vec,
                model_file=f_stat,
                ocean=MOD_OCEAN,
                air_rho=MOD_AIR_RHO,
                ocean_rho=MOD_OCEAN_RHO,
                out=False,
            )
            print(f"  Written : {f_stat}")

    np.savez_compressed(
        NSS_STATS_FILE,
        n=n_acc, mean=mean_acc, m2=m2_acc, std=std_acc,
        sigma=q_sigma, probs=q_probs, quantiles=quant, gauss=gauss,
        hist_lo=h_lo, hist_hi=h_hi, hist_nbins=NSS_HIST_NBINS,
        clip_fraction=clip_frac, reference_model=model,
        r_eff=r_eff, amplitude=NSS_AMPLITUDE,
        info="log10(rho) of the free blocks, ascending blockID order; "
             "quantiles[i] belongs to probs[i] / sigma[i]",
    )
    print(f"  Written : {NSS_STATS_FILE}")

print(f"  elapsed : {time.perf_counter() - t0:.2f} s")

# ===========================================================================
# QC slice plot of the nullspace-shuttled model
# ===========================================================================

if MOD_QC:
    if fviz is None:
        print("\n  MOD_QC: femtic_viz not available -- skipping.")
    else:
        print("\n" + "=" * 72)
        print("QC slice plot of nullspace-shuttled model")
        print("=" * 72)
        (utm_e, utm_n, utm_lat, utm_lon,
         utm_zone, utm_north, site_xys, obs_coords_only) = _resolve_origin_and_sites()
        _plot_slice(
            block_file      = MODEL_OUT,
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

print(f"\n  Total elapsed : {time.perf_counter() - t0_total:.2f} s")


# ---------------------------------------------------------------------------
# Parameter summary
# ---------------------------------------------------------------------------
utl.write_param_summary(fname)
