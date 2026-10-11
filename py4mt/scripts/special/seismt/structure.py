#!/usr/bin/env python3
"""
structure.py — Structural-coupling diagnostics between pairs (and triples)
of interpolated fields on the joint grid produced by ``interpolate.py``.

Renamed from ``cross_gradient.py`` (same pipeline position, same input
file, broadened scope: several structural-coupling measures from the
joint-inversion literature, not just the cross-gradient).

    precompute.py -> interpolate.py -> {SITE_PREFIX}_interp_<method>.nc
                                                     |
                                        structure.py -> figures

Reads the same ``INTERP_FILE`` NetCDF as ``plot_joint.py`` / ``cluster.py``
and is equally agnostic to which interpolation method produced it. For each
configured group of fields it computes one or more structural-coupling
diagnostics and plots them as depth-slice maps and (where wired in, see
"Known limitations" below) vertical cross-sections -- same styling
conventions (basemap, colorbar, marker/label style, VSLICES schema,
isoline toggles) as the rest of the plotting pipeline; see
``README_plot_joint.md`` for the shared conventions this script reuses.

Methods implemented (all computed from the same real-coordinate 3-D
gradients via ``gradient_components()``, so all are plottable on the same
mesh)
------------------------------------------------------------------------
Let g1 = grad(m1), g2 = grad(m2) be the local gradient vectors of two
fields on the joint grid, and theta the angle between them.

  * Cross-gradient (Gallardo & Meju 2003, 2004)
        t = g1 x g2                              -- STRUCTURE_PAIRS, N=2
        |t| = |g1||g2| sin(theta)
    Zero wherever the two fields' local structure is parallel/anti-
    parallel, regardless of magnitude, sign, or units.

  * Cosine similarity / "directed" dot-product constraint
    (Molodtsov, Troyan, Roslov & Zerilli 2013)   -- STRUCTURE_PAIRS, N=2
        cos(theta) = (g1 . g2) / (|g1||g2|)  in [-1, 1]
    Signed: requires no cross-product, but the sign carries a priori
    information about which correlation direction ("directed") is
    expected between the two properties.

  * Squared cosine similarity (Shi, Yu, Zhao, Zhang & Yang 2018)
                                                  -- STRUCTURE_PAIRS, N=2
        cos^2(theta)  in [0, 1]
    Sign-free improvement on Molodtsov et al.'s constraint (no a priori
    correlation direction needed, at the cost of a possible sign
    ambiguity/singularity the original paper discusses).
    Identically, since sin^2 + cos^2 = 1 for the same angle:
        cos^2(theta) == 1 - [normalised cross-gradient]^2
    (a built-in numerical cross-check between the two families of
    measures above, both computed from the same g1/g2 here).

  * Gramian structural-coupling determinant (Zhdanov, Gribenko & Wilson
    2012)                                        -- GRAMIAN_GROUPS, N=2 or 3
        det(Gram(g_1, ..., g_N)),  Gram_ij = g_i . g_j
    At N=2 this is algebraically identical to |t|^2 above (Lagrange's
    identity: det(Gram(g1,g2)) = |g1|^2|g2|^2 - (g1.g2)^2 = |g1 x g2|^2),
    verified in earlier work on this project to ~1e-15 -- included here
    mainly for the N=3 case, which is genuinely novel (a squared scalar
    triple product measuring whether THREE fields' gradients are
    coplanar, not just whether two are parallel). N>3 needs a general
    NxN cofactor expansion that was not built here -- raises ValueError
    rather than guessing at one.

  * Windowed mutual information (optional, off by default; inspired by
    Mandolesi & Jones 2014, who used a *global*, whole-model mutual
    information as a joint-inversion objective, not a spatial diagnostic
    map) -- STRUCTURE_PAIRS, N=2, ENABLE_MUTUAL_INFO
    Unlike everything above, this compares the two fields' *values*
    (log10-transformed per LOG_FIELDS, same convention as the gradients)
    in a local sliding window via histogram-based density estimation,
    not their gradients -- it can register structural correlation even
    where both fields are locally smooth. The window size and bin count
    are free parameters with no literature- or pipeline-verified default
    for this project's grids; MI_WINDOW/MI_BINS are flagged as such
    rather than tuned/guessed at, and the feature defaults OFF. See
    ``windowed_mutual_information_2d()``. Only implemented for
    depth-slice maps (not sections, see below), and only for pairs
    (bivariate MI; no attempt made at a 3-way generalisation here).

Further coupling measures considered but not implemented here
---------------------------------------------------------------
Flagged rather than added speculatively -- these fit a different
computational pattern (inversion regularisers or statistical-distance
measures needing choices this script's config would have to invent) and
were left out of scope for this diagnostic-plotting stage:

  * Joint total variation (JTV) (Haber & Holtzman Gazit 2013, Surv.
    Geophys., 34, 675-695, doi:10.1007/s10712-013-9232-4) -- an
    inversion *regulariser* (couples the models by penalising the sum
    sqrt(|grad m1|^2 + |grad m2|^2)), not naturally a per-point
    diagnostic map the way the measures above are.
  * Variation of information (VI) -- introduced for this application in
    Moorkamp (2021), "Joint inversion of gravity and magnetotelluric
    data from the Ernest Henry IOCG deposit with a variation of
    information constraint", First Int'l Meeting for Applied Geoscience
    & Energy, SEG, pp. 1711-1715 (a conference proceedings paper; no DOI
    could be confirmed for it, flagged rather than fabricated), with the
    full derivation (including dVI/dm) given in Moorkamp (2022),
    "Deciphering the state of the lower crust and upper mantle with
    multi-physics inversion", Geophys. Res. Lett., 49(9), e2021GL096336,
    doi:10.1029/2021GL096336) and joint entropy-based coupling -- both are,
    like MI, global/statistical measures over the whole model rather
    than gradient-based local ones, and would need the same kind of
    windowing choice flagged for MI above.

References
    Gallardo, L. A., and M. A. Meju (2003), Characterization of
    heterogeneous near-surface materials by joint 2D inversion of DC
    resistivity and seismic data, Geophys. Res. Lett., 30(13), 1658,
    doi:10.1029/2003GL017370.

    Gallardo, L. A., and M. A. Meju (2004), Joint two-dimensional DC
    resistivity and seismic travel time inversion with cross-gradients
    constraints, J. Geophys. Res.-Solid Earth, 109, B03311,
    doi:10.1029/2003JB002716.

    Molodtsov, D. M., V. N. Troyan, Y. V. Roslov, and A. Zerilli (2013),
    Joint inversion of seismic traveltimes and magnetotelluric data with
    a directed structural constraint, Geophysical Prospecting, 61(6),
    1218-1228, doi:10.1111/1365-2478.12060.

    Shi, B., P. Yu, C. Zhao, L. Zhang, and H. Yang (2018), Linear
    correlation constrained joint inversion using squared cosine
    similarity of regional residual model vectors, Geophysical Journal
    International, 215(2), 1291-1307, doi:10.1093/gji/ggy336.

    Zhdanov, M. S., A. V. Gribenko, and G. Wilson (2012), Generalized
    joint inversion of multimodal geophysical data using Gramian
    constraints, Geophysical Research Letters, 39(9), L09301,
    doi:10.1029/2012GL051233.

    Mandolesi, E., and A. G. Jones (2014), Magnetotelluric inversion
    based on mutual information, Geophysical Journal International,
    199(1), 242-252, doi:10.1093/gji/ggu258.

Scope / known limitations
-------------------------
Only ``TARGET_GRID == "joint"`` (a genuinely regular UTM-km grid, see
``interpolate.py``) is supported. The reused seismic-tomography native
grid (``TARGET_GRID == "seismic"``) is only structured in (row, col) index
space, not orthogonal in UTM space, so a coordinate-exact 3-D gradient
(the same standard this pipeline holds gradients to elsewhere -- see
``gradient_magnitude()`` in ``plot_dens.py`` et al.) cannot be computed
from index-space finite differences without assuming near-orthogonality.
Rather than silently approximating, this script raises a clear error in
that case. Build ``INTERP_FILE`` with ``TARGET_GRID="joint"`` instead.

Three integration seams, flagged rather than guessed at:
  1. Basemap/marker/label styling -- see plot_map()/plot_section()
     docstrings.
  2. Vertical-section profile projection -- ``_profile_line()`` is a
     stub; see its docstring.
  3. Windowed mutual information is map-only (see above); wiring a
     along-profile version through ``_profile_line()`` once (2) is
     resolved would be a small follow-up, not attempted here.

Authors: Svetlana Byrdina (SMB), Volker Rath (DIAS)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.interpolate import RegularGridInterpolator

import tomomt

# ---------------------------------------------------------------------------
# Site / interpolated-grid selection
# ---------------------------------------------------------------------------

SITE_PREFIX = "ubinas_small"  # "tacna" or "ubinas_small" -- must match precompute.py/interpolate.py

# Explicit path, or None to auto-pick the newest {SITE_PREFIX}_interp_*.nc
# in the current directory (mirrors plot_joint.py / cluster.py behaviour).
INTERP_FILE = "../precompute/ubinas_small/ubinas_interp_nearest.nc"

# ---------------------------------------------------------------------------
# Field groups
# ---------------------------------------------------------------------------
# Pairwise measures (cross-gradient, cosine similarity, squared cosine
# similarity, optional windowed mutual information). Each tuple is
# (field_a, field_b, label). field_a/field_b must be variable names present
# in INTERP_FILE (e.g. "rho_femtic", "vp", "vs", "vpvs", "dens"). label is used in
# titles/filenames; if None, "{field_a}-{field_b}" is used instead.
STRUCTURE_PAIRS = [
    #("rho_femtic", "vs", None),
    ("rho_femtic", "dens", None),
    ("rho_femtic", "vp", None), 
]

# Gramian groups (Zhdanov et al. 2012): 2 or 3 fields per group. Each tuple
# is (fields, label), fields a tuple of 2 or 3 variable names. N=2 is
# algebraically equivalent to the squared cross-gradient magnitude (see
# module docstring) and is included mainly for cross-checking; N=3 is the
# genuinely novel case. Which fields belong in a genuine 3-field group is a
# scientific choice specific to this project, not something to default
# silently -- left commented below rather than guessed at.
GRAMIAN_GROUPS = [
    #(("rho_femtic", "vs"), None),
    (("rho_femtic", "dens"), None),
    (("rho_femtic", "vps"), None),
    # (("rho_femtic", "vp", "vs"), None),
]

# Fields plotted/gradiented (and, for mutual information, compared) in
# log10 space rather than linear -- kept in sync with the log10(rho_femtic)
# convention used throughout plot_joint.py / plot_modem_mesh.py. Extend
# this set if other fields in INTERP_FILE are customarily viewed
# logarithmically.
LOG_FIELDS = {"rho_femtic"}

NORMALIZE_EPS = 1e-12  # gradient-magnitude floor below which ratios are NaN

# ---------------------------------------------------------------------------
# Which measures to compute/plot
# ---------------------------------------------------------------------------

ENABLE_CROSS_GRADIENT = True   # Gallardo & Meju (2003, 2004): |t|
PLOT_CROSSGRAD_NORM = False     # also plot sin(theta) = |t| / (|g1||g2|)

ENABLE_COSINE = False           # Molodtsov et al. (2013): cos(theta), signed
ENABLE_COSINE2 = False          # Shi et al. (2018): cos^2(theta), sign-free

ENABLE_GRAMIAN = False         # Zhdanov et al. (2012): det(Gram(g_1..g_N))
PLOT_GRAMIAN_NORM = False       # also plot det(Gram)/prod(|g_i|^2) in [0, 1]

ENABLE_MUTUAL_INFO = True     # see module docstring -- off by default
MI_WINDOW = 5                  # odd # of grid cells/side; UNVERIFIED default, tune to your grid
MI_BINS = 12                   # histogram bins/field/window; UNVERIFIED default, tune to your data

# ---------------------------------------------------------------------------
# Depth-slice maps
# ---------------------------------------------------------------------------

PLOT_MAPS = True
MAP_DEPTHS = [0., 5., 10., 15., 20., 25., 30., 35.]  # None = every depth level in INTERP_FILE; else list of km
MAP_XLIM, MAP_YLIM = None, None  # None = auto from the grid (safe default)

CMAP_CG = "magma"
CMIN_CG, CMAX_CG = None, None  # None = per-panel auto-scaling

CMAP_CG_NORM = "cividis"
CMIN_CG_NORM, CMAX_CG_NORM = 0.0, 1.0  # sin(theta) is naturally in [0, 1]

CMAP_COS = "RdBu_r"
CMIN_COS, CMAX_COS = -1.0, 1.0  # cos(theta) is naturally in [-1, 1]

CMAP_COS2 = "viridis"
CMIN_COS2, CMAX_COS2 = 0.0, 1.0  # cos^2(theta) is naturally in [0, 1]

CMAP_GRAM = "magma"
CMIN_GRAM, CMAX_GRAM = None, None  # None = per-panel auto-scaling (unbounded)

CMAP_GRAM_NORM = "cividis"
CMIN_GRAM_NORM, CMAX_GRAM_NORM = 0.0, 1.0  # bounded by Hadamard's inequality

CMAP_MI = "inferno"
CMIN_MI, CMAX_MI = 0.0, None  # MI >= 0; upper bound is data-dependent

# ---------------------------------------------------------------------------
# Vertical sections
# ---------------------------------------------------------------------------

PLOT_SECTIONS = True

# Same schema as plot_seis.py / plot_joint.py: each entry is
# (name, (lon1, lat1), (lon2, lat2), n_samples). Populate with verified
# profile endpoints for the active SITE_PREFIX -- left empty here rather
# than guessing, per project policy on unverified geographic values.
VSLICES = []

VE = 1.0  # vertical exaggeration for section panels

# ---------------------------------------------------------------------------
# Isoline overlay (optional)
# ---------------------------------------------------------------------------

ISO_LINES_MAP = False
ISO_LINES_VSLICE = False
ISO_LEVELS = None  # None = auto (matplotlib default contour levels)
ISO_COLOR = "white"
ISO_LINEWIDTH = 0.6

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

OUTPUT_DIR = "../plots_structure/"
FIG_DPI = 600
# Output formats: list of matplotlib-supported extensions (e.g. "png",
# "pdf", "svg"). save_paris() renders/saves once per format and returns
# a list of paths -- call sites use saved.extend(...) accordingly.
PLOT_FORMATS = ["jpg", "pdf"]

# Show figures on screen in addition to saving them -- only actually
# shown if matplotlib.is_interactive() is also True (e.g. Spyder's
# inline/Qt backends), so this is safe to leave on when running headless
# on the DIAS HPC cluster or from a terminal: unconditional plt.show()
# blocks/errors outside an interactive backend, same issue already fixed
# in plot_joint.py via SHOW_PLOTS/_maybe_show() -- mirrored here.
SHOW_PLOTS = False

# Derived at runtime by main() from INTERP_FILE's name (see
# _derive_interp_tag below); placeholder here only so the module has a
# defined value if a plotting function is ever called without going
# through main() first.
INTERP_TAG = "-"


# ===========================================================================
# Interpolation-tag derivation (mirrors plot_joint.py's _derive_interp_tag)
# ===========================================================================

# ===========================================================================
# Thin wrappers around tomomt's shared helpers, each supplying this
# script's own globals -- same pattern plot_joint.py already uses for
# plotpy (now tomomt). See tomomt.py's module docstring for what moved
# here from and why; _title_suffix()/_region()-style config closures that
# don't generalise cleanly across scripts stayed local by design.
# ===========================================================================

def _derive_interp_tag(interp_file):
    return tomomt.derive_interp_tag(interp_file)


def _resolve_interp_file():
    return tomomt.resolve_interp_file(SITE_PREFIX, INTERP_FILE)


def _title_suffix():
    return tomomt.title_suffix(SITE_PREFIX, INTERP_TAG)


def _group_label(fields, label):
    return tomomt.group_label(fields, label)


def _maybe_show(fig):
    tomomt.maybe_show(SHOW_PLOTS)


load_joint_grid = tomomt.load_joint_grid
get_field = tomomt.get_field


# ===========================================================================
# Gradients
# ===========================================================================

def gradient_components(field3d, depth, northing, easting, log_transform=False):
    """
    3-D gradient (d/d_easting, d/d_northing, d/d_depth) of `field3d` via
    numpy.gradient using the grid's own real (not assumed uniform)
    coordinate arrays along each axis -- exact on the non-uniform ModEM /
    interpolation meshes used throughout this pipeline, matching the
    standard set by gradient_magnitude() in the other plot scripts.

    field3d : array, shape (len(depth), len(northing), len(easting))
    log_transform : if True, gradient is computed on log10(field3d) --
        matching the log10(rho_femtic) convention used for resistivity elsewhere
        in the pipeline. NaN/non-positive values propagate as NaN.

    Returns (gx, gy, gz), each shaped like field3d, giving the partial
    derivative along easting, northing, and depth respectively. NaN in the
    input only NaNs the 1-2 immediately dependent output cells (numpy.
    gradient's own behaviour), not the whole array -- masked/no-data
    regions stay masked, not smeared.
    """
    if log_transform:
        with np.errstate(divide="ignore", invalid="ignore"):
            field3d = np.where(field3d > 0, np.log10(field3d), np.nan)

    gz, gy, gx = np.gradient(field3d, depth, northing, easting)
    return gx, gy, gz


def cross_gradient_vector(g1, g2):
    """
    Cross product t = grad(m1) x grad(m2) of two 3-D gradient vector
    fields, each given as (gx, gy, gz) triples of arrays. Returns
    (tx, ty, tz, tmag).
    """
    gx1, gy1, gz1 = g1
    gx2, gy2, gz2 = g2

    tx = gy1 * gz2 - gz1 * gy2
    ty = gz1 * gx2 - gx1 * gz2
    tz = gx1 * gy2 - gy1 * gx2
    tmag = np.sqrt(tx ** 2 + ty ** 2 + tz ** 2)
    return tx, ty, tz, tmag


# ===========================================================================
# Pairwise structural-coupling measures: cross-gradient, cosine, cosine^2
# ===========================================================================

def compute_structure(ds, field_a, field_b):
    """
    End-to-end for one field pair: load both fields on the shared grid,
    compute their gradients (respecting LOG_FIELDS), and return the
    cross-gradient magnitude/normalised form (Gallardo & Meju), the
    signed cosine similarity (Molodtsov et al. 2013), and the squared
    cosine similarity (Shi et al. 2018) -- all derived from the same pair
    of gradient vector fields, so they share one mesh and one set of NaN
    (masked) cells.
    """
    val_a, depth, northing, easting = get_field(ds, field_a)
    val_b, depth_b, northing_b, easting_b = get_field(ds, field_b)
    if not (np.array_equal(depth, depth_b)
            and np.array_equal(northing, northing_b)
            and np.array_equal(easting, easting_b)):
        raise ValueError(
            f"Fields '{field_a}' and '{field_b}' are not on the same grid "
            f"in {ds.encoding.get('source', 'INTERP_FILE')}."
        )

    g_a = gradient_components(val_a, depth, northing, easting,
                               log_transform=field_a in LOG_FIELDS)
    g_b = gradient_components(val_b, depth, northing, easting,
                               log_transform=field_b in LOG_FIELDS)

    tx, ty, tz, tmag = cross_gradient_vector(g_a, g_b)

    mag_a = np.sqrt(sum(c ** 2 for c in g_a))
    mag_b = np.sqrt(sum(c ** 2 for c in g_b))
    denom = mag_a * mag_b
    dot = g_a[0] * g_b[0] + g_a[1] * g_b[1] + g_a[2] * g_b[2]

    with np.errstate(divide="ignore", invalid="ignore"):
        tnorm = np.where(denom > NORMALIZE_EPS, tmag / denom, np.nan)  # sin(theta)
        cos = np.where(denom > NORMALIZE_EPS, dot / denom, np.nan)     # cos(theta), Molodtsov (2013)
    tnorm = np.clip(tnorm, 0.0, 1.0)
    cos = np.clip(cos, -1.0, 1.0)
    cos2 = cos ** 2  # cos^2(theta), Shi et al. (2018); by construction cos2 + tnorm**2 == 1

    out = {
        "depth": depth, "northing": northing, "easting": easting,
        "tx": tx, "ty": ty, "tz": tz,
        "crossgrad_mag": tmag, "crossgrad_norm": tnorm,
        "cosine": cos, "cosine2": cos2,
    }

    if ENABLE_MUTUAL_INFO:
        # Raw (log10-where-configured) *values*, not gradients -- MI
        # compares co-located field values in a window, see
        # windowed_mutual_information_2d().
        def _mi_values(val, field):
            if field in LOG_FIELDS:
                with np.errstate(divide="ignore", invalid="ignore"):
                    return np.where(val > 0, np.log10(val), np.nan)
            return val
        out["_val_a"] = _mi_values(val_a, field_a)
        out["_val_b"] = _mi_values(val_b, field_b)

    return out


# ===========================================================================
# Gramian structural-coupling determinant (Zhdanov et al. 2012)
# ===========================================================================

def compute_gramian(ds, fields):
    """
    Gram-matrix determinant of the gradient vectors of 2 or 3 fields:
        Gram_ij = grad(m_i) . grad(m_j),   det(Gram)
    At N=2, det(Gram) == |grad(m1) x grad(m2)|^2 exactly (Lagrange's
    identity) -- algebraically the same information as the squared
    cross-gradient magnitude, included here mainly for cross-checking.
    At N=3, det(Gram) is the squared scalar triple product
    [grad(m1), grad(m2), grad(m3)]^2, a genuinely different, higher-order
    measure: it vanishes wherever the three gradient vectors are
    coplanar (not just wherever any two are parallel), and is otherwise
    only zero for N>3 configurations this function refuses to touch (see
    below).

    Also returns a normalised form det(Gram)/prod_i(|grad(m_i)|^2),
    which by Hadamard's inequality is bounded in [0, 1] for any N (at
    N=2 it reduces exactly to the existing sin^2(theta) measure).
    """
    n = len(fields)
    if n not in (2, 3):
        raise ValueError(
            f"GRAMIAN_GROUPS entry {fields!r} has {n} fields; the Gramian "
            f"stabiliser (Zhdanov et al. 2012) is implemented here only for "
            f"N=2 (equivalent to the squared cross-gradient magnitude) or "
            f"N=3 (the genuinely novel squared scalar triple product). "
            f"N>3 needs a general NxN cofactor expansion that was not "
            f"built here -- flagged rather than guessed at."
        )

    grads = []
    ref_depth = ref_north = ref_east = None
    for f in fields:
        val, depth, northing, easting = get_field(ds, f)
        if ref_depth is None:
            ref_depth, ref_north, ref_east = depth, northing, easting
        elif not (np.array_equal(depth, ref_depth)
                  and np.array_equal(northing, ref_north)
                  and np.array_equal(easting, ref_east)):
            raise ValueError(
                f"Fields {fields!r} are not all on the same grid in "
                f"{ds.encoding.get('source', 'INTERP_FILE')}."
            )
        grads.append(gradient_components(val, depth, northing, easting,
                                          log_transform=f in LOG_FIELDS))

    dots = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            gxi, gyi, gzi = grads[i]
            gxj, gyj, gzj = grads[j]
            d = gxi * gxj + gyi * gyj + gzi * gzj
            dots[i][j] = d
            dots[j][i] = d

    if n == 2:
        det = dots[0][0] * dots[1][1] - dots[0][1] * dots[1][0]
    else:  # n == 3
        a, b, c = dots[0][0], dots[0][1], dots[0][2]
        d_, e, f_ = dots[1][0], dots[1][1], dots[1][2]
        g_, h, i_ = dots[2][0], dots[2][1], dots[2][2]
        det = a * (e * i_ - f_ * h) - b * (d_ * i_ - f_ * g_) + c * (d_ * h - e * g_)

    # Gram determinants are mathematically >= 0 (Gram matrices are PSD);
    # clip away sub-eps numerical noise from finite-difference roundoff
    # rather than let tiny negatives propagate as NaN through downstream
    # sqrt-like uses.
    det = np.clip(det, 0.0, None)

    diag_prod = dots[0][0]
    for i in range(1, n):
        diag_prod = diag_prod * dots[i][i]
    with np.errstate(divide="ignore", invalid="ignore"):
        gnorm = np.where(diag_prod > NORMALIZE_EPS, det / diag_prod, np.nan)
    gnorm = np.clip(gnorm, 0.0, 1.0)

    return {
        "depth": ref_depth, "northing": ref_north, "easting": ref_east,
        "gramian_det": det, "gramian_norm": gnorm,
    }


# ===========================================================================
# Windowed mutual information (optional; see module docstring)
# ===========================================================================

def windowed_mutual_information_2d(a2d, b2d, window=None, bins=None):
    """
    Local (sliding-window) mutual information between two co-located 2-D
    slices, in nats, via histogram-based joint/marginal density estimation
    within each window:
        MI = sum_xy p(x,y) * log( p(x,y) / (p(x) p(y)) )

    Unlike the gradient-based measures above, this compares field
    *values* directly rather than their spatial derivatives -- it can
    register a structural relationship even where both fields vary
    smoothly rather than sharply. Mandolesi & Jones (2014) used a
    *global*, whole-model mutual information as a joint-inversion
    objective; the local/windowed version here is a diagnostic-mapping
    choice made for this pipeline, not something taken from that paper.

    window, bins : if None, fall back to the module-level MI_WINDOW /
        MI_BINS. Both are free parameters with no literature- or
        pipeline-verified default for this project's grids -- flagged
        here rather than silently tuned; check they suit your grid
        spacing/target feature scale and sample count before trusting
        the output.

    Performance: O(rows * cols * window^2) numpy.histogram2d calls --
    can be slow on large grids. That is why ENABLE_MUTUAL_INFO defaults
    to False.

    Returns a (rows, cols)-shaped array, NaN within half a window of any
    edge or wherever too few finite samples fall in the window.
    """
    window = MI_WINDOW if window is None else window
    bins = MI_BINS if bins is None else bins

    ny, nx = a2d.shape
    half = window // 2
    mi = np.full((ny, nx), np.nan)
    min_valid = max(2 * bins, window)

    for iy in range(half, ny - half):
        for ix in range(half, nx - half):
            wa = a2d[iy - half:iy + half + 1, ix - half:ix + half + 1].ravel()
            wb = b2d[iy - half:iy + half + 1, ix - half:ix + half + 1].ravel()
            mask = np.isfinite(wa) & np.isfinite(wb)
            if mask.sum() < min_valid:
                continue
            wa, wb = wa[mask], wb[mask]
            hist, _, _ = np.histogram2d(wa, wb, bins=bins)
            total = hist.sum()
            if total <= 0:
                continue
            pxy = hist / total
            px = pxy.sum(axis=1)
            py = pxy.sum(axis=0)
            outer = np.outer(px, py)
            nz = pxy > 0
            mi[iy, ix] = np.sum(pxy[nz] * np.log(pxy[nz] / outer[nz]))

    return mi


# ===========================================================================
# Quantity registry (drives the generic plotting functions below)
# ===========================================================================

def _quantity_spec(key):
    """
    Central lookup from a quantity key to its plot styling and metadata.
    Reads current CMAP_*/CMIN_*/CMAX_* module globals at call time (not
    captured earlier), so edits to the config block above take effect
    without touching this function.
    """
    specs = {
        "crossgrad_mag": dict(
            title="cross-gradient |t|", cbar="|grad m1 x grad m2|",
            cmap=CMAP_CG, vmin=CMIN_CG, vmax=CMAX_CG, tag="crossgrad_mag",
        ),
        "crossgrad_norm": dict(
            title="normalised cross-gradient (sin theta)", cbar="sin(theta)  [0-1]",
            cmap=CMAP_CG_NORM, vmin=CMIN_CG_NORM, vmax=CMAX_CG_NORM, tag="crossgrad_norm",
        ),
        "cosine": dict(
            title="cosine similarity (Molodtsov et al. 2013)", cbar="cos(theta)  [-1,1]",
            cmap=CMAP_COS, vmin=CMIN_COS, vmax=CMAX_COS, tag="cosine",
        ),
        "cosine2": dict(
            title="squared cosine similarity (Shi et al. 2018)", cbar="cos^2(theta)  [0,1]",
            cmap=CMAP_COS2, vmin=CMIN_COS2, vmax=CMAX_COS2, tag="cosine2",
        ),
        "gramian_det": dict(
            title="Gramian determinant (Zhdanov et al. 2012)", cbar="det(Gram)",
            cmap=CMAP_GRAM, vmin=CMIN_GRAM, vmax=CMAX_GRAM, tag="gramian",
        ),
        "gramian_norm": dict(
            title="normalised Gramian (Zhdanov et al. 2012)", cbar="det(Gram) / prod|grad|^2  [0,1]",
            cmap=CMAP_GRAM_NORM, vmin=CMIN_GRAM_NORM, vmax=CMAX_GRAM_NORM, tag="gramian_norm",
        ),
        "mutual_info": dict(
            title="windowed mutual information", cbar="MI (nats)",
            cmap=CMAP_MI, vmin=CMIN_MI, vmax=CMAX_MI, tag="mi",
        ),
    }
    return specs[key]


# ===========================================================================
# Plotting -- depth-slice maps
# ===========================================================================

def plot_map(cg, depth_km, group_label, quantity_key):
    """
    Depth-slice map of one quantity for one field group.

    Styling note (integration seam 1): this does its own minimal
    pcolormesh plotting rather than calling into tomomt.py's
    build_panel_figure / finish_panel_colorbar / draw_north_arrow /
    add_latlon_ticks / clipped_markers / clipped_labels, since their
    exact current signatures weren't available to verify against here.
    Swapping these in to match plot_joint.py's house style exactly is a
    straightforward follow-up.

    Uses pcolormesh (shading="nearest" on the grid's own easting/northing
    cell-centre coordinates), not imshow: imshow assumes uniform pixel
    spacing and silently misaligns cells on the non-uniform interpolation
    meshes used throughout this pipeline -- same reasoning as
    plot_section() below, which already used pcolormesh.
    """
    depth, northing, easting = cg["depth"], cg["northing"], cg["easting"]
    iz = int(np.argmin(np.abs(depth - depth_km)))

    if quantity_key == "mutual_info":
        z2d = windowed_mutual_information_2d(cg["_val_a"][iz], cg["_val_b"][iz])
    else:
        z2d = cg[quantity_key][iz]

    spec = _quantity_spec(quantity_key)
    norm = (Normalize(vmin=spec["vmin"], vmax=spec["vmax"])
            if (spec["vmin"] is not None or spec["vmax"] is not None) else None)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(
        easting, northing, z2d, cmap=spec["cmap"], norm=norm,
        shading="nearest", zorder=5,
    )
    ax.set_aspect("equal")
    if ISO_LINES_MAP:
        ax.contour(easting, northing, z2d, levels=ISO_LEVELS,
                   colors=ISO_COLOR, linewidths=ISO_LINEWIDTH, zorder=10)

    ax.set_xlim(MAP_XLIM)
    ax.set_ylim(MAP_YLIM)
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")

    suffix = ", ".join(_title_suffix())
    ax.set_title(f"{group_label}: {spec['title']}\nz = {depth[iz]:.2f} km ({suffix})")

    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label(spec["cbar"])

    fig.tight_layout()

    stem = f"{SITE_PREFIX}_{group_label}_{spec['tag']}_{INTERP_TAG}_map_z{depth[iz]:.2f}km"
    return fig, stem


# ===========================================================================
# Plotting -- vertical sections
# ===========================================================================

def _profile_line(lon1lat1, lon2lat2, n):
    """
    Placeholder straight-line sampler in lon/lat, linearly interpolated to
    n points; projection to UTM easting/northing km is expected to be
    supplied by the pipeline's shared pyproj transformer (same one used
    in plot_seis.py / plot_joint.py) -- wire in that transformer here to
    match VSLICES behaviour exactly. Left as a clearly-marked seam rather
    than re-deriving/guessing the project's UTM transform in isolation.
    """
    raise NotImplementedError(
        "Wire in the pipeline's shared lon/lat -> UTM-km transformer "
        "(as used by plot_seis.py / plot_joint.py) before enabling "
        "PLOT_SECTIONS with a populated VSLICES list."
    )


def plot_section(cg, vslice, group_label, quantity_key):
    """
    Vertical section of one quantity for one field group. Not available
    for quantity_key == "mutual_info" (map-only, see module docstring);
    callers filter that out before calling this.
    """
    if quantity_key == "mutual_info":
        raise NotImplementedError(
            "Windowed mutual information is only implemented for "
            "depth-slice maps, not vertical sections -- see module "
            "docstring (integration seam 3)."
        )

    name, p1, p2, n = vslice
    easting_km, northing_km = _profile_line(p1, p2, n)  # noqa: F841 -- seam, see above

    spec = _quantity_spec(quantity_key)
    interp = RegularGridInterpolator(
        (cg["depth"], cg["northing"], cg["easting"]),
        cg[quantity_key],
        bounds_error=False, fill_value=np.nan,
    )
    depth_grid = cg["depth"]
    dist_km = np.linspace(0, 1, n)  # placeholder along-profile distance; see _profile_line
    pts = np.array([
        [z, ny, ex]
        for z in depth_grid
        for ny, ex in zip(northing_km, easting_km)
    ])
    section = interp(pts).reshape(len(depth_grid), n)

    norm = (Normalize(vmin=spec["vmin"], vmax=spec["vmax"])
            if (spec["vmin"] is not None or spec["vmax"] is not None) else None)

    fig, ax = plt.subplots(figsize=(8, 8 / max(VE, 1e-6) if VE >= 1 else 4))
    im = ax.pcolormesh(dist_km, depth_grid, section, cmap=spec["cmap"], norm=norm,
                       shading="nearest", zorder=5)
    ax.invert_yaxis()
    ax.set_aspect(VE)
    ax.set_xlabel("Distance along profile (km)")
    ax.set_ylabel("Depth (km)")

    if ISO_LINES_VSLICE:
        ax.contour(dist_km, depth_grid, section, levels=ISO_LEVELS,
                   colors=ISO_COLOR, linewidths=ISO_LINEWIDTH, zorder=10)

    ax.text(0.5, 1.02, f"VE = {VE:g}", transform=ax.transAxes,
           ha="center", va="bottom", zorder=21)

    suffix = ", ".join(_title_suffix())
    ax.set_title(f"{group_label}: {spec['title']}\nsection: {name} ({suffix})")

    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label(spec["cbar"])

    fig.tight_layout()

    stem = f"{SITE_PREFIX}_{group_label}_{spec['tag']}_{INTERP_TAG}_sec_{name}"
    return fig, stem


# ===========================================================================
# Delivery helpers
# ===========================================================================

def save_paris(fig, stem, outdir):
    """
    Save `fig` once per format in PLOT_FORMATS, each with its mtime set
    to now in Europe/Paris local time. Shows the figure first if
    SHOW_PLOTS (see _maybe_show). Returns a list of the saved paths --
    call sites use ``saved.extend(save_paris(...))``, not ``.append(...)``.
    """
    _maybe_show(fig)
    return tomomt.save_paris(fig, stem, outdir, PLOT_FORMATS, FIG_DPI)


def zip_outputs(paths, project_name="structure"):
    return tomomt.zip_outputs(paths, project_name, OUTPUT_DIR)


# ===========================================================================
# Main
# ===========================================================================

def main():
    global INTERP_TAG

    interp_file = _resolve_interp_file()
    INTERP_TAG = _derive_interp_tag(interp_file)

    ds = load_joint_grid(interp_file)
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    saved = []

    # -- Pairwise measures: cross-gradient, cosine, cosine^2, (optional) MI --
    pairwise_quantities = []
    if ENABLE_CROSS_GRADIENT:
        pairwise_quantities.append("crossgrad_mag")
        if PLOT_CROSSGRAD_NORM:
            pairwise_quantities.append("crossgrad_norm")
    if ENABLE_COSINE:
        pairwise_quantities.append("cosine")
    if ENABLE_COSINE2:
        pairwise_quantities.append("cosine2")

    for field_a, field_b, label in STRUCTURE_PAIRS:
        group = _group_label((field_a, field_b), label)
        print(f"[structure] {group}: computing pairwise measures...")
        cg = compute_structure(ds, field_a, field_b)

        quantities = list(pairwise_quantities)
        if ENABLE_MUTUAL_INFO:
            quantities.append("mutual_info")

        if PLOT_MAPS:
            depths = MAP_DEPTHS if MAP_DEPTHS is not None else cg["depth"].tolist()
            for z in depths:
                for q in quantities:
                    fig, stem = plot_map(cg, z, group, q)
                    saved.extend(save_paris(fig, stem, outdir))

        if PLOT_SECTIONS:
            section_quantities = [q for q in quantities if q != "mutual_info"]
            if not VSLICES:
                print(f"[structure] {group}: PLOT_SECTIONS=True but VSLICES is empty; skipping.")
            for vslice in VSLICES:
                for q in section_quantities:
                    fig, stem = plot_section(cg, vslice, group, q)
                    saved.extend(save_paris(fig, stem, outdir))

    # -- Gramian groups (N=2 or 3) --
    gramian_quantities = []
    if ENABLE_GRAMIAN:
        gramian_quantities.append("gramian_det")
        if PLOT_GRAMIAN_NORM:
            gramian_quantities.append("gramian_norm")

    for fields, label in GRAMIAN_GROUPS:
        group = _group_label(fields, label)
        print(f"[structure] {group}: computing {len(fields)}-field Gramian...")
        cg = compute_gramian(ds, fields)

        if PLOT_MAPS:
            depths = MAP_DEPTHS if MAP_DEPTHS is not None else cg["depth"].tolist()
            for z in depths:
                for q in gramian_quantities:
                    fig, stem = plot_map(cg, z, group, q)
                    saved.extend(save_paris(fig, stem, outdir))

        if PLOT_SECTIONS:
            if not VSLICES:
                print(f"[structure] {group}: PLOT_SECTIONS=True but VSLICES is empty; skipping.")
            for vslice in VSLICES:
                for q in gramian_quantities:
                    fig, stem = plot_section(cg, vslice, group, q)
                    saved.extend(save_paris(fig, stem, outdir))

    ds.close()

    if saved:
        zpath = zip_outputs(saved, project_name=f"{SITE_PREFIX}_structure")
        print(f"[structure] wrote {len(saved)} figure(s) -> {zpath}")
    else:
        print("[structure] nothing to save (PLOT_MAPS and PLOT_SECTIONS both False, "
              "or all sections skipped/no quantities enabled).")


if __name__ == "__main__":
    main()
