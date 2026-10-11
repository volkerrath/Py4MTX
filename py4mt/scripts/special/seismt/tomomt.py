"""
tomomt.py
=========
Shared helper module for the DIAS MT/seismic joint-imaging pipeline
(Sabancaya / Tacna). Renamed from ``plotpy.py`` (same module, broadened
scope) when a second round of de-duplication pulled in run-management
helpers (safe file/NetCDF writes, output-path joining, PLOT_FORMATS/DPI
figure saving, the Paris-timestamped save+zip delivery convention,
INTERP_FILE resolution and its interpolation-tag parsing, and a couple of
tiny pipeline-wide plumbing one-liners) that had each been copy-pasted
into three or four scripts independently. Imported by ``cluster.py``,
``crossplots.py``, ``interpolate.py``, ``plot_joint.py``, ``precompute.py``,
and ``structure.py``.

A third round (2026-08) consolidated the read-side subset of
``femtic.py`` and ``modem.py`` actually used by this pipeline
(``interpolate.py``/``plot_femtic_mesh.py``'s FEMTIC mesh+resistivity-
block readers; ``precompute.py``'s ModEM model/data/topo readers), so
that ``tomomt.py`` is now the *only* local module any pipeline script
needs to import. This also removes femtic.py's own hard, unconditional
``from ensembles import (...)`` module-level dependency, which every
importer previously inherited even when only reading a mesh file. See
the "SECTION: FEMTIC I/O" / "SECTION: ModEM I/O" blocks below for what
was carried over and, explicitly, what was left behind.

Two families of helpers live here, both are genuinely duplicated,
run-independent pieces pulled out of the scripts above -- not things
guessed at or added speculatively:

1. Plotting primitives (the original ``plotpy.py`` scope): UTM<->lon/lat
   coordinate conversion, colormap loading (matplotlib name / GMT .cpt
   file / plain RGB(A) list), hillshading, ModEM sensitivity-alpha
   helpers, region-clipped scatter/label helpers, the north arrow, the
   deterministic (equal-scale-by-construction) panel/colorbar layout
   engine, the lon/lat tick overlay, the free-text annotation, VE-label
   positioning, and generic profile point sampling/projection.
2. Run-management / I-O plumbing (new in this rename): safe
   file/NetCDF writes that don't choke on a stale read-only leftover,
   output-path joining, figure saving (both the plain PLOT_FORMATS loop
   and the Paris-timestamped save+zip delivery convention used by
   structure.py/crossplots.py), INTERP_FILE auto-resolution, and
   interpolation-tag derivation from an INTERP_FILE name.

None of these read a calling script's global variables — every value
that used to come from a script's own USER SETTINGS is now an explicit
argument (colours, sizes, positions, region bounds, toggles, output
directories, format lists, ...). Each script still owns its own settings
and its own script-specific plotting/analysis code (draw_basemap,
draw_features, compute_vertical_slice_*, plot_vertical_slice, the
region()/colorbar_settings() config-dict builders, ...); it just imports
these shared pieces instead of redefining them.

What deliberately stayed OUT of this module (per-script, not shared),
and why:
  - ``draw_basemap`` (cluster.py / plot_joint.py) — ~90% identical
    bodies, but each closes over that script's own loaded topo/bathymetry
    arrays and style globals; genericising it would mean a large explicit
    parameter list for a function that's realistically likely to diverge
    further (different basemap needs per script) rather than converge.
    Flagged, not moved.
  - ``_region()`` / ``_colorbar_settings()`` (cluster.py / plot_joint.py)
    — thin config-dict builders closing over each script's own USER
    SETTINGS names (which differ slightly, e.g. ``nticks=n_labels`` vs.
    ``nticks=COLORBAR_NTICKS``); exactly the kind of per-script glue
    plotpy.py's own original design section was already meant to leave
    alone.
  - ``_title_suffix()`` — kept per-script: structure.py/crossplots.py
    return a list (``[site, tag]``, joined by the caller), plot_joint.py
    returns a pre-formatted string (``" [site, tag]"``); different
    call-site contracts for a one-line function, not worth forcing a
    shared signature over.
  - The gradient machinery in interpolate.py (spline- or
    finite-difference-based, with seismic-grid Jacobian correction) vs.
    structure.py's gradient_components() (plain numpy.gradient,
    joint-grid only) — similar in spirit, materially different in
    capability; not the same function wearing two names.

Authors: Svetlana Byrdina (SMB) & Volker Rath (DIAS)
AI-assisted development: Claude (Anthropic)
License: GNU General Public License v3 (GPL-3.0-or-later)
AI-generated code — review before use in production.
"""

import glob
import math
import os
import re
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LightSource
from pyproj import Transformer
from scipy.ndimage import gaussian_filter

# Same Europe/Paris delivery-timestamp convention used throughout the
# pipeline (zip filenames and internal file mtimes) -- centralised here
# rather than re-instantiated in every script that saves output.
PARIS_TZ = ZoneInfo("Europe/Paris")


# =====================================================================
# Coordinate transforms (UTM Zone 19S / EPSG:32719 — fixed project-wide)
# =====================================================================
_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32719", always_xy=True)
_to_geo = Transformer.from_crs("EPSG:32719", "EPSG:4326", always_xy=True)


def to_utm_km(lon, lat):
    """Convert geographic lon/lat to UTM Zone 19S easting/northing in km."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    e, n = _to_utm.transform(lon, lat)
    return e / 1e3, n / 1e3


def to_geo(e_km, n_km):
    """Convert UTM Zone 19S easting/northing (km) to geographic lon/lat."""
    lon, lat = _to_geo.transform(np.asarray(e_km) * 1e3, np.asarray(n_km) * 1e3)
    return lon, lat


# =====================================================================
# Colormap loading (matplotlib name / GMT .cpt file / plain RGB(A) list)
# =====================================================================
def load_colormap(spec, name=None):
    """
    Resolve a colourmap spec into a matplotlib Colormap.

    Accepts, in order of precedence:
      - an existing Colormap instance — returned unchanged
      - a path to a GMT ``.cpt`` file — parsed directly, preserving the
        file's own (possibly non-uniform) colour-stop spacing. This lets
        you use the *actual* original palette (e.g. viridisr_vp.cpt) for
        an exact visual comparison against GMT-produced figures, instead
        of a same-ish matplotlib named stand-in.
      - a path to a plain text/CSV file of RGB(A) rows (0-255 or 0-1,
        whitespace- or comma-separated, one colour per line) — built into
        an evenly-spaced ListedColormap. Useful for reusing an exact
        palette exported from another tool (e.g. ParaView, Generic
        Mapping Tools' makecpt, a colleague's colour list) so two
        different figures use pixel-identical colours for comparison.
      - any matplotlib-registered colormap name (built-in, or registered
        by a third-party package such as cmcrameri/cmocean if that
        package has been imported elsewhere in the process) — resolved
        via plt.get_cmap, unchanged from the original behaviour.

    Parameters
    ----------
    spec : str or matplotlib.colors.Colormap
    name : str, optional — name to register the resulting colormap under
           (defaults to the file's base name, or the spec string itself)

    Returns
    -------
    matplotlib.colors.Colormap
    """
    if isinstance(spec, mpl.colors.Colormap):
        return spec

    spec = str(spec)
    ext = os.path.splitext(spec)[1].lower()
    cmap_name = name or os.path.splitext(os.path.basename(spec))[0]

    if ext == ".cpt":
        return _load_cpt_colormap(spec, cmap_name)
    if ext in (".txt", ".csv", ".dat") and os.path.exists(spec):
        return _load_rgb_list_colormap(spec, cmap_name)

    # Not a recognised file — treat as a matplotlib-registered name
    # (built-in, or from a third-party package already imported).
    return plt.get_cmap(spec)


def _parse_cpt_color(tokens):
    """Parse a single .cpt colour field: 'R G B', 'R/G/B', '#hex', or grey."""
    if len(tokens) >= 3:
        r, g, b = (float(t) for t in tokens[:3])
        return (r / 255, g / 255, b / 255)
    tok = tokens[0]
    if tok.startswith("#"):
        return mpl.colors.to_rgb(tok)
    if "/" in tok:
        r, g, b = (float(t) for t in tok.split("/"))
        return (r / 255, g / 255, b / 255)
    v = float(tok)
    return (v / 255, v / 255, v / 255)


def _load_cpt_colormap(path, name):
    """Parse a GMT .cpt colour-palette file into a LinearSegmentedColormap,
    preserving its own colour-stop spacing (not assumed to be uniform)."""
    stops = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line[0] in "BFNbfn":
                continue
            parts = line.split()
            try:
                if len(parts) >= 8:
                    z0 = float(parts[0]); c0 = _parse_cpt_color(parts[1:4])
                    z1 = float(parts[4]); c1 = _parse_cpt_color(parts[5:8])
                elif len(parts) == 4:
                    z0 = float(parts[0]); c0 = _parse_cpt_color([parts[1]])
                    z1 = float(parts[2]); c1 = _parse_cpt_color([parts[3]])
                else:
                    continue
            except ValueError:
                continue
            stops.append((z0, c0))
            stops.append((z1, c1))

    if not stops:
        raise ValueError(f"No colour stops parsed from .cpt file: {path}")

    zs = np.array([s[0] for s in stops], dtype=float)
    zmin, zmax = zs.min(), zs.max()
    span = zmax - zmin if zmax > zmin else 1.0
    seen = {}
    for z, c in stops:
        seen[round((z - zmin) / span, 6)] = c
    positions_colors = sorted(seen.items())
    return mpl.colors.LinearSegmentedColormap.from_list(name, positions_colors)


def _load_rgb_list_colormap(path, name):
    """Build a ListedColormap from a plain text/CSV file of RGB(A) rows.
    Values may be 0-255 or 0-1; whitespace- or comma-separated."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 3:
                rows.append([float(p) for p in parts[:4]])

    if not rows:
        raise ValueError(f"No colour rows parsed from: {path}")

    arr = np.array(rows, dtype=float)
    if arr.max() > 1.0:
        arr[:, :3] /= 255.0
        if arr.shape[1] == 4:
            arr[:, 3] /= 255.0
    return mpl.colors.ListedColormap(arr, name=name)


def export_colormap_to_cpt(cmap, vmin, vmax, outpath, n_steps=32):
    """
    Export a matplotlib Colormap to a GMT-style .cpt file over [vmin, vmax].

    Reverse of load_colormap()'s .cpt import — samples n_steps+1 points
    across the colourmap and writes them as n_steps colour segments, so a
    colourmap actually used here (a matplotlib built-in name, or something
    already imported from a file/package via load_colormap) can be
    re-exported for use in GMT, or shared with a colleague for an exact
    comparison against a figure made with a named/registered colourmap
    rather than a hand-picked .cpt.

    Parameters
    ----------
    cmap : str or matplotlib.colors.Colormap — resolved via load_colormap
           if not already a Colormap instance
    vmin, vmax : float — data range the colourmap is stretched over; the
           .cpt's own z breakpoints are written in this range so it's
           directly usable for the same data in GMT
    outpath : str — output .cpt file path
    n_steps : int — number of colour segments (n_steps+1 sample points)
    """
    cmap = load_colormap(cmap) if not isinstance(cmap, mpl.colors.Colormap) else cmap
    zs = np.linspace(vmin, vmax, n_steps + 1)
    fracs = np.linspace(0.0, 1.0, n_steps + 1)
    rgb = (np.array([cmap(f)[:3] for f in fracs]) * 255).round().astype(int)

    lines = ["# COLOR_MODEL = RGB",
             f"# Exported from matplotlib colormap {cmap.name!r} "
             f"over [{vmin}, {vmax}]"]
    for i in range(n_steps):
        z0, z1 = zs[i], zs[i + 1]
        r0, g0, b0 = rgb[i]
        r1, g1, b1 = rgb[i + 1]
        lines.append(f"{z0:<12.6g} {r0:3d} {g0:3d} {b0:3d}   "
                     f"{z1:<12.6g} {r1:3d} {g1:3d} {b1:3d}")

    r0, g0, b0 = rgb[0]
    r1, g1, b1 = rgb[-1]
    lines.append(f"B {r0} {g0} {b0}")
    lines.append(f"F {r1} {g1} {b1}")
    lines.append("N 128 128 128")

    with open(outpath, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Exported colourmap to: {outpath}")


# =====================================================================
# Hillshade
# =====================================================================
def compute_hillshade(z2d, dx_km, dy_km, azimuth=315, altitude=45, sigma=1.0):
    """Return a [0, 1] hillshade array for a 2-D elevation grid (metres).

    LightSource.hillshade() computes the surface gradient with
    numpy.gradient, which uses one-sided (lower-quality) differences at
    the true boundary rows/columns of whatever array it's given, instead
    of the centred differences used everywhere else — producing a
    visibly different thin stripe right along the top, bottom, and side
    edges of the rendered hillshade, unrelated to the real terrain.
    Padding the elevation by a few pixels before filtering/shading (using
    an odd/slope-preserving reflection, which continues the local
    gradient rather than mirroring values — plain mirroring introduces
    its own artificial kink right at the seam) and cropping the result
    back down to the original shape gives every true-edge pixel a proper
    two-sided gradient too, removing the stripe.
    """
    ls = LightSource(azdeg=azimuth, altdeg=altitude)
    pad = max(int(np.ceil(3 * sigma)), 3) if sigma > 0 else 3
    z_padded = np.pad(z2d, pad, mode="reflect", reflect_type="odd")
    if sigma > 0:
        z_padded = gaussian_filter(z_padded, sigma=sigma)
    hs_padded = ls.hillshade(z_padded, dx=dx_km * 1e3, dy=dy_km * 1e3, vert_exag=1.0)
    return hs_padded[pad:-pad, pad:-pad]


# =====================================================================
# ModEM sensitivity-based alpha helpers
# =====================================================================
def sens_shade_alpha(sens, low, high, max_alpha):
    """
    Map a sensitivity array to a shading alpha in [0, max_alpha]:
    max_alpha at/below `low`, 0 at/above `high`, linearly interpolated
    in between. NaN (missing sensitivity data) is treated as max_alpha —
    conservative, since missing information is not evidence of good
    resolution.
    """
    sens = np.asarray(sens, dtype=float)
    if high == low:
        alpha = np.where(sens <= low, max_alpha, 0.0)
    else:
        frac = np.clip((high - sens) / (high - low), 0.0, 1.0)
        alpha = frac * max_alpha
    return np.where(np.isnan(sens), max_alpha, alpha)


def sens_data_alpha(sens, low, high, base_alpha):
    """
    Map a sensitivity array to a per-cell alpha for the *data layer
    itself* (as opposed to sens_shade_alpha's overlay-on-top alpha): 0
    (fully transparent — whatever is drawn underneath, e.g. the
    topography basemap, shows straight through) at/below `low`,
    base_alpha (the normal data opacity) at/above `high`, linearly
    interpolated in between. NaN (missing sensitivity data) is treated
    as 0 — conservative, same reasoning as blanking.
    """
    sens = np.asarray(sens, dtype=float)
    if high == low:
        alpha = np.where(sens >= high, base_alpha, 0.0)
    else:
        frac = np.clip((sens - low) / (high - low), 0.0, 1.0)
        alpha = frac * base_alpha
    return np.where(np.isnan(sens), 0.0, alpha)


def load_sens_depth_slice(path, use_sensitivity, ref_shape, ref_northing, ref_easting):
    """
    Load a precompute.py horizontal sensitivity/resolution slice NetCDF
    (a single DataArray with northing/easting coords, in either
    orientation) and re-orient it to match a reference resistivity
    slice's own (northing, easting) orientation. Returns None if
    `use_sensitivity` is False, the file doesn't exist, or the loaded
    grid doesn't match (ref_shape, ref_northing, ref_easting) — in which
    case shading/blanking for that depth is simply skipped, with a
    printed WARNING explaining why.

    Was duplicated identically in plot_modem_image.py and
    plot_modem_mesh.py's own per-depth-slice loops.
    """
    if not use_sensitivity:
        return None
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found — sensitivity masking/shading "
              f"is disabled for this depth slice. Check that "
              f"precompute.py found the .sns file (look for its "
              f"own WARNING) and that OUTPUT_DIR there matches NC_DIR here.")
        return None
    _da = xr.open_dataarray(path)
    sy = _da["northing"].values
    sx = _da["easting"].values
    sv = _da.values.copy().astype(float)
    _da.close()

    if sv.shape[0] != len(sy):
        sv = sv.T
    if sy[0] > sy[-1]:
        sy = sy[::-1]
        sv = sv[::-1, :]
    if sx[0] > sx[-1]:
        sx = sx[::-1]
        sv = sv[:, ::-1]

    if sv.shape != ref_shape or not (np.allclose(sy, ref_northing) and
                                      np.allclose(sx, ref_easting)):
        print(f"  WARNING: {path} grid doesn't match the resistivity slice "
              f"— skipping shading/blanking for this depth.")
        return None
    return sv


# =====================================================================
# Robust surface detection (used by compute_vertical_slice_* for the
# topo line/fill on cross-sections)
# =====================================================================
def first_valid_run(valid, min_run=1):
    """
    For each column of a (n_depth, n_seg) boolean array (True = real,
    non-air data), find the shallowest depth index where at least
    min_run CONSECUTIVE cells are valid, rather than just the first
    valid cell on its own.

    A plain "first valid cell" surface pick is vulnerable to a single
    spurious cell — one padding/air cell that wasn't actually assigned a
    true air resistivity, or one column's data landing in the wrong mesh
    cell — being mistaken for the real topographic surface, producing a
    tall, sharp, flat-topped artifact standing above the genuine terrain
    on either side of it. Requiring a short run of consecutive valid
    cells (min_run=1 disables this and reverts to "first valid cell")
    filters that out: real rock, once reached, stays valid for many
    cells going deeper, while a spurious single cell does not.

    Returns (has_data, first_valid_idx), both length n_seg.
    """
    if min_run <= 1:
        has_data = valid.any(axis=0)
        first_valid_idx = np.argmax(valid, axis=0)
        return has_data, first_valid_idx

    n_depth = valid.shape[0]
    run = np.zeros_like(valid, dtype=int)
    run[0] = valid[0].astype(int)
    for d in range(1, n_depth):
        run[d] = np.where(valid[d], run[d - 1] + 1, 0)
    meets = run >= min_run
    has_data = meets.any(axis=0)
    idx_of_run_end = np.argmax(meets, axis=0)
    first_valid_idx = np.clip(idx_of_run_end - (min_run - 1), 0, n_depth - 1)
    return has_data, first_valid_idx


# =====================================================================
# VE-label position resolver (used by plot_vertical_slice)
# =====================================================================
_VE_POS_PRESETS = {
    "upper right": (0.99, 0.99, "right", "top"),
    "upper left":  (0.01, 0.99, "left",  "top"),
    "lower right": (0.99, 0.01, "right", "bottom"),
    "lower left":  (0.01, 0.01, "left",  "bottom"),
}


def resolve_ve_pos(spec):
    """Resolve a VSLICE_VE_POS setting into an (x, y, ha, va) tuple in
    axes-fraction coordinates. `spec` may be one of the named presets
    ("lower right", "lower left", "upper right", "upper left") or an
    explicit (x, y, ha, va) tuple, passed through unchanged."""
    if isinstance(spec, str):
        try:
            return _VE_POS_PRESETS[spec.lower()]
        except KeyError:
            raise ValueError(
                f"VSLICE_VE_POS={spec!r} not recognised; choose one of "
                f"{list(_VE_POS_PRESETS)} or an explicit (x, y, ha, va) tuple."
            )
    return spec



# =====================================================================
# Clip-aware scatter / text helpers
# =====================================================================
def in_region(xe, yn, region):
    """region = (xmin, xmax, ymin, ymax), in the same units as xe/yn."""
    xmin, xmax, ymin, ymax = region
    xe = np.asarray(xe)
    yn = np.asarray(yn)
    return (xe >= xmin) & (xe <= xmax) & (yn >= ymin) & (yn <= ymax)


def clipped_scatter(ax, xe, yn, region, **kwargs):
    """ax.scatter(), restricted to points inside `region`."""
    xe = np.asarray(xe)
    yn = np.asarray(yn)
    mask = in_region(xe, yn, region)
    if not np.any(mask):
        return
    ax.scatter(xe[mask], yn[mask], **kwargs)


# Scatter-style kwarg name -> ax.plot() marker kwarg name. 's' (scatter's
# AREA in points^2) intentionally maps straight to 'markersize' (plot's
# DIAMETER in points) with no sqrt conversion — that reinterpretation is
# the whole point of clipped_markers/markers: a style dict written as
# s=18 was meant to read as "18 pt marker", but scatter renders it as an
# 18 pt^2 area (~4.8 pt diameter), a much smaller and less predictable
# marker than the number in the settings suggests.
_MARKER_KWARG_MAP = {
    "s": "markersize",
    "facecolors": "markerfacecolor",
    "facecolor": "markerfacecolor",
    "edgecolors": "markeredgecolor",
    "edgecolor": "markeredgecolor",
    "linewidths": "markeredgewidth",
    "linewidth": "markeredgewidth",
}


def _scatter_kwargs_to_plot_kwargs(kwargs):
    out = {}
    for k, v in kwargs.items():
        out[_MARKER_KWARG_MAP.get(k, k)] = v
    return out


def clipped_markers(ax, xe, yn, region, **kwargs):
    """
    ax.plot()-based marker scatter, restricted to points inside `region`,
    with TRUE LINEAR markersize (points diameter) rather than scatter's
    area-based `s` (points^2). Accepts the same style-dict kwargs as
    clipped_scatter (s, facecolors, edgecolors, linewidths, marker, alpha,
    zorder, label, ...) — including custom Path markers — and translates
    them to their ax.plot() equivalents (see _MARKER_KWARG_MAP). No lines
    are drawn between points (linestyle="none").
    """
    xe = np.asarray(xe)
    yn = np.asarray(yn)
    mask = in_region(xe, yn, region)
    if not np.any(mask):
        return
    plot_kwargs = _scatter_kwargs_to_plot_kwargs(kwargs)
    plot_kwargs.setdefault("linestyle", "none")
    # ax.scatter() draws a circle if no marker is given; ax.plot() draws
    # NOTHING (not even a visible point) with marker=None and
    # linestyle="none" — a style dict that relied on scatter's implicit
    # default (never specifying marker=...) would silently vanish here
    # without this fallback.
    plot_kwargs.setdefault("marker", "o")
    ax.plot(xe[mask], yn[mask], **plot_kwargs)


def markers(ax, xe, yn, **kwargs):
    """
    ax.plot()-based marker scatter, with TRUE LINEAR markersize (points
    diameter) rather than scatter's area-based `s` (points^2) — same
    kwarg translation as clipped_markers (see _MARKER_KWARG_MAP and its
    marker="o" fallback), but with NO region-clipping. For callers where
    every point is already known to be within the plotted domain (e.g.
    seismicity/MT-site positions already projected onto a vertical
    section's own along-profile/depth axes) and clipping would be
    redundant.
    """
    xe = np.asarray(xe)
    yn = np.asarray(yn)
    if xe.size == 0:
        return
    plot_kwargs = _scatter_kwargs_to_plot_kwargs(kwargs)
    plot_kwargs.setdefault("linestyle", "none")
    plot_kwargs.setdefault("marker", "o")
    ax.plot(xe, yn, **plot_kwargs)


def apply_label_mode(label, mode):
    """
    Transform a label's text according to `mode`:
      "full"      - unchanged (default)
      "none"      - suppressed entirely (returns None)
      "firstN"    - first N characters, e.g. "first3"
      "lastN"     - last N characters, e.g. "last3"
    Unrecognised modes fall back to "full" (the label is used unchanged)
    rather than raising, since a typo here shouldn't crash a whole map.
    """
    if mode is None or mode == "full":
        return label
    if mode == "none":
        return None
    m = re.match(r"^(first|last)(\d+)$", mode)
    if m:
        n = int(m.group(2))
        return label[:n] if m.group(1) == "first" else label[-n:]
    return label


def clipped_labels(ax, xe, yn, labels, style_dict, region):
    """
    Draw text labels for points inside `region`.

    style_dict must include 'offset_x' and 'offset_y' (km); remaining keys
    are passed to ax.text(). An optional 'stroke' key (dict) activates a
    withStroke path-effect. An optional 'mode' key controls how much of
    each label's text is shown — see apply_label_mode() for the options
    ("full" default, "none", "firstN", "lastN"). Callers may pass a
    shared/global style dict directly — it's copied internally, never
    mutated.
    """
    style_dict = dict(style_dict)
    ox = style_dict.pop("offset_x", 0.0)
    oy = style_dict.pop("offset_y", 0.0)
    stroke = style_dict.pop("stroke", None)
    mode = style_dict.pop("mode", "full")
    if mode == "none":
        return
    path_effects = [pe.withStroke(**stroke)] if stroke else []
    xe = np.asarray(xe, dtype=float)
    yn = np.asarray(yn, dtype=float)
    mask = in_region(xe, yn, region)
    for x, y, lbl, inside in zip(xe, yn, labels, mask):
        if not inside:
            continue
        text = apply_label_mode(lbl, mode)
        if text is None:
            continue
        ax.text(x + ox, y + oy, text,
                path_effects=path_effects if path_effects else None,
                **style_dict)


# =====================================================================
# North arrow
# =====================================================================
def draw_north_arrow(ax, x_km, y_km, region, arrow_style, label_style, length_km=4.0):
    if not in_region(np.array([x_km]), np.array([y_km]), region)[0]:
        return
    ax.annotate("", xy=(x_km, y_km + length_km), xytext=(x_km, y_km),
                arrowprops=dict(arrowstyle="-|>", **arrow_style),
                annotation_clip=True)
    ax.text(x_km, y_km + length_km + 0.8, "N",
            ha="center", va="bottom", clip_on=True, **label_style)


# =====================================================================
# Deterministic panel/colorbar figure layout — guarantees equal x/y (km)
# scale on maps BY CONSTRUCTION, and avoids the tight_layout()-plus-
# space-stealing-colorbar breakage that a very wide/short panel (e.g. a
# long, shallow cross-section) could trigger.
# =====================================================================
def build_panel_figure(panel_w_in, panel_h_in, colorbar, size_label="panel"):
    """
    Given a panel's exact physical size in inches, place it (and an
    optional colorbar, added as EXTRA canvas beyond the panel) via
    explicit inch-based axes placement — never matplotlib's automatic
    colorbar space-stealing (fig.colorbar(..., ax=...)) or tight_layout().

    colorbar : dict with keys
        show      (bool)
        position  ("right" | "left" | "bottom" | "top")
        size      (bar length, fraction of the panel edge it's attached to)
        pad       (inches, gap between panel and bar)
        aspect    (bar length / bar thickness)

    Returns (fig, ax, cax) — cax is the colorbar axes, or None if
    colorbar["show"] is False.
    """
    show = colorbar["show"]
    pos = colorbar["position"].lower()
    if pos not in ("right", "left", "bottom", "top"):
        raise ValueError(
            f"colorbar position {colorbar['position']!r} is not valid. "
            "Choose 'right', 'left', 'bottom', or 'top'."
        )

    pad_in = colorbar["pad"]
    if show and pos != "right":
        # For every position except "right", the colorbar's dedicated gap
        # is also exactly where the main axes' OWN decorations render,
        # using ordinary (non-managed) matplotlib layout that doesn't
        # know a colorbar is there: y tick labels + ylabel just left of
        # the axes ("left"), x tick labels + xlabel just below ("bottom"),
        # and the plot title just above ("top"). A plain pad_in (a few
        # tenths of an inch) isn't enough room for that text, so it
        # collides with the colorbar. Reserve extra clearance sized from
        # the actual font sizes in use, on top of the requested pad.
        tick_size = colorbar.get("tick_size", 7)
        label_size = colorbar.get("label_size", 8)
        title_size = colorbar.get("title_size", 9)
        text_in = lambda *sizes: sum(sizes) / 72.0 * 1.6 + 0.05
        extra_clearance = {
            "left":   text_in(tick_size, label_size),
            "bottom": text_in(tick_size, label_size),
            "top":    text_in(title_size),
        }[pos]
        pad_in += extra_clearance

    bar_len_in = bar_thick_in = 0.0
    cbar_w_in = cbar_h_in = 0.0
    if show:
        if pos in ("right", "left"):
            bar_len_in = colorbar["size"] * panel_h_in
            cbar_w_in = bar_thick_in = bar_len_in / colorbar["aspect"]
        else:
            bar_len_in = colorbar["size"] * panel_w_in
            cbar_h_in = bar_thick_in = bar_len_in / colorbar["aspect"]

    fig_w_in = panel_w_in + (cbar_w_in + pad_in if cbar_w_in else 0.0)
    fig_h_in = panel_h_in + (cbar_h_in + pad_in if cbar_h_in else 0.0)
    print(f"Figure size ({size_label}): {fig_w_in:.2f} × {fig_h_in:.2f} in "
          f"({size_label} {panel_w_in:.2f} × {panel_h_in:.2f} in)")

    fig = plt.figure(figsize=(fig_w_in, fig_h_in))

    panel_left   = (cbar_w_in + pad_in) / fig_w_in if (show and pos == "left") else 0.0
    panel_bottom = (cbar_h_in + pad_in) / fig_h_in if (show and pos == "bottom") else 0.0
    panel_w_frac = panel_w_in / fig_w_in
    panel_h_frac = panel_h_in / fig_h_in
    ax = fig.add_axes([panel_left, panel_bottom, panel_w_frac, panel_h_frac])

    cax = None
    if show:
        bar_len_frac = (bar_len_in / fig_h_in) if pos in ("right", "left") \
            else (bar_len_in / fig_w_in)
        if pos == "right":
            cax = fig.add_axes([
                (panel_w_in + pad_in) / fig_w_in,
                panel_bottom + (panel_h_frac - bar_len_frac) / 2,
                cbar_w_in / fig_w_in, bar_len_frac,
            ])
        elif pos == "left":
            cax = fig.add_axes([
                0.0,
                panel_bottom + (panel_h_frac - bar_len_frac) / 2,
                cbar_w_in / fig_w_in, bar_len_frac,
            ])
        elif pos == "top":
            cax = fig.add_axes([
                panel_left + (panel_w_frac - bar_len_frac) / 2,
                (panel_h_in + pad_in) / fig_h_in,
                bar_len_frac, cbar_h_in / fig_h_in,
            ])
        elif pos == "bottom":
            cax = fig.add_axes([
                panel_left + (panel_w_frac - bar_len_frac) / 2,
                0.0,
                bar_len_frac, cbar_h_in / fig_h_in,
            ])

    return fig, ax, cax


def finish_panel_colorbar(cax, mappable, label, colorbar):
    """
    Render the colorbar into the cax returned by build_panel_figure().

    colorbar : dict with keys position, label_size, tick_size, nticks
    """
    if cax is None:
        return None
    pos = colorbar["position"].lower()
    orientation = "vertical" if pos in ("right", "left") else "horizontal"
    cbar = cax.figure.colorbar(mappable, cax=cax, orientation=orientation)
    cbar.set_label(label, fontsize=colorbar["label_size"])
    cbar.ax.tick_params(labelsize=colorbar["tick_size"])
    cbar.locator = mpl.ticker.MaxNLocator(nbins=colorbar["nticks"])
    cbar.update_ticks()
    if pos == "left":
        cax.yaxis.set_ticks_position("left")
        cax.yaxis.set_label_position("left")
    if pos == "top":
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")
    return cbar


# =====================================================================
# Secondary lon/lat axes (cosmetic overlay on a UTM-km plot)
# =====================================================================
def add_latlon_ticks(ax, region, nticks, decimals, label_size, tick_size):
    """
    Replace UTM-km tick labels on the primary axes with lon/lat values.
    No extra axes created — existing ticks are reformatted in-place.

    Tick *positions* are chosen at round lon/lat values (e.g. 0.1/0.2/0.5°
    steps, picked automatically via matplotlib's MaxNLocator) rather than at
    evenly spaced UTM-km positions — the round geographic values are then
    converted back to UTM km to place the ticks.
    """
    xmin, xmax, ymin, ymax = region
    e_mid_m = (xmin + xmax) / 2.0 * 1e3
    n_mid_m = (ymin + ymax) / 2.0 * 1e3
    fmt = f"{{:.{decimals}f}}°"

    # Geographic extent of the map along each edge (mid-line of the other axis)
    lon_min, _ = _to_geo.transform(xmin * 1e3, n_mid_m)
    lon_max, _ = _to_geo.transform(xmax * 1e3, n_mid_m)
    _, lat_min = _to_geo.transform(e_mid_m, ymin * 1e3)
    _, lat_max = _to_geo.transform(e_mid_m, ymax * 1e3)

    # Round tick values (nice 1/2/5-type steps), clipped to the map extent
    lon_locator = mpl.ticker.MaxNLocator(nbins=nticks, steps=[1, 2, 5, 10])
    lat_locator = mpl.ticker.MaxNLocator(nbins=nticks, steps=[1, 2, 5, 10])
    lon_vals = [v for v in lon_locator.tick_values(min(lon_min, lon_max), max(lon_min, lon_max))
                if min(lon_min, lon_max) <= v <= max(lon_min, lon_max)]
    lat_vals = [v for v in lat_locator.tick_values(min(lat_min, lat_max), max(lat_min, lat_max))
                if min(lat_min, lat_max) <= v <= max(lat_min, lat_max)]

    # Convert round lon/lat values back to UTM km for tick placement
    e_ticks_km = np.array([_to_utm.transform(lon, (lat_min + lat_max) / 2.0)[0]
                            for lon in lon_vals]) / 1e3
    n_ticks_km = np.array([_to_utm.transform((lon_min + lon_max) / 2.0, lat)[1]
                            for lat in lat_vals]) / 1e3

    lon_labels = [fmt.format(v) for v in lon_vals]
    lat_labels = [fmt.format(v) for v in lat_vals]

    ax.set_xticks(e_ticks_km)
    ax.set_xticklabels(lon_labels, fontsize=tick_size)
    ax.set_xlabel("Longitude", fontsize=label_size)

    ax.set_yticks(n_ticks_km)
    ax.set_yticklabels(lat_labels, fontsize=tick_size)
    ax.set_ylabel("Latitude", fontsize=label_size)


# =====================================================================
# Free-text annotation
# =====================================================================
def draw_annotation(ax, text, pos, style):
    """Draw one line of free text (e.g. a version tag or "DRAFT"
    watermark) if `text` is truthy."""
    if text:
        ax.text(*pos, text, transform=ax.transAxes, zorder=25, **style)


# =====================================================================
# Contour-level resolution / isoline overlay (depth-slice and
# vertical-section plots)
# =====================================================================
def resolve_iso_spec(spec, key):
    """
    An ISO_LEVELS_* setting may be a single "auto"/list (applied to
    every field) or a dict keyed by field name for per-field control.
    Fields absent from the dict default to "auto".

    Was duplicated identically in plot_joint.py and plot_seis.py.
    """
    if isinstance(spec, dict):
        return spec.get(key, "auto")
    return spec


def resolve_iso_levels(data2d, levels_spec, n_auto=6):
    """
    Resolve an (already per-field-resolved, via resolve_iso_spec if
    needed) ISO_LEVELS_* setting into an explicit list of contour levels
    for one panel. "auto"/None picks n_auto evenly spaced levels
    spanning the finite (non-NaN) data range of this particular panel —
    panels differ, so this is computed fresh each time rather than once
    globally. An explicit list/tuple is used verbatim, unchanged, so
    every panel shares the same levels. Returns [] if there's no usable
    finite data (e.g. an all-air/all-NaN panel) or an explicit level
    list was empty.

    Was duplicated identically in plot_joint.py, plot_modem_image.py,
    plot_modem_mesh.py, and plot_seis.py.
    """
    if levels_spec is None or (isinstance(levels_spec, str) and levels_spec.lower() == "auto"):
        finite = data2d[np.isfinite(data2d)]
        if finite.size == 0:
            return []
        vmin, vmax = float(finite.min()), float(finite.max())
        if vmin == vmax:
            return []
        # Interior points only (exclude the flat/degenerate panel edges)
        return list(np.linspace(vmin, vmax, n_auto + 2)[1:-1])
    return list(levels_spec)


def draw_iso_contours(ax, x, y, data2d, levels_spec, iso_style, n_auto=6,
                       label=False, label_fmt="%.2g", label_fontsize=6, key=None):
    """
    Overlay isolines of data2d on ax. x/y may be 1-D (regular grid) or
    2-D matching data2d's shape (curvilinear grid's own aux coords) —
    ax.contour() accepts either. No-op if there are no usable levels.

    iso_style, label/label_fmt/label_fontsize are the caller's own
    ISO_STYLE/ISO_LABEL/ISO_LABEL_FMT/ISO_LABEL_FONTSIZE settings,
    passed explicitly rather than read from the caller's module
    namespace — same convention as draw_north_arrow/clipped_labels/
    draw_annotation above. `key`, if given, resolves a per-field entry
    via resolve_iso_spec() first (levels_spec may then be a dict keyed
    by field name); omit it when there's only one field and levels_spec
    is always a plain "auto"/list.

    Was duplicated (with only the presence/absence of the `key`
    resolution step differing) in plot_joint.py, plot_modem_image.py,
    plot_modem_mesh.py, and plot_seis.py.
    """
    spec = resolve_iso_spec(levels_spec, key) if key is not None else levels_spec
    levels = resolve_iso_levels(data2d, spec, n_auto)
    if not levels:
        return None
    cs = ax.contour(x, y, data2d, levels=levels, **iso_style)
    if label:
        ax.clabel(cs, fmt=label_fmt, fontsize=label_fontsize, inline=True)
    return cs


# =====================================================================
# Vertical-slice profile helpers (generic — no dependence on a specific
# script's VSLICES list or data arrays)
# =====================================================================
def profile_utm_km(vslice):
    """Return (e_km, n_km) endpoint arrays for a VSLICES entry, converting
    latlon -> UTM if needed."""
    p1 = np.asarray(vslice["p1"], float)
    p2 = np.asarray(vslice["p2"], float)
    if vslice.get("coord", "latlon").lower() == "latlon":
        e1, n1 = to_utm_km([p1[0]], [p1[1]])
        e2, n2 = to_utm_km([p2[0]], [p2[1]])
        return np.array([e1[0], e2[0]]), np.array([n1[0], n2[0]])
    else:
        return np.array([p1[0], p2[0]]), np.array([p1[1], p2[1]])


def profile_labels(index):
    """A/A' for index 0, B/B' for index 1, etc."""
    letter = chr(ord('A') + index)
    return letter, letter + "'"


def sample_profile_points(e_ends, n_ends, npts):
    """
    Return (dist_km, e_pts, n_pts, utm_x, utm_xlabel) for npts evenly
    spaced points along the profile.

    utm_x      : 1-D array — easting when |Δe| >= |Δn|, northing otherwise.
    utm_xlabel : matching axis label.
    dist_km    : cumulative distance from p1 (km).
    """
    e_pts = np.linspace(e_ends[0], e_ends[1], npts)
    n_pts = np.linspace(n_ends[0], n_ends[1], npts)
    dist_km = np.sqrt((e_pts - e_ends[0])**2 + (n_pts - n_ends[0])**2)

    de = abs(e_ends[1] - e_ends[0])
    dn = abs(n_ends[1] - n_ends[0])
    if de >= dn:
        utm_x, utm_xlabel = e_pts, "Easting (km)"
    else:
        utm_x, utm_xlabel = n_pts, "Northing (km)"

    return dist_km, e_pts, n_pts, utm_x, utm_xlabel


def project_points_to_profile(e0, n0, e_ends, n_ends, swath_km,
                               z0=None, zmin_km=None, zmax_km=None):
    """
    Project a set of points (e0, n0) — e.g. a seismicity catalog or MT
    site list — onto the straight-line profile from e_ends/n_ends,
    keeping only those within swath_km of the line (and, if z0 is given,
    within [zmin_km, zmax_km] — either bound may be None for unbounded).

    Returns along-profile distance (km) alone if z0 is None, otherwise
    (along_km, z0_kept).
    """
    e0 = np.asarray(e0)
    n0 = np.asarray(n0)
    de = e_ends[1] - e_ends[0]
    dn = n_ends[1] - n_ends[0]
    L = np.sqrt(de**2 + dn**2)
    if L == 0:
        return (np.array([]), np.array([])) if z0 is not None else np.array([])
    ue, un = de / L, dn / L
    ve = e0 - e_ends[0]
    vn = n0 - n_ends[0]
    along = ve * ue + vn * un
    across = np.abs(ve * (-un) + vn * ue)
    mask = (across <= swath_km) & (along >= 0) & (along <= L)
    if z0 is not None:
        z0 = np.asarray(z0)
        zmin = -np.inf if zmin_km is None else zmin_km
        zmax = np.inf if zmax_km is None else zmax_km
        mask = mask & (z0 >= zmin) & (z0 <= zmax)
        return along[mask], z0[mask]
    return along[mask]


# =====================================================================
# Safe file / NetCDF writes
# =====================================================================
def safe_to_netcdf(obj, path):
    """
    Write a Dataset/DataArray to NetCDF, overwriting any existing file at
    `path` even if it's read-only — e.g. left over from an earlier run
    (possibly by a different user/process, or with different
    permissions), which otherwise makes xarray's own to_netcdf() raise
    PermissionError instead of just overwriting it. Removes the stale
    file first (fixing its permissions first if needed), then writes
    normally.

    Was duplicated byte-for-byte in cluster.py, interpolate.py, and
    precompute.py; consolidated here.
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
    """
    Like open(path, 'w', ...), but first clears a read-only leftover
    file at `path` (same PermissionError issue as safe_to_netcdf(), same
    fix). Only cluster.py used this directly, but it's the same pattern
    as safe_to_netcdf() and belongs alongside it rather than living on
    its own in one script.
    """
    p = Path(path)
    if p.exists():
        try:
            p.unlink()
        except PermissionError:
            os.chmod(p, 0o644)
            p.unlink()
    return open(path, "w", **kwargs)


def resolve_path(directory, name):
    """
    Join a bare filename onto a configured output directory. Generalises
    the ncpath()/outpath() one-liners duplicated (under those two
    different names, same logic) in cluster.py, interpolate.py,
    plot_joint.py, and precompute.py. Each script keeps its own
    `ncpath = lambda name: tomomt.resolve_path(NC_DIR, name)` (or
    equivalent) one-liner rather than calling this with the directory
    spelled out every time.
    """
    return str(Path(directory) / name)


# =====================================================================
# INTERP_FILE resolution / interpolation-tag parsing
# =====================================================================

# Canonical short tag for each recognised interpolation-method name
# fragment, used by derive_interp_tag() below. Adopted from plot_joint.py's
# version (the most robust of the three duplicated implementations found
# across plot_joint.py/structure.py/crossplots.py: normalises several
# spellings and warns instead of silently guessing on an unrecognised
# one) as the single canonical mapping for the whole pipeline.
INTERP_METHOD_TAG_MAP = {
    "kriging": "krig", "krige": "krig", "krig": "krig",
    "rbf": "rbf",
    "idw": "idw",
    "nearest_neighbour": "nn", "nearest_neighbor": "nn",
    "nearest": "nn", "nn": "nn",
}


def derive_interp_tag(filename, method_tag_map=None):
    """
    Pull a short, canonical interpolation-method tag out of an
    INTERP_FILE name of the form '..._interp_<method>.nc' (e.g.
    'saba_interp_kriging.nc' -> 'krig'), via `method_tag_map` (default
    INTERP_METHOD_TAG_MAP above). Prints a warning and falls back to the
    raw, lower-cased method substring if it doesn't match any known key
    substring, rather than raising -- this should still run against
    oddly named files, just with a less friendly output filename tag.

    Consolidates plot_joint.py's (this behaviour), structure.py's, and
    crossplots.py's (both a plainer regex-only version with no
    normalisation or warning) three independent implementations. Adopting
    plot_joint.py's richer behaviour for all three is a deliberate
    upgrade for structure.py/crossplots.py, not a silent behind-the-back
    change: their INTERP_TAG now normalises e.g. "kriging" -> "krig" the
    same way plot_joint.py's output filenames already did, instead of
    keeping the literal regex-captured substring.
    """
    if method_tag_map is None:
        method_tag_map = INTERP_METHOD_TAG_MAP
    stem_name = os.path.splitext(os.path.basename(str(filename)))[0]
    marker = "_interp_"
    method_str = stem_name.split(marker, 1)[1] if marker in stem_name else stem_name
    method_str = method_str.lower()
    for key, tag in method_tag_map.items():
        if key in method_str:
            return tag
    print(f"  WARNING: could not recognise interpolation method from "
          f"INTERP_FILE={filename!r} (looked for one of "
          f"{sorted(set(method_tag_map.values()))}) — using raw "
          f"string {method_str!r} in output filenames instead.")
    return method_str


def resolve_interp_file(site_prefix, explicit=None):
    """
    Return `explicit` if given, else auto-pick the newest
    {site_prefix}_interp_*.nc in the current directory. Raises
    FileNotFoundError with a clear message if neither is available.

    Was duplicated identically in structure.py and crossplots.py as
    `_resolve_interp_file()` (reading INTERP_FILE/SITE_PREFIX as module
    globals); consolidated here with both as explicit arguments.
    """
    if explicit is not None:
        return explicit
    candidates = sorted(glob.glob(f"{site_prefix}_interp_*.nc"))
    if not candidates:
        raise FileNotFoundError(
            f"No {site_prefix}_interp_*.nc found in the current directory; "
            f"set INTERP_FILE explicitly."
        )
    return candidates[-1]


def title_suffix(site_prefix, interp_tag):
    """[site_prefix, interp_tag], for ", ".join(...)-ing onto a plot
    title. Was duplicated identically in structure.py and crossplots.py
    as `_title_suffix()`. (plot_joint.py's own `_title_suffix()` returns
    a pre-formatted string instead of a list and was left as-is — see
    module docstring.)"""
    return [site_prefix, interp_tag]


def group_label(fields, label):
    """label if given, else fields dash-joined (e.g. ("rho", "vp") ->
    "rho-vp"). Was duplicated identically in structure.py and
    crossplots.py as `_group_label()`."""
    return label if label else "-".join(fields)


def maybe_show(show_plots):
    """
    Call plt.show() only if `show_plots` is True *and* matplotlib is
    actually running an interactive backend (mpl.is_interactive() --
    true in Spyder's own console/Qt backend, false for the default
    non-interactive "Agg" backend a plain terminal or batch job gets).
    Whatever was already saved to disk (by save_fig()/save_paris(),
    always called first) is unaffected either way -- this only controls
    the on-screen pop-up.

    Was duplicated in plot_joint.py (as a no-arg `_maybe_show()` reading
    a SHOW_PLOTS global) and, identically to each other, in structure.py
    and crossplots.py (as `_maybe_show(fig)`, with `fig` accepted but
    never actually used in the body -- plt.show() shows whatever
    figure(s) are open regardless of which one is passed in). Each
    script keeps its own zero/one-line wrapper reading its own
    SHOW_PLOTS global, e.g. `def _maybe_show(): tomomt.maybe_show(SHOW_PLOTS)`.
    """
    if show_plots and mpl.is_interactive():
        plt.show()


# =====================================================================
# Figure saving
# =====================================================================
def save_fig(fig, stem, plot_dir, plot_formats, dpi, bbox_inches="tight",
             verbose=True):
    """
    Save `fig` once per extension in `plot_formats` (each entry already
    including its leading dot, e.g. [".png", ".pdf"], matching
    cluster.py's/plot_joint.py's own PLOT_FORMATS convention) under
    `plot_dir`, printing "  Saved: <path>" for each if `verbose`. Returns
    the list of saved paths.

    Was duplicated byte-for-byte in cluster.py and plot_joint.py.
    """
    paths = []
    for fmt in plot_formats:
        out = os.path.join(plot_dir, stem + fmt)
        fig.savefig(out, dpi=dpi, bbox_inches=bbox_inches)
        if verbose:
            print(f"  Saved: {out}")
        paths.append(out)
    return paths


def save_paris(fig, stem, outdir, plot_formats, dpi, close=True):
    """
    Save `fig` once per format in `plot_formats` (bare extensions with no
    leading dot, e.g. ["png", "pdf"], matching structure.py's/
    crossplots.py's own PLOT_FORMATS convention -- note this is the
    opposite convention from save_fig() above's leading-dot formats;
    each script's own PLOT_FORMATS setting already matches whichever of
    the two save functions it calls, so this isn't something callers
    need to convert), each with its mtime set to now in Europe/Paris
    local time (PARIS_TZ above). Closes `fig` afterwards unless
    `close=False`. Returns a list of the saved paths (pathlib.Path).

    Was duplicated identically in structure.py and crossplots.py.
    """
    ts = datetime.now(PARIS_TZ).timestamp()
    paths = []
    for fmt in plot_formats:
        path = Path(outdir) / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi)
        os.utime(path, (ts, ts))
        paths.append(path)
    if close:
        plt.close(fig)
    return paths


def zip_outputs(paths, project_name, output_dir):
    """
    Bundle `paths` into a single zip named <project_name>_YYYYMMDD_HHMM.zip
    (Paris time) under `output_dir`, with each member's internal mtime
    also set to the Paris-local packaging time. Returns the zip path.

    Was duplicated identically in structure.py and crossplots.py.
    """
    now_paris = datetime.now(PARIS_TZ)
    zip_name = f"{project_name}_{now_paris.strftime('%Y%m%d_%H%M')}.zip"
    zip_path = Path(output_dir) / zip_name
    date_time = now_paris.timetuple()[:6]

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            p = Path(p)
            zi = zipfile.ZipInfo(p.name, date_time=date_time)
            zi.compress_type = zipfile.ZIP_DEFLATED
            with open(p, "rb") as f:
                zf.writestr(zi, f.read())
    return zip_path


# =====================================================================
# Joint-grid loading (structure.py / crossplots.py)
# =====================================================================
def load_joint_grid(interp_file):
    """
    Load an INTERP_FILE NetCDF and confirm it is a genuinely regular
    UTM-km "joint" target grid (see interpolate.py's TARGET_GRID), i.e.
    1-D depth/northing/easting coordinate arrays rather than 2-D aux
    coords (the reused "seismic" native grid, structured only in
    row/col index space). Raises ValueError with a clear message
    otherwise, rather than silently proceeding with a grid mode the
    caller isn't written to handle.

    Was duplicated identically (module-name in the error message aside)
    in structure.py and crossplots.py.
    """
    import xarray as xr  # local import: not every tomomt user needs xarray

    ds = xr.open_dataset(interp_file)

    # interpolate.py writes "target_grid_mode"; older or alternate builds may
    # write "target_grid".  Accept either so this function works with all
    # INTERP_FILE versions without requiring a file rebuild.
    grid_mode = (ds.attrs.get("target_grid_mode")
                 or ds.attrs.get("target_grid")
                 or None)
    has_1d_coords = all(
        name in ds.coords and ds.coords[name].ndim == 1
        for name in ("depth", "northing", "easting")
    )
    if grid_mode == "seismic" or not has_1d_coords:
        raise ValueError(
            f"{interp_file} is not a regular 'joint' UTM grid "
            f"(target_grid_mode={grid_mode!r}). This function only supports "
            f"TARGET_GRID='joint'. Re-run interpolate.py with "
            f"TARGET_GRID='joint' to produce a compatible INTERP_FILE."
        )
    return ds


def get_field(ds, key):
    """
    (values, depth, northing, easting) for data variable `key` of an
    xarray Dataset loaded by load_joint_grid() -- values transposed to
    (depth, northing, easting) order and cast to float64. Raises KeyError
    with the list of available fields if `key` isn't present.

    Was duplicated byte-for-byte in structure.py and crossplots.py.
    """
    if key not in ds.data_vars:
        raise KeyError(
            f"Field '{key}' not found in {ds.encoding.get('source', 'INTERP_FILE')}; "
            f"available fields: {sorted(ds.data_vars)}"
        )
    depth = ds.coords["depth"].values
    northing = ds.coords["northing"].values
    easting = ds.coords["easting"].values
    values = ds[key].transpose("depth", "northing", "easting").values.astype(float)
    return values, depth, northing, easting


# =====================================================================
# Map-figure sizing (cluster.py / plot_joint.py)
# =====================================================================
def map_panel_size_in(fig_width_cm, xmin, xmax, ymin, ymax):
    """
    (panel_w_in, panel_h_in) for a map panel fig_width_cm wide, with
    height set so the panel's aspect ratio matches
    (xmax - xmin) : (ymax - ymin) exactly -- a true equal-scale-in-km
    map, not a matplotlib "equal" aspect fudge applied after the fact.

    Was duplicated (same two lines) three times: as the first two lines
    of `create_map_figure()` in cluster.py and plot_joint.py (both now
    call build_map_figure() below instead), and again inside plot_joint.py's
    `create_map_figure_pair()`, which needs the bare width/height (it
    passes them on to its own dual-panel layout, not build_panel_figure
    directly) -- that one now calls this directly.
    """
    map_w_in = fig_width_cm / 2.54
    map_h_in = map_w_in * (ymax - ymin) / (xmax - xmin)
    return map_w_in, map_h_in


def build_map_figure(fig_width_cm, xmin, xmax, ymin, ymax, colorbar_settings,
                      size_label="map"):
    """
    Panel figure sized to fig_width_cm wide, with height set so the map's
    aspect ratio matches (xmax - xmin) : (ymax - ymin) exactly. Thin
    wrapper around build_panel_figure() above with the width/height
    arithmetic factored out into map_panel_size_in().

    Was duplicated identically as `create_map_figure()` in cluster.py and
    plot_joint.py, each closing over its own FIG_WIDTH/xmin/xmax/ymin/ymax
    globals and its own `_colorbar_settings()` closure; both now call this
    with those values passed explicitly.
    """
    map_w_in, map_h_in = map_panel_size_in(fig_width_cm, xmin, xmax, ymin, ymax)
    return build_panel_figure(map_w_in, map_h_in, colorbar_settings,
                               size_label=size_label)


# =====================================================================
# SECTION: FEMTIC I/O (consolidated from femtic.py)
# =====================================================================
# Pulled in so interpolate.py / plot_femtic_mesh.py no longer need to
# import femtic.py for kind="femtic_points" sources. Only the read-side
# subset actually exercised by this pipeline is here -- not a full port
# of femtic.py (6500+ lines covering data-file I/O, distortion decoding,
# the resistivity-block read->NPZ->modify->write workflow, NPZ<->VTK/
# NetCDF conversion, ensemble generation, and a CLI). Left out
# deliberately:
#   - insert_model / read_model_to_npz / modify_model_npz /
#     write_model_from_npz / summarise_model_file / _print_model_summary
#     -- the NPZ round-trip and inversion-side write path; this pipeline
#     only ever reads a resistivity block for visualisation.
#   - get_roughness / make_prior_cov / matrix_reduce / etc. (femtic.py's
#     own Section 2) -- these are re-exports from ensembles.py, which is
#     exactly the hard unconditional-import dependency this consolidation
#     removes; nothing here should reintroduce it.
#   - tet_volumes / build_region_geometry / _point_in_tet /
#     extract_borehole_log / utm_to_model / latlon_to_model and the rest
#     of femtic.py's geometry/coordinate helpers -- not called by any
#     consumer script today. Add them here if/when a script needs them,
#     rather than porting speculatively.
#
# estimate_utm_origin() below originally reached its lat/lon<->UTM
# conversion through femtic.py's own `_utl()` -> bare `import util`.
# util.py's latlon_to_utm_zn()/utm_to_latlon_zn() are copied in verbatim
# below (pyproj primary path + dependency-free Helmert/Bowring-series
# fallback, accurate to <1mm within a zone, in case pyproj is ever
# unavailable at runtime) rather than re-derived, so this keeps femtic.py's
# exact original numerics instead of a from-scratch reimplementation.
# util.py itself is NOT made an import of tomomt.py -- it's a 3200-line
# general-purpose module (workspace HDF5 I/O, petrophysics, archiving,
# ...) and only these two functions are actually exercised by this
# pipeline's FEMTIC-origin workflow.

_ANISO_ISO = 0      # isotropic
_ANISO_TI = 1       # transverse isotropy (rhoXX, rhoYY, strike, dip)
_ANISO_GA = 2       # general anisotropy  (rhoXX, rhoYY, rhoZZ, strike, dip, slant)


def latlon_to_utm_zn(lat, lon, zone, northern):
    """Convert WGS-84 geographic coordinates to UTM easting/northing [m].

    Takes an explicit zone number and hemisphere flag (rather than an
    EPSG code). Uses pyproj when available; falls back to the Helmert /
    Bowring series (accurate to <1mm within a single UTM zone) when
    pyproj is absent. Copied from util.py (VR/Claude Sonnet 4.6).

    Parameters
    ----------
    lat, lon : decimal degrees (positive = N / E)
    zone : UTM zone number 1-60
    northern : True -> Northern hemisphere (false northing = 0)

    Returns
    -------
    E_m, N_m : UTM easting and northing in metres
    """
    try:
        hemi = "north" if northern else "south"
        crs = f"+proj=utm +zone={zone} +{hemi} +datum=WGS84 +units=m"
        tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        E_m, N_m = tr.transform(float(lon), float(lat))
        return float(E_m), float(N_m)
    except Exception:
        pass

    # Helmert / Bowring series fallback (no external dependency)
    a = 6_378_137.0
    f = 1.0 / 298.257_223_563
    k0 = 0.9996
    E0 = 500_000.0
    N0 = 0.0 if northern else 10_000_000.0
    e2 = 2.0 * f - f * f
    lon0_deg = (zone - 1) * 6 - 180 + 3
    lat_r = np.radians(float(lat))
    lon_r = np.radians(float(lon))
    lon0 = math.radians(lon0_deg)
    N_r = a / np.sqrt(1.0 - e2 * np.sin(lat_r) ** 2)
    T = np.tan(lat_r) ** 2
    C = e2 / (1.0 - e2) * np.cos(lat_r) ** 2
    A2 = np.cos(lat_r) * (lon_r - lon0)
    e4, e6 = e2 ** 2, e2 ** 3
    M = a * (
        (1.0 - e2 / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0) * lat_r
        - (3.0 * e2 / 8.0 + 3.0 * e4 / 32.0 + 45.0 * e6 / 1024.0) * np.sin(2.0 * lat_r)
        + (15.0 * e4 / 256.0 + 45.0 * e6 / 1024.0) * np.sin(4.0 * lat_r)
        - (35.0 * e6 / 3072.0) * np.sin(6.0 * lat_r)
    )
    E_m = E0 + k0 * N_r * (
        A2
        + (1.0 - T + C) * A2 ** 3 / 6.0
        + (5.0 - 18.0 * T + T ** 2 + 72.0 * C - 58.0 * e2 / (1.0 - e2)) * A2 ** 5 / 120.0
    )
    N_m = N0 + k0 * (
        M + N_r * np.tan(lat_r) * (
            A2 ** 2 / 2.0
            + (5.0 - T + 9.0 * C + 4.0 * C ** 2) * A2 ** 4 / 24.0
            + (61.0 - 58.0 * T + T ** 2 + 600.0 * C
               - 330.0 * e2 / (1.0 - e2)) * A2 ** 6 / 720.0
        )
    )
    return float(E_m), float(N_m)


def utm_to_latlon_zn(E_m, N_m, zone, northern):
    """Convert UTM easting/northing [m] to WGS-84 (lat, lon) decimal degrees.

    Takes an explicit zone number and hemisphere flag (rather than an
    EPSG code). Uses pyproj when available; falls back to the iterative
    inverse Helmert series (accurate to <1mm within a single UTM zone)
    when pyproj is absent. Copied from util.py (VR/Claude Sonnet 4.6).

    Parameters
    ----------
    E_m, N_m : UTM easting and northing in metres
    zone : UTM zone number 1-60
    northern : True -> Northern hemisphere (false northing = 0)

    Returns
    -------
    lat, lon : decimal degrees (positive = N / E)
    """
    try:
        hemi = "north" if northern else "south"
        crs = f"+proj=utm +zone={zone} +{hemi} +datum=WGS84 +units=m"
        tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = tr.transform(float(E_m), float(N_m))
        return float(lat), float(lon)
    except Exception:
        pass

    # Helmert inverse series fallback (Bowring / Snyder)
    a = 6_378_137.0
    f = 1.0 / 298.257_223_563
    k0 = 0.9996
    E0 = 500_000.0
    N0 = 0.0 if northern else 10_000_000.0
    e2 = 2 * f - f * f
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)
    x = float(E_m) - E0
    y = float(N_m) - N0
    M = y / k0
    mu = M / (a * (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256))
    lat1 = (mu
            + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * np.sin(2 * mu)
            + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * np.sin(4 * mu)
            + (151 * e1 ** 3 / 96) * np.sin(6 * mu)
            + (1097 * e1 ** 4 / 512) * np.sin(8 * mu))
    N1 = a / np.sqrt(1 - e2 * np.sin(lat1) ** 2)
    T1 = np.tan(lat1) ** 2
    C1 = e2 / (1 - e2) * np.cos(lat1) ** 2
    R1 = a * (1 - e2) / (1 - e2 * np.sin(lat1) ** 2) ** 1.5
    D = x / (N1 * k0)
    lat = lat1 - (N1 * np.tan(lat1) / R1) * (
        D ** 2 / 2
        - (5 + 3 * T1 + 10 * C1 - 4 * C1 ** 2 - 9 * e2 / (1 - e2)) * D ** 4 / 24
        + (61 + 90 * T1 + 298 * C1 + 45 * T1 ** 2
           - 252 * e2 / (1 - e2) - 3 * C1 ** 2) * D ** 6 / 720)
    lon = lon0 + (
        D
        - (1 + 2 * T1 + C1) * D ** 3 / 6
        + (5 - 2 * C1 + 28 * T1 - 3 * C1 ** 2
           + 8 * e2 / (1 - e2) + 24 * T1 ** 2) * D ** 5 / 120
    ) / np.cos(lat1)
    return float(np.degrees(lat)), float(np.degrees(lon))


def _detect_block_format(region_line):
    """Detect whether a resistivity-block region line is v4 or v5 format.

    v4 (isotropic only): ``ireg rho rho_lo rho_hi n flag`` (col[1] float rho).
    v5 (iso / TI / general anisotropy): ``ireg aniso_type ...`` (col[1] int 0/1/2).
    Returns ``"v4"`` or ``"v5"``.
    """
    parts = region_line.split()
    if len(parts) < 3:
        return "v4"
    try:
        second = int(parts[1])
    except ValueError:
        return "v4"
    if second not in (0, 1, 2):
        return "v4"
    try:
        float(parts[2])
        return "v5"
    except ValueError:
        return "v4"


def _parse_region_line_v4(line):
    """Parse one v4 region line: ``ireg rho rho_lower rho_upper n flag``."""
    parts = line.split()
    if len(parts) < 6:
        raise ValueError(f"Invalid v4 region line (need >=6 columns): {line!r}")
    ireg = int(parts[0])
    rho = float(parts[1])
    rho_lower = float(parts[2])
    rho_upper = float(parts[3])
    n = float(parts[4])
    flag = int(parts[5])
    return ireg, rho, rho_lower, rho_upper, n, flag


def _parse_region_line_v5(line):
    """Parse one v5 (anisotropic) region line into a dict of its fields.

    Keys: ireg, aniso_type, rho_lo, rho_hi, flag, rhoXX, rhoYY, rhoZZ,
    strike, dip, slant, fix_rhoXX, fix_rhoYY, fix_rhoZZ, fix_strike,
    fix_dip, fix_slant.
    """
    parts = line.split()
    if len(parts) < 2:
        raise ValueError(f"Invalid v5 region line (too short): {line!r}")
    ireg = int(parts[0])
    aniso_type = int(parts[1])

    result = dict(
        ireg=ireg, aniso_type=aniso_type,
        rhoXX=0.0, rhoYY=0.0, rhoZZ=0.0,
        strike=0.0, dip=0.0, slant=0.0,
        rho_lo=0.0, rho_hi=0.0, flag=0,
        fix_rhoXX=0, fix_rhoYY=0, fix_rhoZZ=0,
        fix_strike=0, fix_dip=0, fix_slant=0,
    )

    if aniso_type == _ANISO_ISO:
        if len(parts) < 6:
            raise ValueError(f"v5 ISO region line needs >=6 columns: {line!r}")
        rho = float(parts[2])
        result["rhoXX"] = rho
        result["rhoYY"] = rho
        result["rhoZZ"] = rho
        result["rho_lo"] = float(parts[3])
        result["rho_hi"] = float(parts[4])
        result["flag"] = int(parts[5])

    elif aniso_type == _ANISO_TI:
        if len(parts) < 12:
            raise ValueError(f"v5 TI region line needs >=12 columns: {line!r}")
        result["rhoXX"] = float(parts[2])
        result["rhoYY"] = float(parts[3])
        result["rhoZZ"] = float(parts[2])   # TI: rhoZZ = rhoXX
        result["strike"] = float(parts[4])
        result["dip"] = float(parts[5])
        result["slant"] = 0.0
        result["rho_lo"] = float(parts[6])
        result["rho_hi"] = float(parts[7])
        result["fix_rhoXX"] = int(parts[8])
        result["fix_rhoYY"] = int(parts[9])
        result["fix_strike"] = int(parts[10])
        result["fix_dip"] = int(parts[11])

    elif aniso_type == _ANISO_GA:
        if len(parts) < 16:
            raise ValueError(f"v5 GA region line needs >=16 columns: {line!r}")
        result["rhoXX"] = float(parts[2])
        result["rhoYY"] = float(parts[3])
        result["rhoZZ"] = float(parts[4])
        result["strike"] = float(parts[5])
        result["dip"] = float(parts[6])
        result["slant"] = float(parts[7])
        result["rho_lo"] = float(parts[8])
        result["rho_hi"] = float(parts[9])
        result["fix_rhoXX"] = int(parts[10])
        result["fix_rhoYY"] = int(parts[11])
        result["fix_rhoZZ"] = int(parts[12])
        result["fix_strike"] = int(parts[13])
        result["fix_dip"] = int(parts[14])
        result["fix_slant"] = int(parts[15])
    else:
        raise ValueError(f"Unknown aniso_type {aniso_type} in v5 region line: {line!r}")

    return result


def _parse_region_line(line, fmt="v4"):
    """Parse one region line, returning the v4-compatible 6-tuple
    (ireg, rho, rho_lower, rho_upper, n, flag). For v5 lines, ``rho`` is
    ``rhoXX``; full anisotropic parameters are in _parse_region_line_v5().
    ``n`` is 1.0 for v5 lines (field not present in v5 format).
    """
    if fmt == "v5":
        d = _parse_region_line_v5(line)
        return d["ireg"], d["rhoXX"], d["rho_lo"], d["rho_hi"], 1.0, d["flag"]
    return _parse_region_line_v4(line)


def _infer_ocean_present(region1_line, fmt="v4"):
    """Infer whether region 1 is an 'ocean' fixed block.

    Heuristic (conservative): flag == 1 (fixed) and rho <= 1 Ohm.m (very
    conductive, typical ocean ~0.25 Ohm.m). Override via read_model()'s
    own ``ocean=True/False``.
    """
    _, rho, _, _, _, flag = _parse_region_line(region1_line, fmt=fmt)
    return (flag == 1) and (rho <= 1.0)


def read_model(model_file, model_trans="log10", out=True, *,
               ocean=None, include_fixed=False):
    """Read a FEMTIC resistivity_block_iterX.dat and return a model vector.

    Default behaviour (include_fixed=False): always excludes region 0
    (air), excludes any region with flag == 1, and additionally excludes
    region 1 if it is treated as ocean (auto-inferred unless overridden
    via ``ocean=...``).

    Parameters
    ----------
    model_file : str or Path
    model_trans : "log10" (default, returns log10(rho)) or "none"/"rho"
    out : bool -- print a short info line if True
    ocean : None (auto-infer) | True | False
    include_fixed : bool -- if True, include every region

    Returns
    -------
    np.ndarray -- 1-D vector of model parameters, region-index order
    (selected regions only).
    """
    model_path = Path(model_file)

    with model_path.open("r", encoding="utf-8", errors="replace") as f:
        header = f.readline()
        hdr_parts = header.split()
        if len(hdr_parts) < 2:
            raise ValueError(f"Invalid resistivity block header: {hdr_parts!r}")
        nelem = int(hdr_parts[0])
        nreg = int(hdr_parts[1])

        for _ in range(nelem):
            f.readline()

        if nreg <= 0:
            raise ValueError("No regions in resistivity block (nreg<=0).")

        first_region_line = f.readline()
        if not first_region_line:
            raise ValueError("Unexpected EOF before first region line.")
        fmt = _detect_block_format(first_region_line)

        region_lines = []
        region_rho = np.zeros(nreg, dtype=float)
        region_flag = np.zeros(nreg, dtype=int)

        all_region_lines = [first_region_line] + [f.readline() for _ in range(nreg - 1)]
        for i, line in enumerate(all_region_lines):
            if not line:
                raise ValueError(
                    f"Unexpected EOF while reading region lines: expected {nreg}, got {i}."
                )
            ireg, rho, _, _, _, flag = _parse_region_line(line, fmt=fmt)
            if ireg != i:
                raise ValueError(f"Expected region index {i} at line {i}, got {ireg}.")
            region_lines.append(line)
            region_rho[i] = rho
            region_flag[i] = flag

    ocean_present = False
    if nreg > 1:
        if ocean is None:
            ocean_present = _infer_ocean_present(region_lines[1], fmt=fmt)
        else:
            ocean_present = bool(ocean)

    fixed_mask = np.zeros(nreg, dtype=bool)
    fixed_mask[0] = True  # air always fixed here
    fixed_mask |= (region_flag == 1)
    if nreg > 1 and ocean_present:
        fixed_mask[1] = True

    if include_fixed:
        sel = np.arange(nreg, dtype=int)
    else:
        sel = np.where(~fixed_mask)[0]

    rho_sel = region_rho[sel].astype(float, copy=False)

    if model_trans.lower() == "log10":
        out_vec = np.log10(rho_sel)
    elif model_trans.lower() in ("none", "rho"):
        out_vec = rho_sel
    else:
        raise ValueError(f"Unknown model_trans={model_trans!r}; use 'log10' or 'none'.")

    if out:
        n_fixed = int(fixed_mask.sum())
        print(
            f"read_model: file={model_path.name}, fmt={fmt}, nelem={nelem}, nreg={nreg}, "
            f"ocean_present={ocean_present}, fixed={n_fixed}, returned={out_vec.size}."
        )

    return out_vec


def read_femtic_mesh(mesh_path):
    """Read a FEMTIC TETRA mesh file.

    Returns
    -------
    nodes : ndarray, shape (nn, 3) -- node coordinates [x, y, z]
    conn  : ndarray, shape (nelem, 4) -- tetrahedral connectivity (0-based)
    """
    with open(mesh_path, "r", errors="ignore") as f:
        header = f.readline().strip()
        if header.upper() != "TETRA":
            raise ValueError(f"Unsupported mesh type '{header}', expected 'TETRA'.")

        nn_line = f.readline().split()
        if not nn_line:
            raise ValueError("Missing node count after 'TETRA' header.")
        nn = int(nn_line[0])

        nodes = np.empty((nn, 3), dtype=float)
        for _ in range(nn):
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF while reading node coordinates.")
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Node line has too few columns: {line!r}")
            idx = int(parts[0])
            x, y, z = map(float, parts[1:4])
            if not (0 <= idx < nn):
                raise ValueError(f"Node index {idx} out of range 0..{nn-1}.")
            nodes[idx] = (x, y, z)

        nelem_line = f.readline().split()
        if not nelem_line:
            raise ValueError("Missing element count line after node block.")
        nelem = int(nelem_line[0])

        conn = np.empty((nelem, 4), dtype=int)
        for _ in range(nelem):
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF while reading element block.")
            parts = line.split()
            if len(parts) < 9:
                raise ValueError(f"Element line has too few columns: {line!r}")
            ie = int(parts[0])
            n1, n2, n3, n4 = map(int, parts[-4:])
            if not (0 <= ie < nelem):
                raise ValueError(f"Element index {ie} out of range 0..{nelem-1}.")
            conn[ie] = (n1, n2, n3, n4)

    return nodes, conn


def read_resistivity_block(block_path):
    """Read a FEMTIC resistivity_block_iterX.dat and return region-based data.

    Returns a dict with keys: nelem, nreg, fmt, region_of_elem, region_rho,
    region_rho_lower, region_rho_upper, region_n, region_flag, and (v5
    only) region_aniso_type, region_rhoYY, region_rhoZZ, region_strike,
    region_dip, region_slant.
    """
    with open(block_path, "r", errors="ignore") as f:
        first = f.readline().split()
        if len(first) < 2:
            raise ValueError("First line must contain 'nelem nreg'.")
        nelem = int(first[0])
        nreg = int(first[1])

        region_of_elem = np.empty(nelem, dtype=int)
        for _ in range(nelem):
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF while reading element-region map.")
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Element-region line has too few columns: {line!r}")
            ie = int(parts[0])
            ireg = int(parts[1])
            if not (0 <= ie < nelem):
                raise ValueError(f"Element index {ie} out of range 0..{nelem-1}.")
            region_of_elem[ie] = ireg

        region_rho = np.empty(nreg, dtype=float)
        region_rho_lower = np.empty(nreg, dtype=float)
        region_rho_upper = np.empty(nreg, dtype=float)
        region_n = np.empty(nreg, dtype=float)
        region_flag = np.empty(nreg, dtype=int)
        region_aniso_type = np.zeros(nreg, dtype=int)
        region_rhoYY = np.empty(nreg, dtype=float)
        region_rhoZZ = np.empty(nreg, dtype=float)
        region_strike = np.zeros(nreg, dtype=float)
        region_dip = np.zeros(nreg, dtype=float)
        region_slant = np.zeros(nreg, dtype=float)

        first_region_line = f.readline()
        if not first_region_line:
            raise ValueError("Unexpected EOF before first region line (read_resistivity_block).")
        fmt = _detect_block_format(first_region_line)
        all_rlines = [first_region_line] + [f.readline() for _ in range(nreg - 1)]

        for line in all_rlines:
            if not line:
                raise ValueError("Unexpected EOF while reading region lines.")
            ireg, rho, rho_min, rho_max, n, flag = _parse_region_line(line, fmt=fmt)
            if not (0 <= ireg < nreg):
                raise ValueError(f"Region index {ireg} out of range 0..{nreg-1}.")
            region_rho[ireg] = rho
            region_rho_lower[ireg] = rho_min
            region_rho_upper[ireg] = rho_max
            region_n[ireg] = n
            region_flag[ireg] = flag
            if fmt == "v5":
                d = _parse_region_line_v5(line)
                region_aniso_type[ireg] = d["aniso_type"]
                region_rhoYY[ireg] = d["rhoYY"]
                region_rhoZZ[ireg] = d["rhoZZ"]
                region_strike[ireg] = d["strike"]
                region_dip[ireg] = d["dip"]
                region_slant[ireg] = d["slant"]
            else:
                region_rhoYY[ireg] = rho
                region_rhoZZ[ireg] = rho

    result = {
        "nelem": np.array(nelem, dtype=int),
        "nreg": np.array(nreg, dtype=int),
        "fmt": fmt,
        "region_of_elem": region_of_elem,
        "region_rho": region_rho,
        "region_rho_lower": region_rho_lower,
        "region_rho_upper": region_rho_upper,
        "region_n": region_n,
        "region_flag": region_flag,
    }
    if fmt == "v5":
        result.update({
            "region_aniso_type": region_aniso_type,
            "region_rhoYY": region_rhoYY,
            "region_rhoZZ": region_rhoZZ,
            "region_strike": region_strike,
            "region_dip": region_dip,
            "region_slant": region_slant,
        })
    return result


def build_element_arrays(nodes, conn, region_of_elem, region_rho,
                          region_rho_lower, region_rho_upper, region_n,
                          region_flag, clip_eps=1.0e-30):
    """Build per-element arrays (centroids, log10 resistivity, bounds,
    flags, n) from a FEMTIC mesh + read_resistivity_block() output.

    Returns a dict with keys: centroid, region, log10_resistivity,
    rho_lower, rho_upper, flag, n.
    """
    nodes = np.asarray(nodes, dtype=float)
    conn = np.asarray(conn, dtype=int)
    region_of_elem = np.asarray(region_of_elem, dtype=int)

    nelem = conn.shape[0]
    if region_of_elem.shape[0] != nelem:
        raise ValueError("region_of_elem length does not match number of elements.")

    coords = nodes[conn]
    centroid = coords.mean(axis=1)

    rho = np.clip(region_rho, clip_eps, np.inf)
    rho_min = np.clip(region_rho_lower, clip_eps, np.inf)
    rho_max = np.clip(region_rho_upper, clip_eps, np.inf)

    rid = region_of_elem
    rho_elem = rho[rid]
    rho_min_elem = rho_min[rid]
    rho_max_elem = rho_max[rid]
    n_elem = region_n[rid]
    flag_elem = region_flag[rid]

    log10_rho = np.log10(rho_elem)
    log10_rho_min = np.log10(rho_min_elem)
    log10_rho_max = np.log10(rho_max_elem)

    return {
        "centroid": centroid,
        "region": rid,
        "log10_resistivity": log10_rho,
        "rho_lower": log10_rho_min,
        "rho_upper": log10_rho_max,
        "flag": flag_elem,
        "n": n_elem,
    }


def read_site_position(observe_file, site_number):
    """Return (x_m, y_m) model-local position for site_number from observe.dat.

    Scans linearly for site-header lines matching ``int int float float``
    and returns the (x, y) pair (converted from km) that matches
    site_number (1-based).
    """
    if not os.path.isfile(observe_file):
        raise FileNotFoundError(f"observe.dat not found: {observe_file}")

    with open(observe_file) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                n1 = int(parts[0])
                int(parts[1])
                x_km = float(parts[2])
                y_km = float(parts[3])
            except ValueError:
                continue
            if n1 == site_number:
                return x_km * 1000.0, y_km * 1000.0

    raise ValueError(f"Site {site_number} not found in {observe_file}.  Check SITE_NUMBER.")


def read_site_dat(path, site_names=None):
    """Read site positions from a FEMTIC sitelist CSV (mt_make_sitelist.py).

    Format (comma-separated, no header; ``#``-prefixed lines ignored):
    name, lat, lon, elev, sitenum, easting, northing (easting/northing in
    UTM metres, lat/lon in decimal degrees).

    Returns a list of dicts with keys name, lat, lon, elev, sitenum,
    easting, northing. ``site_names`` (str or list[str], optional)
    restricts to matching rows.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"site.dat not found: {path}")

    if site_names is not None:
        if isinstance(site_names, str):
            site_names = [site_names]
        site_names = set(site_names)

    sites = []
    with open(path) as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.split("#")[0].strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                raise ValueError(
                    f"{path}:{lineno}: expected 7 columns "
                    f"(name,lat,lon,elev,sitenum,easting,northing), "
                    f"got {len(parts)}: {raw.rstrip()}"
                )
            try:
                name = parts[0]
                lat = float(parts[1])
                lon = float(parts[2])
                elev = float(parts[3])
                sitenum = int(parts[4])
                easting = float(parts[5])
                northing = float(parts[6])
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: cannot parse values: {exc}") from exc
            if site_names is not None and name not in site_names:
                continue
            sites.append(dict(
                name=name, lat=lat, lon=lon, elev=elev,
                sitenum=sitenum, easting=easting, northing=northing,
            ))
    return sites


def estimate_utm_origin(calibration_sites, observe_file, zone, northern, *,
                         site_dat=None, out=True):
    """Estimate UTM coordinates of the FEMTIC mesh centre.

    Two methods, selected automatically:

    Bounding-box centre (default when calibration_sites is empty): reads
    all site UTM coordinates from site_dat (mt_make_sitelist.py's sitelist)
    and sets the origin to the midpoint of the bounding box.

    Calibration-site pairs (used when calibration_sites is non-empty):
    each entry gives a site whose model-local position (from observe_file,
    or inline x_km/y_km) and geographic position are both known; with N
    sites the origin is the mean of the N implied offsets.

    Parameters
    ----------
    calibration_sites : list of dict, each with "site" (int), "crs"
        ("latlon" or "utm"), "coords" ([lon, lat] or [E_m, N_m]). May be
        empty -- triggers the bounding-box fallback.
    observe_file : str -- path to observe.dat (calibration-site method)
    zone : int -- UTM zone number
    northern : bool -- hemisphere flag
    site_dat : str or None -- FEMTIC sitelist CSV (bounding-box method)
    out : bool -- print per-site details and result if True

    Returns
    -------
    origin_E, origin_N : float -- estimated UTM coordinates of the mesh
    centre [m].
    """
    if not calibration_sites:
        if site_dat is None:
            raise ValueError(
                "estimate_utm_origin: calibration_sites is empty and "
                "site_dat is None -- cannot estimate origin."
            )
        rows = read_site_dat(site_dat)
        if not rows:
            raise ValueError(f"estimate_utm_origin: site_dat {site_dat!r} is empty.")
        eastings = np.array([r["easting"] for r in rows])
        northings = np.array([r["northing"] for r in rows])
        origin_E = float((eastings.min() + eastings.max()) / 2.0)
        origin_N = float((northings.min() + northings.max()) / 2.0)
        if out:
            center_lat, center_lon = utm_to_latlon_zn(origin_E, origin_N, zone, northern)
            print("Estimating mesh-centre UTM origin from sitelist bounding box:")
            print(f"  sites          : {len(rows)}")
            print(f"  E range        : {eastings.min():.1f} - {eastings.max():.1f} m")
            print(f"  N range        : {northings.min():.1f} - {northings.max():.1f} m")
            print(f"  UTM_ORIGIN_E   = {origin_E:.1f}")
            print(f"  UTM_ORIGIN_N   = {origin_N:.1f}")
            print(f"  UTM_ORIGIN_LAT = {center_lat:.6f}")
            print(f"  UTM_ORIGIN_LON = {center_lon:.6f}")
            print("  Copy these values into the Configuration block.")
            print()
        return origin_E, origin_N

    offsets_E = []
    offsets_N = []

    if out:
        print("Estimating mesh-centre UTM origin from calibration sites:")
        print(f"  {'site':>5}  {'x_model':>10}  {'y_model':>10}  "
              f"{'E_utm':>12}  {'N_utm':>14}  {'dE':>8}  {'dN':>8}")
        print("  " + "-" * 77)

    for entry in calibration_sites:
        site_num = int(entry["site"])
        crs = str(entry["crs"])
        coords = list(entry["coords"])

        if "x_km" in entry and "y_km" in entry:
            x_m = float(entry["x_km"]) * 1e3
            y_m = float(entry["y_km"]) * 1e3
        else:
            x_m, y_m = read_site_position(observe_file, site_num)

        if crs == "latlon":
            lon_deg, lat_deg = coords
            E_site, N_site = latlon_to_utm_zn(lat_deg, lon_deg, zone, northern)
        elif crs == "utm":
            E_site, N_site = float(coords[0]), float(coords[1])
        else:
            raise ValueError(f"Calibration site {site_num}: unknown crs={crs!r}. Use 'latlon' or 'utm'.")

        oE = E_site - x_m
        oN = N_site - y_m
        offsets_E.append(oE)
        offsets_N.append(oN)

        if out:
            print(f"  {site_num:>5}  {x_m/1000:>10.3f}  {y_m/1000:>10.3f}  "
                  f"{E_site:>12.1f}  {N_site:>14.1f}  {oE:>8.1f}  {oN:>8.1f}")

    origin_E = float(np.mean(offsets_E))
    origin_N = float(np.mean(offsets_N))

    if out and len(calibration_sites) > 1:
        print()
        print(f"  {'site':>5}  {'res_E (m)':>10}  {'res_N (m)':>10}")
        print("  " + "-" * 30)
        for entry, oE, oN in zip(calibration_sites, offsets_E, offsets_N):
            print(f"  {int(entry['site']):>5}  {oE - origin_E:>10.2f}  {oN - origin_N:>10.2f}")
        rms_E = float(np.sqrt(np.mean((np.array(offsets_E) - origin_E) ** 2)))
        rms_N = float(np.sqrt(np.mean((np.array(offsets_N) - origin_N) ** 2)))
        print(f"  {'RMS':>5}  {rms_E:>10.2f}  {rms_N:>10.2f}")

    print()
    print("  Estimated mesh-centre UTM origin:")
    print(f"    UTM_ORIGIN_E = {origin_E:.1f}")
    print(f"    UTM_ORIGIN_N = {origin_N:.1f}")
    print("  Copy these values into the Configuration block.")
    print()

    return origin_E, origin_N


# =====================================================================
# SECTION: ModEM I/O (consolidated from modem.py)
# =====================================================================
# Only the three readers this pipeline's precompute.py actually uses.
# Not a port of modem.py (5700+ lines: Jacobian I/O, model writers,
# UBC/RLM format conversion, NetCDF export, denoising/regularisation
# utilities needing numba/pywt). Left out deliberately -- none of these
# are called by any consumer script today; add on demand rather than
# speculatively:
#   - read_data_jac, write_data, read_mod_aniso, and every write_*/
#     convert_* routine
#   - anything gated on numba (@jit) or pywt in the original module --
#     neither read_mod, read_data, nor get_topo touch either.


def read_mod(file=None, modext=".rho", trans="LINEAR", blank=1.e-30, out=True):
    """Read a ModEM model input file. Returns mval in physical units by
    default (trans="LINEAR"); pass trans="LOG10"/"LOGE" to get back
    log-transformed values instead.

    Returns
    -------
    dx, dy, dz : ndarray -- cell sizes along each axis
    mval : ndarray, shape (nx, ny, nz) -- model values
    reference : list[float] -- [x0, y0, z0] reference coordinates
    trans : str -- echoes the requested output transform
    """
    modf = file + modext

    with open(modf, "r") as f:
        lines = f.readlines()

    lines = [line.split() for line in lines]
    dims = [int(sub) for sub in lines[1][0:3]]
    nx, ny, nz = dims
    trns = lines[1][4]
    dx = np.array([float(sub) for sub in lines[2]])
    dy = np.array([float(sub) for sub in lines[3]])
    dz = np.array([float(sub) for sub in lines[4]])

    mval = np.array([])
    for line in lines[5:-2]:
        line = np.flipud(line)
        mval = np.append(mval, np.array([float(sub) for sub in line]))

    if out:
        print("values in " + file + " are: " + trns)

    if trns == "LOGE":
        mval = np.exp(mval)
    elif trns == "LOG10":
        mval = np.power(10.0, mval)
    elif trns == "LINEAR":
        pass
    else:
        raise ValueError(f"Transformation: {trns} not defined!")

    # here mval should be in physical units, not log...
    mval[np.where(np.abs(mval) < blank)] = blank

    if "loge" in trans.lower() or "ln" in trans.lower():
        mval = np.log(mval)
        if out:
            print("values transformed to: " + trans)
    elif "log10" in trans.lower():
        mval = np.log10(mval)
        if out:
            print("values transformed to: " + trans)
    else:
        if out:
            print("values transformed to: " + trans)

    mval = mval.reshape(dims, order="F")

    reference = [float(sub) for sub in lines[-2][0:3]]

    if out:
        print("read_model: %i x %i x %i model read from %s" % (nx, ny, nz, file))

    return dx, dy, dz, mval, reference, trans


def read_data(Datfile=None, modext=".dat", out=True):
    """Read a ModEM input data file.

    Returns
    -------
    Site : ndarray[object] -- site name per data row
    Comp : ndarray[object] -- component code per data row
    Data : ndarray -- data columns (period, geo/model coords, value, error, ...)
    Head : list[str] -- raw header/block-header lines (# and > prefixed)
    """
    file = Datfile + modext

    Data = []
    Site = []
    Comp = []
    Head = []

    with open(file) as fd:
        for line in fd:
            if line.startswith("#") or line.startswith(">"):
                Head.append(line)
                continue

            t = line.split()

            if "PT" in t[7] or "RH" in t[7] or "PH" in t[7]:
                tmp = [
                    float(t[0]), float(t[2]), float(t[3]), float(t[4]),
                    float(t[5]), float(t[6]), float(t[8]),
                    float(t[9]), 0.,
                ]
                Data.append(tmp)
                Site.append([t[1]])
                Comp.append([t[7]])
            else:
                tmp = [
                    float(t[0]), float(t[2]), float(t[3]), float(t[4]),
                    float(t[5]), float(t[6]), float(t[8]),
                    float(t[9]), float(t[10]),
                ]
                Data.append(tmp)
                Comp.append([t[7]])
                Site.append([t[1]])

    Site = [item for sublist in Site for item in sublist]
    Site = np.asarray(Site, dtype=object)
    Comp = [item for sublist in Comp for item in sublist]
    Comp = np.asarray(Comp, dtype=object)
    Data = np.asarray(Data)

    nD = np.shape(Data)
    if out:
        print("readDat: %i data read from %s" % (nD[0], file))

    return Site, Comp, Data, Head


def get_topo(dx=None, dy=None, dz=None, mval=None, ref=[0., 0., 0.],
             mvalair=1.e17, out=True):
    """Extract topography (surface elevation) from a ModEM model.

    Parameters
    ----------
    dx, dy, dz : ndarray -- mesh cell sizes
    mval : ndarray, shape (nx, ny, nz) -- cell resistivities (physical units)
    ref : sequence[float] -- reference coordinates, default [0, 0, 0]
    mvalair : float -- resistivity value marking air cells (physical units)
    out : bool -- print a short info line if True

    Returns
    -------
    xcnt, ycnt : ndarray -- cell-centre coordinates in the x/y plane
    topo : ndarray, shape (nx, ny) -- elevation of the topmost non-air cell
    """
    nx, ny, nz = np.shape(mval)

    x = np.append(0.0, np.cumsum(dx))
    xcnt = 0.5 * (x[0:nx] + x[1:nx + 1]) + ref[0]

    y = np.append(0.0, np.cumsum(dy))
    ycnt = 0.5 * (y[0:ny] + y[1:ny + 1]) + ref[1]

    ztop = np.append(0.0, np.cumsum(dz)) + ref[2]

    topo = np.zeros((nx, ny))
    for ii in np.arange(nx):
        for jj in np.arange(ny):
            col = mval[ii, jj, :]
            nsurf = np.argmax(col < mvalair)
            topo[ii, jj] = ztop[nsurf]

    if out:
        print("get topo: %i x %i cell surfaces marked" % (nx, ny))

    return xcnt, ycnt, topo
