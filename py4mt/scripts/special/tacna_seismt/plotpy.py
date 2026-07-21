"""
plotpy.py
=========
Shared plotting helpers used by tacna_plot_seis.py, tacna_plot_modem_image.py,
and tacna_plot_modem_mesh.py.

These are the genuinely duplicated, run-independent pieces of the three
scripts: UTM<->lon/lat coordinate conversion, colormap loading (matplotlib
name / GMT .cpt file / plain RGB(A) list), hillshading, ModEM
sensitivity-alpha helpers, region-clipped scatter/label helpers, the north
arrow, the deterministic (equal-scale-by-construction) panel/colorbar
layout engine, the lon/lat tick overlay, the free-text annotation, VE-label
positioning, and generic profile point sampling/projection.

None of these read a calling script's global variables — every value that
used to come from a script's own USER SETTINGS is now an explicit
argument (colours, sizes, positions, region bounds, toggles, ...). Each
script still owns its own settings and its own script-specific plotting
code (draw_basemap, draw_features, compute_vertical_slice_*,
plot_vertical_slice); it just imports these shared pieces instead of
redefining them.

Authors: Svetlana Byrdina (SMB) & Volker Rath (DIAS)
AI-assisted development: Claude (Anthropic)
License: GNU General Public License v3 (GPL-3.0-or-later)
AI-generated code — review before use in production.
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LightSource
from pyproj import Transformer
from scipy.ndimage import gaussian_filter


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


def clipped_labels(ax, xe, yn, labels, style_dict, region):
    """
    Draw text labels for points inside `region`.

    style_dict must include 'offset_x' and 'offset_y' (km); remaining keys
    are passed to ax.text(). An optional 'stroke' key (dict) activates a
    withStroke path-effect. Callers may pass a shared/global style dict
    directly — it's copied internally, never mutated.
    """
    style_dict = dict(style_dict)
    ox = style_dict.pop("offset_x", 0.0)
    oy = style_dict.pop("offset_y", 0.0)
    stroke = style_dict.pop("stroke", None)
    path_effects = [pe.withStroke(**stroke)] if stroke else []
    xe = np.asarray(xe, dtype=float)
    yn = np.asarray(yn, dtype=float)
    mask = in_region(xe, yn, region)
    for x, y, lbl, inside in zip(xe, yn, labels, mask):
        if not inside:
            continue
        ax.text(x + ox, y + oy, lbl,
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
