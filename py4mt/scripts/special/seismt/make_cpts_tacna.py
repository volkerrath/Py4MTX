#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 15:16:31 2026

@author: sbyrd
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def matplotlib_cmap_to_cpt(cmap, levels, filename, nan_color=(128, 128, 128),
                            bg_transparent=True, fg_transparent=True,
                            nan_transparent=True):
    """
    Export a discrete matplotlib colormap to a GMT/PyGMT-compatible CPT file.

    Transparency for the background/foreground/NaN swatches
    ----------------------------------------------------------------
    Plain RGB triplets in a .cpt file are always fully opaque — the
    format has no per-swatch alpha channel. To get a genuinely
    transparent swatch (rather than a solid colour that merely looks
    transparent in one particular viewer) GMT reserves the special
    colour "-" for B/F/N, meaning "paint nothing here". That's what
    ``bg_transparent``/``fg_transparent``/``nan_transparent`` write when
    True (the default) — e.g. for vps < cmin, > cmax, or NaN.

    Passing False for any of these instead writes an explicit opaque
    colour for that swatch (black for B, white for F, ``nan_color`` for
    N) — only do this if you actually want a solid fill there.
    """
    ncolors = len(levels) - 1

    with open(filename, "w") as f:
        f.write("# COLOR_MODEL = RGB\n")

        for i in range(ncolors):
            z1 = levels[i]
            z2 = levels[i + 1]

            r, g, b, _ = cmap(i)
            r = int(round(r * 255))
            g = int(round(g * 255))
            b = int(round(b * 255))

            f.write(f"{z1:.6g} {r} {g} {b} {z2:.6g} {r} {g} {b}\n")

        f.write("B -\n" if bg_transparent else "B 0 0 0\n")
        f.write("F -\n" if fg_transparent else "F 255 255 255\n")
        f.write("N -\n" if nan_transparent
                 else f"N {nan_color[0]} {nan_color[1]} {nan_color[2]}\n")


def make_discrete_cpt(cmin, cmax, nbins, mpl_cmap_name, output_cpt,
                       bg_transparent=True, fg_transparent=True,
                       nan_transparent=True):
    """
    Create discrete matplotlib colormap, BoundaryNorm, and export CPT.
    """
    levels = np.linspace(cmin, cmax, nbins + 1)
    cmap = plt.get_cmap(mpl_cmap_name, nbins)
    norm = mcolors.BoundaryNorm(levels, cmap.N)

    matplotlib_cmap_to_cpt(cmap, levels, output_cpt,
                            bg_transparent=bg_transparent,
                            fg_transparent=fg_transparent,
                            nan_transparent=nan_transparent)

    print(f"Saved: {output_cpt}")
    print(f"Range: {cmin} to {cmax}")
    print(f"Bins: {nbins}")
    print(f"Levels: {levels}")

    return cmap, norm, levels

# -----------------------------
# rho CPT
# -----------------------------
make_discrete_cpt(
    cmin=1490,
    cmax=3310,
    nbins=10,
    mpl_cmap_name="jet",
    output_cpt="../colormaps/jet_rho.cpt"
)


# -----------------------------
# VP CPT
# -----------------------------
make_discrete_cpt(
    cmin=4495,
    cmax=6505,
    nbins=10,
    mpl_cmap_name="viridis_r",
    output_cpt="../colormaps/viridisr_vp.cpt"
)

# -----------------------------
# VS CPT
# -----------------------------
make_discrete_cpt(
    cmin=2395,
    cmax=4005,
    nbins=8,
    mpl_cmap_name="viridis_r",
    output_cpt="../colormaps/viridisr_vs.cpt"
)

# -----------------------------
# VP/VS CPT
# -----------------------------
make_discrete_cpt(
    cmin=1.68,
    cmax=2.11,
    nbins=8,
    mpl_cmap_name="hot_r",
    output_cpt="../colormaps/hotr_vps.cpt"
)