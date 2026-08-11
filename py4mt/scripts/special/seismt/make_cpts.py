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


def matplotlib_cmap_to_cpt(cmap, levels, filename, nan_color=(128, 128, 128)):
    """
    Export a discrete matplotlib colormap to a GMT/PyGMT-compatible CPT file.
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

        f.write("B 0 0 0\n")
        f.write("F 255 255 255\n")
        f.write(f"N {nan_color[0]} {nan_color[1]} {nan_color[2]}\n")


def make_discrete_cpt(cmin, cmax, nbins, mpl_cmap_name, output_cpt):
    """
    Create discrete matplotlib colormap, BoundaryNorm, and export CPT.
    """
    levels = np.linspace(cmin, cmax, nbins + 1)
    cmap = plt.get_cmap(mpl_cmap_name, nbins)
    norm = mcolors.BoundaryNorm(levels, cmap.N)

    matplotlib_cmap_to_cpt(cmap, levels, output_cpt)

    print(f"Saved: {output_cpt}")
    print(f"Range: {cmin} to {cmax}")
    print(f"Bins: {nbins}")
    print(f"Levels: {levels}")

    return cmap, norm, levels


# -----------------------------
# VP CPT
# -----------------------------
make_discrete_cpt(
    cmin=4495,
    cmax=6505,
    nbins=10,
    mpl_cmap_name="viridis_r",
    output_cpt="viridisr_vp.cpt"
)

# -----------------------------
# VS CPT
# -----------------------------
make_discrete_cpt(
    cmin=2395,
    cmax=4005,
    nbins=8,
    mpl_cmap_name="viridis_r",
    output_cpt="viridisr_vs.cpt"
)

# -----------------------------
# VP/VS CPT
# -----------------------------
make_discrete_cpt(
    cmin=1.69,
    cmax=2.11,
    nbins=8,
    mpl_cmap_name="hot_r",
    output_cpt="hotr_vps.cpt"
)