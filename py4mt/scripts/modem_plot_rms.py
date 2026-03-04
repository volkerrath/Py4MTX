#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot RMS convergence curves from ModEM NLCG log files.

Extracts nRMS values from .log files and creates convergence plots.

@author: sbyrd
Cleanup: 4 Mar 2026 by Claude (Anthropic)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
#  Configuration
# =============================================================================
DIRECTORY = "/home/sbyrd/Desktop/PEROU/DAHU/SABA200_LF/"

FILENAME_IN = [
    DIRECTORY + "SABA200_50_Alpha03_ZT_NLCG.log",
    DIRECTORY + "SABA200_20_Alpha03_ZT_NLCG.log",
    DIRECTORY + "SABA200_10_Alpha03_ZT_NLCG.log",
]
LEGEND_LABELS = ["prior = 50 Ohmm", "prior = 20 Ohmm", "prior = 10 Ohmm"]

PLOT_FILE = DIRECTORY + "rms.pdf"

# =============================================================================
#  Extract RMS and plot
# =============================================================================
fig, ax = plt.subplots()

for idx, filename in enumerate(FILENAME_IN):
    filename_out = filename.replace(".log", ".csv")
    rms = []

    with open(filename, "r") as file:
        for line in file:
            if (("START:" in line) or ("STARTLS:" in line)) and ("rms=" in line):
                parts = line.split()
                rms.append(float(parts[6]))

    with open(filename_out, "w") as file:
        for i, val in enumerate(rms):
            file.write(f"{i},{val}\n")

    plt.plot(np.arange(len(rms)), rms, label=LEGEND_LABELS[idx])

plt.legend()
plt.xlabel("iteration")
plt.ylabel("nRMS")
plt.grid("on")
plt.tight_layout()
plt.savefig(PLOT_FILE)
print(f"Saved {PLOT_FILE}")
