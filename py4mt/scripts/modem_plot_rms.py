#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot RMS convergence curves from ModEM NLCG log files.

Extracts nRMS values from .log files and creates convergence plots.

@author: sbyrd
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
#  Configuration
# =============================================================================
directory = "/home/sbyrd/Desktop/PEROU/DAHU/SABA200_LF/"

filename_in = [
    directory + "SABA200_50_Alpha03_ZT_NLCG.log",
    directory + "SABA200_20_Alpha03_ZT_NLCG.log",
    directory + "SABA200_10_Alpha03_ZT_NLCG.log",
]
legend_labels = ["prior = 50 Ohmm", "prior = 20 Ohmm", "prior = 10 Ohmm"]

# =============================================================================
#  Extract RMS and plot
# =============================================================================
fig, ax = plt.subplots()

for idx, filename in enumerate(filename_in):
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

    plt.plot(np.arange(len(rms)), rms, label=legend_labels[idx])

plt.legend()
plt.xlabel("iteration")
plt.ylabel("nRMS")
plt.grid("on")
plt.tight_layout()
plt.savefig(directory + "rms.pdf")
print(f"Saved {directory}rms.pdf")
