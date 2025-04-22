#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 21:22:12 2025

@author: vrath
"""

import os
import sys
import numpy as np

import matplotlib.pyplot as plt

alpha = np.arange(0.1,0.95,0.05 )

L = - 1./np.log(alpha)

directory = "/home/vrath/Publish/Current/"

plt.figure()

plt.plot(alpha, L, color="green", 
         marker="o", 
         linestyle="dashed",
         linewidth=1, 
         markersize=7,
         markeredgecolor="red",
         markerfacecolor="white"
         )


plt.text(0.3, 6.5, r"$L = -\frac{1}{\ln\alpha}$", fontsize = 24)

plt.xlabel(r"$\alpha$",fontsize=18)

plt.ylabel(r"$L$ (index difference)",fontsize=18)

plt.tick_params(labelsize='x-large')

plt.grid("on")

plt.tight_layout()

plt.savefig(directory + "L_from_alpha.pdf")
plt.savefig(directory + "L_from_alpha.png")