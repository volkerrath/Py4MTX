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

alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

L = - 1./np.log(alpha)

directory = "/home/vrath/Publish/Current/"
plt.plot(alpha, L, '-bo')
# plt.legend(["prior = 50 Ohmm", "prior = 20 Ohmm", "prior = 10 Ohmm"])
plt.xlabel("alpha")
plt.ylabel("correlation length L (index difference)")
plt.grid("on")

plt.savefig(directory + "L_from_alpha.pdf")
plt.savefig(directory + "L_from_alpha.png")