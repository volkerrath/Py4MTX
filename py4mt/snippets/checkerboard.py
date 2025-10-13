#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 19:05:48 2025

@author: vrath
"""

import numpy as np
import matplotlib.pyplot as plt

def create_checkerboard(nx, ny, nz, size):
    """Create a 3D checkerboard pattern."""
    checkerboard = np.zeros((nx, ny, nz))
    for x in range(0, nx, size):
        for y in range(0, ny, size):
            for z in range(0, nz, size):
                checkerboard[x:x+size, y:y+size, z:z+size] = 1  if (x//size + y//size + z//size) % 2 == 0 else -1
    return checkerboard
def plot_checkerboard(model, slice_index):
    """Plot a slice of the 3D checkerboard model."""
    plt.imshow(model[:, :, slice_index], cmap='gray', origin='lower')
    plt.colorbar()
    plt.title(f'Checkerboard Slice at z={slice_index}')
    plt.show()
    
def simulate_inversion(model, noise_level=0.1):
    """Simulate a tomographic inversion with noise."""
    noise = np.random.normal(0, noise_level, model.shape)
    return model + noise



nx, ny, nz = 20, 20, 20  # Grid dimensions
size = 2  # Size of each checkerboard square
checkerboard_model = create_checkerboard(nx, ny, nz,size)


# Plot a slice
plot_checkerboard(checkerboard_model, slice_index=10)




inverted_model = simulate_inversion(checkerboard_model)
# Plot a slice of the inverted model
plot_checkerboard(inverted_model, slice_index=10)
