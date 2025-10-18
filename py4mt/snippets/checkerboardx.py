import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define grid size
grid_size = 10
checkerboard = np.zeros((grid_size, grid_size, grid_size))

# Create checkerboard pattern
for i in range(grid_size):
    for j in range(grid_size):
        for k in range(grid_size):
            checkerboard[i, j, k] = (i + j + k) % 2

# Visualize the checkerboard
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.voxels(checkerboard, facecolors='cyan', edgecolors='k')
plt.show()
