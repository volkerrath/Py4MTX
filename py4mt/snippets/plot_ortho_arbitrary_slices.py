#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 10:50:27 2025

@author: copilot
"""

"""
Plotting a cut (slice) through a rectangular 3-D finite-difference mesh.
Features:
- Define regular grid by origin, cell counts, and cell sizes
- Fill mesh with a scalar field (synthetic or user-supplied)
- Plot orthogonal slices: XY @ z-index, XZ @ y-index, YZ @ x-index
- Optionally plot an arbitrary plane slice (by interpolation)
Dependencies: numpy, matplotlib, scipy (for arbitrary plane)
"""
import numpy as np
import matplotlib.pyplot as plt

# Optional: for arbitrary plane sampling
from scipy.interpolate import RegularGridInterpolator

# ------------------------
# Mesh definition
# ------------------------
# Origin (lower-left-front) in model coordinates
x0, y0, z0 = 0.0, 0.0, 0.0

# Number of cells
nx, ny, nz = 60, 40, 30

# Cell sizes (uniform here; can be arrays for variable spacing)
dx, dy, dz = 100.0, 100.0, 50.0  # meters

# Cell-edge vectors and cell-center coordinates
x_edges = x0 + np.arange(nx + 1) * dx
y_edges = y0 + np.arange(ny + 1) * dy
z_edges = z0 + np.arange(nz + 1) * dz

# cell centers
xc = x0 + (np.arange(nx) + 0.5) * dx
yc = y0 + (np.arange(ny) + 0.5) * dy
zc = z0 + (np.arange(nz) + 0.5) * dz

# Create 3D meshgrid of centers for synthetic field
Xc, Yc, Zc = np.meshgrid(xc, yc, zc, indexing='ij')  # shape (nx, ny, nz)

# ------------------------
# Scalar field on mesh (example)
# Replace this with your FD model values (shape must be (nx,ny,nz))
# ------------------------
# Example: anisotropic gaussian + linear gradient
field = (1.0
         + 2.0 * np.exp(-(((Xc - xc.mean())**2)/(2*(dx*10)**2)
                          + ((Yc - yc.mean())**2)/(2*(dy*6)**2)
                          + ((Zc - zc.mean())**2)/(2*(dz*4)**2)))
         + 0.001 * Xc - 0.0005 * Yc + 0.002 * Zc)

# ------------------------
# Plot settings: choose slices
# ------------------------
# Indices (integer cell indices)
iz = nz // 2       # XY plane at this z-index
iy = ny // 3       # XZ plane at this y-index
ix = nx // 4       # YZ plane at this x-index

# If you'd rather choose by coordinate instead of index:
# find index by nearest center: iz = np.argmin(np.abs(zc - z_value))

# ------------------------
# Prepare 2D plots for orthogonal cuts
# ------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1) XY slice at z = zc[iz]
ax = axes[0]
# pcolormesh expects 2D grids of edges; use x_edges, y_edges
# swap axes because field has shape (nx,ny)
im1 = ax.pcolormesh(x_edges, y_edges, field[:, :, iz].T, shading='auto', cmap='viridis')
ax.set_title(f'XY slice @ z = {zc[iz]:.1f} (index {iz})')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
fig.colorbar(im1, ax=ax)

# 2) XZ slice at y = yc[iy]
ax = axes[1]
# create edges for x and z
# note: want X vs Z, so transpose appropriate slice
im2 = ax.pcolormesh(x_edges, z_edges, field[:, iy, :].T, shading='auto', cmap='viridis')
ax.set_title(f'XZ slice @ y = {yc[iy]:.1f} (index {iy})')
ax.set_xlabel('X (m)')
ax.set_ylabel('Z (m)')
fig.colorbar(im2, ax=ax)

# 3) YZ slice at x = xc[ix]
ax = axes[2]
im3 = ax.pcolormesh(y_edges, z_edges, field[ix, :, :].T, shading='auto', cmap='viridis')
ax.set_title(f'YZ slice @ x = {xc[ix]:.1f} (index {ix})')
ax.set_xlabel('Y (m)')
ax.set_ylabel('Z (m)')
fig.colorbar(im3, ax=ax)

plt.tight_layout()
plt.show()

# ------------------------
# Optional: arbitrary planar cut (example plane)
# ------------------------
# Define plane by a point p0 and normal n (here slice diagonal through domain)
p0 = np.array([xc.mean(), yc.mean(), zc.mean()])           # plane center point
# Normal vector for plane: choose something oblique
n = np.array([0.2, -0.3, 1.0])
n = n / np.linalg.norm(n)

# Build an orthonormal basis (u,v) spanning the plane
# choose u perpendicular to n
u = np.array([n[1], -n[0], 0.0])
if np.linalg.norm(u) < 1e-8:
    u = np.array([1.0, 0.0, 0.0])
u = u / np.linalg.norm(u)
v = np.cross(n, u)

# sampling grid in plane coordinates (meters): extent controls size of plane patch
extent_x = nx * dx * 0.8
extent_y = ny * dy * 0.8
nu, nv = 200, 200
su = np.linspace(-extent_x/2, extent_x/2, nu)
sv = np.linspace(-extent_y/2, extent_y/2, nv)
SU, SV = np.meshgrid(su, sv, indexing='xy')

# compute 3D sample points on plane P = p0 + SU*u + SV*v
P = p0.reshape((1,1,3)) + SU[...,None]*u.reshape((1,1,3)) + SV[...,None]*v.reshape((1,1,3))
Pts = P.reshape((-1,3))

# Interpolate field (RegularGridInterpolator expects axes in order (x,y,z))
interp = RegularGridInterpolator((xc, yc, zc), field, bounds_error=False, fill_value=np.nan)
Fvals = interp(Pts).reshape((nu, nv))

# Plot arbitrary plane slice
fig2, ax2 = plt.subplots(1,1,figsize=(6,6))
im4 = ax2.imshow(Fvals.T, origin='lower', extent=[su.min(), su.max(), sv.min(), sv.max()],
                 cmap='viridis', aspect='equal')
ax2.set_title('Arbitrary plane slice (local plane coords)')
ax2.set_xlabel('u (m)')
ax2.set_ylabel('v (m)')
fig2.colorbar(im4, ax=ax2)
plt.show()
