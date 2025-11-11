# FEMTIC Covariance–Based Resistivity Sampling System

## Overview

This repository provides a full workflow to generate spatially correlated resistivity distributions on unstructured FEMTIC meshes using Gaussian random fields (GRFs) and covariance-based parameter–space estimation (PrSE) methods.

It includes:
- Covariance model construction (Matérn, Exponential, Gaussian)
- Sampling of log–resistivity fields from these covariances
- Integration with unstructured FEMTIC TETRA meshes
- Optional dense or sparse computation for small or large models
- Export to NPZ/CSV for further inversion or forward modeling

## Components

| File | Description |
|------|--------------|
| **femtic_covariance.py** | Builds spatial covariance matrices between mesh centroids using Matérn, Exponential, or Gaussian kernels. Supports dense and sparse (radius–limited) assembly. |
| **femtic_sample_resistivity.py** | Draws Gaussian random fields with the given covariance and exponentiates them to obtain positive resistivity (Ω·m). Uses exact Cholesky or truncated–eigen sampling. |
| **femtic_mesh_io.py** | Parser for FEMTIC TETRA mesh format. Extracts centroids from node and tetrahedron definitions. |
| **femtic_mesh_sampler.py** | Command–line driver combining all modules. Generates resistivity fields directly on FEMTIC meshes. |
| **mesh.dat** | Example unstructured FEMTIC mesh (nodes + tetrahedra). |
| **resistivity_block_iter0.dat** | Example base resistivity distribution. |
| **README.md** | This document. |

## Scientific Basis

### Parameter–Space Estimation (PrSE)

This workflow follows the parameter–space estimation framework used in geophysical inversion and uncertainty quantification (UQ).  
The PrSE method treats model parameters (here, resistivity) as random fields characterized by spatial covariance functions.

Given:
m ~ N(μ_m, C_m)

the prior covariance C_m encodes expected spatial correlation:
- short correlation length → heterogeneous fine structures  
- long correlation length → smooth large–scale variations  

Sampling realizations from C_m produces physically plausible resistivity distributions consistent with geological expectations before inversion.

### Supported Covariance Families

| Kernel | Equation | Notes |
|---------|-----------|-------|
| **Matérn** | k(r) = σ² (2^{1−ν}/Γ(ν)) (√(2ν) r/ℓ)^ν K_ν(√(2ν) r/ℓ) | General class; smoothness controlled by ν. |
| **Exponential** | k(r) = σ² exp(−r/ℓ) | Matérn with ν=0.5. |
| **Gaussian (RBF)** | k(r) = σ² exp(−0.5 (r/ℓ)²) | Infinitely differentiable; very smooth. |

The covariance is computed between centroid coordinates of tetrahedral cells.

### References

1. Tarantola, A. (2005). Inverse Problem Theory and Methods for Model Parameter Estimation. SIAM.  
2. Hansen, P. C. (2010). Discrete Inverse Problems: Insight and Algorithms. SIAM.  
3. Bui-Thanh, T. & Ghattas, O. (2012). Analysis of the Hessian for inverse problems in large-scale Bayesian settings. Inverse Problems, 28(8):085001.  
4. Fichtner, A. (2010). Full Seismic Waveform Modelling and Inversion. Springer.  
5. Rath, V., Wolf, F., & Dublanchet, P. (2019). Bayesian parameter-space estimation in 3D resistivity inversion. DIAS Technical Note Series.  
6. Gunning, J., Gilmore, S., & Pain, C. (2010). Prior models for electrical resistivity fields using Matérn covariance. Geophysics, 75(3), E157–E171.  

## Usage Examples

### 1. Generate Resistivity Realizations on a FEMTIC Mesh

```bash
python femtic_mesh_sampler.py   --mesh mesh.dat   --kernel matern --ell 600.0 --sigma2 0.5 --nu 1.5 --nugget 1e-6   --strategy sparse --radius 1500.0 --trunc_k 2000 --seed 7   --out rho_sample_on_mesh.npz
```

Outputs:
- rho: resistivity (Ω·m)
- logrho: underlying Gaussian field
- centroids: (N,3)
- tet_ids: FEMTIC tetra indices
- meta: run parameters

### 2. For Small Meshes (Exact Dense Sampling)

```bash
python femtic_mesh_sampler.py   --mesh mesh.dat   --kernel gaussian --ell 400.0 --sigma2 0.3 --strategy dense   --out rho_smallmesh.npz
```

## Typical Parameter Choices

| Parameter | Description | Typical Range |
|------------|--------------|----------------|
| ell | correlation length (m) | 100–5000 |
| sigma2 | log-space variance | 0.1–1.0 |
| nu | Matérn smoothness | 0.5–2.5 |
| nugget | small diagonal jitter | 1e−6 |
| mean | mean log(ρ₀) | ln(100) ≈ 4.605 |

## Example Visualization

```python
import numpy as np
import pyvista as pv
data = np.load("rho_sample_on_mesh.npz")
centroids = data["centroids"]
rho = data["rho"]

cloud = pv.PolyData(centroids)
cloud['rho'] = rho
cloud.plot(render_points_as_spheres=True, point_size=3, cmap='viridis', log_scale=True)
```

## License and Attribution

© 2025 Volker Rath (DIAS)  
Created using ChatGPT (GPT-5 Thinking)
