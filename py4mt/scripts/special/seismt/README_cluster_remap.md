Yes—but I would avoid constructing the geometric union of both meshes as the final statistical mesh. With a highly variable unstructured mesh, that “supermesh” can contain huge numbers of tiny intersection cells and can make clustering depend on discretization rather than geology.

## Recommended approach

Use a separate **common analysis mesh** and map both models onto it by volume overlap.

Let (S_j) be a source cell, (T_k) a target cell, and

[
B_{kj}=\left|T_k\cap S_j\right|
]

the intersection volume. For a cellwise-constant quantity (m_j), calculate

[
m_k =
\frac{\sum_j B_{kj}m_j}
{\sum_j B_{kj}} .
]

The sparse matrix (B) is computed once for each source mesh and reused for every realization or variable.

### 1. Choose the common analysis mesh deliberately

I suggest:

* Restrict it to the volume covered reliably by **both** models.
* Use cells no smaller than the coarse regular mesh in regions where that mesh controls resolution.
* Allow moderate octree refinement where both models have genuinely higher resolution.
* Preserve important horizontal or geological boundaries where possible.
* Treat air, ocean, fixed blocks and padding cells through masks—not as ordinary samples.
* Record the coverage fraction

[
c_k=\frac{\sum_j B_{kj}}{|T_k|}
]

and discard cells with, for example, (c_k<0.95).

The common grid therefore represents the **common information scale**, not the finest available discretization.

## Which quantity should be averaged?

For MT or electrical models, this decision matters.

| Quantity                                        | Recommended mapping                                                               |
| ----------------------------------------------- | --------------------------------------------------------------------------------- |
| Scalar resistivity spanning orders of magnitude | Volume-average (\log_{10}\rho)                                                    |
| Scalar conductivity                             | Volume-average (\sigma), if conductivity is the physical extensive-response proxy |
| Lithological/category label                     | Largest overlap fraction or complete class-fraction vector                        |
| Probability                                     | Volume-weighted probability                                                       |
| Anisotropic conductivity tensor                 | Log-Euclidean or another SPD-preserving tensor average                            |
| Resolution/sensitivity                          | Transfer separately; do not treat it as a model property                          |

For general statistical comparison, mapping (\log_{10}\rho) is usually safer than averaging (\rho): it limits domination by a few very resistive cells and approximately preserves the multiplicative character of resistivity contrasts.

However, neither arithmetic resistivity nor arithmetic conductivity is universally the physically correct upscaling rule. Layer-parallel and layer-normal effective conductivities, for example, lead to arithmetic and harmonic relationships respectively. If the goal is simulation-equivalent upscaling rather than statistical comparison, numerical homogenization is preferable.

## Clustering without mesh bias

A very fine region must not acquire more influence merely because it contains more cells.

For target-cell features (\mathbf z_k), use the cell volume (V_k) in the clustering objective:

[
J=\sum_k V_k,\lVert\mathbf z_k-\boldsymbol\mu_{g(k)}\rVert^2 .
]

For comparing complete models (a) and (b), use a volume-weighted distance:

[
d^2(a,b)=
\frac{\sum_k V_k q_k
\left[m_k^{(a)}-m_k^{(b)}\right]^2}
{\sum_k V_k q_k},
]

where (q_k) can be a sensitivity or resolution weight.

Two useful cases are:

* **Clustering complete models or ensemble realizations:** flatten the common-grid values, use the weighted distance above, optionally perform volume-weighted PCA first, and then use hierarchical clustering or k-medoids.
* **Clustering subsurface regions:** each common cell is an observation and its properties are features. Use volume-weighted k-means/GMM, or spatially constrained clustering if clusters must be connected.

For geological zonation, ordinary k-means often produces spatially fragmented “salt-and-pepper” classes. Better choices include:

* a contiguity-constrained method such as SKATER;
* graph clustering with face-sharing target cells as edges;
* a Potts/Markov spatial penalty;
* post-clustering removal of components smaller than a stated physical volume.

Do not simply add (x,y,z) coordinates as ordinary standardized features unless you explicitly want Euclidean spatial compactness; the result depends strongly on arbitrary coordinate scaling.

## What I would implement

1. Transform both meshes into exactly the same CRS, units, vertical datum and (z)-direction.
2. Define the common analysis domain and masks.
3. Construct a coarse structured or octree target mesh.
4. Compute sparse overlap matrices (B^{(1)}) and (B^{(2)}).
5. Map (\log_{10}\rho), coverage, sensitivity and material fractions separately.
6. Exclude cells poorly resolved in either input model.
7. Standardize features robustly, preferably using weighted medians/MADs.
8. Perform volume-weighted PCA or clustering.
9. Repeat on a second, somewhat coarser analysis mesh.
10. Compare clusterings using adjusted Rand index or variation of information.

The last step is important: if the scientific conclusions change substantially with a modest change in the common grid, the clusters are not mesh-independent.

## Practical implementation options

* **Exact intersection:** tetrahedralize general polyhedral cells and calculate cell–cell intersection volumes. This gives the strongest conservation but can be expensive.
* **Supermesh/Galerkin projection:** mathematically clean for finite-element fields. [`libsupermesh`](https://github.com/firedrakeproject/libsupermesh) implements parallel supermeshing.
* **Deterministic sampling:** place Sobol or stratified points inside every target cell, locate their source cells, and estimate the overlap fractions. This is often much easier for large 3-D geophysical meshes. Store the estimated sampling error and use the same points for all models.
* **VTK/PyVista sampling:** useful for quick inspection, but `sample()` interpolates values at target points and is not a conservative cell-volume transfer. [PyVista documents this distinction](https://docs.pyvista.org/api/core/_autosummary/pyvista.dataobjectfilters.sample).

For your type of FEMTIC models, I would probably use a coarse common octree plus exact box intersections where possible and deterministic Sobol integration for irregular cells. This avoids generating an enormous explicit supermesh.

## Essential checks

Before doing statistics, verify:

* constant-field preservation;
* mapped domain volume;
* preservation of the volume integral of the mapped variable;
* absence of extrapolation outside the source domain;
* no leakage between air, ocean and Earth cells;
* mapped minimum/maximum—first-order overlap averaging should not overshoot;
* convergence when the number of integration samples is increased;
* stability of clusters under changes of target resolution;
* comparison with a blocky synthetic model containing known sharp interfaces.

Also use block bootstrap or spatial cross-validation with blocks at least as large as the coarse source cells. Treating all common-grid voxels as statistically independent would create severe pseudoreplication.

## Relevant literature

* P. E. Farrell, M. D. Piggott, C. C. Pain, G. J. Gorman and C. R. Wilson (2009), “Conservative interpolation between unstructured meshes via supermesh construction,” *Computer Methods in Applied Mechanics and Engineering*, 198, 2632–2642. [https://doi.org/10.1016/j.cma.2009.03.004](https://doi.org/10.1016/j.cma.2009.03.004)

* P. E. Farrell and J. R. Maddison (2011), “Conservative interpolation between volume meshes by local Galerkin projection,” *Computer Methods in Applied Mechanics and Engineering*, 200, 89–100. [https://doi.org/10.1016/j.cma.2010.07.015](https://doi.org/10.1016/j.cma.2010.07.015)

* Carol A. Gotway and Linda J. Young (2002), “Combining incompatible spatial data,” *Journal of the American Statistical Association*, 97, 632–648. This is particularly relevant to change of support and spatial misalignment. [https://doi.org/10.1198/016214502760047140](https://doi.org/10.1198/016214502760047140)

* Philip W. Jones (1999), “First- and second-order conservative remapping schemes for grids in spherical coordinates,” *Monthly Weather Review*, 127, 2204–2210. Although written for spherical grids, it is a foundational conservative-remapping reference. [https://doi.org/10.1175/1520-0493(1999)127%3C2204:FASOCR%3E2.0.CO;2](https://doi.org/10.1175/1520-0493%281999%29127%3C2204:FASOCR%3E2.0.CO;2)

* R. M. Assunção, M. C. Neves, G. Câmara and C. da Costa Freitas (2006), “Efficient regionalization techniques for socio-economic geographical units using minimum spanning trees,” *International Journal of Geographical Information Science*, 20, 797–811. This introduces the spatially constrained SKATER approach. [https://doi.org/10.1080/13658810600665111](https://doi.org/10.1080/13658810600665111)

In short: **use the geometric intersection only to calculate conservative weights; use a resolution-matched common mesh for the statistics; and include volume weights in PCA and clustering.**



BibTeX bibliography

Download

The file contains 18 verified references, with complete author lists, covering conservative mesh transfer, common refinement, tetrahedral and polyhedral meshes, change of spatial support, model comparison, spatial clustering, and relevant Python software.
Python packages

1. MEDCoupling — probably the best match

medcoupling, developed within the SALOME ecosystem, supports:

    Structured and unstructured three-dimensional meshes.

    Tetrahedra, hexahedra, and mixed cell types.

    Conservative cell-to-cell remapping based on actual intersection volumes.

    Direct access to the sparse overlap matrix.

    NumPy and SciPy integration.

    Cell-volume calculations.

Install it inside your existing Python environment:

python -m pip install medcoupling

Precompiled Linux wheels are available for Python 3.11, which should suit your current environment. PyPI package and available wheels

The particularly useful feature is:

import numpy as np
import medcoupling as mc

from MEDCouplingRemapper import MEDCouplingRemapper

# source_mesh: unstructured FEMTIC mesh.
# target_mesh: common regular or octree analysis mesh.
# Both are MEDCoupling mesh objects.

remapper = MEDCouplingRemapper()
remapper.prepare(source_mesh, target_mesh, "P0P0")

# Sparse matrix of intersection volumes:
#
# overlap[k, j] = volume(target_cell_k ∩ source_cell_j)

overlap = remapper.getCrudeCSRMatrix()

# Volume covered by source cells inside each target cell.

covered_volume = np.asarray(overlap.sum(axis=1)).ravel()

# Actual target-cell volumes.

target_volume = (
    target_mesh
    .getMeasureField(True)
    .getArray()
    .toNumPyArray()
    .ravel()
)

coverage = np.divide(
    covered_volume,
    target_volume,
    out=np.zeros_like(target_volume),
    where=target_volume > 0.0,
)

# Map cellwise log-resistivity by overlap-weighted averaging.

source_logrho = np.log10(source_resistivity)

target_logrho = np.divide(
    overlap @ source_logrho,
    covered_volume,
    out=np.full(target_volume.shape, np.nan),
    where=covered_volume > 0.0,
)

# Reject incompletely covered target cells.

target_logrho[coverage < 0.95] = np.nan

"P0P0" means piecewise-constant source cells to piecewise-constant target cells. The official documentation confirms that the underlying matrix contains cell-intersection measures, and its NumPy/SciPy example explicitly demonstrates getCrudeCSRMatrix(). MEDCoupling remapping documentation, MEDCoupling NumPy/SciPy example

For multiple realizations on the same source mesh, compute overlap once:

target_models = overlap @ source_models
target_models /= covered_volume[:, None]

Here source_models has shape:

(number_source_cells, number_models)

This is especially attractive for FEMTIC RTO or MCMC ensembles.

One subtlety: MEDCoupling also provides transferField() with different field natures:

source_field.setNature(mc.IntensiveMaximum)

or:

source_field.setNature(mc.IntensiveConservation)

For partially covered cells, these use different normalizations:
IntensiveMaximum:mk​=∑j​Vkj​∑j​Vkj​mj​​, IntensiveConservation:mk​=Vk​∑j​Vkj​mj​​.

For statistical analysis of log10​ρ, I would generally use the first definition and independently reject cells with insufficient coverage. MEDCoupling field-nature documentation

2. ESMPy — strong alternative

esmpy supports:

    Three-dimensional structured grids.

    Three-dimensional tetrahedral and hexahedral unstructured meshes.

    First-order conservative remapping.

    Reusable sparse remapping weights.

    Parallel execution.

Installation:

conda install -c conda-forge esmpy

The project recommends installing through conda-forge because the Python interface depends on the underlying compiled ESMF library. ESMPy installation instructions

For general FEMTIC-style workflows, I would try MEDCoupling first because its explicit access to the intersection-volume matrix fits your statistical application particularly well.

3. PyVista/VTK — useful, but not conservative by itself

sampled = target_mesh.sample(source_mesh)

This samples source values at target points. It does not compute target-cell averages from intersection volumes. It is useful for:

    visualization;

    quick comparisons;

    generating inspection slices;

    implementing approximate stratified sampling.

It is not a replacement for conservative remapping when cell sizes vary strongly. PyVista sampling documentation

4. Clustering after remapping

For volume-weighted clustering:

from sklearn.cluster import KMeans

features = np.column_stack(
    [
        target_logrho_model_1,
        target_logrho_model_2,
    ]
)

valid = np.all(np.isfinite(features), axis=1)

clustering = KMeans(
    n_clusters=5,
    n_init=20,
    random_state=42,
)

labels = clustering.fit_predict(
    features[valid],
    sample_weight=target_volume[valid],
)

sample_weight=target_volume prevents regions containing many small cells from dominating the clustering. scikit-learn clustering documentation

For spatially connected geological clusters, investigate spopt:

python -m pip install spopt libpysal

It implements SKATER and spatially constrained regionalization. For a three-dimensional mesh, you would need to construct the face-neighbor connectivity graph explicitly; most published examples use two-dimensional geographical regions. SKATER documentation

My recommendation: use medcoupling for the common-mesh overlap matrix, scipy.sparse for ensemble transfers, and scikit-learn or spopt for the subsequent volume-aware clustering.

Yes. Gradients can substantially improve clustering, particularly when the aim is to distinguish geological domains, identify interfaces, or assess whether independently inverted models share structural boundaries.

However, there are two importantly different ways to use them:

1. Include gradients as additional clustering features.
2. Use gradients to control where cluster boundaries are allowed.

For subsurface interpretation, the second approach is often preferable.

### 1. Gradients as clustering features

For a resistivity model, define

[
m(\mathbf{x})=\log_{10}\rho(\mathbf{x}).
]

Then construct a feature vector such as

[
\mathbf{f}(\mathbf{x})
======================

\left[
m,;
L_x\frac{\partial m}{\partial x},;
L_y\frac{\partial m}{\partial y},;
L_z\frac{\partial m}{\partial z}
\right],
]

where (L_x,L_y,L_z) are characteristic lengths.

The length factors make the derivative terms dimensionally comparable to (m):

[
\left[\frac{\partial \log_{10}\rho}{\partial x}\right]
======================================================

\mathrm{m}^{-1},
]

so

[
L_x\frac{\partial \log_{10}\rho}{\partial x}
]

is dimensionless.

For example, with (L_x=1000,\mathrm{m}), the scaled derivative represents the change in (\log_{10}\rho) over approximately one kilometre.

Possible feature sets include:

[
\mathbf f_1 =
\left[
m,;
L|\nabla m|
\right],
]

[
\mathbf f_2 =
\left[
m,;
L_xm_x,;
L_ym_y,;
L_zm_z
\right],
]

or

[
\mathbf f_3 =
\left[
m,;
L_h\sqrt{m_x^2+m_y^2},;
L_vm_z
\right].
]

The third formulation is often useful for volcanic or crustal models because horizontal and vertical structural changes can have different geological meanings.

For example:

* High conductivity and weak gradients: interior of a conductive domain.
* High conductivity and strong gradients: edge of a conductive body.
* High resistivity and weak gradients: resistive host rock.
* Strong vertical gradient: possible layering or cap boundary.
* Strong horizontal gradient: possible fault, conduit margin, or lateral lithological contact.

The potential drawback is that the algorithm may produce a separate “boundary” cluster. That can be desirable for detecting interfaces, but less desirable if the intended clusters are geological units.

### 2. Gradients as boundary indicators

Instead of clustering the gradient values themselves, use them to discourage clusters from crossing sharp interfaces.

For neighboring cells (i) and (j), define a graph affinity:

[
w_{ij}
======

\exp
\left[
-\frac{(m_i-m_j)^2}{2s_m^2}
-\frac{L^2g_{ij}^2}{2s_g^2}
\right],
]

where (g_{ij}) measures the gradient around their shared face.

Then:

* Similar neighboring cells separated by weak gradients remain strongly connected.
* Cells separated by sharp gradients become weakly connected.
* Clusters tend to terminate at physical interfaces.

An alternative formulation is a spatially regularized clustering objective:

[
J
=

\sum_i V_i
\left|
\mathbf f_i-\boldsymbol\mu_{c_i}
\right|^2
+
\lambda
\sum_{(i,j)\in E}
A_{ij}L_{\mathrm{ref}}
\exp
\left[
-\left(\frac{g_{ij}}{g_0}\right)^2
\right]
\mathbf{1}_{c_i\neq c_j}.
]

Here:

* (V_i) is cell volume.
* (A_{ij}) is shared-face area.
* (c_i) is the cluster label.
* (g_0) is a characteristic gradient.
* (L_{\mathrm{ref}}) makes the face penalty comparable to a volume-weighted objective.

At strong gradients, the exponential becomes small, so placing a cluster boundary there is relatively inexpensive.

This formulation is particularly attractive for highly inhomogeneous meshes because both cell volumes and interface areas enter explicitly.

### 3. For two fields: cross-gradients and gradient alignment

Suppose you compare

[
m_1=\log_{10}\rho_1
]

and

[
m_2=\log_{10}\rho_2,
]

or resistivity against seismic velocity.

The cross-gradient is

[
\mathbf t
=========

\nabla m_1
\times
\nabla m_2.
]

Its magnitude is

[
|\mathbf t|
===========

|\nabla m_1|
|\nabla m_2|
\sin\theta.
]

Therefore:

* ( |\mathbf t|\approx 0): the two fields have parallel or antiparallel gradients, or one gradient is negligible.
* Large ( |\mathbf t|): the two fields exhibit differently oriented structures.

For clustering, a normalized form is usually easier to interpret:

[
c_{\times}
==========

\frac{
|\nabla m_1\times\nabla m_2|
}{
|\nabla m_1|
|\nabla m_2|
+
\epsilon
}.
]

Approximately,

[
c_{\times}\approx |\sin\theta|.
]

Alternatively, define structural alignment:

[
a
=

\frac{
|\nabla m_1\cdot\nabla m_2|
}{
|\nabla m_1|
|\nabla m_2|
+
\epsilon
}.
]

Approximately,

[
a\approx |\cos\theta|.
]

A possible feature vector becomes

[
\mathbf f
=========

\left[
m_1,;
m_2,;
L|\nabla m_1|,;
L|\nabla m_2|,;
a
\right].
]

For example, this can distinguish:

* Structures supported by both models.
* Structures present only in the fine-mesh model.
* Structures present only in the coarse model.
* Contradictory interface orientations.

Important caveat: alignment is meaningless where either gradient is close to zero. Mask those cells or downweight the alignment feature there.

The cross-gradient concept is well established in joint geophysical inversion. [Gallardo and Meju, 2004](https://doi.org/10.1029/2003JB002716)

### 4. Compute gradients after transferring to the common mesh

For your specific situation, I recommend:

[
\text{source meshes}
\longrightarrow
\text{conservative common-mesh transfer}
\longrightarrow
\text{common-scale smoothing}
\longrightarrow
\text{gradient calculation}
\longrightarrow
\text{clustering}.
]

Generally,

[
\nabla(Pm)\neq P(\nabla m),
]

where (P) is the remapping operator.

For comparing models defined on very different meshes, computing gradients separately on their original meshes creates a serious resolution mismatch:

* The fine mesh can exhibit large gradients simply because interfaces are resolved over shorter distances.
* The coarse mesh necessarily spreads the same interface over a broader region.
* Clustering then distinguishes mesh resolution rather than geology.

Therefore, transfer both fields first and differentiate them at the same physical smoothing scale.

For piecewise-constant FEMTIC cell properties this is especially relevant: unsmoothed derivatives at cell interfaces are not classical pointwise derivatives.

### 5. Practical Python implementation

Assume `logrho` has array shape

```python
(nz, ny, nx)
```

and grid spacing is expressed in metres.

```python
import numpy as np

from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans


# Common-grid model.

logrho = np.asarray(logrho, dtype=np.float64)

valid = np.isfinite(logrho)

# Grid spacing in metres.

dx = 250.0
dy = 250.0
dz = 100.0

# Physical smoothing scales.

smooth_x = 750.0
smooth_y = 750.0
smooth_z = 300.0

sigma_pixels = (
    smooth_z / dz,
    smooth_y / dy,
    smooth_x / dx,
)

# Mask-aware Gaussian smoothing.

filled = np.where(valid, logrho, 0.0)
weights = valid.astype(np.float64)

numerator = gaussian_filter(
    filled,
    sigma=sigma_pixels,
    mode="nearest",
)

denominator = gaussian_filter(
    weights,
    sigma=sigma_pixels,
    mode="nearest",
)

logrho_smooth = np.divide(
    numerator,
    denominator,
    out=np.full_like(logrho, np.nan),
    where=denominator > 0.95,
)

# Array order is z, y, x.

grad_z, grad_y, grad_x = np.gradient(
    logrho_smooth,
    dz,
    dy,
    dx,
    edge_order=2,
)

grad_horizontal = np.hypot(
    grad_x,
    grad_y,
)

grad_magnitude = np.sqrt(
    grad_x**2
    + grad_y**2
    + grad_z**2
)

# Reference lengths convert gradients into
# approximately comparable dimensionless quantities.

length_horizontal = 1000.0
length_vertical = 500.0

# Explicit gradient weights.

weight_horizontal = 0.5
weight_vertical = 0.5

# Use the same reference scale for model values
# and length-scaled derivatives.

model_median = np.nanmedian(logrho_smooth)

model_scale = np.nanmedian(
    np.abs(logrho_smooth - model_median)
)

model_scale = max(model_scale, 1.0e-12)

features = np.column_stack(
    [
        (
            (logrho_smooth - model_median)
            / model_scale
        ).ravel(),
        (
            weight_horizontal
            * length_horizontal
            * grad_horizontal
            / model_scale
        ).ravel(),
        (
            weight_vertical
            * length_vertical
            * grad_z
            / model_scale
        ).ravel(),
    ]
)

valid_features = np.all(
    np.isfinite(features),
    axis=1,
)

cell_volumes = np.asarray(
    cell_volumes,
    dtype=np.float64,
).ravel()

model = KMeans(
    n_clusters=5,
    n_init=30,
    random_state=42,
)

labels_valid = model.fit_predict(
    features[valid_features],
    sample_weight=cell_volumes[valid_features],
)

labels = np.full(
    features.shape[0],
    -1,
    dtype=np.int32,
)

labels[valid_features] = labels_valid

labels = labels.reshape(logrho.shape)
```

A subtle but important point: if each feature column is independently standardized **after** applying `weight_horizontal` and `weight_vertical`, that standardization largely cancels the chosen weights. Apply the relative weights after normalization, or normalize all gradient terms using a shared physically meaningful scale, as above.

SciPy supports Gaussian derivatives directly, while scikit-learn supports sample-weighted k-means. [SciPy Gaussian filtering](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html), [scikit-learn clustering](https://scikit-learn.org/stable/modules/clustering.html)

### 6. Watershed segmentation: often the most natural gradient-based method

If you principally want geological regions separated by strong contrasts, consider watershed segmentation:

```python
from skimage.segmentation import watershed

gradient_image = np.sqrt(
    grad_x**2
    + grad_y**2
    + grad_z**2
)

segmentation = watershed(
    gradient_image,
    markers=markers,
    mask=valid,
)
```

Here:

* `markers` identify initial domains.
* Large gradients form barriers.
* Regions expand until they meet along strong boundaries.

A good combination is:

1. Use k-means or fuzzy c-means on resistivity values to identify candidate domain interiors.
2. Select low-gradient cells as reliable seeds.
3. Use the gradient magnitude as the watershed landscape.
4. Merge very small regions using a minimum physical volume.

`scikit-image` explicitly documents gradient-based watershed segmentation, including multidimensional images. [scikit-image watershed documentation](https://scikit-image.org/docs/stable/user_guide/tutorial_segmentation.html)

### 7. For ensembles: probability of an interface

For your RTO or MCMC ensembles, gradients can also be used probabilistically.

For realization (r),

[
g^{(r)}(\mathbf x)
==================

\left|
\nabla \log_{10}
\rho^{(r)}(\mathbf x)
\right|.
]

Define an interface probability:

[
p_{\mathrm{edge}}(\mathbf x)
============================

\frac{1}{N}
\sum_{r=1}^{N}
\mathbf 1
\left[
g^{(r)}(\mathbf x)>g_0
\right].
]

Additional useful quantities are

[
\overline{g}(\mathbf x)
=======================

\frac{1}{N}
\sum_r g^{(r)}(\mathbf x)
]

and

[
s_g(\mathbf x)
==============

\operatorname{std}_r
\left[
g^{(r)}(\mathbf x)
\right].
]

Then cluster using, for example,

[
\mathbf f(\mathbf x)
====================

\left[
\overline m,;
s_m,;
L\overline g,;
Ls_g,;
p_{\mathrm{edge}}
\right].
]

Interpretation:

* High (\overline g), high (p_{\mathrm{edge}}), low (s_g): robust interface.
* High (\overline g), low (p_{\mathrm{edge}}): interface appears in only a minority of models.
* High (s_g): uncertain position or strength of an interface.
* Low (s_m), low (\overline g): stable homogeneous region.

For volcanic MT ensembles, this could help distinguish a persistent conductor boundary from an artifact whose location varies substantially among realizations.

### Relevant geophysical references

* Hendrik Paasche and Jens Tronicke, 2007, “Cooperative inversion of 2D geophysical data sets: A zonal approach based on fuzzy C-means cluster analysis,” *Geophysics*, 72, A35–A39. [https://doi.org/10.1190/1.2670341](https://doi.org/10.1190/1.2670341)

* Luis A. Gallardo and Max A. Meju, 2004, “Joint two-dimensional DC resistivity and seismic travel time inversion with cross-gradients constraints,” *Journal of Geophysical Research: Solid Earth*, 109, B03311. [https://doi.org/10.1029/2003JB002716](https://doi.org/10.1029/2003JB002716)

* Jiajia Sun and Yaoguo Li, 2016, “Joint inversion of multiple geophysical data using guided fuzzy c-means clustering,” *Geophysics*, 81, ID37–ID57. [https://doi.org/10.1190/geo2015-0457.1](https://doi.org/10.1190/geo2015-0457.1)

* Jiajia Sun and Yaoguo Li, 2017, “Joint inversion of multiple geophysical and petrophysical data using generalized fuzzy clustering algorithms,” *Geophysical Journal International*, 208, 1201–1216. [https://doi.org/10.1093/gji/ggw442](https://doi.org/10.1093/gji/ggw442)

**Practical recommendation:** For your application, start with common-mesh (\log_{10}\rho), horizontal and vertical gradient magnitudes, cell-volume weighting, and gradient-based watershed boundaries. For ensembles, add interface probability and gradient uncertainty.

