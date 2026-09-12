"""
Author: Claude (Anthropic)
Generated: 2026-09-11
NOTE: AI-generated code. Review before production use.

Sketch for visualizing a FEMTIC v5 full-tensor anisotropic resistivity
model on an UNSTRUCTURED (tetrahedral) mesh with PyVista:
  - build a pv.UnstructuredGrid from node coordinates + cell connectivity
  - clip-plane slice through an isotropic-equivalent scalar cell field
    (geometric mean of eigenvalues) for orientation
  - tensor glyphs (ellipsoids) at a subsampled set of cell centroids,
    built by eigendecomposing the full symmetric 3x3 conductivity
    tensor per cell (FEMTIC v5 ResistivityBlockAnisotropic): eigenvalues
    give principal resistivities, eigenvectors give orientation directly.

Eigendecomposition is vectorized over all cells at once via
np.linalg.eigh's native batching (shape (N,3,3) in, (N,3) / (N,3,3) out)
-- no per-cell Python loop.

Adjust the HDF5 read block to match the actual dataset layout written
by your ResistivityBlockAnisotropic HDF5 extension.
"""

import numpy as np
import pyvista as pv
import h5py


def load_aniso_model(h5_path):
    """
    Load full-tensor anisotropic resistivity model from FEMTIC v5 HDF5,
    unstructured tetrahedral mesh.

    Expected datasets (adjust names/shapes to match your actual schema):
      /mesh/nodes         (n_nodes, 3)      -- x, y, z coordinates
      /mesh/connectivity   (n_cells, 4)      -- node indices per tetrahedron
      /model/conductivity_tensor  (n_cells, 3, 3)  -- symmetric, S/m
        or 6 independent components (xx, yy, zz, xy, xz, yz) per cell --
        see assemble_tensor_from_components below.
    """
    with h5py.File(h5_path, "r") as f:
        nodes = f["mesh/nodes"][...]
        conn = f["mesh/connectivity"][...]
        sigma = f["model/conductivity_tensor"][...]  # (n_cells, 3, 3)
    return dict(nodes=nodes, conn=conn, sigma=sigma)


def assemble_tensor_from_components(sxx, syy, szz, sxy, sxz, syz):
    """Build (n_cells, 3, 3) symmetric tensor field from 6 components,
    if that is how your HDF5 writer stores ResistivityBlockAnisotropic."""
    n = sxx.shape[0]
    sigma = np.zeros((n, 3, 3))
    sigma[:, 0, 0] = sxx
    sigma[:, 1, 1] = syy
    sigma[:, 2, 2] = szz
    sigma[:, 0, 1] = sigma[:, 1, 0] = sxy
    sigma[:, 0, 2] = sigma[:, 2, 0] = sxz
    sigma[:, 1, 2] = sigma[:, 2, 1] = syz
    return sigma


def eigen_decompose_all(sigma):
    """
    Vectorized eigendecomposition over all cells.
    sigma: (N, 3, 3) symmetric conductivity tensors.
    Returns:
      rho:      (N, 3) principal resistivities, sorted descending per cell
      R:        (N, 3, 3) rotation matrices, columns = principal directions,
                 reordered to match rho and forced to proper rotations
                 (det = +1) to avoid mirrored glyphs
    """
    eigvals, eigvecs = np.linalg.eigh(sigma)  # ascending eigvals, batched
    eigvals = np.clip(eigvals, 1e-12, None)
    rho_all = 1.0 / eigvals  # (N, 3), ascending -> resistivity descending
    order = np.argsort(-rho_all, axis=1)  # descending resistivity

    N = sigma.shape[0]
    rho = np.take_along_axis(rho_all, order, axis=1)
    R = np.take_along_axis(eigvecs, order[:, None, :], axis=2)

    dets = np.linalg.det(R)
    flip = dets < 0
    R[flip, :, 2] *= -1
    return rho, R


def build_unstructured_grid(model):
    """Construct a pv.UnstructuredGrid from tetrahedral connectivity."""
    n_cells = model["conn"].shape[0]
    # VTK cell array format: [n_pts, id0, id1, ..., n_pts, id0, ...]
    cells = np.hstack([np.full((n_cells, 1), 4), model["conn"]]).astype(np.int64)
    cell_type = np.full(n_cells, pv.CellType.TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, cell_type, model["nodes"])
    return grid


def main(h5_path, out_html="aniso_model_view.html",
         glyph_fraction=0.02, anisotropy_threshold=1.15, scale=0.4):
    model = load_aniso_model(h5_path)
    rho, R = eigen_decompose_all(model["sigma"])
    rho_eq = np.cbrt(np.prod(rho, axis=1))

    grid = build_unstructured_grid(model)
    grid.cell_data["log10_rho_eq"] = np.log10(rho_eq)

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh_clip_plane(grid, normal="z", scalars="log10_rho_eq",
                                 cmap="viridis", show_edges=False)

    # subsample cells for glyphing -- full-resolution tet meshes are
    # typically far too dense (10^5-10^6 cells) to glyph individually
    centroids = grid.cell_centers().points
    anisotropic = (rho[:, 0] / rho[:, 2]) >= anisotropy_threshold
    candidates = np.nonzero(anisotropic)[0]
    n_glyphs = max(1, int(len(candidates) * glyph_fraction))
    rng = np.random.default_rng(0)
    chosen = rng.choice(candidates, size=min(n_glyphs, len(candidates)),
                         replace=False)

    for idx in chosen:
        ellipsoid = pv.ParametricEllipsoid(
            xradius=scale * np.log10(rho[idx, 0] + 1),
            yradius=scale * np.log10(rho[idx, 1] + 1),
            zradius=scale * np.log10(rho[idx, 2] + 1),
        )
        transform = np.eye(4)
        transform[:3, :3] = R[idx]
        transform[:3, 3] = centroids[idx]
        ellipsoid.transform(transform, inplace=True)
        plotter.add_mesh(ellipsoid, color="orange",
                          smooth_shading=True, opacity=0.85)

    plotter.add_scalar_bar(title="log10(rho_eq) [ohm.m]")
    plotter.show_grid()
    plotter.export_html(out_html)
    print(f"Wrote {out_html} ({len(chosen)} glyphs of "
          f"{len(candidates)} anisotropic cells)")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "model.h5")
