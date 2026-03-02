# modem_insert_multi.py

Generate multiple perturbed models for null-space analysis.

## Provenance

| Field | Value |
|-------|-------|
| Script | `modem_insert_multi.py` |
| Author | vrath |
| Part of | **py4mt** — Python for Magnetotellurics |
| Inversion code | ModEM |
| README generated | 2 March 2026 by Claude (Anthropic), from cleaned source |

## Purpose

Creates a set of perturbed ModEM models (e.g. checkerboard or random
anomaly patterns), then projects each perturbation through the Jacobian
null-space using a pre-computed SVD. This tests what structures the data
can actually resolve.

## Workflow

1. Read the base model and its Jacobian SVD (U, S matrices).
2. For each sample: generate a perturbation pattern via
   `mod.distribute_bodies_ijk` + `mod.insert_body_ijk`.
3. Project the perturbed model onto the Jacobian column space
   with `jac.project_model`.
4. Write the projected model.

## Inputs

| Item | Description |
|------|-------------|
| `ModFile_in` | Path to the base ModEM model. |
| `SVDFile` | `.npz` file containing the Jacobian SVD (`U`, `S`). |

## Configuration

- `model_set` — number of perturbed models to generate.
- `method` — perturbation strategy: `['random', N, padding, distribution, bodymask, seed]` or `['regular', ...]`.
- `bodyval` — perturbation amplitude (log10 resistivity).
- `bodymask` — body dimensions in grid cells `[nx, ny, nz]`.
- `ModDir_out` — output directory for perturbed models.

## Outputs

One `.rho` file per perturbation in `ModDir_out`.

## Dependencies

`numpy`, `scipy.sparse`, py4mt: `modem`, `jac_proc`, `util`, `version`.
