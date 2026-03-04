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
| Last cleanup | 4 March 2026 by Claude (Anthropic) |

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

## Changes in this cleanup (4 Mar 2026)

| Change | Description |
|--------|-------------|
| **UPPERCASE config** | All configuration constants renamed to `UPPER_SNAKE_CASE` (`BLANK`, `RHOAIR`, `MOD_DIR_IN`, `MOD_DIR_OUT`, `MOD_FILE_IN`, `MOD_FILE_OUT`, `MOD_ORIG`, `SVD_FILE`, `MOD_OUT_SINGLE`, `PADDING`, `BODY_MASK`, `BODY_VAL`, `FLIP`, `MODEL_SET`, `METHOD`). |
| **Unused imports** | Removed `numpy.linalg` (not used). |
| **Unused variables** | Removed `rng`, `nan` (never used). |
| **Provenance line** | Added cleanup date to docstring. |

## Configuration

| Constant | Description |
|----------|-------------|
| `MODEL_SET` | Number of perturbed models to generate |
| `METHOD` | Perturbation strategy list |
| `BODY_VAL` | Perturbation amplitude (log10 resistivity) |
| `BODY_MASK` | Body dimensions in grid cells `[nx, ny, nz]` |
| `SVD_FILE` | `.npz` file containing the Jacobian SVD (`U`, `S`) |
| `MOD_DIR_OUT` | Output directory for perturbed models |

## Dependencies

`numpy`, `scipy.sparse`, py4mt: `modem`, `jac_proc`, `util`, `version`.
