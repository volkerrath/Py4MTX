# femtic_ensemble_eof.py

Empirical Orthogonal Function (EOF) analysis of a FEMTIC model ensemble.

## Provenance

| Field | Value |
|-------|-------|
| Script | `femtic_ensemble_eof.py` |
| Author | vrath |
| Part of | **py4mt** — Python for Magnetotellurics |
| Inversion code | FEMTIC |
| README generated | 2 March 2026 by Claude (Anthropic), from cleaned source |

## Purpose

Collects converged FEMTIC inversion results into an ensemble matrix,
computes EOFs via SVD (demeaned), and generates reconstructed model
samples from truncated or individual EOF modes. This provides a
data-driven decomposition of the ensemble variability.

## Inputs

| Item | Description |
|------|-------------|
| `EnsembleDir` | Root directory containing inversion sub-directories. |
| `EnsembleName` | Prefix pattern for directory matching (e.g. `ann_`). |
| `MinRMS` | Maximum acceptable nRMS for inclusion. |
| `MeshFile` | Relative path to the FEMTIC mesh file within each run directory. |

Each run directory must contain `femtic.cnv`, `mesh.dat`, and the final
`resistivity_block_iter<N>.dat`.

## Outputs

| File | Contents |
|------|----------|
| `<EnsembleFile>.npz` | Raw ensemble matrix, mesh nodes and connectivity. |
| `<EnsembleEOF>.npz` | EOFs, principal components, weights, fractional variance, mean model, ensemble, and mesh. |
| `<EnsembleEOF>_trunc<N>.npz` | Truncated EOF reconstruction using modes 0…N (or individual mode N if `GetComponents=True`). |

## Configuration

- `GetComponents` — if `True`, store each EOF mode individually (`_comp`); if `False`, store cumulative truncations (`_trunc`).
- `MinRMS` — convergence threshold.
- `EnsembleFile`, `EnsembleEOF` — base names for output files.

## Dependencies

`numpy`, py4mt modules: `femtic`, `util`, `ensembles`, `version`.
