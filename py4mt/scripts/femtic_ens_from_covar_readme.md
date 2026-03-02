# femtic_ens_from_covar.py

Generate a synthetic model ensemble from a posterior covariance matrix.

## Provenance

| Field | Value |
|-------|-------|
| Script | `femtic_ens_from_covar.py` |
| Author | vrath |
| Part of | **py4mt** — Python for Magnetotellurics |
| Inversion code | FEMTIC |
| README generated | 2 March 2026 by Claude (Anthropic), from cleaned source |

## Purpose

Given a posterior model covariance (e.g. from a randomise-then-optimise
workflow), draws new model samples using the Cholesky decomposition:

    m_new = m_ref + L · z,    z ~ N(0, I)

where L is the Cholesky factor of the covariance. This follows the
method of Osypov et al. (2013).

## Reference

Osypov, K. et al. (2013): Model-uncertainty quantification in seismic
tomography: method and applications. *Geophysical Prospecting*, 61,
1114–1134. doi:10.1111/1365-2478.12058

## Inputs

| Item | Description |
|------|-------------|
| `CovarResults` | `.npz` file containing `rto_cov` (covariance), `rto_avg` (reference model), `rto_var` (variance). |

## Outputs

| File | Contents |
|------|----------|
| `<NewEnsembleFile>.npz` | `new_ens` (ensemble matrix, N×M), `sqrtcov` (Cholesky factor), `ref` (reference model). |

## Configuration

- `CovarResults` — path to the input covariance `.npz`.
- `NewEnsembleSize` — number of samples to draw (default 100).
- `NewEnsembleFile` — output path.

## Dependencies

`numpy`, `scikit-sparse` (cholmod for sparse Cholesky), py4mt: `femtic`, `util`, `version`.
