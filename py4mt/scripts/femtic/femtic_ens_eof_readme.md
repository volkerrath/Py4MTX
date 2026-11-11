# femtic_ens_eof.py

Empirical Orthogonal Function (EOF) analysis of a FEMTIC model ensemble.

## Provenance

| Field | Value |
|-------|-------|
| Script | `femtic_ens_eof.py` |
| Author | vrath |
| Part of | **py4mt** — Python for Magnetotellurics |
| Inversion code | FEMTIC |
| README generated | 2 March 2026 by Claude (Anthropic), from cleaned source |

## Provenance

| Date       | Author | Change                                                          |
|------------|--------|-----------------------------------------------------------------|
| 2025       | vrath  | Created (as femtic_ensemble_eof.py).                            |
| 2026-03-03 | Claude | Renamed file to femtic_ens_eof; params to UPPERCASE.            |
| 2026-08-13 | Claude Sonnet 5 | Added `femtic_ens_eof_summary.md` output at end of run: user-set (UPPERCASE) parameters, script path, and run date/time. |
| 2026-10-28 | Claude Sonnet 5.5 (Anthropic) | The parameter summary `femtic_ens_eof_summary.md` is now written to `ENSEMBLE_DIR` (the working directory set in the script) instead of next to the script: `utl.write_param_summary(fname, out_dir=ENSEMBLE_DIR)`. Needs the `util.py` of the same date. AI-generated; review before production use. |
| 2026-11-11 | Claude Sonnet 5.5 (Anthropic) | New `CNV_FILE` (one name or a list of candidate names; default `["femtic.cnv", "FEMTIC-DABIC.cnv"]`), passed to `fem.find_cnv_file()` and `fem.get_nrms()`. Needs the `femtic.py` of the same date. AI-generated; review before production use. |

## Purpose

Collects converged FEMTIC inversion results into an ensemble matrix,
computes EOFs via SVD (demeaned), and generates reconstructed model
samples from truncated or individual EOF modes. This provides a
data-driven decomposition of the ensemble variability.

## Inputs

| Item | Description |
|------|-------------|
| `ENSEMBLE_DIR` | Root directory containing inversion sub-directories. |
| `ENSEMBLE_NAME` | Prefix pattern for directory matching (e.g. `ann_`). |
| `MIN_RMS` | Maximum acceptable nRMS for inclusion. |
| `CNV_FILE` | Name of the convergence file in each run directory: one name or a list of candidate names (default `["femtic.cnv", "FEMTIC-DABIC.cnv"]`; the first one that exists is used). |
| `MESH_FILE` | Relative path to the FEMTIC mesh file within each run directory. |

Each run directory must contain the convergence file (`CNV_FILE`, default `femtic.cnv` or `FEMTIC-DABIC.cnv`), `mesh.dat`, and the final
`resistivity_block_iter<N>.dat`.

## Outputs

| File | Contents |
|------|----------|
| `<ENSEMBLE_FILE>.npz` | Raw ensemble matrix, mesh nodes and connectivity. |
| `<ENSEMBLE_EOF>.npz` | EOFs, principal components, weights, fractional variance, mean model, ensemble, and mesh. |
| `<ENSEMBLE_EOF>_trunc<N>.npz` | Truncated EOF reconstruction using modes 0…N (or individual mode N if `GET_COMPONENTS=True`). |

## Configuration

- `GET_COMPONENTS` — if `True`, store each EOF mode individually (`_comp`); if `False`, store cumulative truncations (`_trunc`).
- `MIN_RMS` — convergence threshold.
- `ENSEMBLE_FILE`, `ENSEMBLE_EOF` — base names for output files.

## Dependencies

`numpy`, py4mt modules: `femtic`, `util`, `ensembles`, `version`.
