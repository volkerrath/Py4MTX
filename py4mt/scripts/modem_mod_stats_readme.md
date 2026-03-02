# modem_mod_stats.py

Cell-wise statistics and histograms for ModEM model ensembles.

## Provenance

| Field | Value |
|-------|-------|
| Script | `modem_mod_stats.py` |
| Authors | vrath, sbyrd |
| Part of | **py4mt** — Python for Magnetotellurics |
| Inversion code | ModEM |
| README generated | 2 March 2026 by Claude (Anthropic), from cleaned source |

## Purpose

Reads one or more ModEM `.rho` model files and computes cell-wise
statistics: mean, variance, median, and percentile (1σ / 2σ) models.
For single models, prints summary statistics and optionally generates
a log-resistivity histogram. For ensembles, writes statistical summary
models in ModEM and/or RLM format.

## Single model mode

- Masks air cells and padding (configurable horizontal/vertical pad widths).
- Prints mean, median, standard deviation, and quantiles of log₁₀(ρ).
- Optionally saves a histogram as `.png` and/or `.pdf`.

## Ensemble mode (multiple models)

Computes across the ensemble (cell-by-cell):

| Output | Description |
|--------|-------------|
| `_avg.rho` | Mean log-resistivity. |
| `_var.rho` | Standard deviation of log-resistivity. |

Percentile models (15.9th, 50th, 84.1th) are also computed internally.

## Inputs

| Item | Description |
|------|-------------|
| `Models` | List of ModEM model file paths. |

## Outputs

- Histogram plots (`.png`, `.pdf`).
- Optional PDF catalog of all histograms.
- Ensemble average and variance models (`.rho` and/or `.rlm`).

## Configuration

- `hpads`, `vpads` — number of horizontal / vertical padding cells to mask.
- `OutFormat` — `'modem'` and/or `'rlm'`.
- `ModHist` — `True` to generate histograms.
- `PlotFormat` — list of plot formats (e.g. `['.png', '.pdf']`).

## Dependencies

`numpy`, `matplotlib`, py4mt: `modem`, `util`, `version`.
