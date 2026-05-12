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
| Last cleanup | 4 March 2026 by Claude (Anthropic) |

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

## Changes in this cleanup (4 Mar 2026)

| Change | Description |
|--------|-------------|
| **UPPERCASE config** | All configuration constants renamed to `UPPER_SNAKE_CASE` (`HPADS`, `VPADS`, `OUT_FORMAT`, `BLANK`, `RHOAIR`, `MOD_HIST`, `PLOT_FORMAT`, `WORK_DIR`, `M_FILE`, `MODELS`, `PDF_CATALOG`, `PDF_CATALOG_NAME`, `MOD_FILE_ENS`). |
| **Unused imports** | Removed `time` (not used). |
| **Unused variables** | Removed `rng`, `blank` (shadowed), `rhoair` (shadowed by constant). |
| **Fragile counter** | Replaced `imod = -1` + increment with `enumerate()`. |
| **Fragile ensemble init** | Replaced `imod == 0` check with `ModEns is None` guard. |
| **Provenance line** | Added cleanup date to docstring. |

## Configuration

| Constant | Description |
|----------|-------------|
| `HPADS`, `VPADS` | Number of horizontal / vertical padding cells to mask |
| `OUT_FORMAT` | `'modem'` and/or `'rlm'` |
| `MOD_HIST` | `True` to generate histograms |
| `PLOT_FORMAT` | List of plot formats (e.g. `['.png', '.pdf']`) |
| `MODELS` | List of ModEM model file paths |

## Dependencies

`numpy`, `matplotlib`, py4mt: `modem`, `util`, `version`.
