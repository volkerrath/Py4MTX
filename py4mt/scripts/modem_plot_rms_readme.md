# modem_plot_rms.py

Plot nRMS convergence curves from ModEM NLCG log files.

## Provenance

| Field | Value |
|-------|-------|
| Script | `modem_plot_rms.py` |
| Author | sbyrd |
| Part of | **py4mt** — Python for Magnetotellurics |
| Inversion code | ModEM |
| README generated | 2 March 2026 by Claude (Anthropic), from cleaned source |

## Purpose

Parses one or more ModEM `.log` files to extract the normalised RMS
at each iteration, then plots all convergence curves on a single figure.
Also writes iteration/nRMS data to `.csv` files for external use.

## How it works

The script scans each log file for lines containing `START:` or
`STARTLS:` together with `rms=`, extracts the nRMS value (field 6),
and appends it to a list.

## Inputs

| Item | Description |
|------|-------------|
| `filename_in` | List of ModEM `.log` file paths. |
| `legend_labels` | Corresponding legend entries for each run. |

## Outputs

| File | Contents |
|------|----------|
| `<logfile>.csv` | Two-column CSV: iteration number, nRMS. |
| `rms.pdf` | Convergence plot with one curve per log file. |

## Configuration

- `directory` — working directory.
- `filename_in` — list of `.log` file paths.
- `legend_labels` — list of legend strings (must match `filename_in` length).

## Dependencies

`numpy`, `matplotlib`.
