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
| Last cleanup | 4 March 2026 by Claude (Anthropic) |

## Purpose

Parses one or more ModEM `.log` files to extract the normalised RMS
at each iteration, then plots all convergence curves on a single figure.
Also writes iteration/nRMS data to `.csv` files for external use.

## How it works

The script scans each log file for lines containing `START:` or
`STARTLS:` together with `rms=`, extracts the nRMS value (field 6),
and appends it to a list.

## Outputs

| File | Contents |
|------|----------|
| `<logfile>.csv` | Two-column CSV: iteration number, nRMS. |
| `rms.pdf` | Convergence plot with one curve per log file. |

## Changes in this cleanup (4 Mar 2026)

| Change | Description |
|--------|-------------|
| **UPPERCASE config** | All configuration constants renamed to `UPPER_SNAKE_CASE` (`DIRECTORY`, `FILENAME_IN`, `LEGEND_LABELS`, `PLOT_FILE`). |
| **Output path** | Extracted hard-coded `directory + "rms.pdf"` into `PLOT_FILE` constant. |
| **Provenance line** | Added cleanup date to docstring. |

## Configuration

| Constant | Description |
|----------|-------------|
| `DIRECTORY` | Working directory |
| `FILENAME_IN` | List of `.log` file paths |
| `LEGEND_LABELS` | Legend strings (must match `FILENAME_IN` length) |
| `PLOT_FILE` | Output plot path |

## Dependencies

`numpy`, `matplotlib`.
