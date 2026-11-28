# kp_plot.py

Retrieves planetary Kp indices from GFZ Potsdam (JSON API,
https://kp.gfz.de/app/json/) and plots the three-hourly values, the daily maxima
and the storm threshold Kp = 5.

## Usage

```
python kp_plot.py                                       # April 2023 (default)
python kp_plot.py --start 2025-09-01 --end 2025-09-30
python kp_plot.py --start 2025-09-01 --end 2025-09-30 --csv kp.csv --output kp.pdf
python kp_plot.py --size 25 10 --output kp.png          # figure size in cm
python kp_plot.py --title "Kp, {start} to {end}"        # own title; "" = none
python kp_plot.py --help
```

Dates are UTC; the end date is included. The values are printed, with the
maximum and the number of three-hour intervals with Kp >= 5. The figure is shown
at the end.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--start`, `--end` | `2023-04-01`, `2023-04-30` | first and last date (YYYY-MM-DD) |
| `--all` | off | include values that are not definitive (default: definitive only) |
| `--csv FILE` | none | save the data |
| `--output FILE` | none | save the figure as `.pdf`, `.jpg` / `.jpeg` or `.png` (300 dpi); may be repeated |
| `--png FILE` | none | older option for one figure file |
| `--size WIDTH HEIGHT` | `33.02 11.43` | figure size in **cm** |
| `--title` | `auto` | `auto` = "Planetary geomagnetic Kp index (GFZ)"; `""` = none; else the text, with `{start}` and `{end}` |
| `--label-fontsize` | `11` | axis labels [pt] |
| `--tick-fontsize` | `10` | tick labels [pt] |
| `--title-fontsize` | `12` | title [pt] |
| `--legend-fontsize` | `8` | legend [pt] |
| `--daily-linestyle` | `-` | line of the daily maxima |
| `--threshold-linestyle` | `--` | storm threshold |
| `--grid-linestyle` | `:` | grid |

## Functions

| Function | Purpose |
|---|---|
| `get_kp(start, end, definitive=True)` | DataFrame with a UTC time index and the column `Kp` |
| `plot_kp(df, ax=None, daily_max=True, storm_level=5.0, ..., title="auto", size=(33.02, 11.43))` | draws the figure and returns the axes; `size` [cm] is used only if `ax` is `None` |
| `kp_title(title, df)` | the title text for the `title` argument |

## Dependencies

matplotlib, numpy, pandas, requests; network access to `kp.gfz.de`. Check the
terms of GFZ for the use and citation of the data.

## Testing

`test_kp_plot_title.py` (titles). The figure size was checked on made-up data;
the download from GFZ was not run.

## Provenance

| Date | Author | Change |
|---|---|---|
| 2026-10-17 | Volker Rath (DIAS), with the help of ChatGPT (GPT-6) | script created |
| 2026-11-11 | Claude Sonnet 5.5 (Anthropic) | `--title` / `plot_kp(title=...)`. AI-generated; review before use. |
| 2026-11-27 | Claude Sonnet 5.5 (Anthropic) | `--size` / `plot_kp(size=...)`: figure size in cm, default 33.02 x 11.43 (the former 13 x 4.5 inch, fixed in the code). Readme written. AI-generated; review before use. |
