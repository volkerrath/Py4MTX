
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retrieve and plot planetary Kp indices from GFZ Potsdam.

Plots:
    - Three-hourly Kp values
    - Daily maximum Kp values
    - Storm threshold at Kp = 5

Usage
-----
    python kp_plot.py [--start YYYY-MM-DD] [--end YYYY-MM-DD]
                      [--csv FILE] [--output FILE] [--png FILE] [--all]

Examples
--------
    # Plot April 2023 (default)
    python kp_plot.py

    # Plot September 2025
    python kp_plot.py --start 2025-09-01 --end 2025-09-30

    # Save data and figure
    python kp_plot.py --start 2025-09-01 --end 2025-09-30 \
        --csv kp_sep2025.csv --png kp_sep2025.png

    # Save PDF and JPEG figures
    python kp_plot.py --output kp_april2023.pdf --output kp_april2023.jpg

    # Include preliminary and definitive Kp values
    python kp_plot.py --start 2026-10-01 --end 2026-10-17 --all

    # Customize fonts and line styles
    python kp_plot.py --label-fontsize 12 --tick-fontsize 10 \
        --legend-fontsize 7 --daily-linestyle dashdot

    # Show command-line help
    python kp_plot.py --help

Notes
-----
    Dates are interpreted as UTC.
    The end date is inclusive.
    Kp values are sampled at three-hour intervals.
    By default, only definitive GFZ Kp values are requested.


Author: Volker Rath (DIAS)
Created with the help of ChatGPT (GPT-6) on 2026-10-17
"""

import argparse

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests


def get_kp(start="2023-04-01", end="2023-04-30",
           definitive=True):
    """
    Retrieve planetary Kp values from the GFZ JSON API.

    Parameters
    ----------
    start, end : str
        Inclusive UTC calendar dates, YYYY-MM-DD.
    definitive : bool
        Request only definitive Kp indices.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame indexed by UTC time with numeric Kp values.
    """
    t0 = pd.Timestamp(start, tz="UTC")
    t1 = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)

    if t1 <= t0:
        raise ValueError("End date must not precede start date.")

    url = "https://kp.gfz.de/app/json/"
    params = {
        "start": t0.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": (t1 - pd.Timedelta(seconds=1)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "index": "Kp",
        "status": "def" if definitive else "all",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame({
        "time": pd.to_datetime(data["datetime"], utc=True),
        "Kp": pd.to_numeric(data["Kp"], errors="coerce"),
    })

    df = df.set_index("time").sort_index()
    df = df.loc[(df.index >= t0) & (df.index < t1)]

    if df.empty:
        raise ValueError("No Kp data returned for this interval.")

    return df


def plot_kp(df, ax=None, daily_max=True, storm_level=5.0,
            label_fontsize=11, tick_fontsize=10, title_fontsize=12,
            legend_fontsize=8, daily_linestyle="-",
            threshold_linestyle="--", grid_linestyle=":"):
    """
    Plot three-hourly Kp indices and optionally daily maxima.

    Parameters
    ----------
    df : pandas.DataFrame
        UTC-indexed DataFrame containing column 'Kp'.
    ax : matplotlib.axes.Axes or None
        Existing axes, or create new axes.
    daily_max : bool
        Overlay daily maximum Kp values.
    storm_level : float
        Threshold for highlighting geomagnetic storms.

    label_fontsize, tick_fontsize, title_fontsize, legend_fontsize : float
        Font sizes in points; legend defaults to 8 pt.
    daily_linestyle, threshold_linestyle, grid_linestyle : str
        Matplotlib line styles for the daily maximum, threshold, and grid.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Plot axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 4.5))

    kp = df["Kp"]
    storm = kp >= storm_level

    ax.bar(
        kp.index, kp.to_numpy(),
        width=0.115, align="edge",
        color="steelblue", label="3-hourly Kp"
    )

    ax.bar(
        kp.index[storm], kp[storm].to_numpy(),
        width=0.115, align="edge",
        color="firebrick", label="Kp >= 5"
    )

    if daily_max:
        daily = kp.resample("1D").max()
        ax.step(
            daily.index, daily.to_numpy(),
            where="post", color="black",
            linewidth=1.4, linestyle=daily_linestyle, label="Daily maximum"
        )

    ax.axhline(
        storm_level, color="darkred",
        linestyle=threshold_linestyle, linewidth=1,
        label="Storm threshold"
    )

    ax.set_ylabel(r"Planetary $K_p$", fontsize=label_fontsize)
    ax.set_xlabel("Date (UTC)", fontsize=label_fontsize)
    ax.set_ylim(0, 9.2)
    ax.set_yticks(np.arange(0, 10, 1))
    ax.set_title("Planetary geomagnetic Kp index (GFZ)", fontsize=title_fontsize)
    ax.grid(True, linestyle=grid_linestyle, alpha=0.5)
    ax.legend(loc="upper left", fontsize=legend_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
    )

    return ax


def main():
    """Parse command-line arguments, retrieve, plot, and save."""
    parser = argparse.ArgumentParser(
        description="Retrieve and plot GFZ planetary Kp indices"
    )
    parser.add_argument("--start", default="2023-04-01")
    parser.add_argument("--end", default="2023-04-30")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--png", default=None,
                        help="Save PNG figure (legacy option)")
    parser.add_argument(
        "--output", action="append", default=[], metavar="FILE",
        help="Save figure as PDF, JPG/JPEG, or PNG based on extension; "
             "repeat to save multiple formats"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Include non-definitive values"
    )
    parser.add_argument("--label-fontsize", type=float, default=11)
    parser.add_argument("--tick-fontsize", type=float, default=10)
    parser.add_argument("--title-fontsize", type=float, default=12)
    parser.add_argument("--legend-fontsize", type=float, default=8)
    parser.add_argument("--daily-linestyle", default="-")
    parser.add_argument("--threshold-linestyle", default="--")
    parser.add_argument("--grid-linestyle", default=":")
    args = parser.parse_args()

    df = get_kp(
        args.start, args.end, definitive=not args.all
    )

    print(df.to_string())
    print(f"\nMaximum Kp: {df['Kp'].max():.2f}")
    print(f"Intervals with Kp >= 5: {(df['Kp'] >= 5).sum()}")

    if args.csv:
        df.to_csv(args.csv)

    ax = plot_kp(
        df, label_fontsize=args.label_fontsize,
        tick_fontsize=args.tick_fontsize,
        title_fontsize=args.title_fontsize,
        legend_fontsize=args.legend_fontsize,
        daily_linestyle=args.daily_linestyle,
        threshold_linestyle=args.threshold_linestyle,
        grid_linestyle=args.grid_linestyle,
    )
    ax.figure.tight_layout()

    outputs = list(args.output)
    if args.png:
        outputs.append(args.png)

    for filename in outputs:
        extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if extension not in {"pdf", "jpg", "jpeg", "png"}:
            parser.error(
                f"Unsupported output format for {filename!r}; "
                "use .pdf, .jpg, .jpeg, or .png"
            )
        ax.figure.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
