#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plt_sounding_curve_impedance.py
================================
PyGMT translation of plt_sounding_curve_impedance_full2_stddev.sh,
rewritten to consume the output of femtic_data.py directly.

Supported result files (femtic_data.py output):
  result_MT.txt              -- impedance as Re/Im Z  (default)
  result_MT.txt  -appphs     -- impedance as AppRes / Phase
  result_APP_RES_AND_PHS.txt -- native AppRes / Phase data type

Column layout written by write_result_mt() WITH -appphs (no distortion):
  Site  Frequency
  AppRxxCal PhsxxCal  AppRxyCal PhsxyCal  AppRyxCal PhsyxCal  AppRyyCal PhsyyCal
  AppRxxObs PhsxxObs  AppRxyObs PhsxyObs  AppRyxObs PhsyxObs  AppRyyObs PhsyyObs
  AppRxxErr PhsxxErr  AppRxyErr PhsxyErr  AppRyxErr PhsyxErr  AppRyyErr PhsyyErr

Column layout written by write_result_mt() WITHOUT -appphs (no distortion):
  Site  Frequency
  ReZxxCal ImZxxCal  ReZxyCal ImZxyCal  ReZyxCal ImZyxCal  ReZyyCal ImZyyCal
  ReZxxObs ImZxxObs  ReZxyObs ImZxyObs  ReZyxObs ImZyxObs  ReZyyObs ImZyyObs
  ReZxxErr ImZxxErr  ReZxyErr ImZxyErr  ReZyxErr ImZyxErr  ReZyyErr ImZyyErr

Usage
-----
    python plt_sounding_curve_impedance.py [options]

Options
-------
  --result   FILE   Result file to read  (default: result_MT.txt)
  --sites    FILE   Site list file       (default: sites_imp.txt)
                    Two-column whitespace-separated: <site_name> <label>
                    If omitted, all sites in result file are plotted in order.
  --out      PREFIX Output file prefix   (default: imp_all_curv)
  --per-page N      Stations per output PDF page (default: 12)
  --cols     N      Columns per page     (default: 6)
  --appphs          Force AppRes/Phase interpretation of input columns
  --impedance       Force Re/Im impedance interpretation of input columns

Notes
-----
* When impedance (Re/Im Z) is read, apparent resistivity and phase are computed
  internally using the same formulae as femtic_data.py:
      AppRes    = |Z|^2 / (omega * mu0)
      Phase     = atan2(Im Z, Re Z)   [degrees]
      err_AppRes = 2 * |Z_obs| * err / (omega * mu0)
      err_Phase  = asin(err / |Z_obs|) [degrees, capped at 180]
  where err = max(re_err, im_err) per component.
* Phase is plotted sign-flipped (as in the original shell script).
* The --sites file uses site *names* (strings) as produced by femtic_data.py
  when the -name option is given, OR the numeric site IDs if -name was not used.
"""

import argparse
import math
import sys

import numpy as np
import pandas as pd
import pygmt

# ---------------------------------------------------------------------------
# Physical constants  (same values as femtic_data.py)
# ---------------------------------------------------------------------------
MU0     = 4.0 * math.pi * 1.0e-7
RAD2DEG = 180.0 / math.pi

# ---------------------------------------------------------------------------
# Plot style  (matching the shell script)
# ---------------------------------------------------------------------------
COMPS = ["xx", "xy", "yx", "yy"]
COLOR = {
    "xx": "255/128/0",   # orange
    "xy": "255/0/0",     # red
    "yx": "0/0/255",     # blue
    "yy": "0/255/0",     # green
}

RHO_REGION = [1e-4, 1e4, 1e-1, 1e4]   # [T_min, T_max, rho_min, rho_max]
PHS_REGION = [1e-4, 1e4, -180, 180]

RHO_W = 3.8   # cm – panel width
RHO_H = 3.0   # cm – rho panel height
PHS_H = 3.0   # cm – phase panel height

LINE_PEN = "0.5p"
SYM_SIZE = "0.15c"
ERR_CAP  = "0.15c"

# vertical distances between sub-panels / rows
RHO_TO_PHS = RHO_H + 0.5           # shift down from rho to phase panel top
PANEL_DX   = RHO_W + 0.5           # horizontal column pitch
PANEL_DY   = RHO_TO_PHS + PHS_H + 0.7  # full row height including label gap


# ---------------------------------------------------------------------------
# Column-name helpers  (must match femtic_data.py write_result_mt exactly)
# ---------------------------------------------------------------------------

def _cols_appphs() -> list:
    """Header columns for AppRes/Phase output (no distortion)."""
    cols = ["Site", "Frequency"]
    for c in COMPS: cols += [f"AppR{c}Cal", f"Phs{c}Cal"]
    for c in COMPS: cols += [f"AppR{c}Obs", f"Phs{c}Obs"]
    for c in COMPS: cols += [f"AppR{c}Err", f"Phs{c}Err"]
    return cols


def _cols_impedance() -> list:
    """Header columns for Re/Im impedance output (no distortion)."""
    cols = ["Site", "Frequency"]
    for c in COMPS: cols += [f"ReZ{c}Cal", f"ImZ{c}Cal"]
    for c in COMPS: cols += [f"ReZ{c}Obs", f"ImZ{c}Obs"]
    for c in COMPS: cols += [f"ReZ{c}Err", f"ImZ{c}Err"]
    return cols


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def detect_mode(result_file: str) -> str:
    """Peek at header; return 'appphs' or 'impedance'."""
    with open(result_file) as fh:
        header = fh.readline()
    return "appphs" if "AppR" in header else "impedance"


def load_result(result_file: str, mode: str) -> pd.DataFrame:
    """Read femtic_data.py result file into a DataFrame."""
    cols = _cols_appphs() if mode == "appphs" else _cols_impedance()
    df = pd.read_csv(result_file, sep=r"\s+", header=0, names=cols)
    df["Site"] = df["Site"].astype(str)
    return df


def load_sites(sites_file: str) -> list:
    """
    Read optional two-column site list.
    Returns list of (site_name, label) tuples in plotting order.
    """
    pairs = []
    with open(sites_file) as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            name  = parts[0]
            label = parts[1] if len(parts) >= 2 else name
            pairs.append((name, label))
    return pairs


# ---------------------------------------------------------------------------
# Impedance → AppRes / Phase conversion
# ---------------------------------------------------------------------------

def impedance_to_appphs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add AppR*/Phs* columns, computed from ReZ*/ImZ* columns using the
    same arithmetic as femtic_data.py write_result_mt().
    """
    freq  = df["Frequency"].values
    omega = 2.0 * math.pi * freq

    out = df[["Site", "Frequency"]].copy()

    for c in COMPS:
        reC = df[f"ReZ{c}Cal"].values
        imC = df[f"ImZ{c}Cal"].values
        absC = np.hypot(reC, imC)
        out[f"AppR{c}Cal"] = absC**2 / (omega * MU0)
        out[f"Phs{c}Cal"]  = RAD2DEG * np.arctan2(imC, reC)

        reO = df[f"ReZ{c}Obs"].values
        imO = df[f"ImZ{c}Obs"].values
        absO = np.hypot(reO, imO)
        out[f"AppR{c}Obs"] = absO**2 / (omega * MU0)
        out[f"Phs{c}Obs"]  = RAD2DEG * np.arctan2(imO, reO)

        # err = max(re_err, im_err)  — same convention as femtic_data.py
        reE = df[f"ReZ{c}Err"].values
        imE = df[f"ImZ{c}Err"].values
        err = np.maximum(reE, imE)

        out[f"AppR{c}Err"] = 2.0 * absO * err / (omega * MU0)
        ratio = np.where(absO > 0, err / absO, 10.0)
        out[f"Phs{c}Err"] = np.where(
            ratio <= 1.0,
            RAD2DEG * np.arcsin(np.clip(ratio, -1.0, 1.0)),
            180.0,
        )

    return out


# ---------------------------------------------------------------------------
# GMT frame helpers
# ---------------------------------------------------------------------------

def _rho_frame(left: bool) -> list:
    w = "W" if left else "w"
    return [f"{w}sne",
            "xa1f3+lLog(Period,s)",
            "ya1f3g1+lLog(App. Resistivity,@~W@~m)"]


def _phs_frame(left: bool, bottom: bool) -> list:
    w = "W" if left   else "w"
    s = "S" if bottom else "s"
    return [f"{w}{s}ne",
            "xa1f3+lLog(Period,s)",
            "ya45f10g45+lPhase,deg."]


# ---------------------------------------------------------------------------
# Single-station plot
# ---------------------------------------------------------------------------

def plot_station(
    fig: pygmt.Figure,
    df_site: pd.DataFrame,
    label: str,
    show_left: bool,
    show_bottom: bool,
) -> None:
    """
    Plot rho panel (at current origin) then phase panel (shifted down).
    Origin is restored to the rho-panel top before returning.
    """
    period   = 1.0 / df_site["Frequency"].values
    rho_proj = f"X{RHO_W}l/{RHO_H}l"
    phs_proj = f"X{RHO_W}l/{PHS_H}"

    # ------------------------------------------------------------------ rho --
    frame_todo = _rho_frame(show_left)   # drawn once, then set to None

    for c in COMPS:
        col = COLOR[c]

        # model curve
        rho_mod = df_site[f"AppR{c}Cal"].values
        mask = rho_mod > 0
        if mask.any():
            fig.plot(x=period[mask], y=rho_mod[mask],
                     region=RHO_REGION, projection=rho_proj,
                     frame=frame_todo, pen=f"{LINE_PEN},{col}")
            frame_todo = None

        # observed data + error bars
        rho_obs = df_site[f"AppR{c}Obs"].values
        err_rho = df_site[f"AppR{c}Err"].values
        valid = rho_obs > 0
        if valid.any():
            fig.plot(x=period[valid], y=rho_obs[valid],
                     region=RHO_REGION, projection=rho_proj,
                     style=f"c{SYM_SIZE}", pen=f"{LINE_PEN},{col}",
                     no_clip=True)
            fig.plot(data=pd.DataFrame({"x": period[valid],
                                        "y": rho_obs[valid],
                                        "ey": err_rho[valid]}),
                     region=RHO_REGION, projection=rho_proj,
                     style=f"ey{ERR_CAP}/0.3p", pen=f"{LINE_PEN},{col}",
                     no_clip=True)

    if frame_todo is not None:   # nothing was plotted – draw empty frame
        fig.basemap(region=RHO_REGION, projection=rho_proj, frame=frame_todo)

    # site label inside the rho panel
    fig.text(x=period.min(), y=RHO_REGION[3] * 0.6,
             text=label,
             region=RHO_REGION, projection=rho_proj,
             justify="TL", font="8p,Helvetica,black",
             no_clip=True, offset="0.08c/-0.05c")

    # --------------------------------------------------------------- phase --
    fig.shift_origin(yshift=f"-{RHO_TO_PHS}c")
    frame_todo = _phs_frame(show_left, show_bottom)

    for c in COMPS:
        col = COLOR[c]

        # model curve  (sign flip: original script uses -$phi)
        phs_mod = -df_site[f"Phs{c}Cal"].values
        fig.plot(x=period, y=phs_mod,
                 region=PHS_REGION, projection=phs_proj,
                 frame=frame_todo, pen=f"{LINE_PEN},{col}")
        frame_todo = None

        # observed data + error bars
        phs_obs = -df_site[f"Phs{c}Obs"].values
        err_phs = df_site[f"Phs{c}Err"].values
        fig.plot(x=period, y=phs_obs,
                 region=PHS_REGION, projection=phs_proj,
                 style=f"c{SYM_SIZE}", pen=f"{LINE_PEN},{col}",
                 no_clip=True)
        fig.plot(data=pd.DataFrame({"x": period,
                                    "y": phs_obs,
                                    "ey": err_phs}),
                 region=PHS_REGION, projection=phs_proj,
                 style=f"ey{ERR_CAP}/0.3p", pen=f"{LINE_PEN},{col}",
                 no_clip=True)

    # restore origin to rho-panel top so the caller can shift to next panel
    fig.shift_origin(yshift=f"{RHO_TO_PHS}c")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot MT sounding curves from femtic_data.py output.")
    p.add_argument("--result",   default="result_MT.txt",
                   help="Result file  (default: result_MT.txt)")
    p.add_argument("--sites",    default=None,
                   help="Optional two-column site-list file "
                        "(site_name  label).  "
                        "If omitted all sites are plotted in file order.")
    p.add_argument("--out",      default="imp_all_curv",
                   help="Output PDF prefix  (default: imp_all_curv)")
    p.add_argument("--per-page", dest="per_page", type=int, default=12,
                   help="Stations per page  (default: 12)")
    p.add_argument("--cols",     type=int, default=6,
                   help="Columns per page   (default: 6)")
    p.add_argument("--appphs",   action="store_true",
                   help="Force AppRes/Phase column interpretation")
    p.add_argument("--impedance", action="store_true",
                   help="Force Re/Im impedance column interpretation")
    return p.parse_args()


def main():
    args = parse_args()

    # --- detect / override input mode ---
    if args.appphs and args.impedance:
        sys.exit("Error: --appphs and --impedance are mutually exclusive.")
    elif args.appphs:
        mode = "appphs"
    elif args.impedance:
        mode = "impedance"
    else:
        mode = detect_mode(args.result)
        print(f"Auto-detected input mode: {mode}")

    # --- load & optionally convert ---
    df = load_result(args.result, mode)
    if mode == "impedance":
        df = impedance_to_appphs(df)

    # --- determine station order ---
    if args.sites:
        site_pairs = load_sites(args.sites)
    else:
        seen: dict = {}
        for name in df["Site"]:
            if name not in seen:
                seen[name] = name
        site_pairs = list(seen.items())

    n_stations = len(site_pairs)
    per_page   = args.per_page
    n_cols     = args.cols
    n_rows     = math.ceil(per_page / n_cols)

    pygmt.config(FONT_TITLE="14p", FONT_LABEL="12p", MAP_TICK_LENGTH="0.1c")

    fig          = None
    current_page = -1

    for j, (site_name, label) in enumerate(site_pairs):
        page_idx    = j // per_page
        idx_on_page = j % per_page
        col_idx     = idx_on_page % n_cols
        row_idx     = idx_on_page // n_cols

        print(f"Station {j+1}/{n_stations}: site={site_name!r} "
              f"label={label!r} page={page_idx+1} "
              f"row={row_idx} col={col_idx}")

        # ---- new page ----
        if page_idx != current_page:
            if fig is not None:
                out_path = f"{args.out}_{current_page + 1}.pdf"
                fig.savefig(out_path, dpi=300)
                print(f"  Saved {out_path}")
            fig = pygmt.Figure()
            current_page = page_idx
            # Position origin at top-left corner of first panel
            fig.shift_origin(xshift="3c",
                             yshift=f"{3.0 + n_rows * PANEL_DY}c")

        # ---- shift to this panel's position ----
        if idx_on_page == 0:
            pass                          # already at top-left
        elif col_idx == 0:
            # start of a new row: move left and down
            fig.shift_origin(
                xshift=f"-{(n_cols - 1) * PANEL_DX}c",
                yshift=f"-{PANEL_DY}c",
            )
        else:
            fig.shift_origin(xshift=f"{PANEL_DX}c")

        # ---- filter & sort data ----
        df_site = (df[df["Site"] == site_name]
                   .sort_values("Frequency")
                   .reset_index(drop=True))

        if df_site.empty:
            print(f"  WARNING: no data for site '{site_name}', skipping.")
            continue

        # ---- draw ----
        plot_station(fig, df_site, label,
                     show_left=(col_idx == 0),
                     show_bottom=(row_idx == n_rows - 1))

    # ---- save last page ----
    if fig is not None:
        out_path = f"{args.out}_{current_page + 1}.pdf"
        fig.savefig(out_path, dpi=300)
        print(f"  Saved {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
