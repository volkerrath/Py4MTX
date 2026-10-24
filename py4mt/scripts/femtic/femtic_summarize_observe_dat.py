#!/usr/bin/env python3
"""femtic_summarize_observe_dat.py

Summarise the data content of one or more FEMTIC ``observe.dat`` files:
number of sites, frequencies per site, data values per frequency, and
overall data-vector size, broken down by observation type (MT, VTF, PT).

Usage
-----
Edit the USER SECTION below, or pass paths on the command line (they
replace PATHS):

    python femtic_summarize_observe_dat.py observe.dat
    python femtic_summarize_observe_dat.py ensemble_*/observe.dat
    python femtic_summarize_observe_dat.py .          # scan current directory
    python femtic_summarize_observe_dat.py            # uses PATHS

The Py4MTX environment must be set: PY4MTX_ROOT and PY4MTX_DATA are read
from the environment, and <PY4MTX_ROOT>/py4mt/modules/ and
<PY4MTX_ROOT>/py4mt/scripts/ are put on sys.path (version.py and util.py
supply the title line printed at start-up).

Optional LaTeX table: set LATEX_FILE in the USER SECTION to the name of a
.tex file. It gets one booktabs table with, for every file, one row per
observation type and a total row. LATEX_STANDALONE = True writes a complete
document instead of only the table environment.

When ``femtic.py`` is importable (normally the case once the Py4MTX paths
are set) the file is parsed by :func:`femtic.summarise_observe_dat`; the
table itself is always printed by this script, in plain ASCII. Otherwise
the script falls back to a self-contained parser that replicates the same
logic without any external dependencies.

Output columns
--------------
For each observation-type block:

    Obs type  -- MT | VTF | PT
    Sites     -- number of sites in this block
    d/freq    -- data values per frequency per site (MT=8, VTF=4, PT=4)
    Freq tot  -- total frequency rows across all sites in this block
    Data tot  -- Freq tot x d/freq  (contribution to the data vector)

Totals across all blocks are shown at the bottom. Dummy stations (0
frequencies) are reported on an extra line.

AI-generated code -- author: Claude, Anthropic.
Date generated: 2026-06-10.
Review before use in production.

Provenance
----------
    2026-06-10  Claude Sonnet 4.6 (Anthropic)   Created.
    2026-06-12  Claude Sonnet 4.6 (Anthropic)   Replaced Unicode box chars with plain ASCII.
    2026-08-13  Claude Sonnet 5 (Anthropic)     Added femtic_summarize_observe_dat_summary.md
                output: writes CLI parameters, script path, and run date/time via a
                self-contained _write_param_summary() helper (no femtic/util dependency).
    2026-10-24  Claude Sonnet 5.5 (Anthropic)
                Added the usual Py4MTX header (PY4MTX_DATA / PY4MTX_ROOT from
                the environment, py4mt/modules and py4mt/scripts on sys.path,
                version/util title line). Because femtic.py is now found, the
                table is no longer printed by femtic._print_observe_summary
                (it uses Unicode box characters) but always by this script, in
                plain ASCII; parsing is still delegated to femtic.py. Added a
                USER SECTION (PATHS); command-line paths still replace it.
                The fallback message now says why femtic.py could not be
                imported (femtic.py itself needs ensembles.py on the path).
                Added the AI-generated notice above.
    2026-10-24  Claude Sonnet 5.5 (Anthropic)
                Added the optional LaTeX table (LATEX_FILE, LATEX_CAPTION,
                LATEX_LABEL, LATEX_STANDALONE in the USER SECTION; builder
                latex_table()). The printed summary is unchanged.
"""
from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Py4MTX-specific settings and imports
# ---------------------------------------------------------------------------
PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

mypath = [PY4MTX_ROOT + "/py4mt/modules/", PY4MTX_ROOT + "/py4mt/scripts/"]
for pth in mypath:
    if pth not in sys.path:
        sys.path.insert(0, pth)

from version import versionstrg
import util as utl

version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())
titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng + "\n\n")


# ===========================================================================
# USER SECTION -- edit here; every user parameter is in CAPITALS
# ===========================================================================

# observe.dat files, glob patterns, or directories to scan (a directory is
# searched for a file literally named observe.dat). Command-line arguments,
# if given, replace this list.
PATHS = ["."]

# Optional LaTeX table of the summaries. None = off; otherwise the name of
# the .tex file to write (a relative path refers to the current directory).
LATEX_FILE = None

# Caption and label of the table, used as they are (write LaTeX, e.g.
# r"Data-vector size of the ensemble."). None = default caption / no label.
LATEX_CAPTION = None
LATEX_LABEL = None

# False = only the table environment, to be \input into a paper (needs the
# booktabs package); True = a complete document that can be compiled as is.
LATEX_STANDALONE = False

# ===========================================================================
# END OF USER SECTION
# ===========================================================================


def _write_param_summary(fname: str, params: dict) -> str:
    """Write a Markdown summary of this script's user parameters.

    Self-contained (stdlib only). Writes
    ``<script_dir>/<script_stem>_summary.md`` with the script path, run
    date/time, and the resolved user parameters.
    """
    from datetime import datetime

    script_path = os.path.abspath(fname)
    stem = os.path.splitext(os.path.basename(fname))[0]
    out_path = os.path.join(os.path.dirname(script_path), stem + "_summary.md")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Parameter summary: " + stem, "",
        "- **Script path:** `" + script_path + "`",
        "- **Run date/time:** " + now, "",
        "## User-set parameters", "",
        "| Parameter | Value |", "|---|---|",
    ]
    for k in sorted(params):
        vstr = repr(params[k]).replace("|", "\\|").replace("\n", " ")
        lines.append("| `" + k + "` | `" + vstr + "` |")

    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("[write_param_summary] Wrote parameter summary to " + out_path)
    return out_path


# ---------------------------------------------------------------------------
# Observation-type metadata (mirrors femtic.py constants)
# ---------------------------------------------------------------------------

_OBS_DATALEN: dict[str, int] = {"MT": 8, "VTF": 4, "PT": 4}
_OBS_TYPES = set(_OBS_DATALEN)


# ---------------------------------------------------------------------------
# Try to import from femtic.py; fall back to self-contained implementation
# ---------------------------------------------------------------------------

try:
    import femtic as _fem
    _FEMTIC_AVAILABLE = True
except ImportError as _exc:
    _FEMTIC_AVAILABLE = False
    _FEMTIC_ERROR = str(_exc)

    # ------------------------------------------------------------------
    # Self-contained fallback -- minimal observe.dat parser
    # ------------------------------------------------------------------

    def _is_block_header(tokens: list[str]) -> bool:
        if len(tokens) != 2:
            return False
        if tokens[0] not in _OBS_TYPES:
            return False
        try:
            int(tokens[1])
        except Exception:
            return False
        return True

    def _find_end_index(lines: list[str]) -> int:
        for i, line in enumerate(lines):
            toks = line.split()
            if toks and toks[0].upper() == "END":
                return i
        return len(lines)

    def _parse_observe_dat(path: Path) -> dict:
        """Minimal observe.dat parser -- returns same structure as femtic.read_observe_dat."""
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        if not lines:
            raise ValueError(f"Empty file: {path}")

        end_idx = _find_end_index(lines)
        block_starts: list[int] = []
        for i in range(end_idx):
            toks = lines[i].split()
            if _is_block_header(toks):
                block_starts.append(i)

        if not block_starts:
            raise ValueError(f"No data blocks detected in {path}.")

        block_starts.append(end_idx)
        blocks: list[dict] = []

        for bi in range(len(block_starts) - 1):
            start = block_starts[bi]
            stop  = block_starts[bi + 1]
            block_lines = lines[start:stop]

            hdr_toks = block_lines[0].split()
            obs_type  = hdr_toks[0]
            dat_len   = _OBS_DATALEN.get(obs_type, 0)

            sites: list[dict] = []
            li = 1
            while li < len(block_lines):
                toks = block_lines[li].split()
                if not toks:
                    li += 1
                    continue
                if len(toks) != 4:
                    li += 1
                    continue
                # site header -- next line is nfreq
                try:
                    nfreq = int(block_lines[li + 1].split()[0])
                except Exception:
                    li += 1
                    continue
                sites.append({"nfreq": nfreq})
                li = li + 2 + nfreq   # header + nfreq-line + data rows

            blocks.append({"obs_type": obs_type, "sites": sites, "dat_length": dat_len})

        return {"path": str(path), "blocks": blocks}

    def _fallback_summarise(path: Path) -> dict:
        parsed = _parse_observe_dat(path)

        block_summaries: list[dict] = []
        for blk in parsed["blocks"]:
            obs_type = blk["obs_type"]
            dat_len  = blk.get("dat_length", _OBS_DATALEN.get(obs_type, 0))
            nfreqs   = [s["nfreq"] for s in blk["sites"]]
            n_freq_total = sum(nfreqs)
            block_summaries.append({
                "obs_type":        obs_type,
                "n_sites":         len(nfreqs),
                "n_sites_empty":   sum(1 for n in nfreqs if n == 0),
                "dat_length":      dat_len,
                "n_freq_per_site": nfreqs,
                "n_freq_total":    n_freq_total,
                "n_data_total":    n_freq_total * dat_len,
            })

        return {
            "path":          str(path),
            "blocks":        block_summaries,
            "n_sites_total": sum(b["n_sites"]      for b in block_summaries),
            "n_freq_total":  sum(b["n_freq_total"] for b in block_summaries),
            "n_data_total":  sum(b["n_data_total"] for b in block_summaries),
        }


# ---------------------------------------------------------------------------
# Summary and ASCII table (used for both the femtic and the fallback parser)
# ---------------------------------------------------------------------------

def _print_summ(s: dict) -> None:
    SEP  = "  " + "-" * 52
    DSEP = "  " + "=" * 52
    fname = Path(s["path"]).name

    print(f"\n{SEP}")
    print(f"  File          : {fname}")
    print(SEP)
    print(f"  {'Obs type':<10}  {'Sites':>6}  {'d/freq':>6}  {'Freq tot':>9}  {'Data tot':>9}")
    print(SEP)
    for b in s["blocks"]:
        nfreqs = b["n_freq_per_site"]
        print(
            f"  {b['obs_type']:<10}  {b['n_sites']:>6d}  "
            f"{b['dat_length']:>6d}  {b['n_freq_total']:>9d}  "
            f"{b['n_data_total']:>9d}"
        )
        if nfreqs and len(set(nfreqs)) > 1:
            print(f"  {'':10}  nfreq range: {min(nfreqs)}-{max(nfreqs)}")
    print(DSEP)
    print(
        f"  {'Total':<10}  {s['n_sites_total']:>6d}  {'':>6}  "
        f"{s['n_freq_total']:>9d}  {s['n_data_total']:>9d}"
    )
    print(SEP)
    empty = [(b["obs_type"], b.get("n_sites_empty", 0))
             for b in s["blocks"] if b.get("n_sites_empty", 0)]
    if empty:
        txt = ", ".join(f"{t} {n}" for t, n in empty)
        print(f"  Dummy sites   : {txt} (0 frequencies)")


def _summarise(path: Path, *, out: bool = True) -> dict:
    """Summarise one observe.dat; print the ASCII table if *out*."""
    if _FEMTIC_AVAILABLE:
        s = _fem.summarise_observe_dat(path, out=False)
    else:
        s = _fallback_summarise(path)
    if out:
        _print_summ(s)
    return s


# ---------------------------------------------------------------------------
# LaTeX table (optional: LATEX_FILE)
# ---------------------------------------------------------------------------

_TEX_SPECIAL = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _tex_escape(text) -> str:
    """Escape the LaTeX special characters of *text*."""
    return "".join(_TEX_SPECIAL.get(c, c) for c in str(text))


def _tex_int(n: int) -> str:
    """Integer with a thin space as thousands separator (no package needed)."""
    return f"{n:,d}".replace(",", r"\,")


def latex_table(summaries: list[dict], caption: str | None = None,
                label: str | None = None, standalone: bool = False) -> str:
    """LaTeX source (booktabs) of the summaries: one table for all files.

    *summaries* are the dicts returned by ``summarise_observe_dat``. For each
    file the table has a line with the path, one row per observation type, a
    total row and, if there are dummy stations, a footnote line. *caption*
    and *label* are used as they are (they are LaTeX). With *standalone* the
    result is a complete document, otherwise only the table environment.
    """
    from datetime import datetime

    ncol = 5
    if caption is None:
        caption = (r"Data content of the FEMTIC observation files: sites, "
                   r"frequency rows, data values per frequency and site "
                   r"(MT 8, VTF 4, PT 4), and the resulting length of the "
                   r"data vector (frequency rows $\times$ data values per "
                   r"frequency).")

    body = [r"\begin{table}[htbp]", r"  \centering", r"  \small",
            r"  \caption{" + caption + "}"]
    if label:
        body.append(r"  \label{" + label + "}")
    body += [r"  \begin{tabular}{lrrrr}", r"    \toprule",
             r"    Obs type & Sites & d/freq & Freq.\ total & Data total \\",
             r"    \midrule"]
    for k, s in enumerate(summaries):
        if k:
            body.append(r"    \midrule")
        body.append(r"    \multicolumn{" + str(ncol) + r"}{l}{\texttt{"
                    + _tex_escape(s["path"]) + r"}} \\")
        for b in s["blocks"]:
            body.append("    " + " & ".join([
                _tex_escape(b["obs_type"]), _tex_int(b["n_sites"]),
                str(b["dat_length"]), _tex_int(b["n_freq_total"]),
                _tex_int(b["n_data_total"])]) + r" \\")
        body.append("    " + " & ".join([
            r"\textit{Total}", _tex_int(s["n_sites_total"]), "",
            _tex_int(s["n_freq_total"]), _tex_int(s["n_data_total"])]) + r" \\")
        empty = [(b["obs_type"], b.get("n_sites_empty", 0))
                 for b in s["blocks"] if b.get("n_sites_empty", 0)]
        if empty:
            txt = ", ".join(f"{_tex_escape(t)} {n}" for t, n in empty)
            body.append(r"    \multicolumn{" + str(ncol) + r"}{l}{\footnotesize "
                        r"Dummy stations (0 frequencies): " + txt + r"} \\")
    body += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    note = ["% Generated by femtic_summarize_observe_dat.py on " + stamp]
    if standalone:
        out = [r"\documentclass{article}",
               r"\usepackage[a4paper,margin=1.5cm]{geometry}",
               r"\usepackage{booktabs}", r"\begin{document}", ""] + note + [""] \
              + body + ["", r"\end{document}"]
    else:
        out = note + [r"% Needs \usepackage{booktabs} in the preamble."] + body
    return "\n".join(out) + "\n"


def _write_latex(text: str, path: str) -> None:
    """Write the LaTeX source; a failure is reported but does not stop the run."""
    try:
        Path(path).write_text(text, encoding="utf-8")
    except OSError as exc:
        print(f"[error] cannot write LaTeX table to {path}: {exc}", file=sys.stderr)
        return
    print(f"[latex] Wrote LaTeX table to {Path(path).resolve()}")


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def collect_paths(args: list[str]) -> list[Path]:
    paths: list[Path] = []
    for a in args:
        p = Path(a)
        if p.is_dir():
            found = sorted(p.glob("observe.dat"))
            if not found:
                print(f"[warn] No observe.dat found in {p}", file=sys.stderr)
            paths.extend(found)
        elif "*" in a or "?" in a:
            paths.extend(sorted(Path(".").glob(a)))
        else:
            paths.append(p)
    return paths


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description="Summarise data statistics of FEMTIC observe.dat files."
    )
    ap.add_argument(
        "paths", nargs="*", default=list(PATHS),
        metavar="PATH",
        help="observe.dat files, glob patterns, or directories "
             "(default: PATHS in the user section of the script).",
    )
    args = ap.parse_args()

    _write_param_summary(__file__, {
        "PATHS":            args.paths,
        "LATEX_FILE":       LATEX_FILE,
        "LATEX_CAPTION":    LATEX_CAPTION,
        "LATEX_LABEL":      LATEX_LABEL,
        "LATEX_STANDALONE": LATEX_STANDALONE,
    })

    if _FEMTIC_AVAILABLE:
        print("[info] Using femtic.summarise_observe_dat (ASCII table from this script)",
              file=sys.stderr)
    else:
        print(f"[info] femtic.py not importable ({_FEMTIC_ERROR}) "
              f"-- using built-in fallback parser", file=sys.stderr)

    paths = collect_paths(args.paths)
    if not paths:
        print("No observe.dat files found.  Pass paths or a directory.", file=sys.stderr)
        sys.exit(1)

    n_ok = 0
    summaries: list[dict] = []
    for p in paths:
        if not p.exists():
            print(f"[warn] File not found: {p}", file=sys.stderr)
            continue
        try:
            summaries.append(_summarise(p, out=True))
            n_ok += 1
        except Exception as exc:
            print(f"[error] {p.name}: {exc}", file=sys.stderr)

    print(f"\n  {n_ok}/{len(paths)} file(s) processed.")

    if LATEX_FILE is not None:
        if summaries:
            _write_latex(latex_table(summaries, LATEX_CAPTION, LATEX_LABEL,
                                     LATEX_STANDALONE), LATEX_FILE)
        else:
            print("[warn] no file summarised -- no LaTeX table written",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
