#!/usr/bin/env python3
"""femtic_summarize_observe_dat.py

Summarise the data content of one or more FEMTIC ``observe.dat`` files:
number of sites, frequencies per site, data values per frequency, and
overall data-vector size, broken down by observation type (MT, VTF, PT).

Usage
-----
    python femtic_summarize_observe_dat.py observe.dat
    python femtic_summarize_observe_dat.py ensemble_*/observe.dat
    python femtic_summarize_observe_dat.py .          # scan current directory

When ``femtic.py`` is importable the script delegates to
:func:`femtic.summarise_observe_dat` and :func:`femtic._print_observe_summary`.
Otherwise it falls back to a self-contained parser that replicates the same
logic without any external dependencies.

Output columns
--------------
For each observation-type block:

    Obs type  -- MT | VTF | PT
    Sites     -- number of sites in this block
    d/freq    -- data values per frequency per site (MT=8, VTF=4, PT=4)
    Freq tot  -- total frequency rows across all sites in this block
    Data tot  -- Freq tot x d/freq  (contribution to the data vector)

Totals across all blocks are shown at the bottom.

Provenance
----------
    2026-06-10  Claude Sonnet 4.6 (Anthropic)   Created.
    2026-06-12  Claude Sonnet 4.6 (Anthropic)   Replaced Unicode box chars with plain ASCII.
"""
from __future__ import annotations

import sys
from pathlib import Path


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
    _summarise  = _fem.summarise_observe_dat
    _print_summ = _fem._print_observe_summary
    _FEMTIC_AVAILABLE = True
except ImportError:
    _FEMTIC_AVAILABLE = False

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

    def _summarise(path_or_parsed, *, out=True) -> dict:
        if isinstance(path_or_parsed, dict):
            parsed   = path_or_parsed
            src_path = parsed.get("path", "<in-memory>")
        else:
            parsed   = _parse_observe_dat(Path(path_or_parsed))
            src_path = str(path_or_parsed)

        block_summaries: list[dict] = []
        for blk in parsed["blocks"]:
            obs_type = blk["obs_type"]
            dat_len  = blk.get("dat_length", _OBS_DATALEN.get(obs_type, 0))
            sites    = blk["sites"]
            nfreqs   = [s["nfreq"] for s in sites]
            n_freq_total = sum(nfreqs)
            block_summaries.append({
                "obs_type":        obs_type,
                "n_sites":         len(sites),
                "dat_length":      dat_len,
                "n_freq_per_site": nfreqs,
                "n_freq_total":    n_freq_total,
                "n_data_total":    n_freq_total * dat_len,
            })

        n_sites_total = sum(b["n_sites"]      for b in block_summaries)
        n_freq_total  = sum(b["n_freq_total"] for b in block_summaries)
        n_data_total  = sum(b["n_data_total"] for b in block_summaries)

        s = {
            "path":          src_path,
            "blocks":        block_summaries,
            "n_sites_total": n_sites_total,
            "n_freq_total":  n_freq_total,
            "n_data_total":  n_data_total,
        }
        if out:
            _print_summ(s)
        return s

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
        "paths", nargs="*", default=["."],
        metavar="PATH",
        help="observe.dat files, glob patterns, or directories (default: current dir).",
    )
    args = ap.parse_args()

    if _FEMTIC_AVAILABLE:
        print("[info] Using femtic.summarise_observe_dat", file=sys.stderr)
    else:
        print("[info] femtic.py not found -- using built-in fallback parser", file=sys.stderr)

    paths = collect_paths(args.paths)
    if not paths:
        print("No observe.dat files found.  Pass paths or a directory.", file=sys.stderr)
        sys.exit(1)

    n_ok = 0
    for p in paths:
        if not p.exists():
            print(f"[warn] File not found: {p}", file=sys.stderr)
            continue
        try:
            _summarise(p, out=True)
            n_ok += 1
        except Exception as exc:
            print(f"[error] {p.name}: {exc}", file=sys.stderr)

    print(f"\n  {n_ok}/{len(paths)} file(s) processed.")


if __name__ == "__main__":
    main()
