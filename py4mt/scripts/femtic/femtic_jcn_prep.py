#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare directories and data subsets for a jackknife-inspired uncertainty analysis.

Creates sub-directories with template files and generates reduced data sets
(e.g., leave-one-site-out) for FEMTIC inversion runs.

@author:   vrath
@project:  py4mt — Python for Magnetotellurics
@inversion: FEMTIC

Provenance:
    2026-03-03  Claude          Renamed user-set parameters to UPPERCASE;
                                 generated README.
    2026-07-25  Claude Sonnet 5 (Anthropic)
                Added RANDOM_SEED (default None) for optional reproducible
                runs, matching the pattern used in femtic_rto_prep.py /
                femtic_gst_prep.py / femtic_nss.py: rng =
                np.random.default_rng(RANDOM_SEED), resolved seed echoed
                to the console. rng is threaded into the generate_data_fcn
                call for when random "subset" mode is used.
                Flagged (did not fix) a pre-existing, unrelated issue:
                fem.generate_directories() and fem.generate_data_fcn() do
                not exist in the current femtic.py / ensembles.py — this
                script predates the consolidation of directory/data-
                ensemble generation into ensembles.py and will raise
                AttributeError as written. See the KNOWN ISSUE comment
                block near RANDOM_SEED.
    2026-08-13  Claude Sonnet 5 (Anthropic)
                Added femtic_jcn_prep_summary.md output at end of run:
                writes user-set (UPPERCASE) parameters, script path, and
                run date/time via utl.write_param_summary().
    2026-10-11  Claude Sonnet 5.5 (Anthropic)
                Migrated to the current module layout; the script no longer
                raises AttributeError.
                * fem.generate_directories -> ens.generate_directories
                  (ensembles.py; keywords copy_list/link_list/n_samples).
                  observe.dat is *copied* into each member (it is rewritten
                  below); control.dat, mesh.dat and the starting model /
                  distortion files are symlinked, as in femtic_rto_prep.py.
                * fem.generate_data_fcn did not exist anywhere. Replaced by
                  the local function make_jackknife_data(), built on
                  fem.read_observe_dat / fem.write_observe_dat:
                  "site"   -> member i drops the i-th unique site name from
                              every data block (leave-one-site-out);
                  "subset" -> member i keeps SUBSET_SIZE randomly drawn
                              sites (uses rng / RANDOM_SEED).
                  Block header counts are updated; the untouched file is
                  kept as observe_orig.dat in each member directory.
                * N_SAMPLES in "site" mode is now taken from observe.dat
                  (number of unique sites) instead of line 1 of control.dat;
                  a difference to control.dat is reported, not fatal.
                * Fixed dir_base mismatch (directories were created as
                  "jcn_<i>" but the data call used "ens_<i>").
                * distortion_iter0.dat is NOT trimmed when sites are
                  dropped (see NOTE in the configuration block).
    2026-10-28  Claude Sonnet 5.5 (Anthropic)
                The parameter summary femtic_jcn_prep_summary.md is now
                written to the working directory set in this script
                (ENSEMBLE_DIR) instead of next to the script:
                utl.write_param_summary(fname, out_dir=ENSEMBLE_DIR).
                Needs the util.py of the same date (new out_dir argument).
                AI-generated; review before production use.
"""

import os
import sys
from pathlib import Path
import inspect

import numpy as np

PY4MTX_DATA = os.environ["PY4MTX_DATA"]
PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

for _base in [PY4MTX_ROOT + "/py4mt/modules/"]:
    for _p in [Path(_base), *Path(_base).rglob("*")]:
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

import femtic as fem
import ensembles as ens
import util as utl
from version import versionstrg

nan = np.nan
version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())
titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng + "\n\n")

# -----------------------------------------------------------------------
# Reproducibility (optional)
# -----------------------------------------------------------------------
# Set RANDOM_SEED to an integer for a reproducible run (relevant only if/
# when CHOICE_MODE uses random subset selection, e.g. ["subset", N] --
# leave-one-site-out ("site" mode) is deterministic and doesn't consume
# rng draws at all). None (default) uses fresh OS entropy.
RANDOM_SEED = None   # e.g. 20260725 for a reproducible run; None = fresh entropy

rng = np.random.default_rng(RANDOM_SEED)
print(f"RNG seed: {RANDOM_SEED if RANDOM_SEED is not None else '(fresh entropy — not reproducible)'}\n")

# =============================================================================
#  Configuration
# =============================================================================
ENSEMBLE_DIR = r"/home/vrath/work/Ensemble/Ubinas_ens/"
TEMPLATES = ENSEMBLE_DIR + "templates/"
FILES = [
    "control.dat",
    "observe.dat",
    "mesh.dat",
    "resistivity_block_iter0.dat",
    "distortion_iter0.dat",
    "run_femtic_dub.sh",
    "run_femtic_oar.sh",
]

# Files copied into each member (observe.dat is rewritten per member, so it
# must be a real copy) and files symlinked to the templates.
COPY_FILES = ["observe.dat", "run_femtic_dub.sh", "run_femtic_oar.sh"]
LINK_FILES = [f for f in FILES if f not in COPY_FILES]

# "site"   : leave-one-site-out; N_SAMPLES = number of unique sites.
# "subset" : N_SAMPLES members, each keeping SUBSET_SIZE random sites.
CHOICE_MODE = ["site"]
# CHOICE_MODE = ["subset"]
N_SAMPLES = 32        # used in "subset" mode only
SUBSET_SIZE = 20      # used in "subset" mode only

DIR_BASE = ENSEMBLE_DIR + "jcn_"

# NOTE: distortion_iter0.dat holds one entry per site. It is symlinked
# unchanged; if distortion is estimated and sites are dropped, trim or
# disable it for the members yourself.


def make_jackknife_data(dir_base, templates, n_members, choice_mode,
                        file_in="observe.dat", rng=None, out=True):
    """Write a reduced observe.dat into each member directory.

    Returns a list of (member_dir, kept_site_names).
    """
    rng = np.random.default_rng() if rng is None else rng
    src = templates + file_in
    full = fem.read_observe_dat(src, compute_mt_derived=False)

    names = []
    for blk in full["blocks"]:
        for st in blk["sites"]:
            nm = st["site_header_tokens"][0]
            if nm not in names:
                names.append(nm)
    n_all = len(names)

    mode = choice_mode[0]
    if mode == "site":
        if n_members != n_all:
            raise ValueError(f"site mode needs n_members == number of sites "
                             f"({n_all}), got {n_members}")
        keeps = [[n for n in names if n != names[i]] for i in range(n_all)]
    elif mode == "subset":
        k = int(SUBSET_SIZE)
        if not 0 < k <= n_all:
            raise ValueError(f"SUBSET_SIZE={k} must be in 1..{n_all}")
        keeps = []
        for _ in range(n_members):
            pick = set(rng.choice(n_all, size=k, replace=False).tolist())
            keeps.append([n for j, n in enumerate(names) if j in pick])
    else:
        raise ValueError(f"unknown CHOICE_MODE {mode!r}")

    res = []
    for i, keep in enumerate(keeps):
        d = f"{dir_base}{i}/"
        dst = d + file_in
        orig = d + file_in.replace(".dat", "_orig.dat")
        if os.path.exists(dst) and not os.path.exists(orig):
            os.replace(dst, orig)
            import shutil
            shutil.copy2(orig, dst)
        keepset = set(keep)
        red = fem.read_observe_dat(src, compute_mt_derived=False)
        for blk in red["blocks"]:
            blk["sites"] = [s for s in blk["sites"]
                            if s["site_header_tokens"][0] in keepset]
            n = len(blk["sites"])
            toks = blk["header_line"].split()
            blk["n_sites_header"] = n
            blk["header_line"] = f"{toks[0]}    {n}\n"
        fem.write_observe_dat(red, dst)
        res.append((d, keep))
        if out:
            print(f"  {d}: {len(keep)}/{n_all} sites"
                  + (f", dropped {names[i]}" if mode == "site" else ""))
    return res


# =============================================================================
#  Number of members
# =============================================================================
if "site" in CHOICE_MODE:
    _obs = fem.read_observe_dat(TEMPLATES + "observe.dat", compute_mt_derived=False)
    _names = []
    for _b in _obs["blocks"]:
        for _s in _b["sites"]:
            if _s["site_header_tokens"][0] not in _names:
                _names.append(_s["site_header_tokens"][0])
    N_SAMPLES = len(_names)
    try:
        with open(TEMPLATES + "control.dat", "r") as _f:
            _n_ctl = int(_f.readline().split()[0])
        if _n_ctl != N_SAMPLES:
            print(f"WARNING: control.dat line 1 says {_n_ctl}, observe.dat has "
                  f"{N_SAMPLES} unique sites; using {N_SAMPLES}.")
    except (OSError, ValueError, IndexError):
        pass

# =============================================================================
#  Generate directories
# =============================================================================
os.chdir(ENSEMBLE_DIR)

dir_list = ens.generate_directories(
    dir_base=DIR_BASE,
    templates=TEMPLATES,
    copy_list=COPY_FILES,
    link_list=LINK_FILES,
    n_samples=N_SAMPLES,
    out=True,
)

# =============================================================================
#  Draw reduced data sets based on sites
# =============================================================================
data_ensemble = make_jackknife_data(
    dir_base=DIR_BASE,
    templates=TEMPLATES,
    n_members=N_SAMPLES,
    choice_mode=CHOICE_MODE,
    file_in="observe.dat",
    rng=rng,
    out=True,
)


# ---------------------------------------------------------------------------
# Parameter summary
# ---------------------------------------------------------------------------
utl.write_param_summary(fname, out_dir=ENSEMBLE_DIR)
