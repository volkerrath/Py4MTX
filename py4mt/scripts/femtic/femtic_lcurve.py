#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare the run directories for an L-curve study with FEMTIC.

One directory is created per trade-off parameter (alpha) in TRADE_OFF_LIST.
Each directory gets its own copy of control.dat (the only file in COPY_LIST),
in which the value of the TRADE_OFF_PARAMETER keyword is replaced by the
alpha of that run.  All other input files (mesh, data, starting model, ...)
are symlinked to the templates, exactly as in femtic_rto_prep.py.

The directories are named <ENSEMBLE_NAME><alpha>, e.g. lc_0.3, so that
femtic_lcurve_plot.py (SEARCH_STRNG = "lc_*") finds them after the runs.

Workflow:
    1. femtic_lcurve.py       (this script)  -> lc_<alpha>/ directories
    2. run FEMTIC in every lc_<alpha>/       (fixed alpha, INV_METHOD 1)
    3. femtic_lcurve_plot.py                 -> roughness vs. nRMS curve

Edit control.dat in the templates directory for everything that is common
to all runs (ITERATION, STEP_LENGTH, DISTORTION, ...); only the trade-off
parameter is changed here.

control.dat handling (matches the FEMTIC-DABIC v2.7.1 rules):
    * keywords and values are whitespace-separated tokens, a value may be
      on the same line as its keyword or on a later one;
    * a comment is a WHOLE line starting with '#'; comment lines that
      mention TRADE_OFF_PARAMETER, and the different keyword
      TYPE_OF_TRADE_OFF_PARAMETER, are never touched;
    * parsing stops at END;
    * only the first value token after the keyword is replaced; further
      tokens of that value (older "alpha factor" layouts) and everything
      else in the file are kept byte for byte, including line endings.
Both TRADE_OFF_PARAMETER (v2.7.1) and the older TRADE_OFF_PARAM are
recognised (TRADE_OFF_KEYS, first one found wins).

@author:   Claude (Anthropic), for vrath
@project:  py4mt -- Python for Magnetotellurics
@inversion: FEMTIC

Provenance:
    2026-11-17  Claude Sonnet 5.5 (Anthropic)
                Created (date generated: 2026-11-17).  Author: Claude,
                Anthropic.  Modelled on femtic_rto_prep.py (header, USER
                SECTION, COPY_LIST / LINK_LIST via ens.put_files,
                RELATIVE_LINKS, utl.write_param_summary).
                AI-generated code; review before production use.
"""

import os
import sys
import re
import stat
import hashlib
from pathlib import Path
import inspect

import numpy as np

PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

for _base in [PY4MTX_ROOT + "/py4mt/modules/"]:
    for _p in [Path(_base), *Path(_base).rglob("*")]:
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

# femtic must be imported before ensembles (the two import each other); fem
# itself is not used in this script.
import femtic as fem
import ensembles as ens
import util as utl
from version import versionstrg

version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())
titstrng = utl.print_title(version=version, fname=fname, out=False)
print(titstrng + "\n\n")

# =============================================================================
# USER SECTION -- all user-set parameters below are UPPERCASE
# =============================================================================

#: Directory holding the templates/ sub-directory; the run directories are
#: created here.  femtic_lcurve_plot.py uses the same directory as WORK_DIR.
ENSEMBLE_DIR = r"/media/vrath/LargeBack/Ensembles/misti2026/l-curve/"
#: Run directories are <ENSEMBLE_DIR><ENSEMBLE_NAME><alpha>/ .  Keep "lc_"
#: for femtic_lcurve_plot.py (SEARCH_STRNG = "lc_*").
ENSEMBLE_NAME = "lc_"
TEMPLATES = ENSEMBLE_DIR + "templates/"

#: Trade-off parameters (alpha) of the L-curve runs, one run each.  Any
#: sequence of positive numbers; log-spaced values are typical, e.g.
#: [0.01, 0.1, 1., 10., 100.] or list(np.logspace(-2, 2, 9)).  The values
#: are written into control.dat with up to 10 significant digits and the
#: directory name uses 6 (duplicates at that precision are refused).
TRADE_OFF_LIST = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]

#: Keyword(s) of the trade-off parameter in control.dat; the first one found
#: is used.  TRADE_OFF_PARAMETER = FEMTIC-DABIC v2.7.1, TRADE_OFF_PARAM =
#: older control files.
TRADE_OFF_KEYS = ["TRADE_OFF_PARAMETER", "TRADE_OFF_PARAM"]

#: The control file.  It is edited in every run directory, so it must be
#: *copied* (COPY_LIST), never linked -- a link would edit the template.
CONTROL_FILE = "control.dat"
COPY_LIST = [CONTROL_FILE]
#: Symlinked into every run directory.  Names missing in TEMPLATES are
#: reported and skipped (e.g. referencemodel.dat for a run without
#: reference model).
LINK_LIST = ["mesh.dat",
             "observe.dat",
             "resistivity_block_iter0.dat",
             "distortion_iter0.dat",
             "referencemodel.dat",
             "site.dat",
             "run_femtic_dias.sh", "run_femtic_kraken.sh"]
RELATIVE_LINKS = True   # True: portable relative symlinks (default, survives tgz);
                        # False: absolute symlinks

#: True: a run directory that already holds a convergence file (CNV_FILE) is
#: left untouched, so re-running the script with an extended TRADE_OFF_LIST
#: never rewrites the control.dat of finished runs.  False: rewrite all.
SKIP_FINISHED = True
CNV_FILE = ["femtic.cnv", "FEMTIC-DABIC.cnv"]

# =============================================================================
# END OF USER SECTION
# =============================================================================


_TOKEN = re.compile(r"\S+")


def _active_tokens(lines):
    """Tokens of a FEMTIC control file as the program reads them.

    Whole-line '#' comments are skipped, reading stops after END.
    Returns a list of (line_index, start, end, token).
    """
    toks = []
    for i, line in enumerate(lines):
        if line.lstrip().startswith("#"):
            continue
        for m in _TOKEN.finditer(line):
            toks.append((i, m.start(), m.end(), m.group(0)))
            if m.group(0) == "END":
                return toks
    return toks


def _value_index(toks, key):
    """Index in toks of the first value token after keyword key.

    None if the keyword is absent; ValueError if it appears more than once
    or has no value.
    """
    hits = [k for k, t in enumerate(toks) if t[3] == key]
    if not hits:
        return None
    if len(hits) > 1:
        raise ValueError("keyword %s appears %d times (each keyword may "
                         "appear once)" % (key, len(hits)))
    k = hits[0] + 1
    if k >= len(toks) or toks[k][3] == "END":
        raise ValueError("keyword %s has no value" % key)
    return k


def _value_of(lines, key):
    """First value token after key as a string, or None (absent/invalid)."""
    toks = _active_tokens(lines)
    try:
        k = _value_index(toks, key)
    except ValueError:
        return None
    return None if k is None else toks[k][3]


def locate_trade_off(lines, keys):
    """Find the trade-off value token.

    Returns (key, line_index, start, end, old_value_string).
    Raises ValueError if no key is found, a key is ambiguous, or the value
    is not a number.
    """
    toks = _active_tokens(lines)
    for key in keys:
        k = _value_index(toks, key)
        if k is None:
            continue
        i, a, b, old = toks[k]
        try:
            float(old)
        except ValueError:
            raise ValueError("value after %s is %r, not a number" % (key, old))
        return key, i, a, b, old
    raise ValueError("none of the keywords %s found" % list(keys))


def set_trade_off(path, value, keys):
    """Replace the trade-off value in control file path (in place).

    Only the first value token is replaced; the rest of the file, line
    endings included, is written back unchanged.
    Returns (key, old_string, new_string, line_number).
    """
    with open(path, "r", newline="") as f:
        lines = f.read().split("\n")
    key, i, a, b, old = locate_trade_off(lines, keys)
    new = format(float(value), ".10g")
    lines[i] = lines[i][:a] + new + lines[i][b:]
    with open(path, "w", newline="") as f:
        f.write("\n".join(lines))
    return key, old, new, i + 1


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _has_results(directory):
    names = [CNV_FILE] if isinstance(CNV_FILE, str) else list(CNV_FILE)
    return any(os.path.isfile(os.path.join(directory, n)) for n in names)


# =============================================================================
# Check the setup before anything is created
# =============================================================================
ens_dir = os.path.join(ENSEMBLE_DIR, "")
tpl_dir = os.path.join(TEMPLATES, "")
if not os.path.isdir(tpl_dir):
    sys.exit(" Directory: %s does not exist, needs to be copied !" % tpl_dir)

if CONTROL_FILE not in COPY_LIST or CONTROL_FILE in LINK_LIST:
    sys.exit(" %s must be in COPY_LIST and not in LINK_LIST: it is edited in "
             "every run directory, a link would edit the template."
             % CONTROL_FILE)

for _f in COPY_LIST:
    if not os.path.isfile(tpl_dir + _f):
        sys.exit(" COPY_LIST file %s does not exist in %s" % (_f, tpl_dir))

link_missing = [f for f in LINK_LIST if not os.path.exists(tpl_dir + f)]
link_list = [f for f in LINK_LIST if f not in link_missing]
if link_missing:
    print("WARNING: not in %s, not linked: %s\n" % (tpl_dir, link_missing))

tr_values = np.atleast_1d(np.asarray(TRADE_OFF_LIST, dtype=float)).ravel()
if tr_values.size == 0:
    sys.exit(" TRADE_OFF_LIST is empty.")
if not np.all(np.isfinite(tr_values)) or np.any(tr_values <= 0.0):
    sys.exit(" TRADE_OFF_LIST must contain finite values > 0, got %s"
             % tr_values.tolist())
tags = [format(v, "g") for v in tr_values]
dups = sorted({t for t in tags if tags.count(t) > 1})
if dups:
    sys.exit(" TRADE_OFF_LIST has values that give the same directory "
             "name: %s" % dups)

ctl_tpl = tpl_dir + CONTROL_FILE
with open(ctl_tpl, "r", newline="") as _fh:
    tpl_lines = _fh.read().split("\n")
try:
    _key, _i, _a, _b, _old = locate_trade_off(tpl_lines, TRADE_OFF_KEYS)
except ValueError as exc:
    sys.exit(" %s: %s" % (ctl_tpl, exc))
print("Template %s: %s = %s (line %d); set per run from TRADE_OFF_LIST.\n"
      % (CONTROL_FILE, _key, _old, _i + 1))

_inv = _value_of(tpl_lines, "INV_METHOD")
_typ = _value_of(tpl_lines, "TYPE_OF_TRADE_OFF_PARAMETER")
if (_inv is not None and _inv != "1") or (_typ is not None and _typ != "0"):
    print("WARNING: the template has INV_METHOD = %s, "
          "TYPE_OF_TRADE_OFF_PARAMETER = %s.  Separate runs for an L-curve\n"
          "         normally use a FIXED alpha (INV_METHOD 1, "
          "TYPE_OF_TRADE_OFF_PARAMETER 0).  Check that this is intended.\n"
          % (_inv, _typ))

# =============================================================================
# Create the run directories
# =============================================================================
tpl_hash = _sha256(ctl_tpl)

runs = []
for value, tag in zip(tr_values, tags):
    directory = "%s%s%s/" % (ens_dir, ENSEMBLE_NAME, tag)

    if SKIP_FINISHED and _has_results(directory):
        runs.append((value, directory, "skipped"))
        continue

    os.makedirs(directory, exist_ok=True)
    for _f in COPY_LIST:
        # a link left from an earlier setup must not be written through
        if os.path.islink(directory + _f):
            os.remove(directory + _f)
    ens.put_files(COPY_LIST, link_list, directory, tpl_dir,
                  relative_links=RELATIVE_LINKS)
    for _f in COPY_LIST:
        # copy2 keeps the mode of the template; a read-only template would
        # give read-only copies that can be neither edited here nor later
        _path = directory + _f
        os.chmod(_path, os.stat(_path).st_mode | stat.S_IWUSR)

    ctl = directory + CONTROL_FILE
    if os.path.islink(ctl):
        sys.exit(" %s is a link; refusing to edit it." % ctl)
    _key, _old, _new, _line = set_trade_off(ctl, value, TRADE_OFF_KEYS)
    runs.append((value, directory, "%s -> %s" % (_old, _new)))

if _sha256(ctl_tpl) != tpl_hash:
    sys.exit(" The template %s was changed -- this must not happen!" % ctl_tpl)

# =============================================================================
# Summary
# =============================================================================
print("=" * 78)
print("%4s  %-14s  %-18s  %s" % ("run", "alpha", "control.dat", "directory"))
print("-" * 78)
for n, (value, directory, status) in enumerate(runs):
    print("%4d  %-14s  %-18s  %s" % (n, format(value, "g"), status, directory))
print("=" * 78)
n_new = sum(1 for r in runs if r[2] != "skipped")
print("%d run directories written, %d skipped (finished)." %
      (n_new, len(runs) - n_new))
print("Template %s unchanged." % ctl_tpl)

# ---------------------------------------------------------------------------
# Parameter summary
# ---------------------------------------------------------------------------
utl.write_param_summary(fname, out_dir=ENSEMBLE_DIR)
