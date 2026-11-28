#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare the run directories for a starting-model study with FEMTIC.

One directory is created per resistivity in RHO_LIST.  Each directory gets
its own resistivity_block_iter0.dat (MODEL_FILE), written from the template
with EVERY FREE REGION set to that resistivity, i.e. a homogeneous starting
model.  Air, ocean and all other fixed regions (flag 1), the element-to-
region map, the resistivity bounds and the flags are taken over from the
template unchanged.  All other input files (mesh, data, control file, ...)
are symlinked to the templates, exactly as in femtic_lcurve_prep.py and
femtic_rto_prep.py.

The directories are named <ENSEMBLE_NAME><rho>, e.g. ini_100, with rho in
Ohm.m, so that femtic_initial_post.py finds them with the glob "ini_*".

Workflow:
    1. femtic_initial_prep.py  (this script)  -> ini_<rho>/ directories
    2. run FEMTIC in every ini_<rho>/
    3. femtic_initial_post.py  -> best nRMS of every run versus
       log10(fill value), best starting resistivity
       (femtic_plot_conv.py overlays the femtic.cnv of several runs)

Edit control.dat in the templates directory for everything that is common
to all runs; only the starting model is changed here.  The reference model
(referencemodel.dat, the regularisation prior) is NOT changed: if it is in
LINK_LIST it is the same file in all runs, so that only the start differs.

Safety:
    * MODEL_FILE is written per run and must therefore be in neither
      COPY_LIST nor LINK_LIST; a link of that name left in a run directory
      by an earlier setup is removed first, so the template can never be
      written through a link.
    * the template is only read; its checksum is compared before and after.
    * each written model is read back and compared with the template
      (element map, bounds, flags, fixed regions unchanged, free regions
      equal to the requested resistivity), see VERIFY.
    * the starting resistivity is compared with the bounds of the free
      regions of the template before anything is created (ABORT_ON_BOUNDS).
    * air and ocean resistivities are passed to fem.insert_model from the
      template itself, so that they are not replaced by the defaults of
      that function.

@author:   Claude (Anthropic), for vrath
@project:  py4mt -- Python for Magnetotellurics
@inversion: FEMTIC

Provenance:
    2026-11-23  Claude Sonnet 5.5 (Anthropic)
                Created as femtic_initial.py, renamed to femtic_initial_prep.py
                the same day on request (date generated: 2026-11-23).
                Author: Claude, Anthropic.  Modelled on femtic_lcurve_prep.py
                (header, USER SECTION, COPY_LIST / LINK_LIST via
                ens.put_files, RELATIVE_LINKS, SKIP_FINISHED / CNV_FILE,
                template checksum, utl.write_param_summary); the model is
                written with fem.insert_model (as MOD_ITER0_HOMOGENEOUS in
                femtic_rto_prep.py).
                AI-generated code; review before production use.
"""

import os
import sys
import hashlib
from pathlib import Path
import inspect

import numpy as np

PY4MTX_ROOT = os.environ["PY4MTX_ROOT"]

for _base in [PY4MTX_ROOT + "/py4mt/modules/"]:
    for _p in [Path(_base), *Path(_base).rglob("*")]:
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

# femtic must be imported before ensembles (the two import each other).
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
#: created here.
ENSEMBLE_DIR = r"/media/vrath/LargeBack/Ensembles/sabancaya2026/initial_nodist/"
#: Run directories are <ENSEMBLE_DIR><ENSEMBLE_NAME><rho>/ .
ENSEMBLE_NAME = "ini_"
TEMPLATES = ENSEMBLE_DIR + "templates/"

#: Starting resistivities, one run each.  Any sequence of positive numbers,
#: in Ohm.m (or log10(Ohm.m) if RHO_LOG10 is True), e.g. [10., 100., 1000.]
#: or list(np.logspace(0, 3, 7)).  Every free region of the starting model
#: is set to this value.  The directory name uses the value in Ohm.m with 6
#: significant digits (duplicates at that precision are refused).
RHO_LIST =  np.logspace(0., 2., 11)
# RHO_LIST = [10., 30., 100., 300., 1000.]
RHO_LOG10 = False       # True: RHO_LIST is in log10(Ohm.m), e.g. [1., 2., 3.]

#: The starting model that is filled.  It is written in every run directory,
#: so it must be in neither COPY_LIST nor LINK_LIST.
MODEL_FILE = "resistivity_block_iter0.dat"

#: Copied into every run directory (files a run may edit, e.g. observe.dat).
COPY_LIST = []
#: Symlinked into every run directory.  Names missing in TEMPLATES are
#: reported and skipped (e.g. referencemodel.dat for a run without reference
#: model).  referencemodel.dat is linked unchanged: it is the same prior for
#: all runs, only the starting model differs.
LINK_LIST = ["control.dat",
             "mesh.dat",
             "observe.dat",
             "distortion_iter0.dat",
             "referencemodel.dat",
             "site.dat",
             "run_femtic_dias.sh", "run_femtic_kraken.sh"]
RELATIVE_LINKS = True   # True: portable relative symlinks (default, survives tgz);
                        # False: absolute symlinks

#: True: stop before anything is created if a starting resistivity is
#: outside the [lower, upper] bounds of a free region in the template.
#: False: only report it.
ABORT_ON_BOUNDS = True

#: True: read every written model back and compare it with the template.
#: Costs one more read of the block file per run.
VERIFY = True

#: True: a run directory that already holds a convergence file (CNV_FILE) is
#: left untouched, so re-running the script with an extended RHO_LIST never
#: rewrites the starting model of finished runs.  False: rewrite all.
SKIP_FINISHED = True
CNV_FILE = ["femtic.cnv", "FEMTIC-DABIC.cnv"]

# =============================================================================
# END OF USER SECTION
# =============================================================================

#: relative tolerance for the read-back comparison; the block file holds
#: 7 significant digits (%.6e).
_RTOL = 1.0e-5


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _has_results(directory):
    names = [CNV_FILE] if isinstance(CNV_FILE, str) else list(CNV_FILE)
    return any(os.path.isfile(os.path.join(directory, n)) for n in names)


def _free_regions(blk):
    """Boolean mask of the free regions of a block read by fem.read_resistivity_block.

    Fixed: region 0 (air) and every region with flag 1 (this includes the
    ocean, which fem.insert_model only recognises if it has flag 1).
    """
    fixed = np.asarray(blk["region_flag"]) == 1
    fixed[0] = True
    return ~fixed


def _compare_with_template(path, tpl, free, rho):
    """Read the written model back; return a list of problems (empty = ok)."""
    new = fem.read_resistivity_block(path)
    problems = []
    if int(new["nelem"]) != int(tpl["nelem"]) or int(new["nreg"]) != int(tpl["nreg"]):
        return ["nelem/nreg differ from the template"]
    if not np.array_equal(new["region_of_elem"], tpl["region_of_elem"]):
        problems.append("element-region map differs from the template")
    for key in ("region_rho_lower", "region_rho_upper", "region_n"):
        if not np.allclose(new[key], tpl[key], rtol=_RTOL, atol=0.0):
            problems.append("%s differs from the template" % key)
    if not np.array_equal(new["region_flag"], tpl["region_flag"]):
        problems.append("region_flag differs from the template")
    if not np.allclose(new["region_rho"][~free], tpl["region_rho"][~free],
                       rtol=_RTOL, atol=0.0):
        problems.append("fixed regions (air/ocean/...) differ from the template")
    if not np.allclose(new["region_rho"][free], rho, rtol=_RTOL, atol=0.0):
        problems.append("free regions are not all %.7g Ohm.m "
                        "(min %.7g, max %.7g)" % (rho, new["region_rho"][free].min(),
                                                  new["region_rho"][free].max()))
    return problems


# =============================================================================
# Check the setup before anything is created
# =============================================================================
ens_dir = os.path.join(ENSEMBLE_DIR, "")
tpl_dir = os.path.join(TEMPLATES, "")
if not os.path.isdir(tpl_dir):
    sys.exit(" Directory: %s does not exist, needs to be copied !" % tpl_dir)

if MODEL_FILE in COPY_LIST or MODEL_FILE in LINK_LIST:
    sys.exit(" %s must be in neither COPY_LIST nor LINK_LIST: it is written "
             "in every run directory, a link would write into the template."
             % MODEL_FILE)

tpl_model = tpl_dir + MODEL_FILE
if not os.path.isfile(tpl_model):
    sys.exit(" MODEL_FILE %s does not exist in %s" % (MODEL_FILE, tpl_dir))

for _f in COPY_LIST:
    if not os.path.isfile(tpl_dir + _f):
        sys.exit(" COPY_LIST file %s does not exist in %s" % (_f, tpl_dir))

link_missing = [f for f in LINK_LIST if not os.path.exists(tpl_dir + f)]
link_list = [f for f in LINK_LIST if f not in link_missing]
if link_missing:
    print("WARNING: not in %s, not linked: %s\n" % (tpl_dir, link_missing))

rho_in = np.atleast_1d(np.asarray(RHO_LIST, dtype=float)).ravel()
if rho_in.size == 0:
    sys.exit(" RHO_LIST is empty.")
if not np.all(np.isfinite(rho_in)):
    sys.exit(" RHO_LIST must contain finite values, got %s" % rho_in.tolist())
rho_values = np.power(10.0, rho_in) if RHO_LOG10 else rho_in
if not np.all(np.isfinite(rho_values)) or np.any(rho_values <= 0.0):
    sys.exit(" RHO_LIST must give resistivities > 0 Ohm.m, got %s"
             % rho_values.tolist())
tags = [format(v, "g") for v in rho_values]
dups = sorted({t for t in tags if tags.count(t) > 1})
if dups:
    sys.exit(" RHO_LIST has values that give the same directory name: %s"
             % dups)

# --- the template starting model --------------------------------------------
tpl_blk = fem.read_resistivity_block(tpl_model)
free = _free_regions(tpl_blk)
n_free = int(free.sum())
if n_free == 0:
    sys.exit(" %s has no free regions, nothing to fill." % tpl_model)
n_free_lib = fem.read_model(tpl_model, model_trans="log10", out=False).size
if n_free_lib != n_free:
    sys.exit(" %s: %d free regions here, but fem.read_model returns %d "
             "(ocean/fixed-region detection differs); not continuing."
             % (tpl_model, n_free, n_free_lib))

nreg = int(tpl_blk["nreg"])
ocean_present = (nreg > 1 and int(tpl_blk["region_flag"][1]) == 1
                 and float(tpl_blk["region_rho"][1]) <= 1.0)
# fixed values of the template, handed to fem.insert_model so that they are kept
air_rho = float(tpl_blk["region_rho"][0])
ocean_rho = float(tpl_blk["region_rho"][1]) if ocean_present else 0.25

rho_tpl = tpl_blk["region_rho"][free]
print("Template %s: format %s, %d elements, %d regions "
      "(%d free, %d fixed%s)."
      % (MODEL_FILE, tpl_blk["fmt"], int(tpl_blk["nelem"]), nreg, n_free,
         nreg - n_free, ", region 1 = ocean" if ocean_present else ""))
print("  free regions in the template: %.4g ... %.4g Ohm.m (median %.4g); "
      "each run sets all of them to one value."
      % (rho_tpl.min(), rho_tpl.max(), np.median(rho_tpl)))
if tpl_blk["fmt"] == "v5" and np.any(np.asarray(
        tpl_blk["region_aniso_type"])[free] != 0):
    print("  NOTE: the template has anisotropic free regions; fem.insert_model "
          "sets rhoXX and scales rhoYY/rhoZZ by the same factor, so the "
          "anisotropy ratio of each region is kept.")

# --- starting values against the bounds of the free regions ------------------
lo = tpl_blk["region_rho_lower"][free]
hi = tpl_blk["region_rho_upper"][free]
bounds_bad = []
for value in rho_values:
    n_out = int(np.count_nonzero((value < lo) | (value > hi)))
    if n_out:
        bounds_bad.append((value, n_out))
if bounds_bad:
    print("WARNING: starting resistivity outside the bounds of the free regions "
          "(bounds %.4g ... %.4g Ohm.m in the template):" % (lo.min(), hi.max()))
    for value, n_out in bounds_bad:
        print("         %-12s Ohm.m: %d of %d free regions" %
              (format(value, "g"), n_out, n_free))
    if ABORT_ON_BOUNDS:
        sys.exit(" Nothing created.  Change RHO_LIST or the bounds in %s, or "
                 "set ABORT_ON_BOUNDS = False." % tpl_model)
    print()

if "referencemodel.dat" in link_list:
    print("referencemodel.dat is linked unchanged: all runs have the same "
          "reference (prior) model, only the starting model differs.\n")
else:
    print("NOTE: no referencemodel.dat is linked.  Check how FEMTIC treats "
          "a missing reference model: if the starting model then also serves "
          "as the reference, the runs differ in start AND prior.\n")

# =============================================================================
# Create the run directories
# =============================================================================
tpl_hash = _sha256(tpl_model)

runs = []
for value, tag in zip(rho_values, tags):
    directory = "%s%s%s/" % (ens_dir, ENSEMBLE_NAME, tag)

    if SKIP_FINISHED and _has_results(directory):
        runs.append((value, directory, "skipped"))
        continue

    os.makedirs(directory, exist_ok=True)
    ens.put_files(COPY_LIST, link_list, directory, tpl_dir,
                  relative_links=RELATIVE_LINKS)

    model = directory + MODEL_FILE
    # a link (or a file) left from an earlier setup must not be written through
    if os.path.lexists(model):
        os.remove(model)
    fem.insert_model(template=tpl_model,
                     model=np.full(n_free, np.log10(value)),
                     model_file=model,
                     out=False,
                     air_rho=air_rho,
                     ocean_rho=ocean_rho)

    status = "written"
    if VERIFY:
        problems = _compare_with_template(model, tpl_blk, free, value)
        if problems:
            sys.exit(" %s: read-back check failed:\n   %s"
                     % (model, "\n   ".join(problems)))
        status = "written, verified"
    runs.append((value, directory, status))

if _sha256(tpl_model) != tpl_hash:
    sys.exit(" The template %s was changed -- this must not happen!" % tpl_model)

# =============================================================================
# Summary
# =============================================================================
print("=" * 78)
print("%4s  %-12s  %-8s  %-18s  %s" % ("run", "rho [Ohm.m]", "log10", "model", "directory"))
print("-" * 78)
for n, (value, directory, status) in enumerate(runs):
    print("%4d  %-12s  %8.4f  %-18s  %s"
          % (n, format(value, "g"), np.log10(value), status, directory))
print("=" * 78)
n_new = sum(1 for r in runs if r[2] != "skipped")
print("%d run directories written, %d skipped (finished)." %
      (n_new, len(runs) - n_new))
print("Template %s unchanged." % tpl_model)

# ---------------------------------------------------------------------------
# Parameter summary
# ---------------------------------------------------------------------------
utl.write_param_summary(fname, out_dir=ENSEMBLE_DIR)
