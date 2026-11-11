#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_femtic_misfit.py

Pytest suite for femtic_misfit.py (changes of 2026-11-10): output directory
named by OUT_PREFIX, site list for the curves with single pdf / jpg files,
site names of the local site.dat in any case, "site NAME (site#)" titles, and
std / skew / excess kurtosis in the statistics tables.

The tests need no results_iterN.h5 and no h5py: femtic.read_results_data is
replaced by a reader that returns synthetic data (two "runs", four stations,
MT impedance plus VTF tippers numbered 1001.. at the same positions).

Run
---
From the folder containing femtic_misfit.py and this file::

    python -m pytest -q test_femtic_misfit.py

Provenance
----------
Author      : Claude (Anthropic)
Generated   : 2026-11-10
Notice      : This code is AI-generated. Review and test it before any
              production use.
"""
from __future__ import annotations

import os
import re
import sys
from typing import Dict, List

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import femtic as fem  # noqa: E402
import femtic_misfit as fm  # noqa: E402

MU0 = 4.0e-7 * np.pi
NAMES = ["Ann01", "Ann02", "Ann03", "Ann04"]
FREQ = np.logspace(-2, 2, 9)
ROW_DT = np.dtype([
    ("freq", "f8"), ("datatype", "i4"), ("site_id", "i4"), ("site_x", "f8"),
    ("site_y", "f8"), ("site_z", "f8"), ("re_val", "f8"), ("im_val", "f8"),
    ("re_err", "f8"), ("im_err", "f8"), ("cal_re", "f8"), ("cal_im", "f8"),
    ("component", "i4")])


def _synthetic(seed: int) -> dict:
    """Results-data dict of one run (same rows in every run, other calculated values)."""
    rng = np.random.default_rng(1234)             # observed data: the same for all runs
    rng_cal = np.random.default_rng(seed)
    rows = []
    for k in range(4):
        x, y = 1000.0 * (k + 1), 2000.0 + 500.0 * k
        for dt, comps, sid in ((0, (0, 1, 2, 3), k + 1), (3, (0, 1), 1001 + k)):
            for c in comps:
                for f in FREQ:
                    z = np.sqrt(MU0 * 2 * np.pi * f * 100.0) * np.exp(
                        1j * np.radians(45.0 if c in (1, 3) else -135.0))
                    if dt == 3:
                        z = 0.1 * np.exp(1j * 0.3)
                    err = 0.05 * abs(z)
                    obs = z + err * (rng.normal() + 1j * rng.normal())
                    cal = obs + err * (rng_cal.standard_t(3) + 1j * rng_cal.normal())
                    rows.append((f, dt, sid, x, y, 0.0, obs.real, obs.imag, err, err,
                                 cal.real, cal.imag, c))
    rows = np.array(rows, dtype=ROW_DT)
    dtn = np.array([fem.RESULTS_H5_DATATYPE_NAMES[int(d)] for d in rows["datatype"]])
    cn = np.array([fem.RESULTS_H5_COMPONENT_NAMES[d][int(c)]
                   for d, c in zip(dtn, rows["component"])])
    return dict(iterNum=5, nRows=rows.size, nMissingCalc=0, rows=rows,
                datatype_name=dtn, component_name=cn)


@pytest.fixture()
def world(tmp_path, monkeypatch):
    """Two run directories with placeholder results files, a patched reader, a site.dat."""
    runs = []
    for i in (1, 2):
        d = tmp_path / f"run{i}"
        d.mkdir()
        (d / "results_iter5.h5").write_bytes(b"x")
        runs.append(str(d))
    data = {str(tmp_path / f"run{i}" / "results_iter5.h5"): _synthetic(i) for i in (1, 2)}
    monkeypatch.setattr(fem, "read_results_data", lambda src, **kw: data[str(src)])
    # site.dat of mt_make_sitelist.py: name, lat, lon, elev, sitenum, easting, northing
    # (results file: x = north, y = east, relative to the model origin)
    sd = tmp_path / "local"
    sd.mkdir()
    with open(sd / "site.dat", "w") as fh:
        for k, nm in enumerate(NAMES):
            fh.write(f"{nm}, 46.0, 6.0, 500.0, {k + 1}, "
                     f"{300000.0 + 2000.0 + 500.0 * k}, {5000000.0 + 1000.0 * (k + 1)}\n")
    return dict(tmp=tmp_path, runs=runs, local=str(sd))


def _opts(**kw):
    base = dict(METHODS=("crossplot", "histogram", "qq"), PLOT_FORMATS=(".pdf",),
                PLOT_DPI=40, HIST_MIN_N=3, QQ_MIN_N=5, CURVES_WHAT=("rho", "phase"),
                CURVES_SITE_DAT=None, CURVES_COMPS="xy, yx")
    base.update(kw)
    return fm.make_settings(**base)


def _files(root) -> List[str]:
    out = []
    for dp, _, fns in os.walk(root):
        out += [os.path.relpath(os.path.join(dp, f), root) for f in fns]
    return sorted(out)


# ---------------------------------------------------------------------------
# (2) higher moments
# ---------------------------------------------------------------------------
def test_sample_moments_match_scipy():
    from scipy import stats
    rng = np.random.default_rng(1)
    for x in (rng.normal(size=4000), rng.standard_t(4, size=4000), rng.exponential(size=999)):
        m = fm.sample_moments(x)
        assert m["std"] == pytest.approx(np.std(x), rel=1e-12)
        assert m["skew"] == pytest.approx(stats.skew(x), abs=1e-12)
        assert m["kurt"] == pytest.approx(stats.kurtosis(x), abs=1e-12)    # excess


def test_sample_moments_degenerate():
    for x in (np.zeros(6), np.full(6, 0.3), np.array([])):
        m = fm.sample_moments(x)
        assert np.isnan(m["skew"]) and np.isnan(m["kurt"])
    m = fm.sample_moments(np.array([1.0, 2.0, 4.0]))          # n = 3: skew, no kurt
    assert np.isfinite(m["skew"]) and np.isnan(m["kurt"])
    assert fm.sample_moments(np.array([1.0, np.nan, 2.0, 4.0, 9.0]))["kurt"] == pytest.approx(-1.0)


def test_fit_measures_keys():
    m = fm.fit_measures(np.random.default_rng(0).normal(size=100))
    for k in ("n", "nrms", "r_mae", "r_med", "q95", "mean", "std", "skew", "kurt"):
        assert k in m


def test_tables_have_moment_columns(world):
    opts = _opts(OUT_PREFIX="Misf", METHODS=("crossplot", "histogram", "qq"))
    assert fm.run_models(opts, [world["runs"][0]]) == 0
    out = world["tmp"] / "run1" / "Misf"
    for name, cols in (("fit_measures", ("std", "skew", "kurt")),
                       ("hist_stats", ("std", "skew", "kurt")),
                       ("crossplot_stats", ("skew", "kurt")),
                       ("qq_stats", ("skew", "kurt"))):
        txt = (out / f"Misf_{name}.txt").read_text()
        txt.encode("ascii")                                    # plain ASCII only
        hdr = [ln for ln in txt.splitlines() if not ln.startswith(("#", "=", "-"))][0]
        for c in cols:
            assert re.search(rf"\b{c}", hdr), (name, c, hdr)
    # the pooled table: skew / kurt of the first data row are numbers
    fit = [ln for ln in (out / "Misf_fit_measures.txt").read_text().splitlines()
           if ln.startswith("ALL")][0].split()
    assert all(np.isfinite(float(v)) for v in fit[-3:])
    assert "EXCESS" in (out / "Misf_fit_measures.txt").read_text()


def test_ensemble_tables_have_both_norms(world):
    opts = _opts(OUT_PREFIX="Ens", METHODS=("histogram",))
    assert fm.run_ensemble(opts, world["runs"]) == 0
    txt = (world["tmp"] / "Ens" / "Ens_fit_measures.txt").read_text()
    assert " err+syn " in txt and " err " in txt


# ---------------------------------------------------------------------------
# (3) output directory
# ---------------------------------------------------------------------------
def test_outputs_go_to_prefix_directory(world):
    opts = _opts(OUT_PREFIX="Misf", METHODS=("histogram",))
    assert fm.run_models(opts, [world["runs"][0]]) == 0
    files = _files(world["tmp"] / "run1")
    assert files and all(f.startswith("Misf" + os.sep) or f == "results_iter5.h5" for f in files)
    assert os.path.isdir(world["tmp"] / "run1" / "Misf")


def test_several_runs_each_in_own_prefix_directory(world):
    opts = _opts(OUT_PREFIX="Misf", METHODS=("histogram",))
    assert fm.run_models(opts, world["runs"]) == 0
    for i in (1, 2):
        names = os.listdir(world["tmp"] / f"run{i}" / "Misf")
        assert names and all(n.startswith(f"Misf_run{i}_iter5") for n in names)


def test_existing_directory_is_fine(world):
    (world["tmp"] / "run1" / "Misf").mkdir()
    assert fm.run_models(_opts(OUT_PREFIX="Misf", METHODS=("histogram",)),
                         [world["runs"][0]]) == 0


def test_ensemble_goes_to_prefix_directory_in_common_parent(world):
    assert fm.run_ensemble(_opts(OUT_PREFIX="Ens", METHODS=("histogram",)), world["runs"]) == 0
    assert (world["tmp"] / "Ens" / "Ens_runs.txt").is_file()


def test_out_dir_is_used_as_given(world):
    target = world["tmp"] / "somewhere" / "else"
    opts = _opts(OUT_DIR=str(target), OUT_PREFIX="Misf", METHODS=("histogram",))
    assert fm.run_models(opts, [world["runs"][0]]) == 0
    assert (target / "Misf_runs.txt").is_file()
    assert not (target / "Misf").exists()
    assert not (world["tmp"] / "run1" / "Misf").exists()


def test_prefix_dir_off_and_no_prefix(world):
    opts = _opts(OUT_PREFIX="Misf", OUT_PREFIX_DIR=False, METHODS=("histogram",))
    assert fm.run_models(opts, [world["runs"][0]]) == 0
    assert (world["tmp"] / "run1" / "Misf_runs.txt").is_file()
    assert not (world["tmp"] / "run1" / "Misf").exists()
    opts = _opts(OUT_PREFIX=None, METHODS=("histogram",))        # nothing to name a directory
    assert fm.run_models(opts, [world["runs"][1]]) == 0
    assert (world["tmp"] / "run2" / "run2_iter5_runs.txt").is_file()


def test_plot_dir_is_inside_prefix_directory(world):
    opts = _opts(OUT_PREFIX="Misf", PLOT_DIR="figs", METHODS=("histogram",))
    assert fm.run_models(opts, [world["runs"][0]]) == 0
    assert list((world["tmp"] / "run1" / "Misf" / "figs").glob("*.pdf"))


# ---------------------------------------------------------------------------
# (1) site list, single files, names of the local site.dat
# ---------------------------------------------------------------------------
@pytest.fixture()
def titles(monkeypatch):
    """Record the suptitle of every site figure."""
    seen: Dict[int, str] = {}
    orig = fm.plot_site_curves

    def spy(sid, *a, **kw):
        fig = orig(sid, *a, **kw)
        if fig is not None:
            seen[sid] = fig._suptitle.get_text().splitlines()[0]
        return fig

    monkeypatch.setattr(fm, "plot_site_curves", spy)
    return seen


def _curves(world, titles, run_idx=0, **kw):
    opts = _opts(METHODS=("curves",), OUT_PREFIX="Misf", **kw)
    assert fm.run_models(opts, [world["runs"][run_idx]]) == 0
    return world["tmp"] / f"run{run_idx + 1}" / "Misf"


def test_site_list_gives_single_pdf_and_jpg(world, titles):
    out = _curves(world, titles, CURVES_SITE_DAT=os.path.join(world["local"], "site.dat"),
                  CURVES_SITES=["ann02", "Ann04"], PLOT_FORMATS=(".pdf", ".jpg"))
    sub = out / "Misf_curves"
    got = sorted(os.listdir(sub))
    assert got == ["Misf_curves_site2_Ann02.jpg",
                   "Misf_curves_site2_Ann02.pdf",
                   "Misf_curves_site4_Ann04.jpg",
                   "Misf_curves_site4_Ann04.pdf"], got
    assert not (out / "Misf_curves.pdf").exists()      # no multi-page pdf
    assert sorted(titles) == [2, 4]


def test_all_sites_default_is_one_multipage_pdf(world, titles):
    out = _curves(world, titles, PLOT_FORMATS=(".pdf", ".jpg"))
    assert (out / "Misf_curves.pdf").is_file()
    sub = os.listdir(out / "Misf_curves")
    assert len(sub) == 4 and all(s.endswith(".jpg") for s in sub)  # jpgs: always single


def test_multipage_override_both_ways(world, titles):
    out = _curves(world, titles, CURVES_SITES=[1, 2], CURVES_MULTIPAGE=True)
    assert (out / "Misf_curves.pdf").is_file()
    assert not (out / "Misf_curves").exists()
    out = _curves(world, titles, run_idx=1, CURVES_SITES=2, CURVES_MULTIPAGE=False)
    assert len(os.listdir(out / "Misf_curves")) == 2


def test_site_ids_and_merged_ids(world, titles):
    out = _curves(world, titles, CURVES_SITES=[3, "1003", 1004, 99, "nobody"],
                  PLOT_FORMATS=(".pdf",))
    # 3 -> site 3; "1003" and 1004 are the VTF ids merged into sites 3 and 4
    assert sorted(titles) == [3, 4]
    assert len(os.listdir(out / "Misf_curves")) == 2


def test_names_from_local_site_dat_in_working_directory(world, titles, monkeypatch):
    monkeypatch.chdir(world["local"])       # no site.dat near the runs: the local one is found
    _curves(world, titles, CURVES_SITE_DAT="auto", CURVES_SITES="ann03, 1")
    assert titles[3].startswith("site Ann03 (3), x=")
    assert titles[1].startswith("site Ann01 (1), x=")


def test_name_case_in_title_and_file_name(world, titles):
    sd = os.path.join(world["local"], "site.dat")
    for case, nm in (("upper", "ANN02"), ("lower", "ann02"), (None, "Ann02")):
        titles.clear()
        out = _curves(world, titles, CURVES_SITE_DAT=sd, CURVES_SITES=["Ann02"],
                      CURVES_NAME_CASE=case)
        assert titles[2].startswith(f"site {nm} (2), x=")
        assert os.listdir(out / "Misf_curves") == [
            f"Misf_curves_site2_{nm}.pdf"]
        for f in os.listdir(out / "Misf_curves"):
            os.remove(out / "Misf_curves" / f)


def test_without_names_title_has_only_the_number(world, titles):
    _curves(world, titles, CURVES_SITE_DAT=None, CURVES_SITES=[2], CURVES_NAME_CASE="upper")
    assert titles[2].startswith("site 2, x=")


def test_bad_name_case_is_rejected(world, titles):
    opts = _opts(METHODS=("curves",), OUT_PREFIX="Misf", CURVES_NAME_CASE="title")
    assert fm.run_models(opts, [world["runs"][0]]) == 1


def test_names_requested_but_unavailable_say_so(world, titles, capsys):
    _curves(world, titles, CURVES_SITE_DAT=None, CURVES_SITES=["Ann02", 1])
    cap = capsys.readouterr().out
    assert "no site names are available" in cap
    assert sorted(titles) == [1]


def test_select_sites_unit():
    info = {1: dict(name="A01", nrms=1.0, also=[1001]), 2: dict(name="A02", nrms=3.0, also=[]),
            3: dict(name="A02", nrms=2.0, also=[])}          # a name shared by two ids
    assert fm.select_sites(info, "a02, 1", "even", 200) == [2, 3, 1]
    assert fm.select_sites(info, ["A01", 1, 1001], "even", 200) == [1]       # no duplicates
    assert fm.select_sites(info, 2, "worst", 200) == [2, 3]
    assert fm.select_sites(info, None, "id", 2) == [1, 2]


# ---------------------------------------------------------------------------
# (3) the "nrms" method (needs only the nRMS table of femtic_ens_post.py)
# ---------------------------------------------------------------------------
def test_nrms_histogram_goes_to_prefix_directory(tmp_path):
    csv = tmp_path / "ens" / "ann_nrms.csv"
    csv.parent.mkdir()
    rng = np.random.default_rng(3)
    with open(csv, "w") as fh:
        fh.write("member,best_iter,nrms,status\n")
        for k in range(30):
            v = 1.0 + abs(rng.normal(scale=0.4))
            fh.write(f"m{k:02d},12,{v:.4f},{'accepted' if v < 1.5 else 'rejected'}\n")
    opts = _opts(METHODS=("nrms",), NRMS_RESULTS=None, NRMS_CSV=str(csv), OUT_PREFIX="Misf")
    assert fm.run_nrms_histogram(opts) == 0
    out = tmp_path / "ens" / "Misf"
    assert (out / "Misf_nrms_hist.pdf").is_file() and (out / "Misf_nrms_stats.txt").is_file()
    assert sorted(os.listdir(tmp_path / "ens")) == ["Misf", "ann_nrms.csv"]
