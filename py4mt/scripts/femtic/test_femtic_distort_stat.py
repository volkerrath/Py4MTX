#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_femtic_distort_stat.py
===========================

Tests of femtic_distort_stat.py with a small synthetic FEMTIC ensemble
(member directories with femtic.cnv and distortion_iter<N>.dat). Run with
pytest or directly:

    python test_femtic_distort_stat.py

Needs PY4MTX_ROOT / PY4MTX_DATA in the environment (as the script itself),
femtic_distort_stat.py in the same directory, and matplotlib.

@author vrath
@project py4mt
@created 2026-11-13
@modified 2026-11-13

AI-GENERATED CODE NOTICE
------------------------
Author    : Claude Sonnet 5.5 (Anthropic)
Generated : 2026-11-13
AI-generated; review before production use.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("femtic_distort_stat", _HERE / "femtic_distort_stat.py")
ds = importlib.util.module_from_spec(_spec)
sys.modules["femtic_distort_stat"] = ds
_spec.loader.exec_module(ds)
fem = ds.fem

HDR = "Iter#  Retrial#  Alpha  Beta  Damp  Roughness  Distortion  Misfit  RMS  ObjFunc\n"


def write_cnv(path: Path, rms, name="femtic.cnv"):
    rows = [f"{i}  0  10.0  1.0  0.0  5.0  0.1  {r * r:.4f}  {r:.4f}  100.0\n"
            for i, r in enumerate(rms)]
    (path / name).write_text(HDR + "".join(rows))


def write_dist(path: Path, it: int, ids, C):
    """distortion_iter<it>.dat from full matrices C (the file holds C' = C - I)."""
    lines = [f"{len(ids)}\n"]
    for sid, c in zip(ids, C):
        cp = np.asarray(c) - np.eye(2)
        lines.append(f"{sid}  " + "  ".join(f"{v:.8E}" for v in cp.ravel()) + "\n")
    (path / f"distortion_iter{it}.dat").write_text("".join(lines))


def make_ensemble(root: Path, n: int = 12, nsites: int = 3, seed: int = 5):
    """n good members (best iteration 6), true C = C0 + noise; returns the truth."""
    r = np.random.default_rng(seed)
    C0 = np.array([[[1.10, 0.05], [-0.02, 1.20]],
                   [[0.85, 0.20], [0.10, 1.05]],
                   [[1.00, 0.00], [0.00, 1.00]]])[:nsites]
    SIG = 0.03
    mats = []
    for k in range(n):
        d = root / f"ens_{k:03d}"
        d.mkdir()
        write_cnv(d, [8.0, 4.0, 2.5, 1.8, 1.4, 1.2, 1.1 + 0.001 * k, 1.15])   # best = iter 6
        C = C0 + r.normal(scale=SIG, size=C0.shape)
        write_dist(d, 6, [101 + i for i in range(nsites)], C)
        # a worse iteration with different numbers: must not be read
        write_dist(d, 7, [101 + i for i in range(nsites)], C + 5.0)
        mats.append(C)
    return C0, SIG, np.array(mats)


# --------------------------------------------------------------------------
def test_quantity_values_elements_and_prime():
    C = np.array([[[1.1, 0.05], [-0.02, 1.2]], [[0.9, 0.3], [0.2, 1.4]]])
    assert np.allclose(ds.quantity_values(C, "xx"), [1.1, 0.9])
    assert np.allclose(ds.quantity_values(C, "xy"), [0.05, 0.3])
    assert np.allclose(ds.quantity_values(C, "yx"), [-0.02, 0.2])
    assert np.allclose(ds.quantity_values(C, "yy", "prime"), [0.2, 0.4])
    assert np.allclose(ds.quantity_values(C, "xx", "prime"), [0.1, -0.1])
    assert np.allclose(ds.quantity_values(C, "xy", "prime"), [0.05, 0.3])     # off-diagonal unchanged
    assert ds.reference_value("xx") == 1.0 and ds.reference_value("xx", "prime") == 0.0
    assert ds.reference_value("xy") == 0.0 and ds.reference_value("condition") == 1.0


def test_quantity_values_derived_against_svd():
    r = np.random.default_rng(1)
    C = np.eye(2) + 0.4 * r.normal(size=(50, 2, 2))
    sv = np.linalg.svd(C, compute_uv=False)
    assert np.allclose(ds.quantity_values(C, "condition"), sv[:, 0] / sv[:, 1], rtol=1e-10)
    assert np.allclose(ds.quantity_values(C, "gain"), sv.mean(axis=1), rtol=1e-10)
    assert np.allclose(ds.quantity_values(C, "determinant"), np.linalg.det(C), rtol=1e-12)
    # no distortion
    I = np.eye(2)
    assert ds.quantity_values(I, "condition") == 1.0 and ds.quantity_values(I, "gain") == 1.0


def test_quantity_values_nan_inf_and_errors():
    C = np.array([[[np.nan] * 2] * 2, [[1.0, 0.0], [0.0, 0.0]]])
    cond = ds.quantity_values(C, "condition")
    assert np.isnan(cond[0]) and np.isinf(cond[1])                # NaN stays NaN, singular -> inf
    assert np.isnan(ds.quantity_values(C, "xx")[0])
    for bad in ({"key": "zz"}, {"key": "xx", "kind": "half"}):
        try:
            ds.quantity_values(C, **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"no ValueError for {bad}")


def test_best_iteration_rules():
    rows = [dict(Iter=i, RMS=r) for i, r in enumerate([0.9, 3.0, 1.2, 1.2, 2.0])]
    it, rms, cands = ds.best_iteration(rows)
    assert (it, rms) == (2, 1.2)                     # iter 0 (0.9) not eligible; tie -> lower iteration
    assert [c[0] for c in cands] == [1, 2, 3, 4]
    assert ds.best_iteration([dict(Iter=0, RMS=1.0)]) == (None, None, [])
    assert ds.best_iteration([]) == (None, None, [])


def test_scan_members_all_statuses():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        make_ensemble(root, n=2)                                           # ens_000, ens_001: fine
        def new(name):
            d = root / name
            d.mkdir()
            return d
        d = new("ens_bad_nrms");   write_cnv(d, [9, 3, 2.5, 2.2]);   write_dist(d, 3, [1], [np.eye(2)])
        d = new("ens_fallback");   write_cnv(d, [9, 3, 1.2, 1.1, 1.3])                    # best 3, file only for 2
        write_dist(d, 2, [1], [np.eye(2)])
        d = new("ens_fb_toohigh"); write_cnv(d, [9, 3, 1.9, 1.1, 1.3]); write_dist(d, 2, [1], [np.eye(2)])
        d = new("ens_nofile");     write_cnv(d, [9, 3, 1.2])                               # no distortion file at all
        d = new("ens_nocnv");      write_dist(d, 1, [1], [np.eye(2)])
        d = new("ens_badcnv");     (d / "femtic.cnv").write_text("garbage\nfoo\n")
        d = new("ens_noiter");     write_cnv(d, [1.0])                                     # only iter 0
        d = new("ens_dabic");      write_cnv(d, [9, 3, 1.3], name="FEMTIC-DABIC.cnv"); write_dist(d, 2, [1], [np.eye(2)])
        (root / "ens_file.txt").write_text("not a directory")
        (root / "other_000").mkdir()

        members, rej = ds.scan_members(str(root), "ens_", ["femtic.cnv", "FEMTIC-DABIC.cnv"],
                                       1.5, "distortion_iter{iter}.dat", True)
        got = {m["label"]: m for m in members}
        assert set(got) == {"ens_000", "ens_001", "ens_dabic", "ens_fallback"}
        assert got["ens_000"]["iter"] == 6 and got["ens_000"]["status"] == "accepted"
        assert got["ens_fallback"]["iter"] == 2 and got["ens_fallback"]["status"] == "accepted_fallback"
        assert got["ens_fallback"]["nrms"] == 1.2
        assert got["ens_dabic"]["iter"] == 2
        reasons = {r["label"]: r["reason"] for r in rej}
        assert reasons == {"ens_bad_nrms": "rejected_nrms", "ens_fb_toohigh": "no_distortion_file",
                           "ens_nofile": "no_distortion_file", "ens_nocnv": "no_cnv_file",
                           "ens_badcnv": "cnv_unreadable", "ens_noiter": "cnv_no_iterations"}
        # fallback off: ens_fallback is rejected too
        members2, rej2 = ds.scan_members(str(root), "ens_", ["femtic.cnv", "FEMTIC-DABIC.cnv"],
                                         1.5, "distortion_iter{iter}.dat", False)
        assert "ens_fallback" not in {m["label"] for m in members2}
        assert dict((r["label"], r["reason"]) for r in rej2)["ens_fallback"] == "no_distortion_file"
        # NRMS_MAX is applied to the best iteration
        members3, _ = ds.scan_members(str(root), "ens_", ["femtic.cnv", "FEMTIC-DABIC.cnv"],
                                      1.0, "distortion_iter{iter}.dat", True)
        assert members3 == []


def test_read_distortion_and_load_ensemble():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        C0, SIG, mats = make_ensemble(root, n=4)
        ids, C = ds.read_distortion(str(root / "ens_002" / "distortion_iter6.dat"))
        assert ids == [101, 102, 103]
        assert np.allclose(C, mats[2], atol=1e-7)                   # file holds C', read gives C = I + C'
        # a member with fewer sites, one with a malformed file
        d = root / "ens_short"
        d.mkdir(); write_cnv(d, [5, 1.2]); write_dist(d, 1, [101, 103], [C0[0], C0[2]])
        d = root / "ens_broken"
        d.mkdir(); write_cnv(d, [5, 1.2]); (d / "distortion_iter1.dat").write_text("3\n101 a b c d\n")
        members, _ = ds.scan_members(str(root), "ens_", "femtic.cnv", 1.5, "distortion_iter{iter}.dat")
        ok, ids_all, arr, failed = ds.load_ensemble(members)
        assert ids_all == [101, 102, 103]
        assert arr.shape == (5, 3, 2, 2)                            # 4 + short; broken dropped
        assert [f["label"] for f in failed] == ["ens_broken"]
        assert failed[0]["reason"] == "distortion_unreadable"
        short = [m["label"] for m in ok].index("ens_short")
        assert np.isnan(arr[short, 1]).all() and np.isfinite(arr[short, 0]).all()


def test_describe_and_stats_table():
    v = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    d = ds.describe(v, [50.0, 84.1])
    assert d["n"] == 4 and np.isclose(d["mean"], 2.5) and np.isclose(d["std"], np.std([1, 2, 3, 4], ddof=1))
    assert np.isclose(d["p50"], 2.5) and np.isclose(d["p84.1"], np.percentile([1, 2, 3, 4], 84.1))
    e = ds.describe(np.array([np.nan]), [50.0])
    assert e["n"] == 0 and np.isnan(e["mean"]) and np.isnan(e["p50"])
    assert ds.describe(np.array([2.0]), [50.0])["n"] == 1 and np.isnan(ds.describe(np.array([2.0]), [50.0])["std"])

    with tempfile.TemporaryDirectory() as t:
        C0, SIG, mats = make_ensemble(Path(t), n=40)
        rows = ds.stats_table(mats, [101, 102, 103], {102: "SiteB"}, ["xx", "xy"], "full", [16.0, 50.0, 84.0])
        assert len(rows) == 2 + 3 * 2
        pooled = [r for r in rows if r["site_id"] == "ALL" and r["quantity"] == "xx"][0]
        assert pooled["n"] == 40 * 3
        s2 = [r for r in rows if r["site_id"] == 102 and r["quantity"] == "xx"][0]
        assert s2["site_name"] == "SiteB" and s2["n"] == 40
        assert abs(s2["mean"] - 0.85) < 3 * SIG / np.sqrt(40) + 1e-12    # mean of the 40 draws near truth
        assert 0.5 * SIG < s2["std"] < 1.6 * SIG                         # spread near the true sigma
        # CSV: header, row count, ASCII only
        f = Path(t) / "sub" / "stats.csv"
        ds.write_stats_csv(f, rows, [16.0, 50.0, 84.0])
        txt = f.read_text()
        assert txt.isascii()
        rd = list(csv.reader(txt.splitlines()))
        assert rd[0] == ["site_id", "site_name", "quantity", "n", "mean", "std", "p16", "p50", "p84"]
        assert len(rd) == 1 + len(rows)
        tab = ds.format_pooled_table(rows, [16.0, 50.0, 84.0])
        assert tab.isascii() and "xx" in tab and "xy" in tab and "SiteB" not in tab


def test_select_sites():
    ids, names = [101, 102, 103], {102: "SiteB"}
    assert ds.select_sites(ids, names, None) == []
    assert ds.select_sites(ids, names, "all") == [0, 1, 2]
    assert ds.select_sites(ids, names, [103, "SiteB", 103]) == [2, 1]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert ds.select_sites(ids, names, ["nope"]) == []
        assert any("nope" in str(x.message) for x in w)


def test_check_distortion_type():
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "results_iter1.h5"
        assert "not found" in ds.check_distortion_type(p)
        p.write_text("x")
        old = fem.read_results_distortion
        try:
            fem.read_results_distortion = lambda *a, **k: {"type": 1, "type_name": "MATRIX_DIFFERENCE"}
            assert ds.check_distortion_type(p) is None
            fem.read_results_distortion = lambda *a, **k: {"type": 2, "type_name": "GAINS_AND_ROTATIONS"}
            try:
                ds.check_distortion_type(p)
            except RuntimeError as e:
                assert "GAINS_AND_ROTATIONS" in str(e)
            else:
                raise AssertionError("no RuntimeError")
            fem.read_results_distortion = lambda *a, **k: None
            assert "no /distortion" in ds.check_distortion_type(p)

            def no_h5(*a, **k):
                raise ImportError("h5py")
            fem.read_results_distortion = no_h5
            assert ds.check_distortion_type(p).startswith("h5py")
        finally:
            fem.read_results_distortion = old


def test_plot_histograms_structure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = np.random.default_rng(0)
    data = {k: r.normal(1.0, 0.05, 200) for k in ["xx", "xy", "yx", "yy", "condition"]}
    fig = ds.plot_histograms(data, title="T", best_values={"xx": 1.02, "xy": np.nan})
    axes = [a for a in fig.axes if a.get_visible()]
    assert len(axes) == 5 and len(fig.axes) == 6                    # 3 columns x 2 rows, one hidden
    assert fig._suptitle.get_text() == "T"
    assert all(a.get_legend() is None for a in fig.axes)           # single series: no legend
    assert sum(p.get_height() for p in axes[0].patches) == 200     # all values in the bins
    assert axes[0].get_title(loc="right").startswith("n = 200")    # stats in the panel title
    cap = [t.get_text() for t in fig.texts if t.get_text() != "T"]  # one caption names the line styles
    assert len(cap) == 1 and "median" in cap[0] and "no distortion" in cap[0] and "best-nRMS" in cap[0]
    assert all("no distortion" not in t.get_text() for a in axes for t in a.texts)
    plt.close(fig)
    fig = ds.plot_histograms({"xx": r.normal(size=50)}, show_percentiles=False, show_reference=False)
    assert fig.texts == []                                         # nothing to explain: no caption
    plt.close(fig)
    fig = ds.plot_histograms({"xx": np.array([1.0] * 5), "xy": np.array([np.nan])}, density=True)
    assert len(fig.axes) == 2                                      # constant data and all-NaN data do not fail
    plt.close(fig)
    fig = ds.plot_histograms({"xx": r.normal(size=50)}, ranges={"xx": (-1, 1)}, bins=10)
    assert len(fig.axes[0].patches) == 10 and fig.axes[0].get_xlim() == (-1.0, 1.0)
    plt.close(fig)


def _set_config(root: Path, **kw):
    cfg = dict(ENSEMBLE_DIR=str(root), ENSEMBLE_NAME="ens_", CNV_FILE=["femtic.cnv"], NRMS_MAX=1.5,
               ITER_FALLBACK=True, DISTORTION_FILE="distortion_iter{iter}.dat", RESULTS_H5_FILE=None,
               SITE_DAT=None, DISTORTION_KIND="full", QUANTITIES=["xx", "xy", "yx", "yy"],
               PERCENTILES=[2.3, 15.9, 50.0, 84.1, 97.7], BINS=15, DENSITY=False, RANGE=None,
               PER_SITE=None, TITLE="auto", SHOW=False, OUT_DIR=str(root / "results") + "/",
               OUTPUT_FORMATS=["png"], DPI=60)
    cfg.update(kw)
    old = {k: getattr(ds, k) for k in cfg}
    for k, v in cfg.items():
        setattr(ds, k, v)
    return old


def _restore(old):
    for k, v in old.items():
        setattr(ds, k, v)


def test_main_end_to_end_outputs():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        C0, SIG, mats = make_ensemble(root, n=10)
        d = root / "ens_rejected"
        d.mkdir(); write_cnv(d, [9, 3, 2.5]); write_dist(d, 2, [101, 102, 103], C0 + 3.0)
        old = _set_config(root, PER_SITE=[102], SITE_DAT=None)
        try:
            assert ds.main() == 0
        finally:
            _restore(old)
        res = root / "results"
        assert (res / "femtic_distort_stat.png").stat().st_size > 5000
        assert (res / "femtic_distort_stat_site102.png").stat().st_size > 5000
        assert not (res / "femtic_distort_stat_site101.png").exists()
        z = np.load(res / "femtic_distort_stat.npz")
        assert z["C"].shape == (10, 3, 2, 2)                           # rejected member left out
        assert list(z["site_ids"]) == [101, 102, 103]
        assert set(z["iterations"]) == {6}                             # best iteration, not 7
        assert np.allclose(z["C"], mats, atol=1e-7)                    # file content, not iter 7
        rd = list(csv.DictReader((res / "femtic_distort_stat.csv").read_text().splitlines()))
        pooled = [r for r in rd if r["site_id"] == "ALL" and r["quantity"] == "yy"][0]
        assert int(pooled["n"]) == 30
        assert abs(float(pooled["mean"]) - np.mean([1.2, 1.05, 1.0])) < 0.03
        assert (root / "femtic_distort_stat_summary.md").exists()      # parameter summary


def test_main_prime_titles_and_no_member():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        make_ensemble(root, n=3)
        titles = []
        import matplotlib.figure as mf
        orig = mf.Figure.suptitle
        def spy(self, txt, *a, **k):
            titles.append(txt)
            return orig(self, txt, *a, **k)
        mf.Figure.suptitle = spy
        old = _set_config(root, DISTORTION_KIND="prime", QUANTITIES=["xx", "gain"], PER_SITE="all",
                          TITLE={"pooled": "pooled {n_members}x{n_sites}", "site": "S{id}"})
        try:
            ds.main()
            titles_a = list(titles)
            titles.clear()
            ds.TITLE = None
            ds.PER_SITE = None
            ds.main()
            titles_b = list(titles)
            ds.ENSEMBLE_NAME = "nothing_"
            try:
                ds.main()
            except RuntimeError as e:
                assert "no member accepted" in str(e)
            else:
                raise AssertionError("no RuntimeError without members")
        finally:
            _restore(old)
            mf.Figure.suptitle = orig
        assert titles_a == ["pooled 3x3", "S101", "S102", "S103"]
        assert titles_b == []                                          # TITLE = None: no title
        # prime elements are C - I: the xx values are small
        z = np.load(root / "results" / "femtic_distort_stat.npz")
        assert str(z["kind"]) == "prime"


def test_main_stops_on_other_distortion_type():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        make_ensemble(root, n=2)
        (root / "ens_000" / "results_iter6.h5").write_text("x")
        old = _set_config(root, RESULTS_H5_FILE="results_iter{iter}.h5")
        saved = fem.read_results_distortion
        fem.read_results_distortion = lambda *a, **k: {"type": 3, "type_name": "ESTIMATE_GAINS_ONLY"}
        try:
            try:
                ds.main()
            except RuntimeError as e:
                assert "GAINS_ONLY" in str(e)
            else:
                raise AssertionError("no RuntimeError")
        finally:
            fem.read_results_distortion = saved
            _restore(old)


def test_main_site_names_from_sitelist():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        make_ensemble(root, n=4)
        sl = root / "sites.csv"
        sl.write_text("# name, lat, lon, elev, sitenum, easting, northing\n"
                      "SiteA, 45.0, 6.0, 500., 101, 1.0, 2.0\n"
                      "SiteB, 45.1, 6.1, 510., 102, 3.0, 4.0\n")
        old = _set_config(root, SITE_DAT=str(sl), PER_SITE=["SiteB", 103])
        try:
            ds.main()
        finally:
            _restore(old)
        res = root / "results"
        assert (res / "femtic_distort_stat_site102.png").exists()      # found by name
        assert (res / "femtic_distort_stat_site103.png").exists()      # found by ID, no name in the list
        assert not (res / "femtic_distort_stat_site101.png").exists()
        rd = list(csv.DictReader((res / "femtic_distort_stat.csv").read_text().splitlines()))
        names = {r["site_id"]: r["site_name"] for r in rd if r["site_id"] != "ALL"}
        assert names == {"101": "SiteA", "102": "SiteB", "103": ""}


def test_sources_are_ascii():
    for f in ("femtic_distort_stat.py", "test_femtic_distort_stat.py"):
        txt = (_HERE / f).read_text(encoding="utf-8")
        assert txt.isascii(), f"non-ASCII character in {f}"


if __name__ == "__main__":
    n_ok = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n_ok += 1
            print(f"ok  {name}")
    print(f"{n_ok} tests passed")
