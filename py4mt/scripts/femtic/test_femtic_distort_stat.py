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
               PER_SITE=None, TITLE="auto", SHOW=False, PRINT_BEST=True, PLOT_MEMBERS=None, OUT_DIR=str(root / "results") + "/",
               OUTPUT_FORMATS=["png"], JPEG_QUALITY=95, DPI=60)
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


def test_normalize_formats():
    sup = ["png", "pdf", "jpg", "jpeg", "svg", "tif"]
    assert ds.normalize_formats(["png", "pdf"], sup) == ["png", "pdf"]
    assert ds.normalize_formats("pdf", sup) == ["pdf"]
    assert ds.normalize_formats("pdf, JPG;  .Svg", sup) == ["pdf", "jpg", "svg"]   # separators, case, dot
    assert ds.normalize_formats([".PNG", "png", "pdf", "PDF"], sup) == ["png", "pdf"]  # duplicates dropped
    assert ds.normalize_formats(("jpg", "jpeg"), sup) == ["jpg", "jpeg"]
    assert ds.normalize_formats(["xyz"]) == ["xyz"]                  # no check without a list of allowed names
    for bad, why in ((["png", "docx"], "unknown"), ([], "no format"), (None, "no format"),
                     ("", "no format"), (" , ", "no format")):
        try:
            ds.normalize_formats(bad, sup)
        except ValueError as e:
            assert why in str(e), (bad, str(e))
        else:
            raise AssertionError(f"no ValueError for {bad!r}")
    try:
        ds.normalize_formats("docx", sup)
    except ValueError as e:
        assert "'jpg'" in str(e) and "'pdf'" in str(e)               # the message lists the allowed formats


def test_save_figure_formats_and_jpeg_quality():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = np.random.default_rng(0)
    fig = ds.plot_histograms({"xx": r.normal(1, 0.1, 500), "xy": r.normal(0, 0.1, 500)}, title="T")
    with tempfile.TemporaryDirectory() as t:
        stem = Path(t) / "sub" / "fig"
        saved = ds._save_figure(fig, stem, ["png", "pdf", "jpg", "svg", "tif", "eps"], 80)
        assert [Path(x).suffix for x in saved] == [".png", ".pdf", ".jpg", ".svg", ".tif", ".eps"]
        head = {Path(x).suffix: Path(x).read_bytes()[:12] for x in saved}
        assert head[".png"][:8] == b"\x89PNG\r\n\x1a\n"
        assert head[".pdf"][:5] == b"%PDF-"
        assert head[".jpg"][:2] == b"\xff\xd8"
        assert head[".tif"][:4] in (b"II*\x00", b"MM\x00*")
        assert head[".eps"][:2] == b"%!"
        assert b"<svg" in Path(saved[3]).read_bytes()[:600]
        # text stays text in the pdf (Type 42 fonts embedded), checked on the title
        pdf = Path(saved[1]).read_bytes()
        assert b"/FontFile2" in pdf
        # jpeg quality changes the file size
        lo = ds._save_figure(fig, Path(t) / "lo", ["jpg"], 80, 20)[0]
        hi = ds._save_figure(fig, Path(t) / "hi", ["jpg"], 80, 95)[0]
        assert Path(lo).stat().st_size < Path(hi).stat().st_size
        # dpi acts on raster files only: a larger dpi gives a larger png, the same pdf size class
        big = ds._save_figure(fig, Path(t) / "big", ["png"], 160)[0]
        assert Path(big).stat().st_size > Path(saved[0]).stat().st_size
        # ".jpeg" extension is kept as given
        assert ds._save_figure(fig, Path(t) / "j", ["jpeg"], 80)[0].endswith(".jpeg")
    plt.close(fig)


def test_main_output_formats():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        make_ensemble(root, n=4)
        old = _set_config(root, OUTPUT_FORMATS="pdf, .JPG", PER_SITE=[101], JPEG_QUALITY=80)
        try:
            ds.main()
        finally:
            _restore(old)
        res = root / "results"
        names = sorted(p.name for p in res.glob("*.*") if p.suffix in (".png", ".pdf", ".jpg"))
        assert names == ["femtic_distort_stat.jpg", "femtic_distort_stat.pdf",
                         "femtic_distort_stat_site101.jpg", "femtic_distort_stat_site101.pdf"]
        assert (res / "femtic_distort_stat.pdf").read_bytes()[:5] == b"%PDF-"
        assert (res / "femtic_distort_stat.jpg").read_bytes()[:2] == b"\xff\xd8"


def test_main_bad_format_stops_before_reading_the_ensemble():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        make_ensemble(root, n=2)
        for bad in ({"OUTPUT_FORMATS": ["png", "docx"]}, {"OUTPUT_FORMATS": []},
                    {"JPEG_QUALITY": 0}, {"JPEG_QUALITY": 101}):
            old = _set_config(root, **bad)
            try:
                try:
                    ds.main()
                except ValueError as e:
                    assert "OUTPUT_FORMATS" in str(e) or "JPEG_QUALITY" in str(e)
                else:
                    raise AssertionError(f"no ValueError for {bad}")
            finally:
                _restore(old)
        assert not (root / "results").exists()                         # nothing was written


def test_title_and_caption_fit_into_the_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = np.random.default_rng(0)
    long_title = ("Distortion matrix of site ABC123 (C = I + C'): 80 members; marker = best member "
                  "ann_rto_orig-err_000 (nRMS 1.1)")
    for panel, fs in (((3.6, 2.9), 8), ((3.6, 2.9), 12), ((3.0, 2.6), 12)):
        for nq in (1, 2, 5):
            data = {k: r.normal(1, 0.1, 50) for k in ["xx", "xy", "yx", "yy", "condition"][:nq]}
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)           # e.g. "Tight layout not applied"
                fig = ds.plot_histograms(data, title=long_title, best_values={"xx": 1.0},
                                         panel_size=panel, tick_fs=fs, label_fs=fs)
            fig.canvas.draw()
            rend = fig.canvas.get_renderer()
            W, H = fig.bbox.width, fig.bbox.height
            texts = [fig._suptitle] + [x for x in fig.texts if x is not fig._suptitle]
            assert len(texts) == 2
            for tx in texts:
                bb = tx.get_window_extent(rend)
                assert bb.x0 >= 0 and bb.x1 <= W and bb.y0 >= 0 and bb.y1 <= H, (panel, fs, nq, tx.get_text()[:20], bb)
            # the title and caption do not overlap the panels
            top_ax = max(a.get_tightbbox(rend).y1 for a in fig.axes if a.get_visible())
            bot_ax = min(a.get_tightbbox(rend).y0 for a in fig.axes if a.get_visible())
            assert fig._suptitle.get_window_extent(rend).y0 >= top_ax - 1
            assert [x for x in fig.texts if x is not fig._suptitle][0].get_window_extent(rend).y1 <= bot_ax + 1
            plt.close(fig)
    assert ds._wrap_parts(["aaa", "bbb", "ccc"], 100) == "aaa; bbb; ccc"
    assert ds._wrap_parts(["aaa", "bbb", "ccc"], 9) == "aaa; bbb;\nccc"
    assert ds._wrap_parts([], 10) == ""


def test_count_axis_has_whole_number_ticks():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = ds.plot_histograms({"xx": np.random.default_rng(1).normal(1, 0.1, 20)}, bins=5)
    fig.canvas.draw()
    ticks = fig.axes[0].get_yticks()
    assert all(float(x) == int(x) for x in ticks), ticks
    plt.close(fig)


def test_format_best_table():
    member = dict(label="ens_007", iter=6, nrms=1.0765, status="accepted")
    C = np.array([[[1.10, 0.05], [-0.02, 1.20]],
                  [[0.50, 1.00], [1.00, 0.50]],                 # det < 0
                  [[np.nan] * 2] * 2])                          # site missing in this member
    txt = ds.format_best_table(member, C, [101, 102, 103], {101: "SiteA", 102: "SiteB"}, "full")
    lines = txt.splitlines()
    assert txt.isascii()
    assert "ens_007" in lines[0] and "iteration 6" in lines[0] and "nRMS 1.0765" in lines[0]
    assert "ITER_FALLBACK" not in lines[0] and "elements of C = I + C'" in lines[0]
    assert "name" in lines[2] and "Cxx" in lines[2] and "cond(C)" in lines[2]
    r1 = lines[4].split()
    assert r1[:2] == ["101", "SiteA"] and [float(x) for x in r1[2:6]] == [1.1, 0.05, -0.02, 1.2]
    sv = np.linalg.svd(C[0], compute_uv=False)
    assert abs(float(r1[6]) - sv[0] / sv[1]) < 5e-4 and abs(float(r1[7]) - np.linalg.det(C[0])) < 5e-5
    assert "det<0" in lines[5] and "det<0" not in lines[4]
    assert "not in the distortion file" in lines[6] and lines[6].split()[0] == "103"
    assert len(set(len(x) for x in (lines[1], lines[2], lines[3], lines[-1]))) == 1   # rules as wide as the header
    # C' elements, no names, fallback note
    member["status"] = "accepted_fallback"
    txt = ds.format_best_table(member, C[:1], [101], {}, "prime")
    l = txt.splitlines()
    assert "ITER_FALLBACK" in l[0] and "elements of C'" in l[0] and "= I + C'" not in l[0]
    assert "name" not in l[2]
    assert [float(x) for x in l[4].split()[1:5]] == [0.1, 0.05, -0.02, 0.2]            # diagonal minus 1


def test_main_prints_the_best_fit_member():
    import contextlib
    import io
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        C0, SIG, mats = make_ensemble(root, n=5)                    # nRMS of ens_000 .. ens_004: 1.100 .. 1.104
        # make ens_003 the best: its history has a lower nRMS at iteration 6
        write_cnv(root / "ens_003", [8.0, 4.0, 2.5, 1.8, 1.4, 1.2, 1.01, 1.15])
        for flag in (True, False):
            old = _set_config(root, PRINT_BEST=flag, PER_SITE=None)
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    ds.main()
            finally:
                _restore(old)
            out = buf.getvalue()
            if not flag:
                assert "best-fit member" not in out
                continue
            assert "best-fit member ens_003 (iteration 6, nRMS 1.0100)" in out
            row = [ln for ln in out.splitlines() if ln.split()[:1] == ["102"] and len(ln.split()) == 7][0]
            assert [float(x) for x in row.split()[1:5]] == [round(float(v), 4) for v in mats[3][1].ravel()]
    # the setting exists with its default
    assert ds.PRINT_BEST is True


def _members(labels, nrms=None):
    return [dict(label=l, nrms=(1.0 + 0.1 * i if nrms is None else nrms[i])) for i, l in enumerate(labels)]


def test_select_members_forms():
    labs = ["run_000", "run_001", "run_002", "run_010", "run_011", "other_5"]
    ms = _members(labs, nrms=[1.3, 1.2, 1.05, 1.4, 1.1, 1.5])          # best = run_002
    sel = lambda spec: ds.select_members(ms, spec)
    assert sel(None) == list(range(6)) and sel("all") == list(range(6)) and sel("") == list(range(6))
    assert sel("best") == [2]
    assert sel("run_010") == [3]                                       # exact name
    assert sel(["run_011", "run_000"]) == [4, 0]                       # order of the selectors
    assert sel("run_01?") == [3, 4]                                    # patterns
    assert sel("run_*") == [0, 1, 2, 3, 4] and sel(["run_0[01]0"]) == [0, 3]
    assert sel(11) == [4] and sel("11") == [4] and sel(5) == [5]       # numbers: trailing digits
    assert sel(["best", 11, "run_011", "best"]) == [2, 4]              # no repeats
    assert sel(np.int64(1)) == [1]
    assert sel(["best", "run_0*"]) == [2, 0, 1, 3, 4]
    for bad in ("nope", 99, "run_9*", ["best", "nope"], [], True, 1.5, ["run_000", None]):
        try:
            sel(bad)
        except ValueError as e:
            assert "PLOT_MEMBERS" in str(e), (bad, str(e))
        else:
            raise AssertionError(f"no ValueError for {bad!r}")
    try:
        sel("nope")
    except ValueError as e:
        assert "run_000" in str(e)                                     # lists the available names
    # a number that matches two members is ambiguous
    ms2 = _members(["a_007", "b_007", "c_008"])
    for spec in (7, "7"):
        try:
            ds.select_members(ms2, spec)
        except ValueError as e:
            assert "matches 2 members" in str(e) and "a_007" in str(e)
        else:
            raise AssertionError("ambiguous number accepted")
    assert ds.select_members(ms2, "a_007") == [0]                      # the full name is unambiguous
    # a member whose name is all digits: the exact name wins over the number rule
    assert ds.select_members(_members(["7", "x_7"]), "7") == [0]


def test_members_tag():
    assert ds.members_tag(["ens_001"]) == "ens_001"
    assert ds.members_tag(["ens_001", "ens_003"]) == "ens_001+ens_003"
    assert ds.members_tag(["a b", "c/d"]) == "a_b+c_d"                 # file-name safe
    assert ds.members_tag(["x" * 30, "y" * 30, "z" * 30]) == "3members"


def _spy_plots():
    """Wrap ds.plot_histograms; returns (calls, restore)."""
    calls = []
    orig = ds.plot_histograms
    def wrapper(data, **kw):
        calls.append((data, kw))
        return orig(data, **kw)
    ds.plot_histograms = wrapper
    return calls, lambda: setattr(ds, "plot_histograms", orig)


def test_main_plots_one_member_or_a_short_list():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        C0, SIG, mats = make_ensemble(root, n=6)       # ens_000 .. ens_005, nRMS 1.100 .. 1.105
        res = root / "results"
        # --- one member, by number: figure of that member only, statistics of all
        calls, restore = _spy_plots()
        old = _set_config(root, PLOT_MEMBERS=3, PER_SITE="all", TITLE="auto")
        try:
            ds.main()
        finally:
            _restore(old)
            restore()
        assert sorted(p.name for p in res.glob("*.png")) == ["femtic_distort_stat_ens_003.png"]
        assert len(calls) == 1                                          # PER_SITE skipped: one value per site
        data, kw = calls[0]
        assert np.allclose(data["xx"], mats[3][:, 0, 0], atol=1e-7)     # the values of member ens_003 only
        assert np.allclose(data["xy"], mats[3][:, 0, 1], atol=1e-7)
        assert "member ens_003" in kw["title"] and "6 members" not in kw["title"]
        rd = list(csv.DictReader((res / "femtic_distort_stat.csv").read_text().splitlines()))
        assert int(next(r for r in rd if r["site_id"] == "ALL" and r["quantity"] == "xx")["n"]) == 6 * 3
        assert np.load(res / "femtic_distort_stat.npz")["C"].shape[0] == 6    # NPZ keeps all members
        # --- a short list: pooled and per-site figures of those members; marker = best of them
        for f in res.glob("*.png"):
            f.unlink()
        calls, restore = _spy_plots()
        old = _set_config(root, PLOT_MEMBERS=["ens_004", 1], PER_SITE=[102], TITLE="auto")
        try:
            ds.main()
        finally:
            _restore(old)
            restore()
        assert sorted(p.name for p in res.glob("*.png")) == [
            "femtic_distort_stat_ens_004+ens_001.png", "femtic_distort_stat_ens_004+ens_001_site102.png"]
        pooled, kw0 = calls[0]
        assert np.allclose(pooled["xx"], np.array([mats[4][:, 0, 0], mats[1][:, 0, 0]]), atol=1e-7)   # (members, sites)
        site, kw1 = calls[1]
        assert np.allclose(site["xx"], [mats[4][1, 0, 0], mats[1][1, 0, 0]], atol=1e-7)
        assert "2 selected members" in kw0["title"] and "best member ens_001" in kw1["title"]
        assert abs(kw1["best_values"]["xx"] - mats[1][1, 0, 0]) < 1e-7  # ens_001 has the smaller nRMS


def test_main_member_selection_edge_cases():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        make_ensemble(root, n=4)
        res = root / "results"
        # selecting every member = no selection: same file names and title as the default
        calls, restore = _spy_plots()
        old = _set_config(root, PLOT_MEMBERS="ens_*", PER_SITE=[101])
        try:
            ds.main()
        finally:
            _restore(old)
            restore()
        assert sorted(p.name for p in res.glob("*.png")) == ["femtic_distort_stat.png",
                                                              "femtic_distort_stat_site101.png"]
        assert "4 members" in calls[0][1]["title"]
        # "best" and the {members} / {n_members} placeholders of a text title
        for f in res.glob("*.png"):
            f.unlink()
        calls, restore = _spy_plots()
        old = _set_config(root, PLOT_MEMBERS="best", TITLE="{n_members}: {members}")
        try:
            ds.main()
        finally:
            _restore(old)
            restore()
        assert calls[0][1]["title"] == "1: ens_000"                    # smallest nRMS is ens_000 (1.100)
        assert (res / "femtic_distort_stat_ens_000.png").exists()
    # a selector that matches nothing stops before anything is written
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        make_ensemble(root, n=3)
        old = _set_config(root, PLOT_MEMBERS=["ens_001", "ens_099"])
        try:
            try:
                ds.main()
            except ValueError as e:
                assert "ens_099" in str(e) and "ens_000" in str(e)
            else:
                raise AssertionError("no ValueError")
        finally:
            _restore(old)
        assert not (root / "results").exists()


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
