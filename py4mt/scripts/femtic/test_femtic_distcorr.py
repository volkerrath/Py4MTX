#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_femtic_distcorr.py
=======================

Tests of femtic_distcorr.py with small synthetic FEMTIC files (observe.dat,
distortion_iter<N>.dat, femtic.cnv). Run with pytest or directly:

    python test_femtic_distcorr.py

Needs PY4MTX_ROOT / PY4MTX_DATA in the environment (as the script itself),
femtic_distcorr.py in the same directory, and femtic_viz.py / data_viz.py /
matplotlib for the plot tests.

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

import importlib.util
import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("femtic_distcorr", _HERE / "femtic_distcorr.py")
fd = importlib.util.module_from_spec(_spec)
sys.modules["femtic_distcorr"] = fd
_spec.loader.exec_module(fd)
fem = fd.fem

# Distortion of the synthetic run: C' (difference), C = I + C'
CP = {1: [[0.10, 0.05], [-0.02, 0.20]],
      2: [[-0.15, 0.30], [0.10, 0.05]],
      3: [[0.0, 0.0], [0.0, 0.0]],          # identity
      4: [[0.0, 0.0], [0.0, -1.0]]}         # C = diag(1, 0): singular


def _fmt(vals):
    return "    ".join(f"{v:.8E}" for v in vals)


def make_observe(path: Path, legacy: bool = False, seed: int = 3):
    """MT block (sites 1, 2, 3, 4), VTF block (one real, one dummy), PT block."""
    r = np.random.default_rng(seed)
    out = ["MT    4\n"]
    for i, sid in enumerate((1, 2, 3, 4)):
        out.append((f"site{sid} {float(i):.1f} 0.0 0.0\n") if legacy
                   else f"{sid}    {sid}    {float(i):.1f}    0.0\n")
        nf = 3
        out.append(f"{nf}\n")
        for f in np.logspace(1, -1, nf):
            d = r.normal(size=8) * 1e-3
            e = np.abs(r.normal(size=8)) * 1e-4 + 1e-5
            out.append(_fmt([f, *d, *e]) + "\n")
    out.append("VTF    2\n")
    out.append("101    101    0.0    0.0\n2\n")
    for f in (1.0, 0.1):
        out.append(_fmt([f, *r.normal(size=4) * 0.1, *(np.abs(r.normal(size=4)) * 0.01 + 1e-3)]) + "\n")
    out.append("102    102    1.0    0.0\n0\n")
    out.append("PT    1\n1    1    0.0    0.0\n2\n")
    for f in (1.0, 0.1):
        out.append(_fmt([f, *r.normal(size=4), *(np.abs(r.normal(size=4)) * 0.01 + 1e-3)]) + "\n")
    out.append("END\n")
    path.write_text("".join(out))


def make_distortion(path: Path, ids=(1, 2, 3, 4)):
    lines = [f"{len(ids)}\n"]
    for sid in ids:
        c = CP[sid]
        lines.append(f"{sid}  {c[0][0]:.8E}  {c[0][1]:.8E}  {c[1][0]:.8E}  {c[1][1]:.8E}\n")
    path.write_text("".join(lines))


def make_cnv(path: Path):
    hdr = "Iter#  Retrial#  Alpha  Beta  Damp  Roughness  Distortion  Misfit  RMS  ObjFunc\n"
    rms = [9.0, 5.0, 3.2, 2.1, 1.6, 1.3, 1.2, 1.05, 1.07]       # best = iter 7, last = 8
    rows = [f"{i}  0  10.0  1.0  0.0  5.0  0.1  {r * r:.4f}  {r:.4f}  100.0\n"
            for i, r in enumerate(rms)]
    path.write_text(hdr + "".join(rows))


def C_of(sid):
    return np.eye(2) + np.asarray(CP[sid], dtype=float)


# --------------------------------------------------------------------------
def test_physics_remove_recovers_regional_impedance():
    r = np.random.default_rng(0)
    Z = (r.normal(size=(5, 2, 2)) + 1j * r.normal(size=(5, 2, 2))) * 1e-3
    C = C_of(1)
    Zobs = np.einsum("ik,fkj->fij", C, Z).reshape(5, 4)
    data = np.empty((5, 8))
    data[:, 0::2], data[:, 1::2] = Zobs.real, Zobs.imag
    err = np.full((5, 8), 1e-5)
    new, _ = fd.apply_matrix_to_mt(data, err, np.linalg.inv(C))
    back = (new[:, 0::2] + 1j * new[:, 1::2]).reshape(5, 2, 2)
    assert np.allclose(back, Z, rtol=0, atol=1e-15)


def test_error_propagation_matches_monte_carlo():
    r = np.random.default_rng(1)
    data = r.normal(size=(1, 8)) * 1e-3
    err = np.abs(r.normal(size=(1, 8))) * 1e-4 + 2e-5
    M = np.linalg.inv(C_of(2))
    _, e_new = fd.apply_matrix_to_mt(data, err, M)
    n = 400000
    draws = data + r.normal(size=(n, 8)) * err              # independent errors
    mc = np.empty((n, 8))
    z = (draws[:, 0::2] + 1j * draws[:, 1::2]).reshape(n, 2, 2)
    zn = np.einsum("ik,nkj->nij", M, z).reshape(n, 4)
    mc[:, 0::2], mc[:, 1::2] = zn.real, zn.imag
    assert np.allclose(mc.std(axis=0), e_new[0], rtol=0.02)


def test_keep_errors_and_bad_mode():
    d = np.ones((2, 8))
    e = np.full((2, 8), 0.1)
    _, e2 = fd.apply_matrix_to_mt(d, e, np.eye(2) * 2.0, "keep")
    assert np.array_equal(e2, e)
    try:
        fd.apply_matrix_to_mt(d, e, np.eye(2), "bogus")
    except ValueError:
        return
    raise AssertionError("bad error_mode not rejected")


def _run_distcorr(tmp, **kw):
    obs = tmp / "observe.dat"
    dis = tmp / "distortion_iter7.dat"
    ids = kw.pop("ids", (1, 2, 3, 4))
    legacy = kw.pop("legacy", False)
    make_observe(obs, legacy=legacy)
    make_distortion(dis, ids)
    parsed = fem.read_observe_dat(str(obs), compute_mt_derived=False, bootstrap_n=0)
    id_list, C = fd.load_distortion(dis)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rows = fd.distcorr(parsed, id_list, C, **kw)
    return obs, parsed, rows, w


def test_distcorr_statuses_and_values():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        obs, parsed, rows, _ = _run_distcorr(tmp, mode="remove")
        st = {r["id"]: r["status"] for r in rows}
        assert st == {1: "corrected", 2: "corrected", 3: "identity", 4: "singular"}, st
        orig = fem.read_observe_dat(str(obs), compute_mt_derived=False, bootstrap_n=0)
        mt_new = parsed["blocks"][0]["sites"]
        mt_old = orig["blocks"][0]["sites"]
        for k, sid in enumerate((1, 2, 3, 4)):
            if st[sid] == "corrected":
                exp, _ = fd.apply_matrix_to_mt(mt_old[k]["data"], mt_old[k]["error"],
                                               np.linalg.inv(C_of(sid)))
                assert np.allclose(mt_new[k]["data"], exp, rtol=1e-13, atol=0)
                assert not np.allclose(mt_new[k]["data"], mt_old[k]["data"])
            else:
                assert np.array_equal(mt_new[k]["data"], mt_old[k]["data"])
                assert np.array_equal(mt_new[k]["error"], mt_old[k]["error"])
        # VTF / PT untouched
        for b in (1, 2):
            for sn, so in zip(parsed["blocks"][b]["sites"], orig["blocks"][b]["sites"]):
                assert np.array_equal(sn["data"], so["data"])


def test_apply_after_remove_round_trip():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        obs, parsed, rows, _ = _run_distcorr(tmp, mode="remove")
        ids, C = fd.load_distortion(tmp / "distortion_iter7.dat")
        fd.distcorr(parsed, ids, C, mode="apply")
        orig = fem.read_observe_dat(str(obs), compute_mt_derived=False, bootstrap_n=0)
        for sn, so in zip(parsed["blocks"][0]["sites"], orig["blocks"][0]["sites"]):
            assert np.allclose(sn["data"], so["data"], rtol=1e-9, atol=1e-18)
            # (errors do not round-trip: propagation adds up twice)


def test_missing_site_and_unused_row_warn():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        # distortion file lacks site 3 and 4
        _, parsed, rows, w = _run_distcorr(tmp, ids=(1, 2))
        st = {r["id"]: r["status"] for r in rows}
        assert st[3] == "no distortion row" and st[4] == "no distortion row"
        assert any("without a row" in str(x.message) for x in w)
    with tempfile.TemporaryDirectory() as t:
        try:
            _run_distcorr(Path(t), ids=(1, 2), missing="error")
        except RuntimeError:
            return
        raise AssertionError("missing='error' did not stop")


def test_no_matching_ids_raises():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        make_observe(tmp / "observe.dat")
        (tmp / "d.dat").write_text("1\n77  0.1  0.0  0.0  0.1\n")
        parsed = fem.read_observe_dat(str(tmp / "observe.dat"), compute_mt_derived=False,
                                      bootstrap_n=0)
        ids, C = fd.load_distortion(tmp / "d.dat")
        try:
            fd.distcorr(parsed, ids, C)
        except RuntimeError as e:
            assert "MATCH_BY" in str(e)
            return
        raise AssertionError("no-match case did not raise")


def test_legacy_layout_uses_order():
    with tempfile.TemporaryDirectory() as t:
        _, parsed, rows, _ = _run_distcorr(Path(t), legacy=True)
        assert [r["status"] for r in rows] == ["corrected", "corrected", "identity", "singular"]
        assert [r["id"] for r in rows] == [1, 2, 3, 4]


def test_order_needs_equal_counts():
    with tempfile.TemporaryDirectory() as t:
        try:
            _run_distcorr(Path(t), legacy=True, ids=(1, 2))
        except RuntimeError as e:
            assert "equal numbers" in str(e)
            return
        raise AssertionError("order mode with different counts did not raise")


def test_owner_column_layout_and_dummy_vtf():
    """OWNER_ELEMENT -1 layout: 'ID H-ID owner X Y' (MT: 6 tokens with E-type)."""
    row = _fmt([1.0, 2e-3, 1e-3, 2.5e-2, -2e-2, -2.2e-2, 1.9e-2, -1e-3, 2e-3,
                *[2e-4] * 8]) + "\n"
    txt = ("MT    2\n"
           "1    101    1    0    0.0    0.0\n1\n" + row +
           "2    102    1    0    1.0    0.0\n1\n" + row +
           "VTF    2\n"
           "101    101    0    0.0    0.0\n0\n"
           "102    102    0    1.0    0.0\n0\nEND\n")
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / "observe.dat").write_text(txt)
        make_distortion(tmp / "distortion_iter7.dat", ids=(1, 2))
        make_cnv(tmp / "femtic.cnv")
        old = _set_config(tmp)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rc = fd.main()
        finally:
            for k, v in old.items():
                setattr(fd, k, v)
        assert rc == 0
        out = (tmp / "observe_distcorr_iter7.dat").read_text().splitlines()
        for line in ("1    101    1    0    0.0    0.0", "2    102    1    0    1.0    0.0",
                     "101    101    0    0.0    0.0", "102    102    0    1.0    0.0"):
            assert line in out, line
        assert out[-1] == "END"


def test_select_iteration():
    with tempfile.TemporaryDirectory() as t:
        make_cnv(Path(t) / "femtic.cnv")
        assert fd.select_iteration(t, "best", ["femtic.cnv"])[0] == 7
        assert fd.select_iteration(t, "last", ["femtic.cnv"])[0] == 8
        assert fd.select_iteration(t, 3, ["femtic.cnv"]) == (3, None)
        assert fd.select_iteration(t, "5", ["femtic.cnv"]) == (5, None)
        b = fd.select_iteration(t, "best", ["femtic.cnv"])
        assert abs(b[1] - 1.05) < 1e-12


def test_distortion_type_check():
    with tempfile.TemporaryDirectory() as t:
        h5 = Path(t) / "results_iter7.h5"
        assert "not found" in fd.check_distortion_type(h5)
        h5.write_bytes(b"x")
        orig = fem.read_results_distortion
        try:
            fem.read_results_distortion = lambda p, out=False: {
                "type": 1, "type_name": "ESTIMATE_DISTORTION_MATRIX_DIFFERENCE"}
            assert fd.check_distortion_type(h5) is None
            fem.read_results_distortion = lambda p, out=False: {
                "type": 2, "type_name": "ESTIMATE_GAINS_AND_ROTATIONS"}
            try:
                fd.check_distortion_type(h5)
            except RuntimeError as e:
                assert "GAINS_AND_ROTATIONS" in str(e)
            else:
                raise AssertionError("type 2 not rejected")
            fem.read_results_distortion = lambda p, out=False: None
            assert "no /distortion" in fd.check_distortion_type(h5)
        finally:
            fem.read_results_distortion = orig


def _set_config(tmp, **kw):
    cfg = dict(WORK_DIR=str(tmp), OBSERVE_FILE="observe.dat", CNV_FILE=["femtic.cnv"],
               ITERATION="best", DISTORTION_FILE="distortion_iter{iter}.dat",
               OUTPUT_FILE="observe_distcorr_iter{iter}.dat",
               REPORT_FILE="observe_distcorr_iter{iter}_report.txt",
               RESULTS_H5_FILE="results_iter{iter}.h5", MODE="remove",
               ERROR_MODE="propagate", MATCH_BY="auto", MISSING_SITES="keep",
               MAX_COND=1.0e6, VERIFY_OUTPUT=True, PLOT=False)
    cfg.update(kw)
    old = {k: getattr(fd, k) for k in cfg}
    for k, v in cfg.items():
        setattr(fd, k, v)
    return old


def test_main_end_to_end():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        make_observe(tmp / "observe.dat")
        make_distortion(tmp / "distortion_iter7.dat")
        make_cnv(tmp / "femtic.cnv")
        before = (tmp / "observe.dat").read_bytes()
        old = _set_config(tmp)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rc = fd.main()
        finally:
            for k, v in old.items():
                setattr(fd, k, v)
        assert rc == 0
        assert (tmp / "observe.dat").read_bytes() == before            # original untouched
        out = tmp / "observe_distcorr_iter7.dat"
        assert out.is_file()
        rep = (tmp / "observe_distcorr_iter7_report.txt").read_text()
        assert all(ord(ch) < 128 for ch in rep)
        assert "corrected" in rep and "singular" in rep and "identity" in rep
        assert (tmp / "femtic_distcorr_summary.md").is_file()
        # station lines, END preserved
        txt_in = before.decode().splitlines()
        txt_out = out.read_text().splitlines()
        assert txt_out[0] == txt_in[0] and txt_out[-1] == "END"
        assert len(txt_out) == len(txt_in)
        assert "1    1    0.0    0.0" in txt_out and "102    102    1.0    0.0" in txt_out


def test_main_refuses_to_overwrite_input():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        make_observe(tmp / "observe.dat")
        make_distortion(tmp / "distortion_iter7.dat")
        make_cnv(tmp / "femtic.cnv")
        old = _set_config(tmp, OUTPUT_FILE="observe.dat")
        try:
            try:
                fd.main()
            except ValueError as e:
                assert "never overwritten" in str(e)
                return
        finally:
            for k, v in old.items():
                setattr(fd, k, v)
        raise AssertionError("output == input not refused")


def test_verify_output_detects_changes():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        obs, parsed, rows, _ = _run_distcorr(tmp)
        out = tmp / "out.dat"
        fem.write_observe_dat(parsed, str(out))
        ids, C = fd.load_distortion(tmp / "distortion_iter7.dat")
        assert fd.verify_output(str(obs), str(out), ids, C, rows,
                                mode="remove", error_mode="propagate") == []
        # tamper with one MT value of site 1 (first data row, first column after freq)
        lines = out.read_text().splitlines()
        i = next(k for k, l in enumerate(lines) if l.startswith("1    1    0.0"))
        row = lines[i + 2].split()
        row[1] = "9.99999999E+00"
        lines[i + 2] = "    ".join(row)
        out.write_text("\n".join(lines) + "\n")
        probs = fd.verify_output(str(obs), str(out), ids, C, rows,
                                 mode="remove", error_mode="propagate")
        assert probs and "site 1" in probs[0]


def test_report_is_ascii():
    with tempfile.TemporaryDirectory() as t:
        _, _, rows, _ = _run_distcorr(Path(t))
        rep = fd.format_report(rows, iteration=7, nrms=1.05, mode="remove",
                               error_mode="propagate", observe="a", distortion="b",
                               output="c")
        assert all(ord(ch) < 128 for ch in rep)


def test_select_plot_sites():
    with tempfile.TemporaryDirectory() as tdir:
        _, _, rows, _ = _run_distcorr(Path(tdir))
    # statuses: ids 1, 2 corrected, 3 identity, 4 singular; ||C-I||: site 2 > site 1
    assert fd.select_plot_sites(rows, 1) == [1]
    assert fd.select_plot_sites(rows, 6) == [0, 1]
    assert fd.select_plot_sites(rows, "all") == [0, 1]
    assert fd.select_plot_sites(rows, None) == [] and fd.select_plot_sites(rows, 0) == []
    assert fd.select_plot_sites(rows, []) == []
    assert fd.select_plot_sites(rows, [3, "2"]) == [2, 1]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert fd.select_plot_sites(rows, ["nope"]) == []
    assert any("not among" in str(x.message) for x in w)


def test_color_by_component():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    x = [1.0, 2.0]
    ax.plot(x, [1, 2], "-", color="C0", label="XY")        # original, comp 0
    ax.plot(x, [2, 3], "-", color="C1", label="YX")        # original, comp 1
    ax.plot(x, [1, 2.2], "--", color="C2", label="_a")     # new, comp 0 (wrong colour)
    ax.plot(x, [2, 3.2], "--", color="C3", label="_b")     # new, comp 1 (wrong colour)
    ax.legend()
    fd._color_by_component(ax, 2)
    c = [ln.get_color() for ln in ax.lines]
    assert c[0] == c[2] and c[1] == c[3] and c[0] != c[1]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["XY", "YX"]
    plt.close(fig)


def test_plot_files_and_titles():
    import matplotlib
    matplotlib.use("Agg")
    with tempfile.TemporaryDirectory() as tdir:
        tmp = Path(tdir)
        obs, parsed, rows, _ = _run_distcorr(tmp)
        out = tmp / "new.dat"
        fem.write_observe_dat(parsed, str(out))
        cap = []
        orig_save = fd._save_figure

        def spy(fig, stem, formats, dpi):
            cap.append((Path(stem).name, fig.get_suptitle()))
            return orig_save(fig, stem, formats, dpi)

        fd._save_figure = spy
        try:
            saved = fd.plot_orig_vs_corrected(
                str(obs), str(out), rows, stem=tmp / "p", iteration=7, sites=1,
                formats=["png", "pdf"], dpi=60)
            names = sorted(Path(s).name for s in saved)
            assert names == ["p_overview.pdf", "p_overview.png", "p_site2.pdf", "p_site2.png"], names
            for s in saved:
                assert Path(s).stat().st_size > 1000
            assert Path(saved[0]).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
            assert "original (solid)" in cap[0][1] and "all MT sites" in cap[0][1]
            assert "site 2" in cap[1][1] and "cond(C)" in cap[1][1]
            # titles: none / text with placeholders / dict
            cap.clear()
            fd.plot_orig_vs_corrected(str(obs), str(out), rows, stem=tmp / "q", iteration=7,
                                      sites=1, formats=["png"], dpi=50, title=None)
            assert [c[1] for c in cap] == ["", ""]
            cap.clear()
            fd.plot_orig_vs_corrected(str(obs), str(out), rows, stem=tmp / "r", iteration=7,
                                      sites=1, formats=["png"], dpi=50,
                                      title={"site": "S {site} it{iter}", "default": None})
            assert [c[1] for c in cap] == ["", "S 2 it7"]
            # mode "apply" wording, no overview, explicit site list, rho only
            cap.clear()
            fd.plot_orig_vs_corrected(str(obs), str(out), rows, stem=tmp / "s", iteration=7,
                                      mode="apply", overview=False, sites=["1"], what="rho",
                                      formats=["png"], dpi=50)
            assert len(cap) == 1 and "distorted (dashed)" in cap[0][1]
        finally:
            fd._save_figure = orig_save


def test_main_plot_step_and_failure_isolated():
    with tempfile.TemporaryDirectory() as tdir:
        tmp = Path(tdir)
        make_observe(tmp / "observe.dat")
        make_distortion(tmp / "distortion_iter7.dat")
        make_cnv(tmp / "femtic.cnv")
        old = _set_config(tmp, PLOT=True, PLOT_OVERVIEW=True, PLOT_SITES=1,
                          PLOT_WHAT=["rho", "phase"], PLOT_COMPS="xy,yx", PLOT_ERRORS=True,
                          PLOT_OVERVIEW_NSITES=None, PLOT_PERIOD_LIMITS=None,
                          PLOT_FILE="observe_distcorr_iter{iter}_plot", PLOT_FORMATS=["png"],
                          PLOT_DPI=50, PLOT_TITLE="auto", SHOW=False)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                assert fd.main() == 0
            assert (tmp / "observe_distcorr_iter7_plot_overview.png").is_file()
            assert (tmp / "observe_distcorr_iter7_plot_site2.png").is_file()
            # a failing plot step does not undo the output file and returns 0
            (tmp / "observe_distcorr_iter7.dat").unlink()
            fd.PLOT_WHAT = ["bogus"]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                assert fd.main() == 0
            assert (tmp / "observe_distcorr_iter7.dat").is_file()
        finally:
            for k, v in old.items():
                setattr(fd, k, v)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for n, f in tests:
        try:
            f()
            print(f"PASS  {n}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {n}: {type(exc).__name__}: {exc}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
