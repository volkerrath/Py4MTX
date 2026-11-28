#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_modem_data_misfit_measures.py

Pytest checks of the choice of the plotted measures of data fit (2026-11-27)
in modem_data_misfit.py: the settings PANEL_MEASURES,
PANEL_MEASURES_FONTSIZE, HIST_SCALE_MEASURE and SITE_MEASURES, the panel text
boxes (the defaults give the text of before), the histogram curve, the
per-site table / ordering / titles, and main() with invalid values.
Synthetic data; ``modem`` is replaced by a stand-in when it is not
importable. Needs misfit_measures.py and inverse.py (py4mt/modules).

Run, from the folder with modem_data_misfit.py::

    python -m pytest -q test_modem_data_misfit_measures.py

Provenance
----------
Author      : Claude (Claude Sonnet 5.5, Anthropic)
Generated   : 2026-11-27
Notice      : This code is AI-generated. Review and test it before any
              production use.

Changelog
---------
2026-11-27  Claude Sonnet 5.5 (Anthropic), AI-generated, review before use:
            first version.
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for _cand in (os.path.join(HERE, "..", "..", "modules"),
              os.path.join(os.environ.get("PY4MTX_ROOT", ""), "py4mt", "modules")):
    if os.path.isfile(os.path.join(_cand, "misfit_measures.py")):
        sys.path.append(os.path.abspath(_cand))
        break

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

pytest.importorskip("misfit_measures")
try:                                            # pragma: no cover
    import modem  # noqa: F401
except Exception:
    sys.modules["modem"] = types.ModuleType("modem")

import modem_data_misfit as md  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def _residuals(n=400, seed=1):
    rng = np.random.default_rng(seed)
    r1 = rng.normal(0.3, 1.5, n)
    r1[:3] = (25.0, -30.0, 40.0)
    return 10.0 ** rng.uniform(-2, 2, n), r1, 0.5 * r1


def _boxes(fig):
    return [t.get_text() for ax in fig.axes for t in ax.texts
            if t.get_bbox_patch() is not None]


def _hist(**kw):
    f, r1, r2 = _residuals()
    return md.plot_residual_histograms(
        "Z", f, r1, r2, has_syn=False, per_decade=1, axis="frequency", ncols=3,
        panel_size=3.0, limit=10.0, bin_width=0.5, min_n=10 ** 6, **kw)


def _qq(**kw):
    f, r1, r2 = _residuals()
    return md.plot_qq_bands(
        "Z", f, r1, r2, has_syn=False, per_decade=1, axis="frequency", ncols=3,
        panel_size=3.0, limit=10.0, confidence=0.95, min_n=10 ** 6, **kw)


def _cross(**kw):
    rng = np.random.default_rng(2)
    obs = rng.normal(0.0, 1.0, 100)
    cal = obs + rng.normal(0.02, 0.15, 100)
    p = dict(datatype="Z", component="ZXY", part="re", label="Re ZXY",
             freq=10.0 ** rng.uniform(-2, 2, 100), obs=obs, cal=cal, ref=cal,
             err=np.full(100, 0.1), eps=np.zeros(100), n_dropped=0)
    fig = md.plot_crossplot_datatype(
        "Z", [p], title_extra="t", has_syn=False, ncols=2, panel_size=3.0,
        cmap="turbo", color_range=None, marker_size=8.0, alpha=0.8, log_rho=True,
        show_errorbars=False, equal_aspect=True, **kw)
    return fig, (obs - cal) / 0.1


def _info():
    vals = {"A": (1.0, 3.0, 0.1), "B": (2.0, 1.0, -1.5), "C": (3.0, 2.0, 0.4)}
    return {s: dict(n=20, nrms=a, r_med=b, mean=c, x=0.0, y=0.0)
            for s, (a, b, c) in vals.items()}


def test_default_settings():
    assert md.PANEL_MEASURES is None and md.PANEL_MEASURES_FONTSIZE == 8.0
    assert md.HIST_SCALE_MEASURE == "nrms" and md.SITE_MEASURES == "nrms"


def test_default_boxes():
    _, r1, _ = _residuals()
    rms = np.sqrt(np.mean(r1 ** 2))
    out = int(np.sum(np.abs(r1) > 10))
    fig, _ = _hist()
    assert _boxes(fig) == [f"n={r1.size}\nRMS={rms:.2f}\nout: {out}"]
    s, i = md.qq_quartile_line(np.sort(r1))
    fig, _ = _qq()
    assert _boxes(fig) == [f"n={r1.size}\nslope={s:.2f}\nicpt={i:.2f}"]
    fig, r = _cross()
    assert _boxes(fig) == [f"n=100\nnRMS={np.sqrt(np.mean(r ** 2)):.2f}"]


@pytest.mark.parametrize("maker", [_cross, _hist, _qq])
def test_chosen_measures_and_font_size(maker):
    fig, _ = maker(measures=["r_med", "kurt"], box_fontsize=6.0)
    texts = _boxes(fig)
    assert len(texts) == 1
    assert [ln.split("=")[0] for ln in texts[0].split("\n")] == ["rMED", "kurt"]
    got = [t.get_fontsize() for ax in fig.axes for t in ax.texts
           if t.get_bbox_patch() is not None]
    assert got == [6.0]
    plt.close("all")
    assert _boxes(maker(measures=False)[0]) == []


@pytest.mark.parametrize("scale, label", [("nrms", "N(0, RMS^2)"),
                                          ("r_med", "N(0, rMED^2)"), (None, None)])
def test_histogram_scale(scale, label):
    _, r1, _ = _residuals()
    fig, _ = _hist(scale_measure=scale)
    red = [ln for ln in fig.axes[0].lines if ln.get_color() == "red"]
    if scale is None:
        assert red == []
        return
    assert [ln.get_label() for ln in red] == [label]
    want = md.fit_measures(r1)[scale]
    edges = sorted(ln.get_xdata()[0] for ln in fig.axes[0].lines
                   if ln.get_linestyle() == "--")
    assert edges == pytest.approx([-want, want])


def test_site_values_order_and_table(monkeypatch):
    n = 30
    rng = np.random.default_rng(3)
    obs = rng.normal(0, 1, n)
    ds = dict(freq=np.ones(n), datatype_name=np.array(["Full_Impedance"] * n),
              obs_re=obs, ref_re=np.zeros(n), err_re=np.ones(n),
              obs_im=obs, ref_im=np.zeros(n), err_im=np.ones(n),
              site_id=np.array(["A"] * n), site_x=np.zeros(n), site_y=np.zeros(n))
    if not hasattr(md.mdm, "modem_is_real_only_datatype"):      # stand-in
        monkeypatch.setattr(md.mdm, "modem_is_real_only_datatype",
                            lambda name: False, raising=False)
    info = md.site_nrms(ds)["A"]
    want = md.fit_measures(np.concatenate([obs, obs]))
    for key in ("nrms", "r_mae", "r_med", "q95", "mean", "std", "skew", "kurt"):
        assert info[key] == pytest.approx(want[key]), key
    assert md.select_sites(_info(), 2, "worst", 10) == ["C", "B"]
    assert md.select_sites(_info(), 2, "worst", 10, measure="r_med") == ["A", "C"]
    assert md.select_sites(_info(), 1, "worst", 10, measure="mean") == ["B"]
    assert md.select_sites(_info(), 1, "best", 10, measure="mean") == ["A"]
    assert md.format_site_nrms(_info()) == md.format_site_nrms(_info(), ("nrms",))
    head = md.format_site_nrms(_info(), ("r_med", "nrms")).split("\n")[1].split()
    assert head[-2:] == ["nrms", "r_med"]


@pytest.mark.parametrize("measures, want", [
    ("nrms", "nRMS = 1.50, n = 4"),
    (["r_med", "mean"], "rMED = 0.70, mean = -0.20, n = 4"),
    (["nrms", "r_med", "mean"], "n = 4\nnRMS = 1.50, rMED = 0.70, mean = -0.20")])
def test_curve_title(measures, want, monkeypatch):
    monkeypatch.setattr(md, "SITE_MEASURES", measures)
    try:
        dv, markers, comp_class = md._curve_modules()
    except ImportError as exc:
        pytest.skip(f"data_viz not importable: {exc}")
    per = np.logspace(-1, 1, 4)
    series = {"rho_xy": dict(panel="rho", freq=1.0 / per, period=per, obs=per,
                             err=0.1 * per, cal=per[None, :])}
    info = dict(n=4, nrms=1.5, r_med=0.7, mean=-0.2, x=float("nan"), y=float("nan"))
    cfg = md._curves_cfg()
    fig = md.plot_site_curves("A", series, info, M=1, layers=[], cfg=cfg,
                              title_extra="t", dv=dv, markers=markers,
                              comp_class=comp_class, rng=np.random.default_rng(0))
    assert want in fig._suptitle.get_text()


@pytest.mark.parametrize("name, value", [
    ("PANEL_MEASURES", "chi2"), ("HIST_SCALE_MEASURE", "kurt"),
    ("SITE_MEASURES", []), ("PANEL_MEASURES_FONTSIZE", -1)])
def test_main_refuses_invalid_values(name, value, monkeypatch, capsys):
    monkeypatch.setattr(md, name, value)
    monkeypatch.setattr(md, "SHOW", False)
    assert md.main(["single", "nowhere.dat", "nowhere.dat"]) == 1
    assert name in capsys.readouterr().out
