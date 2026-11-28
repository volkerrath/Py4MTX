#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_modem_data_misfit_titles.py

Pytest checks of the title and colour bar settings of modem_data_misfit.py
(added 2026-11-11): FIGURE_TITLE ("auto" / none / text with {placeholders} /
dict per kind of figure), PANEL_TITLES, CROSSPLOT_CBAR_LOCATION and
CROSSPLOT_CBAR_SHRINK, and the keywords of the plot functions behind them;
and (2026-11-11) the SHOW_RMS2 switch: no nRMS2 / RMS2 in the plots and the
table notes unless it is True.
Synthetic panels; ``modem`` and ``inverse`` are replaced by stand-ins when
they are not importable.

Run, from the folder with modem_data_misfit.py::

    python -m pytest -q test_modem_data_misfit_titles.py

Provenance
----------
Author      : Claude (Claude Sonnet 5.5, Anthropic)
Generated   : 2026-11-11
Notice      : This code is AI-generated. Review and test it before any
              production use.

Changelog
---------
2026-11-27  Claude Sonnet 5.5 (Anthropic), AI-generated, review before use:
            adds py4mt/modules to sys.path, as modem_data_misfit.py now
            imports misfit_measures.
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# py4mt/modules: the script imports misfit_measures from there
for _cand in (os.path.join(HERE, "..", "..", "modules"),
              os.path.join(os.environ.get("PY4MTX_ROOT", ""), "py4mt", "modules")):
    if os.path.isfile(os.path.join(_cand, "misfit_measures.py")):
        sys.path.append(os.path.abspath(_cand))
        break

import matplotlib  # noqa: E402

matplotlib.use("Agg")

try:                                            # pragma: no cover
    import inverse  # noqa: F401
except ImportError:
    _m = types.ModuleType("inverse")
    _m.nrms = lambda a: float(np.sqrt(np.mean(np.square(a))))
    _m.r_mae = _m.r_med = _m.q95 = lambda a: 0.0
    sys.modules["inverse"] = _m
try:                                            # pragma: no cover
    import modem  # noqa: F401
except Exception:
    sys.modules["modem"] = types.ModuleType("modem")

import modem_data_misfit as fm  # noqa: E402


# ---------------------------------------------------------------------------
# helpers (_resolve_title, _check_cbar_location)
# ---------------------------------------------------------------------------
def test_resolve_title_forms(capsys):
    r = fm._resolve_title
    assert r("auto", "x", "AUTO") == "AUTO"
    assert r(True, "x", "AUTO") == "AUTO"
    assert r("AUTO ", "x", None) is None
    for none in (None, False, "", "  "):
        assert r(none, "x", "AUTO") == ""
    assert r("t {title} {datatype}", "crossplot", None,
             title="run", datatype="MT") == "t run MT"
    assert r("{key}", "qq", None) == "qq"
    assert r("n={n:.2f}", "k", None, n=1.23456) == "n=1.23"
    assert r("{nope}", "k", None) == "{nope}"                  # as it stands
    assert "cannot fill in" in capsys.readouterr().out
    d = {"crossplot": "C", "qq": None, "default": "auto"}
    assert r(d, "crossplot", "A") == "C"
    assert r(d, "qq", "A") == ""
    assert r(d, "histogram", "A") == "A"
    assert r({"a": "x"}, "b", "A") == "A"                      # no default


@pytest.mark.parametrize("inp, out", [(None, None), (False, None), ("", None),
                                      ("panel", None), ("default", None),
                                      ("right", "right"), (" Top ", "top"),
                                      ("LEFT", "left"), ("bottom", "bottom")])
def test_check_cbar_location(inp, out):
    assert fm._check_cbar_location(inp) == out


@pytest.mark.parametrize("bad", ["middle", "up", 3])
def test_check_cbar_location_errors(bad):
    with pytest.raises(ValueError):
        fm._check_cbar_location(bad)


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------
def test_settings_defaults():
    assert fm.FIGURE_TITLE == "auto"
    assert fm.PANEL_TITLES is True
    assert fm.CROSSPLOT_CBAR_LOCATION is None
    assert fm.CROSSPLOT_CBAR_SHRINK == 0.85
    assert fm._curves_cfg()["title"] == fm.FIGURE_TITLE


def test_curves_title_follows_setting(monkeypatch):
    monkeypatch.setattr(fm, "FIGURE_TITLE", {"curves": "site {site}"})
    assert fm._curves_cfg()["title"] == {"curves": "site {site}"}
    assert fm._resolve_title(fm._curves_cfg()["title"], "curves", "A",
                             site="S1", sid="S1", nrms="1.0") == "site S1"


# ---------------------------------------------------------------------------
# crossplot
# ---------------------------------------------------------------------------
def _panel(rng, label, n=40):
    f = np.logspace(-2, 2, n)
    obs = 10 ** rng.normal(1.0, 0.3, n)
    cal = obs * (1 + 0.05 * rng.standard_normal(n))
    err = 0.05 * obs
    return dict(label=label, component="rho_xy", freq=f, obs=obs, cal=cal,
                err=err, eps=0.02 * obs, ref=cal, n_dropped=0)


def _cross(**kw):
    rng = np.random.default_rng(3)
    panels = [_panel(rng, f"P{i}") for i in range(3)]
    args = dict(title_extra="RUN", has_syn=False, ncols=2, panel_size=3.0,
                cmap="turbo", color_range=None, marker_size=10.0, alpha=0.8,
                log_rho=True, show_errorbars=False, equal_aspect=True)
    args.update(kw)
    fig = fm.plot_crossplot_datatype("MT", panels, **args)
    fig.canvas.draw()
    return fig


def _sup(fig):
    return None if fig._suptitle is None else fig._suptitle.get_text()


def _bar(fig):
    return [a for a in fig.axes if a.get_ylabel() == "log10 frequency (Hz)"
            or a.get_xlabel() == "log10 frequency (Hz)"][0]


def _panels(fig):
    return [a for a in fig.axes if a.get_xlabel() == "observed"]


def _ext(a):
    bb = a.get_window_extent()
    return bb.x0, bb.y0, bb.x1, bb.y1


def test_crossplot_defaults():
    fig = _cross()
    assert _sup(fig) == "MT: observed vs. calculated, RUN"
    assert [a.get_title() for a in _panels(fig)] == ["P0", "P1", "P2"]
    x0, y0, x1, y1 = _ext(_bar(fig))
    assert x0 >= max(_ext(a)[2] for a in _panels(fig)) - 1 and y1 - y0 > x1 - x0


@pytest.mark.parametrize("spec, out", [("Mine", "Mine"), ("", None),
                                       (None, "MT: observed vs. calculated, RUN")])
def test_crossplot_figure_title(spec, out):
    fig = _cross(figure_title=spec)
    assert _sup(fig) == out


def test_crossplot_panel_titles_off():
    fig = _cross(panel_titles=False)
    assert all(a.get_title() == "" for a in _panels(fig))


@pytest.mark.parametrize("loc", ["right", "left", "top", "bottom"])
def test_crossplot_colorbar_sides(loc):
    fig = _cross(cbar_location=loc)
    bar, pan = _bar(fig), _panels(fig)
    x0, y0, x1, y1 = _ext(bar)
    px0 = min(_ext(a)[0] for a in pan)
    px1 = max(_ext(a)[2] for a in pan)
    py0 = min(_ext(a)[1] for a in pan)
    py1 = max(_ext(a)[3] for a in pan)
    if loc == "right":
        assert x0 >= px1 - 1 and y1 - y0 > x1 - x0
    elif loc == "left":
        assert x1 <= px0 + 1 and y1 - y0 > x1 - x0
    elif loc == "top":
        assert y0 >= py1 - 1 and x1 - x0 > y1 - y0
    else:
        assert y1 <= py0 + 1 and x1 - x0 > y1 - y0


def test_crossplot_colorbar_shrink():
    full = _ext(_bar(_cross(cbar_shrink=1.0)))
    half = _ext(_bar(_cross(cbar_shrink=0.5)))
    assert (half[3] - half[1]) == pytest.approx(0.5 * (full[3] - full[1]),
                                                rel=0.1)


def test_crossplot_bad_location():
    with pytest.raises(ValueError):
        _cross(cbar_location="middle")


# ---------------------------------------------------------------------------
# histograms and Q-Q
# ---------------------------------------------------------------------------
def _resid():
    rng = np.random.default_rng(5)
    f = np.logspace(-2, 2, 400)
    return f, rng.standard_normal(400), rng.standard_normal(400)


HIST = dict(has_syn=False, per_decade=1, axis="frequency", ncols=3,
            panel_size=2.5, min_n=5)


def _hist(**kw):
    f, r1, r2 = _resid()
    fig, _ = fm.plot_residual_histograms(
        "MT", f, r1, r2, limit=5.0, bin_width=0.5, run_label="RUN",
        **HIST, **kw)
    return fig


def _qq(**kw):
    f, r1, r2 = _resid()
    fig, _ = fm.plot_qq_bands(
        "MT", f, r1, r2, limit=5.0, confidence=0.95, run_label="RUN",
        **HIST, **kw)
    return fig


@pytest.mark.parametrize("make, word", [(_hist, "normalized residuals"),
                                        (_qq, "normal Q-Q")])
def test_hist_qq_titles(make, word):
    fig = make()
    assert word in _sup(fig) and "RUN" in _sup(fig)
    assert make(figure_title="Mine")._suptitle.get_text() == "Mine"
    assert make(figure_title="")._suptitle is None
    fig = make(panel_titles=False)
    assert all(a.get_title() == "" for a in fig.axes)
    fig = make()
    assert any(a.get_title() for a in fig.axes)


# ---------------------------------------------------------------------------
# through process_runs-style resolution of the dict / placeholders
# ---------------------------------------------------------------------------
def test_dict_keys_per_kind():
    spec = {"crossplot": "X {datatype}", "histogram": "H {group} {axis}",
            "qq": "", "default": "auto"}
    assert fm._resolve_title(spec, "crossplot", None, title="r",
                             datatype="MT") == "X MT"
    assert fm._resolve_title(spec, "histogram", None, title="r", group="g",
                             axis="period", per_decade=2) == "H g period"
    assert fm._resolve_title(spec, "qq", None) == ""
    assert fm._resolve_title(spec, "curves", None) is None


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))


# ---------------------------------------------------------------------------
# nRMS2 switched off (SHOW_RMS2, 2026-11-11): no modelling error is used
# ---------------------------------------------------------------------------
def _texts(fig):
    from matplotlib.text import Text

    out = [t.get_text() for t in fig.findobj(Text)]
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is not None:
            out += [t.get_text() for t in leg.get_texts()]
        out += [p.get_label() for p in ax.patches]
        out += [ln.get_label() for ln in ax.get_lines()]
    return [x for x in out if x]


def _has(fig, word):
    return any(word in t for t in _texts(fig))


def test_show_rms2_default_is_off():
    assert fm.SHOW_RMS2 is False


def test_crossplot_rms2_off_and_on():
    fig = _cross(has_syn=False, ensemble_mean=True)
    assert _has(fig, "nRMS=") and not _has(fig, "RMS2") and not _has(fig, "RMS1")
    assert any(a.get_ylabel() == "calculated (ensemble mean)" for a in fig.axes[:2])
    fig = _cross(has_syn=True)
    assert _has(fig, "nRMS1=") and _has(fig, "nRMS2=")
    fig = _cross(has_syn=False)                       # as before: single run
    assert any(a.get_ylabel() == "calculated" for a in fig.axes[:2])


@pytest.mark.parametrize("make", [_hist, _qq])
def test_hist_qq_rms2_off_and_on(make):
    fig = make()
    for w in ("RMS2", "slope2", "RMS1", "slope1"):
        assert not _has(fig, w), w
    f, r1, r2 = _resid()
    if make is _hist:
        fig, _ = fm.plot_residual_histograms(
            "MT", f, r1, r2, limit=5.0, bin_width=0.5, run_label="RUN",
            **dict(HIST, has_syn=True))
        assert _has(fig, "RMS2=") and _has(fig, "RMS2-normalized")
    else:
        fig, _ = fm.plot_qq_bands(
            "MT", f, r1, r2, limit=5.0, confidence=0.95, run_label="RUN",
            **dict(HIST, has_syn=True))
        assert _has(fig, "slope2=") and _has(fig, "RMS2-normalized")


def test_table_note_follows_rms2():
    assert "err+syn" in fm.fit_measure_note(True)
    assert "err+syn" not in fm.fit_measure_note(False)
    assert "RMS2" not in fm.fit_measure_note(False)
    assert fm.fit_measure_note(True) == fm.FIT_MEASURE_NOTE
