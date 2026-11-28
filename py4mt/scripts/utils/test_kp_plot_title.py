#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_kp_plot_title.py

Pytest checks of the ``title`` option of kp_plot.plot_kp() / ``--title`` of
kp_plot.py (added 2026-11-11). Synthetic Kp series; ``requests`` is replaced by
an empty module when it is not installed (nothing is downloaded).

Run, from the folder with kp_plot.py::

    python -m pytest -q test_kp_plot_title.py

Provenance
----------
Author      : Claude (Claude Sonnet 5.5, Anthropic)
Generated   : 2026-11-11
Notice      : This code is AI-generated. Review and test it before any
              production use.
"""
import importlib.util
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
AUTO = "Planetary geomagnetic Kp index (GFZ)"


@pytest.fixture(scope="module")
def kp():
    try:
        import requests  # noqa: F401
    except ImportError:                          # pragma: no cover
        sys.modules["requests"] = types.ModuleType("requests")
    spec = importlib.util.spec_from_file_location(
        "kp_plot_titletest", os.path.join(HERE, "kp_plot.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def df():
    idx = pd.date_range("2023-04-01", periods=24, freq="3h", tz="UTC")
    return pd.DataFrame({"Kp": np.linspace(0, 8, 24)}, index=idx)


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def test_default_is_unchanged(kp, df):
    assert kp.plot_kp(df).get_title() == AUTO
    assert kp.plot_kp(df, title="auto").get_title() == AUTO
    assert kp.plot_kp(df, title=" Auto ").get_title() == AUTO
    assert kp.plot_kp(df, title=True).get_title() == AUTO


@pytest.mark.parametrize("none", [None, False, "", "  "])
def test_no_title(kp, df, none):
    assert kp.plot_kp(df, title=none).get_title() == ""


def test_text_and_placeholders(kp, df):
    assert kp.plot_kp(df, title="Mine").get_title() == "Mine"
    assert kp.plot_kp(df, title="Kp {start} to {end}").get_title() == \
        "Kp 2023-04-01 to 2023-04-03"
    assert kp.plot_kp(df, title="x {nope}").get_title() == "x {nope}"


def test_cli_option(kp):
    src = open(os.path.join(HERE, "kp_plot.py"), encoding="utf-8").read()
    assert '"--title", default="auto"' in src
    assert "title=args.title" in src


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))
