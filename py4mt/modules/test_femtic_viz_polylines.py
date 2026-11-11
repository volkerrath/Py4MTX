#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_femtic_viz_polylines.py

Pytest suite for the changes of 2026-11-11 in femtic_viz.py / femtic.py:

  * ``slice_intersection_polyline`` / ``slice_intersection_polylines``: the
    line in which two slices (map, ns, ew, vertical plane, dipping plane)
    meet, clipped to the ROI box and to the extent of the slices;
  * the ``polylines`` argument of ``plot_model_slices`` (CRS tags, slab
    projection on vertical sections, plan view on maps, legend, axis limits);
  * ``femtic.polylines_km_to_m`` (km settings of the ens script).

The rendering tests build a small synthetic tetrahedral model (cubes of 1 km
split into 6 tetrahedra, air above z = 0, a 10 Ohm.m conductor at E 2..5 km,
N 1..4 km, depth 1..3 km in a 100 Ohm.m background) in a temporary directory;
no FEMTIC files are needed. Expected line positions are computed here with
independent formulas (dense sampling, closed forms), not with the code under
test.

Run
---
From the folder containing femtic_viz.py, femtic.py and this file::

    python -m pytest -q test_femtic_viz_polylines.py

Provenance
----------
Author      : Claude (Claude Sonnet 5.5, Anthropic)
Generated   : 2026-11-11
Notice      : This code is AI-generated. Review and test it before any
              production use.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import femtic as fem  # noqa: E402
import femtic_viz as fviz  # noqa: E402

BOX = dict(xlim=[-6000.0, 6000.0], ylim=[-6000.0, 6000.0],
           zlim=[-1000.0, 6000.0])


# ---------------------------------------------------------------------------
# UTM helpers: the package's util.py when importable (needs h5py, pyproj),
# else a small stand-in (Krueger series, mm accuracy) so that the polyline
# tests of the lat/lon tag run in minimal environments as well.
# ---------------------------------------------------------------------------
def _stub_util():
    import types

    a, f, k0 = 6378137.0, 1.0 / 298.257223563, 0.9996
    n = f / (2.0 - f)
    A = a / (1.0 + n) * (1.0 + n ** 2 / 4.0 + n ** 4 / 64.0)
    al = [n / 2 - 2 * n ** 2 / 3 + 5 * n ** 3 / 16,
          13 * n ** 2 / 48 - 3 * n ** 3 / 5, 61 * n ** 3 / 240]
    be = [n / 2 - 2 * n ** 2 / 3 + 37 * n ** 3 / 96,
          n ** 2 / 48 + n ** 3 / 15, 17 * n ** 3 / 480]
    de = [2 * n - 2 * n ** 2 / 3 - 2 * n ** 3,
          7 * n ** 2 / 3 - 8 * n ** 3 / 5, 56 * n ** 3 / 15]
    q = 2.0 * math.sqrt(n) / (1.0 + n)

    def latlon_to_utm_zn(lat, lon, zone, northern):
        phi = math.radians(lat)
        lam = math.radians(lon - (6.0 * zone - 183.0))
        t = math.sinh(math.atanh(math.sin(phi))
                      - q * math.atanh(q * math.sin(phi)))
        xi0 = math.atan2(t, math.cos(lam))
        eta0 = math.atanh(math.sin(lam) / math.hypot(1.0, t))
        xi = xi0 + sum(al[j - 1] * math.sin(2 * j * xi0)
                       * math.cosh(2 * j * eta0) for j in (1, 2, 3))
        eta = eta0 + sum(al[j - 1] * math.cos(2 * j * xi0)
                         * math.sinh(2 * j * eta0) for j in (1, 2, 3))
        return (500000.0 + k0 * A * eta,
                (0.0 if northern else 1.0e7) + k0 * A * xi)

    def utm_to_latlon_zn(E_m, N_m, zone, northern):
        xi = (N_m - (0.0 if northern else 1.0e7)) / (k0 * A)
        eta = (E_m - 500000.0) / (k0 * A)
        xi0 = xi - sum(be[j - 1] * math.sin(2 * j * xi)
                       * math.cosh(2 * j * eta) for j in (1, 2, 3))
        eta0 = eta - sum(be[j - 1] * math.cos(2 * j * xi)
                         * math.sinh(2 * j * eta) for j in (1, 2, 3))
        chi = math.asin(math.sin(xi0) / math.cosh(eta0))
        phi = chi + sum(de[j - 1] * math.sin(2 * j * chi) for j in (1, 2, 3))
        lam = math.atan2(math.sinh(eta0), math.cos(xi0))
        return math.degrees(phi), math.degrees(lam) + 6.0 * zone - 183.0

    m = types.ModuleType("util")
    m.latlon_to_utm_zn = latlon_to_utm_zn
    m.utm_to_latlon_zn = utm_to_latlon_zn
    m.__doc__ = "test stand-in for util (UTM conversions only)"
    return m


def _utl():
    """Return util (real if importable, else the stand-in, registered in
    sys.modules because femtic.latlon_to_model imports it lazily)."""
    try:
        import util
        return util
    except Exception:
        stub = _stub_util()
        sys.modules["util"] = stub
        return stub


def test_stub_util_roundtrip_and_known_point():
    u = _stub_util()
    E, N = u.latlon_to_utm_zn(0.0, 9.0, 32, True)          # central meridian
    assert abs(E - 500000.0) < 1e-6 and abs(N) < 1e-6
    for lat, lon in ((45.9, 6.12), (36.0, 9.7), (-33.5, 11.2)):
        zone = 32 if lon > 6.0 else 31
        E, N = u.latlon_to_utm_zn(lat, lon, zone, lat >= 0)
        la, lo = u.utm_to_latlon_zn(E, N, zone, lat >= 0)
        assert abs(la - lat) < 1e-7 and abs(lo - lon) < 1e-7   # ~1 cm
    # one degree of latitude on the central meridian: 110.6 km (WGS84)
    _, N1 = u.latlon_to_utm_zn(45.0, 9.0, 32, True)
    _, N2 = u.latlon_to_utm_zn(46.0, 9.0, 32, True)
    assert 111000.0 < (N2 - N1) < 111200.0


# ---------------------------------------------------------------------------
# synthetic model
# ---------------------------------------------------------------------------
def _make_model(outdir, half=6000.0, h=1000.0, zmin=-1000.0, zmax=6000.0):
    xs = np.arange(-half, half + h / 2, h)
    ys = np.arange(-half, half + h / 2, h)
    zs = np.arange(zmin, zmax + h / 2, h)
    nx, ny, nz = len(xs), len(ys), len(zs)
    nodes = np.array([[xs[i], ys[j], zs[k]] for i in range(nx)
                      for j in range(ny) for k in range(nz)])

    def idx(i, j, k):
        return (i * ny + j) * nz + k

    tets = [(0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7),
            (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7)]
    conn = [[idx(i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1))
             for c in t]
            for i in range(nx - 1) for j in range(ny - 1)
            for k in range(nz - 1) for t in tets]
    conn = np.array(conn)
    cen = nodes[conn].mean(axis=1)
    reg = np.full(len(conn), 2)
    reg[cen[:, 2] < 0.0] = 0
    reg[(cen[:, 0] > 2000) & (cen[:, 0] < 5000) & (cen[:, 1] > 1000)
        & (cen[:, 1] < 4000) & (cen[:, 2] > 1000) & (cen[:, 2] < 3000)] = 3
    mesh = os.path.join(outdir, "mesh.dat")
    with open(mesh, "w") as f:
        f.write(f"TETRA\n{len(nodes)}\n")
        for n, p in enumerate(nodes):
            f.write(f"{n} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        f.write(f"{len(conn)}\n")
        for e, c in enumerate(conn):
            f.write(f"{e} {c[0]} {c[1]} {c[2]} {c[3]}\n")
    block = os.path.join(outdir, "resistivity_block.dat")
    with open(block, "w") as f:
        f.write(f"{len(conn)} 4\n")
        for e, r in enumerate(reg):
            f.write(f"{e} {r}\n")
        f.write("0 1.0e9 1.0e-2 1.0e12 1.0 1\n1 0.25 1.0e-2 1.0e12 1.0 1\n"
                "2 100.0 1.0e-2 1.0e12 1.0 0\n3 10.0 1.0e-2 1.0e12 1.0 0\n")
    return mesh, block


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    return _make_model(str(tmp_path_factory.mktemp("synth")))


def _render(model, slices, polylines=None, tmp=None, **kw):
    """plot_model_slices on the synthetic model; returns the figure."""
    mesh, block = model
    args = dict(
        model_file=block, mesh_file=mesh, slices=slices, clim=[0.0, 3.0],
        display_coords="model", horiz_km=True, depth_km=True,
        nrows=1, ncols=len(slices), panel_height=2.5, dpi=40, out=False,
        plot_file=os.path.join(tmp, "fig.png"))
    args.update(BOX)
    args.update(kw)
    if polylines is not None:
        args["polylines"] = polylines
    return fviz.plot_model_slices(**args)


def _panel_lines(fig, i, name):
    ax = fig.axes[i]
    return [ln for ln in ax.lines if ln.get_label() == name]


# ---------------------------------------------------------------------------
# slice intersection
# ---------------------------------------------------------------------------
def _on_plane(spec, pts, tol=1e-6):
    pt, nrm, _ = fviz._slice_frame(spec)
    return np.max(np.abs((pts - pt) @ nrm)) < tol


def test_map_ns_orthogonal():
    p = fviz.slice_intersection_polyline(
        dict(kind="map", z0=2000.0), dict(kind="ns", x0=3500.0), **BOX)
    assert p.shape == (2, 3)
    np.testing.assert_allclose(np.sort(p[:, 1]), [-6000.0, 6000.0], atol=1e-6)
    np.testing.assert_allclose(p[:, 0], 3500.0, atol=1e-9)
    np.testing.assert_allclose(p[:, 2], 2000.0, atol=1e-9)


def test_ns_ew_vertical_line_depth_is_zlim():
    p = fviz.slice_intersection_polyline(
        dict(kind="ns", x0=1000.0), dict(kind="ew", y0=-2000.0), **BOX)
    np.testing.assert_allclose(p[:, 0], 1000.0, atol=1e-9)
    np.testing.assert_allclose(p[:, 1], -2000.0, atol=1e-9)
    np.testing.assert_allclose(np.sort(p[:, 2]), [-1000.0, 6000.0], atol=1e-6)


def test_vertical_plane_with_map():
    # strike 45 deg through the origin, map at 3 km: the line runs NE and
    # leaves the box where x or y reaches 6 km (corner of the square).
    p = fviz.slice_intersection_polyline(
        dict(kind="plane", point=[0.0, 0.0, 0.0], strike=45.0, dip=90.0),
        dict(kind="map", z0=3000.0), **BOX)
    d = p[1] - p[0]
    np.testing.assert_allclose(d[:2] / np.linalg.norm(d[:2]),
                               np.array([1.0, 1.0]) / math.sqrt(2.0), atol=1e-9)
    np.testing.assert_allclose(np.abs(p[:, :2]).max(), 6000.0, atol=1e-6)
    np.testing.assert_allclose(p[:, 2], 3000.0, atol=1e-9)


def test_dipping_plane_with_map():
    # strike N, dip 45 towards E through the origin: plane x = z. At z = 1 km
    # the line is x = 1 km, z = 1 km, y free.
    pl = dict(kind="plane", point=[0.0, 0.0, 0.0], strike=0.0, dip=45.0)
    p = fviz.slice_intersection_polyline(pl, dict(kind="map", z0=1000.0), **BOX)
    np.testing.assert_allclose(p[:, 0], 1000.0, atol=1e-6)
    np.testing.assert_allclose(p[:, 2], 1000.0, atol=1e-6)
    np.testing.assert_allclose(np.sort(p[:, 1]), [-6000.0, 6000.0], atol=1e-6)


def test_two_vertical_planes_meet_in_vertical_line():
    a = dict(kind="plane", point=[0.0, 0.0, 0.0], strike=0.0, dip=90.0)
    b = dict(kind="plane", point=[0.0, 0.0, 0.0], strike=90.0, dip=90.0)
    p = fviz.slice_intersection_polyline(a, b, **BOX)
    np.testing.assert_allclose(p[:, :2], 0.0, atol=1e-9)


def test_parallel_and_same_plane_give_none():
    assert fviz.slice_intersection_polyline(
        dict(kind="map", z0=1000.0), dict(kind="map", z0=2000.0), **BOX) is None
    assert fviz.slice_intersection_polyline(
        dict(kind="map", z0=1000.0), dict(kind="map", z0=1000.0), **BOX) is None
    assert fviz.slice_intersection_polyline(      # same plane, other spec kind
        dict(kind="ns", x0=0.0),
        dict(kind="plane", point=[0.0, 5.0, 0.0], strike=0.0, dip=90.0),
        **BOX) is None


def test_line_outside_box_gives_none():
    assert fviz.slice_intersection_polyline(
        dict(kind="map", z0=9000.0), dict(kind="ns", x0=0.0), **BOX) is None
    assert fviz.slice_intersection_polyline(
        dict(kind="ns", x0=9000.0), dict(kind="ew", y0=0.0), **BOX) is None


def test_aliases_sn_we_equal_ns_ew():
    a = fviz.slice_intersection_polyline(
        dict(kind="sn", x0=100.0), dict(kind="we", y0=200.0), **BOX)
    b = fviz.slice_intersection_polyline(
        dict(kind="ns", x0=100.0), dict(kind="ew", y0=200.0), **BOX)
    np.testing.assert_allclose(a, b)


def test_n_vertices_evenly_spaced():
    p = fviz.slice_intersection_polyline(
        dict(kind="map", z0=2000.0), dict(kind="ns", x0=0.0), n=5, **BOX)
    assert p.shape == (5, 3)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    np.testing.assert_allclose(seg, seg[0])
    with pytest.raises(ValueError):
        fviz.slice_intersection_polyline(
            dict(kind="map", z0=2000.0), dict(kind="ns", x0=0.0), n=1, **BOX)


def test_slice_extents_clip_the_line():
    # ns panel limited to y in [-1, 2] km, map to y in [0, 5] km: both bound
    # the line, the result is their overlap [0, 2] km.
    p = fviz.slice_intersection_polyline(
        dict(kind="map", z0=2000.0, ylim=[0.0, 5000.0]),
        dict(kind="ns", x0=0.0, ylim=[-1000.0, 2000.0]), **BOX)
    np.testing.assert_allclose(np.sort(p[:, 1]), [0.0, 2000.0], atol=1e-6)
    # a vertical plane's xlim is the along-strike range relative to `point`:
    # strike 90 (E-W), point x = 1 km, range [-500, 500] -> x in [500, 1500]
    q = fviz.slice_intersection_polyline(
        dict(kind="plane", point=[1000.0, 0.0, 0.0], strike=90.0, dip=90.0,
             xlim=[-500.0, 500.0]),
        dict(kind="map", z0=2000.0), **BOX)
    np.testing.assert_allclose(np.sort(q[:, 0]), [500.0, 1500.0], atol=1e-6)
    # dipping plane: down-dip range ylim relative to `point` (dip 90 -> z)
    r = fviz.slice_intersection_polyline(
        dict(kind="plane", point=[0.0, 0.0, 0.0], strike=0.0, dip=45.0,
             ylim=[0.0, 2000.0 * math.sqrt(2.0)]),
        dict(kind="ns", x0=0.0), **BOX)
    # plane x = z meets x = 0 in z = 0 (a horizontal line along y); the
    # down-dip range keeps it, so its end points are y = -/+ 6 km
    np.testing.assert_allclose(np.sort(r[:, 1]), [-6000.0, 6000.0], atol=1e-6)
    np.testing.assert_allclose(r[:, 2], 0.0, atol=1e-6)


def test_unbounded_raises_and_mesh_file_bounds(model):
    a, b = dict(kind="map", z0=2000.0), dict(kind="ns", x0=0.0)
    with pytest.raises(ValueError, match="unbounded"):
        fviz.slice_intersection_polyline(a, b)
    p = fviz.slice_intersection_polyline(a, b, mesh_file=model[0])
    np.testing.assert_allclose(np.sort(p[:, 1]), [-6000.0, 6000.0], atol=1e-6)


def test_unresolved_specs_raise():
    with pytest.raises(ValueError, match="model-local metres"):
        fviz.slice_intersection_polyline(
            dict(kind="ns", x0=(-71.4, "latlon")), dict(kind="map", z0=0.0),
            **BOX)
    with pytest.raises(ValueError, match="profile"):
        fviz.slice_intersection_polyline(
            dict(kind="profile", p1=[0, 0], p2=[1, 1]),
            dict(kind="map", z0=0.0), **BOX)
    with pytest.raises(ValueError, match="dip"):
        fviz.slice_intersection_polyline(
            dict(kind="plane", point=[0, 0, 0], strike=0.0, dip=200.0),
            dict(kind="map", z0=0.0), **BOX)
    with pytest.raises(ValueError, match="unknown slice kind"):
        fviz.slice_intersection_polyline(
            dict(kind="foo"), dict(kind="map", z0=0.0), **BOX)


def _random_spec(rng):
    kind = rng.choice(["map", "ns", "ew", "plane", "plane"])
    if kind == "map":
        return dict(kind="map", z0=float(rng.uniform(0, 5000)))
    if kind == "ns":
        return dict(kind="ns", x0=float(rng.uniform(-4000, 4000)))
    if kind == "ew":
        return dict(kind="ew", y0=float(rng.uniform(-4000, 4000)))
    return dict(kind="plane",
                point=[float(v) for v in rng.uniform(-3000, 3000, 3)],
                strike=float(rng.uniform(0, 360)),
                dip=float(rng.choice([90.0, rng.uniform(20, 160)])))


def test_random_pairs_lie_in_both_planes_and_are_maximal():
    rng = np.random.default_rng(20261111)
    n_found = 0
    for _ in range(300):
        a, b = _random_spec(rng), _random_spec(rng)
        p = fviz.slice_intersection_polyline(a, b, n=3, **BOX)
        if p is None:
            continue
        n_found += 1
        assert _on_plane(a, p) and _on_plane(b, p)
        lo = np.array([BOX["xlim"][0], BOX["ylim"][0], BOX["zlim"][0]])
        hi = np.array([BOX["xlim"][1], BOX["ylim"][1], BOX["zlim"][1]])
        assert np.all(p >= lo - 1e-6) and np.all(p <= hi + 1e-6)
        # maximal: a step beyond either end leaves the box
        d = p[-1] - p[0]
        d /= np.linalg.norm(d)
        for end, sgn in ((p[0], -1.0), (p[-1], 1.0)):
            q = end + sgn * 1.0 * d
            assert np.any(q < lo - 1e-9) or np.any(q > hi + 1e-9)
        # middle vertex is the middle of the segment
        np.testing.assert_allclose(p[1], 0.5 * (p[0] + p[2]), atol=1e-6)
    assert n_found > 100        # the random pairs do intersect most of the time


def test_polylines_helper_entries():
    sl = [dict(kind="map", z0=2000.0, title="map 2 km"),
          dict(kind="ns", x0=3500.0), dict(kind="ew", y0=2500.0),
          dict(kind="map", z0=4000.0)]
    d = fviz.slice_intersection_polylines(sl, **BOX)
    # 4 slices -> 6 pairs; the two maps are parallel -> 5 entries
    assert sorted(d) == ["slice0_slice1", "slice0_slice2", "slice1_slice2",
                         "slice1_slice3", "slice2_slice3"]
    e = d["slice0_slice1"]
    assert e["label"] == "map 2 km x N-S x=3.5 km"
    assert e["max_dist"] == 0.0
    assert e["z_range"] == [2000.0, 2000.0]
    assert e["style"]["linestyle"] == "--"
    pts, crs = e["points"]
    assert crs == "model" and pts.shape == (2, 3)
    sel = fviz.slice_intersection_polylines(sl, pairs=[(0, 2)], style=dict(
        lw=2, c="r"), **BOX)
    assert list(sel) == ["slice0_slice2"]
    assert sel["slice0_slice2"]["style"] == {"lw": 2, "c": "r",
                                             "linestyle": "--"}
    with pytest.raises(ValueError):
        fviz.slice_intersection_polylines(sl, pairs=[(0, 0)], **BOX)
    with pytest.raises(ValueError):
        fviz.slice_intersection_polylines(sl, pairs=[(0, 9)], **BOX)


# ---------------------------------------------------------------------------
# slab clipping
# ---------------------------------------------------------------------------
def _slab_dense(P, point, normal, dmax, n=4001):
    """Independent reference: densely sample the polyline, keep the samples
    within the slab, return them grouped in runs."""
    t = np.linspace(0.0, 1.0, (len(P) - 1) * n + 1)
    seg = np.minimum((t * (len(P) - 1)).astype(int), len(P) - 2)
    s = t * (len(P) - 1) - seg
    X = P[seg] + s[:, None] * (P[seg + 1] - P[seg])
    ok = np.abs((X - point) @ normal) <= dmax + 1e-3
    return X, ok


def test_clip_polyline_whole_and_none():
    P = np.array([[0.0, 0, 0], [1000.0, 0, 0]])
    out = fviz._clip_polyline_to_slab(P, np.zeros(3), np.array([0, 1.0, 0]), None)
    np.testing.assert_allclose(out[0], P)
    # farther than dmax from the plane -> nothing
    assert fviz._clip_polyline_to_slab(
        P, np.array([0.0, 5000.0, 0.0]), np.array([0, 1.0, 0]), 100.0) == []


def test_clip_polyline_drops_transversal_crossing_for_dmax_zero():
    P = np.array([[0.0, -1000, 0], [0.0, 1000, 0]])       # crosses y = 0
    nrm = np.array([0, 1.0, 0])
    assert fviz._clip_polyline_to_slab(P, np.zeros(3), nrm, 0.0) == []
    out = fviz._clip_polyline_to_slab(P, np.zeros(3), nrm, 10.0)
    assert len(out) == 1
    np.testing.assert_allclose(out[0][:, 1], [-10.0, 10.0], atol=2e-3)
    # a polyline that is itself shorter than 1 cm is kept
    tiny = np.array([[0.0, 0, 0], [0.0, 1e-3, 0]])
    assert len(fviz._clip_polyline_to_slab(tiny, np.zeros(3), nrm, 0.0)) == 1


def test_clip_polyline_in_plane_survives_dmax_zero():
    P = np.array([[1.0, 2.0, 3.0], [100.0, 2.0, 30.0], [200.0, 2.0, 5.0]])
    out = fviz._clip_polyline_to_slab(P, np.array([0.0, 2.0, 0.0]),
                                      np.array([0, 1.0, 0]), 0.0)
    assert len(out) == 1
    np.testing.assert_allclose(out[0], P)


def test_clip_polyline_two_pieces_when_it_leaves_and_reenters():
    # zig-zag in y across the slab |y| <= 100
    P = np.array([[0.0, -300, 0], [100, 0, 0], [200, 300, 0], [300, 0, 0],
                  [400, -300, 0]])
    out = fviz._clip_polyline_to_slab(P, np.zeros(3), np.array([0, 1.0, 0]), 100.0)
    # enters at y = -100, leaves at y = +100; re-enters and leaves again
    assert len(out) == 2
    for piece in out:
        assert np.all(np.abs(piece[:, 1]) <= 100.0 + 1e-2)
        assert np.isclose(np.abs(piece[0, 1]), 100.0, atol=1e-2)
        assert np.isclose(np.abs(piece[-1, 1]), 100.0, atol=1e-2)


def test_clip_polyline_matches_dense_sampling():
    rng = np.random.default_rng(7)
    for _ in range(40):
        P = rng.uniform(-3000, 3000, (int(rng.integers(2, 6)), 3))
        nrm = rng.normal(size=3)
        nrm /= np.linalg.norm(nrm)
        pt = rng.uniform(-500, 500, 3)
        dmax = float(rng.uniform(50, 2500))
        pieces = fviz._clip_polyline_to_slab(P, pt, nrm, dmax)
        X, ok = _slab_dense(P, pt, nrm, dmax)
        # total length of the clipped pieces vs the length of the dense
        # samples inside the slab
        L_clip = sum(np.linalg.norm(np.diff(q, axis=0), axis=1).sum()
                     for q in pieces)
        step = np.linalg.norm(np.diff(X, axis=0), axis=1)
        L_ref = step[ok[:-1] & ok[1:]].sum()
        assert abs(L_clip - L_ref) < 2.0e-3 * max(1.0, L_ref) + 5.0, (L_clip, L_ref)
        for q in pieces:
            assert np.all(np.abs((q - pt) @ nrm) <= dmax + 1e-2)


# ---------------------------------------------------------------------------
# preparation: CRS, validation, km conversion
# ---------------------------------------------------------------------------
def _prep(pl, **kw):
    args = dict(projection_dist=1500.0, latlon_to_model_fn=None, utm_zone=32,
                utm_northern=True, utm_origin_e=500000.0,
                utm_origin_n=4000000.0)
    args.update(kw)
    return fviz._prepare_polylines(pl, **args)


def test_prepare_model_utm_and_shorthand():
    pts = [[1000.0, 2000.0], [3000.0, 4000.0, 500.0]][:1]
    out = _prep({"a": [[1000.0, 2000.0], [3000.0, 4000.0]],
                 "b": dict(points=([[503000.0, 4002000.0, 700.0],
                                    [503000.0, 4004000.0, 900.0]], "utm"))})
    np.testing.assert_allclose(out[0]["pts"],
                               [[1000, 2000, 0], [3000, 4000, 0]])
    np.testing.assert_allclose(out[1]["pts"],
                               [[3000, 2000, 700], [3000, 4000, 900]])
    assert out[0]["label"] == "a" and out[1]["label"] == "b"
    assert out[0]["color"] != out[1]["color"]
    assert out[0]["max_dist"] == 1500.0          # from projection_dist
    del pts


def test_prepare_latlon_roundtrip():
    utl = _utl()
    zone, north = 32, True
    e0, n0 = 500000.0, 4000000.0
    # two model points -> lat/lon with the package's own conversions
    model_xy = np.array([[1000.0, 2000.0], [-3000.0, 5000.0]])
    ll = [utl.utm_to_latlon_zn(e0 + x, n0 + y, zone, north) for x, y in model_xy]
    pts = [[lon, lat, 800.0] for lat, lon in ll]
    out = _prep({"c": dict(points=(pts, "latlon"))},
                latlon_to_model_fn=fem.latlon_to_model, utm_zone=zone,
                utm_northern=north, utm_origin_e=e0, utm_origin_n=n0)
    np.testing.assert_allclose(out[0]["pts"][:, :2], model_xy, atol=1e-2)
    np.testing.assert_allclose(out[0]["pts"][:, 2], 800.0)


def test_prepare_options():
    out = _prep({"a": dict(points=[[0, 0], [1, 1]], kinds="vertical",
                           max_dist=0.0, z_range=[3000.0, 1000.0],
                           label=None, style=dict(label="x", lw=2))})
    p = out[0]
    assert p["kinds"] == frozenset({"ns", "ew", "plane"})
    assert p["max_dist"] == 0.0                   # 0 is not "default"
    assert p["z_range"] == (1000.0, 3000.0)
    assert p["label"] == "x"                      # a label in style wins
    assert _prep({"a": dict(points=[[0, 0], [1, 1]], kinds=["map", "we"])}
                 )[0]["kinds"] == frozenset({"map", "ew"})
    assert _prep({}) == [] and _prep(None) == []
    # no projection_dist and no max_dist -> the whole polyline
    assert _prep({"a": [[0, 0], [1, 1]]}, projection_dist=None)[0]["max_dist"] is None


@pytest.mark.parametrize("bad, msg", [
    ({"a": dict(foo=1)}, "points"),
    ({"a": [[0, 0]]}, "n >= 2"),
    ({"a": [[0, 0, 0, 0], [1, 1, 1, 1]]}, "shape"),
    ({"a": [[0, 0], [np.nan, 1]]}, "finite"),
    ({"a": ([[0, 0], [1, 1]], "mercator")}, "CRS"),
    ({"a": ([[0, 0], [1, 1]], "latlon")}, "latlon_to_model_fn"),
    ({"a": dict(points=[[0, 0], [1, 1]], kinds="diagonal")}, "kind"),
    ({"a": dict(points=[[0, 0], [1, 1]], max_dist=-1.0)}, "max_dist"),
    ({"a": dict(points=[[0, 0], [1, 1]], style="r-")}, "style"),
    ({"a": [["x", 0], ["y", 1]]}, "numeric"),
])
def test_prepare_errors(bad, msg):
    with pytest.raises(ValueError, match=msg):
        _prep(bad)
    with pytest.raises(TypeError):
        _prep([("a", [[0, 0], [1, 1]])])


def test_polylines_km_to_m():
    inp = {
        "model": dict(points=[[1.0, 2.0, 0.5], [3.0, 4.0, 1.5]],
                      max_dist=2.0, z_range=[1.0, 3.0], label="m"),
        "utm": dict(points=([[503000.0, 4002000.0, 0.7],
                             [503100.0, 4002100.0, 0.9]], "utm")),
        "ll": ([[6.12, 45.9, 0.1], [6.2, 45.95]][:1] * 2, "latlon"),
        "short": [[1.0, 1.0], [2.0, 2.0]],
    }
    import copy
    keep = copy.deepcopy(inp)
    out = fem.polylines_km_to_m(inp)
    assert inp == keep                                 # not modified
    pts, crs = out["model"]["points"]
    assert crs == "model"
    np.testing.assert_allclose(pts, [[1000, 2000, 500], [3000, 4000, 1500]])
    assert out["model"]["max_dist"] == 2000.0
    assert out["model"]["z_range"] == [1000.0, 3000.0]
    assert out["model"]["label"] == "m"
    pts, crs = out["utm"]["points"]                    # E, N metres stay
    assert crs == "utm"
    np.testing.assert_allclose(pts, [[503000, 4002000, 700],
                                     [503100, 4002100, 900]])
    pts, crs = out["ll"]["points"]                     # degrees stay, z km->m
    assert crs == "latlon"
    np.testing.assert_allclose(pts, [[6.12, 45.9, 100.0]] * 2)
    pts, crs = out["short"]["points"]
    np.testing.assert_allclose(pts, [[1000, 1000], [2000, 2000]])
    assert fem.polylines_km_to_m(None) is None and fem.polylines_km_to_m({}) is None
    # the result is accepted by plot_model_slices' preparation
    assert len(_prep(out, latlon_to_model_fn=fem.latlon_to_model)) == 4
    with pytest.raises(ValueError):
        fem.polylines_km_to_m({"a": dict(points=([[0, 0], [1, 1]], "foo"))})
    with pytest.raises(ValueError):
        fem.polylines_km_to_m({"a": dict(label="x")})


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
FAULT = np.array([[-4000.0, -3000.0, 500.0], [0.0, 0.0, 2500.0],
                  [4000.0, 3000.0, 4500.0]])


def _ref_section(P, point, normal, u_ax, v_ax, dmax):
    """Independent: densely sampled slab points -> in-plane coordinates."""
    X, ok = _slab_dense(P, point, normal, dmax)
    X = X[ok]
    return X @ u_ax, X @ v_ax


def test_default_and_empty_polylines_give_the_same_axes(model, tmp_path):
    sl = [dict(kind="map", z0=2000.0), dict(kind="ns", x0=3000.0)]
    f0 = _render(model, sl, tmp=str(tmp_path))
    f1 = _render(model, sl, polylines={}, tmp=str(tmp_path))
    for i in range(2):
        assert f0.axes[i].get_xlim() == f1.axes[i].get_xlim()
        assert f0.axes[i].get_ylim() == f1.axes[i].get_ylim()
        assert len(f0.axes[i].lines) == len(f1.axes[i].lines)


def test_map_plan_view_all_vertices(model, tmp_path):
    sl = [dict(kind="map", z0=2000.0)]
    fig = _render(model, sl, polylines={"fault": FAULT}, tmp=str(tmp_path))
    (ln,) = _panel_lines(fig, 0, "fault")
    np.testing.assert_allclose(ln.get_xdata(), FAULT[:, 0] / 1000.0)
    np.testing.assert_allclose(ln.get_ydata(), FAULT[:, 1] / 1000.0)
    assert ln.get_zorder() == 7


def test_map_z_range_and_kinds(model, tmp_path):
    sl = [dict(kind="map", z0=500.0), dict(kind="map", z0=2000.0),
          dict(kind="ns", x0=0.0)]
    pl = {"zr": dict(points=FAULT, z_range=[1500.0, 2500.0]),
          "only_ns": dict(points=FAULT, kinds="ns"),
          "hor": dict(points=FAULT, kinds="horizontal")}
    fig = _render(model, sl, polylines=pl, tmp=str(tmp_path))
    assert not _panel_lines(fig, 0, "zr")            # z0 = 0.5 km: outside
    assert len(_panel_lines(fig, 1, "zr")) == 1      # z0 = 2 km: inside
    assert not _panel_lines(fig, 0, "only_ns") and not _panel_lines(fig, 1, "only_ns")
    assert len(_panel_lines(fig, 2, "only_ns")) == 1
    assert len(_panel_lines(fig, 0, "hor")) == 1 and not _panel_lines(fig, 2, "hor")


def test_vertical_sections_slab_projection(model, tmp_path):
    D = 1500.0
    sl = [dict(kind="ns", x0=3000.0), dict(kind="ew", y0=2500.0),
          dict(kind="plane", point=[0.0, 0.0, 0.0], strike=45.0, dip=90.0),
          dict(kind="plane", point=[0.0, 0.0, 0.0], strike=0.0, dip=60.0)]
    fig = _render(model, sl, polylines={"f": dict(points=FAULT, max_dist=D)},
                  tmp=str(tmp_path))
    ez = np.array([0, 0, -1.0])
    # (index, normal, u_ax, v_ax, point, display function)
    s45 = math.sin(math.radians(45.0))
    u60, v60 = np.array([0, 1.0, 0]), -np.array([0.5, 0.0, math.sin(math.radians(60))])
    cases = [
        (0, np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), ez,
         np.array([3000.0, 0, 0]), lambda u, v: (u / 1e3, v / 1e3)),
        (1, np.array([0, 1.0, 0]), np.array([1.0, 0, 0]), ez,
         np.array([0, 2500.0, 0]), lambda u, v: (u / 1e3, v / 1e3)),
        (2, np.array([s45, -s45, 0]), np.array([s45, s45, 0]), ez,
         np.zeros(3), lambda u, v: (u / 1e3, v / 1e3)),
        (3, np.array([math.sin(math.radians(60)), 0, -0.5]), u60, v60,
         np.zeros(3), lambda u, v: (u / 1e3, v / 1e3)),
    ]
    for i, nrm, u_ax, v_ax, pt, disp in cases:
        (ln,) = _panel_lines(fig, i, "f")
        U, V = _ref_section(FAULT, pt, nrm, u_ax, v_ax, D)
        X, Y = disp(U, V)
        got = np.column_stack([ln.get_xdata(), ln.get_ydata()])
        got = got[np.isfinite(got).all(axis=1)]
        # the drawn piece spans the same x / y ranges as the dense reference
        np.testing.assert_allclose([got[:, 0].min(), got[:, 0].max()],
                                   [X.min(), X.max()], atol=2e-3)
        np.testing.assert_allclose([got[:, 1].min(), got[:, 1].max()],
                                   [Y.min(), Y.max()], atol=2e-3)


def test_vertical_slab_outside_gives_no_line_and_zero_dist(model, tmp_path):
    sl = [dict(kind="ns", x0=-6000.0), dict(kind="ns", x0=3500.0)]
    # the box outline of the conductor, drawn in the ns section through its
    # middle with max_dist = 1500 (faces at +/- 1500) and with 0
    cx = np.array([[2000.0, 1000, 1000], [2000, 4000, 1000],
                   [2000, 4000, 3000], [2000, 1000, 3000], [2000, 1000, 1000]])
    pl = {"face": dict(points=cx, max_dist=1500.0),
          "face0": dict(points=cx, max_dist=0.0)}
    fig = _render(model, sl, polylines=pl, tmp=str(tmp_path))
    assert not _panel_lines(fig, 0, "face")            # 8 km away
    (ln,) = _panel_lines(fig, 1, "face")
    np.testing.assert_allclose(np.column_stack([ln.get_xdata(), ln.get_ydata()]),
                               [[1, -1], [4, -1], [4, -3], [1, -3], [1, -1]])
    assert not _panel_lines(fig, 1, "face0")           # not in the plane x=3.5


def test_intersection_lines_drawn_in_both_slices(model, tmp_path):
    sl = [dict(kind="map", z0=2000.0), dict(kind="ns", x0=3500.0),
          dict(kind="ew", y0=2500.0)]
    d = fviz.slice_intersection_polylines(sl, **BOX)
    assert sorted(d) == ["slice0_slice1", "slice0_slice2", "slice1_slice2"]
    fig = _render(model, sl, polylines=d, tmp=str(tmp_path))
    lab = {k: v["label"] for k, v in d.items()}
    # map x ns: x = 3.5 km on the map (vertical line), z = -2 km in the ns panel
    (m,) = _panel_lines(fig, 0, lab["slice0_slice1"])
    np.testing.assert_allclose(m.get_xdata(), 3.5)
    np.testing.assert_allclose(sorted(m.get_ydata()), [-6.0, 6.0])
    (n,) = _panel_lines(fig, 1, lab["slice0_slice1"])
    np.testing.assert_allclose(n.get_ydata(), -2.0)
    # the vertical line of ns x ew is a point on the map (z_range excludes
    # nothing: it spans the whole depth range) and a vertical line on ew
    (e,) = _panel_lines(fig, 2, lab["slice1_slice2"])
    np.testing.assert_allclose(e.get_xdata(), 3.5)
    np.testing.assert_allclose(sorted(e.get_ydata()), [-6.0, 1.0])
    # not on a panel that does not contain it (map x ns line on the ew panel)
    assert not _panel_lines(fig, 2, lab["slice0_slice1"])
    # a map elsewhere does not show the line of two sections at other depths
    fig2 = _render(model, [dict(kind="map", z0=500.0)] + sl[1:],
                   polylines={"v": dict(points=d["slice1_slice2"]["points"],
                                        z_range=[2000.0, 2000.0])},
                   tmp=str(tmp_path))
    assert not _panel_lines(fig2, 0, "v")


def test_labels_legend_and_axis_limits_unchanged(model, tmp_path):
    sl = [dict(kind="map", z0=2000.0), dict(kind="ns", x0=3000.0)]
    # no figure-level limits: autoscaled axes; a far-away polyline must not
    # change them
    kw = dict(xlim=None, ylim=None, zlim=None)
    f0 = _render(model, sl, tmp=str(tmp_path), **kw)
    far = np.array([[-90000.0, -90000.0, 0.0], [90000.0, 90000.0, 9000.0]])
    f1 = _render(model, sl, polylines={
        "far": dict(points=far, label="far one", max_dist=None),
        "quiet": dict(points=FAULT, label=None)}, tmp=str(tmp_path), **kw)
    for i in range(2):
        assert f0.axes[i].get_xlim() == f1.axes[i].get_xlim()
        assert f0.axes[i].get_ylim() == f1.axes[i].get_ylim()
        legend = f1.axes[i].get_legend()
        assert legend is not None
        texts = [t.get_text() for t in legend.get_texts()]
        assert "far one" in texts and "quiet" not in texts
    # no label at all -> no legend
    f2 = _render(model, sl, polylines={"q": dict(points=FAULT, label=None)},
                 tmp=str(tmp_path))
    assert f2.axes[0].get_legend() is None


def test_style_aliases_do_not_collide(model, tmp_path):
    sl = [dict(kind="map", z0=2000.0)]
    fig = _render(model, sl, polylines={"s": dict(points=FAULT, style=dict(
        c="red", lw=3, ls=":", marker="o", zorder=9))}, tmp=str(tmp_path))
    (ln,) = _panel_lines(fig, 0, "s")
    assert ln.get_color() == "red" and ln.get_linewidth() == 3
    assert ln.get_linestyle() == ":" and ln.get_zorder() == 9
    assert ln.get_marker() == "o"


def test_display_coords_utm_and_latlon_shift_the_line(model, tmp_path):
    utl = _utl()
    e0, n0, zone = 500000.0, 4000000.0, 32
    sl = [dict(kind="map", z0=2000.0), dict(kind="ns", x0=3000.0)]
    pts = np.array([[1000.0, 1000.0, 1000.0], [3000.0, 3000.0, 2000.0]])
    fig = _render(model, sl, polylines={"p": pts}, tmp=str(tmp_path),
                  display_coords="utm", utm_origin_e=e0, utm_origin_n=n0,
                  utm_zone=zone, utm_northern=True)
    (ln,) = _panel_lines(fig, 0, "p")                # UTM km on the map
    np.testing.assert_allclose(ln.get_xdata(), (pts[:, 0] + e0) / 1000.0)
    np.testing.assert_allclose(ln.get_ydata(), (pts[:, 1] + n0) / 1000.0)
    # the same polyline given in lat/lon lands on the same place
    ll = [utl.utm_to_latlon_zn(e0 + x, n0 + y, zone, True) for x, y in pts[:, :2]]
    pl = {"p": dict(points=([[lon, lat, z] for (lat, lon), z in zip(ll, pts[:, 2])],
                            "latlon"))}
    fig = _render(model, sl, polylines=pl, tmp=str(tmp_path),
                  display_coords="latlon", utm_origin_e=e0, utm_origin_n=n0,
                  utm_zone=zone, utm_northern=True,
                  latlon_to_model_fn=fem.latlon_to_model,
                  utm_to_latlon_fn=utl.utm_to_latlon_zn)
    (ln,) = _panel_lines(fig, 0, "p")                # latlon display: UTM m
    np.testing.assert_allclose(ln.get_xdata(), (pts[:, 0] + e0), atol=0.05)
    np.testing.assert_allclose(ln.get_ydata(), (pts[:, 1] + n0), atol=0.05)


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))
