# data_viz.py — Matplotlib helpers for MT transfer-function plots

`data_viz.py` contains small, composable Matplotlib plotters for
magnetotelluric transfer-function data. All plotters work with a **tidy
`pandas.DataFrame`** produced by `data_proc.dataframe_from_edi()`, and can
also accept an EDI-style data dict directly (auto-converted internally via
`datadict_to_plot_df()`).

## MT plotters

| Function | Plot type | Axes scale |
|----------|-----------|------------|
| `add_rho` | Apparent resistivity curves | log–log |
| `add_rhoplus` | D⁺/rho-plus test: ρ_a and ρ⁺ with violation markers | log–log |
| `add_phase` | Impedance phase | semilog-x |
| `add_tipper` | Tipper real/imaginary | semilog-x |
| `add_pt` | Phase tensor components | semilog-x |

All plotters accept an optional Matplotlib `Axes` and return the axes. If
`ax=None`, a new figure/axes pair is created automatically.

---

## D⁺ / rho-plus test plot

### `add_rhoplus(data, comps='xy,yx', ax=None, ...)`

Overlays the measured apparent resistivity ρ_a (solid line) and the D⁺
upper bound ρ⁺ (dashed line, same colour) for each requested component.
Periods where ρ_a > ρ⁺ — i.e. violations of the 1-D consistency test — are
highlighted with red crosses.

The D⁺ (also known as **dplus** in Cordell's mtcode) is the necessary and
sufficient condition for a scalar impedance dataset to be compatible with any
1-D conductivity profile (Parker 1980; Parker & Whaler 1981). A violation at
even one period proves no 1-D model can explain the data.

```python
# Quickest usage — pass a raw EDI dict
ax = data_viz.add_rhoplus(site, comps="xy,yx")

# With a pre-built DataFrame
df = data_proc.dataframe_from_edi(site)
ax = data_viz.add_rhoplus(df, comps="xy,yx", show_errors=True)
```

Key extra arguments:

| Argument | Default | Effect |
|----------|---------|--------|
| `n_lambda_per_freq` | `4` | D⁺ λ-grid density |
| `violation_marker` | `"x"` | Matplotlib marker for violations |
| `violation_color` | `"red"` | Colour for violation markers |
| `violation_size` | `120` | Scatter marker size |

Requires `data_proc.compute_rhoplus` (lazy import).

**References:** Parker (1980) *JGR* 85; Parker & Whaler (1981) *JGR* 86;
Parker & Booker (1996) *PEPI* 98;
Cordell et al. (2022) mtcode doi:10.5281/zenodo.6784201
https://github.com/darcycordell/mtcode

---

## Expected DataFrame columns

**Required:** `freq` [Hz]

**Optional but recommended:** `period` [s] (computed from `freq` if missing)

### Apparent resistivity / phase

- `rho_xx`, `rho_xy`, `rho_yx`, `rho_yy` [Ω·m]
- `phi_xx`, `phi_xy`, `phi_yx`, `phi_yy` [deg]
- Optional error columns: `rho_xy_err`, `phi_xy_err`, … (suffix configurable
  via `error_suffix`)

### Tipper

- `Tx_re`, `Tx_im`, `Ty_re`, `Ty_im`
- Optional: `Tx_re_err`, `Tx_im_err`, `Ty_re_err`, `Ty_im_err`

### Phase tensor

Phase tensor components are real; column names keep the historical `_re` suffix:

- `ptxx_re`, `ptxy_re`, `ptyx_re`, `ptyy_re`
- Optional: `ptxx_re_err`, `ptxy_re_err`, `ptyx_re_err`, `ptyy_re_err`

---

## Common plotter arguments

All MT plotters share these keyword arguments:

- `show_errors` (bool) — show ±1σ fill-between envelopes (default `False`)
- `error_suffix` (str) — suffix for error columns (default `"_err"`)
- `error_alpha` (float) — transparency for error envelopes (default `0.25`)
- `comps` (str) — comma-separated component list, e.g. `"xy,yx"` (default varies by plotter; not available for `add_tipper` / `add_pt`)
- `legend` (bool) — show legend (default `True`)
- `xlim` (tuple or None) — period-axis limits `(lo, hi)` in seconds; either element may be `None` to keep the auto-computed bound (default `None`)
- `ylim` (tuple or None) — y-axis limits `(lo, hi)`; either element may be `None` (default `None`)

All six arguments are consumed by the plotter and are **not** forwarded to the underlying Matplotlib call — they are safe to include in a shared `PLTARGS` dict passed to every plotter regardless of plot type.

Additional `**line_kw` are forwarded to the underlying `ax.loglog` / `ax.semilogx` call. Two commonly useful overrides are:

- `linestyle` (str) — override the default line style. All plotters pop this from `**line_kw` before the plot call. For `add_tipper` and `add_pt`, the override is applied uniformly to all components; when omitted, each component keeps its own default (`"-"` for real parts, `"--"` for imaginary parts in `add_tipper`).
- `marker` (str or None) — override the marker symbol. `add_rho` and `add_phase` have **no default marker** (marker-free lines by default); pass `marker="o"` to restore the old behaviour. `add_tipper` and `add_pt` keep their meaningful per-component defaults (`o`, `^`, `s`, `d`) but accept a uniform override via `marker=`. `add_rhoplus` applies the marker only to the ρ_a line; the ρ⁺ line always has no marker.

### Phase curves that leave the +-180 deg window (`add_phase`, 2026-11-20)

The phase of a complex number lives in a window of 360 deg (here -180 to
+180).  Where a curve leaves it on one side and re-enters on the other, two
neighbouring points differ by more than 180 deg, and a plain line (and a
plain error band) is drawn straight through the jump: a long vertical streak.
`add_phase` now detects such jumps (more than half the window between two
neighbouring plotted points; points that are NaN or have a non-positive period
never count) and

- interrupts the solid line and the `show_errors` band there, and
- joins the two points with a separate line drawn in `wrap_style`.

| Argument | Default | Meaning |
|----------|---------|---------|
| `wrap_mode` | `"direct"` | `"direct"`: one straight line from point to point. `"short"`: two short stubs, from the point to the window edge it leaves at and from the opposite edge to the next point (much less ink if the phase jitters around +-180 deg). `"off"`: old behaviour, line straight through the jump |
| `wrap_style` | `None` | dict of Line2D keywords, applied on top of the module default `data_viz.PHASE_WRAP_STYLE` (`{"linestyle": ":"}`); colour, linewidth, alpha and zorder not given are taken from the solid curve. `False`: no connection (the curve is just interrupted) |
| `wrap_range` | `(-180, 180)` | the phase window; use `(0, 360)` for phases given in that range |

```python
data_viz.add_phase(df, comps="xy,yx")                               # dotted, as the curve
data_viz.add_phase(df, comps="xy,yx",
                   wrap_style={"linestyle": (0, (1, 3)), "alpha": 0.5, "color": "0.4"})
data_viz.add_phase(df, comps="xy,yx", wrap_mode="short")            # stubs only
data_viz.add_phase(df, comps="xy,yx", wrap_style=False)             # gap, no connection
data_viz.PHASE_WRAP_STYLE.update(linestyle="--")                    # for all later calls
```

Curves without a line (`linestyle="none"`, `linewidth=0`, i.e. markers only,
as used for observed data in the FEMTIC/ModEM misfit plots) get no
connection; their error band is still interrupted.  A jump of more than half
the window is always taken as a wrap, never as a real change of the phase
through the middle of the window.  The three arguments are consumed by
`add_phase` only: do not put them in a `PLTARGS` dict that is also passed to
`add_rho`, `add_tipper` or `add_pt` (those would forward them to Matplotlib).
`"ls"` is accepted as an alias of `"linestyle"`.  Tests:
`scripts/general/test_data_viz_phase_wrap.py` (41 cases).

#### The same for phase lines drawn by other code (2026-11-20, later)

`add_phase` is built on two public helpers that other plotters use:

```python
k = data_viz.mark_phase_wraps(ax, line, wrap_mode="direct", wrap_style=None,
                              wrap_range=(-180.0, 180.0))
idx = data_viz.phase_wrap_index(x, y, lo=-180.0, hi=180.0)
```

`mark_phase_wraps` takes a `Line2D` that has already been drawn (for
`ax.errorbar` the data line `container.lines[0]`), inserts the NaN gaps and
adds the connection (same arguments as in `add_phase`).  It returns the wrap
indices `k` (between point `k` and `k+1` of the original data; empty if there
is none or `wrap_mode="off"`), which are the `breaks` for an error band; a
line without a visible stroke (markers only) gets no connection but its
indices are still returned.  The connection lines carry the gid
`data_viz.PHASE_WRAP_GID`, so that a caller that restyles the new lines of an
axes after plotting (`femtic_misfit._draw`, `femtic_viz.plot_data_ensemble`)
can leave them alone.  `wrap_style=True` is the same as `None` (the default
style).  Users of the helpers: `femtic_misfit.py` (and `femtic_ens_misfit.py`,
`femtic_mod_misfit.py`, `modem_data_misfit.py`), `femtic_viz.plot_data_ensemble`
(and the callers `femtic_rto_prep.py`, `femtic_gst_prep.py`,
`femtic_distcorr.py`), `viz.plot_rhophas` (opt-in), `mt_plot_data.py`,
`mt_get_averages.py`, `mt_aniso1d_forward.py`.  Tests of these:
`scripts/general/test_phase_wrap_plotters.py` (49 cases).

> **Note for callers using a shared `PLTARGS` dict:** because `ylim` varies per panel, pass per-call overrides as `**{**PLTARGS, "ylim": (...)}` rather than mutating `PLTARGS["ylim"]` before each call.

> **Note on `femtic_viz.plot_data_ensemble`:** that function applies per-component markers *post-hoc* (by patching `Line2D` objects after each plotter call) rather than forwarding `marker=` through `**line_kw`. This avoids any duplicate-keyword collision and works with all `data_viz` versions.

---

## Data dict support

In addition to DataFrames, the main plotters accept a data dict (the dict
returned by `data_proc.load_edi()`) with keys `freq`, `Z`, `Z_err`, `T`,
`T_err`, `P`, `P_err`. Conversion to the plotting DataFrame happens
automatically via `datadict_to_plot_df()`.

---

## Recommended workflow

```python
import matplotlib.pyplot as plt
import data_proc
import data_viz

site = data_proc.load_edi("SITE.edi")
site["P"], site["P_err"] = data_proc.compute_pt(site["Z"], site.get("Z_err"))
df = data_proc.dataframe_from_edi(site)

fig, axs = plt.subplots(2, 3, figsize=(14, 8))

data_viz.add_rho(df, ax=axs[0, 0], comps="xy,yx", show_errors=True)
data_viz.add_rhoplus(site, ax=axs[0, 1], comps="xy,yx")   # D+ test
data_viz.add_phase(df, ax=axs[0, 2], comps="xy,yx", show_errors=True)
data_viz.add_tipper(df, ax=axs[1, 0], show_errors=True)
data_viz.add_pt(df, ax=axs[1, 1], show_errors=True)

fig.tight_layout()
plt.show()
```

---

## Generic subplot helpers

Two utility functions for rendering lists of small datasets into subplot grids:

- `plot_gridx(data_list, nrows, ncols, figsize)` — supports
  `type ∈ {"line", "scatter", "image", "hist"}`
- `plot_grid(data_list, nrows, ncols, figsize)` — simple `{"x", "y", "style"}`
  line datasets

Both remove empty axes automatically.

---

Author: Volker Rath (DIAS)
Modified: 2026-03-16 — add_rhoplus (D+/rho+ test plot); Claude Sonnet 4.6 (Anthropic)
Modified: 2026-03-30 — linestyle override via **line_kw (all plotters); removed bogus required-key guard from add_tipper/add_pt; per-call PLTARGS pattern documented; Claude Sonnet 4.6 (Anthropic)
Modified: 2026-04-12 — removed hardcoded marker="o" from add_rho, add_phase, add_rhoplus (default now no marker); pop marker from **line_kw so callers can supply it without TypeError; add uniform marker override to add_tipper and add_pt (same pattern as existing linestyle override); Claude Sonnet 4.6 (Anthropic)
Modified: 2026-11-20 -- `add_phase`: phase curves that leave the +-180 deg window are interrupted and the jump is drawn as a connection in a style dict (`wrap_mode`, `wrap_style`, `wrap_range`, module default `PHASE_WRAP_STYLE`); `ls` accepted as alias of `linestyle`; `_maybe_fill_between(breaks=...)`. Claude Sonnet 5.5 (Anthropic); AI-generated, review before production use
Modified: 2026-11-20 (b) -- public helpers `mark_phase_wraps(ax, line, ...)` and `phase_wrap_index(x, y, lo, hi)`, constant `PHASE_WRAP_GID`; `add_phase` uses `mark_phase_wraps`; `wrap_style=True` accepted (= default style). Used by femtic_misfit, femtic_viz.plot_data_ensemble, viz.plot_rhophas (opt-in) and several plotting scripts. Claude Sonnet 5.5 (Anthropic); AI-generated, review before production use
