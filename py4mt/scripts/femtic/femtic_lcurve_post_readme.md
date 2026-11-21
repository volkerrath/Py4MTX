# femtic_lcurve_post

Post-process an L-curve study with FEMTIC: harvest the finished runs made by
`femtic_lcurve_prep.py`, plot the L-curve, and estimate the trade-off
parameter at the corner (point of maximum curvature).

It combines `femtic_lcurve_plot.py` (run harvesting, plot style) with the
corner search of `lc.py` as it now lives in `inverse.py` of Py4MTX
(`calc_lc_corner`, `curvature`; Menger curvature, default). A cubic-spline
curvature estimate is available as an option and is always calculated
alongside. The script imports these functions; it has no copy of them.

---

## Workflow

| Step | Action |
|------|--------|
| 1 | `femtic_lcurve_prep.py` creates `lc_<alpha>/` for every value in `TRADE_OFF_LIST` |
| 2 | Run FEMTIC in every `lc_<alpha>/` (fixed alpha: `INV_METHOD 1`, `TYPE_OF_TRADE_OFF_PARAMETER 0`) |
| 3 | `femtic_lcurve_post.py` reads the last line of `femtic.cnv` of each run, plots the L-curve, marks the corner, writes the report |
| 4 | Put the reported alpha (or a value near it) into `TRADE_OFF_PARAMETER` of the production `control.dat` |

Needs at least 3 finished runs for `"menger"`, 4 for `"spline"` (5 with
smoothing). A corner has to be bracketed by runs, so use roughly half a
decade or finer in alpha around it. Extend `TRADE_OFF_LIST` and re-run
step 1 (`SKIP_FINISHED` leaves finished runs alone) if the corner is at the
edge of the sampled range; the script warns about that.

---

## Input

| File | Location | Description |
|------|----------|-------------|
| `femtic.cnv` (or `FEMTIC-DABIC.cnv`) | `<ENSEMBLE_DIR><ENSEMBLE_NAME>*/` | Convergence file of every run, last row is used (alpha, roughness, misfit, nRMS, objective function, iteration) |

Run directories without a usable convergence file, or with non-positive or
non-finite values, are reported and skipped. Two runs with the same alpha
stop the script (the pairs are listed).

## Output

Stem `PLOT_NAME` (default `<ENSEMBLE_DIR>L-curve`):

| File | Description |
|------|-------------|
| `<PLOT_NAME>.<ext>` | L-curve (`PLOT_FORMATS`): x = roughness, y = nRMS or misfit, every point labelled with its alpha, corner marked by a star (`CORNER_COLOR`, default yellow); axis titles on both axes, with the scale factor on linear axes |
| `<PLOT_NAME>_curvature.<ext>` | Curvature versus alpha, both methods (`PLOT_CURVATURE`) |
| `<PLOT_NAME>_corner.txt` | Corner estimates and a table of all runs (same text as printed) |
| `<PLOT_NAME>_corner.npz` | The same numbers as arrays: `alpha`, `roughness`, `misfit`, `nrms`, `objfc`, `iteration`, `kappa_menger`, `kappa_spline_runs`, `corner_alpha`, `corner_method`, `corner_alpha_menger`, `corner_alpha_spline`, `rough_is_squared`; with the spline also `alpha_fine`, `kappa_spline` |
| `femtic_lcurve_post_summary.md` | User-set (UPPERCASE) parameters, script path, run date/time, written to `ENSEMBLE_DIR` |

---

## Corner estimate

Both estimates work in the plane

- u = ln(nRMS) (data norm),
- v = 0.5 ln(Roughness) if `ROUGH_IS_SQUARED`, else ln(Roughness) (model norm),

with the runs ordered by alpha. This does not depend on `PLOT_WHAT`.

### `"menger"` (default, `inverse.calc_lc_corner`)

The corner is the run with the largest Menger curvature 1/R, where R is the
radius of the circle through the run's point and its two neighbours
(`calc_lc_corner`, `curvature`, `circumradius` of `inverse.py`, the code of
`lc.py`). Only runs can be chosen, and not the first or the last one. The
script calls `inverse.curvature` for the values in the table and
`inverse.calc_lc_corner` for the corner, so a change in `inverse.py` is
picked up.

What the script adds around `inverse.py`:

| Item | inverse.py | script |
|------|------------|--------|
| Returned index | `calc_lc_corner` gives an index into the n-2 interior values (run index = returned + 1) | adds 1, the corner is reported as a run |
| Collinear or coincident points | `curvature` prints a message and returns the number 0 for the whole curve (`calc_lc_corner` then returns 0, which would silently be taken for the first interior point) | detected: "no estimate" for `"menger"`; with `CORNER_METHOD = "menger"` the script stops and suggests `"spline"` |
| Sign of the bend | unsigned curvature only | sign of the turn at the selected run, warning if it is not an L-shaped bend |

An opposite (non-L) bend can also have a large unsigned curvature; the
script warns if the selected run is not an L-shaped bend.

### `"spline"` (option)

Cubic splines u(t), v(t) in t = log10(alpha) and the analytic signed curvature

    kappa(t) = (u' v'' - u'' v') / (u'^2 + v'^2)^(3/2)

positive for an L-shaped bend, evaluated on a fine grid between the 2nd and
the last-but-one run. alpha is then found between the runs, is less tied to
how the alphas happen to be spaced, and only L-shaped bends count.

`SPLINE_SMOOTH`:

| Value | Meaning |
|-------|---------|
| `0.0` | Interpolating cubic spline (default) |
| `"gcv"` | Smoothing spline (`inverse.make_spline`), penalty by generalised cross validation (noisy runs); needs 5 runs |
| float > 0 | Penalty `lam` of `inverse.make_spline` (`scipy.interpolate.make_smoothing_spline`); depends on the data scale, a fixed value can oversmooth |

If the spline curvature is nowhere positive there is no spline estimate
("no estimate: ..." in the report); selecting `"spline"` then stops the
script.

### Which one?

No method is generally better; the L-curve corner is a heuristic (Engl and
Grever 1994 show it does not give a convergent parameter choice rule in
general). Both are printed, `CORNER_METHOD` only selects which one is marked
and reported first. Look at the L-curve and the curvature plot. An
interpolating spline can oscillate through a sharp corner (negative lobes in
the curvature plot).

### Lambda

FEMTIC-DABIC v2.7.1 minimises Phi = Phi_d + alpha^2 Phi_m, so the weight of
the roughness term is lambda = alpha^`LAMBDA_POWER` with `LAMBDA_POWER = 2`;
use 1 for older versions (Phi_d + alpha Phi_m), `None` to skip. The value for
`TRADE_OFF_PARAMETER` in `control.dat` is **alpha**, not lambda.

---

## Check before use: ROUGH_IS_SQUARED

Curvature in log-log space changes if only one axis is squared. If the
`Roughness` column of your `femtic.cnv` is the squared term Phi_m = |R m|^2,
the model norm is its square root (`ROUGH_IS_SQUARED = True`, default); if it
is already a norm, set `False`. The script cannot tell. Run both and compare;
if the corner differs, find out which convention your FEMTIC version uses
(in the synthetic test below the Menger result moved by one run when the
wrong setting was used).

---

## Configuration variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENSEMBLE_DIR` | `str` | `.../misti2026/l-curve/` | Directory with the run directories (as in `femtic_lcurve_prep.py`) |
| `ENSEMBLE_NAME` | `str` | `"lc_"` | Prefix of the run directories |
| `CNV_FILE` | `str` or `list` | `["femtic.cnv", "FEMTIC-DABIC.cnv"]` | Convergence file name(s); first existing wins |
| `CORNER_METHOD` | `str` | `"menger"` | `"menger"` or `"spline"`; selects the marked/reported estimate |
| `ROUGH_IS_SQUARED` | `bool` | `True` | Roughness column is the squared term (model norm = sqrt) |
| `SPLINE_SMOOTH` | `float` or `"gcv"` | `0.0` | Spline smoothing, see above |
| `LAMBDA_POWER` | `int` or `None` | `2` | lambda = alpha^LAMBDA_POWER (2: v2.7.1, 1: older, None: off) |
| `PLOT_NAME` | `str` | `ENSEMBLE_DIR + "L-curve"` | Stem of the output files |
| `PLOT_WHAT` | `str` | `"nrms"` | y axis: `"nrms"` or `"misfit"` (corner unaffected) |
| `PLOT_TITLE` | `str`, `dict`, `None` | `"auto"` | `"auto"` = "L-curve"; None/False/"" = none; a string (`{what}` = `PLOT_WHAT`); resolved by `fviz.resolve_title` |
| `FONTSIZE` | `int` | `9` | Font size |
| `PLOT_LOG_X`, `PLOT_LOG_Y` | `bool` | `True` | Log10 axes |
| `PLOT_XLIM`, `PLOT_YLIM` | `list` or `None` | `None` | Axis limits [min, max] in data units (before `SCALE_X` / `SCALE_Y`) |
| `SCALE_X`, `SCALE_Y` | `"auto"`, `None` or `float` | `"auto"` | Scale factor of a linear axis (x: roughness, y: nRMS or misfit); see below |
| `PLOT_SIZE_CM` | `list` | `[8.5, 8.5]` | Figure size in cm |
| `PLOT_CURVATURE` | `bool` | `True` | Second figure: curvature versus alpha |
| `CORNER_COLOR` | `str` | `"yellow"` | Fill colour of the corner marker in the L-curve (any matplotlib colour; edge is black) |
| `CORNER_MARKER` | `str` | `"*"` | Marker style of the corner (`"*"` = star) |
| `CORNER_SIZE` | `float` | `14` | Marker size in points |
| `PLOT_FORMATS` | `list` of `str` | `["pdf", "svg", "jpg"]` | Output formats |
| `PLOT_DPI` | `int` | `600` | Resolution of raster formats |

Not carried over from `femtic_lcurve_plot.py`: the `LC_dat.npz` file
(replaced by `<PLOT_NAME>_corner.npz`). `SCALE_ROUGH` / `SCALE_MISFIT` are
replaced by `SCALE_X` / `SCALE_Y`, see below.

### Axis scale (linear axes)

The axis titles are "model roughness" (x) and "nRMS" or "misfit" (y). For a
**linear** axis a scale factor can be applied: the plotted values are
data / factor and the title gets the suffix "/ 10^n", e.g.
"model roughness / 10^6" means the axis shows roughness divided by 10^6.

| Setting | Effect |
|---------|--------|
| `"auto"` (default) | Power of ten, multiple of 3, chosen from the data if its largest value is >= 1e4 or < 1e-2; otherwise no scaling |
| `None` or `1` | No scaling |
| number > 0 | Fixed factor; a power of ten gives "/ 10^n", any other value "/ 2" etc. |

A logarithmic axis is never scaled (the tick labels already show the powers
of ten); a number set for a log axis is ignored with a note. Unlike
`SCALE_ROUGH` / `SCALE_MISFIT` of `femtic_lcurve_plot.py`, `SCALE_Y` also
applies to nRMS, `"auto"` exists, and the corner star, annotations, spline
and `PLOT_XLIM` / `PLOT_YLIM` (given in data units) follow the scaling. The
`.txt` / `.npz` files always hold unscaled values.

---

## Test status

Tested with synthetic Tikhonov L-curves (13 runs, known corner at
alpha = 4.2e-4), written in the format of `femtic.cnv`. The test was run
first with a local copy of the Menger code and again after the switch to
`inverse.py` (the repository version, identical to the uploaded file): the
reports are identical and the curvature values agree to 3e-15.

| Method | Result | Note |
|--------|--------|------|
| `"menger"` | 3e-4 | Nearest run to the true corner (runs only) |
| `"spline"` | 5.3e-4 | Between the runs |

Monte Carlo (60 trials per case, 0.5 and 0.25 decade spacing, 0 / 2 / 5 %
noise): noise-free at 0.5 decade spacing the interpolating spline had a
median error of 0.05 decade against 0.13 decade for Menger. With noise the
advantage was small or gone; `"gcv"` smoothing was about equal or slightly
better than interpolation; a fixed `lam = 1e-2` oversmoothed.

Also tested: 3 runs (Menger works, spline stops with "needs at least 4
runs"), 4 runs, 5 runs for `"gcv"`, duplicate alpha, two runs with identical
roughness and nRMS (Menger: "no estimate", `"menger"` stops cleanly,
`"spline"` still works), missing or invalid convergence files, corner at the
edge of the sampled range, an opposite bend, missing or empty `ENSEMBLE_DIR`,
bad `CORNER_METHOD` / `SPLINE_SMOOTH`, and the plot options (no curvature
figure, single format, no title, linear axes with limits, title placeholder).

**Not tested:** real FEMTIC output (no access to any), and the Py4MTX
modules `femtic`, `femtic_viz`, `inverse`, `util` were used as in the repository, but
uninstalled third-party packages (h5py, pyvista, ...) had to be stubbed in
the test environment. Run it on one real study and check the plot before
relying on the number.

Known cosmetic limit: point labels overlap where runs cluster near the
corner in the 8.5 cm figure.

---

## Dependencies

| Module | Source | Purpose |
|--------|--------|---------|
| `numpy` | PyPI | Arrays, curvature |
| `scipy` | PyPI | `CubicSpline`; >= 1.10, because `inverse.py` imports `make_smoothing_spline` at the top |
| `matplotlib` | PyPI | Figures |
| `femtic` | Py4MTX | `read_cnv` |
| `femtic_viz` | Py4MTX | `resolve_title` |
| `inverse` | Py4MTX | `calc_lc_corner`, `curvature` (Menger corner), `make_spline` (smoothing splines); itself imports `aniso` |
| `util` | Py4MTX | `print_title`, `write_param_summary` |
| `version` | Py4MTX | Version string for the title banner |

Needs the environment variable `PY4MTX_ROOT`.

---

## Provenance

| Date | Author | Description |
|------|--------|-------------|
| 2026-11-17 | Claude Sonnet 5.5 (Anthropic) | Created. Built from `femtic_lcurve_plot.py` and `lc.py`; spline corner added as option. AI-generated; review before production use. |
| 2026-11-17 | Claude Sonnet 5.5 (Anthropic) | Revised on request: Menger corner and smoothing splines are imported from `inverse.py` instead of local copies; collinear / coincident points now give "no menger estimate". AI-generated; review before production use. |
| 2026-11-17 | Claude Sonnet 5.5 (Anthropic) | Corner marker in the L-curve is a yellow star by default; `CORNER_COLOR`, `CORNER_MARKER`, `CORNER_SIZE` added. AI-generated; review before production use. |
| 2026-11-18 | Claude Sonnet 5.5 (Anthropic) | Scale factors for linear axes (`SCALE_X`, `SCALE_Y`, "auto" or a number) shown in the axis titles; `PLOT_XLIM` / `PLOT_YLIM` now in data units. AI-generated; review before production use. |

---

## References

DOIs confirmed by web search (2026-11-17).

- Hansen, P. C. (1992). Analysis of discrete ill-posed problems by means of
  the L-curve. *SIAM Review*, 34(4), 561-580. doi:10.1137/1034115
- Hansen, P. C. and O'Leary, D. P. (1993). The use of the L-curve in the
  regularization of discrete ill-posed problems. *SIAM Journal on Scientific
  Computing*, 14(6), 1487-1503. doi:10.1137/0914086
- Engl, H. and Grever, W. (1994). Using the L-curve for determining
  optimal regularization parameters. *Numerische Mathematik*, 69, 25-31.
  doi:10.1007/s002110050078
