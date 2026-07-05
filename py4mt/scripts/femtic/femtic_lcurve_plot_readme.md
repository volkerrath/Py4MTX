# femtic_lcurve_plot

Plot the L-curve (model roughness vs. data misfit / nRMS) from a set of
FEMTIC inversion runs carried out at different regularisation parameters α.
Each run lives in its own sub-directory; the final-iteration line of
`femtic.cnv` is harvested from every matching directory and assembled into
the curve.

---

## Workflow

| Step | Action |
|------|--------|
| 1 | Glob `WORK_DIR` for directories matching `SEARCH_STRNG` |
| 2 | Read the **last line** of `femtic.cnv` in each directory |
| 3 | Extract α, roughness, misfit, nRMS, objective function |
| 4 | Sort runs by ascending α |
| 5 | Plot roughness (y) vs. misfit or nRMS (x); annotate each point with its α |
| 6 | Save `PLOT_NAME.pdf` and `PLOT_NAME.png` |

---

## Input

| File | Location | Description |
|------|----------|-------------|
| `femtic.cnv` | `<run_dir>/femtic.cnv` | FEMTIC convergence log; one line per iteration |

### `femtic.cnv` column layout

```
Iter#  Retrial#  Alpha  Damp  Roughness  Misfit  RMS  ObjFunc
  0       1        2     3       4         5       6      7
```

Only the **last line** (final iteration) is used from each file.

---

## Output

| File | Description |
|------|-------------|
| `<PLOT_NAME>.pdf` | Vector L-curve plot |
| `<PLOT_NAME>.png` | Raster L-curve plot |

---

## Configuration variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `WORK_DIR` | `str` | — | Root directory containing the per-α run sub-directories |
| `PLOT_NAME` | `str` | — | Output file base path (no extension); title derived from this |
| `PLOT_WHAT` | `str` | `"nrms"` | x-axis quantity: `"nrms"` for normalised RMS, anything else for raw misfit |
| `PLOT_LOG_X` | `bool` | `False` | Apply log₁₀ scale to the x-axis (misfit / nRMS) |
| `PLOT_LOG_Y` | `bool` | `False` | Apply log₁₀ scale to the y-axis (roughness) |
| `PLOT_XLIM` | `list` or `None` | `None` | x-axis limits `[xmin, xmax]`; `None` = matplotlib auto |
| `PLOT_YLIM` | `list` or `None` | `None` | y-axis limits `[ymin, ymax]`; `None` = matplotlib auto |
| `SCALE_ROUGH` | `float` or `None` | `None` | Divide roughness values by this factor before plotting; shown as "×10ⁿ" next to the roughness label. `None`/`1` disables. |
| `SCALE_MISFIT` | `float` or `None` | `None` | Divide misfit values by this factor before plotting; shown as "×10ⁿ" next to the misfit label. Ignored when `PLOT_WHAT = "nrms"`. `None`/`1` disables. |
| `SEARCH_STRNG` | `str` | `"*L2"` | Glob pattern passed to `utl.get_filelist` to identify run directories |

### `PLOT_WHAT` detail

| Value | x-axis quantity | Formula shown |
|-------|----------------|---------------|
| `"nrms"` | Normalised RMS | $\mathrm{nRMS}=\sqrt{\frac{1}{N}\sum_i\left[\mathbf{C}_d^{-1/2}(d_i^\mathrm{obs}-d_i^\mathrm{pred})\right]^2}$ |
| anything else | Raw misfit | $\Vert\mathbf{C}_d^{-1/2}(\mathbf{d}_\mathrm{obs}-\mathbf{d}_\mathrm{calc})\Vert_2$ |

### Axis scaling factors

`SCALE_ROUGH` and `SCALE_MISFIT` divide the plotted data by a constant so
that large values (e.g. roughness in the tens of thousands) display with a
compact axis. The factor is echoed next to the axis label, e.g. setting
`SCALE_ROUGH = 1e3` turns the roughness label into:

```
roughness x10^3
```

If the factor is not a clean power of ten, it is shown verbatim
(e.g. `x2.5`) instead. `SCALE_MISFIT` has no effect when
`PLOT_WHAT = "nrms"`, since nRMS values are already normalised and do not
need rescaling. Both default to `None` (no scaling). Data written to
`LC_dat.npz` is always unscaled, regardless of these settings.

`PLOT_XLIM` and `PLOT_YLIM` are entered in **raw, unscaled units** and are
automatically divided by `SCALE_ROUGH` / `SCALE_MISFIT` internally — no
manual adjustment needed when turning scaling on or off.

### Log-axis combinations

| `PLOT_LOG_X` | `PLOT_LOG_Y` | Effect |
|:---:|:---:|--------|
| `False` | `False` | Linear–linear (default) |
| `True` | `False` | Log–linear (log misfit) |
| `False` | `True` | Linear–log (log roughness) |
| `True` | `True` | Log–log (true L-curve in log space) |

---

## Point annotations

Each data point is labelled with its α value, rounded to one significant
figure via:

```python
round(a[k], -int(np.floor(np.log10(abs(a[k])))))
```

Labels are offset from the marker centre by `(markersize + 2)` points in
both x and y, using `ax.annotate` with `textcoords="offset points"`.  This
keeps the text clear of the marker regardless of axis scale.  Adjust the
offset tuple `ann_offset` in the script if crowding still occurs.

---

## Dependencies

| Module | Source | Purpose |
|--------|--------|---------|
| `numpy` | PyPI | Array operations, argsort, log10 |
| `matplotlib` | PyPI | Plotting and annotation |
| `femtic` | Py4MTX | (imported; not directly called in current version) |
| `util` | Py4MTX | `get_filelist`, `print_title` |
| `version` | Py4MTX | Version string for title banner |

---

## Provenance

| Date | Author | Description |
|------|--------|-------------|
| 2025 | vrath | Created |
| 2026-03-03 | vrath / Claude Sonnet 4.6 | Renamed user-set parameters to UPPERCASE |
| 2026-06-17 | vrath / Claude Sonnet 4.6 | Added `PLOT_LOG_X` / `PLOT_LOG_Y` for independent log₁₀ axes |
| 2026-06-18 | vrath / Claude Sonnet 4.6 | Added `PLOT_XLIM` / `PLOT_YLIM`; offset annotations from markers via `xytext` |
| 2026-07-05 | vrath / Claude Sonnet 5 | Added `SCALE_ROUGH` / `SCALE_MISFIT` optional axis scaling factors, displayed as "×10ⁿ" in the axis label; not applied to `nrms` or to saved `LC_dat.npz`; `PLOT_XLIM`/`PLOT_YLIM` auto-rescaled to match |
