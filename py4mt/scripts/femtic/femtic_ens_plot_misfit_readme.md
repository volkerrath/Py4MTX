# femtic_ens_plot_misfit.py

nRMS histogram of a FEMTIC ensemble, drawn from the results `.npz` of
`femtic_ens_post.py`.

The histogram shows the per-member best nRMS of **all scanned members**:
accepted members (including those taken from the fallback iteration,
`ITER_FALLBACK`) and rejected ones stacked, the acceptance threshold
(`NRMS_MAX`) as a dashed line and the median of the accepted members. It is the
quickest check of how many inversions converged and where the cut lies.

The table of all scanned members is stored in the `.npz` as
`<P>_nrms_table` (with `<P>_nrms_max`), so the script needs nothing but the
results file. For results written before 2026-11-08 the CSV of
`femtic_ens_post.py` (`NRMS_CSV`) can be given instead. With a random
subsample (`ENS_SUBSAMPLE_N` in `femtic_ens_post.py`) the members that were
accepted but not drawn (status `accepted_not_drawn`) count as accepted, so the
histogram describes the whole scan.

Since 2026-11-08 this is a separate script; the code was a part of
`femtic_ens_post.py` (`_plot_nrms_hist`). The drawing is now
`femtic_viz.plot_nrms_histogram`, the table I/O `femtic.read_nrms_table`
and `femtic.write_nrms_table`.

---

## Configuration parameters

Defaults in the tables are the documented defaults; the UPPERCASE user section of
the script holds the values currently set. Every setting has a `#:` comment
there with the full text.

### Input

| Parameter | Type | Description |
|---|---|---|
| `ENSEMBLE_DIR` | str | Ensemble directory (used to build the default paths). |
| `ENSEMBLE_PREFIX` | str | `ENSEMBLE_PREFIX` of `femtic_ens_post.py`, including the suffix for `"perturbed"` or a subsample. |
| `RESULTS_DIR` | str | Where `femtic_ens_post.py` wrote its results; the figure goes here as well. |
| `ENSEMBLE_RESULTS` | str | The results `.npz`; default `RESULTS_DIR + ENSEMBLE_PREFIX.upper() + "_results.npz"`. |
| `NPZ_PREFIX` | str or None | Key prefix inside the `.npz`; `None` = detect. |
| `NRMS_CSV` | str or None | Fallback for results files without the nRMS table: the CSV written by `femtic_ens_post.py`, e.g. `RESULTS_DIR + ENSEMBLE_PREFIX + "_nrms.csv"`. `None` = no fallback. Used only if the `.npz` holds no table; if neither exists, the histogram shows the accepted members of the ensemble only. |
| `NRMS_MAX` | float or None | Acceptance threshold marked in the plot. `None` = the value stored in the `.npz`; if the file has none, no line is drawn. |

### Histogram and figure

| Parameter | Type | Default | Description |
|---|---|---|---|
| `MOD_NRMS_BINS` | int or str | `"auto"` | Bins: an int, or any rule `numpy.histogram_bin_edges` accepts (`"auto"`, `"fd"`, `"sturges"`, ...). |
| `MOD_NRMS_HIST_FILE` | str or None | `None` | Extension-less output path; one file per `MOD_PLOT_FORMAT` entry. `None` = `RESULTS_DIR/<prefix>_nrms_hist`. |
| `MOD_PLOT_FORMAT` | list of str | `["pdf"]` | File extensions. |
| `MOD_DPI` | int | `400` | Figure DPI. |
| `MOD_TICK_FONTSIZE`, `MOD_LABEL_FONTSIZE` | int or None | `16`, `16` | Axis annotation and label sizes (`None` = Matplotlib's default). |
| `MOD_NRMS_TITLE_FONTSIZE` | int or None | `None` | Title size. `None` = Matplotlib's default: in the old code the font-size argument of the title was lost inside a comment, so the title always had the default size; that stays the default. |
| `MOD_SHOW_IN_SPYDER` | bool | `True` | Also display the saved figure inline when running inside Spyder. |

---

## Output files

```
<RESULTS_DIR>/<prefix>_nrms_hist.<fmt>     the histogram (one file per MOD_PLOT_FORMAT entry)
femtic_ens_plot_misfit_summary.md          settings of the run (next to the script)
```

---

## Quick start

```bash
conda activate EM
python femtic_ens_post.py          # once: writes the .npz with the nRMS table
# edit ENSEMBLE_DIR, ENSEMBLE_PREFIX in femtic_ens_plot_misfit.py
python femtic_ens_plot_misfit.py
```

---

## Relationship to other scripts

| Script | Purpose |
|---|---|
| `femtic_ens_post.py` | Writes the results `.npz` (and the CSV) this script reads. |
| `femtic_ens_plot_slices.py`, `femtic_ens_plot_bh.py` | Slice and borehole figures from the same `.npz`. |
| `femtic_ens_repair.py` | Convergence diagnostic over all scanned directories (nRMS bar chart, histogram) and REPAIR of non-converged ones. |

---

## Changelog

| Date | Author | Change |
|---|---|---|
| 2026-11-08 | Claude Sonnet 5.5 (Anthropic) | Created: the nRMS histogram of `femtic_ens_post.py` as a stand-alone script that reads the table from the results `.npz` (`<P>_nrms_table`, `<P>_nrms_max`; CSV fallback). Drawing in `femtic_viz.plot_nrms_histogram`, table I/O in `femtic.py`. `MOD_NRMS_TITLE_FONTSIZE` added. |
