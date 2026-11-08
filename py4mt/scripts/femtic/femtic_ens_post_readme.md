# femtic_ens_post.py

Algorithm-agnostic postprocessing of a FEMTIC ensemble: **calculations only**.

Collects all converged inversion members, computes summary statistics in
log10(rho) space, the optional sensitivity measures, the empirical covariance
and a bootstrap variance, and writes

- the results `.npz` (`ENSEMBLE_RESULTS`) with **everything the plot scripts
  need**: ensemble matrix, statistics, member list, best member, the perturbed
  models (`ENS_MODEL_SOURCE = "both"`), and the nRMS table of all scanned
  members;
- the nRMS table as CSV (`NRMS_CSV`);
- the parameter summary `femtic_ens_post_summary.md`, including the directory
  scan report.

It draws nothing. Since 2026-11-08 the figures come from three separate
scripts that read the results `.npz` (see "Relationship to other scripts"):

| Script | Makes |
|---|---|
| `femtic_ens_plot_slices.py` | best-member and statistics slice panels |
| `femtic_ens_plot_bh.py` | borehole profiles and heat maps |
| `femtic_ens_plot_misfit.py` | nRMS histogram |

so that colours, slices, vertical exaggeration, the reference model or the
borehole list can be changed and re-plotted in seconds, without rescanning
the member directories. The former `PLOT_ONLY` mode of this script is what
these scripts do; the settings `PLOT_ONLY*`, `MOD_QC`, `MOD_STATS*`,
`MOD_SLICES`, `MOD_BH*`, `MOD_REF_*` and the other `MOD_*` plot parameters
moved with them (their documentation is in the readmes of the plot scripts).

Optionally only **N members drawn at random** from the accepted ones are used
(`ENS_SUBSAMPLE_N`, see "Random subsample").

Supersedes `femtic_rto_post.py`.

For a convergence-rate diagnostic over **every** scanned directory and a
REPAIR procedure for non-converged directories, see the companion script
`femtic_ens_repair.py` (and `femtic_ens_repair_readme.md`).

---

## Workflow

```
Ensemble sub-directories  (rto_*, gst_*, member_*, ...)
          |
          v
  nRMS filter  (NRMS_MAX, ITER_FALLBACK)          -> accepted members
          |
          v
  optional random draw of N members  (ENS_SUBSAMPLE_N)
          |
          v
  Load log10(rho)  ->  N_members x N_free matrix
          |
          +--> mean, variance, median, MAD, percentiles, Q-differences
          +--> bootstrap variance                (BOOTSTRAP_VAR)
          +--> sensitivity, SimRC                (COMPUTE_SENS, COMPUTE_SIMRC)
          +--> empirical covariance              (COMPUTE_COV, COV_METHOD)
          |
          v
  <PREFIX>_results.npz     <PREFIX>_nrms.csv     femtic_ens_post_summary.md
          |
          +--> femtic_ens_plot_slices.py   best + statistics slice figures
          +--> femtic_ens_plot_bh.py       borehole profiles, heat maps
          +--> femtic_ens_plot_misfit.py   nRMS histogram
```

---

## Configuration parameters

All parameters are in the UPPERCASE user section at the top of the script.

### Ensemble input

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ENSEMBLE_DIR` | str | — | Root directory containing ensemble sub-directories. |
| `ENSEMBLE_NAME` | str | `"rto_*"` | Glob matched against sub-directory names. Only actual directories among the matches are scanned as ensemble runs (`os.path.isdir()` checked in step (1)); a stray file that happens to match the glob is skipped rather than being mistaken for a failed inversion run. |
| `ENSEMBLE_PREFIX` | str | `"rto"` | Prefix for `.npz` output keys and default filenames. Set to `"gst"`, `"ens"`, etc. as appropriate. |
| `ENSEMBLE_INDEX_RANGE` | tuple or None | `None` | Expected member numbers `(first, last)`, inclusive, e.g. `(0, 255)` for 256 members named `<ENSEMBLE_NAME>0` ... `<ENSEMBLE_NAME>255`. Used only by the "Directory numbering" part of the scan report in the parameter summary. `None` = the range found in the names (cannot show a missing first or last member). |
| `CNV_FILE` | str or list | `"femtic.cnv"` | Name of the convergence file in each member directory; a list gives candidates (first existing wins). |
| `NRMS_MAX` | float | `1.4` | Members whose final nRMS exceeds this value are skipped. |
| `ITER_FALLBACK` | bool | `True` | A member's model is taken from its best-nRMS iteration N. If `resistivity_block_iter<N>.dat` does not exist, `True` uses the best iteration that has a block file (iteration > 0, lowest nRMS among those with a file, ties: lower iteration -- usually N-1) and applies `NRMS_MAX` to *that* iteration's nRMS; the member enters with that iteration's model, sensitivity file and nRMS. It is logged and has status `accepted_fallback` in `NRMS_CSV`. With no usable alternative the member is skipped as `no_model_file`. `False` = skip without looking for another iteration. |
| `ENS_MODEL_SOURCE` | str | `"converged"` | Which model of each member enters the ensemble. `"converged"`: the best-fit model `resistivity_block_iter<N>.dat` (as before). `"perturbed"`: the member's **perturbed reference** (the RTO prior draw), file `ENS_PERTURBED_FILE`, so that the prior ensemble is plotted instead of the converged one; see "Plotting the perturbed references" below. `"both"`: the run is the converged run, and the perturbed models of the same members are loaded in addition for the **borehole figure** (`MOD_BH`), which then lets `femtic_ens_plot_bh.py` show both ensembles on one colour bar; see "Converged and perturbed together" below. |
| `ENS_PERTURBED_FILE` | str | `"referencemodel.dat"` | File name, inside each member directory, of the perturbed reference. `femtic_rto_prep.py` writes the draw to `referencemodel.dat` (a real file per member). `resistivity_block_iter0.dat` holds it only if the ensemble was prepared with `MOD_START_FROM_PERTURBED = True`; otherwise it is a link to the shared, unperturbed template. |
| `ENS_PERTURBED_SUFFIX` | str | `"_perturbed"` | Appended to `ENSEMBLE_PREFIX` when `ENS_MODEL_SOURCE = "perturbed"`, so the results `.npz`, nRMS table, figures and `.npz` keys never overwrite those of the converged run. |


#### Using the perturbed references

```python
ENS_MODEL_SOURCE = "perturbed"      # default "converged"
# ENS_PERTURBED_FILE = "referencemodel.dat"
```

With `"perturbed"` the whole run (mean, median, percentiles, covariance, the
best member) describes the **prior** ensemble instead of the posterior one.
Rules:

- **Same members.** Which members take part is still decided by the converged inversion (`NRMS_MAX`, `ITER_FALLBACK`, a block file must exist for the best iteration), so the prior and the converged ensemble contain the same members and compare like for like. Raise `NRMS_MAX` to include every member. `"best"` is the member with the best converged nRMS; its perturbed reference is drawn.
- **Separate outputs.** `ENSEMBLE_PREFIX` gets `ENS_PERTURBED_SUFFIX` (`annecy_rto` -> `annecy_rto_perturbed`): results `.npz` (`ANNECY_RTO_PERTURBED_results.npz`), `*_nrms.csv` and `.npz` keys are separate files / keys. The `.npz` stores `<P>_model_source` and `<P>_model_file_name`. In the plot scripts set the same `ENSEMBLE_PREFIX` (or `ENSEMBLE_RESULTS`) to plot this run; the stored source is reported and used for the figure titles.
- **Switched off.** `COMPUTE_SIMRC` (it correlates the models with the converged responses; a message is printed). `COMPUTE_SENS` stays on: the sensitivities are those of the converged inversion.
- **Checks.** A member without `ENS_PERTURBED_FILE` is skipped (status `no_perturbed_file` in `NRMS_CSV`). A warning is printed if the file is a symbolic link in some members (shared template = unperturbed) or if all members are identical (zero spread).
- **Reference model.** The base, *unperturbed* reference (`MOD_REF_FILE` in the slice and borehole plot scripts) is never replaced by a member's file.

#### Converged and perturbed together (`"both"`)

```python
ENS_MODEL_SOURCE = "both"
```

The run is the **converged** run: `ENSEMBLE_PREFIX` is unchanged, and the statistics, covariance and results `.npz` are those of the converged ensemble. In addition every member's `ENS_PERTURBED_FILE` is read (row for row, same member order) and stored.

- **Stored.** The results `.npz` keeps `<P>_model_source = "converged"` and gains `<P>_ens_perturbed` (log10 rho, same shape as `<P>_ens`) and `<P>_perturbed_file_name`. `femtic_ens_plot_bh.py` uses them to draw every borehole for **both** ensembles (heat map, lines, converged and perturbed side by side or in separate figures; `MOD_BH_HEAT_FIGURE`).
- **Skipped members.** A member without `ENS_PERTURBED_FILE` is skipped *entirely* (`no_perturbed_file` in `NRMS_CSV`), so both matrices always hold the same members. A perturbed model with a different number of free parameters stops the run.
- **Warnings.** As for `"perturbed"`: symbolic-link or zero-spread perturbed files are reported. `COMPUTE_SIMRC` stays on (the ensemble is the converged one).

### Random subsample

```python
ENS_SUBSAMPLE_N      = 32        # None = all accepted members
ENS_SUBSAMPLE_SEED   = 42        # None = a different draw each run
ENS_SUBSAMPLE_SUFFIX = "_sub{n}" # may contain {n} and {seed}
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ENS_SUBSAMPLE_N` | int or None | `None` | `None` = use all accepted members. An integer N draws N members at random, **without replacement**, from the **accepted** members (those that passed `NRMS_MAX` / `ITER_FALLBACK` and have a model file; with `"perturbed"` / `"both"` also a perturbed file). Everything is then computed from these N only: statistics, sensitivity, SimRC, covariance, bootstrap, the `"best"` member (= best nRMS **within the draw**), the member list and the results `.npz`. `N >=` number of accepted members: all are used (a message is printed). |
| `ENS_SUBSAMPLE_SEED` | int or None | `None` | Seed of the draw. `None` = fresh OS entropy; an integer gives a reproducible draw. |
| `ENS_SUBSAMPLE_SUFFIX` | str | `"_sub{n}"` | Appended to `ENSEMBLE_PREFIX` when a draw is made, so the `.npz`, the nRMS table and the keys of the subsample never overwrite those of the full ensemble (`annecy_rto` -> `annecy_rto_sub32`; results file `ANNECY_RTO_SUB32_results.npz`). `{n}` = `ENS_SUBSAMPLE_N`, `{seed}` = `ENS_SUBSAMPLE_SEED`, e.g. `"_sub{n}_s{seed}"` to keep several draws side by side. The suffix is added after `ENS_PERTURBED_SUFFIX` if both apply. |

Notes:

- **What is drawn from.** The draw is made from the *accepted* members of the directory scan that is run now (the scan still reads every accepted member's model once). Drawing from an existing results `.npz` without rescanning is not implemented. For a fast run switch `COMPUTE_SENS` / `COMPUTE_SIMRC` off.
- **Sensitivity / SimRC** are computed for the drawn members only (their rows are remapped; members whose HDF5 file is missing are skipped there as in a full run).
- **Reproducibility.** The rows drawn are stored in the `.npz` as `<P>_subsample_idx` (indices into the accepted members in scan order), `<P>_subsample_n_accepted` and `<P>_subsample_seed` (`-1` = no seed), so a draw made with `ENS_SUBSAMPLE_SEED = None` can be reproduced afterwards.
- **nRMS table.** `NRMS_CSV` and `<P>_nrms_table` still list **all** scanned members (the histogram of the convergence rate is not about the draw); the accepted members that were not drawn get the status `accepted_not_drawn` (counted as accepted by the histogram). The drawn members are those in `<P>_model_list`.
- **Plotting.** Point the plot scripts at the subsample by using the same `ENSEMBLE_PREFIX` (with the suffix) or by setting `ENSEMBLE_RESULTS` to the subsample file.
- **Use.** Typical uses: how the spread and the percentiles depend on the ensemble size (draw N = 16, 32, 64 ... with different seeds), a quick preview of a large ensemble, or checking that a result does not hinge on a few members.

### Statistics

| Parameter | Type | Default | Description |
|---|---|---|---|
| `PERCENTILES` | list of float | `[2.3, 15.9, 50.0, 84.1, 97.7]` | Percentile levels (2-σ / 1-σ normal-equivalent). |
| `QDIFF_PAIRS` | list of `(lo, hi)` | `[(15.9, 84.1), (2.3, 97.7)]` | Percentile-pair differences `\|P_hi - P_lo\|` computed as extra, outlier-robust spread statistics (both values must also appear in `PERCENTILES`). The default gives both a 1-sigma-equivalent (`15.9`/`84.1`) and 2-sigma-equivalent (`2.3`/`97.7`) spread. Saved as `<P>_qdiff_<lo>_<hi>` and plottable under the same key in `MOD_STATS_WHAT`. |


### Sensitivity statistics

Low ensemble spread (`VAR`/`ERR`/`MAD` above) is ambiguous on its own: it
can mean the data genuinely constrain a cell, or that regularisation
collapsed a null-space cell regardless of the data. This section adds
two independent sensitivity diagnostics to tell the two apart. Either diagnostic can be switched off independently and
degrades gracefully (printed warning, corresponding `.npz` keys and
panels in the plot script skipped) if its required per-member files are
missing for some or all members.

> **Two aggregation orders, computed side by side.** FEMTIC's per-block
> sensitivity in `results_iterX.h5` (whichever `SENS_KIND` dataset is
> used) is confirmed **not** normalised to each member's own max, so
> there is no single "correct" way to bring per-member sensitivities onto
> a common, threshold-friendly scale before aggregating across the
> ensemble. Both orders below are always computed when `COMPUTE_SENS`
> succeeds (suffixes `_na`/`_an` throughout — config, `.npz` keys,
> `MOD_STATS_WHAT`/`MOD_ALPHA_SOURCE` names in `femtic_ens_plot_slices.py`):
> - **`_na` (normalise, then average)** — each member's raw vector is
>   divided by *its own* max right after reading, so every member
>   contributes equally regardless of its absolute sensitivity scale,
>   then `mean`/`median`/`min`/`max`/`std`/`cv` are taken across the
>   (now per-member-normalised) ensemble. Its own max is generally
>   `< 1` (different members peak at different cells).
> - **`_an` (average, then normalise)** — `mean`/`median`/`min`/`max`/
>   `std`/`cv` are taken across the raw ensemble first (a larger-
>   magnitude member dominates, as a plain ensemble mean would), then
>   the resulting aggregate is divided by *its own* max, so it always
>   peaks at exactly `1.0`.
>
> Both are max-normalised statistics.
>
> **log10.** All sensitivity aggregates (`mean`/`median`/`min`/`max`/`std`,
> `_raw`/`_na`/`_an`) and the SimRC sums (`simrc_coef`/`simrc_corr`) are
> strictly positive and are stored, plotted and used in `MOD_ALPHA_SOURCE` (`femtic_ens_plot_slices.py`)
> as **log10** (values <= 0 -> NaN). Normalised sensitivities are
> therefore <= 0, e.g. `"sens_mean_an > -3."` keeps cells within three
> decades of the maximum. `sens_cv_*` are ratios and stay linear.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `COMPUTE_SENS` | bool | `True` | (1) Per-member linearised cumulative sensitivity, aggregated across the ensemble. Requires a `results_iter{numit}.h5` alongside each accepted member's `resistivity_block_iterX.dat` (FEMTIC's current, settled 2026-09-13 merged HDF5 output — see note below). |
| `SENS_H5_PATTERN` | str | `"results_iter{numit}.h5"` | Per-member HDF5 filename, relative to the member's run directory, with `{numit}` substituted by that member's best (accepted) iteration — the same iteration used for `ens_matrix`. Older ensembles predating the 2026-09-13 merged schema (separate `model_iterX.h5`) are not supported here. |
| `SENS_H5_GROUP` | str or `None` | `None` | HDF5 group containing the sensitivity dataset(s); `None` tries the file root, then falls back to `"model/sensitivity"` automatically (FEMTIC's real location in `results_iterX.h5`). |
| `SENS_KIND` | str | `"volume_normalised"` | Which precomputed per-block sensitivity dataset to use: `"raw"` (`sum\|J[:,block]\|`) or `"volume_normalised"` (the same, divided by block volume — sensitivity density). Passed as `fem.read_h5_sensitivity()`'s `cumsens_key`. Confirmed against an actual `results_iterX.h5` that neither dataset is normalised to a member's own max, so two independent aggregation orders are always computed instead of picking one — see `sens_mean_na`/`sens_mean_an` below. |
| `SENS_JACOBIAN_KEY` | str | `"jacobian"` | Dataset name for a raw 2-D Jacobian, used only if `SENS_KIND`'s dataset is absent. In practice this never resolves against `results_iterX.h5` (the raw Jacobian only lives in the separate, fixed-name `jacobian.h5`, deliberately not consulted here — see below), so a missing `SENS_KIND` dataset just means that member is skipped for sensitivity. |
| `SENS_ERROR_KEY` | str or `None` | `"data_errors"` | Dataset name for a 1-D per-datum error vector used to weight the Jacobian sum if the fallback above ever fires; `None` disables weighting. |

> **No `jacobian.h5` fallback, by design.** If a member's `results_iter{numit}.h5` doesn't have sensitivity for that iteration (e.g. sensitivity calculation was disabled in `control.dat` for that step), this script does **not** fall back to reading `jacobian.h5` — that file is separate, has a fixed name, and may hold an entirely different (arbitrary "last computed") iteration than `numit`. That member is simply skipped for `COMPUTE_SENS`, with a printed warning, same as any other missing-file case.
| `COMPUTE_SIMRC` | bool | `True` | (2) Ensemble-native "SimRC" sensitivity from the model ensemble and the calculated responses (`cal_re`/`cal_im`, errors `re_err`/`im_err`) in `/data/data` of each member's `results_iter<numit>.h5` (same file as the sensitivity). Real parts of all rows plus imaginary parts of complex datatypes; columns with error <= 0 dropped. Members with NaN calculated values or a different data composition are skipped. |
| `SIMRC_ERR_WEIGHT` | bool | `True` | Divide each data column by its error (first usable member), so `simrc_coef` sums over mixed datatypes are dimensionless. `simrc_corr` is unaffected. |
| `SIMRC_CHUNK` | int | `200` | Data columns processed per chunk when accumulating the model/data cross-covariance — bounds peak memory to `O(SIMRC_CHUNK * n_free)` instead of forming the full `(n_data, n_free)` matrix. |

**Display options moved.** `SENS_ROI_NORMALIZE` (re-centre every plotted `sens_*`
panel on the ROI's own maximum) and the ROI value-range diagnostic belong to the
plotting and are now in `femtic_ens_plot_slices.py`.

**Why two measures instead of one.** (1) is a classical linearised
diagnostic (Christiansen & Auken, 2012, *Geophysics* 77, WB171,
doi:10.1190/geo2011-0393.1) computed once per member and then
aggregated — the aggregation (mean/median/min/max/std/CV across members)
guards against trusting a single, possibly locally atypical,
linearisation point; a large coefficient of variation (`<P>_sens_cv_na`/`<P>_sens_cv_an`)
flags cells where the linearised sensitivity itself is strongly
model-dependent (i.e. the problem is locally nonlinear there), which is
exactly where (2) should be trusted instead. (2) (Bobe, Keller & Van De
Vijver, 2021, *Geophysical Prospecting*, doi:10.1111/1365-2478.13068) is
built directly from the finite ensemble spread rather than an
infinitesimal derivative, so it remains valid in that nonlinear regime,
at the cost of needing a converged forward run (not just a Jacobian) per
member. SimRC in short: for each datum, the ensemble covariance between the
forward responses and each model parameter, divided by that parameter's
ensemble variance, is a regression coefficient that equals the
differential sensitivity for a linear forward problem; dividing by both
standard deviations gives a correlation. Summing `|coef|` / `|corr|` over
all data gives the cumulative measures stored here.

**Model iterX.h5 schema is provisional.** No writer for this file exists
elsewhere in the codebase yet; `fem.read_h5_sensitivity()`'s dataset
names are all configurable arguments (see `femtic_readme.md`) precisely
so this section can be pointed at whatever layout the actual FEMTIC HDF5
export ends up using once that pipeline is finalised (see the open item
"HDF5 archive schema compatibility with FEMTIC v5").


### Covariance

| Parameter | Type | Default | Description |
|---|---|---|---|
| `COMPUTE_COV` | bool | `True` | Set `False` to skip covariance estimation entirely. Statistics, percentiles, and slice plots are unaffected; only the `*_cov*` keys are omitted from the `.npz`. |
| `COV_METHOD` | str | `"full"` | `"full"` = dense empirical covariance. `"low_rank"` = thin SVD of the centred ensemble instead — see below. |
| `SPARSIFY` | bool | `True` | (`COV_METHOD="full"` only) Threshold small entries in the covariance to create a CSR sparse version. |
| `SPARSE_THRESH` | float | `1e-8` | Relative threshold: entries with `|C_ij| / max|C| ≤ SPARSE_THRESH` are zeroed. |

**`COV_METHOD="low_rank"`.** The empirical covariance of `n_members`
samples has rank ≤ `n_members - 1`. Since `n_members` (tens to a few
hundred) is normally far smaller than `n_free` (thousands to hundreds of
thousands of mesh cells), forming the dense `n_free × n_free` covariance
is both the slowest step in the script and, past a few thousand free
parameters, not something that fits in memory at all (e.g. `n_free=1e5`
→ 80 GB just to store it).

Instead, `"low_rank"` takes the thin SVD of the centred ensemble matrix
`Xc` (shape `n_members × n_free`):

```
Xc = U S Vᵀ         (economy SVD, cost O(n_members² · n_free))
C  = Xcᵀ Xc / (m-1) = Vᵀᵀ diag(S²/(m-1)) Vᵀ     — exact, not an approximation
```

This is stored as `f"{P}_cov_eigval"` (`(r,)`, `r = min(n_members, n_free)`)
and `f"{P}_cov_eigvec"` (`(n_free, r)`) in place of the dense `f"{P}_cov"`.
The full covariance reconstructs exactly as
`eigvec @ diag(eigval) @ eigvec.T`, and downstream consumers that only
need matrix-vector products, low-rank sampling, or leading eigenpairs
(e.g. `gdm.py`, prior sampling in `ensembles.py`) can use the factors
directly without ever forming the dense matrix. Cost drops to
`O(n_members² · n_free)` time and `O(n_members · n_free)` memory.

**Other ways to speed up `COV_METHOD="full"`**, if the dense matrix is
genuinely needed downstream:
- Make sure NumPy/SciPy are linked against a multi-threaded BLAS (OpenBLAS
  or MKL) — the `Xᵀ X` product inside `empirical_covariance` is a single
  `dgemm` call and will use all cores automatically; check with
  `numpy.show_config()` and set `OMP_NUM_THREADS` / `MKL_NUM_THREADS`
  before launching Python if it's pinned to one core.
- Cast `ens_matrix` to `float32` before the covariance call if the extra
  precision isn't needed — halves memory traffic and roughly doubles
  BLAS throughput on most hardware.
- If only a fixed block/neighbourhood structure of `C` is ever queried
  (e.g. covariance restricted to a region of interest), compute that
  sub-block directly (`Xc[:, idx].T @ Xc[:, idx]`) instead of the full
  matrix and slicing afterwards.
- For truly large `n_free` where even the low-rank factors are awkward,
  a randomized SVD (`sklearn.utils.extmath.randomized_svd` on `Xc`, or
  `scipy.sparse.linalg.svds`) approximates the same factorisation with
  fewer passes over the data — usually unnecessary here since the thin
  SVD above is already exact and cheap, but relevant if `n_members`
  itself becomes large (thousands).
- `joblib.Parallel` / `multiprocessing` can shard the outer product sum
  across process pools for the dense case, but for this problem shape
  (`n_members` small, `n_free` large) the algorithmic fix (`"low_rank"`)
  outperforms parallelizing the naive computation by orders of magnitude,
  so that's the recommended first step rather than adding process-level
  parallelism to the dense path.


### Bootstrap variance estimation

| Parameter | Type | Default | Description |
|---|---|---|---|
| `BOOTSTRAP_VAR` | bool | `False` | Enable an alternative bootstrap estimate of the ensemble variance, computed and saved alongside the plug-in `VAR`. |
| `BOOTSTRAP_N` | int | `500` | Number of bootstrap resamples. |
| `BOOTSTRAP_SEED` | int / `None` | `None` | `None` = fresh OS entropy; an integer makes the bootstrap reproducible. |

The plug-in variance (`np.var(ens_matrix, axis=0)`) uses each member
exactly once — a single point estimate. When `N_members` is small (order
30–100, typical for RTO/GST ensembles), that estimate itself can be
noisy. `BOOTSTRAP_VAR=True` resamples the `N_members` members with
replacement `BOOTSTRAP_N` times, computes the plug-in variance of each
resample, and reports:

- **`var_boot`** — the mean plug-in variance across all resamples. Generally
  a smoother, more stable estimate than the single `VAR` value, though for
  well-behaved ensembles the two are usually close (see the worked example
  in the script's own smoke-test: with 40 synthetic members drawn from a
  known distribution, plug-in and bootstrap variance agreed to within a
  few percent).
- **`var_boot_se`** — the bootstrap standard error *of `var_boot` itself*
  (i.e. how much the variance estimate would be expected to jitter from
  one 32-64-member ensemble to another) — a diagnostic of estimator
  noise, not a spread statistic of the model. Not plotted by default;
  add `"var_boot_se"` to `MOD_STATS_WHAT` and an entry to `MOD_STATS_STYLE`
  in `femtic_ens_plot_slices.py` if you want a slice figure of it.
- **`err_boot`** = `sqrt(var_boot)`, on the same scale as `MAD`/`QDIFF`,
  stored in the `.npz` when `BOOTSTRAP_VAR=True`; `femtic_ens_plot_slices.py`
  plots it when `"err_boot"` is in its `MOD_STATS_WHAT`.

Cost is `O(BOOTSTRAP_N × n_members × n_free)` time and `O(n_free)` memory
(running sums, not a stored `(BOOTSTRAP_N, n_free)` array) — independent
of mesh size in memory, and linear in `BOOTSTRAP_N` in time. `BOOTSTRAP_N
= 500` is a reasonable default; reduce for a quick look, increase (e.g.
2000+) to shrink `var_boot_se` further if the plug-in-vs-bootstrap spread
matters for your analysis.


### Output

| Parameter | Type | Default | Description |
|---|---|---|---|
| `RESULTS_DIR` | str | `<ENSEMBLE_DIR>/results/` | Directory for **all** outputs of this script (results `.npz`, nRMS table). The plot scripts read the `.npz` from here and write their figures here by default. Created if missing. |
| `ENSEMBLE_RESULTS` | str | `<RESULTS_DIR>/<PREFIX.upper()>_results.npz` | Path of the output `.npz` file. |
| `NRMS_CSV` | str | `<RESULTS_DIR>/<PREFIX>_nrms.csv` | Per-member best-nRMS table (CSV): member, best iteration, nRMS, status (`accepted`, `accepted_fallback`, `accepted_not_drawn`, `rejected_nrms_max`, `no_model_file`, `no_perturbed_file`). Always written. The same table is stored in the `.npz` (`<P>_nrms_table`). |
| `OUT` | bool | `True` | Verbose console output. |

### Block-file reading

| Parameter | Type | Default | Description |
|---|---|---|---|
| `MOD_OCEAN` | bool / None | `None` | Ocean flag handed to the resistivity-block reader; must match the FEMTIC inversion that produced the ensemble. `None` = auto-infer; `True`/`False` forces ocean-present/absent. The air / ocean sentinel resistivities and the mesh belong to the plot scripts. |


---

## Output files

### `.npz` archive

Keys follow the pattern `<PREFIX>_<stat>` (`<P>` below; `ENSEMBLE_PREFIX` with
`ENS_PERTURBED_SUFFIX` / `ENS_SUBSAMPLE_SUFFIX` where they apply). The plot
scripts read the file with `femtic.read_ensemble_results()`.

| Key | Shape | Description |
|---|---|---|
| `<P>_model_list` | `(N, 3)` | `[block_file, n_iter, nRMS]` per accepted member. |
| `<P>_ens` | `(N_members, N_free)` | Stacked ensemble in log₁₀(Ω·m). Rows are the converged models, or the members' perturbed references with `ENS_MODEL_SOURCE = "perturbed"`. |
| `<P>_model_source` | str | `"converged"` or `"perturbed"`: which model of each member the rows of `<P>_ens` are (`<P>_model_list` then lists the perturbed files). Absent in files from before 2026-10-31 (= converged). |
| `<P>_ens_perturbed` | `(N_members, N_free)` | Only with `ENS_MODEL_SOURCE = "both"`: the members' perturbed models in log10(Ohm.m), row for row as `<P>_ens` (which stays the converged ensemble, `<P>_model_source = "converged"`). Read by `femtic_ens_plot_bh.py` for the borehole figure. `<P>_perturbed_file_name` holds the file name. |
| `<P>_model_file_name` | str | `ENS_PERTURBED_FILE`, or `resistivity_block_iter<N>.dat`. |
| `<P>_cov` | `(N_free, N_free)` | Empirical covariance matrix. Present only if `COMPUTE_COV=True` and `COV_METHOD="full"`. |
| `<P>_cov_eigval` | `(r,)` | Covariance eigenvalues, `r = min(N_members, N_free)`. Present only if `COMPUTE_COV=True` and `COV_METHOD="low_rank"`. |
| `<P>_cov_eigvec` | `(N_free, r)` | Covariance eigenvectors; `C = eigvec @ diag(eigval) @ eigvec.T`. Present only if `COMPUTE_COV=True` and `COV_METHOD="low_rank"`. |
| `<P>_avg` | `(N_free,)` | Element-wise mean over members. |
| `<P>_var` | `(N_free,)` | Element-wise variance over members (plug-in, ddof=0). |
| `<P>_err` | `(N_free,)` | `sqrt(var)` — standard deviation, on the same scale as `MAD`/`QDIFF`. |
| `<P>_med` | `(N_free,)` | Element-wise median over members. |
| `<P>_mad` | `(N_free,)` | Median absolute deviation. |
| `<P>_prc` | `(N_prc, N_free)` | Percentile values at `PERCENTILES` levels. |
| `<P>_prc_levels` | `(N_prc,)` | The `PERCENTILES` levels themselves, for self-describing output. |
| `<P>_qdiff_<lo>_<hi>` | `(N_free,)` | `\|P_hi - P_lo\|` per free parameter, one key per `QDIFF_PAIRS` entry (e.g. `<P>_qdiff_15_9_84_1`, `<P>_qdiff_2_3_97_7`). |
| `<P>_var_boot` | `(N_free,)` | Bootstrap mean variance across `BOOTSTRAP_N` resamples. Present only if `BOOTSTRAP_VAR=True`. |
| `<P>_err_boot` | `(N_free,)` | `sqrt(var_boot)`. Present only if `BOOTSTRAP_VAR=True`. |
| `<P>_var_boot_se` | `(N_free,)` | Bootstrap standard error of `var_boot` itself (estimator-noise diagnostic, not a model spread statistic). Present only if `BOOTSTRAP_VAR=True`. |
| `<P>_sens_mean_raw` | `(N_free,)` | Mean per-member cumulative sensitivity across the ensemble, **no normalisation**: `sens_matrix.mean(axis=0)` in `SENS_KIND`'s native units (`"raw"` = `sum\|J\|`, or `"volume_normalised"` = `sum\|J\|`/block volume). Present only if `COMPUTE_SENS=True` and at least one member's `results_iterX.h5` was found. Plottable as the `"sens_mean_raw"` `MOD_STATS` panel, on the same slice geometry as the model itself. |
| `<P>_sens_mean_na` | `(N_free,)` | Mean per-member cumulative sensitivity across the ensemble, **normalise-then-average**: each member's raw vector is divided by its own max before averaging (Christiansen & Auken, 2012). Present only if `COMPUTE_SENS=True` and at least one member's `results_iterX.h5` was found. |
| `<P>_sens_median_na/_min_na/_max_na/_std_na` | `(N_free,)` | Corresponding median, min, max, and standard deviation (ddof=1) for the `_na` aggregation. Same presence condition as `<P>_sens_mean_na`. |
| `<P>_sens_cv_na` | `(N_free,)` | Coefficient of variation, `sens_std_na/sens_mean_na` — large values flag cells where the linearised sensitivity is strongly model-dependent (nonlinear regime). Same presence condition as `<P>_sens_mean_na`. |
| `<P>_sens_mean_an` | `(N_free,)` | Mean per-member cumulative sensitivity across the ensemble, **average-then-normalise**: the raw ensemble is averaged first, then the result is divided by its own max (always peaks at exactly `1.0`). Same presence condition as `<P>_sens_mean_na`. |
| `<P>_sens_median_an/_min_an/_max_an/_std_an` | `(N_free,)` | Corresponding median, min, max, and standard deviation (ddof=1) for the `_an` aggregation, all divided by the same renormalisation factor as `<P>_sens_mean_an`. Same presence condition as `<P>_sens_mean_na`. |
| `<P>_sens_cv_an` | `(N_free,)` | Coefficient of variation, `sens_std_an/sens_mean_an` (renormalisation cancels in the ratio, so this equals the raw ensemble's CV). Same presence condition as `<P>_sens_mean_na`. |
| `<P>_simrc_coef` | `(N_free,)` | Cumulative `\|regression coefficient\|` between the forward-response ensemble and the model ensemble (Bobe, Keller & Van De Vijver, 2021). Present only if `COMPUTE_SIMRC=True` and at least one member's result files were found. |
| `<P>_simrc_corr` | `(N_free,)` | Cumulative `\|correlation\|`, the normalised counterpart of `<P>_simrc_coef`. Same presence condition. |
| `<P>_best` | `(N_free,)` | log10(rho) of the best-nRMS member (within the draw when subsampled), free blocks. Plotted as the `"best"` panel by `femtic_ens_plot_slices.py` (the block-file structure comes from a template, see there). |
| `<P>_best_member` / `_best_file` / `_best_iter` / `_best_nrms` | str / str / int / float | Directory name, model file, iteration and nRMS of that member. |
| `<P>_nrms_table` | `(n_scanned, 4)` str | nRMS table of **all** scanned members, accepted or not: member, best iteration, nRMS, status (same content as `NRMS_CSV`). Read by `femtic_ens_plot_misfit.py`. Absent in files from before 2026-11-08 (the CSV is then used). |
| `<P>_nrms_max` | float | `NRMS_MAX` of the run (the threshold line in the histogram). |
| `<P>_subsample_idx` | `(N,)` int | Only with `ENS_SUBSAMPLE_N`: indices of the drawn members among the accepted members (scan order). |
| `<P>_subsample_n_accepted` | int | Only with `ENS_SUBSAMPLE_N`: number of accepted members the draw was made from. |
| `<P>_subsample_seed` | int | Only with `ENS_SUBSAMPLE_N`: `ENS_SUBSAMPLE_SEED`, `-1` if it was `None`. |

If `COMPUTE_COV=False`, none of the `<P>_cov*` keys are present.
All `<P>_sens_*` keys except `<P>_sens_cv_*`, and `<P>_simrc_coef`/`<P>_simrc_corr`, are **log10** since 2026-09-25 (non-positive values -> NaN).
If `COMPUTE_SENS=False`, or no accepted member's `results_iterX.h5` was found, none of the `<P>_sens_*` keys are present.
If `COMPUTE_SIMRC=False`, or no accepted member's result files were found, neither `<P>_simrc_coef` nor `<P>_simrc_corr` is present.

### Figures

This script writes no figures and no statistics block files. The block files
of the statistics panels, the slice figures, the borehole figures and
`<MOD_BH_FILE>.npz` are written by the plot scripts; see their readmes.

### Parameter summary and directory scan report

At the end of every run `femtic_ens_post_summary.md` is written by
`utl.write_param_summary()` (the UPPERCASE settings, script path, run
date/time; next to the script, as this script calls it without `out_dir`).
After every run a section
**"Directory scan report"** is appended to that file. It accounts for every
scanned directory and says why each one that is *not* in the ensemble was left
out:

| Reason | Meaning |
|---|---|
| `no_cnv_file` | none of the `CNV_FILE` names exists in the directory |
| `cnv_unreadable` | the convergence file has no usable header row |
| `cnv_no_iterations` | the convergence file has no data rows or no iteration > 0 |
| `rejected_nrms` | best nRMS (iterations > 0) above `NRMS_MAX` |
| `no_model_file` | no `resistivity_block_iter<N>.dat` for the best iteration and no usable fallback (the detail names the iteration and why no fallback was possible) |
| `no_perturbed_file` | `ENS_MODEL_SOURCE` `"perturbed"` / `"both"`: `ENS_PERTURBED_FILE` is missing (the member itself converged) |

The section holds the counts (scanned = accepted + not accepted; a warning line
appears if that does not add up), a count-per-reason table, a per-directory
table (directory, reason, detail), the members accepted via `ITER_FALLBACK`
(best iteration -> iteration used), and glob matches that are not directories.
`NRMS_CSV` is unchanged: it has no row for directories without a usable
convergence file, the report does. If the summary file cannot be appended to,
the report is written to `femtic_ens_post_scan_report.md` in its place.
**Directory numbering.** The report always gives the number of glob matches next
to the number of directories scanned (they differ when a match is a file or a
broken symbolic link). It then reads the number that follows `ENSEMBLE_NAME` in
every matching name and compares it with the expected range
(`ENSEMBLE_INDEX_RANGE`, default: the range found in the names). It lists, if
present:

- numbers with **no entry at all** (nothing starts with `<ENSEMBLE_NAME><number>`: never created, deleted, renamed, or elsewhere) -- this is how a "255 scanned, 256 expected" difference is traced to a number;
- numbers matched **only by a non-directory** (file, broken symbolic link);
- numbers matched by **more than one name** (e.g. `X` and `X_restart`);
- names with a number outside the expected range, and names without a number.

Zero-padded numbers are fine (`007` = 7). With the default (inferred) range a
missing first or last member cannot be seen; set `ENSEMBLE_INDEX_RANGE = (0, 255)`
for a known ensemble size.

Plain ASCII only (other characters in directory names show as `?`).


## Quick start

```bash
conda activate EM
# 1. Compute (slow: scans the ensemble). Edit ENSEMBLE_DIR, ENSEMBLE_NAME,
#    ENSEMBLE_PREFIX, NRMS_MAX at the top of the script.
python femtic_ens_post.py

# 2. Plot (fast, repeat as often as needed; each script has its own settings)
python femtic_ens_plot_slices.py
python femtic_ens_plot_bh.py
python femtic_ens_plot_misfit.py

# Variants of step 1:
#   ENS_MODEL_SOURCE = "perturbed" | "both"          prior (perturbed) models
#   ENS_SUBSAMPLE_N = 32, ENS_SUBSAMPLE_SEED = 1     32 random members
```

---

## Bug fix vs. femtic_rto_post.py

The original script computed mean, variance, and median with `axis=1`,
which reduced over the **free-parameter** axis rather than the **member**
axis.  The ensemble matrix has shape `(N_members, N_free)`, so the
correct reduction axis is `axis=0`.  All aggregate statistics are now
correct.

---

## Relationship to other scripts

| Script | Purpose |
|---|---|
| `femtic_rto_prep.py` | Generate RTO ensemble members. |
| `femtic_gst_prep.py` | Generate GST ensemble members. |
| **`femtic_ens_post.py`** | Postprocess any ensemble: statistics, sensitivity, covariance, bootstrap; writes the results `.npz`. No plotting. |
| `femtic_ens_plot_slices.py` | Reads the `.npz`: best-member and statistics slice panels (`MOD_QC` / `MOD_STATS` of the former `femtic_ens_post.py`). |
| `femtic_ens_plot_bh.py` | Reads the `.npz`: borehole profiles and heat maps (`MOD_BH` of the former `femtic_ens_post.py`). |
| `femtic_ens_plot_misfit.py` | Reads the `.npz`: nRMS histogram (formerly part of `femtic_ens_post.py`). |
| `femtic_ens_repair.py` | Convergence diagnostic over all scanned directories and REPAIR of non-converged ones; shares the scan-report helpers (`femtic.build_scan_report`) with this script. |
| `femtic_ens_plot.py` | Per-member figures and borehole logs (older, independent script; overlaps with `femtic_ens_plot_bh.py`). |
| `femtic_mod_math.py` | Write average and smoothed-median as block files from an N-subset. |
| `femtic_mod_edit.py` | Apply arithmetic operations to a single model. |
| `femtic_mod_plot_slice.py` | Plot slice figures from a single model file. |

---

## Changelog

| Date | Author | Change |
|---|---|---|
| 2026-11-08 | Claude Sonnet 5.5 (Anthropic) | **Split into a calculation script and three plot scripts.** `femtic_ens_post.py` now only scans, computes and writes the results `.npz`, the nRMS CSV and the parameter summary with the scan report; all plotting (`MOD_QC`, `MOD_STATS*`, `MOD_SLICES`, `MOD_BH*`, `MOD_REF_*`, the other `MOD_*` plot settings, `SENS_ROI_NORMALIZE`, the nRMS histogram) and `PLOT_ONLY` / `PLOT_ONLY_NPZ` / `PLOT_ONLY_TEMPLATE` were removed; the plot scripts `femtic_ens_plot_slices.py`, `femtic_ens_plot_bh.py` and `femtic_ens_plot_misfit.py` read the `.npz`. Reusable code moved to `femtic.py` (section 6c: `read_ensemble_results`, `resolve_results_template`, `write_nrms_table`, alpha helpers, `ReferenceModel`, `BoreholeMesh`, `build_scan_report`, ...) and `femtic_viz.py` (`plot_nrms_histogram`, `plot_boreholes`, `draw_borehole_figures`, ...). New keys in the `.npz`: `<P>_nrms_table`, `<P>_nrms_max`. **New option: random subsample** (`ENS_SUBSAMPLE_N`, `ENS_SUBSAMPLE_SEED`, `ENS_SUBSAMPLE_SUFFIX`): N members drawn at random without replacement from the accepted members; everything is computed from the draw; new keys `<P>_subsample_idx`, `<P>_subsample_n_accepted`, `<P>_subsample_seed`; output names get the suffix. The duplicate scan-report helpers in `femtic_ens_repair.py` were replaced by the shared `femtic.build_scan_report` / `femtic.append_scan_report`. |
| 2025-04-30 | vrath | Created as `femtic_rto_post.py`. |
| 2026-03-03 | Claude (Anthropic) | Renamed user-set parameters to UPPERCASE; generated README. |
| 2026-05-27 | vrath / Claude Sonnet 4.6 (Anthropic) | Added `femtic_viz` import; `PLOT_QC` block with minimal `plot_model_slices` call. |
| 2026-06-11 | vrath / Claude Sonnet 4.6 (Anthropic) | Renamed → `femtic_ens_post.py`; fixed `axis` bug in mean/var/median/MAD; replaced thin `PLOT_QC` block with full CRS-aware `_plot_slice()` helper; added `PLOT_STATS` block (writes block files + figures for avg/var/med/MAD); added `ENSEMBLE_PREFIX` config var for generic naming. |
| 2026-07-07 | vrath / Claude Sonnet 5 (Anthropic) | Renamed the entire plotting config surface to match `femtic_gst_prep.py` / `femtic_rto_prep.py` exactly (`MOD_*` prefix throughout: mesh, ocean/air, UTM origin, display coords, site overlay, slice specs, colormap/limits, figure layout). Added `MOD_OCEAN`/`MOD_AIR_RHO`, `MOD_SITE_NUMBER` (observe.dat fallback), `MOD_AIR_COLOR`, `MOD_ALPHA_FILE/MODE/BLANK_THRESH`, `MOD_PANEL_WIDTH`, `MOD_FIGSIZE`. Removed a latent duplicate `MOD_XLIM/YLIM/ZLIM` assignment that silently discarded the first (non-`None`) values. A config block can now be copied between `femtic_ens_post.py` and the ensemble-generation scripts with no renaming. |
| 2026-07-09 | vrath / Claude Sonnet 5 (Anthropic) | Merged `MOD_QC_DPI` / `MOD_STATS_DPI` into a single `MOD_DPI` knob, matching `femtic_gst_prep.py` and `femtic_nss.py` (one figure-DPI setting per script, not one per plot type). `_plot_slice()` no longer takes a `dpi` argument; it reads `MOD_DPI` directly. |
| 2026-07-17 | Claude Sonnet 5 (Anthropic) | `scipy.sparse`: migrated from legacy matrix to array-equivalent API — `scs.csr_matrix(tmp)` → `scs.csr_array(tmp)` when building the sparsified empirical covariance (`ens_covs`). No functional change; `ens_covs` is only used for its `.nnz` count. |
| 2026-07-25 | Claude Sonnet 5 (Anthropic) | Added `COMPUTE_COV` (skip covariance entirely) and `COV_METHOD="low_rank"` (exact thin-SVD factorisation of the centred ensemble, avoiding the dense `N_free × N_free` matrix — see Covariance section). `MOD_STATS` now also writes a block file and slice figure for each `PERCENTILES` level (`p2_3`, `p50`, `p97_7`, …), included by default in `MOD_STATS_WHAT`. Added `<P>_prc_levels` to the `.npz` output. |
| 2026-07-25 | Claude Sonnet 5 (Anthropic) | Added `MOD_STATS_CLIM` for per-statistic colour scaling (`VAR`/`MAD`/`QDIFF` default to a fixed `[-2, 2]` range instead of silently reusing `MOD_CLIM`, which made them blank; set a key to `None` for auto per-panel scaling instead). Added `QDIFF_PAIRS` (default `[(15.9, 84.1)]`): percentile-difference spread statistics, saved to the `.npz` and plottable via `MOD_STATS`. Added `MOD_ROI_AUTO`/`MOD_ROI_PAD_XY`/`MOD_ROI_ZLIM`: automatic `MOD_XLIM`/`MOD_YLIM`/`MOD_ZLIM` from the site bounding box, which also activates `femtic_viz.py`'s existing aspect-ratio panel-width logic so map/ns/ew panels size themselves differently. Changed `MOD_NROWS`/`MOD_NCOLS` defaults to `2`/`2`. Also fixed a sign bug in `femtic_viz.py`'s `plot_model_slices` (ns/ew curtain panels rendered blank/upside down whenever `MOD_ZLIM` was set) — the bug was dormant under the old `MOD_ZLIM=None` default and is fixed as part of enabling `MOD_ROI_AUTO`. |
| 2026-07-25 | Claude Sonnet 5 (Anthropic) | Added `MOD_TICK_FONTSIZE` / `MOD_LABEL_FONTSIZE`, passed through to every `_plot_slice()` call (`MOD_QC` and `MOD_STATS`). Axis tick labels, axis labels, panel titles, and colourbar text were previously fixed at `plot_model_slices`' internal defaults with no way to override them from this script. |
| 2026-07-25 | Claude Sonnet 5 (Anthropic) | Added `MOD_SHOW_IN_SPYDER` (default `True`): when running inside Spyder, every saved figure is also displayed inline via `plt.show()` (`fviz.plot_model_slices`' new `show=` parameter), without changing what gets saved to disk. No effect outside Spyder. |
| 2026-07-25 | Claude Sonnet 5 (Anthropic) | Added `(2.3, 97.7)` to the default `QDIFF_PAIRS` (alongside `(15.9, 84.1)`), giving both 1-sigma- and 2-sigma-equivalent spread statistics. Added `<P>_err = sqrt(var)`: `MOD_STATS_WHAT`'s default now plots `"err"` instead of `"var"`, since `var` (in (log10 Ω·m)²) was never on the same scale as `MAD`/`QDIFF` (log10 Ω·m); `var` remains available on request. Added `BOOTSTRAP_VAR`/`BOOTSTRAP_N`/`BOOTSTRAP_SEED` and the new `_bootstrap_variance()` helper: an alternative bootstrap estimate of the ensemble variance (resample members with replacement, average the plug-in variance of each replicate), reporting `var_boot`, `err_boot` (added to `MOD_STATS_WHAT` automatically when enabled), and `var_boot_se` (the bootstrap estimate's own standard error, an estimator-noise diagnostic). See the new "Bootstrap variance estimation" section. |
| 2026-08-10 | Claude Sonnet 5 (Anthropic) | `femtic_viz.py` fix: `"N-S"` curtain panels (`MOD_SLICES` entries with `kind="ns"`) were plotting mirrored left-right relative to true geography — verified against two real ensemble QC figures where a resistive body sitting north-central in the `"map"` panels appeared on the `"S"` side of the `"N-S"` panel. Root cause and fix are entirely inside `fviz.plot_model_slices`' internal `_axis_slice_params()` helper (see `femtic_viz_readme.md` for details); no changes needed in this script beyond picking up the updated `femtic_viz.py`. `"ew"` and `"map"` panels were unaffected. Also added `MOD_TICK_DECIMALS` (default `None` = unchanged formatting): controls the number of decimal digits shown on depth / easting-northing / lat-lon axis tick labels, passed through to `fviz.plot_model_slices`' new `tick_decimals` parameter in `_plot_slice()`. |
| 2026-08-12 | Claude Sonnet 5 (Anthropic) | Step (1)'s scan loop now checks `os.path.isdir(d)` before looking for `femtic.cnv`/the model file inside it. `dir_list` comes from a glob-style match (`utl.get_filelist(searchstr=[ENSEMBLE_NAME+"*"])`) against `ENSEMBLE_DIR`, which can in principle return non-directory matches (e.g. a stray file sharing the `ENSEMBLE_NAME` prefix); previously such an entry fell through to the `femtic.cnv` check and got printed as a skipped ensemble member as if it were a failed inversion run. Non-directory matches are now skipped immediately with their own log message. No change for genuine run directories. |
| 2026-08-12 | Claude Sonnet 5 (Anthropic) | A convergence diagnostic (nRMS bar chart / binned histogram over all scanned directories) and a REPAIR procedure (rebuild non-converged directories with a starting model averaged from 2 random converged members, for a restart) were prototyped in this script across several iterations today, then moved out into a new standalone script, `femtic_ens_repair.py` (see `femtic_ens_repair_readme.md`), once the design settled — keeping this script focused on its original scope (summary statistics, covariance, QC/statistics slice plots). No `MOD_CONV*`/`MOD_REPAIR*` config or related step remains here. `fviz.plot_convergence_bar()` / `fviz.plot_convergence_histogram()` (added to `femtic_viz.py` during the same work; see `femtic_viz_readme.md`) are unaffected and now used by `femtic_ens_repair.py` instead. |
| 2026-08-13 | Claude Sonnet 5 (Anthropic) | Added `femtic_ens_post_summary.md` output at end of run: writes user-set (UPPERCASE) parameters, script path, and run date/time. |
| 2026-08-21 | Claude Sonnet 5 (Anthropic) | Added `COMPUTE_VAR_REDUX` (default `True`): the scan loop now also reads each accepted member's iter0 (prior) model into `ens_matrix_prior`, from which `<P>_var_prior` (variance of the prior ensemble) and `<P>_var_redux = 1 - var/var_prior` (fractional variance reduction from the inversion) are computed and saved to the `.npz`. Computed only if every accepted member's `resistivity_block_iter0.dat` is found; otherwise skipped with a warning, with no effect on the rest of the run. `"var_redux"` is added to `MOD_STATS_WHAT` automatically (same pattern as `err_boot` for `BOOTSTRAP_VAR`), with `MOD_STATS_CLIM` defaults `var_prior: [-.0, .5]` (matching `var`) and `var_redux: [0.0, 1.0]` (bounded fraction; override to `None` for auto-scaling). |
| 2026-08-21 | Claude Sonnet 5 (Anthropic) | Added `MOD_STATS_BLANK_BY_REDUX` (default `False`) + `REDUX_EPS` (default `0.1`): when enabled, free parameters with `var_redux < REDUX_EPS` are blanked (`MOD_STATS_BLANK_MODE`, default `"blank"`) in every `MOD_STATS` plot except `var_redux`'s own, using the same alpha/blanking mechanism as `MOD_ALPHA_FILE`/`MOD_ALPHA_MODE`/`MOD_ALPHA_BLANK_THRESH` but sourced from the in-memory `var_redux` array. `_plot_slice()` now accepts optional per-call `alpha_file`/`alpha_mode`/`alpha_blank_thresh` overrides (`None` = fall back to the existing module-level `MOD_ALPHA_*` settings), so `MOD_QC` and any run with the new option left off are unaffected. |
| 2026-08-24 | Claude Sonnet 5 (Anthropic) | Step (1)'s best-iteration selection fixed: (a) `dir_list` is now filtered to `os.path.isdir()` entries immediately after `utl.get_filelist()`, since that function matches on filename only (`fnmatch` over `os.listdir`) and can return non-directory hits — folding the earlier 2026-08-12 inline `os.path.isdir(d)` check into an upfront filter rather than a per-iteration skip. (b) The "best" iteration for each ensemble member is now the one with the **smallest nRMS across the entire `femtic.cnv` convergence history**, not simply the last row — FEMTIC does not guarantee monotonic nRMS reduction, so the last iteration was not necessarily the best. `iter0` (prior/starting model) is excluded from eligibility; a `WARNING` is printed if `iter0`'s nRMS is `<=` the best eligible `iter>0` nRMS, since that indicates the inversion did not improve on the starting model. Ties (identical minimum nRMS at two-plus iterations) resolve to the first (lowest-iteration) occurrence. Runs with no eligible `iter>0` row are skipped with a message rather than raising. |
| 2026-09-02 | Claude Sonnet 5 (Anthropic) | Fixed the best-iteration column lookup: replaced the `"4.3"`/`"5."` version-string switch on `nrms_col` (`6` vs `8`) with `fem.read_cnv()`, which reads column positions from `femtic.cnv`'s own header row. The `Beta`/`Distortion` columns only appear when distortion parameters are being inverted, shifting `RMS`'s column independently of FEMTIC version -- a 4.3 run *with* distortion has `RMS` at the same column as a typical 5.x run, not at plain 4.3's position, so the version-only switch silently read the wrong column (`Roughness` instead of `RMS`) for that combination. The now-unused `FEMTIC` config variable was removed. Iteration/nRMS selection logic (iter0 exclusion, tie-breaking, the iter0-nRMS warning) is unchanged. |
| 2026-09-06 | Claude Sonnet 5 (Anthropic) | Added `MOD_SHOW_MODEL_CENTRE` (default `True`): marks the model origin on `"map"` panels whenever `MOD_DISPLAY_COORDS` is `"utm"`/`"latlon"`, via `fviz.plot_model_slices`'s new `show_model_centre` parameter. Override style with a `MOD_MAP_MARKERS` entry carrying `"is_model_centre": True`. |
| 2026-09-08 | Claude Sonnet 5 (Anthropic) | Added a "Sensitivity statistics" section (new config block, between `REDUX_EPS` and Covariance) mitigating the ambiguity that low ensemble spread (`VAR`/`ERR`/`MAD`) can mean either genuine data control or a regularisation-collapsed null-space cell. Two independent diagnostics, each separately switchable: (1) `COMPUTE_SENS` — per-member linearised cumulative sensitivity (Christiansen & Auken, 2012, doi:10.1190/geo2011-0393.1), read from each accepted member's Jacobian via the new `fem.read_h5_sensitivity()` (`SENS_H5_PATTERN="model_iter{numit}.h5"`, schema-provisional — see `femtic_readme.md`), aggregated across the ensemble (`sens_mean`/`_median`/`_min`/`_max`/`_std`/`_cv`); large `sens_cv` flags cells where the linearised sensitivity is itself model-dependent (nonlinear regime), where (2) should be trusted instead. (2) `COMPUTE_SIMRC` — ensemble-native "SimRC" sensitivity (Bobe, Keller & Van De Vijver, 2021, doi:10.1111/1365-2478.13068): per-member forward ("cal") responses collected via the new `fem.collect_forward_response()` (`SIMRC_RESULT_FILES`, `SIMRC_SITE_FILE`) into a data ensemble, then cumulative `\|regression coefficient\|`/`\|correlation\|` against the model ensemble computed in `SIMRC_CHUNK`-sized data-column chunks (no Jacobian needed). Model rows are matched to exactly the members that also produced a forward-response row (`simrc_keep_idx`), since `COMPUTE_SENS`/`COMPUTE_VAR_REDUX` can drop different members than `COMPUTE_SIMRC` does. Both diagnostics degrade gracefully (printed warning, `.npz` keys and `MOD_STATS_WHAT` entries omitted) when required per-member files are missing, following the existing `COMPUTE_VAR_REDUX` pattern. Added `COMPUTE_NULL_SPACE_FLAG` (default `True`): flags cells with `ERR` below its own `FLAG_SPREAD_PERCENTILE` **and** low sensitivity by every available measure (`AND`, not `OR` — conservative) as `<P>_flag_null_space`, plottable via `MOD_STATS` the same way `var_redux` is. New `.npz` keys: `<P>_sens_mean/_median/_min/_max/_std/_cv/_low_mask`, `<P>_simrc_coef/_corr/_low_mask`, `<P>_flag_null_space`. New auto-added `MOD_STATS_WHAT` entries: `"sens_mean"`, `"sens_cv"`, `"simrc_corr"`, `"flag_null_space"`. |
| 2026-09-17 | Claude Sonnet 5 (Anthropic) | Generalised the `MOD_STATS` blanking/fading scheme: replaced the single hardcoded `MOD_STATS_BLANK_BY_REDUX` boolean with `MOD_STATS_BLANK_SOURCES`, a list of `(source_name, direction, thresh)` triples so `sens_mean`, `sens_cv`, `simrc_corr`, and `flag_null_space` can each be used as a blanking source (singly or combined via the new `MOD_STATS_BLANK_COMBINE`, `"and"`/`"or"`, default `"and"`, with a printed warning whenever more than one source is listed). `direction` is `"below"` (var_redux/sens_mean/simrc_corr: low means poorly resolved), `"above"` (sens_cv: high means the linearised sensitivity itself is untrustworthy there — a different question from resolution, per Bobe, Keller & Van De Vijver, 2021, doi:10.1111/1365-2478.13068, on differential-vs-regression sensitivity divergence under nonlinearity), or `"flag"` (flag_null_space is already boolean). Default is an empty list (blanking off), matching the previous default. Internally each source becomes a plain boolean mask in Python before being combined and written as one 0/1 alpha-block file, so `femtic_viz.py`'s existing alpha mechanism is untouched. Added `SENS_CV_HIGH_THRESH` (default `1.0`) and `NULL_SPACE_COMBINE` (`"and"`/`"or"`, default `"and"`, matching prior hardcoded behaviour), generalising `flag_null_space`'s own `sens_low_mask`/`simrc_low_mask` combination the same way. |
| 2026-09-16 | Claude Sonnet 5 (Anthropic) | Adapted `COMPUTE_SENS` to FEMTIC's settled (2026-09-13) HDF5 schema: `model_iterX.h5`/`data_iterX.h5` were merged into one `results_iterX.h5` per iteration (sibling `/model`, `/data`, `/distortion` groups), with the per-block cumulative sensitivity now precomputed by FEMTIC itself at `/model/sensitivity/{raw,volume_normalised}` — no Jacobian assembly needed. `SENS_H5_PATTERN` changed from `"model_iter{numit}.h5"` to `"results_iter{numit}.h5"` (older, pre-2026-09-13 ensembles are no longer supported by this script). Replaced the hardcoded `SENS_CUMSENS_KEY` with `SENS_KIND` (`"raw"` \| `"volume_normalised"`), matching `fem.read_h5_sensitivity()`'s updated `cumsens_key` default and automatic `model/sensitivity` group fallback (`femtic.py`, same date). No fallback to `jacobian.h5` if a member's `results_iter{numit}.h5` lacks sensitivity for that iteration (by design — `jacobian.h5` is a separate, fixed-name file that may hold a different iteration entirely): that member is skipped for sensitivity with a printed warning, same as before. Updated stale `model_iterX.h5` wording throughout this README and the script's comments/warnings to match. |
| 2026-09-19 | Claude Sonnet 5 (Anthropic) | Plot-extent consistency with `femtic_rto_prep.py` / `femtic_gst_prep.py`: no functional change to this script. Its `MOD_ROI_AUTO`/`MOD_ROI_PAD_XY`/`MOD_ROI_ZLIM` block is the reference; the two prep scripts now carry the same variables with the same defaults and semantics. Corrected stale unit/default statements for `MOD_XLIM`/`YLIM`/`ZLIM`, `MOD_ROI_PAD_XY` and `MOD_ROI_ZLIM` in this README (values are km, not metres; defaults `2.0` and `[-1.0, 7.0]`). |
| 2026-09-22 | Claude Sonnet 5 (Anthropic) | Added a simple, max-normalised "small spread" blanking option alongside the existing sensitivity one: `SPREAD_LOW_THRESH_FRAC` (default `0.05`, mirroring `SENS_LOW_THRESH_FRAC`) and `spread_low_mask` (`ens_err < SPREAD_LOW_THRESH_FRAC * max(ens_err)`, always computed, saved as `<P>_spread_low_mask`). Registered as `source_name="err"` in `MOD_STATS_BLANK_SOURCES` (keyed to match the `"err"` entry in the internal stat map, so it's correctly excluded from its own plot like the other sources). Does not change `flag_null_space`, which still uses the percentile-based `FLAG_SPREAD_PERCENTILE`. |
| 2026-09-23 | Claude Sonnet 5 (Anthropic) | Fixed `MOD_STATS_BLANK_SOURCES` not blanking anything regardless of threshold: `"sens_mean"` and `"err"` are now compared on a consistently max-normalised scale wherever the blank mask is built, so a plain fraction (`SENS_LOW_THRESH_FRAC`/`SPREAD_LOW_THRESH_FRAC`, e.g. `0.05` = "5% of max") is always a valid threshold regardless of `SENS_KIND`, physical units, or ensemble size — previously an absolute number like `0.01` was compared against `sens_mean`'s raw (`SENS_KIND`- and run-dependent) scale, which for `"volume_normalised"` sensitivity in particular could sit nowhere near typical `sens_mean` values, silently blanking nothing no matter what threshold was tried. Added `SENS_RENORMALISE` (default `True`): since FEMTIC normalises each member's own per-block sensitivity to that member's own max=1 before writing it, `sens_mean`/`_median`/`_min`/`_max`/`_std` are now renormalised to `sens_mean`'s own max=1 immediately after ensemble aggregation, matching FEMTIC's convention and making `MOD_STATS_CLIM["sens_mean"]` default to the fixed `[0, 1]` range (previously auto-scaled). `MOD_STATS_BLANK_SOURCES`'s default list now references `SENS_LOW_THRESH_FRAC`/`SPREAD_LOW_THRESH_FRAC` directly instead of a guessed absolute number. Blanking now applies to every `MOD_STATS` panel, including a listed source's own plot (previously excluded); `MOD_QC` is still unaffected. Added two new plottable diagnostic panels, analogous to `flag_null_space`: `"spread_low_mask"` (always available) and `"sens_low_mask"` (when `COMPUTE_SENS` succeeds) — boolean 0/1 maps of exactly which cells each criterion would blank, auto-added to `MOD_STATS_WHAT`, useful for tuning thresholds visually before combining them via `MOD_STATS_BLANK_SOURCES`. *(Superseded same-day by the entry above once the max=1 assumption behind `SENS_RENORMALISE` was checked against real data and found false.)* |
| 2026-09-23b | Claude Sonnet 5 (Anthropic) | Per user confirmation against an actual `results_iter10.h5`: per-member sensitivity is **not** normalised to that member's own max, as previously assumed. Replaced the single `SENS_RENORMALISE` switch with two side-by-side aggregation pipelines, always both computed when `COMPUTE_SENS` succeeds (suffixes `_na`/`_an` throughout config, `.npz` keys, and `MOD_STATS_WHAT`/`MOD_STATS_BLANK_SOURCES` entries): **`_na`** (normalise, then average) divides each member's raw vector by its own max right after reading, then aggregates across the (now per-member-normalised) ensemble; **`_an`** (average, then normalise) aggregates the raw ensemble first, then divides the resulting aggregate by its own max. `flag_null_space`'s low-sensitivity combination now considers `sens_low_mask_na`, `sens_low_mask_an`, and `simrc_low_mask` (was: a single `sens_low_mask` + `simrc_low_mask`). The old singular `sens_mean`/`_median`/`_min`/`_max`/`_std`/`_cv`/`_low_mask` names and `SENS_RENORMALISE` are gone; every consumer (blanking, plotting, `.npz` output) now works from the two suffixed sets. |
| 2026-09-23c | Claude Sonnet 5 (Anthropic) | Changed `MOD_STATS_BLANK_SOURCES`' default from three separately-ANDed sources (`sens_mean_na`/`_an`, `err`) to `flag_null_space` alone. Rationale: low spread and low sensitivity only need blanking when they **coincide** (regularisation parked a cell near the prior and the ensemble never got pushed away) — low spread with high sensitivity is a genuine result, and low sensitivity with high spread is already visible as spread in `VAR`/`ERR`/`MAD`, so neither alone should hide a cell. `flag_null_space` already encodes exactly that `AND`, and does so using `FLAG_SPREAD_PERCENTILE` (a percentile of the spread distribution) rather than an absolute fraction-of-max threshold, so it can't silently collapse to zero blanked cells the way a miscalibrated `SPREAD_LOW_THRESH_FRAC` ANDed directly against the sensitivity sources just did in practice. The three individual sources remain available, commented out, for per-criterion diagnostic tuning. |
| 2026-09-24 | Claude Sonnet 5 (Anthropic) | Added a third sensitivity panel, `"sens_mean_raw"`: the plain ensemble mean of `sens_matrix` with no normalisation at all, in `SENS_KIND`'s native physical units (`"raw"` = `sum\|J\|`, or `"volume_normalised"` = `sum\|J\|`/block volume) — alongside the existing dimensionless, max-normalised `sens_mean_na`/`_an`. Plotted on the same `MOD_STATS` slice geometry (same template, mesh, ROI, `MOD_XLIM`/`YLIM`/`ZLIM`) as every other panel, including the model itself. Auto-scaled `MOD_STATS_CLIM` by default (its scale is run- and `SENS_KIND`-dependent). New `.npz` key: `<P>_sens_mean_raw`. Not wired into `MOD_STATS_BLANK_SOURCES` (its un-normalised scale makes a fixed threshold meaningless, same reasoning as `sens_mean_na`/`_an` before normalisation). |
| 2026-09-25 | Claude Opus 5.5 (Anthropic) | Simplification + blanking fix. **Root cause of "no blanked areas":** `fem.insert_model()` writes `10**v`, but `plot_model_slices` reads alpha files as *raw* values; the 0/1 mask therefore arrived as 1/10 and never fell below the 0.5 threshold. The alpha block is now written so its raw values equal alpha (`_write_alpha_block`) and plotted with the new `alpha_mode="direct"`. Replaced `MOD_ALPHA_FILE`/`_BLANK_THRESH`, `MOD_STATS_BLANK_SOURCES`/`_COMBINE`/`_MODE` with a single `MOD_ALPHA_SOURCE` expression (`"<name> <op> <value>"`) plus `MOD_ALPHA_MODE` (`blank`/`fade`), `MOD_ALPHA_FADE_WIDTH`, `MOD_ALPHA_QC`, all in the shared slice block. Sensitivity aggregates and SimRC sums are now log10 (memory, `.npz`, plots). Removed `COMPUTE_VAR_REDUX`/`REDUX_EPS`/`var_prior`/`var_redux` and the iter0 reads; removed `SENS_LOW_THRESH_FRAC`, `SENS_CV_HIGH_THRESH`, `SIMRC_LOW_THRESH_FRAC`, `SPREAD_LOW_THRESH_FRAC`, `COMPUTE_NULL_SPACE_FLAG`, `FLAG_SPREAD_PERCENTILE`, `NULL_SPACE_COMBINE` and the `*_low_mask`/`flag_null_space` arrays and panels. `MOD_STATS_CLIM` replaced by `MOD_STATS_STYLE` (per-panel `cmap`/`clim`/`label`) in the shared slice block. Fixed `clim=None` silently falling back to `MOD_CLIM` (auto-scaling never happened). MOD_STATS block files use sentinel air (1e30)/ocean (1e-30) resistivities so statistic values cannot be mistaken for air/ocean. Requires the 2026-09-25 `femtic_viz.py`. |
| 2026-09-25b | Claude Opus 5.5 (Anthropic) | Diagnostics: per-member `COMPUTE_SENS`/`COMPUTE_SIMRC` skip reasons (exception type and message, size or composition mismatch) are collected and summarised once after the scan loop (first 5 listed). The `MOD_ALPHA_SOURCE` "not available" message now says why: flag off, or no usable member plus the first failure reason. |
| 2026-09-25c | Claude Opus 5.5 (Anthropic) | **Fix:** sensitivity vectors from `results_iterX.h5` have one entry per resistivity block *including fixed ones* (e.g. 54588 vs 54587 free parameters), so every member was skipped and no `sens_*` statistics existed -- which also means the old `flag_null_space` blanking can never have worked on such runs. New `_sens_to_free()` maps the vector onto the free parameters using the first member's region table (`len == n_free` as is; `== nreg` -> `sens[free_idx]`; `== nreg-1` -> `sens[free_idx-1]`) and prints the chosen mapping once. Block order is verified once against `/model/blocks["blockID"]` (identity confirmed in the log; a permutation is undone; unreadable -> warning, identity assumed). |
| 2026-09-25d | Claude Opus 5.5 (Anthropic) | `COMPUTE_SIMRC` now reads the calculated responses and errors from `/data/data` of the same `results_iter<numit>.h5` (`fem.read_results_data`) instead of per-member `result_MT.txt`/`result_VTF.txt`, which current FEMTIC runs no longer write ("no per-member forward-response files found"). Removed `SIMRC_RESULT_FILES`, `SIMRC_SITE_FILE`, `SIMRC_DATA_KIND`; new `SIMRC_ERR_WEIGHT` (default `True`). Imaginary parts of real-valued datatypes (APP_RES_AND_PHS, PT, NMT2_APP_RES_AND_PHS) and columns with error <= 0 are dropped; data composition (datatype, site, frequency, component) is checked against the first member. |
| 2026-09-25e | Claude Opus 5.5 (Anthropic) | New `MOD_TITLE_FONTSIZE` (default `None` = `MOD_LABEL_FONTSIZE + 6`) for the file name shown as figure title, passed as `figure_title_fontsize` to `plot_model_slices`. |
| 2026-09-26 | Claude Sonnet 5 (Anthropic) | Model-centre marker: picked up `femtic_viz.py`'s 2026-09-26 default-style change (thin black `"+"` → yellow star with a black edge, higher `zorder`) — no functional change here, `MOD_SHOW_MODEL_CENTRE`/`MOD_MAP_MARKERS` wiring is unchanged. |
| 2026-09-26b | Claude Sonnet 5 (Anthropic) | Added a region-of-interest diagnostic for the log10 sensitivity panels: every `sens_*`/`simrc_*` entry in `MOD_STATS_WHAT` that gets plotted now prints its original (whole-mesh-normalised, as stored) min/max restricted to the free parameters inside the ROI (`MOD_XLIM`/`MOD_YLIM`/`MOD_ZLIM`, including `MOD_ROI_AUTO`) — meant to make picking a `MOD_ALPHA_SOURCE` threshold easier. New `SENS_ROI_NORMALIZE` (default `False`): optionally re-centres each `sens_*` panel (not `sens_cv_*`) to the ROI's own max instead of the whole-mesh max, display-only (the `.npz` and `MOD_ALPHA_SOURCE` are unaffected); `simrc_coef`/`simrc_corr` are diagnosed but never renormalised by this switch, per explicit request. New helper `_roi_free_mask()`, built on `fem.build_region_geometry()` and the same `free_idx` mapping as `_sens_to_free()`; degrades gracefully if mesh/region geometry can't be read. See the "Sensitivity statistics" section above for the full parameter description. |
| 2026-09-26c | Claude Sonnet 5 (Anthropic) | Simplified the panel-selection interface per user request ("set panels to plot by a list of sources and one blanking option; determine the other settings from these"): `MOD_ALPHA_SOURCE` was already the one blanking option (2026-09-25). `MOD_QC` was a second switch that had to be kept in sync with `MOD_STATS_WHAT` by hand; it is now the special `"best"` entry in `MOD_STATS_WHAT` itself (default `MOD_STATS_WHAT` gained `"best"` as its first entry, preserving the previous `MOD_QC=True` default). `MOD_QC`/`MOD_STATS` are still set, but derived (`MOD_QC = "best" in MOD_STATS_WHAT`; `MOD_STATS = bool(set(MOD_STATS_WHAT) - {"best"})`) so every downstream use of those two names is unchanged; turning either off is now just editing the one list. `MOD_NROWS`/`MOD_NCOLS` (sized to the unrelated `MOD_SLICES` list) and the `COMPUTE_*`/`BOOTSTRAP_VAR` `.npz`-computation switches were deliberately left untouched — see "Panels to plot" above for the scope reasoning. |
| 2026-10-07 | Claude Sonnet 5.5 (Anthropic) | Vertical sections are configured in `MOD_SLICES` as `dict(kind="plane", point=[x, y] km, strike=azimuth deg, dip=90)`; the default list ends with a SW-NE (strike 45) section through the model origin. New helper `_plane_spec_km_to_m()` converts such entries (km to m) in `_plot_slice()`; other kinds still go through `fem.resolve_slice_positions`, original order kept. Same limits as the other slices: along-strike range = `MOD_XLIM` x `MOD_YLIM` projected onto the strike (optional per-entry `xlim`), depth range = `MOD_ZLIM`; horizontal axis in model km. Requires updated `femtic_viz.py`. Also fixed a missing comma in the default `MOD_STATS_WHAT` that merged `"sens_median_an"` and `"sens_cv_an"` into the non-existent key `"sens_median_ansens_cv_an"`, so neither panel was plotted. |
| 2026-10-08 | Claude Sonnet 5.5 (Anthropic) | New config `CNV_FILE` (default `"femtic.cnv"`, single name or candidate list) replaces the hard-coded `femtic.cnv` in step (1)'s scan loop; uses `fem.resolve_cnv_path()`. AI-generated; review before production use. |
| 2026-10-11 | Claude Sonnet 5.5 (Anthropic) | New optional `MOD_BH` block (step 9): 1-D resistivity-depth borehole profiles from the ensemble. List of dicts like `MOD_SLICES` (`name`, `x`/`y` as model-local km or `(v,"utm")`/`(v,"latlon")`, `z_top` (km or `"surface"`), `z_bot`, `dz`, `what` = list of `"avg"`, `"med"`, `"percentiles"` / single percentile keys, `"ensemble"`, `"best"`, `alpha` = ensemble opacity, optional `xlim`, `style`). New globals `MOD_BH_FILE`, `MOD_BH_NPZ`, `MOD_BH_ALPHA`, `MOD_BH_XLIM`, `MOD_BH_PANEL_SIZE`, `MOD_BH_NCOLS`, `MOD_BH_SHARE_DEPTH`, `MOD_BH_LEGEND_LOC`, `MOD_BH_STYLE`; shares `MOD_PLOT_FORMAT`, `MOD_DPI`, `MOD_CLIM`, fonts, `MOD_DEPTH_KM`, `MOD_DISPLAY_COORDS`. Step (5) (origin / site resolution) now also runs when only `MOD_BH` is set. Element search once per borehole, reused for all statistics and members; verified against `fem.extract_borehole_log` on a synthetic mesh. New helpers `_bh_*` and `_plot_boreholes()`. Provenance: AI-generated, review before production use. |
| 2026-10-11b | Claude Sonnet 5.5 (Anthropic) | `MOD_BH`: new `sites` key (`"all"` or list of names; template expanded to one borehole per site of `MOD_SITE_DAT`), `MOD_BH_PER_FIG` (paging), `MOD_BH_NCOLS = None` now up to 4 columns, unique borehole names. Depth documented as z positive downward. Mesh reading: `fviz.read_femtic_mesh` was removed; `fem.read_femtic_mesh` is the only reader, the former x/y swap is off by default and available as `MOD_MESH_SWAP_XY` (needs the matching `femtic.py` / `femtic_viz.py` of 2026-10-11). This also makes `_roi_free_mask()` use the same axis convention as the plots. |
| 2026-10-11c | Claude Sonnet 5.5 (Anthropic) | `MOD_BH`: `n_random` / `seed` keys (random subset of the sites of a `sites` template); `"sigma1"` / `"sigma2"` in `what` (1-/2-sigma quantile pairs of the members, drawn as two lines with one legend entry); default entry now draws the ensemble at `alpha=0.05` beneath the median and the 1-sigma quantiles; new `MOD_BH_STYLE` keys `"sigma1"`, `"sigma2"`. |
| 2026-10-27 | Claude Sonnet 5.5 (Anthropic) | Optional reference model: `MOD_REF_FILE` (block file on the `MOD_MESH` mesh; `None` = off), `MOD_REF_SLICES`, `MOD_REF_BH`, `MOD_REF_LEVELS`, `MOD_REF_STYLE`, `MOD_REF_LABEL`. Slices: iso-resistivity lines of the reference on the best panel and every `MOD_STATS` panel (via `femtic_viz.plot_model_slices(reference_*)`; levels always in the reference's own log10 resistivity). Boreholes: dashed `"ref"` line (`MOD_BH_STYLE["ref"]`), added to every `MOD_BH` panel unless the entry has `ref=False`, or listed in `what`; saved as `ref_<name>` in the `.npz`, `reference_file` in its header. A missing or wrong-mesh reference is reported and switched off, the run continues. New `MOD_VERT_EXAG` (default `None` = figures unchanged; checked pixel-identical to the previous version): vertical exaggeration of the ns / ew / plane panels from real axis units, also for `MOD_DISPLAY_COORDS="latlon"`. Needs the 2026-10-27 `femtic_viz.py`. AI-generated; review before production use. |
| 2026-10-27b | Claude Sonnet 5.5 (Anthropic) | **Plot-only mode.** New `PLOT_ONLY` (default `False`), `PLOT_ONLY_NPZ`, `PLOT_ONLY_TEMPLATE`: with `PLOT_ONLY=True` steps (1)-(4b) are replaced by reading the existing results `.npz` and steps (5)-(9) (best panel, `MOD_STATS_WHAT` panels, `MOD_ALPHA_SOURCE`, boreholes, nRMS histogram from `NRMS_CSV`) run unchanged on the rebuilt variables; nothing is scanned or recomputed and the `.npz` is never rewritten. Percentile levels come from the file; missing `avg ... mad / prc` are recomputed from the stored members. Template = best member's block file from the `.npz`, or `PLOT_ONLY_TEMPLATE` (then the best panel is drawn from the stored best vector). Default `MOD_STATS_WHAT` / `MOD_STATS_STYLE` list the sens / simrc / bootstrap entries regardless of the `COMPUTE_*` flags in this mode. The scan / statistics block is indented under `else:` of `if PLOT_ONLY:`; the full-run path is otherwise unchanged (figures, `.npz` and borehole `.npz` identical to the previous version on a synthetic ensemble; plot-only figures pixel-identical to the full run that wrote the file). `femtic_ens_plot.py` gained the matching `RESULTS_NPZ` mode. AI-generated; review before production use. |
| 2026-10-27c | Claude Sonnet 5.5 (Anthropic) | New `ITER_FALLBACK` (default `True`): when the best-nRMS iteration N has no `resistivity_block_iter<N>.dat`, the member is no longer dropped as `no_model_file` but represented by the best iteration that has a block file (usually N-1), with `NRMS_MAX` checked against that iteration's nRMS. New `NRMS_CSV` status `accepted_fallback` (counted as accepted in the histogram, whose legend shows how many members use a fallback iteration). `ITER_FALLBACK=False` = previous behaviour; with nothing missing the results are identical to the previous version. AI-generated; review before production use. |
| 2026-10-28 | Claude Sonnet 5.5 (Anthropic) | The `MOD_ALPHA_SOURCE` mask is no longer applied to the sensitivity panels themselves (`sens_*`, `simrc_*`): shading a sensitivity map by sensitivity is circular. New `MOD_ALPHA_EXCLUDE` (key prefixes, default `("sens_", "simrc_")`; `()` restores masking of every panel). Other panels and the best panel (`MOD_ALPHA_QC`) are unchanged. AI-generated; review before production use. |
| 2026-10-29 | Claude Sonnet 5.5 (Anthropic) | `MOD_BH`: new `"heatmap"` item for `what` (density heat map of the ensemble along the borehole: per-depth histogram of the members' log10 rho, `pcolormesh` behind the other items, one shared colour bar per figure). New globals `MOD_BH_HEAT_BINS`, `MOD_BH_HEAT_NORM` (`"fraction"` / `"row_max"`), colour scale `MOD_BH_HEAT_CMAP` (default `Greys`), `MOD_BH_HEAT_SCALE` (`"linear"` / `"sqrt"` / `"log"`), `MOD_BH_HEAT_VMIN`, `MOD_BH_HEAT_VMAX`, `MOD_BH_HEAT_CBAR`, `MOD_BH_STYLE["heatmap"]` (pcolormesh keywords), per-entry `heat_bins`. Members outside the bin range are not drawn but stay in the denominator; auto colour maximum = 99th percentile of the non-zero fractions (linear / sqrt). `.npz`: `heat_<name>`, `heat_edges_<name>`; header: `heat_unit`, `heat_bins`. Nothing changes unless `"heatmap"` is listed in an entry: with the default `MOD_BH` the borehole figure is pixel-identical and the `.npz` arrays identical to the previous version (tested on a synthetic mesh); heat maps checked against an independent histogram. AI-generated; review before production use. |
| 2026-10-31 | Claude Sonnet 5.5 (Anthropic) | New `ENS_MODEL_SOURCE` (`"converged"` by default, or `"perturbed"`), `ENS_PERTURBED_FILE` (default `"referencemodel.dat"`), `ENS_PERTURBED_SUFFIX` (default `"_perturbed"`): the ensemble can be built from each member's perturbed reference (the RTO prior draw) instead of its converged model, so all statistics, slices, boreholes and heat maps describe the prior ensemble. Same member set as the converged run (selection unchanged); `MOD_REF_FILE` (base reference) unchanged; `ENSEMBLE_PREFIX` gets the suffix so results do not overwrite; `<P>_model_source` / `<P>_model_file_name` in the `.npz`, `ensemble_source` in the borehole header; `PLOT_ONLY` reads the stored source (older files: converged); `COMPUTE_SIMRC` off in this mode; warnings for missing files (`no_perturbed_file`), symbolic links and zero spread. Default behaviour unchanged (scan result identical to the previous version on a synthetic ensemble). AI-generated; review before production use. |
| 2026-11-02 | Claude Sonnet 5.5 (Anthropic) | Borehole heat maps of the converged **and** the perturbed models on a common colour bar, same options as `femtic_ens_plot.py`. `ENS_MODEL_SOURCE = "both"`: the run is the converged run and each member's `ENS_PERTURBED_FILE` is loaded in addition (a member lacking it is skipped, `no_perturbed_file`); `MOD_BH` draws every borehole for both ensembles (heat map, `"ensemble"`, statistics, `"best"`), side by side in one figure with one colour bar (`MOD_BH_HEAT_FIGURE = "combined"`) or as two figures (`"separate"`); the perturbed profiles go to `<MOD_BH_FILE>_perturbed.npz`, the perturbed matrix is stored as `<P>_ens_perturbed` so `PLOT_ONLY` can draw it. New `MOD_BH_HEAT_SHARE_SCALE` (default `False`: one scale per figure as before; `True`: one scale for all figures), `MOD_BH_HEAT_SCALE_FILE` / `MOD_BH_HEAT_SCALE_IO` (save / reuse the resolved scale as JSON across runs; same format as `femtic_ens_plot.py`). The borehole `.npz` header gains `colour_scale`. With the defaults, figures and `.npz` arrays are identical to the previous version. No change needed for the 2-element plane `point` (`_plane_spec_km_to_m` pads it). AI-generated; review before production use. |
| 2026-11-06 | Claude Sonnet 5.5 (Anthropic) | **Directory scan report** appended to `femtic_ens_post_summary.md` after a full run: every scanned directory that is not in the ensemble, with a reason (`no_cnv_file`, `cnv_unreadable`, `cnv_no_iterations`, `rejected_nrms`, `no_model_file`, `no_perturbed_file`) and detail, counts per reason, members accepted via `ITER_FALLBACK`, and ignored non-directory glob matches. Uses the path returned by `utl.write_param_summary()`; falls back to `femtic_ens_post_scan_report.md` if the summary cannot be appended to. New helpers `build_scan_report()` / `append_scan_report()` (same code as in `femtic_ens_repair.py`); no change to member selection, `NRMS_CSV`, the `.npz` or any figure. Tested by executing the edited scan loop and the helpers on a synthetic ensemble (all six reasons, fallback, non-directory matches); the full script was not run. AI-generated; review before production use. |
| 2026-11-06b | Claude Sonnet 5.5 (Anthropic) | Scan report: the number of glob matches is always shown next to the directories scanned (they differ for a file or broken symbolic link), and a new **"Directory numbering"** part reads the number after `ENSEMBLE_NAME` in each name and lists numbers with no entry at all, numbers matched only by a non-directory, numbers matched by several names and unnumbered names (prompted by a report of 255 directories scanned for an ensemble numbered 0..255). New optional `ENSEMBLE_INDEX_RANGE = (first, last)`, default `None` = inferred from the names. No change to member selection or any output other than the summary. Tested by executing the edited scan loop and helpers on a synthetic ensemble; the full script was not run. AI-generated; review before production use. |

