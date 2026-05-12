# modem_compress.py

Apply spectral / basis-function compression to a ModEM resistivity model,
reconstruct, report accuracy, and optionally write the result.

## Provenance

| Field | Value |
|-------|-------|
| Script | `modem_compress.py` |
| Author | vrath |
| Part of | **py4mt** — Python for Magnetotellurics |
| Inversion code | ModEM |
| Written | April 2026 |

## Purpose

Reads a ModEM `.rho` model, compresses it in a chosen spectral or
basis-function representation, reconstructs the model, reports
reconstruction errors (RMS, relative RMS, max), and writes the result
to a new `.rho` file.

Six compression methods are implemented.  One is selected at runtime by
setting `METHOD`.  All six parameter blocks are present in the
configuration section; only the one matching `METHOD` is used.

An optional truncation sweep (`RUN_ANALYSIS = True`) runs the
corresponding analysis function before compression, printing a table of
reconstruction accuracy versus compression ratio.  This is the recommended
first step when using a new model or a new method.

## Workflow

```
read model (.rho)
    │
    ├─ [optional] truncation analysis → accuracy vs ratio table
    │
    ├─ compress  (forward transform + truncation)
    │
    ├─ reconstruct (inverse transform)
    │
    ├─ report errors (RMS, rel. RMS, max)
    │
    └─ [optional] write reconstructed model (_compressed_<METHOD>.rho)
```

## Methods

| `METHOD` | Basis | Truncation | Best suited for |
|---|---|---|---|
| `'dct'` | 3-D DCT-II | Radial (L2 wavenumber sphere) | Smooth, spatially stationary models |
| `'dct_sep'` | 3-D DCT-II | Separable box per axis | Anisotropic compression needs |
| `'wavelet'` | 3-D DWT | Hard threshold by count, fraction, or amplitude | Spatially localised anomalies |
| `'legdct'` | Legendre-z × DCT-xy | Independent per axis | Models with strong depth gradients |
| `'bspdct'` | B-spline-z × DCT-xy | Independent per axis | Real models with expanding cell size |
| `'kl'` | Karhunen–Loève / PCA | Mode count or fraction | Ensemble-based uncertainty analysis |

## Global configuration

| Constant | Default | Description |
|---|---|---|
| `RHOAIR` | `1e17` | Air resistivity (Ω·m); used for air-cell masking |
| `MOD_FILE_IN` | `…/TAC_100` | Input model (without extension) |
| `MOD_FILE_OUT` | `MOD_FILE_IN + "_compressed"` | Output base name (method suffix appended) |
| `METHOD` | `"dct"` | Active compression method |
| `WRITE_OUTPUT` | `True` | Write reconstructed model to `.rho` file |
| `WRITE_NPZ` | `True` | Write compressed coefficients + metadata to `.npz` archive |
| `RUN_ANALYSIS` | `True` | Run truncation sweep before compressing |

Output file name: `MOD_FILE_OUT + "_" + METHOD + ".rho"`, e.g.
`TAC_100_compressed_bspdct.rho`.  Running all six methods therefore
produces six distinguishable files.

## Method parameters

All parameters live in named dicts at the top of the configuration
section.  Only the dict matching the active `METHOD` is read; all
others are ignored.  Switching methods requires changing only the
`METHOD` string.

### `DCT` — radial DCT-II truncation

Keeps the *n* lowest-wavenumber 3-D DCT-II coefficients, ranked by
L2 wavenumber k = √(kx² + ky² + kz²).

| Key | Default | Description |
|---|---|---|
| `frac_keep` | `0.05` | Fraction of total coefficients to retain |
| `n_keep` | `None` | Explicit count (overrides `frac_keep` if set) |
| `kmax` | `None` | Maximum L2 wavenumber cutoff (alternative) |
| `n_levels` | `20` | Levels for the truncation analysis sweep |

Specify exactly one of `frac_keep`, `n_keep`, or `kmax`.

### `DCT_SEP` — separable (box) DCT-II truncation

Retains a rectangular sub-block of the 3-D DCT coefficient array,
with independent compression ratios for each spatial axis.  No
truncation analysis sweep is available for this variant.

| Key | Default | Description |
|---|---|---|
| `frac_x` | `0.4` | Fraction of nx coefficients to retain along x |
| `frac_y` | `0.4` | Fraction of ny coefficients to retain along y |
| `frac_z` | `0.4` | Fraction of nz coefficients to retain along z |
| `nx_keep` | `None` | Explicit count along x (overrides `frac_x`) |
| `ny_keep` | `None` | Explicit count along y (overrides `frac_y`) |
| `nz_keep` | `None` | Explicit count along z (overrides `frac_z`) |

Fractions are converted to integer counts at runtime using the actual
model dimensions, so the configuration is dimension-agnostic.

### `WAVELET` — 3-D DWT with hard thresholding

Requires `PyWavelets` (`pip install PyWavelets`).  Specify exactly one
of `frac_keep`, `n_keep`, or `thresh`.

| Key | Default | Description |
|---|---|---|
| `wavelet` | `"db4"` | Wavelet family: `'db4'`, `'sym4'`, `'coif2'`, … |
| `level` | `None` | Decomposition depth (`None` = maximum for array size) |
| `frac_keep` | `0.05` | Fraction of largest-magnitude coefficients to keep |
| `n_keep` | `None` | Explicit count |
| `thresh` | `None` | Hard amplitude threshold (keep all \|c\| ≥ thresh) |
| `n_levels` | `20` | Levels for the truncation analysis sweep |

Recommended wavelets for smooth geophysical models: `'db4'` (default),
`'sym4'`, `'coif2'`.  Shorter wavelets (`'db2'`, `'haar'`) are faster
but produce more ringing artefacts near sharp boundaries.

### `LEGDCT` — Legendre-z × DCT-xy

Applies a 1-D Legendre polynomial expansion along the depth axis and
a 2-D DCT-II along the two horizontal axes.  Global support along z
(each Legendre order influences the full depth column).

| Key | Default | Description |
|---|---|---|
| `frac_leg` | `0.4` | Fraction of nz cells used as Legendre orders |
| `n_leg` | `None` | Explicit Legendre order count (overrides `frac_leg`) |
| `frac_dct` | `0.4` | Fraction of nx / ny cells used as DCT coefficients |
| `nx_dct` | `None` | Explicit count along x (overrides `frac_dct`) |
| `ny_dct` | `None` | Explicit count along y (overrides `frac_dct`) |
| `n_levels` | `20` | Levels for the truncation analysis sweep |

The truncation sweep couples `frac_leg` and `frac_dct` via
`frac_leg = frac_dct = √(frac_keep)` so that a single parameter
controls the overall compression ratio.

### `BSPDCT` — B-spline-z × DCT-xy

Applies a 1-D B-spline basis expansion along the depth axis and a
2-D DCT-II along the horizontal axes.  Locally supported along z:
each basis function spans only a few adjacent cells.  Knots are placed
adaptively using `knot_style`, which can concentrate basis functions
near the surface where cells are finest and MT sensitivity is highest.

| Key | Default | Description |
|---|---|---|
| `frac_basis` | `0.4` | Fraction of nz cells used as B-spline basis functions |
| `n_basis` | `None` | Explicit basis count (overrides `frac_basis`) |
| `k` | `3` | Spline degree (3 = cubic) |
| `knot_style` | `"quantile"` | Knot placement: `'quantile'`, `'uniform'`, or `'log'` |
| `frac_dct` | `0.4` | Fraction of nx / ny cells used as DCT coefficients |
| `nx_dct` | `None` | Explicit count along x |
| `ny_dct` | `None` | Explicit count along y |
| `n_levels` | `20` | Levels for the truncation analysis sweep |

`knot_style` options:

| Value | Knot positions | When to use |
|---|---|---|
| `'quantile'` | Depth quantiles of cell centres | Real models with expanding cells — **default** |
| `'uniform'` | Equally spaced in normalised depth | Models with uniform cell sizes |
| `'log'` | Logarithmically spaced | Depth axis spanning many decades |

`n_basis` must satisfy `n_basis ≥ k + 1` (minimum: 4 for cubic
splines).  The number of free interior knots is `n_basis − k − 1`;
setting `n_basis = k + 1` gives a single polynomial with no interior
knots.

### `KL` — Karhunen–Loève / PCA

Builds the KL basis from an ensemble of existing models by SVD of the
centred ensemble matrix, then projects the target model onto the
leading eigenmodes.  The basis is theoretically optimal for that
ensemble: it minimises mean-square reconstruction error for any fixed
number of modes.

| Key | Default | Description |
|---|---|---|
| `ensemble_files` | (3 TAC files) | List of `.rho` file paths (without extension) to form the ensemble |
| `n_modes` | `20` | Number of KL modes to retain (`None` = all available) |
| `frac_modes` | `None` | Fraction of available modes (alternative to `n_modes`) |
| `svd_method` | `"auto"` | SVD algorithm: `'auto'`, `'exact'`, `'randomized'`, `'truncated'` |
| `n_oversamples` | `10` | Extra random projections for `'randomized'` SVD |
| `n_power_iter` | `4` | Power iterations for `'randomized'` SVD (more → better on flat spectra) |
| `random_state` | `None` | Random seed for reproducibility of `'randomized'` SVD |
| `n_levels` | `20` | Levels for the KL truncation analysis sweep |

`svd_method` selection guide:

| Value | Algorithm | When to use |
|---|---|---|
| `'auto'` | Selects below automatically | Safe default |
| `'exact'` | Full economy SVD (`numpy.linalg.svd`) | All modes needed, or `n_modes ≈ n_models` |
| `'randomized'` | Halko et al. 2011 via sklearn | `n_modes ≪ n_models` — strongly preferred for large ensembles |
| `'truncated'` | ARPACK (`scipy.sparse.linalg.svds`) | No sklearn; sparse matrices |

`'auto'` selects `'randomized'` when `n_modes < 0.5 × min(n_models, n_cells)`,
otherwise `'exact'`.  If sklearn is absent the randomized path silently
falls back to `'truncated'`.

The KL truncation analysis prints reconstruction accuracy versus number
of modes, alongside the cumulative explained variance from the ensemble
SVD — this is the primary tool for choosing `n_modes`.

## Reconstruction errors

All methods report the same three scalar errors after reconstruction:

| Metric | Formula | Interpretation |
|---|---|---|
| RMS | √(mean((m_rec − m)²)) | Absolute error in log-resistivity |
| Rel. RMS | RMS / √(mean(m²)) | Fractional error relative to model amplitude |
| Max | max(\|m_rec − m\|) | Worst-case error (log-resistivity) |

Errors are computed on the full model array including padding cells.
Air cells are excluded from compression but re-masked before writing,
so they do not contribute to the error metrics.

## Output

### Reconstructed model (`.rho`)

If `WRITE_OUTPUT = True`, the reconstructed model is written to

```
{MOD_FILE_OUT}_{METHOD}.rho
```

in `LOGE` (natural-log resistivity) format, matching the storage
convention of the input file.  Air cells are reset to `log(RHOAIR)`
before writing.

### Compressed archive (`.npz`)

If `WRITE_NPZ = True`, a NumPy compressed archive is written to

```
{MOD_FILE_OUT}_{METHOD}.npz
```

The archive contains everything needed to reconstruct the model from
scratch — the compressed coefficients, the mesh, the reference point,
and reconstruction error metrics — without re-running the compression.

Load with:

```python
import numpy as np
data = np.load("TAC_100_compressed_bspdct.npz", allow_pickle=True)
```

#### Arrays present in every archive

| Key | Shape / type | Description |
|---|---|---|
| `dx` | (nx,) | Cell sizes along x (m) |
| `dy` | (ny,) | Cell sizes along y (m) |
| `dz` | (nz,) | Cell sizes along z (m) |
| `reference` | (3,) | Model reference point [x, y, z] (m) |
| `rms_err` | scalar | RMS reconstruction error (log-resistivity) |
| `rel_err` | scalar | Relative RMS error |
| `max_err` | scalar | Maximum absolute error (log-resistivity) |
| `method` | str | Compression method used (e.g. `'bspdct'`) |
| `trans` | str | Model transform stored (`'LOGE'`) |
| `mod_file_in` | str | Source model path |

#### Method-specific coefficient arrays

**`dct`**

| Key | Shape | Description |
|---|---|---|
| `coeff` | (n_keep,) | Retained DCT-II coefficients, flat |
| `keep_mask` | (nx, ny, nz) bool | True where a coefficient is retained |
| `shape` | (3,) int | Original model shape |

Reconstruct: `mod.dct_to_model(coeff, tuple(shape), keep_mask)`

**`dct_sep`**

| Key | Shape | Description |
|---|---|---|
| `coeff_block` | (nx_keep, ny_keep, nz_keep) | Low-wavenumber corner of the DCT array |
| `shape_full` | (3,) int | Original model shape |
| `shape_keep` | (3,) int | Shape of `coeff_block` |

Reconstruct: `mod.dct_separable_to_model(coeff_block, tuple(shape_full))`

**`wavelet`**

| Key | Shape | Description |
|---|---|---|
| `wavelet` | str | Wavelet name (e.g. `'db4'`) |
| `wb_keys` | str | Comma-separated subband labels (e.g. `'aaa,aad,ada,…'`) |
| `wb_<key>` | varies | One array per subband; key matches entry in `wb_keys` |

Reconstruct:

```python
wb_keys = str(data['wb_keys']).split(',')
coeffs  = {k: data['wb_' + k] for k in wb_keys}
rho_rec = mod.wavelet_to_model(coeffs, wavelet=str(data['wavelet']),
                               shape=tuple(rho.shape))
```

**`legdct`**

| Key | Shape | Description |
|---|---|---|
| `C` | (nx_dct, ny_dct, n_leg) | Compressed coefficient array |
| `n_leg` | scalar int | Legendre orders used |
| `nx_dct` | scalar int | DCT coefficients retained along x |
| `ny_dct` | scalar int | DCT coefficients retained along y |
| `shape` | (3,) int | Original model shape |

Reconstruct:

```python
params  = dict(n_leg=int(data['n_leg']), nx_dct=int(data['nx_dct']),
               ny_dct=int(data['ny_dct']))
rho_rec = mod.legdct_to_model(data['C'], tuple(data['shape']), params)
```

**`bspdct`**

| Key | Shape | Description |
|---|---|---|
| `C` | (nx_dct, ny_dct, n_basis) | Compressed coefficient array |
| `n_basis` | scalar int | B-spline basis functions used |
| `k` | scalar int | Spline degree |
| `knot_style` | str | Knot placement used (`'quantile'`, `'uniform'`, or `'log'`) |
| `nx_dct` | scalar int | DCT coefficients retained along x |
| `ny_dct` | scalar int | DCT coefficients retained along y |
| `B` | (nz, n_basis) | B-spline collocation matrix |
| `Bpinv` | (n_basis, nz) | Pre-computed pseudo-inverse of B |
| `shape` | (3,) int | Original model shape |

Reconstruct:

```python
params  = dict(n_basis=int(data['n_basis']), k=int(data['k']),
               knot_style=str(data['knot_style']),
               nx_dct=int(data['nx_dct']), ny_dct=int(data['ny_dct']),
               B=data['B'], Bpinv=data['Bpinv'])
rho_rec = mod.bspdct_to_model(data['C'], tuple(data['shape']), params)
```

**`kl`**

| Key | Shape | Description |
|---|---|---|
| `alpha` | (n_modes,) | KL expansion coefficients (scores) for the target model |
| `modes` | (n_modes, n_cells) | KL eigenmodes (right singular vectors) |
| `singular_values` | (n_modes,) | Singular values of the centred ensemble matrix |
| `mean_model` | (n_cells,) | Ensemble mean (flat) |
| `shape` | (3,) int | Original model shape |

Reconstruct:

```python
rho_rec = mod.kl_to_model(data['alpha'], data['modes'],
                           data['mean_model'], shape=tuple(data['shape']))
```

Note: `modes` and `mean_model` are stored flat (n_cells = nx × ny × nz).
The `alpha` vector alone is sufficient if `modes` and `mean_model` are
already available from a previous run; saving `alpha` only (and
re-loading the basis separately) can substantially reduce archive size
for large models with many modes.

## Dependencies

| Package | Required by | Install |
|---|---|---|
| `numpy` | All methods | — |
| `scipy` | All methods (`scipy.fft`, `scipy.special`, `scipy.interpolate`) | — |
| `PyWavelets` | `'wavelet'` only | `pip install PyWavelets` |
| `sklearn` | `'kl'` with `svd_method='randomized'` | `pip install scikit-learn` |
| py4mt: `modem`, `util`, `version` | All | — |

`PyWavelets` and `sklearn` are optional: the script exits with a clear
error message if `'wavelet'` is selected without PyWavelets, and the
KL randomized SVD falls back automatically to ARPACK if sklearn is
absent.

## Usage notes

Set `RUN_ANALYSIS = True` on the first run for any new model or new
method.  The truncation analysis prints a table of RMS error versus
compression ratio across 20 logarithmically spaced levels, which is
the primary tool for choosing compression parameters.  Once parameters
are decided, set `RUN_ANALYSIS = False` to skip the sweep on subsequent
runs.

For the `'kl'` method, `ensemble_files` should point to models that
represent the range of structural variability relevant to your
problem — for example, several inversion iterations, results from
different starting models, or models from a checkerboard / null-space
test suite such as `modem_insert_multi.py`.  The more diverse and
representative the ensemble, the more informative the KL basis.

The `'bspdct'` method with `knot_style='quantile'` is generally
recommended over `'legdct'` for real ModEM models because B-splines
are locally supported (a change in a deep control point does not
affect the near-surface fit) and the quantile knot placement
automatically adapts to the cell-size progression of the mesh.
