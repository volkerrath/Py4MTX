# misfit_measures.py

Measures of data fit of error-normalized residuals, and the choice of the
measures that the misfit figures show. Used by `femtic_misfit.py` (the engine of
`femtic_mod_misfit.py` / `femtic_ens_misfit.py`) and by
`scripts/modem/modem_data_misfit.py`. All functions are stateless.

## Measures

r = error-normalized residuals (non-finite values are ignored); the last column
is the value expected for r ~ N(0,1).

| Name | Label | Definition | Expected |
|---|---|---|---|
| `nrms` | nRMS | sqrt(mean(r^2)) | 1 |
| `r_mae` | rMAE | sqrt(pi/2) * mean(abs(r)) | 1 |
| `r_med` | rMED | median(abs(r)) / 0.67449 | 1 |
| `q95` | q95 | 95th percentile of abs(r) | 1.96 |
| `mean` | mean | mean(r), the bias | 0 |
| `std` | std | sqrt(m2) | 1 |
| `skew` | skew | m3 / m2^1.5 | 0 |
| `kurt` | kurt | m4 / m2^2 - 3 (excess kurtosis) | 0 |

with m_k = mean((r - mean(r))^k). `nrms`, `r_mae`, `r_med` and `q95` are
computed by `inverse.py`. The moments are the plain 1/n estimators; `skew` needs
n >= 3, `kurt` n >= 4 and a sample that is not constant (else NaN).
`SIGNED_MEASURES` = (`mean`, `skew`, `kurt`) can be negative: the scripts judge
them by the absolute value (worst / best) and use a diverging colour scale.

Aliases accepted in the settings (`MEASURE_ALIASES`): `rms` = `nrms`, `rmae` =
`r_mae`, `rmed` = `r_med`, `bias` = `mean`, `intercept` = `icpt`.

For the use of normalized misfits in three-dimensional magnetotelluric modelling
see Baba (2023), cited below; the histogram figures follow that paper.

## Functions

| Function | Purpose |
|---|---|
| `fit_measures(r)` | dict with `n` and the eight measures |
| `sample_moments(r)` | dict with `std`, `skew`, `kurt` |
| `qq_quartile_line(sample_sorted)` | `(slope, intercept)` of the quartile line of a normal Q-Q plot (needs scipy) |
| `site_measures(spec, *, setting="SITE_MEASURES")` | the setting `SITE_MEASURES` as a tuple of names; `None` = `("nrms",)` |
| `box_items(spec, kind, *, setting="PANEL_MEASURES")` | the entries of the text box of a `"crossplot"`, `"histogram"` or `"qq"` panel |
| `box_text(items, kind, *, n, r1, r2, has_syn, limit=None, out_pct=nan)` | the text of such a box |
| `hist_scale(spec, *, setting="HIST_SCALE_MEASURE")` | key of `HIST_SCALES`, or `None` |
| `site_value(d, measure="nrms")` | one value of a per-site table entry (NaN if absent) |

Constants: `MEASURE_LABELS`, `FIT_MEASURE_NAMES`, `SIGNED_MEASURES`,
`MEASURE_ALIASES`, `BOX_KINDS`, `BOX_ITEMS`, `BOX_DEFAULT`, `HIST_SCALES`.

Panel boxes:

- entries: `n`, the measures, `slope`, `icpt`, `n_out` (histogram, Q-Q), `out_pct`
  (Q-Q);
- defaults (`BOX_DEFAULT`): crossplot `n`, `nrms`; histogram `n`, `nrms`, `n_out`;
  Q-Q `n`, `slope`, `icpt`;
- `spec`: `None` = default, `False` / `""` / `()` = nothing, a name, `"a, b"`, a
  list, or a dict `{kind: list, "default": list}`.

Histogram scale (`HIST_SCALES`): `nrms`, `r_mae`, `r_med` are used as they are as
the standard deviation of the Gaussian curve; `q95` is divided by
1.959963984540054.

The settings that use these functions are described in
`femtic_misfit_readme.md` ("Choosing the plotted measures").

## Dependencies

numpy, scipy (for the Q-Q line), `inverse.py` (which imports `aniso.py`).

## Testing

`scripts/general/test_femtic_misfit_measures.py` and
`scripts/modem/test_modem_data_misfit_measures.py`, on synthetic data.

## Reference

Baba, K. (2023): A simple method to evaluate the uncertainty of magnetotelluric
forward modeling for practical three-dimensional conductivity structure models.
Earth, Planets and Space 75, 67. https://doi.org/10.1186/s40623-023-01832-5

## Provenance

| Date | Author | Change |
|---|---|---|
| 2026-11-27 | Claude Sonnet 5.5 (Anthropic) | New module. `sample_moments`, `fit_measures` and `qq_quartile_line` moved here from `femtic_misfit.py` without a change of their results; new: the choice of the plotted measures. AI-generated; review before production use. |

Author of py4mt: Volker Rath (DIAS).
