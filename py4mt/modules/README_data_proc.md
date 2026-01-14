# data_proc.py — EDI / MT data processing utilities

This module contains utilities to read/write EDI-style MT data, compute derived quantities
(e.g., phase tensor and invariants), and convert an EDI dictionary into a tidy `pandas.DataFrame`.

## New / updated in this version

### Apparent resistivity + phase with optional uncertainties

#### `compute_rhophas(freq, Z, Z_err=None, *, err_kind="var", err_method="analytic", nsim=200, ...)`

Computes apparent resistivity and phase from the complex impedance tensor:

- `rho_a = |Z|^2 / (mu0 * omega)` with `omega = 2π f`
- `phi = angle(Z)` in degrees

Optional uncertainty propagation:

- `err_method="none"`: no errors returned
- `err_method="analytic"`: fast first-order (delta-method) propagation
- `err_method="bootstrap"`: parametric bootstrap / Monte-Carlo perturbation of `Z`
- `err_method="both"`: returns both analytic and bootstrap error estimates

`err_kind` controls how `Z_err` is interpreted (`"var"` or `"std"`) and how returned errors are reported.

Example:

```python
rho, phi, rho_err, phi_err = compute_rhophas(
    edi["freq"], edi["Z"], edi.get("Z_err"),
    err_kind=edi.get("err_kind", "var"),
    err_method="bootstrap",
    nsim=500,
)
```

### Error-method switch for compute procedures

The following compute procedures now support a consistent error-method switch:

- `compute_pt(...)`
- `compute_zdet(...)`
- `compute_zssq(...)`
- `compute_rhophas(...)`

Common keyword pattern:

- `err_method={"none","analytic","bootstrap","both"}`
- `err_kind={"var","std"}`
- `nsim` for bootstrap
- `fd_eps` for analytic finite-difference Jacobians (where applicable)

## Notes

- Bootstrap here is implemented as **parametric Monte-Carlo**: `Z` is perturbed using the provided `Z_err`
  (interpreted as variance or standard deviation), assuming independent Gaussian perturbations of the complex entries.
- Analytic propagation is a first-order approximation and is substantially faster; bootstrap is more robust but
  can be slower depending on `nsim`.

## Bootstrap / Monte-Carlo uncertainty references (BibTeX)

Copy/paste as needed:

```bibtex
@article{Efron1979Bootstrap,
  title   = {Bootstrap Methods: Another Look at the Jackknife},
  author  = {Efron, Bradley},
  journal = {The Annals of Statistics},
  year    = {1979},
  volume  = {7},
  number  = {1},
  pages   = {1--26},
  doi     = {10.1214/aos/1176344552}
}

@article{EiselEgbert2001Stability,
  title   = {On the stability of magnetotelluric transfer function estimates and the reliability of their variances},
  author  = {Eisel, Markus and Egbert, Gary D.},
  journal = {Geophysical Journal International},
  year    = {2001},
  volume  = {144},
  number  = {1},
  pages   = {65--82},
  doi     = {10.1046/j.1365-246x.2001.00292.x}
}

@article{NeukirchGarcia2014Nonstationary,
  title   = {Nonstationary magnetotelluric data processing with instantaneous parameter},
  author  = {Neukirch, M. and Garc{\'i}a, X.},
  journal = {Journal of Geophysical Research: Solid Earth},
  year    = {2014},
  volume  = {119},
  number  = {3},
  pages   = {1634--1654},
  doi     = {10.1002/2013JB010494}
}

@article{Chen2012EMDMarineMT,
  title   = {Using empirical mode decomposition to process marine magnetotelluric data},
  author  = {Chen, J. and others},
  journal = {Geophysical Journal International},
  year    = {2012},
  volume  = {190},
  number  = {1},
  pages   = {293--309},
  doi     = {10.1111/j.1365-246X.2012.05536.x}
}

@article{UsuiEtAl2024RRMS,
  title   = {New robust remote reference estimator using robust multivariate linear regression},
  author  = {Usui, Yoshiya and Uyeshima, Makoto and Sakanaka, Shin'ya and Hashimoto, Tasuku and Ichiki, Masahiro and Kaida, Toshiki and Yamaya, Yusuke and Ogawa, Yasuo and Masuda, Masataka and Akiyama, Takahiro},
  journal = {Geophysical Journal International},
  year    = {2024},
  volume  = {238},
  number  = {2},
  pages   = {943--959},
  doi     = {10.1093/gji/ggae199}
}

@article{UsuiEtAl2025FRB_MT,
  title   = {Application of the fast and robust bootstrap method to the uncertainty analysis of the magnetotelluric transfer function},
  author  = {Usui, Yoshiya and Uyeshima, Makoto and Sakanaka, Shin'ya and Hashimoto, Tasuku and Ichiki, Masahiro and Kaida, Toshiki and Yamaya, Yusuke and Ogawa, Yasuo and Masuda, Masataka and Akiyama, Takahiro},
  journal = {Geophysical Journal International},
  year    = {2025},
  volume  = {242},
  number  = {1},
  doi     = {10.1093/gji/ggaf162}
}

@article{SalibianBarrera2008FRB,
  title   = {Fast and robust bootstrap},
  author  = {Salibi{\'a}n-Barrera, Mat{\'i}as and Van Aelst, Stefan and Willems, Gert},
  journal = {Statistical Methods \& Applications},
  year    = {2008},
  volume  = {17},
  pages   = {41--71},
  doi     = {10.1007/s10260-007-0048-6}
}
```
