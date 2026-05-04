# plotrjmcmc.py — Legacy rjMCMC plotting (Geoscience Australia)

`plotrjmcmc.py` is a Python translation of Ross Brodie's original MATLAB
plotting routines for rjMCMC (reversible-jump MCMC) 1-D inversion results,
developed at Geoscience Australia.  It was adapted for the py4mt codebase by
Volker Rath (2019).

> **Status: legacy.** For new work, prefer `transdim_viz.py`, which provides
> a cleaner, axes-based API and integrates directly with the `transdim.py`
> sampler.  `plotrjmcmc.py` is retained for reading results produced by older
> GA-format rjMCMC output files.

---

## Provenance

| Field | Value |
|-------|-------|
| Original author | Rakib Hassan (Geoscience Australia) — 2017/10/17 |
| Python port | rakib.hassan@ga.gov.au — 2018 |
| py4mt adaptation | Volker Rath (DIAS) — 2019 (colorbar, sizes) |

---

## Overview

The module defines a `Results` class that reads a GA-format rjMCMC output
directory and generates multi-panel diagnostic figures.

### `Results` class

```python
Results(
    path,
    outputFileName,
    plotSizeInches='11x8',
    maxDepth=2000,
    resLims=[1., 100000.],
    zLog=False,
    colormap='gray_r',
)
```

| Parameter | Description |
|-----------|-------------|
| `path` | Path to the rjMCMC station output directory |
| `outputFileName` | Base filename for saved figures |
| `plotSizeInches` | Figure size as `"WxH"` string (default `"11x8"`) |
| `maxDepth` | Maximum depth for profile plots [m] (default `2000`) |
| `resLims` | Resistivity axis limits `[min, max]` (default `[1, 100000]`) |
| `zLog` | If `True`, use log scale on the depth axis |
| `colormap` | Matplotlib colormap for probability density plots (default `"gray_r"`) |

**Plot flags** (set on the instance before calling plot methods):

| Attribute | Default | Effect |
|-----------|---------|--------|
| `_plotLowestMisfitModels` | `False` | Overlay the lowest-misfit models |
| `_plotMisfits` | `True` | Include misfit convergence panel |
| `_plotSynthetic` | `False` | Overlay synthetic data (if available) |

---

## Dependencies

| Package | Role |
|---------|------|
| `numpy` | Array operations |
| `matplotlib` | All plotting |

---

## Notes

- Reads GA-format binary or ASCII rjMCMC output files; file layout is
  specific to the Geoscience Australia rjMCMC code.
- Uses `matplotlib.style.use('fast')` globally on import.
- For results produced by `transdim.py`, use `transdim_viz.py` instead.

---

Author: Rakib Hassan (Geoscience Australia); adapted by Volker Rath (DIAS)
