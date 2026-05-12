#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script-style driver for transdimensional (rjMCMC) isotropic 1-D MT inversion.

This is intentionally NOT a CLI. Edit the USER CONFIG and run:

    python mt_transdim_iso1d.py

Preserved conventions: PY4MTX environment variables, explicit sys.path
setup, startup title print, example in-file model (MODEL0 / MODEL0_ANISO).
Helpers are imported from transdim.py; plotting from transdim_viz.py.
Data I/O uses data_proc.py (load_edi, load_npz, compute_rhophas, compute_pt)
via transdim.load_site().

The sampler uses reversible-jump MCMC (Green 1995) so that the number of
layers *k* is itself a free parameter. Multiple independent chains are
run in parallel via joblib.

Likelihood modes
----------------
This isotropic driver is restricted to the isotropic likelihoods:

- ``"Zdet"`` — Re/Im of determinant impedance ``Z_det``
- ``"rhoa"`` — log10 apparent resistivity derived from ``Z_det``

@author:    Volker Rath (DIAS)
@project:   py4mt — Python for Magnetotellurics
@created:   2026-03-07 — Claude (Opus 4.6, Anthropic)
@modified:  2026-03-07 — anisotropic example block, viz split
@modified:  2026-03-08 — data I/O via data_proc
@modified:  2026-03-09 — Z_det likelihood for isotropic; Z-component + PT
                          likelihood for anisotropic; helpers moved to transdim.py

Author: Volker Rath (DIAS)
Created with the help of ChatGPT (GPT-5 Thinking) on 2026-03-11 UTC
"""

from __future__ import annotations

import json
import os
import sys
import inspect
import warnings
from pathlib import Path
import glob

import numpy as np

SUPPRESS_FUTUREWARNINGS = True
if SUPPRESS_FUTUREWARNINGS:
    warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
#  Environment variables / path settings
# =============================================================================
PY4MTX_DATA = os.environ.get("PY4MTX_DATA", "")
PY4MTX_ROOT = os.environ.get("PY4MTX_ROOT", "")

if not PY4MTX_ROOT:
    sys.exit("PY4MTX_ROOT not set! Exit.")
if not PY4MTX_DATA:
    sys.exit("PY4MTX_DATA not set! Exit.")

for _base in [PY4MTX_ROOT + "/py4mt/modules/"]:
    for _p in [Path(_base), *Path(_base).rglob("*")]:
        if _p.is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

import util
from version import versionstrg

import transdim
import transdim_viz

version, _ = versionstrg()
fname = inspect.getfile(inspect.currentframe())

titstrng = util.print_title(version=version, fname=fname, out=False)
print(titstrng + "\n\n")

PLOT_OBSERVED = True

# =============================================================================
#  Example starting model — ISOTROPIC
# =============================================================================
N_LAYER = 5
H_M = np.r_[np.logspace(np.log10(500.0), np.log10(1000.0), N_LAYER - 1), 0.0]
RHO_BG = 300.0

MODEL0 = {
    "prior_name": "transdim_start_iso",
    "h_m": H_M,
    "sigma_min": (1.0 / RHO_BG) * np.ones(N_LAYER, dtype=float),
    "sigma_max": (1.0 / RHO_BG) * np.ones(N_LAYER, dtype=float),
    "strike_deg": np.zeros(N_LAYER, dtype=float),
    "is_iso": np.ones(N_LAYER, dtype=bool),
    "is_fix": np.zeros(N_LAYER, dtype=bool),
}

# =============================================================================
#  Example starting model — ANISOTROPIC
# =============================================================================
N_LAYER_ANISO = 6
H_M_ANISO = np.r_[np.logspace(np.log10(500.0), np.log10(1000.0), N_LAYER - 1), 0.0]

RHO_MAX_ANISO = np.array([200.0, 300.0, 500.0, 800.0, 300.0, 100.0])
RHO_MIN_ANISO = np.array([200.0, 100.0, 100.0, 200.0, 300.0, 100.0])
STRIKE_ANISO = np.array([0.0, 30.0, 45.0, 45.0, 0.0, 0.0])

MODEL0_ANISO = {
    "prior_name": "transdim_start_aniso",
    "h_m": H_M_ANISO,
    "sigma_min": 1.0 / RHO_MAX_ANISO,
    "sigma_max": 1.0 / RHO_MIN_ANISO,
    "strike_deg": STRIKE_ANISO,
    "is_iso": (RHO_MAX_ANISO == RHO_MIN_ANISO),
    "is_fix": np.zeros(N_LAYER_ANISO, dtype=bool),
}

# =============================================================================
#  USER CONFIG — forward model
# =============================================================================
USE_ANISO = False

# =============================================================================
#  USER CONFIG — likelihood
# =============================================================================
LIKELIHOOD_MODE = "Zdet"
Z_COMPS = ("xx", "xy", "yx", "yy")
USE_PT = True
PT_COMPS = ("xx", "xy", "yx", "yy")

# =============================================================================
#  USER CONFIG — data
# =============================================================================
MCMC_DATA = PY4MTX_ROOT + "/py4mt/data/edi/mcmc/"

INPUT_FORMAT = "npz"
EDI_DIR = MCMC_DATA
INPUT_GLOB = MCMC_DATA + "*proc.npz"

MODEL_NPZ = MCMC_DATA + "model0.npz"
MODEL_DIRECT = MODEL0
MODEL_DIRECT_SAVE_PATH = MODEL_NPZ
MODEL_DIRECT_OVERWRITE = True

NOISE_LEVEL = 0.02
SIGMA_FLOOR = 0.0

ERR_METHOD = "bootstrap"
ERR_NSIM = 200
COMPUTE_PT = True

# =============================================================================
#  USER CONFIG — prior bounds
# =============================================================================
K_MIN = 1
K_MAX = 20
DEPTH_MIN = 50.0
DEPTH_MAX = 3000.0
LOG10_RHO_BOUNDS = (-1.0, 4.0)

LOG10_ANISO_BOUNDS = (0.0, 1.5)
STRIKE_BOUNDS_DEG = (-90.0, 90.0)

# =============================================================================
#  USER CONFIG — sampler
# =============================================================================
N_ITERATIONS = 250_000
BURN_IN = 50_000
THIN = 10

PROPOSAL_WEIGHTS = (0.20, 0.20, 0.25, 0.35)

SIGMA_BIRTH_RHO = 0.03
SIGMA_MOVE_Z = 50.0
SIGMA_CHANGE_RHO = 0.05

SIGMA_BIRTH_ANISO = 0.10
SIGMA_BIRTH_STRIKE = 10.0
SIGMA_CHANGE_ANISO = 0.05
SIGMA_CHANGE_STRIKE = 5.0

# =============================================================================
#  USER CONFIG — parallel chains
# =============================================================================
N_CHAINS = 12
N_JOBS = 12
BASE_SEED = transdim.generate_seed()

# =============================================================================
#  USER CONFIG — output
# =============================================================================
OUTDIR = MCMC_DATA + "rjmcmc_iso"

DEPTH_GRID_MAX = 3000.0

lower68, upper68, _ = util.get_percentile(nsig=1)
lower95, upper95, _ = util.get_percentile(nsig=2)
QPAIRS = ((lower95, upper95), (lower68, upper68))

PROGRESSBAR = True

# =============================================================================
#  Metadata helpers
# =============================================================================
def _json_safe(obj):
    """Convert common NumPy / pathlib objects to JSON-safe Python objects."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, complex):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    return obj


def export_metadata(
    station: str,
    site: dict,
    results: dict,
    outdir: Path,
    input_path: str | Path,
    fig_paths: dict,
    initial_model,
    likelihood_mode: str,
    use_aniso: bool,
    prior,
    config,
    base_seed: int,
    n_chains: int,
    n_jobs: int,
    z_comps=None,
    use_pt=None,
    pt_comps=None,
) -> tuple[Path, Path]:
    """Write per-station RJMCMC metadata to JSON and NPZ files."""
    n_layers = np.asarray(results.get("n_layers", []), dtype=int)
    layer_hist = {}
    if n_layers.size:
        binc = np.bincount(n_layers)
        layer_hist = {int(i): int(v) for i, v in enumerate(binc) if v > 0}

    meta = {
        "station": station,
        "created_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "input_path": str(input_path),
        "outdir": str(outdir),
        "use_aniso": bool(use_aniso),
        "likelihood_mode": str(likelihood_mode),
        "z_comps": None if z_comps is None else list(z_comps),
        "use_pt": None if use_pt is None else bool(use_pt),
        "pt_comps": None if pt_comps is None else list(pt_comps),
        "n_frequencies": int(len(site["frequencies"])),
        "frequency_min_hz": float(np.nanmin(site["frequencies"])),
        "frequency_max_hz": float(np.nanmax(site["frequencies"])),
        "has_Z": bool("Z" in site),
        "has_Zdet": bool("Zdet" in site),
        "has_PT": bool("PT" in site),
        "prior": {
            "k_min": int(prior.k_min),
            "k_max": int(prior.k_max),
            "depth_min": float(prior.depth_min),
            "depth_max": float(prior.depth_max),
            "log_rho_min": float(prior.log_rho_min),
            "log_rho_max": float(prior.log_rho_max),
            "log_aniso_min": float(prior.log_aniso_min),
            "log_aniso_max": float(prior.log_aniso_max),
            "strike_min": float(prior.strike_min),
            "strike_max": float(prior.strike_max),
        },
        "config": {
            "n_iterations": int(config.n_iterations),
            "burn_in": int(config.burn_in),
            "thin": int(config.thin),
            "proposal_weights": [float(v) for v in config.proposal_weights],
            "sigma_birth_rho": float(config.sigma_birth_rho),
            "sigma_move_z": float(config.sigma_move_z),
            "sigma_change_rho": float(config.sigma_change_rho),
            "sigma_birth_aniso": float(config.sigma_birth_aniso),
            "sigma_birth_strike": float(config.sigma_birth_strike),
            "sigma_change_aniso": float(config.sigma_change_aniso),
            "sigma_change_strike": float(config.sigma_change_strike),
            "verbose": bool(config.verbose),
        },
        "parallel": {
            "n_chains": int(n_chains),
            "n_jobs": int(n_jobs),
            "base_seed": int(base_seed),
        },
        "initial_model": {
            "k": int(initial_model.k),
            "n_layers": int(initial_model.n_layers),
            "depths_m": np.asarray(initial_model.depths, dtype=float),
            "log10_rho": np.asarray(initial_model.log_resistivities, dtype=float),
            "resistivity_ohm_m": np.asarray(initial_model.get_resistivities(), dtype=float),
            "is_anisotropic": bool(initial_model.is_anisotropic),
            "aniso_ratios": None if initial_model.aniso_ratios is None else np.asarray(initial_model.aniso_ratios, dtype=float),
            "strikes_deg": None if initial_model.strikes is None else np.asarray(initial_model.strikes, dtype=float),
        },
        "posterior": {
            "n_samples": int(len(results.get("models", []))),
            "median_n_layers": float(np.median(n_layers)) if n_layers.size else None,
            "mode_n_layers": int(np.bincount(n_layers).argmax()) if n_layers.size else None,
            "gelman_rubin": float(results.get("gelman_rubin", np.nan)),
            "layer_histogram": layer_hist,
        },
        "figures": {k: str(v) for k, v in fig_paths.items()},
    }

    json_path = outdir / f"{station}_rjmcmc_metadata.json"
    npz_path = outdir / f"{station}_rjmcmc_metadata.npz"

    with open(json_path, "w", encoding="utf-8") as fobj:
        json.dump(_json_safe(meta), fobj, indent=2, sort_keys=True)

    np.savez_compressed(
        str(npz_path),
        metadata_json=np.array(json.dumps(_json_safe(meta), sort_keys=True), dtype=object),
        station=np.array(station, dtype=object),
        input_path=np.array(str(input_path), dtype=object),
        outdir=np.array(str(outdir), dtype=object),
        likelihood_mode=np.array(str(likelihood_mode), dtype=object),
        use_aniso=np.array(bool(use_aniso)),
        n_frequencies=np.array(int(len(site["frequencies"]))),
        frequency_hz=np.asarray(site["frequencies"], dtype=float),
        n_layers=np.asarray(results.get("n_layers", [])),
        gelman_rubin=np.array(float(results.get("gelman_rubin", np.nan))),
        initial_depths_m=np.asarray(initial_model.depths, dtype=float),
        initial_log10_rho=np.asarray(initial_model.log_resistivities, dtype=float),
        initial_resistivity_ohm_m=np.asarray(initial_model.get_resistivities(), dtype=float),
        initial_aniso_ratios=np.asarray(
            [] if initial_model.aniso_ratios is None else initial_model.aniso_ratios,
            dtype=float,
        ),
        initial_strikes_deg=np.asarray(
            [] if initial_model.strikes is None else initial_model.strikes,
            dtype=float,
        ),
    )

    return json_path, npz_path


# =============================================================================
#  Build objects from USER CONFIG
# =============================================================================
prior = transdim.Prior(
    k_min=K_MIN,
    k_max=K_MAX,
    depth_min=DEPTH_MIN,
    depth_max=DEPTH_MAX,
    log_rho_min=LOG10_RHO_BOUNDS[0],
    log_rho_max=LOG10_RHO_BOUNDS[1],
    log_aniso_min=LOG10_ANISO_BOUNDS[0],
    log_aniso_max=LOG10_ANISO_BOUNDS[1],
    strike_min=STRIKE_BOUNDS_DEG[0],
    strike_max=STRIKE_BOUNDS_DEG[1],
)

config = transdim.RjMCMCConfig(
    n_iterations=N_ITERATIONS,
    burn_in=BURN_IN,
    thin=THIN,
    proposal_weights=PROPOSAL_WEIGHTS,
    sigma_birth_rho=SIGMA_BIRTH_RHO,
    sigma_move_z=SIGMA_MOVE_Z,
    sigma_change_rho=SIGMA_CHANGE_RHO,
    sigma_birth_aniso=SIGMA_BIRTH_ANISO,
    sigma_birth_strike=SIGMA_BIRTH_STRIKE,
    sigma_change_aniso=SIGMA_CHANGE_ANISO,
    sigma_change_strike=SIGMA_CHANGE_STRIKE,
    verbose=PROGRESSBAR,
)

likelihood_mode = LIKELIHOOD_MODE
if likelihood_mode.lower() not in {"zdet", "rhoa"}:
    raise ValueError("Isotropic driver supports only likelihood_mode='Zdet' or 'rhoa'.")

print(f"Likelihood mode: {likelihood_mode}")

# =============================================================================
#  Run
# =============================================================================
outdir = transdim.ensure_dir(OUTDIR)

if USE_ANISO and not transdim.has_aniso():
    sys.exit(
        "USE_ANISO=True but aniso.py not found on PYTHONPATH. "
        "Place aniso.py in the working directory or set PYTHONPATH. Exit."
    )

if MODEL_DIRECT is not None:
    initial_model = transdim.model0_to_layered(MODEL_DIRECT, USE_ANISO)
    if MODEL_DIRECT_OVERWRITE or not Path(MODEL_DIRECT_SAVE_PATH).exists():
        np.savez_compressed(
            MODEL_DIRECT_SAVE_PATH,
            **{k: np.asarray(v) for k, v in MODEL_DIRECT.items()},
        )
else:
    with np.load(MODEL_NPZ, allow_pickle=True) as npz:
        m0 = {k: npz[k] for k in npz.files}
    initial_model = transdim.model0_to_layered(m0, USE_ANISO)

print(f"Starting model: k={initial_model.k} interfaces, "
      f"{initial_model.n_layers} layers, "
      f"anisotropic={initial_model.is_anisotropic}")
print()

in_files = sorted(glob.glob(INPUT_GLOB))
if not in_files:
    raise FileNotFoundError(f"No inputs matched: {INPUT_GLOB}")

print(f"Found {len(in_files)} input file(s) ({INPUT_FORMAT}).\n")

for f in in_files:
    site = transdim.load_site(
        f,
        noise_level=NOISE_LEVEL,
        sigma_floor=SIGMA_FLOOR,
        err_method=ERR_METHOD,
        err_nsim=ERR_NSIM,
        do_compute_pt=COMPUTE_PT,
    )
    station = str(site.get("station", Path(f).stem))
    print(f"{'='*70}")
    print(f"  Station: {station}")
    print(f"{'='*70}")
    print(f"  Frequencies: {len(site['frequencies'])}, "
          f"range {site['frequencies'].min():.4f}–{site['frequencies'].max():.1f} Hz")

    has_Z = "Z" in site
    has_Zdet = "Zdet" in site
    has_PT = "PT" in site
    print(f"  Z tensor:     {'yes' if has_Z else 'no'}")
    print(f"  Z_det:        {'yes' if has_Zdet else 'no'}")
    print(f"  Phase tensor: {'yes' if has_PT else 'no'}")
    print()

    if PLOT_OBSERVED:
        print("yet to come!")

    lmode = likelihood_mode.lower().strip()
    kw_sampler = dict(
        frequencies=site["frequencies"],
        observed=site["rho_a"],
        sigma=site["sigma"],
        prior=prior,
        config=config,
        n_chains=N_CHAINS,
        n_jobs=N_JOBS,
        base_seed=BASE_SEED,
        use_aniso=USE_ANISO,
        likelihood_mode=lmode,
        initial_model=initial_model,
    )

    if lmode == "zdet":
        if not has_Zdet:
            print("  WARNING: Z_det not available; falling back to rhoa likelihood.")
            kw_sampler["likelihood_mode"] = "rhoa"
        else:
            kw_sampler["observed_Zdet"] = site["Zdet"]
            if "Zdet_err" in site:
                kw_sampler["Zdet_sigma"] = site["Zdet_err"]
            else:
                kw_sampler["Zdet_sigma"] = 0.05 * np.abs(site["Zdet"])
                print("  NOTE: No Zdet_err; using 5% of |Zdet| as uncertainty.")
            if "rho_a_det" in site:
                kw_sampler["observed"] = site["rho_a_det"]
                Zabs = np.abs(site["Zdet"])
                with np.errstate(divide="ignore", invalid="ignore"):
                    sig_log = np.where(
                        Zabs > 0,
                        2.0 * np.asarray(kw_sampler["Zdet_sigma"]) / (Zabs * np.log(10)),
                        NOISE_LEVEL,
                    )
                kw_sampler["sigma"] = np.maximum(sig_log, SIGMA_FLOOR)

    results = transdim.run_parallel_rjmcmc(**kw_sampler)

    print(f"\nPosterior summary for {station}:")
    print(f"  Median number of layers: {np.median(results['n_layers']):.0f}")
    print(f"  Mode number of layers:   {np.bincount(results['n_layers']).argmax()}")
    print(f"  R-hat: {results.get('gelman_rubin', np.nan):.4f}")
    print()

    npz_path = outdir / f"{station}_rjmcmc.npz"
    transdim.save_results_npz(results, npz_path)

    sum_dict = transdim.build_rjmcmc_summary(
        station, results, DEPTH_GRID_MAX, QPAIRS, USE_ANISO)
    sum_path = outdir / f"{station}_rjmcmc_summary.npz"
    np.savez_compressed(str(sum_path), **{
        k: np.asarray(v) if not isinstance(v, (str, dict)) else str(v)
        for k, v in sum_dict.items()
    })
    print(f"Wrote: {sum_path}")

    fig_path = outdir / f"{station}_rjmcmc.png"
    transdim_viz.plot_results(
        results,
        true_model=None,
        frequencies=site["frequencies"],
        observed=kw_sampler["observed"],
        depth_max=DEPTH_GRID_MAX,
        use_aniso=USE_ANISO,
        save_path=str(fig_path),
    )

    qc_path = outdir / f"{station}_rjmcmc_qc.png"
    transdim_viz.plot_qc(
        results,
        frequencies=site["frequencies"],
        observed=kw_sampler["observed"],
        sigma=kw_sampler["sigma"],
        station=station,
        use_aniso=USE_ANISO,
        observed_Z=site.get("Z", None),
        observed_Z_err=site.get("Z_err", None),
        z_comps=("xy", "yx"),
        show_pt=has_PT or (USE_ANISO and has_Z),
        observed_PT=site.get("PT", None),
        observed_PT_err=site.get("PT_err", None),
        save_path=str(qc_path),
    )

    model_path = outdir / f"{station}_rjmcmc_model.png"
    transdim_viz.plot_posterior_model(
        results,
        depth_max=DEPTH_GRID_MAX,
        true_model=None,
        station=station,
        use_aniso=USE_ANISO,
        save_path=str(model_path),
    )

    fig_paths = {
        "results": fig_path,
        "qc": qc_path,
        "model": model_path,
    }
    meta_json, meta_npz = export_metadata(
        station=station,
        site=site,
        results=results,
        outdir=outdir,
        input_path=f,
        fig_paths=fig_paths,
        initial_model=initial_model,
        likelihood_mode=kw_sampler["likelihood_mode"],
        use_aniso=USE_ANISO,
        prior=prior,
        config=config,
        base_seed=BASE_SEED,
        n_chains=N_CHAINS,
        n_jobs=N_JOBS,
        z_comps=kw_sampler.get("z_comps", None),
        use_pt=kw_sampler.get("use_pt", None),
        pt_comps=kw_sampler.get("pt_comps", None),
    )
    print(f"Wrote: {meta_json}")
    print(f"Wrote: {meta_npz}")
    print()

print("\nDone.\n")
