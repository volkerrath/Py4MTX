#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transdim_viz.py — Plotting routines for transdimensional rjMCMC results.
=========================================================================

Separated from transdim.py so that the core sampler module has no
matplotlib dependency (useful for headless / HPC environments).

All functions take the results dict returned by
``transdim.run_rjmcmc`` or ``transdim.run_parallel_rjmcmc``.

Contents
--------
    plot_results                — multi-panel diagnostic figure
    plot_resistivity_profile    — single-panel posterior ρ(z)
    plot_dimension_histogram    — posterior on number of layers
    plot_data_fit               — observed vs. posterior-predicted ρ_a(f)
    plot_chain_traces           — per-chain log-likelihood traces
    plot_aniso_profile          — posterior anisotropy ratio vs. depth
    plot_strike_profile         — posterior strike angle vs. depth

@author:    Volker Rath (DIAS) / Claude (Opus 4.6, Anthropic)
@project:   py4mt — Python for Magnetotellurics
@created:   2026-03-07 — split from transdim.py
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

import transdim


# =============================================================================
#  Composite diagnostic figure
# =============================================================================

def plot_results(
    results: Dict,
    true_model: Optional[transdim.LayeredModel],
    frequencies: np.ndarray,
    observed: np.ndarray,
    depth_max: float = 3000.0,
    use_aniso: bool = False,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Multi-panel diagnostic figure for (multi-chain) rjMCMC results.

    Panels:
      1. Posterior resistivity profile (median + 90 % credible interval)
      2. Number-of-layers histogram
      3. Data fit (posterior samples vs. observations)
      4. Log-likelihood traces (per chain if available)
      5. (aniso only) Posterior anisotropy-ratio profile

    Returns the ``matplotlib.figure.Figure``.
    """
    n_panels = 5 if use_aniso else 4
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 8))

    depth_grid = np.linspace(1, depth_max, 500)

    plot_resistivity_profile(
        axes[0], results, depth_grid, true_model=true_model, depth_max=depth_max)
    plot_dimension_histogram(
        axes[1], results, true_model=true_model)
    plot_data_fit(
        axes[2], results, frequencies, observed, use_aniso=use_aniso)
    plot_chain_traces(
        axes[3], results)

    if use_aniso:
        plot_aniso_profile(
            axes[4], results, depth_grid, true_model=true_model, depth_max=depth_max)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    plt.close(fig)
    return fig


# =============================================================================
#  Individual panels (reusable)
# =============================================================================

def plot_resistivity_profile(
    ax: plt.Axes,
    results: Dict,
    depth_grid: np.ndarray,
    true_model: Optional[transdim.LayeredModel] = None,
    depth_max: float = 3000.0,
) -> None:
    """Posterior resistivity profile with credible interval."""
    prof = transdim.compute_posterior_profile(results["models"], depth_grid)

    ax.fill_betweenx(
        prof["depth"], prof["p05"], prof["p95"],
        alpha=0.25, color="steelblue", label="90% credible")
    ax.plot(prof["median"], prof["depth"], "b-", lw=2, label="Median")
    ax.plot(prof["mean"], prof["depth"], "b--", lw=1, alpha=0.7, label="Mean")

    if true_model is not None:
        _overlay_true_model(ax, true_model, depth_max)

    ax.set_xscale("log")
    ax.set_xlabel("Resistivity (Ω·m)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Posterior Resistivity Profile", fontweight="bold")
    ax.invert_yaxis()
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_dimension_histogram(
    ax: plt.Axes,
    results: Dict,
    true_model: Optional[transdim.LayeredModel] = None,
) -> None:
    """Posterior histogram of the number of layers."""
    k_vals = results["n_layers"]
    bins = np.arange(k_vals.min() - 0.5, k_vals.max() + 1.5, 1)
    ax.hist(k_vals, bins=bins, color="steelblue", edgecolor="white",
            density=True, alpha=0.8)
    if true_model is not None:
        ax.axvline(true_model.n_layers, color="red", ls="--", lw=2,
                   label=f"True k={true_model.n_layers}")
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel("Posterior Probability")
    ax.set_title("Model Dimension", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_data_fit(
    ax: plt.Axes,
    results: Dict,
    frequencies: np.ndarray,
    observed: np.ndarray,
    use_aniso: bool = False,
    n_show: int = 50,
) -> None:
    """Overlay observed data and posterior-predicted apparent resistivities."""
    ax.loglog(frequencies, observed, "ko", ms=6, label="Observed")

    n_show = min(n_show, len(results["models"]))
    indices = np.random.choice(len(results["models"]), n_show, replace=False)

    for idx in indices:
        m = results["models"][idx]
        if use_aniso and m.is_anisotropic:
            pred = transdim.mt_forward_1d_anisotropic(
                m.get_thicknesses(), m.get_resistivities(), frequencies,
                m.aniso_ratios, m.strikes)
            ax.loglog(frequencies, pred["rho_a_xy"], "b-", alpha=0.05, lw=1)
        else:
            pred = transdim.mt_forward_1d_isotropic(
                m.get_thicknesses(), m.get_resistivities(), frequencies)
            ax.loglog(frequencies, pred, "b-", alpha=0.05, lw=1)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Apparent Resistivity (Ω·m)")
    ax.set_title("Data Fit", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")


def plot_chain_traces(
    ax: plt.Axes,
    results: Dict,
) -> None:
    """Per-chain log-likelihood traces (or merged trace if no chain info)."""
    if "chains" in results:
        colors = plt.cm.tab10(np.linspace(0, 1, len(results["chains"])))
        for i, cr in enumerate(results["chains"]):
            ax.plot(cr["log_likes"], "-", lw=0.3, alpha=0.6,
                    color=colors[i], label=f"Chain {i}")
        ax.legend(fontsize=7, ncol=2)
    else:
        ax.plot(results["log_likes"], "k-", lw=0.3, alpha=0.5)

    ax.set_xlabel("Sample Index (post burn-in)")
    ax.set_ylabel("Log-Likelihood")
    ax.set_title("Chain Convergence", fontweight="bold")
    ax.grid(True, alpha=0.3)


def plot_aniso_profile(
    ax: plt.Axes,
    results: Dict,
    depth_grid: np.ndarray,
    true_model: Optional[transdim.LayeredModel] = None,
    depth_max: float = 3000.0,
) -> None:
    """Posterior anisotropy-ratio profile with credible interval."""
    aprof = transdim.compute_posterior_aniso_profile(results["models"], depth_grid)

    ax.fill_betweenx(
        aprof["depth"], aprof["aniso_p05"], aprof["aniso_p95"],
        alpha=0.25, color="darkorange", label="90% credible")
    ax.plot(aprof["aniso_median"], aprof["depth"],
            color="darkorange", lw=2, label="Median")

    if true_model is not None and true_model.is_anisotropic:
        true_depths = np.concatenate(([0], true_model.interfaces, [depth_max]))
        for i in range(len(true_model.aniso_ratios)):
            ax.plot(
                [true_model.aniso_ratios[i], true_model.aniso_ratios[i]],
                [true_depths[i], true_depths[i + 1]],
                "r-", lw=2, label="True" if i == 0 else None)

    ax.set_xlabel("Anisotropy Ratio (ρ_max/ρ_min)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Posterior Anisotropy", fontweight="bold")
    ax.invert_yaxis()
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_strike_profile(
    ax: plt.Axes,
    results: Dict,
    depth_grid: np.ndarray,
    true_model: Optional[transdim.LayeredModel] = None,
    depth_max: float = 3000.0,
) -> None:
    """Posterior strike-angle profile with credible interval."""
    aprof = transdim.compute_posterior_aniso_profile(results["models"], depth_grid)

    ax.fill_betweenx(
        aprof["depth"], aprof["strike_p05"], aprof["strike_p95"],
        alpha=0.25, color="seagreen", label="90% credible")
    ax.plot(aprof["strike_median"], aprof["depth"],
            color="seagreen", lw=2, label="Median")

    if true_model is not None and true_model.is_anisotropic:
        true_depths = np.concatenate(([0], true_model.interfaces, [depth_max]))
        for i in range(len(true_model.strikes)):
            ax.plot(
                [true_model.strikes[i], true_model.strikes[i]],
                [true_depths[i], true_depths[i + 1]],
                "r-", lw=2, label="True" if i == 0 else None)

    ax.set_xlabel("Strike (°)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Posterior Strike", fontweight="bold")
    ax.invert_yaxis()
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


# =============================================================================
#  QC summary figure  (cf. plot_qc.png)
# =============================================================================

# Z-component index map
_ZCOMP_IDX = {"xx": (0, 0), "xy": (0, 1), "yx": (1, 0), "yy": (1, 1)}
_ZCOMP_LABELS = {"xx": r"$Z_{xx}$", "xy": r"$Z_{xy}$",
                 "yx": r"$Z_{yx}$", "yy": r"$Z_{yy}$"}
_PTCOMP_LABELS = {"xx": r"$\Phi_{xx}$", "xy": r"$\Phi_{xy}$",
                  "yx": r"$\Phi_{yx}$", "yy": r"$\Phi_{yy}$"}

_COMP_COLORS = {"xx": "purple", "xy": "blue", "yx": "red", "yy": "green"}


def plot_qc(
    results: Dict,
    frequencies: np.ndarray,
    observed: np.ndarray,
    sigma: np.ndarray,
    station: str = "",
    use_aniso: bool = False,
    observed_Z: Optional[np.ndarray] = None,
    observed_Z_err: Optional[np.ndarray] = None,
    z_comps: tuple = ("xy", "yx"),
    show_pt: bool = False,
    observed_PT: Optional[np.ndarray] = None,
    observed_PT_err: Optional[np.ndarray] = None,
    pt_comps: tuple = ("xx", "xy", "yx", "yy"),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """QC figure with flexible data-fit panels.

    Modes (selected automatically):

    1. **ρ_a + phase** (default) — when ``observed_Z`` is None.
       Panels A (apparent resistivity vs period) and B (phase vs period).

    2. **Impedance** — when ``observed_Z`` is provided.
       Panels A (Re Z components vs period) and B (Im Z components vs period).

    3. **Phase tensor** — when ``show_pt=True`` an additional row of panels
       shows the selected PT components vs period.

    The bottom row always contains:
       Panel C (data misfit vs sampling step) and Panel D (k histogram).

    Parameters
    ----------
    observed_Z : (n_freq, 2, 2) complex, optional
        Observed impedance tensor.  When provided, panels A/B show Z.
    observed_Z_err : (n_freq, 2, 2) float, optional
        Impedance uncertainties (absolute).
    z_comps : tuple of str
        Z components to plot (subset of ``'xx', 'xy', 'yx', 'yy'``).
    show_pt : bool
        If True, add a row for phase-tensor components.
    observed_PT : (n_freq, 2, 2) float, optional
        Observed phase tensor.  Computed from ``observed_Z`` if not given.
    observed_PT_err : (n_freq, 2, 2) float, optional
        PT uncertainties.
    pt_comps : tuple of str
        PT components to plot.
    """
    periods = 1.0 / frequencies
    has_chains = "chains" in results
    impedance_mode = observed_Z is not None

    # ---- determine layout --------------------------------------------------
    n_rows = 2
    if show_pt:
        n_rows = 3

    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows))
    fig.suptitle(
        f"MT Station {station} — 1D rj-McMC Inversion" if station
        else "1D rj-McMC Inversion QC",
        fontsize=14, fontweight="bold",
    )

    # ---- best-fit models per chain -----------------------------------------
    if has_chains:
        best_models = []
        for cr in results["chains"]:
            best_idx = int(np.argmax(cr["log_likes"]))
            best_models.append(cr["models"][best_idx])
    else:
        best_idx = int(np.argmax(results["log_likes"]))
        best_models = [results["models"][best_idx]]

    # ---- compute best-fit predictions ---------------------------------------
    best_preds = []     # list of dicts with rho_a, phase_deg, Z_re, Z_im
    best_Z = []         # list of (nf, 2, 2) complex tensors (for impedance/PT panels)
    for m in best_models:
        if use_aniso and m.is_anisotropic:
            pred = transdim.mt_forward_1d_anisotropic_impedance(
                m.get_thicknesses(), m.get_resistivities(), frequencies,
                m.aniso_ratios, m.strikes)
            best_preds.append(pred)
            best_Z.append(pred["Z"])
        else:
            pred = transdim.mt_forward_1d_isotropic(
                m.get_thicknesses(), m.get_resistivities(), frequencies,
                full_output=True)
            best_preds.append(pred)
            # Reconstruct (nf, 2, 2) tensor for impedance/PT panels
            nf = len(frequencies)
            Z_scalar = pred["Z_re"] + 1j * pred["Z_im"]
            Z_tensor = np.zeros((nf, 2, 2), dtype=np.complex128)
            Z_tensor[:, 0, 1] = Z_scalar     # Zxy
            Z_tensor[:, 1, 0] = -Z_scalar    # Zyx = -Zxy
            best_Z.append(Z_tensor)

    if impedance_mode:
        # ---- Panel A: Re(Z) vs Period --------------------------------------
        ax = axes[0, 0]
        for comp in z_comps:
            i, j = _ZCOMP_IDX[comp]
            color = _COMP_COLORS[comp]
            obs_re = observed_Z[:, i, j].real
            if observed_Z_err is not None:
                ax.errorbar(periods, obs_re, yerr=observed_Z_err[:, i, j],
                            fmt="s", ms=4, capsize=2, color=color, alpha=0.7,
                            label=f"Obs Re({_ZCOMP_LABELS[comp]})")
            else:
                ax.plot(periods, obs_re, "s", ms=4, color=color, alpha=0.7,
                        label=f"Obs Re({_ZCOMP_LABELS[comp]})")
            for Zp in best_Z:
                ax.plot(periods, Zp[:, i, j].real, "-", color=color,
                        lw=1.2, alpha=0.6)

        ax.set_xscale("log")
        ax.set_xlabel("Period [s]")
        ax.set_ylabel("Re(Z)")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3, which="both")
        ax.text(0.02, 0.02, "A", transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="bottom")

        # ---- Panel B: Im(Z) vs Period --------------------------------------
        ax = axes[0, 1]
        for comp in z_comps:
            i, j = _ZCOMP_IDX[comp]
            color = _COMP_COLORS[comp]
            obs_im = observed_Z[:, i, j].imag
            if observed_Z_err is not None:
                ax.errorbar(periods, obs_im, yerr=observed_Z_err[:, i, j],
                            fmt="s", ms=4, capsize=2, color=color, alpha=0.7,
                            label=f"Obs Im({_ZCOMP_LABELS[comp]})")
            else:
                ax.plot(periods, obs_im, "s", ms=4, color=color, alpha=0.7,
                        label=f"Obs Im({_ZCOMP_LABELS[comp]})")
            for Zp in best_Z:
                ax.plot(periods, Zp[:, i, j].imag, "-", color=color,
                        lw=1.2, alpha=0.6)

        ax.set_xscale("log")
        ax.set_xlabel("Period [s]")
        ax.set_ylabel("Im(Z)")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3, which="both")
        ax.text(0.02, 0.02, "B", transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="bottom")

    else:
        # ---- Panel A: Apparent Resistivity vs Period -----------------------
        ax = axes[0, 0]
        ax.errorbar(periods, observed, yerr=observed * sigma * np.log(10),
                    fmt="rs", ms=5, capsize=2, zorder=5, label="Observations")
        for pred in best_preds:
            rho_a = pred.get("rho_a", pred.get("rho_a_xy"))
            ax.plot(periods, rho_a, "b-", lw=1.5, alpha=0.8)
        ax.plot([], [], "b-", lw=1.5, label="Best fit per chain")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Period [s]")
        ax.set_ylabel("Apparent Resistivity [Ω·m]")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which="both")
        ax.text(0.02, 0.02, "A", transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="bottom")

        # ---- Panel B: Phase vs Period --------------------------------------
        ax = axes[0, 1]
        for pred in best_preds:
            if "phase_deg" in pred:
                ax.plot(periods, pred["phase_deg"], "b-", lw=1.5, alpha=0.8)
            else:
                # aniso: derive from Zxy
                Zp = pred["Z"]
                phase = np.abs(np.degrees(np.arctan2(
                    Zp[:, 0, 1].imag, Zp[:, 0, 1].real)))
                ax.plot(periods, phase, "b-", lw=1.5, alpha=0.8)
        ax.plot([], [], "b-", lw=1.5, label="Best fit per chain")
        ax.set_xscale("log")
        ax.set_xlabel("Period [s]")
        ax.set_ylabel("Apparent Phase [Deg]")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which="both")
        ax.text(0.02, 0.02, "B", transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="bottom")

    # ---- Phase-tensor row (optional) ---------------------------------------
    if show_pt:
        # Compute observed PT if not provided
        if observed_PT is None and observed_Z is not None:
            observed_PT = transdim.compute_phase_tensor(observed_Z)

        # PT predictions from best-fit models
        best_PT = [transdim.compute_phase_tensor(Zp) for Zp in best_Z]

        # Left: PT_xx, PT_xy
        ax = axes[1, 0]
        for comp in pt_comps[:2] if len(pt_comps) >= 2 else pt_comps:
            i, j = _ZCOMP_IDX[comp]
            color = _COMP_COLORS[comp]
            if observed_PT is not None:
                if observed_PT_err is not None:
                    ax.errorbar(periods, observed_PT[:, i, j],
                                yerr=observed_PT_err[:, i, j],
                                fmt="s", ms=4, capsize=2, color=color,
                                alpha=0.7,
                                label=f"Obs {_PTCOMP_LABELS[comp]}")
                else:
                    ax.plot(periods, observed_PT[:, i, j], "s", ms=4,
                            color=color, alpha=0.7,
                            label=f"Obs {_PTCOMP_LABELS[comp]}")
            for PTp in best_PT:
                ax.plot(periods, PTp[:, i, j], "-", color=color,
                        lw=1.2, alpha=0.6)

        ax.set_xscale("log")
        ax.set_xlabel("Period [s]")
        ax.set_ylabel("Phase Tensor")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3, which="both")
        ax.text(0.02, 0.02, "PT-1", transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="bottom")

        # Right: PT_yx, PT_yy
        ax = axes[1, 1]
        for comp in pt_comps[2:] if len(pt_comps) > 2 else pt_comps[-1:]:
            i, j = _ZCOMP_IDX[comp]
            color = _COMP_COLORS[comp]
            if observed_PT is not None:
                if observed_PT_err is not None:
                    ax.errorbar(periods, observed_PT[:, i, j],
                                yerr=observed_PT_err[:, i, j],
                                fmt="s", ms=4, capsize=2, color=color,
                                alpha=0.7,
                                label=f"Obs {_PTCOMP_LABELS[comp]}")
                else:
                    ax.plot(periods, observed_PT[:, i, j], "s", ms=4,
                            color=color, alpha=0.7,
                            label=f"Obs {_PTCOMP_LABELS[comp]}")
            for PTp in best_PT:
                ax.plot(periods, PTp[:, i, j], "-", color=color,
                        lw=1.2, alpha=0.6)

        ax.set_xscale("log")
        ax.set_xlabel("Period [s]")
        ax.set_ylabel("Phase Tensor")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3, which="both")
        ax.text(0.02, 0.02, "PT-2", transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="bottom")

    # ---- Misfit and k-histogram (always last row) --------------------------
    row_cd = n_rows - 1

    # ---- Panel C: Misfit trace ---------------------------------------------
    ax = axes[row_cd, 0]
    burn_in = results.get("burn_in", 0)
    if has_chains:
        for i, cr in enumerate(results["chains"]):
            if "full_ll_trace" in cr:
                misfit = -2.0 * cr["full_ll_trace"]
                steps = np.arange(1, len(misfit) + 1)
                ax.plot(steps, misfit, "-", lw=0.6, alpha=0.7,
                        label=f"Chain {i}" if i < 6 else None)
    elif "full_ll_trace" in results:
        misfit = -2.0 * results["full_ll_trace"]
        steps = np.arange(1, len(misfit) + 1)
        ax.plot(steps, misfit, "b-", lw=0.5, alpha=0.5)

    if burn_in > 0:
        ax.axvline(burn_in, color="k", ls=":", lw=1, alpha=0.7)
    n_data = len(frequencies)
    ax.axhline(n_data, color="k", ls=":", lw=1, alpha=0.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Sampling Steps")
    ax.set_ylabel("Data Misfit")
    ax.grid(True, alpha=0.3, which="both")
    if has_chains:
        ax.legend(fontsize=7, ncol=2)
    ax.text(0.02, 0.02, "C", transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="bottom")

    # ---- Panel D: k histogram ----------------------------------------------
    ax = axes[row_cd, 1]
    k_vals = results["n_layers"]
    bins = np.arange(k_vals.min() - 0.5, k_vals.max() + 1.5, 1)
    ax.hist(k_vals, bins=bins, color="steelblue", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.98, "D", transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="top")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    plt.close(fig)
    return fig


# =============================================================================
#  Posterior model figure  (cf. results.png)
# =============================================================================

def plot_posterior_model(
    results: Dict,
    depth_max: float = 5000.0,
    log_rho_range: tuple = (-1.0, 5.0),
    n_rho_bins: int = 200,
    true_model: Optional[transdim.LayeredModel] = None,
    station: str = "",
    use_aniso: bool = False,
    depth_scale: str = "linear",
    strike_range: tuple = (-90.0, 90.0),
    n_strike_bins: int = 120,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Posterior-model figure: 2-D histograms + change-point frequency.

    Isotropic mode (default):
        E  — 2-D histogram of ρ vs depth + overlaid statistics
        F  — change-point frequency vs depth

    Anisotropic mode (``use_aniso=True``):
        E1 — 2-D histogram of ρ_max vs depth
        E2 — 2-D histogram of ρ_min vs depth
        E3 — 2-D histogram of strike vs depth
        F  — change-point frequency vs depth

    Parameters
    ----------
    depth_scale : ``"linear"`` (default) or ``"log"``
        Controls the depth axis spacing.
    strike_range : (float, float)
        Bin range for the strike histogram [degrees].
    n_strike_bins : int
        Number of bins for the strike histogram.
    """
    from matplotlib.colors import LogNorm
    import matplotlib.gridspec as gridspec

    # ---- depth grid --------------------------------------------------------
    if depth_scale == "log":
        depth_grid = np.logspace(0, np.log10(depth_max), 600)
    else:
        depth_grid = np.linspace(1, depth_max, 600)

    log_rho_bins = np.linspace(log_rho_range[0], log_rho_range[1], n_rho_bins + 1)
    strike_bins = np.linspace(strike_range[0], strike_range[1], n_strike_bins + 1)

    # ---- compute shared data -----------------------------------------------
    cpf = transdim.compute_changepoint_frequency(results["models"], depth_grid)

    if use_aniso:
        # Three histograms: rho_max, rho_min, strike
        h_rmax = transdim.compute_posterior_histogram(
            results["models"], depth_grid, log_rho_bins, prop="rho")
        h_rmin = transdim.compute_posterior_histogram(
            results["models"], depth_grid, log_rho_bins, prop="rho_min")
        h_str = transdim.compute_posterior_histogram(
            results["models"], depth_grid, strike_bins, prop="strike")

        prof_rmax = transdim.compute_posterior_profile(
            results["models"], depth_grid)
        prof_rmin = transdim.compute_posterior_rhomin_profile(
            results["models"], depth_grid)
        aprof = transdim.compute_posterior_aniso_profile(
            results["models"], depth_grid)

        n_hist_panels = 3
    else:
        h_rmax = transdim.compute_posterior_histogram(
            results["models"], depth_grid, log_rho_bins, prop="rho")
        prof_rmax = transdim.compute_posterior_profile(
            results["models"], depth_grid)
        n_hist_panels = 1

    # ---- figure layout -----------------------------------------------------
    fig = plt.figure(figsize=(4.5 * n_hist_panels + 3, 10))
    ratios = [3] * n_hist_panels + [1]
    gs = gridspec.GridSpec(1, n_hist_panels + 1, width_ratios=ratios, wspace=0.08)

    # ---- helper: draw one rho histogram panel ------------------------------
    def _draw_rho_panel(ax, phist, prof, label, panel_tag, show_true_rho=None):
        h2d = np.ma.masked_where(phist["hist2d"] < 1, phist["hist2d"])
        rho_edges = 10 ** phist["value_bins"]
        depth_edges = np.concatenate(
            ([depth_grid[0]], 0.5 * (depth_grid[:-1] + depth_grid[1:]),
             [depth_grid[-1]]))

        pcm = ax.pcolormesh(
            rho_edges, depth_edges, h2d,
            norm=LogNorm(vmin=1, vmax=max(phist["hist2d"].max(), 2)),
            cmap="RdYlGn_r", shading="flat", rasterized=True)
        fig.colorbar(pcm, ax=ax, label="Log freq.", shrink=0.5, pad=0.02)

        ax.plot(prof["median"], depth_grid, "k-", lw=2.0, label="Median")
        ax.plot(prof["p05"], depth_grid, "k--", lw=1.5, alpha=0.8)
        ax.plot(prof["p95"], depth_grid, "k--", lw=1.5, alpha=0.8,
                label="10th & 90th %")
        ax.plot(prof["mean"], depth_grid, "b-", lw=1.5, label="Mean")
        ax.plot(phist["mode"], depth_grid, "g-", lw=1.5, label="Mode")

        if show_true_rho is not None:
            _overlay_true_staircase(ax, show_true_rho, true_model, depth_max)

        ax.set_xscale("log")
        if depth_scale == "log":
            ax.set_yscale("log")
        ax.invert_yaxis()
        ax.set_xlabel(f"{label} [Ω·m]")
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.2, which="both")
        ax.text(0.02, 0.02, panel_tag, transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="bottom")

    # ---- helper: draw strike histogram panel -------------------------------
    def _draw_strike_panel(ax, phist, aprof, panel_tag):
        h2d = np.ma.masked_where(phist["hist2d"] < 1, phist["hist2d"])
        str_edges = phist["value_bins"]
        depth_edges = np.concatenate(
            ([depth_grid[0]], 0.5 * (depth_grid[:-1] + depth_grid[1:]),
             [depth_grid[-1]]))

        pcm = ax.pcolormesh(
            str_edges, depth_edges, h2d,
            norm=LogNorm(vmin=1, vmax=max(phist["hist2d"].max(), 2)),
            cmap="RdYlGn_r", shading="flat", rasterized=True)
        fig.colorbar(pcm, ax=ax, label="Log freq.", shrink=0.5, pad=0.02)

        ax.plot(aprof["strike_median"], depth_grid, "k-", lw=2.0,
                label="Median")
        ax.plot(aprof["strike_p05"], depth_grid, "k--", lw=1.5, alpha=0.8)
        ax.plot(aprof["strike_p95"], depth_grid, "k--", lw=1.5, alpha=0.8,
                label="10th & 90th %")

        if true_model is not None and true_model.is_anisotropic:
            td = np.concatenate(([0], true_model.interfaces, [depth_max]))
            for ii in range(len(true_model.strikes)):
                ax.plot([true_model.strikes[ii], true_model.strikes[ii]],
                        [td[ii], td[ii + 1]],
                        "r-", lw=2, label="True" if ii == 0 else None)

        if depth_scale == "log":
            ax.set_yscale("log")
        ax.invert_yaxis()
        ax.set_xlabel("Strike [°]")
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.2, which="both")
        ax.text(0.02, 0.02, panel_tag, transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="bottom")

    # ---- draw histogram panels ---------------------------------------------
    if use_aniso:
        ax1 = fig.add_subplot(gs[0])
        true_rmax = (true_model.get_resistivities()
                     if true_model is not None else None)
        _draw_rho_panel(ax1, h_rmax, prof_rmax, "ρ_max", "E1",
                        show_true_rho=true_rmax)
        ax1.set_ylabel("Depth [m]")
        if station:
            ax1.set_title(f"Station {station}", fontsize=12, fontweight="bold")

        ax2 = fig.add_subplot(gs[1], sharey=ax1)
        true_rmin = None
        if true_model is not None and true_model.is_anisotropic:
            true_rmin = (true_model.get_resistivities()
                         / true_model.aniso_ratios)
        _draw_rho_panel(ax2, h_rmin, prof_rmin, "ρ_min", "E2",
                        show_true_rho=true_rmin)
        plt.setp(ax2.get_yticklabels(), visible=False)

        ax3 = fig.add_subplot(gs[2], sharey=ax1)
        _draw_strike_panel(ax3, h_str, aprof, "E3")
        plt.setp(ax3.get_yticklabels(), visible=False)

        ax_cpf = fig.add_subplot(gs[3], sharey=ax1)
    else:
        ax1 = fig.add_subplot(gs[0])
        _draw_rho_panel(ax1, h_rmax, prof_rmax, "Resistivity", "E",
                        show_true_rho=(true_model.get_resistivities()
                                       if true_model is not None else None))
        ax1.set_ylabel("Depth [m]")
        if station:
            ax1.set_title(f"Station {station}", fontsize=12, fontweight="bold")

        ax_cpf = fig.add_subplot(gs[1], sharey=ax1)

    # ---- change-point frequency panel --------------------------------------
    ax_cpf.plot(cpf, depth_grid, "b-", lw=1.2)
    ax_cpf.fill_betweenx(depth_grid, 0, cpf, alpha=0.15, color="steelblue")
    ax_cpf.set_xlabel("Change point\nfrequency")
    if depth_scale == "log":
        ax_cpf.set_yscale("log")
    ax_cpf.invert_yaxis()
    ax_cpf.grid(True, alpha=0.2)
    plt.setp(ax_cpf.get_yticklabels(), visible=False)
    ax_cpf.text(0.05, 0.02, "F", transform=ax_cpf.transAxes,
                fontsize=16, fontweight="bold", va="bottom")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    plt.close(fig)
    return fig


# =============================================================================
#  Private helpers
# =============================================================================

def _overlay_true_model(
    ax: plt.Axes,
    true_model: transdim.LayeredModel,
    depth_max: float,
) -> None:
    """Draw a staircase true-model profile on an existing axes."""
    true_depths = np.concatenate(([0], true_model.interfaces, [depth_max]))
    true_rho = true_model.get_resistivities()
    for i in range(len(true_rho)):
        ax.plot(
            [true_rho[i], true_rho[i]],
            [true_depths[i], true_depths[i + 1]],
            "r-", lw=2, label="True" if i == 0 else None)
        if i < len(true_rho) - 1:
            ax.plot(
                [true_rho[i], true_rho[i + 1]],
                [true_depths[i + 1], true_depths[i + 1]],
                "r-", lw=2)


def _overlay_true_staircase(
    ax: plt.Axes,
    values: np.ndarray,
    true_model: transdim.LayeredModel,
    depth_max: float,
) -> None:
    """Draw a generic staircase overlay for arbitrary per-layer values."""
    if values is None or true_model is None:
        return
    true_depths = np.concatenate(([0], true_model.interfaces, [depth_max]))
    for i in range(len(values)):
        ax.plot(
            [values[i], values[i]],
            [true_depths[i], true_depths[i + 1]],
            "r-", lw=2, label="True" if i == 0 else None)
        if i < len(values) - 1:
            ax.plot(
                [values[i], values[i + 1]],
                [true_depths[i + 1], true_depths[i + 1]],
                "r-", lw=2)
