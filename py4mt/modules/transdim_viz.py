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

def plot_qc(
    results: Dict,
    frequencies: np.ndarray,
    observed: np.ndarray,
    sigma: np.ndarray,
    station: str = "",
    use_aniso: bool = False,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Four-panel QC figure: ρ_a, phase, misfit trace, k histogram.

    Panels
    ------
    A — Apparent resistivity vs period: observations (red) + best-fit per chain (blue)
    B — Phase vs period: same layout
    C — Data misfit (χ²) vs sampling step (log–log), one curve per chain,
        dashed vertical line at burn-in
    D — Histogram of number of layers (post burn-in)
    """
    has_chains = "chains" in results
    periods = 1.0 / frequencies

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"MT Station {station} — 1D rj-McMC Inversion" if station
        else "1D rj-McMC Inversion QC",
        fontsize=14, fontweight="bold",
    )

    # ---- find best-fit model per chain (or overall) ------------------------
    if has_chains:
        best_models = []
        for cr in results["chains"]:
            best_idx = int(np.argmax(cr["log_likes"]))
            best_models.append(cr["models"][best_idx])
    else:
        best_idx = int(np.argmax(results["log_likes"]))
        best_models = [results["models"][best_idx]]

    # ---- compute best-fit predictions --------------------------------------
    best_preds = []
    for m in best_models:
        pred = transdim.mt_forward_1d_isotropic_full(
            m.get_thicknesses(), m.get_resistivities(), frequencies)
        best_preds.append(pred)

    # ---- Panel A: Apparent Resistivity vs Period ---------------------------
    ax = axes[0, 0]
    ax.errorbar(periods, observed, yerr=observed * sigma * np.log(10),
                fmt="rs", ms=5, capsize=2, zorder=5, label="Observations")
    for pred in best_preds:
        ax.plot(periods, pred["rho_a"], "b-", lw=1.5, alpha=0.8)
    # single legend entry for chains
    ax.plot([], [], "b-", lw=1.5, label="Best fit for each chain")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Period [s]")
    ax.set_ylabel("Apparent Resistivity [Ω·m]")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    ax.text(0.02, 0.02, "A", transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="bottom")

    # ---- Panel B: Phase vs Period ------------------------------------------
    ax = axes[0, 1]
    # Observed phase: approximate from observed rho_a is not available,
    # so we show best-fit phase curves only vs any "observed" phase if provided.
    # Plot best-fit phase curves:
    for pred in best_preds:
        ax.plot(periods, pred["phase_deg"], "b-", lw=1.5, alpha=0.8)
    ax.plot([], [], "b-", lw=1.5, label="Best fit for each chain")
    ax.set_xscale("log")
    ax.set_xlabel("Period [s]")
    ax.set_ylabel("Apparent Phase [Deg]")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    ax.text(0.02, 0.02, "B", transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="bottom")

    # ---- Panel C: Misfit trace ---------------------------------------------
    ax = axes[1, 0]
    burn_in = results.get("burn_in", 0)
    if has_chains:
        for i, cr in enumerate(results["chains"]):
            if "full_ll_trace" in cr:
                misfit = -2.0 * cr["full_ll_trace"]
                steps = np.arange(1, len(misfit) + 1)
                ax.plot(steps, misfit, "-", lw=0.6, alpha=0.7,
                        label=f"Chain {i}" if i < 6 else None)
    else:
        if "full_ll_trace" in results:
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

    # ---- Panel D: Number-of-layers histogram -------------------------------
    ax = axes[1, 1]
    k_vals = results["n_layers"]
    bins = np.arange(k_vals.min() - 0.5, k_vals.max() + 1.5, 1)
    ax.hist(k_vals, bins=bins, color="steelblue", edgecolor="white",
            alpha=0.8)
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
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Two-panel posterior-model figure: 2-D histogram + change-point frequency.

    Panels
    ------
    E — Colour-coded 2-D histogram of log10(ρ) vs depth (log frequency scale),
        overlaid with median, mean, mode, and 10th/90th percentile profiles.
    F — Change-point frequency vs depth (where the posterior places interfaces).
    """
    from matplotlib.colors import LogNorm
    import matplotlib.gridspec as gridspec

    depth_grid = np.linspace(1, depth_max, 600)

    # Compute posterior histogram
    log_rho_bins = np.linspace(log_rho_range[0], log_rho_range[1], n_rho_bins + 1)
    phist = transdim.compute_posterior_histogram(
        results["models"], depth_grid, log_rho_bins)

    # Compute standard profile statistics
    prof = transdim.compute_posterior_profile(results["models"], depth_grid)

    # Compute change-point frequency
    cpf = transdim.compute_changepoint_frequency(results["models"], depth_grid)

    # ---- Figure layout: E gets ~75% width, F gets ~25% --------------------
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.05)

    # ---- Panel E: 2-D histogram -------------------------------------------
    ax_e = fig.add_subplot(gs[0])

    hist2d = phist["hist2d"]
    hist2d_masked = np.ma.masked_where(hist2d < 1, hist2d)

    rho_edges = 10 ** phist["log_rho_bins"]
    depth_edges = np.concatenate(([depth_grid[0]],
                                  0.5 * (depth_grid[:-1] + depth_grid[1:]),
                                  [depth_grid[-1]]))

    pcm = ax_e.pcolormesh(
        rho_edges, depth_edges, hist2d_masked,
        norm=LogNorm(vmin=1, vmax=hist2d.max()),
        cmap="RdYlGn_r", shading="flat", rasterized=True)
    fig.colorbar(pcm, ax=ax_e, label="Log frequency", shrink=0.6, pad=0.02)

    # Overlay statistics
    ax_e.plot(prof["median"], depth_grid, "k-", lw=2.0, label="Median")
    ax_e.plot(prof["p05"], depth_grid, "k--", lw=1.5, alpha=0.8)
    ax_e.plot(prof["p95"], depth_grid, "k--", lw=1.5, alpha=0.8,
              label="10th & 90th percentile")
    ax_e.plot(prof["mean"], depth_grid, "b-", lw=1.5, label="Mean")
    ax_e.plot(phist["mode"], depth_grid, "g-", lw=1.5, label="Mode")

    if true_model is not None:
        _overlay_true_model(ax_e, true_model, depth_max)

    ax_e.set_xscale("log")
    ax_e.set_xlabel("Resistivity [Ω·m]")
    ax_e.set_ylabel("Depth [m]")
    ax_e.invert_yaxis()
    ax_e.legend(fontsize=9, loc="lower left")
    ax_e.grid(True, alpha=0.2, which="both")
    ax_e.text(0.02, 0.02, "E", transform=ax_e.transAxes,
              fontsize=16, fontweight="bold", va="bottom")
    if station:
        ax_e.set_title(f"Station {station}", fontsize=12, fontweight="bold")

    # ---- Panel F: Change-point frequency -----------------------------------
    ax_f = fig.add_subplot(gs[1], sharey=ax_e)

    ax_f.plot(cpf, depth_grid, "b-", lw=1.2)
    ax_f.fill_betweenx(depth_grid, 0, cpf, alpha=0.15, color="steelblue")
    ax_f.set_xlabel("Change point\nfrequency")
    ax_f.invert_yaxis()
    ax_f.grid(True, alpha=0.2)
    plt.setp(ax_f.get_yticklabels(), visible=False)
    ax_f.text(0.05, 0.02, "F", transform=ax_f.transAxes,
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
