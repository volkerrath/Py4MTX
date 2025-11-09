"""
femtic_borehole_viz.py

Plotting utilities for vertical borehole resistivity profiles.
Pure Matplotlib (no seaborn, no styles). Single-axis plots only.

Author: Volker Rath (DIAS)
Created by ChatGPT (GPT-5 Thinking)
"""

from typing import Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

def plot_vertical_profile(
    z: np.ndarray,
    v: np.ndarray,
    label: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    invert_z: bool = True,
    logx: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a single vertical resistivity profile v(z)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 6.5))
    else:
        fig = ax.figure
    ax.plot(v, z, label=label)
    ax.set_xlabel("Resistivity")
    ax.set_ylabel("Z")
    if logx:
        ax.set_xscale("log")
    if invert_z:
        ax.invert_yaxis()
    if label:
        ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig, ax

def plot_vertical_profiles(
    z: np.ndarray,
    profiles: List[np.ndarray],
    labels: Optional[List[str]] = None,
    logx: bool = True,
    z_positive_down: bool = True,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot multiple vertical profiles on a single Matplotlib axis."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 6.5))
    else:
        fig = ax.figure
    for i, prof in enumerate(profiles):
        lab = labels[i] if (labels and i < len(labels)) else None
        ax.plot(prof, z, label=lab)
    ax.set_xlabel("Resistivity")
    ax.set_ylabel("Z")
    if logx:
        ax.set_xscale("log")
    if z_positive_down:
        if z.size >= 2 and z[1] < z[0]:
            ax.invert_yaxis()
    else:
        if z.size >= 2 and z[1] > z[0]:
            ax.invert_yaxis()
    if labels:
        ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig, ax
